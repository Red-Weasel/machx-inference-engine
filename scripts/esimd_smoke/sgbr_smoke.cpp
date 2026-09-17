// /tmp/sgbr_smoke.cpp
//
// Step 0 — disambiguate: does cooperative subgroup block read
// (sub_group::load → __spirv_SubgroupBlockReadINTEL<ushort>) work on
// BMG-G31 stepping C0 (IP 20.2.0)?
//
// Memory file (project_v2_phase1_freeze_diagnosis.md) claims yes.
// Existing src/ops/gemm_q4k_esimd.cpp comment claims no.
// One controlled test resolves this.
//
// Test layout: 8 rows × 16 ushorts each = 256 bytes.  Each row is read
// as ONE cooperative SG transaction (16 lanes form a single 32-byte
// LSC.ugm load).  Lane lid receives src[row*16 + lid] per the
// sub_group::load contract.
//
// Pass: returned sum matches host-computed expected sum AND no GuC
// reset in journalctl.
// Fail: sum=0 (kernel timed out, GuC CT failed) OR sum != expected
// (load returned wrong data).

#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    auto dev = q.get_device();
    std::printf("Device: %s\n",
                dev.get_info<sycl::info::device::name>().c_str());
    std::printf("Driver: %s\n",
                dev.get_info<sycl::info::device::driver_version>().c_str());

    constexpr uint32_t ROWS = 8;
    constexpr uint32_t COLS = 16;       // ushorts per row
    constexpr uint32_t TOTAL = ROWS * COLS;

    // Initialize known data: src[r*16 + c] = (r*16 + c) & 0xFFFF.
    std::vector<uint16_t> src_h(TOTAL);
    uint64_t expected_byte_sum = 0;
    for (uint32_t r = 0; r < ROWS; ++r) {
        for (uint32_t c = 0; c < COLS; ++c) {
            const uint16_t v = uint16_t((r * COLS + c) & 0xFFFFu);
            src_h[r * COLS + c] = v;
            expected_byte_sum += uint32_t(v & 0xFFu);
            expected_byte_sum += uint32_t((v >> 8) & 0xFFu);
        }
    }

    // 16-byte alignment satisfies LSC requirements (sub_group::load
    // wants the base aligned to the per-lane element size; 16-byte is safer).
    auto* src_d = sycl::aligned_alloc_device<uint16_t>(16, TOTAL, q);
    auto* sum_d = sycl::malloc_device<uint32_t>(1, q);
    if (!src_d || !sum_d) {
        std::fprintf(stderr, "alloc failed\n");
        return 2;
    }
    std::printf("src_d=%p (mod16=%lu)  sum_d=%p  expected_byte_sum=%llu\n",
                (void*)src_d,
                reinterpret_cast<uintptr_t>(src_d) % 16u,
                (void*)sum_d,
                (unsigned long long)expected_byte_sum);

    q.memcpy(src_d, src_h.data(), TOTAL * sizeof(uint16_t)).wait();
    q.memset(sum_d, 0, sizeof(uint32_t)).wait();

    std::printf("Issuing kernel launch (1 SG of 16 lanes, 8 cooperative rows)...\n");
    std::fflush(stdout);

    auto e = q.submit([&](sycl::handler& h) {
        h.parallel_for<class SgBlockReadSmoke>(
            sycl::nd_range<1>(16, 16),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                const auto sg = it.get_sub_group();
                uint32_t lane_sum = 0;

#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    // sub_group::load expects a multi_ptr.  Each row is
                    // 16 ushorts (32 bytes), one cooperative SG load.
                    auto mptr = sycl::address_space_cast<
                                    sycl::access::address_space::global_space,
                                    sycl::access::decorated::yes>(
                        const_cast<uint16_t*>(src_d) + r * 16);
                    const uint16_t v = sg.load(mptr);
                    lane_sum += uint32_t(v & 0xFFu);
                    lane_sum += uint32_t((v >> 8) & 0xFFu);
                }

                const uint32_t total = sycl::reduce_over_group(
                    sg, lane_sum, sycl::plus<uint32_t>());
                if (it.get_local_id(0) == 0) *sum_d = total;
            });
    });

    // Bound the wait — if the GPU hangs we want to know quickly.
    e.wait();

    uint32_t got = 0;
    q.memcpy(&got, sum_d, sizeof(uint32_t)).wait();

    sycl::free(src_d, q);
    sycl::free(sum_d, q);

    std::printf("\nResult: got=%u  expected=%llu\n",
                got, (unsigned long long)expected_byte_sum);

    if (got == uint32_t(expected_byte_sum)) {
        std::printf("\n=== PASS ===\n");
        std::printf("Cooperative __spirv_SubgroupBlockReadINTEL works on this HW.\n");
        std::printf("ESIMD GEMM via 1D block reads is unblocked.\n");
        return 0;
    }
    if (got == 0) {
        std::fprintf(stderr, "\n=== FAIL: got=0 ===\n");
        std::fprintf(stderr,
            "Kernel likely hung — check `journalctl -k -g 'xe.*coredump'` for GuC reset.\n");
        std::fprintf(stderr,
            "Cooperative subgroup block read confirmed BROKEN on this HW;\n");
        std::fprintf(stderr,
            "no IGC variant of __spirv_SubgroupBlockReadINTEL is functional.\n");
        return 1;
    }
    std::fprintf(stderr, "\n=== FAIL: got=%u expected=%llu (Δ=%lld) ===\n",
                 got, (unsigned long long)expected_byte_sum,
                 (long long)int64_t(got) - int64_t(expected_byte_sum));
    std::fprintf(stderr,
        "Cooperative SG block read returned WRONG data — kernel ran but read pattern is wrong.\n");
    return 1;
}
