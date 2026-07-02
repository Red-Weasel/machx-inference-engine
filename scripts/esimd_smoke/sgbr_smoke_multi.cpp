// /tmp/sgbr_smoke_multi.cpp
//
// Step 0 (extended): test multi-element-per-lane cooperative subgroup
// block reads on BMG-G31 C0.
//
// Rationale: the single-ushort-per-lane form (sg.load(ptr)) was just
// confirmed working.  But ESIMD GEMM weight loads need to fill SIMD
// registers (typically 8 or 16 elements per lane = full GRF).  The
// existing src/ops/gemm_q4k_esimd.cpp source comment specifically
// flagged the multi-element transposed form as broken:
//
//   `lsc_load.ugm (M1_NM,1) d32x8t (__spirv_SubgroupBlockReadINTEL)`
//
// We need the NON-transposed multi-element form (vec<T,N> per lane,
// contiguous-row layout) to be functional.  Test it here.

#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

template <int N> class SgBlockReadMulti_T;

template <int N>
int run_test(sycl::queue& q, const char* label) {
    using vec_t = sycl::vec<uint16_t, N>;

    constexpr uint32_t LANES_PER_SG = 16;
    // Per row: LANES_PER_SG × N ushorts contiguous.
    constexpr uint32_t COLS = LANES_PER_SG * N;
    constexpr uint32_t ROWS = 8;
    const uint32_t TOTAL = ROWS * COLS;

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

    auto* src_d = sycl::aligned_alloc_device<uint16_t>(64, TOTAL, q);
    auto* sum_d = sycl::malloc_device<uint32_t>(1, q);
    if (!src_d || !sum_d) {
        std::fprintf(stderr, "alloc failed\n");
        return 2;
    }
    q.memcpy(src_d, src_h.data(), TOTAL * sizeof(uint16_t)).wait();
    q.memset(sum_d, 0, sizeof(uint32_t)).wait();

    std::printf("[%s] N=%d, row=%u ushorts (%u bytes), expected=%llu\n",
                label, N, COLS, COLS * 2, (unsigned long long)expected_byte_sum);
    std::fflush(stdout);

    const uint32_t COLS_local = COLS;
    auto e = q.submit([&](sycl::handler& h) {
        h.parallel_for<class SgBlockReadMulti_T<N>>(
            sycl::nd_range<1>(LANES_PER_SG, LANES_PER_SG),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                const auto sg = it.get_sub_group();
                uint32_t lane_sum = 0;

                for (int r = 0; r < ROWS; ++r) {
                    auto mptr = sycl::address_space_cast<
                                    sycl::access::address_space::global_space,
                                    sycl::access::decorated::yes>(
                        const_cast<uint16_t*>(src_d) + r * COLS_local);
                    // Multi-element cooperative load: each lane gets N
                    // ushorts.  16 lanes × N elements = full row per call.
                    vec_t vv = sg.load<N>(mptr);
                    #pragma unroll
                    for (int i = 0; i < N; ++i) {
                        const uint16_t v = vv[i];
                        lane_sum += uint32_t(v & 0xFFu);
                        lane_sum += uint32_t((v >> 8) & 0xFFu);
                    }
                }

                const uint32_t total = sycl::reduce_over_group(
                    sg, lane_sum, sycl::plus<uint32_t>());
                if (it.get_local_id(0) == 0) *sum_d = total;
            });
    });
    e.wait();

    uint32_t got = 0;
    q.memcpy(&got, sum_d, sizeof(uint32_t)).wait();
    sycl::free(src_d, q);
    sycl::free(sum_d, q);

    if (got == uint32_t(expected_byte_sum)) {
        std::printf("[%s] PASS — got=%u\n", label, got);
        return 0;
    }
    std::fprintf(stderr,
        "[%s] FAIL — got=%u expected=%llu (Δ=%lld)\n",
        label, got, (unsigned long long)expected_byte_sum,
        (long long)int64_t(got) - int64_t(expected_byte_sum));
    return 1;
}

int main() {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    auto dev = q.get_device();
    std::printf("Device: %s\n",
                dev.get_info<sycl::info::device::name>().c_str());
    std::printf("Driver: %s\n\n",
                dev.get_info<sycl::info::device::driver_version>().c_str());

    int rc = 0;
    // Test progressively higher element counts.  Stop at first failure
    // (one of these is the "highest BW form that works").
    rc |= run_test<2>(q, "N=2  (32 B/row)");
    if (rc) { std::fprintf(stderr, "stopping at first failure\n"); return rc; }

    rc |= run_test<4>(q, "N=4  (64 B/row)");
    if (rc) { std::fprintf(stderr, "stopping at first failure\n"); return rc; }

    rc |= run_test<8>(q, "N=8  (128 B/row)");
    if (rc) { std::fprintf(stderr, "stopping at first failure\n"); return rc; }

    std::printf("\n=== ALL MULTI-ELEMENT FORMS PASS ===\n");
    std::printf("Cooperative SG block read works at N=2,4,8 (16-128 byte rows).\n");
    std::printf("ESIMD GEMM weight loads via multi-element 1D block reads is unblocked.\n");
    return 0;
}
