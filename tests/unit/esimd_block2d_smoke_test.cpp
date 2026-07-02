// tests/unit/esimd_block2d_smoke_test.cpp
//
// Phase 1 (v2.0) Days 1-2: validates that the IGC SPIR-V 2D-block-load
// builtin links and round-trips raw bytes on BMG-G31. This is the
// prerequisite plumbing for the register-resident Q4_K micro-kernel
// (gemm_q4_K_esimd) that follows in Days 3-10.
//
// Smoke check: m8k32v1 — load an 8×32-byte (= 256 B / 4 GRF) tile via
// cooperative sub-group of 16 lanes, sum all bytes, compare against
// host-computed expected sum.  The earlier sub-GRF variants (m1k16v1
// at 16 B) silently zero-pad on Xe2 — see comments in gemm_q4k_esimd.cpp.

#include "ie/ops.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

int run_smoke_tile256(sycl::queue& q) {
    // Xe2 2D-block-load hardware constraints:
    //   * surface base 64 B aligned (USM device alloc satisfies this),
    //   * surface width ≥ 64 B,
    //   * pitch ≥ 64 B and 16-B aligned.
    // We allocate an 8 × 64-B surface; the load reads cols [0..32) of
    // all 8 rows (= 256 bytes).
    constexpr uint32_t W = 64;          // surface width (bytes)
    constexpr uint32_t H = 8;           // rows
    constexpr uint32_t P = 64;          // pitch (bytes)
    constexpr uint32_t LOAD_W = 32;
    constexpr uint32_t LOAD_H = 8;

    std::vector<uint8_t> src_h(W * H, 0);
    std::mt19937 rng(0xC0DEC0DEu);
    for (auto& b : src_h) b = uint8_t(rng() & 0xFFu);

    uint64_t expected_sum64 = 0;
    for (uint32_t r = 0; r < LOAD_H; ++r)
        for (uint32_t c = 0; c < LOAD_W; ++c)
            expected_sum64 += src_h[r * P + c];
    const uint32_t expected = uint32_t(expected_sum64);

    // 2D block load on Xe2 requires 64-B aligned base. Force it.
    auto* src_d = sycl::aligned_alloc_device<uint8_t>(64, W * H, q);
    auto* sum_d = sycl::malloc_device<uint32_t>(1,    q);
    std::printf("  src_d=%p alignment-mod-64=%lu\n",
                (void*)src_d, reinterpret_cast<uintptr_t>(src_d) % 64u);
    q.memcpy(src_d, src_h.data(), W * H).wait();
    q.memset(sum_d, 0, sizeof(uint32_t)).wait();

    ie::esimd_block2d_smoke_tile256(q, src_d, sum_d, W, H, P).wait();

    uint32_t got = 0;
    q.memcpy(&got, sum_d, sizeof(uint32_t)).wait();

    sycl::free(src_d, q);
    sycl::free(sum_d, q);

    if (got == expected) {
        std::printf("[smoke tile256] PASS — 8×32-u8 tile sum = %u.\n", got);
        return 0;
    } else {
        std::fprintf(stderr,
            "[smoke tile256] FAIL: got sum=%u expected=%u (Δ=%d)\n",
            got, expected, int(int64_t(got) - int64_t(expected)));
        return 1;
    }
}

}  // namespace

int main() {
    sycl::queue q{sycl::gpu_selector_v,
                  sycl::property::queue::in_order{}};
    auto dev = q.get_device();
    std::printf("Device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
    std::printf("Vendor: %s\n", dev.get_info<sycl::info::device::vendor>().c_str());

    int rc = run_smoke_tile256(q);

    if (rc == 0) std::printf("\nESIMD 2D-block-load smoke check PASS.\n");
    else         std::fprintf(stderr, "\nESIMD 2D-block-load smoke check FAILED.\n");
    return rc;
}
