// tools/onednn_multidev_test.cpp — validates the per-device oneDNN ctx map.
//
// Proves the Step-1 fix in gemm_onednn.cpp:ctx_for. The old static singleton
// built ONE engine/stream bound to the first queue it saw (card 0); calling
// gemm_fp16_onednn with a card-1 buffer then routed card-1 USM through that
// card-0-bound stream → UR_RESULT_ERROR_DEVICE_LOST (the reason oneDNN was
// force-disabled on every multi-card layer-split path). With one engine PER
// device, each card's GEMM uses its own engine/stream/prim-cache.
//
// We run a small fp16 matmul on EACH B70, INTERLEAVED (dev0, dev1, dev0, dev1),
// and verify every result against a host fp32 reference. A per-device-confused
// cache (or the old singleton) surfaces immediately on the first dev1 call:
// either a thrown DEVICE_LOST or wrong numbers.
//
// usage: ie-onednn-multidev-test   (needs >=2 B70s; SKIPs cleanly otherwise)
#include "ie/allocator.hpp"
#include "ie/ops.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

// y[M,N] = A[M,K] @ B[K,N], row-major fp16, checked against an fp32 host ref.
// Inputs are multiples of 0.25 (exact in fp16) so the reference is unambiguous.
bool run_on_device(ie::DeviceFleet& fleet, uint32_t d, int round) {
    const uint32_t M = 64, K = 256, N = 128;
    auto& dev = fleet.dev(d);
    auto& q = dev.queue();

    std::vector<sycl::half> hA(size_t(M) * K), hB(size_t(K) * N),
        hY(size_t(M) * N, sycl::half(0.f));
    for (uint32_t m = 0; m < M; ++m)
        for (uint32_t k = 0; k < K; ++k)
            hA[size_t(m) * K + k] = sycl::half(float((m + k + round) % 4) * 0.25f);
    for (uint32_t k = 0; k < K; ++k)
        for (uint32_t n = 0; n < N; ++n)
            hB[size_t(k) * N + n] = sycl::half(float((k + n) % 4) * 0.25f);

    auto* dA = static_cast<sycl::half*>(dev.malloc(hA.size() * sizeof(sycl::half)));
    auto* dB = static_cast<sycl::half*>(dev.malloc(hB.size() * sizeof(sycl::half)));
    auto* dY = static_cast<sycl::half*>(dev.malloc(hY.size() * sizeof(sycl::half)));
    if (!dA || !dB || !dY) {
        std::fprintf(stderr, "dev %u: malloc failed\n", d);
        return false;
    }
    q.memcpy(dA, hA.data(), hA.size() * sizeof(sycl::half)).wait();
    q.memcpy(dB, hB.data(), hB.size() * sizeof(sycl::half)).wait();

    bool ok = true;
    try {
        ie::gemm_fp16_onednn(q, dA, dB, dY, M, N, K, {}).wait();
        q.memcpy(hY.data(), dY, hY.size() * sizeof(sycl::half)).wait();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "dev %u (round %d): gemm_fp16_onednn THREW: %s\n",
                     d, round, e.what());
        ok = false;
    }
    dev.free(dA);
    dev.free(dB);
    dev.free(dY);
    if (!ok) return false;

    size_t bad = 0;
    float maxerr = 0.f;
    for (uint32_t m = 0; m < M; ++m)
        for (uint32_t n = 0; n < N; ++n) {
            float acc = 0.f;
            for (uint32_t k = 0; k < K; ++k)
                acc += float(hA[size_t(m) * K + k]) * float(hB[size_t(k) * N + n]);
            float err = std::fabs(acc - float(hY[size_t(m) * N + n]));
            if (err > maxerr) maxerr = err;
            if (err > 0.25f) ++bad;  // fp16 output rounding is < 0.125 at these magnitudes
        }
    if (bad) {
        std::fprintf(stderr, "dev %u (round %d): %zu/%u elems wrong (maxerr %g)\n",
                     d, round, bad, M * N, maxerr);
        return false;
    }
    std::printf("  dev %u (round %d): OK (%ux%u x K=%u, maxerr %g)\n",
                d, round, M, N, K, maxerr);
    return true;
}

}  // namespace

int main() {
    ie::DeviceFleet fleet;
    if (auto e = fleet.init(2); !e.empty()) {
        std::fprintf(stderr, "fleet init: %s\n", e.c_str());
        return 1;
    }
    std::printf("onednn-multidev: %u device(s) bound\n", fleet.size());
    if (fleet.size() < 2) {
        std::printf("only %u device — multi-device oneDNN test SKIPPED\n", fleet.size());
        return 0;
    }

    // Interleave devices so a device-confused cache surfaces on the first dev1 hit.
    bool ok = true;
    for (int round = 0; round < 2; ++round)
        for (uint32_t d = 0; d < fleet.size(); ++d)
            ok = run_on_device(fleet, d, round) && ok;

    if (!ok) {
        std::fprintf(stderr,
                     "FAIL: per-device oneDNN ctx map did not isolate the devices\n");
        return 1;
    }
    std::printf("PASS: gemm_fp16_onednn ran on all %u devices interleaved — "
                "no DEVICE_LOST, all numerically correct\n",
                fleet.size());
    return 0;
}
