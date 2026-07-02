// tools/multi_gpu_probe.cpp — P-A: validate the DeviceFleet (multi-GPU foundation).
// Enumerates the fleet, then round-trips a buffer dev0 → (copy_across) → dev1 →
// host and asserts it survives bit-identically. The first concrete step of the
// layer-split build (docs/superpowers/specs/2026-06-12-multi-gpu-layer-split-design.md).
//
// usage: ie-multi-gpu-probe [n_devices=2]
#include "ie/allocator.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    const uint32_t n = (argc > 1) ? uint32_t(std::atoi(argv[1])) : 2u;
    ie::DeviceFleet fleet;
    if (auto e = fleet.init(n); !e.empty()) {
        std::fprintf(stderr, "fleet init: %s\n", e.c_str());
        return 1;
    }
    std::printf("DeviceFleet: %u device(s) bound (requested %u)\n", fleet.size(), n);
    for (uint32_t i = 0; i < fleet.size(); ++i)
        std::printf("  dev %u: %s\n", i,
                    fleet.dev(i).device().get_info<sycl::info::device::name>().c_str());
    if (fleet.size() < 2) {
        std::printf("only %u device available — cross-device test skipped\n", fleet.size());
        return 0;
    }

    // Round-trip: host pattern → dev0 → copy_across → dev1 → host; verify exact.
    const size_t N = 4096;
    std::vector<float> src(N), dst(N, 0.f);
    for (size_t i = 0; i < N; ++i) src[i] = float(i) * 1.5f - 3.0f;
    void* d0 = fleet.dev(0).malloc(N * sizeof(float));
    void* d1 = fleet.dev(1).malloc(N * sizeof(float));
    if (!d0 || !d1) { std::fprintf(stderr, "malloc failed\n"); return 1; }

    fleet.dev(0).queue().memcpy(d0, src.data(), N * sizeof(float)).wait();
    fleet.copy_across(/*si=*/0, d1, /*di=*/1, d0, N * sizeof(float));   // dev0 → dev1
    fleet.dev(1).queue().memcpy(dst.data(), d1, N * sizeof(float)).wait();

    size_t mism = 0;
    for (size_t i = 0; i < N; ++i) if (src[i] != dst[i]) ++mism;
    fleet.dev(0).free(d0);
    fleet.dev(1).free(d1);

    if (mism) {
        std::fprintf(stderr, "FAIL: %zu/%zu floats differ after dev0->dev1 round-trip\n", mism, N);
        return 1;
    }
    std::printf("cross-device round-trip dev0->dev1: OK (%zu floats bit-identical)\n", N);

    // All-reduce (TP foundation): dev d's buffer = (d+1); after all-reduce every
    // buffer must hold sum(d+1) over all devices (1+2 = 3 for 2 cards).
    {
        const size_t M = 2048;
        std::vector<sycl::half*> bufs(fleet.size());
        float expected = 0.f;
        for (uint32_t d = 0; d < fleet.size(); ++d) {
            expected += float(d + 1);
            bufs[d] = static_cast<sycl::half*>(fleet.dev(d).malloc(M * sizeof(sycl::half)));
            std::vector<sycl::half> init(M, sycl::half(float(d + 1)));
            fleet.dev(d).queue().memcpy(bufs[d], init.data(), M * sizeof(sycl::half)).wait();
        }
        fleet.all_reduce_sum_fp16(bufs, M);
        size_t armis = 0;
        for (uint32_t d = 0; d < fleet.size(); ++d) {
            std::vector<sycl::half> got(M);
            fleet.dev(d).queue().memcpy(got.data(), bufs[d], M * sizeof(sycl::half)).wait();
            for (size_t i = 0; i < M; ++i) if (float(got[i]) != expected) ++armis;
            fleet.dev(d).free(bufs[d]);
        }
        if (armis) { std::fprintf(stderr, "FAIL all-reduce: %zu/%zu wrong (expected %g)\n", armis, M * fleet.size(), expected); return 1; }
        std::printf("all-reduce SUM across %u devices: OK (every buffer == %g)\n", fleet.size(), expected);
    }
    return 0;
}
