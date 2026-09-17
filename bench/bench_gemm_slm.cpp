// bench/bench_gemm_slm.cpp — Phase 3 FP16 GEMM gate.
//
// Bench the SLM-tiled `ie::gemm_fp16` against:
//   - bench/baseline/phase0 naive joint_matrix (≈47 TFLOPS = 26% peak)
// Phase 3 gate target: ≥ 70% of 183 TFLOPS = 128 TFLOPS.
//
//   ./bench_gemm_slm [--m M --n N --k K] [--iters N] [--verify]

#include "ie/ops.hpp"

#include <sycl/sycl.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr const char* kBold = "\033[1m";
constexpr const char* kReset = "\033[0m";
constexpr const char* kGreen = "\033[32m";
constexpr const char* kYell  = "\033[33m";

}  // namespace

int main(int argc, char** argv) {
    int M = 4096, N = 4096, K = 4096, iters = 20, warmup = 5;
    bool verify = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--m" && i+1 < argc) M = std::atoi(argv[++i]);
        else if (a == "--n" && i+1 < argc) N = std::atoi(argv[++i]);
        else if (a == "--k" && i+1 < argc) K = std::atoi(argv[++i]);
        else if (a == "--iters" && i+1 < argc) iters = std::atoi(argv[++i]);
        else if (a == "--verify") verify = true;
    }

    sycl::device dev;
    bool found = false;
    for (const auto& d : sycl::device::get_devices()) {
        if (d.is_gpu() && d.get_info<sycl::info::device::name>().find("0xe223") != std::string::npos) {
            dev = d; found = true; break;
        }
    }
    if (!found) for (const auto& d : sycl::device::get_devices()) if (d.is_gpu()) { dev = d; found = true; break; }
    if (!found) { std::fputs("no GPU\n", stderr); return 2; }
    sycl::queue q(dev, sycl::property::queue::enable_profiling{});

    std::printf("Device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
    std::printf("Shape : M=%d N=%d K=%d  (FP16 in / FP32 out)\n", M, N, K);
    std::printf("Kernel: ie::gemm_fp16 (SLM-tiled, BM=128 BN=128 BK=16, 4×4 SGs/WG)\n\n");

    // Init host
    std::vector<sycl::half> hA(size_t(M) * K), hB(size_t(K) * N);
    std::mt19937 rng(0xC0FFEE);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto& x : hA) x = sycl::half(dist(rng));
    for (auto& x : hB) x = sycl::half(dist(rng));

    auto* dA = sycl::malloc_device<sycl::half>(size_t(M) * K, q);
    auto* dB = sycl::malloc_device<sycl::half>(size_t(K) * N, q);
    auto* dC = sycl::malloc_device<float>(size_t(M) * N, q);
    if (!dA || !dB || !dC) { std::fputs("device alloc fail\n", stderr); return 3; }
    q.memcpy(dA, hA.data(), hA.size() * sizeof(sycl::half)).wait();
    q.memcpy(dB, hB.data(), hB.size() * sizeof(sycl::half)).wait();

    // Warmup
    for (int i = 0; i < warmup; ++i) ie::gemm_fp16(q, dA, dB, dC, M, N, K).wait();

    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();
    for (int i = 0; i < iters; ++i) ie::gemm_fp16(q, dA, dB, dC, M, N, K).wait();
    auto t1 = clk::now();
    const double sec = std::chrono::duration<double>(t1 - t0).count() / iters;
    const double flops = 2.0 * double(M) * double(N) * double(K);
    const double tflops = flops / sec / 1e12;
    const double peak = 183.0;
    const double pct  = 100.0 * tflops / peak;
    const char* color = (pct >= 70.0) ? kGreen : kYell;

    std::printf("Result\n  iters     : %d\n  time/iter : %.3f ms\n", iters, sec * 1e3);
    std::printf("  %sTFLOPS    : %.2f%s\n", color, tflops, kReset);
    std::printf("  %% peak    : %s%.1f%%%s of %.0f TFLOPS\n", color, pct, kReset, peak);
    std::printf("  gate (70%% peak = %.0f TFLOPS): %s\n",
                peak * 0.70, pct >= 70.0 ? "PASS" : "MISS");

    if (verify) {
        std::vector<float> hC(size_t(M) * N);
        q.memcpy(hC.data(), dC, hC.size() * sizeof(float)).wait();
        // Spot-check a single cell against scalar fp32 reference
        const int row = 5, col = 7;
        float ref = 0.f;
        for (int k = 0; k < K; ++k) ref += float(hA[row*K+k]) * float(hB[k*N+col]);
        const float got = hC[row*N + col];
        const float err = std::fabs(ref - got) / std::fmax(std::fabs(ref), 1e-6f);
        std::printf("  verify    : C[%d,%d] ref=%.4f got=%.4f rel=%.2g  %s\n",
                    row, col, ref, got, err, err < 1e-2f ? "OK" : "FAIL");
        if (err >= 1e-2f) { sycl::free(dA, q); sycl::free(dB, q); sycl::free(dC, q); return 4; }
    }

    sycl::free(dA, q); sycl::free(dB, q); sycl::free(dC, q);
    return 0;
}
