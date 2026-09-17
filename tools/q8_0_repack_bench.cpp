// tools/q8_0_repack_bench.cpp — P3b Q6_K-repack decode-GEMV GATE (2026-06-14).
//
// The dense ffn_down Q6_K decode falls to scalar gemm_q6_K at K=12288 (140 GB/s,
// the #1 decode cliff, dequant-ALU-bound). Hypothesis: repack Q6_K -> 8-bit at
// load (SoA Q8_0: int8 qs col-major + fp16 d/32-block) so the GEMV streams weights
// with only int8 dp4a (no 6-bit unpack) -> bandwidth-bound. Reads ~29% more bytes,
// so it only wins if the int-dot kernel clears ~250+ GB/s. This tool measures that
// DIRECTLY on the real shape (K=12288, N=4096) — no loader/dispatch plumbing.
//   gate: if gemv_q8_0_soa_q8 >= ~1.4x the scalar gemm_q6_K AND max_rel sane -> GO.
#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"
#include "ie/quantize.hpp"
#include "ie/allocator.hpp"

#include <sycl/sycl.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
static double median(std::vector<double> v) {
    if (v.empty()) return 0; std::sort(v.begin(), v.end()); return v[v.size() / 2];
}

int main(int argc, char** argv) {
    uint32_t K = (argc > 1) ? std::atoi(argv[1]) : 12288;   // dense ffn_down K
    uint32_t N = (argc > 2) ? std::atoi(argv[2]) : 4096;    // hidden
    const int runs = 25, inner = 10;

    ie::DeviceAllocator alloc;
    if (auto e = alloc.init(); !e.empty()) { std::fprintf(stderr, "alloc: %s\n", e.c_str()); return 1; }
    auto& q = alloc.queue();
    std::printf("q8_0-repack-gate  K=%u N=%u  device=%s\n", K, N,
                q.get_device().get_info<sycl::info::device::name>().c_str());

    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 0.05f);
    std::vector<float> Wf(uint64_t(N) * K), af(K);
    for (auto& v : Wf) v = nd(rng);
    for (auto& v : af) v = nd(rng);

    std::vector<float> y_ref(N);
    for (uint32_t n = 0; n < N; ++n) {
        double s = 0; const float* w = &Wf[uint64_t(n) * K];
        for (uint32_t k = 0; k < K; ++k) s += double(w[k]) * double(af[k]);
        y_ref[n] = float(s);
    }

    // Q6_K (per column, K elems) — the production format.
    const uint32_t q6bpc = K / 256;
    std::vector<ie::block_q6_K> Wq6(uint64_t(N) * q6bpc);
    for (uint32_t n = 0; n < N; ++n)
        ie::quantize_row_q6_K(&Wf[uint64_t(n) * K], &Wq6[uint64_t(n) * q6bpc], K);

    // SoA Q8_0 repack: qs int8 col-major [N*K] + d fp16 [N*K/32].
    const uint32_t bpc = K / 32;
    std::vector<int8_t>   qsW(uint64_t(N) * K);
    std::vector<uint16_t> dW(uint64_t(N) * bpc);
    for (uint32_t n = 0; n < N; ++n)
        for (uint32_t b = 0; b < bpc; ++b) {
            const float* x = &Wf[uint64_t(n) * K + uint64_t(b) * 32];
            float amax = 0; for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[i]));
            const float d = amax / 127.f, inv = amax > 0 ? 127.f / amax : 0.f;
            dW[uint64_t(n) * bpc + b] = ie::fp32_to_fp16(d);
            for (int i = 0; i < 32; ++i)
                qsW[uint64_t(n) * K + uint64_t(b) * 32 + i] = int8_t(std::lround(x[i] * inv));
        }

    std::vector<sycl::half> a16(K);
    for (uint32_t k = 0; k < K; ++k) a16[k] = sycl::half(af[k]);

    auto* d_Wq6 = sycl::malloc_device<ie::block_q6_K>(Wq6.size(), q);
    auto* d_qsW = sycl::malloc_device<int8_t>(qsW.size(), q);
    auto* d_dW  = sycl::malloc_device<uint16_t>(dW.size(), q);
    auto* d_a16 = sycl::malloc_device<sycl::half>(K, q);
    auto* d_q8  = sycl::malloc_device<ie::block_q8_1x>(K / 32, q);
    auto* d_y   = sycl::malloc_device<sycl::half>(N, q);
    q.memcpy(d_Wq6, Wq6.data(), Wq6.size() * sizeof(ie::block_q6_K)).wait();
    q.memcpy(d_qsW, qsW.data(), qsW.size()).wait();
    q.memcpy(d_dW,  dW.data(),  dW.size() * 2).wait();
    q.memcpy(d_a16, a16.data(), K * sizeof(sycl::half)).wait();

    std::vector<sycl::half> hy(N);
    // Relative RMS error vs fp32 ref (per-element max_rel is meaningless here —
    // many y_ref[n] are ≈0 since the mean is 0, so it divides by noise).
    auto rel_rms = [&]() {
        q.memcpy(hy.data(), d_y, N * sizeof(sycl::half)).wait();
        double se = 0, sr = 0;
        for (uint32_t n = 0; n < N; ++n) {
            const double e = double(float(hy[n])) - y_ref[n];
            se += e * e; sr += y_ref[n] * double(y_ref[n]);
        }
        return std::sqrt(se / sr);
    };

    const double wb_q6 = double(Wq6.size()) * sizeof(ie::block_q6_K);
    const double wb_q8 = double(qsW.size()) + double(dW.size()) * 2.0;

    // warmup (JIT)
    ie::gemm_q6_K(q, d_a16, d_Wq6, d_y, 1, K, N); q.wait();
    ie::quantize_q8_1(q, d_a16, d_q8, K);
    ie::gemv_q8_0_soa_q8(q, d_q8, d_qsW, d_dW, d_y, K, N); q.wait();

    std::vector<double> t6, t8;
    for (int r = 0; r < runs; ++r) {
        const double t0 = now_ms();
        for (int i = 0; i < inner; ++i) ie::gemm_q6_K(q, d_a16, d_Wq6, d_y, 1, K, N);
        q.wait(); t6.push_back((now_ms() - t0) / inner);
    }
    const double q6ms = median(t6), q6rel = rel_rms();
    for (int r = 0; r < runs; ++r) {
        const double t0 = now_ms();
        for (int i = 0; i < inner; ++i) {
            ie::quantize_q8_1(q, d_a16, d_q8, K);
            ie::gemv_q8_0_soa_q8(q, d_q8, d_qsW, d_dW, d_y, K, N);
        }
        q.wait(); t8.push_back((now_ms() - t0) / inner);
    }
    const double q8ms = median(t8), q8rel = rel_rms();

    std::printf("Q6_K scalar (gemm_q6_K) : %.4f ms  %6.1f GB/s  rel_rms %.4f\n",
                q6ms, wb_q6 / q6ms / 1e6, q6rel);
    std::printf("Q8_0 repack (int-dot)   : %.4f ms  %6.1f GB/s  rel_rms %.4f\n",
                q8ms, wb_q8 / q8ms / 1e6, q8rel);
    std::printf("=> speedup %.2fx   (weight %.0f MB Q6_K vs %.0f MB Q8_0, +%.0f%% bytes)\n",
                q6ms / q8ms, wb_q6 / 1e6, wb_q8 / 1e6, (wb_q8 / wb_q6 - 1) * 100);
    std::printf("VERDICT: %s\n", (q6ms / q8ms >= 1.4 && q8rel < 0.03)
                ? "GO — repack clears the 1.4x gate, correctness sane"
                : "NO-GO — int-dot repack does not beat the scalar cliff enough");
    return 0;
}
