// tools/qwen4_quant_test.cpp — qwen4exp (Qwen3.8-Flash-Next) quant-path gate.
//
// No model file needed. Validates, against host references:
//   1. GPU Q5_1 GEMV (gemv_q5_1) vs a CPU fp64 reference built on
//      ref::dequant_q5_1_buffer — random blocks (fixed seed), K=640 N=2560
//      (the ffn_down_exps per-expert shape) + K=64 N=7 edge; max rel err
//      <= 2e-3. Plus x = unit vectors: the same GEMV must reproduce the
//      dequantized columns <= 1e-3 rel.
//   2. Batched-rows variant (gemm_q5_1), M in {1, 4, 33}, <= 2e-3.
//   3. Expert-slice addressing (q5_1_expert_slice) on a synthetic
//      [K=64, N=32, E=512] bank, e in {0, 7, 511}: GEMV the slice, compare
//      vs CPU fp64 ref indexed independently over the full bank.
//   4. convert_bf16_to_f16: exhaustive over all 65536 bf16 bit patterns vs
//      an independent fp32-promote-then-_Float16-RTNE reference (exact bits;
//      NaN in -> NaN out, bits not compared).
//   5. Q5_K: ref::dequant_q5_K already exists (dequant_ref.hpp) — gate checks
//      convert_q5_k_to_f16_buffer buffer indexing on random blocks + a
//      hand-crafted block with known y = d*sc*q - dmin*m values.
//
// Rel err denominator: max(|ref|, 0.01 * max|ref| over the output) for the
// random-x GEMVs (guards cancellation-shrunk outputs), max(|ref|, 1e-6) for
// the unit-vector / all-positive cases.
//
// usage: ie-qwen4-quant-test          exit 0 = GATE PASSED, 1 = FAILED
#include "ie/qwen4_quant.hpp"
#include "ie/dequant_ref.hpp"
#include "ie/quant_blocks.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include <sycl/sycl.hpp>

using namespace ie;

namespace {

int g_fail = 0;

void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? "\033[32mOK\033[0m" : "\033[31mFAIL\033[0m", what);
    if (!ok) ++g_fail;
}

void check_rel(double max_rel, double tol, const char* what) {
    const bool ok = (max_rel <= tol) && std::isfinite(max_rel);
    std::printf("  [%s] %-52s max rel %.3e (tol %.1e)\n",
                ok ? "\033[32mOK\033[0m" : "\033[31mFAIL\033[0m", what, max_rel, tol);
    if (!ok) ++g_fail;
}

void set_half_bits(sycl::half* p, uint16_t bits) { std::memcpy(p, &bits, 2); }
uint16_t half_bits(const sycl::half* p) { uint16_t b; std::memcpy(&b, p, 2); return b; }

block_q5_1 rand_q51(std::mt19937& rng) {
    std::uniform_real_distribution<float> dm(0.01f, 1.0f);
    std::uniform_int_distribution<int> byte(0, 255);
    block_q5_1 b;
    b.d = fp32_to_fp16(dm(rng));
    b.m = fp32_to_fp16(dm(rng));
    for (auto& v : b.qh) v = uint8_t(byte(rng));
    for (auto& v : b.qs) v = uint8_t(byte(rng));
    return b;
}

// f16-quantized random activation in [-1,1]; returns the exact f16 value as
// float and stores its bits into dst.
float rand_act_f16(std::mt19937& rng, sycl::half* dst) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    const uint16_t bits = fp32_to_fp16(u(rng));
    set_half_bits(dst, bits);
    return fp16_to_fp32(bits);
}

// Dequant column-packed Q5_1 W[K,N] -> Wd[n*K + k] fp32 (the golden).
std::vector<float> dequant_W_q51(const block_q5_1* W, uint32_t K, uint32_t N) {
    std::vector<float> Wd(size_t(K) * N);
    const uint32_t bpc = K / 32;
    for (uint32_t n = 0; n < N; ++n)
        ref::dequant_q5_1_buffer(W + size_t(n) * bpc, K, Wd.data() + size_t(n) * K);
    return Wd;
}

// y_ref[m*N+n] = sum_k Wd[n*K+k] * xf[m*K+k], fp64 accumulate.
std::vector<double> gemm_ref_f64(const std::vector<float>& Wd,
                                 const std::vector<float>& xf,
                                 uint32_t M, uint32_t K, uint32_t N) {
    std::vector<double> y(size_t(M) * N);
    for (uint32_t m = 0; m < M; ++m)
        for (uint32_t n = 0; n < N; ++n) {
            double acc = 0.0;
            const float* wc = Wd.data() + size_t(n) * K;
            const float* xr = xf.data() + size_t(m) * K;
            for (uint32_t k = 0; k < K; ++k) acc += double(wc[k]) * double(xr[k]);
            y[size_t(m) * N + n] = acc;
        }
    return y;
}

double max_rel_guarded(const sycl::half* got, const std::vector<double>& ref) {
    double linf = 0.0;
    for (double r : ref) linf = std::max(linf, std::fabs(r));
    const double floor_ = std::max(0.01 * linf, 1e-6);
    double mr = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double g = double(float(got[i]));
        mr = std::max(mr, std::fabs(g - ref[i]) / std::max(std::fabs(ref[i]), floor_));
    }
    return mr;
}

}  // namespace

int main() {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    std::printf("\n\033[1mqwen4exp quant gate\033[0m  device: %s\n",
                q.get_device().get_info<sycl::info::device::name>().c_str());

    std::mt19937 rng(0x51C0DEu);   // fixed seed

    // =====================================================================
    std::printf("\n\033[1m1. GPU Q5_1 GEMV vs CPU fp64 reference\033[0m\n");
    struct Shape { uint32_t K, N; };
    for (const Shape sh : {Shape{640, 2560}, Shape{64, 7}}) {
        const uint32_t K = sh.K, N = sh.N, bpc = K / 32;
        std::vector<block_q5_1> W(size_t(N) * bpc);
        for (auto& b : W) b = rand_q51(rng);
        const auto Wd = dequant_W_q51(W.data(), K, N);

        auto* dW = sycl::malloc_shared<block_q5_1>(W.size(), q);
        auto* dx = sycl::malloc_shared<sycl::half>(K, q);
        auto* dy = sycl::malloc_shared<sycl::half>(N, q);
        std::memcpy(dW, W.data(), W.size() * sizeof(block_q5_1));

        std::vector<float> xf(K);
        for (uint32_t k = 0; k < K; ++k) xf[k] = rand_act_f16(rng, &dx[k]);
        gemv_q5_1(q, dx, dW, dy, K, N).wait();
        const auto yref = gemm_ref_f64(Wd, xf, 1, K, N);
        char what[96];
        std::snprintf(what, sizeof what, "gemv_q5_1 K=%u N=%u random x", K, N);
        check_rel(max_rel_guarded(dy, yref), 2e-3, what);

        if (K == 640) {
            // x = every unit vector: y must reproduce dequantized column k.
            double mr = 0.0;
            for (uint32_t k = 0; k < K; ++k) {
                q.wait();
                for (uint32_t i = 0; i < K; ++i) set_half_bits(&dx[i], 0);
                set_half_bits(&dx[k], 0x3C00);   // f16 1.0
                gemv_q5_1(q, dx, dW, dy, K, N).wait();
                for (uint32_t n = 0; n < N; ++n) {
                    const double r = double(Wd[size_t(n) * K + k]);
                    const double g = double(float(dy[n]));
                    mr = std::max(mr, std::fabs(g - r) / std::max(std::fabs(r), 1e-6));
                }
            }
            check_rel(mr, 1e-3, "gemv_q5_1 x=unit vectors (all 640) == dequant cols");
        }
        sycl::free(dW, q); sycl::free(dx, q); sycl::free(dy, q);
    }

    // =====================================================================
    std::printf("\n\033[1m2. batched rows (gemm_q5_1), M in {1,4,33}\033[0m\n");
    {
        const uint32_t K = 640, N = 2560, bpc = K / 32;
        std::vector<block_q5_1> W(size_t(N) * bpc);
        for (auto& b : W) b = rand_q51(rng);
        const auto Wd = dequant_W_q51(W.data(), K, N);
        auto* dW = sycl::malloc_shared<block_q5_1>(W.size(), q);
        std::memcpy(dW, W.data(), W.size() * sizeof(block_q5_1));

        for (const uint32_t M : {1u, 4u, 33u}) {
            auto* dx = sycl::malloc_shared<sycl::half>(size_t(M) * K, q);
            auto* dy = sycl::malloc_shared<sycl::half>(size_t(M) * N, q);
            std::vector<float> xf(size_t(M) * K);
            for (size_t i = 0; i < xf.size(); ++i) xf[i] = rand_act_f16(rng, &dx[i]);
            gemm_q5_1(q, dx, dW, dy, M, K, N).wait();
            const auto yref = gemm_ref_f64(Wd, xf, M, K, N);
            char what[96];
            std::snprintf(what, sizeof what, "gemm_q5_1 M=%u K=%u N=%u", M, K, N);
            check_rel(max_rel_guarded(dy, yref), 2e-3, what);
            sycl::free(dx, q); sycl::free(dy, q);
        }
        sycl::free(dW, q);
    }

    // =====================================================================
    std::printf("\n\033[1m3. expert-slice addressing on a [K,N,E] Q5_1 bank\033[0m\n");
    {
        const uint32_t K = 64, N = 32, E = 512, bpc = K / 32;
        check(q5_1_expert_stride_bytes(K, N) == uint64_t(bpc) * 24 * N,
              "q5_1_expert_stride_bytes(64,32) == (K/32)*24*N == 1536");
        check(q5_1_expert_stride_bytes(640, 2560) == 1228800ull,
              "q5_1_expert_stride_bytes(640,2560) == 1228800 (real down bank)");

        std::vector<block_q5_1> bank(size_t(E) * N * bpc);
        for (auto& b : bank) b = rand_q51(rng);
        auto* dbank = sycl::malloc_shared<block_q5_1>(bank.size(), q);
        std::memcpy(dbank, bank.data(), bank.size() * sizeof(block_q5_1));
        auto* dx = sycl::malloc_shared<sycl::half>(K, q);
        auto* dy = sycl::malloc_shared<sycl::half>(N, q);
        std::vector<float> xf(K);
        for (uint32_t k = 0; k < K; ++k) xf[k] = rand_act_f16(rng, &dx[k]);

        for (const uint32_t e : {0u, 7u, 511u}) {
            const void* slice = q5_1_expert_slice(dbank, K, N, e);
            gemv_q5_1(q, dx, slice, dy, K, N).wait();
            // CPU ref: index the FULL bank independently (block-index math).
            std::vector<double> yref(N);
            for (uint32_t n = 0; n < N; ++n) {
                double acc = 0.0;
                for (uint32_t b = 0; b < bpc; ++b) {
                    float w32[32];
                    ref::dequant_q5_1(&bank[(size_t(e) * N + n) * bpc + b], w32);
                    for (int i = 0; i < 32; ++i)
                        acc += double(w32[i]) * double(xf[b * 32 + i]);
                }
                yref[n] = acc;
            }
            char what[96];
            std::snprintf(what, sizeof what, "expert e=%u slice GEMV vs full-bank ref", e);
            check_rel(max_rel_guarded(dy, yref), 2e-3, what);
        }
        sycl::free(dbank, q); sycl::free(dx, q); sycl::free(dy, q);
    }

    // =====================================================================
    std::printf("\n\033[1m4. convert_bf16_to_f16 — exhaustive 65536 patterns\033[0m\n");
    {
        std::vector<uint16_t> src(65536);
        std::vector<sycl::half> dst(65536);
        for (uint32_t u = 0; u < 65536; ++u) src[u] = uint16_t(u);
        convert_bf16_to_f16(src.data(), dst.data(), src.size());

        uint32_t bad = 0; uint32_t first_bad = 0; uint32_t n_nan = 0;
        for (uint32_t u = 0; u < 65536; ++u) {
            const float f = std::bit_cast<float>(uint32_t(u) << 16);
            const uint16_t got = half_bits(&dst[u]);
            if (std::isnan(f)) {
                ++n_nan;
                // NaN in -> NaN out: exp all-ones + nonzero mantissa.
                if (!((got & 0x7C00u) == 0x7C00u && (got & 0x03FFu) != 0)) {
                    if (!bad) first_bad = u; ++bad;
                }
            } else {
                const _Float16 r = (_Float16)f;    // independent RTNE oracle
                const uint16_t want = std::bit_cast<uint16_t>(r);
                if (got != want) { if (!bad) first_bad = u; ++bad; }
            }
        }
        char what[96];
        std::snprintf(what, sizeof what,
                      "all 65536 bf16 patterns exact (%u NaN handled), %u bad%s0x%04x",
                      n_nan, bad, bad ? ", first " : " @ ", first_bad);
        check(bad == 0, what);
    }

    // =====================================================================
    std::printf("\n\033[1m5. Q5_K coverage\033[0m\n");
    {
        // Existence finding: ref::dequant_q5_K + dequant_q5_K_buffer already
        // exist in include/ie/dequant_ref.hpp — no new reference dequant was
        // needed; convert_q5_k_to_f16_buffer wraps it.
        std::printf("  (ref::dequant_q5_K pre-exists in dequant_ref.hpp; helper wraps it)\n");

        // 5a. buffer indexing on random blocks (vs per-block ref + RTNE).
        std::uniform_real_distribution<float> dm(0.01f, 0.5f);
        std::uniform_int_distribution<int> byte(0, 255);
        const size_t NB = 8;
        std::vector<block_q5_K> blocks(NB);
        for (auto& b : blocks) {
            b.d = fp32_to_fp16(dm(rng)); b.dmin = fp32_to_fp16(dm(rng));
            for (auto& v : b.scales) v = uint8_t(byte(rng));
            for (auto& v : b.qh) v = uint8_t(byte(rng));
            for (auto& v : b.qs) v = uint8_t(byte(rng));
        }
        std::vector<sycl::half> out(NB * 256);
        convert_q5_k_to_f16_buffer(blocks.data(), NB * 256, out.data());
        uint32_t bad = 0;
        for (size_t i = 0; i < NB; ++i) {
            float y[256];
            ref::dequant_q5_K(&blocks[i], y);
            for (int j = 0; j < 256; ++j)
                if (half_bits(&out[i * 256 + j]) != fp32_to_fp16(y[j])) ++bad;
        }
        check(bad == 0, "convert_q5_k_to_f16_buffer == RTNE(ref::dequant_q5_K), 8 blocks");

        // 5b. hand-crafted block: y = d*sc*q - dmin*m with known values.
        block_q5_K hb{};
        hb.d = fp32_to_fp16(0.5f); hb.dmin = fp32_to_fp16(0.25f);
        hb.scales[0] = 2;  hb.scales[4] = 3;   // sub 0: sc=2, m=3
        hb.scales[1] = 5;  hb.scales[5] = 1;   // sub 1: sc=5, m=1
        hb.qs[0] = 0x37;                       // elem 0 low-nib 7, elem 32 hi-nib 3
        hb.qh[0] = 0x03;                       // 5th bit set for elems 0 and 32
        sycl::half hout[256];
        convert_q5_k_to_f16_buffer(&hb, 256, hout);
        // elem 0 : q = 7+16 = 23 -> 0.5*2*23 - 0.25*3 = 22.25
        // elem 32: q = 3+16 = 19 -> 0.5*5*19 - 0.25*1 = 47.25
        // elem 1 : q = 0        -> 0        - 0.25*3 = -0.75
        // elem 64: sub 2 sc=m=0 -> 0
        check(float(hout[0])  == 22.25f, "hand block elem 0  == d*sc*q - dmin*m = 22.25");
        check(float(hout[32]) == 47.25f, "hand block elem 32 == 47.25");
        check(float(hout[1])  == -0.75f, "hand block elem 1  == -0.75");
        check(float(hout[64]) == 0.0f,   "hand block elem 64 == 0");
    }

    // =====================================================================
    std::printf("\n%s\n\n", g_fail == 0
        ? "\033[1;32mGATE PASSED\033[0m"
        : "\033[1;31mGATE FAILED\033[0m");
    return g_fail == 0 ? 0 : 1;
}
