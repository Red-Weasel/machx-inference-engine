// tests/unit/gemv_q1_0_soa_test.cpp — numeric correctness of the native Q1_0
// (Prism 1-bit sign-only) SoA W1A8 int-dot decode GEMV vs a reference fp16 path.
//
// Pipeline mirrors gemv_q2_0_soa_test's "kernel numeric correctness" gate:
//   1. Random fp32 weights [K,N]; quantize each 128-elem block to Q1_0 EXACTLY
//      like llama.cpp-prism quantize_row_q1_0_ref (d = mean-abs = Σ|x|/128,
//      bit = (x >= 0), value = (2*bit-1)*d).
//   2. Reference: dequant_q1_0 (ref) → fp16 weight [K,N]; fp16 GEMV vs a random
//      fp16 activation (host, fp32 accum) → y_ref[N].
//   3. Device: repack_q1_0_to_soa → device streams; quantize_q8_1(act) →
//      gemv_q1_0_soa_q8 → y_dev[N].
//   4. Compare. int8-activation W1A8 is NOT bit-exact vs fp16-dequant, but the
//      integer part (Σ sign·q8) IS exact (sign bits are exactly representable), so
//      the only error is q8 activation rounding — tighter than Q2's. Gate on
//      cosine ≥ 0.9999 and small median rel-err. A big mismatch = packing/scale
//      or lane-map bug.

#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"
#include "ie/dequant_ref.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>

using namespace ie;

namespace {
int g_fail = 0;

// Quantize one 128-element block to Q1_0, verbatim per quantize_row_q1_0_ref:
// d = mean-abs = Σ|x|/128; bit = (x >= 0); packed LSB-first in natural order.
void quantize_block_q1_0(const float* x, block_q1_0& b) {
    float sum_abs = 0.f;
    for (int j = 0; j < 128; ++j) sum_abs += std::fabs(x[j]);
    const float d = sum_abs / 128.f;   // mean-abs (NOT amax)
    b.d = fp32_to_fp16(d);
    for (int j = 0; j < 16; ++j) b.qs[j] = 0;
    for (int j = 0; j < 128; ++j) {
        const uint8_t bit = (x[j] >= 0.f) ? 1u : 0u;   // bit = (x >= 0)
        if (bit) b.qs[j >> 3] |= uint8_t(1u << (j & 7));
    }
}
}  // namespace

int main() {
    sycl::queue q{sycl::default_selector_v};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.f);

    // The 5 Bonsai-27B Q1_0 GEMV shapes (K a multiple of 128; K%512==0 too).
    struct Shape { uint32_t K, N; const char* name; };
    const Shape shapes[] = {
        {5120, 1024,  "attn_k/v  [5120,1024]"},
        {5120, 6144,  "attn_gate [5120,6144]"},
        {17408, 5120, "ffn_down  [17408,5120]"},
        {6144, 5120,  "ssm_out   [6144,5120]"},
        {5120, 256,   "small-N   [5120,256]"},
    };

    for (const auto& s : shapes) {
        const uint32_t K = s.K, N = s.N;
        const uint32_t bpc = K / 128;

        // ---- random weights [K,N] (column-major: N cols, each K contiguous) ----
        std::vector<float> Wf(uint64_t(K) * N);
        for (auto& v : Wf) v = nd(rng);

        // ---- AoS Q1_0 [N cols][bpc blocks] + reference fp16 dequant [K,N] ----
        std::vector<block_q1_0> aos(uint64_t(N) * bpc);
        std::vector<sycl::half> Wref(uint64_t(K) * N);  // [k*N + n] fp16 dequant
        float blk[128], dq[128];
        for (uint32_t n = 0; n < N; ++n) {
            for (uint32_t b = 0; b < bpc; ++b) {
                for (int e = 0; e < 128; ++e) blk[e] = Wf[uint64_t(b * 128 + e) * N + n];
                block_q1_0& bb = aos[uint64_t(n) * bpc + b];
                quantize_block_q1_0(blk, bb);
                ref::dequant_q1_0(&bb, dq);
                for (int e = 0; e < 128; ++e)
                    Wref[uint64_t(b * 128 + e) * N + n] = sycl::half(dq[e]);
            }
        }

        // ---- random activation [K] fp16 ----
        std::vector<sycl::half> Ah(K);
        for (uint32_t k = 0; k < K; ++k) Ah[k] = sycl::half(0.5f * nd(rng));

        // ---- reference GEMV: y_ref[n] = Σ_k Wref[k,n]·A[k] (fp32 accum) ----
        std::vector<float> y_ref(N, 0.f);
        for (uint32_t n = 0; n < N; ++n) {
            float acc = 0.f;
            for (uint32_t k = 0; k < K; ++k)
                acc += float(Wref[uint64_t(k) * N + n]) * float(Ah[k]);
            y_ref[n] = acc;
        }

        // ---- repack to SoA (host) + upload ----
        std::vector<uint8_t>  h_qs(uint64_t(N) * (K / 8), 0);
        std::vector<uint16_t> h_d(uint64_t(N) * (K / 128), 0);
        repack_q1_0_to_soa(aos.data(), K, N, h_qs.data(), h_d.data());

        auto* d_qs = sycl::malloc_device<uint8_t>(h_qs.size(), q);
        auto* d_d  = sycl::malloc_device<uint16_t>(h_d.size(), q);
        auto* d_A  = sycl::malloc_device<sycl::half>(K, q);
        auto* d_q8 = sycl::malloc_device<block_q8_1x>(K / 32, q);
        auto* d_y  = sycl::malloc_device<sycl::half>(N, q);
        q.memcpy(d_qs, h_qs.data(), h_qs.size()).wait();
        q.memcpy(d_d,  h_d.data(),  h_d.size() * sizeof(uint16_t)).wait();
        q.memcpy(d_A,  Ah.data(),   K * sizeof(sycl::half)).wait();

        // ---- device: quantize act → gemv_q1_0_soa_q8 ----
        sycl::event qe = quantize_q8_1(q, d_A, d_q8, K);
        gemv_q1_0_soa_q8(q, d_q8, d_qs, d_d, d_y, K, N, {qe}).wait();

        std::vector<sycl::half> y_dev(N);
        q.memcpy(y_dev.data(), d_y, N * sizeof(sycl::half)).wait();

        // ---- compare: cosine + median relative error ----
        double dot = 0, na = 0, nb = 0;
        std::vector<double> rel;
        for (uint32_t n = 0; n < N; ++n) {
            const double a = y_ref[n], b = float(y_dev[n]);
            dot += a * b; na += a * a; nb += b * b;
            const double denom = std::max(std::fabs(a), 1e-3);
            rel.push_back(std::fabs(a - b) / denom);
        }
        const double cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
        std::sort(rel.begin(), rel.end());
        const double med = rel[rel.size() / 2];
        const double p95 = rel[size_t(rel.size() * 0.95)];

        const bool ok = cos >= 0.9999 && med <= 0.05;
        std::printf("  %-24s cos=%.6f  med_rel=%.4f  p95_rel=%.4f  %s\n",
                    s.name, cos, med, p95, ok ? "OK" : "FAIL");
        if (!ok) ++g_fail;

        sycl::free(d_qs, q); sycl::free(d_d, q); sycl::free(d_A, q);
        sycl::free(d_q8, q); sycl::free(d_y, q);
    }

    std::printf(g_fail ? "\nRESULT: %d FAIL\n" : "\nRESULT: all shapes OK\n", g_fail);
    return g_fail ? 1 : 0;
}
