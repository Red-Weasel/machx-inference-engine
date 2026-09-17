// tests/unit/gemv_q2_0_soa_test.cpp — numeric correctness of the native Q2_0
// (Prism ternary 2-bit) SoA W2A8 int-dot decode GEMV vs a reference fp16 path.
//
// Pipeline mirrors the task's "kernel numeric correctness" gate:
//   1. Random fp32 weights [K,N]; quantize each 128-elem block to Q2_0 EXACTLY
//      like llama.cpp-prism quantize_row_q2_0_ref (d = amax, code = round(w/d)+1
//      clamped [0,3], value = (code-1)*d).
//   2. Reference: dequant_q2_0 (ref) → fp16 weight [K,N]; fp16 GEMV vs a random
//      fp16 activation (host, fp32 accum) → y_ref[N].
//   3. Device: repack_q2_0_to_soa → device streams; quantize_q8_1(act) →
//      gemv_q2_0_soa_q8 → y_dev[N].
//   4. Compare. int8-activation W2A8 is NOT bit-exact vs fp16-dequant, so we gate
//      on cosine ≥ 0.999 and median rel-err small. A big mismatch = packing/scale
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

// Quantize one 128-element block to Q2_0, verbatim per quantize_row_q2_0_ref.
void quantize_block_q2_0(const float* x, block_q2_0& b) {
    float amax = 0.f;
    for (int j = 0; j < 128; ++j) amax = std::max(amax, std::fabs(x[j]));
    const float d = amax;
    const float id = d > 0.f ? 1.f / d : 0.f;
    b.d = fp32_to_fp16(d);
    for (int j = 0; j < 32; ++j) b.qs[j] = 0;
    for (int j = 0; j < 128; ++j) {
        int q = int(std::lround(x[j] * id)) + 1;
        if (q < 0) q = 0;
        if (q > 3) q = 3;
        b.qs[j >> 2] |= uint8_t(q << ((j & 3) * 2));
    }
}
}  // namespace

int main() {
    sycl::queue q{sycl::default_selector_v};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.f);

    // A few shapes from the Ternary-Bonsai-27B GGUF (K multiple of 128).
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

        // ---- AoS Q2_0 [N cols][bpc blocks] + reference fp16 dequant [K,N] ----
        std::vector<block_q2_0> aos(uint64_t(N) * bpc);
        std::vector<sycl::half> Wref(uint64_t(K) * N);  // [k*N + n] fp16 dequant
        float blk[128], dq[128];
        for (uint32_t n = 0; n < N; ++n) {
            for (uint32_t b = 0; b < bpc; ++b) {
                for (int e = 0; e < 128; ++e) blk[e] = Wf[uint64_t(b * 128 + e) * N + n];
                block_q2_0& bb = aos[uint64_t(n) * bpc + b];
                quantize_block_q2_0(blk, bb);
                ref::dequant_q2_0(&bb, dq);
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
        std::vector<uint8_t>  h_qs(uint64_t(N) * (K / 4), 0);
        std::vector<uint16_t> h_d(uint64_t(N) * (K / 128), 0);
        repack_q2_0_to_soa(aos.data(), K, N, h_qs.data(), h_d.data());

        auto* d_qs = sycl::malloc_device<uint8_t>(h_qs.size(), q);
        auto* d_d  = sycl::malloc_device<uint16_t>(h_d.size(), q);
        auto* d_A  = sycl::malloc_device<sycl::half>(K, q);
        auto* d_q8 = sycl::malloc_device<block_q8_1x>(K / 32, q);
        auto* d_y  = sycl::malloc_device<sycl::half>(N, q);
        q.memcpy(d_qs, h_qs.data(), h_qs.size()).wait();
        q.memcpy(d_d,  h_d.data(),  h_d.size() * sizeof(uint16_t)).wait();
        q.memcpy(d_A,  Ah.data(),   K * sizeof(sycl::half)).wait();

        // ---- device: quantize act → gemv_q2_0_soa_q8 ----
        sycl::event qe = quantize_q8_1(q, d_A, d_q8, K);
        gemv_q2_0_soa_q8(q, d_q8, d_qs, d_d, d_y, K, N, {qe}).wait();

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

        const bool ok = cos >= 0.999 && med <= 0.05;
        std::printf("  %-24s cos=%.6f  med_rel=%.4f  p95_rel=%.4f  %s\n",
                    s.name, cos, med, p95, ok ? "OK" : "FAIL");
        if (!ok) ++g_fail;

        sycl::free(d_qs, q); sycl::free(d_d, q); sycl::free(d_A, q);
        sycl::free(d_q8, q); sycl::free(d_y, q);
    }

    // ---- BATCHED-T case (gemv_q2_0_soa_q8_batched, IE_Q2_BATCHED prefill):
    // T=5 exercises the wrapper's T-slicing (one T_TILE=4 tile + a single-row
    // remainder through the T==1 kernel). Same pipeline as above extended to T
    // activation rows (row t's Q8_1 blocks at t*(K/32), one batched
    // quantize_q8_1 over [T,K]); same gates (cos ≥ 0.999, med_rel ≤ 0.05),
    // applied PER ROW vs the fp16-dequant reference.
    {
        const uint32_t K = 5120, N = 1024, T = 5;
        const uint32_t bpc = K / 128;

        // ---- random weights [K,N] + AoS Q2_0 + reference fp16 dequant ----
        std::vector<float> Wf(uint64_t(K) * N);
        for (auto& v : Wf) v = nd(rng);
        std::vector<block_q2_0> aos(uint64_t(N) * bpc);
        std::vector<sycl::half> Wref(uint64_t(K) * N);  // [k*N + n] fp16 dequant
        float blk[128], dq[128];
        for (uint32_t n = 0; n < N; ++n) {
            for (uint32_t b = 0; b < bpc; ++b) {
                for (int e = 0; e < 128; ++e) blk[e] = Wf[uint64_t(b * 128 + e) * N + n];
                block_q2_0& bb = aos[uint64_t(n) * bpc + b];
                quantize_block_q2_0(blk, bb);
                ref::dequant_q2_0(&bb, dq);
                for (int e = 0; e < 128; ++e)
                    Wref[uint64_t(b * 128 + e) * N + n] = sycl::half(dq[e]);
            }
        }

        // ---- random activations [T,K] fp16 (row-major) ----
        std::vector<sycl::half> Ah(uint64_t(T) * K);
        for (auto& v : Ah) v = sycl::half(0.5f * nd(rng));

        // ---- reference GEMM: y_ref[t*N+n] = Σ_k Wref[k,n]·A[t,k] (fp32) ----
        std::vector<float> y_ref(uint64_t(T) * N, 0.f);
        for (uint32_t t = 0; t < T; ++t)
            for (uint32_t n = 0; n < N; ++n) {
                float acc = 0.f;
                for (uint32_t k = 0; k < K; ++k)
                    acc += float(Wref[uint64_t(k) * N + n]) * float(Ah[uint64_t(t) * K + k]);
                y_ref[uint64_t(t) * N + n] = acc;
            }

        // ---- repack to SoA (host) + upload ----
        std::vector<uint8_t>  h_qs(uint64_t(N) * (K / 4), 0);
        std::vector<uint16_t> h_d(uint64_t(N) * (K / 128), 0);
        repack_q2_0_to_soa(aos.data(), K, N, h_qs.data(), h_d.data());

        auto* d_qs = sycl::malloc_device<uint8_t>(h_qs.size(), q);
        auto* d_d  = sycl::malloc_device<uint16_t>(h_d.size(), q);
        auto* d_A  = sycl::malloc_device<sycl::half>(uint64_t(T) * K, q);
        auto* d_q8 = sycl::malloc_device<block_q8_1x>(uint64_t(T) * (K / 32), q);
        auto* d_y  = sycl::malloc_device<sycl::half>(uint64_t(T) * N, q);
        q.memcpy(d_qs, h_qs.data(), h_qs.size()).wait();
        q.memcpy(d_d,  h_d.data(),  h_d.size() * sizeof(uint16_t)).wait();
        q.memcpy(d_A,  Ah.data(),   uint64_t(T) * K * sizeof(sycl::half)).wait();

        // ---- device: batched quantize [T,K] → batched int-dot GEMV ----
        sycl::event qe = quantize_q8_1(q, d_A, d_q8, T * K);
        gemv_q2_0_soa_q8_batched(q, d_q8, d_qs, d_d, d_y, K, N, T, {qe}).wait();

        std::vector<sycl::half> y_dev(uint64_t(T) * N);
        q.memcpy(y_dev.data(), d_y, uint64_t(T) * N * sizeof(sycl::half)).wait();

        // ---- compare PER ROW: cosine + median relative error ----
        for (uint32_t t = 0; t < T; ++t) {
            double dot = 0, na = 0, nb = 0;
            std::vector<double> rel;
            for (uint32_t n = 0; n < N; ++n) {
                const double a = y_ref[uint64_t(t) * N + n];
                const double b = float(y_dev[uint64_t(t) * N + n]);
                dot += a * b; na += a * a; nb += b * b;
                const double denom = std::max(std::fabs(a), 1e-3);
                rel.push_back(std::fabs(a - b) / denom);
            }
            const double cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
            std::sort(rel.begin(), rel.end());
            const double med = rel[rel.size() / 2];
            const double p95 = rel[size_t(rel.size() * 0.95)];

            const bool ok = cos >= 0.999 && med <= 0.05;
            std::printf("  batched T=5 row %u [5120,1024]  cos=%.6f  med_rel=%.4f  p95_rel=%.4f  %s\n",
                        t, cos, med, p95, ok ? "OK" : "FAIL");
            if (!ok) ++g_fail;
        }

        sycl::free(d_qs, q); sycl::free(d_d, q); sycl::free(d_A, q);
        sycl::free(d_q8, q); sycl::free(d_y, q);
    }

    std::printf(g_fail ? "\nRESULT: %d FAIL\n" : "\nRESULT: all shapes OK\n", g_fail);
    return g_fail ? 1 : 0;
}
