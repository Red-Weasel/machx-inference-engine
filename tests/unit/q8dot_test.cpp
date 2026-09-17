// tests/unit/q8dot_test.cpp — integer-dot GEMV (P1a) correctness.
//
// Validates quantize_q8_1 + gemv_q4_K_q8 against a CPU fp64 reference of the
// SAME math chain (Q4_K dequant exactly as the engine does it; activations
// through the same Q8_1 rounding).  Two comparisons:
//   (a) vs the q8-chain reference: tight tolerance (kernel implements the
//       intended integer math exactly);
//   (b) vs the full-precision fp reference: loose tolerance (~1% rel) —
//       bounds the activation-quantization error itself.

#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

bool g_fail = false;
void report(const char* name, bool ok, const char* extra = "") {
    std::printf("  %s%-52s%s %s%s\n", ok ? "\033[32m" : "\033[31m", name,
                "\033[0m", ok ? "OK" : "FAIL", extra);
    if (!ok) g_fail = true;
}

void get_scale_min_k4(int j, const uint8_t* s, uint8_t& sc, uint8_t& m) {
    if (j < 4) { sc = s[j] & 63; m = s[j + 4] & 63; }
    else {
        sc = (s[j + 4] & 0x0F) | ((s[j - 4] >> 6) << 4);
        m  = (s[j + 4] >>   4) | ((s[j    ] >> 6) << 4);
    }
}

}  // namespace

int main() {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    std::printf("Device: %s\n\n",
                q.get_device().get_info<sycl::info::device::name>().c_str());

    constexpr uint32_t K = 2048, N = 512;
    const uint32_t bpc = K / 256;

    // Random Q4_K weights (valid packed blocks) + fp16 activations.
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> byte_d(0, 255);
    std::uniform_real_distribution<float> xf(-2.f, 2.f);

    std::vector<ie::block_q4_K> W(uint64_t(N) * bpc);
    for (auto& b : W) {
        b.d    = uint16_t(0x2E66);   // ~0.1 in fp16
        b.dmin = uint16_t(0x2A66);   // ~0.05
        for (auto& s : b.scales) s = uint8_t(byte_d(rng));
        for (auto& v : b.qs)     v = uint8_t(byte_d(rng));
    }
    std::vector<sycl::half> x(K);
    for (auto& v : x) v = sycl::half(xf(rng));

    // CPU: dequant W exactly as the engine lattice does.
    auto wval = [&](uint32_t n, uint32_t k) {
        const ie::block_q4_K& b = W[uint64_t(n) * bpc + k / 256];
        const uint32_t e = k % 256;
        const int g = int(e / 64), rem = int(e % 64);
        const int hi_nib = rem / 32, i = rem % 32;
        const int sub = g * 2 + hi_nib;
        uint8_t sc, m;
        get_scale_min_k4(sub, b.scales, sc, m);
        const float d    = float(sycl::half(sycl::bit_cast<sycl::half>(b.d)));
        const float dmin = float(sycl::half(sycl::bit_cast<sycl::half>(b.dmin)));
        const uint8_t qb = b.qs[g * 32 + i];
        const int q4 = hi_nib ? (qb >> 4) : (qb & 0x0F);
        return d * float(sc) * float(q4) - dmin * float(m);
    };

    // CPU Q8_1 chain (mirrors quantize_q8_1).
    std::vector<float> xq(K);
    for (uint32_t b = 0; b < K / 32; ++b) {
        float amax = 0.f;
        for (uint32_t i = 0; i < 32; ++i)
            amax = std::max(amax, std::fabs(float(x[b * 32 + i])));
        const float d = amax / 127.f, inv = amax > 0.f ? 127.f / amax : 0.f;
        for (uint32_t i = 0; i < 32; ++i)
            xq[b * 32 + i] = d * std::round(float(x[b * 32 + i]) * inv);
    }
    std::vector<double> ref_q8(N, 0.0), ref_fp(N, 0.0);
    for (uint32_t n = 0; n < N; ++n)
        for (uint32_t k = 0; k < K; ++k) {
            ref_q8[n] += double(wval(n, k)) * double(xq[k]);
            ref_fp[n] += double(wval(n, k)) * double(float(x[k]));
        }

    // GPU.
    auto* dW  = sycl::malloc_device<ie::block_q4_K>(W.size(), q);
    auto* dx  = sycl::malloc_device<sycl::half>(K, q);
    auto* dq8 = sycl::malloc_device<ie::block_q8_1x>(K / 32, q);
    auto* dy  = sycl::malloc_device<sycl::half>(N, q);
    q.memcpy(dW, W.data(), W.size() * sizeof(ie::block_q4_K)).wait();
    q.memcpy(dx, x.data(), K * sizeof(sycl::half)).wait();
    ie::quantize_q8_1(q, dx, dq8, K).wait();
    ie::gemv_q4_K_q8(q, dq8, dW, dy, K, N).wait();
    std::vector<sycl::half> y(N);
    q.memcpy(y.data(), dy, N * sizeof(sycl::half)).wait();

    // (a) vs q8-chain reference.
    {
        float max_rel = 0.f;
        for (uint32_t n = 0; n < N; ++n) {
            const float r = float(ref_q8[n]);
            const float g = float(y[n]);
            const float rel = std::fabs(g - r) / std::max(1.0f, std::fabs(r));
            max_rel = std::max(max_rel, rel);
        }
        char extra[64];
        std::snprintf(extra, sizeof(extra), "  max_rel=%.2e (gate 5e-3)", max_rel);
        report("gemv_q4_K_q8 vs q8-chain CPU ref", max_rel < 5e-3f, extra);
    }
    // (b) vs full-precision reference — activation-quant noise, measured
    // against the SIGNAL SCALE (rms of the reference vector).  Per-element
    // relative error is meaningless on random data: outputs near zero from
    // sign cancellation make tiny absolute noise look huge.
    {
        double rms = 0.0;
        for (uint32_t n = 0; n < N; ++n) rms += ref_fp[n] * ref_fp[n];
        rms = std::sqrt(rms / N);
        float max_scaled = 0.f;
        for (uint32_t n = 0; n < N; ++n) {
            const float e = std::fabs(float(y[n]) - float(ref_fp[n]));
            max_scaled = std::max(max_scaled, float(e / rms));
        }
        char extra[64];
        std::snprintf(extra, sizeof(extra),
                      "  max_err/rms=%.2e (gate 3e-2, rms=%.0f)", max_scaled, rms);
        report("gemv_q4_K_q8 quant noise vs signal scale", max_scaled < 3e-2f, extra);
    }

    sycl::free(dW, q); sycl::free(dx, q); sycl::free(dq8, q); sycl::free(dy, q);
    std::printf("\n%s\n", g_fail ? "FAILURES" : "ALL PASS");
    return g_fail ? 1 : 0;
}
