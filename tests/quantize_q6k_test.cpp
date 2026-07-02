// tests/quantize_q6k_test.cpp — P3e Task 2 (Step D1). Round-trip the host Q6_K
// encoder through the engine's reference decoder; assert Q6_K error bounds.
#undef NDEBUG
#include "ie/quantize.hpp"
#include "ie/dequant_ref.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

int main() {
    constexpr int K = 1024;            // 4 superblocks
    std::vector<float> x(K), y(K);

    // Deterministic pseudo-random data in [-2, 2], with per-block magnitude
    // variation to exercise the scale search.
    uint32_t s = 1234567u;
    auto rnd = [&]() { s = s * 1103515245u + 12345u; return float((s >> 16) & 0x7fff) / 16384.f - 1.f; };
    float absmax = 0;
    for (int b = 0; b < K / 256; ++b) {
        const float mag = 0.25f + 0.75f * b;     // 0.25, 1.0, 1.75, 2.5
        for (int i = 0; i < 256; ++i) {
            const float v = rnd() * mag;
            x[b * 256 + i] = v;
            absmax = std::fmax(absmax, std::fabs(v));
        }
    }

    std::vector<ie::block_q6_K> blocks(K / 256);
    ie::quantize_row_q6_K(x.data(), blocks.data(), K);
    for (int b = 0; b < K / 256; ++b)
        ie::ref::dequant_q6_K(&blocks[b], y.data() + b * 256);

    double se = 0, sx = 0;
    float  maxerr = 0;
    for (int i = 0; i < K; ++i) {
        const float e = std::fabs(x[i] - y[i]);
        maxerr = std::fmax(maxerr, e);
        se += double(e) * e;
        sx += double(x[i]) * x[i];
    }
    const double rms_rel = std::sqrt(se / sx);
    std::printf("quantize_q6k: rms_rel=%.4f%%  maxabs=%.5f (absmax=%.3f, %.2f%% of absmax)\n",
                rms_rel * 100.0, maxerr, absmax, 100.0 * maxerr / absmax);
    std::fflush(stdout);

    // Q6_K is 6-bit. The decisive correctness signal is max abs error ≈
    // step/2 ≈ 1.5% of absmax (a packing/scale bug blows this up). RMS-rel on
    // UNIFORM stress data is ~1.5% (≈ step/√12 over a flat signal); real
    // Gaussian-ish weights quantize far tighter.
    assert(maxerr < 0.02f * absmax);              // < 2% of absmax — tight
    assert(rms_rel < 0.02);                       // < 2% on uniform stress data

    // All-zero block → zero output, d == 0.
    std::vector<float> z(256, 0.f), zr(256, 1.f);
    ie::block_q6_K zb;
    ie::quantize_row_q6_K(z.data(), &zb, 256);
    ie::ref::dequant_q6_K(&zb, zr.data());
    for (int i = 0; i < 256; ++i) assert(zr[i] == 0.f);

    // ---- Q4_K encoder round-trip (4-bit → looser bounds) ----
    std::vector<float> y4(K);
    std::vector<ie::block_q4_K> b4(K / 256);
    ie::quantize_row_q4_K(x.data(), b4.data(), K);
    for (int b = 0; b < K / 256; ++b)
        ie::ref::dequant_q4_K(&b4[b], y4.data() + b * 256);
    double se4 = 0, sx4 = 0; float maxerr4 = 0;
    for (int i = 0; i < K; ++i) {
        const float e = std::fabs(x[i] - y4[i]);
        maxerr4 = std::fmax(maxerr4, e); se4 += double(e) * e; sx4 += double(x[i]) * x[i];
    }
    const double rms4 = std::sqrt(se4 / sx4);
    std::printf("quantize_q4k: rms_rel=%.4f%%  maxabs=%.5f (%.2f%% of absmax)\n",
                rms4 * 100.0, maxerr4, 100.0 * maxerr4 / absmax);
    std::fflush(stdout);
    assert(maxerr4 < 0.10f * absmax);   // 4-bit: ~step/2
    assert(rms4 < 0.08);                 // uniform stress data
    ie::block_q4_K z4;
    ie::quantize_row_q4_K(z.data(), &z4, 256);
    std::vector<float> z4r(256, 1.f);
    ie::ref::dequant_q4_K(&z4, z4r.data());
    for (int i = 0; i < 256; ++i) assert(z4r[i] == 0.f);

    std::printf("quantize_q6k_test: all OK (Q6_K + Q4_K round-trip within bounds)\n");
    return 0;
}
