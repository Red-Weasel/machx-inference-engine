// tests/unit/cpu_moe_gemv_test.cpp — host Q4_K/Q5_K GEMV vs ie::ref dequant.
// No GPU. Drives the shipped cpu_gemv_* / cpu_moe_expert_q8 entry points.

#include "ie/cpu_moe_gemv.hpp"
#include "ie/dequant_ref.hpp"
#include "ie/quant_blocks.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using ie::block_q4_K;
using ie::block_q5_K;
using ie::fp32_to_fp16;

namespace {

int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? "OK" : "FAIL", what);
    if (!ok) ++g_fail;
}

void fill_q4(block_q4_K* w, int nblocks, std::mt19937& rng) {
    std::uniform_int_distribution<int> u8(0, 255), u6(1, 63);
    for (int i = 0; i < nblocks; ++i) {
        w[i].d    = fp32_to_fp16(0.02f);
        w[i].dmin = fp32_to_fp16(0.002f);
        for (int s = 0; s < 12; ++s) w[i].scales[s] = uint8_t(u6(rng) | ((u6(rng) & 3) << 6));
        for (int q = 0; q < 128; ++q) w[i].qs[q] = uint8_t(u8(rng));
    }
}
void fill_q5(block_q5_K* w, int nblocks, std::mt19937& rng) {
    std::uniform_int_distribution<int> u8(0, 255), u6(1, 63);
    for (int i = 0; i < nblocks; ++i) {
        w[i].d    = fp32_to_fp16(0.02f);
        w[i].dmin = fp32_to_fp16(0.002f);
        for (int s = 0; s < 12; ++s) w[i].scales[s] = uint8_t(u6(rng) | ((u6(rng) & 3) << 6));
        for (int q = 0; q < 32; ++q) w[i].qh[q] = uint8_t(u8(rng));
        for (int q = 0; q < 128; ++q) w[i].qs[q] = uint8_t(u8(rng));
    }
}

}  // namespace

int main() {
    std::mt19937 rng(0xC0DE);
    std::normal_distribution<float> nd(0.f, 1.f);

    // Shapes: one super-block per column — still the real packed layout.
    constexpr uint32_t K = 256, N = 16;
    std::vector<block_q4_K> W4(size_t(N) * (K / 256));
    std::vector<block_q5_K> W5(size_t(N) * (K / 256));
    fill_q4(W4.data(), int(W4.size()), rng);
    fill_q5(W5.data(), int(W5.size()), rng);
    std::vector<float> x(K), y4(N), y5(N), r4(N), r5(N);
    for (float& v : x) v = nd(rng);

    // Golden: ie::ref dequant each column, f32 dot. This is the contract.
    float colw[256];
    for (uint32_t n = 0; n < N; ++n) {
        ie::ref::dequant_q4_K(&W4[n], colw);
        float s = 0.f;
        for (uint32_t k = 0; k < K; ++k) s += x[k] * colw[k];
        r4[n] = s;
        ie::ref::dequant_q5_K(&W5[n], colw);
        s = 0.f;
        for (uint32_t k = 0; k < K; ++k) s += x[k] * colw[k];
        r5[n] = s;
    }

    ie::cpu_gemv_q4k_f32(x.data(), W4.data(), y4.data(), K, N);
    ie::cpu_gemv_q5k_f32(x.data(), W5.data(), y5.data(), K, N);
    double max4 = 0, nrm4 = 0, max5 = 0, nrm5 = 0;
    for (uint32_t n = 0; n < N; ++n) {
        max4 = std::max(max4, std::fabs(double(y4[n]) - r4[n]));
        nrm4 = std::max(nrm4, std::fabs(double(r4[n])));
        max5 = std::max(max5, std::fabs(double(y5[n]) - r5[n]));
        nrm5 = std::max(nrm5, std::fabs(double(r5[n])));
    }
    check(max4 < 1e-3 * (nrm4 + 1.0), "cpu_gemv_q4k_f32 vs ie::ref dequant GEMV");
    check(max5 < 1e-3 * (nrm5 + 1.0), "cpu_gemv_q5k_f32 vs ie::ref dequant GEMV");
    std::printf("      q4 f32 max|d|=%.3e |ref|=%.3e\n", max4, nrm4);
    std::printf("      q5 f32 max|d|=%.3e |ref|=%.3e\n", max5, nrm5);

    // Q8 path: golden is dequant(W) · (d_q8 * qs). Same packed W, shipped quantize.
    std::vector<ie::block_q8_K> xq(K / 256);
    ie::cpu_quantize_q8_K(x.data(), xq.data(), int(K));
    std::vector<float> y4q(N), r4q(N);
    for (uint32_t n = 0; n < N; ++n) {
        ie::ref::dequant_q4_K(&W4[n], colw);
        float s = 0.f;
        for (uint32_t k = 0; k < K; ++k) s += colw[k] * xq[0].d * float(xq[0].qs[k]);
        r4q[n] = s;
    }
    ie::cpu_gemv_q4k_q8(xq.data(), W4.data(), y4q.data(), K, N);
    double maxq = 0, nrmq = 0;
    for (uint32_t n = 0; n < N; ++n) {
        maxq = std::max(maxq, std::fabs(double(y4q[n]) - r4q[n]));
        nrmq = std::max(nrmq, std::fabs(double(r4q[n])));
    }
    check(maxq < 2e-2 * (nrmq + 1.0), "cpu_gemv_q4k_q8 vs ie::ref dequant · Q8_K(x)");
    std::printf("      q4 q8  max|d|=%.3e |ref|=%.3e\n", maxq, nrmq);

    // Tiny expert: H=EF=256, shipped cpu_moe_expert_q8 must run and stay finite.
    std::vector<block_q4_K> gate(256), up(256);
    std::vector<block_q5_K> down(256);
    fill_q4(gate.data(), 256, rng);
    fill_q4(up.data(), 256, rng);
    fill_q5(down.data(), 256, rng);
    std::vector<float> yexp(256, 0.f);
    ie::cpu_moe_expert_q8(x.data(), gate.data(), up.data(), down.data(),
                          256, 256, 10.f, yexp.data());
    int nfin = 0;
    for (float v : yexp) if (std::isfinite(v)) ++nfin;
    check(nfin == 256, "cpu_moe_expert_q8 produces 256 finite outputs");

    // G1b: the per-32-scale expert against the f32-activation composition of
    // the shipped f32 GEMVs (the reference path) — activation quantization is
    // the only difference, so the two must agree to a few percent of the
    // output scale, and the per-32 variant must not be worse than the per-256.
    {
        // The finiteness expert above uses d = 0.02, which drives gate/up to
        // ~90 and pins the SwiGLU at the clamp — chaotic in the rounding, no
        // use as a precision test. Rescale into the real operating range
        // (d ~ 4e-5: gate/up O(1), clamp inactive) for the comparison.
        for (auto& b : gate) { b.d = fp32_to_fp16(4e-5f); b.dmin = fp32_to_fp16(4e-6f); }
        for (auto& b : up)   { b.d = fp32_to_fp16(4e-5f); b.dmin = fp32_to_fp16(4e-6f); }
        for (auto& b : down) { b.d = fp32_to_fp16(4e-5f); b.dmin = fp32_to_fp16(4e-6f); }
        // One expert shares ONE quantized activation vector across its 256
        // outputs, so a single input is one correlated draw of the error —
        // compare the two activation formats on the MEAN over 16 inputs.
        std::vector<float> g32(256), u32(256), m32(256), yref(256), ys(256, 0.f), xs(256);
        double e8_sum = 0, es_sum = 0, worst_s = 0;
        for (int trial = 0; trial < 16; ++trial) {
            for (float& v : xs) v = nd(rng);
            ie::cpu_gemv_q4k_f32(xs.data(), gate.data(), g32.data(), 256, 256);
            ie::cpu_gemv_q4k_f32(xs.data(), up.data(),   u32.data(), 256, 256);
            for (int i = 0; i < 256; ++i) {
                const float g = std::min(g32[i], 10.f), u = std::min(std::max(u32[i], -10.f), 10.f);
                m32[i] = (g / (1.f + std::exp(-g))) * u;
            }
            ie::cpu_gemv_q5k_f32(m32.data(), down.data(), yref.data(), 256, 256);
            ie::cpu_moe_expert_q8(xs.data(), gate.data(), up.data(), down.data(),
                                  256, 256, 10.f, yexp.data());
            ie::cpu_moe_expert_q8s_nt(xs.data(), gate.data(), up.data(), down.data(),
                                      256, 256, 10.f, ys.data(), 8);
            double e8 = 0, es = 0, nrm = 0;
            for (int i = 0; i < 256; ++i) {
                e8 = std::max(e8, std::fabs(double(yexp[i]) - yref[i]));
                es = std::max(es, std::fabs(double(ys[i]) - yref[i]));
                nrm = std::max(nrm, std::fabs(double(yref[i])));
            }
            e8_sum += e8 / (nrm + 1e-30); es_sum += es / (nrm + 1e-30);
            worst_s = std::max(worst_s, es / (nrm + 1e-30));
        }
        check(worst_s < 5e-2, "cpu_moe_expert_q8s_nt vs f32-activation expert (per-32 scales): every input within 5%");
        check(es_sum <= e8_sum * 1.05, "per-32 activation scales not worse than per-256 (mean over 16 inputs)");
        std::printf("      expert mean max|d|/|ref| vs f32: q8 %.3e  q8s %.3e  (worst q8s %.3e)\n",
                    e8_sum / 16, es_sum / 16, worst_s);
    }

    std::printf("%s (%d failure%s)\n", g_fail ? "GATE FAILED" : "GATE PASSED",
                g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
