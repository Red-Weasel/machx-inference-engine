// tests/unit/cpu_moe_mxfp4_test.cpp — the CPU MXFP4 expert kernel (docs/deepseek41/34, Phase 13
// step 2): the nibble table and the E8M0 scale against the reference formulas, the AVX2 GEMV
// against the scalar dequant + fp32 dot on random planes, and the expert composition (gate, up,
// silu-clamp, down) against a scalar recomputation. No model, no GPU.
#include "ie/cpu_moe_mxfp4.hpp"
#include "ie/deepseek41_experts.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {
int g_fail = 0;
void check(bool ok, const char* w, const std::string& d = "") { std::printf("%s%s%s\n", ok ? "[ ok ] " : "[FAIL] ", w, d.empty() ? "" : ("  (" + d + ")").c_str()); if (!ok) ++g_fail; }
// the reference nibble formula: exp = (nb>>1)&3, mant = nb&1, mag = exp ? (2+mant) << (exp-1) : mant, sign = bit 3
int nibble_ref(uint32_t nb) { const int ex = int((nb >> 1) & 3u), mant = int(nb & 1u); const int mag = ex ? ((2 + mant) << (ex - 1)) : mant; return (nb & 8u) ? -mag : mag; }
double rel(const std::vector<float>& a, const std::vector<float>& b) { double m = 0, s = 0; for (size_t i = 0; i < a.size(); ++i) { m = std::max(m, std::fabs(double(a[i]) - double(b[i]))); s = std::max(s, std::fabs(double(b[i]))); } return s > 0 ? m / s : 0; }
}  // namespace

int main() {
    // 1. the tables
    { bool all = true; for (uint32_t nb = 0; nb < 16; ++nb) all &= ie::cpu_mxfp4_nibble_int(nb) == nibble_ref(nb);
      check(all, "the 16 nibble values equal the exponent-branch reference formula"); }
    { bool all = true;
      for (uint32_t e = 2; e < 255; ++e) all &= ie::cpu_mxfp4_e8m0_half(uint8_t(e)) == std::ldexp(1.0f, int(e) - 128);   // 2^(e-127) / 2
      check(all, "E8M0 half-scale equals 2^(e-128) for e in 2..254");
      check(ie::cpu_mxfp4_e8m0_half(0) == std::ldexp(1.0f, -128) && ie::cpu_mxfp4_e8m0_half(1) == std::ldexp(1.0f, -127), "the two subnormal scales (e = 0, 1)"); }

    // 2. the GEMV on random planes, shapes of the model (gate/up K = 5120, N = 2304; down K = 2304, N = 5120) and an odd one
    std::mt19937 rng(7);
    for (auto [K, N] : {std::pair<uint32_t, uint32_t>{5120, 2304}, {2304, 5120}, {256, 40}}) {
        const uint32_t nb = K / 32;
        std::vector<uint8_t> qs(size_t(N) * nb * 16), e(size_t(N) * nb);
        for (auto& b : qs) b = uint8_t(rng() & 0xFF);
        for (auto& s : e) s = uint8_t(120 + (rng() % 16));            // scales around 2^-7..2^8
        std::vector<float> x(K); for (auto& v : x) v = float(int(rng() % 2001) - 1000) / 1000.f;
        std::vector<float> w(size_t(N) * K); ie::cpu_dequant_mxfp4_ref(qs.data(), e.data(), w.data(), K, N);
        std::vector<float> ref(N, 0.f);
        for (uint32_t n = 0; n < N; ++n) { double acc = 0; for (uint32_t k = 0; k < K; ++k) acc += double(w[size_t(n) * K + k]) * x[k]; ref[n] = float(acc); }
        std::vector<float> y(N, 0.f);
        ie::cpu_gemv_mxfp4_f32(x.data(), qs.data(), e.data(), y.data(), K, N, 8);
        const double r = rel(y, ref);
        char buf[64]; std::snprintf(buf, sizeof buf, "K %u N %u rel %.2e", K, N, r);
        check(r < 1e-5, "AVX2 GEMV == scalar dequant + double-accumulated dot within 1e-5", buf);
        std::vector<float> y1(N, 0.f); ie::cpu_gemv_mxfp4_f32(x.data(), qs.data(), e.data(), y1.data(), K, N, 1);
        check(y1 == y, "the result does not depend on the team size (rows are independent)");
    }

    // 3. the expert composition over a random slot laid out by ds41_slot_layout
    {
        const uint32_t H = 5120, EF = 2304;
        ie::Ds4SlotLayout lay; if (auto er = ie::ds41_slot_layout(H, EF, lay); !er.empty()) { std::printf("[FAIL] layout: %s\n", er.c_str()); return 1; }
        std::vector<uint8_t> slot(lay.bytes);
        for (auto& b : slot) b = uint8_t(rng() & 0xFF);
        for (const auto* m : {&lay.gate, &lay.up, &lay.down}) for (uint64_t i = 0; i < m->len1; ++i) slot[m->off1 + i] = uint8_t(120 + (rng() % 12));
        std::vector<float> x(H); for (auto& v : x) v = float(int(rng() % 2001) - 1000) / 1000.f;
        std::vector<float> scratch(2 * EF), out(H);
        ie::cpu_expert_mxfp4(slot.data(), lay, x.data(), scratch.data(), out.data(), 10.f, 8);
        // the scalar recomputation
        std::vector<float> wg(size_t(EF) * H), wu(size_t(EF) * H), wd(size_t(H) * EF);
        ie::cpu_dequant_mxfp4_ref(slot.data() + lay.gate.off0, slot.data() + lay.gate.off1, wg.data(), H, EF);
        ie::cpu_dequant_mxfp4_ref(slot.data() + lay.up.off0,   slot.data() + lay.up.off1,   wu.data(), H, EF);
        ie::cpu_dequant_mxfp4_ref(slot.data() + lay.down.off0, slot.data() + lay.down.off1, wd.data(), EF, H);
        std::vector<float> h(EF), ref(H);
        for (uint32_t i = 0; i < EF; ++i) {
            double g = 0, u = 0; for (uint32_t k = 0; k < H; ++k) { g += double(wg[size_t(i) * H + k]) * x[k]; u += double(wu[size_t(i) * H + k]) * x[k]; }
            float gg = float(g), uu = float(u); gg = std::min(gg, 10.f); uu = std::max(std::min(uu, 10.f), -10.f);
            h[i] = (gg / (1.f + std::exp(-gg))) * uu;
        }
        for (uint32_t n = 0; n < H; ++n) { double acc = 0; for (uint32_t k = 0; k < EF; ++k) acc += double(wd[size_t(n) * EF + k]) * h[k]; ref[n] = float(acc); }
        const double r = rel(out, ref);
        char buf[64]; std::snprintf(buf, sizeof buf, "rel %.2e over %u outputs", r, H);
        check(r < 1e-4, "cpu_expert_mxfp4 == scalar gate/up/silu-clamp/down within 1e-4", buf);
    }
    std::printf("%s\n", g_fail ? "cpu_moe_mxfp4_test: FAILURE(S)" : "cpu_moe_mxfp4_test: all OK");
    return g_fail ? 1 : 0;
}
