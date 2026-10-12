// tests/unit/cpu_moe_mxfp4_test.cpp — the CPU MXFP4 expert kernel (docs/deepseek41/34, Phase 13
// step 2): the nibble table and the E8M0 scale against the reference formulas, the AVX2 GEMV
// against the scalar dequant + fp32 dot on random planes, and the expert composition (gate, up,
// silu-clamp, down) against a scalar recomputation. No model, no GPU.
#include "ie/cpu_moe_mxfp4.hpp"
#include "ie/deepseek41_experts.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <sys/mman.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {
int g_fail = 0;
void check(bool ok, const char* w, const std::string& d = "") { std::printf("%s%s%s\n", ok ? "[ ok ] " : "[FAIL] ", w, d.empty() ? "" : ("  (" + d + ")").c_str()); if (!ok) ++g_fail; }
// the reference nibble formula: exp = (nb>>1)&3, mant = nb&1, mag = exp ? (2+mant) << (exp-1) : mant, sign = bit 3
int nibble_ref(uint32_t nb) { const int ex = int((nb >> 1) & 3u), mant = int(nb & 1u); const int mag = ex ? ((2 + mant) << (ex - 1)) : mant; return (nb & 8u) ? -mag : mag; }
double rel(const std::vector<float>& a, const std::vector<float>& b) { double m = 0, s = 0; for (size_t i = 0; i < a.size(); ++i) { m = std::max(m, std::fabs(double(a[i]) - double(b[i]))); s = std::max(s, std::fabs(double(b[i]))); } return s > 0 ? m / s : 0; }
// --bench (B66): one expert at MiMo's shape (H 4096, EF 2048) on R = 1, 2, 4, 8 rows, cycling 64 distinct slots (~0.8 GB,
// past the caches); the team as the CPU leg's (run it with OMP_NUM_THREADS / OMP_PLACES for the cores). ms per expert
// call and the weight bytes' rate.
// --bench-usm: the slots in sycl::malloc_host memory (what the pinned expert arena is) instead of the process heap
// --bench-hot: one slot (13.4 MB, in the 30 MB L3) called again and again: the kernel's rate with the weights cached
int bench(bool usm, uint32_t NS) {
    const uint32_t H = 4096, EF = 2048;
    ie::Ds4SlotLayout lay; if (auto er = ie::ds41_slot_layout(H, EF, lay); !er.empty()) { std::printf("layout: %s\n", er.c_str()); return 1; }
    std::mt19937 rng(7);
    std::vector<uint8_t> heap;
    std::unique_ptr<sycl::queue> q;
    uint8_t* slots_p = nullptr;
    if (usm) { q = std::make_unique<sycl::queue>(sycl::gpu_selector_v); slots_p = sycl::malloc_host<uint8_t>(size_t(NS) * lay.bytes, *q); }
    else { heap.resize(size_t(NS) * lay.bytes); slots_p = heap.data(); }
    std::printf("slots in %s\n", usm ? "sycl::malloc_host" : "the process heap");
    for (size_t i = 0; i < size_t(NS) * lay.bytes; ++i) slots_p[i] = uint8_t(rng() & 0xFF);
    for (uint32_t s = 0; s < NS; ++s) for (const auto* m : {&lay.gate, &lay.up, &lay.down}) for (uint64_t i = 0; i < m->len1; ++i) slots_p[size_t(s) * lay.bytes + m->off1 + i] = uint8_t(118 + (rng() % 16));
    std::vector<float> xs(size_t(ie::kCpuExpertRows) * H); for (auto& v : xs) v = float(int(rng() % 2001) - 1000) / 1000.f;
    std::vector<float> scratch(size_t(ie::kCpuExpertRows) * 2 * EF), out(size_t(ie::kCpuExpertRows) * H);
    for (const uint32_t R : {1u, 2u, 4u, 8u}) {
        const float* xp[ie::kCpuExpertRows]; for (uint32_t r = 0; r < R; ++r) xp[r] = xs.data() + size_t(r) * H;
        double best = 1e30;
        for (int rep = 0; rep < (NS == 1 ? 50 : 3); ++rep) {
            const auto t0 = std::chrono::steady_clock::now();
            for (uint32_t s = 0; s < NS; ++s) {
                const void* slot = slots_p + size_t(s) * lay.bytes;
                if (R == 1) ie::cpu_expert_mxfp4(slot, lay, xp[0], scratch.data(), out.data(), 10.f, 0);
                else ie::cpu_expert_mxfp4_rows(slot, lay, xp, R, scratch.data(), out.data(), 10.f, 0);
            }
            best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / NS);
        }
        std::printf("R %u: %.3f ms per expert (%.3f per row), weights %.1f GB/s\n", R, best, best / R, lay.bytes / (best * 1e6));
    }
    {   // the one-region leg over the same slots, one-row jobs, fp16 stores discarded
        std::vector<ie::CpuExpertJob> jobs(NS);
        for (uint32_t s = 0; s < NS; ++s) { jobs[s] = ie::CpuExpertJob{}; jobs[s].slot = slots_p + size_t(s) * lay.bytes; jobs[s].xs[0] = xs.data(); jobs[s].R = 1; jobs[s].out_row = 0; }
        std::vector<float> fs(ie::cpu_experts_fused_scratch(H, EF));
        double best = 1e30;
        for (int rep = 0; rep < (NS == 1 ? 50 : 3); ++rep) {
            const auto t0 = std::chrono::steady_clock::now();
            ie::cpu_experts_mxfp4_fused(jobs.data(), jobs.size(), lay, fs.data(), 10.f, 0, [](void*, uint32_t, uint32_t, uint32_t, const float*) {}, nullptr);
            best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / NS);
        }
        std::printf("fused R 1: %.3f ms per expert, weights %.1f GB/s\n", best, lay.bytes / (best * 1e6));
    }
    if (usm) sycl::free(slots_p, *q);
    return 0;
}
// --bench-big 4k|2m: 2048 slots (~27 GB) in one region, 4 KB pages (MADV_NOHUGEPAGE) or transparent 2 MB pages
// (MADV_HUGEPAGE), cycled once per pass: a footprint like the pinned arena's, whose page tables do not stay cached
int bench_big(bool huge) {
    const uint32_t H = 4096, EF = 2048, NS = 2048;
    ie::Ds4SlotLayout lay; if (auto er = ie::ds41_slot_layout(H, EF, lay); !er.empty()) { std::printf("layout: %s\n", er.c_str()); return 1; }
    const size_t bytes = (size_t(NS) * lay.bytes + (2u << 20) - 1) & ~size_t((2u << 20) - 1);
    auto* p = static_cast<uint8_t*>(std::aligned_alloc(2u << 20, bytes));
    if (!p) { std::printf("alloc of %zu bytes failed\n", bytes); return 1; }
    madvise(p, bytes, huge ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);
    std::memset(p, 0x37, bytes);   // codes 7 / 3: no NaN, no denormal products
    for (uint32_t s = 0; s < NS; ++s) for (const auto* m : {&lay.gate, &lay.up, &lay.down}) std::memset(p + size_t(s) * lay.bytes + m->off1, 124, m->len1);
    std::vector<float> x(H); for (uint32_t i = 0; i < H; ++i) x[i] = float(int(i % 2001) - 1000) / 1000.f;
    std::vector<float> scratch(2 * EF), out(H);
    std::printf("%u slots, %.1f GB, %s pages\n", NS, bytes / 1e9, huge ? "2 MB (THP)" : "4 KB");
    for (int pass = 0; pass < 3; ++pass) {
        const auto t0 = std::chrono::steady_clock::now();
        for (uint32_t s = 0; s < NS; ++s) ie::cpu_expert_mxfp4(p + size_t(s) * lay.bytes, lay, x.data(), scratch.data(), out.data(), 10.f, 0);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / NS;
        std::printf("pass %d: %.3f ms per expert, weights %.1f GB/s\n", pass, ms, lay.bytes / (ms * 1e6));
    }
    std::free(p);
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--bench") return bench(false, 64);
    if (argc > 1 && std::string(argv[1]) == "--bench-usm") return bench(true, 64);
    if (argc > 1 && std::string(argv[1]) == "--bench-hot") return bench(false, 1);
    if (argc > 2 && std::string(argv[1]) == "--bench-big") return bench_big(std::string(argv[2]) == "2m");
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

    // 4. the multi-row expert: every row bit-identical to the one-row expert on that row, for R = 1..kCpuExpertRows
    {
        const uint32_t H = 5120, EF = 2304, RM = ie::kCpuExpertRows;
        ie::Ds4SlotLayout lay; if (auto er = ie::ds41_slot_layout(H, EF, lay); !er.empty()) { std::printf("[FAIL] layout: %s\n", er.c_str()); return 1; }
        std::vector<uint8_t> slot(lay.bytes);
        for (auto& b : slot) b = uint8_t(rng() & 0xFF);
        for (const auto* m : {&lay.gate, &lay.up, &lay.down}) for (uint64_t i = 0; i < m->len1; ++i) slot[m->off1 + i] = uint8_t(120 + (rng() % 12));
        std::vector<float> xs(size_t(RM) * H); for (auto& v : xs) v = float(int(rng() % 2001) - 1000) / 1000.f;
        std::vector<float> one(size_t(RM) * H), scratch1(2 * EF);
        for (uint32_t r = 0; r < RM; ++r) ie::cpu_expert_mxfp4(slot.data(), lay, xs.data() + size_t(r) * H, scratch1.data(), one.data() + size_t(r) * H, 10.f, 8);
        bool all = true;
        for (uint32_t R = 1; R <= RM; ++R) {
            const float* xp[ie::kCpuExpertRows]; for (uint32_t r = 0; r < R; ++r) xp[r] = xs.data() + size_t(RM - R + r) * H;   // rows RM-R .. RM-1
            std::vector<float> scratch(size_t(R) * 2 * EF), out(size_t(R) * H);
            ie::cpu_expert_mxfp4_rows(slot.data(), lay, xp, R, scratch.data(), out.data(), 10.f, 8);
            all &= std::memcmp(out.data(), one.data() + size_t(RM - R) * H, size_t(R) * H * 4) == 0;
        }
        check(all, "cpu_expert_mxfp4_rows == cpu_expert_mxfp4 per row, bit for bit, R = 1..8");
    }
    // 5. (B66) checksums of the one-row and multi-row experts' exact output bits, for DS4.1's and MiMo's shapes, with and
    // without the SwiGLU clamp: run once as is (the team form) and once with IE_CPU_MXFP4_TEAM=0 -- the checksum lines must
    // be equal, and equal v0.2.21's (B66: 940eb86e012ee103 62d8378a282d4042 083e4939d56bbb5e 65a1bf8d8d1f8a32)
    for (const auto& [H, EF] : {std::pair<uint32_t, uint32_t>{5120, 2304}, {4096, 2048}})
        for (const float limit : {10.f, 0.f}) {
            ie::Ds4SlotLayout lay; if (auto er = ie::ds41_slot_layout(H, EF, lay); !er.empty()) { std::printf("[FAIL] layout: %s\n", er.c_str()); return 1; }
            std::vector<uint8_t> slot(lay.bytes);
            for (auto& b : slot) b = uint8_t(rng() & 0xFF);
            for (const auto* m : {&lay.gate, &lay.up, &lay.down}) for (uint64_t i = 0; i < m->len1; ++i) slot[m->off1 + i] = uint8_t(118 + (rng() % 16));
            const uint32_t RM = ie::kCpuExpertRows;
            std::vector<float> xs(size_t(RM) * H); for (auto& v : xs) v = float(int(rng() % 20001) - 10000) / 1000.f;
            std::vector<float> one(size_t(RM) * H), scratch1(2 * EF), many(size_t(RM) * H), scratch(size_t(RM) * 2 * EF);
            for (uint32_t r = 0; r < RM; ++r) ie::cpu_expert_mxfp4(slot.data(), lay, xs.data() + size_t(r) * H, scratch1.data(), one.data() + size_t(r) * H, limit, 8);
            const float* xp[ie::kCpuExpertRows]; for (uint32_t r = 0; r < RM; ++r) xp[r] = xs.data() + size_t(r) * H;
            ie::cpu_expert_mxfp4_rows(slot.data(), lay, xp, RM, scratch.data(), many.data(), limit, 8);
            auto fnv = [](const std::vector<float>& v) { uint64_t h = 1469598103934665603ull; const auto* p = reinterpret_cast<const uint8_t*>(v.data());
                                                         for (size_t i = 0; i < v.size() * 4; ++i) { h ^= p[i]; h *= 1099511628211ull; } return h; };
            std::printf("checksum H %u EF %u limit %.0f: one %016llx rows %016llx\n", H, EF, double(limit),
                        (unsigned long long)fnv(one), (unsigned long long)fnv(many));
            // the one-region leg (cpu_experts_mxfp4_fused): RM one-row jobs, then one RM-row job, at two team sizes; each
            // output row bit-identical to the per-expert calls above
            for (const int nt : {12, 5}) {
                std::vector<ie::CpuExpertJob> jobs;
                for (uint32_t r = 0; r < RM; ++r) { ie::CpuExpertJob j{}; j.slot = slot.data(); j.xs[0] = xp[r]; j.R = 1; j.out_row = r; jobs.push_back(j); }
                { ie::CpuExpertJob j{}; j.slot = slot.data(); for (uint32_t r = 0; r < RM; ++r) j.xs[r] = xp[r]; j.R = RM; j.out_row = RM; jobs.push_back(j); }
                struct Sink { uint32_t H; std::vector<float> rows; } sink{H, std::vector<float>(size_t(2) * RM * H, -1.f)};
                std::vector<float> fs(ie::cpu_experts_fused_scratch(H, EF));
                ie::cpu_experts_mxfp4_fused(jobs.data(), jobs.size(), lay, fs.data(), limit, nt,
                    [](void* c, uint32_t row, uint32_t k0, uint32_t k1, const float* src) {
                        auto& sk = *static_cast<Sink*>(c); std::copy(src, src + (k1 - k0), sk.rows.data() + size_t(row) * sk.H + k0); }, &sink);
                const bool same1 = std::memcmp(sink.rows.data(), one.data(), size_t(RM) * H * 4) == 0;
                const bool sameR = std::memcmp(sink.rows.data() + size_t(RM) * H, many.data(), size_t(RM) * H * 4) == 0;
                char buf[64]; std::snprintf(buf, sizeof buf, "H %u EF %u limit %.0f, %d threads", H, EF, double(limit), nt);
                check(same1 && sameR, "the one-region leg == the per-expert calls, bit for bit (one-row and 8-row jobs)", buf);
            }
        }
    std::printf("team elementwise: %s\n", ie::cpu_mxfp4_team() ? "on" : "off");
    std::printf("%s\n", g_fail ? "cpu_moe_mxfp4_test: FAILURE(S)" : "cpu_moe_mxfp4_test: all OK");
    return g_fail ? 1 : 0;
}
