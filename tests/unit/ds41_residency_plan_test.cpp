// tests/unit/ds41_residency_plan_test.cpp — the V4.1 three-tier planner, host-only.
#include "ie/deepseek41_residency.hpp"

#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

// Not assert(): this target is compiled with -DNDEBUG, which compiled every assert out and left
// a test that could not fail (Phase 8 gate, finding 6). CHECK fails the process regardless.
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "CHECK FAILED at line %d: %s\n", __LINE__, #c); return 1; } } while (0)

int main() {
    const uint32_t L = 40, E = 384;
    const uint64_t slot = 18800640;                  // 17.93 MiB: w1+w3+w2 FP4 planes + E8M0 scales
    const uint64_t GiB = 1ull << 30;

    // The real box: 2 x B70 with ~23 GiB each left for experts after 4.4 GiB of dense per
    // card and a reserve; ~190 GiB of pinnable host RAM.
    ie::Ds41ResidencyPlan p;
    std::string e = ie::ds41_plan_residency(L, E, slot, 2, 23 * GiB, 190 * GiB, 8, p);
    CHECK(e.empty());
    std::printf("2 cards: static %u + stream %u VRAM, %u pinned, %u mmap per layer | VRAM %.1f GiB, pinned %.1f GiB, mmap %.1f GiB\n",
                p.vram_static_per_layer, p.vram_stream_per_layer, p.pinned_per_layer, p.mmap_per_layer,
                double(p.vram_bytes_total) / GiB, double(p.host_pinned_bytes) / GiB, double(p.mmap_bytes) / GiB);
    CHECK(p.first_layer[0] == 0 && p.first_layer[1] == 20 && p.first_layer[2] == 40);
    CHECK(p.vram_static_per_layer + p.vram_stream_per_layer + p.pinned_per_layer + p.mmap_per_layer >= E);
    CHECK(p.vram_bytes_total <= 2 * 23 * GiB && p.host_pinned_bytes <= 190 * GiB);
    // every expert has a tier; nothing is refused (V4's planner would have)
    CHECK(p.mmap_per_layer > 0 && "V4.1 needs the third tier on this box");
    CHECK(p.vram_bytes_total + p.host_pinned_bytes + p.mmap_bytes >= uint64_t(E) * slot * L);

    // Tier hit accounting: a ranking where selections concentrate on the top ranks must
    // yield a VRAM share far above the uniform static fraction.
    std::vector<std::vector<uint32_t>> rank(L, std::vector<uint32_t>(E));
    std::vector<std::vector<uint64_t>> cnt(L, std::vector<uint64_t>(E));
    for (uint32_t l = 0; l < L; ++l) {
        std::iota(rank[l].begin(), rank[l].end(), 0u);
        for (uint32_t r = 0; r < E; ++r) cnt[l][r] = uint64_t(1000000 / (r + 1));   // Zipf-ish, rank r = expert r
    }
    const auto h = ie::ds41_tier_hits(p, rank, cnt);
    const double uniform_vram = double(p.vram_static_per_layer) / E;
    std::printf("Zipf profile: hits VRAM %.1f%%  pinned %.1f%%  mmap %.1f%%   (uniform VRAM share would be %.1f%%)\n",
                100 * h.vram, 100 * h.pinned, 100 * h.mmap, 100 * uniform_vram);
    CHECK(h.vram > 2 * uniform_vram);
    // and a uniform profile is the control: VRAM share == static fraction
    for (auto& c : cnt) std::fill(c.begin(), c.end(), 1);
    const auto u = ie::ds41_tier_hits(p, rank, cnt);
    std::printf("uniform profile: VRAM %.1f%%  (expected %.1f%%)\n", 100 * u.vram, 100 * uniform_vram);
    CHECK(std::abs(u.vram - uniform_vram) < 1e-9);

    // one card, and a box with enough RAM that the mmap tier vanishes
    ie::Ds41ResidencyPlan q;
    e = ie::ds41_plan_residency(L, E, slot, 1, 46 * GiB, 512 * GiB, 8, q);
    CHECK(e.empty() && q.mmap_per_layer == 0);
    std::printf("1 card, 512 GiB RAM: mmap tier %u (vanishes when RAM suffices)\n", q.mmap_per_layer);
    CHECK(!ie::ds41_plan_residency(L, E, slot, 0, 1, 1, 1, q).empty());
    std::printf("ds41_residency_plan_test: all OK\n");
    return 0;
}
