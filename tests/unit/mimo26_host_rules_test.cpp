// tests/unit/mimo26_host_rules_test.cpp -- the MiMo-V2.6 serving rules that need no device (include/ie/mimo26_host_rules.hpp,
// docs/mimo26/P7_FIX64_FIX70.md): the ring's slot runs, the servable prefix (the P3b ring rule), the continuation test, the
// host-slot admission policy (#70) and the drafter's fp16 range check (#64). CPU only, no SYCL:
//   g++ -std=c++20 -O1 -Wall -Wextra -I include tests/unit/mimo26_host_rules_test.cpp src/model/mimo26_host_rules.cpp
#include "ie/mimo26_host_rules.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("[%s] %s\n", ok ? " ok " : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

float from_bits(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

std::vector<std::pair<uint32_t, uint32_t>> runs_of(uint32_t p0, uint32_t p1, uint32_t R) {
    uint32_t r[4] = {~0u, ~0u, ~0u, ~0u};
    const uint32_t n = ie::mimo26_ring_runs(p0, p1, R, r);
    std::vector<std::pair<uint32_t, uint32_t>> v;
    for (uint32_t i = 0; i < n; ++i) v.push_back({r[2 * i], r[2 * i + 1]});
    return v;
}

std::vector<int32_t> seq(int32_t a, int32_t b) { std::vector<int32_t> v; for (int32_t i = a; i < b; ++i) v.push_back(i); return v; }

void test_ring_runs() {
    using P = std::vector<std::pair<uint32_t, uint32_t>>;
    check(runs_of(5, 5, 10).empty(), "ring runs: an empty range has none");
    check(runs_of(0, 4, 0).empty(), "ring runs: R = 0 (linear) has none");
    check(runs_of(3, 7, 10) == P{{3, 7}}, "ring runs: [3,7) in R 10 is one run [3,7)");
    check(runs_of(8, 13, 10) == P{{8, 10}, {0, 3}}, "ring runs: [8,13) wraps: [8,10) + [0,3)");
    check(runs_of(0, 10, 10) == P{{0, 10}}, "ring runs: exactly R from slot 0 is the whole ring in one run");
    check(runs_of(5, 15, 10) == P{{5, 10}, {0, 5}}, "ring runs: exactly R from slot 5 is two runs covering the ring");
    check(runs_of(7, 10, 10) == P{{7, 10}}, "ring runs: a range ending on the ring's end does not wrap");
    check(runs_of(10, 12, 10) == P{{0, 2}}, "ring runs: a range starting at R maps to slot 0");
    check(runs_of(0, 25, 10) == P{{5, 10}, {0, 5}}, "ring runs: a range longer than R keeps its last R positions [15,25)");
    // brute force: the runs are disjoint, cover exactly the slots of the last min(n, R) positions, and never exceed R
    std::mt19937 g(64070);
    bool ok = true;
    for (int it = 0; it < 20000 && ok; ++it) {
        const uint32_t R = 1 + g() % 300, p0 = g() % 5000, p1 = p0 + g() % 700;
        std::set<uint32_t> want;
        for (uint32_t p = std::max(p0, p1 > R ? p1 - R : 0u); p < p1; ++p) want.insert(p % R);
        std::set<uint32_t> got; size_t total = 0;
        for (auto [a, b] : runs_of(p0, p1, R)) {
            if (!(a < b && b <= R)) ok = false;
            for (uint32_t s = a; s < b; ++s) got.insert(s);
            total += b - a;
        }
        if (got != want || total != want.size()) ok = false;
    }
    check(ok, "ring runs: 20,000 random ranges -- disjoint runs, exactly the slots of the positions the ring holds");
}

void test_servable() {
    const auto h10 = seq(1, 11);   // ids 1..10 at positions 0..9
    auto with = [](std::vector<int32_t> v, std::initializer_list<int32_t> more) { v.insert(v.end(), more); return v; };
    check(ie::mimo26_servable(h10, 10, 0, 128, with(h10, {11})) == 10, "servable: an extending prompt is served whole (linear caches)");
    check(ie::mimo26_servable(h10, 10, 0, 128, h10) == 9, "servable: an identical prompt leaves its last token to run");
    check(ie::mimo26_servable({1, 2, 3, 4}, 4, 0, 128, {1, 2, 9, 4, 5}) == 2, "servable: the common prefix up to the first difference");
    check(ie::mimo26_servable({}, 0, 2176, 128, {1, 2, 3}) == 0, "servable: an empty state serves nothing");
    check(ie::mimo26_servable(h10, 10, 2176, 128, {}) == 0, "servable: an empty prompt is served nothing");
    // the ring rule (Mimo26Forward::written_end, P3b-3): 10,000 held, rejected rows written to 10,005, ring 2,176, window 128
    const auto h = seq(0, 10000);
    auto req = [&](uint32_t share) { auto v = seq(0, int32_t(share)); v.push_back(-7); v.push_back(-8); return v; };
    check(ie::mimo26_servable(h, 10005, 2176, 128, req(9000)) == 9000, "servable: a prefix whose window the ring still holds (10,005 - 9,000 + 128 <= 2,176)");
    check(ie::mimo26_servable(h, 10005, 2176, 128, req(7957)) == 7957, "servable: the boundary written_end - L + window == ring is served");
    check(ie::mimo26_servable(h, 10005, 2176, 128, req(7956)) == 0, "servable: one position further back the ring lost the window: nothing");
    check(ie::mimo26_servable(h, 10005, 0, 128, req(7956)) == 7956, "servable: linear SWA caches (ring 0) have no such limit");
    check(ie::mimo26_servable(h, 9990, 2176, 128, req(7946)) == 0, "servable: written_end below the held size is taken as the held size");
    check(ie::mimo26_servable(h, 9990, 2176, 128, req(7952)) == 7952, "servable: ... and the boundary moves with it");
    // image positions carry negative ids and match like any other id
    check(ie::mimo26_servable({5, -1, -2, -3, 6}, 5, 0, 128, {5, -1, -2, -3, 6, 7}) == 5, "servable: negative (image) ids match like any other id");
    check(ie::mimo26_servable({5, -1, -2, -3, 6}, 5, 0, 128, {5, -1, -9, -3, 6, 7}) == 2, "servable: a different image diverges at its first position");
}

void test_continues() {
    check(ie::mimo26_continues(135000, 135000, 1024), "continues: served up to the prompt's end");
    check(ie::mimo26_continues(136500, 135000, 1024), "continues: served into the generated tail");
    check(ie::mimo26_continues(133976, 135000, 1024), "continues: re-reading exactly `slack` prompt positions still continues");
    check(!ie::mimo26_continues(133975, 135000, 1024), "continues: one more and it is a branch");
    check(!ie::mimo26_continues(3000, 135000, 1024), "continues: a shared system prompt only is a branch");
}

void test_plan_admit() {
    using A = ie::Mimo26Admit;
    const uint64_t G = 1ull << 30;
    std::vector<size_t> v = {99};
    // empty store
    check(ie::mimo26_plan_admit({}, 4 * G, 0, 16 * G, 100 * G, 24 * G, -1, v) == A::kOk && v.empty(), "admit: an empty store takes a slot that fits");
    check(ie::mimo26_plan_admit({}, 17 * G, 5, 16 * G, 100 * G, 24 * G, -1, v) == A::kTooLarge && v.empty(), "admit: a slot larger than the budget is refused (too large)");
    check(ie::mimo26_plan_admit({}, 4 * G, 0, 16 * G, 27 * G, 24 * G, -1, v) == A::kFloor && v.empty(), "admit: MemAvailable would fall under the floor: refused (floor)");
    check(ie::mimo26_plan_admit({}, 3 * G, 0, 16 * G, 27 * G, 24 * G, -1, v) == A::kOk && v.empty(), "admit: MemAvailable exactly at the floor afterwards is allowed");
    // the owner's #70 pattern: a continued 4 GiB conversation A, then one-shot side requests
    std::vector<ie::Mimo26SlotCost> s = {{4 * G, 1, 7}, {G, 2, 0}, {G, 3, 0}};
    check(ie::mimo26_plan_admit(s, G, 0, 6 * G + G / 2, 100 * G, 24 * G, -1, v) == A::kOk && v == std::vector<size_t>{1},
          "admit: over budget, the least recently used ONE-SHOT slot goes first, not the older continued one");
    check(ie::mimo26_plan_admit(s, 2 * G, 0, 6 * G + G / 2, 100 * G, 24 * G, -1, v) == A::kOk && (v == std::vector<size_t>{1, 2}),
          "admit: one-shot slots are evicted oldest first until the newcomer fits");
    check(ie::mimo26_plan_admit(s, 3 * G, 0, 6 * G + G / 2, 100 * G, 24 * G, -1, v) == A::kBudget && v.empty(),
          "admit: a one-shot newcomer never evicts a continued conversation: refused (budget), nothing evicted");
    check(ie::mimo26_plan_admit(s, 3 * G, 2, 6 * G + G / 2, 100 * G, 24 * G, -1, v) == A::kOk && (v == std::vector<size_t>{1, 2, 0}),
          "admit: a continued newcomer evicts the one-shot slots first, then continued ones");
    // LRU among continued slots, and `keep` is never a victim
    std::vector<ie::Mimo26SlotCost> c = {{4 * G, 9, 3}, {4 * G, 4, 1}, {4 * G, 6, 2}};
    check(ie::mimo26_plan_admit(c, 4 * G, 1, 12 * G, 100 * G, 24 * G, -1, v) == A::kOk && v == std::vector<size_t>{1},
          "admit: among continued slots the least recently used goes first");
    check(ie::mimo26_plan_admit(c, 4 * G, 1, 12 * G, 100 * G, 24 * G, 1, v) == A::kOk && v == std::vector<size_t>{2},
          "admit: the slot being restored (`keep`) is skipped");
    check(ie::mimo26_plan_admit(c, 12 * G, 1, 12 * G, 100 * G, 24 * G, 0, v) == A::kBudget && v.empty(),
          "admit: when only `keep` would make room, the newcomer is refused (budget)");
    // the MemAvailable floor: evictions give their bytes back
    std::vector<ie::Mimo26SlotCost> m = {{2 * G, 1, 0}, {G, 2, 0}};
    check(ie::mimo26_plan_admit(m, 8 * G, 0, 64 * G, 30 * G, 24 * G, -1, v) == A::kOk && v == std::vector<size_t>{0},
          "admit: under the floor, evicting 2 GiB lifts MemAvailable to exactly the need");
    check(ie::mimo26_plan_admit(m, 10 * G, 0, 64 * G, 30 * G, 24 * G, -1, v) == A::kFloor && v.empty(),
          "admit: when every eviction together cannot meet the floor: refused (floor), nothing evicted");
    // both bounds failing reports the budget
    check(ie::mimo26_plan_admit({{4 * G, 1, 3}}, 4 * G, 0, 6 * G, 20 * G, 24 * G, -1, v) == A::kBudget && v.empty(),
          "admit: over budget and under the floor with nothing evictable reports the budget");
    // ties in tick keep index order (a stable, deterministic plan)
    std::vector<ie::Mimo26SlotCost> t = {{G, 5, 0}, {G, 5, 0}, {G, 5, 0}};
    check(ie::mimo26_plan_admit(t, 2 * G, 0, 3 * G, 100 * G, 24 * G, -1, v) == A::kOk && (v == std::vector<size_t>{0, 1}),
          "admit: equal ticks are evicted in index order");
}

// #77: one-shot victims go cheapest-to-rebuild first (fewest tokens), least recently used among equals. The #70 pattern:
// the owner's first turn (123,782 tokens, 3.63 GiB, one-shot until its first continuation) is the OLDEST state when the side
// requests (8k tokens, 0.63 GiB each) fill the budget; before #77 the 20th side state evicted it.
void test_plan_admit_cost() {
    using A = ie::Mimo26Admit;
    const uint64_t G = 1ull << 30, side = G * 63 / 100;
    std::vector<size_t> v;
    std::vector<ie::Mimo26SlotCost> s = {{G * 363 / 100, 1, 0, 123782}};
    for (uint64_t k = 0; k < 19; ++k) s.push_back({side, 2 + k, 0, 8000});   // 3.63 + 19 x 0.63 = 15.6 GiB of 16
    check(ie::mimo26_plan_admit(s, side, 0, 16 * G, 100 * G, 24 * G, -1, v) == A::kOk && v == std::vector<size_t>{1},
          "admit cost: the 20th side state evicts the oldest SIDE state, not the older 123,782-token first turn");
    std::vector<ie::Mimo26SlotCost> m = {{2 * G, 1, 0, 60000}, {G, 2, 0, 3000}, {G, 3, 0, 2000}};
    check(ie::mimo26_plan_admit(m, G, 0, 4 * G, 100 * G, 24 * G, -1, v) == A::kOk && v == std::vector<size_t>{2},
          "admit cost: among one-shot states the fewest tokens go first (2,000 before 3,000, whatever their age)");
    check(ie::mimo26_plan_admit(m, 3 * G, 0, 4 * G, 100 * G, 24 * G, -1, v) == A::kOk && (v == std::vector<size_t>{2, 1, 0}),
          "admit cost: the costly state goes last, when nothing cheaper is left to make room");
    std::vector<ie::Mimo26SlotCost> e = {{G, 7, 0, 5000}, {G, 3, 0, 5000}, {G, 5, 0, 5000}};
    check(ie::mimo26_plan_admit(e, G, 0, 3 * G, 100 * G, 24 * G, -1, v) == A::kOk && v == std::vector<size_t>{1},
          "admit cost: equal token counts fall back to least recently used");
    // continued conversations keep their LRU order (only the one-shot tier is ranked by cost), and still come after
    std::vector<ie::Mimo26SlotCost> c = {{G, 9, 2, 1000}, {G, 4, 1, 90000}, {G, 6, 0, 50000}};
    check(ie::mimo26_plan_admit(c, 2 * G, 1, 3 * G, 100 * G, 24 * G, -1, v) == A::kOk && (v == std::vector<size_t>{2, 1}),
          "admit cost: one-shot first, then continued conversations least recently used first (not by size)");
}

// #78: on a swap for a continuation the slot being restored is dropped once the prompt has run (the live state supersedes
// it), so it must not count against the byte budget when the live conversation is kept first. It still holds its RAM until
// then: the MemAvailable floor is unchanged.
void test_plan_admit_swap() {
    using A = ie::Mimo26Admit;
    const uint64_t G = 1ull << 30;
    std::vector<size_t> v;
    const std::vector<ie::Mimo26SlotCost> s = {{6 * G, 1, 2, 150000}, {3 * G, 2, 3, 80000}};   // slot 0 is being restored
    check(ie::mimo26_plan_admit(s, 5 * G, 1, 11 * G, 100 * G, 24 * G, 0, v) == A::kOk && v == std::vector<size_t>{1},
          "admit swap: before #78 the restored slot's 6 GiB counts: the live state must evict slot 1 to fit");
    check(ie::mimo26_plan_admit(s, 5 * G, 1, 11 * G, 100 * G, 24 * G, 0, v, true) == A::kOk && v.empty(),
          "admit swap: a slot that leaves after the prompt does not count against the budget: nothing evicted");
    check(ie::mimo26_plan_admit(s, 8 * G, 0, 10 * G, 100 * G, 24 * G, 0, v) == A::kBudget && v.empty(),
          "admit swap: counted, a one-shot live state is refused ('held by conversations it may not evict')");
    check(ie::mimo26_plan_admit(s, 7 * G, 0, 10 * G, 100 * G, 24 * G, 0, v, true) == A::kOk && v.empty(),
          "admit swap: not counted, the same live state is admitted (3 + 7 <= 10)");
    check(ie::mimo26_plan_admit(s, 8 * G, 0, 10 * G, 100 * G, 24 * G, 0, v, true) == A::kBudget && v.empty(),
          "admit swap: slots it may not evict still count (3 + 8 > 10: refused)");
    check(ie::mimo26_plan_admit(s, 5 * G, 1, 64 * G, 28 * G, 24 * G, 0, v, true) == A::kOk && v == std::vector<size_t>{1},
          "admit swap: the floor still counts the leaving slot's RAM (MemAvailable 28 < 24 + 5 until slot 1 is evicted)");
    check(ie::mimo26_plan_admit(s, 5 * G, 1, 10 * G, 100 * G, 24 * G, -1, v, true) == A::kOk && v == std::vector<size_t>{0},
          "admit swap: without a slot being restored the flag changes nothing");
}

// #87: the prompt-end snapshot of the SWA rings. P = 74,327 (the owner's prompt), ring 2,176, window 128, margin 128.
void test_prompt_snapshot() {
    uint32_t s0 = 0, hs = 0;
    const uint32_t P = 74327, R = 2176, W = 128, M = 128;
    check(ie::mimo26_prompt_snapshot(P, P, R, W, M, 250000, s0, hs) && s0 == P - W - M && hs == s0 + R,
          "snapshot: positions [P - window - margin, P), described by written_end s0 + ring");
    // what the restored state serves, through the servable rule
    const auto have = seq(0, int32_t(P));
    auto req = [&](uint32_t share, bool extra) { auto v = seq(0, int32_t(share)); if (extra) { v.push_back(-5); v.push_back(-6); } return v; };
    check(ie::mimo26_servable(have, hs, R, W, req(P, true)) == P, "snapshot: a prompt diverging exactly at P (a discarded reply) is served to P");
    check(ie::mimo26_servable(have, hs, R, W, req(P, false)) == P - 1, "snapshot: the same prompt again (regenerate) is served to P - 1");
    check(ie::mimo26_servable(have, hs, R, W, req(P - M, true)) == P - M, "snapshot: a divergence `margin` before P is served");
    check(ie::mimo26_servable(have, hs, R, W, req(P - M - 1, true)) == 0, "snapshot: one position further back the window is not held: nothing");
    // the live state after a 16,384-token reply does not serve P: the #87 incident
    check(ie::mimo26_servable(seq(0, int32_t(P + 16384)), P + 16384, R, W, req(P, true)) == 0, "snapshot: the live state after the reply serves nothing (#87)");
    // an older written_end at the prompt's end (a rewind before this prompt) bounds the range from below
    check(ie::mimo26_prompt_snapshot(P, P + 2100, R, W, M, 250000, s0, hs) && s0 == P + 2100 - R && hs == P + 2100,
          "snapshot: a ring already written past P holds only [written_end - ring, P)");
    check(ie::mimo26_prompt_snapshot(200, 200, R, W, M, 250000, s0, hs) && s0 == 0 && hs == R, "snapshot: a short prompt from 0");
    check(!ie::mimo26_prompt_snapshot(P, P, 0, W, M, 250000, s0, hs), "snapshot: linear SWA caches (ring 0) need none");
    check(!ie::mimo26_prompt_snapshot(0, 0, R, W, M, 250000, s0, hs), "snapshot: an empty prompt needs none");
    // near the capacity no reply can push the ring past the window: the live state always serves [P - margin, P]
    const uint32_t cap = P + R - W - M;
    check(ie::mimo26_prompt_snapshot(P, P, R, W, M, cap, s0, hs) && hs == cap, "snapshot: s0 + ring == capacity still fits");
    check(!ie::mimo26_prompt_snapshot(P, P, R, W, M, cap - 1, s0, hs), "snapshot: past the capacity none is needed");
    bool ok = true;   // ... because every state the capacity allows then serves the margin
    for (uint32_t hi = P; hi <= cap - 1 && ok; ++hi)
        ok = ie::mimo26_servable(have, hi, R, W, req(P - M, true)) == P - M;
    check(ok, "snapshot: with capacity < s0 + ring every written_end up to the capacity serves P - margin from the live state");
    check(!ie::mimo26_prompt_snapshot(P, P, R, W, R - W, 250000, s0, hs), "snapshot: a margin of ring - window or more is refused");
}

void test_first_non_f16() {
    std::vector<float> x = {0.f, -0.f, 1.f, -1.f, 65504.f, -65504.f, from_bits(0x477FEFFFu) /* just under 65520 */, 1e-8f, from_bits(1) /* denormal */};
    check(ie::mimo26_first_non_f16(x.data(), x.size()) == x.size(), "f16 range: finite values up to just under 65,520 all fit (they round to <= 65,504)");
    check(ie::mimo26_first_non_f16(x.data(), 0) == 0, "f16 range: an empty span");
    const std::pair<uint32_t, const char*> bad[] = {
        {0x477FF000u, "65,520 (rounds to inf)"}, {0xC77FF000u, "-65,520"}, {0x47800000u, "65,536"}, {0x7F800000u, "+inf"},
        {0xFF800000u, "-inf"}, {0x7FC00000u, "a quiet NaN"}, {0x7F800001u, "a signalling NaN"}, {0xFFFFFFFFu, "a negative NaN"},
        {0x7F7FFFFFu, "FLT_MAX"}};
    for (auto [bits, name] : bad) {
        std::vector<float> y(10000, 3.f);
        y[5000] = from_bits(bits);
        y[7000] = from_bits(0x7FC00000u);
        check(ie::mimo26_first_non_f16(y.data(), y.size()) == 5000, std::string("f16 range: ") + name + " is found, the first of two");
    }
    std::vector<float> z(8193, 2.f); z[8192] = from_bits(0x7F800000u);
    check(ie::mimo26_first_non_f16(z.data(), z.size()) == 8192, "f16 range: an element past a block boundary is found");
}

void test_scan() {
    // fp32: counts, first non-finite, largest finite magnitude and where (all by bits)
    std::vector<float> x = {1.f, -3.f, 2.f, from_bits(0x7FC00000u), -7.5f, from_bits(0xFF800000u), 7.f};
    auto s = ie::mimo26_scan_f32(x.data(), x.size());
    check(s.non_finite == 2 && s.first_bad == 3 && s.max_abs == 7.5f && s.max_at == 4, "scan f32: two non-finite, the first at 3; max finite |x| 7.5 at 4");
    std::vector<float> clean = {0.f, -0.f, from_bits(1), 65504.f, -1e30f};
    s = ie::mimo26_scan_f32(clean.data(), clean.size());
    check(s.non_finite == 0 && s.first_bad == -1 && s.max_abs == 1e30f && s.max_at == 4, "scan f32: a clean span (a denormal, -0, 1e30)");
    s = ie::mimo26_scan_f32(clean.data(), 0);
    check(s.non_finite == 0 && s.first_bad == -1 && s.max_at == -1, "scan f32: an empty span");
    std::vector<float> all_bad = {from_bits(0x7F800000u), from_bits(0xFFC00001u)};
    s = ie::mimo26_scan_f32(all_bad.data(), all_bad.size());
    check(s.non_finite == 2 && s.first_bad == 0 && s.max_at == -1, "scan f32: nothing finite, no max");
    // fp16 bit patterns: 1.0 = 0x3C00, -2.0 = 0xC000, 65504 = 0x7BFF, +inf 0x7C00, NaN 0x7E00, the smallest denormal 0x0001
    std::vector<uint16_t> h = {0x3C00, 0xC000, 0x7E00, 0x7BFF, 0x0001, 0xFC00, 0x8000};
    auto t = ie::mimo26_scan_f16(h.data(), h.size());
    check(t.non_finite == 2 && t.first_bad == 2 && t.max_abs == 65504.f && t.max_at == 3, "scan f16: NaN and -inf found, max finite 65,504 at 3");
    std::vector<uint16_t> small = {0x0001, 0x8003, 0x0200};
    t = ie::mimo26_scan_f16(small.data(), small.size());
    check(t.non_finite == 0 && t.max_at == 2 && t.max_abs == 3.0517578125e-05f, "scan f16: denormals decode exactly (0x0200 = 2^-15)");
}

}  // namespace

int main() {
    test_ring_runs();
    test_servable();
    test_continues();
    test_plan_admit();
    test_plan_admit_cost();
    test_plan_admit_swap();
    test_prompt_snapshot();
    test_first_non_f16();
    test_scan();
    std::printf("\nMIMO26 HOST RULES: %s\n", g_fail ? "FAILURE(S)" : "PASS");
    return g_fail ? 1 : 0;
}
