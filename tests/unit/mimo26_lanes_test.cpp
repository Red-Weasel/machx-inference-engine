// tests/unit/mimo26_lanes_test.cpp -- MiMo-V2.6 lanes' host bookkeeping (P4 B1, include/ie/mimo26_lanes.hpp): the per-lane
// VRAM arithmetic against the design's numbers, the fit refusal, the commit rule (eos / length / outbox / the repetition
// window), admission to the least recently used idle lane, the serial round robin, a mid-run finish and re-admit. CPU only:
//   g++ -std=c++20 -O1 -Wall -Wextra -I include tests/unit/mimo26_lanes_test.cpp src/model/mimo26_lanes.cpp
#include "ie/mimo26_lanes.hpp"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("[%s] %s\n", ok ? " ok " : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

// MiMo-V2.6-Flash's attention geometry: 48 layers, 9 full (4 kv heads, V padded to 192), 39 SWA (8 kv heads, V 128),
// head_dim 192; the full layers are every 5th from layer 0, nine of them (not the checkpoint's hybrid pattern -- only the counts matter here)
std::vector<ie::Mimo26LaneLayer> mimo_layers(uint32_t L0, uint32_t L1) {
    std::vector<ie::Mimo26LaneLayer> v;
    for (uint32_t L = L0; L < L1; ++L) {
        const bool full = L % 5 == 0 && L / 5 < 9;
        v.push_back({!full, full ? 4u : 8u});
    }
    return v;
}

void test_bytes() {
    const auto all = mimo_layers(0, 48);
    uint32_t n_full = 0; for (const auto& l : all) n_full += !l.swa;
    check(n_full == 9 && all.size() == 48, "fixture: 9 full + 39 SWA layers");
    // per token of context: the 9 full layers' K + padded V = 9 x 4 x (192 + 192) x 2 B = 27,648 B (the design's 27.6 KB/token)
    const uint64_t d = ie::mimo26_lane_kv_bytes(all, 192, 128, 2176, 32769) - ie::mimo26_lane_kv_bytes(all, 192, 128, 2176, 32768);
    check(d == 27648, "one more position costs 27,648 B (9 full layers x 4 kv x 384 x 2 B)");
    // the SWA rings alone: 39 x 8 x 2,176 x 320 x 2 B = 434.5 MB (+ the K slack rows)
    std::vector<ie::Mimo26LaneLayer> swa; for (const auto& l : all) if (l.swa) swa.push_back(l);
    const uint64_t rings = ie::mimo26_lane_kv_bytes(swa, 192, 128, 2176, 0);
    check(rings == 39ull * 8 * 2176 * 320 * 2 + 39ull * 64 * 192 * 2, "SWA rings: 39 x 8 x 2,176 x (192 + 128) x 2 B + slack");
    // one lane at 32K, both cards together: ~1.34 GB (the design's ~1.36 GB, ~0.68 GB per card)
    const uint64_t lane = ie::mimo26_lane_kv_bytes(all, 192, 128, 2176, 32768);
    check(lane > 1'300'000'000ull && lane < 1'400'000'000ull, "one lane at ctx 32,768: " + std::to_string(lane / 1000000) + " MB (design ~1,360)");
    const uint64_t c0 = ie::mimo26_lane_kv_bytes(mimo_layers(0, 24), 192, 128, 2176, 32768), c1 = ie::mimo26_lane_kv_bytes(mimo_layers(24, 48), 192, 128, 2176, 32768);
    check(c0 + c1 == lane, "the cards' shares add up to the lane");
    // linear SWA caches (ring 0): ctx slots per kv head
    const uint64_t lin = ie::mimo26_lane_kv_bytes({{true, 8}}, 192, 128, 0, 1000);
    check(lin == (8ull * 1000 + 64) * 192 * 2 + 8ull * 1000 * 128 * 2, "ring 0: an SWA layer keeps ctx slots");
}

void test_fit() {
    const uint64_t G = 1ull << 30;
    check(ie::mimo26_lanes_fit(4 * G, G, 2, G, 0, 32768).empty(), "fit: 2 x 1 GiB + 1 GiB reserve in 4 GiB free");
    check(ie::mimo26_lanes_fit(3 * G, G, 2, G, 0, 32768).empty(), "fit: exactly full fits");
    const std::string e = ie::mimo26_lanes_fit(3 * G - 1, G, 2, G, 1, 32768);
    check(!e.empty() && e.find("card 1") != std::string::npos && e.find("2048 MiB") != std::string::npos, "fit: one byte short is refused, naming the card and the need: " + e);
    check(ie::mimo26_lanes_fit(0, G, 0, 0, 0, 0).empty(), "fit: no extra lanes always fits");
    // P4 B14: --parallel 16 -- 15 extra lanes of one card's share at 32K (~0.67 GB) against a 20 GiB-free card and a 1.5 GiB reserve
    const uint64_t c0 = ie::mimo26_lane_kv_bytes(mimo_layers(0, 24), 192, 128, 2176, 32768), res = 1536ull << 20;
    check(ie::mimo26_lanes_fit(15 * c0 + res, c0, 15, res, 0, 32768).empty() && !ie::mimo26_lanes_fit(15 * c0 + res - 1, c0, 15, res, 0, 32768).empty(),
          "fit: 15 extra lanes at 32K admitted exactly at 15 x " + std::to_string(c0 >> 20) + " MiB + the reserve, refused a byte short");
    const std::string e16 = ie::mimo26_lanes_fit(8 * G, c0, 15, res, 0, 32768);
    check(e16.find("15 more lane(s) at ctx 32768 need " + std::to_string((15 * c0) >> 20) + " MiB on card 0, which has 8192 MiB free less a 1536 MiB reserve") !=
              std::string::npos, "fit: --parallel 16 on 8 GiB free refused with the numbers: " + e16);
}

void test_commit() {
    ie::Mimo26Lane l; l.max_new = 3;
    const std::vector<int32_t> eos = {7, 9};
    check(ie::mimo26_lane_commit(l, 5, eos, false) && l.next == 5 && l.outbox == std::vector<int32_t>{5} && l.n_new == 1, "commit: a plain id goes on");
    check(!ie::mimo26_lane_commit(l, 9, eos, false) && l.finish == "stop" && l.n_new == 1 && l.outbox.size() == 1 && l.next == -1,
          "commit: an eos id stops the lane and is not committed");
    ie::Mimo26Lane m; m.max_new = 2;
    check(ie::mimo26_lane_commit(m, 9, eos, true) && m.n_new == 1, "commit: ignore_eos commits an eos id");
    check(!ie::mimo26_lane_commit(m, 4, eos, true) && m.finish == "length" && m.outbox == std::vector<int32_t>({9, 4}), "commit: max_new ends it with length, the last id committed");
    ie::Mimo26Lane r; r.max_new = 10000;
    r.recent.assign(ie::Mimo26Lane::kRecent, 1);
    for (int32_t i = 0; i < 20; ++i) ie::mimo26_lane_commit(r, 100 + i, {}, false);
    check(r.recent.size() == ie::Mimo26Lane::kRecent && r.recent.back() == 119 && r.recent[r.recent.size() - 20] == 100,
          "commit: the repetition window keeps the last 512 ids");
}

void test_set() {
    ie::Mimo26LaneSet s(3);
    check(s.size() == 3 && s.busy() == 0 && s.next() == -1, "set: 3 idle lanes, nothing to step");
    std::vector<int32_t> p(600); for (int32_t i = 0; i < 600; ++i) p[size_t(i)] = i;
    const int a = s.admit(p, 4, 11), b = s.admit({1, 2, 3}, 4, 12), c = s.admit({4}, 4, 13);
    check(a == 0 && b == 1 && c == 2 && s.busy() == 3, "admit: lanes 0, 1, 2 in order");
    check(s.admit({5}, 4, 14) == -1, "admit: every lane busy -> -1");
    check(s.lane(0).recent.size() == 512 && s.lane(0).recent.front() == 88 && s.lane(0).rng == 11 && s.lane(0).prefill_at == 0,
          "admit: the window is the prompt's last 512, rng = seed, prefill from 0");
    std::vector<int> order; for (int k = 0; k < 7; ++k) order.push_back(s.next());
    check(order == std::vector<int>({0, 1, 2, 0, 1, 2, 0}), "next: round robin 0 1 2 0 1 2 0");
    // lane 1 finishes mid-run: the others go on, lane 1 is skipped
    s.lane(1).live = {1, 2, 3, 42};
    s.release(1);
    order.clear(); for (int k = 0; k < 4; ++k) order.push_back(s.next());
    check(order == std::vector<int>({2, 0, 2, 0}) && s.busy() == 2, "release: the finished lane leaves the round robin");
    check(s.lru_idle() == 1 && s.lane(1).live.size() == 4, "release: the idle lane keeps its live ids");
    // a new sequence takes the idle lane and rejoins the round robin after the last stepped lane (0)
    const int d = s.admit({8, 8}, 2, 15);
    check(d == 1 && s.lane(1).outbox.empty() && s.lane(1).finish.empty() && s.lane(1).n_new == 0 && s.lane(1).live.size() == 4,
          "admit mid-run: the idle lane, a clean sampler / outbox, its live ids left for the prefix step");
    order.clear(); for (int k = 0; k < 3; ++k) order.push_back(s.next());
    check(order == std::vector<int>({1, 2, 0}), "admit mid-run: it rejoins the round robin in index order");
    // LRU among several idle lanes: the one stepped longest ago
    s.release(0); s.release(2);   // 0 was stepped after 2
    check(s.lru_idle() == 2, "lru_idle: the lane stepped longest ago");
    s.release(1);
    check(s.busy() == 0 && s.next() == -1, "all released: nothing to step");
    ie::Mimo26LaneSet one(0);
    check(one.size() == 1 && one.admit({1}, 1, 0) == 0 && one.next() == 0 && one.next() == 0, "a set of one lane (the default) always steps lane 0");
}

}  // namespace

int main() {
    test_bytes();
    test_fit();
    test_commit();
    test_set();
    std::printf("\nMIMO26 LANES TEST: %s\n", g_fail ? "FAILURE(S)" : "PASS");
    return g_fail ? 1 : 0;
}
