// tests/unit/lanes_auto_test.cpp -- host-only (no GPU, no SYCL): P4 B30's lane pick (include/ie/lanes_auto.hpp), with
// fake free-VRAM numbers per arch. The plan per arch (one card, a refusing switch, an arch without lanes), the measured pick
// (the edges, never above what fits, never below what fits, the same rule as an explicit --parallel's admission), the 27B's
// load-time budget as the engine checks it, and the load line. The parser side (no --parallel = auto, an explicit
// --parallel N untouched, the --max-queue default) is tests/unit/serve_options_test.cpp. Does NOT link ie_core.
#undef NDEBUG
#include "ie/lanes_auto.hpp"
#include "ie/q35m_lanes.hpp"   // Q35mLaneShape / q35m_lane_bytes / q35m_lanes_fit: the explicit admission the pick must agree with
#include "ie/q27_lanes.hpp"    // q27_lane_ctx
#include "ie/q4e_lanes.hpp"    // q4e_lane_ctx

#include <cstdio>
#include <string>
#include <vector>

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("[%s] %s\n", ok ? " ok " : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

constexpr uint64_t GiB = 1ull << 30, MiB = 1ull << 20;

// One card of the crown split (layers [0, 20): 5 full attention + 15 DeltaNet; the q35m_lanes_test shape)
ie::Q35mLaneShape crown_card() {
    ie::Q35mLaneShape sh;
    sh.n_full = 5; sh.n_kv_heads = 2; sh.head_dim = 256;
    sh.n_lin = 15; sh.v_heads = 32; sh.k_head_dim = 128; sh.v_head_dim = 128; sh.conv_channels = 8192; sh.conv_kernel = 4;
    return sh;
}
// One card of the 27B split (32 layers: 8 full attention + 24 DeltaNet; the q27_lanes_test shape)
// A lane's DeltaNet part (the crown's sticky checkpoint, Q35mLanesModel::init_sticky's per-lane need)
uint64_t dn_bytes(ie::Q35mLaneShape sh) { sh.n_full = 0; return ie::q35m_lane_bytes(sh, 0); }

// the pick is the largest n <= n_max every card holds: n fits, n + 1 does not (or n == n_max)
bool maximal(const std::vector<ie::LanesCardRoom>& cards, uint32_t n_max, uint32_t n) {
    auto fits = [&](uint32_t k) {
        for (const auto& c : cards) if (ie::lanes_auto_need(c, k) > c.avail) return false;
        return true;
    };
    return n >= 1 && n <= n_max && (n == 1 || fits(n)) && (n == n_max || !fits(n + 1));
}

void test_plan() {
    using A = ie::LanesArch;
    static_assert(ie::kLanesAuto == 0 && ie::kLanesAutoMax == 16);
    for (A a : {A::kCrownSplit, A::kQwen35Split}) {
        const auto p = ie::lanes_auto_plan(a, 2, true, 0);
        check(p.n == 16 && p.measure && p.slot_ctx == 0, "plan: a split arch on two cards measures up to 16 lanes at its default ctx");
        check(ie::lanes_auto_plan(a, 2, true, 8192).slot_ctx == 8192, "plan: a split arch keeps an explicit --slot-ctx");
    }
    for (A a : {A::kMimo26, A::kDeepSeek41, A::kFlashNext}) {
        const auto p = ie::lanes_auto_plan(a, 2, true, 0);
        check(p.n == 4 && !p.measure && p.slot_ctx == 16384, "plan: a host-expert arch picks 4 lanes at slot ctx 16384");
        const auto q = ie::lanes_auto_plan(a, 2, true, 32768);
        check(q.n == 4 && q.slot_ctx == 32768, "plan: a host-expert arch keeps an explicit --slot-ctx");
        check(ie::lanes_auto_plan(a, 3, true, 0).n == 4, "plan: three cards = the same pick");
    }
    for (A a : {A::kCrownSplit, A::kQwen35Split, A::kMimo26, A::kDeepSeek41, A::kFlashNext}) {
        const auto one = ie::lanes_auto_plan(a, 1, true, 0), off = ie::lanes_auto_plan(a, 2, false, 0);
        check(one.n == 1 && !one.measure && one.slot_ctx == 0, "plan: one card = one lane (the lanes need the two-card pipe)");
        check(off.n == 1 && !off.measure, "plan: a switch that turns the arch's lanes off or refuses them = one lane");
    }
    for (uint32_t cards : {1u, 2u, 4u}) {
        const auto p = ie::lanes_auto_plan(A::kOther, cards, true, 4096);
        check(p.n == 1 && !p.measure && p.slot_ctx == 4096, "plan: an arch without request lanes = one lane on " + std::to_string(cards) + " card(s)");
    }
    // the lane contexts the archs then derive (their own rules, unchanged): the crown min(ctx, 32768), the 27B
    // min(ctx, 65536), the host-expert archs min(ctx, 16384) under auto
    check(ie::q4e_lane_ctx(ie::lanes_auto_plan(A::kCrownSplit, 2, true, 0).slot_ctx, 262144) == 32768 &&
              ie::q27_lane_ctx(ie::lanes_auto_plan(A::kQwen35Split, 2, true, 0).slot_ctx, 262144) == 65536 &&
              ie::q4e_lane_ctx(ie::lanes_auto_plan(A::kFlashNext, 2, true, 0).slot_ctx, 262144) == 16384 &&
              ie::q4e_lane_ctx(ie::lanes_auto_plan(A::kFlashNext, 2, true, 0).slot_ctx, 8192) == 8192,
          "plan: lane ctx = the arch's default (crown 32768, 27B 65536) or 16384 on the host-expert archs, capped at --ctx");
}

void test_fit_edges() {
    const uint64_t lane = 350 * MiB, per = 31 * MiB, keep = 3 * GiB;
    std::vector<ie::LanesCardRoom> c{{0, lane, per, keep}};
    c[0].avail = ie::lanes_auto_need(c[0], 16);
    check(ie::lanes_auto_fit(c, 16) == 16, "fit: exactly 16 lanes' need fits 16");
    c[0].avail -= 1;
    check(ie::lanes_auto_fit(c, 16) == 15, "fit: one byte short of 16 picks 15");
    c[0].avail = ie::lanes_auto_need(c[0], 2);
    check(ie::lanes_auto_fit(c, 16) == 2, "fit: exactly two lanes' need picks 2");
    c[0].avail -= 1;
    check(ie::lanes_auto_fit(c, 16) == 1, "fit: one byte short of two picks 1");
    c[0].avail = 0;
    check(ie::lanes_auto_fit(c, 16) == 1, "fit: no free memory still answers 1 (never 0: lane 0 is the load's own state)");
    c[0].avail = 1000 * GiB;
    check(ie::lanes_auto_fit(c, 16) == 16 && ie::lanes_auto_fit(c, 5) == 5 && ie::lanes_auto_fit(c, 1) == 1,
          "fit: plenty of memory picks n_max (16, or a lower cap)");
    check(ie::lanes_auto_fit(std::vector<ie::LanesCardRoom>{}, 16) == 16, "fit: no card constraint = n_max");
    // two cards: the tighter one decides
    std::vector<ie::LanesCardRoom> two{{0, lane, per, keep}, {0, lane, per, keep}};
    two[0].avail = ie::lanes_auto_need(two[0], 16);
    two[1].avail = ie::lanes_auto_need(two[1], 9);
    check(ie::lanes_auto_fit(two, 16) == 9, "fit: card 1 holds 9 and card 0 16 -> 9");
    std::swap(two[0], two[1]);
    check(ie::lanes_auto_fit(two, 16) == 9, "fit: the same with the cards swapped");
    // never above what fits and never below it, over a sweep of free memory, keeps and lane sizes, on two uneven cards
    bool ok = true;
    uint64_t picks = 0;
    for (uint64_t a0 = 0; a0 <= 14 * GiB; a0 += 97 * MiB)
        for (uint64_t lb : {90 * MiB, 350 * MiB, 2200 * MiB})
            for (uint64_t pl : {uint64_t(0), 31 * MiB})
                for (uint64_t kp : {1536 * MiB, 1536 * MiB + 2 * GiB})
                    for (uint32_t n_max : {16u, 12u, 1u}) {
                        std::vector<ie::LanesCardRoom> cs{{a0, lb, pl, kp}, {a0 + 300 * MiB, lb + 7 * MiB, pl, kp + 64 * MiB}};
                        const uint32_t n = ie::lanes_auto_fit(cs, n_max);
                        ok = ok && maximal(cs, n_max, n);
                        ++picks;
                    }
    check(ok, "fit: " + std::to_string(picks) + " picks, each the largest n <= n_max that every card holds");
    bool mono = true;   // more free memory never picks fewer lanes
    uint32_t prev = 1;
    for (uint64_t a = 0; a <= 12 * GiB; a += 13 * MiB) {
        const uint32_t n = ie::lanes_auto_fit(std::vector<ie::LanesCardRoom>{{a, lane, per, keep}}, 16);
        mono = mono && n >= prev;
        prev = n;
    }
    check(mono, "fit: monotone in the free memory");
}

void test_crown() {
    const ie::Q35mLaneShape sh = crown_card();
    const uint64_t lane = ie::q35m_lane_bytes(sh, 32768), ck = dn_bytes(sh), res = 1536 * MiB, rows = 150 * MiB;
    check(lane == 335544320ull + 15ull * (2097152ull + 49152ull) && ck == 15ull * (2097152ull + 49152ull),
          "crown: an extra lane at 32K = 350.2 MiB per card, its sticky checkpoint (the DeltaNet part) 30.7 MiB");
    // the pick agrees with an explicit --parallel's admission (q35m_lanes_fit) when nothing more is kept: the pick is admitted,
    // one more lane is refused
    bool agree = true;
    for (uint64_t a = 0; a <= 9 * GiB; a += 61 * MiB) {
        const uint32_t n = ie::lanes_auto_fit(std::vector<ie::LanesCardRoom>{{a, lane, 0, res + rows}}, 16);
        if (n > 1) agree = agree && ie::q35m_lanes_fit(a, lane, n, 32768, res + rows, 0).empty();
        if (n < 16) agree = agree && !ie::q35m_lanes_fit(a, lane, n + 1, 32768, res + rows, 0).empty();
    }
    check(agree, "crown: with only the reserve kept, the pick == the most lanes an explicit --parallel would be admitted");
    // Dream's crown flags (--gpus 2 --ctx 262144): the B29 gate's 16-lane load had 3.65 GiB free after the lanes, the sampler
    // buffers and 16 checkpoints (491 MiB), so ~4.14 GiB before the checkpoints -- the arithmetic, not a measurement
    const uint64_t after = uint64_t(3.65 * double(GiB)) + 16 * ck + 8 * MiB;
    const uint64_t keep = res + rows + ie::kLanesAutoCacheFloor;
    const uint64_t before = after + 15 * lane + rows;
    std::vector<ie::LanesCardRoom> c{{before, lane, ck, keep}, {before + 512 * MiB, lane, ck, keep}};
    check(ie::lanes_auto_fit(c, 16) == 16, "crown: the B29 gate's numbers pick 16 (the cache floor 1.5 GiB + 16 checkpoints fit)");
    const uint64_t margin = before - ie::lanes_auto_need(c[0], 16);
    check(margin > 612 * MiB && margin < 812 * MiB,   // the 2 GiB floor's 100..300 MiB + the 512 MiB it gave back
          "crown: ... with a margin of " + std::to_string(margin >> 20) + " MiB (a desktop app on that card can tip it to 15)");
    c[0].avail -= margin + 1;
    check(ie::lanes_auto_fit(c, 16) == 15, "crown: one byte less than 16 lanes' need on card 0 -> 15");
    // after a 16-lane pick, the sticky check (n checkpoints + the reserve, Q35mLanesModel::init_sticky) passes and the
    // prompt cache's budget (free - reserve, P4 B29) is at least the floor
    c[0].avail += margin + 1;
    const uint64_t left = c[0].avail - 15 * lane - rows;   // what init_lanes leaves
    check(left >= 16 * ck + res && left - 16 * ck - res >= ie::kLanesAutoCacheFloor,
          "crown: the pick leaves the sticky check passing and >= 1.5 GiB of prompt-cache budget");
    // --ctx 32768 (the B29 4/16-lane loads: ~11.6 GiB free before the lanes): 16, with room to spare
    std::vector<ie::LanesCardRoom> c32{{11600 * MiB, lane, ck, keep}, {11800 * MiB, lane, ck, keep}};
    check(ie::lanes_auto_fit(c32, 16) == 16, "crown: --ctx 32768 (~11.6 GiB free) -> 16");
    // a crowded card: 7 GiB free, the cache on -> fewer lanes; the cache off (--no-prompt-cache) and sticky off -> more
    std::vector<ie::LanesCardRoom> crowd{{7 * GiB, lane, ck, keep}};
    std::vector<ie::LanesCardRoom> crowd_off{{7 * GiB, lane, 0, res + rows}};
    const uint32_t n_on = ie::lanes_auto_fit(crowd, 16), n_off = ie::lanes_auto_fit(crowd_off, 16);
    check(n_on == 11 && n_off == 16,   // 9 under the 2 GiB floor; the 512 MiB it gave back plus the margin fit 2 more lanes
          "crown: 7 GiB free -> " + std::to_string(n_on) + " with the cache floor and checkpoints, " +
                                        std::to_string(n_off) + " with the cache and sticky lanes off");
    // --slot-ctx 16384 halves the lanes' KV: more fit on the same card
    std::vector<ie::LanesCardRoom> c16{{7 * GiB, ie::q35m_lane_bytes(sh, 16384), ck, keep}};
    check(ie::lanes_auto_fit(c16, 16) == 16, "crown: 7 GiB free at --slot-ctx 16384 -> 16");
}


void test_line() {
    const std::vector<ie::LanesCardRoom> c{{9500 * MiB, 350 * MiB, 31 * MiB, 3686 * MiB}, {9932 * MiB, 350 * MiB, 31 * MiB, 3686 * MiB}};
    const std::string s = ie::lanes_auto_line("qwen35moe", 16, 32768, "the most, up to 16, that fit", c);
    check(s.find("[lanes] auto (no --parallel): qwen35moe picks 16 request lanes (lanes 1..15 at ctx 32768): the most") == 0 &&
              s.find("card 0: 9.28 GiB free, 15 x 0.342 extra lanes + 16 x 0.030 checkpoints + 3.60 kept = 9.21") != std::string::npos &&
              s.find("card 1: 9.70 GiB free") != std::string::npos && s.find("--parallel N sets the count") != std::string::npos,
          "line: N, lane ctx and every card's numbers: " + s);
    const std::string one = ie::lanes_auto_line("glm5next", 1, 0, "no request lanes on this arch");
    check(one == "[lanes] auto (no --parallel): glm5next serves one request at a time: no request lanes on this arch; "
                 "--parallel N sets the count", "line: one lane: " + one);
    const std::string host = ie::lanes_auto_line("mimo_v2", 4, 16384, "the fixed pick");
    check(host.find("mimo_v2 picks 4 request lanes (lanes 1..3 at ctx 16384): the fixed pick;") != std::string::npos &&
              host.find("card") == std::string::npos,
          "line: a fixed pick names no card numbers (the arch's own lane lines do): " + host);
    const std::string nock = ie::lanes_auto_line("qwen35", 2, 65536, "", std::vector<ie::LanesCardRoom>{{4 * GiB, GiB, 0, GiB}});
    check(nock.find("card 0: 4.00 GiB free, 1 x 1.000 extra lanes + 1.00 kept = 2.00") != std::string::npos &&
              nock.find("checkpoints") == std::string::npos,
          "line: no checkpoint term when lanes keep none: " + nock);
}

}  // namespace

int main() {
    test_plan();
    test_fit_edges();
    test_crown();
    test_line();
    std::printf("%s (%d failure(s))\n", g_fail ? "FAIL" : "ALL PASS", g_fail);
    return g_fail ? 1 : 0;
}
