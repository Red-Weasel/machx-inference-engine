// include/ie/lanes_auto.hpp -- P4 B30 "lanes size themselves": `ie serve` without --parallel (or with --parallel auto) lets
// the load pick its request lanes; Engine::parallel() (and /props total_slots) then report the pick. --parallel N given
// explicitly is never changed. No SYCL here: tests/unit/lanes_auto_test.cpp.
//
//   * The crown split (qwen35moe) and the 27B split (qwen35): the most lanes, up to 16, that fit at the arch's default lane
//     context, measured from what each card has free at load (init_lanes' auto mode) beside the reserve init_lanes keeps.
//     The crown also keeps every lane's B29 sticky checkpoint and a floor for the prompt cache's snapshots; the 27B stays
//     within its load-time VRAM budget (q27_budget_room).
//   * MiMo-V2.6, DeepSeek-V4.1 and Flash-Next (qwen4exp): a lane's state comes out of the in-VRAM expert cache, so every
//     reserved lane slows every lane: a fixed pick (kLanesAutoHost lanes at kLanesAutoHostSlotCtx), which the arch's own
//     load-time fit check then admits or refuses with its numbers, as it does an explicit --parallel.
//   * Every other arch, one card, or a switch that turns the arch's lanes off or refuses them: one lane, as before.
#pragma once

#include <cstdint>
#include <cstdio>
#include <span>
#include <string>
#include <vector>

namespace ie {

// EngineOptions::parallel 0 = auto: the load picks N.
inline constexpr uint32_t kLanesAuto = 0;
inline constexpr uint32_t kLanesAutoMax = 16;   // = kMaxParallel (engine.hpp static_asserts it)

// The host-expert archs' pick. Measured on MiMo-V2.6 (~/ds41_work/p60/b23/LOG.md, 2026-09-29, lanes_gate --gate scale,
// drafter on): --parallel 4 --slot-ctx 16384 decoded 23.5-23.9 tok/s for one request and 26.4 for four, against 15.8-18.9
// and 19.0 for sixteen at --parallel 16 --slot-ctx 32768, whose 15 extra lanes left 39 / 29 static expert slots per layer
// instead of 69 / 56. A PLACEHOLDER on DeepSeek-V4.1 and Flash-Next (not measured at this pick): V4.1's lanes gain from the
// two cards' overlap (x1.73 at --parallel 2, README), and its 15 extra lanes at 32K took 10 / 4 static slots per layer;
// Flash-Next's extra lane takes about 0.55 GiB per card at 32K out of its expert cache (README).
inline constexpr uint32_t kLanesAutoHost = 4;
inline constexpr uint32_t kLanesAutoHostSlotCtx = 16384;

// The VRAM per card the crown's auto pick leaves for the prompt cache's snapshots, beside the reserve (P4 B29 budgets them
// out of what the lanes leave free). A pick: the B29 gate's 16-lane swarm replay passed with 2.15 GiB of snapshots per card
// (--ctx 262144, ~/ds41_work/p60/b29gate/runs/r2_new_swarm). 1.5 GiB, not 2: the B30 gate measured the 16th lane's margin at
// ~260 MiB with a 2 GiB floor while a browser rendering on card 0 has taken up to 540 MiB above baseline, which would drop
// Dream's load to 15 lanes; 1.5 GiB leaves ~770 MiB (sticky lanes keep conversations in their lanes, so the cache mostly
// holds shared prefixes: a 30K-token one is ~0.34 GiB per card).
inline constexpr uint64_t kLanesAutoCacheFloor = 1536ull << 20;

// The archs whose loads pick lanes; kOther serves one request at a time.
enum class LanesArch : uint8_t { kOther, kCrownSplit, kQwen35Split, kMimo26, kDeepSeek41, kFlashNext };

struct LanesAutoPlan {
    uint32_t n = 1;          // the lanes; with `measure`, the most init_lanes may pick
    uint32_t slot_ctx = 0;   // the --slot-ctx the load uses (0 = the arch's default)
    bool measure = false;    // init_lanes picks n by the free VRAM it measures
};

// What auto gives a load of `arch` on `cards` cards. lanes_ok = false: something on this load turns the arch's lanes off or
// refuses them (IE_Q35MOE_LANES / IE_QWEN35_LANES / IE_Q4E_LANES = 0, --spec on the 27B, IE_P2P on qwen4exp, IE_DS41_SPEC /
// _PROFILE_OUT / _DUMP_ROUTING / _EP on deepseek41). slot_ctx: --slot-ctx (0 = not given).
inline LanesAutoPlan lanes_auto_plan(LanesArch arch, uint32_t cards, bool lanes_ok, uint32_t slot_ctx) {
    LanesAutoPlan p;
    p.slot_ctx = slot_ctx;
    if (cards < 2 || !lanes_ok) return p;
    switch (arch) {
        case LanesArch::kCrownSplit: case LanesArch::kQwen35Split:
            p.n = kLanesAutoMax; p.measure = true; break;
        case LanesArch::kMimo26: case LanesArch::kDeepSeek41: case LanesArch::kFlashNext:
            p.n = kLanesAutoHost; if (!slot_ctx) p.slot_ctx = kLanesAutoHostSlotCtx; break;
        case LanesArch::kOther: break;
    }
    return p;
}

// One card as a measured pick sees it: `avail` bytes; every extra lane takes `lane_bytes`, every lane (lane 0 too)
// `per_lane` more, and `keep` bytes stay free.
struct LanesCardRoom { uint64_t avail = 0, lane_bytes = 0, per_lane = 0, keep = 0; };

inline uint64_t lanes_auto_need(const LanesCardRoom& c, uint32_t n) {
    return uint64_t(n - 1) * c.lane_bytes + uint64_t(n) * c.per_lane + c.keep;
}

// The largest n in [1, n_max] that every card holds (lanes_auto_need <= avail); 1 when no n > 1 fits.
inline uint32_t lanes_auto_fit(std::span<const LanesCardRoom> cards, uint32_t n_max) {
    for (uint32_t n = n_max; n > 1; --n) {
        bool fits = true;
        for (const LanesCardRoom& c : cards) fits = fits && lanes_auto_need(c, n) <= c.avail;
        if (fits) return n;
    }
    return 1;
}

// init_lanes' auto mode (Qwen35MoeSplitModel / Qwen35SplitModel, n_lanes = kLanesAuto): what a card keeps beside the
// reserve, and what was measured and picked.
struct LanesAutoFit {
    uint32_t n_max = kLanesAutoMax;      // in: the most lanes to pick
    bool dn_checkpoint = false;          // in: every lane also keeps a copy of its DeltaNet state (the crown's sticky lanes)
    uint64_t keep = 0;                   // in: more bytes every card keeps free (the crown's kLanesAutoCacheFloor)
    std::vector<LanesCardRoom> cards;    // out: each card's numbers
    uint32_t n = 1;                      // out: the pick
};

// The 27B split's load-time VRAM budget (Engine::load's [budget] check) as a room: a card holds `fixed` (weights, the live
// KV and DeltaNet state, the prompt cache at its caps, the reserve) plus the extra lanes' banks of `bank` bytes, refused above
// 92 % of its `gmem` (that check compares in double; the truncated cap keeps a pick within it).
inline LanesCardRoom q27_budget_room(uint64_t gmem, uint64_t fixed, uint64_t bank) {
    const uint64_t cap = uint64_t(0.92 * double(gmem));
    return {cap > fixed ? cap - fixed : 0, bank, 0, 0};
}

// P4 B45 (4): the 27B split's prompt-cache prefix bound beside a resident vision tower (Engine::load's [budget] check). The
// cache's worst case on a card is max_entries x (per_tok x max_prefix + dnb) bytes; when the card's total (`fixed` = weights,
// the live KV and DeltaNet state, the banks, the tower, the reserve + that cache) crosses the 92 % line, the bound shrinks
// to the largest prefix (a multiple of 512) that fits, never below min_prefix -- the load then prints
// "[budget] card 1: prompt cache max X -> Y GB ... to fit the vision tower" instead of refusing. fits == false = the refusal
// stands (the fixed part alone is over the line, or the room is below min_prefix). An explicit IE_PROMPT_CACHE_MAX_PREFIX is
// the caller's rule: the load does not call this then.
struct Q27CacheFit { bool fits = false; bool shrunk = false; uint32_t max_prefix = 0; uint64_t cache_bytes = 0; };
inline uint64_t q27_cache_bytes(uint32_t max_entries, uint64_t per_tok, uint64_t dnb, uint32_t max_prefix) {
    return uint64_t(max_entries) * (per_tok * max_prefix + dnb);
}
inline Q27CacheFit q27_cache_fit(uint64_t gmem, uint64_t fixed, uint32_t max_entries, uint64_t per_tok, uint64_t dnb,
                                 uint32_t max_prefix, uint32_t min_prefix = 1024) {
    Q27CacheFit r;
    r.max_prefix = max_prefix;
    r.cache_bytes = q27_cache_bytes(max_entries, per_tok, dnb, max_prefix);
    const uint64_t cap = uint64_t(0.92 * double(gmem));   // (the check compares in double; a total <= this truncated cap passes it)
    if (fixed + r.cache_bytes <= cap) { r.fits = true; return r; }
    if (fixed >= cap || max_entries == 0 || per_tok == 0) return r;   // not the cache's overflow, or nothing to shrink
    const uint64_t per_entry = (cap - fixed) / max_entries;
    if (per_entry <= dnb) return r;
    uint64_t pre = ((per_entry - dnb) / per_tok) / 512 * 512;
    if (pre < min_prefix || pre >= max_prefix) return r;
    r.max_prefix = uint32_t(pre);
    r.cache_bytes = q27_cache_bytes(max_entries, per_tok, dnb, r.max_prefix);
    r.fits = fixed + r.cache_bytes <= cap;
    r.shrunk = r.fits;
    return r;
}

// The load's one line about the pick: the arch, N, the lane context, why, and each measured card's numbers.
inline std::string lanes_auto_line(const char* arch, uint32_t n, uint32_t lane_ctx, const std::string& why,
                                   std::span<const LanesCardRoom> cards = {}) {
    std::string s = std::string("[lanes] auto (no --parallel): ") + arch;
    s += n > 1 ? " picks " + std::to_string(n) + " request lanes (lanes 1.." + std::to_string(n - 1) + " at ctx " +
                     std::to_string(lane_ctx) + ")"
               : std::string(" serves one request at a time");
    if (!why.empty()) s += ": " + why;
    const double G = 1073741824.0;
    for (size_t i = 0; i < cards.size(); ++i) {
        const LanesCardRoom& c = cards[i];
        char b[96], k[48] = "";
        if (c.per_lane) std::snprintf(k, sizeof k, " + %u x %.3f checkpoints", n, double(c.per_lane) / G);
        std::snprintf(b, sizeof b, "; card %zu: %.2f GiB free, %u x %.3f extra lanes", i, double(c.avail) / G, n - 1,
                      double(c.lane_bytes) / G);
        s += b + std::string(k);
        std::snprintf(b, sizeof b, " + %.2f kept = %.2f", double(c.keep) / G, double(lanes_auto_need(c, n)) / G);
        s += b;
    }
    return s + "; --parallel N sets the count";
}

}  // namespace ie
