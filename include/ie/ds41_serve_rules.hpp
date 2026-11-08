// include/ie/ds41_serve_rules.hpp -- P4 B6b (docs/deepseek41/P4_B6B_SERVE.md): the DeepSeek-V4.1 serving rules at --parallel > 1
// that need no device, so they are host-testable (tests/unit/ds41_serve_rules_test.cpp). Header-only, no SYCL.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace ie {

// One lane as the choice sees it: owned by a request or idle, holding a conversation or empty, its position capacity, how many
// positions of the prompt its OWN state serves (Ds41Forward::prefix_servable), the planned end of the last prompt it holds
// (0: none), its last release (larger = more recent).
struct Ds41LaneView { bool idle = false, occupied = false; uint32_t cap = 0, match = 0, last_tp = 0; uint64_t tick = 0; };

// The lane for a prompt, among the IDLE lanes it fits (prompt < cap): MiMo B4's rule (mimo26_choose_lane, ba287bd) with
// V4.1's continuation preference.
//   1. The lanes that leave the reply room first: cap - prompt >= min(budget, cap / 4) -- the request's whole max_tokens, or a
//      quarter of the lane when the budget is larger (the server's default 16,384 and Dream's window - prompt budgets would
//      otherwise pin every request to the biggest lane). Only when no idle lane leaves that room, the others: the reply is then
//      cut at the lane's room (no queueing).
//   2. Within either set: the lane whose own state serves the most of the prompt, when that is at least `min_tokens` or it
//      reaches the lane's last prompt end (a follow-up of the lane's conversation, or the same prompt again: taking the lane
//      loses nothing, however short the conversation); then an empty lane; then the smallest capacity (short prompts leave the
//      big lane to long ones); then the least recently released; then the lower index.
// budget 0 = unlimited. Returns the lane's index, or -1 when no idle lane fits the prompt.
inline int ds41_choose_lane(const std::vector<Ds41LaneView>& lanes, uint32_t prompt, uint32_t budget, uint32_t min_tokens) {
    const uint32_t want = budget ? budget : UINT32_MAX;
    int best = -1;
    std::tuple<int, int, uint32_t, int, uint32_t, uint64_t> bestk{};   // (short of room, big match, ~match, occupied, cap, tick): smallest wins
    for (size_t i = 0; i < lanes.size(); ++i) {
        const Ds41LaneView& l = lanes[i];
        if (!l.idle || l.cap <= prompt) continue;
        const bool roomy = l.cap - prompt >= std::min(want, l.cap / 4);
        const bool big = l.match >= min_tokens || (l.last_tp && l.match >= l.last_tp);
        const auto key = std::make_tuple(roomy ? 0 : 1, big ? -1 : 0, big ? ~l.match : 0u, l.occupied ? 1 : 0, l.cap, l.tick);
        if (best < 0 || key < bestk) { best = int(i); bestk = key; }
    }
    return best;
}

// A request's reply budget on a lane (Ds41Generator::run's rule, in 64-bit so an unlimited budget cannot wrap): the whole
// max_tokens when the prompt and it fit the lane's positions, else the positions left. max_tokens 0 = unlimited.
inline uint32_t ds41_reply_budget(uint32_t prompt, uint32_t max_tokens, uint32_t cap) {
    if (prompt >= cap) return 0;
    return max_tokens == 0 || uint64_t(prompt) + max_tokens > cap ? cap - prompt : max_tokens;
}

// The auto static tier on one card (Ds41Forward::init_resident; P4 B14 moved the arithmetic here so the lanes' admission is
// host-tested): the slots per layer `for_experts` bytes hold at per_slot bytes a slot (one slot in every layer of the tier), at
// most experts_here. The extra lanes' state was allocated before the budget query, so it is already out of for_experts.
inline uint32_t ds41_static_slots(uint64_t for_experts, uint64_t per_slot, uint32_t experts_here) {
    const uint64_t n = per_slot ? for_experts / per_slot : 0;
    return uint32_t(n < experts_here ? n : experts_here);
}

// The refusal when those slots cannot hold the stream partition plus one static slot (slots <= stream), with the numbers and,
// at --parallel > 1, what the `extra` lanes of lane_bytes at lane_ctx take on this card. "" = the tier fits.
inline std::string ds41_static_refusal(uint32_t slots, uint32_t stream, uint64_t for_experts, uint32_t extra, uint64_t lane_bytes,
                                       uint32_t lane_ctx) {
    if (slots > stream) return {};
    return "init_resident: VRAM left for experts (" + std::to_string(for_experts >> 20) + " MiB) gives " + std::to_string(slots) +
           " slots/layer, not enough for the stream partition" +
           (extra ? " (the " + std::to_string(extra) + " extra lane(s) at ctx " + std::to_string(lane_ctx) + " take " +
                        std::to_string((lane_bytes * extra) >> 20) + " MiB on this card: fewer lanes or a smaller lane context)"
                  : std::string());
}

// P4 B19 (~/ds41_work/p60/b19): the PCIe share of the pinned misses (q*) for one lane-pipe step on a card. The tier's q* (0.30)
// was tuned with ONE card working; with two or more lanes in flight both cards run steps at once, their CPU expert legs share the
// same E-cores while each card's PCIe link carries a fraction of what it can (estimated ~27 % at --parallel 16 from the B14 runs;
// not measured by a counter), so the step may use `lanes_q` instead
// (IE_DS41_QSTAR_LANES). Only a one-row decode step (rows == 1) with busy >= 2 lanes in flight takes it; lanes_q < 0 = unset (the
// default): the tier's own q* everywhere, byte for byte today's split.
inline float ds41_step_qstar(float tier_q, float lanes_q, uint32_t rows, uint32_t busy) {
    return lanes_q >= 0.f && rows == 1 && busy >= 2 ? std::min(lanes_q, 1.f) : tier_q;
}

// 2026-11-04: which residency ranking a load takes. The served-traffic decode profile this deployment writes
// (IE_DS41_PROFILE_OUT, rewritten at the end of every request) is the box's own workload, and a ranking has to match the
// workload it serves (docs/deepseek41/69): over 870K generated tokens of Dream's traffic the September chat profile covered
// 38 % of the expert selections with 64 resident experts a layer and left 9 % of them to the NVMe tier, the served profile
// 57 % and 1.6 % (ranking_compare, 2026-11-04: code-writing replies had fallen to 6-7 tok/s on disk reads). Precedence:
// IE_DS41_RANKING (an explicit choice) > the served profile once it holds at least `min_tokens` generated tokens (a few
// hundred would rank by noise) > the decode-phase chat ranking beside the model > the held-out prefill profile.
struct Ds41RankingChoice { std::string path; const char* why; };
inline Ds41RankingChoice ds41_choose_ranking(const char* env_ranking, const char* env_profile, uint64_t profile_tokens,
                                             uint64_t min_tokens, bool chat_exists, const std::string& dir) {
    if (env_ranking && *env_ranking) return {env_ranking, "IE_DS41_RANKING"};
    if (env_profile && *env_profile && profile_tokens >= min_tokens) return {env_profile, "the served-traffic profile (IE_DS41_PROFILE_OUT)"};
    if (chat_exists) return {dir + "/ie_ranking_decode_chat.txt", "the decode-phase chat ranking beside the model"};
    return {dir + "/ie_ranking_heldout.txt", "the held-out prefill profile"};
}
// The generated-token count a profile's header carries ("...: N generated tokens"); 0 when the text has no such line.
inline uint64_t ds41_profile_generated_tokens(const std::string& head) {
    const auto at = head.find(" generated tokens");
    if (at == std::string::npos) return 0;
    size_t b = at;
    while (b > 0 && head[b - 1] >= '0' && head[b - 1] <= '9') --b;
    if (b == at) return 0;
    return std::stoull(head.substr(b, at - b));
}

}  // namespace ie
