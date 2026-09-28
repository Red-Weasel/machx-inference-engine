// tests/unit/ds41_serve_rules_test.cpp -- the DeepSeek-V4.1 serving rules at --parallel > 1 that need no device
// (include/ie/ds41_serve_rules.hpp, P4 B6b, docs/deepseek41/P4_B6B_SERVE.md): the lane choice -- MiMo B4's reply-room rule
// (ba287bd, its 16 checks carried over) and V4.1's continuation preference -- and the reply budget. CPU only, no SYCL:
//   g++ -std=c++20 -O1 -Wall -Wextra -I include tests/unit/ds41_serve_rules_test.cpp
#include "ie/ds41_serve_rules.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("[%s] %s\n", ok ? " ok " : "FAIL", what);
    if (!ok) ++g_fail;
}

using V = ie::Ds41LaneView;
// lanes as the MiMo B4 gate ran them: --ctx 32768 --parallel 4 --slot-ctx 16384 = lane 0 at 32,768 positions, lanes 1-3 at
// 16,384; min_slot_tokens 1,024
std::vector<V> lanes4() { std::vector<V> v(4); v[0].cap = 32768; for (int i = 1; i < 4; ++i) v[i].cap = 16384; for (auto& l : v) l.idle = true; return v; }
int pick(const std::vector<V>& v, uint32_t prompt, uint32_t budget) { return ie::ds41_choose_lane(v, prompt, budget, 1024); }

void test_reply_room() {   // MiMo B4's checks (tests/unit/mimo26_host_rules_test.cpp, ba287bd), on V4.1's rule
    auto v = lanes4();
    check(pick(v, 15993, 2000) == 0, "room: the B4 gate's case -- a 15,993-token prompt with max_tokens 2,000 takes idle lane 0, not a 16,384 lane (391 left)");
    v[0].occupied = true; v[0].match = 5;
    check(pick(v, 15993, 2000) == 0, "room: ... also when lane 0 holds another conversation (the reply room outranks an empty lane)");
    v = lanes4(); v[0].idle = false;
    check(pick(v, 15993, 2000) == 1, "room: no idle lane leaves the reply room (lane 0 busy): the smallest lane (the reply is cut there, no queueing)");
    v = lanes4();
    check(pick(v, 1620, 16384) == 1, "room: a 1,620-token prompt with the server's default budget (16,384) takes a small lane (room >= a quarter of it)");
    check(pick(v, 1620, 31000) == 1, "room: ... and with a window-sized budget (Dream: window - prompt)");
    check(pick(v, 1620, 0) == 1, "room: ... and unlimited (0)");
    check(pick(v, 1620, UINT32_MAX) == 1, "room: ... and the server's unlimited (UINT32_MAX, max_tokens 0 on the wire)");
    check(pick(v, 13000, 16384) == 0, "room: a 13,000-token prompt leaves 3,384 < 4,096 on a small lane: lane 0");
    check(pick(v, 12288, 16384) == 1, "room: a 12,288-token prompt leaves exactly a quarter (4,096) of a small lane: the small lane");
    check(pick(v, 16200, 100) == 1, "room: a budget smaller than the room left (100 <= 184): the small lane");
    v = lanes4(); v[2].occupied = true; v[2].match = 5000;
    check(pick(v, 5100, 2000) == 2, "room: a big prefix match on a lane that leaves the room beats the empty lanes");
    v = lanes4(); v[1].occupied = true; v[1].match = 15992;
    check(pick(v, 15993, 2000) == 0, "room: the gate's case repeated -- its own lane 1 (match 15,992) leaves 391: lane 0 re-prefills rather than cut the reply");
    v[0].idle = false;
    check(pick(v, 15993, 2000) == 1, "room: ... with lane 0 busy, the prefix match decides among the lanes left (lane 1, the reply cut at 391)");
    v = lanes4(); v[0].idle = false;
    check(pick(v, 17000, 2000) == -1, "room: a prompt only lane 0 fits, lane 0 busy: none (the request waits for it)");
    v = lanes4(); v[1].occupied = true; v[1].match = 200;
    check(pick(v, 2000, 2000) == 2, "room: no big match -- an empty lane before one holding a conversation");
    v = lanes4(); v[1].tick = 5; v[2].tick = 3; v[3].tick = 9;
    check(pick(v, 2000, 2000) == 2, "room: equal lanes -- the least recently released (tick 3)");
    v = lanes4(); for (auto& l : v) l.idle = false;
    check(pick(v, 10, 10) == -1, "room: every lane busy: none");
}

void test_continuation() {   // V4.1: a match that reaches the lane's last prompt end continues its conversation
    auto v = lanes4();
    v[3].occupied = true; v[3].match = 611; v[3].last_tp = 611;          // a short image chat (611 positions) held by lane 3
    check(pick(v, 700, 400) == 3, "cont: a follow-up of a short conversation (match 611 = its last prompt end) stays on its lane, not an empty one");
    v[3].match = 300;                                                    // a shorter shared prefix: not a continuation, under 1,024
    check(pick(v, 700, 400) == 1, "cont: a shared prefix that stops short of the lane's last prompt end, under 1,024: an empty lane first");
    v = lanes4(); v[2].occupied = true; v[2].match = 2599; v[2].last_tp = 2599;   // the same 2,600-token prompt again (cached = T - 1)
    v[1].occupied = true; v[1].match = 2048; v[1].last_tp = 2599;                 // another lane serving only its first chunk
    check(pick(v, 2600, 96) == 2, "cont: the same prompt again -- the lane serving the most (2,599) wins");
    v = lanes4(); v[1].occupied = true; v[1].match = 0; v[1].last_tp = 500;
    check(pick(v, 400, 96) == 2, "cont: last_tp set but nothing served (match 0): not a continuation, an empty lane first");
    v = lanes4(); v[1].occupied = true; v[1].match = 611; v[1].last_tp = 611; v[1].cap = 700;   // the continuation lane lacks the room
    check(pick(v, 690, 400) == 2, "cont: the reply room still outranks a continuation (lane 1 would leave 10 of 400)");
    v = lanes4();
    check(ie::ds41_choose_lane(v, 700, 400, UINT32_MAX) == 1, "cont: with the prefix cache off (min_tokens UINT32_MAX, no matches): an empty lane by capacity, then index");
}

void test_reply_budget() {   // Ds41Generator::run's rule, in 64-bit
    check(ie::ds41_reply_budget(2600, 96, 32768) == 96, "budget: max_tokens that fits: itself");
    check(ie::ds41_reply_budget(2600, 0, 32768) == 30168, "budget: 0 = unlimited: the positions left");
    check(ie::ds41_reply_budget(2600, UINT32_MAX, 32768) == 30168, "budget: the server's unlimited (UINT32_MAX) cannot wrap: the positions left");
    check(ie::ds41_reply_budget(2600, UINT32_MAX - 1000, 32768) == 30168, "budget: a near-UINT32_MAX budget: the positions left");
    check(ie::ds41_reply_budget(32000, 2000, 32768) == 768, "budget: too little room: the positions left");
    check(ie::ds41_reply_budget(32768, 10, 32768) == 0, "budget: a prompt filling the lane: 0");
}

void test_static_tier() {   // P4 B14: the auto static tier and the lanes' admission (Ds41Forward::init_resident)
    const uint64_t per = 180ull << 20;   // one slot in every layer of the tier (synthetic; not the model's bytes)
    check(ie::ds41_static_slots(10 * per + per / 2, per, 256) == 10, "static: whole slots only");
    check(ie::ds41_static_slots(1000 * per, per, 128) == 128, "static: at most the experts this card holds");
    check(ie::ds41_static_slots(5 * per, 0, 128) == 0, "static: a zero slot size gives no slots");
    check(ie::ds41_static_refusal(9, 8, 9 * per, 15, 245ull << 20, 32768).empty(), "static: one slot above the stream partition fits");
    const std::string m = ie::ds41_static_refusal(8, 8, 8 * per, 15, 245ull << 20, 32768);
    check(m == "init_resident: VRAM left for experts (1440 MiB) gives 8 slots/layer, not enough for the stream partition "
               "(the 15 extra lane(s) at ctx 32768 take 3675 MiB on this card: fewer lanes or a smaller lane context)",
          ("static: --parallel 16 refused with the lanes' numbers: " + m).c_str());
    check(ie::ds41_static_refusal(3, 8, 3 * per, 0, 0, 0) ==
              "init_resident: VRAM left for experts (540 MiB) gives 3 slots/layer, not enough for the stream partition",
          "static: --parallel 1's message is unchanged (no lanes clause)");
}

}  // namespace

void test_step_qstar() {   // P4 B19: the lanes' PCIe share (IE_DS41_QSTAR_LANES)
    using ie::ds41_step_qstar;
    check(ds41_step_qstar(0.30f, -1.f, 1, 16) == 0.30f, "qstar: unset (< 0) keeps the tier's q* at 16 lanes in flight");
    check(ds41_step_qstar(0.30f, -1.f, 1, 1) == 0.30f, "qstar: unset keeps the tier's q* for a lone lane");
    check(ds41_step_qstar(0.30f, 0.60f, 1, 2) == 0.60f, "qstar: set, a one-row step with 2 lanes in flight takes the lanes' share");
    check(ds41_step_qstar(0.30f, 0.60f, 1, 16) == 0.60f, "qstar: ... and with 16");
    check(ds41_step_qstar(0.30f, 0.60f, 1, 1) == 0.30f, "qstar: set, a lone lane in flight keeps the tier's q* (solo is tuned)");
    check(ds41_step_qstar(0.30f, 0.60f, 1, 0) == 0.30f, "qstar: set, no lane count (the serial path) keeps the tier's q*");
    check(ds41_step_qstar(0.30f, 0.60f, 2, 4) == 0.30f, "qstar: set, a 2-row lookup verify keeps its own share (the tier picks qstar_multi)");
    check(ds41_step_qstar(0.30f, 0.60f, 8, 4) == 0.30f, "qstar: ... an 8-row verify too");
    check(ds41_step_qstar(0.30f, 0.60f, 2048, 4) == 0.30f, "qstar: ... and a prefill chunk (the tier's continuation split)");
    check(ds41_step_qstar(0.30f, 0.f, 1, 3) == 0.f, "qstar: 0 is a value (every miss on the CPU), not unset");
    check(ds41_step_qstar(0.30f, 1.5f, 1, 3) == 1.f, "qstar: above 1 is clamped to 1 (every miss over PCIe)");
    check(ds41_step_qstar(0.60f, 0.60f, 1, 3) == 0.60f, "qstar: equal to the tier's: nothing changes");
}

int main() {
    test_reply_room();
    test_continuation();
    test_reply_budget();
    test_static_tier();
    test_step_qstar();
    std::printf("\nDS41 SERVE RULES: %s\n", g_fail ? "FAILURE(S)" : "PASS");
    return g_fail ? 1 : 0;
}
