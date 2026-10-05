// tests/unit/dn_ladder_test.cpp -- P4 B57: the in-place restart points' host bookkeeping (ie/dn_ladder_plan.hpp).
#include "ie/dn_ladder_plan.hpp"

#include <cstdio>
#include <vector>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

int main() {
    using namespace ie;
    // the step: the wanted spacing, doubled until the regular slots cover the context
    CHECK(dn_ladder_step(65536, 16, 8192) == 8192);
    CHECK(dn_ladder_step(200000, 16, 8192) == 16384);
    CHECK(dn_ladder_step(262144, 16, 8192) == 16384);
    CHECK(dn_ladder_step(1000000, 16, 8192) == 65536);
    CHECK(dn_ladder_step(4096, 16, 100) == 512);
    DnLadderPlan p;
    p.init(8192, 4);
    CHECK(p.slots() == 6 && p.slot_prompt_end() == 4 && p.slot_reply_end() == 5);
    // regular depths strictly between pos and end
    CHECK(p.next_regular(0, 8192) == 0);
    CHECK(p.next_regular(0, 8193) == 8192);
    CHECK(p.next_regular(8192, 30000) == 16384);
    CHECK(p.next_regular(8191, 30000) == 8192);
    CHECK(p.next_regular(24576, 30000) == 0);
    // a ring over the multiples of the step
    CHECK(p.regular_slot(8192) == 0 && p.regular_slot(32768) == 3 && p.regular_slot(40960) == 0);
    p.depth[p.regular_slot(8192)] = 8192;
    p.depth[p.regular_slot(16384)] = 16384;
    p.depth[p.slot_prompt_end()] = 20000;
    p.depth[p.slot_reply_end()] = 20100;
    CHECK(p.best(0) == -1 && p.best(8191) == -1);
    CHECK(p.depth[unsigned(p.best(8192))] == 8192);
    CHECK(p.depth[unsigned(p.best(19999))] == 16384);
    CHECK(p.depth[unsigned(p.best(20000))] == 20000);
    CHECK(p.depth[unsigned(p.best(999999))] == 20100);
    p.drop_above(16384);
    CHECK(p.depth[unsigned(p.best(999999))] == 16384 && p.depth[p.slot_prompt_end()] == 0);
    p.clear();
    CHECK(p.best(999999) == -1);
    // the shared leading tokens
    const std::vector<int32_t> a{1, 2, 3, 4, 5}, b{1, 2, 3, 9, 5, 6}, c{};
    CHECK(dn_ladder_lcp(a, b) == 3 && dn_ladder_lcp(a, a) == 5 && dn_ladder_lcp(a, c) == 0);
    std::printf(fails ? "dn_ladder_test: %d FAILED\n" : "dn_ladder_test: OK\n", fails);
    return fails ? 1 : 0;
}
