// tests/unit/q4e_lane_ladder_test.cpp -- host-only gate of include/ie/q4e_lane_ladder.hpp (2026-11-10): the record and the
// slots of a request lane's restart points, driven the way the engine drives them (chunks of 2048, step 8192).
#include "ie/q4e_lane_ladder.hpp"

#include <cstdio>
#include <numeric>
#include <vector>

using ie::Q4eLaneLadder;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) { std::printf("  [%s] %s\n", ok ? "OK" : "FAIL", what); if (!ok) ++g_fail; }
// reads [from, to) in 2048-row chunks; every regular copy "succeeds"; returns the copies taken
int read(Q4eLaneLadder& l, const std::vector<int32_t>& ids, uint32_t from, uint32_t to) {
    int copies = 0;
    for (uint32_t p = from; p < to; p += 2048) {
        const uint32_t n = std::min<uint32_t>(2048, to - p);
        const int s = l.regular_slot(p, n);
        if (!l.append(ids.data(), p, n)) return -1;
        if (s >= 0) { l.set(s, p + n); ++copies; }
    }
    return copies;
}
}  // namespace

int main() {
    std::printf("q4e_lane_ladder_test\n");
    std::vector<int32_t> a(40000);
    std::iota(a.begin(), a.end(), 1000);
    {
        std::printf("1. a cold read, then prompts that leave it\n");
        Q4eLaneLadder l; l.init(8192, 32, 4);
        check(read(l, a, 0, 33584) == 4, "33,584 tokens in 2048-row chunks: copies at 8192, 16384, 24576, 32768");
        int s = -1;
        std::vector<int32_t> b(a.begin(), a.begin() + 33584);
        check(l.best(b, &s) == 32768, "the same prompt: the deepest copy below its last row");
        b[28655] = -7;
        check(l.best(b, &s) == 24576 && s == 2, "a change at 28,655: the copy at 24,576 (slot 2)");
        b = std::vector<int32_t>(a.begin(), a.begin() + 33584); b[5] = -7;
        check(l.best(b, &s) == 0, "a change at token 5: nothing to restart from");
        b = std::vector<int32_t>(a.begin(), a.begin() + 16384);
        check(l.best(b, &s) == 8192, "a prompt that ends ON a copy's depth takes the one below (a row must be left to read)");
        b = std::vector<int32_t>(a.begin(), a.begin() + 16385);
        check(l.best(b, &s) == 16384, "one token longer: that copy");
    }
    {
        std::printf("2. a restore drops the record and the copies above it\n");
        Q4eLaneLadder l; l.init(8192, 32, 4);
        read(l, a, 0, 33584);
        l.keep(24576);
        check(l.rec.size() == 24576 && l.depth[3] == 0 && l.depth[2] == 24576, "keep(24576): the record ends there, the 32,768 copy is gone, the 24,576 one stays");
        std::vector<int32_t> c(a.begin(), a.begin() + 24576);
        for (int i = 0; i < 9000; ++i) c.push_back(-100 - i);
        check(read(l, c, 24576, uint32_t(c.size())) == 1 && l.depth[3] == 32768, "the new tail is read from there: a new copy at 32,768");
        std::vector<int32_t> old(a.begin(), a.begin() + 33584);
        int s = -1;
        check(l.best(old, &s) == 24576, "the OLD prompt now shares only 24,576: the new 32,768 copy is not offered to it");
        l.keep(0);
        check(l.rec.empty() && l.best(old) == 0, "keep(0) leaves nothing");
    }
    {
        std::printf("3. turn boundaries\n");
        Q4eLaneLadder l; l.init(8192, 32, 4);
        read(l, a, 0, 10000);
        const int t1 = l.turn_slot(10000);
        check(t1 == 32, "the first boundary takes the first turn slot");
        l.set(t1, 10000);
        check(l.turn_slot(10000) == -1, "a boundary a slot already holds is not copied again");
        check(l.turn_slot(12000) == -1, "a boundary beyond the record is refused");
        read(l, a, 10000, 10500); l.set(l.turn_slot(10500), 10500);
        read(l, a, 10500, 11000); l.set(l.turn_slot(11000), 11000);
        read(l, a, 11000, 11500); l.set(l.turn_slot(11500), 11500);
        read(l, a, 11500, 12000); const int t5 = l.turn_slot(12000); l.set(t5, 12000);
        check(t5 == 32 && l.depth[32] == 12000, "the fifth boundary reuses the oldest turn slot");
        std::vector<int32_t> b(a.begin(), a.begin() + 12000); b[11200] = -1; b.push_back(5);
        check(l.best(b) == 11000, "a rewrite inside the last-but-one turn restarts at the boundary before it");
    }
    {
        std::printf("4. gaps and failed copies\n");
        Q4eLaneLadder l; l.init(8192, 32, 4);
        read(l, a, 0, 4096);
        check(!l.append(a.data(), 6144, 2048) && l.rec.size() == 4096, "a chunk that does not continue the record is refused");
        check(!l.whole_to(6144) && l.whole_to(4096), "whole_to tells the engine not to copy past a gap");
        l.set(0, 8192);
        check(l.depth[0] == 0, "a copy deeper than the record is not kept");
        Q4eLaneLadder off;
        check(off.best(a) == 0 && off.regular_slot(0, 16384) == -1 && off.turn_slot(5) == -1, "an uninitialised ladder does nothing");
        Q4eLaneLadder pp; pp.init(8192, 32, 4);
        check(pp.append_piece(a.data(), 0, 2048) && pp.append_piece(a.data() + 2048, 2048, 2048) && pp.rec.size() == 4096 && pp.rec[2048] == a[2048],
              "append_piece takes a piece's own tokens (the lane pipe's steps)");
        check(!pp.append_piece(a.data() + 8192, 8192, 2048) && pp.rec.size() == 4096, "a piece that leaves a gap is refused");
        Q4eLaneLadder r; r.init(8192, 2, 0);
        read(r, a, 0, 30000);
        check(r.depth[0] == 24576 && r.depth[1] == 16384, "a ring of 2 keeps the two deepest multiples");
    }
    {
        std::printf("5. a restart point as a reason to choose the lane\n");
        Q4eLaneLadder l; l.init(8192, 32, 4);
        read(l, a, 0, 33584);
        std::vector<int32_t> same(a.begin(), a.begin() + 33584); same[20000] = -3; same.push_back(9);
        check(l.own(same) == 16384, "the lane's own conversation with its history changed: the lane is its own (restart at 16,384)");
        std::vector<int32_t> other(a.begin(), a.begin() + 9000);
        for (int i = 0; i < 2000; ++i) other.push_back(-500 - i);
        check(l.best(other) == 8192 && l.own(other) == 0, "an 11K-token prompt sharing a 9K prefix with a 33.6K conversation: not steered here");
        std::vector<int32_t> half(a.begin(), a.begin() + 17000);
        for (int i = 0; i < 500; ++i) half.push_back(-900 - i);
        check(l.own(half) == 16384, "a branch that shares at least half of the record: its own");
        std::vector<int32_t> lng(a.begin(), a.begin() + 9000);
        for (int i = 0; i < 30000; ++i) lng.push_back(-2000 - i);
        check(l.own(lng) == 8192, "a prompt as long as the record that leaves it early: its own (the conversation's start changed)");
    }
    std::printf("q4e_lane_ladder_test: %s (%d failures)\n", g_fail ? "FAIL" : "OK", g_fail);
    return g_fail ? 1 : 0;
}
