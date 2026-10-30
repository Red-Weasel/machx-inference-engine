// tests/unit/dense_lookup_policy_test.cpp -- the prompt-lookup draft policy of the dense fleet class
// (include/ie/dense_lookup.hpp), host only: a scripted decode drives it and the decisions are checked.
#include "ie/dense_lookup.hpp"

#include <cstdio>
#include <vector>

using ie::DenseLookupPolicy;
using Action = DenseLookupPolicy::Action;

static int fails = 0;
#define CHECK(cond) do { if (!(cond)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++fails; } } while (0)

static std::vector<int32_t> range(int32_t a, int32_t b) { std::vector<int32_t> v; for (int32_t i = a; i <= b; ++i) v.push_back(i); return v; }
static std::vector<int32_t> cat(std::vector<int32_t> a, const std::vector<int32_t>& b) { a.insert(a.end(), b.begin(), b.end()); return a; }

int main() {
    DenseLookupPolicy::Params p;
    p.rows = 16; p.min_match = 4; p.probe = 4; p.strong = 6; p.strong_rows = 8; p.block = 24; p.block_max = 384;

    {   // 1. no repeat in the history: every step is a plain forward
        DenseLookupPolicy pol(p);
        pol.context(range(10, 17));
        for (int32_t t = 18; t < 30; ++t) {
            auto d = pol.step(t, uint32_t(t - 10), 1000);
            CHECK(d.action == Action::kPlain); CHECK(!d.rollback);
        }
        CHECK(pol.stats().plain == 12); CHECK(pol.stats().rounds == 0);
    }
    {   // 2. a repeated span: a strong round, served row by row; followed to its end the next round grows; a mismatch
        //    rolls back to the first row not followed and nothing is blocked when 3+ rows were followed
        DenseLookupPolicy pol(p);
        const auto A = range(1, 20);
        pol.context(cat(A, A));                          // 40 tokens: the span repeats once
        auto d = pol.step(1, 40, 1000);                  // ... and the caller gives 1 again
        CHECK(d.action == Action::kRound); CHECK(!d.rollback);
        CHECK(d.vin.size() == 8);                        // strong_rows (8) rows: the token + 7 drafts
        CHECK(d.vin[0] == 1); CHECK(d.vin[1] == 2); CHECK(d.vin[7] == 8);
        pol.round_started(40, d.vin);
        CHECK(pol.round_open());
        for (int32_t t = 2; t <= 8; ++t) {               // the sampler follows every draft
            d = pol.step(t, uint32_t(39 + t), 1000);
            CHECK(d.action == Action::kServeRow); CHECK(d.row == uint32_t(t - 1)); CHECK(!d.rollback);
        }
        CHECK(!pol.round_open()); CHECK(pol.stats().hits == 7); CHECK(pol.stats().rounds == 1); CHECK(pol.stats().drafted == 7);
        d = pol.step(9, 48, 1000);                       // a new round, grown: min(max(8, 4) * 2, 16) = 16 rows
        CHECK(d.action == Action::kRound); CHECK(!d.rollback);
        CHECK(d.vin.size() == 16); CHECK(d.vin[0] == 9); CHECK(d.vin[1] == 10); CHECK(d.vin[12] == 1);   // ... 20, 1, 2, ...
        pol.round_started(48, d.vin);
        for (int32_t t = 10; t <= 14; ++t) { d = pol.step(t, uint32_t(39 + t), 1000); CHECK(d.action == Action::kServeRow); }
        d = pol.step(99, 54, 1000);                      // the sampler leaves the drafts at row 6
        CHECK(d.rollback); CHECK(d.rollback_to == 54);   // rows 0..5 (positions 48..53) stay; the KV length goes back to 54
        CHECK(d.action == Action::kPlain);               // 99 never followed anything
        CHECK(!pol.round_open()); CHECK(pol.stats().misses == 1); CHECK(pol.stats().hits == 12);
        d = pol.step(100, 55, 1000);                     // 6 rows were followed: no block, but no repeat either
        CHECK(d.action == Action::kPlain);
    }
    {   // 3. a round with fewer than 3 rows followed blocks the lookups for `block` steps, then they come back
        DenseLookupPolicy pol(p);
        const auto B = range(1, 8);
        pol.context(cat(B, B));                          // 16 tokens
        auto d = pol.step(1, 16, 1000);
        CHECK(d.action == Action::kRound); CHECK(d.vin.size() >= 3);
        pol.round_started(16, d.vin);
        d = pol.step(42, 17, 1000);                      // the second row is not followed: 1 row followed < 3
        CHECK(d.rollback); CHECK(d.rollback_to == 17); CHECK(d.action == Action::kPlain);
        // the next 23 steps re-create the repeat (1..8 again) -- blocked, every step plain (the block counts down at the top
        // of a step, the 35B's rule: the miss's own step and 23 more are blocked, the 24th is free)
        uint32_t pos = 18; int blocked_rounds = 0;
        for (int32_t k = 0; k < 23; ++k) {
            d = pol.step(B[size_t(k % 8)], pos++, 1000);
            if (d.action == Action::kRound) ++blocked_rounds;
        }
        CHECK(blocked_rounds == 0);
        // the block is over: the span 1..8 repeats in the history, so a draft comes
        int rounds_after = 0;
        for (int32_t k = 0; k < 8 && !rounds_after; ++k) {
            d = pol.step(B[size_t(k % 8)], pos++, 1000);
            if (d.action == Action::kRound) { ++rounds_after; pol.round_started(pos - 1, d.vin); }
        }
        CHECK(rounds_after == 1);
    }
    {   // 4. the context's end caps a round: the rows after this one must fit; one row of room = plain
        DenseLookupPolicy pol(p);
        const auto A = range(1, 20);
        pol.context(cat(A, A));
        auto d = pol.step(1, 100 - 4, 100);              // room = 3 rows after this one
        CHECK(d.action == Action::kRound); CHECK(d.vin.size() == 4);
        DenseLookupPolicy pol2(p);
        pol2.context(cat(A, A));
        d = pol2.step(1, 100 - 2, 100);                  // room = 1: a draft needs two rows
        CHECK(d.action == Action::kPlain);
        DenseLookupPolicy pol3(p);
        pol3.context(cat(A, A));
        d = pol3.step(1, 100 - 1, 100);                  // the last position: nothing fits
        CHECK(d.action == Action::kPlain);
    }
    {   // 5. flush: the rows not followed at the request's end roll the KV back; a used-up round does not
        DenseLookupPolicy pol(p);
        const auto A = range(1, 20);
        pol.context(cat(A, A));
        auto d = pol.step(1, 40, 1000);
        CHECK(d.action == Action::kRound);
        pol.round_started(40, d.vin);
        CHECK(pol.step(2, 41, 1000).action == Action::kServeRow);
        CHECK(pol.step(3, 42, 1000).action == Action::kServeRow);
        uint32_t to = 0;
        CHECK(pol.flush(to)); CHECK(to == 43); CHECK(!pol.round_open());
        CHECK(!pol.flush(to));
        CHECK(pol.stats().misses == 1);
    }
    {   // 6. the rows cap: the longest round is `rows` (8..256)
        DenseLookupPolicy::Params q = p; q.rows = 4;
        DenseLookupPolicy pol(q);
        CHECK(pol.params().rows == 8);
        q.rows = 1000;
        DenseLookupPolicy pol2(q);
        CHECK(pol2.params().rows == 256);
    }
    if (fails) { std::fprintf(stderr, "%d failure(s)\n", fails); return 1; }
    std::printf("dense_lookup_policy_test: PASS\n");
    return 0;
}
