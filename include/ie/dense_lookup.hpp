// include/ie/dense_lookup.hpp -- the prompt-lookup draft policy of the dense fleet class (DenseModelTP::look_step), host only.
//
// 2026-10-28 (serving campaign P2, docs/research/2026-10-28-nanbeige43-serving-campaign.md): the 35B split's look_step
// rule (qwen35moe_split.hpp, P4 B53) on a KV-only model, with the bookkeeping here so it is tested without a GPU
// (tests/unit/dense_lookup_policy_test.cpp). The engine's decode loop forwards one token a step. When the context's last
// tokens repeat an earlier span (the prompt, then every token given), the tokens that followed that span are run with
// the token in ONE forward -- a "round", every row's logits kept -- and the next steps are answered from those rows
// while the caller's token is the drafted one. A row not followed costs nothing on a KV-only model: the KV lengths go
// back to that position (the kernels read start_pos, so the stale rows past it are never read) -- no re-run, unlike
// the DeltaNet models. Probe first: a round is `probe` rows (`strong_rows` on a repeated span of `strong` tokens) and
// doubles, up to `rows`, after each round followed to its end; a round with fewer than 3 rows followed blocks lookups
// for `block` tokens, twice as long after each such round in a row (at most `block_max`).
#pragma once

#include "ie/ngram_draft.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace ie {

// (at namespace scope: a nested struct's default member values are not usable in the outer class's default arguments)
struct DenseLookupParams {
    uint32_t rows = 128;        // the longest round (the rows of one forward), 8..256
    uint32_t min_match = 16;    // the repeated span that starts a draft
    uint32_t probe = 8, block = 24, block_max = 384, strong = 24, strong_rows = 32;
};

class DenseLookupPolicy {
public:
    using Params = DenseLookupParams;
    struct Stats { uint64_t rounds = 0, drafted = 0, hits = 0, misses = 0, plain = 0; };
    enum class Action : uint8_t { kServeRow, kPlain, kRound };
    struct Decision {
        Action               action = Action::kPlain;
        uint32_t             row = 0;            // kServeRow: the open round's row whose logits answer this step
        std::vector<int32_t> vin;                // kRound: the forward's rows -- the token, then the drafts
        bool                 rollback = false;   // first: the KV lengths go back to `rollback_to` (rows not followed)
        uint32_t             rollback_to = 0;
    };

    explicit DenseLookupPolicy(const Params& p = Params()) : p_(p) {
        p_.rows = std::min<uint32_t>(256, std::max<uint32_t>(8, p_.rows));
        reset();
    }
    const Params& params() const noexcept { return p_; }
    const Stats&  stats()  const noexcept { return st_; }
    bool          round_open() const noexcept { return next_ < n_rows_; }

    // The history the lookup reads: the prompt's ids (the request's prefill); every token given after is pushed by step.
    void context(const std::vector<int32_t>& ids) { idx_.reset(ids); idx_on_ = true; }
    // A request starts: the counters, the streak and the block go back (the index is replaced by context()).
    void reset() { st_ = {}; n_rows_ = next_ = 0; streak_ = block_ = 0; block_len_ = p_.block; grow_ = p_.probe; }

    // The caller forwards `token` at `pos` (the logits of that position are wanted).
    Decision step(int32_t token, uint32_t pos, uint32_t max_ctx) {
        Decision d;
        if (idx_on_) idx_.push(token);
        if (block_) --block_;
        // A row of the open round for exactly this position and input token: its logits are there already.
        if (next_ < n_rows_ && pos == base_ + next_ && token == vin_[next_]) {
            d.action = Action::kServeRow; d.row = next_; ++next_; ++st_.hits;
            if (next_ == n_rows_) {   // followed to its end: the next round may be longer
                ++streak_; block_len_ = p_.block;
                grow_ = std::min<uint32_t>(std::max<uint32_t>(n_rows_, p_.probe) * 2u, p_.rows);
                n_rows_ = next_ = 0;
            }
            return d;
        }
        if (next_ < n_rows_) { d.rollback = true; d.rollback_to = base_ + next_; close_unfollowed(); }
        std::vector<int32_t> look;
        if (idx_on_ && !block_ && pos != 0 && pos + 2 <= max_ctx) {
            const uint32_t room = max_ctx - pos - 1;   // the rows after this one that still fit the context
            if (!streak_) look = idx_.draft(std::min(std::min(p_.rows, p_.strong_rows) - 1, room), p_.strong);
            if (look.size() < 2) look = idx_.draft(std::min((streak_ ? grow_ : p_.probe) - 1, room), p_.min_match);
            if (look.size() < 2) look.clear();
        }
        if (look.empty()) { d.action = Action::kPlain; ++st_.plain; return d; }
        d.action = Action::kRound;
        d.vin.reserve(1 + look.size());
        d.vin.push_back(token);
        d.vin.insert(d.vin.end(), look.begin(), look.end());
        return d;
    }
    // After a kRound's forward ran: the round is open at `pos` with `vin`'s rows, row 0 answered this step.
    void round_started(uint32_t pos, const std::vector<int32_t>& vin) {
        base_ = pos; vin_ = vin; n_rows_ = uint32_t(vin.size()); next_ = 1;
        ++st_.rounds; st_.drafted += n_rows_ - 1;
    }
    // The request ends: the rows not followed -> the KV lengths go back to `rollback_to` (true), else nothing (false).
    bool flush(uint32_t& rollback_to) {
        if (next_ < n_rows_) { rollback_to = base_ + next_; close_unfollowed(); return true; }
        n_rows_ = next_ = 0;
        return false;
    }

private:
    void close_unfollowed() {
        const uint32_t followed = next_;   // row 0 (the token itself) and the drafts the sampler took
        ++st_.misses; streak_ = 0;
        if (followed < 3) { block_ = block_len_; block_len_ = std::min<uint32_t>(block_len_ * 2u, p_.block_max); }
        n_rows_ = next_ = 0;
    }

    Params         p_;
    Stats          st_;
    Ds41NgramIndex idx_;
    bool           idx_on_ = false;
    std::vector<int32_t> vin_;
    uint32_t base_ = 0, n_rows_ = 0, next_ = 0;
    uint32_t streak_ = 0, block_ = 0, block_len_ = 0, grow_ = 0;
};

}  // namespace ie
