// include/ie/dn_ladder_plan.hpp -- P4 B57: the host bookkeeping of the in-place restart points (ie/dn_ladder.hpp has the
// device copies and the description). No SYCL here: host-tested in tests/unit/dn_ladder_test.cpp.
#pragma once

#include <algorithm>
#include <cstdint>
#include <span>
#include <vector>

namespace ie {

// The depths' bookkeeping (host-tested: tests/unit/dn_ladder_test.cpp).
struct DnLadderPlan {
    uint32_t step = 0;                 // regular depths are multiples of `step` (0 = no regular depths)
    uint32_t n_regular = 0;            // regular slots (a ring over the multiples of step: slot = (depth / step - 1) % n_regular)
    std::vector<uint32_t> depth;       // [n_regular + 2] the depth each slot holds (0 = empty); the last two: prompt end, reply end
    void init(uint32_t step_, uint32_t n_regular_) { step = step_; n_regular = n_regular_; depth.assign(n_regular + 2, 0); }
    uint32_t slots() const { return uint32_t(depth.size()); }
    uint32_t slot_prompt_end() const { return n_regular; }
    uint32_t slot_reply_end() const { return n_regular + 1; }
    // the next regular depth strictly above pos and strictly below end (0 = none)
    uint32_t next_regular(uint32_t pos, uint32_t end) const {
        if (!step || !n_regular) return 0;
        const uint64_t d = (uint64_t(pos) / step + 1) * step;
        return d < end ? uint32_t(d) : 0;
    }
    uint32_t regular_slot(uint32_t d) const { return (d / step - 1) % n_regular; }
    // the slot holding the deepest depth <= limit (-1 = none)
    int best(uint32_t limit) const {
        int b = -1;
        for (uint32_t i = 0; i < depth.size(); ++i)
            if (depth[i] && depth[i] <= limit && (b < 0 || depth[i] > depth[uint32_t(b)])) b = int(i);
        return b;
    }
    void drop_above(uint32_t d) { for (auto& x : depth) if (x > d) x = 0; }
    void clear() { std::fill(depth.begin(), depth.end(), 0u); }
};

// The step for a context: `want` (>= 512, a multiple of 512), raised so n_regular slots cover the context.
inline uint32_t dn_ladder_step(uint32_t max_ctx, uint32_t n_regular, uint32_t want) {
    uint64_t s = std::max<uint32_t>(512u, want / 512u * 512u);
    if (n_regular) while (s * n_regular < max_ctx) s *= 2;
    return uint32_t(std::min<uint64_t>(s, 1u << 30));
}

// The common leading tokens of the live sequence and a prompt.
inline uint32_t dn_ladder_lcp(std::span<const int32_t> live, std::span<const int32_t> ids) {
    const size_t n = std::min(live.size(), ids.size());
    size_t i = 0;
    while (i < n && live[i] == ids[i]) ++i;
    return uint32_t(i);
}

}  // namespace ie
