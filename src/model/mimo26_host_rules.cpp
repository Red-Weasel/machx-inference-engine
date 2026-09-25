// src/model/mimo26_host_rules.cpp — MiMo-V2.6 serving rules that need no device. See include/ie/mimo26_host_rules.hpp.
#include "ie/mimo26_host_rules.hpp"

#include <algorithm>
#include <cstring>

namespace ie {

uint32_t mimo26_ring_runs(uint32_t p0, uint32_t p1, uint32_t R, uint32_t runs[4]) {
    if (R == 0 || p1 <= p0) return 0;
    if (p1 - p0 > R) p0 = p1 - R;                       // the ring holds only the last R positions
    const uint32_t a = p0 % R, n = p1 - p0;
    if (a + n <= R) { runs[0] = a; runs[1] = a + n; return 1; }
    runs[0] = a; runs[1] = R; runs[2] = 0; runs[3] = a + n - R;
    return 2;
}

uint32_t mimo26_servable(const std::vector<int32_t>& have, uint32_t written_end, uint32_t ring, uint32_t window,
                         const std::vector<int32_t>& ids) {
    if (ids.empty()) return 0;
    const size_t lim = std::min(have.size(), ids.size() - 1);   // at least one token runs, for the logits
    uint32_t L = 0;
    while (L < lim && have[L] == ids[L]) ++L;
    // the ring's oldest valid position is written_end - ring, not have.size() - ring: a verify's rejected rows were
    // written past the held end and overwrote ring slots of older positions
    if (ring && std::max<size_t>(have.size(), written_end) - L + window > ring) L = 0;
    return L;
}

bool mimo26_continues(uint32_t served, uint32_t prompt_end, uint32_t slack) {
    return uint64_t(served) + slack >= prompt_end;
}

Mimo26Admit mimo26_plan_admit(const std::vector<Mimo26SlotCost>& slots, uint64_t bytes, uint32_t uses, uint64_t budget,
                              uint64_t avail, uint64_t keep_free, int keep, std::vector<size_t>& victims,
                              bool keep_leaves) {
    victims.clear();
    if (bytes > budget) return Mimo26Admit::kTooLarge;
    uint64_t held = 0;
    for (size_t i = 0; i < slots.size(); ++i)
        if (!(keep_leaves && int(i) == keep)) held += slots[i].bytes;
    auto tier = [&](bool continued) {
        std::vector<size_t> v;
        for (size_t i = 0; i < slots.size(); ++i)
            if (int(i) != keep && (slots[i].uses > 0) == continued) v.push_back(i);
        std::stable_sort(v.begin(), v.end(), [&](size_t a, size_t b) {
            if (!continued && slots[a].tokens != slots[b].tokens) return slots[a].tokens < slots[b].tokens;   // one-shot: cheapest first
            return slots[a].tick < slots[b].tick;
        });
        return v;
    };
    std::vector<size_t> order = tier(false);
    if (uses > 0) { const auto c = tier(true); order.insert(order.end(), c.begin(), c.end()); }
    uint64_t freed = 0;
    auto in_budget = [&] { return held - freed + bytes <= budget; };
    auto above_floor = [&] { return avail + freed >= keep_free + bytes; };
    for (size_t k = 0; k < order.size() && !(in_budget() && above_floor()); ++k) { victims.push_back(order[k]); freed += slots[order[k]].bytes; }
    if (in_budget() && above_floor()) return Mimo26Admit::kOk;
    victims.clear();
    return in_budget() ? Mimo26Admit::kFloor : Mimo26Admit::kBudget;
}

bool mimo26_prompt_snapshot(uint32_t P, uint32_t hi, uint32_t ring, uint32_t window, uint32_t margin, uint32_t max_ctx,
                            uint32_t& s0, uint32_t& hi_syn) {
    if (ring == 0 || P == 0 || uint64_t(window) + margin >= ring) return false;
    const uint32_t a = P > window + margin ? P - window - margin : 0u, b = hi > ring ? hi - ring : 0u;
    s0 = std::max(a, b);
    if (uint64_t(s0) + ring > max_ctx) return false;
    hi_syn = s0 + ring;
    return true;
}

size_t mimo26_first_non_f16(const float* x, size_t n) {
    constexpr uint32_t kLimit = 0x477FF000u;   // 65520.0f: the smallest magnitude that rounds to nearest even to fp16 inf
    constexpr size_t kBlock = 4096;
    for (size_t i = 0; i < n; i += kBlock) {
        const size_t e = std::min(n, i + kBlock);
        uint32_t bad = 0;                        // a branch-free pass per block (vectorisable), then the position
        for (size_t j = i; j < e; ++j) { uint32_t u; std::memcpy(&u, x + j, 4); bad |= uint32_t((u & 0x7FFFFFFFu) >= kLimit); }
        if (!bad) continue;
        for (size_t j = i; j < e; ++j) { uint32_t u; std::memcpy(&u, x + j, 4); if ((u & 0x7FFFFFFFu) >= kLimit) return j; }
    }
    return n;
}

Mimo26Scan mimo26_scan_f32(const float* x, size_t n) {
    Mimo26Scan s;
    uint32_t best = 0;                                        // finite magnitudes order as their bit patterns
    for (size_t i = 0; i < n; ++i) {
        uint32_t u; std::memcpy(&u, x + i, 4);
        const uint32_t a = u & 0x7FFFFFFFu;
        if (a >= 0x7F800000u) { if (!s.non_finite++) s.first_bad = int64_t(i); continue; }
        if (s.max_at < 0 || a > best) { best = a; s.max_at = int64_t(i); }
    }
    if (s.max_at >= 0) std::memcpy(&s.max_abs, &best, 4);
    return s;
}

Mimo26Scan mimo26_scan_f16(const uint16_t* x, size_t n) {
    Mimo26Scan s;
    uint16_t best = 0;
    for (size_t i = 0; i < n; ++i) {
        const uint16_t a = x[i] & 0x7FFFu;
        if (a >= 0x7C00u) { if (!s.non_finite++) s.first_bad = int64_t(i); continue; }
        if (s.max_at < 0 || a > best) { best = a; s.max_at = int64_t(i); }
    }
    if (s.max_at >= 0) {                                      // the magnitude's exact fp32 value
        const uint32_t e = best >> 10, m = best & 0x3FFu;
        if (e == 0) s.max_abs = float(m) * 5.9604644775390625e-08f;   // a denormal: m x 2^-24, exact
        else { const uint32_t u = ((e + 112u) << 23) | (m << 13); std::memcpy(&s.max_abs, &u, 4); }
    }
    return s;
}

}  // namespace ie
