// src/model/deepseek41_residency.cpp — see the header.
#include "ie/deepseek41_residency.hpp"

#include <algorithm>

namespace ie {

std::string ds41_plan_residency(uint32_t layers, uint32_t n_experts, uint64_t slot_bytes,
                                uint32_t n_cards, uint64_t vram_budget_per_card,
                                uint64_t host_pin_cap_total, uint32_t min_stream_slots,
                                Ds41ResidencyPlan& out) {
    out = Ds41ResidencyPlan{};
    if (!layers || !n_experts || !slot_bytes || !n_cards) return "ds41_plan_residency: zero-sized request";
    if (n_cards > layers) return "ds41_plan_residency: more cards than layers";
    out.n_cards = n_cards; out.n_experts = n_experts; out.slot_bytes = slot_bytes;

    // contiguous layer ranges, as even as integer division allows
    out.first_layer.resize(n_cards + 1);
    for (uint32_t c = 0; c <= n_cards; ++c) out.first_layer[c] = layers * c / n_cards;

    // One V4 plan per card over its own layers, with the host cap split in proportion to the
    // card's layer count. V4's planner refuses when experts have no home; we ask it for the
    // coverage it CAN give by capping slots at what the budgets allow and taking the remainder
    // as the mmap tier. The math is per layer and identical across cards with equal layer
    // counts, so the per-layer figures below are the first card's (the tightest, if uneven).
    uint32_t vram_static = n_experts, pinned = 0, stream = 0;
    out.card.resize(n_cards);
    for (uint32_t c = 0; c < n_cards; ++c) {
        const uint32_t nl = out.first_layer[c + 1] - out.first_layer[c];
        const uint64_t layer_slot_total = uint64_t(nl) * slot_bytes;      // one slot in every layer of this card
        const uint64_t host_cap = host_pin_cap_total * nl / layers;
        // slots this card can hold per layer, static + stream
        const uint32_t vram_slots = uint32_t(std::min<uint64_t>(n_experts, vram_budget_per_card / layer_slot_total));
        const uint32_t pin_slots  = uint32_t(std::min<uint64_t>(n_experts, host_cap / layer_slot_total));
        Ds4ResidencyPlan p{};
        p.slots_per_layer  = vram_slots;
        p.stream_slots     = std::min(min_stream_slots, vram_slots);
        p.static_slots     = vram_slots - p.stream_slots;
        p.pinned_experts   = std::min(pin_slots, n_experts - p.static_slots);
        p.layer_slot_total = layer_slot_total;
        p.vram_bytes       = uint64_t(p.slots_per_layer) * layer_slot_total;
        p.host_bytes       = uint64_t(p.pinned_experts) * layer_slot_total;
        out.card[c] = p;
        vram_static = std::min(vram_static, p.static_slots);
        pinned = c == 0 ? p.pinned_experts : std::min(pinned, p.pinned_experts);
        stream = c == 0 ? p.stream_slots : std::min(stream, p.stream_slots);
        out.vram_bytes_total   += p.vram_bytes;
        out.host_pinned_bytes  += p.host_bytes;
    }
    out.vram_static_per_layer = vram_static;
    out.vram_stream_per_layer = stream;
    out.pinned_per_layer      = pinned;
    out.mmap_per_layer        = n_experts - std::min(n_experts, vram_static + pinned);
    out.mmap_bytes            = uint64_t(out.mmap_per_layer) * slot_bytes * layers;
    return {};
}

Ds41TierHits ds41_tier_hits(const Ds41ResidencyPlan& plan,
                            const std::vector<std::vector<uint32_t>>& ranking,
                            const std::vector<std::vector<uint64_t>>& counts) {
    Ds41TierHits h{};
    uint64_t by_tier[3] = {0, 0, 0};
    for (size_t L = 0; L < ranking.size() && L < counts.size(); ++L) {
        for (uint32_t rank = 0; rank < ranking[L].size(); ++rank) {
            const uint32_t e = ranking[L][rank];
            if (e >= counts[L].size()) continue;
            by_tier[plan.tier_of_rank(rank)] += counts[L][e];
            h.total += counts[L][e];
        }
    }
    if (h.total) { h.vram = double(by_tier[0]) / double(h.total); h.pinned = double(by_tier[1]) / double(h.total); h.mmap = double(by_tier[2]) / double(h.total); }
    return h;
}

}  // namespace ie
