// include/ie/deepseek41_residency.hpp — V4.1 expert residency plan: VRAM / pinned RAM / mmap.
//
// V4's ds4_plan_residency (expert_stream.hpp) already does the hard part — static never-evicted
// VRAM slots, a FIFO stream tier backed by a pinned host arena, all sized from real per-layer
// slot bytes — and its per-layer ranking file (`ds4-expert-priority 1`) is model-agnostic. It
// has one rule V4.1 cannot meet: it REFUSES an expert that fits neither VRAM nor pinned RAM,
// because V4-Flash always fits. V4.1's 268.95 GiB of experts against ~46 GiB of VRAM and
// ~190 GiB of pinnable RAM leaves ~12% with no home. This plan adds that third tier: experts
// served from the safetensors mmap on miss (the OS page cache, then the NVMe at the 10.3 GB/s
// measured in docs/deepseek41/06). Nothing is refused; the cost of the third tier is reported.
//
// Cards are used PIPELINE-by-layer, not hidden-dim expert-TP as V4 does: each card owns a
// contiguous range of layers, their dense weights and their expert slots, so no per-expert
// cross-card reduction and single-device contexts throughout (the 83554c2 mirror lesson).
#pragma once

#include "ie/expert_stream.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace ie {

struct Ds41ResidencyPlan {
    uint32_t n_cards = 0;
    std::vector<uint32_t> first_layer;        // [n_cards+1]: card c owns [first_layer[c], first_layer[c+1])
    std::vector<Ds4ResidencyPlan> card;       // V4's plan for each card's own layers
    uint32_t n_experts = 0;
    uint64_t slot_bytes = 0;                  // one expert, one layer (w1+w3+w2 + scales)
    // per layer: how many experts (by priority rank) land in each tier
    uint32_t vram_static_per_layer = 0;       // rank [0, vram_static)          -> VRAM, never evicted
    uint32_t vram_stream_per_layer = 0;       // FIFO slots, backed by the pinned arena
    uint32_t pinned_per_layer = 0;            // rank [vram_static, vram_static+pinned) -> pinned RAM
    uint32_t mmap_per_layer = 0;              // the rest                        -> mmap on miss
    uint64_t vram_bytes_total = 0, host_pinned_bytes = 0, mmap_bytes = 0;
    // an expert's tier, by (layer, priority rank): 0 = VRAM static, 1 = pinned, 2 = mmap
    uint32_t tier_of_rank(uint32_t rank) const {
        return rank < vram_static_per_layer ? 0u : rank < vram_static_per_layer + pinned_per_layer ? 1u : 2u;
    }
};

// Plans V4.1's experts over `n_cards` cards with `layers` layers split contiguously.
//   slot_bytes           one expert at one layer, from the bound tensors (17.93 MiB here)
//   vram_budget_per_card VRAM left for experts after the dense weights and the KV/workspace reserve
//   host_pin_cap_total   pinnable host RAM for the stream arenas of ALL cards (a whole-box resource)
//   min_stream_slots     evictable FIFO slots to keep per layer (V4's knob)
// Never refuses for lack of a home: whatever does not fit is `mmap_per_layer`. Returns "" or a
// diagnostic for a genuinely impossible request (zero budget, non-tiling split).
std::string ds41_plan_residency(uint32_t layers, uint32_t n_experts, uint64_t slot_bytes,
                                uint32_t n_cards, uint64_t vram_budget_per_card,
                                uint64_t host_pin_cap_total, uint32_t min_stream_slots,
                                Ds41ResidencyPlan& out);

// Given the plan and a per-layer ranking (most-selected first, the file format V4 reads with
// ds4_expert_priority_read_layers), the per-token expected traffic for a routing profile: the
// share of selections that hit VRAM / pinned / mmap. `counts[L][e]` are selection counts.
struct Ds41TierHits { double vram = 0, pinned = 0, mmap = 0; uint64_t total = 0; };
Ds41TierHits ds41_tier_hits(const Ds41ResidencyPlan& plan,
                            const std::vector<std::vector<uint32_t>>& ranking,
                            const std::vector<std::vector<uint64_t>>& counts);

}  // namespace ie
