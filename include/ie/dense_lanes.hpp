// include/ie/dense_lanes.hpp -- the dense fleet class's request lanes (DenseModelTP, `ie serve --parallel N`): the host-side
// rules, no SYCL, unit-tested on the CPU (tests/unit/dense_lanes_test.cpp). 2026-10-29, serving campaign P3
// (docs/research/2026-10-28-nanbeige43-serving-campaign.md); the 27B's q27_lanes.hpp is the model.
//
// A lane is one sequence's KV on every card (head-sharded: this card's kv heads at the lane's context). Lane 0 is the
// state the load allocated at --ctx; lanes 1..n-1 are allocated at --slot-ctx. The class is ONE pipe stage over all its
// cards (the tensor-parallel forward already runs both cards inside one call), so a lane's prefill pieces are plain
// pf_chunk-row cuts with no pipelined kinds: a 1-row piece is a decode row.
#pragma once

#include "ie/lanes_serve.hpp"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace ie {

// One card's slice of a lane's KV: every layer, this card's kv heads, the head dim.
struct DenseLaneShape { uint32_t n_layers = 0, n_kv_heads_card = 0, head_dim = 0; };

// Device bytes of ONE lane on a card at `ctx` positions: K + V, fp16 (KvCache::init without the int8 shadow).
inline uint64_t dense_lane_bytes(const DenseLaneShape& s, uint32_t ctx) {
    return 2ull * s.n_layers * s.n_kv_heads_card * uint64_t(ctx) * s.head_dim * sizeof(uint16_t);
}

// Lanes 1..n-1's context: --slot-ctx, 0 = 65,536 (the engine's lanes default), never above --ctx.
inline uint32_t dense_lane_ctx(uint32_t slot_ctx, uint32_t max_ctx) {
    const uint32_t c = slot_ctx ? slot_ctx : 65536u;
    return c < max_ctx ? c : max_ctx;
}

// The load-time admission of --parallel n_lanes on one card: the n_lanes - 1 extra lanes of lane_bytes each plus the reserve
// must fit what the card reports free (after the weights, lane 0's KV and the workspace). "" = fits; else the refusal.
inline std::string dense_lanes_fit(uint64_t free_bytes, uint64_t lane_bytes, uint32_t n_lanes, uint32_t lane_ctx, uint64_t reserve,
                                   uint32_t card) {
    if (n_lanes <= 1) return {};
    const uint64_t need = uint64_t(n_lanes - 1) * lane_bytes + reserve;
    if (free_bytes >= need) return {};
    char b[320];
    std::snprintf(b, sizeof b, "request lanes: card %u has %.2f GiB free; %u extra lane(s) at ctx %u need %.2f GiB "
                  "(%.3f GiB each) + a %.2f GiB reserve. Lower --parallel or --slot-ctx.", card, free_bytes / 1073741824.0,
                  n_lanes - 1, lane_ctx, (need - reserve) / 1073741824.0, lane_bytes / 1073741824.0, reserve / 1073741824.0);
    return b;
}

// [p, end) in pf_chunk-row pieces (the last one shorter).
inline void dense_lane_pieces(uint32_t p, uint32_t end, uint32_t pf_chunk, std::vector<LanesChunk>& out) {
    const uint32_t piece = pf_chunk ? pf_chunk : 1u;
    for (uint32_t q = p; q < end; q += piece) out.emplace_back(q, end - q < piece ? end - q : piece);
}

// The lane plan for a T-token prompt with [0, reused) restored: the pipe runs [reused, Tp) in pf_chunk pieces, where Tp is
// the prompt-cache boundary snap_at (the cache on and snap_at past what was restored: prompt_end inserts there and runs the
// tail [snap_at, T) in its turn, the 27B's protocol) else T. No shared-prefix mark yet (dense inserts no shared prefix).
inline LanesPlan dense_lane_plan(uint32_t T, uint32_t reused, uint32_t snap_at, bool cache_on, uint32_t pf_chunk) {
    LanesPlan p;
    p.Tp = (cache_on && snap_at > reused && snap_at <= T) ? snap_at : T;
    dense_lane_pieces(reused, p.Tp, pf_chunk, p.chunks);
    return p;
}

}  // namespace ie
