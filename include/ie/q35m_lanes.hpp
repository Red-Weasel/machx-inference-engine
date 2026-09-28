// include/ie/q35m_lanes.hpp -- the Qwen3.6/3.8-35B-A3B crown split (qwen35moe, Qwen35MoeSplitModel) request lanes: the
// host-side rules (P4 B10, docs/lanes/LANES_SERVE.md).
//
// The crown runs as ONE model object over two cards (layers [0, 20) on card 0, [20, 40) + the head on card 1). A request lane
// is one sequence's KV + DeltaNet state on both cards (Qwen35MoeSplitModel::select_lane); the shared lanes module
// (ie/lanes_serve.hpp) serves them. No SYCL here: the per-lane VRAM arithmetic, the prefill chunk rule and the prefill plan
// are unit-tested on the CPU (tests/unit/q35m_lanes_test.cpp).
#pragma once

#include "ie/lanes_serve.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace ie {

// One card's per-lane state shape (what Qwen35MoeSplitModel::load allocates per card: KvCache + DeltaNetState).
struct Q35mLaneShape {
    uint32_t n_full = 0, n_kv_heads = 0, head_dim = 0;                       // KV: fp16 K and V [n_full, n_kv_heads, ctx, head_dim]
    uint32_t n_lin = 0, v_heads = 0, k_head_dim = 0, v_head_dim = 0;         // DeltaNet state f32 [n_lin, v_heads, v_hd, k_hd]
    uint32_t conv_channels = 0, conv_kernel = 0;                             // conv state fp16 [n_lin, channels, kernel - 1]
};

// Device bytes of ONE extra lane on a card at `ctx` positions (KvCache::init + DeltaNetState::init; no snapshot: the prompt
// cache is the engine's shared FleetPrefixCache, not per lane).
inline uint64_t q35m_lane_bytes(const Q35mLaneShape& s, uint32_t ctx) {
    const uint64_t kv = 2ull * s.n_full * s.n_kv_heads * uint64_t(ctx) * s.head_dim * 2;
    const uint64_t dn = s.n_lin ? uint64_t(s.n_lin) * (uint64_t(s.v_heads) * s.v_head_dim * s.k_head_dim * 4 +
                                                     uint64_t(s.conv_channels) * (s.conv_kernel - 1) * 2) : 0;
    return kv + dn;
}

// The load-time admission of --parallel n_lanes on one card (Qwen35MoeSplitModel::init_lanes; P4 B14 moved it here so it is
// host-tested): the n_lanes - 1 extra lanes of lane_bytes each plus the reserve must fit what the card reports free (after the
// weights, lane 0's state and the largest step's workspace). "" = fits; else the refusal, with the numbers.
inline std::string q35m_lanes_fit(uint64_t free_bytes, uint64_t lane_bytes, uint32_t n_lanes, uint32_t lane_ctx, uint64_t reserve,
                                  uint32_t card) {
    const uint64_t need = uint64_t(n_lanes - 1) * lane_bytes + reserve;
    if (free_bytes >= need) return {};
    char b[320];
    std::snprintf(b, sizeof b, "request lanes: card %u has %.2f GiB free; %u extra lane(s) at ctx %u need %.2f GiB "
                  "(%.3f GiB each) + a %.2f GiB reserve. Lower --parallel or --slot-ctx.", card, free_bytes / 1073741824.0,
                  n_lanes - 1, lane_ctx, (need - reserve) / 1073741824.0, lane_bytes / 1073741824.0, reserve / 1073741824.0);
    return b;
}

// The crown split's prefill chunk at --parallel 1 (Engine::generate, the kQwen35Moe branch of its pf_chunk rule): 512, or 8192
// when the context is >= 8192 (or IE_QWEN36_MOE_ONEDNN is set) unless IE_QWEN36_NO_MOE_ONEDNN is; IE_QWEN35_PREFILL_CHUNK
// (1..max_ctx) overrides; never above max_ctx. The env values are passed in (nullptr = unset).
inline uint32_t q35m_prefill_chunk(uint32_t max_ctx, const char* no_onednn, const char* onednn, const char* override_chunk) {
    uint32_t c = max_ctx < 512u ? max_ctx : 512u;
    if (no_onednn == nullptr && (max_ctx >= 8192u || onednn != nullptr)) c = max_ctx < 8192u ? max_ctx : 8192u;
    if (override_chunk) {
        const int v = std::atoi(override_chunk);
        if (v >= 1 && uint32_t(v) <= max_ctx) c = uint32_t(v);
    }
    return c;
}

// [p, end) in pf_chunk-row pieces: --parallel 1's prefill_to loop on the crown split (no pipelined range there).
inline void q35m_chunks(uint32_t p, uint32_t end, uint32_t pf_chunk, std::vector<LanesChunk>& out) {
    const uint32_t piece = pf_chunk ? pf_chunk : 1u;
    for (uint32_t q = p; q < end; q += piece) out.emplace_back(q, end - q < piece ? end - q : piece);
}

// ---- P4 B15: the shared prefix (the system prompt + tools; Engine::chat computes it, the crown and Flash-Next use it) ----

// IE_SHARED_PREFIX: on unless "0".
inline bool shared_prefix_enabled(const char* env) { return !(env && env[0] == '0' && env[1] == 0); }
// IE_SHARED_PREFIX_MIN: the smallest shared prefix worth a snapshot, tokens (default 1024; a value < 1 keeps the default).
inline uint32_t shared_prefix_min(const char* env) {
    if (env) { const int v = std::atoi(env); if (v >= 1) return uint32_t(v); }
    return 1024u;
}
// The shared prefix's length when its tokens (sys) are a STRICT prefix of the prompt's (ids) and at least `min_tokens`; else 0.
inline uint32_t shared_prefix_accept(const std::vector<int32_t>& sys, const std::vector<int32_t>& ids, uint32_t min_tokens) {
    if (sys.empty() || sys.size() >= ids.size() || sys.size() < min_tokens) return 0;
    for (size_t i = 0; i < sys.size(); ++i)
        if (sys[i] != ids[i]) return 0;
    return uint32_t(sys.size());
}
// The boundary a request uses: the shared prefix when the prompt cache is on and it lies below the snapshot boundary, else 0.
inline uint32_t shared_prefix_boundary(uint32_t shared_len, uint32_t snap_at, bool cache_on) {
    return (cache_on && shared_len > 0 && shared_len < snap_at) ? shared_len : 0u;
}

// The prefill plan of a T-token prompt with [0, reused) restored: --parallel 1's prefill_to(share_at), insert,
// prefill_to(snap_at), insert, prefill_to(T). The pipe's part ends at the snapshot boundary when one is due (snap_at > reused);
// snap_at == T when the cache is off. P4 B15: with a shared-prefix boundary inside the pipe's part (reused < share_at < Tp) the
// pieces are cut there too and plan.mark = share_at (LanesModel::mark snapshots it); the pieces after it are the same whether
// [0, share_at) was prefilled or restored.
inline LanesPlan q35m_plan(uint32_t T, uint32_t snap_at, uint32_t reused, uint32_t pf_chunk, uint32_t share_at = 0) {
    LanesPlan p;
    p.Tp = snap_at > reused ? snap_at : T;
    if (share_at > reused && share_at < p.Tp) {
        q35m_chunks(reused, share_at, pf_chunk, p.chunks);
        q35m_chunks(share_at, p.Tp, pf_chunk, p.chunks);
        p.mark = share_at;
    } else {
        q35m_chunks(reused, p.Tp, pf_chunk, p.chunks);
    }
    return p;
}

}  // namespace ie
