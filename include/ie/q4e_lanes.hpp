// include/ie/q4e_lanes.hpp -- Qwen3.8-Flash-Next (qwen4exp) request lanes: the host-side rules (P4 B8,
// docs/qwen4exp/P4_B8_LANES.md; design ~/ds41_work/p60/p4_b6_b8_design.md sections 1.3, 3.3, 3.4, 5.3).
//
// Flash-Next runs as TWO model objects on two cards (Qwen4ExpModel [0, 24) and [24, 48)). A request lane is one sequence's
// state on each of them (Qwen4ExpModel::select_lane); the shared lanes module (ie/lanes_serve.hpp) serves them. No SYCL here:
// the per-lane VRAM arithmetic, the snapshot match and the prefill plan are unit-tested on the CPU
// (tests/unit/q4e_lanes_test.cpp).
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace ie {

// One stage's per-lane state shape (the numbers Qwen4ExpModel::init_runtime allocates per lane).
struct Q4eLaneShape {
    uint32_t n_full = 0;         // full-attention layers on the stage (KV + QSA indexer caches); >= 1 like init_runtime
    uint32_t n_lin  = 0;         // Gated DeltaNet layers on the stage; >= 1 like init_runtime
    uint32_t n_kv_heads = 0, head_dim = 0;
    uint32_t idx_head_dim = 0;   // 0 = no QSA indexer
    uint32_t v_heads = 0, state = 0;               // DeltaNet recurrent state [v_heads, state, state] f32 per layer
    uint32_t conv_channels = 0, conv_kernel = 0;   // DeltaNet conv state [conv_channels, conv_kernel - 1] f16 per layer
    uint64_t ple_conv_floats = 0;                  // PLE conv state (0 unless the stage owns the PLE block)
};

// Device bytes of ONE extra lane at `ctx` positions: KV (f16 K and V), the indexer's raw keys (f16 [ctx]) and pooled keys
// (f32 [ctx / 4]), the DeltaNet state + conv, the PLE conv, and the prompt-cache snapshot of the recurrent state (DeltaNet
// + PLE conv; KV and indexer keys stay in place, so a snapshot needs no copy of them).
inline uint64_t q4e_lane_bytes(const Q4eLaneShape& s, uint32_t ctx) {
    const uint64_t kv  = 2ull * s.n_full * s.n_kv_heads * uint64_t(ctx) * s.head_dim * 2;
    const uint64_t idx = s.idx_head_dim
        ? uint64_t(s.n_full) * ctx * s.idx_head_dim * 2 + uint64_t(s.n_full) * (ctx / 4) * s.idx_head_dim * 4 : 0;
    const uint64_t rec = uint64_t(s.n_lin) * (uint64_t(s.v_heads) * s.state * s.state * 4 +
                                              uint64_t(s.conv_channels) * (s.conv_kernel - 1) * 2);
    const uint64_t ple = s.ple_conv_floats * 4;
    return kv + idx + 2 * (rec + ple);   // the live recurrent state + its snapshot
}

// The expert-cache floor a stage keeps after its extra lanes (a floor picked for P4 B8; not measured).
inline constexpr uint64_t kQ4eLaneMinSlots = 64;

// The load-time admission of the extra lanes on one stage (Qwen4ExpModel::init_runtime; P4 B14 moved it here so it is
// host-tested): `extra` lanes of lane_bytes come out of the expert-cache budget; `left` gets what remains. "" = at least
// kQ4eLaneMinSlots slots per layer remain (slot_bytes = one expert slot summed over the stage's layers); else the refusal,
// with the numbers.
inline std::string q4e_lanes_cache_fit(uint64_t budget, uint64_t lane_bytes, uint32_t extra, uint32_t lane_ctx,
                                       uint64_t slot_bytes, uint64_t& left) {
    const uint64_t total = lane_bytes * extra;
    left = budget > total ? budget - total : 0;
    if (left / slot_bytes >= kQ4eLaneMinSlots) return {};
    return "qwen4exp lanes: " + std::to_string(extra) + " extra lane(s) of " + std::to_string(lane_bytes >> 20) + " MiB at ctx " +
           std::to_string(lane_ctx) + " leave " + std::to_string(left >> 20) + " MiB of the " + std::to_string(budget >> 20) +
           " MiB expert-cache budget = " + std::to_string(left / slot_bytes) + " slots/layer (< " +
           std::to_string(kQ4eLaneMinSlots) + "); lower --parallel or --slot-ctx";
}

// The lane context for lanes 1..n-1: --slot-ctx (0 = 32,768), at most lane 0's context.
inline uint32_t q4e_lane_ctx(uint32_t slot_ctx, uint32_t max_ctx) {
    const uint32_t c = slot_ctx ? slot_ctx : 32768u;
    return c < max_ctx ? c : max_ctx;
}

// Strict-prefix match of a lane's snapshot on the prompt: the engine restores a Flash-Next snapshot only when its tokens are a
// STRICT prefix of the new prompt (the recurrent state cannot be truncated; at least one prompt row runs for the first logits).
inline uint32_t q4e_snap_match(const std::vector<int32_t>& snap, std::span<const int32_t> ids) {
    if (snap.empty() || snap.size() >= ids.size()) return 0;
    for (size_t i = 0; i < snap.size(); ++i)
        if (snap[i] != ids[i]) return 0;
    return uint32_t(snap.size());
}

// The prefill pieces of [p, end) exactly as `--parallel 1` cuts them when uncontended (Engine::generate's prefill_to): more
// than kQ4ePipeChunk rows go through Q4eBundle::fwd_pipelined in kQ4ePipeChunk-row chunks; otherwise (or with
// IE_Q4E_NO_PIPELINE) the serial loop's pf_chunk-row chunks. The same pieces in the turn and through the lane pipe, so a
// lane's numerics never depend on what the other lanes do.
inline constexpr uint32_t kQ4ePipeChunk = 2048;   // Q4eBundle::fwd_pipelined's PC
inline void q4e_plan_range(uint32_t p, uint32_t end, uint32_t pf_chunk, bool pipeline,
                           std::vector<std::pair<uint32_t, uint32_t>>& out) {
    if (end <= p) return;
    const uint32_t piece = (pipeline && end - p > kQ4ePipeChunk) ? kQ4ePipeChunk : (pf_chunk ? pf_chunk : 1u);
    for (uint32_t q = p; q < end; q += piece) out.emplace_back(q, end - q < piece ? end - q : piece);
}
// P4 B15: [p, end) cut as --parallel 1 cuts it when the prefill is split at the shared-prefix boundary `mark`
// (prefill_to(mark), then prefill_to(end)): q4e_plan_range on each side. mark outside (p, end): one range.
inline void q4e_plan_split(uint32_t p, uint32_t end, uint32_t mark, uint32_t pf_chunk, bool pipeline,
                           std::vector<std::pair<uint32_t, uint32_t>>& out) {
    if (mark > p && mark < end) { q4e_plan_range(p, mark, pf_chunk, pipeline, out); q4e_plan_range(mark, end, pf_chunk, pipeline, out); }
    else q4e_plan_range(p, end, pf_chunk, pipeline, out);
}
// The largest piece q4e_plan_range makes.
inline uint32_t q4e_plan_max_rows(uint32_t pf_chunk, bool pipeline) {
    return pipeline ? kQ4ePipeChunk : pf_chunk;
}

// P4 B17 rows (Qwen4ExpModel::forward_rows): the most rows a group may carry on a stage. The grouped MoE body resolves every
// expert of the group's union into the stage's cache at once (a later resolve must not evict an earlier one's slot), so the
// union -- at most rows x top_k experts a layer -- must fit the slots a layer has; and 16 rows at most (the batched int-dot
// GEMV's largest bucket, the pinned router staging).
inline constexpr uint32_t kQ4eMaxRows = 16;
inline uint32_t q4e_rows_max_group(uint32_t slots_per_layer, uint32_t top_k) {
    if (top_k == 0) return 0;
    const uint32_t g = slots_per_layer / top_k;
    return g < kQ4eMaxRows ? g : kQ4eMaxRows;
}

// The T == 1 decode route a stage runs (the env switches that change it) and the per-layer facts forward_rows depends on. Rows
// mirror only the default route; anything else keeps the per-lane pipe (the reason is logged at load).
struct Q4eRowsRoute {
    bool dense_q8 = true;          // IE_Q4E_DENSE_Q8 (default on)
    bool a16 = false;              // IE_Q4E_A16=1
    bool smallk = false;           // IE_GEMV_SMALLK (the split-K T == 1 int-dot GEMV)
    bool decode_grouped = true;    // IE_Q4E_DECODE_GROUPED (default on; 0 = the per-pick decode body)
    bool moe_alt = false;          // IE_Q4E_MOE_EXPMAJOR / IE_Q4E_SV_NO_G1 / _NO_G2 / _NO_RED (non-default MoE routes)
    bool tp = false;               // verify-only expert parallelism is set
    bool dn_mixed = false;         // some DeltaNet layer has qkv and gate in different dtypes (the T == 1 gate reuses qkv's act)
    uint32_t n_lanes = 1;
    uint32_t slots_per_layer = 0, top_k = 0;
};
inline std::string q4e_rows_route_refusal(const Q4eRowsRoute& r) {
    if (r.n_lanes < 2) return "one lane";
    if (!r.dense_q8) return "IE_Q4E_DENSE_Q8=0 (the F16 dense route is not rows-gated)";
    if (r.a16) return "IE_Q4E_A16=1 (the A16 leaves are not rows-gated)";
    if (r.smallk) return "IE_GEMV_SMALLK (the split-K one-row GEMV has no row-exact batched twin)";
    if (!r.decode_grouped) return "IE_Q4E_DECODE_GROUPED=0 (the per-pick decode MoE body)";
    if (r.moe_alt) return "a non-default MoE route (IE_Q4E_MOE_EXPMAJOR / IE_Q4E_SV_NO_*)";
    if (r.tp) return "verify expert parallelism";
    if (r.dn_mixed) return "a DeltaNet layer's qkv and gate differ in dtype";
    if (q4e_rows_max_group(r.slots_per_layer, r.top_k) < 2)
        return "the expert cache's " + std::to_string(r.slots_per_layer) + " slots/layer hold fewer than 2 rows' top-" +
               std::to_string(r.top_k);
    return {};
}

// Why a Flash-Next load refuses image requests, "" when it takes them: the per-request image staging is engine-global
// (Engine members written before the generation gate), so --parallel > 1 -- request lanes or time-sliced -- refuses images.
// Engine::vision_status_json (/props "vision") and Engine::chat's refusal both come from here, so they cannot disagree.
inline std::string q4e_vision_refusal(uint32_t parallel) {
    if (parallel > 1) return "image inputs need --parallel 1 (per-request image staging is engine-global, not slot-safe)";
    return {};
}

// Why a DeepSeek-V4-Flash load refuses image requests, "" when it takes them (P4 B11): no vision sidecar, or --parallel > 1
// (its image staging, Engine::ds4_pending_imgs_, is engine-global like Flash-Next's). Engine::vision_status_json (/props
// "vision") and Engine::chat's refusal both come from here, so they cannot disagree.
inline std::string ds4_vision_refusal(bool sidecar_loaded, uint32_t parallel) {
    if (!sidecar_loaded) return "this deepseek4 load has no vision sidecar (*-Native.safetensors beside the GGUF, or $IE_DS4_VISION)";
    return q4e_vision_refusal(parallel);
}

}  // namespace ie
