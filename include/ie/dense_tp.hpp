// include/ie/dense_tp.hpp — TP: multi-GPU TENSOR-PARALLEL dense forward.
//
// ADDITIVE: never touches DenseModel (single-GPU) or DenseModelSplit (layer-
// split). Where layer-split puts whole layers on one card and runs them
// serially, TP splits EVERY layer across all cards (Megatron column-then-row)
// so both cards compute the same layer concurrently, with two all-reduces per
// layer. Target: ~1.4-2x faster decode than layer-split (which leaves each card
// idle ~50%). TP is NOT bit-exact vs single-GPU (the partial-sum reduction order
// differs) — validated by cosine>=0.999 + exact greedy argmax, not equality.
// Spec: docs/superpowers/specs/2026-06-12-tensor-parallel-decode-design.md.
//
// SCOPE (v1): Q4_K/Q6_K weights everywhere + F16 column-parallel weights
// (qwen3-8b's attn_v). F16 row-parallel weights are unsupported (no target has
// one: 72B is all-Q4_K; qwen3-8b's row-parallel o-proj/ffn_down are Q4_K/Q6_K).
// llama Q/K un-permute composed with the column split is a follow-up (validate
// on qwen3/qwen2 first). Same fp16 GEMV/GEMM path as the split — no int-dot.
#pragma once

#include "ie/allocator.hpp"          // DeviceFleet
#include "ie/deltanet_state.hpp"     // the FleetPrefixCache's DN accessors (a never-inited dummy: dense has no DeltaNet)
#include "ie/dense_transformer.hpp"  // DenseLayerWeights, DenseQuantPtr
#include "ie/dense_lookup.hpp"       // DenseLookupPolicy: the prompt-lookup drafts' bookkeeping (2026-10-28, serving campaign P2)
#include "ie/gguf.hpp"
#include "ie/kv_cache.hpp"
#include "ie/model_config.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ie {

class DenseModelTP {
public:
    DenseModelTP() = default;
    ~DenseModelTP();
    DenseModelTP(const DenseModelTP&) = delete;
    DenseModelTP& operator=(const DenseModelTP&) = delete;

    // Split every layer across all of `fleet`'s devices. `fleet` and `g` must
    // outlive this object. Returns error text on any unsupported geometry
    // (n_q/n_kv not divisible by n_dev; a column/row split that misses a
    // 256-superblock boundary; an F16 row-parallel weight; tied-embeddings).
    std::string load(DeviceFleet& fleet, const GgufReader& g, const DenseConfig& cfg,
                     uint32_t max_ctx = 2048);

    const DenseConfig& config() const noexcept { return cfg_; }

    // Forward of `T` tokens at `start_pos`, last token's logits → `out_logits_host`
    // (host fp16, size vocab). Prefill (T>1, start_pos=0, reset_kv=true) and decode
    // (T=1, start_pos=len, reset_kv=false) share one code path. Two host-bounce
    // all-reduces per layer (o-proj + ffn_down partials).
    // 2026-10-28 (P2): `all_logits_dev` = a card-0 [T x vocab] device buffer that receives EVERY row's logits (a lookup
    // round's rows, T <= kLookRowsMax, through the batched SoA int-dot lm_head); out_logits_host may then be null.
    std::string forward(const int32_t* input_ids, uint32_t T, uint32_t start_pos,
                        bool reset_kv, sycl::half* out_logits_host, sycl::half* all_logits_dev = nullptr);

    // 2026-10-28 (serving campaign P1): every card's workspace sized for `max_T` tokens now -- at the load, so the prompt
    // cache's room is measured after it and the first request allocates nothing. forward grows it on demand otherwise.
    std::string ensure_workspace(uint32_t max_T);

    // 2026-10-28 (P1): the FleetPrefixCache's per-card surface (fleet_prefix_cache.hpp; the KV-only shape of
    // Qwen3MoeTpModel). Dense has no DeltaNet: dev_has_dn() is false and dn_state() a never-inited dummy. A card's KV is
    // head-sharded (nkv kv heads a card) but self-contained -- the all-reduce is downstream of attention -- so a per-card
    // prefix snapshot restores exactly. One sequence (n_lanes 1) until the request lanes.
    DeviceFleet*   fleet()                      const noexcept { return fleet_; }
    uint32_t       n_devices()                  const noexcept { return n_dev_; }
    KvCache&       kv_cache(uint32_t dev)                      { return kv_[dev]; }
    DeltaNetState& dn_state(uint32_t /*dev*/)                  { return dn_dummy_; }
    bool           dev_has_kv(uint32_t dev)     const          { return kv_[dev].ready(); }
    bool           dev_has_dn(uint32_t /*dev*/) const          { return false; }
    uint32_t       n_lanes()                    const noexcept { return 1; }
    uint32_t       max_ctx()                    const noexcept { return kv_.empty() ? 0u : kv_[0].config().max_ctx; }

    // ---- 2026-10-28 (serving campaign P2): prompt-lookup drafts at --parallel 1 (the 35B split's look_step rule,
    // qwen35moe_split.hpp P4 B53, on a KV-only model; the bookkeeping is DenseLookupPolicy, dense_lookup.hpp, host-tested).
    // look_step(token, pos) gives the logits of position `pos` for the input `token`, as forward(T = 1) does. When the
    // context's last tokens repeat an earlier span (the prompt, then every token given), the tokens that followed it run
    // with `token` in ONE forward (a round: every row's logits kept on card 0 through the batched SoA int-dot lm_head, so
    // a round is at most kLookRowsMax rows) and the next calls are answered from those rows while the caller's token is
    // the drafted one. A row not followed costs only the KV lengths going back to it (the kernels read start_pos; no
    // re-run). The rows of a round carry the prefill kernels' numerics (naive attention, the T-row GEMVs), as a prompt's
    // rows do. Needs a Q6_K lm_head with its SoA repack and the Q8 decode path.
    static constexpr uint32_t kLookRowsMax = 16;
    std::string look_step_init(const DenseLookupParams& p);
    void        look_step_context(const int32_t* ids, uint32_t n);
    void        look_step_reset();   // the counters (a line when IE_DENSE_LOOKUP_STATS is set), the streak, the block
    std::string look_step(int32_t token, uint32_t pos, sycl::half* logits_host);
    std::string look_step_flush();   // the request ends: rows not followed -> the KV lengths back

    std::vector<uint64_t> device_bytes() const { return dev_bytes_; }

private:
    struct LookStep {
        std::unique_ptr<DenseLookupPolicy> pol;   // null = not initialised
        uint32_t    rows = 0;                     // the longest round (<= kLookRowsMax)
        sycl::half* d_all = nullptr;              // card 0: [rows x vocab], a round's logits
        void*       q8_rows = nullptr;            // card 0: [rows x hidden / 32] block_q8_1x, the round's normed rows quantized
        double      ms_round = 0, ms_plain = 0;
    } ls_;
    void rollback_to(uint32_t pos);               // every card, every layer: the KV length back to `pos`
    // Per-device, per-layer split weights (column slices + row re-packs).
    struct TpLayer {
        // replicated (full) on every card:
        float* attn_norm = nullptr;
        float* ffn_norm  = nullptr;
        float* attn_q_norm = nullptr;   // qwen3 [head_dim], replicated
        float* attn_k_norm = nullptr;
        // column-parallel (this card's output-head slice):
        DenseQuantPtr attn_q, attn_k, attn_v;     // [N_*_c, H]
        DenseQuantPtr ffn_gate, ffn_up;           // [F_c, H]
        float* attn_q_bias = nullptr;             // sliced to this card's heads
        float* attn_k_bias = nullptr;
        float* attn_v_bias = nullptr;
        // row-parallel (this card's input-K slice, re-packed):
        DenseQuantPtr attn_output;                // [H, N_q_c]
        DenseQuantPtr ffn_down;                   // [H, F_c]
    };

    // Per-device workspace, sized to this card's owned dims.
    struct Workspace {
        uint32_t T = 0;
        sycl::half *x = nullptr, *x_normed = nullptr, *q = nullptr, *k = nullptr,
                   *v = nullptr, *attn_out = nullptr, *attn_block = nullptr,
                   *gate = nullptr, *up = nullptr, *h = nullptr;
        int32_t* positions = nullptr;
        int32_t* ids = nullptr;        // 2026-10-28 (P1): the host ids' device copy (T; read by the embedding on card 0)
        void*    q8 = nullptr;         // 2026-10-28 (P1 fix): the decode step's Q8_1 activation scratch (DenseModel's ws_q8_)
        // 2026-10-28 (P3-B step 2): the fa2 decode kernel's split-K partials for this card's q heads, sized once for
        // the KV's max context (DenseModel::ensure_attn_partials' rule). Null = the naive kernel at T = 1 (as before).
        float* attn_partials = nullptr;
    };

    // Per-card head/ffn ownership (Megatron split bookkeeping).
    struct Shard {
        uint32_t q_head0 = 0, nq = 0;     // q-heads [q_head0, q_head0+nq)
        uint32_t kv_head0 = 0, nkv = 0;   // kv-heads
        uint32_t Nq = 0;                  // nq*head_dim (column dim of q / o-proj K-slice)
        uint32_t Nkv = 0;                 // nkv*head_dim
        uint32_t f0 = 0, fc = 0;          // ffn intermediate [f0, f0+fc) (256-aligned)
    };

    DeviceFleet* fleet_ = nullptr;
    DenseConfig  cfg_;
    uint32_t     n_dev_ = 0;

    std::vector<Shard>                    shard_;     // [dev]
    std::vector<std::vector<TpLayer>>     layers_;    // [dev][L]
    std::vector<KvCache>                  kv_;        // [dev]
    DeltaNetState                         dn_dummy_;  // never inited (ready() false): the FleetPrefixCache's DN accessors
    sycl::half*                           d_logits_ = nullptr;   // card 0: the last row's logits [vocab], allocated at load
    // 2026-10-28 (P1 fix): the lm_head's SoA-Q6 streams for the int-dot decode kernel (DenseModel's universal Q6_K lm_head)
    void* out_soa_q6_lo_ = nullptr; void* out_soa_q6_hi_ = nullptr; void* out_soa_q6_sc_ = nullptr; void* out_soa_q6_d_ = nullptr;
    std::vector<Workspace>                ws_;        // [dev]
    std::vector<std::vector<void*>>       owned_;     // [dev] device ptrs to free
    std::vector<uint64_t>                 dev_bytes_; // [dev] weight bytes

    // top-level (card 0):
    void*  token_embd_ = nullptr;  DType token_embd_dtype_ = DType::kCount;
    float* output_norm_ = nullptr;
    void*  output_ = nullptr;      DType output_dtype_ = DType::kCount;
    std::vector<float*> rope_freqs_per_dev_;   // [dev] (null if no scaling)

    std::string ensure_ws(uint32_t dev, uint32_t max_T);
    void free_all();
};

}  // namespace ie
