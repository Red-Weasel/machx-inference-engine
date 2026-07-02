// include/ie/qwen36.hpp — full model: load Qwen3.6-35B-A3B GGUF + forward.
//
// Phase 8 deliverable. Wires every primitive built in phases 1–7 into a 40-
// layer forward pass:
//   embedding → 40 × { rms_norm(1+w) → (FullAttn | DeltaNet) → residual →
//                       rms_norm(1+w) → MoE → residual } → rms_norm(1+w)
//                                                       → lm_head_GEMV
//
// Caller responsibility:
//   * keep `GgufReader` alive while the model is loaded (init copies all
//     tensor data to device, but read-only side dictionary is borrowed).
//   * provide a `KvCache` and `DeltaNetState` sized for at least the
//     intended max context.

#pragma once

#include "ie/allocator.hpp"
#include "ie/deltanet_state.hpp"
#include "ie/gguf.hpp"
#include "ie/kv_cache.hpp"
#include "ie/tokenizer.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <string>
#include <functional>
#include <vector>

namespace ie {

struct QwenConfig {
    uint32_t n_layers           = 40;
    uint32_t hidden             = 2048;
    uint32_t n_q_heads          = 16;
    uint32_t n_kv_heads         = 2;
    uint32_t head_dim           = 256;
    uint32_t vocab              = 248320;
    uint32_t rope_dim           = 64;       // partial-rotary (head_dim * 0.25)
    float    rope_theta         = 1.0e7f;
    uint32_t n_experts          = 256;
    uint32_t experts_topk       = 8;
    uint32_t expert_ffn         = 512;
    uint32_t full_attn_interval = 4;
    uint32_t ssm_inner          = 4096;     // 32 v-heads × 128 head_dim
    uint32_t ssm_n_v_heads      = 32;
    uint32_t ssm_n_k_heads      = 16;
    uint32_t ssm_head_dim       = 128;
    uint32_t ssm_conv_kernel    = 4;
    float    rms_eps            = 1.0e-6f;
};

struct QuantPtr { void* p = nullptr; DType dt = DType::kCount; };

struct LayerWeights {
    // Block-level norms (always present)
    float* attn_norm = nullptr;
    float* post_attn_norm = nullptr;

    // MoE FFN (every layer). Dtype is Q4_K or Q6_K per-layer.
    float* ffn_gate_inp = nullptr;
    float* ffn_gate_inp_shexp = nullptr;
    QuantPtr ffn_gate_exps;
    QuantPtr ffn_up_exps;
    QuantPtr ffn_down_exps;
    QuantPtr ffn_gate_shexp;
    QuantPtr ffn_up_shexp;
    QuantPtr ffn_down_shexp;

    // Full-attn weights (only on layers where i % 4 == 3, else nullptr/empty)
    QuantPtr attn_q;
    QuantPtr attn_k;
    QuantPtr attn_v;
    QuantPtr attn_output;
    float* attn_q_norm = nullptr;
    float* attn_k_norm = nullptr;

    // DeltaNet weights (only on i % 4 != 3)
    QuantPtr attn_qkv;
    QuantPtr attn_gate;
    float* ssm_a = nullptr;
    QuantPtr ssm_alpha;
    QuantPtr ssm_beta;
    float* ssm_conv1d = nullptr;
    sycl::half* ssm_conv1d_fp16 = nullptr;
    float* ssm_dt_bias = nullptr;
    float* ssm_norm = nullptr;
    sycl::half* ssm_norm_fp16 = nullptr;
    QuantPtr ssm_out;
};

// Generic per-position DeltaNet checkpoint (defined in qwen35_dense; reused here
// since it depends only on DeltaNetState, which the Crown shares). Forward-declared
// so the spec-verify `ckpt` param below stays a pointer with no header coupling.
struct Qwen35SpecCheckpoint;

// Native MTP/NextN draft head for the Crown (blk.<n_layers> = blk.40). Loaded only
// for spec decode. The head is a FULL-ATTN + MoE transformer block + the nextn input
// projection (enorm/hnorm/eh_proj) + shared_head_norm; the lm_head is shared (output_).
// HYBRID layout (like the 27B MtpHead but MoE FFN): attn q/k/v/output + eh_proj are
// dequanted to fp16 (gemv_fp16, M=1 draft); the 256 expert banks stay QUANTIZED and
// run through the Crown's moe_router + moe_decode_* kernels (can't fp16-expand 256
// experts). The draft need NOT be bit-exact — losslessness comes from the verify
// forward (already has the hooks) + commit_to_n; draft quality only affects acceptance.
struct CrownMtpHead {
    // Norms (F32) and nextn projections.
    float* enorm = nullptr;        // RMSNorm on prev target hidden
    float* hnorm = nullptr;        // RMSNorm on token embedding
    float* shead_norm = nullptr;   // final norm before lm_head
    float* attn_norm = nullptr;
    float* post_attn_norm = nullptr;
    float* attn_q_norm = nullptr;
    float* attn_k_norm = nullptr;
    sycl::half* eh_proj = nullptr;     // [2H,H] dequant fp16 (gemv_fp16)
    // Attn weights dequanted to fp16 (M=1 draft GEMVs).
    sycl::half* w_attn_q = nullptr;    // [H, N_qg] (Q + gate fold)
    sycl::half* w_attn_k = nullptr;    // [H, N_kv]
    sycl::half* w_attn_v = nullptr;    // [H, N_kv]
    sycl::half* w_attn_out = nullptr;  // [N_q, H]
    sycl::half* w_lm_head = nullptr;   // = model output_ (shared) dequant fp16, [H, vocab]
    // MoE FFN router (F32; BF16 cast at load) + experts dequanted to fp16 (a
    // per-expert fp16 loop — sidesteps the unsloth UD quant's Q5_K ffn_down,
    // which has no decode kernel). Shared expert skipped in the draft.
    float*      ffn_gate_inp = nullptr;   // [H, E] router
    sycl::half* exp_gate = nullptr;       // [E][H, E_ffn]
    sycl::half* exp_up   = nullptr;       // [E][H, E_ffn]
    sycl::half* exp_down = nullptr;       // [E][E_ffn, H]
    // Token embedding (independent upload for the per-draft-step lookup).
    void*  te_dev = nullptr; DType te_dtype = DType::kCount;
    // Head full-attn KV cache (n_kv_heads * max_ctx * head_dim).
    sycl::half* kc = nullptr; sycl::half* vc = nullptr;
    uint32_t max_ctx = 0;
    // Per-step scratch (all [H] or projection-width, fp16).
    sycl::half *d_e=nullptr, *d_hn=nullptr, *d_en=nullptr, *d_cat=nullptr, *d_x=nullptr;
    sycl::half *d_xn=nullptr, *d_qg=nullptr, *d_q=nullptr, *d_gate=nullptr, *d_k=nullptr;
    sycl::half *d_v=nullptr, *d_ao=nullptr, *d_blk=nullptr, *d_moe=nullptr, *d_logits1=nullptr;
    sycl::half *d_fg=nullptr, *d_fu=nullptr, *d_fh=nullptr, *d_tmp=nullptr;  // MoE expert scratch
    int32_t *d_pos1=nullptr, *d_tok1=nullptr;
    std::vector<void*> owned;     // head-owned device allocations (freed at model dtor)
    bool loaded = false;
};

class QwenModel {
public:
    QwenModel() = default;
    ~QwenModel();
    QwenModel(const QwenModel&) = delete;
    QwenModel& operator=(const QwenModel&) = delete;

    // Load all weights from `g` to device memory via `alloc`. Caches must be
    // initialized separately (caller passes them in to forward()).
    std::string load(DeviceAllocator& alloc, const GgufReader& g, const QwenConfig& cfg);

    const QwenConfig& config() const noexcept { return cfg_; }

    // Prefill-crown (2026-06-10): true when the MoE expert tensors
    // (ffn_gate_exps / ffn_up_exps / ffn_down_exps) were repacked at load
    // into per-expert SoA streams (ie/quant_soa.hpp).  Disable with
    // IE_NO_MOE_SOA=1 for order-controlled A/Bs.
    bool moe_exps_soa() const noexcept { return moe_exps_soa_; }

    // Forward pass on the given prompt token IDs at positions [start_pos, start_pos+T).
    //   input_ids: device pointer to int32 [T] token IDs
    //   T: number of tokens
    //   start_pos: position in the cache for the first new token
    //   kv: full-attention KV cache (pre-initialized for ≥ start_pos+T tokens)
    //   dn: DeltaNet recurrent state (pre-initialized; reset to zero on first call)
    //   out_logits: device pointer to fp16 [vocab] — logits for the LAST token only
    //
    // Workspace is allocated on first call and reused; configurable via the
    // optional `max_T` field (defaults to 1, growable).
    // Spec-decode verify hooks (all default null = unchanged Crown path, byte-
    // identical → gate-safe). Mirror Qwen35DenseModel::forward:
    //   * all_logits != null → lm_head on ALL T positions → [T, vocab].
    //   * hidden_pre_norm != null → copy the pre-output_norm residual ([T,hidden],
    //     the h_i the native MTP head consumes) into it.
    //   * ckpt != null → CHECKPOINT MODE: run each DeltaNet layer's conv +
    //     recurrence as T single-token scans, snapshotting state AFTER each
    //     position (rollback-free commit). MoE is stateless → untouched.
    sycl::event forward(sycl::queue& q,
                        const int32_t* input_ids, uint32_t T, uint32_t start_pos,
                        KvCache& kv, DeltaNetState& dn,
                        sycl::half* out_logits,
                        sycl::half* all_logits = nullptr,
                        sycl::half* hidden_pre_norm = nullptr,
                        Qwen35SpecCheckpoint* ckpt = nullptr);

    // Sized workspace for at least `max_T` tokens. Call before forward() when
    // doing prefill of more than 1 token.
    std::string ensure_workspace(uint32_t max_T);

    // ---- MTP self-speculative decode (single-GPU, GREEDY) — mirrors the 27B ----
    // Load the native blk.40 NextN head on demand (~0.6 GB). Idempotent.
    std::string load_mtp_head(const GgufReader& g, uint32_t max_ctx);
    bool mtp_loaded() const noexcept { return mtp_.loaded; }
    using SpecEmit = std::function<bool(int32_t)>;
    // Run MTP self-spec GREEDY decode. Caller prefilled the prompt into kv/dn up to
    // start_pos and passes the target's pre-output_norm hidden at the last prompt
    // position (h_last, device fp16 [hidden]) + the target argmax there (tn). Emits up
    // to max_new committed tokens through emit IN ORDER (lossless == plain greedy).
    // K = draft length (4 optimal). Reuses gemv_q4_K_q8s_batched for the verify.
    std::string spec_generate(sycl::queue& q, KvCache& kv, DeltaNetState& dn,
                              sycl::half* h_last, int32_t tn,
                              uint32_t start_pos, uint32_t max_new, uint32_t K,
                              const SpecEmit& emit);

    // Set a path prefix for per-layer activation dumps. When non-empty, every
    // forward() call writes 42 fp32 binary files: ${prefix}_L<NN>.bin where
    // NN is 00 (embed output), 01..40 (residual after each of 40 layers),
    // 41 (final-norm output). Each file is just T*hidden float32 elements,
    // little-endian.  The filename `<prefix>_L<NN>.meta` carries `T H` text.
    // Set to empty string to disable. Default = empty.
    void set_dump_prefix(std::string prefix) { dump_prefix_ = std::move(prefix); }

    // Ensure FA-2 split-K partials scratch is sized for the given ctx.
    // Idempotent — grows monotonically. Caller passes the KV cache's
    // max_ctx so partials cover the worst-case ctx the model will see.
    std::string ensure_attn_partials(uint32_t max_ctx);

    // Profiling — per-section ms accumulator, opt-in.  When enabled,
    // forward() inserts q.wait() boundaries around each labelled section
    // (loses pipelining, kernel-level wall time is accurate per section).
    void set_profile(bool b);
    void dump_profile(std::FILE* out) const;
    void clear_profile();

    // Partial-stack logit capture (PR #1: DeltaNet-as-draft acceptance-rate
    // measurement).  When non-empty, forward() will at each cut layer N apply
    // final-norm + lm_head to the LAST token's residual and write [vocab]
    // fp16 logits into per_cut_out_logits[i].  N=cut_layer means "after layer
    // index N has fully completed" (so N=10 means residual after layer 9 is
    // scored — but for symmetry with dump_residual we use 1-based: N=10
    // captures after L01..L10 = post-layer-10 residual, equivalent to the
    // existing _L10.bin dump).
    //
    // Buffers must be device pointers, each sized [vocab] fp16, valid for
    // the lifetime of subsequent forward() calls.  Sizes of cut_layers and
    // per_cut_out_logits must match.  cut_layers is automatically sorted.
    // Pass an empty list (or call clear_partial_logits_capture) to disable.
    //
    // This path is purely observational: it does not modify KV cache or
    // DeltaNet state, and the full-stack out_logits returned by forward()
    // is bit-identical to the no-capture case.  PPL preserved by construction.
    void set_partial_logits_capture(std::vector<uint32_t> cut_layers,
                                    std::vector<sycl::half*> per_cut_out_logits);
    void clear_partial_logits_capture();

private:
    DeviceAllocator* alloc_ = nullptr;
    QwenConfig cfg_;
    std::string dump_prefix_;     // empty = no dump

    // Top-level weights
    void*  token_embd_ = nullptr;          // Q4_K or Q6_K [hidden, vocab]
    DType  token_embd_dtype_ = DType::kCount;
    float* output_norm_ = nullptr;         // [hidden] FP32
    void*  output_ = nullptr;              // Q4_K or Q6_K [hidden, vocab] (lm_head)
    DType  output_dtype_ = DType::kCount;

    std::vector<LayerWeights> layers_;
    CrownMtpHead mtp_;     // native MTP/NextN draft head (loaded only for spec)
    // MTP draft head internals (defined in qwen36.cpp; only called from spec_generate).
    void mtp_head_forward(sycl::queue& q, const sycl::half* h_src, int32_t e_tok, int32_t pos);
    void mtp_draft(sycl::queue& q, const sycl::half* h_last, int32_t tn,
                   uint32_t p_base, uint32_t K, std::vector<int32_t>& out);
    std::vector<void*>        owned_;      // device pointers to free at dtor
    bool moe_exps_soa_ = false;            // expert tensors in per-expert SoA layout

    // Workspace (allocated lazily, sized for max_T tokens)
    uint32_t    ws_T_ = 0;
    sycl::half* ws_x_ = nullptr;           // [T, hidden]
    sycl::half* ws_x_residual_ = nullptr;
    sycl::half* ws_x_normed_ = nullptr;
    sycl::half* ws_q_full_ = nullptr;      // [T, n_q*head*2] (raw q_proj output, interleaved Q|gate per head)
    sycl::half* ws_q_split_ = nullptr;     // [T, n_q*head] (Q after split)
    sycl::half* ws_gate_split_ = nullptr;  // [T, n_q*head] (output-gate after split)
    sycl::half* ws_k_ = nullptr;           // [T, n_kv*head]
    sycl::half* ws_v_ = nullptr;
    sycl::half* ws_attn_out_ = nullptr;    // [T, n_q*head]
    sycl::half* ws_attn_block_ = nullptr;  // [T, hidden]
    // FA-2 split-K partials: [n_chunks_max, n_q_heads, head_dim+2] FP32.
    // Sized for max_ctx; reused across all full-attn layers within a forward.
    float*      ws_attn_partials_ = nullptr;
    uint32_t    ws_attn_partials_ctx_ = 0;
    // Profiling state (only used when set_profile(true) is called)
    bool        profile_enabled_ = false;
    mutable std::vector<std::pair<std::string, double>> profile_totals_;
    // Partial-stack logit capture (PR #1).  Empty = disabled.
    std::vector<uint32_t>     partial_cuts_;          // sorted ascending, 1..n_layers
    std::vector<sycl::half*>  partial_out_logits_;    // device [vocab] fp16, parallel to partial_cuts_
    sycl::half*               ws_partial_normed_ = nullptr;  // [hidden] scratch (last-token rms_norm output)
    int32_t*    ws_positions_ = nullptr;   // [T]
    // DeltaNet workspace
    sycl::half* ws_qkv_ = nullptr;         // [T, ssm_inner*2]
    sycl::half* ws_qkv_silu_ = nullptr;
    sycl::half* ws_alpha_fp16_ = nullptr;  // [T, n_v_heads]
    sycl::half* ws_beta_fp16_ = nullptr;
    sycl::half* ws_z_fp16_ = nullptr;      // [T, ssm_inner]
    float*      ws_q_fp32_ = nullptr;      // [T, n_v_heads, head_dim] (32 heads, post-GQA)
    float*      ws_k_fp32_ = nullptr;
    float*      ws_v_fp32_ = nullptr;
    float*      ws_q_fp32_pre_ = nullptr;  // [T, n_k_heads, head_dim] (16 heads, pre-GQA)
    float*      ws_k_fp32_pre_ = nullptr;
    float*      ws_a_fp32_ = nullptr;      // [T, n_v_heads]
    float*      ws_b_fp32_ = nullptr;
    float*      ws_g_fp32_ = nullptr;
    float*      ws_beta_fp32_ = nullptr;
    float*      ws_recurrence_out_ = nullptr;  // [T, n_v_heads, head_dim]
    sycl::half* ws_gated_norm_ = nullptr;
    // MoE workspace
    int32_t*    ws_topk_idx_ = nullptr;    // [T, k]
    sycl::half* ws_topk_w_ = nullptr;
    sycl::half* ws_gate_o_ = nullptr;      // [T, expert_ffn]
    sycl::half* ws_up_o_ = nullptr;
    sycl::half* ws_h_o_ = nullptr;
    sycl::half* ws_h_routed_ = nullptr;    // [K_top, expert_ffn] — fused MoE stage 1 → stage 2 (T=1 only)
    // Scatter-gather MoE prefill scratch: gathered input rows for one expert,
    // sized for the worst case (all tokens route to one expert).
    sycl::half* ws_moe_x_packed_ = nullptr;   // [max_T, hidden]
    int32_t*    ws_moe_token_idx_ = nullptr;  // [max_T*K_top] sorted-by-expert
    sycl::half* ws_moe_token_w_ = nullptr;    // [max_T*K_top]
    // Multi-expert prefill (item 10) workspace.  Sized for max_T_moe = min(max_T, 4096):
    //   ws_moe_h_packed_:    stage-1 output [max_T_moe * K_top, E_ffn]
    //   ws_moe_out_packed_:  stage-2 output [max_T_moe * K_top, H] — atomics-free
    //                        per-(packed_row) write, then reduce kernel sums per token.
    //   ws_moe_tk_to_packed_: inverse map for the reduce kernel,
    //                        tk_to_packed[t*K_top + k_slot] → packed_row
    //   ws_moe_expert_offsets_: [E+1] expert offset table
    //   ws_moe_y_fp32_: legacy fp32 accumulator (kept for the older atomic-add
    //                  variant — currently unused; will retire once the no-atomic
    //                  path proves out at all T).
    uint32_t*   ws_moe_expert_offsets_ = nullptr;
    sycl::half* ws_moe_h_packed_ = nullptr;
    sycl::half* ws_moe_out_packed_ = nullptr;
    sycl::half* ws_moe_up_packed_ = nullptr;   // oneDNN large-M MoE: up-proj out [max_T_moe*K_top, E_ffn]
    sycl::half* ws_moe_btf16_ = nullptr;       // oneDNN large-M MoE: per-expert dequant Bt [H*E_ffn]
    sycl::half* ws_moe_xp_ = nullptr;          // E5: expert-sorted gathered x
    void*       ws_q8_ = nullptr;              // P1a: Q8_1 activation scratch (decode)
    void*       ws_moe_xp_q8_ = nullptr;       // P1b-2: Q8_1s over ws_moe_xp_ (prefill)
    void*       ws_moe_h_q8_ = nullptr;        // prefill-crown: Q8_1s over h_packed

    uint32_t*   ws_moe_tk_to_packed_ = nullptr;
    float*      ws_moe_y_fp32_ = nullptr;
    uint32_t    ws_T_moe_ = 0;
    sycl::half* ws_eo_ = nullptr;          // [T, hidden]
    sycl::half* ws_sh_g_ = nullptr;        // [T] sigmoid scalar
};

}  // namespace ie
