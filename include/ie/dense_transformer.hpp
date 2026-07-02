// include/ie/dense_transformer.hpp — P2: metadata-driven dense GQA
// transformer (Qwen3-dense family; Llama 3.x next). Forward structure:
//   embedding → n × { rms(attn_norm) → GQA attn (QK-norm, full RoPE)
//                     → residual → rms(ffn_norm) → SwiGLU FFN → residual }
//   → rms(output_norm) → lm_head
//
// Caller responsibility (same contract as QwenModel):
//   * keep `GgufReader` alive while the model is loaded.
//   * provide a `KvCache` sized with n_layers_full = cfg.n_layers,
//     n_kv_heads = cfg.n_kv_heads, head_dim = cfg.head_dim (every layer is
//     full-attention — no DeltaNet, no layer-interval indexing).
#pragma once

#include "ie/allocator.hpp"
#include "ie/gguf.hpp"
#include "ie/kv_cache.hpp"
#include "ie/model_config.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace ie {

struct DenseQuantPtr {
    void* p = nullptr; DType dt = DType::kCount;
    // EXL3 (kEXL3) only: the per-feature sign/scale side-vectors (device F16) and
    // the trellis bit-width. Null/0 for every non-EXL3 weight (byte-identical).
    void* suh = nullptr; void* svh = nullptr; uint32_t bits = 0;
    // SoA-Q4_0 fast decode streams (gemma4): device nibble + fp16-scale streams
    // for gemv_q4_0_soa_q8. Null unless the loader repacked this Q4_0 weight.
    void* soa_qs = nullptr; void* soa_d = nullptr;
    // Reordered Q4_K (llama-SYCL 3-region global SoA) for gemv_q4_K_reorder_q8.
    // Null unless the loader repacked this Q4_K weight (IE_QWEN35_Q4K_REORDER).
    void* reorder = nullptr;
};

// P3a: row permutation that INVERTS convert_hf_to_gguf.py LlamaModel.permute,
// applied to attn_q (n_heads = n_q_heads) and attn_k (n_heads = n_kv_heads) at
// load so our NEOX rope_partial(_ff) is exactly correct. Returns perm of size
// n_heads*head_dim with perm[dest] = src: destination row
//   h*hd + b*(hd/2) + k   sources from   h*hd + 2k + b   (k∈[0,hd/2), b∈{0,1}).
// qwen3/qwen2 do NOT call this (their GGUF rows already match NEOX pairing).
std::vector<uint32_t> llama_qk_unpermute_rows(uint32_t n_heads, uint32_t head_dim);

struct DenseLayerWeights {
    float* attn_norm = nullptr;       // blk.N.attn_norm.weight       F32
    float* ffn_norm  = nullptr;       // blk.N.ffn_norm.weight        F32
    DenseQuantPtr attn_q, attn_k, attn_v, attn_output;   // Q4_K/Q6_K/F16(transposed [K,N])
    float* attn_q_norm = nullptr;     // [head_dim] F32 — Qwen3 (absent for Qwen2)
    float* attn_k_norm = nullptr;
    float* attn_q_bias = nullptr;     // [n_q*head_dim] F32 — Qwen2 (absent for Qwen3)
    float* attn_k_bias = nullptr;     // [n_kv*head_dim] F32
    float* attn_v_bias = nullptr;     // [n_kv*head_dim] F32
    DenseQuantPtr ffn_gate, ffn_up, ffn_down;
    // Q6_K-repack decode optimization (IE_DENSE_Q6K_REPACK): a SoA Q8_0 copy of a
    // Q6_K ffn_down, used by the T==1 decode GEMV (gemv_q8_0_soa_q8, 2.96× vs the
    // scalar gemm_q6_K cliff). nullptr → not repacked (prefill always uses ffn_down).
    int8_t*   ffn_down_q8_qs = nullptr;   // int8 qs, column-major [N*K]
    uint16_t* ffn_down_q8_d  = nullptr;   // fp16 d per 32-block  [N*K/32]
};

class DenseModel {
public:
    DenseModel() = default;
    ~DenseModel();
    DenseModel(const DenseModel&) = delete;
    DenseModel& operator=(const DenseModel&) = delete;

    // Load all weights from `g` to device memory via `alloc`. Hard-fails
    // (returns error text) on any tensor dtype outside {Q4_K, Q6_K, F16} —
    // no silent empty-event dispatch paths exist for this model.
    std::string load(DeviceAllocator& alloc, const GgufReader& g,
                     const DenseConfig& cfg);
    const DenseConfig& config() const noexcept { return cfg_; }

    // KV cache must be inited with n_layers_full = cfg.n_layers,
    // n_kv_heads = cfg.n_kv_heads, head_dim = cfg.head_dim.
    sycl::event forward(sycl::queue& q,
                        const int32_t* input_ids, uint32_t T, uint32_t start_pos,
                        KvCache& kv, sycl::half* out_logits);

    // Sized workspace for at least `max_T` tokens. Call before forward()
    // when doing prefill of more than 1 token.
    std::string ensure_workspace(uint32_t max_T);

    // Ensure FA-2 split-K partials scratch is sized for the given ctx.
    // Idempotent — grows monotonically.
    std::string ensure_attn_partials(uint32_t max_ctx);

    // Per-layer activation dumps, same file naming as QwenModel:
    // ${prefix}_L<NN>.bin, NN = 00 (embed) .. n_layers (residuals) ..
    // n_layers+1 (final-norm output), plus `.meta` carrying `T H`.
    void set_dump_prefix(std::string p) { dump_prefix_ = std::move(p); }

private:
    DeviceAllocator* alloc_ = nullptr;
    DenseConfig cfg_;
    std::string dump_prefix_;     // empty = no dump

    // Top-level weights
    void*  token_embd_ = nullptr;  DType token_embd_dtype_ = DType::kCount;
    float* output_norm_ = nullptr;
    void*  output_ = nullptr;      DType output_dtype_ = DType::kCount;  // tied → token_embd_
    void*  output_suh_ = nullptr;  void* output_svh_ = nullptr;  uint32_t output_bits_ = 0;  // kEXL3 lm_head
    // P3a: llama3 RoPE frequency factors (rope_freqs.weight [rope_dim/2] F32),
    // shared by all layers. nullptr → no scaling (qwen3, llama-2/3.0).
    float* rope_freqs_ = nullptr;

    std::vector<DenseLayerWeights> layers_;
    std::vector<void*> owned_;     // device pointers to free at dtor

    // Workspace (allocated lazily, sized for max_T tokens):
    //   ws_x_/ws_x_normed_ [T,H]; ws_q_ [T,n_q*hd]; ws_k_/ws_v_ [T,n_kv*hd];
    //   ws_attn_out_ [T,n_q*hd]; ws_attn_block_ [T,H];
    //   ws_gate_/ws_up_/ws_h_ [T,ffn]; ws_positions_ [T];
    //   ws_q8_ (decode int-dot Q8_1 scratch, sized max(H,ffn)/32 blocks);
    //   ws_attn_partials_ (FA-2 split-K, same sizing rule as QwenModel).
    uint32_t    ws_T_ = 0;
    sycl::half* ws_x_ = nullptr;
    sycl::half* ws_x_normed_ = nullptr;
    sycl::half* ws_q_ = nullptr;
    sycl::half* ws_k_ = nullptr;
    sycl::half* ws_v_ = nullptr;
    sycl::half* ws_attn_out_ = nullptr;
    sycl::half* ws_attn_block_ = nullptr;
    sycl::half* ws_gate_ = nullptr;
    sycl::half* ws_up_ = nullptr;
    sycl::half* ws_h_ = nullptr;
    int32_t*    ws_positions_ = nullptr;
    void*       ws_q8_ = nullptr;
    float*      ws_attn_partials_ = nullptr;
    uint32_t    ws_attn_partials_ctx_ = 0;
};

}  // namespace ie
