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
#include "ie/dense_transformer.hpp"  // DenseLayerWeights, DenseQuantPtr
#include "ie/gguf.hpp"
#include "ie/kv_cache.hpp"
#include "ie/model_config.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
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
    std::string forward(const int32_t* input_ids, uint32_t T, uint32_t start_pos,
                        bool reset_kv, sycl::half* out_logits_host);

    std::vector<uint64_t> device_bytes() const { return dev_bytes_; }

private:
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
