// include/ie/qwen3moe_split.hpp — Qwen3 standard MoE (`kQwen3Moe`, e.g.
// Qwen3-Coder-30B-A3B / Tongyi-DeepResearch) multi-GPU LAYER-SPLIT.
//
// ADDITIVE companion to the validated single-GPU `Qwen3MoeModel` (qwen3moe.{hpp,cpp})
// — that file is NEVER edited, so its PPL gate cannot move. This mirrors the crown
// `Qwen35MoeSplitModel` ORCHESTRATION (device-by-device forward, per-card weights +
// KV, one residual copy per card boundary) but for the simpler dense-attention
// qwen3moe: standard Qwen3 QK-norm GQA attention (NO DeltaNet, NO shared expert,
// NO output gate) + top-k MoE FFN.
//
// APPROACH B (owner runs Q4_K_M / Q6_K GGUFs, no all-Q8_0): the experts keep the
// qwen3moe layout (AoS Q4_K / per-expert-SoA Q6_K, per-expert byte stride =
// nbytes/E) and the MoE runs the qwen3moe int-dot W4A8/W6A8 kernel sequence
// (moe_prefill_gate_up_silu_q{4,6}k_q8 → moe_prefill_down_q{4,6}k_q8_gen →
// moe_prefill_reduce), inlined per card. The attention projections ride
// dense::gemv_q_T (Q4_K/Q6_K/dequant-to-fp16), exactly like the single-GPU path.
// The router is a GPU gemv (F16 transposed router weight) + host softmax/top-K
// (matching the single-GPU DEFAULT numerics), NOT the crown's moe_router (which is
// hard-locked to E==256/K==8 and returns a no-op at qwen3moe's E==128).
//
// `--gpus 1` keeps using `Qwen3MoeModel`; `kQwen3Moe && --gpus>1` uses this.
#pragma once

#include "ie/allocator.hpp"          // DeviceFleet, DeviceAllocator
#include "ie/dense_split.hpp"        // LayerPlan
#include "ie/dense_transformer.hpp"  // DenseQuantPtr, dense::upload*
#include "ie/gguf.hpp"
#include "ie/kv_cache.hpp"
#include "ie/model_config.hpp"       // Qwen3MoeConfig

#include <sycl/sycl.hpp>
#include <cstdint>
#include <string>
#include <vector>

namespace ie {

class Qwen3MoeSplitModel {
public:
    Qwen3MoeSplitModel() = default;
    ~Qwen3MoeSplitModel();
    Qwen3MoeSplitModel(const Qwen3MoeSplitModel&) = delete;
    Qwen3MoeSplitModel& operator=(const Qwen3MoeSplitModel&) = delete;

    // Load every layer L onto plan.dev_of_layer[L]; embed→plan.embed_dev,
    // lm_head→plan.head_dev. Per-card full-attn KV sized to that card's layer
    // count. `fleet` and `g` must outlive this object. "" on success.
    std::string load(DeviceFleet& fleet, const LayerPlan& plan,
                     const GgufReader& g, const Qwen3MoeConfig& cfg,
                     uint32_t max_ctx = 2048);

    const Qwen3MoeConfig& config() const noexcept { return cfg_; }
    std::vector<uint64_t> device_bytes() const { return dev_bytes_; }
    uint32_t n_devices() const noexcept { return n_dev_; }

    // Device-by-device forward. Host ids in, last token's logits → host fp16 [vocab].
    std::string forward(const int32_t* input_ids, uint32_t T, uint32_t start_pos,
                        bool reset_kv, sycl::half* out_logits_host);

private:
    // Per-layer weights. Attention projections ride the dense GEMV path
    // (DenseQuantPtr); the MoE experts keep the qwen3moe expert-bank layout
    // (raw device buffer + per-expert byte stride + dtype + SoA flag). Mirrors
    // Qwen3MoeModel::Layer.
    struct LayerW {
        float* attn_norm   = nullptr;
        float* ffn_norm    = nullptr;              // GGUF key "ffn_norm.weight"
        float* attn_q_norm = nullptr;
        float* attn_k_norm = nullptr;
        DenseQuantPtr attn_q, attn_k, attn_v, attn_output;
        // MoE router: device F16 [hidden, E] (transposed) for the GPU-gemm router.
        sycl::half* router_w_dev = nullptr;
        // MoE experts (stacked banks; per-expert stride = nbytes/E).
        void* gate_exps = nullptr; DType gate_dt = DType::kCount; uint64_t gate_stride = 0; bool gate_soa = false;
        void* up_exps   = nullptr; DType up_dt   = DType::kCount; uint64_t up_stride   = 0; bool up_soa   = false;
        void* down_exps = nullptr; DType down_dt = DType::kCount; uint64_t down_stride = 0; bool down_soa = false;
    };

    // Per-card forward scratch. Mirrors Qwen3MoeModel::ws_* (dense attn + fused
    // MoE staging), one instance per device.
    struct Workspace {
        uint32_t T = 0;
        sycl::half *x = nullptr, *x_normed = nullptr;
        sycl::half *q = nullptr, *k = nullptr, *v = nullptr;
        sycl::half *attn_out = nullptr, *attn_block = nullptr;
        int32_t* positions = nullptr;
        float*   attn_partials = nullptr; uint32_t partials_ctx = 0;
        // router
        sycl::half* router_logits = nullptr;   // [T, E]
        // fused MoE staging (expert-sorted, int-dot q8)
        sycl::half* xp_packed  = nullptr;   // [T*K, H]      gathered expert-sorted x
        void*       xp_q8      = nullptr;   // [T*K, H/32]   block_q8_1s (stage-1 act)
        sycl::half* h_packed   = nullptr;   // [T*K, E_ffn]  silu(gate)*up
        void*       h_q8       = nullptr;   // [T*K, E_ffn/32] block_q8_1s (stage-2 act)
        sycl::half* out_packed = nullptr;   // [T*K, H]      per-packed-row down out
        sycl::half* ffn_y      = nullptr;   // [T, H]        reduced MoE output
        uint32_t*   expert_offsets = nullptr; // [E+1]
        int32_t*    sorted_idx     = nullptr; // [T*K]
        int32_t*    tk_to_packed   = nullptr; // [T*K]
        sycl::half* weights_packed = nullptr; // [T*K]
    };

    DeviceFleet*   fleet_ = nullptr;
    LayerPlan      plan_;
    Qwen3MoeConfig cfg_;
    uint32_t       n_dev_ = 0;
    bool           moe_exps_soa_ = false;   // Q6_K experts repacked AoS→per-expert SoA

    std::vector<LayerW> layers_;
    void*  token_embd_ = nullptr;  DType token_embd_dtype_ = DType::kCount;
    float* output_norm_ = nullptr;
    void*  output_ = nullptr;      DType output_dtype_ = DType::kCount;

    std::vector<std::vector<void*>> owned_;    // [dev] → device ptrs to free
    std::vector<Workspace>          ws_;       // [dev]
    std::vector<KvCache>            kv_;        // [dev], sized to the dev's layer count
    std::vector<uint32_t>           kv_local_; // [global L] → local full-attn index on its dev
    std::vector<uint64_t>           dev_bytes_;// [dev] weight bytes (reporting)
    std::vector<float>              host_logits_; // [T*E] router-logit host scratch (reused)

    // Inlined qwen3moe fused-MoE sequence for one layer on one card → ws_[dev].ffn_y.
    void moe_ffn(uint32_t dev, const LayerW& w, uint32_t T);

    std::string ensure_ws(uint32_t dev, uint32_t max_T);
    void free_ws(uint32_t dev);
    void free_all();
};

}  // namespace ie
