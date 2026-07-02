// include/ie/qwen3moe.hpp — Qwen3 standard MoE (`qwen3moe`, e.g. Qwen3-Coder-30B-A3B).
//
// Forward = the Qwen3-dense attention block (QK-norm + partial RoPE + GQA full
// attention) with the dense FFN replaced by a top-k Mixture-of-Experts:
//   embedding → n × { rms(attn_norm) → attn → +res → rms(ffn_norm) → MoE → +res }
//             → rms(output_norm) → lm_head
// NO DeltaNet, NO shared expert. Additive — new files only; the crown
// (qwen35moe) MoE kernels are untouched (they hardcode 256 experts / fused
// scatter-gather; this v1 uses an UNFUSED per-token, per-expert path for
// correctness — router on host, expert GEMVs on device — to be optimized later).
#pragma once

#include "ie/allocator.hpp"
#include "ie/dense_transformer.hpp"   // DenseQuantPtr, upload helpers
#include "ie/gguf.hpp"
#include "ie/kv_cache.hpp"
#include "ie/model_config.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace ie {

class Qwen3MoeModel {
public:
    Qwen3MoeModel() = default;
    ~Qwen3MoeModel();
    Qwen3MoeModel(const Qwen3MoeModel&) = delete;
    Qwen3MoeModel& operator=(const Qwen3MoeModel&) = delete;

    std::string load(DeviceAllocator& alloc, const GgufReader& g, const Qwen3MoeConfig& cfg);
    const Qwen3MoeConfig& config() const noexcept { return cfg_; }

    // KV must be inited with n_layers_full = cfg.dense.n_layers, n_kv_heads =
    // cfg.dense.n_kv_heads, head_dim = cfg.dense.head_dim (every layer is full attn).
    sycl::event forward(sycl::queue& q, const int32_t* input_ids, uint32_t T,
                        uint32_t start_pos, KvCache& kv, sycl::half* out_logits);

    std::string ensure_workspace(uint32_t max_T);
    // FA-2 split-K decode partials scratch (mirror qwen35_dense / dense). Sized
    // for the largest ctx the caller will pass; alloc grows on demand. Decode
    // (T==1) uses full_attention_fa2_decode; prefill (T>1) stays on naive attn.
    std::string ensure_attn_partials(uint32_t max_ctx);

private:
    struct Layer {
        float* attn_norm = nullptr;
        float* ffn_norm  = nullptr;
        float* attn_q_norm = nullptr;
        float* attn_k_norm = nullptr;
        DenseQuantPtr attn_q, attn_k, attn_v, attn_output;
        // MoE: router weights kept on HOST (F32 [n_experts, hidden]) for the
        // decode (T==1) path; ALSO uploaded transposed to device F16 [hidden,
        // n_experts] for prefill, so router logits = ws_xn[T,H] @ router_w_devᵀ
        // run as one GPU gemm instead of a single-threaded host dot loop
        // (was ~66% of prefill wall-clock — Tier-3 router lever, 2026-06-12).
        std::vector<float> router_w;           // [n_experts * hidden]
        sycl::half*        router_w_dev = nullptr;  // device F16 [hidden, n_experts]
        // soa flags: per-expert SoA repack only pays for Q6_K (AoS ql/qh badly
        // uncoalesced → 2.4× decode); Q4_K AoS is already coalesced so SoA there
        // regresses (~38% on gate) — so SoA is applied per-tensor by dtype.
        void* gate_exps = nullptr; DType gate_dt = DType::kCount; uint64_t gate_stride = 0; bool gate_soa = false;
        void* up_exps   = nullptr; DType up_dt   = DType::kCount; uint64_t up_stride   = 0; bool up_soa   = false;
        void* down_exps = nullptr; DType down_dt = DType::kCount; uint64_t down_stride = 0; bool down_soa = false;
    };

    // Fused MoE FFN over T>1 tokens (prefill). Reads post-ffn-norm activations
    // from ws_xn_, leaves the per-token MoE output in ws_ffn_y_ [T,H].
    void moe_ffn_fused_prefill(sycl::queue& q, const Layer& w, uint32_t T);
    // IE_QWEN3MOE_MOE_XMX per-expert fp16-on-XMX prefill (gate/up GEMM →
    // SiLU·mul → down GEMM), writing UNWEIGHTED per-expert outputs into
    // ws_out_packed_ (the routing weight is applied later by moe_prefill_reduce).
    // Reads the host expert-offset slices from `pk`. Caller runs the reduce.
    void moe_xmx_prefill(sycl::queue& q, const Layer& w,
                         const struct MoePacking& pk, uint32_t T);
    // Fused MoE FFN for a single token (decode, T==1). Reads ws_xn_ [1,H],
    // writes the MoE output into ws_ffn_y_ [1,H].
    void moe_ffn_decode(sycl::queue& q, const Layer& w);

    DeviceAllocator* alloc_ = nullptr;
    Qwen3MoeConfig cfg_;
    std::vector<Layer> layers_;
    std::vector<void*> owned_;
    bool moe_exps_soa_ = false;    // experts repacked AoS→per-expert SoA at load
                                   // (coalesced int-dot reads; opt-out IE_NO_MOE_SOA)

    void*  token_embd_ = nullptr;  DType token_embd_dtype_ = DType::kCount;
    float* output_norm_ = nullptr;
    void*  output_ = nullptr;      DType output_dtype_ = DType::kCount;

    // workspace
    uint32_t ws_T_ = 0;
    sycl::half *ws_x_ = nullptr, *ws_xn_ = nullptr, *ws_q_ = nullptr, *ws_k_ = nullptr,
               *ws_v_ = nullptr, *ws_attn_out_ = nullptr, *ws_block_ = nullptr;
    sycl::half *ws_gate_ = nullptr, *ws_up_ = nullptr, *ws_eh_ = nullptr,
               *ws_eout_ = nullptr, *ws_acc_ = nullptr;     // per-expert MoE scratch
    // --- fused MoE FFN staging (T>1 prefill) ---
    uint32_t      ws_T_moe_ = 0;                 // capacity of the *_packed buffers
    uint32_t*     ws_expert_offsets_ = nullptr;  // [E+1] device
    int32_t*      ws_sorted_idx_     = nullptr;  // [T*K_top] device (token id per packed row)
    sycl::half*   ws_sorted_w_       = nullptr;  // [T*K_top] device (router weight per packed row)
    int32_t*      ws_tk_to_packed_   = nullptr;  // [T*K_top] device ((t,kslot) -> packed row)
    sycl::half*   ws_weights_packed_ = nullptr;  // [T*K_top] device (weight indexed by tk_to_packed slot)
    sycl::half*   ws_xp_packed_      = nullptr;  // [T*K_top, H] expert-sorted gathered input
    void*         ws_xp_q8_          = nullptr;  // [T*K_top, H/32] block_q8_1s (int-dot stage-1 in)
    void*         ws_h_q8_           = nullptr;  // [T*K_top, E_ffn/32] block_q8_1s (int-dot stage-2 in)
    sycl::half*   ws_h_packed_       = nullptr;  // [T*K_top, E_ffn] stage-1 output
    sycl::half*   ws_out_packed_     = nullptr;  // [T*K_top, H] stage-2 output
    sycl::half*   ws_ffn_y_          = nullptr;  // [T, H] reduced MoE output
    // --- IE_QWEN3MOE_MOE_XMX prefill fp16-on-XMX scratch (default-OFF; lazily
    //     sized on first use). Per-expert single-expert GEMM scratch so
    //     gemm_fp16's full-tile (round_up(M,8)×round_up(N,16)) row-tail spill
    //     lands in unused rows, never in another expert's slice. ---
    sycl::half*   ws_xmx_wbt_   = nullptr;  // [max(H,E_ffn) * max(E_ffn,H)] dequant B^T fp16
    float*        ws_xmx_cf32_  = nullptr;  // [round_up(TK,8) * max(E_ffn,H)] gemm_fp16 fp32 C
    sycl::half*   ws_xmx_gate_  = nullptr;  // [round_up(TK,8) * E_ffn] gate fp16 (per-expert base)
    sycl::half*   ws_xmx_up_    = nullptr;  // [round_up(TK,8) * E_ffn] up   fp16
    sycl::half*   ws_xmx_h_     = nullptr;  // [round_up(TK,8) * E_ffn] silu*mul fp16
    sycl::half*   ws_xmx_out_   = nullptr;  // [round_up(TK,8) * H]     down fp16 (per-expert base)
    uint32_t      ws_xmx_cap_   = 0;        // round_up(TK,8) the scratch is sized for
    std::string ensure_moe_xmx_scratch(uint32_t T);  // [T,H,EF] -> lazy alloc
    // --- fused MoE FFN decode (T==1) ---
    int32_t*      ws_topk_idx_       = nullptr;  // [K_top] device expert ids
    sycl::half*   ws_topk_w_         = nullptr;  // [K_top] device renormed weights
    std::vector<float> host_router_logits_;      // [E] reused per token
    sycl::half*        ws_router_logits_ = nullptr;  // device [T*E] prefill logits
    std::vector<float> host_logits_;             // [T*E] prefill logits (host copy)
    int32_t* ws_pos_ = nullptr;
    std::vector<sycl::half> host_xn_;   // [T*hidden] for the host router
    // --- FA-2 decode (T==1) attention partials scratch ---
    float*   ws_attn_partials_     = nullptr;  // [n_chunks_max, n_q_heads, head_dim+2] FP32
    uint32_t ws_attn_partials_ctx_ = 0;        // ctx the scratch is sized for

    void free_all();
};

}  // namespace ie
