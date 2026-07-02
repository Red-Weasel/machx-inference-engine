// include/ie/gptoss.hpp — gpt-oss (OpenAI MoE, `gpt-oss`): GQA dense attention
// with three non-standard pieces + top-4 MoE.
//
// Forward = embedding → n × { rms(attn_norm) → GQA-attn(+q/k/v/o BIAS, per-head
//   SINK, alternating SWA[even]/full[odd]) → +res → rms(post_attn_norm) →
//   top-4 MoE(+router/expert BIAS, clamped gated-SwiGLU/OAI) → +res }
//   → rms(output_norm) → lm_head.
//
// vs qwen3moe (the structural base): NO QK-norm, NO shared expert, NO DeltaNet.
// The deltas that earned a fork (rather than branching qwen3moe and risking the
// Coder/crown bit-exact gates): biases everywhere, per-head attention sinks,
// alternating sliding-window, MXFP4 experts (oneDNN-only — no int-dot kernel),
// and clamped gated-SwiGLU. All heavy ops are shared free functions; this class
// just orchestrates them (single-GPU, mirrors Qwen3MoeModel).
#pragma once

#include "ie/allocator.hpp"
#include "ie/dense_transformer.hpp"   // DenseQuantPtr, dense::upload helpers
#include "ie/gguf.hpp"
#include "ie/kv_cache.hpp"
#include "ie/model_config.hpp"
#include "ie/qwen3moe_pack.hpp"       // MoePacking + build_moe_packing

#include <sycl/sycl.hpp>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ie {

class GptOssModel {
public:
    GptOssModel() = default;
    ~GptOssModel();
    GptOssModel(const GptOssModel&) = delete;
    GptOssModel& operator=(const GptOssModel&) = delete;

    std::string load(DeviceAllocator& alloc, const GgufReader& g, const GptOssConfig& cfg);
    const GptOssConfig& config() const noexcept { return cfg_; }

    // KV must be inited with n_layers_full = cfg.dense.n_layers, n_kv_heads =
    // cfg.dense.n_kv_heads (8), head_dim = cfg.dense.head_dim (64). Every layer is
    // full-attention at the cache level; sliding-window is a softmax-mask concern
    // applied INSIDE attention, not a cache-geometry one (so KV is not shrunk).
    sycl::event forward(sycl::queue& q, const int32_t* input_ids, uint32_t T,
                        uint32_t start_pos, KvCache& kv, sycl::half* out_logits);

    std::string ensure_workspace(uint32_t max_T);
    // FA-2 split-K decode partials scratch (mirror qwen3moe). Lazily grown.
    std::string ensure_attn_partials(uint32_t max_ctx);

private:
    struct Layer {
        float* attn_norm      = nullptr;   // F32 [hidden]  (pre-attn)
        float* post_attn_norm = nullptr;   // F32 [hidden]  (pre-MoE; gguf post_attention_norm)
        // Q8_0 weights → dequanted to F16 [K,N] at load (upload_weight_auto).
        DenseQuantPtr attn_q, attn_k, attn_v, attn_output;
        // attention biases (F32 device vectors). q-dim = n_q*head_dim, kv-dim =
        // n_kv*head_dim, o-dim = hidden.
        float* attn_q_bias = nullptr;      // [n_q_heads*head_dim]   (4096)
        float* attn_k_bias = nullptr;      // [n_kv_heads*head_dim]  (512)
        float* attn_v_bias = nullptr;      // [n_kv_heads*head_dim]  (512)
        float* attn_o_bias = nullptr;      // [hidden]               (2880)
        float* attn_sinks  = nullptr;      // [n_q_heads]            (64) per-head softmax sink
        bool   is_swa = false;             // even layers (L%2==0) → window 128, odd → full

        // router: F32 [E,hidden] host (decode dot) + transposed F16 [hidden,E]
        // device (prefill gemm), plus the F32[E] router bias.
        std::vector<float> router_w;
        sycl::half*        router_w_dev = nullptr;
        float*             router_bias  = nullptr;   // F32 [n_experts]

        // experts (MXFP4) — load-time SoA repack into aligned per-expert planes:
        //   *_qs : nibble plane, per-expert stride N*(K/2) bytes
        //   *_e  : E8M0 exponent plane, per-expert stride N*(K/32) bytes
        // Decode (T=1) → gemv_mxfp4_soa_q8 int-dot; prefill (T≥2) → dequant_mxfp4_soa_to_Bt.
        // gate/up: K=hidden, N=expert_ffn.  down: K=expert_ffn, N=hidden.
        uint8_t* gate_qs = nullptr; uint8_t* gate_e = nullptr;
        uint8_t* up_qs   = nullptr; uint8_t* up_e   = nullptr;
        uint8_t* down_qs = nullptr; uint8_t* down_e = nullptr;
        uint64_t gate_qs_stride = 0, gate_e_stride = 0;
        uint64_t up_qs_stride   = 0, up_e_stride   = 0;
        uint64_t down_qs_stride = 0, down_e_stride = 0;
        // per-expert biases (F32). gate/up packed [n_experts, expert_ffn]; down
        // packed [n_experts, hidden]. Added before activation (gate/up) / before
        // the weighted reduce (down).
        float* gate_bias = nullptr;        // F32 [n_experts*expert_ffn]
        float* up_bias   = nullptr;        // F32 [n_experts*expert_ffn]
        float* down_bias = nullptr;        // F32 [n_experts*hidden]
    };

    // top-4 MoE for T tokens (prefill OR decode — unified per-expert oneDNN path,
    // MXFP4 has no int-dot kernel). Reads post-norm activations from ws_xn_, leaves
    // the per-token MoE output in ws_moe_y_ [T,H].
    void moe_ffn(sycl::queue& q, const Layer& w, uint32_t T);

    DeviceAllocator* alloc_ = nullptr;
    GptOssConfig cfg_;
    std::vector<Layer> layers_;
    std::vector<void*> owned_;

    void*  token_embd_ = nullptr;  DType token_embd_dtype_ = DType::kCount;
    float* output_norm_ = nullptr;
    DenseQuantPtr output_;         // lm_head fp16 fallback (or A/B opt-out path)
    // lm_head W8A8 int-dot (Q8_0 SoA): halves the fp16 weight read (1.16→0.58 GB,
    // the single biggest decode gemv) — reusable for the 120b non-expert fit.
    int8_t*  lmhead_qs_ = nullptr; // int8 qs col-major [N*K]
    uint16_t* lmhead_d_ = nullptr; // per-32 fp16 d [N*(K/32)]
    uint32_t lmhead_K_ = 0, lmhead_N_ = 0;   // nonzero → use the int-dot lm_head

    // ---- workspace (single device) ----
    uint32_t ws_T_ = 0;
    sycl::half *ws_x_ = nullptr, *ws_xn_ = nullptr, *ws_block_ = nullptr,
               *ws_q_ = nullptr, *ws_k_ = nullptr, *ws_v_ = nullptr,
               *ws_attn_out_ = nullptr;
    int32_t* ws_pos_ = nullptr;
    // MoE oneDNN scratch (mirrors qwen3next's oneDNN MoE path).
    sycl::half *ws_moe_logits_ = nullptr;  // [T,E]   router logits (device gemm)
    sycl::half *ws_moe_xp_     = nullptr;  // [T*K,H] expert-sorted gathered input
    sycl::half *ws_moe_h_      = nullptr;  // [T*K,E_ffn] gate (then activated)
    sycl::half *ws_moe_h2_     = nullptr;  // [T*K,E_ffn] up
    sycl::half *ws_moe_out_    = nullptr;  // [T*K,H] per-packed-row down output
    sycl::half *ws_moe_y_      = nullptr;  // [T,H]   reduced MoE output
    sycl::half *ws_moe_wpk_    = nullptr;  // [T*K]   router weight per packed row
    uint32_t   *ws_moe_offsets_= nullptr;  // [E+1]
    int32_t    *ws_moe_sorted_ = nullptr;  // [T*K]   token id per packed row
    int32_t    *ws_moe_tk2pk_  = nullptr;  // [T*K]   (t*K+kslot) -> packed row
    sycl::half *ws_moe_btf16_  = nullptr;  // [H*E_ffn] per-expert dequant→fp16 Bt (reused)
    void       *ws_q8_x_  = nullptr;       // [hidden/32]   block_q8_1x — decode act (gate/up)
    void       *ws_q8_h_  = nullptr;       // [E_ffn/32]    block_q8_1x — decode act (down)
    uint32_t    ws_T_moe_ = 0;             // capacity of the *_moe buffers keyed on T*K

    std::vector<float>       host_logits_;        // [T*E] router logits (host copy)
    std::vector<std::vector<std::pair<uint32_t,float>>> host_routes_;  // [T] per-token (expert,weight)
    MoePacking pk_;                                // reused host packing

    // FA-2 decode (T==1) attention partials scratch
    float*   ws_attn_partials_     = nullptr;
    uint32_t ws_attn_partials_ctx_ = 0;

    void free_all();
};

}  // namespace ie
