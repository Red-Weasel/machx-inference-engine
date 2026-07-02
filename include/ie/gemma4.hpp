// include/ie/gemma4.hpp — Google Gemma 4 (`gemma4`): dense 31B + MoE 26B-A4B.
//
// The defining trait is PER-LAYER head geometry: sliding-window layers use
// head_dim 256 + more KV heads (+ a V projection); global/full-attention layers
// use head_dim 512 + fewer KV heads and REUSE the K projection as V. On top of
// that: plain-w RMSNorm (no 1+w), attention softmax scale 1.0 (QK-norm carries
// magnitude; we pre-scale Q by sqrt(head_dim)), partial RoPE on global layers,
// sandwich norms, GeGLU FFN, a dual-path MoE (shared dense FFN + routed experts)
// per layer, and final logit soft-capping (30). See the plan/spec:
// docs/superpowers/plans/2026-06-13-gemma4-arch.md and project_gemma4_arch.md.
//
// Additive — new files only. Forward uses an UNFUSED per-expert MoE path for
// correctness first (host routing + device expert GEMVs), like qwen3moe v1.
#pragma once

#include "ie/allocator.hpp"
#include "ie/dense_transformer.hpp"   // DenseQuantPtr, dense:: upload helpers
#include "ie/gguf.hpp"
#include "ie/kv_cache.hpp"
#include "ie/model_config.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace ie {

class Gemma4Model {
public:
    Gemma4Model() = default;
    ~Gemma4Model();
    Gemma4Model(const Gemma4Model&) = delete;
    Gemma4Model& operator=(const Gemma4Model&) = delete;

    std::string load(DeviceAllocator& alloc, const GgufReader& g, const GemmaConfig& cfg);
    const GemmaConfig& config() const noexcept { return cfg_; }

    // Single-GPU forward (26B/31B Q4_0 fit one 32 GB B70). Sliding-window layers
    // are served with full causal attention — exact for T <= sliding_window.
    // hidden_pre_norm (optional, [T*H] device): when non-null, the residual stream
    // BEFORE output_norm is copied here — the [T,hidden] tensor the gemma4-assistant
    // MTP draft head consumes as its `inp_h` (the EAGLE self-spec hidden handoff).
    // all_logits (optional, [T*vocab] device): when non-null, the per-position
    // softcapped logits for ALL T rows (the MTP spec-decode VERIFY needs argmax
    // at every draft position, not just the last). out_logits still gets row T-1.
    sycl::event forward(sycl::queue& q, const int32_t* input_ids, uint32_t T,
                        uint32_t start_pos, KvCache& kv, sycl::half* out_logits,
                        sycl::half* hidden_pre_norm = nullptr,
                        sycl::half* all_logits = nullptr);

    // ---- gemma4-assistant MTP draft-head support ----
    // The draft head reads (never writes) the TARGET's KV cache at its last two
    // layers (SWA head layers → target L(n-2), global → target L(n-1)) and embeds
    // tokens with the target's tok_embd. These expose exactly what the head needs.
    const sycl::half* kcache(uint32_t L) const { return kcache_[L]; }
    const sycl::half* vcache(uint32_t L) const { return vcache_[L]; }
    uint32_t kv_ctx()            const noexcept { return kv_ctx_; }
    uint32_t n_layers()          const noexcept { return cfg_.n_layers; }
    uint32_t layer_head_dim(uint32_t L) const { return layers_[L].head_dim; }
    uint32_t layer_n_kv(uint32_t L)     const { return layers_[L].n_kv; }
    const void* token_embd()     const noexcept { return token_embd_; }
    DType token_embd_dtype()     const noexcept { return token_embd_dtype_; }
    uint32_t hidden()            const noexcept { return cfg_.hidden; }

    std::string ensure_workspace(uint32_t max_T);
    std::string ensure_attn_partials(uint32_t max_ctx);
    // Gemma self-manages per-layer KV (variable head geometry → the shared
    // uniform KvCache can't express it). Allocate before forward.
    std::string ensure_kv(uint32_t max_ctx);

    // ---- MTP self-speculative decode (gemma4-assistant draft head) ----
    // Gemma 4 ships its official ~0.9B MTP draft head as a SEPARATE GGUF
    // (mtp-gemma-4-*-Q8_0.gguf). Loaded on demand (--spec) → Q8_0 SoA int-dot.
    // The head's Q-only attention SHARES this model's KV (read-only, last 2 layers).
    std::string load_mtp_head(const GgufReader& head_g, uint32_t max_ctx);
    bool mtp_loaded() const noexcept { return mtp_.loaded; }
    // GREEDY MTP self-spec decode (strictly token-lossless vs plain greedy).
    // Caller must have prefilled `start_pos` tokens and passes the target argmax
    // `tn` @ the last prompt position + `d_hlast` = the pre-output_norm hidden
    // there (backbone-wide). Drafts K, verifies forward(T=K), accepts the longest
    // argmax-matching prefix; calls emit() per committed token IN ORDER (emit
    // returns false → stop). Stops after max_new committed tokens.
    std::string spec_generate(sycl::queue& q, sycl::half* d_hlast, int32_t tn,
                              uint32_t start_pos, uint32_t max_new, uint32_t K,
                              const std::function<bool(int32_t)>& emit);

private:
    struct Layer {
        bool     is_swa   = true;
        uint32_t head_dim = 0;     // 256 (swa) / 512 (global)
        uint32_t n_kv     = 0;     // KV heads this layer
        uint32_t n_rot    = 0;     // rotated dims
        float    rope_theta = 10000.f;

        // norms (all plain-w, loaded as-is; F32)
        float* attn_norm = nullptr;
        float* attn_q_norm = nullptr;     // [head_dim]
        float* attn_k_norm = nullptr;     // [head_dim]
        float* post_attn_norm = nullptr;
        float* ffn_norm = nullptr;        // shared-FFN pre-norm
        float* post_ffw_norm_1 = nullptr; // shared-FFN post-norm (MoE layers)
        float* pre_ffw_norm_2 = nullptr;  // expert pre-norm (MoE layers)
        float* post_ffw_norm_2 = nullptr; // expert post-norm (MoE layers)
        float* post_ffw_norm = nullptr;   // outer post-FFN norm

        DenseQuantPtr attn_q, attn_k, attn_v, attn_output;  // attn_v null on global layers (V=K)
        bool has_v = true;

        // shared dense GeGLU FFN (present every layer)
        DenseQuantPtr ffn_gate, ffn_up, ffn_down;

        // MoE (when is_moe): router + fused gate_up experts + down experts + scales.
        // Router runs on host (correctness-first), so its scales are host-side.
        bool                is_moe = false;
        std::vector<float>  router_w;            // F32 [n_experts, hidden] (row e = e*hidden)
        std::vector<float>  router_in_scale_h;   // ffn_gate_inp.scale [hidden] (per-channel)
        sycl::half*         router_w_dev = nullptr;        // F16 [n_experts, hidden] (GPU router gemv_q_T)
        float*              router_in_scale_dev = nullptr; // F32 [hidden] (= router_in_scale_h, or all-ones)
        std::vector<float>  down_scale_h;        // ffn_down_exps.scale [n_experts] (per-expert)
        void* gate_up_exps = nullptr; uint64_t gate_up_stride = 0; // Q4_0 [hidden, 2*expert_ffn] per expert
        void* down_exps    = nullptr; uint64_t down_stride    = 0; // Q4_0 [expert_ffn, hidden] per expert
        // SoA-Q4_0 expert banks (IE_GEMMA4_MOE_SOA) — stored INSTEAD of AoS above
        // (gate_up_exps/down_exps null on the SoA path; no memory doubling).
        uint8_t* gu_qs = nullptr; uint16_t* gu_d = nullptr; uint64_t gu_qs_stride = 0, gu_d_stride = 0;
        uint8_t* dn_qs = nullptr; uint16_t* dn_d = nullptr; uint64_t dn_qs_stride = 0, dn_d_stride = 0;
        float               layer_out_scale_val = 1.0f;  // [1] per-layer output scalar (host)
    };

    DeviceAllocator* alloc_ = nullptr;
    GemmaConfig cfg_;
    std::vector<Layer> layers_;
    std::vector<void*> owned_;

    void*  token_embd_ = nullptr;  DType token_embd_dtype_ = DType::kCount;  // tied lm_head
    // SoA-Q6 streams for the tied lm_head (fast decode GEMV via gemv_q6_soa_q8).
    // Null unless token_embd is Q6_K and the SoA path is on. AoS token_embd_ kept
    // for the embedding lookup.
    uint8_t *lmh_q6_lo_ = nullptr, *lmh_q6_hi_ = nullptr;
    int8_t  *lmh_q6_sc_ = nullptr;
    uint16_t *lmh_q6_d_ = nullptr;
    float* output_norm_ = nullptr;
    float* rope_freqs_ = nullptr;   // [head_dim_global/2] proportional-rope freq factors (global layers)

    uint32_t max_head_dim_ = 0;     // max over layers (512) for workspace sizing
    uint32_t max_n_kv_ = 0;

    // workspace (sized for the largest per-layer geometry)
    uint32_t ws_T_ = 0;
    sycl::half *ws_x_ = nullptr, *ws_xn_ = nullptr, *ws_q_ = nullptr, *ws_k_ = nullptr,
               *ws_v_ = nullptr, *ws_attn_out_ = nullptr, *ws_block_ = nullptr,
               *ws_attn_out2_ = nullptr;
    sycl::half *ws_gate_ = nullptr, *ws_up_ = nullptr, *ws_h_ = nullptr,
               *ws_moe_y_ = nullptr, *ws_shared_y_ = nullptr;
    sycl::half *ws_rin_ = nullptr, *ws_router_logits_ = nullptr;  // GPU-router scratch ([T,H], [T,E])
    // per-(token,expert) MoE scratch (single token at a time)
    sycl::half *ws_e_gateup_ = nullptr, *ws_e_h_ = nullptr, *ws_e_out_ = nullptr;
    // batched fused-MoE scratch (IE_GEMMA4_FUSED_MOE) — expert-sorted packing
    // over T*K routed rows: gather → per-expert batched gemm_q4_0 → reduce.
    sycl::half *ws_xp_packed_ = nullptr, *ws_gu_packed_ = nullptr,
               *ws_h_packed_ = nullptr, *ws_out_packed_ = nullptr,
               *ws_weights_packed_ = nullptr;
    int32_t *ws_sorted_idx_ = nullptr, *ws_tk_to_packed_ = nullptr;
    // int-dot W4A8 path (default within fused; opt-out IE_GEMMA4_NO_Q8):
    // q8_1s-quantized activation streams + device expert_offsets.
    void *ws_xq8_ = nullptr, *ws_hq8_ = nullptr;
    uint32_t *ws_expert_offsets_ = nullptr;
    // int-dot W4A8 dense/attn projections (IE_GEMMA4_INTDOT_PROJ): a per-forward
    // q8_1s staging stream for the projection input, consumed by the split-K
    // gemm_q4_0_q8 (handles any K, incl. the large-K o-proj).
    void *ws_projq8_ = nullptr;
    // weight-stationary fp16 GEMM scratch (IE_GEMMA4_ONEDNN): per-projection
    // dequant Q4_0→fp16 Bt[K,N] (grows to the largest projection K*N seen).
    sycl::half *ws_btf16_ = nullptr;
    uint64_t ws_btf16_cap_ = 0;
    // block_q8_1x activation stream for the SoA-Q4_0 decode GEMV (gemv_q4_0_soa_q8).
    void *ws_q8_ = nullptr;
    // Batched-verify staging: T rows of block_q8_1x for gemv_q4_0_soa_q8_batched
    // (the MTP spec-decode VERIFY forward, T=K≤16 — reads each weight column ONCE
    // and dots vs all T activation rows, the decode-class kernel the prefill path
    // can't match for tiny T). Sized for max_T rows of the widest projection K.
    void *ws_q8b_ = nullptr;
    sycl::half *ws_attn_partials_ = nullptr; uint32_t ws_attn_partials_ctx_ = 0;
    float* ones_hd_ = nullptr;      // all-ones [max_head_dim] for the weightless V-norm
    int32_t* ws_positions_ = nullptr;
    uint32_t ws_positions_cap_ = 0;

    // self-managed per-layer KV cache (sized by each layer's n_kv*head_dim)
    std::vector<sycl::half*> kcache_, vcache_;
    uint32_t kv_ctx_ = 0;

    // ---- MTP draft head (gemma4-assistant), loaded only with --spec ----
    // SoA-Q8_0 weight (int8 qs col-contiguous + fp16/32-block scales) for the
    // int-dot W8A8 decode GEMV. All buffers tracked in owned_ (freed in dtor).
    struct MtpW { int8_t* qs = nullptr; uint16_t* d = nullptr; uint32_t K = 0, N = 0; };
    struct MtpLayer {
        bool     is_swa = true;
        uint32_t head_dim = 0, n_kv = 0, n_rot = 0;
        float    theta = 1e4f, out_scale = 1.0f;
        float   *attn_norm=nullptr, *attn_q_norm=nullptr, *post_attn_norm=nullptr,
                *ffn_norm=nullptr, *post_ffw_norm=nullptr, *rope_freqs=nullptr;
        MtpW wq, wo, fg, fu, fd;
    };
    struct Mtp {
        bool loaded = false;
        uint32_t H=0, BB=0, F=0, vocab=0, n_q=0, max_Nq=0, max_K=0;
        float eps = 1e-6f;
        MtpW pre_proj, post_proj, tok_embd;   // tok_embd = head output/logits proj
        float* output_norm = nullptr;
        std::vector<MtpLayer> L;
        // T=1 draft scratch
        sycl::half *xemb=nullptr, *xh=nullptr, *inph=nullptr, *cur=nullptr, *curn=nullptr,
                   *qbuf=nullptr, *ao=nullptr, *blk=nullptr, *attnout=nullptr,
                   *fgb=nullptr, *fub=nullptr, *fhb=nullptr, *hnext=nullptr, *logits=nullptr;
        void* actq8 = nullptr;          // block_q8_1x activation stream (max head K)
        int32_t *pos=nullptr, *tokid=nullptr;
    };
    Mtp mtp_;
    // run the head over (inp_h_dev[BB], tok) at draft position p (reads target
    // KV[0,p)); leaves draft logits in mtp_.logits, next inp_h in mtp_.hnext.
    void mtp_run(sycl::queue& q, const sycl::half* inp_h, int32_t tok, uint32_t p);

    // Weight-stationary oneDNN fp16 MoE GEMM (prefill, large-M). For each expert
    // e with packed row-slice [expert_offsets[e], expert_offsets[e+1]): dequant
    // its SoA-Q4_0 weight [Kd,Nd] → fp16 Bt (ws_btf16_, BIT-IDENTICAL to AoS),
    // then gemm_fp16_onednn(in_e[n_e,Kd], Bt[Kd,Nd]) → out_e[n_e,Nd]. Output is
    // UNWEIGHTED (moe_prefill_reduce applies the routing weight). The same lever
    // that put Coder/qwen3moe MoE prefill ahead of llama (oneDNN ~1.65× our
    // hand-rolled gemm_fp16 + fp16-direct, no q8 quantize). Caller must have
    // grown ws_btf16_ to >= Kd*Nd. ws_btf16_ is reused per expert (in-order
    // queue serializes WAR/RAW). Gemma fuses gate+up so the gate_up GEMM is one
    // [H,2*EF] call; down is [EF,H].
    void moe_onednn_proj(sycl::queue& q, const sycl::half* in_packed,
                         const uint8_t* qs, const uint16_t* d,
                         uint64_t qs_stride, uint64_t d_stride,
                         sycl::half* out_packed, const uint32_t* expert_offsets,
                         uint32_t Kd, uint32_t Nd);
};

}  // namespace ie
