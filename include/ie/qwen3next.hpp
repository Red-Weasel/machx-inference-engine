// include/ie/qwen3next.hpp — Qwen3-Next-80B multi-GPU LAYER-SPLIT loader.
// ADDITIVE: composes the validated DeltaNet/attn load (qwen35_dense) + MoE
// expert/router load (qwen3moe) onto a DeviceFleet via the dense_split LayerPlan.
// Never touches DenseModelSplit / crown / qwen35_dense / qwen3moe. Step 1 = load
// only (no forward). Design: docs/superpowers/specs/2026-06-12-qwen3next-hybrid-
// split-architecture-review.md.
#pragma once

#include "ie/allocator.hpp"          // DeviceFleet, DeviceAllocator
#include "ie/dense_split.hpp"        // LayerPlan (reused as-is)
#include "ie/dense_transformer.hpp"  // DenseQuantPtr, dense::upload*, upload_weight_auto
#include "ie/deltanet_state.hpp"     // DeltaNetState
#include "ie/gguf.hpp"
#include "ie/kv_cache.hpp"
#include "ie/model_config.hpp"       // Qwen3NextConfig

#include <sycl/sycl.hpp>
#include <cstdint>
#include <string>
#include <vector>

namespace ie {

class Qwen3NextModel {
public:
    Qwen3NextModel() = default;
    ~Qwen3NextModel();
    Qwen3NextModel(const Qwen3NextModel&) = delete;
    Qwen3NextModel& operator=(const Qwen3NextModel&) = delete;

    // Load every weight onto plan.dev_of_layer[L]'s card; embed → plan.embed_dev,
    // lm_head → plan.head_dev. Per-card hybrid caches sized to that card's layer
    // mix. `fleet` and `g` must outlive this object. "" on success, else error.
    // max_ctx sizes each card's KvCache (default 2048 = the load-smoke gate;
    // ie::Engine passes opts.max_ctx for real serving).
    std::string load(DeviceFleet& fleet, const LayerPlan& plan,
                     const GgufReader& g, const Qwen3NextConfig& cfg,
                     uint32_t max_ctx = 2048);

    const Qwen3NextConfig& config() const noexcept { return cfg_; }
    std::vector<uint64_t> device_bytes() const { return dev_bytes_; }

    // Prefix-cache support (FleetPrefixCache, mirroring the crown PrefixCache).
    // The per-card hybrid state (kv_[dev]/dn_[dev]) is the SAME KvCache/DeltaNetState
    // the crown cache snapshots, so snapshot/restore is "the crown design once per
    // card". Non-const cache accessors so the engine can restore INTO the live state
    // (copy_prefix_from/copy_from write the destination); reads use the const-ready
    // probes. Additive; no behavior change.
    uint32_t        n_devices()              const noexcept { return n_dev_; }
    DeviceFleet*    fleet()                  const noexcept { return fleet_; }
    KvCache&        kv_cache(uint32_t dev)                  { return kv_[dev]; }
    DeltaNetState&  dn_state(uint32_t dev)                  { return dn_[dev]; }
    bool            dev_has_kv(uint32_t dev) const          { return kv_[dev].ready(); }
    bool            dev_has_dn(uint32_t dev) const          { return dn_[dev].ready(); }

    // F1 FORWARD SCAFFOLD — runs the split pipeline end-to-end with the per-layer
    // body STUBBED (attn + MoE contributions zeroed → x passes through unchanged).
    // Builds + runs without crashing; output logits are garbage (expected for F1).
    // Mirrors DenseModelSplit::forward: embedding on embed_dev → device-by-device
    // layer loop with ONE residual copy at each boundary → output_norm + lm_head on
    // head_dev → last token's logits to `out_logits_host` (host fp16, size vocab).
    std::string forward(const int32_t* input_ids, uint32_t T, uint32_t start_pos,
                        bool reset_kv, sycl::half* out_logits_host);

private:
    struct LayerW {
        bool is_linear = false;
        float* attn_norm = nullptr;          // F32
        float* post_attn_norm = nullptr;     // F32 (pre-FFN norm; no separate ffn_norm)
        // DeltaNet (is_linear)
        DenseQuantPtr attn_qkv;              // Q5_K [H, conv_ch=8192]
        DenseQuantPtr attn_gate;             // Q4_K [H, ssm_inner=4096] z-gate
        DenseQuantPtr ssm_ba;                // Q8_0 [H, 2*n_v=64] fused [beta|alpha]
        DenseQuantPtr ssm_out;               // Q8_0 [ssm_inner, H]
        float* ssm_a = nullptr;              // F32 [n_v=32]
        float* ssm_dt_bias = nullptr;        // F32 [n_v=32]
        float* ssm_conv1d = nullptr; sycl::half* ssm_conv1d_fp16 = nullptr;  // [4, 8192]
        float* ssm_norm  = nullptr; sycl::half* ssm_norm_fp16  = nullptr;    // [128]
        // full-attn (!is_linear)
        DenseQuantPtr attn_q;                // Q4_K [H, 8192] joint Q|gate
        DenseQuantPtr attn_k, attn_v;        // Q8_0 [H, 512]
        DenseQuantPtr attn_output;           // Q6_K [4096, H]
        float* attn_q_norm = nullptr; float* attn_k_norm = nullptr;          // F32 [256]
        // MoE (every layer)
        std::vector<float> router_w; sycl::half* router_w_dev = nullptr;     // F32[E,H] + F16[H,E]
        void* gate_exps = nullptr; DType gate_dt = DType::kCount; uint64_t gate_stride = 0;
        void* up_exps   = nullptr; DType up_dt   = DType::kCount; uint64_t up_stride   = 0;
        void* down_exps = nullptr; DType down_dt = DType::kCount; uint64_t down_stride = 0;
        // EXL3 expert banks (when *_dt==kEXL3): the trellis is *_exps above (per-expert byte stride
        // *_stride); the F16 suh/svh side-banks + bit-width ride here. Per-expert in the forward:
        // suh + e*K_elems, svh + e*N_elems (gate/up K=H,N=EF; down K=EF,N=H). Null on the Q4_K path.
        void* gate_suh = nullptr; void* gate_svh = nullptr; uint32_t gate_bits = 0;
        void* up_suh   = nullptr; void* up_svh   = nullptr; uint32_t up_bits   = 0;
        void* down_suh = nullptr; void* down_svh = nullptr; uint32_t down_bits = 0;
        // shared expert
        float* ffn_gate_inp_shexp = nullptr;             // F32 [H] sigmoid gate
        DenseQuantPtr gate_shexp, up_shexp, down_shexp;  // Q8_0 typ.
    };

    // Per-card forward scratch. Pipeline buffers (F1) + gated full-attn scratch
    // (F2). DeltaNet/MoE buffers arrive in F3-F4. Mirrors (trimmed)
    // DenseModelSplit::Workspace. These buffers are tracked HERE (not in owned_)
    // so ensure_ws can free+re-alloc them on a T-grow without leaking; free_all
    // frees them via free_ws.
    struct Workspace {
        uint32_t T = 0;
        sycl::half *x = nullptr, *x_normed = nullptr, *attn_block = nullptr;
        int32_t* positions = nullptr;
        // F2 gated full-attn scratch (qwen3next dims: H=2048, n_q=16, n_kv=2,
        // head_dim=256; Q|gate=8192; K/V=2·256=512; attn_out=n_q·256=4096).
        sycl::half *qg = nullptr;        // [T, 8192]  joint Q|gate projection
        sycl::half *q = nullptr;         // [T, 4096]  Q after split (n_q·head_dim)
        sycl::half *gate = nullptr;      // [T, 4096]  σ-gate after split
        sycl::half *k = nullptr;         // [T, 512]   K projection
        sycl::half *v = nullptr;         // [T, 512]   V projection
        sycl::half *attn_out = nullptr;  // [T, 4096]  SDPA output (pre out_proj)
        float      *attn_partials = nullptr;  // FA-2 decode partials (T==1)
        uint32_t    partials_ctx = 0;         // ctx the partials buffer is sized for
        // F3 gated-DeltaNet scratch (qwen3next deltas vs the 27B: n_v_heads=32,
        // ssm_inner=4096, conv_ch=8192, fused ssm_ba [T,64]=β|α, repeat 16→32).
        //   conv_ch = ssm_inner + 2·n_k·state = 4096 + 2·16·128 = 8192
        //   kw      = n_k·state = 16·128 = 2048  (q/k pre-repeat width)
        //   rep     = n_v/n_k = 32/16 = 2  → q/k repeated width = 4096 = ssm_inner
        sycl::half *dn_qkv = nullptr;    // [T, 8192]  attn_qkv proj (conv input)
        sycl::half *dn_conv = nullptr;   // [T, 8192]  post-conv1d (silu fused)
        float      *dn_qpre = nullptr;   // [T, 2048]  q pre-repeat (fp32, l2-normed)
        float      *dn_kpre = nullptr;   // [T, 2048]  k pre-repeat (fp32, l2-normed)
        float      *dn_vpre = nullptr;   // [T, 4096]  v (fp32, = ssm_inner)
        float      *dn_qrep = nullptr;   // [T, 4096]  q repeated 16→32
        float      *dn_krep = nullptr;   // [T, 4096]  k repeated 16→32
        sycl::half *dn_ba = nullptr;     // [T, 64]    ssm_ba proj = [β(32)|α(32)]
        sycl::half *dn_alpha = nullptr;  // [T, 32]    α slice = ba cols [32,64)
        sycl::half *dn_beta_h = nullptr; // [T, 32]    β slice = ba cols [0,32)
        float      *dn_g = nullptr;      // [T, 32]    g from compute_g_beta_h16
        float      *dn_beta = nullptr;   // [T, 32]    β (recurrence) from compute_g_beta_h16
        float      *dn_out = nullptr;    // [T, 4096]  recurrence output (fp32)
        sycl::half *dn_z = nullptr;      // [T, 4096]  z-gate (attn_gate proj, = ssm_inner)
        sycl::half *dn_gn = nullptr;     // [T, 4096]  gated_rms_norm output (fp16)
        // F4 MoE + shared-expert scratch (qwen3next dims: H=2048, E=512, K=10,
        // E_ffn=512). The fused-prefill chain mirrors qwen3moe::moe_ffn_fused_prefill
        // (gather → gate_up_silu_q8 → down_q6k_q8_gen[E_ffn=512] → reduce); the
        // T==1 decode chain mirrors qwen3moe::moe_ffn_decode (2 launches). Shared
        // expert (every layer) mirrors qwen36.cpp:1516-1572 (σ-gate → swiglu →
        // down → scaled_add). Buffers tracked HERE (not owned_) → freed by free_ws.
        sycl::half *moe_logits = nullptr;    // [T, E]    router logits (GPU gemm)
        sycl::half *moe_xp     = nullptr;    // [T*K, H]  expert-sorted gathered input
        sycl::half *moe_h      = nullptr;    // [T*K, E_ffn]  gate*up*silu per packed row
        sycl::half *moe_out    = nullptr;    // [T*K, H]  per-packed-row down output
        sycl::half *moe_y      = nullptr;    // [T, H]    reduced MoE output (per token)
        sycl::half *moe_wpk    = nullptr;    // [T*K]     router weight per packed row
        void       *moe_xp_q8  = nullptr;    // [T*K]·(H/32)     block_q8_1s of moe_xp
        void       *moe_h_q8   = nullptr;    // [T*K]·(E_ffn/32) block_q8_1s of moe_h
        uint32_t   *moe_offsets = nullptr;   // [E+1]     expert prefix offsets
        int32_t    *moe_sorted_idx = nullptr;// [T*K]     token id per packed row
        int32_t    *moe_tk_to_packed = nullptr; // [T*K]  (t*K+kslot) -> packed row
        int32_t    *moe_topk_idx = nullptr;  // [K]       decode top-k expert ids
        sycl::half *moe_topk_w   = nullptr;  // [K]       decode top-k weights
        // shared-expert scratch
        sycl::half *sh_gate = nullptr;       // [T, E_ffn]  shexp gate proj
        sycl::half *sh_up   = nullptr;       // [T, E_ffn]  shexp up proj
        sycl::half *sh_h    = nullptr;       // [T, E_ffn]  swiglu(gate,up)
        sycl::half *sh_eo   = nullptr;       // [T, H]      shexp down proj
        sycl::half *sh_g    = nullptr;       // [T]         per-token σ-gate scalar
        // EXL3-only scratch: fused in_proj_qkvz projection (sliced → dn_qkv|dn_z) + per-expert
        // MoE working buffers (gemv_exl3 has no batched MoE kernel yet → one expert at a time).
        sycl::half *dn_qkvz = nullptr;       // [T, 12288]  EXL3 fused qkvz (conv_ch + SI)
        sycl::half *ex_gate = nullptr;       // [E_ffn]     one expert gate proj
        sycl::half *ex_up   = nullptr;       // [E_ffn]     one expert up proj
        sycl::half *ex_h    = nullptr;       // [E_ffn]     swiglu(gate,up)
        sycl::half *ex_down = nullptr;       // [H]         one expert down proj
        sycl::half *ex_w    = nullptr;       // [T*K]       all routing weights (uploaded 1×/layer)
        // EXL3 fused-batched MoE (perf pass): row-batched scratch (R = T*K active rows).
        sycl::half *moe_h2     = nullptr;    // [T*K, E_ffn]  up-proj output (batched, parallel to moe_h)
        int32_t    *moe_rowtok = nullptr;    // [T*K]         token id per active row (x-gather index)
        // oneDNN MoE-prefill (Step 2, large-M long-ctx lever): per-expert fp16 Bt
        // dequant target, REUSED across experts (gate→moe_h, up→moe_h2, down→moe_out).
        sycl::half *moe_btf16  = nullptr;    // [H*E_ffn]     per-expert dequant→fp16 Bt
    };

    DeviceFleet*  fleet_ = nullptr;
    LayerPlan     plan_;
    Qwen3NextConfig cfg_;
    uint32_t      n_dev_ = 0;
    std::vector<Workspace> ws_;              // [dev]

    std::vector<LayerW> layers_;             // global index L; ptrs live on dev_of_layer[L]
    void*  token_embd_ = nullptr;  DType token_embd_dtype_ = DType::kCount;
    float* output_norm_ = nullptr;
    void*  output_ = nullptr;      DType output_dtype_ = DType::kCount;
    void*  output_suh_ = nullptr;  void* output_svh_ = nullptr;  uint32_t output_bits_ = 0;  // kEXL3 lm_head

    std::vector<std::vector<void*>> owned_;  // [dev] device ptrs to free
    std::vector<DeltaNetState>      dn_;      // [dev], sized to dev's linear-layer count
    std::vector<KvCache>            kv_;      // [dev], sized to dev's attn-layer count
    std::vector<uint32_t>           dn_local_;// [global L] local DeltaNet idx (linear layers)
    std::vector<uint32_t>           kv_local_;// [global L] local KV idx (attn layers)
    std::vector<uint64_t>           dev_bytes_;

    std::string ensure_ws(uint32_t dev, uint32_t max_T);
    void free_ws(uint32_t dev);   // free a card's Workspace buffers (not in owned_)
    void free_all();
};

}  // namespace ie
