// include/ie/qwen35_split.hpp — Qwen3.6-27B (`kQwen35Dense`) multi-GPU LAYER-SPLIT.
//
// ADDITIVE companion to the validated single-GPU `Qwen35DenseModel` (qwen35_dense.
// {hpp,cpp}, PPL 5.34) — that file is NEVER edited. This mirrors the proven 80B
// fleet scaffold (`Qwen3NextModel`, qwen3next.{hpp,cpp}): the SAME hybrid family
// (gated-DeltaNet linear layers + gated full-attn layers) MINUS the MoE — the FFN
// is a plain dense SwiGLU. Per layer L → plan.dev_of_layer[L]'s card; embed →
// plan.embed_dev, lm_head → plan.head_dev. Device-by-device forward with ONE
// residual hand-off per card boundary, host-logits bounce (mirroring the TP /
// next_ engine paths). `--gpus 1` keeps using `Qwen35DenseModel`; `--gpus N` uses
// this. Per-layer math is lifted VERBATIM from qwen35_dense.cpp's forward — only
// the orchestration becomes per-card.
//
// 27B DeltaNet conventions (vs the 80B's): SEPARATE ssm_alpha/ssm_beta projections
// (N-padded to 64 → batched gemm + extract_cols), TILE repeat 16→48 (interleave
// =false), n_v_heads=48, conv_channels=10240, ssm_inner=6144. NOT the 80B's fused
// ssm_ba / interleave repeat / MoE.
//
// Design: docs/superpowers/specs/2026-06-20-qwen35-27b-multigpu-split.md.
#pragma once

#include "ie/allocator.hpp"          // DeviceFleet, DeviceAllocator
#include "ie/dense_split.hpp"        // LayerPlan (reused as-is)
#include "ie/dense_transformer.hpp"  // DenseQuantPtr, dense::upload*
#include "ie/deltanet_state.hpp"     // DeltaNetState
#include "ie/gguf.hpp"
#include "ie/kv_cache.hpp"
#include "ie/model_config.hpp"       // Qwen35Config

#include <sycl/sycl.hpp>
#include <cstdint>
#include <string>
#include <vector>

namespace ie {

class Qwen35SplitModel {
public:
    Qwen35SplitModel() = default;
    ~Qwen35SplitModel();
    Qwen35SplitModel(const Qwen35SplitModel&) = delete;
    Qwen35SplitModel& operator=(const Qwen35SplitModel&) = delete;

    // Load every weight onto plan.dev_of_layer[L]'s card; embed → plan.embed_dev,
    // lm_head → plan.head_dev. Per-card hybrid caches sized to that card's layer
    // mix. `fleet` and `g` must outlive this object. "" on success, else error.
    // max_ctx sizes each card's KvCache (Engine passes opts.max_ctx).
    std::string load(DeviceFleet& fleet, const LayerPlan& plan,
                     const GgufReader& g, const Qwen35Config& cfg,
                     uint32_t max_ctx = 2048);

    const Qwen35Config& config() const noexcept { return cfg_; }
    std::vector<uint64_t> device_bytes() const { return dev_bytes_; }
    uint32_t n_devices() const noexcept { return n_dev_; }

    // Device-by-device forward. Host ids in, last token's logits to host fp16
    // (size vocab). reset_kv clears per-card hybrid state for a fresh sequence.
    std::string forward(const int32_t* input_ids, uint32_t T, uint32_t start_pos,
                        bool reset_kv, sycl::half* out_logits_host);

private:
    // Phase 2 weight handle. A Q8_0 GGUF weight is stored PACKED as SoA int8
    // (no F16 doubling → ~13.5 GB/card): q8_qs[n*K+k] int8 column-contiguous +
    // q8_d[n*(K/32)+b] fp16 per-32-block scale (de-interleaved from on-disk AoS
    // block_q8_0 — bit-exact, no requant). Consumed by gemv_q8_0_soa_q8 on decode
    // and a SoA→fp16 dequant + gemm_fp16 on prefill. Non-Q8_0 tensors (Q4_K/Q6_K
    // packed, Q5_K→F16) fall back to `fp` (q8_qs == nullptr). See sgemv().
    struct SplitW {
        int8_t*   q8_qs = nullptr;   // [N*K] int8 (Q8_0-SoA)
        uint16_t* q8_d  = nullptr;   // [N*(K/32)] fp16 bits
        uint32_t  K = 0, N = 0;
        DenseQuantPtr fp;            // fallback when q8_qs == nullptr
    };

    // Per-layer weights. EITHER linear (DeltaNet) or full-attn populated; the FFN
    // + the two norms are shared. Mirror Qwen35LayerWeights, sans single-GPU notes.
    struct LayerW {
        bool is_linear = false;
        float* attn_norm = nullptr;            // F32 [hidden]
        float* post_attn_norm = nullptr;       // F32 [hidden]
        SplitW ffn_gate, ffn_up, ffn_down;     // dense SwiGLU
        // full-attn
        SplitW attn_q;                         // joint Q|gate [5120,12288]
        SplitW attn_k, attn_v;                 // [5120,1024]
        SplitW attn_output;                    // [6144,5120]
        float* attn_q_norm = nullptr;          // F32 [head_dim 256]
        float* attn_k_norm = nullptr;          // F32 [head_dim 256]
        // linear (DeltaNet)
        SplitW attn_qkv;                       // fused q|k|v conv input [5120,10240]
        SplitW attn_gate;                      // z-gate input [5120,6144]
        float* ssm_a = nullptr;                // F32 [n_v 48] (A_log)
        DenseQuantPtr ssm_alpha;               // F32→fp16 N-padded proj [5120,64] (small, stays F16)
        DenseQuantPtr ssm_beta;                // F32→fp16 N-padded proj [5120,64]
        float* ssm_conv1d = nullptr; sycl::half* ssm_conv1d_fp16 = nullptr;  // [4,10240]
        float* ssm_dt_bias = nullptr;          // F32 [48]
        float* ssm_norm = nullptr; sycl::half* ssm_norm_fp16 = nullptr;      // [128]
        SplitW ssm_out;                        // [6144,5120]
    };

    // Per-card forward scratch (mirror Qwen35DenseModel's ws_* fields, in a struct
    // so ensure_ws can free+re-alloc on a T-grow). Tracked HERE, not in owned_.
    struct Workspace {
        uint32_t T = 0;
        sycl::half *x = nullptr, *x_normed = nullptr, *attn_block = nullptr;
        int32_t* positions = nullptr;
        // full-attn
        sycl::half *qg = nullptr;        // [T, N_qg 12288] joint Q|gate
        sycl::half *q = nullptr;         // [T, N_q 6144]
        sycl::half *gate = nullptr;      // [T, N_q] per-head σ-gate
        sycl::half *k = nullptr;         // [T, N_kv 1024]
        sycl::half *v = nullptr;         // [T, N_kv]
        sycl::half *attn_out = nullptr;  // [T, N_q]
        float      *attn_partials = nullptr;  // FA-2 decode partials (T==1)
        uint32_t    partials_ctx = 0;
        // DeltaNet (27B dims: n_v=48, conv_ch=10240, ssm_inner=6144, Vd=6144)
        sycl::half *dn_qkv = nullptr;    // [T, conv_ch 10240]
        sycl::half *dn_conv = nullptr;   // [T, conv_ch]
        sycl::half *dn_z = nullptr;      // [T, SI 6144] z-gate (attn_gate proj)
        float      *dn_qpre = nullptr;   // [T, Vd 6144] fp32 q predelta
        float      *dn_kpre = nullptr;   // [T, Vd] fp32 k predelta
        float      *dn_vpre = nullptr;   // [T, Vd] fp32 v
        float      *dn_g = nullptr;      // [T, Nv 48] fp32 g
        float      *dn_beta = nullptr;   // [T, Nv] fp32 β
        float      *dn_out = nullptr;    // [T, Vd] fp32 recurrence output
        float      *dn_qrep = nullptr;   // [T, Vd] fp32 q post-repeat (16→48 tile)
        float      *dn_krep = nullptr;   // [T, Vd] fp32 k post-repeat
        sycl::half *dn_alpha_h = nullptr;// [T, Nv] fp16 α compacted
        sycl::half *dn_beta_h = nullptr; // [T, Nv] fp16 β compacted
        sycl::half *dn_alpha64 = nullptr;// [T, Nvp 64] fp16 α proj (batched gemm out)
        sycl::half *dn_beta64 = nullptr; // [T, Nvp 64] fp16 β proj
        // FFN
        sycl::half *ffn_gate = nullptr;  // [T, F 17408]
        sycl::half *ffn_up = nullptr;    // [T, F]
        sycl::half *ffn_h = nullptr;     // [T, F]
    };

    DeviceFleet*  fleet_ = nullptr;
    LayerPlan     plan_;
    Qwen35Config  cfg_;
    uint32_t      n_dev_ = 0;
    std::vector<Workspace> ws_;               // [dev]

    std::vector<LayerW> layers_;              // global index L; ptrs on dev_of_layer[L]
    void*  token_embd_ = nullptr;  DType token_embd_dtype_ = DType::kCount;   // packed AoS (lookup)
    float* output_norm_ = nullptr;
    SplitW output_;                           // lm_head (Q8_0-SoA when packed)

    std::vector<std::vector<void*>> owned_;   // [dev] device ptrs to free
    std::vector<DeltaNetState>      dn_;       // [dev], sized to dev's linear-layer count
    std::vector<KvCache>            kv_;       // [dev], sized to dev's full-attn count
    std::vector<uint32_t>           dn_local_; // [global L] local DeltaNet idx
    std::vector<uint32_t>           kv_local_; // [global L] local KV idx
    std::vector<uint64_t>           dev_bytes_;
    // Phase-2 per-card scratch (NOT in Workspace — weight-sized / T-independent,
    // allocated once): int-dot activation (block_q8_1x, decode) + the SoA→fp16
    // prefill dequant target (grown to the largest weight, reused across projections).
    std::vector<void*>       act_q8_;          // [dev] block_q8_1x [Kmax/32]
    std::vector<sycl::half*> prefill_bt_;       // [dev] fp16 [K*N]max dequant scratch
    std::vector<uint64_t>    prefill_bt_cap_;   // [dev] element capacity

    // Q8_0-SoA aware GEMV: out[T,N] = A[T,K] @ W. Decode (T==1) int-dot; prefill
    // dequant-to-fp16 + gemm; non-Q8_0 → dense::gemv_q_T. Runs on dev's queue.
    sycl::event sgemv(uint32_t dev, const sycl::half* A, const SplitW& w,
                      sycl::half* out, uint32_t K, uint32_t N, uint32_t T);

    std::string ensure_ws(uint32_t dev, uint32_t max_T);
    void free_ws(uint32_t dev);
    void free_all();
};

}  // namespace ie
