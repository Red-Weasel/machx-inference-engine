// include/ie/qwen35_tp.hpp — Qwen3.6-27B (`kQwen35Dense`) TENSOR-PARALLEL decode.
//
// ADDITIVE: never touches Qwen35DenseModel (single-GPU), Qwen35SplitModel (layer-
// split, the correctness ORACLE), DenseModelTP, or the crown. Where the layer-
// split puts whole layers on one card and runs the cards serially (one bus active
// at a time), TP runs BOTH cards on the SAME token concurrently (Megatron column-
// then-row) so both memory buses work at once → ~1.4× decode on this no-P2P board.
//
// PHASE 0 (this build): FFN-slice TP ONLY. The dense SwiGLU FFN is split
// (gate/up column-parallel, down row-parallel, 1 all-reduce/layer); the gated
// full-attn and gated-DeltaNet blocks stay REPLICATED (full weights on both cards,
// each card runs the identical block on the replicated x → bit-identical residual,
// no all-reduce). This isolates the FFN-TP win and MEASURES the no-P2P all-reduce
// cost on this board with the simplest delta — the GO/NO-GO gate before the harder
// gated-attn head-shard (Phase 1) and the novel DeltaNet v-head shard (Phase 2).
//
// Reuse: the per-layer hybrid math is lifted VERBATIM from qwen35_split.cpp (which
// itself lifted it from the validated single-GPU qwen35_dense.cpp). The Q8_0-SoA
// pack + the sgemv (decode int-dot / prefill dequant+gemm) mirror Qwen35SplitModel.
// The all-reduce + per-card overlap mirror DenseModelTP. TP is NOT bit-exact vs the
// split (reduction order differs) → validated by cosine ≥0.999/layer + greedy match.
//
// Spec: docs/superpowers/specs/2026-06-20-qwen35-27b-hybrid-tensor-parallel.md
//       docs/superpowers/specs/2026-06-20-hybrid-tp-SCOPING.md
#pragma once

#include "ie/allocator.hpp"          // DeviceFleet, DeviceAllocator
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

class Qwen35TpModel {
public:
    Qwen35TpModel() = default;
    ~Qwen35TpModel();
    Qwen35TpModel(const Qwen35TpModel&) = delete;
    Qwen35TpModel& operator=(const Qwen35TpModel&) = delete;

    // Load: every card holds every layer. Phase 0 — attn/DeltaNet weights are
    // REPLICATED full on every card; only the FFN gate/up (column slice) and
    // ffn_down (row slice) are sharded. n_dev must divide the FFN superblock count
    // and (later phases) the head counts. `fleet`/`g` must outlive this object.
    std::string load(DeviceFleet& fleet, const GgufReader& g,
                     const Qwen35Config& cfg, uint32_t max_ctx = 2048);

    const Qwen35Config& config() const noexcept { return cfg_; }
    std::vector<uint64_t> device_bytes() const { return dev_bytes_; }
    uint32_t n_devices() const noexcept { return n_dev_; }

    // Concurrent forward (both cards, same token). Host ids in, last token's logits
    // to host fp16 (size vocab). reset_kv clears per-card hybrid state.
    std::string forward(const int32_t* input_ids, uint32_t T, uint32_t start_pos,
                        bool reset_kv, sycl::half* out_logits_host);

private:
    // Q8_0-SoA weight handle (PACKED, ~13.5 GB/card): q8_qs[n*K+k] int8 column-
    // contiguous + q8_d[n*(K/32)+b] fp16 per-32-block scale. Non-Q8_0 weights
    // (Q4_K/Q6_K/F16) fall back to `fp` (q8_qs == nullptr). Same as the split's SplitW.
    struct SplitW {
        int8_t*   q8_qs = nullptr;   // [N*K] int8 (Q8_0-SoA)
        uint16_t* q8_d  = nullptr;   // [N*(K/32)] fp16 bits
        uint32_t  K = 0, N = 0;
        DenseQuantPtr fp;            // fallback when q8_qs == nullptr
    };

    // Per-layer weights. Phase 0: attn/DeltaNet replicated FULL; FFN sharded.
    struct LayerW {
        bool is_linear = false;
        float* attn_norm = nullptr;            // F32 [hidden] (replicated)
        float* post_attn_norm = nullptr;       // F32 [hidden] (replicated)
        // FFN (SHARDED): gate/up column-parallel [fc,H]; down row-parallel [H,fc].
        SplitW ffn_gate, ffn_up, ffn_down;
        // gated full-attn (REPLICATED full in Phase 0)
        SplitW attn_q;                         // joint Q|gate [5120,12288]
        SplitW attn_k, attn_v;                 // [5120,1024]
        SplitW attn_output;                    // [6144,5120]
        float* attn_q_norm = nullptr;          // F32 [head_dim 256]
        float* attn_k_norm = nullptr;
        // gated-DeltaNet (REPLICATED full in Phase 0)
        SplitW attn_qkv;                       // fused q|k|v conv input [5120,10240]
        SplitW attn_gate;                      // z-gate input [5120,6144]
        float* ssm_a = nullptr;                // F32 [n_v 48]
        DenseQuantPtr ssm_alpha;               // F32→fp16 N-padded proj [5120,64]
        DenseQuantPtr ssm_beta;                // F32→fp16 N-padded proj [5120,64]
        float* ssm_conv1d = nullptr; sycl::half* ssm_conv1d_fp16 = nullptr;  // [4,10240]
        float* ssm_dt_bias = nullptr;          // F32 [48]
        float* ssm_norm = nullptr; sycl::half* ssm_norm_fp16 = nullptr;      // [128]
        SplitW ssm_out;                        // [6144,5120]
    };

    // Per-card forward scratch. attn/DeltaNet scratch is FULL (replicated work);
    // the FFN scratch (ffn_gate/up/h) is sized to this card's fc.
    struct Workspace {
        uint32_t T = 0;
        sycl::half *x = nullptr, *x_normed = nullptr, *attn_block = nullptr;
        int32_t* positions = nullptr;
        // full-attn (full dims)
        sycl::half *qg = nullptr, *q = nullptr, *gate = nullptr, *k = nullptr,
                   *v = nullptr, *attn_out = nullptr;
        float      *attn_partials = nullptr;
        uint32_t    partials_ctx = 0;
        // DeltaNet (full dims)
        sycl::half *dn_qkv = nullptr, *dn_conv = nullptr, *dn_z = nullptr;
        float      *dn_qpre = nullptr, *dn_kpre = nullptr, *dn_vpre = nullptr;
        float      *dn_g = nullptr, *dn_beta = nullptr, *dn_out = nullptr;
        float      *dn_qrep = nullptr, *dn_krep = nullptr;
        sycl::half *dn_alpha_h = nullptr, *dn_beta_h = nullptr;
        sycl::half *dn_alpha64 = nullptr, *dn_beta64 = nullptr;
        // FFN (sharded: fc cols)
        sycl::half *ffn_gate = nullptr, *ffn_up = nullptr, *ffn_h = nullptr;
        sycl::half *ffn_part = nullptr;   // [T,H] ffn_down partial (all-reduce buffer)
    };

    // Per-card FFN shard bookkeeping (256-superblock contiguous range of F).
    struct Shard { uint32_t f0 = 0, fc = 0; };

    DeviceFleet*  fleet_ = nullptr;
    Qwen35Config  cfg_;
    uint32_t      n_dev_ = 0;
    std::vector<Shard>     shard_;            // [dev]
    std::vector<Workspace> ws_;               // [dev]
    std::vector<LayerW>    layers_;           // [L] — replicated per card via build_split below
    std::vector<std::vector<LayerW>> dlayers_;// [dev][L] — per-card weight ptrs

    void*  token_embd_ = nullptr;  DType token_embd_dtype_ = DType::kCount;
    float* output_norm_ = nullptr;
    SplitW output_;                           // lm_head on card 0

    std::vector<std::vector<void*>> owned_;   // [dev] device ptrs to free
    std::vector<DeltaNetState>      dn_;       // [dev] (Phase 0: every card, full n_v)
    std::vector<KvCache>            kv_;       // [dev] (Phase 0: every card, full n_kv)
    std::vector<uint64_t>           dev_bytes_;
    std::vector<void*>       act_q8_;          // [dev] block_q8_1x [Kmax/32]
    std::vector<sycl::half*> prefill_bt_;       // [dev]
    std::vector<uint64_t>    prefill_bt_cap_;   // [dev]

    sycl::event sgemv(uint32_t dev, const sycl::half* A, const SplitW& w,
                      sycl::half* out, uint32_t K, uint32_t N, uint32_t T);

    std::string ensure_ws(uint32_t dev, uint32_t max_T);
    void free_ws(uint32_t dev);
    void free_all();
};

}  // namespace ie
