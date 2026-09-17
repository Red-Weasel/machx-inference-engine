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
#include "ie/qwen35_dense.hpp"       // MtpHead + Qwen35SpecCheckpoint (spec-decode)

#include <functional>

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
                     uint32_t max_ctx = 2048, bool int8_kv = false);

    const Qwen35Config& config() const noexcept { return cfg_; }

    // Per-card state accessors for the multi-GPU FleetPrefixCache (prompt/KV cache).
    // 27B is a DeltaNet+full-attn hybrid (dense FFN), so it snapshots BOTH per-card
    // KvCache and DeltaNetState — identical surface to Qwen3NextModel/crown-split.
    DeviceFleet*    fleet()                     const noexcept { return fleet_; }
    KvCache&        kv_cache(uint32_t dev)                     { return kv_[dev]; }
    DeltaNetState&  dn_state(uint32_t dev)                     { return dn_[dev]; }
    bool            dev_has_kv(uint32_t dev)    const          { return kv_[dev].ready(); }
    bool            dev_has_dn(uint32_t dev)    const          { return dn_[dev].ready(); }
    std::vector<uint64_t> device_bytes() const { return dev_bytes_; }

    // --- Batched multi-sequence decode (decode-throughput campaign, Phase 2a).
    // A slot bank holds a full per-card (KvCache @ slot_ctx, DeltaNetState)
    // pair. Prefill runs on the live kv_/dn_ singletons exactly as today; a
    // finished prefill is bank_store()d device-to-device, and forward_slots()
    // then reads/writes slot state IN the banks (the live state is untouched).
    std::string alloc_slot_banks(uint32_t n_slots, uint32_t slot_ctx);
    void        free_slot_banks();
    uint32_t    n_slot_banks() const noexcept { return n_banks_; }
    uint32_t    slot_ctx()     const noexcept { return slot_ctx_; }
    std::string bank_store(uint32_t slot, uint32_t depth);   // live → bank[slot]
    std::string bank_load(uint32_t slot, uint32_t depth);    // bank[slot] → live
    KvCache&       bank_kv(uint32_t slot, uint32_t dev) { return bank_kv_[slot][dev]; }
    DeltaNetState& bank_dn(uint32_t slot, uint32_t dev) { return bank_dn_[slot][dev]; }
    // Serving-stepper hooks: device slot-logits rows ([n_banks, vocab] fp16 on
    // head_dev) for on-device per-slot sampling, and the head device index.
    sycl::half* slot_logits_dev() const noexcept { return d_slot_logits_; }
    uint32_t    head_dev()        const noexcept { return plan_.head_dev; }

    // One decode step for N active slots: ids[i] is slot i's pending token at
    // sequence position positions[i] (arbitrary, per slot); slot state is read
    // and advanced in bank[slots[i]]. Per-slot logits land at
    // out_logits_host + i*vocab (fp16). Weight GEMMs are row-batched across
    // slots (the spec-verify batched kernels); attention, causal conv, and the
    // DeltaNet recurrence run per slot against bank state.
    std::string forward_slots(uint32_t N, const int32_t* ids,
                              const uint32_t* positions, const uint32_t* slots,
                              sycl::half* out_logits_host);
private:
    // Per-card layer walk for one group of slots (pure enqueue; used by
    // forward_slots' 2-card group pipeline and its serial fallback).
    std::string stage_card_slots(uint32_t dev, uint32_t T,
                                 const uint32_t* positions,
                                 const uint32_t* slots);
public:
    uint32_t n_devices() const noexcept { return n_dev_; }

    // Device-by-device forward. Host ids in, last token's logits to host fp16
    // (size vocab). reset_kv clears per-card hybrid state for a fresh sequence.
    // Spec-decode hooks (all optional; buffers live on plan.head_dev's card):
    //   all_logits_dev      — [T, vocab] lm_head over EVERY row (verify).
    //   hidden_pre_norm_dev — [T, hidden] final residual BEFORE output_norm.
    //   ckpts               — per-card DeltaNet checkpoints (size n_devices());
    //                         when set (and each ckpt.K >= T), the DeltaNet conv +
    //                         recurrence decompose into T single-token steps with
    //                         per-position state snapshots (byte-identical output).
    std::string forward(const int32_t* input_ids, uint32_t T, uint32_t start_pos,
                        bool reset_kv, sycl::half* out_logits_host,
                        sycl::half* all_logits_dev = nullptr,
                        sycl::half* hidden_pre_norm_dev = nullptr,
                        std::vector<Qwen35SpecCheckpoint>* ckpts = nullptr);

    // ---- MTP self-speculative GREEDY decode (--spec) on the split path ----
    // Same lossless draft→verify→accept→commit machinery as the single-GPU
    // Qwen35DenseModel::spec_generate, with per-card KV/DeltaNet bookkeeping.
    // The NextN head lives on plan.head_dev (+~0.6 GB there, spec-only).
    using SpecEmit = std::function<bool(int32_t)>;
    std::string load_mtp_head(const GgufReader& g, uint32_t max_ctx);
    bool mtp_loaded() const noexcept { return mtp_.loaded; }
    // Self-contained: resets per-card state, chunk-prefills ids[0..P0) (exporting
    // the conditioning hidden), then runs the draft/verify loop. Emits committed
    // tokens in order via `emit` (return false to stop). Greedy/argmax only.
    std::string spec_generate(const int32_t* ids, uint32_t P0, uint32_t pf_chunk,
                              uint32_t max_new, uint32_t K, const SpecEmit& emit,
                              double* prefill_ms_out);

    // ---- PIPELINED PREFILL (2 cards): card 1 computes chunk c while card 0 ----
    // runs chunk c+1 — the layer-split's idle-card tax removed for prefill.
    // Falls back to the serial per-chunk path unless (n_dev==2, embed on dev 0,
    // head on dev 1, T_total > chunk). Same math per chunk as forward(); logits
    // of the LAST token → out_logits_host. Gated at the engine by IE_QWEN35_PIPELINE.
    std::string forward_pipelined(const int32_t* ids, uint32_t T_total,
                                  uint32_t start_pos, bool reset_kv, uint32_t chunk,
                                  sycl::half* out_logits_host,
                                  sycl::half* hidden_last_dev = nullptr);

private:
    // Slot banks (batched decode). bank_kv_[slot][dev] / bank_dn_[slot][dev];
    // d_slot_logits_ is an [n_banks, vocab] fp16 staging buffer on head_dev.
    std::vector<std::vector<KvCache>>       bank_kv_;
    std::vector<std::vector<DeltaNetState>> bank_dn_;
    sycl::half* d_slot_logits_ = nullptr;
    uint32_t    n_banks_ = 0, slot_ctx_ = 0;
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
        int32_t* ids = nullptr;          // [T] input ids (embed_dev only) — persistent,
                                         // was a per-forward malloc/free
        sycl::half* logits = nullptr;    // [vocab] lm_head out (head_dev only) — ditto
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
    std::vector<void*>       act_q8_;          // [dev] block_q8_1x [16, Kmax/32] (row 0 =
                                               // decode; rows for T≤16 spec verify)
    MtpHead mtp_;                              // NextN draft head on head_dev (--spec only)
    bool    spec_verify_gemv_ = false;         // route sgemv T∈[2,16] → batched int-dot
    std::vector<sycl::half*> prefill_bt_;       // [dev] fp16 [K*N]max dequant scratch
    std::vector<uint64_t>    prefill_bt_cap_;   // [dev] element capacity

    // Q8_0-SoA aware GEMV: out[T,N] = A[T,K] @ W. Decode (T==1) int-dot; prefill
    // dequant-to-fp16 + gemm; non-Q8_0 → dense::gemv_q_T. Runs on dev's queue.
    sycl::event sgemv(uint32_t dev, const sycl::half* A, const SplitW& w,
                      sycl::half* out, uint32_t K, uint32_t N, uint32_t T);

    // Enqueue one card's full layer stage for a prefill chunk (no waits/prof/ckpt;
    // used by forward_pipelined — the serial forward keeps its own validated loop).
    void stage_card_prefill(uint32_t dev, uint32_t T, uint32_t start_pos);

    std::string ensure_ws(uint32_t dev, uint32_t max_T);
    void free_ws(uint32_t dev);
    void free_all();
};

}  // namespace ie
