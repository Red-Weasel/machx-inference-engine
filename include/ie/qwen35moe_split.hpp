// include/ie/qwen35moe_split.hpp — Qwen3.6-35B-A3B crown (`kQwen35Moe`) multi-GPU
// LAYER-SPLIT for all-Q8_0 GGUFs (~34.4 GiB → 2 cards).
//
// ADDITIVE companion to the validated single-GPU `QwenModel` (qwen36.{hpp,cpp}) —
// that file is NEVER edited, so the crown PPL 6.4527 gate cannot move. This mirrors
// the 27B `Qwen35SplitModel` scaffold (same gated-DeltaNet + gated full-attn hybrid
// family, same Q8_0-SoA layer-split orchestration) but the FFN is the crown's top-k
// **MoE** (256 experts / top-8 + a shared expert), not a dense SwiGLU.
//
// Crown DeltaNet conventions (from QwenConfig, NOT the 27B's): SEPARATE
// ssm_alpha/ssm_beta projections (N-padded → batched gemm + extract_cols), TILE
// repeat 16→32 (rep=2, interleave=false), n_v_heads=32, ssm_inner=4096,
// conv_channels=8192. Per-layer math lifted VERBATIM from qwen36.cpp's forward —
// only the orchestration becomes per-card and the MoE becomes a per-expert Q8_0
// int-dot loop (Phase 1: no grouped Q8_0-expert kernel exists yet).
//
// `--gpus 1` keeps using `QwenModel`; `kQwen35Moe && --gpus>1` uses this.
#pragma once

#include "ie/allocator.hpp"          // DeviceFleet, DeviceAllocator
#include "ie/dense_split.hpp"        // LayerPlan
#include "ie/dense_transformer.hpp"  // DenseQuantPtr, dense::upload*
#include "ie/deltanet_state.hpp"     // DeltaNetState
#include "ie/gguf.hpp"
#include "ie/kv_cache.hpp"
#include "ie/qwen36.hpp"             // QwenConfig

#include <sycl/sycl.hpp>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace ie {

class Qwen35MoeSplitModel {
public:
    Qwen35MoeSplitModel() = default;
    ~Qwen35MoeSplitModel();
    Qwen35MoeSplitModel(const Qwen35MoeSplitModel&) = delete;
    Qwen35MoeSplitModel& operator=(const Qwen35MoeSplitModel&) = delete;

    // Load every layer L onto plan.dev_of_layer[L]; embed→plan.embed_dev,
    // lm_head→plan.head_dev. Per-card hybrid caches sized to that card's layer mix.
    // `fleet` and `g` must outlive this object. "" on success.
    std::string load(DeviceFleet& fleet, const LayerPlan& plan,
                     const GgufReader& g, const QwenConfig& cfg,
                     uint32_t max_ctx = 2048, bool int8_kv = false);

    const QwenConfig& config() const noexcept { return cfg_; }
    std::vector<uint64_t> device_bytes() const { return dev_bytes_; }
    uint32_t n_devices() const noexcept { return n_dev_; }

    // Per-card state accessors — same surface Qwen3NextModel exposes, so the
    // multi-GPU FleetPrefixCache (prompt/KV cache) can snapshot/restore this
    // model's per-card KvCache + DeltaNetState identically.
    // With request lanes (below) they are the SELECTED lane's state on that card.
    DeviceFleet*    fleet()                  const noexcept { return fleet_; }
    KvCache&        kv_cache(uint32_t dev)                  { return kv_at(dev); }
    DeltaNetState&  dn_state(uint32_t dev)                  { return dn_at(dev); }
    bool            dev_has_kv(uint32_t dev) const          { return kv_[dev].ready(); }
    bool            dev_has_dn(uint32_t dev) const          { return dn_[dev].ready(); }

    // Device-by-device forward. Host ids in, last token's logits → host fp16 [vocab].
    std::string forward(const int32_t* input_ids, uint32_t T, uint32_t start_pos,
                        bool reset_kv, sycl::half* out_logits_host);

    // P4 B10 (docs/lanes/LANES_SERVE.md): request lanes for `ie serve --parallel N` on the two-card split. A lane is
    // one sequence's KV + DeltaNet state on every card; lane 0 is the state load() allocated (nothing else changes with one
    // lane). init_lanes sizes every card's workspace for max_rows, then allocates lanes 1..n-1 at lane_ctx positions: refused,
    // with the numbers, when a card would keep less than reserve_bytes free after them.
    std::string init_lanes(uint32_t n_lanes, uint32_t lane_ctx, uint32_t max_rows, uint64_t reserve_bytes);
    uint32_t    n_lanes() const noexcept { return 1u + uint32_t(lane_kv_.size()); }
    uint32_t    lane_ctx(uint32_t lane) const noexcept { return lane ? lane_ctx_ : max_ctx_; }
    uint64_t    lane_bytes(uint32_t dev) const noexcept { return dev < lane_bytes_.size() ? lane_bytes_[dev] : 0; }
    uint32_t    head_dev() const noexcept { return plan_.head_dev; }
    void        select_lane(uint32_t lane);   // every card: kv_cache/dn_state/forward act on the lane
    void        reset_state();                // the selected lane on every card: a new sequence
    // One card's part of one step of `lane` (a card-pipe stage). Card 0: embeds ids, runs its layers, copies the residual rows
    // [T, hidden] fp16 out to x_host. Card 1: copies them in, runs its layers and the head: the last row's logits land in
    // logits() (device, card 1). pos0 0 resets the lane on the card first. Touches only that card's objects, so the two cards'
    // stages may run at once on two threads (for different lanes, or card 1 of chunk k beside card 0 of chunk k + 1).
    // Returns when the card's work is done. The kernels are forward()'s, card by card.
    std::string forward_stage(uint32_t dev, uint32_t lane, const int32_t* ids, uint32_t T, uint32_t pos0,
                              sycl::half* x_host);
    sycl::half* logits() const noexcept { return lane_logits_; }
    // P4 B14 phase 1b (rows): one card's part of ONE decode step of G = 2..kMaxRows lanes at once (a rows-mode group of the
    // card pipe), one row per lane, lane i at pos0[i] (never 0: a decode row never starts a sequence). Card 0: embeds the G
    // ids, runs its layers, copies the G residual rows [G, hidden] fp16 out to x_host. Card 1: copies them in, runs its
    // layers and the head: row i's logits land in rows_logits() + i * vocab (device, card 1). Every weight is read once for
    // the G rows (the batched int-dot GEMVs, the MoE decode kernels over G rows); the conv, the DeltaNet recurrence, the
    // alpha/beta projections and the attention run per lane against that lane's own state. Each row's bytes are the
    // one-lane forward_stage's (ie-q35m-rows-test --exact proves the shared kernels row for row). Needs rows_ok().
    static constexpr uint32_t kMaxRows = 16;
    std::string forward_stage_rows(uint32_t dev, std::span<const uint32_t> lanes, const int32_t* ids, const uint32_t* pos0,
                                   sycl::half* x_host);
    // "" = the rows path can run this checkpoint (every dense projection and the head Q8_0 int-dot, the experts Q8_0, the lanes
    // initialised); else why not.
    std::string rows_off_reason() const;
    bool        rows_ok() const { return rows_off_reason().empty(); }
    sycl::half* rows_logits() const noexcept { return rows_logits_; }

private:
    // A Q8_0 matrix weight stored PACKED as SoA int8 (no F16 doubling): q8_qs[n*K+k]
    // int8 column-contiguous + q8_d[n*(K/32)+b] fp16 per-32-block scale (de-interleaved
    // from on-disk AoS block_q8_0 — bit-exact). Consumed by gemv_q8_0_soa_q8 (decode) +
    // dequant_q8_0_soa_to_Bt+gemm (prefill). Non-Q8_0 → `fp` fallback (q8_qs == nullptr).
    struct SplitW {
        int8_t*   q8_qs = nullptr;   // [N*K] int8
        uint16_t* q8_d  = nullptr;   // [N*(K/32)] fp16 bits
        uint32_t  K = 0, N = 0;
        DenseQuantPtr fp;            // fallback when q8_qs == nullptr
    };

    // Per-layer MoE experts: E experts, each a Q8_0-SoA [K,N] matrix laid out
    // contiguously (expert e at qs + e*qs_stride, d + e*d_stride). gate/up: K=H,
    // N=E_ffn. down: K=E_ffn, N=H.
    struct ExpertsW {
        // Storage dtype: kQ8_0 = requant SoA (default path, qs/d below); kQ4_K/kQ6_K
        // = native crown-fused-kernel path (raw AoS bank in `blob`, opt-in via
        // IE_Q35MOE_NATIVE_KQUANT — halves expert VRAM, no requant-to-Q8 inflation).
        DType     dtype = DType::kQ8_0;
        // Q8_0 SoA planes (requant path):
        int8_t*   qs = nullptr;      // [E * N * K]
        uint16_t* d  = nullptr;      // [E * N * (K/32)]
        // Native Q4_K/Q6_K expert bank (consumed by moe_{decode,prefill}_*_q{4,6}k):
        void*     blob = nullptr;                // raw AoS [E,N,K] Q4_K/Q6_K
        uint64_t  expert_stride_bytes = 0;       // nbytes / E
        bool      soa = false;                   // false = raw AoS layout
        uint32_t  K = 0, N = 0, E = 0;
        uint64_t  qs_stride = 0;     // N*K  (per expert)
        uint64_t  d_stride  = 0;     // N*(K/32)
    };

    // Per-layer weights. EITHER linear (DeltaNet) or full-attn populated; the MoE
    // FFN + the two norms are shared. Mirrors qwen36.cpp LayerWeights.
    struct LayerW {
        bool is_linear = false;
        float* attn_norm = nullptr;
        float* post_attn_norm = nullptr;
        // full-attn
        SplitW attn_q, attn_k, attn_v, attn_output;
        float* attn_q_norm = nullptr;
        float* attn_k_norm = nullptr;
        // linear (DeltaNet) — crown dims
        SplitW attn_qkv;             // fused q|k|v conv input [H, conv_ch]
        SplitW attn_gate;            // z-gate input [H, ssm_inner]
        float* ssm_a = nullptr;      // [n_v]
        DenseQuantPtr ssm_alpha;     // F32→fp16 N-padded [H,64]
        DenseQuantPtr ssm_beta;
        float* ssm_conv1d = nullptr; sycl::half* ssm_conv1d_fp16 = nullptr;
        float* ssm_dt_bias = nullptr;
        float* ssm_norm = nullptr; sycl::half* ssm_norm_fp16 = nullptr;
        SplitW ssm_out;              // [ssm_inner, H]
        // MoE FFN (every layer)
        float*   ffn_gate_inp = nullptr;         // router [H, E] F32
        ExpertsW exp_gate, exp_up, exp_down;     // Q8_0-SoA experts
        // shared expert
        float*   ffn_gate_inp_shexp = nullptr;   // [H] sigmoid router
        SplitW   sh_gate, sh_up, sh_down;        // Q8_0-SoA shared expert
    };

    // Per-card forward scratch. Mirrors qwen36.cpp ws_* + the 27B split DeltaNet.
    struct Workspace {
        uint32_t T = 0;
        sycl::half *x = nullptr, *x_normed = nullptr, *attn_block = nullptr;
        int32_t* positions = nullptr;
        // full-attn
        sycl::half *qg = nullptr, *q = nullptr, *gate = nullptr;
        sycl::half *k = nullptr, *v = nullptr, *attn_out = nullptr;
        float      *attn_partials = nullptr; uint32_t partials_ctx = 0;
        // DeltaNet (crown dims: n_v=32, conv_ch=8192, ssm_inner=4096, Vd=4096)
        sycl::half *dn_qkv = nullptr, *dn_conv = nullptr, *dn_z = nullptr;
        float      *dn_qpre = nullptr, *dn_kpre = nullptr, *dn_vpre = nullptr;
        float      *dn_g = nullptr, *dn_beta = nullptr, *dn_out = nullptr;
        float      *dn_qrep = nullptr, *dn_krep = nullptr;
        sycl::half *dn_alpha_h = nullptr, *dn_beta_h = nullptr;
        sycl::half *dn_alpha64 = nullptr, *dn_beta64 = nullptr;
        // MoE
        int32_t*    topk_idx = nullptr;   // [T, K_top]
        sycl::half* topk_w = nullptr;     // [T, K_top]
        sycl::half* gate_o = nullptr;     // [T, E_ffn]
        sycl::half* up_o = nullptr;       // [T, E_ffn]
        sycl::half* ffn_h = nullptr;      // [T, E_ffn]
        sycl::half* moe_y = nullptr;      // [T, H] accumulator
        sycl::half* eo = nullptr;         // [T, H] expert/shared out
        // GPU-resident MoE stage buffers (no host pull)
        sycl::half* moe_hp = nullptr;     // [maxTK, E_ffn] stage-1 silu(gate)*up
        sycl::half* moe_yp = nullptr;     // [maxTK, H] stage-2 weighted down (y_packed)
        void*       x_q8 = nullptr;       // [T, H/32] block_q8_1x (stage-1 activation)
        void*       h_q8 = nullptr;       // [maxTK, E_ffn/32] block_q8_1x (stage-2 activation)
        // Expert-batched PREFILL (T>1) MoE scratch (the IE_Q35MOE_NO_PREFILL_GEMM
        // path). Host counting-sort of the TK token-slots by expert + expert-sorted
        // gather, then weight-stationary Q8_0 GEMMs. Decode (T==1) never touches these.
        sycl::half* moe_xp    = nullptr;  // [maxTK, H] expert-sorted gathered x
        void*       xp_q8     = nullptr;  // [maxTK, H/32] block_q8_1x (quantized moe_xp)
        uint32_t*   moe_eoff  = nullptr;  // [E+1] expert_offsets (device)
        int32_t*    moe_sidx  = nullptr;  // [maxTK] sorted token index per packed row
        sycl::half* moe_sw    = nullptr;  // [maxTK] routing weight per packed row
        uint32_t*   moe_tk2pk = nullptr;  // [maxTK] (t,kslot) → packed row (reduce map)
    };

    DeviceFleet* fleet_ = nullptr;
    LayerPlan    plan_;
    QwenConfig   cfg_;
    uint32_t     n_dev_ = 0;
    std::vector<Workspace> ws_;

    std::vector<LayerW> layers_;
    void*  token_embd_ = nullptr;  DType token_embd_dtype_ = DType::kCount;
    float* output_norm_ = nullptr;
    SplitW output_;

    std::vector<std::vector<void*>> owned_;
    std::vector<DeltaNetState>      dn_;
    std::vector<KvCache>            kv_;
    std::vector<uint32_t>           dn_local_;
    std::vector<uint32_t>           kv_local_;
    std::vector<uint64_t>           dev_bytes_;
    std::vector<void*>       act_q8_;       // [dev] block_q8_1x [Kmax/32]
    std::vector<sycl::half*> prefill_bt_;    // [dev] fp16 [K*N]max dequant scratch
    std::vector<uint64_t>    prefill_bt_cap_;

    // P4 B10 request lanes: lanes 1..n-1 ([lane - 1][dev]; lane 0 = kv_/dn_), the lane each card acts on, the logits a
    // card-1 stage leaves.
    std::vector<std::vector<KvCache>>       lane_kv_;
    std::vector<std::vector<DeltaNetState>> lane_dn_;
    std::vector<uint32_t>                   cur_;          // [dev]
    std::vector<uint64_t>                   lane_bytes_;   // [dev] one extra lane
    uint32_t    max_ctx_ = 0, lane_ctx_ = 0;
    sycl::half* lane_logits_ = nullptr;                    // [vocab] on head_dev
    KvCache&       kv_at(uint32_t dev) { return cur_[dev] ? lane_kv_[cur_[dev] - 1][dev] : kv_[dev]; }
    DeltaNetState& dn_at(uint32_t dev) { return cur_[dev] ? lane_dn_[cur_[dev] - 1][dev] : dn_[dev]; }
    // P4 B14 rows: a lane's state by index (the rows path never touches cur_), the G-row int-dot activation per card
    // [kMaxRows x Kmax/32 block_q8_1x], the G rows' logits on card 1 [kMaxRows x vocab]
    KvCache&       lane_kv(uint32_t lane, uint32_t dev) { return lane ? lane_kv_[lane - 1][dev] : kv_[dev]; }
    DeltaNetState& lane_dn(uint32_t lane, uint32_t dev) { return lane ? lane_dn_[lane - 1][dev] : dn_[dev]; }
    std::vector<void*> act_rows_;
    sycl::half*        rows_logits_ = nullptr;
    void        run_layers_rows(uint32_t dev, std::span<const uint32_t> lanes, const uint32_t* pos0);
    // the rows path's dense projection: quantize_q8_1 over G x K + gemv_q8_0_soa_q8_batched (forward_slots' forced route)
    sycl::event srows(uint32_t dev, const sycl::half* A, const SplitW& w, sycl::half* out, uint32_t K, uint32_t N, uint32_t G);

    // forward()'s pieces (kept in its order): the embedding into card embed_dev's x, one card's layers, the head.
    std::string embed(const int32_t* input_ids, uint32_t T);
    void        run_layers(uint32_t dev, uint32_t T, uint32_t start_pos, bool dbg_timing, double& t_attn, double& t_moe);
    void        head(uint32_t T, sycl::half* d_logits);

    // Q8_0-SoA aware GEMV (decode int-dot / prefill dequant+gemm / non-Q8 fallback).
    sycl::event sgemv(uint32_t dev, const sycl::half* A, const SplitW& w,
                      sycl::half* out, uint32_t K, uint32_t N, uint32_t T);
    // Per-expert Q8_0 int-dot GEMV against ExpertsW slice e (decode/prefill).
    sycl::event egemv(uint32_t dev, const sycl::half* A, const ExpertsW& w, uint32_t e,
                      sycl::half* out, uint32_t T);

    std::string ensure_ws(uint32_t dev, uint32_t max_T);
    void free_ws(uint32_t dev);
    void free_all();
};

}  // namespace ie
