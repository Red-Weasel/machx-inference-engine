// include/ie/qwen4exp.hpp — Qwen3.8-Flash-Next (`qwen4exp`) model runtime.
//
// Campaign #3 (docs/qwen4-flash-next-port-campaign.md). Op-level ground truth
// lives in docs/qwen4/1*.md — every layout note below cites those specs.
//
// v0 bring-up placement (single B70, correctness before speed):
//  - DEVICE, F16: every non-expert matrix (attn, DeltaNet, HC up/down, shexp,
//    ple_key/ple_value, embeddings, lm_head), dequantized host-side at load.
//    ~17.2 GiB — fits one 32 GB card with KV + workspaces to spare. Native
//    quant residency (Q8_0 SoA etc.) is a later perf phase.
//  - DEVICE, F32: norms, hc gammas + inject, router (fp32 per spec 15), ssm
//    alpha/beta ([K, 64]-padded F16 like qwen35 — the one transposed layout),
//    ssm a/dt/conv, ple conv + gammas.
//  - HOST (GGUF mmap, raw quant): the 3 routed-expert banks per block
//    (gate/up Q4_K or Q5_K, down Q5_1; streamed + dequantized per token by the
//    MoE path) and the 26.8 GiB IQ4_NL PLE table (row-gather at hash time,
//    spec 12 §6).
#pragma once

#include "ie/allocator.hpp"
#include "ie/deltanet_state.hpp"
#include "ie/gguf.hpp"
#include "ie/kv_cache.hpp"
#include "ie/model_config.hpp"
#include "ie/qwen4_ple.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

namespace ie {

// W8 (Q8-SoA) dense residency default: ON since the 2026-08-27 PPL gate
// (streaming +0.42% / chunked-prefill −0.90% vs F16; spec lossless + bvs
// bit-exact under Q8). IE_Q4E_DENSE_Q8=0 (or "off") restores F16 dense.
// Only tensors stored Q8_0 in the GGUF take the SoA path (load-site rule);
// everything else stays F16 regardless of this switch.
inline bool qwen4exp_dense_q8() noexcept {
    const char* v = std::getenv("IE_Q4E_DENSE_Q8");
    return !(v && (v[0] == '0' || v[0] == 'o' || v[0] == 'O'));
}

// Device weights for one block. Pointers are F16 unless noted. Layout classes:
//  - projection matrices consumed by gemv_fp16 are TRANSPOSED to [K, N]
//    row-major (W[k*N+n]) at load, per the gemv_fp16 contract in ops.hpp;
//  - the HC up/down mats keep GGUF-NATIVE row-per-output order — that is the
//    layout contract of the qwen4_hc kernels (qwen4_hc.hpp);
//  - token_embd keeps GGUF-native [vocab rows of hidden] for row-gather;
//  - ssm_alpha/beta use the qwen35 [K, 64]-padded transposed layout.
// Null when the tensor does not exist on this block (DeltaNet vs full-attn vs
// PLE schedule — Qwen4ExpConfig::is_full_attn()).
struct Qwen4ExpLayer {
    // -- hyper-connections (spec 11): both sites on every block -------------
    sycl::half* hc_attn_up     = nullptr;  // [320 -> 10240]
    sycl::half* hc_attn_down   = nullptr;  // [10240 -> 320]
    float*      hc_attn_norm   = nullptr;  // [10240] gamma, (1+w) pre-folded
    float*      hc_attn_inject = nullptr;  // [10240, 4]
    sycl::half* hc_ffn_up      = nullptr;
    sycl::half* hc_ffn_down    = nullptr;
    float*      hc_ffn_norm    = nullptr;
    float*      hc_ffn_inject  = nullptr;
    // [K, N]-transposed copies for the T>16 prefill mix's gemm_fp16 path
    // (built once at init_runtime; ~26 MB/layer). v2 keeps the [N, K]
    // originals — its per-token GEMV wants row-contiguous outputs.
    sycl::half* hc_attn_up_t   = nullptr;
    sycl::half* hc_attn_down_t = nullptr;
    sycl::half* hc_ffn_up_t    = nullptr;
    sycl::half* hc_ffn_down_t  = nullptr;

    // Q8_0-SoA residency (perf: native int-dot GEMV, half the F16 reads).
    // When .qs != nullptr the F16 copy is NOT allocated and the projection
    // runs gemv_q8_0_soa_q8_* on quantize_q8_1 activations (qwen35-certified
    // pattern). Layout: qs[n*K + b*32 + i], d[n*(K/32) + b] (exact copy of
    // qwen35_split's build_split repack).
    // qs [N][K] int8 + d n-major [N][K/32] f16 (the decode-GEMV planes) +
    // dt KB-major [K/32][N] f16 (the oneDNN s8 prefill GEMM's scale plane —
    // oneDNN consumes attr scales in plain layout; handing it the n-major
    // plane is the documented 125x-wrong trap).
    struct Q8W { int8_t* qs = nullptr; uint16_t* d = nullptr;
                 uint16_t* dt = nullptr; };

    // -- Gated DeltaNet (spec 13; linear blocks only) -----------------------
    sycl::half* attn_qkv  = nullptr;   // [2560 -> 10240] fused Q|K|V (F16 fallback)
    Q8W         qkv_q8;                //   ... or Q8_0-SoA
    sycl::half* attn_gate = nullptr;   // [2560 -> 6144] output-gate stream z
    Q8W         gate_q8;
    sycl::half* ssm_alpha = nullptr;   // TRANSPOSED [2560, 64] (48 cols + zero pad), qwen35 convention
    sycl::half* ssm_beta  = nullptr;   // TRANSPOSED [2560, 64]
    float*      ssm_a     = nullptr;   // [48]  (= -exp(A_log), GGUF-precomputed)
    float*      ssm_dt    = nullptr;   // [48]  dt bias
    sycl::half* ssm_conv  = nullptr;   // [4, 10240] depthwise conv (F16 per depthwise_conv1d_causal)
    sycl::half* ssm_norm  = nullptr;   // [128] gated-RMS weight (raw, no +1)
    sycl::half* ssm_out   = nullptr;   // [6144 -> 2560]
    Q8W         out_q8;

    // -- full attention + QSA indexer (spec 10; il%4==3 only) ---------------
    sycl::half* attn_q      = nullptr;  // [2560 -> 12288] per-head [q|gate] interleaved
    Q8W         q_q8;                   //   ... or Q8-SoA (requantized at load)
    sycl::half* attn_k      = nullptr;  // [2560 -> 512]
    sycl::half* attn_v      = nullptr;  // [2560 -> 512]
    sycl::half* attn_output = nullptr;  // [6144 -> 2560]
    Q8W         attnout_q8;             //   ... or Q8-SoA (requantized at load)
    float*      attn_q_norm = nullptr;  // [256]
    float*      attn_k_norm = nullptr;  // [256]
    sycl::half* idx_q_proj  = nullptr;  // [2560 -> 512]  (BF16 in file -> F16)
    sycl::half* idx_k_proj  = nullptr;  // [2560 -> 128]
    float*      idx_q_norm  = nullptr;  // [128] gamma, (1+w) pre-folded
    float*      idx_k_norm  = nullptr;  // [128]

    // -- MoE (spec 15): router + shared expert device-resident --------------
    // Router W downcast to F16 for the GPU GEMV (qwen3next-certified pattern);
    // the softmax itself runs fp32 on host (route_from_logits).
    sycl::half* ffn_gate_inp       = nullptr;  // [2560 -> 512] transposed [K,N]
    sycl::half* ffn_gate_inp_shexp = nullptr;  // [2560] shared-expert scalar gate
    sycl::half* ffn_gate_shexp     = nullptr;  // [2560 -> 640]
    Q8W         shg_q8;                        //   ... or Q8-SoA (requant)
    sycl::half* ffn_up_shexp       = nullptr;  // [2560 -> 640]
    Q8W         shu_q8;                        //   ... or Q8-SoA (requant)
    sycl::half* ffn_down_shexp     = nullptr;  // [640 -> 2560]
    Q8W         shd_q8;                        //   ... or Q8-SoA (requant)

    // -- routed expert banks: HOST, raw GGUF quant (streamed) ---------------
    const GgufTensorInfo* gate_exps = nullptr;  // [2560, 640, 512] Q4_K (2 blocks Q5_K)
    const GgufTensorInfo* up_exps   = nullptr;  // [2560, 640, 512] Q4_K (2 blocks Q5_K)
    const GgufTensorInfo* down_exps = nullptr;  // [640, 2560, 512] Q5_1 (5 blocks Q8_0)

    // Runtime view of each bank: fast path streams the RAW quant slice
    // (Q4_K gate/up via gemv_q4_K_q8, Q5_1 down via gemv_q5_1); any other
    // dtype is pre-dequantized at load to a host F16 [E][K,N]-transposed
    // fallback consumed by gemv_fp16.
    struct BankView {
        const uint8_t*    raw = nullptr;   // host mmap, native path (else null)
        DType             dt  = DType::kCount;
        const sycl::half* f16 = nullptr;   // host F16 fallback (else null)
        uint64_t slice_bytes = 0;          // per-expert bytes of the ACTIVE repr
        uint32_t K = 0, N = 0;
    };
    BankView gate_bv, up_bv, down_bv;

    // -- PLE (spec 12; ple.layers blocks only, blk.1 on the real file) ------
    sycl::half* ple_key        = nullptr;  // [2560 -> 10240]
    sycl::half* ple_value      = nullptr;  // [2560 -> 2560]
    float*      ple_conv       = nullptr;  // [4, 10240] depthwise, dilation 3
    float*      ple_norm_query = nullptr;  // [10240] gammas, (1+w) pre-folded
    float*      ple_norm_key   = nullptr;
    float*      ple_norm_conv  = nullptr;
};

class Qwen4ExpModel {
public:
    ~Qwen4ExpModel();

    // Loads onto `alloc`'s device (single-GPU v0). `g` must outlive the model:
    // the routed-expert banks and the PLE table stay host-resident inside its
    // mmap. `vram_budget_bytes` gates projected device residency BEFORE any
    // upload (0 = default 28 GiB); IE_ALLOW_OOM=1 bypasses.
    // `layer_lo`/`layer_hi` restrict this instance to blocks [lo, hi) — the
    // 2-GPU pipeline loads one instance per card (globals bind by role:
    // token_embd/PLE with layer 0's owner, lm_head/output_hc with the tail).
    std::string load(DeviceAllocator& alloc, const GgufReader& g,
                     const Qwen4ExpConfig& cfg, uint64_t vram_budget_bytes = 0,
                     uint32_t layer_lo = 0, uint32_t layer_hi = UINT32_MAX);

    const Qwen4ExpConfig& config() const noexcept { return cfg_; }
    const Qwen4ExpLayer&  layer(uint32_t il) const { return layers_[il]; }

    // Globals (device, F16/F32 as noted).
    sycl::half* token_embd = nullptr;   // [vocab, 2560] row-gather source
    sycl::half* lm_head    = nullptr;   // [2560 -> vocab] (F16 fallback)
    Qwen4ExpLayer::Q8W lmh_q8;          //   ... or Q8_0-SoA
    sycl::half* out_hc_up   = nullptr;  // [320 -> 10240] final HC merge (spec 11 §5)
    sycl::half* out_hc_down = nullptr;  // [10240 -> 320]
    sycl::half* out_hc_up_t   = nullptr;  // [K, N] copies (prefill mix gemm)
    sycl::half* out_hc_down_t = nullptr;
    float*      out_hc_norm = nullptr;  // [10240]
    const GgufTensorInfo* ple_table = nullptr;  // [160, 320001536] IQ4_NL, host

    uint64_t device_bytes() const noexcept { return dev_bytes_; }
    uint64_t host_bank_bytes() const noexcept { return host_bytes_; }

    // -- runtime (v0, single GPU; correctness before speed) -----------------
    // init_runtime: KV (12 full-attn layers) + DeltaNet state (36 linear) +
    // PLE conv state + all workspaces for chunks up to `max_chunk` tokens.
    // forward: run T tokens at absolute position `start_pos`; when
    // `logits_out` != nullptr the LAST token's fp16 logits [vocab] are written
    // there (device pointer; model-owned logits() buffer is also valid).
    // v0 attention is EXACT dense attention, which equals QSA only while
    // start_pos + T <= indexer.top_k + compress_ratio - 1 (2051 on the real
    // file) — beyond that forward() refuses honestly (QSA lands next).
    std::string init_runtime(uint32_t max_ctx, uint32_t max_chunk);
    std::string forward(const int32_t* tokens_host, uint32_t T, uint32_t start_pos,
                        sycl::half* logits_out);
    // Pipeline stage entry: wide_in_host == nullptr -> embed here (needs
    // layer_lo == 0); wide_out_host != nullptr -> stop after our blocks and
    // download the wide state; otherwise run final merge + lm_head (needs
    // layer_hi == n_layers). tokens_host is required whenever this range owns
    // the PLE block.
    // wide_in_device: this range's wide input already sits in wide_ (the
    // previous stage PUSHED it over P2P) — wide_in_host is ignored.
    // Per-slot suspend/resume for interleaved serving (the 27B Phase-1b
    // HostSlotStash pattern): the FULL per-generation state — KV prefix
    // packed by depth, QSA indexer caches (raw keys + pooled block keys +
    // blk_done), DeltaNet state/conv, PLE conv+hist — lands in host RAM.
    // Unlike snapshot()/restore() (prompt-cache rewind of ONE conversation,
    // which can leave KV content in place), a suspended slot's cache rows
    // WILL be overwritten by the next slot, so content is copied both ways.
    // stash leaves the live state untouched; unstash overwrites it and sets
    // the depth markers. MTP state is NOT stashed (parallel serving runs
    // plain decode; refuse spec+parallel at the call site).
    struct SlotState {
        std::vector<uint8_t>  kv_k, kv_v;   // [L_full*n_kv][depth*hd] f16 packed
        std::vector<uint8_t>  idxk;         // [nf][depth*idx_hd] f16 packed
        std::vector<uint8_t>  blkk;         // [nf][ceil(depth/4)*idx_hd] f32
        std::vector<uint8_t>  dns, dnc;     // DeltaNet f32 state + f16 conv
        std::vector<uint8_t>  ple;          // PLE conv f32 (empty if no PLE)
        PleHistory            hist{};
        std::vector<uint32_t> blk_done;
        uint32_t              depth = 0;
    };
    std::string stash_slot(SlotState& s, uint32_t depth);
    std::string unstash_slot(const SlotState& s);

    std::string forward_range(const int32_t* tokens_host, uint32_t T,
                              uint32_t start_pos, const float* wide_in_host,
                              float* wide_out_host, sycl::half* logits_out,
                              bool wide_in_device = false);
    // Vision (docs/qwen4/16_vision_port.md §4). set_vision stages merged ViT
    // rows (f32 [n_rows, hidden], block-major) that overwrite the embed-gather
    // rows at absolute positions [t0, t0+n_rows) on the head stage. set_mrope
    // stages the 3-stream interleaved M-RoPE table for the prompt: pos3 is
    // [3, n_total] stream-major (T,H,W); positions >= n_total (decode
    // continuation) rope at (pos + delta) on all streams, while d_pos_ / the
    // QSA masking positions stay LINEAR token indices. Both persist across
    // forward calls until reset_state(). Not supported with the MTP head.
    std::string set_vision(const float* rows_f32, uint32_t t0, uint32_t n_rows);
    void        set_mrope(const int32_t* pos3, uint32_t n_total, int32_t delta);
    // Drop staged vision/mrope WITHOUT touching KV/DeltaNet state — the
    // prompt-cache restore path (which skips the pos==0 reset) calls this so a
    // previous request's staging can never leak into the current one.
    void        clear_vision() {
        vis_spans_.clear(); vis_rows_.clear();
        mrope_n_ = 0; mrope_delta_ = 0;
    }
    void        reset_state();
    sycl::half* logits() const noexcept { return logits_; }
    sycl::queue& queue() const { return alloc_->queue(); }
    // P2P interstage handoff (IE_P2P): when a peer pointer is set, a
    // wide-out forward_range PUSHES wide_ straight into the next stage's
    // wide_ from THIS device's queue (the verified direction on this
    // fabric — peer reads are broken) instead of bouncing through
    // wide_out_host. Requires both models' allocators in ONE shared
    // context with peer access enabled (gpu_p2p_shared_context).
    float* wide_device() noexcept { return wide_; }
    void   set_wide_peer(float* peer_wide_dev) noexcept { wide_peer_ = peer_wide_dev; }
    // P2P-pipelined prefill double banks. Bank 0 is the original wide_; bank 1
    // is allocated on demand. Producer alternates set_wide_peer(peer->
    // wide_bank(k & 1)) per chunk while the consumer calls use_wide_bank(k & 1)
    // before its forward — chunk k+1's push then never clobbers chunk k's
    // in-flight read (the single-buffer hazard that forced the host bounce).
    std::string alloc_wide_bank2();
    float* wide_bank(uint32_t idx) noexcept { return idx ? wide_alt_ : wide_bank0_; }
    void   use_wide_bank(uint32_t idx) noexcept { wide_ = wide_bank(idx); }
    uint32_t max_chunk() const noexcept { return max_chunk_; }

    // -- MTP draft head (speculative decode; tensors from mtp-head.gguf,
    // fetched from the HF checkpoint — the UD GGUF ships without them) ------
    struct Mtp {
        bool loaded = false;
        // fusion front-end
        float*      pre_fc_norm_hidden = nullptr;   // [10240] grouped gamma (+1 folded)
        float*      pre_fc_norm_embed  = nullptr;   // [2560]
        sycl::half* fc_hidden = nullptr;            // [2560 -> 2560] (transposed [K,N])
        sycl::half* fc_embed  = nullptr;            // [2560 -> 2560]
        // one full QSA decoder layer, engine-shaped (reuses run_block kernels)
        Qwen4ExpLayer lw;
        // final mixer (GR read, no inject)
        float*      out_hc_norm = nullptr;
        sycl::half* out_hc_up = nullptr, *out_hc_down = nullptr;
        // draft-layer state
        KvCache  kv;                    // 1 full-attn layer
        sycl::half* idx_kcache = nullptr;
        float*      blk_keys   = nullptr;
        uint32_t    blk_done   = 0;
    };
    // Binds + uploads the MTP head (call after load(), before init_runtime()
    // so the adaptive expert cache budgets around it). Expert bank goes
    // device-resident F16 (~4.7 GiB).
    // main_g (optional): tail-of-pipeline stages don't load token_embd, but
    // the MTP embed path needs it — pass the MAIN model's reader to pull it.
    std::string load_mtp(const GgufReader& mtp_g,
                         const GgufReader* main_g = nullptr);
    bool mtp_loaded() const noexcept { return mtp_.loaded; }
    // Ingest the last forward chunk into the MTP layer's context (call after
    // forward(); uses the chunk's final wide rows still in wide_). Position i
    // pairs (wide_i, token_{i+1}); the chunk's last position waits for the
    // next chunk's first token via a 1-row carry.
    std::string mtp_ingest(const int32_t* tokens_host, uint32_t T, uint32_t start_pos);
    // Draft K tokens greedily from the current state (after a forward whose
    // sampled next token is t_next at position pos). Fills draft[0..K).
    std::string mtp_draft(int32_t t_next, uint32_t pos, uint32_t K, int32_t* draft);
    // Logits for EVERY row of the last forward (verify step): mixed_ still
    // holds all T final-merge rows. out: device [T, vocab] f16.
    std::string logits_rows(uint32_t T, sycl::half* out);

    // -- prompt-prefix snapshot (agent multi-turn TTFT) ---------------------
    // The DeltaNet/PLE recurrences cannot rewind, so prefix reuse = snapshot
    // the recurrent state at a chosen depth and restore it when the next
    // prompt extends that prefix (KV + QSA idx caches are position-indexed
    // and stay in place; only depth markers rewind). Cost: ~115 MB D2D per
    // direction. One snapshot slot (the growing-conversation agent pattern).
    std::string snapshot(uint32_t depth);
    bool        has_snapshot() const noexcept { return snap_depth_ > 0; }
    uint32_t    snapshot_depth() const noexcept { return snap_depth_; }
    std::string restore();     // rewinds to snapshot_depth()
    void        drop_snapshot() noexcept { snap_depth_ = 0; }

    // -- spec-verify mode ---------------------------------------------------
    // While on, forward() with T in [2,16] computes every row with kernels
    // BIT-IDENTICAL to T sequential T=1 decode steps (per-row gemv leaves,
    // decode attention looped per row, decode MoE routing/experts per row).
    // Required for lossless speculative decode: the verify batch IS the
    // greedy reference chain. Prefill batches are unaffected.
    // Such a forward also checkpoints the DeltaNet/PLE recurrent state AFTER
    // every row; commit_verify(n) then adopts the state as of row n-1 (the
    // accepted prefix) — no snapshot/restore/replay forward needed.
    void set_spec_verify(bool on) noexcept { spec_verify_ = on; }
    std::string commit_verify(uint32_t n_commit);

    // Replicated multi-GPU topologies: reuse `d`'s prepared bank views
    // (pinned host USM, legal across a shared-context fleet) instead of
    // pinning a second ~66 GB copy. Call BEFORE init_runtime; the donor
    // must already be initialized and outlive this instance.
    void set_bank_donor(const Qwen4ExpModel* d) noexcept { bank_donor_ = d; }

    // -- verify-only expert parallelism (replicated topology) ---------------
    // This instance computes verify-MoE cells ONLY for experts in [lo, hi);
    // after its cells land, `sync(L)` runs (the driver exchanges peer cells
    // into verify_cells() before the ordered reduce). Reference/T=1 lanes are
    // untouched (fully replicated) — cross-card determinism of identical
    // kernels on identical replicated inputs keeps spec lossless.
    void set_tp_experts(uint32_t e_lo, uint32_t e_hi,
                        std::function<void(uint32_t)> sync) {
        tp_e_lo_ = e_lo; tp_e_hi_ = e_hi; tp_sync_ = std::move(sync);
    }
    sycl::half* verify_cells() noexcept { return moe_yT_; }
    // Per-cell ownership for the current verify layer (1 = computed here),
    // [T*NU] entries, filled before tp_sync_ runs. Driver-readable.
    std::vector<uint8_t> tp_cell_owner_;

    // Oracle parity hooks: stage outputs copied to HOST buffers when set
    // (each [T, hidden] F16). Used only by the parity gate.
    struct ParityCapture {
        sycl::half* mixed_attn = nullptr;  // post hc_attn mix (block input)
        sycl::half* mixer_out  = nullptr;  // DeltaNet/attention out, pre-combine
        sycl::half* mixed_ffn  = nullptr;  // post hc_ffn mix (MoE input)
        sycl::half* moe_out    = nullptr;  // MoE out, pre-combine
    };
    // Run block L THROUGH THE REAL forward path on an injected wide residual
    // state (host fp32 [T, 4, hidden]) at positions [0, T). Caller must
    // reset_state() first — the oracle runs layers with fresh caches. Writes
    // the post-block wide state to wide_out_host (same shape); tokens_host is
    // required for the PLE block (hash inputs), ignored elsewhere.
    std::string run_block_parity(uint32_t L, const float* wide_in_host, uint32_t T,
                                 const int32_t* tokens_host, float* wide_out_host,
                                 ParityCapture* cap = nullptr);

    // Per-tensor placement rows for the load report/audit tool.
    struct PlacementRow { std::string name; const char* where; const char* dt; uint64_t bytes; };
    const std::vector<PlacementRow>& placement() const noexcept { return placement_; }

private:
    void free_all();
    std::string prepare_bank_views();
    void run_block(uint32_t L, uint32_t T, uint32_t start_pos,
                   const int32_t* tokens_host, ParityCapture* cap);

    DeviceAllocator*   alloc_ = nullptr;
    Qwen4ExpConfig     cfg_{};
    std::vector<Qwen4ExpLayer> layers_;
    std::vector<void*> owned_;
    uint64_t dev_bytes_  = 0;
    uint64_t host_bytes_ = 0;
    std::vector<PlacementRow> placement_;

    // Host F16 fallback storage for non-fast-path banks (kept alive here).
    std::vector<std::vector<sycl::half>> f16_banks_;

    // Runtime state.
    KvCache       kv_;
    DeltaNetState dn_;
    std::vector<int32_t> full_idx_, lin_idx_;   // layer -> cache-local index
    uint32_t max_ctx_ = 0, max_chunk_ = 0;
    uint32_t layer_lo_ = 0, layer_hi_ = 0;      // owned block range [lo, hi)
    // Snapshot storage (device): DeltaNet recurrent+conv state, PLE conv
    // state; host: PLE history, per-layer QSA block counters, KV lengths.
    float*      snap_dn_state_ = nullptr;
    sycl::half* snap_dn_conv_  = nullptr;
    float*      snap_ple_conv_ = nullptr;
    PleHistory  snap_ple_hist_{};
    std::vector<uint32_t> snap_blk_done_;
    uint32_t    snap_depth_ = 0;
    bool        spec_verify_ = false;
    // run_block() is void; a block-level fault (e.g. verify expert union >
    // cache slots) latches here and forward_range()/run_block_parity()
    // surface it instead of computing on corrupt slots.
    std::string block_err_;
    const Qwen4ExpModel* bank_donor_ = nullptr;
    uint32_t tp_e_lo_ = 0, tp_e_hi_ = 0xFFFFFFFFu;
    std::function<void(uint32_t)> tp_sync_;
    // spec-verify per-position state checkpoints ([slab t] = state after
    // consuming verify row t); lazily sized to the largest verify T seen.
    float*      vck_dn_state_ = nullptr;   // [cap, n_lin * se] f32
    sycl::half* vck_dn_conv_  = nullptr;   // [cap, n_lin * ce] f16
    float*      vck_ple_conv_ = nullptr;   // [cap, kPleStateRows * kPleSI] f32
    PleHistory  vck_ple_hist_[16] = {};
    uint32_t    vck_cap_ = 0, vck_T_ = 0, vck_pos_ = 0;
    uint32_t    vck_base_ = 0;   // slab offset: a verify forward that EXTENDS
                                 // the previous one (start == pos+T) appends
                                 // its checkpoints, so a T=1 row-0 pass (run
                                 // concurrently with the draft on the other
                                 // card) plus a T=K tail commit uniformly.
    Mtp mtp_{};
    // MTP scratch: fused wide input, boundary carry, embed rows.
    float*      mtp_wide_ = nullptr;      // [MT, 4, 2560]
    sycl::half* mtp_x16_  = nullptr;      // [MT, 4, 2560] normed wide, f16
    sycl::half* mtp_e16_  = nullptr;      // [MT, 2560] normed embed rows
    float*      mtp_carry_ = nullptr;     // [4, 2560] prev chunk's last wide row
    uint32_t    mtp_carry_pos_ = 0;
    bool        mtp_have_carry_ = false;
    std::string mtp_layer_fwd(uint32_t T, uint32_t start_pos);
    PleHistory ple_hist_{};
    float*   ple_conv_state_ = nullptr;   // [9, 10240] f32
    float*   ple_ws_ = nullptr;
    // Workspaces (device USM; sized for max_chunk_ = MT tokens).
    int32_t* d_tokens_ = nullptr;         // [MT]
    int32_t* d_pos_    = nullptr;         // [MT]
    int32_t* d_pos3_   = nullptr;         // [3, MT] imrope T/H/W streams (vision)
    sycl::half* emb_ = nullptr;           // [MT, H]
    // Vision splice + M-RoPE staging (set_vision / set_mrope; reset_state
    // clears). set_vision APPENDS a span — multi-image prompts call it once
    // per <|image_pad|> run, rows concatenated in call order.
    struct VisSpan { uint32_t t0, n, row0; };
    std::vector<VisSpan> vis_spans_;
    std::vector<sycl::half> vis_rows_;    // concat [sum n, H] f16
    std::vector<int32_t> mrope3_;         // [3, mrope_n_] host
    uint32_t mrope_n_ = 0;
    int32_t  mrope_delta_ = 0;
    float*   wide_  = nullptr;            // [MT, 4, H]
    float*   wide_peer_ = nullptr;        // next stage's wide_ (P2P push)
    float*   wide_bank0_ = nullptr;       // init_runtime's wide_ (bank 0)
    float*   wide_alt_  = nullptr;        // bank 1 (alloc_wide_bank2)
    float*   xn_ws_ = nullptr;            // [MT, 4H]
    float*   lo_ws_ = nullptr;            // [MT, r]
    sycl::half* hc_x16_  = nullptr;       // prefill HC mix: F16 xn [MT, 4H]
    sycl::half* hc_lo16_ = nullptr;       // prefill HC mix: F16 lo [MT, r]
    sycl::half* mixed_ = nullptr;         // [MT, H]
    float*   inj_   = nullptr;            // [MT, 4]
    sycl::half* blockout_ = nullptr;      // [MT, H]
    sycl::half* dn_qkv_ = nullptr;        // [MT, 10240]
    sycl::half* dn_conv_ = nullptr;       // [MT, 10240]
    float *dn_qpre_ = nullptr, *dn_kpre_ = nullptr, *dn_vpre_ = nullptr;
    float *dn_qrep_ = nullptr, *dn_krep_ = nullptr;
    sycl::half *dn_ab64_ = nullptr;       // [MT, 128] alpha|beta padded
    sycl::half *dn_a48_ = nullptr, *dn_b48_ = nullptr;   // [MT, 48]
    float *dn_g_ = nullptr, *dn_beta_ = nullptr;         // [MT, 48]
    float* dn_out_ = nullptr;             // [MT, 6144]
    sycl::half *dn_z_ = nullptr, *dn_gn_ = nullptr;      // [MT, 6144]
    sycl::half *qg_ = nullptr, *aq_ = nullptr, *agate_ = nullptr;  // [MT,12288]/[MT,6144]/[MT,6144]
    sycl::half *ak_ = nullptr, *av_ = nullptr;           // [MT, 512]
    sycl::half* attn_out_ = nullptr;      // [MT, 6144]
    sycl::half* router_out_ = nullptr;    // [512]
    void*    act_q8_ = nullptr;           // q8_1 stream of one [H] row
    sycl::half *moe_g_ = nullptr, *moe_u_ = nullptr, *moe_h_ = nullptr;  // [640]
    sycl::half* moe_y_ = nullptr;         // [H]
    sycl::half *sh_g_ = nullptr, *sh_u_ = nullptr, *sh_h_ = nullptr, *sh_y_ = nullptr, *sh_s_ = nullptr;
    float*   moe_acc_ = nullptr;          // [H] fp32 accumulator
    // MoE VRAM expert cache (perf iteration 2): per layer, C slots each
    // holding one expert's gate|up|down slices in the bank's active repr;
    // host-tracked LRU by expert id. Slices are immutable weight copies, so
    // cache state never affects numerics — only which bytes move over PCIe.
    struct ECache {
        void* base = nullptr;                 // device, C * slot_bytes
        uint64_t gate_off = 0, up_off = 0, down_off = 0, slot_bytes = 0;
        std::vector<int16_t>  slot_of;        // [n_experts] -> slot or -1
        std::vector<int32_t>  expert_in;      // [C] -> expert id or -1
        std::vector<uint64_t> last_use;       // [C] LRU stamps
    };
    std::vector<ECache> ecache_;
    uint64_t ecache_clock_ = 0;
    uint32_t ecache_slots_ = 0;
    // Prefill expert-upload prefetch queue (in-order, same device/context as
    // compute): with half-cache waves, wave i+1's slot refills overlap wave
    // i's kernels — see the wave loop's eviction-safety argument.
    std::unique_ptr<sycl::queue> copyq_;
public:
    uint64_t ecache_hits = 0, ecache_misses = 0;   // perf counters (host)
    // Decode router-sync breakdown (grouped T=1 path): time blocked in the
    // per-layer logits D2H .wait() vs host routing + job build + H2D submit.
    double t_route_wait = 0, t_route_host = 0, t_moe_submit = 0;
private:
    sycl::half* e16_ = nullptr;           // [MT, H] PLE embedding f16
    sycl::half *ple_key_out_ = nullptr, *ple_v_out_ = nullptr;  // [MT,10240]/[MT,H]
    sycl::half* logits_ = nullptr;        // [vocab]
    // Prefill batching (T>1): fp32 GEMM C scratch + expert-major MoE buffers.
    float*      gemm_c_    = nullptr;     // [MT, 12288] fp32
    sycl::half* router_all_ = nullptr;    // [MT, 512]
    void*       act_q8T_   = nullptr;     // [MT] q8_1 streams of H
    sycl::half* dq_ws_     = nullptr;     // T>16 dense-Q8 dequant scratch
                                          // (largest dense mat, gemv layout)
    float*      moe_accT_  = nullptr;     // [MT, H] fp32
    sycl::half *moe_gT_ = nullptr, *moe_uT_ = nullptr, *moe_hT_ = nullptr;  // [MT, EF]
    sycl::half* moe_yT_    = nullptr;     // [MT, H]
    float*      moe_rsum_  = nullptr;     // [PICKS+16, EF/32] dpas-down rowsums
    int32_t*    moe_tok_idx_ = nullptr;   // [MT * top_k] device
    float*      moe_tok_w_   = nullptr;   // [MT * top_k] device
    // Pinned host staging (sycl::malloc_host): pageable-vector H2D staged a
    // bounce copy at EVERY per-layer submit — measured inside the 6-8 ms/token
    // decode moe-submit bubble (2026-08-31). Same indexing as the old vectors.
    int32_t* h_tok_idx_ = nullptr;
    float*   h_tok_w_   = nullptr;
    sycl::half* h_router16_ = nullptr;    // pinned router-logits D2H target
    float*      qsa_scoresT_  = nullptr;  // chunk-batched QSA [MT, max_ctx/4]
    int32_t*    qsa_blk_selT_ = nullptr;  // [MT, top_k/4]
    int32_t*    qsa_selT_     = nullptr;  // [MT, sel_cap]
    int32_t*    qsa_nselT_    = nullptr;  // [MT]
    float*      qsa_partG_    = nullptr;  // fused-attend group scratch [64,nq,16,hd+2]
    void*       act_q8s_  = nullptr;      // block_q8_1s decode activation
    void*       act_q8sT_ = nullptr;      // [MT rows] _1s activations
    void*       xg_q8s_   = nullptr;      // gathered _1s mega rows
    int32_t*    moe_tiles_ = nullptr;     // grouped prefill tile jobs [J x 4]
    int32_t*    moe_cells_ = nullptr;     // per-token pick cells + nk tail
    std::vector<int32_t> h_tiles_, h_cells_;
    void*       xg_q8_     = nullptr;     // [MT] gathered q8 streams
    sycl::half* xg16_      = nullptr;     // [MT, H] gathered f16 rows
    sycl::half* sgateT_    = nullptr;     // [MT] batched shexp scalar gates

    // -- QSA indexer state (12 full-attn layers; spec docs/qwen4/10_*.md) ----
    // Raw indexer keys are cached UN-normed/UN-roped; complete blocks of 4 are
    // pooled(mean,fp32)+normed+roped ONCE into blk_keys_ (immutable after) —
    // the incremental improvement over the reference's per-step re-pool.
    sycl::half* idx_kcache_ = nullptr;    // [n_full, max_ctx, 128] raw keys
    float*      blk_keys_   = nullptr;    // [n_full, max_ctx/4, 128] pooled+normed+roped
    std::vector<uint32_t> blk_done_;      // per full layer: blocks materialized
    sycl::half* idx_q_ = nullptr;         // [MT, 4, 128] roped indexer queries
    sycl::half* idx_kraw_ = nullptr;      // [MT, 128] this chunk's raw keys
    float*      qsa_scores_ = nullptr;    // [max_ctx/4] one query's block scores
    int32_t*    qsa_sel_ = nullptr;       // [sel_cap] selected token positions
    int32_t*    qsa_blk_sel_ = nullptr;   // [512] top-k block ids (device)
    int32_t*    qsa_nsel_ = nullptr;      // [1] device-side selection count
    float*      qsa_part_ = nullptr;      // [n_q, 16, hd+2] split-K partials
    sycl::half *qsa_gk_ = nullptr, *qsa_gv_ = nullptr;  // gathered KV [2, sel_cap, 256]
    uint32_t    qsa_sel_cap_ = 0;
};

}  // namespace ie
