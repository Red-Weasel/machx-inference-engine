// include/ie/qwen35_dense.hpp — P3d: Qwen3.6-27B (`qwen35` dense-hybrid).
//
// The crown family (gated-DeltaNet linear layers + gated full-attention every
// `full_attn_interval`th layer) MINUS the MoE: the FFN is a plain dense SwiGLU
// (the P2 path). 64 transformer layers (48 linear + 16 full) + 1 trailing
// NextN/MTP layer we SKIP (text-only decode). Forward, per layer il:
//   rms(attn_norm) → { linear: DeltaNet  |  full: gated full-attn }
//                  → +residual → rms(post_attention_norm) → SwiGLU → +residual
//   → rms(output_norm) → lm_head
//
// IRON RULE: this is NEW code. `src/model/qwen36.cpp` (the crown) is NEVER
// edited; we CALL the shared `src/ops/*` leaf free functions and re-derive the
// DeltaNet/attention orchestration here (the crown's inline `forward` is a
// reference to read, not code to call). Shared GEMV/upload helpers come from
// `src/model/dense_dispatch.hpp` (the P2 copies), same copy-not-hoist discipline.
//
// Caller contract (mirrors QwenModel — hybrid caches, NOT the dense "all full"):
//   * keep `GgufReader` alive while loaded.
//   * KvCache sized n_layers_full = 16 (full-attn layers only), n_kv_heads = 4,
//     head_dim = 256, indexed by full_idx = il / full_attn_interval.
//   * DeltaNetState sized n_layers_linear = 48, n_v_heads = 48, k/v head_dim 128,
//     conv_channels = 10240, conv_kernel = 4, indexed dn_idx = il - il/interval.
#pragma once

#include "ie/allocator.hpp"
#include "ie/deltanet_state.hpp"
#include "ie/dense_transformer.hpp"   // DenseQuantPtr
#include "ie/gguf.hpp"
#include "ie/kv_cache.hpp"
#include "ie/model_config.hpp"        // Qwen35Config

#include <sycl/sycl.hpp>

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace ie {

// SoA-Q6 repacked weight (fast Q6_K decode path, gemv_q6_soa.cpp). A canonical
// Q6_K [K,N] weight is repacked at load into per-column natural-order bit-plane
// streams (q6_lo 4-bit, q6_hi 2-bit, per-16 int8 scales, fp16 super-scale,
// ~6.5 bpw).  When `lo != nullptr` the dense forward routes this weight's decode
// GEMV through gemv_q6_soa_q8 (int-dot W6A8) and prefill through
// dequant_q6_soa_to_Bt + gemm_fp16, instead of the AoS Q6_K path.
struct Q6SoaW {
    uint8_t*  lo = nullptr;   // [N*(K/2)]   4-bit low nibbles
    uint8_t*  hi = nullptr;   // [N*(K/4)]   2-bit high bits
    int8_t*   sc = nullptr;   // [N*(K/16)]  per-16 int8 scales
    uint16_t* d  = nullptr;   // [N*(K/256)] fp16 super-scale
    uint32_t  K = 0, N = 0;
    bool active() const noexcept { return lo != nullptr; }
};

// SoA-Q8_0 fast-decode mirror (de-interleaved AoS block_q8_0 → column-contiguous
// int8 qs + per-32-block fp16 d, ~8.5 bpw, bit-exact — no requant for native
// Q8_0). When `qs != nullptr` the dense forward routes decode through the ~80%-BW
// int-dot gemv_q8_0_soa_q8 and prefill through dequant_q8_0_soa_to_Bt + gemm_fp16,
// instead of expanding the weight to F16 (2× the bytes). Mirrors qwen35_split's
// SplitW Q8_0 path, brought to the single-GPU decode (2026-06-22 GEMV grind).
struct Q8SoaW {
    int8_t*   qs = nullptr;   // [N*K]        int8 quants, column-contiguous
    uint16_t* d  = nullptr;   // [N*(K/32)]   per-32-block fp16 scale (raw bits)
    uint32_t  K = 0, N = 0;
    bool active() const noexcept { return qs != nullptr; }
};

// SoA-Q4 repacked weight (fast Q4_K decode path, gemv_q4_soa.cpp). Mirrors
// Q6SoaW for Q4_K weights: a canonical block_q4_K [K,N] is repacked at load into
// per-column NATURAL-ORDER streams (q 4-bit nibbles, per-32 int8 s_raw/m_raw —
// the unpacked 6-bit scale/min, per-256 fp16 d/dmin, ~4.625 bpw). When `q !=
// nullptr` the dense forward routes this weight's decode GEMV through the int-dot
// W4A8 gemv_q4_soa_q8 (vs the AoS W4A16 gemv_q4_K) and prefill through
// dequant_q4_soa_to_Bt + gemm_fp16. Gated opt-IN (IE_QWEN35_Q4_SOA) so the
// default forward stays byte-identical to the AoS Q4_K path.
struct Q4SoaW {
    uint8_t*  q    = nullptr; // [N*(K/2)]   4-bit nibbles, 2 elems/byte
    int8_t*   sc   = nullptr; // [N*(K/32)]  per-32 int8 s_raw (unpacked 6-bit)
    int8_t*   mn   = nullptr; // [N*(K/32)]  per-32 int8 m_raw (unpacked 6-bit)
    uint16_t* d    = nullptr; // [N*(K/256)] per-256 fp16 super-scale d
    uint16_t* dmin = nullptr; // [N*(K/256)] per-256 fp16 super-min   dmin
    uint32_t  K = 0, N = 0;
    bool active() const noexcept { return q != nullptr; }
};

// Per-layer weights. A layer is EITHER linear (DeltaNet) or full-attention;
// only the matching block is populated (the other stays null). The FFN +
// the two norms are shared by both kinds.
struct Qwen35LayerWeights {
    bool is_linear = false;           // recurrent_layer(il)

    // --- shared (both kinds) ---
    float* attn_norm = nullptr;       // blk.N.attn_norm.weight            F32 [hidden]
    float* post_attn_norm = nullptr;  // blk.N.post_attention_norm.weight  F32 [hidden]
    DenseQuantPtr ffn_gate, ffn_up, ffn_down;   // Q4_K/Q6_K, [K,N]
    // SoA-Q6 fast-decode mirror (set when the corresponding DenseQuantPtr was
    // Q6_K and the SoA path is enabled; AoS pointer is freed after repack).
    Q6SoaW ffn_gate_soa, ffn_up_soa, ffn_down_soa;
    Q6SoaW attn_q_soa, ssm_out_soa;
    // SoA-Q4 fast-decode mirror (set when the corresponding DenseQuantPtr was
    // Q4_K and the opt-IN SoA path IE_QWEN35_Q4_SOA is enabled; AoS pointer is
    // left null after repack). One per Q4_K-capable decode GEMV weight.
    Q4SoaW ffn_gate_q4soa, ffn_up_q4soa, ffn_down_q4soa;
    Q4SoaW attn_q_q4soa, ssm_out_q4soa, attn_v_q4soa;
    Q4SoaW attn_qkv_q4soa, attn_gate_q4soa;

    // --- full-attention layers ---
    // attn_q is the JOINT Q|gate projection [5120, 12288] = n_q(24)·head(256)·2
    // (landmine: NOT n_q·head — the ·2 folds the per-head sigmoid gate; split
    // with split_q_gate_per_head after the projection).
    DenseQuantPtr attn_q;             // joint Q|gate           [5120, 12288]
    DenseQuantPtr attn_k;             // Q5_K → requant SoA-Q8 (or F16 fallback) [5120, 1024]
    DenseQuantPtr attn_v;             //                         [5120, 1024]
    DenseQuantPtr attn_output;        // Q5_K → requant SoA-Q8 (or F16 fallback) [6144, 5120]
    Q8SoaW attn_k_q8, attn_output_q8; // SoA-Q8 fast-decode mirror (Q5_K requantized)
    float* attn_q_norm = nullptr;     // [head_dim 256]  F32  (per-head Q-norm)
    float* attn_k_norm = nullptr;     // [head_dim 256]  F32

    // --- linear (DeltaNet) layers ---
    // attn_qkv is the FUSED q|k|v conv-input projection [5120, 10240].
    // conv_channels = d_inner(6144) + 2·n_k_heads(16)·k_head_dim(128) = 10240.
    // BAN SI2=SI*2 (crown identity, false here). Split offsets after conv:
    //   q_conv @ 0 (2048=16·128), k_conv @ 2048 (2048), v_conv @ 4096 (6144).
    DenseQuantPtr attn_qkv;           // fused q|k|v conv input  [5120, 10240]
    DenseQuantPtr attn_gate;          // gated-norm z input      [5120, 6144]
    Q6SoaW attn_qkv_soa, attn_gate_soa;
    float* ssm_a = nullptr;           // ssm_a              F32 [n_v_heads 48] (A_log)
    DenseQuantPtr ssm_alpha;          // ssm_alpha.weight   F32→fp16 [K,N]=[5120,48] proj
    DenseQuantPtr ssm_beta;           // ssm_beta.weight    F32→fp16 [K,N]=[5120,48] proj
    float* ssm_conv1d = nullptr;      // ssm_conv1d.weight  F32 [4, 10240]
    sycl::half* ssm_conv1d_fp16 = nullptr;
    float* ssm_dt_bias = nullptr;     // ssm_dt.bias        F32 [48]
    float* ssm_norm = nullptr;        // ssm_norm.weight    F32 [128]
    sycl::half* ssm_norm_fp16 = nullptr;
    DenseQuantPtr ssm_out;            // ssm_out.weight  Q8_0 → F16 [6144, 5120]
    Q8SoaW ssm_out_q8;                // SoA-Q8_0 fast-decode mirror (native Q8_0 only)
};

// Spec-decode per-position DeltaNet checkpoint sidecar.
//
// Captures the DeltaNet RECURRENT + CONV state AFTER each of the K verify
// positions, per linear layer, during a single checkpoint-mode verify forward.
// Then commit_to_n() copies the slice for the n accepted positions straight
// into the live DeltaNetState — eliminating the rollback RE-FORWARD (a 2nd
// weight read) the spec loop previously paid on every partial accept, and the
// run-to-run DeltaNet non-determinism that re-running the recurrence triggered.
//
// Memory: K · n_linear · (state_elems·4 + conv_elems·2) bytes.  For the 27B
// (K=4, n_lin=48, state 48·128·128 fp32 = 3 MiB, conv 10240·3 fp16 = 60 KiB):
//   ≈ 4 · 48 · (3 MiB + 60 KiB) ≈ 588 MiB.  K=6 ≈ 882 MiB.
struct Qwen35SpecCheckpoint {
    float*      ckpt_state = nullptr;   // [K, n_lin, state_elems_per_layer] fp32
    sycl::half* ckpt_conv  = nullptr;   // [K, n_lin, conv_elems_per_layer]  fp16
    uint32_t    K = 0;
    uint32_t    n_lin = 0;
    uint64_t    state_elems = 0;        // per layer (= dn.state_elems_per_layer())
    uint64_t    conv_elems  = 0;        // per layer (= dn.conv_elems_per_layer())
    DeviceAllocator* alloc = nullptr;

    // Allocate for K positions sized to `dn`'s geometry. Idempotent if already
    // sized for the same K/geometry.
    std::string init(DeviceAllocator& a, const DeltaNetState& dn, uint32_t K_);
    void        free_storage() noexcept;
    ~Qwen35SpecCheckpoint() noexcept { free_storage(); }

    // Commit exactly `accepted` verify positions: copy the snapshot taken AFTER
    // position (accepted-1) into the live DeltaNetState (recurrent + conv, all
    // linear layers). `accepted` in [1, K]. KV is handled by the caller via
    // kv.set_length (positional; no copy needed).
    std::string commit_to_n(sycl::queue& q, DeltaNetState& dn,
                            uint32_t accepted) const;
};

// ===========================================================================
// MTP (NextN) draft head — the native blk.<n_transformer_layers> layer.
//
// A full-attention transformer block plus the NextN concat projection
// (eh_proj over [enorm(embed) ; hnorm(h_i)]) + shared lm_head. Drafts K tokens
// autoregressively given (h_i, prev_tok). Owns its weights + a private 1-layer
// KV cache (positional; re-seeded from pos 0 every draft burst — the burst is
// self-contained causal context). Loaded ONLY when spec-decode is requested
// (~0.6 GB), so it does not change the default memory footprint.
//
// Lifted verbatim from the validated tools/ie_qwen35_spec.cpp MtpHead (the
// lossless + net-1.20× oracle). Concat order is [enorm(e) ; hnorm(h)].
// ===========================================================================
struct MtpHead {
    bool loaded = false;
    DeviceAllocator* alloc = nullptr;

    // geometry (mirrors the dense config)
    uint32_t H = 0, HD = 0, N_q = 0, N_qg = 0, N_kv = 0, F = 0, rope_n = 0,
             vocab = 0, mtp_blk = 0;
    float eps = 0, rope_theta = 0;
    uint32_t n_q_heads = 0, n_kv_heads = 0, max_ctx = 0;

    // weights (all device fp16 / fp32 like the dense path; matrices dequanted
    // to F16 [K,N] at load so they ride gemv_fp16).
    sycl::half *eh_proj = nullptr, *w_attn_q = nullptr, *w_attn_k = nullptr,
               *w_attn_v = nullptr, *w_attn_out = nullptr;
    sycl::half *w_ffn_gate = nullptr, *w_ffn_up = nullptr, *w_ffn_down = nullptr,
               *w_lm_head = nullptr;
    float *enorm = nullptr, *hnorm = nullptr, *shead_norm = nullptr,
          *attn_norm = nullptr, *post_attn_norm = nullptr,
          *attn_q_norm = nullptr, *attn_k_norm = nullptr;
    void* te_dev = nullptr; DType te_dtype = DType::kCount;

    // private 1-layer KV (sized max_ctx — only the per-burst window is ever used)
    sycl::half *mtp_kc = nullptr, *mtp_vc = nullptr;

    // per-step T=1 scratch
    sycl::half *d_h=nullptr, *d_e=nullptr, *d_hn=nullptr, *d_en=nullptr,
               *d_cat=nullptr, *d_x=nullptr, *d_xn=nullptr, *d_qg=nullptr,
               *d_q=nullptr, *d_gate=nullptr, *d_k=nullptr, *d_v=nullptr,
               *d_ao=nullptr, *d_blk=nullptr, *d_fg=nullptr, *d_fu=nullptr,
               *d_fh=nullptr, *d_logits1=nullptr;
    int32_t *d_pos1=nullptr, *d_tok1=nullptr;

    std::vector<void*> owned;

    ~MtpHead();
    // Allocate all weights from the GGUF + scratch. `g` must stay alive (only
    // read here). Returns error text on failure. Idempotent: a 2nd call is a
    // no-op once `loaded`.
    std::string load(DeviceAllocator& a, const GgufReader& g,
                     const Qwen35Config& cfg, uint32_t max_ctx_);

    // internal forward helpers (used by draft())
    void embed(sycl::queue& q, int32_t tok);
    void build_x(sycl::queue& q, const sycl::half* h_src, int32_t e_tok);
    void run_layer(sycl::queue& q, int32_t pos);

    // Draft K tokens from (h_last device-ptr, tn). Fills `out` with g_1..g_K,
    // growing the MTP KV from pos = p_base (use 0 for a fresh burst).
    void draft(sycl::queue& q, const sycl::half* h_last, int32_t tn,
               uint32_t p_base, uint32_t K, std::vector<int32_t>& out);
};

class Qwen35DenseModel {
public:
    Qwen35DenseModel() = default;
    ~Qwen35DenseModel();
    Qwen35DenseModel(const Qwen35DenseModel&) = delete;
    Qwen35DenseModel& operator=(const Qwen35DenseModel&) = delete;

    // Load all weights to device. Hard-fails (returns error text) on any
    // dtype it cannot handle; Q5_K (attn_k/attn_output) and Q8_0 (ssm_out)
    // are dequanted to F16 [K,N] at load via dequant_q5_K_to_Bt/q8_0_to_Bt.
    // Skips blk.<n_transformer_layers> (the NextN/MTP layer).
    std::string load(DeviceAllocator& alloc, const GgufReader& g,
                     const Qwen35Config& cfg);
    const Qwen35Config& config() const noexcept { return cfg_; }

    // Hybrid caches: kv holds the 16 full-attn layers, dn holds the 48 linear.
    // Spec-decode verify primitive (both default null = unchanged behaviour):
    //   * all_logits   != null → run lm_head on ALL T positions → [T, vocab]
    //     (instead of last-token only); out_logits still gets the last position.
    //   * hidden_pre_norm != null → copy the residual stream BEFORE output_norm
    //     (the [T, hidden] tensor the native MTP head consumes as h_i) into it.
    //   * ckpt != null → CHECKPOINT MODE (spec-decode): run each linear layer's
    //     conv + DeltaNet recurrence as T single-token scans, snapshotting the
    //     {recurrent, conv} state AFTER each position into `ckpt` (per layer).
    //     Output is byte-identical to the normal T-batched path (the scans are
    //     sequential; T×(T=1) == 1×(T=K)). Lets the caller commit n accepted
    //     positions with NO rollback re-forward. ckpt must be init()'d for ≥ T.
    //     Default-off → normal decode/prefill/crown path is untouched.
    sycl::event forward(sycl::queue& q,
                        const int32_t* input_ids, uint32_t T, uint32_t start_pos,
                        KvCache& kv, DeltaNetState& dn,
                        sycl::half* out_logits,
                        sycl::half* all_logits = nullptr,
                        sycl::half* hidden_pre_norm = nullptr,
                        Qwen35SpecCheckpoint* ckpt = nullptr);

    std::string ensure_workspace(uint32_t max_T);
    std::string ensure_attn_partials(uint32_t max_ctx);

    // ---- MTP self-speculative decode (single-GPU, GREEDY only) ----
    // Load the native MTP/NextN head (blk.<n_transformer_layers>) on demand. Only
    // called when --spec is requested; ~0.6 GB. Idempotent. `g` must stay alive.
    std::string load_mtp_head(const GgufReader& g, uint32_t max_ctx);
    bool mtp_loaded() const noexcept { return mtp_.loaded; }

    // One emitted token of the spec loop. `on_token(id)` returns false to abort
    // (the loop stops after the current committed batch is delivered up to id).
    using SpecEmit = std::function<bool(int32_t)>;

    // Run MTP self-speculative GREEDY decode. Caller has already prefilled the
    // prompt into kv/dn (the model's hybrid caches) up to `start_pos`, and passes
    // the target's pre-output_norm hidden at the last prompt position (h_last,
    // device fp16 [hidden]) plus the target argmax there (tn) — both produced by a
    // forward(.., hidden_pre_norm=) over the prompt. Emits up to max_new committed
    // tokens through `emit` IN ORDER (lossless: identical to plain greedy). Stops
    // early if emit() returns false. K = draft length (4 optimal). Bit-for-bit
    // identical token stream to plain greedy decode on §1-stable prompts.
    std::string spec_generate(sycl::queue& q, KvCache& kv, DeltaNetState& dn,
                              sycl::half* h_last, int32_t tn,
                              uint32_t start_pos, uint32_t max_new, uint32_t K,
                              const SpecEmit& emit);

    // SoA-Q6 fast-decode GEMV dispatch: T==1 → quantize act to Q8_1 + int-dot
    // gemv_q6_soa_q8; T≥2 → dequant SoA→fp16 + gemm_fp16. `A`/`y` are device
    // buffers, K/N the weight geometry. Requires w.active().
    // `x_q8_pre` (IE_QWEN35_QUANT_HOIST): if non-null AND T==1, the activation
    // `A` has already been quantized to a block_q8_1x stream (by the fused
    // norm+quant); skip the internal quantize_q8_1 and int-dot against it. Pure
    // launch-elimination, bit-identical (same rounded-fp16 inputs).
    sycl::event gemv_q6soa_T(sycl::queue& q, const sycl::half* A,
                             const Q6SoaW& w, sycl::half* y, uint32_t T,
                             const void* x_q8_pre = nullptr);
    // SoA-Q8_0 dispatch: T==1 → int-dot gemv_q8_0_soa_q8 (~80% BW, vs the F16-
    // expanded path's 2× bytes); T≥2 → dequant SoA→fp16 + gemm_fp16 (no batched
    // Q8 int-dot kernel yet). Requires w.active().
    sycl::event gemv_q8soa_T(sycl::queue& q, const sycl::half* A,
                             const Q8SoaW& w, sycl::half* y, uint32_t T,
                             const void* x_q8_pre = nullptr);
    // SoA-Q4 dispatch (mirrors gemv_q6soa_T): T==1 → quantize act to Q8_1 +
    // int-dot gemv_q4_soa_q8; T≥2 → dequant SoA-Q4→fp16 + gemm_fp16 (reuses the
    // q6soa_bt_/q6soa_c_ prefill scratch). Requires w.active().
    sycl::event gemv_q4soa_T(sycl::queue& q, const sycl::half* A,
                             const Q4SoaW& w, sycl::half* y, uint32_t T,
                             const void* x_q8_pre = nullptr);
    void set_dump_prefix(std::string p) { dump_prefix_ = std::move(p); }

private:
    DeviceAllocator* alloc_ = nullptr;
    Qwen35Config cfg_;
    std::string dump_prefix_;

    void*  token_embd_ = nullptr;  DType token_embd_dtype_ = DType::kCount;
    float* output_norm_ = nullptr;
    void*  output_ = nullptr;      DType output_dtype_ = DType::kCount;
    Q6SoaW output_soa_;            // SoA-Q6 lm_head (when output is Q6_K + SoA on)
    bool   q6_soa_on_ = false;     // gate: IE_QWEN35_NO_Q6_SOA opt-out
    bool   q4_soa_on_ = false;     // gate: IE_QWEN35_Q4_SOA opt-IN (default OFF →
                                   // default forward byte-identical to AoS Q4_K)
    bool   batched_verify_on_ = true;  // small-T batched int-dot verify (spec-decode);
                                       // opt-out IE_QWEN35_NO_BATCHED_VERIFY
    bool   quant_hoist_on_ = false;    // gate: IE_QWEN35_QUANT_HOIST opt-IN (default
                                       // OFF → default forward byte-identical). When
                                       // ON, a norm's Q8_1 stream is computed once
                                       // (fused norm byproduct / one explicit quant)
                                       // and reused by all its SoA-GEMV consumers,
                                       // eliminating the redundant per-GEMV quants.
    bool   fa2_prefill_on_ = true;     // tiled FA-2 for T>1 attn (prefill + spec
                                       // verify); opt-out IE_QWEN35_NO_FA2_PREFILL

    // Per-device prefill scratch for the SoA-Q6 → fp16 dequant→gemm path (grown
    // on demand; reused across projections — single GPU, one gen at a time).
    sycl::half* q6soa_bt_ = nullptr; uint64_t q6soa_bt_cap_ = 0;
    float*      q6soa_c_  = nullptr; uint64_t q6soa_c_cap_  = 0;

    std::vector<Qwen35LayerWeights> layers_;   // size n_transformer_layers (64)
    std::vector<void*> owned_;

    MtpHead mtp_;   // native MTP/NextN draft head (loaded only when --spec)

    // Workspace (lazy, sized for max_T). Superset of the dense path's plus the
    // DeltaNet scratch (conv I/O, q/k/v predelta fp32, g/beta, state out).
    uint32_t    ws_T_ = 0;
    sycl::half* ws_x_ = nullptr;          // [T, H]
    sycl::half* ws_x_normed_ = nullptr;   // [T, H]
    sycl::half* ws_attn_block_ = nullptr; // [T, H]  (attn or DeltaNet out-proj result)
    // full-attn scratch
    sycl::half* ws_qg_ = nullptr;         // [T, n_q·head·2] joint Q|gate
    sycl::half* ws_q_ = nullptr;          // [T, n_q·head]
    sycl::half* ws_gate_ = nullptr;       // [T, n_q·head] (full-attn per-head gate)
    sycl::half* ws_k_ = nullptr;          // [T, n_kv·head]
    sycl::half* ws_v_ = nullptr;          // [T, n_kv·head]
    sycl::half* ws_attn_out_ = nullptr;   // [T, n_q·head]
    // DeltaNet scratch
    sycl::half* ws_qkv_ = nullptr;        // [T, conv_channels 10240]
    sycl::half* ws_conv_ = nullptr;       // [T, conv_channels] (post conv+silu)
    sycl::half* ws_dn_z_ = nullptr;       // [T, d_inner 6144]  gated-norm z
    float*      ws_qpre_ = nullptr;       // [T, n_v·128] fp32  q predelta (post repeat)
    float*      ws_kpre_ = nullptr;       // [T, n_v·128] fp32  k predelta (post repeat)
    float*      ws_vpre_ = nullptr;       // [T, n_v·128] fp32  v
    float*      ws_g_ = nullptr;          // [T, n_v 48] fp32   (g after compute_g_beta)
    float*      ws_beta_ = nullptr;       // [T, n_v 48] fp32   (beta)
    float*      ws_dn_out_ = nullptr;     // [T, n_v·128] fp32  recurrence output
    float*      ws_qrep_ = nullptr;       // [T, n_v·128] fp32  q post-repeat (16→48)
    float*      ws_krep_ = nullptr;       // [T, n_v·128] fp32  k post-repeat
    sycl::half* ws_alpha_h_ = nullptr;    // [T, n_v] fp16  alpha (compacted) → compute_g_beta_h16
    sycl::half* ws_beta_h_ = nullptr;     // [T, n_v] fp16  beta (compacted)
    sycl::half* ws_alpha64_ = nullptr;    // [T, pad64(n_v)] fp16  alpha proj (batched gemm out)
    sycl::half* ws_beta64_ = nullptr;     // [T, pad64(n_v)] fp16  beta proj
    // FFN scratch
    sycl::half* ws_ffn_gate_ = nullptr;   // [T, ffn]
    sycl::half* ws_ffn_up_ = nullptr;     // [T, ffn]
    sycl::half* ws_ffn_h_ = nullptr;      // [T, ffn]
    int32_t*    ws_positions_ = nullptr;  // [T]
    void*       ws_q8_ = nullptr;         // decode int-dot Q8_1 scratch
    // Quant-hoist (IE_QWEN35_QUANT_HOIST) persistent Q8_1 streams for the two
    // decode norms whose SoA-GEMV consumers re-quantize the SAME vector:
    //   ws_attn_norm_q8_: pre-attn norm (attn_qkv/attn_gate or attn_q/k/v)
    //   ws_ffn_in_q8_:    post-attn/pre-FFN norm (ffn_gate + ffn_up)
    // Sized (H/32)*sizeof(block_q8_1x) — decode T==1 only. nullptr when gate OFF.
    void*       ws_attn_norm_q8_ = nullptr;
    void*       ws_ffn_in_q8_    = nullptr;
    float*      ws_attn_partials_ = nullptr;
    uint32_t    ws_attn_partials_ctx_ = 0;
};

}  // namespace ie
