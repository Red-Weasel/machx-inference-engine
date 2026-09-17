// include/ie/deepseek4.hpp — DeepSeek-V4-Flash (`deepseek4`) weight binding
// (Phase 1) and the assembled forward pass (Phase 5).
//
// `DeepSeek4Model` below is unchanged Phase-1 territory: host-side binding and
// shape validation over the GGUF mmap, no device memory, no compute.
//
// `DeepSeek4Runtime` (bottom of this file) is Phase 5: it uploads the
// always-resident weight set, drives the Phase 2/3/4 primitives through the
// reference's DecoderLayer and Model, and streams the 120.393 GB routed-expert
// set through `include/ie/expert_stream.hpp`.
//
// Every pointer below is a non-owning view into the GgufReader's mmap (the
// reader merges all four shards into one tensor table, so a caller only opens
// shard 1). The model owns nothing and frees nothing; it is invalidated when
// the reader closes.
//
// Structure verified 2026-08-01 against a complete 1328-tensor parse of
// DeepSeek-V4-Flash-0731-UD-Q3_K_XL (see docs/deepseek4/21_tensor_manifest_verified.md).
// Where this header and that document disagree the header wins: the document
// records `attn_compressor_*` at ONE shape, the file carries TWO (see
// DeepSeek4LayerKind).
#pragma once

#include "ie/deepseek4_cache.hpp"
#include "ie/expert_stream.hpp"
#include "ie/gguf.hpp"
#include "ie/model_config.hpp"

#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace ie {

// Per-layer variation, computed ONCE from the config so the binder never
// branches on a bare layer index. Two independent schedules drive it:
//
//   compress_ratios[L]  (deepseek4.attention.compress_ratios, 46 entries; the
//                        3-entry surplus past block_count is ignored here)
//     0   → no long-range compressor at all           (layers 0-1)
//     4   → CSA block: compressor at 2*head_dim + the lightning indexer
//           (reference DeepseekV4CSACompressor: kv_proj/gate_proj → 2*head_dim,
//            position_bias [compress_rate, 2*head_dim])   (even layers >= 2)
//     128 → HCA block: compressor at head_dim, NO indexer
//           (reference DeepseekV4HCACompressor: kv_proj/gate_proj → head_dim,
//            position_bias [compress_rate, head_dim])     (odd layers)
//
//   hash_layer_count    (deepseek4.hash_layer_count = 3)
//     L < count → `ffn_gate_tid2eid` present, `exp_probs_b.bias` ABSENT
//                 (reference DeepseekV4HashRouter — a token-id→expert-id table)
//     otherwise → the reverse.
//
// The two schedules are independent: layer 2 is both a hash-router layer and a
// CSA/indexer layer.
struct DeepSeek4LayerKind {
    uint32_t compress_ratio = 0;  // compress_ratios[L], verbatim
    uint32_t compressor_dim = 0;  // 2*head_dim (ratio 4) / head_dim (ratio 128) / 0
    bool     has_compressor = false;
    bool     has_indexer    = false;   // == (compress_ratio == 4)
    bool     hash_router    = false;   // == (L < hash_layer_count)
};

// One layer's bound tensors. A pointer is null IF AND ONLY IF `kind` says the
// tensor is absent for this layer; load() hard-errors on any other absence, and
// equally on a tensor that is present when `kind` says it must not be.
struct DeepSeek4Layer {
    DeepSeek4LayerKind kind;

    // DENSE below marks a weight bound at whatever dtype the FILE declares
    // (`Binder::bind_dense`, gated on ds4_dense_uploadable) because the two
    // shipped quants disagree on it: UD-Q3_K_XL says Q8_0 and UD-Q8_K_XL says
    // BF16 — except indexer_proj, which is F32 in Q3 and BF16 in Q8. Everything
    // annotated with a concrete dtype is bound at exactly that dtype in both.

    // -- attention (all 43 layers) ---------------------------------------
    const GgufTensorInfo* attn_norm      = nullptr;  // F32   [hidden]
    const GgufTensorInfo* attn_q_a       = nullptr;  // DENSE [hidden, q_lora_rank]
    const GgufTensorInfo* attn_q_a_norm  = nullptr;  // F32   [q_lora_rank]
    const GgufTensorInfo* attn_q_b       = nullptr;  // DENSE [q_lora_rank, n_q_heads*head_dim]
    const GgufTensorInfo* attn_kv        = nullptr;  // DENSE [hidden, head_dim]
    const GgufTensorInfo* attn_kv_a_norm = nullptr;  // F32   [head_dim]
    const GgufTensorInfo* attn_sinks     = nullptr;  // F32   [n_q_heads]
    const GgufTensorInfo* attn_output_a  = nullptr;  // DENSE [hidden, o_lora_rank*o_groups]
    const GgufTensorInfo* attn_output_b  = nullptr;  // DENSE [o_lora_rank*o_groups, hidden]

    // -- long-range compressor (kind.has_compressor) ----------------------
    const GgufTensorInfo* compressor_kv   = nullptr; // DENSE [hidden, compressor_dim]
    const GgufTensorInfo* compressor_gate = nullptr; // DENSE [hidden, compressor_dim]
    const GgufTensorInfo* compressor_ape  = nullptr; // F32   [compressor_dim, compress_ratio]
    const GgufTensorInfo* compressor_norm = nullptr; // F32   [head_dim]

    // -- lightning indexer (kind.has_indexer) -----------------------------
    const GgufTensorInfo* indexer_q_b             = nullptr; // DENSE [q_lora_rank, idx_heads*idx_head_dim]
    const GgufTensorInfo* indexer_proj            = nullptr; // DENSE [hidden, idx_heads]  (F32 in Q3, BF16 in Q8)
    const GgufTensorInfo* indexer_compressor_kv   = nullptr; // DENSE [hidden, 2*idx_head_dim]
    const GgufTensorInfo* indexer_compressor_gate = nullptr; // DENSE [hidden, 2*idx_head_dim]
    const GgufTensorInfo* indexer_compressor_ape  = nullptr; // F32   [2*idx_head_dim, compress_ratio]
    const GgufTensorInfo* indexer_compressor_norm = nullptr; // F32   [idx_head_dim]

    // -- MoE (all 43 layers) ----------------------------------------------
    const GgufTensorInfo* ffn_norm       = nullptr;  // F32  [hidden]
    const GgufTensorInfo* ffn_gate_inp   = nullptr;  // DENSE [hidden, n_experts] (BF16 in the
                                                     // Unsloth 0731 files, F32 in the Vision-Exp
                                                     // GGUF; dequantised to fp32 either way)
    // Routed experts. dtype is NOT fixed by role or by model: it is read from
    // each tensor's own GGUF type_id. In UD-Q3_K_XL gate/up are IQ3_XXS on 42
    // layers and MXFP4 on blk.26 (Unsloth mixed precision), down is MXFP4
    // everywhere — but nothing here assumes that.
    const GgufTensorInfo* ffn_gate_exps  = nullptr;  // [hidden, expert_ffn, n_experts]
    const GgufTensorInfo* ffn_up_exps    = nullptr;  // [hidden, expert_ffn, n_experts]
    const GgufTensorInfo* ffn_down_exps  = nullptr;  // [expert_ffn, hidden, n_experts]
    const GgufTensorInfo* ffn_gate_shexp = nullptr;  // DENSE [hidden, expert_ffn]
    const GgufTensorInfo* ffn_up_shexp   = nullptr;  // DENSE [hidden, expert_ffn]
    const GgufTensorInfo* ffn_down_shexp = nullptr;  // DENSE [expert_ffn, hidden]
    // Exactly one of these two is bound, per kind.hash_router.
    const GgufTensorInfo* ffn_gate_tid2eid = nullptr; // I32 [n_experts_used, vocab]
    const GgufTensorInfo* exp_probs_b      = nullptr; // F32 [n_experts]
    // Vision-Exp files only (all 43 layers, hash layers included): the router
    // bias the reference applies to IMAGE tokens instead of `exp_probs_b`
    // (inference/model.py Gate.forward, `bias_vl`). Null on 0731 files.
    const GgufTensorInfo* exp_probs_b_vl   = nullptr; // F32 [n_experts] or null

    // -- hyper-connections (all 43 layers) --------------------------------
    // mix = (2 + hc_count) * hc_count = 24; fn is the transpose of the
    // reference's [mix, hc_count*hidden]. See docs/deepseek4/23_hyper_connections_solved.md.
    const GgufTensorInfo* hc_attn_fn    = nullptr;   // F32 [hc_count*hidden, mix]
    const GgufTensorInfo* hc_attn_base  = nullptr;   // F32 [mix]
    const GgufTensorInfo* hc_attn_scale = nullptr;   // F32 [3]
    const GgufTensorInfo* hc_ffn_fn     = nullptr;   // F32 [hc_count*hidden, mix]
    const GgufTensorInfo* hc_ffn_base   = nullptr;   // F32 [mix]
    const GgufTensorInfo* hc_ffn_scale  = nullptr;   // F32 [3]
};

// Non-layer tensors.
struct DeepSeek4Globals {
    // token_embd/output are DENSE (per-tensor dtype): Q8_0 / Q6_K in UD-Q3_K_XL,
    // both BF16 in UD-Q8_K_XL.
    const GgufTensorInfo* token_embd      = nullptr; // DENSE [hidden, vocab]
    const GgufTensorInfo* output          = nullptr; // DENSE [hidden, vocab]  (untied lm_head)
    const GgufTensorInfo* output_norm     = nullptr; // F32   [hidden]
    const GgufTensorInfo* output_hc_fn    = nullptr; // F32   [hc_count*hidden, hc_count]
    const GgufTensorInfo* output_hc_base  = nullptr; // F32   [hc_count]
    const GgufTensorInfo* output_hc_scale = nullptr; // F32   [1]
};

class DeepSeek4Model {
public:
    // Binds every tensor the config implies and cross-checks that NOTHING in the
    // file was left over. Returns "" on success, else a diagnostic naming the
    // offending tensor. Never warns-and-continues, never silently skips.
    //
    // `g` must be the reader for shard 1 of the split GGUF (GgufReader::open
    // pulls in the siblings itself); it must outlive this object.
    std::string load(const GgufReader& g, const DeepSeek4Config& cfg);

    const DeepSeek4Config&             config()  const noexcept { return cfg_; }
    const DeepSeek4Globals&            globals() const noexcept { return globals_; }
    const std::vector<DeepSeek4Layer>& layers()  const noexcept { return layers_; }
    // Number of distinct GGUF tensors bound by the last successful load().
    uint64_t n_bound() const noexcept { return n_bound_; }

    // The per-layer descriptor, exposed so callers (and tests) can derive layer
    // variation from the config exactly the way load() does. Returns "" on
    // success, or a diagnostic if compress_ratios is too short / carries a
    // ratio this loader does not model.
    static std::string layer_kind(const DeepSeek4Config& cfg, uint32_t layer,
                                  DeepSeek4LayerKind& out);

private:
    DeepSeek4Config             cfg_;
    DeepSeek4Globals            globals_;
    std::vector<DeepSeek4Layer> layers_;
    uint64_t                    n_bound_ = 0;
};

// ===========================================================================
// PHASE 5 — the assembled forward pass
// ===========================================================================
//
// PRECISION.  Activations are fp32 end to end, matching the Phase 2/3 primitives
// (which are fp32 because the reference is: the hyper-connection casts to
// float, the softmaxes are `dtype=torch.float32`, the indexer scorer casts both
// operands).  The always-resident WEIGHTS keep their packed GGUF form on the
// device wherever this file carries an element-wise device decoder for their
// dtype (Q8_0, Q6_K — see `ds4_dense_keeps_packed`), and are dequantised once at
// load into fp16 [N, K] otherwise; both forms are read by the same fp32-activation,
// fp32-accumulate kernels.  The routed experts stay packed (IQ3_XXS / MXFP4) and
// go through the Phase-4 device GEMVs.
//
// WHY PACKED.  Expanding the resident set to fp16 cost 14.785 GB against the
// 7.81 GB the Q3 GGUF actually stores, and every one of those bytes came out of
// the VRAM expert arena (docs/deepseek4/40_ALPHA_OMEGA_PHASE_PLAN.md, "the gap
// this exposed").  Nothing about the packed path is an approximation: the device
// readers reproduce `ie::ref` (dequant_ref.hpp, bit-exact vs ggml) operation for
// operation, so they are STRICTLY more accurate than the fp16 copy they replace,
// which rounded every dequantised weight to 11 significant bits.
//
// WHAT IS DELIBERATELY NOT HERE.  No batching over sequences, no tokenizer
// (Phase 6), no tensor parallelism.  `forward()` runs one sequence on one
// device; `Ds4Options::device_ordinal` picks it.
// ---------------------------------------------------------------------------

// One always-resident dense matrix, row-major [N, K], read as
// y[n] = Σ_k x[k] · w[n][k].  That is the GGUF layout for every 2-D weight in
// this model ({shape[0]=K, shape[1]=N} stores rows of length K), so neither
// upload transposes anything.
//
// EXACTLY ONE of `w`, `p` and `q8` is non-null:
//   w   fp16 [N, K]        — the dtype had no device decoder and was expanded.
//   p   packed GGUF bytes  — N rows of `ds4_dense_row_bytes(dt, K)`, verbatim
//                            file bytes, decoded element-wise in the kernel.
//   q8  Q8_0-SoA           — the weight was REQUANTISED at load (see
//                            `ds4_dense_requantises`).  `q8` is the int8 plane
//                            [N][K] and `q8d` the fp16 scale plane [N][K/32],
//                            both inside the single allocation `q8` owns; only
//                            `q8` is freed.  Layout and rationale:
//                            include/ie/ds4_decode_gemv.hpp.
struct Ds4Dense {
    sycl::half* w   = nullptr;
    uint8_t*    p   = nullptr;
    int8_t*     q8  = nullptr;
    sycl::half* q8d = nullptr;      // interior of `q8`'s allocation — never freed
    sycl::half* q8dt = nullptr;     // the SAME scales kb-major [K/32][N] — the
                                    // orientation oneDNN's weight-decompression
                                    // matmul reads (attr scales are consumed in
                                    // PLAIN layout; the strides on the scales md
                                    // are ignored, measured 2026-08-08). Interior
                                    // of `q8`'s allocation too — never freed.
    DType       dt = DType::kF16;   // kF16 iff `w`; the file's own dtype iff `p`;
                                    // kQ8_0 iff `q8`
    uint32_t    K = 0, N = 0;
};

// ---------------------------------------------------------------------------
// The always-resident packing policy — ONE place, so the loader's byte
// accounting and any residency arithmetic derived from a tensor table cannot
// drift apart.
// ---------------------------------------------------------------------------

// True iff a 2-D always-resident weight of this dtype and row length is kept in
// its packed GGUF form on the device.  A dtype qualifies only when this
// translation unit carries an element-wise device decoder for it that is
// bit-exact against `ie::ref` at every T; everything else expands to fp16,
// which is correct and simply costs 2 B/element.
//
// `K` (the row length) must be a whole number of blocks: an element-wise
// decoder addresses `k` inside a block, so a ragged final block is not
// addressable.  Every DS4 K is a multiple of 256, but the check is enforced,
// not assumed.
bool ds4_dense_keeps_packed(DType dt, uint64_t K) noexcept;

// Bytes one packed row of `K` elements occupies.  0 if `dt` is not packable.
uint64_t ds4_dense_row_bytes(DType dt, uint64_t K) noexcept;

// Device bytes ONE always-resident 2-D weight [K, N] costs under that policy:
// the packed GGUF size when it stays packed, else K*N*2 for the fp16 expansion.
//
// NOTE — THIS IS THE PACKED-OR-EXPANDED POLICY ONLY.  A BF16 weight that
// `upload_dense` REQUANTISES (`ds4_dense_requantises` below) costs
// `ds4_dense_requant_bytes` instead, which is 1.0625 B/element rather than the 2
// this function reports.  The two are deliberately separate functions: this one
// answers "what does the dtype policy cost", which is what the residency
// spreadsheets in deepseek4_residency_test §4/§8c ask, and changing it would
// silently rewrite their arithmetic.  A caller that wants the LOADER's actual
// byte count must consult both.
uint64_t ds4_dense_device_bytes(DType dt, uint64_t K, uint64_t N) noexcept;

// ---------------------------------------------------------------------------
// REQUANTISATION — the third residency form, and the only LOSSY one
// ---------------------------------------------------------------------------
//
// WHY.  In UD-Q8_K_XL every always-resident dense projection is BF16 in the
// file, and BF16 has no device decoder, so `upload_dense` expanded it to fp16 —
// 2 B/element.  At decode those projections are pure weight streaming at M=1:
// 6.92 GB per card per token (7.45 GB counting the LM head), which a tuned GEMV
// already runs at 91.8% of this card's streaming roofline.  Compute was done;
// the only remaining lever was to move fewer bytes.
//
// Requantising BF16 to Q8_0 at load costs 1.0625 B/element instead of 2.
// MEASURED cache-cold on a B70 over the real per-card call set (511 dense calls
// per card per token): 7.453 GB / 13.565 ms -> 3.959 GB / 7.869 ms, i.e. 1.88x
// fewer bytes for 1.72x less time, 5.70 ms/card/token.  It also returns ~3.5 GB
// per card to the expert arena, which for this quant is the binding constraint.
//
// WHAT IT COSTS, STATED PLAINLY.  This is LOSSY and it is the first lossy thing
// done to the always-resident set.  BF16 carries an 8-bit mantissa; Q8_0 carries
// 8-bit integers with one fp16 scale per 32 elements, so each weight moves by up
// to d/2 with d = amax/127 over its block.  deepseek4_dense_gate_test bounds
// that against the BF16 ORIGINAL — not against the quantised bytes, which would
// measure nothing — and prints observed against bound.
//
// It is not out of family: this model's routed experts are already MXFP4 (4-bit)
// and the UD-Q3_K_XL sibling ships these very tensors as Q8_0/Q6_K through the
// packed path.  It is NOT applied blindly, though; see `Ds4DenseQuant`.

// The dtype-and-shape half of the policy: true iff a weight of this dtype and
// row length CAN be requantised when IE_DS4_DENSE_Q8 is enabled (default on).
// BF16, F16 and F32 use the same fp32 row source; the latter two also cover the
// Vision-Exp source-preserving GGUF. Packed dtypes are excluded. K must be
// nonzero and a whole number of 32-element Q8_0 blocks.
bool ds4_dense_requantises(DType dt, uint64_t K) noexcept;

// Device bytes a requantised [K, N] weight occupies: the int8 plane plus the
// fp16 scale plane twice (n-major and kb-major), K*N + 2*N*(K/32)*2 = 1.125*K*N.
uint64_t ds4_dense_requant_bytes(uint64_t K, uint64_t N) noexcept;

// The per-CALL-SITE half of the policy.  `upload_dense` takes this and it is
// stated at every call site rather than inferred, because "which tensors may go
// to 8 bits" is a numerical judgement about what CONSUMES the result, and no
// dtype or shape can express it.
//
//   kAuto      requantise when `ds4_dense_requantises` allows it.  The default
//              for ordinary projections whose output flows into arithmetic.
//   kKeepWide  never requantise; take the packed-or-fp16 policy unchanged.
//              Used where the measured win is absent AND the file's own packed
//              form is an acceptable resting place — `token_embd`, whose gather
//              reads one 8 KB row per token and whose packed residency is 1.06 GB
//              of deliberate VRAM saving.
//   kWideF16   never requantised AND never packed: materialise fp16.  It exists
//              because `kKeepWide` resolves to the FILE's dtype, so a site that
//              wants one behaviour gets two: fp16 on an F16/F32-dense GGUF, the
//              file's packed blocks on a Q8_0-dense one, and with them the
//              per-element `dense_packed` kernel (measured 2026-09-12 on the
//              ggml-org MXFP4 file: the three indexer projections cost 935.7 ms
//              of a 4360.7 ms two-card prefill GPU budget, 7.43 ms/call against
//              0.24 for `dense_f16`; removing them took prefill +16.6%).
//
//              WHAT IT IS AND IS NOT.  On an F16/BF16-source weight it is a
//              precision choice: it declines the requantiser's real loss.  On a
//              Q8_0-source weight it is a PERFORMANCE choice that COSTS a little
//              precision, and the comment this replaced had that backwards.
//              `dense_packed` decodes Q8_0 as d(fp16) * qs(int8) accumulated in
//              fp32 — at most 19 significant bits into 24 — so the packed route
//              is BIT-EXACT to the file's own values, while the fp16 image rounds
//              them to 11 bits (~2^-11 relative, against the file's own ~2^-8
//              quantisation error, so roughly an eighth more of an error that is
//              already there).  Widening cannot recover precision the file never
//              stored.  Taken purely on precision, a Q8_0 source is better served
//              by `kAuto`: the requantiser recovers d and qs bit-for-bit, the
//              decode GEMV keeps activations and accumulator in fp32, and it
//              SAVES VRAM (1.0625 vs 2 B/element) — at the cost of a per-chunk
//              conversion at T > 1.  Choose `kWideF16` for the prefill kernel it
//              buys, not for precision it does not.
enum class Ds4DenseQuant { kAuto, kKeepWide, kWideF16 };

// Where a dense weight comes to rest on the device.  `upload_dense` has three
// branches and this names which one it takes, so the policy is testable without
// a GPU or a model.
//
//   kPacked       the file's own blocks, decoded per element by `dense_packed`
//   kRequantised  the Q8_0-SoA planes, read by the decode GEMV / oneDNN s8 route
//   kFp16         materialised fp16, read by `dense_f16`
enum class Ds4DenseRoute { kPacked, kRequantised, kFp16 };

// The placement decision as a pure function.  `K` is the full contraction length,
// `Ks` this card's column slice (they differ only under a kCols split).
Ds4DenseRoute ds4_dense_route(DType dt, uint64_t K, uint64_t Ks, Ds4DenseQuant quant) noexcept;

// Requantises `nrows` rows of `K` fp32 values into the Q8_0-SoA layout.
// `qs` must hold nrows*K int8 and `d` nrows*(K/32) halves.
//
// The block quantiser is ggml's, with ONE deliberate improvement: ggml computes
// the reciprocal from the fp32 scale and then stores the scale as fp16, so its
// decode multiplies by a scale the quantiser never saw.  These bytes are ours —
// they are never written to a file and never compared against ggml — so the
// scale is rounded to fp16 FIRST and the reciprocal taken from the rounded
// value.  The decoded weight is then exactly d16*round(w/d16), the error is
// exactly |d16|/2 per element with no second fp16 term, and the gate's bound is
// that expression rather than that expression plus a fudge.
void ds4_requantise_q8_soa(const float* src, uint32_t K, uint32_t nrows,
                           int8_t* qs, sycl::half* d) noexcept;

// The whole upload: allocates `d`'s single Q8_0-SoA allocation for THIS CARD's
// [Ks, Ns] slice of a [K, N] weight and fills both planes, pulling rows from
// `next_rows(r0, nr, out)` in host-sized chunks.  `next_rows` yields nr FULL
// rows of K floats beginning at this card's row r0 — it owns the row (N) slice,
// this function owns the column (K) slice at [K0, K0+Ks).  Reports the bytes
// allocated through `bytes`.  `upload_dense` is one caller and
// deepseek4_dense_gate_test §8 is the other; there is no third copy of this
// arithmetic, which matters because no GGUF on the development box carries a
// BF16 dense tensor and the gate is therefore the only thing that runs it.
std::string ds4_build_q8_soa(sycl::queue& q, uint32_t K, uint32_t K0, uint32_t Ks, uint32_t Ns,
                             Ds4Dense& d, uint64_t& bytes,
                             const std::function<std::string(uint32_t, uint32_t,
                                                             std::vector<float>&)>& next_rows);

// ---------------------------------------------------------------------------
// NON-EXPERT TENSOR PARALLELISM — how one always-resident 2-D weight [K, N] is
// divided between the cards.
// ---------------------------------------------------------------------------
//
//   kMirror  every card holds the whole weight.  The only option when
//            n_cards == 1, and the right answer for every weight whose output
//            is consumed by something that needs it whole (see the decision
//            table in tests/unit/deepseek4_residency_test.cpp §8c, which is
//            executable: `kSplitRows`/`kSplitCols` there ARE this policy, and
//            §15i checks that they predict a real load's bytes exactly).
//   kRows    the OUTPUT dim N is partitioned; card c holds rows
//            [c*N/n, (c+1)*N/n).  A row is contiguous in EVERY layout this
//            loader uses (packed rows are `ds4_dense_row_bytes` apart, fp16 rows
//            are K halves apart), so no quantisation block is ever cut and — the
//            load-bearing part — NO SUM IS SPLIT.  Card c's result is a SLICE of
//            the mirrored result, bit for bit, not a partial of it.
//   kCols    the CONTRACTION dim K is partitioned; card c holds, of every row,
//            elements [c*K/n, (c+1)*K/n).  Card c's result is a PARTIAL SUM that
//            is meaningless until the other cards' are added to it.  For a
//            packed dtype K/n must be a whole number of quantisation blocks;
//            `upload_dense` refuses BY NAME rather than rounding.
//
// Every kCols weight therefore costs a cross-card reduction, and with
// `can_access_peer == 0` that reduction goes through pinned host memory.  Which
// weights earn one is an arithmetic question, answered per tensor in
// docs/deepseek4/34; the two that do are `attn_output_b` (whose reduction is
// new) and `ffn_down_shexp` (whose partial folds into the routed-expert
// reduction that already existed, so it costs nothing).
enum class Ds4DenseSplit { kMirror, kRows, kCols };

// This card's share of the attention heads, output groups and shared-expert
// intermediate columns.  All three are derived from ONE split of the head axis,
// because they have to agree: output group g consumes attention heads
// [g*NH/G, (g+1)*NH/G), so a card that owns groups [g0, g0+gc) must own exactly
// the heads those groups read or `attn_output_a` would be indexing another
// card's attention output.  `load_prepare` checks that identity rather than
// assuming it.
struct Ds4AttnSlice {
    uint32_t n_cards = 1, card = 0;
    uint32_t nh0 = 0, nhc = 0;   // query heads      [nh0, nh0+nhc)
    uint32_t g0  = 0, gc  = 0;   // output groups    [g0,  g0+gc)
    uint32_t ef0 = 0, efc = 0;   // shared-expert intermediate columns
    // false means MIRRORED — nhc/gc/efc are then the model's full extents and
    // every weight below is uploaded whole.  Always false when n_cards == 1.
    bool     split = false;
};

// True iff `DeepSeek4Runtime::upload_dense` can turn a weight of this dtype into
// a device weight at all — i.e. it either stays packed (a device decoder exists)
// or `dequant_rows` can expand it to fp16.
//
// WHY THIS IS A SET AND NOT A CONSTANT.  The same tensor ROLE carries different
// dtypes in different quants of the same model: UD-Q3_K_XL stores every
// attention projection, shared expert and `token_embd` as Q8_0 and `output` as
// Q6_K, while UD-Q8_K_XL stores all of them — and `indexer.proj`, which is F32
// in Q3 — as BF16.  A loader that binds a per-role dtype constant can therefore
// only ever load one of the two files.  The loader binds each tensor's own
// `type_id` and gates it on THIS predicate, which keeps a mis-declared dtype a
// hard, named error instead of an "accept anything" path.
bool ds4_dense_uploadable(DType dt) noexcept;

// Device-resident, non-expert weights of one layer.
struct Ds4LayerRT {
    DeepSeek4LayerKind kind;

    float*   attn_norm      = nullptr;   // [H]
    Ds4Dense q_a, q_b, kv, o_a, o_b;
    float*   q_a_norm       = nullptr;   // [q_lora_rank]
    float*   kv_a_norm      = nullptr;   // [head_dim]
    float*   sinks          = nullptr;   // [n_q_heads]

    Ds4Dense comp_kv, comp_gate;
    float*   comp_ape       = nullptr;   // [compress_ratio, compressor_dim]
    float*   comp_norm      = nullptr;   // [head_dim]

    Ds4Dense idx_q_b, idx_proj, idx_kv, idx_gate;
    float*   idx_ape        = nullptr;   // [compress_ratio, 2*index_head_dim]
    float*   idx_norm       = nullptr;   // [index_head_dim]

    float*   ffn_norm       = nullptr;   // [H]
    float*   router_w       = nullptr;   // [n_experts, H]   fp32 (small, fragile chain)
    float*   router_bias    = nullptr;   // [n_experts] or null (hash layers)
    float*   router_bias_vl = nullptr;   // [n_experts] or null (text-only files) — image-token routing bias
    int32_t* tid2eid        = nullptr;   // [vocab, top_k]  or null
    Ds4Dense sh_gate, sh_up, sh_down;

    float*   hc_attn_fn     = nullptr;   // [mix, hc*H]
    float*   hc_attn_base   = nullptr;   // [mix]
    float*   hc_attn_scale  = nullptr;   // [3]
    float*   hc_ffn_fn      = nullptr;
    float*   hc_ffn_base    = nullptr;
    float*   hc_ffn_scale   = nullptr;

    // Routed experts: the slot layout the streaming arena stores them in.
    Ds4SlotLayout slot{};
};

struct Ds4Options {
    uint32_t device_ordinal      = 0;      // which B70
    // Hidden-dim expert tensor parallelism (docs/deepseek4/33 §2.1, Option A).
    // `n_cards` runtimes, each on its own device, each holding
    // `moe_intermediate_size / n_cards` of EVERY expert; `card` says which
    // slice THIS runtime owns.  1/0 is the single-device engine, byte for byte.
    //
    // load() honours this fully: the plan, the pinned arena and the uploaded
    // bytes are all the card's slice.  `forward()` on a SINGLE runtime does NOT
    // and never will — the sliced `down` GEMV yields a PARTIAL sum that has to
    // be reduced against the other cards' (`ds4_tp_reduce_host`), and one
    // runtime has no way to reach them.  It therefore REFUSES when n_cards > 1.
    // `DeepSeek4TpRuntime` (below) is the thing that drives n_cards runtimes in
    // lockstep and performs that reduction; it is the ONLY supported way to run
    // a sliced load.
    //
    // These two fields are set BY THE ORCHESTRATOR, per card.  A caller building
    // a Ds4TpOptions must leave them at 1/0; `DeepSeek4TpRuntime::load` refuses
    // rather than overwriting them, because silently overwriting a field the
    // caller set is how a two-card run ends up with two copies of card 0.
    uint32_t n_cards             = 1;
    uint32_t card                = 0;
    // Also TP-split the ALWAYS-RESIDENT (non-expert) set, not just the routed
    // experts.  Ignored when n_cards == 1 — there is nothing to split.
    //
    // WHAT IT SPLITS, AND WHAT IT COSTS.  `attn_q_b`, `attn_output_a`,
    // `ffn_gate_shexp` and `ffn_up_shexp` split by rows (no reduction — see
    // Ds4DenseSplit::kRows); `attn_output_b` and `ffn_down_shexp` split by
    // columns and yield partials.  The shared expert's partial folds into the
    // routed-expert reduction that already ran, so the WHOLE cost of this option
    // is ONE additional host-staged reduction per layer, of T*hidden fp32.
    // Everything else stays mirrored, per tensor, for reasons recorded in
    // deepseek4_residency_test §8c — most sharply `attn_kv`, which cannot be
    // head-split at all because `head_count_kv == 1`.
    //
    // WHY IT IS A CHOICE AND NOT A CONSTANT.  docs/deepseek4/33 §2.5 measured
    // the two configurations on a scheduler harness and got opposite answers per
    // quant: Q8 is DMA-bound, so the slots the split buys are worth more than
    // the reduction costs; Q3 at its residency was already compute-bound, where
    // the extra slots buy little and the reduction is pure loss.  That
    // measurement rests on an ASSUMED 0.70 ms/layer non-expert compute time
    // (doc 33 §7.1), so it is not strong enough to hard-wire either way.  The
    // default is on because the memory arithmetic — which does NOT depend on
    // that assumption — favours it for both quants.
    bool     split_non_expert    = true;
    // VRAM to leave for the streamed expert cache.  0 = $DS4_CACHE_MB, else
    // "whatever is left after the resident set", floored at one slot per layer.
    uint64_t expert_cache_bytes  = 0;
    // Device bytes something ELSE already holds on the card(s) and the expert
    // cache derivation must leave alone (the Vision-Exp tower the engine
    // uploads before this runtime plans its arena).  Ignored when
    // expert_cache_bytes / $DS4_CACHE_MB set the budget explicitly.
    uint64_t vram_reserved_extra = 0;
    // 0 = $DS4_SLOTS, else derive from expert_cache_bytes.  Both env fallbacks
    // exist because `EngineOptions` carries no equivalent field, so a served
    // engine could otherwise not reach a configuration `ie-ds4-bench --slots`
    // can — see the DS4_* precedence blocks in load_prepare().
    uint32_t slots_per_layer     = 0;
    // Pin only the first `pin_layers` layers' experts (0 = all 43).  A partial
    // pool is HONEST, not a fallback: `forward()` fails loudly if the router
    // selects a layer whose experts were never pinned.
    uint32_t pin_layers          = 0;
    // Hard ceiling on the pinned HOST bytes, i.e. on the routed experts that are
    // not permanently VRAM-resident.  Exceeding it fails the load; it is never
    // met by pinning less than the model needs.
    //
    // 0 resolves, most explicit first: $DS4_PIN_CAP_GB, then a cap DERIVED FROM
    // LIVE MEMORY (`ds4_host_pin_cap_live` — this is the default), and only when
    // /proc/meminfo cannot be read at all does kDs4HostPinCapDefault apply.  The
    // derived cap is the one that can see the page cache, which is the variable
    // behind the recorded systemd-oomd kill; see ds4_host_pin_cap_live.
    uint64_t host_pin_cap_bytes  = 0;
    // Evictable slots per layer; every VRAM slot above this is static and its
    // host copy is never allocated.  0 = $DS4_STREAM_SLOTS if set, else
    // ALL slots streaming (2026-09-01: LRU + all-streaming = +35% decode; see
    // load_prepare), and always floored at n_experts_used so one token's
    // routing can never exceed the streaming partition.
    uint32_t min_stream_slots    = 0;
    // The order in which experts claim the STATIC (never-evicted, VRAM-only)
    // partition.  Entry 0 gets the first static slot, and the experts past
    // `static_slots` fall through to the pinned host arena and stream.
    //
    // EMPTY = index order, 0,1,2,...  — which is what this was HARD-WIRED to
    // before.  That default was justified on the grounds that `noaux_tc` routing
    // trains toward equalised utilisation, so any static set is as good as any
    // other; but the supporting benchmark ran on a randomly-initialised model,
    // where utilisation is uniform BY CONSTRUCTION and the test could not have
    // failed.  The default is retained because nothing measured beats it (see
    // the note below), but it is now a CHOICE the caller can override rather
    // than a property of the code.
    //
    // What the trained weights actually say (measured 2026-08-02 by reading
    // `blk.*.exp_probs_b.bias` out of the real GGUF — 40 tensors, layers 3..42;
    // layers 0-2 are hash-routed and carry no bias):
    //   * The bias is added to the score only INSIDE the top-k argsort
    //     (deepseek4_ops.hpp), so it lives in sqrt-softplus score units and is
    //     invariant under adding a constant to all 256 entries.  "std as % of
    //     mean" — the figure previously quoted as 0.2-0.6% — is therefore not a
    //     meaningful statistic: the mean is pure gauge freedom.  Measured over
    //     all 40 layers it is 0.21%-1.13% anyway, not 0.2-0.6%.
    //   * The defensible figures are absolute: per-layer centred std 0.019-0.128
    //     (median 0.050), centred range 0.126-1.688 (median 0.465), MAD 0.007.
    //     The distribution is a tight spike plus a one-sided left tail — 25 of
    //     10240 (expert,layer) slots sit more than 0.5 below their layer median.
    //   * Crucially this cannot answer the residency question EITHER WAY.  The
    //     balancing bias records how much correction the router NEEDED, not the
    //     residual imbalance, so a tight bias is equally consistent with "already
    //     balanced" and "small nudge sufficed".  Utilisation is measured by
    //     counting selections, which needs a real forward pass.
    //   * One genuinely useful negative result: the suppressed experts are
    //     layer-idiosyncratic (174 suppression events over 126 distinct ids
    //     against a uniform expectation of 126.4; adjacent-layer Jaccard 0.006).
    //     So there is no globally hot or cold expert set for index order to get
    //     systematically wrong — and equally none for any static order to exploit.
    // Hence: mechanism now exists and is tested; default unchanged; and the
    // claim that index order is *justified* is withdrawn, not replaced.
    //
    // Must be a permutation of [0, n_experts) when non-empty; load() refuses a
    // partial, duplicated or out-of-range order by name rather than filling in.
    std::vector<uint32_t> expert_priority;
    // The PER-LAYER order, which is the one that matters and the one the field
    // above cannot express.  `expert_priority_layers[L]` is layer L's full
    // permutation; supplying it wins over `expert_priority`.  Empty = fall
    // through to the single-permutation field, then the file, then index order.
    //
    // WHY IT EXISTS.  The comment below concluded index order was near-optimal.
    // That conclusion was drawn from the HASH layers 0-2 alone, because they are
    // the only ones computable without a forward pass — and it does not
    // generalise to the 40 top-k layers, which is 93% of the model.  Measured on
    // a 512-step real decode (2026-08-03, $DS4_EXPERT_TRACE + the §17 scorer in
    // deepseek4_residency_test), scoring a ranking on decode steps it was not
    // built from:
    //     index order        31.5% of selections already resident
    //     one global order   42.5%
    //     PER-LAYER order    77.9%
    // Layers 3-42 concentrate 83.8-95.8% of their decode selections (median
    // 90.8%) in their OWN top 80 of 256; layer 4's hot set is not layer 40's, so
    // any single permutation averages the skew away.  It costs no extra pinned
    // host RAM — every layer still pins the same COUNT — and it took the modelled
    // decode expert traffic from 533.0 to 199.5 MB/card/token.
    std::vector<std::vector<uint32_t>> expert_priority_layers;
    // A MEASURED order, read from a `ds4-expert-priority 1` file (see
    // ds4_expert_priority_read_layers, which accepts BOTH the single-permutation
    // file and the per-layer form).  Consulted only when both fields above are
    // empty; empty here in turn falls back to $DS4_EXPERT_PRIORITY_FILE, and
    // then to index order.  This is the hook that lets a profiling run feed a
    // real utilisation ranking back into a later load with NO code change.
    //
    // WHAT STATIC ANALYSIS OF THE REAL FILES SAYS ABOUT THE DEFAULT (measured
    // 2026-08-02 over UD-Q3_K_XL's own tensors; see the report for the commands):
    //   * Layers 0-2 hash-route through `ffn_gate_tid2eid`, a frozen
    //     [vocab, top_k] I32 table, so their expert usage is EXACTLY computable
    //     from a token-frequency distribution with no forward pass at all.  The
    //     table is not flat — per-expert entry counts run 2678..3397 against a
    //     3030.0 uniform, and they fall monotonically with expert id
    //     (pearson(id, count) = -0.977; block means 9767 -> 8404 from experts
    //     [0,32) to [224,256)).
    //   * But that skew runs THE SAME WAY index order does, so index order
    //     already captures 97.5-99.0% of the entire available gain: at 48 static
    //     slots, index order takes 20.0204% of layer-0-2 picks against 20.0524%
    //     for the count-optimal order and 18.7500% for a random one.  The whole
    //     residual is +0.032 percentage points, on 3 of 43 layers.
    //   * Under a skewed token distribution the tables would say much more, but
    //     this engine has measured no such distribution, and assuming one is how
    //     the previous residency justification went wrong.
    // Hence: index order stays the default, now for a QUANTIFIED reason rather
    // than an assumed one, and the file hook is how a measurement replaces it.
    std::string expert_priority_file;
    // Count routed-expert selections per layer during forward().  Off by default
    // — it is a measurement tool, not part of inference.  $DS4_EXPERT_PROFILE,
    // when set to a path, turns this on AND makes release() dump the profile
    // there, which is what closes the measure -> feed-back loop without a code
    // change on either side.
    bool     profile_experts     = false;
    // DEFAULT OFF as of 2026-08-03, on measured evidence.  Speculation was
    // designed for a link with idle time; the fetch pipeline in expert_stream.cpp
    // removed that idle time, so speculative bytes now come straight out of
    // demand fetches on a saturated link.  Real model, 2x B70, Q8, ctx 4096:
    //
    //                       prefetch ON   prefetch OFF
    //   decode tg4096          8.02          12.90     <- the reason
    //   prefill pp2048        85.21          97.04
    //   prefill pp512         89.22          72.15     <- the cost, single chunk
    //   pp512 dma_busy_frac    1.107          0.698
    //
    // It measured 1,900 speculations for 18 hits (0.95%) even before the
    // pipeline: a chunk touches each expert ONCE, so a streaming slot yields no
    // reuse to speculate toward, and layer L+1's demand fetches churn the whole
    // streaming partition ~16x before most groups run.
    //
    // The pp512 column is a genuine loss, not noise to ignore: that is a SINGLE
    // chunk from a cold cache, the one case where speculation pays. The right fix
    // is to speculate only while the cache is cold rather than all-or-nothing;
    // until someone builds that, decode is what makes this engine usable and it
    // wins by 1.61x. Set `prefetch = true` (or drop `--no-prefetch`) to restore.
    bool     prefetch            = false;  // idle-time speculative prefetch
    // Drives the routed-expert block with the TOKEN-CHUNKED residency loop
    // instead of the expert-grouped one (see `moe_expert_grouped`).  It is the
    // reference the default is gated bit-for-bit against, in ONE process, which
    // is why it is an option and not only the $IE_DS4_MOE_TOKEN_CHUNK env var:
    // a static-once getenv cannot be flipped between two runtimes in one test.
    // $IE_DS4_MOE_TOKEN_CHUNK and $IE_DS4_MOE_TOKEN_MAJOR force it on at load.
    bool     moe_token_chunk     = false;
    // How many INDEPENDENT SEQUENCES this runtime keeps attention state for.
    // 1 (the default) is the historical behaviour exactly: one set of per-layer
    // caches, one conversation at a time.
    //
    // Each additional slot costs one more copy of the PERSISTENT per-layer cache
    // state — the sliding ring, the compressor window buffers, the overlap
    // slices and the growing compressed-entry arrays.  It costs NO additional
    // transient scratch (that is shared, see Ds4CacheScratch) and no additional
    // workspace, because slots are stepped one at a time: this option makes a
    // scheduler able to interleave sequences, it does not by itself batch them
    // into one forward.  Deliberately named `seq_slots` and not `max_seqs` so it
    // cannot be misread as `max_seq`, which is the unrelated per-call token
    // bound two lines below.
    //
    // $DS4_SEQ_SLOTS overrides it at load, the same way $DS4_SLOTS overrides
    // slots_per_layer.
    uint32_t seq_slots           = 1;
    uint32_t max_seq             = 4096;   // largest T in one forward() call
    // Largest total context the mask / kv scratch is sized for.  The compressed
    // entry list grows at 1/compress_rate of the token rate and is unbounded in
    // principle, so this is a HARD sizing bound: forward() returns an error
    // naming the overflow rather than writing past the workspace.
    uint32_t max_context         = 8192;
};

// One card's cache state at a depth, in pinned host memory (Phase L3).
struct Ds4HostSnapshot {
    std::vector<Ds4LayerHostSnap> layers;
    uint32_t depth = 0;
    uint64_t bytes() const noexcept {
        uint64_t n = 0;
        for (const auto& l : layers) n += l.bytes();
        return n;
    }
};

class DeepSeek4Runtime {
public:
    DeepSeek4Runtime() = default;
    ~DeepSeek4Runtime() { release(); }
    DeepSeek4Runtime(const DeepSeek4Runtime&)            = delete;
    DeepSeek4Runtime& operator=(const DeepSeek4Runtime&) = delete;

    // Uploads the always-resident set and fills the pinned expert arena from
    // `m`'s bound tensors.  `m` (and the GgufReader behind it) must outlive
    // this object.  Returns "" or a diagnostic.  Reports progress to stderr
    // because filling 120 GB takes minutes, and a silent load looks like a hang.
    std::string load(const DeepSeek4Model& m, const Ds4Options& opt);
    void        release() noexcept;

    // One forward pass over `T` tokens starting at absolute position `pos0`.
    // `logits` receives [T, vocab] fp32 on the host (only the LAST row is
    // computed when `last_only`, in which case it receives [1, vocab]).
    // The per-layer caches carry context across calls, so decode is
    // forward(&tok, 1, pos, ...) repeatedly.
    std::string forward(const int32_t* input_ids, uint32_t T, uint32_t pos0,
                        float* logits, bool last_only = true);

    void reset_context() noexcept;    // clears every layer cache of EVERY slot

    // ---- Vision-Exp image blocks (docs/deepseek4/70_VISION_EXP_PORT_PLAN.md P3) ----
    // Stage one image block: `rows` [n, hidden] f32 replaces the embedding rows
    // of prompt positions [t0, t0+n).  The block's IMAGE_START row sits at
    // t0 + start_off (the rows before it are compress-pad rows) and its
    // IMAGE_END row at t0+n-1.  Every row of the block routes with the
    // `exp_probs_b_vl` bias; the rows from IMAGE_START to IMAGE_END attend
    // bidirectionally inside the sliding window (reference
    // get_window_topk_idxs_visible).  The whole block must be prefilled in ONE
    // forward() call (the reference asserts the same); a chunk that straddles a
    // block is refused by name.  Rows are copied.  Staging persists across
    // reset_context() — the engine re-arms right after its pos==0 reset — and
    // is dropped by clear_vision() or the next set_vision() after a clear.
    std::string set_vision(const float* rows, uint32_t t0, uint32_t n, uint32_t start_off);
    void        clear_vision() noexcept { vis_spans_.clear(); }

    // ---- sequence slots (Ds4Options::seq_slots) -----------------------------
    //
    // A slot is one conversation's attention state.  `forward()` always acts on
    // the ACTIVE slot; `select_seq` says which that is.  Nothing else in the
    // runtime is per-slot — the workspaces, the expert cache and the weights are
    // shared — so switching slots is a host-side index write and costs nothing.
    //
    // What this buys is CORRECTNESS for interleaved serving: two sequences can
    // be advanced in any order without either one seeing the other's window
    // state.  What it does NOT buy is throughput: two slots stepped one after
    // the other read the active weights twice, exactly as two sequential
    // requests do today.  Batching them into one forward is a separate piece of
    // work (docs/deepseek4/60_CONTINUOUS_BATCHING.md, piece 3) and this is its
    // prerequisite, not a substitute for it.
    uint32_t seq_slots() const noexcept { return seq_slots_; }
    uint32_t active_seq() const noexcept { return seq_; }
    // Returns "" on success, or a diagnostic naming the bad slot.  Refuses
    // rather than clamping: a silently-clamped slot id is two conversations
    // sharing one cache, which produces plausible text and no error.
    std::string select_seq(uint32_t slot) noexcept;
    // Clears ONE slot's layer caches, leaving every other slot untouched.
    std::string reset_seq(uint32_t slot) noexcept;
    // Live state / allocated bytes of ONE slot (Σ over its layers).  The
    // no-argument accessors below report the ACTIVE slot, which is what they
    // reported before slots existed.
    uint64_t seq_state_bytes(uint32_t slot, uint32_t elem_bytes = 4) const noexcept;
    uint64_t seq_allocated_bytes(uint32_t slot) const noexcept;
    // Transient scratch shared by every slot — Σ over the layers.  Reported
    // apart from the per-slot figures because adding a slot does not add any of
    // it, and a capacity model that folds it into the per-sequence cost
    // overstates that cost by roughly two orders of magnitude at prefill widths.
    uint64_t cache_scratch_bytes() const noexcept;

    // ---- prompt cache: one snapshot of the ACTIVE slot ----------------------
    //
    // `reset_context()` clears the layer caches and nothing else, which is the
    // proof that the layer caches ARE the whole of this runtime's per-context
    // state: capture them and you have captured the conversation.
    //
    // Single endpoint, following Gemma4 (`gemma4.hpp:87`) rather than a trie —
    // the common chat pattern is "this conversation, one turn longer", and one
    // endpoint serves it.  The TOKENS that produced the snapshot are NOT stored
    // here; the caller owns them and does its own longest-prefix match, because
    // under tensor parallelism every card holds the same tokens and only one
    // copy should exist.
    //
    // VRAM: `snapshot_bytes()`, which is the per-sequence figure — 22.55 MiB
    // constant + 6.88 kB per token of context across all layers (fp16 compressed
    // entries since docs/deepseek4/73 Phase 2) — and it is
    // drawn from the SAME reserve the expert arena is derived from.  Snapshot
    // endpoints and live sequence slots must therefore be budgeted together.
    //
    // `snapshot_context` frees any prior snapshot FIRST, so a failed capture
    // leaves no snapshot rather than a stale one, and returns a diagnostic
    // instead of throwing when VRAM is short (caller: skip the cache, carry on).
    std::string snapshot_context();
    // Restores the snapshot into the ACTIVE slot and returns the context depth
    // it was taken at (0 = nothing restored, caller must prefill from scratch).
    // The caller continues its prefill at pos0 = that depth; positions are
    // absolute and the matched tokens are identical, so this is exact.
    uint32_t    restore_context();
    void        free_snapshot() noexcept;
    uint32_t    snapshot_depth() const noexcept { return snap_depth_; }
    uint64_t    snapshot_bytes() const noexcept;
    // Host-resident form (docs/deepseek4/72 Phase L3): the active slot's state
    // into pinned host memory (all layers; waits for the copies), and back.
    // restore_from_host returns the depth restored (0 = nothing: empty or failed,
    // and then the slot is reset so the caller prefills from scratch).
    std::string snapshot_to_host(Ds4HostSnapshot& out);
    uint32_t    restore_from_host(const Ds4HostSnapshot& in);

    sycl::queue&        queue() noexcept { return *q_; }
    const Ds4HostArena& arena() const noexcept { return arena_; }
    Ds4ExpertCache&     cache() noexcept { return cache_; }
    uint64_t            resident_bytes() const noexcept { return resident_bytes_; }
    // VRAM the packed-residency policy did NOT spend, i.e. what expanding every
    // resident weight to fp16 would have cost on top of resident_bytes().
    uint64_t            resident_saved_bytes() const noexcept { return resident_saved_; }
    uint32_t            layers_pinned() const noexcept { return pinned_layers_; }
    // Routed-expert slot bytes this card packed out of the file during load().
    // Its whole point is the load TRACE: sampled per layer while the single file
    // walk runs, two cards' counters climbing together is the observable that
    // distinguishes the interleaved drive from "card 0 fully, then card 1".
    uint64_t            packed_expert_bytes() const noexcept { return packed_bytes_; }
    // The static/streamed split this load actually used.
    const Ds4ResidencyPlan& residency() const noexcept { return plan_.card; }
    // ...and the whole-box view of it, including the other cards' share.
    const Ds4TpResidencyPlan& tp_residency() const noexcept { return plan_; }
    // The intermediate-dimension slice of every expert this card holds.
    const Ds4ExpertSlice& expert_slice() const noexcept { return slice_; }
    // This card's share of the ALWAYS-RESIDENT set.  `split == false` means the
    // whole of it is mirrored here (always so when n_cards == 1).
    const Ds4AttnSlice&   attn_slice() const noexcept { return aslice_; }
    // The residency priority order this load actually resolved: entry i is the
    // expert holding static VRAM slot i, and everything from `static_slots` on
    // is host-pinned and streams.  Always a full permutation of [0, n_experts),
    // even when the caller supplied nothing (then it is the identity).
    const std::vector<uint32_t>& expert_priority() const noexcept { return prio_; }
    // ...and the per-layer form, which is what residency actually uses.  Entry
    // [L][i] is the expert holding layer L's static VRAM slot i.  Always
    // n_layers full permutations, even when every layer was given the same one.
    const std::vector<std::vector<uint32_t>>& expert_priority_layers() const noexcept {
        return prio_layer_;
    }
    // Static VRAM slot of expert `e` IN LAYER `L`, or -1 if it streams there.
    // The inverse of expert_priority_layers()[L] over its first `static_slots`
    // entries.  Layer-qualified because residency is now a per-layer decision.
    int32_t static_slot_of(uint32_t L, uint32_t e) const noexcept {
        const uint64_t i = uint64_t(L) * cfg_.n_experts + e;
        return e < cfg_.n_experts && i < prio_rank_.size() ? prio_rank_[i] : -1;
    }
    // Where the resolved order came from, for a log line and for a test that has
    // to prove the file was actually consulted rather than defaulted past.
    enum class PriorityOrigin { kIndex, kOptionField, kFile };
    PriorityOrigin     priority_origin() const noexcept { return prio_from_; }
    const std::string& priority_source()  const noexcept { return prio_src_; }

    // The measured expert-selection counts, if `Ds4Options::profile_experts` (or
    // $DS4_EXPERT_PROFILE) turned counting on.  `active()` is false otherwise.
    const Ds4ExpertProfile& expert_profile() const noexcept { return profile_; }

    // The routed-expert accumulator, [T, hidden] fp32 on THIS card's device, as
    // it stands after the last layer of the last forward().  Not part of the
    // inference API: it is exposed because it is the ONE buffer hidden-dim
    // expert-TP makes a partial of, so comparing it between a one-card and a
    // two-card run is the sharpest test available that the split is a partition
    // and not an approximation.  Reading it needs this runtime's own queue.
    const float* moe_accumulator() const noexcept { return ws_moe_; }

    // Per-forward instrumentation, all MEASURED (profiling events / host clock),
    // never modelled.
    struct Timing {
        double   wall_seconds  = 0.0;
        double   dma_seconds   = 0.0;   // transfer-queue busy time
        uint64_t bytes_fetched = 0;
        uint64_t hits = 0, misses = 0;
    };
    const Timing& last_timing() const noexcept { return timing_; }

    // Host time THIS card spent BLOCKED on its own compute queue during the last
    // forward(), split by the site that blocked and counted per site.  Measured
    // with steady_clock immediately around the drain, so it is the host's view:
    // "the GPU had not finished and this thread could not submit".  It is the
    // only way to tell a decode step that is GPU-bound from one that is
    // submission-bound, because both show the same wall.
    //
    // The four sites are ALL of them, and each is load-bearing for a different
    // reason (see the comment at each drain in deepseek4.cpp):
    //   router — the routing-ids D2H the host chunker has to read;
    //   moe    — the per-group eviction-safety barrier against the transfer queue;
    //   post   — the one per-layer drain that makes the next layer's cache grow safe;
    //   other  — the prologue/epilogue copies and the compressor-position H2Ds.
    // Reset by forward_prologue(), so it describes ONE forward.
    struct HostWaits {
        double   router = 0.0, moe = 0.0, post = 0.0, other = 0.0;
        uint64_t n_router = 0, n_moe = 0, n_post = 0, n_other = 0;
        double   total() const noexcept { return router + moe + post + other; }
        uint64_t count() const noexcept { return n_router + n_moe + n_post + n_other; }
    };
    const HostWaits& last_host_waits() const noexcept { return waits_; }

    // Rows the device-side logits buffer currently holds.  Exposed for the same
    // reason `moe_accumulator()` is: "this buffer is sized by what is USED, not
    // by max_seq" is a claim about an allocation, and an allocation is invisible
    // from outside unless something states its size.  1 after load; grown only
    // by a `last_only == false` forward, which is the only caller that needs
    // more.  deepseek4_sched_gate_test §3 gates both halves of that.
    uint32_t logits_rows() const noexcept { return ws_logits_T_; }

    // ---- context cost, read off the per-layer caches themselves ----
    //
    // Both are Σ over the layers of THIS card, at the context depth the caches
    // are at right now, and both come from `Ds4LayerCache`'s own accounting
    // (`state_bytes` / `allocated_bytes`) rather than from a per-token constant.
    // The distinction matters for a benchmark: `kv_state_bytes` is the LIVE
    // state (what a decode step actually reads), `kv_allocated_bytes` is what
    // the growable buffers currently hold, which only ever grows and includes
    // the slack a geometric growth policy leaves behind.  0 before load().
    //
    // `elem_bytes` is the element width the state would be QUOTED at — 2 for the
    // bf16 reference / fp16 engine figure, 4 for what this fp32 engine actually
    // allocates.  It is a parameter and not a constant because the published
    // 6 880 B/token growth rate is the 2-byte number, and silently reporting the
    // 4-byte one against it would look like a 2x regression that is not there.
    //
    // WITH SEQUENCE SLOTS: `kv_state_bytes` / `kv_growing_bytes` describe the
    // ACTIVE slot — they answer "how deep is this conversation", which is what
    // they always answered.  `kv_allocated_bytes` is WHOLE-RUNTIME (every slot
    // plus the shared scratch), because it always answered "what does this
    // runtime hold".  At the default one slot all three are unchanged.
    uint64_t kv_state_bytes(uint32_t elem_bytes = 4) const noexcept;
    uint64_t kv_allocated_bytes() const noexcept;
    // The part of `kv_state_bytes` that GROWS with context — the compressed
    // entry arrays only.  `kv_state_bytes - kv_growing_bytes` is therefore the
    // constant part (the sliding ring, the window buffers, the overlap slices),
    // and the two together are what a context-scaling measurement needs: a
    // single total cannot distinguish "the constant is large" from "the growth
    // rate is large", and those have opposite fixes.
    uint64_t kv_growing_bytes(uint32_t elem_bytes = 4) const noexcept;
    // Total tokens the layer-0 cache has seen since the last reset_context().
    // The context depth every other per-context figure is indexed by.
    uint64_t context_length() const noexcept;

private:
    friend class DeepSeek4TpRuntime;

    // Uploads one 2-D always-resident weight under ds4_dense_keeps_packed and
    // charges resident_bytes_ with what it actually spent.  `split` says which
    // axis (if any) this card takes a slice of; see Ds4DenseSplit.  A slice that
    // does not divide, or that would cut a quantisation block, is refused by
    // name — never rounded, never silently mirrored.
    // `quant` says whether this particular tensor may be requantised to Q8_0 —
    // a numerical judgement about the CONSUMER, so it is stated per call site.
    //
    // `stage_at_load` says whether a REQUANTISED weight of this size must have
    // its fp16 T > 1 staging buffer reserved while a failure is still a named
    // load error.  True for every per-layer projection, which prefill submits at
    // T > 1 on every chunk.  FALSE for the LM head alone: `forward_epilogue`
    // passes `TL = last_only ? 1 : T`, so generation only ever asks it at T == 1
    // and reserving its 529 MB per card would spend VRAM that generation never
    // touches — VRAM the expert-arena derivation does not know is gone, because
    // the staging buffer is not charged to resident_bytes_.  Teacher-forced
    // scoring (tools/ie-ds4-ppl) does ask at T > 1; it grows the buffer then,
    // and ds4_scratch_grow says so on stderr if it cannot.
    std::string upload_dense(const GgufTensorInfo* ti, Ds4Dense& d,
                             Ds4DenseSplit split = Ds4DenseSplit::kMirror,
                             Ds4DenseQuant quant = Ds4DenseQuant::kAuto,
                             bool stage_at_load = true);
    std::string upload_layer(const DeepSeek4Layer& w, Ds4LayerRT& rt);
    std::string alloc_workspace();
    // The kv-axis extent every kv-shaped workspace is allocated to, and the
    // bound layer_forward_pre checks the live kv length against.  ONE formula,
    // asked by both, so a change to the sizing cannot leave the check behind.
    // Split into its two segments (docs/deepseek4/73 Phase 1): the sliding
    // buffer holds at most sliding_window-1+T rows, the compressed caches at
    // most max_context/min_ratio+1 entries; only the latter grows with --ctx.
    uint32_t    ws_kv_capacity() const noexcept;
    uint32_t    ws_sliding_capacity() const noexcept;
    uint32_t    ws_compressed_capacity() const noexcept;
    // Query rows per strip for the ctx-scaled workspaces (indexer scores,
    // block bias, concatenated mask): [ws_strip_, kv] instead of [max_seq, kv].
    // Derived once in alloc_workspace from $IE_DS4_STRIP_MB (default 512).
    uint32_t    ws_strip_rows() const noexcept;

    // ---- load(), split at the seam the TP orchestrator needs ----
    //
    // `load()` is exactly
    //     load_prepare
    //   -> for each layer { load_advise_layer; for each expert load_pack_expert }
    //   -> load_finish
    // and the single-card path calls it in that order, so this is a refactor and
    // not a second code path.
    //
    // THE SEAM IS THE FILE WALK.  `load_prepare` is everything a card can do
    // ALONE — its device, its mirrored always-resident weights, its residency
    // plan, its pinned arena and its VRAM cache — and it touches no routed-expert
    // byte.  `load_pack_expert` reads exactly one expert's slice out of the mmap
    // and lands it in this card's arena or VRAM slot.  Splitting there is what
    // lets `DeepSeek4TpRuntime` run the prepares CONCURRENTLY (one thread per
    // card) and then drive ONE walk over the file in which both cards' slices of
    // the SAME expert are packed back to back — so the expert's pages are read
    // from disk once and consumed twice, instead of once per card an hour apart.
    // Loading card 0 to completion and then card 1 was the defect: one GPU maxed
    // out while the other sat idle, and 120 GB of file defeats the page cache
    // long before the second pass arrives.
    std::string load_prepare(const DeepSeek4Model& m, const Ds4Options& opt);
    // Undo the reader's MADV_RANDOM over layer `l`'s three expert tensors.  Once
    // per layer covers the whole pass: readahead is a property of the mapping,
    // not a queued request.
    void        load_advise_layer(uint32_t l) noexcept;
    // Packs THIS card's slice of expert `e` of layer `l`.  Writes into the pinned
    // host arena, or — when `e` won a static VRAM slot — through this card's own
    // `pack_stage_` and straight into that slot, after which the host bytes are
    // dropped.  `pack_stage_` is per-runtime and never shared between cards, so
    // the interleaved drive needs no synchronisation at all around it.
    std::string load_pack_expert(uint32_t l, uint32_t e);
    std::string load_finish();

    // ---- the forward pass, split at the seams the TP orchestrator needs ----
    //
    // `forward()` is exactly prologue -> for each layer { pre; mid; post } ->
    // epilogue, and the single-card path calls it that way, so the split is a
    // refactor and not a second code path.  `DeepSeek4TpRuntime` interleaves the
    // same calls across cards and inserts a cross-card reduction at each seam.
    //
    // THERE ARE TWO SEAMS, AND EACH IS A BUFFER THAT HOLDS A PARTIAL SUM.
    //
    //   `ws_sub_` after `layer_forward_pre`.  Present ONLY when the non-expert
    //   set is TP-split (Ds4Options::split_non_expert): `attn_output_b` then
    //   contracts over this card's half of the group axis, so the attention
    //   output is a partial.  Mirrored, `ws_sub_` is already the whole answer and
    //   the orchestrator skips this reduction — which is exactly what makes the
    //   mirrored configuration cost one reduction per layer and the split one
    //   cost two.
    //
    //   `ws_moe_` after `layer_forward_mid`.  Always present when n_cards > 1:
    //   the routed experts are hidden-dim sliced on every configuration.  When
    //   the non-expert set is split the SHARED expert's `down` is sliced on the
    //   same axis and its partial is added into `ws_moe_` before this seam, so it
    //   rides the reduction that already existed and adds none of its own.
    //
    // `layer_forward_post` is what remains once both sums are whole: the FFN-site
    // hyper-connection mix, and the per-layer drain.
    std::string forward_prologue(const int32_t* input_ids, uint32_t T, uint32_t pos0);
    std::string layer_forward_pre(uint32_t L, uint32_t T, uint32_t pos0);
    std::string layer_forward_mid(uint32_t L, uint32_t T);
    std::string layer_forward_post(uint32_t L, uint32_t T);
    std::string forward_epilogue(uint32_t T, float* logits, bool last_only);
    std::string layer_forward(uint32_t L, uint32_t T, uint32_t pos0, const int32_t* input_ids);

    // ---- the routed-expert block, EXPERT-GROUPED (the default since the
    // streaming-chunk measurement below) ------------------------------------
    //
    // WHAT THE TOKEN-CHUNKED PREDECESSOR DID, AND WHY IT COST WHAT IT DID.
    // `layer_forward_mid` used to split the batch into TOKEN RANGES small enough
    // that the union of their streamed experts fit the layer's streaming
    // partition, and call `ds4_experts_forward_batched` once per range.  That
    // makes the residency budget bound the BATCH, so at
    // `kDs4MinStreamSlots == 12` a 128-token prefill became ~15-30 ranges of
    // 4-9 tokens each, and an expert routed to by two different ranges was
    // FETCHED, and its weights RE-READ by a second GEMM, once per range.
    //
    // WHAT THIS DOES INSTEAD.  The batch stays whole (bounded only by
    // `bws_.max_tokens`); it is the EXPERTS that are grouped.  One packing, one
    // gather+quantise, then for each group of at most `stream_slots()` streamed
    // experts: acquire, gate/up GEMMs, the SwiGLU chain over exactly that
    // group's packed rows, down GEMMs.  One scatter-accumulate at the end.
    //
    // Because `build_moe_packing` sorts the packed rows by ascending expert id,
    // a group taken as a consecutive run of OCCUPIED expert ids owns a
    // CONTIGUOUS row range — which is what lets the elementwise chain run per
    // group without changing a single value.
    //
    // WHY IT IS BIT-IDENTICAL, not merely close.  Every packed row's arithmetic
    // is independent of every other's: `ds4_expert_gemm_q8` keeps one
    // accumulator chain per row (gated bit-for-bit against the M==1 GEMV in
    // deepseek4_experts_test §7), the cast/SwiGLU/cast stages are elementwise,
    // and `quantize_q8_1` is blockwise over a row length that is a multiple of
    // 32 — so splitting any of them at a row boundary cannot move a rounding.
    // The only cross-row reduction is `ds4_expert_scatter_accum`, and it still
    // runs ONCE over the whole batch in ascending kslot order.  Grouping
    // therefore changes WHICH experts are resident when, and nothing else.
    //
    // It cannot do more work than the token-chunked path on any input: each
    // distinct expert is acquired exactly once per batch (the old path acquired
    // it once per range it appeared in), so acquires, H2D bytes, GEMM launches
    // and slot-arena reads are all >= the new counts and never below them.
    std::string moe_expert_grouped(uint32_t L, uint32_t t0, uint32_t nt, uint32_t EFc,
                                   float swiglu_limit);

    const DeepSeek4Model*      m_ = nullptr;
    DeepSeek4Config            cfg_{};
    Ds4Options                 opt_{};
    // Speculation policy for THIS forward pass, set once at the top of
    // forward().  The 2026-08-03 first-cold-chunk win (pp512 72.15 -> 89.22)
    // was FALSIFIED on the 2026-08-09 engine (pp512 ~110 either way — the
    // idle transfer windows are gone), so the auto rule is default OFF and
    // opt-in via IE_DS4_PREFETCH_FIRST=1; `--prefetch` (opt_.prefetch) still
    // forces speculation everywhere.  See the forward() site for the A/B.
    bool                       spec_this_pass_ = false;
    std::vector<sycl::queue>   q_store_;
    sycl::queue*               q_ = nullptr;
    std::vector<Ds4LayerRT>    rt_;
    // [seq_slots_ * n_layers], slot-major: slot s layer L is at s*n_layers + L.
    // Slot-major and not layer-major because a slot is what gets created,
    // destroyed and reset as a unit.
    std::vector<Ds4LayerCache> caches_;
    // [n_layers].  The transient working buffers every slot shares at a layer.
    std::vector<Ds4CacheScratch> scratch_;
    uint32_t                   seq_slots_ = 1;
    uint32_t                   seq_       = 0;   // active slot
    // [n_layers] when a snapshot is held, empty otherwise.  Not a slot: nothing
    // ever runs a forward against it, so it binds no scratch and allocates only
    // the live prefix of each array.
    std::vector<Ds4LayerCache> snap_;
    uint32_t                   snap_depth_ = 0;
    Ds4LayerCache&       layer_cache(uint32_t L) noexcept {
        return caches_[size_t(seq_) * rt_.size() + L];
    }
    const Ds4LayerCache& layer_cache(uint32_t L) const noexcept {
        return caches_[size_t(seq_) * rt_.size() + L];
    }
    Ds4HostArena               arena_;
    Ds4ExpertCache             cache_;
    DS4ExpertWorkspace         xws_{};
    // Expert-major batch workspace for the routed-expert block.  Sized once at
    // load for `kDs4BatchTokenCap` tokens; `layer_forward_pre` bounds each
    // streaming chunk by that cap so the packed buffers are never overrun.
    DS4ExpertBatchWs           bws_{};

    // globals
    Ds4Dense    embd_{};             // token_embd [vocab, H]
    Ds4Dense    lm_{};               // output     [vocab, H]
    float*      onorm_  = nullptr;   // [H]
    float*      hh_fn_  = nullptr;   // [hc, hc*H]
    float*      hh_base_= nullptr;   // [hc]
    float*      hh_scale_ = nullptr; // [1]

    // workspace (device)
    float *ws_streams_ = nullptr, *ws_streams2_ = nullptr;
    float *ws_post_ = nullptr, *ws_comb_ = nullptr, *ws_coll_ = nullptr;
    float *ws_norm_ = nullptr, *ws_qres_ = nullptr, *ws_q_ = nullptr, *ws_kv_ = nullptr;
    float *ws_attn_ = nullptr, *ws_grp_ = nullptr, *ws_sub_ = nullptr;
    float *ws_mask_ = nullptr, *ws_bias_ = nullptr, *ws_maskc_ = nullptr;
    float *ws_cos_ = nullptr, *ws_sin_ = nullptr, *ws_ccos_ = nullptr, *ws_csin_ = nullptr;
    float *ws_kcos_ = nullptr, *ws_ksin_ = nullptr;
    float *ws_cmp_ = nullptr, *ws_pool_ = nullptr;
    float *ws_iq_ = nullptr, *ws_iw_ = nullptr, *ws_iscore_ = nullptr;
    float *ws_rlogit_ = nullptr, *ws_rw_ = nullptr;
    // `ws_moe_` is ONE allocation of 2 * max_seq * hidden floats holding TWO
    // [T, hidden] accumulators back to back: the routed-expert sum at
    // `ws_moe_`, and the shared expert's output at `ws_moe_ + T*hidden`
    // (`ds4_shared_out`).  They are adjacent so that when BOTH are partials —
    // which is exactly the non-expert-split configuration, where
    // `ffn_down_shexp` is column-sliced like the routed `down` — one
    // `ds4_tp_reduce_host` call over 2*T*hidden floats reduces them together.
    // Two separate calls would be two host round trips for the same 32 KB, and
    // doc 33 §2.5 measured that round trip to be latency-bound (16.83 us at
    // 32 KB, 18.44 us at 64 KB), so the payload is nearly free and the second
    // trip would not be.
    //
    // Keeping them SEPARATE rather than accumulating the shared expert into
    // `ws_moe_` is what preserves `moe_accumulator()`'s contract: that buffer
    // means "the routed-expert sum" in every configuration, so comparing it
    // between a one-card and a two-card run stays the sharp partition test it
    // was built to be.
    float *ws_moe_ = nullptr, *ws_shg_ = nullptr, *ws_shu_ = nullptr;
    float* ds4_shared_out(uint32_t T) const noexcept {
        return ws_moe_ + uint64_t(T) * cfg_.hidden;
    }
    float *ws_logits_ = nullptr;
    int32_t *ws_pos_ = nullptr, *ws_spos_ = nullptr, *ws_cpos_ = nullptr;
    int32_t *ws_ridx_ = nullptr, *ws_topk_ = nullptr, *ws_ids_ = nullptr;
    float *inv_main_dev_ = nullptr, *inv_comp_dev_ = nullptr;
    std::vector<float>   inv_main_, inv_comp_;
    // The router's decision, read back to the host once per layer.  PINNED USM,
    // not std::vector, and written by a KERNEL rather than a `memcpy` — see
    // layer_forward_mid, where the reason is measured.  Both are
    // `max_seq * n_experts_used` long and only the first `T * K` of each is live.
    int32_t* h_ridx_ = nullptr;
    float*   h_rw_   = nullptr;
    std::vector<uint32_t> h_slots_;
    // Sliding-window query positions, staged host-side.  A MEMBER rather than a
    // stack-local because the H2D that reads it is no longer waited for, and
    // the source of an outstanding copy has to outlive it.  Rewritten once per
    // layer, always after that layer's predecessor drained.
    std::vector<int32_t> h_spos_;

    // Vision-Exp staging (set_vision).  `ws_imgmask_` / `ws_vleft_` /
    // `ws_vright_` are per-chunk [max_seq] int32 device arrays; `vis_chunk_`
    // says the chunk in flight carries image rows — when false every kernel
    // takes the text-only path, untouched.  `h_vis_` is the host source of the
    // three arrays (member: the H2D is not waited for) and `h_vis_rows_` the
    // PINNED source of the row splice (pageable sources stall the immediate
    // command list, see sycl-kernel-name-collision-trap / ds4_vision.cpp).
    struct VisSpan { uint32_t t0, n, start_off; std::vector<float> rows; };
    std::vector<VisSpan> vis_spans_;
    std::vector<int32_t> h_vis_;
    float*   h_vis_rows_  = nullptr;
    int32_t *ws_imgmask_ = nullptr, *ws_vleft_ = nullptr, *ws_vright_ = nullptr;
    bool     vis_chunk_   = false;
    std::string vision_prologue(uint32_t T, uint32_t pos0);

    Ds4TpResidencyPlan plan_{};
    Ds4ExpertSlice     slice_{};
    Ds4AttnSlice       aslice_{};
    // Residency priority (see Ds4Options::expert_priority) and its inverse over
    // the static partition: prio_[i] = expert in static slot i; prio_rank_[e] =
    // e's static slot, or -1 when e streams.
    // prio_ is LAYER 0's order, kept because the log line and the existing
    // accessor are about "the" order and every single-permutation configuration
    // makes all 43 layers agree anyway.  prio_layer_[L] is the real thing.
    std::vector<uint32_t>              prio_;
    std::vector<std::vector<uint32_t>> prio_layer_;
    // [layer][expert] -> that expert's static VRAM slot IN THAT LAYER, or -1 if
    // it streams.  Was [expert]: one 256-entry array applied to all 43 layers,
    // which forced the same 80 experts to be permanently resident everywhere.
    std::vector<int32_t>  prio_rank_;
    PriorityOrigin        prio_from_ = PriorityOrigin::kIndex;
    std::string           prio_src_;
    // Selection counting, and the path release() dumps it to (empty = no dump).
    Ds4ExpertProfile      profile_;
    std::string           profile_path_;
    // ---- optional TEMPORAL routing trace ($DS4_EXPERT_TRACE) ----
    // `Ds4ExpertProfile` COUNTS selections.  A count cannot answer the only
    // question a replacement policy depends on — "did token t+1 route to an
    // expert token t also routed to" — because that is a statement about ORDER,
    // and the histogram has thrown the order away.  This records the sequence
    // instead: one record per (forward, token, layer) naming the K experts the
    // router chose, dumped as text by release().
    //
    // OFF unless $DS4_EXPERT_TRACE names a file.  Card 0 only: under expert-TP
    // both cards run the identical router GEMV on the identical hidden state,
    // so a second copy would be the same file written twice.  Capped, because a
    // long serving run must not grow host RAM without bound; the dump says how
    // many records were dropped rather than silently truncating.
    // ---- lifetime cache accounting ($DS4_CACHE_BREAKDOWN) ----
    // `Ds4ExpertCache::Stats` is reset by every forward_prologue, so nothing
    // downstream can see a whole run's behaviour, and the harness that COULD
    // aggregate it sums both cards and both phases into one hit rate.  A single
    // blended number cannot answer "did the residency order do what it was
    // supposed to", because prefill touches nearly every expert once and decode
    // touches six, and they move in opposite directions.  Kept separately, per
    // phase, and dumped by release() only when asked for.
    struct CacheTotals {
        uint64_t acquires = 0, static_hits = 0, stream_hits = 0, misses = 0;
        uint64_t unavailable = 0, bytes = 0, forwards = 0;
    };
    CacheTotals ctot_decode_{}, ctot_prefill_{};
    void        accumulate_cache(uint32_t T) noexcept;

    std::string           trace_path_;
    // $DS4_ACT_DUMP=<path> (card 0, T==1 only): the RMS-normed layer inputs at
    // the attention site and the FFN site, f32 [hidden] per (token, layer, site),
    // for the Delta-Forest block-similarity probe (tools/deepseek4/delta_forest_probe.py).
    std::FILE*            act_dump_ = nullptr;
    std::vector<float>    h_act_;
    void act_dump(uint32_t L, uint32_t site, const float* dev_row);
    std::vector<int32_t>  trace_buf_;    // flat records, kDs4TraceFields + K each
    std::vector<int32_t>  trace_ids_;    // THIS forward's input token ids
    uint64_t              trace_seq_     = 0;
    uint64_t              trace_dropped_ = 0;
    uint64_t resident_bytes_ = 0;
    uint64_t resident_saved_ = 0;
    uint32_t pinned_layers_  = 0;
    uint32_t ws_T_           = 0;
    uint32_t ws_strip_       = 0;   // query rows per strip, see ws_strip_rows()
    bool     kv16_values_    = true;    // $IE_DS4_KV16, see load_prepare
    bool     kv16_keys_      = false;
    // Every byte alloc_workspace() put on this card (attention/indexer
    // workspaces, the expert-major batch workspace, the XMX fp16 staging), so
    // the expert-arena derivation can subtract what is really there instead of
    // a copy of the sizing formulas.
    uint64_t ws_bytes_       = 0;
    // Was this card's compute queue created with enable_profiling?  Kept so a
    // forward can REFUSE when `ie::g_profiler` is set but the events it would
    // harvest do not carry timestamps — see forward_prologue.
    bool     q_profiling_    = false;
    // Rows `ws_logits_` currently holds.  1 after alloc_workspace, because
    // `last_only` is true at every call site there is; grown by
    // forward_epilogue, and only by it, if a caller ever passes false.
    uint32_t ws_logits_T_    = 0;
    Timing   timing_{};
    HostWaits waits_{};
    // Drains this card's compute queue and charges the elapsed HOST time to
    // `acc`/`n`.  The queue is in-order, so waiting on it is exactly waiting on
    // the last thing submitted — every call site below replaced an `evt.wait()`
    // or a bare `q.wait()` with this and changed nothing about the ordering.
    void drain(double& acc, uint64_t& n);
    // One static-slot staging buffer, sized to this card's widest layer slot and
    // reused for every statically-resident expert.  Owned by THIS runtime; the
    // TP driver never writes through another card's.  Released by load_finish().
    std::vector<uint8_t> pack_stage_;
    // Routed-expert slot bytes this card has packed so far.  Read by the TP
    // orchestrator's load trace to show both cards climbing together.
    uint64_t             packed_bytes_ = 0;
};

// ===========================================================================
// THE TWO-CARD ORCHESTRATOR — hidden-dim expert-TP, driven in lockstep
// ===========================================================================
//
// WHAT IT IS.  `n_cards` `DeepSeek4Runtime`s, one per device, each holding
// `moe_intermediate_size / n_cards` of EVERY routed expert and a FULL MIRRORED
// copy of every non-expert weight (docs/deepseek4/33 §2.1 "Option A", §2.5).
// It is what makes Q3 fit: 102 VRAM slots/layer instead of 51, and 78.068 GB of
// pinned host RAM against the 84 GB cap where one card needs 102.052 GB.
//
// WHAT LOCKSTEP MEANS HERE, PRECISELY.  Every card runs the same tokens through
// the same mirrored weights, so every card computes the same attention output,
// the same hyper-connection, the same RMSNorm and — the load-bearing one — the
// same router decision.  Only the routed-expert block differs, and only in which
// half of the intermediate dimension it contracts over.  Card c's `ws_moe_` is
// therefore `Σ_k w_k · down_c(swiglu(gate_c(x), up_c(x)))`, a partial of the true
// `Σ_k w_k · down(swiglu(gate(x), up(x)))`, and the cards' partials sum to it
// exactly because the slice is a partition of the contraction.
//
// THAT SAMENESS IS CHECKED, NOT ASSUMED.  Mirrored weights and identical inputs
// SHOULD give bit-identical results on two identical devices, but "should" is
// not a property this engine gets to rest a logit on.  After the router runs,
// the orchestrator compares every card's selected expert ids AND their routing
// weights against card 0's, bit for bit, and REFUSES the whole forward pass by
// name if any card disagrees.  Different ids would mean the partials belong to
// different experts and summing them is meaningless; identical ids with
// different weights would mean the mirrored activations have drifted.  Because
// the routing weights are a continuous function of `ws_norm_`, comparing them
// bit-exactly is in practice a checksum of the entire mirrored activation chain.
//
// THE REDUCTION GOES THROUGH HOST.  Level-Zero P2P is unavailable between the
// two B70s in this box (`can_access_peer == 0` both directions; a D2D copy fails
// with `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`), so `ds4_tp_reduce_host` stages
// through PINNED host memory — one buffer per card, in that card's own context.
//
// THE CARDS RUN CONCURRENTLY.  Each card's per-layer segment is driven on its
// own host thread — card 0 on the calling thread, cards 1..n-1 on workers this
// class owns — so both GPUs are submitting at the same instant.  They used to
// run STRICTLY SEQUENTIALLY: card 0's segment ran to completion (it ends in a
// `q.wait()`), then card 1's, then the reduction, which left one 32 GB GPU idle
// at every instant.  The only barrier now is the cross-card reduction, which is
// a genuine data dependency: `layer_forward_post` consumes the SUM.  What the
// forward records about this is in `Ds4TpForwardTrace`, and the property is
// falsifiable — under the sequential drive the cards' [begin, end] intervals
// were disjoint by construction.
//
// WHAT MAKES THAT SAFE.  Each `DeepSeek4Runtime` owns its queue, its
// workspaces, its KV caches, its expert cache and its pinned arena; a card's
// thread touches only its own runtime.  The three pieces of process-wide state
// the forward path reaches are the profiler tag `g_ds4_site` (thread_local), a
// magic-static env read, and `ie::g_profiler` — and that last one is a shared
// `std::vector` with no lock, so a forward WITH THE PROFILER ON falls back to
// the sequential drive rather than racing it.  That fallback is recorded in the
// trace (`Ds4TpForwardTrace::sequential`), not silent, and it mirrors what the
// other TP models do: timing mode adds barriers.
//
// THE NON-EXPERT SET IS ALSO SPLIT, PARTIALLY, AND ON PURPOSE.  Under
// `Ds4Options::split_non_expert` (default on) the query heads, the attention
// output projection and the shared expert are divided between the cards as well;
// everything else — the shared KV head, the compressors, the lightning indexer,
// the router, the hyper-connections, `token_embd` and the LM head — stays
// mirrored because splitting it would buy less VRAM than its reduction costs.
// The per-tensor arithmetic is in deepseek4_residency_test §8c.  Two
// consequences are visible from out here: there are TWO reductions per layer
// instead of one, and each layer is driven in THREE segments instead of two.
//
// WHAT IS NOT IMPLEMENTED, AND REFUSES RATHER THAN APPROXIMATING:
//   * a sliced runtime driven through its OWN `forward()` still refuses loudly.
//   * a head/group geometry that does not divide by n_cards is a named error at
//     load time, not a silent fall back to mirroring.
struct Ds4TpOptions {
    // Shared by every card.  `card`, `n_cards` and `device_ordinal` are owned by
    // this struct's own fields and MUST be left at their defaults here; load()
    // refuses instead of overwriting them.
    Ds4Options base{};
    uint32_t   n_cards = 2;
    // Card c runs on GPU `device_ordinals[c]`.  Empty means the identity map
    // {0, 1, ... n_cards-1}.  Ordinals must be distinct: two runtimes on one
    // device each size their expert arena from that device's whole
    // `global_mem_size`, so the residency plan they print would be a fiction.
    std::vector<uint32_t> device_ordinals;
    // Explicit opt-in to putting more than one card on the SAME device.  It is a
    // rehearsal mode for when the second card is occupied: the arithmetic and the
    // staging are real, the residency plan is NOT.  Anything measured under it
    // must say so.  Without this, a repeated ordinal is a hard, named error.
    bool same_device_rehearsal = false;
};

// What `DeepSeek4TpRuntime::load` actually did, recorded as it did it.  This
// exists because "both cards filled together" and "the file was walked once" are
// exactly the two properties the sequential load got wrong, and neither is
// visible from the outside once the load has finished.
struct Ds4TpLoadTrace {
    // Longest run of consecutive expert packs served to the SAME card during the
    // single file walk.  The interleaved drive gives 1 for every n_cards > 1;
    // loading card 0 to completion and then card 1 gives
    // pinned_layers * n_experts.  One number, and it discriminates the defect.
    uint64_t max_same_card_run = 0;
    // Wall seconds the concurrent prepare phase took, and the single walk.
    double   prepare_seconds = 0.0;
    double   pack_seconds    = 0.0;
    // Per card, seconds from the start of the prepare phase: when that card's
    // `load_prepare` began and ended.  This is where the mirrored always-resident
    // set is uploaded — 7.887 GB per card for Q3 — so it is the phase in which one
    // GPU used to max out while the other sat at its compositor's 34 MiB.  Running
    // it on a thread per card makes the intervals OVERLAP, and an overlap is a
    // falsifiable statement: under the old sequential load card c+1's begin was
    // strictly after card c's end.
    std::vector<double> prepare_begin, prepare_end;
    // Per card, at the end: routed-expert slot bytes packed.
    std::vector<uint64_t> packed_bytes;
    // Sampled once per layer of the walk: every card's cumulative packed bytes at
    // that instant.  A sequential load shows card 1 pinned at 0 for the whole
    // first half; the interleaved drive shows the cards within one expert of each
    // other at EVERY sample.
    struct Sample {
        double                seconds;
        std::vector<uint64_t> per_card;
    };
    std::vector<Sample> timeline;
};

// What `DeepSeek4TpRuntime::forward` did about CONCURRENCY, recorded as it did
// it.  This exists for the same reason `Ds4TpLoadTrace` does: "both cards were
// computing at the same time" is invisible once a forward has finished — the
// logits are identical either way — so the only way to hold the claim to
// account is to have the forward say when each card started and stopped.
//
// A SEGMENT is one card's share of one phase of one layer.  There are THREE
// phases: 0 is `layer_forward_pre` (up to the attention-output partial), 1 is
// `layer_forward_mid` (hyper-connection mix, router, routed experts and the
// shared expert, ending at the MoE partial), 2 is `layer_forward_post` (the
// FFN-site mix on the reduced sum).  Segment index s covers layer s/3, phase
// s%3.  Times are seconds from the entry to `forward()`, on
// `std::chrono::steady_clock`.
struct Ds4TpForwardTrace {
    uint32_t n_cards = 0;
    // Row-major [segment * n_cards + card].  Cleared and refilled by every
    // forward(), so this describes the LAST call only.
    std::vector<double> begin, end;
    // Segments in which every card's interval intersects every other card's,
    // out of `total`.  The sequential drive gives 0: card c+1's begin was
    // strictly after card c's end, because card c's segment ends in a q.wait().
    uint64_t overlapped = 0, total = 0;
    // Σ over segments of the n-card interval intersection
    // (max(0, min_c end_c − max_c begin_c)), and Σ of every card's segment
    // durations.  `overlap_seconds` is exactly 0.0 under a sequential drive.
    double overlap_seconds = 0.0, busy_seconds = 0.0;
    // Set when this forward ran the cards ONE AFTER THE OTHER after all: the
    // kernel profiler was on and its entry list is an unsynchronised
    // std::vector.  A true statement about the run, printed rather than hidden.
    bool sequential = false;

    // ---- the wall-clock decomposition of ONE forward() -------------------
    // Every field is measured on the orchestrator's own steady_clock, and the
    // five of them TILE `wall_seconds` by construction:
    //
    //   wall = prologue + span + reduce + epilogue + gap
    //
    // where `span` is Σ over segments of (max_c end_c − min_c begin_c) — the
    // wall time the drive actually spent inside run_cards — and `gap` is
    // whatever is left: the fork/join handshakes, the lockstep check, the trace
    // bookkeeping and any host work between the phases.  `gap` is the residual
    // on purpose: it is the term that cannot hide, because it is defined as
    // what the four measured terms do not explain.
    //
    // `card_busy[c]` is Σ over segments of card c's own interval, so
    // `span − max_c card_busy[c]` is the price of imperfect overlap plus the
    // handshakes inside run_cards, and `card_busy[c] − card_waits[c].total()`
    // is the host time card c spent SUBMITTING rather than blocked.
    double wall_seconds = 0.0, span_seconds = 0.0, reduce_seconds = 0.0;
    double prologue_seconds = 0.0, epilogue_seconds = 0.0;
    // Σ span for phase 0 (pre), 1 (mid), 2 (post).
    double phase_span[3] = {0.0, 0.0, 0.0};
    std::vector<double> card_busy;                        // per card
    std::vector<DeepSeek4Runtime::HostWaits> card_waits;  // per card, this forward
    uint64_t reductions = 0;                              // this forward only
    uint32_t tokens = 0;                                  // T
};

class DeepSeek4TpRuntime {
public:
    // Both out of line: `Worker` is incomplete here, and every implicitly
    // generated body that could destroy a `unique_ptr<Worker>` — the
    // constructor's own exception cleanup included — needs it complete.
    DeepSeek4TpRuntime();
    ~DeepSeek4TpRuntime();
    DeepSeek4TpRuntime(const DeepSeek4TpRuntime&)            = delete;
    DeepSeek4TpRuntime& operator=(const DeepSeek4TpRuntime&) = delete;

    // Loads every card from the same bound model.  Each card's slice, pinned
    // arena and VRAM cache come out of `DeepSeek4Runtime`'s own load steps
    // unchanged; this adds the card->device binding, verifies it landed, and
    // checks that the cards' slices tile the expert exactly.  Returns "" or a
    // diagnostic naming the card.
    //
    // IT DOES NOT LOAD THE CARDS ONE AFTER THE OTHER.  It runs every card's
    // `load_prepare` on its own thread — so both devices' always-resident sets
    // upload at the same time instead of one GPU maxing out while the other sits
    // idle — and then drives ONE walk over the routed-expert tensors in which
    // both cards' slices of the same expert are packed back to back.  Under
    // hidden-dim expert-TP the cards need the two halves of the SAME expert, so
    // a per-card walk read `down` twice (its slice is strided, so both cards
    // touch every page of it) with an hour and 120 GB of other traffic in
    // between, which no page cache on this box can bridge.  See
    // `Ds4TpLoadTrace` for what the load records about itself.
    std::string load(const DeepSeek4Model& m, const Ds4TpOptions& opt);

    // What the last load() did.  Empty before the first load.
    const Ds4TpLoadTrace& load_trace() const noexcept { return trace_; }
    // What the last forward() did about running the cards at the same time AND
    // where its wall went.  $IE_DS4_TP_TRACE=1 prints the same numbers to
    // stderr after every call.  Empty before the first forward.
    const Ds4TpForwardTrace& forward_trace() const noexcept { return ftrace_; }
    void        release() noexcept;

    // Same contract as DeepSeek4Runtime::forward — `logits` receives [T, vocab]
    // (or [1, vocab] when `last_only`).  Every card computes the identical
    // logits; card 0's are the ones copied out.
    std::string forward(const int32_t* input_ids, uint32_t T, uint32_t pos0,
                        float* logits, bool last_only = true);

    void reset_context() noexcept;

    // Vision-Exp image blocks: same contract as DeepSeek4Runtime::set_vision,
    // applied to every card (each card runs its own prologue over the same ids).
    std::string set_vision(const float* rows, uint32_t t0, uint32_t n, uint32_t start_off);
    void        clear_vision() noexcept;

    // ---- sequence slots, in lockstep across the cards ----------------------
    // Every card holds the SAME slot count and the SAME active slot, because
    // every card runs the whole attention for the sequence being stepped.  A
    // card that drifted to a different slot would attend over another
    // conversation's window state on its share of the heads and produce fluent
    // nonsense, so both calls are all-or-nothing: they apply to every card or
    // they report which card refused and change nothing.
    uint32_t    seq_slots() const noexcept { return rts_.empty() ? 0 : rts_[0]->seq_slots(); }
    uint32_t    active_seq() const noexcept { return rts_.empty() ? 0 : rts_[0]->active_seq(); }
    std::string select_seq(uint32_t slot) noexcept;
    std::string reset_seq(uint32_t slot) noexcept;

    // ---- prompt cache, all-or-nothing across the cards ---------------------
    // Each card snapshots ITS OWN share of the attention state (the non-expert
    // set is split, so no card holds the whole thing).  A snapshot that existed
    // on some cards and not others would restore a conversation on part of the
    // heads and a blank one on the rest, so any card's failure frees them all.
    std::string snapshot_context();
    uint32_t    restore_context();
    void        free_snapshot() noexcept;
    uint32_t    snapshot_depth() const noexcept;
    uint64_t    snapshot_bytes() const noexcept;
    // Host-resident form, one Ds4HostSnapshot per card (Phase L3).
    std::string snapshot_to_host(std::vector<Ds4HostSnapshot>& per_card);
    uint32_t    restore_from_host(const std::vector<Ds4HostSnapshot>& per_card);

    uint32_t          n_cards() const noexcept { return uint32_t(rts_.size()); }
    DeepSeek4Runtime& card(uint32_t c) { return *rts_[c]; }
    uint32_t          device_of(uint32_t c) const noexcept { return ordinals_[c]; }
    // Pinned host bytes this class holds for the cross-card staging (grows with
    // the largest T seen, never shrinks).
    uint64_t          stage_bytes() const noexcept { return stage_n_ * 4ull * rts_.size(); }
    // Cross-card reductions performed, and host seconds spent in them.
    uint64_t          reductions() const noexcept { return reductions_; }
    double            reduce_seconds() const noexcept { return reduce_s_; }

private:
    std::string ensure_stage(uint64_t n);

    // ---- the per-card forward workers ----
    // One persistent thread per card ABOVE card 0; card 0 runs on the calling
    // thread.  Persistent because the alternative — spawning threads inside the
    // layer loop — pays a thread creation per card per phase per layer, which on
    // the real model is 61*2*(n-1) spawns for every single decoded token.
    // Created by load(), joined and destroyed by release().
    struct Worker;
    std::vector<std::unique_ptr<Worker>> workers_;

    // Runs `fn(c)` for EVERY card at the same time and joins them ALL before
    // returning — including when one of them fails, so a failing card can never
    // leave a sibling running into a released workspace.  Returns the
    // LOWEST-numbered failing card's message (with `*out_card` set to it), or ""
    // when every card succeeded.  A thrown exception on any card is caught and
    // turned into that card's message; it is never allowed to escape a worker
    // thread, where it would call std::terminate.
    //
    // `seg` is the Ds4TpForwardTrace segment index to record the per-card
    // [begin, end] into, or UINT32_MAX to record nothing.
    std::string run_cards(uint32_t seg, const std::chrono::steady_clock::time_point& t0,
                          const std::function<std::string(uint32_t)>& fn, uint32_t* out_card);

    std::vector<std::unique_ptr<DeepSeek4Runtime>> rts_;
    std::vector<uint32_t> ordinals_;
    std::vector<float*>   stages_;      // pinned, one per card, in that card's context
    uint64_t              stage_n_ = 0; // floats per card
    Ds4TpReducer          reducer_;     // the sliced prefill-size reduction (Phase I)
    uint64_t              reductions_ = 0;
    double                reduce_s_ = 0.0;
    Ds4TpLoadTrace        trace_{};
    Ds4TpForwardTrace     ftrace_{};
};

// ---------------------------------------------------------------------------
// Two Phase-5-owned device primitives, exposed because the forward test gates
// them directly against the parity blobs.
// ---------------------------------------------------------------------------

// DeepseekV4HyperHead.forward (modeling_deepseek_v4.py:955) — the final stream
// collapse before the shared RMSNorm.  Structurally NOT `ds4_hyper_connection`:
// `hc_fn` is [hc_mult, hc_mult*hidden] (not [(2+hc)*hc, ...]), there is no
// `post`, no `comb` and no Sinkhorn — only the `pre` sigmoid gate.
//   x        [n_tokens, hc_mult, hidden]
//   hc_fn    [hc_mult, hc_mult*hidden]
//   hc_base  [hc_mult]     hc_scale [1]
//   y        [n_tokens, hidden]
sycl::event ds4_hyper_head(sycl::queue& q,
                           const float* x, const float* hc_fn, const float* hc_base,
                           const float* hc_scale, float* y,
                           uint32_t n_tokens, uint32_t hidden, uint32_t hc_mult,
                           float rms_eps, float hc_eps,
                           const std::vector<sycl::event>& deps = {});

// The DecoderLayer residual mix (modeling_deepseek_v4.py:1135, 1141):
//   out[t, h, d] = post[t, h] * sub[t, d] + Σ_j comb[t, j, h] * streams[t, j, d]
// `comb` is consumed TRANSPOSED — the sum is over the FIRST hc axis.  Sinkhorn
// yields a doubly-stochastic but NON-symmetric matrix, so using comb[t,h,j]
// instead is a silent long-context corruption, not a no-op.
//   streams, out  [n_tokens, hc_mult, hidden]   (out may equal streams exactly)
//   post          [n_tokens, hc_mult]
//   comb          [n_tokens, hc_mult, hc_mult]
//   sub           [n_tokens, hidden]
// Other overlaps are unsupported. Exact in-place hc_mult != 4 uses temporary
// output and waits for copy-back; hc_mult == 4 and disjoint output are async.
sycl::event ds4_hc_mix(sycl::queue& q,
                       const float* streams, const float* post, const float* comb,
                       const float* sub, float* out,
                       uint32_t n_tokens, uint32_t hidden, uint32_t hc_mult,
                       const std::vector<sycl::event>& deps = {});

// ---------------------------------------------------------------------------
// Test seams for the packed-residency decoders.
// ---------------------------------------------------------------------------
// These route through the SAME `Ds4Dense` dispatchers `forward()` uses — they
// are not reimplementations — so a test can prove the in-kernel decode is
// bit-exact against `ie::ref` without loading a 128 GB model.  `w` is a device
// pointer to verbatim GGUF bytes: N rows of `ds4_dense_row_bytes(dt, K)`.
// `dt` must satisfy ds4_dense_keeps_packed(dt, K); anything else is a caller
// bug and the call is refused with a diagnostic rather than run on the fp16
// branch with a null weight.
std::string ds4_dense_packed_gemv(sycl::queue& q, const float* x, const void* w, DType dt,
                                  float* y, uint32_t T, uint32_t K, uint32_t N);
std::string ds4_grouped_packed_gemv(sycl::queue& q, const float* x, const void* w, DType dt,
                                    float* y, uint32_t T, uint32_t G, uint32_t IPG,
                                    uint32_t OPG);

// The same seams for the fp16-expanded branch — the one every BF16 weight in the
// Q8_K_XL model actually takes (BF16 is deliberately not packed, see
// `ds4_dense_uploadable`), and therefore the branch that carries ~66% of prefill.
// It had no numerical coverage at all until deepseek4_dense_gate_test; a mutant
// that transposed the activation index was caught by nothing in the repo.
// `w` is fp16 [N, K] device memory, the layout `upload_dense` produces.
std::string ds4_dense_f16_gemv(sycl::queue& q, const float* x, const sycl::half* w,
                               float* y, uint32_t T, uint32_t K, uint32_t N);
std::string ds4_grouped_f16_gemv(sycl::queue& q, const float* x, const sycl::half* w,
                                 float* y, uint32_t T, uint32_t G, uint32_t IPG,
                                 uint32_t OPG);

// The same seams for the REQUANTISED branch — the one every BF16 dense weight in
// the Q8_K_XL model takes once `Ds4DenseQuant::kAuto` applies.  `qs`/`d` are the
// two planes `ds4_requantise_q8_soa` produces, uploaded verbatim; the call goes
// through `dense_w`/`grouped_w`, so a test exercises the SAME dispatcher (and
// therefore the same T==1 vs T>1 routing) the forward pass does, including the
// fp16 materialisation the prefill route needs.  Refused by name if K is not a
// whole number of Q8_0 blocks rather than reading a ragged final block.
std::string ds4_dense_q8_gemv(sycl::queue& q, const float* x, const int8_t* qs,
                              const sycl::half* d, float* y, uint32_t T, uint32_t K,
                              uint32_t N);
std::string ds4_grouped_q8_gemv(sycl::queue& q, const float* x, const int8_t* qs,
                                const sycl::half* d, float* y, uint32_t T, uint32_t G,
                                uint32_t IPG, uint32_t OPG);

}  // namespace ie
