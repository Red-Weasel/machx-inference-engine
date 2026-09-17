// include/ie/deepseek4_attn.hpp — DeepSeek-V4 attention stack (Phase 3).
//
// Covers the pieces of the reference attention block that have no equivalent in
// the existing op set:
//   * interleaved partial RoPE on the TRAILING rope slice, with the dual-theta
//     (main 10000 / compress 160000 + YaRN) frequency tables, and the
//     output-side conjugate rotation at position -i;
//   * shared-KV MQA core attention with per-head learnable sinks and an
//     additive per-query block bias;
//   * the HCA / CSA compressor window pooling (both shape families);
//   * the Lightning Indexer scorer and its top-k selection.
//
// Conventions match include/ie/ops.hpp and include/ie/deepseek4_ops.hpp:
// device-USM pointers, a deps vector, returns sycl::event.  FP32 in / FP32 out
// throughout, for the same reason Phase 2 is fp32: the reference runs these
// tensors in fp32 (softmaxes are `dtype=torch.float32`, the RoPE rotation is
// `.float()`, the indexer scorer casts both operands to float), and the parity
// blobs are fp32.
//
// REUSE ASSESSMENT (verified by reading the code, not assumed)
// -----------------------------------------------------------
// * `full_attention_gptoss` (src/ops/attention.cpp:342) — the SINK MATH IS THE
//   SAME.  The reference concatenates `sinks` as an extra logit column,
//   subtracts the row max, softmaxes, then drops the sink column
//   (modeling_deepseek_v4.py:733-741); that is algebraically identical to the
//   gpt-oss kernel's "fold the sink into the denominator only" online form.
//   The KERNEL itself is NOT reusable: it is fp16, it derives its mask from
//   `(causal, window)` index arithmetic instead of taking an additive mask (V4
//   needs the compressor's per-query `block_bias`), it indexes a
//   `[n_kv_heads, max_ctx, head_dim]` slab rather than a contiguous
//   `[n_kv, head_dim]` run, and its per-lane register arrays are `[16]`, which
//   overflows at V4's head_dim=512 (dpl=32).  So `ds4_attention` below is a new
//   kernel that reproduces the same sink algebra.
// * `rope_partial` / `rope_yarn` (src/ops/elementwise.cpp:588 / 711) — the
//   MATH DOES NOT MATCH and they are NOT reused.  Two independent mismatches:
//   (a) pairing — those kernels use the NEOX pairing `(r, r + n_rotary/2)`,
//       V4 uses the INTERLEAVED pairing `(2r, 2r+1)`
//       (modeling:335-339 `x[..., 0::2]` / `x[..., 1::2]`);
//   (b) placement — those kernels rotate the FIRST `n_rotary` dims, V4 rotates
//       the LAST `rope_dim` dims (modeling:357 `x[..., -rope_dim:]`).
//   `rope_yarn` also applies ggml's mscale to cos/sin; V4 forces
//   `attention_factor = 1.0` (configuration_deepseek_v4.py:298-302).  Reusing
//   either would corrupt every rotated channel.

#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>
#include <vector>

namespace ie {

// ---------------------------------------------------------------------------
// RoPE frequency tables.  DeepseekV4RotaryEmbedding (modeling:75) builds ONE
// table per *rope-type label* — "main" and "compress" — not per layer type.
//   main     : rope_type "default", theta = rope_theta (10000), no scaling.
//   compress : rope_type "yarn",    theta = compress_rope_theta (160000),
//              factor 16, beta_fast 32, beta_slow 1,
//              original_max_position_embeddings 65536,
//              attention_factor forced to 1.0 by the V4 config.
// Sliding layers use "main"; every CSA/HCA branch — including the indexer, for
// both its compressed keys and its queries — uses "compress" (modeling:382,
// 491, 610, 777).
// ---------------------------------------------------------------------------
struct Ds4RopeConfig {
    float    theta            = 10000.f;
    bool     yarn             = false;
    float    factor           = 1.f;
    float    beta_fast        = 32.f;
    float    beta_slow        = 1.f;
    uint32_t original_max_pos = 0;
    float    attention_factor = 1.f;  // multiplies cos/sin; 1.0 for both V4 tables
};

// Host-side inv_freq, `rope_dim / 2` entries.  Transcribed from
// DeepseekV4RotaryEmbedding.compute_default_rope_parameters (modeling:114) and
// transformers' `_compute_yarn_parameters` (modeling_rope_utils.py:327).
std::vector<float> ds4_rope_inv_freq(const Ds4RopeConfig& cfg, uint32_t rope_dim);

// cos/sin tables: cos[p, r] = cos(positions[p] * inv_freq[r]) * attention_factor.
//   inv_freq  [half_dim]
//   positions [n_pos] int32
//   cos, sin  [n_pos, half_dim]
sycl::event ds4_rope_cos_sin(sycl::queue& q,
                             const float* inv_freq, const int32_t* positions,
                             float* cos, float* sin,
                             uint32_t n_pos, uint32_t half_dim,
                             float attention_factor,
                             const std::vector<sycl::event>& deps = {});

// Interleaved partial RoPE on the TRAILING `rope_dim` channels of each head.
// apply_rotary_pos_emb (modeling:342):
//   cos/sin are repeat_interleave(2) to `rope_dim`, then
//   rotated = rope*cos + rotate_half(rope)*sin  with
//   rotate_half(x) = stack((-x[1::2], x[0::2]), -1).flatten(-2)
// i.e. for each pair p:
//   y[2p]   = x[2p]  *cos[p] - x[2p+1]*sin[p]
//   y[2p+1] = x[2p+1]*cos[p] + x[2p]  *sin[p]
// The leading `head_dim - rope_dim` "nope" channels are copied unchanged.
//
//   x, y  [n_tokens, n_heads, head_dim]   (may alias exactly; other overlap unsupported)
//   rope_dim is even and at most head_dim. Nonzero n_tokens requires
//   positive n_heads/head_dim. Invalid dimensions, byte spans and null live
//   operands throw before submission. Empty work preserves dependencies.
//   cos, sin [n_tokens, rope_dim/2]
//   sin_sign  +1 for the forward rotation; -1 is the conjugate rotation the
//             reference applies to the ATTENTION OUTPUT (modeling:868,
//             `apply_rotary_pos_emb(attn_output.transpose(1,2), cos, -sin)`),
//             which undoes the rotation V picked up because K == V.  Omitting
//             it silently corrupts long context.
sycl::event ds4_rope_apply(sycl::queue& q,
                           const float* x, const float* cos, const float* sin,
                           float* y,
                           uint32_t n_tokens, uint32_t n_heads,
                           uint32_t head_dim, uint32_t rope_dim,
                           float sin_sign,
                           const std::vector<sycl::event>& deps = {});

// Absolute positions of the compressed entries a compressor is about to emit:
//   positions[i] = i * compress_rate + first_window_position   (modeling:419, 551)
// Host helper — the caller uploads the result for ds4_rope_cos_sin.
std::vector<int32_t> ds4_compress_positions(uint32_t n_win, uint32_t compress_rate,
                                            uint32_t first_window_position);

// ---------------------------------------------------------------------------
// Weighted RMSNorm in fp32 — DeepseekV4RMSNorm (modeling:46).
//   y = weight * (x * rsqrt(mean(x^2) + eps))
// The existing `rms_norm_f32w` (ops.hpp:41) takes fp16 activations; the
// compressor's kv_norm runs on fp32 rows, hence this fp32-in/fp32-out variant.
//   x, y [n_rows, n]   w [n]
// ---------------------------------------------------------------------------
sycl::event ds4_rms_norm(sycl::queue& q,
                         const float* x, const float* w, float* y,
                         uint32_t n_rows, uint32_t n, float eps,
                         const std::vector<sycl::event>& deps = {});

// ---------------------------------------------------------------------------
// Shared-KV MQA core attention with per-head sinks — eager_attention_forward
// (modeling:717) as V4 calls it (modeling:849, `attention_interface(self, q, kv, kv, ...)`).
//
//   q_in  [T, n_heads, head_dim]
//   kv    [n_kv, head_dim]        SINGLE kv head, read as both K and V
//   mask  [T, n_kv] additive, or nullptr.  This is the model-level sliding
//         causal mask with the compressor's `block_bias` concatenated on the
//         kv axis (modeling:840-844).  -inf entries are skipped exactly.
//   sinks [n_heads] or nullptr
//   y     [T, n_heads, head_dim]
//   scaling  head_dim^-0.5, applied to the logits BEFORE the mask is added.
// head_dim must be a multiple of 16 and at most 512 (the per-lane register
// budget, same constraint full_attention_gemma carries).  V4-Flash's 512 and
// the indexer's 128 both satisfy it; anything else throws std::invalid_argument
// rather than silently overflowing device memory.
// ---------------------------------------------------------------------------
sycl::event ds4_attention(sycl::queue& q,
                          const float* q_in, const float* kv,
                          const float* mask, const float* sinks,
                          float* y,
                          uint32_t T, uint32_t n_heads, uint32_t head_dim,
                          uint32_t n_kv, float scaling,
                          const std::vector<sycl::event>& deps = {});

// TWO-SEGMENT KV (docs/deepseek4/73 Phase 1).  The kv axis the kernels walk is
// the model's cat([sliding, compressed]) — and until Phase 1 the runtime built
// that concatenation with a device memcpy of the ENTIRE compressed cache into a
// staging buffer on every forward (~1.1 GB per decoded token at 100K context,
// 10.7 GB at 1M).  The kernels now read column i < n_a from `a` and column
// i >= n_a from `b + (i - n_a)`, in the SAME column order as the contiguous
// buffer they used to be handed, so the result is bit-identical to
// ds4_attention over the concatenation and the copy is gone.
//   a  [n_a, head_dim] fp32   the sliding buffer (127 + T rows, cache scratch)
//   b  [n_b, head_dim]        the compressed entries, read from the cache in
//                             place — fp16 when `b_f16` (Phase 2's default for
//                             the values: exactly what the XMX kernel converted
//                             to on every call), else fp32 (IE_DS4_KV16=0)
// n_kv == n_a + n_b.  `b` may be null when n_b == 0, and must carry at least
// 64 finite rows past n_b otherwise (Ds4LayerCache pads and zeroes them).  The
// plain ds4_attention above is this with a single fp32 segment.
struct Ds4KvSegs {
    const float* a     = nullptr;
    uint32_t     n_a   = 0;
    const void*  b     = nullptr;
    uint32_t     n_b   = 0;
    bool         b_f16 = true;
    // DSpark P2 (docs/deepseek41/55): a T-row decode step's own kv rows [a_new_n, head_dim], NOT yet in the ring `a`.
    // Column i of segment a is this step's row j = (i - a_new_base) mod n_a for query row t when j < a_new_n and
    // j <= t (read from a_new: the same column index the one-row step wrote it to, so the summation order is the
    // one-row step's), and the old ring's row i otherwise. null = every column from `a` (prefill, T = 1 before P2).
    const float* a_new = nullptr;
    uint32_t     a_new_base = 0, a_new_n = 0;
    bool         a_new_nocausal = false;   // diagnostic only: every row reads all of this step's rows (a negative control)
    // Phase 26 (docs/deepseek41/68): the indexer's picks for segment b, [T, n_picks] int32 (n_picks is the ROW
    // PITCH) in ASCENDING index order with the negative unreachable sentinels AFTER the valid picks. When given, the split kernels visit segment b
    // through this list -- O(n_picks) instead of O(n_b) -- instead of scanning every column and consulting the
    // mask. Ascending is REQUIRED for bit-identity: it is the order the dense scan visited them in, and the
    // online softmax is order-dependent. ds41_sort_picks_asc produces it (ds4_indexer_topk emits by descending
    // score). null = scan densely, exactly as before.
    const int32_t* b_picks = nullptr;
    uint32_t       n_picks = 0;
    // Phase 33: per-row count of valid picks, [T] int32 (ds41_sort_picks_asc writes it). The contiguous partition
    // sizes each slice's run from it, so a short row spreads over the slices; null = size by n_picks.
    const int32_t* b_pick_n = nullptr;
};
// `xmx_kv_prepared` is forwarded to ds4_attention_xmx_segs (below) when the
// prefill dispatch takes the XMX route; the fp32 kernels read the segments in
// place and ignore it.
sycl::event ds4_attention_segs(sycl::queue& q,
                               const float* q_in, Ds4KvSegs kv,
                               const float* mask, const float* sinks,
                               float* y,
                               uint32_t T, uint32_t n_heads, uint32_t head_dim,
                               float scaling, bool xmx_kv_prepared = false,
                               const std::vector<sycl::event>& deps = {});

// docs/deepseek4/72 Phase K: block-sparse XMX flash attention for prefill
// (src/ops/deepseek4_attn_xmx.cpp).  Same contract as ds4_attention, fp16
// operands with fp32 accumulation — NOT bit-identical to the fp32 kernel.
// ds4_attention dispatches to it when `ds4_attention_xmx_eligible` (T > 16,
// DSpark P2 (docs/deepseek41/55): when set, a 2..8-row step keeps the one-row split factor (64 slices) so each
// row's tree combine matches a one-row step bit for bit; the V4.1 resident runtime sets it, nothing else does.
extern bool g_ds4_attn_split_fixed64;
// head_dim 512, n_heads a multiple of 32, enough SLM; IE_DS4_ATTN_XMX=0 opts out —
// the default since its gate passed 2026-09-02).
bool ds4_attention_xmx_eligible(const sycl::device& dev, uint32_t T, uint32_t n_heads,
                                uint32_t head_dim);
sycl::event ds4_attention_xmx(sycl::queue& q,
                              const float* q_in, const float* kv,
                              const float* mask, const float* sinks,
                              float* y,
                              uint32_t T, uint32_t n_heads, uint32_t head_dim,
                              uint32_t n_kv, float scaling,
                              const std::vector<sycl::event>& deps = {});
// Two-segment form (see Ds4KvSegs).  The fp16 pre-pass stages segment a (and
// the few b rows that share its last 64-key block) in the per-queue staging
// buffer; every later block is read straight from b.  Block boundaries are
// those of the old contiguous layout and b holds the same saturated fp16
// values that layout was converted to, so this is bit-identical to
// ds4_attention_xmx over the fp32 concatenation.
// `kv_prepared`: the caller already ran that pre-pass on this queue for these
// exact segments (a previous query-row strip of the same layer) — skip it and
// reuse the staging buffer.  Phase 1 strips a layer's T query rows so the
// [T, n_kv] mask is a [strip, n_kv] one; the staging is per LAYER, not per
// strip, which is what this flag buys.
// `kv_rows` (V4.1 Phase 44, docs/deepseek41/84): the keys are PER QUERY ROW -- row t's column c at
// kv_rows[(t * kv.n_a + c) * head_dim], fp16, kv.n_a a multiple of 64, kv.b null, `mask` [T, kv.n_a] -- a gathered
// axis instead of one shared cache; no staging pass. null = the shared segments above, the same kernel as before.
sycl::event ds4_attention_xmx_segs(sycl::queue& q,
                                   const float* q_in, Ds4KvSegs kv,
                                   const float* mask, const float* sinks,
                                   float* y,
                                   uint32_t T, uint32_t n_heads, uint32_t head_dim,
                                   float scaling, bool kv_prepared,
                                   const std::vector<sycl::event>& deps = {},
                                   const sycl::half* kv_rows = nullptr);
// Bytes the XMX path will allocate lazily on `q` for these extents (Q staging
// [T, n_heads, 512] fp16 + KV staging [n_stage_max padded to 64, 512] fp16 —
// with a two-segment call n_stage is segment a's rows, i.e. the sliding
// buffer, not the context), so the expert-arena derivation can count them
// before they exist.
uint64_t ds4_attention_xmx_ws_bytes(uint32_t T, uint32_t n_heads, uint32_t n_stage_max) noexcept;
// Allocate that staging on `q` NOW (idempotent; grows only), so the bytes are
// real when the arena is derived rather than appearing at the first prefill.
std::string ds4_attention_xmx_reserve(sycl::queue& q, uint32_t T, uint32_t n_heads,
                                      uint32_t n_stage_max);

// Sliding causal mask, the model-level input to the above:
//   mask[t, k] = 0 when (k <= positions[t] && k > positions[t] - window) else
//                std::numeric_limits<float>::lowest()
// `window == 0` degenerates to plain causal.  Matches transformers'
// create_sliding_window_causal_mask, including its use of `finfo.min` rather
// than -inf.
// THE KEYS ARE ASSUMED TO SIT AT POSITIONS 0..n_kv-1: `k` above is the key's ROW
// INDEX, compared directly with `positions[t]`.  A caller whose keys start at an
// offset (a segment of a longer prompt, a chunk at pos0 > 0) must pass positions
// RELATIVE to its first key, or every row past `window` sees no keys at all —
// V4.1's bounded replay did exactly that (docs/deepseek41/28, gate 10 finding 1).
sycl::event ds4_sliding_causal_mask(sycl::queue& q,
                                    const int32_t* positions, float* mask,
                                    uint32_t T, uint32_t n_kv, uint32_t window,
                                    const std::vector<sycl::event>& deps = {});

// Vision-Exp variant — inference/model.py get_window_topk_idxs_visible: inside
// an [IMAGE_START, IMAGE_END] span the window is widened on both sides so the
// whole span attends to itself (bidirectional), text tokens are unchanged.
//   left_add[t]  max(left - (window-1), 0), left = distance from IMAGE_START
//   right[t]     distance to IMAGE_END (0 outside a span)
//   mask[t, k] = 0 when (k <= p + right[t]) && (k > p - window - left_add[t])
// Only dispatched for a chunk carrying image rows; the plain kernel above is
// untouched.
sycl::event ds4_sliding_causal_mask_vis(sycl::queue& q,
                                        const int32_t* positions,
                                        const int32_t* left_add, const int32_t* right,
                                        float* mask,
                                        uint32_t T, uint32_t n_kv, uint32_t window,
                                        const std::vector<sycl::event>& deps = {});

// ---------------------------------------------------------------------------
// Compressor window pooling — the shared body of DeepseekV4HCACompressor
// (modeling:412-418) and DeepseekV4CSACompressor (modeling:644-675) /
// DeepseekV4Indexer (modeling:528-550).  Emits the PRE-kv_norm pooled rows;
// the caller then runs ds4_rms_norm and ds4_rope_apply.
//
// `overlap == false` (HCA, compress_rate 128, src_width == width):
//     out[w, c] = Σ_{j<rate} softmax_j(gate[w, j, c] + bias[j, c]) * kv[w, j, c]
//
// `overlap == true` (CSA and the indexer, compress_rate 4, src_width == 2*width):
//     2*rate slots per window.  Slots [0, rate) are window w-1's Ca half
//     (`[..., :width]`), slots [rate, 2*rate) are window w's Cb half
//     (`[..., width:]`).  Window 0's Ca slots come from the cache's overlap
//     slice; when there is none they are zero-kv / -inf-gate (softmax weight 0).
//
//     DIVERGENCE FROM THE REFERENCE — deliberate, and numerically identical.
//     The reference adds `position_bias` to the whole chunk BEFORE the Ca/Cb
//     split, so the overlap slice it saves (modeling:298) already carries the
//     bias.  Here `Ds4LayerCache` stores the RAW projected gate and this kernel
//     adds the bias at use time.  Ca slot j always takes bias row j, columns
//     [0, width) — whether that row came from this call's window w-1 or from
//     the previous call's saved slice — so the sum is identical either way, and
//     the cache stays a pure state container with no dependency on layer
//     weights.  (Found by §4b of the Phase 3 gate, which drives the compressor
//     across several forward calls: the cache and the kernel originally
//     disagreed about which of them owned the bias, and the single-call checks
//     in §4 could not see it.)
//
//   chunk_kv, chunk_gate  [n_win, rate, src_width]
//   position_bias         [rate, src_width]
//   prior_kv, prior_gate  [rate, width] or nullptr  (overlap only; RAW gate,
//                         i.e. position_bias NOT pre-added)
//   out                   [n_win, width]
// The softmax is over the slot axis, INDEPENDENTLY PER CHANNEL — the reference
// softmaxes a [.., 2*rate, width] tensor along dim=2 (modeling:674), not a
// scalar-per-slot gate.
// ---------------------------------------------------------------------------
sycl::event ds4_compress_pool(sycl::queue& q,
                              const float* chunk_kv, const float* chunk_gate,
                              const float* position_bias,
                              const float* prior_kv, const float* prior_gate,
                              float* out,
                              uint32_t n_win, uint32_t rate, uint32_t width,
                              bool overlap,
                              const std::vector<sycl::event>& deps = {});

// ---------------------------------------------------------------------------
// Lightning Indexer — DeepseekV4IndexerScorer (modeling:446):
//   scores[t, e] = Σ_h ReLU(q[t, h, :] · keys[e, :]) * softmax_scale
//                        * (w_proj[t, h] * weights_scaling)
//   softmax_scale   = index_head_dim^-0.5
//   weights_scaling = index_n_heads^-0.5
//
//   q_in    [T, n_heads, head_dim]   (already RoPE'd at the query positions)
//   keys    [n_keys, head_dim]       (the indexer's compressed entries)
//   w_proj  [T, n_heads]             (the RAW weights_proj output, unscaled)
//   scores  [T, n_keys]
// ---------------------------------------------------------------------------
sycl::event ds4_indexer_score(sycl::queue& q,
                              const float* q_in, const float* keys, const float* w_proj,
                              float* scores,
                              uint32_t T, uint32_t n_heads, uint32_t head_dim,
                              uint32_t n_keys,
                              float softmax_scale, float weights_scaling,
                              const std::vector<sycl::event>& deps = {});
// V4.1 Phase 45 (docs/deepseek41/85): the same scores for the keys inside a row's CANDIDATE blocks -- `keep`
// [T, ceil(n_keys / keep_block)], nonzero = kept, keep_block even -- and `masked` written for every other key
// without running its dot products: exactly what ds4_indexer_score followed by ds41_candidate_apply produces, bit
// for bit (the kept keys' arithmetic is untouched). n_heads >= 16.
sycl::event ds4_indexer_score_candidates(sycl::queue& q,
                                         const float* q_in, const float* keys, const float* w_proj,
                                         float* scores,
                                         uint32_t T, uint32_t n_heads, uint32_t head_dim,
                                         uint32_t n_keys,
                                         float softmax_scale, float weights_scaling,
                                         const uint8_t* keep, uint32_t keep_block, float masked,
                                         const std::vector<sycl::event>& deps = {});
// Keys stored fp16 (the Phase 2 cache): same arithmetic, each key element is
// widened to fp32 as it is read.
sycl::event ds4_indexer_score(sycl::queue& q,
                              const float* q_in, const sycl::half* keys, const float* w_proj,
                              float* scores,
                              uint32_t T, uint32_t n_heads, uint32_t head_dim,
                              uint32_t n_keys,
                              float softmax_scale, float weights_scaling,
                              const std::vector<sycl::event>& deps = {});

// Top-k selection with V4's causal threshold and -1 sentinel (modeling:577-584):
//   causal_threshold[t] = (positions[t] + 1) / compress_rate
//   entries with index >= threshold are masked to -inf before the topk
//   picks that still land >= threshold come back as -1
// Emits `min(index_topk, n_keys)` indices per query in DESCENDING score order,
// ties broken by ascending entry index (a strict total order, so the result is
// deterministic).  This is a DISCRETE selection: an index mismatch is a hard
// failure, not a tolerance question.
//
//   scores    [T, n_keys]
//   positions [T] int32
//   out       [T, min(index_topk, n_keys)] int32
sycl::event ds4_indexer_topk(sycl::queue& q,
                             const float* scores, const int32_t* positions,
                             int32_t* out,
                             uint32_t T, uint32_t n_keys, uint32_t index_topk,
                             uint32_t compress_rate,
                             const std::vector<sycl::event>& deps = {});

// The radix-select shape of the above — O(n_keys) with constant SLM, the only
// shape usable at ctx-200k entry counts.  Lives in its OWN translation unit
// (src/ops/deepseek4_topk_radix.cpp — the header there records the miscompile
// that forced the split).  Call through ds4_indexer_topk, which owns the
// shape choice; this symbol is public for that dispatcher and the gate test.
// `top_k` must already be clamped to n_keys; K2 = pow2 >= top_k.
sycl::event ds4_indexer_topk_radix256(sycl::queue& q,
                                      const float* scores, const int32_t* positions,
                                      int32_t* out,
                                      uint32_t T, uint32_t n_keys, uint32_t top_k,
                                      uint32_t compress_rate, uint32_t K2,
                                      const std::vector<sycl::event>& deps = {});

// Per-query block bias over the compressed entries.
//
// HCA / dense (modeling:436-443): every causally-legal entry is visible.
//   bias[t, e] = (e >= (positions[t] + 1) / compress_rate) ? -inf : 0
sycl::event ds4_block_bias_dense(sycl::queue& q,
                                 const int32_t* positions, float* bias,
                                 uint32_t T, uint32_t n_keys, uint32_t compress_rate,
                                 const std::vector<sycl::event>& deps = {});

// CSA / sparse (modeling:699-702): only the indexer's picks are visible.
//   bias[t, :] = -inf, then bias[t, top_k_indices[t, j]] = 0 for every j with
//   top_k_indices[t, j] >= 0.
sycl::event ds4_block_bias_topk(sycl::queue& q,
                                const int32_t* top_k_indices, float* bias,
                                uint32_t T, uint32_t n_keys, uint32_t top_k,
                                const std::vector<sycl::event>& deps = {});

}  // namespace ie
