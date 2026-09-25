// include/ie/deepseek4_ops.hpp — DeepSeek-V4 compute primitives (Phase 2).
//
// Five primitives that have no equivalent in the existing op set:
//   (a) ds4_hyper_connection   — fused mHC mapping, ONE kernel launch/invocation
//   (b) ds4_unweighted_rms_norm— RMSNorm with NO learnable weight
//   (c) ds4_grouped_linear     — block-diagonal grouped projection (o_a_proj)
//   (d) ds4_router_topk / ds4_router_hash — MoE routing (sqrtsoftplus + noaux_tc)
//   (e) ds4_swiglu_clamped     — SwiGLU with V4's asymmetric clamp
//
// Conventions match include/ie/ops.hpp: device-USM pointers, deps vector,
// returns sycl::event.
//
// Precision: FP32 in / FP32 out for all five.  This is not a placeholder —
// it is what the reference does.  DeepseekV4HyperConnection.forward() casts
// its input and `fn` to fp32 explicitly (modeling_deepseek_v4.py:940-941) and
// DeepseekV4UnweightedRMSNorm reduces in fp32; the routers' score/normalise
// chain is likewise numerically fragile (sqrt of a softplus, then a
// reciprocal) and is kept in fp32 in every DeepSeek reference port.  The
// quantised weight paths (expert GEMMs, o_b_proj) are separate ops and are
// NOT in this file.

#pragma once

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ie {

// ---------------------------------------------------------------------------
// (b) UnweightedRMSNorm — DeepseekV4UnweightedRMSNorm (modeling:66)
//     y[r, j] = x[r, j] * rsqrt(mean_j(x[r, :]^2) + eps)
//     No learnable weight.  x, y are [n_rows, hidden].
// ---------------------------------------------------------------------------
sycl::event ds4_unweighted_rms_norm(sycl::queue& q,
                                    const float* x, float* y,
                                    uint32_t n_rows, uint32_t hidden,
                                    float eps,
                                    const std::vector<sycl::event>& deps = {});

// ---------------------------------------------------------------------------
// (a) Fused hyper-connection — DeepseekV4HyperConnection.forward (modeling:876)
//
//   streams   [n_tokens, hc_mult, hidden]     (flatten(2) is contiguous)
//   fn        [mix, hc_mult*hidden]           mix = (2 + hc_mult) * hc_mult
//   base      [mix]
//   scale     [3]                             (pre, post, comb)
//   post      [n_tokens, hc_mult]             out
//   comb      [n_tokens, hc_mult, hc_mult]    out (row-major, doubly stochastic)
//   collapsed [n_tokens, hidden]              out
//
// `fn` must be the row-major [mix, hc_mult*hidden] layout of the reference
// nn.Parameter.  (The GGUF stores it transposed as (hc_mult*hidden, mix); the
// loader is responsible for presenting the reference layout here.)
//
// ONE kernel launch: one work-group per token; the hc×hc Sinkhorn loop runs
// entirely in private registers of a single work-item, so the 20 iterations
// cost zero extra launches.  40 dependent launches per invocation × 2 sites ×
// 43 layers would otherwise dominate decode.
sycl::event ds4_hyper_connection(sycl::queue& q,
                                 const float* streams,
                                 const float* fn, const float* base, const float* scale,
                                 float* post, float* comb, float* collapsed,
                                 uint32_t n_tokens, uint32_t hidden, uint32_t hc_mult,
                                 uint32_t sinkhorn_iters,
                                 float rms_eps, float hc_eps,
                                 const std::vector<sycl::event>& deps = {});

// Hyper-connection with the CONSUMER RMSNorm folded into its tail.
//
// WHY.  Every one of the three hyper-connection sites in the model is spelled
//
//     ds4_hyper_connection(..., collapsed);        // launch 1 (or 2, split path)
//     ds4_rms_norm(collapsed, w, normed, T, H);    // launch 3
//
// and `collapsed` is read by NOTHING ELSE — the norm is its only consumer.  At
// decode that norm is 4096 floats: ~4.5 us of measured GPU time per call
// against a 1.43 us submit floor, i.e. nearly all fixed cost, and the model
// issues it 86 times per token per card at these two sites.  Folding it into
// the tail removes the launch AND removes a full round trip of `collapsed`
// through memory.
//
// HOW, AND WHAT IT COSTS.  The tail already splits the 4096-wide collapse
// across G2 work-groups, and an RMS needs the whole row, so each work-group
// recomputes the WHOLE collapse to obtain the sum of squares and then writes
// only its own slice.  That is a G2-fold redundant read of `streams` (64 KB at
// the real shape, L2-resident) traded against one whole launch.  The trade is
// stated explicitly rather than assumed: the implementation measures it.
//
// BIT-IDENTICAL to the two-call sequence above, not merely close.  The collapse
// accumulates in the same order, the sum of squares uses the same 4-way strided
// unroll over the same 256-wide work-group and the same `reduce_over_group`
// that `ds4_rms_norm` uses, and the store is the same `w[j] * (x[j] * r)`.
// tests/unit/deepseek4_ops_gate_test.cpp requires EXACT equality.
//
// `collapsed` is still written (it is part of the contract of the op it
// replaces).  `norm_w` is the RMSNorm's learnable weight [hidden]; `normed` is
// [n_tokens, hidden].  Shapes this op's split path does not cover fall back to
// the two-launch sequence internally, so it is always correct and merely stops
// saving a launch.
sycl::event ds4_hyper_connection_norm(sycl::queue& q,
                                      const float* streams,
                                      const float* fn, const float* base, const float* scale,
                                      const float* norm_w,
                                      float* post, float* comb, float* collapsed, float* normed,
                                      uint32_t n_tokens, uint32_t hidden, uint32_t hc_mult,
                                      uint32_t sinkhorn_iters,
                                      float rms_eps, float hc_eps, float norm_eps,
                                      const std::vector<sycl::event>& deps = {});

// ---------------------------------------------------------------------------
// (c) Grouped linear — DeepseekV4GroupedLinear.forward (modeling:303)
//
//   x [n_tokens, n_groups, in_per_group]
//   w [n_groups * out_per_group, in_per_group]   (nn.Linear layout; the block
//                                                 for group g is rows
//                                                 [g*out_per_group, (g+1)*out_per_group))
//   y [n_tokens, n_groups, out_per_group]
//
//   y[t, g, o] = Σ_h x[t, g, h] * w[g*out_per_group + o, h]
//
// V4-Flash: n_groups = o_groups = 8, in_per_group = n_heads*head_dim/8 = 4096,
// out_per_group = o_lora_rank = 1024.
sycl::event ds4_grouped_linear(sycl::queue& q,
                               const float* x, const float* w, float* y,
                               uint32_t n_tokens, uint32_t n_groups,
                               uint32_t in_per_group, uint32_t out_per_group,
                               const std::vector<sycl::event>& deps = {});

// ---------------------------------------------------------------------------
// (d) MoE routing.
//
// DeepseekV4TopKRouter.forward (modeling:1033), transcribed:
//     logits  = x @ Wᵀ
//     scores  = sqrt(softplus(logits))
//     indices = topk(scores + e_score_correction_bias, k).indices
//     weights = scores.gather(1, indices)                 <-- RAW scores, NO bias
//     weights = weights / (weights.sum(-1) + 1e-20)
//     return logits, weights * routed_scaling_factor, indices
//
// *** The noaux_tc bias enters the ARGSORT ONLY.  The returned weight is the
//     unbiased score at the biased-selected index.  Adding the bias to the
//     returned weight is the classic silent-accuracy bug in V3/V4 ports. ***
//
// `bias` may be null (== all-zero correction bias).
// `indices` are emitted in DESCENDING (scores+bias) order, which is what
// torch.topk(..., sorted=False) actually produces on CPU/CUDA and what the
// reference blobs contain; ties resolve to the lower expert index.
//
// The renormalisation is unconditional, exactly as in the reference — V4's
// config sets norm_topk_prob=true and the reference code has no branch on it.
//
//   x       [n_tokens, hidden]
//   w       [n_experts, hidden]
//   bias    [n_experts] or nullptr
//   logits  [n_tokens, n_experts]   out
//   weights [n_tokens, top_k]       out (already × routed_scaling)
//   indices [n_tokens, top_k]       out
sycl::event ds4_router_topk(sycl::queue& q,
                            const float* x, const float* w, const float* bias,
                            float* logits, float* weights, int32_t* indices,
                            uint32_t n_tokens, uint32_t hidden,
                            uint32_t n_experts, uint32_t top_k,
                            float routed_scaling,
                            const std::vector<sycl::event>& deps = {});

// DeepseekV4HashRouter.forward (modeling:1054): identical scoring/normalise
// chain, but selection is the frozen table lookup `tid2eid[input_ids]` — no
// argsort, no bias.
//
//   tid2eid   [vocab, top_k]  (device)
//   input_ids [n_tokens]      (device)
sycl::event ds4_router_hash(sycl::queue& q,
                            const float* x, const float* w,
                            const int32_t* tid2eid, const int32_t* input_ids,
                            float* logits, float* weights, int32_t* indices,
                            uint32_t n_tokens, uint32_t hidden,
                            uint32_t n_experts, uint32_t top_k,
                            float routed_scaling,
                            const std::vector<sycl::event>& deps = {});

// Vision-Exp variants (inference/model.py Gate.forward with `bias_vl`).  Only
// dispatched for a forward chunk that carries IMAGE rows; the two functions
// above are untouched so the text path stays bit-identical.
//   img_mask [n_tokens]  1 for an image-block row, 0 for a text token
//   bias_vl  [n_experts] the `exp_probs_b_vl` bias
// topk: image rows select over scores + bias_vl, text rows over scores + bias
//       (bias may be null); weights come from the UNBIASED scores either way.
// hash: image rows select top-k over scores + bias_vl (they never hash-route);
//       text rows take the frozen table like ds4_router_hash.
sycl::event ds4_router_topk_vl(sycl::queue& q,
                               const float* x, const float* w, const float* bias,
                               const float* bias_vl, const int32_t* img_mask,
                               float* logits, float* weights, int32_t* indices,
                               uint32_t n_tokens, uint32_t hidden,
                               uint32_t n_experts, uint32_t top_k,
                               float routed_scaling,
                               const std::vector<sycl::event>& deps = {});
sycl::event ds4_router_hash_vl(sycl::queue& q,
                               const float* x, const float* w,
                               const int32_t* tid2eid, const int32_t* input_ids,
                               const float* bias_vl, const int32_t* img_mask,
                               float* logits, float* weights, int32_t* indices,
                               uint32_t n_tokens, uint32_t hidden,
                               uint32_t n_experts, uint32_t top_k,
                               float routed_scaling,
                               const std::vector<sycl::event>& deps = {});

// ---------------------------------------------------------------------------
// (e) Clamped SwiGLU — DeepseekV4MLP.forward (modeling:974) and
//     DeepseekV4Experts._apply_gate.
//
//     gate = gate.clamp(max=limit)              <-- UPPER bound only
//     up   = up.clamp(min=-limit, max=limit)    <-- both sides
//     y    = silu(gate) * up
//
// Element-wise over `n` elements; `gate`, `up`, `y` may alias `gate`/`up`.
sycl::event ds4_swiglu_clamped(sycl::queue& q,
                               const float* gate, const float* up, float* y,
                               size_t n, float limit,
                               const std::vector<sycl::event>& deps = {});

// Float inputs, half output: the same fp32 arithmetic followed by the same
// half rounding as ds4_swiglu_clamped + cast_fp32_to_fp16, in one launch.
// Output storage must not overlap either input. Zero work preserves deps;
// nonzero work requires nonnull operands and representable byte spans.
sycl::event ds4_swiglu_clamped_to_f16(sycl::queue& q,
                                      const float* gate, const float* up,
                                      sycl::half* y, size_t n, float limit,
                                      const std::vector<sycl::event>& deps = {});

// MiMo-V2.6 (docs/mimo26/P7_FIX64_FIX70.md section 6, fix-list #74): the rows of ds4_swiglu_clamped_to_f16's output that
// overflowed. Over `rows` rows of `EF`: a row whose fp16 products hold an inf / NaN (by bits: a product >= 65520 rounds to
// inf) is recomputed from the fp32 gate / up -- the same expression -- times 2^-k, k the smallest that keeps its largest
// product under 65520, and k is written to shift[row]; a clean row is NOT written (shift[row] = 0), so it stays bit for
// bit what ds4_swiglu_clamped_to_f16 stored. A non-finite fp32 product stays non-finite (k is capped).
sycl::event ds4_swiglu_f16_rescale_overflow(sycl::queue& q, const float* gate, const float* up, sycl::half* y, int32_t* shift,
                                            uint32_t rows, uint32_t EF, float limit, const std::vector<sycl::event>& deps = {});
// The same for ds4_swiglu_clamped_h's rows (the int-dot / decode route: gate / up are fp16 there). A row rescaled by 2^-k
// quantizes (quantize_q8_1: a per-32-block scale) to the same q8 codes, so the int-dot down row is exactly 2^-k times.
sycl::event ds4_swiglu_h_rescale_overflow(sycl::queue& q, const sycl::half* gate, const sycl::half* up, sycl::half* y, int32_t* shift,
                                          uint32_t rows, uint32_t EF, float limit, const std::vector<sycl::event>& deps = {});
// w[r] *= 2^shift[r] (exact): the routing weight of a row stored x 2^-k, so a fp32 scatter of the linear down GEMM's
// output adds exactly the unscaled row.
sycl::event ds4_scale_pow2_rows(sycl::queue& q, float* w, const int32_t* shift, uint32_t rows,
                                const std::vector<sycl::event>& deps = {});

// fp16-in / fp16-out SwiGLU — collapses a FOUR-launch chain into ONE.
//
// Every routed-expert site in the engine spells the same thing:
//
//     cast_fp16_to_fp32(gate_h, gate_f, n);     // launch 1
//     cast_fp16_to_fp32(up_h,   up_f,   n);     // launch 2
//     ds4_swiglu_clamped(gate_f, up_f, h_f, n); // launch 3
//     cast_fp32_to_fp16(h_f,    h_h,    n);     // launch 4
//
// because the expert GEMMs are fp16 and this op was fp32-only.  At decode the
// three cast launches move a few KB each; they are pure submission overhead
// (~1 us apiece of host time, and decode issues thousands of launches per
// token).  This entry point does the same arithmetic in the same order and
// removes three launches per call.
//
// BIT-IDENTICAL to the four-launch chain, not merely close.  fp16 -> fp32 is
// exact (fp16 is a subset of fp32), the interior arithmetic is the same fp32
// expression, and the final store is the same `sycl::half(...)` rounding that
// cast_fp32_to_fp16 applies (src/ops/elementwise.cpp:378).  There is no
// intermediate that the chain rounds and this does not, or vice versa.
// tests/unit/deepseek4_ops_gate_test.cpp checks that over the full fp16 input
// range and requires EXACT equality, not a tolerance.
//
// `y` may alias `gate` or `up`.
sycl::event ds4_swiglu_clamped_h(sycl::queue& q,
                                 const sycl::half* gate, const sycl::half* up,
                                 sycl::half* y, size_t n, float limit,
                                 const std::vector<sycl::event>& deps = {});

}  // namespace ie
