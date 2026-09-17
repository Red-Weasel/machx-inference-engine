// include/ie/inkling_ops.hpp — Inkling-Small primitive ops (port phase P3).
//
// NOT A RUNTIME.  These are the architecture's novel operators, built and gated
// one at a time ahead of the forward pass (docs/inkling/00_PORT_PLAN.md).
// Reference: ~/models/inkling-reference/modeling_inkling.py (transformers
// 5.14.1).
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>
#include <vector>

namespace ie {

// Depthwise CAUSAL short convolution with a residual add —
// `InklingShortConvolution` (modeling_inkling.py:501) over `causal_conv1d_fn`
// (:461).  Four of these run per layer (attn / k / v / mlp streams).
//
//   y[t][c] = x[t][c] + sum_{j=0..K-1} w[j][c] * x[t-(K-1)+j][c]   (x[<0] = 0)
//
//   x, y : [T, C] fp32, row-major, MUST NOT ALIAS (position t reads t-K+1..t)
//   w    : [C, K] fp32 — CHANNEL-MAJOR, taps contiguous: the GGUF layout
//          verbatim (printed [4, C]; ne[0]=4 is the row length, so a row IS the
//          4 taps of one channel).  No load-time transpose needed.
//
// No bias: the reference builds its Conv1d with bias=False and the GGUF carries
// none, so this API cannot express one.  Channels never mix (groups == C).
// fp32 accumulation, matching the reference's explicit .float() cast.
sycl::event ink_shortconv(sycl::queue& q, const float* x, const float* w, float* y,
                          uint32_t T, uint32_t C, uint32_t K,
                          const std::vector<sycl::event>& deps = {});

// Relative position bias — `InklingRelativeLogits` (modeling_inkling.py:131).
//
//   bias[h][i][j] = (0 <= d < rel_extent)
//                     ? sum_k relative_states[i][h][k] * proj[k][d]  where
//                       d = q_pos[i] - k_pos[j]
//                     : 0
//
// Causality/padding are NOT applied here — they live in the attention mask,
// exactly as in the reference.  Fused: no [H,T,rel_extent] scratch.
//
//   relative_states : [T, n_heads, d_rel] fp32   proj : [d_rel, rel_extent] fp32
//   q_pos : [T] int32   k_pos : [n_kv] int32     bias : [n_heads, T, n_kv] fp32
sycl::event ink_rel_logits(sycl::queue& q, const float* relative_states, const float* proj,
                           const int32_t* q_pos, const int32_t* k_pos, float* bias,
                           uint32_t T, uint32_t n_heads, uint32_t d_rel,
                           uint32_t rel_extent, uint32_t n_kv,
                           const std::vector<sycl::event>& deps = {});

// MoE router — `InklingTopkRouter.forward` (modeling_inkling.py:356).
//
// THE SHARED EXPERTS ARE NORMALISED JOINTLY WITH THE ROUTED ONES: the k chosen
// routed logits and both shared logits go through ONE log-softmax, so a shared
// expert's weight depends on which routed experts won.  A separate additive
// shared term — what every other MoE in this engine does — is WRONG here.
// Selection uses sigmoid(logit) + bias; the WEIGHTS use logsigmoid of the RAW
// logit.  Winners are emitted in descending score, smaller index first on ties.
//
//   logits [T, n_routed+n_shared], bias [n_routed] (e_score_correction_bias)
//   -> out_idx [T,k] int32, out_w [T,k], out_shared [T,n_shared] (shared_gammas)
//   k <= 16.  scale = route_scale * global_scale (config expert_weights_scale
//   and the `ffn_gscale` tensor).
sycl::event ink_moe_router(sycl::queue& q, const float* logits, const float* bias,
                           int32_t* out_idx, float* out_w, float* out_shared,
                           uint32_t T, uint32_t n_routed, uint32_t n_shared,
                           uint32_t k, float route_scale, float global_scale,
                           const std::vector<sycl::event>& deps = {});

// Length-dependent attention temperature — `InklingAttention.forward`
// (modeling_inkling.py:254-261).
//
//   tau[i] = 1 + alpha * log( max( (q_pos[i]+1) / n_floor, 1 ) )
//
// ONLY on FULL-ATTENTION layers (`!InklingConfig::is_sliding(L)` — 7 of 42 on
// this model), and the caller must scale BOTH the query AND the position bias
// by it, not just the query.  The clamp is on the RATIO (inside the log), so
// tau == 1 below n_floor (128000 here) and only diverges past it.
sycl::event ink_log_scaling_tau(sycl::queue& q, const int32_t* q_pos, float* tau,
                                uint32_t T, float alpha, float n_floor,
                                const std::vector<sycl::event>& deps = {});

// Attention — `eager_attention_forward` (modeling_inkling.py:157) with the
// relative position bias, online-softmax in fp32 (the reference softmaxes in
// float32 explicitly, so this is its behaviour, not an upgrade).
// GQA `repeat_kv` is applied as an INDEX MAP (query head h reads KV head
// h/(n_heads/n_kv_heads)), never materialised.  Masked columns are skipped
// before the dot product.
//
//   q [T,n_heads,head_dim]  k,v [n_kv,n_kv_heads,head_dim]
//   bias [n_heads,T,n_kv] or null   mask [T,n_kv] or null (additive)
//   out [T,n_heads,head_dim].  head_dim <= 16*64.
sycl::event ink_attention(sycl::queue& q_, const float* q, const float* k, const float* v,
                          const float* bias, const float* mask, float* out,
                          uint32_t T, uint32_t n_heads, uint32_t n_kv_heads,
                          uint32_t head_dim, uint32_t n_kv, float scaling,
                          const std::vector<sycl::event>& deps = {});

// ---- assembly helpers (P2) — generic fp32, NOT the performance path --------
// They exist so a decoder layer can be composed and gated against the reference
// before the loader and the engine's tuned GEMMs are wired in.

// y[T,N] = x[T,K] * w[N,K]^T (weights output-major, as GGUF and torch store).
sycl::event ink_matmul_nt(sycl::queue& q, const float* x, const float* w, float* y,
                          uint32_t T, uint32_t K, uint32_t N,
                          const std::vector<sycl::event>& deps = {});

// RMSNorm over the last axis of [rows, dim] — `InklingRMSNorm`.  Serves the
// layer norms (dim = hidden) and the PER-HEAD q/k norms (rows = T*heads,
// dim = head_dim) alike.
sycl::event ink_rms_norm(sycl::queue& q, const float* x, const float* w, float* y,
                         uint32_t rows, uint32_t dim, float eps,
                         const std::vector<sycl::event>& deps = {});

// y = silu(gate) * up.
sycl::event ink_swiglu(sycl::queue& q, const float* gate, const float* up, float* y,
                       uint64_t n, const std::vector<sycl::event>& deps = {});

// y = a + b.
sycl::event ink_add(sycl::queue& q, const float* a, const float* b, float* y,
                    uint64_t n, const std::vector<sycl::event>& deps = {});

}  // namespace ie
