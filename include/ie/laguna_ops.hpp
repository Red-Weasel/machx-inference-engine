// include/ie/laguna_ops.hpp — Laguna S 2.1 primitive ops (port phase P3).
//
// NOT A RUNTIME.  docs/laguna/00_PORT_PLAN.md.
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>
#include <vector>

namespace ie {

// Per-head attention output gating — `LagunaAttention.forward`, applied BEFORE
// o_proj:  attn[t, h, :] *= softplus(gate_logits[t, h]).
//
// ONE scalar per head, broadcast across head_dim (the GGUF's
// `attn_gate.weight` is [hidden, heads], which settles per-head vs per-channel).
// softplus is evaluated in fp32 in the stable form — the naive log(1+exp(x))
// overflows to inf past x ~ 88 and an inf gate silently destroys the layer.
// `n_heads` is PER LAYER on this model (48 full / 72 sliding).
//
//   attn        : [T, n_heads*head_dim] fp32, modified IN PLACE
//   gate_logits : [T, n_heads] fp32, the RAW g_proj output (pre-softplus)
sycl::event lag_attn_gate(sycl::queue& q, float* attn, const float* gate_logits,
                          uint32_t T, uint32_t n_heads, uint32_t head_dim,
                          const std::vector<sycl::event>& deps = {});

// RoPE apply — `apply_rotary_pos_emb` (upstream modeling_laguna.py:271).
//
// NOT interchangeable with `ds4_rope_apply`: Laguna rotates the LEADING
// `rope_dim` dims with NON-INTERLEAVED rotate_half pairing (d pairs with
// d + rope_dim/2); DeepSeek-V4 rotates the TRAILING slice with ADJACENT pairing.
// The YaRN frequency TABLE is shared (`ds4_rope_inv_freq`); the apply is not.
// `rope_dim` is PER LAYER (64 full-attention / 128 sliding).
//
//   x, y [T, n_heads, head_dim] fp32 (may alias)   cos, sin [T, rope_dim] fp32
sycl::event lag_rope_apply(sycl::queue& q, const float* x, const float* cos_in,
                           const float* sin_in, float* y,
                           uint32_t T, uint32_t n_heads, uint32_t head_dim,
                           uint32_t rope_dim,
                           const std::vector<sycl::event>& deps = {});

// MoE router — `LagunaTopKRouter.forward` (upstream modeling_laguna.py:169).
//
// SIGMOID scoring, and the WEIGHTS ARE THAT SAME SIGMOID (renormalised to sum 1
// when `norm_topk_prob`).  Contrast Inkling, whose weights are a softmax over
// logsigmoid taken JOINTLY with its shared experts — the two routers are not
// interchangeable and neither failure mode crashes.
// `e_score_correction_bias` shifts SELECTION ONLY; the weights come from the
// UNBIASED scores (aux-loss-free balancing, arXiv:2408.15664).
// `softcap <= 0` disables softcapping (it is 0 on this checkpoint).
// Laguna's shared expert is SEPARATE — not in the gate and not in this sum.
//
//   logits [T, n_experts] (raw)   bias [n_experts]
//   -> out_idx [T,k] int32, out_w [T,k].  k <= 32.
sycl::event lag_moe_router(sycl::queue& q, const float* logits, const float* bias,
                           int32_t* out_idx, float* out_w,
                           uint32_t T, uint32_t n_experts, uint32_t k,
                           bool norm_topk_prob, float softcap,
                           const std::vector<sycl::event>& deps = {});

}  // namespace ie
