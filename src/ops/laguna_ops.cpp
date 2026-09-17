// src/ops/laguna_ops.cpp — Laguna S 2.1 primitive ops (port phase P3).
//
// SCOPE, HONEST: individual operators, each gated against the reference in
// isolation.  There is no Laguna runtime yet; this file exists so the one piece
// of that architecture the engine has never built is proven before assembly.
// Everything else Laguna needs (YaRN + partial rotary, alternating SWA, QK
// RMSNorm, top-k MoE with expert streaming) already has a sibling in this
// engine — see docs/laguna/00_PORT_PLAN.md.
//
// Reference: ~/models/... upstream `modeling_laguna.py` (fetched 2026-08-09).

#include "ie/laguna_ops.hpp"

#include "ie/kernel_profiler.hpp"

#include <sycl/sycl.hpp>

namespace ie {

namespace {
constexpr int kWG = 256;
}

// PER-HEAD ATTENTION OUTPUT GATING — `LagunaAttention.forward` (the
// `if self.gating:` block, upstream modeling_laguna.py):
//
//     gate = softplus(g_proj(hidden_states).float()).to(dtype)   # [T, n_heads]
//     attn = (attn.view(T, n_heads, head_dim) * gate.unsqueeze(-1)).view(...)
//
// applied BEFORE o_proj, not after — the gate scales the per-head attention
// outputs while they are still separable, so folding it into o_proj's input is
// the only correct place for it.
//
// THREE THINGS THIS ENCODES:
// 1. `gating: "per-head"` means ONE scalar per head, BROADCAST across head_dim
//    (`gate.unsqueeze(-1)`).  The alternative the upstream code also supports
//    (`gating: True`) is one scalar per CHANNEL, i.e. g_proj emits
//    n_heads*head_dim.  The GGUF settles it: `attn_gate.weight` is
//    [hidden, heads] — per-head — and the shape is asserted by
//    laguna_manifest_test.
// 2. softplus is computed in FP32 (`.float()` in the reference) even when the
//    tensors are bf16, and softplus(x) = log1p(exp(x)) is evaluated in the
//    numerically stable form: for large x it is x + log1p(exp(-x)), which
//    avoids exp() overflowing.  The naive log(1+exp(x)) returns inf for
//    x > ~88 in fp32; a gate of inf silently destroys the layer.
// 3. `n_heads` is PER LAYER on this model (48 on full-attention layers, 72 on
//    sliding), so the caller passes the layer's own count — there is no model-
//    wide head count to cache.
//
//   attn : [T, n_heads * head_dim] fp32, modified IN PLACE
//   gate_logits : [T, n_heads] fp32 (the raw g_proj output, pre-softplus)
sycl::event lag_attn_gate(sycl::queue& q, float* attn, const float* gate_logits,
                          uint32_t T, uint32_t n_heads, uint32_t head_dim,
                          const std::vector<sycl::event>& deps) {
    return ie::ps(q, "lag_attn_gate", [&](sycl::handler& h) {
        h.depends_on(deps);
        const uint64_t total  = uint64_t(T) * n_heads * head_dim;
        const uint64_t global = ((total + kWG - 1) / kWG) * kWG;
        h.parallel_for(sycl::nd_range<1>(global, kWG), [=](sycl::nd_item<1> it) {
            const uint64_t i = it.get_global_id(0);
            if (i >= total) return;
            const uint32_t t = uint32_t(i / (uint64_t(n_heads) * head_dim));
            const uint32_t hd = uint32_t((i / head_dim) % n_heads);
            const float x = gate_logits[uint64_t(t) * n_heads + hd];
            // Stable softplus: log1p(exp(-|x|)) + max(x, 0).
            const float sp = sycl::log1p(sycl::exp(-sycl::fabs(x))) + sycl::fmax(x, 0.f);
            attn[i] *= sp;
        });
    });
}

// ROPE APPLY — `apply_rotary_pos_emb` (upstream modeling_laguna.py:271) over
// `rotate_half` (:263).
//
// THIS IS NOT THE DEEPSEEK-V4 ROPE, AND THE DIFFERENCE IS SILENT.  Both models
// use PARTIAL rotary, so it is tempting to reuse `ds4_rope_apply`.  Two things
// differ, either of which produces plausible-looking garbage:
//
//   1. WHICH DIMS ROTATE.  Laguna takes the LEADING `rope_dim` dims
//      (`q[..., :rotary_dim]` rotates, `q[..., rotary_dim:]` passes through).
//      DeepSeek-V4 rotates the TRAILING slice (its `nope` channels come first).
//   2. THE PAIRING.  Laguna is NON-INTERLEAVED "rotate_half": the rotary block
//      splits in two and dim d pairs with d + rope_dim/2 — the docstring says
//      so outright ("Removes the interleaving of cos and sin from GLM").
//      DeepSeek-V4 pairs ADJACENT dims (2p, 2p+1).
//
// What IS shared is the frequency table: Laguna's full-attention layers use the
// same YaRN NTK-by-parts form `ds4_rope_inv_freq` already computes, and its
// sliding layers use the plain form.  So reuse the TABLE, not the APPLY.
//
//     out[d]              = x[d]*cos[d] - x[d + h]*sin[d]        for d < h
//     out[d + h]          = x[d + h]*cos[d + h] + x[d]*sin[d + h]
//     out[>= rope_dim]    = x[>= rope_dim]                        (passthrough)
//   where h = rope_dim/2, and cos/sin are [T, rope_dim] with the reference's
//   duplicated layout (cos[d] == cos[d + h]).
//
//   x, y : [T, n_heads, head_dim] fp32 (may alias: each work-item reads both
//          partners before writing, and writes only its own pair)
//   cos, sin : [T, rope_dim] fp32
sycl::event lag_rope_apply(sycl::queue& q, const float* x, const float* cos_in,
                           const float* sin_in, float* y,
                           uint32_t T, uint32_t n_heads, uint32_t head_dim,
                           uint32_t rope_dim,
                           const std::vector<sycl::event>& deps) {
    return ie::ps(q, "lag_rope_apply", [&](sycl::handler& h) {
        h.depends_on(deps);
        const uint64_t total  = uint64_t(T) * n_heads * head_dim;
        const uint64_t global = ((total + kWG - 1) / kWG) * kWG;
        const uint32_t half   = rope_dim / 2u;
        h.parallel_for(sycl::nd_range<1>(global, kWG), [=](sycl::nd_item<1> it) {
            const uint64_t i = it.get_global_id(0);
            if (i >= total) return;
            const uint32_t d = uint32_t(i % head_dim);
            const uint32_t t = uint32_t(i / (uint64_t(n_heads) * head_dim));
            if (d >= rope_dim) { if (y != x) y[i] = x[i]; return; }   // passthrough
            if (d >= half) return;                 // the low lane owns the pair
            const uint64_t lo = i, hi = i + half;
            const float x1 = x[lo], x2 = x[hi];
            const float c1 = cos_in[uint64_t(t) * rope_dim + d];
            const float s1 = sin_in[uint64_t(t) * rope_dim + d];
            const float c2 = cos_in[uint64_t(t) * rope_dim + d + half];
            const float s2 = sin_in[uint64_t(t) * rope_dim + d + half];
            y[lo] = x1 * c1 - x2 * s1;             // rotate_half: (-x2, x1)
            y[hi] = x2 * c2 + x1 * s2;
        });
    });
}

// MoE ROUTER — `LagunaTopKRouter.forward` (upstream modeling_laguna.py:169).
//
// SIGMOID SCORING, NOT SOFTMAX (the class docstring leads with it), and the
// difference from Inkling's router is total even though both models are
// 256-expert MoEs by the same vendor-adjacent lineage:
//
//   Laguna:  score  = sigmoid(logit)                      <- the WEIGHT too
//            select = topk(score + e_score_correction_bias)
//            weight = score[selected], then NORMALISED to sum 1 if
//                     `norm_topk_prob` (true on this model)
//            shared expert is SEPARATE — not in the gate, not in the sum
//
//   Inkling: select = topk(sigmoid(logit) + bias)   but
//            weight = softmax over LOGSIGMOID of the RAW logits, JOINTLY with
//                     the shared experts
//
// So: Laguna's weights come from the SAME sigmoid used for selection (Inkling's
// do not), and Laguna renormalises the top-k to sum 1 (Inkling's normalisation
// includes the shared experts and does not).  Wiring either router the other
// way produces a model that runs and is quietly wrong.
//
// AUX-LOSS-FREE LOAD BALANCING (arXiv:2408.15664): `e_score_correction_bias`
// shifts SELECTION only — the returned weights are read from the UNBIASED
// scores.  Adding the bias into the weight is the classic misreading and it
// changes every routed contribution.
//
// SOFTCAPPING: `moe_router_logit_softcapping` is 0 (disabled) on this
// checkpoint, so it is NOT implemented here rather than implemented untested —
// pass `softcap <= 0`.  A future checkpoint that enables it must add
// `tanh(logit/c)*c` BEFORE the sigmoid and re-gate.
//
// Selection order: descending score, smaller index first on ties (a strict
// total order, so deterministic).  The downstream expert sum is
// order-independent; the gate compares the (expert -> weight) mapping.
//
//   logits [T, n_experts] fp32 (RAW router output)   bias [n_experts] fp32
//   out_idx [T, k] int32   out_w [T, k] fp32
sycl::event lag_moe_router(sycl::queue& q, const float* logits, const float* bias,
                           int32_t* out_idx, float* out_w,
                           uint32_t T, uint32_t n_experts, uint32_t k,
                           bool norm_topk_prob, float softcap,
                           const std::vector<sycl::event>& deps) {
    return ie::ps(q, "lag_moe_router", [&](sycl::handler& h) {
        h.depends_on(deps);
        const uint32_t global = ((T + kWG - 1) / kWG) * kWG;
        h.parallel_for(sycl::nd_range<1>(global, kWG), [=](sycl::nd_item<1> it) {
            const uint32_t t = uint32_t(it.get_global_id(0));
            if (t >= T) return;
            const float* row = logits + uint64_t(t) * n_experts;
            constexpr uint32_t kMaxK = 32;
            uint32_t sel[kMaxK];
            float    w[kMaxK];
            float prev_s = 0.f;
            uint32_t prev_i = 0;
            for (uint32_t r = 0; r < k && r < kMaxK; ++r) {
                float best_s = -1e30f, best_score = 0.f;
                uint32_t best_i = 0;
                bool found = false;
                for (uint32_t e = 0; e < n_experts; ++e) {
                    float lg = row[e];
                    if (softcap > 0.f) lg = sycl::tanh(lg / softcap) * softcap;
                    const float score = 1.f / (1.f + sycl::exp(-lg));   // sigmoid
                    const float sc = score + bias[e];                   // SELECTION only
                    const bool after = r == 0 ? true
                                    : (sc < prev_s || (sc == prev_s && e > prev_i));
                    if (!after) continue;
                    if (!found || sc > best_s || (sc == best_s && e < best_i)) {
                        best_s = sc; best_i = e; best_score = score; found = true;
                    }
                }
                sel[r] = best_i;
                w[r]   = best_score;     // UNBIASED score is the weight
                prev_s = best_s; prev_i = best_i;
            }
            if (norm_topk_prob) {
                float sum = 0.f;
                for (uint32_t r = 0; r < k; ++r) sum += w[r];
                const float inv = (sum > 0.f) ? (1.f / sum) : 0.f;
                for (uint32_t r = 0; r < k; ++r) w[r] *= inv;
            }
            for (uint32_t r = 0; r < k; ++r) {
                out_idx[uint64_t(t) * k + r] = int32_t(sel[r]);
                out_w[uint64_t(t) * k + r]   = w[r];
            }
        });
    });
}

}  // namespace ie
