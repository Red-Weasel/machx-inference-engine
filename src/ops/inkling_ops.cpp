// src/ops/inkling_ops.cpp — Inkling-Small primitive ops (port phase P3).
//
// SCOPE, HONEST: individual operators, each gated against the reference in
// isolation.  There is no Inkling runtime and no forward pass yet; this file
// exists so the novel pieces of that architecture are built and PROVEN one at a
// time rather than all at once inside a forward that cannot be debugged.  That
// is the sequence the DeepSeek-V4 port used and the reason its numerics held.
//
// Reference: models/inkling-reference/modeling_inkling.py
// (transformers 5.14.1, extracted 2026-08-09; the working venv stays at 5.12.0).

#include "ie/inkling_ops.hpp"

#include "ie/kernel_profiler.hpp"

#include <sycl/sycl.hpp>

#include <limits>

namespace ie {

namespace {
constexpr int kWG = 256;
constexpr int kSGNorm = 16;
}

// SHORT CONVOLUTION — `InklingShortConvolution.forward` (modeling_inkling.py:501)
// over `causal_conv1d_fn` (:461).
//
// THE REFERENCE, UNROLLED.  `causal_conv1d_fn` runs a DEPTHWISE conv1d
// (groups == channels, so each channel has its own K taps and channels never
// mix) with `padding = K-1` and then TRUNCATES to the input length:
//
//     out = F.conv1d(x, w.unsqueeze(1), bias, padding=K-1, groups=C)[:, :, :L]
//
// Left-padding by K-1 and keeping the FIRST L outputs is what makes it causal:
// output position t sees input positions t-(K-1) .. t and nothing later, with
// out-of-range positions reading as zero.  Written out, for channel c:
//
//     y_conv[t][c] = sum_{j=0..K-1} w[c][j] * x[t - (K-1) + j][c]      (x[<0] = 0)
//
// The module then adds the residual and that residual is the ORIGINAL input,
// captured BEFORE the padding mask and the convolution (:512) — not the
// convolved value:
//
//     y[t][c] = x[t][c] + y_conv[t][c]
//
// FP32 THROUGHOUT: the module casts to float on entry (:510) and back only at
// the end, so the accumulation is fp32 regardless of the weight dtype.
//
// WEIGHT LAYOUT IS [C][K] — CHANNEL-MAJOR, TAPS CONTIGUOUS — and this was got
// WRONG first (2026-08-09).  GGUF prints `shortconv_attn.weight` as [4, 4096]
// and ne[0] is the ROW LENGTH, so a row is 4 taps and there are C rows: the
// file stores channel c's four taps adjacently.  The kernel originally indexed
// w[j*C + c] ([K][C]) and its golden test AGREED, because the golden generator
// was written from the same wrong assumption — a self-consistent pair that
// tested nothing about the real file.  `inkling_manifest_test`, which asserts
// shapes against the ACTUAL GGUF, is what exposed it.  Lesson already recorded
// for `attn_rel_proj`: an orientation is not verified until a real tensor says
// so.  Indexing is now w[c*K + j], matching the file with no load-time
// transpose.
//
// NO BIAS: the module constructs `nn.Conv1d(..., bias=False)` (:497), and the
// GGUF carries no shortconv bias tensor.  A bias argument is therefore NOT
// accepted here — an API that cannot express the wrong thing.
//
// One work-item per (token, channel).  Every read is from the ORIGINAL input,
// so `x` and `y` MUST NOT alias: position t reads t-3..t, which a later
// work-item may already have overwritten.  The caller supplies distinct
// buffers; `ink_shortconv` does not check because a device-side aliasing check
// is not possible and a silent wrong answer is worse than a documented rule.
sycl::event ink_shortconv(sycl::queue& q, const float* x, const float* w, float* y,
                          uint32_t T, uint32_t C, uint32_t K,
                          const std::vector<sycl::event>& deps) {
    return ie::ps(q, "ink_shortconv", [&](sycl::handler& h) {
        h.depends_on(deps);
        const uint64_t total  = uint64_t(T) * C;
        const uint64_t global = ((total + kWG - 1) / kWG) * kWG;
        h.parallel_for(sycl::nd_range<1>(global, kWG), [=](sycl::nd_item<1> it) {
            const uint64_t i = it.get_global_id(0);
            if (i >= total) return;
            const uint32_t t = uint32_t(i / C);
            const uint32_t c = uint32_t(i % C);
            float acc = 0.f;
            // Ascending j — the same order F.conv1d accumulates its taps.
            for (uint32_t j = 0; j < K; ++j) {
                const int64_t src = int64_t(t) - int64_t(K - 1) + int64_t(j);
                if (src < 0) continue;                      // left zero padding
                acc += w[uint64_t(c) * K + j] * x[uint64_t(src) * C + c];
            }
            y[i] = x[i] + acc;                              // residual (:542)
        });
    });
}

// RELATIVE POSITION BIAS — `InklingRelativeLogits.forward`
// (modeling_inkling.py:131).  The reference, in three lines:
//
//     rel_logits   = (relative_states @ proj).transpose(1, 2)   # [H, T, extent]
//     distance     = q_pos[:, None] - k_pos[None, :]            # [T, n_kv]
//     position_bias = rel_logits.gather(-1, clamp(distance, 0, extent-1))
//                       .masked_fill(distance < 0 | distance >= extent, 0)
//
// `proj` is a trained bank of bias-vs-distance profiles (the class docstring):
// each token mixes the d_rel profiles into ONE bias value per BACKWARD distance.
// So the bias for (head h, query i, key j) is the profile value at distance
// q_pos[i] - k_pos[j], and it is ZERO outside [0, rel_extent) — causality and
// padding are NOT this op's job, they stay in the attention mask.
//
// FUSED, no intermediate.  The reference materialises `rel_logits`
// [H, T, rel_extent] first and then gathers; here each output element does its
// own d_rel-long dot (d_rel is 16 on this model), which removes the scratch
// buffer entirely.  The clamp is deliberately NOT applied to the value read:
// the reference clamps only to make `gather` legal and then MASKS those lanes
// to zero, so reading them at all would be wasted work — the range test comes
// first here and short-circuits.
//
//   relative_states : [T, n_heads, d_rel] fp32
//   proj            : [d_rel, rel_extent] fp32
//   q_pos           : [T] int32,  k_pos : [n_kv] int32
//   bias            : [n_heads, T, n_kv] fp32  (the layout attention wants)
sycl::event ink_rel_logits(sycl::queue& q, const float* relative_states, const float* proj,
                           const int32_t* q_pos, const int32_t* k_pos, float* bias,
                           uint32_t T, uint32_t n_heads, uint32_t d_rel,
                           uint32_t rel_extent, uint32_t n_kv,
                           const std::vector<sycl::event>& deps) {
    return ie::ps(q, "ink_rel_logits", [&](sycl::handler& h) {
        h.depends_on(deps);
        const uint64_t total  = uint64_t(n_heads) * T * n_kv;
        const uint64_t global = ((total + kWG - 1) / kWG) * kWG;
        h.parallel_for(sycl::nd_range<1>(global, kWG), [=](sycl::nd_item<1> it) {
            const uint64_t idx = it.get_global_id(0);
            if (idx >= total) return;
            const uint32_t j = uint32_t(idx % n_kv);
            const uint32_t i = uint32_t((idx / n_kv) % T);
            const uint32_t hh = uint32_t(idx / (uint64_t(n_kv) * T));
            const int64_t dist = int64_t(q_pos[i]) - int64_t(k_pos[j]);
            if (dist < 0 || dist >= int64_t(rel_extent)) { bias[idx] = 0.f; return; }
            const float* rs = relative_states + (uint64_t(i) * n_heads + hh) * d_rel;
            float acc = 0.f;
            for (uint32_t d = 0; d < d_rel; ++d)
                acc += rs[d] * proj[uint64_t(d) * rel_extent + uint64_t(dist)];
            bias[idx] = acc;
        });
    });
}

// MoE ROUTER — `InklingTopkRouter.forward` (modeling_inkling.py:356).
//
// THE NOVELTY, AND IT IS NOT A DETAIL: Inkling's TWO SHARED EXPERTS ARE
// NORMALISED JOINTLY WITH THE ROUTED ONES.  Every other MoE in this engine
// softmaxes the top-k routed scores and then adds a shared-expert term with its
// own fixed weight.  Here the router emits `n_routed + n_shared` logits, the
// chosen k routed logits and BOTH shared logits go into ONE log-softmax, and
// the shared experts' weights (`shared_gammas`) therefore DEPEND on which
// routed experts won.  Wiring the shared path as a separate additive term would
// be silently wrong.
//
// The reference, step by step (and this kernel does exactly these):
//   scores            = sigmoid(router_logits)                        (:361)
//   scores_for_choice = scores[:n_routed] + e_score_correction_bias   (:363)
//   topk_indices      = topk(scores_for_choice, k)                    (:364)
//   topk_logits       = cat(RAW routed logits at topk_indices,
//                           RAW shared logits)                        (:368)
//   topk_log_probs    = logsigmoid(topk_logits)                       (:369)
//   weights           = exp(log_probs - logsumexp(log_probs))         (:370)
//   weights          *= route_scale * global_scale                    (:372)
//
// NOTE the two places the reference uses RAW logits, not the sigmoid scores:
// selection uses `scores + bias`, but the WEIGHTS come from `logsigmoid` of the
// raw logits.  Conflating them is the obvious way to get plausible-but-wrong
// routing weights.
//
// SELECTION ORDER.  `torch.topk(..., sorted=False)` leaves slot order
// unspecified, and the downstream expert sum is order-independent, so this
// kernel emits the k winners in DESCENDING score with the smaller index first
// on a tie — a strict total order, hence deterministic.  The gate compares the
// (expert -> weight) MAPPING rather than slot positions, because that is what
// the model actually consumes.
//
//   logits  : [T, n_routed + n_shared] fp32 (router output, RAW)
//   bias    : [n_routed] fp32 (`e_score_correction_bias`)
//   out_idx : [T, k] int32      out_w : [T, k] fp32
//   out_shared : [T, n_shared] fp32 (`shared_gammas`)
sycl::event ink_moe_router(sycl::queue& q, const float* logits, const float* bias,
                           int32_t* out_idx, float* out_w, float* out_shared,
                           uint32_t T, uint32_t n_routed, uint32_t n_shared,
                           uint32_t k, float route_scale, float global_scale,
                           const std::vector<sycl::event>& deps) {
    return ie::ps(q, "ink_moe_router", [&](sycl::handler& h) {
        h.depends_on(deps);
        const uint32_t global = ((T + kWG - 1) / kWG) * kWG;
        h.parallel_for(sycl::nd_range<1>(global, kWG), [=](sycl::nd_item<1> it) {
            const uint32_t t = uint32_t(it.get_global_id(0));
            if (t >= T) return;
            const uint32_t n_tot = n_routed + n_shared;
            const float* row = logits + uint64_t(t) * n_tot;
            // --- selection over sigmoid(routed) + bias, descending, ties by
            //     smaller index.  k is 6 on this model, so a k-pass scan beats
            //     any sort and keeps the order exact.
            constexpr uint32_t kMaxK = 16;
            uint32_t sel[kMaxK];
            float    prev_s = 0.f;
            uint32_t prev_i = 0;
            for (uint32_t r = 0; r < k && r < kMaxK; ++r) {
                float best_s = -1e30f;
                uint32_t best_i = 0;
                bool found = false;
                for (uint32_t e = 0; e < n_routed; ++e) {
                    const float sc = 1.f / (1.f + sycl::exp(-row[e])) + bias[e];
                    // strictly after the previous pick in (score desc, index asc)
                    const bool after = r == 0 ? true
                                    : (sc < prev_s || (sc == prev_s && e > prev_i));
                    if (!after) continue;
                    if (!found || sc > best_s || (sc == best_s && e < best_i)) {
                        best_s = sc; best_i = e; found = true;
                    }
                }
                sel[r] = best_i;
                prev_s = best_s; prev_i = best_i;
            }
            // --- joint log-softmax over the k routed winners AND the shared ---
            float lp[kMaxK + 4];
            const uint32_t n_j = k + n_shared;
            for (uint32_t r = 0; r < k; ++r) {
                const float x = row[sel[r]];
                lp[r] = -sycl::log(1.f + sycl::exp(-x));          // logsigmoid
            }
            for (uint32_t sIdx = 0; sIdx < n_shared; ++sIdx) {
                const float x = row[n_routed + sIdx];
                lp[k + sIdx] = -sycl::log(1.f + sycl::exp(-x));
            }
            float mx = lp[0];
            for (uint32_t r = 1; r < n_j; ++r) mx = sycl::fmax(mx, lp[r]);
            float sum = 0.f;
            for (uint32_t r = 0; r < n_j; ++r) sum += sycl::exp(lp[r] - mx);
            const float lse = mx + sycl::log(sum);
            const float scale = route_scale * global_scale;
            for (uint32_t r = 0; r < k; ++r) {
                out_idx[uint64_t(t) * k + r] = int32_t(sel[r]);
                out_w[uint64_t(t) * k + r]   = sycl::exp(lp[r] - lse) * scale;
            }
            for (uint32_t sIdx = 0; sIdx < n_shared; ++sIdx)
                out_shared[uint64_t(t) * n_shared + sIdx] =
                    sycl::exp(lp[k + sIdx] - lse) * scale;
        });
    });
}

// LENGTH-DEPENDENT ATTENTION TEMPERATURE — `InklingAttention.forward`
// (modeling_inkling.py:254-261).
//
//     tau[i] = 1 + alpha * log( clamp( (q_pos[i]+1) / n_floor, min=1 ) )
//
// THREE THINGS THAT ARE EASY TO GET WRONG, all of them silent:
// 1. IT RUNS ONLY ON FULL-ATTENTION LAYERS.  The reference guards with
//    `if not self.is_sliding` (:254) — on this model that is 7 layers of 42
//    (5, 11, 17, 23, 29, 35, 41; see InklingConfig::is_sliding, verified against
//    the real `attn_rel_proj` extents).  Applying it everywhere changes the
//    sliding layers' scores.
// 2. IT SCALES THE QUERY *AND* THE POSITION BIAS (:260-261), not just the
//    query.  Scaling only q leaves the relative bias at the wrong temperature.
// 3. THE CLAMP IS INSIDE THE LOG (`.clamp(min=1.0)` on the RATIO), so tau is
//    exactly 1 for every position below `n_floor` — 128000 on this model, i.e.
//    the whole of any ordinary context.  A clamp applied to the log instead
//    would give the same answer here and diverge past the floor, which is
//    precisely where it matters and where nobody would be testing.
//
// Computed in fp32 as the reference does (:253 "original impl applies log
// scaling in f32").  `tau` is [T]; the caller multiplies q[T,H,D] and
// bias[H,T,KV] by tau[i] along their token axis.
sycl::event ink_log_scaling_tau(sycl::queue& q, const int32_t* q_pos, float* tau,
                                uint32_t T, float alpha, float n_floor,
                                const std::vector<sycl::event>& deps) {
    return ie::ps(q, "ink_log_scaling_tau", [&](sycl::handler& h) {
        h.depends_on(deps);
        const uint32_t global = ((T + kWG - 1) / kWG) * kWG;
        h.parallel_for(sycl::nd_range<1>(global, kWG), [=](sycl::nd_item<1> it) {
            const uint32_t i = uint32_t(it.get_global_id(0));
            if (i >= T) return;
            const float ratio = (float(q_pos[i]) + 1.f) / n_floor;
            tau[i] = 1.f + alpha * sycl::log(sycl::fmax(ratio, 1.f));
        });
    });
}

// ATTENTION — `eager_attention_forward` (modeling_inkling.py:157) as Inkling
// calls it, i.e. WITH the relative position bias:
//
//     w = (Q @ K^T) * scaling + position_bias + attention_mask
//     w = softmax(w, dim=-1, dtype=float32)
//     out = w @ V
//
// GQA: `repeat_kv` (:145) maps KV head g to query heads [g*n_rep, (g+1)*n_rep),
// so query head h reads KV head h / n_rep.  That is an INDEX MAP, not a copy —
// materialising the repeat, which the reference does, would be pure traffic
// here.
//
// ONLINE SOFTMAX in fp32, one sub-group per (query token, head).  The reference
// softmaxes in float32 explicitly (:177) even when the tensors are bf16, so
// fp32 here is the reference behaviour, not an upgrade.  Masked entries are
// skipped BEFORE the dot product, exactly as `ds4_attention` does — the sliding
// layers mask ~everything outside a 512-wide band, and paying for those dots
// would dominate.
//
//   q    : [T, n_heads, head_dim]        k, v : [n_kv, n_kv_heads, head_dim]
//   bias : [n_heads, T, n_kv] or null    mask : [T, n_kv] or null (additive)
//   out  : [T, n_heads, head_dim]
sycl::event ink_attention(sycl::queue& q_, const float* q, const float* k, const float* v,
                          const float* bias, const float* mask, float* out,
                          uint32_t T, uint32_t n_heads, uint32_t n_kv_heads,
                          uint32_t head_dim, uint32_t n_kv, float scaling,
                          const std::vector<sycl::event>& deps) {
    constexpr int kSG = 16;
    return ie::ps(q_, "ink_attention", [&](sycl::handler& h) {
        h.depends_on(deps);
        const uint32_t dpl = (head_dim + kSG - 1) / kSG;
        const uint32_t n_rep = n_heads / n_kv_heads;
        h.parallel_for(sycl::nd_range<2>({T, size_t(n_heads) * kSG}, {1, kSG}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(kSG)]] {
            const uint32_t t    = uint32_t(it.get_group(0));
            const uint32_t hd   = uint32_t(it.get_group(1));
            const uint32_t lane = uint32_t(it.get_local_id(1));
            auto sg = it.get_sub_group();
            const uint32_t kvh = hd / n_rep;              // repeat_kv, as a map

            float qv[64];
            for (uint32_t d = 0; d < dpl; ++d) {
                const uint32_t dim = lane * dpl + d;
                qv[d] = dim < head_dim
                          ? q[(uint64_t(t) * n_heads + hd) * head_dim + dim] : 0.f;
            }
            float m = -std::numeric_limits<float>::infinity(), l = 0.f;
            float acc[64];
            for (uint32_t d = 0; d < dpl; ++d) acc[d] = 0.f;

            for (uint32_t j = 0; j < n_kv; ++j) {
                float add = 0.f;
                if (mask) {
                    const float mv = mask[uint64_t(t) * n_kv + j];
                    if (!(mv > -1e30f)) continue;         // masked: contributes 0
                    add += mv;
                }
                if (bias) add += bias[(uint64_t(hd) * T + t) * n_kv + j];
                const float* krow = k + (uint64_t(j) * n_kv_heads + kvh) * head_dim;
                float part = 0.f;
                for (uint32_t d = 0; d < dpl; ++d) {
                    const uint32_t dim = lane * dpl + d;
                    if (dim < head_dim) part += qv[d] * krow[dim];
                }
                const float sc =
                    sycl::reduce_over_group(sg, part, sycl::plus<float>()) * scaling + add;
                const float m_new = sycl::fmax(m, sc);
                const float alpha = (m == -std::numeric_limits<float>::infinity())
                                      ? 0.f : sycl::exp(m - m_new);
                const float e = sycl::exp(sc - m_new);
                const float* vrow = v + (uint64_t(j) * n_kv_heads + kvh) * head_dim;
                for (uint32_t d = 0; d < dpl; ++d) {
                    const uint32_t dim = lane * dpl + d;
                    acc[d] = acc[d] * alpha + (dim < head_dim ? e * vrow[dim] : 0.f);
                }
                l = l * alpha + e;
                m = m_new;
            }
            const float inv = (l > 0.f) ? (1.f / l) : 0.f;
            for (uint32_t d = 0; d < dpl; ++d) {
                const uint32_t dim = lane * dpl + d;
                if (dim < head_dim)
                    out[(uint64_t(t) * n_heads + hd) * head_dim + dim] = acc[d] * inv;
            }
        });
    });
}

// ---------------------------------------------------------------------------
// Assembly helpers (P2).  Small, generic, fp32 — enough to compose a decoder
// layer against the reference before the real loader and the engine's tuned
// GEMMs are wired in.  They are NOT the performance path; `ink_layer_dense`
// exists to gate the ORDER of operations, which is where port bugs live once
// the individual operators are correct.
// ---------------------------------------------------------------------------

// y[T,N] = x[T,K] * w[N,K]^T  (row-major, weights stored output-major as GGUF
// and torch both do).
sycl::event ink_matmul_nt(sycl::queue& q, const float* x, const float* w, float* y,
                          uint32_t T, uint32_t K, uint32_t N,
                          const std::vector<sycl::event>& deps) {
    return ie::ps(q, "ink_matmul_nt", [&](sycl::handler& h) {
        h.depends_on(deps);
        const uint64_t total  = uint64_t(T) * N;
        const uint64_t global = ((total + kWG - 1) / kWG) * kWG;
        h.parallel_for(sycl::nd_range<1>(global, kWG), [=](sycl::nd_item<1> it) {
            const uint64_t i = it.get_global_id(0);
            if (i >= total) return;
            const uint32_t t = uint32_t(i / N), n = uint32_t(i % N);
            float acc = 0.f;
            for (uint32_t k = 0; k < K; ++k)
                acc += x[uint64_t(t) * K + k] * w[uint64_t(n) * K + k];
            y[i] = acc;
        });
    });
}

// RMSNorm over the LAST axis of [rows, dim] — `InklingRMSNorm` (:99):
//   y = x * rsqrt(mean(x^2) + eps) * weight,  accumulated in fp32.
// Used both for the layer norms (dim = hidden) and the PER-HEAD q/k norms
// (rows = T*n_heads, dim = head_dim) — the reference applies the same module
// to a [T, heads, head_dim] view, which is exactly this with rows folded.
sycl::event ink_rms_norm(sycl::queue& q, const float* x, const float* w, float* y,
                         uint32_t rows, uint32_t dim, float eps,
                         const std::vector<sycl::event>& deps) {
    return ie::ps(q, "ink_rms_norm", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(size_t(rows) * kSGNorm, kSGNorm),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSGNorm)]] {
            const uint32_t r = uint32_t(it.get_group(0));
            const uint32_t lane = uint32_t(it.get_local_id(0));
            auto sg = it.get_sub_group();
            const float* xr = x + uint64_t(r) * dim;
            float acc = 0.f;
            for (uint32_t d = lane; d < dim; d += kSGNorm) acc += xr[d] * xr[d];
            const float ss = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
            const float inv = sycl::rsqrt(ss / float(dim) + eps);
            for (uint32_t d = lane; d < dim; d += kSGNorm)
                y[uint64_t(r) * dim + d] = xr[d] * inv * w[d];
        });
    });
}

// SwiGLU: y = silu(gate) * up, elementwise.  silu(x) = x * sigmoid(x).
sycl::event ink_swiglu(sycl::queue& q, const float* gate, const float* up, float* y,
                       uint64_t n, const std::vector<sycl::event>& deps) {
    return ie::ps(q, "ink_swiglu", [&](sycl::handler& h) {
        h.depends_on(deps);
        const uint64_t global = ((n + kWG - 1) / kWG) * kWG;
        h.parallel_for(sycl::nd_range<1>(global, kWG), [=](sycl::nd_item<1> it) {
            const uint64_t i = it.get_global_id(0);
            if (i >= n) return;
            const float g = gate[i];
            y[i] = (g / (1.f + sycl::exp(-g))) * up[i];
        });
    });
}

// Elementwise add, y = a + b.
sycl::event ink_add(sycl::queue& q, const float* a, const float* b, float* y,
                    uint64_t n, const std::vector<sycl::event>& deps) {
    return ie::ps(q, "ink_add", [&](sycl::handler& h) {
        h.depends_on(deps);
        const uint64_t global = ((n + kWG - 1) / kWG) * kWG;
        h.parallel_for(sycl::nd_range<1>(global, kWG), [=](sycl::nd_item<1> it) {
            const uint64_t i = it.get_global_id(0);
            if (i < n) y[i] = a[i] + b[i];
        });
    });
}

}  // namespace ie
