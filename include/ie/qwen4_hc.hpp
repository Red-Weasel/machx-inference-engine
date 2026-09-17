// include/ie/qwen4_hc.hpp — Qwen3.8-Flash-Next (qwen4exp) hyper-connections
// residual-stream ops.  Spec: docs/qwen4/11_hyper_connections.md.
//
// LAYOUT CONTRACT (all buffers device USM):
//  * Wide residual state: FP32 [T, hc, H] contiguous, t-major then stream then
//    channel — element (t, s, h) lives at index ((uint64_t(t) * hc) + s) * H + h.
//    D = hc * H is the flattened per-token width (qwen4exp: 4 * 2560 = 10240).
//  * gamma (grouped-norm weight): F32 [D].  PRE-FOLDED (1 + w) by the GGUF
//    converter — applied as a plain multiply, NOT (1 + gamma).
//  * w_down: F16 [r][D] row-major (row per output — the GGUF-native layout of
//    hc_*_down {ne0=10240, ne1=320}; loader dequants Q8_0 rows straight in,
//    no transpose).
//  * w_up:   F16 [D][r] row-major (GGUF hc_*_up {320, 10240}, same convention).
//  * w_inject: F32 [hc][D] row-major (GGUF hc_*_inject {10240, 4}).
//  * mixed / block_out: F16 [T, H] row-major.  inj_logits: F32 [T, hc].
//
// MATH (per token t; all accumulation fp32, norms fp32; eps from config, 1e-6):
//   MIX (read path, 2 kernel launches):
//     xn[j]  = x[j] * rsqrt(mean_h(x[s*H+..]^2) + eps) * gamma[j]   (RMS per
//              2560-stream group s = j / H; gamma spans all D)
//     lo[i]  = silu( (Σ_j w_down[i][j]·xn[j]) / hc )                i < r
//     gate[d]= sigmoid( Σ_k w_up[d][k]·lo[k] )                      d < D
//     mixed[h]     = (1/hc) Σ_s xn[s*H+h]·gate[s*H+h]
//     inj_logits[s]= Σ_j w_inject[s][j]·xn[j]      (from the PRE-block xn)
//   The /hc (= /4) divisors sit BETWEEN the GEMV and the nonlinearity — kept
//   explicit (never folded into the weights) for bit-parity vs the reference.
//   COMBINE (write path, 1 launch; streams NEVER re-mixed):
//     w[s]        = 2·sigmoid(inj_logits[s] / hc)
//     res[s,h]   += w[s] · block_out[h]        (added to the RAW wide state)
//   FINAL MERGE = qwen4_hc_mix with w_inject == nullptr (mixed only) — this IS
//   the model's final norm before lm_head.

#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>
#include <vector>

namespace ie {

// wide[t, s, :] = emb[t, :] for every stream s — the hc-way expand init
// (hidden_states.repeat(1,1,hc)).  emb is the F16 [T, H] embedding row block.
sycl::event qwen4_hc_expand_streams(sycl::queue& q,
                                    const sycl::half* emb,   // F16 [T, H]
                                    float* wide,             // F32 [T, hc, H]
                                    uint32_t T, uint32_t H, uint32_t hc,
                                    const std::vector<sycl::event>& deps = {});

// MIX (and, with w_inject == nullptr, the FINAL MERGE).  Two kernel launches:
//   1. grouped RMS norm + down GEMV + silu   -> xn_ws [T, D], lo_ws [T, r]
//   2. up GEMV + sigmoid gate + mean collapse + inject GEMV
//      -> mixed [T, H] F16, inj_logits [T, hc] F32 (skipped when w_inject
//         == nullptr; inj_logits may then be nullptr).
// xn_ws / lo_ws are caller-provided F32 scratch ([T*hc*H] and [T*r] elements).
// Returns the stage-2 event (stage 1 is chained internally).
sycl::event qwen4_hc_mix(sycl::queue& q,
                         const float* x,            // F32 [T, hc, H] wide state
                         const float* gamma,        // F32 [hc*H], pre-folded (1+w)
                         const sycl::half* w_down,  // F16 [r][hc*H]
                         const sycl::half* w_up,    // F16 [hc*H][r]
                         const float* w_inject,     // F32 [hc][hc*H], or nullptr
                         float* xn_ws,              // F32 scratch [T, hc*H]
                         float* lo_ws,              // F32 scratch [T, r]
                         sycl::half* mixed,         // F16 out [T, H]
                         float* inj_logits,         // F32 out [T, hc], or nullptr
                         uint32_t T, uint32_t H, uint32_t hc, uint32_t r,
                         float eps,
                         const std::vector<sycl::event>& deps = {});

// COMBINE: res[t,s,:] += 2·sigmoid(inj_logits[t,s]/hc) · block_out[t,:].
// In-place on the RAW pre-norm wide state; one launch.
// v2 of MIX: identical math, decomposed into device-wide launches so any T
// (decode T=1 included) fills the GPU. v1 runs ONE work-group per token —
// measured 268 ms of a 300 ms decode step (88% of GPU time) on the real
// model; v2 replaces it on the model path. v1 stays as the fp64-gate anchor.
// Same signature/layout contract as qwen4_hc_mix.
sycl::event qwen4_hc_mix_v2(sycl::queue& q,
                            const float* x, const float* gamma,
                            const sycl::half* w_down, const sycl::half* w_up,
                            const float* w_inject,
                            float* xn_ws, float* lo_ws,
                            sycl::half* mixed, float* inj_logits,
                            uint32_t T, uint32_t H, uint32_t hc, uint32_t r,
                            float eps,
                            const std::vector<sycl::event>& deps = {});

sycl::event qwen4_hc_combine(sycl::queue& q,
                             float* res,                  // F32 [T, hc, H], in-place
                             const sycl::half* block_out, // F16 [T, H]
                             const float* inj_logits,     // F32 [T, hc]
                             uint32_t T, uint32_t H, uint32_t hc,
                             const std::vector<sycl::event>& deps = {});

// PREFILL MIX (T > 16): qwen4_hc_mix_v2's math with the two low-rank
// projections routed through gemm_fp16 over [K, N]-TRANSPOSED weight copies
// (w_down_t F16 [hc*H][r], w_up_t F16 [r][hc*H] — built once at load;
// qwen4exp keeps the [N, K] originals for v2). v2's subgroup-GEMV
// re-streams both weight mats once PER TOKEN (40% of prefill GPU busy at
// T=1024, 7.9 ms/call). Down/up run W16A16 (xn and lo rounded to F16 for
// the GEMM A operands — the dense-prefill scratch-dequant convention);
// inject keeps v2's fp32 dot verbatim. Decode/verify (T <= 16) stay on v2:
// prefill numerics differ in fp order only, are DETERMINISTIC (fixed XMX
// tiling; the oneDNN nt gemm was bisected bit-nondeterministic 2026-08-27
// and evicted), and are gated by chunked PPL + run2 bit-eq. Extra scratch:
// xn16_ws F16 [T, hc*H], lo16_ws F16 [T, r], gemm_ws F32 (>= T x hc*H).
sycl::event qwen4_hc_mix_prefill(sycl::queue& q,
                            const float* x, const float* gamma,
                            const sycl::half* w_down_t, const sycl::half* w_up_t,
                            const float* w_inject,
                            float* xn_ws, sycl::half* xn16_ws,
                            sycl::half* lo16_ws, float* gemm_ws,
                            sycl::half* mixed, float* inj_logits,
                            uint32_t T, uint32_t H, uint32_t hc, uint32_t r,
                            float eps,
                            const std::vector<sycl::event>& deps = {});

}  // namespace ie
