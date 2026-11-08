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

// 8-bit HC mixing weights (2026-11-08): the GGUF's Q8_0 blocks of hc_*_down / hc_*_up as they lie on disk, split SoA the
// way qwen4exp's Q8W holds dense weights -- q[n*K + k] int8 and one F16 scale (raw bits) per 32 along each row,
// d[n*(K/32) + b]. down: r rows of K = D; up: D rows of K = r. qwen4_hc_mix_v2 / _xn read them at T <= 4 instead of the F16
// copy: about half the bytes, and the file's own values (the F16 copy rounds every q*d to 11 bits). Needs D % 16 == 0,
// r % 64 == 0, H % 4 == 0; otherwise (or T > 4) the F16 path runs.
struct Qwen4HcQ8 {
    const int8_t*   down_q = nullptr;   // [r][D]
    const uint16_t* down_d = nullptr;   // [r][D/32] F16 bits
    const int8_t*   up_q   = nullptr;   // [D][r], or lane-interleaved (up_lanes)
    const uint16_t* up_d   = nullptr;   // [D][r/32] F16 bits, or lane-interleaved (up_lanes)
    bool up_lanes = false;              // up_q / up_d in qwen4_hc_q8_interleave_up's layout (the fast up kernel)
    bool ok() const { return down_q && down_d && up_q && up_d; }
};

// The up matrix's planes [D][r] int8 / [D][r/32] F16 bits repacked lane-interleaved (host side, at load): rows in groups of
// 16, a group's 16-byte chunks lane-major -- qi[((g*(r/16) + c)*16 + l)*16 + u] = row (16g + l)'s byte 16c + u,
// di[(g*(r/32) + b)*16 + l] = row (16g + l)'s block-b scale. A lane then owns one row and a subgroup's 16 lanes read 256
// contiguous bytes a step. Needs D % 16 == 0, r % 32 == 0.
void qwen4_hc_q8_interleave_up(const int8_t* q, const uint16_t* d, uint32_t D, uint32_t r, int8_t* qi, uint16_t* di);

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
                            const std::vector<sycl::event>& deps = {},
                            const Qwen4HcQ8* w8 = nullptr);   // the 8-bit down/up at T <= 4 (see Qwen4HcQ8)

sycl::event qwen4_hc_combine(sycl::queue& q,
                             float* res,                  // F32 [T, hc, H], in-place
                             const sycl::half* block_out, // F16 [T, H]
                             const float* inj_logits,     // F32 [T, hc]
                             uint32_t T, uint32_t H, uint32_t hc,
                             const std::vector<sycl::event>& deps = {});

// COMBINE + the following mix's norm in ONE launch (2026-11-08): the combine above, then xn_ws[t,s,:] = rms-norm(res[t,s,:])
// * gamma[s,:] exactly as qwen4_hc_mix_v2's first stage computes it (same split-H reduction). Pair with qwen4_hc_mix_v2_xn,
// which is qwen4_hc_mix_v2 minus that first stage (it reads xn_ws instead of x + gamma). Both bit-identical to the
// separate launches (ie-qwen4-hc-test).
sycl::event qwen4_hc_combine_norm(sycl::queue& q,
                                  float* res, const sycl::half* block_out,
                                  const float* inj_logits,
                                  const float* gamma,          // the FOLLOWING mix's norm weights [hc*H]
                                  float* xn_ws,                // F32 out [T, hc*H]
                                  uint32_t T, uint32_t H, uint32_t hc, float eps,
                                  const std::vector<sycl::event>& deps = {});
sycl::event qwen4_hc_mix_v2_xn(sycl::queue& q,
                               const float* xn_ws,            // F32 [T, hc*H], pre-normed
                               const sycl::half* w_down, const sycl::half* w_up,
                               const float* w_inject,
                               float* lo_ws, sycl::half* mixed, float* inj_logits,
                               uint32_t T, uint32_t H, uint32_t hc, uint32_t r,
                               const std::vector<sycl::event>& deps = {},
                               const Qwen4HcQ8* w8 = nullptr);

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
