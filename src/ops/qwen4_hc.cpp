// src/ops/qwen4_hc.cpp — Qwen3.8-Flash-Next (qwen4exp) hyper-connections
// residual-stream kernels.  Spec: docs/qwen4/11_hyper_connections.md.
//
// Layout contract lives in include/ie/qwen4_hc.hpp.  Correctness gate:
// tools/qwen4_hc_test.cpp (GPU vs CPU fp64 reference, no model file).
//
// MIX is 2 launches per site (the decode budget from the DSV4 launch-overhead
// lesson): stage 1 fuses the grouped RMS norm with the down GEMV + silu;
// stage 2 fuses the up GEMV + sigmoid gate + mean collapse + inject GEMV.
// COMBINE is 1 flat launch.  Both use one workgroup per token in the mix
// stages — fine for prefill (T workgroups) and launch-bound anyway at T=1.
//
// Numerics: all accumulation fp32; precise sycl::exp / sycl::sqrt (NOT the
// native:: variants) so the parity gate's tolerances hold with margin.  The
// /hc divisors (== /4 for qwen4exp) sit between GEMV and nonlinearity,
// exactly as the reference computes them — never folded into the weights.

#include "ie/kernel_profiler.hpp"
#include "ie/ops.hpp"
#include "ie/qwen4_hc.hpp"

#include <sycl/sycl.hpp>
#include <cstdlib>

namespace ie {

namespace {

constexpr uint32_t kWG = 256;  // workgroup size for the per-token mix stages
constexpr uint32_t kSG = 16;   // subgroup size

inline float silu_f(float a) { return a / (1.f + sycl::exp(-a)); }
inline float sigmoid_f(float a) { return 1.f / (1.f + sycl::exp(-a)); }

}  // namespace

sycl::event qwen4_hc_expand_streams(sycl::queue& q,
                                    const sycl::half* emb, float* wide,
                                    uint32_t T, uint32_t H, uint32_t hc,
                                    const std::vector<sycl::event>& deps) {
    const uint64_t n = uint64_t(T) * hc * H;
    return ie::ps(q, "q4hc_expand", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>((n + kWG - 1) / kWG * kWG, kWG),
                       [=](sycl::nd_item<1> it) {
            const uint64_t i = it.get_global_id(0);
            if (i >= n) return;
            const uint64_t t = i / (uint64_t(hc) * H);
            const uint64_t d = i % H;
            wide[i] = float(emb[t * H + d]);
        });
    });
}

// ---------------------------------------------------------------------------
// MIX stage 1: grouped RMS norm (per 2560-stream group, gamma over all D) +
// down GEMV (D -> r) + /hc + silu.  One WG per token.
//   xn[t, j] = x[t, j] * rsqrt(mean_h x[t, s*H+..]^2 + eps) * gamma[j]
//   lo[t, i] = silu( (Sum_j w_down[i][j] * xn[t, j]) / hc )
// ---------------------------------------------------------------------------
static sycl::event qwen4_hc_mix_s1(sycl::queue& q,
                                   const float* x, const float* gamma,
                                   const sycl::half* w_down,
                                   float* xn, float* lo,
                                   uint32_t T, uint32_t H, uint32_t hc,
                                   uint32_t r, float eps,
                                   const std::vector<sycl::event>& deps) {
    const uint32_t D = hc * H;
    return ie::ps(q, "q4hc_mix_s1", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(uint64_t(T) * kWG, kWG),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
            const uint32_t t   = uint32_t(it.get_group(0));
            const uint32_t lid = uint32_t(it.get_local_id(0));
            auto grp = it.get_group();
            const float* xt  = x  + uint64_t(t) * D;
            float*       xnt = xn + uint64_t(t) * D;

            // Per-stream RMS norm; group collectives keep the WG converged.
            for (uint32_t s = 0; s < hc; ++s) {
                float ss = 0.f;
                for (uint32_t i = lid; i < H; i += kWG) {
                    const float v = xt[uint64_t(s) * H + i];
                    ss = sycl::fma(v, v, ss);
                }
                ss = sycl::reduce_over_group(grp, ss, sycl::plus<float>());
                const float inv = 1.f / sycl::sqrt(ss / float(H) + eps);
                for (uint32_t i = lid; i < H; i += kWG) {
                    const uint32_t j = s * H + i;
                    xnt[j] = xt[j] * inv * gamma[j];
                }
            }
            sycl::group_barrier(grp);

            // Down GEMV: one subgroup per output i, lanes split the D dot.
            auto sg = it.get_sub_group();
            const uint32_t sgid = uint32_t(sg.get_group_linear_id());
            const uint32_t lane = uint32_t(sg.get_local_linear_id());
            const uint32_t nsg  = kWG / kSG;
            for (uint32_t i = sgid; i < r; i += nsg) {
                const sycl::half* wr = w_down + uint64_t(i) * D;
                float acc = 0.f;
                for (uint32_t j = lane; j < D; j += kSG)
                    acc = sycl::fma(float(wr[j]), xnt[j], acc);
                acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
                if (lane == 0)
                    lo[uint64_t(t) * r + i] = silu_f(acc / float(hc));
            }
        });
    });
}

// ---------------------------------------------------------------------------
// MIX stage 2: up GEMV (r -> D) + sigmoid gate + mean collapse (+ inject
// GEMV D -> hc, from the PRE-block xn).  One WG per token; lo staged in SLM.
//   gate[d]   = sigmoid( Sum_k w_up[d][k] * lo[k] )
//   mixed[h]  = (1/hc) Sum_s xn[s*H+h] * gate[s*H+h]
//   inj[s]    = Sum_j w_inject[s][j] * xn[j]        (when w_inject != nullptr)
// ---------------------------------------------------------------------------
static sycl::event qwen4_hc_mix_s2(sycl::queue& q,
                                   const float* xn, const float* lo,
                                   const sycl::half* w_up,
                                   const float* w_inject,
                                   sycl::half* mixed, float* inj_logits,
                                   uint32_t T, uint32_t H, uint32_t hc,
                                   uint32_t r,
                                   const std::vector<sycl::event>& deps) {
    const uint32_t D = hc * H;
    const bool has_inject = (w_inject != nullptr);
    return ie::ps(q, "q4hc_mix_s2", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<float, 1> lo_s(sycl::range<1>(r), h);
        h.parallel_for(sycl::nd_range<1>(uint64_t(T) * kWG, kWG),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
            const uint32_t t   = uint32_t(it.get_group(0));
            const uint32_t lid = uint32_t(it.get_local_id(0));
            auto grp = it.get_group();
            const float* xnt = xn + uint64_t(t) * D;

            for (uint32_t k = lid; k < r; k += kWG)
                lo_s[k] = lo[uint64_t(t) * r + k];
            sycl::group_barrier(grp);

            // Inject GEMV (whole-WG reduction per stream; collectives converge).
            if (has_inject) {
                for (uint32_t s = 0; s < hc; ++s) {
                    const float* wi = w_inject + uint64_t(s) * D;
                    float acc = 0.f;
                    for (uint32_t j = lid; j < D; j += kWG)
                        acc = sycl::fma(wi[j], xnt[j], acc);
                    acc = sycl::reduce_over_group(grp, acc, sycl::plus<float>());
                    if (lid == 0) inj_logits[uint64_t(t) * hc + s] = acc;
                }
            }

            // Up GEMV + gate + mean collapse: one lane per output channel h.
            for (uint32_t ch = lid; ch < H; ch += kWG) {
                float m = 0.f;
                for (uint32_t s = 0; s < hc; ++s) {
                    const uint32_t d = s * H + ch;
                    const sycl::half* wr = w_up + uint64_t(d) * r;
                    float g = 0.f;
                    for (uint32_t k = 0; k < r; ++k)
                        g = sycl::fma(float(wr[k]), lo_s[k], g);
                    m = sycl::fma(xnt[d], sigmoid_f(g), m);
                }
                mixed[uint64_t(t) * H + ch] = sycl::half(m / float(hc));
            }
        });
    });
}

sycl::event qwen4_hc_mix(sycl::queue& q,
                         const float* x, const float* gamma,
                         const sycl::half* w_down, const sycl::half* w_up,
                         const float* w_inject,
                         float* xn_ws, float* lo_ws,
                         sycl::half* mixed, float* inj_logits,
                         uint32_t T, uint32_t H, uint32_t hc, uint32_t r,
                         float eps,
                         const std::vector<sycl::event>& deps) {
    auto e1 = qwen4_hc_mix_s1(q, x, gamma, w_down, xn_ws, lo_ws,
                              T, H, hc, r, eps, deps);
    return qwen4_hc_mix_s2(q, xn_ws, lo_ws, w_up, w_inject, mixed, inj_logits,
                           T, H, hc, r, {e1});
}

// ---------------------------------------------------------------------------
// COMBINE: res[t,s,:] += 2*sigmoid(inj[t,s]/hc) * block_out[t,:], in-place on
// the RAW pre-norm wide state.  Streams are never re-mixed.  One flat launch.
// ---------------------------------------------------------------------------
sycl::event qwen4_hc_combine(sycl::queue& q,
                             float* res, const sycl::half* block_out,
                             const float* inj_logits,
                             uint32_t T, uint32_t H, uint32_t hc,
                             const std::vector<sycl::event>& deps) {
    const uint64_t n = uint64_t(T) * hc * H;
    const float inv_hc = 1.f / float(hc);
    return ie::ps(q, "q4hc_combine", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>((n + kWG - 1) / kWG * kWG, kWG),
                       [=](sycl::nd_item<1> it) {
            const uint64_t i = it.get_global_id(0);
            if (i >= n) return;
            const uint64_t ch = i % H;
            const uint64_t s  = (i / H) % hc;
            const uint64_t t  = i / (uint64_t(hc) * H);
            const float w = 2.f * sigmoid_f(inj_logits[t * hc + s] * inv_hc);
            res[i] = sycl::fma(w, float(block_out[t * H + ch]), res[i]);
        });
    });
}


// L1 (shared by v2 and the prefill mix): WG of 8 subgroups per (t, s) stream
// row — split-H sumsq reduced in SLM, then a cooperative normalize+gamma
// write. (One subgroup per row left the GPU ~4 WGs at T=1.)
static sycl::event qwen4_hc_norm_l1(sycl::queue& q,
                                    const float* x, const float* gamma,
                                    float* xn_ws,
                                    uint32_t T, uint32_t H, uint32_t hc,
                                    float eps,
                                    const std::vector<sycl::event>& deps) {
    const uint32_t D = hc * H;
    constexpr int SG = 16;
    return ie::ps(q, "q4hc2_norm", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<float, 1> part(8, h);
        h.parallel_for(sycl::nd_range<1>(uint64_t(T) * hc * 128, 128),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint32_t row = uint32_t(it.get_group(0));
            const uint32_t lid = uint32_t(it.get_local_id(0));
            const float* xr = x + uint64_t(row) * H;
            float p = 0.f;
            for (uint32_t i = lid; i < H; i += 128) p += xr[i] * xr[i];
            auto sg = it.get_sub_group();
            const float ps2 = sycl::reduce_over_group(sg, p, sycl::plus<float>());
            if (sg.get_local_id()[0] == 0) part[sg.get_group_id()[0]] = ps2;
            sycl::group_barrier(it.get_group());
            float ss = 0.f;
            for (int j = 0; j < 8; ++j) ss += part[j];
            const float rs = sycl::rsqrt(ss / float(H) + eps);
            const uint32_t s = row % hc;
            float* o = xn_ws + uint64_t(row / hc) * D + uint64_t(s) * H;
            const float* gm = gamma + uint64_t(s) * H;
            for (uint32_t i = lid; i < H; i += 128) o[i] = xr[i] * rs * gm[i];
        });
    });
}

// T<=4 wide variants of L2/L3 (decode + spec-verify shapes). The v2 grids
// collapse at T=1 — L2 runs (r+hc)=324 SIXTEEN-LANE subgroups and L3 runs
// T*H/256 = 10 WGs on a 32-core B70 (measured 52 + 24 us/call, 2 calls/layer
// = 7.3 ms/token). v3 keeps the math but re-grids: L2 gets a 128-lane WG per
// output row (8 sub-group partials, SLM-combined); L3 gets a 64-lane WG per
// (t, hh) with one subgroup per stream. fp32 sums are REASSOCIATED vs v2
// (grouped partials) — same contract as the norm's split-H reduce; PPL-gated.
static sycl::event q4hc2_down_v3(sycl::queue& q, sycl::event e1,
                                 const float* xn_ws, const sycl::half* w_down,
                                 const float* w_inject, float* lo_ws,
                                 float* inj_logits,
                                 uint32_t T, uint32_t D, uint32_t hc,
                                 uint32_t r) {
    constexpr int SG = 16, WGL = 128;         // 8 subgroups per row
    const uint32_t rows2 = r + (w_inject ? hc : 0);
    return ie::ps(q, "q4hc2_down", [&](sycl::handler& h) {
        h.depends_on({e1});
        sycl::local_accessor<float, 1> part(WGL / SG, h);
        h.parallel_for(sycl::nd_range<1>(uint64_t(T) * rows2 * WGL, WGL),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint32_t g2 = uint32_t(it.get_group(0));
            const uint32_t t = g2 / rows2, i = g2 % rows2;
            const uint32_t lid = uint32_t(it.get_local_id(0));
            auto sg = it.get_sub_group();
            const float* xn = xn_ws + uint64_t(t) * D;
            float p = 0.f;
            if (i < r) {
                const auto* wr8 = reinterpret_cast<const sycl::vec<sycl::half, 8>*>(
                    w_down + uint64_t(i) * D);
                for (uint32_t j8 = lid; j8 < D / 8; j8 += WGL) {
                    const sycl::vec<sycl::half, 8> v = wr8[j8];
                    const float* xj = xn + j8 * 8;
                    #pragma unroll
                    for (int u2 = 0; u2 < 8; ++u2) p += float(v[u2]) * xj[u2];
                }
            } else {
                const uint32_t s2 = i - r;
                const auto* wi4 = reinterpret_cast<const sycl::vec<float, 4>*>(
                    w_inject + uint64_t(s2) * D);
                const auto* xn4 = reinterpret_cast<const sycl::vec<float, 4>*>(xn);
                for (uint32_t j4 = lid; j4 < D / 4; j4 += WGL) {
                    const sycl::vec<float, 4> a = wi4[j4], b = xn4[j4];
                    #pragma unroll
                    for (int u2 = 0; u2 < 4; ++u2) p += a[u2] * b[u2];
                }
            }
            const float ps2 = sycl::reduce_over_group(sg, p, sycl::plus<float>());
            if (sg.get_local_id()[0] == 0) part[sg.get_group_id()[0]] = ps2;
            sycl::group_barrier(it.get_group());
            if (lid != 0) return;
            float acc = 0.f;
            for (int j = 0; j < WGL / SG; ++j) acc += part[j];
            if (i < r) {
                acc /= float(hc);
                lo_ws[uint64_t(t) * r + i] = acc / (1.f + sycl::exp(-acc));
            } else {
                inj_logits[uint64_t(t) * hc + (i - r)] = acc;
            }
        });
    });
}

static sycl::event q4hc2_up_v3(sycl::queue& q, sycl::event e2,
                               const float* xn_ws, const sycl::half* w_up,
                               const float* lo_ws, sycl::half* mixed,
                               uint32_t T, uint32_t H, uint32_t hc, uint32_t r) {
    constexpr int SG = 16;
    const uint32_t WGL = hc * SG;             // one subgroup per stream
    return ie::ps(q, "q4hc2_up", [&](sycl::handler& h) {
        h.depends_on({e2});
        sycl::local_accessor<float, 1> sd(hc, h);
        h.parallel_for(sycl::nd_range<1>(uint64_t(T) * H * WGL, WGL),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint32_t g2 = uint32_t(it.get_group(0));
            const uint32_t t = g2 / H, hh = g2 % H;
            const uint32_t lid = uint32_t(it.get_local_id(0));
            auto sg = it.get_sub_group();
            const uint32_t s2 = uint32_t(sg.get_group_id()[0]);
            const uint32_t lane = uint32_t(sg.get_local_id()[0]);
            const uint32_t D = hc * H;
            const auto* wu8 = reinterpret_cast<const sycl::vec<sycl::half, 8>*>(
                w_up + uint64_t(s2 * H + hh) * r);
            const float* lo = lo_ws + uint64_t(t) * r;
            float p = 0.f;
            for (uint32_t k8 = lane; k8 < r / 8; k8 += SG) {
                const sycl::vec<sycl::half, 8> v = wu8[k8];
                #pragma unroll
                for (int u2 = 0; u2 < 8; ++u2) p += float(v[u2]) * lo[k8 * 8 + u2];
            }
            const float d = sycl::reduce_over_group(sg, p, sycl::plus<float>());
            if (lane == 0) sd[s2] = d;
            sycl::group_barrier(it.get_group());
            if (lid != 0) return;
            const float* xn = xn_ws + uint64_t(t) * D;
            float m = 0.f;
            for (uint32_t ss = 0; ss < hc; ++ss) {
                const float gt = 1.f / (1.f + sycl::exp(-sd[ss]));
                m += xn[uint64_t(ss) * H + hh] * gt;
            }
            mixed[uint64_t(t) * H + hh] = sycl::half(m / float(hc));
        });
    });
}

sycl::event qwen4_hc_mix_v2(sycl::queue& q,
                            const float* x, const float* gamma,
                            const sycl::half* w_down, const sycl::half* w_up,
                            const float* w_inject,
                            float* xn_ws, float* lo_ws,
                            sycl::half* mixed, float* inj_logits,
                            uint32_t T, uint32_t H, uint32_t hc, uint32_t r,
                            float eps,
                            const std::vector<sycl::event>& deps) {
    const uint32_t D = hc * H;
    constexpr int SG = 16;
    auto e1 = qwen4_hc_norm_l1(q, x, gamma, xn_ws, T, H, hc, eps, deps);
    static const bool v3_on = [] {
        const char* v = std::getenv("IE_Q4E_HC_V3");
        return !(v && v[0] == '0');
    }();
    if (v3_on && T <= 4 && (D % 8) == 0 && (r % 8) == 0) {
        auto e2 = q4hc2_down_v3(q, e1, xn_ws, w_down, w_inject, lo_ws,
                                inj_logits, T, D, hc, r);
        return q4hc2_up_v3(q, e2, xn_ws, w_up, lo_ws, mixed, T, H, hc, r);
    }
    // L2: one subgroup per (t, i) output where i < r computes
    // lo[i] = silu(dot(w_down[i], xn)/hc) and rows i in [r, r+hc) compute the
    // inject logits (same dot shape over xn vs w_inject; no silu, no /hc) —
    // the inject fold removes the old 4-WG L4 launch entirely.
    const uint32_t rows2 = r + (w_inject ? hc : 0);
    auto e2 = ie::ps(q, "q4hc2_down", [&](sycl::handler& h) {
        h.depends_on({e1});
        h.parallel_for(sycl::nd_range<1>(uint64_t(T) * rows2 * SG, SG),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint32_t g2 = uint32_t(it.get_group(0));
            const uint32_t t = g2 / rows2, i = g2 % rows2;
            const uint32_t lid = uint32_t(it.get_local_id(0));
            auto sg = it.get_sub_group();
            const float* xn = xn_ws + uint64_t(t) * D;
            if (i < r) {
                // 8-wide vectorized dot: the row fits L2, the cost is the
                // scalar dependency chain — vec loads cut instructions ~8x.
                const auto* wr8 = reinterpret_cast<const sycl::vec<sycl::half, 8>*>(
                    w_down + uint64_t(i) * D);
                float p = 0.f;
                for (uint32_t j8 = lid; j8 < D / 8; j8 += SG) {
                    const sycl::vec<sycl::half, 8> v = wr8[j8];
                    const float* xj = xn + j8 * 8;
                    #pragma unroll
                    for (int u2 = 0; u2 < 8; ++u2) p += float(v[u2]) * xj[u2];
                }
                const float acc =
                    sycl::reduce_over_group(sg, p, sycl::plus<float>()) / float(hc);
                if (lid == 0)
                    lo_ws[uint64_t(t) * r + i] = acc / (1.f + sycl::exp(-acc));
            } else {
                const uint32_t s = i - r;
                const auto* wi4 = reinterpret_cast<const sycl::vec<float, 4>*>(
                    w_inject + uint64_t(s) * D);
                const auto* xn4 = reinterpret_cast<const sycl::vec<float, 4>*>(xn);
                float p = 0.f;
                for (uint32_t j4 = lid; j4 < D / 4; j4 += SG) {
                    const sycl::vec<float, 4> a = wi4[j4], b = xn4[j4];
                    #pragma unroll
                    for (int u2 = 0; u2 < 4; ++u2) p += a[u2] * b[u2];
                }
                const float acc = sycl::reduce_over_group(sg, p, sycl::plus<float>());
                if (lid == 0) inj_logits[uint64_t(t) * hc + s] = acc;
            }
        });
    });
    // L3: WG of 256 lanes; lo staged in SLM once per WG; per-lane k-loop
    // vectorized via half8 loads of the w_up row.
    return ie::ps(q, "q4hc2_up", [&](sycl::handler& h) {
        h.depends_on({e2});
        sycl::local_accessor<float, 1> slo(r, h);
        const uint64_t n = uint64_t(T) * H;
        h.parallel_for(sycl::nd_range<1>((n + 255) / 256 * 256, 256),
                       [=](sycl::nd_item<1> it) {
            const uint64_t g2 = it.get_global_id(0);
            const uint32_t lid = uint32_t(it.get_local_id(0));
            // All lanes of a WG share one t (H=2560 >= 256: a WG never spans
            // two tokens because WGs tile g2 = t*H + hh contiguously and
            // 256 | H). Stage lo[t] once.
            const uint32_t t_wg = uint32_t((uint64_t(it.get_group(0)) * 256) / H);
            for (uint32_t k = lid; k < r; k += 256)
                slo[k] = lo_ws[uint64_t(t_wg) * r + k];
            sycl::group_barrier(it.get_group());
            if (g2 >= n) return;
            const uint32_t t = uint32_t(g2 / H), hh = uint32_t(g2 % H);
            const float* xn = xn_ws + uint64_t(t) * D;
            float m = 0.f;
            for (uint32_t s = 0; s < hc; ++s) {
                const auto* wu = reinterpret_cast<const sycl::vec<sycl::half, 8>*>(
                    w_up + uint64_t(s * H + hh) * r);
                float d = 0.f;
                for (uint32_t k8 = 0; k8 < r / 8; ++k8) {
                    const sycl::vec<sycl::half, 8> v = wu[k8];
                    #pragma unroll
                    for (int u2 = 0; u2 < 8; ++u2)
                        d += float(v[u2]) * slo[k8 * 8 + u2];
                }
                const float gt = 1.f / (1.f + sycl::exp(-d));
                m += xn[uint64_t(s) * H + hh] * gt;
            }
            mixed[uint64_t(t) * H + hh] = sycl::half(m / float(hc));
        });
    });
}

sycl::event qwen4_hc_mix_prefill(sycl::queue& q,
                            const float* x, const float* gamma,
                            const sycl::half* w_down_t, const sycl::half* w_up_t,
                            const float* w_inject,
                            float* xn_ws, sycl::half* xn16_ws,
                            sycl::half* lo16_ws, float* gemm_ws,
                            sycl::half* mixed, float* inj_logits,
                            uint32_t T, uint32_t H, uint32_t hc, uint32_t r,
                            float eps,
                            const std::vector<sycl::event>& deps) {
    const uint32_t D = hc * H;
    auto e1 = qwen4_hc_norm_l1(q, x, gamma, xn_ws, T, H, hc, eps, deps);
    // Inject: v2's fp32 vec4 dot verbatim (one subgroup per (t, s)) — the
    // inject logits gate the residual WRITE, so they keep the exact path.
    sycl::event e_inj = e1;
    if (w_inject) {
        constexpr int SG = 16;
        e_inj = ie::ps(q, "q4hcp_inj", [&](sycl::handler& h) {
            h.depends_on({e1});
            h.parallel_for(sycl::nd_range<1>(uint64_t(T) * hc * SG, SG),
                           [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                const uint32_t g2 = uint32_t(it.get_group(0));
                const uint32_t t = g2 / hc, s = g2 % hc;
                const uint32_t lid = uint32_t(it.get_local_id(0));
                auto sg = it.get_sub_group();
                const float* xn = xn_ws + uint64_t(t) * D;
                const auto* wi4 = reinterpret_cast<const sycl::vec<float, 4>*>(
                    w_inject + uint64_t(s) * D);
                const auto* xn4 = reinterpret_cast<const sycl::vec<float, 4>*>(xn);
                float p = 0.f;
                for (uint32_t j4 = lid; j4 < D / 4; j4 += SG) {
                    const sycl::vec<float, 4> a = wi4[j4], b = xn4[j4];
                    #pragma unroll
                    for (int u2 = 0; u2 < 4; ++u2) p += a[u2] * b[u2];
                }
                const float acc = sycl::reduce_over_group(sg, p, sycl::plus<float>());
                if (lid == 0) inj_logits[uint64_t(t) * hc + s] = acc;
            });
        });
    }
    // Down: xn rounded to F16, one GEMM [T, D] x [D, r] -> gemm_ws [T, r].
    // w_down_t/w_up_t are the [K, N]-TRANSPOSED load-time copies (gemm_fp16's
    // B contract; feeding the [N, K] originals was garbage — caught by the
    // hc-test prefill case). In-place-NT alternatives both lost: oneDNN nt is
    // bit-nondeterministic at these shapes (2026-08-27 bisect, k-slice
    // atomics — the scorer instruments and run2 bit-eq need deterministic
    // prefill) and the hand NT kernel ran 3x slow (uncoalesced B staging).
    auto e2 = cast_fp32_to_fp16(q, xn_ws, xn16_ws, uint64_t(T) * D, {e1});
    auto e3 = gemm_fp16(q, xn16_ws, w_down_t, gemm_ws, T, r, D, {e2});
    // lo16 = f16(silu(c / hc)) — the GEMM-A operand for the up projection.
    auto e4 = ie::ps(q, "q4hcp_lo", [&](sycl::handler& h) {
        h.depends_on({e3});
        const uint64_t n = uint64_t(T) * r;
        h.parallel_for(sycl::nd_range<1>((n + 255) / 256 * 256, 256),
                       [=](sycl::nd_item<1> it) {
            const uint64_t i = it.get_global_id(0);
            if (i >= n) return;
            const float a = gemm_ws[i] / float(hc);
            lo16_ws[i] = sycl::half(a / (1.f + sycl::exp(-a)));
        });
    });
    // Up: one GEMM [T, r] x [r, D] -> gemm_ws [T, D] (gate logits).
    auto e5 = gemm_fp16(q, lo16_ws, w_up_t, gemm_ws, T, D, r, {e4});
    // Collapse: mixed[t, ch] = (1/hc) Sum_s xn[s*H+ch] * sigmoid(g[s*H+ch]).
    return ie::ps(q, "q4hcp_collapse", [&](sycl::handler& h) {
        h.depends_on({e5, e_inj});
        const uint64_t n = uint64_t(T) * H;
        h.parallel_for(sycl::nd_range<1>((n + 255) / 256 * 256, 256),
                       [=](sycl::nd_item<1> it) {
            const uint64_t i = it.get_global_id(0);
            if (i >= n) return;
            const uint32_t t = uint32_t(i / H), ch = uint32_t(i % H);
            const float* xn = xn_ws + uint64_t(t) * D;
            const float* g  = gemm_ws + uint64_t(t) * D;
            float m = 0.f;
            for (uint32_t s = 0; s < hc; ++s) {
                const uint32_t d = s * H + ch;
                m += xn[d] * (1.f / (1.f + sycl::exp(-g[d])));
            }
            mixed[i] = sycl::half(m / float(hc));
        });
    });
}

}  // namespace ie
