// src/model/inkling.cpp — see include/ie/inkling.hpp.
#include "ie/inkling.hpp"

#include "ie/allocator.hpp"
#include "ie/dequant_ref.hpp"
#include "ie/ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <fstream>
#include <numeric>
#include <random>
#include <thread>

namespace ie {
// Kernel names unique to this file (the SYCL kernel-name collision trap: two TUs with same-named kernels in an anonymous
// namespace ran the wrong device code).
namespace inkling_k {
class RmsK; class CastFHK; class AddK; template <class T> class RowsScaledK; template <class T> class SwigluK;
class SconvK; class SconvShiftK; class GemvF32K; class RouterK; class GatherK; class ScatterK;
template <int GQ> class MoeGateUpK; template <int DQ> class MoeDownK; class AttnDecK; class AttnCombK; class AttnPreK;
template <int NC> class GemvQ8F32K;
}

namespace {
constexpr uint32_t kWG = 256;

// y = x / rms(x) * w * post over rows of n (y and/or y16). In place is fine: a work-group reads its row before writing it.
sycl::event ink2_rmsnorm(sycl::queue& q, const float* x, const float* w, float* y, sycl::half* y16, uint32_t rows, uint32_t n,
                         float eps, float post = 1.f) {
    return q.parallel_for<inkling_k::RmsK>(sycl::nd_range<1>(uint64_t(rows) * kWG, kWG), [=](sycl::nd_item<1> it) {
        const uint64_t r = it.get_group(0);
        const uint32_t l = uint32_t(it.get_local_id(0));
        const float* xr = x + r * n;
        float s = 0.f;
        for (uint32_t i = l; i < n; i += kWG) { const float v = xr[i]; s += v * v; }
        s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
        const float inv = 1.0f / sycl::sqrt(s / float(n) + eps);
        for (uint32_t i = l; i < n; i += kWG) {
            const float v = xr[i] * inv * w[i] * post;
            if (y)   y[r * n + i] = v;
            if (y16) y16[r * n + i] = sycl::half(v);
        }
    });
}
sycl::event ink2_f2h(sycl::queue& q, const float* a, sycl::half* b, uint64_t n) {
    return q.parallel_for<inkling_k::CastFHK>(sycl::range<1>(n), [=](sycl::id<1> i) { b[i] = sycl::half(a[i]); });
}
sycl::event ink2_add(sycl::queue& q, float* x, const float* y, uint64_t n) {
    return q.parallel_for<inkling_k::AddK>(sycl::range<1>(n), [=](sycl::id<1> i) { x[i] += y[i]; });
}
// dst[t, i] = (dst[t, i] if accumulate) + scale * s[t] * o[t, i]: a SwiGLU path's output with its per-row scale
template <class In>
sycl::event ink2_rows_scaled(sycl::queue& q, float* dst, const In* o, const float* s, uint32_t T, uint32_t H, float scale, bool accumulate) {
    return q.parallel_for<inkling_k::RowsScaledK<In>>(sycl::range<1>(uint64_t(T) * H), [=](sycl::id<1> i) {
        const float v = scale * s[i / H] * float(o[i]);
        dst[i] = accumulate ? dst[i] + v : v;
    });
}
// SwiGLU per row in fp32 with a power-of-two scale: h16[r] = gm_r * silu(g[r]) * u[r] / s[r], gm_r = gamma[r * gstride] (or 1),
// s[r] >= 1 chosen so max|h16[r]| <= 4096 (callers multiply back by s). Inkling's outlier channels (|ffn_down| 677 x a 17.9
// conv tap on the oracle prompt) put rows near f16's range; the scale keeps the next f16 GEMV finite.
template <class In>
sycl::event ink2_swiglu_rows(sycl::queue& q, const In* g, const In* u, sycl::half* h16, float* s, uint32_t R, uint32_t n,
                             const float* gamma, uint32_t gstride) {
    return q.parallel_for<inkling_k::SwigluK<In>>(sycl::nd_range<1>(uint64_t(R) * kWG, kWG), [=](sycl::nd_item<1> it) {
        const uint64_t r = it.get_group(0);
        const uint32_t l = uint32_t(it.get_local_id(0));
        const In* gr = g + r * n; const In* ur = u + r * n;
        const float gm = gamma ? gamma[r * gstride] : 1.0f;
        float mx = 0.f;
        for (uint32_t i = l; i < n; i += kWG) {
            const float gv = float(gr[i]), uv = float(ur[i]);
            mx = sycl::fmax(mx, sycl::fabs(gm * (gv / (1.0f + sycl::exp(-gv)) * uv)));
        }
        mx = sycl::reduce_over_group(it.get_group(), mx, sycl::maximum<float>());
        float sc = 1.0f;
        while (mx / sc > 4096.0f) sc *= 2.0f;              // power of two: the division is exact
        const float inv = 1.0f / sc;
        for (uint32_t i = l; i < n; i += kWG) {
            const float gv = float(gr[i]), uv = float(ur[i]);
            h16[r * n + i] = sycl::half(gm * (gv / (1.0f + sycl::exp(-gv)) * uv) * inv);
        }
        if (l == 0) s[r] = sc;
    });
}
// shortconv: y[t][c] = x[t][c] + sum_j w[c][j] * xx[t + j][c], xx = the state's 3 rows then x's T rows (the last tap is the
// current input) -- ggml_ssm_conv's order, plus the built-in residual. y must not alias x.
sycl::event ink2_sconv(sycl::queue& q, const float* x, const float* st, const float* w, float* y, uint32_t T, uint32_t C) {
    return q.parallel_for<inkling_k::SconvK>(sycl::range<1>(uint64_t(T) * C), [=](sycl::id<1> id) {
        const uint64_t i = id[0];
        const uint32_t t = uint32_t(i / C), c = uint32_t(i % C);
        float acc = 0.f;
        for (uint32_t j = 0; j < 4; ++j) {
            const uint32_t rr = t + j;
            const float v = rr < 3 ? st[uint64_t(rr) * C + c] : x[uint64_t(rr - 3) * C + c];
            acc += w[uint64_t(c) * 4 + j] * v;
        }
        y[i] = x[i] + acc;
    });
}
// ... then the state becomes xx's last 3 rows
sycl::event ink2_sconv_shift(sycl::queue& q, const float* x, float* st, uint32_t T, uint32_t C) {
    return q.parallel_for<inkling_k::SconvShiftK>(sycl::range<1>(C), [=](sycl::id<1> id) {
        const uint32_t c = uint32_t(id[0]);
        float v[3];
        for (uint32_t r = 0; r < 3; ++r) { const uint32_t xr = T + r; v[r] = xr < 3 ? st[uint64_t(xr) * C + c] : x[uint64_t(xr - 3) * C + c]; }
        for (uint32_t r = 0; r < 3; ++r) st[uint64_t(r) * C + c] = v[r];
    });
}
// y[t, n] = sum_k W[n, k] * x[t, k]   (W row-major [N][K], fp32 throughout: the router, as llama.cpp computes it)
sycl::event ink2_gemv_f32(sycl::queue& q, const float* W, const float* x, float* y, uint32_t T, uint32_t K, uint32_t N) {
    return q.parallel_for<inkling_k::GemvF32K>(sycl::nd_range<1>(uint64_t(T) * N * kWG, kWG), [=](sycl::nd_item<1> it) {
        const uint64_t g = it.get_group(0);
        const uint64_t t = g / N, n = g % N;
        const uint32_t l = uint32_t(it.get_local_id(0));
        const float* wr = W + n * K;
        const float* xr = x + t * K;
        float s = 0.f;
        for (uint32_t k = l; k < K; k += kWG) s += wr[k] * xr[k];
        s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
        if (l == 0) y[t * N + n] = s;
    });
}
// The router, one work-group per row: every routed expert's selection score sigmoid(logit) + bias at once, k rounds of a
// group max over (order-preserving score bits << 32 | ~index) -> the picks in descending score (smaller index on ties);
// then the WEIGHTS: a softmax over logsigmoid of [the picks' RAW logits, the shared logits] x scale (llama.cpp:
// soft_max(-softplus(-x)) x expert_weights_scale x gscale). idx [T][k], w [T][k + n_shared] (the shared ones last).
sycl::event ink2_router(sycl::queue& q, const float* logits, const float* bias, int32_t* idx, float* wout, uint32_t T,
                        uint32_t n_routed, uint32_t n_shared, uint32_t k, float scale) {
    constexpr uint32_t WG = 256, kPer = 4, kMaxK = 16;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<int32_t, 1> isel(sycl::range<1>(kMaxK), h);
        h.parallel_for<inkling_k::RouterK>(sycl::nd_range<1>(uint64_t(T) * WG, WG), [=](sycl::nd_item<1> it) {
            const uint64_t t = it.get_group(0);
            const uint32_t lid = uint32_t(it.get_local_id(0));
            const float* row = logits + t * (n_routed + n_shared);
            uint64_t key[kPer];
            for (uint32_t j = 0; j < kPer; ++j) {
                const uint32_t e = lid + j * WG;
                key[j] = 0;                                        // 0 = no expert (never a real key)
                if (e >= n_routed) continue;
                float sc = 1.f / (1.f + sycl::exp(-row[e])) + bias[e];
                uint32_t u;
                if (!(sc == sc)) u = 0u;                           // NaN: below every number
                else {
                    if (sc == 0.f) sc = 0.f;                       // -0 orders as +0
                    u = sycl::bit_cast<uint32_t>(sc);
                    u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
                }
                key[j] = (uint64_t(u) << 32) | uint64_t(0xFFFFFFFFu - e);
            }
            for (uint32_t r = 0; r < k; ++r) {
                uint64_t best = 0;
                for (uint32_t j = 0; j < kPer; ++j) best = key[j] > best ? key[j] : best;
                best = sycl::reduce_over_group(it.get_group(), best, sycl::maximum<uint64_t>());
                for (uint32_t j = 0; j < kPer; ++j)
                    if (best != 0 && key[j] == best) { isel[r] = int32_t(0xFFFFFFFFu - uint32_t(best & 0xFFFFFFFFu)); key[j] = 0; }
            }
            sycl::group_barrier(it.get_group());
            if (lid == 0) {
                float v[kMaxK + 4];
                const uint32_t nw = k + n_shared;
                float m = -INFINITY;
                for (uint32_t r = 0; r < nw; ++r) {
                    const float x = r < k ? row[isel[r]] : row[n_routed + (r - k)];
                    const float ls = sycl::fmin(x, 0.f) - sycl::log1p(sycl::exp(-sycl::fabs(x)));   // logsigmoid
                    v[r] = ls; m = sycl::fmax(m, ls);
                }
                float sum = 0.f;
                for (uint32_t r = 0; r < nw; ++r) { v[r] = sycl::exp(v[r] - m); sum += v[r]; }
                for (uint32_t r = 0; r < k; ++r) idx[t * k + r] = isel[r];
                for (uint32_t r = 0; r < nw; ++r) wout[t * nw + r] = v[r] / sum * scale;
            }
        });
    });
}
// dst[r, :] = src[rows[r], :]
sycl::event ink2_gather_rows(sycl::queue& q, sycl::half* dst, const sycl::half* src, const int32_t* rows, uint32_t R, uint32_t H) {
    return q.parallel_for<inkling_k::GatherK>(sycl::range<1>(uint64_t(R) * H), [=](sycl::id<1> i) {
        const uint64_t r = i / H, c = i % H;
        dst[i] = src[uint64_t(rows[r]) * H + c];
    });
}
// acc[rows[r], :] += w[r] * s[r] * y[r, :]   (an expert's rows are distinct: no write conflict)
sycl::event ink2_scatter_add(sycl::queue& q, float* acc, const sycl::half* y, const int32_t* rows, const float* w, const float* s,
                             uint32_t R, uint32_t H) {
    return q.parallel_for<inkling_k::ScatterK>(sycl::range<1>(uint64_t(R) * H), [=](sycl::id<1> i) {
        const uint64_t r = i / H, c = i % H;
        acc[uint64_t(rows[r]) * H + c] += w[r] * s[r] * float(y[i]);
    });
}

// ---- decode MoE: the K picked experts in three launches (gate+up, ink2_swiglu_rows, down). Expert k's raw GGUF slices sit in
// its cache slot: base + slot[k] * stride. The lane lattices of moe_decode_gate_up_silu_q4k / moe_decode_down_q6k
// (src/ops/moe_fused.cpp), on AoS blocks; Q5_K = the Q4_K lattice plus bit `sub` of qh.
struct InkSlots { int32_t s[16]; };   // the picks' slots, by value: no upload
inline float ink2_h2f_bits(const uint8_t* p) { return float(sycl::bit_cast<sycl::half>(*reinterpret_cast<const uint16_t*>(p))); }
inline void ink2_scale_min_k4(int j, const uint8_t* sc, float& s, float& m) {
    if (j < 4) { s = float(sc[j] & 63); m = float(sc[j + 4] & 63); }
    else { s = float((sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4)); m = float((sc[j + 4] >> 4) | ((sc[j] >> 6) << 4)); }
}
// g[k, n], u[k, n] (fp32) = the Q4_K (GQ 4) / Q5_K (GQ 5) gate / up rows n of expert k (gate at slot + 0, up at + off_up) . x
template <int GQ>
sycl::event ink2_moe_gate_up(sycl::queue& q, const sycl::half* x, const uint8_t* base, uint64_t stride, uint64_t off_up,
                             InkSlots slot, float* g, float* u, uint32_t H, uint32_t EF, uint32_t K) {
    constexpr int SG = 16, NW = 32, WG = NW * SG;
    constexpr uint64_t BB = GQ == 4 ? 144 : 176;
    const uint32_t bpc = H / 256, wgs = (EF + NW - 1) / NW;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<sycl::half, 1> A(sycl::range<1>(H), h);
        h.parallel_for<inkling_k::MoeGateUpK<GQ>>(sycl::nd_range<2>({uint64_t(K), uint64_t(wgs) * WG}, {1, WG}),
                                                  [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint32_t k = uint32_t(it.get_group(0)), lid = uint32_t(it.get_local_id(1));
            const uint32_t lane = lid % SG, n = uint32_t(it.get_group(1)) * NW + lid / SG;
            for (uint32_t i = lid; i < H; i += WG) A[i] = x[i];
            sycl::group_barrier(it.get_group());
            if (n >= EF) return;
            const uint8_t* gw = base + uint64_t(slot.s[k]) * stride;
            const uint8_t* uw = gw + off_up;
            const int sub = int(lane) >> 1, half = int(lane) & 1, grp = sub >> 1, hi = sub & 1;
            const int qs_off = (GQ == 4 ? 16 : 48) + grp * 32 + half * 16, a_off = grp * 64 + hi * 32 + half * 16, sh = hi * 4;
            float ga = 0.f, ua = 0.f;
            for (uint32_t b = 0; b < bpc; ++b) {
                const uint8_t* gb = gw + (uint64_t(n) * bpc + b) * BB;    // d, dmin, scales[12], (qh[32],) qs[128]
                const uint8_t* ub = uw + (uint64_t(n) * bpc + b) * BB;
                float gs, gm, us, um;
                ink2_scale_min_k4(sub, gb + 4, gs, gm); ink2_scale_min_k4(sub, ub + 4, us, um);
                const float gd = ink2_h2f_bits(gb) * gs, gmn = ink2_h2f_bits(gb + 2) * gm;
                const float ud = ink2_h2f_bits(ub) * us, umn = ink2_h2f_bits(ub + 2) * um;
                const sycl::half* a = &A[b * 256 + a_off];
                float sg_ = 0.f, su_ = 0.f, sa = 0.f;
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const float av = float(a[i]);
                    int gq = (gb[qs_off + i] >> sh) & 0xF, uq = (ub[qs_off + i] >> sh) & 0xF;
                    if constexpr (GQ == 5) {
                        gq |= ((gb[16 + half * 16 + i] >> sub) & 1) << 4;
                        uq |= ((ub[16 + half * 16 + i] >> sub) & 1) << 4;
                    }
                    sg_ += av * float(gq); su_ += av * float(uq); sa += av;
                }
                ga += gd * sg_ - gmn * sa;
                ua += ud * su_ - umn * sa;
            }
            ga = sycl::reduce_over_group(it.get_sub_group(), ga, sycl::plus<float>());
            ua = sycl::reduce_over_group(it.get_sub_group(), ua, sycl::plus<float>());
            if (lane == 0) { g[uint64_t(k) * EF + n] = ga; u[uint64_t(k) * EF + n] = ua; }
        });
    });
}
// acc[n] += sum_k w[k] * s[k] * (down row n of expert k . h16[k])  -- DQ 5 (Q5_K) or 6 (Q6_K); base points at the slots'
// down slices; h16 [K, EF] and s [K] are ink2_swiglu_rows' output; w the router's weights (already x scale x gscale).
template <int DQ>
sycl::event ink2_moe_down(sycl::queue& q, const sycl::half* h16, const float* s, const uint8_t* base, uint64_t stride,
                          InkSlots slot, const float* w, float* acc, uint32_t H, uint32_t EF, uint32_t K) {
    constexpr int SG = 16, NW = 32, WG = NW * SG;
    constexpr uint64_t BB = DQ == 5 ? 176 : 210;
    const uint32_t bpc = EF / 256, wgs = (H + NW - 1) / NW;
    const uint64_t hn = uint64_t(K) * EF;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<sycl::half, 1> Hs(sycl::range<1>(hn), h);
        h.parallel_for<inkling_k::MoeDownK<DQ>>(sycl::nd_range<1>(uint64_t(wgs) * WG, WG),
                                                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint32_t lid = uint32_t(it.get_local_id(0)), lane = lid % SG, n = uint32_t(it.get_group(0)) * NW + lid / SG;
            for (uint64_t i = lid; i < hn; i += WG) Hs[i] = h16[i];
            sycl::group_barrier(it.get_group());
            if (n >= H) return;
            float total = 0.f;
            for (uint32_t k = 0; k < K; ++k) {
                const uint8_t* dw = base + uint64_t(slot.s[k]) * stride;
                const sycl::half* hk = &Hs[uint64_t(k) * EF];
                float ka = 0.f;
                for (uint32_t b = 0; b < bpc; ++b) {
                    const uint8_t* blk = dw + (uint64_t(n) * bpc + b) * BB;
                    if constexpr (DQ == 5) {   // block_q5_K: d, dmin, scales[12], qh[32], qs[128]
                        const int sub = int(lane) >> 1, half = int(lane) & 1, grp = sub >> 1, hi = sub & 1;
                        float sc, mn; ink2_scale_min_k4(sub, blk + 4, sc, mn);
                        const float d = ink2_h2f_bits(blk) * sc, m = ink2_h2f_bits(blk + 2) * mn;
                        const uint8_t* qs = blk + 48 + grp * 32 + half * 16;
                        const uint8_t* qh = blk + 16 + half * 16;
                        const sycl::half* a = hk + b * 256 + grp * 64 + hi * 32 + half * 16;
                        float saq = 0.f, sa = 0.f;
                        #pragma unroll
                        for (int i = 0; i < 16; ++i) {
                            const int qv = ((qs[i] >> (hi * 4)) & 0xF) | (((qh[i] >> sub) & 1) << 4);
                            const float av = float(a[i]);
                            saq += av * float(qv); sa += av;
                        }
                        ka += d * saq - m * sa;
                    } else {                   // block_q6_K: ql[128], qh[64], scales[16] (int8), d; gemv_q6_K's lattice
                        const int half_q = int(lane) >> 3, sub = (int(lane) >> 1) & 3, lh = int(lane) & 1, l0 = lh * 16;
                        const uint8_t* ql = blk + half_q * 64 + (sub & 1) * 32 + l0;
                        const uint8_t* qh = blk + 128 + half_q * 32 + l0;
                        const float d = ink2_h2f_bits(blk + 208) * float(int8_t(blk[192 + half_q * 8 + sub * 2 + lh]));
                        const int qsh = sub * 2, lsh = (sub & 2) ? 4 : 0;
                        const sycl::half* a = hk + b * 256 + half_q * 128 + sub * 32 + l0;
                        float saq = 0.f;
                        #pragma unroll
                        for (int i = 0; i < 16; ++i) {
                            const int qv = int(((ql[i] >> lsh) & 0xF) | (((qh[i] >> qsh) & 3) << 4)) - 32;
                            saq += float(a[i]) * float(qv);
                        }
                        ka += d * saq;
                    }
                }
                total += w[k] * s[k] * ka;
            }
            total = sycl::reduce_over_group(it.get_sub_group(), total, sycl::plus<float>());
            if (lane == 0) acc[n] += total;
        });
    });
}

// ---- attention. A key at position pk is visible to a query at pq iff pk >= 0, pk <= pq and (window == 0 or pq - pk < window);
// its score is tau * (q.k * scale + bias), bias = r[h] . proj[:, pq - pk] while pq - pk < E, else 0 (llama.cpp's banded
// flash attention: q and the relative logits both x tau on full layers; tau = 1 on sliding ones). kpos: the ring's positions
// (sliding layers) or nullptr (full layers: slot j holds position j). Decode (one query row): split over the keys -- one
// work-group per (KV head g, 64-key split) scores the n_rep query heads that share g, a split-local softmax and P.V into
// part [nh][S][hd + 2]; ink2_attn_combine merges the splits.
constexpr uint32_t kAttnKS = 64;
sycl::event ink2_attn_decode(sycl::queue& q, const float* qv, const float* k, const float* v, const int32_t* kpos,
                             const float* r, const float* proj, uint32_t E, uint32_t d_rel, int64_t pq, uint32_t window,
                             float tau, float* part, uint32_t nh, uint32_t nkv, uint32_t hd, uint32_t n_kv, float scale) {
    constexpr uint32_t WG = 256, SG = 16;
    const uint32_t S = (n_kv + kAttnKS - 1) / kAttnKS, n_rep = nh / nkv;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> qs(sycl::range<1>(uint64_t(n_rep) * hd), h);
        sycl::local_accessor<float, 1> rs(sycl::range<1>(uint64_t(n_rep) * d_rel), h);
        sycl::local_accessor<float, 1> sc(sycl::range<1>(uint64_t(n_rep) * kAttnKS), h);
        sycl::local_accessor<float, 1> ml(sycl::range<1>(uint64_t(n_rep) * 2), h);
        h.parallel_for<inkling_k::AttnDecK>(sycl::nd_range<2>({uint64_t(nkv), uint64_t(S) * WG}, {1, WG}),
                                            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint32_t g = uint32_t(it.get_group(0)), s = uint32_t(it.get_group(1)), lid = uint32_t(it.get_local_id(1));
            const uint32_t j0 = s * kAttnKS, nk = sycl::min(kAttnKS, n_kv - j0);
            for (uint32_t i = lid; i < n_rep * hd; i += WG) qs[i] = qv[uint64_t(g) * n_rep * hd + i];      // heads g*n_rep.. adjacent
            for (uint32_t i = lid; i < n_rep * d_rel; i += WG) rs[i] = r[uint64_t(g) * n_rep * d_rel + i];
            sycl::group_barrier(it.get_group());
            for (uint32_t i = lid; i < n_rep * kAttnKS; i += WG) {
                const uint32_t rr = i / kAttnKS, jj = i % kAttnKS;
                float val = -INFINITY;
                if (jj < nk) {
                    const int64_t pk = kpos ? int64_t(kpos[j0 + jj]) : int64_t(j0 + jj);
                    if (pk >= 0 && pk <= pq && (window == 0 || pq - pk < int64_t(window))) {
                        const float* kr = k + (uint64_t(j0 + jj) * nkv + g) * hd;
                        float d = 0.f;
                        for (uint32_t x = 0; x < hd; ++x) d += qs[rr * hd + x] * kr[x];
                        const int64_t dist = pq - pk;
                        float b = 0.f;
                        if (dist < int64_t(E)) for (uint32_t x = 0; x < d_rel; ++x) b += rs[rr * d_rel + x] * proj[uint64_t(x) * E + uint64_t(dist)];
                        val = tau * (d * scale + b);
                    }
                }
                sc[i] = val;
            }
            sycl::group_barrier(it.get_group());
            auto sg = it.get_sub_group();
            const uint32_t sgid = uint32_t(sg.get_group_id()[0]), lane = uint32_t(sg.get_local_id()[0]);
            if (sgid < n_rep) {                                   // sub-group r: head r's split max and sum
                float m = -INFINITY;
                for (uint32_t jj = lane; jj < kAttnKS; jj += SG) m = sycl::fmax(m, sc[sgid * kAttnKS + jj]);
                m = sycl::reduce_over_group(sg, m, sycl::maximum<float>());
                float l = 0.f;
                for (uint32_t jj = lane; jj < kAttnKS; jj += SG) {
                    const float p = m == -INFINITY ? 0.f : sycl::exp(sc[sgid * kAttnKS + jj] - m);
                    sc[sgid * kAttnKS + jj] = p; l += p;
                }
                l = sycl::reduce_over_group(sg, l, sycl::plus<float>());
                if (lane == 0) { ml[2 * sgid] = m; ml[2 * sgid + 1] = l; }
            }
            sycl::group_barrier(it.get_group());
            for (uint32_t i = lid; i < n_rep * hd; i += WG) {
                const uint32_t rr = i / hd, d = i % hd;
                float o = 0.f;
                for (uint32_t jj = 0; jj < nk; ++jj) {
                    const float p = sc[rr * kAttnKS + jj];
                    if (p != 0.f) o += p * v[(uint64_t(j0 + jj) * nkv + g) * hd + d];
                }
                float* pp = part + (uint64_t(g * n_rep + rr) * S + s) * (hd + 2);
                pp[d] = o;
                if (d == 0) { pp[hd] = ml[2 * rr]; pp[hd + 1] = ml[2 * rr + 1]; }
            }
        });
    });
}
// out[h][d] = sum_s e^(m_s - M) o_s[d] / sum_s e^(m_s - M) l_s over ink2_attn_decode's splits (all-masked splits: none)
sycl::event ink2_attn_combine(sycl::queue& q, const float* part, float* out, uint32_t nh, uint32_t hd, uint32_t n_kv) {
    const uint32_t S = (n_kv + kAttnKS - 1) / kAttnKS;
    return q.parallel_for<inkling_k::AttnCombK>(sycl::nd_range<1>(uint64_t(nh) * hd, hd), [=](sycl::nd_item<1> it) {
        const uint32_t h = uint32_t(it.get_group(0)), d = uint32_t(it.get_local_id(0));
        const float* ph = part + uint64_t(h) * S * (hd + 2);
        float M = -INFINITY;
        for (uint32_t s = 0; s < S; ++s) M = sycl::fmax(M, ph[uint64_t(s) * (hd + 2) + hd]);
        float L = 0.f, o = 0.f;
        if (M != -INFINITY)
            for (uint32_t s = 0; s < S; ++s) {
                const float* ps = ph + uint64_t(s) * (hd + 2);
                if (ps[hd] == -INFINITY) continue;
                const float w = sycl::exp(ps[hd] - M);
                L += w * ps[hd + 1]; o += w * ps[d];
            }
        out[uint64_t(h) * hd + d] = L > 0.f ? o / L : 0.f;
    });
}
// Prefill (T query rows): one sub-group per (row t, head h) walking the keys with an online softmax; lane x < d_rel also
// carries r[t][h][x] * proj[x][dist] into the score's reduction. tau per row from its position (alpha 0 = sliding: 1).
sycl::event ink2_attn_prefill(sycl::queue& q, const float* qv, const float* k, const float* v, const int32_t* kpos,
                              const float* r, const float* proj, uint32_t E, uint32_t d_rel, int64_t pos0, uint32_t window,
                              float alpha, float n_floor, float* out, uint32_t T, uint32_t nh, uint32_t nkv, uint32_t hd,
                              uint32_t n_kv, float scale) {
    constexpr int SG = 16;
    const uint32_t dpl = hd / SG, n_rep = nh / nkv;
    return q.parallel_for<inkling_k::AttnPreK>(sycl::nd_range<2>({uint64_t(T), uint64_t(nh) * SG}, {1, SG}),
                                               [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
        const uint32_t t = uint32_t(it.get_group(0)), hh = uint32_t(it.get_group(1)), lane = uint32_t(it.get_local_id(1));
        auto sg = it.get_sub_group();
        const uint32_t kvh = hh / n_rep;
        const int64_t pq = pos0 + int64_t(t);
        const float tau = alpha > 0.f ? 1.f + alpha * sycl::log(sycl::fmax(float(pq + 1) / n_floor, 1.f)) : 1.f;
        float qr[16], acc[16];
        for (uint32_t d = 0; d < dpl; ++d) { qr[d] = qv[(uint64_t(t) * nh + hh) * hd + lane * dpl + d]; acc[d] = 0.f; }
        const float rv = lane < d_rel ? r[(uint64_t(t) * nh + hh) * d_rel + lane] : 0.f;
        float m = -INFINITY, l = 0.f;
        for (uint32_t j = 0; j < n_kv; ++j) {
            const int64_t pk = kpos ? int64_t(kpos[j]) : int64_t(j);
            if (pk < 0 || pk > pq || (window != 0 && pq - pk >= int64_t(window))) continue;   // uniform over the sub-group
            const float* kr = k + (uint64_t(j) * nkv + kvh) * hd + lane * dpl;
            float part = 0.f;
            for (uint32_t d = 0; d < dpl; ++d) part += qr[d] * kr[d];
            const int64_t dist = pq - pk;
            part = part * scale + ((dist < int64_t(E) && lane < d_rel) ? rv * proj[uint64_t(lane) * E + uint64_t(dist)] : 0.f);
            const float s = tau * sycl::reduce_over_group(sg, part, sycl::plus<float>());
            const float m_new = sycl::fmax(m, s);
            const float a = m == -INFINITY ? 0.f : sycl::exp(m - m_new);
            const float e = sycl::exp(s - m_new);
            const float* vr = v + (uint64_t(j) * nkv + kvh) * hd + lane * dpl;
            for (uint32_t d = 0; d < dpl; ++d) acc[d] = acc[d] * a + e * vr[d];
            l = l * a + e; m = m_new;
        }
        const float inv = l > 0.f ? 1.f / l : 0.f;
        for (uint32_t d = 0; d < dpl; ++d) out[(uint64_t(t) * nh + hh) * hd + lane * dpl + d] = acc[d] * inv;
    });
}

// Decode's Q8_0 projections with an fp32 OUTPUT: gemv_q8_0_soa_f16_g's lattice (global-activation leaf: unpack the 32 int8
// of a block, fp32 FMA against the f16 x, x the block's scale, sub-group reduce) storing float. Inkling's rows outgrow f16:
// on the oracle prompt a shared expert's down projection put one channel past 65504 at layer 41 (decode), and the inf
// reached the residual; prefill never saw it (its GEMMs write fp32).
template <int NC>
sycl::event ink2_gemv_q8_f32_t(sycl::queue& q, const sycl::half* x, const int8_t* qs_W, const uint16_t* d_W, float* y,
                               uint32_t K, uint32_t N) {
    constexpr int SG = 16, WG = NC * SG;
    const uint32_t bpc = K / 32, n_wgs = (N + NC - 1) / NC;
    return q.parallel_for<inkling_k::GemvQ8F32K<NC>>(sycl::nd_range<1>(uint64_t(n_wgs) * WG, WG),
                                                     [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const uint32_t lid = uint32_t(it.get_local_id(0)), lane = lid % SG, n = uint32_t(it.get_group(0)) * NC + lid / SG;
        if (n >= N) return;
        const int8_t* wcol = qs_W + uint64_t(n) * K;
        const uint16_t* dcol = d_W + uint64_t(n) * bpc;
        float acc = 0.f;
        for (uint32_t b = lane; b < bpc; b += SG) {
            const sycl::vec<uint32_t, 4>* w128 = reinterpret_cast<const sycl::vec<uint32_t, 4>*>(wcol + uint64_t(b) * 32);
            const sycl::vec<sycl::half, 8>* xv = reinterpret_cast<const sycl::vec<sycl::half, 8>*>(x + uint64_t(b) * 32);
            const sycl::vec<uint32_t, 4> wa = w128[0], wb = w128[1];
            int8_t wv[32];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint32_t v = wa[j], u = wb[j];
                wv[4 * j + 0] = int8_t(uint8_t(v));       wv[4 * j + 1] = int8_t(uint8_t(v >> 8));
                wv[4 * j + 2] = int8_t(uint8_t(v >> 16)); wv[4 * j + 3] = int8_t(uint8_t(v >> 24));
                wv[16 + 4 * j + 0] = int8_t(uint8_t(u));       wv[16 + 4 * j + 1] = int8_t(uint8_t(u >> 8));
                wv[16 + 4 * j + 2] = int8_t(uint8_t(u >> 16)); wv[16 + 4 * j + 3] = int8_t(uint8_t(u >> 24));
            }
            sycl::vec<sycl::half, 8> xq[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) xq[j] = xv[j];
            float bacc = 0.f;
            #pragma unroll
            for (int i = 0; i < 32; ++i) bacc += float(wv[i]) * float(xq[i >> 3][i & 7]);
            acc += float(sycl::bit_cast<sycl::half>(dcol[b])) * bacc;
        }
        acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
        if (lane == 0) y[n] = acc;
    });
}
// the leaf's work-group shapes (gemv_q8_0_soa_f16_g's B70 sweep): 4 columns per group for narrow N, 16 to K 12288, else 32
sycl::event ink2_gemv_q8_f32(sycl::queue& q, const sycl::half* x, const InkQ8& W, float* y, uint32_t K, uint32_t N) {
    if (K <= 12288u) return N <= 256u ? ink2_gemv_q8_f32_t<4>(q, x, W.qs, W.d, y, K, N) : ink2_gemv_q8_f32_t<16>(q, x, W.qs, W.d, y, K, N);
    return ink2_gemv_q8_f32_t<32>(q, x, W.qs, W.d, y, K, N);
}

// Host dequant of `rows` rows of K elements from a GGUF tensor's raw rows into fp32.
std::string dequant_rows(const GgufTensorInfo* t, uint64_t row0, uint64_t rows, uint32_t K, float* out, const uint8_t* base = nullptr) {
    const uint64_t row_bytes = t->nbytes / (t->shape[1] * (t->n_dims > 2 ? t->shape[2] : 1));
    const uint8_t* p = (base ? base : t->data) + row0 * row_bytes;
    const size_t n = size_t(rows) * K;
    switch (t->dtype) {
        case DType::kF32:  std::memcpy(out, p, n * 4); return {};
        case DType::kQ8_0: ref::dequant_q8_0_buffer(p, n, out); return {};
        case DType::kQ4_K: ref::dequant_q4_K_buffer(p, n, out); return {};
        case DType::kQ5_K: ref::dequant_q5_K_buffer(p, n, out); return {};
        case DType::kQ6_K: ref::dequant_q6_K_buffer(p, n, out); return {};
        default: return "inkling: no host dequant for the dtype of " + std::string(t->name);
    }
}
uint64_t mem_available_bytes() {
    std::ifstream f("/proc/meminfo"); std::string k; uint64_t v = 0; std::string unit;
    while (f >> k >> v >> unit) if (k == "MemAvailable:") return v * 1024ull;
    return 0;
}
double env_f(const char* name, double def) { const char* v = std::getenv(name); return v && *v ? std::atof(v) : def; }
float tau_at(const InklingConfig& c, int64_t pos) {
    if (c.log_scaling_n_floor == 0) return 1.f;
    return 1.f + c.log_scaling_alpha * std::log(std::max(float(pos + 1) / float(c.log_scaling_n_floor), 1.f));
}
}  // namespace

template <class T> T* InklingModel::dev_alloc(uint64_t n) {
    T* p = static_cast<T*>(malloc_device_checked(*q_, n * sizeof(T)));   // the VRAM page-alias self-test on the big ones
    if (p) { allocs_.push_back(p); dev_bytes_ += n * sizeof(T); }
    return p;
}

// " (card 2: layers 22-41)" when the layers are split over two cards, "" on one card
void InklingModel::to_device(void* dst, const void* src, uint64_t n) {
    for (uint64_t off = 0; off < n; off += hstage_n_) {
        const uint64_t c = std::min(hstage_n_, n - off);
        std::memcpy(hstage_, static_cast<const uint8_t*>(src) + off, c);
        q_->memcpy(static_cast<uint8_t*>(dst) + off, hstage_, c).wait();
    }
}
void InklingModel::from_device(void* dst, const void* src, uint64_t n) {
    for (uint64_t off = 0; off < n; off += hstage_n_) {
        const uint64_t c = std::min(hstage_n_, n - off);
        q_->memcpy(hstage_, static_cast<const uint8_t*>(src) + off, c).wait();
        std::memcpy(static_cast<uint8_t*>(dst) + off, hstage_, c);
    }
}

std::string InklingModel::stage_tag() const {
    if (l0_ == 0 && l1_ == cfg_.n_layers) return {};
    return " (card " + std::to_string(l0_ == 0 ? 1 : 2) + ": layers " + std::to_string(l0_) + "-" + std::to_string(l1_ - 1) + ")";
}

InklingModel::~InklingModel() {
    if (q_) { try { q_->wait(); } catch (...) {} for (void* p : allocs_) sycl::free(p, *q_); }
}

std::string InklingModel::upload_f32(const GgufTensorInfo* t, uint64_t n, float*& out) {
    if (!t) return "inkling: missing tensor";
    if (t->dtype != DType::kF32) return "inkling: " + std::string(t->name) + " is not F32";
    if (t->nbytes != n * 4) return "inkling: " + std::string(t->name) + " holds " + std::to_string(t->nbytes / 4) + " floats, expected " + std::to_string(n);
    out = dev_alloc<float>(n);
    if (!out) return "inkling: device alloc failed for " + std::string(t->name);
    to_device(out, t->data, n * 4);
    return {};
}

// GGUF Q8_0 [K (ne0), N (ne1)(, n_slices)] -> slice `slice` on the card in SoA: qs [N][K], d [N][K/32].
std::string InklingModel::upload_q8(const GgufTensorInfo* t, uint32_t K, uint32_t N, InkQ8& out, uint32_t slice, uint32_t n_slices) {
    if (!t) return "inkling: missing tensor";
    const uint64_t s2 = t->n_dims > 2 ? t->shape[2] : 1;
    if (t->shape[0] != K || t->shape[1] != N || s2 != n_slices)
        return "inkling: " + std::string(t->name) + " shape [" + std::to_string(t->shape[0]) + "," + std::to_string(t->shape[1]) + "," +
               std::to_string(s2) + "] != [" + std::to_string(K) + "," + std::to_string(N) + "," + std::to_string(n_slices) + "]";
    if (t->dtype != DType::kQ8_0 || K % 32) return "inkling: " + std::string(t->name) + " must be Q8_0 with K % 32 == 0";
    const uint32_t bpc = K / 32;
    const uint8_t* src = t->data + uint64_t(slice) * (t->nbytes / n_slices);
    std::vector<int8_t> qs(uint64_t(N) * K);
    std::vector<uint16_t> d(uint64_t(N) * bpc);
    const uint32_t nt = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    std::vector<std::thread> ts;
    for (uint32_t w = 0; w < nt; ++w) ts.emplace_back([&, w] {
        for (uint64_t n = w; n < N; n += nt)
            for (uint32_t b = 0; b < bpc; ++b) {
                const uint8_t* blk = src + (n * bpc + b) * 34;   // block_q8_0: fp16 d, int8 qs[32]
                std::memcpy(&d[n * bpc + b], blk, 2);
                std::memcpy(&qs[n * K + uint64_t(b) * 32], blk + 2, 32);
            }
    });
    for (auto& th : ts) th.join();
    out.qs = dev_alloc<int8_t>(qs.size());
    out.d = dev_alloc<uint16_t>(d.size());
    if (!out.qs || !out.d) return "inkling: device alloc failed for " + std::string(t->name);
    to_device(out.qs, qs.data(), qs.size());
    to_device(out.d, d.data(), d.size() * 2);
    wscratch_n_ = std::max(wscratch_n_, uint64_t(K) * N);
    return {};
}

// Prefill GEMM over a Q8_0 projection: C[T, N] (fp32) = A[T, K] (f16) x W, W dequantized into the F16 scratch first
// (the in-order queue finishes one projection's GEMM before the next dequant overwrites the scratch).
void InklingModel::gemm_q8(const sycl::half* A, const InkQ8& W, float* C, uint32_t T, uint32_t K, uint32_t N) {
    dequant_q8_soa_f16(*q_, W.qs, W.d, wscratch_, K, N);
    gemm_fp16(*q_, A, wscratch_, C, T, N, K);
}

std::string InklingModel::load(sycl::queue& q, const GgufReader& g, const InklingConfig& cfg, sycl::queue* q2) {
    if (!q2) return load_stage(q, g, cfg, 0, cfg.n_layers);
    // two cards: the dense layers and half the MoE layers on the first, the other half and the head (~0.9 GiB) on the second
    const uint32_t split = cfg.dense_layers + (cfg.n_layers - cfg.dense_layers) / 2;
    if (split == 0 || split >= cfg.n_layers) return "inkling: no two-card split for " + std::to_string(cfg.n_layers) + " layers";
    if (auto e = load_stage(q, g, cfg, 0, split); !e.empty()) return e;
    next_ = std::make_unique<InklingModel>();
    return next_->load_stage(*q2, g, cfg, split, cfg.n_layers);
}

std::string InklingModel::load_stage(sycl::queue& q, const GgufReader& g, const InklingConfig& cfg, uint32_t l0, uint32_t l1) {
    if (!q.is_in_order()) return "inkling: the queue must be in-order";
    q_ = &q; g_ = &g; cfg_ = cfg; l0_ = l0; l1_ = l1;
    hstage_n_ = 64ull << 20;
    hstage_ = sycl::malloc_host<uint8_t>(hstage_n_, q);
    if (!hstage_) return "inkling: pinned staging alloc failed";
    allocs_.push_back(hstage_);
    const char* fm = std::getenv("IE_INKLING_FUSED_MOE");   // =0: decode runs the expert-major loop
    fused_moe_ = !(fm && std::string(fm) == "0");
    const uint32_t H = cfg.hidden, hd = cfg.head_dim, nh = cfg.n_q_heads, E = cfg.n_experts, S = cfg.n_shared_experts;
    const uint32_t EF = cfg.expert_ffn, dr = cfg.d_rel;
    if (cfg.head_dim != cfg.v_head_dim) return "inkling: key/value head dims differ";
    if (cfg.shortconv_kernel != 4) return "inkling: shortconv kernel " + std::to_string(cfg.shortconv_kernel) + " (only 4 is bound)";
    if (S != 2) return "inkling: " + std::to_string(S) + " shared experts (only 2 are bound)";
    if (cfg.n_experts_used + S > 20 || cfg.n_experts_used > 16) return "inkling: too many experts per token";
    if (dr > 16 || hd % 16 || hd > 256) return "inkling: d_rel <= 16 and head_dim a multiple of 16 (<= 256) are bound";
    if (EF % 256 || H % 256) return "inkling: hidden and expert_ffn must be multiples of 256";
    auto T = [&](const std::string& n) { return g.find_tensor(n); };
    std::string e;
    if (l0_ == 0) {                           // the first card embeds
        tok_embd_ = T("token_embd.weight");
        if (!tok_embd_ || tok_embd_->dtype != DType::kQ8_0 || tok_embd_->shape[0] != H) return "inkling: token_embd must be Q8_0 [hidden, vocab]";
        if (e = upload_f32(T("token_embd_norm.weight"), H, embd_norm_); !e.empty()) return e;
    }
    if (l1_ == cfg.n_layers) {                // the last card runs the head
        if (e = upload_f32(T("output_norm.weight"), H, out_norm_); !e.empty()) return e;
        if (e = upload_q8(T("output.weight"), H, cfg.vocab, head_); !e.empty()) return e;
    }
    wscratch_n_ = 0;                          // the head only ever runs as a GEMV: the prefill scratch is sized by the layers
    layers_.resize(cfg.n_layers);             // indexed by layer; only this card's [l0_, l1_) are loaded
    for (uint32_t L = l0_; L < l1_; ++L) {
        InklingLayer& w = layers_[L];
        const std::string b = "blk." + std::to_string(L) + ".";
        w.sliding = cfg.is_sliding(L); w.moe = L >= cfg.dense_layers;
        w.n_kv_heads = L < cfg.n_kv_heads.size() ? uint32_t(cfg.n_kv_heads[L]) : 0;
        w.rel_extent = cfg.rel_extent_for(L);
        if (!w.n_kv_heads || nh % w.n_kv_heads) return "inkling: layer " + std::to_string(L) + ": bad KV head count";
        const uint32_t kvd = w.n_kv_heads * hd;
        kvd_max_ = std::max(kvd_max_, kvd);
        if (e = upload_f32(T(b + "attn_norm.weight"), H, w.attn_norm); !e.empty()) return e;
        if (e = upload_f32(T(b + "ffn_norm.weight"), H, w.ffn_norm); !e.empty()) return e;
        if (e = upload_f32(T(b + "attn_q_norm.weight"), hd, w.q_norm); !e.empty()) return e;
        if (e = upload_f32(T(b + "attn_k_norm.weight"), hd, w.k_norm); !e.empty()) return e;
        if (e = upload_f32(T(b + "attn_rel_proj.weight"), uint64_t(dr) * w.rel_extent, w.rel_proj); !e.empty()) return e;   // [d_rel][E]
        if (e = upload_f32(T(b + "shortconv_k.weight"), uint64_t(kvd) * 4, w.sc_k); !e.empty()) return e;                 // [C][4]
        if (e = upload_f32(T(b + "shortconv_v.weight"), uint64_t(kvd) * 4, w.sc_v); !e.empty()) return e;
        if (e = upload_f32(T(b + "shortconv_attn.weight"), uint64_t(H) * 4, w.sc_attn); !e.empty()) return e;
        if (e = upload_f32(T(b + "shortconv_mlp.weight"), uint64_t(H) * 4, w.sc_mlp); !e.empty()) return e;
        if (const auto* gt = T(b + "ffn_gscale.weight"); gt && gt->dtype == DType::kF32 && gt->nbytes == 4) std::memcpy(&w.gscale, gt->data, 4);
        else return "inkling: layer " + std::to_string(L) + ": ffn_gscale must be one F32";
        if (e = upload_q8(T(b + "attn_q.weight"), H, nh * hd, w.wq); !e.empty()) return e;
        if (e = upload_q8(T(b + "attn_k.weight"), H, kvd, w.wk); !e.empty()) return e;
        if (e = upload_q8(T(b + "attn_v.weight"), H, kvd, w.wv); !e.empty()) return e;
        if (e = upload_q8(T(b + "attn_r.weight"), H, nh * dr, w.wr); !e.empty()) return e;
        if (e = upload_q8(T(b + "attn_output.weight"), nh * hd, H, w.wo); !e.empty()) return e;
        if (!w.moe) {
            if (e = upload_q8(T(b + "ffn_gate.weight"), H, cfg.ffn_dense, w.ffn_gate); !e.empty()) return e;
            if (e = upload_q8(T(b + "ffn_up.weight"), H, cfg.ffn_dense, w.ffn_up); !e.empty()) return e;
            if (e = upload_q8(T(b + "ffn_down.weight"), cfg.ffn_dense, H, w.ffn_down); !e.empty()) return e;
        } else {
            if (e = upload_f32(T(b + "ffn_gate_inp.weight"), uint64_t(H) * (E + S), w.router); !e.empty()) return e;
            if (e = upload_f32(T(b + "exp_probs_b.bias"), E, w.router_bias); !e.empty()) return e;
            for (uint32_t s = 0; s < S; ++s) {
                if (e = upload_q8(T(b + "ffn_gate_shexp.weight"), H, EF, w.sh_gate[s], s, S); !e.empty()) return e;
                if (e = upload_q8(T(b + "ffn_up_shexp.weight"), H, EF, w.sh_up[s], s, S); !e.empty()) return e;
                if (e = upload_q8(T(b + "ffn_down_shexp.weight"), EF, H, w.sh_down[s], s, S); !e.empty()) return e;
            }
            w.gate_exps = T(b + "ffn_gate_exps.weight"); w.up_exps = T(b + "ffn_up_exps.weight"); w.down_exps = T(b + "ffn_down_exps.weight");
            if (!w.gate_exps || !w.up_exps || !w.down_exps) return "inkling: layer " + std::to_string(L) + " misses an expert bank";
            auto kq = [](DType d) { return d == DType::kQ4_K ? 4 : d == DType::kQ5_K ? 5 : d == DType::kQ6_K ? 6 : 0; };
            w.gu_q = kq(w.gate_exps->dtype); w.down_q = kq(w.down_exps->dtype);
            if ((w.gu_q != 4 && w.gu_q != 5) || kq(w.up_exps->dtype) != w.gu_q)
                return "inkling: layer " + std::to_string(L) + ": gate/up experts must both be Q4_K or both Q5_K";
            if (w.down_q != 5 && w.down_q != 6) return "inkling: layer " + std::to_string(L) + ": down experts are neither Q5_K nor Q6_K";
            w.gate_slice = w.gate_exps->nbytes / E; w.up_slice = w.up_exps->nbytes / E; w.down_slice = w.down_exps->nbytes / E;
            w.gate_src = w.gate_exps->data; w.up_src = w.up_exps->data; w.down_src = w.down_exps->data;
        }
    }
    wscratch_ = dev_alloc<sycl::half>(wscratch_n_);
    if (!wscratch_) return "inkling: prefill F16 scratch alloc failed";
    return pin_experts();
}

// The routed experts in pinned host memory, so a cache miss is a DMA from pinned pages rather than a copy through pageable
// ones. Refused (the mapping stays the source) when it would leave less than IE_INKLING_RAM_FLOOR_GB (30) available, or with
// IE_INKLING_PIN=0. The copy runs on 16 threads from the page cache.
std::string InklingModel::pin_experts() {
    if (const char* v = std::getenv("IE_INKLING_PIN"); v && *v == '0') { std::printf("[inkling] experts: unpinned (IE_INKLING_PIN=0)\n"); return {}; }
    uint64_t need = 0;
    for (const auto& w : layers_) if (w.moe) need += w.gate_exps->nbytes + w.up_exps->nbytes + w.down_exps->nbytes;
    const uint64_t avail = mem_available_bytes(), floor_b = uint64_t(env_f("IE_INKLING_RAM_FLOOR_GB", 30.0) * 1073741824.0);
    if (avail < need + floor_b) {
        std::printf("[inkling] experts: NOT pinned (%.1f GiB needed, %.1f GiB available, floor %.0f GiB) -- misses read the mapping%s\n",
                    need / 1073741824.0, avail / 1073741824.0, floor_b / 1073741824.0, stage_tag().c_str());
        return {};
    }
    for (auto& w : layers_) {
        if (!w.moe) continue;
        for (const auto& bank : {std::pair<const GgufTensorInfo*, const uint8_t**>{w.gate_exps, &w.gate_src},
                                 {w.up_exps, &w.up_src}, {w.down_exps, &w.down_src}}) {
            const GgufTensorInfo* t = bank.first; const uint8_t** src = bank.second;
            uint8_t* p = sycl::malloc_host<uint8_t>(t->nbytes, *q_);
            if (!p) return "inkling: pinned alloc failed (" + std::to_string(t->nbytes >> 20) + " MiB)";
            allocs_.push_back(p); pinned_bytes_ += t->nbytes;
            const uint32_t nt = 16; const uint64_t per = (t->nbytes + nt - 1) / nt;
            std::vector<std::thread> ts;
            for (uint32_t k = 0; k < nt; ++k) ts.emplace_back([=] {
                const uint64_t a = k * per, b = std::min<uint64_t>(t->nbytes, a + per);
                if (a < b) std::memcpy(p + a, t->data + a, b - a);
            });
            for (auto& th : ts) th.join();
            *src = p;
        }
    }
    std::printf("[inkling] experts: %.1f GiB pinned in host memory%s\n", pinned_bytes_ / 1073741824.0, stage_tag().c_str());
    return {};
}

std::string InklingModel::init_runtime(uint32_t max_ctx, uint32_t max_chunk) {
    const uint32_t H = cfg_.hidden, hd = cfg_.head_dim, nh = cfg_.n_q_heads, E = cfg_.n_experts, S = cfg_.n_shared_experts;
    const uint32_t K = cfg_.n_experts_used, EF = cfg_.expert_ffn, F = std::max(cfg_.ffn_dense, EF), dr = cfg_.d_rel;
    max_ctx_ = max_ctx; max_chunk_ = max_chunk;
    // gemm_fp16 stores whole 8x16 tiles (its contract): rows are padded to a multiple of 8 so a ragged T never spills
    const uint64_t C = (uint64_t(max_chunk) + 7) / 8 * 8, qd = uint64_t(nh) * hd, kvd = kvd_max_;
    ring_ = std::min<uint32_t>(max_ctx, cfg_.sliding_window + uint32_t(C));
    ring_pos_.assign(ring_, -1);
    kc_.assign(cfg_.n_layers, nullptr); vc_.assign(cfg_.n_layers, nullptr);
    for (uint32_t L = l0_; L < l1_; ++L) {
        const uint64_t rows = layers_[L].sliding ? ring_ : max_ctx;
        const uint64_t lkv = uint64_t(layers_[L].n_kv_heads) * hd;
        kc_[L] = dev_alloc<float>(rows * lkv);
        vc_[L] = dev_alloc<float>(rows * lkv);
        if (!kc_[L] || !vc_[L]) return "inkling: KV alloc failed at layer " + std::to_string(L) + " (" + std::to_string(rows) + " positions)";
    }
    cstate_layer_ = 3 * (2 * kvd + 2 * uint64_t(H));
    cstate_ = dev_alloc<float>(cstate_layer_ * cfg_.n_layers);
    csnap_ = dev_alloc<float>(cstate_layer_ * cfg_.n_layers);
    d_ring_pos_ = dev_alloc<int32_t>(std::max<uint32_t>(1, ring_));
    x_ = dev_alloc<float>(C * H); xn_ = dev_alloc<float>(C * H); acc_ = dev_alloc<float>(C * H); fy_ = dev_alloc<float>(C * H);
    ffo_ = dev_alloc<float>(C * H); ao_ = dev_alloc<float>(C * H);
    qf_ = dev_alloc<float>(C * qd); kf_ = dev_alloc<float>(C * kvd); vf_ = dev_alloc<float>(C * kvd);
    ka_ = dev_alloc<float>(C * kvd); va_ = dev_alloc<float>(C * kvd); rf_ = dev_alloc<float>(C * nh * dr);
    attn_ = dev_alloc<float>(C * qd);
    fg_ = dev_alloc<float>(C * F); fu_ = dev_alloc<float>(C * F); fh16_ = dev_alloc<sycl::half>(C * F);
    rlog_ = dev_alloc<float>(C * (E + S)); rw_ = dev_alloc<float>(C * (K + S)); ridx_ = dev_alloc<int32_t>(C * K);
    d_rows_ = dev_alloc<int32_t>(C * K); d_rw_ = dev_alloc<float>(C * K); sc_ = dev_alloc<float>(C); gs_ = dev_alloc<float>(C);
    xn16_ = dev_alloc<sycl::half>(C * H); a16_ = dev_alloc<sycl::half>(C * qd);
    gx_ = dev_alloc<sycl::half>(C * H); gg_ = dev_alloc<sycl::half>(C * EF); gu_ = dev_alloc<sycl::half>(C * EF);
    gh_ = dev_alloc<sycl::half>(C * EF); gy_ = dev_alloc<sycl::half>(C * H);
    logits16_ = dev_alloc<sycl::half>(cfg_.vocab);
    attn_part_ = dev_alloc<float>(qd / hd * ((std::max(max_ctx, ring_) + kAttnKS - 1) / kAttnKS) * (hd + 2));
    for (void* p : {(void*)cstate_, (void*)csnap_, (void*)d_ring_pos_, (void*)x_, (void*)xn_, (void*)acc_, (void*)fy_, (void*)ffo_,
                    (void*)ao_, (void*)qf_, (void*)kf_, (void*)vf_, (void*)ka_, (void*)va_, (void*)rf_, (void*)attn_, (void*)fg_,
                    (void*)fu_, (void*)fh16_, (void*)rlog_, (void*)rw_, (void*)ridx_, (void*)d_rows_, (void*)d_rw_, (void*)sc_,
                    (void*)gs_, (void*)xn16_, (void*)a16_, (void*)gx_, (void*)gg_, (void*)gu_, (void*)gh_, (void*)gy_,
                    (void*)logits16_, (void*)attn_part_})
        if (!p) return "inkling: workspace alloc failed";
    if (next_ || l0_ > 0) {                   // two cards: the residual's pinned hand-off buffer (one per card's context)
        hx_ = sycl::malloc_host<float>(C * H, *q_);
        if (!hx_) return "inkling: pinned hand-off buffer alloc failed";
        allocs_.push_back(hx_);
    }
    // ---- the expert cache: the card's free memory minus the reserve, the same slot count on every MoE layer, each layer's
    // slots sized for its own quants (one Q5_K gate/up layer and two Q6_K down layers would otherwise size every slot) ----
    auto align = [](uint64_t v) { return (v + 255) / 256 * 256; };
    ec_.assign(cfg_.n_layers, {});
    uint64_t per_slot_all = 0; uint32_t n_moe = 0;
    for (uint32_t L = 0; L < cfg_.n_layers; ++L) {
        const InklingLayer& w = layers_[L];
        if (!w.moe) continue;
        InklingECache& c = ec_[L];
        c.off_up = align(w.gate_slice); c.off_down = c.off_up + align(w.up_slice); c.slot_bytes = c.off_down + align(w.down_slice);
        per_slot_all += c.slot_bytes; ++n_moe;
    }
    const auto& dev = q_->get_device();
    uint64_t free_b = dev.has(sycl::aspect::ext_intel_free_memory) ? dev.get_info<sycl::ext::intel::info::device::free_memory>() : 0;
    if (!free_b) { const uint64_t total = dev.get_info<sycl::info::device::global_mem_size>(); free_b = total > dev_bytes_ ? total - dev_bytes_ : 0; }
    const uint64_t reserve = uint64_t(env_f("IE_INKLING_RESERVE_GB", 1.5) * 1073741824.0);
    uint64_t budget = free_b > reserve ? free_b - reserve : 0;
    if (const char* v = std::getenv("IE_INKLING_ECACHE_GB"); v && *v) budget = std::min<uint64_t>(budget, uint64_t(std::atof(v) * 1073741824.0));
    ec_slots_ = n_moe ? uint32_t(std::min<uint64_t>(E, budget / per_slot_all)) : 0;
    if (ec_slots_ < K + 2) return "inkling: room for only " + std::to_string(ec_slots_) + " expert slots per layer (need >= " + std::to_string(K + 2) + ")";
    for (uint32_t L = 0; L < cfg_.n_layers; ++L) {
        if (!layers_[L].moe) continue;
        InklingECache& c = ec_[L];
        c.base = dev_alloc<uint8_t>(uint64_t(ec_slots_) * c.slot_bytes);
        if (!c.base) return "inkling: expert-cache alloc failed at layer " + std::to_string(L);
        c.slot_of.assign(E, -1); c.expert_of.assign(ec_slots_, -1); c.stamp.assign(ec_slots_, 0);
    }
    std::printf("[inkling] expert cache: %u slots per MoE layer (%.0f%% of %u experts), %.2f GiB; %.2f GiB on the card in all%s\n",
                ec_slots_, 100.0 * ec_slots_ / E, E, uint64_t(ec_slots_) * per_slot_all / 1073741824.0, dev_bytes_ / 1073741824.0,
                stage_tag().c_str());
    pos_ = 0; snap_pos_ = 0;
    return next_ ? next_->init_runtime(max_ctx, max_chunk) : std::string();
}

// The slot holding expert e of layer L, filling it on a miss (LRU victim). The queue is in-order, so a slot reused inside one
// layer pass is only overwritten after the kernels that read its previous expert.
const uint8_t* InklingModel::acquire(const InklingLayer& w, uint32_t L, int32_t e) {
    InklingECache& c = ec_[L];
    int32_t s = c.slot_of[e];
    if (s >= 0) { ++ec_hits_; c.stamp[s] = ++ec_clock_; return c.base + uint64_t(s) * c.slot_bytes; }
    ++ec_misses_;
    s = int32_t(std::min_element(c.stamp.begin(), c.stamp.end()) - c.stamp.begin());
    if (c.expert_of[s] >= 0) c.slot_of[c.expert_of[s]] = -1;
    c.expert_of[s] = int16_t(e); c.slot_of[e] = int16_t(s); c.stamp[s] = ++ec_clock_;
    uint8_t* p = c.base + uint64_t(s) * c.slot_bytes;
    q_->memcpy(p,              w.gate_src + uint64_t(e) * w.gate_slice, w.gate_slice);
    q_->memcpy(p + c.off_up,   w.up_src   + uint64_t(e) * w.up_slice,   w.up_slice);
    q_->memcpy(p + c.off_down, w.down_src + uint64_t(e) * w.down_slice, w.down_slice);
    return p;
}

// A SwiGLU FFN over T rows of xn16_ (a dense layer's FFN, a shared expert): dst[t] (=|+=) scale * down(gm_t * swiglu(gate x,
// up x)), with gm_t = gamma[t * gamma_stride] (a shared expert's router weight, before down as llama.cpp applies it) or 1.
void InklingModel::swiglu_ffn(const InkQ8& gate, const InkQ8& up, const InkQ8& down, uint32_t F, uint32_t T, float* dst,
                              bool accumulate, float scale, const float* gamma, uint32_t gamma_stride) {
    sycl::queue& q = *q_;
    const uint32_t H = cfg_.hidden;
    if (T == 1) {                                     // fp32 outputs: decode rows outgrow f16 (ink2_gemv_q8_f32)
        ink2_gemv_q8_f32(q, xn16_, gate, fg_, H, F);
        ink2_gemv_q8_f32(q, xn16_, up, fu_, H, F);
        ink2_swiglu_rows<float>(q, fg_, fu_, fh16_, sc_, 1, F, gamma, gamma_stride);
        ink2_gemv_q8_f32(q, fh16_, down, fy_, F, H);
        ink2_rows_scaled<float>(q, dst, fy_, sc_, 1, H, scale, accumulate);
    } else {
        gemm_q8(xn16_, gate, fg_, T, H, F);
        gemm_q8(xn16_, up, fu_, T, H, F);
        ink2_swiglu_rows<float>(q, fg_, fu_, fh16_, sc_, T, F, gamma, gamma_stride);
        gemm_q8(fh16_, down, fy_, T, F, H);
        ink2_rows_scaled<float>(q, dst, fy_, sc_, T, H, scale, accumulate);
    }
}

// The routed experts into acc_ (h_idx_ holds the picks; rw_ the weights [T][k + shared]). Decode: the k picks in three
// launches; prefill: expert-major, each picked expert acquired once and run over every row that picked it, least-picked
// first so the LRU ends the chunk holding the experts this chunk used most.
std::string InklingModel::moe_experts(const InklingLayer& w, uint32_t L, uint32_t T) {
    sycl::queue& q = *q_;
    const uint32_t H = cfg_.hidden, EF = cfg_.expert_ffn, E = cfg_.n_experts, K = cfg_.n_experts_used;
    const uint32_t NW = K + cfg_.n_shared_experts;
    if (T == 1 && fused_moe_) {
        InkSlots sl{};
        for (uint32_t k = 0; k < K; ++k) {
            const int32_t e = h_idx_[k];
            if (e < 0 || uint32_t(e) >= E) return "inkling: router picked expert " + std::to_string(e);
            sl.s[k] = int32_t((acquire(w, L, e) - ec_[L].base) / ec_[L].slot_bytes);
        }
        const InklingECache& c = ec_[L];
        if (w.gu_q == 4) ink2_moe_gate_up<4>(q, xn16_, c.base, c.slot_bytes, c.off_up, sl, fg_, fu_, H, EF, K);
        else             ink2_moe_gate_up<5>(q, xn16_, c.base, c.slot_bytes, c.off_up, sl, fg_, fu_, H, EF, K);
        ink2_swiglu_rows<float>(q, fg_, fu_, gh_, gs_, K, EF, nullptr, 0);
        if (w.down_q == 5) ink2_moe_down<5>(q, gh_, gs_, c.base + c.off_down, c.slot_bytes, sl, rw_, acc_, H, EF, K);
        else               ink2_moe_down<6>(q, gh_, gs_, c.base + c.off_down, c.slot_bytes, sl, rw_, acc_, H, EF, K);
        return {};
    }
    std::vector<uint32_t> cnt(E + 1, 0);
    for (uint64_t i = 0; i < uint64_t(T) * K; ++i) {
        const int32_t e = h_idx_[i];
        if (e < 0 || uint32_t(e) >= E) return "inkling: router picked expert " + std::to_string(e);
        ++cnt[uint32_t(e) + 1];
    }
    for (uint32_t e = 0; e < E; ++e) cnt[e + 1] += cnt[e];             // offsets
    h_rows_.assign(uint64_t(T) * K, 0); h_rw_.assign(uint64_t(T) * K, 0.f);
    std::vector<uint32_t> fill(cnt.begin(), cnt.end() - 1);
    for (uint32_t t = 0; t < T; ++t)
        for (uint32_t k = 0; k < K; ++k) {
            const uint32_t e = uint32_t(h_idx_[uint64_t(t) * K + k]), at = fill[e]++;
            h_rows_[at] = int32_t(t); h_rw_[at] = h_w_[uint64_t(t) * NW + k];
        }
    to_device(d_rows_, h_rows_.data(), h_rows_.size() * 4);
    to_device(d_rw_, h_rw_.data(), h_rw_.size() * 4);
    std::vector<uint32_t> order(E);
    std::iota(order.begin(), order.end(), 0u);
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return cnt[a + 1] - cnt[a] < cnt[b + 1] - cnt[b]; });
    for (const uint32_t e : order) {
        const uint32_t off = cnt[e], R = cnt[e + 1] - cnt[e];
        if (!R) continue;
        const uint8_t* slot = acquire(w, L, int32_t(e));
        const uint8_t *sg = slot, *su = slot + ec_[L].off_up, *sd = slot + ec_[L].off_down;
        const sycl::half* A = xn16_ + uint64_t(h_rows_[off]) * H;          // one row: read it in place
        if (R > 1) { ink2_gather_rows(q, gx_, xn16_, d_rows_ + off, R, H); A = gx_; }
        if (w.gu_q == 4) {
            if (R == 1) { gemv_q4_K(q, A, sg, gg_, H, EF); gemv_q4_K(q, A, su, gu_, H, EF); }
            else        { gemv_q4_K_rows(q, A, sg, gg_, H, EF, R); gemv_q4_K_rows(q, A, su, gu_, H, EF, R); }
        } else {
            if (R == 1) { gemv_q5_K(q, A, sg, gg_, H, EF); gemv_q5_K(q, A, su, gu_, H, EF); }
            else        { gemv_q5_K_rows(q, A, sg, gg_, H, EF, R); gemv_q5_K_rows(q, A, su, gu_, H, EF, R); }
        }
        ink2_swiglu_rows<sycl::half>(q, gg_, gu_, gh_, gs_, R, EF, nullptr, 0);
        if (w.down_q == 5) {
            if (R == 1) gemv_q5_K(q, gh_, sd, gy_, EF, H); else gemv_q5_K_rows(q, gh_, sd, gy_, EF, H, R);
        } else if (R == 1) {
            gemv_q6_K(q, gh_, sd, gy_, EF, H);
        } else {
            for (uint32_t r0 = 0; r0 < R; r0 += 8)                         // gemm_q6_K takes at most 8 rows (more: it does NOTHING)
                gemm_q6_K(q, gh_ + uint64_t(r0) * EF, sd, gy_ + uint64_t(r0) * H, std::min(8u, R - r0), EF, H);
        }
        ink2_scatter_add(q, acc_, gy_, d_rows_ + off, d_rw_ + off, gs_, R, H);
    }
    return {};
}

// shortconv over T rows of x (C channels) into y, then the rolling state advances
void InklingModel::sconv(float* x, float* y, float* state, const float* w, uint32_t T, uint32_t C) {
    ink2_sconv(*q_, x, state, w, y, T, C);
    ink2_sconv_shift(*q_, x, state, T, C);
}

uint32_t InklingModel::restart_at(uint32_t want) {
    if (next_) next_->restart_at(want);       // the cards hold the same positions and snapshots: the same answer
    if (want >= pos_) return pos_;
    const uint32_t W = cfg_.sliding_window;
    auto ring_has = [&](uint32_t p) {
        const uint32_t lo = p > W - 1 ? p - (W - 1) : 0;
        for (uint32_t x = lo; x < p; ++x) if (ring_pos_[x % ring_] != int64_t(x)) return false;
        return true;
    };
    if (snap_pos_ && snap_pos_ <= want && ring_has(snap_pos_)) {
        q_->memcpy(cstate_, csnap_, cstate_layer_ * cfg_.n_layers * sizeof(float)).wait();
        for (auto& v : ring_pos_) if (v >= int64_t(snap_pos_)) v = -1;
        pos_ = snap_pos_;
        return pos_;
    }
    pos_ = 0; snap_pos_ = 0;
    std::fill(ring_pos_.begin(), ring_pos_.end(), -1);
    return 0;
}

std::string InklingModel::forward(const int32_t* ids, uint32_t T, uint32_t pos0, float* logits_host) {
    if (!x_) return "inkling: init_runtime first";
    if (T == 0 || T > max_chunk_) return "inkling: T out of range";
    if (pos0 != 0 && pos0 != pos_) return "inkling: forward at " + std::to_string(pos0) + " but the sequence is at " + std::to_string(pos_);
    if (uint64_t(pos0) + T > max_ctx_) return "inkling: context full";
    if (!next_) return stage_forward(ids, T, pos0, logits_host);
    if (auto e = stage_forward(ids, T, pos0, nullptr); !e.empty()) return e;
    // the residual [T, hidden] to the second card: card 1 -> its pinned buffer -> card 2's pinned buffer -> card 2
    const uint64_t bytes = uint64_t(T) * cfg_.hidden * 4;
    q_->memcpy(hx_, x_, bytes).wait();
    std::memcpy(next_->hx_, hx_, bytes);
    next_->q_->memcpy(next_->x_, next_->hx_, bytes);
    next_->probe = probe;
    return next_->stage_forward(nullptr, T, pos0, logits_host);
}

std::string InklingModel::prefill(const int32_t* ids, uint32_t n, uint32_t pos0, float* logits_host,
                                  const std::function<bool(uint32_t)>& after_chunk) {
    if (n == 0) return "inkling: empty prefill";
    static const bool overlap = !(std::getenv("IE_INKLING_OVERLAP") && *std::getenv("IE_INKLING_OVERLAP") == '0');
    if (!next_ || n <= max_chunk_ || !overlap) {   // one card, one chunk, or IE_INKLING_OVERLAP=0: one chunk at a time
        for (uint32_t off = 0; off < n; off += max_chunk_) {
            const uint32_t T = std::min(max_chunk_, n - off);
            const bool last = off + T == n;
            if (auto e = forward(ids + off, T, pos0 + off, last ? logits_host : nullptr); !e.empty()) return e;
            if (!last && after_chunk && !after_chunk(pos0 + off + T)) return {};
        }
        return {};
    }
    if (!x_) return "inkling: init_runtime first";
    if (pos0 != 0 && pos0 != pos_) return "inkling: prefill at " + std::to_string(pos0) + " but the sequence is at " + std::to_string(pos_);
    if (uint64_t(pos0) + n > max_ctx_) return "inkling: context full";
    // card 2 runs chunk k on its own thread while this thread runs chunk k+1 on card 1; the cards share no state (each
    // takes its own prompt-end snapshot, in chunk order)
    std::future<std::string> card2;
    uint32_t card2_end = 0;
    bool stop = false;
    for (uint32_t off = 0; off < n && !stop; off += max_chunk_) {
        const uint32_t T = std::min(max_chunk_, n - off), p = pos0 + off;
        const bool last = off + T == n;
        std::string e = stage_forward(ids + off, T, p, nullptr);
        if (card2.valid()) {
            const std::string e2 = card2.get();
            if (e.empty()) e = e2;
            if (e.empty() && after_chunk && !after_chunk(card2_end)) stop = true;   // ends after this chunk, already on card 1
        }
        if (!e.empty()) return e;
        const uint64_t bytes = uint64_t(T) * cfg_.hidden * 4;          // the residual to card 2 (idle now)
        q_->memcpy(hx_, x_, bytes).wait();
        std::memcpy(next_->hx_, hx_, bytes);
        next_->q_->memcpy(next_->x_, next_->hx_, bytes);
        next_->probe = probe;
        card2 = std::async(std::launch::async, [this, T, p, out = last ? logits_host : nullptr] {
            return next_->stage_forward(nullptr, T, p, out);
        });
        card2_end = p + T;
    }
    return card2.get();
}

// This card's part of a forward: the embeddings (first card; otherwise x_ already holds the residual), its layers, the
// head (last card, when logits_host is given), the prompt-end snapshot.
std::string InklingModel::stage_forward(const int32_t* ids, uint32_t T, uint32_t pos0, float* logits_host) {
    sycl::queue& q = *q_;
    const uint32_t H = cfg_.hidden, hd = cfg_.head_dim, nh = cfg_.n_q_heads, K = cfg_.n_experts_used, S = cfg_.n_shared_experts;
    const uint32_t E = cfg_.n_experts, dr = cfg_.d_rel, qd = nh * hd, NW = K + S;
    const float eps = cfg_.rms_eps, kq_scale = 1.0f / float(hd);    // Inkling's attention scale is 1/head_dim
    if (pos0 == 0) {
        std::fill(ring_pos_.begin(), ring_pos_.end(), -1);
        q.memset(cstate_, 0, cstate_layer_ * cfg_.n_layers * sizeof(float));
        snap_pos_ = 0;
    }

    // IE_INKLING_DUMP=1: layer 0's stages, named as llama.cpp's graph names them (whole-tensor sums, for llama-eval-callback)
    static const bool dump = std::getenv("IE_INKLING_DUMP") != nullptr;
    auto dumpf = [&](const char* name, const float* d, uint64_t n) {
        q.wait(); std::vector<float> h(n); q.memcpy(h.data(), d, n * 4).wait();
        double s = 0; for (float v : h) s += v;
        std::printf("[dump] %-26s sum %+14.6f  [0..2] %+.5f %+.5f %+.5f\n", name, s, h[0], h[1], h[2]);
    };

    // ---- embeddings (host dequant of the rows), then token_embd_norm ----
    if (l0_ == 0) {
        h_emb_.resize(uint64_t(T) * H);
        for (uint32_t t = 0; t < T; ++t) {
            if (ids[t] < 0 || uint32_t(ids[t]) >= cfg_.vocab) return "inkling: token id out of range";
            if (auto e = dequant_rows(tok_embd_, uint64_t(ids[t]), 1, H, h_emb_.data() + uint64_t(t) * H); !e.empty()) return e;
        }
        to_device(x_, h_emb_.data(), h_emb_.size() * 4);
        if (dump) dumpf("embd", x_, uint64_t(T) * H);
        ink2_rmsnorm(q, x_, embd_norm_, x_, nullptr, T, H, eps);
        if (dump) dumpf("inkling_embd_norm", x_, uint64_t(T) * H);
    }

    // ---- the sliding layers' ring: slot (pos % ring_) <- this chunk's positions ----
    for (uint32_t t = 0; t < T; ++t) ring_pos_[(pos0 + t) % ring_] = int64_t(pos0 + t);
    h_rpos_.resize(ring_);
    for (uint32_t j = 0; j < ring_; ++j) h_rpos_[j] = int32_t(ring_pos_[j]);
    to_device(d_ring_pos_, h_rpos_.data(), uint64_t(ring_) * 4);

    h_idx_.resize(uint64_t(T) * K); h_w_.resize(uint64_t(T) * NW);
    for (uint32_t L = l0_; L < l1_; ++L) {
        const InklingLayer& w = layers_[L];
        const uint32_t nkv = w.n_kv_heads, kvd = nkv * hd, Er = w.rel_extent;
        float* cs = cstate_ + uint64_t(L) * cstate_layer_;
        float *st_k = cs, *st_v = cs + 3 * uint64_t(kvd_max_), *st_a = cs + 6 * uint64_t(kvd_max_), *st_m = st_a + 3 * uint64_t(H);
        // ---- attention ----
        ink2_rmsnorm(q, x_, w.attn_norm, xn_, xn16_, T, H, eps);
        if (T == 1) {
            ink2_gemv_q8_f32(q, xn16_, w.wq, qf_, H, qd);
            ink2_gemv_q8_f32(q, xn16_, w.wk, ka_, H, kvd);
            ink2_gemv_q8_f32(q, xn16_, w.wv, va_, H, kvd);
            ink2_gemv_q8_f32(q, xn16_, w.wr, rf_, H, nh * dr);
        } else {
            gemm_q8(xn16_, w.wq, qf_, T, H, qd);
            gemm_q8(xn16_, w.wk, ka_, T, H, kvd);
            gemm_q8(xn16_, w.wv, va_, T, H, kvd);
            gemm_q8(xn16_, w.wr, rf_, T, H, nh * dr);
        }
        const bool D = dump && L == 0;
        if (D) { dumpf("inkling_attn_norm-0", xn_, uint64_t(T) * H); dumpf("inkling_attn_q-0", qf_, uint64_t(T) * qd);
                 dumpf("inkling_attn_k-0", ka_, uint64_t(T) * kvd); dumpf("inkling_attn_v-0", va_, uint64_t(T) * kvd);
                 dumpf("inkling_attn_r-0", rf_, uint64_t(T) * nh * dr); }
        sconv(ka_, kf_, st_k, w.sc_k, T, kvd);                                // k/v shortconvs on the flat projections
        sconv(va_, vf_, st_v, w.sc_v, T, kvd);
        if (D) { dumpf("inkling_attn_k_sconv-0", kf_, uint64_t(T) * kvd); dumpf("inkling_attn_v_sconv-0", vf_, uint64_t(T) * kvd); }
        ink2_rmsnorm(q, qf_, w.q_norm, qf_, nullptr, T * nh, hd, eps);        // per-head q/k RMSNorm, after the convs
        ink2_rmsnorm(q, kf_, w.k_norm, kf_, nullptr, T * nkv, hd, eps);
        if (D) { dumpf("inkling_attn_q_norm-0", qf_, uint64_t(T) * qd); dumpf("inkling_attn_k_norm-0", kf_, uint64_t(T) * kvd); }
        if (!w.sliding) {
            q.memcpy(kc_[L] + uint64_t(pos0) * kvd, kf_, uint64_t(T) * kvd * 4);
            q.memcpy(vc_[L] + uint64_t(pos0) * kvd, vf_, uint64_t(T) * kvd * 4);
        } else {                                                          // the ring: rows at slots (pos % ring_), wrapping once at most
            const uint32_t s0 = pos0 % ring_, first = std::min(T, ring_ - s0);
            q.memcpy(kc_[L] + uint64_t(s0) * kvd, kf_, uint64_t(first) * kvd * 4);
            q.memcpy(vc_[L] + uint64_t(s0) * kvd, vf_, uint64_t(first) * kvd * 4);
            if (first < T) {
                q.memcpy(kc_[L], kf_ + uint64_t(first) * kvd, uint64_t(T - first) * kvd * 4);
                q.memcpy(vc_[L], vf_ + uint64_t(first) * kvd, uint64_t(T - first) * kvd * 4);
            }
        }
        const uint32_t n_kv = w.sliding ? ring_ : pos0 + T;
        const int32_t* kpos = w.sliding ? d_ring_pos_ : nullptr;
        const uint32_t window = w.sliding ? cfg_.sliding_window : 0;
        if (T == 1) {
            ink2_attn_decode(q, qf_, kc_[L], vc_[L], kpos, rf_, w.rel_proj, Er, dr, int64_t(pos0), window,
                             w.sliding ? 1.f : tau_at(cfg_, pos0), attn_part_, nh, nkv, hd, n_kv, kq_scale);
            ink2_attn_combine(q, attn_part_, attn_, nh, hd, n_kv);
        } else {
            ink2_attn_prefill(q, qf_, kc_[L], vc_[L], kpos, rf_, w.rel_proj, Er, dr, int64_t(pos0), window,
                              (w.sliding || !cfg_.log_scaling_n_floor) ? 0.f : cfg_.log_scaling_alpha, float(cfg_.log_scaling_n_floor),
                              attn_, T, nh, nkv, hd, n_kv, kq_scale);
        }
        if (D) dumpf("kqv_out-0", attn_, uint64_t(T) * qd);
        ink2_f2h(q, attn_, a16_, uint64_t(T) * qd);
        if (T == 1) ink2_gemv_q8_f32(q, a16_, w.wo, ao_, qd, H);
        else        gemm_q8(a16_, w.wo, ao_, T, qd, H);
        if (D) dumpf("inkling_attn_o-0", ao_, uint64_t(T) * H);
        sconv(ao_, ffo_, st_a, w.sc_attn, T, H);
        if (D) dumpf("inkling_attn_sconv-0", ffo_, uint64_t(T) * H);
        ink2_add(q, x_, ffo_, uint64_t(T) * H);
        // ---- FFN ----
        ink2_rmsnorm(q, x_, w.ffn_norm, xn_, xn16_, T, H, eps);
        if (D) dumpf("inkling_ffn_norm-0", xn_, uint64_t(T) * H);
        if (!w.moe) {
            swiglu_ffn(w.ffn_gate, w.ffn_up, w.ffn_down, cfg_.ffn_dense, T, acc_, false, w.gscale, nullptr, 0);
        } else {
            ink2_gemv_f32(q, w.router, xn_, rlog_, T, H, E + S);
            ink2_router(q, rlog_, w.router_bias, ridx_, rw_, T, E, S, K, cfg_.expert_weights_scale * w.gscale);
            from_device(h_idx_.data(), ridx_, uint64_t(T) * K * 4);
            if (!(T == 1 && fused_moe_)) from_device(h_w_.data(), rw_, uint64_t(T) * NW * 4);   // decode: the weights stay on the card
            q.memset(acc_, 0, uint64_t(T) * H * 4);
            for (uint32_t s = 0; s < S; ++s)                                  // shared experts: their router weight scales the SwiGLU
                swiglu_ffn(w.sh_gate[s], w.sh_up[s], w.sh_down[s], cfg_.expert_ffn, T, acc_, true, 1.f, rw_ + K + s, NW);
            if (auto e = moe_experts(w, L, T); !e.empty()) return e;
        }
        if (D) dumpf("inkling_dense_ffn_out-0", acc_, uint64_t(T) * H);
        sconv(acc_, ffo_, st_m, w.sc_mlp, T, H);
        if (D) dumpf("inkling_ffn_sconv-0", ffo_, uint64_t(T) * H);
        ink2_add(q, x_, ffo_, uint64_t(T) * H);
        if (D) dumpf("l_out-0", x_, uint64_t(T) * H);
        if (probe) { q.wait(); probe(L, q, x_, T); }
    }
    // ---- head: the last row, RMSNorm x 1/logit_scale_denom, then the padded ids -inf ----
    if (logits_host) {
        ink2_rmsnorm(q, x_ + uint64_t(T - 1) * H, out_norm_, nullptr, xn16_, 1, H, eps, 1.0f / cfg_.logit_scale_denom);
        gemv_q8_0_soa_f16_g(q, xn16_, head_.qs, head_.d, logits16_, H, cfg_.vocab);
        std::vector<sycl::half> l16(cfg_.vocab);
        from_device(l16.data(), logits16_, cfg_.vocab * sizeof(sycl::half));
        for (uint32_t i = 0; i < cfg_.vocab; ++i) logits_host[i] = float(l16[i]);
        const uint32_t live = cfg_.vocab_unpadded ? std::min(cfg_.vocab_unpadded, cfg_.vocab) : cfg_.vocab;
        for (uint32_t i = live; i < cfg_.vocab; ++i) logits_host[i] = -INFINITY;
    }
    if (T > 1) {                                                          // a prompt's end: the restart point (restart_at)
        q.memcpy(csnap_, cstate_, cstate_layer_ * cfg_.n_layers * sizeof(float));
        snap_pos_ = pos0 + T;
    }
    q.wait();
    pos_ = pos0 + T;
    return {};
}

// Each novel kernel against a host reference: the Q8_0 projection, the router, the stateful shortconv, decode attention
// (and the prefill kernel on the same row), the fused decode MoE; on synthetic inputs and the real weights.
std::string InklingModel::self_check() {
    sycl::queue& q = *q_;
    const uint32_t H = cfg_.hidden, hd = cfg_.head_dim, nh = cfg_.n_q_heads;
    std::mt19937 rng(7); std::normal_distribution<float> nd(0.f, 1.f);
    auto rel_err = [](const std::vector<double>& ref, const float* got, size_t n) {
        double num = 0, den = 0;
        for (size_t i = 0; i < n; ++i) { const double d = double(got[i]) - ref[i]; num += d * d; den += ref[i] * ref[i]; }
        return std::sqrt(num / std::max(den, 1e-30));
    };
    auto report = [](const char* what, double rel, double bound) -> std::string {
        std::printf("[inkling self-check] %-52s rel err %.2e\n", what, rel);
        return rel < bound ? std::string() : "inkling self-check FAILED: " + std::string(what);
    };
    std::string err;
    if (const std::string tag = stage_tag(); !tag.empty()) std::printf("[inkling self-check]%s\n", tag.c_str());
    // (1) this card's first layer's attn_q through the decode GEMV
    {
        const InklingLayer& w0 = layers_[l0_];
        const auto* t = g_->find_tensor("blk." + std::to_string(l0_) + ".attn_q.weight");
        const uint32_t N = nh * hd;
        std::vector<sycl::half> x16(H); std::vector<float> xf(H);
        for (uint32_t i = 0; i < H; ++i) { x16[i] = sycl::half(nd(rng)); xf[i] = float(x16[i]); }
        sycl::half* dx = sycl::malloc_device<sycl::half>(H, q); float* dy = sycl::malloc_device<float>(N, q);
        q.memcpy(dx, x16.data(), H * 2).wait();
        ink2_gemv_q8_f32(q, dx, w0.wq, dy, H, N);
        std::vector<float> y32(N); q.memcpy(y32.data(), dy, N * 4).wait();
        sycl::free(dx, q); sycl::free(dy, q);
        std::vector<double> ref(N); std::vector<float> row(H), got(N);
        for (uint32_t n = 0; n < N && err.empty(); ++n) {
            err = dequant_rows(t, n, 1, H, row.data());
            double s = 0; for (uint32_t k = 0; k < H; ++k) s += double(row[k]) * xf[k];
            ref[n] = s; got[n] = y32[n];
        }
        if (err.empty()) err = report(("attn_q Q8_0 SoA GEMV, fp32 out (layer " + std::to_string(l0_) + ")").c_str(), rel_err(ref, got.data(), N), 1e-5);
    }
    // (2) the router: 3 rows of real-scale logits (sigmoid selection + bias, joint logsigmoid softmax with the shared logits)
    if (err.empty()) {
        const uint32_t E = cfg_.n_experts, S = cfg_.n_shared_experts, K = cfg_.n_experts_used, T = 3, NW = K + S;
        std::vector<float> lg(uint64_t(T) * (E + S)), bias(E);
        for (auto& v : lg) v = 2.f * nd(rng);
        for (auto& v : bias) v = 0.1f * nd(rng);
        const float scale = 8.f * 0.75f;
        float *dl = sycl::malloc_device<float>(lg.size(), q), *db = sycl::malloc_device<float>(E, q), *dw = sycl::malloc_device<float>(T * NW, q);
        int32_t* di = sycl::malloc_device<int32_t>(T * K, q);
        q.memcpy(dl, lg.data(), lg.size() * 4); q.memcpy(db, bias.data(), E * 4).wait();
        ink2_router(q, dl, db, di, dw, T, E, S, K, scale);
        std::vector<int32_t> gi(T * K); std::vector<float> gw(T * NW);
        q.memcpy(gi.data(), di, gi.size() * 4); q.memcpy(gw.data(), dw, gw.size() * 4).wait();
        for (void* p : {(void*)dl, (void*)db, (void*)dw, (void*)di}) sycl::free(p, q);
        double worst = 0; bool picks_ok = true;
        for (uint32_t t = 0; t < T; ++t) {
            const float* row = lg.data() + uint64_t(t) * (E + S);
            std::vector<uint32_t> ord(E); std::iota(ord.begin(), ord.end(), 0u);
            auto score = [&](uint32_t e) { return 1.0 / (1.0 + std::exp(-double(row[e]))) + bias[e]; };
            std::stable_sort(ord.begin(), ord.end(), [&](uint32_t a, uint32_t b) { return score(a) > score(b); });
            std::vector<double> v(NW); double m = -1e300, sum = 0;
            for (uint32_t r = 0; r < NW; ++r) {
                const double x = r < K ? row[ord[r]] : row[E + r - K];
                v[r] = std::min(x, 0.0) - std::log1p(std::exp(-std::fabs(x))); m = std::max(m, v[r]);
            }
            for (auto& x : v) { x = std::exp(x - m); sum += x; }
            for (uint32_t r = 0; r < NW; ++r) {
                if (r < K && uint32_t(gi[t * K + r]) != ord[r]) picks_ok = false;
                worst = std::max(worst, std::fabs(gw[t * NW + r] - v[r] / sum * scale) / scale);
            }
        }
        std::printf("[inkling self-check] router: picks %s, max weight err %.2e (of the scale)\n", picks_ok ? "identical" : "DIFFER", worst);
        if (!picks_ok || worst > 1e-5) err = "inkling self-check FAILED: router";
    }
    // (3) the stateful shortconv: 5 rows then 1 row continue exactly as 6 rows from a zero state
    if (err.empty()) {
        const uint32_t C = 300, T1 = 5;
        std::vector<float> x(6 * C), w(C * 4);
        for (auto& v : x) v = nd(rng);
        for (auto& v : w) v = 0.5f * nd(rng);
        float *dx = sycl::malloc_device<float>(x.size(), q), *dy = sycl::malloc_device<float>(x.size(), q);
        float *dw = sycl::malloc_device<float>(w.size(), q), *dst = sycl::malloc_device<float>(3 * C, q);
        q.memcpy(dx, x.data(), x.size() * 4); q.memcpy(dw, w.data(), w.size() * 4); q.memset(dst, 0, 3 * C * 4).wait();
        sconv(dx, dy, dst, dw, T1, C);
        sconv(dx + T1 * C, dy + T1 * C, dst, dw, 1, C);
        std::vector<float> y(x.size()); q.memcpy(y.data(), dy, y.size() * 4).wait();
        for (void* p : {(void*)dx, (void*)dy, (void*)dw, (void*)dst}) sycl::free(p, q);
        std::vector<double> ref(x.size());
        for (uint32_t t = 0; t < 6; ++t)
            for (uint32_t c = 0; c < C; ++c) {
                double a = 0;
                for (uint32_t j = 0; j < 4; ++j) { const int64_t src = int64_t(t) - 3 + j; if (src >= 0) a += double(w[c * 4 + j]) * x[src * C + c]; }
                ref[uint64_t(t) * C + c] = x[uint64_t(t) * C + c] + a;
            }
        err = report("shortconv, 5 rows + 1 row with the rolling state", rel_err(ref, y.data(), y.size()), 1e-6);
    }
    // (4) attention, one query row: decode (split-K) and prefill kernels vs host fp64. Full: 1101 keys at positions 0..1100,
    // E 1024 (the oldest keys past the bias), tau 1.3. Sliding: a 512-window at 700 over a ring of shuffled positions (E 512)
    for (int full = 1; full >= 0 && err.empty(); --full) {
        const uint32_t nkv = layers_[l0_].n_kv_heads, dr = cfg_.d_rel, E = full ? 1024 : 512, n_kv = full ? 1101 : 520, window = full ? 0 : 512;
        const int64_t pq = full ? 1100 : 700; const float tau = full ? 1.3f : 1.f;
        std::vector<float> hq(nh * hd), hk(uint64_t(n_kv) * nkv * hd), hv(hk.size()), hr(nh * dr), hp(uint64_t(dr) * E);
        std::vector<int32_t> kp(n_kv);
        for (auto& v : hq) v = nd(rng); for (auto& v : hk) v = nd(rng); for (auto& v : hv) v = nd(rng);
        for (auto& v : hr) v = nd(rng); for (auto& v : hp) v = 0.3f * nd(rng);
        for (uint32_t j = 0; j < n_kv; ++j)          // the ring: positions 200..700 shuffled, 100..108 (outside the window), empty slots
            kp[j] = full ? int32_t(j) : j < 501 ? int32_t(200 + (j * 37) % 501) : j < 510 ? int32_t(100 + (j - 501)) : -1;
        float *dq = sycl::malloc_device<float>(hq.size(), q), *dk = sycl::malloc_device<float>(hk.size(), q), *dv = sycl::malloc_device<float>(hv.size(), q);
        float *dr_ = sycl::malloc_device<float>(hr.size(), q), *dp = sycl::malloc_device<float>(hp.size(), q);
        float *o1 = sycl::malloc_device<float>(hq.size(), q), *o2 = sycl::malloc_device<float>(hq.size(), q);
        int32_t* dkp = sycl::malloc_device<int32_t>(n_kv, q);
        q.memcpy(dq, hq.data(), hq.size() * 4); q.memcpy(dk, hk.data(), hk.size() * 4); q.memcpy(dv, hv.data(), hv.size() * 4);
        q.memcpy(dr_, hr.data(), hr.size() * 4); q.memcpy(dp, hp.data(), hp.size() * 4); q.memcpy(dkp, kp.data(), n_kv * 4).wait();
        const float scl = 1.f / float(hd);
        ink2_attn_decode(q, dq, dk, dv, full ? nullptr : dkp, dr_, dp, E, dr, pq, window, tau, attn_part_, nh, nkv, hd, n_kv, scl);
        ink2_attn_combine(q, attn_part_, o1, nh, hd, n_kv);
        // the prefill kernel takes tau from the position: pick alpha/n_floor so position 700 gives the same tau
        const float alpha = full ? (tau - 1.f) / std::log(float(pq + 1) / 100.f) : 0.f;
        ink2_attn_prefill(q, dq, dk, dv, full ? nullptr : dkp, dr_, dp, E, dr, pq, window, alpha, 100.f, o2, 1, nh, nkv, hd, n_kv, scl);
        std::vector<float> a(hq.size()), b(hq.size());
        q.memcpy(a.data(), o1, a.size() * 4); q.memcpy(b.data(), o2, b.size() * 4).wait();
        for (void* p : {(void*)dq, (void*)dk, (void*)dv, (void*)dr_, (void*)dp, (void*)o1, (void*)o2, (void*)dkp}) sycl::free(p, q);
        std::vector<double> ref(hq.size());
        const uint32_t n_rep = nh / nkv;
        for (uint32_t h = 0; h < nh; ++h) {
            const uint32_t g = h / n_rep;
            std::vector<double> s(n_kv, -1e300); double m = -1e300;
            for (uint32_t j = 0; j < n_kv; ++j) {
                const int64_t pk = kp[j];
                if (pk < 0 || pk > pq || (window && pq - pk >= window)) continue;
                double d = 0; for (uint32_t x = 0; x < hd; ++x) d += double(hq[h * hd + x]) * hk[(uint64_t(j) * nkv + g) * hd + x];
                double bb = 0; const int64_t dist = pq - pk;
                if (dist < int64_t(E)) for (uint32_t x = 0; x < dr; ++x) bb += double(hr[h * dr + x]) * hp[uint64_t(x) * E + dist];
                s[j] = tau * (d * scl + bb); m = std::max(m, s[j]);
            }
            double l = 0; std::vector<double> o(hd, 0.0);
            for (uint32_t j = 0; j < n_kv; ++j) {
                if (s[j] <= -1e299) continue;
                const double p = std::exp(s[j] - m); l += p;
                for (uint32_t x = 0; x < hd; ++x) o[x] += p * hv[(uint64_t(j) * nkv + g) * hd + x];
            }
            for (uint32_t x = 0; x < hd; ++x) ref[h * hd + x] = o[x] / l;
        }
        const std::string tag = std::string(full ? "attention, full layer (tau 1.3), " : "attention, sliding ring, ");
        err = report((tag + "decode split-K").c_str(), rel_err(ref, a.data(), a.size()), 1e-5);
        if (err.empty()) err = report((tag + "prefill kernel").c_str(), rel_err(ref, b.data(), b.size()), 1e-5);
    }
    // (5) the fused decode MoE vs a host fp64 reference over the real experts, one layer per (gate/up, down) quant pair
    if (err.empty()) {
        const uint32_t K = cfg_.n_experts_used, EF = cfg_.expert_ffn, NW = K + cfg_.n_shared_experts;
        const int32_t picks[16] = {3, 17, 37, 64, 99, 128, 150, 201, 230, 255, 5, 7, 11, 13, 19, 23};
        std::vector<sycl::half> x16(H); std::vector<double> xd(H);
        for (uint32_t i = 0; i < H; ++i) { x16[i] = sycl::half(nd(rng)); xd[i] = float(x16[i]); }
        std::vector<float> wk(NW, 0.f);
        for (uint32_t k = 0; k < K; ++k) wk[k] = 0.3f + 0.1f * std::fabs(nd(rng));
        h_idx_.assign(picks, picks + K); h_w_ = wk;
        const bool keep = fused_moe_;
        std::vector<int> seen;
        for (uint32_t L = 0; L < cfg_.n_layers && err.empty(); ++L) {
            const InklingLayer& w = layers_[L];
            if (!w.moe) continue;
            const int pair = w.gu_q * 10 + w.down_q;
            if (std::find(seen.begin(), seen.end(), pair) != seen.end()) continue;
            seen.push_back(pair);
            std::vector<double> ref(H, 0.0), hh(EF);
            std::vector<float> row(std::max(H, EF));
            for (uint32_t k = 0; k < K && err.empty(); ++k) {
                const uint64_t e = uint64_t(picks[k]);
                for (uint32_t n = 0; n < EF && err.empty(); ++n) {
                    double gv = 0, uv = 0;
                    err = dequant_rows(w.gate_exps, e * EF + n, 1, H, row.data(), w.gate_src);
                    for (uint32_t i = 0; i < H; ++i) gv += row[i] * xd[i];
                    if (err.empty()) err = dequant_rows(w.up_exps, e * EF + n, 1, H, row.data(), w.up_src);
                    for (uint32_t i = 0; i < H; ++i) uv += row[i] * xd[i];
                    hh[n] = gv / (1.0 + std::exp(-gv)) * uv;
                }
                for (uint32_t n = 0; n < H && err.empty(); ++n) {
                    err = dequant_rows(w.down_exps, e * H + n, 1, EF, row.data(), w.down_src);
                    double y = 0; for (uint32_t i = 0; i < EF; ++i) y += row[i] * hh[i];
                    ref[n] += wk[k] * y;
                }
            }
            double rel[2] = {0, 0};
            for (int f = 0; f < 2 && err.empty(); ++f) {
                fused_moe_ = f == 1;
                q.memcpy(xn16_, x16.data(), uint64_t(H) * 2);
                q.memcpy(rw_, wk.data(), uint64_t(NW) * 4);
                q.memset(acc_, 0, uint64_t(H) * 4).wait();
                if (auto e = moe_experts(w, L, 1); !e.empty()) { err = e; break; }
                std::vector<float> got(H); q.memcpy(got.data(), acc_, uint64_t(H) * 4).wait();
                rel[f] = rel_err(ref, got.data(), H);
            }
            fused_moe_ = keep;
            if (!err.empty()) break;
            std::printf("[inkling self-check] decode MoE, gate/up Q%d_K, down Q%d_K (layer %u): fused rel err %.2e | loop %.2e\n",
                        w.gu_q, w.down_q, L, rel[1], rel[0]);
            if (rel[1] >= 5e-3 || rel[0] >= 5e-3) err = "inkling self-check FAILED: decode MoE (layer " + std::to_string(L) + ")";
        }
    }
    if (err.empty() && next_) err = next_->self_check();
    return err;
}

}  // namespace ie
