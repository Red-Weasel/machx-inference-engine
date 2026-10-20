// src/model/laguna.cpp — see include/ie/laguna.hpp.
#include "ie/laguna.hpp"

#include "ie/allocator.hpp"
#include "ie/dequant_ref.hpp"
#include "ie/inkling_ops.hpp"
#include "ie/laguna_ops.hpp"
#include "ie/ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <numeric>
#include <random>
#include <thread>

namespace ie {
// Kernel names are unique to this file (the SYCL kernel-name collision trap: two TUs with same-named kernels in an
// anonymous namespace ran the wrong device code).
namespace laguna_k {
class RmsK; class CastFHK; class CastHFK; class AddF32K; class GemvF32K; class AddRowsHK; class AddRowsFK;
class GatherK; class ScatterK; template <class T> class SwigluRowsK; class MoeGateUpK; template <int QT> class MoeDownK;
class AttnDecK; class AttnCombK;
}

namespace {
constexpr uint32_t kWG = 256;

sycl::event lag_rmsnorm(sycl::queue& q, const float* x, const float* w, float* y, sycl::half* y16,
                        uint32_t rows, uint32_t n, float eps) {
    return q.parallel_for<laguna_k::RmsK>(sycl::nd_range<1>(uint64_t(rows) * kWG, kWG), [=](sycl::nd_item<1> it) {
        const uint64_t r = it.get_group(0);
        const uint32_t l = uint32_t(it.get_local_id(0));
        const float* xr = x + r * n;
        float s = 0.f;
        for (uint32_t i = l; i < n; i += kWG) { const float v = xr[i]; s += v * v; }
        s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
        const float inv = 1.0f / sycl::sqrt(s / float(n) + eps);
        for (uint32_t i = l; i < n; i += kWG) {
            const float v = xr[i] * inv * w[i];
            if (y)   y[r * n + i] = v;
            if (y16) y16[r * n + i] = sycl::half(v);
        }
    });
}
sycl::event lag_f16_to_f32(sycl::queue& q, const sycl::half* a, float* b, uint64_t n) {
    return q.parallel_for<laguna_k::CastHFK>(sycl::range<1>(n), [=](sycl::id<1> i) { b[i] = float(a[i]); });
}
sycl::event lag_f32_to_f16(sycl::queue& q, const float* a, sycl::half* b, uint64_t n) {
    return q.parallel_for<laguna_k::CastFHK>(sycl::range<1>(n), [=](sycl::id<1> i) { b[i] = sycl::half(a[i]); });
}
sycl::event lag_add_f32(sycl::queue& q, float* x, const float* y, uint64_t n) {
    return q.parallel_for<laguna_k::AddF32K>(sycl::range<1>(n), [=](sycl::id<1> i) { x[i] += y[i]; });
}
// x[t, i] += s[t] * o[t, i]   (o f16 / f32: a SwiGLU path's output with its per-row scale)
sycl::event lag_add_rows_scaled(sycl::queue& q, float* x, const sycl::half* o, const float* s, uint32_t T, uint32_t H) {
    return q.parallel_for<laguna_k::AddRowsHK>(sycl::range<1>(uint64_t(T) * H), [=](sycl::id<1> i) { x[i] += s[i / H] * float(o[i]); });
}
sycl::event lag_add_rows_scaled_f32(sycl::queue& q, float* x, const float* o, const float* s, uint32_t T, uint32_t H) {
    return q.parallel_for<laguna_k::AddRowsFK>(sycl::range<1>(uint64_t(T) * H), [=](sycl::id<1> i) { x[i] += s[i / H] * o[i]; });
}
// SwiGLU per row in fp32 with a power-of-two scale: h16[r] = (silu(g[r]) * u[r]) / s[r], s[r] >= 1 chosen so
// max|h16[r]| <= 4096. Ordinary rows get s = 1 exactly; the BOS row's massive activations (76,964 at layer 46 on the
// oracle prompt) would otherwise overflow f16 into inf and the next GEMV into NaN. Callers multiply back by s.
template <class In>
sycl::event lag_swiglu_rows(sycl::queue& q, const In* g, const In* u, sycl::half* h16, float* s, uint32_t R, uint32_t n) {
    return q.parallel_for<laguna_k::SwigluRowsK<In>>(sycl::nd_range<1>(uint64_t(R) * kWG, kWG), [=](sycl::nd_item<1> it) {
        const uint64_t r = it.get_group(0);
        const uint32_t l = uint32_t(it.get_local_id(0));
        const In* gr = g + r * n; const In* ur = u + r * n;
        float mx = 0.f;
        for (uint32_t i = l; i < n; i += kWG) {
            const float gv = float(gr[i]), uv = float(ur[i]);
            mx = sycl::fmax(mx, sycl::fabs(gv / (1.0f + sycl::exp(-gv)) * uv));
        }
        mx = sycl::reduce_over_group(it.get_group(), mx, sycl::maximum<float>());
        float sc = 1.0f;
        while (mx / sc > 4096.0f) sc *= 2.0f;              // power of two: the division is exact
        const float inv = 1.0f / sc;
        for (uint32_t i = l; i < n; i += kWG) {
            const float gv = float(gr[i]), uv = float(ur[i]);
            h16[r * n + i] = sycl::half(gv / (1.0f + sycl::exp(-gv)) * uv * inv);
        }
        if (l == 0) s[r] = sc;
    });
}
// dst[r, :] = src[rows[r], :]
sycl::event lag_gather_rows(sycl::queue& q, sycl::half* dst, const sycl::half* src, const int32_t* rows, uint32_t R, uint32_t H) {
    return q.parallel_for<laguna_k::GatherK>(sycl::range<1>(uint64_t(R) * H), [=](sycl::id<1> i) {
        const uint64_t r = i / H, c = i % H;
        dst[i] = src[uint64_t(rows[r]) * H + c];
    });
}
// acc[rows[r], :] += scale * w[r] * s[r] * y[r, :]   (an expert's rows are distinct: no write conflict)
sycl::event lag_scatter_add(sycl::queue& q, float* acc, const sycl::half* y, const int32_t* rows, const float* w, const float* s,
                            uint32_t R, uint32_t H, float scale) {
    return q.parallel_for<laguna_k::ScatterK>(sycl::range<1>(uint64_t(R) * H), [=](sycl::id<1> i) {
        const uint64_t r = i / H, c = i % H;
        acc[uint64_t(rows[r]) * H + c] += scale * w[r] * s[r] * float(y[i]);
    });
}
// y[t, n] = sum_k W[n, k] * x[t, k]   (W row-major [N][K], fp32 throughout: the router, as llama.cpp computes it)
sycl::event lag_gemv_f32(sycl::queue& q, const float* W, const float* x, float* y, uint32_t T, uint32_t K, uint32_t N) {
    return q.parallel_for<laguna_k::GemvF32K>(sycl::nd_range<1>(uint64_t(T) * N * kWG, kWG), [=](sycl::nd_item<1> it) {
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

// ---- decode MoE: the K picked experts in three launches (gate+up, lag_swiglu_rows, down) instead of five per expert.
// Expert k's raw GGUF slices sit in its cache slot: base + slot[k] * stride. The lane lattices are those of
// moe_decode_gate_up_silu_q4k / moe_decode_down_q6k (src/ops/moe_fused.cpp), on AoS blocks.
struct LagSlots { int32_t s[16]; };   // the picks' slots, by value: no upload
inline float lag_h2f(const uint8_t* p) { return float(sycl::bit_cast<sycl::half>(*reinterpret_cast<const uint16_t*>(p))); }
// Q4_K / Q5_K sub-block scale and min (ggml get_scale_min_k4) for sub-block j of a block's 12 scale bytes.
inline void lag_scale_min_k4(int j, const uint8_t* sc, float& s, float& m) {
    if (j < 4) { s = float(sc[j] & 63); m = float(sc[j + 4] & 63); }
    else { s = float((sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4)); m = float((sc[j + 4] >> 4) | ((sc[j] >> 6) << 4)); }
}

// g[k, n], u[k, n] (fp32) = the Q4_K gate / up rows n of expert k (gate at slot + 0, up at slot + off_up) . x[H] (f16).
sycl::event lag_moe_gate_up_q4k(sycl::queue& q, const sycl::half* x, const uint8_t* base, uint64_t stride, uint64_t off_up,
                                LagSlots slot, float* g, float* u, uint32_t H, uint32_t EF, uint32_t K) {
    constexpr int SG = 16, NW = 32, WG = NW * SG;
    const uint32_t bpc = H / 256, wgs = (EF + NW - 1) / NW;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<sycl::half, 1> A(sycl::range<1>(H), h);
        h.parallel_for<laguna_k::MoeGateUpK>(sycl::nd_range<2>({uint64_t(K), uint64_t(wgs) * WG}, {1, WG}),
                                             [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint32_t k = uint32_t(it.get_group(0)), lid = uint32_t(it.get_local_id(1));
            const uint32_t lane = lid % SG, n = uint32_t(it.get_group(1)) * NW + lid / SG;
            for (uint32_t i = lid; i < H; i += WG) A[i] = x[i];
            sycl::group_barrier(it.get_group());
            if (n >= EF) return;
            const uint8_t* gw = base + uint64_t(slot.s[k]) * stride;
            const uint8_t* uw = gw + off_up;
            const int sub = int(lane) >> 1, half = int(lane) & 1, grp = sub >> 1, hi = sub & 1;
            const int qs_off = 16 + grp * 32 + half * 16, a_off = grp * 64 + hi * 32 + half * 16, sh = hi * 4;
            float ga = 0.f, ua = 0.f;
            for (uint32_t b = 0; b < bpc; ++b) {
                const uint8_t* gb = gw + (uint64_t(n) * bpc + b) * 144;   // block_q4_K: d, dmin, scales[12], qs[128]
                const uint8_t* ub = uw + (uint64_t(n) * bpc + b) * 144;
                float gs, gm, us, um;
                lag_scale_min_k4(sub, gb + 4, gs, gm); lag_scale_min_k4(sub, ub + 4, us, um);
                const float gd = lag_h2f(gb) * gs, gmn = lag_h2f(gb + 2) * gm, ud = lag_h2f(ub) * us, umn = lag_h2f(ub + 2) * um;
                const sycl::half* a = &A[b * 256 + a_off];
                float sg_ = 0.f, su_ = 0.f, sa = 0.f;
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const float av = float(a[i]);
                    sg_ += av * float((gb[qs_off + i] >> sh) & 0xF);
                    su_ += av * float((ub[qs_off + i] >> sh) & 0xF);
                    sa  += av;
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

// acc[n] += scale * sum_k w[k] * s[k] * (down row n of expert k . h16[k])  -- QT 5 (Q5_K) or 6 (Q6_K); base points at the
// slots' down slices. h16 [K, EF] and s [K] are lag_swiglu_rows' output (s undoes its per-row scale).
template <int QT>
sycl::event lag_moe_down(sycl::queue& q, const sycl::half* h16, const float* s, const uint8_t* base, uint64_t stride,
                         LagSlots slot, const float* w, float* acc, uint32_t H, uint32_t EF, uint32_t K, float scale) {
    constexpr int SG = 16, NW = 32, WG = NW * SG;
    constexpr uint64_t BB = QT == 5 ? 176 : 210;
    const uint32_t bpc = EF / 256, wgs = (H + NW - 1) / NW;
    const uint64_t hn = uint64_t(K) * EF;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<sycl::half, 1> Hs(sycl::range<1>(hn), h);
        h.parallel_for<laguna_k::MoeDownK<QT>>(sycl::nd_range<1>(uint64_t(wgs) * WG, WG),
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
                    if constexpr (QT == 5) {   // block_q5_K: d, dmin, scales[12], qh[32], qs[128]; the Q4_K lattice + bit `sub` of qh
                        const int sub = int(lane) >> 1, half = int(lane) & 1, grp = sub >> 1, hi = sub & 1;
                        float sc, mn; lag_scale_min_k4(sub, blk + 4, sc, mn);
                        const float d = lag_h2f(blk) * sc, m = lag_h2f(blk + 2) * mn;
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
                        const float d = lag_h2f(blk + 208) * float(int8_t(blk[192 + half_q * 8 + sub * 2 + lh]));
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
            if (lane == 0) acc[n] += scale * total;
        });
    });
}

// ---- decode attention (one query row), split over the keys. One work-group per (KV head g, kAttnKS-key split s) scores
// the split's keys for ALL n_rep query heads that share g (each K/V row is read once for the group, not once per query
// head), takes the split's own softmax and P.V, and writes per head (o[hd], max, sum) to part [nh][S][hd + 2];
// lag_attn_combine merges the splits. Masked keys (mask <= -1e30) are skipped, as ink_attention skips them. ink_attention
// ran one sub-group per head walking every key in sequence: 1.33 ms per layer at 800 positions.
constexpr uint32_t kAttnKS = 64;
sycl::event lag_attn_decode(sycl::queue& q, const float* qv, const float* k, const float* v, const float* mask, float* part,
                            uint32_t nh, uint32_t nkv, uint32_t hd, uint32_t n_kv, float scale) {
    constexpr uint32_t WG = 256, SG = 16;
    const uint32_t S = (n_kv + kAttnKS - 1) / kAttnKS, n_rep = nh / nkv;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> qs(sycl::range<1>(uint64_t(n_rep) * hd), h);
        sycl::local_accessor<float, 1> sc(sycl::range<1>(uint64_t(n_rep) * kAttnKS), h);
        sycl::local_accessor<float, 1> ml(sycl::range<1>(uint64_t(n_rep) * 2), h);
        h.parallel_for<laguna_k::AttnDecK>(sycl::nd_range<2>({uint64_t(nkv), uint64_t(S) * WG}, {1, WG}),
                                           [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint32_t g = uint32_t(it.get_group(0)), s = uint32_t(it.get_group(1)), lid = uint32_t(it.get_local_id(1));
            const uint32_t j0 = s * kAttnKS, nk = sycl::min(kAttnKS, n_kv - j0);
            for (uint32_t i = lid; i < n_rep * hd; i += WG) qs[i] = qv[uint64_t(g) * n_rep * hd + i];   // heads g*n_rep.. are adjacent
            sycl::group_barrier(it.get_group());
            for (uint32_t i = lid; i < n_rep * kAttnKS; i += WG) {
                const uint32_t r = i / kAttnKS, jj = i % kAttnKS;
                float val = -INFINITY;
                if (jj < nk) {
                    const float mv = mask[j0 + jj];
                    if (mv > -1e30f) {
                        const float* kr = k + (uint64_t(j0 + jj) * nkv + g) * hd;
                        float d = 0.f;
                        for (uint32_t x = 0; x < hd; ++x) d += qs[r * hd + x] * kr[x];
                        val = d * scale + mv;
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
                const uint32_t r = i / hd, d = i % hd;
                float o = 0.f;
                for (uint32_t jj = 0; jj < nk; ++jj) {
                    const float p = sc[r * kAttnKS + jj];
                    if (p != 0.f) o += p * v[(uint64_t(j0 + jj) * nkv + g) * hd + d];
                }
                float* pp = part + (uint64_t(g * n_rep + r) * S + s) * (hd + 2);
                pp[d] = o;
                if (d == 0) { pp[hd] = ml[2 * r]; pp[hd + 1] = ml[2 * r + 1]; }
            }
        });
    });
}
// out[h][d] = sum_s e^(m_s - M) o_s[d] / sum_s e^(m_s - M) l_s over the splits of lag_attn_decode (all-masked splits: none)
sycl::event lag_attn_combine(sycl::queue& q, const float* part, float* out, uint32_t nh, uint32_t hd, uint32_t n_kv) {
    const uint32_t S = (n_kv + kAttnKS - 1) / kAttnKS;
    return q.parallel_for<laguna_k::AttnCombK>(sycl::nd_range<1>(uint64_t(nh) * hd, hd), [=](sycl::nd_item<1> it) {
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
        default: return "laguna: no host dequant for the dtype of " + std::string(t->name);
    }
}

// ggml's rope_yarn correction dim (ggml-cpu ops.cpp), float math as ggml does it, so the table matches the oracle's.
float yarn_corr_dim(int n_dims, int n_ctx_orig, float n_rot, float base) {
    return n_dims * std::log(float(n_ctx_orig) / (n_rot * 2.0f * float(M_PI))) / (2.0f * std::log(base));
}

uint64_t mem_available_bytes() {
    std::ifstream f("/proc/meminfo"); std::string k; uint64_t v = 0; std::string unit;
    while (f >> k >> v >> unit) if (k == "MemAvailable:") return v * 1024ull;
    return 0;
}
double env_f(const char* name, double def) { const char* v = std::getenv(name); return v && *v ? std::atof(v) : def; }
}  // namespace

template <class T> T* LagunaModel::dev_alloc(uint64_t n) {
    T* p = static_cast<T*>(malloc_device_checked(*q_, n * sizeof(T)));   // the VRAM page-alias self-test on the big ones
    if (p) { allocs_.push_back(p); dev_bytes_ += n * sizeof(T); }
    return p;
}

void LagunaModel::to_device(void* dst, const void* src, uint64_t n) {
    for (uint64_t off = 0; off < n; off += hstage_n_) {
        const uint64_t c = std::min(hstage_n_, n - off);
        std::memcpy(hstage_, static_cast<const uint8_t*>(src) + off, c);
        q_->memcpy(static_cast<uint8_t*>(dst) + off, hstage_, c).wait();
    }
}
void LagunaModel::from_device(void* dst, const void* src, uint64_t n) {
    for (uint64_t off = 0; off < n; off += hstage_n_) {
        const uint64_t c = std::min(hstage_n_, n - off);
        q_->memcpy(hstage_, static_cast<const uint8_t*>(src) + off, c).wait();
        std::memcpy(static_cast<uint8_t*>(dst) + off, hstage_, c);
    }
}

// " (card 2: layers 24-47)" when the layers are split over two cards, "" on one card
std::string LagunaModel::stage_tag() const {
    if (l0_ == 0 && l1_ == cfg_.n_layers) return {};
    return " (card " + std::to_string(l0_ == 0 ? 1 : 2) + ": layers " + std::to_string(l0_) + "-" + std::to_string(l1_ - 1) + ")";
}

LagunaModel::~LagunaModel() {
    if (q_) { try { q_->wait(); } catch (...) {} for (void* p : allocs_) sycl::free(p, *q_); }
}

std::string LagunaModel::upload_f32(const GgufTensorInfo* t, uint64_t n, float*& out) {
    if (!t) return "laguna: missing tensor";
    if (t->dtype != DType::kF32) return "laguna: " + std::string(t->name) + " is not F32";
    out = dev_alloc<float>(n);
    if (!out) return "laguna: device alloc failed for " + std::string(t->name);
    to_device(out, t->data, n * 4);
    return {};
}

// GGUF Q8_0 [K (ne0), N (ne1)] = N rows of K/32 blocks  ->  device SoA (gemv_q8_0_soa_f16_*): qs [N][K], d [N][K/32].
std::string LagunaModel::upload_q8_soa(const GgufTensorInfo* t, uint32_t K, uint32_t N, LagQ8& out) {
    if (!t) return "laguna: missing tensor";
    if (t->shape[0] != K || t->shape[1] != N)
        return "laguna: " + std::string(t->name) + " shape [" + std::to_string(t->shape[0]) + "," + std::to_string(t->shape[1]) +
               "] != [" + std::to_string(K) + "," + std::to_string(N) + "]";
    if (t->dtype != DType::kQ8_0 || K % 32) return "laguna: " + std::string(t->name) + " must be Q8_0 with K % 32 == 0";
    const uint32_t bpc = K / 32;
    std::vector<int8_t> qs(uint64_t(N) * K);
    std::vector<uint16_t> d(uint64_t(N) * bpc);
    const uint32_t nt = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    std::vector<std::thread> ts;
    for (uint32_t w = 0; w < nt; ++w) ts.emplace_back([&, w] {
        for (uint64_t n = w; n < N; n += nt)
            for (uint32_t b = 0; b < bpc; ++b) {
                const uint8_t* blk = t->data + (n * bpc + b) * 34;   // block_q8_0: fp16 d, int8 qs[32]
                std::memcpy(&d[n * bpc + b], blk, 2);
                std::memcpy(&qs[n * K + uint64_t(b) * 32], blk + 2, 32);
            }
    });
    for (auto& th : ts) th.join();
    out.qs = dev_alloc<int8_t>(qs.size());
    out.d = dev_alloc<uint16_t>(d.size());
    if (!out.qs || !out.d) return "laguna: device alloc failed for " + std::string(t->name);
    to_device(out.qs, qs.data(), qs.size());
    to_device(out.d, d.data(), d.size() * 2);
    wscratch_n_ = std::max(wscratch_n_, uint64_t(K) * N);
    return {};
}

// Prefill GEMM over a Q8_0 projection: C[T, N] (fp32) = A[T, K] (f16) x W, W dequantized into the F16 scratch first
// (the in-order queue finishes one projection's GEMM before the next dequant overwrites the scratch).
void LagunaModel::gemm_q8(const sycl::half* A, const LagQ8& W, float* C, uint32_t T, uint32_t K, uint32_t N) {
    dequant_q8_soa_f16(*q_, W.qs, W.d, wscratch_, K, N);
    gemm_fp16(*q_, A, wscratch_, C, T, N, K);
}

std::string LagunaModel::load(sycl::queue& q, const GgufReader& g, const LagunaConfig& cfg, sycl::queue* q2) {
    if (!q2) return load_stage(q, g, cfg, 0, cfg.n_layers);
    // two cards: the dense layers and half the MoE layers on the first, the other half and the head on the second
    const uint32_t split = cfg.dense_layers + (cfg.n_layers - cfg.dense_layers) / 2;
    if (split == 0 || split >= cfg.n_layers) return "laguna: no two-card split for " + std::to_string(cfg.n_layers) + " layers";
    if (auto e = load_stage(q, g, cfg, 0, split); !e.empty()) return e;
    next_ = std::make_unique<LagunaModel>();
    return next_->load_stage(*q2, g, cfg, split, cfg.n_layers);
}

std::string LagunaModel::load_stage(sycl::queue& q, const GgufReader& g, const LagunaConfig& cfg, uint32_t l0, uint32_t l1) {
    if (!q.is_in_order()) return "laguna: the queue must be in-order";
    q_ = &q; g_ = &g; cfg_ = cfg; l0_ = l0; l1_ = l1;
    hstage_n_ = 64ull << 20;
    hstage_ = sycl::malloc_host<uint8_t>(hstage_n_, q);
    if (!hstage_) return "laguna: pinned staging alloc failed";
    allocs_.push_back(hstage_);
    const char* fm = std::getenv("IE_LAGUNA_FUSED_MOE");   // =0: decode runs the expert-major loop (five launches per expert)
    fused_moe_ = !(fm && std::string(fm) == "0") && cfg.n_experts_used <= 16;
    const uint32_t H = cfg.hidden, hd = cfg.head_dim, nkv = cfg.n_kv_heads, E = cfg.n_experts;
    if (cfg.head_dim != cfg.v_head_dim) return "laguna: key/value head dims differ";
    auto T = [&](const std::string& n) { return g.find_tensor(n); };
    if (l0_ == 0) {                           // the first card embeds
        tok_embd_ = T("token_embd.weight");
        if (!tok_embd_ || tok_embd_->dtype != DType::kQ8_0 || tok_embd_->shape[0] != H) return "laguna: token_embd must be Q8_0 [hidden, vocab]";
    }
    if (l1_ == cfg.n_layers) {                // the last card runs the head
        if (auto e = upload_f32(T("output_norm.weight"), H, out_norm_); !e.empty()) return e;
        const auto* o = T("output.weight");
        if (!o || o->dtype != DType::kQ6_K || o->shape[0] != H || o->shape[1] != cfg.vocab) return "laguna: output must be Q6_K [hidden, vocab]";
        head_q6_ = dev_alloc<uint8_t>(o->nbytes);
        if (!head_q6_) return "laguna: head alloc failed";
        to_device(head_q6_, o->data, o->nbytes);
    }
    layers_.resize(cfg.n_layers);             // indexed by layer; only this card's [l0_, l1_) are loaded
    for (uint32_t L = l0_; L < l1_; ++L) {
        LagunaLayer& w = layers_[L];
        const std::string b = "blk." + std::to_string(L) + ".";
        w.n_heads = cfg.heads_for(L); w.sliding = cfg.is_sliding(L);
        w.rope_dim = cfg.rope_dim_for(L); w.rope_theta = cfg.rope_theta_for(L);
        w.moe = L >= cfg.dense_layers;
        max_heads_ = std::max(max_heads_, w.n_heads);
        const uint32_t qd = w.n_heads * hd;
        std::string e;
        if (e = upload_f32(T(b + "attn_norm.weight"), H, w.attn_norm); !e.empty()) return e;
        if (e = upload_f32(T(b + "ffn_norm.weight"), H, w.ffn_norm); !e.empty()) return e;
        if (e = upload_f32(T(b + "attn_q_norm.weight"), hd, w.q_norm); !e.empty()) return e;
        if (e = upload_f32(T(b + "attn_k_norm.weight"), hd, w.k_norm); !e.empty()) return e;
        if (e = upload_q8_soa(T(b + "attn_q.weight"), H, qd, w.wq); !e.empty()) return e;
        if (e = upload_q8_soa(T(b + "attn_k.weight"), H, nkv * hd, w.wk); !e.empty()) return e;
        if (e = upload_q8_soa(T(b + "attn_v.weight"), H, nkv * hd, w.wv); !e.empty()) return e;
        if (e = upload_q8_soa(T(b + "attn_output.weight"), qd, H, w.wo); !e.empty()) return e;
        if (e = upload_q8_soa(T(b + "attn_gate.weight"), H, w.n_heads, w.wg); !e.empty()) return e;   // per-head gate (XS.2/S.2)
        if (!w.moe) {
            if (e = upload_q8_soa(T(b + "ffn_gate.weight"), H, cfg.ffn_dense, w.ffn_gate); !e.empty()) return e;
            if (e = upload_q8_soa(T(b + "ffn_up.weight"), H, cfg.ffn_dense, w.ffn_up); !e.empty()) return e;
            if (e = upload_q8_soa(T(b + "ffn_down.weight"), cfg.ffn_dense, H, w.ffn_down); !e.empty()) return e;
        } else {
            if (e = upload_f32(T(b + "ffn_gate_inp.weight"), uint64_t(H) * E, w.router); !e.empty()) return e;
            if (e = upload_f32(T(b + "exp_probs_b.bias"), E, w.router_bias); !e.empty()) return e;
            const uint32_t S = cfg.shared_expert_ffn;
            if (e = upload_q8_soa(T(b + "ffn_gate_shexp.weight"), H, S, w.sh_gate); !e.empty()) return e;
            if (e = upload_q8_soa(T(b + "ffn_up_shexp.weight"), H, S, w.sh_up); !e.empty()) return e;
            if (e = upload_q8_soa(T(b + "ffn_down_shexp.weight"), S, H, w.sh_down); !e.empty()) return e;
            w.gate_exps = T(b + "ffn_gate_exps.weight"); w.up_exps = T(b + "ffn_up_exps.weight"); w.down_exps = T(b + "ffn_down_exps.weight");
            if (!w.gate_exps || !w.up_exps || !w.down_exps) return "laguna: layer " + std::to_string(L) + " misses an expert bank";
            if (w.gate_exps->dtype != DType::kQ4_K || w.up_exps->dtype != DType::kQ4_K)
                return "laguna: layer " + std::to_string(L) + ": gate/up experts are not Q4_K (no binding for that dtype yet)";
            if (w.down_exps->dtype != DType::kQ5_K && w.down_exps->dtype != DType::kQ6_K)
                return "laguna: layer " + std::to_string(L) + ": down experts are neither Q5_K nor Q6_K";
            w.down_q5 = w.down_exps->dtype == DType::kQ5_K;
            w.gate_slice = w.gate_exps->nbytes / E; w.up_slice = w.up_exps->nbytes / E; w.down_slice = w.down_exps->nbytes / E;
            w.gate_src = w.gate_exps->data; w.up_src = w.up_exps->data; w.down_src = w.down_exps->data;
        }
    }
    wscratch_ = dev_alloc<sycl::half>(wscratch_n_);
    if (!wscratch_) return "laguna: prefill F16 scratch alloc failed";
    return pin_experts();
}

// The routed experts in pinned host memory, so a cache miss is a DMA from pinned pages rather than a copy through
// pageable ones. Refused (the mapping stays the source) when it would leave less than IE_LAGUNA_RAM_FLOOR_GB (30)
// available, or with IE_LAGUNA_PIN=0. The copy runs on 16 threads from the page cache.
std::string LagunaModel::pin_experts() {
    if (const char* v = std::getenv("IE_LAGUNA_PIN"); v && *v == '0') { std::printf("[laguna] experts: unpinned (IE_LAGUNA_PIN=0)\n"); return {}; }
    uint64_t need = 0;
    for (const auto& w : layers_) if (w.moe) need += w.gate_exps->nbytes + w.up_exps->nbytes + w.down_exps->nbytes;
    const uint64_t avail = mem_available_bytes(), floor_b = uint64_t(env_f("IE_LAGUNA_RAM_FLOOR_GB", 30.0) * 1073741824.0);
    if (avail < need + floor_b) {
        std::printf("[laguna] experts: NOT pinned (%.1f GiB needed, %.1f GiB available, floor %.0f GiB) -- misses read the mapping%s\n",
                    need / 1073741824.0, avail / 1073741824.0, floor_b / 1073741824.0, stage_tag().c_str());
        return {};
    }
    for (auto& w : layers_) {
        if (!w.moe) continue;
        for (const auto& bank : {std::pair<const GgufTensorInfo*, const uint8_t**>{w.gate_exps, &w.gate_src},
                                 {w.up_exps, &w.up_src}, {w.down_exps, &w.down_src}}) {
            const GgufTensorInfo* t = bank.first; const uint8_t** src = bank.second;
            uint8_t* p = sycl::malloc_host<uint8_t>(t->nbytes, *q_);
            if (!p) return "laguna: pinned alloc failed (" + std::to_string(t->nbytes >> 20) + " MiB)";
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
    std::printf("[laguna] experts: %.1f GiB pinned in host memory%s\n", pinned_bytes_ / 1073741824.0, stage_tag().c_str());
    return {};
}

std::string LagunaModel::init_runtime(uint32_t max_ctx, uint32_t max_chunk) {
    const uint32_t H = cfg_.hidden, hd = cfg_.head_dim, nkv = cfg_.n_kv_heads, E = cfg_.n_experts;
    const uint32_t K = cfg_.n_experts_used, F = std::max(cfg_.ffn_dense, cfg_.shared_expert_ffn), EF = cfg_.expert_ffn;
    max_ctx_ = max_ctx; max_chunk_ = max_chunk;
    // gemm_fp16 stores whole 8x16 tiles (its contract): rows are padded to a multiple of 8 so a ragged T never spills
    const uint64_t C = (uint64_t(max_chunk) + 7) / 8 * 8, qd = uint64_t(max_heads_) * hd;
    ring_ = std::min<uint32_t>(max_ctx, cfg_.sliding_window + uint32_t(C));
    ring_pos_.assign(ring_, -1);
    kc_.assign(cfg_.n_layers, nullptr); vc_.assign(cfg_.n_layers, nullptr);
    for (uint32_t L = l0_; L < l1_; ++L) {
        const uint64_t rows = layers_[L].sliding ? ring_ : max_ctx;
        kc_[L] = dev_alloc<float>(rows * nkv * hd);
        vc_[L] = dev_alloc<float>(rows * nkv * hd);
        if (!kc_[L] || !vc_[L]) return "laguna: KV alloc failed at layer " + std::to_string(L) + " (" + std::to_string(rows) + " positions)";
    }
    x_ = dev_alloc<float>(C * H); xn_ = dev_alloc<float>(C * H); acc_ = dev_alloc<float>(C * H); fy_ = dev_alloc<float>(C * H);
    qf_ = dev_alloc<float>(C * qd); kf_ = dev_alloc<float>(C * nkv * hd); vf_ = dev_alloc<float>(C * nkv * hd);
    gf_ = dev_alloc<float>(C * max_heads_); attn_ = dev_alloc<float>(C * qd);
    fg_ = dev_alloc<float>(C * F); fu_ = dev_alloc<float>(C * F); fh16_ = dev_alloc<sycl::half>(C * F);
    rlog_ = dev_alloc<float>(C * E); rw_ = dev_alloc<float>(C * K); ridx_ = dev_alloc<int32_t>(C * K);
    d_rows_ = dev_alloc<int32_t>(C * K); d_rw_ = dev_alloc<float>(C * K); sc_ = dev_alloc<float>(C); gs_ = dev_alloc<float>(C);
    cos_full_ = dev_alloc<float>(C * hd); sin_full_ = dev_alloc<float>(C * hd);
    cos_swa_ = dev_alloc<float>(C * hd); sin_swa_ = dev_alloc<float>(C * hd);
    mask_full_ = dev_alloc<float>(C * max_ctx); mask_swa_ = dev_alloc<float>(C * std::max<uint64_t>(1, ring_));
    xn16_ = dev_alloc<sycl::half>(C * H); q16_ = dev_alloc<sycl::half>(C * qd);
    k16_ = dev_alloc<sycl::half>(C * nkv * hd); v16_ = dev_alloc<sycl::half>(C * nkv * hd); g16_ = dev_alloc<sycl::half>(C * max_heads_);
    a16_ = dev_alloc<sycl::half>(C * qd); o16_ = dev_alloc<sycl::half>(C * H);
    f1_ = dev_alloc<sycl::half>(F); f2_ = dev_alloc<sycl::half>(F); f3_ = dev_alloc<sycl::half>(F);
    gx_ = dev_alloc<sycl::half>(C * H); gg_ = dev_alloc<sycl::half>(C * EF); gu_ = dev_alloc<sycl::half>(C * EF);
    gh_ = dev_alloc<sycl::half>(C * EF); gy_ = dev_alloc<sycl::half>(C * H);
    logits16_ = dev_alloc<sycl::half>(cfg_.vocab);
    attn_part_ = dev_alloc<float>(uint64_t(max_heads_) * ((std::max(max_ctx, ring_) + kAttnKS - 1) / kAttnKS) * (hd + 2));
    for (void* p : {(void*)x_, (void*)xn_, (void*)acc_, (void*)fy_, (void*)qf_, (void*)kf_, (void*)vf_, (void*)gf_, (void*)attn_,
                    (void*)fg_, (void*)fu_, (void*)fh16_, (void*)rlog_, (void*)rw_, (void*)ridx_, (void*)d_rows_, (void*)d_rw_,
                    (void*)sc_, (void*)gs_, (void*)cos_full_, (void*)sin_full_, (void*)cos_swa_, (void*)sin_swa_, (void*)mask_full_,
                    (void*)mask_swa_, (void*)xn16_, (void*)q16_, (void*)k16_, (void*)v16_, (void*)g16_, (void*)a16_, (void*)o16_,
                    (void*)f1_, (void*)f2_, (void*)f3_, (void*)gx_, (void*)gg_, (void*)gu_, (void*)gh_, (void*)gy_, (void*)logits16_,
                    (void*)attn_part_})
        if (!p) return "laguna: workspace alloc failed";
    if (next_ || l0_ > 0) {                   // two cards: the residual's pinned hand-off buffer (one per card's context)
        hx_ = sycl::malloc_host<float>(C * H, *q_);
        if (!hx_) return "laguna: pinned hand-off buffer alloc failed";
        allocs_.push_back(hx_);
    }
    // ---- the expert cache: the card's free memory minus the reserve, split evenly over the MoE layers ----
    uint64_t gmax = 0, umax = 0, dmax = 0; uint32_t n_moe = 0;
    for (const auto& w : layers_) if (w.moe) { ++n_moe; gmax = std::max(gmax, w.gate_slice); umax = std::max(umax, w.up_slice); dmax = std::max(dmax, w.down_slice); }
    auto align = [](uint64_t v) { return (v + 255) / 256 * 256; };
    off_up_ = align(gmax); off_down_ = off_up_ + align(umax); slot_bytes_ = off_down_ + align(dmax);
    const auto& dev = q_->get_device();
    uint64_t free_b = dev.has(sycl::aspect::ext_intel_free_memory) ? dev.get_info<sycl::ext::intel::info::device::free_memory>() : 0;
    if (!free_b) { const uint64_t total = dev.get_info<sycl::info::device::global_mem_size>(); free_b = total > dev_bytes_ ? total - dev_bytes_ : 0; }
    const uint64_t reserve = uint64_t(env_f("IE_LAGUNA_RESERVE_GB", 1.5) * 1073741824.0);
    uint64_t budget = free_b > reserve ? free_b - reserve : 0;
    if (const char* v = std::getenv("IE_LAGUNA_ECACHE_GB"); v && *v) budget = std::min<uint64_t>(budget, uint64_t(std::atof(v) * 1073741824.0));
    ec_slots_ = n_moe ? uint32_t(std::min<uint64_t>(E, budget / (uint64_t(n_moe) * slot_bytes_))) : 0;
    if (ec_slots_ < 12) return "laguna: room for only " + std::to_string(ec_slots_) + " expert slots per layer (need >= 12: a decode step selects 10)";
    ec_.assign(cfg_.n_layers, {});
    for (uint32_t L = 0; L < cfg_.n_layers; ++L) {
        if (!layers_[L].moe) continue;
        LagunaECache& c = ec_[L];
        c.base = dev_alloc<uint8_t>(uint64_t(ec_slots_) * slot_bytes_);
        if (!c.base) return "laguna: expert-cache alloc failed at layer " + std::to_string(L);
        c.slot_of.assign(E, -1); c.expert_of.assign(ec_slots_, -1); c.stamp.assign(ec_slots_, 0);
    }
    std::printf("[laguna] expert cache: %u slots per MoE layer (%.0f%% of %u experts), %.2f GiB; %.2f GiB on the card in all%s\n",
                ec_slots_, 100.0 * ec_slots_ / E, E, uint64_t(ec_slots_) * slot_bytes_ * n_moe / 1073741824.0, dev_bytes_ / 1073741824.0,
                stage_tag().c_str());
    pos_ = 0;
    return next_ ? next_->init_runtime(max_ctx, max_chunk) : std::string();
}

// The slot holding expert e of layer L, filling it on a miss (LRU victim). The queue is in-order, so a slot reused
// inside one layer pass is only overwritten after the kernels that read its previous expert.
const uint8_t* LagunaModel::acquire(const LagunaLayer& w, uint32_t L, int32_t e) {
    LagunaECache& c = ec_[L];
    int32_t s = c.slot_of[e];
    if (s >= 0) { ++ec_hits_; c.stamp[s] = ++ec_clock_; return c.base + uint64_t(s) * slot_bytes_; }
    ++ec_misses_;
    s = int32_t(std::min_element(c.stamp.begin(), c.stamp.end()) - c.stamp.begin());
    if (c.expert_of[s] >= 0) c.slot_of[c.expert_of[s]] = -1;
    c.expert_of[s] = int16_t(e); c.slot_of[e] = int16_t(s); c.stamp[s] = ++ec_clock_;
    uint8_t* p = c.base + uint64_t(s) * slot_bytes_;
    q_->memcpy(p,             w.gate_src + uint64_t(e) * w.gate_slice, w.gate_slice);
    q_->memcpy(p + off_up_,   w.up_src   + uint64_t(e) * w.up_slice,   w.up_slice);
    q_->memcpy(p + off_down_, w.down_src + uint64_t(e) * w.down_slice, w.down_slice);
    return p;
}

// A SwiGLU FFN over T rows (layer 0's dense FFN, a shared expert): dst[t] += s[t] * down(swiglu(gate x, up x) / s[t]).
std::string LagunaModel::ffn_dense(const LagQ8& gate, const LagQ8& up, const LagQ8& down, uint32_t F, uint32_t T, float* dst) {
    sycl::queue& q = *q_;
    const uint32_t H = cfg_.hidden;
    if (T == 1) {
        gemv_q8_0_soa_f16_rows_dual(q, xn16_, gate.qs, gate.d, f1_, up.qs, up.d, f2_, H, F, 1);
        lag_swiglu_rows<sycl::half>(q, f1_, f2_, f3_, sc_, 1, F);
        gemv_q8_0_soa_f16_g(q, f3_, down.qs, down.d, o16_, F, H);
        lag_add_rows_scaled(q, dst, o16_, sc_, 1, H);
    } else {
        gemm_q8(xn16_, gate, fg_, T, H, F);
        gemm_q8(xn16_, up, fu_, T, H, F);
        lag_swiglu_rows<float>(q, fg_, fu_, fh16_, sc_, T, F);
        gemm_q8(fh16_, down, fy_, T, F, H);
        lag_add_rows_scaled_f32(q, dst, fy_, sc_, T, H);
    }
    return {};
}

// The routed experts, expert-major: rows grouped by expert, each expert acquired once and run over its rows with the
// multi-row kernels, its outputs scattered into acc_ with weight * expert_weights_scale (* the row's SwiGLU scale).
std::string LagunaModel::moe_experts(const LagunaLayer& w, uint32_t L, uint32_t T) {
    sycl::queue& q = *q_;
    const uint32_t H = cfg_.hidden, EF = cfg_.expert_ffn, E = cfg_.n_experts, K = cfg_.n_experts_used;
    if (T == 1 && fused_moe_) {   // decode: the K picks (distinct experts, distinct slots) in three launches
        LagSlots sl{};
        for (uint32_t k = 0; k < K; ++k) {
            const int32_t e = h_idx_[k];
            if (e < 0 || uint32_t(e) >= E) return "laguna: router picked expert " + std::to_string(e);
            sl.s[k] = int32_t((acquire(w, L, e) - ec_[L].base) / slot_bytes_);
        }
        const uint8_t* base = ec_[L].base;
        lag_moe_gate_up_q4k(q, xn16_, base, slot_bytes_, off_up_, sl, fg_, fu_, H, EF, K);
        lag_swiglu_rows<float>(q, fg_, fu_, gh_, gs_, K, EF);
        if (w.down_q5) lag_moe_down<5>(q, gh_, gs_, base + off_down_, slot_bytes_, sl, rw_, acc_, H, EF, K, cfg_.expert_weights_scale);
        else           lag_moe_down<6>(q, gh_, gs_, base + off_down_, slot_bytes_, sl, rw_, acc_, H, EF, K, cfg_.expert_weights_scale);
        return {};
    }
    std::vector<uint32_t> cnt(E + 1, 0);
    for (uint64_t i = 0; i < uint64_t(T) * K; ++i) {
        const int32_t e = h_idx_[i];
        if (e < 0 || uint32_t(e) >= E) return "laguna: router picked expert " + std::to_string(e);
        ++cnt[uint32_t(e) + 1];
    }
    for (uint32_t e = 0; e < E; ++e) cnt[e + 1] += cnt[e];             // offsets
    h_rows_.assign(uint64_t(T) * K, 0); h_rw_.assign(uint64_t(T) * K, 0.f);
    std::vector<uint32_t> fill(cnt.begin(), cnt.end() - 1);
    for (uint32_t t = 0; t < T; ++t)
        for (uint32_t k = 0; k < K; ++k) {
            const uint32_t e = uint32_t(h_idx_[uint64_t(t) * K + k]), at = fill[e]++;
            h_rows_[at] = int32_t(t); h_rw_[at] = h_w_[uint64_t(t) * K + k];
        }
    to_device(d_rows_, h_rows_.data(), h_rows_.size() * 4);
    to_device(d_rw_, h_rw_.data(), h_rw_.size() * 4);
    const float scale = cfg_.expert_weights_scale;
    // Least-picked first: the LRU then ends the chunk holding the experts this chunk used MOST (what decode is likely to
    // pick next), not the highest-numbered ones an index-order walk leaves behind.
    std::vector<uint32_t> order(E);
    std::iota(order.begin(), order.end(), 0u);
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return cnt[a + 1] - cnt[a] < cnt[b + 1] - cnt[b]; });
    for (const uint32_t e : order) {
        const uint32_t off = cnt[e], R = cnt[e + 1] - cnt[e];
        if (!R) continue;
        const uint8_t* slot = acquire(w, L, int32_t(e));
        const uint8_t *sg = slot, *su = slot + off_up_, *sd = slot + off_down_;
        const sycl::half* A = xn16_ + uint64_t(h_rows_[off]) * H;          // one row: read it in place
        if (R > 1) { lag_gather_rows(q, gx_, xn16_, d_rows_ + off, R, H); A = gx_; }
        if (R == 1) { gemv_q4_K(q, A, sg, gg_, H, EF); gemv_q4_K(q, A, su, gu_, H, EF); }
        else        { gemv_q4_K_rows(q, A, sg, gg_, H, EF, R); gemv_q4_K_rows(q, A, su, gu_, H, EF, R); }
        lag_swiglu_rows<sycl::half>(q, gg_, gu_, gh_, gs_, R, EF);
        if (w.down_q5) {
            if (R == 1) gemv_q5_K(q, gh_, sd, gy_, EF, H); else gemv_q5_K_rows(q, gh_, sd, gy_, EF, H, R);
        } else if (R == 1) {
            gemv_q6_K(q, gh_, sd, gy_, EF, H);
        } else {
            for (uint32_t r0 = 0; r0 < R; r0 += 8)                         // gemm_q6_K takes at most 8 rows (more: it does NOTHING)
                gemm_q6_K(q, gh_ + uint64_t(r0) * EF, sd, gy_ + uint64_t(r0) * H, std::min(8u, R - r0), EF, H);
        }
        lag_scatter_add(q, acc_, gy_, d_rows_ + off, d_rw_ + off, gs_, R, H, scale);
    }
    return {};
}

bool LagunaModel::rewind(uint32_t pos) {
    if (next_ && !next_->rewind(pos)) return false;   // the cards hold the same positions: the same answer
    if (pos > pos_) return false;
    if (pos == 0) { pos_ = 0; std::fill(ring_pos_.begin(), ring_pos_.end(), -1); return true; }
    const uint32_t W = cfg_.sliding_window, lo = pos > W - 1 ? pos - (W - 1) : 0;   // the keys position `pos` will still attend to
    for (uint32_t p = lo; p < pos; ++p) if (ring_pos_[p % ring_] != int64_t(p)) return false;
    for (auto& v : ring_pos_) if (v >= int64_t(pos)) v = -1;            // positions past the rewind point are gone
    pos_ = pos;
    return true;
}

std::string LagunaModel::forward(const int32_t* ids, uint32_t T, uint32_t pos0, float* logits_host) {
    if (!x_) return "laguna: init_runtime first";
    if (T == 0 || T > max_chunk_) return "laguna: T out of range";
    if (pos0 != 0 && pos0 != pos_) return "laguna: forward at " + std::to_string(pos0) + " but the sequence is at " + std::to_string(pos_);
    if (uint64_t(pos0) + T > max_ctx_) return "laguna: context full";
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

std::string LagunaModel::prefill(const int32_t* ids, uint32_t n, uint32_t pos0, float* logits_host,
                                 const std::function<bool(uint32_t)>& after_chunk) {
    if (n == 0) return "laguna: empty prefill";
    static const bool overlap = !(std::getenv("IE_LAGUNA_OVERLAP") && *std::getenv("IE_LAGUNA_OVERLAP") == '0');
    if (!next_ || n <= max_chunk_ || !overlap) {   // one card, one chunk, or IE_LAGUNA_OVERLAP=0: one chunk at a time
        for (uint32_t off = 0; off < n; off += max_chunk_) {
            const uint32_t T = std::min(max_chunk_, n - off);
            const bool last = off + T == n;
            if (auto e = forward(ids + off, T, pos0 + off, last ? logits_host : nullptr); !e.empty()) return e;
            if (!last && after_chunk && !after_chunk(pos0 + off + T)) return {};
        }
        return {};
    }
    if (!x_) return "laguna: init_runtime first";
    if (pos0 != 0 && pos0 != pos_) return "laguna: prefill at " + std::to_string(pos0) + " but the sequence is at " + std::to_string(pos_);
    if (uint64_t(pos0) + n > max_ctx_) return "laguna: context full";
    // card 2 runs chunk k on its own thread while this thread runs chunk k+1 on card 1; the cards share no state
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
// head (last card, when logits_host is given).
std::string LagunaModel::stage_forward(const int32_t* ids, uint32_t T, uint32_t pos0, float* logits_host) {
    if (pos0 == 0) std::fill(ring_pos_.begin(), ring_pos_.end(), -1);
    sycl::queue& q = *q_;
    const uint32_t H = cfg_.hidden, hd = cfg_.head_dim, nkv = cfg_.n_kv_heads, E = cfg_.n_experts, K = cfg_.n_experts_used;
    const uint32_t n_kv = pos0 + T;
    const float eps = cfg_.rms_eps, kq_scale = 1.0f / std::sqrt(float(hd));
    // IE_LAGUNA_DUMP=1: layer 0's stages, named as llama.cpp's graph names them (sum | first row [0..2] | last row [0..2]).
    static const bool dump = std::getenv("IE_LAGUNA_DUMP") != nullptr;
    auto dumpf = [&](const char* name, const float* d, uint64_t rows, uint64_t cols) {
        q.wait(); std::vector<float> h(rows * cols); q.memcpy(h.data(), d, h.size() * 4).wait();
        double s = 0; for (float v : h) s += v;
        const float* f = h.data(); const float* l = h.data() + (rows - 1) * cols;
        std::printf("[dump] %-22s sum %+12.5f  first-row %+.4f %+.4f %+.4f  last-row %+.4f %+.4f %+.4f\n", name, s, f[0], f[1], f[2], l[0], l[1], l[2]);
    };

    // ---- embeddings (host dequant of the rows) ----
    if (l0_ == 0) {
        h_emb_.resize(uint64_t(T) * H);
        for (uint32_t t = 0; t < T; ++t) {
            if (ids[t] < 0 || uint32_t(ids[t]) >= cfg_.vocab) return "laguna: token id out of range";
            if (auto e = dequant_rows(tok_embd_, uint64_t(ids[t]), 1, H, h_emb_.data() + uint64_t(t) * H); !e.empty()) return e;
        }
        to_device(x_, h_emb_.data(), h_emb_.size() * 4);
        if (dump) dumpf("embd", x_, T, H);
    }

    // ---- rope tables (ggml rope_yarn, NEOX pairing (d, d + n_dims/2)) ----
    auto build_rope = [&](uint32_t n_dims, float base, float freq_scale, float ext_factor, float beta_fast, float beta_slow,
                          uint32_t n_ctx_orig, float* dcos, float* dsin) {
        h_cos_.assign(uint64_t(T) * n_dims, 0.f); h_sin_.assign(uint64_t(T) * n_dims, 0.f);
        float corr0 = 0.f, corr1 = 0.f;
        if (ext_factor != 0.f) {
            corr0 = std::max(0.0f, std::floor(yarn_corr_dim(int(n_dims), int(n_ctx_orig), beta_fast, base)));
            corr1 = std::min(float(n_dims - 1), std::ceil(yarn_corr_dim(int(n_dims), int(n_ctx_orig), beta_slow, base)));
        }
        const float theta_scale = std::pow(base, -2.0f / float(n_dims));
        // YaRN magnitude scaling, exactly as llama.cpp ends up applying it: llama_context sets yarn_attn_factor to
        // get_mscale(factor) / (1 + 0.1 ln factor) = 1 (x rope.scaling.attn_factor, absent = 1) and ggml's rope_yarn
        // multiplies by 1 + 0.1 ln(1/freq_scale), so the rotated dims of the YaRN layers carry 1 + 0.1 ln(factor)
        // (1.3466 at factor 32; measured against the oracle: Q at position 0 was 0.74x without it). Plain rope: 1.
        const float m = ext_factor != 0.f ? 1.0f + 0.1f * std::log(1.0f / freq_scale) : 1.0f;
        const uint32_t half = n_dims / 2;
        for (uint32_t t = 0; t < T; ++t) {
            float theta = float(pos0 + t);
            for (uint32_t i0 = 0; i0 < n_dims; i0 += 2) {
                const float theta_extrap = theta, theta_interp = freq_scale * theta_extrap;
                float th = theta_interp;
                if (ext_factor != 0.f) {
                    const float y = (float(i0 / 2) - corr0) / std::max(0.001f, corr1 - corr0);
                    const float ramp_mix = (1.0f - std::min(1.0f, std::max(0.0f, y))) * ext_factor;
                    th = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
                }
                const float c = std::cos(th) * m, s = std::sin(th) * m;
                const uint32_t i = i0 / 2;
                h_cos_[uint64_t(t) * n_dims + i] = c; h_cos_[uint64_t(t) * n_dims + i + half] = c;
                h_sin_[uint64_t(t) * n_dims + i] = s; h_sin_[uint64_t(t) * n_dims + i + half] = s;
                theta *= theta_scale;
            }
        }
        to_device(dcos, h_cos_.data(), h_cos_.size() * 4);
        to_device(dsin, h_sin_.data(), h_sin_.size() * 4);
        q.wait();
    };
    build_rope(cfg_.rope_dim_full, cfg_.rope_theta_full, 1.0f / cfg_.yarn_factor, 1.0f, cfg_.yarn_beta_fast, cfg_.yarn_beta_slow,
               cfg_.yarn_orig_ctx, cos_full_, sin_full_);
    build_rope(cfg_.rope_dim_swa, cfg_.rope_theta_swa, 1.0f, 0.0f, 0.f, 0.f, cfg_.ctx_train, cos_swa_, sin_swa_);

    // ---- masks. Full layers [T, n_kv]: causal. Sliding layers [T, ring_]: the ring slot j holds position ring_pos_[j];
    // a query at p sees it iff p_k <= p and p - p_k < window (llama.cpp SWA_STANDARD). Attention sums over keys, so the
    // ring's order does not matter -- only which positions the mask admits.
    for (uint32_t t = 0; t < T; ++t) ring_pos_[(pos0 + t) % ring_] = int64_t(pos0 + t);
    h_mask_.assign(uint64_t(T) * n_kv, 0.f);
    for (uint32_t t = 0; t < T; ++t)
        for (uint32_t j = pos0 + t + 1; j < n_kv; ++j) h_mask_[uint64_t(t) * n_kv + j] = -INFINITY;
    to_device(mask_full_, h_mask_.data(), h_mask_.size() * 4);
    h_mask_.assign(uint64_t(T) * ring_, -INFINITY);
    for (uint32_t t = 0; t < T; ++t) {
        const int64_t pq = int64_t(pos0 + t);
        for (uint32_t j = 0; j < ring_; ++j) {
            const int64_t pk = ring_pos_[j];
            if (pk >= 0 && pk <= pq && pq - pk < int64_t(cfg_.sliding_window)) h_mask_[uint64_t(t) * ring_ + j] = 0.f;
        }
    }
    to_device(mask_swa_, h_mask_.data(), h_mask_.size() * 4);

    h_idx_.resize(uint64_t(T) * K); h_w_.resize(uint64_t(T) * K);
    for (uint32_t L = l0_; L < l1_; ++L) {
        const LagunaLayer& w = layers_[L];
        const uint32_t nh = w.n_heads, qd = nh * hd, kvd = nkv * hd;
        const bool D = dump && L == 0;
        // ---- attention ----
        lag_rmsnorm(q, x_, w.attn_norm, xn_, xn16_, T, H, eps);
        if (D) dumpf("attn_norm-0", xn_, T, H);
        if (T == 1) {
            gemv_q8_0_soa_f16_g(q, xn16_, w.wq.qs, w.wq.d, q16_, H, qd);
            gemv_q8_0_soa_f16_g(q, xn16_, w.wk.qs, w.wk.d, k16_, H, kvd);
            gemv_q8_0_soa_f16_g(q, xn16_, w.wv.qs, w.wv.d, v16_, H, kvd);
            lag_f16_to_f32(q, q16_, qf_, qd);
            lag_f16_to_f32(q, k16_, kf_, kvd);
            lag_f16_to_f32(q, v16_, vf_, kvd);
        } else {                                                              // batched, fp32 out (N multiples of 16)
            gemm_q8(xn16_, w.wq, qf_, T, H, qd);
            gemm_q8(xn16_, w.wk, kf_, T, H, kvd);
            gemm_q8(xn16_, w.wv, vf_, T, H, kvd);
        }
        // the gate's width (48 / 72) is not a whole 16-column tile (gemm_fp16 would spill into the next row): the rows GEMV
        gemv_q8_0_soa_f16_rows(q, xn16_, w.wg.qs, w.wg.d, g16_, H, nh, T);
        lag_f16_to_f32(q, g16_, gf_, uint64_t(T) * nh);
        lag_rmsnorm(q, qf_, w.q_norm, qf_, nullptr, T * nh, hd, eps);
        lag_rmsnorm(q, kf_, w.k_norm, kf_, nullptr, T * nkv, hd, eps);
        const float* cs = w.sliding ? cos_swa_ : cos_full_;
        const float* sn = w.sliding ? sin_swa_ : sin_full_;
        lag_rope_apply(q, qf_, cs, sn, qf_, T, nh, hd, w.rope_dim);
        lag_rope_apply(q, kf_, cs, sn, kf_, T, nkv, hd, w.rope_dim);
        if (D) { dumpf("Qcur_rope-0", qf_, T, qd); dumpf("Kcur_rope-0", kf_, T, kvd); dumpf("Vcur-0", vf_, T, kvd); }
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
        if (T == 1) {                                                     // decode: split over the keys
            const uint32_t nk = w.sliding ? ring_ : n_kv;
            lag_attn_decode(q, qf_, kc_[L], vc_[L], w.sliding ? mask_swa_ : mask_full_, attn_part_, nh, nkv, hd, nk, kq_scale);
            lag_attn_combine(q, attn_part_, attn_, nh, hd, nk);
        } else
            ink_attention(q, qf_, kc_[L], vc_[L], nullptr, w.sliding ? mask_swa_ : mask_full_, attn_, T, nh, nkv, hd,
                          w.sliding ? ring_ : n_kv, kq_scale);
        if (D) dumpf("attn_out-0", attn_, T, qd);
        lag_attn_gate(q, attn_, gf_, T, nh, hd);
        if (D) dumpf("attn_gated-0", attn_, T, qd);
        lag_f32_to_f16(q, attn_, a16_, uint64_t(T) * qd);
        if (T == 1) { gemv_q8_0_soa_f16_g(q, a16_, w.wo.qs, w.wo.d, o16_, qd, H); lag_f16_to_f32(q, o16_, fy_, H); }
        else          gemm_q8(a16_, w.wo, fy_, T, qd, H);
        if (D) dumpf("attn_o_proj-0", fy_, T, H);
        lag_add_f32(q, x_, fy_, uint64_t(T) * H);
        if (D) dumpf("ffn_inp-0", x_, T, H);
        // ---- FFN ----
        lag_rmsnorm(q, x_, w.ffn_norm, xn_, xn16_, T, H, eps);
        if (D) dumpf("ffn_norm-0", xn_, T, H);
        if (!w.moe) {
            if (auto e = ffn_dense(w.ffn_gate, w.ffn_up, w.ffn_down, cfg_.ffn_dense, T, x_); !e.empty()) return e;
            if (D) dumpf("l_out-0", x_, T, H);
        } else {
            lag_gemv_f32(q, w.router, xn_, rlog_, T, H, E);
            lag_moe_router(q, rlog_, w.router_bias, ridx_, rw_, T, E, K, cfg_.expert_weights_norm, 0.0f);
            from_device(h_idx_.data(), ridx_, h_idx_.size() * 4);
            if (!(T == 1 && fused_moe_)) from_device(h_w_.data(), rw_, h_w_.size() * 4);   // fused decode: the weights stay on the card
            q.memset(acc_, 0, uint64_t(T) * H * 4);
            if (auto e = ffn_dense(w.sh_gate, w.sh_up, w.sh_down, cfg_.shared_expert_ffn, T, acc_); !e.empty()) return e;
            if (auto e = moe_experts(w, L, T); !e.empty()) return e;
            lag_add_f32(q, x_, acc_, uint64_t(T) * H);
        }
        if (probe) { q.wait(); probe(L, q, x_, T); }
    }
    // ---- head: the last row ----
    if (logits_host) {
        lag_rmsnorm(q, x_ + uint64_t(T - 1) * H, out_norm_, nullptr, xn16_, 1, H, eps);
        gemv_q6_K(q, xn16_, head_q6_, logits16_, H, cfg_.vocab);
        std::vector<sycl::half> l16(cfg_.vocab);
        from_device(l16.data(), logits16_, cfg_.vocab * sizeof(sycl::half));
        for (uint32_t i = 0; i < cfg_.vocab; ++i) logits_host[i] = float(l16[i]);
    }
    q.wait();
    pos_ = pos0 + T;
    return {};
}

// Each quant binding vs a host dequant + fp32 matvec on the same f16 input (rows: R distinct inputs through the
// multi-row kernel, each checked against its own matvec). A layout mismatch shows as a large error.
std::string LagunaModel::self_check() {
    sycl::queue& q = *q_;
    const uint32_t H = cfg_.hidden, EF = cfg_.expert_ffn;
    std::mt19937 rng(7); std::normal_distribution<float> nd(0.f, 1.f);
    uint8_t* dw = sycl::malloc_device<uint8_t>(slot_bytes_ ? slot_bytes_ : (64u << 20), q);
    auto check = [&](const std::string& what, const GgufTensorInfo* t, const uint8_t* host_src, uint64_t row0, uint32_t K, uint32_t N,
                     uint32_t R, uint32_t rows_check, auto&& run) -> std::string {
        std::vector<sycl::half> x16(uint64_t(R) * K); std::vector<float> xf(x16.size());
        for (size_t i = 0; i < x16.size(); ++i) { x16[i] = sycl::half(nd(rng)); xf[i] = float(x16[i]); }
        sycl::half* dx = sycl::malloc_device<sycl::half>(x16.size(), q); sycl::half* dy = sycl::malloc_device<sycl::half>(uint64_t(R) * N, q);
        q.memcpy(dx, x16.data(), x16.size() * 2).wait();
        run(dx, dy);
        std::vector<sycl::half> y16(uint64_t(R) * N); q.memcpy(y16.data(), dy, y16.size() * 2).wait();
        sycl::free(dx, q); sycl::free(dy, q);
        std::vector<float> wrow(K);
        double num = 0, den = 0;
        for (uint32_t n = 0; n < rows_check; ++n) {
            if (auto e = dequant_rows(t, row0 + n, 1, K, wrow.data(), host_src); !e.empty()) return e;
            for (uint32_t r = 0; r < R; ++r) {
                double ref = 0; for (uint32_t k = 0; k < K; ++k) ref += double(wrow[k]) * xf[uint64_t(r) * K + k];
                const double d = float(y16[uint64_t(r) * N + n]) - ref; num += d * d; den += ref * ref;
            }
        }
        const double rel = std::sqrt(num / std::max(den, 1e-30));
        std::printf("[laguna self-check] %-40s rel err %.2e\n", what.c_str(), rel);
        return rel < 5e-3 ? std::string() : "laguna self-check FAILED: " + what;
    };
    std::string err;
    if (const std::string tag = stage_tag(); !tag.empty()) std::printf("[laguna self-check]%s\n", tag.c_str());
    {   // the Q8_0 SoA projections: this card's first layer's attn_q (the decode GEMV) and attn_gate (R rows in one launch)
        const LagunaLayer& w0 = layers_[l0_];
        const uint32_t qd0 = w0.n_heads * cfg_.head_dim;
        const std::string b0 = "blk." + std::to_string(l0_) + ".", ln = "(layer " + std::to_string(l0_);
        err = check("attn_q Q8_0 SoA " + ln + ", 1 row)", g_->find_tensor(b0 + "attn_q.weight"), nullptr, 0, H, qd0, 1, qd0,
                    [&](sycl::half* dx, sycl::half* dy) { gemv_q8_0_soa_f16_g(q, dx, w0.wq.qs, w0.wq.d, dy, H, qd0); q.wait(); });
        if (err.empty())
            err = check("attn_gate Q8_0 SoA rows " + ln + ", R=5)", g_->find_tensor(b0 + "attn_gate.weight"), nullptr, 0, H, w0.n_heads, 5,
                        w0.n_heads, [&](sycl::half* dx, sycl::half* dy) { gemv_q8_0_soa_f16_rows(q, dx, w0.wg.qs, w0.wg.d, dy, H, w0.n_heads, 5); q.wait(); });
    }
    for (uint32_t L = 0; L < cfg_.n_layers && err.empty(); ++L) {
        const LagunaLayer& w = layers_[L];
        if (!w.moe) continue;
        const uint32_t e = 37;
        const uint64_t g_rows = uint64_t(e) * EF;            // expert e's first row in the bank
        q.memcpy(dw, w.gate_src + uint64_t(e) * w.gate_slice, w.gate_slice).wait();
        err = check("expert gate Q4_K (1 row)", w.gate_exps, w.gate_src, g_rows, H, EF, 1, EF, [&](sycl::half* dx, sycl::half* dy) { gemv_q4_K(q, dx, dw, dy, H, EF); q.wait(); });
        if (err.empty()) err = check("expert gate Q4_K rows (R=5)", w.gate_exps, w.gate_src, g_rows, H, EF, 5, 256, [&](sycl::half* dx, sycl::half* dy) { gemv_q4_K_rows(q, dx, dw, dy, H, EF, 5); q.wait(); });
        break;
    }
    bool did5 = false, did6 = false;
    for (uint32_t L = 0; L < cfg_.n_layers && err.empty() && !(did5 && did6); ++L) {
        const LagunaLayer& w = layers_[L];
        if (!w.moe || (w.down_q5 && did5) || (!w.down_q5 && did6)) continue;
        const uint32_t e = 201;
        const uint64_t d_rows = uint64_t(e) * H;
        q.memcpy(dw, w.down_src + uint64_t(e) * w.down_slice, w.down_slice).wait();
        const std::string tag = std::string("expert down ") + (w.down_q5 ? "Q5_K" : "Q6_K") + " (layer " + std::to_string(L) + ")";
        err = check(tag + " 1 row", w.down_exps, w.down_src, d_rows, EF, H, 1, H, [&](sycl::half* dx, sycl::half* dy) {
            if (w.down_q5) gemv_q5_K(q, dx, dw, dy, EF, H); else gemv_q6_K(q, dx, dw, dy, EF, H); q.wait(); });
        if (err.empty()) err = check(tag + " rows (R=11)", w.down_exps, w.down_src, d_rows, EF, H, 11, 512, [&](sycl::half* dx, sycl::half* dy) {
            if (w.down_q5) gemv_q5_K_rows(q, dx, dw, dy, EF, H, 11);
            else for (uint32_t r0 = 0; r0 < 11; r0 += 8) gemm_q6_K(q, dx + uint64_t(r0) * EF, dw, dy + uint64_t(r0) * H, std::min(8u, 11 - r0), EF, H);
            q.wait(); });
        (w.down_q5 ? did5 : did6) = true;
    }
    sycl::free(dw, q);
    if (!err.empty()) return err;
    // The decode MoE (one row, K picks), the fused launches and the expert-major loop, each against a host fp64 reference
    // (gate/up/SwiGLU/down/weights, no f16 rounding): one layer per down type.
    {
        const uint32_t K = cfg_.n_experts_used;
        const int32_t picks[16] = {3, 17, 37, 64, 99, 128, 150, 201, 230, 255, 5, 7, 11, 13, 19, 23};
        std::vector<sycl::half> x16(H); std::vector<double> xd(H);
        for (uint32_t i = 0; i < H; ++i) { x16[i] = sycl::half(nd(rng)); xd[i] = float(x16[i]); }
        std::vector<float> wk(K); float ws = 0.f;
        for (auto& v : wk) { v = std::fabs(nd(rng)) + 0.1f; ws += v; }
        for (auto& v : wk) v /= ws;
        h_idx_.assign(picks, picks + K); h_w_ = wk;
        const bool keep = fused_moe_;
        bool d5 = false, d6 = false;
        for (uint32_t L = 0; L < cfg_.n_layers && err.empty() && !(d5 && d6); ++L) {
            const LagunaLayer& w = layers_[L];
            if (!w.moe || (w.down_q5 && d5) || (!w.down_q5 && d6)) continue;
            (w.down_q5 ? d5 : d6) = true;
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
                    ref[n] += double(cfg_.expert_weights_scale) * wk[k] * y;
                }
            }
            double rel[2] = {0, 0};
            for (int f = 0; f < 2 && err.empty(); ++f) {
                fused_moe_ = f == 1;
                q.memcpy(xn16_, x16.data(), uint64_t(H) * 2);
                q.memcpy(rw_, wk.data(), uint64_t(K) * 4);
                q.memset(acc_, 0, uint64_t(H) * 4).wait();
                if (auto e = moe_experts(w, L, 1); !e.empty()) { err = e; break; }
                std::vector<float> got(H); q.memcpy(got.data(), acc_, uint64_t(H) * 4).wait();
                double num = 0, den = 0;
                for (uint32_t n = 0; n < H; ++n) { const double d = got[n] - ref[n]; num += d * d; den += ref[n] * ref[n]; }
                rel[f] = std::sqrt(num / std::max(den, 1e-30));
            }
            fused_moe_ = keep;
            if (!err.empty()) break;
            std::printf("[laguna self-check] decode MoE, %u picks, down %s (layer %u): fused rel err %.2e | loop %.2e\n", K,
                        w.down_q5 ? "Q5_K" : "Q6_K", L, rel[1], rel[0]);
            if (rel[1] >= 5e-3) err = "laguna self-check FAILED: fused decode MoE (layer " + std::to_string(L) + ")";
        }
        if (!err.empty()) return err;
    }
    // decode attention (split over the keys) vs ink_attention, one query row: 72 and 48 heads over the KV heads, 777 keys
    // (not a whole split), a seventh of them and the tail masked
    for (const uint32_t nhh : {72u, 48u}) {
        const uint32_t nkv = cfg_.n_kv_heads, hd = cfg_.head_dim, nk = 777;
        std::vector<float> hq(uint64_t(nhh) * hd), hk(uint64_t(nk) * nkv * hd), hv(hk.size()), hm(nk, 0.f);
        for (auto& x : hq) x = nd(rng);
        for (auto& x : hk) x = nd(rng);
        for (auto& x : hv) x = nd(rng);
        for (uint32_t j = 0; j < nk; ++j) if (j % 7 == 3 || j > 700) hm[j] = -INFINITY;
        float* dq = sycl::malloc_device<float>(hq.size(), q); float* dk = sycl::malloc_device<float>(hk.size(), q);
        float* dv = sycl::malloc_device<float>(hv.size(), q); float* dm = sycl::malloc_device<float>(nk, q);
        float* o1 = sycl::malloc_device<float>(hq.size(), q); float* o2 = sycl::malloc_device<float>(hq.size(), q);
        q.memcpy(dq, hq.data(), hq.size() * 4); q.memcpy(dk, hk.data(), hk.size() * 4);
        q.memcpy(dv, hv.data(), hv.size() * 4); q.memcpy(dm, hm.data(), nk * 4);
        const float scl = 1.0f / std::sqrt(float(hd));
        ink_attention(q, dq, dk, dv, nullptr, dm, o1, 1, nhh, nkv, hd, nk, scl);
        lag_attn_decode(q, dq, dk, dv, dm, attn_part_, nhh, nkv, hd, nk, scl);
        lag_attn_combine(q, attn_part_, o2, nhh, hd, nk);
        std::vector<float> a(hq.size()), b(hq.size());
        q.memcpy(a.data(), o1, a.size() * 4); q.memcpy(b.data(), o2, b.size() * 4).wait();
        for (void* p : {(void*)dq, (void*)dk, (void*)dv, (void*)dm, (void*)o1, (void*)o2}) sycl::free(p, q);
        double num = 0, den = 0;
        for (size_t i = 0; i < a.size(); ++i) { const double d = double(b[i]) - a[i]; num += d * d; den += double(a[i]) * a[i]; }
        const double rel = std::sqrt(num / std::max(den, 1e-30));
        std::printf("[laguna self-check] decode attention, %u heads, %u keys: split-K vs ink_attention rel err %.2e\n", nhh, nk, rel);
        if (!(rel < 1e-5)) return "laguna self-check FAILED: decode attention (" + std::to_string(nhh) + " heads)";
    }
    if (l1_ == cfg_.n_layers) {
        const auto* o = g_->find_tensor("output.weight");
        if (auto e = check("output head Q6_K (first 4096 rows)", o, nullptr, 0, H, cfg_.vocab, 1, 4096, [&](sycl::half* dx, sycl::half* dy) {
                gemv_q6_K(q, dx, head_q6_, dy, H, cfg_.vocab); q.wait(); }); !e.empty()) return e;
    }
    return next_ ? next_->self_check() : std::string();
}

}  // namespace ie
