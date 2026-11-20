// src/ops/hy4flash_small.cpp — Hy4-flash: the fp16 matrix products of a decode token as direct kernels.
//
// At T = 1 the bring-up forward sent ~250 small products a token through oneDNN (q_a, kv_a, the absorbed query, scores,
// context, values, the indexer's maps, the n-gram maps). A oneDNN call costs tens of microseconds whatever the size --
// primitive lookup, memory objects, its own submit -- and these operands are a few MB at most. Same contracts as
// gemm_nt_f16_onednn / gemm_bmm_nt_f16_onednn / gemm_fp16_onednn (include/ie/ops.hpp), fp32 accumulation.
#include "ie/hy4flash_q8.hpp"
#include "ie/kernel_profiler.hpp"

#include <sycl/sycl.hpp>

namespace ie {

namespace {
using h8 = sycl::vec<sycl::half, 8>;
// n8 groups of 8 fp16 pairs at (16-byte aligned) a and b
inline float dot_h8(const h8* a, const h8* b, uint32_t n8) {
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    for (uint32_t i = 0; i < n8; ++i) {
        const h8 x = a[i], y = b[i];
        s0 += float(x[0]) * float(y[0]) + float(x[4]) * float(y[4]);
        s1 += float(x[1]) * float(y[1]) + float(x[5]) * float(y[5]);
        s2 += float(x[2]) * float(y[2]) + float(x[6]) * float(y[6]);
        s3 += float(x[3]) * float(y[3]) + float(x[7]) * float(y[7]);
    }
    return (s0 + s1) + (s2 + s3);
}
}  // namespace

// y [T, N] fp32 = x16 [T, K] . W^T, W fp16 [N, K]; K a multiple of 32. A sub-group of 16 lanes per output row strides
// the row's 32-weight blocks (the W8A16 kernels' shape), 16 rows a work-group.
sycl::event hy4f_f16_gemv_f32(sycl::queue& q, const sycl::half* x16, const sycl::half* w16, float* y, uint32_t K, uint32_t N, uint32_t T) {
    constexpr int SG = 16, NPW = 16, WGI = NPW * SG;
    const uint32_t kb = K / 32, wgs = (N + NPW - 1) / NPW;
    return ie::ps(q, "hy4f_f16_gemv_f32", [&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<2>({size_t(T), size_t(wgs) * WGI}, {1, size_t(WGI)}), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint32_t lid = uint32_t(it.get_local_id(1)), wg = uint32_t(it.get_group(1));
            const uint32_t t = uint32_t(it.get_global_id(0)), n = wg * NPW + lid / SG, lane = lid % SG;
            if (n >= N) return;
            const sycl::half* w = w16 + uint64_t(n) * K; const sycl::half* xr = x16 + uint64_t(t) * K;
            float a = 0.f;
            for (uint32_t b = lane; b < kb; b += SG)
                a += dot_h8(reinterpret_cast<const h8*>(w + uint64_t(b) * 32), reinterpret_cast<const h8*>(xr + uint64_t(b) * 32), 4);
            a = sycl::reduce_over_group(it.get_sub_group(), a, sycl::plus<float>());
            if (lane == 0) y[uint64_t(t) * N + n] = a;
        });
    });
}

// y [T, N] fp32 = x32 [T, K] . W^T, W fp32 [N, K]; K a multiple of 8 (the router: 512 x 3072, read back for the host's
// top-k). A sub-group of 16 lanes per row, each lane taking every 16th block of 8 floats as one vector load -- the
// lanes of the kernel it replaces each read one float per cache line (32.6 us a call against 11 us of weight traffic).
sycl::event hy4f_f32_gemv_f32(sycl::queue& q, const float* x32, const float* w32, float* y, uint32_t K, uint32_t N, uint32_t T) {
    constexpr int SG = 16;
    const uint32_t kb = K / 8;
    using f8 = sycl::vec<float, 8>;
    return ie::ps(q, "hy4f_f32_gemv_f32", [&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<3>({size_t(T), size_t(N), size_t(SG)}, {1, 1, size_t(SG)}), [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint32_t t = uint32_t(it.get_global_id(0)), n = uint32_t(it.get_global_id(1)), lane = uint32_t(it.get_local_id(2));
            const f8* w = reinterpret_cast<const f8*>(w32 + uint64_t(n) * K); const f8* xr = reinterpret_cast<const f8*>(x32 + uint64_t(t) * K);
            f8 acc(0.f);
            for (uint32_t b = lane; b < kb; b += SG) acc += w[b] * xr[b];
            float a = ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]));
            a = sycl::reduce_over_group(it.get_sub_group(), a, sycl::plus<float>());
            if (lane == 0) y[uint64_t(t) * N + n] = a;
        });
    });
}

// An indexed layer past its recall budget: row b attends the keys sel[b, 0 .. cnt) (any positions of the cache).
// scores [B, NH, S] fp32 = a16 [B, NH, LW] . kv[sel [B, S]]^T (0 where sel is -1); LW a multiple of 8.
sycl::event hy4f_f16_scores_sel(sycl::queue& q, const sycl::half* a16, const sycl::half* kv, const int32_t* sel, float* scores,
                                uint32_t B, uint32_t NH, uint32_t LW, uint32_t S) {
    const uint32_t n8 = LW / 8;
    return ie::ps(q, "hy4f_f16_scores_sel", [&](sycl::handler& h) {
        h.parallel_for(sycl::range<3>(B, S, NH), [=](sycl::id<3> i) {
            const uint64_t bh = uint64_t(i[0]) * NH + i[2];
            const int32_t k = sel[uint64_t(i[0]) * S + i[1]];
            scores[bh * S + i[1]] = k < 0 ? 0.f : dot_h8(reinterpret_cast<const h8*>(a16 + bh * LW), reinterpret_cast<const h8*>(kv + uint64_t(k) * LW), n8);
        });
    });
}
// ctx16 [B, NH, LW] = sum over s <= hi[b] of p16 [B, NH, S] * kv[sel [B, S]]. One work-item per output element walking
// the row's keys, as hy4f_f16_mix_f16 does over a window (a first form with 64 dims a work-item left a decode token 288
// work-items: 1,164 us a call against 40 for the window kernel).
sycl::event hy4f_f16_mix_sel(sycl::queue& q, const sycl::half* p16, const sycl::half* kv, const int32_t* sel, const int32_t* hi, sycl::half* ctx16,
                             uint32_t B, uint32_t NH, uint32_t LW, uint32_t S) {
    return ie::ps(q, "hy4f_f16_mix_sel", [&](sycl::handler& h) {
        h.parallel_for(sycl::range<3>(B, NH, LW), [=](sycl::id<3> i) {
            const uint64_t b = i[0], bh = b * NH + i[1], d = i[2];
            const sycl::half* p = p16 + bh * S; const int32_t* sl = sel + b * S; const int32_t n = hi[b] + 1;
            float s0 = 0.f, s1 = 0.f;
            int32_t k = 0;
            for (; k + 1 < n; k += 2) { s0 += float(p[k]) * float(kv[uint64_t(sl[k]) * LW + d]); s1 += float(p[k + 1]) * float(kv[uint64_t(sl[k + 1]) * LW + d]); }
            if (k < n) s0 += float(p[k]) * float(kv[uint64_t(sl[k]) * LW + d]);
            ctx16[bh * LW + d] = sycl::half(s0 + s1);
        });
    });
}

// The compressed indexer on the card. Its query heads rotated at the row's position: qrot [T, heads, hd] from q32, the
// last rope_dim dims of every head turned by the row's (cos | sin) table cs [T, rope_dim].
sycl::event hy4f_ix_rotate(sycl::queue& q, const float* q32, const float* cs, float* qrot, uint32_t T, uint32_t heads, uint32_t hd, uint32_t rope_dim) {
    const uint32_t pass = hd - rope_dim, nf = rope_dim / 2;
    return ie::ps(q, "hy4f_ix_rotate", [&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>(T, heads), [=](sycl::id<2> i) {
            const float* s = q32 + (uint64_t(i[0]) * heads + i[1]) * hd; float* d = qrot + (uint64_t(i[0]) * heads + i[1]) * hd;
            const float* c = cs + uint64_t(i[0]) * rope_dim;
            for (uint32_t j = 0; j < pass; ++j) d[j] = s[j];
            for (uint32_t f = 0; f < nf; ++f) {
                const float a = s[pass + 2 * f], b = s[pass + 2 * f + 1];
                d[pass + 2 * f] = a * c[f] - b * c[nf + f]; d[pass + 2 * f + 1] = b * c[f] + a * c[nf + f];
            }
        });
    });
}
// score [T, GC]: query row t against group g < own[t] (the rest is not written) =
// sum over heads of max(qrot[t, h, :] . grp[g, :], 0) * hw[t, h] * hs  -- Hy4fIndexer::recall's sum, on the card: a
// sub-group of 16 lanes per (row, group), a lane a head (a first form with one work-item per (row, group) took 192 us for
// 977 groups). hd a multiple of 8.
sycl::event hy4f_ix_scores(sycl::queue& q, const float* qrot, const float* hw, const float* grp, const int32_t* own, float* score,
                           uint32_t T, uint32_t heads, uint32_t hd, uint32_t GC, float hs) {
    constexpr int SG = 16;
    const uint32_t n8 = hd / 8;
    using f8 = sycl::vec<float, 8>;
    return ie::ps(q, "hy4f_ix_scores", [&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<3>({size_t(T), size_t(GC), size_t(SG)}, {1, 1, size_t(SG)}), [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint64_t t = it.get_global_id(0), g = it.get_global_id(1); const uint32_t lane = uint32_t(it.get_local_id(2));
            float part = 0.f;
            if (int32_t(g) < own[t]) {
                const f8* k = reinterpret_cast<const f8*>(grp + g * hd);
                for (uint32_t hh = lane; hh < heads; hh += SG) {
                    const f8* d = reinterpret_cast<const f8*>(qrot + (t * heads + hh) * hd);
                    f8 acc(0.f);
                    for (uint32_t j = 0; j < n8; ++j) acc += d[j] * k[j];
                    const float dot = ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]));
                    if (dot > 0.f) part += hw[t * heads + hh] * hs * dot;
                }
            }
            const float sum = sycl::reduce_over_group(it.get_sub_group(), part, sycl::plus<float>());
            if (lane == 0 && int32_t(g) < own[t]) score[t * GC + g] = sum;
        });
    });
}

// Grouped: y [M, G * OPG] fp32, y[m, g, o] = a16[m, g, :IPG] . w16[g * OPG + o, :IPG]; IPG a multiple of 8.
// (G = 1: plain y = a . W^T for a short K -- the attention scores of a decode token against a window of keys.)
sycl::event hy4f_f16_bmm_f32(sycl::queue& q, const sycl::half* a16, const sycl::half* w16, float* y, uint32_t M, uint32_t G, uint32_t IPG, uint32_t OPG) {
    const uint32_t n8 = IPG / 8;
    return ie::ps(q, "hy4f_f16_bmm_f32", [&](sycl::handler& h) {
        h.parallel_for(sycl::range<3>(M, G, OPG), [=](sycl::id<3> i) {
            const uint64_t mg = uint64_t(i[0]) * G + i[1];
            y[mg * OPG + i[2]] = dot_h8(reinterpret_cast<const h8*>(a16 + mg * IPG), reinterpret_cast<const h8*>(w16 + (uint64_t(i[1]) * OPG + i[2]) * IPG), n8);
        });
    });
}

// The absorbed query written where the scores read it: a16 [M * G, LW], columns [0, OPG) of row (m, g) =
// half(x16[m, g, :IPG] . w16[g * OPG + o, :IPG])  -- hy4f_f16_bmm_f32 and the copy into place in one launch.
sycl::event hy4f_f16_bmm_place(sycl::queue& q, const sycl::half* x16, const sycl::half* w16, sycl::half* a16, uint32_t M, uint32_t G, uint32_t IPG, uint32_t OPG, uint32_t LW) {
    const uint32_t n8 = IPG / 8;
    return ie::ps(q, "hy4f_f16_bmm_place", [&](sycl::handler& h) {
        h.parallel_for(sycl::range<3>(M, G, OPG), [=](sycl::id<3> i) {
            const uint64_t mg = uint64_t(i[0]) * G + i[1];
            a16[mg * LW + i[2]] = sycl::half(dot_h8(reinterpret_cast<const h8*>(x16 + mg * IPG), reinterpret_cast<const h8*>(w16 + (uint64_t(i[1]) * OPG + i[2]) * IPG), n8));
        });
    });
}

// The values through W_v and the gate in one launch: y16[m, g, o] = half((c16[m, g, :IPG] . w16[g * OPG + o, :IPG]) * sigmoid(g32[m, g, o]))
sycl::event hy4f_f16_bmm_gate(sycl::queue& q, const sycl::half* c16, const sycl::half* w16, const float* g32, sycl::half* y16, uint32_t M, uint32_t G, uint32_t IPG, uint32_t OPG) {
    const uint32_t n8 = IPG / 8;
    return ie::ps(q, "hy4f_f16_bmm_gate", [&](sycl::handler& h) {
        h.parallel_for(sycl::range<3>(M, G, OPG), [=](sycl::id<3> i) {
            const uint64_t mg = uint64_t(i[0]) * G + i[1];
            const float v = dot_h8(reinterpret_cast<const h8*>(c16 + mg * IPG), reinterpret_cast<const h8*>(w16 + (uint64_t(i[1]) * OPG + i[2]) * IPG), n8);
            y16[mg * OPG + i[2]] = sycl::half(v * (1.0f / (1.0f + sycl::exp(-g32[mg * OPG + i[2]]))));
        });
    });
}

// One work-group of 256 per (row, head). The scores: a work-item per key, its whole row read as independent 16-byte loads
// against the query in local memory. The softmax in local memory. The weighted sum: a work-item per (8-dim chunk, key
// partition) -- the 72 chunks of a 576-wide key are 72 neighbouring work-items, so a key's row is one coalesced read -- and
// the partitions' sums added in local memory. (The three-launch form spent ~74 us a layer a token on ~0.6 MB of keys, mostly
// its context kernel's one work-item per output walking every key; a first fused form with a sub-group per key, a
// reduction and a fresh memory wait each, took 81 us. 2026-11-19.)
sycl::event hy4f_attn_decode(sycl::queue& q, const sycl::half* a16, const sycl::half* keys, const int32_t* sel, uint32_t S, const int32_t* lo, const int32_t* hi,
                             const float* sinks, sycl::half* ctx16, uint32_t rows, uint32_t NH, uint32_t LW, float scaling, bool use_sink) {
    constexpr int WG = 256;
    using f8 = sycl::vec<float, 8>;
    const uint32_t nc = LW / 8, P = WG / nc;                    // chunks of 8 dims; key partitions of the weighted sum (3 at 576)
    return ie::ps(q, "hy4f_attn_decode", [&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sc(sycl::range<1>(kAttnMaxKeys), h);
        sycl::local_accessor<f8, 1> qs(sycl::range<1>(kAttnMaxLW / 8), h);           // the query as fp32 (one conversion, not one a key)
        sycl::local_accessor<float, 1> part(sycl::range<1>(size_t(WG) * 8), h);       // [P, LW]: P * nc <= WG
        h.parallel_for(sycl::nd_range<2>({size_t(rows) * NH, size_t(WG)}, {1, size_t(WG)}), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
            const uint32_t rh = uint32_t(it.get_global_id(0)), r = rh / NH, hh = rh % NH;
            const int32_t lid = int32_t(it.get_local_linear_id());
            const int32_t l = lo[r], n = hi[r] - l + 1;
            const h8* qv = reinterpret_cast<const h8*>(a16 + uint64_t(rh) * LW);
            for (uint32_t c = uint32_t(lid); c < nc; c += WG) qs[c] = qv[c].convert<float>();
            const int32_t* sr = sel ? sel + uint64_t(r) * S : nullptr;
            it.barrier(sycl::access::fence_space::local_space);
            for (int32_t j = lid; j < n; j += WG) {                     // the scores: a work-item per key
                const h8* kp = reinterpret_cast<const h8*>(keys + uint64_t(sr ? sr[l + j] : l + j) * LW);
                f8 acc(0.f);
                for (uint32_t c = 0; c < nc; ++c) acc += kp[c].convert<float>() * qs[c];
                sc[j] = (((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]))) * scaling;
            }
            it.barrier(sycl::access::fence_space::local_space);
            const float sink = sinks[hh];                                // the softmax, the sink logit in its denominator
            float m = use_sink ? sink : -3.0e38f;
            for (int32_t j = lid; j < n; j += WG) m = sycl::fmax(m, sc[j]);
            m = sycl::reduce_over_group(it.get_group(), m, sycl::maximum<float>());
            float s = 0.f;
            for (int32_t j = lid; j < n; j += WG) { const float e = sycl::exp(sc[j] - m); sc[j] = e; s += e; }
            s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>()) + (use_sink ? sycl::exp(sink - m) : 0.f);
            const float inv = 1.0f / s;
            it.barrier(sycl::access::fence_space::local_space);
            const uint32_t c = uint32_t(lid) % nc, pp = uint32_t(lid) / nc;   // the weighted sum: chunk c of partition pp's keys
            if (pp < P) {
                f8 a(0.f);
                #pragma unroll 4
                for (int32_t j = int32_t(pp); j < n; j += int32_t(P))
                    a += reinterpret_cast<const h8*>(keys + uint64_t(sr ? sr[l + j] : l + j) * LW)[c].convert<float>() * sc[j];
                for (int e = 0; e < 8; ++e) part[size_t(pp) * LW + c * 8 + uint32_t(e)] = a[e];
            }
            it.barrier(sycl::access::fence_space::local_space);
            for (uint32_t d = uint32_t(lid); d < LW; d += WG) {
                float t = 0.f;
                for (uint32_t p2 = 0; p2 < P; ++p2) t += part[size_t(p2) * LW + d];
                ctx16[uint64_t(rh) * LW + d] = sycl::half(t * inv);
            }
        });
    });
}

// c16 [M, N] fp16 = p16 [M, K] . b16 [K, N]   (the context of a decode token: its attention weights over K keys)
sycl::event hy4f_f16_mix_f16(sycl::queue& q, const sycl::half* p16, const sycl::half* b16, sycl::half* c16, uint32_t M, uint32_t N, uint32_t K) {
    return ie::ps(q, "hy4f_f16_mix_f16", [&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>(M, N), [=](sycl::id<2> i) {
            const sycl::half* p = p16 + uint64_t(i[0]) * K;
            float s0 = 0.f, s1 = 0.f;
            uint32_t k = 0;
            for (; k + 1 < K; k += 2) { s0 += float(p[k]) * float(b16[uint64_t(k) * N + i[1]]); s1 += float(p[k + 1]) * float(b16[uint64_t(k + 1) * N + i[1]]); }
            if (k < K) s0 += float(p[k]) * float(b16[uint64_t(k) * N + i[1]]);
            c16[i[0] * N + i[1]] = sycl::half(s0 + s1);
        });
    });
}

}  // namespace ie
