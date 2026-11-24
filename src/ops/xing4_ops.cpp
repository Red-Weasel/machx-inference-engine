// src/ops/xing4_ops.cpp -- Xing4 kernels (include/ie/xing4_ops.hpp). Written for this model's shapes: 4 streams of 3584,
// MLA rows of 576, 64 experts of 1024. Every kernel and helper carries the Xg4 prefix.
#include "ie/xing4_ops.hpp"
#include "ie/dp4a_native.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace ie {

namespace {

constexpr uint32_t kXg4SG = 16;

inline float xg4_h2f(uint16_t b) { return float(sycl::bit_cast<sycl::half>(b)); }
inline float xg4_sig(float x) { return 1.0f / (1.0f + sycl::exp(-x)); }

// One Q8_0 block (32 int8 weights) against 32 fp16 activations, fp32 sum (without the block scale).
inline float xg4_blk(const int8_t* wq, const sycl::half* xh) {
    const sycl::vec<int32_t, 8> w = *reinterpret_cast<const sycl::vec<int32_t, 8>*>(wq);
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const sycl::vec<sycl::half, 8> xv = *reinterpret_cast<const sycl::vec<sycl::half, 8>*>(xh + 8 * i);
        const int32_t a = w[2 * i], b = w[2 * i + 1];
        s += float(int8_t(a)) * float(xv[0]) + float(int8_t(a >> 8)) * float(xv[1]) + float(int8_t(a >> 16)) * float(xv[2]) +
             float(int8_t(a >> 24)) * float(xv[3]) + float(int8_t(b)) * float(xv[4]) + float(int8_t(b >> 8)) * float(xv[5]) +
             float(int8_t(b >> 16)) * float(xv[6]) + float(int8_t(b >> 24)) * float(xv[7]);
    }
    return s;
}

// 8 words of signed bytes (32 weights) against 32 activations: the sums of the two halves of 16
inline void xg4_dot32s(const sycl::vec<int32_t, 8>& w, const sycl::half* xh, float& s0, float& s1) {
    s0 = 0.f; s1 = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const sycl::vec<sycl::half, 8> xv = *reinterpret_cast<const sycl::vec<sycl::half, 8>*>(xh + 8 * i);
        const int32_t a = w[2 * i], b = w[2 * i + 1];
        const float v = float(int8_t(a)) * float(xv[0]) + float(int8_t(a >> 8)) * float(xv[1]) + float(int8_t(a >> 16)) * float(xv[2]) +
                        float(int8_t(a >> 24)) * float(xv[3]) + float(int8_t(b)) * float(xv[4]) + float(int8_t(b >> 8)) * float(xv[5]) +
                        float(int8_t(b >> 16)) * float(xv[6]) + float(int8_t(b >> 24)) * float(xv[7]);
        if (i < 2) s0 += v; else s1 += v;
    }
}
// unsigned bytes against activations: the dot and the sum of the activations
inline float xg4_dot32u(const sycl::vec<uint32_t, 8>& w, const sycl::half* xh, float& sx) {
    float s = 0.f, t = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const sycl::vec<sycl::half, 8> xv = *reinterpret_cast<const sycl::vec<sycl::half, 8>*>(xh + 8 * i);
        const uint32_t a = w[2 * i], b = w[2 * i + 1];
        s += float(a & 255u) * float(xv[0]) + float((a >> 8) & 255u) * float(xv[1]) + float((a >> 16) & 255u) * float(xv[2]) +
             float(a >> 24) * float(xv[3]) + float(b & 255u) * float(xv[4]) + float((b >> 8) & 255u) * float(xv[5]) +
             float((b >> 16) & 255u) * float(xv[6]) + float(b >> 24) * float(xv[7]);
        t += ((float(xv[0]) + float(xv[1])) + (float(xv[2]) + float(xv[3]))) + ((float(xv[4]) + float(xv[5])) + (float(xv[6]) + float(xv[7])));
    }
    sx = t;
    return s;
}

// Weights 32b .. 32b + 31 of row `row` (K inputs) against x (the 32 activations of that unit), scales applied.
inline float xg4_unit(const Xg4W& w, size_t row, uint32_t K, uint32_t b, const sycl::half* x) {
    switch (w.fmt) {
    default:
        return xg4_h2f(w.d[row * (K / 32) + b]) * xg4_blk(w.qs + row * K + size_t(b) * 32, x);
    case kXg4Q6K: {
        const uint32_t blk = b >> 3, u = b & 7, n = u >> 2, qd = u & 3;
        const uint8_t* ql = reinterpret_cast<const uint8_t*>(w.qs) + row * (K / 2) + size_t(blk) * 128 + n * 64 + (qd & 1) * 32;
        const uint8_t* qh = w.qh + row * (K / 4) + size_t(blk) * 64 + n * 32;
        const int8_t* sc = reinterpret_cast<const int8_t*>(w.sc) + row * (K / 16) + size_t(blk) * 16 + 2 * u;
        const sycl::vec<uint32_t, 8> L = *reinterpret_cast<const sycl::vec<uint32_t, 8>*>(ql);
        const sycl::vec<uint32_t, 8> Hb = *reinterpret_cast<const sycl::vec<uint32_t, 8>*>(qh);
        const uint32_t sl = (qd >> 1) * 4, sh = 2 * qd;
        sycl::vec<int32_t, 8> v;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t qv = ((L[i] >> sl) & 0x0F0F0F0Fu) | (((Hb[i] >> sh) & 0x03030303u) << 4);
            v[i] = int32_t(((qv | 0x80808080u) - 0x20202020u) ^ 0x80808080u);      // bytewise q - 32
        }
        float s0, s1; xg4_dot32s(v, x, s0, s1);
        return xg4_h2f(w.d[row * (K / 256) + blk]) * (float(sc[0]) * s0 + float(sc[1]) * s1);
    }
    case kXg4Q5K:
    case kXg4Q4K: {
        const uint32_t blk = b >> 3, sb = b & 7, j = sb >> 1;
        const sycl::vec<uint32_t, 8> L = *reinterpret_cast<const sycl::vec<uint32_t, 8>*>(reinterpret_cast<const uint8_t*>(w.qs) + row * (K / 2) + size_t(blk) * 128 + j * 32);
        const uint32_t sl = (sb & 1) * 4;
        sycl::vec<uint32_t, 8> v;
        if (w.fmt == kXg4Q5K) {
            const sycl::vec<uint32_t, 8> Hb = *reinterpret_cast<const sycl::vec<uint32_t, 8>*>(w.qh + row * (K / 8) + size_t(blk) * 32);
#pragma unroll
            for (int i = 0; i < 8; ++i) v[i] = ((L[i] >> sl) & 0x0F0F0F0Fu) | (((Hb[i] >> sb) & 0x01010101u) << 4);
        } else {
#pragma unroll
            for (int i = 0; i < 8; ++i) v[i] = (L[i] >> sl) & 0x0F0F0F0Fu;
        }
        float sx; const float dot = xg4_dot32u(v, x, sx);
        const uint8_t* sm = w.sc + (row * (K / 32) + b) * 2;
        const uint16_t* dd = w.d + (row * (K / 256) + blk) * 2;
        return xg4_h2f(dd[0]) * float(sm[0]) * dot - xg4_h2f(dd[1]) * float(sm[1]) * sx;
    }
    }
}

// Weights 32b .. 32b + 31 against int8 activations xq (that unit's 32), xd its scale, xs its scaled sum: integer dots.
// Unit b (32 weights) of row `row` against the int8 activation row (xq [K], xd / xs [K/32]): integer dots.
// Q6_K: unit b covers, in super-block b / 8, half n = (b % 8) / 4 and positions l0 .. l0 + 7 (l0 = 8 (b % 4)) of each of the
// four 32-wide quarters -- the 8 + 8 ql bytes and 8 qh bytes no other unit reads (24 bytes for 32 weights, nothing twice).
inline float xg4_unit_i8(const Xg4W& w, size_t row, uint32_t K, uint32_t b, const int8_t* xq, const float* xd, const float* xs) {
    switch (w.fmt) {
    default: {
        const sycl::vec<uint32_t, 8> X = *reinterpret_cast<const sycl::vec<uint32_t, 8>*>(xq + size_t(b) * 32);
        const sycl::vec<uint32_t, 8> Wv = *reinterpret_cast<const sycl::vec<uint32_t, 8>*>(w.qs + row * K + size_t(b) * 32);
        int32_t id = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) id = dp4a_ss_native(Wv[i], X[i], id);
        return xg4_h2f(w.d[row * (K / 32) + b]) * xd[b] * float(id);
    }
    case kXg4Q6K: {
        const uint32_t blk = b >> 3, u = b & 7, n = u >> 2, l0 = (u & 3) * 8;
        const uint8_t* qlb = reinterpret_cast<const uint8_t*>(w.qs) + row * (K / 2) + size_t(blk) * 128 + n * 64 + l0;
        const sycl::vec<uint32_t, 2> A = *reinterpret_cast<const sycl::vec<uint32_t, 2>*>(qlb);
        const sycl::vec<uint32_t, 2> B = *reinterpret_cast<const sycl::vec<uint32_t, 2>*>(qlb + 32);
        const sycl::vec<uint32_t, 2> Q = *reinterpret_cast<const sycl::vec<uint32_t, 2>*>(w.qh + row * (K / 4) + size_t(blk) * 64 + n * 32 + l0);
        const int8_t* sc = reinterpret_cast<const int8_t*>(w.sc) + row * (K / 16) + size_t(blk) * 16 + n * 8 + (l0 >> 4);
        const size_t ab = size_t(blk) * 8 + n * 4;                 // the activation block of quarter 0
        float acc = 0.f;
#pragma unroll
        for (int qd = 0; qd < 4; ++qd) {
            const sycl::vec<uint32_t, 2> X = *reinterpret_cast<const sycl::vec<uint32_t, 2>*>(xq + (ab + qd) * 32 + l0);
            int32_t id = 0;
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const uint32_t lo = (((qd & 1) ? B[i] : A[i]) >> ((qd >> 1) * 4)) & 0x0F0F0F0Fu;
                const uint32_t qv = lo | (((Q[i] >> (2 * qd)) & 0x03030303u) << 4);
                id = dp4a_ss_native(((qv | 0x80808080u) - 0x20202020u) ^ 0x80808080u, X[i], id);
            }
            acc += float(sc[2 * qd]) * xd[ab + qd] * float(id);
        }
        return xg4_h2f(w.d[row * (K / 256) + blk]) * acc;
    }
    case kXg4Q5K:
    case kXg4Q4K: {
        const sycl::vec<uint32_t, 8> X = *reinterpret_cast<const sycl::vec<uint32_t, 8>*>(xq + size_t(b) * 32);
        const uint32_t blk = b >> 3, sb = b & 7, j = sb >> 1;
        const sycl::vec<uint32_t, 8> L = *reinterpret_cast<const sycl::vec<uint32_t, 8>*>(reinterpret_cast<const uint8_t*>(w.qs) + row * (K / 2) + size_t(blk) * 128 + j * 32);
        const uint32_t sl = (sb & 1) * 4;
        int32_t id = 0;
        if (w.fmt == kXg4Q5K) {
            const sycl::vec<uint32_t, 8> Hb = *reinterpret_cast<const sycl::vec<uint32_t, 8>*>(w.qh + row * (K / 8) + size_t(blk) * 32);
#pragma unroll
            for (int i = 0; i < 8; ++i) id = dp4a_ss_native(((L[i] >> sl) & 0x0F0F0F0Fu) | (((Hb[i] >> sb) & 0x01010101u) << 4), X[i], id);
        } else {
#pragma unroll
            for (int i = 0; i < 8; ++i) id = dp4a_ss_native((L[i] >> sl) & 0x0F0F0F0Fu, X[i], id);
        }
        const uint8_t* sm = w.sc + (row * (K / 32) + b) * 2;
        const uint16_t* dd = w.d + (row * (K / 256) + blk) * 2;
        return xg4_h2f(dd[0]) * float(sm[0]) * xd[b] * float(id) - xg4_h2f(dd[1]) * float(sm[1]) * xs[b];
    }
    }
}

// 32 fp16 values -> int8, scale amax / 127, scaled sum
inline void xg4_qblk(const sycl::half* v, int8_t* q, float* d, float* s) {
    float f[32], am = 0.f;
#pragma unroll
    for (int i = 0; i < 32; ++i) { f[i] = float(v[i]); am = sycl::fmax(am, sycl::fabs(f[i])); }
    const float dd = am / 127.f, id = am > 0.f ? 127.f / am : 0.f;
    int32_t sum = 0;
    sycl::vec<int32_t, 8> o;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        int32_t w = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) { const int32_t qi = int32_t(sycl::rint(f[4 * i + k] * id)); sum += qi; w |= (qi & 255) << (8 * k); }
        o[i] = w;
    }
    *reinterpret_cast<sycl::vec<int32_t, 8>*>(q) = o;
    *d = dd; *s = dd * float(sum);
}

// one weight (the embedding lookup)
inline float xg4_w1(const Xg4W& w, size_t row, uint32_t K, uint32_t k) {
    switch (w.fmt) {
    default: return xg4_h2f(w.d[row * (K / 32) + k / 32]) * float(w.qs[row * K + k]);
    case kXg4Q6K: {
        const uint32_t blk = k / 256, p = k % 256, n = p / 128, i = p % 128, qd = i / 32, l = i % 32;
        const uint8_t lo = (uint8_t(w.qs[row * (K / 2) + blk * 128 + n * 64 + (qd & 1) * 32 + l]) >> ((qd >> 1) * 4)) & 15;
        const uint8_t hi = (w.qh[row * (K / 4) + blk * 64 + n * 32 + l] >> (2 * qd)) & 3;
        return xg4_h2f(w.d[row * (K / 256) + blk]) * float(int8_t(w.sc[row * (K / 16) + k / 16])) * float(int(lo | (hi << 4)) - 32);
    }
    case kXg4Q5K:
    case kXg4Q4K: {
        const uint32_t blk = k / 256, p = k % 256, sb = p / 32, l = p % 32;
        uint32_t qv = (uint8_t(w.qs[row * (K / 2) + blk * 128 + (sb / 2) * 32 + l]) >> ((sb & 1) * 4)) & 15;
        if (w.fmt == kXg4Q5K) qv |= ((w.qh[row * (K / 8) + blk * 32 + l] >> sb) & 1) << 4;
        const uint8_t* sm = w.sc + (row * (K / 32) + k / 32) * 2;
        const uint16_t* dd = w.d + (row * (K / 256) + blk) * 2;
        return xg4_h2f(dd[0]) * float(sm[0]) * float(qv) - xg4_h2f(dd[1]) * float(sm[1]);
    }
    }
}

}  // namespace

// ---------------------------------------------------------------------------------------------------- hyper-connections
// Prompt rows (2026-11-20): a work-group per (16 tokens, segment): the 16 tokens' S slice and the 24 fn rows' slice in local
// memory, 256 columns at a time, each lane two of the 400 (row, token) dots -- fn is read once per 16 tokens, not per token
// (the per-token form re-read the 1.4 MB fn for every token: 20 % of a 5,000-token prompt read). Same sums, same order per
// lane as a plain dot over the segment; IE_XG4_HCP_OLD=1 runs the per-token form.
static sycl::event xg4_hc_parts_tiled(sycl::queue& q, const float* S, const float* fn, float* part, uint32_t T, uint32_t HH) {
    constexpr uint32_t P = kXg4HcParts, WG = 256, NM = 24, TB = 16, LS = 256, SP = LS + 1;
    const uint32_t seg = HH / P;
    if (HH % P || seg % LS) throw std::runtime_error("xg4_hc_parts: hc * dim must split into 28 segments of a multiple of 256");
    const uint32_t nblk = (T + TB - 1) / TB;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> Ss(sycl::range<1>(size_t(TB) * SP), h);
        sycl::local_accessor<float, 1> Fs(sycl::range<1>(size_t(NM) * LS), h);
        h.parallel_for<class Xg4HcPartsTK>(sycl::nd_range<1>(size_t(nblk) * P * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const uint32_t g = uint32_t(it.get_group(0)), tb = g / P, p = g % P, l = uint32_t(it.get_local_id(0));
            const uint32_t t0 = tb * TB, nt = sycl::min(TB, T - t0);
            // this lane's dots: pairs u = l and l + 256 of 400 (row m = u / 16, token tt = u % 16; m == 24: the sum of squares)
            const uint32_t u0 = l, u1 = l + WG;
            const uint32_t m0 = u0 / TB, tt0 = u0 % TB, m1 = u1 / TB, tt1 = u1 % TB;
            const bool has1 = u1 < (NM + 1) * TB;
            float v0 = 0.f, v1 = 0.f;
            for (uint32_t c0 = 0; c0 < seg; c0 += LS) {
                const size_t col = size_t(p) * seg + c0;
                for (uint32_t i = l; i < TB * LS; i += WG) {
                    const uint32_t r = i / LS, c = i % LS;
                    Ss[r * SP + c] = r < nt ? S[size_t(t0 + r) * HH + col + c] : 0.f;
                }
                for (uint32_t i = l; i < NM * LS; i += WG) {
                    const uint32_t r = i / LS, c = i % LS;
                    Fs[i] = fn[size_t(r) * HH + col + c];
                }
                sycl::group_barrier(it.get_group());
                {
                    const float* sr = &Ss[tt0 * SP];
                    if (m0 < NM) { const float* fr = &Fs[m0 * LS]; for (uint32_t c = 0; c < LS; ++c) v0 += fr[c] * sr[c]; }
                    else for (uint32_t c = 0; c < LS; ++c) v0 += sr[c] * sr[c];
                }
                if (has1) {
                    const float* sr = &Ss[tt1 * SP];
                    if (m1 < NM) { const float* fr = &Fs[m1 * LS]; for (uint32_t c = 0; c < LS; ++c) v1 += fr[c] * sr[c]; }
                    else for (uint32_t c = 0; c < LS; ++c) v1 += sr[c] * sr[c];
                }
                sycl::group_barrier(it.get_group());
            }
            if (tt0 < nt) part[(size_t(t0 + tt0) * (NM + 1) + m0) * P + p] = v0;
            if (has1 && tt1 < nt) part[(size_t(t0 + tt1) * (NM + 1) + m1) * P + p] = v1;
        });
    });
}

sycl::event xg4_hc_parts(sycl::queue& q, const float* S, const float* fn, float* part, uint32_t T, uint32_t HH) {
    static const bool old = [] { const char* v = std::getenv("IE_XG4_HCP_OLD"); return v && *v == '1'; }();
    if (!old && T >= 16) return xg4_hc_parts_tiled(q, S, fn, part, T, HH);
    // a work-group per (token, mix row, segment): 25 x 28 of them a token, each one contiguous 512-float slice of an fn row
    // (the sum of squares is row 24); S is re-read per row from the L2.
    constexpr uint32_t P = kXg4HcParts, WG = 256, NM = 24;
    const uint32_t seg = HH / P;
    if (HH % P || seg % WG) throw std::runtime_error("xg4_hc_parts: hc * dim must split into 28 segments of a multiple of 256");
    const uint32_t per = seg / WG;
    return q.parallel_for<class Xg4HcPartsK>(sycl::nd_range<1>(size_t(T) * (NM + 1) * P * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        const uint32_t g = uint32_t(it.get_group(0)), t = g / ((NM + 1) * P), r = g % ((NM + 1) * P), m = r / P, p = r % P;
        const uint32_t l = uint32_t(it.get_local_id(0));
        const float* s = S + size_t(t) * HH + size_t(p) * seg;
        float v = 0.f;
        if (m < NM) {
            const float* f = fn + size_t(m) * HH + size_t(p) * seg;
            for (uint32_t i = 0; i < per; ++i) v += f[l + i * WG] * s[l + i * WG];
        } else {
            for (uint32_t i = 0; i < per; ++i) { const float x = s[l + i * WG]; v += x * x; }
        }
        v = sycl::reduce_over_group(it.get_group(), v, sycl::plus<float>());
        if (l == 0) part[(size_t(t) * (NM + 1) + m) * P + p] = v;
    });
}

sycl::event xg4_hc_gate(sycl::queue& q, const float* S, const float* part, const float* base, const float* scale, const float* w,
                        float* post, float* comb, sycl::half* x16, float* x32, uint32_t T, uint32_t H, float norm_eps,
                        float hc_eps, float clamp_lo, float clamp_hi, uint32_t iters, Xg4A a) {
    constexpr uint32_t P = kXg4HcParts, WG = 512, NM = 24, HC = 4;
    const uint32_t HH = HC * H;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> mix(sycl::range<1>(NM + 1), h);
        sycl::local_accessor<float, 1> gate(sycl::range<1>(HC), h);
        sycl::local_accessor<float, 1> col(sycl::range<1>(H), h);
        h.parallel_for<class Xg4HcGateK>(sycl::nd_range<1>(size_t(T) * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const uint32_t t = uint32_t(it.get_group(0)), l = uint32_t(it.get_local_id(0));
            const auto sg = it.get_sub_group();
            if (l <= NM) {
                float a = 0.f;
                for (uint32_t p = 0; p < P; ++p) a += part[(size_t(t) * (NM + 1) + l) * P + p];
                mix[l] = a;
            }
            sycl::group_barrier(it.get_group());
            if (sg.get_group_linear_id() == 0) {                   // lane = entry (i, j) of the 4 x 4 mix
                const uint32_t ln = uint32_t(sg.get_local_linear_id()), i = ln / HC, j = ln % HC;
                const float r = sycl::rsqrt(mix[NM] / float(HH) + norm_eps);
                float v = mix[2 * HC + ln] * r * scale[2] + base[2 * HC + ln];
                v = sycl::fmin(sycl::fmax(v, clamp_lo), clamp_hi);
                // row max over j: lanes of the same row are ln ^ 1, ln ^ 2
                float mx = sycl::fmax(v, sycl::permute_group_by_xor(sg, v, 1));
                mx = sycl::fmax(mx, sycl::permute_group_by_xor(sg, mx, 2));
                float c = sycl::exp(v - mx);
                for (uint32_t it2 = 0; it2 < iters; ++it2) {
                    float rs = c + sycl::permute_group_by_xor(sg, c, 1); rs = rs + sycl::permute_group_by_xor(sg, rs, 2);
                    c = c / (rs + hc_eps);
                    float cs = c + sycl::permute_group_by_xor(sg, c, 4); cs = cs + sycl::permute_group_by_xor(sg, cs, 8);
                    c = c / (cs + hc_eps);
                }
                comb[size_t(t) * HC * HC + ln] = c;
                if (ln < HC) {
                    gate[ln] = xg4_sig(mix[ln] * r * scale[0] + base[ln]);
                    post[size_t(t) * HC + ln] = 2.0f * xg4_sig(mix[HC + ln] * r * scale[1] + base[HC + ln]);
                }
                (void)i; (void)j;
            }
            sycl::group_barrier(it.get_group());
            const float g0 = gate[0], g1 = gate[1], g2 = gate[2], g3 = gate[3];
            const float* s = S + size_t(t) * HH;
            float ss = 0.f;
            for (uint32_t d = l; d < H; d += WG) {
                const float v = g0 * s[d] + g1 * s[H + d] + g2 * s[2 * H + d] + g3 * s[3 * H + d];
                col[d] = v; ss += v * v;
            }
            ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
            const float rr = sycl::rsqrt(ss / float(H) + norm_eps);
            for (uint32_t d = l; d < H; d += WG) {
                const float v = w[d] * (col[d] * rr);
                x16[size_t(t) * H + d] = sycl::half(v);
                if (x32) x32[size_t(t) * H + d] = v;
            }
            if (a.q) {
                sycl::group_barrier(it.get_group());
                for (uint32_t b = l; b < H / 32; b += WG)
                    xg4_qblk(x16 + size_t(t) * H + size_t(b) * 32, a.q + size_t(t) * H + size_t(b) * 32, a.d + size_t(t) * (H / 32) + b, a.s + size_t(t) * (H / 32) + b);
            }
        });
    });
}

sycl::event xg4_hc_post(sycl::queue& q, float* S, const float* post, const float* comb, const float* y, uint32_t T, uint32_t H) {
    return q.parallel_for<class Xg4HcPostK>(sycl::range<1>(size_t(T) * H), [=](sycl::id<1> id) {
        const size_t i = id[0]; const uint32_t t = uint32_t(i / H), d = uint32_t(i % H);
        float* s = S + size_t(t) * 4 * H;
        const float* c = comb + size_t(t) * 16; const float* p = post + size_t(t) * 4;
        const float o0 = s[d], o1 = s[H + d], o2 = s[2 * H + d], o3 = s[3 * H + d], yv = y[i];
        s[d]         = p[0] * yv + c[0] * o0 + c[1] * o1 + c[2] * o2 + c[3] * o3;
        s[H + d]     = p[1] * yv + c[4] * o0 + c[5] * o1 + c[6] * o2 + c[7] * o3;
        s[2 * H + d] = p[2] * yv + c[8] * o0 + c[9] * o1 + c[10] * o2 + c[11] * o3;
        s[3 * H + d] = p[3] * yv + c[12] * o0 + c[13] * o1 + c[14] * o2 + c[15] * o3;
    });
}

sycl::event xg4_stream_mean_norm(sycl::queue& q, const float* S, const int32_t* rows, const float* w, sycl::half* x16, uint32_t n,
                                 uint32_t H, float eps) {
    constexpr uint32_t WG = 256;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> col(sycl::range<1>(H), h);
        h.parallel_for<class Xg4StreamMeanK>(sycl::nd_range<1>(size_t(n) * WG, WG), [=](sycl::nd_item<1> it) {
            const uint32_t j = uint32_t(it.get_group(0)), l = uint32_t(it.get_local_id(0));
            const uint32_t t = rows ? uint32_t(rows[j]) : j;
            const float* s = S + size_t(t) * 4 * H;
            float ss = 0.f;
            for (uint32_t d = l; d < H; d += WG) {
                const float v = (((s[d] + s[H + d]) + s[2 * H + d]) + s[3 * H + d]) * 0.25f;
                col[d] = v; ss += v * v;
            }
            ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
            const float rr = sycl::rsqrt(ss / float(H) + eps);
            for (uint32_t d = l; d < H; d += WG) x16[size_t(j) * H + d] = sycl::half(w[d] * (col[d] * rr));
        });
    });
}

sycl::event xg4_rope_rows(sycl::queue& q, const int32_t* rpos, const float* inv_freq, float* cs, uint32_t T) {
    return q.parallel_for<class Xg4RopeRowsK>(sycl::range<1>(size_t(T) * 32), [=](sycl::id<1> id) {
        const uint32_t t = uint32_t(id[0] / 32), i = uint32_t(id[0] % 32);
        const float a = float(rpos[t]) * inv_freq[i];
        cs[size_t(t) * 64 + i] = sycl::cos(a); cs[size_t(t) * 64 + 32 + i] = sycl::sin(a);
    });
}

sycl::event xg4_embed(sycl::queue& q, const int32_t* tok, const Xg4W& w, float* S, uint32_t T, uint32_t H) {
    return q.parallel_for<class Xg4EmbedK>(sycl::range<1>(size_t(T) * H), [=](sycl::id<1> id) {
        const uint32_t t = uint32_t(id[0] / H), k = uint32_t(id[0] % H);
        const float v = xg4_w1(w, size_t(tok[t]), H, k);
        float* s = S + size_t(t) * 4 * H;
        s[k] = v; s[H + k] = v; s[2 * H + k] = v; s[3 * H + k] = v;
    });
}

sycl::event xg4_act_q8(sycl::queue& q, const sycl::half* x, uint32_t ldx, uint32_t K, uint32_t T, Xg4A a) {
    const uint32_t nb = K / 32;
    return q.parallel_for<class Xg4ActQ8K>(sycl::range<1>(size_t(T) * nb), [=](sycl::id<1> id) {
        const uint32_t t = uint32_t(id[0] / nb), b = uint32_t(id[0] % nb);
        xg4_qblk(x + size_t(t) * ldx + size_t(b) * 32, a.q + size_t(t) * K + size_t(b) * 32, a.d + size_t(t) * nb + b, a.s + size_t(t) * nb + b);
    });
}

// ------------------------------------------------------------------------------------------------------------- Q8 GEMV
namespace {
template <int TR> class Xg4GemvK;
template <int TR>
sycl::event xg4_gemv_t(sycl::queue& q, const sycl::half* x, uint32_t ldx, const Xg4W w, float* y32,
                       sycl::half* y16, uint32_t ldy, uint32_t N, uint32_t K, uint32_t n_mats, uint32_t xms, uint32_t yms, const Xg4A a) {
    constexpr uint32_t RPG = 8, WG = RPG * kXg4SG;              // a row per sub-group, 8 rows a work-group
    const uint32_t gpm = (N + RPG - 1) / RPG, nb = K / 32;
    return q.parallel_for<Xg4GemvK<TR>>(sycl::nd_range<1>(size_t(n_mats) * gpm * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        const uint32_t g = uint32_t(it.get_group(0)), mat = g / gpm, rb = g % gpm;
        const auto sg = it.get_sub_group();
        const uint32_t lane = uint32_t(sg.get_local_linear_id()), n = rb * RPG + uint32_t(sg.get_group_linear_id());
        if (n >= N) return;
        const size_t row = size_t(mat) * N + n;
        const sycl::half* xb = x + size_t(mat) * xms;
        float acc[TR];
#pragma unroll
        for (int t = 0; t < TR; ++t) acc[t] = 0.f;
        if (a.q && w.fmt != kXg4Q8) {
            for (uint32_t b = lane; b < nb; b += kXg4SG)
#pragma unroll
                for (int t = 0; t < TR; ++t)
                    acc[t] += xg4_unit_i8(w, row, K, b, a.q + size_t(t) * K, a.d + size_t(t) * nb, a.s + size_t(t) * nb);
        } else if (w.fmt == kXg4Q8) {
            const int8_t* wq = w.qs + row * K; const uint16_t* wd = w.d + row * nb;
            for (uint32_t b = lane; b < nb; b += kXg4SG) {
                const float sc = xg4_h2f(wd[b]);
#pragma unroll
                for (int t = 0; t < TR; ++t) acc[t] += sc * xg4_blk(wq + size_t(b) * 32, xb + size_t(t) * ldx + size_t(b) * 32);
            }
        } else {
            for (uint32_t b = lane; b < nb; b += kXg4SG)
#pragma unroll
                for (int t = 0; t < TR; ++t) acc[t] += xg4_unit(w, row, K, b, xb + size_t(t) * ldx + size_t(b) * 32);
        }
#pragma unroll
        for (int t = 0; t < TR; ++t) {
            const float v = sycl::reduce_over_group(sg, acc[t], sycl::plus<float>());
            if (lane == 0) {
                const size_t o = size_t(t) * ldy + size_t(mat) * yms + n;
                if (y32) y32[o] = v;
                if (y16) y16[o] = sycl::half(v);
            }
        }
    });
}
}  // namespace

namespace {
template <int NB> class Xg4GemvSmallK;
// K = 32 * NB with NB <= 8: 16 / NB rows a sub-group, a block a lane (no idle lanes for the MLA absorb matrices).
template <int NB>
sycl::event xg4_gemv_small(sycl::queue& q, const sycl::half* x, uint32_t ldx, const Xg4W w, float* y32,
                           sycl::half* y16, uint32_t ldy, uint32_t N, uint32_t T, uint32_t n_mats, uint32_t xms, uint32_t yms) {
    constexpr uint32_t K = 32 * NB, RPS = kXg4SG / NB, NSG = 8, RPG = RPS * NSG, WG = NSG * kXg4SG;
    const uint32_t gpm = (N + RPG - 1) / RPG;
    return q.parallel_for<Xg4GemvSmallK<NB>>(sycl::nd_range<1>(size_t(T) * n_mats * gpm * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        const uint32_t g = uint32_t(it.get_group(0)), t = g / (n_mats * gpm), r = g % (n_mats * gpm), mat = r / gpm, rb = r % gpm;
        const auto sg = it.get_sub_group();
        const uint32_t lane = uint32_t(sg.get_local_linear_id());
        const uint32_t n = rb * RPG + uint32_t(sg.get_group_linear_id()) * RPS + lane / NB, b = lane % NB;
        const bool ok = n < N;
        const size_t row = size_t(mat) * N + (ok ? n : 0);
        float v = xg4_unit(w, row, K, b, x + size_t(t) * ldx + size_t(mat) * xms + size_t(b) * 32);
#pragma unroll
        for (uint32_t o = 1; o < NB; o <<= 1) v += sycl::permute_group_by_xor(sg, v, o);
        if (ok && b == 0) {
            const size_t oi = size_t(t) * ldy + size_t(mat) * yms + n;
            if (y32) y32[oi] = v;
            if (y16) y16[oi] = sycl::half(v);
        }
    });
}
}  // namespace

sycl::event xg4_gemv(sycl::queue& q, const sycl::half* x, uint32_t ldx, const Xg4W& w, float* y32,
                     sycl::half* y16, uint32_t ldy, uint32_t N, uint32_t K, uint32_t T, uint32_t n_mats,
                     uint32_t x_mat_stride, uint32_t y_mat_stride, Xg4A a) {
    if (K % 32 || (w.fmt != kXg4Q8 && K % 256)) throw std::runtime_error("xg4_gemv: K must be a multiple of 32 (256 for K-quants)");
    if (K == 128) return xg4_gemv_small<4>(q, x, ldx, w, y32, y16, ldy, N, T, n_mats, x_mat_stride, y_mat_stride);
    if (K == 256) return xg4_gemv_small<8>(q, x, ldx, w, y32, y16, ldy, N, T, n_mats, x_mat_stride, y_mat_stride);
    if (a.q && n_mats != 1) throw std::runtime_error("xg4_gemv: int8 activations with n_mats 1 only");
    sycl::event e;
    for (uint32_t t0 = 0; t0 < T; t0 += 8) {
        const uint32_t n = std::min<uint32_t>(8, T - t0);
        const sycl::half* xx = x + size_t(t0) * ldx;
        Xg4A aa = a;
        if (a.q) { aa.q += size_t(t0) * K; aa.d += size_t(t0) * (K / 32); aa.s += size_t(t0) * (K / 32); }
        Xg4A a4 = aa, a2 = aa;
        if (a.q) { a4.q += size_t(4) * K; a4.d += size_t(4) * (K / 32); a4.s += size_t(4) * (K / 32);
                   a2.q += size_t(2) * K; a2.d += size_t(2) * (K / 32); a2.s += size_t(2) * (K / 32); }
        float* a_ = y32 ? y32 + size_t(t0) * ldy : nullptr; sycl::half* b = y16 ? y16 + size_t(t0) * ldy : nullptr;
        if (n == 8)      e = xg4_gemv_t<8>(q, xx, ldx, w, a_, b, ldy, N, K, n_mats, x_mat_stride, y_mat_stride, aa);
        else if (n >= 4) { e = xg4_gemv_t<4>(q, xx, ldx, w, a_, b, ldy, N, K, n_mats, x_mat_stride, y_mat_stride, aa);
                           if (n > 4) e = xg4_gemv(q, xx + size_t(4) * ldx, ldx, w, a_ ? a_ + size_t(4) * ldy : nullptr, b ? b + size_t(4) * ldy : nullptr, ldy, N, K, n - 4, n_mats, x_mat_stride, y_mat_stride, a4); }
        else if (n >= 2) { e = xg4_gemv_t<2>(q, xx, ldx, w, a_, b, ldy, N, K, n_mats, x_mat_stride, y_mat_stride, aa);
                           if (n > 2) e = xg4_gemv_t<1>(q, xx + size_t(2) * ldx, ldx, w, a_ ? a_ + size_t(2) * ldy : nullptr, b ? b + size_t(2) * ldy : nullptr, ldy, N, K, n_mats, x_mat_stride, y_mat_stride, a2); }
        else             e = xg4_gemv_t<1>(q, xx, ldx, w, a_, b, ldy, N, K, n_mats, x_mat_stride, y_mat_stride, aa);
    }
    return e;
}

sycl::event xg4_rmsnorm16(sycl::queue& q, const float* x, uint32_t ld, const float* w, sycl::half* y, uint32_t ld16,
                          uint32_t T, uint32_t n, float eps, Xg4A a) {
    constexpr uint32_t WG = 256;
    return q.parallel_for<class Xg4RmsNormK>(sycl::nd_range<1>(size_t(T) * WG, WG), [=](sycl::nd_item<1> it) {
        const uint32_t t = uint32_t(it.get_group(0)), l = uint32_t(it.get_local_id(0));
        const float* r = x + size_t(t) * ld;
        float ss = 0.f;
        for (uint32_t i = l; i < n; i += WG) ss += r[i] * r[i];
        ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
        const float rr = sycl::rsqrt(ss / float(n) + eps);
        for (uint32_t i = l; i < n; i += WG) y[size_t(t) * ld16 + i] = sycl::half(w[i] * (r[i] * rr));
        if (a.q) {
            sycl::group_barrier(it.get_group());
            for (uint32_t b = l; b < n / 32; b += WG)
                xg4_qblk(y + size_t(t) * ld16 + size_t(b) * 32, a.q + size_t(t) * n + size_t(b) * 32, a.d + size_t(t) * (n / 32) + b, a.s + size_t(t) * (n / 32) + b);
        }
    });
}

// ------------------------------------------------------------------------------------------------------------------ MLA
sycl::event xg4_kv_append(sycl::queue& q, const float* kv, uint32_t ldkv, const float* w, const float* cs, sycl::half* pool,
                          const int32_t* tab, uint32_t maxp, const int32_t* rlane, const int32_t* rpos, uint32_t T, float eps) {
    constexpr uint32_t WG = 256, KL = 512, RD = 64, LW = KL + RD;
    return q.parallel_for<class Xg4KvAppendK>(sycl::nd_range<1>(size_t(T) * WG, WG), [=](sycl::nd_item<1> it) {
        const uint32_t t = uint32_t(it.get_group(0)), l = uint32_t(it.get_local_id(0));
        const float* r = kv + size_t(t) * ldkv;
        float ss = 0.f;
        for (uint32_t i = l; i < KL; i += WG) ss += r[i] * r[i];
        ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
        const float rr = sycl::rsqrt(ss / float(KL) + eps);
        const uint32_t p = uint32_t(rpos[t]);
        const size_t slot = size_t(tab[size_t(rlane[t]) * maxp + p / kXg4Page]) * kXg4Page + p % kXg4Page;
        sycl::half* o = pool + slot * LW;
        for (uint32_t i = l; i < KL; i += WG) o[i] = sycl::half(w[i] * (r[i] * rr));
        if (l < RD / 2) {
            const float c = cs[size_t(t) * RD + l], s = cs[size_t(t) * RD + RD / 2 + l];
            const float x0 = r[KL + 2 * l], x1 = r[KL + 2 * l + 1];
            o[KL + l] = sycl::half(x0 * c - x1 * s);
            o[KL + RD / 2 + l] = sycl::half(x1 * c + x0 * s);
        }
    });
}

sycl::event xg4_q_split(sycl::queue& q, const float* qf, const float* cs, sycl::half* qn16, sycl::half* a16, uint32_t T, uint32_t NH) {
    constexpr uint32_t ND = 128, RD = 64, QD = ND + RD, LW = 576;
    return q.parallel_for<class Xg4QSplitK>(sycl::range<1>(size_t(T) * NH * (ND + RD / 2)), [=](sycl::id<1> id) {
        const size_t i = id[0]; const uint32_t th = uint32_t(i / (ND + RD / 2)), k = uint32_t(i % (ND + RD / 2)), t = th / NH;
        const float* r = qf + size_t(th) * QD;
        if (k < ND) { qn16[size_t(th) * ND + k] = sycl::half(r[k]); return; }
        const uint32_t p = k - ND;
        const float c = cs[size_t(t) * RD + p], s = cs[size_t(t) * RD + RD / 2 + p];
        const float x0 = r[ND + 2 * p], x1 = r[ND + 2 * p + 1];
        a16[size_t(th) * LW + 512 + p] = sycl::half(x0 * c - x1 * s);
        a16[size_t(th) * LW + 512 + RD / 2 + p] = sycl::half(x1 * c + x0 * s);
    });
}

// ------------------------------------------------------------------------------------------------------------------ MoE
sycl::event xg4_router(sycl::queue& q, const float* x32, const float* wg, const float* bias, int32_t* job_e, float* job_w,
                       uint32_t T, uint32_t H, uint32_t E, uint32_t k, bool norm, float scale, float* logit_ws) {
    // logits: a sub-group an expert, 8 experts a work-group (fp32 weights and activations: the routing is near-tie sensitive)
    constexpr uint32_t RPG = 8, WG = RPG * kXg4SG;
    if (E > 256 || k > 16) throw std::runtime_error("xg4_router: at most 256 experts and top-16");
    const uint32_t gpt = (E + RPG - 1) / RPG;
    q.parallel_for<class Xg4RouterLogitsK>(sycl::nd_range<1>(size_t(T) * gpt * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        const uint32_t g = uint32_t(it.get_group(0)), t = g / gpt;
        const auto sg = it.get_sub_group();
        const uint32_t lane = uint32_t(sg.get_local_linear_id()), e = (g % gpt) * RPG + uint32_t(sg.get_group_linear_id());
        if (e >= E) return;
        const sycl::vec<float, 4>* w = reinterpret_cast<const sycl::vec<float, 4>*>(wg + size_t(e) * H);
        const sycl::vec<float, 4>* x = reinterpret_cast<const sycl::vec<float, 4>*>(x32 + size_t(t) * H);
        float a = 0.f;
        for (uint32_t i = lane; i < H / 4; i += kXg4SG) { const auto wv = w[i], xv = x[i]; a += wv[0] * xv[0] + wv[1] * xv[1] + wv[2] * xv[2] + wv[3] * xv[3]; }
        a = sycl::reduce_over_group(sg, a, sycl::plus<float>());
        if (lane == 0) logit_ws[size_t(t) * E + e] = a;
    });
    // selection: one sub-group a token, each lane holding E / 16 experts; k rounds of a sub-group arg-max
    if (E % kXg4SG) throw std::runtime_error("xg4_router: experts must be a multiple of 16");
    return q.parallel_for<class Xg4RouterTopkK>(sycl::nd_range<1>(size_t(T) * kXg4SG, kXg4SG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        const uint32_t t = uint32_t(it.get_group(0));
        const auto sg = it.get_sub_group();
        const uint32_t lane = uint32_t(sg.get_local_linear_id()), per = E / kXg4SG;
        float sc[16], key[16];
        for (uint32_t i = 0; i < per && i < 16; ++i) {
            const uint32_t e = lane * per + i;
            sc[i] = xg4_sig(logit_ws[size_t(t) * E + e]); key[i] = sc[i] + (bias ? bias[e] : 0.f);
        }
        float sum = 0.f, wsel[16]; int32_t esel[16];
        for (uint32_t j = 0; j < k; ++j) {
            float bv = -3.0e38f; int32_t bi = 0x7fffffff;
            for (uint32_t i = 0; i < per && i < 16; ++i) if (key[i] > bv) { bv = key[i]; bi = int32_t(lane * per + i); }
            const float mv = sycl::reduce_over_group(sg, bv, sycl::maximum<float>());
            const int32_t me = sycl::reduce_over_group(sg, bv == mv ? bi : 0x7fffffff, sycl::minimum<int32_t>());
            const uint32_t owner = uint32_t(me) / per, slot = uint32_t(me) % per;
            float wv = 0.f;
            if (lane == owner) { wv = sc[slot]; key[slot] = -3.0e38f; }
            wv = sycl::group_broadcast(sg, wv, owner);
            esel[j] = me; wsel[j] = wv; sum += wv;
        }
        if (lane == 0) {
            const float den = norm ? sum + 1e-20f : 1.f;
            for (uint32_t j = 0; j < k; ++j) { job_e[size_t(t) * (k + 1) + j] = esel[j]; job_w[size_t(t) * (k + 1) + j] = wsel[j] / den * scale; }
            job_e[size_t(t) * (k + 1) + k] = -1; job_w[size_t(t) * (k + 1) + k] = 1.f;
        }
    });
}

sycl::event xg4_moe_gate_up(sycl::queue& q, const sycl::half* x, const Xg4W& g_, const Xg4W& u_, const Xg4W& gs_, const Xg4W& us_,
                            const int32_t* job_e, float* h, uint32_t T, uint32_t per_tok,
                            uint32_t N, uint32_t K, Xg4A a) {
    constexpr uint32_t RPG = 8, WG = RPG * kXg4SG;
    const uint32_t gpj = (N + RPG - 1) / RPG, nb = K / 32, J = T * per_tok;
    const Xg4W wg = g_, wu = u_, wgs = gs_, wus = us_;
    return q.parallel_for<class Xg4MoeGateUpK>(sycl::nd_range<1>(size_t(J) * gpj * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        const uint32_t g = uint32_t(it.get_group(0)), j = g / gpj, rb = g % gpj;
        const auto sg = it.get_sub_group();
        const uint32_t lane = uint32_t(sg.get_local_linear_id()), r = rb * RPG + uint32_t(sg.get_group_linear_id());
        if (r >= N) return;
        const int32_t e = job_e ? job_e[j] : -1;
        const Xg4W gw = e >= 0 ? wg : wgs, uw = e >= 0 ? wu : wus;
        const size_t row = e >= 0 ? size_t(e) * N + r : size_t(r);
        const sycl::half* xr = x + size_t(j / per_tok) * K;
        float ag = 0.f, au = 0.f;
        if (a.q && gw.fmt != kXg4Q8) {
            const size_t tt = j / per_tok;
            for (uint32_t b = lane; b < nb; b += kXg4SG) {
                const int8_t* xq = a.q + tt * K; const float* xd = a.d + tt * nb; const float* xs = a.s + tt * nb;
                ag += xg4_unit_i8(gw, row, K, b, xq, xd, xs);
                au += xg4_unit_i8(uw, row, K, b, xq, xd, xs);
            }
        } else {
            for (uint32_t b = lane; b < nb; b += kXg4SG) {
                ag += xg4_unit(gw, row, K, b, xr + size_t(b) * 32);
                au += xg4_unit(uw, row, K, b, xr + size_t(b) * 32);
            }
        }
        ag = sycl::reduce_over_group(sg, ag, sycl::plus<float>());
        au = sycl::reduce_over_group(sg, au, sycl::plus<float>());
        if (lane == 0) h[size_t(j) * N + r] = ag / (1.0f + sycl::exp(-ag)) * au;
    });
}

sycl::event xg4_moe_down(sycl::queue& q, const sycl::half* hh, const float* hs, const Xg4W& dn_, const Xg4W& ds_, const int32_t* job_e, const float* job_w, float* y, uint32_t T, uint32_t per_tok,
                         uint32_t N, uint32_t K, Xg4A a) {
    constexpr uint32_t RPG = 8, WG = RPG * kXg4SG;
    const uint32_t gpt = (N + RPG - 1) / RPG, nb = K / 32;
    const Xg4W dn = dn_, ds = ds_;
    const Xg4A ax = a;
    return q.parallel_for<class Xg4MoeDownK>(sycl::nd_range<1>(size_t(T) * gpt * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        const uint32_t g = uint32_t(it.get_group(0)), t = g / gpt, rb = g % gpt;
        const auto sg = it.get_sub_group();
        const uint32_t lane = uint32_t(sg.get_local_linear_id()), n = rb * RPG + uint32_t(sg.get_group_linear_id());
        if (n >= N) return;
        float acc = 0.f;
        for (uint32_t s = 0; s < per_tok; ++s) {
            const uint32_t j = t * per_tok + s;
            const int32_t e = job_e ? job_e[j] : -1;
            const Xg4W ww = e >= 0 ? dn : ds;
            const size_t row = e >= 0 ? size_t(e) * N + n : size_t(n);
            const sycl::half* xr = hh + size_t(j) * K;
            float a = 0.f;
            if (ax.q && ww.fmt != kXg4Q8) for (uint32_t b = lane; b < nb; b += kXg4SG) a += xg4_unit_i8(ww, row, K, b, ax.q + size_t(j) * K, ax.d + size_t(j) * nb, ax.s + size_t(j) * nb);
            else for (uint32_t b = lane; b < nb; b += kXg4SG) a += xg4_unit(ww, row, K, b, xr + size_t(b) * 32);
            acc += (job_w ? job_w[j] : 1.f) * (hs ? hs[j] : 1.f) * a;
        }
        acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
        if (lane == 0) y[size_t(t) * N + n] = acc;
    });
}

// ------------------------------------------------------------------------------------------------- MoE, expert-grouped
sycl::event xg4_moe_group(sycl::queue& q, const int32_t* job_e, uint32_t J, uint32_t E, int32_t* off, int32_t* list) {
    constexpr uint32_t WG = 256;
    if (E + 1 > 512) throw std::runtime_error("xg4_moe_group: too many experts");
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<int32_t, 1> cnt(sycl::range<1>(512), h), cur(sycl::range<1>(512), h);
        h.parallel_for<class Xg4MoeGroupK>(sycl::nd_range<1>(WG, WG), [=](sycl::nd_item<1> it) {
            const uint32_t l = uint32_t(it.get_local_id(0));
            for (uint32_t e = l; e <= E; e += WG) cnt[e] = 0;
            sycl::group_barrier(it.get_group());
            for (uint32_t j = l; j < J; j += WG) {
                const int32_t e = job_e[j]; const uint32_t g = e < 0 ? E : uint32_t(e);
                sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space>(cnt[g]).fetch_add(1);
            }
            sycl::group_barrier(it.get_group());
            if (l == 0) { int32_t a = 0; for (uint32_t e = 0; e <= E; ++e) { off[e] = a; cur[e] = a; a += cnt[e]; } off[E + 1] = a; }
            sycl::group_barrier(it.get_group());
            for (uint32_t j = l; j < J; j += WG) {
                const int32_t e = job_e[j]; const uint32_t g = e < 0 ? E : uint32_t(e);
                const int32_t at = sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space>(cur[g]).fetch_add(1);
                list[at] = int32_t(j);
            }
        });
    });
}

namespace {
template <int TR> class Xg4MoeGuGrpK;
template <int TR> class Xg4MoeDnGrpK;
}  // namespace

// Group g < E: expert g's jobs list[off[g] .. off[g + 1]); group E: the shared matrices. job_e null (a dense layer): one
// group, the shared matrices, jobs 0 .. J. A work-group per (group, 8 output rows); the group's jobs TR at a time share
// every weight load.
template <int TR>
static sycl::event xg4_moe_gate_up_grp_t(sycl::queue& q, const sycl::half* x, const Xg4W g_, const Xg4W u_, const Xg4W gs_, const Xg4W us_,
                                         const Xg4A ax, const int32_t* off, const int32_t* list, float* h, uint32_t J,
                                         uint32_t per_tok, uint32_t N, uint32_t K, uint32_t E, bool dense) {
    constexpr uint32_t RPG = 8, WG = RPG * kXg4SG;
    const uint32_t gpe = (N + RPG - 1) / RPG, nb = K / 32, NG = dense ? 1 : E + 1;
    return q.parallel_for<Xg4MoeGuGrpK<TR>>(sycl::nd_range<1>(size_t(NG) * gpe * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        const uint32_t g = uint32_t(it.get_group(0)) / gpe, rb = uint32_t(it.get_group(0)) % gpe;
        const auto sg = it.get_sub_group();
        const uint32_t lane = uint32_t(sg.get_local_linear_id()), r = rb * RPG + uint32_t(sg.get_group_linear_id());
        if (r >= N) return;
        const int32_t j0 = dense ? 0 : off[g], j1 = dense ? int32_t(J) : off[g + 1];
        if (j0 >= j1) return;
        const bool shared = dense || g == E;
        const size_t row = shared ? size_t(r) : size_t(g) * N + r;
        const Xg4W gw = shared ? gs_ : g_, uw = shared ? us_ : u_;
        for (int32_t c0 = j0; c0 < j1; c0 += TR) {
            const int32_t n = sycl::min(int32_t(TR), j1 - c0);
            int32_t jj[TR]; float ag[TR], au[TR];
#pragma unroll
            for (int i = 0; i < TR; ++i) { jj[i] = i < n ? (dense ? c0 + i : list[c0 + i]) : -1; ag[i] = 0.f; au[i] = 0.f; }
            if (ax.q && gw.fmt != kXg4Q8) {
                for (uint32_t b = lane; b < nb; b += kXg4SG) {
#pragma unroll
                    for (int i = 0; i < TR; ++i) {
                        if (i >= n) break;
                        const size_t tt = size_t(jj[i] / int32_t(per_tok));
                        const int8_t* xq = ax.q + tt * K; const float* xd = ax.d + tt * nb; const float* xs = ax.s + tt * nb;
                        ag[i] += xg4_unit_i8(gw, row, K, b, xq, xd, xs);
                        au[i] += xg4_unit_i8(uw, row, K, b, xq, xd, xs);
                    }
                }
            } else if (gw.fmt == kXg4Q8 && uw.fmt == kXg4Q8) {
                const int8_t* gq = gw.qs + row * K; const int8_t* uq = uw.qs + row * K;
                const uint16_t* gd = gw.d + row * nb; const uint16_t* ud = uw.d + row * nb;
                for (uint32_t b = lane; b < nb; b += kXg4SG) {
                    const float sg_ = xg4_h2f(gd[b]), su = xg4_h2f(ud[b]);
#pragma unroll
                    for (int i = 0; i < TR; ++i) {
                        if (i >= n) break;
                        const sycl::half* xr = x + size_t(jj[i] / int32_t(per_tok)) * K + size_t(b) * 32;
                        ag[i] += sg_ * xg4_blk(gq + size_t(b) * 32, xr);
                        au[i] += su * xg4_blk(uq + size_t(b) * 32, xr);
                    }
                }
            } else {
                for (uint32_t b = lane; b < nb; b += kXg4SG) {
#pragma unroll
                    for (int i = 0; i < TR; ++i) {
                        if (i >= n) break;
                        const sycl::half* xr = x + size_t(jj[i] / int32_t(per_tok)) * K + size_t(b) * 32;
                        ag[i] += xg4_unit(gw, row, K, b, xr);
                        au[i] += xg4_unit(uw, row, K, b, xr);
                    }
                }
            }
#pragma unroll
            for (int i = 0; i < TR; ++i) {
                if (i >= n) break;
                const float a = sycl::reduce_over_group(sg, ag[i], sycl::plus<float>()), u = sycl::reduce_over_group(sg, au[i], sycl::plus<float>());
                if (lane == 0) h[size_t(jj[i]) * N + r] = a / (1.0f + sycl::exp(-a)) * u;
            }
        }
    });
}

template <int TR>
static sycl::event xg4_moe_down_grp_t(sycl::queue& q, const sycl::half* hh, const float* hs, const Xg4W dn_, const Xg4W ds_, const Xg4A ax, const int32_t* off, const int32_t* list, const float* job_w, float* yj,
                                      uint32_t J, uint32_t N, uint32_t K, uint32_t E, bool dense) {
    constexpr uint32_t RPG = 8, WG = RPG * kXg4SG;
    const uint32_t gpe = (N + RPG - 1) / RPG, nb = K / 32, NG = dense ? 1 : E + 1;
    return q.parallel_for<Xg4MoeDnGrpK<TR>>(sycl::nd_range<1>(size_t(NG) * gpe * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        const uint32_t g = uint32_t(it.get_group(0)) / gpe, rb = uint32_t(it.get_group(0)) % gpe;
        const auto sg = it.get_sub_group();
        const uint32_t lane = uint32_t(sg.get_local_linear_id()), n_ = rb * RPG + uint32_t(sg.get_group_linear_id());
        if (n_ >= N) return;
        const int32_t j0 = dense ? 0 : off[g], j1 = dense ? int32_t(J) : off[g + 1];
        if (j0 >= j1) return;
        const bool shared = dense || g == E;
        const size_t row = shared ? size_t(n_) : size_t(g) * N + n_;
        const Xg4W ww = shared ? ds_ : dn_;
        for (int32_t c0 = j0; c0 < j1; c0 += TR) {
            const int32_t n = sycl::min(int32_t(TR), j1 - c0);
            int32_t jj[TR]; float a[TR];
#pragma unroll
            for (int i = 0; i < TR; ++i) { jj[i] = i < n ? (dense ? c0 + i : list[c0 + i]) : -1; a[i] = 0.f; }
            if (ax.q && ww.fmt != kXg4Q8) {
                for (uint32_t b = lane; b < nb; b += kXg4SG)
#pragma unroll
                    for (int i = 0; i < TR; ++i) {
                        if (i >= n) break;
                        const size_t jr = size_t(jj[i]);
                        a[i] += xg4_unit_i8(ww, row, K, b, ax.q + jr * K, ax.d + jr * nb, ax.s + jr * nb);
                    }
            } else if (ww.fmt == kXg4Q8) {
                const int8_t* wq = ww.qs + row * K; const uint16_t* wd = ww.d + row * nb;
                for (uint32_t b = lane; b < nb; b += kXg4SG) {
                    const float sc = xg4_h2f(wd[b]);
#pragma unroll
                    for (int i = 0; i < TR; ++i) { if (i >= n) break; a[i] += sc * xg4_blk(wq + size_t(b) * 32, hh + size_t(jj[i]) * K + size_t(b) * 32); }
                }
            } else {
                for (uint32_t b = lane; b < nb; b += kXg4SG)
#pragma unroll
                    for (int i = 0; i < TR; ++i) { if (i >= n) break; a[i] += xg4_unit(ww, row, K, b, hh + size_t(jj[i]) * K + size_t(b) * 32); }
            }
#pragma unroll
            for (int i = 0; i < TR; ++i) {
                if (i >= n) break;
                const float v = sycl::reduce_over_group(sg, a[i], sycl::plus<float>());
                if (lane == 0) yj[size_t(jj[i]) * N + n_] = (job_w && !dense ? job_w[jj[i]] : 1.f) * (hs ? hs[jj[i]] : 1.f) * v;
            }
        }
    });
}

sycl::event xg4_moe_gate_up_grp(sycl::queue& q, const sycl::half* x, const Xg4W& g, const Xg4W& u, const Xg4W& gs, const Xg4W& us,
                                const int32_t* off, const int32_t* list, float* h, uint32_t J,
                                uint32_t per_tok, uint32_t N, uint32_t K, uint32_t E, bool dense, Xg4A a) {
    return xg4_moe_gate_up_grp_t<8>(q, x, g, u, gs, us, a, off, list, h, J, per_tok, N, K, E, dense);
}
sycl::event xg4_moe_down_grp(sycl::queue& q, const sycl::half* hh, const float* hs, const Xg4W& dn, const Xg4W& ds, const int32_t* off,
                             const int32_t* list, const float* job_w, float* yj,
                             uint32_t J, uint32_t N, uint32_t K, uint32_t E, bool dense, Xg4A a) {
    return xg4_moe_down_grp_t<8>(q, hh, hs, dn, ds, a, off, list, job_w, yj, J, N, K, E, dense);
}
// h (fp32, J rows of N) -> fp16 for the down projection. A row whose magnitude would overflow fp16 (Xing4's <_end> token
// drives the expert hidden past 65504 in layer 29: inf, then NaN through the cache) is stored as h / 2^k with hs = 2^k,
// which the down kernels multiply back in; every other row has hs = 1 and is converted exactly as before.
sycl::event xg4_h_pack(sycl::queue& q, const float* h, sycl::half* h16, float* hs, uint32_t J, uint32_t N) {
    constexpr uint32_t WG = 256;
    return q.parallel_for<class Xg4HPackK>(sycl::nd_range<1>(size_t(J) * WG, WG), [=](sycl::nd_item<1> it) {
        const uint32_t j = uint32_t(it.get_group(0)), l = uint32_t(it.get_local_id(0));
        const float* r = h + size_t(j) * N;
        float m = 0.f;
        for (uint32_t i = l; i < N; i += WG) m = sycl::fmax(m, sycl::fabs(r[i]));
        m = sycl::reduce_over_group(it.get_group(), m, sycl::maximum<float>());
        float s = 1.f;
        while (m / s > 16384.f && s < 1e30f) s *= 2.f;   // a power of two: the rescale itself is exact
        const float inv = 1.f / s;
        for (uint32_t i = l; i < N; i += WG) h16[size_t(j) * N + i] = sycl::half(r[i] * inv);
        if (l == 0) hs[j] = s;
    });
}

sycl::event xg4_moe_sum(sycl::queue& q, const float* yj, float* y, uint32_t T, uint32_t per_tok, uint32_t N) {
    return q.parallel_for<class Xg4MoeSumK>(sycl::range<1>(size_t(T) * N), [=](sycl::id<1> id) {
        const size_t i = id[0]; const uint32_t t = uint32_t(i / N), n = uint32_t(i % N);
        float a = 0.f;
        for (uint32_t s = 0; s < per_tok; ++s) a += yj[(size_t(t) * per_tok + s) * N + n];
        y[i] = a;
    });
}

}  // namespace ie
