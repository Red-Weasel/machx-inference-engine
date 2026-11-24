// src/model/hy4flash_forward.cpp — Hy4-flash device forward, bring-up form. See include/ie/hy4flash_forward.hpp and
// docs/hy4flash/00_PORT_AND_ABLITERATION_PLAN.md. Every kernel here is a plain one (correctness first); the GEMMs are
// oneDNN's. Kernel names are unique to this file (the SYCL kernel-name collision trap).
#include "ie/hy4flash_forward.hpp"
#include "ie/kernel_profiler.hpp"

#include "ie/mimo26_ops.hpp"
#include "ie/ops.hpp"

#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <algorithm>
#include <sys/mman.h>
#include <unistd.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>

namespace ie {

namespace {
// Every launch of this file is tagged for the engine's kernel profiler (ie::g_profiler; ie-hy4flash-run --kprofile): the
// same call on the queue, plus its event pushed under a name when a profiler is installed. A pointer compare otherwise.
struct KpQ {
    sycl::queue& q; const char* name;
    template <class K, class... A> sycl::event parallel_for(A&&... a) { sycl::event e = q.parallel_for<K>(std::forward<A>(a)...); if (g_profiler) [[unlikely]] g_profiler->push(name, e, q); return e; }
    template <class... A> sycl::event memcpy(A&&... a) { sycl::event e = q.memcpy(std::forward<A>(a)...); if (g_profiler) [[unlikely]] g_profiler->push(name, e, q); return e; }
    template <class... A> sycl::event fill(A&&... a) { sycl::event e = q.fill(std::forward<A>(a)...); if (g_profiler) [[unlikely]] g_profiler->push(name, e, q); return e; }
};
// IE_HY4F_SMALLK=0: the first versions of the small per-token kernels (one work-item walking a whole row), for comparison.
// The kernel profile of 2026-11-19 had them at 38 us (key append), 364 us (n-gram statistics), 34 us (softmax) a call.
bool smallk() { static const bool on = [] { const char* v = std::getenv("IE_HY4F_SMALLK"); return !(v && *v == '0'); }(); return on; }


double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// IE_HY4F_PLANT=<name> (diagnostic): one deliberate arithmetic fault, to show the mini gate fails on it.
const std::string& plant() {
    static const std::string p = [] { const char* v = std::getenv("IE_HY4F_PLANT"); return std::string(v ? v : ""); }();
    return p;
}

// score / probability workspace of one attention row block. IE_HY4F_SCORE_ELEMS (diagnostic) shrinks it so a small test
// runs the multi-block loop the long real prompts take.
uint64_t score_elems() {
    static const uint64_t n = [] { const char* v = std::getenv("IE_HY4F_SCORE_ELEMS"); const long long x = v ? std::atoll(v) : 0; return x > 0 ? uint64_t(x) : (32ull << 20); }();
    return n;
}

inline float dsig(float x) { return 1.0f / (1.0f + sycl::exp(-x)); }

// mixes[t, m] = fn[m] . flat[t] for m < n_mix, mixes[t, n_mix] = mean(flat[t]^2); flat = the hc streams of a token.
// Two launches: 64 partial sums per (token, mix) spread over the card, then their sum in a fixed order. (One work-item
// walking all hc * dim elements took 0.23 ms a call -- 21 % of a decode token on the real model, profile of 2026-11-18.)
constexpr uint32_t kHcParts = 64;
sycl::event hc_mixes(sycl::queue& q, const float* S, const float* fn, float* mixes, float* part, uint32_t T, uint32_t n_mix, uint32_t HH, bool sum = true) {
    const uint32_t M1 = n_mix + 1, seg = HH / kHcParts;       // HH = hc * dim is a multiple of 64 for every supported shape
    const sycl::event parts = KpQ{q, "Hy4fHcMixPartK"}.parallel_for<class Hy4fHcMixPartK>(sycl::range<2>(size_t(T) * M1, kHcParts), [=](sycl::id<2> i) {
        const size_t tm = i[0]; const uint32_t t = uint32_t(tm / M1), m = uint32_t(tm % M1), p = uint32_t(i[1]);
        const float* f = S + size_t(t) * HH + size_t(p) * seg;
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        if (m < n_mix) {
            const float* w = fn + size_t(m) * HH + size_t(p) * seg;
            for (uint32_t d = 0; d + 3 < seg; d += 4) { a0 += w[d] * f[d]; a1 += w[d + 1] * f[d + 1]; a2 += w[d + 2] * f[d + 2]; a3 += w[d + 3] * f[d + 3]; }
        } else {
            for (uint32_t d = 0; d + 3 < seg; d += 4) { a0 += f[d] * f[d]; a1 += f[d + 1] * f[d + 1]; a2 += f[d + 2] * f[d + 2]; a3 += f[d + 3] * f[d + 3]; }
        }
        part[tm * kHcParts + p] = (a0 + a1) + (a2 + a3);
    });
    if (!sum) return parts;                 // hc_apply sums the parts itself (one launch less a step)
    return KpQ{q, "Hy4fHcMixSumK"}.parallel_for<class Hy4fHcMixSumK>(sycl::range<1>(size_t(T) * M1), [=](sycl::id<1> i) {
        const size_t tm = i[0];
        float a = 0.f;
        for (uint32_t p = 0; p < kHcParts; ++p) a += part[tm * kHcParts + p];
        mixes[tm] = (tm % M1) < n_mix ? a : a / float(HH);
    });
}
// pre[t, i] = sigmoid(mix_i * r * scale[0] + base[i]) + hc_eps; post[t, i] = mag * sigmoid(mix_{hc+i} * r * scale[1] + base[hc+i]) + hc_eps
// (post may be null: the head's collapse has no post); r = rsqrt(mean(flat^2) + eps). Then x[t, d] = sum_i pre[t, i] * S[t, i, d].
sycl::event hc_apply(sycl::queue& q, const float* S, const float* mixes, const float* base, const float* scale, float* pre, float* post,
                     float* x, uint32_t T, uint32_t hc, uint32_t H, float eps, float hc_eps, float mag, const float* part = nullptr, uint32_t HH = 0) {
    const uint32_t n_mix = post ? 2 * hc : hc;
    KpQ{q, "Hy4fHcGatesK"}.parallel_for<class Hy4fHcGatesK>(sycl::range<1>(T), [=](sycl::id<1> ti) {
        const uint32_t t = uint32_t(ti[0]);
        float mloc[17];                                            // (hc <= 8, checked at init)
        if (part) {                                                // hc_mixes left its partial sums: the same sums, in the same order
            for (uint32_t m = 0; m <= n_mix; ++m) {
                float a = 0.f;
                for (uint32_t pp = 0; pp < kHcParts; ++pp) a += part[(size_t(t) * (n_mix + 1) + m) * kHcParts + pp];
                mloc[m] = m < n_mix ? a : a / float(HH);
            }
        }
        const float* mx = part ? mloc : mixes + size_t(t) * (n_mix + 1);
        const float r = sycl::rsqrt(mx[n_mix] + eps);
        for (uint32_t i = 0; i < hc; ++i) {
            pre[size_t(t) * hc + i] = dsig(mx[i] * r * scale[0] + base[i]) + hc_eps;
            if (post) post[size_t(t) * hc + i] = mag * dsig(mx[hc + i] * r * scale[1] + base[hc + i]) + hc_eps;
        }
    });
    return KpQ{q, "Hy4fHcCollapseK"}.parallel_for<class Hy4fHcCollapseK>(sycl::range<2>(T, H), [=](sycl::id<2> i) {
        const uint32_t t = uint32_t(i[0]), d = uint32_t(i[1]);
        float a = 0.f;
        for (uint32_t s = 0; s < hc; ++s) a += pre[size_t(t) * hc + s] * S[(size_t(t) * hc + s) * H + d];
        x[size_t(t) * H + d] = a;
    });
}
// S[t, i, d] += post[t, i] * y[t, d]
sycl::event post_add(sycl::queue& q, float* S, const float* post, const float* y, uint32_t T, uint32_t hc, uint32_t H) {
    return KpQ{q, "Hy4fPostAddK"}.parallel_for<class Hy4fPostAddK>(sycl::range<2>(size_t(T) * hc, H), [=](sycl::id<2> i) {
        const size_t r = i[0]; const uint32_t d = uint32_t(i[1]);
        S[r * H + d] += post[r] * y[(r / hc) * H + d];
    });
}
// q [T, NH, nope + rope] fp32 -> qn [T * NH, nope] fp16 and a [T * NH, lw] fp16 columns [kl, kl + rope) = the rope slice rotated by
// the row's cos / sin (cs = [T, rope]: rope / 2 cosines then rope / 2 sines).
sycl::event q_split(sycl::queue& q, const float* q32, const float* cs, sycl::half* qn, sycl::half* a, uint32_t T, uint32_t NH,
                    uint32_t nope, uint32_t rope, uint32_t kl, uint32_t lw) {
    const uint32_t qk = nope + rope, nf = rope / 2;
    return KpQ{q, "Hy4fQSplitK"}.parallel_for<class Hy4fQSplitK>(sycl::range<2>(size_t(T) * NH, qk), [=](sycl::id<2> i) {
        const size_t r = i[0]; const uint32_t j = uint32_t(i[1]); const size_t t = r / NH;
        const float* s = q32 + r * qk;
        if (j < nope) { qn[r * nope + j] = sycl::half(s[j]); return; }
        const uint32_t k = j - nope, f = k / 2;
        const float c = cs[t * rope + f], sn = cs[t * rope + nf + f];
        const float x0 = s[nope + 2 * f], x1 = s[nope + 2 * f + 1];
        a[r * lw + kl + k] = sycl::half((k & 1u) ? x1 * c + x0 * sn : x0 * c - x1 * sn);
    });
}
// a [R, lw] fp16 columns [0, kl) = half(x [R, kl])
sycl::event place_cols(sycl::queue& q, const float* x, sycl::half* a, size_t R, uint32_t kl, uint32_t lw) {
    return KpQ{q, "Hy4fPlaceK"}.parallel_for<class Hy4fPlaceK>(sycl::range<2>(R, kl), [=](sycl::id<2> i) { a[i[0] * lw + i[1]] = sycl::half(x[i[0] * kl + i[1]]); });
}
// kv [T, kl + rope] fp32 -> cache rows [T, kl + rope] fp16: the latent RMS-normed with w [kl], the rope slice rotated.
sycl::event kv_append(sycl::queue& q, const float* kv, const float* w, const float* cs, sycl::half* dst, sycl::half* dst2, uint32_t T, uint32_t kl, uint32_t rope, float eps) {
    const uint32_t lw = kl + rope, nf = rope / 2;
    if (smallk()) {        // 16 lanes a row: each sums and writes every 16th element, the norm through the sub-group
        constexpr int SG = 16;
        return KpQ{q, "Hy4fKvAppend2K"}.parallel_for<class Hy4fKvAppend2K>(sycl::nd_range<2>({size_t(T), size_t(SG)}, {1, size_t(SG)}), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
            const size_t t = it.get_global_id(0); const uint32_t lane = uint32_t(it.get_local_id(1));
            const float* s = kv + t * lw; sycl::half* o = dst + t * lw;
            float a = 0.f;
            for (uint32_t d = lane; d < kl; d += SG) a += s[d] * s[d];
            a = sycl::reduce_over_group(it.get_sub_group(), a, sycl::plus<float>());
            const float r = sycl::rsqrt(a / float(kl) + eps);
            sycl::half* o2 = dst2 ? dst2 + t * lw : nullptr;          // (the windowed layers' copy for host RAM)
            for (uint32_t d = lane; d < kl; d += SG) { const sycl::half v = sycl::half(s[d] * r * w[d]); o[d] = v; if (o2) o2[d] = v; }
            for (uint32_t f = lane; f < nf; f += SG) {
                const float c = cs[t * rope + f], sn = cs[t * rope + nf + f];
                const float x0 = s[kl + 2 * f], x1 = s[kl + 2 * f + 1];
                const sycl::half v0 = sycl::half(x0 * c - x1 * sn), v1 = sycl::half(x1 * c + x0 * sn);
                o[kl + 2 * f] = v0; o[kl + 2 * f + 1] = v1;
                if (o2) { o2[kl + 2 * f] = v0; o2[kl + 2 * f + 1] = v1; }
            }
        });
    }
    return KpQ{q, "Hy4fKvAppendK"}.parallel_for<class Hy4fKvAppendK>(sycl::range<1>(T), [=](sycl::id<1> ti) {
        const size_t t = ti[0];
        const float* s = kv + t * lw; sycl::half* o = dst + t * lw;
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        for (uint32_t d = 0; d + 3 < kl; d += 4) { a0 += s[d] * s[d]; a1 += s[d + 1] * s[d + 1]; a2 += s[d + 2] * s[d + 2]; a3 += s[d + 3] * s[d + 3]; }
        const float r = sycl::rsqrt(((a0 + a1) + (a2 + a3)) / float(kl) + eps);
        for (uint32_t d = 0; d < kl; ++d) o[d] = sycl::half(s[d] * r * w[d]);
        for (uint32_t f = 0; f < nf; ++f) {
            const float c = cs[t * rope + f], sn = cs[t * rope + nf + f];
            const float x0 = s[kl + 2 * f], x1 = s[kl + 2 * f + 1];
            o[kl + 2 * f] = sycl::half(x0 * c - x1 * sn); o[kl + 2 * f + 1] = sycl::half(x1 * c + x0 * sn);
        }
        if (dst2) for (uint32_t d = 0; d < lw; ++d) dst2[t * lw + d] = o[d];
    });
}
// p [rows * NH, n_keys] fp16 = softmax over the allowed keys of (score * scaling) with the head's sink logit in the
// denominator. Row r allows keys lo[r] .. hi[r] (and, with `allow`, only those whose byte is set).
// dst [S, LW] = kv[sel[s]] (zeros where sel is -1): the keys one row of an indexed layer attends, side by side
sycl::event gather_keys(sycl::queue& q, const sycl::half* kv, const int32_t* sel, sycl::half* dst, uint32_t S, uint32_t LW) {
    return KpQ{q, "Hy4fGatherK"}.parallel_for<class Hy4fGatherK>(sycl::range<1>(S), [=](sycl::id<1> i) {
        const int32_t k = sel[i[0]]; sycl::half* o = dst + i[0] * LW;
        if (k < 0) { for (uint32_t d = 0; d < LW; ++d) o[d] = sycl::half(0.f); return; }
        const sycl::half* s = kv + size_t(k) * LW;
        for (uint32_t d = 0; d < LW; ++d) o[d] = s[d];
    });
}

sycl::event sink_softmax(sycl::queue& q, const float* scores, sycl::half* p, const int32_t* lo, const int32_t* hi, const uint8_t* allow,
                         const float* sinks, uint32_t rows, uint32_t NH, uint32_t n_keys, float scaling, bool use_sink) {
    if (smallk()) {        // 16 lanes a (row, head): the maximum and the sum through the sub-group, each lane writing its keys
        constexpr int SG = 16;
        return KpQ{q, "Hy4fSinkSoftmax2K"}.parallel_for<class Hy4fSinkSoftmax2K>(sycl::nd_range<3>({size_t(rows), size_t(NH), size_t(SG)}, {1, 1, size_t(SG)}), [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(SG)]] {
            const size_t r = it.get_global_id(0); const uint32_t h = uint32_t(it.get_global_id(1)); const int32_t lane = int32_t(it.get_local_id(2));
            const float* s = scores + (r * NH + h) * n_keys; sycl::half* o = p + (r * NH + h) * n_keys;
            const uint8_t* al = allow ? allow + r * n_keys : nullptr;
            const int32_t l = lo[r], u = hi[r];
            const float sink = sinks[h];
            float m = use_sink ? sink : -3.0e38f;
            for (int32_t k = lane; k < int32_t(n_keys); k += SG) if (k >= l && k <= u && (!al || al[k])) m = sycl::fmax(m, s[k] * scaling);
            m = sycl::reduce_over_group(it.get_sub_group(), m, sycl::maximum<float>());
            float sum = 0.f;
            for (int32_t k = lane; k < int32_t(n_keys); k += SG) if (k >= l && k <= u && (!al || al[k])) sum += sycl::exp(s[k] * scaling - m);
            sum = sycl::reduce_over_group(it.get_sub_group(), sum, sycl::plus<float>()) + (use_sink ? sycl::exp(sink - m) : 0.f);
            const float inv = 1.0f / sum;
            for (int32_t k = lane; k < int32_t(n_keys); k += SG)
                o[k] = (k >= l && k <= u && (!al || al[k])) ? sycl::half(sycl::exp(s[k] * scaling - m) * inv) : sycl::half(0.f);
        });
    }
    return KpQ{q, "Hy4fSinkSoftmaxK"}.parallel_for<class Hy4fSinkSoftmaxK>(sycl::range<2>(rows, NH), [=](sycl::id<2> i) {
        const size_t r = i[0]; const uint32_t h = uint32_t(i[1]);
        const float* s = scores + (r * NH + h) * n_keys; sycl::half* o = p + (r * NH + h) * n_keys;
        const uint8_t* al = allow ? allow + r * n_keys : nullptr;
        const int32_t l = lo[r], u = hi[r];
        const float sink = sinks[h];
        float m = use_sink ? sink : -3.0e38f;
        for (int32_t k = l; k <= u; ++k) if (!al || al[k]) m = sycl::fmax(m, s[k] * scaling);
        float sum = use_sink ? sycl::exp(sink - m) : 0.f;
        for (int32_t k = l; k <= u; ++k) if (!al || al[k]) sum += sycl::exp(s[k] * scaling - m);
        const float inv = 1.0f / sum;
        for (int32_t k = 0; k < int32_t(n_keys); ++k)
            o[k] = (k >= l && k <= u && (!al || al[k])) ? sycl::half(sycl::exp(s[k] * scaling - m) * inv) : sycl::half(0.f);
    });
}
// y[i] = half(v[i] * sigmoid(g[i]))
sycl::event gate_mul(sycl::queue& q, const float* v, const float* g, sycl::half* y, size_t n) {
    return KpQ{q, "Hy4fGateMulK"}.parallel_for<class Hy4fGateMulK>(sycl::range<1>(n), [=](sycl::id<1> i) { y[i] = sycl::half(v[i] * dsig(g[i])); });
}
// BF16 bits in place -> fp16 (the device rounds; BF16's 8 significant bits fit fp16's 11 wherever fp16 has the range).
sycl::event bf16_to_f16_inplace(sycl::queue& q, sycl::half* p, size_t n) {
    uint16_t* b = reinterpret_cast<uint16_t*>(p);
    return KpQ{q, "Hy4fBf16ToF16K"}.parallel_for<class Hy4fBf16ToF16K>(sycl::range<1>(n), [=](sycl::id<1> i) {
        const float f = sycl::bit_cast<float>(uint32_t(b[i]) << 16);
        p[i] = sycl::half(f);
    });
}

// (Hy4fOptions::emulate) w [n_blocks * B] fp16 rounded in place to a weight format, block by block. fmt: 1 mxfp4, 2 mxfp6,
// 3 fp8, 4 int8, 5 q6s, 6 q5a, 7 q4a, 8 q3a, 9 q2a, 10 t158, 11 b1 (see the option's comment). B = 32 or 16.
sycl::event emulate_format(sycl::queue& q, sycl::half* w, size_t n_blocks, uint32_t B, int fmt) {
    return KpQ{q, "Hy4fEmulateFormatK"}.parallel_for<class Hy4fEmulateFormatK>(sycl::range<1>(n_blocks), [=](sycl::id<1> bi) {
        sycl::half* p = w + bi[0] * B;
        float x[32]; float amax = 0.f, lo = 3.0e38f, hi = -3.0e38f;
        for (uint32_t i = 0; i < B; ++i) { x[i] = float(p[i]); amax = sycl::fmax(amax, sycl::fabs(x[i])); lo = sycl::fmin(lo, x[i]); hi = sycl::fmax(hi, x[i]); }
        if (!(amax > 0.f)) return;
        if (fmt == 1 || fmt == 2) {
            const float gmax = fmt == 1 ? 6.0f : 7.5f;
            auto grid = [=](float a) -> float {            // a in [0, gmax] -> the nearest value of the format
                if (fmt == 1) {
                    const float mids[7] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f}; const float vals[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
                    int k = 0; for (int m = 0; m < 7; ++m) k += mids[m] < a ? 1 : 0;
                    return vals[k];
                }
                const float step = a < 2.0f ? 0.125f : (a < 4.0f ? 0.25f : 0.5f);
                return sycl::fmin(sycl::rint(a / step) * step, 7.5f);
            };
            const float e0 = sycl::ceil(sycl::log2(amax / gmax));
            float best = 3.0e38f, bs = 1.f;
            for (int de = -1; de <= 1; ++de) {
                const float sc = sycl::exp2(e0 + float(de));
                float err = 0.f;
                for (uint32_t i = 0; i < B; ++i) { const float g = grid(sycl::fmin(sycl::fabs(x[i]) / sc, gmax)) * sc; const float d = (x[i] < 0.f ? -g : g) - x[i]; err += d * d; }
                if (err < best) { best = err; bs = sc; }
            }
            for (uint32_t i = 0; i < B; ++i) { const float g = grid(sycl::fmin(sycl::fabs(x[i]) / bs, gmax)) * bs; p[i] = sycl::half(x[i] < 0.f ? -g : g); }
        } else if (fmt == 3) {
            const float sc = sycl::exp2(sycl::ceil(sycl::log2(amax / 448.0f)));
            for (uint32_t i = 0; i < B; ++i) {
                const float a = sycl::fmin(sycl::fabs(x[i]) / sc, 448.0f);
                float r;
                if (a < 0.015625f) r = sycl::rint(a * 512.0f) / 512.0f;
                else { const float qn = sycl::exp2(sycl::floor(sycl::log2(a)) - 3.0f); r = sycl::fmin(sycl::rint(a / qn) * qn, 448.0f); }
                p[i] = sycl::half((x[i] < 0.f ? -r : r) * sc);
            }
        } else if (fmt == 4 || fmt == 5) {
            const float qmax = fmt == 4 ? 127.0f : 31.0f, qmin = fmt == 4 ? -127.0f : -32.0f;
            const float d = amax / qmax;
            for (uint32_t i = 0; i < B; ++i) p[i] = sycl::half(sycl::fmin(sycl::fmax(sycl::rint(x[i] / d), qmin), qmax) * d);
        } else if (fmt == 10 || fmt == 11) {
            // ternary {-s, 0, +s} (1.58 bits) and sign {-s, +s} (1 bit), s fitted per block: the least-squares scale for the
            // chosen codes (ternary: the threshold 0.7 * mean|x|, then s = the mean magnitude of the kept elements)
            float mean_abs = 0.f;
            for (uint32_t i = 0; i < B; ++i) mean_abs += sycl::fabs(x[i]);
            mean_abs /= float(B);
            if (fmt == 11) { for (uint32_t i = 0; i < B; ++i) p[i] = sycl::half(x[i] < 0.f ? -mean_abs : mean_abs); return; }
            const float th = 0.7f * mean_abs; float sum = 0.f, cnt = 0.f;
            for (uint32_t i = 0; i < B; ++i) if (sycl::fabs(x[i]) > th) { sum += sycl::fabs(x[i]); cnt += 1.f; }
            const float sc = cnt > 0.f ? sum / cnt : 0.f;
            for (uint32_t i = 0; i < B; ++i) p[i] = sycl::half(sycl::fabs(x[i]) > th ? (x[i] < 0.f ? -sc : sc) : 0.f);
        } else {
            const float levels = fmt == 6 ? 31.0f : (fmt == 7 ? 15.0f : (fmt == 8 ? 7.0f : 3.0f));
            const float step = (hi - lo) / levels;
            if (!(step > 0.f)) return;
            for (uint32_t i = 0; i < B; ++i) p[i] = sycl::half(lo + sycl::rint((x[i] - lo) / step) * step);
        }
    });
}
int emulate_code(const std::string& f, uint32_t& block) {
    block = 32;
    if (f == "mxfp4") return 1;
    if (f == "mxfp6") return 2;
    if (f == "fp8") return 3;
    if (f == "int8") return 4;
    if (f == "q6s") { block = 16; return 5; }
    if (f == "q5a") return 6;
    if (f == "q4a") { block = 16; return 7; }
    if (f == "q3a") { block = 16; return 8; }      // 3-bit integer, scale and offset per 16
    if (f == "q2a") { block = 16; return 9; }      // 2-bit integer, scale and offset per 16
    if (f == "t158") return 10;                    // ternary, one scale per 32
    if (f == "b1") return 11;                      // sign, one scale per 32
    return 0;
}

// IE_HY4F_RANGE=1 (diagnostic): the magnitude of a device buffer -- rms, largest |x|, smallest non-zero |x|, and how many
// values fp16 cannot hold (over 65504) or holds badly (non-zero under 6.1e-5, the subnormal range) -- printed per layer for
// every fp32 buffer that is about to be cast to fp16 or handed to a GEMM as fp16.
// A pass of up to this many rows is a decode pass (a drafted block being verified): direct kernels that read every matrix
// and every expert once for all its rows, no residency swaps. Longer passes are prompt chunks (oneDNN, the chunk kernels).
constexpr uint32_t kSmallRows = 8, kSmallJobs = 128;
// IE_HY4F_FUSE_SMALL=1 (OPT-IN): a decode pass runs the stream gates + collapse + norm, and each sublayer's end (the other
// card's half + norm + post-add), as one launch each with the same arithmetic (src/ops/hy4flash_fuse.cpp; the tokens are
// byte-identical). Measured 2026-11-19, 300 tokens, interleaved: 26.3 / 27.7 tok/s fused, 27.2 / 27.1 not -- 323 fewer
// launches a token, but a fused kernel holds a row on one of the card's 32 cores (16.9 + 14.1 us against 11.4 + 7.5 for
// the spread-out parts), and the launches it saves were not on the critical path.
// IE_HY4F_PP_GROUP=0: a prompt chunk's resident experts as one matrix-vector product per (row, expert) pick (the first
// form). On (default): expert-major XMX tiles, each expert's weights read once per 32 of its picks.
// IE_HY4F_PP_STREAM=0: a prompt chunk's missed experts on the CPU from RAM (the first form). On (default, with
// IE_HY4F_PP_GROUP): they are copied to the card through pinned buffers, kStageSlots at a time, and run there expert-major.
bool pp_stream() { static const bool on = [] { const char* v = std::getenv("IE_HY4F_PP_STREAM"); return !(v && *v == '0'); }(); return on; }
constexpr uint32_t kStageSlots = 16;
// A few persistent threads for the host side of the stream (page cache -> pinned buffer copies)
struct CopyPool {
    std::vector<std::thread> th; std::mutex m; std::condition_variable cv, done; std::function<void(uint32_t)> job;
    uint32_t n = 0, gen = 0, left = 0; bool stop = false;
    explicit CopyPool(uint32_t k) : n(k) {
        for (uint32_t i = 0; i < k; ++i) th.emplace_back([this, i] {
            uint32_t seen = 0;
            for (;;) {
                std::function<void(uint32_t)> f;
                { std::unique_lock<std::mutex> l(m); cv.wait(l, [&] { return stop || gen != seen; }); if (stop) return; seen = gen; f = job; }
                f(i);
                { std::lock_guard<std::mutex> l(m); if (--left == 0) done.notify_one(); }
            }
        });
    }
    void run(std::function<void(uint32_t)> f) { std::unique_lock<std::mutex> l(m); job = std::move(f); left = n; ++gen; cv.notify_all(); done.wait(l, [&] { return left == 0; }); }
    ~CopyPool() { { std::lock_guard<std::mutex> l(m); stop = true; } cv.notify_all(); for (auto& t : th) t.join(); }
};
CopyPool& copy_pool() { static const uint32_t k = [] { const char* v = std::getenv("IE_HY4F_STREAM_THREADS"); const int x = v ? std::atoi(v) : 8; return uint32_t(x > 0 ? x : 8); }(); static CopyPool p(k); return p; }
// IE_HY4F_PP_ATTN_XMX=0: a prompt chunk's indexed attention as the vector kernels (scores, softmax, sum). On: the XMX rows kernel.
bool pp_attn_xmx() { static const bool on = [] { const char* v = std::getenv("IE_HY4F_PP_ATTN_XMX"); return !(v && *v == '0'); }(); return on; }
bool pp_group() { static const bool on = [] { const char* v = std::getenv("IE_HY4F_PP_GROUP"); return !(v && *v == '0'); }(); return on; }
bool fuse_small_steps() { static const bool on = [] { const char* v = std::getenv("IE_HY4F_FUSE_SMALL"); return v && *v == '1'; }(); return on; }
bool range_on() { static const bool on = [] { const char* v = std::getenv("IE_HY4F_RANGE"); return v && *v == '1'; }(); return on; }
void range32(sycl::queue& q, const char* what, uint32_t L, const float* p, size_t n) {
    if (!range_on() || !n) return;
    std::vector<float> h(n);
    KpQ{q, "memcpy"}.memcpy(h.data(), p, n * 4).wait();
    double s2 = 0, mx = 0, mn = 1e300; size_t over = 0, tiny = 0, bad = 0;
    for (float v : h) {
        const double a = std::fabs(double(v));
        if (!(a < 1e30)) { ++bad; continue; }
        s2 += a * a; mx = std::max(mx, a); if (a > 0) mn = std::min(mn, a);
        over += a > 65504.0; tiny += a > 0 && a < 6.1e-5;
    }
    std::fprintf(stderr, "range L%02u %-10s rms %.3e max %.3e min %.3e  over-fp16 %zu  subnormal-in-fp16 %zu (%.1f %%)  non-finite %zu\n", L, what,
                 std::sqrt(s2 / double(n)), mx, mn > 1e299 ? 0.0 : mn, over, tiny, 100.0 * double(tiny) / double(n), bad);
}

// S[r, :] -= strength * sum_j (S[r, :] . B[j]) B[j] for every stream row r (B: k orthonormal rows of dim H)
sycl::event ablate_streams(sycl::queue& q, float* S, const float* B, size_t rows, uint32_t H, uint32_t k, float strength, float* coef) {
    KpQ{q, "Hy4fAblateDotK"}.parallel_for<class Hy4fAblateDotK>(sycl::range<2>(rows, k), [=](sycl::id<2> i) {
        const float* s = S + i[0] * H; const float* b = B + i[1] * H;
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        for (uint32_t d = 0; d + 3 < H; d += 4) { a0 += s[d] * b[d]; a1 += s[d + 1] * b[d + 1]; a2 += s[d + 2] * b[d + 2]; a3 += s[d + 3] * b[d + 3]; }
        coef[i[0] * k + i[1]] = ((a0 + a1) + (a2 + a3)) * strength;
    });
    return KpQ{q, "Hy4fAblateSubK"}.parallel_for<class Hy4fAblateSubK>(sycl::range<2>(rows, H), [=](sycl::id<2> i) {
        float a = 0.f;
        for (uint32_t j = 0; j < k; ++j) a += coef[i[0] * k + j] * B[size_t(j) * H + i[1]];
        S[i[0] * H + i[1]] -= a;
    });
}

}  // namespace

// IE_HY4F_RAW_SMALL=0: every fp16 product through oneDNN again (the bring-up behaviour, for comparison)
static bool raw_small() { static const bool on = [] { const char* v = std::getenv("IE_HY4F_RAW_SMALL"); return !(v && *v == '0'); }(); return on; }
// y32 [T, N] = x16 [T, K] . W^T, W fp16: a direct kernel for a few rows, oneDNN's GEMM for a chunk
static void mm16(sycl::queue& q, const sycl::half* x16, const sycl::half* w16, float* y32, uint32_t T, uint32_t N, uint32_t K) {
    if (T <= kSmallRows && K % 32 == 0 && raw_small()) hy4f_f16_gemv_f32(q, x16, w16, y32, K, N, T);
    else gemm_nt_f16_onednn(q, x16, w16, y32, T, N, K);
}
static void bmm16(sycl::queue& q, const sycl::half* a16, const sycl::half* w16, float* y32, uint32_t M, uint32_t G, uint32_t IPG, uint32_t OPG) {
    if (M <= kSmallRows && IPG % 8 == 0 && raw_small()) hy4f_f16_bmm_f32(q, a16, w16, y32, M, G, IPG, OPG);
    else gemm_bmm_nt_f16_onednn(q, a16, w16, y32, M, G, IPG, OPG);
}

// IE_HY4F_PREDICT_DIAG=1 (a measurement, decode only): how well a layer's expert picks can be told BEFORE its attention
// runs -- the layer's own router on the FFN's mix of the streams as they ENTER the layer. Printed at exit.
struct PredictDiag {
    static constexpr uint32_t Ks[4] = {16, 24, 32, 48};
    uint64_t rows = 0, picks = 0, misses = 0, hit_all[4] = {}, hit_miss[4] = {}, nonres[4] = {}, prev_miss = 0, prev_nonres = 0;
    std::vector<float> early; std::vector<std::vector<int32_t>> prev;
    ~PredictDiag() {
        if (!rows) return;
        std::fprintf(stderr, "[predict] %llu (layer, token) rows, %.2f misses a row\n", (unsigned long long)rows, double(misses) / rows);
        for (int i = 0; i < 4; ++i)
            std::fprintf(stderr, "[predict] early router, top %u: %.1f %% of the real picks, %.1f %% of the real MISSES; %.2f experts a row to fetch\n", Ks[i],
                         100.0 * hit_all[i] / std::max<uint64_t>(picks, 1), 100.0 * hit_miss[i] / std::max<uint64_t>(misses, 1), double(nonres[i]) / rows);
        std::fprintf(stderr, "[predict] the previous token's picks: %.1f %% of the real misses; %.2f experts a row to fetch\n", 100.0 * prev_miss / std::max<uint64_t>(misses, 1), double(prev_nonres) / rows);
    }
};
static PredictDiag g_pdiag;
// IE_HY4F_HOSTPROF=1 (a measurement): a decode token's host time per layer segment, printed at exit
struct HostProf {
    bool on = [] { const char* v = std::getenv("IE_HY4F_HOSTPROF"); return v && *v == '1'; }();
    bool prompt = [] { const char* v = std::getenv("IE_HY4F_HOSTPROF"); return v && *v == '1' && v[1] == 'p'; }();   // IE_HY4F_HOSTPROF=1p: the prompt chunks instead (ms a TOKEN of the prompt)
    double seg[8] = {}; uint64_t layers = 0, tokens = 0;
    std::chrono::steady_clock::time_point t;
    void mark(int i) { if (!on) return; const auto n = std::chrono::steady_clock::now(); seg[i] += std::chrono::duration<double, std::milli>(n - t).count(); t = n; }
    ~HostProf() {
        if (!on || !tokens) return;
        static const char* nm[8] = {"attn submit", "attn TP+wait", "ffn submit to router", "router wait", "topk+shared+moe (leg inside)", "ffn tail submit", "layer wait", "between cards+head"};
        std::fprintf(stderr, "[hostprof] %llu decode tokens, ms a token:", (unsigned long long)tokens);
        for (int i = 0; i < 8; ++i) std::fprintf(stderr, " %s %.2f;", nm[i], seg[i] / double(tokens));
        std::fprintf(stderr, "\n");
    }
};
static HostProf g_hp;
static bool predict_diag() { static const bool on = [] { const char* v = std::getenv("IE_HY4F_PREDICT_DIAG"); return v && *v == '1'; }(); return on; }

// Words between a card buffer and the card's PINNED block, as a kernel on the compute engine. A copy command of a few KiB
// to or from pageable host memory is read across the BAR on the CPU (~19 us a KiB) or handed to the copy engine
// (src/core/allocator.cpp, measured 2026-10-20); this forward issued 139 of them a decode token: 6.9 ms of a card's
// 39 ms (kernel profile, 2026-11-19). n_bytes a multiple of 4.
static sycl::event copy_words(sycl::queue& q, void* dst, const void* src, size_t n_bytes, const char* tag) {
    auto* d = static_cast<uint32_t*>(dst); const auto* sp = static_cast<const uint32_t*>(src);
    return KpQ{q, tag}.parallel_for<class Hy4fCopyWordsK>(sycl::range<1>(n_bytes / 4), [=](sycl::id<1> i) { d[i] = sp[i]; });
}

constexpr uint32_t kLogitRows = 64;                        // the head's logits buffer, rows (the head runs block by block)
constexpr size_t kIxScoreElems = size_t(8) << 20;        // the card's group-score workspace of one row block (32 MiB)
// IE_HY4F_RECALL_GPU=0: the indexer's group scores on the host (the first form; a row costs context / 4 * 2,048 products)
bool recall_gpu() { static const bool on = [] { const char* v = std::getenv("IE_HY4F_RECALL_GPU"); return !(v && *v == '0'); }(); return on; }
// IE_HY4F_SEL_KERNELS=0: an indexed layer's chosen keys gathered row by row and sent through oneDNN (the first form)
bool sel_kernels() { static const bool on = [] { const char* v = std::getenv("IE_HY4F_SEL_KERNELS"); return !(v && *v == '0'); }(); return on; }
// IE_HY4F_KV_WINDOW=0: every layer's keys on the cards for the whole context (the first form: 48 KiB a token of context,
// 12 GiB at 250,000 -- 39 expert slots a layer -- although 30 of the 42 layers read their last 512 keys only)
bool kv_window() { static const bool on = [] { const char* v = std::getenv("IE_HY4F_KV_WINDOW"); return !(v && *v == '0'); }(); return on; }
// IE_HY4F_TP=0: every layer's attention on its own card (the first form). Default with two cards: a sliding layer's heads
// are split over both -- the cards took turns before (each executing 43-50 % of the time), and a launch with its wait
// costs the host 5.5 us, so the hand-offs are cheap.
bool tp_on() { static const bool on = [] { const char* v = std::getenv("IE_HY4F_TP"); return !(v && *v == '0'); }(); return on; }
constexpr size_t kPinAllow = size_t(1) << 20;            // a key mask up to this size goes through the pinned stage (decode rows)

struct Hy4fForward::Card {
    sycl::queue* q = nullptr;
    uint32_t L0 = 0, L1 = 0;
    bool tier_on = false; uint32_t tier_L0 = 0;
    Ds41ExpertTier tier;
    Hy4fQ8Store q8; bool q8_on = false;
    sycl::half *q8_g16 = nullptr, *q8_u16 = nullptr, *q8_y16 = nullptr, *q8_hj = nullptr;
    // a pass of 2 .. 8 rows: its resident experts as jobs (slot, a bit per row), each row's picks as (job, weight)
    int32_t *mr_slot = nullptr, *mr_rowjobs = nullptr, *pin_mr = nullptr; uint32_t* mr_mask = nullptr; float *mr_roww = nullptr, *mr_yj = nullptr; sycl::half* mr_hj = nullptr;
    // pinned (USM host) staging in this card's context: the host reads and writes these, kernels copy to / from the card
    float *pin_S = nullptr, *pin_cos = nullptr, *pin_rlog = nullptr, *pin_lat = nullptr, *pin_ycpu = nullptr, *pin_emb = nullptr, *pin_logits = nullptr;
    int32_t* pin_rng = nullptr; std::vector<void*> pinned;
    std::vector<Dense> helper; uint32_t hL0 = 0; bool any_tp = false;   // the OTHER card's split layers: this card's half of their heads
    sycl::half* pin_tpx = nullptr; float *pin_tpo = nullptr, *tp_o32 = nullptr;   // a split layer's input and partial output on their way between the cards
    sycl::half* pin_kv = nullptr; uint32_t n_kv_stage = 0;     // the windowed layers' new key rows of a call [n_kv_stage, max_tokens, LW], for host RAM
    float *eg_mask = nullptr, *pin_egmask = nullptr; uint32_t eg_mask_dirty = 0;   // [T] 1 = the row owns an n-gram; rows that may not hold 1
    float *pin_gcs = nullptr, *ix_gcs = nullptr;   // a group's rope (cos | sin) on its way to the card's pooling
    float *pin_ixq = nullptr, *pin_ixw = nullptr; uint8_t* pin_allow = nullptr;   // the indexer's query read-back; a decode row's key mask
    float* q8_ycpu = nullptr; uint64_t q8_promo_tok = ~0ull; uint32_t q8_promo_left = 0;
    // a chunk's jobs (row, expert): the residents' tables on the card, the misses' on the host
    int32_t *q8_jslot = nullptr, *q8_jrow = nullptr, *q8_rs = nullptr; float *q8_jw = nullptr, *q8_yrows = nullptr; sycl::half* q8_hrows = nullptr;
    std::vector<int32_t> h_jslot, h_jrow, h_rs, h_crow; std::vector<float> h_jw, h_cw; std::vector<Hy4fQ8Store::Planes> h_cpl;
    // a chunk's picks expert-major (src/ops/hy4flash_moe_grp.cpp): jobs of up to 32 picks of one expert
    uint64_t* grp_jbase = nullptr; int32_t *grp_j0 = nullptr, *grp_jn = nullptr; size_t grp_jcap = 0;
    // a chunk's missed experts streamed to the card: two staging areas of kStageSlots experts, their pinned host buffers,
    // and a copy queue (the DMA beside the compute)
    uint8_t* stg = nullptr; uint8_t* pin_stg[2] = {nullptr, nullptr}; std::unique_ptr<sycl::queue> qc;
    std::vector<uint64_t> h_gbase; std::vector<int32_t> h_gj0, h_gjn, h_gpos, h_gcnt;
    sycl::half* dq16 = nullptr; size_t dq_need = 0;      // q8_dense: the fp16 scratch a many-row GEMM expands a matrix into
    std::vector<Dense> dense;
    std::vector<void*> owned; uint64_t bytes = 0, kv_bytes = 0;   // kv_bytes: the part of `bytes` that is context cache (grows with ctx)
    // Elastic context (the indexed layers' latent caches): each cache is a reserved range of ctx rows, backed for el_caps[el_tier]
    // tokens; the VRAM of the rest holds expert slots (Hy4fQ8Store context pieces) until a conversation grows into it.
    int32_t extra_piece = -1;   // the store piece held for the caller (the image tower), released by release_extra()
    bool el_on = false; uint32_t el_tier = 0; std::vector<uint32_t> el_caps; std::vector<uint64_t> el_back;   // [tier] tokens, bytes backed a cache
    std::vector<uintptr_t> el_va; uint64_t el_full = 0; std::vector<std::vector<sycl::ext::oneapi::experimental::physical_mem>> el_ph;   // [cache]; [tier][cache]
    // the per-row workspace lives in an arena (alloc_rows): its own VRAM for max_tokens rows, or -- a prompt chunk of more rows
    // (big_tokens) -- the expert slots' lent tails (Hy4fQ8Store::lend); `measuring` lays it out without memory to size it
    uint8_t *arena_cur = nullptr, *arena_end = nullptr, *small_arena = nullptr; uint64_t small_bytes = 0, big_need = 0;
    bool arena_on = false, measuring = false, big = false;
    bool any_indexed = false, any_dense_ffn = false, any_engram = false; Card* other_card = nullptr;
    uint32_t nh_max = 0, nh_min = 0; size_t allow_cap = 0;
    float* mix_part = nullptr;
    float *S = nullptr, *mix = nullptr, *pre = nullptr, *post = nullptr, *x = nullptr, *xn32 = nullptr, *qa32 = nullptr, *qr32 = nullptr;
    float *q32 = nullptr, *qabs32 = nullptr, *kv32 = nullptr, *scores = nullptr, *v32 = nullptr, *gate32 = nullptr, *o32 = nullptr, *y32 = nullptr;
    sycl::half *xn16 = nullptr, *qr16 = nullptr, *qn16 = nullptr, *a16 = nullptr, *p16 = nullptr, *ctx16 = nullptr, *attn16 = nullptr;
    float *ix_tk32 = nullptr, *ix_q32 = nullptr, *ix_hw32 = nullptr;
    int32_t* sel = nullptr; sycl::half* kg16 = nullptr;
    float *ix_qrot = nullptr, *ix_cs = nullptr, *ix_sc = nullptr, *pin_ixcs = nullptr; int32_t *ix_own = nullptr, *pin_ixown = nullptr;   // the indexer's scoring on the card      // an indexed layer's chosen keys [T, S]; one row's keys gathered [S, LW]
    float *ff_g32 = nullptr, *ff_u32 = nullptr; sycl::half* ff_h16 = nullptr;
    float *rlogits = nullptr, *lat32 = nullptr, *moe32 = nullptr, *rout32 = nullptr, *sd32 = nullptr; sycl::half* lat16 = nullptr;
    float *cs_s = nullptr, *cs_i = nullptr;
    int32_t *lo = nullptr, *hi = nullptr; uint8_t* allow = nullptr;
    int32_t *lo_s = nullptr, *hi_s = nullptr, *lo_i = nullptr, *hi_i = nullptr;   // a call's key ranges: sliding layers / indexed layers
    float *eg_emb32 = nullptr, *eg_v32 = nullptr, *eg_k32 = nullptr, *eg_stat = nullptr; sycl::half* eg_emb16 = nullptr;
    sycl::half *ex_gu = nullptr, *ex_dn = nullptr, *ex_x16 = nullptr, *ex_h16 = nullptr; float *ex_gu32 = nullptr, *ex_d32 = nullptr, *ex_w = nullptr; int32_t* ex_rows = nullptr;
    float *abl_B = nullptr, *abl_coef = nullptr; uint32_t abl_cap = 0; uint64_t abl_version = 0;
    float *wm_A = nullptr, *wm_B = nullptr; uint64_t wm_version = 0; uint32_t wm_cap = 0;   // this card's layers of the write maps
    float* steer = nullptr; uint64_t steer_version = 0;      // this card's layers of the steering vectors [L1 - L0, hc, dim]
    sycl::half* head = nullptr; float *final_norm = nullptr, *head_fn = nullptr, *head_base = nullptr, *head_scale = nullptr, *logits = nullptr;
};

// One persistent host thread: a chunk's matrix steps block the thread that issues them, so the other card's half of a
// split layer is driven from here while the calling thread drives the home card's (persistent: oneDNN's primitive cache
// is per thread). A decode row's kernels do not block and are issued inline.
struct Hy4fForward::TpWorker {
    std::thread th; std::mutex m; std::condition_variable cv;
    std::function<std::string()> job; std::string result; bool has_job = false, done = true, stop = false;
    TpWorker() {
        th = std::thread([this] {
            std::unique_lock<std::mutex> lk(m);
            for (;;) {
                cv.wait(lk, [this] { return has_job || stop; });
                if (stop) return;
                auto j = std::move(job); has_job = false;
                lk.unlock();
                std::string r;
                try { r = j(); } catch (const sycl::exception& e) { r = std::string("sycl: ") + e.what(); } catch (const std::exception& e) { r = e.what(); }
                lk.lock();
                result = std::move(r); done = true; cv.notify_all();
            }
        });
    }
    ~TpWorker() { { std::lock_guard<std::mutex> lk(m); stop = true; } cv.notify_all(); if (th.joinable()) th.join(); }
    void post(std::function<std::string()> j) { { std::lock_guard<std::mutex> lk(m); job = std::move(j); has_job = true; done = false; } cv.notify_all(); }
    std::string wait() { std::unique_lock<std::mutex> lk(m); cv.wait(lk, [this] { return done; }); return std::move(result); }
};

// The drafter (Hy4fOptions::drafter asks for it; IE_HY4F_DFLY=0 leaves it out whatever the options say). Everything lives
// on the LAST card, where the target layers it reads are.
struct Hy4fForward::Dfly {
    bool on = false; uint32_t W = 0, FI = 0;                // W = block rows (1 + drafts), FI = its MLP width
    std::vector<Dense> layers; std::vector<float*> ln1, ln2;
    float *final_norm = nullptr, *enorm = nullptr, *hnorm = nullptr, *fgate = nullptr, *hidden_norm = nullptr, *embed_norm = nullptr;
    sycl::half *eh_proj = nullptr, *gate_proj = nullptr, *up_proj = nullptr, *down_proj = nullptr;
    float *th32 = nullptr, *x32 = nullptr, *e32 = nullptr, *h32 = nullptr, *f32 = nullptr, *pe32 = nullptr, *g32 = nullptr, *u32 = nullptr, *d32 = nullptr, *mg32 = nullptr, *mu32 = nullptr;
    sycl::half *z16 = nullptr, *hm16 = nullptr, *hc16 = nullptr, *mh16 = nullptr, *keys = nullptr;
    int32_t* sel_dev = nullptr; float* pin_e = nullptr; int32_t* pin_sel = nullptr;
    // the target's final hidden (normed: the head's input) of the last call's rows
    sycl::half* th16 = nullptr; uint32_t th_row0 = 0, th_rows = 0;
    int32_t* pin_rng = nullptr; sycl::half* pin_hc = nullptr;
    std::vector<float> conf_w; float conf_b = 0.f;          // the confidence head (a logit per draft: its acceptance rate as trained)
    // the target's last call: per target layer, the keys each row attended (when the recall ran) or "every key" (dense)
    // (row0 = the call's row the first stored choice belongs to: a prompt chunk keeps only its last row's)
    struct Snap { uint32_t T = 0, row0 = 0, pos0 = 0; bool masked = false; std::vector<int32_t> sel, cnt; };
    std::vector<Snap> snap;
};
bool dfly_on() { static const bool on = [] { const char* v = std::getenv("IE_HY4F_DFLY"); return !(v && *v == '0'); }(); return on; }

Hy4fForward::Hy4fForward() = default;
Hy4fForward::~Hy4fForward() { free_all(); }

template <class T> T* Hy4fForward::dev(Card& c, size_t n) {
    if (c.arena_on) {                                            // (alloc_rows: carved from the arena, 256-byte aligned)
        const uintptr_t at = (reinterpret_cast<uintptr_t>(c.arena_cur) + 255) & ~uintptr_t(255), end = at + (n ? n : 1) * sizeof(T);
        if (!c.measuring && end > reinterpret_cast<uintptr_t>(c.arena_end)) return nullptr;
        c.arena_cur = reinterpret_cast<uint8_t*>(end);
        return reinterpret_cast<T*>(at);
    }
    T* p = sycl::malloc_device<T>(n ? n : 1, *c.q);
    if (p) { c.owned.push_back(p); c.bytes += n * sizeof(T); }
    return p;
}

uint64_t Hy4fForward::vram_bytes(size_t card) const {
    if (card >= cards_.size()) return 0;
    return cards_[card]->bytes + (cards_[card]->tier_on ? cards_[card]->tier.vram_bytes() : 0) + (cards_[card]->q8_on ? cards_[card]->q8.vram_bytes() : 0);
}

void Hy4fForward::free_all() {
    tpw_.reset(); dfly_.reset();
    for (auto& c : cards_) if (c->q) c->q->wait();
    for (auto& c : cards_) {
        if (!c->q) continue;
        if (c->tier_on) { c->tier.free_storage(*c->q); c->tier_on = false; }
        if (c->q8_on) { c->q8.free_all(*c->q); c->q8_on = false; }
        if (c->el_on) {
            namespace ex = sycl::ext::oneapi::experimental;
            try {
                c->q->wait();
                for (size_t b = 0; b < c->el_va.size(); ++b) {
                    ex::unmap(reinterpret_cast<void*>(c->el_va[b]), c->el_back[c->el_tier], c->q->get_context());
                    ex::free_virtual_mem(c->el_va[b], c->el_full, c->q->get_context());
                }
            } catch (...) {}
            c->el_ph.clear(); c->el_va.clear(); c->el_on = false;
        }
        for (void* p : c->owned) sycl::free(p, *c->q);
        std::sort(c->pinned.begin(), c->pinned.end()); c->pinned.erase(std::unique(c->pinned.begin(), c->pinned.end()), c->pinned.end());
        for (void* p : c->pinned) sycl::free(p, *c->q);
        c->owned.clear(); c->pinned.clear(); c->bytes = 0;
    }
    for (auto& [mp, mb] : kv_mirror_) if (mp) ::munmap(mp, mb);
    kv_mirror_.clear();
    cards_.clear(); indexers_.clear(); ix_dev_groups_.clear(); hist_.clear(); pos3_.clear(); n_pos_ = 0;
}

void Hy4fForward::rewind(uint32_t n) {
    if (n >= n_pos_) return;
    n_pos_ = n; hist_.resize(n); pos3_.resize(size_t(n) * 3);
    for (auto& ix : indexers_) {
        if (n <= ix.n_tokens()) { ix.pending = 0; ix.rewind(n); }
        else ix.pending = n - ix.n_tokens();              // (the first rows still waiting on the card stay)
    }
    for (size_t i = 0; i < indexers_.size(); ++i) ix_dev_groups_[i] = std::min(ix_dev_groups_[i], indexers_[i].n_groups());
    // a sliding layer continues at n with the keys n - (window - 1) .. n - 1: when the card's window starts later, they
    // come back from host RAM
    if (!m_) return;
    const auto& cfg = m_->config(); const size_t LW = size_t(cfg.kv_lora_rank) + cfg.rope_dim; const uint32_t Wn = cfg.window;
    const uint32_t w0 = n >= Wn - 1 ? n - (Wn - 1) : 0;
    for (auto& cp : cards_) {
        if (!cp->q) continue;
        bool any = false;
        for (auto* vec : {&cp->dense, &cp->helper})
            for (Dense& d : *vec) if (d.kv_win && w0 < d.kv_base) {
                if (n > w0) cp->q->memcpy(d.kv, kv_mirror_[size_t(d.kv_host)].first + size_t(w0) * LW, size_t(n - w0) * LW * 2);
                d.kv_base = w0; any = true;
            }
        if (any) cp->q->wait();
    }
}

std::string Hy4fForward::upload_card(Card& c) {
    const auto& m = *m_; const auto& cfg = m.config();
    sycl::queue& q = *c.q;
    const uint32_t H = cfg.dim, hc = cfg.hc_mult, KL = cfg.kv_lora_rank, RD = cfg.rope_dim, ND = cfg.nope_dim, VD = cfg.v_dim, LW = KL + RD;
    std::string err;
    std::vector<float> f;
    auto up32 = [&](const Ds41Tensor& t) -> float* {
        if (!err.empty()) return nullptr;
        if (auto e = hy4f_to_f32(t, f); !e.empty()) { err = e; return nullptr; }
        float* p = dev<float>(c, f.size());
        if (!p) { err = "device allocation failed (" + t.w->name + ")"; return nullptr; }
        KpQ{q, "memcpy"}.memcpy(p, f.data(), f.size() * 4).wait(); return p;
    };
    // a matrix as fp16 [N, K]: BF16 bytes go up as they are and are converted on the card
    auto up16 = [&](const Ds41Tensor& t) -> sycl::half* {
        if (!err.empty()) return nullptr;
        const size_t n = size_t(t.w->numel());
        sycl::half* p = dev<sycl::half>(c, n);
        if (!p) { err = "device allocation failed (" + t.w->name + ")"; return nullptr; }
        if (t.w->dtype_str == "BF16") { KpQ{q, "memcpy"}.memcpy(p, t.w->data, n * 2); bf16_to_f16_inplace(q, p, n).wait(); return p; }
        if (auto e = hy4f_to_f32(t, f); !e.empty()) { err = e; return nullptr; }
        float* tmp = sycl::malloc_device<float>(n, q);
        if (!tmp) { err = "device allocation failed (staging " + t.w->name + ")"; return nullptr; }
        KpQ{q, "memcpy"}.memcpy(tmp, f.data(), n * 4); cast_fp32_to_fp16(q, tmp, p, n).wait(); sycl::free(tmp, q);
        return p;
    };
    auto up16_bits = [&](const std::vector<uint16_t>& bits, const char* what) -> sycl::half* {
        if (!err.empty()) return nullptr;
        sycl::half* p = dev<sycl::half>(c, bits.size());
        if (!p) { err = std::string("device allocation failed (") + what + ")"; return nullptr; }
        KpQ{q, "memcpy"}.memcpy(p, bits.data(), bits.size() * 2); bf16_to_f16_inplace(q, p, bits.size()).wait(); return p;
    };
    // several [n] vectors stacked into one fp32 [k, n] block
    auto up32_stack = [&](const std::vector<Ds41Tensor>& ts) -> float* {
        if (!err.empty()) return nullptr;
        std::vector<float> all;
        for (const auto& t : ts) { if (auto e = hy4f_to_f32(t, f); !e.empty()) { err = e; return nullptr; } all.insert(all.end(), f.begin(), f.end()); }
        float* p = dev<float>(c, all.size());
        if (!p) { err = "device allocation failed (stack)"; return nullptr; }
        KpQ{q, "memcpy"}.memcpy(p, all.data(), all.size() * 4).wait(); return p;
    };

    // a big matrix [N, K]: 8-bit SoA under q8_dense (BF16 sources with K a multiple of 32), fp16 otherwise
    // IE_HY4F_Q8_DENSE_SET=qb,gate,o,mlp,lat,sh (diagnostic): which kinds q8_dense converts; unset = the default set
    static const std::string q8_set = [] { const char* v = std::getenv("IE_HY4F_Q8_DENSE_SET"); return std::string(v ? v : "gate,o,mlp,lat,sh,qb"); }();
    auto in_set = [&](const char* kind) { const std::string k = std::string(",") + q8_set + ","; return k.find(std::string(",") + kind + ",") != std::string::npos; };
    auto upm = [&](const Ds41Tensor& t, uint32_t N, uint32_t K, const char* kind) -> Mat {
        Mat mt;
        if (!err.empty()) return mt;
        if (!opt_.q8_dense || !in_set(kind) || t.w->dtype_str != "BF16" || K % 32 || size_t(t.w->numel()) != size_t(N) * K) { mt.w16 = up16(t); return mt; }
        std::vector<int8_t> qs(size_t(N) * K); std::vector<uint16_t> ds(size_t(N) * (K / 32));
        hy4f_q8_from_bf16_mt(t.w->data, N, K, qs.data(), ds.data());
        int8_t* pq = dev<int8_t>(c, qs.size()); uint16_t* pd = dev<uint16_t>(c, ds.size());
        if (!pq || !pd) { err = "device allocation failed (" + t.w->name + ", 8-bit)"; return mt; }
        KpQ{q, "memcpy"}.memcpy(pq, qs.data(), qs.size()); KpQ{q, "memcpy"}.memcpy(pd, ds.data(), ds.size() * 2).wait();
        mt.qs = pq; mt.d = pd; c.dq_need = std::max(c.dq_need, size_t(N) * K); ++n_q8_mats_;
        return mt;
    };
    // raw BF16 [N, K] as a big matrix (8-bit under q8_dense, else fp16): a slice of a checkpoint tensor
    auto upm_raw = [&](const uint8_t* bf16, uint32_t N, uint32_t K, const char* kind, const std::string& name) -> Mat {
        Mat mt;
        if (!err.empty()) return mt;
        if (!opt_.q8_dense || !in_set(kind) || K % 32) {
            sycl::half* p = dev<sycl::half>(c, size_t(N) * K);
            if (!p) { err = "device allocation failed (" + name + ")"; return mt; }
            KpQ{q, "memcpy"}.memcpy(p, bf16, size_t(N) * K * 2); bf16_to_f16_inplace(q, p, size_t(N) * K).wait(); mt.w16 = p; return mt;
        }
        std::vector<int8_t> qs(size_t(N) * K); std::vector<uint16_t> ds(size_t(N) * (K / 32));
        hy4f_q8_from_bf16_mt(bf16, N, K, qs.data(), ds.data());
        int8_t* pq = dev<int8_t>(c, qs.size()); uint16_t* pd = dev<uint16_t>(c, ds.size());
        if (!pq || !pd) { err = "device allocation failed (" + name + ", 8-bit)"; return mt; }
        KpQ{q, "memcpy"}.memcpy(pq, qs.data(), qs.size()); KpQ{q, "memcpy"}.memcpy(pd, ds.data(), ds.size() * 2).wait();
        mt.qs = pq; mt.d = pd; c.dq_need = std::max(c.dq_need, size_t(N) * K); ++n_q8_mats_;
        return mt;
    };
    // The attention weights of heads h0 .. h0 + nh - 1 of a layer and the layer's key cache on this card. A slice of q_b /
    // gate is a run of rows, a slice of o a run of columns (256 a head: whole 32-weight groups, so the 8-bit values are the
    // unsplit matrix's), kv_b is cut per head as before.
    auto attn_heads = [&](Dense& d, const Hy4fLayer& lw, uint32_t L, uint32_t h0, uint32_t nh, bool home, bool with_kv = true) {
        const uint32_t NHl = lw.n_heads, QKd = cfg.qk_dim(), QLr = cfg.q_lora_rank, Hd = cfg.dim;
        d.nh = nh; d.h0 = h0;
        d.q_a_norm = up32(lw.q_a_norm); d.kv_a_norm = up32(lw.kv_a_norm);
        d.q_a = up16(lw.q_a); d.kv_a = up16(lw.kv_a);
        if (!err.empty()) return;
        if (h0 == 0 && nh == NHl) {
            d.sinks = up32(lw.sinks);
            d.q_b = upm(lw.q_b, NHl * QKd, QLr, "qb"); d.gate = upm(lw.gate, NHl * VD, Hd, "gate"); d.o = upm(lw.o, Hd, NHl * VD, "o");
        } else {
            if (auto e = hy4f_to_f32(lw.sinks, f); !e.empty()) { err = e; return; }
            d.sinks = dev<float>(c, nh);
            if (!d.sinks) { err = "device allocation failed (sinks)"; return; }
            KpQ{q, "memcpy"}.memcpy(d.sinks, f.data() + h0, size_t(nh) * 4).wait();
            d.q_b = upm_raw(lw.q_b.w->data + size_t(h0) * QKd * QLr * 2, nh * QKd, QLr, "qb", lw.q_b.w->name);
            d.gate = upm_raw(lw.gate.w->data + size_t(h0) * VD * Hd * 2, nh * VD, Hd, "gate", lw.gate.w->name);
            std::vector<uint8_t> cols(size_t(Hd) * nh * VD * 2);
            for (uint32_t r = 0; r < Hd; ++r) std::memcpy(cols.data() + size_t(r) * nh * VD * 2, lw.o.w->data + (size_t(r) * NHl * VD + size_t(h0) * VD) * 2, size_t(nh) * VD * 2);
            d.o = upm_raw(cols.data(), Hd, nh * VD, "o", lw.o.w->name);
        }
        if (!err.empty()) return;
        // kv_b_proj [NH * (nope + v), kl] split per head: W_k transposed ([NH * kl, nope], the bmm's [G * OPG, IPG]) for the
        // absorbed query, W_v widened to the latent row ([NH * v, kl + rope], zero under the rope columns) for the values.
        if (lw.kv_b.w->dtype_str != "BF16") { err = "kv_b_proj is not BF16 (" + lw.kv_b.w->dtype_str + ")"; return; }
        {
            std::vector<uint16_t> wk(size_t(nh) * KL * ND), wv(size_t(nh) * VD * LW, 0);
            const uint8_t* sb = lw.kv_b.w->data;
            for (uint32_t hh = 0; hh < nh; ++hh) {
                const uint32_t h = h0 + hh;
                for (uint32_t i = 0; i < ND; ++i) {
                    const uint8_t* row = sb + (size_t(h) * (ND + VD) + i) * KL * 2;
                    for (uint32_t j = 0; j < KL; ++j) { uint16_t b; std::memcpy(&b, row + size_t(j) * 2, 2); wk[(size_t(hh) * KL + j) * ND + i] = b; }
                }
                for (uint32_t o = 0; o < VD; ++o)
                    std::memcpy(wv.data() + (size_t(hh) * VD + o) * LW, sb + (size_t(h) * (ND + VD) + ND + o) * KL * 2, size_t(KL) * 2);
            }
            d.wk_t = up16_bits(wk, "kv_b keys"); d.wv = up16_bits(wv, "kv_b values");
        }
        if (with_kv) {
            const uint32_t wcap = uint32_t((2ull * (uint64_t(cfg.window) + rows_max()) + 63u) / 64u * 64u);
            d.kv_win = !lw.indexed && kv_window() && wcap < ctx_;
            d.kv_cap = d.kv_win ? wcap : ctx_; d.kv_base = 0;
            if (c.el_on && !d.kv_win && d.kv_cap == ctx_) {
                namespace ex = sycl::ext::oneapi::experimental;
                try {
                    const uintptr_t va = ex::reserve_virtual_mem(c.el_full, q.get_context());
                    if (c.el_ph.empty()) c.el_ph.resize(c.el_caps.size());
                    c.el_ph[0].emplace_back(q.get_device(), q.get_context(), c.el_back[0]);
                    c.el_ph[0].back().map(va, c.el_back[0], ex::address_access_mode::read_write);
                    c.el_va.push_back(va); d.kv = reinterpret_cast<sycl::half*>(va);
                } catch (const sycl::exception& x) { err = std::string("device allocation failed (latent cache, elastic): ") + x.what(); return; }
                c.bytes += c.el_back[0]; c.kv_bytes += c.el_back[0];
                KpQ{q, "fill"}.fill(d.kv, sycl::half(0.f), size_t(c.el_back[0] / 2)).wait();
            } else {
            d.kv = dev<sycl::half>(c, size_t(d.kv_cap) * LW);
            if (!d.kv) { err = "device allocation failed (latent cache)"; return; }
            c.kv_bytes += size_t(d.kv_cap) * LW * 2;
            KpQ{q, "fill"}.fill(d.kv, sycl::half(0.f), size_t(d.kv_cap) * LW).wait();
            }
            if (d.kv_win) {
                if (kv_mirror_.size() < cfg.n_layers) kv_mirror_.resize(cfg.n_layers, {nullptr, 0});
                if (!kv_mirror_[L].first) {            // one copy of the layer's keys in host RAM, whichever card asks first
                    const size_t mb = size_t(ctx_) * LW * 2;
                    void* mp = ::mmap(nullptr, mb, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
                    if (mp == MAP_FAILED) { err = "cannot map host RAM for a sliding layer's keys"; return; }
                    kv_mirror_[L] = {static_cast<sycl::half*>(mp), mb};
                }
                d.kv_host = int32_t(L); d.kv_mirror = home;
                if (home) d.kv_stage = c.n_kv_stage++;
            }
        }
    };
    Card* other = nullptr;
    for (auto& cp : cards_) if (cp.get() != &c && cards_.size() == 2) other = cp.get();
    auto can_split = [&](const Hy4fLayer& lw) {
        return other && tp_on() && !lw.indexed && lw.n_heads % 2 == 0 && (size_t(lw.n_heads / 2) * cfg.v_dim) % 32 == 0 && !opt_.stop_after &&
               lw.q_b.w->dtype_str == "BF16" && lw.gate.w->dtype_str == "BF16" && lw.o.w->dtype_str == "BF16";
    };
    c.dense.assign(c.L1 - c.L0, Dense{});
    c.nh_max = 0;
    bool any_dense_ffn = false, any_engram = false, any_indexed = false;
    for (uint32_t L = c.L0; L < c.L1 && err.empty(); ++L) {
        const Hy4fLayer& lw = m.layers()[L]; Dense& d = c.dense[L - c.L0];
        const uint32_t NH = lw.n_heads;
        d.hc_a_fn = up32(lw.hc_attn_fn); d.hc_a_base = up32(lw.hc_attn_base); d.hc_a_scale = up32(lw.hc_attn_scale);
        d.hc_f_fn = up32(lw.hc_ffn_fn); d.hc_f_base = up32(lw.hc_ffn_base); d.hc_f_scale = up32(lw.hc_ffn_scale);
        d.attn_norm = up32(lw.attn_norm); d.ffn_norm = up32(lw.ffn_norm);
        d.post_norm = up32(lw.post_norm);
        // a sliding layer's heads over both cards (IE_HY4F_TP): this card keeps the first half, the other card loads the rest
        const bool split = can_split(lw);
        d.tp = split; if (split) c.any_tp = true;
        attn_heads(d, lw, L, 0, split ? NH / 2 : NH, true);
        if (!err.empty()) break;
        c.nh_max = std::max(c.nh_max, d.nh); c.nh_min = c.nh_min ? std::min(c.nh_min, d.nh) : d.nh;
        if (lw.indexed) {
            any_indexed = true;
            d.ix_q = up16(lw.ix_q); d.ix_head_w = up16(lw.ix_head_w); d.ix_tok_k = up16(lw.ix_tok_k);
            ix_pend_cap_ = rows_max() + 1024;
            d.ix_pend = dev<float>(c, size_t(ix_pend_cap_) * cfg.ix_head_dim);
            if (!d.ix_pend) { err = "device allocation failed (indexer keys)"; break; }
            d.ix_grp = dev<float>(c, (size_t(ctx_) / cfg.ix_ratio + 2) * cfg.ix_head_dim);
            if (!d.ix_grp) { err = "device allocation failed (indexer group keys)"; break; }
            c.kv_bytes += (size_t(ctx_) / cfg.ix_ratio + 2) * cfg.ix_head_dim * 4;
            d.ix = int32_t(indexers_.size()); ix_dev_groups_.push_back(0);
            Hy4fIndexer ix; ix.ratio = cfg.ix_ratio; ix.heads = cfg.ix_heads; ix.hd = cfg.ix_head_dim; ix.group_topk = cfg.ix_group_topk;
            ix.rope_dim = RD; ix.theta = cfg.theta_indexed; ix.eps = cfg.norm_eps; std::copy(cfg.mrope_indexed, cfg.mrope_indexed + 3, ix.mrope);
            if (ix.hd > 128 || ix.hd < RD || RD > 256) { err = "indexer head width out of the supported range"; break; }
            std::string e;
            if (!(e = hy4f_to_f32(lw.ix_tok_norm, ix.tok_norm)).empty() || !(e = hy4f_to_f32(lw.ix_kv, ix.kv)).empty() || !(e = hy4f_to_f32(lw.ix_kg, ix.kg)).empty() ||
                !(e = hy4f_to_f32(lw.ix_key_norm, ix.key_norm)).empty() || !(e = hy4f_to_f32(lw.ix_ape, ix.ape)).empty()) { err = e; break; }
            {   // the pooling weights on the card too: a decode row's group keys are pooled there
                auto upv = [&](const std::vector<float>& v) -> float* { float* p = dev<float>(c, v.size()); if (p) KpQ{q, "memcpy"}.memcpy(p, v.data(), v.size() * 4).wait(); return p; };
                d.ixg_tok_norm = upv(ix.tok_norm); d.ixg_kv = upv(ix.kv); d.ixg_kg = upv(ix.kg); d.ixg_key_norm = upv(ix.key_norm); d.ixg_ape = upv(ix.ape);
                if (!d.ixg_tok_norm || !d.ixg_kv || !d.ixg_kg || !d.ixg_key_norm || !d.ixg_ape) { err = "device allocation failed (indexer pooling)"; break; }
                d.ixg_ring = dev<float>(c, size_t(8) * ix.hd); if (!d.ixg_ring) { err = "device allocation failed (indexer ring)"; break; }
                if (ix.ratio > 8) d.ixg_kv = nullptr;                  // (the card's pooling holds 8 slots; wider groups stay on the host)
            }
            indexers_.push_back(std::move(ix));
        }
        if (!lw.moe) {
            any_dense_ffn = true;
            d.mlp_gate = upm(lw.mlp_gate, cfg.inter_dim, cfg.dim, "mlp"); d.mlp_up = upm(lw.mlp_up, cfg.inter_dim, cfg.dim, "mlp"); d.mlp_down = upm(lw.mlp_down, cfg.dim, cfg.inter_dim, "mlp"); d.mlp_post_norm = up32(lw.mlp_post_norm);
        } else {
            d.router_w = up32(lw.router_w); d.moe_norm = up32(lw.moe_norm);
            if (auto e = hy4f_to_f32(lw.router_bias, d.router_b); !e.empty()) { err = e; break; }
            d.lat_down = upm(lw.lat_down, cfg.latent_dim, cfg.dim, "lat"); d.lat_up = upm(lw.lat_up, cfg.dim, cfg.latent_dim, "lat");
            d.sh_gate = upm(lw.sh_gate, cfg.moe_inter_dim, cfg.dim, "sh"); d.sh_up = upm(lw.sh_up, cfg.moe_inter_dim, cfg.dim, "sh"); d.sh_down = upm(lw.sh_down, cfg.dim, cfg.moe_inter_dim, "sh");
        }
        if (lw.engram >= 0) {
            any_engram = true;
            const Hy4fEngram& g = m.engrams()[size_t(lw.engram)];
            d.eg_value_w = up16(g.value_w); d.eg_value_b = up32(g.value_b);
            for (const auto& kw : g.key_w) d.eg_key_w.push_back(up16(kw));
            d.eg_key_b = up32_stack(g.key_b); d.eg_norm_key = up32_stack(g.norm_key); d.eg_norm_query = up32_stack(g.norm_query);
            d.eg_scales = up32_stack(g.hc_scales); d.eg_pre_gamma = up32(g.pre_gamma);
            for (const auto& hn : g.head_norms) { if (auto e = hy4f_to_f32(hn, f); !e.empty()) { err = e; break; } d.eg_head_norms.insert(d.eg_head_norms.end(), f.begin(), f.end()); }
        }
    }
    if (!err.empty()) return err;
    if (c.dq_need) { c.dq16 = dev<sycl::half>(c, c.dq_need); if (!c.dq16) return "device allocation failed (the 8-bit matrices' fp16 scratch)"; }

    // ---- workspaces ----
    // this card's half of the OTHER card's split layers (it computes them while that card computes its own half)
    if (other) {
        c.hL0 = other->L0; c.helper.assign(other->L1 - other->L0, Dense{});
        for (uint32_t L = other->L0; L < other->L1 && err.empty(); ++L) {
            const Hy4fLayer& lw = m.layers()[L];
            if (!can_split(lw)) continue;
            Dense& d = c.helper[L - other->L0];
            attn_heads(d, lw, L, lw.n_heads / 2, lw.n_heads - lw.n_heads / 2, false);
            c.nh_max = std::max(c.nh_max, d.nh); c.nh_min = c.nh_min ? std::min(c.nh_min, d.nh) : d.nh;
        }
        if (!err.empty()) return err;
    }
    // the drafter: on the last card, when its tensors are there and every target layer it reads lives on this card
    // (asked for with Hy4fOptions::drafter; a mask token outside the embedding table is a broken checkpoint: no drafter)
    if (&c == cards_.back().get() && opt_.drafter && dfly_on() && cfg.dfly_layers && cfg.dfly_drafts && cfg.dfly_mask_id >= 0 && uint32_t(cfg.dfly_mask_id) < cfg.vocab_size &&
        m.embed.w && m.embed.w->shape.size() == 2 && uint64_t(cfg.dfly_mask_id) < uint64_t(m.embed.w->shape[0]) && cfg.dfly_kv_layers.size() == cfg.dfly_layers &&
        opt_.max_tokens >= cfg.dfly_drafts + 1 && cfg.dfly_drafts + 1 <= kSmallRows && m.source().find("model.dfly_layers.final_norm.weight")) {
        bool here = true;
        for (uint32_t L : cfg.dfly_kv_layers) here = here && L >= c.L0 && L < c.L1 && m.layers()[L].indexed;
        auto T_ = [&](const std::string& n) { Ds41Tensor t; t.w = m.source().find(n); if (!t.w && err.empty()) err = "the drafter's " + n + " is missing"; return t; };
        if (here) {
            dfly_ = std::make_unique<Dfly>(); Dfly& D = *dfly_;
            D.W = cfg.dfly_drafts + 1; D.layers.assign(cfg.dfly_layers, Dense{}); D.snap.assign(cfg.dfly_layers, {});
            for (uint32_t i2 = 0; i2 < cfg.dfly_layers && err.empty(); ++i2) {
                const std::string p = "model.dfly_layers." + std::to_string(i2) + ".";
                Hy4fLayer lf; lf.indexed = true;
                lf.q_a = T_(p + "self_attn.q_a_proj.weight"); lf.q_a_norm = T_(p + "self_attn.q_a_layernorm.weight"); lf.q_b = T_(p + "self_attn.q_b_proj.weight");
                lf.kv_a = T_(p + "self_attn.kv_a_proj_with_mqa.weight"); lf.kv_a_norm = T_(p + "self_attn.kv_a_layernorm.weight"); lf.kv_b = T_(p + "self_attn.kv_b_proj.weight");
                lf.gate = T_(p + "self_attn.linear_gate.weight"); lf.o = T_(p + "self_attn.o_proj.weight"); lf.sinks = T_(p + "self_attn.learnable_sink_param");
                const Ds41Tensor pn = T_(p + "self_attn.post_proj_norm.weight"), l1 = T_(p + "input_layernorm.weight"), l2 = T_(p + "post_attention_layernorm.weight");
                const Ds41Tensor mg = T_(p + "mlp.gate_proj.weight"), mu = T_(p + "mlp.up_proj.weight"), md = T_(p + "mlp.down_proj.weight");
                if (!err.empty()) break;
                lf.n_heads = uint32_t(lf.sinks.w->numel());
                Dense& d = D.layers[i2];
                attn_heads(d, lf, 0, 0, lf.n_heads, false, false);
                if (!err.empty()) break;
                d.post_norm = up32(pn); D.ln1.push_back(up32(l1)); D.ln2.push_back(up32(l2));
                D.FI = uint32_t(mg.w->shape[0]);
                d.mlp_gate = upm(mg, D.FI, cfg.dim, "mlp"); d.mlp_up = upm(mu, D.FI, cfg.dim, "mlp"); d.mlp_down = upm(md, cfg.dim, D.FI, "mlp");
                c.nh_max = std::max(c.nh_max, d.nh);
            }
            if (err.empty()) {
                D.final_norm = up32(T_("model.dfly_layers.final_norm.weight"));
                D.enorm = up32(T_("model.dfly_layers.mtp_fusion.enorm.weight")); D.hnorm = up32(T_("model.dfly_layers.mtp_fusion.hnorm.weight")); D.fgate = up32(T_("model.dfly_layers.mtp_fusion.gate"));
                D.eh_proj = up16(T_("model.dfly_layers.mtp_fusion.eh_proj.weight"));
                D.hidden_norm = up32(T_("model.dfly_layers.hidden_correction.hidden_norm.weight")); D.embed_norm = up32(T_("model.dfly_layers.hidden_correction.embed_norm.weight"));
                D.gate_proj = up16(T_("model.dfly_layers.hidden_correction.gate_proj.weight")); D.up_proj = up16(T_("model.dfly_layers.hidden_correction.up_proj.weight"));
                D.down_proj = up16(T_("model.dfly_layers.hidden_correction.down_proj.weight"));
                Ds41Tensor cw, cb; cw.w = m.source().find("model.dfly_layers.confidence_head.proj.weight"); cb.w = m.source().find("model.dfly_layers.confidence_head.proj.bias");
                std::vector<float> fb;
                if (cw.w && cb.w && hy4f_to_f32(cw, D.conf_w).empty() && hy4f_to_f32(cb, fb).empty() && D.conf_w.size() == cfg.dim && fb.size() == 1) D.conf_b = fb[0]; else D.conf_w.clear();
            }
            if (err.empty()) {
                const size_t W = D.W, Hd = cfg.dim, S = size_t(cfg.ix_group_topk) * cfg.ix_ratio;
                bool k = true;
                auto F = [&](float*& ptr, size_t n) { ptr = dev<float>(c, n); k = k && ptr; };
                auto Hh = [&](sycl::half*& ptr, size_t n) { ptr = dev<sycl::half>(c, n); k = k && ptr; };
                F(D.th32, Hd); Hh(D.th16, size_t(kSmallRows) * Hd); F(D.x32, W * Hd); F(D.e32, W * Hd); F(D.h32, W * Hd); F(D.f32, Hd); F(D.pe32, Hd); F(D.g32, Hd); F(D.u32, Hd); F(D.d32, Hd);
                F(D.mg32, W * D.FI); F(D.mu32, W * D.FI); Hh(D.mh16, W * D.FI); Hh(D.z16, 2 * Hd); Hh(D.hm16, Hd); Hh(D.hc16, Hd); Hh(D.keys, (S + W + 64) * LW);
                D.sel_dev = dev<int32_t>(c, S); k = k && D.sel_dev;
                const size_t nLd = D.layers.size();
                D.pin_e = sycl::malloc_host<float>((W + 1) * Hd, *c.q); D.pin_sel = sycl::malloc_host<int32_t>(nLd * S, *c.q);
                D.pin_rng = sycl::malloc_host<int32_t>(nLd * 2 * W, *c.q); D.pin_hc = sycl::malloc_host<sycl::half>(Hd, *c.q);
                for (void* pp : {static_cast<void*>(D.pin_e), static_cast<void*>(D.pin_sel), static_cast<void*>(D.pin_rng), static_cast<void*>(D.pin_hc)}) if (pp) c.pinned.push_back(pp);
                k = k && D.pin_e && D.pin_sel && D.pin_rng && D.pin_hc;
                if (k) KpQ{q, "fill"}.fill(D.keys, sycl::half(0.f), (S + W + 64) * LW).wait();      // (the rows past a key list are read, never used)
                if (!k) err = "device allocation failed (the drafter's scratch)";
                else D.on = true;
            }
            if (!err.empty()) return err;
        }
    }
    c.any_indexed = any_indexed; c.any_dense_ffn = any_dense_ffn; c.any_engram = any_engram; c.other_card = other;
    if (auto e = rows_setup(c); !e.empty()) return e;
    return {};
}


std::string Hy4fForward::alloc_rows(Card& c, size_t T) {
    const auto& cfg = m_->config();
    const uint32_t H = cfg.dim, hc = cfg.hc_mult, KL = cfg.kv_lora_rank, RD = cfg.rope_dim, ND = cfg.nope_dim, VD = cfg.v_dim, LW = KL + RD;
    const bool any_indexed = c.any_indexed, any_dense_ffn = c.any_dense_ffn, any_engram = c.any_engram; Card* other = c.other_card;
    c.arena_on = true;
    struct Off { Card& c; ~Off() { c.arena_on = false; } } off_{c};
    const size_t NH = c.nh_max, QL = cfg.q_lora_rank, QK = ND + RD, E = cfg.n_experts;
    bool ok = true;
    auto A32 = [&](float*& p, size_t n) { p = dev<float>(c, n); ok = ok && p; };
    auto A16 = [&](sycl::half*& p, size_t n) { p = dev<sycl::half>(c, n); ok = ok && p; };
    auto PIN = [&](float*& p, size_t n) { if (p) return; p = sycl::malloc_host<float>(n ? n : 1, *c.q); ok = ok && p; if (p) c.pinned.push_back(p); };
    PIN(c.pin_S, T * hc * H); PIN(c.pin_cos, T * cfg.rope_dim * 2); PIN(c.pin_rlog, T * cfg.n_experts); PIN(c.pin_lat, T * cfg.latent_dim); PIN(c.pin_ycpu, T * cfg.latent_dim);
    PIN(c.pin_emb, T * cfg.engram_width()); PIN(c.pin_logits, std::min<size_t>(T, kLogitRows) * cfg.vocab_size);
    if (c.n_kv_stage) { if (!c.pin_kv) c.pin_kv = sycl::malloc_host<sycl::half>(size_t(c.n_kv_stage) * T * (cfg.kv_lora_rank + cfg.rope_dim), *c.q); ok = ok && c.pin_kv; if (c.pin_kv) c.pinned.push_back(c.pin_kv); }
    if (any_indexed) { PIN(c.pin_ixq, T * cfg.ix_heads * cfg.ix_head_dim); PIN(c.pin_ixw, T * cfg.ix_heads); if (!c.pin_allow) c.pin_allow = sycl::malloc_host<uint8_t>(kPinAllow, *c.q); ok = ok && c.pin_allow; if (c.pin_allow) c.pinned.push_back(c.pin_allow); }
    if (!c.pin_rng) c.pin_rng = sycl::malloc_host<int32_t>(T * 4, *c.q); ok = ok && c.pin_rng; if (c.pin_rng) c.pinned.push_back(c.pin_rng);
    if (other) { if (!c.pin_tpx) c.pin_tpx = sycl::malloc_host<sycl::half>(T * H, *c.q); ok = ok && c.pin_tpx; if (c.pin_tpx) c.pinned.push_back(c.pin_tpx); PIN(c.pin_tpo, T * H); c.tp_o32 = dev<float>(c, T * H); ok = ok && c.tp_o32; }
    A32(c.S, T * hc * H); A32(c.mix, T * (2 * hc + 1)); A32(c.mix_part, T * (2 * hc + 1) * kHcParts); A32(c.pre, T * hc); A32(c.post, T * hc);
    A32(c.x, T * H); A32(c.xn32, T * H); A32(c.qa32, T * QL); A32(c.qr32, T * QL);
    A32(c.q32, T * NH * QK); A32(c.qabs32, T * NH * KL); A32(c.kv32, T * LW);
    A32(c.scores, score_elems()); A32(c.v32, T * NH * VD); A32(c.gate32, T * NH * VD); A32(c.o32, T * H); A32(c.y32, T * H);
    A16(c.xn16, T * H); A16(c.qr16, T * QL); A16(c.qn16, T * NH * ND); A16(c.a16, T * NH * LW); A16(c.p16, score_elems());
    A16(c.ctx16, T * NH * LW); A16(c.attn16, T * NH * VD);
    if (any_indexed) {
        A32(c.ix_tk32, T * cfg.ix_head_dim); A32(c.ix_q32, T * cfg.ix_heads * cfg.ix_head_dim); A32(c.ix_hw32, T * cfg.ix_heads);
        const size_t S = size_t(cfg.ix_group_topk) * cfg.ix_ratio;
        c.sel = dev<int32_t>(c, T * S); ok = ok && c.sel; A16(c.kg16, S * LW);
        A32(c.ix_qrot, T * cfg.ix_heads * cfg.ix_head_dim); A32(c.ix_cs, T * cfg.ix_head_dim); A32(c.ix_sc, kIxScoreElems);
        c.ix_own = dev<int32_t>(c, T); ok = ok && c.ix_own;
        PIN(c.pin_ixcs, T * cfg.ix_head_dim); PIN(c.pin_gcs, size_t(9) * cfg.ix_head_dim); A32(c.ix_gcs, cfg.ix_head_dim); if (!c.pin_ixown) c.pin_ixown = sycl::malloc_host<int32_t>(T, *c.q); ok = ok && c.pin_ixown; if (c.pin_ixown) c.pinned.push_back(c.pin_ixown);
    }
    const size_t FI = std::max<size_t>(any_dense_ffn ? cfg.inter_dim : 0, cfg.moe_inter_dim);
    A32(c.ff_g32, T * FI); A32(c.ff_u32, T * FI); A16(c.ff_h16, T * FI);
    A32(c.rlogits, T * E); A32(c.lat32, T * cfg.latent_dim); A32(c.moe32, T * cfg.latent_dim); A32(c.rout32, T * H); A32(c.sd32, T * H);
    A16(c.lat16, T * cfg.latent_dim);
    A32(c.cs_s, T * RD); A32(c.cs_i, T * RD);
    // the recall mask holds rows * n_keys bytes of one attention block; a block of a layer with H heads has up to
    // score_elems / (H * n_keys) rows, so the layer with the fewest heads sets the size (found in review 2026-11-18)
    c.lo = dev<int32_t>(c, T); c.hi = dev<int32_t>(c, T);
    c.lo_s = dev<int32_t>(c, T); c.hi_s = dev<int32_t>(c, T); c.lo_i = dev<int32_t>(c, T); c.hi_i = dev<int32_t>(c, T); ok = ok && c.lo_s && c.hi_s && c.lo_i && c.hi_i; c.allow_cap = score_elems() / std::max<size_t>(1, c.nh_min); c.allow = dev<uint8_t>(c, c.allow_cap);
    ok = ok && c.lo && c.hi && c.allow;
    if (any_engram) {
        const size_t W = cfg.engram_width();
        A32(c.eg_mask, T); PIN(c.pin_egmask, T); if (c.eg_mask && !c.measuring) { KpQ{*c.q, "fill"}.fill(c.eg_mask, 1.f, T).wait(); } c.eg_mask_dirty = 0;
        A32(c.eg_emb32, T * W); A16(c.eg_emb16, T * W); A32(c.eg_v32, T * H); A32(c.eg_k32, T * hc * H); A32(c.eg_stat, T * (hc + 1));
    }
    if (opt_.q8_experts) {
        const size_t MI = cfg.moe_inter_dim, LAT = cfg.latent_dim;
        A16(c.ex_x16, T * LAT); A16(c.ex_h16, T * MI); A16(c.q8_g16, T * MI); A16(c.q8_u16, T * MI); A16(c.q8_y16, T * LAT); A32(c.ex_w, T);
        const size_t P = T * cfg.top_k;
        A16(c.q8_hj, size_t(cfg.top_k) * MI); A32(c.q8_ycpu, T * LAT);
        A16(c.q8_hrows, P * MI); A32(c.q8_yrows, P * LAT); A32(c.q8_jw, P);
        c.grp_jcap = P / 32 + 3 * cfg.n_experts + 8;
        if (pp_stream()) {
            const uint64_t sb = Hy4fQ8Store::slot_size(uint32_t(MI), uint32_t(LAT), Hy4fQFmt{opt_.q_bits_gu, opt_.q_bits_dn});
            c.stg = dev<uint8_t>(c, size_t(2) * kStageSlots * sb);
            for (int b = 0; b < 2; ++b) { if (!c.pin_stg[b]) c.pin_stg[b] = sycl::malloc_host<uint8_t>(size_t(kStageSlots) * sb, *c.q); if (c.pin_stg[b]) c.pinned.push_back(c.pin_stg[b]); }
            if (!c.qc) c.qc = std::make_unique<sycl::queue>(c.q->get_context(), c.q->get_device(), sycl::property_list{sycl::property::queue::in_order{}});
            ok = ok && c.stg && c.pin_stg[0] && c.pin_stg[1];
        }
        c.grp_jbase = dev<uint64_t>(c, c.grp_jcap); c.grp_j0 = dev<int32_t>(c, c.grp_jcap); c.grp_jn = dev<int32_t>(c, c.grp_jcap);
        ok = ok && c.grp_jbase && c.grp_j0 && c.grp_jn;
        c.q8_jslot = dev<int32_t>(c, P); c.q8_jrow = dev<int32_t>(c, P); c.q8_rs = dev<int32_t>(c, T + 1); ok = ok && c.q8_jslot && c.q8_jrow && c.q8_rs;
        c.ex_rows = dev<int32_t>(c, T); ok = ok && c.ex_rows;
        c.mr_slot = dev<int32_t>(c, kSmallJobs); c.mr_mask = dev<uint32_t>(c, kSmallJobs); c.mr_rowjobs = dev<int32_t>(c, size_t(kSmallRows) * cfg.top_k); c.mr_roww = dev<float>(c, size_t(kSmallRows) * cfg.top_k);
        c.mr_hj = dev<sycl::half>(c, size_t(kSmallJobs) * kSmallRows * MI); c.mr_yj = dev<float>(c, size_t(kSmallJobs) * kSmallRows * LAT);
        if (!c.pin_mr) c.pin_mr = sycl::malloc_host<int32_t>(size_t(kSmallJobs) * 2 + size_t(kSmallRows) * cfg.top_k * 2, *c.q); if (c.pin_mr) c.pinned.push_back(c.pin_mr);
        ok = ok && c.mr_slot && c.mr_mask && c.mr_rowjobs && c.mr_roww && c.mr_hj && c.mr_yj && c.pin_mr;
    }
    if (opt_.exact_experts) {
        const size_t MI = cfg.moe_inter_dim, LAT = cfg.latent_dim;
        A16(c.ex_gu, 2 * MI * LAT); A16(c.ex_dn, LAT * MI); A16(c.ex_x16, T * LAT); A16(c.ex_h16, T * MI);
        A32(c.ex_gu32, T * 2 * MI); A32(c.ex_d32, T * LAT); A32(c.ex_w, T);
        c.ex_rows = dev<int32_t>(c, T); ok = ok && c.ex_rows;
    }
    if (!ok) return "device allocation failed (workspaces)";
    return {};
}

// The arena layouts: the first pass lays out rows_max() rows without memory, which also allocates every pinned host buffer
// at that size (each is allocated once); then max_tokens rows in the card's own arena. big_need: what a big chunk needs lent.
std::string Hy4fForward::rows_setup(Card& c) {
    auto measure = [&](size_t T, uint64_t& need) -> std::string {
        c.measuring = true; c.arena_cur = reinterpret_cast<uint8_t*>(uintptr_t(4096)); c.arena_end = nullptr;
        auto e = alloc_rows(c, T); c.measuring = false;
        need = uint64_t(reinterpret_cast<uintptr_t>(c.arena_cur)) - 4096 + 4096;
        return e;
    };
    uint64_t big = 0;
    if (auto e = measure(rows_max(), big); !e.empty()) return e;
    if (auto e = measure(opt_.max_tokens, c.small_bytes); !e.empty()) return e;
    c.big_need = rows_max() > opt_.max_tokens ? big : 0;
    c.small_arena = dev<uint8_t>(c, c.small_bytes);
    if (!c.small_arena) return "device allocation failed (workspaces)";
    c.arena_cur = c.small_arena; c.arena_end = c.small_arena + c.small_bytes;
    if (auto e = alloc_rows(c, opt_.max_tokens); !e.empty()) return e;
    std::fprintf(stderr, "[hy4-flash] card scratch for %u-row passes: %.2f GiB%s\n", opt_.max_tokens, double(c.small_bytes) / double(1ull << 30),
                 c.big_need ? (", " + std::to_string(rows_max()) + "-row prompt chunks: " + std::to_string(double(c.big_need) / double(1ull << 30)).substr(0, 4) + " GiB lent by the expert slots").c_str() : "");
    return {};
}

// A call of more than max_tokens rows runs in the lent arena; the first call of max_tokens rows or fewer after it gives the
// slots back (and refills them). Every card waits first: no kernel may still read either range.
std::string Hy4fForward::rows_mode(bool big) {
    bool any = false; for (auto& cp : cards_) any = any || cp->big != big;
    if (!any) return {};
    for (auto& cp : cards_) { cp->q->wait(); if (cp->qc) cp->qc->wait(); }
    for (auto& cp : cards_) {
        Card& c = *cp;
        if (c.big == big) continue;
        try {
            if (big) {
                uint8_t* p = c.q8.lend(*c.q);
                if (!p) return "lending the expert slots failed";
                c.arena_cur = p; c.arena_end = p + c.q8.lendable();
                if (auto e = alloc_rows(c, rows_max()); !e.empty()) return e;
            } else {
                c.q8.take_back(*c.q, st_.rows);
                c.arena_cur = c.small_arena; c.arena_end = c.small_arena + c.small_bytes;
                if (auto e = alloc_rows(c, opt_.max_tokens); !e.empty()) return e;
            }
        } catch (const sycl::exception& x) { return std::string("lending the expert slots: ") + x.what(); }
        c.big = big;
    }
    return {};
}

std::string Hy4fForward::init(const std::vector<sycl::queue*>& qs, const Hy4FlashModel& m, const Hy4fOptions& o) {
    free_all();
    m_ = &m; opt_ = o;
    const auto& cfg = m.config();
    if (qs.empty() || qs.size() > 2) return "one or two cards";
    for (auto* q : qs) if (!q || !q->is_in_order()) return "the queues must be in-order";
    if (cfg.n_experts > 1024) return "more than 1024 routed experts";
    if ((cfg.hc_mult * cfg.dim) % (kHcParts * 4)) return "hc_mult * hidden_size must be a multiple of 256";
    if (cfg.hc_mult > 8) return "more than 8 residual streams";
    if (opt_.max_tokens == 0 || opt_.max_ctx == 0) return "max_tokens / max_ctx must be positive";
    ctx_ = (opt_.max_ctx + 63u) / 64u * 64u;
    const uint32_t nL = cfg.n_layers, hc = cfg.hc_mult, H = cfg.dim;
    uint32_t split = qs.size() == 2 ? (opt_.split_layer ? opt_.split_layer : nL / 2) : nL;
    if (split == 0 || split > nL) return "split_layer out of range";
    uint32_t first_moe = nL;
    for (uint32_t L = 0; L < nL; ++L) if (cfg.moe[L]) { first_moe = L; break; }
    for (uint32_t L = first_moe; L < nL; ++L) if (!cfg.moe[L]) return "a dense layer after the first MoE layer is not supported";
    if (opt_.q8_experts && opt_.exact_experts) return "q8_experts and exact_experts are two different expert paths: pick one";
    if (first_moe < nL && !m.has_pack() && !opt_.exact_experts && !opt_.q8_experts) return "the directory has no complete expert pack (run scripts/hy4flash_oracle/pack.py)";

    std::vector<std::vector<uint32_t>> ranking = opt_.ranking;
    if (ranking.empty()) { ranking.assign(nL, std::vector<uint32_t>(cfg.n_experts)); for (auto& r : ranking) std::iota(r.begin(), r.end(), 0u); }

    try {
        // every card and its layer range first: a card loads half the heads of the OTHER card's split layers, so it
        // must know that card when it is uploaded (2026-11-19: created one by one, the first card built no helper halves
        // and the second read an empty list -- three copy-engine faults on a card)
        for (size_t ci = 0; ci < qs.size(); ++ci) {
            cards_.push_back(std::make_unique<Card>());
            Card& c = *cards_.back();
            c.q = qs[ci]; c.L0 = ci == 0 ? 0 : split; c.L1 = (ci + 1 == qs.size()) ? nL : split;
            // Elastic context (IE_HY4F_ELASTIC_CTX=0: the whole cache at load, as before 2026-11-19): with the 8-bit slots in
            // the lending layout, an indexed layer's cache is backed for 32,768 tokens and grows by tiers (x2) to ctx
            static const bool elastic = [] { const char* v = std::getenv("IE_HY4F_ELASTIC_CTX"); return !(v && *v == '0'); }();
            const uint32_t el_first = 32768;
            if (elastic && opt_.q8_experts && opt_.big_tokens > opt_.max_tokens && ctx_ > el_first && std::max(c.L0, first_moe) < c.L1) {
                const uint64_t g = 2ull << 20, row = (uint64_t(cfg.kv_lora_rank) + cfg.rope_dim) * 2;
                c.el_full = (uint64_t(ctx_) * row + g - 1) / g * g;
                for (uint64_t cap = el_first; cap < ctx_; cap *= 2) c.el_caps.push_back(uint32_t(cap));
                c.el_caps.push_back(ctx_);
                for (uint32_t cap : c.el_caps) c.el_back.push_back(cap == ctx_ ? c.el_full : std::min(c.el_full, (uint64_t(cap) * row + g - 1) / g * g));
                c.el_on = true;
            }
        }
        for (size_t ci = 0; ci < qs.size(); ++ci) {
            Card& c = *cards_[ci];
            if (auto e = upload_card(c); !e.empty()) return "card " + std::to_string(ci) + ": " + e;
            sycl::queue& q = *c.q;
            if (ci + 1 == qs.size()) {   // the head lives with the last layers
                std::vector<float> f;
                auto up32 = [&](const Ds41Tensor& t, float*& p) -> std::string {
                    if (auto e = hy4f_to_f32(t, f); !e.empty()) return e;
                    p = dev<float>(c, f.size()); if (!p) return "device allocation failed (head)";
                    KpQ{q, "memcpy"}.memcpy(p, f.data(), f.size() * 4).wait(); return {};
                };
                std::string e;
                if (!(e = up32(m.final_norm, c.final_norm)).empty() || !(e = up32(m.head_fn, c.head_fn)).empty() ||
                    !(e = up32(m.head_base, c.head_base)).empty() || !(e = up32(m.head_scale, c.head_scale)).empty()) return e;
                const size_t n = size_t(cfg.vocab_size) * H;
                // the logits of at most kLogitRows rows at a time (a chunk of a prompt needs its last row's; the full-row modes
                // run the head block by block): 0.5 MB a row of the chunk was VRAM the expert slots did not get
                c.head = dev<sycl::half>(c, n); c.logits = dev<float>(c, size_t(std::min<uint32_t>(opt_.max_tokens, kLogitRows)) * cfg.vocab_size);
                if (!c.head || !c.logits) return "device allocation failed (head)";
                if (m.lm_head.w->dtype_str != "BF16") return "lm_head is not BF16";
                KpQ{q, "memcpy"}.memcpy(c.head, m.lm_head.w->data, n * 2); bf16_to_f16_inplace(q, c.head, n).wait();
            }
            const uint32_t first = std::max(c.L0, first_moe);
            if (first < c.L1 && opt_.q8_experts) {
                const Hy4fQFmt qfmt{opt_.q_bits_gu, opt_.q_bits_dn};
                uint32_t slots = ci < opt_.q8_slots_card.size() && opt_.q8_slots_card[ci] ? opt_.q8_slots_card[ci] : opt_.q8_slots;
                {   // the store is the card's last big allocation: size it to what is free, or check what was asked for
                    const sycl::device dv = q.get_device();
                    const uint64_t per_slot = uint64_t(c.L1 - first) * Hy4fQ8Store::slot_size(cfg.moe_inter_dim, cfg.latent_dim, qfmt);
                    if (dv.has(sycl::aspect::ext_intel_free_memory)) {
                        q.wait();
                        // the margin left free: a serve session with an image, a 25,216-token prompt and a 600-token reply took at most
                        // 0.40 GiB of it (IE_HY4F_VRAM_LOW=1, 2026-11-19); 0.8 GiB keeps that twice. IE_HY4F_VRAM_RESERVE_MB (was 1434)
                        static const uint64_t reserve_mb = [] { const char* v = std::getenv("IE_HY4F_VRAM_RESERVE_MB"); const long long x = v ? std::atoll(v) : 0; return x > 0 ? uint64_t(x) : uint64_t(820); }();
                        const uint64_t fr = dv.get_info<sycl::ext::intel::info::device::free_memory>(), reserve = reserve_mb << 20;
                        const uint32_t fit = fr > reserve ? uint32_t(std::min<uint64_t>((fr - reserve) / per_slot, cfg.n_experts)) : 0;
                        // sized from free VRAM, which moves by tens of MB from one load to the next: rounded down to a multiple
                        // of 4, so the count -- which experts run on the card and which on the CPU, and with it the exact
                        // tokens -- is the same load after load (2026-11-19: two loads gave different greedy text; pinned, the same)
                        // (a multiple of 2 was tried 2026-11-19: card 1 came up with 164 slots on one load and 166 on the next)
                        if (!slots) slots = fit >= 4 ? fit / 4 * 4 : fit;
                        else if (slots > fit) { std::fprintf(stderr, "card %zu: %u expert slots a layer asked for, %u fit its free VRAM (%.2f GiB) with the margin -- using %u\n", ci, slots, fit, double(fr) / double(1ull << 30), fit); slots = fit; }
                        {   // where the card's memory goes (2026-11-19: is all of the VRAM used?)
                            const double G = double(1ull << 30), tot = double(dv.get_info<sycl::info::device::global_mem_size>());
                            const double ex = double(std::min(slots, fit)) * double(per_slot);
                            std::fprintf(stderr, "[hy4-flash] card %zu VRAM %.2f GiB: weights and scratch %.2f, context cache %.2f (ctx %u), other (driver, image tower) %.2f, "
                                         "experts %.2f (%u slots a layer x %u layers), left free %.2f\n", ci, tot / G, double(c.bytes - c.kv_bytes) / G, double(c.kv_bytes) / G, ctx_,
                                         (tot - double(fr) - double(c.bytes)) / G, ex / G, std::min(slots, fit), c.L1 - first, (double(fr) - ex) / G);
                        }
                    } else if (!slots) slots = 120;      // (free VRAM not readable: the bring-up count)
                }
                q8_slots_used_.resize(qs.size(), 0); q8_slots_used_[ci] = slots;
                const uint64_t lend = c.big_need ? (c.big_need + (c.L1 - first) - 1) / (c.L1 - first) + (1ull << 20) : 0;
                std::vector<uint64_t> pieces;               // the VRAM each later context tier takes from the slots, a layer
                if (c.el_on) for (size_t t = 0; t + 1 < c.el_caps.size(); ++t)
                    pieces.push_back((uint64_t(c.el_va.size()) * (c.el_back[t + 1] - c.el_back[t]) + (c.L1 - first) - 1) / (c.L1 - first));
                if (ci == 0 && opt_.extra_piece_bytes) { c.extra_piece = int32_t(pieces.size()); pieces.push_back((opt_.extra_piece_bytes + (c.L1 - first) - 1) / (c.L1 - first)); }
                if (auto e = c.q8.init(q, m, first, c.L1 - first, slots, ranking, qfmt, lend, pieces); !e.empty()) return "card " + std::to_string(ci) + " 8-bit experts: " + e;
                if (c.q8.ctx_pieces() != pieces.size()) return "card " + std::to_string(ci) + ": the expert slots cannot hold the context pieces (IE_HY4F_ELASTIC_CTX=0 allocates the whole context cache)";
                if (c.el_on) std::fprintf(stderr, "[hy4-flash] card %zu: context cache backed for %u tokens (%.2f GiB); the VRAM of the rest (to %u tokens, %.2f GiB) holds expert slots until a conversation needs it\n",
                                          ci, c.el_caps[0], double(c.el_va.size() * c.el_back[0]) / double(1ull << 30), ctx_, double(c.el_va.size() * (c.el_full - c.el_back[0])) / double(1ull << 30));
                c.q8_on = true; st_.q8_build_s += c.q8.stats().build_s;
            }
            if (first < c.L1 && !opt_.exact_experts && !opt_.q8_experts) {
                Ds41ExpertSource s;
                s.store = &m.pack(); s.H = cfg.latent_dim; s.EF = cfg.moe_inter_dim; s.n_experts = cfg.n_experts; s.top_k = cfg.top_k;
                s.n_text_layers = nL;
                s.layers.resize(nL);
                for (uint32_t L = 0; L < nL; ++L) {
                    const auto& lw = m.layers()[L];
                    if (lw.moe) s.layers[L] = Ds41ExpertLayer{lw.exp_w1.data(), lw.exp_w3.data(), lw.exp_w2.data()};
                }
                s.fp32_out = true;
                const uint32_t E = cfg.n_experts;
                const uint32_t n_static = std::min(E, opt_.n_static ? opt_.n_static : E);
                const uint32_t n_pinned = std::min(E - n_static, opt_.n_pinned ? opt_.n_pinned : E - n_static);
                if (auto e = c.tier.init(q, s, first, c.L1 - first, ranking, n_static, n_pinned, opt_.stream_slots, rows_max()); !e.empty())
                    return "card " + std::to_string(ci) + " expert tier: " + e;
                c.tier_on = true; c.tier_L0 = first;
            }
        }
    } catch (const sycl::exception& e) {
        return std::string("sycl: ") + e.what();
    }
    big_ok_ = rows_max() > opt_.max_tokens && !cards_.empty();
    for (auto& cp : cards_) big_ok_ = big_ok_ && cp->q8_on && cp->q8.lendable() >= cp->big_need;
    if (rows_max() > opt_.max_tokens && !big_ok_) std::fprintf(stderr, "[hy4-flash] %u-row prompt chunks need the 8-bit experts' slots to lend: chunks stay at %u rows\n", rows_max(), opt_.max_tokens);
    const size_t T = rows_max();
    h_x_.resize(T * hc * H); h_stage_.resize(T * std::max<size_t>({size_t(hc) * H, size_t(cfg.n_experts), size_t(cfg.ix_heads) * cfg.ix_head_dim}));
    h_cos_s_.resize(T * cfg.rope_dim); h_cos_i_.resize(T * cfg.rope_dim);
    h_ridx_.resize(T * cfg.top_k); h_rw_.resize(T * cfg.top_k); h_lo_.resize(T); h_hi_.resize(T);
    routes_.assign(nL, {});
    { const double b = st_.q8_build_s; st_ = {}; st_.q8_build_s = b; }
    { const char* v = std::getenv("IE_HY4F_PROF"); prof_on_ = v && *v == '1'; prof_.clear(); }
    n_pos_ = 0; hist_.clear();
    return {};
}

// y32 [T, N] = x16 [T, K] . W^T for a matrix held as fp16 or as 8-bit (q8_dense)
void Hy4fForward::mm(Card& c, const Mat& w, const sycl::half* x16, float* y32, uint32_t T, uint32_t N, uint32_t K) {
    sycl::queue& q = *c.q;
    if (w.w16) { mm16(q, x16, w.w16, y32, T, N, K); return; }
    if (T == 1) { hy4f_q8_gemv_f32(q, x16, w.qs, w.d, y32, K, N, T); return; }
    if (T <= kSmallRows) { hy4f_q8_gemv_rows_f32(q, x16, w.qs, w.d, y32, K, N, T); return; }   // a few rows: the matrix read once for all of them
    hy4f_q8_dequant_f16(q, w.qs, w.d, c.dq16, N, K);
    gemm_nt_f16_onednn(q, x16, c.dq16, y32, T, N, K);
}

std::string Hy4fForward::engram(Card& c, const Dense& d, const Hy4fEngram& eg, uint32_t T, uint32_t pos0) {
    const auto& cfg = m_->config(); sycl::queue& q = *c.q;
    const uint32_t H = cfg.dim, hc = cfg.hc_mult, NHd = cfg.engram_n_heads(), HD = cfg.engram_head_dim(), W = cfg.engram_width();
    const float deps = cfg.torch_default_eps;
    // host: hash -> table rows -> fp8 round trip -> per-head norm
    h_emb_.resize(size_t(T) * W);
    std::vector<int32_t>& lookup = eg_lookup_[eg.id];             // (aligned copy of the token -> compressed id table, made once)
    if (lookup.empty()) { lookup.resize(eg.table_size); std::memcpy(lookup.data(), eg.lookup.w->data, size_t(eg.table_size) * 4); }
    const bool f8 = eg.fp8 != nullptr;
    if (!f8 && eg.table.w->dtype_str != "BF16") return "the n-gram table is not BF16";
    const uint8_t* tab = f8 ? eg.fp8 : eg.table.w->data;
    const size_t stride = f8 ? 4 + size_t(HD) : size_t(HD) * 2;        // an FP8 row: its scale, then one code a value
    int32_t ngram[8];
    // every row this call needs, paged in together (the tables are mapped from the drive), then read
    h_eg_rows_.resize(size_t(T) * NHd); h_eg_addr_.resize(size_t(T) * NHd);
    for (uint32_t t = 0; t < T; ++t) {
        const uint32_t p = pos0 + t;
        for (uint32_t j = 0; j < cfg.engram_max_ngram; ++j) ngram[j] = p >= j ? hist_[p - j] : int32_t(cfg.engram_pad_id);
        hy4f_engram_rows(eg, cfg, lookup.data(), ngram, h_eg_rows_.data() + size_t(t) * NHd);
        for (uint32_t h = 0; h < NHd; ++h) h_eg_addr_[size_t(t) * NHd + h] = tab + h_eg_rows_[size_t(t) * NHd + h] * stride;
    }
    hy4f_prefault(h_eg_addr_.data(), h_eg_addr_.size(), stride, 0);
    for (uint32_t t = 0; t < T; ++t) {
        const uint64_t* rows = h_eg_rows_.data() + size_t(t) * NHd;
        for (uint32_t h = 0; h < NHd; ++h) {
            float* e = h_emb_.data() + size_t(t) * W + size_t(h) * HD;
            const uint8_t* src = tab + rows[h] * stride;
            if (f8) hy4f_fp8_decode_row(src, HD, e);                   // (the stored rounding: the same values, bit for bit)
            else {
                for (uint32_t k = 0; k < HD; ++k) { uint16_t b; std::memcpy(&b, src + size_t(k) * 2, 2); const uint32_t u = uint32_t(b) << 16; std::memcpy(&e[k], &u, 4); }
                if (cfg.engram_fp8) hy4f_fp8_round_trip(e, HD);
            }
            double ms = 0;
            for (uint32_t k = 0; k < HD; ++k) ms += double(e[k]) * e[k];
            const float r = 1.0f / std::sqrt(float(ms / HD) + deps);
            const float* w = d.eg_head_norms.data() + size_t(h) * HD;
            for (uint32_t k = 0; k < HD; ++k) e[k] = e[k] * r * w[k];
        }
    }
    std::memcpy(c.pin_emb, h_emb_.data(), size_t(T) * W * 4);
    copy_words(q, c.eg_emb32, c.pin_emb, size_t(T) * W * 4, "pin_to_card");
    if (eg_any_masked_ || c.eg_mask_dirty) {       // image rows add nothing here (the reference's engram_mask); the buffer holds 1 otherwise
        const uint32_t n = std::max(T, c.eg_mask_dirty);
        for (uint32_t t = 0; t < n; ++t) c.pin_egmask[t] = t < T && !h_egmask_[t] ? 0.f : 1.f;
        copy_words(q, c.eg_mask, c.pin_egmask, size_t(n) * 4, "pin_to_card");
        c.eg_mask_dirty = eg_any_masked_ ? T : 0;
    }
    cast_fp32_to_fp16(q, c.eg_emb32, c.eg_emb16, size_t(T) * W);
    mm16(q, c.eg_emb16, d.eg_value_w, c.eg_v32, T, H, W);
    for (uint32_t i = 0; i < hc; ++i) mm16(q, c.eg_emb16, d.eg_key_w[i], c.eg_k32 + size_t(i) * T * H, T, H, W);
    // stat[t, i < hc] = the stream's gate; stat[t, hc] = rsqrt(mean(v^2) + eps) of the value (bias included)
    {
        float* S = c.S; float* stat = c.eg_stat; const float* v = c.eg_v32; const float* k32 = c.eg_k32;
        const float* vb = d.eg_value_b; const float* kb = d.eg_key_b; const float* nk = d.eg_norm_key; const float* nq = d.eg_norm_query;
        const float* sc = d.eg_scales; const float* pg = d.eg_pre_gamma;
        const float hinv = 1.0f / std::sqrt(float(H));
        if (smallk()) {    // 16 lanes a (row, stream): three sums through the sub-group instead of one work-item over the width
            constexpr int SG = 16;
            KpQ{q, "Hy4fEngramStat2K"}.parallel_for<class Hy4fEngramStat2K>(sycl::nd_range<3>({size_t(T), size_t(hc) + 1, size_t(SG)}, {1, 1, size_t(SG)}), [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(SG)]] {
                const size_t t = it.get_global_id(0); const uint32_t i = uint32_t(it.get_global_id(1)), lane = uint32_t(it.get_local_id(2));
                if (i == hc) {
                    float a = 0.f;
                    for (uint32_t dd = lane; dd < H; dd += SG) { const float x = v[t * H + dd] + vb[dd]; a += x * x; }
                    a = sycl::reduce_over_group(it.get_sub_group(), a, sycl::plus<float>());
                    if (lane == 0) stat[t * (hc + 1) + hc] = sycl::rsqrt(a / float(H) + deps);
                    return;
                }
                const float* kk = k32 + (size_t(i) * T + t) * H; const float* s = S + (t * hc + i) * H;
                float mk = 0.f, mq = 0.f, dot = 0.f;
                for (uint32_t dd = lane; dd < H; dd += SG) {
                    const float kv = kk[dd] + kb[size_t(i) * H + dd], sv = s[dd];
                    mk += kv * kv; mq += sv * sv; dot += kv * nk[size_t(i) * H + dd] * sv * nq[size_t(i) * H + dd];
                }
                mk = sycl::reduce_over_group(it.get_sub_group(), mk, sycl::plus<float>());
                mq = sycl::reduce_over_group(it.get_sub_group(), mq, sycl::plus<float>());
                dot = sycl::reduce_over_group(it.get_sub_group(), dot, sycl::plus<float>());
                if (lane != 0) return;
                float g = dot * sycl::rsqrt(mk / float(H) + deps) * sycl::rsqrt(mq / float(H) + deps) * hinv;
                const float mag = sycl::sqrt(sycl::fmax(sycl::fabs(g), 1e-6f));
                g = g < 0.f ? -mag : (g > 0.f ? mag : 0.f);
                stat[t * (hc + 1) + i] = dsig(g);
            });
        } else
        KpQ{q, "Hy4fEngramStatK"}.parallel_for<class Hy4fEngramStatK>(sycl::range<2>(T, hc + 1), [=](sycl::id<2> id) {
            const size_t t = id[0]; const uint32_t i = uint32_t(id[1]);
            if (i == hc) {
                float a = 0.f;
                for (uint32_t dd = 0; dd < H; ++dd) { const float x = v[t * H + dd] + vb[dd]; a += x * x; }
                stat[t * (hc + 1) + hc] = sycl::rsqrt(a / float(H) + deps);
                return;
            }
            const float* kk = k32 + (size_t(i) * T + t) * H; const float* s = S + (t * hc + i) * H;
            float mk = 0.f, mq = 0.f, dot = 0.f;
            for (uint32_t dd = 0; dd < H; ++dd) {
                const float kv = kk[dd] + kb[size_t(i) * H + dd], sv = s[dd];
                mk += kv * kv; mq += sv * sv; dot += kv * nk[size_t(i) * H + dd] * sv * nq[size_t(i) * H + dd];
            }
            float g = dot * sycl::rsqrt(mk / float(H) + deps) * sycl::rsqrt(mq / float(H) + deps) * hinv;
            const float mag = sycl::sqrt(sycl::fmax(sycl::fabs(g), 1e-6f));
            g = g < 0.f ? -mag : (g > 0.f ? mag : 0.f);
            stat[t * (hc + 1) + i] = dsig(g);
        });
        const float* egm = c.eg_mask;
        KpQ{q, "Hy4fEngramAddK"}.parallel_for<class Hy4fEngramAddK>(sycl::range<2>(size_t(T) * hc, H), [=](sycl::id<2> id) {
            const size_t r = id[0]; const uint32_t dd = uint32_t(id[1]); const size_t t = r / hc; const uint32_t i = uint32_t(r % hc);
            const float main = pg[dd] * (v[t * H + dd] + vb[dd]) * stat[t * (hc + 1) + hc] * egm[t];
            S[r * H + dd] += (stat[t * (hc + 1) + i] + sc[size_t(i) * H + dd]) * main;
        });
    }
    return {};
}

std::string Hy4fForward::attention(Card& c, const Dense& d, const Hy4fLayer& lw, uint32_t L, uint32_t T, uint32_t pos0) {
    if (auto e = attention_core(c, d, lw, L, T, pos0); !e.empty()) return e;
    if (!tail_fused_) mimo26_rms_norm(*c.q, c.o32, d.post_norm, nullptr, c.y32, T, m_->config().dim, m_->config().norm_eps);
    return {};
}

bool Hy4fForward::has_drafter() const { return dfly_ && dfly_->on; }
uint32_t Hy4fForward::draft_width() const { return has_drafter() ? dfly_->W - 1 : 0; }

std::string Hy4fForward::draft(uint32_t row, int32_t bonus, std::vector<int32_t>& drafts, const std::function<int32_t(uint32_t, const float*)>& pick,
                               std::vector<float>* conf, float stop_below) {
    drafts.clear(); if (conf) conf->clear();
    if (!has_drafter()) return "this load has no drafter";
    Dfly& D = *dfly_; Card& c = *cards_.back(); sycl::queue& q = *c.q;
    const auto& cfg = m_->config();
    const uint32_t H = cfg.dim, W = D.W, QL = cfg.q_lora_rank, KL = cfg.kv_lora_rank, RD = cfg.rope_dim, ND = cfg.nope_dim, VD = cfg.v_dim, QK = ND + RD, LW = KL + RD, nf = RD / 2;
    const uint32_t V = cfg.vocab_size, S = cfg.ix_group_topk * cfg.ix_ratio;
    const float eps = cfg.norm_eps;
    if (bonus < 0 || uint32_t(bonus) >= V) return "the bonus token is out of range";
    if (m_->embed.w->dtype_str != "BF16") return "the token embedding is not BF16";
    if (row < D.th_row0 || row >= D.th_row0 + D.th_rows) return "the drafter has no final hidden of the target's last call (row " + std::to_string(row) + ")";
    const uint8_t* emb = m_->embed.w->data;
    auto embed_row = [&](int32_t id, float* dst) { const uint8_t* src = emb + size_t(id) * H * 2; for (uint32_t k = 0; k < H; ++k) { uint16_t b; std::memcpy(&b, src + size_t(k) * 2, 2); const uint32_t u = uint32_t(b) << 16; std::memcpy(&dst[k], &u, 4); } };
    // IE_HY4F_DFLY_DUMP=<dir>: the first calls' inputs and intermediates as files, for scripts/hy4flash_oracle/dfly_oracle.py
    static const char* dump_dir = std::getenv("IE_HY4F_DFLY_DUMP"); static int dump_n = 0;
    const bool dump = dump_dir && *dump_dir && dump_n < 8;
    const std::string dpre = dump ? std::string(dump_dir) + "/c" + std::to_string(dump_n) + "." : std::string();
    auto dumpd = [&](const std::string& name, const void* dev_ptr, size_t bytes) {
        if (!dump) return;
        std::vector<uint8_t> hb(bytes); q.wait(); q.memcpy(hb.data(), dev_ptr, bytes).wait();
        std::ofstream f(dpre + name, std::ios::binary); f.write(reinterpret_cast<const char*>(hb.data()), std::streamsize(bytes));
    };
    std::vector<uint32_t> dump_hist;
    try {
        // the target's final hidden of the last accepted row (the head's normed input, kept by forward())
        { float* th = D.th32; const sycl::half* xs = D.th16 + size_t(row - D.th_row0) * H; q.parallel_for(sycl::range<1>(H), [=](sycl::id<1> i) { th[i] = float(xs[i]); }); }
        // the block: [the token just sampled, the mask token ...] at the positions after the sequence
        embed_row(bonus, D.pin_e);
        for (uint32_t j = 1; j < W; ++j) embed_row(cfg.dfly_mask_id, D.pin_e + size_t(j) * H);
        copy_words(q, D.e32, D.pin_e, size_t(H) * 4, "pin_to_card"); copy_words(q, D.x32, D.pin_e, size_t(W) * H * 4, "pin_to_card");
        const int32_t p0 = next_position();
        for (uint32_t j = 0; j < W; ++j) { const int32_t pj = p0 + int32_t(j); const int32_t pos3[3] = {pj, pj, pj}; hy4f_rope_row(cfg.theta_indexed, cfg.mrope_indexed, RD, pos3, c.pin_cos + size_t(j) * RD, c.pin_cos + size_t(j) * RD + nf); }
        copy_words(q, c.cs_i, c.pin_cos, size_t(W) * RD * 4, "pin_to_card");
        // slot 0 += gate * eh_proj([enorm(embedding) | hnorm(target hidden)])
        mimo26_rms_norm(q, D.e32, D.enorm, D.z16, nullptr, 1, H, eps);
        mimo26_rms_norm(q, D.th32, D.hnorm, D.z16 + H, nullptr, 1, H, eps);
        mm16(q, D.z16, D.eh_proj, D.f32, 1, H, 2 * H);
        { float* x = D.x32; const float* f = D.f32; const float* g = D.fgate; q.parallel_for(sycl::range<1>(H), [=](sycl::id<1> i) { x[i] += g[i] * f[i]; }); }
        const float scaling = 1.0f / std::sqrt(float(QK));
        dumpd("th.f32", D.th32, size_t(H) * 4); dumpd("cs.f32", c.cs_i, size_t(W) * RD * 4); dumpd("x0.f32", D.x32, size_t(W) * H * 4);
        for (size_t li = 0; li < D.layers.size(); ++li) {
            const Dense& d = D.layers[li]; const uint32_t NH = d.nh;
            const Dense& tgt = c.dense[cfg.dfly_kv_layers[li] - c.L0];
            mimo26_rms_norm(q, D.x32, D.ln1[li], c.xn16, c.xn32, W, H, eps);
            mm16(q, c.xn16, d.q_a, c.qa32, W, QL, H);
            mimo26_rms_norm(q, c.qa32, d.q_a_norm, c.qr16, c.qr32, W, QL, eps);
            mm(c, d.q_b, c.qr16, c.q32, W, NH * QK, QL);
            q_split(q, c.q32, c.cs_i, c.qn16, c.a16, W, NH, ND, RD, KL, LW);
            if (ND % 8 == 0 && raw_small()) hy4f_f16_bmm_place(q, c.qn16, d.wk_t, c.a16, W, NH, ND, KL, LW);
            else { bmm16(q, c.qn16, d.wk_t, c.qabs32, W, NH, ND, KL); place_cols(q, c.qabs32, c.a16, size_t(W) * NH, KL, LW); }
            // the keys: the target layer's latents at the positions its indexer chose for the last accepted row (every
            // position while the sequence is inside the recall budget), then the block's own
            const Dfly::Snap& sn = D.snap[li];
            if (!sn.T || row >= sn.T || row < sn.row0) return "the drafter has no recall choices of the target's last call (row " + std::to_string(row) + ")";
            uint32_t n_hist = 0;
            if (sn.masked) {
                n_hist = uint32_t(sn.cnt[row - sn.row0]);
                int32_t* ps = D.pin_sel + li * size_t(S);               // (a region per layer: nothing here waits for the card)
                std::memcpy(ps, sn.sel.data() + size_t(row - sn.row0) * S, size_t(n_hist) * 4);
                copy_words(q, D.sel_dev, ps, size_t(n_hist ? n_hist : 1) * 4, "pin_to_card");
                gather_keys(q, tgt.kv, D.sel_dev, D.keys, n_hist, LW);
            } else {
                n_hist = sn.pos0 + row + 1;
                if (n_hist > S) return "internal: a dense recall choice longer than the budget";
                copy_words(q, D.keys, tgt.kv, size_t(n_hist) * LW * 2, "card");
            }
            dump_hist.push_back(n_hist); dumpd("keys" + std::to_string(li) + ".f16", D.keys, size_t(n_hist) * LW * 2);
            mm16(q, c.xn16, d.kv_a, c.kv32, W, LW, H);
            kv_append(q, c.kv32, d.kv_a_norm, c.cs_i, D.keys + size_t(n_hist) * LW, nullptr, W, KL, RD, eps);
            mm(c, d.gate, c.xn16, c.gate32, W, NH * VD, H);
            // slot j sees the history and the block's slots 0 .. j. A short key list goes through the direct kernels; a long
            // one through oneDNN at a length rounded up (its primitives are per shape; the softmax's range hides the rest)
            const uint32_t n_keys = n_hist + W;
            const bool direct = n_keys <= 1024 && LW % 8 == 0 && raw_small();
            const uint32_t n_pad = direct ? n_keys : (n_keys + 63u) / 64u * 64u;
            if (uint64_t(W) * NH * n_pad > score_elems()) return "the drafter's block exceeds the attention workspace";
            int32_t* pr = D.pin_rng + li * size_t(2 * W);
            for (uint32_t j = 0; j < W; ++j) { pr[j] = 0; pr[size_t(W) + j] = int32_t(n_hist + j); }
            copy_words(q, c.lo, pr, size_t(W) * 4, "pin_to_card"); copy_words(q, c.hi, pr + W, size_t(W) * 4, "pin_to_card");
            if (direct) hy4f_f16_bmm_f32(q, c.a16, D.keys, c.scores, W * NH, 1, LW, n_keys);
            else gemm_nt_f16_onednn(q, c.a16, D.keys, c.scores, W * NH, n_pad, LW);
            sink_softmax(q, c.scores, c.p16, c.lo, c.hi, nullptr, d.sinks, W, NH, n_pad, scaling, true);
            if (direct) hy4f_f16_mix_f16(q, c.p16, D.keys, c.ctx16, W * NH, LW, n_keys);
            else gemm_fp16_onednn(q, c.p16, D.keys, c.ctx16, W * NH, LW, n_pad);
            if (LW % 8 == 0 && raw_small()) hy4f_f16_bmm_gate(q, c.ctx16, d.wv, c.gate32, c.attn16, W, NH, LW, VD);
            else { bmm16(q, c.ctx16, d.wv, c.v32, W, NH, LW, VD); gate_mul(q, c.v32, c.gate32, c.attn16, size_t(W) * NH * VD); }
            mm(c, d.o, c.attn16, c.o32, W, H, NH * VD);
            mimo26_rms_norm(q, c.o32, d.post_norm, nullptr, c.y32, W, H, eps);
            mimo26_axpy(q, c.y32, 1.f, D.x32, size_t(W) * H);
            // the dense SwiGLU
            mimo26_rms_norm(q, D.x32, D.ln2[li], c.xn16, nullptr, W, H, eps);
            mm(c, d.mlp_gate, c.xn16, D.mg32, W, D.FI, H); mm(c, d.mlp_up, c.xn16, D.mu32, W, D.FI, H);
            mimo26_swiglu_f32(q, D.mg32, D.mu32, D.mh16, size_t(W) * D.FI);
            mm(c, d.mlp_down, D.mh16, c.o32, W, H, D.FI);
            mimo26_axpy(q, c.o32, 1.f, D.x32, size_t(W) * H);
            dumpd("x" + std::to_string(li + 1) + ".f32", D.x32, size_t(W) * H * 4);
        }
        mimo26_rms_norm(q, D.x32, D.final_norm, nullptr, D.h32, W, H, eps);
        dumpd("h.f32", D.h32, size_t(W) * H * 4);
        // slot k's token from its hidden corrected by the token before it: a chain, each step through the target's head.
        // The confidence head (the acceptance rate of draft k given the drafts before it, as trained) ends the chain where
        // the drafts so far are unlikely to survive together: the head is 742 MB a read, and a draft nobody verifies is waste.
        const bool want_conf = (conf || stop_below > 0.f) && !D.conf_w.empty();
        int32_t prev = bonus; double cum = 1.0;
        float* pe = D.pin_e + size_t(W) * H;                           // (its own pinned row: the block's rows may still be on their way)
        for (uint32_t k = 1; k < W; ++k) {
            embed_row(prev, pe);
            copy_words(q, D.pe32, pe, size_t(H) * 4, "pin_to_card");
            mimo26_rms_norm(q, D.h32 + size_t(k) * H, D.hidden_norm, D.z16, nullptr, 1, H, eps);
            mimo26_rms_norm(q, D.pe32, D.embed_norm, D.z16 + H, nullptr, 1, H, eps);
            mm16(q, D.z16, D.gate_proj, D.g32, 1, H, 2 * H); mm16(q, D.z16, D.up_proj, D.u32, 1, H, 2 * H);
            mimo26_swiglu_f32(q, D.g32, D.u32, D.hm16, H);
            mm16(q, D.hm16, D.down_proj, D.d32, 1, H, H);
            { sycl::half* o = D.hc16; const float* hh = D.h32 + size_t(k) * H; const float* dd = D.d32; q.parallel_for(sycl::range<1>(H), [=](sycl::id<1> i) { o[i] = sycl::half(hh[i] + dd[i]); }); }
            if (want_conf) {                                           // sigmoid(w . corrected hidden + b)
                copy_words(q, D.pin_hc, D.hc16, size_t(H) * 2, "card_to_pin").wait();
                double a = D.conf_b; for (uint32_t i = 0; i < H; ++i) a += double(D.conf_w[i]) * double(float(D.pin_hc[i]));
                const double cf = 1.0 / (1.0 + std::exp(-a));
                if (conf) conf->push_back(float(cf));
                cum *= cf;
                if (stop_below > 0.f && cum < double(stop_below)) break;
            }
            gemm_nt_f16_onednn(q, D.hc16, c.head, c.logits, 1, V, H);
            copy_words(q, c.pin_logits, c.logits, size_t(V) * 4, "card_to_pin").wait();
            if (dump) { std::ofstream f(dpre + "logits" + std::to_string(k) + ".f32", std::ios::binary); f.write(reinterpret_cast<const char*>(c.pin_logits), std::streamsize(size_t(V) * 4)); }
            uint32_t best = 0;
            if (pick) best = uint32_t(pick(k - 1, c.pin_logits));
            else for (uint32_t v = 1; v < V; ++v) if (c.pin_logits[v] > c.pin_logits[best]) best = v;
            if (best >= V) return "the drafter's sampler returned a token out of range";
            drafts.push_back(int32_t(best)); prev = int32_t(best);
        }
        q.wait();                                                      // (the pinned tables are forward()'s too)
        if (dump) {
            std::ofstream f(dpre + "meta.json");
            f << "{\"bonus\": " << bonus << ", \"p0\": " << p0 << ", \"W\": " << W << ", \"n_hist\": [";
            for (size_t i2 = 0; i2 < dump_hist.size(); ++i2) f << (i2 ? ", " : "") << dump_hist[i2];
            f << "], \"drafts\": [";
            for (size_t i2 = 0; i2 < drafts.size(); ++i2) f << (i2 ? ", " : "") << drafts[i2];
            f << "]}\n"; ++dump_n;
        }
    } catch (const sycl::exception& e) {
        return std::string("sycl: ") + e.what();
    }
    return {};
}

// The rows' sliding rope table and key ranges on the card that computes the other half of a split layer.
void Hy4fForward::stage_helper(Card& p, uint32_t T, uint32_t pos0) {
    const auto& cfg = m_->config(); sycl::queue& q = *p.q;
    const size_t RDb = size_t(T) * cfg.rope_dim * 4;
    std::memcpy(p.pin_cos, h_cos_s_.data(), RDb);
    copy_words(q, p.cs_s, p.pin_cos, RDb, "pin_to_card");
    const uint32_t Wn = cfg.window, k0 = pos0 >= Wn - 1 ? pos0 - (Wn - 1) : 0;
    int32_t* rg = p.pin_rng;
    for (uint32_t t = 0; t < T; ++t) { const uint32_t pp = pos0 + t; rg[t] = int32_t((pp >= Wn - 1 ? pp - (Wn - 1) : 0) - k0); rg[size_t(T) + t] = int32_t(pp - k0); }
    copy_words(q, p.lo_s, rg, size_t(T) * 4, "pin_to_card"); copy_words(q, p.hi_s, rg + T, size_t(T) * 4, "pin_to_card").wait();
}

// A sliding layer on both cards: the home card's normed input goes to the other card, each computes its half of the heads
// through its half of o, the other card's partial output comes back and the two are summed before the output norm.
std::string Hy4fForward::attention_tp(Card& home, Card& help, const Dense& dh, const Dense& dp, const Hy4fLayer& lw, uint32_t L, uint32_t T, uint32_t pos0) {
    const auto& cfg = m_->config(); const uint32_t H = cfg.dim;
    sycl::queue& qh = *home.q; sycl::queue& qp = *help.q;
    // a decode row's hand-off is a few KiB: a kernel on the compute engine; a chunk's is megabytes: the copy engine
    // (through the small-copy kernels a 512-row chunk lost 5 % of its read to these two hand-offs a layer)
    auto mv = [](sycl::queue& q, void* dst, const void* src, size_t n, const char* tag) {
        return n > (size_t(256) << 10) ? KpQ{q, "memcpy"}.memcpy(dst, src, n) : copy_words(q, dst, src, n, tag);
    };
    mv(qh, home.pin_tpx, home.xn16, size_t(T) * H * 2, "card_to_pin").wait();
    std::memcpy(help.pin_tpx, home.pin_tpx, size_t(T) * H * 2);
    mv(qp, help.xn16, help.pin_tpx, size_t(T) * H * 2, "pin_to_card");
    if (T > kSmallRows && !prof_on_ && !g_profiler && !range_on()) {
        // a chunk: the other card's half from the worker thread, the home card's from this one
        if (!tpw_) tpw_ = std::make_unique<TpWorker>();
        tpw_->post([&]() -> std::string {
            if (auto e = attention_core(help, dp, lw, L, T, pos0); !e.empty()) return e;
            mv(qp, help.pin_tpo, help.o32, size_t(T) * H * 4, "card_to_pin").wait();
            return {};
        });
        std::string eh;
        try { eh = attention_core(home, dh, lw, L, T, pos0); } catch (const sycl::exception& e) { eh = std::string("sycl: ") + e.what(); }
        const std::string ep = tpw_->wait();                 // (always: the job holds references to this frame)
        if (!eh.empty()) return eh;
        if (!ep.empty()) return ep;
    } else {
        if (auto e = attention_core(help, dp, lw, L, T, pos0); !e.empty()) return e;
        sycl::event back = mv(qp, help.pin_tpo, help.o32, size_t(T) * H * 4, "card_to_pin");
        if (auto e = attention_core(home, dh, lw, L, T, pos0); !e.empty()) return e;
        back.wait();
    }
    std::memcpy(home.pin_tpo, help.pin_tpo, size_t(T) * H * 4);
    mv(qh, home.tp_o32, home.pin_tpo, size_t(T) * H * 4, "pin_to_card");
    if (tail_fused_) { tp_add_ = home.tp_o32; return {}; }    // (the caller's fused end adds the other half)
    {
        float* o = home.o32; const float* a = home.tp_o32;
        KpQ{qh, "Hy4fTpAddK"}.parallel_for<class Hy4fTpAddK>(sycl::range<1>(size_t(T) * H), [=](sycl::id<1> i) { o[i] += a[i]; });
    }
    mimo26_rms_norm(qh, home.o32, dh.post_norm, nullptr, home.y32, T, H, cfg.norm_eps);
    return {};
}

std::string Hy4fForward::attention_core(Card& c, const Dense& d, const Hy4fLayer& lw, uint32_t L, uint32_t T, uint32_t pos0) {
    const auto& cfg = m_->config(); sycl::queue& q = *c.q;
    const uint32_t H = cfg.dim, NH = d.nh, QL = cfg.q_lora_rank, KL = cfg.kv_lora_rank, RD = cfg.rope_dim, ND = cfg.nope_dim, VD = cfg.v_dim;
    const uint32_t QK = ND + RD, LW = KL + RD;
    const float eps = cfg.norm_eps;
    const float* cs = lw.indexed ? c.cs_i : c.cs_s;
    // the query: low rank, norm, heads; the content slice absorbed through W_k, the rope slice rotated
    mm16(q, c.xn16, d.q_a, c.qa32, T, QL, H);
    mimo26_rms_norm(q, c.qa32, d.q_a_norm, c.qr16, c.qr32, T, QL, eps);
    mm(c, d.q_b, c.qr16, c.q32, T, NH * QK, QL);
    tick(q, "a.q_a+norm+q_b");
    range32(q, "qa", L, c.qa32, size_t(T) * QL); range32(q, "q", L, c.q32, size_t(T) * NH * QK);
    q_split(q, c.q32, cs, c.qn16, c.a16, T, NH, ND, RD, KL, LW);
    const bool fuse_small = T <= kSmallRows && raw_small() && !range_on();      // (the range diagnostic reads the intermediates)
    if (fuse_small && ND % 8 == 0) hy4f_f16_bmm_place(q, c.qn16, d.wk_t, c.a16, T, NH, ND, KL, LW);
    else { bmm16(q, c.qn16, d.wk_t, c.qabs32, T, NH, ND, KL); place_cols(q, c.qabs32, c.a16, size_t(T) * NH, KL, LW); }
    tick(q, "a.q_split+absorb");
    range32(q, "qabs", L, c.qabs32, size_t(T) * NH * KL);
    // the keys: this chunk's latents into the cache
    mm16(q, c.xn16, d.kv_a, c.kv32, T, LW, H);
    uint32_t kb = 0;                                    // the position the card's key buffer starts at
    if (d.kv_win) {
        const uint32_t Wn = cfg.window, w0 = pos0 >= Wn - 1 ? pos0 - (Wn - 1) : 0;
        if (pos0 < d.kv_base || w0 < d.kv_base) return "internal: a sliding layer's key window does not cover position " + std::to_string(pos0);
        if (pos0 - d.kv_base + T > d.kv_cap) {             // make room: the rows the window still needs move to the front
            const uint32_t src = w0 - d.kv_base, n = pos0 - w0;
            if (src < n) return "internal: a sliding layer's key window cannot be moved in place";
            if (n) {
                sycl::half* base = d.kv; const size_t off = size_t(src) * LW;
                KpQ{q, "Hy4fKvMoveK"}.parallel_for<class Hy4fKvMoveK>(sycl::range<1>(size_t(n) * LW), [=](sycl::id<1> i) { base[i] = base[off + i]; });
            }
            d.kv_base = w0;
        }
        kb = d.kv_base;
    }
    kv_append(q, c.kv32, d.kv_a_norm, cs, d.kv + size_t(pos0 - kb) * LW, nullptr, T, KL, RD, eps);
    // (the copy for host RAM as one block of words: the append kernel writing it itself, two bytes at a time across the
    // bus, cost a 512-row chunk 10 % of its time)
    if (d.kv_win && d.kv_mirror) copy_words(q, c.pin_kv + size_t(d.kv_stage) * rows_max() * LW, d.kv + size_t(pos0 - kb) * LW, size_t(T) * LW * 2, "card_to_pin");
    tick(q, "a.kv_a+append");
    mm(c, d.gate, c.xn16, c.gate32, T, NH * VD, H);
    tick(q, "a.gemm_gate");
    range32(q, "kv", L, c.kv32, size_t(T) * LW); range32(q, "gate", L, c.gate32, size_t(T) * NH * VD);

    // which keys each row may see
    const uint32_t k1 = pos0 + T;
    uint32_t k0 = 0;
    bool masked = false, sel_on_card = false;     // sel_on_card: the recall wrote c.sel / c.lo / c.hi itself
    if (!lw.indexed) {
        const uint32_t Wn = cfg.window;
        k0 = pos0 >= Wn - 1 ? pos0 - (Wn - 1) : 0;
    } else {
        const auto ti = std::chrono::steady_clock::now();
        const uint32_t IXD = cfg.ix_head_dim, IXH = cfg.ix_heads;
        Hy4fIndexer& ix = indexers_[size_t(d.ix)];
        if (ix.n_tokens() + ix.pending != pos0) return "layer " + std::to_string(L) + ": the indexer holds " + std::to_string(ix.n_tokens() + ix.pending) + " token keys at position " + std::to_string(pos0);
        // The token keys stay on the card until a recall needs them (context past group_topk * ratio tokens) or the
        // buffer fills: within the budget every key is allowed, and reading 128 floats back cost a host round trip in
        // each of the 12 indexed layers of every token (2.7 ms a token, 2026-11-18).
        auto sync_keys = [&]() {
            if (!ix.pending) return;
            h_ixp_.resize(size_t(ix.pending) * IXD);
            KpQ{q, "memcpy"}.memcpy(h_ixp_.data(), d.ix_pend, h_ixp_.size() * 4).wait();
            const uint32_t n = ix.pending; ix.pending = 0;
            for (uint32_t r = 0; r < n; ++r) ix.append(h_ixp_.data() + size_t(r) * IXD, pos3_.data());
        };
        if (ix.pending + T > ix_pend_cap_) sync_keys();
        mm16(q, c.xn16, d.ix_tok_k, d.ix_pend + size_t(ix.pending) * IXD, T, IXD, H);
        ix.pending += T;
        for (uint32_t t = 0; t < T; ++t) { h_lo_[t] = 0; h_hi_[t] = int32_t(pos0 + t); }
        if ((k1 - 1) / cfg.ix_ratio + 1 > cfg.ix_group_topk) {     // some row is past the recall budget
          // A decode row's recall on the card (IE_HY4F_IX_GPU=0: on the host, the first form): the groups' scores, the exact
          // top group_topk - 1 and the key list go straight to the attention, and a group whose last token just arrived is
          // pooled on the card -- no read-back, no host selection (three host round trips a layer a token before).
          static const bool ix_gpu = [] { const char* v = std::getenv("IE_HY4F_IX_GPU"); return !(v && *v == '0'); }();
          if (ix_gpu && T == 1 && recall_gpu() && !(ref_recall_ && force_recall_) && !(dfly_ && dfly_->on) && d.ixg_kv) {
            uint32_t& gn = ix_dev_groups_[size_t(d.ix)];
            const uint32_t ng = ix.n_groups(), ratio = cfg.ix_ratio, G = cfg.ix_group_topk, S = G * ratio, p = pos0, own = p / ratio;
            if (gn < ng) {          // groups the host built that the card does not hold yet
                const size_t gb = size_t(ng - gn) * IXD * 4;
                if (gb <= kPinAllow) { std::memcpy(c.pin_allow, ix.grp_keys.data() + size_t(gn) * IXD, gb); copy_words(q, d.ix_grp + size_t(gn) * IXD, c.pin_allow, gb, "pin_to_card"); }
                else KpQ{q, "memcpy"}.memcpy(d.ix_grp + size_t(gn) * IXD, ix.grp_keys.data() + size_t(gn) * IXD, gb);
                gn = ng;
            }
            if (gn >= own) {
                // this token's normed key into the ring of its group; slots of the group the host holds (the group a prompt
                // ended inside) are copied in from the host's normed keys once
                hy4f_ix_norm_into(q, d.ix_pend + size_t(ix.pending - 1) * IXD, d.ixg_tok_norm, d.ixg_ring + size_t(p % ratio) * IXD, IXD, ix.eps);
                if (ix.n_tokens() > own * ratio && d.ixg_ring_seeded != own) {
                    const uint32_t n0 = own * ratio, n1 = std::min(ix.n_tokens(), p);
                    if (n1 > n0) {
                        std::memcpy(c.pin_gcs + IXD, ix.tok_keys.data() + size_t(n0) * IXD, size_t(n1 - n0) * IXD * 4);
                        copy_words(q, d.ixg_ring, c.pin_gcs + IXD, size_t(n1 - n0) * IXD * 4, "pin_to_card");
                    }
                    d.ixg_ring_seeded = own;
                }
                mm16(q, c.qr16, d.ix_q, c.ix_q32, T, IXH * IXD, QL);
                mm16(q, c.xn16, d.ix_head_w, c.ix_hw32, T, IXH, H);
                const uint32_t rd = ix.rope_dim;
                hy4f_rope_row(ix.theta, ix.mrope, rd, pos3_.data() + size_t(p) * 3, c.pin_ixcs, c.pin_ixcs + rd / 2);
                c.pin_ixown[0] = int32_t(own);
                copy_words(q, c.ix_cs, c.pin_ixcs, size_t(rd) * 4, "pin_to_card"); copy_words(q, c.ix_own, c.pin_ixown, 4, "pin_to_card");
                hy4f_ix_rotate(q, c.ix_q32, c.ix_cs, c.ix_qrot, T, IXH, IXD, rd);
                const float hs = 1.0f / std::sqrt(float(IXH)) / std::sqrt(float(IXD));
                hy4f_ix_scores(q, c.ix_qrot, c.ix_hw32, d.ix_grp, c.ix_own, c.ix_sc, 1, IXH, IXD, own, hs);
                hy4f_ix_select(q, c.ix_sc, own, G - 1, ratio, p, S, c.sel, c.lo, c.hi);
                if (ref_recall_ && p < ref_n_) {        // (the mini gate) the card's picks against the reference's
                    std::vector<int32_t> hs2(S); KpQ{q, "memcpy"}.memcpy(hs2.data(), c.sel, size_t(S) * 4).wait();
                    std::vector<uint32_t> mine; for (uint32_t i = 0; i < G - 1; ++i) mine.push_back(uint32_t(hs2[size_t(i) * ratio]) / ratio);
                    const int32_t* ref = ref_recall_ + (size_t(d.ix) * ref_n_ + p) * G;
                    std::vector<uint32_t> want(ref, ref + (G - 1)); std::sort(want.begin(), want.end());
                    ++ref_diffs_.recall_rows;
                    if (want != mine) ref_diffs_.recalls.push_back({L, p});
                }
                // this token completes its group: pooled on the card when all of the group's token keys are still there
                if ((p + 1) % ratio == 0 && gn == own) {
                    hy4f_rope_row(ix.theta, ix.mrope, rd, pos3_.data() + size_t(own) * ratio * 3, c.pin_gcs, c.pin_gcs + rd / 2);
                    copy_words(q, c.ix_gcs, c.pin_gcs, size_t(rd) * 4, "pin_to_card");
                    hy4f_ix_group_build(q, d.ixg_ring, d.ixg_kv, d.ixg_kg, d.ixg_key_norm, d.ixg_ape,
                                        c.ix_gcs, d.ix_grp + size_t(own) * IXD, ratio, IXD, rd, ix.eps);
                    gn = own + 1;
                }
                sel_on_card = true; masked = true;
            }
          }
          if (!sel_on_card && T == 1 && std::getenv("IE_HY4F_IX_DEBUG")) std::fprintf(stderr, "[ix] layer %u pos %u host recall: gn %u host groups %u own %u host tokens %u pending %u kv %d\n", L, pos0, ix_dev_groups_[size_t(d.ix)], ix.n_groups(), pos0 / cfg.ix_ratio, ix.n_tokens(), ix.pending, d.ixg_kv ? 1 : 0);
          if (!sel_on_card) {
            sync_keys();
            mm16(q, c.qr16, d.ix_q, c.ix_q32, T, IXH * IXD, QL);
            mm16(q, c.xn16, d.ix_head_w, c.ix_hw32, T, IXH, H);
            const uint32_t G = cfg.ix_group_topk, ratio = cfg.ix_ratio, S = G * ratio;
            std::vector<std::vector<uint32_t>> picked;
            // IE_HY4F_RECALL_THREADS (default 14; 1 = the first form, one row after the other on the calling thread)
            static const int rth = [] { const char* v = std::getenv("IE_HY4F_RECALL_THREADS"); const int n = v ? std::atoi(v) : 0; return n > 0 ? n : 14; }();
            if (recall_gpu()) {
                // The scores of every group against every row are computed on the card (a row costs context / ratio * heads
                // * head_dim products: on the host that was 6 ms a layer a token at 250,000 tokens of context, and it
                // grew with the square of a prompt's length); the host only picks each row's best groups.
                uint32_t& gn = ix_dev_groups_[size_t(d.ix)]; const uint32_t ng = ix.n_groups();
                if (gn > ng) gn = ng;
                if (gn < ng) {
                    const size_t gb = size_t(ng - gn) * IXD * 4;
                    if (gb <= kPinAllow) { std::memcpy(c.pin_allow, ix.grp_keys.data() + size_t(gn) * IXD, gb); copy_words(q, d.ix_grp + size_t(gn) * IXD, c.pin_allow, gb, "pin_to_card"); }
                    else KpQ{q, "memcpy"}.memcpy(d.ix_grp + size_t(gn) * IXD, ix.grp_keys.data() + size_t(gn) * IXD, gb);
                    gn = ng;
                }
                const uint32_t rd = ix.rope_dim;
                for (uint32_t t = 0; t < T; ++t) {
                    hy4f_rope_row(ix.theta, ix.mrope, rd, pos3_.data() + size_t(pos0 + t) * 3, c.pin_ixcs + size_t(t) * rd, c.pin_ixcs + size_t(t) * rd + rd / 2);
                    c.pin_ixown[t] = int32_t((pos0 + t) / ratio);
                }
                copy_words(q, c.ix_cs, c.pin_ixcs, size_t(T) * rd * 4, "pin_to_card"); copy_words(q, c.ix_own, c.pin_ixown, size_t(T) * 4, "pin_to_card");
                hy4f_ix_rotate(q, c.ix_q32, c.ix_cs, c.ix_qrot, T, IXH, IXD, rd);
                const uint32_t GC = std::max<uint32_t>(1, (pos0 + T - 1) / ratio);          // the last row's own group bounds every row's
                const float hs = 1.0f / std::sqrt(float(IXH)) / std::sqrt(float(IXD));
                const uint32_t rb = uint32_t(std::max<size_t>(1, std::min<size_t>(T, kIxScoreElems / GC)));
                if (size_t(GC) > kIxScoreElems) return "context too long for the indexer's score workspace";
                // IE_HY4F_PP_SELECT_GPU=0: the scores back to the host, which picks (the first form). On: a chunk's rows pick on the
                // card from the scores where they are, straight into the attention's key lists (exact: the host's rule)
                static const bool sel_gpu = [] { const char* v = std::getenv("IE_HY4F_PP_SELECT_GPU"); return !(v && *v == '0'); }();
                if (sel_gpu && T > kSmallRows && !(dfly_ && dfly_->on) && !ref_recall_) {
                    for (uint32_t r0 = 0; r0 < T; r0 += rb) {
                        const uint32_t rows = std::min(rb, T - r0);
                        hy4f_ix_scores(q, c.ix_qrot + size_t(r0) * IXH * IXD, c.ix_hw32 + size_t(r0) * IXH, d.ix_grp, c.ix_own + r0, c.ix_sc, rows, IXH, IXD, GC, hs);
                        hy4f_ix_select_rows(q, c.ix_sc, GC, rows, pos0 + r0, G - 1, ratio, S, c.sel + size_t(r0) * S, c.lo + r0, c.hi + r0);
                    }
                    sel_on_card = true; masked = true;
                } else {
                h_ixsc_.resize(size_t(T) * GC);
                for (uint32_t r0 = 0; r0 < T; r0 += rb) {
                    const uint32_t rows = std::min(rb, T - r0); const size_t sb = size_t(rows) * GC * 4;
                    hy4f_ix_scores(q, c.ix_qrot + size_t(r0) * IXH * IXD, c.ix_hw32 + size_t(r0) * IXH, d.ix_grp, c.ix_own + r0, c.ix_sc, rows, IXH, IXD, GC, hs);
                    if (sb <= kPinAllow) { copy_words(q, c.pin_allow, c.ix_sc, sb, "card_to_pin").wait(); std::memcpy(h_ixsc_.data() + size_t(r0) * GC, c.pin_allow, sb); }
                    else KpQ{q, "memcpy"}.memcpy(h_ixsc_.data() + size_t(r0) * GC, c.ix_sc, sb).wait();
                }
                ix.select_rows(h_ixsc_.data(), GC, pos0, T, picked, rth);
                }
            } else {
                copy_words(q, c.pin_ixq, c.ix_q32, size_t(T) * IXH * IXD * 4, "card_to_pin"); copy_words(q, c.pin_ixw, c.ix_hw32, size_t(T) * IXH * 4, "card_to_pin").wait();
                ix.recall_rows(c.pin_ixq, c.pin_ixw, pos0, T, nullptr, 0, &picked, rth, pos3_.data() + size_t(pos0) * 3);
            }
            // The keys each row attends, ascending (-1 pads the row): the recalled groups, then the row's own group up to
            // itself -- at most group_topk * ratio of them, wherever they lie in the context. (The first form scored EVERY
            // key of the context for every row and masked the rest: the cost of a token grew with the context.)
            if (!sel_on_card) {                                 // (the card picked: its key lists are written already)
            h_sel_.assign(size_t(T) * S, -1); h_cnt_.assign(T, 0);
            for (uint32_t t = 0; t < T; ++t) {
                const uint32_t p = pos0 + t, own = p / ratio; int32_t* row = h_sel_.data() + size_t(t) * S; uint32_t n = 0;
                if (own + 1 <= G) { for (uint32_t k = 0; k <= p; ++k) row[n++] = int32_t(k); }          // within the budget: every key
                else {
                    const int32_t* ref = ref_recall_ && p < ref_n_ ? ref_recall_ + (size_t(d.ix) * ref_n_ + p) * G : nullptr;   // G - 1 earlier groups, then the row's own
                    if (ref) {
                        std::vector<uint32_t> want(ref, ref + (G - 1)); std::sort(want.begin(), want.end());
                        ++ref_diffs_.recall_rows;
                        if (want != picked[t]) ref_diffs_.recalls.push_back({L, p});
                    }
                    if (ref && force_recall_) {
                        std::vector<uint32_t> ks;
                        for (uint32_t i = 0; i < G; ++i)
                            for (uint32_t s2 = 0; s2 < ratio; ++s2) { const uint64_t k = uint64_t(uint32_t(ref[i])) * ratio + s2; if (k <= p) ks.push_back(uint32_t(k)); }
                        std::sort(ks.begin(), ks.end()); ks.erase(std::unique(ks.begin(), ks.end()), ks.end());
                        for (uint32_t k : ks) if (n < S) row[n++] = int32_t(k);
                    } else {
                        for (uint32_t g : picked[t]) for (uint32_t s2 = 0; s2 < ratio; ++s2) row[n++] = int32_t(g * ratio + s2);
                        for (uint32_t k = own * ratio; k <= p; ++k) row[n++] = int32_t(k);
                    }
                }
                h_cnt_[t] = int32_t(n);
            }
            }
            masked = true;
          }
        }
        if (dfly_ && dfly_->on) {            // the keys each row of this call attended in a layer the drafter reads
            for (size_t i2 = 0; i2 < cfg.dfly_kv_layers.size(); ++i2) if (cfg.dfly_kv_layers[i2] == L) {
                auto& sn = dfly_->snap[i2]; sn.T = T; sn.pos0 = pos0; sn.masked = masked; sn.row0 = T <= kSmallRows ? 0 : T - 1;
                if (masked) {
                    const size_t S2 = size_t(cfg.ix_group_topk) * cfg.ix_ratio;
                    sn.sel.assign(h_sel_.begin() + long(size_t(sn.row0) * S2), h_sel_.begin() + long(size_t(T) * S2)); sn.cnt.assign(h_cnt_.begin() + sn.row0, h_cnt_.begin() + T);
                }
            }
        }
        st_.ms_indexer += ms_since(ti);
    }
    const float scaling = 1.0f / std::sqrt(float(QK));
    // IE_HY4F_FUSED_ATTN=0: the three-launch form (scores, softmax, context) for a few rows, for comparison
    static const bool fused_attn = [] { const char* v = std::getenv("IE_HY4F_FUSED_ATTN"); return !(v && *v == '0'); }();
    const bool fa = fused_attn && fuse_small && LW % 8 == 0 && LW <= kAttnMaxLW;
    // IE_HY4F_ATTN_XMX=0: the vector kernel above instead of the XMX one (the scores and the context as matrix tiles)
    static const bool attn_xmx = [] { const char* v = std::getenv("IE_HY4F_ATTN_XMX"); return !(v && *v == '0'); }();
    const bool fx = fa && attn_xmx && hy4f_attn_xmx_fits(NH, LW, KL);
    if (masked) {
        // each row against the keys it chose: gathered into one compact list, then the same three steps as a window
        const uint32_t S = cfg.ix_group_topk * cfg.ix_ratio;
        if (uint64_t(NH) * S > score_elems()) return "the recall budget exceeds the attention workspace";
        const size_t sb = size_t(T) * S * 4;
        if (!sel_on_card) {
            if (sb <= kPinAllow) { std::memcpy(c.pin_allow, h_sel_.data(), sb); copy_words(q, c.sel, c.pin_allow, sb, "pin_to_card"); }
            else KpQ{q, "memcpy"}.memcpy(c.sel, h_sel_.data(), sb);
            for (uint32_t t = 0; t < T; ++t) { c.pin_rng[t] = 0; c.pin_rng[size_t(T) + t] = h_cnt_[t] - 1; }
            copy_words(q, c.lo, c.pin_rng, size_t(T) * 4, "pin_to_card"); copy_words(q, c.hi, c.pin_rng + T, size_t(T) * 4, "pin_to_card");
        }
        if (fx) hy4f_attn_decode_xmx(q, c.a16, d.kv, c.sel, S, 0, c.lo, c.hi, d.sinks, c.ctx16, T, NH, scaling, plant() != "nosink");
        else if (fa && S <= kAttnMaxKeys) hy4f_attn_decode(q, c.a16, d.kv, c.sel, S, c.lo, c.hi, d.sinks, c.ctx16, T, NH, LW, scaling, plant() != "nosink");
        else if (pp_attn_xmx() && T > kSmallRows && hy4f_attn_xmx_fits(NH, LW, KL))
            hy4f_attn_rows_xmx(q, c.a16, d.kv, c.sel, S, c.lo, c.hi, d.sinks, c.ctx16, T, NH, scaling, plant() != "nosink");
        else if (LW % 8 == 0 && sel_kernels() && T > kSmallRows) {            // (a few rows: gathered keys through oneDNN are faster, 221 -> ~30 us a row)
            // every row of a block at once: its scores straight from the cache through its selection, the softmax, the sum
            const uint32_t blk = uint32_t(std::max<uint64_t>(1, std::min<uint64_t>(T, score_elems() / (uint64_t(NH) * S))));
            for (uint32_t r0 = 0; r0 < T; r0 += blk) {
                const uint32_t rows = std::min(blk, T - r0);
                hy4f_f16_scores_sel(q, c.a16 + size_t(r0) * NH * LW, d.kv, c.sel + size_t(r0) * S, c.scores, rows, NH, LW, S);
                sink_softmax(q, c.scores, c.p16, c.lo + r0, c.hi + r0, nullptr, d.sinks, rows, NH, S, scaling, plant() != "nosink");
                hy4f_f16_mix_sel(q, c.p16, d.kv, c.sel + size_t(r0) * S, c.hi + r0, c.ctx16 + size_t(r0) * NH * LW, rows, NH, LW, S);
            }
        } else
        for (uint32_t r = 0; r < T; ++r) {
            gather_keys(q, d.kv, c.sel + size_t(r) * S, c.kg16, S, LW);
            gemm_nt_f16_onednn(q, c.a16 + size_t(r) * NH * LW, c.kg16, c.scores, NH, S, LW);
            sink_softmax(q, c.scores, c.p16, c.lo + r, c.hi + r, nullptr, d.sinks, 1, NH, S, scaling, plant() != "nosink");
            gemm_fp16_onednn(q, c.p16, c.kg16, c.ctx16 + size_t(r) * NH * LW, NH, LW, S);
        }
    } else {
    const uint32_t n_keys = std::min(d.kv_cap - (k0 - kb), (k1 - k0 + 63u) / 64u * 64u);
    if (uint64_t(NH) * n_keys > score_elems()) return "context too long for the attention workspace (an indexed layer within its recall budget, or a window)";
    const int32_t* lo = lw.indexed ? c.lo_i : c.lo_s; const int32_t* hi = lw.indexed ? c.hi_i : c.hi_s;
    const uint32_t blk = uint32_t(std::max<uint64_t>(1, std::min<uint64_t>(T, score_elems() / (uint64_t(NH) * n_keys))));
    const sycl::half* keys = d.kv + size_t(k0 - kb) * LW;
    if (fx) hy4f_attn_decode_xmx(q, c.a16, keys, nullptr, 0, n_keys, lo, hi, d.sinks, c.ctx16, T, NH, scaling, plant() != "nosink");
    else if (fa && n_keys <= kAttnMaxKeys) hy4f_attn_decode(q, c.a16, keys, nullptr, 0, lo, hi, d.sinks, c.ctx16, T, NH, LW, scaling, plant() != "nosink");
    else if (pp_attn_xmx() && T > kSmallRows && hy4f_attn_xmx_fits(NH, LW, KL))      // prompt rows over contiguous keys (a window)
        hy4f_attn_rows_xmx(q, c.a16, keys, nullptr, 0, lo, hi, d.sinks, c.ctx16, T, NH, scaling, plant() != "nosink");
    else
    for (uint32_t r0 = 0; r0 < T; r0 += blk) {
        const uint32_t rows = std::min(blk, T - r0);
        // (a few rows against a window of keys: direct kernels; a long key list goes through oneDNN, which reads it once)
        const bool direct = rows <= kSmallRows && n_keys <= 1024 && LW % 8 == 0 && raw_small();
        if (direct) hy4f_f16_bmm_f32(q, c.a16 + size_t(r0) * NH * LW, keys, c.scores, rows * NH, 1, LW, n_keys);
        else gemm_nt_f16_onednn(q, c.a16 + size_t(r0) * NH * LW, keys, c.scores, rows * NH, n_keys, LW);
        sink_softmax(q, c.scores, c.p16, lo + r0, hi + r0, nullptr, d.sinks, rows, NH, n_keys, scaling, plant() != "nosink");
        if (direct) hy4f_f16_mix_f16(q, c.p16, keys, c.ctx16 + size_t(r0) * NH * LW, rows * NH, LW, n_keys);
        else gemm_fp16_onednn(q, c.p16, keys, c.ctx16 + size_t(r0) * NH * LW, rows * NH, LW, n_keys);
    }
    }
    // values through W_v, the gate, the output projection and its norm
    tick(q, "a.indexer+scores+softmax+ctx");
    if (fuse_small && LW % 8 == 0) hy4f_f16_bmm_gate(q, c.ctx16, d.wv, c.gate32, c.attn16, T, NH, LW, VD);
    else { bmm16(q, c.ctx16, d.wv, c.v32, T, NH, LW, VD); gate_mul(q, c.v32, c.gate32, c.attn16, size_t(T) * NH * VD); }
    range32(q, "v", L, c.v32, size_t(T) * NH * VD);
    tick(q, "a.values+gate");
    mm(c, d.o, c.attn16, c.o32, T, H, NH * VD);
    tick(q, "a.gemm_o");
    range32(q, "o", L, c.o32, size_t(T) * H);
    return {};
}

// The routed experts from the source's BF16 stacks: moe32 [T, latent] = sum over a row's picks of w * down(silu(min(gate, lim)) *
// clamp(up, +-lim)), gate | up = gate_up_proj[e] . lat. One expert at a time: its two matrices go up, the rows that picked it
// are gathered, three GEMMs, a weighted scatter.
std::string Hy4fForward::exact_moe(Card& c, const Hy4fLayer& lw, uint32_t T) {
    const auto& cfg = m_->config(); sycl::queue& q = *c.q;
    const uint32_t E = cfg.n_experts, TK = cfg.top_k, MI = cfg.moe_inter_dim, LAT = cfg.latent_dim;
    const float lim = cfg.swiglu_limit;
    if (!lw.exp_gate_up.w || !lw.exp_down.w) return "the full-precision experts are not in the source directory (this model directory keeps only the 8-bit pack)";
    if (lw.exp_gate_up.w->dtype_str != "BF16" || lw.exp_down.w->dtype_str != "BF16") return "the source's expert stacks are not BF16";
    std::vector<std::vector<int32_t>> rows(E); std::vector<std::vector<float>> wts(E);
    for (uint32_t t = 0; t < T; ++t)
        for (uint32_t k = 0; k < TK; ++k) { const int32_t e = h_ridx_[size_t(t) * TK + k]; rows[size_t(e)].push_back(int32_t(t)); wts[size_t(e)].push_back(h_rw_[size_t(t) * TK + k]); }
    uint32_t emu_block = 32; const int emu = emulate_code(opt_.emulate, emu_block);
    if (!opt_.emulate.empty() && (!emu || LAT % emu_block || MI % emu_block)) return "unknown weight format '" + opt_.emulate + "' (or widths that are not a multiple of its block)";
    uint32_t emu_hot_block = 32; const int emu_hot = emulate_code(opt_.emulate_hot, emu_hot_block);
    if (!opt_.emulate_hot.empty() && opt_.emulate_hot != "exact" && !emu_hot) return "unknown weight format '" + opt_.emulate_hot + "'";
    uint32_t layer_ord = 0;                                   // this layer's place among the MoE layers
    for (uint32_t l = 0; l < m_->layers().size() && &m_->layers()[l] != &lw; ++l) layer_ord += cfg.moe[l];
    float* moe = c.moe32; const float* lat = c.lat32;
    KpQ{q, "fill"}.fill(moe, 0.f, size_t(T) * LAT);
    for (uint32_t e = 0; e < E; ++e) {
        const uint32_t n = uint32_t(rows[e].size());
        if (!n) continue;
        const size_t gu_n = size_t(2) * MI * LAT, dn_n = size_t(LAT) * MI;
        KpQ{q, "memcpy"}.memcpy(c.ex_gu, lw.exp_gate_up.w->data + size_t(e) * gu_n * 2, gu_n * 2); bf16_to_f16_inplace(q, c.ex_gu, gu_n);
        KpQ{q, "memcpy"}.memcpy(c.ex_dn, lw.exp_down.w->data + size_t(e) * dn_n * 2, dn_n * 2); bf16_to_f16_inplace(q, c.ex_dn, dn_n);
        {   // the expert's format: the hot set's (when one is given and it is in it), else the default
            const bool is_hot = !opt_.emulate_hot_set.empty() && layer_ord < opt_.emulate_hot_set.size() && e < opt_.emulate_hot_set[layer_ord].size() && opt_.emulate_hot_set[layer_ord][e];
            const int f = is_hot ? emu_hot : emu; const uint32_t fb = is_hot ? emu_hot_block : emu_block;
            if (f) { emulate_format(q, c.ex_gu, gu_n / fb, fb, f); emulate_format(q, c.ex_dn, dn_n / fb, fb, f); }
        }
        KpQ{q, "memcpy"}.memcpy(c.ex_rows, rows[e].data(), size_t(n) * 4); KpQ{q, "memcpy"}.memcpy(c.ex_w, wts[e].data(), size_t(n) * 4);
        const int32_t* ri = c.ex_rows; const float* rw = c.ex_w; sycl::half* x16 = c.ex_x16; sycl::half* h16 = c.ex_h16; const float* gu = c.ex_gu32; const float* d32 = c.ex_d32;
        KpQ{q, "Hy4fExactGatherK"}.parallel_for<class Hy4fExactGatherK>(sycl::range<2>(n, LAT), [=](sycl::id<2> i) { x16[i[0] * LAT + i[1]] = sycl::half(lat[size_t(ri[i[0]]) * LAT + i[1]]); });
        gemm_nt_f16_onednn(q, c.ex_x16, c.ex_gu, c.ex_gu32, n, 2 * MI, LAT);
        KpQ{q, "Hy4fExactSwigluK"}.parallel_for<class Hy4fExactSwigluK>(sycl::range<2>(n, MI), [=](sycl::id<2> i) {
            const float g = sycl::fmin(gu[i[0] * 2 * MI + i[1]], lim);
            const float u = sycl::fmin(sycl::fmax(gu[i[0] * 2 * MI + MI + i[1]], -lim), lim);
            h16[i[0] * MI + i[1]] = sycl::half(g * dsig(g) * u);
        });
        if (range_on() && e < 2) { q.wait(); range32(q, "ex_gu", layer_ord, c.ex_gu32, size_t(n) * 2 * MI); }
        gemm_nt_f16_onednn(q, c.ex_h16, c.ex_dn, c.ex_d32, n, LAT, MI);
        KpQ{q, "Hy4fExactScatterK"}.parallel_for<class Hy4fExactScatterK>(sycl::range<2>(n, LAT), [=](sycl::id<2> i) { moe[size_t(ri[i[0]]) * LAT + i[1]] += rw[i[0]] * d32[i[0] * LAT + i[1]]; }).wait();
    }
    return {};
}

// The routed experts from the 8-bit store: per expert picked in this call, its rows gathered (fp16), gate and up in one
// W8A16 launch, the clamped SwiGLU, down, and a weighted scatter into moe32 [T, latent].
std::string Hy4fForward::q8_moe(Card& c, uint32_t L, uint32_t T) {
    const auto& cfg = m_->config(); sycl::queue& q = *c.q;
    const uint32_t E = cfg.n_experts, TK = cfg.top_k, MI = cfg.moe_inter_dim, LAT = cfg.latent_dim;
    const float lim = cfg.swiglu_limit;
    std::vector<std::vector<int32_t>> rows(E); std::vector<std::vector<float>> wts(E);
    for (uint32_t t = 0; t < T; ++t)
        for (uint32_t k = 0; k < TK; ++k) { const int32_t e = h_ridx_[size_t(t) * TK + k]; rows[size_t(e)].push_back(int32_t(t)); wts[size_t(e)].push_back(h_rw_[size_t(t) * TK + k]); }
    // one token (decode): its 16 experts in two launches (IE_HY4F_Q8_FUSED=0 keeps the per-expert chain, for comparison)
    static const bool fused = [] { const char* v = std::getenv("IE_HY4F_Q8_FUSED"); return !(v && *v == '0'); }();
    // IE_HY4F_Q8_CPU=0 turns the CPU leg off (every miss is copied to the card, the bring-up behaviour); IE_HY4F_CPU_THREADS
    static const bool cpu_leg = [] { const char* v = std::getenv("IE_HY4F_Q8_CPU"); return !(v && *v == '0'); }();
    static const int cpu_threads = [] { const char* v = std::getenv("IE_HY4F_CPU_THREADS"); const int n = v ? std::atoi(v) : 0; return n > 0 ? n : 0; }();   // 0 = the leg's default
    if (T == 1 && fused && TK <= 16 && cpu_leg) {
        // hits on the card (two launches), misses on the CPU from RAM, in parallel; an expert that keeps missing earns a slot
        // IE_HY4F_PROMOTE_PER_TOKEN: uploads a token may spend on promotions (default 4, about 3 ms; 0 = fixed residency)
        static const uint32_t promo_budget = [] { const char* v = std::getenv("IE_HY4F_PROMOTE_PER_TOKEN"); return v ? uint32_t(std::atoi(v)) : 4u; }();
        if (c.q8_promo_tok != st_.rows) { c.q8_promo_tok = st_.rows; c.q8_promo_left = promo_budget; }
        Hy4fQ8Store::Planes pg[16], pc[16]; float wg[16], wc[16]; uint32_t ng = 0, nc = 0, missk[16], nm = 0;
        const uint64_t mark = c.q8.mark();
        for (uint32_t k = 0; k < TK; ++k) {                      // the residents first: their slots are then safe from eviction
            const uint32_t e = uint32_t(h_ridx_[k]);
            c.q8.note_pick(L, e, st_.rows);
            if (c.q8.resident(L, e)) { if (auto er = c.q8.acquire(q, L, e, pg[ng]); !er.empty()) return er; wg[ng++] = h_rw_[k]; }
            else missk[nm++] = k;
        }
        // (IE_HY4F_ASYNC_PROMOTE=0: the upload on this queue and the expert on the card at once -- the card waits for the copy)
        static const bool async_promote = [] { const char* v = std::getenv("IE_HY4F_ASYNC_PROMOTE"); return !(v && *v == '0'); }();
        for (uint32_t i = 0; i < nm; ++i) {
            const uint32_t k = missk[i], e = uint32_t(h_ridx_[k]);
            if (c.q8_promo_left) {
                if (async_promote) { if (c.q8.promote_async(q, L, e, st_.rows, mark)) --c.q8_promo_left; }   // copied beside the compute; still a CPU miss this time
                else if (c.q8.promote(q, L, e, st_.rows, mark, pg[ng])) { wg[ng++] = h_rw_[k]; --c.q8_promo_left; continue; }
            }
            pc[nc] = c.q8.ram_planes(L, e); wc[nc++] = h_rw_[k];
        }
        if (ng) { cast_fp32_to_fp16(q, c.lat32, c.ex_x16, LAT); hy4f_q8_moe_decode(q, c.ex_x16, pg, wg, ng, c.q8_hj, c.moe32, LAT, MI, lim, c.q8.fmt()); }
        else KpQ{q, "fill"}.fill(c.moe32, 0.f, size_t(LAT));
        if (nc) {
            std::fill_n(c.pin_ycpu, LAT, 0.f);
            const auto tc = std::chrono::steady_clock::now();
            // IE_HY4F_DIAG_SKIP_CPU=1 (a timing diagnostic, WRONG output): the missed experts contribute nothing
            static const bool skip_cpu = [] { const char* v = std::getenv("IE_HY4F_DIAG_SKIP_CPU"); return v && *v == '1'; }();
            if (!skip_cpu) hy4f_q8_moe_cpu(c.pin_lat, pc, wc, nc, c.pin_ycpu, LAT, MI, lim, cpu_threads, c.q8.fmt());
            st_.q8_ms_cpu += ms_since(tc);
            copy_words(q, c.q8_ycpu, c.pin_ycpu, size_t(LAT) * 4, "pin_to_card");            // (the layer's closing wait runs before the host writes pin_ycpu again)
            mimo26_axpy(q, c.q8_ycpu, 1.f, c.moe32, size_t(LAT));
        }
        st_.q8_cpu += nc;
        st_.q8_hits = 0; st_.q8_uploads = 0; st_.q8_promotions = 0;
        for (const auto& cc : cards_) if (cc->q8_on) { st_.q8_hits += cc->q8.stats().hits; st_.q8_uploads += cc->q8.stats().uploads; st_.q8_promotions += cc->q8.stats().promotions; }
        return {};
    }
    if (T == 1 && fused && TK <= 16) {
        Hy4fQ8Store::Planes pl[16]; float wj[16];
        for (uint32_t k = 0; k < TK; ++k) {
            if (auto er = c.q8.acquire(q, L, uint32_t(h_ridx_[k]), pl[k]); !er.empty()) return er;
            wj[k] = h_rw_[k];
        }
        // IE_HY4F_PROF=1 (diagnostic): wait after the uploads and after the kernels so each has its own time
        static const bool prof = [] { const char* v = std::getenv("IE_HY4F_PROF"); return v && *v == '1'; }();
        auto tp = std::chrono::steady_clock::now();
        if (prof) { q.wait(); st_.q8_ms_upload += ms_since(tp); tp = std::chrono::steady_clock::now(); }
        cast_fp32_to_fp16(q, c.lat32, c.ex_x16, LAT);
        hy4f_q8_moe_decode(q, c.ex_x16, pl, wj, TK, c.q8_hj, c.moe32, LAT, MI, lim, c.q8.fmt());
        if (prof) { q.wait(); st_.q8_ms_kernel += ms_since(tp); }
        st_.q8_hits = 0; st_.q8_uploads = 0; st_.q8_promotions = 0;
        for (const auto& cc : cards_) if (cc->q8_on) { st_.q8_hits += cc->q8.stats().hits; st_.q8_uploads += cc->q8.stats().uploads; st_.q8_promotions += cc->q8.stats().promotions; }
        return {};
    }
    if (T > 1 && T <= kSmallRows && fused && cpu_leg && TK <= 16 && !cov_on_ && !cov_x_) {
        // a decode pass of a few rows: every resident expert any row picked is one job on the card (read once), the rest go
        // to the CPU grouped by expert (read once each); nothing is swapped in or out
        int32_t* js = c.pin_mr; uint32_t* jm = reinterpret_cast<uint32_t*>(c.pin_mr + kSmallJobs);
        int32_t* rj = c.pin_mr + 2 * kSmallJobs; float* rw = reinterpret_cast<float*>(c.pin_mr + 2 * kSmallJobs + size_t(kSmallRows) * TK);
        uint32_t nj = 0;
        std::vector<std::pair<uint32_t, uint32_t>> miss;                // (expert, row * TK + k)
        for (uint32_t t = 0; t < T; ++t)
            for (uint32_t k = 0; k < TK; ++k) {
                const uint32_t e = uint32_t(h_ridx_[size_t(t) * TK + k]);
                c.q8.note_pick(L, e, st_.rows + t);
                const int32_t sl = c.q8.slot_touch(L, e);
                rw[size_t(t) * TK + k] = h_rw_[size_t(t) * TK + k];
                if (sl < 0) { rj[size_t(t) * TK + k] = -1; miss.emplace_back(e, t * TK + k); continue; }
                uint32_t j = 0;
                while (j < nj && js[j] != sl) ++j;
                if (j == nj) { js[nj] = sl; jm[nj] = 0; ++nj; }
                jm[j] |= 1u << t; rj[size_t(t) * TK + k] = int32_t(j);
            }
        std::sort(miss.begin(), miss.end());
        const uint32_t nc = uint32_t(miss.size());
        c.h_cpl.resize(nc); c.h_crow.resize(nc); c.h_cw.resize(nc);
        for (uint32_t i = 0; i < nc; ++i) { c.h_cpl[i] = c.q8.ram_planes(L, miss[i].first); c.h_crow[i] = int32_t(miss[i].second / TK); c.h_cw[i] = h_rw_[miss[i].second]; }
        copy_words(q, c.mr_slot, js, size_t(kSmallJobs) * 4, "pin_to_card"); copy_words(q, c.mr_mask, jm, size_t(kSmallJobs) * 4, "pin_to_card");
        copy_words(q, c.mr_rowjobs, rj, size_t(kSmallRows) * TK * 4, "pin_to_card"); copy_words(q, c.mr_roww, rw, size_t(kSmallRows) * TK * 4, "pin_to_card");
        cast_fp32_to_fp16(q, c.lat32, c.ex_x16, size_t(T) * LAT);
        hy4f_q8_moe_mrows(q, c.ex_x16, c.q8.slab(L), c.q8.slot_bytes(), c.q8.offsets(), c.mr_slot, c.mr_mask, c.mr_rowjobs, c.mr_roww, nj, T, TK,
                          c.mr_hj, c.mr_yj, c.moe32, LAT, MI, lim, c.q8.fmt());
        if (nc) {
            std::fill_n(c.pin_ycpu, size_t(T) * LAT, 0.f);
            const auto tc = std::chrono::steady_clock::now();
            hy4f_q8_moe_cpu_rows(c.pin_lat, c.h_cpl.data(), c.h_crow.data(), c.h_cw.data(), nc, T, c.pin_ycpu, LAT, MI, lim, cpu_threads, c.q8.fmt(), nullptr);
            st_.q8_ms_cpu += ms_since(tc);
            copy_words(q, c.q8_ycpu, c.pin_ycpu, size_t(T) * LAT * 4, "pin_to_card");
            mimo26_axpy(q, c.q8_ycpu, 1.f, c.moe32, size_t(T) * LAT);
        }
        q.wait();                    // the job tables above are pinned memory of this call
        st_.q8_cpu += nc;
        st_.q8_hits = 0; st_.q8_uploads = 0; st_.q8_promotions = 0;
        for (const auto& cc : cards_) if (cc->q8_on) { st_.q8_hits += cc->q8.stats().hits; st_.q8_uploads += cc->q8.stats().uploads; st_.q8_promotions += cc->q8.stats().promotions; }
        return {};
    }
    if (T > 1 && fused && cpu_leg) {
        // a chunk: its (row, pick) jobs in row order -- the residents on the card in three launches, the rest on the CPU
        // from RAM in parallel. (No promotion here: a chunk touches most of a layer's experts once or twice.)
        const size_t P = size_t(T) * TK;
        c.h_jslot.resize(P); c.h_jrow.resize(P); c.h_jw.resize(P); c.h_rs.assign(size_t(T) + 1, 0);
        std::vector<int32_t> jpick(P);
        std::vector<std::pair<uint32_t, uint32_t>> miss;                // (expert, row * TK + k)
        uint32_t ng = 0;
        for (uint32_t t = 0; t < T; ++t) {
            for (uint32_t k = 0; k < TK; ++k) {
                const uint32_t e = uint32_t(h_ridx_[size_t(t) * TK + k]);
                c.q8.note_pick(L, e, st_.rows + t);
                const int32_t s = cov_on_ ? -1 : c.q8.slot_touch(L, e);          // (recording: every job on the CPU, where its hidden activation is at hand)
                if (s >= 0) { c.h_jslot[ng] = s; c.h_jrow[ng] = int32_t(t); c.h_jw[ng] = h_rw_[size_t(t) * TK + k]; jpick[ng] = int32_t(t * TK + k); ++ng; }
                else miss.emplace_back(e, t * TK + k);
            }
            c.h_rs[t + 1] = int32_t(ng);
        }
        std::sort(miss.begin(), miss.end());
        const uint32_t nc = uint32_t(miss.size());
        c.h_cpl.resize(nc); c.h_crow.resize(nc); c.h_cw.resize(nc);
        for (uint32_t i = 0; i < nc; ++i) { c.h_cpl[i] = c.q8.ram_planes(L, miss[i].first); c.h_crow[i] = int32_t(miss[i].second / TK); c.h_cw[i] = h_rw_[miss[i].second]; }
        const bool grp = pp_group() && c.q8.fmt().gu == 8 && c.q8.fmt().dn == 8 && T >= 32;
        bool streamed = false;                                       // the misses ran on the card (no CPU leg, no rebalance below)
        uint32_t nc_cpu = 0; std::thread hyb;                        // streamed: the lightest misses on the CPU beside the stream
        if (grp) {
            // expert-major: counting sort of the resident picks by slot (rows ascending within a slot), jobs of <= 32
            uint32_t nslot = 0; for (uint32_t i = 0; i < ng; ++i) nslot = std::max(nslot, uint32_t(c.h_jslot[i]) + 1);
            c.h_gcnt.assign(size_t(nslot) + 1, 0);
            for (uint32_t i = 0; i < ng; ++i) ++c.h_gcnt[size_t(c.h_jslot[i]) + 1];
            for (uint32_t sl = 0; sl < nslot; ++sl) c.h_gcnt[sl + 1] += c.h_gcnt[sl];
            std::vector<int32_t> at(c.h_gcnt.begin(), c.h_gcnt.end() - 1);
            std::vector<int32_t> grow(ng); std::vector<float> gw(ng);
            c.h_gpos.assign(P, -1);
            for (uint32_t i = 0; i < ng; ++i) {             // (in row order, so a slot's picks come out rows ascending)
                const int32_t j = at[size_t(c.h_jslot[i])]++;
                grow[j] = c.h_jrow[i]; gw[j] = c.h_jw[i]; c.h_gpos[size_t(jpick[i])] = j;
            }
            c.h_gbase.clear(); c.h_gj0.clear(); c.h_gjn.clear();
            const uint64_t slab = reinterpret_cast<uint64_t>(c.q8.slab(L)), sb = c.q8.slot_bytes();
            for (uint32_t sl = 0; sl < nslot; ++sl)
                for (int32_t j = c.h_gcnt[sl]; j < c.h_gcnt[sl + 1]; j += 32) {
                    c.h_gbase.push_back(slab + uint64_t(sl) * sb); c.h_gj0.push_back(j); c.h_gjn.push_back(std::min<int32_t>(32, c.h_gcnt[sl + 1] - j));
                }
            const uint32_t nj_res = uint32_t(c.h_gbase.size());
            // the missed experts streamed: their picks after the residents' in j order (expert, then rows ascending)
            // (IE_HY4F_STREAM_MIN_ROWS, default 2048: a chunk streams an expert for all its rows, so a short chunk pays the
            // whole expert for few rows -- at 512 rows the CPU leg is as fast; serving reads 512-row chunks today)
            static const uint32_t stream_min = [] { const char* v = std::getenv("IE_HY4F_STREAM_MIN_ROWS"); return v ? uint32_t(std::atoi(v)) : 2048u; }();
            streamed = pp_stream() && c.stg && nc && !cov_on_ && T >= stream_min;
            std::vector<uint32_t> mex, mbeg, gjob0; std::vector<int32_t> promo_slot;
            std::vector<std::pair<uint32_t, int32_t>> promo;
            if (streamed) {
                std::vector<uint32_t> amex, abeg;                 // every missed expert and its picks' range in miss
                for (uint32_t i = 0; i < nc; ++i) if (i == 0 || miss[i].first != miss[i - 1].first) { amex.push_back(miss[i].first); abeg.push_back(i); }
                abeg.push_back(nc);
                // The CPU is idle while the experts stream: it takes the missed experts with the FEWEST picks (its cost grows
                // with the picks, a streamed expert costs its whole bytes however few rows picked it), as many as it can
                // finish while the rest stream. IE_HY4F_HYB_US_PICK (us a pick on the CPU, default 15) and
                // IE_HY4F_HYB_MS_EXPERT (ms an expert on the link, default 0.36); IE_HY4F_HYB_US_PICK=0: everything streams.
                static const double us_pick = [] { const char* v = std::getenv("IE_HY4F_HYB_US_PICK"); return v ? std::atof(v) : 15.0; }();
                static const double ms_exp = [] { const char* v = std::getenv("IE_HY4F_HYB_MS_EXPERT"); return v ? std::atof(v) : 0.36; }();
                std::vector<uint8_t> on_cpu(amex.size(), 0);
                if (us_pick > 0) {
                    std::vector<uint32_t> ord(amex.size()); for (uint32_t i = 0; i < ord.size(); ++i) ord[i] = i;
                    std::stable_sort(ord.begin(), ord.end(), [&](uint32_t x, uint32_t y) { return abeg[x + 1] - abeg[x] < abeg[y + 1] - abeg[y]; });
                    double cpu_ms = 0;
                    for (size_t k = 0; k < ord.size(); ++k) {
                        const double add = double(abeg[ord[k] + 1] - abeg[ord[k]]) * us_pick / 1000.0;
                        if (cpu_ms + add > double(ord.size() - k - 1) * ms_exp) break;
                        cpu_ms += add; on_cpu[ord[k]] = 1;
                    }
                }
                c.h_cpl.clear(); c.h_crow.clear(); c.h_cw.clear();
                uint32_t ns = 0;                                  // streamed picks so far (their j = ng + ns)
                for (uint32_t i = 0; i < amex.size(); ++i) {
                    if (on_cpu[i]) {
                        for (uint32_t m = abeg[i]; m < abeg[i + 1]; ++m) { c.h_cpl.push_back(c.q8.ram_planes(L, amex[i])); c.h_crow.push_back(int32_t(miss[m].second / TK)); c.h_cw.push_back(h_rw_[miss[m].second]); }
                        continue;
                    }
                    mex.push_back(amex[i]); mbeg.push_back(ns);
                    for (uint32_t m = abeg[i]; m < abeg[i + 1]; ++m) {
                        const uint32_t p = miss[m].second, j = ng + ns++;
                        if (grow.size() <= j) { grow.resize(j + 1); gw.resize(j + 1); }
                        grow[j] = int32_t(p / TK); gw[j] = h_rw_[p]; c.h_gpos[p] = int32_t(j);
                    }
                }
                mbeg.push_back(ns);
                nc_cpu = uint32_t(c.h_cpl.size());
                static const uint32_t rb_budget = [] { const char* v = std::getenv("IE_HY4F_REBALANCE"); return v ? uint32_t(std::atoi(v)) : 32u; }();
                promo = c.q8.plan_promotions(L, st_.rows + T, rb_budget);       // (bookkeeping now; the bytes from the staging copies below)
                promo_slot.assign(cfg.n_experts, -1);
                for (const auto& [e, sl] : promo) promo_slot[e] = sl;
                const uint64_t stg = reinterpret_cast<uint64_t>(c.stg), sb = c.q8.slot_bytes();
                for (uint32_t g0 = 0; g0 < mex.size(); g0 += kStageSlots) {
                    gjob0.push_back(uint32_t(c.h_gbase.size()));
                    const uint32_t b = (g0 / kStageSlots) % 2;
                    for (uint32_t i = g0; i < std::min<uint32_t>(uint32_t(mex.size()), g0 + kStageSlots); ++i)
                        for (uint32_t j = ng + mbeg[i]; j < ng + mbeg[i + 1]; j += 32) {
                            c.h_gbase.push_back(stg + (uint64_t(b) * kStageSlots + (i - g0)) * sb); c.h_gj0.push_back(int32_t(j)); c.h_gjn.push_back(int32_t(std::min<uint32_t>(32, ng + mbeg[i + 1] - j)));
                        }
                }
                gjob0.push_back(uint32_t(c.h_gbase.size()));
            }
            const uint32_t nj = uint32_t(c.h_gbase.size()), njp = uint32_t(grow.size());
            if (nj > c.grp_jcap) return "a chunk's expert jobs exceed the table";
            if (njp) { KpQ{q, "memcpy"}.memcpy(c.q8_jrow, grow.data(), size_t(njp) * 4); KpQ{q, "memcpy"}.memcpy(c.q8_jw, gw.data(), size_t(njp) * 4); }
            KpQ{q, "memcpy"}.memcpy(c.q8_jslot, c.h_gpos.data(), P * 4);
            if (nj) { KpQ{q, "memcpy"}.memcpy(c.grp_jbase, c.h_gbase.data(), size_t(nj) * 8); KpQ{q, "memcpy"}.memcpy(c.grp_j0, c.h_gj0.data(), size_t(nj) * 4); KpQ{q, "memcpy"}.memcpy(c.grp_jn, c.h_gjn.data(), size_t(nj) * 4); }
            cast_fp32_to_fp16(q, c.lat32, c.ex_x16, size_t(T) * LAT);
            auto run_jobs = [&](uint32_t a, uint32_t b) {
                if (b <= a) return;
                hy4f_moe_grp_gu(q, c.ex_x16, c.grp_jbase + a, c.grp_j0 + a, c.grp_jn + a, c.q8_jrow, b - a, c.q8.offsets(), c.q8_hrows, LAT, MI, lim);
                hy4f_moe_grp_down(q, c.q8_hrows, c.grp_jbase + a, c.grp_j0 + a, c.grp_jn + a, c.q8_jw, b - a, c.q8.offsets(), c.q8_yrows, LAT, MI);
            };
            run_jobs(0, nj_res);                                   // (before any promotion overwrites a resident slot)
            if (streamed && nc_cpu) hyb = std::thread([&, nc_cpu] {
                std::fill_n(c.pin_ycpu, size_t(T) * LAT, 0.f);
                hy4f_q8_moe_cpu_rows(c.pin_lat, c.h_cpl.data(), c.h_crow.data(), c.h_cw.data(), nc_cpu, T, c.pin_ycpu, LAT, MI, lim, cpu_threads, c.q8.fmt(), nullptr);
            });
            if (streamed) {
                const uint64_t sb = c.q8.slot_bytes(); uint8_t* slab = const_cast<uint8_t*>(c.q8.slab(L));
                sycl::event dma[2], comp[2]; bool hd[2] = {false, false}, hc[2] = {false, false};
                const auto ts = std::chrono::steady_clock::now();
                for (uint32_t g = 0; g * kStageSlots < mex.size(); ++g) {
                    const uint32_t b = g % 2, g0 = g * kStageSlots, ne = std::min<uint32_t>(uint32_t(mex.size()) - g0, kStageSlots);
                    uint8_t* st = c.stg + uint64_t(b) * kStageSlots * sb;
                    std::vector<sycl::event> deps; if (hc[b]) deps.push_back(comp[b]);
                    if (c.q8.ram_pinned(L)) {                      // the pack is pinned: the card reads each expert from it directly
                        sycl::event last;
                        for (uint32_t i = 0; i < ne; ++i) last = c.qc->memcpy(st + uint64_t(i) * sb, c.q8.ram_slot(L, mex[g0 + i]), sb, i == 0 ? deps : std::vector<sycl::event>{});
                        dma[b] = last; hd[b] = true;
                    } else {
                    if (hd[b]) dma[b].wait();                      // the pinned buffer is free again
                    uint8_t* pin = c.pin_stg[b];
                    const uint32_t units = ne * 4; const uint64_t ub = (sb + 3) / 4;
                    copy_pool().run([&](uint32_t tid) {
                        for (uint32_t u = tid; u < units; u += copy_pool().n) {
                            const uint32_t i = u / 4; const uint64_t o = uint64_t(u % 4) * ub, len = std::min<uint64_t>(ub, sb - std::min<uint64_t>(sb, o));
                            if (len) std::memcpy(pin + uint64_t(i) * sb + o, reinterpret_cast<const uint8_t*>(c.q8.ram_planes(L, mex[g0 + i]).gate_qs) + o, len);
                        }
                    });
                    dma[b] = c.qc->memcpy(st, pin, uint64_t(ne) * sb, deps); hd[b] = true;
                    }
                    q.ext_oneapi_submit_barrier({dma[b]});
                    for (uint32_t i = 0; i < ne; ++i) if (promo_slot[mex[g0 + i]] >= 0) { q.memcpy(slab + uint64_t(promo_slot[mex[g0 + i]]) * sb, st + uint64_t(i) * sb, sb); promo_slot[mex[g0 + i]] = -2; }
                    run_jobs(gjob0[g], gjob0[g + 1]);
                    comp[b] = q.ext_oneapi_submit_barrier(); hc[b] = true;
                }
                for (const auto& [e, sl] : promo) if (promo_slot[e] >= 0) q.memcpy(slab + uint64_t(sl) * sb, c.q8.ram_planes(L, e).gate_qs, sb);   // (not picked in this chunk)
                st_.q8_ms_cpu += ms_since(ts); st_.q8_rebalanced += uint32_t(promo.size());
            }
            hy4f_moe_grp_sum(q, c.q8_yrows, c.q8_jslot, T, TK, c.moe32, LAT);
            if (hyb.joinable()) {
                hyb.join();
                copy_words(q, c.q8_ycpu, c.pin_ycpu, size_t(T) * LAT * 4, "pin_to_card");
                mimo26_axpy(q, c.q8_ycpu, 1.f, c.moe32, size_t(T) * LAT);
                st_.q8_cpu += nc_cpu;
            }
        } else {
        if (ng) { KpQ{q, "memcpy"}.memcpy(c.q8_jslot, c.h_jslot.data(), size_t(ng) * 4); KpQ{q, "memcpy"}.memcpy(c.q8_jrow, c.h_jrow.data(), size_t(ng) * 4); KpQ{q, "memcpy"}.memcpy(c.q8_jw, c.h_jw.data(), size_t(ng) * 4); }
        KpQ{q, "memcpy"}.memcpy(c.q8_rs, c.h_rs.data(), (size_t(T) + 1) * 4);
        cast_fp32_to_fp16(q, c.lat32, c.ex_x16, size_t(T) * LAT);
        hy4f_q8_moe_rows(q, c.ex_x16, c.q8.slab(L), c.q8.slot_bytes(), c.q8.offsets(), c.q8_jslot, c.q8_jrow, c.q8_jw, c.q8_rs, ng, T,
                         c.q8_hrows, c.q8_yrows, c.moe32, LAT, MI, lim, c.q8.fmt());
        }
        if (nc && !streamed) {
            std::fill_n(c.pin_ycpu, size_t(T) * LAT, 0.f);
            const auto tc = std::chrono::steady_clock::now();
            if (cov_on_) h_hall_.resize(size_t(nc) * MI);
            static const bool skip_cpu_pp = [] { const char* v = std::getenv("IE_HY4F_DIAG_SKIP_CPU"); return v && *v == '1'; }();   // (timing only: WRONG output)
            if (!skip_cpu_pp || cov_on_) hy4f_q8_moe_cpu_rows(c.pin_lat, c.h_cpl.data(), c.h_crow.data(), c.h_cw.data(), nc, T, c.pin_ycpu, LAT, MI, lim, cpu_threads, c.q8.fmt(), cov_on_ ? h_hall_.data() : nullptr);
            st_.q8_ms_cpu += ms_since(tc);
            if (cov_on_) {
                uint32_t ord = 0; for (uint32_t l = 0; l < L; ++l) ord += cfg.moe[l];
                uint32_t n_moe = 0; for (uint32_t l = 0; l < cfg.n_layers; ++l) n_moe += cfg.moe[l];
                if (cov_hth_.empty()) { cov_hth_.assign(size_t(n_moe) * MI * MI, 0.0); cov_nh_.assign(n_moe, 0); }
                hy4f_gram_add(h_hall_.data(), nc, MI, cov_hth_.data() + size_t(ord) * MI * MI, cpu_threads);
                cov_nh_[ord] += nc;
            }
            copy_words(q, c.q8_ycpu, c.pin_ycpu, size_t(T) * LAT * 4, "pin_to_card");
            mimo26_axpy(q, c.q8_ycpu, 1.f, c.moe32, size_t(T) * LAT);
        }
        if (cov_x_) {                // the recorder: this call's latent inputs (read back with the router's logits)
            uint32_t ord = 0; for (uint32_t l = 0; l < L; ++l) ord += cfg.moe[l];
            uint32_t n_moe = 0; for (uint32_t l = 0; l < cfg.n_layers; ++l) n_moe += cfg.moe[l];
            if (cov_xtx_.empty()) { cov_xtx_.assign(size_t(n_moe) * LAT * LAT, 0.0); cov_nx_.assign(n_moe, 0); }
            hy4f_gram_add(c.pin_lat, T, LAT, cov_xtx_.data() + size_t(ord) * LAT * LAT, cpu_threads);
            cov_nx_[ord] += T;
        }
        q.wait();                    // the job tables and the CPU leg's sums above are host memory of this call
        if (!streamed) st_.q8_cpu += nc; else st_.q8_streamed += nc - nc_cpu;
        // The chunk has shown which of the layer's experts this text uses: bring the best of the missed ones in now
        // (IE_HY4F_REBALANCE = swaps a layer a chunk, default 32, 0 = off). The uploads run behind this layer's work.
        static const uint32_t rebalance_budget = [] { const char* v = std::getenv("IE_HY4F_REBALANCE"); return v ? uint32_t(std::atoi(v)) : 32u; }();
        if (rebalance_budget && !cov_on_ && !streamed) st_.q8_rebalanced += c.q8.rebalance(q, L, st_.rows + T, rebalance_budget);
        st_.q8_hits = 0; st_.q8_uploads = 0; st_.q8_promotions = 0;
        for (const auto& cc : cards_) if (cc->q8_on) { st_.q8_hits += cc->q8.stats().hits; st_.q8_uploads += cc->q8.stats().uploads; st_.q8_promotions += cc->q8.stats().promotions; }
        return {};
    }
    if (c.q8.fmt().gu != 8 || c.q8.fmt().dn != 8) return "the per-expert chain (IE_HY4F_Q8_FUSED=0 / IE_HY4F_Q8_CPU=0, or more than 16 experts a token) reads 8-bit experts only";
    float* moe = c.moe32; const float* lat = c.lat32;
    KpQ{q, "fill"}.fill(moe, 0.f, size_t(T) * LAT);
    const int32_t* ri = c.ex_rows; const float* rw = c.ex_w; sycl::half* x16 = c.ex_x16; sycl::half* h16 = c.ex_h16;
    const sycl::half* g16 = c.q8_g16; const sycl::half* u16 = c.q8_u16; const sycl::half* y16 = c.q8_y16;
    for (uint32_t e = 0; e < E; ++e) {
        const uint32_t n = uint32_t(rows[e].size());
        if (!n) continue;
        Hy4fQ8Store::Planes pl;
        if (auto er = c.q8.acquire(q, L, e, pl); !er.empty()) return er;
        KpQ{q, "memcpy"}.memcpy(c.ex_rows, rows[e].data(), size_t(n) * 4); KpQ{q, "memcpy"}.memcpy(c.ex_w, wts[e].data(), size_t(n) * 4);
        KpQ{q, "Hy4fQ8GatherK"}.parallel_for<class Hy4fQ8GatherK>(sycl::range<2>(n, LAT), [=](sycl::id<2> i) { x16[i[0] * LAT + i[1]] = sycl::half(lat[size_t(ri[i[0]]) * LAT + i[1]]); });
        gemv_q8_0_soa_f16_rows_dual(q, c.ex_x16, pl.gate_qs, pl.gate_d, c.q8_g16, pl.up_qs, pl.up_d, c.q8_u16, LAT, MI, n);
        KpQ{q, "Hy4fQ8SwigluK"}.parallel_for<class Hy4fQ8SwigluK>(sycl::range<1>(size_t(n) * MI), [=](sycl::id<1> i) {
            const float g = sycl::fmin(float(g16[i]), lim);
            const float u = sycl::fmin(sycl::fmax(float(u16[i]), -lim), lim);
            h16[i] = sycl::half(g * dsig(g) * u);
        });
        gemv_q8_0_soa_f16_rows(q, c.ex_h16, pl.down_qs, pl.down_d, c.q8_y16, MI, LAT, n);
        KpQ{q, "Hy4fQ8ScatterK"}.parallel_for<class Hy4fQ8ScatterK>(sycl::range<2>(n, LAT), [=](sycl::id<2> i) { moe[size_t(ri[i[0]]) * LAT + i[1]] += rw[i[0]] * float(y16[i[0] * LAT + i[1]]); });
    }
    q.wait();                        // the row lists and weights above are host vectors of this call
    st_.q8_hits = 0; st_.q8_uploads = 0;
    for (const auto& cc : cards_) if (cc->q8_on) { st_.q8_hits += cc->q8.stats().hits; st_.q8_uploads += cc->q8.stats().uploads; }
    return {};
}

// (IE_HY4F_PROF=1) wait for everything enqueued so far and charge the time since the last tick to `name`
void Hy4fForward::tick(sycl::queue& q, const char* name) {
    if (!prof_on_) return;
    q.wait();
    const auto now = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(now - prof_t_).count();
    prof_t_ = now;
    for (auto& p : prof_) if (p.first == name) { p.second += ms; return; }
    prof_.emplace_back(name, ms);
}

std::string Hy4fForward::write_cov(const std::string& prefix) const {
    if (cov_xtx_.empty()) return "no calibration data was recorded (multi-token calls with --q8 only)";
    auto put = [&](const std::string& name, const std::vector<double>& v) {
        std::FILE* f = std::fopen((prefix + name).c_str(), "wb");
        if (!f) return false;
        const bool ok = std::fwrite(v.data(), 8, v.size(), f) == v.size();
        std::fclose(f); return ok;
    };
    if (!put(".xtx.f64", cov_xtx_) || (!cov_hth_.empty() && !put(".hth.f64", cov_hth_))) return "cannot write " + prefix + ".*.f64";
    std::FILE* f = std::fopen((prefix + ".json").c_str(), "w");
    if (!f) return "cannot write " + prefix + ".json";
    std::fprintf(f, "{\"latent\": %u, \"moe_inter\": %u, \"tokens\": [", m_->config().latent_dim, m_->config().moe_inter_dim);
    for (size_t i = 0; i < cov_nx_.size(); ++i) std::fprintf(f, "%s%llu", i ? ", " : "", static_cast<unsigned long long>(cov_nx_[i]));
    std::fprintf(f, "], \"jobs\": [");
    for (size_t i = 0; i < cov_nh_.size(); ++i) std::fprintf(f, "%s%llu", i ? ", " : "", static_cast<unsigned long long>(cov_nh_[i]));
    std::fprintf(f, "]}\n"); std::fclose(f);
    return {};
}

bool Hy4fForward::pack_in_ram() const { for (const auto& c : cards_) if (c->q8_on && !c->q8.warm_done()) return false; return true; }
double Hy4fForward::pack_ram_gib() const { double b = 0; for (const auto& c : cards_) if (c->q8_on) b += double(c->q8.warm_bytes()); return b / double(1ull << 30); }

std::string Hy4fForward::set_steering(const float* vectors, float alpha) {
    const auto& cfg = m_->config();
    ++steer_version_;
    if (!vectors || alpha == 0.f) { steer_.clear(); steer_alpha_ = 0.f; return {}; }
    steer_.assign(vectors, vectors + size_t(cfg.n_layers) * cfg.hc_mult * cfg.dim); steer_alpha_ = alpha;
    return {};
}

std::string Hy4fForward::set_expert_mask(const std::vector<std::pair<uint32_t, uint32_t>>& layer_expert) {
    const auto& cfg = m_->config();
    mask_.clear();
    if (layer_expert.empty()) return {};
    mask_.assign(cfg.n_layers, {});
    std::vector<uint32_t> per(cfg.n_layers, 0);
    for (const auto& [L, e] : layer_expert) {
        if (L >= cfg.n_layers || e >= cfg.n_experts || !cfg.moe[L]) return "expert mask: layer " + std::to_string(L) + " expert " + std::to_string(e) + " is not a routed expert";
        if (mask_[L].empty()) mask_[L].assign(cfg.n_experts, 0);
        if (!mask_[L][e]) { mask_[L][e] = 1; ++per[L]; }
    }
    for (uint32_t L = 0; L < cfg.n_layers; ++L)
        if (per[L] + cfg.top_k > cfg.n_experts) return "expert mask: layer " + std::to_string(L) + " would have fewer than top_k experts left";
    return {};
}

std::string Hy4fForward::set_write_maps(const float* A, const float* B, uint32_t r) {
    if (r > 64) return "write maps of rank above 64";
    ++wm_version_; wm_r_ = r; if (wm_gain_.empty()) wm_gain_.resize(size_t(2) * m_->config().n_layers);
    if (!r) { wm_A_.clear(); wm_B_.clear(); return {}; }
    const size_t n = size_t(2) * m_->config().n_layers * r * m_->config().dim;
    wm_A_.assign(A, A + n); wm_B_.assign(B, B + n);
    return {};
}

std::string Hy4fForward::set_ablation(const float* basis, uint32_t k, float strength) {
    if (k > 256) return "more than 256 ablation directions";
    abl_k_ = k; abl_strength_ = strength; ++abl_version_;
    if (k) abl_basis_.assign(basis, basis + size_t(k) * m_->config().dim);
    return {};
}

std::string Hy4fForward::run_card(Card& c, size_t ci, const int32_t* ids, uint32_t T, uint32_t pos0, std::vector<float>& logits, bool all_rows) {
    (void)ids;
    const auto& m = *m_; const auto& cfg = m.config(); sycl::queue& q = *c.q;
    const uint32_t H = cfg.dim, hc = cfg.hc_mult, HH = hc * H, E = cfg.n_experts, TK = cfg.top_k, LAT = cfg.latent_dim, MI = cfg.moe_inter_dim;
    const float eps = cfg.norm_eps;
    const size_t RDb = size_t(T) * cfg.rope_dim * 4;
    std::memcpy(c.pin_S, h_x_.data(), size_t(T) * HH * 4); std::memcpy(c.pin_cos, h_cos_s_.data(), RDb); std::memcpy(c.pin_cos + size_t(T) * cfg.rope_dim, h_cos_i_.data(), RDb);
    copy_words(q, c.S, c.pin_S, size_t(T) * HH * 4, "pin_to_card");
    copy_words(q, c.cs_s, c.pin_cos, RDb, "pin_to_card"); copy_words(q, c.cs_i, c.pin_cos + size_t(T) * cfg.rope_dim, RDb, "pin_to_card");
    {   // which keys each row may see, for both layer kinds, sent once a call (was two small uploads in every layer)
        const uint32_t Wn = cfg.window, k0 = pos0 >= Wn - 1 ? pos0 - (Wn - 1) : 0;
        int32_t* rg = c.pin_rng;
        for (uint32_t t = 0; t < T; ++t) {
            const uint32_t p = pos0 + t;
            rg[t] = int32_t((p >= Wn - 1 ? p - (Wn - 1) : 0) - k0); rg[size_t(T) + t] = int32_t(p - k0);
            rg[size_t(2) * T + t] = 0; rg[size_t(3) * T + t] = int32_t(p);
        }
        copy_words(q, c.lo_s, rg, size_t(T) * 4, "pin_to_card"); copy_words(q, c.hi_s, rg + T, size_t(T) * 4, "pin_to_card");
        copy_words(q, c.lo_i, rg + size_t(2) * T, size_t(T) * 4, "pin_to_card"); copy_words(q, c.hi_i, rg + size_t(3) * T, size_t(T) * 4, "pin_to_card").wait();
    }
    if (abl_k_ && c.abl_version != abl_version_) {                  // the candidate changed: its basis to this card
        if (abl_k_ > c.abl_cap) {
            c.abl_B = dev<float>(c, size_t(abl_k_) * H); c.abl_coef = dev<float>(c, size_t(rows_max()) * hc * abl_k_); c.abl_cap = abl_k_;
            if (!c.abl_B || !c.abl_coef) return "device allocation failed (ablation basis)";
        }
        KpQ{q, "memcpy"}.memcpy(c.abl_B, abl_basis_.data(), size_t(abl_k_) * H * 4).wait(); c.abl_version = abl_version_;
    }
    auto ablate = [&]() { if (abl_k_) ablate_streams(q, c.S, c.abl_B, size_t(T) * hc, H, abl_k_, abl_strength_, c.abl_coef); };
    if (!steer_.empty() && c.steer_version != steer_version_) {
        if (!c.steer) { c.steer = dev<float>(c, size_t(c.L1 - c.L0) * HH); if (!c.steer) return "device allocation failed (steering vectors)"; }
        KpQ{q, "memcpy"}.memcpy(c.steer, steer_.data() + size_t(c.L0) * HH, size_t(c.L1 - c.L0) * HH * 4).wait(); c.steer_version = steer_version_;
    }
    // a sublayer's write y (c.y32, [T, H]) before it is added to the streams: recorded, then mapped (see set_write_maps)
    auto write_hook = [&](uint32_t L, uint32_t kind, const float* gain) {
        if (wrec_) {
            const uint32_t R = std::min(T, wrec_rows_); const size_t at = wrec_->size();
            wrec_->resize(at + size_t(wrec_rows_) * H, 0.f);          // (a call shorter than `rows`: zero rows first)
            KpQ{q, "memcpy"}.memcpy(wrec_->data() + at + size_t(wrec_rows_ - R) * H, c.y32 + size_t(T - R) * H, size_t(R) * H * 4).wait();
        }
        if (!wm_r_) return;
        // on the host (a search tool's path: the y of a call round-trips; the result of a search is baked into the weights)
        const uint32_t r = wm_r_; const size_t per = size_t(2) * r * H;
        const float* A = wm_A_.data() + size_t(L) * per + size_t(kind) * r * H; const float* B = wm_B_.data() + size_t(L) * per + size_t(kind) * r * H;
        bool any = false; for (size_t i = 0; i < size_t(r) * H && !any; ++i) any = B[i] != 0.f;
        if (!any) return;
        auto& gh = wm_gain_[size_t(L) * 2 + kind];
        if (gh.empty()) { gh.resize(H); KpQ{q, "memcpy"}.memcpy(gh.data(), gain, size_t(H) * 4).wait(); }
        wm_y_.resize(size_t(T) * H);
        KpQ{q, "memcpy"}.memcpy(wm_y_.data(), c.y32, size_t(T) * H * 4).wait();
        const float* g = gh.data(); float* yh = wm_y_.data();
        #pragma omp parallel for schedule(static) if (T > 4)
        for (int64_t t = 0; t < int64_t(T); ++t) {
            float* yr = yh + size_t(t) * H; float coef[64];
            for (uint32_t k = 0; k < r; ++k) { double sacc = 0; const float* a = A + size_t(k) * H; for (uint32_t jx = 0; jx < H; ++jx) sacc += double(a[jx]) * yr[jx]; coef[k] = float(sacc); }
            for (uint32_t k = 0; k < r; ++k) { const float ck = coef[k]; const float* b = B + size_t(k) * H; for (uint32_t jx = 0; jx < H; ++jx) yr[jx] += ck * b[jx]; }
            double ss = 0; uint32_t n = 0;
            for (uint32_t jx = 0; jx < H; ++jx) if (std::fabs(g[jx]) > 1e-6f) { const double u = double(yr[jx]) / g[jx]; ss += u * u; ++n; }
            const float sc = n ? float(1.0 / std::sqrt(ss / double(n) + double(eps))) : 1.f;
            for (uint32_t jx = 0; jx < H; ++jx) yr[jx] *= sc;
        }
        KpQ{q, "memcpy"}.memcpy(c.y32, wm_y_.data(), size_t(T) * H * 4).wait();
    };
    if (c.any_tp) stage_helper(*cards_[1 - ci], T, pos0);          // the other card computes half of this card's split layers
    if (ci == 0) ablate();                                         // the embeddings
    for (uint32_t L = c.L0; L < c.L1; ++L) {
        const Hy4fLayer& lw = m.layers()[L]; const Dense& d = c.dense[L - c.L0];
        if (opt_.stop_after && L >= opt_.stop_after) {          // (diagnostic) the first K layers only
            if (dump_) { const size_t at = dump_->size(); const uint32_t r0 = dump_last_ ? T - 1 : 0; dump_->resize(at + size_t(T - r0) * HH);
                         KpQ{q, "memcpy"}.memcpy(dump_->data() + at, c.S + size_t(r0) * HH, size_t(T - r0) * HH * 4).wait(); }
            stopped_ = true; logits.clear();
            return {};
        }
        if (dump_) {
            const size_t at = dump_->size(); const uint32_t r0 = dump_last_ ? T - 1 : 0;
            dump_->resize(at + size_t(T - r0) * HH);
            KpQ{q, "memcpy"}.memcpy(dump_->data() + at, c.S + size_t(r0) * HH, size_t(T - r0) * HH * 4).wait();
        }
        if (lw.engram >= 0 && plant() != "noengram") {
            const auto t0 = std::chrono::steady_clock::now();
            if (auto e = engram(c, d, m.engrams()[size_t(lw.engram)], T, pos0); !e.empty()) return "layer " + std::to_string(L) + " n-gram: " + e;
            ablate();
            st_.ms_engram += ms_since(t0);
        }
        // ---- attention ----
        const bool hpo = g_hp.on && (T == 1) == !g_hp.prompt;
        if (hpo) { g_hp.mark(7); }
        auto ta = std::chrono::steady_clock::now();
        if (prof_on_) { q.wait(); prof_t_ = std::chrono::steady_clock::now(); }
        if (predict_diag() && T == 1 && lw.moe && c.q8_on) {
            hc_mixes(q, c.S, d.hc_f_fn, c.mix, c.mix_part, T, 2 * hc, HH, false);
            hc_apply(q, c.S, c.mix, d.hc_f_base, d.hc_f_scale, c.pre, c.post, c.x, T, hc, H, eps, cfg.hc_eps, cfg.hc_magnitude, c.mix_part, HH);
            mimo26_rms_norm(q, c.x, d.ffn_norm, c.xn16, c.xn32, T, H, eps);
            hy4f_f32_gemv_f32(q, c.xn32, d.router_w, c.rlogits, H, E, T);
            copy_words(q, c.pin_rlog, c.rlogits, size_t(E) * 4, "card_to_pin").wait();
            g_pdiag.early.assign(c.pin_rlog, c.pin_rlog + E);
        }
        hc_mixes(q, c.S, d.hc_a_fn, c.mix, c.mix_part, T, 2 * hc, HH, false);
        tick(q, "a.hc_mixes");
        const bool fz = fuse_small_steps() && T <= kSmallRows && H <= 4096 && hc <= 8 && !range_on() && !prof_on_ && !abl_k_ && steer_.empty() && !wm_r_ && !wrec_;
        if (fz) hy4f_hc_apply_norm(q, c.S, c.mix_part, kHcParts, d.hc_a_base, d.hc_a_scale, c.pre, c.post, c.x, d.attn_norm, c.xn16, c.xn32, T, hc, H, HH, eps, cfg.hc_eps, cfg.hc_magnitude, eps);
        else {
        hc_apply(q, c.S, c.mix, d.hc_a_base, d.hc_a_scale, c.pre, c.post, c.x, T, hc, H, eps, cfg.hc_eps, cfg.hc_magnitude, c.mix_part, HH);
        tick(q, "a.hc_apply");
        range32(q, "stream", L, c.S, size_t(T) * HH); range32(q, "x_attn", L, c.x, size_t(T) * H);
        mimo26_rms_norm(q, c.x, d.attn_norm, c.xn16, c.xn32, T, H, eps);
        tick(q, "a.rms_norm");
        }
        tail_fused_ = fz; tp_add_ = nullptr;
        if (hpo) g_hp.mark(0);
        if (d.tp) {
            Card& hp = *cards_[1 - ci];
            if (hp.hL0 != c.L0 || L - c.L0 >= hp.helper.size() || !hp.helper[L - c.L0].nh) return "internal: layer " + std::to_string(L) + " is split but the other card holds no half of it";
            if (auto e = attention_tp(c, hp, d, hp.helper[L - c.L0], lw, L, T, pos0); !e.empty()) return e;
        }
        else if (auto e = attention(c, d, lw, L, T, pos0); !e.empty()) return e;
        tick(q, "a.o_norm");
        if (fz) hy4f_norm_post_add(q, c.o32, tp_add_, nullptr, d.post_norm, c.post, c.S, T, hc, H, eps);
        else { write_hook(L, 0, d.post_norm); post_add(q, c.S, c.post, c.y32, T, hc, H); }
        tail_fused_ = false; tp_add_ = nullptr;
        ablate();
        tick(q, "a.post_add");
        // The wait that times attention and FFN apart. IE_HY4F_STAGE_TIMES=0 drops it (measured on the real model,
        // 300 tokens: 14.58 tok/s without it, 14.77 with -- it costs nothing, so it stays on).
        static const bool stage_times = [] { const char* v = std::getenv("IE_HY4F_STAGE_TIMES"); return !(v && *v == '0'); }();
        if (hpo) g_hp.mark(0);
        if (stage_times || prof_on_) q.wait();
        if (hpo) g_hp.mark(1);
        st_.ms_attn += ms_since(ta);
        // ---- FFN ----
        ta = std::chrono::steady_clock::now();
        if (prof_on_) prof_t_ = std::chrono::steady_clock::now();
        hc_mixes(q, c.S, d.hc_f_fn, c.mix, c.mix_part, T, 2 * hc, HH, false);
        tick(q, "f.hc_mixes");
        if (fz) hy4f_hc_apply_norm(q, c.S, c.mix_part, kHcParts, d.hc_f_base, d.hc_f_scale, c.pre, c.post, c.x, d.ffn_norm, c.xn16, c.xn32, T, hc, H, HH, eps, cfg.hc_eps, cfg.hc_magnitude, eps);
        else {
        hc_apply(q, c.S, c.mix, d.hc_f_base, d.hc_f_scale, c.pre, c.post, c.x, T, hc, H, eps, cfg.hc_eps, cfg.hc_magnitude, c.mix_part, HH);
        tick(q, "f.hc_apply");
        mimo26_rms_norm(q, c.x, d.ffn_norm, c.xn16, c.xn32, T, H, eps);
        tick(q, "f.rms_norm");
        }
        bool ffn_tail_done = false;
        if (!lw.moe) {
            const uint32_t FI = cfg.inter_dim;
            mm(c, d.mlp_gate, c.xn16, c.ff_g32, T, FI, H);
            mm(c, d.mlp_up, c.xn16, c.ff_u32, T, FI, H);
            range32(q, "ffn_gate", L, c.ff_g32, size_t(T) * FI); range32(q, "ffn_up", L, c.ff_u32, size_t(T) * FI);
            mimo26_swiglu_f32(q, c.ff_g32, c.ff_u32, c.ff_h16, size_t(T) * FI);
            mm(c, d.mlp_down, c.ff_h16, c.o32, T, H, FI);
            if (fz) { hy4f_norm_post_add(q, c.o32, nullptr, nullptr, d.mlp_post_norm, c.post, c.S, T, hc, H, eps); ffn_tail_done = true; }
            else mimo26_rms_norm(q, c.o32, d.mlp_post_norm, nullptr, c.y32, T, H, eps);
            routes_[L].clear();
        } else {
            // the router on the full-width input (fp32), read back for the host's top-k
            const auto tr0 = std::chrono::steady_clock::now();
            // (the latent projection first: it does not depend on the picks, and in decode its read-back for the CPU leg
            // shares the router's wait -- one host round trip a layer instead of two)
            mm(c, d.lat_down, c.xn16, c.lat32, T, LAT, H);
            if (T > 8) gemm_nt_f32_onednn(q, c.xn32, d.router_w, c.rlogits, T, E, H);
            else if (smallk() && H % 8 == 0) hy4f_f32_gemv_f32(q, c.xn32, d.router_w, c.rlogits, H, E, T);
            else mimo26_router_logits(q, c.xn32, d.router_w, c.rlogits, T, H, E);
            if (c.q8_on) copy_words(q, c.pin_lat, c.lat32, size_t(T) * LAT * 4, "card_to_pin");
            { auto ev = copy_words(q, c.pin_rlog, c.rlogits, size_t(T) * E * 4, "card_to_pin"); if (hpo) g_hp.mark(2); ev.wait(); if (hpo) g_hp.mark(3); }
            st_.ms_router += ms_since(tr0);
            tick(q, "f.router+readback");
            const float* rbias = d.router_b.data();
            if (!mask_.empty() && !mask_[L].empty()) {          // masked experts: a bias no score can overcome
                h_bias_.assign(d.router_b.begin(), d.router_b.end());
                for (uint32_t e = 0; e < E; ++e) if (mask_[L][e]) h_bias_[e] = -1e30f;
                rbias = h_bias_.data();
            }
            for (uint32_t t = 0; t < T; ++t)
                hy4f_route(c.pin_rlog + size_t(t) * E, rbias, E, TK, cfg.norm_topk_prob, cfg.route_scale,
                           h_ridx_.data() + size_t(t) * TK, h_rw_.data() + size_t(t) * TK);
            routes_[L].assign(h_ridx_.begin(), h_ridx_.begin() + size_t(T) * TK);
            tick(q, "f.route_host_topk");
            if (predict_diag() && T == 1 && c.q8_on && g_pdiag.early.size() == E) {
                auto& g = g_pdiag;
                std::vector<float> sc(E); std::vector<uint32_t> ord(E);
                for (uint32_t e = 0; e < E; ++e) { sc[e] = 1.0f / (1.0f + std::exp(-g.early[e])) + rbias[e]; ord[e] = e; }
                std::sort(ord.begin(), ord.end(), [&](uint32_t a, uint32_t b) { return sc[a] > sc[b] || (sc[a] == sc[b] && a < b); });
                std::vector<uint32_t> rank(E); for (uint32_t i = 0; i < E; ++i) rank[ord[i]] = i;
                if (g.prev.size() < cfg.n_layers) g.prev.resize(cfg.n_layers);
                ++g.rows;
                for (uint32_t k = 0; k < TK; ++k) {
                    const uint32_t e = uint32_t(h_ridx_[k]); const bool miss = !c.q8.resident(L, e);
                    ++g.picks; g.misses += miss;
                    for (int i = 0; i < 4; ++i) if (rank[e] < PredictDiag::Ks[i]) { ++g.hit_all[i]; g.hit_miss[i] += miss; }
                    if (miss && std::find(g.prev[L].begin(), g.prev[L].end(), int32_t(e)) != g.prev[L].end()) ++g.prev_miss;
                }
                for (int i = 0; i < 4; ++i) for (uint32_t r = 0; r < PredictDiag::Ks[i]; ++r) g.nonres[i] += !c.q8.resident(L, ord[r]);
                for (int32_t e : g.prev[L]) g.prev_nonres += !c.q8.resident(L, uint32_t(e));
                g.prev[L].assign(h_ridx_.begin(), h_ridx_.begin() + TK);
                g.early.clear();
            }
            if (ref_routes_) {             // (the mini gate) compare with the reference's picks; use them when forcing
                uint32_t ord = 0;
                for (uint32_t l = 0; l < L; ++l) ord += cfg.moe[l];
                for (uint32_t t = 0; t < T && pos0 + t < ref_n_ && ord < ref_route_layers_; ++t) {
                    const int32_t* ref = ref_routes_ + (size_t(ord) * ref_n_ + pos0 + t) * TK;
                    int32_t* mine = h_ridx_.data() + size_t(t) * TK; float* w = h_rw_.data() + size_t(t) * TK;
                    std::vector<int32_t> a(mine, mine + TK), b(ref, ref + TK);
                    std::sort(a.begin(), a.end()); std::sort(b.begin(), b.end());
                    ++ref_diffs_.route_rows;
                    if (a != b) ref_diffs_.routes.push_back({L, pos0 + t});
                    if (force_routes_ && a != b) {
                        const float* lg = c.pin_rlog + size_t(t) * E;
                        float sum = 0.f;
                        for (uint32_t k = 0; k < TK; ++k) { mine[k] = ref[k]; w[k] = 1.0f / (1.0f + std::exp(-lg[ref[k]])); sum += w[k]; }
                        for (uint32_t k = 0; k < TK; ++k) w[k] = (cfg.norm_topk_prob ? w[k] / (sum + 1e-20f) : w[k]) * cfg.route_scale;
                    }
                }
            }
            // the routed experts in the latent width
            // IE_HY4F_LAT_DUMP=<prefix> (diagnostic): the experts' latent inputs of this call, [T, latent] fp32 appended to
            // <prefix>.L<layer>.f32 -- what a data-aware weight format would be fitted on (their covariance spectrum).
            if (static const char* lp = std::getenv("IE_HY4F_LAT_DUMP"); lp && *lp) {
                std::vector<float> hl(size_t(T) * LAT);
                KpQ{q, "memcpy"}.memcpy(hl.data(), c.lat32, hl.size() * 4).wait();
                char nm[512]; std::snprintf(nm, sizeof nm, "%s.L%02u.f32", lp, L);
                if (FILE* fp = std::fopen(nm, "ab")) { std::fwrite(hl.data(), 4, hl.size(), fp); std::fclose(fp); }
            }
            // the shared expert on the full-width input -- queued BEFORE the routed experts: it does not depend on them, and
            // the cards then compute it while the host computes the missed experts (the cards used to idle through that)
            mm(c, d.sh_gate, c.xn16, c.ff_g32, T, MI, H);
            tick(q, "f.gemm_sh_gate");
            mm(c, d.sh_up, c.xn16, c.ff_u32, T, MI, H);
            tick(q, "f.gemm_sh_up");
            range32(q, "sh_gate", L, c.ff_g32, size_t(T) * MI); range32(q, "sh_up", L, c.ff_u32, size_t(T) * MI);
            mimo26_swiglu_f32(q, c.ff_g32, c.ff_u32, c.ff_h16, size_t(T) * MI);
            tick(q, "f.swiglu");
            mm(c, d.sh_down, c.ff_h16, c.sd32, T, H, MI);
            tick(q, "f.gemm_sh_down");
            if (opt_.exact_experts) { if (auto e = exact_moe(c, lw, T); !e.empty()) return "layer " + std::to_string(L) + " exact experts: " + e; }
            else if (opt_.q8_experts) { if (auto e = q8_moe(c, L, T); !e.empty()) return "layer " + std::to_string(L) + " 8-bit experts: " + e; }
            else if (auto e = c.tier.moe(q, L, c.lat32, h_ridx_.data(), h_rw_.data(), T, c.moe32, cfg.swiglu_limit); !e.empty()) return "layer " + std::to_string(L) + " tier: " + e;
            if (hpo) g_hp.mark(4);
            range32(q, "lat_in", L, c.lat32, size_t(T) * LAT); range32(q, "moe_out", L, c.moe32, size_t(T) * LAT);
            tick(q, "f.experts");
            cast_fp32_to_fp16(q, c.moe32, c.lat16, size_t(T) * LAT);
            mm(c, d.lat_up, c.lat16, c.rout32, T, H, LAT);
            tick(q, "f.gemm_lat_up");
            range32(q, "routed", L, c.rout32, size_t(T) * H); range32(q, "shared", L, c.sd32, size_t(T) * H);
            if (fz) { hy4f_norm_post_add(q, c.rout32, nullptr, c.sd32, d.moe_norm, c.post, c.S, T, hc, H, eps); ffn_tail_done = true; }
            else {
            mimo26_axpy(q, c.sd32, 1.f, c.rout32, size_t(T) * H);
            mimo26_rms_norm(q, c.rout32, d.moe_norm, nullptr, c.y32, T, H, eps);
            }
        }
        tick(q, "f.axpy+norm");
        if (!ffn_tail_done) { write_hook(L, 1, lw.moe ? d.moe_norm : d.mlp_post_norm); post_add(q, c.S, c.post, c.y32, T, hc, H); }
        if (!steer_.empty()) {
            float* S = c.S; const float* v = c.steer + size_t(L - c.L0) * HH; const float al = steer_alpha_;
            KpQ{q, "Hy4fSteerK"}.parallel_for<class Hy4fSteerK>(sycl::range<2>(T, HH), [=](sycl::id<2> i) { S[i[0] * HH + i[1]] += al * v[i[1]]; });
        }
        ablate();
        tick(q, "f.post_add");
        if (hpo) g_hp.mark(5);
        q.wait(); st_.ms_ffn += ms_since(ta);
        if (hpo) { g_hp.mark(6); ++g_hp.layers; }
    }
    // the windowed layers' new key rows into host RAM (the kernels wrote them to the pinned stage; the waits below cover them)
    auto mirror_keys = [&]() {
        if (!c.n_kv_stage) return;
        const size_t LWk = size_t(cfg.kv_lora_rank) + cfg.rope_dim;
        for (const Dense& dd : c.dense) if (dd.kv_win && dd.kv_mirror)
            std::memcpy(kv_mirror_[size_t(dd.kv_host)].first + size_t(pos0) * LWk, c.pin_kv + size_t(dd.kv_stage) * rows_max() * LWk, size_t(T) * LWk * 2);
    };
    if (ci + 1 < cards_.size()) {
        copy_words(q, c.pin_S, c.S, size_t(T) * HH * 4, "card_to_pin").wait();      // the streams to the next card
        std::memcpy(h_x_.data(), c.pin_S, size_t(T) * HH * 4);
        mirror_keys();
        return {};
    }
    // ---- the head: collapse the streams, norm, logits ----
    hc_mixes(q, c.S, c.head_fn, c.mix, c.mix_part, T, hc, HH);
    hc_apply(q, c.S, c.mix, c.head_base, c.head_scale, c.pre, nullptr, c.x, T, hc, H, eps, cfg.hc_eps, 0.f);
    mimo26_rms_norm(q, c.x, c.final_norm, c.xn16, nullptr, T, H, eps);
    if (dfly_ && dfly_->on) {                 // the drafter reads the last accepted row's final hidden
        Dfly& D = *dfly_; D.th_row0 = T <= kSmallRows ? 0 : T - 1; D.th_rows = T - D.th_row0;
        copy_words(q, D.th16, c.xn16 + size_t(D.th_row0) * H, size_t(D.th_rows) * H * 2, "card");
    }
    const uint32_t r0 = all_rows ? 0 : T - 1, rows = T - r0, V = cfg.vocab_size;
    logits.resize(size_t(rows) * V);
    for (uint32_t b0 = 0; b0 < rows; b0 += kLogitRows) {
        const uint32_t nb = std::min(kLogitRows, rows - b0);
        gemm_nt_f16_onednn(q, c.xn16 + size_t(r0 + b0) * H, c.head, c.logits, nb, V, H);
        copy_words(q, c.pin_logits, c.logits, size_t(nb) * V * 4, "card_to_pin").wait();
        std::memcpy(logits.data() + size_t(b0) * V, c.pin_logits, size_t(nb) * V * 4);
    }
    mirror_keys();
    return {};
}

// The first card's extra piece (Hy4fOptions::extra_piece_bytes) leaves the expert slots: its VRAM is free for the caller.
std::string Hy4fForward::release_extra() {
    if (cards_.empty() || cards_[0]->extra_piece < 0) return "this load holds no extra piece";
    Card& c = *cards_[0];
    if (c.q8.ctx_gone(uint32_t(c.extra_piece))) return {};
    try { if (!c.q8.release_ctx(*c.q, uint32_t(c.extra_piece))) return "the extra piece could not be released"; }
    catch (const sycl::exception& x) { return std::string("releasing the extra piece: ") + x.what(); }
    return {};
}

// The indexed layers' caches hold `need` tokens: a card whose backing is short takes the next context piece from its
// expert slots (their experts become misses) and maps that VRAM behind the caches; a card backed for more than twice
// the need gives the top tier back to the slots, refilled with the best-scoring experts (a new, shorter conversation).
std::string Hy4fForward::fit_context(uint32_t need) {
    namespace ex = sycl::ext::oneapi::experimental;
    for (size_t ci = 0; ci < cards_.size(); ++ci) {
        Card& c = *cards_[ci];
        if (!c.el_on) continue;
        sycl::queue& q = *c.q;
        try {
            while (c.el_caps[c.el_tier] < need) {
                if (c.el_tier + 1 >= c.el_caps.size()) return "internal: the context cache has no further tier";
                const uint32_t t = c.el_tier + 1;
                if (!c.q8.release_ctx(q, t - 1)) return "internal: the expert slots hold no further context piece"; const uint64_t off = c.el_back[t - 1], add = c.el_back[t] - off;
                for (size_t b = 0; b < c.el_va.size(); ++b) {
                    c.el_ph[t].emplace_back(q.get_device(), q.get_context(), add);
                    c.el_ph[t].back().map(c.el_va[b] + off, add, ex::address_access_mode::read_write);
                    KpQ{q, "fill"}.fill(reinterpret_cast<sycl::half*>(c.el_va[b] + off), sycl::half(0.f), size_t(add / 2));
                }
                q.wait();
                c.el_tier = t; c.bytes += c.el_va.size() * add; c.kv_bytes += c.el_va.size() * add;
                std::fprintf(stderr, "[hy4-flash] card %zu: context cache grown to %u tokens (%u expert slots a layer now)\n", ci, c.el_caps[t], c.q8.vram_slots());
            }
            while (c.el_tier > 0 && uint64_t(need) * 2 <= c.el_caps[c.el_tier - 1] && !c.q8.lent()) {
                const uint32_t t = c.el_tier; const uint64_t off = c.el_back[t - 1], add = c.el_back[t] - off;
                q.wait();
                for (size_t b = 0; b < c.el_va.size(); ++b) ex::unmap(reinterpret_cast<void*>(c.el_va[b] + off), add, q.get_context());
                c.el_ph[t].clear();
                c.el_tier = t - 1; c.bytes -= c.el_va.size() * add; c.kv_bytes -= c.el_va.size() * add;
                if (auto e = c.q8.restore_ctx(q, st_.rows, t - 1); !e.empty()) { std::fprintf(stderr, "[hy4-flash] card %zu: %s (the slots stay as they are)\n", ci, e.c_str()); break; }
                std::fprintf(stderr, "[hy4-flash] card %zu: context cache back to %u tokens (%u expert slots a layer now)\n", ci, c.el_caps[c.el_tier], c.q8.vram_slots());
            }
        } catch (const sycl::exception& x) { return std::string("context cache growth: ") + x.what(); }
    }
    return {};
}

std::string Hy4fForward::forward(const int32_t* ids, uint32_t T, std::vector<float>& logits, bool all_rows, const Hy4fRowInputs* in) {
    if (cards_.empty()) return "not initialised";
    const auto& cfg = m_->config();
    if (T == 0 || T > (big_ok_ ? rows_max() : opt_.max_tokens)) return "T out of range";
    if (big_ok_) if (auto e = rows_mode(T > opt_.max_tokens); !e.empty()) return e;
    const uint32_t pos0 = n_pos_;
    if (pos0 + T > opt_.max_ctx) return "context capacity " + std::to_string(opt_.max_ctx) + " exceeded";
    const uint32_t H = cfg.dim, hc = cfg.hc_mult, RD = cfg.rope_dim, nf = RD / 2;
    const auto t0 = std::chrono::steady_clock::now();
    // experts uploaded beside the last call's compute (promote_async) become resident here, at the same point for the same tokens
    for (auto& cp : cards_) if (cp->q8_on) cp->q8.activate_pending();
    if (auto e = fit_context(pos0 + T); !e.empty()) return e;
    if (g_hp.on && (T == 1) == !g_hp.prompt) g_hp.t = std::chrono::steady_clock::now();
    if (m_->embed.w->dtype_str != "BF16") return "the token embedding is not BF16";
    const uint8_t* emb = m_->embed.w->data;
    pos3_.resize(size_t(pos0) * 3); h_egmask_.assign(T, 1); eg_any_masked_ = false;
    size_t n_emb = 0;
    for (uint32_t t = 0; t < T; ++t) {
        const int32_t id = ids[t];
        if (id < 0 || uint32_t(id) >= cfg.vocab_size) { hist_.resize(pos0); pos3_.resize(size_t(pos0) * 3); return "token id " + std::to_string(id) + " out of range"; }
        float* row = h_x_.data() + size_t(t) * hc * H;
        const bool own = in && in->is_embed && in->is_embed[t];
        if (own) std::memcpy(row, in->embeds + (n_emb++) * H, size_t(H) * 4);                      // an image row: the tower's vector
        else {
            const uint8_t* src = emb + size_t(id) * H * 2;
            for (uint32_t k = 0; k < H; ++k) { uint16_t b; std::memcpy(&b, src + size_t(k) * 2, 2); const uint32_t u = uint32_t(b) << 16; std::memcpy(&row[k], &u, 4); }
        }
        for (uint32_t i = 1; i < hc; ++i) std::memcpy(row + size_t(i) * H, row, size_t(H) * 4);
        // the row's 3-D position: given (an image's rows), else one past the largest axis of the row before it
        int32_t pos3[3];
        if (in && in->pos3) { pos3[0] = in->pos3[size_t(t) * 3]; pos3[1] = in->pos3[size_t(t) * 3 + 1]; pos3[2] = in->pos3[size_t(t) * 3 + 2]; }
        else pos3[0] = pos3[1] = pos3[2] = next_position();
        pos3_.insert(pos3_.end(), pos3, pos3 + 3);
        hy4f_rope_row(cfg.theta_swa, cfg.mrope_swa, RD, pos3, h_cos_s_.data() + size_t(t) * RD, h_cos_s_.data() + size_t(t) * RD + nf);
        hy4f_rope_row(cfg.theta_indexed, cfg.mrope_indexed, RD, pos3, h_cos_i_.data() + size_t(t) * RD, h_cos_i_.data() + size_t(t) * RD + nf);
        // visual rows carry projector output, not a token: no n-gram is defined over them (nor over the pad id)
        if (own || id == cfg.image_token_id || id == cfg.video_token_id || uint32_t(id) == cfg.engram_pad_id) { h_egmask_[t] = 0; eg_any_masked_ = true; }
        hist_.push_back(id);
    }
    // The n-gram tables are mapped from the drive (127 GB): ask for every row this call will read, for all four n-gram
    // layers, now -- the kernel reads them while the layers before each lookup compute (2026-11-18: on unseen text the
    // rows of a decode token cost about 5 ms, each lookup waiting for its own 16 page faults on the slower drive).
    {
        const uint32_t NHd = cfg.engram_n_heads(), HD = cfg.engram_head_dim();
        const long page = ::sysconf(_SC_PAGESIZE);
        int32_t ngram[8]; uint64_t rows[64];
        for (const Hy4fEngram& eg : m_->engrams()) {
            if ((!eg.fp8 && eg.table.w->dtype_str != "BF16") || NHd > 64) continue;
            std::vector<int32_t>& lookup = eg_lookup_[eg.id];
            if (lookup.empty()) { lookup.resize(eg.table_size); std::memcpy(lookup.data(), eg.lookup.w->data, size_t(eg.table_size) * 4); }
            const uint8_t* tab = eg.fp8 ? eg.fp8 : eg.table.w->data;
            const size_t stride = eg.fp8 ? 4 + size_t(HD) : size_t(HD) * 2;
            for (uint32_t t = 0; t < T; ++t) {
                const uint32_t p = pos0 + t;
                for (uint32_t j = 0; j < cfg.engram_max_ngram; ++j) ngram[j] = p >= j ? hist_[p - j] : int32_t(cfg.engram_pad_id);
                hy4f_engram_rows(eg, cfg, lookup.data(), ngram, rows);
                for (uint32_t h = 0; h < NHd; ++h) {
                    const uintptr_t a0 = reinterpret_cast<uintptr_t>(tab + rows[h] * stride), a1 = a0 + stride - 1;
                    const uintptr_t b0 = a0 & ~uintptr_t(page - 1), b1 = a1 & ~uintptr_t(page - 1);
                    ::madvise(reinterpret_cast<void*>(b0), size_t(b1 - b0) + size_t(page), MADV_WILLNEED);
                }
            }
        }
    }
    std::string err;
    try {
        stopped_ = false;
        for (size_t ci = 0; ci < cards_.size() && err.empty() && !stopped_; ++ci) err = run_card(*cards_[ci], ci, ids, T, pos0, logits, all_rows);
    } catch (const sycl::exception& e) {
        err = std::string("sycl: ") + e.what();
    }
    if (!err.empty()) {                       // leave the position where it was: the caller may rewind and retry
        hist_.resize(pos0); pos3_.resize(size_t(pos0) * 3);
        for (auto& ix : indexers_) ix.rewind(pos0);
        for (size_t i = 0; i < indexers_.size(); ++i) ix_dev_groups_[i] = std::min(ix_dev_groups_[i], indexers_[i].n_groups());
        return err;
    }
    n_pos_ = pos0 + T;
    if (g_hp.on && (T == 1) == !g_hp.prompt) { g_hp.mark(7); g_hp.tokens += T; }
    // IE_HY4F_VRAM_LOW=1 (a measurement): each card's lowest free VRAM after a call, printed when it falls
    if (static const bool vl = [] { const char* v = std::getenv("IE_HY4F_VRAM_LOW"); return v && *v == '1'; }(); vl) {
        static double low[2] = {1e30, 1e30};
        for (size_t ci = 0; ci < cards_.size() && ci < 2; ++ci) {
            const sycl::device dv = cards_[ci]->q->get_device();
            if (!dv.has(sycl::aspect::ext_intel_free_memory)) continue;
            const double fr = double(dv.get_info<sycl::ext::intel::info::device::free_memory>()) / double(1ull << 20);
            if (fr < low[ci] - 16) { low[ci] = fr; std::fprintf(stderr, "[hy4-flash] card %zu free VRAM low: %.0f MiB (after a %u-row call)\n", ci, fr, T); }
        }
    }
    // IE_HY4F_TRACE=<file> (a measurement): every call's picks appended -- uint32 T, then per MoE layer T x top-k int16
    if (static const char* tf = std::getenv("IE_HY4F_TRACE"); tf && *tf) {
        static FILE* fp = std::fopen(tf, "ab");
        if (fp) {
            const auto& cfg2 = m_->config(); std::fwrite(&T, 4, 1, fp);
            std::vector<int16_t> b16(size_t(T) * cfg2.top_k);
            for (uint32_t L = 0; L < cfg2.n_layers; ++L) {
                if (!cfg2.moe[L]) continue;
                for (size_t i = 0; i < b16.size(); ++i) b16[i] = i < routes_[L].size() ? int16_t(routes_[L][i]) : int16_t(-1);
                std::fwrite(b16.data(), 2, b16.size(), fp);
            }
            std::fflush(fp);
        }
    }
    st_.ms_total += ms_since(t0); ++st_.calls; st_.rows += T;
    return {};
}

}  // namespace ie
