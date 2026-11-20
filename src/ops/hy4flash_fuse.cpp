// src/ops/hy4flash_fuse.cpp -- Hy4-flash decode: the small steps around attention and the FFN in one launch each.
// Each fused kernel repeats its parts' arithmetic in the same order (the same per-lane strides of a 256-lane group,
// the same reduction), so the results are the parts' own bits. Prefixed hy4ffu_ (the kernel-name collision trap).
#include "ie/hy4flash_q8.hpp"
#include "ie/kernel_profiler.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>

namespace ie {

namespace {
inline float hy4ffu_sig(float x) { return 1.0f / (1.0f + sycl::exp(-x)); }
}

// The stream gates (from hc_mixes' partial sums), the collapse of the hc streams and the RMS norm of the result:
// hc_apply + mimo26_rms_norm for a few rows. pre / post [T, hc] and x [T, H] are written as hc_apply writes them,
// xn16 / xn32 (either may be null) as mimo26_rms_norm(x, w) writes them. post null = the head's form (n_mix = hc).
sycl::event hy4f_hc_apply_norm(sycl::queue& q, const float* S, const float* part, uint32_t n_parts, const float* base, const float* scale,
                               float* pre, float* post, float* x, const float* w, sycl::half* xn16, float* xn32,
                               uint32_t T, uint32_t hc, uint32_t H, uint32_t HH, float eps, float hc_eps, float mag, float norm_eps) {
    constexpr uint32_t WG = 256, MAXK = 16;                // H <= WG * MAXK (checked by the caller: kFuseMaxH)
    const uint32_t n_mix = post ? 2 * hc : hc, nk = (H + WG - 1) / WG;
    return ie::ps(q, "hy4ffu_hc_apply_norm", [&](sycl::handler& h) {
        sycl::local_accessor<float, 1> ml(sycl::range<1>(32), h), gp(sycl::range<1>(16), h);
        h.parallel_for(sycl::nd_range<1>(size_t(T) * WG, WG), [=](sycl::nd_item<1> it) {
            const uint32_t t = uint32_t(it.get_group(0)), i = uint32_t(it.get_local_id(0));
            if (i <= n_mix) {                      // each mix's partial sums, in HcGatesK's order, one lane a mix
                const float* pr = part + (size_t(t) * (n_mix + 1) + i) * n_parts;
                float a = 0.f;
                for (uint32_t pp = 0; pp < n_parts; ++pp) a += pr[pp];
                ml[i] = i < n_mix ? a : a / float(HH);
            }
            sycl::group_barrier(it.get_group());
            if (i < hc) {
                const float r = sycl::rsqrt(ml[n_mix] + eps);
                const float pv = hy4ffu_sig(ml[i] * r * scale[0] + base[i]) + hc_eps;
                pre[size_t(t) * hc + i] = pv; gp[i] = pv;
                if (post) post[size_t(t) * hc + i] = mag * hy4ffu_sig(ml[hc + i] * r * scale[1] + base[hc + i]) + hc_eps;
            }
            sycl::group_barrier(it.get_group());
            // HcCollapseK, then mimo26_rms_norm on the collapsed row (the same lanes and strides); every load before a store
            float xv[MAXK]; float ss = 0.f;
            #pragma unroll
            for (uint32_t m = 0; m < MAXK; ++m) {
                const uint32_t k = i + m * WG;
                if (m < nk && k < H) {
                    float a = 0.f;
                    for (uint32_t s2 = 0; s2 < hc; ++s2) a += gp[s2] * S[(size_t(t) * hc + s2) * H + k];
                    xv[m] = a; ss += a * a;
                } else xv[m] = 0.f;
            }
            ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
            const float inv = sycl::rsqrt(ss / float(H) + norm_eps);
            #pragma unroll
            for (uint32_t m = 0; m < MAXK; ++m) {
                const uint32_t k = i + m * WG;
                if (m < nk && k < H) {
                    x[size_t(t) * H + k] = xv[m];
                    const float v = xv[m] * inv * w[k];
                    if (xn16) xn16[size_t(t) * H + k] = sycl::half(v);
                    if (xn32) xn32[size_t(t) * H + k] = v;
                }
            }
        });
    });
}

sycl::event hy4f_norm_post_add(sycl::queue& q, float* o, const float* add, const float* axpy_x, const float* w, const float* post, float* S,
                               uint32_t T, uint32_t hc, uint32_t H, float eps) {
    constexpr uint32_t WG = 256, MAXK = 16;
    const uint32_t nk = (H + WG - 1) / WG;
    return ie::ps(q, "hy4ffu_norm_post_add", [&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(size_t(T) * WG, WG), [=](sycl::nd_item<1> it) {
            const uint32_t t = uint32_t(it.get_group(0)), i = uint32_t(it.get_local_id(0));
            float vv[MAXK]; float ss = 0.f;
            #pragma unroll
            for (uint32_t m = 0; m < MAXK; ++m) {
                const uint32_t k = i + m * WG;
                if (m < nk && k < H) {
                    float v = o[size_t(t) * H + k];
                    if (add) v += add[size_t(t) * H + k];
                    if (axpy_x) v += 1.f * axpy_x[size_t(t) * H + k];
                    vv[m] = v; ss += v * v;
                } else vv[m] = 0.f;
            }
            ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
            const float inv = sycl::rsqrt(ss / float(H) + eps);
            float pv[8];
            for (uint32_t s2 = 0; s2 < hc && s2 < 8; ++s2) pv[s2] = post[size_t(t) * hc + s2];
            for (uint32_t s2 = 0; s2 < hc; ++s2) {
                float* Sr = S + (size_t(t) * hc + s2) * H;
                float sv[MAXK];
                #pragma unroll
                for (uint32_t m = 0; m < MAXK; ++m) { const uint32_t k = i + m * WG; sv[m] = (m < nk && k < H) ? Sr[k] : 0.f; }
                #pragma unroll
                for (uint32_t m = 0; m < MAXK; ++m) { const uint32_t k = i + m * WG; if (m < nk && k < H) Sr[k] = sv[m] + pv[s2] * (vv[m] * inv * w[k]); }
            }
            #pragma unroll
            for (uint32_t m = 0; m < MAXK; ++m) { const uint32_t k = i + m * WG; if (m < nk && k < H) o[size_t(t) * H + k] = vv[m]; }
        });
    });
}

}  // namespace ie
