// src/ops/deepseek41_ops.cpp — V4.1's hyper-connection split.
//
// WHY THIS EXISTS RATHER THAN REUSING ds4_hyper_connection.
//
// V4 and V4.1 compute identical mHC coefficients — pre, post and the Sinkhorn-normalised comb
// are the same arithmetic, step for step (docs/deepseek41/03 addendum). What differs is WHEN
// `pre` is consumed:
//
//   V4   : hc_mixes and the stream collapse are ONE call — the collapse uses the `pre` that
//          same call just produced (src/model/deepseek4.cpp:3399, 4076).
//   V4.1 : "the coefficients a sublayer computes are used by the *next* one" (Block.forward,
//          model.py:968). Attention collapses with the pre_mix handed in from the previous
//          layer's FFN; the FFN collapses with the pre attention produced. This is the tech
//          report's "Single-Pass mHC, a revision of V4's mHC" made concrete.
//
// So V4.1 needs `pre` as an OUTPUT, and needs the collapse to take a pre from elsewhere.
// ds4_hyper_connection fuses the two and has four dispatch paths (split/fused x two tilings);
// threading a `pre` output through all of them to serve one caller would not be a surgical
// change. These two small kernels are.
//
// THE DUPLICATION IS GUARDED, NOT HOPED AT: tests/unit/ds41_hc_parity_test.cpp requires
// ds41_hc_mixes to reproduce ds4_hyper_connection's `post` and `comb` on the real layer-0
// hc tensors to within fp32 reduction-order noise (1e-5; the two kernels tile the RMS
// reduction differently, so bit-exactness is the wrong bar), and it must ALSO show that 19
// and 21 Sinkhorn iterations disagree with the reference -- so it proves it can see an
// iteration-count defect rather than passing on a trivial input.
#include "ie/deepseek41_upload.hpp"
#include "ie/kernel_profiler.hpp"

#include <cstdlib>
#include <string>

namespace ie {

namespace {

inline float ds41_sigmoid(float x) { return 1.0f / (1.0f + sycl::exp(-x)); }

// hc_mult == 4 Sinkhorn over a private 4x4, transcribed from ds4_sinkhorn4
// (src/ops/deepseek4_ops.cpp:93): column-normalise once, then (iters-1) x (row, column),
// every division carrying +eps.
inline void ds41_sinkhorn4(float* m, uint32_t iters, float eps) {
    for (int c = 0; c < 4; ++c) {
        float cs = 0.f;
        for (int r = 0; r < 4; ++r) cs += m[r * 4 + c];
        cs += eps;
        for (int r = 0; r < 4; ++r) m[r * 4 + c] /= cs;
    }
    for (uint32_t t = 1; t < iters; ++t) {
        for (int r = 0; r < 4; ++r) {
            float rs = 0.f;
            for (int c = 0; c < 4; ++c) rs += m[r * 4 + c];
            rs += eps;
            for (int c = 0; c < 4; ++c) m[r * 4 + c] /= rs;
        }
        for (int c = 0; c < 4; ++c) {
            float cs = 0.f;
            for (int r = 0; r < 4; ++r) cs += m[r * 4 + c];
            cs += eps;
            for (int r = 0; r < 4; ++r) m[r * 4 + c] /= cs;
        }
    }
}

}  // namespace

sycl::event ds41_hc_mixes(sycl::queue& q,
                          const float* streams, const float* fn, const float* base,
                          const float* scale, float* pre, float* post, float* comb,
                          uint32_t n_tokens, uint32_t hidden, uint32_t hc_mult,
                          uint32_t sinkhorn_iters, float rms_eps, float hc_eps,
                          const std::vector<sycl::event>& deps, float* scratch) {
    const uint32_t flat = hc_mult * hidden;
    const uint32_t mix  = (2u + hc_mult) * hc_mult;
    constexpr uint32_t WG = 256;
    static const bool t1_path = [] { const char* e = std::getenv("IE_DS41_HCMIX_T1"); return !(e && *e && std::string(e) == "0"); }();   // kill switch, for the before/after measurement
    if (scratch && t1_path) {   // DSpark P2 (docs/deepseek41/48 C.1 item 16): the same chunked shape for every row of a T-row decode step
        // DECODE (Phase 11). The general kernel below gives each of the `mix` (24) rows to ONE lane
        // that walks all `flat` (20,480) elements serially, in one work-group for the whole GPU:
        // 232 of 256 lanes idle and ~700 us per call at T = 1, where a prefill's thousands of
        // work-groups hide the same shape. A first version spread each row over the work-group's
        // 256 lanes and still measured ~700 us -- one work-group cannot pull 2 MB of `fn` at any
        // useful rate. Here (mix + 1) x kDs41HcChunks work-groups each reduce one row's chunk (the
        // extra row is sum of x^2 for the RMS) into `scratch`, and a one-work-group finish sums the
        // chunks and runs the same tail (sigmoids, softmax, sinkhorn). Summation order differs
        // from the general kernel's, so this shape is taken at T = 1 only (prefill digits unchanged).
        const uint32_t CH = kDs41HcChunks, chunk = (flat + CH - 1) / CH, RG = (mix + 1) * CH;   // groups per row
        float* part = scratch;                                            // [n_tokens, (mix + 1), CH]
        ie::ps(q, "ds41_hc_mixes_part", [&](sycl::handler& h) {
            h.depends_on(deps);
            h.parallel_for(sycl::nd_range<1>(size_t(n_tokens) * RG * WG, WG), [=](sycl::nd_item<1> it) {
                const uint32_t g = uint32_t(it.get_group(0)), tok = g / RG, rg = g % RG, r = rg / CH, c = rg % CH, lid = uint32_t(it.get_local_id(0));
                const uint32_t i0 = c * chunk, i1 = sycl::min(i0 + chunk, flat);
                const float* xs = streams + size_t(tok) * flat;
                float a = 0.f;
                if (r < mix) { const float* fr = fn + size_t(r) * flat; for (uint32_t i = i0 + lid; i < i1; i += WG) a += fr[i] * xs[i]; }
                else         { for (uint32_t i = i0 + lid; i < i1; i += WG) { const float v = xs[i]; a += v * v; } }
                const float tot = sycl::reduce_over_group(it.get_group(), a, sycl::plus<float>());
                if (lid == 0) part[(size_t(tok) * (mix + 1) + r) * CH + c] = tot;
            });
        });
        return ie::ps(q, "ds41_hc_mixes_finish", [&](sycl::handler& h) {
            sycl::local_accessor<float, 1> spart(sycl::range<1>(size_t(mix)), h);
            h.parallel_for(sycl::nd_range<1>(size_t(n_tokens) * WG, WG), [=](sycl::nd_item<1> it) {
                const uint32_t lid = uint32_t(it.get_local_id(0)), tok = uint32_t(it.get_group(0));
                const float* pt = part + size_t(tok) * (mix + 1) * CH;
                float ss = 0.f; for (uint32_t c = 0; c < CH; ++c) ss += pt[size_t(mix) * CH + c];
                const float rms = sycl::rsqrt(ss / float(flat) + rms_eps);
                if (lid < mix) { float a = 0.f; for (uint32_t c = 0; c < CH; ++c) a += pt[size_t(lid) * CH + c]; spart[lid] = a * rms; }
                it.barrier(sycl::access::fence_space::local_space);
                if (lid == 0) {
                    const uint32_t hc = hc_mult;
                    for (uint32_t s = 0; s < hc; ++s) {
                        pre [size_t(tok) * hc + s] = ds41_sigmoid(spart[s] * scale[0] + base[s]) + hc_eps;
                        post[size_t(tok) * hc + s] = 2.f * ds41_sigmoid(spart[hc + s] * scale[1] + base[hc + s]);
                    }
                    float m4[16];
                    for (uint32_t r = 0; r < hc; ++r) {
                        float mx = -3.402823466e+38f;
                        for (uint32_t c = 0; c < hc; ++c) {
                            const uint32_t o = 2u * hc + r * hc + c;
                            const float l = spart[o] * scale[2] + base[o];
                            m4[r * hc + c] = l;
                            mx = sycl::fmax(mx, l);
                        }
                        float den = 0.f;
                        for (uint32_t c = 0; c < hc; ++c) {
                            const float e = sycl::exp(m4[r * hc + c] - mx);
                            m4[r * hc + c] = e;
                            den += e;
                        }
                        for (uint32_t c = 0; c < hc; ++c) m4[r * hc + c] = m4[r * hc + c] / den + hc_eps;
                    }
                    ds41_sinkhorn4(m4, sinkhorn_iters, hc_eps);
                    for (uint32_t e = 0; e < hc * hc; ++e) comb[size_t(tok) * hc * hc + e] = m4[e];
                }
            });
        });
    }
    return ie::ps(q, "ds41_hc_mixes", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<float, 1> spart(sycl::range<1>(size_t(mix) * 1), h);
        sycl::local_accessor<float, 1> ssum(sycl::range<1>(WG), h);
        h.parallel_for(sycl::nd_range<1>(size_t(n_tokens) * WG, WG), [=](sycl::nd_item<1> it) {
            const uint32_t tok = uint32_t(it.get_group(0));
            const uint32_t lid = uint32_t(it.get_local_id(0));
            const float* xs = streams + size_t(tok) * flat;

            // RMS over the whole flattened hc*hidden stream — one statistic per token.
            float acc = 0.f;
            for (uint32_t i = lid; i < flat; i += WG) { const float v = xs[i]; acc += v * v; }
            ssum[lid] = acc;
            it.barrier(sycl::access::fence_space::local_space);
            for (uint32_t s = WG / 2; s; s >>= 1) {
                if (lid < s) ssum[lid] += ssum[lid + s];
                it.barrier(sycl::access::fence_space::local_space);
            }
            const float rms = sycl::rsqrt(ssum[0] / float(flat) + rms_eps);

            // mixes = (fn @ x) * rms
            for (uint32_t r = lid; r < mix; r += WG) {
                const float* fr = fn + size_t(r) * flat;
                float a = 0.f;
                for (uint32_t i = 0; i < flat; ++i) a += fr[i] * xs[i];
                spart[r] = a * rms;
            }
            it.barrier(sycl::access::fence_space::local_space);

            if (lid == 0) {
                const uint32_t hc = hc_mult;
                for (uint32_t s = 0; s < hc; ++s) {
                    pre [size_t(tok) * hc + s] = ds41_sigmoid(spart[s] * scale[0] + base[s]) + hc_eps;
                    post[size_t(tok) * hc + s] = 2.f * ds41_sigmoid(spart[hc + s] * scale[1] + base[hc + s]);
                }
                float m4[16];
                for (uint32_t r = 0; r < hc; ++r) {
                    float mx = -3.402823466e+38f;
                    for (uint32_t c = 0; c < hc; ++c) {
                        const uint32_t o = 2u * hc + r * hc + c;
                        const float l = spart[o] * scale[2] + base[o];
                        m4[r * hc + c] = l;
                        mx = sycl::fmax(mx, l);
                    }
                    float den = 0.f;
                    for (uint32_t c = 0; c < hc; ++c) {
                        const float e = sycl::exp(m4[r * hc + c] - mx);
                        m4[r * hc + c] = e;
                        den += e;
                    }
                    for (uint32_t c = 0; c < hc; ++c) m4[r * hc + c] = m4[r * hc + c] / den + hc_eps;
                }
                ds41_sinkhorn4(m4, sinkhorn_iters, hc_eps);
                for (uint32_t e = 0; e < hc * hc; ++e) comb[size_t(tok) * hc * hc + e] = m4[e];
            }
        });
    });
}

sycl::event ds41_hc_collapse(sycl::queue& q, const float* streams, const float* pre, float* y,
                             uint32_t n_tokens, uint32_t hidden, uint32_t hc_mult,
                             const std::vector<sycl::event>& deps) {
    // hc_pre (model.py:957): y[t, d] = sum_h pre[t, h] * streams[t, h, d].
    return ie::ps(q, "ds41_hc_collapse", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(size_t(n_tokens) * hidden), [=](sycl::id<1> gid) {
            const uint32_t t = uint32_t(gid[0] / hidden), d = uint32_t(gid[0] % hidden);
            const float* xs = streams + size_t(t) * hc_mult * hidden;
            float a = 0.f;
            for (uint32_t s = 0; s < hc_mult; ++s) a += pre[size_t(t) * hc_mult + s] * xs[size_t(s) * hidden + d];
            y[gid[0]] = a;
        });
    });
}

// Engram gate + residual (model.py:350-367). One work-group per (token, hc copy):
//   rstd = rsqrt(mean(h^2)+eps) * rsqrt(mean(key^2)+eps)         -- per copy, NOT jointly
//   dot  = sum_d h*qk*key * rstd * dim^-0.5
//   gate = sigmoid(copysign(sqrt(max(|dot|, 1e-6)), dot))         -- signed sqrt
//   out  = h + gate * value
sycl::event ds41_engram_gate(sycl::queue& q, const float* h, const float* key, const float* value,
                             const float* qk, float* out, float* gate_out,
                             uint32_t n_tokens, uint32_t hidden, uint32_t hc_mult, float eps,
                             const std::vector<sycl::event>& deps) {
    constexpr uint32_t WG = 256;
    const float inv_sqrt_dim = 1.0f / sycl::sqrt(float(hidden));
    return ie::ps(q, "ds41_engram_gate", [&](sycl::handler& hd) {
        hd.depends_on(deps);
        sycl::local_accessor<float, 1> red(sycl::range<1>(3 * WG), hd);
        hd.parallel_for(sycl::nd_range<1>(size_t(n_tokens) * hc_mult * WG, WG), [=](sycl::nd_item<1> it) {
            const uint32_t g = uint32_t(it.get_group(0)), lid = uint32_t(it.get_local_id(0));
            const uint32_t t = g / hc_mult, c = g % hc_mult;
            const float* hp = h   + (size_t(t) * hc_mult + c) * hidden;
            const float* kp = key + (size_t(t) * hc_mult + c) * hidden;
            const float* wp = qk  + size_t(c) * hidden;
            float sh = 0.f, sk = 0.f, sd = 0.f;
            for (uint32_t d = lid; d < hidden; d += WG) {
                const float hv = hp[d], kv = kp[d];
                sh += hv * hv; sk += kv * kv; sd += hv * wp[d] * kv;
            }
            red[lid] = sh; red[WG + lid] = sk; red[2 * WG + lid] = sd;
            it.barrier(sycl::access::fence_space::local_space);
            for (uint32_t s = WG / 2; s; s >>= 1) {
                if (lid < s) { red[lid] += red[lid + s]; red[WG + lid] += red[WG + lid + s];
                               red[2 * WG + lid] += red[2 * WG + lid + s]; }
                it.barrier(sycl::access::fence_space::local_space);
            }
            const float rstd = sycl::rsqrt(red[0] / float(hidden) + eps) *
                               sycl::rsqrt(red[WG] / float(hidden) + eps);
            const float dot  = red[2 * WG] * rstd * inv_sqrt_dim;
            const float mag  = sycl::sqrt(sycl::fmax(sycl::fabs(dot), 1e-6f));
            const float z    = sycl::copysign(mag, dot);
            const float gate = 1.0f / (1.0f + sycl::exp(-z));
            if (lid == 0) gate_out[size_t(t) * hc_mult + c] = gate;
            const float* vp = value + size_t(t) * hidden;
            float* op = out + (size_t(t) * hc_mult + c) * hidden;
            for (uint32_t d = lid; d < hidden; d += WG) op[d] = hp[d] + gate * vp[d];
        });
    });
}

}  // namespace ie
