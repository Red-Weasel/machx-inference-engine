// src/ops/qwen4_vision_gpu.cpp — qwen4exp vision tower, GPU path (P3).
//
// Spec: docs/qwen4/16_vision_port.md §3. Oracle: the CPU path (encode()),
// itself gated against the numpy oracle. All GEMMs ride the existing XMX
// gemm_fp16 (A[M,K]f16 × B[K,N]f16 → C[M,N]f32); the glue kernels here are
// bias/activation fusions, LayerNorm, the h/w vision rope, and a bidirectional
// packed attention kernel (no mask, no KV cache — one frame per image).

#include "ie/qwen4_vision.hpp"

#include "ie/allocator.hpp"
#include "ie/ops.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace ie {

struct Qwen4Vision::GpuState {
    DeviceAllocator* alloc = nullptr;
    struct BlockW {
        float *ln1_w, *ln1_b, *ln2_w, *ln2_b;      // [1152] f32
        sycl::half *qkv, *out, *up, *down;         // Bt [K, N] f16
        float *qkv_b, *out_b, *up_b, *down_b;      // [N] f32
    };
    sycl::half* patch = nullptr;                   // [1536, 1152]
    float* patch_b = nullptr;
    std::vector<BlockW> blk;
    float *post_w = nullptr, *post_b = nullptr;
    sycl::half *mm0 = nullptr, *mm2 = nullptr;     // [4608,4608], [4608,2560]
    float *mm0_b = nullptr, *mm2_b = nullptr;
    // scratch, sized on first encode (grown if a larger image arrives)
    uint32_t cap = 0;
    sycl::half *px = nullptr;                      // [N, 1536]
    float *posadd = nullptr;                       // [N, 1152]
    float *cs = nullptr, *sn = nullptr;            // [N, 36]
    float *res = nullptr;                          // [N, 1152]
    float *C = nullptr;                            // [N, 4608] gemm out
    sycl::half *xh = nullptr;                      // [N, 4608] gemm A staging
    sycl::half *qkv = nullptr;                     // [N, 3456]
    sycl::half *ffn = nullptr;                     // [N, 4304]
    sycl::half *att = nullptr;                     // [N, 1152]
};

namespace {

// y[i,j] = act(C[i,j] + b[j]) as f16. act: 0 none, 1 gelu_tanh, 2 gelu_erf.
sycl::event bias_act_f16(sycl::queue& q, const float* C, const float* b,
                         sycl::half* y, uint32_t M, uint32_t N, int act) {
    return q.parallel_for(sycl::range<1>(size_t(M) * N), [=](sycl::id<1> i) {
        const uint32_t j = uint32_t(i % N);
        float v = C[i] + b[j];
        if (act == 1)
            v = 0.5f * v * (1.0f + sycl::tanh(0.7978845608028654f * (v + 0.044715f * v * v * v)));
        else if (act == 2)
            v = 0.5f * v * (1.0f + sycl::erf(v * 0.70710678118654752f));
        y[i] = sycl::half(v);
    });
}

// res[i,j] += C[i,j] + b[j]
sycl::event bias_add_res(sycl::queue& q, const float* C, const float* b,
                         float* res, uint32_t M, uint32_t N) {
    return q.parallel_for(sycl::range<1>(size_t(M) * N), [=](sycl::id<1> i) {
        res[i] += C[i] + b[uint32_t(i % N)];
    });
}

// LayerNorm rows of f32 src -> f16 dst. One work-group per row.
sycl::event layer_norm_f16(sycl::queue& q, const float* src, const float* w,
                           const float* b, sycl::half* dst,
                           uint32_t M, uint32_t N) {
    constexpr uint32_t WG = 256;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> red(WG, h);
        h.parallel_for(sycl::nd_range<1>(size_t(M) * WG, WG),
                       [=](sycl::nd_item<1> it) {
            const uint32_t row = uint32_t(it.get_group(0));
            const uint32_t l = uint32_t(it.get_local_id(0));
            const float* x = src + size_t(row) * N;
            float s = 0;
            for (uint32_t j = l; j < N; j += WG) s += x[j];
            red[l] = s;
            it.barrier(sycl::access::fence_space::local_space);
            for (uint32_t st = WG / 2; st; st >>= 1) {
                if (l < st) red[l] += red[l + st];
                it.barrier(sycl::access::fence_space::local_space);
            }
            const float mu = red[0] / float(N);
            it.barrier(sycl::access::fence_space::local_space);
            float v = 0;
            for (uint32_t j = l; j < N; j += WG) { const float c = x[j] - mu; v += c * c; }
            red[l] = v;
            it.barrier(sycl::access::fence_space::local_space);
            for (uint32_t st = WG / 2; st; st >>= 1) {
                if (l < st) red[l] += red[l + st];
                it.barrier(sycl::access::fence_space::local_space);
            }
            const float inv = sycl::rsqrt(red[0] / float(N) + kVisLnEps);
            sycl::half* y = dst + size_t(row) * N;
            for (uint32_t j = l; j < N; j += WG)
                y[j] = sycl::half((x[j] - mu) * inv * w[j] + b[j]);
        });
    });
}

// Vision rope over q and k inside packed qkv [N, 3, heads, 72]:
// pairs (r, r+36), dims 0..17 by h-angle, 18..35 by w-angle (cs/sn [N, 36]).
sycl::event rope_hw(sycl::queue& q, sycl::half* qkv, const float* cs,
                    const float* sn, uint32_t N) {
    const uint32_t items = N * 2 * kVisHeads * kVisRot;   // q,k only
    return q.parallel_for(sycl::range<1>(items), [=](sycl::id<1> id) {
        uint32_t i = uint32_t(id);
        const uint32_t r = i % kVisRot;         i /= kVisRot;
        const uint32_t hd = i % kVisHeads;      i /= kVisHeads;
        const uint32_t qk = i % 2;              i /= 2;
        const uint32_t tok = i;
        sycl::half* v = qkv + ((size_t(tok) * 3 + qk) * kVisHeads + hd) * kVisHeadDim;
        const float c = cs[size_t(tok) * kVisRot + r];
        const float s = sn[size_t(tok) * kVisRot + r];
        const float a = float(v[r]), b2 = float(v[r + kVisRot]);
        v[r] = sycl::half(a * c - b2 * s);
        v[r + kVisRot] = sycl::half(a * s + b2 * c);
    });
}

// Bidirectional packed attention: out[i, hd*72..] = softmax(q_i · K / sqrt(72)) V.
// One work-group per (token, head); scores staged in SLM (N floats — caller
// guards N <= 12288).
sycl::event attn_full(sycl::queue& q, const sycl::half* qkv, sycl::half* out,
                      uint32_t N) {
    constexpr uint32_t WG = 128;
    const float scale = 1.0f / std::sqrt(float(kVisHeadDim));
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sc(N, h);
        sycl::local_accessor<float, 1> qv(kVisHeadDim, h);
        sycl::local_accessor<float, 1> red(WG, h);
        h.parallel_for(sycl::nd_range<2>({size_t(N), size_t(kVisHeads) * WG},
                                         {1, WG}),
                       [=](sycl::nd_item<2> it) {
            const uint32_t tok = uint32_t(it.get_global_id(0));
            const uint32_t hd = uint32_t(it.get_group(1));
            const uint32_t l = uint32_t(it.get_local_id(1));
            auto row = [&](uint32_t t2, uint32_t which) {
                return qkv + ((size_t(t2) * 3 + which) * kVisHeads + hd) * kVisHeadDim;
            };
            const sycl::half* qr = row(tok, 0);
            for (uint32_t d = l; d < kVisHeadDim; d += WG) qv[d] = float(qr[d]);
            it.barrier(sycl::access::fence_space::local_space);
            // scores
            for (uint32_t j = l; j < N; j += WG) {
                const sycl::half* kr = row(j, 1);
                float d2 = 0;
                for (uint32_t d = 0; d < kVisHeadDim; ++d) d2 += qv[d] * float(kr[d]);
                sc[j] = d2 * scale;
            }
            it.barrier(sycl::access::fence_space::local_space);
            // max
            float mx = -1e30f;
            for (uint32_t j = l; j < N; j += WG) mx = sycl::fmax(mx, sc[j]);
            red[l] = mx;
            it.barrier(sycl::access::fence_space::local_space);
            for (uint32_t st = WG / 2; st; st >>= 1) {
                if (l < st) red[l] = sycl::fmax(red[l], red[l + st]);
                it.barrier(sycl::access::fence_space::local_space);
            }
            mx = red[0];
            it.barrier(sycl::access::fence_space::local_space);
            // exp + sum
            float sm = 0;
            for (uint32_t j = l; j < N; j += WG) {
                const float e = sycl::native::exp(sc[j] - mx);
                sc[j] = e;
                sm += e;
            }
            red[l] = sm;
            it.barrier(sycl::access::fence_space::local_space);
            for (uint32_t st = WG / 2; st; st >>= 1) {
                if (l < st) red[l] += red[l + st];
                it.barrier(sycl::access::fence_space::local_space);
            }
            const float inv = 1.0f / red[0];
            it.barrier(sycl::access::fence_space::local_space);
            // weighted V, lane d owns output dims d, d+WG, ...
            for (uint32_t d = l; d < kVisHeadDim; d += WG) {
                float acc = 0;
                for (uint32_t j = 0; j < N; ++j)
                    acc += sc[j] * float(row(j, 2)[d]);
                out[size_t(tok) * kVisHid + hd * kVisHeadDim + d] =
                    sycl::half(acc * inv);
            }
        });
    });
}

// host f32 [out, in] -> device f16 Bt [in, out]
sycl::half* upload_bt(DeviceAllocator& a, const std::vector<float>& w,
                      uint32_t out_d, uint32_t in_d) {
    std::vector<sycl::half> bt(size_t(in_d) * out_d);
    for (uint32_t o = 0; o < out_d; ++o)
        for (uint32_t k = 0; k < in_d; ++k)
            bt[size_t(k) * out_d + o] = sycl::half(w[size_t(o) * in_d + k]);
    auto* d = static_cast<sycl::half*>(a.malloc(bt.size() * 2));
    a.queue().memcpy(d, bt.data(), bt.size() * 2).wait();
    return d;
}

float* upload_f32(DeviceAllocator& a, const std::vector<float>& w) {
    auto* d = static_cast<float*>(a.malloc(w.size() * 4));
    a.queue().memcpy(d, w.data(), w.size() * 4).wait();
    return d;
}

}  // namespace

std::string Qwen4Vision::encode_gpu(DeviceAllocator& alloc, const float* img,
                                    uint32_t H, uint32_t W,
                                    std::vector<float>& out) {
    if (!loaded()) return "qwen4 vision: not loaded";
    const uint32_t f = kVisPatch * kVisMerge;
    if (H % f || W % f || !H || !W) return "qwen4 vision: H/W must be multiples of 32";
    const uint32_t gh = H / kVisPatch, gw = W / kVisPatch;
    const uint32_t N = gh * gw;
    if (N > 12288) return "qwen4 vision gpu: patch grid too large (cap via IE_VIS_MAX_PX)";
    const uint32_t pd = 3 * 2 * kVisPatch * kVisPatch;   // 1536
    const uint32_t md = kVisHid * 4;                     // 4608
    sycl::queue& q = alloc.queue();

    // ---- one-time weight upload ---------------------------------------------
    if (!gpu_) {
        gpu_ = std::make_unique<GpuState>();
        GpuState& g = *gpu_;
        g.alloc = &alloc;
        g.patch = upload_bt(alloc, patch_w_, kVisHid, pd);
        g.patch_b = upload_f32(alloc, patch_b_);
        g.blk.resize(kVisDepth);
        for (uint32_t L = 0; L < kVisDepth; ++L) {
            const Block& s = blocks_[L];
            auto& d = g.blk[L];
            d.ln1_w = upload_f32(alloc, s.ln1_w); d.ln1_b = upload_f32(alloc, s.ln1_b);
            d.ln2_w = upload_f32(alloc, s.ln2_w); d.ln2_b = upload_f32(alloc, s.ln2_b);
            d.qkv = upload_bt(alloc, s.qkv_w, 3 * kVisHid, kVisHid);
            d.out = upload_bt(alloc, s.out_w, kVisHid, kVisHid);
            d.up = upload_bt(alloc, s.up_w, kVisFfn, kVisHid);
            d.down = upload_bt(alloc, s.down_w, kVisHid, kVisFfn);
            d.qkv_b = upload_f32(alloc, s.qkv_b); d.out_b = upload_f32(alloc, s.out_b);
            d.up_b = upload_f32(alloc, s.up_b);   d.down_b = upload_f32(alloc, s.down_b);
        }
        g.post_w = upload_f32(alloc, post_ln_w_);
        g.post_b = upload_f32(alloc, post_ln_b_);
        g.mm0 = upload_bt(alloc, mm0_w_, md, md);
        g.mm0_b = upload_f32(alloc, mm0_b_);
        g.mm2 = upload_bt(alloc, mm2_w_, kVisOutD, md);
        g.mm2_b = upload_f32(alloc, mm2_b_);
    }
    GpuState& g = *gpu_;
    if (g.alloc != &alloc) return "qwen4 vision gpu: allocator changed between calls";
    if (g.cap < N) {
        auto am = [&](size_t bytes) { return alloc.malloc(bytes); };
        g.px = static_cast<sycl::half*>(am(size_t(N) * pd * 2));
        g.posadd = static_cast<float*>(am(size_t(N) * kVisHid * 4));
        g.cs = static_cast<float*>(am(size_t(N) * kVisRot * 4));
        g.sn = static_cast<float*>(am(size_t(N) * kVisRot * 4));
        g.res = static_cast<float*>(am(size_t(N) * kVisHid * 4));
        g.C = static_cast<float*>(am(size_t(N) * md * 4));
        g.xh = static_cast<sycl::half*>(am(size_t(N) * md * 2));
        g.qkv = static_cast<sycl::half*>(am(size_t(N) * 3 * kVisHid * 2));
        g.ffn = static_cast<sycl::half*>(am(size_t(N) * kVisFfn * 2));
        g.att = static_cast<sycl::half*>(am(size_t(N) * kVisHid * 2));
        g.cap = N;
    }

    // ---- host prep: block-major patches (f16), pos-embed interp, rope tables —
    // identical ordering/taps to the CPU path (encode()).
    std::vector<sycl::half> px(size_t(N) * pd);
    std::vector<float> posadd(size_t(N) * kVisHid);
    std::vector<float> cs(size_t(N) * kVisRot), sn(size_t(N) * kVisRot);
    {
        const uint32_t pp = kVisPatch * kVisPatch;
        uint32_t idx = 0;
        for (uint32_t br = 0; br < gh / kVisMerge; ++br)
            for (uint32_t bc = 0; bc < gw / kVisMerge; ++bc)
                for (uint32_t ir = 0; ir < kVisMerge; ++ir)
                    for (uint32_t ic = 0; ic < kVisMerge; ++ic, ++idx) {
                        const uint32_t pr = br * kVisMerge + ir, pc = bc * kVisMerge + ic;
                        sycl::half* d = &px[size_t(idx) * pd];
                        for (uint32_t c = 0; c < 3; ++c)
                            for (uint32_t y = 0; y < kVisPatch; ++y)
                                for (uint32_t x2 = 0; x2 < kVisPatch; ++x2) {
                                    const sycl::half v = sycl::half(
                                        img[(size_t(c) * H + pr * kVisPatch + y) * W
                                            + pc * kVisPatch + x2]);
                                    sycl::half* t2 = &d[(size_t(c) * 2) * pp
                                                        + y * kVisPatch + x2];
                                    t2[0] = v; t2[pp] = v;
                                }
                        // pos embed (bilinear, align_corners=True) + rope angles
                        auto taps = [&](float pos, uint32_t size, uint32_t (&tp)[2],
                                        float (&wt)[2]) {
                            const float denom = size > 1 ? float(size - 1) : 1.f;
                            const float src = pos * float(kVisGridSide - 1) / denom;
                            const float fl = std::floor(src);
                            for (int k = 0; k < 2; ++k) {
                                const float raw = fl + k;
                                tp[k] = uint32_t(std::min(std::max(raw, 0.f),
                                                          float(kVisGridSide - 1)));
                                wt[k] = std::max(1.f - std::fabs(src - raw), 0.f);
                            }
                        };
                        uint32_t th[2], tw[2]; float wh[2], ww[2];
                        taps(float(pr), gh, th, wh);
                        taps(float(pc), gw, tw, ww);
                        float* pa = &posadd[size_t(idx) * kVisHid];
                        std::memset(pa, 0, kVisHid * 4);
                        for (int a = 0; a < 2; ++a)
                            for (int b = 0; b < 2; ++b) {
                                const float wgt = wh[a] * ww[b];
                                const float* rw = &pos_embd_[
                                    (size_t(th[a]) * kVisGridSide + tw[b]) * kVisHid];
                                for (uint32_t j = 0; j < kVisHid; ++j) pa[j] += wgt * rw[j];
                            }
                        for (uint32_t r2 = 0; r2 < 18; ++r2) {
                            const float invf = std::exp(-2.f * float(r2) / float(kVisRot)
                                                        * std::log(kVisTheta));
                            cs[size_t(idx) * kVisRot + r2] = std::cos(float(pr) * invf);
                            sn[size_t(idx) * kVisRot + r2] = std::sin(float(pr) * invf);
                            cs[size_t(idx) * kVisRot + 18 + r2] = std::cos(float(pc) * invf);
                            sn[size_t(idx) * kVisRot + 18 + r2] = std::sin(float(pc) * invf);
                        }
                    }
    }
    q.memcpy(g.px, px.data(), px.size() * 2);
    q.memcpy(g.posadd, posadd.data(), posadd.size() * 4);
    q.memcpy(g.cs, cs.data(), cs.size() * 4);
    q.memcpy(g.sn, sn.data(), sn.size() * 4);

    // ---- forward (in-order queue) -------------------------------------------
    gemm_fp16(q, g.px, g.patch, g.C, N, kVisHid, pd);
    // res = C + patch_bias + posadd  (fold patch bias into posadd add)
    q.parallel_for(sycl::range<1>(size_t(N) * kVisHid),
                   [C = g.C, pa = g.posadd, res = g.res, pb = g.patch_b,
                    hd2 = kVisHid](sycl::id<1> i) {
        res[i] = C[i] + pa[i] + pb[uint32_t(i % hd2)];
    });

    for (uint32_t L = 0; L < kVisDepth; ++L) {
        auto& d = g.blk[L];
        layer_norm_f16(q, g.res, d.ln1_w, d.ln1_b, g.xh, N, kVisHid);
        gemm_fp16(q, g.xh, d.qkv, g.C, N, 3 * kVisHid, kVisHid);
        bias_act_f16(q, g.C, d.qkv_b, g.qkv, N, 3 * kVisHid, 0);
        rope_hw(q, g.qkv, g.cs, g.sn, N);
        attn_full(q, g.qkv, g.att, N);
        gemm_fp16(q, g.att, d.out, g.C, N, kVisHid, kVisHid);
        bias_add_res(q, g.C, d.out_b, g.res, N, kVisHid);
        layer_norm_f16(q, g.res, d.ln2_w, d.ln2_b, g.xh, N, kVisHid);
        gemm_fp16(q, g.xh, d.up, g.C, N, kVisFfn, kVisHid);
        bias_act_f16(q, g.C, d.up_b, g.ffn, N, kVisFfn, 1);
        gemm_fp16(q, g.ffn, d.down, g.C, N, kVisHid, kVisFfn);
        bias_add_res(q, g.C, d.down_b, g.res, N, kVisHid);
    }

    const uint32_t Nm = N / 4;
    layer_norm_f16(q, g.res, g.post_w, g.post_b, g.xh, N, kVisHid);
    gemm_fp16(q, g.xh, g.mm0, g.C, Nm, md, md);
    bias_act_f16(q, g.C, g.mm0_b, g.xh, Nm, md, 2);
    gemm_fp16(q, g.xh, g.mm2, g.C, Nm, kVisOutD, md);
    out.resize(size_t(Nm) * kVisOutD);
    // bias into the f32 download
    q.parallel_for(sycl::range<1>(out.size()),
                   [C = g.C, b = g.mm2_b, od = kVisOutD](sycl::id<1> i) {
        C[i] += b[uint32_t(i % od)];
    });
    q.memcpy(out.data(), g.C, out.size() * 4).wait();
    return {};
}

Qwen4Vision::Qwen4Vision() = default;
Qwen4Vision::~Qwen4Vision() = default;

}  // namespace ie
