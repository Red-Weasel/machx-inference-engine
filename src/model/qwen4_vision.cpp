// src/model/qwen4_vision.cpp — qwen4exp vision tower, CPU fp32 reference path.
//
// Spec: docs/qwen4/16_vision_port.md §3. Golden oracle:
// tests/oracle/qwen4_vision_oracle.py (same GGUF, same synthetic image).

#include "ie/qwen4_vision.hpp"

#include "ie/gguf.hpp"
#include "ie/quant_blocks.hpp"  // fp16_to_fp32

#include <algorithm>
#include <cmath>
#include <cstring>

namespace ie {
namespace {

// Materialize any F16/F32 gguf tensor as f32.
std::string to_f32(const GgufReader& r, const char* name, std::vector<float>& out,
                   size_t expect) {
    const GgufTensorInfo* ti = r.find_tensor(name);
    if (!ti) return std::string("qwen4 vision: missing tensor ") + name;
    size_t n = 1;
    for (uint32_t d = 0; d < ti->n_dims; ++d) n *= ti->shape[d];
    if (n != expect)
        return std::string("qwen4 vision: ") + name + " has " + std::to_string(n) +
               " elems, want " + std::to_string(expect);
    out.resize(n);
    if (ti->dtype == DType::kF32) {
        std::memcpy(out.data(), ti->data, n * 4);
    } else if (ti->dtype == DType::kF16) {
        const uint16_t* h = reinterpret_cast<const uint16_t*>(ti->data);
        for (size_t i = 0; i < n; ++i) out[i] = fp16_to_fp32(h[i]);
    } else {
        return std::string("qwen4 vision: ") + name + " unexpected dtype";
    }
    return {};
}

void layer_norm(const float* x, const float* w, const float* b, float* y,
                uint32_t n, uint32_t d) {
    for (uint32_t i = 0; i < n; ++i) {
        const float* xi = x + size_t(i) * d;
        float* yi = y + size_t(i) * d;
        double mu = 0;
        for (uint32_t j = 0; j < d; ++j) mu += xi[j];
        mu /= d;
        double var = 0;
        for (uint32_t j = 0; j < d; ++j) { double c = xi[j] - mu; var += c * c; }
        var /= d;
        const float inv = 1.0f / std::sqrt(float(var) + kVisLnEps);
        for (uint32_t j = 0; j < d; ++j)
            yi[j] = (float(xi[j] - mu)) * inv * w[j] + b[j];
    }
}

// y[n, dout] = x[n, din] * W^T + b, W stored [dout, din] row-major (gguf order).
void gemm_bias(const float* x, const float* w, const float* b, float* y,
               uint32_t n, uint32_t din, uint32_t dout) {
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < int64_t(n); ++i) {
        const float* xi = x + size_t(i) * din;
        float* yi = y + size_t(i) * dout;
        for (uint32_t o = 0; o < dout; ++o) {
            const float* wo = w + size_t(o) * din;
            double acc = 0;
            for (uint32_t j = 0; j < din; ++j) acc += double(xi[j]) * wo[j];
            yi[o] = float(acc) + b[o];
        }
    }
}

inline float gelu_tanh(float x) {   // gelu_pytorch_tanh (ViT MLP)
    return 0.5f * x * (1.0f + std::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}
inline float gelu_erf(float x) {    // nn.GELU default (merger)
    return 0.5f * x * (1.0f + std::erf(x * 0.70710678118654752f));
}

}  // namespace

std::string Qwen4Vision::load(const std::string& mmproj_path) {
    GgufReader r;
    if (auto e = r.open(mmproj_path); !e.empty()) return e;

    const size_t pw = size_t(kVisHid) * 3 * kVisPatch * kVisPatch;  // one temporal slice
    std::vector<float> w0, w1;
    if (auto e = to_f32(r, "v.patch_embd.weight", w0, pw); !e.empty()) return e;
    if (auto e = to_f32(r, "v.patch_embd.weight.1", w1, pw); !e.empty()) return e;
    // Fuse the two temporal slices into [1152, C*T*ph*pw] with the reference's
    // per-patch layout [C, T, ph, pw] (slice t is strided into position).
    const uint32_t pp = kVisPatch * kVisPatch;
    patch_w_.assign(size_t(kVisHid) * 3 * 2 * pp, 0.f);
    for (uint32_t o = 0; o < kVisHid; ++o)
        for (uint32_t c = 0; c < 3; ++c)
            for (uint32_t p = 0; p < pp; ++p) {
                const size_t src = (size_t(o) * 3 + c) * pp + p;
                float* dst = &patch_w_[((size_t(o) * 3 + c) * 2) * pp + p];
                dst[0] = w0[src];        // t = 0
                dst[pp] = w1[src];       // t = 1
            }
    if (auto e = to_f32(r, "v.patch_embd.bias", patch_b_, kVisHid); !e.empty()) return e;
    if (auto e = to_f32(r, "v.position_embd.weight", pos_embd_,
                        size_t(kVisGridSide) * kVisGridSide * kVisHid); !e.empty()) return e;
    if (auto e = to_f32(r, "v.post_ln.weight", post_ln_w_, kVisHid); !e.empty()) return e;
    if (auto e = to_f32(r, "v.post_ln.bias", post_ln_b_, kVisHid); !e.empty()) return e;
    const uint32_t md = kVisHid * kVisMerge * kVisMerge;   // 4608
    if (auto e = to_f32(r, "mm.0.weight", mm0_w_, size_t(md) * md); !e.empty()) return e;
    if (auto e = to_f32(r, "mm.0.bias", mm0_b_, md); !e.empty()) return e;
    if (auto e = to_f32(r, "mm.2.weight", mm2_w_, size_t(kVisOutD) * md); !e.empty()) return e;
    if (auto e = to_f32(r, "mm.2.bias", mm2_b_, kVisOutD); !e.empty()) return e;

    blocks_.resize(kVisDepth);
    for (uint32_t L = 0; L < kVisDepth; ++L) {
        Block& blk = blocks_[L];
        const std::string p = "v.blk." + std::to_string(L) + ".";
        struct { const char* sfx; std::vector<float>* v; size_t n; } items[] = {
            {"ln1.weight", &blk.ln1_w, kVisHid}, {"ln1.bias", &blk.ln1_b, kVisHid},
            {"ln2.weight", &blk.ln2_w, kVisHid}, {"ln2.bias", &blk.ln2_b, kVisHid},
            {"attn_qkv.weight", &blk.qkv_w, size_t(3) * kVisHid * kVisHid},
            {"attn_qkv.bias", &blk.qkv_b, size_t(3) * kVisHid},
            {"attn_out.weight", &blk.out_w, size_t(kVisHid) * kVisHid},
            {"attn_out.bias", &blk.out_b, kVisHid},
            {"ffn_up.weight", &blk.up_w, size_t(kVisFfn) * kVisHid},
            {"ffn_up.bias", &blk.up_b, kVisFfn},
            {"ffn_down.weight", &blk.down_w, size_t(kVisHid) * kVisFfn},
            {"ffn_down.bias", &blk.down_b, kVisHid},
        };
        for (auto& it : items)
            if (auto e = to_f32(r, (p + it.sfx).c_str(), *it.v, it.n); !e.empty()) return e;
    }
    return {};
}

std::string Qwen4Vision::encode(const float* img, uint32_t H, uint32_t W,
                                std::vector<float>& out) const {
    if (!loaded()) return "qwen4 vision: not loaded";
    const uint32_t f = kVisPatch * kVisMerge;
    if (H % f || W % f || !H || !W) return "qwen4 vision: H/W must be multiples of 32";
    const uint32_t gh = H / kVisPatch, gw = W / kVisPatch;
    const uint32_t N = gh * gw;
    const uint32_t pp = kVisPatch * kVisPatch;
    const uint32_t pd = 3 * 2 * pp;                 // 1536 per-patch input

    // Block-major patch extraction, layout [C, T, ph, pw]; (h, w) ids alongside.
    std::vector<float> px(size_t(N) * pd);
    std::vector<uint32_t> ph(N), pwid(N);
    {
        uint32_t idx = 0;
        for (uint32_t br = 0; br < gh / kVisMerge; ++br)
            for (uint32_t bc = 0; bc < gw / kVisMerge; ++bc)
                for (uint32_t ir = 0; ir < kVisMerge; ++ir)
                    for (uint32_t ic = 0; ic < kVisMerge; ++ic, ++idx) {
                        const uint32_t pr = br * kVisMerge + ir, pc = bc * kVisMerge + ic;
                        ph[idx] = pr; pwid[idx] = pc;
                        float* d = &px[size_t(idx) * pd];
                        for (uint32_t c = 0; c < 3; ++c)
                            for (uint32_t y = 0; y < kVisPatch; ++y)
                                for (uint32_t x2 = 0; x2 < kVisPatch; ++x2) {
                                    const float v = img[(size_t(c) * H + pr * kVisPatch + y) * W
                                                        + pc * kVisPatch + x2];
                                    float* t2 = &d[(size_t(c) * 2) * pp + y * kVisPatch + x2];
                                    t2[0] = v; t2[pp] = v;   // duplicated frame
                                }
                    }
    }

    // Patch embed + interpolated pos embed (bilinear, align_corners=True).
    std::vector<float> x(size_t(N) * kVisHid);
    gemm_bias(px.data(), patch_w_.data(), patch_b_.data(), x.data(), N, pd, kVisHid);
    for (uint32_t i = 0; i < N; ++i) {
        auto taps = [&](float idx, uint32_t size, uint32_t (&tp)[2], float (&wt)[2]) {
            const float denom = size > 1 ? float(size - 1) : 1.f;
            const float src = idx * float(kVisGridSide - 1) / denom;
            const float fl = std::floor(src);
            for (int k = 0; k < 2; ++k) {
                float raw = fl + k;
                tp[k] = uint32_t(std::min(std::max(raw, 0.f), float(kVisGridSide - 1)));
                wt[k] = std::max(1.f - std::fabs(src - raw), 0.f);
            }
        };
        uint32_t th[2], tw[2]; float wh[2], ww[2];
        taps(float(ph[i]), gh, th, wh);
        taps(float(pwid[i]), gw, tw, ww);
        float* xi = &x[size_t(i) * kVisHid];
        for (int a = 0; a < 2; ++a)
            for (int b = 0; b < 2; ++b) {
                const float wgt = wh[a] * ww[b];
                const float* row = &pos_embd_[(size_t(th[a]) * kVisGridSide + tw[b]) * kVisHid];
                for (uint32_t j = 0; j < kVisHid; ++j) xi[j] += wgt * row[j];
            }
    }

    // Rope tables: dims 0..17 rotate by h, 18..35 by w; pairs (i, i+36).
    std::vector<float> cosb(size_t(N) * kVisRot), sinb(size_t(N) * kVisRot);
    for (uint32_t i = 0; i < N; ++i) {
        for (uint32_t r2 = 0; r2 < 18; ++r2) {
            const float invf = std::exp(-2.f * float(r2) / float(kVisRot) * std::log(kVisTheta));
            const float ah = float(ph[i]) * invf, aw = float(pwid[i]) * invf;
            cosb[size_t(i) * kVisRot + r2] = std::cos(ah);
            sinb[size_t(i) * kVisRot + r2] = std::sin(ah);
            cosb[size_t(i) * kVisRot + 18 + r2] = std::cos(aw);
            sinb[size_t(i) * kVisRot + 18 + r2] = std::sin(aw);
        }
    }

    // Transformer blocks (full bidirectional attention over the frame).
    std::vector<float> h(size_t(N) * kVisHid), qkv(size_t(N) * 3 * kVisHid),
        attn(size_t(N) * kVisHid), ffn(size_t(N) * kVisFfn);
    for (uint32_t L = 0; L < kVisDepth; ++L) {
        const Block& blk = blocks_[L];
        layer_norm(x.data(), blk.ln1_w.data(), blk.ln1_b.data(), h.data(), N, kVisHid);
        gemm_bias(h.data(), blk.qkv_w.data(), blk.qkv_b.data(), qkv.data(),
                  N, kVisHid, 3 * kVisHid);
        // rope q and k in place: token i, head hd, pair (r, r+36)
        for (uint32_t qk = 0; qk < 2; ++qk)
#pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < int64_t(N); ++i) {
                float* base = &qkv[size_t(i) * 3 * kVisHid + qk * kVisHid];
                const float* ci = &cosb[size_t(i) * kVisRot];
                const float* si = &sinb[size_t(i) * kVisRot];
                for (uint32_t hd = 0; hd < kVisHeads; ++hd) {
                    float* v2 = base + size_t(hd) * kVisHeadDim;
                    for (uint32_t r2 = 0; r2 < kVisRot; ++r2) {
                        const float a = v2[r2], b = v2[r2 + kVisRot];
                        v2[r2] = a * ci[r2] - b * si[r2];
                        v2[r2 + kVisRot] = a * si[r2] + b * ci[r2];
                    }
                }
            }
        const float scale = 1.0f / std::sqrt(float(kVisHeadDim));
#pragma omp parallel for schedule(static) collapse(2)
        for (int64_t hd = 0; hd < int64_t(kVisHeads); ++hd)
            for (int64_t i = 0; i < int64_t(N); ++i) {
                const float* qi = &qkv[size_t(i) * 3 * kVisHid + hd * kVisHeadDim];
                std::vector<float> sc(N);
                float mx = -1e30f;
                for (uint32_t j = 0; j < N; ++j) {
                    const float* kj = &qkv[size_t(j) * 3 * kVisHid + kVisHid + hd * kVisHeadDim];
                    double d2 = 0;
                    for (uint32_t r2 = 0; r2 < kVisHeadDim; ++r2) d2 += double(qi[r2]) * kj[r2];
                    sc[j] = float(d2) * scale;
                    mx = std::max(mx, sc[j]);
                }
                double sum = 0;
                for (uint32_t j = 0; j < N; ++j) { sc[j] = std::exp(sc[j] - mx); sum += sc[j]; }
                float* oi = &attn[size_t(i) * kVisHid + hd * kVisHeadDim];
                for (uint32_t r2 = 0; r2 < kVisHeadDim; ++r2) {
                    double acc = 0;
                    for (uint32_t j = 0; j < N; ++j)
                        acc += double(sc[j]) *
                               qkv[size_t(j) * 3 * kVisHid + 2 * kVisHid + hd * kVisHeadDim + r2];
                    oi[r2] = float(acc / sum);
                }
            }
        gemm_bias(attn.data(), blk.out_w.data(), blk.out_b.data(), h.data(),
                  N, kVisHid, kVisHid);
        for (size_t i = 0; i < size_t(N) * kVisHid; ++i) x[i] += h[i];

        layer_norm(x.data(), blk.ln2_w.data(), blk.ln2_b.data(), h.data(), N, kVisHid);
        gemm_bias(h.data(), blk.up_w.data(), blk.up_b.data(), ffn.data(),
                  N, kVisHid, kVisFfn);
        for (size_t i = 0; i < size_t(N) * kVisFfn; ++i) ffn[i] = gelu_tanh(ffn[i]);
        gemm_bias(ffn.data(), blk.down_w.data(), blk.down_b.data(), h.data(),
                  N, kVisFfn, kVisHid);
        for (size_t i = 0; i < size_t(N) * kVisHid; ++i) x[i] += h[i];
    }

    // post_ln -> group 4 (block-major consecutive) -> mm.0 -> erf gelu -> mm.2
    layer_norm(x.data(), post_ln_w_.data(), post_ln_b_.data(), h.data(), N, kVisHid);
    const uint32_t Nm = N / 4, md = kVisHid * 4;
    std::vector<float> merged(size_t(Nm) * md);
    gemm_bias(h.data(), mm0_w_.data(), mm0_b_.data(), merged.data(), Nm, md, md);
    for (auto& v2 : merged) v2 = gelu_erf(v2);
    out.resize(size_t(Nm) * kVisOutD);
    gemm_bias(merged.data(), mm2_w_.data(), mm2_b_.data(), out.data(), Nm, md, kVisOutD);
    return {};
}

}  // namespace ie

// M-RoPE position table for a mixed text+image token sequence — HF
// get_rope_index port (docs/qwen4/16_vision_port.md §4). `imgs` must be
// sorted by t0 and non-overlapping.
void ie::qwen4_build_mrope3(uint32_t n, const std::vector<Qwen4VisGrid>& imgs,
                            std::vector<int32_t>& pos3, int32_t& delta) {
    pos3.assign(size_t(3) * n, 0);
    size_t next = 0;
    int32_t cur = 0, mx = 0;
    for (uint32_t t = 0; t < n; ) {
        if (next < imgs.size() && t == imgs[next].t0) {
            const auto& im = imgs[next];
            for (uint32_t r = 0; r < im.gh2; ++r)
                for (uint32_t c = 0; c < im.gw2; ++c) {
                    const uint32_t i = im.t0 + r * im.gw2 + c;
                    pos3[i] = cur;
                    pos3[size_t(n) + i] = cur + int32_t(r);
                    pos3[size_t(2) * n + i] = cur + int32_t(c);
                }
            mx = std::max(mx, cur + int32_t(std::max(im.gh2, im.gw2)) - 1);
            cur += int32_t(std::max(im.gh2, im.gw2));
            t += im.gh2 * im.gw2;
            ++next;
        } else {
            pos3[t] = pos3[size_t(n) + t] = pos3[size_t(2) * n + t] = cur;
            mx = std::max(mx, cur);
            ++cur;
            ++t;
        }
    }
    delta = mx + 1 - int32_t(n);
}
