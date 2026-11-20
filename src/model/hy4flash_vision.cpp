// src/model/hy4flash_vision.cpp — Tencent Hy4-flash: the image loader and the vision tower (include/ie/hy4flash_vision.hpp).
// The plan's arithmetic is Python's (this file is built with -fp-model=precise, src/CMakeLists.txt). The tower's GEMMs go
// through oneDNN (fp16 operands, fp32 accumulation); activations are fp32 between blocks.
#include "ie/hy4flash_vision.hpp"

#include "ie/ds4_vision.hpp"       // ds4_pil_resize: PIL's bicubic, the processor's resample
#include "ie/ops.hpp"
#include "stb/stb_image.h"          // implementation instantiated in qwen4_image.cpp

#include <algorithm>
#include <cmath>
#include <cstring>

namespace ie {

std::string hy4f_vis_plan(uint32_t height, uint32_t width, Hy4fVisGeom& g, uint64_t max_px) {
    g = {};
    if (!height || !width) return "vision: the image is empty";
    const double h = height, w = width, f = kHy4fVisFactor;
    if (std::max(h, w) / std::min(h, w) > 200.0) return "vision: the image's aspect ratio is above 200";
    double hb = std::nearbyint(h / f) * f, wb = std::nearbyint(w / f) * f;        // Python's round: half to even
    if (hb * wb > double(max_px)) {
        const double beta = std::sqrt(h * w / double(max_px));
        hb = std::max(f, std::floor(h / beta / f) * f);
        wb = std::max(f, std::floor(w / beta / f) * f);
    } else if (hb * wb < double(kHy4fVisMinPx)) {
        const double beta = std::sqrt(double(kHy4fVisMinPx) / (h * w));
        hb = std::ceil(h * beta / f) * f;
        wb = std::ceil(w * beta / f) * f;
    }
    g.h = uint32_t(hb); g.w = uint32_t(wb);
    g.grid_h = g.h / kHy4fVisPatch; g.grid_w = g.w / kHy4fVisPatch;
    if (g.grid_h < 2 * kHy4fVisMerge || g.grid_w < 2 * kHy4fVisMerge) return "vision: the image is too narrow for the projector (a merged grid needs two rows and two columns)";
    return {};
}

std::string hy4f_load_image_mem(const void* bytes, size_t nbytes, std::vector<float>& pv, Hy4fVisGeom& g, uint64_t max_px) {
    int sw = 0, sh = 0, comp = 0;
    uint8_t* rgb = stbi_load_from_memory(static_cast<const stbi_uc*>(bytes), int(nbytes), &sw, &sh, &comp, 3);
    if (!rgb) return std::string("vision: image decode failed: ") + stbi_failure_reason();
    std::string err = hy4f_vis_plan(uint32_t(sh), uint32_t(sw), g, max_px);
    if (!err.empty()) { stbi_image_free(rgb); return err; }
    std::vector<uint8_t> canvas(size_t(g.w) * g.h * 3);
    if (int(g.w) == sw && int(g.h) == sh) std::memcpy(canvas.data(), rgb, canvas.size());
    else ds4_pil_resize(rgb, sw, sh, canvas.data(), int(g.w), int(g.h));
    stbi_image_free(rgb);
    static constexpr float kMean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
    static constexpr float kStd[3]  = {0.26862954f, 0.26130258f, 0.27577711f};
    const uint32_t P = kHy4fVisPatch;
    pv.assign(size_t(g.patches()) * kHy4fVisPatchDim, 0.f);
    float* out = pv.data();
    for (uint32_t gy = 0; gy < g.grid_h; ++gy)
        for (uint32_t gx = 0; gx < g.grid_w; ++gx)
            for (uint32_t c = 0; c < 3; ++c)
                for (uint32_t py = 0; py < P; ++py)
                    for (uint32_t px = 0; px < P; ++px) {
                        const uint8_t v = canvas[(size_t(gy * P + py) * g.w + gx * P + px) * 3 + c];
                        const float r = float(double(v) * (1.0 / 255.0));
                        *out++ = (r - kMean[c]) / kStd[c];
                    }
    return {};
}

int32_t hy4f_vis_positions(const Hy4fVisGeom& g, int32_t start, std::vector<int32_t>& pos3) {
    const uint32_t R = g.rows(), C = g.cols();
    pos3.resize(size_t(R) * C * 3);
    for (uint32_t r = 0; r < R; ++r)
        for (uint32_t c = 0; c < C; ++c) {
            int32_t* p = pos3.data() + (size_t(r) * C + c) * 3;
            p[0] = start; p[1] = start + int32_t(r); p[2] = start + int32_t(c);
        }
    return start + int32_t(std::max(R - 1, C - 1)) + 1;
}

namespace {
constexpr uint32_t kD = 1152, kHeads = 16, kHd = 72, kFfn = 4304, kLayers = 27, kMergeIn = 4 * kD, kGrid = 128;
constexpr float kProbScale = 256.f;        // f16 probabilities stored as p * 256: a uniform row over 16,384 keys keeps its digits

inline float bf16_at(const uint8_t* p, size_t i) { uint16_t b; std::memcpy(&b, p + i * 2, 2); const uint32_t u = uint32_t(b) << 16; float f; std::memcpy(&f, &u, 4); return f; }

// per row: mean and 1 / sqrt(var + eps)  (LayerNorm)  or  0 and 1 / sqrt(mean(x^2) + eps)  (RMSNorm)
sycl::event norm_stat(sycl::queue& q, const float* x, float* stat, uint32_t M, uint32_t N, float eps, bool center) {
    return q.parallel_for(sycl::range<1>(M), [=](sycl::id<1> i) {
        const float* r = x + size_t(i[0]) * N;
        float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
        if (center) { for (uint32_t k = 0; k + 3 < N; k += 4) { s0 += r[k]; s1 += r[k + 1]; s2 += r[k + 2]; s3 += r[k + 3]; } }
        const float mean = center ? ((s0 + s1) + (s2 + s3)) / float(N) : 0.f;
        float v0 = 0.f, v1 = 0.f, v2 = 0.f, v3 = 0.f;
        for (uint32_t k = 0; k + 3 < N; k += 4) {
            const float a = r[k] - mean, b = r[k + 1] - mean, c = r[k + 2] - mean, d = r[k + 3] - mean;
            v0 += a * a; v1 += b * b; v2 += c * c; v3 += d * d;
        }
        stat[i[0] * 2] = mean; stat[i[0] * 2 + 1] = sycl::rsqrt(((v0 + v1) + (v2 + v3)) / float(N) + eps);
    });
}
sycl::event norm_apply(sycl::queue& q, const float* x, const float* stat, const float* w, const float* b, sycl::half* y, uint32_t M, uint32_t N) {
    return q.parallel_for(sycl::range<2>(M, N), [=](sycl::id<2> i) {
        const float v = (x[i[0] * N + i[1]] - stat[i[0] * 2]) * stat[i[0] * 2 + 1] * w[i[1]] + (b ? b[i[1]] : 0.f);
        y[i[0] * N + i[1]] = sycl::half(v);
    });
}
}  // namespace

struct Hy4fVision::Impl {
    sycl::queue* q = nullptr;
    std::vector<void*> owned; uint64_t bytes = 0, max_px = 0; uint32_t n_max = 0; bool open = false;
    uint32_t out_dim = 0; float eps = 1e-5f;
    std::vector<float> pos_table;          // host [128 * 128, 1152]: the learned grid (the checkpoint's rows 1 ..)
    std::vector<float> newline;            // host [out_dim]
    struct Block { float *ln1w, *ln1b, *ln2w, *ln2b, *bq, *bk, *bv, *bo, *b1, *b2; sycl::half *wq, *wk, *wv, *wo, *w1, *w2; };
    std::vector<Block> blocks;
    sycl::half *w_patch = nullptr, *w_conv = nullptr, *w_out = nullptr; float *b_patch = nullptr, *b_conv = nullptr, *b_out = nullptr, *w_rms = nullptr;
    // scratch
    sycl::half *px16 = nullptr, *xh16 = nullptr, *q_hm = nullptr, *k_hm = nullptr, *v_hm = nullptr, *P16 = nullptr, *O16 = nullptr, *att16 = nullptr, *mh16 = nullptr, *g16 = nullptr, *gh16 = nullptr;
    float *x32 = nullptr, *C32 = nullptr, *S32 = nullptr, *Cm32 = nullptr, *stat = nullptr, *pos32 = nullptr, *Cg32 = nullptr, *out32 = nullptr;
    uint32_t tile_attn = 1024, tile_mlp = 2048;

    template <class T> T* dev(size_t n) {
        T* p = sycl::malloc_device<T>(n ? n : 1, *q);
        if (p) { owned.push_back(p); bytes += n * sizeof(T); }
        return p;
    }
    // a BF16 / F32 tensor as fp16 [rows, cols] (row-major as stored) or as fp32
    std::string up16(const Hy4FlashModel& m, const std::string& name, size_t want, sycl::half*& out, const std::vector<uint32_t>* perm = nullptr) {
        const SafeTensorInfo* t = m.source().find(name);
        if (!t) return "vision: the checkpoint has no " + name;
        if (t->dtype_str != "BF16") return "vision: " + name + " is not BF16";
        size_t n = 1; for (auto d : t->shape) n *= size_t(d);
        if (n != want) return "vision: " + name + " has " + std::to_string(n) + " values, expected " + std::to_string(want);
        std::vector<sycl::half> h(n);
        for (size_t i = 0; i < n; ++i) h[i] = sycl::half(bf16_at(t->data, perm ? (*perm)[i] : i));
        out = dev<sycl::half>(n);
        if (!out) return "vision: device allocation failed (" + name + ")";
        q->memcpy(out, h.data(), n * 2).wait();
        return {};
    }
    std::string host32(const Hy4FlashModel& m, const std::string& name, size_t want, std::vector<float>& out, size_t skip = 0) {
        const SafeTensorInfo* t = m.source().find(name);
        if (!t) return "vision: the checkpoint has no " + name;
        if (t->dtype_str != "BF16") return "vision: " + name + " is not BF16";
        size_t n = 1; for (auto d : t->shape) n *= size_t(d);
        if (n != want + skip) return "vision: " + name + " has " + std::to_string(n) + " values, expected " + std::to_string(want + skip);
        out.resize(want);
        for (size_t i = 0; i < want; ++i) out[i] = bf16_at(t->data, skip + i);
        return {};
    }
    std::string up32(const Hy4FlashModel& m, const std::string& name, size_t want, float*& out) {
        std::vector<float> h;
        if (auto e = host32(m, name, want, h); !e.empty()) return e;
        out = dev<float>(want);
        if (!out) return "vision: device allocation failed (" + name + ")";
        q->memcpy(out, h.data(), want * 4).wait();
        return {};
    }
};

Hy4fVision::Hy4fVision() : impl_(std::make_unique<Impl>()) {}
Hy4fVision::~Hy4fVision() { free_all(); }
bool Hy4fVision::ready() const { return impl_->open; }
uint64_t Hy4fVision::max_px() const { return impl_->max_px; }
uint64_t Hy4fVision::device_bytes() const { return impl_->bytes; }
void Hy4fVision::free_all() {
    Impl& I = *impl_;
    if (I.q) { I.q->wait(); for (void* p : I.owned) sycl::free(p, *I.q); }
    I.owned.clear(); I.blocks.clear(); I.bytes = 0; I.open = false; I.q = nullptr;
}

std::string Hy4fVision::init(sycl::queue& q, const Hy4FlashModel& m, uint64_t max_px) {
    free_all();
    Impl& I = *impl_; I.q = &q;
    I.out_dim = m.config().dim;
    I.max_px = std::max<uint64_t>(kHy4fVisMinPx, std::min<uint64_t>(max_px, kHy4fVisRefMaxPx));
    I.n_max = uint32_t(I.max_px / (kHy4fVisPatch * kHy4fVisPatch));
    std::string e;
    auto ok = [&](std::string r) { if (e.empty() && !r.empty()) e = r; return e.empty(); };
    // the patch embedding [1152, 3, 16, 16] is [1152, 768] against a (channel, y, x) patch as it is stored
    ok(I.up16(m, "vit.embeddings.patch_embedding.weight", size_t(kD) * kHy4fVisPatchDim, I.w_patch));
    ok(I.up32(m, "vit.embeddings.patch_embedding.bias", kD, I.b_patch));
    ok(I.host32(m, "vit.embeddings.position_embedding.weight", size_t(kGrid) * kGrid * kD, I.pos_table, kD));   // row 0 is the unused class token's
    I.blocks.resize(kLayers);
    for (uint32_t l = 0; l < kLayers && e.empty(); ++l) {
        const std::string p = "vit.layers." + std::to_string(l) + ".";
        auto& b = I.blocks[l];
        ok(I.up32(m, p + "input_layernorm.weight", kD, b.ln1w)); ok(I.up32(m, p + "input_layernorm.bias", kD, b.ln1b));
        ok(I.up32(m, p + "post_attention_layernorm.weight", kD, b.ln2w)); ok(I.up32(m, p + "post_attention_layernorm.bias", kD, b.ln2b));
        ok(I.up16(m, p + "self_attn.q_proj.weight", size_t(kD) * kD, b.wq)); ok(I.up32(m, p + "self_attn.q_proj.bias", kD, b.bq));
        ok(I.up16(m, p + "self_attn.k_proj.weight", size_t(kD) * kD, b.wk)); ok(I.up32(m, p + "self_attn.k_proj.bias", kD, b.bk));
        ok(I.up16(m, p + "self_attn.v_proj.weight", size_t(kD) * kD, b.wv)); ok(I.up32(m, p + "self_attn.v_proj.bias", kD, b.bv));
        ok(I.up16(m, p + "self_attn.o_proj.weight", size_t(kD) * kD, b.wo)); ok(I.up32(m, p + "self_attn.o_proj.bias", kD, b.bo));
        ok(I.up16(m, p + "mlp.dense_h_to_4h.weight", size_t(kFfn) * kD, b.w1)); ok(I.up32(m, p + "mlp.dense_h_to_4h.bias", kFfn, b.b1));
        ok(I.up16(m, p + "mlp.dense_4h_to_h.weight", size_t(kD) * kFfn, b.w2)); ok(I.up32(m, p + "mlp.dense_4h_to_h.bias", kD, b.b2));
    }
    ok(I.up32(m, "vit.perceive.norm.weight", kD, I.w_rms));
    {   // the 2 x 2 convolution [4608, 1152, 2, 2] against four gathered patches laid out (ky, kx, channel)
        std::vector<uint32_t> perm(size_t(kMergeIn) * kMergeIn);
        for (uint32_t o = 0; o < kMergeIn; ++o)
            for (uint32_t ky = 0; ky < 2; ++ky) for (uint32_t kx = 0; kx < 2; ++kx) for (uint32_t c = 0; c < kD; ++c)
                perm[size_t(o) * kMergeIn + (size_t(ky) * 2 + kx) * kD + c] = uint32_t(((size_t(o) * kD + c) * 2 + ky) * 2 + kx);
        ok(I.up16(m, "vit.perceive.proj.0.weight", size_t(kMergeIn) * kMergeIn, I.w_conv, &perm));
    }
    ok(I.up32(m, "vit.perceive.proj.0.bias", kMergeIn, I.b_conv));
    ok(I.up16(m, "vit.perceive.proj.2.weight", size_t(I.out_dim) * kMergeIn, I.w_out));
    ok(I.up32(m, "vit.perceive.proj.2.bias", I.out_dim, I.b_out));
    ok(I.host32(m, "vit.perceive.image_newline", I.out_dim, I.newline));
    if (!e.empty()) { free_all(); return e; }
    const size_t N = I.n_max, NB = N / 4;
    I.tile_attn = uint32_t(std::min<size_t>(N, 1024)); I.tile_mlp = uint32_t(std::min<size_t>(N, 2048));
    bool a = true;
    auto A16 = [&](sycl::half*& p, size_t n) { p = I.dev<sycl::half>(n); a = a && p; };
    auto A32 = [&](float*& p, size_t n) { p = I.dev<float>(n); a = a && p; };
    A16(I.px16, N * kHy4fVisPatchDim); A32(I.x32, N * kD); A32(I.C32, N * kD); A32(I.pos32, N * kD); A32(I.stat, N * 2); A16(I.xh16, N * kD);
    A16(I.q_hm, N * kD); A16(I.k_hm, N * kD); A16(I.v_hm, N * kD); A32(I.S32, size_t(I.tile_attn) * N); A16(I.P16, size_t(I.tile_attn) * N); A16(I.O16, size_t(I.tile_attn) * kHd); A16(I.att16, N * kD);
    A32(I.Cm32, size_t(I.tile_mlp) * kFfn); A16(I.mh16, size_t(I.tile_mlp) * kFfn);
    A16(I.g16, NB * kMergeIn); A32(I.Cg32, NB * kMergeIn); A16(I.gh16, NB * kMergeIn); A32(I.out32, NB * I.out_dim);
    if (!a) { free_all(); return "vision: device allocation failed (scratch for " + std::to_string(N) + " patches)"; }
    I.open = true;
    return {};
}

std::string Hy4fVision::encode(const std::vector<float>& pv, const Hy4fVisGeom& g, std::vector<float>& merged,
                               const std::vector<uint32_t>& capture, std::vector<std::vector<float>>* captured) {
    Impl& I = *impl_;
    if (!I.open) return "vision: the tower is not loaded";
    const uint32_t N = g.patches(), gh = g.grid_h, gw = g.grid_w;
    if (!N || N > I.n_max) return "vision: " + std::to_string(N) + " patches exceed the tower's scratch (" + std::to_string(I.n_max) + ")";
    if (pv.size() != size_t(N) * kHy4fVisPatchDim) return "vision: pixel_values do not match the grid";
    if ((gh | gw) & 1) return "vision: the patch grid must be even";
    sycl::queue& q = *I.q;
    try {
        // the pixels as fp16; the position grid resampled on the host (F.interpolate, bilinear, align_corners False, the
        // reference's explicit scale (size + 0.1) / 128: source = (i + 0.5) * 128 / (size + 0.1) - 0.5, clamped at 0)
        {
            std::vector<sycl::half> h(pv.size());
            for (size_t i = 0; i < pv.size(); ++i) h[i] = sycl::half(pv[i]);
            q.memcpy(I.px16, h.data(), h.size() * 2);
            std::vector<float> pos(size_t(N) * kD);
            const float sy = float(kGrid) / (float(gh) + 0.1f), sx = float(kGrid) / (float(gw) + 0.1f);
            for (uint32_t y = 0; y < gh; ++y) {
                const float fy = std::max(0.f, (float(y) + 0.5f) * sy - 0.5f); const uint32_t y0 = std::min<uint32_t>(uint32_t(fy), kGrid - 1), y1 = std::min<uint32_t>(y0 + 1, kGrid - 1); const float ly = fy - float(y0);
                for (uint32_t x = 0; x < gw; ++x) {
                    const float fx = std::max(0.f, (float(x) + 0.5f) * sx - 0.5f); const uint32_t x0 = std::min<uint32_t>(uint32_t(fx), kGrid - 1), x1 = std::min<uint32_t>(x0 + 1, kGrid - 1); const float lx = fx - float(x0);
                    const float* p00 = I.pos_table.data() + (size_t(y0) * kGrid + x0) * kD; const float* p01 = I.pos_table.data() + (size_t(y0) * kGrid + x1) * kD;
                    const float* p10 = I.pos_table.data() + (size_t(y1) * kGrid + x0) * kD; const float* p11 = I.pos_table.data() + (size_t(y1) * kGrid + x1) * kD;
                    float* o = pos.data() + (size_t(y) * gw + x) * kD;
                    for (uint32_t d = 0; d < kD; ++d) o[d] = (1.f - ly) * ((1.f - lx) * p00[d] + lx * p01[d]) + ly * ((1.f - lx) * p10[d] + lx * p11[d]);
                }
            }
            q.memcpy(I.pos32, pos.data(), pos.size() * 4).wait();
        }
        gemm_nt_f16_onednn(q, I.px16, I.w_patch, I.C32, N, kD, kHy4fVisPatchDim);
        {
            float* x = I.x32; const float* C = I.C32; const float* b = I.b_patch; const float* pos = I.pos32;
            q.parallel_for(sycl::range<2>(N, kD), [=](sycl::id<2> i) { x[i[0] * kD + i[1]] = C[i[0] * kD + i[1]] + b[i[1]] + pos[i[0] * kD + i[1]]; });
        }
        const float scale = 1.0f / std::sqrt(float(kHd));
        if (captured) captured->clear();
        for (uint32_t l = 0; l < kLayers; ++l) {
            const auto& b = I.blocks[l];
            norm_stat(q, I.x32, I.stat, N, kD, I.eps, true); norm_apply(q, I.x32, I.stat, b.ln1w, b.ln1b, I.xh16, N, kD);
            // q, k, v with their biases, head-major [heads, N, 72]
            const sycl::half* W3[3] = {b.wq, b.wk, b.wv}; const float* B3[3] = {b.bq, b.bk, b.bv}; sycl::half* D3[3] = {I.q_hm, I.k_hm, I.v_hm};
            for (int t = 0; t < 3; ++t) {
                gemm_nt_f16_onednn(q, I.xh16, W3[t], I.C32, N, kD, kD);
                const float* C = I.C32; const float* bb = B3[t]; sycl::half* d = D3[t];
                q.parallel_for(sycl::range<2>(N, kD), [=](sycl::id<2> i) {
                    const size_t h = i[1] / kHd, j = i[1] % kHd;
                    d[(h * N + i[0]) * kHd + j] = sycl::half(C[i[0] * kD + i[1]] + bb[i[1]]);
                });
            }
            for (uint32_t h = 0; h < kHeads; ++h)
                for (uint32_t r0 = 0; r0 < N; r0 += I.tile_attn) {
                    const uint32_t n = std::min(I.tile_attn, N - r0);
                    gemm_nt_f16_onednn(q, I.q_hm + (size_t(h) * N + r0) * kHd, I.k_hm + size_t(h) * N * kHd, I.S32, n, N, kHd);
                    {
                        const float* S = I.S32; sycl::half* P = I.P16;
                        // 16 lanes a row: the maximum and the sum through the sub-group, each lane writing its keys (one
                        // work-item walking a whole row took 6.2 s an image of 4,104 patches)
                        constexpr int SG = 16;
                        q.parallel_for(sycl::nd_range<2>({size_t(n), size_t(SG)}, {1, size_t(SG)}), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
                            const size_t r = it.get_global_id(0); const uint32_t lane = uint32_t(it.get_local_id(1));
                            const float* s = S + r * N; sycl::half* p = P + r * N;
                            float m = -3.0e38f;
                            for (uint32_t k = lane; k < N; k += SG) m = sycl::fmax(m, s[k] * scale);
                            m = sycl::reduce_over_group(it.get_sub_group(), m, sycl::maximum<float>());
                            float sum = 0.f;
                            for (uint32_t k = lane; k < N; k += SG) sum += sycl::exp(s[k] * scale - m);
                            sum = sycl::reduce_over_group(it.get_sub_group(), sum, sycl::plus<float>());
                            const float inv = kProbScale / sum;
                            for (uint32_t k = lane; k < N; k += SG) p[k] = sycl::half(sycl::exp(s[k] * scale - m) * inv);
                        });
                    }
                    gemm_fp16_onednn(q, I.P16, I.v_hm + size_t(h) * N * kHd, I.O16, n, kHd, N);
                    {
                        const sycl::half* O = I.O16; sycl::half* att = I.att16;
                        q.parallel_for(sycl::range<2>(n, kHd), [=](sycl::id<2> i) { att[(r0 + i[0]) * kD + h * kHd + i[1]] = sycl::half(float(O[i[0] * kHd + i[1]]) * (1.0f / kProbScale)); });
                    }
                }
            gemm_nt_f16_onednn(q, I.att16, b.wo, I.C32, N, kD, kD);
            {
                float* x = I.x32; const float* C = I.C32; const float* bb = b.bo;
                q.parallel_for(sycl::range<2>(N, kD), [=](sycl::id<2> i) { x[i[0] * kD + i[1]] += C[i[0] * kD + i[1]] + bb[i[1]]; });
            }
            norm_stat(q, I.x32, I.stat, N, kD, I.eps, true); norm_apply(q, I.x32, I.stat, b.ln2w, b.ln2b, I.xh16, N, kD);
            for (uint32_t r0 = 0; r0 < N; r0 += I.tile_mlp) {
                const uint32_t n = std::min(I.tile_mlp, N - r0);
                gemm_nt_f16_onednn(q, I.xh16 + size_t(r0) * kD, b.w1, I.Cm32, n, kFfn, kD);
                {
                    const float* C = I.Cm32; const float* bb = b.b1; sycl::half* y = I.mh16;
                    q.parallel_for(sycl::range<2>(n, kFfn), [=](sycl::id<2> i) {
                        const float v = C[i[0] * kFfn + i[1]] + bb[i[1]];
                        y[i[0] * kFfn + i[1]] = sycl::half(0.5f * v * (1.0f + sycl::erf(v * 0.70710678f)));
                    });
                }
                gemm_nt_f16_onednn(q, I.mh16, b.w2, I.C32, n, kD, kFfn);
                {
                    float* x = I.x32 + size_t(r0) * kD; const float* C = I.C32; const float* bb = b.b2;
                    q.parallel_for(sycl::range<2>(n, kD), [=](sycl::id<2> i) { x[i[0] * kD + i[1]] += C[i[0] * kD + i[1]] + bb[i[1]]; });
                }
            }
            if (captured && std::find(capture.begin(), capture.end(), l) != capture.end()) {
                captured->emplace_back(size_t(N) * kD);
                q.memcpy(captured->back().data(), I.x32, size_t(N) * kD * 4).wait();
            }
        }
        // the projector: RMSNorm, the 2 x 2 convolution as a GEMM over gathered cells, GELU, the linear map
        const uint32_t R = gh / 2, Cc = gw / 2, NB = R * Cc;
        norm_stat(q, I.x32, I.stat, N, kD, I.eps, false); norm_apply(q, I.x32, I.stat, I.w_rms, nullptr, I.xh16, N, kD);
        {
            const sycl::half* x = I.xh16; sycl::half* gdst = I.g16;
            q.parallel_for(sycl::range<2>(NB, kMergeIn), [=](sycl::id<2> i) {
                const size_t cell = i[0], r = cell / Cc, c = cell % Cc, k = i[1] / kD, d = i[1] % kD, ky = k / 2, kx = k % 2;
                gdst[cell * kMergeIn + i[1]] = x[((r * 2 + ky) * gw + c * 2 + kx) * kD + d];
            });
        }
        gemm_nt_f16_onednn(q, I.g16, I.w_conv, I.Cg32, NB, kMergeIn, kMergeIn);
        {
            const float* C = I.Cg32; const float* bb = I.b_conv; sycl::half* y = I.gh16;
            q.parallel_for(sycl::range<2>(NB, kMergeIn), [=](sycl::id<2> i) {
                const float v = C[i[0] * kMergeIn + i[1]] + bb[i[1]];
                y[i[0] * kMergeIn + i[1]] = sycl::half(0.5f * v * (1.0f + sycl::erf(v * 0.70710678f)));
            });
        }
        const uint32_t OD = I.out_dim;
        gemm_nt_f16_onednn(q, I.gh16, I.w_out, I.out32, NB, OD, kMergeIn);
        std::vector<float> cells(size_t(NB) * OD);
        q.memcpy(cells.data(), I.out32, cells.size() * 4).wait();
        std::vector<float> bo(OD);
        q.memcpy(bo.data(), I.b_out, size_t(OD) * 4).wait();
        merged.assign(size_t(g.tokens()) * OD, 0.f);
        for (uint32_t r = 0; r < R; ++r) {
            for (uint32_t c = 0; c < Cc; ++c) {
                float* o = merged.data() + (size_t(r) * (Cc + 1) + c) * OD; const float* s = cells.data() + (size_t(r) * Cc + c) * OD;
                for (uint32_t d = 0; d < OD; ++d) o[d] = s[d] + bo[d];
            }
            std::memcpy(merged.data() + (size_t(r) * (Cc + 1) + Cc) * OD, I.newline.data(), size_t(OD) * 4);
        }
    } catch (const sycl::exception& ex) {
        return std::string("vision: sycl: ") + ex.what();
    }
    return {};
}

}  // namespace ie
