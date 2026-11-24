// src/model/inkling2_vision.cpp — Inkling 2 Mini: the image processor and the vision tower (include/ie/inkling2_vision.hpp).
// The structure follows src/model/hy4flash_vision.cpp (oneDNN GEMMs with fp16 operands, fp32 accumulation and fp32
// activations between blocks; attention as S = Q K^T per head, a sub-group softmax, P V).
#include "ie/inkling2_vision.hpp"

#include "ie/ds4_vision.hpp"       // ds4_pil_resize: PIL's bicubic
#include "ie/ops.hpp"
#include "ie/safetensors.hpp"
#include "stb/stb_image.h"          // implementation instantiated in qwen4_image.cpp

#include <algorithm>
#include <cmath>
#include <cstring>

namespace ie {

std::string inkling2_image_from_rgb(const uint8_t* rgb, uint32_t h, uint32_t w, Inkling2Image& out) {
    out = {};
    if (!h || !w) return "vision: the image is empty";
    if (h > kInk2VisMaxSide || w > kInk2VisMaxSide)
        return "vision: the image is " + std::to_string(w) + " x " + std::to_string(h) + "; Inkling 2 takes sides up to " +
               std::to_string(kInk2VisMaxSide) + " pixels";
    const uint32_t B = kInk2VisBlock, R = (h + B - 1) / B, C = (w + B - 1) / B;
    out.rows = R; out.cols = C;
    out.pixels.assign(size_t(R) * C * B * B * 3, 0.f);
    // pad value -1 on the 0-255 scale; x * (1/255) then (x - 0.5) / 0.5, in fp32 as torch does
    const float rs = 0.00392156862745098f;
    auto norm = [&](float v) { return (v * rs - 0.5f) / 0.5f; };
    const float padv = norm(-1.f);
    for (uint32_t r = 0; r < R; ++r)
        for (uint32_t c = 0; c < C; ++c) {
            float* o = out.pixels.data() + (size_t(r) * C + c) * B * B * 3;
            for (uint32_t y = 0; y < B; ++y)
                for (uint32_t x = 0; x < B; ++x) {
                    const uint32_t Y = r * B + y, X = c * B + x;
                    for (uint32_t ch = 0; ch < 3; ++ch)
                        o[(size_t(y) * B + x) * 3 + ch] = (Y < h && X < w) ? norm(float(rgb[(size_t(Y) * w + X) * 3 + ch])) : padv;
                }
        }
    return {};
}

std::string inkling2_load_image(const void* bytes, size_t nbytes, Inkling2Image& out, uint32_t max_tokens, bool* resized) {
    if (resized) *resized = false;
    int sw = 0, sh = 0, comp = 0;
    uint8_t* px = stbi_load_from_memory(static_cast<const stbi_uc*>(bytes), int(nbytes), &sw, &sh, &comp, 4);
    if (!px) return std::string("vision: image decode failed: ") + stbi_failure_reason();
    std::vector<uint8_t> rgb(size_t(sw) * sh * 3);
    bool alpha = false;
    for (size_t i = 0; i < size_t(sw) * sh; ++i) alpha = alpha || px[i * 4 + 3] != 255;
    for (int y = 0; y < sh; ++y)
        for (int x = 0; x < sw; ++x) {
            const uint8_t* p = px + (size_t(y) * sw + x) * 4;
            uint8_t* o = rgb.data() + (size_t(y) * sw + x) * 3;
            if (!alpha || comp != 4) { o[0] = p[0]; o[1] = p[1]; o[2] = p[2]; continue; }
            const int bg = ((y / 8 + x / 8) % 2 == 0) ? 204 : 153, a = p[3];   // convert_to_rgb's checkerboard
            for (int ch = 0; ch < 3; ++ch) o[ch] = uint8_t((int(p[ch]) * a + bg * (255 - a) + 127) / 255);
        }
    stbi_image_free(px);
    const uint32_t B = kInk2VisBlock;
    auto toks = [&](uint32_t h, uint32_t w) { return uint64_t((h + B - 1) / B) * ((w + B - 1) / B); };
    uint32_t th = uint32_t(sh), tw = uint32_t(sw);
    if (std::max(th, tw) > kInk2VisMaxSide) {
        const double f = double(kInk2VisMaxSide) / std::max(th, tw);
        th = std::max<uint32_t>(1, uint32_t(th * f)); tw = std::max<uint32_t>(1, uint32_t(tw * f));
    }
    if (max_tokens && toks(th, tw) > max_tokens) {
        const double f = std::sqrt(double(max_tokens) / double(toks(th, tw)));
        th = std::max<uint32_t>(1, uint32_t(th * f)); tw = std::max<uint32_t>(1, uint32_t(tw * f));
        while (toks(th, tw) > max_tokens) { th = std::max<uint32_t>(1, th * 97 / 100); tw = std::max<uint32_t>(1, tw * 97 / 100); }
    }
    if (th != uint32_t(sh) || tw != uint32_t(sw)) {
        std::vector<uint8_t> small(size_t(tw) * th * 3);
        ds4_pil_resize(rgb.data(), sw, sh, small.data(), int(tw), int(th));
        if (resized) *resized = true;
        return inkling2_image_from_rgb(small.data(), th, tw, out);
    }
    return inkling2_image_from_rgb(rgb.data(), uint32_t(sh), uint32_t(sw), out);
}

namespace {
constexpr uint32_t kD = 1024, kHeads = 16, kHd = 64, kFfn = 4096, kLayers = 23, kPatchDim = 1200, kMergeIn = 4096,
                   kProjHid = 4096, kGrid = 32, kOut = 2304;
constexpr float kProbScale = 256.f;
constexpr float kTheta = 10000.f;

inline float bf16_at(const uint8_t* p, size_t i) { uint16_t b; std::memcpy(&b, p + i * 2, 2); const uint32_t u = uint32_t(b) << 16; float f; std::memcpy(&f, &u, 4); return f; }

sycl::event ln_stat(sycl::queue& q, const float* x, float* stat, uint32_t M, uint32_t N, float eps) {
    return q.parallel_for(sycl::range<1>(M), [=](sycl::id<1> i) {
        const float* r = x + size_t(i[0]) * N;
        float s = 0.f;
        for (uint32_t k = 0; k < N; ++k) s += r[k];
        const float mean = s / float(N);
        float v = 0.f;
        for (uint32_t k = 0; k < N; ++k) { const float a = r[k] - mean; v += a * a; }
        stat[i[0] * 2] = mean; stat[i[0] * 2 + 1] = 1.f / sycl::sqrt(v / float(N) + eps);
    });
}
template <class T>
sycl::event ln_apply(sycl::queue& q, const float* x, const float* stat, const float* w, const float* b, T* y, uint32_t M, uint32_t N) {
    return q.parallel_for(sycl::range<2>(M, N), [=](sycl::id<2> i) {
        y[i[0] * N + i[1]] = T((x[i[0] * N + i[1]] - stat[i[0] * 2]) * stat[i[0] * 2 + 1] * w[i[1]] + b[i[1]]);
    });
}
inline float gelu(float v) { return 0.5f * v * (1.0f + sycl::erf(v * 0.70710678118654752f)); }
}  // namespace

struct Inkling2Vision::Impl {
    sycl::queue* q = nullptr;
    SafetensorsModel st;
    std::vector<void*> owned, sowned; uint64_t bytes = 0, sbytes = 0; uint32_t max_tok = 0, n_max = 0, cap = 0, res = 0; bool open = false;
    bool scratch_dev = false;                              // dev() allocations go to the scratch list
    std::vector<float> pos_table;                          // host [32 * 32, 1024]
    std::vector<float> cls;                                // host [1024]: class + cls position
    struct Block { float *ln1w, *ln1b, *ln2w, *ln2b, *bq, *bk, *bv, *bo, *b1, *b2, *g1, *g2; sycl::half *wq, *wk, *wv, *wo, *w1, *w2; };
    std::vector<Block> blocks;
    sycl::half *w_patch = nullptr, *w_in = nullptr, *w_out = nullptr;
    float *lnp_w = nullptr, *lnp_b = nullptr;
    sycl::half *px16 = nullptr, *xh16 = nullptr, *q_hm = nullptr, *k_hm = nullptr, *v_hm = nullptr, *P16 = nullptr, *O16 = nullptr,
               *att16 = nullptr, *mh16 = nullptr, *g16 = nullptr, *gh16 = nullptr;
    float *x32 = nullptr, *C32 = nullptr, *S32 = nullptr, *Cm32 = nullptr, *stat = nullptr, *pos32 = nullptr, *Cg32 = nullptr,
          *out32 = nullptr, *rope = nullptr;
    uint32_t tile_attn = 1024, tile_mlp = 2048;

    template <class T> T* dev(size_t n) {
        T* p = sycl::malloc_device<T>(n ? n : 1, *q);
        if (p) { (scratch_dev ? sowned : owned).push_back(p); (scratch_dev ? sbytes : bytes) += n * sizeof(T); }
        return p;
    }
    void free_scratch() {
        if (q) { q->wait(); for (void* p : sowned) sycl::free(p, *q); }
        sowned.clear(); sbytes = 0; cap = 0;
    }
    bool alloc_scratch(uint32_t tokens);
    const SafeTensorInfo* find(const std::string& name, size_t want, std::string& err) {
        const SafeTensorInfo* t = st.find("model.visual." + name);
        if (!t) { err = "vision: the checkpoint has no model.visual." + name; return nullptr; }
        if (t->dtype_str != "BF16") { err = "vision: " + name + " is not BF16"; return nullptr; }
        size_t n = 1; for (auto d : t->shape) n *= size_t(d);
        if (n != want) { err = "vision: " + name + " has " + std::to_string(n) + " values, expected " + std::to_string(want); return nullptr; }
        return t;
    }
    std::string up16(const std::string& name, size_t want, sycl::half*& out) {
        std::string e;
        const SafeTensorInfo* t = find(name, want, e);
        if (!t) return e;
        std::vector<sycl::half> h(want);
        for (size_t i = 0; i < want; ++i) h[i] = sycl::half(bf16_at(t->data, i));
        out = dev<sycl::half>(want);
        if (!out) return "vision: device allocation failed (" + name + ")";
        q->memcpy(out, h.data(), want * 2).wait();
        return {};
    }
    std::string host32(const std::string& name, size_t want, std::vector<float>& out) {
        std::string e;
        const SafeTensorInfo* t = find(name, want, e);
        if (!t) return e;
        out.resize(want);
        for (size_t i = 0; i < want; ++i) out[i] = bf16_at(t->data, i);
        return {};
    }
    std::string up32(const std::string& name, size_t want, float*& out) {
        std::vector<float> h;
        if (auto e = host32(name, want, h); !e.empty()) return e;
        out = dev<float>(want);
        if (!out) return "vision: device allocation failed (" + name + ")";
        q->memcpy(out, h.data(), want * 4).wait();
        return {};
    }
};

Inkling2Vision::Inkling2Vision() : impl_(std::make_unique<Impl>()) {}
Inkling2Vision::~Inkling2Vision() { free_all(); }
bool Inkling2Vision::ready() const { return impl_->open; }
uint32_t Inkling2Vision::max_tokens() const { return impl_->max_tok; }
uint64_t Inkling2Vision::device_bytes() const { return impl_->bytes + impl_->sbytes; }
uint32_t Inkling2Vision::capacity() const { return impl_->cap; }
void Inkling2Vision::free_all() {
    Impl& I = *impl_;
    I.free_scratch();
    if (I.q) { I.q->wait(); for (void* p : I.owned) sycl::free(p, *I.q); }
    I.owned.clear(); I.blocks.clear(); I.bytes = 0; I.open = false; I.q = nullptr;
}
bool Inkling2Vision::Impl::alloc_scratch(uint32_t tokens) {
    free_scratch();
    const size_t N = 4 * size_t(tokens) + 1, NB = tokens;
    n_max = uint32_t(N);
    tile_attn = uint32_t(std::min<size_t>(N, 1024)); tile_mlp = uint32_t(std::min<size_t>(N, 2048));
    bool a = true;
    scratch_dev = true;
    auto A16 = [&](sycl::half*& p, size_t n) { p = dev<sycl::half>(n); a = a && p; };
    auto A32 = [&](float*& p, size_t n) { p = dev<float>(n); a = a && p; };
    A16(px16, N * kPatchDim); A32(x32, N * kD); A32(C32, N * kD); A32(pos32, N * kD); A32(stat, N * 2); A16(xh16, N * kD);
    A16(q_hm, N * kD); A16(k_hm, N * kD); A16(v_hm, N * kD); A32(S32, size_t(tile_attn) * N); A16(P16, size_t(tile_attn) * N);
    A16(O16, size_t(tile_attn) * kHd); A16(att16, N * kD); A32(rope, N * kHd);
    A32(Cm32, size_t(tile_mlp) * kFfn); A16(mh16, size_t(tile_mlp) * kFfn);
    A16(g16, NB * kMergeIn); A32(Cg32, NB * kProjHid); A16(gh16, NB * kProjHid); A32(out32, NB * kOut);
    scratch_dev = false;
    if (!a) { free_scratch(); return false; }
    cap = tokens;
    return true;
}
bool Inkling2Vision::grow(uint32_t tokens) {
    Impl& I = *impl_;
    if (!I.open) return false;
    tokens = std::min(tokens, I.max_tok);
    if (tokens <= I.cap) return true;
    {   // the scratch for `tokens`, less what the current one frees, must leave 256 MiB of the card free
        const uint64_t N = 4ull * tokens + 1, NB = tokens, ta = std::min<uint64_t>(N, 1024), tm = std::min<uint64_t>(N, 2048);
        const uint64_t need = N * kPatchDim * 2 + N * kD * 4 * 3 + N * 8 + N * kD * 2 * 5 + ta * N * 6 + ta * kHd * 2 + N * kHd * 4 +
                              tm * kFfn * 6 + NB * (kMergeIn * 2 + kProjHid * 6 + kOut * 4);
        const sycl::device d = I.q->get_device();
        if (d.has(sycl::aspect::ext_intel_free_memory)) {
            const uint64_t fr = d.get_info<sycl::ext::intel::info::device::free_memory>() + I.sbytes;
            if (fr < need + (256ull << 20)) return false;
        }
    }
    try { if (I.alloc_scratch(tokens)) return true; } catch (...) {}
    I.alloc_scratch(I.res);                                 // the resident scratch fitted before; it fits again
    return false;
}
void Inkling2Vision::shrink() {
    Impl& I = *impl_;
    if (I.open && I.cap > I.res) I.alloc_scratch(I.res);
}

std::string Inkling2Vision::init(sycl::queue& q, const std::string& dir, uint32_t max_tokens, uint32_t resident_tokens) {
    free_all();
    Impl& I = *impl_;
    I.q = &q;
    if (auto e = I.st.open(dir); !e.empty()) { I.q = nullptr; return "vision: " + e; }
    I.max_tok = std::max<uint32_t>(1, max_tokens);
    std::string e;
    auto ok = [&](std::string r) { if (e.empty() && !r.empty()) e = r; return e.empty(); };
    const std::string bb = "backbone.";
    ok(I.up16(bb + "embeddings.patch_embedding.weight", size_t(kD) * kPatchDim, I.w_patch));
    ok(I.host32(bb + "embeddings.position_embedding", size_t(kGrid) * kGrid * kD, I.pos_table));
    {
        std::vector<float> a, b;
        ok(I.host32(bb + "embeddings.class_embedding", kD, a));
        ok(I.host32(bb + "embeddings.cls_position_embedding", kD, b));
        if (e.empty()) { I.cls.resize(kD); for (uint32_t i = 0; i < kD; ++i) I.cls[i] = a[i] + b[i]; }
    }
    ok(I.up32(bb + "ln_pre.weight", kD, I.lnp_w)); ok(I.up32(bb + "ln_pre.bias", kD, I.lnp_b));
    I.blocks.resize(kLayers);
    for (uint32_t l = 0; l < kLayers && e.empty(); ++l) {
        const std::string p = bb + "layers." + std::to_string(l) + ".";
        auto& b = I.blocks[l];
        ok(I.up32(p + "ln_1.weight", kD, b.ln1w)); ok(I.up32(p + "ln_1.bias", kD, b.ln1b));
        ok(I.up32(p + "ln_2.weight", kD, b.ln2w)); ok(I.up32(p + "ln_2.bias", kD, b.ln2b));
        ok(I.up32(p + "ls_1.gamma", kD, b.g1)); ok(I.up32(p + "ls_2.gamma", kD, b.g2));
        ok(I.up16(p + "self_attn.q_proj.weight", size_t(kD) * kD, b.wq)); ok(I.up32(p + "self_attn.q_proj.bias", kD, b.bq));
        ok(I.up16(p + "self_attn.k_proj.weight", size_t(kD) * kD, b.wk)); ok(I.up32(p + "self_attn.k_proj.bias", kD, b.bk));
        ok(I.up16(p + "self_attn.v_proj.weight", size_t(kD) * kD, b.wv)); ok(I.up32(p + "self_attn.v_proj.bias", kD, b.bv));
        ok(I.up16(p + "self_attn.out_proj.weight", size_t(kD) * kD, b.wo)); ok(I.up32(p + "self_attn.out_proj.bias", kD, b.bo));
        ok(I.up16(p + "mlp.c_fc.weight", size_t(kFfn) * kD, b.w1)); ok(I.up32(p + "mlp.c_fc.bias", kFfn, b.b1));
        ok(I.up16(p + "mlp.c_proj.weight", size_t(kD) * kFfn, b.w2)); ok(I.up32(p + "mlp.c_proj.bias", kD, b.b2));
    }
    ok(I.up16("proj_in.weight", size_t(kProjHid) * kMergeIn, I.w_in));
    ok(I.up16("proj_out.weight", size_t(kOut) * kProjHid, I.w_out));
    if (!e.empty()) { free_all(); return e; }
    I.res = resident_tokens ? std::min(resident_tokens, I.max_tok) : I.max_tok;
    if (!I.alloc_scratch(I.res)) { free_all(); return "vision: device allocation failed (scratch for " + std::to_string(I.res) + " image tokens)"; }
    I.open = true;
    return {};
}

std::string Inkling2Vision::encode(const Inkling2Image& img, std::vector<float>& rows, std::vector<std::vector<float>>* captured) {
    Impl& I = *impl_;
    if (!I.open) return "vision: the tower is not loaded";
    const uint32_t T = img.tokens(), NP = 4 * T, N = NP + 1, R = img.rows, C = img.cols;
    if (!T || T > I.cap) return "vision: " + std::to_string(T) + " image tokens exceed the tower's scratch (" + std::to_string(I.cap) + ")";
    if (img.pixels.size() != size_t(T) * kInk2VisBlock * kInk2VisBlock * 3) return "vision: pixels do not match the grid";
    sycl::queue& q = *I.q;
    if (captured) captured->clear();
    auto cap = [&](const float* src) {
        if (!captured) return;
        captured->emplace_back(size_t(N) * kD);
        q.memcpy(captured->back().data(), src, size_t(N) * kD * 4).wait();
    };
    try {
        {   // patches [4T, 1200] in token-major, then 2 x 2 (row, col) order, each flattened (py, px, channel); position
            // embeddings, class row and RoPE angles on the host
            std::vector<sycl::half> h(size_t(NP) * kPatchDim);
            const uint32_t B = kInk2VisBlock, P = 20;
            for (uint32_t t = 0; t < T; ++t)
                for (uint32_t m = 0; m < 4; ++m) {
                    const uint32_t my = m / 2, mx = m % 2;
                    sycl::half* o = h.data() + (size_t(t) * 4 + m) * kPatchDim;
                    const float* src = img.pixels.data() + size_t(t) * B * B * 3;
                    for (uint32_t py = 0; py < P; ++py)
                        for (uint32_t px = 0; px < P; ++px)
                            for (uint32_t ch = 0; ch < 3; ++ch)
                                o[(size_t(py) * P + px) * 3 + ch] = sycl::half(src[(size_t(my * P + py) * B + mx * P + px) * 3 + ch]);
                }
            q.memcpy(I.px16, h.data(), h.size() * 2);
            std::vector<float> pos(size_t(N) * kD, 0.f), rope(size_t(N) * kHd);
            const uint32_t GH = 2 * R, GW = 2 * C;
            auto taps = [&](uint32_t i, uint32_t size, int64_t t[2], float w[2]) {
                const float src = (float(i) + 0.5f) * float(kGrid) / float(size) - 0.5f, fl = std::floor(src);
                for (int o = 0; o < 2; ++o) {
                    const int64_t raw = int64_t(fl) + o;
                    t[o] = std::min<int64_t>(std::max<int64_t>(raw, 0), kGrid - 1);
                    w[o] = std::max(0.f, 1.f - std::fabs(src - fl - float(o)));
                }
            };
            float inv[16];
            for (int j = 0; j < 16; ++j) inv[j] = 1.f / std::pow(kTheta, float(2 * j) / 32.f);
            auto angles = [&](float* r, float hp, float wp) {            // [64]: cos for pairs 0..31 then sin 0..31
                for (int p = 0; p < 32; ++p) {
                    const float f = p < 16 ? wp * inv[p] : hp * inv[p - 16];
                    r[p] = std::cos(f); r[32 + p] = std::sin(f);
                }
            };
            std::memcpy(pos.data(), I.cls.data(), kD * 4);              // row 0: the class token, its own position
            angles(rope.data(), 0.f, 0.f);
            for (uint32_t t = 0; t < T; ++t)
                for (uint32_t m = 0; m < 4; ++m) {
                    const uint32_t row = (t / C) * 2 + m / 2, col = (t % C) * 2 + m % 2, n = 1 + t * 4 + m;
                    int64_t ht[2], wt[2]; float hw[2], ww[2];
                    taps(row, GH, ht, hw); taps(col, GW, wt, ww);
                    float* o = pos.data() + size_t(n) * kD;
                    for (int a = 0; a < 2; ++a)
                        for (int b = 0; b < 2; ++b) {
                            const float wgt = hw[a] * ww[b];
                            const float* s = I.pos_table.data() + (size_t(ht[a]) * kGrid + size_t(wt[b])) * kD;
                            for (uint32_t d = 0; d < kD; ++d) o[d] += s[d] * wgt;
                        }
                    angles(rope.data() + size_t(n) * kHd, float(row + 1), float(col + 1));
                }
            q.memcpy(I.pos32, pos.data(), pos.size() * 4);
            q.memcpy(I.rope, rope.data(), rope.size() * 4).wait();
        }
        gemm_nt_f16_onednn(q, I.px16, I.w_patch, I.C32 + kD, NP, kD, kPatchDim);     // rows 1..: the patches
        {
            float* x = I.x32; const float* Cc = I.C32; const float* pos = I.pos32;
            q.parallel_for(sycl::range<2>(N, kD), [=](sycl::id<2> i) {
                x[i[0] * kD + i[1]] = (i[0] ? Cc[i[0] * kD + i[1]] : 0.f) + pos[i[0] * kD + i[1]];
            });
        }
        const float eps = 1e-5f;
        ln_stat(q, I.x32, I.stat, N, kD, eps);
        ln_apply<float>(q, I.x32, I.stat, I.lnp_w, I.lnp_b, I.x32, N, kD);
        q.wait();
        cap(I.x32);
        const float scale = 1.0f / 8.0f;
        for (uint32_t l = 0; l < kLayers; ++l) {
            const auto& b = I.blocks[l];
            ln_stat(q, I.x32, I.stat, N, kD, eps); ln_apply<sycl::half>(q, I.x32, I.stat, b.ln1w, b.ln1b, I.xh16, N, kD);
            const sycl::half* W3[3] = {b.wq, b.wk, b.wv}; const float* B3[3] = {b.bq, b.bk, b.bv}; sycl::half* D3[3] = {I.q_hm, I.k_hm, I.v_hm};
            for (int t = 0; t < 3; ++t) {
                gemm_nt_f16_onednn(q, I.xh16, W3[t], I.C32, N, kD, kD);
                const float* Cc = I.C32; const float* bias = B3[t]; sycl::half* d = D3[t]; const float* rp = I.rope; const bool rot = t < 2;
                q.parallel_for(sycl::range<2>(N, kD / 2), [=](sycl::id<2> i) {     // bias, RoPE on q / k (interleaved pairs)
                    const size_t n = i[0], c0 = 2 * i[1], h = c0 / kHd, j = c0 % kHd, p = j / 2;
                    const float x0 = Cc[n * kD + c0] + bias[c0], x1 = Cc[n * kD + c0 + 1] + bias[c0 + 1];
                    float y0 = x0, y1 = x1;
                    if (rot) {
                        const float cs = rp[n * kHd + p], sn = rp[n * kHd + 32 + p];
                        y0 = x0 * cs - x1 * sn; y1 = x1 * cs + x0 * sn;
                    }
                    d[(h * N + n) * kHd + j] = sycl::half(y0);
                    d[(h * N + n) * kHd + j + 1] = sycl::half(y1);
                });
            }
            for (uint32_t h = 0; h < kHeads; ++h)
                for (uint32_t r0 = 0; r0 < N; r0 += I.tile_attn) {
                    const uint32_t n = std::min(I.tile_attn, N - r0);
                    gemm_nt_f16_onednn(q, I.q_hm + (size_t(h) * N + r0) * kHd, I.k_hm + size_t(h) * N * kHd, I.S32, n, N, kHd);
                    {
                        const float* S = I.S32; sycl::half* Pp = I.P16;
                        constexpr int SG = 16;
                        q.parallel_for(sycl::nd_range<2>({size_t(n), size_t(SG)}, {1, size_t(SG)}), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
                            const size_t r = it.get_global_id(0); const uint32_t lane = uint32_t(it.get_local_id(1));
                            const float* s = S + r * N; sycl::half* p = Pp + r * N;
                            float m = -3.0e38f;
                            for (uint32_t k = lane; k < N; k += SG) m = sycl::fmax(m, s[k] * scale);
                            m = sycl::reduce_over_group(it.get_sub_group(), m, sycl::maximum<float>());
                            float sum = 0.f;
                            for (uint32_t k = lane; k < N; k += SG) sum += sycl::exp(s[k] * scale - m);
                            sum = sycl::reduce_over_group(it.get_sub_group(), sum, sycl::plus<float>());
                            const float invs = kProbScale / sum;
                            for (uint32_t k = lane; k < N; k += SG) p[k] = sycl::half(sycl::exp(s[k] * scale - m) * invs);
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
                float* x = I.x32; const float* Cc = I.C32; const float* bias = b.bo; const float* g = b.g1;
                q.parallel_for(sycl::range<2>(N, kD), [=](sycl::id<2> i) { x[i[0] * kD + i[1]] += g[i[1]] * (Cc[i[0] * kD + i[1]] + bias[i[1]]); });
            }
            ln_stat(q, I.x32, I.stat, N, kD, eps); ln_apply<sycl::half>(q, I.x32, I.stat, b.ln2w, b.ln2b, I.xh16, N, kD);
            for (uint32_t r0 = 0; r0 < N; r0 += I.tile_mlp) {
                const uint32_t n = std::min(I.tile_mlp, N - r0);
                gemm_nt_f16_onednn(q, I.xh16 + size_t(r0) * kD, b.w1, I.Cm32, n, kFfn, kD);
                {
                    const float* Cc = I.Cm32; const float* bias = b.b1; sycl::half* y = I.mh16;
                    q.parallel_for(sycl::range<2>(n, kFfn), [=](sycl::id<2> i) { y[i[0] * kFfn + i[1]] = sycl::half(gelu(Cc[i[0] * kFfn + i[1]] + bias[i[1]])); });
                }
                gemm_nt_f16_onednn(q, I.mh16, b.w2, I.C32, n, kD, kFfn);
                {
                    float* x = I.x32 + size_t(r0) * kD; const float* Cc = I.C32; const float* bias = b.b2; const float* g = b.g2;
                    q.parallel_for(sycl::range<2>(n, kD), [=](sycl::id<2> i) { x[i[0] * kD + i[1]] += g[i[1]] * (Cc[i[0] * kD + i[1]] + bias[i[1]]); });
                }
            }
            if (l == 0) { q.wait(); cap(I.x32); }
        }
        q.wait();
        cap(I.x32);
        {   // the patch rows (no class token), 4 a token: [T, 4096] -> proj_in -> GELU -> proj_out
            const float* x = I.x32 + kD; sycl::half* g = I.g16;
            q.parallel_for(sycl::range<1>(size_t(NP) * kD), [=](sycl::id<1> i) { g[i[0]] = sycl::half(x[i[0]]); });
        }
        gemm_nt_f16_onednn(q, I.g16, I.w_in, I.Cg32, T, kProjHid, kMergeIn);
        {
            const float* Cc = I.Cg32; sycl::half* y = I.gh16;
            q.parallel_for(sycl::range<1>(size_t(T) * kProjHid), [=](sycl::id<1> i) { y[i[0]] = sycl::half(gelu(Cc[i[0]])); });
        }
        gemm_nt_f16_onednn(q, I.gh16, I.w_out, I.out32, T, kOut, kProjHid);
        rows.resize(size_t(T) * kOut);
        q.memcpy(rows.data(), I.out32, rows.size() * 4).wait();
    } catch (const sycl::exception& ex) {
        return std::string("vision: sycl: ") + ex.what();
    }
    return {};
}

}  // namespace ie
