// src/model/ds4_vision.cpp — DeepSeek-V4-Flash-Vision-Exp vision tower (GPU) +
// the host-side geometry / block layout / image preprocessing.
//
// Spec: docs/deepseek4/70_VISION_EXP_PORT_PLAN.md. Reference: official
// inference/vision.py + image_processor.py. Gate: tests/unit/ds4_vision_gpu_test
// against tests/oracle/ds4_vision_oracle.py goldens.
//
// GEMMs ride the XMX gemm_fp16 (A[M,K]f16 × Bt[K,N]f16 → C[M,N]f32). Every C
// buffer is sized for round_up(M, 8) rows so the kernel's full-tile store
// (gemm_fp16.cpp CONTRACT) never lands outside it.

#include "ie/ds4_vision.hpp"

#include "ie/allocator.hpp"
#include "ie/ops.hpp"
#include "ie/safetensors.hpp"

#include "stb/stb_image.h"   // implementation instantiated in qwen4_image.cpp

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>

namespace ie {

// ===========================================================================
// Host geometry — verbatim ports of image_processor.py (double precision where
// Python uses float).
// ===========================================================================
uint32_t ds4_vis_grid_tokens(uint32_t n_llm_h, uint32_t n_llm_w) {
    uint32_t num = n_llm_h * (n_llm_w + 1) + 2;
    if (n_llm_h % 2 == 1) num += n_llm_w + 1;
    num += (n_llm_h + 1) / 2 * (n_llm_w + 1) % 2 * 2;
    return num;
}

namespace {

struct Grid { uint32_t n_llm_h, n_llm_w, num; };

Grid grid_tokens_px(uint32_t best_h, uint32_t best_w) {
    const uint32_t p = kDs4VisPatch, r = kDs4VisDown;
    const uint32_t n_llm_h = ((best_h / p) + r - 1) / r;
    const uint32_t n_llm_w = ((best_w / p) + r - 1) / r;
    return {n_llm_h, n_llm_w, ds4_vis_grid_tokens(n_llm_h, n_llm_w)};
}

struct Solve { uint32_t n_llm_h, n_llm_w, best_h, best_w, num; };

Solve solve_resize_ratio(double height, double width, uint32_t max_n_token) {
    const double p = kDs4VisPatch, r3 = kDs4VisDown;
    const double r = height / width;
    const double max_w_f = std::sqrt((double(max_n_token) - 2.0) / r + 0.25) - 0.5;
    const double max_h_f = max_w_f * r;
    uint32_t best_w, best_h;
    if (max_w_f < 1.0) {
        const uint32_t max_w = 1;
        uint32_t max_h = (max_n_token - 2) / (max_w + 1);
        if (max_h % 2 == 1) max_h -= 1;
        best_w = uint32_t(max_w * p * r3);
        best_h = uint32_t(max_h * p * r3);
    } else if (max_h_f < 2.0) {
        const uint32_t max_h = 2;
        const uint32_t max_w = ((max_n_token - 2) / max_h) - 1;
        best_w = uint32_t(max_w * p * r3);
        best_h = uint32_t(max_h * p * r3);
    } else {
        const uint32_t max_w = uint32_t(std::floor(max_w_f));
        uint32_t max_h = uint32_t(std::floor(max_h_f));
        if (max_h % 2 == 1) max_h -= 1;
        const double beta = std::min(max_w * p * r3 / width, max_h * p * r3 / height);
        best_w = uint32_t(std::floor(width * beta / p)) * kDs4VisPatch;
        best_h = uint32_t(std::floor(height * beta / p)) * kDs4VisPatch;
    }
    const Grid g = grid_tokens_px(best_h, best_w);
    return {g.n_llm_h, g.n_llm_w, best_h, best_w, g.num};
}

}  // namespace

Ds4VisGeom ds4_vis_plan(uint32_t width0, uint32_t height0) {
    const uint32_t p = kDs4VisPatch;
    double width = width0, height = height0;
    if (width > height * double(kDs4VisMaxWh)) width = height * double(kDs4VisMaxWh);
    if (0 < width * height && width * height < double(kDs4VisMinPx)) {
        const double ratio = std::sqrt(double(kDs4VisMinPx) / (width * height));
        width  = std::trunc(width * ratio);     // Python int()
        height = std::trunc(height * ratio);
    }
    uint32_t best_w = uint32_t(std::ceil(width / p)) * p;
    uint32_t best_h = uint32_t(std::ceil(height / p)) * p;
    // safe_resize
    const uint32_t limit = kDs4VisMaxTok - (kDs4VisPadTo - 1);
    Grid g = grid_tokens_px(best_h, best_w);
    uint32_t budget = limit;
    while (g.num > limit) {
        const Solve s = solve_resize_ratio(height, width, budget);
        g = {s.n_llm_h, s.n_llm_w, s.num};
        best_h = s.best_h; best_w = s.best_w;
        budget -= 1;
    }
    Ds4VisGeom out;
    out.best_w = best_w; out.best_h = best_h;
    out.n_vit_h = best_h / p; out.n_vit_w = best_w / p;
    out.n_llm_h = g.n_llm_h; out.n_llm_w = g.n_llm_w;
    out.plain = width0 >= kDs4VisMaxWh * height0;
    return out;
}

void ds4_vis_build_block(uint32_t n_llm_h, uint32_t n_llm_w, uint32_t start_pos,
                         std::vector<int32_t>& types, std::vector<int32_t>& perm) {
    const uint32_t compress_pad = kDs4VisPadTo - 1 - start_pos % kDs4VisPadTo;
    const uint32_t pad_h = n_llm_h % 2;
    const uint32_t rows = n_llm_h + pad_h;
    const uint32_t row_len = n_llm_w + 1;
    const uint32_t pad_last = rows / 2 * row_len % 2 * 2;
    // body types in grid order, then the (row-pair, column) interleave
    std::vector<int32_t> body(size_t(rows) * row_len, kDs4VisPad);
    for (uint32_t r = 0; r < n_llm_h; ++r) {
        for (uint32_t c = 0; c < n_llm_w; ++c) body[size_t(r) * row_len + c] = kDs4VisImage;
        body[size_t(r) * row_len + n_llm_w] = kDs4VisNewline;
    }
    std::vector<int32_t> image_idx(size_t(rows) * row_len, -1);
    for (uint32_t r = 0; r < n_llm_h; ++r)
        for (uint32_t c = 0; c < n_llm_w; ++c)
            image_idx[size_t(r) * row_len + c] = int32_t(r * n_llm_w + c);
    // order = arange(rows*row_len).view(rows/2, 2, row_len).transpose(1,2).reshape(-1)
    std::vector<uint32_t> order;
    order.reserve(size_t(rows) * row_len);
    for (uint32_t pr = 0; pr < rows / 2; ++pr)
        for (uint32_t c = 0; c < row_len; ++c)
            for (uint32_t k = 0; k < 2; ++k)
                order.push_back((pr * 2 + k) * row_len + c);
    types.clear();
    types.reserve(compress_pad + 1 + order.size() + pad_last + 1);
    for (uint32_t i = 0; i < compress_pad; ++i) types.push_back(kDs4VisPad);
    types.push_back(kDs4VisStart);
    perm.clear();
    for (uint32_t o : order) {
        types.push_back(body[o]);
        if (image_idx[o] >= 0) perm.push_back(image_idx[o]);
    }
    for (uint32_t i = 0; i < pad_last; ++i) types.push_back(kDs4VisPad);
    types.push_back(kDs4VisEnd);
}

// ===========================================================================
// Image preprocessing — PIL semantics (ImageOps.pad / Image.resize, BICUBIC
// with the reducing support widened for downscale, u8 rounding between the
// two separable passes).
// ===========================================================================
namespace {

float pil_bicubic(float x) {
    constexpr float a = -0.5f;
    x = std::fabs(x);
    if (x < 1.0f) return ((a + 2.0f) * x - (a + 3.0f)) * x * x + 1.0f;
    if (x < 2.0f) return (((x - 5.0f) * x + 8.0f) * x - 4.0f) * a;
    return 0.0f;
}

// One separable pass along `len_in` -> `len_out`; ch channels interleaved,
// `stride_px` between consecutive pixels along the pass, `n_lines` lines with
// `stride_line` apart (u8 in, u8 out).
void pil_resample_1d(const uint8_t* src, uint8_t* dst, int len_in, int len_out, int ch,
                     size_t stride_px_in, size_t stride_line_in,
                     size_t stride_px_out, size_t stride_line_out, int n_lines) {
    const double scale = double(len_in) / double(len_out);
    const double filterscale = std::max(scale, 1.0);
    const double support = 2.0 * filterscale;
    std::vector<double> w;
    for (int i = 0; i < len_out; ++i) {
        const double center = (i + 0.5) * scale;
        int xmin = int(std::floor(center - support)); if (xmin < 0) xmin = 0;
        int xmax = int(std::ceil(center + support)); if (xmax > len_in) xmax = len_in;
        w.assign(size_t(xmax - xmin), 0.0);
        double ws = 0;
        for (int x = xmin; x < xmax; ++x) {
            const double v = pil_bicubic(float((x - center + 0.5) / filterscale));
            w[size_t(x - xmin)] = v; ws += v;
        }
        for (double& v : w) v /= (ws != 0 ? ws : 1.0);
        for (int line = 0; line < n_lines; ++line)
            for (int c = 0; c < ch; ++c) {
                double acc = 0;
                for (int x = xmin; x < xmax; ++x)
                    acc += w[size_t(x - xmin)] *
                           src[size_t(line) * stride_line_in + size_t(x) * stride_px_in + c];
                const int q = int(std::lround(acc));
                dst[size_t(line) * stride_line_out + size_t(i) * stride_px_out + c] =
                    uint8_t(std::clamp(q, 0, 255));
            }
    }
}

// PIL Image.resize(BICUBIC): horizontal then vertical pass, u8 between.
void pil_resize(const uint8_t* src, int sw, int sh, uint8_t* dst, int dw, int dh) {
    std::vector<uint8_t> mid(size_t(dw) * sh * 3);
    pil_resample_1d(src, mid.data(), sw, dw, 3, 3, size_t(sw) * 3, 3, size_t(dw) * 3, sh);
    pil_resample_1d(mid.data(), dst, sh, dh, 3, size_t(dw) * 3, 3, size_t(dw) * 3, 3, dw);
}

}  // namespace

std::string ds4_load_image_mem(const void* bytes, size_t nbytes, std::vector<float>& px,
                               uint32_t& H, uint32_t& W, Ds4VisGeom& geom) {
    return ds4_load_image_planned(bytes, nbytes, [](uint32_t w, uint32_t h) { return ds4_vis_plan(w, h); }, px, H, W, geom);
}

std::string ds4_load_image_planned(const void* bytes, size_t nbytes,
                                   const std::function<Ds4VisGeom(uint32_t, uint32_t)>& plan,
                                   std::vector<float>& px, uint32_t& H, uint32_t& W, Ds4VisGeom& geom) {
    int sw = 0, sh = 0, comp = 0;
    uint8_t* rgb = stbi_load_from_memory(static_cast<const stbi_uc*>(bytes), int(nbytes),
                                         &sw, &sh, &comp, 3);
    if (!rgb) return std::string("vision: image decode failed: ") + stbi_failure_reason();
    geom = plan(uint32_t(sw), uint32_t(sh));
    if (!geom.best_w || !geom.best_h) { stbi_image_free(rgb); return "vision: the image plan is empty"; }
    const int bw = int(geom.best_w), bh = int(geom.best_h);
    std::vector<uint8_t> canvas(size_t(bw) * bh * 3, 127);
    if (geom.plain) {
        pil_resize(rgb, sw, sh, canvas.data(), bw, bh);
    } else {
        // ImageOps.contain: shrink to fit inside (bw, bh) keeping aspect
        int cw = bw, chh = bh;
        const double im_ratio = double(sw) / double(sh), dest_ratio = double(bw) / double(bh);
        if (im_ratio != dest_ratio) {
            // Python round() = half-to-even (std::nearbyint under the default mode)
            if (im_ratio > dest_ratio) {
                const int nh = int(std::nearbyint(double(sh) / double(sw) * bw));
                if (nh != bh) chh = nh;
            } else {
                const int nw = int(std::nearbyint(double(sw) / double(sh) * bh));
                if (nw != bw) cw = nw;
            }
        }
        std::vector<uint8_t> resized(size_t(cw) * chh * 3);
        pil_resize(rgb, sw, sh, resized.data(), cw, chh);
        // ImageOps.pad centring (0.5, 0.5): offset along the short axis only
        int x0 = 0, y0 = 0;
        if (cw != bw) x0 = int(std::nearbyint((bw - cw) * 0.5));
        else          y0 = int(std::nearbyint((bh - chh) * 0.5));
        for (int y = 0; y < chh; ++y)
            std::memcpy(&canvas[(size_t(y0 + y) * bw + x0) * 3], &resized[size_t(y) * cw * 3],
                        size_t(cw) * 3);
    }
    stbi_image_free(rgb);
    H = uint32_t(bh); W = uint32_t(bw);
    px.resize(size_t(3) * bh * bw);
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < bh; ++y)
            for (int x = 0; x < bw; ++x) {
                float v = (float(canvas[(size_t(y) * bw + x) * 3 + c]) / 255.0f - 0.5f) / 0.5f;
                // official casts to bf16 before patchifying
                uint32_t u; std::memcpy(&u, &v, 4);
                const uint32_t lsb = (u >> 16) & 1u;
                u = (u + 0x7FFFu + lsb) & 0xFFFF0000u;
                std::memcpy(&v, &u, 4);
                px[(size_t(c) * bh + y) * bw + x] = v;
            }
    return {};
}

std::string ds4_load_image(const std::string& path, std::vector<float>& px,
                           uint32_t& H, uint32_t& W, Ds4VisGeom& geom) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return "ds4 vision: cannot open " + path;
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return ds4_load_image_mem(bytes.data(), bytes.size(), px, H, W, geom);
}

// ===========================================================================
// Weights + GPU state
// ===========================================================================
using Ds4vFind = std::function<const SafeTensorInfo*(const std::string&)>;

// One device buffer of a tower: a weight (src set; in_d != 0 -> f16 Bt [in_d, out_d], else f32 [out_d]) or scratch.
struct Ds4vSlot { void** pp; size_t bytes; const SafeTensorInfo* src; uint32_t out_d, in_d; size_t off; };

struct Ds4Vision::Impl {
    ~Impl() {
        if (!alloc) return;
        sycl::queue& q = alloc->queue();
        for (void* p : {static_cast<void*>(h_px), static_cast<void*>(h_cs), static_cast<void*>(h_sn),
                        static_cast<void*>(h_out), hw})
            if (p) sycl::free(p, q);
    }
    SafetensorsReader st;
    Ds4VisionOptions opt;
    Ds4vFind find;
    // transient mode: the converted weights (pinned), and every device buffer's offset in the one encode block
    void* hw = nullptr; size_t hw_bytes = 0, block_bytes = 0;
    uint64_t hw_hash = 0;          // sampled hash of the staged weights, checked before every encode
    std::vector<Ds4vSlot> slots;
    void collect_slots();
    bool open = false;
    std::vector<float> ctrl[4];                    // start, pad, newline, end [4096] f32

    DeviceAllocator* alloc = nullptr;
    struct BlockW {
        float *n1 = nullptr, *n2 = nullptr;                  // [1024]
        sycl::half *qkv = nullptr, *wo = nullptr;            // Bt [1024,3072], [1024,1024]
        sycl::half *w1 = nullptr, *w2 = nullptr;             // Bt [1024,5632], [2816,1024]
        float *qkv_b = nullptr, *wo_b = nullptr;             // [3072], [1024]
    };
    sycl::half* patch = nullptr;   float* patch_b = nullptr; // Bt [588,1024], [1024]
    std::vector<BlockW> blk;
    float* norm = nullptr;                                   // [1024]
    sycl::half *a1 = nullptr, *a2 = nullptr;                 // Bt [9216,4096], [4096,4096]
    float *a1_b = nullptr, *a2_b = nullptr;                  // [4096]
    // scratch (grown on demand): Mp = round_up(N patches, 8), Bp = round_up(blocks, 8)
    uint32_t cap_n = 0, cap_b = 0;
    sycl::half* px = nullptr;      // [Mp, 588]
    float* res = nullptr;          // [Mp, 1024]
    sycl::half* xh = nullptr;      // [Mp, 1024]
    float* C = nullptr;            // [Mp, 5632]
    sycl::half* qkv = nullptr;     // [Mp, 3072]
    sycl::half* att = nullptr;     // [Mp, 1024]
    sycl::half* ffn = nullptr;     // [Mp, 2816]
    float *cs = nullptr, *sn = nullptr;   // [N, 32]
    sycl::half* h_px = nullptr;    // pinned host staging [Mp, 588]
    float *h_cs = nullptr, *h_sn = nullptr;   // pinned host staging [Mp, 32]
    sycl::half* unf = nullptr;     // [Bp, 9216]
    sycl::half* ah = nullptr;      // [Bp, 4096]
    float* Ca = nullptr;           // [Bp, 4096]
    float* h_out = nullptr;        // pinned host download [Bp, 4096]
    // GEMM attention (P4): per-head Q [Mp,64] f16, K^T Bt [64,Nc] f16, V [Nc,64] f16,
    // scores S [Mp,Nc] f32, probabilities P [Mp,Nc] f16, O [Mp,64] f32; Nc = round_up(N,128)
    uint32_t cap_nc = 0;
    sycl::half *qh = nullptr, *kT = nullptr, *vh = nullptr, *P = nullptr;
    float *S = nullptr, *O = nullptr;
};

namespace {

inline float bf16_at(const uint8_t* p, size_t i) {
    uint16_t h; std::memcpy(&h, p + i * 2, 2);
    const uint32_t u = uint32_t(h) << 16;
    float f; std::memcpy(&f, &u, 4);
    return f;
}

std::string want(const Ds4vFind& find, const std::string& name, std::vector<int64_t> shape,
                 const SafeTensorInfo*& out) {
    out = find(name);
    if (!out) return name + ": not found";
    if (out->dtype_str != "BF16") return name + ": expected BF16, found " + out->dtype_str;
    if (out->shape != shape) {
        std::string s = name + ": unexpected shape [";
        for (auto d : out->shape) s += std::to_string(d) + ",";
        return s + "]";
    }
    return {};
}

// Pinned (host-USM) staging for every H2D copy in this file.  A pageable
// std::vector source makes the Level Zero immediate command list stage the
// copy itself, page by page, at 100% host CPU — a 0.93 GB upload took >9 min
// that way (observed 2026-09-01).  Pinned staging is the engine convention
// (docs/deepseek4/20_pcie_bandwidth_measured.md: pinned 26.5 GB/s).
// Allocated ONCE at the largest tensor's size (aligner.w1, 75.5 MB): growing it
// by free+realloc mid-upload left the compute queue dead a few kernels later
// (stale translation after a host-USM free is the working theory, 2026-09-01).
struct Stage {
    sycl::queue& q;
    void*  p = nullptr;
    size_t cap = 0;
    Stage(sycl::queue& qq, size_t bytes) : q(qq) {
        p = sycl::malloc_host(bytes, q);
        cap = p ? bytes : 0;
    }
    ~Stage() { if (p) sycl::free(p, q); }
    void* get(size_t bytes) { return bytes <= cap ? p : nullptr; }
};

// BF16 [out, in] -> device f16 Bt [in, out]
sycl::half* ds4v_upload_bt(DeviceAllocator& a, Stage& st, const SafeTensorInfo& t, uint32_t out_d, uint32_t in_d) {
    const size_t n = size_t(in_d) * out_d;
    auto* bt = static_cast<sycl::half*>(st.get(n * 2));
    if (!bt) return nullptr;
    for (uint32_t o = 0; o < out_d; ++o)
        for (uint32_t k = 0; k < in_d; ++k)
            bt[size_t(k) * out_d + o] = sycl::half(bf16_at(t.data, size_t(o) * in_d + k));
    auto* d = static_cast<sycl::half*>(a.malloc(n * 2));
    if (d) a.queue().memcpy(d, bt, n * 2).wait();
    return d;
}

float* ds4v_upload_f32(DeviceAllocator& a, Stage& st, const SafeTensorInfo& t) {
    const size_t n = t.numel();
    auto* h = static_cast<float*>(st.get(n * 4));
    if (!h) return nullptr;
    for (size_t i = 0; i < n; ++i) h[i] = bf16_at(t.data, i);
    auto* d = static_cast<float*>(a.malloc(n * 4));
    if (d) a.queue().memcpy(d, h, n * 4).wait();
    return d;
}

// ---- kernels ---------------------------------------------------------------
// Every kernel-launching function here carries the `ds4v_` prefix ON PURPOSE.
// SYCL derives the device kernel name from the enclosing function's mangled
// name, and `ie::(anonymous)::bias_act_f16` / `bias_add_res` / `attn_full`
// already exist in src/ops/qwen4_vision_gpu.cpp.  Two device images with the
// same kernel name are an ODR violation the runtime resolves silently to ONE
// of them: launching this file's bias kernel executed the wrong code and
// killed the queue (UR_RESULT_ERROR_DEVICE_LOST, GPU CAT error at address 0;
// 2026-09-01, bisected with the trace marks below and the probe seam in
// tests/unit/ds4_vision_gpu_test.cpp).  Do not reuse names across TUs.

// RMSNorm rows of f32 src -> f16 dst: y = w * x * rsqrt(mean(x^2) + eps).
sycl::event ds4v_rms_norm_f16(sycl::queue& q, const float* src, const float* w, sycl::half* dst,
                         uint32_t M, uint32_t N, float eps) {
    constexpr uint32_t WG = 256;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> red(WG, h);
        h.parallel_for(sycl::nd_range<1>(size_t(M) * WG, WG), [=](sycl::nd_item<1> it) {
            const uint32_t row = uint32_t(it.get_group(0));
            const uint32_t l = uint32_t(it.get_local_id(0));
            const float* x = src + size_t(row) * N;
            float s = 0;
            for (uint32_t j = l; j < N; j += WG) s += x[j] * x[j];
            red[l] = s;
            it.barrier(sycl::access::fence_space::local_space);
            for (uint32_t st = WG / 2; st; st >>= 1) {
                if (l < st) red[l] += red[l + st];
                it.barrier(sycl::access::fence_space::local_space);
            }
            const float inv = sycl::rsqrt(red[0] / float(N) + eps);
            sycl::half* y = dst + size_t(row) * N;
            for (uint32_t j = l; j < N; j += WG) y[j] = sycl::half(w[j] * (x[j] * inv));
        });
    });
}

// y[i,j] = act(C[i,j] + b[j]) as f16. act 0 none, 1 gelu_erf. b may be null.
sycl::event ds4v_bias_act_f16(sycl::queue& q, const float* C, const float* b, sycl::half* y,
                         uint32_t M, uint32_t N, int act) {
    return q.parallel_for(sycl::range<1>(size_t(M) * N), [=](sycl::id<1> i) {
        float v = C[i] + (b ? b[uint32_t(i % N)] : 0.f);
        if (act == 1) v = 0.5f * v * (1.0f + sycl::erf(v * 0.70710678118654752f));
        y[i] = sycl::half(v);
    });
}

// res[i,j] += C[i,j] + b[j]   (b may be null)
sycl::event ds4v_bias_add_res(sycl::queue& q, const float* C, const float* b, float* res,
                         uint32_t M, uint32_t N) {
    return q.parallel_for(sycl::range<1>(size_t(M) * N), [=](sycl::id<1> i) {
        res[i] += C[i] + (b ? b[uint32_t(i % N)] : 0.f);
    });
}

// res[i,j] = C[i,j] + b[j]
sycl::event ds4v_bias_set(sycl::queue& q, const float* C, const float* b, float* res,
                     uint32_t M, uint32_t N) {
    return q.parallel_for(sycl::range<1>(size_t(M) * N), [=](sycl::id<1> i) {
        res[i] = C[i] + b[uint32_t(i % N)];
    });
}

// SwiGLU: C [M, 2F] = [gate ‖ up] -> y [M, F] f16 = silu(gate) * up
sycl::event ds4v_swiglu_f16(sycl::queue& q, const float* C, sycl::half* y, uint32_t M, uint32_t F) {
    return q.parallel_for(sycl::range<1>(size_t(M) * F), [=](sycl::id<1> i) {
        const uint32_t r = uint32_t(i / F), j = uint32_t(i % F);
        const float g = C[size_t(r) * 2 * F + j], u = C[size_t(r) * 2 * F + F + j];
        y[i] = sycl::half(g / (1.0f + sycl::native::exp(-g)) * u);
    });
}

// 2-D RoPE over q and k inside packed qkv [N, 3, 16, 64]: pair (r, r+32),
// r < 16 uses the h-angle, r >= 16 the w-angle (cs/sn [N, 32]).
sycl::event ds4v_rope2d(sycl::queue& q, sycl::half* qkv, const float* cs, const float* sn, uint32_t N) {
    const uint32_t items = N * 2 * kDs4VisHeads * kDs4VisRot;
    return q.parallel_for(sycl::range<1>(items), [=](sycl::id<1> id) {
        uint32_t i = uint32_t(id);
        const uint32_t r = i % kDs4VisRot;      i /= kDs4VisRot;
        const uint32_t hd = i % kDs4VisHeads;   i /= kDs4VisHeads;
        const uint32_t qk = i % 2;              i /= 2;
        const uint32_t tok = i;
        sycl::half* v = qkv + ((size_t(tok) * 3 + qk) * kDs4VisHeads + hd) * kDs4VisHeadDim;
        const float c = cs[size_t(tok) * kDs4VisRot + r];
        const float s = sn[size_t(tok) * kDs4VisRot + r];
        const float a = float(v[r]), b2 = float(v[r + kDs4VisRot]);
        v[r] = sycl::half(a * c - b2 * s);
        v[r + kDs4VisRot] = sycl::half(b2 * c + a * s);
    });
}

// Full bidirectional attention over one image: one work-group per (token, head).
sycl::event ds4v_attn_full(sycl::queue& q, const sycl::half* qkv, sycl::half* out, uint32_t N) {
    constexpr uint32_t WG = 128;
    const float scale = 1.0f / std::sqrt(float(kDs4VisHeadDim));
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sc(N, h);
        sycl::local_accessor<float, 1> qv(kDs4VisHeadDim, h);
        sycl::local_accessor<float, 1> red(WG, h);
        h.parallel_for(sycl::nd_range<2>({size_t(N), size_t(kDs4VisHeads) * WG}, {1, WG}),
                       [=](sycl::nd_item<2> it) {
            const uint32_t tok = uint32_t(it.get_global_id(0));
            const uint32_t hd = uint32_t(it.get_group(1));
            const uint32_t l = uint32_t(it.get_local_id(1));
            auto row = [&](uint32_t t2, uint32_t which) {
                return qkv + ((size_t(t2) * 3 + which) * kDs4VisHeads + hd) * kDs4VisHeadDim;
            };
            const sycl::half* qr = row(tok, 0);
            for (uint32_t d = l; d < kDs4VisHeadDim; d += WG) qv[d] = float(qr[d]);
            it.barrier(sycl::access::fence_space::local_space);
            for (uint32_t j = l; j < N; j += WG) {
                const sycl::half* kr = row(j, 1);
                float d2 = 0;
                for (uint32_t d = 0; d < kDs4VisHeadDim; ++d) d2 += qv[d] * float(kr[d]);
                sc[j] = d2 * scale;
            }
            it.barrier(sycl::access::fence_space::local_space);
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
            float sm = 0;
            for (uint32_t j = l; j < N; j += WG) {
                const float e = sycl::native::exp(sc[j] - mx);
                sc[j] = e; sm += e;
            }
            red[l] = sm;
            it.barrier(sycl::access::fence_space::local_space);
            for (uint32_t st = WG / 2; st; st >>= 1) {
                if (l < st) red[l] += red[l + st];
                it.barrier(sycl::access::fence_space::local_space);
            }
            const float inv = 1.0f / red[0];
            it.barrier(sycl::access::fence_space::local_space);
            for (uint32_t d = l; d < kDs4VisHeadDim; d += WG) {
                float acc = 0;
                for (uint32_t j = 0; j < N; ++j) acc += sc[j] * float(row(j, 2)[d]);
                out[size_t(tok) * kDs4VisDim + hd * kDs4VisHeadDim + d] = sycl::half(acc * inv);
            }
        });
    });
}

// ---- GEMM attention (P4) ---------------------------------------------------
// The naive kernel above (one work-group per (token, head), scores in SLM)
// was 1.55 s of a 1.64 s encode at 2562 patches.  This route runs each head
// as two XMX GEMMs: S = Q_h K_h^T (K^T staged as Bt [64, Nc]), row softmax to
// f16 P, O = P V_h (V_h as Bt [Nc, 64]).  Nc = round_up(N, 128) keeps the
// GEMM's column store tile-aligned (gemm_fp16.cpp CONTRACT); padded key
// columns hold zero K rows and get P = 0, padded V rows are zero.
// Per head: qh[t][d] = qkv[t][0][h][d]; kT[d][t] = qkv[t][1][h][d] (0 for t >= N);
// vh[t][d] = qkv[t][2][h][d] (0 for t >= N).
sycl::event ds4v_attn_gather(sycl::queue& q, const sycl::half* qkv, sycl::half* qh, sycl::half* kT,
                             sycl::half* vh, uint32_t N, uint32_t Nc, uint32_t head) {
    constexpr uint32_t HD = kDs4VisHeadDim, NH = kDs4VisHeads;
    return q.parallel_for(sycl::range<1>(size_t(Nc) * HD), [=](sycl::id<1> id) {
        const uint32_t i = uint32_t(id), t = i / HD, d = i % HD;
        const sycl::half zero(0.f);
        if (t < N) {
            const sycl::half* base = qkv + (size_t(t) * 3 * NH + head) * HD;
            qh[i] = base[d];
            kT[size_t(d) * Nc + t] = base[size_t(NH) * HD + d];
            vh[i] = base[size_t(2 * NH) * HD + d];
        } else {
            kT[size_t(d) * Nc + t] = zero;
            vh[i] = zero;
        }
    });
}

// P[t, k] = softmax_k(scale * S[t, k]) over k < N, 0 for k >= N; one WG per row.
sycl::event ds4v_softmax_rows(sycl::queue& q, const float* S, sycl::half* P, uint32_t rows, uint32_t N, uint32_t Nc,
                              float scale, float pscale) {
    constexpr uint32_t WG = 256;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> red(WG, h);
        h.parallel_for(sycl::nd_range<1>(size_t(rows) * WG, WG), [=](sycl::nd_item<1> it) {
            const uint32_t row = uint32_t(it.get_group(0)), l = uint32_t(it.get_local_id(0));
            const float* s = S + size_t(row) * Nc;
            sycl::half* p = P + size_t(row) * Nc;
            float mx = -1e30f;
            for (uint32_t k = l; k < N; k += WG) mx = sycl::fmax(mx, s[k] * scale);
            red[l] = mx;
            it.barrier(sycl::access::fence_space::local_space);
            for (uint32_t st = WG / 2; st; st >>= 1) {
                if (l < st) red[l] = sycl::fmax(red[l], red[l + st]);
                it.barrier(sycl::access::fence_space::local_space);
            }
            mx = red[0];
            it.barrier(sycl::access::fence_space::local_space);
            float sm = 0.f;
            for (uint32_t k = l; k < N; k += WG) sm += sycl::native::exp(s[k] * scale - mx);
            red[l] = sm;
            it.barrier(sycl::access::fence_space::local_space);
            for (uint32_t st = WG / 2; st; st >>= 1) {
                if (l < st) red[l] += red[l + st];
                it.barrier(sycl::access::fence_space::local_space);
            }
            const float inv = pscale / red[0];
            for (uint32_t k = l; k < Nc; k += WG)
                p[k] = k < N ? sycl::half(sycl::native::exp(s[k] * scale - mx) * inv) : sycl::half(0.f);
        });
    });
}

// att[t, head*64 + d] = f16(O[t, d])
sycl::event ds4v_attn_scatter(sycl::queue& q, const float* O, sycl::half* att, uint32_t N, uint32_t head, float mul) {
    constexpr uint32_t HD = kDs4VisHeadDim, D = kDs4VisDim;
    return q.parallel_for(sycl::range<1>(size_t(N) * HD), [=](sycl::id<1> id) {
        const uint32_t i = uint32_t(id), t = i / HD, d = i % HD;
        att[size_t(t) * D + head * HD + d] = sycl::half(O[i] * mul);      // mul == 1 leaves the value bit for bit
    });
}

// Aligner input: F.unfold(3, stride 3) over the [1024, n_h, n_w] feature map
// zero-padded to multiples of 3. dst [bh*bw, 9216], feature = c*9 + kh*3 + kw.
sycl::event ds4v_unfold3(sycl::queue& q, const sycl::half* src, sycl::half* dst,
                    uint32_t n_h, uint32_t n_w, uint32_t bh, uint32_t bw) {
    constexpr uint32_t R = kDs4VisDown, D = kDs4VisDim, F = D * R * R;
    return q.parallel_for(sycl::range<1>(size_t(bh) * bw * F), [=](sycl::id<1> id) {
        const size_t i = id;
        const uint32_t f = uint32_t(i % F), blk = uint32_t(i / F);
        const uint32_t c = f / (R * R), kh = (f / R) % R, kw = f % R;
        const uint32_t bi = blk / bw, bj = blk % bw;
        const uint32_t r = bi * R + kh, cc = bj * R + kw;
        dst[i] = (r < n_h && cc < n_w) ? src[(size_t(r) * n_w + cc) * D + c] : sycl::half(0.f);
    });
}

inline uint32_t round8(uint32_t n) { return (n + 7u) & ~7u; }

}  // namespace

namespace ds4vis_detail {
void bias_act_f16_probe(void* queue, const float* C, const float* b, void* y_half,
                        uint32_t M, uint32_t N, int act) {
    ds4v_bias_act_f16(*static_cast<sycl::queue*>(queue), C, b, static_cast<sycl::half*>(y_half), M, N, act).wait();
}
}  // namespace ds4vis_detail

Ds4Vision::Ds4Vision() : impl_(std::make_unique<Impl>()) {}
Ds4Vision::~Ds4Vision() = default;
bool Ds4Vision::loaded() const { return impl_->open; }

namespace {
// shape-check everything an upload will read, while a failure is still a load error; fill the control rows
std::string ds4v_check(const Ds4vFind& find, const Ds4VisionOptions& o, std::vector<float> ctrl[4]) {
    const char* names[4] = {"image_start", "image_pad", "image_newline", "image_end"};
    for (int k = 0; k < 4; ++k) {
        if (k == 1 && !o.pad_row) continue;
        const SafeTensorInfo* t;
        if (auto e = want(find, names[k], {int64_t(o.out_dim)}, t); !e.empty()) return e;
        ctrl[k].resize(o.out_dim);
        for (uint32_t i = 0; i < o.out_dim; ++i) ctrl[k][i] = bf16_at(t->data, i);
    }
    const SafeTensorInfo* t;
    std::string e;
    const int64_t D = kDs4VisDim, Fd = kDs4VisFfn, O = o.out_dim;
    if (e = want(find, "vision.patch_embed.proj.weight", {D, 3 * 14 * 14}, t); !e.empty()) return e;
    if (e = want(find, "vision.patch_embed.proj.bias", {D}, t); !e.empty()) return e;
    if (e = want(find, "vision.norm.weight", {D}, t); !e.empty()) return e;
    if (e = want(find, "aligner.w1.weight", {O, D * 9}, t); !e.empty()) return e;
    if (e = want(find, "aligner.w1.bias", {O}, t); !e.empty()) return e;
    if (e = want(find, "aligner.w2.weight", {O, O}, t); !e.empty()) return e;
    if (e = want(find, "aligner.w2.bias", {O}, t); !e.empty()) return e;
    for (uint32_t L = 0; L < kDs4VisDepth; ++L) {
        const std::string p = "vision.blocks." + std::to_string(L) + ".";
        if (e = want(find, p + "norm1.weight", {D}, t); !e.empty()) return e;
        if (e = want(find, p + "norm2.weight", {D}, t); !e.empty()) return e;
        if (e = want(find, p + "attn.wqkv.weight", {3 * D, D}, t); !e.empty()) return e;
        if (e = want(find, p + "attn.wqkv.bias", {3 * D}, t); !e.empty()) return e;
        if (e = want(find, p + "attn.wo.weight", {D, D}, t); !e.empty()) return e;
        if (e = want(find, p + "attn.wo.bias", {D}, t); !e.empty()) return e;
        if (e = want(find, p + "mlp.w1.weight", {2 * Fd, D}, t); !e.empty()) return e;
        if (e = want(find, p + "mlp.w2.weight", {D, Fd}, t); !e.empty()) return e;
    }
    return {};
}
}  // namespace

std::string Ds4Vision::load(const std::string& path) {
    Impl& I = *impl_;
    if (auto e = I.st.open(path); !e.empty()) return "ds4 vision: " + e;
    if (I.st.tensors().size() != 267)
        return "ds4 vision: sidecar has " + std::to_string(I.st.tensors().size()) + " tensors, expected 267";
    I.opt = Ds4VisionOptions{};
    I.opt.max_patches = kMaxPatches; I.opt.max_blocks = kMaxBlocks;
    I.find = [&I](const std::string& n) { return I.st.find(n); };
    if (auto e = ds4v_check(I.find, I.opt, I.ctrl); !e.empty()) return "ds4 vision: " + e;
    I.open = true;
    return {};
}

std::string Ds4Vision::load_from(Ds4vFind find, const Ds4VisionOptions& opt) {
    Impl& I = *impl_;
    if (I.open) return "ds4 vision: already loaded";
    if (!opt.out_dim || !opt.max_patches || !opt.max_blocks) return "ds4 vision: empty options";
    I.opt = opt; I.find = std::move(find);
    if (auto e = ds4v_check(I.find, I.opt, I.ctrl); !e.empty()) return "ds4 vision: " + e;
    I.open = true;
    return {};
}

uint32_t Ds4Vision::out_dim() const { return impl_->opt.out_dim; }

std::string Ds4Vision::assemble_rows(const std::vector<float>& aligned, const std::vector<int32_t>& types,
                                     const std::vector<int32_t>& perm, std::vector<float>& rows) const {
    const Impl& I = *impl_;
    if (!I.open) return "ds4 vision: not loaded";
    const uint32_t OD = I.opt.out_dim;
    const size_t n_img = aligned.size() / OD;
    rows.assign(types.size() * OD, 0.f);
    size_t k = 0;
    for (size_t i = 0; i < types.size(); ++i) {
        float* dst = &rows[i * OD];
        switch (types[i]) {
            case kDs4VisStart:   std::memcpy(dst, I.ctrl[0].data(), OD * 4); break;
            case kDs4VisPad:     std::memcpy(dst, I.ctrl[1].data(), OD * 4); break;
            case kDs4VisNewline: std::memcpy(dst, I.ctrl[2].data(), OD * 4); break;
            case kDs4VisEnd:     std::memcpy(dst, I.ctrl[3].data(), OD * 4); break;
            case kDs4VisImage: {
                if (k >= perm.size()) return "ds4 vision: more IMAGE rows than perm entries";
                const int32_t src = perm[k++];
                if (src < 0 || size_t(src) >= n_img) return "ds4 vision: perm index out of range";
                std::memcpy(dst, &aligned[size_t(src) * OD], OD * 4);
                break;
            }
            default: return "ds4 vision: bad row type " + std::to_string(types[i]);
        }
    }
    if (k != perm.size()) return "ds4 vision: IMAGE rows (" + std::to_string(k) + ") != perm (" +
                                 std::to_string(perm.size()) + ")";
    return {};
}

uint64_t Ds4Vision::device_bytes() {
    const uint64_t D = kDs4VisDim, Fd = kDs4VisFfn, O = kDs4VisOut, PD = 3 * kDs4VisPatch * kDs4VisPatch;
    const uint64_t weights = PD * D * 2 + D * 4 +
        uint64_t(kDs4VisDepth) * (2 * D * 4 + 3 * D * D * 2 + 3 * D * 4 + D * D * 2 + D * 4 + 2 * Fd * D * 2 + D * Fd * 2) +
        D * 4 + uint64_t(D * 9) * O * 2 + O * 4 + O * O * 2 + O * 4;
    const uint64_t Mp = kMaxPatches, Bp = kMaxBlocks;
    const uint64_t Nc = (Mp + 127u) & ~127u, HD = kDs4VisHeadDim;
    const uint64_t scratch = Mp * PD * 2 + Mp * D * 4 + Mp * D * 2 + Mp * 2 * Fd * 4 + Mp * 3 * D * 2 +
        Mp * D * 2 + Mp * Fd * 2 + 2 * Mp * kDs4VisRot * 4 + Bp * uint64_t(D * 9) * 2 + Bp * O * 2 + Bp * O * 4 +
        Mp * HD * 2 + HD * Nc * 2 + Nc * HD * 2 + Mp * Nc * 4 + Mp * Nc * 2 + Mp * HD * 4;
    return weights + scratch;
}

std::string Ds4Vision::upload(DeviceAllocator& alloc) {
    Impl& I = *impl_;
    if (!I.open) return "ds4 vision: not loaded";
    if (I.opt.transient) return stage_host(alloc);
    if (I.alloc) return I.alloc == &alloc ? std::string{} : "ds4 vision: allocator changed between calls";
    const uint32_t PD = 3 * kDs4VisPatch * kDs4VisPatch;
    const uint32_t D = kDs4VisDim, Fd = kDs4VisFfn, O = I.opt.out_dim, UF = D * 9;
    sycl::queue& q = alloc.queue();
    static const bool trace = std::getenv("IE_DS4_VIS_TRACE") != nullptr;
    const auto t_start = std::chrono::steady_clock::now();
    auto mark = [&](const char* what) {
        if (!trace) return;
        q.wait();
        std::fprintf(stderr, "[ds4-vis] %-14s %.3fs\n", what,
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count());
    };
    {
        I.alloc = &alloc;
        Stage st(q, size_t(UF) * O * 2);   // aligner.w1 is the largest upload
        auto T = [&](const std::string& n) { return I.find(n); };
        bool ok = true;
        I.patch   = ds4v_upload_bt(alloc, st, *T("vision.patch_embed.proj.weight"), D, PD);  ok &= I.patch != nullptr;
        I.patch_b = ds4v_upload_f32(alloc, st, *T("vision.patch_embed.proj.bias"));          ok &= I.patch_b != nullptr;
        I.blk.resize(kDs4VisDepth);
        for (uint32_t L = 0; L < kDs4VisDepth; ++L) {
            const std::string pre = "vision.blocks." + std::to_string(L) + ".";
            auto& b = I.blk[L];
            b.n1    = ds4v_upload_f32(alloc, st, *T(pre + "norm1.weight"));             ok &= b.n1 != nullptr;
            b.n2    = ds4v_upload_f32(alloc, st, *T(pre + "norm2.weight"));             ok &= b.n2 != nullptr;
            b.qkv   = ds4v_upload_bt(alloc, st, *T(pre + "attn.wqkv.weight"), 3 * D, D); ok &= b.qkv != nullptr;
            b.qkv_b = ds4v_upload_f32(alloc, st, *T(pre + "attn.wqkv.bias"));           ok &= b.qkv_b != nullptr;
            b.wo    = ds4v_upload_bt(alloc, st, *T(pre + "attn.wo.weight"), D, D);      ok &= b.wo != nullptr;
            b.wo_b  = ds4v_upload_f32(alloc, st, *T(pre + "attn.wo.bias"));             ok &= b.wo_b != nullptr;
            b.w1    = ds4v_upload_bt(alloc, st, *T(pre + "mlp.w1.weight"), 2 * Fd, D);  ok &= b.w1 != nullptr;
            b.w2    = ds4v_upload_bt(alloc, st, *T(pre + "mlp.w2.weight"), D, Fd);      ok &= b.w2 != nullptr;
        }
        I.norm = ds4v_upload_f32(alloc, st, *T("vision.norm.weight"));                  ok &= I.norm != nullptr;
        I.a1   = ds4v_upload_bt(alloc, st, *T("aligner.w1.weight"), O, UF);             ok &= I.a1 != nullptr;
        I.a1_b = ds4v_upload_f32(alloc, st, *T("aligner.w1.bias"));                     ok &= I.a1_b != nullptr;
        I.a2   = ds4v_upload_bt(alloc, st, *T("aligner.w2.weight"), O, O);              ok &= I.a2 != nullptr;
        I.a2_b = ds4v_upload_f32(alloc, st, *T("aligner.w2.bias"));                     ok &= I.a2_b != nullptr;
        if (!ok) return "ds4 vision: device/pinned allocation failed during weight upload";
        if (!I.st.tensors().empty()) I.st.close();
        mark("upload");
    }
    {   // scratch, once, for the largest block the token budget allows
        const uint32_t Mp = I.opt.max_patches, Bp = I.opt.max_blocks;
        if (I.cap_n < Mp) {
        auto am = [&](size_t bytes) { return alloc.malloc(bytes); };
        I.px  = static_cast<sycl::half*>(am(size_t(Mp) * PD * 2));
        I.res = static_cast<float*>(am(size_t(Mp) * D * 4));
        I.xh  = static_cast<sycl::half*>(am(size_t(Mp) * D * 2));
        I.C   = static_cast<float*>(am(size_t(Mp) * 2 * Fd * 4));
        I.qkv = static_cast<sycl::half*>(am(size_t(Mp) * 3 * D * 2));
        I.att = static_cast<sycl::half*>(am(size_t(Mp) * D * 2));
        I.ffn = static_cast<sycl::half*>(am(size_t(Mp) * Fd * 2));
        I.cs  = static_cast<float*>(am(size_t(Mp) * kDs4VisRot * 4));
        I.sn  = static_cast<float*>(am(size_t(Mp) * kDs4VisRot * 4));
        if (I.h_px) sycl::free(I.h_px, q);
        if (I.h_cs) sycl::free(I.h_cs, q);
        if (I.h_sn) sycl::free(I.h_sn, q);
        I.h_px = sycl::malloc_host<sycl::half>(size_t(Mp) * PD, q);
        I.h_cs = sycl::malloc_host<float>(size_t(Mp) * kDs4VisRot, q);
        I.h_sn = sycl::malloc_host<float>(size_t(Mp) * kDs4VisRot, q);
        if (!I.px || !I.res || !I.xh || !I.C || !I.qkv || !I.att || !I.ffn || !I.cs || !I.sn ||
            !I.h_px || !I.h_cs || !I.h_sn)
            return "ds4 vision: scratch allocation failed";
        I.cap_n = Mp;
    }
        {   // GEMM-attention scratch for the same cap
            const uint32_t Nc = (Mp + 127u) & ~127u;
            I.qh = static_cast<sycl::half*>(alloc.malloc(size_t(Mp) * kDs4VisHeadDim * 2));
            I.kT = static_cast<sycl::half*>(alloc.malloc(size_t(kDs4VisHeadDim) * Nc * 2));
            I.vh = static_cast<sycl::half*>(alloc.malloc(size_t(Nc) * kDs4VisHeadDim * 2));
            const uint32_t Tq = I.opt.attn_tile ? std::min(I.opt.attn_tile, Mp) : Mp;   // query rows per GEMM
            I.S  = static_cast<float*>(alloc.malloc(size_t(Tq) * Nc * 4));
            I.P  = static_cast<sycl::half*>(alloc.malloc(size_t(Tq) * Nc * 2));
            I.O  = static_cast<float*>(alloc.malloc(size_t(Mp) * kDs4VisHeadDim * 4));
            if (!I.qh || !I.kT || !I.vh || !I.S || !I.P || !I.O) return "ds4 vision: attention scratch allocation failed";
            I.cap_nc = Nc;
        }
        if (I.cap_b < Bp) {
        auto am = [&](size_t bytes) { return alloc.malloc(bytes); };
        I.unf = static_cast<sycl::half*>(am(size_t(Bp) * UF * 2));
        I.ah  = static_cast<sycl::half*>(am(size_t(Bp) * O * 2));
        I.Ca  = static_cast<float*>(am(size_t(Bp) * O * 4));
        if (I.h_out) sycl::free(I.h_out, q);
        I.h_out = sycl::malloc_host<float>(size_t(Bp) * O, q);
        if (!I.unf || !I.ah || !I.Ca || !I.h_out) return "ds4 vision: aligner scratch allocation failed";
        I.cap_b = Bp;
    }

    }
    return {};
}

namespace {
// A cheap fingerprint of the pinned tower weights: 4 MiB sampled across the buffer. A
// full hash of 1.5 GiB would cost ~0.4 s per image; this costs ~2 ms and still catches
// anything that overwrites a region (docs/deepseek41/101).
uint64_t ds4v_sample_hash(const void* p, size_t bytes) {
    const auto* b = static_cast<const uint8_t*>(p);
    const size_t chunk = 4096, want = 4u << 20;
    const size_t chunks = bytes / chunk, take = std::min(chunks, want / chunk);
    if (!chunks) return 0;
    const size_t stride = std::max<size_t>(1, chunks / std::max<size_t>(1, take));
    uint64_t h = 0xCBF29CE484222325ull;
    for (size_t c = 0; c < chunks; c += stride)
        for (size_t i = 0; i < chunk; i += 64) { h ^= b[c * chunk + i]; h *= 0x100000001B3ull; }
    return h;
}
}  // namespace

std::string Ds4Vision::encode_gpu(DeviceAllocator& alloc, const float* img, uint32_t H, uint32_t W,
                                  std::vector<float>& aligned) {
    Impl& I = *impl_;
    if (!I.open) return "ds4 vision: not loaded";
    const uint32_t p = kDs4VisPatch;
    if (!H || !W || H % p || W % p) return "ds4 vision: H/W must be multiples of 14";
    const uint32_t n_h = H / p, n_w = W / p, N = n_h * n_w;
    const uint32_t bh = (n_h + kDs4VisDown - 1) / kDs4VisDown, bw = (n_w + kDs4VisDown - 1) / kDs4VisDown;
    const uint32_t NB = bh * bw;
    const uint32_t PD = 3 * p * p;              // 588
    const uint32_t D = kDs4VisDim, Fd = kDs4VisFfn, O = I.opt.out_dim, UF = D * 9;
    sycl::queue& q = alloc.queue();
    // IE_DS4_VIS_TRACE=1: wait after every stage and print elapsed time (hang
    // localisation; serialises the queue, never on by default).
    static const bool trace = std::getenv("IE_DS4_VIS_TRACE") != nullptr;
    const auto t_start = std::chrono::steady_clock::now();
    auto mark = [&](const char* what) {
        if (!trace) return;
        q.wait();
        std::fprintf(stderr, "[ds4-vis] %-14s %.3fs\n", what,
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count());
    };

    if (std::string e = upload(alloc); !e.empty()) return e;
    if (I.alloc != &alloc) return "ds4 vision: allocator changed between calls";
    // transient: weights + scratch in ONE device block that lives for this call (freed after the queue drains)
    struct Lease {
        sycl::queue& q; Impl& I; void* p = nullptr;
        ~Lease() {
            if (!p) return;
            try { q.wait(); } catch (...) {}
            sycl::free(p, q);
            for (auto& sl : I.slots) *sl.pp = nullptr;
        }
    } lease{q, I};
    if (I.opt.transient) {
        lease.p = sycl::malloc_device(I.block_bytes, q);
        if (!lease.p) return "ds4 vision: no device memory for the " + std::to_string(I.block_bytes >> 20) + " MiB encode block";
        for (auto& sl : I.slots) *sl.pp = static_cast<uint8_t*>(lease.p) + sl.off;
        // The staged weights live for the whole process. If anything has overwritten them,
        // every image after it would encode to nonsense -- and the model then answers with
        // one token repeated (seen live 2026-09-19). Check, say so, and restage.
        if (I.hw_hash && ds4v_sample_hash(I.hw, I.hw_bytes) != I.hw_hash) {
            std::fprintf(stderr, "[ds4-vis] the staged tower weights changed since they were loaded — restaging "
                                 "(an image encoded with them would be nonsense)\n");
            if (std::string e = restage_host(); !e.empty()) return "ds4 vision: restage after corruption: " + e;
        }
        q.memcpy(lease.p, I.hw, I.hw_bytes);
        mark("weights-h2d");
    }
    const uint32_t Mp = round8(N), Bp = round8(NB);
    if (Mp > I.cap_n || Bp > I.cap_b)
        return "ds4 vision: image of " + std::to_string(N) + " patches / " + std::to_string(NB) +
               " aligner rows exceeds the preallocated scratch (" + std::to_string(I.cap_n) + " / " +
               std::to_string(I.cap_b) + ")";
    // ---- host prep: patches [N, 588] (c, y, x) in (row, col) order; rope tables
    // (written straight into the pinned staging, then ONE copy each)
    sycl::half* px = I.h_px;
    float *cs = I.h_cs, *sn = I.h_sn;
    for (uint32_t pr = 0; pr < n_h; ++pr)
        for (uint32_t pc = 0; pc < n_w; ++pc) {
            const uint32_t idx = pr * n_w + pc;
            sycl::half* d = &px[size_t(idx) * PD];
            for (uint32_t c = 0; c < 3; ++c)
                for (uint32_t y = 0; y < p; ++y)
                    for (uint32_t x = 0; x < p; ++x)
                        d[(c * p + y) * p + x] =
                            sycl::half(img[(size_t(c) * H + pr * p + y) * W + pc * p + x]);
            for (uint32_t r = 0; r < kDs4VisRot / 2; ++r) {
                // inv[r] = theta^(-2r/32) over dim = rope_dim = 32
                const float inv = std::exp(-2.f * float(r) / float(kDs4VisRot) * std::log(kDs4VisTheta));
                cs[size_t(idx) * kDs4VisRot + r]      = std::cos(float(pr) * inv);
                sn[size_t(idx) * kDs4VisRot + r]      = std::sin(float(pr) * inv);
                cs[size_t(idx) * kDs4VisRot + 16 + r] = std::cos(float(pc) * inv);
                sn[size_t(idx) * kDs4VisRot + 16 + r] = std::sin(float(pc) * inv);
            }
        }
    q.memcpy(I.px, px, size_t(N) * PD * 2);
    q.memcpy(I.cs, cs, size_t(N) * kDs4VisRot * 4);
    q.memcpy(I.sn, sn, size_t(N) * kDs4VisRot * 4);
    mark("host-prep");

    // IE_DS4_VIS_NAIVE_ATTN=1: the original one-WG-per-(token,head) kernel (A/B lever).
    static const bool naive_attn = std::getenv("IE_DS4_VIS_NAIVE_ATTN") != nullptr;
    const float attn_scale = 1.0f / std::sqrt(float(kDs4VisHeadDim));
    // ---- forward (in-order queue) ------------------------------------------
    gemm_fp16(q, I.px, I.patch, I.C, N, D, PD);
    ds4v_bias_set(q, I.C, I.patch_b, I.res, N, D);
    mark("patch-embed");
    for (uint32_t L = 0; L < kDs4VisDepth; ++L) {
        auto& b = I.blk[L];
        if (trace && L && L % 8 == 0) mark("blocks");
        const bool m0 = trace && L == 0;
        ds4v_rms_norm_f16(q, I.res, b.n1, I.xh, N, D, kDs4VisEps);      if (m0) mark("b0 norm1");
        gemm_fp16(q, I.xh, b.qkv, I.C, N, 3 * D, D);               if (m0) mark("b0 qkv gemm");
        ds4v_bias_act_f16(q, I.C, b.qkv_b, I.qkv, N, 3 * D, 0);        if (m0) mark("b0 qkv bias");
        ds4v_rope2d(q, I.qkv, I.cs, I.sn, N);                           if (m0) mark("b0 rope");
        if (naive_attn) {
            ds4v_attn_full(q, I.qkv, I.att, N);
        } else {
            const uint32_t Nc = (N + 127u) & ~127u;
            const uint32_t Tq = I.opt.attn_tile ? std::min(I.opt.attn_tile, N) : N;   // softmax is per query row: tiling rows is exact
            static const char* ps_env = std::getenv("IE_DS4_VIS_PSCALE");                 // A/B lever
            const float pscale = ps_env ? float(std::atof(ps_env)) : I.opt.prob_scale;
            for (uint32_t hd = 0; hd < kDs4VisHeads; ++hd) {
                ds4v_attn_gather(q, I.qkv, I.qh, I.kT, I.vh, N, Nc, hd);
                for (uint32_t r0 = 0; r0 < N; r0 += Tq) {
                    const uint32_t n = std::min(Tq, N - r0);
                    gemm_fp16(q, I.qh + size_t(r0) * kDs4VisHeadDim, I.kT, I.S, n, Nc, kDs4VisHeadDim);
                    ds4v_softmax_rows(q, I.S, I.P, n, N, Nc, attn_scale, pscale);
                    gemm_fp16(q, I.P, I.vh, I.O, n, kDs4VisHeadDim, Nc);
                    ds4v_attn_scatter(q, I.O, I.att + size_t(r0) * kDs4VisDim, n, hd, 1.0f / pscale);
                }
            }
        }
        if (m0) mark("b0 attn");
        gemm_fp16(q, I.att, b.wo, I.C, N, D, D);                   if (m0) mark("b0 wo gemm");
        ds4v_bias_add_res(q, I.C, b.wo_b, I.res, N, D);                 if (m0) mark("b0 wo res");
        ds4v_rms_norm_f16(q, I.res, b.n2, I.xh, N, D, kDs4VisEps);      if (m0) mark("b0 norm2");
        gemm_fp16(q, I.xh, b.w1, I.C, N, 2 * Fd, D);               if (m0) mark("b0 w1 gemm");
        ds4v_swiglu_f16(q, I.C, I.ffn, N, Fd);                          if (m0) mark("b0 swiglu");
        gemm_fp16(q, I.ffn, b.w2, I.C, N, D, Fd);                  if (m0) mark("b0 w2 gemm");
        ds4v_bias_add_res(q, I.C, nullptr, I.res, N, D);                if (m0) mark("b0 w2 res");
    }
    mark("blocks-done");
    ds4v_rms_norm_f16(q, I.res, I.norm, I.xh, N, D, kDs4VisEps);
    ds4v_unfold3(q, I.xh, I.unf, n_h, n_w, bh, bw);
    mark("unfold");
    gemm_fp16(q, I.unf, I.a1, I.Ca, NB, O, UF);
    ds4v_bias_act_f16(q, I.Ca, I.a1_b, I.ah, NB, O, 1);
    gemm_fp16(q, I.ah, I.a2, I.Ca, NB, O, O);
    aligned.resize(size_t(NB) * O);
    q.parallel_for(sycl::range<1>(aligned.size()), [C = I.Ca, bb = I.a2_b, od = O](sycl::id<1> i) {
        C[i] += bb[uint32_t(i % od)];
    });
    q.memcpy(I.h_out, I.Ca, aligned.size() * 4).wait();
    std::memcpy(aligned.data(), I.h_out, aligned.size() * 4);
    mark("aligner");
    return {};
}

// Every device buffer of the tower in one fixed order (weights first), with its offset in the encode block.
void Ds4Vision::Impl::collect_slots() {
    slots.clear();
    const uint32_t PD = 3 * kDs4VisPatch * kDs4VisPatch, D = kDs4VisDim, Fd = kDs4VisFfn, O = opt.out_dim, UF = D * 9;
    auto W2 = [&](sycl::half*& ptr, const std::string& n, uint32_t out_d, uint32_t in_d) {
        slots.push_back({reinterpret_cast<void**>(&ptr), size_t(out_d) * in_d * 2, find(n), out_d, in_d, 0});
    };
    auto W1 = [&](float*& ptr, const std::string& n, uint32_t len) {
        slots.push_back({reinterpret_cast<void**>(&ptr), size_t(len) * 4, find(n), len, 0, 0});
    };
    W2(patch, "vision.patch_embed.proj.weight", D, PD);  W1(patch_b, "vision.patch_embed.proj.bias", D);
    blk.resize(kDs4VisDepth);
    for (uint32_t L = 0; L < kDs4VisDepth; ++L) {
        const std::string pre = "vision.blocks." + std::to_string(L) + ".";
        auto& b = blk[L];
        W1(b.n1, pre + "norm1.weight", D);             W1(b.n2, pre + "norm2.weight", D);
        W2(b.qkv, pre + "attn.wqkv.weight", 3 * D, D); W1(b.qkv_b, pre + "attn.wqkv.bias", 3 * D);
        W2(b.wo, pre + "attn.wo.weight", D, D);        W1(b.wo_b, pre + "attn.wo.bias", D);
        W2(b.w1, pre + "mlp.w1.weight", 2 * Fd, D);    W2(b.w2, pre + "mlp.w2.weight", D, Fd);
    }
    W1(norm, "vision.norm.weight", D);
    W2(a1, "aligner.w1.weight", O, UF);  W1(a1_b, "aligner.w1.bias", O);
    W2(a2, "aligner.w2.weight", O, O);   W1(a2_b, "aligner.w2.bias", O);
    const size_t n_weights = slots.size();
    const size_t Mp = opt.max_patches, Bp = opt.max_blocks, Nc = (Mp + 127u) & ~size_t(127);
    const size_t Tq = opt.attn_tile ? std::min<size_t>(opt.attn_tile, Mp) : Mp, HD = kDs4VisHeadDim;
    auto S = [&](auto*& ptr, size_t bytes) { slots.push_back({reinterpret_cast<void**>(&ptr), bytes, nullptr, 0, 0, 0}); };
    S(px, Mp * PD * 2);  S(res, Mp * D * 4);  S(xh, Mp * D * 2);  S(C, Mp * 2 * Fd * 4);  S(qkv, Mp * 3 * D * 2);
    S(att, Mp * D * 2);  S(ffn, Mp * Fd * 2);  S(cs, Mp * kDs4VisRot * 4);  S(sn, Mp * kDs4VisRot * 4);
    S(qh, Mp * HD * 2);  S(kT, HD * Nc * 2);  S(vh, Nc * HD * 2);  S(this->S, Tq * Nc * 4);  S(P, Tq * Nc * 2);  S(this->O, Mp * HD * 4);
    S(unf, Bp * UF * 2);  S(ah, Bp * O * 2);  S(Ca, Bp * O * 4);
    size_t off = 0;
    for (size_t i = 0; i < slots.size(); ++i) {
        slots[i].off = off;
        off += (slots[i].bytes + 4095) & ~size_t(4095);
        if (i + 1 == n_weights) hw_bytes = off;
    }
    block_bytes = off;
}

uint64_t Ds4Vision::encode_bytes() const {
    Impl& I = *impl_;
    if (!I.open) return 0;
    if (!I.opt.transient) return device_bytes();
    if (I.slots.empty()) I.collect_slots();
    return I.block_bytes;
}

std::string Ds4Vision::stage_host(DeviceAllocator& alloc) {
    Impl& I = *impl_;
    if (!I.open) return "ds4 vision: not loaded";
    if (!I.opt.transient) return "ds4 vision: stage_host is the transient mode's";
    if (I.alloc) return I.alloc == &alloc ? std::string{} : "ds4 vision: allocator changed between calls";
    sycl::queue& q = alloc.queue();
    I.collect_slots();
    for (auto& sl : I.slots) *sl.pp = nullptr;
    const size_t Mp = I.opt.max_patches, Bp = I.opt.max_blocks, PD = 3 * kDs4VisPatch * kDs4VisPatch;
    I.hw    = sycl::malloc_host(I.hw_bytes, q);
    I.h_px  = sycl::malloc_host<sycl::half>(Mp * PD, q);
    I.h_cs  = sycl::malloc_host<float>(Mp * kDs4VisRot, q);
    I.h_sn  = sycl::malloc_host<float>(Mp * kDs4VisRot, q);
    I.h_out = sycl::malloc_host<float>(Bp * I.opt.out_dim, q);
    I.alloc = &alloc;                                   // the destructor frees the pinned buffers through it
    if (!I.hw || !I.h_px || !I.h_cs || !I.h_sn || !I.h_out) return "ds4 vision: pinned host allocation failed";
    fill_host();
    I.cap_n = uint32_t(Mp); I.cap_b = uint32_t(Bp); I.cap_nc = uint32_t((Mp + 127u) & ~size_t(127));
    return {};
}

void Ds4Vision::fill_host() {
    Impl& I = *impl_;
    auto* base = static_cast<uint8_t*>(I.hw);
    for (const auto& sl : I.slots) {
        if (!sl.src) break;                             // weights come first
        const uint8_t* src = sl.src->data;
        if (sl.in_d) {                                  // BF16 [out, in] -> f16 Bt [in, out]
            auto* bt = reinterpret_cast<sycl::half*>(base + sl.off);
            const uint32_t out_d = sl.out_d, in_d = sl.in_d;
            #pragma omp parallel for schedule(static)
            for (uint32_t o = 0; o < out_d; ++o)
                for (uint32_t k = 0; k < in_d; ++k)
                    bt[size_t(k) * out_d + o] = sycl::half(bf16_at(src, size_t(o) * in_d + k));
        } else {
            auto* f = reinterpret_cast<float*>(base + sl.off);
            for (uint32_t i = 0; i < sl.out_d; ++i) f[i] = bf16_at(src, i);
        }
    }
    I.hw_hash = ds4v_sample_hash(I.hw, I.hw_bytes);
}

std::string Ds4Vision::restage_host() {
    Impl& I = *impl_;
    if (!I.hw || !I.open) return "the tower is not staged";
    fill_host();
    return {};
}

}  // namespace ie
