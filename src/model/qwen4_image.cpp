// src/model/qwen4_image.cpp — image file -> normalized [3, H, W] f32 for the
// qwen4exp vision tower (docs/qwen4/16_vision_port.md §2).
//
// Semantics match Qwen2VLImageProcessorFast + llama.cpp mtmd-image.cpp
// calc_size_preserved_ratio: smart_resize to multiples of 32 inside the
// [65536, 16777216] pixel budget (bilinear), then (x/255 - 0.5) / 0.5.

#include "ie/qwen4_vision.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#include "stb/stb_image.h"

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace ie {

static void smart_resize_dims(int h, int w, uint32_t& h_bar, uint32_t& w_bar) {
    static constexpr int kFactor = 32;         // patch 16 * merge 2
    static constexpr int64_t kMinPx = 65536;   // preprocessor_config shortest_edge
    // preprocessor_config longest_edge; IE_VIS_MAX_PX caps it (the CPU reference
    // ViT is O(N^2) in patches — test lever until the P3 GPU path lands).
    static const int64_t kMaxPx = [] {
        if (const char* v = std::getenv("IE_VIS_MAX_PX")) {
            const int64_t p = std::atoll(v);
            if (p >= kMinPx) return p;
        }
        return int64_t(16777216);
    }();
    auto round_f = [](float x) { return std::max(kFactor, int(std::round(x / kFactor)) * kFactor); };
    auto floor_f = [](float x) { return std::max(kFactor, int(std::floor(x / kFactor)) * kFactor); };
    auto ceil_f  = [](float x) { return int(std::ceil(x / kFactor)) * kFactor; };
    int hb = round_f(float(h)), wb = round_f(float(w));
    if (int64_t(hb) * wb > kMaxPx) {
        const float beta = std::sqrt(float(int64_t(h) * w) / float(kMaxPx));
        hb = floor_f(float(h) / beta);
        wb = floor_f(float(w) / beta);
    } else if (int64_t(hb) * wb < kMinPx) {
        const float beta = std::sqrt(float(kMinPx) / float(int64_t(h) * w));
        hb = ceil_f(float(h) * beta);
        wb = ceil_f(float(w) * beta);
    }
    h_bar = uint32_t(hb);
    w_bar = uint32_t(wb);
}

// Bilinear resize (align_corners=False half-pixel centres — F.interpolate /
// PIL-free fast-processor convention), u8 RGB -> f32 RGB.
static void resize_bilinear(const uint8_t* src, int sh, int sw,
                            float* dst, uint32_t dh, uint32_t dw) {
    for (uint32_t y = 0; y < dh; ++y) {
        const float fy = (float(y) + 0.5f) * float(sh) / float(dh) - 0.5f;
        const int y0 = std::clamp(int(std::floor(fy)), 0, sh - 1);
        const int y1 = std::min(y0 + 1, sh - 1);
        const float wy = std::clamp(fy - float(y0), 0.f, 1.f);
        for (uint32_t x = 0; x < dw; ++x) {
            const float fx = (float(x) + 0.5f) * float(sw) / float(dw) - 0.5f;
            const int x0 = std::clamp(int(std::floor(fx)), 0, sw - 1);
            const int x1 = std::min(x0 + 1, sw - 1);
            const float wx = std::clamp(fx - float(x0), 0.f, 1.f);
            for (int c = 0; c < 3; ++c) {
                const float p00 = src[(size_t(y0) * sw + x0) * 3 + c];
                const float p01 = src[(size_t(y0) * sw + x1) * 3 + c];
                const float p10 = src[(size_t(y1) * sw + x0) * 3 + c];
                const float p11 = src[(size_t(y1) * sw + x1) * 3 + c];
                dst[(size_t(c) * dh + y) * dw + x] =
                    (p00 * (1 - wx) + p01 * wx) * (1 - wy) +
                    (p10 * (1 - wx) + p11 * wx) * wy;
            }
        }
    }
}

static std::string finish_load(uint8_t* rgb, int sh, int sw,
                               std::vector<float>& out,
                               uint32_t& H, uint32_t& W) {
    smart_resize_dims(sh, sw, H, W);
    out.resize(size_t(3) * H * W);
    resize_bilinear(rgb, sh, sw, out.data(), H, W);
    stbi_image_free(rgb);
    for (auto& v : out) v = (v / 255.0f - 0.5f) / 0.5f;
    return {};
}

std::string qwen4_load_image(const std::string& path, std::vector<float>& out,
                             uint32_t& H, uint32_t& W) {
    int sw = 0, sh = 0, comp = 0;
    uint8_t* rgb = stbi_load(path.c_str(), &sw, &sh, &comp, 3);
    if (!rgb) return "qwen4 image: cannot decode " + path + " (" +
                     (stbi_failure_reason() ? stbi_failure_reason() : "?") + ")";
    return finish_load(rgb, sh, sw, out, H, W);
}

std::string qwen4_load_image_mem(const void* bytes, size_t nbytes,
                                 std::vector<float>& out,
                                 uint32_t& H, uint32_t& W) {
    int sw = 0, sh = 0, comp = 0;
    uint8_t* rgb = stbi_load_from_memory(static_cast<const stbi_uc*>(bytes),
                                         int(nbytes), &sw, &sh, &comp, 3);
    if (!rgb) return std::string("qwen4 image: cannot decode buffer (") +
                     (stbi_failure_reason() ? stbi_failure_reason() : "?") + ")";
    return finish_load(rgb, sh, sw, out, H, W);
}

}  // namespace ie
