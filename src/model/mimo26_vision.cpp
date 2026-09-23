// src/model/mimo26_vision.cpp — MiMo-V2.6 vision input: the resize plan and the image loader (P6.1a).
//
// Built with -fp-model=precise (src/CMakeLists.txt): icpx's fast model compiles x / 14 as x * (1/14) and moved
// DeepSeek-V4.1's plan off Python's by one patch on 6 of 156 sizes (docs/deepseek41/94, "Found on the way").
#include "ie/mimo26_vision.hpp"
#include "ie/ds4_vision.hpp"      // ds4_pil_resize: PIL's bicubic, the processor's resample
#include "stb/stb_image.h"         // implementation instantiated in qwen4_image.cpp

#include <algorithm>
#include <cmath>

namespace ie {

std::string mimo26_vis_plan(uint32_t height, uint32_t width, MimoVisGeom& g, uint64_t max_px) {
    g = {};
    if (!height || !width) return "vision: the image is empty";
    const double h = height, w = width, f = kMimoVisFactor;
    if (std::max(h, w) / std::min(h, w) > 200.0) return "vision: the image's aspect ratio is above 200";
    // round(height / factor) * factor: Python rounds half to even, as std::nearbyint does in the default mode
    double hb = std::nearbyint(h / f) * f, wb = std::nearbyint(w / f) * f;
    if (hb * wb > double(max_px)) {
        const double beta = std::sqrt(h * w / double(max_px));
        hb = std::max(f, std::floor(h / beta / f) * f);
        wb = std::max(f, std::floor(w / beta / f) * f);
    } else if (hb * wb < double(kMimoVisMinPx)) {
        const double beta = std::sqrt(double(kMimoVisMinPx) / (h * w));
        hb = std::ceil(h * beta / f) * f;
        wb = std::ceil(w * beta / f) * f;
    }
    g.h = uint32_t(hb);
    g.w = uint32_t(wb);
    g.grid_h = g.h / kMimoVisPatch;
    g.grid_w = g.w / kMimoVisPatch;
    return {};
}

std::string mimo26_load_image_mem(const void* bytes, size_t nbytes, std::vector<float>& pv, MimoVisGeom& g,
                                  uint64_t max_px) {
    int sw = 0, sh = 0, comp = 0;
    uint8_t* rgb = stbi_load_from_memory(static_cast<const stbi_uc*>(bytes), int(nbytes), &sw, &sh, &comp, 3);
    if (!rgb) return std::string("vision: image decode failed: ") + stbi_failure_reason();
    std::string err = mimo26_vis_plan(uint32_t(sh), uint32_t(sw), g, max_px);
    if (!err.empty()) {
        stbi_image_free(rgb);
        return err;
    }
    std::vector<uint8_t> canvas(size_t(g.w) * g.h * 3);
    ds4_pil_resize(rgb, sw, sh, canvas.data(), int(g.w), int(g.h));
    stbi_image_free(rgb);
    // rescale (float64 product, cast to float32) then (x - mean) / std in float32: the processor's numpy steps
    static constexpr float kMean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
    static constexpr float kStd[3]  = {0.26862954f, 0.26130258f, 0.27577711f};
    const uint32_t P = kMimoVisPatch, M = kMimoVisMerge;
    pv.assign(size_t(g.patches()) * kMimoVisPatchDim, 0.f);
    size_t n = 0;
    for (uint32_t by = 0; by < g.grid_h / M; ++by)
        for (uint32_t bx = 0; bx < g.grid_w / M; ++bx)
            for (uint32_t my = 0; my < M; ++my)
                for (uint32_t mx = 0; mx < M; ++mx, ++n) {
                    const uint32_t y0 = (by * M + my) * P, x0 = (bx * M + mx) * P;
                    float* out = &pv[n * kMimoVisPatchDim];
                    for (uint32_t c = 0; c < 3; ++c)
                        for (uint32_t t = 0; t < 2; ++t)          // a still image is its own second frame
                            for (uint32_t py = 0; py < P; ++py)
                                for (uint32_t px = 0; px < P; ++px) {
                                    const uint8_t v = canvas[(size_t(y0 + py) * g.w + x0 + px) * 3 + c];
                                    const float r = float(double(v) * (1.0 / 255.0));
                                    *out++ = (r - kMean[c]) / kStd[c];
                                }
                }
    return {};
}

}  // namespace ie
