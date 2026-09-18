// src/model/ds41_vision.cpp — see include/ie/ds41_vision.hpp. Host code only: no kernels live here (the tower's
// kernels are ds4_vision.cpp's, and a second TU with same-named kernel helpers is the 2026-09-01 collision trap).

#include "ie/ds41_vision.hpp"

#include "stb/stb_image.h"   // implementation instantiated in qwen4_image.cpp

#include <algorithm>
#include <cmath>

namespace ie {

namespace {

constexpr uint32_t kP = kDs4VisPatch, kR = kDs4VisDown;

struct Grid { uint32_t h, w; };
Grid llm_grid(uint32_t best_h, uint32_t best_w) { return {((best_h / kP) + kR - 1) / kR, ((best_w / kP) + kR - 1) / kR}; }

// solve_resize_ratio(): the largest aspect-preserving pixel size whose token grid fits the budget.
void solve_resize_ratio(double height, double width, uint32_t max_n, uint32_t& best_h, uint32_t& best_w) {
    const double r = height / width;
    const double max_w_f = std::sqrt((double(max_n) - 2.0) / r + 0.25) - 0.5;
    const double max_h_f = max_w_f * r;
    const uint32_t cell = kP * kR;
    if (max_w_f < 1.0) { best_h = (max_n - 2) / 2 * cell; best_w = cell; return; }       // very tall: one column
    if (max_h_f < 1.0) { best_h = cell; best_w = (max_n - 3) * cell; return; }           // very wide: one row
    const double beta = std::min(std::floor(max_w_f) * cell / width, std::floor(max_h_f) * cell / height);
    best_h = uint32_t(std::floor(height * beta / kP)) * kP;
    best_w = uint32_t(std::floor(width * beta / kP)) * kP;
}

}  // namespace

Ds4VisGeom ds41_vis_plan(uint32_t width0, uint32_t height0) {
    Ds4VisGeom out;
    if (!width0 || !height0) return out;
    double width = width0, height = height0;
    if (width * height < double(kDs41VisMinPx)) {
        const double ratio = std::sqrt(double(kDs41VisMinPx) / (width * height));
        width  = std::trunc(width * ratio);      // Python int()
        height = std::trunc(height * ratio);
    }
    uint32_t best_w = uint32_t(std::ceil(width / kP)) * kP;
    uint32_t best_h = uint32_t(std::ceil(height / kP)) * kP;
    Grid g = llm_grid(best_h, best_w);
    if (ds41_vis_tokens(g.h, g.w) > kDs41VisMaxTok) {      // safe_resize
        solve_resize_ratio(height, width, kDs41VisMaxTok, best_h, best_w);
        g = llm_grid(best_h, best_w);
    }
    out.best_w = best_w; out.best_h = best_h;
    out.n_vit_h = best_h / kP; out.n_vit_w = best_w / kP;
    out.n_llm_h = g.h; out.n_llm_w = g.w;
    out.plain = false;
    return out;
}

void ds41_vis_types(uint32_t n_llm_h, uint32_t n_llm_w, std::vector<int32_t>& types, std::vector<int32_t>& perm) {
    types.clear(); perm.clear();
    types.reserve(ds41_vis_tokens(n_llm_h, n_llm_w));
    types.push_back(kDs4VisStart);
    for (uint32_t r = 0; r < n_llm_h; ++r) {
        for (uint32_t c = 0; c < n_llm_w; ++c) { types.push_back(kDs4VisImage); perm.push_back(int32_t(r * n_llm_w + c)); }
        types.push_back(kDs4VisNewline);
    }
    types.push_back(kDs4VisEnd);
}

std::string ds41_image_size(const void* bytes, size_t nbytes, uint32_t& width, uint32_t& height) {
    int w = 0, h = 0, comp = 0;
    if (!stbi_info_from_memory(static_cast<const stbi_uc*>(bytes), int(nbytes), &w, &h, &comp) || w <= 0 || h <= 0)
        return std::string("vision: not a decodable image: ") + (stbi_failure_reason() ? stbi_failure_reason() : "unknown format");
    width = uint32_t(w); height = uint32_t(h);
    return {};
}

std::string ds41_load_image_mem(const void* bytes, size_t nbytes, std::vector<float>& px,
                                uint32_t& H, uint32_t& W, Ds4VisGeom& geom) {
    return ds4_load_image_planned(bytes, nbytes, [](uint32_t w, uint32_t h) { return ds41_vis_plan(w, h); }, px, H, W, geom);
}

Ds4VisionOptions ds41_vision_options(uint32_t lm_hidden) {
    Ds4VisionOptions o;
    o.out_dim = lm_hidden;
    o.max_patches = kDs41VisMaxPatches;
    o.max_blocks = kDs41VisMaxTok;
    o.pad_row = false;
    o.attn_tile = kDs41VisAttnTile;
    o.prob_scale = kDs41VisProbScale;
    o.transient = true;
    return o;
}

}  // namespace ie
