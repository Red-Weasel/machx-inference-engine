// include/ie/hy4flash_vision.hpp — Tencent Hy4-flash: image input. Reference: transformers' HYV4FlashVisionTransformer
// (modeling_hy_v4_flash.py) and HYV4FlashImageProcessorPil; oracle: scripts/hy4flash_oracle/vision_oracle.py.
//
// The processor: smart_resize to multiples of 32 (patch 16 x merge 2) inside [512 * 512, max_px] pixels, PIL bicubic,
// rescale 1/255, CLIP mean / std, patches row by row, each a (channel, y, x) vector of 768 values.
// The tower: a 16 x 16 patch embedding with bias, a learned 128 x 128 position grid resampled (bilinear) to the image's
// grid, 27 pre-norm blocks (LayerNorm, 16-head attention over the whole image with biases, GELU MLP 4304), then the
// projector: RMSNorm, a 2 x 2 stride-2 convolution to 4608, GELU, a linear map to the text width (3072). Every merged
// row of the grid is closed by the learned `image_newline` vector: an image of grid (gh, gw) is
// (gh / 2) * (gw / 2 + 1) rows of the text model.
#pragma once
#include "ie/hy4flash.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ie {

inline constexpr uint32_t kHy4fVisPatch = 16, kHy4fVisMerge = 2, kHy4fVisFactor = 32, kHy4fVisPatchDim = 3 * 16 * 16;
inline constexpr uint64_t kHy4fVisMinPx = 512ull * 512ull, kHy4fVisRefMaxPx = 2048ull * 2048ull;

struct Hy4fVisGeom {
    uint32_t h = 0, w = 0;                 // the resized canvas, multiples of 32
    uint32_t grid_h = 0, grid_w = 0;       // the patch grid
    uint32_t patches() const { return grid_h * grid_w; }
    uint32_t rows() const { return grid_h / kHy4fVisMerge; }            // merged rows
    uint32_t cols() const { return grid_w / kHy4fVisMerge + 1; }        // merged columns + the newline
    uint32_t tokens() const { return rows() * cols(); }
};
// transformers' smart_resize(height, width, factor 32, min 512 * 512, max_px), Python's rounding. "" on success.
std::string hy4f_vis_plan(uint32_t height, uint32_t width, Hy4fVisGeom& g, uint64_t max_px);
// Decode (JPEG / PNG / BMP), resize, normalise, lay out: pixel_values [patches, 768] f32. "" on success.
std::string hy4f_load_image_mem(const void* bytes, size_t nbytes, std::vector<float>& pixel_values, Hy4fVisGeom& g, uint64_t max_px);
// The text model's 3-D positions of an image whose first row sits at position `start`: pos3 [tokens, 3] = (start,
// start + row, start + column), row by row; returns the position the text continues at (start + max(rows - 1, cols - 1) + 1).
int32_t hy4f_vis_positions(const Hy4fVisGeom& g, int32_t start, std::vector<int32_t>& pos3);

class Hy4fVision {
public:
    Hy4fVision();
    ~Hy4fVision();
    // The checkpoint's `vit.*` tensors as fp16 on the card of `q`, scratch for images up to max_px pixels. "" on success.
    std::string init(sycl::queue& q, const Hy4FlashModel& m, uint64_t max_px);
    bool     ready() const;
    uint64_t max_px() const;
    uint64_t device_bytes() const;
    // pixel_values [patches, 768] -> merged [tokens, text width] f32, the newline vector closing every merged row.
    // capture / captured: blocks whose output [patches, 1152] is also returned (the oracle's checkpoints).
    std::string encode(const std::vector<float>& pixel_values, const Hy4fVisGeom& g, std::vector<float>& merged,
                       const std::vector<uint32_t>& capture = {}, std::vector<std::vector<float>>* captured = nullptr);
    void free_all();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ie
