// include/ie/mimo26_vision.hpp — MiMo-V2.6 vision input (docs/mimo26/00_PORT_PLAN.md, P6).
//
// The checkpoint's processor is transformers' Qwen2VLImageProcessor(Pil) with its preprocessor_config.json:
// smart_resize to multiples of 32 (patch 16 x merge 2) inside [3,136, 12,845,056] pixels, PIL bicubic, rescale
// 1/255, CLIP mean/std, a still image doubled in time (temporal patch 2), and the patches laid out merge block by
// merge block, each patch a (channel, time, y, x) vector of 1,536 values.
#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace ie {

inline constexpr uint32_t kMimoVisPatch    = 16;
inline constexpr uint32_t kMimoVisMerge    = 2;
inline constexpr uint32_t kMimoVisFactor   = kMimoVisPatch * kMimoVisMerge;   // 32
inline constexpr uint32_t kMimoVisPatchDim = 3 * 2 * kMimoVisPatch * kMimoVisPatch;   // 1,536
inline constexpr uint64_t kMimoVisMinPx    = 3136;
inline constexpr uint64_t kMimoVisMaxPx    = 12845056;

struct MimoVisGeom {
    uint32_t h = 0, w = 0;              // the resized canvas, multiples of 32
    uint32_t grid_h = 0, grid_w = 0;    // the patch grid (h / 16, w / 16)
    uint32_t patches() const { return grid_h * grid_w; }
    uint32_t tokens() const { return patches() / (kMimoVisMerge * kMimoVisMerge); }
};

// transformers' smart_resize(height, width, factor=32, min_pixels, max_pixels) with Python's rounding (half to
// even). "" on success, else the reason (an aspect ratio above 200 is refused, as the processor does).
std::string mimo26_vis_plan(uint32_t height, uint32_t width, MimoVisGeom& geom, uint64_t max_px = kMimoVisMaxPx);

// Decode (JPEG / PNG / BMP), resize, normalise and lay out: pixel_values [patches, 1,536] f32. "" on success.
std::string mimo26_load_image_mem(const void* bytes, size_t nbytes, std::vector<float>& pixel_values,
                                  MimoVisGeom& geom, uint64_t max_px = kMimoVisMaxPx);

class DeviceAllocator;
struct SafeTensorInfo;

// The mimovl tower (docs/mimo26/00_PORT_PLAN.md, "Tower facts"), transient like DeepSeek-V4.1's: the converted f16
// weights live in pinned host memory; ONE device block (weights + scratch) exists only inside encode_gpu(), because
// the language model's cards are full at steady state.
struct MimoVisionOptions {
    uint32_t max_patches = 8192;    // scratch cap: 2,048 image tokens (a 1920x1088 frame is 8,160 patches)
    uint32_t attn_tile   = 1024;    // query rows per attention GEMM (softmax is per row, so tiling is exact)
    uint32_t mlp_rows    = 2048;    // rows per SwiGLU pass (bounds the 4,608-wide intermediates)
    float    prob_scale  = 256.f;   // f16 probabilities stored as p * scale (docs/deepseek41/94)
    bool     stream_weights = true; // block weights copied per block from pinned host memory into two device slots (P6.3):
                                    // the encode block holds scratch + patch embed + merger (~0.8 GB) instead of every weight (~2 GB)
};

class MimoVision {
public:
    MimoVision();
    ~MimoVision();
    // The 364 `visual.*` BF16 tensors from any source (the checkpoint's shard); `find` must stay valid until staged.
    std::string load_from(std::function<const SafeTensorInfo*(const std::string&)> find, const MimoVisionOptions& opt = {});
    // Convert the weights into pinned host memory (else the first encode_gpu() does it).
    std::string stage_host(DeviceAllocator& alloc);
    uint64_t encode_bytes() const;    // the device block one encode leases
    bool loaded() const;
    // pixel_values [patches, 1,536] (mimo26_load_image_mem) -> merged [tokens, 4,096] f32. `capture`: blocks whose
    // output is also returned in ROW order (the gate's per-block checkpoints).
    std::string encode_gpu(DeviceAllocator& alloc, const std::vector<float>& pixel_values, const MimoVisGeom& geom,
                           std::vector<float>& merged, const std::vector<uint32_t>& capture = {},
                           std::vector<std::vector<float>>* captured = nullptr);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ie
