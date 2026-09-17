// include/ie/qwen4_vision.hpp — Qwen3.8-Flash-Next (qwen4exp) vision tower.
//
// Spec: docs/qwen4/16_vision_port.md. CPU fp32 reference path (P1): loads
// mmproj-F16.gguf, encodes one preprocessed image into merged embeddings the
// text model splices at image-token positions. GPU path arrives in P3.
//
// The input image is already smart-resized and normalized ([3, H, W] f32,
// H and W multiples of 32 = patch(16) * merge(2)); patch extraction, position
// ids, pos-embed interpolation and the 2x2 merge all use the block-major
// ordering shared with the reference (vision_utils.py:118-125).

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ie {

class DeviceAllocator;

// Fixed qwen4exp vision geometry (GGUF + config.json, spec §1).
inline constexpr uint32_t kVisPatch    = 16;
inline constexpr uint32_t kVisMerge    = 2;
inline constexpr uint32_t kVisHid      = 1152;
inline constexpr uint32_t kVisHeads    = 16;
inline constexpr uint32_t kVisHeadDim  = 72;    // 1152/16
inline constexpr uint32_t kVisRot      = 36;    // head_dim/2: 18 h + 18 w freqs
inline constexpr uint32_t kVisFfn      = 4304;
inline constexpr uint32_t kVisDepth    = 27;
inline constexpr uint32_t kVisGridSide = 48;    // sqrt(2304) learned pos table
inline constexpr uint32_t kVisOutD     = 2560;  // text model width
inline constexpr float    kVisLnEps    = 1e-6f;
inline constexpr float    kVisTheta    = 1e4f;

class Qwen4Vision {
public:
    Qwen4Vision();
    ~Qwen4Vision();

    // Reads mmproj GGUF, materializes all weights as f32. "" on success.
    std::string load(const std::string& mmproj_path);

    // img: [3, H, W] f32, normalized ((x/255 - 0.5) / 0.5), H % 32 == W % 32 == 0.
    // out: [(H/32)*(W/32), 2560] f32, block-major merged-token order.
    // "" on success.
    std::string encode(const float* img, uint32_t H, uint32_t W,
                       std::vector<float>& out) const;

    // GPU path (P3, src/ops/qwen4_vision_gpu.cpp): same contract as encode().
    // First call uploads the weights (~0.9 GiB f16) to alloc's device; they
    // stay resident for the object's lifetime. Patch grid is capped at 12288
    // patches (attention-kernel SLM bound) — cap resolution via IE_VIS_MAX_PX.
    std::string encode_gpu(DeviceAllocator& alloc, const float* img,
                           uint32_t H, uint32_t W, std::vector<float>& out);

    bool loaded() const { return !blocks_.empty(); }

private:
    struct Block {
        std::vector<float> ln1_w, ln1_b, ln2_w, ln2_b;
        std::vector<float> qkv_w, qkv_b;      // [3456, 1152], [3456]
        std::vector<float> out_w, out_b;      // [1152, 1152], [1152]
        std::vector<float> up_w, up_b;        // [4304, 1152], [4304]
        std::vector<float> down_w, down_b;    // [1152, 4304], [1152]
    };
    std::vector<float> patch_w_;              // [1152, 1536] (t0|t1 fused, [C,T,ph,pw])
    std::vector<float> patch_b_;              // [1152]
    std::vector<float> pos_embd_;             // [2304, 1152]
    std::vector<float> post_ln_w_, post_ln_b_;
    std::vector<float> mm0_w_, mm0_b_;        // [4608, 4608], [4608]
    std::vector<float> mm2_w_, mm2_b_;        // [2560, 4608], [2560]
    std::vector<Block> blocks_;

    struct GpuState;                  // device weights + scratch (gpu cpp)
    std::unique_ptr<GpuState> gpu_;
};

// (src/model/qwen4_image.cpp) Decode + smart_resize + normalize an image file
// into the encode() input format. Sets H, W (multiples of 32). "" on success.
std::string qwen4_load_image(const std::string& path, std::vector<float>& out,
                             uint32_t& H, uint32_t& W);

// Same, from an in-memory encoded image (JPEG/PNG/BMP bytes) — the server's
// base64 data-URI path.
std::string qwen4_load_image_mem(const void* bytes, size_t nbytes,
                                 std::vector<float>& out,
                                 uint32_t& H, uint32_t& W);

// (src/model/qwen4_vision.cpp) Interleaved M-RoPE position table for a token
// sequence with image-pad spans (docs/qwen4/16_vision_port.md §4, HF
// get_rope_index): text runs linear on all three streams; each image span gets
// T=cur, H=cur+row, W=cur+col over its merged grid and advances cur by
// max(gh2, gw2). Returns pos3 [3, n] stream-major and the decode rope delta
// (max_pos + 1 - n).
struct Qwen4VisGrid { uint32_t t0, gh2, gw2; };   // token offset, merged grid
void qwen4_build_mrope3(uint32_t n_tokens, const std::vector<Qwen4VisGrid>& imgs,
                        std::vector<int32_t>& pos3, int32_t& delta);

}  // namespace ie
