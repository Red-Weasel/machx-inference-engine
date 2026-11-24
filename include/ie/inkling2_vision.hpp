// include/ie/inkling2_vision.hpp — Inkling 2 Mini's vision tower (model.visual.*): the image processor and the encoder that
// turns one image into the rows replacing its placeholder tokens (<|unused_200054|>, one per 40 x 40 pixel block) in the
// language model's input, after embed_norm (the reference's inputs_embeds.masked_scatter).
//
// Processor (Inkling2ImageProcessor): no resize; the RGB image is padded to multiples of 40 with -1 on the 0-255 scale,
// rescaled by 1/255 and normalised (x - 0.5) / 0.5; one image token = one 40 x 40 block = 2 x 2 native 20 x 20 patches.
// Tower (Inkling2VisionModel): patch embedding [1024, 1200] (no bias) + the 32 x 32 learned position grid resampled
// bilinearly (align_corners False) to the 2R x 2C patch grid; a class token (class + cls position) in front; ln_pre;
// 23 pre-LN blocks (16 heads of 64, axial 2D RoPE theta 1e4 over interleaved pairs, width then height frequencies, the
// class token at (0, 0) and patch (r, c) at (r + 1, c + 1); layer scales ls_1 / ls_2; GELU (erf) MLP 4096); the patch
// rows (no class token) in groups of 4 -> proj_in [4096, 4096] -> GELU -> proj_out [2304, 4096] (no biases).
// GEMMs through oneDNN with fp16 operands and fp32 accumulation; fp32 between blocks.
#pragma once
#include <sycl/sycl.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ie {

inline constexpr uint32_t kInk2VisBlock = 40;              // pixels per image token side
inline constexpr uint32_t kInk2VisMaxSide = 4096;          // the README's largest recommended side

struct Inkling2Image {
    uint32_t rows = 0, cols = 0;                           // image tokens down / across
    std::vector<float> pixels;                             // [rows * cols, 40, 40, 3] normalised (the processor's output)
    uint32_t tokens() const { return rows * cols; }
};
// Decode (stb: PNG / JPEG / BMP / GIF first frame / ...) and preprocess one image. RGBA is composited onto the
// processor's 8-pixel checkerboard (204 / 153) as Inkling2ImageProcessor.convert_to_rgb does. "" on success.
// max_tokens > 0: an image needing more tokens is first scaled down (PIL bicubic, aspect kept) until it fits -- the
// reference never resizes (it takes sides up to 4096 px); this keeps a large screenshot within the tower's scratch and
// the context. *resized (optional) says whether it happened.
std::string inkling2_load_image(const void* bytes, size_t nbytes, Inkling2Image& out, uint32_t max_tokens = 0, bool* resized = nullptr);
// The same from raw RGB8 [h, w, 3].
std::string inkling2_image_from_rgb(const uint8_t* rgb, uint32_t h, uint32_t w, Inkling2Image& out);

class Inkling2Vision {
public:
    Inkling2Vision();
    ~Inkling2Vision();
    // Read model.visual.* from <dir>/model-00001-of-00001.safetensors onto q's card; scratch for images up to max_tokens
    // image tokens (4 patches each). "" on success.
    // resident_tokens (0 = max_tokens): the scratch kept on the card between images; a larger image grows it for its
    // encode (grow) and shrink() returns it, so the resident footprint is the weights + a resident_tokens scratch.
    std::string init(sycl::queue& q, const std::string& dir, uint32_t max_tokens = 2048, uint32_t resident_tokens = 0);
    bool ready() const;
    uint32_t max_tokens() const;
    uint32_t capacity() const;                               // image tokens the current scratch takes
    bool grow(uint32_t tokens);                              // scratch for `tokens` (false: no room; the resident scratch kept)
    void shrink();                                           // back to the resident scratch
    uint64_t device_bytes() const;
    // One image -> rows [tokens, 2304] (fp32, host). captured (optional): {after ln_pre, after layer 0, the last hidden
    // state}, each [1 + patches, 1024], for comparing with the reference.
    std::string encode(const Inkling2Image& img, std::vector<float>& rows, std::vector<std::vector<float>>* captured = nullptr);
    void free_all();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ie
