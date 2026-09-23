// include/ie/ds4_vision.hpp — DeepSeek-V4-Flash-Vision-Exp vision tower + aligner.
//
// Spec: docs/deepseek4/70_VISION_EXP_PORT_PLAN.md. Reference: the official
// inference/vision.py + inference/image_processor.py. Oracle:
// tests/oracle/ds4_vision_oracle.py (numpy, gated against the official torch
// modules). Weights come from the native safetensors sidecar
// (`…-Native.safetensors`, 267 BF16 tensors), not from the GGUF.
//
// Pipeline: pixels [3,H,W] (H,W multiples of 14, normalised (v/255-0.5)/0.5)
//   -> patch embed (Linear 588->1024 + bias)
//   -> 32 x { RMSNorm, wqkv(+b), 2-D RoPE over the full 64-d head, full
//             bidirectional attention, wo(+b), RMSNorm, SwiGLU 1024->2816->1024 }
//   -> RMSNorm -> 3x3 unfold (9216) -> w1(+b) -> GELU(erf) -> w2(+b)
//   = aligner rows [n_llm_h * n_llm_w, 4096] in grid (row-major) order.
// The LM block (`ds4_vis_build_block` + `assemble_rows`) then lays those rows
// out in the official N-layout with the four control rows.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace ie {

class DeviceAllocator;
struct SafeTensorInfo;

inline constexpr uint32_t kDs4VisPatch   = 14;
inline constexpr uint32_t kDs4VisDim     = 1024;
inline constexpr uint32_t kDs4VisHeads   = 16;
inline constexpr uint32_t kDs4VisHeadDim = 64;     // 1024/16
inline constexpr uint32_t kDs4VisRot     = 32;     // head_dim/2: 16 h-freqs ‖ 16 w-freqs
inline constexpr uint32_t kDs4VisFfn     = 2816;
inline constexpr uint32_t kDs4VisDepth   = 32;
inline constexpr uint32_t kDs4VisDown    = 3;      // aligner downsample ratio
inline constexpr uint32_t kDs4VisOut     = 4096;   // LM hidden
inline constexpr uint32_t kDs4VisMaxTok  = 384;
inline constexpr uint32_t kDs4VisMinPx   = 147456;
inline constexpr uint32_t kDs4VisMaxWh   = 8;
inline constexpr uint32_t kDs4VisPadTo   = 4;      // COMPRESS_PAD_TO
inline constexpr float    kDs4VisEps     = 1e-6f;
inline constexpr float    kDs4VisTheta   = 1e4f;

// Row types of the image block (image_processor.py).
enum Ds4VisType : int32_t {
    kDs4VisStart = 0, kDs4VisPad = 1, kDs4VisImage = 2, kDs4VisNewline = 3, kDs4VisEnd = 4
};

// Official load_image() geometry for a source image of (width, height) px.
struct Ds4VisGeom {
    uint32_t best_w = 0, best_h = 0;      // resized canvas, multiples of 14
    uint32_t n_vit_h = 0, n_vit_w = 0;    // patch grid
    uint32_t n_llm_h = 0, n_llm_w = 0;    // aligner grid (ceil(n_vit/3))
    bool     plain = false;               // true: plain resize; false: contain + grey pad
};
Ds4VisGeom ds4_vis_plan(uint32_t width, uint32_t height);

// Official grid_tokens(): rows the block occupies EXCLUDING the compress pad.
uint32_t ds4_vis_grid_tokens(uint32_t n_llm_h, uint32_t n_llm_w);

// Official build_image_block(): `types` [n_rows] and `perm` [n_llm_h*n_llm_w]
// (aligner row for each IMAGE slot, in sequence order). n_rows =
// grid_tokens + (3 - start_pos % 4).
void ds4_vis_build_block(uint32_t n_llm_h, uint32_t n_llm_w, uint32_t start_pos,
                         std::vector<int32_t>& types, std::vector<int32_t>& perm);

// The tower's runtime shape. The defaults are the V4 Vision-Exp sidecar; DeepSeek-V4.1 runs the SAME tower
// (32 x 1024, 2-D RoPE, SwiGLU 2816, 3x3 aligner) into a 5120-wide LM with a 1,024-token image budget and no
// pad row (include/ie/ds41_vision.hpp).
struct Ds4VisionOptions {
    uint32_t out_dim     = kDs4VisOut;   // LM hidden the aligner projects to
    uint32_t max_patches = 3584;         // scratch caps (kMaxPatches / kMaxBlocks for the sidecar)
    uint32_t max_blocks  = 400;
    bool     pad_row     = true;         // the sidecar carries `image_pad`
    uint32_t attn_tile   = 0;            // query rows per attention GEMM; 0 = every row in one GEMM
    // The attention probabilities are an f16 GEMM input. Past a few thousand keys most of them sit below f16's
    // normal range (6.1e-5) and lose their low bits; stored as p * prob_scale they keep them (the row sum is then
    // prob_scale, and the scatter divides it back out). 1 = the sidecar's original arithmetic, bit for bit.
    float    prob_scale  = 1.0f;
    // transient: the converted weights live in pinned host memory and the device copy (weights + scratch, one
    // block) exists only inside encode_gpu(). For a language model whose cards are full at steady state: the
    // encode runs before the prompt's prefill, in the VRAM that prefill is about to use.
    bool     transient   = false;
};

class Ds4Vision {
public:
    Ds4Vision();
    ~Ds4Vision();

    // The tower's tensors from any source (a checkpoint's own shards): `find` returns the BF16 tensor of that
    // name or null, and must stay valid until the weights are uploaded (persistent) or staged (transient).
    std::string load_from(std::function<const SafeTensorInfo*(const std::string&)> find, const Ds4VisionOptions& opt);
    // transient only: convert the weights into pinned host memory now (else the first encode_gpu() does it).
    std::string stage_host(DeviceAllocator& alloc);
    // Convert the weights into the pinned buffer again: every encode checks a sampled hash of it first, because
    // staged weights that something else overwrote encode every later image to nonsense (docs/deepseek41/101).
    std::string restage_host();
    uint32_t out_dim() const;
    // device bytes this instance holds while it encodes (weights + scratch), by its own options
    uint64_t encode_bytes() const;

    // Opens the safetensors sidecar (kept mapped until the first upload).
    std::string load(const std::string& sidecar_path);
    bool loaded() const;

    // One-time weight upload (~0.93 GiB f16) + scratch for the largest block the
    // 384-token budget allows (3584 patches, 400 aligner rows) on alloc's device.
    // Call it BEFORE a language model plans its VRAM residency on the same card;
    // encode_gpu() calls it lazily otherwise. `alloc` must outlive this object.
    std::string upload(DeviceAllocator& alloc);
    static constexpr uint32_t kMaxPatches = 3584;   // > 9 * 384 (3x3 unfold of the token budget)
    static constexpr uint32_t kMaxBlocks  = 400;
    // Device bytes upload() claims (weights + scratch), for VRAM planning.
    static uint64_t device_bytes();

    // img: [3, H, W] f32 normalised, H % 14 == W % 14 == 0.
    // aligned: [n_llm_h * n_llm_w, 4096] f32, grid order. Patch count is capped
    // at kMaxPatches.
    std::string encode_gpu(DeviceAllocator& alloc, const float* img, uint32_t H, uint32_t W,
                           std::vector<float>& aligned);

    // rows: [types.size(), 4096] — control rows for START/PAD/NEWLINE/END,
    // aligned[perm[k]] for the k-th IMAGE row.
    std::string assemble_rows(const std::vector<float>& aligned, const std::vector<int32_t>& types,
                              const std::vector<int32_t>& perm, std::vector<float>& rows) const;

private:
    void fill_host();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Decode (JPEG/PNG/BMP) + official resize/pad + normalise -> encode_gpu input.
// Sets H = geom.best_h, W = geom.best_w. "" on success.
std::string ds4_load_image_mem(const void* bytes, size_t nbytes, std::vector<float>& px,
                               uint32_t& H, uint32_t& W, Ds4VisGeom& geom);
// PIL Image.resize(BICUBIC) on packed u8 RGB: the resampler the loaders here use (MiMo-V2.6's loader too).
void ds4_pil_resize(const uint8_t* src, int sw, int sh, uint8_t* dst, int dw, int dh);
std::string ds4_load_image(const std::string& path, std::vector<float>& px,
                           uint32_t& H, uint32_t& W, Ds4VisGeom& geom);
// The same decode + PIL-exact resize/pad + normalise, on a canvas planned by `plan(src_w, src_h)`.
std::string ds4_load_image_planned(const void* bytes, size_t nbytes,
                                   const std::function<Ds4VisGeom(uint32_t, uint32_t)>& plan,
                                   std::vector<float>& px, uint32_t& H, uint32_t& W, Ds4VisGeom& geom);

// Test seam (tests/unit/ds4_vision_gpu_test --probe): the encoder's own
// elementwise kernel, callable on caller-supplied buffers.
namespace ds4vis_detail {
void bias_act_f16_probe(void* queue, const float* C, const float* b, void* y_half,
                        uint32_t M, uint32_t N, int act);
}

}  // namespace ie
