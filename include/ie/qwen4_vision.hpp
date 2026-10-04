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

// Fixed Qwen3.8 vision-tower geometry (GGUF + config.json, spec §1). The same tower
// (27 blocks, hidden 1152, patch 16, merge 2) fronts Flash-Next, the 27B and the
// 35B-A3B; only the projector's OUTPUT width differs (mm.2: 2560 / 5120 / 2048),
// so that one is read from the mmproj (out_d(), P4 B45), not fixed here.
inline constexpr uint32_t kVisPatch    = 16;
inline constexpr uint32_t kVisMerge    = 2;
inline constexpr uint32_t kVisHid      = 1152;
inline constexpr uint32_t kVisHeads    = 16;
inline constexpr uint32_t kVisHeadDim  = 72;    // 1152/16
inline constexpr uint32_t kVisRot      = 36;    // head_dim/2: 18 h + 18 w freqs
inline constexpr uint32_t kVisFfn      = 4304;
inline constexpr uint32_t kVisDepth    = 27;
inline constexpr uint32_t kVisGridSide = 48;    // sqrt(2304) learned pos table
inline constexpr float    kVisLnEps    = 1e-6f;
inline constexpr float    kVisTheta    = 1e4f;
// The GPU path's patch cap (attention-kernel SLM bound; src/ops/qwen4_vision_gpu.cpp).
inline constexpr uint32_t kVisMaxPatches = 12288;

class Qwen4Vision {
public:
    Qwen4Vision();
    ~Qwen4Vision();

    // Reads mmproj GGUF (F32 / F16 / BF16 tensors), materializes all weights as f32.
    // The projector's output width comes from mm.2.weight's shape. "" on success.
    std::string load(const std::string& mmproj_path);

    // The text width the projector emits (mm.2's rows): 2560 Flash-Next, 5120 the 27B,
    // 2048 the 35B-A3B. 0 before load().
    uint32_t out_d() const noexcept { return out_d_; }

    // img: [3, H, W] f32, normalized ((x/255 - 0.5) / 0.5), H % 32 == W % 32 == 0.
    // out: [(H/32)*(W/32), out_d()] f32, block-major merged-token order.
    // "" on success.
    std::string encode(const float* img, uint32_t H, uint32_t W,
                       std::vector<float>& out) const;

    // GPU path (P3, src/ops/qwen4_vision_gpu.cpp): same contract as encode().
    // First call uploads the weights (~0.9 GiB f16) to alloc's device; they
    // stay resident for the object's lifetime. Patch grid is capped at
    // kVisMaxPatches (attention-kernel SLM bound) — cap resolution via IE_VIS_MAX_PX.
    std::string encode_gpu(DeviceAllocator& alloc, const float* img,
                           uint32_t H, uint32_t W, std::vector<float>& out);

    // P4 B45: the one-time weight upload encode_gpu's first call does, callable at
    // load so a card budget sees the tower before any image arrives. Idempotent for
    // the same allocator; a different allocator after the first is an error.
    std::string upload_gpu(DeviceAllocator& alloc);
    bool        gpu_resident() const noexcept { return gpu_ != nullptr; }
    // What upload_gpu places on the device (f16 weights + f32 biases/norms), from the
    // loaded shapes; 0 before load().
    uint64_t    gpu_weight_bytes() const noexcept;
    // The per-encode scratch encode_gpu grows to (kept for the object's lifetime) for
    // a grid of n_patches patches; the budget worst case is gpu_scratch_bytes(kVisMaxPatches).
    uint64_t    gpu_scratch_bytes(uint32_t n_patches) const noexcept;

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
    std::vector<float> mm2_w_, mm2_b_;        // [out_d_, 4608], [out_d_]
    std::vector<Block> blocks_;
    uint32_t out_d_ = 0;                      // mm.2's output width (read at load)

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

// P4 B45 (host arithmetic shared by the models that splice a tower's rows; tested
// host-only in tests/unit/qwen_vision_pos_test.cpp).
//
// The [3, T] stream-major rope positions of a forward piece [start, start + T):
// a prompt position (< n, the table's length) takes its table entry, a position
// past the prompt (decode) ropes at linear + delta on all three streams (docs/qwen4/
// 14_mrope.md §4.2). With no image (the table linear, delta 0) every stream is the
// token position: rope_imrope3 is then bit-identical to rope_partial.
inline void qwen4_mrope3_slice(const int32_t* table, uint32_t n, int32_t delta,
                               uint32_t start, uint32_t T, int32_t* out) {
    for (uint32_t s = 0; s < 3; ++s)
        for (uint32_t t = 0; t < T; ++t) {
            const uint32_t ap = start + t;
            out[size_t(s) * T + t] = ap < n ? table[size_t(s) * n + ap] : int32_t(ap) + delta;
        }
}

// A staged image span: its n rows sit at prompt positions [t0, t0 + n) and at
// rows [row0, row0 + n) of the concatenated projector output.
struct Qwen4VisSpan { uint32_t t0, n, row0; };
// Where the piece [start, start + T) meets the spans: for each overlap, the piece
// row to overwrite (dst), the staged row to copy from (src) and the row count.
// A piece may straddle an image or hold several; a 1-row decode piece inside a
// span (never produced by the engine: images are prompt rows) gives one row.
struct Qwen4VisSplice { uint32_t dst, src, n; };
inline std::vector<Qwen4VisSplice> qwen4_vis_splice_ranges(const std::vector<Qwen4VisSpan>& spans,
                                                           uint32_t start, uint32_t T) {
    std::vector<Qwen4VisSplice> out;
    for (const Qwen4VisSpan& vs : spans) {
        const uint32_t lo = start > vs.t0 ? start : vs.t0;
        const uint32_t hi = (start + T) < (vs.t0 + vs.n) ? (start + T) : (vs.t0 + vs.n);
        if (lo < hi) out.push_back({lo - start, vs.row0 + (lo - vs.t0), hi - lo});
    }
    return out;
}

}  // namespace ie
