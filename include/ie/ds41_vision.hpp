// include/ie/ds41_vision.hpp — DeepSeek-V4.1-Flash vision: the image plan and the token span.
//
// Reference: the checkpoint's own inference/image_processor.py + vision.py. The tower itself is the V4 Vision-Exp
// tower (include/ie/ds4_vision.hpp) at other sizes: a 5120-wide LM, a 1,024-token image budget, no pad row, and
// the weights in the checkpoint's shards (`vision.*`, `aligner.*`, `image_start` / `image_newline` / `image_end`).
// An image is    [IMAGE_START] + ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_END]
// and every one of those positions carries `image_token_id` in the reference's input_ids.
// Spec + gate: docs/deepseek41/94_PHASE55_VISION_TOWER_2026-09-18.md.

#pragma once

#include "ie/ds4_vision.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace ie {

inline constexpr uint32_t kDs41VisMaxTok = 1024;      // vision_max_n_token
inline constexpr uint32_t kDs41VisMinPx  = 295936;    // vision_min_pixels (544 * 544)
inline constexpr uint32_t kDs41VisMaxPatches = 9216;  // 3x3 patches per token, rounded up to the attention tile
inline constexpr uint32_t kDs41VisAttnTile   = 2304;  // query rows per attention GEMM (scores 2304 x 9216 f32 = 81 MiB)
inline constexpr float    kDs41VisProbScale  = 256.0f; // f16 attention probabilities over up to 9,216 keys (docs/deepseek41/94)

// plan_image_grid(): the canvas (multiples of 14), the patch grid and the token grid for a source of w x h px.
// vision_max_wh_ratio is null in this checkpoint: the image is always contained and grey-padded (`plain` false).
Ds4VisGeom ds41_vis_plan(uint32_t width, uint32_t height);

// num_image_tokens(): positions the span occupies.
inline uint32_t ds41_vis_tokens(uint32_t n_llm_h, uint32_t n_llm_w) { return n_llm_h * (n_llm_w + 1) + 2; }

// image_token_types() in the tower's row enum (Ds4VisType, no pad rows); `perm` is the identity (reading order).
void ds41_vis_types(uint32_t n_llm_h, uint32_t n_llm_w, std::vector<int32_t>& types, std::vector<int32_t>& perm);

// The source size from the header alone (no decode): what the plan, and so the span's length, needs.
std::string ds41_image_size(const void* bytes, size_t nbytes, uint32_t& width, uint32_t& height);

// Decode (JPEG / PNG / BMP) + the official contain / grey-pad / normalise, ready for Ds4Vision::encode_gpu.
std::string ds41_load_image_mem(const void* bytes, size_t nbytes, std::vector<float>& px,
                                uint32_t& H, uint32_t& W, Ds4VisGeom& geom);

// The tower options for this checkpoint: transient (the cards are full at steady state).
Ds4VisionOptions ds41_vision_options(uint32_t lm_hidden);

}  // namespace ie
