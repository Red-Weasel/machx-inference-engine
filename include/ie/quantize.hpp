// include/ie/quantize.hpp — host fp32→GGUF-quant encoders (import-time, P3e).
//
// The engine had only DEcoders (dequant_ref.hpp / dequant_kernels.cpp); import
// needs the ENcoder so HF/AWQ weights become a real quantized GGUF the loader
// accepts (token_embd/output require Q4_K/Q6_K, not F16). Faithful port of
// ggml-quants.c so the output is bit-compatible with llama.cpp's quantizer.
#pragma once

#include "ie/quant_blocks.hpp"

#include <cstdint>

namespace ie {

// Quantize `k` fp32 values (k % 256 == 0) into k/256 Q6_K blocks.
// Port of ggml-quants.c quantize_row_q6_K_ref (+ make_qx_quants rmse_type=1).
void quantize_row_q6_K(const float* x, block_q6_K* out, int64_t k);

// Quantize `k` fp32 values (k % 256 == 0) into k/256 Q4_K blocks.
// Port of ggml-quants.c quantize_row_q4_K_ref (+ make_qkx2_quants).
void quantize_row_q4_K(const float* x, block_q4_K* out, int64_t k);

}  // namespace ie
