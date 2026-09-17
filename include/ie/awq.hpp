// include/ie/awq.hpp — AWQ 4-bit dequantization (ingestion-time, host-side).
//
// AWQ stores a Linear as three tensors (group size G, default 128, pack 8):
//   qweight int32 [in, out/8]   — 8 nibbles packed along the OUTPUT axis
//   qzeros  int32 [in/G, out/8] — per-group zero codes, same packing
//   scales  fp16  [in/G, out]   — per-group fp16 scale
// Dequant:  w[i,o] = (q[i,o] - z[i//G, o]) * s[i//G, o]   (zero = raw 4-bit code).
//
// ⚠ The nibble order within each int32 is INTERLEAVED, not sequential. Logical
// column j∈[0,8) is at bit-shift `kAwqShift[j]` = {0,16,4,20,8,24,12,28}
// (reverse order [0,4,1,5,2,6,3,7]). qzeros use the SAME interleave. Getting
// this wrong runs fine and produces garbage. (vLLM awq_triton.py.)
//
// This is the AWQ→fp16 first stage of P3e (the fp16 then re-quantizes to Q4_K
// at load — a format llama.cpp can't ingest natively). Pure host, no SYCL.
#pragma once

#include "ie/quant_blocks.hpp"   // fp16_to_fp32

#include <bit>
#include <cstdint>
#include <string>

namespace ie {

// AWQ GEMM pack order: logical column j → bit shift within the int32.
inline constexpr int kAwqShift[8] = {0, 16, 4, 20, 8, 24, 12, 28};

// fp32_to_fp16 / fp16_to_fp32 live in quant_blocks.hpp (included above).

struct AwqConfig {
    std::string quant_method = "awq";   // "awq" | "gptq"
    int  bits        = 4;
    int  group_size  = 128;
    std::string version = "gemm";       // AWQ: "gemm" (interleaved) | "gemv" | "marlin"
    bool zero_point  = true;            // AWQ
    bool desc_act    = false;           // GPTQ act-order (true → g_idx permutation)
    bool sym         = true;            // GPTQ symmetric
};

// Parse a HF `config.json` body → AwqConfig (reads quantization_config).
// Returns "" on success, error text otherwise.
std::string parse_awq_config(const std::string& config_json, AwqConfig& out);

// Dequantize an AWQ (gemm pack order) Linear to fp16, row-major [in, out].
// Pointers must be suitably aligned (caller memcpys from the mmap if needed).
// `out_features` must be a multiple of 8 and `in_features` a multiple of
// `group_size`. Returns "" on success, error text otherwise.
std::string awq_dequant_to_fp16(const int32_t* qweight,   // [in, out/8]
                                const int32_t* qzeros,     // [in/G, out/8]
                                const uint16_t* scales,    // [in/G, out] fp16
                                int64_t in_features,
                                int64_t out_features,
                                int64_t group_size,
                                uint16_t* w_out);          // [in, out] fp16

// Dequantize a classic AutoGPTQ 4-bit Linear to fp16, row-major [in, out].
// GPTQ differs from AWQ (vLLM/AutoGPTQ): qweight packs 8 values along the INPUT
// axis (natural nibble order, shift = (i%8)*4); qzeros pack along the OUTPUT
// axis; the zero is applied as (z + 1); and an optional `g_idx[in]` maps each
// input row to its group (act-order). g_idx == nullptr → contiguous i/group_size.
//   w[i,o] = scales[g, o] * (q[i,o] - (z[g, o] + 1)),  g = g_idx ? g_idx[i] : i/G
// `in_features` mult of 8, `out_features` mult of 8. NOTE: validated against the
// documented format + a synthetic oracle; validate vs a real AutoGPTQ checkpoint
// (P3e Task 4) before trusting on production models.
std::string gptq_dequant_to_fp16(const int32_t* qweight,  // [in/8, out]
                                 const int32_t* qzeros,    // [in/G, out/8]
                                 const uint16_t* scales,   // [num_groups, out] fp16
                                 const int32_t* g_idx,     // [in] or nullptr
                                 int64_t in_features,
                                 int64_t out_features,
                                 int64_t group_size,
                                 uint16_t* w_out);         // [in, out] fp16

}  // namespace ie
