// include/ie/qwen4_quant.hpp — Qwen3.8-Flash-Next (qwen4exp) quant helpers.
//
// The routed-expert down bank (GGUF ffn_down_exps [640, 2560, 512] per block)
// is Q5_1: each output column = K/32 consecutive 24-B block_q5_1 (K=640 → 20
// blocks), each expert's [K, N] slice contiguous inside the [K, N, E] bank.
// The bank is host-resident and streamed per token, so the down projection
// needs a native Q5_1 GEMV (dequant-to-f16 would double the stream traffic).
//
// Also here: the BF16→F16 load-time conversion for the 24 indexer projection
// tensors, and the Q5_K→F16 host conversion for the 2 expert banks the loader
// keeps as F16. Kernels live in src/ops/gemv_q5_1.cpp.

#pragma once

#include "ie/quant_blocks.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <sycl/sycl.hpp>

namespace ie {

// W5A16 GEMV: y[1,N] = A[1,K] @ W[K,N].
//   A: [1, K] FP16 row-major (the activation)
//   W: [K, N] Q5_1 column-packed (column n = K/32 consecutive block_q5_1;
//      w = d*q + m, q = 5-bit unsigned from ql nibble + qh bit)
//   y: [1, N] FP16; fp32 accumulate. K % 32 == 0.
// Mirrors gemv_q4_0's WG/SG shape (correctness-first; tiling later).
sycl::event gemv_q5_1(sycl::queue& q,
                      const sycl::half* A, const void* W_packed,
                      sycl::half* y,
                      uint32_t K, uint32_t N,
                      const std::vector<sycl::event>& deps = {});

// Grouped expert-pick variant (spec-verify MoE): one launch for ALL picks;
// per (pick, column) the solo body runs verbatim -> bit-identical to the
// per-pick gemv_q5_1 calls. jobs = [P x 3] int32 {vt, slot, ycell}.
sycl::event gemv_q5_1_grouped(sycl::queue& q,
                      const sycl::half* h_rows,
                      const uint8_t* slot_base, uint64_t slot_bytes,
                      uint64_t down_off,
                      const int32_t* jobs,
                      sycl::half* ystage, uint32_t y_stride,
                      uint32_t K, uint32_t N, uint32_t P,
                      const std::vector<sycl::event>& deps = {});

// Grouped-tiles down-proj for the prefill expert-major MoE: one launch per
// layer; per (job, column) each row runs the gemv_q5_1 body verbatim ->
// bit-identical per row to the per-expert gemm_q5_1/gemv_q5_1 path.
// tile_jobs = [J x 4] int32 {x_row0, rows, slot, y_row0} over the mega
// h-rows / y-rows buffers.
sycl::event gemv_q5_1_grouped_tiles(sycl::queue& q,
                      const sycl::half* h_mega,
                      const uint8_t* slot_base, uint64_t slot_bytes,
                      uint64_t down_off,
                      const int32_t* tile_jobs,
                      sycl::half* y_mega,
                      uint32_t K, uint32_t N, uint32_t J,
                      const std::vector<sycl::event>& deps = {});

// Batched-rows variant: y[M,N] = A[M,K] @ W[K,N]. Same layout; per-row
// numerics identical to gemv_q5_1 (each weight block dequantized once per
// column and reused across the SLM-staged rows). Any M (tiled internally).
sycl::event gemm_q5_1(sycl::queue& q,
                      const sycl::half* A, const void* W_packed,
                      sycl::half* y,
                      uint32_t M, uint32_t K, uint32_t N,
                      const std::vector<sycl::event>& deps = {});

// ---------------------------------------------------------------------------
// Expert-slice addressing for a host-resident [K, N, E] Q5_1 bank.
// Expert e's [K, N] column-packed slice is contiguous at byte offset
//   e * (K/32) * sizeof(block_q5_1) * N        (= e * (K/32)*24*N).
// ---------------------------------------------------------------------------
inline uint64_t q5_1_expert_stride_bytes(uint32_t K, uint32_t N) noexcept {
    static_assert(sizeof(block_q5_1) == 24, "block_q5_1 must be 24 bytes");
    return uint64_t(K / 32) * sizeof(block_q5_1) * N;
}

inline const void* q5_1_expert_slice(const void* bank, uint32_t K, uint32_t N,
                                     uint32_t e) {
    if (K % 32 != 0) {   // rows must be whole Q5_1 blocks; violation = loader bug
        std::fprintf(stderr, "q5_1_expert_slice: K=%u not a multiple of 32\n", K);
        std::abort();
    }
    return static_cast<const uint8_t*>(bank) + q5_1_expert_stride_bytes(K, N) * e;
}

// ---------------------------------------------------------------------------
// BF16 → F16 host conversion (indexer projections, load time).
// bf16 is the top 16 bits of fp32: promote src[i]<<16 to fp32, then
// round-to-nearest-even to IEEE binary16 (any NaN in → NaN out).
// ---------------------------------------------------------------------------
void convert_bf16_to_f16(const uint16_t* src, sycl::half* dst, size_t n);

// ---------------------------------------------------------------------------
// Q5_K → F16 host conversion (the 2 expert banks the loader keeps as F16).
// packed = whole block_q5_K super-blocks, n % 256 == 0; out[i] = f16 RTNE of
// ie::ref::dequant_q5_K's fp32 value (y = d*sc*q - dmin*m per 32-elem group).
// ---------------------------------------------------------------------------
void convert_q5_k_to_f16_buffer(const void* packed, size_t n, sycl::half* out);

}  // namespace ie
