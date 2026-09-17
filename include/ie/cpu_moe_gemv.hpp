// include/ie/cpu_moe_gemv.hpp — host Q4_K/Q5_K GEMV for GLM-5.3-Flash CPU-miss.
//
// Integer path (Q8_K activations) is the measured DRAM-speed lane (~55 GB/s
// on 265K, faster than PCIe fill). f32 path matches ie::ref dequant GEMV
// (quality reference; too slow to replace PCIe).
#pragma once

#include "ie/quant_blocks.hpp"

#include <cstdint>

namespace ie {

// ggml intermediate activation block (not a GGUF on-disk type).
struct alignas(4) block_q8_K {
    float   d;
    int8_t  qs[256];
    int16_t bsums[16];
};
static_assert(sizeof(block_q8_K) == 292, "block_q8_K");

// G1b (2026-09-02): the same 256-element activation block with ONE SCALE PER
// 32 ELEMENTS (matching the Q4_K/Q5_K sub-block granularity), so an outlier
// channel coarsens 31 neighbours instead of 255. block_q8_K measured 0.08-5.5%
// rms error per expert on heavy-tailed inputs (oracle); this is the remedy.
struct alignas(4) block_q8_Ks {
    float   d[8];       // per-32 scales
    int8_t  qs[256];
    int16_t bsums[16];  // per-16 sums of qs (two per sub-block)
};
static_assert(sizeof(block_q8_Ks) == 320, "block_q8_Ks");
void cpu_quantize_q8_Ks(const float* x, block_q8_Ks* y, int k);

void cpu_quantize_q8_K(const float* x, block_q8_K* y, int k);

// Column-major packed W: nblocks = K/256 super-blocks per output column.
void cpu_gemv_q4k_f32(const float* x, const block_q4_K* W, float* y,
                      uint32_t K, uint32_t N);
void cpu_gemv_q4k_q8(const block_q8_K* x, const block_q4_K* W, float* y,
                     uint32_t K, uint32_t N);
void cpu_gemv_q5k_f32(const float* x, const block_q5_K* W, float* y,
                      uint32_t K, uint32_t N);
void cpu_gemv_q5k_q8(const block_q8_K* x, const block_q5_K* W, float* y,
                     uint32_t K, uint32_t N);

// One decode expert: y[H] = down(swiglu_clamp(gate(x), up(x))).
// gate/up are Q4_K [H, EF] packed, down is Q5_K [EF, H] packed.
void cpu_moe_expert_q8(const float* x,
                       const block_q4_K* gate, const block_q4_K* up,
                       const block_q5_K* down,
                       uint32_t H, uint32_t EF, float clamp, float* y);
// Same, with the OpenMP team sized by the caller (the 8-thread entry point above
// is unchanged): a worker pinned to n cores passes n.
void cpu_moe_expert_q8_nt(const float* x,
                          const block_q4_K* gate, const block_q4_K* up,
                          const block_q5_K* down,
                          uint32_t H, uint32_t EF, float clamp, float* y, int nthreads);
// Same expert with per-32 activation scales (block_q8_Ks) for gate/up AND the
// down input; ~8 extra float multiply-adds per 256-block over the q8 path.
void cpu_moe_expert_q8s_nt(const float* x,
                           const block_q4_K* gate, const block_q4_K* up,
                           const block_q5_K* down,
                           uint32_t H, uint32_t EF, float clamp, float* y, int nthreads);

}  // namespace ie
