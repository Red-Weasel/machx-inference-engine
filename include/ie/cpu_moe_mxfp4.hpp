// include/ie/cpu_moe_mxfp4.hpp — the host side of the V4.1 decode miss path (Phase 13,
// docs/deepseek41/34): one routed expert computed on the CPU from the pinned host arena's slot,
// so a miss need not cross PCIe. The slot is exactly what the GPU reads (ds41_slot_layout):
// for each of gate / up / down, a nibble plane [N][K/2] in the engine's SoA order — 16-byte
// blocks, one per 32 consecutive k of a row, byte j holding k = j in its low nibble and
// k = j + 16 in its high nibble — and an E8M0 scale plane [N][K/32], one byte per block.
// A weight is mxfp4_nibble_int(nibble) * mxfp4_e8m0_half(scale): the same two formulas as
// src/ops/deepseek4_experts.cpp, transcribed here; the unit test holds them equal.
#pragma once
#include "ie/expert_stream.hpp"

#include <cstdint>

namespace ie {

// The nibble magnitude table {0,1,2,3,4,6,8,12} with the sign in bit 3, and the E8M0 half-scale.
int   cpu_mxfp4_nibble_int(uint32_t nb);
float cpu_mxfp4_e8m0_half(uint8_t e);

// w[n*K + k] fp32 from the planes -- the scalar reference for the tests, not for speed.
void cpu_dequant_mxfp4_ref(const uint8_t* qs, const uint8_t* e, float* w, uint32_t K, uint32_t N);

// y[n] = sum_k w[n][k] * x[k] over the planes, fp32 accumulation; rows split over `nthreads`
// (an OpenMP team of that size; 0 = the runtime's default). AVX2 + FMA when compiled with them.
void cpu_gemv_mxfp4_f32(const float* x, const uint8_t* qs, const uint8_t* e, float* y,
                        uint32_t K, uint32_t N, int nthreads);

// One expert from its slot: gate and up over x[H], silu with the clamp (`swiglu_limit`, 0 = none)
// times up, then down -> out[H]. `scratch` holds 2 * EF floats. The three GEMVs each split their
// rows over the team; `nthreads` as above.
void cpu_expert_mxfp4(const void* slot, const Ds4SlotLayout& lay, const float* x, float* scratch,
                      float* out, float swiglu_limit, int nthreads);

}  // namespace ie
