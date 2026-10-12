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

// R <= kCpuExpertRows rows through ONE expert: row r's out[r * H] is bit-identical to cpu_expert_mxfp4 on xs[r] (the
// same block order, FMAs and final sum per row), but each weight block is read and decoded once for all R rows
// instead of once per row -- the per-row cost was the ~19 MB weight read. `scratch` holds R * 2 * EF floats.
constexpr uint32_t kCpuExpertRows = 8;
// B66, on by default: the SwiGLU (spans of 64 entries) and the worker's fp16 / fp32 row stores run on the whole team instead
// of one thread; every entry is the same function of the same input, so the results are bit-identical (MiMo A-B-A:
// -7.6 % ms/token on top of the 1 ms blocktime). IE_CPU_MXFP4_TEAM=0 restores v0.2.21's serial form.
bool cpu_mxfp4_team();
void cpu_expert_mxfp4_rows(const void* slot, const Ds4SlotLayout& lay, const float* const* xs, uint32_t R,
                           float* scratch, float* out, float swiglu_limit, int nthreads);

// B66: a list of experts in ONE team region. Each thread takes an even share of every expert's gate / up rows and their
// SwiGLU entries, then (one barrier) its share of the down rows, and hands those output entries to `store`; the next
// expert's gate / up go to the other half of the scratch, so one barrier per expert suffices. Every entry is computed as
// cpu_expert_mxfp4 (R = 1) / cpu_expert_mxfp4_rows (R > 1) computes it -- the same row dot per output row, the same
// scalar SwiGLU per entry -- so the results are bit-identical; the per-expert path opens ~5 regions per expert.
struct CpuExpertJob { const void* slot; const float* xs[kCpuExpertRows]; uint32_t R; uint32_t out_row; };
// entries [k0, k1) of output row `row` (job.out_row + r), fp32 at src[0 .. k1 - k0)
using CpuRowStore = void (*)(void* ctx, uint32_t row, uint32_t k0, uint32_t k1, const float* src);
constexpr size_t cpu_experts_fused_scratch(uint32_t H, uint32_t EF) { return size_t(2) * kCpuExpertRows * 2 * EF + size_t(kCpuExpertRows) * H; }
void cpu_experts_mxfp4_fused(const CpuExpertJob* jobs, size_t n, const Ds4SlotLayout& lay, float* scratch,
                             float swiglu_limit, int nthreads, CpuRowStore store, void* ctx);

}  // namespace ie
