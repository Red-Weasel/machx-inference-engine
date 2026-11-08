// include/ie/qwen4_bank.hpp -- Qwen3.8-Flash-Next (qwen4exp) routed-expert banks in the quant types the community builds
// carry beyond Q4_K gate/up and Q5_1 down (2026-11-07, Qwen 4 day-one readiness, docs/research/2026-11-07-qwen4-readiness-plan.md):
//   Q8_0  every down bank of Unsloth's UD-Q5_K_XL / UD-Q6_K_XL / Q8_0 builds, and the Q8_0 build's gate/up banks
//   Q5_K  UD-Q5_K_XL gate/up (47 of 48 layers; our UD-Q4_K_XL carries one such layer)
//   Q6_K  UD-Q6_K_XL gate/up
//   Q5_0  bartowski's down banks (a 640-wide row is never a K-quant); Q4_0 (ggml-org, bartowski Q4_0)
// Before this every such bank was dequantised at load to a pinned host F16 copy (2 bytes an element, `gemv_fp16` per pick):
// the Q8_0 file needed ~225 GiB of RAM and ran at 0.7 tok/s for the GitHub issue-3 reporter. These kernels stream the raw
// bank bytes like the Q4_K / Q5_1 families do.
//
// A16 kernels: f16 activations, weights dequantised in registers to fp32 exactly as include/ie/dequant_ref.hpp does
// (w = d*q; (q-8)*d; (q-16)*d; d*sc*q - dmin*m; d*sc*q), fp32 accumulate per lane, one subgroup reduce, a half store --
// the gemv_q5_1 family's shape. ONE column walk per kind (src/ops/qwen4_bank.cpp) is shared by the solo GEMV, the grouped
// pick kernels, the grouped tiles and the batched rows, so every variant is bit-identical to the solo kernel per output
// (the lossless contract the Q5_1 / Q4_K families keep; ie-qwen4-bank-test checks it on synthetic banks against an fp64
// CPU reference).
//
// Weight layout: W[K, N] column-packed -- output column n = K/block consecutive blocks (the GGUF [K, N, E] bank's order;
// expert e's [K, N] slice is contiguous, e * (K/block) * sizeof(block) * N bytes in). K must be a whole number of blocks.
#pragma once

#include "ie/dtype.hpp"

#include <cstdint>
#include <vector>

#include <sycl/sycl.hpp>

namespace ie {

enum class BankQ : uint8_t { kQ8_0, kQ5_K, kQ6_K, kQ5_0, kQ4_0 };

// The kinds these kernels stream; false for any other dtype or a K that is not a whole number of blocks (32 for the
// 32-block types, 256 for the K-quants).
bool qwen4_bank_kind(DType dt, uint32_t K, BankQ& out) noexcept;
const char* qwen4_bank_name(BankQ q) noexcept;
uint64_t qwen4_bank_col_bytes(BankQ q, uint32_t K) noexcept;   // one output column's bytes

// y[N] = A[K] @ W[K, N].  A: f16 row; y: f16; fp32 accumulate.
sycl::event qwen4_bank_gemv(sycl::queue& q, BankQ kind,
                            const sycl::half* A, const void* W, sycl::half* y,
                            uint32_t K, uint32_t N,
                            const std::vector<sycl::event>& deps = {});

// Grouped gate+up over an expert-pick job list, one launch for all picks x both mats (gemv_q4_K_q8s_grouped_dual's
// contract): jobs = [P x 3] int32 {vt, slot, ycell}; pick p reads the f16 activation row x + vt*K and the slices at
// slot_base + slot*slot_bytes + gate_off / up_off; writes g_out[p*N ..] and u_out[p*N ..].
// slot_tab (every grouped pick kernel here; null = the slot arithmetic): an address table [slot][3] {gate, up, down}, the
// job's slot an index into it (a VRAM cache slot or an expert's slices in pinned host memory -- device routing's
// read-through miss). The dual kernels read entries 0 / 1 of the slot's row; the single-matrix qwen4_bank_gemv_grouped
// reads entry 0 of the row it is handed, so the down call passes slot_tab + 2.
sycl::event qwen4_bank_gemv_grouped_dual(sycl::queue& q, BankQ kind,
                            const sycl::half* x,
                            const uint8_t* slot_base, uint64_t slot_bytes,
                            uint64_t gate_off, uint64_t up_off,
                            const int32_t* jobs,
                            sycl::half* g_out, sycl::half* u_out,
                            uint32_t K, uint32_t N, uint32_t P,
                            const std::vector<sycl::event>& deps = {},
                            const uint8_t* const* slot_tab = nullptr);

// The grouped dual gate/up with swiglu fused into the epilogue (2026-11-08, the decode host gap, option 1): the same
// job list, h_out[p*N ..] = silu(f16(g)) * f16(u) exactly as swiglu() computes it from the dual kernel's f16 outputs.
// Bit-identical to the dual kernel followed by swiglu(); A16 for every BankQ, the int-dot (A8) form for Q5_K / Q6_K.
sycl::event qwen4_bank_gemv_grouped_dual_swiglu(sycl::queue& q, BankQ kind,
                            const sycl::half* x,
                            const uint8_t* slot_base, uint64_t slot_bytes,
                            uint64_t gate_off, uint64_t up_off,
                            const int32_t* jobs,
                            sycl::half* h_out,
                            uint32_t K, uint32_t N, uint32_t P,
                            const std::vector<sycl::event>& deps = {},
                            const uint8_t* const* slot_tab = nullptr);
sycl::event qwen4_bank_gemv_grouped_dual_q8s_swiglu(sycl::queue& q, BankQ kind,
                            const void* x_q8s,
                            const uint8_t* slot_base, uint64_t slot_bytes,
                            uint64_t gate_off, uint64_t up_off,
                            const int32_t* jobs,
                            sycl::half* h_out,
                            uint32_t K, uint32_t N, uint32_t P,
                            const std::vector<sycl::event>& deps = {},
                            const uint8_t* const* slot_tab = nullptr);

// Grouped down over the same job list (gemv_q5_1_grouped's contract): pick p reads h_rows + p*K and the slice at
// slot_base + slot*slot_bytes + off; writes ystage + ycell*y_stride.
sycl::event qwen4_bank_gemv_grouped(sycl::queue& q, BankQ kind,
                            const sycl::half* h_rows,
                            const uint8_t* slot_base, uint64_t slot_bytes,
                            uint64_t off,
                            const int32_t* jobs,
                            sycl::half* ystage, uint32_t y_stride,
                            uint32_t K, uint32_t N, uint32_t P,
                            const std::vector<sycl::event>& deps = {},
                            const uint8_t* const* slot_tab = nullptr);

// Grouped tiles for the prefill expert-major MoE (gemm_f16_grouped_tiles' contract): tile_jobs = [J x 4] int32
// {x_row0, rows (<= 16), slot, y_row0}; row t of job j reads A_mega + (x_row0 + t)*K and writes out + (y_row0 + t)*N;
// mats = 1 (slice off0 -> out0) or 2 (off0 -> out0 and off1 -> out1, gate and up). Per row == the solo kernel.
sycl::event qwen4_bank_gemv_grouped_tiles(sycl::queue& q, BankQ kind,
                            const sycl::half* A_mega,
                            const uint8_t* slot_base, uint64_t slot_bytes,
                            uint64_t off0, uint64_t off1,
                            const int32_t* tile_jobs,
                            sycl::half* out0, sycl::half* out1,
                            uint32_t K, uint32_t N, uint32_t J, uint32_t mats,
                            const std::vector<sycl::event>& deps = {});

// Batched rows: y[M, N] = A[M, K] @ W[K, N], any M (16-row chunks inside). Per row == the solo kernel.
sycl::event qwen4_bank_gemm(sycl::queue& q, BankQ kind,
                            const sycl::half* A, const void* W, sycl::half* y,
                            uint32_t M, uint32_t K, uint32_t N,
                            const std::vector<sycl::event>& deps = {});

// Int-dot variants for the K-quant gate/up banks (2026-11-07; docs/research/2026-11-07-flashnext-q6-profile.md: the A16
// Q6_K grouped-dual kernel ran ALU-bound at 103 GB/s, 12.5 ms a decode step on the UD-Q6_K_XL, where the Q4_K int-dot kernel
// runs 2.6 ms at 338 GB/s). x_q8s = block_q8_1s rows (quantize_q8_1s: 32 int8 + d + the two 16-element half-sums), the
// Q4_K path's activations; the sub-block dots run as packed dp4a (Q5_K: q5 = nibble | qh bit << 4, minus dmin*m*sum(x) per
// 32-sub-block; Q6_K: q6 = nibble | 2 qh bits << 4, the -32 offset folded through the half-sums). The numerics are the Q4_K
// int-dot path's (int8-rounded activations, ~0.4 % RMS), NOT bit-identical to the A16 kernels; dual == solo per pick.
// qwen4_bank_has_q8dot = Q5_K or Q6_K; the other kinds have no such variant (an empty event is returned).
bool qwen4_bank_has_q8dot(BankQ kind) noexcept;
sycl::event qwen4_bank_gemv_q8s(sycl::queue& q, BankQ kind,
                            const void* x_q8s, const void* W, sycl::half* y, uint32_t K, uint32_t N,
                            const std::vector<sycl::event>& deps = {});
sycl::event qwen4_bank_gemv_grouped_dual_q8s(sycl::queue& q, BankQ kind,
                            const void* x_q8s, const uint8_t* slot_base, uint64_t slot_bytes,
                            uint64_t gate_off, uint64_t up_off, const int32_t* jobs,
                            sycl::half* g_out, sycl::half* u_out, uint32_t K, uint32_t N, uint32_t P,
                            const std::vector<sycl::event>& deps = {},
                            const uint8_t* const* slot_tab = nullptr);
// The prefill tiles (qwen4_bank_gemv_grouped_tiles' tile_jobs contract, gate and up): x_q8s_mega = the gathered Q8_1s rows;
// row t of job j reads row x_row0 + t. Per row bit-identical to qwen4_bank_gemv_q8s (the lane's packed weight words are
// built once a super-block and dotted against every row).
sycl::event qwen4_bank_gemv_grouped_tiles_q8s(sycl::queue& q, BankQ kind,
                            const void* x_q8s_mega, const uint8_t* slot_base, uint64_t slot_bytes,
                            uint64_t gate_off, uint64_t up_off, const int32_t* tile_jobs,
                            sycl::half* g_out, sycl::half* u_out, uint32_t K, uint32_t N, uint32_t J,
                            const std::vector<sycl::event>& deps = {});
// The int8-DPAS (XMX) grouped tiles over Q8_0 / Q5_K / Q6_K gate/up banks (src/ops/qwen4_bank_dpas.cpp; gemm_q4k_dpas.cpp's
// shape): the same tile-job contract and Q8_1s rows; the fp32 scale chain runs K-span-ascending (the dpas-form order, not the
// s-form kernels'), so rows are NOT bit-identical to qwen4_bank_gemv_q8s -- prefill numerics, gated by chunked PPL like the
// Q4_K path. N % 16 == 0. qwen4_bank_has_q8dpas = the three kinds above.
bool qwen4_bank_has_q8dpas(BankQ kind) noexcept;
// The DOWN tiles for Q8_0 down banks (every Unsloth build): the exact-int f16-DPAS form of gemv_q5_1_f16d_grouped_tiles
// without its min term -- B = f16(q) exact, A = the f16 activation rows (x, xbase-indexed), d_b per 32-block in fp32,
// block-ascending per output. Rows beyond a job's `rows` within its last 8-row group are READ from x (not written): x must
// hold ceil(rows / 8) * 8 rows past each job's x_row0, as moe_hT_ does. N % 16 == 0, K % 32 == 0. Q8_0 only for now.
bool qwen4_bank_has_down_f16d(BankQ kind) noexcept;
sycl::event qwen4_bank_gemv_grouped_tiles_down_f16d(sycl::queue& q, BankQ kind,
                            const sycl::half* x, const uint8_t* slot_base, uint64_t slot_bytes,
                            uint64_t down_off, const int32_t* tile_jobs, sycl::half* y,
                            uint32_t K, uint32_t N, uint32_t J,
                            const std::vector<sycl::event>& deps = {});
sycl::event qwen4_bank_gemv_grouped_tiles_q8d(sycl::queue& q, BankQ kind,
                            const void* x_q8s_mega, const uint8_t* slot_base, uint64_t slot_bytes,
                            uint64_t gate_off, uint64_t up_off, const int32_t* tile_jobs,
                            sycl::half* g_out, sycl::half* u_out, uint32_t K, uint32_t N, uint32_t J,
                            const std::vector<sycl::event>& deps = {});

}  // namespace ie
