// include/ie/moe_q8.hpp — fully GPU-resident Q8_0-expert MoE kernels for the crown
// layer-split (Qwen35MoeSplitModel). The Q8_0 analogs of the crown's Q4_K/Q6_K
// grouped MoE kernels: router output (topk_idx/topk_w stay ON DEVICE) + per-card
// Q8_0-SoA expert planes → moe output, with NO host round-trip and NO per-expert
// host loop. Int-dot (dp4a) over the block_q8_1x activation stream, mirroring
// gemv_q8_0_soa_q8. Per token-slot tk=(t*K+k) the expert id is read from device
// topk_idx; expert weights are the SoA planes built by build_experts
// (qs[e*qs_stride + n*K + k] int8, d[e*d_stride + n*(K/32)] fp16).
#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>
#include <vector>

namespace ie {

// Stage 1: h_out[T*K, E_ffn] = silu(gate)·up. x_q8 = T block_q8_1x rows (H/32 each).
sycl::event moe_gate_up_silu_q8(sycl::queue& q, const void* x_q8,
                                const int8_t* g_qs, const uint16_t* g_d,
                                const int8_t* u_qs, const uint16_t* u_d,
                                uint64_t qs_stride, uint64_t d_stride,
                                const int32_t* topk_idx, sycl::half* h_out,
                                uint32_t T, uint32_t K, uint32_t H, uint32_t E_ffn);

// Stage 2: y_packed[T*K, H] = topk_w · (down · h). h_q8 = T*K block_q8_1x rows (E_ffn/32).
sycl::event moe_down_q8(sycl::queue& q, const void* h_q8,
                        const int8_t* d_qs, const uint16_t* d_d,
                        uint64_t qs_stride, uint64_t d_stride,
                        const int32_t* topk_idx, const sycl::half* topk_w,
                        sycl::half* y_packed, uint32_t T, uint32_t K,
                        uint32_t E_ffn, uint32_t H);

// Stage 3: y[T,H] = sum_k y_packed[(t*K+k), :].
sycl::event moe_reduce_q8(sycl::queue& q, const sycl::half* y_packed, sycl::half* y,
                          uint32_t T, uint32_t K, uint32_t H);

// ===========================================================================
// EXPERT-BATCHED Q8_0 MoE PREFILL (T>1). Weight-stationary GEMM analog of the
// per-token-slot decode kernels above: one WG per (expert, output-col-chunk)
// loops the expert's M routed token rows, reading each weight column once and
// reusing it across the tile — vs the decode kernels re-streaming each expert's
// weights ~TK/E times. Activation rows are EXPERT-SORTED (moe_gather_rows +
// quantize_q8_1 → block_q8_1x), partitioned by expert_offsets[E+1]. The per-
// (token,column) dp4a, lane→block order and SG-reduce are identical to
// moe_gate_up_silu_q8/moe_down_q8, so the result is BIT-IDENTICAL (PPL exact).
// ===========================================================================

// Stage 1 (batched): h_packed[TK, E_ffn] = silu(gate)·up over expert-sorted
// block_q8_1x rows (xq8_packed = TK rows of H/32 blocks). gate/up SoA planes:
// col n, K-index k → qs[e*qs_stride + n*H + k], scale d[e*d_stride + n*(H/32) + b].
sycl::event moe_prefill_gate_up_silu_q8(sycl::queue& q, const void* xq8_packed,
                                        const int8_t* g_qs, const uint16_t* g_d,
                                        const int8_t* u_qs, const uint16_t* u_d,
                                        uint64_t qs_stride, uint64_t d_stride,
                                        const uint32_t* expert_offsets,
                                        sycl::half* h_packed,
                                        uint32_t E, uint32_t H, uint32_t E_ffn,
                                        const std::vector<sycl::event>& deps = {});

// Stage 2 (batched): out_packed[TK, H] = sorted_w · (down · h) over expert-sorted
// block_q8_1x rows (hq8_packed = TK rows of E_ffn/32 blocks). down SoA planes:
// col n, K-index k → qs[e*qs_stride + n*E_ffn + k], scale d[e*d_stride + n*(E_ffn/32)+b].
// The routing weight is folded HERE (half(sorted_w·acc), matching decode's
// half(topk_w·acc) rounding) so the reduce is a pure sum (PPL-exact).
sycl::event moe_prefill_down_q8(sycl::queue& q, const void* hq8_packed,
                                const int8_t* d_qs, const uint16_t* d_d,
                                uint64_t qs_stride, uint64_t d_stride,
                                const uint32_t* expert_offsets,
                                const sycl::half* sorted_w,
                                sycl::half* out_packed,
                                uint32_t E, uint32_t H, uint32_t E_ffn,
                                const std::vector<sycl::event>& deps = {});

// Stage 3 (batched): y[T,H] = Σ_kslot out_packed[tk_to_packed[t,kslot], :]. The
// weight is already in out_packed → pure sum (mirrors moe_reduce_q8 but gathers
// the K expert-sorted rows of token t via the tk_to_packed inverse map).
sycl::event moe_prefill_reduce_sum(sycl::queue& q, const sycl::half* out_packed,
                                   const uint32_t* tk_to_packed, sycl::half* y,
                                   uint32_t T, uint32_t K, uint32_t H,
                                   const std::vector<sycl::event>& deps = {});

}  // namespace ie
