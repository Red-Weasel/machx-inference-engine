// include/ie/moe_qwen3.hpp — qwen3moe-owned generalized int-dot MoE down kernels.
//
// The crown's int-dot stage-2 down kernels (moe_fused.cpp,
// moe_prefill_down_packed_q{4,6}k_q8) are HARD-LOCKED to E_ffn == 512: the SG
// (16 lanes × one q8 block × 32 elems = 512) covers the FULL K=E_ffn reduction
// in one pass, and the SLM stage / lane→block indexing bake in q8_per_row==16.
// Qwen3-Coder-30B has E_ffn=768 (q8_per_row=24, blocks_per_col=3), so it fell
// back to the fp16-activation down kernels (≈83% of qwen3moe prefill GPU time).
//
// These functions GENERALIZE the same W4A8 int-dot technique to any E_ffn that
// is a multiple of 256: each lane walks the q8 blocks {lane, lane+16, …} <
// E_ffn/32, deriving the Q-block b_in = j/8 and 32-elem sub-group sb = j%8 from
// the q8-block index j; the SG-reduce then sums every lane's partials over the
// full K. At E_ffn=512 (n_blk_per_lane==1) this reduces bit-for-bit to the
// crown kernel. moe_fused.cpp is untouched (P2/P3 iron rule); dev_fp16_to_fp32
// is copied (copy-not-hoist). ESIMD-safe: plain SLM + vec loads, no block2d.
#pragma once

#include "ie/quant_blocks.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <vector>

namespace ie {

// y_out[T*K_top, H] = down(h_in) per expert, int-dot over a block_q8_1s stream
// of h_packed (quantize_q8_1s once per layer). Same layout/semantics as
// moe_prefill_down_packed_q6k_q8 but for any E_ffn % 256 == 0.
sycl::event moe_prefill_down_q6k_q8_gen(sycl::queue& q,
                                        const void* hq8_packed,
                                        const void* down_W,
                                        const uint32_t* expert_offsets,
                                        sycl::half* out_packed,
                                        uint32_t E, uint32_t H, uint32_t E_ffn,
                                        uint64_t expert_stride_bytes, bool soa,
                                        const std::vector<sycl::event>& deps = {});

sycl::event moe_prefill_down_q4k_q8_gen(sycl::queue& q,
                                        const void* hq8_packed,
                                        const void* down_W,
                                        const uint32_t* expert_offsets,
                                        sycl::half* out_packed,
                                        uint32_t E, uint32_t H, uint32_t E_ffn,
                                        uint64_t expert_stride_bytes, bool soa,
                                        const std::vector<sycl::event>& deps = {});

// h_packed[T*K_top, E_ffn] = silu(gate(x)) * up(x) per expert, int-dot W6A8
// over a block_q8_1s stream of the expert-sorted x_packed (quantize_q8_1s once
// per layer). The Q6_K analog of moe_fused.cpp:moe_prefill_gate_up_silu_q4k_q8
// (Q6_K gate/up banks, e.g. bartowski Q6_K of Qwen3-30B-A3B / Tongyi-DeepResearch).
// Contracts over H; requires H % 512 == 0. soa selects the per-expert SoA repack.
sycl::event moe_prefill_gate_up_silu_q6k_q8(sycl::queue& q,
                                            const void* xq8_packed,
                                            const void* gate_W, const void* up_W,
                                            const uint32_t* expert_offsets,
                                            sycl::half* h_packed,
                                            uint32_t E, uint32_t H, uint32_t E_ffn,
                                            uint64_t expert_stride_bytes, bool soa,
                                            const std::vector<sycl::event>& deps = {});

}  // namespace ie
