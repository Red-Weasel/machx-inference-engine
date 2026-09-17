// include/ie/dequant.hpp — device-side dequant kernels.
//
// Each function launches one SYCL kernel that dequantizes `n_elements` worth
// of packed input into a contiguous FP16 device buffer. `n_elements` MUST be
// a multiple of the block size for the format (32 for Q8_0; 256 for K-quants).
//
// These standalone dequant kernels are useful for tests, validation, and
// "spill-to-fp16" fallback paths. The performance-critical path in the
// engine is dequant-fused-into-XMX (Phase 3), not these.

#pragma once

#include <sycl/sycl.hpp>

#include <cstddef>

namespace ie {

sycl::event dequant_q8_0(sycl::queue& q, const void* packed_in,
                         sycl::half* out, size_t n_elements,
                         const std::vector<sycl::event>& deps = {});
sycl::event dequant_q4_K(sycl::queue& q, const void* packed_in,
                         sycl::half* out, size_t n_elements,
                         const std::vector<sycl::event>& deps = {});
sycl::event dequant_q5_K(sycl::queue& q, const void* packed_in,
                         sycl::half* out, size_t n_elements,
                         const std::vector<sycl::event>& deps = {});
sycl::event dequant_q6_K(sycl::queue& q, const void* packed_in,
                         sycl::half* out, size_t n_elements,
                         const std::vector<sycl::event>& deps = {});

// Transposed full-matrix dequant for the prefill fp16-GEMM path
// (docs/prefill_attack_plan_2026-06-09.md, E1).  Input W is the usual
// GGUF packed layout: N output columns, each with K/256 super-blocks along K.
// Output is B^T fp16 [K, N] row-major (out[k*N + n]) — the layout gemm_fp16
// consumes directly.  Requires K % 256 == 0 and N % 64 == 0.
sycl::event dequant_q4_K_to_Bt(sycl::queue& q, const void* packed_in,
                               sycl::half* out, uint32_t K, uint32_t N,
                               const std::vector<sycl::event>& deps = {});
sycl::event dequant_q6_K_to_Bt(sycl::queue& q, const void* packed_in,
                               sycl::half* out, uint32_t K, uint32_t N,
                               const std::vector<sycl::event>& deps = {});
// MXFP4 (OpenAI gpt-oss MoE expert format): E8M0 half-scaled shared exponent +
// 32 FP4 nibbles via a 16-value signed LUT. Bit-exact with llama
// ggml_e8m0_to_fp32_half + kvalues_mxfp4. n_elements % 32 == 0.
sycl::event dequant_mxfp4(sycl::queue& q, const void* packed_in,
                          sycl::half* out, size_t n_elements,
                          const std::vector<sycl::event>& deps = {});
// MXFP4 → B^T fp16 [K,N] (out[k*N+n]) — feeds the per-expert oneDNN MoE GEMM,
// mirroring dequant_q4_K_to_Bt. K % 32 == 0, N % 64 == 0.
sycl::event dequant_mxfp4_to_Bt(sycl::queue& q, const void* packed_in,
                                sycl::half* out, uint32_t K, uint32_t N,
                                const std::vector<sycl::event>& deps = {});
// MXFP4 SoA (load-time repack: aligned qs_plane[N*(K/2)] + e_plane[N*(K/32)]) →
// B^T fp16 [K,N].  Bit-identical to dequant_mxfp4_to_Bt (the repack is a pure split
// of the 17-byte block into its e + qs planes).  Prefill path for gpt-oss experts.
// K%32==0 (N need NOT be %64 — the launch rounds the dim-1 global up to WG and a tail
// guard skips the padding; required for TP column slices, N=efc).  Declared in
// dequant.hpp; implemented in ops/gemv_mxfp4.cpp.
sycl::event dequant_mxfp4_soa_to_Bt(sycl::queue& q,
                                    const uint8_t* qs_plane, const uint8_t* e_plane,
                                    sycl::half* out, uint32_t K, uint32_t N,
                                    const std::vector<sycl::event>& deps = {});
// Q6_K SoA (per-expert MoE repack, repack_moe_q6k_soa_host) → B^T fp16 [K,N].
// `soa_bank_expert_slice` is the per-expert region base (bank + e*expert_stride);
// the kernel reads the four de-interleaved planes (ql|qh|scales|d) at the
// q6k_soa_offsets within that region. Bit-identical output to dequant_q6_K_to_Bt
// on the same weight (the SoA repack is a pure byte reorder). K%256==0, N%64==0.
sycl::event dequant_q6_K_soa_to_Bt(sycl::queue& q, const void* soa_bank_expert_slice,
                                   sycl::half* out, uint32_t K, uint32_t N,
                                   const std::vector<sycl::event>& deps = {});
// Q4_K SoA (per-expert MoE repack, repack_moe_q4k_soa_host) → B^T fp16 [K,N].
// `soa_bank_expert_slice` = per-expert region base (bank + e*expert_stride). Reads
// the three de-interleaved planes (qs|scales|dm) at q4k_soa_offsets; scales stay the
// PACKED 6-bit get_scale_min_k4 form (NOT the dense-unpacked layout that
// dequant_q4_soa_to_Bt expects) and dm = {d,dmin} fp16 pairs. Bit-identical output to
// dequant_q4_K_to_Bt on the same weight (pure byte reorder). K%256==0, N%64==0.
sycl::event dequant_q4_moe_soa_to_Bt(sycl::queue& q, const void* soa_bank_expert_slice,
                                     sycl::half* out, uint32_t K, uint32_t N,
                                     const std::vector<sycl::event>& deps = {});
// Q5_K / Q8_0 transposed dequant (P3d — the Qwen3.6-27B GGUF carries Q5_K
// attn_k/attn_output and Q8_0 ssm_out, which the dense GEMV set lacks). Same
// B^T [K,N] output as the Q4_K/Q6_K variants. Q5_K needs K%256==0; Q8_0 K%32==0.
sycl::event dequant_q5_K_to_Bt(sycl::queue& q, const void* packed_in,
                               sycl::half* out, uint32_t K, uint32_t N,
                               const std::vector<sycl::event>& deps = {});
sycl::event dequant_q8_0_to_Bt(sycl::queue& q, const void* packed_in,
                               sycl::half* out, uint32_t K, uint32_t N,
                               const std::vector<sycl::event>& deps = {});

}  // namespace ie
