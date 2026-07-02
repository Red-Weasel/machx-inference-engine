// include/ie/ops.hpp — public element-wise + linear-algebra op API.
//
// Conventions:
//  * All pointers are device USM (sycl::malloc_device), unless noted.
//  * All kernels return a sycl::event so callers can chain.
//  * Tensor layout is row-major (last dim contiguous) and FP16 (sycl::half)
//    unless a function name says otherwise.

#pragma once

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ie {

// y[*, j] = (x[*, j] / sqrt(mean(x[*]²) + eps)) * w[j]
// Shapes: x, y are [n_rows, hidden]. w is [hidden]. eps is from config.
// One workgroup per row.
sycl::event rms_norm(sycl::queue& q,
                     const sycl::half* x, const sycl::half* w,
                     sycl::half* y,
                     uint32_t n_rows, uint32_t hidden,
                     float eps,
                     const std::vector<sycl::event>& deps = {});

// Gemma3-style RMSNorm: y[*, j] = (x[*, j] / rms(x)) * (1.0 + w[j])
// Used for `Qwen3_5MoeRMSNorm` (block-level: input_layernorm,
// post_attention_norm, output_norm). research/05 §9 corrected.
// w is FP32 (matches GGUF F32 norm tensors).
sycl::event rms_norm_one_plus_w(sycl::queue& q,
                                const sycl::half* x, const float* w,
                                sycl::half* y,
                                uint32_t n_rows, uint32_t hidden,
                                float eps,
                                const std::vector<sycl::event>& deps = {});

// Same as rms_norm but with FP32 weight (per-head QK-Norm on attention layers).
sycl::event rms_norm_f32w(sycl::queue& q,
                          const sycl::half* x, const float* w,
                          sycl::half* y,
                          uint32_t n_rows, uint32_t hidden,
                          float eps,
                          const std::vector<sycl::event>& deps = {});

// v1.5-C: rms_norm_f32w fused with Q8_1 activation-block emission (q8_out
// receives n_rows × hidden/32 block_q8_1x).  Quantizes the rounded fp16
// outputs — numerics identical to quantize_q8_1 run after the norm.
sycl::event rms_norm_f32w_q8(sycl::queue& q,
                             const sycl::half* x, const float* w,
                             sycl::half* y, void* q8_out,
                             uint32_t n_rows, uint32_t hidden,
                             float eps,
                             const std::vector<sycl::event>& deps = {});

// Pure type cast kernels.
// Fused Q/K/V split + fp16→fp32 cast for the DeltaNet block.  Replaces
// the per-token-per-split loop (3*T launches) with ONE launch.
//   src is [T, SI2] fp16 (post-conv1d/silu); each row is concat
//        [Q (k_dim*head_dim), K (k_dim*head_dim), V (v_dim*head_dim)].
//   q_dst, k_dst, v_dst are [T, k_dim*head_dim] / [T, k_dim*head_dim] /
//        [T, v_dim*head_dim] fp32 buffers.
//
// Caller-supplied strides (k_dim*head_dim and v_dim*head_dim) describe
// where Q ends, K ends, V ends within each src row.
sycl::event cast_qkv_split_fp16_to_fp32(sycl::queue& q,
                                        const sycl::half* src,
                                        float* q_dst, float* k_dst, float* v_dst,
                                        uint32_t T, uint32_t k_total, uint32_t v_total,
                                        const std::vector<sycl::event>& deps = {});

sycl::event cast_fp16_to_fp32(sycl::queue& q, const sycl::half* x, float* y, uint64_t n,
                              const std::vector<sycl::event>& deps = {});
sycl::event cast_fp32_to_fp16(sycl::queue& q, const float* x, sycl::half* y, uint64_t n,
                              const std::vector<sycl::event>& deps = {});

// Embedding lookup: y[t, h] = dequant(token_embd[h, token_ids[t]])
// `token_embd` is the Q4_K- or Q6_K-packed table with shape [hidden, vocab]
// in GGUF (column-per-token layout — each token's embedding is `hidden`
// consecutive elements packed as `hidden/256` super-blocks).
sycl::event embedding_lookup_q6k(sycl::queue& q,
                                 const int32_t* token_ids,
                                 const void* token_embd_q6k,
                                 sycl::half* y,
                                 uint32_t n_tokens, uint32_t hidden,
                                 const std::vector<sycl::event>& deps = {});
sycl::event embedding_lookup_q4k(sycl::queue& q,
                                 const int32_t* token_ids,
                                 const void* token_embd_q4k,
                                 sycl::half* y,
                                 uint32_t n_tokens, uint32_t hidden,
                                 const std::vector<sycl::event>& deps = {});
// Q8_0 token_embd (block = fp16 scale + 32 int8). Unblocks full-Q8_0 GGUFs whose
// embedding ships as Q8_0 (e.g. *-Q8_0 quants). y[i] = d * qs[i].
sycl::event embedding_lookup_q8_0(sycl::queue& q,
                                  const int32_t* token_ids,
                                  const void* token_embd_q8_0,
                                  sycl::half* y,
                                  uint32_t n_tokens, uint32_t hidden,
                                  const std::vector<sycl::event>& deps = {});

// repeat_interleave_heads: replicate each KV head `repeat` times along the head
// axis. Two conventions (model-specific, set by `interleave`):
//   interleave=false (default, TILE): y[t,h_out,d]=x[t, h_out % n_in_heads, d]
//     — the GGUF tiled-repeat convention used by qwen35 (27B), VALIDATED.
//   interleave=true  (INTERLEAVE):    y[t,h_out,d]=x[t, h_out / repeat, d]
//     — k-head h → v-heads {repeat*h .. repeat*h+repeat-1}; qwen3next's ggml
//       REPEAT (q_conv_predelta vh = r + repeat*h).
//   x: [T, n_in_heads, head_dim]   (FP32)
//   y: [T, n_in_heads * repeat, head_dim]  (FP32)
sycl::event repeat_interleave_heads(sycl::queue& q,
                                    const float* x, float* y,
                                    uint32_t T, uint32_t n_in_heads,
                                    uint32_t head_dim, uint32_t repeat,
                                    bool interleave = false,
                                    const std::vector<sycl::event>& deps = {});

// Split the Q-proj output into (Q, gate) when the model has `attn_output_gate=True`.
// Input layout per HF transformers (qwen3_5_moe.modeling):
//   qk = q_proj(x).view(*B, n_heads, 2*head_dim)
//   q, gate = qk.chunk(2, dim=-1)
// In flat memory the per-head pattern is [q_h(head_dim), gate_h(head_dim)] for
// each head h — so the flat split is interleaved, NOT first-half/second-half.
//   qk      : [T, n_heads, 2*head_dim]   (flat T·n_heads·2·head_dim FP16 elements)
//   q_out   : [T, n_heads, head_dim]
//   gate_out: [T, n_heads, head_dim]
sycl::event split_q_gate_per_head(sycl::queue& q,
                                  const sycl::half* qk,
                                  sycl::half* q_out, sycl::half* gate_out,
                                  uint32_t T, uint32_t n_heads, uint32_t head_dim,
                                  const std::vector<sycl::event>& deps = {});

// SwiGLU: y[i] = silu(gate[i]) * up[i] = gate[i] * sigmoid(gate[i]) * up[i]
sycl::event swiglu(sycl::queue& q,
                   const sycl::half* gate, const sycl::half* up,
                   sycl::half* y, uint64_t n,
                   const std::vector<sycl::event>& deps = {});

// gpt-oss clamped gated-SwiGLU (ggml SWIGLU_OAI): out = (x·sigmoid(alpha·x))·(z+1)
// where x=min(gate,limit), z=clamp(up,-limit,limit). gpt-oss: alpha=1.702, limit=7.0.
sycl::event swiglu_oai(sycl::queue& q,
                       const sycl::half* gate, const sycl::half* up,
                       sycl::half* y, uint64_t n, float alpha, float limit,
                       const std::vector<sycl::event>& deps = {});

// GeGLU (Gemma): y[i] = gelu_tanh(gate[i]) * up[i], tanh-approx gelu
// (HF `gelu_pytorch_tanh`). Gemma's FFN activation.
sycl::event geglu(sycl::queue& q,
                  const sycl::half* gate, const sycl::half* up,
                  sycl::half* y, uint64_t n,
                  const std::vector<sycl::event>& deps = {});

// Row-wise GeGLU over an interleaved [n_rows, 2*EF] buffer (Gemma 4 fused-MoE).
// Each row r holds [gate(EF) | up(EF)] contiguously; produces y[n_rows, EF] with
//   y[r,j] = gelu_tanh(gu[r, gate_off + j]) * gu[r, up_off + j].
// `swap` selects gate=second-half / up=first-half (mirrors IE_GEMMA4_SWAP_GATEUP).
// One launch over n_rows*EF; identical fp32 gelu-tanh math to `geglu`.
sycl::event geglu_rows(sycl::queue& q,
                       const sycl::half* gu_packed,   // [n_rows, 2*EF]
                       sycl::half* y,                 // [n_rows, EF]
                       uint32_t n_rows, uint32_t EF, bool swap,
                       const std::vector<sycl::event>& deps = {});

// Gemma 4 int-dot W4A8 MoE projection: y[TK,N] = q8(x)[TK,K] @ Q4_0 W[K,N],
// over ALL experts in one launch (expert_offsets[E+1] partition the TK rows).
// W column-packed Q4_0 (column n = K/32 contiguous 18-byte blocks). Serves
// gate_up (K=H, N=2*EF) and down (K=E_ffn, N=H). x quantized by quantize_q8_1s.
sycl::event moe_prefill_proj_q4_0_q8(sycl::queue& q,
                                     const void* xq8_packed,   // block_q8_1s [TK, K/32]
                                     const void* W_q4_0,       // Q4_0 [K, N] per expert
                                     const uint32_t* expert_offsets,
                                     sycl::half* out_packed,    // [TK, N]
                                     uint32_t E, uint32_t K, uint32_t N,
                                     uint64_t expert_stride_bytes,
                                     const std::vector<sycl::event>& deps = {});

// SoA-Q4_0 version of moe_prefill_proj_q4_0_q8: weights read from per-expert
// contiguous nibble + fp16-scale banks (16 B lane-coalesced) instead of 18 B
// AoS blocks. Serves prefill (T-batched) AND decode (T=1). qs_bank[e]=qs_stride,
// d_bank[e]=d_stride. Expert banks store SoA instead of AoS (no doubling).
sycl::event moe_prefill_proj_q4_0_soa_q8(sycl::queue& q,
                                         const void* xq8_packed,
                                         const uint8_t* qs_bank, const uint16_t* d_bank,
                                         const uint32_t* expert_offsets,
                                         sycl::half* out_packed,
                                         uint32_t E, uint32_t K, uint32_t N,
                                         uint64_t qs_stride, uint64_t d_stride,
                                         const std::vector<sycl::event>& deps = {});

// XMX/DPAS version of moe_prefill_proj_q4_0_q8: y[TK,N] = x_fp16[TK,K] @ Q4_0
// W[K,N] over ALL experts in one launch. fp16 A (NOT q8), dequant-to-fp16 B in
// SLM, mat_mad. Requires K % 256 == 0 and N % 64 == 0 (serves gate_up K=H;
// down K=E_ffn=704 stays on the q8 path). M-loop is per-expert (re-dequant only
// ~ n_tok_e/16 ×, small for MoE) so weight amortization is fine here.
sycl::event moe_prefill_proj_q4_0_xmx(sycl::queue& q,
                                      const sycl::half* x_fp16,   // [TK, K]
                                      const void* W_q4_0,         // Q4_0 [K, N] per expert
                                      const uint32_t* expert_offsets,
                                      sycl::half* out_packed,     // [TK, N]
                                      uint32_t E, uint32_t K, uint32_t N,
                                      uint64_t expert_stride_bytes,
                                      const std::vector<sycl::event>& deps = {});

// y[i] = a[i] + b[i]
sycl::event residual_add(sycl::queue& q,
                         const sycl::half* a, const sycl::half* b,
                         sycl::half* y, uint64_t n,
                         const std::vector<sycl::event>& deps = {});

// y[i] = x[i] * sigmoid(g[i])
//
// Used for the Qwen3.6 full-attention `attn_output_gate` parameterization:
// after attention, multiply each element by sigmoid(gate_input) before the
// final out_proj. (research/04 §5.1 quirk #5).
sycl::event sigmoid_gate(sycl::queue& q,
                         const sycl::half* x, const sycl::half* g,
                         sycl::half* y, uint64_t n,
                         const std::vector<sycl::event>& deps = {});

// y[i] = x[i] * sigmoid(x[i])
sycl::event silu(sycl::queue& q,
                 const sycl::half* x, sycl::half* y, uint64_t n,
                 const std::vector<sycl::event>& deps = {});

// Partial-rotary RoPE.
//
// Applies rotation to the first `n_rotary` of `head_dim` per head; the
// remaining dims pass through unchanged. Rotation is the standard
// "interleaved-pair" form used by Qwen / Llama:
//
//   for r in [0, n_rotary/2):
//     theta = pos * (theta_base ^ (-2r / n_rotary))
//     (x_r, x_{r + n_rotary/2}) = (x_r·cos − x_{...}·sin, x_r·sin + x_{...}·cos)
//
// `x` and `y` may alias. Shape: [n_tokens, n_heads, head_dim] FP16.
// `positions` is [n_tokens] int32 (token positions in the sequence).
sycl::event rope_partial(sycl::queue& q,
                         const sycl::half* x, const int32_t* positions,
                         sycl::half* y,
                         uint32_t n_tokens, uint32_t n_heads,
                         uint32_t head_dim, uint32_t n_rotary,
                         float theta_base,
                         const std::vector<sycl::event>& deps = {});

// P3a: rope_partial with llama3 frequency factors (rope_freqs.weight).
//   freq_factors: device F32 [n_rotary/2]; per ggml the pair frequency becomes
//   inv_freq(r) / freq_factors[r]. nullptr → bit-identical to rope_partial
//   (the dense forward passes null for qwen3/qwen2).
sycl::event rope_partial_ff(sycl::queue& q,
                            const sycl::half* x, const int32_t* positions,
                            sycl::half* y,
                            uint32_t n_tokens, uint32_t n_heads,
                            uint32_t head_dim, uint32_t n_rotary,
                            float theta_base, const float* freq_factors,
                            const std::vector<sycl::event>& deps = {});

// YaRN RoPE (NEOX, ggml-exact). NTK-by-parts theta interpolation between
// extrapolation (high-freq dims) and theta*freq_scale (low-freq dims) blended by
// the corr-dim ramp, PLUS the YaRN magnitude scale on cos/sin. EFFECTIVE mscale =
// attn_factor*(1+0.1*ln(1/freq_scale)) is applied at ALL positions (gpt-oss:
// 1.34657). freq_scale=1/rope.scaling.factor; orig_ctx=original_context_length;
// ext_factor=1.0 for yarn (0 -> reduces to plain scaled rope). NEOX pair (r,r+n_rot/2).
sycl::event rope_yarn(sycl::queue& q,
                      const sycl::half* x, const int32_t* positions,
                      sycl::half* y,
                      uint32_t n_tokens, uint32_t n_heads,
                      uint32_t head_dim, uint32_t n_rotary,
                      float freq_base, float freq_scale, uint32_t orig_ctx,
                      float ext_factor, float attn_factor,
                      float beta_fast, float beta_slow,
                      const std::vector<sycl::event>& deps = {});

// FP16 in / FP32 out GEMM via SYCL joint_matrix.
//   A: [M, K] FP16 row-major
//   B: [K, N] FP16 row-major
//   C: [M, N] FP32 row-major
//
// Uses SLM tiling, double buffering, and (where possible) the 2D block-load
// extension. Phase 3 gate target: ≥ 70% of 183 TFLOPS dense FP16 on 4096³.
sycl::event gemm_fp16(sycl::queue& q,
                      const sycl::half* A, const sycl::half* B,
                      float* C,
                      uint32_t M, uint32_t N, uint32_t K,
                      const std::vector<sycl::event>& deps = {});

// P2: dense-weight fp16 GEMV for F16 GGUF tensors (e.g. qwen3 attn_v).
//   A    : [K] FP16 activation
//   W_kn : [K, N] FP16 row-major (GGUF stores [N, K] row-major; loader transposes before upload)
//   y    : [N] FP16
// K and N have no alignment constraints beyond N ≥ 1, K ≥ 1.
sycl::event gemv_fp16(sycl::queue& q,
                      const sycl::half* A, const sycl::half* W_kn,
                      sycl::half* y, uint32_t K, uint32_t N,
                      const std::vector<sycl::event>& deps = {});

// P1a (2026-06-09): integer-dot (dp4a) decode path.
// quantize_q8_1: fp16 activations [K] → block_q8_1x stream (K % 32 == 0).
// Run ONCE per unique activation vector, reuse across that vector's GEMVs.
sycl::event quantize_q8_1(sycl::queue& q,
                          const sycl::half* x, void* out_q8,
                          uint32_t K,
                          const std::vector<sycl::event>& deps = {});
// Prefill-crown: fp16 [K] -> block_q8_1s stream (48 B blocks with split
// half-sums s0/s1).  Feeds the int-dot MoE prefill kernels (IE_MOE_Q8).
sycl::event quantize_q8_1s(sycl::queue& q,
                           const sycl::half* x, void* out_q8,
                           uint64_t K,
                           const std::vector<sycl::event>& deps = {});
// W4A8 GEMV: y[1,N] = dequant(W_q4K) @ dequant(x_q8).  INT8 dot inner loops.
sycl::event gemv_q4_K_q8(sycl::queue& q,
                         const void* x_q8, const void* W_packed,
                         sycl::half* y,
                         uint32_t K, uint32_t N,
                         const std::vector<sycl::event>& deps = {});
// REORDERED Q4_K decode (llama-SYCL's 3-region global SoA layout — the 52% BW
// trick). repack_q4_K_to_reorder: host [N,K] AoS → nibbles|scales|dm regions
// (out sized = N*(K/256)*144 B, same as AoS). gemv_q4_K_reorder_q8: same dp4a as
// gemv_q4_K_q8 but reads the pure-contiguous nibble stream. See reference_llama_q4k_reorder.
void repack_q4_K_to_reorder(const void* W_blocks, uint32_t K, uint32_t N, uint8_t* out);
sycl::event gemv_q4_K_reorder_q8(sycl::queue& q,
                         const void* x_q8, const void* W_reorder,
                         sycl::half* y,
                         uint32_t K, uint32_t N,
                         const std::vector<sycl::event>& deps = {});

// BATCHED-T W4A8 GEMV (spec-decode verify): reads each Q4_K weight column ONCE,
// dots against T staged Q8_1 activation rows → y[T,N] (y[t*N+n]). x_q8 = T
// contiguous block_q8_1x streams (row t at block offset t*(K/32)). Per-row
// numerics identical to gemv_q4_K_q8. T ≤ 16. See gemv_q8dot.cpp.
sycl::event gemv_q4_K_q8_batched(sycl::queue& q,
                         const void* x_q8, const void* W_packed,
                         sycl::half* y,
                         uint32_t K, uint32_t N, uint32_t T,
                         const std::vector<sycl::event>& deps = {});

// gemv_q4_K_q8s_batched — spec-decode VERIFY isum-elimination variant. Same idot
// as gemv_q4_K_q8_batched but reads the precomputed per-16 half-block sums
// (block_q8_1s.s0/s1) for the Σq8 bias term, dropping the redundant isum dp4a
// (halves the inner dp4a — the T=4 verify ALU bottleneck). x_q8s = T contiguous
// block_q8_1s streams (quantize_q8_1s). y[T,N]. Lossless-gated. See gemv_q8dot.cpp.
sycl::event gemv_q4_K_q8s_batched(sycl::queue& q,
                         const void* x_q8s, const void* W_packed,
                         sycl::half* y,
                         uint32_t K, uint32_t N, uint32_t T,
                         const std::vector<sycl::event>& deps = {});

// W8A8 GEMV over a SoA-repacked Q8_0 weight (qs int8 col-major + fp16 d per
// 32-block). Q6_K-repack decode prototype — no 6-bit unpack, pure int8 dp4a.
sycl::event gemv_q8_0_soa_q8(sycl::queue& q,
                             const void* x_q8, const int8_t* qs_W, const uint16_t* d_W,
                             sycl::half* y,
                             uint32_t K, uint32_t N,
                             const std::vector<sycl::event>& deps = {});

// Same as gemv_q8_0_soa_q8 but reads the activation from global (no SLM staging) —
// higher occupancy on large-K projections (bit-identical output). See gemv_q8dot.cpp.
sycl::event gemv_q8_0_soa_q8_g(sycl::queue& q,
                               const void* x_q8, const int8_t* qs_W, const uint16_t* d_W,
                               sycl::half* y,
                               uint32_t K, uint32_t N,
                               const std::vector<sycl::event>& deps = {});

// v1.5-D2: gated_rms_norm fused with Q8_1 emission (per-head rows; hidden
// must be a multiple of 32).
sycl::event gated_rms_norm_q8(sycl::queue& q,
                              const float* x, const sycl::half* z,
                              const sycl::half* weight,
                              sycl::half* y, void* q8_out,
                              uint32_t n_rows, uint32_t hidden, float eps,
                              const std::vector<sycl::event>& deps = {});

// v1.5-D: integer-dot staged Q6_K GEMV (lm_head shape; needs Q8_1 x).
sycl::event gemv_q6_K_q8(sycl::queue& q,
                         const void* x_q8, const void* W_packed,
                         sycl::half* y,
                         uint32_t K, uint32_t N,
                         const std::vector<sycl::event>& deps = {});

// P3b Task 1: K-tiled integer-dot Q6_K GEMV (dense ffn_down, big K). Same
// int-dot math as gemv_q6_K_q8 but stages the weight slab + q8 stream one
// K-tile at a time, so SLM footprint is fixed (≈29 KiB) regardless of K —
// usable at K=12288 where gemv_q6_K_q8's whole-column slab overflows SLM.
// NOT bit-identical to gemv_q6_K_q8/gemm_q6_K (tile-wise fp32 accumulation).
sycl::event gemv_q6_K_q8_ktiled(sycl::queue& q,
                                const void* x_q8, const void* W_packed,
                                sycl::half* y,
                                uint32_t K, uint32_t N,
                                const std::vector<sycl::event>& deps = {});

// Fast Q6_K decode GEMV via load-time SoA-Q6 repack (gemv_q6_soa.cpp).
// repack_q6_K_to_soa: host de-interleave of canonical block_q6_K [K,N] into
// per-column natural-order bit-plane streams (q6_lo 4-bit, q6_hi 2-bit, per-16
// int8 scales, fp16 super-scale) — stays ~6.5 bpw.  Caller must ZERO q6_lo/q6_hi
// first (the bit-plane writes are read-modify-write).
void repack_q6_K_to_soa(const void* W_blocks, uint32_t K, uint32_t N,
                        uint8_t* q6_lo, uint8_t* q6_hi,
                        int8_t* q6_sc, uint16_t* q6_d);
// gemv_q6_soa_q8: int-dot W6A8 GEMV over the SoA-Q6 streams (split-K, Q8_1 act
// staged in SLM).  NOT bit-exact vs gemv_q6_K (int8 act + fp fold) — PPL-gated.
sycl::event gemv_q6_soa_q8(sycl::queue& q,
                           const void* x_q8,
                           const uint8_t* q6_lo, const uint8_t* q6_hi,
                           const int8_t* q6_sc, const uint16_t* q6_d,
                           sycl::half* y,
                           uint32_t K, uint32_t N,
                           const std::vector<sycl::event>& deps = {});
// BATCHED-T int-dot W6A8 GEMV over the SoA-Q6 streams (spec-decode verify):
// reads each weight column ONCE, dots against T staged Q8_1 activation rows →
// y[T,N] (y[t*N+n]). x_q8 = T contiguous block_q8_1x streams (row t at block
// offset t*(K/32)). Per-row numerics identical to gemv_q6_soa_q8. T ≤ 16.
sycl::event gemv_q6_soa_q8_batched(sycl::queue& q,
                           const void* x_q8,
                           const uint8_t* q6_lo, const uint8_t* q6_hi,
                           const int8_t* q6_sc, const uint16_t* q6_d,
                           sycl::half* y,
                           uint32_t K, uint32_t N, uint32_t T,
                           const std::vector<sycl::event>& deps = {});
// SoA-Q6 → fp16 Bt[K,N] (prefill: feeds gemm_fp16, like dequant_q6_K_to_Bt).
sycl::event dequant_q6_soa_to_Bt(sycl::queue& q,
                                 const uint8_t* q6_lo, const uint8_t* q6_hi,
                                 const int8_t* q6_sc, const uint16_t* q6_d,
                                 sycl::half* Bt, uint32_t K, uint32_t N,
                                 const std::vector<sycl::event>& deps = {});

// Fast Q4_K decode GEMV via load-time SoA-Q4 repack (gemv_q4_soa.cpp). Mirrors
// the Q6-SoA path for Q4_K weights: closes the dense-decode BW gap by replacing
// the AoS gemv_q4_K (W4A16, byte-loads + (sub,half,g,hi_nib) lattice) with an
// int-dot W4A8 GEMV over per-column NATURAL-ORDER streams.
//   repack_q4_K_to_soa: host de-interleave of block_q4_K [K,N] into per-column
//     q4_q (4-bit nibbles, 2 elems/byte), per-32 int8 s_raw/m_raw (the UNPACKED
//     6-bit get_scale_min_k4 values), and per-256 fp16 d/dmin. ~4.625 bpw.
//     The reconstructed weight value (d·s_raw·q4 − dmin·m_raw) is BIT-IDENTICAL
//     to the AoS gemv_q4_K dequant (pure layout move; ie-q4soa-test proves it).
//   Caller allocates: q4_q N*(K/2)B, q4_sc/q4_mn N*(K/32) int8, q4_d/q4_dmin
//     N*(K/256) uint16 (raw fp16). K % 256 == 0.
void repack_q4_K_to_soa(const void* W_blocks, uint32_t K, uint32_t N,
                        uint8_t* q4_q, int8_t* q4_sc, int8_t* q4_mn,
                        uint16_t* q4_d, uint16_t* q4_dmin);
// gemv_q4_soa_q8: int-dot W4A8 GEMV over the SoA-Q4 streams (split-K, Q8_1 act
// staged in SLM; IE_QWEN35_Q4_SOA_GMEM=1 → global-act variant). NOT bit-exact vs
// gemv_q4_K (int8 act + fp fold) — PPL-gated. x quantized by quantize_q8_1.
sycl::event gemv_q4_soa_q8(sycl::queue& q,
                           const void* x_q8,
                           const uint8_t* q4_q,
                           const int8_t* q4_sc, const int8_t* q4_mn,
                           const uint16_t* q4_d, const uint16_t* q4_dmin,
                           sycl::half* y,
                           uint32_t K, uint32_t N,
                           const std::vector<sycl::event>& deps = {});
// SoA-Q4 → fp16 Bt[K,N] (prefill: feeds gemm_fp16, like dequant_q4_K_to_Bt).
sycl::event dequant_q4_soa_to_Bt(sycl::queue& q,
                                 const uint8_t* q4_q,
                                 const int8_t* q4_sc, const int8_t* q4_mn,
                                 const uint16_t* q4_d, const uint16_t* q4_dmin,
                                 sycl::half* Bt, uint32_t K, uint32_t N,
                                 const std::vector<sycl::event>& deps = {});

// P1b: oneDNN matmul (fp16 in/out, fp32 accumulate).  E1 prefill GEMM.
sycl::event gemm_fp16_onednn(sycl::queue& q,
                             const sycl::half* A, const sycl::half* B,
                             sycl::half* y,
                             uint32_t M, uint32_t N, uint32_t K,
                             const std::vector<sycl::event>& deps = {});

// W4A16 GEMV (decode-critical).
//   A: [1, K] FP16 row-major (the activation)
//   W: [K, N] Q4_K-packed (column-major when packed: each column has K/256 super-blocks)
//   y: [1, N] FP16 row-major
//
// `K` must be a multiple of 256. `N` is the output dim. For Qwen3.6 the
// columns of W typically run along the second tensor axis (e.g. attn_q.weight
// shape [in=2048, out=8192]). The GGUF writes packed bytes following the
// reference C struct layout (research/03 §2.2): each column's super-blocks
// laid out sequentially.
//
// Phase 3 gate: ≥ 60% of 608 GB/s effective (≥ 365 GB/s) on a typical decode
// projection (e.g. K=2048, N=2048).
sycl::event gemv_q4_K(sycl::queue& q,
                      const sycl::half* A, const void* W_packed,
                      sycl::half* y,
                      uint32_t K, uint32_t N,
                      const std::vector<sycl::event>& deps = {});

// Fused MXFP4 decode GEMV (gpt-oss MoE experts) over the load-time SoA repack
// (aligned qs_plane[N*(K/2)] + e_plane[N*(K/32)]; see gemv_mxfp4.cpp / repack).
// Replaces the decode dequant_mxfp4_to_Bt + M=1 oneDNN GEMM (which materialized fp16
// and re-read it — ~10.8 GB/token vs a 1.27 GB packed-weight floor).  K%32==0.
//   gemv_mxfp4_soa_f16: fp16 activation, BIT-FAITHFUL (same d·LUT values).
//   gemv_mxfp4_soa_q8 : W4A8 int-dot (dp4a_ss); x pre-quantized to block_q8_1x via
//     quantize_q8_1.  The ALU-efficient B70 decode lever.  PPL-gated (not bit-exact).
sycl::event gemv_mxfp4_soa_f16(sycl::queue& q,
                               const sycl::half* A,
                               const uint8_t* qs_plane, const uint8_t* e_plane,
                               sycl::half* y, uint32_t K, uint32_t N,
                               const std::vector<sycl::event>& deps = {});
sycl::event gemv_mxfp4_soa_q8(sycl::queue& q,
                              const void* x_q8,
                              const uint8_t* qs_plane, const uint8_t* e_plane,
                              sycl::half* y, uint32_t K, uint32_t N,
                              const std::vector<sycl::event>& deps = {});
//   gemv_mxfp4_soa_q8_x2: FUSED gate+up — one launch, shared SLM-staged x, both
//     N=efc outputs (gate->gate_y, up->up_y), per-column biases folded post-reduce.
//     BIT-IDENTICAL to two gemv_mxfp4_soa_q8 + add_bias; cuts host launches 4->1 per
//     expert (gpt-oss MoE decode is host-launch-bound). Decode (1 row) only.
sycl::event gemv_mxfp4_soa_q8_x2(sycl::queue& q, const void* x_q8,
                                 const uint8_t* gate_qs, const uint8_t* gate_e,
                                 const uint8_t* up_qs,   const uint8_t* up_e,
                                 const float* gate_bias, const float* up_bias,
                                 sycl::half* gate_y, sycl::half* up_y,
                                 uint32_t K, uint32_t N,
                                 const std::vector<sycl::event>& deps = {});

// Multi-row GEMM with Q4_K weights, M ≤ 8 per launch. Caller chunks M
// into M_TILE-sized passes for larger T. Amortizes the weight-read cost
// M× across rows by SLM-staging A[M, K]. See gemv_q4_K for layout.
sycl::event gemm_q4_K(sycl::queue& q,
                      const sycl::half* A, const void* W_packed,
                      sycl::half* y,
                      uint32_t M, uint32_t K, uint32_t N,
                      const std::vector<sycl::event>& deps = {});

// W4A16 GEMV/GEMM with Q4_0 weights (Gemma 4 QAT format). Same W[K,N]
// column-packed layout as gemv_q4_K but Q4_0 blocks (32 elems, 18 bytes,
// one FP16 scale, symmetric `w = d*(q-8)`). K % 32 == 0.
sycl::event gemv_q4_0(sycl::queue& q,
                      const sycl::half* A, const void* W_packed,
                      sycl::half* y,
                      uint32_t K, uint32_t N,
                      const std::vector<sycl::event>& deps = {});
sycl::event gemm_q4_0(sycl::queue& q,
                      const sycl::half* A, const void* W_packed,
                      sycl::half* y,
                      uint32_t M, uint32_t K, uint32_t N,
                      const std::vector<sycl::event>& deps = {});

// int-dot W4A8 batched GEMM: y[M,N] = q8(x)[M,K] @ Q4_0 W[K,N], split-K over
// the K dimension so it handles ARBITRARY K (unlike moe_prefill_proj_q4_0_q8's
// MAX_BPL register cap). x is pre-quantized to block_q8_1s [M, K/32] by
// quantize_q8_1s. Same dp4a fold as the gemma4 fused-MoE kernel; used for the
// large-K dense/attn projections (o-proj, and all 31B-dense projections).
sycl::event gemm_q4_0_q8(sycl::queue& q,
                         const void* xq8_packed,    // block_q8_1s [M, K/32]
                         const void* W_q4_0,        // Q4_0 [K, N] column-packed
                         sycl::half* y,             // [M, N]
                         uint32_t M, uint32_t K, uint32_t N,
                         const std::vector<sycl::event>& deps = {});

// Transposing dequant Q4_0 W[K,N] (column-packed) → fp16 Bt[K,N] row-major,
// the layout gemm_fp16 / gemm_fp16_onednn consume (weight-stationary GEMM).
sycl::event dequant_q4_0_to_Bt(sycl::queue& q,
                               const void* W_packed, sycl::half* Bt,
                               uint32_t K, uint32_t N,
                               const std::vector<sycl::event>& deps = {});

// SoA variant: same fp16 Bt[K,N] row-major output as dequant_q4_0_to_Bt (BIT-
// IDENTICAL), but reads the de-interleaved SoA streams (soa_qs: per-column K/2
// nibble bytes; soa_d: per-column K/32 fp16 scales) produced by repack_q4_0_to_soa
// instead of the AoS block_q4_0. Lets a model keep ONLY the SoA copy (no AoS) and
// still run the weight-stationary onednn prefill GEMM. K % 32 == 0.
sycl::event dequant_q4_0_soa_to_Bt(sycl::queue& q,
                                   const uint8_t* soa_qs, const uint16_t* soa_d,
                                   sycl::half* Bt, uint32_t K, uint32_t N,
                                   const std::vector<sycl::event>& deps = {});

// SoA-Q4_0 fast decode GEMV (W4A8 int-dot, ~80% BW template). repack_q4_0_to_soa
// splits the AoS block_q4_0 [K,N] into contiguous per-column nibble + fp16-scale
// streams (host); gemv_q4_0_soa_q8 does the split-K int-dot decode GEMV over them
// (x quantized by quantize_q8_1 → block_q8_1x). K % 32 == 0.
void repack_q4_0_to_soa(const void* W_blocks, uint32_t K, uint32_t N,
                        uint8_t* q4_qs, uint16_t* q4_d);
sycl::event gemv_q4_0_soa_q8(sycl::queue& q,
                             const void* x_q8, const uint8_t* q4_qs,
                             const uint16_t* q4_d, sycl::half* y,
                             uint32_t K, uint32_t N,
                             const std::vector<sycl::event>& deps = {});

// Fused multi-bank variant: up to 3 SoA-Q4_0 projections that SHARE the same
// quantized activation x_q8 (same K) are computed in ONE kernel launch. Work-
// groups tile the concatenated column space, so the tiny GQA k/v projections
// ride a full, well-occupied grid instead of 3 under-sized launches. Each bank
// b in [0,n) reads qs[b]/d[b] (N[b] columns) and writes y[b]. Decode lever for
// q/k/v (shared attn_norm input) and gate/up (shared ffn_norm input).
sycl::event gemv_q4_0_soa_q8_multi(sycl::queue& q,
                                   const void* x_q8,
                                   const uint8_t* const qs[3],
                                   const uint16_t* const d[3],
                                   sycl::half* const y[3],
                                   const uint32_t N[3], int n_banks,
                                   uint32_t K,
                                   const std::vector<sycl::event>& deps = {});

// BATCHED-T (spec-decode verify): each Q4_0-SoA weight column read ONCE, dotted
// against T staged Q8_1 rows → y[T,N]. x_q8 = T contiguous block_q8_1x streams
// (row t at t*(K/32)). Per-row bit-identical to gemv_q4_0_soa_q8. T ≤ 8.
sycl::event gemv_q4_0_soa_q8_batched(sycl::queue& q,
                                     const void* x_q8, const uint8_t* q4_qs,
                                     const uint16_t* q4_d, sycl::half* y,
                                     uint32_t K, uint32_t N, uint32_t T,
                                     const std::vector<sycl::event>& deps = {});

// XMX (joint_matrix mat_mad) variant of gemm_q4_K.  Uses Battlemage
// dpas instructions for the inner dot product.  Requires N % 64 == 0
// and K % 256 == 0 (one Q4_K block per K iter).  Caller falls back to
// gemm_q4_K if those constraints aren't met.
sycl::event gemm_q4_K_xmx(sycl::queue& q,
                          const sycl::half* A, const void* W_packed,
                          sycl::half* y,
                          uint32_t M, uint32_t K, uint32_t N,
                          const std::vector<sycl::event>& deps = {});

// XMX (joint_matrix mat_mad) GEMM with Q4_0 weights (Gemma 4 QAT). fp16 A,
// dequant-to-fp16 B, fp32 accumulate on the DPAS engine. M ≤ 16 per call
// (caller tiles M in 16s); requires N % 64 == 0 and K % 256 == 0 (else caller
// falls back to gemm_q4_0_q8 / gemv_q4_0). Same contract as gemm_q4_K_xmx.
sycl::event gemm_q4_0_xmx(sycl::queue& q,
                          const sycl::half* A, const void* W_packed,
                          sycl::half* y,
                          uint32_t M, uint32_t K, uint32_t N,
                          const std::vector<sycl::event>& deps = {});

// =====================================================================
// Phase 1 (v2.0): ESIMD tile-load smoke test — UNSAFE on BMG-G31.
// =====================================================================
// Every variant tried (2D block load, 1D block read, per-lane global
// indexing, SLM-staged) causes UR_RESULT_ERROR_DEVICE_LOST on BMG-G31
// C0 (IP 20.2.0).  Guarded by IE_ENABLE_UNSAFE_BMG_ESIMD_EXPERIMENTS.
#ifdef IE_ENABLE_UNSAFE_BMG_ESIMD_EXPERIMENTS
sycl::event esimd_block2d_smoke_tile256(sycl::queue& q,
                                        const uint8_t* src,
                                        uint32_t* out_sum,
                                        uint32_t width_bytes,
                                        uint32_t height_rows,
                                        uint32_t pitch_bytes,
                                        const std::vector<sycl::event>& deps = {});
#endif // IE_ENABLE_UNSAFE_BMG_ESIMD_EXPERIMENTS

// XMX variant for Q6_K weights (used for lm_head, attn_v, ffn_down_*
// in Q4_K_M models).  Same layout contract as gemm_q4_K_xmx.
sycl::event gemm_q6_K_xmx(sycl::queue& q,
                          const sycl::half* A, const void* W_packed,
                          sycl::half* y,
                          uint32_t M, uint32_t K, uint32_t N,
                          const std::vector<sycl::event>& deps = {});

// GEMV W6A16 — same shape contract as `gemv_q4_K`, but W is Q6_K-packed.
// Used for `ffn_down_exps` (Q6_K), `ffn_down_shexp` (Q6_K), `attn_v` (Q6_K),
// `output.weight` (Q6_K) etc. K must be a multiple of 256.
sycl::event gemv_q6_K(sycl::queue& q,
                      const sycl::half* A, const void* W_packed,
                      sycl::half* y,
                      uint32_t K, uint32_t N,
                      const std::vector<sycl::event>& deps = {});

// Multi-row Q6_K GEMM, M ≤ M_TILE per launch.  Mirrors gemm_q4_K's tile
// structure with Q6_K dequant in the inner loop.  Used by gemv_q_T for
// T>1 (prefill) on Q6_K weights — closes the per-row gemv_q6_K fallback
// gap from before this kernel existed.
sycl::event gemm_q6_K(sycl::queue& q,
                      const sycl::half* A, const void* W_packed,
                      sycl::half* y,
                      uint32_t M, uint32_t K, uint32_t N,
                      const std::vector<sycl::event>& deps = {});

// EXL3 (QTIP-based, cb=0) trellis-decode GEMV: y[1,N] = A[1,K] @ W_rot[K,N],
// W_rot decoded on the fly from the EXL3 tail-biting trellis `codes` (raw int16
// bytes, [K/16, N/16, 16*bits]). DECODE-ONLY — the Hadamard incoherence + suh/svh
// scales wrap this kernel (see gemv_exl3_forward). K, N multiples of 16. cb=0.
sycl::event gemv_exl3(sycl::queue& q,
                      const sycl::half* A, const void* codes,
                      sycl::half* y,
                      uint32_t K, uint32_t N, uint32_t bits,
                      const std::vector<sycl::event>& deps = {});

// Full EXL3 linear forward (the engine entry point): y = had128(A⊙suh) @ W_rot,
// then had128(·)⊙svh. Manages per-device scratch internally. suh is F16[K], svh
// F16[N]. Validated bit-exact vs the materialized weight (ie-exl3-test [C]).
sycl::event gemv_exl3_forward(sycl::queue& q,
                              const sycl::half* A, const void* codes,
                              const sycl::half* suh, const sycl::half* svh,
                              sycl::half* y,
                              uint32_t K, uint32_t N, uint32_t bits,
                              const std::vector<sycl::event>& deps = {});

// Row-batched EXL3 trellis-decode GEMV for fused MoE: for each row r∈[0,R),
// y[r, :N] = A[r, :K] @ W_rot(expert = row_expert[r]), where expert e's trellis
// lives at codes_base + e*expert_stride_bytes. DECODE-ONLY (the per-expert
// Hadamard/suh/svh wrap via hadamard_transform_moe). `bits` is uniform across
// experts of a layer (the per-layer-variable, per-expert-uniform EXL3 landmine).
// Mirrors gemv_exl3 exactly with a leading row dim + per-row expert gather.
sycl::event gemv_exl3_moe(sycl::queue& q,
                          const sycl::half* A, const void* codes_base,
                          uint64_t expert_stride_bytes,
                          const int32_t* row_expert,
                          sycl::half* y,
                          uint32_t K, uint32_t N, uint32_t R, uint32_t bits,
                          const std::vector<sycl::event>& deps = {});

// 128-point Sylvester Walsh–Hadamard transform (norm 1/sqrt(128)) applied
// block-wise to `x` (length `n`, a multiple of 128) → `y`. EXL3's runtime
// incoherence step. Optional per-element scales: pre_scale applied BEFORE the
// transform (suh), post_scale AFTER (svh); both null → a plain orthogonal WHT.
sycl::event hadamard_transform(sycl::queue& q,
                               const sycl::half* x, sycl::half* y,
                               uint32_t n,
                               const sycl::half* pre_scale = nullptr,
                               const sycl::half* post_scale = nullptr,
                               const std::vector<sycl::event>& deps = {});

// Row-batched 128-pt Hadamard for fused EXL3 MoE: x/y are [R, n] (row-major);
// each row r applies the transform per 128-block with PER-EXPERT scales gathered
// from the suh/svh banks at expert row_expert[r]. pre_base/post_base are the full
// banks [E*n] (suh applied before, svh after); either null → skip that scale.
// In-place (x==y) is safe. Mirrors hadamard_transform with a row/expert gather.
sycl::event hadamard_transform_moe(sycl::queue& q,
                                   const sycl::half* x, sycl::half* y,
                                   uint32_t n, uint32_t R,
                                   const int32_t* row_expert,
                                   const sycl::half* pre_base,
                                   const sycl::half* post_base,
                                   const std::vector<sycl::event>& deps = {});

// Native F16 embedding lookup: gather row `tokens[t]` from embd[vocab, hidden]
// (row-major, F16) into y[t, hidden] (F16). For EXL3 models whose token_embd
// ships as plain F16 (kept faithful — not re-quantized). Mirrors embedding_lookup_q6k.
sycl::event embedding_lookup_f16(sycl::queue& q,
                                 const sycl::half* embd, const int32_t* tokens,
                                 sycl::half* y, uint32_t n_tokens, uint32_t hidden,
                                 const std::vector<sycl::event>& deps = {});

// =====================================================================
// Sampler kernels
// =====================================================================

// Argmax over [vocab] FP16 logits. Output: single int32 (host or device).
sycl::event sample_argmax(sycl::queue& q,
                          const sycl::half* logits, int32_t* out,
                          uint32_t vocab,
                          const std::vector<sycl::event>& deps = {});

// Apply repetition penalty in-place: logits[id] /= penalty if logits[id] > 0
//                                   logits[id] *= penalty if logits[id] < 0
// `recent_ids`: device pointer to [n_recent] int32 token IDs (USM device memory).
// `penalty == 1.0` is a no-op (skip).
sycl::event repetition_penalty(sycl::queue& q,
                               sycl::half* logits, uint32_t vocab,
                               const int32_t* recent_ids, uint32_t n_recent,
                               float penalty,
                               const std::vector<sycl::event>& deps = {});

// One-shot sampling: applies temperature, top-k, top-p, min-p in that order
// to `logits` (in-place writes log-probs after softmax to logits buffer for
// inspection), then does multinomial selection with seed `rng_state`. Output:
// single int32 in `*out`. `top_k=0` disables top-k; `top_p=1.0` disables
// top-p; `min_p=0` disables min-p; `temperature<=0` selects argmax.
sycl::event sample_softmax_topk_topp(sycl::queue& q,
                                     sycl::half* logits, int32_t* out,
                                     uint32_t vocab,
                                     float temperature, uint32_t top_k,
                                     float top_p, float min_p,
                                     uint64_t rng_state,
                                     const std::vector<sycl::event>& deps = {});

// =====================================================================
// MoE ops (research/06_moe_math.md)
// =====================================================================

// Top-k softmax router. Per token: logits = W_gate @ x ∈ ℝ^E (fp32),
// softmax over E, pick top-k, renormalize the k weights to sum to 1, **sort
// the top-k by ascending expert index** (required for bit-exact reproduction
// of HF's expert iteration order — see research/06 §6).
//
//   x          : [n_tokens, hidden]            FP16
//   W_gate     : [n_experts, hidden]           FP32  (router weights — F32 in GGUF)
//   topk_idx   : [n_tokens, k]                 INT32 (sorted ascending)
//   topk_w     : [n_tokens, k]                 FP16  (renormalized)
//
// Phase 6 v1 supports only n_experts=256 and k=8.
sycl::event moe_router(sycl::queue& q,
                       const sycl::half* x, const float* W_gate,
                       int32_t* topk_idx, sycl::half* topk_w,
                       uint32_t n_tokens, uint32_t hidden,
                       uint32_t n_experts, uint32_t k,
                       const std::vector<sycl::event>& deps = {});

// On-device softmax + top-K from logits ALREADY on device (generalized
// moe_router stage 2; arbitrary E≤1024, arbitrary K). topk_idx ascending,
// topk_w renormalized. One WG = E lanes.
sycl::event moe_topk_from_logits(sycl::queue& q, const sycl::half* logits,
                                 int32_t* topk_idx, sycl::half* topk_w,
                                 uint32_t E, uint32_t K,
                                 const std::vector<sycl::event>& deps = {});

// Build the MoePacking arrays on device for the T==1 decode case from a
// top-K route (topk_idx ascending). Single work-item.
sycl::event moe_build_pack_decode(sycl::queue& q,
                                  const int32_t* topk_idx, const sycl::half* topk_w,
                                  uint32_t* expert_offsets, int32_t* sorted_idx,
                                  int32_t* tk_to_packed, sycl::half* weights_packed,
                                  uint32_t E, uint32_t K,
                                  const std::vector<sycl::event>& deps = {});

// v1.4 shared-expert decode fusion: dual gate/up GEMV + SiLU combine +
// the sigmoid gate-scalar dot (one extra WG) in ONE launch; and the down
// GEMV with the scaled-accumulate epilogue (y += sh_g * acc).
sycl::event gemv_q4_K_shexp_gate_up(sycl::queue& q,
                                    const sycl::half* A,
                                    const void* W_gate_packed,
                                    const void* W_up_packed,
                                    const float* W_shg,
                                    sycl::half* y_h, sycl::half* sh_g_out,
                                    uint32_t K, uint32_t N,
                                    const std::vector<sycl::event>& deps = {});
// v1.4: T==1 alpha/beta dual GEMV + compute_g_beta fused (single WG;
// valid for N <= 32 only — the g/beta tail runs WG-locally).  Numerics
// identical to the unfused chain (accumulators round through fp16 at the
// point the old path round-tripped through ws_alpha/beta_fp16_).
sycl::event gemv_q4_K_dual_ssm_gbeta(sycl::queue& q,
                                     const sycl::half* A,
                                     const void* W_alpha_packed,
                                     const void* W_beta_packed,
                                     const float* A_log, const float* dt_bias,
                                     float* g_out, float* beta_out,
                                     uint32_t K, uint32_t N,
                                     const std::vector<sycl::event>& deps = {});
sycl::event gemv_q4_K_down_accum(sycl::queue& q,
                                 const sycl::half* A, const void* W_packed,
                                 const sycl::half* sh_g, sycl::half* y,
                                 uint32_t K, uint32_t N,
                                 const std::vector<sycl::event>& deps = {});

// Per-token sigmoid scalar gate for the shared expert: out = sigmoid(W_shg · x)
//   x       : [n_tokens, hidden]    FP16
//   W_shg   : [hidden]              FP32
//   gate_out: [n_tokens]            FP16 — one scalar per token
sycl::event shared_expert_gate(sycl::queue& q,
                               const sycl::half* x, const float* W_shg,
                               sycl::half* gate_out,
                               uint32_t n_tokens, uint32_t hidden,
                               const std::vector<sycl::event>& deps = {});

// y[i] += scale * x[i]   (scale is a host fp16 scalar, no broadcast needed)
sycl::event scaled_add(sycl::queue& q,
                       const sycl::half* x, sycl::half scale,
                       sycl::half* y, uint64_t n,
                       const std::vector<sycl::event>& deps = {});

// Same shape as scaled_add but reads the scalar from a device pointer
// (single fp16 element at *scale_dev). Used by the shared-expert path during
// decode to avoid a host roundtrip on the per-token sigmoid gate.
sycl::event scaled_add_dev_scalar(sycl::queue& q,
                                  const sycl::half* x, const sycl::half* scale_dev,
                                  sycl::half* y, uint64_t n,
                                  const std::vector<sycl::event>& deps = {});

// Per-token row-broadcast scaled add, used by the shared-expert prefill path.
//   y[t, h] += scale_per_tok[t] * x[t, h]   for t ∈ [0, T), h ∈ [0, H).
// scale_per_tok is a device pointer to T fp16 scalars (one per token).
// Replaces the T-iteration host-memcpy + per-token scaled_add loop.
sycl::event scaled_add_per_token_row(sycl::queue& q,
                                     const sycl::half* x,
                                     const sycl::half* scale_per_tok,
                                     sycl::half* y,
                                     uint32_t T, uint32_t H,
                                     const std::vector<sycl::event>& deps = {});

// Column-broadcast bias add:  y[t, n] += bias[n]  for t ∈ [0,T), n ∈ [0,N).
// `bias` is a device pointer to N fp32 values. Used by archs with attention
// bias (e.g. Qwen2) after the q/k/v projection GEMV.
sycl::event add_bias(sycl::queue& q, sycl::half* y, const float* bias,
                     uint32_t T, uint32_t N,
                     const std::vector<sycl::event>& deps = {});

// =====================================================================
// Fused MoE decode kernels — collapse the 47 launches/layer of the naive
// per-expert loop into 2-3 kernels. T=1 only (decode); prefill keeps the
// scalar gemv_q_T loop. Reads topk_idx/topk_w directly from device — no
// host roundtrip.
// =====================================================================

// XMX (joint_matrix) variant of moe_prefill_gate_up_silu_q4k — v2 rewrite
// (2026-06-09).  Same contract as the scalar kernel (expert-sorted x_packed
// in, h_packed out); inner dot products run on XMX with vectorized Q4_K
// dequant into SLM B-tiles.  Returns a no-op event when shapes don't match
// (H % 256, E_ffn % 64).  Opt-in via IE_MOE_XMX=1 (see qwen36.cpp dispatch).
sycl::event moe_prefill_gate_up_silu_q4k_xmx(sycl::queue& q,
                                             const sycl::half* x_packed,
                                             const void* gate_W, const void* up_W,
                                             const uint32_t* expert_offsets,
                                             sycl::half* h_packed,
                                             uint32_t E, uint32_t H, uint32_t E_ffn,
                                             uint64_t expert_stride_bytes,
                                             const std::vector<sycl::event>& deps = {});

// Multi-expert prefill stage 1: gate + up + swiglu for all active experts
// in one launch.  Inputs are pre-sorted by expert id so each expert's tokens
// form a contiguous slice [expert_offsets[e], expert_offsets[e+1]) of
// `sorted_token_idx`.  Output `h_packed` has layout
//   h_packed[(expert_offsets[e] + i) * E_ffn + n] for i in [0, n_tok_e).
// One WG per (expert, n_chunk_in_E_ffn).  Inside each WG, M_TILE=8 tokens
// at a time amortize the weight read across rows.
//
// Total launches per MoE layer: 1 (replaces T-token × per-token-fused-MoE
// dispatch which was 2T launches).
// E5 (2026-06-09): takes pre-gathered, expert-sorted `x_packed` (built once
// per layer by moe_gather_rows) instead of x + sorted_token_idx.  Before,
// every n-chunk WG of an expert re-gathered the same rows through the index
// indirection — ~E_ffn/N_PER_WG× redundant scattered reads of x per layer.
// P1b-2: integer-dot stage 1 (llama.cpp's MoE prefill technique) —
// consumes a block_q8_1x stream over the expert-sorted x_packed.
sycl::event moe_prefill_gate_up_silu_q4k_q8(sycl::queue& q,
                                            const void* xq8_packed,
                                            const void* gate_W, const void* up_W,
                                            const uint32_t* expert_offsets,
                                            sycl::half* h_packed,
                                            uint32_t E, uint32_t H, uint32_t E_ffn,
                                            uint64_t expert_stride_bytes, bool soa,
                                            const std::vector<sycl::event>& deps = {});

sycl::event moe_prefill_gate_up_silu_q4k(sycl::queue& q,
                                         const sycl::half* x_packed,     // [T*K_top, H] expert-sorted
                                         const void* gate_W, const void* up_W,
                                         const uint32_t* expert_offsets, // [E+1] device
                                         sycl::half* h_packed,           // [T*K_top, E_ffn]
                                         uint32_t E, uint32_t H, uint32_t E_ffn,
                                         uint64_t expert_stride_bytes, bool soa,
                                         const std::vector<sycl::event>& deps = {});

// Multi-expert prefill stage 2: down + per-(token,expert) weight scaling
// + atomic-add into `y_fp32` (fp32 because atomic_ref<sycl::half> isn't
// portably supported on the level-zero backend). Caller must zero
// `y_fp32` before this call and merge it back with a separate fp32→fp16
// op (`moe_prefill_merge_fp32_to_fp16`).
sycl::event moe_prefill_down_q4k(sycl::queue& q,
                                 const sycl::half* h_packed,
                                 const void* down_W,
                                 const uint32_t* expert_offsets,
                                 const int32_t* sorted_token_idx,
                                 const sycl::half* sorted_token_w,
                                 float* y_fp32,                         // [T, H]
                                 uint32_t E, uint32_t H, uint32_t E_ffn,
                                 uint64_t expert_stride_bytes,
                                 const std::vector<sycl::event>& deps = {});
sycl::event moe_prefill_down_q6k(sycl::queue& q,
                                 const sycl::half* h_packed,
                                 const void* down_W,
                                 const uint32_t* expert_offsets,
                                 const int32_t* sorted_token_idx,
                                 const sycl::half* sorted_token_w,
                                 float* y_fp32,
                                 uint32_t E, uint32_t H, uint32_t E_ffn,
                                 uint64_t expert_stride_bytes,
                                 const std::vector<sycl::event>& deps = {});

// Merge the multi-expert prefill stage 2 fp32 accumulator into the fp16
// MoE output buffer.  y_fp16[i] += sycl::half(y_fp32[i]).
sycl::event moe_prefill_merge_fp32_to_fp16(sycl::queue& q,
                                           const float* y_fp32,
                                           sycl::half* y_fp16,
                                           uint64_t n,
                                           const std::vector<sycl::event>& deps = {});

// Atomics-free multi-expert prefill stage 2.  Writes per-packed-row
// output (no race; each WG owns its rows), to be reduced per-token by
// `moe_prefill_reduce`.  Same layout assumptions as the atomic variant.
sycl::event moe_prefill_down_packed_q4k(sycl::queue& q,
                                        const sycl::half* h_packed,
                                        const void* down_W,
                                        const uint32_t* expert_offsets,
                                        sycl::half* out_packed,           // [T*K_top, H]
                                        uint32_t E, uint32_t H, uint32_t E_ffn,
                                        uint64_t expert_stride_bytes,
                                        const std::vector<sycl::event>& deps = {});
sycl::event moe_prefill_down_packed_q6k(sycl::queue& q,
                                        const sycl::half* h_packed,
                                        const void* down_W,
                                        const uint32_t* expert_offsets,
                                        sycl::half* out_packed,
                                        uint32_t E, uint32_t H, uint32_t E_ffn,
                                        uint64_t expert_stride_bytes,
                                        const std::vector<sycl::event>& deps = {});

// Prefill-crown: integer-dot stage-2 down kernels (IE_MOE_Q8).  Consume a
// block_q8_1s stream over h_packed (quantize_q8_1s once per layer).  One SG
// covers the full K = E_ffn = 512 reduction (16 lanes x 32 elements = 2
// Q4_K blocks / lane-mapped Q6_K sub-blocks), so expert weights are loaded
// into registers ONCE per (expert, column) and reused across every routed
// token row.  Requires E_ffn == 512; callers fall back to the fp16 v2
// kernels otherwise.
sycl::event moe_prefill_down_packed_q4k_q8(sycl::queue& q,
                                        const void* hq8_packed,
                                        const void* down_W,
                                        const uint32_t* expert_offsets,
                                        sycl::half* out_packed,
                                        uint32_t E, uint32_t H, uint32_t E_ffn,
                                        uint64_t expert_stride_bytes, bool soa,
                                        const std::vector<sycl::event>& deps = {});
sycl::event moe_prefill_down_packed_q6k_q8(sycl::queue& q,
                                        const void* hq8_packed,
                                        const void* down_W,
                                        const uint32_t* expert_offsets,
                                        sycl::half* out_packed,
                                        uint32_t E, uint32_t H, uint32_t E_ffn,
                                        uint64_t expert_stride_bytes, bool soa,
                                        const std::vector<sycl::event>& deps = {});

// Reduce per-token: y[t, h] += Σ_kslot weights_packed[tk_to_packed[t,kslot]] *
//                                    out_packed[tk_to_packed[t,kslot], h]
// One WG per (token, h_chunk) — atomic-free, race-free.
sycl::event moe_prefill_reduce(sycl::queue& q,
                               const sycl::half* out_packed,         // [T*K_top, H]
                               const uint32_t* tk_to_packed,         // [T, K_top]
                               const sycl::half* weights_packed,     // [T*K_top]
                               sycl::half* y,                        // [T, H]
                               uint32_t T, uint32_t K_top, uint32_t H,
                               const std::vector<sycl::event>& deps = {});

// Fused gate+up+silu for K_top experts in one launch (Q4_K weights).
//
//   x        : [H]                        FP16, post-RMS-norm input
//   gate_W   : [E_total, E_ffn, H/256]    Q4_K-packed (gate experts, full bank)
//   up_W     : [E_total, E_ffn, H/256]    Q4_K-packed (up experts, full bank)
//   topk_idx : [K_top]                    INT32 device-resident, expert IDs
//   h_out    : [K_top, E_ffn]             FP16 — silu(gate(x))*up(x) per expert
//
// Per-expert byte stride (passed as `expert_stride_bytes`) = (H/256) * E_ffn *
// sizeof(block_q4_K). Per layout `[E_total, E_ffn, H/256]`, expert e starts
// at `gate_W + e * expert_stride_bytes`.
//
// One WG per (k, n-chunk). WG cooperatively loads x[H] into SLM once (~4 KiB).
// Each sub-group computes one output column n: gate[n] and up[n] in parallel
// on the same lane lattice, then `swiglu(gate, up) = silu(gate) * up`.
sycl::event moe_decode_gate_up_silu_q4k(sycl::queue& q,
                                        const sycl::half* x,
                                        const void* gate_W, const void* up_W,
                                        const int32_t* topk_idx,
                                        sycl::half* h_out,
                                        uint32_t H, uint32_t E_ffn, uint32_t K_top,
                                        uint64_t expert_stride_bytes, bool soa,
                                        const std::vector<sycl::event>& deps = {});

// Gather rows: y_packed[i] = x[indices[i]] for i in [0, n_tok).
// Used by scatter-gather MoE prefill to pack the input rows that route
// to a single expert into a contiguous [n_tok, hidden] buffer.
sycl::event moe_gather_rows(sycl::queue& q,
                            const sycl::half* x,        // [T, hidden]
                            const int32_t* indices,     // [n_tok] device
                            sycl::half* y_packed,       // [n_tok, hidden]
                            uint32_t n_tok, uint32_t hidden,
                            const std::vector<sycl::event>& deps = {});

// Scatter-add with per-row weight: y[indices[i]] += weights[i] * x_packed[i].
// `y` is the accumulated MoE output [T, hidden]; `x_packed` is the per-expert
// down-projection output [n_tok, hidden]. Caller supplies a device array of
// `n_tok` token indices (where to add) and `n_tok` fp16 scale weights (the
// router's per-(token, expert) weight).
sycl::event moe_scatter_add(sycl::queue& q,
                            const sycl::half* x_packed,  // [n_tok, hidden]
                            const int32_t* indices,      // [n_tok]
                            const sycl::half* weights,   // [n_tok]
                            sycl::half* y,               // [T, hidden]
                            uint32_t n_tok, uint32_t hidden,
                            const std::vector<sycl::event>& deps = {});

// Fused down + per-expert weight scale + accumulate-into-residual.
//
//   h         : [K_top, E_ffn]            FP16 (output of stage 1)
//   down_W    : [E_total, H, E_ffn/256]   Q4_K- or Q6_K-packed (down experts)
//   topk_idx  : [K_top]                   INT32 device, expert IDs
//   topk_w    : [K_top]                   FP16  device, renormalized weights
//   y_out     : [H]                       FP16  (zero-initialized; kernel WRITES,
//                                                does not accumulate prior content)
//
// One WG per H output chunk. Loops K_top experts inside the kernel, summing
// `topk_w[k] * (down_e_k row n) · h[k]` per output element. The kernel
// expects `y_out` to be initialized to 0 by the caller (a single memset).
sycl::event moe_decode_down_q4k(sycl::queue& q,
                                const sycl::half* h,
                                const void* down_W,
                                const int32_t* topk_idx,
                                const sycl::half* topk_w,
                                sycl::half* y_out,
                                uint32_t H, uint32_t E_ffn, uint32_t K_top,
                                uint64_t expert_stride_bytes, bool soa,
                                const std::vector<sycl::event>& deps = {});
sycl::event moe_decode_down_q6k(sycl::queue& q,
                                const sycl::half* h,
                                const void* down_W,
                                const int32_t* topk_idx,
                                const sycl::half* topk_w,
                                sycl::half* y_out,
                                uint32_t H, uint32_t E_ffn, uint32_t K_top,
                                uint64_t expert_stride_bytes, bool soa,
                                const std::vector<sycl::event>& deps = {});

// =====================================================================
// Gated DeltaNet ops (research/05_deltanet_math.md)
// =====================================================================

// gated_rms_norm: y[i] = (weight[i] * x[i]/rms(x)) * silu(z[i])
//
// Reduce over last dim (`hidden`). Inputs/output are conventional stride patterns:
//   x  : [n_rows, hidden] FP32 (recurrence output)
//   z  : [n_rows, hidden] FP16 (the gate stream — original input dtype)
//   weight: [hidden]      FP16
//   y  : [n_rows, hidden] FP16
sycl::event gated_rms_norm(sycl::queue& q,
                           const float* x, const sycl::half* z, const sycl::half* weight,
                           sycl::half* y,
                           uint32_t n_rows, uint32_t hidden, float eps,
                           const std::vector<sycl::event>& deps = {});

// l2_norm_scale: y[i] = x[i] * rsqrt(sum(x²) + eps) * scale
//   x, y: [n_rows, head_dim] FP32. eps is INSIDE the sqrt (FLA convention).
sycl::event l2_norm_scale(sycl::queue& q,
                          const float* x, float* y,
                          uint32_t n_rows, uint32_t head_dim,
                          float scale, float eps,
                          const std::vector<sycl::event>& deps = {});

// v1.4 fusion: cast_qkv_split + 2× repeat_interleave_heads + 2× l2_norm_scale
// in ONE launch.  src rows are [Q (skh·shd) | K (skh·shd) | V (2·skh·shd)]
// fp16; q/k outputs are tiled-repeated ([h0..h{skh-1}, h0..h{skh-1}]),
// L2-normalized per head (q with qscale, k with 1.0); v casts to fp32.
// Math identical to the unfused chain.
sycl::event dn_qkv_split_norm_fused(sycl::queue& q,
                                    const sycl::half* src,
                                    float* q_out, float* k_out, float* v_out,
                                    uint32_t T, uint32_t skh, uint32_t shd,
                                    float qscale, float eps,
                                    const std::vector<sycl::event>& deps = {});

// compute_g_beta: g = -exp(A_log) * softplus(a + dt_bias);  β = sigmoid(b)
//   a, b: [n_rows, n_heads] FP32   (output of in_proj_a, in_proj_b)
//   A_log, dt_bias: [n_heads] FP32
//   g_out, beta_out: [n_rows, n_heads] FP32
sycl::event compute_g_beta(sycl::queue& q,
                           const float* a, const float* b,
                           const float* A_log, const float* dt_bias,
                           float* g_out, float* beta_out,
                           uint32_t n_rows, uint32_t n_heads,
                           const std::vector<sycl::event>& deps = {});

// Fused fp16-input variant of `compute_g_beta` — eliminates the two
// cast_fp16_to_fp32 launches that precede it on the decode hot path.
// Same math; reads `a_h16`, `b_h16` as FP16 and converts inline.
sycl::event compute_g_beta_h16(sycl::queue& q,
                               const sycl::half* a_h16, const sycl::half* b_h16,
                               const float* A_log, const float* dt_bias,
                               float* g_out, float* beta_out,
                               uint32_t n_rows, uint32_t n_heads,
                               const std::vector<sycl::event>& deps = {});

// deltanet_recurrence: gated delta-rule scan, T-agnostic (decode + prefill).
//   q, k, v   : [B, T, n_v_heads, head_dim] FP32, post-l2norm and post-Q-scale
//   g, beta   : [B, T, n_v_heads] FP32
//   state     : [B, n_v_heads, k_head_dim, v_head_dim] FP32, in/out
//   out       : [B, T, n_v_heads, v_head_dim] FP32
//
// Phase 5 v1 only supports k_head_dim == v_head_dim == 128 (the Qwen3.6 shape).
sycl::event deltanet_recurrence(sycl::queue& q,
                                const float* q_in, const float* k_in, const float* v_in,
                                const float* g_in, const float* beta_in,
                                float* state,
                                float* out,
                                uint32_t B, uint32_t T,
                                uint32_t n_v_heads,
                                uint32_t k_head_dim, uint32_t v_head_dim,
                                const std::vector<sycl::event>& deps = {});

// Depthwise causal conv1d (kernel=4 in Qwen3.6 DeltaNet).
//
//   x: [T, channels] FP16   (T tokens, channels-last; in Qwen this is Q+K+V concat = 8192)
//   w: [kernel, channels] FP16    (one filter per channel, no cross-channel mixing)
//   conv_state: [kernel-1, channels] FP16  in/out, holds the last (kernel-1) tokens
//                                         from the previous call (for streaming decode)
//   y: [T, channels] FP16
//
// Causal: y[t, c] = sum_{k=0..kernel-1} x_padded[t - k, c] * w[k, c]
//   where x_padded[t < 0, c] is read from `conv_state` (last (kernel-1) tokens before
//   the current chunk). On exit, `conv_state` contains the last (kernel-1) tokens of
//   the current x. Pass nullptr / a zeroed state for fresh prefill from t=0.
//
// Activation (silu) is NOT applied here — caller can chain.
sycl::event depthwise_conv1d_causal(sycl::queue& q,
                                    const sycl::half* x, const sycl::half* w,
                                    sycl::half* conv_state,
                                    sycl::half* y,
                                    uint32_t T, uint32_t channels, uint32_t kernel,
                                    const std::vector<sycl::event>& deps = {});

// Per-row symmetric INT8 KV quantization helpers.
//
// Shapes are contiguous rows:
//   k_src/v_src   : [n_rows, head_dim] FP16
//   k_dst/v_dst   : [n_rows, head_dim] INT8, values clamped to [-127, 127]
//   k_scales/v_scales : [n_rows] FP16, max_abs(row) / 127 (or 1 for all-zero rows)
//
// These are validation/general-purpose ops. The decode FA-2 INT8 path keeps
// dequantization fused into the attention tile load to avoid materializing a
// full fp16 cache on the hot path.
sycl::event quantize_kv_to_int8(sycl::queue& q,
                                const sycl::half* k_src,
                                const sycl::half* v_src,
                                int8_t* k_dst,
                                int8_t* v_dst,
                                sycl::half* k_scales,
                                sycl::half* v_scales,
                                uint32_t n_rows,
                                uint32_t head_dim,
                                const std::vector<sycl::event>& deps = {});

sycl::event dequantize_kv_from_int8(sycl::queue& q,
                                    const int8_t* k_src,
                                    const int8_t* v_src,
                                    const sycl::half* k_scales,
                                    const sycl::half* v_scales,
                                    sycl::half* k_dst,
                                    sycl::half* v_dst,
                                    uint32_t n_rows,
                                    uint32_t head_dim,
                                    const std::vector<sycl::event>& deps = {});

// Full-attention forward (Qwen3.6 hybrid arch — applied to layers where i % 4 == 3).
//
// Inputs (all device USM, FP16):
//   q       : [T, n_q_heads, head_dim]   already QK-normed and RoPE'd
//   k_in    : [T, n_kv_heads, head_dim]  already QK-normed and RoPE'd
//   v_in    : [T, n_kv_heads, head_dim]
//   k_cache : [n_kv_heads, max_ctx, head_dim] for THIS layer (already sliced)
//   v_cache : same
// Parameters:
//   start_pos : where in the cache the first new token goes
//   T         : number of new tokens (1 for decode, T>1 for prefill)
//   n_q_heads, n_kv_heads, head_dim, max_ctx
// Output:
//   y       : [T, n_q_heads, head_dim] FP16 — pre-output-gate, pre-out_proj
//
// Side-effect: writes (k_in, v_in) into (k_cache, v_cache) at positions
// [start_pos, start_pos + T).
//
// Performs scaled-dot-product attention with a causal mask: each output token
// at index t (in [0, T)) attends to KV positions [0, start_pos + t]. GQA: Q
// head h reads from KV head `h * n_kv_heads / n_q_heads`.
sycl::event full_attention(sycl::queue& q,
                           const sycl::half* q_in,
                           const sycl::half* k_in,
                           const sycl::half* v_in,
                           sycl::half* k_cache,
                           sycl::half* v_cache,
                           sycl::half* y,
                           uint32_t T,
                           uint32_t start_pos,
                           uint32_t n_q_heads,
                           uint32_t n_kv_heads,
                           uint32_t head_dim,
                           uint32_t max_ctx,
                           const std::vector<sycl::event>& deps = {});

// gpt-oss attention: full_attention + two gpt-oss pieces, as a COPY (crown's
// full_attention stays byte-identical). (1) per-head softmax SINK `sinks[h]` — a
// virtual always-attended key with NO value, folded into the denominator only
// (mass leaks out); (2) sliding-window `window` — query at pos p attends keys
// (p-window, p]; window==0 → full causal. Used for BOTH prefill and decode (T=1
// works in naive attention; the split-K decode + sink is a later perf lever).
// head_dim ≤ 256 (dpl ≤ 16). sinks==nullptr / window==0 → identical to full_attention.
sycl::event full_attention_gptoss(sycl::queue& q,
                                  const sycl::half* q_in,
                                  const sycl::half* k_in,
                                  const sycl::half* v_in,
                                  sycl::half* k_cache,
                                  sycl::half* v_cache,
                                  sycl::half* y,
                                  uint32_t T,
                                  uint32_t start_pos,
                                  uint32_t n_q_heads,
                                  uint32_t n_kv_heads,
                                  uint32_t head_dim,
                                  uint32_t max_ctx,
                                  uint32_t window,
                                  const float* sinks,
                                  const std::vector<sycl::event>& deps = {});

// Query-row-block FA-2 prefill (T>1). Drop-in for full_attention. Keeps naive's
// ALU-efficient split-head-dim + subgroup-reduce inner loop, but runs ONE query
// row per SUBGROUP with Br rows/WG sharing a K/V SLM tile → cuts naive's
// redundant KV HBM reads ~Br× while staying fully occupied. Any head_dim.
// Measured 2.05× vs naive @ T=16384 (past-L2 regime), parity ≤4K. See
// attention.cpp for the full roofline rationale + the two rejected variants.
sycl::event full_attention_fa2_prefill_v2(sycl::queue& q,
                                          const sycl::half* q_in,
                                          const sycl::half* k_in,
                                          const sycl::half* v_in,
                                          sycl::half* k_cache,
                                          sycl::half* v_cache,
                                          sycl::half* y,
                                          uint32_t T,
                                          uint32_t start_pos,
                                          uint32_t n_q_heads,
                                          uint32_t n_kv_heads,
                                          uint32_t head_dim,
                                          uint32_t max_ctx,
                                          const std::vector<sycl::event>& deps = {});

// "Tile" FA-2 prefill — a faithful structural port of llama.cpp-SYCL's
// flash_attn_tile (fattn-tile.hpp). Big WG (256 work-items = 16 subgroups of
// 16) processes one q_head × a 16-query-row block; all lanes cooperatively
// stream half2-vectorized COALESCED K/V SLM tiles (nbatch_fa=64 KV positions),
// compute the full [16×64] Q·Kᵀ score tile + online softmax + [16×64]·[64×128]
// P·V — no per-key subgroup reduce (unlike v2). hd=128 fp16 causal only; falls
// through to v2 for any other head_dim. Drop-in for full_attention. Gated A/B
// via IE_FA2_PREFILL_TILE. SIMD (sub_group/SLM/half2) only — no joint_matrix /
// ESIMD / block2d / lsc_load.
sycl::event full_attention_fa2_prefill_tile(sycl::queue& q,
                                            const sycl::half* q_in,
                                            const sycl::half* k_in,
                                            const sycl::half* v_in,
                                            sycl::half* k_cache,
                                            sycl::half* v_cache,
                                            sycl::half* y,
                                            uint32_t T,
                                            uint32_t start_pos,
                                            uint32_t n_q_heads,
                                            uint32_t n_kv_heads,
                                            uint32_t head_dim,
                                            uint32_t max_ctx,
                                            const std::vector<sycl::event>& deps = {});

// head_dim 256/512 "wide" tile kernel (Gemma's SWA-256 + global-512 layers).
// A faithful templated generalization of full_attention_fa2_prefill_tile to
// HD∈{256,512} (Bc 64/32 to keep the K+V+Q+P SLM ≤ 128 KB/WG). Same numerics
// (unscaled-Q + post-dot 1/sqrt(HD) scale → Gemma's pre-scaled-Q gives net 1.0,
// matching naive/v2; online softmax, per-lane complete KQ dots + VKQ FMA — NO
// per-key subgroup reduce, the v2 floor). head_dim ∉ {256,512} falls through to
// v2. Drop-in for full_attention. SIMD only — no joint_matrix/ESIMD/block2d/lsc.
// `window` > 0 → sliding-window attention (key k attends iff q-window < k <= q),
// e.g. Gemma's SWA layers (window = sliding_window). 0 = full causal (default;
// Gemma global layers, Coder/crown). Out-of-window KV tiles are SKIPPED (the long-
// ctx lever: at 16K an SWA layer touches ~window keys, not ~T).
sycl::event full_attention_fa2_prefill_tile_gemma(sycl::queue& q,
                                                  const sycl::half* q_in,
                                                  const sycl::half* k_in,
                                                  const sycl::half* v_in,
                                                  sycl::half* k_cache,
                                                  sycl::half* v_cache,
                                                  sycl::half* y,
                                                  uint32_t T,
                                                  uint32_t start_pos,
                                                  uint32_t n_q_heads,
                                                  uint32_t n_kv_heads,
                                                  uint32_t head_dim,
                                                  uint32_t max_ctx,
                                                  uint32_t window = 0,
                                                  const std::vector<sycl::event>& deps = {});

// GQA head-packing variant of the tile kernel (mirrors llama-SYCL's `ncols2`
// GQA head-packing, fattn-tile.hpp). Packs G=4 GQA-sibling q-heads of the SAME
// kv_head into one WG (16 subgroups reinterpreted as G heads × Br=4 query rows),
// so the cooperatively-loaded half2 K/V SLM tile is loaded ONCE and reused by
// all G heads — removing up to gqa× redundant K/V HBM/L2 reads + the per-head
// SLM-stage barriers. Numerically IDENTICAL to full_attention_fa2_prefill_tile
// (heads independent; same online softmax, fp32 accumulate, unscaled-Q +
// post-dot scale, fp32 P). Requires gqa % 4 == 0 and head_dim==128; else falls
// through to v2. Gated A/B via IE_FA2_TILE_GQA. SIMD only — no joint_matrix /
// ESIMD / block2d / lsc_load.
sycl::event full_attention_fa2_prefill_tile_gqa(sycl::queue& q,
                                                const sycl::half* q_in,
                                                const sycl::half* k_in,
                                                const sycl::half* v_in,
                                                sycl::half* k_cache,
                                                sycl::half* v_cache,
                                                sycl::half* y,
                                                uint32_t T,
                                                uint32_t start_pos,
                                                uint32_t n_q_heads,
                                                uint32_t n_kv_heads,
                                                uint32_t head_dim,
                                                uint32_t max_ctx,
                                                const std::vector<sycl::event>& deps = {});

// Bounded register-staging variant of the tile kernel. IDENTICAL geometry,
// SLM layout, online softmax, P·V and numerics (unscaled-Q + post-dot scale,
// fp32 accumulate) to full_attention_fa2_prefill_tile, EXCEPT the KQ dot is
// restructured to llama-SYCL's bounded chunked register-staging (CPE=4 half2
// chunks + 2 parity accumulators per dot) to lift it off its ~15%-vector-ALU
// floor by breaking the single-accumulator carried dependency WITHOUT a
// register spill (footprint stays bounded). Output matches the default tile
// kernel within fp16 tol. Gated A/B via IE_FA2_TILE_REGTILE. Falls through to
// v2 for head_dim != 128. SIMD only — no joint_matrix / ESIMD / block2d /
// lsc_load.
sycl::event full_attention_fa2_prefill_tile_regtile(sycl::queue& q,
                                                    const sycl::half* q_in,
                                                    const sycl::half* k_in,
                                                    const sycl::half* v_in,
                                                    sycl::half* k_cache,
                                                    sycl::half* v_cache,
                                                    sycl::half* y,
                                                    uint32_t T,
                                                    uint32_t start_pos,
                                                    uint32_t n_q_heads,
                                                    uint32_t n_kv_heads,
                                                    uint32_t head_dim,
                                                    uint32_t max_ctx,
                                                    const std::vector<sycl::event>& deps = {});

// XMX-accelerated FA-2 prefill. Same FlashAttention-2 online-softmax algorithm
// as full_attention_fa2_prefill_v2, but the QK^T and P·V matmuls run on the
// B70 XMX matrix engine via SYCL joint_matrix (fp16 in, fp32 accumulate) — the
// large-T prefill regime makes both products true matrix-matrix GEMMs (~8× the
// vector-ALU throughput). Drop-in for full_attention. Optimized for head_dim
// 128 (Coder) but general. joint_matrix only — no ESIMD/block2d. See
// attention.cpp for the a/b/accumulator layout choice (QK^T uses a col_major
// b-operand for the K transpose; P·V uses row_major).
sycl::event full_attention_fa2_prefill_xmx(sycl::queue& q,
                                           const sycl::half* q_in,
                                           const sycl::half* k_in,
                                           const sycl::half* v_in,
                                           sycl::half* k_cache,
                                           sycl::half* v_cache,
                                           sycl::half* y,
                                           uint32_t T,
                                           uint32_t start_pos,
                                           uint32_t n_q_heads,
                                           uint32_t n_kv_heads,
                                           uint32_t head_dim,
                                           uint32_t max_ctx,
                                           const std::vector<sycl::event>& deps = {});

// Like full_attention but with [32]-sized per-lane registers → supports
// head_dim up to 512 (Gemma 4 global layers). Crown's full_attention untouched.
sycl::event full_attention_gemma(sycl::queue& q,
                                 const sycl::half* q_in, const sycl::half* k_in,
                                 const sycl::half* v_in,
                                 sycl::half* k_cache, sycl::half* v_cache,
                                 sycl::half* y, uint32_t T, uint32_t start_pos,
                                 uint32_t n_q_heads, uint32_t n_kv_heads,
                                 uint32_t head_dim, uint32_t max_ctx,
                                 const std::vector<sycl::event>& deps = {});

// Read-only attention over an EXISTING (shared) KV cache — no append. Used by the
// gemma4-assistant MTP draft head, whose Q-only layers attend the TARGET model's
// KV cache (it writes no K/V). A single query [n_q_heads, head_dim] attends cache
// positions [0, ctx_len). Scale 1/sqrt(head_dim) (caller pre-scales Q → net 1.0,
// like full_attention_gemma). [32]-sized regs → head_dim up to 512. See attention.cpp.
sycl::event read_attention_gemma(sycl::queue& q,
                                 const sycl::half* q_in,
                                 const sycl::half* k_cache, const sycl::half* v_cache,
                                 sycl::half* y, uint32_t ctx_len,
                                 uint32_t n_q_heads, uint32_t n_kv_heads,
                                 uint32_t head_dim, uint32_t max_ctx,
                                 const std::vector<sycl::event>& deps = {});

// FlashAttention-2 split-K, decode-only (T=1). Replaces the naive
// WG-per-q_head pattern (which re-reads KV gqa× because each q in a GQA
// group hits global memory independently). FA-2 splits the ctx into
// chunks of Bc=64 along K, dispatches WG-per-(kv_head, chunk) — KV reads
// are coalesced and cooperatively SLM-staged inside each WG, so each KV
// element is read once per layer.
//
// The first kernel writes per-chunk partials (m_local, l_local, out_local)
// to `partials_scratch`. The second kernel merges chunks per q-head with
// the standard online-softmax merge formula.
//
// Shapes:
//   q_in              : [n_q_heads, head_dim] FP16  (T=1 single-token Q)
//   k_in, v_in        : [n_kv_heads, head_dim] FP16 (T=1 K/V to append)
//   k_cache, v_cache  : [n_kv_heads, max_ctx, head_dim] FP16 (in/out)
//   y                 : [n_q_heads, head_dim] FP16 (pre-out_proj output)
//   partials_scratch  : [n_chunks_max, n_q_heads, head_dim + 2] FP32
//                       device buffer, sized for the largest ctx the
//                       caller will ever pass; Bc-chunked along ctx.
//
// `start_pos` = position of the new token; effective ctx_len = start_pos + 1.
//
// Per-call sub-kernel timing: pass a non-null AttnProfileData* (and use a
// profiling-enabled queue) to measure append / partial / combine separately.
// The struct accumulates nanoseconds with +=, so callers can sum across layers.
struct AttnProfileData {
    uint64_t append_ns  = 0;
    uint64_t partial_ns = 0;
    uint64_t combine_ns = 0;
    void reset() noexcept { append_ns = partial_ns = combine_ns = 0; }
    double append_ms()  const noexcept { return double(append_ns)  * 1e-6; }
    double partial_ms() const noexcept { return double(partial_ns) * 1e-6; }
    double combine_ms() const noexcept { return double(combine_ns) * 1e-6; }
    double total_ms()   const noexcept { return append_ms() + partial_ms() + combine_ms(); }
};

sycl::event full_attention_fa2_decode(sycl::queue& q,
                                      const sycl::half* q_in,
                                      const sycl::half* k_in,
                                      const sycl::half* v_in,
                                      sycl::half* k_cache,
                                      sycl::half* v_cache,
                                      sycl::half* y,
                                      float* partials_scratch,
                                      uint32_t start_pos,
                                      uint32_t n_q_heads,
                                      uint32_t n_kv_heads,
                                      uint32_t head_dim,
                                      uint32_t max_ctx,
                                      const std::vector<sycl::event>& deps = {},
                                      AttnProfileData* prof = nullptr);

// gpt-oss DECODE (T=1) split-K FA: COPY of full_attention_fa2_decode + the per-head
// softmax SINK folded ONCE in the combine pass (virtual key, denominator only).
// Used for the FULL-attention layers (window==0); the windowed even layers stay on
// the bounded naive full_attention_gptoss.  head_dim=64.  sinks==nullptr → base.
sycl::event full_attention_fa2_decode_gptoss(sycl::queue& q,
                                             const sycl::half* q_in,
                                             const sycl::half* k_in,
                                             const sycl::half* v_in,
                                             sycl::half* k_cache,
                                             sycl::half* v_cache,
                                             sycl::half* y,
                                             float* partials_scratch,
                                             uint32_t start_pos,
                                             uint32_t n_q_heads,
                                             uint32_t n_kv_heads,
                                             uint32_t head_dim,
                                             uint32_t max_ctx,
                                             const float* sinks,
                                             const std::vector<sycl::event>& deps = {});

// gpt-oss PREFILL (T>1) wide-tile attention: fa2_tile_wide_impl at HD=64 + per-head
// SINK.  The window param handles both SWA (even, 128) and full (odd, 0) layers.
// Replaces the naive O(T·ctx) full_attention_gptoss for prefill.
sycl::event full_attention_gptoss_prefill_tile(sycl::queue& q,
                                               const sycl::half* q_in, const sycl::half* k_in,
                                               const sycl::half* v_in, sycl::half* k_cache,
                                               sycl::half* v_cache, sycl::half* y,
                                               uint32_t T, uint32_t start_pos,
                                               uint32_t n_q_heads, uint32_t n_kv_heads,
                                               uint32_t max_ctx, uint32_t window,
                                               const float* sinks,
                                               const std::vector<sycl::event>& deps = {});

// Ground-up port of llama.cpp's flash_attn_ext_vec decode kernel — the structural
// fix for the 3.85× long-ctx decode gap (ours latency-bound/collapses, llama
// BW-bound/flat). Key difference vs our split-K kernel: warp-split so multiple KV
// positions are scored in PARALLEL with narrow (nthreads_KQ-lane) reduces, deferred
// per-tile softmax, SLM-staged scores, per-thread VKQ output. Same signature as
// full_attention_fa2_decode (drop-in). See attention.cpp. Gated A/B.
sycl::event full_attention_fa2_decode_vec(sycl::queue& q,
                                          const sycl::half* q_in,
                                          const sycl::half* k_in,
                                          const sycl::half* v_in,
                                          sycl::half* k_cache,
                                          sycl::half* v_cache,
                                          sycl::half* y,
                                          float* partials_scratch,
                                          uint32_t start_pos,
                                          uint32_t n_q_heads,
                                          uint32_t n_kv_heads,
                                          uint32_t head_dim,
                                          uint32_t max_ctx,
                                          const std::vector<sycl::event>& deps = {},
                                          AttnProfileData* prof = nullptr);

// Latency-optimized FA-2 split-K decode: identical algorithm & partials format
// to full_attention_fa2_decode, but the inner KV loop is BLOCKED (B positions per
// iteration) so the B independent score reduce_over_group calls pipeline and hide
// latency (the decode-attn kernel is ~11% peak BW = latency-bound, not BW-bound).
// Bit-identical numerics (same softmax order). Gated A/B vs the v1 kernel.
sycl::event full_attention_fa2_decode_v2(sycl::queue& q,
                                         const sycl::half* q_in,
                                         const sycl::half* k_in,
                                         const sycl::half* v_in,
                                         sycl::half* k_cache,
                                         sycl::half* v_cache,
                                         sycl::half* y,
                                         float* partials_scratch,
                                         uint32_t start_pos,
                                         uint32_t n_q_heads,
                                         uint32_t n_kv_heads,
                                         uint32_t head_dim,
                                         uint32_t max_ctx,
                                         const std::vector<sycl::event>& deps = {},
                                         AttnProfileData* prof = nullptr);

// INT8-cache variant of FA-2 split-K decode.  Same algorithm as
// `full_attention_fa2_decode`, but reads K/V from a per-row symmetric-INT8
// cache (1 fp16 scale per (kv_head, position) row of head_dim INT8s).
// Halves the long-ctx KV bandwidth.  The new token's K/V are quantized
// inline (max-abs / 127 → fp16 scale; round to int8) and written to both
// the INT8 cache and (optionally — if k_cache/v_cache != nullptr) to the
// fp16 shadow cache for any path that still needs fp16.
sycl::event full_attention_fa2_decode_int8(sycl::queue& q,
                                           const sycl::half* q_in,
                                           const sycl::half* k_in,
                                           const sycl::half* v_in,
                                           int8_t* k_int8_cache,
                                           int8_t* v_int8_cache,
                                           sycl::half* k_scales_cache,
                                           sycl::half* v_scales_cache,
                                           sycl::half* k_fp16_shadow,
                                           sycl::half* v_fp16_shadow,
                                           sycl::half* y,
                                           float* partials_scratch,
                                           uint32_t start_pos,
                                           uint32_t n_q_heads,
                                           uint32_t n_kv_heads,
                                           uint32_t head_dim,
                                           uint32_t max_ctx,
                                           const std::vector<sycl::event>& deps = {},
                                           AttnProfileData* prof = nullptr);

}  // namespace ie
