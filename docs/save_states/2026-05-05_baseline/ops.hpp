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

// repeat_interleave_heads: replicate each KV head `repeat` times along the head
// axis. y[t, h_out, d] = x[t, h_out / repeat, d].
//   x: [T, n_in_heads, head_dim]   (FP32)
//   y: [T, n_in_heads * repeat, head_dim]  (FP32)
sycl::event repeat_interleave_heads(sycl::queue& q,
                                    const float* x, float* y,
                                    uint32_t T, uint32_t n_in_heads,
                                    uint32_t head_dim, uint32_t repeat,
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

// Multi-row GEMM with Q4_K weights, M ≤ 8 per launch. Caller chunks M
// into M_TILE-sized passes for larger T. Amortizes the weight-read cost
// M× across rows by SLM-staging A[M, K]. See gemv_q4_K for layout.
sycl::event gemm_q4_K(sycl::queue& q,
                      const sycl::half* A, const void* W_packed,
                      sycl::half* y,
                      uint32_t M, uint32_t K, uint32_t N,
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
// `recent_ids` is host pointer — caller passes a small device buffer with the
// repetition window. `penalty == 1.0` is a no-op (skip).
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

// =====================================================================
// Fused MoE decode kernels — collapse the 47 launches/layer of the naive
// per-expert loop into 2-3 kernels. T=1 only (decode); prefill keeps the
// scalar gemv_q_T loop. Reads topk_idx/topk_w directly from device — no
// host roundtrip.
// =====================================================================

// XMX (joint_matrix) variant of moe_prefill_gate_up_silu_q4k.  Same
// signature, but the inner dot product runs on XMX.  Falls back to the
// scalar variant when caller's shapes don't match (H % 256 != 0 etc.).
sycl::event moe_prefill_gate_up_silu_q4k_xmx(sycl::queue& q,
                                             const sycl::half* x,
                                             const void* gate_W, const void* up_W,
                                             const uint32_t* expert_offsets,
                                             const int32_t* sorted_token_idx,
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
sycl::event moe_prefill_gate_up_silu_q4k(sycl::queue& q,
                                         const sycl::half* x,            // [T, H] all tokens
                                         const void* gate_W, const void* up_W,
                                         const uint32_t* expert_offsets, // [E+1] device
                                         const int32_t* sorted_token_idx,// [T*K_top] device
                                         sycl::half* h_packed,           // [T*K_top, E_ffn]
                                         uint32_t E, uint32_t H, uint32_t E_ffn,
                                         uint64_t expert_stride_bytes,
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
                                        uint64_t expert_stride_bytes,
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
                                uint64_t expert_stride_bytes,
                                const std::vector<sycl::event>& deps = {});
sycl::event moe_decode_down_q6k(sycl::queue& q,
                                const sycl::half* h,
                                const void* down_W,
                                const int32_t* topk_idx,
                                const sycl::half* topk_w,
                                sycl::half* y_out,
                                uint32_t H, uint32_t E_ffn, uint32_t K_top,
                                uint64_t expert_stride_bytes,
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
