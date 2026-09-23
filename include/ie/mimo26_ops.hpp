// include/ie/mimo26_ops.hpp — the device ops MiMo-V2.6's forward needs beyond ops.hpp (P2,
// docs/mimo26/00_PORT_PLAN.md). Conventions match ops.hpp: device-USM pointers, deps vector, returns
// sycl::event. Everything here is a plain bring-up kernel -- correctness first, the P4 levers later.
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>
#include <vector>

namespace ie {

// y[t, :] = x[t, :] / sqrt(mean(x[t, :]^2) + eps) * w   -- the fp32 residual stream in, fp16 out (a GEMM's
// A operand) and/or fp32 out (the router's and the expert tier's input). Either output may be null.
sycl::event mimo26_rms_norm(sycl::queue& q, const float* x, const float* w, sycl::half* y16, float* y32,
                            uint32_t T, uint32_t H, float eps, const std::vector<sycl::event>& deps = {});

// The fused projection's fp32 rows [T, n_q*hd + n_kv*hd + n_kv*hdv] (unsharded [Q | K | V] order) ->
// contiguous fp16 Q [T, n_q, hd], K [T, n_kv, hd], V [T, n_kv, v_stride] (the first hdv of each V row written;
// v_stride = hdv is plain, 192 is the full layers' padded V -- see mimo26_attention).
sycl::event mimo26_split_qkv(sycl::queue& q, const float* qkv, sycl::half* Q, sycl::half* K, sycl::half* V,
                             uint32_t T, uint32_t n_q, uint32_t n_kv, uint32_t hd, uint32_t hdv,
                             const std::vector<sycl::event>& deps = {}, uint32_t v_stride = 0);

// GQA attention with a K head dim `hd` and a V head dim `hdv` (192 / 128). Appends K [T, n_kv, hd] and
// V [T, n_kv, v_stride] (first hdv of each row) at positions [pos0, pos0 + T) of the layer's caches
// k_cache [n_kv, slots, hd] / v_cache [n_kv, slots, v_stride], then y [T, n_q, hdv] = softmax(Q K^T / sqrt(hd)) V
// over the causal keys, `window` > 0 restricting query position p to keys (p - window, p] (llama.cpp's standard
// SWA), and `sinks` [n_q] (may be null) a per-head virtual key of that logit and no value (mass leaks out of the
// softmax, as in gpt-oss / ggml_soft_max_ext with sinks). `ring` = 0: slots = max_ctx, position p at slot p;
// `ring` = R > 0: a ring of R slots, position p at slot p mod R -- needs R >= window + T (the chunk's own keys plus
// the window before it; P3a). v_stride 0 = hdv. hd, hdv multiples of 16 and <= 256.
sycl::event mimo26_attention(sycl::queue& q, const sycl::half* Q, const sycl::half* K, const sycl::half* V,
                             sycl::half* k_cache, sycl::half* v_cache, sycl::half* y,
                             uint32_t T, uint32_t pos0, uint32_t n_q, uint32_t n_kv, uint32_t hd, uint32_t hdv,
                             uint32_t max_ctx, uint32_t window, const float* sinks,
                             const std::vector<sycl::event>& deps = {}, uint32_t ring = 0, uint32_t v_stride = 0);

// Split-K decode attention for T <= 8 query rows (P3a/P4: mimo26_attention walks every key of a head serially in one
// sub-group, 680 ms/token at 32k). Same semantics and cache layout as mimo26_attention (append included, window, sinks,
// ring, v_stride), different summation order. Pass 1: a work-group per (row, kv head, key split) -- one sub-group per q
// head of the GQA group, the split's K/V staged in SLM tiles once for all of them -- writes (m, l, acc[hdv]) partials;
// pass 2 merges the splits per (row, head) and folds the sink in. `partials` holds T * n_q * max_splits * (hdv + 2)
// floats; max_splits = mimo26_decode_max_splits(). hd 192 and hdv 128 (compile-time in the kernel), n_q / n_kv <= 16.
constexpr uint32_t mimo26_decode_max_splits() { return 64; }
sycl::event mimo26_attention_decode(sycl::queue& q, const sycl::half* Q, const sycl::half* K, const sycl::half* V,
                                    sycl::half* k_cache, sycl::half* v_cache, sycl::half* y, float* partials,
                                    uint32_t T, uint32_t pos0, uint32_t n_q, uint32_t n_kv, uint32_t hd, uint32_t hdv,
                                    uint32_t max_ctx, uint32_t window, const float* sinks,
                                    const std::vector<sycl::event>& deps = {}, uint32_t ring = 0, uint32_t v_stride = 0);

// XMX prefill attention for K head dim 192 / V head dim 128 (the chunks, T > 8; P3a/P4: full_attention_fa2_prefill_xmx
// runs one q head per work-group, V padded to 192, the softmax one lane per row; mimo26_attention walks keys serially).
// Same semantics as mimo26_attention (window, sinks, ring, v_stride). A work-group takes 64 rows = 64 / gqa tokens x the
// gqa q heads of one kv head, so every K/V block is read once for all of them: S^T = Kblk . Q^T and O += P . Vblk on
// joint_matrix, the online softmax spread over 256 lanes. Throws unless n_q / n_kv divides 64 and ring % 64 == 0. A
// LINEAR k_cache needs 64 rows of slack past its last head (the last block's 8-row K tiles are read whole and masked);
// v_cache rows past the causal end are never read.
sycl::event mimo26_attention_prefill_xmx(sycl::queue& q, const sycl::half* Q, const sycl::half* K, const sycl::half* V,
                                         sycl::half* k_cache, sycl::half* v_cache, sycl::half* y,
                                         uint32_t T, uint32_t pos0, uint32_t n_q, uint32_t n_kv, uint32_t max_ctx,
                                         uint32_t window, const float* sinks, const std::vector<sycl::event>& deps = {},
                                         uint32_t v_stride = 0, uint32_t ring = 0);

// y [T, n_q, hdv] <- x [T, n_q, x_stride] (the first hdv of every head row): the full layers' XMX FA-2 output
// (192-wide rows over the padded V) compacted for o_proj.
sycl::event mimo26_compact_heads(sycl::queue& q, const sycl::half* x, sycl::half* y, uint32_t T, uint32_t n_q,
                                 uint32_t x_stride, uint32_t hdv, const std::vector<sycl::event>& deps = {});

// y [M, N] (fp32) = x [M, K] (fp16, row stride x_stride) . W^T for an FP8 weight kept as the checkpoint stores it (P4 lever
// 3): w8 E4M3 [N, K] bytes and srow [N, K / 128] per-row F32 scales (mimo26_fp8_rows). y[m][n] = sum over the 128-column
// blocks b of srow[n][b] * sum_k x[m][k] * e4m3(w8[n][k]): fp32 products and sums, no fp16 rounding of the weight (the
// fp16-dequant path rounds every weight to 11 bits). M 1..8 (the decode steps); K a multiple of 128.
sycl::event mimo26_gemv_fp8(sycl::queue& q, const sycl::half* x, uint32_t x_stride, const uint8_t* w8, const float* srow,
                            float* y, uint32_t M, uint32_t K, uint32_t N, const std::vector<sycl::event>& deps = {});

// out [N, K] fp16 = half(e4m3(w8[n][k]) * srow[n][k / 128]) -- the fp16 weight a chunk's oneDNN GEMM reads, element for
// element the host dequant's (mimo26_fp8_rows_to_f32, then fp16). K a multiple of 128.
sycl::event mimo26_fp8_to_f16(sycl::queue& q, const uint8_t* w8, const float* srow, sycl::half* out, uint32_t N, uint32_t K,
                              const std::vector<sycl::event>& deps = {});

// logits[t, e] = sum_k x[t, k] * w[e, k], all fp32 (the router: llama.cpp keeps ffn_gate_inp in f32).
sycl::event mimo26_router_logits(sycl::queue& q, const float* x, const float* w, float* logits,
                                 uint32_t T, uint32_t H, uint32_t E, const std::vector<sycl::event>& deps = {});

// y[i] += alpha * x[i]  (fp32 residual adds; alpha = 1, or the attention value scale 0.707).
sycl::event mimo26_axpy(sycl::queue& q, const float* x, float alpha, float* y, uint64_t n,
                        const std::vector<sycl::event>& deps = {});

// y[i] = silu(gate[i]) * up[i], fp32 in (two GEMM outputs) -> fp16 out (the down GEMM's A operand).
sycl::event mimo26_swiglu_f32(sycl::queue& q, const float* gate, const float* up, sycl::half* y, uint64_t n,
                              const std::vector<sycl::event>& deps = {});

}  // namespace ie
