// include/ie/prefill_gemm.hpp — P4 B48: the 27B split's prefill pieces that run as whole-matrix products on the matrix
// engines. src/ops/prefill_gemm.cpp (the SYCL kernels) and src/ops/gemm_onednn_b48.cpp (the oneDNN products; a file of
// its own, so src/ops/gemm_onednn.cpp -- an input of DeepSeek-V4.1's numerics key -- stays untouched).
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace ie {

// ---- the oneDNN products (abort in a build without oneDNN: callers branch on onednn_available()) -------------------

// full_attention_prefill_gemm's two products, fp16 operands and an fp32 result. Their own primitive caches: the context
// is N of the scores and K of the output product (rounded by the caller), so each cache is emptied at 1,024 shapes.
//   scores  S [M, N] = Q [M, K] @ Kc [N, K]^T   (Kc = one kv head's cached keys, read in place)
//   output  O [M, N] = P [M, K] @ V  [K, N]     (V  = one kv head's cached values, read in place)
sycl::event gemm_attn_qk_onednn(sycl::queue& q,
                                const sycl::half* Q, const sycl::half* Kc,
                                float* S,
                                uint32_t M, uint32_t N, uint32_t K,
                                const std::vector<sycl::event>& deps = {});
sycl::event gemm_attn_pv_onednn(sycl::queue& q,
                                const sycl::half* P, const sycl::half* V,
                                float* O,
                                uint32_t M, uint32_t N, uint32_t K,
                                const std::vector<sycl::event>& deps = {});

// y [M, N] fp16 = A [M, K] fp16 @ W [N, K]^T, W = Q8_0-SoA read in place (oneDNN weights decompression, the math in
// fp16 on the matrix engines): no expand-to-fp16 pass over the weight.
//   qs [N, K]          int8, n-major (the SoA plane as it lies)
//   dt [K / group, N]  fp16 scales, kb-MAJOR (oneDNN reads attr scales in plain layout; q8_scales_kb_major makes it)
// gemm_nt_s8_onednn (ie/ops.hpp) with an fp16 result. Returns false, having submitted nothing, when oneDNN has no
// jitted kernel for the shape (a reference kernel is 25x slower than what it would replace).
bool gemm_nt_s8_f16_onednn(sycl::queue& q,
                           const sycl::half* A, const int8_t* qs, const sycl::half* dt,
                           sycl::half* y,
                           uint32_t M, uint32_t N, uint32_t K, uint32_t group,
                           sycl::event* out);
// The implementation oneDNN picked for that shape ("jit:gemm...", or the reason there is none).
std::string onednn_nt_s8_f16_impl(sycl::queue& q, uint32_t M, uint32_t N, uint32_t K, uint32_t group);

// full_attention_fa2_decode_rows (ie/ops.hpp) for a group of CONSECUTIVE positions on ONE cache -- the spec verify's
// rows: row i at start_pos + i, every row attending the rows before it. The caller appends the group's k / v rows
// first (the kernels append per kernel kind, in another order); group_end = the position after the last row. The XMX
// append leaves positions below group_end alone (it otherwise zeroes the 15 rows after each token, which would wipe
// the group's own later rows); a row's result is the single-row kernel's: the partial pass masks keys past the row's
// context, so a real row there contributes the same exact 0 a zeroed one does. The plan must be batched.
struct FaDecodeRowsPlan;
sycl::event full_attention_fa2_decode_rows_shared(sycl::queue& q,
                                                  const FaDecodeRowsPlan& plan,
                                                  const sycl::half* q_in,
                                                  const sycl::half* k_in,
                                                  const sycl::half* v_in,
                                                  sycl::half* y,
                                                  float* partials,
                                                  uint32_t layer_slot,
                                                  uint32_t group_end);

// ---- the SYCL kernels -------------------------------------------------------------------------------------------------

// d [N, nb] (fp16 bits, n-major: the Q8_0-SoA scale plane) -> dt [nb, N] (kb-major).
sycl::event q8_scales_kb_major(sycl::queue& q, const uint16_t* d, sycl::half* dt, uint32_t N, uint32_t nb);

// deltanet_recurrence (ie/ops.hpp) for a prefill chunk, B = 1, head dims 128 / 128: the same recurrence per state
// column with the column held in registers (one subgroup a column), q_t / k_t read where they lie; no SLM staging and
// no group barrier. Same arguments and layouts; the two dots of a step are summed in another order (fp32 rounding).
sycl::event deltanet_scan_prefill(sycl::queue& q,
                                  const float* q_in, const float* k_in, const float* v_in,
                                  const float* g_in, const float* beta_in,
                                  float* state, float* out,
                                  uint32_t T, uint32_t n_v_heads);

// full_attention_prefill_gemm (P4 B48) — causal full attention for a prefill chunk as whole-matrix products on the
// matrix engines: per kv head and per block of query rows (the heads that share the kv head, stacked),
//   S = Q @ K^T          oneDNN, fp16 x fp16 -> fp32, the cached keys read in place
//   P = exp(S*scale - rowmax) * 2^10 as fp16, 0 above the diagonal; L = the row sums of the rounded P
//   O = P @ V            oneDNN, fp16 x fp16 -> fp32, the cached values read in place
//   y = O / L
// Same contract as full_attention (appends k_in / v_in at [start_pos, start_pos + T), then attends; y [T, n_q, hd]).
// The 2^10 keeps weights down to 6e-8 of a row's largest in fp16's normal range. Numerics differ from the other
// prefill kernels in the last bits (the weights are rounded to fp16 before the output product).
//
// Shapes are rounded so that a primitive is built once and reused (oneDNN's GPU matmul has no jitted kernel for
// runtime dimensions, and building one costs ~4 ms): the key count to a multiple of kAttnGemmKeyStep (never above
// max_ctx), the token count to a multiple of kAttnGemmRowStep, the rows of a block to a power of two. Rounded-up rows
// read whatever q_in / the scratch hold there and their results are never read; rounded-up keys get a zero weight,
// and the value cache's rows [ctx, rounded ctx) are ZEROED here so that 0 * value stays 0 -- rows past the cache's
// length, which the next append overwrites.
//
// The scratch is the caller's, on q's device, sized for T_cap tokens (a multiple of kAttnGemmRowStep; q_in must hold
// T_cap token rows too). Everything is enqueued on q (in order); nothing waits.
constexpr uint32_t kAttnGemmKeyStep = 512;
constexpr uint32_t kAttnGemmRowStep = 64;

struct AttnGemmScratch {
    sycl::half* qh = nullptr;   // [n_q_heads, T_cap, head_dim]  q_in, head-major
    float*      s  = nullptr;   // [rows, keys]                  scores
    sycl::half* p  = nullptr;   // [rows, keys]                  weights
    float*      o  = nullptr;   // [gqa * T_cap, head_dim]       P @ V
    float*      l  = nullptr;   // [gqa * T_cap]                 row sums
    uint64_t    sp_elems = 0;   // capacity of s and of p, in elements
    // full_attention_prefill_gemm_sel only: one bit a (token, key), [T_cap, mask_words] words.
    uint32_t*   mask = nullptr;
    uint32_t    mask_words = 0;
};

// Rows per block for `rows_all` (rounded) query rows of one kv head at `keys` (rounded) keys: all of them when
// rows_all * keys fits sp_elems, else the largest power of two that does (never below 16: the caller sizes sp_elems
// for 16 rows at its largest context).
inline uint32_t attn_gemm_rows(uint32_t rows_all, uint32_t keys, uint64_t sp_elems) {
    const uint64_t fit = keys ? sp_elems / keys : rows_all;
    if (fit >= rows_all) return rows_all;
    uint32_t r = 16;
    while (uint64_t(r) * 2 <= fit) r *= 2;
    return r;
}

sycl::event full_attention_prefill_gemm(sycl::queue& q,
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
                                        const AttnGemmScratch& scr,
                                        const std::vector<sycl::event>& deps = {});

// The same attention over a SELECTED key set per token (P4 B49: Flash-Next's sparse attention, where a token attends
// the keys an indexer picked plus its own tail). sel_rows [T, sel_stride] lists token t's n_sel_rows[t] key positions
// (each <= start_pos + t); the lists become a bit mask and a key outside a token's list gets weight 0, so the result
// is the softmax over exactly the listed keys. The caller has already appended the chunk's k / v rows. The scores are
// computed for every key up to the chunk's end, so the cost grows with the context where a gathering kernel's does
// not: the caller bounds the context it sends here. Needs scr.mask with (rounded keys + 31) / 32 <= scr.mask_words.
sycl::event full_attention_prefill_gemm_sel(sycl::queue& q,
                                            const sycl::half* q_in,
                                            sycl::half* k_cache,
                                            sycl::half* v_cache,
                                            sycl::half* y,
                                            uint32_t T,
                                            uint32_t start_pos,
                                            uint32_t n_q_heads,
                                            uint32_t n_kv_heads,
                                            uint32_t head_dim,
                                            uint32_t max_ctx,
                                            const int32_t* sel_rows, uint32_t sel_stride,
                                            const int32_t* n_sel_rows,
                                            const AttnGemmScratch& scr);

}  // namespace ie
