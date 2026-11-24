// include/ie/xing4_ops.hpp -- the Xing4 kernels (src/ops/xing4_ops.cpp). Every kernel name carries the Xg4 prefix
// (same-named helpers in two translation units run the wrong device code: docs/ENGINE_PLAYBOOK.md section 2).
#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>

namespace ie {

// A weight matrix on the card, any of four formats (row-major planes, rows independent):
//   fmt 0 Q8_0: qs [N][K] int8, d [N][K/32] fp16
//   fmt 1 Q6_K: qs [N][K/2] (ggml ql), qh [N][K/4], sc [N][K/16] int8, d [N][K/256] fp16
//   fmt 2 Q5_K: qs [N][K/2], qh [N][K/8], sc [N][K/32][2] (6-bit scale, min unpacked), d [N][K/256][2] (d, dmin)
//   fmt 3 Q4_K: qs [N][K/2], sc [N][K/32][2], d [N][K/256][2]
// Values exactly as ggml's dequantize_row_q{8_0,6_K,5_K,4_K}. Stacked matrices (experts, heads) are more rows.
struct Xg4W {
    const int8_t* qs = nullptr; const uint16_t* d = nullptr; const uint8_t* qh = nullptr; const uint8_t* sc = nullptr;
    uint32_t fmt = 0;
};
enum : uint32_t { kXg4Q8 = 0, kXg4Q6K = 1, kXg4Q5K = 2, kXg4Q4K = 3 };

// Activations rounded to int8 for the integer-dot path: per row t of K values, q [t*K + k], per 32 values the scale
// d [t*(K/32) + b] (amax / 127) and the sum of the rounded values times d, s [t*(K/32) + b] (the K-quant minimum term).
// Rounded from the fp16 activation. q == nullptr: the kernels read the fp16 activations instead.
struct Xg4A { int8_t* q = nullptr; float* d = nullptr; float* s = nullptr; };
// x16 [T, ldx] -> a (rows of K)
sycl::event xg4_act_q8(sycl::queue& q, const sycl::half* x, uint32_t ldx, uint32_t K, uint32_t T, Xg4A a);

// --- hyper-connections (4 streams, Sinkhorn), exact to modeling_xing4_0.Xing4_0HyperConnection ------------------------
// S [T, hc * H] fp32. part [T, 25, kXg4HcParts]: for m < 24 the dot of fn row m with the token's streams, m = 24 their
// sum of squares, in kXg4HcParts segments (the gate kernel sums them in a fixed order).
constexpr uint32_t kXg4HcParts = 28;
sycl::event xg4_hc_parts(sycl::queue& q, const float* S, const float* fn, float* part, uint32_t T, uint32_t HH);
// Per token: pre = sigmoid(mix_i r s0 + b_i), post = 2 sigmoid(mix_{4+i} r s1 + b_{4+i}), comb = Sinkhorn(exp(clamp(logits) -
// rowmax)) with `iters` (row, column) rounds, r = rsqrt(mean(S^2) + eps); collapsed = sum_i pre_i S_i; then RMSNorm with w ->
// x16 (fp16) and x32 (fp32, may be null). post [T, 4], comb [T, 16] (row i = the weights of new stream i over old j).
sycl::event xg4_hc_gate(sycl::queue& q, const float* S, const float* part, const float* base, const float* scale, const float* w,
                        float* post, float* comb, sycl::half* x16, float* x32, uint32_t T, uint32_t H, float norm_eps,
                        float hc_eps, float clamp_lo, float clamp_hi, uint32_t iters, Xg4A a = {});
// S'[t, i, d] = post[t, i] y[t, d] + sum_j comb[t, i, j] S[t, j, d]   (in place)
sycl::event xg4_hc_post(sycl::queue& q, float* S, const float* post, const float* comb, const float* y, uint32_t T, uint32_t H);
// x[j, d] = (1/4) sum_i S[rows[j], i, d] -> RMSNorm(w) -> x16[j]   (rows null: row j)
sycl::event xg4_stream_mean_norm(sycl::queue& q, const float* S, const int32_t* rows, const float* w, sycl::half* x16, uint32_t n,
                                 uint32_t H, float eps);

// --- Q8_0-SoA matrices (qs[n*K + k], d[n*(K/32) + b] fp16 bits) -------------------------------------------------------
// y[t, n] = W[n] . x[t] for T rows sharing each weight load (T <= 8 a pass); y32 and/or y16 (either may be null).
// x row stride ldx (>= K), y row stride ldy (>= N). n_mats > 1: a batch of same-shape matrices (heads): matrix h reads
// x + h * K (inside a row of stride ldx) and writes y + h * N; its weights start at h * N rows.
sycl::event xg4_gemv(sycl::queue& q, const sycl::half* x, uint32_t ldx, const Xg4W& w, float* y32,
                     sycl::half* y16, uint32_t ldy, uint32_t N, uint32_t K, uint32_t T, uint32_t n_mats = 1,
                     uint32_t x_mat_stride = 0, uint32_t y_mat_stride = 0, Xg4A a = {});   // a: x as int8 (n_mats 1)
// RMSNorm of fp32 rows [T, n] (stride ld) with w -> fp16 [T, n] (stride ld16)
sycl::event xg4_rmsnorm16(sycl::queue& q, const float* x, uint32_t ld, const float* w, sycl::half* y, uint32_t ld16,
                          uint32_t T, uint32_t n, float eps, Xg4A a = {});

// --- MLA pieces ---------------------------------------------------------------------------------------------------------
// The latent cache is paged: kXg4Page keys a page, tab [lanes][maxp] the pages of each lane (the same for every layer),
// a layer's pool [pages * kXg4Page, 576] fp16. Row r of a call belongs to lane rlane[r] at position rpos[r].
constexpr uint32_t kXg4Page = 64;
// kv [T, ldkv] fp32 (the kv_a output at column 0): latent = RMSNorm(kv[:512]) * w, rope part rotated at the row's position
// (cs [T, 64]: 32 cosines then 32 sines; interleaved pairs in, the reference's de-interleaved order out) -> the row's slot.
sycl::event xg4_kv_append(sycl::queue& q, const float* kv, uint32_t ldkv, const float* w, const float* cs, sycl::half* pool,
                          const int32_t* tab, uint32_t maxp, const int32_t* rlane, const int32_t* rpos, uint32_t T, float eps);
// cs [T, 64] for the rows' positions (YaRN frequencies inv_freq [32])
sycl::event xg4_rope_rows(sycl::queue& q, const int32_t* rpos, const float* inv_freq, float* cs, uint32_t T);
// S [T, 4, H] = the token's embedding (a Q8_0-SoA table [vocab, H]) in each stream
sycl::event xg4_embed(sycl::queue& q, const int32_t* tok, const Xg4W& w, float* S, uint32_t T, uint32_t H);
// attention over the paged cache (src/ops/xing4_attn.cpp): ctx16 [rows, NH, 576] (the last 64 written as zeros)
sycl::event xg4_attn_decode(sycl::queue& q, const sycl::half* a16, const sycl::half* pool, const int32_t* tab, uint32_t maxp,
                            const int32_t* rlane, const int32_t* rpos, sycl::half* ctx16, uint32_t rows, uint32_t NH, float scaling,
                            uint32_t max_keys);
// prompt rows in groups of up to 4 rows of one lane (groups: [first row, count] pairs): a key block read once for the group
sycl::event xg4_attn_rows4(sycl::queue& q, const sycl::half* a16, const sycl::half* pool, const int32_t* tab, uint32_t maxp,
                           const int32_t* rlane, const int32_t* rpos, const int32_t* groups, uint32_t n_groups, sycl::half* ctx16,
                           uint32_t NH, float scaling);
sycl::event xg4_attn_rows(sycl::queue& q, const sycl::half* a16, const sycl::half* pool, const int32_t* tab, uint32_t maxp,
                          const int32_t* rlane, const int32_t* rpos, sycl::half* ctx16, uint32_t rows, uint32_t NH, float scaling);
// q [T, NH, 192] fp32 -> qn16 [T, NH, 128] fp16 (the nope part) and a16 [T, NH, 576] columns 512.. = the roped part.
sycl::event xg4_q_split(sycl::queue& q, const float* qf, const float* cs, sycl::half* qn16, sycl::half* a16, uint32_t T, uint32_t NH);

// --- MoE ----------------------------------------------------------------------------------------------------------------
// Router: logits = Wg (fp32 [E, H]) . x32; top-k of sigmoid + bias; weights = the chosen sigmoids (normalised) * scale.
// Writes job_e [T, k + 1] (the last of a token's jobs = the shared expert, e = -1) and job_w [T, k + 1] (shared: 1).
sycl::event xg4_router(sycl::queue& q, const float* x32, const float* wg, const float* bias, int32_t* job_e, float* job_w,
                       uint32_t T, uint32_t H, uint32_t E, uint32_t k, bool norm, float scale, float* logit_ws);   // logit_ws [T, E]
// h[j, r] = silu(G_e[r] . x[t]) * (U_e[r] . x[t]) for job j = t * per_tok + s, e = job_e[j] (< 0: the shared matrices; job_e
// null: every job is shared). Banks: [E][N][K] SoA planes (gate, up separately).
sycl::event xg4_moe_gate_up(sycl::queue& q, const sycl::half* x, const Xg4W& g, const Xg4W& u, const Xg4W& gs, const Xg4W& us,
                            const int32_t* job_e, float* h, uint32_t T, uint32_t per_tok,
                            uint32_t N, uint32_t K, Xg4A a = {});
// y[t, d] = sum_s job_w[t, s] * (D_e[d] . h[t * per_tok + s])   (job_w null: weight 1)
sycl::event xg4_moe_down(sycl::queue& q, const sycl::half* h, const float* hs, const Xg4W& dn, const Xg4W& ds, const int32_t* job_e, const float* job_w, float* y, uint32_t T, uint32_t per_tok,
                         uint32_t N, uint32_t K, Xg4A a = {});

// Expert-grouped MoE (batches of rows: decode lanes, prompt chunks). xg4_moe_group sorts the jobs by expert: off [E + 2]
// (group E = the shared expert), list [J] = job indices. The grouped kernels read each expert's weights once per 8 of its
// jobs; job_e / off / list unused for a dense layer (dense = true: one group, jobs 0 .. J). down writes per-job rows yj
// [J, N] (times job_w), xg4_moe_sum adds a token's per_tok rows in order.
sycl::event xg4_moe_group(sycl::queue& q, const int32_t* job_e, uint32_t J, uint32_t E, int32_t* off, int32_t* list);
sycl::event xg4_moe_gate_up_grp(sycl::queue& q, const sycl::half* x, const Xg4W& g, const Xg4W& u, const Xg4W& gs, const Xg4W& us,
                                const int32_t* off, const int32_t* list, float* h, uint32_t J,
                                uint32_t per_tok, uint32_t N, uint32_t K, uint32_t E, bool dense, Xg4A a = {});
sycl::event xg4_moe_down_grp(sycl::queue& q, const sycl::half* hh, const float* hs, const Xg4W& dn, const Xg4W& ds, const int32_t* off, const int32_t* list, const float* job_w, float* yj,
                             uint32_t J, uint32_t N, uint32_t K, uint32_t E, bool dense, Xg4A a = {});
// h (fp32) -> fp16 with a per-row power-of-two scale hs where a row would overflow fp16 (1 elsewhere)
sycl::event xg4_h_pack(sycl::queue& q, const float* h, sycl::half* h16, float* hs, uint32_t J, uint32_t N);
sycl::event xg4_moe_sum(sycl::queue& q, const float* yj, float* y, uint32_t T, uint32_t per_tok, uint32_t N);

// XMX versions for prompt chunks (src/ops/xing4_moe_xmx.cpp): tiles of 32 jobs of one group. xg4_moe_tiles writes the
// tile list (tiles[0] = count, then (group, first row) pairs) from off; max_tiles = J / 32 + E + 1 bounds it.
sycl::event xg4_moe_tiles(sycl::queue& q, const int32_t* off, uint32_t E, int32_t* tiles, uint32_t max_tiles);
sycl::event xg4_moe_gate_up_xmx(sycl::queue& q, const sycl::half* x, const int8_t* g_qs, const uint16_t* g_d, const int8_t* u_qs,
                                const uint16_t* u_d, const int8_t* gs_qs, const uint16_t* gs_d, const int8_t* us_qs,
                                const uint16_t* us_d, const int32_t* off, const int32_t* list, const int32_t* tiles,
                                uint32_t max_tiles, float* h, uint32_t J, uint32_t per_tok, uint32_t N, uint32_t K,
                                uint32_t E, bool dense);
sycl::event xg4_moe_down_xmx(sycl::queue& q, const sycl::half* hh, const float* hs, const int8_t* d_qs, const uint16_t* d_d, const int8_t* ds_qs,
                             const uint16_t* ds_d, const int32_t* off, const int32_t* list, const int32_t* tiles,
                             uint32_t max_tiles, const float* job_w, float* yj, uint32_t J, uint32_t N, uint32_t K, uint32_t E,
                             bool dense);
sycl::event xg4_q8_dequant(sycl::queue& q, const int8_t* qs, const uint16_t* d, sycl::half* w16, uint32_t N, uint32_t K);

}  // namespace ie
