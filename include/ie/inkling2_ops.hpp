// include/ie/inkling2_ops.hpp — Inkling 2 Mini (`inkling2_loop`) primitive ops, phase 1 (correct, not fast).
//
// Every op is the formula of docs/inkling2/SPEC.md (section numbers below) in fp32 on the device. Most take a
// `round` flag: when set, the result is rounded to the nearest bf16 (ties to even) at exactly the point where the
// publisher's reference casts back to bf16 (SPEC 9, the list under "Every place where the reference rounds").
// The engine's own arithmetic is round = false; round = true exists so a per-call comparison with the reference's
// goldens can get under its bf16 noise floor (SPEC 16.5). scripts/inkling2/ref_forward.py --bf16 is the same switch.
//
// All functions submit to `q` and return the event. They assume an IN-ORDER queue (the runtime makes one): no
// dependency lists. Pointers are device USM. Matrices are row-major; weights are stored [out, in] as in the file.
// Weight matrices are raw BF16 bits (uint16_t), activations fp32.
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>

namespace ie {

// Round fp32 to the nearest bf16, ties to even, returned as fp32 (finite inputs). Host and device.
inline float i2_bf16_round(float x) {
    uint32_t u = sycl::bit_cast<uint32_t>(x);
    u = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;
    return sycl::bit_cast<float>(u);
}
inline float i2_bf16_to_f32(uint16_t b) { return sycl::bit_cast<float>(uint32_t(b) << 16); }

// FP8 E4M3 ("fn": bias 7, no inf, 0x7F/0xFF NaN) built from bits, so the device needs no table. Same values as
// ie::e4m3_to_f32 (include/ie/fp8.hpp); tests/unit/inkling2_ops_test.cpp checks all 256 codes against it.
inline float i2_e4m3(uint8_t b) {
    const uint32_t s = b >> 7, e = (b >> 3) & 15u, m = b & 7u;
    float v;
    if (e == 0) v = float(m) * 0.001953125f;                                   // m/8 * 2^-6
    else if (e == 15 && m == 7) v = sycl::bit_cast<float>(0x7FC00000u);        // NaN
    else v = sycl::bit_cast<float>(((e + 120u) << 23) | (m << 20));            // (1 + m/8) * 2^(e-7)
    return s ? -v : v;
}
// E2M1 code -> value (SPEC 10.1): 0, 0.5, 1, 1.5, 2, 3, 4, 6, then the same negated (bit 3 = sign).
inline float i2_e2m1(uint32_t c) {
    const uint32_t m = c & 7u;
    const float v = m < 4 ? float(m) * 0.5f : (m == 4 ? 2.f : m == 5 ? 3.f : m == 6 ? 4.f : 6.f);
    return (c & 8u) ? -v : v;
}
// UE8M0 exponent byte -> 2^(b-127) for b in 1..254 (SPEC 10.2: the checkpoint's bytes are 113..118).
inline float i2_ue8m0(uint8_t b) { return sycl::bit_cast<float>(uint32_t(b) << 23); }

// y[t][n] = sum_k x[t][k] * W[n][k]   (x [T, K] fp32, W [N, K] bf16, y [T, N]); fp32 accumulation.
// Several projections of the same K (each its own input, weights and output) in one launch: the sub-group kernel
// over all their rows (fast, T <= 4, n <= 8), otherwise one generic launch each. Same values as separate calls.
struct I2LinSeg { const float* x; const uint16_t* W; uint32_t N; float* y; };
sycl::event i2_linear_bf16_multi(sycl::queue& q, uint32_t n, const I2LinSeg* seg, uint32_t T, uint32_t K, bool round,
                                 bool fast, bool rowwise = false);
// rowwise (batched decoding, T rows of different conversations): every row's sum is formed exactly as with T = 1 (the
// work-group-per-row kernel keeps serving the few-row projections for any T).
// fast: few tokens (T <= 4) use a sub-group kernel with a different fp32 summation order; false keeps the generic
// kernel's order (the bf16-emulation mode, whose per-call agreement with the reference is gated, passes false).
sycl::event i2_linear_bf16(sycl::queue& q, const float* x, uint32_t T, uint32_t K, const uint16_t* W, uint32_t N,
                           float* y, bool round, bool fast = true, bool rowwise = false);

// SPEC 1.1 / 1.2. y[r] = (w ?) w * (x[r] * rsqrt(mean(x[r]^2) + eps)). x [rows, dim]; y row r at y + r * y_stride.
sycl::event i2_rmsnorm(sycl::queue& q, const float* x, uint32_t rows, uint32_t dim, const float* w, float eps,
                       float* y, uint32_t y_stride, bool round);
// Up to 8 such norms of the same shape (each its own input, weight, output) in one launch; the same values.
struct I2NormSeg { const float* x; const float* w; float* y; };
sycl::event i2_rmsnorm_multi(sycl::queue& q, uint32_t n, const I2NormSeg* seg, uint32_t rows, uint32_t dim, float eps,
                             uint32_t y_stride, bool round);

// SPEC 1.3 / 1.4: y = L2clip(x * scale, thr) per row of `dim`. scale (nullable) is [heads, dim], row r uses
// scale row r % heads (rows = T * heads). y may alias x.
sycl::event i2_scale_l2clip(sycl::queue& q, const float* x, uint32_t rows, uint32_t dim, const float* scale,
                            uint32_t heads, float thr, float* y, bool round);

// y[t][i] = x[t][i] * f[t] for i < len (the tau multiply of SPEC 1.6). y may alias x.
sycl::event i2_mul_token(sycl::queue& q, const float* x, uint32_t T, uint32_t len, const float* f, float* y, bool round);

// SPEC 1.8 static short conv: y[t][c] = (sum_j w[c][j] * x[t-3+j][c] + x[t][c]) / sqrt(5), K = 4 taps, w [C, 4]
// (last tap = current token). Token t is at absolute position pos0 + t. `hist` is a ring of `hist_rows` rows [C] holding
// the conv's INPUTS by position (row = position % hist_rows); inputs before position 0 are zero. The kernel reads
// the 3 inputs before pos0 from it, then this call's T inputs are stored into it, so the ring needs hist_rows >= 3 and
// a caller that wants to rewind to position n later must keep hist_rows >= (positions written after n) + 3.
// x and y must not alias.
sycl::event i2_sconv_static(sycl::queue& q, const float* x, uint32_t T, uint32_t C, const float* w, float* hist,
                            uint32_t hist_rows, uint32_t pos0, float* y, bool round);

// SPEC 1.9 dynamic short conv (global layers' k / v): W[t][c][lag] = static_w[lag][c] + dyn[t][(c / (C/heads)) * 4
// + lag]; y[t][c] = (sum_lag W * x[t-lag][c] + x[t][c]) / 3 (lag 0 = current token). dyn [T, heads * 4] is the
// [32, H] projection of the linear's INPUT. hist / hist_rows / pos0 as above.
sycl::event i2_sconv_dynamic(sycl::queue& q, const float* x, uint32_t T, uint32_t C, const float* dyn,
                             const float* static_w, uint32_t heads, float* hist, uint32_t hist_rows, uint32_t pos0,
                             float* y, bool round);

// SPEC 1.10 / 5.3 attn_res read-out. Sources src[0..n-1] each [T, H] (the last one is the partial state); queries
// Q [nq, H]. out [nq][T][H] (query-major, so each query's read-out is a contiguous [T, H]). n <= 8, nq <= 4.
// Never rounded (the reference keeps the read-outs in fp32).
struct I2Sources { const float* p[8]; };
// fast: one sweep over the sources and a batched reduction (the 1/rms factor leaves the dot products: fp32 order only;
// n <= 8, nq <= 4); false: the original order (the bf16-emulation mode).
sycl::event i2_attn_res(sycl::queue& q, const I2Sources& src, uint32_t n, uint32_t T, uint32_t H, const float* Q,
                        uint32_t nq, float eps, float* out, bool fast = false, float* scratch = nullptr, bool rowwise = false);
// (fast, T <= 16 (any T when rowwise) and a scratch of T * 16 * 40 floats: the columns are cut into 16 slices over two
// launches.)

// SPEC 1.11 last step: out = (1 - tanh(f)) * h + tanh(g) * m   (n elements, fp32, never rounded).
sycl::event i2_gate_combine(sycl::queue& q, const float* h, const float* m, const float* f, const float* g, uint64_t n,
                            float* out);

// y = x * sigmoid(gate)   (SPEC 3 step 9). y may alias x.
sycl::event i2_mul_sigmoid(sycl::queue& q, const float* x, const float* gate, uint64_t n, float* y, bool round);
// y = a + b.
sycl::event i2_add(sycl::queue& q, const float* a, const float* b, uint64_t n, float* y, bool round);
// Directional ablation of a write into the stream, per token row t (one work-group per row, fp32, never rounded):
//   y[t] = (a ? a[t] : 0) + x[t] - strength * sum_k (x[t] . r_k) r_k
// r = dirs [k, dim], orthonormal rows (the caller checks), k <= kI2AblateMaxK. y may alias x (a == nullptr) or a.
// keep_norm: the ablated write is rescaled to the length of x (a norm-preserving variant, for experiments).
constexpr uint32_t kI2AblateMaxK = 256;
sycl::event i2_ablate_add(sycl::queue& q, const float* a, const float* x, uint32_t rows, uint32_t dim, const float* dirs,
                          uint32_t k, float strength, float* y, bool keep_norm = false);
// ---- batched decoding: T rows, each ONE token of its own conversation (slot). Row t is at position pos[t] and uses the
// state of slot[t]: a state array given by its slot-0 base pointer starts slot[t] * stride floats further on. Every row
// is computed exactly as the same token alone (T = 1) would be. Null pos = the usual chunk of one conversation.
struct I2Rows { const int32_t* pos = nullptr; const int32_t* slot = nullptr; };
// ---- fused fp32-path kernels (phase 4: fewer launches; same values as the separate kernels up to reduction order)
// The per-head work between the projections and the attention core, one launch: query scale + L2 clip (+ tau on a
// global layer) and the relative states' scale + clip (+ tau); key / value causal conv (static, or dynamic with dyn_k /
// dyn_v) over the input ring (stores this chunk's inputs), key clip / value norm, the cache-row stores. HD <= 128.
struct I2AttnPrep {
    uint32_t T = 0, pos0 = 0, nh = 0, nkv = 0, hd = 0, d_rel = 0, crow = 0, ring = 0;
    bool glob = false;
    float qk_clip = 0.f, rel_clip = 0.f, eps = 0.f;
    float *qb = nullptr, *rb = nullptr;
    const float *q_scale = nullptr, *rel_scale = nullptr, *tau = nullptr;
    const float *kp = nullptr, *vp = nullptr, *dyn_k = nullptr, *dyn_v = nullptr, *k_conv = nullptr, *v_conv = nullptr;
    float *ck = nullptr, *cv = nullptr, *Kc = nullptr, *Vc = nullptr;
    I2Rows rows;                                                    // batched: per-row position / slot
    uint64_t kv_stride = 0, c_stride = 0;                           // elements per slot: Kc / Vc, ck / cv
    bool kv16 = false;                                              // Kc / Vc hold bf16 (uint16) values
};
sycl::event i2_attn_prep(sycl::queue& q, const I2AttnPrep& p);
// y = static conv of x (as i2_sconv_static, inputs stored into the ring) and res = a + y (a null: res = y; res null: none).
sycl::event i2_conv_add(sycl::queue& q, const float* x, uint32_t T, uint32_t C, const float* w, float* hist,
                        uint32_t hist_rows, uint32_t pos0, float* y, const float* a, float* res, I2Rows rows = {},
                        uint64_t h_stride = 0, const float* addv = nullptr);   // addv [C]: res = a + (y + addv)
// y[t] = x[t] + v (v [H], every row); y may alias x. Steering on the unfused paths.
sycl::event i2_add_rows(sycl::queue& q, const float* x, uint32_t T, uint32_t H, const float* v, float* y);
// The MoE tail: m = expert combine (as i2_expert_combine), y = static conv of m (m stored into the ring), out = a + y.
sycl::event i2_moe_out(sycl::queue& q, const float* ey, const float* wts, uint32_t T, uint32_t k, uint32_t H,
                       const float* w, float* hist, uint32_t hist_rows, uint32_t pos0, float* y, const float* a, float* out,
                       I2Rows rows = {}, uint64_t h_stride = 0, const float* addv = nullptr);
// A stream write with an ablation set folded in (kd <= 4 directions): y = the write as i2_conv_add (moe false: conv of
// x [T, H]) or i2_moe_out (moe true: conv of the expert combine of x = ey [T, k_top, H] with wts), before the removal;
// res = a + remove(y) (a null: remove(y)), remove as i2_ablate_add. part: T * ceil(H / 256) * 5 floats; cnt: T uint32,
// zero on entry, left zero. One launch per write.
struct I2WriteAbl {
    bool moe = false, keep_norm = false;
    uint32_t T = 0, H = 0, k_top = 0, hist_rows = 0, pos0 = 0, kd = 0;
    const float *x = nullptr, *wts = nullptr, *w = nullptr, *a = nullptr, *dirs = nullptr;
    float *hist = nullptr, *y = nullptr, *res = nullptr, *part = nullptr;
    uint32_t* cnt = nullptr;
    float strength = 1.f;
    I2Rows rows;                                                    // batched: per-row position / slot
    uint64_t h_stride = 0;                                          // floats per slot of hist
    const float* addv = nullptr;                                    // steering [H]: res = a + (remove(y) + addv)
};
sycl::event i2_write_ablate(sycl::queue& q, const I2WriteAbl& p);
// Up to 4 i2_gate_combine calls of the same length in one launch.
struct I2GateSeg { const float* h; const float* m; const float* f; const float* g; float* out; };
sycl::event i2_gate_combine_multi(sycl::queue& q, uint32_t n, const I2GateSeg* seg, uint64_t len);
// The gate in two launches (fp32 fast path, up to 4 gates, K = 2H of 8-multiples, gate rank R <= 128):
// i2_gate_down_norm: gd[t] = W_down . [h[t] / rms(h[t]), m[t] / rms(m[t])] (the norms formed in the kernel: no cat
// buffer, no norm launch; a different fp32 order for the rms); i2_gate_up_combine: per element n,
// f = W_forget[n] . gd[t], g = W_gain[n] . gd[t] (each formed as i2_linear_bf16's one-row kernel forms it), and
// out = (1 - tanh f) h + tanh(g) m (as i2_gate_combine).
struct I2GateDownSeg { const float* h; const float* m; const uint16_t* W; float* gd; };
sycl::event i2_gate_down_norm(sycl::queue& q, uint32_t n, const I2GateDownSeg* seg, uint32_t T, uint32_t H, uint32_t R, float eps);
struct I2GateUpSeg { const float* gd; const uint16_t* Wf; const uint16_t* Wg; const float* h; const float* m; float* out; };
sycl::event i2_gate_up_combine(sycl::queue& q, uint32_t n, const I2GateUpSeg* seg, uint32_t T, uint32_t H, uint32_t R);

// x = bf16_round(x) in place.
sycl::event i2_round(sycl::queue& q, float* x, uint64_t n);
// x = cap * tanh(x / cap) in place (SPEC 1.7 / 8).
sycl::event i2_softcap(sycl::queue& q, float* x, uint64_t n, float cap);

// SPEC 3 steps 5 (bias), 7, 8, fused; no T x context buffer. Query t sits at absolute position P = pos0 + t and
// reads the cached keys / values of positions j = 0 .. P (window > 0: only P - j < window). Key position j lives in
// row j (ring == 0: a linear entry) or row j % ring (a ring entry; the caller keeps ring >= window - 1 + T so no row a
// query of this call needs was overwritten by this call's own writes). Kc / Vc rows are [nkv, hd].
//   bias(t, h, j) = 0 <= P - j < extent ? sum_x rel[t][h][x] * proj[x][P - j] : 0     (rel [T, nh, d_rel], proj [d_rel, extent])
//   s = ((q . k) * hd^-0.5) + bias;  p = softmax(s ++ sink) without the sink column;  o = p @ v
// Three passes over the keys (max, denominator, weighted sum) so p can be rounded exactly where the reference rounds it.
// Query head h reads KV head h / (nh / nkv). sink [T, nh], o [T, nh * hd]. hd <= 256.
sycl::event i2_attention(sycl::queue& q, const float* Q, const float* Kc, const float* Vc, uint32_t ring,
                         const float* rel, const float* proj, uint32_t d_rel, uint32_t extent,
                         const float* sink, float* o, uint32_t T, uint32_t pos0, uint32_t nh, uint32_t nkv,
                         uint32_t hd, uint32_t window, bool round);

// The same attention (fp32 only; no rounding), split over the keys: for each (query t, KV head, split of the visible
// keys) one work-group scores its keys for the nh / nkv query heads that share the KV head (each K / V row is read once
// for all of them) with an online softmax, and writes a partial (max, sum, weighted values) to `part`; a second kernel
// merges the partials and the sink. Same values as i2_attention up to fp32 summation order. `part` holds
// T * nh * n_split * (hd + 2) floats; i2_attention_splits() gives n_split for a part buffer of `part_floats` (at least
// min_keys keys per split).
uint32_t i2_attention_splits(uint32_t T, uint32_t pos0, uint32_t nh, uint32_t hd, uint32_t window, uint64_t part_floats,
                             uint32_t min_keys = 64);
sycl::event i2_attention_split(sycl::queue& q, const float* Q, const float* Kc, const float* Vc, uint32_t ring,
                               const float* rel, const float* proj, uint32_t d_rel, uint32_t extent,
                               const float* sink, float* o, uint32_t T, uint32_t pos0, uint32_t nh, uint32_t nkv,
                               uint32_t hd, uint32_t window, float* part, uint64_t part_floats, uint32_t min_keys = 64,
                               const float* gate = nullptr,    // gate: o *= sigmoid(gate) (SPEC 3 step 9) in the merge
                               I2Rows rows = {}, uint64_t kv_stride = 0, uint32_t ns_max = 0,
                               bool kv16 = false);   // kv16: Kc / Vc hold bf16 (uint16) values; kv_stride in elements
// rows (batched): row t has the splits i2_attention_splits(1, pos[t], ...) gives (part_floats = the T = 1 budget), its
// partials at part + t * nh * ns_max * (hd + 2); ns_max >= every row's split count (the launch's split range).

// Store T rows of `width` floats for positions pos0 .. pos0 + T - 1 into a cache entry: row = pos (ring == 0) or
// pos % ring.
sycl::event i2_store_rows(sycl::queue& q, const float* src, uint32_t T, uint32_t width, float* dst, uint32_t pos0,
                          uint32_t ring);

// SPEC 7.1 router, per token: scores = sigmoid(logits); own = top-k of (scores + bias), lower expert id first on
// exact ties, then sorted ascending; margin = k-th minus (k+1)-th selection score; used = forced (sorted
// ascending) if given, else own; w = scores[used] / (sum + 1e-20) * route_scale (the bias is not in w).
// logits [T, E]; forced / used / own [T, k] int32; w [T, k]; margin [T]. E <= 256, k <= 16.
sycl::event i2_router(sycl::queue& q, const float* logits, const float* bias, uint32_t T, uint32_t E, uint32_t k,
                      float route_scale, const int32_t* forced, int32_t* used, float* w, int32_t* own, float* margin);

// One expert bank (one tensor, all experts) in its stored form, decoded inside i2_expert_matvec (SPEC 10).
struct I2ExpertBank {
    int            fmt   = 0;        // 0 = NVFP4 (codes U8 two per byte + E4M3 block-16 scales + per-expert factor)
                                     // 1 = MXFP8 (E4M3 values + UE8M0 block-32 exponents)
    const uint8_t* w     = nullptr;  // NVFP4: [E, rows, cols / 2]; MXFP8: [E, rows, cols]
    const uint8_t* s     = nullptr;  // NVFP4: [E, rows, cols / 16]; MXFP8: [E, rows, cols / 32]
    const float*   fac   = nullptr;  // NVFP4: [E] = amax[e] / outer_scale_denominator
    uint32_t       rows  = 0, cols = 0;
};
// y[t][j][r] = sum_c W[idx[t][j]][r][c] * x(t, j)[c] for the k picked experts of each token. x(t, j) = x + t * cols
// (x_per_slot false: the token's MoE input) or x + (t * k + j) * cols (true: that slot's activation). y [T, k, rows].
// round_w rounds each decoded weight to bf16 (the reference's dequantised weights), round rounds y.
// Prefill MoE (fp32 path): i2_moe_group sorts the T * k slots by expert (order: ascending slot ids per expert) and cuts
// each expert's run into tiles of <= `tile` slots ({expert, first index into order, count} int32 triples; ntiles[0] =
// their number, at most ceil(T * k / tile) + E); i2_expert_grouped then computes every slot's rows like
// i2_expert_matvec's fast path (bit-identical values) with the weight loads shared across a tile. E <= 256.
constexpr uint32_t kI2MoeTile = 16;
sycl::event i2_moe_group(sycl::queue& q, const int32_t* used, uint32_t T, uint32_t k, uint32_t E, uint32_t tile,
                         int32_t* order, int32_t* tiles, int32_t* ntiles);
sycl::event i2_expert_grouped(sycl::queue& q, const I2ExpertBank& b, const float* x, bool x_per_slot, uint32_t k,
                              const int32_t* order, const int32_t* tiles, const int32_t* ntiles, uint32_t maxtiles,
                              float* y, bool round);
// The same products on the matrix engines (NVFP4 banks with rows % 128 == 0, cols % 32 == 0: i2_expert_xmx_ok): tiles
// of <= `tile` (16, 32, 64) slots from i2_moe_group(..., tile, ...); inputs rounded to fp16 (split: fp16 + the fp16
// remainder, two product passes), weights exact in fp16, fp32 sums in another order (not bit-identical).
// Sampling on the card (2026-11-23): one draw per row from logits [n][V] (modified in place by the penalties), the
// distribution of Ds41Generator::sample_row: presence / frequency deltas (pf pairs) then the repetition penalty over the
// recent ids (once per occurrence), then temperature <= 0: argmax (the lowest index among equal maxima); else the
// candidates are cut to top_k (the k-th largest logit, ties kept), the top_p nucleus (the largest logit whose and every
// larger logit's mass reaches top_p, ties kept) and the min_p floor, and the draw is a Gumbel-max over them with the
// row's seed (logits more than 25 nats under the maximum are left out: under 3e-6 of the mass). out[r] = -1: a
// non-finite logit. cand: [n][V] int32 scratch.
struct I2SampleRow {
    float    temp = 0.f, top_p = 1.f, min_p = 0.f, rep = 1.f;
    uint32_t top_k = 0, rec_off = 0, rec_n = 0, pf_off = 0, pf_n = 0, pad = 0;
    uint64_t seed = 0;
};
sycl::event i2_sample_rows(sycl::queue& q, float* logits, uint32_t n, uint32_t V, const I2SampleRow* rows, const int32_t* recent,
                           const int32_t* pf_id, const float* pf_d, int32_t* cand, int32_t* out);
bool i2_expert_xmx_ok(const I2ExpertBank& b);
sycl::event i2_expert_xmx(sycl::queue& q, const I2ExpertBank& b, const float* x, bool x_per_slot, uint32_t k,
                          const int32_t* order, const int32_t* tiles, const int32_t* ntiles, uint32_t maxtiles,
                          uint32_t tile, bool split, float* y);
// The w13 bank (rows 2j / 2j + 1 = gate / up of unit j) straight to the capped SwiGLU act [slots, rows / 2] (fp32
// path; the sums as i2_expert_matvec / i2_expert_grouped's fast path, the activation as i2_expert_act). b.cols % 32 == 0.
sycl::event i2_expert_w13_act(sycl::queue& q, const I2ExpertBank& b, const float* x, const int32_t* idx, uint32_t T,
                              uint32_t k, float* act, float gcap, float ucap);
sycl::event i2_expert_grouped_w13_act(sycl::queue& q, const I2ExpertBank& b, const float* x, uint32_t k,
                                      const int32_t* order, const int32_t* tiles, const int32_t* ntiles, uint32_t maxtiles,
                                      float* act, float gcap, float ucap);
// fast: the sub-group kernel (fp32 path only: 16-byte weight loads, block scale / expert factor applied per block / row,
// a different fp32 summation order); false or round_w: the generic kernel.
sycl::event i2_expert_matvec(sycl::queue& q, const I2ExpertBank& b, const float* x, bool x_per_slot,
                             const int32_t* idx, uint32_t T, uint32_t k, float* y, bool round_w, bool round,
                             bool fast = true);

// SPEC 7.2 activation: gu [R, 2I] rows interleaved (even = gate, odd = up) -> act [R, I] =
// gcap tanh(g / gcap) * sigmoid(g) * ucap tanh(u / ucap).
sycl::event i2_expert_act(sycl::queue& q, const float* gu, uint32_t R, uint32_t I, float gcap, float ucap, float* act,
                          bool round);
// out[t] = sum_j w[t][j] * y[t][j]   (fp32, slots in order); y [T, k, H].
sycl::event i2_expert_combine(sycl::queue& q, const float* y, const float* w, uint32_t T, uint32_t k, uint32_t H,
                              float* out, bool round);

}  // namespace ie
