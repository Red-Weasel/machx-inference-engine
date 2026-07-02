// src/ops/attention.cpp — full-attention forward (Qwen3.6 hybrid arch).
//
// Two-stage submission:
//   1) Append (k_in, v_in) into the cache at positions [start_pos, start_pos+T)
//   2) Online-softmax SDPA over the cache, GQA, causal-masked
//
// Kernel-level layout:
//   * One subgroup of SG_SIZE=16 lanes per (output_token t, q-head h).
//   * Lanes split head_dim across 16 chunks of head_dim/16 = 16 elems each.
//   * No SLM materialization of scores — online softmax (FA-1 style) keeps
//     running stats (m, l) and an `out_local` accumulator per lane.
//
// This naive variant scales to arbitrary ctx_len (no SLM cap), is correctness-
// first, and serves as the validation reference for the FA-2-tiled variant
// scheduled for Phase 9.

#include "ie/kernel_profiler.hpp"
#include "ie/ops.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

// IE_FA_NAIVE_PRECISE_EXP — replace `sycl::native::exp` with `sycl::exp` inside
// `full_attention`'s online-softmax loop.  Default OFF.
//
// Motivation (2026-05-02 bisect, Step 6):
// Steps 3-5 ruled out SG-reduce ordering, chunk-size-dependent algorithmic
// math, and cache write/read race.  Step 4 (A1 vs A2) confirmed the kernel
// is intrinsically non-deterministic.  Strongest remaining suspect:
// `sycl::native::exp` is "implementation-defined precision" per SYCL spec —
// if Xe2's hardware approximation has thermal/cache-state-dependent rounding
// that varies between invocations, the per-position online softmax updates
// would diverge run-to-run, exactly matching observed pattern.
//
// Hypothesis test: replace 2 native::exp calls inside `full_attention`'s
// inner softmax update with the precise IEEE `sycl::exp` variant.  If
// non-determinism vanishes (A1 == A2 bit-exact), native::exp is the source
// and ship the fix.  If still non-deterministic, pivot to memory-bounds
// audit on q_in / k_cache / v_cache indexing.
#ifndef IE_FA_NAIVE_PRECISE_EXP
#define IE_FA_NAIVE_PRECISE_EXP 0
#endif

// IE_FA_NAIVE_HARD_FENCE — insert an explicit `q.wait()` between the append
// kernel and the compute kernel inside `full_attention`.  Default OFF.
//
// Motivation (2026-05-02 bisect, Step 4):
// `ie-validate-chunking` Step 4 (A1 vs A2 comparison) confirmed
// `full_attention` is intrinsically non-deterministic — same code, same
// fixed inputs, fresh state, yet I7attno_ varies by ~1e-3 max_abs across
// runs and L41 varies by ~0.44.  Suspect #1 (after Step 3 ruled out
// reduce_over_group ordering): the compute kernel reads K/V cache
// positions that the append kernel just wrote on the same in-order queue.
// `compute_evt depends_on(append_evt)` should be a full memory fence per
// SYCL spec; if SYCL/L0/IGC enforces the fence incompletely, compute
// could read stale or partially-written cache data.
//
// Hypothesis test: replacing the implicit fence (depends_on) with an
// explicit host-side wait should eliminate any ordering bug.  If
// non-determinism vanishes, the fence enforcement is the issue.
#ifndef IE_FA_NAIVE_HARD_FENCE
#define IE_FA_NAIVE_HARD_FENCE 0
#endif

// IE_FA_NAIVE_DETERMINISTIC_REDUCE — replace the SG-wide score reduction in
// `full_attention` (naive T>1 prefill path) with an explicit butterfly XOR
// reduction in fixed mask order (8, 4, 2, 1).  Default OFF.
//
// Motivation (2026-05-02 bisect, see docs/bisect_chunking_validator.txt):
// `ie-validate-chunking` localized a chunk-size-dependent + run-to-run
// non-deterministic divergence to the `full_attention` kernel.  Same Q/K/V
// inputs (verified bit-exact via I1qproj_..I6krope_ dumps) produce different
// I7attno_ outputs at max_abs varying 30x across runs of identical code
// (1.46e-3, 8.5e-3, 4.93e-2 in three runs).  Only suspect not yet ruled out
// is `sycl::reduce_over_group` emitting different reduction orders / IGC
// codegen across launches with different total WG counts.  Pinning the
// butterfly order isolates this.
#ifndef IE_FA_NAIVE_DETERMINISTIC_REDUCE
#define IE_FA_NAIVE_DETERMINISTIC_REDUCE 0
#endif

// FA-2 split-K decode super-chunking target.  Determines how many super-chunks
// we aim to produce per kv_head; CHUNKS_PER_WG = ceil(n_chunks / TARGET_SUPER).
//
// Regression history: 5be3385 introduced TARGET_SUPER=12, replacing the prior
// fixed CHUNKS_PER_WG=8.  At long ctx (≥16k) this produced exactly 24 WGs
// (1 per Xe-core on B70) — single deep wave with no pipeline hiding for
// per-chunk launch/barrier overhead.  Measured -33% at ctx=32k INT8 KV
// (24.3 → 16.4 tok/s) vs the prior fixed-8 baseline.  The 5be3385 commit-time
// bench only covered short prompts (51-219 tok), so the long-ctx regression
// went undiagnosed.
//
// 64 keeps the short-ctx fan-out the adaptive logic was added for
// (n_chunks ≤ 64 → CHUNKS_PER_WG=1, full 16-WG fan-out at ctx=512) AND
// recovers long-ctx pipelining: at ctx=32k (n_chunks=512) it picks
// CHUNKS_PER_WG=8, matching the pre-regression fixed value and giving 128
// WGs / 5.3 waves on 24 Xe-cores.  Bench: 32k INT8 17.4 → 29.1 tok/s (+67%),
// 16k 25.4 → 34.9 (+37%); short-ctx unchanged or marginally better.
#ifndef IE_FA2_TARGET_SUPER
#define IE_FA2_TARGET_SUPER 64
#endif

namespace ie {

namespace {

constexpr int SG_SIZE = 16;

// Deterministic SG-wide sum for SG_SIZE=16: butterfly XOR with fixed mask
// order 8, 4, 2, 1.  Every lane gets the full sum.  Used by full_attention
// when IE_FA_NAIVE_DETERMINISTIC_REDUCE is set (debug/validation path).
inline float fa_naive_butterfly_sum16(sycl::sub_group sg, float x) {
    x += sycl::permute_group_by_xor(sg, x, 8);
    x += sycl::permute_group_by_xor(sg, x, 4);
    x += sycl::permute_group_by_xor(sg, x, 2);
    x += sycl::permute_group_by_xor(sg, x, 1);
    return x;
}

sycl::event empty_event(sycl::queue& q, const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.single_task([]() {});
    });
}

}  // namespace

sycl::event quantize_kv_to_int8(sycl::queue& q,
                                const sycl::half* k_src,
                                const sycl::half* v_src,
                                int8_t* k_dst,
                                int8_t* v_dst,
                                sycl::half* k_scales,
                                sycl::half* v_scales,
                                uint32_t n_rows,
                                uint32_t head_dim,
                                const std::vector<sycl::event>& deps) {
    if (n_rows == 0 || head_dim == 0) return empty_event(q, deps);

    constexpr uint32_t WG = 256;
    return ie::ps(q, "quant_kv_int8", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<2>({n_rows, WG}, {1, WG}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t row = uint32_t(it.get_group(0));
            const uint32_t lid = uint32_t(it.get_local_id(1));
            const uint64_t base = uint64_t(row) * head_dim;

            float k_local_max = 0.0f;
            float v_local_max = 0.0f;
            for (uint32_t d = lid; d < head_dim; d += WG) {
                k_local_max = sycl::fmax(k_local_max, sycl::fabs(float(k_src[base + d])));
                v_local_max = sycl::fmax(v_local_max, sycl::fabs(float(v_src[base + d])));
            }

            const float k_max = sycl::reduce_over_group(
                it.get_group(), k_local_max, sycl::maximum<float>{});
            const float v_max = sycl::reduce_over_group(
                it.get_group(), v_local_max, sycl::maximum<float>{});
            const float k_scale = (k_max > 0.0f) ? (k_max / 127.0f) : 1.0f;
            const float v_scale = (v_max > 0.0f) ? (v_max / 127.0f) : 1.0f;
            const float k_inv = (k_max > 0.0f) ? (127.0f / k_max) : 0.0f;
            const float v_inv = (v_max > 0.0f) ? (127.0f / v_max) : 0.0f;

            if (lid == 0) {
                k_scales[row] = sycl::half(k_scale);
                v_scales[row] = sycl::half(v_scale);
            }

            for (uint32_t d = lid; d < head_dim; d += WG) {
                const int k_q = int(sycl::round(float(k_src[base + d]) * k_inv));
                const int v_q = int(sycl::round(float(v_src[base + d]) * v_inv));
                k_dst[base + d] = int8_t(k_q < -127 ? -127 : (k_q > 127 ? 127 : k_q));
                v_dst[base + d] = int8_t(v_q < -127 ? -127 : (v_q > 127 ? 127 : v_q));
            }
        });
    });
}

sycl::event dequantize_kv_from_int8(sycl::queue& q,
                                    const int8_t* k_src,
                                    const int8_t* v_src,
                                    const sycl::half* k_scales,
                                    const sycl::half* v_scales,
                                    sycl::half* k_dst,
                                    sycl::half* v_dst,
                                    uint32_t n_rows,
                                    uint32_t head_dim,
                                    const std::vector<sycl::event>& deps) {
    if (n_rows == 0 || head_dim == 0) return empty_event(q, deps);

    const uint64_t total = uint64_t(n_rows) * head_dim;
    constexpr uint32_t WG = 256;
    const uint64_t global = ((total + WG - 1) / WG) * WG;
    return ie::ps(q, "dequant_kv_int8", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(global, WG),
                       [=](sycl::nd_item<1> it) {
            const uint64_t idx = it.get_global_id(0);
            if (idx >= total) return;
            const uint32_t row = uint32_t(idx / head_dim);
            const float ks = float(k_scales[row]);
            const float vs = float(v_scales[row]);
            k_dst[idx] = sycl::half(float(k_src[idx]) * ks);
            v_dst[idx] = sycl::half(float(v_src[idx]) * vs);
        });
    });
}

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
                           const std::vector<sycl::event>& deps) {
    // 1. Append (k_in, v_in) to the cache at [start_pos, start_pos+T).
    auto append_evt = ie::ps(q, "attn_naive_append", [&](sycl::handler& h) {
        h.depends_on(deps);
        const uint32_t total = T * n_kv_heads * head_dim;
        constexpr uint32_t WG = 256;
        const uint32_t global = ((total + WG - 1) / WG) * WG;
        h.parallel_for(sycl::nd_range<1>(global, WG),
                       [=](sycl::nd_item<1> it) {
            const uint32_t idx = uint32_t(it.get_global_id(0));
            if (idx >= total) return;
            const uint32_t d  = idx % head_dim;
            const uint32_t kv = (idx / head_dim) % n_kv_heads;
            const uint32_t t  = idx / (n_kv_heads * head_dim);
            const uint64_t in_off  = (uint64_t(t) * n_kv_heads + kv) * head_dim + d;
            const uint64_t out_off = (uint64_t(kv) * max_ctx + (start_pos + t)) * head_dim + d;
            k_cache[out_off] = k_in[in_off];
            v_cache[out_off] = v_in[in_off];
        });
    });

#if IE_FA_NAIVE_HARD_FENCE
    // Step 4 cache-race hypothesis test: explicit host-side wait on append
    // before compute submits.  Replaces the implicit fence from
    // depends_on(append_evt) with a hard host-blocking wait.  If the
    // implicit fence was the source of non-determinism, A1 vs A2 should
    // become bit-exact with this enabled.
    append_evt.wait();
#endif

    // 2. Attention compute. One subgroup per (t, h).
    return ie::ps(q, "attn_naive_compute", [&](sycl::handler& h) {
        h.depends_on(append_evt);
        const uint32_t dpl = head_dim / SG_SIZE;       // dims per lane
        const float    scale = 1.0f / sycl::sqrt(float(head_dim));
        const uint32_t gqa_ratio = n_q_heads / n_kv_heads;

        h.parallel_for(sycl::nd_range<2>({T, uint64_t(n_q_heads) * SG_SIZE}, {1, SG_SIZE}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t t       = uint32_t(it.get_group(0));
            const uint32_t h       = uint32_t(it.get_group(1));
            const uint32_t lane    = uint32_t(it.get_local_id(1));
            const uint32_t kv_head = h / gqa_ratio;
            const uint32_t pos_t   = start_pos + t;
            const uint32_t ctx_len = pos_t + 1;
            auto sg = it.get_sub_group();

            // Cache the lane's 16 Q dims in registers (head_dim is dpl × SG_SIZE).
            float q_vals[16];
            #pragma unroll
            for (uint32_t d_loc = 0; d_loc < dpl; ++d_loc) {
                const uint64_t q_off = (uint64_t(t) * n_q_heads + h) * head_dim
                                       + lane * dpl + d_loc;
                q_vals[d_loc] = float(q_in[q_off]);
            }

            // Online softmax accumulators.
            float m = -std::numeric_limits<float>::infinity();
            float l = 0.f;
            float out_local[16];
            #pragma unroll
            for (uint32_t d_loc = 0; d_loc < dpl; ++d_loc) out_local[d_loc] = 0.f;

            for (uint32_t i = 0; i < ctx_len; ++i) {
                // Score: dot(q, K[i, kv_head, :]) / sqrt(head_dim)
                float partial = 0.f;
                const uint64_t kv_row =
                    (uint64_t(kv_head) * max_ctx + i) * head_dim + lane * dpl;
                #pragma unroll
                for (uint32_t d_loc = 0; d_loc < dpl; ++d_loc) {
                    partial += q_vals[d_loc] * float(k_cache[kv_row + d_loc]);
                }
#if IE_FA_NAIVE_DETERMINISTIC_REDUCE
                const float s_i = fa_naive_butterfly_sum16(sg, partial) * scale;
#else
                const float s_i =
                    sycl::reduce_over_group(sg, partial, sycl::plus<float>()) * scale;
#endif

                const float m_new = sycl::fmax(m, s_i);
#if IE_FA_NAIVE_PRECISE_EXP
                const float alpha = sycl::exp(m - m_new);
                const float e     = sycl::exp(s_i - m_new);
#else
                const float alpha = sycl::native::exp(m - m_new);
                const float e     = sycl::native::exp(s_i - m_new);
#endif

                // Update output and l.
                #pragma unroll
                for (uint32_t d_loc = 0; d_loc < dpl; ++d_loc) {
                    const uint64_t v_off =
                        (uint64_t(kv_head) * max_ctx + i) * head_dim + lane * dpl + d_loc;
                    out_local[d_loc] = out_local[d_loc] * alpha
                                     + e * float(v_cache[v_off]);
                }
                l = l * alpha + e;
                m = m_new;
            }

            // Normalize and write output.
            const float inv_l = 1.f / l;
            #pragma unroll
            for (uint32_t d_loc = 0; d_loc < dpl; ++d_loc) {
                const uint64_t y_off = (uint64_t(t) * n_q_heads + h) * head_dim
                                       + lane * dpl + d_loc;
                y[y_off] = sycl::half(out_local[d_loc] * inv_l);
            }
        });
    });
}

// full_attention_gptoss — COPY of full_attention (crown's stays byte-identical)
// + the two gpt-oss attention pieces: per-head softmax SINK and sliding WINDOW.
// Used for both prefill (T>1) and decode (T==1). See ops.hpp for semantics.
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
                                  const std::vector<sycl::event>& deps) {
    // 1. Append (k_in, v_in) to the cache at [start_pos, start_pos+T).
    auto append_evt = ie::ps(q, "attn_gptoss_append", [&](sycl::handler& h) {
        h.depends_on(deps);
        const uint32_t total = T * n_kv_heads * head_dim;
        constexpr uint32_t WG = 256;
        const uint32_t global = ((total + WG - 1) / WG) * WG;
        h.parallel_for(sycl::nd_range<1>(global, WG),
                       [=](sycl::nd_item<1> it) {
            const uint32_t idx = uint32_t(it.get_global_id(0));
            if (idx >= total) return;
            const uint32_t d  = idx % head_dim;
            const uint32_t kv = (idx / head_dim) % n_kv_heads;
            const uint32_t t  = idx / (n_kv_heads * head_dim);
            const uint64_t in_off  = (uint64_t(t) * n_kv_heads + kv) * head_dim + d;
            const uint64_t out_off = (uint64_t(kv) * max_ctx + (start_pos + t)) * head_dim + d;
            k_cache[out_off] = k_in[in_off];
            v_cache[out_off] = v_in[in_off];
        });
    });

    // 2. Attention compute. One subgroup per (t, h).
    return ie::ps(q, "attn_gptoss_compute", [&](sycl::handler& h) {
        h.depends_on(append_evt);
        const uint32_t dpl = head_dim / SG_SIZE;       // dims per lane
        const float    scale = 1.0f / sycl::sqrt(float(head_dim));
        const uint32_t gqa_ratio = n_q_heads / n_kv_heads;

        h.parallel_for(sycl::nd_range<2>({T, uint64_t(n_q_heads) * SG_SIZE}, {1, SG_SIZE}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t t       = uint32_t(it.get_group(0));
            const uint32_t h       = uint32_t(it.get_group(1));
            const uint32_t lane    = uint32_t(it.get_local_id(1));
            const uint32_t kv_head = h / gqa_ratio;
            const uint32_t pos_t   = start_pos + t;
            const uint32_t ctx_len = pos_t + 1;
            // sliding-window lower bound: query pos_t attends keys (pos_t-window, pos_t].
            // window==0 → wstart=0 → full causal (identical to full_attention).
            const uint32_t wstart  = (window && pos_t + 1 > window) ? (pos_t + 1 - window) : 0u;
            auto sg = it.get_sub_group();

            float q_vals[16];
            #pragma unroll
            for (uint32_t d_loc = 0; d_loc < dpl; ++d_loc) {
                const uint64_t q_off = (uint64_t(t) * n_q_heads + h) * head_dim
                                       + lane * dpl + d_loc;
                q_vals[d_loc] = float(q_in[q_off]);
            }

            float m = -std::numeric_limits<float>::infinity();
            float l = 0.f;
            float out_local[16];
            #pragma unroll
            for (uint32_t d_loc = 0; d_loc < dpl; ++d_loc) out_local[d_loc] = 0.f;

            for (uint32_t i = wstart; i < ctx_len; ++i) {
                float partial = 0.f;
                const uint64_t kv_row =
                    (uint64_t(kv_head) * max_ctx + i) * head_dim + lane * dpl;
                #pragma unroll
                for (uint32_t d_loc = 0; d_loc < dpl; ++d_loc) {
                    partial += q_vals[d_loc] * float(k_cache[kv_row + d_loc]);
                }
                const float s_i =
                    sycl::reduce_over_group(sg, partial, sycl::plus<float>()) * scale;

                const float m_new = sycl::fmax(m, s_i);
                const float alpha = sycl::native::exp(m - m_new);
                const float e     = sycl::native::exp(s_i - m_new);

                #pragma unroll
                for (uint32_t d_loc = 0; d_loc < dpl; ++d_loc) {
                    const uint64_t v_off =
                        (uint64_t(kv_head) * max_ctx + i) * head_dim + lane * dpl + d_loc;
                    out_local[d_loc] = out_local[d_loc] * alpha
                                     + e * float(v_cache[v_off]);
                }
                l = l * alpha + e;
                m = m_new;
            }

            // Fold the per-head sink into the DENOMINATOR only (virtual key, no value):
            // m_f = max(m, s); rescale by corr; l' = l·corr + exp(s - m_f). Mass leaks out.
            float corr = 1.f, ll = l;
            if (sinks) {
                const float s = sinks[h];
                const float m_f = sycl::fmax(m, s);
                corr = (m == -std::numeric_limits<float>::infinity())
                         ? 0.f : sycl::native::exp(m - m_f);
                ll   = l * corr + sycl::native::exp(s - m_f);
            }
            const float inv_l = (ll > 0.f) ? (1.f / ll) : 0.f;
            #pragma unroll
            for (uint32_t d_loc = 0; d_loc < dpl; ++d_loc) {
                const uint64_t y_off = (uint64_t(t) * n_q_heads + h) * head_dim
                                       + lane * dpl + d_loc;
                y[y_off] = sycl::half(out_local[d_loc] * corr * inv_l);
            }
        });
    });
}

// full_attention_gemma — byte-for-byte copy of full_attention with the per-lane
// register arrays sized [32] instead of [16], so head_dim up to 512 (= dpl 32
// at SG_SIZE=16) works. Gemma 4's GLOBAL layers use head_dim 512; the shared
// full_attention's [16] arrays overflow there (GPU page fault). COPY-not-edit
// per the crown-untouchable rule — the crown's full_attention is unchanged.
sycl::event full_attention_gemma(sycl::queue& q,
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
                                 const std::vector<sycl::event>& deps) {
    auto append_evt = ie::ps(q, "attn_gemma_append", [&](sycl::handler& h) {
        h.depends_on(deps);
        const uint32_t total = T * n_kv_heads * head_dim;
        constexpr uint32_t WG = 256;
        const uint32_t global = ((total + WG - 1) / WG) * WG;
        h.parallel_for(sycl::nd_range<1>(global, WG),
                       [=](sycl::nd_item<1> it) {
            const uint32_t idx = uint32_t(it.get_global_id(0));
            if (idx >= total) return;
            const uint32_t d  = idx % head_dim;
            const uint32_t kv = (idx / head_dim) % n_kv_heads;
            const uint32_t t  = idx / (n_kv_heads * head_dim);
            const uint64_t in_off  = (uint64_t(t) * n_kv_heads + kv) * head_dim + d;
            const uint64_t out_off = (uint64_t(kv) * max_ctx + (start_pos + t)) * head_dim + d;
            k_cache[out_off] = k_in[in_off];
            v_cache[out_off] = v_in[in_off];
        });
    });

    return ie::ps(q, "attn_gemma_compute", [&](sycl::handler& h) {
        h.depends_on(append_evt);
        const uint32_t dpl = head_dim / SG_SIZE;
        const float    scale = 1.0f / sycl::sqrt(float(head_dim));
        const uint32_t gqa_ratio = n_q_heads / n_kv_heads;

        h.parallel_for(sycl::nd_range<2>({T, uint64_t(n_q_heads) * SG_SIZE}, {1, SG_SIZE}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t t       = uint32_t(it.get_group(0));
            const uint32_t h       = uint32_t(it.get_group(1));
            const uint32_t lane    = uint32_t(it.get_local_id(1));
            const uint32_t kv_head = h / gqa_ratio;
            const uint32_t pos_t   = start_pos + t;
            const uint32_t ctx_len = pos_t + 1;
            auto sg = it.get_sub_group();

            float q_vals[32];               // [16]→[32] for head_dim up to 512
            #pragma unroll
            for (uint32_t d_loc = 0; d_loc < dpl; ++d_loc) {
                const uint64_t q_off = (uint64_t(t) * n_q_heads + h) * head_dim
                                       + lane * dpl + d_loc;
                q_vals[d_loc] = float(q_in[q_off]);
            }

            float m = -std::numeric_limits<float>::infinity();
            float l = 0.f;
            float out_local[32];            // [16]→[32]
            #pragma unroll
            for (uint32_t d_loc = 0; d_loc < dpl; ++d_loc) out_local[d_loc] = 0.f;

            for (uint32_t i = 0; i < ctx_len; ++i) {
                float partial = 0.f;
                const uint64_t kv_row =
                    (uint64_t(kv_head) * max_ctx + i) * head_dim + lane * dpl;
                #pragma unroll
                for (uint32_t d_loc = 0; d_loc < dpl; ++d_loc)
                    partial += q_vals[d_loc] * float(k_cache[kv_row + d_loc]);
                const float s_i =
                    sycl::reduce_over_group(sg, partial, sycl::plus<float>()) * scale;
                const float m_new = sycl::fmax(m, s_i);
                const float alpha = sycl::native::exp(m - m_new);
                const float e     = sycl::native::exp(s_i - m_new);
                #pragma unroll
                for (uint32_t d_loc = 0; d_loc < dpl; ++d_loc) {
                    const uint64_t v_off =
                        (uint64_t(kv_head) * max_ctx + i) * head_dim + lane * dpl + d_loc;
                    out_local[d_loc] = out_local[d_loc] * alpha + e * float(v_cache[v_off]);
                }
                l = l * alpha + e;
                m = m_new;
            }

            const float inv_l = 1.f / l;
            #pragma unroll
            for (uint32_t d_loc = 0; d_loc < dpl; ++d_loc) {
                const uint64_t y_off = (uint64_t(t) * n_q_heads + h) * head_dim
                                       + lane * dpl + d_loc;
                y[y_off] = sycl::half(out_local[d_loc] * inv_l);
            }
        });
    });
}

// read_attention_gemma — READ-ONLY attention over an already-populated KV cache,
// no append. The gemma4-assistant MTP draft head has Q-only attention layers that
// SHARE (read) the TARGET model's KV cache (it writes no K/V of its own). A single
// query token [n_q_heads, head_dim] attends cache positions [0, ctx_len). Scale
// 1/sqrt(head_dim) (the caller pre-scales Q by sqrt(head_dim) → net 1.0, matching
// full_attention_gemma / Gemma's f_attention_scale=1.0). Same [32]-sized per-lane
// registers as full_attention_gemma → head_dim up to 512. T=1 (single draft token).
sycl::event read_attention_gemma(sycl::queue& q,
                                 const sycl::half* q_in,
                                 const sycl::half* k_cache,
                                 const sycl::half* v_cache,
                                 sycl::half* y,
                                 uint32_t ctx_len,
                                 uint32_t n_q_heads,
                                 uint32_t n_kv_heads,
                                 uint32_t head_dim,
                                 uint32_t max_ctx,
                                 const std::vector<sycl::event>& deps) {
    return ie::ps(q, "attn_gemma_read", [&](sycl::handler& h) {
        h.depends_on(deps);
        const uint32_t dpl = head_dim / SG_SIZE;
        const float    scale = 1.0f / sycl::sqrt(float(head_dim));
        const uint32_t gqa_ratio = n_q_heads / n_kv_heads;

        h.parallel_for(sycl::nd_range<2>({1, uint64_t(n_q_heads) * SG_SIZE}, {1, SG_SIZE}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t hh      = uint32_t(it.get_group(1));
            const uint32_t lane    = uint32_t(it.get_local_id(1));
            const uint32_t kv_head = hh / gqa_ratio;
            auto sg = it.get_sub_group();

            float q_vals[32];
            #pragma unroll
            for (uint32_t d_loc = 0; d_loc < dpl; ++d_loc) {
                const uint64_t q_off = uint64_t(hh) * head_dim + lane * dpl + d_loc;
                q_vals[d_loc] = float(q_in[q_off]);
            }

            float m = -std::numeric_limits<float>::infinity();
            float l = 0.f;
            float out_local[32];
            #pragma unroll
            for (uint32_t d_loc = 0; d_loc < dpl; ++d_loc) out_local[d_loc] = 0.f;

            for (uint32_t i = 0; i < ctx_len; ++i) {
                float partial = 0.f;
                const uint64_t kv_row =
                    (uint64_t(kv_head) * max_ctx + i) * head_dim + lane * dpl;
                #pragma unroll
                for (uint32_t d_loc = 0; d_loc < dpl; ++d_loc)
                    partial += q_vals[d_loc] * float(k_cache[kv_row + d_loc]);
                const float s_i =
                    sycl::reduce_over_group(sg, partial, sycl::plus<float>()) * scale;
                const float m_new = sycl::fmax(m, s_i);
                const float alpha = sycl::native::exp(m - m_new);
                const float e     = sycl::native::exp(s_i - m_new);
                #pragma unroll
                for (uint32_t d_loc = 0; d_loc < dpl; ++d_loc) {
                    const uint64_t v_off =
                        (uint64_t(kv_head) * max_ctx + i) * head_dim + lane * dpl + d_loc;
                    out_local[d_loc] = out_local[d_loc] * alpha + e * float(v_cache[v_off]);
                }
                l = l * alpha + e;
                m = m_new;
            }

            const float inv_l = (l > 0.f) ? (1.f / l) : 0.f;
            #pragma unroll
            for (uint32_t d_loc = 0; d_loc < dpl; ++d_loc) {
                const uint64_t y_off = uint64_t(hh) * head_dim + lane * dpl + d_loc;
                y[y_off] = sycl::half(out_local[d_loc] * inv_l);
            }
        });
    });
}

// =====================================================================
// FlashAttention-2 split-K, decode-only (T=1)
// =====================================================================
//
// Why: the naive `full_attention` above runs WG-per-q_head, so each
// kv_head's K/V is read GQA-times redundantly (gqa=8 on Qwen3.6 → 8×).
// This is the dominant decode cost at long ctx — measured 5.4 ms/full-
// attn-layer at ctx=4k, ~85% of which is wasted KV bandwidth.
//
// FA-2 split-K layout:
//   pass 1 (partial): WG per (kv_head, ctx_chunk_of_Bc=64).  Each WG
//     cooperatively SLM-loads its (Bc × head_dim) K and V tiles ONCE,
//     then runs the gqa SGs that share this kv_head over the tile,
//     producing per-chunk online-softmax partials (m, l, out).  KV is
//     read exactly once per layer per token now.
//   pass 2 (combine): WG per q_head.  Merges per-chunk partials with
//     the standard online-softmax-merge formula:
//       m* = max_c m_c
//       l* = Σ_c l_c · exp(m_c − m*)
//       o* = Σ_c o_c · exp(m_c − m*) / l*
//
// Workspace: `partials_scratch` is FP32 [n_chunks_max, n_q_heads, head_dim+2].
// Caller sizes it for the largest ctx it will ever pass.
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
                                      const std::vector<sycl::event>& deps,
                                      AttnProfileData* prof) {
    constexpr uint32_t Bc = 64;
    // Adaptive super-chunking — see INT8 path for rationale.
    constexpr uint32_t TARGET_SUPER = IE_FA2_TARGET_SUPER;
    const uint32_t ctx_len = start_pos + 1;
    const uint32_t n_chunks = (ctx_len + Bc - 1) / Bc;
    const uint32_t CHUNKS_PER_WG =
        std::max<uint32_t>(1u, (n_chunks + TARGET_SUPER - 1) / TARGET_SUPER);
    const uint32_t n_super_chunks = (n_chunks + CHUNKS_PER_WG - 1) / CHUNKS_PER_WG;
    const uint32_t gqa = n_q_heads / n_kv_heads;

    // 1. Append (k_in, v_in) for THIS new token to the cache at start_pos.
    auto append_evt = ie::ps(q, "fa2_append_fp16", [&](sycl::handler& h) {
        h.depends_on(deps);
        const uint32_t total = n_kv_heads * head_dim;
        constexpr uint32_t WG = 256;
        const uint32_t global = ((total + WG - 1) / WG) * WG;
        h.parallel_for(sycl::nd_range<1>(global, WG),
                       [=](sycl::nd_item<1> it) {
            const uint32_t idx = uint32_t(it.get_global_id(0));
            if (idx >= total) return;
            const uint32_t d  = idx % head_dim;
            const uint32_t kv = idx / head_dim;
            const uint64_t in_off  = uint64_t(kv) * head_dim + d;
            const uint64_t out_off = (uint64_t(kv) * max_ctx + start_pos) * head_dim + d;
            k_cache[out_off] = k_in[in_off];
            v_cache[out_off] = v_in[in_off];
        });
    });

    // 2. Partial pass: WG per (kv_head, super_chunk_of_8). gqa SGs per WG.
    //    Each WG processes CHUNKS_PER_WG=8 Bc-tiles serially while keeping
    //    its own running (m, l, out) across them — eliminates 87.5% of WG
    //    launches at long ctx.
    auto partial_evt = ie::ps(q, "fa2_partial_fp16", [&](sycl::handler& h) {
        h.depends_on(append_evt);
        sycl::local_accessor<sycl::half, 1> K_slm({uint64_t(Bc) * head_dim}, h);
        sycl::local_accessor<sycl::half, 1> V_slm({uint64_t(Bc) * head_dim}, h);

        const uint32_t WG_ITEMS = gqa * SG_SIZE;          // 8 × 16 = 128
        const uint32_t dpl = head_dim / SG_SIZE;          // 16
        const float scale = 1.0f / sycl::sqrt(float(head_dim));

        h.parallel_for(sycl::nd_range<2>({uint64_t(n_kv_heads) * n_super_chunks, WG_ITEMS},
                                          {1, WG_ITEMS}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t kv_super_id = uint32_t(it.get_group(0));
            const uint32_t kv          = kv_super_id / n_super_chunks;
            const uint32_t super       = kv_super_id % n_super_chunks;
            const uint32_t lid         = uint32_t(it.get_local_id(1));
            const uint32_t sg_id       = lid / SG_SIZE;
            const uint32_t lane        = lid % SG_SIZE;
            const uint32_t my_q        = kv * gqa + sg_id;
            auto sg = it.get_sub_group();

            // Per-SG q (registered in private memory); per-lane dpl dims.
            float q_vals[16];
            #pragma unroll
            for (uint32_t d_loc = 0; d_loc < dpl; ++d_loc) {
                q_vals[d_loc] = float(q_in[uint64_t(my_q) * head_dim + lane * dpl + d_loc]);
            }
            float m = -std::numeric_limits<float>::infinity();
            float l = 0.f;
            float out_local[16];
            #pragma unroll
            for (uint32_t d_loc = 0; d_loc < dpl; ++d_loc) out_local[d_loc] = 0.f;

            // Loop CHUNKS_PER_WG inner chunks, accumulating into local m, l, out.
            const uint32_t hd_log2 = 31 - sycl::clz(head_dim);
            const uint32_t hd_mask = head_dim - 1;
            const uint32_t chunk_first = super * CHUNKS_PER_WG;
            for (uint32_t cc = 0; cc < CHUNKS_PER_WG; ++cc) {
                const uint32_t chunk = chunk_first + cc;
                if (chunk >= n_chunks) break;

                const uint32_t chunk_start = chunk * Bc;
                const uint32_t chunk_end   = sycl::min(chunk_start + Bc, ctx_len);
                const uint32_t chunk_n     = chunk_end - chunk_start;

                const uint32_t tile_size = chunk_n * head_dim;
                for (uint32_t i = lid; i < tile_size; i += WG_ITEMS) {
                    const uint32_t tk = i >> hd_log2;
                    const uint32_t d  = i & hd_mask;
                    const uint64_t off =
                        (uint64_t(kv) * max_ctx + chunk_start + tk) * head_dim + d;
                    K_slm[i] = k_cache[off];
                    V_slm[i] = v_cache[off];
                }
                sycl::group_barrier(it.get_group());

                for (uint32_t i = 0; i < chunk_n; ++i) {
                    float partial = 0.f;
                    #pragma unroll
                    for (uint32_t d_loc = 0; d_loc < dpl; ++d_loc) {
                        partial += q_vals[d_loc] *
                                   float(K_slm[i * head_dim + lane * dpl + d_loc]);
                    }
                    const float s_i =
                        sycl::reduce_over_group(sg, partial, sycl::plus<float>()) * scale;
                    const float m_new = sycl::fmax(m, s_i);
                    const float alpha = sycl::native::exp(m - m_new);
                    const float e     = sycl::native::exp(s_i - m_new);
                    #pragma unroll
                    for (uint32_t d_loc = 0; d_loc < dpl; ++d_loc) {
                        out_local[d_loc] = out_local[d_loc] * alpha +
                                           e * float(V_slm[i * head_dim + lane * dpl + d_loc]);
                    }
                    l = l * alpha + e;
                    m = m_new;
                }
                sycl::group_barrier(it.get_group());  // before next K tile load
            }

            // Write one super-partial per (super, my_q).
            const uint64_t base =
                (uint64_t(super) * n_q_heads + my_q) * (uint64_t(head_dim) + 2);
            if (lane == 0) {
                partials_scratch[base + 0] = m;
                partials_scratch[base + 1] = l;
            }
            #pragma unroll
            for (uint32_t d_loc = 0; d_loc < dpl; ++d_loc) {
                partials_scratch[base + 2 + lane * dpl + d_loc] = out_local[d_loc];
            }
        });
    });

    // 3. Combine pass: now iterates over n_super_chunks (not n_chunks).
    const uint32_t n_partials = n_super_chunks;
    auto combine_evt = ie::ps(q, "fa2_combine_fp16", [&](sycl::handler& h) {
        h.depends_on(partial_evt);
        const uint32_t dpl = head_dim / SG_SIZE;
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_q_heads) * SG_SIZE, SG_SIZE),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t qh   = uint32_t(it.get_group(0));
            const uint32_t lane = uint32_t(it.get_local_id(0));

            float m_global = -std::numeric_limits<float>::infinity();
            for (uint32_t c = 0; c < n_partials; ++c) {
                const uint64_t base =
                    (uint64_t(c) * n_q_heads + qh) * (uint64_t(head_dim) + 2);
                m_global = sycl::fmax(m_global, partials_scratch[base + 0]);
            }
            float l_global = 0.f;
            float out_local[16];
            #pragma unroll
            for (uint32_t d_loc = 0; d_loc < dpl; ++d_loc) out_local[d_loc] = 0.f;

            for (uint32_t c = 0; c < n_partials; ++c) {
                const uint64_t base =
                    (uint64_t(c) * n_q_heads + qh) * (uint64_t(head_dim) + 2);
                const float m_c = partials_scratch[base + 0];
                const float l_c = partials_scratch[base + 1];
                const float w   = sycl::native::exp(m_c - m_global);
                l_global += l_c * w;
                #pragma unroll
                for (uint32_t d_loc = 0; d_loc < dpl; ++d_loc) {
                    out_local[d_loc] +=
                        w * partials_scratch[base + 2 + lane * dpl + d_loc];
                }
            }
            const float inv_l = 1.0f / l_global;
            #pragma unroll
            for (uint32_t d_loc = 0; d_loc < dpl; ++d_loc) {
                const uint64_t y_off = uint64_t(qh) * head_dim + lane * dpl + d_loc;
                y[y_off] = sycl::half(out_local[d_loc] * inv_l);
            }
        });
    });
    if (prof) {
        combine_evt.wait();
        auto dur = [](const sycl::event& e) -> uint64_t {
            return e.get_profiling_info<sycl::info::event_profiling::command_end>()
                 - e.get_profiling_info<sycl::info::event_profiling::command_start>();
        };
        prof->append_ns  += dur(append_evt);
        prof->partial_ns += dur(partial_evt);
        prof->combine_ns += dur(combine_evt);
    }
    return combine_evt;
}

// full_attention_fa2_decode_gptoss — COPY of full_attention_fa2_decode (the base
// kernel stays byte-identical) + the gpt-oss per-head softmax SINK.  A sink is a
// virtual always-attended key with NO value: it enters the softmax DENOMINATOR only
// (mass leaks out).  In the split-K layout the sink is layer/head-global, so it
// folds ONCE in the combine pass after the per-chunk partials are merged:
//   m' = max(m_global, sink_h);  l_final = l_global·exp(m_global−m') + exp(sink_h−m');
//   out_final = out_local·exp(m_global−m') / l_final.
// Used for gpt-oss DECODE on the FULL-attention (odd) layers — the dominant decode
// cost at long ctx (the windowed even layers stay on the bounded naive path).
// head_dim=64 (dpl=4).  sinks==nullptr → identical to full_attention_fa2_decode.
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
                                             const std::vector<sycl::event>& deps) {
    constexpr uint32_t Bc = 64;
    constexpr uint32_t TARGET_SUPER = IE_FA2_TARGET_SUPER;
    const uint32_t ctx_len = start_pos + 1;
    const uint32_t n_chunks = (ctx_len + Bc - 1) / Bc;
    const uint32_t CHUNKS_PER_WG =
        std::max<uint32_t>(1u, (n_chunks + TARGET_SUPER - 1) / TARGET_SUPER);
    const uint32_t n_super_chunks = (n_chunks + CHUNKS_PER_WG - 1) / CHUNKS_PER_WG;
    const uint32_t gqa = n_q_heads / n_kv_heads;

    // 1. Append (k_in, v_in) for THIS new token to the cache at start_pos.
    auto append_evt = ie::ps(q, "attn_gptoss_fa2_append", [&](sycl::handler& h) {
        h.depends_on(deps);
        const uint32_t total = n_kv_heads * head_dim;
        constexpr uint32_t WG = 256;
        const uint32_t global = ((total + WG - 1) / WG) * WG;
        h.parallel_for(sycl::nd_range<1>(global, WG),
                       [=](sycl::nd_item<1> it) {
            const uint32_t idx = uint32_t(it.get_global_id(0));
            if (idx >= total) return;
            const uint32_t d  = idx % head_dim;
            const uint32_t kv = idx / head_dim;
            const uint64_t in_off  = uint64_t(kv) * head_dim + d;
            const uint64_t out_off = (uint64_t(kv) * max_ctx + start_pos) * head_dim + d;
            k_cache[out_off] = k_in[in_off];
            v_cache[out_off] = v_in[in_off];
        });
    });

    // 2. Partial pass (identical to the base kernel).
    auto partial_evt = ie::ps(q, "attn_gptoss_fa2_partial", [&](sycl::handler& h) {
        h.depends_on(append_evt);
        sycl::local_accessor<sycl::half, 1> K_slm({uint64_t(Bc) * head_dim}, h);
        sycl::local_accessor<sycl::half, 1> V_slm({uint64_t(Bc) * head_dim}, h);
        const uint32_t WG_ITEMS = gqa * SG_SIZE;
        const uint32_t dpl = head_dim / SG_SIZE;
        const float scale = 1.0f / sycl::sqrt(float(head_dim));

        h.parallel_for(sycl::nd_range<2>({uint64_t(n_kv_heads) * n_super_chunks, WG_ITEMS},
                                          {1, WG_ITEMS}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t kv_super_id = uint32_t(it.get_group(0));
            const uint32_t kv          = kv_super_id / n_super_chunks;
            const uint32_t super       = kv_super_id % n_super_chunks;
            const uint32_t lid         = uint32_t(it.get_local_id(1));
            const uint32_t sg_id       = lid / SG_SIZE;
            const uint32_t lane        = lid % SG_SIZE;
            const uint32_t my_q        = kv * gqa + sg_id;
            auto sg = it.get_sub_group();

            float q_vals[16];
            #pragma unroll
            for (uint32_t d_loc = 0; d_loc < dpl; ++d_loc)
                q_vals[d_loc] = float(q_in[uint64_t(my_q) * head_dim + lane * dpl + d_loc]);
            float m = -std::numeric_limits<float>::infinity();
            float l = 0.f;
            float out_local[16];
            #pragma unroll
            for (uint32_t d_loc = 0; d_loc < dpl; ++d_loc) out_local[d_loc] = 0.f;

            const uint32_t hd_log2 = 31 - sycl::clz(head_dim);
            const uint32_t hd_mask = head_dim - 1;
            const uint32_t chunk_first = super * CHUNKS_PER_WG;
            for (uint32_t cc = 0; cc < CHUNKS_PER_WG; ++cc) {
                const uint32_t chunk = chunk_first + cc;
                if (chunk >= n_chunks) break;
                const uint32_t chunk_start = chunk * Bc;
                const uint32_t chunk_end   = sycl::min(chunk_start + Bc, ctx_len);
                const uint32_t chunk_n     = chunk_end - chunk_start;
                const uint32_t tile_size = chunk_n * head_dim;
                for (uint32_t i = lid; i < tile_size; i += WG_ITEMS) {
                    const uint32_t tk = i >> hd_log2;
                    const uint32_t d  = i & hd_mask;
                    const uint64_t off =
                        (uint64_t(kv) * max_ctx + chunk_start + tk) * head_dim + d;
                    K_slm[i] = k_cache[off];
                    V_slm[i] = v_cache[off];
                }
                sycl::group_barrier(it.get_group());
                for (uint32_t i = 0; i < chunk_n; ++i) {
                    float partial = 0.f;
                    #pragma unroll
                    for (uint32_t d_loc = 0; d_loc < dpl; ++d_loc)
                        partial += q_vals[d_loc] *
                                   float(K_slm[i * head_dim + lane * dpl + d_loc]);
                    const float s_i =
                        sycl::reduce_over_group(sg, partial, sycl::plus<float>()) * scale;
                    const float m_new = sycl::fmax(m, s_i);
                    const float alpha = sycl::native::exp(m - m_new);
                    const float e     = sycl::native::exp(s_i - m_new);
                    #pragma unroll
                    for (uint32_t d_loc = 0; d_loc < dpl; ++d_loc)
                        out_local[d_loc] = out_local[d_loc] * alpha +
                                           e * float(V_slm[i * head_dim + lane * dpl + d_loc]);
                    l = l * alpha + e;
                    m = m_new;
                }
                sycl::group_barrier(it.get_group());
            }
            const uint64_t base =
                (uint64_t(super) * n_q_heads + my_q) * (uint64_t(head_dim) + 2);
            if (lane == 0) {
                partials_scratch[base + 0] = m;
                partials_scratch[base + 1] = l;
            }
            #pragma unroll
            for (uint32_t d_loc = 0; d_loc < dpl; ++d_loc)
                partials_scratch[base + 2 + lane * dpl + d_loc] = out_local[d_loc];
        });
    });

    // 3. Combine pass + per-head SINK fold (the only delta vs the base kernel).
    const uint32_t n_partials = n_super_chunks;
    auto combine_evt = ie::ps(q, "attn_gptoss_fa2_combine", [&](sycl::handler& h) {
        h.depends_on(partial_evt);
        const uint32_t dpl = head_dim / SG_SIZE;
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_q_heads) * SG_SIZE, SG_SIZE),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t qh   = uint32_t(it.get_group(0));
            const uint32_t lane = uint32_t(it.get_local_id(0));

            float m_global = -std::numeric_limits<float>::infinity();
            for (uint32_t c = 0; c < n_partials; ++c) {
                const uint64_t base =
                    (uint64_t(c) * n_q_heads + qh) * (uint64_t(head_dim) + 2);
                m_global = sycl::fmax(m_global, partials_scratch[base + 0]);
            }
            float l_global = 0.f;
            float out_local[16];
            #pragma unroll
            for (uint32_t d_loc = 0; d_loc < dpl; ++d_loc) out_local[d_loc] = 0.f;
            for (uint32_t c = 0; c < n_partials; ++c) {
                const uint64_t base =
                    (uint64_t(c) * n_q_heads + qh) * (uint64_t(head_dim) + 2);
                const float m_c = partials_scratch[base + 0];
                const float l_c = partials_scratch[base + 1];
                const float w   = sycl::native::exp(m_c - m_global);
                l_global += l_c * w;
                #pragma unroll
                for (uint32_t d_loc = 0; d_loc < dpl; ++d_loc)
                    out_local[d_loc] += w * partials_scratch[base + 2 + lane * dpl + d_loc];
            }
            // Per-head SINK: virtual key, denominator only.  Numerically stable for
            // sink ≷ m_global.  sinks==nullptr → no sink (base behaviour).
            float scale_o = 1.0f;
            float l_final = l_global;
            if (sinks) {
                const float sink = sinks[qh];
                const float m_prime = sycl::fmax(m_global, sink);
                scale_o = sycl::native::exp(m_global - m_prime);
                l_final = l_global * scale_o + sycl::native::exp(sink - m_prime);
            }
            const float inv_l = scale_o / l_final;
            #pragma unroll
            for (uint32_t d_loc = 0; d_loc < dpl; ++d_loc) {
                const uint64_t y_off = uint64_t(qh) * head_dim + lane * dpl + d_loc;
                y[y_off] = sycl::half(out_local[d_loc] * inv_l);
            }
        });
    });
    return combine_evt;
}

// =====================================================================
// FA-2 split-K decode v2 — BLOCKED inner loop (latency-bound fix)
// =====================================================================
// The v1 partial pass does one subgroup reduce_over_group PER KV position on a
// serial online-softmax chain → at long ctx it's latency-bound (~11% peak BW,
// 67 GB/s @16K). v2 processes BLK positions per iteration: the BLK score dots +
// their BLK reduce_over_group calls are independent, so they pipeline and hide
// latency; only the softmax accumulation stays serial. SAME partials format and
// SAME numeric order → the combine pass is reused unchanged and PPL is identical.
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
                                         const std::vector<sycl::event>& deps,
                                         AttnProfileData* prof) {
    constexpr uint32_t Bc = 64;
    constexpr uint32_t BLK = 16;                      // softmax-defer tile (positions/rescale)
    constexpr uint32_t TARGET_SUPER = IE_FA2_TARGET_SUPER;
    const uint32_t ctx_len = start_pos + 1;
    const uint32_t n_chunks = (ctx_len + Bc - 1) / Bc;
    const uint32_t CHUNKS_PER_WG =
        std::max<uint32_t>(1u, (n_chunks + TARGET_SUPER - 1) / TARGET_SUPER);
    const uint32_t n_super_chunks = (n_chunks + CHUNKS_PER_WG - 1) / CHUNKS_PER_WG;
    const uint32_t gqa = n_q_heads / n_kv_heads;

    // 1. Append (identical to v1).
    auto append_evt = ie::ps(q, "fa2_append_fp16", [&](sycl::handler& h) {
        h.depends_on(deps);
        const uint32_t total = n_kv_heads * head_dim;
        constexpr uint32_t WG = 256;
        const uint32_t global = ((total + WG - 1) / WG) * WG;
        h.parallel_for(sycl::nd_range<1>(global, WG), [=](sycl::nd_item<1> it) {
            const uint32_t idx = uint32_t(it.get_global_id(0));
            if (idx >= total) return;
            const uint32_t d  = idx % head_dim;
            const uint32_t kv = idx / head_dim;
            const uint64_t in_off  = uint64_t(kv) * head_dim + d;
            const uint64_t out_off = (uint64_t(kv) * max_ctx + start_pos) * head_dim + d;
            k_cache[out_off] = k_in[in_off];
            v_cache[out_off] = v_in[in_off];
        });
    });

    // 2. Partial pass — NO-SLM + softmax-defer (latency-bound fix).
    auto partial_evt = ie::ps(q, "fa2_partial_fp16", [&](sycl::handler& h) {
        h.depends_on(append_evt);
        const uint32_t WG_ITEMS = gqa * SG_SIZE;
        const uint32_t dpl = head_dim / SG_SIZE;
        const float scale = 1.0f / sycl::sqrt(float(head_dim));

        h.parallel_for(sycl::nd_range<2>({uint64_t(n_kv_heads) * n_super_chunks, WG_ITEMS},
                                          {1, WG_ITEMS}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t kv_super_id = uint32_t(it.get_group(0));
            const uint32_t kv          = kv_super_id / n_super_chunks;
            const uint32_t super       = kv_super_id % n_super_chunks;
            const uint32_t lid         = uint32_t(it.get_local_id(1));
            const uint32_t sg_id       = lid / SG_SIZE;
            const uint32_t lane        = lid % SG_SIZE;
            const uint32_t my_q        = kv * gqa + sg_id;
            auto sg = it.get_sub_group();

            float q_vals[16];
            #pragma unroll
            for (uint32_t d_loc = 0; d_loc < dpl; ++d_loc)
                q_vals[d_loc] = float(q_in[uint64_t(my_q) * head_dim + lane * dpl + d_loc]);
            float m = -std::numeric_limits<float>::infinity();
            float l = 0.f;
            float out_local[16];
            #pragma unroll
            for (uint32_t d_loc = 0; d_loc < dpl; ++d_loc) out_local[d_loc] = 0.f;

            const uint32_t chunk_first = super * CHUNKS_PER_WG;
            for (uint32_t cc = 0; cc < CHUNKS_PER_WG; ++cc) {
                const uint32_t chunk = chunk_first + cc;
                if (chunk >= n_chunks) break;
                const uint32_t chunk_start = chunk * Bc;
                const uint32_t chunk_end   = sycl::min(chunk_start + Bc, ctx_len);
                const uint32_t chunk_n     = chunk_end - chunk_start;

                // NO SLM staging (llama fattn-vec): read K/V straight from cache —
                // gqa-sibling subgroups read the same rows ~concurrently so L2
                // absorbs the redundancy, and we drop the cooperative SLM load +
                // 2 barriers/chunk (pure latency on this 11%-BW latency-bound path).
                const uint64_t kvb =
                    (uint64_t(kv) * max_ctx + chunk_start) * head_dim + lane * dpl;

                // SOFTMAX-DEFER (llama fattn-vec technique): per TILE of positions,
                // pass 1 computes all scores (independent reduces → pipelined ILP)
                // and the tile max via a cheap register fmax (no exp); then rescale
                // the output accumulator ONCE per tile (not per position); pass 2
                // accumulates e·V with a fixed reference max. Drops the O(N) out
                // rescales + alpha-exps to O(N/TILE); only the weight exp stays
                // per-position. Numerically equivalent (PPL-gated, not bit-exact).
                for (uint32_t i0 = 0; i0 < chunk_n; i0 += BLK) {
                    const uint32_t ntile = sycl::min(BLK, chunk_n - i0);
                    float s[BLK];
                    float tile_max = -std::numeric_limits<float>::infinity();
                    #pragma unroll
                    for (uint32_t b = 0; b < BLK; ++b) {
                        const uint32_t i = i0 + b;
                        float partial = 0.f;
                        if (i < chunk_n) {
                            #pragma unroll
                            for (uint32_t d_loc = 0; d_loc < dpl; ++d_loc)
                                partial += q_vals[d_loc] *
                                    float(k_cache[kvb + uint64_t(i) * head_dim + d_loc]);
                        }
                        s[b] = sycl::reduce_over_group(sg, partial, sycl::plus<float>()) * scale;
                        if (b < ntile) tile_max = sycl::fmax(tile_max, s[b]);
                    }
                    const float m_new = sycl::fmax(m, tile_max);
                    const float alpha = sycl::native::exp(m - m_new);
                    #pragma unroll
                    for (uint32_t d_loc = 0; d_loc < dpl; ++d_loc)
                        out_local[d_loc] *= alpha;            // rescale ONCE per tile
                    l *= alpha;
                    for (uint32_t b = 0; b < ntile; ++b) {
                        const float e = sycl::native::exp(s[b] - m_new);
                        const uint32_t i = i0 + b;
                        #pragma unroll
                        for (uint32_t d_loc = 0; d_loc < dpl; ++d_loc)
                            out_local[d_loc] += e *
                                float(v_cache[kvb + uint64_t(i) * head_dim + d_loc]);
                        l += e;
                    }
                    m = m_new;
                }
            }

            const uint64_t base =
                (uint64_t(super) * n_q_heads + my_q) * (uint64_t(head_dim) + 2);
            if (lane == 0) { partials_scratch[base + 0] = m; partials_scratch[base + 1] = l; }
            #pragma unroll
            for (uint32_t d_loc = 0; d_loc < dpl; ++d_loc)
                partials_scratch[base + 2 + lane * dpl + d_loc] = out_local[d_loc];
        });
    });

    // 3. Combine pass — identical to v1.
    const uint32_t n_partials = n_super_chunks;
    auto combine_evt = ie::ps(q, "fa2_combine_fp16", [&](sycl::handler& h) {
        h.depends_on(partial_evt);
        const uint32_t dpl = head_dim / SG_SIZE;
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_q_heads) * SG_SIZE, SG_SIZE),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t qh   = uint32_t(it.get_group(0));
            const uint32_t lane = uint32_t(it.get_local_id(0));
            float m_global = -std::numeric_limits<float>::infinity();
            for (uint32_t c = 0; c < n_partials; ++c) {
                const uint64_t base = (uint64_t(c) * n_q_heads + qh) * (uint64_t(head_dim) + 2);
                m_global = sycl::fmax(m_global, partials_scratch[base + 0]);
            }
            float l_global = 0.f;
            float out_local[16];
            #pragma unroll
            for (uint32_t d_loc = 0; d_loc < dpl; ++d_loc) out_local[d_loc] = 0.f;
            for (uint32_t c = 0; c < n_partials; ++c) {
                const uint64_t base = (uint64_t(c) * n_q_heads + qh) * (uint64_t(head_dim) + 2);
                const float m_c = partials_scratch[base + 0];
                const float l_c = partials_scratch[base + 1];
                const float w   = sycl::native::exp(m_c - m_global);
                l_global += l_c * w;
                #pragma unroll
                for (uint32_t d_loc = 0; d_loc < dpl; ++d_loc)
                    out_local[d_loc] += w * partials_scratch[base + 2 + lane * dpl + d_loc];
            }
            const float inv_l = 1.0f / l_global;
            #pragma unroll
            for (uint32_t d_loc = 0; d_loc < dpl; ++d_loc) {
                const uint64_t y_off = uint64_t(qh) * head_dim + lane * dpl + d_loc;
                y[y_off] = sycl::half(out_local[d_loc] * inv_l);
            }
        });
    });
    if (prof) {
        combine_evt.wait();
        auto dur = [](const sycl::event& e) -> uint64_t {
            return e.get_profiling_info<sycl::info::event_profiling::command_end>()
                 - e.get_profiling_info<sycl::info::event_profiling::command_start>();
        };
        prof->append_ns  += dur(append_evt);
        prof->partial_ns += dur(partial_evt);
        prof->combine_ns += dur(combine_evt);
    }
    return combine_evt;
}

// =====================================================================
// FA-2 split-K decode VEC — faithful port of llama flash_attn_ext_vec
// =====================================================================
// Root-cause fix for the 3.85x decode gap. v1/v2 map SG=16 lanes across the
// SAME head_dim and score positions one-at-a-time with a full width-16
// reduce_over_group on the serial online-softmax chain (v2 batched only the
// exp, NOT the reduce → flat). llama's vec path instead maps:
//   lane = head_dim slice, with NTH_KQ=8 lanes cooperating on ONE K·Q dot.
// So one SG16 step scores TWO KV positions (lane group lane/8 ∈ {0,1}); the
// reduce is a NARROW width-8 butterfly (xor 4,2,1) within each contiguous
// 8-lane group, NOT a full width-16 reduce. NTH_KQ score dots over a tile are
// filled into a register array with the butterfly reduces issued back-to-back
// (OFF the softmax chain — running max via plain register fmax), THEN one
// exp-rescale of m/l/VKQ per tile, THEN P@V as pure per-lane FMA: each lane
// OWNS VPL=head_dim/NTH_V output dims and reads the softmax weight from SLM —
// zero cross-lane in the V combine (that is why lane=D-slice wins). K/V loaded
// as vectorized sycl::vec<half,8> (16B), not scalar half.
//
// Reuses the v1/v2 super-chunk split-K + partials_scratch format + combine pass
// VERBATIM (same {m, l, out[head_dim]} layout) so combine is shared and the
// partials buffer fits unchanged. Partials are UNNORMALIZED until combine.
// Correct for head_dim=128 (Coder), gqa = n_q_heads / n_kv_heads. NCOLS=1
// (decode T=1: one q-head per subgroup, gqa siblings re-read K/V and rely on L2,
// exactly as llama's vec path at cols_per_block=1). No ESIMD/block2d/joint_matrix.
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
                                          const std::vector<sycl::event>& deps,
                                          AttnProfileData* prof) {
    constexpr uint32_t Bc = 64;
    constexpr uint32_t TARGET_SUPER = IE_FA2_TARGET_SUPER;
    const uint32_t ctx_len = start_pos + 1;
    const uint32_t n_chunks = (ctx_len + Bc - 1) / Bc;
    const uint32_t CHUNKS_PER_WG =
        std::max<uint32_t>(1u, (n_chunks + TARGET_SUPER - 1) / TARGET_SUPER);
    const uint32_t n_super_chunks = (n_chunks + CHUNKS_PER_WG - 1) / CHUNKS_PER_WG;
    const uint32_t gqa = n_q_heads / n_kv_heads;

    // 1. Append (k_in, v_in) for THIS new token to the cache (identical to v1/v2).
    auto append_evt = ie::ps(q, "fa2_append_fp16", [&](sycl::handler& h) {
        h.depends_on(deps);
        const uint32_t total = n_kv_heads * head_dim;
        constexpr uint32_t WG = 256;
        const uint32_t global = ((total + WG - 1) / WG) * WG;
        h.parallel_for(sycl::nd_range<1>(global, WG), [=](sycl::nd_item<1> it) {
            const uint32_t idx = uint32_t(it.get_global_id(0));
            if (idx >= total) return;
            const uint32_t d  = idx % head_dim;
            const uint32_t kv = idx / head_dim;
            const uint64_t in_off  = uint64_t(kv) * head_dim + d;
            const uint64_t out_off = (uint64_t(kv) * max_ctx + start_pos) * head_dim + d;
            k_cache[out_off] = k_in[in_off];
            v_cache[out_off] = v_in[in_off];
        });
    });

    // 2. Partial pass — VEC layout. One subgroup (SG16) per (kv_head, super, q-head).
    //    Lanes split into NTH_KQ=8-lane dot groups → 2 KV positions scored/step.
    auto partial_evt = ie::ps(q, "fa2_partial_fp16_vec", [&](sycl::handler& h) {
        h.depends_on(append_evt);

        // A/B knobs (maintainer): NTH_KQ lanes cooperate on one K·Q dot, so the
        // SG16 scores SG_SIZE/NTH_KQ positions per step. NTH_V is the analogous
        // V-column fan-out. Default 8 (= POS_PER_STEP 2). Try 4 to score 4
        // KV-positions/step (more reduce ILP, smaller Q slice/lane). Both must
        // divide SG_SIZE; DPL_KQ register array (q_vals[]) is sized to the
        // worst case (NTH_KQ small → bigger slice), so changing them is safe.
        constexpr uint32_t NTH_KQ = 8;
        constexpr uint32_t NTH_V  = 8;
        const uint32_t WG_ITEMS = gqa * SG_SIZE;           // gqa subgroups / WG
        const uint32_t POS_PER_STEP = SG_SIZE / NTH_KQ;    // = 2
        const uint32_t DPL_KQ = head_dim / NTH_KQ;         // 16 dims/lane for the dot
        // P@V output partition: ALL SG_SIZE lanes own DISTINCT out-dim slices
        // (the 2 lane_v_groups each own half the head_dim), so P@V runs at full
        // SG-width ILP — no redundant recompute, no guarded write-drop.
        const uint32_t VPL    = head_dim / SG_SIZE;        // 8 out-dims/lane (distinct)
        const float scale = 1.0f / sycl::sqrt(float(head_dim));

        // SLM: per-subgroup softmax weights for one POS_PER_STEP-batch of scores.
        // Layout: [sg_id][POS_PER_STEP]. Tiny (gqa*2 floats).
        sycl::local_accessor<float, 1> KQ_slm({uint64_t(gqa) * POS_PER_STEP}, h);

        h.parallel_for(sycl::nd_range<2>({uint64_t(n_kv_heads) * n_super_chunks, WG_ITEMS},
                                          {1, WG_ITEMS}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t kv_super_id = uint32_t(it.get_group(0));
            const uint32_t kv          = kv_super_id / n_super_chunks;
            const uint32_t super       = kv_super_id % n_super_chunks;
            const uint32_t lid         = uint32_t(it.get_local_id(1));
            const uint32_t sg_id       = lid / SG_SIZE;
            const uint32_t lane        = lid % SG_SIZE;
            const uint32_t my_q        = kv * gqa + sg_id;
            auto sg = it.get_sub_group();

            // Lane role within the SG16:
            //   KQ dot: lane_kq_group = lane / NTH_KQ ∈ {0,1} selects which of the
            //           POS_PER_STEP positions this lane helps score; lane_in_kq =
            //           lane % NTH_KQ selects which DPL_KQ-slice of head_dim.
            const uint32_t lane_kq_group = lane / NTH_KQ;   // 0..POS_PER_STEP-1
            const uint32_t lane_in_kq    = lane % NTH_KQ;   // 0..NTH_KQ-1
            //   V P@V: each of the SG_SIZE lanes owns a DISTINCT out-dim slice
            //          [lane*VPL, +VPL). The two lane_v_groups (lanes 0-7, 8-15)
            //          thus cover the two halves of head_dim — no overlap, so the
            //          P@V FMA runs at full SG-width ILP and the write needs no
            //          guard. (NTH_V retained only as the documented A/B knob.)
            const uint32_t out_dim0     = lane * VPL;       // this lane's first out-dim
            (void)NTH_V;

            // Q slice this lane needs for the dot (lane_in_kq owns DPL_KQ dims).
            // Sized to the worst case in the A/B range: NTH_KQ=4 → head_dim/4=32
            // (head_dim≤128). Only [0,DPL_KQ) is touched (runtime-bounded loops).
            float q_vals[32];
            #pragma unroll
            for (uint32_t d_loc = 0; d_loc < DPL_KQ; ++d_loc) {
                q_vals[d_loc] = float(q_in[uint64_t(my_q) * head_dim
                                           + lane_in_kq * DPL_KQ + d_loc]) * scale;
            }

            // Output accumulator: this lane owns out-dims [lane_in_v*VPL, +VPL).
            float m = -3.0e38f;                 // -FLT_MAX/2, not -inf (NaN on step 0)
            float l = 0.f;
            float out_local[16];
            #pragma unroll
            for (uint32_t d_loc = 0; d_loc < VPL; ++d_loc) out_local[d_loc] = 0.f;

            const uint32_t chunk_first = super * CHUNKS_PER_WG;
            for (uint32_t cc = 0; cc < CHUNKS_PER_WG; ++cc) {
                const uint32_t chunk = chunk_first + cc;
                if (chunk >= n_chunks) break;
                const uint32_t chunk_start = chunk * Bc;
                const uint32_t chunk_end   = sycl::min(chunk_start + Bc, ctx_len);
                const uint32_t chunk_n     = chunk_end - chunk_start;

                // Score POS_PER_STEP positions per step; defer the softmax update.
                for (uint32_t p0 = 0; p0 < chunk_n; p0 += POS_PER_STEP) {
                    // ---- Pass A: raw dots into register, narrow butterfly reduce,
                    //              running max via register fmax (OFF softmax chain).
                    const uint32_t pos = p0 + lane_kq_group;      // position this lane scores
                    float s_lane = -3.0e38f;
                    if (pos < chunk_n) {
                        const uint64_t kbase =
                            (uint64_t(kv) * max_ctx + chunk_start + pos) * head_dim
                            + lane_in_kq * DPL_KQ;
                        float acc = 0.f;
                        // Vectorized 16B K load (sycl::vec<half,8>), DPL_KQ=16 = 2 vecs.
                        #pragma unroll
                        for (uint32_t v8 = 0; v8 < DPL_KQ; v8 += 8) {
                            sycl::vec<sycl::half, 8> kv8;
                            kv8.load(0, sycl::address_space_cast<
                                            sycl::access::address_space::global_space,
                                            sycl::access::decorated::no>(
                                            const_cast<const sycl::half*>(k_cache) + kbase + v8));
                            #pragma unroll
                            for (uint32_t e = 0; e < 8; ++e)
                                acc += q_vals[v8 + e] * float(kv8[e]);
                        }
                        // Narrow width-NTH_KQ butterfly reduce (xor 4,2,1 for NTH_KQ=8).
                        #pragma unroll
                        for (uint32_t off = NTH_KQ >> 1; off > 0; off >>= 1)
                            acc += sycl::permute_group_by_xor(sg, acc, off);
                        s_lane = acc;     // every lane in the 8-group now has the full dot
                    }
                    // Broadcast each group's score to SLM so all lanes can read both.
                    // lane_in_kq==0 of each group writes its group's score.
                    if (pos < chunk_n && lane_in_kq == 0)
                        KQ_slm[sg_id * POS_PER_STEP + lane_kq_group] = s_lane;
                    sycl::group_barrier(sg);

                    // Tile max over the POS_PER_STEP scores (register fmax, no exp).
                    const uint32_t ntile = sycl::min(POS_PER_STEP, chunk_n - p0);
                    float tile_max = -3.0e38f;
                    #pragma unroll
                    for (uint32_t pp = 0; pp < POS_PER_STEP; ++pp)
                        if (pp < ntile)
                            tile_max = sycl::fmax(tile_max,
                                                  KQ_slm[sg_id * POS_PER_STEP + pp]);

                    // ---- One online-softmax rescale of m/l/VKQ for the tile.
                    const float m_new = sycl::fmax(m, tile_max);
                    const float alpha = sycl::native::exp(m - m_new);   // old-accum rescale
                    #pragma unroll
                    for (uint32_t d_loc = 0; d_loc < VPL; ++d_loc)
                        out_local[d_loc] *= alpha;                      // BEFORE P@V
                    l *= alpha;

                    // ---- Pass B: P@V — pure per-lane FMA, weight from SLM, V from HBM.
                    for (uint32_t pp = 0; pp < ntile; ++pp) {
                        const float e =
                            sycl::native::exp(KQ_slm[sg_id * POS_PER_STEP + pp] - m_new);
                        l += e;
                        const uint64_t vbase =
                            (uint64_t(kv) * max_ctx + chunk_start + p0 + pp) * head_dim
                            + out_dim0;
                        #pragma unroll
                        for (uint32_t v8 = 0; v8 < VPL; v8 += 8) {
                            sycl::vec<sycl::half, 8> vv8;
                            vv8.load(0, sycl::address_space_cast<
                                            sycl::access::address_space::global_space,
                                            sycl::access::decorated::no>(
                                            const_cast<const sycl::half*>(v_cache) + vbase + v8));
                            #pragma unroll
                            for (uint32_t ee = 0; ee < 8; ++ee)
                                out_local[v8 + ee] += e * float(vv8[ee]);
                        }
                    }
                    m = m_new;
                    sycl::group_barrier(sg);   // KQ_slm reuse next iter
                }
            }

            // Write one UNNORMALIZED super-partial per (super, my_q).
            // Each lane owns the DISTINCT out-dim slice [out_dim0, out_dim0+VPL):
            // lanes 0-15 tile the full head_dim exactly once (VPL=head_dim/SG_SIZE),
            // so every out-dim is written exactly once by the lane that computed it
            // — no guard, no double-write, no skipped dim.
            const uint64_t base =
                (uint64_t(super) * n_q_heads + my_q) * (uint64_t(head_dim) + 2);
            if (lane == 0) { partials_scratch[base + 0] = m; partials_scratch[base + 1] = l; }
            #pragma unroll
            for (uint32_t d_loc = 0; d_loc < VPL; ++d_loc)
                partials_scratch[base + 2 + out_dim0 + d_loc] = out_local[d_loc];
        });
    });

    // 3. Combine pass — identical layout to v1/v2 (global-max rescale + normalize).
    const uint32_t n_partials = n_super_chunks;
    auto combine_evt = ie::ps(q, "fa2_combine_fp16_vec", [&](sycl::handler& h) {
        h.depends_on(partial_evt);
        const uint32_t dpl = head_dim / SG_SIZE;
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_q_heads) * SG_SIZE, SG_SIZE),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t qh   = uint32_t(it.get_group(0));
            const uint32_t lane = uint32_t(it.get_local_id(0));
            float m_global = -3.0e38f;
            for (uint32_t c = 0; c < n_partials; ++c) {
                const uint64_t base = (uint64_t(c) * n_q_heads + qh) * (uint64_t(head_dim) + 2);
                m_global = sycl::fmax(m_global, partials_scratch[base + 0]);
            }
            float l_global = 0.f;
            float out_local[16];
            #pragma unroll
            for (uint32_t d_loc = 0; d_loc < dpl; ++d_loc) out_local[d_loc] = 0.f;
            for (uint32_t c = 0; c < n_partials; ++c) {
                const uint64_t base = (uint64_t(c) * n_q_heads + qh) * (uint64_t(head_dim) + 2);
                const float m_c = partials_scratch[base + 0];
                const float l_c = partials_scratch[base + 1];
                const float w   = sycl::native::exp(m_c - m_global);
                l_global += l_c * w;
                #pragma unroll
                for (uint32_t d_loc = 0; d_loc < dpl; ++d_loc)
                    out_local[d_loc] += w * partials_scratch[base + 2 + lane * dpl + d_loc];
            }
            const float inv_l = l_global > 0.f ? 1.0f / l_global : 0.f;   // guard l==0
            #pragma unroll
            for (uint32_t d_loc = 0; d_loc < dpl; ++d_loc) {
                const uint64_t y_off = uint64_t(qh) * head_dim + lane * dpl + d_loc;
                y[y_off] = sycl::half(out_local[d_loc] * inv_l);
            }
        });
    });
    if (prof) {
        combine_evt.wait();
        auto dur = [](const sycl::event& e) -> uint64_t {
            return e.get_profiling_info<sycl::info::event_profiling::command_end>()
                 - e.get_profiling_info<sycl::info::event_profiling::command_start>();
        };
        prof->append_ns  += dur(append_evt);
        prof->partial_ns += dur(partial_evt);
        prof->combine_ns += dur(combine_evt);
    }
    return combine_evt;
}

// =====================================================================
// FA-2 split-K decode with INT8 KV cache
// =====================================================================
//
// Identical algorithm to full_attention_fa2_decode, but:
//   1. Append step quantizes the new (k, v) per-row to symmetric INT8 +
//      one fp16 scale per (kv_head, position) row.  Writes BOTH the INT8
//      cache + scale arrays AND (optionally) the fp16 shadow cache that
//      naive prefill paths still depend on.
//   2. Partial pass cooperatively SLM-loads INT8 K and V tiles + the
//      per-row scales, then dequant inline (s * int8) when reading from
//      SLM.  KV bandwidth is ~half (256 INT8 vs 512 FP16 bytes per row,
//      plus a tiny scale).
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
                                           const std::vector<sycl::event>& deps,
                                           AttnProfileData* prof) {
    constexpr uint32_t Bc = 64;
    // Adaptive super-chunking: aim for ≥ TARGET_SUPER super-chunks per kv_head
    // so we keep the GPU well-subscribed even at short ctx.  See top-of-file
    // IE_FA2_TARGET_SUPER block for the regression history that drove the
    // value from 12 (introduced in 5be3385) to 64.  TL;DR: 12 saturated B70's
    // 24 Xe-cores in a single wave at ≥16k ctx with no pipeline hiding; 64
    // keeps short-ctx fan-out (n_chunks ≤ 64 → CHUNKS_PER_WG=1) AND restores
    // multi-wave occupancy at long ctx (32k → CHUNKS_PER_WG=8, 5.3 waves).
    constexpr uint32_t TARGET_SUPER = IE_FA2_TARGET_SUPER;
    const uint32_t ctx_len = start_pos + 1;
    const uint32_t n_chunks = (ctx_len + Bc - 1) / Bc;
    const uint32_t CHUNKS_PER_WG =
        std::max<uint32_t>(1u, (n_chunks + TARGET_SUPER - 1) / TARGET_SUPER);
    const uint32_t n_super_chunks = (n_chunks + CHUNKS_PER_WG - 1) / CHUNKS_PER_WG;
    const uint32_t gqa = n_q_heads / n_kv_heads;

    // 1. Append + quantize. One WG per kv_head (n_kv_heads is small, e.g. 2),
    //    head_dim work-items inside. Cooperatively reduce max-abs, derive
    //    scale, write INT8 + scale (and fp16 shadow if requested).
    const sycl::half* k_in_local = k_in;
    const sycl::half* v_in_local = v_in;
    auto append_evt = ie::ps(q, "fa2_append_int8", [&](sycl::handler& h) {
        h.depends_on(deps);
        const uint32_t WG = head_dim;          // = 256 for Qwen3.6

        h.parallel_for(sycl::nd_range<2>({n_kv_heads, WG}, {1, WG}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t kv = uint32_t(it.get_group(0));
            const uint32_t d  = uint32_t(it.get_local_id(1));

            const uint64_t in_off  = uint64_t(kv) * head_dim + d;
            const uint64_t out_off = (uint64_t(kv) * max_ctx + start_pos) * head_dim + d;
            const uint64_t scale_off = uint64_t(kv) * max_ctx + start_pos;

            const float k_val = float(k_in_local[in_off]);
            const float v_val = float(v_in_local[in_off]);
            if (k_fp16_shadow) k_fp16_shadow[out_off] = k_in_local[in_off];
            if (v_fp16_shadow) v_fp16_shadow[out_off] = v_in_local[in_off];

            const float k_max = sycl::reduce_over_group(
                it.get_group(), sycl::fabs(k_val), sycl::maximum<float>{});
            const float v_max = sycl::reduce_over_group(
                it.get_group(), sycl::fabs(v_val), sycl::maximum<float>{});

            const float k_scale = (k_max > 0.f) ? (k_max / 127.0f) : 1.0f;
            const float v_scale = (v_max > 0.f) ? (v_max / 127.0f) : 1.0f;
            const float k_inv   = (k_max > 0.f) ? (127.0f / k_max) : 0.0f;
            const float v_inv   = (v_max > 0.f) ? (127.0f / v_max) : 0.0f;

            const int k_q = int(sycl::round(k_val * k_inv));
            const int v_q = int(sycl::round(v_val * v_inv));
            k_int8_cache[out_off] = int8_t(sycl::clamp(k_q, -127, 127));
            v_int8_cache[out_off] = int8_t(sycl::clamp(v_q, -127, 127));

            if (d == 0) {
                k_scales_cache[scale_off] = sycl::half(k_scale);
                v_scales_cache[scale_off] = sycl::half(v_scale);
            }
        });
    });

    // 2. Partial pass: super-chunked — one WG per (kv_head, super_chunk).
    //    Each WG processes CHUNKS_PER_WG inner Bc-tiles serially with running
    //    online-softmax accumulators; writes ONE super-partial per (super, my_q).
    //    Mirrors fp16 full_attention_fa2_decode (this used to be one WG per
    //    n_chunks, paying ~8× more launch overhead at long ctx).
    auto partial_evt = ie::ps(q, "fa2_partial_int8", [&](sycl::handler& h) {
        h.depends_on(append_evt);
        // INT8 tiles: 16KB each (was 32KB as fp16) → total SLM ~32KB → 3 WGs/subslice on Xe2 128KB SLM.
        sycl::local_accessor<int8_t, 1>     K_slm({uint64_t(Bc) * head_dim}, h);
        // (this is fa2_partial_int8 — kernel name already registered by ie::ps above)
        sycl::local_accessor<int8_t, 1>     V_slm({uint64_t(Bc) * head_dim}, h);
        sycl::local_accessor<sycl::half, 1> K_scales_slm({Bc}, h);
        sycl::local_accessor<sycl::half, 1> V_scales_slm({Bc}, h);

        const uint32_t WG_ITEMS = gqa * SG_SIZE;
        const uint32_t dpl = head_dim / SG_SIZE;
        const float scale_attn = 1.0f / sycl::sqrt(float(head_dim));

        h.parallel_for(sycl::nd_range<2>({uint64_t(n_kv_heads) * n_super_chunks, WG_ITEMS},
                                          {1, WG_ITEMS}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t kv_super_id = uint32_t(it.get_group(0));
            const uint32_t kv          = kv_super_id / n_super_chunks;
            const uint32_t super       = kv_super_id % n_super_chunks;
            const uint32_t lid         = uint32_t(it.get_local_id(1));
            const uint32_t sg_id       = lid / SG_SIZE;
            const uint32_t lane        = lid % SG_SIZE;
            const uint32_t my_q        = kv * gqa + sg_id;
            auto sg = it.get_sub_group();

            // Per-SG q (registers); per-lane dpl dims.
            float q_vals[16];
            #pragma unroll
            for (uint32_t d_loc = 0; d_loc < dpl; ++d_loc) {
                q_vals[d_loc] = float(q_in[uint64_t(my_q) * head_dim + lane * dpl + d_loc]);
            }
            float m = -std::numeric_limits<float>::infinity();
            float l = 0.f;
            float out_local[16];
            #pragma unroll
            for (uint32_t d_loc = 0; d_loc < dpl; ++d_loc) out_local[d_loc] = 0.f;

            // Loop CHUNKS_PER_WG inner chunks, accumulating m/l/out across them.
            const uint32_t hd_log2 = 31 - sycl::clz(head_dim);
            const uint32_t hd_mask = head_dim - 1;
            const uint32_t chunk_first = super * CHUNKS_PER_WG;
            for (uint32_t cc = 0; cc < CHUNKS_PER_WG; ++cc) {
                const uint32_t chunk = chunk_first + cc;
                if (chunk >= n_chunks) break;

                const uint32_t chunk_start = chunk * Bc;
                const uint32_t chunk_end   = sycl::min(chunk_start + Bc, ctx_len);
                const uint32_t chunk_n     = chunk_end - chunk_start;

                // Load scales + raw INT8 tiles; dequant is inlined in the compute loop.
                // Single barrier (was 2) because scales are no longer needed during load.
                for (uint32_t i = lid; i < chunk_n; i += WG_ITEMS) {
                    const uint64_t s_off = uint64_t(kv) * max_ctx + chunk_start + i;
                    K_scales_slm[i] = k_scales_cache[s_off];
                    V_scales_slm[i] = v_scales_cache[s_off];
                }
                const uint32_t tile_size = chunk_n * head_dim;
                for (uint32_t i = lid; i < tile_size; i += WG_ITEMS) {
                    const uint32_t tk = i >> hd_log2;
                    const uint32_t d  = i & hd_mask;
                    const uint64_t off =
                        (uint64_t(kv) * max_ctx + chunk_start + tk) * head_dim + d;
                    K_slm[i] = k_int8_cache[off];
                    V_slm[i] = v_int8_cache[off];
                }
                sycl::group_barrier(it.get_group());

                for (uint32_t i = 0; i < chunk_n; ++i) {
                    const float ks = float(K_scales_slm[i]);
                    const float vs = float(V_scales_slm[i]);
                    float partial = 0.f;
                    #pragma unroll
                    for (uint32_t d_loc = 0; d_loc < dpl; ++d_loc) {
                        partial += q_vals[d_loc] *
                                   (float(K_slm[i * head_dim + lane * dpl + d_loc]) * ks);
                    }
                    const float s_i =
                        sycl::reduce_over_group(sg, partial, sycl::plus<float>()) * scale_attn;
                    const float m_new = sycl::fmax(m, s_i);
                    const float alpha = sycl::native::exp(m - m_new);
                    const float e     = sycl::native::exp(s_i - m_new);
                    #pragma unroll
                    for (uint32_t d_loc = 0; d_loc < dpl; ++d_loc) {
                        out_local[d_loc] = out_local[d_loc] * alpha +
                                           e * (float(V_slm[i * head_dim + lane * dpl + d_loc]) * vs);
                    }
                    l = l * alpha + e;
                    m = m_new;
                }
                sycl::group_barrier(it.get_group());  // before next K/V tile load
            }

            // Write one super-partial per (super, my_q).
            const uint64_t base =
                (uint64_t(super) * n_q_heads + my_q) * (uint64_t(head_dim) + 2);
            if (lane == 0) {
                partials_scratch[base + 0] = m;
                partials_scratch[base + 1] = l;
            }
            #pragma unroll
            for (uint32_t d_loc = 0; d_loc < dpl; ++d_loc) {
                partials_scratch[base + 2 + lane * dpl + d_loc] = out_local[d_loc];
            }
        });
    });

    // 3. Combine pass: iterates n_super_chunks (was n_chunks before super-chunk port).
    const uint32_t n_partials = n_super_chunks;
    auto combine_evt = ie::ps(q, "fa2_combine_int8", [&](sycl::handler& h) {
        h.depends_on(partial_evt);
        const uint32_t dpl = head_dim / SG_SIZE;
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_q_heads) * SG_SIZE, SG_SIZE),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t qh   = uint32_t(it.get_group(0));
            const uint32_t lane = uint32_t(it.get_local_id(0));

            float m_global = -std::numeric_limits<float>::infinity();
            for (uint32_t c = 0; c < n_partials; ++c) {
                const uint64_t base =
                    (uint64_t(c) * n_q_heads + qh) * (uint64_t(head_dim) + 2);
                m_global = sycl::fmax(m_global, partials_scratch[base + 0]);
            }
            float l_global = 0.f;
            float out_local[16];
            #pragma unroll
            for (uint32_t d_loc = 0; d_loc < dpl; ++d_loc) out_local[d_loc] = 0.f;
            for (uint32_t c = 0; c < n_partials; ++c) {
                const uint64_t base =
                    (uint64_t(c) * n_q_heads + qh) * (uint64_t(head_dim) + 2);
                const float m_c = partials_scratch[base + 0];
                const float l_c = partials_scratch[base + 1];
                const float w   = sycl::native::exp(m_c - m_global);
                l_global += l_c * w;
                #pragma unroll
                for (uint32_t d_loc = 0; d_loc < dpl; ++d_loc) {
                    out_local[d_loc] +=
                        w * partials_scratch[base + 2 + lane * dpl + d_loc];
                }
            }
            const float inv_l = 1.0f / l_global;
            #pragma unroll
            for (uint32_t d_loc = 0; d_loc < dpl; ++d_loc) {
                const uint64_t y_off = uint64_t(qh) * head_dim + lane * dpl + d_loc;
                y[y_off] = sycl::half(out_local[d_loc] * inv_l);
            }
        });
    });
    if (prof) {
        combine_evt.wait();
        auto dur = [](const sycl::event& e) -> uint64_t {
            return e.get_profiling_info<sycl::info::event_profiling::command_end>()
                 - e.get_profiling_info<sycl::info::event_profiling::command_start>();
        };
        prof->append_ns  += dur(append_evt);
        prof->partial_ns += dur(partial_evt);
        prof->combine_ns += dur(combine_evt);
    }
    return combine_evt;
}

// =====================================================================
// FlashAttention-2 tiled prefill (T>1)
// =====================================================================
//
// Replaces the naive `full_attention` for T>1 prefill.  The naive variant
// dispatches WG-per-(t,h) = T × n_q_heads WGs (e.g. 8192 at T=512, 16
// q_heads), each WG reading the full KV cache scattered.  GQA shares K/V
// across `gqa = n_q_heads / n_kv_heads = 8` q_heads so the naive layout
// reads each KV element 8× redundantly.
//
// Tiled FA-2 prefill layout:
//   1 WG per (kv_head, br_chunk_of_Br=8) — at T=512, 2 × 64 = 128 WGs.
//   gqa SGs per WG (one per q_head in the group), 16 lanes per SG.
//   Each WG cooperatively SLM-loads K_tile and V_tile (Bc × head_dim) ONCE
//   per K-tile, then all gqa SGs reuse it across the SG's Br Q-rows.
//   KV is read exactly once per kv_head per K-tile, eliminating the 8×
//   redundancy.  WG count drops 64× vs naive.
//
// Per-SG state in registers:
//   q_vals[Br][dpl=16]    — this q_head's Q values for the Br rows
//   out_local[Br][dpl=16] — accumulator
//   m[Br], l[Br]          — online-softmax running stats
//   ≈ 272 floats/lane = 34 GRFs of 128 — comfortable headroom.
//
// SLM per WG:
//   K_slm[Bc=64][head_dim=256] halfs = 32 KB
//   V_slm[Bc=64][head_dim=256] halfs = 32 KB
//   Total 64 KB → fits 2-3 WGs/subslice on B70's 192 KB Xe-core SLM.
//
// Causal mask is applied per-(q_row, k_pos) pair against absolute
// positions (start_pos + br_chunk*Br + r vs k_tile_start + k_local).

sycl::event full_attention_fa2_prefill(sycl::queue& q,
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
                                       const std::vector<sycl::event>& deps) {
    if (T == 0) return {};
    constexpr uint32_t Br = 8;
    constexpr uint32_t Bc = 64;
    const uint32_t gqa = n_q_heads / n_kv_heads;
    const uint32_t ctx_len = start_pos + T;
    const uint32_t n_br_chunks = (T + Br - 1) / Br;
    const uint32_t WG_ITEMS = gqa * SG_SIZE;       // gqa SGs × 16 lanes
    const uint32_t dpl = head_dim / SG_SIZE;

    // 1. Append (k_in, v_in) for THESE T tokens to cache at [start_pos, start_pos+T).
    auto append_evt = ie::ps(q, "fa2_prefill_append", [&](sycl::handler& h) {
        h.depends_on(deps);
        const uint32_t total = T * n_kv_heads * head_dim;
        constexpr uint32_t WG = 256;
        const uint32_t global = ((total + WG - 1) / WG) * WG;
        h.parallel_for(sycl::nd_range<1>(global, WG),
                       [=](sycl::nd_item<1> it) {
            const uint32_t idx = uint32_t(it.get_global_id(0));
            if (idx >= total) return;
            const uint32_t d  = idx % head_dim;
            const uint32_t kv = (idx / head_dim) % n_kv_heads;
            const uint32_t t  = idx / (n_kv_heads * head_dim);
            const uint64_t in_off  = (uint64_t(t) * n_kv_heads + kv) * head_dim + d;
            const uint64_t out_off = (uint64_t(kv) * max_ctx + (start_pos + t)) * head_dim + d;
            k_cache[out_off] = k_in[in_off];
            v_cache[out_off] = v_in[in_off];
        });
    });

    // 2. Tiled FA-2 compute.
    return ie::ps(q, "fa2_prefill_compute", [&](sycl::handler& h) {
        h.depends_on(append_evt);
        sycl::local_accessor<sycl::half, 2> K_slm({Bc, head_dim}, h);
        sycl::local_accessor<sycl::half, 2> V_slm({Bc, head_dim}, h);
        const float scale = 1.0f / sycl::sqrt(float(head_dim));

        h.parallel_for(sycl::nd_range<2>({uint64_t(n_kv_heads) * n_br_chunks, WG_ITEMS},
                                          {1, WG_ITEMS}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t wg0     = uint32_t(it.get_group(0));
            const uint32_t kv_head = wg0 / n_br_chunks;
            const uint32_t br_idx  = wg0 % n_br_chunks;
            const uint32_t lid     = uint32_t(it.get_local_id(1));
            const uint32_t sg_id   = lid / SG_SIZE;            // 0..gqa-1
            const uint32_t lane    = lid % SG_SIZE;
            const uint32_t q_head  = kv_head * gqa + sg_id;
            const uint32_t br_base = br_idx * Br;              // local within T
            auto sg = it.get_sub_group();

            // 2a. Cache Q values for this SG's q_head over all Br rows.
            float q_vals[Br][16];                              // dpl <= 16; head_dim<=256
            #pragma unroll
            for (uint32_t r = 0; r < Br; ++r) {
                const uint32_t t_local = br_base + r;
                if (t_local < T) {
                    const uint64_t q_base = (uint64_t(t_local) * n_q_heads + q_head)
                                              * head_dim + lane * dpl;
                    #pragma unroll
                    for (uint32_t d_loc = 0; d_loc < 16; ++d_loc) {
                        if (d_loc < dpl) {
                            q_vals[r][d_loc] = float(q_in[q_base + d_loc]);
                        } else {
                            q_vals[r][d_loc] = 0.f;
                        }
                    }
                } else {
                    #pragma unroll
                    for (uint32_t d_loc = 0; d_loc < 16; ++d_loc) q_vals[r][d_loc] = 0.f;
                }
            }

            // 2b. Init online-softmax state per Br row.
            float m_run[Br], l_run[Br], out_local[Br][16];
            #pragma unroll
            for (uint32_t r = 0; r < Br; ++r) {
                m_run[r] = -std::numeric_limits<float>::infinity();
                l_run[r] = 0.f;
                #pragma unroll
                for (uint32_t d_loc = 0; d_loc < 16; ++d_loc) out_local[r][d_loc] = 0.f;
            }

            // 2c. Loop over K-tiles in [0, ctx_len).
            const uint32_t n_k_tiles = (ctx_len + Bc - 1) / Bc;
            for (uint32_t kt = 0; kt < n_k_tiles; ++kt) {
                const uint32_t k_tile_start = kt * Bc;
                const uint32_t k_tile_len   = sycl::min(Bc, ctx_len - k_tile_start);

                // 2c-i. Cooperative K, V tile load.  WG_ITEMS lanes split
                // Bc * head_dim halfs each for K and V.
                {
                    const uint32_t total = Bc * head_dim;
                    for (uint32_t i = lid; i < total; i += WG_ITEMS) {
                        const uint32_t k_local = i / head_dim;
                        const uint32_t d       = i % head_dim;
                        if (k_local < k_tile_len) {
                            const uint64_t off =
                                (uint64_t(kv_head) * max_ctx + k_tile_start + k_local)
                                  * head_dim + d;
                            K_slm[k_local][d] = k_cache[off];
                            V_slm[k_local][d] = v_cache[off];
                        } else {
                            K_slm[k_local][d] = sycl::half(0.f);
                            V_slm[k_local][d] = sycl::half(0.f);
                        }
                    }
                }
                sycl::group_barrier(it.get_group());

                // 2c-ii. For each Br row, score against this K-tile, online softmax.
                #pragma unroll
                for (uint32_t r = 0; r < Br; ++r) {
                    const uint32_t t_local = br_base + r;
                    if (t_local >= T) continue;
                    const uint32_t abs_q_pos = start_pos + t_local;

                    // For each K position in this tile.
                    for (uint32_t k_local = 0; k_local < k_tile_len; ++k_local) {
                        const uint32_t k_abs = k_tile_start + k_local;
                        // Causal mask: skip K positions > Q position.
                        if (k_abs > abs_q_pos) break;  // tile ordered ascending → safe to break

                        // Lane partial: dot(q[lane*dpl..(lane+1)*dpl], K[k_local][lane*dpl..])
                        float partial = 0.f;
                        #pragma unroll
                        for (uint32_t d_loc = 0; d_loc < 16; ++d_loc) {
                            if (d_loc < dpl) {
                                partial += q_vals[r][d_loc]
                                         * float(K_slm[k_local][lane * dpl + d_loc]);
                            }
                        }
                        const float s_i =
                            sycl::reduce_over_group(sg, partial, sycl::plus<float>()) * scale;

                        // Online softmax update for row r.
                        const float m_new = sycl::fmax(m_run[r], s_i);
                        const float alpha = sycl::native::exp(m_run[r] - m_new);
                        const float e     = sycl::native::exp(s_i - m_new);

                        #pragma unroll
                        for (uint32_t d_loc = 0; d_loc < 16; ++d_loc) {
                            if (d_loc < dpl) {
                                out_local[r][d_loc] = out_local[r][d_loc] * alpha
                                                    + e * float(V_slm[k_local][lane * dpl + d_loc]);
                            }
                        }
                        l_run[r] = l_run[r] * alpha + e;
                        m_run[r] = m_new;
                    }
                }
                sycl::group_barrier(it.get_group());
            }

            // 2d. Normalize and write output.
            #pragma unroll
            for (uint32_t r = 0; r < Br; ++r) {
                const uint32_t t_local = br_base + r;
                if (t_local >= T) continue;
                const float inv_l = (l_run[r] > 0.f) ? (1.0f / l_run[r]) : 0.f;
                const uint64_t y_base = (uint64_t(t_local) * n_q_heads + q_head)
                                          * head_dim + lane * dpl;
                #pragma unroll
                for (uint32_t d_loc = 0; d_loc < 16; ++d_loc) {
                    if (d_loc < dpl) {
                        y[y_base + d_loc] = sycl::half(out_local[r][d_loc] * inv_l);
                    }
                }
            }
        });
    });
}

// ---------------------------------------------------------------------------
// full_attention_fa2_prefill_v2 — query-row-block FA-2 prefill (split head_dim).
//
// Three prior attempts and why they lost (all measured 2026-06-22):
//   * `attn_naive_compute` (baseline): HBM-bandwidth-bound on REDUNDANT KV
//     re-reads — each of T query rows re-streams the whole causal KV → O(T²)
//     traffic (~137 GB/layer @T=4096, ~80% of peak BW, 73% of prefill). BUT it
//     dispatches T·n_q_heads subgroups (max parallelism) and its inner loop is
//     ALU-efficient: head_dim split across 16 lanes (dpl MACs) + ONE subgroup
//     reduce per key.  Fastest so far precisely because the inner loop is good.
//   * old `fa2_prefill_compute`: SLM-tiled (kills HBM) but only n_kv·T/8 WGs
//     (under-occupied) AND looped Br rows SERIALLY inside one subgroup.
//   * v1 thread-per-row (one lane = one full query row): killed the subgroup
//     reduce but SERIALIZED all head_dim MACs onto one lane → 16× more per-lane
//     work + register spill → 2–3.5× SLOWER than naive.  The reduce is cheaper
//     than serializing head_dim.
//
// v2 keeps naive's GOOD inner loop and only fixes its ONE flaw (redundant HBM):
//   * SAME split-head-dim + subgroup-reduce inner loop as naive (ALU-efficient,
//     works for any head_dim incl. crown's 256).
//   * ONE query row per SUBGROUP (not per lane) → naive-level parallelism.
//   * Br subgroups per WG run in PARALLEL (not a serial row loop) and SHARE one
//     K/V SLM tile → each KV element read from HBM once per (q_head, q-block)
//     instead of once per query row → ~Br× less redundant HBM.
//   * One q_head per WG → n_q_heads·⌈T/Br⌉ WGs of Br·16 lanes → high occupancy.
//
// Layout: group(0) = q_head·n_q_blocks + q_block ; local(1) = Br·SG_SIZE lanes.
//   subgroup s (0..Br-1) owns query row q_block·Br + s ; lane splits head_dim.
// SLM per WG: (K+V)·Bc·head_dim halfs (Bc=32 → 16/32 KB at hd 128/256).
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
                                          const std::vector<sycl::event>& deps) {
    if (T == 0) return {};
    constexpr uint32_t Br = 16;             // query rows per WG = subgroups per WG
    constexpr uint32_t Bc = 32;             // key tile depth (32·256·2·2 = 32 KB SLM)
    const uint32_t gqa        = n_q_heads / n_kv_heads;
    const uint32_t ctx_len    = start_pos + T;
    const uint32_t n_q_blocks = (T + Br - 1) / Br;
    const uint32_t WG_ITEMS   = Br * SG_SIZE;
    const uint32_t dpl        = head_dim / SG_SIZE;   // dims per lane

    // 1. Append (k_in, v_in) for these T tokens — identical to naive.
    auto append_evt = ie::ps(q, "fa2_v2_append", [&](sycl::handler& h) {
        h.depends_on(deps);
        const uint32_t total = T * n_kv_heads * head_dim;
        constexpr uint32_t WG = 256;
        const uint32_t global = ((total + WG - 1) / WG) * WG;
        h.parallel_for(sycl::nd_range<1>(global, WG),
                       [=](sycl::nd_item<1> it) {
            const uint32_t idx = uint32_t(it.get_global_id(0));
            if (idx >= total) return;
            const uint32_t d  = idx % head_dim;
            const uint32_t kv = (idx / head_dim) % n_kv_heads;
            const uint32_t t  = idx / (n_kv_heads * head_dim);
            const uint64_t in_off  = (uint64_t(t) * n_kv_heads + kv) * head_dim + d;
            const uint64_t out_off = (uint64_t(kv) * max_ctx + (start_pos + t)) * head_dim + d;
            k_cache[out_off] = k_in[in_off];
            v_cache[out_off] = v_in[in_off];
        });
    });

    // 2. Query-row-block tiled compute (one subgroup per row, Br rows/WG).
    return ie::ps(q, "fa2_v2_compute", [&](sycl::handler& h) {
        h.depends_on(append_evt);
        sycl::local_accessor<sycl::half, 1> K_slm(Bc * head_dim, h);
        sycl::local_accessor<sycl::half, 1> V_slm(Bc * head_dim, h);
        const float scale = 1.0f / sycl::sqrt(float(head_dim));

        h.parallel_for(sycl::nd_range<2>({uint64_t(n_q_heads) * n_q_blocks, WG_ITEMS},
                                          {1, WG_ITEMS}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t wg0     = uint32_t(it.get_group(0));
            const uint32_t q_head  = wg0 / n_q_blocks;
            const uint32_t q_block = wg0 % n_q_blocks;
            const uint32_t kv_head = q_head / gqa;
            const uint32_t lid     = uint32_t(it.get_local_id(1));
            const uint32_t sg_id   = lid / SG_SIZE;             // 0..Br-1 → row
            const uint32_t lane    = lid % SG_SIZE;
            const uint32_t t_local = q_block * Br + sg_id;      // this subgroup's query row
            const uint32_t abs_q   = start_pos + t_local;
            const bool     active  = (t_local < T);
            auto sg = it.get_sub_group();

            // This lane's slice (dpl dims) of the subgroup's Q row, in registers.
            float q_vals[16];
            if (active) {
                const uint64_t qb = (uint64_t(t_local) * n_q_heads + q_head) * head_dim
                                    + lane * dpl;
                for (uint32_t d = 0; d < dpl; ++d) q_vals[d] = float(q_in[qb + d]);
            }

            float m = -std::numeric_limits<float>::infinity();
            float l = 0.f;
            float out_local[16];
            for (uint32_t d = 0; d < dpl; ++d) out_local[d] = 0.f;

            const uint32_t n_tiles = (ctx_len + Bc - 1) / Bc;
            for (uint32_t kt = 0; kt < n_tiles; ++kt) {
                const uint32_t k0   = kt * Bc;
                const uint32_t klen = sycl::min(Bc, ctx_len - k0);

                // Cooperative K/V tile load — all Br·16 lanes participate.
                const uint32_t tile_n = klen * head_dim;
                for (uint32_t i = lid; i < tile_n; i += WG_ITEMS) {
                    const uint32_t kl = i / head_dim;
                    const uint32_t d  = i % head_dim;
                    const uint64_t off = (uint64_t(kv_head) * max_ctx + k0 + kl) * head_dim + d;
                    K_slm[i] = k_cache[off];
                    V_slm[i] = v_cache[off];
                }
                sycl::group_barrier(it.get_group());

                if (active && k0 <= abs_q) {
                    const uint32_t kmax = sycl::min(klen, abs_q - k0 + 1);
                    for (uint32_t kl = 0; kl < kmax; ++kl) {
                        const uint32_t koff = kl * head_dim + lane * dpl;
                        float partial = 0.f;
                        for (uint32_t d = 0; d < dpl; ++d)
                            partial += q_vals[d] * float(K_slm[koff + d]);
                        const float s_i =
                            sycl::reduce_over_group(sg, partial, sycl::plus<float>()) * scale;

                        const float m_new = sycl::fmax(m, s_i);
                        const float alpha = sycl::native::exp(m - m_new);
                        const float e     = sycl::native::exp(s_i - m_new);
                        for (uint32_t d = 0; d < dpl; ++d)
                            out_local[d] = out_local[d] * alpha + e * float(V_slm[koff + d]);
                        l = l * alpha + e;
                        m = m_new;
                    }
                }
                sycl::group_barrier(it.get_group());
            }

            if (active) {
                const float inv_l = (l > 0.f) ? (1.0f / l) : 0.f;
                const uint64_t yb = (uint64_t(t_local) * n_q_heads + q_head) * head_dim
                                    + lane * dpl;
                for (uint32_t d = 0; d < dpl; ++d)
                    y[yb + d] = sycl::half(out_local[d] * inv_l);
            }
        });
    });
}

// ===========================================================================
// full_attention_fa2_prefill_tile — faithful port of llama.cpp-SYCL's
// flash_attn_tile (ggml/src/ggml-sycl/fattn-tile.hpp), adapted to OUR buffers.
// ===========================================================================
//
// llama-SYCL's prefill attention does NOT use the XMX/joint_matrix engine — it
// uses a SIMD "tile" kernel: a big workgroup (256 work-items = 16 subgroups of
// 16) processes a block of `ncols` query rows that SHARE a cooperatively-loaded,
// half2-vectorized K/V SLM tile (`nbatch_fa`=64 KV positions per tile), iterating
// KV tiles with FA-2 online softmax. The lever vs our v2 is the vectorized,
// fully-coalesced SLM load pattern (llama's flash_attn_tile_load_tile) plus the
// full [Br×Bc] score tile (no per-key subgroup reduce).
//
// This is a SIMPLIFIED, hd=128-only port (no ALiBi / softcap / sinks / quantized
// KV — plain causal mask + scale). For any other head_dim it falls through to v2.
//
// Geometry (hardcoded for hd=128):
//   WG  = 256 work-items (NSG=16 subgroups × SG_SIZE=16).  Br = ncols = 16
//   query rows per WG; Bc = nbatch_fa = 64 KV positions per tile.
//   grid: {n_q_heads * n_q_blocks, 256}; group(0) = q_head*n_q_blocks + q_block.
//
// Work distribution across the 256 lanes (lid = 0..255):
//   * K/V SLM staging: half2-vectorized — each tile is Bc*hd/2 = 4096 half2;
//     256 lanes each move 16 contiguous half2 (fully coalesced, key-major).
//   * Score tile S[16×64] = 1024 cells: 4 cells/lane; cell = dot(Q_row, K_key)
//     over 128 dims in registers (fp32), causal-masked, stored to SLM.
//   * Online softmax: one lane per row (lid 0..15), 240 idle — short step.
//   * P·V into O_slm[16×128] = 2048 cells: 8 cells/lane; accumulate onto the
//     alpha-rescaled O_slm (true FA-2 O += P·V).
//
// SLM per WG: Q 4KB + K 16KB + V 16KB + S(f32) 4KB + P(f16) 2KB + O(f32) 8KB +
//   m,l(f32) 128B ≈ 50 KB — well under the 128 KB/WG cap.
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
                                            const std::vector<sycl::event>& deps) {
    if (T == 0) return {};
    if (head_dim != 128)
        return full_attention_fa2_prefill_v2(q, q_in, k_in, v_in, k_cache, v_cache,
                                             y, T, start_pos, n_q_heads, n_kv_heads,
                                             head_dim, max_ctx, deps);

    constexpr uint32_t HD       = 128;          // head_dim (compile-time)
    constexpr uint32_t Br       = 16;           // query rows per WG (ncols)
    constexpr uint32_t Bc       = 64;           // KV positions per tile (nbatch_fa)
    constexpr uint32_t WG_ITEMS = 256;          // 16 subgroups × 16 lanes
    const uint32_t gqa        = n_q_heads / n_kv_heads;
    const uint32_t ctx_len    = start_pos + T;
    const uint32_t n_q_blocks = (T + Br - 1) / Br;

    // 1. Append (k_in, v_in) for these T tokens — identical to v2.
    auto append_evt = ie::ps(q, "fa2_tile_append", [&](sycl::handler& h) {
        h.depends_on(deps);
        const uint32_t total = T * n_kv_heads * head_dim;
        constexpr uint32_t WG = 256;
        const uint32_t global = ((total + WG - 1) / WG) * WG;
        h.parallel_for(sycl::nd_range<1>(global, WG),
                       [=](sycl::nd_item<1> it) {
            const uint32_t idx = uint32_t(it.get_global_id(0));
            if (idx >= total) return;
            const uint32_t d  = idx % head_dim;
            const uint32_t kv = (idx / head_dim) % n_kv_heads;
            const uint32_t t  = idx / (n_kv_heads * head_dim);
            const uint64_t in_off  = (uint64_t(t) * n_kv_heads + kv) * head_dim + d;
            const uint64_t out_off = (uint64_t(kv) * max_ctx + (start_pos + t)) * head_dim + d;
            k_cache[out_off] = k_in[in_off];
            v_cache[out_off] = v_in[in_off];
        });
    });

    // 2. Tile FA-2 compute. One WG per (q_head, q_block).
    //
    // FAITHFUL llama flash_attn_tile mapping (SG16-adapted, hd=128):
    //   nthreads=256 → NSG=16 subgroups of SG=16 lanes.
    //   ncols = Br = 16  → cpw = ncols/NSG = 1  (ONE query row per subgroup)
    //                      np  = NSG/ncols = 1  (ONE subgroup per column).
    //   nbatch_fa = Bc = 64 KV positions per tile.
    //
    // Per-lane register accumulators (llama's KQ_acc / VKQ — NO idle lanes):
    //   * KQ_acc[KVPL]  : KVPL = nbatch_fa/(np*SG) = 64/16 = 4 raw scores.
    //     Lane L of subgroup r computes the COMPLETE dot Q[row r]·K[kv] over the
    //     full head_dim for its 4 KV positions kv ∈ {L, L+16, L+32, L+48}.
    //     There is NO per-key subgroup reduce — each lane owns whole dots. All
    //     16 lanes × 16 subgroups = 256 lanes each producing 4 complete scores.
    //   * VKQ[DPL]      : DPL = head_dim/SG = 128/16 = 8 output dims.
    //     Lane L of subgroup r owns out-dims {L*8 .. L*8+7} of query row r. P·V is
    //     a pure per-lane FMA loop over the 64 KV positions; the softmax weight is
    //     broadcast from the SG-shared P_slm row — ZERO cross-lane in the V combine
    //     (this is why lane=out-dim-slice wins).
    //
    // Q is loaded ONCE per WG into SLM (scaled), resident across all KV tiles.
    // The running online-softmax max is a width-16 warp_reduce_max over each
    // lane's 4 scores (only place the subgroup cooperates in KQ).
    constexpr uint32_t NSG  = WG_ITEMS / SG_SIZE;   // 16 subgroups
    constexpr uint32_t KVPL = Bc / SG_SIZE;         // 4 KV positions per lane
    constexpr uint32_t DPL  = HD / SG_SIZE;         // 8 out-dims per lane
    static_assert(NSG == Br, "one subgroup per query row (cpw==1, np==1)");

    return ie::ps(q, "fa2_tile_compute", [&](sycl::handler& h) {
        h.depends_on(append_evt);
        sycl::local_accessor<sycl::half, 1> Q_slm(Br * HD, h);   // [16×128] scaled, resident
        sycl::local_accessor<sycl::half, 1> K_slm(Bc * HD, h);   // [64×128] key-major
        sycl::local_accessor<sycl::half, 1> V_slm(Bc * HD, h);   // [64×128] key-major
        sycl::local_accessor<float, 1> P_slm(Br * Bc, h);   // [16×64] softmax wts (fp32 — matches v2 precision)
        const float scale = 1.0f / sycl::sqrt(float(HD));

        h.parallel_for(sycl::nd_range<2>({uint64_t(n_q_heads) * n_q_blocks, WG_ITEMS},
                                          {1, WG_ITEMS}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t wg0     = uint32_t(it.get_group(0));
            const uint32_t q_head  = wg0 / n_q_blocks;
            const uint32_t q_block = wg0 % n_q_blocks;
            const uint32_t kv_head = q_head / gqa;
            const uint32_t lid     = uint32_t(it.get_local_id(1));   // 0..255
            const uint32_t sg_id   = lid / SG_SIZE;                  // 0..15 → query row
            const uint32_t lane    = lid % SG_SIZE;                  // 0..15
            const uint32_t q0      = q_block * Br;                   // first query row (local)
            auto sg = it.get_sub_group();

            // This subgroup owns query row r = sg_id (cpw==1, np==1).
            const uint32_t r      = sg_id;
            const uint32_t tl     = q0 + r;
            const uint32_t abs_q  = start_pos + tl;
            const bool     active = (tl < T);

            // half2 views of the cache (rows are 128 contiguous halfs → even
            // base offsets → 4-byte aligned half2 reads).
            const sycl::half2* k_cache2 = reinterpret_cast<const sycl::half2*>(k_cache);
            const sycl::half2* v_cache2 = reinterpret_cast<const sycl::half2*>(v_cache);
            sycl::half2* K_slm2 = reinterpret_cast<sycl::half2*>(
                K_slm.get_multi_ptr<sycl::access::decorated::no>().get());
            sycl::half2* V_slm2 = reinterpret_cast<sycl::half2*>(
                V_slm.get_multi_ptr<sycl::access::decorated::no>().get());

            // --- Load this WG's Q tile [Br×HD] into SLM ONCE, scaled (zero-pad OOB). ---
            for (uint32_t i = lid; i < Br * HD; i += WG_ITEMS) {
                const uint32_t rr = i / HD;
                const uint32_t d  = i % HD;
                const uint32_t t2 = q0 + rr;
                // Store Q UNSCALED (full fp16 precision); the 1/sqrt(d) scale is
                // applied to the fp32 score post-dot (matches v2 — pre-scaling Q in
                // fp16 lost mantissa bits and moved Coder PPL +0.86% vs v2).
                Q_slm[i] = (t2 < T)
                    ? q_in[(uint64_t(t2) * n_q_heads + q_head) * HD + d]
                    : sycl::half(0);
            }
            sycl::group_barrier(it.get_group());

            // --- Per-subgroup online-softmax stats + per-lane VKQ register tile. ---
            float m = -std::numeric_limits<float>::infinity();   // running max (per row)
            float l = 0.f;                                        // running denom (per row)
            float vkq[DPL];                                       // lane owns out-dims r*?  no: dims lane*DPL..
            #pragma unroll
            for (uint32_t d = 0; d < DPL; ++d) vkq[d] = 0.f;

            // Causal extent for this WG (uniform across lanes → safe break).
            const uint32_t wg_last_row = sycl::min(q0 + Br - 1, T - 1);
            const uint32_t abs_q_last  = start_pos + wg_last_row;

            const uint32_t n_tiles = (ctx_len + Bc - 1) / Bc;
            for (uint32_t kt = 0; kt < n_tiles; ++kt) {
                const uint32_t k0   = kt * Bc;
                const uint32_t klen = sycl::min(Bc, ctx_len - k0);
                if (k0 > abs_q_last) break;     // future for the whole WG (uniform)

                // --- Cooperative half2-vectorized K/V tile load [Bc×HD] key-major. ---
                // Bc*HD/2 = 4096 half2; 256 lanes → 16 half2 each, contiguous.
                constexpr uint32_t TILE_H2 = (Bc * HD) / 2;   // 4096
                for (uint32_t i = lid; i < TILE_H2; i += WG_ITEMS) {
                    const uint32_t kl = i / (HD / 2);         // key position in tile
                    const uint32_t d2 = i % (HD / 2);         // half2 index within row
                    if (kl < klen) {
                        const uint64_t off2 =
                            ((uint64_t(kv_head) * max_ctx + k0 + kl) * HD) / 2 + d2;
                        K_slm2[i] = k_cache2[off2];
                        V_slm2[i] = v_cache2[off2];
                    } else {
                        K_slm2[i] = sycl::half2(0, 0);
                        V_slm2[i] = sycl::half2(0, 0);
                    }
                }
                sycl::group_barrier(it.get_group());

                // --- iter_KQ: per-lane COMPLETE dots → KQ_acc[KVPL] (no idle lanes). ---
                // Lane L of subgroup r computes the full head_dim dot of its query
                // row against KV positions {L, L+16, L+32, L+48} of this tile.
                // half2 read of both Q (resident SLM) and K (tile SLM): HD/2 steps.
                float kq[KVPL];
                #pragma unroll
                for (uint32_t kk = 0; kk < KVPL; ++kk) kq[kk] = 0.f;

                if (active) {
                    const sycl::half2* qrow2 =
                        reinterpret_cast<const sycl::half2*>(&Q_slm[r * HD]);
                    #pragma unroll
                    for (uint32_t kk = 0; kk < KVPL; ++kk) {
                        const uint32_t c = lane + kk * SG_SIZE;        // KV pos in tile
                        const sycl::half2* krow2 =
                            reinterpret_cast<const sycl::half2*>(&K_slm[c * HD]);
                        float acc = 0.f;
                        #pragma unroll
                        for (uint32_t d2 = 0; d2 < HD / 2; ++d2) {
                            const sycl::half2 qd = qrow2[d2];
                            const sycl::half2 kd = krow2[d2];
                            acc += float(qd.x()) * float(kd.x())
                                 + float(qd.y()) * float(kd.y());
                        }
                        kq[kk] = acc * scale;   // scale the fp32 score (Q stored unscaled)
                    }
                }

                // --- Apply causal mask, compute this tile's per-row max. ---
                float tile_max = -std::numeric_limits<float>::infinity();
                #pragma unroll
                for (uint32_t kk = 0; kk < KVPL; ++kk) {
                    const uint32_t c     = lane + kk * SG_SIZE;
                    const uint32_t abs_k = k0 + c;
                    const bool keep = active && (c < klen) && (abs_k <= abs_q);
                    if (!keep) kq[kk] = -std::numeric_limits<float>::infinity();
                    tile_max = sycl::fmax(tile_max, kq[kk]);
                }
                // width-16 max over the subgroup → row max over all 64 KV positions.
                tile_max = sycl::reduce_over_group(sg, tile_max, sycl::maximum<float>());

                const float m_new = sycl::fmax(m, tile_max);
                const float alpha = (m == -std::numeric_limits<float>::infinity())
                                    ? 0.f : sycl::native::exp(m - m_new);

                // --- exp weights → P_slm[row][kv], denom update, VKQ rescale. ---
                float row_l_add = 0.f;
                #pragma unroll
                for (uint32_t kk = 0; kk < KVPL; ++kk) {
                    const uint32_t c = lane + kk * SG_SIZE;
                    const float e = (kq[kk] == -std::numeric_limits<float>::infinity())
                                    ? 0.f : sycl::native::exp(kq[kk] - m_new);
                    if (active) P_slm[r * Bc + c] = e;
                    row_l_add += e;
                }
                row_l_add = sycl::reduce_over_group(sg, row_l_add, sycl::plus<float>());
                l = l * alpha + row_l_add;
                m = m_new;
                // rescale this lane's VKQ accumulators ONCE per tile (FA-2).
                #pragma unroll
                for (uint32_t d = 0; d < DPL; ++d) vkq[d] *= alpha;

                sycl::group_barrier(it.get_group());  // P_slm ready for all lanes

                // --- VKQ += P·V (pure per-lane FMA, no cross-lane). ---
                // Lane L owns out-dims {L*DPL .. L*DPL+DPL-1}; weight from P_slm.
                if (active && k0 <= abs_q) {
                    const uint32_t kmax = sycl::min(klen, abs_q - k0 + 1);
                    for (uint32_t c = 0; c < kmax; ++c) {
                        const float p = float(P_slm[r * Bc + c]);
                        const uint32_t vbase = c * HD + lane * DPL;
                        #pragma unroll
                        for (uint32_t d = 0; d < DPL; ++d)
                            vkq[d] += p * float(V_slm[vbase + d]);
                    }
                }
                sycl::group_barrier(it.get_group());  // before next K/V tile load
            }

            // --- Normalize and write y (active rows only). ---
            if (active) {
                const float inv_l = (l > 0.f) ? (1.0f / l) : 0.f;
                const uint64_t yb = (uint64_t(tl) * n_q_heads + q_head) * HD + lane * DPL;
                #pragma unroll
                for (uint32_t d = 0; d < DPL; ++d)
                    y[yb + d] = sycl::half(vkq[d] * inv_l);
            }
        });
    });
}

// ===========================================================================
// fa2_tile_wide_impl<HD,Bc> — head_dim 256/512 tile kernel (Gemma SWA/global).
// ===========================================================================
//
// A faithful TEMPLATED generalization of full_attention_fa2_prefill_tile (hd128)
// to head_dim 256 (Bc=64) and 512 (Bc=32). IDENTICAL structure + numerics:
// unscaled-Q in SLM + post-dot 1/sqrt(HD) scale (so Gemma's pre-scaled-Q by
// sqrt(HD) → net scale 1.0, matching naive/v2), per-lane COMPLETE KQ dots (no
// per-key subgroup reduce — the v2 floor the tile kernel lifts), online softmax,
// per-lane VKQ FMA (lane L owns out-dims {L*DPL..}), half2-vectorized coalesced
// K/V SLM loads. KVPL=Bc/16 scores/lane, DPL=HD/16 out-dims/lane.
// SLM/WG (128 KB cap): Q(Br·HD·2)+K(Bc·HD·2)+V+P(Br·Bc·4):
//   hd256/Bc64 = 8+32+32+4 = 76 KB ; hd512/Bc32 = 16+32+32+2 = 82 KB.  Both fit.
// The hd128 tile kernel is UNTOUCHED (crown/Coder gate path). SIMD only — no
// joint_matrix / ESIMD / block2d / lsc_load.
// SG (subgroup size) is a template param defaulting to the global SG_SIZE(16). At
// SG=32 the whole body retargets via the shadow below: every DPL=HD/SG, KVPL=Bc/SG,
// lane mapping and reduce_over_group folds 32 lanes — capturing llama's SIMD32
// issue-efficiency edge. Register-safe (hot path 16→9 live fp32/lane, no spill).
template <uint32_t HD, uint32_t Bc, bool REGDOT, uint32_t SG = SG_SIZE, bool SINK = false>
static sycl::event fa2_tile_wide_impl(sycl::queue& q,
                                      const sycl::half* q_in, const sycl::half* k_in,
                                      const sycl::half* v_in, sycl::half* k_cache,
                                      sycl::half* v_cache, sycl::half* y,
                                      uint32_t T, uint32_t start_pos,
                                      uint32_t n_q_heads, uint32_t n_kv_heads,
                                      uint32_t max_ctx, uint32_t window,
                                      const std::vector<sycl::event>& deps,
                                      const float* sinks = nullptr) {
    if (T == 0) return {};   // guard T-1 underflow (dispatcher also guards; defensive)
    constexpr uint32_t SG_SIZE  = SG;            // shadow the global; whole body retargets
    constexpr uint32_t Br       = 16;
    constexpr uint32_t WG_ITEMS = Br * SG_SIZE;  // 256 @SG16 (identical), 512 @SG32
    const uint32_t gqa        = n_q_heads / n_kv_heads;
    const uint32_t ctx_len    = start_pos + T;
    const uint32_t n_q_blocks = (T + Br - 1) / Br;

    // 1. Append (k_in, v_in) — identical to the hd128 tile kernel.
    auto append_evt = ie::ps(q, "fa2_tilew_append", [&](sycl::handler& h) {
        h.depends_on(deps);
        const uint32_t total = T * n_kv_heads * HD;
        constexpr uint32_t WG = 256;
        const uint32_t global = ((total + WG - 1) / WG) * WG;
        h.parallel_for(sycl::nd_range<1>(global, WG), [=](sycl::nd_item<1> it) {
            const uint32_t idx = uint32_t(it.get_global_id(0));
            if (idx >= total) return;
            const uint32_t d  = idx % HD;
            const uint32_t kv = (idx / HD) % n_kv_heads;
            const uint32_t t  = idx / (n_kv_heads * HD);
            const uint64_t in_off  = (uint64_t(t) * n_kv_heads + kv) * HD + d;
            const uint64_t out_off = (uint64_t(kv) * max_ctx + (start_pos + t)) * HD + d;
            k_cache[out_off] = k_in[in_off];
            v_cache[out_off] = v_in[in_off];
        });
    });

    constexpr uint32_t KVPL = Bc / SG_SIZE;   // scores per lane
    constexpr uint32_t DPL  = HD / SG_SIZE;   // out-dims per lane
    constexpr uint32_t CPE  = 4;              // regtile chunk width (half2)
    static_assert(WG_ITEMS / SG_SIZE == Br, "one subgroup per query row");
    static_assert((HD / 2) % CPE == 0, "CPE must divide HD/2");
    static_assert(Bc % SG_SIZE == 0 && Bc / SG_SIZE >= 1, "Bc must tile the subgroup (KVPL>=1)");

    return ie::ps(q, "fa2_tilew_compute", [&](sycl::handler& h) {
        h.depends_on(append_evt);
        sycl::local_accessor<sycl::half, 1> Q_slm(Br * HD, h);
        sycl::local_accessor<sycl::half, 1> K_slm(Bc * HD, h);
        sycl::local_accessor<sycl::half, 1> V_slm(Bc * HD, h);
        sycl::local_accessor<float, 1>      P_slm(Br * Bc, h);
        const float scale = 1.0f / sycl::sqrt(float(HD));

        h.parallel_for(sycl::nd_range<2>({uint64_t(n_q_heads) * n_q_blocks, WG_ITEMS},
                                          {1, WG_ITEMS}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t wg0     = uint32_t(it.get_group(0));
            const uint32_t q_head  = wg0 / n_q_blocks;
            const uint32_t q_block = wg0 % n_q_blocks;
            const uint32_t kv_head = q_head / gqa;
            const uint32_t lid     = uint32_t(it.get_local_id(1));
            const uint32_t sg_id   = lid / SG_SIZE;
            const uint32_t lane    = lid % SG_SIZE;
            const uint32_t q0      = q_block * Br;
            auto sg = it.get_sub_group();
            const uint32_t r      = sg_id;
            const uint32_t tl     = q0 + r;
            const uint32_t abs_q  = start_pos + tl;
            const bool     active = (tl < T);

            const sycl::half2* k_cache2 = reinterpret_cast<const sycl::half2*>(k_cache);
            const sycl::half2* v_cache2 = reinterpret_cast<const sycl::half2*>(v_cache);
            sycl::half2* K_slm2 = reinterpret_cast<sycl::half2*>(
                K_slm.get_multi_ptr<sycl::access::decorated::no>().get());
            sycl::half2* V_slm2 = reinterpret_cast<sycl::half2*>(
                V_slm.get_multi_ptr<sycl::access::decorated::no>().get());

            // Q tile [Br×HD] into SLM ONCE, UNSCALED (zero-pad OOB).
            for (uint32_t i = lid; i < Br * HD; i += WG_ITEMS) {
                const uint32_t rr = i / HD;
                const uint32_t d  = i % HD;
                const uint32_t t2 = q0 + rr;
                Q_slm[i] = (t2 < T)
                    ? q_in[(uint64_t(t2) * n_q_heads + q_head) * HD + d]
                    : sycl::half(0);
            }
            sycl::group_barrier(it.get_group());

            float m = -std::numeric_limits<float>::infinity();
            float l = 0.f;
            float vkq[DPL];
            #pragma unroll
            for (uint32_t d = 0; d < DPL; ++d) vkq[d] = 0.f;

            const uint32_t wg_last_row = sycl::min(q0 + Br - 1, T - 1);
            const uint32_t abs_q_last  = start_pos + wg_last_row;
            // Sliding-window lower bound (window>0). A key k is in-window for query q
            // iff k > q-window. Across this WG's queries [q0..], the loosest bound is
            // the FIRST query's: wstart = abs_q_first - window + 1. Tiles entirely
            // below wstart are skipped (all keys out-of-window for every WG query);
            // the per-key window mask below handles the boundary tiles. window==0 =
            // full causal (Gemma global layers, Coder/crown).
            const uint32_t abs_q_first = start_pos + q0;
            const uint32_t wstart = (window && abs_q_first + 1 > window)
                                    ? (abs_q_first + 1 - window) : 0;
            const uint32_t n_tiles = (ctx_len + Bc - 1) / Bc;
            for (uint32_t kt = 0; kt < n_tiles; ++kt) {
                const uint32_t k0   = kt * Bc;
                const uint32_t klen = sycl::min(Bc, ctx_len - k0);
                if (k0 > abs_q_last) break;             // future for the whole WG
                if (window && k0 + Bc <= wstart) continue;  // fully before the window

                constexpr uint32_t TILE_H2 = (Bc * HD) / 2;
                for (uint32_t i = lid; i < TILE_H2; i += WG_ITEMS) {
                    const uint32_t kl = i / (HD / 2);
                    const uint32_t d2 = i % (HD / 2);
                    if (kl < klen) {
                        const uint64_t off2 =
                            ((uint64_t(kv_head) * max_ctx + k0 + kl) * HD) / 2 + d2;
                        K_slm2[i] = k_cache2[off2];
                        V_slm2[i] = v_cache2[off2];
                    } else {
                        K_slm2[i] = sycl::half2(0, 0);
                        V_slm2[i] = sycl::half2(0, 0);
                    }
                }
                sycl::group_barrier(it.get_group());

                float kq[KVPL];
                #pragma unroll
                for (uint32_t kk = 0; kk < KVPL; ++kk) kq[kk] = 0.f;
                if (active) {
                    const sycl::half2* qrow2 =
                        reinterpret_cast<const sycl::half2*>(&Q_slm[r * HD]);
                    if constexpr (REGDOT) {
                        // Bounded register-staging dot (the _regtile lever): each of
                        // the KVPL dots keeps TWO parity fp32 accumulators (even/odd
                        // CPE-chunk) → 2·KVPL independent MAD chains, breaking the
                        // single-acc carried dependency that pins the long hd512 dot
                        // (HD/2=256 serial FMAs) at the ~15% vector-ALU floor. Live
                        // staging = CPE·(1+KVPL) half2, bounded (no spill).
                        const sycl::half2* krow2[KVPL];
                        #pragma unroll
                        for (uint32_t kk = 0; kk < KVPL; ++kk) {
                            const uint32_t c = lane + kk * SG_SIZE;
                            krow2[kk] = reinterpret_cast<const sycl::half2*>(&K_slm[c * HD]);
                        }
                        float acc0[KVPL], acc1[KVPL];
                        #pragma unroll
                        for (uint32_t kk = 0; kk < KVPL; ++kk) { acc0[kk] = 0.f; acc1[kk] = 0.f; }
                        constexpr uint32_t NCHUNK = (HD / 2) / CPE;
                        #pragma unroll
                        for (uint32_t ch = 0; ch < NCHUNK; ++ch) {
                            const uint32_t base = ch * CPE;
                            sycl::half2 Q_k[CPE];
                            #pragma unroll
                            for (uint32_t e = 0; e < CPE; ++e) Q_k[e] = qrow2[base + e];
                            float* acc = (ch & 1u) ? acc1 : acc0;
                            #pragma unroll
                            for (uint32_t kk = 0; kk < KVPL; ++kk) {
                                sycl::half2 K_k[CPE];
                                #pragma unroll
                                for (uint32_t e = 0; e < CPE; ++e) K_k[e] = krow2[kk][base + e];
                                float a = 0.f;
                                #pragma unroll
                                for (uint32_t e = 0; e < CPE; ++e)
                                    a += float(Q_k[e].x()) * float(K_k[e].x())
                                       + float(Q_k[e].y()) * float(K_k[e].y());
                                acc[kk] += a;
                            }
                        }
                        #pragma unroll
                        for (uint32_t kk = 0; kk < KVPL; ++kk)
                            kq[kk] = (acc0[kk] + acc1[kk]) * scale;
                    } else {
                        #pragma unroll
                        for (uint32_t kk = 0; kk < KVPL; ++kk) {
                            const uint32_t c = lane + kk * SG_SIZE;
                            const sycl::half2* krow2 =
                                reinterpret_cast<const sycl::half2*>(&K_slm[c * HD]);
                            float acc = 0.f;
                            #pragma unroll
                            for (uint32_t d2 = 0; d2 < HD / 2; ++d2) {
                                const sycl::half2 qd = qrow2[d2];
                                const sycl::half2 kd = krow2[d2];
                                acc += float(qd.x()) * float(kd.x())
                                     + float(qd.y()) * float(kd.y());
                            }
                            kq[kk] = acc * scale;
                        }
                    }
                }

                float tile_max = -std::numeric_limits<float>::infinity();
                #pragma unroll
                for (uint32_t kk = 0; kk < KVPL; ++kk) {
                    const uint32_t c     = lane + kk * SG_SIZE;
                    const uint32_t abs_k = k0 + c;
                    // causal (abs_k<=abs_q) AND sliding-window (abs_k>abs_q-window).
                    const bool keep = active && (c < klen) && (abs_k <= abs_q)
                                      && (window == 0 || abs_k + window > abs_q);
                    if (!keep) kq[kk] = -std::numeric_limits<float>::infinity();
                    tile_max = sycl::fmax(tile_max, kq[kk]);
                }
                tile_max = sycl::reduce_over_group(sg, tile_max, sycl::maximum<float>());

                const float m_new = sycl::fmax(m, tile_max);
                const float alpha = (m == -std::numeric_limits<float>::infinity())
                                    ? 0.f : sycl::native::exp(m - m_new);

                float row_l_add = 0.f;
                #pragma unroll
                for (uint32_t kk = 0; kk < KVPL; ++kk) {
                    const uint32_t c = lane + kk * SG_SIZE;
                    const float e = (kq[kk] == -std::numeric_limits<float>::infinity())
                                    ? 0.f : sycl::native::exp(kq[kk] - m_new);
                    if (active) P_slm[r * Bc + c] = e;
                    row_l_add += e;
                }
                row_l_add = sycl::reduce_over_group(sg, row_l_add, sycl::plus<float>());
                l = l * alpha + row_l_add;
                m = m_new;
                #pragma unroll
                for (uint32_t d = 0; d < DPL; ++d) vkq[d] *= alpha;

                sycl::group_barrier(it.get_group());

                if (active && k0 <= abs_q) {
                    const uint32_t kmax = sycl::min(klen, abs_q - k0 + 1);
                    for (uint32_t c = 0; c < kmax; ++c) {
                        const float p = float(P_slm[r * Bc + c]);
                        const uint32_t vbase = c * HD + lane * DPL;
                        #pragma unroll
                        for (uint32_t d = 0; d < DPL; ++d)
                            vkq[d] += p * float(V_slm[vbase + d]);
                    }
                }
                sycl::group_barrier(it.get_group());
            }

            if (active) {
                // gpt-oss per-head softmax SINK (virtual key, denominator only) folds
                // ONCE at row end. if constexpr(SINK=false) → byte-identical to the base
                // tile (crown/gemma/Coder path unchanged — verified by the crown gate).
                float scale_o = 1.0f, l_final = l;
                if constexpr (SINK) {
                    const float sink = sinks[q_head];
                    const float m_prime = sycl::fmax(m, sink);
                    scale_o = sycl::native::exp(m - m_prime);
                    l_final = l * scale_o + sycl::native::exp(sink - m_prime);
                }
                const float inv_l = (l_final > 0.f) ? (scale_o / l_final) : 0.f;
                const uint64_t yb = (uint64_t(tl) * n_q_heads + q_head) * HD + lane * DPL;
                #pragma unroll
                for (uint32_t d = 0; d < DPL; ++d)
                    y[yb + d] = sycl::half(vkq[d] * inv_l);
            }
        });
    });
}

// full_attention_gptoss_prefill_tile — gpt-oss PREFILL (T>1) wide-tile attention:
// fa2_tile_wide_impl at HD=64 with the per-head SINK (SINK=true).  window param
// handles both the SWA (even, window=128) and full (odd, window=0) layers — same
// kernel for all layers (the wide tile already supports the Gemma window).  Replaces
// the naive O(T·ctx) full_attention_gptoss for prefill.  REGDOT on (bounded reg dot).
sycl::event full_attention_gptoss_prefill_tile(sycl::queue& q,
                                               const sycl::half* q_in, const sycl::half* k_in,
                                               const sycl::half* v_in, sycl::half* k_cache,
                                               sycl::half* v_cache, sycl::half* y,
                                               uint32_t T, uint32_t start_pos,
                                               uint32_t n_q_heads, uint32_t n_kv_heads,
                                               uint32_t max_ctx, uint32_t window,
                                               const float* sinks,
                                               const std::vector<sycl::event>& deps) {
    if (T == 0) return {};
    // Bc=64 (HD=64 → SLM Q2+K8+V8+P4 = 22 KB → fits; occupancy A/B vs Bc=32 later).
    return fa2_tile_wide_impl<64, 64, /*REGDOT=*/true, /*SG=*/SG_SIZE, /*SINK=*/true>(
        q, q_in, k_in, v_in, k_cache, v_cache, y, T, start_pos,
        n_q_heads, n_kv_heads, max_ctx, window, deps, sinks);
}

// Dispatcher: head_dim 256/512 → wide tile; else fall through to v2.
sycl::event full_attention_fa2_prefill_tile_gemma(sycl::queue& q,
                                                  const sycl::half* q_in,
                                                  const sycl::half* k_in,
                                                  const sycl::half* v_in,
                                                  sycl::half* k_cache,
                                                  sycl::half* v_cache,
                                                  sycl::half* y,
                                                  uint32_t T, uint32_t start_pos,
                                                  uint32_t n_q_heads, uint32_t n_kv_heads,
                                                  uint32_t head_dim, uint32_t max_ctx,
                                                  uint32_t window,
                                                  const std::vector<sycl::event>& deps) {
    if (T == 0) return {};
    // REGDOT (bounded register-staged KQ dot) — default ON (the long hd512 dot
    // is carried-dependency bound; regtile breaks it). Opt-out IE_GEMMA4_NO_TILE_REGDOT.
    static const bool regdot = std::getenv("IE_GEMMA4_NO_TILE_REGDOT") == nullptr;
    // SMALLBC — DEFAULT ON (huge win: 8K +54%, 16K +59%). The large Bc (64 @hd256 /
    // 32 @hd512) puts the K+V+Q+P SLM at 76/82 KB → only 1 WG/Xe-core (128 KB cap) =
    // no latency hiding (occupancy-starved). Halving Bc (32/16) → ~42/49 KB → 2-3
    // WG/core → +54-59%. Beats llama @8K (873 vs 758). Opt-out IE_GEMMA4_NO_TILE_SMALLBC.
    static const bool smallbc = std::getenv("IE_GEMMA4_NO_TILE_SMALLBC") == nullptr;
    // SG32 (subgroup size 32): captures llama's SIMD32 issue-efficiency edge on the
    // wide-tile — hd128 (Coder/dense) AND hd256 (Gemma SWA layers = 96% of gemma 16K
    // attention; crown/27B/80B long-ctx tile). Register-safe (design-vetted: hot path
    // 16→9 live fp32/lane at hd128, 13 at hd256, no spill; IGC accepts SIMD32 on
    // BMG-G31). DEFAULT-OFF env gate; NOT bit-exact (the 32-lane reduce_over_group
    // reassociates fp → argmax-identical, Coder PPL identical to 4 dp) → opt-in.
    // hd128/hd256 only (hd512's smallbc Bc=16 → KVPL=0). +5-7% attention-heavy long-ctx.
    static const bool sg32 = std::getenv("IE_FA2_TILE_SG32") != nullptr;
    if (head_dim == 128) {   // Coder/dense reuse (env-gated callers); occupancy A/B
        // TINYBC (hd128 Bc=16): higher occupancy than smallbc's Bc=32 (SLM ~13 KB →
        // ~9 WG/Xe-core vs ~5). A/B for the Coder 16K attention wall (93.5% of 16K
        // prefill, full-causal). Env IE_TILE_HD128_BC16.
        static const bool tinybc = std::getenv("IE_TILE_HD128_BC16") != nullptr;
        if (tinybc && !sg32)   // tinybc Bc=16 incompatible with SG32 (KVPL=0)
            return regdot ? fa2_tile_wide_impl<128, 16, true >(q, q_in, k_in, v_in, k_cache, v_cache, y, T, start_pos, n_q_heads, n_kv_heads, max_ctx, window, deps)
                          : fa2_tile_wide_impl<128, 16, false>(q, q_in, k_in, v_in, k_cache, v_cache, y, T, start_pos, n_q_heads, n_kv_heads, max_ctx, window, deps);
        if (smallbc)
            return sg32
                ? (regdot ? fa2_tile_wide_impl<128, 32, true , 32>(q, q_in, k_in, v_in, k_cache, v_cache, y, T, start_pos, n_q_heads, n_kv_heads, max_ctx, window, deps)
                          : fa2_tile_wide_impl<128, 32, false, 32>(q, q_in, k_in, v_in, k_cache, v_cache, y, T, start_pos, n_q_heads, n_kv_heads, max_ctx, window, deps))
                : (regdot ? fa2_tile_wide_impl<128, 32, true >(q, q_in, k_in, v_in, k_cache, v_cache, y, T, start_pos, n_q_heads, n_kv_heads, max_ctx, window, deps)
                          : fa2_tile_wide_impl<128, 32, false>(q, q_in, k_in, v_in, k_cache, v_cache, y, T, start_pos, n_q_heads, n_kv_heads, max_ctx, window, deps));
        return sg32
            ? (regdot ? fa2_tile_wide_impl<128, 64, true , 32>(q, q_in, k_in, v_in, k_cache, v_cache, y, T, start_pos, n_q_heads, n_kv_heads, max_ctx, window, deps)
                      : fa2_tile_wide_impl<128, 64, false, 32>(q, q_in, k_in, v_in, k_cache, v_cache, y, T, start_pos, n_q_heads, n_kv_heads, max_ctx, window, deps))
            : (regdot ? fa2_tile_wide_impl<128, 64, true >(q, q_in, k_in, v_in, k_cache, v_cache, y, T, start_pos, n_q_heads, n_kv_heads, max_ctx, window, deps)
                      : fa2_tile_wide_impl<128, 64, false>(q, q_in, k_in, v_in, k_cache, v_cache, y, T, start_pos, n_q_heads, n_kv_heads, max_ctx, window, deps));
    }
    if (head_dim == 256) {
        // SG32 at hd256 — register-safe (Bc=32 → KVPL=1, DPL=8 = 13 fp32/lane). Targets
        // Gemma's SWA layers (hd256, ~96% of gemma 16K attention) + crown/27B/80B long-ctx.
        if (smallbc)
            return sg32
                ? (regdot ? fa2_tile_wide_impl<256, 32, true , 32>(q, q_in, k_in, v_in, k_cache, v_cache, y, T, start_pos, n_q_heads, n_kv_heads, max_ctx, window, deps)
                          : fa2_tile_wide_impl<256, 32, false, 32>(q, q_in, k_in, v_in, k_cache, v_cache, y, T, start_pos, n_q_heads, n_kv_heads, max_ctx, window, deps))
                : (regdot ? fa2_tile_wide_impl<256, 32, true >(q, q_in, k_in, v_in, k_cache, v_cache, y, T, start_pos, n_q_heads, n_kv_heads, max_ctx, window, deps)
                          : fa2_tile_wide_impl<256, 32, false>(q, q_in, k_in, v_in, k_cache, v_cache, y, T, start_pos, n_q_heads, n_kv_heads, max_ctx, window, deps));
        return sg32
            ? (regdot ? fa2_tile_wide_impl<256, 64, true , 32>(q, q_in, k_in, v_in, k_cache, v_cache, y, T, start_pos, n_q_heads, n_kv_heads, max_ctx, window, deps)
                      : fa2_tile_wide_impl<256, 64, false, 32>(q, q_in, k_in, v_in, k_cache, v_cache, y, T, start_pos, n_q_heads, n_kv_heads, max_ctx, window, deps))
            : (regdot ? fa2_tile_wide_impl<256, 64, true >(q, q_in, k_in, v_in, k_cache, v_cache, y, T, start_pos, n_q_heads, n_kv_heads, max_ctx, window, deps)
                      : fa2_tile_wide_impl<256, 64, false>(q, q_in, k_in, v_in, k_cache, v_cache, y, T, start_pos, n_q_heads, n_kv_heads, max_ctx, window, deps));
    }
    if (head_dim == 512) {
        // SG32 forces Bc=32 (the default smallbc Bc=16 → KVPL=16/32=0). Register-safe:
        // DPL=16/KVPL=1 = 21 fp32/lane (< the ~28-32 BMG spill cliff, per the design).
        // hd512 = Gemma's GLOBAL layers, which do FULL-causal O(T²) attention → they
        // DOMINATE gemma's long-ctx attention cost (the windowed SWA hd256 is cheaper),
        // so this is the BIGGER gemma-16K SG32 lever, completing all-attention-layers.
        if (sg32)
            return regdot ? fa2_tile_wide_impl<512, 32, true , 32>(q, q_in, k_in, v_in, k_cache, v_cache, y, T, start_pos, n_q_heads, n_kv_heads, max_ctx, window, deps)
                          : fa2_tile_wide_impl<512, 32, false, 32>(q, q_in, k_in, v_in, k_cache, v_cache, y, T, start_pos, n_q_heads, n_kv_heads, max_ctx, window, deps);
        if (smallbc)
            return regdot ? fa2_tile_wide_impl<512, 16, true >(q, q_in, k_in, v_in, k_cache, v_cache, y, T, start_pos, n_q_heads, n_kv_heads, max_ctx, window, deps)
                          : fa2_tile_wide_impl<512, 16, false>(q, q_in, k_in, v_in, k_cache, v_cache, y, T, start_pos, n_q_heads, n_kv_heads, max_ctx, window, deps);
        return regdot ? fa2_tile_wide_impl<512, 32, true >(q, q_in, k_in, v_in, k_cache, v_cache, y, T, start_pos, n_q_heads, n_kv_heads, max_ctx, window, deps)
                      : fa2_tile_wide_impl<512, 32, false>(q, q_in, k_in, v_in, k_cache, v_cache, y, T, start_pos, n_q_heads, n_kv_heads, max_ctx, window, deps);
    }
    return full_attention_fa2_prefill_v2(q, q_in, k_in, v_in, k_cache, v_cache, y,
                                         T, start_pos, n_q_heads, n_kv_heads,
                                         head_dim, max_ctx, deps);
}

// ===========================================================================
// full_attention_fa2_prefill_tile_regtile — bounded register-staging KQ dot.
// ===========================================================================
//
// IDENTICAL to full_attention_fa2_prefill_tile in EVERY respect (geometry,
// SLM layout, online softmax, P·V, y-write, numerics: unscaled-Q + post-dot
// scale, fp32 accumulate) EXCEPT the KQ inner dot, which is restructured to
// llama-SYCL's bounded register-staging (fattn-tile.hpp flash_attn_tile_iter_KQ).
//
// THE BOTTLENECK it fixes:  the default tile KQ dot accumulates the full
// head_dim into a SINGLE fp32 `acc` over HD/2=64 serial iterations:
//     for (d2=0..63) acc += qx*kx + qy*ky;
// That carried `acc` dependency (load-to-use + FMA latency on the critical
// path of every iteration) pins the dot at ~15% of vector-ALU peak.
//
// THE LEVER (bounded, vs the failed unbounded 4-acc spill):
//   * Walk the head_dim in BOUNDED chunks of CPE half2 (CPE=4 → 8 elems).
//     Per chunk: stage CPE half2 of the Q-row (Q_k[CPE]) and CPE half2 of
//     EACH of the KVPL=4 K-rows (K_k[KVPL][CPE]) into tiny register arrays,
//     then MAD. Live staging footprint = CPE*(1+KVPL) = 20 half2 — bounded
//     and constant; it does NOT grow with head_dim.
//   * ILP without exploding accumulators: each of the KVPL=4 dots keeps TWO
//     partial fp32 accumulators (acc0/acc1, even/odd chunk parity), combined
//     ONCE at the end. So 4 dots × 2 = 8 live fp32 accumulators (bounded),
//     each accumulator's serial chain is only HALF as deep, and the 4 dots ×
//     2 parities give 8 independent MAD chains overlapping per chunk. This is
//     the difference vs the user's 0.54× attempt, which `#pragma unroll`-ed
//     the WHOLE KVPL kk-loop over the full dot → ~16 live accs + 32 half2
//     temps → register SPILL. Here only ONE bounded CPE chunk is live at a
//     time; the accumulator count is fixed at 2*KVPL=8 regardless of HD.
//
// Everything else is byte-for-byte the default kernel. Output matches the
// default tile kernel within fp16 tol (same fp32 accumulate; only the
// summation grouping of the dot changes — associativity-level fp32 reorder).
//
// Gated A/B via IE_FA2_TILE_REGTILE (default OFF → default tile unchanged).
// SIMD only — no joint_matrix / ESIMD / block2d / lsc_load. Falls through to
// v2 for head_dim != 128.
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
                                                    const std::vector<sycl::event>& deps) {
    if (T == 0) return {};
    if (head_dim != 128)
        return full_attention_fa2_prefill_v2(q, q_in, k_in, v_in, k_cache, v_cache,
                                             y, T, start_pos, n_q_heads, n_kv_heads,
                                             head_dim, max_ctx, deps);

    constexpr uint32_t HD       = 128;
    constexpr uint32_t Br       = 16;
    constexpr uint32_t Bc       = 64;
    constexpr uint32_t WG_ITEMS = 256;
    const uint32_t gqa        = n_q_heads / n_kv_heads;
    const uint32_t ctx_len    = start_pos + T;
    const uint32_t n_q_blocks = (T + Br - 1) / Br;

    // 1. Append (k_in, v_in) — identical to the default tile kernel.
    auto append_evt = ie::ps(q, "fa2_tile_rt_append", [&](sycl::handler& h) {
        h.depends_on(deps);
        const uint32_t total = T * n_kv_heads * head_dim;
        constexpr uint32_t WG = 256;
        const uint32_t global = ((total + WG - 1) / WG) * WG;
        h.parallel_for(sycl::nd_range<1>(global, WG),
                       [=](sycl::nd_item<1> it) {
            const uint32_t idx = uint32_t(it.get_global_id(0));
            if (idx >= total) return;
            const uint32_t d  = idx % head_dim;
            const uint32_t kv = (idx / head_dim) % n_kv_heads;
            const uint32_t t  = idx / (n_kv_heads * head_dim);
            const uint64_t in_off  = (uint64_t(t) * n_kv_heads + kv) * head_dim + d;
            const uint64_t out_off = (uint64_t(kv) * max_ctx + (start_pos + t)) * head_dim + d;
            k_cache[out_off] = k_in[in_off];
            v_cache[out_off] = v_in[in_off];
        });
    });

    constexpr uint32_t NSG  = WG_ITEMS / SG_SIZE;   // 16 subgroups
    constexpr uint32_t KVPL = Bc / SG_SIZE;         // 4 KV positions per lane
    constexpr uint32_t DPL  = HD / SG_SIZE;         // 8 out-dims per lane
    // Bounded register-staging chunk width (in half2). CPE=4 → 8 head_dim
    // elements per chunk; HD/2 = 64 half2 / CPE = 16 chunks. Live staging =
    // CPE*(1+KVPL) = 20 half2. Must divide HD/2 (=64) evenly.
    constexpr uint32_t CPE = 4;
    static_assert((HD / 2) % CPE == 0, "CPE must divide HD/2");
    static_assert(NSG == Br, "one subgroup per query row (cpw==1, np==1)");

    return ie::ps(q, "fa2_tile_rt_compute", [&](sycl::handler& h) {
        h.depends_on(append_evt);
        sycl::local_accessor<sycl::half, 1> Q_slm(Br * HD, h);
        sycl::local_accessor<sycl::half, 1> K_slm(Bc * HD, h);
        sycl::local_accessor<sycl::half, 1> V_slm(Bc * HD, h);
        sycl::local_accessor<float, 1> P_slm(Br * Bc, h);
        const float scale = 1.0f / sycl::sqrt(float(HD));

        h.parallel_for(sycl::nd_range<2>({uint64_t(n_q_heads) * n_q_blocks, WG_ITEMS},
                                          {1, WG_ITEMS}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t wg0     = uint32_t(it.get_group(0));
            const uint32_t q_head  = wg0 / n_q_blocks;
            const uint32_t q_block = wg0 % n_q_blocks;
            const uint32_t kv_head = q_head / gqa;
            const uint32_t lid     = uint32_t(it.get_local_id(1));
            const uint32_t sg_id   = lid / SG_SIZE;
            const uint32_t lane    = lid % SG_SIZE;
            const uint32_t q0      = q_block * Br;
            auto sg = it.get_sub_group();

            const uint32_t r      = sg_id;
            const uint32_t tl     = q0 + r;
            const uint32_t abs_q  = start_pos + tl;
            const bool     active = (tl < T);

            const sycl::half2* k_cache2 = reinterpret_cast<const sycl::half2*>(k_cache);
            const sycl::half2* v_cache2 = reinterpret_cast<const sycl::half2*>(v_cache);
            sycl::half2* K_slm2 = reinterpret_cast<sycl::half2*>(
                K_slm.get_multi_ptr<sycl::access::decorated::no>().get());
            sycl::half2* V_slm2 = reinterpret_cast<sycl::half2*>(
                V_slm.get_multi_ptr<sycl::access::decorated::no>().get());

            // --- Load this WG's Q tile [Br×HD] into SLM ONCE, UNSCALED. ---
            for (uint32_t i = lid; i < Br * HD; i += WG_ITEMS) {
                const uint32_t rr = i / HD;
                const uint32_t d  = i % HD;
                const uint32_t t2 = q0 + rr;
                Q_slm[i] = (t2 < T)
                    ? q_in[(uint64_t(t2) * n_q_heads + q_head) * HD + d]
                    : sycl::half(0);
            }
            sycl::group_barrier(it.get_group());

            float m = -std::numeric_limits<float>::infinity();
            float l = 0.f;
            float vkq[DPL];
            #pragma unroll
            for (uint32_t d = 0; d < DPL; ++d) vkq[d] = 0.f;

            const uint32_t wg_last_row = sycl::min(q0 + Br - 1, T - 1);
            const uint32_t abs_q_last  = start_pos + wg_last_row;

            const uint32_t n_tiles = (ctx_len + Bc - 1) / Bc;
            for (uint32_t kt = 0; kt < n_tiles; ++kt) {
                const uint32_t k0   = kt * Bc;
                const uint32_t klen = sycl::min(Bc, ctx_len - k0);
                if (k0 > abs_q_last) break;

                constexpr uint32_t TILE_H2 = (Bc * HD) / 2;
                for (uint32_t i = lid; i < TILE_H2; i += WG_ITEMS) {
                    const uint32_t kl = i / (HD / 2);
                    const uint32_t d2 = i % (HD / 2);
                    if (kl < klen) {
                        const uint64_t off2 =
                            ((uint64_t(kv_head) * max_ctx + k0 + kl) * HD) / 2 + d2;
                        K_slm2[i] = k_cache2[off2];
                        V_slm2[i] = v_cache2[off2];
                    } else {
                        K_slm2[i] = sycl::half2(0, 0);
                        V_slm2[i] = sycl::half2(0, 0);
                    }
                }
                sycl::group_barrier(it.get_group());

                // --- iter_KQ: BOUNDED register-staging KQ dot. ---
                // KVPL=4 dots, each split into TWO parity accumulators
                // (acc0=even chunks, acc1=odd chunks) → 8 fixed fp32 accs.
                float kq[KVPL];
                #pragma unroll
                for (uint32_t kk = 0; kk < KVPL; ++kk) kq[kk] = 0.f;

                if (active) {
                    const sycl::half2* qrow2 =
                        reinterpret_cast<const sycl::half2*>(&Q_slm[r * HD]);
                    // Per-dot K-row half2 base pointers (KVPL=4).
                    const sycl::half2* krow2[KVPL];
                    #pragma unroll
                    for (uint32_t kk = 0; kk < KVPL; ++kk) {
                        const uint32_t c = lane + kk * SG_SIZE;   // KV pos in tile
                        krow2[kk] = reinterpret_cast<const sycl::half2*>(&K_slm[c * HD]);
                    }
                    // Two parity partials per dot → 8 independent fp32 chains.
                    float acc0[KVPL], acc1[KVPL];
                    #pragma unroll
                    for (uint32_t kk = 0; kk < KVPL; ++kk) { acc0[kk] = 0.f; acc1[kk] = 0.f; }

                    // Walk head_dim in BOUNDED CPE-half2 chunks. Only Q_k +
                    // K_k (CPE*(1+KVPL) half2) are live at any time.
                    constexpr uint32_t NCHUNK = (HD / 2) / CPE;     // 16
                    #pragma unroll
                    for (uint32_t ch = 0; ch < NCHUNK; ++ch) {
                        const uint32_t base = ch * CPE;
                        // Stage CPE half2 of the Q-row.
                        sycl::half2 Q_k[CPE];
                        #pragma unroll
                        for (uint32_t e = 0; e < CPE; ++e) Q_k[e] = qrow2[base + e];
                        // Stage CPE half2 of each K-row, MAD into the parity acc.
                        // Accumulator selected by chunk parity → the two chains
                        // (ch even / ch odd) are independent across chunks.
                        float* acc = (ch & 1u) ? acc1 : acc0;
                        #pragma unroll
                        for (uint32_t kk = 0; kk < KVPL; ++kk) {
                            sycl::half2 K_k[CPE];
                            #pragma unroll
                            for (uint32_t e = 0; e < CPE; ++e) K_k[e] = krow2[kk][base + e];
                            float a = 0.f;
                            #pragma unroll
                            for (uint32_t e = 0; e < CPE; ++e)
                                a += float(Q_k[e].x()) * float(K_k[e].x())
                                   + float(Q_k[e].y()) * float(K_k[e].y());
                            acc[kk] += a;
                        }
                    }
                    #pragma unroll
                    for (uint32_t kk = 0; kk < KVPL; ++kk)
                        kq[kk] = (acc0[kk] + acc1[kk]) * scale;
                }

                // --- Causal mask + per-row max (identical to default). ---
                float tile_max = -std::numeric_limits<float>::infinity();
                #pragma unroll
                for (uint32_t kk = 0; kk < KVPL; ++kk) {
                    const uint32_t c     = lane + kk * SG_SIZE;
                    const uint32_t abs_k = k0 + c;
                    const bool keep = active && (c < klen) && (abs_k <= abs_q);
                    if (!keep) kq[kk] = -std::numeric_limits<float>::infinity();
                    tile_max = sycl::fmax(tile_max, kq[kk]);
                }
                tile_max = sycl::reduce_over_group(sg, tile_max, sycl::maximum<float>());

                const float m_new = sycl::fmax(m, tile_max);
                const float alpha = (m == -std::numeric_limits<float>::infinity())
                                    ? 0.f : sycl::native::exp(m - m_new);

                float row_l_add = 0.f;
                #pragma unroll
                for (uint32_t kk = 0; kk < KVPL; ++kk) {
                    const uint32_t c = lane + kk * SG_SIZE;
                    const float e = (kq[kk] == -std::numeric_limits<float>::infinity())
                                    ? 0.f : sycl::native::exp(kq[kk] - m_new);
                    if (active) P_slm[r * Bc + c] = e;
                    row_l_add += e;
                }
                row_l_add = sycl::reduce_over_group(sg, row_l_add, sycl::plus<float>());
                l = l * alpha + row_l_add;
                m = m_new;
                #pragma unroll
                for (uint32_t d = 0; d < DPL; ++d) vkq[d] *= alpha;

                sycl::group_barrier(it.get_group());

                if (active && k0 <= abs_q) {
                    const uint32_t kmax = sycl::min(klen, abs_q - k0 + 1);
                    for (uint32_t c = 0; c < kmax; ++c) {
                        const float p = float(P_slm[r * Bc + c]);
                        const uint32_t vbase = c * HD + lane * DPL;
                        #pragma unroll
                        for (uint32_t d = 0; d < DPL; ++d)
                            vkq[d] += p * float(V_slm[vbase + d]);
                    }
                }
                sycl::group_barrier(it.get_group());
            }

            if (active) {
                const float inv_l = (l > 0.f) ? (1.0f / l) : 0.f;
                const uint64_t yb = (uint64_t(tl) * n_q_heads + q_head) * HD + lane * DPL;
                #pragma unroll
                for (uint32_t d = 0; d < DPL; ++d)
                    y[yb + d] = sycl::half(vkq[d] * inv_l);
            }
        });
    });
}

// ===========================================================================
// full_attention_fa2_prefill_tile_gqa — GQA head-packing variant of the tile
// kernel (mirrors llama-SYCL's `ncols2` GQA head-packing, fattn-tile.hpp ~727).
// ===========================================================================
//
// PROBLEM the standard tile kernel has: it maps ONE q_head per WG, so the `gqa`
// q-heads that SHARE a kv_head are `gqa` separate WGs each independently
// streaming the SAME K/V tiles from HBM/L2 into SLM (up to gqa× redundant K/V
// reads + gqa× the SLM-stage barriers). On Coder (32 q / 4 kv → gqa=8) at 16K
// prefill where attention is 61% of wall and the redundant K/V traffic blows
// past L2, that redundancy hits HBM.
//
// FIX: pack G GQA-sibling q-heads of the SAME kv_head into ONE WG, sharing ONE
// cooperatively-loaded K/V SLM tile. The K/V tile is loaded ONCE per WG (keyed
// on kv_head) and reused by all G heads. Heads are numerically INDEPENDENT —
// they only share the input K/V tile; there is NO cross-head reduction. The
// per-row online softmax, fp32 accumulate, unscaled-Q + post-dot scale, and
// fp32 P are IDENTICAL to the standard tile kernel, so output is numerically
// identical (modulo nothing — same ops, same order per (head,row)).
//
// Geometry (hd=128, G=4, Br=4): WG = 256 = NSG=16 subgroups × SG=16 lanes.
//   The 16 subgroups are reinterpreted as a (head_in_group, q_row) grid:
//       sg_id (0..15) → head_in_group = sg_id / Br   (0..G-1 = 0..3)
//                       q_row         = sg_id % Br   (0..Br-1 = 0..3)
//   So each WG owns G=4 q-heads (all sharing one kv_head) × Br=4 query rows.
//   KVPL = Bc/SG = 4 (KV pos/lane), DPL = HD/SG = 8 (out-dims/lane) — unchanged.
//
// Grid: iterate (kv_head, head_group, q_block). head-groups per kv_head =
//   gqa/G. dim0 = n_kv_heads * (gqa/G) * n_q_blocks. From wg0:
//       q_block   = wg0 % n_q_blocks
//       hg_lin    = wg0 / n_q_blocks                  (0 .. n_kv_heads*(gqa/G)-1)
//       kv_head   = hg_lin / (gqa/G)
//       head_grp  = hg_lin % (gqa/G)                  (which G-block within kv_head)
//   The absolute q_head for a subgroup = kv_head*gqa + head_grp*G + head_in_group.
//   (q_head/gqa == kv_head holds, so all G heads in the WG share kv_head — the
//   K/V tile load uses kv_head directly.)
//
// SLM per WG (G=4, Br=4 → G*Br = NSG = 16, same as standard tile):
//   Q_slm  G*Br*HD*2  = 16*128*2 = 4096 B   (resident, UNSCALED, zero-pad OOB)
//   K_slm  Bc*HD*2    = 64*128*2 = 16384 B  (shared across all G heads)
//   V_slm  Bc*HD*2    = 16384 B
//   P_slm  G*Br*Bc*4  = 16*64*4  = 4096 B   (fp32 softmax weights)
//   total ≈ 40 KB  (<< 128 KB cap).  The K/V tile is loaded ONCE and amortized
//   over G heads — that is the entire win.
//
// Per-head y-write offset: a subgroup writes its row tl = q0 + q_row of its own
// q_head = kv_head*gqa + head_grp*G + head_in_group, dims lane*DPL..+DPL:
//   y[(tl * n_q_heads + q_head) * HD + lane*DPL + d].
//
// Requires gqa % G == 0 and head_dim==128; else falls through to v2 (caller may
// also just use the standard tile kernel — this is a gated A/B).
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
                                                const std::vector<sycl::event>& deps) {
    if (T == 0) return {};
    const uint32_t gqa_chk = (n_kv_heads == 0) ? 0 : n_q_heads / n_kv_heads;
    constexpr uint32_t G_PACK = 4;   // GQA-sibling heads packed per WG
    // Fall through to v2 when the GQA head-packing can't apply cleanly.
    if (head_dim != 128 || n_kv_heads == 0 ||
        (n_q_heads % n_kv_heads) != 0 || (gqa_chk % G_PACK) != 0)
        return full_attention_fa2_prefill_v2(q, q_in, k_in, v_in, k_cache, v_cache,
                                             y, T, start_pos, n_q_heads, n_kv_heads,
                                             head_dim, max_ctx, deps);

    constexpr uint32_t HD       = 128;
    constexpr uint32_t G        = G_PACK;        // heads per WG
    constexpr uint32_t Br       = 4;             // query rows per head per WG
    constexpr uint32_t Bc       = 64;            // KV positions per tile
    constexpr uint32_t WG_ITEMS = 256;           // 16 subgroups × 16 lanes
    static_assert(G * Br == WG_ITEMS / SG_SIZE,  // == NSG == 16
                  "G*Br must equal the number of subgroups (16)");
    const uint32_t gqa        = n_q_heads / n_kv_heads;
    const uint32_t groups_per_kv = gqa / G;      // head-groups per kv_head
    const uint32_t ctx_len    = start_pos + T;
    const uint32_t n_q_blocks = (T + Br - 1) / Br;

    // 1. Append (k_in, v_in) for these T tokens — identical to the tile kernel.
    auto append_evt = ie::ps(q, "fa2_tile_gqa_append", [&](sycl::handler& h) {
        h.depends_on(deps);
        const uint32_t total = T * n_kv_heads * head_dim;
        constexpr uint32_t WG = 256;
        const uint32_t global = ((total + WG - 1) / WG) * WG;
        h.parallel_for(sycl::nd_range<1>(global, WG),
                       [=](sycl::nd_item<1> it) {
            const uint32_t idx = uint32_t(it.get_global_id(0));
            if (idx >= total) return;
            const uint32_t d  = idx % head_dim;
            const uint32_t kv = (idx / head_dim) % n_kv_heads;
            const uint32_t t  = idx / (n_kv_heads * head_dim);
            const uint64_t in_off  = (uint64_t(t) * n_kv_heads + kv) * head_dim + d;
            const uint64_t out_off = (uint64_t(kv) * max_ctx + (start_pos + t)) * head_dim + d;
            k_cache[out_off] = k_in[in_off];
            v_cache[out_off] = v_in[in_off];
        });
    });

    // 2. Tile FA-2 compute. One WG per (kv_head, head_group, q_block); each WG
    //    owns G GQA-sibling heads (same kv_head) × Br query rows per head, and
    //    loads the kv_head's K/V tile ONCE for all G heads.
    constexpr uint32_t KVPL = Bc / SG_SIZE;         // 4 KV positions per lane
    constexpr uint32_t DPL  = HD / SG_SIZE;         // 8 out-dims per lane

    // dim0 = n_kv_heads * groups_per_kv * n_q_blocks
    const uint64_t grid0 = uint64_t(n_kv_heads) * groups_per_kv * n_q_blocks;

    return ie::ps(q, "fa2_tile_gqa_compute", [&](sycl::handler& h) {
        h.depends_on(append_evt);
        sycl::local_accessor<sycl::half, 1> Q_slm(G * Br * HD, h);  // [16×128] resident, UNSCALED
        sycl::local_accessor<sycl::half, 1> K_slm(Bc * HD, h);      // [64×128] shared by all G heads
        sycl::local_accessor<sycl::half, 1> V_slm(Bc * HD, h);      // [64×128] shared by all G heads
        sycl::local_accessor<float, 1> P_slm(G * Br * Bc, h);       // [16×64] fp32 softmax wts
        const float scale = 1.0f / sycl::sqrt(float(HD));

        h.parallel_for(sycl::nd_range<2>({grid0, WG_ITEMS}, {1, WG_ITEMS}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t wg0      = uint32_t(it.get_group(0));
            const uint32_t q_block  = wg0 % n_q_blocks;
            const uint32_t hg_lin   = wg0 / n_q_blocks;             // 0 .. n_kv_heads*groups_per_kv-1
            const uint32_t kv_head  = hg_lin / groups_per_kv;
            const uint32_t head_grp = hg_lin % groups_per_kv;       // which G-block within kv_head
            const uint32_t lid      = uint32_t(it.get_local_id(1)); // 0..255
            const uint32_t sg_id    = lid / SG_SIZE;                // 0..15
            const uint32_t lane     = lid % SG_SIZE;                // 0..15
            // Reinterpret the 16 subgroups as (head_in_group, q_row).
            const uint32_t hig      = sg_id / Br;                   // 0..G-1
            const uint32_t r        = sg_id % Br;                   // 0..Br-1 (query row within head)
            const uint32_t q_head   = kv_head * gqa + head_grp * G + hig;
            const uint32_t q0       = q_block * Br;                 // first query row (local) for this head
            auto sg = it.get_sub_group();

            // This subgroup owns query row r of head q_head.
            const uint32_t tl     = q0 + r;
            const uint32_t abs_q  = start_pos + tl;
            const bool     active = (tl < T);

            const sycl::half2* k_cache2 = reinterpret_cast<const sycl::half2*>(k_cache);
            const sycl::half2* v_cache2 = reinterpret_cast<const sycl::half2*>(v_cache);
            sycl::half2* K_slm2 = reinterpret_cast<sycl::half2*>(
                K_slm.get_multi_ptr<sycl::access::decorated::no>().get());
            sycl::half2* V_slm2 = reinterpret_cast<sycl::half2*>(
                V_slm.get_multi_ptr<sycl::access::decorated::no>().get());

            // --- Load this WG's Q tile [G×Br×HD] into SLM ONCE, UNSCALED. ---
            // Layout: Q_slm[(hig*Br + rr) * HD + d] for head index hig, row rr.
            // The flat index maps to its (hig, rr) so each of the G heads reads
            // its OWN q_in head. zero-pad OOB rows.
            for (uint32_t i = lid; i < G * Br * HD; i += WG_ITEMS) {
                const uint32_t slot = i / HD;          // 0..G*Br-1 = (hig*Br + rr)
                const uint32_t d    = i % HD;
                const uint32_t li_hig = slot / Br;     // head index within group
                const uint32_t li_rr  = slot % Br;     // row within head
                const uint32_t t2     = q0 + li_rr;
                const uint32_t qh      = kv_head * gqa + head_grp * G + li_hig;
                Q_slm[i] = (t2 < T)
                    ? q_in[(uint64_t(t2) * n_q_heads + qh) * HD + d]
                    : sycl::half(0);
            }
            sycl::group_barrier(it.get_group());

            // --- Per-subgroup online-softmax stats + per-lane VKQ register tile. ---
            float m = -std::numeric_limits<float>::infinity();
            float l = 0.f;
            float vkq[DPL];
            #pragma unroll
            for (uint32_t d = 0; d < DPL; ++d) vkq[d] = 0.f;

            // Causal extent for this WG (uniform across lanes → safe break).
            // All G heads share the same q-row range (q0..q0+Br-1) so the causal
            // last-row is identical across heads in the WG.
            const uint32_t wg_last_row = sycl::min(q0 + Br - 1, T - 1);
            const uint32_t abs_q_last  = start_pos + wg_last_row;

            const uint32_t n_tiles = (ctx_len + Bc - 1) / Bc;
            for (uint32_t kt = 0; kt < n_tiles; ++kt) {
                const uint32_t k0   = kt * Bc;
                const uint32_t klen = sycl::min(Bc, ctx_len - k0);
                if (k0 > abs_q_last) break;   // future for the whole WG (uniform)

                // --- Cooperative half2-vectorized K/V tile load (ONCE per WG). ---
                // Shared by all G heads of this kv_head — the amortized win.
                constexpr uint32_t TILE_H2 = (Bc * HD) / 2;   // 4096
                for (uint32_t i = lid; i < TILE_H2; i += WG_ITEMS) {
                    const uint32_t kl = i / (HD / 2);
                    const uint32_t d2 = i % (HD / 2);
                    if (kl < klen) {
                        const uint64_t off2 =
                            ((uint64_t(kv_head) * max_ctx + k0 + kl) * HD) / 2 + d2;
                        K_slm2[i] = k_cache2[off2];
                        V_slm2[i] = v_cache2[off2];
                    } else {
                        K_slm2[i] = sycl::half2(0, 0);
                        V_slm2[i] = sycl::half2(0, 0);
                    }
                }
                sycl::group_barrier(it.get_group());

                // --- iter_KQ: per-lane COMPLETE dots → kq[KVPL]. ---
                // The subgroup's q_head reads its own Q_slm slot (hig*Br + r).
                float kq[KVPL];
                #pragma unroll
                for (uint32_t kk = 0; kk < KVPL; ++kk) kq[kk] = 0.f;

                if (active) {
                    const uint32_t qslot = hig * Br + r;        // this subgroup's Q row slot
                    const sycl::half2* qrow2 =
                        reinterpret_cast<const sycl::half2*>(&Q_slm[qslot * HD]);
                    #pragma unroll
                    for (uint32_t kk = 0; kk < KVPL; ++kk) {
                        const uint32_t c = lane + kk * SG_SIZE;
                        const sycl::half2* krow2 =
                            reinterpret_cast<const sycl::half2*>(&K_slm[c * HD]);
                        float acc = 0.f;
                        #pragma unroll
                        for (uint32_t d2 = 0; d2 < HD / 2; ++d2) {
                            const sycl::half2 qd = qrow2[d2];
                            const sycl::half2 kd = krow2[d2];
                            acc += float(qd.x()) * float(kd.x())
                                 + float(qd.y()) * float(kd.y());
                        }
                        kq[kk] = acc * scale;
                    }
                }

                // --- Apply causal mask, compute this tile's per-row max. ---
                float tile_max = -std::numeric_limits<float>::infinity();
                #pragma unroll
                for (uint32_t kk = 0; kk < KVPL; ++kk) {
                    const uint32_t c     = lane + kk * SG_SIZE;
                    const uint32_t abs_k = k0 + c;
                    const bool keep = active && (c < klen) && (abs_k <= abs_q);
                    if (!keep) kq[kk] = -std::numeric_limits<float>::infinity();
                    tile_max = sycl::fmax(tile_max, kq[kk]);
                }
                tile_max = sycl::reduce_over_group(sg, tile_max, sycl::maximum<float>());

                const float m_new = sycl::fmax(m, tile_max);
                const float alpha = (m == -std::numeric_limits<float>::infinity())
                                    ? 0.f : sycl::native::exp(m - m_new);

                // --- exp weights → P_slm[slot][kv], denom update, VKQ rescale. ---
                // P_slm slot indexes by THIS subgroup's (hig, r): slot = hig*Br + r.
                const uint32_t pslot = hig * Br + r;
                float row_l_add = 0.f;
                #pragma unroll
                for (uint32_t kk = 0; kk < KVPL; ++kk) {
                    const uint32_t c = lane + kk * SG_SIZE;
                    const float e = (kq[kk] == -std::numeric_limits<float>::infinity())
                                    ? 0.f : sycl::native::exp(kq[kk] - m_new);
                    if (active) P_slm[pslot * Bc + c] = e;
                    row_l_add += e;
                }
                row_l_add = sycl::reduce_over_group(sg, row_l_add, sycl::plus<float>());
                l = l * alpha + row_l_add;
                m = m_new;
                #pragma unroll
                for (uint32_t d = 0; d < DPL; ++d) vkq[d] *= alpha;

                sycl::group_barrier(it.get_group());  // P_slm ready

                // --- VKQ += P·V (pure per-lane FMA, no cross-lane). ---
                if (active && k0 <= abs_q) {
                    const uint32_t kmax = sycl::min(klen, abs_q - k0 + 1);
                    for (uint32_t c = 0; c < kmax; ++c) {
                        const float p = P_slm[pslot * Bc + c];
                        const uint32_t vbase = c * HD + lane * DPL;
                        #pragma unroll
                        for (uint32_t d = 0; d < DPL; ++d)
                            vkq[d] += p * float(V_slm[vbase + d]);
                    }
                }
                sycl::group_barrier(it.get_group());  // before next K/V tile load
            }

            // --- Normalize and write y for THIS subgroup's (q_head, row). ---
            if (active) {
                const float inv_l = (l > 0.f) ? (1.0f / l) : 0.f;
                const uint64_t yb = (uint64_t(tl) * n_q_heads + q_head) * HD + lane * DPL;
                #pragma unroll
                for (uint32_t d = 0; d < DPL; ++d)
                    y[yb + d] = sycl::half(vkq[d] * inv_l);
            }
        });
    });
}

// ===========================================================================
// full_attention_fa2_prefill_xmx — FlashAttention-2 prefill on the XMX engine.
// ===========================================================================
//
// Same online-softmax algorithm as full_attention_fa2_prefill_v2, but at large
// T the QK^T ([Br×d]·[d×Bc]) and P·V ([Br×Bc]·[Bc×d]) products are true
// matrix-matrix GEMMs — exactly what the B70 XMX matrix engine accelerates
// (~8× the vector ALU). We run them through SYCL joint_matrix (fp16 in, fp32
// accumulate) using the SAME sanctioned path as gemm_fp16.cpp (NOT block2d /
// ESIMD / lsc_load).
//
// One WG == one subgroup (SG_SIZE=16 lanes) processing one (q_head, q_block).
// The subgroup owns Br=16 query rows and streams the causal KV in Bc=64-column
// tiles. joint_matrix MAD/load/store are subgroup-collective, so a single
// subgroup issues all the matrix ops; the 16 lanes also cooperatively do the
// SLM loads and the elementwise softmax pass.
//
// ----- layout choice (the #1 correctness risk) -----
//   joint_matrix_mad computes  acc[M×N] += A[M×K] · B[K×N].
//
//   S = Q · K^T   (S:[Br×Bc], M=Br, N=Bc, K=head_dim contraction)
//     A = Q_smem  [Br × head_dim]               → use::a, ROW_MAJOR, stride=head_dim
//     B = K^T     [head_dim × Bc]               → use::b, COL_MAJOR, stride=head_dim
//         K_smem is stored key-major [Bc][head_dim] (matches the cache layout).
//         Loading it as a col_major b-operand with leading dim = head_dim makes
//         the matrix element (k=dim, n=key) read K_smem[key*head_dim + dim] =
//         K^T[dim][key].  THIS is how we get the transpose for free — no
//         explicit transpose pass, no block2d.
//     acc = float accumulator, ROW_MAJOR store → S_smem[Br×Bc].
//
//   O = P · V     (O:[Br×head_dim], M=Br, N=head_dim, K=Bc contraction)
//     A = P_smem  [Br × Bc]    (softmax weights as fp16)  → use::a, ROW_MAJOR, stride=Bc
//     B = V_smem  [Bc × head_dim] (key-major == [K × N])  → use::b, ROW_MAJOR, stride=head_dim
//     acc = float accumulator for the P·V of THIS tile (then rescaled+added to O_smem).
//
// ----- online-softmax rescale order (per tile, FA-2) -----
//   1. S_smem = (Q·K^T)*scale  via XMX.
//   2. cooperative pass: causal-mask S_smem; per-row new max m_new = max(m, rowmax(S));
//      alpha = exp(m - m_new); P_smem = exp(S - m_new); l = l*alpha + rowsum(P).
//   3. rescale O_smem *= alpha   (BEFORE adding this tile's P·V).
//   4. Otile = P·V  via XMX; O_smem += Otile.
//   5. m = m_new.
//   After all tiles: y = O_smem / l.
//
// SLM per WG (head_dim=128): Q 4KB + K 16KB + V 16KB + S/P 8KB(f) /2KB(h) + O 8KB
//   ≈ 52 KB — well under the 128 KB/WG cap. General for head_dim multiple of 16,
//   Bc multiple of TN=16, Br multiple of TM=8.
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
                                           const std::vector<sycl::event>& deps) {
    if (T == 0) return {};
    namespace mat = sycl::ext::oneapi::experimental::matrix;
    using fp16 = sycl::half;

    constexpr uint32_t TM = 8;
    constexpr uint32_t TN = 16;
    constexpr uint32_t TK = 16;
    // MULTI-SUBGROUP: NSG subgroups per WG, each owns Br=16 query rows of the
    // same q_head, ALL sharing one K/V SLM tile (loaded cooperatively, once).
    // This fills the EU (NSG×16 work-items vs 16), parallelizes the scalar
    // online-softmax NSG×, and amortizes the K/V SLM load across NSG× rows —
    // the three things that bottlenecked the 1-subgroup kernel.
    constexpr uint32_t Br  = 16;           // query rows / SUBGROUP (multiple of TM)
    constexpr uint32_t Bc  = 64;           // key tile depth        (multiple of TN)
    constexpr uint32_t MR  = Br / TM;      // 2 accumulator tiles in M
    // NSG chosen at runtime so the per-WG SLM stays under the device cap for ANY
    // head_dim (the kernel is NOT hd=128-only). Per sg: Q(Br*hd*2)+S(Br*Bc*4)+
    // P(Br*Bc*2)+O(Br*hd*4)+m,l(Br*8); the WG also shares K+V (Bc*hd*4).
    //   hd=128 → NSG=4 (~105KB, byte-identical to the tuned path);
    //   hd=256 → NSG=1 (~96KB) instead of failing submission at 184KB.
    constexpr uint32_t SLM_CAP = 112u * 1024u;             // safe budget < 128KB
    const uint32_t per_sg_slm = Br*head_dim*2 + Br*Bc*4 + Br*Bc*2 + Br*head_dim*4 + Br*8;
    const uint32_t shared_slm = Bc*head_dim*4;             // K + V fp16
    const uint32_t NSG = std::max(1u, std::min(8u,
                              (SLM_CAP > shared_slm) ? (SLM_CAP - shared_slm) / per_sg_slm : 1u));
    const uint32_t WG_ITEMS = NSG * SG_SIZE;

    const uint32_t gqa        = n_q_heads / n_kv_heads;
    const uint32_t ctx_len    = start_pos + T;
    const uint32_t n_q_blocks = (T + Br - 1) / Br;
    const uint32_t n_super    = (n_q_blocks + NSG - 1) / NSG;  // WGs of NSG q-blocks
    const uint32_t hd         = head_dim;

    // 1. Append (k_in, v_in) — identical to the v2 / naive append.
    auto append_evt = ie::ps(q, "fa2_xmx_append", [&](sycl::handler& h) {
        h.depends_on(deps);
        const uint32_t total = T * n_kv_heads * head_dim;
        constexpr uint32_t WG = 256;
        const uint32_t global = ((total + WG - 1) / WG) * WG;
        h.parallel_for(sycl::nd_range<1>(global, WG),
                       [=](sycl::nd_item<1> it) {
            const uint32_t idx = uint32_t(it.get_global_id(0));
            if (idx >= total) return;
            const uint32_t d  = idx % head_dim;
            const uint32_t kv = (idx / head_dim) % n_kv_heads;
            const uint32_t t  = idx / (n_kv_heads * head_dim);
            const uint64_t in_off  = (uint64_t(t) * n_kv_heads + kv) * head_dim + d;
            const uint64_t out_off = (uint64_t(kv) * max_ctx + (start_pos + t)) * head_dim + d;
            k_cache[out_off] = k_in[in_off];
            v_cache[out_off] = v_in[in_off];
        });
    });

    // 2. XMX FA-2 compute. One subgroup per (q_head, q_block).
    return ie::ps(q, "fa2_xmx_compute", [&](sycl::handler& h) {
        h.depends_on(append_evt);
        // Per-subgroup regions are stacked NSG-deep; K/V are shared by the WG.
        sycl::local_accessor<fp16, 1>  Q_slm(NSG * Br * hd, h);  // [NSG][Br × hd] row-major
        sycl::local_accessor<fp16, 1>  K_slm(Bc * hd, h);       // [Bc × hd] key-major (shared)
        sycl::local_accessor<fp16, 1>  V_slm(Bc * hd, h);       // [Bc × hd] key-major (shared)
        sycl::local_accessor<float, 1> S_slm(NSG * Br * Bc, h);  // [NSG][Br × Bc] scores
        sycl::local_accessor<fp16, 1>  P_slm(NSG * Br * Bc, h);  // [NSG][Br × Bc] softmax weights
        sycl::local_accessor<float, 1> O_slm(NSG * Br * hd, h);  // [NSG][Br × hd] running output
        sycl::local_accessor<float, 1> m_slm(NSG * Br, h);       // running row max
        sycl::local_accessor<float, 1> l_slm(NSG * Br, h);       // running row sum
        const float scale = 1.0f / sycl::sqrt(float(head_dim));

        h.parallel_for(sycl::nd_range<2>({uint64_t(n_q_heads) * n_super, WG_ITEMS},
                                          {1, WG_ITEMS}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t wg0     = uint32_t(it.get_group(0));
            const uint32_t q_head  = wg0 / n_super;
            const uint32_t super   = wg0 % n_super;
            const uint32_t kv_head = q_head / gqa;
            const uint32_t lid     = uint32_t(it.get_local_id(1));   // 0..NSG*16-1
            const uint32_t sg_id   = lid / SG_SIZE;                  // 0..NSG-1
            const uint32_t lane    = lid % SG_SIZE;                  // 0..15
            const uint32_t q_block = super * NSG + sg_id;            // this sg's q-block
            const bool     sg_act  = (q_block < n_q_blocks);         // last super may idle some sgs
            const uint32_t q0      = q_block * Br;                   // first query row (local)
            auto sg = it.get_sub_group();

            // Per-subgroup SLM windows (K/V are shared by the whole WG).
            const uint32_t Qoff = sg_id * Br * hd;
            const uint32_t Soff = sg_id * Br * Bc;
            const uint32_t Poff = sg_id * Br * Bc;
            const uint32_t Ooff = sg_id * Br * hd;
            const uint32_t MLo  = sg_id * Br;
            auto Qp = Q_slm.get_multi_ptr<sycl::access::decorated::no>() + Qoff;
            auto Kp = K_slm.get_multi_ptr<sycl::access::decorated::no>();
            auto Vp = V_slm.get_multi_ptr<sycl::access::decorated::no>();
            auto Sp = S_slm.get_multi_ptr<sycl::access::decorated::no>() + Soff;
            auto Pp = P_slm.get_multi_ptr<sycl::access::decorated::no>() + Poff;
            auto Op = O_slm.get_multi_ptr<sycl::access::decorated::no>() + Ooff;

            // --- Load this sg's Q tile [Br × hd] into its SLM window (zero-pad). ---
            for (uint32_t i = lane; i < Br * hd; i += SG_SIZE) {
                const uint32_t r = i / hd;
                const uint32_t d = i % hd;
                const uint32_t tl = q0 + r;
                Q_slm[Qoff + i] = (sg_act && tl < T)
                    ? q_in[(uint64_t(tl) * n_q_heads + q_head) * hd + d]
                    : fp16(0);
            }
            for (uint32_t r = lane; r < Br; r += SG_SIZE) {
                m_slm[MLo + r] = -std::numeric_limits<float>::infinity();
                l_slm[MLo + r] = 0.f;
            }
            for (uint32_t i = lane; i < Br * hd; i += SG_SIZE) O_slm[Ooff + i] = 0.f;
            sycl::group_barrier(it.get_group());

            // WG-wide causal extent (uniform → break stays collective): the last
            // valid query row across all NSG sgs in this super-block.
            const uint32_t wg_last_blk = sycl::min(super * NSG + NSG, n_q_blocks) - 1;
            const uint32_t wg_last_row = sycl::min(wg_last_blk * Br + Br - 1, T - 1);
            const uint32_t abs_q_last_wg = start_pos + wg_last_row;

            const uint32_t n_tiles = (ctx_len + Bc - 1) / Bc;
            for (uint32_t kt = 0; kt < n_tiles; ++kt) {
                const uint32_t k0   = kt * Bc;
                const uint32_t klen = sycl::min(Bc, ctx_len - k0);
                if (k0 > abs_q_last_wg) break;   // future for the whole WG (uniform)

                // --- Cooperative K/V tile load [Bc × hd] key-major (ALL WG lanes). ---
                for (uint32_t i = lid; i < Bc * hd; i += WG_ITEMS) {
                    const uint32_t kl = i / hd;
                    const uint32_t d  = i % hd;
                    if (kl < klen) {
                        const uint64_t off = (uint64_t(kv_head) * max_ctx + k0 + kl) * hd + d;
                        K_slm[i] = k_cache[off];
                        V_slm[i] = v_cache[off];
                    } else {
                        K_slm[i] = fp16(0);
                        V_slm[i] = fp16(0);
                    }
                }
                sycl::group_barrier(it.get_group());

                // --- (a) S = (Q · K^T) * scale  via XMX → S_slm[Br × Bc]. ---
                // A=Q row_major [Br×hd], B=K^T col_major (K_slm key-major, ld=hd).
                // No unroll on mr: MR=8 → keep one acc fragment live (avoid spill).
                for (uint32_t mr = 0; mr < MR; ++mr) {
                    for (uint32_t nc = 0; nc < Bc; nc += TN) {
                        mat::joint_matrix<sycl::sub_group, float, mat::use::accumulator,
                                          TM, TN> acc;
                        mat::joint_matrix_fill(sg, acc, 0.0f);
                        for (uint32_t kk = 0; kk < hd; kk += TK) {
                            mat::joint_matrix<sycl::sub_group, fp16, mat::use::a,
                                              TM, TK, mat::layout::row_major> a_tile;
                            mat::joint_matrix<sycl::sub_group, fp16, mat::use::b,
                                              TK, TN, mat::layout::col_major> b_tile;
                            // A: row (mr*TM), col (kk) of Q_slm[Br×hd], stride=hd.
                            mat::joint_matrix_load(sg, a_tile,
                                Qp + (mr * TM) * hd + kk, /*stride=*/hd);
                            // B = K^T: element (k=dim, n=key). K_slm key-major
                            // [Bc×hd]; col_major load with ld=hd reads
                            // K_slm[(nc+n)*hd + (kk+k)] = K^T[kk+k][nc+n]. The
                            // base offset selects key column block nc, dim block kk.
                            mat::joint_matrix_load(sg, b_tile,
                                Kp + nc * hd + kk, /*stride=*/hd);
                            mat::joint_matrix_mad(sg, acc, a_tile, b_tile, acc);
                        }
                        mat::joint_matrix_store(sg, acc,
                            Sp + (mr * TM) * Bc + nc,
                            /*stride=*/Bc, mat::layout::row_major);
                    }
                }
                sycl::group_barrier(it.get_group());

                // --- (b)+(c) causal mask + online softmax (one lane per row). ---
                // 16 lanes, Br=16 rows → lane == row.
                for (uint32_t r = lane; r < Br; r += SG_SIZE) {
                    const uint32_t tl    = q0 + r;
                    const bool     activ = sg_act && (tl < T);
                    const uint32_t abs_q = start_pos + tl;
                    const uint32_t base  = Soff + r * Bc;
                    const uint32_t pbase = Poff + r * Bc;

                    // row max over valid (causal & in-range) keys.
                    float m_old = m_slm[MLo + r];
                    float row_m = m_old;
                    for (uint32_t c = 0; c < Bc; ++c) {
                        const uint32_t abs_k = k0 + c;
                        const bool keep = activ && (c < klen) && (abs_k <= abs_q);
                        float s = keep ? (S_slm[base + c] * scale)
                                       : -std::numeric_limits<float>::infinity();
                        S_slm[base + c] = s;            // store masked+scaled score
                        row_m = sycl::fmax(row_m, s);
                    }
                    const float m_new = row_m;
                    const float alpha = (m_old == -std::numeric_limits<float>::infinity())
                                        ? 0.f : sycl::native::exp(m_old - m_new);
                    float row_l = 0.f;
                    for (uint32_t c = 0; c < Bc; ++c) {
                        const float s = S_slm[base + c];
                        const float e = (s == -std::numeric_limits<float>::infinity())
                                        ? 0.f : sycl::native::exp(s - m_new);
                        P_slm[pbase + c] = fp16(e);
                        row_l += e;
                    }
                    // rescale running stats + O for this row.
                    l_slm[MLo + r] = l_slm[MLo + r] * alpha + row_l;
                    m_slm[MLo + r] = m_new;
                    for (uint32_t d = 0; d < hd; ++d)
                        O_slm[Ooff + r * hd + d] *= alpha;   // FA-2: rescale O before += P·V
                }
                sycl::group_barrier(it.get_group());

                // --- (d) O = O*alpha + P · V  via XMX, into O_slm[Br × hd]. ---
                // O_slm was already rescaled by alpha in step (b). Seed the
                // accumulator FROM O_slm (so the MAD accumulates onto the
                // rescaled O — true FA-2 O += P·V), then store back.
                // A=P row_major [Br×Bc] ld=Bc, B=V row_major [Bc×hd] ld=hd.
                // (Op is this sg's O window, defined at top with the Ooff offset.)
                for (uint32_t mr = 0; mr < MR; ++mr) {
                    for (uint32_t nd = 0; nd < hd; nd += TN) {
                        mat::joint_matrix<sycl::sub_group, float, mat::use::accumulator,
                                          TM, TN> acc;
                        mat::joint_matrix_load(sg, acc,
                            Op + (mr * TM) * hd + nd, /*stride=*/hd,
                            mat::layout::row_major);
                        for (uint32_t kk = 0; kk < Bc; kk += TK) {
                            mat::joint_matrix<sycl::sub_group, fp16, mat::use::a,
                                              TM, TK, mat::layout::row_major> a_tile;
                            mat::joint_matrix<sycl::sub_group, fp16, mat::use::b,
                                              TK, TN, mat::layout::row_major> b_tile;
                            mat::joint_matrix_load(sg, a_tile,
                                Pp + (mr * TM) * Bc + kk, /*stride=*/Bc);
                            mat::joint_matrix_load(sg, b_tile,
                                Vp + kk * hd + nd, /*stride=*/hd);
                            mat::joint_matrix_mad(sg, acc, a_tile, b_tile, acc);
                        }
                        mat::joint_matrix_store(sg, acc,
                            Op + (mr * TM) * hd + nd, /*stride=*/hd,
                            mat::layout::row_major);
                    }
                }
                sycl::group_barrier(it.get_group());
            }

            // --- Normalize O by 1/l and write y (active rows only). ---
            if (sg_act) {
                for (uint32_t i = lane; i < Br * hd; i += SG_SIZE) {
                    const uint32_t r  = i / hd;
                    const uint32_t d  = i % hd;
                    const uint32_t tl = q0 + r;
                    if (tl >= T) continue;
                    const float l = l_slm[MLo + r];
                    const float inv_l = (l > 0.f) ? (1.0f / l) : 0.f;
                    y[(uint64_t(tl) * n_q_heads + q_head) * hd + d] =
                        fp16(O_slm[Ooff + i] * inv_l);
                }
            }
        });
    });
}

}  // namespace ie
