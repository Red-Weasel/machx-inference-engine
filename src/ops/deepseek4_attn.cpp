// src/ops/deepseek4_attn.cpp — DeepSeek-V4 attention stack (Phase 3).
//
// Every routine is a transcription of
// transformers/models/deepseek_v4/modeling_deepseek_v4.py; the reference line
// numbers are quoted at each site and in include/ie/deepseek4_attn.hpp.
// Where the order of operations is observable in fp32 (the softmax max
// subtraction, the sink column, the position-bias-before-split in the CSA
// pooling) the reference order is preserved exactly.
//
// Accumulation follows the Phase 2 convention (src/ops/deepseek4_ops.cpp):
// fp32 accumulators split over `kSG` sub-group lanes, so a length-N reduction
// has a serial chain of N/kSG plus a log-depth tree reduce.

#include "ie/deepseek4_attn.hpp"

#include "ie/kernel_profiler.hpp"
#include "ie/deepseek41_topk_split.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

namespace ie {

namespace {

constexpr int kSG = 16;   // sub-group size (repo-wide idiom)
constexpr int kWG = 256;  // 16 sub-groups per work-group

constexpr float kNegInf = -std::numeric_limits<float>::infinity();
// Below this, a mask entry is a "disallowed" sentinel rather than a bias: both
// -inf (indexer block bias) and float::lowest() (sliding causal mask) fall here,
// and exp() of anything at or below it underflows to exactly 0.0.
constexpr float kMaskedCut = -1e30f;

// The two-segment kv axis (Ds4KvSegs, deepseek4_attn.hpp): column i < n_a is
// row i of `a` (fp32), column i >= n_a is row i - n_a of `b` (fp16) — the SAME
// column order the old contiguous staging buffer had.  Each kernel below
// expresses its per-column arithmetic as ONE generic lambda and calls it under
// a branch on the segment; `i` is uniform across the sub-group, so the branch
// is a select and the group reductions inside stay convergent.  On the fp32
// segment `float(krow[d])` is a no-op, so that path is the pre-Phase-1 code
// instruction for instruction.

// Order-preserving map float -> uint32 (IEEE754, no NaN expected), packed with
// the complement of the entry index so that a plain uint64 max yields
// "largest score, smallest index on a tie" — a strict total order, which is
// what makes the top-k output deterministic.
inline uint64_t pack_key(float v, uint32_t idx) {
    uint32_t u = sycl::bit_cast<uint32_t>(v);
    u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return (uint64_t(u) << 32) | uint64_t(0xFFFFFFFFu - idx);
}

// DSpark P2: the ring column for query row t (see Ds4KvSegs::a_new); `i` is uniform across the sub-group
inline const float* ds4_seg_a_row(const Ds4KvSegs& kv, uint32_t i, uint32_t t, uint32_t head_dim) {
    if (kv.a_new) { const uint32_t j = (i + kv.n_a - kv.a_new_base) % kv.n_a; if (j < kv.a_new_n && (j <= t || kv.a_new_nocausal)) return kv.a_new + size_t(j) * head_dim; }
    return kv.a + size_t(i) * head_dim;
}

}  // namespace

// ---------------------------------------------------------------------------
// RoPE frequency tables (host).
// ---------------------------------------------------------------------------
std::vector<float> ds4_rope_inv_freq(const Ds4RopeConfig& cfg, uint32_t rope_dim) {
    const uint32_t half = rope_dim / 2u;
    std::vector<float> inv(half);

    // compute_default_rope_parameters (modeling:114):
    //   inv_freq = 1 / base ** (arange(0, dim, 2) / dim)
    if (!cfg.yarn) {
        for (uint32_t r = 0; r < half; ++r)
            inv[r] = float(1.0 / std::pow(double(cfg.theta),
                                          double(2u * r) / double(rope_dim)));
        return inv;
    }

    // _compute_yarn_parameters (modeling_rope_utils.py:327).  NTK-by-parts:
    // blend extrapolation (high-frequency dims) with 1/factor interpolation
    // (low-frequency dims) across a linear ramp between the two correction dims.
    const double base = double(cfg.theta);
    const double dim  = double(rope_dim);
    const double omax = double(cfg.original_max_pos);
    auto corr_dim = [&](double rot) {
        return (dim * std::log(omax / (rot * 2.0 * M_PI))) / (2.0 * std::log(base));
    };
    // `truncate` defaults to True in transformers, so floor/ceil are applied.
    double low  = std::floor(corr_dim(double(cfg.beta_fast)));
    double high = std::ceil(corr_dim(double(cfg.beta_slow)));
    low  = std::max(low, 0.0);
    high = std::min(high, dim - 1.0);
    if (low == high) high += 0.001;  // prevent singularity (reference comment)

    for (uint32_t r = 0; r < half; ++r) {
        const double pos_freq = std::pow(base, double(2u * r) / dim);
        const double extrap   = 1.0 / pos_freq;
        const double interp   = 1.0 / (double(cfg.factor) * pos_freq);
        double ramp = (double(r) - low) / (high - low);
        ramp = std::min(std::max(ramp, 0.0), 1.0);
        const double ext_factor = 1.0 - ramp;
        inv[r] = float(interp * (1.0 - ext_factor) + extrap * ext_factor);
    }
    return inv;
}

std::vector<int32_t> ds4_compress_positions(uint32_t n_win, uint32_t compress_rate,
                                            uint32_t first_window_position) {
    std::vector<int32_t> p(n_win);
    for (uint32_t i = 0; i < n_win; ++i)
        p[i] = int32_t(i * compress_rate + first_window_position);
    return p;
}

sycl::event ds4_rope_cos_sin(sycl::queue& q,
                             const float* inv_freq, const int32_t* positions,
                             float* cos_out, float* sin_out,
                             uint32_t n_pos, uint32_t half_dim,
                             float attention_factor,
                             const std::vector<sycl::event>& deps) {
    return ie::ps(q, "ds4_rope_cos_sin", [&](sycl::handler& h) {
        h.depends_on(deps);
        const uint32_t total  = n_pos * half_dim;
        const uint32_t global = ((total + kWG - 1) / kWG) * kWG;
        h.parallel_for(sycl::nd_range<1>(global, kWG), [=](sycl::nd_item<1> it) {
            const uint32_t i = uint32_t(it.get_global_id(0));
            if (i >= total) return;
            const uint32_t r = i % half_dim;
            const uint32_t p = i / half_dim;
            // freqs = inv_freq @ position_ids  (modeling:165), then cos/sin
            // scaled by attention_scaling (1.0 for both V4 tables).
            const float th = float(positions[p]) * inv_freq[r];
            cos_out[i] = sycl::cos(th) * attention_factor;
            sin_out[i] = sycl::sin(th) * attention_factor;
        });
    });
}

// ---------------------------------------------------------------------------
// Interleaved partial RoPE on the trailing rope slice — apply_rotary_pos_emb
// (modeling:342) with rotate_half (modeling:335).
// ---------------------------------------------------------------------------
sycl::event ds4_rope_apply(sycl::queue& q,
                           const float* x, const float* cos_in, const float* sin_in,
                           float* y,
                           uint32_t n_tokens, uint32_t n_heads,
                           uint32_t head_dim, uint32_t rope_dim,
                           float sin_sign,
                           const std::vector<sycl::event>& deps) {
    if (rope_dim > head_dim || (rope_dim & 1u) ||
        (n_tokens && (!n_heads || !head_dim)))
        throw std::invalid_argument("ds4_rope_apply: invalid head or rotary dimensions");
    const uint64_t rows = uint64_t(n_tokens) * n_heads;
    if (head_dim && rows > std::numeric_limits<size_t>::max() / sizeof(float) / head_dim)
        throw std::invalid_argument("ds4_rope_apply: tensor byte span overflows size_t");
    const size_t total = size_t(rows) * head_dim;
    if (total && (!x || !y || (rope_dim && (!cos_in || !sin_in))))
        throw std::invalid_argument("ds4_rope_apply: null live operand");
    return ie::ps(q, "ds4_rope_apply", [&](sycl::handler& h) {
        h.depends_on(deps);
        const uint32_t half = rope_dim / 2u;
        const uint32_t nope = head_dim - rope_dim;
        if (!total || (x == y && !half)) {
            h.single_task([] {});
            return;
        }
        if (x == y) {
            // The other channels already have their final values. One item
            // owns a rotary pair, consuming both inputs before either store.
            const size_t pairs = size_t(rows) * half;
            auto rotate_pairs = [&]<typename Index>() {
                const Index count = Index(pairs);
                h.parallel_for(sycl::nd_range<1>((pairs + kWG - 1) / kWG * kWG, kWG),
                               [=](sycl::nd_item<1> it) {
                    const Index i = Index(it.get_global_id(0));
                    if (i >= count) return;
                    const Index row = i / half;
                    const uint32_t p = uint32_t(i % half);
                    const Index t = row / n_heads;
                    const Index j = row * head_dim + nope + Index(p) * 2;
                    const float c = cos_in[t * half + p];
                    const float s = sin_in[t * half + p] * sin_sign;
                    const float a = x[j], b = x[j + 1];
                    y[j] = a * c - b * s;
                    y[j + 1] = b * c + a * s;
                });
            };
            if (total <= std::numeric_limits<uint32_t>::max())
                rotate_pairs.template operator()<uint32_t>();
            else
                rotate_pairs.template operator()<size_t>();
            return;
        }
        // Retain the narrow address arithmetic for ordinary disjoint tensors;
        // wider tensors use a separately compiled, overflow-safe mapping.
        auto copy_and_rotate = [&]<typename Index>() {
            const Index row_width = Index(n_heads) * head_dim;
            const Index count = Index(total);
            h.parallel_for(sycl::nd_range<1>((total + kWG - 1) / kWG * kWG, kWG),
                           [=](sycl::nd_item<1> it) {
                const Index i = Index(it.get_global_id(0));
                if (i >= count) return;
                const uint32_t d = uint32_t(i % head_dim);
                const uint32_t t = uint32_t(i / row_width);
                if (d < nope) { y[i] = x[i]; return; }
                const uint32_t rd = d - nope;
                if (rd & 1u) return;
                const uint32_t p = rd / 2u;
                const float c = cos_in[size_t(t) * half + p];
                const float s = sin_in[size_t(t) * half + p] * sin_sign;
                const float a = x[i], b = x[i + 1];
                y[i] = a * c - b * s;
                y[i + 1] = b * c + a * s;
            });
        };
        if (total <= std::numeric_limits<uint32_t>::max())
            copy_and_rotate.template operator()<uint32_t>();
        else
            copy_and_rotate.template operator()<size_t>();
    });
}

// ---------------------------------------------------------------------------
// Weighted RMSNorm, fp32 — DeepseekV4RMSNorm (modeling:46).
// ---------------------------------------------------------------------------
sycl::event ds4_rms_norm(sycl::queue& q,
                         const float* x, const float* w, float* y,
                         uint32_t n_rows, uint32_t n, float eps,
                         const std::vector<sycl::event>& deps) {
    return ie::ps(q, "ds4_rms_norm", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(size_t(n_rows) * kWG, kWG),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
            const uint32_t row = uint32_t(it.get_group(0));
            const uint32_t lid = uint32_t(it.get_local_id(0));
            const float* xr = x + size_t(row) * n;
            float*       yr = y + size_t(row) * n;
            float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
            uint32_t i = lid;
            for (; i + 3u * kWG < n; i += 4u * kWG) {
                const float v0 = xr[i], v1 = xr[i + kWG];
                const float v2 = xr[i + 2u * kWG], v3 = xr[i + 3u * kWG];
                s0 += v0 * v0; s1 += v1 * v1; s2 += v2 * v2; s3 += v3 * v3;
            }
            for (; i < n; i += kWG) { const float v = xr[i]; s0 += v * v; }
            const float ss = sycl::reduce_over_group(it.get_group(),
                                                     (s0 + s1) + (s2 + s3),
                                                     sycl::plus<float>());
            const float r = sycl::rsqrt(ss / float(n) + eps);
            for (uint32_t j = lid; j < n; j += kWG) yr[j] = w[j] * (xr[j] * r);
        });
    });
}

// ---------------------------------------------------------------------------
// Core attention — eager_attention_forward (modeling:717) as V4 invokes it.
//
// TWO SHAPES, one dispatcher (ds4_attention, at the bottom of this block).
//
//   PLAIN   one sub-group per (query token, head).  n_heads*T sub-groups.
//   SPLIT   `SP` sub-groups per (query token, head), each walking a strided
//           1-in-SP slice of the kv axis and keeping its OWN online-softmax
//           state, combined through SLM at the end.  n_heads*T*SP sub-groups.
//
// WHY.  At decode T=1 the plain shape launches `n_heads` work-groups of kSG=16
// work-items — 32 x 16 = 512 work-items against a B70's 256 compute units and
// 1024-wide work-groups.  The kernel is not slow because the arithmetic is
// hard; it is slow because almost none of the machine is running.  Measured on
// this box at the real decode shape (T=1, n_heads 32, head_dim 512, n_kv 1152,
// 55.6% of columns live), best of three runs of 60 iterations:
//
//     plain                        2.828 ms
//     split SP=8                   0.295 ms   9.6x
//     split SP=16                  0.165 ms  17.2x
//     split SP=32                  0.116 ms  24.5x
//
// and at n_kv 4224 (ctx 16k, 15.2% live): plain 3.311 ms, SP=32 0.130 ms, 25.5x.
//
// WHY STRIDED, NOT BLOCKED.  Sub-group s takes columns s, s+SP, s+2*SP, ... —
// NOT a contiguous block.  The mask-first skip above means a sub-group's real
// cost is the number of LIVE columns it draws, and the live set is two very
// unevenly distributed pieces: a dense contiguous sliding band (128 columns)
// followed by the indexer's top-512 scattered over the compressed entries.  A
// blocked partition would hand the sliding band entirely to sub-group 0 and
// leave the rest waiting on it; a stride-SP partition deals both pieces round
// robin, so every sub-group draws within one column of the same live count for
// any mask that is not periodic with period SP.  (The dense band has period 1
// and the indexer's picks are a uniform random subset; neither is.)
//
// EVERY LIVE COLUMN EXACTLY ONCE.  The union over s in [0, SP) of
// {s, s+SP, s+2SP, ...} n [0, n_kv) is exactly [0, n_kv), and the sets are
// pairwise disjoint, because congruence mod SP partitions the integers.  A
// dropped or double-counted column is the dangerous failure mode here, so
// deepseek4_decode_gate_test checks the visit count of every column directly
// rather than trusting that sentence.
//
// NOT BIT-IDENTICAL, AND WHY THAT IS ACCEPTABLE.  The logits are unchanged —
// same lane->dim map, same per-column sub-group reduce — but the softmax
// numerator and denominator are now summed as SP independent chains plus a
// combine instead of one chain, and floating-point addition is not associative.
// deepseek4_decode_gate_test derives the reassociation bound from SP and the
// live count, prints observed-vs-bound, and backs it with negative controls.
//
// head_dim must be a multiple of kSG and at most kSG*32 = 512 (V4-Flash
// head_dim is exactly 512; the indexer's 128 also fits).  Larger head dims
// would need a bigger per-lane array — the same constraint
// full_attention_gemma carries.
// ---------------------------------------------------------------------------
namespace {

sycl::event ds4_attention_plain(sycl::queue& q,
                                const float* q_in, Ds4KvSegs kv,
                                const float* mask, const float* sinks,
                                float* y,
                                uint32_t T, uint32_t n_heads, uint32_t head_dim,
                                uint32_t n_kv, float scaling,
                                const std::vector<sycl::event>& deps) {
    // HEADS PER WORK-GROUP: MEASURED AT ZERO, so this kernel keeps one head per work-group.  Decode runs
    // 64 heads x 40 layers = 2,560 work-groups per token and `ds4_attention` measured 13.12 ms/token there,
    // i.e. 5.1 us per work-group, which looked exactly like launch/occupancy cost.  Packing 8, 16 or 32
    // heads into a work-group (a pure geometry change -- each sub-group already owns one head outright, so
    // it is bit-identical) moved it to 13.15: NOTHING.  The cost was the per-slice scan of the pick list,
    // which `pickpart` below fixes (13.12 -> 7.29 ms/token).  Recorded so the next person does not spend
    // the afternoon on work-group geometry.  2026-09-15.
    return ie::ps(q, "ds4_attention", [&](sycl::handler& h) {
        h.depends_on(deps);
        const uint32_t dpl = head_dim / kSG;   // dims per lane, <= 32
        h.parallel_for(sycl::nd_range<2>({T, size_t(n_heads) * kSG}, {1, kSG}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(kSG)]] {
            const uint32_t t    = uint32_t(it.get_group(0));
            const uint32_t hd   = uint32_t(it.get_group(1));
            const uint32_t lane = uint32_t(it.get_local_id(1));
            auto sg = it.get_sub_group();

            float qv[32];
            for (uint32_t d = 0; d < dpl; ++d)
                qv[d] = q_in[(size_t(t) * n_heads + hd) * head_dim + lane * dpl + d];

            float m = kNegInf, l = 0.f;
            float acc[32];
            for (uint32_t d = 0; d < dpl; ++d) acc[d] = 0.f;

            for (uint32_t i = 0; i < n_kv; ++i) {
                // Test the mask FIRST.  A masked column contributes exactly zero
                // no matter what Q.K is, so computing that dot product is pure
                // waste — and it is the dominant cost here (head_dim = 512 fused
                // multiply-adds plus a sub-group reduction, versus one float
                // load).  The old ordering paid it for every column and then
                // discarded the result, which threw away the whole point of the
                // lightning indexer: it selects top_k = 512 of the compressed
                // entries precisely so attention does not have to look at the
                // rest.  At ctx 4096 a ratio-4 layer masks ~44% of its columns;
                // the fraction grows with context, since n_kv grows and top_k
                // does not.
                //
                // Two sentinels reach here and both must be caught: the indexer's
                // block bias fills with -inf (ds4_block_bias_topk) while the
                // sliding causal mask fills with float::lowest()
                // (ds4_sliding_causal_mask).  Anything at or below -1e30 is one
                // of them: exp() of it underflows to exactly 0.0, so no
                // legitimate bias can live there.
                //
                // EXACTNESS.  For a column that the loop below would have
                // processed, e = exp(s - m_new) is exactly 0.0 and alpha is
                // exactly 1.0, so the online-softmax state is bit-identical
                // whether the column is skipped or walked — PROVIDED at least one
                // column is unmasked, which causality guarantees (a query token
                // always sees itself inside the 128-wide window).  The degenerate
                // all-masked case would differ, and there `sinks` (non-null on
                // every DeepSeek-V4 call site) drives both variants to zero anyway.
                float mv = 0.f;
                if (mask) {
                    mv = mask[size_t(t) * n_kv + i];
                    if (!(mv > kMaskedCut)) continue;
                }
                auto column = [&](const auto* krow) {
                    float part = 0.f;
                    for (uint32_t d = 0; d < dpl; ++d) part += qv[d] * float(krow[d]);
                    float s = sycl::reduce_over_group(sg, part, sycl::plus<float>()) * scaling + mv;
                    // Retained for the non-masked routes into this kernel and for a
                    // Q.K that is itself non-finite.
                    if (!(s > kNegInf)) return;

                    const float m_new = sycl::fmax(m, s);
                    const float alpha = (m == kNegInf) ? 0.f : sycl::exp(m - m_new);
                    const float e     = sycl::exp(s - m_new);
                    for (uint32_t d = 0; d < dpl; ++d)
                        acc[d] = acc[d] * alpha + e * float(krow[d]);   // K == V (modeling:825, 849)
                    l = l * alpha + e;
                    m = m_new;
                };
                if (i < kv.n_a)
                    column(ds4_seg_a_row(kv, i, t, head_dim) + lane * dpl);
                else if (kv.b_f16)
                    column(static_cast<const sycl::half*>(kv.b) + size_t(i - kv.n_a) * head_dim + lane * dpl);
                else
                    column(static_cast<const float*>(kv.b) + size_t(i - kv.n_a) * head_dim + lane * dpl);
            }

            // Per-head learnable sink: the reference concatenates it as an extra
            // logit column and drops the corresponding probability afterwards
            // (modeling:733-741), i.e. it only enters the denominator.
            float corr = 1.f, ll = l;
            if (sinks) {
                const float s   = sinks[hd];
                const float m_f = sycl::fmax(m, s);
                corr = (m == kNegInf) ? 0.f : sycl::exp(m - m_f);
                ll   = l * corr + sycl::exp(s - m_f);
            }
            const float inv = (ll > 0.f) ? (1.f / ll) : 0.f;
            for (uint32_t d = 0; d < dpl; ++d)
                y[(size_t(t) * n_heads + hd) * head_dim + lane * dpl + d] =
                    acc[d] * corr * inv;
        });
    });
}

// PREFILL, BATCHED REDUCTIONS — the plain kernel with `NB` live columns' dot
// products in flight at once.
//
// WHY.  Profiling put `ds4_attention` at 36% of a prefill chunk, and the tiling
// experiment (below, kept as an instrument) proved the kernel is NOT
// memory-bound: a shared tile cut K reads ~5x at ctx 4096 and still ran 25%
// SLOWER.  What is left is the REDUCTION: the plain shape does one
// `reduce_over_group` per LIVE column per (token, head) — a 4-step shuffle
// chain whose latency nothing hides, ~21 M of them per layer at pp1024.  This
// variant gathers NB live columns, computes their NB partial dots, and issues
// the NB reductions back to back so the shuffle latencies overlap instead of
// serialising.
//
// BIT-IDENTICAL, BY CONSTRUCTION.  Batching changes only WHEN a score is
// computed, never the arithmetic that computes it (same lane->dim map, same
// per-column sub-group reduce, same operand order) and never the order of the
// online-softmax STATE updates: the NB scores are applied to (m, l, acc) one at
// a time in ascending column order, exactly as the plain kernel applies them.
// The mask-first skip, the sentinels and the sink correction are verbatim.
template <uint32_t NB>
sycl::event ds4_attention_batched(sycl::queue& q,
                                  const float* q_in, Ds4KvSegs kv,
                                  const float* mask, const float* sinks,
                                  float* y,
                                  uint32_t T, uint32_t n_heads, uint32_t head_dim,
                                  uint32_t n_kv, float scaling,
                                  const std::vector<sycl::event>& deps) {
    return ie::ps(q, "ds4_attention", [&](sycl::handler& h) {
        h.depends_on(deps);
        const uint32_t dpl = head_dim / kSG;
        h.parallel_for(sycl::nd_range<2>({T, size_t(n_heads) * kSG}, {1, kSG}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(kSG)]] {
            const uint32_t t    = uint32_t(it.get_group(0));
            const uint32_t hd   = uint32_t(it.get_group(1));
            const uint32_t lane = uint32_t(it.get_local_id(1));
            auto sg = it.get_sub_group();

            float qv[32];
            for (uint32_t d = 0; d < dpl; ++d)
                qv[d] = q_in[(size_t(t) * n_heads + hd) * head_dim + lane * dpl + d];

            float m = kNegInf, l = 0.f;
            float acc[32];
            for (uint32_t d = 0; d < dpl; ++d) acc[d] = 0.f;

            uint32_t bidx[NB];
            float    bmv[NB];
            uint32_t nb = 0;
            const float* mrow = mask ? mask + size_t(t) * n_kv : nullptr;

            // One pass over the columns, flushing whenever NB live ones are
            // collected.  `flush` is the plain kernel's per-column body,
            // applied in ascending order over the batch.
            auto flush = [&]() {
                float part[NB], sc[NB];
                #pragma unroll
                for (uint32_t b = 0; b < NB; ++b) part[b] = 0.f;
                auto dot = [&](const auto* krow) {
                    float pp = 0.f;
                    for (uint32_t d = 0; d < dpl; ++d) pp += qv[d] * float(krow[d]);
                    return pp;
                };
                for (uint32_t b = 0; b < nb; ++b) {
                    const uint32_t i = bidx[b];
                    part[b] = i < kv.n_a ? dot(ds4_seg_a_row(kv, i, t, head_dim) + lane * dpl)
                            : kv.b_f16   ? dot(static_cast<const sycl::half*>(kv.b) + size_t(i - kv.n_a) * head_dim + lane * dpl)
                                         : dot(static_cast<const float*>(kv.b) + size_t(i - kv.n_a) * head_dim + lane * dpl);
                }
                // The NB reductions issue back to back; their shuffle latencies
                // overlap.  Each is the SAME reduction the plain kernel does.
                for (uint32_t b = 0; b < nb; ++b)
                    sc[b] = sycl::reduce_over_group(sg, part[b], sycl::plus<float>());
                for (uint32_t b = 0; b < nb; ++b) {
                    const float s2 = sc[b] * scaling + bmv[b];
                    if (!(s2 > kNegInf)) continue;
                    const float m_new = sycl::fmax(m, s2);
                    const float alpha = (m == kNegInf) ? 0.f : sycl::exp(m - m_new);
                    const float e     = sycl::exp(s2 - m_new);
                    auto axpy = [&](const auto* krow) {
                        for (uint32_t d = 0; d < dpl; ++d)
                            acc[d] = acc[d] * alpha + e * float(krow[d]);
                    };
                    const uint32_t i = bidx[b];
                    if (i < kv.n_a)
                        axpy(ds4_seg_a_row(kv, i, t, head_dim) + lane * dpl);
                    else if (kv.b_f16)
                        axpy(static_cast<const sycl::half*>(kv.b) + size_t(i - kv.n_a) * head_dim + lane * dpl);
                    else
                        axpy(static_cast<const float*>(kv.b) + size_t(i - kv.n_a) * head_dim + lane * dpl);
                    l = l * alpha + e;
                    m = m_new;
                }
                nb = 0;
            };

            for (uint32_t i = 0; i < n_kv; ++i) {
                float mv = 0.f;
                if (mrow) {
                    mv = mrow[i];
                    if (!(mv > kMaskedCut)) continue;
                }
                bidx[nb] = i;
                bmv[nb]  = mv;
                if (++nb == NB) flush();
            }
            if (nb) flush();

            float corr = 1.f, ll = l;
            if (sinks) {
                const float sv  = sinks[hd];
                const float m_f = sycl::fmax(m, sv);
                corr = (m == kNegInf) ? 0.f : sycl::exp(m - m_f);
                ll   = l * corr + sycl::exp(sv - m_f);
            }
            const float inv = (ll > 0.f) ? (1.f / ll) : 0.f;
            for (uint32_t d = 0; d < dpl; ++d)
                y[(size_t(t) * n_heads + hd) * head_dim + lane * dpl + d] =
                    acc[d] * corr * inv;
        });
    });
}

// PREFILL TILE — the plain kernel's arithmetic with the K/V rows staged in SLM
// and REUSED across a block of query tokens.
//
// WHY.  The plain shape gives one sub-group per (query token, head), so every
// query token walks the KV axis and reads each K row independently: after the
// 2026-08-09 expert-GEMM work `ds4_attention` was the #1 prefill kernel at
// 1.68 s of a 4.70 s chunk (36%).  TQ query tokens sharing one staged tile cut
// that K/V traffic by up to TQ.
//
// BIT-IDENTICAL, NOT MERELY CLOSE — and that is why it is tiled THIS way.
// Tiles are walked in ASCENDING column order and columns within a tile in
// ascending order, so every accumulator sees exactly the update sequence the
// plain kernel gives it.  Nothing is reassociated (unlike the SP-split decode
// shape, which needs a derived bound); the SLM is a CACHE, not a change of
// algorithm.  Each sub-group keeps its own m/l/acc, and the mask-first skip,
// the sentinel handling and the sink correction are the plain kernel verbatim.
//
// GEOMETRY.  Work-group = TQ sub-groups (one query token each) x kSG lanes;
// SLM = TK * head_dim floats (16 KB at TK=8, head_dim=512).
template <uint32_t TQ, uint32_t TK>
sycl::event ds4_attention_tile(sycl::queue& q,
                               const float* q_in, Ds4KvSegs kv,
                               const float* mask, const float* sinks,
                               float* y,
                               uint32_t T, uint32_t n_heads, uint32_t head_dim,
                               uint32_t n_kv, float scaling,
                               const std::vector<sycl::event>& deps) {
    const uint32_t n_blk = (T + TQ - 1u) / TQ;
    return ie::ps(q, "ds4_attention", [&](sycl::handler& h) {
        h.depends_on(deps);
        const uint32_t dpl = head_dim / kSG;
        sycl::local_accessor<float, 1> ktile(sycl::range<1>(size_t(TK) * head_dim), h);
        h.parallel_for(sycl::nd_range<2>({size_t(n_blk), size_t(n_heads) * TQ * kSG},
                                         {1, size_t(TQ) * kSG}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(kSG)]] {
            const uint32_t blk = uint32_t(it.get_group(0));
            const uint32_t hd  = uint32_t(it.get_group(1));
            const uint32_t lid = uint32_t(it.get_local_id(1));
            auto sg = it.get_sub_group();
            const uint32_t sub  = lid / kSG;
            const uint32_t lane = lid % kSG;
            const uint32_t t    = blk * TQ + sub;
            const bool     ok   = t < T;

            float qv[32];
            if (ok)
                for (uint32_t d = 0; d < dpl; ++d)
                    qv[d] = q_in[(size_t(t) * n_heads + hd) * head_dim + lane * dpl + d];

            float m = kNegInf, l = 0.f;
            float acc[32];
            for (uint32_t d = 0; d < dpl; ++d) acc[d] = 0.f;

            for (uint32_t i0 = 0; i0 < n_kv; i0 += TK) {
                const uint32_t nk = sycl::min(uint32_t(TK), n_kv - i0);
                it.barrier(sycl::access::fence_space::local_space);
                for (uint32_t e = lid; e < nk * head_dim; e += TQ * kSG)
                {
                    const uint32_t row = i0 + e / head_dim, col = e % head_dim;
                    ktile[e] = row < kv.n_a ? kv.a[size_t(row) * head_dim + col]
                             : kv.b_f16     ? float(static_cast<const sycl::half*>(kv.b)[size_t(row - kv.n_a) * head_dim + col])
                                            : static_cast<const float*>(kv.b)[size_t(row - kv.n_a) * head_dim + col];
                }
                it.barrier(sycl::access::fence_space::local_space);
                if (!ok) continue;
                for (uint32_t j = 0; j < nk; ++j) {
                    const uint32_t i = i0 + j;
                    float mv = 0.f;
                    if (mask) {
                        mv = mask[size_t(t) * n_kv + i];
                        if (!(mv > kMaskedCut)) continue;
                    }
                    const float* krow =
                        ktile.get_multi_ptr<sycl::access::decorated::no>().get() +
                        size_t(j) * head_dim + lane * dpl;
                    float part = 0.f;
                    for (uint32_t d = 0; d < dpl; ++d) part += qv[d] * krow[d];
                    const float sc =
                        sycl::reduce_over_group(sg, part, sycl::plus<float>()) * scaling + mv;
                    if (!(sc > kNegInf)) continue;
                    const float m_new = sycl::fmax(m, sc);
                    const float alpha = (m == kNegInf) ? 0.f : sycl::exp(m - m_new);
                    const float e     = sycl::exp(sc - m_new);
                    for (uint32_t d = 0; d < dpl; ++d)
                        acc[d] = acc[d] * alpha + e * krow[d];
                    l = l * alpha + e;
                    m = m_new;
                }
            }
            if (!ok) return;
            float corr = 1.f, ll = l;
            if (sinks) {
                const float sv  = sinks[hd];
                const float m_f = sycl::fmax(m, sv);
                corr = (m == kNegInf) ? 0.f : sycl::exp(m - m_f);
                ll   = l * corr + sycl::exp(sv - m_f);
            }
            const float inv = (ll > 0.f) ? (1.f / ll) : 0.f;
            for (uint32_t d = 0; d < dpl; ++d)
                y[(size_t(t) * n_heads + hd) * head_dim + lane * dpl + d] =
                    acc[d] * corr * inv;
        });
    });
}

// SP sub-groups per (t, head).  Same arithmetic as the plain kernel per slice;
// the only new code is the combine after the barrier.
//
// THE COMBINE IS A TREE, AND THAT IS WHAT SETS THE SLM BUDGET.
//
// The obvious combine writes all SP partial accumulators to SLM at once and has
// every work-item sum the SP of them that belong to its output element.  That
// costs SP * head_dim floats of SLM — 128 KB at SP=64, head_dim 512, which no
// device here will launch, so SP was capped at 32 and the kernel ran with 1024
// sub-groups at T=1.
//
// A pairwise tree over the slice axis needs only the UPPER HALF of the still-live
// slices resident at any round: peak (SP/2) * head_dim floats, half as much, so
// SP=64 fits in 64 KB and the decode shape gets 2048 sub-groups instead of 1024.
// Each round has the upper half write and the lower half accumulate, 2*log2(SP)
// barriers in total (12 at SP=64) against the flat form's 1 — and it is still
// FASTER, because a barrier on this device costs ~0.08 us while the flat form
// made every work-item re-read SP values out of SLM.
//
// Each slice is rescaled by exp(m_s - gm) ONCE, before the tree, instead of
// once per term inside the summation, so the tree adds log2(SP) roundings where
// the flat combine added 5*SP.  The bound in deepseek4_decode_gate_test's
// `split_const` is derived from exactly that and is tighter than the one it
// replaces.
//
// Measured on B70 against the previously shipped ladder (flat combine), best of
// three runs of 40, live 640 columns:
//     T=1  ctx4096   SP=32 flat 0.0962 -> SP=64 tree 0.0808 ms   1.19x
//     T=1  ctx16k    SP=32 flat 0.1066 -> SP=64 tree 0.0915 ms   1.17x
//     T=1  d=128     SP=32 flat 0.0330 -> SP=64 tree 0.0289 ms   1.14x
//     T=2  ctx4096   SP=16 flat 0.1732 -> SP=32 tree 0.1330 ms   1.30x
//     T=4  ctx4096   SP=16 flat 0.3847 -> SP=16 tree 0.2358 ms   1.63x
//     T=8  ctx4096   SP=8  flat 0.6868 -> SP=16 tree 0.4686 ms   1.47x
//     T=16 ctx4096   SP=8  flat 1.4290 -> SP=16 tree 0.9250 ms   1.54x
//
// FALSIFIED ALONG THE WAY, and recorded because it looks like the obvious win:
// holding the key row in registers so it is read ONCE instead of twice (it is
// needed as K for the dot and again as V for the accumulator).  That needs a
// third dpl-long per-lane array — 96 floats at head_dim 512 — which SPILLS at
// the default 128-GRF budget and runs 1.43x SLOWER (0.137 vs 0.096).  It can be
// made to pay with `sycl::ext::intel::experimental::grf_size<256>` (the PROPERTY
// spelling; the `[[intel::grf_size(256)]]` ATTRIBUTE is rejected by icpx 2026.1,
// which is why an earlier attempt concluded large GRF was unavailable), and then
// it is bit-identical and 1.11x — but large GRF also halves the work-group limit
// to 512 work-items, which forbids SP=64 outright.  The tree at SP=64 is worth
// more than the second read is (1.19x vs 1.11x at T=1, and 1.63x vs 1.17x at
// T=4), and the two cannot be combined, so the register cache is not here.
template <uint32_t SP>
sycl::event ds4_attention_split(sycl::queue& q,
                                const float* q_in, Ds4KvSegs kv,
                                const float* mask, const float* sinks,
                                float* y,
                                uint32_t T, uint32_t n_heads, uint32_t head_dim,
                                uint32_t n_kv, float scaling,
                                const std::vector<sycl::event>& deps) {
    constexpr uint32_t WG = SP * kSG;
    // Phase 30: partition segment b by PICK INDEX rather than by column mod SP (see the loop below).
    // Read once into a static, then copied to a plain local -- a SYCL kernel may not capture a static.
    // 0 = stride by column (Phase 26), 1 = contiguous run per slice sized by the row's valid count (Phase 30/33,
    // default), 2 = rank-strided (Phase 33, measured: recovers short context but moves a greedy near-tie at 24k).
    static const int pickpart_env = [] { const char* e = std::getenv("IE_DS4_ATTN_PICKPART"); if (!e || !*e) return 1; const int v = std::atoi(e); return (v >= 0 && v <= 2) ? v : 1; }();
    const int pickpart = pickpart_env;
    return ie::ps(q, "ds4_attention", [&](sycl::handler& h) {
        h.depends_on(deps);
        const uint32_t dpl = head_dim / kSG;
        // The tree combine's staging buffer: only the upper half of the still-live
        // slices is resident at any round, so [SP/2][head_dim] is enough, plus SP
        // running maxima and SP running sums.  ds4_attn_split_factor caps SP with
        // exactly this formula, so it fits the device by construction.
        sycl::local_accessor<float, 1> sacc(sycl::range<1>(size_t(SP / 2) * head_dim), h);
        sycl::local_accessor<float, 1> sml(sycl::range<1>(2 * SP), h);
        h.parallel_for(sycl::nd_range<2>({T, size_t(n_heads) * WG}, {1, WG}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(kSG)]] {
            const uint32_t t   = uint32_t(it.get_group(0));
            const uint32_t hd  = uint32_t(it.get_group(1));
            auto sg = it.get_sub_group();
            const uint32_t s    = uint32_t(sg.get_group_linear_id());   // slice
            const uint32_t lane = uint32_t(sg.get_local_linear_id());

            float qv[32];
            for (uint32_t d = 0; d < dpl; ++d)
                qv[d] = q_in[(size_t(t) * n_heads + hd) * head_dim + lane * dpl + d];

            float m = kNegInf, l = 0.f;
            float acc[32];
            for (uint32_t d = 0; d < dpl; ++d) acc[d] = 0.f;

            // The stride-SP slice this sub-group owns.  The per-column arithmetic lives in ONE lambda so the
            // dense and gathered paths below cannot drift apart: whichever enumerates the columns, the score,
            // the sentinel test and the online-softmax update are the same code on the same operands.
            auto do_col = [&](uint32_t i) {
                float mv = 0.f;
                if (mask) {
                    // Test the mask FIRST.  A masked column contributes exactly zero however Q.K comes out, so
                    // computing that dot product is pure waste -- head_dim fused multiply-adds plus a sub-group
                    // reduction against one float load.  Two sentinels reach here and both must be caught: the
                    // indexer's block bias fills with -inf (ds4_block_bias_topk) and the sliding causal mask
                    // with float::lowest() (ds4_sliding_causal_mask); anything at or below -1e30 is one of them,
                    // since exp() of it underflows to exactly 0.0 and no legitimate bias can live there.
                    //
                    // EXACTNESS.  For a column the body would have processed, e = exp(s - m_new) is exactly 0.0
                    // and alpha exactly 1.0, so the online-softmax state is bit-identical whether the column is
                    // skipped or walked -- PROVIDED at least one column is unmasked, which causality guarantees
                    // (a query always sees itself inside the 128-wide window).  The degenerate all-masked case
                    // would differ, and there `sinks` (non-null on every DeepSeek-V4 call site) drives both
                    // variants to zero anyway.
                    mv = mask[size_t(t) * n_kv + i];
                    if (!(mv > kMaskedCut)) return;
                }
                auto column = [&](const auto* krow) {
                    float part = 0.f;
                    for (uint32_t d = 0; d < dpl; ++d) part += qv[d] * float(krow[d]);
                    float sc = sycl::reduce_over_group(sg, part, sycl::plus<float>()) * scaling + mv;
                    if (!(sc > kNegInf)) return;

                    const float m_new = sycl::fmax(m, sc);
                    const float alpha = (m == kNegInf) ? 0.f : sycl::exp(m - m_new);
                    const float e     = sycl::exp(sc - m_new);
                    for (uint32_t d = 0; d < dpl; ++d)
                        acc[d] = acc[d] * alpha + e * float(krow[d]);
                    l = l * alpha + e;
                    m = m_new;
                };
                if (i < kv.n_a)
                    column(ds4_seg_a_row(kv, i, t, head_dim) + lane * dpl);
                else if (kv.b_f16)
                    column(static_cast<const sycl::half*>(kv.b) + size_t(i - kv.n_a) * head_dim + lane * dpl);
                else
                    column(static_cast<const float*>(kv.b) + size_t(i - kv.n_a) * head_dim + lane * dpl);
            };

            // THE GATHERED PATH (Phase 26, docs/deepseek41/68).  The dense form below walks every column of the
            // slice and consults the mask, which costs one full sub-group iteration per MASKED column -- and at
            // decode the indexer leaves only ~640 of n_kv columns live, so at ctx 262,144 that is 262,272
            // iterations per (query, head) to reach 640.  Measured at 121.3 ms/token, the largest decode term
            // once residency is fixed.  When the caller supplies the indexer's picks, segment b is visited
            // DIRECTLY instead: O(n_picks) rather than O(NC).
            //
            // BIT-IDENTICAL, and the argument has three parts.  (1) Segment a is still walked densely on the
            // same stride, and every a column precedes every b column, so the two passes visit the slice in the
            // same ascending order one pass did.  (2) `b_picks` is required ASCENDING (ds41_sort_picks_asc puts
            // it that way, because ds4_indexer_topk emits by descending score), so the b columns arrive in
            // ascending order too.  (3) The slice filter `i % SP == s` keeps each sub-group's column set exactly
            // what it was, so the per-slice partials the combine tree reduces are unchanged.  A pick outside the
            // slice, negative (the topk's unreachable sentinel) or past n_kv is skipped, and every surviving
            // column still goes through the same mask test, so a pick whose bias is somehow not live is dropped
            // exactly as the dense path would drop it.
            if (kv.b_picks && kv.n_picks) {
                const uint32_t a_end = sycl::min(kv.n_a, n_kv);
                for (uint32_t i = s; i < a_end; i += SP) do_col(i);
                // SEGMENT B IS PARTITIONED BY PICK INDEX, not by column mod SP.  The first form had EVERY
                // slice scan the WHOLE pick list to find the ~n_picks/SP entries on its own stride -- SP
                // redundant scans of 512 entries to do ~16 columns of work each.  A contiguous run per
                // slice needs no scan at all: at index_topk = 512 and SP = 32 that is 32x fewer iterations.
                //
                // NOT bit-identical, deliberately: this is a different PARTITION of exactly the same
                // columns, so the per-slice partials the combine tree reduces are grouped differently.  The
                // columns visited, their ascending order within a slice and the arithmetic per column are
                // all unchanged, so it is judged on the engine's logits bar rather than bit-equality.
                // IE_DS4_ATTN_PICKPART=0 restores the stride partition.
                //
                // ROW-IDENTICAL across T (P2) because n_picks is the constant row PITCH and the sort puts a
                // row's valid picks at [0, n_valid) with the -1 sentinels after them: the run each slice owns
                // is then the same list positions holding the same picks whether the row came in a T-row step
                // or a one-row step. Either premise broken (a count for a pitch, sentinels at the front) and
                // the partition shifts with T -- that was the multi-test failure of 2026-09-15.
                const int32_t* pr = kv.b_picks + size_t(t) * kv.n_picks;
                if (pickpart == 2) {
                    // Phase 33 (docs/deepseek41/73): RANK-STRIDED. Position == rank because the valid picks sit at
                    // [0, n_valid) with the sentinels after them, so this is T-independent like the contiguous run,
                    // but a short row (NC < 512) is spread over min(n_valid, SP) slices instead of n_valid / 8.
                    for (uint32_t j = s; j < kv.n_picks; j += SP) {
                        const int32_t p = pr[j];
                        if (p < 0) continue;
                        const uint32_t i = kv.n_a + uint32_t(p);
                        if (i < n_kv) do_col(i);
                    }
                } else if (pickpart == 1) {
                    // Phase 33: the run is sized by the ROW's valid count when the caller supplies it, not by the
                    // pitch. n_valid is the row's causal reach, so it is the same in a T-row step and a one-row
                    // step (T-independent), and at T = 1 it equals the pitch-or-NC count the Phase 30 form used
                    // (no sentinels at a one-row step), so the partition -- and every bit -- is unchanged there.
                    // Sized by the pitch, a short row (NC < 512) put all its picks in n_valid / 8 slices.
                    const uint32_t nv = kv.b_pick_n ? uint32_t(kv.b_pick_n[t]) : kv.n_picks;
                    const uint32_t per = (nv + SP - 1u) / SP;
                    const uint32_t j0 = s * per, j1 = sycl::min(j0 + per, nv);
                    for (uint32_t j = j0; j < j1; ++j) {
                        const int32_t p = pr[j];
                        if (p < 0) continue;
                        const uint32_t i = kv.n_a + uint32_t(p);
                        if (i < n_kv) do_col(i);
                    }
                } else {
                    for (uint32_t j = 0; j < kv.n_picks; ++j) {
                        const int32_t p = pr[j];
                        if (p < 0) continue;
                        const uint32_t i = kv.n_a + uint32_t(p);
                        if (i >= n_kv || (i % SP) != s) continue;
                        do_col(i);
                    }
                }
            } else {
                for (uint32_t i = s; i < n_kv; i += SP) do_col(i);
            }

            // Combine, part 1: the two scalars.  Every work-item recomputes them
            // (SP is small and they are already in SLM; broadcasting would cost
            // another barrier).  The loop order over p is fixed, so the result
            // does not depend on which work-item computed it or on scheduling.
            //
            // A slice that drew no live column at all has m = -inf and l = 0;
            // its weight exp(m - gm) would be exp(-inf - gm) = NaN when gm is
            // also -inf, so it is forced to 0 rather than evaluated.  Its acc is
            // exactly zero, so that is not an approximation.
            if (lane == 0) { sml[s] = m; sml[SP + s] = l; }
            it.barrier(sycl::access::fence_space::local_space);
            float gm = kNegInf;
            for (uint32_t p = 0; p < SP; ++p) gm = sycl::fmax(gm, sml[p]);
            float gl = 0.f;
            for (uint32_t p = 0; p < SP; ++p)
                gl += (sml[p] == kNegInf) ? 0.f : sml[SP + p] * sycl::exp(sml[p] - gm);

            // Combine, part 2: rescale ONCE to the global max, then a pairwise
            // tree down the slice axis.  Round `stride` has slices
            // [stride, 2*stride) hand their accumulator to slice s - stride;
            // after log2(SP) rounds slice 0 holds the whole sum.  Every slice is
            // folded in exactly once, because the rounds partition [1, SP) into
            // the half-open blocks [stride, 2*stride).
            const float w = (m == kNegInf) ? 0.f : sycl::exp(m - gm);
            for (uint32_t d = 0; d < dpl; ++d) acc[d] *= w;
            for (uint32_t stride = SP / 2u; stride > 0u; stride >>= 1) {
                if (s >= stride && s < 2u * stride)
                    for (uint32_t d = 0; d < dpl; ++d)
                        sacc[size_t(s - stride) * head_dim + lane * dpl + d] = acc[d];
                it.barrier(sycl::access::fence_space::local_space);
                if (s < stride)
                    for (uint32_t d = 0; d < dpl; ++d)
                        acc[d] += sacc[size_t(s) * head_dim + lane * dpl + d];
                it.barrier(sycl::access::fence_space::local_space);
            }

            // Sink algebra, unchanged from the plain kernel — it only ever saw
            // the FINAL (m, l), which is exactly (gm, gl).
            float corr = 1.f, ll = gl;
            if (sinks) {
                const float sk  = sinks[hd];
                const float m_f = sycl::fmax(gm, sk);
                corr = (gm == kNegInf) ? 0.f : sycl::exp(gm - m_f);
                ll   = gl * corr + sycl::exp(sk - m_f);
            }
            const float inv = (ll > 0.f) ? (1.f / ll) : 0.f;
            const float out_scale = corr * inv;

            if (s == 0)
                for (uint32_t d = 0; d < dpl; ++d)
                    y[(size_t(t) * n_heads + hd) * head_dim + lane * dpl + d] =
                        acc[d] * out_scale;
        });
    });
}

}  // namespace

// The split factor, from the two things that bound it.
//
// PARALLELISM.  Re-measured on B70 with the TREE combine, at n_kv 1152, head_dim
// 512, 640 live columns (best of three runs of 40; all times ms).  The `flat`
// rows are the previously shipped combine, kept so the ladder change is visible:
//
//                       T=1     T=2     T=4     T=8     T=16
//     flat SP=8       0.2879  0.2925  0.3340  0.6868  1.4290
//     flat SP=16      0.1620  0.1732  0.3847  0.7862  1.3109
//     flat SP=32      0.0962  0.1960  0.4061  0.8175  1.6324
//     tree SP=8       0.3069  0.2979  0.3384  0.6553  1.0846
//     tree SP=16      0.1722  0.1808  0.2358  0.4686  0.9250
//     tree SP=32      0.1008  0.1330  0.2725  0.5426  1.0909
//     tree SP=64      0.0808  0.1670  0.3377  0.6969  1.4030
//
// Same governing quantity as before — total sub-groups in flight, T*n_heads*SP,
// wants to be a couple of thousand on this card — but the tree combine both
// raises the reachable SP (its staging buffer is half the size) and makes the
// larger SPs cheaper, so the whole ladder moves up.  At T >= 32 the plain shape
// already has enough work-groups and splitting is a pure loss, unchanged.
//
// CAPACITY.  The tree's staging buffer is (SP/2)*head_dim floats plus 2*SP for
// the running maxima and sums.  At SP=64, head_dim 512 that is 64.5 KB: it fits
// B70's 128 KB and does NOT fit the integrated Graphics 12.70's 64 KB, where the
// loop below falls back to SP=32 (33 KB).  The cap is applied rather than
// assumed because a failed launch here is a hard error, not a slow path —
// confirmed the hard way, forcing an oversized SP on the iGPU returns
// UR_RESULT_ERROR_OUT_OF_RESOURCES.
//
// WHAT ELSE WAS TRIED AT THE DECODE SHAPE, AND FALSIFIED
// -----------------------------------------------------
// All at T=1, n_heads 32, head_dim 512, n_kv 1152, 640 live columns, against the
// then-shipped 0.0962 ms.  Reference points: a bare streaming read of the live
// key rows once per head is 0.0345 ms (41.9 MB, 1216 GB/s); the flat combine was
// 0.0122; walking the mask and touching nothing else is 0.0050.
//
// * INTERLEAVED lane->dim map (lane l owns dims l, l+kSG, ... so one SIMD16 load
//   covers 64 contiguous bytes instead of touching 16 cache lines 128 B apart):
//   0.1373 ms, 1.43x SLOWER.  The BLOCKED map this kernel uses gives each lane a
//   contiguous 128-byte run, which IGC folds into wide per-lane loads; the
//   "coalesced" map defeats that.  This is the opposite of the repo's usual
//   sg_dot idiom and was checked rather than assumed BECAUSE it is the opposite.
// * SHARING KV ACROSS HEADS THROUGH SLM.  head_count_kv is 1, so all 32 query
//   heads read the same rows; staging each column in SLM once per work-group and
//   giving the work-group HSG head sub-groups cuts GLOBAL kv traffic from 83.9 MB
//   to 1.31 MB — 64x — and is still no faster (0.0901 ms at its best, HSG=32 with
//   64 column groups).  The bytes were never the constraint; the number of load
//   INSTRUCTIONS is, and moving them to SLM does not reduce it.
// * HOLDING THE KEY ROW IN REGISTERS so it is read once instead of twice: spills
//   at 128 GRF (0.1370, 1.43x slower).  With grf_size<256> it is bit-identical
//   and 1.11x — but large GRF caps the work-group at 512 work-items, which
//   forbids SP=64, and SP=64 is worth more.  See the note on the split kernel.
// * grf_size<256> ALONE, without that register cache: 0.1867 ms, 1.94x slower.
// * BLOCKED instead of STRIDED column partition: identical (0.0345 both) in the
//   isolated read probe at SP=32, so the strided partition costs nothing for the
//   load balance it buys.  The kernel keeps strided for the balance argument,
//   now knowing the alternative is not faster rather than assuming it.
bool g_ds4_attn_split_fixed64 = false;   // DSpark P2 (docs/deepseek41/55): a multi-row decode step keeps T = 1's 64 slices so
                                          // each row's tree combine is the one-row step's, bit for bit (set by the V4.1 runtime)
static uint32_t ds4_attn_split_factor(const sycl::device& dev,
                                      uint32_t T, uint32_t head_dim) {
    uint32_t sp = 0;
    if      (T <= 1u)  sp = 64u;
    else if (g_ds4_attn_split_fixed64 && T <= 8u) sp = 64u;
    else if (T <= 2u)  sp = 32u;
    else if (T <= 16u) sp = 16u;
    else               return 0u;                       // plain shape wins

    const uint64_t slm = dev.get_info<sycl::info::device::local_mem_size>();
    const size_t   max_wg = dev.get_info<sycl::info::device::max_work_group_size>();
    while (sp > 1u &&
           (uint64_t(sp / 2u) * head_dim * sizeof(float)
                + uint64_t(2u * sp) * sizeof(float) > slm ||
            size_t(sp) * kSG > max_wg))
        sp /= 2u;
    return (sp > 1u) ? sp : 0u;
}

sycl::event ds4_attention(sycl::queue& q,
                          const float* q_in, const float* kv,
                          const float* mask, const float* sinks,
                          float* y,
                          uint32_t T, uint32_t n_heads, uint32_t head_dim,
                          uint32_t n_kv, float scaling,
                          const std::vector<sycl::event>& deps) {
    return ds4_attention_segs(q, q_in, Ds4KvSegs{kv, n_kv, nullptr, 0u, false}, mask, sinks, y,
                              T, n_heads, head_dim, scaling, /*xmx_kv_prepared=*/false, deps);
}

sycl::event ds4_attention_segs(sycl::queue& q,
                               const float* q_in, Ds4KvSegs kv,
                               const float* mask, const float* sinks,
                               float* y,
                               uint32_t T, uint32_t n_heads, uint32_t head_dim,
                               float scaling, bool xmx_kv_prepared,
                               const std::vector<sycl::event>& deps) {
    const uint32_t n_kv = kv.n_a + kv.n_b;
    // Not a runtime condition — head_dim is a model constant (512 for V4-Flash
    // attention, 128 for the indexer, both fine).  But violating it would
    // overflow the per-lane register arrays and silently corrupt device memory,
    // so it fails loudly instead of quietly.
    if (head_dim % kSG != 0 || head_dim / kSG > 32)
        throw std::invalid_argument(
            "ds4_attention: head_dim must be a multiple of " + std::to_string(kSG)
            + " and at most " + std::to_string(kSG * 32) + ", got "
            + std::to_string(head_dim));

    switch (ds4_attn_split_factor(q.get_device(), T, head_dim)) {
        case 64u: return ds4_attention_split<64>(q, q_in, kv, mask, sinks, y,
                                                 T, n_heads, head_dim, n_kv, scaling, deps);
        case 32u: return ds4_attention_split<32>(q, q_in, kv, mask, sinks, y,
                                                 T, n_heads, head_dim, n_kv, scaling, deps);
        case 16u: return ds4_attention_split<16>(q, q_in, kv, mask, sinks, y,
                                                 T, n_heads, head_dim, n_kv, scaling, deps);
        case 8u:  return ds4_attention_split<8>(q, q_in, kv, mask, sinks, y,
                                                T, n_heads, head_dim, n_kv, scaling, deps);
        case 4u:  return ds4_attention_split<4>(q, q_in, kv, mask, sinks, y,
                                                T, n_heads, head_dim, n_kv, scaling, deps);
        case 2u:  return ds4_attention_split<2>(q, q_in, kv, mask, sinks, y,
                                                T, n_heads, head_dim, n_kv, scaling, deps);
        default: {
            if (kv.a_new) throw std::invalid_argument("ds4_attention: the multi-row ring overlay (a_new) is honoured by the split kernels only (T <= 8)");   // gate P2 finding 5
            // docs/deepseek4/72 Phase K: the XMX flash kernel takes every
            // eligible prefill shape (IE_DS4_ATTN_XMX=0 restores the fp32 path).
            if (ds4_attention_xmx_eligible(q.get_device(), T, n_heads, head_dim))
                return ds4_attention_xmx_segs(q, q_in, kv, mask, sinks, y,
                                              T, n_heads, head_dim, scaling,
                                              xmx_kv_prepared, deps);
            // Prefill (T > 16): stage K/V in SLM and share the tile across TQ
            // query tokens.  BIT-IDENTICAL to the plain kernel — same ascending
            // column order, so no accumulator is reassociated; the SLM is a
            // cache, not a change of algorithm.  $IE_DS4_ATTN_TILE=0 restores
            // the plain shape.
            // DEFAULT OFF — MEASURED AND FALSIFIED 2026-08-09.  Real model,
            // clean runs: pp512 187-194 -> 148.7, pp2048 212-215 -> 159.5.
            // WHY IT LOSES, and it is structural rather than a tuning miss:
            // this attention is SPARSE.  The plain kernel tests the mask FIRST
            // and never touches a masked column's K row — at ctx 4096 a ratio-4
            // layer masks ~44% of columns, and the indexer exists precisely so
            // attention can skip the rest.  A shared tile must stage EVERY
            // column of the tile because the TQ query tokens sharing it have
            // DIFFERENT live sets (the union approaches "all columns" as TQ
            // grows), so tiling trades a 45% read saving for a reuse factor it
            // never gets to collect, and adds two barriers per tile.
            // FlashAttention-style tiling assumes DENSE attention; this one is
            // not dense, which is the whole point of the lightning indexer.
            // Kept as an instrument ($IE_DS4_ATTN_TILE=1) because it is
            // bit-identical (verified byte-for-byte over 6 shapes) and would
            // become the right shape for any DENSE-mask call site.
            static const bool tile_on = [] {
                const char* e = std::getenv("IE_DS4_ATTN_TILE");
                return e && *e && std::string(e) != "0";
            }();
            const uint64_t slm_sz =
                q.get_device().get_info<sycl::info::device::local_mem_size>();
            const size_t max_wg2 =
                q.get_device().get_info<sycl::info::device::max_work_group_size>();
            // DEFAULT OFF — MEASURED AND FALSIFIED 2026-08-09 (the third
            // attention experiment to fail, and the three together bound the
            // problem).  pp512 195.8 -> 191.4, pp2048 223.7 -> 212.0.
            // WHY: batching breaks the plain kernel's L1 REUSE.  There, a
            // column's K row is read for the dot product and then immediately
            // again for the accumulator update — back to back, so the second
            // read is hot.  Batching NB columns puts NB-1 other rows between
            // those two reads, and at head_dim 512 (2 KB/row) they evict it.
            // The reduction latency this was meant to hide costs less than the
            // cache miss it causes.
            static const bool batch_on = [] {
                const char* e = std::getenv("IE_DS4_ATTN_BATCH");
                return e && *e && std::string(e) != "0";
            }();
            if (batch_on && !tile_on && T > 16)
                return ds4_attention_batched<4>(q, q_in, kv, mask, sinks, y,
                                                T, n_heads, head_dim, n_kv, scaling, deps);
            if (tile_on && T > 16 &&
                uint64_t(8) * head_dim * sizeof(float) <= slm_sz &&
                size_t(8) * kSG <= max_wg2)
                return ds4_attention_tile<8, 8>(q, q_in, kv, mask, sinks, y,
                                                T, n_heads, head_dim, n_kv, scaling, deps);
            return ds4_attention_plain(q, q_in, kv, mask, sinks, y,
                                       T, n_heads, head_dim, n_kv, scaling, deps);
        }
    }
}

sycl::event ds4_sliding_causal_mask(sycl::queue& q,
                                    const int32_t* positions, float* mask,
                                    uint32_t T, uint32_t n_kv, uint32_t window,
                                    const std::vector<sycl::event>& deps) {
    return ie::ps(q, "ds4_sliding_causal_mask", [&](sycl::handler& h) {
        h.depends_on(deps);
        const uint32_t total  = T * n_kv;
        const uint32_t global = ((total + kWG - 1) / kWG) * kWG;
        const float    lowest = std::numeric_limits<float>::lowest();
        h.parallel_for(sycl::nd_range<1>(global, kWG), [=](sycl::nd_item<1> it) {
            const uint32_t i = uint32_t(it.get_global_id(0));
            if (i >= total) return;
            const int64_t k = int64_t(i % n_kv);
            const int64_t p = int64_t(positions[i / n_kv]);
            const bool ok = (k <= p) && (window == 0u || k > p - int64_t(window));
            mask[i] = ok ? 0.f : lowest;
        });
    });
}

sycl::event ds4_sliding_causal_mask_vis(sycl::queue& q,
                                        const int32_t* positions,
                                        const int32_t* left_add, const int32_t* right,
                                        float* mask,
                                        uint32_t T, uint32_t n_kv, uint32_t window,
                                        const std::vector<sycl::event>& deps) {
    return ie::ps(q, "ds4_sliding_causal_mask", [&](sycl::handler& h) {
        h.depends_on(deps);
        const uint32_t total  = T * n_kv;
        const uint32_t global = ((total + kWG - 1) / kWG) * kWG;
        const float    lowest = std::numeric_limits<float>::lowest();
        h.parallel_for(sycl::nd_range<1>(global, kWG), [=](sycl::nd_item<1> it) {
            const uint32_t i = uint32_t(it.get_global_id(0));
            if (i >= total) return;
            const uint32_t t = i / n_kv;
            const int64_t k = int64_t(i % n_kv);
            const int64_t p = int64_t(positions[t]);
            const bool ok = (k <= p + int64_t(right[t])) &&
                            (window == 0u || k > p - int64_t(window) - int64_t(left_add[t]));
            mask[i] = ok ? 0.f : lowest;
        });
    });
}

// ---------------------------------------------------------------------------
// Compressor window pooling.
// ---------------------------------------------------------------------------
sycl::event ds4_compress_pool(sycl::queue& q,
                              const float* chunk_kv, const float* chunk_gate,
                              const float* position_bias,
                              const float* prior_kv, const float* prior_gate,
                              float* out,
                              uint32_t n_win, uint32_t rate, uint32_t width,
                              bool overlap,
                              const std::vector<sycl::event>& deps) {
    return ie::ps(q, "ds4_compress_pool", [&](sycl::handler& h) {
        h.depends_on(deps);
        const uint32_t src_w  = overlap ? 2u * width : width;
        const uint32_t slots  = overlap ? 2u * rate : rate;
        const uint32_t total  = n_win * width;
        const uint32_t global = ((total + kWG - 1) / kWG) * kWG;
        h.parallel_for(sycl::nd_range<1>(global, kWG), [=](sycl::nd_item<1> it) {
            const uint32_t i = uint32_t(it.get_global_id(0));
            if (i >= total) return;
            const uint32_t c = i % width;    // channel — the softmax is per channel
            const uint32_t w = i / width;    // window

            // Pass 1: running max of the gate logits.
            float mx = kNegInf;
            for (uint32_t j = 0; j < slots; ++j) {
                float g;
                if (!overlap) {
                    g = chunk_gate[(size_t(w) * rate + j) * src_w + c]
                      + position_bias[size_t(j) * src_w + c];
                } else if (j < rate) {
                    if (w > 0) {
                        g = chunk_gate[(size_t(w - 1) * rate + j) * src_w + c]
                          + position_bias[size_t(j) * src_w + c];
                    } else if (prior_gate) {
                        // Slot j of the Ca half always takes bias row j,
                        // columns [0, width) — whether the row came from this
                        // call's window w-1 or from the previous call's saved
                        // slice.  See the header note on why the cache stores
                        // the RAW gate rather than the reference's bias-added one.
                        g = prior_gate[size_t(j) * width + c]
                          + position_bias[size_t(j) * src_w + c];
                    } else {
                        g = kNegInf;
                    }
                } else {
                    const uint32_t jj = j - rate;
                    g = chunk_gate[(size_t(w) * rate + jj) * src_w + width + c]
                      + position_bias[size_t(jj) * src_w + width + c];
                }
                mx = sycl::fmax(mx, g);
            }

            // Pass 2: softmax over the slot axis (fp32, as the reference forces
            // with `dtype=torch.float32`) and the gated sum.
            float den = 0.f, num = 0.f;
            for (uint32_t j = 0; j < slots; ++j) {
                float g, v;
                if (!overlap) {
                    g = chunk_gate[(size_t(w) * rate + j) * src_w + c]
                      + position_bias[size_t(j) * src_w + c];
                    v = chunk_kv[(size_t(w) * rate + j) * src_w + c];
                } else if (j < rate) {
                    if (w > 0) {
                        g = chunk_gate[(size_t(w - 1) * rate + j) * src_w + c]
                          + position_bias[size_t(j) * src_w + c];
                        v = chunk_kv[(size_t(w - 1) * rate + j) * src_w + c];
                    } else if (prior_gate) {
                        g = prior_gate[size_t(j) * width + c]
                          + position_bias[size_t(j) * src_w + c];
                        v = prior_kv[size_t(j) * width + c];
                    } else {
                        g = kNegInf; v = 0.f;
                    }
                } else {
                    const uint32_t jj = j - rate;
                    g = chunk_gate[(size_t(w) * rate + jj) * src_w + width + c]
                      + position_bias[size_t(jj) * src_w + width + c];
                    v = chunk_kv[(size_t(w) * rate + jj) * src_w + width + c];
                }
                const float e = (g > kNegInf) ? sycl::exp(g - mx) : 0.f;
                den += e;
                num += e * v;
            }
            out[i] = (den > 0.f) ? (num / den) : 0.f;
        });
    });
}

// ---------------------------------------------------------------------------
// Lightning Indexer scorer — DeepseekV4IndexerScorer.forward (modeling:455).
//
// TWO SHAPES, one dispatcher (ds4_indexer_score, at the bottom of this block).
//
//   DIMLANE  the original: one sub-group per (query, key), the head loop serial
//            inside it, and the head_dim dot product split ACROSS the lanes —
//            so every head costs a sub-group reduction.  At the decode shape
//            (index_n_heads 64) that is 64 reduce-and-shuffle round trips per
//            key, each a 4-deep dependent shuffle chain, for 8 fused multiply
//            adds of actual work apiece.
//   HEADLANE lane l owns whole HEADS (l, l+kSG, l+2*kSG, ...) and walks the
//            entire head_dim itself, so there is exactly ONE sub-group
//            reduction per key instead of n_heads.  Same fused-multiply-add
//            count, same memory, 64x fewer cross-lane reductions.
//
// Measured on B70 (best of three runs of 60), index_n_heads 64, head_dim 128:
//     T=1  n_keys 1024    dimlane 0.0633 ms   headlane 0.0108 ms   5.9x
//     T=1  n_keys 4096    dimlane 0.0961 ms   headlane 0.0250 ms   3.8x
//     T=4  n_keys 1024    dimlane 0.1277 ms   headlane 0.0291 ms   4.4x
//     T=4  n_keys 4096    dimlane 0.3793 ms   headlane 0.0896 ms   4.2x
//
// FALSIFIED, so it is not here.  (a) The old shape is NOT dispatch-starved:
// packing its one-sub-group work-groups into work-groups of 2/4/8/16/32/64
// sub-groups is bit-identical AND changes the time by less than 1% at every
// size, so the 1024 tiny work-groups were never the problem.  (b) Staging q
// TRANSPOSED through SLM so the per-lane read is contiguous across lanes is
// SLOWER than reading it straight from global (0.0136 vs 0.0108 ms at n_keys
// 1024): q is 32 KB, it is resident in L1 after the first key, and the SLM
// staging costs more than the gather it removes.
//
// NOT BIT-IDENTICAL, AND WHY.  Two reassociations, in opposite directions:
//   * the dot product moves from "8-long per-lane chain + 4-level tree" to
//     "four 32-long chains + 2 levels", which is LESS accurate;
//   * the sum over heads moves from one 64-long serial chain to "4-long
//     per-lane chain + 4-level tree", which is MORE accurate — and it is the
//     longer of the two chains, so the net is a wash.
// deepseek4_decode_gate_test §8 measures both against a double-precision host
// reference and requires the new shape to be no worse, and §9 measures what
// actually matters downstream: whether ds4_indexer_topk still selects the same
// keys.
// ---------------------------------------------------------------------------
namespace {

template <typename TK>
sycl::event ds4_indexer_score_dimlane(sycl::queue& q,
                                      const float* q_in, const TK* keys,
                                      const float* w_proj, float* scores,
                                      uint32_t T, uint32_t n_heads, uint32_t head_dim,
                                      uint32_t n_keys,
                                      float softmax_scale, float weights_scaling,
                                      const std::vector<sycl::event>& deps) {
    return ie::ps(q, "ds4_indexer_score", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<2>({T, size_t(n_keys) * kSG}, {1, kSG}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(kSG)]] {
            const uint32_t t    = uint32_t(it.get_group(0));
            const uint32_t e    = uint32_t(it.get_group(1));
            const uint32_t lane = uint32_t(it.get_local_id(1));
            auto sg = it.get_sub_group();
            const TK* krow = keys + size_t(e) * head_dim;

            float total = 0.f;
            for (uint32_t hh = 0; hh < n_heads; ++hh) {
                const float* qrow = q_in + (size_t(t) * n_heads + hh) * head_dim;
                float part = 0.f;
                for (uint32_t d = lane; d < head_dim; d += kSG) part += qrow[d] * float(krow[d]);
                const float dot = sycl::reduce_over_group(sg, part, sycl::plus<float>());
                // scores = relu(q·K) * softmax_scale ; weights = w_proj * weights_scaling
                total += sycl::fmax(dot, 0.f) * softmax_scale
                       * (w_proj[size_t(t) * n_heads + hh] * weights_scaling);
            }
            if (lane == 0) scores[size_t(t) * n_keys + e] = total;
        });
    });
}

// SGS keys per work-group (one per sub-group); lane l owns heads l, l+kSG, ...
template <uint32_t SGS, typename TK, bool Keep = false>
sycl::event ds4_indexer_score_headlane(sycl::queue& q,
                                       const float* q_in, const TK* keys,
                                       const float* w_proj, float* scores,
                                       uint32_t T, uint32_t n_heads, uint32_t head_dim,
                                       uint32_t n_keys,
                                       float softmax_scale, float weights_scaling,
                                       const std::vector<sycl::event>& deps,
                                       const uint8_t* keep = nullptr, uint32_t keep_block = 1, float masked = 0.f) {
    constexpr uint32_t WG = SGS * kSG;
    const uint32_t keep_nb = (n_keys + keep_block - 1u) / keep_block;
    const uint32_t n_grp = (n_keys + SGS - 1u) / SGS;
    return ie::ps(q, "ds4_indexer_score", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<2>({T, size_t(n_grp) * WG}, {1, WG}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(kSG)]] {
            const uint32_t t = uint32_t(it.get_group(0));
            auto sg = it.get_sub_group();
            // `e` is uniform inside the sub-group, so this early return retires
            // whole sub-groups and never leaves the reduce below unreached.
            const uint32_t e = uint32_t(it.get_group(1)) * SGS
                             + uint32_t(sg.get_group_linear_id());
            if (e >= n_keys) return;
            const uint32_t lane = uint32_t(sg.get_local_linear_id());
            // Keep (V4.1 Phase 45): a key outside the row's candidate blocks takes the value the mask would give it,
            // and its dot products are never run -- `e` is uniform in the sub-group, so this retires it whole
            if (Keep && !keep[size_t(t) * keep_nb + e / keep_block]) { if (lane == 0) scores[size_t(t) * n_keys + e] = masked; return; }
            const TK* krow = keys + size_t(e) * head_dim;

            float tot = 0.f;
            // The loop bound is uniform; only the CONTRIBUTION is predicated, so
            // n_heads that is not a multiple of kSG costs idle lanes, not
            // divergent control flow.
            for (uint32_t hb = 0; hb < n_heads; hb += kSG) {
                const uint32_t hh = hb + lane;
                float dot = 0.f;
                if (hh < n_heads) {
                    const float* qrow = q_in + (size_t(t) * n_heads + hh) * head_dim;
                    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
                    uint32_t d = 0;
                    for (; d + 3u < head_dim; d += 4u) {
                        a0 += qrow[d]      * float(krow[d]);
                        a1 += qrow[d + 1u] * float(krow[d + 1u]);
                        a2 += qrow[d + 2u] * float(krow[d + 2u]);
                        a3 += qrow[d + 3u] * float(krow[d + 3u]);
                    }
                    for (; d < head_dim; ++d) a0 += qrow[d] * float(krow[d]);
                    dot = (a0 + a1) + (a2 + a3);
                    tot += sycl::fmax(dot, 0.f) * softmax_scale
                         * (w_proj[size_t(t) * n_heads + hh] * weights_scaling);
                }
            }
            const float r = sycl::reduce_over_group(sg, tot, sycl::plus<float>());
            if (lane == 0) scores[size_t(t) * n_keys + e] = r;
        });
    });
}

// Two keys share each query vector load. Keep the four dot-product chains and
// head-lane reduction of headlane; explicit FMA preserves its contracted math.
// The dispatcher guarantees aligned vec4 rows and head_dim divisible by four.
template <typename TK, bool Keep = false>
sycl::event ds4_indexer_score_keytile(sycl::queue& q,
                                     const float* q_in, const TK* keys,
                                     const float* w_proj, float* scores,
                                     uint32_t T, uint32_t n_heads, uint32_t head_dim,
                                     uint32_t n_keys,
                                     float softmax_scale, float weights_scaling,
                                     const std::vector<sycl::event>& deps,
                                     const uint8_t* keep = nullptr, uint32_t keep_block = 2, float masked = 0.f) {
    constexpr uint32_t KEYS = 2, SGS = 16, WG = SGS * kSG;
    const uint32_t keep_nb = (n_keys + keep_block - 1u) / keep_block;   // keep_block is even: a key pair shares its block
    const size_t n_grp = (size_t(n_keys) + KEYS * SGS - 1) / (KEYS * SGS);
    return ie::ps(q, "ds4_indexer_score", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<2>({T, n_grp * WG}, {1, WG}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(kSG)]] {
            auto sg = it.get_sub_group();
            const uint32_t t = uint32_t(it.get_group(0));
            const uint32_t lane = uint32_t(sg.get_local_linear_id());
            const size_t e0 = (it.get_group(1) * SGS + sg.get_group_linear_id()) * KEYS;
            if (e0 >= n_keys) return; // Uniform across the whole sub-group.
            if (Keep && !keep[size_t(t) * keep_nb + e0 / keep_block]) {   // V4.1 Phase 45: the pair is outside the candidates
                if (sg.leader()) for (uint32_t e = 0; e < KEYS; ++e) if (e0 + e < n_keys) scores[size_t(t) * n_keys + e0 + e] = masked;
                return;
            }
            float total[KEYS] = {};
            for (uint32_t hb = 0; hb < n_heads; hb += kSG) {
                const uint32_t hh = hb + lane;
                if (hh < n_heads) {
                    const float* qrow = q_in + (size_t(t) * n_heads + hh) * head_dim;
                    float sums[KEYS][4] = {};
                    for (uint32_t d = 0; d < head_dim; d += 4) {
                        sycl::vec<float, 4> qv;
                        qv.load(0, static_cast<const float*>(__builtin_assume_aligned(
                            qrow + d, alignof(sycl::vec<float, 4>))));
                        #pragma unroll
                        for (uint32_t e = 0; e < KEYS; ++e) {
                            if (e0 + e < n_keys) {
                                sycl::vec<TK, 4> kv;
                                kv.load(0, static_cast<const TK*>(__builtin_assume_aligned(
                                    keys + (e0 + e) * head_dim + d, alignof(sycl::vec<TK, 4>))));
                                #pragma unroll
                                for (uint32_t p = 0; p < 4; ++p)
                                    sums[e][p] = sycl::fma(qv[p], float(kv[p]), sums[e][p]);
                            }
                        }
                    }
                    #pragma unroll
                    for (uint32_t e = 0; e < KEYS; ++e) {
                        const float dot = (sums[e][0] + sums[e][1]) + (sums[e][2] + sums[e][3]);
                        total[e] = sycl::fma(sycl::fmax(dot, 0.f) * softmax_scale,
                                            w_proj[size_t(t) * n_heads + hh] * weights_scaling,
                                            total[e]);
                    }
                }
            }
            #pragma unroll
            for (uint32_t e = 0; e < KEYS; ++e) {
                const float r = sycl::reduce_over_group(sg, total[e], sycl::plus<float>());
                if (sg.leader() && e0 + e < n_keys)
                    scores[size_t(t) * n_keys + e0 + e] = r;
            }
        });
    });
}

template <typename TK>
sycl::event ds4_indexer_score_any(sycl::queue& q,
                                  const float* q_in, const TK* keys, const float* w_proj,
                                  float* scores,
                                  uint32_t T, uint32_t n_heads, uint32_t head_dim,
                                  uint32_t n_keys,
                                  float softmax_scale, float weights_scaling,
                                  const std::vector<sycl::event>& deps,
                                  const uint8_t* keep = nullptr, uint32_t keep_block = 0, float masked = 0.f) {
    if (keep) {   // V4.1 Phase 45: the two shapes that serve V4.1 (index_n_heads 32), with the candidate skip
        if (n_heads < uint32_t(kSG) || keep_block == 0 || keep_block % 2)
            throw std::invalid_argument("ds4_indexer_score (candidates): needs n_heads >= 16 and an even block size");
        if (T >= 64 && head_dim >= 64 && head_dim % 4 == 0 && n_keys >= 32
            && reinterpret_cast<uintptr_t>(q_in) % alignof(sycl::vec<float, 4>) == 0
            && reinterpret_cast<uintptr_t>(keys) % alignof(sycl::vec<TK, 4>) == 0)
            return ds4_indexer_score_keytile<TK, true>(q, q_in, keys, w_proj, scores, T, n_heads, head_dim, n_keys,
                                                      softmax_scale, weights_scaling, deps, keep, keep_block, masked);
        return ds4_indexer_score_headlane<16, TK, true>(q, q_in, keys, w_proj, scores, T, n_heads, head_dim, n_keys,
                                                       softmax_scale, weights_scaling, deps, keep, keep_block, masked);
    }
    if (T >= 64 && n_heads >= uint32_t(kSG) && head_dim >= 64 && head_dim % 4 == 0
        && n_keys >= 32
        && reinterpret_cast<uintptr_t>(q_in) % alignof(sycl::vec<float, 4>) == 0
        && reinterpret_cast<uintptr_t>(keys) % alignof(sycl::vec<TK, 4>) == 0)
        return ds4_indexer_score_keytile(q, q_in, keys, w_proj, scores, T, n_heads,
                                        head_dim, n_keys, softmax_scale, weights_scaling, deps);
    // Below kSG heads the head-lane shape would leave most of the sub-group
    // idle for the whole head_dim walk, which costs more than the reductions it
    // saves.  V4-Flash has index_n_heads 64, so decode always takes the fast
    // shape; the guard exists for the small shapes the unit tests drive.
    if (n_heads >= uint32_t(kSG))
        return ds4_indexer_score_headlane<16, TK>(q, q_in, keys, w_proj, scores, T, n_heads,
                                                  head_dim, n_keys, softmax_scale,
                                                  weights_scaling, deps);
    return ds4_indexer_score_dimlane<TK>(q, q_in, keys, w_proj, scores, T, n_heads, head_dim,
                                         n_keys, softmax_scale, weights_scaling, deps);
}

}  // namespace

sycl::event ds4_indexer_score(sycl::queue& q,
                              const float* q_in, const float* keys, const float* w_proj,
                              float* scores,
                              uint32_t T, uint32_t n_heads, uint32_t head_dim,
                              uint32_t n_keys,
                              float softmax_scale, float weights_scaling,
                              const std::vector<sycl::event>& deps) {
    return ds4_indexer_score_any<float>(q, q_in, keys, w_proj, scores, T, n_heads, head_dim,
                                        n_keys, softmax_scale, weights_scaling, deps);
}

sycl::event ds4_indexer_score_candidates(sycl::queue& q,
                                         const float* q_in, const float* keys, const float* w_proj,
                                         float* scores,
                                         uint32_t T, uint32_t n_heads, uint32_t head_dim,
                                         uint32_t n_keys,
                                         float softmax_scale, float weights_scaling,
                                         const uint8_t* keep, uint32_t keep_block, float masked,
                                         const std::vector<sycl::event>& deps) {
    return ds4_indexer_score_any<float>(q, q_in, keys, w_proj, scores, T, n_heads, head_dim,
                                        n_keys, softmax_scale, weights_scaling, deps, keep, keep_block, masked);
}

sycl::event ds4_indexer_score(sycl::queue& q,
                              const float* q_in, const sycl::half* keys, const float* w_proj,
                              float* scores,
                              uint32_t T, uint32_t n_heads, uint32_t head_dim,
                              uint32_t n_keys,
                              float softmax_scale, float weights_scaling,
                              const std::vector<sycl::event>& deps) {
    return ds4_indexer_score_any<sycl::half>(q, q_in, keys, w_proj, scores, T, n_heads, head_dim,
                                             n_keys, softmax_scale, weights_scaling, deps);
}

// ---------------------------------------------------------------------------
// Indexer top-k.  Two implementations of ONE specification, and the
// specification is exact: the packed keys are pairwise distinct (the entry
// index occupies the low 32 bits), so "the j-th largest packed key" is a
// uniquely determined value and both routines return it.  This is not a
// tolerance question and there is no tolerance below — the two shapes are
// required to agree EXACTLY, and deepseek4_decode_gate_test checks that they
// do, index for index.
//
//   SCAN     the original: `top_k` rounds, each a full work-group scan of all
//            n_keys followed by a work-group reduction.  At index_topk = 512
//            that is 512 dependent reduce-and-barrier round trips inside a
//            SINGLE work-group (the launch is one work-group per query token,
//            so decode runs exactly one), each re-reading the score row from
//            global memory.
//   BITONIC  sort the whole padded key array once in SLM, then read the top
//            `top_k` off the end.  55 barriers at n_keys 1024, 78 at 4096,
//            versus 512 — and the scores are read once, not 512 times.
//
// Measured on B70 (best of three runs of 100), index_topk 512:
//     T=1 n_keys=1024   scan 0.998 ms   bitonic 0.012 ms   83.8x
//     T=1 n_keys=4096   scan 2.171 ms   bitonic 0.043 ms   50.7x
//     T=64 n_keys=1024  scan 0.984 ms   bitonic 0.023 ms   42.1x
//
// PADDING CAN NEVER BE SELECTED.  The sort runs over N = next power of two
// >= n_keys, with slots [n_keys, N) filled by pack_key(-inf, e).  For any real
// entry r < n_keys and any pad p >= n_keys: if score[r] > -inf then key(r) has
// the larger high word; if score[r] == -inf (masked by the causal threshold)
// the high words tie and the low word decides, and 0xFFFFFFFF-r > 0xFFFFFFFF-p
// because r < p.  Either way key(r) > key(p), so all n_keys real entries sort
// strictly above every pad, and top_k = min(index_topk, n_keys) <= n_keys only
// ever reaches real ones.  (This matters: an out-of-range index here would be
// scattered straight into ds4_block_bias_topk's bias row.)
// ---------------------------------------------------------------------------
namespace {

sycl::event ds4_indexer_topk_scan(sycl::queue& q,
                                  const float* scores, const int32_t* positions,
                                  int32_t* out,
                                  uint32_t T, uint32_t n_keys, uint32_t top_k,
                                  uint32_t compress_rate,
                                  const std::vector<sycl::event>& deps) {
    return ie::ps(q, "ds4_indexer_topk", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(size_t(T) * kWG, kWG),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
            const uint32_t t   = uint32_t(it.get_group(0));
            const uint32_t lid = uint32_t(it.get_local_id(0));
            const auto grp = it.get_group();
            // causal_threshold = (position_ids + 1) // compress_rate (modeling:578)
            const int64_t thr = (int64_t(positions[t]) + 1) / int64_t(compress_rate);

            uint64_t prev = ~uint64_t(0);
            for (uint32_t j = 0; j < top_k; ++j) {
                uint64_t best = 0;
                for (uint32_t e = lid; e < n_keys; e += kWG) {
                    // entries at or beyond the threshold are masked to -inf
                    // before the topk, exactly as the reference does.
                    const float v = (int64_t(e) >= thr) ? kNegInf
                                                        : scores[size_t(t) * n_keys + e];
                    const uint64_t k = pack_key(v, e);
                    if (k < prev && k > best) best = k;
                }
                best = sycl::reduce_over_group(grp, best, sycl::maximum<uint64_t>());
                const int32_t idx = int32_t(0xFFFFFFFFu - uint32_t(best & 0xFFFFFFFFu));
                if (lid == 0)
                    out[size_t(t) * top_k + j] = (int64_t(idx) >= thr) ? -1 : idx;
                prev = best;
            }
        });
    });
}

template <uint32_t WG>
sycl::event ds4_indexer_topk_bitonic(sycl::queue& q,
                                     const float* scores, const int32_t* positions,
                                     int32_t* out,
                                     uint32_t T, uint32_t n_keys, uint32_t top_k,
                                     uint32_t compress_rate, uint32_t N,
                                     const std::vector<sycl::event>& deps) {
    return ie::ps(q, "ds4_indexer_topk", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<uint64_t, 1> buf(sycl::range<1>(N), h);
        h.parallel_for(sycl::nd_range<1>(size_t(T) * WG, WG),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
            const uint32_t t   = uint32_t(it.get_group(0));
            const uint32_t lid = uint32_t(it.get_local_id(0));
            const int64_t thr = (int64_t(positions[t]) + 1) / int64_t(compress_rate);

            for (uint32_t e = lid; e < N; e += WG) {
                const float v = (e < n_keys && int64_t(e) < thr)
                                    ? scores[size_t(t) * n_keys + e] : kNegInf;
                buf[e] = pack_key(v, e);
            }
            it.barrier(sycl::access::fence_space::local_space);

            // Textbook bitonic sort, ascending.  Every work-item reaches every
            // barrier: the compare-exchange loop is inside, the barrier is not.
            for (uint32_t k = 2; k <= N; k <<= 1) {
                for (uint32_t j = k >> 1; j > 0; j >>= 1) {
                    for (uint32_t i = lid; i < N; i += WG) {
                        const uint32_t ixj = i ^ j;
                        if (ixj > i) {
                            const bool up = ((i & k) == 0);
                            const uint64_t a = buf[i], b = buf[ixj];
                            if ((a > b) == up) { buf[i] = b; buf[ixj] = a; }
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }
            }

            // Descending read-out: the j-th largest key sits at N-1-j.
            for (uint32_t j = lid; j < top_k; j += WG) {
                const int32_t idx =
                    int32_t(0xFFFFFFFFu - uint32_t(buf[N - 1 - j] & 0xFFFFFFFFu));
                out[size_t(t) * top_k + j] = (int64_t(idx) >= thr) ? -1 : idx;
            }
        });
    });
}


template <uint32_t WG, uint32_t NN>
sycl::event ds4_indexer_topk_bitonic_fixed(sycl::queue& q,
                                     const float* scores, const int32_t* positions,
                                     int32_t* out,
                                     uint32_t T, uint32_t n_keys, uint32_t top_k,
                                     uint32_t compress_rate,
                                     const std::vector<sycl::event>& deps) {
    constexpr uint32_t N = NN;
    constexpr uint32_t SG = 16;
    return ie::ps(q, "ds4_indexer_topk", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<uint64_t, 1> buf(sycl::range<1>(N), h);
        h.parallel_for(sycl::nd_range<1>(size_t(T) * WG, WG),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint32_t t   = uint32_t(it.get_group(0));
            const uint32_t lid = uint32_t(it.get_local_id(0));
            const int64_t thr = (int64_t(positions[t]) + 1) / int64_t(compress_rate);

            for (uint32_t e = lid; e < N; e += WG) {
                const float v = (e < n_keys && int64_t(e) < thr)
                                    ? scores[size_t(t) * n_keys + e] : kNegInf;
                buf[e] = pack_key(v, e);
            }
            it.barrier(sycl::access::fence_space::local_space);

            // Fixed padded extent makes every sorting stage a compile-time
            // distance. Long exchanges use SLM; the final four distances stay
            // in subgroup registers. Packed uint64 keys preserve exact ties.
            #pragma unroll
            for (uint32_t k = 2; k <= N; k <<= 1) {
                uint32_t j = k >> 1;
                #pragma unroll
                for (; j >= SG; j >>= 1) {
                    for (uint32_t i = lid; i < N; i += WG) {
                        const uint32_t ixj = i ^ j;
                        if (ixj > i) {
                            const bool up = (i & k) == 0;
                            const uint64_t a = buf[i], b = buf[ixj];
                            if ((a > b) == up) { buf[i] = b; buf[ixj] = a; }
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }
                for (uint32_t i = lid; i < N; i += WG) {
                    uint64_t v = buf[i];
                    #pragma unroll
                    for (uint32_t d = SG / 2; d; d >>= 1) {
                        if (d > j) continue;
                        const uint64_t other = sycl::permute_group_by_xor(it.get_sub_group(), v, d);
                        const bool take_min = ((i & k) == 0) == ((i & d) == 0);
                        v = take_min ? sycl::min(v, other) : sycl::max(v, other);
                    }
                    buf[i] = v;
                }
                it.barrier(sycl::access::fence_space::local_space);
            }

            // Descending read-out: the j-th largest key sits at N-1-j.
            for (uint32_t j = lid; j < top_k; j += WG) {
                const int32_t idx =
                    int32_t(0xFFFFFFFFu - uint32_t(buf[N - 1 - j] & 0xFFFFFFFFu));
                out[size_t(t) * top_k + j] = (int64_t(idx) >= thr) ? -1 : idx;
            }
        });
    });
}

}  // namespace

sycl::event ds4_indexer_topk(sycl::queue& q,
                             const float* scores, const int32_t* positions,
                             int32_t* out,
                             uint32_t T, uint32_t n_keys, uint32_t index_topk,
                             uint32_t compress_rate,
                             const std::vector<sycl::event>& deps) {
    const uint32_t top_k = (index_topk < n_keys) ? index_topk : n_keys;
    // Phase 39 (docs/deepseek41/79): decode rows over more than 8,192 keys take the two-level split -- chunks sorted
    // side by side, then one sort over the chunk winners -- the same picks in the same order, bit for bit.
    if (ds41_topk_split_wanted(T, n_keys, top_k))
        return ds41_indexer_topk_split(q, scores, positions, out, T, n_keys, top_k, compress_rate, deps);

    uint32_t N = 1u;
    while (N < n_keys) N <<= 1;
    uint32_t K2 = 1u;
    while (K2 < top_k) K2 <<= 1;
    const sycl::device dev = q.get_device();
    const uint64_t slm    = dev.get_info<sycl::info::device::local_mem_size>();
    const size_t   max_wg = dev.get_info<sycl::info::device::max_work_group_size>();

    // Shape choice, in preference order (IE_DS4_TOPK=scan|bitonic|radix forces
    // one for A/B):
    //   radix   O(n) passes, CONSTANT SLM — the only shape alive at ctx 200k
    //           (n_keys 50000: bitonic needs 512 KB of SLM, scan does 512 full
    //           rescans). Used once the full padded bitonic sort cannot fit;
    //           it performs 8 histogram passes plus a top_k-sized sort.
    //   bitonic full padded sort in SLM — best at small n_keys.
    //   scan    O(top_k * n_keys), needs nothing — the fallback of last resort.
    static const int forced = [] {
        const char* e = std::getenv("IE_DS4_TOPK");
        if (!e || !*e) return 0;
        const std::string s(e);
        return s == "scan" ? 1 : s == "bitonic" ? 2 : s == "radix" ? 3 : 0;
    }();
    const bool bitonic_fits = uint64_t(N) * sizeof(uint64_t) <= slm && max_wg >= 256;
    const bool radix_fits   = max_wg >= 256 &&
                              (256u * 4u + uint64_t(K2) * 8u + 8u) <= slm;
    // Radix only where the bitonic cannot fit: at shapes both can run the
    // bitonic measures faster (n_keys 4096: 0.099 ms vs radix 0.110 ms — the
    // atomic-free rotation has a fixed serialization cost), and past the SLM
    // wall the alternative was the O(top_k*n) scan, which radix beats by
    // orders of magnitude.  2026-08-09.
    const bool want_radix = forced == 3 || (forced == 0 && radix_fits && !bitonic_fits);

    if (want_radix && radix_fits) {
        // WG=256 ONLY.  At WG=1024 (64 sub-groups at SIMD16) the SLM ATOMIC
        // histogram updates are silently lost on this stack — measured
        // 2026-08-09 with a standalone probe: identical kernel, WG=256 selects
        // the exact k-th key, WG=1024 returns an all-zero histogram (the
        // bitonic at WG=1024 is fine, so it is local atomics specifically,
        // not SLM or the launch).  256 lanes are ample for histogram passes.
        return ds4_indexer_topk_radix256(q, scores, positions, out, T, n_keys,
                                         top_k, compress_rate, K2, deps);
    }
    if (forced != 1 && bitonic_fits) {
        // Measured B70 shapes: WG256 packs prefill better through N=4096;
        // decode and N=8192 need WG1024. Other devices keep the generic path.
        struct Choice { sycl::device device; bool fixed; };
        static thread_local std::optional<Choice> choice;
        if (!choice || choice->device != dev) {
            const bool fixed = dev.get_backend() == sycl::backend::ext_oneapi_level_zero &&
                dev.is_gpu() && dev.get_info<sycl::info::device::vendor_id>() == 0x8086 &&
                dev.get_info<sycl::info::device::name>().find("B70") != std::string::npos;
            choice.emplace(Choice{dev, fixed});
        }
        if (choice->fixed && max_wg >= 1024) {
            const bool compact = N == 256 || (T >= 128 && N <= 4096);
            // Only measured padded extents; tiny and very large rows retain
            // the original bitonic/radix/scan fallback and forced-path controls.
            auto launch_fixed = [&]<uint32_t NN>() {
                if (compact)
                    return ds4_indexer_topk_bitonic_fixed<256, NN>(q, scores, positions,
                        out, T, n_keys, top_k, compress_rate, deps);
                return ds4_indexer_topk_bitonic_fixed<1024, NN>(q, scores, positions,
                    out, T, n_keys, top_k, compress_rate, deps);
            };
            switch (N) {
                case 256:  return launch_fixed.operator()<256>();
                case 1024: return launch_fixed.operator()<1024>();
                case 2048: return launch_fixed.operator()<2048>();
                case 4096: return launch_fixed.operator()<4096>();
                case 8192: return launch_fixed.operator()<8192>();
            }
        }
        if (max_wg >= 1024)
            return ds4_indexer_topk_bitonic<1024>(q, scores, positions, out, T, n_keys,
                                                  top_k, compress_rate, N, deps);
        return ds4_indexer_topk_bitonic<256>(q, scores, positions, out, T, n_keys,
                                             top_k, compress_rate, N, deps);
    }
    return ds4_indexer_topk_scan(q, scores, positions, out, T, n_keys, top_k,
                                 compress_rate, deps);
}

sycl::event ds4_block_bias_dense(sycl::queue& q,
                                 const int32_t* positions, float* bias,
                                 uint32_t T, uint32_t n_keys, uint32_t compress_rate,
                                 const std::vector<sycl::event>& deps) {
    return ie::ps(q, "ds4_block_bias_dense", [&](sycl::handler& h) {
        h.depends_on(deps);
        const uint32_t total  = T * n_keys;
        const uint32_t global = ((total + kWG - 1) / kWG) * kWG;
        h.parallel_for(sycl::nd_range<1>(global, kWG), [=](sycl::nd_item<1> it) {
            const uint32_t i = uint32_t(it.get_global_id(0));
            if (i >= total) return;
            const int64_t e   = int64_t(i % n_keys);
            const int64_t thr = (int64_t(positions[i / n_keys]) + 1) / int64_t(compress_rate);
            bias[i] = (e >= thr) ? kNegInf : 0.f;
        });
    });
}

sycl::event ds4_block_bias_topk(sycl::queue& q,
                                const int32_t* top_k_indices, float* bias,
                                uint32_t T, uint32_t n_keys, uint32_t top_k,
                                const std::vector<sycl::event>& deps) {
    auto fill = ie::ps(q, "ds4_block_bias_topk_fill", [&](sycl::handler& h) {
        h.depends_on(deps);
        const uint32_t total  = T * n_keys;
        const uint32_t global = ((total + kWG - 1) / kWG) * kWG;
        h.parallel_for(sycl::nd_range<1>(global, kWG), [=](sycl::nd_item<1> it) {
            const uint32_t i = uint32_t(it.get_global_id(0));
            if (i < total) bias[i] = kNegInf;
        });
    });
    return ie::ps(q, "ds4_block_bias_topk_scatter", [&](sycl::handler& h) {
        h.depends_on(fill);
        const uint32_t total  = T * top_k;
        const uint32_t global = ((total + kWG - 1) / kWG) * kWG;
        h.parallel_for(sycl::nd_range<1>(global, kWG), [=](sycl::nd_item<1> it) {
            const uint32_t i = uint32_t(it.get_global_id(0));
            if (i >= total) return;
            const int32_t idx = top_k_indices[i];
            if (idx < 0) return;                    // the -1 sentinel is dropped
            bias[size_t(i / top_k) * n_keys + uint32_t(idx)] = 0.f;
        });
    });
}

}  // namespace ie
