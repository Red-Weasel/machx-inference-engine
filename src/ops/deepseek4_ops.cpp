// src/ops/deepseek4_ops.cpp — DeepSeek-V4 compute primitives (Phase 2).
//
// Every routine here is a line-by-line transcription of
// transformers/models/deepseek_v4/modeling_deepseek_v4.py.  Where the order of
// operations is observable in fp32 (softmax max-subtraction, Sinkhorn
// alternation, the router's bias-on-selection-only rule) the reference order is
// preserved exactly.
//
// Accumulation strategy: every reduction of length N uses 4 independent fp32
// partial accumulators per lane and then a sub-group / work-group tree reduce.
// With SG=16 lanes and 4 partials the longest serial dependency chain is
// N/64 additions instead of N, which is what the parity tolerances are argued
// from (see tests/unit/deepseek4_parity_test.cpp).

#include "ie/deepseek4_ops.hpp"

#include "ie/deepseek4_attn.hpp"   // ds4_rms_norm — the fallback of the fused-norm tail
#include "ie/kernel_profiler.hpp"

#include <sycl/sycl.hpp>

#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ie {

namespace {

constexpr int kSG = 16;   // sub-group size (repo-wide idiom, see elementwise.cpp)
constexpr int kWG = 256;  // 16 sub-groups per work-group

// torch.nn.functional.softplus(x, beta=1, threshold=20): the linear branch
// above the threshold is part of the reference definition, not an optimisation.
inline float softplus_ref(float x) {
    return x > 20.f ? x : sycl::log1p(sycl::exp(x));
}

inline float sigmoid_ref(float x) { return 1.f / (1.f + sycl::exp(-x)); }

// Dot product of two contiguous fp32 rows, split across the `kSG` lanes of one
// sub-group, 4 partial accumulators per lane.  Returns the full sum on every
// lane (reduce_over_group broadcasts).
template <typename SubGroup>
inline float sg_dot(const SubGroup& sg, const float* a, const float* b, uint32_t n) {
    const uint32_t lane = uint32_t(sg.get_local_linear_id());
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    uint32_t i = lane;
    for (; i + 3u * kSG < n; i += 4u * kSG) {
        a0 += a[i]            * b[i];
        a1 += a[i + kSG]      * b[i + kSG];
        a2 += a[i + 2u * kSG] * b[i + 2u * kSG];
        a3 += a[i + 3u * kSG] * b[i + 3u * kSG];
    }
    for (; i < n; i += kSG) a0 += a[i] * b[i];
    return sycl::reduce_over_group(sg, (a0 + a1) + (a2 + a3), sycl::plus<float>());
}

// Same, over the half-open element range [lo, hi) of both rows.  The lane→
// element mapping is `lo + lane + m*kSG`, i.e. identical to sg_dot's whenever
// lo is a multiple of kSG, which is what keeps the loads coalesced when a mix
// row is split across several sub-groups.
template <typename SubGroup>
inline float sg_dot_range(const SubGroup& sg, const float* a, const float* b,
                          uint32_t lo, uint32_t hi) {
    const uint32_t lane = uint32_t(sg.get_local_linear_id());
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    uint32_t i = lo + lane;
    for (; i + 3u * kSG < hi; i += 4u * kSG) {
        a0 += a[i]            * b[i];
        a1 += a[i + kSG]      * b[i + kSG];
        a2 += a[i + 2u * kSG] * b[i + 2u * kSG];
        a3 += a[i + 3u * kSG] * b[i + 3u * kSG];
    }
    for (; i < hi; i += kSG) a0 += a[i] * b[i];
    return sycl::reduce_over_group(sg, (a0 + a1) + (a2 + a3), sycl::plus<float>());
}

// hc_mult == 4 Sinkhorn, entirely in registers.  `m` is a private array indexed
// only by compile-time constants inside fully unrolled loops, so it never
// reaches SLM.  This is the transcription the header always claimed ("the hc×hc
// Sinkhorn loop runs entirely in private registers of a single work-item") but
// which the SLM version below did not actually deliver: 20 iterations × 2
// normalisation passes over a local_accessor is ~1600 dependent SLM round-trips
// on one work-item while the rest of the work-group waits at the barrier.
//
// The arithmetic is the reference's, in the reference's order: column pass
// first, then (iters-1) × (row pass, column pass), each divisor formed as the
// running sum + eps.  Division is kept as division (not a reciprocal multiply)
// so the result matches the SLM path exactly.
inline void ds4_sinkhorn4(float* m, uint32_t iters, float eps) {
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        float cs = 0.f;
        #pragma unroll
        for (int r = 0; r < 4; ++r) cs += m[r * 4 + c];
        cs += eps;
        #pragma unroll
        for (int r = 0; r < 4; ++r) m[r * 4 + c] /= cs;
    }
    for (uint32_t t = 1; t < iters; ++t) {
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            float rs = 0.f;
            #pragma unroll
            for (int c = 0; c < 4; ++c) rs += m[r * 4 + c];
            rs += eps;
            #pragma unroll
            for (int c = 0; c < 4; ++c) m[r * 4 + c] /= rs;
        }
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            float cs = 0.f;
            #pragma unroll
            for (int r = 0; r < 4; ++r) cs += m[r * 4 + c];
            cs += eps;
            #pragma unroll
            for (int r = 0; r < 4; ++r) m[r * 4 + c] /= cs;
        }
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// (b) UnweightedRMSNorm
// ---------------------------------------------------------------------------
sycl::event ds4_unweighted_rms_norm(sycl::queue& q,
                                    const float* x, float* y,
                                    uint32_t n_rows, uint32_t hidden,
                                    float eps,
                                    const std::vector<sycl::event>& deps) {
    return ie::ps(q, "ds4_unweighted_rms_norm", [&](sycl::handler& h) {
        h.depends_on(deps);
        if (hidden == 512) {
            // Same two-term per-lane sum and same workgroup reduction as the
            // generic 512-wide case. Keep both values across the reduction so
            // the output pass does not reload the input, including in-place.
            h.parallel_for(sycl::nd_range<1>(size_t(n_rows) * kWG, kWG),
                           [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
                constexpr uint32_t H = 512;
                const uint32_t lid = uint32_t(it.get_local_id(0));
                const size_t row = it.get_group(0);
                float values[2];
                float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
                for (uint32_t i = lid; i < H; i += kWG) {
                    const float v = x[row * H + i];
                    values[i / kWG] = v;
                    s0 += v * v;
                }
                const float ss = sycl::reduce_over_group(it.get_group(),
                    (s0 + s1) + (s2 + s3), sycl::plus<float>());
                const float r = sycl::rsqrt(ss / float(H) + eps);
                for (uint32_t j = lid; j < H; j += kWG)
                    y[row * H + j] = values[j / kWG] * r;
            });
            return;
        }
        h.parallel_for(sycl::nd_range<1>(size_t(n_rows) * kWG, kWG),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
            const uint32_t row = uint32_t(it.get_group(0));
            const uint32_t lid = uint32_t(it.get_local_id(0));
            const float* xr = x + size_t(row) * hidden;
            float*       yr = y + size_t(row) * hidden;

            float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
            uint32_t i = lid;
            for (; i + 3u * kWG < hidden; i += 4u * kWG) {
                const float v0 = xr[i], v1 = xr[i + kWG];
                const float v2 = xr[i + 2u * kWG], v3 = xr[i + 3u * kWG];
                s0 += v0 * v0; s1 += v1 * v1; s2 += v2 * v2; s3 += v3 * v3;
            }
            for (; i < hidden; i += kWG) { const float v = xr[i]; s0 += v * v; }
            const float ss = sycl::reduce_over_group(it.get_group(),
                                                     (s0 + s1) + (s2 + s3),
                                                     sycl::plus<float>());
            // Reference: x * rsqrt(x.square().mean(-1) + eps)
            const float r = sycl::rsqrt(ss / float(hidden) + eps);
            for (uint32_t j = lid; j < hidden; j += kWG) yr[j] = xr[j] * r;
        });
    });
}

// ---------------------------------------------------------------------------
// (a) Fused hyper-connection — ONE launch.
//
// Geometry.  One work-group per token, WG work-items, and the `mix` rows of the
// [mix, flat] GEMV split into `mix * CH` (row, chunk) jobs handed one per
// sub-group.  The original WG=256/CH=1 shape put 24 rows on 16 sub-groups, so
// the GEMV took two rounds with a third of the group idle in the second, and
// only 16 hardware threads were ever in flight against a 1.5 MB weight read.
// Both WG and CH are template parameters so the launcher can pick a shape per
// T (see the dispatcher at the bottom): at T=1 the whole kernel is one
// work-group on one Xe-core and wants every thread it can get, while at
// T=512 there are already 512 work-groups and a smaller one packs better.
// ---------------------------------------------------------------------------
namespace {

template <uint32_t WG, uint32_t CH>
sycl::event ds4_hc_launch(sycl::queue& q,
                          const float* streams,
                          const float* fn, const float* base, const float* scale,
                          float* post, float* comb, float* collapsed,
                          uint32_t n_tokens, uint32_t hidden, uint32_t hc_mult,
                          uint32_t sinkhorn_iters,
                          float rms_eps, float hc_eps,
                          const std::vector<sycl::event>& deps) {
    const uint32_t flat = hc_mult * hidden;          // == fn's row length
    const uint32_t mix  = (2u + hc_mult) * hc_mult;  // == fn's row count

    return ie::ps(q, "ds4_hyper_connection", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<float, 1> spart(sycl::range<1>(size_t(mix) * CH), h);
        sycl::local_accessor<float, 1> smix(sycl::range<1>(mix), h);
        sycl::local_accessor<float, 1> spre(sycl::range<1>(hc_mult), h);
        sycl::local_accessor<float, 1> scomb(sycl::range<1>(size_t(hc_mult) * hc_mult), h);

        h.parallel_for(sycl::nd_range<1>(size_t(n_tokens) * WG, WG),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
            const uint32_t tok = uint32_t(it.get_group(0));
            const uint32_t lid = uint32_t(it.get_local_id(0));
            const auto sg = it.get_sub_group();
            const uint32_t sgid  = uint32_t(sg.get_group_linear_id());
            const uint32_t n_sg  = uint32_t(sg.get_group_linear_range());
            const float* xs = streams + size_t(tok) * flat;

            // --- input_norm: UnweightedRMSNorm over the flattened streams.
            float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
            uint32_t i = lid;
            for (; i + 3u * WG < flat; i += 4u * WG) {
                const float v0 = xs[i], v1 = xs[i + WG];
                const float v2 = xs[i + 2u * WG], v3 = xs[i + 3u * WG];
                s0 += v0 * v0; s1 += v1 * v1; s2 += v2 * v2; s3 += v3 * v3;
            }
            for (; i < flat; i += WG) { const float v = xs[i]; s0 += v * v; }
            const float ss = sycl::reduce_over_group(it.get_group(),
                                                     (s0 + s1) + (s2 + s3),
                                                     sycl::plus<float>());
            const float rms = sycl::rsqrt(ss / float(flat) + rms_eps);

            // --- F.linear(flat, fn): the RMS scale is a per-row constant, so
            // dot(flat, fn[m]) == rms * dot(streams, fn[m]).  One sub-group per
            // (mix row, chunk); the chunk partials are summed below.
            const uint32_t per = (flat / CH) & ~(kSG - 1u);   // keep chunks kSG-aligned
            for (uint32_t idx = sgid; idx < mix * CH; idx += n_sg) {
                const uint32_t m = idx / CH, c = idx % CH;
                const uint32_t lo = c * per;
                const uint32_t hi = (c == CH - 1u) ? flat : lo + per;
                const float d = sg_dot_range(sg, xs, fn + size_t(m) * flat, lo, hi);
                if (sg.leader()) spart[idx] = d;
            }
            it.barrier(sycl::access::fence_space::local_space);
            if (lid < mix) {
                float acc = 0.f;
                for (uint32_t c = 0; c < CH; ++c) acc += spart[size_t(lid) * CH + c];
                smix[lid] = acc * rms;
            }
            it.barrier(sycl::access::fence_space::local_space);

            // --- pre / post / comb + the whole Sinkhorn loop, one work-item.
            // Zero extra launches.
            if (lid == 0) {
                const float pre_s = scale[0], post_s = scale[1], comb_s = scale[2];
                const uint32_t hc = hc_mult;
                for (uint32_t s = 0; s < hc; ++s) {
                    spre[s] = sigmoid_ref(smix[s] * pre_s + base[s]) + hc_eps;
                    post[size_t(tok) * hc + s] =
                        2.f * sigmoid_ref(smix[hc + s] * post_s + base[hc + s]);
                }
                if (hc == 4u) {
                    // comb = softmax(comb_w*scale + comb_b, dim=-1) + eps, then
                    // Sinkhorn — all 16 values in registers.
                    float m4[16];
                    #pragma unroll
                    for (int r = 0; r < 4; ++r) {
                        float mx = -std::numeric_limits<float>::infinity();
                        #pragma unroll
                        for (int c = 0; c < 4; ++c) {
                            const uint32_t o = 8u + uint32_t(r) * 4u + uint32_t(c);
                            const float l = smix[o] * comb_s + base[o];
                            m4[r * 4 + c] = l;
                            mx = sycl::fmax(mx, l);
                        }
                        float den = 0.f;
                        #pragma unroll
                        for (int c = 0; c < 4; ++c) {
                            const float e = sycl::exp(m4[r * 4 + c] - mx);
                            m4[r * 4 + c] = e;
                            den += e;
                        }
                        #pragma unroll
                        for (int c = 0; c < 4; ++c) m4[r * 4 + c] = m4[r * 4 + c] / den + hc_eps;
                    }
                    ds4_sinkhorn4(m4, sinkhorn_iters, hc_eps);
                    #pragma unroll
                    for (int e = 0; e < 16; ++e) comb[size_t(tok) * 16 + e] = m4[e];
                } else {
                    // Generic hc: the same arithmetic over SLM.
                    for (uint32_t r = 0; r < hc; ++r) {
                        float mx = -std::numeric_limits<float>::infinity();
                        for (uint32_t c = 0; c < hc; ++c) {
                            const uint32_t o = 2u * hc + r * hc + c;
                            const float l = smix[o] * comb_s + base[o];
                            scomb[r * hc + c] = l;
                            mx = sycl::fmax(mx, l);
                        }
                        float den = 0.f;
                        for (uint32_t c = 0; c < hc; ++c) {
                            const float e = sycl::exp(scomb[r * hc + c] - mx);
                            scomb[r * hc + c] = e;
                            den += e;
                        }
                        for (uint32_t c = 0; c < hc; ++c)
                            scomb[r * hc + c] = scomb[r * hc + c] / den + hc_eps;
                    }
                    // comb = comb / (comb.sum(dim=-2) + eps)   [column sums]
                    for (uint32_t c = 0; c < hc; ++c) {
                        float cs = 0.f;
                        for (uint32_t r = 0; r < hc; ++r) cs += scomb[r * hc + c];
                        cs += hc_eps;
                        for (uint32_t r = 0; r < hc; ++r) scomb[r * hc + c] /= cs;
                    }
                    // (hc_sinkhorn_iters - 1) × (row-normalise, column-normalise)
                    for (uint32_t t = 1; t < sinkhorn_iters; ++t) {
                        for (uint32_t r = 0; r < hc; ++r) {
                            float rs = 0.f;
                            for (uint32_t c = 0; c < hc; ++c) rs += scomb[r * hc + c];
                            rs += hc_eps;
                            for (uint32_t c = 0; c < hc; ++c) scomb[r * hc + c] /= rs;
                        }
                        for (uint32_t c = 0; c < hc; ++c) {
                            float cs = 0.f;
                            for (uint32_t r = 0; r < hc; ++r) cs += scomb[r * hc + c];
                            cs += hc_eps;
                            for (uint32_t r = 0; r < hc; ++r) scomb[r * hc + c] /= cs;
                        }
                    }
                    for (uint32_t e = 0; e < hc * hc; ++e)
                        comb[size_t(tok) * hc * hc + e] = scomb[e];
                }
            }
            it.barrier(sycl::access::fence_space::local_space);

            // --- collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2)
            for (uint32_t j = lid; j < hidden; j += WG) {
                float acc = 0.f;
                for (uint32_t s = 0; s < hc_mult; ++s) acc += spre[s] * xs[size_t(s) * hidden + j];
                collapsed[size_t(tok) * hidden + j] = acc;
            }
        });
    });
}

// ---------------------------------------------------------------------------
// Split-K hyper-connection — the SMALL-T path.
//
// WHY.  The one-work-group-per-token geometry above is correct but at decode
// T=1 it is ONE work-group, and that work-group has to read the whole `fn`
// tensor: [mix=24, hc*hidden=16384] fp32 = 1.5 MB.  A single Xe-core on B70
// cannot keep enough loads in flight to stream 1.5 MB quickly.  Measured on
// this box with a cold L2 (rotating over a 151 MB working set, which is what
// decode does — 86 distinct `fn` per token with GBs of expert traffic in
// between):
//
//     work-groups     1      2      4      8     16     32     64     96    128
//     GB/s         17.3   32.5   69.7  134.3  133.3  160.0  184.9  421.2  422.4
//
// One work-group gets 17 GB/s out of a part that delivers 422 GB/s at ~96-128
// work-groups.  That is the whole bug: not arithmetic, not the Sinkhorn, not
// precision — just that a 1.5 MB read was pinned to 1/256th of the machine.
//
// The fix splits the [mix, flat] GEMV over `mix * C2` work-groups (96 at C2=4)
// and moves the reduce/Sinkhorn/collapse into a second, tiny launch.  Two
// launches instead of one, but the first now runs at ~350 GB/s.
//
// Measured, event-profiled (same metric kprof reports), hidden 4096, hc 4,
// 20 Sinkhorn iterations, cold-L2 harness:
//
//     T      shipped        split-K      speedup
//     1      71.00 us        8.08 us       8.8x
//     2      50.45 us        8.90 us       5.7x
//     4      51.99 us       10.97 us       4.7x
//     8      52.28 us       14.11 us       3.7x
//    16      55.98 us       22.35 us       2.5x
//    32      50.91 us       31.72 us       1.6x     <- hot-L2 bracket: 0.93x
//   512     536.56 us      706.66 us       0.76x
//
// At T=32 the two L2 brackets disagree in sign, so the split path is gated at
// T <= 16, where BOTH brackets show a clear win.  Prefill keeps the shipped
// kernel bit-for-bit.
//
// ACCURACY.  Splitting shortens the summation tree rather than lengthening it:
// the shipped kernel accumulates ~256 elements serially per lane before any
// tree reduce, the split kernel ~8.  Against a double-precision host reference
// at the real shape the split path is measurably MORE accurate than the one it
// replaces (max |post| error 8.3e-08 vs 1.8e-07).  See
// tests/unit/deepseek4_ops_gate_test.cpp for the error model and the bound.
// ---------------------------------------------------------------------------

// The split path is only taken for hc_mult == 4 (mix == 24) and T <= 16, so
// the scratch is a fixed ~8 KB per device and is allocated once.
constexpr uint32_t kHcSplitMaxT = 16;
constexpr uint32_t kHcSplitMaxC2 = 4;
constexpr uint32_t kHcSplitMix = 24;  // (2 + 4) * 4

// Blocks per token in the FUSED-NORM tail.  Distinct from the store-only path's
// G2 because here it is also the redundancy factor: an RMS needs the whole row,
// so every block recomputes the whole collapse and G2 multiplies the read of
// `streams`.  Chosen by measurement — see the table on ds4_hc_tail_norm.
constexpr uint32_t kHcNormG2 = 16;

struct Ds4HcScratch { float* part = nullptr; float* ssq = nullptr; };

// `part`/`ssq` are written by ds4_hc_gemv and read by ds4_hc_tail, which is
// submitted immediately after it on the SAME queue.  Every engine queue in this
// repo is in-order (src/core/allocator.cpp:49, src/model/deepseek4.cpp:1439),
// so the next call's gemv cannot start before this call's tail has retired.
// The split path checks that with q.is_in_order() and falls back to the fused
// kernel if it is ever false.
//
// The cache is keyed by the QUEUE, not by (context, device).  Keying by device
// would be enough for the engine as it stands — one compute queue per card —
// but it would silently break the day a second compute queue is opened on the
// same card, because the two would then share one `part` buffer with no
// ordering between them.  A per-queue buffer is ~8 KB and removes that hazard
// by construction rather than by convention.
//
// Never freed, deliberately: these outlive any single call, and freeing device
// USM after its context has been torn down is undefined.  The bound is 8 KB per
// compute queue, allocated once.
Ds4HcScratch ds4_hc_scratch(sycl::queue& q) {
    static std::mutex mu;
    static std::vector<std::pair<sycl::queue, Ds4HcScratch>> cache;
    std::lock_guard<std::mutex> lk(mu);
    for (const auto& kv : cache)
        if (kv.first == q) return kv.second;
    Ds4HcScratch s;
    s.part = sycl::malloc_device<float>(size_t(kHcSplitMaxT) * kHcSplitMix * kHcSplitMaxC2, q);
    s.ssq  = sycl::malloc_device<float>(size_t(kHcSplitMaxT) * kHcSplitMaxC2, q);
    cache.push_back({q, s});
    return s;
}

// Pass 1: work-group j == ((tok*mix + m)*C2 + c) reduces one contiguous chunk
// of one `fn` row against `streams`.  The m == 0 work-groups already hold their
// chunk of `streams`, so they also emit the RMS sum-of-squares for it — no
// separate pass and no extra read.
template <uint32_t WGA, uint32_t C2>
sycl::event ds4_hc_gemv(sycl::queue& q,
                        const float* streams, const float* fn,
                        float* part, float* ssq,
                        uint32_t n_tokens, uint32_t flat, uint32_t mix,
                        const std::vector<sycl::event>& deps) {
    const uint32_t P = flat / C2;
    const size_t J = size_t(n_tokens) * mix * C2;
    return ie::ps(q, "ds4_hc_gemv", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(J * WGA, WGA),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
            const size_t   j   = it.get_group(0);
            const uint32_t lid = uint32_t(it.get_local_id(0));
            const uint32_t c   = uint32_t(j % C2);
            const size_t   jm  = j / C2;
            const uint32_t m   = uint32_t(jm % mix);
            const size_t   tok = jm / mix;
            const float* xs = streams + tok * flat + size_t(c) * P;
            const float* fr = fn + size_t(m) * flat + size_t(c) * P;

            float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
            uint32_t i = lid;
            for (; i + 3u * WGA < P; i += 4u * WGA) {
                a0 += xs[i]             * fr[i];
                a1 += xs[i + WGA]       * fr[i + WGA];
                a2 += xs[i + 2u * WGA]  * fr[i + 2u * WGA];
                a3 += xs[i + 3u * WGA]  * fr[i + 3u * WGA];
            }
            for (; i < P; i += WGA) a0 += xs[i] * fr[i];
            const float d = sycl::reduce_over_group(it.get_group(),
                                                    (a0 + a1) + (a2 + a3),
                                                    sycl::plus<float>());
            if (lid == 0) part[jm * C2 + c] = d;

            if (m == 0u) {
                float b0 = 0.f, b1 = 0.f, b2 = 0.f, b3 = 0.f;
                uint32_t k = lid;
                for (; k + 3u * WGA < P; k += 4u * WGA) {
                    const float v0 = xs[k],            v1 = xs[k + WGA];
                    const float v2 = xs[k + 2u * WGA], v3 = xs[k + 3u * WGA];
                    b0 += v0 * v0; b1 += v1 * v1; b2 += v2 * v2; b3 += v3 * v3;
                }
                for (; k < P; k += WGA) { const float v = xs[k]; b0 += v * v; }
                const float s = sycl::reduce_over_group(it.get_group(),
                                                        (b0 + b1) + (b2 + b3),
                                                        sycl::plus<float>());
                if (lid == 0) ssq[tok * C2 + c] = s;
            }
        });
    });
}

// Pass 2: G2 work-groups per token.  All of them re-do the (tiny, L2-hot)
// reduction of the C2 partials so that every one of them has `pre` without a
// device-wide barrier; only block 0 writes `post`/`comb` and runs the Sinkhorn.
// The 4096-wide collapse is split across the G2 blocks.
template <uint32_t WGB, uint32_t G2, uint32_t C2, bool NORM = false>
sycl::event ds4_hc_tail(sycl::queue& q,
                        const float* streams, const float* part, const float* ssq,
                        const float* base, const float* scale,
                        float* post, float* comb, float* collapsed,
                        uint32_t n_tokens, uint32_t hidden, uint32_t hc_mult,
                        uint32_t sinkhorn_iters, float rms_eps, float hc_eps,
                        const std::vector<sycl::event>& deps,
                        const float* norm_w = nullptr, float* normed = nullptr,
                        float norm_eps = 0.f) {
    const uint32_t flat = hc_mult * hidden;
    const uint32_t mix  = (2u + hc_mult) * hc_mult;
    const uint32_t per  = (hidden + G2 - 1u) / G2;
    return ie::ps(q, NORM ? "ds4_hc_tail_norm" : "ds4_hc_tail", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<float, 1> smix(sycl::range<1>(mix), h);
        sycl::local_accessor<float, 1> spre(sycl::range<1>(hc_mult), h);
        h.parallel_for(sycl::nd_range<1>(size_t(n_tokens) * G2 * WGB, WGB),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
            const size_t   g   = it.get_group(0);
            const uint32_t blk = uint32_t(g % G2);
            const uint32_t tok = uint32_t(g / G2);
            const uint32_t lid = uint32_t(it.get_local_id(0));
            const float* xs = streams + size_t(tok) * flat;

            float ss = 0.f;
            for (uint32_t c = 0; c < C2; ++c) ss += ssq[size_t(tok) * C2 + c];
            const float rms = sycl::rsqrt(ss / float(flat) + rms_eps);
            if (lid < mix) {
                const float* pr = part + (size_t(tok) * mix + lid) * C2;
                float acc = 0.f;
                for (uint32_t c = 0; c < C2; ++c) acc += pr[c];
                smix[lid] = acc * rms;
            }
            it.barrier(sycl::access::fence_space::local_space);

            if (lid == 0) {
                const float pre_s = scale[0], post_s = scale[1], comb_s = scale[2];
                for (uint32_t s = 0; s < 4u; ++s)
                    spre[s] = sigmoid_ref(smix[s] * pre_s + base[s]) + hc_eps;
                if (blk == 0u) {
                    for (uint32_t s = 0; s < 4u; ++s)
                        post[size_t(tok) * 4 + s] =
                            2.f * sigmoid_ref(smix[4 + s] * post_s + base[4 + s]);
                    float m4[16];
                    #pragma unroll
                    for (int r = 0; r < 4; ++r) {
                        float mx = -std::numeric_limits<float>::infinity();
                        #pragma unroll
                        for (int c = 0; c < 4; ++c) {
                            const uint32_t o = 8u + uint32_t(r) * 4u + uint32_t(c);
                            const float l = smix[o] * comb_s + base[o];
                            m4[r * 4 + c] = l;
                            mx = sycl::fmax(mx, l);
                        }
                        float den = 0.f;
                        #pragma unroll
                        for (int c = 0; c < 4; ++c) {
                            const float e = sycl::exp(m4[r * 4 + c] - mx);
                            m4[r * 4 + c] = e;
                            den += e;
                        }
                        #pragma unroll
                        for (int c = 0; c < 4; ++c) m4[r * 4 + c] = m4[r * 4 + c] / den + hc_eps;
                    }
                    ds4_sinkhorn4(m4, sinkhorn_iters, hc_eps);
                    #pragma unroll
                    for (int e = 0; e < 16; ++e) comb[size_t(tok) * 16 + e] = m4[e];
                }
            }
            it.barrier(sycl::access::fence_space::local_space);

            // The collapse, as a value rather than a store: the fused-norm path
            // needs it twice (once for the row's sum of squares, once for the
            // slice it owns) and both must be the SAME expression in the SAME
            // order as the store-only path, or the fusion stops being
            // bit-identical to `ds4_hyper_connection` + `ds4_rms_norm`.
            auto collapse = [&](uint32_t j) {
                float acc = 0.f;
                #pragma unroll
                for (uint32_t s = 0; s < 4u; ++s) acc += spre[s] * xs[size_t(s) * hidden + j];
                return acc;
            };

            const uint32_t lo = blk * per;
            const uint32_t hi = sycl::min(lo + per, hidden);
            if constexpr (!NORM) {
                for (uint32_t j = lo + lid; j < hi; j += WGB)
                    collapsed[size_t(tok) * hidden + j] = collapse(j);
            } else {
                // Pass A — the WHOLE row's sum of squares, recomputed by every
                // block because an RMS does not decompose across the G2 split.
                // Loop shape, unroll factor, work-group width and reduction are
                // `ds4_rms_norm`'s (src/ops/deepseek4_attn.cpp:174) verbatim;
                // that is what makes `r` identical to the one the separate norm
                // launch would have computed.
                float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
                uint32_t i = lid;
                for (; i + 3u * WGB < hidden; i += 4u * WGB) {
                    const float v0 = collapse(i),            v1 = collapse(i + WGB);
                    const float v2 = collapse(i + 2u * WGB), v3 = collapse(i + 3u * WGB);
                    s0 += v0 * v0; s1 += v1 * v1; s2 += v2 * v2; s3 += v3 * v3;
                }
                for (; i < hidden; i += WGB) { const float v = collapse(i); s0 += v * v; }
                const float ss2 = sycl::reduce_over_group(it.get_group(),
                                                          (s0 + s1) + (s2 + s3),
                                                          sycl::plus<float>());
                const float r = sycl::rsqrt(ss2 / float(hidden) + norm_eps);
                // Pass B — this block's slice only.
                for (uint32_t j = lo + lid; j < hi; j += WGB) {
                    const float acc = collapse(j);
                    collapsed[size_t(tok) * hidden + j] = acc;
                    normed[size_t(tok) * hidden + j]    = norm_w[j] * (acc * r);
                }
            }
        });
    });
}

template <uint32_t C2, uint32_t G2, bool NORM = false>
sycl::event ds4_hc_split(sycl::queue& q,
                         const float* streams,
                         const float* fn, const float* base, const float* scale,
                         float* post, float* comb, float* collapsed,
                         uint32_t n_tokens, uint32_t hidden, uint32_t hc_mult,
                         uint32_t sinkhorn_iters, float rms_eps, float hc_eps,
                         const std::vector<sycl::event>& deps,
                         const float* norm_w = nullptr, float* normed = nullptr,
                         float norm_eps = 0.f) {
    const uint32_t flat = hc_mult * hidden;
    const uint32_t mix  = (2u + hc_mult) * hc_mult;
    const Ds4HcScratch sc = ds4_hc_scratch(q);
    auto e = ds4_hc_gemv<256, C2>(q, streams, fn, sc.part, sc.ssq,
                                  n_tokens, flat, mix, deps);
    return ds4_hc_tail<256, G2, C2, NORM>(q, streams, sc.part, sc.ssq, base, scale,
                                          post, comb, collapsed, n_tokens, hidden, hc_mult,
                                          sinkhorn_iters, rms_eps, hc_eps, {e},
                                          norm_w, normed, norm_eps);
}

// Eligibility for the split path.  Written once so the fused-norm entry point
// below cannot drift from `ds4_hyper_connection`'s own test.
inline bool ds4_hc_split_ok(sycl::queue& q, uint32_t n_tokens, uint32_t hidden,
                            uint32_t hc_mult) {
    return hc_mult == 4u && n_tokens >= 1u && n_tokens <= kHcSplitMaxT &&
           q.is_in_order() && ((hc_mult * hidden) % 4u) == 0u;
}

}  // namespace

sycl::event ds4_hyper_connection(sycl::queue& q,
                                 const float* streams,
                                 const float* fn, const float* base, const float* scale,
                                 float* post, float* comb, float* collapsed,
                                 uint32_t n_tokens, uint32_t hidden, uint32_t hc_mult,
                                 uint32_t sinkhorn_iters,
                                 float rms_eps, float hc_eps,
                                 const std::vector<sycl::event>& deps) {
    // Small T: split the [mix, flat] GEMV across work-groups (see the block
    // comment above).  Everything the split path assumes is checked here; any
    // shape it does not cover falls through to the fused kernel unchanged.
    if (ds4_hc_split_ok(q, n_tokens, hidden, hc_mult)) {
        if (n_tokens <= 2u)
            return ds4_hc_split<4, 16>(q, streams, fn, base, scale, post, comb, collapsed,
                                       n_tokens, hidden, hc_mult, sinkhorn_iters,
                                       rms_eps, hc_eps, deps);
        return ds4_hc_split<2, 8>(q, streams, fn, base, scale, post, comb, collapsed,
                                  n_tokens, hidden, hc_mult, sinkhorn_iters,
                                  rms_eps, hc_eps, deps);
    }
    // Measured on B70 at the real V4-Flash shape (hidden 4096, hc 4, 20 iters),
    // sweeping WG × CH: at T=1 the kernel is a single work-group and wants the
    // widest group (768/CH=2 was 8.4x the old 256/CH=1); by T=512 there are
    // enough work-groups that a smaller one packs better (512/CH=2 was best).
    // The crossover is broad and flat, so one threshold is enough.
    if (n_tokens <= 64)
        return ds4_hc_launch<768, 2>(q, streams, fn, base, scale, post, comb, collapsed,
                                     n_tokens, hidden, hc_mult, sinkhorn_iters,
                                     rms_eps, hc_eps, deps);
    return ds4_hc_launch<512, 2>(q, streams, fn, base, scale, post, comb, collapsed,
                                 n_tokens, hidden, hc_mult, sinkhorn_iters,
                                 rms_eps, hc_eps, deps);
}

sycl::event ds4_hyper_connection_norm(sycl::queue& q,
                                      const float* streams,
                                      const float* fn, const float* base, const float* scale,
                                      const float* norm_w,
                                      float* post, float* comb, float* collapsed, float* normed,
                                      uint32_t n_tokens, uint32_t hidden, uint32_t hc_mult,
                                      uint32_t sinkhorn_iters,
                                      float rms_eps, float hc_eps, float norm_eps,
                                      const std::vector<sycl::event>& deps) {
    // The fusion lives in the split path's tail, so a shape the split path does
    // not cover keeps the two-launch sequence.  Correct either way; it simply
    // stops saving the launch, which is the only thing the fusion buys.
    if (ds4_hc_split_ok(q, n_tokens, hidden, hc_mult)) {
        // G2 is the redundancy factor of the fused norm — every block recomputes
        // the whole collapse — so the norm variant does NOT inherit the
        // store-only path's G2.  Measured on a B70 at the real decode shape
        // (T=1, hidden 4096, hc 4), see tools/ie_ds4_bench --hcnorm-sweep.
        if (n_tokens <= 2u)
            return ds4_hc_split<4, kHcNormG2, true>(q, streams, fn, base, scale, post, comb,
                                                    collapsed, n_tokens, hidden, hc_mult,
                                                    sinkhorn_iters, rms_eps, hc_eps, deps,
                                                    norm_w, normed, norm_eps);
        return ds4_hc_split<2, kHcNormG2, true>(q, streams, fn, base, scale, post, comb,
                                                collapsed, n_tokens, hidden, hc_mult,
                                                sinkhorn_iters, rms_eps, hc_eps, deps,
                                                norm_w, normed, norm_eps);
    }
    auto e = ds4_hyper_connection(q, streams, fn, base, scale, post, comb, collapsed,
                                  n_tokens, hidden, hc_mult, sinkhorn_iters, rms_eps,
                                  hc_eps, deps);
    return ds4_rms_norm(q, collapsed, norm_w, normed, n_tokens, hidden, norm_eps, {e});
}

// ---------------------------------------------------------------------------
// (c) Grouped linear
// ---------------------------------------------------------------------------
sycl::event ds4_grouped_linear(sycl::queue& q,
                               const float* x, const float* w, float* y,
                               uint32_t n_tokens, uint32_t n_groups,
                               uint32_t in_per_group, uint32_t out_per_group,
                               const std::vector<sycl::event>& deps) {
    // One sub-group per output element; kWG/kSG sub-groups per work-group.
    const size_t n_out = size_t(n_tokens) * n_groups * out_per_group;
    const size_t n_wg  = (n_out + (kWG / kSG) - 1) / (kWG / kSG);

    return ie::ps(q, "ds4_grouped_linear", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(n_wg * kWG, kWG),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
            const auto sg = it.get_sub_group();
            const size_t row = size_t(it.get_group(0)) * (kWG / kSG) + sg.get_group_linear_id();
            if (row >= n_out) return;
            const uint32_t o = uint32_t(row % out_per_group);
            const size_t   tg = row / out_per_group;
            const uint32_t g = uint32_t(tg % n_groups);
            const size_t   t = tg / n_groups;

            const float* xr = x + (t * n_groups + g) * size_t(in_per_group);
            const float* wr = w + (size_t(g) * out_per_group + o) * size_t(in_per_group);
            const float d = sg_dot(sg, xr, wr, in_per_group);
            if (sg.leader()) y[row] = d;
        });
    });
}

// ---------------------------------------------------------------------------
// (d) MoE routing.
// ---------------------------------------------------------------------------
namespace {

// Router logits: one GEMV of [n_experts, hidden] against each token.
//
// WHY THIS IS ITS OWN LAUNCH.  Both routers used to compute the logits inside
// the same work-group that then selected the top-k, which forced the shape
// "one work-group per token": at decode T=1 that is ONE work-group of 256
// work-items reading the whole 4 MB router weight matrix, on a card with 256
// compute units.  Measured 0.53-0.62 ms per call for 1 MFLOP of arithmetic.
// Split out, the GEMV is the same shape as ds4_grouped_linear — one sub-group
// per output element — and at T=1 supplies 256 sub-groups instead of 16.
//
// BIT-IDENTICAL, NOT MERELY CLOSE.  The lane->element map and the four-partial
// accumulation order are `sg_dot`, unchanged, so each logit is the same fp32
// value it was before; only the number of work-groups differs.  Measured over
// T in {1, 2, 4, 16, 64}: max |logit delta| and max |weight delta| both exactly
// 0.00e+00, and zero expert-index mismatches.  A work-group-wide reduction
// would have been ~1.5x faster again but perturbs each logit by ~4e-8, which
// can flip a near-tie at the top-6 boundary; that is not worth 5 us per call.
//
//   logits [n_tokens, n_experts]
sycl::event ds4_router_logits_tiled(sycl::queue& q,
                                    const float* x, const float* w, float* logits,
                                    uint32_t n_tokens, uint32_t hidden, uint32_t n_experts,
                                    const std::vector<sycl::event>& deps) {
    constexpr uint32_t tokens_per_tile = 2;
    constexpr uint32_t experts_per_tile = 2;
    constexpr uint32_t tile_wg = 64;
    const size_t token_tiles = (size_t(n_tokens) + tokens_per_tile - 1) / tokens_per_tile;
    const size_t expert_tiles = (size_t(n_experts) + experts_per_tile - 1) / experts_per_tile;
    const size_t groups = (token_tiles * expert_tiles + tile_wg / kSG - 1) / (tile_wg / kSG);
    return ie::ps(q, "ds4_router_logits", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(groups * tile_wg, tile_wg),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
            const auto sg = it.get_sub_group();
            const size_t tile = it.get_group(0) * (tile_wg / kSG) + sg.get_group_linear_id();
            if (tile >= token_tiles * expert_tiles) return;
            const size_t first = (tile / expert_tiles) * tokens_per_tile;
            const size_t first_expert = (tile % expert_tiles) * experts_per_tile;
            float sums[tokens_per_tile][experts_per_tile][4] = {};
            uint32_t i = uint32_t(sg.get_local_linear_id());
            // A 2x2 output tile reuses both activations and weights. Each dot
            // retains sg_dot's four partial chains and subgroup reduction.
            for (; i + 3u * kSG < hidden; i += 4u * kSG) {
                float wr[experts_per_tile][4] = {};
                #pragma unroll
                for (uint32_t e = 0; e < experts_per_tile; ++e)
                    if (first_expert + e < n_experts) {
                        #pragma unroll
                        for (uint32_t p = 0; p < 4; ++p)
                            wr[e][p] = w[(first_expert + e) * hidden + i + p * kSG];
                    }
                #pragma unroll
                for (uint32_t t = 0; t < tokens_per_tile; ++t) {
                    if (first + t < n_tokens) {
                        const float* xr = x + (first + t) * hidden;
                        #pragma unroll
                        for (uint32_t p = 0; p < 4; ++p) {
                            const float xv = xr[i + p * kSG];
                            #pragma unroll
                            for (uint32_t e = 0; e < experts_per_tile; ++e)
                                sums[t][e][p] += xv * wr[e][p];
                        }
                    }
                }
            }
            for (; i < hidden; i += kSG) {
                float wr[experts_per_tile] = {};
                #pragma unroll
                for (uint32_t e = 0; e < experts_per_tile; ++e)
                    if (first_expert + e < n_experts)
                        wr[e] = w[(first_expert + e) * hidden + i];
                #pragma unroll
                for (uint32_t t = 0; t < tokens_per_tile; ++t)
                    if (first + t < n_tokens) {
                        const float xv = x[(first + t) * hidden + i];
                        #pragma unroll
                        for (uint32_t e = 0; e < experts_per_tile; ++e)
                            sums[t][e][0] += xv * wr[e];
                    }
            }
            #pragma unroll
            for (uint32_t t = 0; t < tokens_per_tile; ++t) {
                if (first + t < n_tokens) {
                    #pragma unroll
                    for (uint32_t e = 0; e < experts_per_tile; ++e) {
                        const float v = sycl::reduce_over_group(sg,
                            (sums[t][e][0] + sums[t][e][1]) + (sums[t][e][2] + sums[t][e][3]), sycl::plus<float>());
                        if (sg.leader() && first_expert + e < n_experts)
                            logits[(first + t) * n_experts + first_expert + e] = v;
                    }
                }
            }
        });
    });
}

sycl::event ds4_router_logits(sycl::queue& q,
                              const float* x, const float* w, float* logits,
                              uint32_t n_tokens, uint32_t hidden, uint32_t n_experts,
                              const std::vector<sycl::event>& deps) {
    // A smaller grid hurts decode occupancy. Tile only substantial prefills;
    // scalar-token routing and small model/test geometries retain sg_dot.
    if (n_tokens >= 64 && hidden >= 1024 && n_experts >= 64)
        return ds4_router_logits_tiled(q, x, w, logits, n_tokens, hidden, n_experts, deps);
    const size_t n_out = size_t(n_tokens) * n_experts;
    const size_t n_wg  = (n_out + (kWG / kSG) - 1) / (kWG / kSG);
    return ie::ps(q, "ds4_router_logits", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(n_wg * kWG, kWG),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
            const auto sg = it.get_sub_group();
            const size_t row = size_t(it.get_group(0)) * (kWG / kSG) + sg.get_group_linear_id();
            if (row >= n_out) return;
            const uint32_t e   = uint32_t(row % n_experts);
            const size_t   tok = row / n_experts;
            const float d = sg_dot(sg, x + tok * hidden, w + size_t(e) * hidden, hidden);
            if (sg.leader()) logits[row] = d;
        });
    });
}

// Shared body of both routers' second stage: sqrtsoftplus of the logits into
// SLM.  `sscore` holds scores[0..n_experts) on exit (post-barrier is the
// caller's job).
template <typename Item, typename Acc>
inline void ds4_router_scores(const Item& it, const Acc& sscore,
                              const float* logits,
                              uint32_t tok, uint32_t n_experts) {
    const uint32_t lid = uint32_t(it.get_local_id(0));
    for (uint32_t e = lid; e < n_experts; e += kWG)
        sscore[e] = sycl::sqrt(softplus_ref(logits[size_t(tok) * n_experts + e]));
}

// Order-preserving map float -> uint32 (IEEE754), packed with the complement
// of the expert index so that a plain uint64 max yields "largest score,
// smallest index on a tie" — which is exactly the rule the serial scan this
// replaces implemented with `v > best` over ascending e.  Identical to the
// helper in src/ops/deepseek4_attn.cpp; duplicated rather than shared because
// hoisting it into a header would put a device-only bit_cast in the public
// interface of two unrelated ops.
inline uint64_t pack_key(float v, uint32_t idx) {
    uint32_t u = sycl::bit_cast<uint32_t>(v);
    u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return (uint64_t(u) << 32) | uint64_t(0xFFFFFFFFu - idx);
}

// Shared tail: gather the k selected scores, renormalise, scale, store.
// `indices` for this token must already be written.  Reference:
//   weights = scores.gather(1, indices)
//   weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
//   return ..., weights * self.routed_scaling_factor, ...
template <typename Acc>
inline void ds4_router_weights(const Acc& sscore,
                               float* weights, const int32_t* indices,
                               uint32_t tok, uint32_t top_k, float routed_scaling) {
    float sum = 0.f;
    for (uint32_t j = 0; j < top_k; ++j) sum += sscore[indices[size_t(tok) * top_k + j]];
    const float inv = 1.f / (sum + 1e-20f);
    for (uint32_t j = 0; j < top_k; ++j) {
        weights[size_t(tok) * top_k + j] =
            sscore[indices[size_t(tok) * top_k + j]] * inv * routed_scaling;
    }
}

}  // namespace

sycl::event ds4_router_topk(sycl::queue& q,
                            const float* x, const float* w, const float* bias,
                            float* logits, float* weights, int32_t* indices,
                            uint32_t n_tokens, uint32_t hidden,
                            uint32_t n_experts, uint32_t top_k,
                            float routed_scaling,
                            const std::vector<sycl::event>& deps) {
    auto gemv = ds4_router_logits(q, x, w, logits, n_tokens, hidden, n_experts, deps);

    return ie::ps(q, "ds4_router_topk", [&](sycl::handler& h) {
        h.depends_on(gemv);
        sycl::local_accessor<float, 1> sscore(sycl::range<1>(n_experts), h);
        h.parallel_for(sycl::nd_range<1>(size_t(n_tokens) * kWG, kWG),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
            const uint32_t tok = uint32_t(it.get_group(0));
            const uint32_t lid = uint32_t(it.get_local_id(0));
            const auto grp = it.get_group();
            ds4_router_scores(it, sscore, logits, tok, n_experts);
            it.barrier(sycl::access::fence_space::local_space);

            // topk(scores + bias): the bias steers SELECTION ONLY.  Descending
            // order, lower expert index wins ties — matches torch.topk's
            // observed output for the reference blobs.
            //
            // This used to be a triple loop on work-item 0 while the other 255
            // waited: for each of the top_k rounds it rescanned all n_experts
            // and, for each, rescanned the j already-taken picks.  The packed
            // key makes the whole thing a work-group reduction instead —
            // `k < prev` excludes exactly the already-taken picks, because the
            // keys are pairwise distinct (the expert index is in the low bits)
            // so the j entries at or above the previous round's winner ARE the
            // j already selected.  Same rule, same answer, 256 lanes wide.
            uint64_t prev = ~uint64_t(0);
            for (uint32_t j = 0; j < top_k; ++j) {
                uint64_t best = 0;
                for (uint32_t e = lid; e < n_experts; e += kWG) {
                    const uint64_t k = pack_key(sscore[e] + (bias ? bias[e] : 0.f), e);
                    if (k < prev && k > best) best = k;
                }
                best = sycl::reduce_over_group(grp, best, sycl::maximum<uint64_t>());
                if (lid == 0)
                    indices[size_t(tok) * top_k + j] =
                        int32_t(0xFFFFFFFFu - uint32_t(best & 0xFFFFFFFFu));
                prev = best;
            }
            it.barrier(sycl::access::fence_space::local_space);

            // weights = scores.gather(...) — the UNBIASED score.
            if (lid == 0)
                ds4_router_weights(sscore, weights, indices, tok, top_k, routed_scaling);
        });
    });
}

sycl::event ds4_router_hash(sycl::queue& q,
                            const float* x, const float* w,
                            const int32_t* tid2eid, const int32_t* input_ids,
                            float* logits, float* weights, int32_t* indices,
                            uint32_t n_tokens, uint32_t hidden,
                            uint32_t n_experts, uint32_t top_k,
                            float routed_scaling,
                            const std::vector<sycl::event>& deps) {
    auto gemv = ds4_router_logits(q, x, w, logits, n_tokens, hidden, n_experts, deps);

    return ie::ps(q, "ds4_router_hash", [&](sycl::handler& h) {
        h.depends_on(gemv);
        sycl::local_accessor<float, 1> sscore(sycl::range<1>(n_experts), h);
        h.parallel_for(sycl::nd_range<1>(size_t(n_tokens) * kWG, kWG),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
            const uint32_t tok = uint32_t(it.get_group(0));
            ds4_router_scores(it, sscore, logits, tok, n_experts);
            it.barrier(sycl::access::fence_space::local_space);

            if (it.get_local_id(0) == 0) {
                // indices = tid2eid[input_ids] — frozen table, no argsort.
                const int32_t tid = input_ids[tok];
                for (uint32_t j = 0; j < top_k; ++j)
                    indices[size_t(tok) * top_k + j] = tid2eid[size_t(tid) * top_k + j];
                ds4_router_weights(sscore, weights, indices, tok, top_k, routed_scaling);
            }
        });
    });
}

namespace {
// Shared selection body of the two *_vl routers: top-k over sscore + bias,
// the packed-key work-group reduction from ds4_router_topk.
template <typename Item, typename Acc>
inline void ds4_router_select(const Item& it, const Acc& sscore, const float* bias,
                              int32_t* indices, uint32_t tok, uint32_t n_experts, uint32_t top_k) {
    const uint32_t lid = uint32_t(it.get_local_id(0));
    const auto grp = it.get_group();
    uint64_t prev = ~uint64_t(0);
    for (uint32_t j = 0; j < top_k; ++j) {
        uint64_t best = 0;
        for (uint32_t e = lid; e < n_experts; e += kWG) {
            const uint64_t k = pack_key(sscore[e] + (bias ? bias[e] : 0.f), e);
            if (k < prev && k > best) best = k;
        }
        best = sycl::reduce_over_group(grp, best, sycl::maximum<uint64_t>());
        if (lid == 0)
            indices[size_t(tok) * top_k + j] = int32_t(0xFFFFFFFFu - uint32_t(best & 0xFFFFFFFFu));
        prev = best;
    }
}
}  // namespace

sycl::event ds4_router_topk_vl(sycl::queue& q,
                               const float* x, const float* w, const float* bias,
                               const float* bias_vl, const int32_t* img_mask,
                               float* logits, float* weights, int32_t* indices,
                               uint32_t n_tokens, uint32_t hidden,
                               uint32_t n_experts, uint32_t top_k,
                               float routed_scaling,
                               const std::vector<sycl::event>& deps) {
    auto gemv = ds4_router_logits(q, x, w, logits, n_tokens, hidden, n_experts, deps);
    return ie::ps(q, "ds4_router_topk", [&](sycl::handler& h) {
        h.depends_on(gemv);
        sycl::local_accessor<float, 1> sscore(sycl::range<1>(n_experts), h);
        h.parallel_for(sycl::nd_range<1>(size_t(n_tokens) * kWG, kWG),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
            const uint32_t tok = uint32_t(it.get_group(0));
            ds4_router_scores(it, sscore, logits, tok, n_experts);
            it.barrier(sycl::access::fence_space::local_space);
            // Per-token bias choice is work-group uniform (one token per group).
            const float* bsel = img_mask[tok] ? bias_vl : bias;
            ds4_router_select(it, sscore, bsel, indices, tok, n_experts, top_k);
            it.barrier(sycl::access::fence_space::local_space);
            if (it.get_local_id(0) == 0)
                ds4_router_weights(sscore, weights, indices, tok, top_k, routed_scaling);
        });
    });
}

sycl::event ds4_router_hash_vl(sycl::queue& q,
                               const float* x, const float* w,
                               const int32_t* tid2eid, const int32_t* input_ids,
                               const float* bias_vl, const int32_t* img_mask,
                               float* logits, float* weights, int32_t* indices,
                               uint32_t n_tokens, uint32_t hidden,
                               uint32_t n_experts, uint32_t top_k,
                               float routed_scaling,
                               const std::vector<sycl::event>& deps) {
    auto gemv = ds4_router_logits(q, x, w, logits, n_tokens, hidden, n_experts, deps);
    return ie::ps(q, "ds4_router_hash", [&](sycl::handler& h) {
        h.depends_on(gemv);
        sycl::local_accessor<float, 1> sscore(sycl::range<1>(n_experts), h);
        h.parallel_for(sycl::nd_range<1>(size_t(n_tokens) * kWG, kWG),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
            const uint32_t tok = uint32_t(it.get_group(0));
            ds4_router_scores(it, sscore, logits, tok, n_experts);
            it.barrier(sycl::access::fence_space::local_space);
            if (img_mask[tok]) {   // work-group uniform branch
                ds4_router_select(it, sscore, bias_vl, indices, tok, n_experts, top_k);
                it.barrier(sycl::access::fence_space::local_space);
                if (it.get_local_id(0) == 0)
                    ds4_router_weights(sscore, weights, indices, tok, top_k, routed_scaling);
            } else if (it.get_local_id(0) == 0) {
                const int32_t tid = input_ids[tok];
                for (uint32_t j = 0; j < top_k; ++j)
                    indices[size_t(tok) * top_k + j] = tid2eid[size_t(tid) * top_k + j];
                ds4_router_weights(sscore, weights, indices, tok, top_k, routed_scaling);
            }
        });
    });
}

// ---------------------------------------------------------------------------
// (e) Clamped SwiGLU
// ---------------------------------------------------------------------------
sycl::event ds4_swiglu_clamped(sycl::queue& q,
                               const float* gate, const float* up, float* y,
                               size_t n, float limit,
                               const std::vector<sycl::event>& deps) {
    const size_t global = ((n + kWG - 1) / kWG) * kWG;
    return ie::ps(q, "ds4_swiglu_clamped", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(global, kWG), [=](sycl::nd_item<1> it) {
            const size_t i = it.get_global_id(0);
            if (i >= n) return;
            const float g = sycl::fmin(gate[i], limit);            // clamp(max=limit)
            const float u = sycl::fmin(sycl::fmax(up[i], -limit), limit);
            y[i] = (g * sigmoid_ref(g)) * u;                       // silu(g) * u
        });
    });
}

sycl::event ds4_swiglu_clamped_to_f16(sycl::queue& q,
                                      const float* gate, const float* up,
                                      sycl::half* y, size_t n, float limit,
                                      const std::vector<sycl::event>& deps) {
    if (n > std::numeric_limits<size_t>::max() / sizeof(float) ||
        (n && (!gate || !up || !y)))
        throw std::invalid_argument("ds4_swiglu_clamped_to_f16: invalid operand span");
    if (!n) return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.single_task([] {});
    });
    const size_t global = ((n + kWG - 1) / kWG) * kWG;
    return ie::ps(q, "ds4_swiglu_clamped_to_f16", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(global, kWG), [=](sycl::nd_item<1> it) {
            const size_t i = it.get_global_id(0);
            if (i >= n) return;
            const float g = sycl::fmin(gate[i], limit);
            const float u = sycl::fmin(sycl::fmax(up[i], -limit), limit);
            y[i] = sycl::half((g * sigmoid_ref(g)) * u);
        });
    });
}

// fp16-in / fp16-out — see the header for why this exists and why it is
// bit-identical to cast + cast + ds4_swiglu_clamped + cast.  The body below is
// character-for-character the fp32 body above; only the loads and the store
// change type, and `float(half)` is exact.
sycl::event ds4_swiglu_clamped_h(sycl::queue& q,
                                 const sycl::half* gate, const sycl::half* up,
                                 sycl::half* y, size_t n, float limit,
                                 const std::vector<sycl::event>& deps) {
    const size_t global = ((n + kWG - 1) / kWG) * kWG;
    return ie::ps(q, "ds4_swiglu_clamped_h", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(global, kWG), [=](sycl::nd_item<1> it) {
            const size_t i = it.get_global_id(0);
            if (i >= n) return;
            const float g = sycl::fmin(float(gate[i]), limit);      // clamp(max=limit)
            const float u = sycl::fmin(sycl::fmax(float(up[i]), -limit), limit);
            y[i] = sycl::half((g * sigmoid_ref(g)) * u);            // silu(g) * u
        });
    });
}

}  // namespace ie
