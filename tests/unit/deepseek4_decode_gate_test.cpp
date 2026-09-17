// tests/unit/deepseek4_decode_gate_test.cpp — gate for the decode-shape rework
// of ds4_attention, ds4_router_topk and ds4_indexer_topk.
//
// WHAT CHANGED
// ------------
// All three kernels launched one work-group per query token (or per (token,
// head)) and therefore ran decode, T=1, on a handful of the B70's 256 compute
// units.  Each now spreads the same arithmetic over many more work-groups:
//
//   ds4_attention     SP sub-groups per (token, head), each walking a stride-SP
//                     slice of the kv axis with its own online-softmax state,
//                     combined through SLM.  SP is chosen by T (32 at T=1).
//   ds4_router_topk   the [n_experts, hidden] GEMV split into its own launch,
//                     one sub-group per (token, expert); the top-k selection
//                     turned from a serial scan on work-item 0 into a
//                     work-group reduction over packed keys.
//   ds4_indexer_topk  top_k rounds of "scan everything and reduce" replaced by
//                     one bitonic sort of the packed keys in SLM.
//
// WHAT IS CLAIMED, AND HOW EACH CLAIM IS GATED
// --------------------------------------------
// Two of the three are EXACT, and are gated as exact — no tolerance at all:
//
//   §4 ds4_router_topk is bit-identical to the fused kernel it replaces.  The
//      dot product keeps `sg_dot`'s lane map and four-accumulator order, so the
//      logits are the same fp32 values; the packed-key selection implements the
//      same "largest score, lowest index on a tie, previously-taken excluded"
//      rule.  §4 checks logits, indices AND weights bit for bit.
//
//   §5 ds4_indexer_topk is bit-identical to the scan it replaces.  The packed
//      keys are pairwise distinct (the entry index occupies the low 32 bits),
//      so "the j-th largest" is a uniquely determined entry and sorting cannot
//      disagree with selecting.  §5 checks every emitted index.
//
// ds4_attention is NOT bit-identical and cannot be: SP independent online
// softmaxes plus a combine sum the numerator and denominator in a different
// order, and fp32 addition is not associative.  The logits are unchanged (same
// lane->dim map, same per-column sub-group reduce), so the ONLY new error is
// reassociation of the softmax accumulation, and §3 bounds exactly that from
// the split structure — see the derivation above `split_const`.  §2 first shows
// the partition itself is sound (every live column visited exactly once), and
// §6 shows the §3 bound is sharp enough to be worth passing, by breaking the
// kernel three ways and watching every break exceed it.
//
// This test does NOT need $DS4_PARITY2_DIR.  deepseek4_attn_test, the
// blob-based gate, cannot run on this machine and correctly exits 1; nothing
// here depends on it.
#undef NDEBUG  // build is Release (-DNDEBUG); asserts must stay live here
#include "ie/deepseek4_attn.hpp"
#include "ie/deepseek4_ops.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <utility>
#include <vector>

namespace {

int g_fail = 0;
void check(bool ok, const char* what) {
    if (!ok) { std::printf("  FAIL: %s\n", what); ++g_fail; }
}

constexpr int   kSG = 16;
constexpr int   kWG = 256;
constexpr float kNegInf    = -std::numeric_limits<float>::infinity();
constexpr float kMaskedCut = -1e30f;
constexpr double kU = 5.9604644775390625e-08;  // 2^-24, fp32 unit roundoff

inline float softplus_host(float x) { return x > 20.f ? x : std::log1p(std::exp(x)); }

// ===========================================================================
// Verbatim copies of the PRE-CHANGE kernels.  These are the references the new
// shapes are measured against; they are transcriptions, not re-derivations, so
// that "bit-identical to what shipped before" is a checkable statement.
// ===========================================================================

sycl::event ref_attn_plain(sycl::queue& q, const float* q_in, const float* kv,
                           const float* mask, const float* sinks, float* y,
                           uint32_t T, uint32_t n_heads, uint32_t head_dim,
                           uint32_t n_kv, float scaling) {
    return q.submit([&](sycl::handler& h) {
        const uint32_t dpl = head_dim / kSG;
        h.parallel_for(sycl::nd_range<2>({T, size_t(n_heads) * kSG}, {1, kSG}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(kSG)]] {
            const uint32_t t = uint32_t(it.get_group(0));
            const uint32_t hd = uint32_t(it.get_group(1));
            const uint32_t lane = uint32_t(it.get_local_id(1));
            auto sg = it.get_sub_group();
            float qv[32];
            for (uint32_t d = 0; d < dpl; ++d)
                qv[d] = q_in[(size_t(t) * n_heads + hd) * head_dim + lane * dpl + d];
            float m = kNegInf, l = 0.f;
            float acc[32];
            for (uint32_t d = 0; d < dpl; ++d) acc[d] = 0.f;
            for (uint32_t i = 0; i < n_kv; ++i) {
                float mv = 0.f;
                if (mask) { mv = mask[size_t(t) * n_kv + i]; if (!(mv > kMaskedCut)) continue; }
                const float* krow = kv + size_t(i) * head_dim + lane * dpl;
                float part = 0.f;
                for (uint32_t d = 0; d < dpl; ++d) part += qv[d] * krow[d];
                float s = sycl::reduce_over_group(sg, part, sycl::plus<float>()) * scaling + mv;
                if (!(s > kNegInf)) continue;
                const float m_new = sycl::fmax(m, s);
                const float alpha = (m == kNegInf) ? 0.f : sycl::exp(m - m_new);
                const float e = sycl::exp(s - m_new);
                for (uint32_t d = 0; d < dpl; ++d) acc[d] = acc[d] * alpha + e * krow[d];
                l = l * alpha + e;
                m = m_new;
            }
            float corr = 1.f, ll = l;
            if (sinks) {
                const float s = sinks[hd];
                const float m_f = sycl::fmax(m, s);
                corr = (m == kNegInf) ? 0.f : sycl::exp(m - m_f);
                ll = l * corr + sycl::exp(s - m_f);
            }
            const float inv = (ll > 0.f) ? (1.f / ll) : 0.f;
            for (uint32_t d = 0; d < dpl; ++d)
                y[(size_t(t) * n_heads + hd) * head_dim + lane * dpl + d] = acc[d] * corr * inv;
        });
    });
}

template <typename SubGroup>
inline float sg_dot_ref(const SubGroup& sg, const float* a, const float* b, uint32_t n) {
    const uint32_t lane = uint32_t(sg.get_local_linear_id());
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    uint32_t i = lane;
    for (; i + 3u * kSG < n; i += 4u * kSG) {
        a0 += a[i] * b[i];
        a1 += a[i + kSG] * b[i + kSG];
        a2 += a[i + 2u * kSG] * b[i + 2u * kSG];
        a3 += a[i + 3u * kSG] * b[i + 3u * kSG];
    }
    for (; i < n; i += kSG) a0 += a[i] * b[i];
    return sycl::reduce_over_group(sg, (a0 + a1) + (a2 + a3), sycl::plus<float>());
}

sycl::event ref_router_fused(sycl::queue& q, const float* x, const float* w,
                             const float* bias, float* logits, float* weights,
                             int32_t* indices, uint32_t n_tokens, uint32_t hidden,
                             uint32_t n_experts, uint32_t top_k, float routed_scaling) {
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sscore(sycl::range<1>(n_experts), h);
        h.parallel_for(sycl::nd_range<1>(size_t(n_tokens) * kWG, kWG),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
            const uint32_t tok = uint32_t(it.get_group(0));
            const auto sg = it.get_sub_group();
            const uint32_t sgid = uint32_t(sg.get_group_linear_id());
            const uint32_t n_sg = uint32_t(sg.get_group_linear_range());
            const float* xr = x + size_t(tok) * hidden;
            for (uint32_t e = sgid; e < n_experts; e += n_sg) {
                const float l = sg_dot_ref(sg, xr, w + size_t(e) * hidden, hidden);
                if (sg.leader()) {
                    logits[size_t(tok) * n_experts + e] = l;
                    sscore[e] = sycl::sqrt(l > 20.f ? l : sycl::log1p(sycl::exp(l)));
                }
            }
            it.barrier(sycl::access::fence_space::local_space);
            if (it.get_local_id(0) == 0) {
                int32_t* idx = indices + size_t(tok) * top_k;
                for (uint32_t j = 0; j < top_k; ++j) {
                    float best = kNegInf;
                    int32_t best_e = -1;
                    for (uint32_t e = 0; e < n_experts; ++e) {
                        bool taken = false;
                        for (uint32_t p = 0; p < j; ++p) taken |= (idx[p] == int32_t(e));
                        if (taken) continue;
                        const float v = sscore[e] + (bias ? bias[e] : 0.f);
                        if (v > best) { best = v; best_e = int32_t(e); }
                    }
                    idx[j] = best_e;
                }
                float sum = 0.f;
                for (uint32_t j = 0; j < top_k; ++j) sum += sscore[idx[j]];
                const float inv = 1.f / (sum + 1e-20f);
                for (uint32_t j = 0; j < top_k; ++j)
                    weights[size_t(tok) * top_k + j] = sscore[idx[j]] * inv * routed_scaling;
            }
        });
    });
}

// ds4_indexer_score, PRE-CHANGE shape: one sub-group per (query, key), the head
// loop serial inside it, the head_dim dot split ACROSS the lanes — so every head
// costs its own sub-group reduction.
sycl::event ref_iscore_dimlane(sycl::queue& q, const float* q_in, const float* keys,
                               const float* w_proj, float* scores, uint32_t T,
                               uint32_t n_heads, uint32_t head_dim, uint32_t n_keys,
                               float sms, float ws) {
    return q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<2>({T, size_t(n_keys) * kSG}, {1, kSG}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(kSG)]] {
            const uint32_t t = uint32_t(it.get_group(0));
            const uint32_t e = uint32_t(it.get_group(1));
            const uint32_t lane = uint32_t(it.get_local_id(1));
            auto sg = it.get_sub_group();
            const float* krow = keys + size_t(e) * head_dim;
            float total = 0.f;
            for (uint32_t hh = 0; hh < n_heads; ++hh) {
                const float* qrow = q_in + (size_t(t) * n_heads + hh) * head_dim;
                float part = 0.f;
                for (uint32_t d = lane; d < head_dim; d += kSG) part += qrow[d] * krow[d];
                const float dot = sycl::reduce_over_group(sg, part, sycl::plus<float>());
                total += sycl::fmax(dot, 0.f) * sms * (w_proj[size_t(t) * n_heads + hh] * ws);
            }
            if (lane == 0) scores[size_t(t) * n_keys + e] = total;
        });
    });
}

// The head-lane shape, reproduced so it can be deliberately broken (§8b).
//   BUG 0  faithful — must reproduce the shipped kernel bit for bit
//   BUG 1  the last head block is dropped
//   BUG 2  the relu is omitted (negative dots leak into the sum)
//   BUG 3  the lane->head map strides by kSG+1, so some heads are visited twice
//          and others not at all
template <int BUG>
sycl::event iscore_headlane_variant(sycl::queue& q, const float* q_in, const float* keys,
                                    const float* w_proj, float* scores, uint32_t T,
                                    uint32_t n_heads, uint32_t head_dim, uint32_t n_keys,
                                    float sms, float ws) {
    constexpr uint32_t SGS = 16, WG = SGS * kSG;
    constexpr uint32_t STRIDE = (BUG == 3) ? uint32_t(kSG) + 1u : uint32_t(kSG);
    const uint32_t nh = (BUG == 1) ? n_heads - uint32_t(kSG) : n_heads;
    const uint32_t n_grp = (n_keys + SGS - 1u) / SGS;
    return q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<2>({T, size_t(n_grp) * WG}, {1, WG}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(kSG)]] {
            const uint32_t t = uint32_t(it.get_group(0));
            auto sg = it.get_sub_group();
            const uint32_t e = uint32_t(it.get_group(1)) * SGS
                             + uint32_t(sg.get_group_linear_id());
            if (e >= n_keys) return;
            const uint32_t lane = uint32_t(sg.get_local_linear_id());
            const float* krow = keys + size_t(e) * head_dim;
            float tot = 0.f;
            for (uint32_t hb = 0; hb < nh; hb += STRIDE) {
                const uint32_t hh = hb + lane;
                if (hh < n_heads) {
                    const float* qrow = q_in + (size_t(t) * n_heads + hh) * head_dim;
                    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
                    uint32_t d = 0;
                    for (; d + 3u < head_dim; d += 4u) {
                        a0 += qrow[d] * krow[d];
                        a1 += qrow[d + 1u] * krow[d + 1u];
                        a2 += qrow[d + 2u] * krow[d + 2u];
                        a3 += qrow[d + 3u] * krow[d + 3u];
                    }
                    for (; d < head_dim; ++d) a0 += qrow[d] * krow[d];
                    const float dot = (a0 + a1) + (a2 + a3);
                    const float act = (BUG == 2) ? dot : sycl::fmax(dot, 0.f);
                    tot += act * sms * (w_proj[size_t(t) * n_heads + hh] * ws);
                }
            }
            const float r = sycl::reduce_over_group(sg, tot, sycl::plus<float>());
            if (lane == 0) scores[size_t(t) * n_keys + e] = r;
        });
    });
}

// Census of the lane->head map: increment visit[h] once for every head the map
// would PROCESS.  Launched with n_keys == 1 so a head's expected count is 1.
template <uint32_t STRIDE>
sycl::event head_census(sycl::queue& q, uint32_t* visit, uint32_t n_heads) {
    return q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(kSG, kSG),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
            const uint32_t lane = uint32_t(it.get_local_id(0));
            for (uint32_t hb = 0; hb < n_heads; hb += STRIDE) {
                const uint32_t hh = hb + lane;
                if (hh >= n_heads) continue;
                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device> a(visit[hh]);
                a.fetch_add(1u);
            }
        });
    });
}

sycl::event ref_itopk_scan(sycl::queue& q, const float* scores, const int32_t* positions,
                           int32_t* out, uint32_t T, uint32_t n_keys,
                           uint32_t index_topk, uint32_t compress_rate) {
    return q.submit([&](sycl::handler& h) {
        const uint32_t top_k = (index_topk < n_keys) ? index_topk : n_keys;
        h.parallel_for(sycl::nd_range<1>(size_t(T) * kWG, kWG),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
            const uint32_t t = uint32_t(it.get_group(0));
            const uint32_t lid = uint32_t(it.get_local_id(0));
            const auto grp = it.get_group();
            const int64_t thr = (int64_t(positions[t]) + 1) / int64_t(compress_rate);
            uint64_t prev = ~uint64_t(0);
            for (uint32_t j = 0; j < top_k; ++j) {
                uint64_t best = 0;
                for (uint32_t e = lid; e < n_keys; e += kWG) {
                    const float v = (int64_t(e) >= thr) ? kNegInf : scores[size_t(t) * n_keys + e];
                    uint32_t u = sycl::bit_cast<uint32_t>(v);
                    u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
                    const uint64_t k = (uint64_t(u) << 32) | uint64_t(0xFFFFFFFFu - e);
                    if (k < prev && k > best) best = k;
                }
                best = sycl::reduce_over_group(grp, best, sycl::maximum<uint64_t>());
                const int32_t idx = int32_t(0xFFFFFFFFu - uint32_t(best & 0xFFFFFFFFu));
                if (lid == 0) out[size_t(t) * top_k + j] = (int64_t(idx) >= thr) ? -1 : idx;
                prev = best;
            }
        });
    });
}

// ===========================================================================
// The split kernel, reproduced here so it can be deliberately broken.  Mirrors
// the shipped TREE combine: rescale each slice by exp(m_s - gm) once, then fold
// the slice axis pairwise through a staging buffer of only (SP/2)*head_dim.
//   BUG 0  faithful — must reproduce the shipped kernel bit for bit (§1) and
//          pass §3's bound (§6)
//   BUG 1  the last slice is dropped from the combine
//   BUG 2  the running maxima are combined wrongly (slice 0's max is used as
//          the global max instead of the maximum over slices)
//   BUG 3  the partition strides by SP+1, so one residue class of columns is
//          never visited
// ===========================================================================
template <uint32_t SP, int BUG>
sycl::event split_variant(sycl::queue& q, const float* q_in, const float* kv,
                          const float* mask, const float* sinks, float* y,
                          uint32_t T, uint32_t n_heads, uint32_t head_dim,
                          uint32_t n_kv, float scaling) {
    constexpr uint32_t WG = SP * kSG;
    constexpr uint32_t STRIDE = (BUG == 3) ? SP + 1u : SP;
    constexpr uint32_t NCOMB = (BUG == 1) ? SP - 1u : SP;
    return q.submit([&](sycl::handler& h) {
        const uint32_t dpl = head_dim / kSG;
        sycl::local_accessor<float, 1> sacc(sycl::range<1>(size_t(SP / 2) * head_dim), h);
        sycl::local_accessor<float, 1> sml(sycl::range<1>(2 * SP), h);
        h.parallel_for(sycl::nd_range<2>({T, size_t(n_heads) * WG}, {1, WG}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(kSG)]] {
            const uint32_t t = uint32_t(it.get_group(0));
            const uint32_t hd = uint32_t(it.get_group(1));
            auto sg = it.get_sub_group();
            const uint32_t s = uint32_t(sg.get_group_linear_id());
            const uint32_t lane = uint32_t(sg.get_local_linear_id());

            float qv[32];
            for (uint32_t d = 0; d < dpl; ++d)
                qv[d] = q_in[(size_t(t) * n_heads + hd) * head_dim + lane * dpl + d];
            float m = kNegInf, l = 0.f;
            float acc[32];
            for (uint32_t d = 0; d < dpl; ++d) acc[d] = 0.f;
            for (uint32_t i = s; i < n_kv; i += STRIDE) {
                float mv = 0.f;
                if (mask) { mv = mask[size_t(t) * n_kv + i]; if (!(mv > kMaskedCut)) continue; }
                const float* krow = kv + size_t(i) * head_dim + lane * dpl;
                float part = 0.f;
                for (uint32_t d = 0; d < dpl; ++d) part += qv[d] * krow[d];
                float sc = sycl::reduce_over_group(sg, part, sycl::plus<float>()) * scaling + mv;
                if (!(sc > kNegInf)) continue;
                const float m_new = sycl::fmax(m, sc);
                const float alpha = (m == kNegInf) ? 0.f : sycl::exp(m - m_new);
                const float e = sycl::exp(sc - m_new);
                for (uint32_t d = 0; d < dpl; ++d) acc[d] = acc[d] * alpha + e * krow[d];
                l = l * alpha + e;
                m = m_new;
            }
            if (lane == 0) { sml[s] = m; sml[SP + s] = l; }
            it.barrier(sycl::access::fence_space::local_space);

            float gm;
            if (BUG == 2) {
                gm = sml[0];
            } else {
                gm = kNegInf;
                for (uint32_t p = 0; p < NCOMB; ++p) gm = sycl::fmax(gm, sml[p]);
            }
            float gl = 0.f;
            for (uint32_t p = 0; p < NCOMB; ++p)
                gl += (sml[p] == kNegInf) ? 0.f : sml[SP + p] * sycl::exp(sml[p] - gm);

            // BUG 1 drops the last slice by zeroing its contribution before the
            // tree, which is what "not combined" means in this shape.
            const float w = (m == kNegInf || (BUG == 1 && s >= NCOMB))
                                ? 0.f : sycl::exp(m - gm);
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

            float corr = 1.f, ll = gl;
            if (sinks) {
                const float sk = sinks[hd];
                const float m_f = sycl::fmax(gm, sk);
                corr = (gm == kNegInf) ? 0.f : sycl::exp(gm - m_f);
                ll = gl * corr + sycl::exp(sk - m_f);
            }
            const float inv = (ll > 0.f) ? (1.f / ll) : 0.f;
            const float os = corr * inv;
            if (s == 0)
                for (uint32_t d = 0; d < dpl; ++d)
                    y[(size_t(t) * n_heads + hd) * head_dim + lane * dpl + d] = acc[d] * os;
        });
    });
}

// Census of the partition: increment visit[t][i] once for every column the
// stride-STRIDE partition would PROCESS (i.e. that survives the mask test).
// Launched with n_heads == 1 so a live column's expected count is exactly 1.
template <uint32_t SP, uint32_t STRIDE>
sycl::event census(sycl::queue& q, const float* mask, uint32_t* visit,
                   uint32_t T, uint32_t n_kv) {
    constexpr uint32_t WG = SP * kSG;
    return q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<2>({T, WG}, {1, WG}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(kSG)]] {
            const uint32_t t = uint32_t(it.get_group(0));
            auto sg = it.get_sub_group();
            const uint32_t s = uint32_t(sg.get_group_linear_id());
            if (sg.get_local_linear_id() != 0) return;   // one counter per slice
            for (uint32_t i = s; i < n_kv; i += STRIDE) {
                if (!(mask[size_t(t) * n_kv + i] > kMaskedCut)) continue;
                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device>
                    a(visit[size_t(t) * n_kv + i]);
                a.fetch_add(1u);
            }
        });
    });
}

// ===========================================================================
// Test data
// ===========================================================================

// The real decode mask: a `window`-wide sliding causal band over the local rows
// (float::lowest() outside it) concatenated with the indexer's block bias,
// which is -inf everywhere except `top_k` scattered compressed columns.
std::vector<float> make_mask(uint32_t T, uint32_t n_sl, uint32_t n_c, uint32_t window,
                             uint32_t top_k, uint64_t seed) {
    const uint32_t n_kv = n_sl + n_c;
    std::vector<float> m(size_t(T) * n_kv, kNegInf);
    std::mt19937_64 rng(seed);
    const float lowest = std::numeric_limits<float>::lowest();
    for (uint32_t t = 0; t < T; ++t) {
        const int64_t p = int64_t(t) + int64_t(n_sl) - 1;   // decode: the band is full
        for (uint32_t i = 0; i < n_sl; ++i) {
            const bool ok = (int64_t(i) <= p) && (p - int64_t(i) < int64_t(window));
            m[size_t(t) * n_kv + i] = ok ? 0.f : lowest;
        }
        std::vector<uint32_t> idx(n_c);
        for (uint32_t i = 0; i < n_c; ++i) idx[i] = i;
        std::shuffle(idx.begin(), idx.end(), rng);
        for (uint32_t j = 0; j < std::min(top_k, n_c); ++j)
            m[size_t(t) * n_kv + n_sl + idx[j]] = 0.f;
    }
    return m;
}

struct AttnCase {
    uint32_t T, H, D, n_kv;
    std::vector<float> q, kv, mask, sinks;
    float scaling;
};

AttnCase make_case(uint32_t T, uint32_t H, uint32_t D, uint32_t n_sl, uint32_t n_c,
                   uint32_t top_k, uint64_t seed) {
    AttnCase c;
    c.T = T; c.H = H; c.D = D; c.n_kv = n_sl + n_c;
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    c.q.resize(size_t(T) * H * D);
    c.kv.resize(size_t(c.n_kv) * D);
    c.sinks.resize(H);
    for (auto& v : c.q) v = nd(rng) * 0.3f;
    for (auto& v : c.kv) v = nd(rng) * 0.3f;
    for (auto& v : c.sinks) v = nd(rng) * 0.5f;
    c.mask = make_mask(T, n_sl, n_c, 128, top_k, seed ^ 0xbeef);
    c.scaling = 1.f / std::sqrt(float(D));
    return c;
}

// Same case, but every LIVE column in one residue class mod SP carries a large
// positive additive bias, so that slice's running maximum sits `boost` above
// every other slice's.  See §6 for why this regime has to exist.
AttnCase boost_slice(AttnCase c, uint32_t SP, uint32_t residue, float boost) {
    for (uint32_t t = 0; t < c.T; ++t)
        for (uint32_t i = residue; i < c.n_kv; i += SP) {
            float& m = c.mask[size_t(t) * c.n_kv + i];
            if (m > kMaskedCut) m += boost;      // live columns only
        }
    return c;
}

// ---------------------------------------------------------------------------
// Double-precision reference, plus the per-element ABSOLUTE CONTRIBUTION SUM
//
//     A[t,h,d] = ( SUM_i w_i |kv[i,d]| ) / ( SUM_i w_i + w_sink )
//
// which is what every bound below is stated against.  It is the right
// denominator and |y| is not: an attention output is a convex combination of
// value rows, so its components pass through zero and a relative error there is
// unbounded no matter how correct the kernel is.  A is that same combination
// with the signs removed, so it is bounded away from zero exactly when any
// weight reaches the element at all.
// ---------------------------------------------------------------------------
struct HostRef {
    std::vector<double> y, absum;
    uint32_t max_live_per_slice = 0;   // for the SP actually used
    uint32_t n_live_max = 0;
};

HostRef host_reference(const AttnCase& c, uint32_t SP) {
    HostRef r;
    r.y.assign(size_t(c.T) * c.H * c.D, 0.0);
    r.absum.assign(r.y.size(), 0.0);
    const double scaling = double(c.scaling);
    for (uint32_t t = 0; t < c.T; ++t) {
        std::vector<uint32_t> live;
        for (uint32_t i = 0; i < c.n_kv; ++i)
            if (c.mask[size_t(t) * c.n_kv + i] > kMaskedCut) live.push_back(i);
        r.n_live_max = std::max<uint32_t>(r.n_live_max, uint32_t(live.size()));
        std::vector<uint32_t> per(SP, 0);
        for (uint32_t i : live) ++per[i % SP];
        for (uint32_t v : per) r.max_live_per_slice = std::max(r.max_live_per_slice, v);

        for (uint32_t h = 0; h < c.H; ++h) {
            std::vector<double> logit(live.size());
            double mx = -std::numeric_limits<double>::infinity();
            for (size_t z = 0; z < live.size(); ++z) {
                const uint32_t i = live[z];
                double dot = 0.0;
                for (uint32_t d = 0; d < c.D; ++d)
                    dot += double(c.q[(size_t(t) * c.H + h) * c.D + d]) *
                           double(c.kv[size_t(i) * c.D + d]);
                logit[z] = dot * scaling + double(c.mask[size_t(t) * c.n_kv + i]);
                mx = std::max(mx, logit[z]);
            }
            mx = std::max(mx, double(c.sinks[h]));
            double den = std::exp(double(c.sinks[h]) - mx);
            std::vector<double> w(live.size());
            for (size_t z = 0; z < live.size(); ++z) {
                w[z] = std::exp(logit[z] - mx);
                den += w[z];
            }
            for (uint32_t d = 0; d < c.D; ++d) {
                double num = 0.0, absn = 0.0;
                for (size_t z = 0; z < live.size(); ++z) {
                    const double kvv = double(c.kv[size_t(live[z]) * c.D + d]);
                    num += w[z] * kvv;
                    absn += w[z] * std::fabs(kvv);
                }
                r.y[(size_t(t) * c.H + h) * c.D + d] = num / den;
                r.absum[(size_t(t) * c.H + h) * c.D + d] = absn / den;
            }
        }
    }
    return r;
}

// ---------------------------------------------------------------------------
// THE BOUND.
//
// The logits are identical between the plain and split shapes (same lane->dim
// map, same per-column sub-group reduce), so the only difference is HOW the
// softmax numerator and denominator are accumulated.
//
// Within one slice the kernel runs the online recurrence
//     acc <- acc*alpha + e*k ,   l <- l*alpha + e
// once per live column of that slice.  Each step contributes two roundings to
// every already-accumulated term (the multiply by alpha and the fma), so after
// L steps a term carries a relative perturbation bounded by gamma_{2L}; and
// alpha is itself an exp, specified to 3 ulp, through which a term created at
// step j passes for every step j+1..L.  That is (2 + 3)*L*u per term inside a
// slice.
//
// The combine is a TREE, and that changes this term.  Each slice is rescaled to
// the global maximum ONCE — one 3-ulp exp plus one multiply, 4u — and the SP
// rescaled accumulators are then folded pairwise in log2(SP) rounds, one
// rounding each.  The flat combine it replaces re-evaluated exp(m_p - gm) inside
// a serial sum of SP terms and cost 5*SP*u, so this is strictly tighter for
// every SP >= 2.  A constant 16u covers the creating exp(s - m_new), the sink
// fold, the final reciprocal and the output multiply.
//
//     C_split = 5*Lmax + log2(SP) + 4 + 16   Lmax = max live columns in a slice
//     C_plain = 5*Nlive + 16                 the same formula at SP = 1
//
// and the claim gated in §3 is  |y_kernel - y_exact| <= C * u * A  elementwise.
// C_split is far below C_plain across the whole decode regime — the split is not
// merely faster, it is better conditioned, and raising SP now IMPROVES the bound
// on both terms (more slices means fewer live columns each, and the combine
// grows only logarithmically).  §3 prints both so that is visible rather than
// asserted.
// ---------------------------------------------------------------------------
double split_const(uint32_t max_live_per_slice, uint32_t SP) {
    double log2sp = 0.0;
    for (uint32_t v = SP; v > 1u; v >>= 1) log2sp += 1.0;
    return 5.0 * double(max_live_per_slice) + log2sp + 4.0 + 16.0;
}
double plain_const(uint32_t n_live) { return 5.0 * double(n_live) + 16.0; }

// ---------------------------------------------------------------------------
// ds4_indexer_score: exact scores, the ABSOLUTE CONTRIBUTION SUM they are
// measured against, and the bound each shape carries.
//
//     score[t,e] = SUM_h relu(q[t,h,:] . keys[e,:]) * sms * (w[t,h] * ws)
//     A[t,e]     = SUM_h ( SUM_d |q[t,h,d] keys[e,d]| ) * sms * |w[t,h] * ws|
//
// A is the right denominator and |score| is not: the relu zeroes roughly half
// the head terms, so a score can sit arbitrarily close to zero while individual
// head contributions are large, and a relative error against |score| would be
// unbounded no matter how correct the kernel is.
//
// THE BOUND.  Two summations, and the two shapes associate both differently.
//
//   dot product, length D:
//     dimlane   each lane runs a serial chain of D/kSG, then a tree of depth
//               log2(kSG)              -> C_dot = D/16 + 4
//     headlane  one lane runs FOUR chains of D/4, combined as (a0+a1)+(a2+a3)
//                                      -> C_dot = D/4 + 2
//   sum over heads, length n_heads:
//     dimlane   one serial chain       -> C_head = n_heads
//     headlane  a per-lane chain of n_heads/kSG, then a tree of depth log2(kSG)
//                                      -> C_head = n_heads/16 + 4
//
// A term created by the dot passes through the relu (exact — it is a max), then
// two multiplies (2u), then the head summation.  So
//
//     |delta score| <= (C_dot + 2 + C_head) * u * A
//
// At the decode shape (D = 128, n_heads = 64) that is 78u for the old shape and
// 44u for the new one: the head sum is the longer chain and the new shape
// shortens it by more than it lengthens the dot.  §7 prints both constants and
// both observed deviations, so "the new shape is better conditioned" is a
// measured statement rather than an assertion.
// ---------------------------------------------------------------------------
double iscore_const_dimlane(uint32_t head_dim, uint32_t n_heads) {
    return (double(head_dim) / 16.0 + 4.0) + 2.0 + double(n_heads);
}
double iscore_const_headlane(uint32_t head_dim, uint32_t n_heads) {
    return (double(head_dim) / 4.0 + 2.0) + 2.0 + (double(n_heads) / 16.0 + 4.0);
}

struct IScoreRef {
    std::vector<double> score, absum;
};

IScoreRef iscore_reference(const std::vector<float>& q, const std::vector<float>& k,
                           const std::vector<float>& w, uint32_t T, uint32_t n_heads,
                           uint32_t head_dim, uint32_t n_keys, float sms, float ws) {
    IScoreRef r;
    r.score.assign(size_t(T) * n_keys, 0.0);
    r.absum.assign(r.score.size(), 0.0);
    for (uint32_t t = 0; t < T; ++t)
        for (uint32_t e = 0; e < n_keys; ++e) {
            double tot = 0.0, abs_tot = 0.0;
            for (uint32_t h = 0; h < n_heads; ++h) {
                double dot = 0.0, mag = 0.0;
                for (uint32_t d = 0; d < head_dim; ++d) {
                    const double p = double(q[(size_t(t) * n_heads + h) * head_dim + d]) *
                                     double(k[size_t(e) * head_dim + d]);
                    dot += p;
                    mag += std::fabs(p);
                }
                const double wgt = double(sms) * double(w[size_t(t) * n_heads + h]) * double(ws);
                tot     += std::max(dot, 0.0) * wgt;
                abs_tot += mag * std::fabs(wgt);
            }
            r.score[size_t(t) * n_keys + e] = tot;
            r.absum[size_t(t) * n_keys + e] = abs_tot;
        }
    return r;
}

// Worst |delta| / A over the tensor.  Elements where A is exactly zero (no
// weight reaches them at all) must match exactly, and are counted separately.
//
// Non-finite outputs are counted, NOT folded into `worst`: std::max(x, NaN)
// returns x, so a NaN would otherwise vanish silently and a kernel that
// overflowed to inf/NaN would be scored as passing.  §6's overflow control
// produces exactly that, so this counter is load-bearing.
double worst_over_absum(const std::vector<float>& got, const std::vector<double>& ref,
                        const std::vector<double>& absum, size_t* zero_mismatch,
                        size_t* nonfinite) {
    double worst = 0.0;
    *zero_mismatch = 0;
    *nonfinite = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        if (!std::isfinite(got[i])) { ++*nonfinite; continue; }
        const double d = std::fabs(double(got[i]) - ref[i]);
        if (absum[i] <= 0.0) { if (d != 0.0) ++*zero_mismatch; continue; }
        worst = std::max(worst, d / absum[i]);
    }
    return worst;
}

struct AttnDev {
    float *q = nullptr, *kv = nullptr, *mask = nullptr, *sinks = nullptr, *y = nullptr;
    size_t ny = 0;
    void alloc(sycl::queue& qu, const AttnCase& c) {
        ny = size_t(c.T) * c.H * c.D;
        q = sycl::malloc_device<float>(c.q.size(), qu);
        kv = sycl::malloc_device<float>(c.kv.size(), qu);
        mask = sycl::malloc_device<float>(c.mask.size(), qu);
        sinks = sycl::malloc_device<float>(c.sinks.size(), qu);
        y = sycl::malloc_device<float>(ny, qu);
        qu.memcpy(q, c.q.data(), c.q.size() * 4).wait();
        qu.memcpy(kv, c.kv.data(), c.kv.size() * 4).wait();
        qu.memcpy(mask, c.mask.data(), c.mask.size() * 4).wait();
        qu.memcpy(sinks, c.sinks.data(), c.sinks.size() * 4).wait();
    }
    void free(sycl::queue& qu) {
        sycl::free(q, qu); sycl::free(kv, qu); sycl::free(mask, qu);
        sycl::free(sinks, qu); sycl::free(y, qu);
    }
};

double time_ms(sycl::queue& q, int reps, const std::function<void()>& fn) {
    fn(); q.wait();
    double best = 1e30;
    for (int trial = 0; trial < 3; ++trial) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) fn();
        q.wait();
        best = std::min(best, std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - t0).count() / reps);
    }
    return best;
}

// The SP the shipped dispatcher will pick, mirrored here so the bound describes
// the shape actually under test.  §1 fails loudly if the two ever disagree.
uint32_t expected_sp(const sycl::device& dev, uint32_t T, uint32_t head_dim) {
    uint32_t sp = 0;
    if      (T <= 1u)  sp = 64u;
    else if (T <= 2u)  sp = 32u;
    else if (T <= 16u) sp = 16u;
    else               return 0u;
    const uint64_t slm = dev.get_info<sycl::info::device::local_mem_size>();
    const size_t max_wg = dev.get_info<sycl::info::device::max_work_group_size>();
    while (sp > 1u && (uint64_t(sp / 2u) * head_dim * sizeof(float)
                           + uint64_t(2u * sp) * sizeof(float) > slm ||
                       size_t(sp) * kSG > max_wg))
        sp /= 2u;
    return (sp > 1u) ? sp : 0u;
}

struct Shape { const char* name; uint32_t T, H, D, n_sl, n_c, top_k; };

}  // namespace

// ===========================================================================
int main() {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    const sycl::device dev = q.get_device();
    std::printf("device: %s   (%u CUs, %llu B SLM, max WG %zu)\n\n",
                dev.get_info<sycl::info::device::name>().c_str(),
                dev.get_info<sycl::info::device::max_compute_units>(),
                (unsigned long long)dev.get_info<sycl::info::device::local_mem_size>(),
                dev.get_info<sycl::info::device::max_work_group_size>());

    // The shapes that matter: V4-Flash decode on a two-card head split
    // (n_heads 32 of 64), head_dim 512, sliding window 128, indexer top-512.
    // The last one is deliberately ragged — n_kv is coprime with every SP — so
    // the strided partition's tail is exercised rather than assumed.
    const Shape shapes[] = {
        {"decode ctx4096",      1, 32, 512, 128, 1024, 512},
        {"decode ctx16k",       1, 32, 512, 128, 4096, 512},
        {"indexer d=128 T=1",   1, 32, 128, 128, 1024, 512},
        {"spec-decode T=4",     4, 32, 512, 128, 1024, 512},
        {"ragged n_kv",         1,  8, 512, 100,  677, 311},
    };

    // ---------------------------------------------------------------------
    std::puts("§1 the shipped kernel IS the split shape this test reasons about");
    for (const Shape& sh : shapes) {
        const uint32_t SP = expected_sp(dev, sh.T, sh.D);
        if (!SP) { std::printf("  %-22s dispatches to the plain shape\n", sh.name); continue; }
        const AttnCase c = make_case(sh.T, sh.H, sh.D, sh.n_sl, sh.n_c, sh.top_k, 101);
        AttnDev dv; dv.alloc(q, c);
        std::vector<float> shipped(dv.ny), local(dv.ny);
        ie::ds4_attention(q, dv.q, dv.kv, dv.mask, dv.sinks, dv.y,
                          c.T, c.H, c.D, c.n_kv, c.scaling).wait();
        q.memcpy(shipped.data(), dv.y, dv.ny * 4).wait();
        switch (SP) {
            case 64: split_variant<64, 0>(q, dv.q, dv.kv, dv.mask, dv.sinks, dv.y,
                                          c.T, c.H, c.D, c.n_kv, c.scaling).wait(); break;
            case 32: split_variant<32, 0>(q, dv.q, dv.kv, dv.mask, dv.sinks, dv.y,
                                          c.T, c.H, c.D, c.n_kv, c.scaling).wait(); break;
            case 16: split_variant<16, 0>(q, dv.q, dv.kv, dv.mask, dv.sinks, dv.y,
                                          c.T, c.H, c.D, c.n_kv, c.scaling).wait(); break;
            default: check(false, "expected_sp returned a factor this test has no "
                                  "instantiation for — the ladder and the test have "
                                  "drifted apart"); continue;
        }
        q.memcpy(local.data(), dv.y, dv.ny * 4).wait();
        size_t mism = 0;
        for (size_t i = 0; i < dv.ny; ++i)
            if (std::memcmp(&shipped[i], &local[i], 4) != 0) ++mism;
        std::printf("  %-22s SP=%-3u  bitwise mismatches vs local copy %zu\n",
                    sh.name, SP, mism);
        check(mism == 0, "shipped ds4_attention does not match split_variant<SP,0> — "
                         "the dispatcher and this test disagree about the shape");
        dv.free(q);
    }

    // ---------------------------------------------------------------------
    std::puts("\n§2 the partition visits every live column EXACTLY once");
    {
        const AttnCase c = make_case(3, 1, 512, 100, 677, 311, 202);
        float* dmask = sycl::malloc_device<float>(c.mask.size(), q);
        uint32_t* dv = sycl::malloc_device<uint32_t>(size_t(c.T) * c.n_kv, q);
        q.memcpy(dmask, c.mask.data(), c.mask.size() * 4).wait();
        std::vector<uint32_t> vis(size_t(c.T) * c.n_kv);

        auto run_census = [&](const char* label, const std::function<void()>& launch,
                              bool expect_ok) {
            q.memset(dv, 0, vis.size() * 4).wait();
            launch();
            q.wait();
            q.memcpy(vis.data(), dv, vis.size() * 4).wait();
            size_t missed = 0, doubled = 0, phantom = 0, live = 0;
            for (size_t i = 0; i < vis.size(); ++i) {
                const bool is_live = c.mask[i] > kMaskedCut;
                if (is_live) ++live;
                if (is_live && vis[i] == 0) ++missed;
                if (is_live && vis[i] > 1) ++doubled;
                if (!is_live && vis[i] != 0) ++phantom;
            }
            std::printf("  %-30s live %5zu  missed %5zu  double %5zu  masked-but-visited %5zu\n",
                        label, live, missed, doubled, phantom);
            const bool ok = (missed == 0 && doubled == 0 && phantom == 0);
            if (expect_ok)
                check(ok, "the stride partition does not cover the live set exactly once");
            else
                check(!ok, "NEGATIVE CONTROL DID NOT FIRE: a stride-(SP+1) partition was "
                           "reported as complete — the census cannot see dropped columns");
        };
        run_census("SP=64 stride 64",
                   [&]{ census<64, 64>(q, dmask, dv, c.T, c.n_kv); }, true);
        run_census("SP=32 stride 32",
                   [&]{ census<32, 32>(q, dmask, dv, c.T, c.n_kv); }, true);
        run_census("SP=16 stride 16",
                   [&]{ census<16, 16>(q, dmask, dv, c.T, c.n_kv); }, true);
        run_census("SP=8  stride 8",
                   [&]{ census<8, 8>(q, dmask, dv, c.T, c.n_kv); }, true);
        run_census("SP=64 stride 65 (BROKEN)",
                   [&]{ census<64, 65>(q, dmask, dv, c.T, c.n_kv); }, false);
        run_census("SP=32 stride 33 (BROKEN)",
                   [&]{ census<32, 33>(q, dmask, dv, c.T, c.n_kv); }, false);
        sycl::free(dmask, q); sycl::free(dv, q);
    }

    // ---------------------------------------------------------------------
    std::puts("\n§3 ds4_attention vs a double-precision host softmax,\n"
              "   bound derived from the split structure (see split_const)");
    for (const Shape& sh : shapes) {
        const uint32_t SP = expected_sp(dev, sh.T, sh.D);
        const AttnCase c = make_case(sh.T, sh.H, sh.D, sh.n_sl, sh.n_c, sh.top_k, 303);
        const HostRef ref = host_reference(c, SP ? SP : 1);
        AttnDev dv; dv.alloc(q, c);
        std::vector<float> got(dv.ny), gotp(dv.ny);

        ie::ds4_attention(q, dv.q, dv.kv, dv.mask, dv.sinks, dv.y,
                          c.T, c.H, c.D, c.n_kv, c.scaling).wait();
        q.memcpy(got.data(), dv.y, dv.ny * 4).wait();
        size_t zbad = 0, nf = 0;
        const double obs = worst_over_absum(got, ref.y, ref.absum, &zbad, &nf);
        const double bound = (SP ? split_const(ref.max_live_per_slice, SP)
                                 : plain_const(ref.n_live_max)) * kU;

        ref_attn_plain(q, dv.q, dv.kv, dv.mask, dv.sinks, dv.y,
                       c.T, c.H, c.D, c.n_kv, c.scaling).wait();
        q.memcpy(gotp.data(), dv.y, dv.ny * 4).wait();
        size_t zbadp = 0, nfp = 0;
        const double obsp = worst_over_absum(gotp, ref.y, ref.absum, &zbadp, &nfp);
        const double boundp = plain_const(ref.n_live_max) * kU;
        check(nf == 0 && nfp == 0, "ds4_attention produced a non-finite output");

        std::printf("  %-22s SP=%-3u live<=%4u (<=%3u per slice)\n",
                    sh.name, SP, ref.n_live_max, ref.max_live_per_slice);
        std::printf("      shipped observed %.3e vs bound %.3e  (%6.2f%% of bound)\n",
                    obs, bound, 100.0 * obs / bound);
        std::printf("      plain   observed %.3e vs bound %.3e  (%6.2f%% of bound)\n",
                    obsp, boundp, 100.0 * obsp / boundp);
        check(obs <= bound, "ds4_attention exceeds its derived reassociation bound");
        check(zbad == 0, "ds4_attention is nonzero where no weight reaches the element");
        check(obsp <= boundp, "the plain reference shape exceeds ITS derived bound — "
                              "the derivation itself is wrong");
        dv.free(q);
    }

    // ---------------------------------------------------------------------
    std::puts("\n§4 ds4_router_topk is BIT-IDENTICAL to the fused kernel it replaces");
    {
        const uint32_t H = 4096, E = 256, K = 6;
        std::mt19937_64 rng(404);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (uint32_t T : {1u, 2u, 4u, 16u}) {
            std::vector<float> hx(size_t(T) * H), hw(size_t(E) * H), hb(E);
            for (auto& v : hx) v = nd(rng) * 0.05f;
            for (auto& v : hw) v = nd(rng) * 0.02f;
            for (auto& v : hb) v = nd(rng) * 0.01f;
            float* dx = sycl::malloc_device<float>(hx.size(), q);
            float* dw = sycl::malloc_device<float>(hw.size(), q);
            float* db = sycl::malloc_device<float>(hb.size(), q);
            float* dl = sycl::malloc_device<float>(size_t(T) * E, q);
            float* dwt = sycl::malloc_device<float>(size_t(T) * K, q);
            int32_t* di = sycl::malloc_device<int32_t>(size_t(T) * K, q);
            q.memcpy(dx, hx.data(), hx.size() * 4).wait();
            q.memcpy(dw, hw.data(), hw.size() * 4).wait();
            q.memcpy(db, hb.data(), hb.size() * 4).wait();

            std::vector<float> l0(size_t(T) * E), l1(l0.size());
            std::vector<float> w0(size_t(T) * K), w1(w0.size());
            std::vector<int32_t> i0(size_t(T) * K), i1(i0.size());
            ref_router_fused(q, dx, dw, db, dl, dwt, di, T, H, E, K, 2.5f).wait();
            q.memcpy(l0.data(), dl, l0.size() * 4).wait();
            q.memcpy(w0.data(), dwt, w0.size() * 4).wait();
            q.memcpy(i0.data(), di, i0.size() * 4).wait();
            q.memset(dl, 0, l0.size() * 4).wait();
            ie::ds4_router_topk(q, dx, dw, db, dl, dwt, di, T, H, E, K, 2.5f).wait();
            q.memcpy(l1.data(), dl, l1.size() * 4).wait();
            q.memcpy(w1.data(), dwt, w1.size() * 4).wait();
            q.memcpy(i1.data(), di, i1.size() * 4).wait();

            size_t bl = 0, bw = 0, bi = 0;
            for (size_t z = 0; z < l0.size(); ++z) if (std::memcmp(&l0[z], &l1[z], 4)) ++bl;
            for (size_t z = 0; z < w0.size(); ++z) if (std::memcmp(&w0[z], &w1[z], 4)) ++bw;
            for (size_t z = 0; z < i0.size(); ++z) if (i0[z] != i1[z]) ++bi;

            // How close the 6th and 7th ranked experts come, so the exactness
            // claim comes with a statement of how much headroom it has.
            double min_gap = 1e30;
            for (uint32_t t = 0; t < T; ++t) {
                std::vector<float> v(E);
                for (uint32_t e = 0; e < E; ++e)
                    v[e] = std::sqrt(softplus_host(l0[size_t(t) * E + e])) + hb[e];
                std::sort(v.begin(), v.end(), std::greater<float>());
                min_gap = std::min(min_gap, double(v[K - 1]) - double(v[K]));
            }
            std::printf("  T=%-3u logits %zu  weights %zu  indices %zu mismatched"
                        "   (top-6/7 score gap %.3e)\n", T, bl, bw, bi, min_gap);
            check(bl == 0, "ds4_router_topk logits are not bit-identical to the fused kernel");
            check(bw == 0, "ds4_router_topk weights are not bit-identical to the fused kernel");
            check(bi == 0, "ds4_router_topk selected different experts");
            sycl::free(dx, q); sycl::free(dw, q); sycl::free(db, q);
            sycl::free(dl, q); sycl::free(dwt, q); sycl::free(di, q);
        }
    }

    // ---------------------------------------------------------------------
    std::puts("\n§5 ds4_indexer_topk is index-for-index identical to the scan it replaces");
    {
        std::mt19937_64 rng(505);
        std::normal_distribution<float> nd(0.f, 1.f);
        // The two real decode sizes, a non-power-of-two n_keys, a threshold low
        // enough to force -1 sentinels, and one n_keys below index_topk so
        // top_k saturates.
        struct S { uint32_t T, n_keys; int32_t pos; };
        const S cases[] = {
            {1, 1024, 4095}, {1, 4096, 16383}, {4, 1024, 4095},
            {1, 1000, 3999}, {1, 1024, 2047},  {1,  300, 1199}, {16, 1024, 4095},
            // The radix shape's own regimes (2026-08-09): the bitonic SLM
            // boundary, and the ctx-200k class (n_keys 50000) where bitonic
            // cannot fit and the scan shape was the only — unusable — option.
            {1, 16384, 65535}, {1, 50000, 199999},
        };
        for (const S& s : cases) {
            const uint32_t top_k = std::min(512u, s.n_keys);
            std::vector<float> hs(size_t(s.T) * s.n_keys);
            for (auto& v : hs) v = nd(rng);
            // Force exact ties so the tie-break rule is actually exercised.
            for (size_t z = 0; z + 7 < hs.size(); z += 7) hs[z] = 0.5f;
            std::vector<int32_t> hp(s.T, s.pos);
            float* dsc = sycl::malloc_device<float>(hs.size(), q);
            int32_t* dp = sycl::malloc_device<int32_t>(hp.size(), q);
            int32_t* dout = sycl::malloc_device<int32_t>(size_t(s.T) * top_k, q);
            q.memcpy(dsc, hs.data(), hs.size() * 4).wait();
            q.memcpy(dp, hp.data(), hp.size() * 4).wait();
            std::vector<int32_t> o0(size_t(s.T) * top_k), o1(o0.size());
            ref_itopk_scan(q, dsc, dp, dout, s.T, s.n_keys, 512, 4).wait();
            q.memcpy(o0.data(), dout, o0.size() * 4).wait();
            q.memset(dout, 0x5A, o1.size() * 4).wait();
            ie::ds4_indexer_topk(q, dsc, dp, dout, s.T, s.n_keys, 512, 4).wait();
            q.memcpy(o1.data(), dout, o1.size() * 4).wait();
            size_t bad = 0, oob = 0, sent = 0;
            for (size_t z = 0; z < o0.size(); ++z) {
                if (o0[z] != o1[z]) {
                    if (!bad)
                        std::printf("    first mismatch at slot %zu: ref=%d got=%d "
                                    "(next ref/got: %d/%d %d/%d)\n",
                                    z, o0[z], o1[z],
                                    z + 1 < o0.size() ? o0[z + 1] : -99,
                                    z + 1 < o1.size() ? o1[z + 1] : -99,
                                    z + 2 < o0.size() ? o0[z + 2] : -99,
                                    z + 2 < o1.size() ? o1[z + 2] : -99);
                    ++bad;
                }
                if (o1[z] == -1) ++sent;
                else if (o1[z] < 0 || uint32_t(o1[z]) >= s.n_keys) ++oob;
            }
            std::printf("  T=%-3u n_keys=%-5u pos=%-6d top_k=%-4u  mismatched %zu  "
                        "out-of-range %zu  (-1 sentinels %zu)\n",
                        s.T, s.n_keys, s.pos, top_k, bad, oob, sent);
            check(bad == 0, "ds4_indexer_topk disagrees with the scan implementation");
            check(oob == 0, "ds4_indexer_topk emitted an index outside [0, n_keys) — "
                            "a padded sort slot escaped into the output");
            sycl::free(dsc, q); sycl::free(dp, q); sycl::free(dout, q);
        }
    }

    // ---------------------------------------------------------------------
    // -----------------------------------------------------------------------
    // §6 exists to show §3's bound is worth passing.  Three deliberate breaks,
    // over TWO data regimes, because one of the three is invisible in the first.
    //
    // WHY TWO REGIMES.  BUG 2 replaces the global running max with slice 0's.
    // That does NOT change the answer: writing gm' for the substitute,
    // a' = a*exp(gm-gm') and gl' = gl*exp(gm-gm'), and the final
    //     y = a'*exp(gm') / (gl'*exp(gm') + exp(sink))
    // has exp(gm') cancel top and bottom — y is independent of gm' in exact
    // arithmetic.  The running max is a RANGE device, not a value device: its
    // only job is to keep exp() inside fp32.  So on decode-like data, where
    // every slice maximum sits within a few units of every other, the broken
    // combine is genuinely harmless and passing the bound proves nothing about
    // it.  (Discovered by this control failing to fire, not assumed.)
    //
    // Regime B therefore pushes one residue class's logits +95 above the rest,
    // which is what the max subtraction is FOR.  The faithful combine weights
    // it by exp(m_p - gm) <= 1 and stays finite; the broken one evaluates
    // exp(+95) = 1.8e41, overflows fp32, and returns inf/NaN — which is why
    // worst_over_absum counts non-finite outputs separately instead of letting
    // std::max swallow them.
    // -----------------------------------------------------------------------
    std::puts("\n§6 negative controls — each break must be caught in at least one regime");
    {
        const uint32_t SP = 64;   // the factor the ladder picks at T=1
        const AttnCase base = make_case(1, 32, 512, 128, 1024, 512, 606);
        struct Regime { const char* name; AttnCase c; };
        Regime regimes[2] = {
            {"A uniform logits", base},
            {"B one slice +95", boost_slice(base, SP, 5, 95.f)},
        };
        bool caught_any[4] = {false, false, false, false};
        for (Regime& rg : regimes) {
            const AttnCase& c = rg.c;
            const HostRef ref = host_reference(c, SP);
            const double bound = split_const(ref.max_live_per_slice, SP) * kU;
            AttnDev dv; dv.alloc(q, c);
            std::vector<float> got(dv.ny);
            std::printf("  regime %s (bound %.3e)\n", rg.name, bound);

            // caught == exceeded the bound OR produced a non-finite output.
            auto probe = [&](const char* label, const std::function<void()>& launch) {
                launch(); q.wait();
                q.memcpy(got.data(), dv.y, dv.ny * 4).wait();
                size_t zbad = 0, nf = 0;
                const double obs = worst_over_absum(got, ref.y, ref.absum, &zbad, &nf);
                const bool caught = (obs > bound) || (nf > 0);
                std::printf("    %-34s observed %.3e  non-finite %6zu  %s\n",
                            label, obs, nf, caught ? "CAUGHT" : "within");
                return caught;
            };
            const bool ok0 = probe("faithful (control of controls)", [&]{
                split_variant<64, 0>(q, dv.q, dv.kv, dv.mask, dv.sinks, dv.y,
                                     c.T, c.H, c.D, c.n_kv, c.scaling); });
            const bool c1 = probe("BROKEN: last partial dropped", [&]{
                split_variant<64, 1>(q, dv.q, dv.kv, dv.mask, dv.sinks, dv.y,
                                     c.T, c.H, c.D, c.n_kv, c.scaling); });
            const bool c2 = probe("BROKEN: running max not combined", [&]{
                split_variant<64, 2>(q, dv.q, dv.kv, dv.mask, dv.sinks, dv.y,
                                     c.T, c.H, c.D, c.n_kv, c.scaling); });
            const bool c3 = probe("BROKEN: partition misses columns", [&]{
                split_variant<64, 3>(q, dv.q, dv.kv, dv.mask, dv.sinks, dv.y,
                                     c.T, c.H, c.D, c.n_kv, c.scaling); });

            // The faithful variant must stay inside the bound in BOTH regimes —
            // including the one that overflows the broken combine.  That is the
            // positive claim: the max subtraction does its job.
            check(!ok0, "the faithful split variant failed its own bound");
            caught_any[1] |= c1;
            caught_any[2] |= c2;
            caught_any[3] |= c3;
            dv.free(q);
        }
        // Each break has to be visible SOMEWHERE.  Requiring every break in
        // every regime would be wrong, not stricter: under regime B one slice
        // holds essentially all the softmax mass, so dropping columns outside
        // it genuinely does not change the answer, and asserting otherwise
        // would be asserting something false.
        check(caught_any[1], "NEGATIVE CONTROL DID NOT FIRE: dropping a partial was never "
                             "caught, so the combine's coverage is untested");
        check(caught_any[2], "NEGATIVE CONTROL DID NOT FIRE: an uncombined running max "
                             "survived a 95-unit logit spread, so the max combine is untested");
        check(caught_any[3], "NEGATIVE CONTROL DID NOT FIRE: a partition that misses columns "
                             "was never caught, so the partition is untested");
    }

    // ---------------------------------------------------------------------
    // §7 ds4_indexer_score changed from the dim-lane shape (one sub-group per
    // key, n_heads sub-group reductions) to the head-lane shape (lane l owns
    // heads l, l+kSG, ..., ONE reduction per key).  It is not bit-identical —
    // both the dot product and the sum over heads reassociate — so it is gated
    // the same way ds4_attention is: a bound derived from the two chain
    // structures, printed against the observed deviation for BOTH shapes, with
    // negative controls that must exceed it.
    // ---------------------------------------------------------------------
    std::puts("\n§7 ds4_indexer_score vs a double-precision host reference,\n"
              "   bounds derived from the two summation structures");
    struct IShape { const char* name; uint32_t T, H, D, n_keys; };
    const IShape ishapes[] = {
        {"decode ctx4096",   1, 64, 128, 1024},
        {"decode ctx16k",    1, 64, 128, 4096},
        {"spec-decode T=4",  4, 64, 128, 1024},
        {"ragged n_keys",    1, 64, 128,  777},
        {"n_heads 40 (<kSG mult)", 1, 40, 128, 512},
        {"n_heads 8 (dimlane path)", 1, 8, 128, 512},
    };
    for (const IShape& sh : ishapes) {
        std::mt19937_64 rng(0x15c0 ^ sh.n_keys ^ (uint64_t(sh.H) << 20));
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> hq(size_t(sh.T) * sh.H * sh.D), hk(size_t(sh.n_keys) * sh.D),
                           hw(size_t(sh.T) * sh.H);
        for (auto& v : hq) v = nd(rng) * 0.3f;
        for (auto& v : hk) v = nd(rng) * 0.3f;
        for (auto& v : hw) v = nd(rng) * 0.5f;
        const float sms = 1.f / std::sqrt(float(sh.D));
        const float ws  = 1.f / std::sqrt(float(sh.H));
        const IScoreRef ref = iscore_reference(hq, hk, hw, sh.T, sh.H, sh.D, sh.n_keys, sms, ws);

        float* dq = sycl::malloc_device<float>(hq.size(), q);
        float* dk = sycl::malloc_device<float>(hk.size(), q);
        float* dw = sycl::malloc_device<float>(hw.size(), q);
        float* dsc = sycl::malloc_device<float>(ref.score.size(), q);
        q.memcpy(dq, hq.data(), hq.size() * 4).wait();
        q.memcpy(dk, hk.data(), hk.size() * 4).wait();
        q.memcpy(dw, hw.data(), hw.size() * 4).wait();
        std::vector<float> s_new(ref.score.size()), s_old(ref.score.size());

        ie::ds4_indexer_score(q, dq, dk, dw, dsc, sh.T, sh.H, sh.D, sh.n_keys, sms, ws).wait();
        q.memcpy(s_new.data(), dsc, s_new.size() * 4).wait();
        q.memset(dsc, 0, s_new.size() * 4).wait();
        ref_iscore_dimlane(q, dq, dk, dw, dsc, sh.T, sh.H, sh.D, sh.n_keys, sms, ws).wait();
        q.memcpy(s_old.data(), dsc, s_old.size() * 4).wait();

        size_t zn = 0, zo = 0, nfn = 0, nfo = 0;
        const double obs_new = worst_over_absum(s_new, ref.score, ref.absum, &zn, &nfn);
        const double obs_old = worst_over_absum(s_old, ref.score, ref.absum, &zo, &nfo);
        // The shipped dispatcher takes the head-lane shape only at n_heads >= kSG;
        // below that it IS ref_iscore_dimlane and must match it bit for bit.
        const bool head_path = sh.H >= uint32_t(kSG);
        const double bnd_new = (head_path ? iscore_const_headlane(sh.D, sh.H)
                                          : iscore_const_dimlane(sh.D, sh.H)) * kU;
        const double bnd_old = iscore_const_dimlane(sh.D, sh.H) * kU;
        std::printf("  %-26s T=%-3u n_keys=%-5u n_heads=%-3u  %s\n",
                    sh.name, sh.T, sh.n_keys, sh.H, head_path ? "head-lane" : "dim-lane");
        std::printf("      shipped observed %.3e vs bound %.3e  (%6.2f%% of bound, C=%.0f)\n",
                    obs_new, bnd_new, 100.0 * obs_new / bnd_new,
                    head_path ? iscore_const_headlane(sh.D, sh.H)
                              : iscore_const_dimlane(sh.D, sh.H));
        std::printf("      dimlane observed %.3e vs bound %.3e  (%6.2f%% of bound, C=%.0f)\n",
                    obs_old, bnd_old, 100.0 * obs_old / bnd_old,
                    iscore_const_dimlane(sh.D, sh.H));
        check(nfn == 0 && nfo == 0, "ds4_indexer_score produced a non-finite score");
        check(obs_new <= bnd_new, "ds4_indexer_score exceeds its derived bound");
        check(obs_old <= bnd_old, "the dim-lane reference exceeds ITS derived bound — "
                                  "the derivation itself is wrong");
        check(zn == 0 && zo == 0,
              "ds4_indexer_score is nonzero where no head contributes anything");
        if (!head_path) {
            size_t bits = 0;
            for (size_t z = 0; z < s_new.size(); ++z)
                if (std::memcmp(&s_new[z], &s_old[z], 4) != 0) ++bits;
            std::printf("      below kSG heads: bitwise mismatches vs dim-lane %zu\n", bits);
            check(bits == 0, "n_heads < kSG must still take the dim-lane shape unchanged");
        }

        // ---- §7b negative controls, on the decode shape only ----
        if (sh.T == 1 && sh.n_keys == 1024 && sh.H == 64) {
            std::puts("      negative controls (each break must exceed the bound):");
            auto probe = [&](const char* label, const std::function<void()>& launch) {
                q.memset(dsc, 0, s_new.size() * 4).wait();
                launch(); q.wait();
                std::vector<float> got(s_new.size());
                q.memcpy(got.data(), dsc, got.size() * 4).wait();
                size_t zb = 0, nf = 0;
                const double o = worst_over_absum(got, ref.score, ref.absum, &zb, &nf);
                const bool caught = (o > bnd_new) || (nf > 0);
                std::printf("        %-32s observed %.3e  %s\n", label, o,
                            caught ? "CAUGHT" : "within");
                return caught;
            };
            const bool ok0 = probe("faithful (control of controls)", [&]{
                iscore_headlane_variant<0>(q, dq, dk, dw, dsc, sh.T, sh.H, sh.D,
                                           sh.n_keys, sms, ws); });
            const bool c1 = probe("BROKEN: last head block dropped", [&]{
                iscore_headlane_variant<1>(q, dq, dk, dw, dsc, sh.T, sh.H, sh.D,
                                           sh.n_keys, sms, ws); });
            const bool c2 = probe("BROKEN: relu omitted", [&]{
                iscore_headlane_variant<2>(q, dq, dk, dw, dsc, sh.T, sh.H, sh.D,
                                           sh.n_keys, sms, ws); });
            const bool c3 = probe("BROKEN: lane->head map strides kSG+1", [&]{
                iscore_headlane_variant<3>(q, dq, dk, dw, dsc, sh.T, sh.H, sh.D,
                                           sh.n_keys, sms, ws); });
            check(!ok0, "the faithful head-lane variant failed its own bound");
            check(c1, "NEGATIVE CONTROL DID NOT FIRE: dropping a head block was not caught");
            check(c2, "NEGATIVE CONTROL DID NOT FIRE: dropping the relu was not caught");
            check(c3, "NEGATIVE CONTROL DID NOT FIRE: a lane->head map that skips heads "
                      "was not caught");
            // and the faithful variant must BE the shipped kernel, bit for bit
            q.memset(dsc, 0, s_new.size() * 4).wait();
            iscore_headlane_variant<0>(q, dq, dk, dw, dsc, sh.T, sh.H, sh.D,
                                       sh.n_keys, sms, ws).wait();
            std::vector<float> local(s_new.size());
            q.memcpy(local.data(), dsc, local.size() * 4).wait();
            size_t bits = 0;
            for (size_t z = 0; z < s_new.size(); ++z)
                if (std::memcmp(&s_new[z], &local[z], 4) != 0) ++bits;
            std::printf("        shipped vs local copy: bitwise mismatches %zu\n", bits);
            check(bits == 0, "shipped ds4_indexer_score does not match "
                             "iscore_headlane_variant<0> — the test and the dispatcher "
                             "disagree about the shape");
        }
        sycl::free(dq, q); sycl::free(dk, q); sycl::free(dw, q); sycl::free(dsc, q);
    }

    // ---------------------------------------------------------------------
    // §7c the lane->head map covers every head exactly once.  Same argument as
    // §2's column census, same failure mode: a map that silently skips a head
    // is the dangerous break, so it is counted rather than argued.
    // ---------------------------------------------------------------------
    std::puts("\n§7c the lane->head map visits every head EXACTLY once");
    {
        uint32_t* dv = sycl::malloc_device<uint32_t>(128, q);
        auto run = [&](const char* label, uint32_t n_heads,
                       const std::function<void()>& launch, bool expect_ok) {
            q.memset(dv, 0, 128 * 4).wait();
            launch(); q.wait();
            std::vector<uint32_t> vis(n_heads);
            q.memcpy(vis.data(), dv, size_t(n_heads) * 4).wait();
            size_t missed = 0, doubled = 0;
            for (uint32_t h = 0; h < n_heads; ++h) {
                if (vis[h] == 0) ++missed;
                if (vis[h] > 1) ++doubled;
            }
            std::printf("  %-34s n_heads %3u  missed %3zu  double %3zu\n",
                        label, n_heads, missed, doubled);
            const bool ok = (missed == 0 && doubled == 0);
            if (expect_ok) check(ok, "the lane->head map does not cover every head once");
            else check(!ok, "NEGATIVE CONTROL DID NOT FIRE: a stride-(kSG+1) head map was "
                            "reported as complete — the census cannot see skipped heads");
        };
        run("stride kSG, n_heads 64", 64, [&]{ head_census<kSG>(q, dv, 64); }, true);
        run("stride kSG, n_heads 40", 40, [&]{ head_census<kSG>(q, dv, 40); }, true);
        run("stride kSG+1 (BROKEN)",  64, [&]{ head_census<kSG + 1>(q, dv, 64); }, false);
        sycl::free(dv, q);
    }

    // ---------------------------------------------------------------------
    // §8 the DOWNSTREAM discrete effect.  ds4_indexer_score feeds
    // ds4_indexer_topk, which turns the scores into a selection — and a
    // selection is a discrete decision that a 1-ulp score change can flip.  So
    // "the deviation is inside the bound" is not on its own enough; what has to
    // hold is that any key the two shapes disagree about is a genuine near-tie
    // at the selection boundary, not a key with a clearly different score.
    //
    // The check: let `thr` be the EXACT (fp64) score of the top_k-th ranked key.
    // Every key selected by one shape and not the other must have an exact
    // score within `slack` of thr, where slack is the sum of the two shapes'
    // derived bounds scaled by the largest absolute contribution sum.  A swap
    // outside that window is a real defect and fails.
    // ---------------------------------------------------------------------
    std::puts("\n§8 ds4_indexer_score -> ds4_indexer_topk: every selection difference\n"
              "   must be a near-tie at the boundary, not a genuine disagreement");
    for (uint32_t n_keys : {1024u, 4096u}) {
        const uint32_t T = 1, H = 64, D = 128, index_topk = 512;
        const uint32_t top_k = std::min(index_topk, n_keys);
        std::mt19937_64 rng(0x709 ^ n_keys);
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> hq(size_t(T) * H * D), hk(size_t(n_keys) * D), hw(size_t(T) * H);
        for (auto& v : hq) v = nd(rng) * 0.3f;
        for (auto& v : hk) v = nd(rng) * 0.3f;
        for (auto& v : hw) v = nd(rng) * 0.5f;
        const float sms = 1.f / std::sqrt(float(D)), ws = 1.f / std::sqrt(float(H));
        const IScoreRef ref = iscore_reference(hq, hk, hw, T, H, D, n_keys, sms, ws);

        float* dq = sycl::malloc_device<float>(hq.size(), q);
        float* dk = sycl::malloc_device<float>(hk.size(), q);
        float* dw = sycl::malloc_device<float>(hw.size(), q);
        float* dsc = sycl::malloc_device<float>(size_t(T) * n_keys, q);
        int32_t* dp = sycl::malloc_device<int32_t>(T, q);
        int32_t* dout = sycl::malloc_device<int32_t>(size_t(T) * top_k, q);
        const int32_t pos = int32_t(n_keys * 4 - 1);   // threshold admits every key
        q.memcpy(dq, hq.data(), hq.size() * 4).wait();
        q.memcpy(dk, hk.data(), hk.size() * 4).wait();
        q.memcpy(dw, hw.data(), hw.size() * 4).wait();
        q.memcpy(dp, &pos, 4).wait();

        std::vector<int32_t> sel_new(top_k), sel_old(top_k);
        ie::ds4_indexer_score(q, dq, dk, dw, dsc, T, H, D, n_keys, sms, ws).wait();
        ie::ds4_indexer_topk(q, dsc, dp, dout, T, n_keys, index_topk, 4).wait();
        q.memcpy(sel_new.data(), dout, sel_new.size() * 4).wait();
        ref_iscore_dimlane(q, dq, dk, dw, dsc, T, H, D, n_keys, sms, ws).wait();
        ie::ds4_indexer_topk(q, dsc, dp, dout, T, n_keys, index_topk, 4).wait();
        q.memcpy(sel_old.data(), dout, sel_old.size() * 4).wait();

        double max_absum = 0.0;
        for (double a : ref.absum) max_absum = std::max(max_absum, a);
        const double slack = (iscore_const_headlane(D, H) + iscore_const_dimlane(D, H))
                             * kU * max_absum;
        std::vector<double> sorted = ref.score;
        std::sort(sorted.begin(), sorted.end(), std::greater<double>());
        const double thr = sorted[top_k - 1];
        const double gap = sorted[top_k - 1] - sorted[top_k];   // boundary separation

        std::vector<char> in_new(n_keys, 0), in_old(n_keys, 0);
        for (int32_t v : sel_new) if (v >= 0) in_new[size_t(v)] = 1;
        for (int32_t v : sel_old) if (v >= 0) in_old[size_t(v)] = 1;
        size_t differ = 0, outside = 0;
        for (uint32_t e = 0; e < n_keys; ++e) {
            if (in_new[e] == in_old[e]) continue;
            ++differ;
            if (std::fabs(ref.score[e] - thr) > slack) ++outside;
        }
        std::printf("  n_keys=%-5u top_k=%-4u  selections differing %3zu  "
                    "outside the near-tie window %zu\n"
                    "      boundary gap %.3e   slack %.3e   (gap/slack %.2f)\n",
                    n_keys, top_k, differ, outside, gap, slack, gap / slack);
        check(outside == 0, "ds4_indexer_score changed a top-k selection between two keys "
                            "whose exact scores are further apart than the two shapes' "
                            "combined error bounds — that is a defect, not a tie-break");
        sycl::free(dq, q); sycl::free(dk, q); sycl::free(dw, q);
        sycl::free(dsc, q); sycl::free(dp, q); sycl::free(dout, q);
    }

    // ---------------------------------------------------------------------
    std::puts("\n§9 measured before/after (best of 3 runs of 40; ms per call)");
    std::printf("  %-34s %5s %7s %10s %10s %8s\n",
                "kernel / shape", "T", "n", "before", "after", "speedup");
    for (const Shape& sh : shapes) {
        const AttnCase c = make_case(sh.T, sh.H, sh.D, sh.n_sl, sh.n_c, sh.top_k, 707);
        AttnDev dv; dv.alloc(q, c);
        const double before = time_ms(q, 40, [&]{
            ref_attn_plain(q, dv.q, dv.kv, dv.mask, dv.sinks, dv.y,
                           c.T, c.H, c.D, c.n_kv, c.scaling); });
        const double after = time_ms(q, 40, [&]{
            ie::ds4_attention(q, dv.q, dv.kv, dv.mask, dv.sinks, dv.y,
                              c.T, c.H, c.D, c.n_kv, c.scaling); });
        std::printf("  ds4_attention    %-17s %5u %7u %10.4f %10.4f %7.2fx\n",
                    sh.name, sh.T, c.n_kv, before, after, before / after);
        dv.free(q);
    }
    {
        const uint32_t H = 4096, E = 256, K = 6;
        std::mt19937_64 rng(808);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (uint32_t T : {1u, 4u, 64u}) {
            std::vector<float> hx(size_t(T) * H), hw(size_t(E) * H), hb(E);
            for (auto& v : hx) v = nd(rng) * 0.05f;
            for (auto& v : hw) v = nd(rng) * 0.02f;
            for (auto& v : hb) v = nd(rng) * 0.01f;
            float* dx = sycl::malloc_device<float>(hx.size(), q);
            float* dw = sycl::malloc_device<float>(hw.size(), q);
            float* db = sycl::malloc_device<float>(hb.size(), q);
            float* dl = sycl::malloc_device<float>(size_t(T) * E, q);
            float* dwt = sycl::malloc_device<float>(size_t(T) * K, q);
            int32_t* di = sycl::malloc_device<int32_t>(size_t(T) * K, q);
            q.memcpy(dx, hx.data(), hx.size() * 4).wait();
            q.memcpy(dw, hw.data(), hw.size() * 4).wait();
            q.memcpy(db, hb.data(), hb.size() * 4).wait();
            const double before = time_ms(q, 40, [&]{
                ref_router_fused(q, dx, dw, db, dl, dwt, di, T, H, E, K, 2.5f); });
            const double after = time_ms(q, 40, [&]{
                ie::ds4_router_topk(q, dx, dw, db, dl, dwt, di, T, H, E, K, 2.5f); });
            std::printf("  ds4_router_topk  %-17s %5u %7u %10.4f %10.4f %7.2fx\n",
                        "hidden 4096", T, E, before, after, before / after);
            sycl::free(dx, q); sycl::free(dw, q); sycl::free(db, q);
            sycl::free(dl, q); sycl::free(dwt, q); sycl::free(di, q);
        }
    }
    {
        std::mt19937_64 rng(909);
        std::normal_distribution<float> nd(0.f, 1.f);
        const std::pair<uint32_t, uint32_t> ik[] = {{1, 1024}, {1, 4096}, {64, 1024}};
        for (const auto& pr : ik) {
            const uint32_t T = pr.first, n_keys = pr.second;
            const uint32_t top_k = std::min(512u, n_keys);
            std::vector<float> hs(size_t(T) * n_keys);
            for (auto& v : hs) v = nd(rng);
            std::vector<int32_t> hp(T, int32_t(n_keys * 4 - 1));
            float* dsc = sycl::malloc_device<float>(hs.size(), q);
            int32_t* dp = sycl::malloc_device<int32_t>(hp.size(), q);
            int32_t* dout = sycl::malloc_device<int32_t>(size_t(T) * top_k, q);
            q.memcpy(dsc, hs.data(), hs.size() * 4).wait();
            q.memcpy(dp, hp.data(), hp.size() * 4).wait();
            const double before = time_ms(q, 40, [&]{
                ref_itopk_scan(q, dsc, dp, dout, T, n_keys, 512, 4); });
            const double after = time_ms(q, 40, [&]{
                ie::ds4_indexer_topk(q, dsc, dp, dout, T, n_keys, 512, 4); });
            std::printf("  ds4_indexer_topk %-17s %5u %7u %10.4f %10.4f %7.2fx\n",
                        "index_topk 512", T, n_keys, before, after, before / after);
            sycl::free(dsc, q); sycl::free(dp, q); sycl::free(dout, q);
        }
    }
    {
        const uint32_t H = 64, D = 128;
        std::mt19937_64 rng(1010);
        std::normal_distribution<float> nd(0.f, 1.f);
        const std::pair<uint32_t, uint32_t> sc[] = {{1, 1024}, {1, 4096}, {4, 1024}, {4, 4096}};
        for (const auto& pr : sc) {
            const uint32_t T = pr.first, n_keys = pr.second;
            std::vector<float> hq(size_t(T) * H * D), hk(size_t(n_keys) * D), hw(size_t(T) * H);
            for (auto& v : hq) v = nd(rng) * 0.3f;
            for (auto& v : hk) v = nd(rng) * 0.3f;
            for (auto& v : hw) v = nd(rng) * 0.5f;
            const float sms = 1.f / std::sqrt(float(D)), ws = 1.f / std::sqrt(float(H));
            float* dq = sycl::malloc_device<float>(hq.size(), q);
            float* dk = sycl::malloc_device<float>(hk.size(), q);
            float* dw = sycl::malloc_device<float>(hw.size(), q);
            float* ds = sycl::malloc_device<float>(size_t(T) * n_keys, q);
            q.memcpy(dq, hq.data(), hq.size() * 4).wait();
            q.memcpy(dk, hk.data(), hk.size() * 4).wait();
            q.memcpy(dw, hw.data(), hw.size() * 4).wait();
            const double before = time_ms(q, 40, [&]{
                ref_iscore_dimlane(q, dq, dk, dw, ds, T, H, D, n_keys, sms, ws); });
            const double after = time_ms(q, 40, [&]{
                ie::ds4_indexer_score(q, dq, dk, dw, ds, T, H, D, n_keys, sms, ws); });
            std::printf("  ds4_indexer_score %-16s %5u %7u %10.4f %10.4f %7.2fx\n",
                        "64 heads, d=128", T, n_keys, before, after, before / after);
            sycl::free(dq, q); sycl::free(dk, q); sycl::free(dw, q); sycl::free(ds, q);
        }
    }

    if (g_fail) { std::printf("\nFAILED: %d check(s)\n", g_fail); return 1; }
    std::puts("\nOK — partition is exact, attention is inside its derived bound with every "
              "control firing, router and indexer top-k are bit-identical.");
    return 0;
}
