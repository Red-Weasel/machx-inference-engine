// src/ops/mimo26_ops.cpp — MiMo-V2.6 bring-up kernels (P2, docs/mimo26/00_PORT_PLAN.md). See mimo26_ops.hpp.
// The attention is full_attention_gptoss (src/ops/attention.cpp) with the V head dim decoupled from the K
// head dim; everything else is elementwise or one reduction per row.
#include "ie/mimo26_ops.hpp"

#include "ie/deepseek41_upload.hpp"   // ds41_e4m3
#include "ie/kernel_profiler.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ie {

namespace {
constexpr uint32_t kSg = 16;   // sub-group width the attention and router kernels reduce over

// K [T, n_kv, hd] and V [T, n_kv, vs] (first hdv of each row) into the caches at positions [pos0, pos0 + T): slot
// pos mod ring for a ring (ring > 0), pos for a linear cache of `slots` rows per kv head.
sycl::event mimo26_attn_append(sycl::queue& q, const sycl::half* K, const sycl::half* V, sycl::half* k_cache, sycl::half* v_cache,
                               uint32_t T, uint32_t pos0, uint32_t n_kv, uint32_t hd, uint32_t hdv, uint32_t slots, uint32_t ring,
                               uint32_t vs, const std::vector<sycl::event>& deps) {
    return ie::ps(q, "mimo26_attn_append", [&](sycl::handler& h) {
        h.depends_on(deps);
        const uint32_t nk = n_kv * hd, nv = n_kv * hdv, R = ring;
        h.parallel_for(sycl::range<2>(T, nk + nv), [=](sycl::id<2> id) {
            const uint32_t t = uint32_t(id[0]), c = uint32_t(id[1]);
            const uint32_t slot = R ? (pos0 + t) % R : pos0 + t;
            if (c < nk) {
                const uint32_t kvh = c / hd, d = c % hd;
                k_cache[(size_t(kvh) * slots + slot) * hd + d] = K[size_t(t) * nk + c];
            } else {
                const uint32_t cc = c - nk, kvh = cc / hdv, d = cc % hdv;
                v_cache[(size_t(kvh) * slots + slot) * vs + d] = V[(size_t(t) * n_kv + kvh) * vs + d];
            }
        });
    });
}
}  // namespace

// The XMX prefill attention's geometry (mimo26_attention_prefill_xmx). Named namespace: the kernel-name collision trap.
namespace mimo26px {
namespace mat  = sycl::ext::oneapi::experimental::matrix;
namespace imat = sycl::ext::intel::experimental::matrix;
constexpr uint32_t SG = 16, NSG = 16, WG = SG * NSG;   // 256 lanes
constexpr uint32_t HT = 64;                             // rows per work-group: HT / gqa tokens x the gqa heads of one kv head
constexpr uint32_t KB = 64;                             // keys per block
constexpr uint32_t DK = 192, DV = 128;                  // the K / V head dims it is built for
constexpr uint32_t TM = 8, TN = 16, TK = 16;
constexpr uint32_t KPL = KB * HT / WG;                  // keys per lane in the softmax (16: 4 lanes per row)
constexpr float kNegInf = -std::numeric_limits<float>::infinity();
}  // namespace mimo26px

sycl::event mimo26_rms_norm(sycl::queue& q, const float* x, const float* w, sycl::half* y16, float* y32,
                            uint32_t T, uint32_t H, float eps, const std::vector<sycl::event>& deps) {
    return ie::ps(q, "mimo26_rms_norm", [&](sycl::handler& h) {
        h.depends_on(deps);
        constexpr uint32_t WG = 256;
        h.parallel_for(sycl::nd_range<1>(size_t(T) * WG, WG), [=](sycl::nd_item<1> it) {
            const uint32_t t = uint32_t(it.get_group(0)), i = uint32_t(it.get_local_id(0));
            const float* xr = x + size_t(t) * H;
            float ss = 0.f;
            for (uint32_t k = i; k < H; k += WG) ss += xr[k] * xr[k];
            ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
            const float inv = sycl::rsqrt(ss / float(H) + eps);
            for (uint32_t k = i; k < H; k += WG) {
                const float v = xr[k] * inv * w[k];
                if (y16) y16[size_t(t) * H + k] = sycl::half(v);
                if (y32) y32[size_t(t) * H + k] = v;
            }
        });
    });
}

sycl::event mimo26_split_qkv(sycl::queue& q, const float* qkv, sycl::half* Q, sycl::half* K, sycl::half* V,
                             uint32_t T, uint32_t n_q, uint32_t n_kv, uint32_t hd, uint32_t hdv,
                             const std::vector<sycl::event>& deps, uint32_t v_stride) {
    if (!v_stride) v_stride = hdv;
    return ie::ps(q, "mimo26_split_qkv", [&](sycl::handler& h) {
        h.depends_on(deps);
        const uint32_t nq = n_q * hd, nk = n_kv * hd, nv = n_kv * hdv, N = nq + nk + nv, vs = v_stride;
        h.parallel_for(sycl::range<2>(T, N), [=](sycl::id<2> id) {
            const uint32_t t = uint32_t(id[0]), c = uint32_t(id[1]);
            const sycl::half v = sycl::half(qkv[size_t(t) * N + c]);
            if (c < nq)           Q[size_t(t) * nq + c] = v;
            else if (c < nq + nk) K[size_t(t) * nk + (c - nq)] = v;
            else { const uint32_t cc = c - nq - nk; V[(size_t(t) * n_kv + cc / hdv) * vs + cc % hdv] = v; }
        });
    });
}

sycl::event mimo26_compact_heads(sycl::queue& q, const sycl::half* x, sycl::half* y, uint32_t T, uint32_t n_q,
                                 uint32_t x_stride, uint32_t hdv, const std::vector<sycl::event>& deps) {
    return ie::ps(q, "mimo26_compact_heads", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<2>(size_t(T) * n_q, hdv), [=](sycl::id<2> id) {
            y[id[0] * hdv + id[1]] = x[id[0] * x_stride + id[1]];
        });
    });
}

sycl::event mimo26_attention(sycl::queue& q, const sycl::half* Q, const sycl::half* K, const sycl::half* V,
                             sycl::half* k_cache, sycl::half* v_cache, sycl::half* y,
                             uint32_t T, uint32_t pos0, uint32_t n_q, uint32_t n_kv, uint32_t hd, uint32_t hdv,
                             uint32_t max_ctx, uint32_t window, const float* sinks,
                             const std::vector<sycl::event>& deps, uint32_t ring, uint32_t v_stride) {
    if (!v_stride) v_stride = hdv;
    const uint32_t slots = ring ? ring : max_ctx;   // rows per kv head in the caches
    // 1. append K and V rows to the caches at [pos0, pos0 + T) (slot = pos mod ring for a ring)
    auto append = mimo26_attn_append(q, K, V, k_cache, v_cache, T, pos0, n_kv, hd, hdv, slots, ring, v_stride, deps);
    // 2. one sub-group per (query row, head): online softmax over the key range, sink folded into the denominator
    return ie::ps(q, "mimo26_attn_compute", [&](sycl::handler& h) {
        h.depends_on(append);
        const uint32_t dk = hd / kSg, dv = hdv / kSg;          // dims per lane
        const float scale = 1.0f / sycl::sqrt(float(hd));
        const uint32_t gqa = n_q / n_kv, R = ring, vs = v_stride;
        h.parallel_for(sycl::nd_range<2>({T, size_t(n_q) * kSg}, {1, kSg}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(kSg)]] {
            const uint32_t t = uint32_t(it.get_group(0)), hq = uint32_t(it.get_group(1)), lane = uint32_t(it.get_local_id(1));
            const uint32_t kvh = hq / gqa, p = pos0 + t, n_keys = p + 1;
            const uint32_t k0 = (window && n_keys > window) ? n_keys - window : 0u;
            auto sg = it.get_sub_group();
            float qv[16];
            for (uint32_t d = 0; d < dk; ++d) qv[d] = float(Q[(size_t(t) * n_q + hq) * hd + lane * dk + d]);
            float m = -std::numeric_limits<float>::infinity(), l = 0.f, acc[16];
            for (uint32_t d = 0; d < dv; ++d) acc[d] = 0.f;
            for (uint32_t i = k0; i < n_keys; ++i) {
                const uint32_t si = R ? i % R : i;
                const sycl::half* kr = k_cache + (size_t(kvh) * slots + si) * hd + lane * dk;
                float part = 0.f;
                for (uint32_t d = 0; d < dk; ++d) part += qv[d] * float(kr[d]);
                const float s = sycl::reduce_over_group(sg, part, sycl::plus<float>()) * scale;
                const float m_new = sycl::fmax(m, s), a = sycl::native::exp(m - m_new), e = sycl::native::exp(s - m_new);
                const sycl::half* vr = v_cache + (size_t(kvh) * slots + si) * vs + lane * dv;
                for (uint32_t d = 0; d < dv; ++d) acc[d] = acc[d] * a + e * float(vr[d]);
                l = l * a + e; m = m_new;
            }
            float corr = 1.f, ll = l;
            if (sinks) {
                const float s = sinks[hq], m_f = sycl::fmax(m, s);
                corr = (m == -std::numeric_limits<float>::infinity()) ? 0.f : sycl::native::exp(m - m_f);
                ll = l * corr + sycl::native::exp(s - m_f);
            }
            const float inv = ll > 0.f ? 1.f / ll : 0.f;
            for (uint32_t d = 0; d < dv; ++d) y[(size_t(t) * n_q + hq) * hdv + lane * dv + d] = sycl::half(acc[d] * corr * inv);
        });
    });
}

sycl::event mimo26_attention_decode(sycl::queue& q, const sycl::half* Q, const sycl::half* K, const sycl::half* V,
                                    sycl::half* k_cache, sycl::half* v_cache, sycl::half* y, float* partials,
                                    uint32_t T, uint32_t pos0, uint32_t n_q, uint32_t n_kv, uint32_t hd, uint32_t hdv,
                                    uint32_t max_ctx, uint32_t window, const float* sinks,
                                    const std::vector<sycl::event>& deps, uint32_t ring, uint32_t v_stride) {
    // hd / hdv are compile-time in the kernel (runtime-bounded per-lane arrays spill to private memory: the first build
    // of this kernel measured 0.28 ms per call at 16k): MiMo's 192 / 128 only
    if (hd != 192 || hdv != 128 || !n_kv || n_q % n_kv || n_q / n_kv > 16)
        throw std::invalid_argument("mimo26_attention_decode: needs hd 192, hdv 128, n_q / n_kv <= 16");
    if (!v_stride) v_stride = hdv;
    const uint32_t slots = ring ? ring : max_ctx;
    auto append = mimo26_attn_append(q, K, V, k_cache, v_cache, T, pos0, n_kv, hd, hdv, slots, ring, v_stride, deps);
    // the widest key range any row sees decides the split count (every row uses the same split geometry over its own range)
    const uint32_t last_keys = pos0 + T;
    const uint32_t span = window && last_keys > window ? window : last_keys;
    constexpr uint32_t kMinKeys = 64;
    const uint32_t want = std::max(1u, 256u / std::max(1u, n_kv * T));
    const uint32_t NS = std::min({mimo26_decode_max_splits(), want, (span + kMinKeys - 1) / kMinKeys});
    const uint32_t gqa = n_q / n_kv, PW = hdv + 2;
    constexpr uint32_t Bc = 32;
    auto part = ie::ps(q, "mimo26_attn_decode_part", [&](sycl::handler& h) {
        h.depends_on(append);
        constexpr uint32_t HD = 192, HDV = 128, dk = HD / kSg, dv = HDV / kSg;
        sycl::local_accessor<sycl::half, 1> Ks(sycl::range<1>(Bc * HD), h), Vs(sycl::range<1>(Bc * HDV), h);
        const uint32_t vs = v_stride, R = ring;
        const float scale = 1.0f / sycl::sqrt(float(HD));
        h.parallel_for(sycl::nd_range<3>({T, size_t(n_kv) * NS, size_t(gqa) * kSg}, {1, 1, size_t(gqa) * kSg}),
                       [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(kSg)]] {
            const uint32_t t = uint32_t(it.get_group(0)), g1 = uint32_t(it.get_group(1));
            const uint32_t kvh = g1 / NS, s = g1 % NS;
            const uint32_t lid = uint32_t(it.get_local_id(2)), sgi = lid / kSg, lane = lid % kSg, hq = kvh * gqa + sgi;
            const uint32_t p = pos0 + t, n_keys = p + 1, k0 = (window && n_keys > window) ? n_keys - window : 0u;
            const uint32_t range = n_keys - k0, per = (range + NS - 1) / NS;
            const uint32_t a = k0 + std::min(range, s * per), b = k0 + std::min(range, (s + 1) * per);
            auto sg = it.get_sub_group();
            float qv[dk];
            #pragma unroll
            for (uint32_t d = 0; d < dk; ++d) qv[d] = float(Q[(size_t(t) * n_q + hq) * HD + lane * dk + d]);
            float m = -std::numeric_limits<float>::infinity(), l = 0.f, acc[dv];
            #pragma unroll
            for (uint32_t d = 0; d < dv; ++d) acc[d] = 0.f;
            const uint32_t nthr = gqa * kSg;
            for (uint32_t base = a; base < b; base += Bc) {
                const uint32_t nb = std::min(Bc, b - base);
                sycl::group_barrier(it.get_group());
                // the tile's K and V rows into SLM, 16 bytes per load (the rows are 384 / 256 bytes, 16-byte aligned)
                for (uint32_t i = lid; i < nb * (HD / 8); i += nthr) {
                    const uint32_t r = i / (HD / 8), c8 = i % (HD / 8), key = base + r, si = R ? key % R : key;
                    sycl::vec<sycl::half, 8> v;
                    v.load(0, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(k_cache + (size_t(kvh) * slots + si) * HD + c8 * 8));
                    v.store(0, Ks.template get_multi_ptr<sycl::access::decorated::no>() + r * HD + c8 * 8);
                }
                for (uint32_t i = lid; i < nb * (HDV / 8); i += nthr) {
                    const uint32_t r = i / (HDV / 8), c8 = i % (HDV / 8), key = base + r, si = R ? key % R : key;
                    sycl::vec<sycl::half, 8> v;
                    v.load(0, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(v_cache + (size_t(kvh) * slots + si) * vs + c8 * 8));
                    v.store(0, Vs.template get_multi_ptr<sycl::access::decorated::no>() + r * HDV + c8 * 8);
                }
                sycl::group_barrier(it.get_group());
                for (uint32_t j = 0; j < nb; ++j) {
                    sycl::vec<sycl::half, 4> k4[dk / 4];   // this lane's 12 K dims: 24 bytes, 8-byte aligned
                    #pragma unroll
                    for (uint32_t u = 0; u < dk / 4; ++u) k4[u].load(u, Ks.template get_multi_ptr<sycl::access::decorated::no>() + j * HD + lane * dk);
                    float pdot = 0.f;
                    #pragma unroll
                    for (uint32_t d = 0; d < dk; ++d) pdot += qv[d] * float(k4[d / 4][d % 4]);
                    const float sc = sycl::reduce_over_group(sg, pdot, sycl::plus<float>()) * scale;
                    const float m_new = sycl::fmax(m, sc), al = sycl::native::exp(m - m_new), e = sycl::native::exp(sc - m_new);
                    sycl::vec<sycl::half, dv> v8;          // this lane's 8 V dims: 16 bytes
                    v8.load(0, Vs.template get_multi_ptr<sycl::access::decorated::no>() + j * HDV + lane * dv);
                    #pragma unroll
                    for (uint32_t d = 0; d < dv; ++d) acc[d] = acc[d] * al + e * float(v8[d]);
                    l = l * al + e; m = m_new;
                }
            }
            float* out = partials + ((size_t(t) * n_q + hq) * NS + s) * PW;
            #pragma unroll
            for (uint32_t d = 0; d < dv; ++d) out[lane * dv + d] = acc[d];
            if (lane == 0) { out[HDV] = m; out[HDV + 1] = l; }
        });
    });
    // 2. merge the splits per (row, head), sink folded into the denominator
    return ie::ps(q, "mimo26_attn_decode_merge", [&](sycl::handler& h) {
        h.depends_on(part);
        h.parallel_for(sycl::range<3>(T, n_q, hdv), [=](sycl::id<3> id) {
            const uint32_t t = uint32_t(id[0]), hq = uint32_t(id[1]), d = uint32_t(id[2]);
            const float* base = partials + (size_t(t) * n_q + hq) * NS * PW;
            float M = -std::numeric_limits<float>::infinity();
            for (uint32_t s = 0; s < NS; ++s) M = sycl::fmax(M, base[s * PW + hdv]);
            if (sinks) M = sycl::fmax(M, sinks[hq]);
            float L = sinks ? sycl::native::exp(sinks[hq] - M) : 0.f, O = 0.f;
            for (uint32_t s = 0; s < NS; ++s) {
                const float ms = base[s * PW + hdv];
                if (ms == -std::numeric_limits<float>::infinity()) continue;   // an empty split
                const float w = sycl::native::exp(ms - M);
                L += base[s * PW + hdv + 1] * w;
                O += base[s * PW + d] * w;
            }
            y[(size_t(t) * n_q + hq) * hdv + d] = sycl::half(L > 0.f ? O / L : 0.f);
        });
    });
}

sycl::event mimo26_attention_prefill_xmx(sycl::queue& q, const sycl::half* Q, const sycl::half* K, const sycl::half* V,
                                         sycl::half* k_cache, sycl::half* v_cache, sycl::half* y,
                                         uint32_t T, uint32_t pos0, uint32_t n_q, uint32_t n_kv, uint32_t max_ctx,
                                         uint32_t window, const float* sinks, const std::vector<sycl::event>& deps,
                                         uint32_t v_stride, uint32_t ring) {
    using namespace mimo26px;
    if (!v_stride) v_stride = DV;
    const uint32_t gqa = n_q / n_kv;
    if (!gqa || n_q % n_kv || HT % gqa) throw std::invalid_argument("mimo26_attention_prefill_xmx: n_q / n_kv must divide 64");
    if (ring % KB) throw std::invalid_argument("mimo26_attention_prefill_xmx: the ring must be a multiple of 64 slots");
    const uint32_t slots = ring ? ring : max_ctx;
    auto append = mimo26_attn_append(q, K, V, k_cache, v_cache, T, pos0, n_kv, DK, DV, slots, ring, v_stride, deps);
    const uint32_t n_tt = HT / gqa, n_tb = (T + n_tt - 1) / n_tt;   // tokens per work-group, token blocks
    // The Q^T and V tiles load straight from global memory: joint_matrix loads from SLM measured 4-6x slower on this
    // stack (ie-mimo26-attn-bench --ablate, 16k keys: 336 -> 50 ms per call). V for every whole key block (the causal
    // edge's block is staged zero-filled: a stale or NaN row times P = 0 must not reach O); Q when a 16-row column tile is
    // one token's 16 heads (gqa 16, one stride) -- otherwise it is staged transposed in SLM.
    auto submit = [&]<bool kQG>() {
    return ie::ps(q, "mimo26_attn_prefill_xmx", [&](sycl::handler& h) {
        h.depends_on(append);
        sycl::local_accessor<sycl::half, 1> Qt(sycl::range<1>(kQG ? 1 : size_t(DK) * HT), h);   // Q^T [DK][HT] (staged Q)
        sycl::local_accessor<float, 1>      St(sycl::range<1>(size_t(KB) * HT), h);   // S^T [KB][HT]
        sycl::local_accessor<sycl::half, 1> P (sycl::range<1>(size_t(HT) * KB), h);   // P [HT][KB]
        sycl::local_accessor<sycl::half, 1> Vs(sycl::range<1>(size_t(KB) * DV), h);   // the block's V [KB][DV]
        sycl::local_accessor<float, 1>      red(sycl::range<1>(WG), h);
        sycl::local_accessor<float, 1>      mrun(sycl::range<1>(HT), h), lrun(sycl::range<1>(HT), h);
        sycl::local_accessor<float, 1>      alpha(sycl::range<1>(HT), h), mnew(sycl::range<1>(HT), h);
        const float scale = 1.0f / sycl::sqrt(float(DK));
        const uint32_t vs = v_stride, R = ring;
        h.parallel_for(sycl::nd_range<2>({size_t(n_tb), size_t(n_kv) * WG}, {1, WG}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint32_t tb  = n_tb - 1 - uint32_t(it.get_group(0));   // the longest key ranges start first
            const uint32_t kvh = uint32_t(it.get_group(1));
            const uint32_t lid = uint32_t(it.get_local_id(1));
            auto sg = it.get_sub_group();
            auto grp = it.get_group();
            const uint32_t s  = uint32_t(sg.get_group_linear_id());
            const uint32_t t0 = tb * n_tt, t_last = sycl::min(t0 + n_tt, T) - 1;
            // row = tt * gqa + g: token t0 + tt, q head kvh * gqa + g
            if constexpr (!kQG) for (uint32_t i = lid; i < HT * DK; i += WG) {
                const uint32_t row = i / DK, k = i % DK, t = t0 + row / gqa;
                Qt[k * HT + row] = t < T ? Q[(size_t(t) * n_q + kvh * gqa + row % gqa) * DK + k] : sycl::half(0.f);
            }
            if (lid < HT) { mrun[lid] = kNegInf; lrun[lid] = 0.f; }
            // O [HT][DV]: sub-group s owns rows [32 (s >> 3), +32) (4 row tiles) x columns [16 (s & 7), +16)
            const uint32_t orow0 = (s >> 3) * 32, ocol = (s & 7) * TN;
            mat::joint_matrix<sycl::sub_group, float, mat::use::accumulator, TM, TN> acc[4];
            #pragma unroll
            for (int mr = 0; mr < 4; ++mr) mat::joint_matrix_fill(sg, acc[mr], 0.0f);
            // this lane's softmax row and keys: row hh, keys [KPL jj, KPL jj + KPL) of each block
            const uint32_t hh = lid & (HT - 1), jj = lid / HT;
            const uint32_t t_row = t0 + hh / gqa, p_row = pos0 + t_row;
            const bool row_live = t_row < T;
            const uint32_t lo_row = (window && p_row + 1 > window) ? p_row + 1 - window : 0u;   // first allowed key
            const uint32_t kend = pos0 + t_last + 1;
            const uint32_t p_first = pos0 + t0;
            const uint32_t kbeg = ((window && p_first + 1 > window) ? p_first + 1 - window : 0u) / KB * KB;
            sycl::group_barrier(grp);

            for (uint32_t n0 = kbeg; n0 < kend; n0 += KB) {
                // a whole block's V tiles come from the cache (R % 64 == 0: a block never wraps the ring); the causal edge's
                // block is staged, rows past kend zero
                const bool whole = n0 + KB <= kend;
                const uint32_t slot0 = R ? n0 % R : n0;
                if (!whole) for (uint32_t i = lid; i < KB * (DV / 8); i += WG) {
                    const uint32_t r = i / (DV / 8), c8 = i % (DV / 8), key = n0 + r;
                    sycl::vec<sycl::half, 8> v(sycl::half(0.f));
                    if (key < kend) v.load(0, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(v_cache + (size_t(kvh) * slots + slot0 + r) * vs + c8 * 8));
                    v.store(0, Vs.template get_multi_ptr<sycl::access::decorated::no>() + r * DV + c8 * 8);
                }
                // S^T [KB][HT] = Kblk [KB][DK] . Q^T [DK][HT]: 8 key-row tiles x 4 row-column tiles; sub-group s takes key
                // tile s >> 1 and row-column tiles 2 (s & 1) + {0, 1}. K straight from the cache (rows past kend are masked;
                // in a ring an 8-row tile starts at a multiple of 8 and R is one, so it never wraps)
                {
                    const uint32_t rt = s >> 1, c0 = (s & 1) * 2;
                    mat::joint_matrix<sycl::sub_group, float, mat::use::accumulator, TM, TN> sacc[2];
                    mat::joint_matrix_fill(sg, sacc[0], 0.0f);
                    mat::joint_matrix_fill(sg, sacc[1], 0.0f);
                    if (n0 + rt * TM < kend) {
                        const uint32_t key = n0 + rt * TM;
                        const sycl::half* kbase = k_cache + (size_t(kvh) * slots + (R ? key % R : key)) * DK;
                        #pragma unroll 4
                        for (uint32_t kk = 0; kk < DK; kk += TK) {
                            mat::joint_matrix<sycl::sub_group, sycl::half, mat::use::a, TM, TK, mat::layout::row_major> a_tile;
                            mat::joint_matrix_load(sg, a_tile,
                                sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(kbase + kk), DK);
                            #pragma unroll
                            for (uint32_t j = 0; j < 2; ++j) {
                                if constexpr (kQG) {   // column tile c0 + j = token t0 + c0 + j's 16 heads, col-major from Q
                                    if (t0 + c0 + j < T) {
                                        mat::joint_matrix<sycl::sub_group, sycl::half, mat::use::b, TK, TN, mat::layout::col_major> bq;
                                        mat::joint_matrix_load(sg, bq, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                                            Q + (size_t(t0 + c0 + j) * n_q + kvh * gqa) * DK + kk), DK);
                                        mat::joint_matrix_mad(sg, sacc[j], a_tile, bq, sacc[j]);
                                    }
                                } else {
                                    mat::joint_matrix<sycl::sub_group, sycl::half, mat::use::b, TK, TN, mat::layout::row_major> b_tile;
                                    mat::joint_matrix_load(sg, b_tile, Qt.template get_multi_ptr<sycl::access::decorated::no>() + kk * HT + (c0 + j) * TN, HT);
                                    mat::joint_matrix_mad(sg, sacc[j], a_tile, b_tile, sacc[j]);
                                }
                            }
                        }
                    }
                    #pragma unroll
                    for (uint32_t j = 0; j < 2; ++j)
                        imat::joint_matrix_apply(sg, sacc[j], [&](float& v, size_t r, size_t c) { St[(rt * TM + r) * HT + (c0 + j) * TN + c] = v; });
                }
                sycl::group_barrier(grp);
                // online softmax: 4 lanes per row, KPL keys each
                float sc[KPL]; float bm = kNegInf;
                #pragma unroll
                for (uint32_t r = 0; r < KPL; ++r) {
                    const uint32_t kl = jj * KPL + r, key = n0 + kl;
                    const bool ok = row_live && key <= p_row && key >= lo_row;
                    sc[r] = ok ? St[kl * HT + hh] * scale : kNegInf;
                    bm = sycl::fmax(bm, sc[r]);
                }
                red[lid] = bm;
                sycl::group_barrier(grp);
                if (lid < HT) {
                    float m4 = kNegInf;
                    #pragma unroll
                    for (uint32_t j = 0; j < WG / HT; ++j) m4 = sycl::fmax(m4, red[j * HT + lid]);
                    const float m_old = mrun[lid], m_n = sycl::fmax(m_old, m4);
                    mnew[lid] = m_n;
                    alpha[lid] = (m_old == kNegInf) ? 0.f : sycl::native::exp(m_old - m_n);
                }
                sycl::group_barrier(grp);
                {
                    const float m_n = mnew[hh]; float ps = 0.f;
                    #pragma unroll
                    for (uint32_t r = 0; r < KPL; ++r) {
                        const float e = sc[r] > kNegInf ? sycl::native::exp(sc[r] - m_n) : 0.f;
                        P[hh * KB + jj * KPL + r] = sycl::half(e);
                        ps += e;
                    }
                    red[lid] = ps;
                }
                sycl::group_barrier(grp);
                if (lid < HT) {
                    float l4 = 0.f;
                    #pragma unroll
                    for (uint32_t j = 0; j < WG / HT; ++j) l4 += red[j * HT + lid];
                    lrun[lid] = lrun[lid] * alpha[lid] + l4;
                    mrun[lid] = mnew[lid];
                }
                // O = O * alpha + P [HT][KB] . Vblk [KB][DV]
                #pragma unroll
                for (int mr = 0; mr < 4; ++mr)
                    imat::joint_matrix_apply(sg, acc[mr], [&](float& v, size_t r, size_t) { v *= alpha[orow0 + mr * TM + r]; });
                #pragma unroll
                for (uint32_t kk = 0; kk < KB; kk += TK) {
                    mat::joint_matrix<sycl::sub_group, sycl::half, mat::use::b, TK, TN, mat::layout::row_major> b_tile;
                    if (whole) mat::joint_matrix_load(sg, b_tile, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                                   v_cache + (size_t(kvh) * slots + slot0 + kk) * vs + ocol), vs);
                    else mat::joint_matrix_load(sg, b_tile, Vs.template get_multi_ptr<sycl::access::decorated::no>() + kk * DV + ocol, DV);
                    #pragma unroll
                    for (int mr = 0; mr < 4; ++mr) {
                        mat::joint_matrix<sycl::sub_group, sycl::half, mat::use::a, TM, TK, mat::layout::row_major> a_tile;
                        mat::joint_matrix_load(sg, a_tile, P.template get_multi_ptr<sycl::access::decorated::no>() + (orow0 + mr * TM) * KB + kk, KB);
                        mat::joint_matrix_mad(sg, acc[mr], a_tile, b_tile, acc[mr]);
                    }
                }
                sycl::group_barrier(grp);
            }
            // the sink (a virtual key of logit sinks[head] and no value) and 1/l
            if (lid < HT) {
                const float m = mrun[lid], l = lrun[lid];
                float corr = 1.f, ll = l;
                if (sinks) {
                    const float sv = sinks[kvh * gqa + lid % gqa], m_f = sycl::fmax(m, sv);
                    corr = (m == kNegInf) ? 0.f : sycl::native::exp(m - m_f);
                    ll = l * corr + sycl::native::exp(sv - m_f);
                }
                alpha[lid] = ll > 0.f ? corr / ll : 0.f;
            }
            sycl::group_barrier(grp);
            #pragma unroll
            for (int mr = 0; mr < 4; ++mr)
                imat::joint_matrix_apply(sg, acc[mr], [&](float& v, size_t r, size_t c) {
                    const uint32_t row = orow0 + mr * TM + uint32_t(r), t = t0 + row / gqa;
                    if (t < T) y[(size_t(t) * n_q + kvh * gqa + row % gqa) * DV + ocol + c] = sycl::half(v * alpha[row]);
                });
        });
    });
    };
    return gqa == 16 ? submit.template operator()<true>() : submit.template operator()<false>();
}

// V4.1's FP8 rows GEMV (gemv_fp8_rows, src/ops/gemv_fp8.cpp) with MiMo's scales: one F32 per row and 128 columns
// instead of one E8M0 per 32 x 32 block. A sub-group per output row, lanes striding the 32-column blocks, E4M3 decoded
// through a 256-entry fp16 table (every E4M3 value is exact in fp16), the M activation rows accumulated together.
template <int M, int NC>
static sycl::event mimo26_gemv_fp8_t(sycl::queue& q, const sycl::half* x, uint32_t x_stride, const uint8_t* w, const float* srow,
                                     float* y, uint32_t K, uint32_t N, const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE = 16, WG_ITEMS = NC * SG_SIZE;
    const uint32_t blocks = K / 32, SK = K / 128;
    const uint32_t n_wgs = (N + NC - 1) / NC;
    return ie::ps(q, "mimo26_gemv_fp8", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<sycl::half, 1> lut16(sycl::range<1>(256u), h);
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid = uint32_t(it.get_local_id(0)), wgid = uint32_t(it.get_group(0));
            const uint32_t sg_id = lid / SG_SIZE, lane = lid % SG_SIZE, n = wgid * NC + sg_id;
            for (uint32_t i = lid; i < 256u; i += WG_ITEMS) lut16[i] = sycl::half(ds41_e4m3(uint8_t(i)));
            sycl::group_barrier(it.get_group());
            if (n >= N) return;
            const uint8_t* wrow = w + uint64_t(n) * K;
            const float* sr = srow + uint64_t(n) * SK;
            float acc[M];
            #pragma unroll
            for (int m = 0; m < M; ++m) acc[m] = 0.f;
            for (uint32_t b = lane; b < blocks; b += SG_SIZE) {
                const auto* w128 = reinterpret_cast<const sycl::vec<uint32_t, 4>*>(wrow + uint64_t(b) * 32);
                const sycl::vec<uint32_t, 4> wa = w128[0], wb = w128[1];
                sycl::vec<sycl::half, 8> xq[M][4];
                #pragma unroll
                for (int m = 0; m < M; ++m) {
                    const auto* xv = reinterpret_cast<const sycl::vec<sycl::half, 8>*>(x + uint64_t(m) * x_stride + uint64_t(b) * 32);
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) xq[m][j] = xv[j];
                }
                float bacc[M];
                #pragma unroll
                for (int m = 0; m < M; ++m) bacc[m] = 0.f;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const uint32_t v = wa[j], u = wb[j];
                    const float d0 = float(lut16[v & 0xFFu]), d1 = float(lut16[(v >> 8) & 0xFFu]), d2 = float(lut16[(v >> 16) & 0xFFu]), d3 = float(lut16[v >> 24]);
                    const float e0 = float(lut16[u & 0xFFu]), e1 = float(lut16[(u >> 8) & 0xFFu]), e2 = float(lut16[(u >> 16) & 0xFFu]), e3 = float(lut16[u >> 24]);
                    #pragma unroll
                    for (int m = 0; m < M; ++m) {
                        bacc[m] += d0 * float(xq[m][(4 * j + 0) >> 3][(4 * j + 0) & 7]);
                        bacc[m] += d1 * float(xq[m][(4 * j + 1) >> 3][(4 * j + 1) & 7]);
                        bacc[m] += d2 * float(xq[m][(4 * j + 2) >> 3][(4 * j + 2) & 7]);
                        bacc[m] += d3 * float(xq[m][(4 * j + 3) >> 3][(4 * j + 3) & 7]);
                        bacc[m] += e0 * float(xq[m][(16 + 4 * j + 0) >> 3][(16 + 4 * j + 0) & 7]);
                        bacc[m] += e1 * float(xq[m][(16 + 4 * j + 1) >> 3][(16 + 4 * j + 1) & 7]);
                        bacc[m] += e2 * float(xq[m][(16 + 4 * j + 2) >> 3][(16 + 4 * j + 2) & 7]);
                        bacc[m] += e3 * float(xq[m][(16 + 4 * j + 3) >> 3][(16 + 4 * j + 3) & 7]);
                    }
                }
                const float sc = sr[b >> 2];
                #pragma unroll
                for (int m = 0; m < M; ++m) acc[m] += sc * bacc[m];
            }
            #pragma unroll
            for (int m = 0; m < M; ++m) {
                const float r = sycl::reduce_over_group(it.get_sub_group(), acc[m], sycl::plus<float>());
                if (lane == 0) y[uint64_t(m) * N + n] = r;
            }
        });
    });
}

template <int M>
static sycl::event mimo26_gemv_fp8_m(sycl::queue& q, const sycl::half* x, uint32_t x_stride, const uint8_t* w, const float* srow,
                                     float* y, uint32_t K, uint32_t N, const std::vector<sycl::event>& deps) {
    if (N <= 256u) return mimo26_gemv_fp8_t<M, 4>(q, x, x_stride, w, srow, y, K, N, deps);
    if (N % 32u == 0) return mimo26_gemv_fp8_t<M, 16>(q, x, x_stride, w, srow, y, K, N, deps);
    return mimo26_gemv_fp8_t<M, 8>(q, x, x_stride, w, srow, y, K, N, deps);
}

sycl::event mimo26_gemv_fp8(sycl::queue& q, const sycl::half* x, uint32_t x_stride, const uint8_t* w8, const float* srow,
                            float* y, uint32_t M, uint32_t K, uint32_t N, const std::vector<sycl::event>& deps) {
    if (K % 128) throw std::invalid_argument("mimo26_gemv_fp8: K must be a multiple of 128");
    switch (M) {
        case 1: return mimo26_gemv_fp8_m<1>(q, x, x_stride, w8, srow, y, K, N, deps);
        case 2: return mimo26_gemv_fp8_m<2>(q, x, x_stride, w8, srow, y, K, N, deps);
        case 3: return mimo26_gemv_fp8_m<3>(q, x, x_stride, w8, srow, y, K, N, deps);
        case 4: return mimo26_gemv_fp8_m<4>(q, x, x_stride, w8, srow, y, K, N, deps);
        case 5: return mimo26_gemv_fp8_m<5>(q, x, x_stride, w8, srow, y, K, N, deps);
        case 6: return mimo26_gemv_fp8_m<6>(q, x, x_stride, w8, srow, y, K, N, deps);
        case 7: return mimo26_gemv_fp8_m<7>(q, x, x_stride, w8, srow, y, K, N, deps);
        case 8: return mimo26_gemv_fp8_m<8>(q, x, x_stride, w8, srow, y, K, N, deps);
        default: throw std::invalid_argument("mimo26_gemv_fp8: M must be 1..8");
    }
}

sycl::event mimo26_fp8_to_f16(sycl::queue& q, const uint8_t* w8, const float* srow, sycl::half* out, uint32_t N, uint32_t K,
                              const std::vector<sycl::event>& deps) {
    if (K % 128) throw std::invalid_argument("mimo26_fp8_to_f16: K must be a multiple of 128");
    const uint32_t SK = K / 128;
    return ie::ps(q, "mimo26_fp8_to_f16", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<2>(N, K / 8), [=](sycl::id<2> id) {
            const uint32_t n = uint32_t(id[0]), k8 = uint32_t(id[1]) * 8;
            const float sc = srow[uint64_t(n) * SK + k8 / 128];
            const sycl::vec<uint32_t, 2> v = *reinterpret_cast<const sycl::vec<uint32_t, 2>*>(w8 + uint64_t(n) * K + k8);
            sycl::vec<sycl::half, 8> o;
            #pragma unroll
            for (int j = 0; j < 8; ++j) o[j] = sycl::half(ds41_e4m3(uint8_t(v[j >> 2] >> (8 * (j & 3)))) * sc);
            *reinterpret_cast<sycl::vec<sycl::half, 8>*>(out + uint64_t(n) * K + k8) = o;
        });
    });
}

sycl::event mimo26_router_logits(sycl::queue& q, const float* x, const float* w, float* logits,
                                 uint32_t T, uint32_t H, uint32_t E, const std::vector<sycl::event>& deps) {
    return ie::ps(q, "mimo26_router_logits", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<2>({T, size_t(E) * kSg}, {1, kSg}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(kSg)]] {
            const uint32_t t = uint32_t(it.get_group(0)), e = uint32_t(it.get_group(1)), lane = uint32_t(it.get_local_id(1));
            const float* xr = x + size_t(t) * H;
            const float* wr = w + size_t(e) * H;
            float part = 0.f;
            for (uint32_t k = lane; k < H; k += kSg) part += xr[k] * wr[k];
            const float s = sycl::reduce_over_group(it.get_sub_group(), part, sycl::plus<float>());
            if (lane == 0) logits[size_t(t) * E + e] = s;
        });
    });
}

sycl::event mimo26_axpy(sycl::queue& q, const float* x, float alpha, float* y, uint64_t n,
                        const std::vector<sycl::event>& deps) {
    return ie::ps(q, "mimo26_axpy", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { y[i] += alpha * x[i]; });
    });
}

sycl::event mimo26_swiglu_f32(sycl::queue& q, const float* gate, const float* up, sycl::half* y, uint64_t n,
                              const std::vector<sycl::event>& deps) {
    return ie::ps(q, "mimo26_swiglu_f32", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
            const float g = gate[i];
            y[i] = sycl::half(g / (1.f + sycl::native::exp(-g)) * up[i]);
        });
    });
}

}  // namespace ie
