// src/ops/hy4flash_ix_gpu.cpp -- the Hy4-flash compressed indexer of a decode row on the card: the pooled key of a group
// whose last token just arrived, and the row's recall (the best group_topk - 1 earlier groups by score, then the keys
// they cover and the row's own group up to itself) written straight into the attention's key list. The host used to do
// both: three host round trips in each of the 12 indexed layers of every token, and a selection over context / ratio
// scores (2026-11-19: +3.6 ms a token at 25,000 tokens of context).
//
// The selection is EXACT with the host's rule (Hy4fIndexer::select_rows: score descending, then the lower group index
// first; the picked groups ascending): a radix select of the threshold, then the groups above it and the first of the
// groups equal to it, in index order. The pooled key is the host's arithmetic in fp32 (its sums of squares in float, not
// double: equal to the last bits, the selection then sees the same order except at exact near-ties).
// Every kernel-launching helper is prefixed hy4fix_ (the SYCL kernel-name collision trap).
#include "ie/hy4flash_q8.hpp"
#include "ie/kernel_profiler.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>

namespace ie {

// out [hd] = the token norm of raw [hd] (x * rsqrt(mean(x^2) + eps) * w) -- a decode token's key into the ring its group pools
sycl::event hy4f_ix_norm_into(sycl::queue& q, const float* raw, const float* tok_norm, float* out, uint32_t hd, float eps) {
    constexpr int WG = 128;
    return ie::ps(q, "hy4fix_norm_into", [&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(WG, WG), [=](sycl::nd_item<1> it) {
            const uint32_t d = uint32_t(it.get_local_id(0));
            const float x = d < hd ? raw[d] : 0.f;
            const float ms = sycl::reduce_over_group(it.get_group(), x * x, sycl::plus<float>());
            if (d < hd) out[d] = x * (1.0f / sycl::sqrt(ms / float(hd) + eps)) * tok_norm[d];
        });
    });
}

// grp_out [hd] = the group's key: nk [ratio, hd] its token keys AFTER the token norm -> value
// kv . k and gate kg . k + ape[s] per slot -> softmax over the slots per dim -> the weighted values -> key norm
// (key_norm) -> the rope of the group's first position on the last rope_dim dims (pairs 2i, 2i + 1; cs / sn [rope_dim/2]).
sycl::event hy4f_ix_group_build(sycl::queue& q, const float* nk, const float* kv, const float* kg, const float* key_norm,
                                const float* ape, const float* cs, float* grp_out, uint32_t ratio, uint32_t hd, uint32_t rope_dim, float eps) {
    constexpr int WG = 128, RMAX = 8;
    return ie::ps(q, "hy4fix_group_build", [&](sycl::handler& h) {
        sycl::local_accessor<float, 1> k(sycl::range<1>(size_t(RMAX) * WG), h), pooled(sycl::range<1>(WG), h);
        h.parallel_for(sycl::nd_range<1>(WG, WG), [=](sycl::nd_item<1> it) {
            const uint32_t d = uint32_t(it.get_local_id(0)); auto g = it.get_group();
            for (uint32_t s = 0; s < ratio; ++s) if (d < hd) k[s * WG + d] = nk[size_t(s) * hd + d];
            sycl::group_barrier(g);
            float pv = 0.f;
            if (d < hd) {
                float val[RMAX], gat[RMAX];
                const float* rv = kv + size_t(d) * hd; const float* rg = kg + size_t(d) * hd;
                for (uint32_t s = 0; s < ratio; ++s) {
                    float av = 0.f, ag = 0.f;
                    for (uint32_t j = 0; j < hd; ++j) { const float kj = k[s * WG + j]; av += rv[j] * kj; ag += rg[j] * kj; }
                    val[s] = av; gat[s] = ag + ape[size_t(s) * hd + d];
                }
                float mx = gat[0];
                for (uint32_t s = 1; s < ratio; ++s) mx = sycl::fmax(mx, gat[s]);
                float den = 0.f, num = 0.f;
                for (uint32_t s = 0; s < ratio; ++s) { const float e = sycl::exp(gat[s] - mx); den += e; num += e * val[s]; }
                pv = num / den;
            }
            const float pms = sycl::reduce_over_group(g, pv * pv, sycl::plus<float>());
            const float pr = 1.0f / sycl::sqrt(pms / float(hd) + eps);
            if (d < hd) pooled[d] = pv * pr * key_norm[d];
            sycl::group_barrier(g);
            if (d < hd) {
                const uint32_t pass = hd - rope_dim;
                float o = pooled[d];
                if (d >= pass) {
                    const uint32_t i = (d - pass) / 2;
                    const float a = pooled[pass + 2 * i], b = pooled[pass + 2 * i + 1];
                    o = ((d - pass) % 2 == 0) ? a * cs[i] - b * cs[rope_dim / 2 + i] : b * cs[i] + a * cs[rope_dim / 2 + i];
                }
                grp_out[d] = o;
            }
        });
    });
}

// One row at position p (own = p / ratio > group_topk - 1): score [GC = own] -> the keep = group_topk - 1 best groups
// (score descending, the lower index first among equals), ascending; sel [S] = their keys (g * ratio + s), then
// own * ratio .. p, then -1; lo[0] = 0, hi[0] = the count - 1.
sycl::event hy4f_ix_select(sycl::queue& q, const float* score, uint32_t GC, uint32_t keep, uint32_t ratio, uint32_t p, uint32_t S,
                           int32_t* sel, int32_t* lo, int32_t* hi) {
    constexpr int WG = 1024;
    return ie::ps(q, "hy4fix_select", [&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> hist(sycl::range<1>(256), h);
        sycl::local_accessor<uint32_t, 1> sh(sycl::range<1>(4), h);
        h.parallel_for(sycl::nd_range<1>(WG, WG), [=](sycl::nd_item<1> it) {
            const uint32_t lid = uint32_t(it.get_local_id(0)); auto g = it.get_group();
            // a float's bits as an unsigned that orders like the float (larger score = larger key)
            auto key = [&](uint32_t i) -> uint32_t { const uint32_t b = sycl::bit_cast<uint32_t>(score[i]); return (b & 0x80000000u) ? ~b : (b | 0x80000000u); };
            // radix select, most significant byte first: the keep-th largest key
            uint32_t prefix = 0, pmask = 0, need = keep;            // need: how many of the keys matching prefix are still to take
            for (int shift = 24; shift >= 0; shift -= 8) {
                for (uint32_t b = lid; b < 256; b += WG) hist[b] = 0;
                sycl::group_barrier(g);
                for (uint32_t i = lid; i < GC; i += WG) {
                    const uint32_t k = key(i);
                    if ((k & pmask) == prefix) sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space>(hist[(k >> shift) & 255]).fetch_add(1u);
                }
                sycl::group_barrier(g);
                if (lid == 0) {
                    uint32_t acc = 0, b = 255;
                    for (;; --b) { if (acc + hist[b] >= need) break; acc += hist[b]; if (b == 0) break; }
                    sh[0] = b; sh[1] = need - acc;
                }
                sycl::group_barrier(g);
                prefix |= sh[0] << shift; pmask |= 255u << shift; need = sh[1];
                sycl::group_barrier(g);
            }
            const uint32_t thr = prefix, n_eq_take = need;          // every key > thr, and the first n_eq_take keys == thr
            // each lane a contiguous block of groups: its count of equals and of picks, scanned over the lanes in index order
            const uint32_t per = (GC + WG - 1) / WG, b0 = sycl::min(GC, lid * per), b1 = sycl::min(GC, b0 + per);
            uint32_t eq = 0;
            for (uint32_t i = b0; i < b1; ++i) eq += key(i) == thr;
            const uint32_t eq_before = sycl::exclusive_scan_over_group(g, eq, sycl::plus<uint32_t>());
            uint32_t pick = 0, e2 = eq_before;
            for (uint32_t i = b0; i < b1; ++i) { const uint32_t k = key(i); if (k > thr) ++pick; else if (k == thr) { if (e2 < n_eq_take) ++pick; ++e2; } }
            uint32_t at = sycl::exclusive_scan_over_group(g, pick, sycl::plus<uint32_t>());
            e2 = eq_before;
            for (uint32_t i = b0; i < b1; ++i) {
                const uint32_t k = key(i); bool take = false;
                if (k > thr) take = true; else if (k == thr) { take = e2 < n_eq_take; ++e2; }
                if (take) { for (uint32_t s = 0; s < ratio; ++s) sel[size_t(at) * ratio + s] = int32_t(i * ratio + s); ++at; }
            }
            // the row's own group up to itself, then the padding
            const uint32_t own = p / ratio, base = keep * ratio, n_own = p - own * ratio + 1, n = base + n_own;
            for (uint32_t j = lid; j < S; j += WG) if (j >= base) sel[j] = j < n ? int32_t(own * ratio + (j - base)) : -1;
            if (lid == 0) { lo[0] = 0; hi[0] = int32_t(n) - 1; }
        });
    });
}


// Many rows at once (a prompt chunk): row t at position p = pos0 + t, its scores at score + t * stride (groups 0 .. own - 1
// read). own = p / ratio <= keep: every key 0 .. p (the host's within-the-budget rule); otherwise the selection above.
sycl::event hy4f_ix_select_rows(sycl::queue& q, const float* score, size_t stride, uint32_t rows, uint32_t pos0, uint32_t keep,
                                uint32_t ratio, uint32_t S, int32_t* sel, int32_t* lo, int32_t* hi) {
    constexpr int WG = 1024;
    return ie::ps(q, "hy4fix_select_rows", [&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> hist(sycl::range<1>(256), h);
        sycl::local_accessor<uint32_t, 1> sh(sycl::range<1>(4), h);
        h.parallel_for(sycl::nd_range<1>(size_t(rows) * WG, WG), [=](sycl::nd_item<1> it) {
            const uint32_t t = uint32_t(it.get_group(0)), lid = uint32_t(it.get_local_id(0)); auto g = it.get_group();
            const uint32_t p = pos0 + t, own = p / ratio;
            int32_t* srow = sel + uint64_t(t) * S;
            if (own <= keep) {                                   // within the budget: every key up to the row
                for (uint32_t j = lid; j < S; j += WG) srow[j] = j <= p ? int32_t(j) : -1;
                if (lid == 0) { lo[t] = 0; hi[t] = int32_t(p); }
                return;
            }
            const float* sc = score + uint64_t(t) * stride; const uint32_t GC = own;
            auto key = [&](uint32_t i) -> uint32_t { const uint32_t b = sycl::bit_cast<uint32_t>(sc[i]); return (b & 0x80000000u) ? ~b : (b | 0x80000000u); };
            uint32_t prefix = 0, pmask = 0, need = keep;
            for (int shift = 24; shift >= 0; shift -= 8) {
                for (uint32_t b = lid; b < 256; b += WG) hist[b] = 0;
                sycl::group_barrier(g);
                for (uint32_t i = lid; i < GC; i += WG) {
                    const uint32_t k = key(i);
                    if ((k & pmask) == prefix) sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space>(hist[(k >> shift) & 255]).fetch_add(1u);
                }
                sycl::group_barrier(g);
                if (lid == 0) {
                    uint32_t acc = 0, b = 255;
                    for (;; --b) { if (acc + hist[b] >= need) break; acc += hist[b]; if (b == 0) break; }
                    sh[0] = b; sh[1] = need - acc;
                }
                sycl::group_barrier(g);
                prefix |= sh[0] << shift; pmask |= 255u << shift; need = sh[1];
                sycl::group_barrier(g);
            }
            const uint32_t thr = prefix, n_eq_take = need;
            const uint32_t per = (GC + WG - 1) / WG, b0 = sycl::min(GC, lid * per), b1 = sycl::min(GC, b0 + per);
            uint32_t eq = 0;
            for (uint32_t i = b0; i < b1; ++i) eq += key(i) == thr;
            const uint32_t eq_before = sycl::exclusive_scan_over_group(g, eq, sycl::plus<uint32_t>());
            uint32_t pick = 0, e2 = eq_before;
            for (uint32_t i = b0; i < b1; ++i) { const uint32_t k = key(i); if (k > thr) ++pick; else if (k == thr) { if (e2 < n_eq_take) ++pick; ++e2; } }
            uint32_t at = sycl::exclusive_scan_over_group(g, pick, sycl::plus<uint32_t>());
            e2 = eq_before;
            for (uint32_t i = b0; i < b1; ++i) {
                const uint32_t k = key(i); bool take = false;
                if (k > thr) take = true; else if (k == thr) { take = e2 < n_eq_take; ++e2; }
                if (take) { for (uint32_t s2 = 0; s2 < ratio; ++s2) srow[uint64_t(at) * ratio + s2] = int32_t(i * ratio + s2); ++at; }
            }
            const uint32_t base = keep * ratio, n_own = p - own * ratio + 1, n = base + n_own;
            for (uint32_t j = lid; j < S; j += WG) if (j >= base) srow[j] = j < n ? int32_t(own * ratio + (j - base)) : -1;
            if (lid == 0) { lo[t] = 0; hi[t] = int32_t(n) - 1; }
        });
    });
}

}  // namespace ie
