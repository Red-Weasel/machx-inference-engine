// src/ops/deepseek4_topk_radix.cpp — the radix-select top-k shape, in its OWN
// translation unit.  2026-08-09.
//
// WHAT THE SHAPE IS FOR.  O(n_keys) selection with CONSTANT SLM (~21 KB:
// per-sub-group histogram rows + survivors + broadcast), where the bitonic
// shape needs the whole padded key array in SLM — dead past n_keys 16384 =
// ctx 64k at compress ratio 4 — and the scan shape is O(top_k * n_keys), 512
// full rescans per token.  The dispatcher uses radix ONLY past the bitonic's
// SLM wall: below it the bitonic measures faster (n_keys 4096: 0.099 ms vs
// 0.110 ms), above it the scan was never usable — at the ctx-200000 target
// (n_keys 50000) this shape is what makes the sparse indexer viable at all.
//
// WHY ATOMIC-FREE, AND WHY ITS OWN TU.  The first implementation used
// sycl::atomic_ref on SLM for the histogram and the survivor append.  On this
// stack (icpx 2025 era, spir64 JIT, B70) SLM atomics SILENTLY LOSE UPDATES in
// two probed regimes: (a) WG=1024 — even a constant-trip-count loop of
// fetch_add(1) reads back wrong; (b) WG=256 with a DYNAMICALLY-bounded loop —
// the identical kernel histograms correctly when n_keys is a compile-time
// constant and reads back ALL-ZERO when n_keys is a runtime value (isolated
// 2026-08-09 with a standalone probe pair; the constant build unrolls and the
// atomics survive, the dynamic build strips them).  Nothing else in this
// repository uses local-space atomics — keep it that way.  This kernel is
// race-free by STRUCTURE instead: per-sub-group histogram rows updated one
// lane at a time under a uniform trip count (so the sub-group barrier never
// diverges), and a count/exclusive-scan/write collection.
//
// Selection runs over the SAME 64-bit pack_key strict total order the other
// shapes use, so ties and sentinels resolve identically: 8 MSB-first radix
// passes pin the exact top_k-th largest key, a collection pass gathers the
// >= threshold survivors (exactly top_k — keys are unique by index), and an
// ascending bitonic over K2 = pow2(top_k) keys reproduces the exact
// descending emission, -1 sentinels included.  Gated by
// deepseek4_decode_gate_test §5 against the scan reference over nine shapes,
// including n_keys 16384 and the ctx-200k-class 50000.

#include "ie/deepseek4_attn.hpp"
#include "ie/kernel_profiler.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <limits>

namespace ie {
namespace {

constexpr int   kSG     = 16;
constexpr float kNegInf = -std::numeric_limits<float>::infinity();

// "largest score, smallest index on a tie" — identical to deepseek4_attn.cpp.
inline uint64_t pack_key(float v, uint32_t idx) {
    uint32_t u = sycl::bit_cast<uint32_t>(v);
    u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return (uint64_t(u) << 32) | uint64_t(0xFFFFFFFFu - idx);
}

}  // namespace

sycl::event ds4_indexer_topk_radix256(sycl::queue& q,
                                      const float* scores, const int32_t* positions,
                                      int32_t* out,
                                      uint32_t T, uint32_t n_keys, uint32_t top_k,
                                      uint32_t compress_rate, uint32_t K2,
                                      const std::vector<sycl::event>& deps) {
    // WG: 256 was the original, and its "WG=256 ONLY" warning below is about an SLM-ATOMIC histogram
    // that no longer exists here -- this shape is atomic-free by structure, so more lanes are testable.
    // More lanes halve the per-pass iteration count, which is what a decode call (ONE work-group per
    // query row, nothing to hide its latency) is bound by. SLM at 512: hist_sg 32 KB + hist 2 + surv 4.
    static const uint32_t WGe = [] {
        const char* e = std::getenv("IE_DS4_TOPK_WG");
        const long v = e && *e ? std::atol(e) : 512;      // 512 default: 4.33 -> 2.45 ms/token at ctx 32,768, bit-identical
        return (v == 256 || v == 512 || v == 1024) ? uint32_t(v) : 512u;
    }();
    const uint32_t WG  = (uint64_t(WGe / uint32_t(kSG)) * 256u * 4u + uint64_t(WGe) * 4u + uint64_t(K2) * 8u + 8u)
                         <= q.get_device().get_info<sycl::info::device::local_mem_size>()
                         && WGe <= q.get_device().get_info<sycl::info::device::max_work_group_size>() ? WGe : 256u;
    const uint32_t nSG = WG / uint32_t(kSG);
    return ie::ps(q, "ds4_indexer_topk", [&](sycl::handler& h) {
        h.depends_on(deps);
        // NO SLM ATOMICS ANYWHERE IN THIS KERNEL.  sycl::atomic_ref on local
        // memory silently LOSES updates inside dynamically-bounded loops on
        // this stack (probed 2026-08-09: the identical kernel with
        // compile-time-constant n_keys histograms correctly; with runtime
        // n_keys the histogram reads back all-zero.  WG=1024 loses local
        // atomics even in constant loops).  Everything below is plain SLM
        // reads/writes made race-free by structure: per-sub-group histogram
        // rows updated one lane at a time, and a count/exclusive-scan/write
        // collection.
        sycl::local_accessor<uint32_t, 1> hist_sg(sycl::range<1>(size_t(nSG) * 256), h);
        sycl::local_accessor<uint32_t, 1> hist(sycl::range<1>(std::max(256u, WG)), h);   // 256 bins AND WG lane counts
        sycl::local_accessor<uint64_t, 1> surv(sycl::range<1>(K2), h);
        sycl::local_accessor<uint32_t, 1> bcast(sycl::range<1>(2), h);
        h.parallel_for(sycl::nd_range<1>(size_t(T) * WG, WG),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
            const uint32_t t   = uint32_t(it.get_group(0));
            const uint32_t lid = uint32_t(it.get_local_id(0));
            const auto     sg  = it.get_sub_group();
            const uint32_t sgi = uint32_t(sg.get_group_linear_id());
            const uint32_t sgl = uint32_t(sg.get_local_linear_id());
            const int64_t  thr = (int64_t(positions[t]) + 1) / int64_t(compress_rate);
            const float*   row = scores + size_t(t) * n_keys;
            const auto key_at = [&](uint32_t e) -> uint64_t {
                const float v = (int64_t(e) < thr) ? row[e] : kNegInf;
                return pack_key(v, e);
            };
            // Uniform trip count so the sub-group rotation below never sees
            // divergent lanes at its barrier.
            const uint32_t iters = (n_keys + WG - 1u) / WG;
            // After pass p: `prefix` holds the top (p+1)*8 bits of the
            // top_k-th largest key; `want` its 1-based rank among the keys
            // sharing that prefix.
            uint64_t prefix     = 0;
            uint32_t shift_done = 0;
            uint32_t want       = top_k;
            for (uint32_t pass = 0; pass < 8; ++pass) {
                const uint32_t sh = 56u - pass * 8u;
                for (uint32_t i = lid; i < nSG * 256u; i += WG) hist_sg[i] = 0;
                it.barrier(sycl::access::fence_space::local_space);
                for (uint32_t ii = 0; ii < iters; ++ii) {
                    const uint32_t e    = lid + ii * WG;
                    const bool     ok   = e < n_keys;
                    const uint64_t k    = ok ? key_at(e) : 0;
                    const bool     live = ok &&
                        !(shift_done && (k >> (64u - shift_done)) != prefix);
                    const uint32_t d = uint32_t((k >> sh) & 0xFFu);
                    // One lane of the sub-group updates its OWN row at a time:
                    // plain read-modify-write, no concurrent writer possible.
                    //
                    // THIS ROTATION IS NOT THIS KERNEL'S COST -- measured, 2026-09-15, because it looked
                    // like the obvious suspect for long-context decode (this kernel is EXACTLY linear in
                    // n_keys: 4.00x on n_keys x4, ~26-34 ms/token at ctx 200k-262k).  A 4-bit/per-lane-row
                    // variant that removes the rotation ENTIRELY measured 4.31 ms/token against this
                    // form's 4.26 at ctx 32,768 -- no difference, because 16 passes double the key reads
                    // by as much as the rotation saved.  The kernel is MEMORY-LATENCY bound with ONE
                    // work-group per query row at decode, so the lever is more work-groups over the key
                    // range (a two-level histogram), not a cheaper histogram.  The variant was removed
                    // rather than kept behind a flag; docs/deepseek41/68 records it.
                    for (uint32_t l = 0; l < uint32_t(kSG); ++l) {
                        if (sgl == l && live) hist_sg[sgi * 256u + d] += 1u;
                        sycl::group_barrier(sg);
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);
                if (lid < 256) {
                    uint32_t s = 0;
                    for (uint32_t r = 0; r < nSG; ++r) s += hist_sg[r * 256u + lid];
                    hist[lid] = s;
                }
                it.barrier(sycl::access::fence_space::local_space);
                if (lid == 0) {
                    uint32_t cum = 0, digit = 0, w = want;
                    for (int32_t b = 255; b >= 0; --b) {
                        const uint32_t c = hist[uint32_t(b)];
                        if (cum + c >= w) { digit = uint32_t(b); w -= cum; break; }
                        cum += c;
                    }
                    bcast[0] = digit;
                    bcast[1] = w;
                }
                it.barrier(sycl::access::fence_space::local_space);
                prefix = (prefix << 8) | uint64_t(bcast[0]);
                want   = bcast[1];
                shift_done += 8;
                it.barrier(sycl::access::fence_space::local_space);
            }
            const uint64_t kth = prefix;
            // Collection, atomic-free: count per lane, exclusive-scan the
            // per-lane counts (hist doubles as the scan array — the passes are
            // done with it), then each lane writes its survivors at its base.
            uint32_t mine = 0;
            for (uint32_t e = lid; e < n_keys; e += WG) mine += key_at(e) >= kth ? 1u : 0u;
            hist[lid] = mine;
            it.barrier(sycl::access::fence_space::local_space);
            if (lid == 0) {
                uint32_t run = 0;
                for (uint32_t i = 0; i < WG; ++i) {
                    const uint32_t c = hist[i];
                    hist[i] = run;
                    run += c;
                }
            }
            it.barrier(sycl::access::fence_space::local_space);
            for (uint32_t i = lid; i < K2; i += WG) surv[i] = 0;
            it.barrier(sycl::access::fence_space::local_space);
            uint32_t at = hist[lid];
            for (uint32_t e = lid; e < n_keys; e += WG) {
                const uint64_t k = key_at(e);
                if (k >= kth) surv[at++] = k;
            }
            it.barrier(sycl::access::fence_space::local_space);
            // Ascending bitonic over K2 keys (pads are 0 = smallest); the j-th
            // largest sits at K2-1-j — the identical read-out the bitonic
            // shape uses.
            for (uint32_t k = 2; k <= K2; k <<= 1) {
                for (uint32_t j = k >> 1; j > 0; j >>= 1) {
                    for (uint32_t i = lid; i < K2; i += WG) {
                        const uint32_t ixj = i ^ j;
                        if (ixj > i) {
                            const bool up = ((i & k) == 0);
                            const uint64_t a = surv[i], b = surv[ixj];
                            if ((a > b) == up) { surv[i] = b; surv[ixj] = a; }
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }
            }
            for (uint32_t j = lid; j < top_k; j += WG) {
                const int32_t idx =
                    int32_t(0xFFFFFFFFu - uint32_t(surv[K2 - 1 - j] & 0xFFFFFFFFu));
                out[size_t(t) * top_k + j] = (int64_t(idx) >= thr) ? -1 : idx;
            }
        });
    });
}

}  // namespace ie
