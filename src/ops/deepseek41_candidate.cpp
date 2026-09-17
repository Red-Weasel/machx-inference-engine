// src/ops/deepseek41_candidate.cpp — V4.1's TWO-LEVEL candidate top-k, level one.  2026-09-15.
//
// WHAT IT IS.  Above `candidate_topk_blocks * candidate_block_size` compressed latents (2048 * 8 =
// 16,384 for this checkpoint) V4.1 does not run its fine top-512 over the whole latent cache.  One
// layer -- `candidate_source_layer_id`, 20 here -- first keeps the 2,048 highest-scoring BLOCKS of 8
// consecutive latents, and layers 24/28/32/36 then score with their own weights but only inside
// those blocks.  Below 16,384 latents `min(topk_blocks, n_blocks)` keeps every block and the whole
// stage is a no-op, which is why the engine never needed it until Phase 24 took the context past 16k.
//
// THE REFERENCE, term by term (inference/model.py:583-610, `select_candidate_blocks`):
//   * a block's score is the MAX over its 8 latents (`amax`), with latents the query cannot yet
//     reach already at -inf, which is what makes a block score of -inf mean "not reachable yet";
//   * the block holding this query's NEWEST reachable latent is PINNED IN at +inf
//     (`last = (compress_len - 1) // block_size`), because it is only partly filled and would
//     otherwise lose to an older full block while holding the most recent tokens;
//   * `topk(min(topk_blocks, n_blocks))`, then keep only the picks whose value is `> -inf`, so a
//     query with fewer reachable blocks than topk_blocks does not admit -inf padding;
//   * the mask is expanded by block_size back to the latent axis.
//
// WHERE THE APPROXIMATION ACTUALLY LIVES, proved rather than assumed.  Within ONE layer's own scores
// this stage cannot change that layer's fine top-k at all, whenever index_topk <= candidate_topk_blocks
// (512 <= 2048 here).  Let e* be one of the layer's top-512 entries.  Its block's score is the max over
// that block, so it is >= s(e*).  For the block to fall outside the top 2048, at least 2048 blocks must
// score strictly higher, and each of those contains an entry scoring above s(e*) -- distinct blocks,
// distinct entries -- so at least 2048 entries would beat e*.  But e* is in the top 512, so at most 511
// do.  Contradiction, so every block holding a top-512 entry is inside the top 512 blocks, let alone the
// top 2048.  This is why the reference has the source layer fall through UNMASKED: masking itself would
// be a no-op.  It follows that the whole accuracy cost of the two-level scheme is CROSS-LAYER -- layers
// 24/28/32/36 are pruned by a ranking computed from layer 20's weights, not their own -- which is what
// makes an end-to-end retrieval check at depth the right gate, and IE_DS41_CAND=0 the arm to measure
// against.
//
// TIE-BREAK, stated because it is the one place this can differ from the reference.  Selection runs
// over the same 64-bit `pack_key` strict total order the rest of this engine's top-k shapes use --
// largest score, and on an equal score the SMALLER block index wins -- so there are no ties and the
// result is deterministic.  `torch.topk(sorted=False)` breaks an equal score arbitrarily, so on a
// genuine tie at exactly the 2,048th block the two can pick different blocks.  Block scores are
// maxima of 8 continuous floats, so this needs two blocks whose best latent scores are bit-equal.
//
// NO SLM ATOMICS.  Same hard constraint as src/ops/deepseek4_topk_radix.cpp, and for the same
// measured reason: `sycl::atomic_ref` on local memory silently loses updates inside
// dynamically-bounded loops on this stack.  The histogram here is race-free by STRUCTURE -- one lane
// of a sub-group updates its own row at a time under a uniform trip count.

#include "ie/deepseek41_candidate.hpp"
#include "ie/kernel_profiler.hpp"

#include <sycl/sycl.hpp>
#include <cstdint>
#include <limits>
#include <algorithm>

namespace ie {
namespace {

constexpr int   kDs41CandSG = 16;
constexpr float kDs41CandNegInf = -std::numeric_limits<float>::infinity();

// "largest score, smallest index on a tie" -- the same strict total order as deepseek4_attn.cpp and
// deepseek4_topk_radix.cpp, so ties and sentinels resolve identically across every top-k in the engine.
inline uint64_t ds41_cand_pack_key(float v, uint32_t idx) {
    uint32_t u = sycl::bit_cast<uint32_t>(v);
    u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return (uint64_t(u) << 32) | uint64_t(0xFFFFFFFFu - idx);
}

}  // namespace

sycl::event ds41_candidate_block_scores(sycl::queue& q,
                                        const float* scores, const int32_t* positions,
                                        float* bscore,
                                        uint32_t T, uint32_t n_keys, uint32_t block_size,
                                        uint32_t compress_rate,
                                        const std::vector<sycl::event>& deps) {
    const uint32_t NB = (n_keys + block_size - 1u) / block_size;
    return ie::ps(q, "ds41_candidate_bscore", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(size_t(T) * NB), [=](sycl::id<1> i) {
            const uint32_t t = uint32_t(i[0] / NB), b = uint32_t(i[0] % NB);
            const int64_t thr = (int64_t(positions[t]) + 1) / int64_t(compress_rate);
            const float* row = scores + size_t(t) * n_keys;
            // the partly-filled block holding this query's newest reachable latent is pinned in
            if (thr > 0 && b == uint32_t((thr - 1) / int64_t(block_size))) {
                bscore[i] = -kDs41CandNegInf; return;                  // +inf
            }
            float m = kDs41CandNegInf;
            const uint32_t e0 = b * block_size, e1 = sycl::min(e0 + block_size, n_keys);
            for (uint32_t e = e0; e < e1; ++e)
                if (int64_t(e) < thr) m = sycl::fmax(m, row[e]);
            bscore[i] = m;
        });
    });
}

sycl::event ds41_candidate_select(sycl::queue& q,
                                  const float* bscore, uint8_t* keep,
                                  uint32_t T, uint32_t n_keys, uint32_t block_size,
                                  uint32_t topk_blocks,
                                  const std::vector<sycl::event>& deps) {
    const uint32_t NB = (n_keys + block_size - 1u) / block_size;
    const uint32_t K  = std::min(topk_blocks, NB);
    constexpr uint32_t WG  = 256;
    constexpr uint32_t nSG = WG / uint32_t(kDs41CandSG);
    return ie::ps(q, "ds41_candidate_select", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<uint32_t, 1> hist_sg(sycl::range<1>(size_t(nSG) * 256), h);
        sycl::local_accessor<uint32_t, 1> hist(sycl::range<1>(256), h);
        sycl::local_accessor<uint32_t, 1> bcast(sycl::range<1>(2), h);
        h.parallel_for(sycl::nd_range<1>(size_t(T) * WG, WG),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kDs41CandSG)]] {
            const uint32_t t   = uint32_t(it.get_group(0));
            const uint32_t lid = uint32_t(it.get_local_id(0));
            const auto     sg  = it.get_sub_group();
            const uint32_t sgi = uint32_t(sg.get_group_linear_id());
            const uint32_t sgl = uint32_t(sg.get_local_linear_id());
            const float*   row = bscore + size_t(t) * NB;
            uint8_t*       out = keep   + size_t(t) * NB;
            // K == NB: every block is picked, so the only filter left is the reference's `> -inf`
            if (K >= NB) {
                for (uint32_t b = lid; b < NB; b += WG) out[b] = row[b] > kDs41CandNegInf ? 1 : 0;
                return;
            }
            const uint32_t iters = (NB + WG - 1u) / WG;              // uniform, for the sub-group rotation
            uint64_t prefix = 0; uint32_t shift_done = 0, want = K;
            for (uint32_t pass = 0; pass < 8; ++pass) {
                const uint32_t sh = 56u - pass * 8u;
                for (uint32_t i = lid; i < nSG * 256u; i += WG) hist_sg[i] = 0;
                it.barrier(sycl::access::fence_space::local_space);
                for (uint32_t ii = 0; ii < iters; ++ii) {
                    const uint32_t b  = lid + ii * WG;
                    const bool     ok = b < NB;
                    const uint64_t k  = ok ? ds41_cand_pack_key(row[b], b) : 0;
                    const bool     live = ok && !(shift_done && (k >> (64u - shift_done)) != prefix);
                    const uint32_t d = uint32_t((k >> sh) & 0xFFu);
                    for (uint32_t l = 0; l < uint32_t(kDs41CandSG); ++l) {
                        if (sgl == l && live) hist_sg[sgi * 256u + d] += 1u;
                        sycl::group_barrier(sg);
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);
                if (lid < 256) { uint32_t s = 0; for (uint32_t r = 0; r < nSG; ++r) s += hist_sg[r * 256u + lid]; hist[lid] = s; }
                it.barrier(sycl::access::fence_space::local_space);
                if (lid == 0) {
                    uint32_t cum = 0, digit = 0, w = want;
                    for (int32_t b = 255; b >= 0; --b) {
                        const uint32_t cc = hist[uint32_t(b)];
                        if (cum + cc >= w) { digit = uint32_t(b); w -= cum; break; }
                        cum += cc;
                    }
                    bcast[0] = digit; bcast[1] = w;
                }
                it.barrier(sycl::access::fence_space::local_space);
                prefix = (prefix << 8) | uint64_t(bcast[0]);
                want = bcast[1]; shift_done += 8;
                it.barrier(sycl::access::fence_space::local_space);
            }
            const uint64_t kth = prefix;    // the exact K-th largest key; no ties, so exactly K are >= it
            for (uint32_t b = lid; b < NB; b += WG)
                out[b] = (ds41_cand_pack_key(row[b], b) >= kth && row[b] > kDs41CandNegInf) ? 1 : 0;
        });
    });
}

sycl::event ds41_candidate_apply(sycl::queue& q,
                                 const uint8_t* keep, float* scores,
                                 uint32_t T, uint32_t n_keys, uint32_t block_size,
                                 const std::vector<sycl::event>& deps) {
    const uint32_t NB = (n_keys + block_size - 1u) / block_size;
    return ie::ps(q, "ds41_candidate_apply", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(size_t(T) * n_keys), [=](sycl::id<1> i) {
            const uint32_t t = uint32_t(i[0] / n_keys), e = uint32_t(i[0] % n_keys);
            if (!keep[size_t(t) * NB + e / block_size]) scores[i] = kDs41CandNegInf;
        });
    });
}

sycl::event ds41_sort_picks_asc(sycl::queue& q,
                                const int32_t* picks, int32_t* out, int32_t* n_valid,
                                uint32_t T, uint32_t k, uint32_t stride,
                                const std::vector<sycl::event>& deps) {
    // Bitonic sort, one work-group per query row, the row staged in SLM padded to a power of two with INT32_MAX
    // so the pad sorts to the END and never displaces a real pick. NO SLM ATOMICS (the constraint at the top of
    // this file): a bitonic network is pure compare-exchange under barriers, which is why it is the right shape
    // here. k is at most index_topk = 512, so P is at most 512 and one 256-lane group does it in 45 steps.
    //
    // The -1 sentinels are staged as INT32_MAX too, so they sort to the END with the pads, and the WHOLE row pitch
    // is written back (sentinels and pads as -1). That is what makes the row T-independent for the attention
    // kernel: a row's valid picks then occupy [0, n_valid) whatever k was, and k is min(index_topk, NC), which
    // DIFFERS between a T-row step and the one-row steps it must equal (NC counts this step's own latents). The
    // first form sorted sentinels to the front and wrote only [0, k), so a row's picks shifted with k and the
    // kernel's pitch -- fed k, not stride -- read row 1 and up from the wrong offset (multi test: T-1 rows wrong
    // on the 12-token prompt, none on the 2,048-token one where k == stride).
    //
    // The INPUT pitch is k, not stride: ds4_indexer_topk clamps top_k to min(index_topk, n_keys) and emits
    // out[t * top_k + j], so its rows are packed at the emitted width (ds4_block_bias_topk reads them that way).
    // Reading the input at `stride` took row 1 and up from row 0's tail -- the same T-1-rows failure.
    uint32_t P = 1; while (P < k) P <<= 1;
    const uint32_t WG = std::min<uint32_t>(256, P);
    return ie::ps(q, "ds41_sort_picks_asc", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<int32_t, 1> a(sycl::range<1>(P), h);
        h.parallel_for(sycl::nd_range<1>(size_t(T) * WG, WG), [=](sycl::nd_item<1> it) {
            const uint32_t t = uint32_t(it.get_group(0)), lid = uint32_t(it.get_local_id(0));
            for (uint32_t i = lid; i < P; i += WG) {
                const int32_t v = i < k ? picks[size_t(t) * k + i] : -1;
                a[i] = v < 0 ? INT32_MAX : v;
            }
            it.barrier(sycl::access::fence_space::local_space);
            for (uint32_t len = 2; len <= P; len <<= 1)
                for (uint32_t st = len >> 1; st > 0; st >>= 1) {
                    for (uint32_t i = lid; i < P; i += WG) {
                        const uint32_t j = i ^ st;
                        if (j > i) {
                            const bool up = ((i & len) == 0);
                            const int32_t x = a[i], y = a[j];
                            if ((x > y) == up) { a[i] = y; a[j] = x; }
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }
            for (uint32_t i = lid; i < stride; i += WG)
                out[size_t(t) * stride + i] = (i < P && a[i] != INT32_MAX) ? a[i] : -1;
            // Phase 33: the row's valid count = the first INT32_MAX in the sorted row (one binary search, lane 0).
            if (n_valid && lid == 0) {
                uint32_t lo = 0, hi = P;
                while (lo < hi) { const uint32_t mid = (lo + hi) >> 1; if (a[mid] == INT32_MAX) hi = mid; else lo = mid + 1; }
                n_valid[t] = int32_t(lo);
            }
        });
    });
}

}  // namespace ie
