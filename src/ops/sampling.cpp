// src/ops/sampling.cpp — sampling kernels.
//
// All operate on a single token's logits buffer of size `vocab` (≤ 256k for
// Qwen3.6). Cheap enough that performance is dominated by launch overhead;
// the kernels here are correctness-first.

#include "ie/kernel_profiler.hpp"
#include "ie/ops.hpp"

#include <sycl/sycl.hpp>

#include <limits>

namespace ie {

namespace {

// xorshift64 — small state RNG that lives on device.
inline uint64_t xorshift64(uint64_t& s) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
}
inline float to_unit_float(uint64_t r) {
    // top 24 bits → [0, 1). Slightly biased but fine for sampling.
    return float(r >> 40) * (1.0f / float(1u << 24));
}

}  // namespace

sycl::event sample_argmax(sycl::queue& q,
                          const sycl::half* logits, int32_t* out,
                          uint32_t vocab,
                          const std::vector<sycl::event>& deps) {
    constexpr uint32_t WG = 256;
    return ie::ps(q, "argmax", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<float, 1>   max_v(WG, h);
        sycl::local_accessor<int32_t, 1> max_i(WG, h);
        h.parallel_for(sycl::nd_range<1>(WG, WG),
                       [=](sycl::nd_item<1> it) {
            const uint32_t lid = uint32_t(it.get_local_id(0));
            float local_max = -std::numeric_limits<float>::infinity();
            int32_t local_idx = -1;
            for (uint32_t i = lid; i < vocab; i += WG) {
                const float v = float(logits[i]);
                if (v > local_max) { local_max = v; local_idx = int32_t(i); }
            }
            max_v[lid] = local_max;
            max_i[lid] = local_idx;
            sycl::group_barrier(it.get_group());
            // Tree reduce within WG.
            for (uint32_t step = WG / 2; step > 0; step /= 2) {
                if (lid < step) {
                    if (max_v[lid + step] > max_v[lid]) {
                        max_v[lid] = max_v[lid + step];
                        max_i[lid] = max_i[lid + step];
                    }
                }
                sycl::group_barrier(it.get_group());
            }
            if (lid == 0) *out = max_i[0];
        });
    });
}

sycl::event repetition_penalty(sycl::queue& q,
                               sycl::half* logits, uint32_t vocab,
                               const int32_t* recent_ids, uint32_t n_recent,
                               float penalty,
                               const std::vector<sycl::event>& deps) {
    if (penalty == 1.0f || n_recent == 0) {
        return q.submit([&](sycl::handler& h) { h.depends_on(deps); h.single_task([](){}); });
    }
    return ie::ps(q, "rep_penalty", [&](sycl::handler& h) {
        h.depends_on(deps);
        constexpr uint32_t WG = 64;
        const uint32_t global = ((n_recent + WG - 1) / WG) * WG;
        h.parallel_for(sycl::nd_range<1>(global, WG),
                       [=](sycl::nd_item<1> it) {
            const uint32_t i = uint32_t(it.get_global_id(0));
            if (i >= n_recent) return;
            const int32_t id = recent_ids[i];
            if (id < 0 || uint32_t(id) >= vocab) return;
            const float v = float(logits[id]);
            const float adjusted = (v > 0.f) ? v / penalty : v * penalty;
            // Race-tolerant: if multiple recent_ids[*] hit the same id, the
            // last writer wins, which mirrors HF's behaviour.
            logits[id] = sycl::half(adjusted);
        });
    });
}

sycl::event sample_softmax_topk_topp(sycl::queue& q,
                                     sycl::half* logits, int32_t* out,
                                     uint32_t vocab,
                                     float temperature, uint32_t top_k,
                                     float top_p, float min_p,
                                     uint64_t rng_state,
                                     const std::vector<sycl::event>& deps) {
    if (temperature <= 0.f) return sample_argmax(q, logits, out, vocab, deps);

    // v1.3 (2026-06-09): WG-parallel sampler — replaces the Phase 7 serial
    // single_task (which walked the 248k vocab once per top-k/top-p round on
    // ONE thread).  One WG of 512 lanes; lane l owns the strided slice
    // {l, l+512, ...}.
    //
    // Key insight: softmax order is monotonic in raw logits, so ALL selection
    // (max, top-k, min-p threshold, top-p extraction order) happens in logit
    // space; probabilities are computed in fp32 only for the extracted
    // winners.  No half-precision prob round-trips (more precise than the
    // old serial version, which stored probs back into the half logits).
    //
    // Extraction: each lane caches a (local-max, idx) over its slice; per
    // round a WG-wide max-reduce on sign-fixed (logit, idx) packed keys picks
    // the global winner, then ONLY the owner lane tombstones it in `logits`
    // and rescans its slice.  Stop on top_k reached, top-p mass reached,
    // min-p floor crossed, or K_CAP.
    //
    // Contract notes (this kernel had no callers before — semantics set here):
    //   - The final CDF draw runs in DESCENDING-probability order
    //     (llama.cpp convention), not vocab-index order.
    //   - At most K_CAP=1024 candidates are considered even when top_k==0;
    //     for real LLM distributions the tail beyond the top-1024 carries
    //     negligible mass.  Exact full-vocab multinomial is NOT provided.
    //   - `logits` is mutated (winners tombstoned) — per-step scratch, same
    //     as the serial version's in-place behaviour.
    constexpr uint32_t WG    = 512;
    constexpr uint32_t K_CAP = 1024;
    const uint32_t k_eff = (top_k > 0 && top_k < K_CAP) ? top_k : K_CAP;

    return ie::ps(q, "topk_topp", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<int32_t, 1> keep_idx(K_CAP, h);
        sycl::local_accessor<float, 1>   keep_p(K_CAP, h);
        sycl::local_accessor<int32_t, 1> ctl(1, h);

        h.parallel_for(sycl::nd_range<1>(WG, WG),
                       [=](sycl::nd_item<1> it) {
            const uint32_t lid = uint32_t(it.get_local_id(0));
            auto grp = it.get_group();

            // Monotonic uint mapping for signed-float compare-by-bits.
            auto fkey = [](float v) {
                const uint32_t b = sycl::bit_cast<uint32_t>(v);
                return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
            };

            // Pass 1: global max logit.
            float lmax = -std::numeric_limits<float>::infinity();
            for (uint32_t i = lid; i < vocab; i += WG)
                lmax = sycl::fmax(lmax, float(logits[i]));
            lmax = sycl::reduce_over_group(grp, lmax, sycl::maximum<float>());

            // Pass 2: full-distribution partition function (temperature in).
            const float inv_temp = 1.0f / temperature;
            float z = 0.f;
            for (uint32_t i = lid; i < vocab; i += WG)
                z += sycl::native::exp((float(logits[i]) - lmax) * inv_temp);
            z = sycl::reduce_over_group(grp, z, sycl::plus<float>());
            const float invZ = 1.0f / z;

            // min-p in logit space: p/p_max >= min_p  <=>
            //   l >= lmax + T * ln(min_p).
            const float l_floor = (min_p > 0.f)
                ? lmax + temperature * sycl::log(min_p)
                : -std::numeric_limits<float>::infinity();

            // Per-lane local max over its slice (>= floor).
            auto rescan = [&](float& v, int32_t& vi) {
                v  = -std::numeric_limits<float>::infinity();
                vi = -1;
                for (uint32_t i = lid; i < vocab; i += WG) {
                    const float xv = float(logits[i]);
                    if (xv >= l_floor && xv > v) { v = xv; vi = int32_t(i); }
                }
            };
            float   lv;  int32_t li;
            rescan(lv, li);

            // Extraction rounds.
            uint32_t n_keep = 0;
            for (uint32_t kk = 0; kk < k_eff; ++kk) {
                // Inverted index in the low bits: max-reduce then prefers
                // the LOWEST index among exact logit ties (matches the CPU
                // golden / conventional first-wins scan order).
                const uint64_t my_key = (li >= 0)
                    ? ((uint64_t(fkey(lv)) << 32) | (0xFFFFFFFFu - uint32_t(li)))
                    : 0ull;
                const uint64_t win = sycl::reduce_over_group(
                    grp, my_key, sycl::maximum<uint64_t>());
                if (win == 0ull) break;                 // nothing left >= floor
                const int32_t win_i = int32_t(0xFFFFFFFFu - uint32_t(win & 0xFFFFFFFFu));

                if (lid == 0) {
                    const float wl = float(logits[win_i]);
                    const float p  = sycl::native::exp((wl - lmax) * inv_temp) * invZ;
                    keep_idx[n_keep] = win_i;
                    keep_p[n_keep]   = p;
                }
                // Owner tombstones the winner and rescans only its slice;
                // every other lane's cached local max is still valid.
                if (uint32_t(win_i) % WG == lid) {
                    logits[win_i] = sycl::half(-65504.0f);
                    rescan(lv, li);
                }
                n_keep += 1;
                sycl::group_barrier(grp);
                if (lid == 0) {
                    float cum = 0.f;
                    for (uint32_t i = 0; i < n_keep; ++i) cum += keep_p[i];
                    ctl[0] = (top_p < 1.0f && cum >= top_p) ? 0 : 1;
                }
                sycl::group_barrier(grp);
                if (ctl[0] == 0) break;
            }

            // Final: renormalize the kept set and draw (lane 0).
            if (lid == 0) {
                float zk = 0.f;
                for (uint32_t i = 0; i < n_keep; ++i) zk += keep_p[i];
                int32_t picked = (n_keep > 0) ? keep_idx[0] : 0;
                if (zk > 0.f) {
                    uint64_t state = rng_state ^ 0x9E3779B97F4A7C15ull;
                    (void)xorshift64(state);            // avalanche
                    const float r = to_unit_float(xorshift64(state)) * zk;
                    float c = 0.f;
                    for (uint32_t i = 0; i < n_keep; ++i) {
                        c += keep_p[i];
                        if (keep_p[i] > 0.f && c > r) { picked = keep_idx[i]; break; }
                    }
                }
                *out = picked;
            }
        });
    });
}

}  // namespace ie
