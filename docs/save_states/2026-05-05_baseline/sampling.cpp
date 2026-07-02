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

    // Phase 7 v1: serial sampling kernel — one work-item does the full pipeline.
    // Vocab=248k means each pass over logits is ~250 µs, dominated by reduce.
    // A WG-parallel version is in the Phase 9 backlog; this is simple and
    // correct. We use a workgroup of 1 thread but allocate scratch as device
    // memory for the sorted indices.
    return ie::ps(q, "topk_topp", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.single_task([=]() {
            // 1. Apply temperature, find max for numerical-safe softmax.
            float m = -std::numeric_limits<float>::infinity();
            const float inv_temp = 1.0f / temperature;
            for (uint32_t i = 0; i < vocab; ++i) {
                const float v = float(logits[i]) * inv_temp;
                logits[i] = sycl::half(v);
                if (v > m) m = v;
            }
            // 2. Compute exp(l - m) and sum.
            float Z = 0.f;
            for (uint32_t i = 0; i < vocab; ++i) {
                const float p = sycl::native::exp(float(logits[i]) - m);
                logits[i] = sycl::half(p);
                Z += p;
            }
            // 3. Normalize.
            const float invZ = 1.0f / Z;
            float p_max = 0.f;
            for (uint32_t i = 0; i < vocab; ++i) {
                const float p = float(logits[i]) * invZ;
                logits[i] = sycl::half(p);
                if (p > p_max) p_max = p;
            }
            // 4. Apply min-p: drop probs below `min_p * p_max`.
            if (min_p > 0.f) {
                const float thresh = min_p * p_max;
                for (uint32_t i = 0; i < vocab; ++i) {
                    if (float(logits[i]) < thresh) logits[i] = sycl::half(0.f);
                }
            }
            // 5. Top-k (skip if 0): in-place find the k-th highest, zero everything below.
            //    Implemented as `pass k times: find max, save, mark used`. O(k * vocab).
            //    For small k this is fine; for large k a sort would be faster.
            if (top_k > 0 && top_k < vocab) {
                float kth_thresh = -1.f;
                for (uint32_t kk = 0; kk < top_k; ++kk) {
                    float best = -1.f;
                    for (uint32_t i = 0; i < vocab; ++i) {
                        const float p = float(logits[i]);
                        if (p > best && p < kth_thresh + 1.f) {
                            // Clamp via "already considered" flag: use sentinel below.
                        }
                    }
                    // Simpler implementation: collect indices manually.
                    // (Re-doing this loop properly below.)
                    (void)best;
                }
                // Replace with a simple "find threshold" pass:
                // Collect probs into a fixed-size buffer? No — we do it differently:
                // sort by partial selection, but to avoid extra memory we just do
                // O(vocab * log(vocab)) selection by repeated max-and-zero. That's
                // O(top_k * vocab). For top_k=40 and vocab=248k = 10M ops — fast.
                for (uint32_t kk = 0; kk < top_k; ++kk) {
                    float best = -1.f;
                    int32_t best_i = -1;
                    for (uint32_t i = 0; i < vocab; ++i) {
                        const float p = float(logits[i]);
                        // Skip already-zeroed and already-selected (we mark with a
                        // sign bit hack: negative = "kept"). Use -p - 2 to mark.
                        if (p > best) { best = p; best_i = int32_t(i); }
                    }
                    if (best_i >= 0) {
                        // Mark as kept: set to -p - 2.
                        logits[best_i] = sycl::half(-best - 2.0f);
                    }
                }
                // Now restore kept items, zero others.
                for (uint32_t i = 0; i < vocab; ++i) {
                    const float p = float(logits[i]);
                    if (p < -1.f) {
                        logits[i] = sycl::half(-(p + 2.0f));
                    } else {
                        logits[i] = sycl::half(0.f);
                    }
                }
            }
            // 6. Top-p (nucleus): zero probs in the bottom (1-top_p) cumulative tail.
            //    Done by repeated max-extraction until cumulative sum ≥ top_p.
            if (top_p < 1.0f) {
                // Mark kept by negation again.
                float cum = 0.f;
                while (cum < top_p) {
                    float best = -1.f;
                    int32_t best_i = -1;
                    for (uint32_t i = 0; i < vocab; ++i) {
                        const float p = float(logits[i]);
                        if (p > best) { best = p; best_i = int32_t(i); }
                    }
                    if (best_i < 0 || best <= 0.f) break;
                    cum += best;
                    logits[best_i] = sycl::half(-best - 2.0f);
                }
                for (uint32_t i = 0; i < vocab; ++i) {
                    const float p = float(logits[i]);
                    if (p < -1.f) logits[i] = sycl::half(-(p + 2.0f));
                    else          logits[i] = sycl::half(0.f);
                }
            }
            // 7. Renormalize (important after top-k / top-p / min-p truncation).
            float Z2 = 0.f;
            for (uint32_t i = 0; i < vocab; ++i) Z2 += float(logits[i]);
            if (Z2 <= 0.f) {
                // Degenerate — fall back to argmax.
                float best = -std::numeric_limits<float>::infinity();
                int32_t best_i = 0;
                for (uint32_t i = 0; i < vocab; ++i) {
                    const float p = float(logits[i]);
                    if (p > best) { best = p; best_i = int32_t(i); }
                }
                *out = best_i;
                return;
            }
            const float invZ2 = 1.0f / Z2;
            // 8. Multinomial draw — inversion sampling. xorshift64 with a
            //    seed of 1..few can yield a top byte of 0; mix the seed to
            //    avoid pathological r=0. Use strict `>` so probability-0
            //    indices never get picked over the actual non-zero one when
            //    r happens to be 0.
            uint64_t state = rng_state ^ 0x9E3779B97F4A7C15ull;
            (void)xorshift64(state);                         // discard first to avalanche
            const float r = to_unit_float(xorshift64(state));
            float cum = 0.f;
            int32_t picked = -1;
            for (uint32_t i = 0; i < vocab; ++i) {
                const float p = float(logits[i]) * invZ2;
                cum += p;
                if (p > 0.f && cum > r) { picked = int32_t(i); break; }
            }
            if (picked < 0) {
                // Safety: scan for last non-zero index (shouldn't normally trigger).
                for (uint32_t i = vocab; i-- > 0;) {
                    if (float(logits[i]) > 0.f) { picked = int32_t(i); break; }
                }
                if (picked < 0) picked = int32_t(vocab) - 1;
            }
            *out = picked;
        });
    });
}

}  // namespace ie
