// src/ops/moe.cpp — MoE router + small utility ops.

#include "ie/ops.hpp"

#include <sycl/sycl.hpp>

#include <limits>
#include <mutex>
#include <unordered_map>
#include "ie/kernel_profiler.hpp"

namespace ie {

// One workgroup per token: 256 work-items, each owning one expert's logit.
// Pipeline:
//   (a) lane e computes logits[e] = Σ_h W_gate[e,h] * x[h]   (fp32)
//   (b) WG-wide reduce-max for numerical safety
//   (c) exp(l - m), reduce-sum, divide → fp32 probs
//   (d) lane 0 selects top-k (serial loop over E), renormalizes, sorts asc by
//       expert index (bit-exact match to HF's `torch.where` iteration order)
//   (e) lanes 0..k-1 write the output
sycl::event moe_router(sycl::queue& q,
                       const sycl::half* x, const float* W_gate,
                       int32_t* topk_idx, sycl::half* topk_w,
                       uint32_t n_tokens, uint32_t hidden,
                       uint32_t n_experts, uint32_t k,
                       const std::vector<sycl::event>& deps) {
    if (n_experts != 256 || k != 8) {
        // Phase 6 v1: only the Qwen3.6 router shape.
        return ie::ps(q, "moe_router", [&](sycl::handler& h) { h.depends_on(deps); h.single_task([](){}); });
    }
    constexpr uint32_t WG = 256;

    // v1.4 (2026-06-09): at T==1 the single-WG kernel is occupancy-starved —
    // one Xe core streams the 2 MB fp32 router matrix at ~30 GB/s = 66 µs;
    // × 40 layers = 2.6 ms/token, the #1 decode kernel by census.  Two-stage
    // split: stage 1 puts one WG (64 lanes) on EACH expert's dot (full-GPU
    // bandwidth), stage 2 reruns the existing softmax+top-8 on the cached
    // logits.  T>1 keeps the fused single kernel (T WGs saturate the GPU).
    if (n_tokens == 1) {
        // Per-DEVICE logits scratch.  A single `static float*` here is a cross-card
        // landmine on a layer-split / TP fleet: it is allocated in whichever device's
        // context makes the FIRST T==1 call (card 0 on the crown split), and a later
        // card-1 kernel that dereferences that pointer page-faults the device →
        // UR_RESULT_ERROR_DEVICE_LOST (decode hangs on card 1's first MoE layer).
        // Key the scratch by device so each card owns its own 256-float buffer.
        // Single-GPU keeps exactly one entry → behaviourally identical.  Mirrors the
        // per-device oneDNN ctx fix (gemm_onednn.cpp:ctx_for).
        static std::unordered_map<sycl::device, float*> logits_bufs;
        static std::mutex logits_mu;
        float* logits_buf;
        {
            std::lock_guard<std::mutex> lk(logits_mu);
            float*& slot = logits_bufs[q.get_device()];
            if (!slot) slot = sycl::malloc_device<float>(256, q);
            logits_buf = slot;
        }
        ie::ps(q, "moe_router_dot", [&](sycl::handler& h) {
            h.depends_on(deps);
            constexpr uint32_t LWG = 64;
            h.parallel_for(sycl::nd_range<1>(256 * LWG, LWG),
                           [=, lb = logits_buf](sycl::nd_item<1> it)
                               [[sycl::reqd_sub_group_size(16)]] {
                const uint32_t e   = uint32_t(it.get_group(0));
                const uint32_t lid = uint32_t(it.get_local_id(0));
                const auto* wv4 = reinterpret_cast<const sycl::vec<float, 4>*>(
                    W_gate + uint64_t(e) * hidden);
                const auto* xv4 = reinterpret_cast<const sycl::vec<sycl::half, 4>*>(x);
                float p = 0.f;
                for (uint32_t hh = lid; hh < hidden / 4; hh += LWG) {
                    const auto w4 = wv4[hh];
                    const auto x4 = xv4[hh];
                    p += w4[0] * float(x4[0]) + w4[1] * float(x4[1]) +
                         w4[2] * float(x4[2]) + w4[3] * float(x4[3]);
                }
                p = sycl::reduce_over_group(it.get_group(), p, sycl::plus<float>());
                if (lid == 0) lb[e] = p;
            });
        });
        return ie::ps(q, "moe_router_top8", [&](sycl::handler& h) {
            sycl::local_accessor<int32_t, 1> idx_slm(8, h);
            sycl::local_accessor<float, 1>   w_slm(8, h);
            h.parallel_for(sycl::nd_range<1>(WG, WG),
                           [=, lb = logits_buf](sycl::nd_item<1> it) {
                const uint32_t lid = uint32_t(it.get_local_id(0));
                const uint32_t e   = lid;
                auto grp = it.get_group();
                const float logit = lb[e];

                float m = sycl::reduce_over_group(grp, logit, sycl::maximum<float>());
                const float ev = sycl::native::exp(logit - m);
                const float s  = sycl::reduce_over_group(grp, ev, sycl::plus<float>());
                const float p  = ev / s;

                bool live = true;
                for (uint32_t kk = 0; kk < 8; ++kk) {
                    const uint64_t my_key = live
                        ? ((uint64_t(sycl::bit_cast<uint32_t>(p)) << 32) | e) : 0ull;
                    const uint64_t win = sycl::reduce_over_group(
                        grp, my_key, sycl::maximum<uint64_t>());
                    const uint32_t win_e = uint32_t(win & 0xFFFFFFFFu);
                    if (e == win_e) {
                        live = false;
                        idx_slm[kk] = int32_t(e);
                        w_slm[kk]   = sycl::bit_cast<float>(uint32_t(win >> 32));
                    }
                }
                sycl::group_barrier(grp);
                if (lid == 0) {
                    float ren = 0.f;
                    for (uint32_t kk = 0; kk < 8; ++kk) ren += w_slm[kk];
                    const float inv = 1.0f / ren;
                    for (uint32_t kk = 0; kk < 8; ++kk) w_slm[kk] *= inv;
                    for (uint32_t i = 0; i + 1 < 8; ++i)
                        for (uint32_t j = 0; j + 1 + i < 8; ++j)
                            if (idx_slm[j] > idx_slm[j + 1]) {
                                int32_t ti = idx_slm[j]; idx_slm[j] = idx_slm[j + 1]; idx_slm[j + 1] = ti;
                                float   tw = w_slm[j];   w_slm[j]   = w_slm[j + 1];   w_slm[j + 1]   = tw;
                            }
                }
                sycl::group_barrier(grp);
                if (lid < 8) {
                    topk_idx[lid] = idx_slm[lid];
                    topk_w[lid]   = sycl::half(w_slm[lid]);
                }
            });
        });
    }

    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<int32_t, 1> idx_slm(8, h);
        sycl::local_accessor<float, 1>   w_slm(8, h);

        h.parallel_for(sycl::nd_range<1>(uint64_t(n_tokens) * WG, WG),
                       [=](sycl::nd_item<1> it) {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t token = uint32_t(it.get_group(0));
            const uint32_t e     = lid;

            // (a) per-expert logit in fp32.
            // v1.2-B (2026-06-09): vectorized — vec<float,4> weight loads ×
            // vec<half,4> activation loads (both rows 16 B / 8 B aligned:
            // strides 8192 B / 4096 B).  4 partial sums break the serial FMA
            // dependency chain; summation order changes → PPL-gated.
            const sycl::half* xt = x + uint64_t(token) * hidden;
            const float* wr = W_gate + uint64_t(e) * hidden;
            const auto* wv4 = reinterpret_cast<const sycl::vec<float, 4>*>(wr);
            const auto* xv4 = reinterpret_cast<const sycl::vec<sycl::half, 4>*>(xt);
            float l0 = 0.f, l1 = 0.f, l2 = 0.f, l3 = 0.f;
            for (uint32_t hh = 0; hh < hidden / 4; ++hh) {
                const auto w4 = wv4[hh];
                const auto x4 = xv4[hh];
                l0 += w4[0] * float(x4[0]);
                l1 += w4[1] * float(x4[1]);
                l2 += w4[2] * float(x4[2]);
                l3 += w4[3] * float(x4[3]);
            }
            const float logit = (l0 + l1) + (l2 + l3);

            // (b) WG-wide max
            float m = sycl::reduce_over_group(it.get_group(), logit, sycl::maximum<float>());

            // (c) softmax in fp32
            const float ev = sycl::native::exp(logit - m);
            const float s  = sycl::reduce_over_group(it.get_group(), ev, sycl::plus<float>());
            const float p  = ev / s;

            // (d) top-k via 8 WG-wide max-reduces on packed (prob, idx) keys
            // (positive-float bits are monotonic), then renorm + id-sort on
            // lane 0 over just the 8 winners.
            // v1.2-B: replaces a serial 2048-iteration scan on lane 0 with
            // 255 lanes idle.  Tie-break differs from the old scan (highest
            // expert id wins an exact-fp tie instead of lowest) — benign,
            // PPL-gated.
            bool live = true;
            for (uint32_t kk = 0; kk < 8; ++kk) {
                const uint64_t my_key = live
                    ? ((uint64_t(sycl::bit_cast<uint32_t>(p)) << 32) | e)
                    : 0ull;
                const uint64_t win = sycl::reduce_over_group(
                    it.get_group(), my_key, sycl::maximum<uint64_t>());
                const uint32_t win_e = uint32_t(win & 0xFFFFFFFFu);
                if (e == win_e) {
                    live = false;
                    idx_slm[kk] = int32_t(e);
                    w_slm[kk]   = sycl::bit_cast<float>(uint32_t(win >> 32));
                }
            }
            sycl::group_barrier(it.get_group());

            if (lid == 0) {
                float ren = 0.f;
                for (uint32_t kk = 0; kk < 8; ++kk) ren += w_slm[kk];
                const float inv = 1.0f / ren;
                for (uint32_t kk = 0; kk < 8; ++kk) w_slm[kk] *= inv;
                // Sort ascending by expert id (bubble sort — k=8, trivial).
                for (uint32_t i = 0; i + 1 < 8; ++i) {
                    for (uint32_t j = 0; j + 1 + i < 8; ++j) {
                        if (idx_slm[j] > idx_slm[j + 1]) {
                            int32_t ti = idx_slm[j]; idx_slm[j] = idx_slm[j + 1]; idx_slm[j + 1] = ti;
                            float   tw = w_slm[j];   w_slm[j]   = w_slm[j + 1];   w_slm[j + 1]   = tw;
                        }
                    }
                }
            }
            sycl::group_barrier(it.get_group());

            // (e) write outputs
            if (lid < 8) {
                topk_idx[uint64_t(token) * 8 + lid] = idx_slm[lid];
                topk_w  [uint64_t(token) * 8 + lid] = sycl::half(w_slm[lid]);
            }
        });
    });
}

sycl::event shared_expert_gate(sycl::queue& q,
                               const sycl::half* x, const float* W_shg,
                               sycl::half* gate_out,
                               uint32_t n_tokens, uint32_t hidden,
                               const std::vector<sycl::event>& deps) {
    constexpr int SG = 16;
    return ie::ps(q, "shared_exp_gate", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_tokens) * SG, SG),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint32_t token = uint32_t(it.get_group(0));
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            auto sg = it.get_sub_group();
            const sycl::half* xt = x + uint64_t(token) * hidden;
            float partial = 0.f;
            for (uint32_t i = lid; i < hidden; i += SG) {
                partial += float(xt[i]) * W_shg[i];
            }
            const float dot = sycl::reduce_over_group(sg, partial, sycl::plus<float>());
            if (lid == 0) {
                gate_out[token] = sycl::half(1.0f / (1.0f + sycl::native::exp(-dot)));
            }
        });
    });
}

sycl::event scaled_add(sycl::queue& q,
                       const sycl::half* x, sycl::half scale,
                       sycl::half* y, uint64_t n,
                       const std::vector<sycl::event>& deps) {
    return ie::ps(q, "scaled_add", [&](sycl::handler& h) {
        h.depends_on(deps);
        constexpr uint64_t WG = 256;
        const uint64_t global = ((n + WG - 1) / WG) * WG;
        const float s = float(scale);
        h.parallel_for(sycl::nd_range<1>(global, WG),
                       [=](sycl::nd_item<1> it) {
            const uint64_t i = it.get_global_id(0);
            if (i >= n) return;
            y[i] = sycl::half(float(y[i]) + float(x[i]) * s);
        });
    });
}

sycl::event moe_gather_rows(sycl::queue& q,
                            const sycl::half* x,
                            const int32_t* indices,
                            sycl::half* y_packed,
                            uint32_t n_tok, uint32_t hidden,
                            const std::vector<sycl::event>& deps) {
    return ie::ps(q, "moe_gather", [&](sycl::handler& h) {
        h.depends_on(deps);
        constexpr uint32_t WG = 256;
        const uint64_t total  = uint64_t(n_tok) * hidden;
        const uint64_t global = ((total + WG - 1) / WG) * WG;
        h.parallel_for(sycl::nd_range<1>(global, WG),
                       [=](sycl::nd_item<1> it) {
            const uint64_t i = it.get_global_id(0);
            if (i >= total) return;
            const uint32_t row = uint32_t(i / hidden);
            const uint32_t col = uint32_t(i % hidden);
            y_packed[i] = x[uint64_t(indices[row]) * hidden + col];
        });
    });
}

sycl::event moe_scatter_add(sycl::queue& q,
                            const sycl::half* x_packed,
                            const int32_t* indices,
                            const sycl::half* weights,
                            sycl::half* y,
                            uint32_t n_tok, uint32_t hidden,
                            const std::vector<sycl::event>& deps) {
    return ie::ps(q, "moe_scatter", [&](sycl::handler& h) {
        h.depends_on(deps);
        constexpr uint32_t WG = 256;
        const uint64_t total  = uint64_t(n_tok) * hidden;
        const uint64_t global = ((total + WG - 1) / WG) * WG;
        h.parallel_for(sycl::nd_range<1>(global, WG),
                       [=](sycl::nd_item<1> it) {
            const uint64_t i = it.get_global_id(0);
            if (i >= total) return;
            const uint32_t row = uint32_t(i / hidden);
            const uint32_t col = uint32_t(i % hidden);
            const float w = float(weights[row]);
            const uint64_t y_off = uint64_t(indices[row]) * hidden + col;
            // Atomic-free: each token-row of `y` is written by at most one
            // expert at a time (caller enforces via per-expert serialization),
            // and the multiple expert contributions are added at the
            // per-expert call (each scatter_add is one expert's contribution).
            // For the accumulation across the K_top experts to be race-free
            // we rely on the caller running scatter_adds serially per layer.
            y[y_off] = sycl::half(float(y[y_off]) + float(x_packed[i]) * w);
        });
    });
}

sycl::event scaled_add_dev_scalar(sycl::queue& q,
                                  const sycl::half* x, const sycl::half* scale_dev,
                                  sycl::half* y, uint64_t n,
                                  const std::vector<sycl::event>& deps) {
    return ie::ps(q, "scaled_add_scl", [&](sycl::handler& h) {
        h.depends_on(deps);
        constexpr uint64_t WG = 256;
        const uint64_t global = ((n + WG - 1) / WG) * WG;
        h.parallel_for(sycl::nd_range<1>(global, WG),
                       [=](sycl::nd_item<1> it) {
            const uint64_t i = it.get_global_id(0);
            if (i >= n) return;
            const float s = float(scale_dev[0]);
            y[i] = sycl::half(float(y[i]) + float(x[i]) * s);
        });
    });
}

// Per-token row scaled add (2026-05-05): y[t, h] += scale_per_tok[t] * x[t, h].
// Used by the shared-expert prefill path to replace its T-iteration host-
// memcpy + per-token scaled_add loop with one device-side launch.
sycl::event scaled_add_per_token_row(sycl::queue& q,
                                     const sycl::half* x,
                                     const sycl::half* scale_per_tok,
                                     sycl::half* y,
                                     uint32_t T, uint32_t H,
                                     const std::vector<sycl::event>& deps) {
    return ie::ps(q, "scaled_add_pt", [&](sycl::handler& h) {
        h.depends_on(deps);
        constexpr uint64_t WG = 256;
        const uint64_t total = uint64_t(T) * uint64_t(H);
        const uint64_t global = ((total + WG - 1) / WG) * WG;
        h.parallel_for(sycl::nd_range<1>(global, WG),
                       [=](sycl::nd_item<1> it) {
            const uint64_t i = it.get_global_id(0);
            if (i >= total) return;
            const uint32_t t = uint32_t(i / uint64_t(H));
            const float s = float(scale_per_tok[t]);
            y[i] = sycl::half(float(y[i]) + float(x[i]) * s);
        });
    });
}

// ---------------------------------------------------------------------------
// moe_topk_from_logits — generalized device softmax + top-K (stage 2 of
// moe_router, arbitrary E≤1024, arbitrary K) reading logits ALREADY on device.
// Matches the host route_from_logits algorithm bit-for-algorithm (full fp32
// softmax → bit-packed top-K argmax → renorm by top-K sum → ascending index
// sort). Used by the qwen3moe on-device decode router to kill the per-layer
// host round-trip. One WG = E lanes.
// ---------------------------------------------------------------------------
sycl::event moe_topk_from_logits(sycl::queue& q, const sycl::half* logits,
                                 int32_t* topk_idx, sycl::half* topk_w,
                                 uint32_t E, uint32_t K,
                                 const std::vector<sycl::event>& deps) {
    const uint32_t WG = E;   // one lane per expert
    return ie::ps(q, "moe_topk_from_logits", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<int32_t, 1> idx_slm(K, h);
        sycl::local_accessor<float, 1>   w_slm(K, h);
        h.parallel_for(sycl::nd_range<1>(WG, WG), [=](sycl::nd_item<1> it) {
            const uint32_t e = uint32_t(it.get_local_id(0));
            auto grp = it.get_group();
            const float logit = float(logits[e]);
            const float m  = sycl::reduce_over_group(grp, logit, sycl::maximum<float>());
            const float ev = sycl::native::exp(logit - m);
            const float s  = sycl::reduce_over_group(grp, ev, sycl::plus<float>());
            const float p  = ev / s;
            bool live = true;
            for (uint32_t kk = 0; kk < K; ++kk) {
                const uint64_t my_key = live
                    ? ((uint64_t(sycl::bit_cast<uint32_t>(p)) << 32) | e) : 0ull;
                const uint64_t win = sycl::reduce_over_group(grp, my_key, sycl::maximum<uint64_t>());
                const uint32_t win_e = uint32_t(win & 0xFFFFFFFFu);
                if (e == win_e) {
                    live = false;
                    idx_slm[kk] = int32_t(e);
                    w_slm[kk]   = sycl::bit_cast<float>(uint32_t(win >> 32));
                }
            }
            sycl::group_barrier(grp);
            if (e == 0) {
                float ren = 0.f;
                for (uint32_t kk = 0; kk < K; ++kk) ren += w_slm[kk];
                const float inv = 1.0f / ren;
                for (uint32_t kk = 0; kk < K; ++kk) w_slm[kk] *= inv;
                for (uint32_t i = 0; i + 1 < K; ++i)
                    for (uint32_t j = 0; j + 1 + i < K; ++j)
                        if (idx_slm[j] > idx_slm[j + 1]) {
                            int32_t ti = idx_slm[j]; idx_slm[j] = idx_slm[j + 1]; idx_slm[j + 1] = ti;
                            float   tw = w_slm[j];   w_slm[j]   = w_slm[j + 1];   w_slm[j + 1]   = tw;
                        }
            }
            sycl::group_barrier(grp);
            if (e < K) { topk_idx[e] = idx_slm[e]; topk_w[e] = sycl::half(w_slm[e]); }
        });
    });
}

// ---------------------------------------------------------------------------
// moe_build_pack_decode — build the MoePacking arrays ON DEVICE for the T==1
// decode case from a top-K route (topk_idx ascending, topk_w). For one token
// the expert-sorted packing is trivial: each of the K distinct experts gets one
// row, in ascending-expert (== topk) order. expert_offsets[e] = #{topk < e};
// sorted_idx = all token 0; tk_to_packed = identity; weights_packed = topk_w.
// Single work-item (E+K tiny serial work, ~ns) — kept on device so the layer
// loop never stalls on the host.
// ---------------------------------------------------------------------------
sycl::event moe_build_pack_decode(sycl::queue& q,
                                  const int32_t* topk_idx, const sycl::half* topk_w,
                                  uint32_t* expert_offsets, int32_t* sorted_idx,
                                  int32_t* tk_to_packed, sycl::half* weights_packed,
                                  uint32_t E, uint32_t K,
                                  const std::vector<sycl::event>& deps) {
    return ie::ps(q, "moe_build_pack_decode", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.single_task([=]() {
            uint32_t off = 0;
            for (uint32_t e = 0; e <= E; ++e) {
                expert_offsets[e] = off;
                if (e < E)
                    for (uint32_t kk = 0; kk < K; ++kk)
                        if (uint32_t(topk_idx[kk]) == e) ++off;
            }
            for (uint32_t kk = 0; kk < K; ++kk) {
                sorted_idx[kk]     = 0;
                tk_to_packed[kk]   = int32_t(kk);
                weights_packed[kk] = topk_w[kk];
            }
        });
    });
}

}  // namespace ie
