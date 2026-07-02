// src/ops/moe.cpp — MoE router + small utility ops.

#include "ie/ops.hpp"

#include <sycl/sycl.hpp>

#include <limits>
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
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<float, 1>   probs(WG, h);
        sycl::local_accessor<int32_t, 1> idx_slm(8, h);
        sycl::local_accessor<float, 1>   w_slm(8, h);

        h.parallel_for(sycl::nd_range<1>(uint64_t(n_tokens) * WG, WG),
                       [=](sycl::nd_item<1> it) {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t token = uint32_t(it.get_group(0));
            const uint32_t e     = lid;

            // (a) per-expert logit in fp32
            const sycl::half* xt = x + uint64_t(token) * hidden;
            const float* wr = W_gate + uint64_t(e) * hidden;
            float logit = 0.f;
            for (uint32_t hh = 0; hh < hidden; ++hh) {
                logit += wr[hh] * float(xt[hh]);
            }

            // (b) WG-wide max
            float m = sycl::reduce_over_group(it.get_group(), logit, sycl::maximum<float>());

            // (c) softmax in fp32
            const float ev = sycl::native::exp(logit - m);
            const float s  = sycl::reduce_over_group(it.get_group(), ev, sycl::plus<float>());
            const float p  = ev / s;
            probs[e] = p;
            sycl::group_barrier(it.get_group());

            // (d) top-k + renorm + sort, serial on lane 0.
            if (lid == 0) {
                for (uint32_t kk = 0; kk < 8; ++kk) {
                    float best = -1.f;
                    int32_t best_idx = -1;
                    for (uint32_t ee = 0; ee < 256; ++ee) {
                        const float pe = probs[ee];
                        if (pe > best) { best = pe; best_idx = int32_t(ee); }
                    }
                    idx_slm[kk] = best_idx;
                    w_slm[kk]   = best;
                    if (best_idx >= 0) probs[best_idx] = -2.f;
                }
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

}  // namespace ie
