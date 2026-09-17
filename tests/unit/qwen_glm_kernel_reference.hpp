// Frozen pre-round kernel expressions, 2026-09-10. Independent of production dispatch.
#pragma once
#include <sycl/sycl.hpp>
#include <vector>
namespace frozen {
sycl::event rope_partial(sycl::queue& q,
                         const sycl::half* x, const int32_t* positions,
                         sycl::half* y,
                         uint32_t n_tokens, uint32_t n_heads,
                         uint32_t head_dim, uint32_t n_rotary,
                         float theta_base,
                         const std::vector<sycl::event>& deps) {
    return q.submit( [&](sycl::handler& h) {
        h.depends_on(deps);
        constexpr uint32_t WG = 64;
        // Grid: (n_tokens * n_heads, head_dim / 2 rotary pairs + non-rotary copies)
        // Simpler: one work-item per (token, head, dim).
        const uint64_t global_x = uint64_t(n_tokens) * uint64_t(n_heads);
        const uint64_t global_y = ((head_dim + WG - 1) / WG) * WG;

        h.parallel_for(sycl::nd_range<2>({global_x, global_y}, {1, WG}),
                       [=](sycl::nd_item<2> it) {
            const uint32_t th = uint32_t(it.get_global_id(0));
            const uint32_t d  = uint32_t(it.get_global_id(1));
            if (d >= head_dim) return;
            const uint32_t token = th / n_heads;
            const uint32_t head  = th % n_heads;
            const uint32_t base_idx = (token * n_heads + head) * head_dim;
            const sycl::half* xr = x + base_idx;
            sycl::half*       yr = y + base_idx;

            if (d >= n_rotary) {
                // Pass-through dim (the upper 192 of 256 for Qwen3.6 with factor 0.25).
                yr[d] = xr[d];
                return;
            }

            const uint32_t half = n_rotary / 2;
            // 2026-06-10 dense-nondeterminism fix: ONE work-item owns the
            // whole pair (r, r+half) — reads both inputs, writes both
            // outputs.  The previous one-item-per-dim layout had the
            // cos-side item (d) and sin-side item (d+half) each reading
            // BOTH x[r] and x[r+half] while writing one of them; with the
            // in-place calls every model path makes (x == y) that is a
            // cross-work-item read/write race.  At the crown shape
            // (n_rotary=64, half=32) both sides share one 64-wide WG and
            // the race never manifested; at the dense qwen3 shape
            // (n_rotary=128, half=64) the two sides land in DIFFERENT
            // work-groups and it fired constantly (run-to-run PPL spread
            // 19.0-19.4; bisect: docs/dense_nondeterminism_2026-06-10.md).
            // Items d in [half, n_rotary) are now no-ops; per-element
            // arithmetic expressions are unchanged.
            if (d >= half) return;
            const uint32_t r = d;
            const int32_t pos = positions[token];
            const float    inv_freq =
                sycl::native::exp(-float(2u * r) / float(n_rotary)
                                  * sycl::log(theta_base));
            const float    angle = float(pos) * inv_freq;
            const float    cs = sycl::cos(angle);
            const float    sn = sycl::sin(angle);
            const float    a  = float(xr[r]);
            const float    b  = float(xr[r + half]);
            yr[r]        = sycl::half(a * cs - b * sn);
            yr[r + half] = sycl::half(a * sn + b * cs);
        });
    });
}
sycl::event rope_imrope3(sycl::queue& q,
                         const sycl::half* x, const int32_t* positions,
                         sycl::half* y,
                         uint32_t n_tokens, uint32_t n_heads,
                         uint32_t head_dim, uint32_t n_rotary,
                         float theta_base,
                         const std::vector<sycl::event>& deps) {
    return q.submit( [&](sycl::handler& h) {
        h.depends_on(deps);
        constexpr uint32_t WG = 64;
        const uint64_t global_x = uint64_t(n_tokens) * uint64_t(n_heads);
        const uint64_t global_y = ((head_dim + WG - 1) / WG) * WG;
        h.parallel_for(sycl::nd_range<2>({global_x, global_y}, {1, WG}),
                       [=](sycl::nd_item<2> it) {
            const uint32_t th = uint32_t(it.get_global_id(0));
            const uint32_t d  = uint32_t(it.get_global_id(1));
            if (d >= head_dim) return;
            const uint32_t token = th / n_heads;
            const uint32_t head  = th % n_heads;
            const uint32_t base_idx = (token * n_heads + head) * head_dim;
            const sycl::half* xr = x + base_idx;
            sycl::half*       yr = y + base_idx;
            if (d >= n_rotary) { yr[d] = xr[d]; return; }
            const uint32_t half = n_rotary / 2;
            if (d >= half) return;   // pair (r, r+half) owned by item r (race note above)
            const uint32_t r = d;
            const int32_t pos = positions[size_t(r % 3u) * n_tokens + token];
            const float    inv_freq =
                sycl::native::exp(-float(2u * r) / float(n_rotary)
                                  * sycl::log(theta_base));
            const float    angle = float(pos) * inv_freq;
            const float    cs = sycl::cos(angle);
            const float    sn = sycl::sin(angle);
            const float    a  = float(xr[r]);
            const float    b  = float(xr[r + half]);
            yr[r]        = sycl::half(a * cs - b * sn);
            yr[r + half] = sycl::half(a * sn + b * cs);
        });
    });
}
sycl::event kda_gate(sycl::queue& q,
                     const float* pre, const float* dt_bias, const float* a,
                     float* g_out,
                     uint32_t n_rows, uint32_t n_heads, uint32_t head_dim,
                     float lower_bound,
                     const std::vector<sycl::event>& deps) {
    const uint32_t D = n_heads * head_dim;
    return q.submit( [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(uint64_t(n_rows) * D), [=](sycl::id<1> gid) {
            const uint64_t i  = gid[0];
            const uint32_t c  = uint32_t(i % D);
            const uint32_t hh = c / head_dim;
            const float x = -(pre[i] + dt_bias[c]) * a[hh];
            g_out[i] = lower_bound / (1.f + sycl::native::exp(-x));
        });
    });
}
sycl::event depthwise_conv1d_causal(sycl::queue& q,
                                    const sycl::half* x, const sycl::half* w,
                                    sycl::half* conv_state,
                                    sycl::half* y,
                                    uint32_t T, uint32_t channels, uint32_t kernel,
                                    const std::vector<sycl::event>& deps) {
    if (T == 0 || channels == 0 || kernel == 0) return {};

    auto compute_evt = q.submit( [&](sycl::handler& h) {
        h.depends_on(deps);
        constexpr uint32_t WG_C = 64;
        const uint64_t global_t = T;
        const uint64_t global_c = ((channels + WG_C - 1) / WG_C) * WG_C;
        h.parallel_for(sycl::nd_range<2>({global_t, global_c}, {1, WG_C}),
                       [=](sycl::nd_item<2> it) {
            const uint32_t t = uint32_t(it.get_global_id(0));
            const uint32_t c = uint32_t(it.get_global_id(1));
            if (c >= channels) return;
            float acc = 0.f;
            for (uint32_t k = 0; k < kernel; ++k) {
                const int32_t src_t = int32_t(t) - int32_t(k);
                float v = 0.f;
                if (src_t >= 0) {
                    v = float(x[uint64_t(src_t) * channels + c]);
                } else if (conv_state) {
                    // Past tokens stored in conv_state[(kernel-1) + src_t, c].
                    const int32_t past_idx = int32_t(kernel - 1) + src_t;  // 0..kernel-2
                    if (past_idx >= 0)
                        v = float(conv_state[uint64_t(past_idx) * channels + c]);
                }
                // GGUF conv weight has shape [kernel, channels] with kernel as
                // the leading (contiguous) dim — element (k, c) at offset k + c*kernel.
                // PyTorch/llama.cpp convention: W[0] multiplies the oldest tap
                // (t-(K-1)), W[K-1] multiplies the current tap (t). Since our
                // loop variable k indexes "lag from current" (src_t = t - k),
                // we need W[K-1-k] here.
                const uint32_t wk = (kernel - 1) - k;
                acc += v * float(w[uint64_t(wk) + uint64_t(c) * kernel]);
            }
            // Inline SiLU: silu(x) = x * sigmoid(x) = x / (1 + exp(-x)).
            // Saves a separate silu launch in the DeltaNet pipeline.
            const float silu = acc / (1.0f + sycl::native::exp(-acc));
            y[uint64_t(t) * channels + c] = sycl::half(silu);

            // v1.4 fusion: at T==1 each item owns channel c outright (its
            // state taps were read above), so the state shift-and-append
            // happens here and the separate writeback launch is skipped.
            // Ascending p reads slot p+1 before any overwrite of p+1.
            if (T == 1 && conv_state && kernel > 1) {
                for (uint32_t p = 0; p + 1 < kernel - 1; ++p)
                    conv_state[uint64_t(p) * channels + c] =
                        conv_state[uint64_t(p + 1) * channels + c];
                conv_state[uint64_t(kernel - 2) * channels + c] = x[c];
            }
        });
    });

    if (!conv_state || T == 1 || kernel == 1) return compute_evt;

    // A short chunk retains part of the old history. One item owns the
    // channel's entire shift, reading p+T before overwriting that slot.
    // Separate workgroups per history position would race with one another.
    if (T < kernel - 1) {
        return q.submit( [&](sycl::handler& h) {
            h.depends_on(compute_evt);
            h.parallel_for(sycl::range<1>(channels), [=](sycl::id<1> id) {
                const uint32_t c = uint32_t(id[0]);
                const uint32_t past = kernel - 1;
                for (uint32_t p = 0; p < past; ++p)
                    conv_state[uint64_t(p) * channels + c] = p + T < past
                        ? conv_state[uint64_t(p + T) * channels + c]
                        : x[uint64_t(p + T - past) * channels + c];
            });
        });
    }

    // Update conv_state with the last (kernel-1) tokens of x (after compute completes).
    return q.submit( [&](sycl::handler& h) {
        h.depends_on(compute_evt);
        const uint32_t past = kernel - 1;
        constexpr uint32_t WG_C = 64;
        const uint64_t global_c = ((channels + WG_C - 1) / WG_C) * WG_C;
        const uint64_t global_p = past;
        h.parallel_for(sycl::nd_range<2>({global_p, global_c}, {1, WG_C}),
                       [=](sycl::nd_item<2> it) {
            const uint32_t p = uint32_t(it.get_global_id(0));   // 0..past-1
            const uint32_t c = uint32_t(it.get_global_id(1));
            if (c >= channels) return;
            conv_state[uint64_t(p) * channels + c] =
                x[uint64_t(T - past + p) * channels + c];
        });
    });
}

}
