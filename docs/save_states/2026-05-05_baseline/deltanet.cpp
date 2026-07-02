// src/ops/deltanet.cpp — Gated DeltaNet primitives.
//
// Implements: gated_rms_norm, l2_norm_scale, compute_g_beta,
//             compute_g_beta_h16, deltanet_recurrence.
// Reference: research/05_deltanet_math.md §7-12.
//
// History note (2026-05-03): a 28-step bisect was run against a
// stochastic non-determinism observed in `deltanet_recurrence`'s state
// read on Xe2 BMG-G31.  All software-level fixes (atomics, fences,
// USM-host, DMA round-trip, fp16 clamping, internal kernel rewrites)
// failed.  Conclusion: HW-level pipeline non-determinism, software-
// irreducible from the SYCL layer.  Production paths chunk prefill
// externally at T=256 and do not exercise the bug.  Steps 15–26
// scaffolding has been removed; `docs/bisect_step25_26_summary.md`
// captures the full investigation history.

#include "ie/ops.hpp"

#include <sycl/sycl.hpp>
#include "ie/kernel_profiler.hpp"

// IE_DN_RECURRENCE_REWRITE — clean alternative kernel.
// When ON, kernel writes its result to a scratch buffer and a single
// q.memcpy(state, scratch) updates state[].  EU never stores to state[].
// Validated to preserve baseline math but does NOT fix the stochastic
// non-determinism — kept only as a structurally-cleaner kernel option.
#ifndef IE_DN_RECURRENCE_REWRITE
#define IE_DN_RECURRENCE_REWRITE 0
#endif

namespace ie {

// ===== gated_rms_norm =====
sycl::event gated_rms_norm(sycl::queue& q,
                           const float* x, const sycl::half* z, const sycl::half* weight,
                           sycl::half* y,
                           uint32_t n_rows, uint32_t hidden, float eps,
                           const std::vector<sycl::event>& deps) {
    constexpr int SG = 16;
    return ie::ps(q, "dn_gated_rms", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_rows) * SG, SG),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint32_t row = uint32_t(it.get_group(0));
            const uint32_t lid = uint32_t(it.get_local_id(0));
            auto sg = it.get_sub_group();
            const float* xr = x + row * hidden;

            float partial = 0.f;
            for (uint32_t i = lid; i < hidden; i += SG) {
                const float v = xr[i];
                partial += v * v;
            }
            float sum_sq = sycl::reduce_over_group(sg, partial, sycl::plus<float>());
            const float rsqrt = sycl::native::rsqrt(sum_sq / float(hidden) + eps);

            sycl::half* yr = y + row * hidden;
            for (uint32_t i = lid; i < hidden; i += SG) {
                const float xn = xr[i] * rsqrt;
                const float w  = float(weight[i]);
                const float gz = float(z[row * hidden + i]);
                const float silu = gz / (1.f + sycl::native::exp(-gz));
                yr[i] = sycl::half(w * xn * silu);
            }
        });
    });
}

// ===== l2_norm_scale =====
sycl::event l2_norm_scale(sycl::queue& q,
                          const float* x, float* y,
                          uint32_t n_rows, uint32_t head_dim, float scale, float eps,
                          const std::vector<sycl::event>& deps) {
    constexpr int SG = 16;
    return ie::ps(q, "dn_l2_norm", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_rows) * SG, SG),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint32_t row = uint32_t(it.get_group(0));
            const uint32_t lid = uint32_t(it.get_local_id(0));
            auto sg = it.get_sub_group();
            const float* xr = x + row * head_dim;
            float* yr = y + row * head_dim;

            float partial = 0.f;
            for (uint32_t i = lid; i < head_dim; i += SG) {
                partial += xr[i] * xr[i];
            }
            const float sum_sq = sycl::reduce_over_group(sg, partial, sycl::plus<float>());
            const float r = sycl::native::rsqrt(sum_sq + eps) * scale;
            for (uint32_t i = lid; i < head_dim; i += SG) yr[i] = xr[i] * r;
        });
    });
}

// ===== compute_g_beta =====
sycl::event compute_g_beta(sycl::queue& q,
                           const float* a, const float* b,
                           const float* A_log, const float* dt_bias,
                           float* g_out, float* beta_out,
                           uint32_t n_rows, uint32_t n_heads,
                           const std::vector<sycl::event>& deps) {
    return ie::ps(q, "dn_g_beta", [&](sycl::handler& hdl) {
        hdl.depends_on(deps);
        constexpr uint32_t WG = 64;
        const uint64_t total = uint64_t(n_rows) * n_heads;
        const uint64_t global = ((total + WG - 1) / WG) * WG;
        hdl.parallel_for(sycl::nd_range<1>(global, WG),
                         [=](sycl::nd_item<1> it) {
            const uint64_t i = it.get_global_id(0);
            if (i >= total) return;
            const uint32_t h = uint32_t(i % n_heads);
            const float a_h = a[i] + dt_bias[h];
            const float ax = sycl::fabs(a_h);
            const float sp = sycl::fmax(a_h, 0.f) + sycl::log1p(sycl::native::exp(-ax));
            g_out[i]    = A_log[h] * sp;
            beta_out[i] = 1.0f / (1.0f + sycl::native::exp(-b[i]));
        });
    });
}

// Fused fp16-input variant of compute_g_beta — saves two cast_fp16_to_fp32
// launches per DeltaNet layer on the decode hot path.
sycl::event compute_g_beta_h16(sycl::queue& q,
                               const sycl::half* a_h16, const sycl::half* b_h16,
                               const float* A_log, const float* dt_bias,
                               float* g_out, float* beta_out,
                               uint32_t n_rows, uint32_t n_heads,
                               const std::vector<sycl::event>& deps) {
    return ie::ps(q, "dn_g_beta_h16", [&](sycl::handler& hdl) {
        hdl.depends_on(deps);
        constexpr uint32_t WG = 64;
        const uint64_t total = uint64_t(n_rows) * n_heads;
        const uint64_t global = ((total + WG - 1) / WG) * WG;
        hdl.parallel_for(sycl::nd_range<1>(global, WG),
                         [=](sycl::nd_item<1> it) {
            const uint64_t i = it.get_global_id(0);
            if (i >= total) return;
            const uint32_t h = uint32_t(i % n_heads);
            const float a_h = float(a_h16[i]) + dt_bias[h];
            const float ax = sycl::fabs(a_h);
            const float sp = sycl::fmax(a_h, 0.f) + sycl::log1p(sycl::native::exp(-ax));
            g_out[i]    = A_log[h] * sp;
            beta_out[i] = 1.0f / (1.0f + sycl::native::exp(-float(b_h16[i])));
        });
    });
}

// ===== deltanet_recurrence =====
//
// One subgroup-of-128 work-group per (b, h).  Each lane (lid in [0, V_DIM))
// owns one column of the state matrix S[:, lid] (k_head_dim values, fp32).
// Loop over t=0..T-1 sequentially.  For each t: cooperatively load q_t, k_t,
// v_t, g_t, β_t into SLM / subgroup-broadcast scalars, then run the 5-step
// gated-delta recurrence.
//
// SLM: q_slm[K_DIM] + k_slm[K_DIM] + v_slm[V_DIM] + scalars[2] ≈ 2 KiB.
// Per-thread state: K_DIM (=128) fp32 in private memory ≈ 512 B / lane.
sycl::event deltanet_recurrence(sycl::queue& q,
                                const float* q_in, const float* k_in, const float* v_in,
                                const float* g_in, const float* beta_in,
                                float* state,
                                float* out,
                                uint32_t B, uint32_t T,
                                uint32_t n_v_heads, uint32_t k_head_dim, uint32_t v_head_dim,
                                const std::vector<sycl::event>& deps) {
    constexpr int K_DIM_MAX = 128;
    if (k_head_dim != K_DIM_MAX || v_head_dim != K_DIM_MAX) {
        // Phase 5 v1 supports only the Qwen3.6 shape (128/128).  Add
        // specializations here when other models land.
        sycl::event e;
        return e;
    }

#if IE_DN_RECURRENCE_REWRITE
    // Alternative path: kernel writes to a scratch buffer; a single
    // q.memcpy updates state[].  EU never stores to state[].
    {
        const size_t state_bytes =
            uint64_t(B) * n_v_heads * K_DIM_MAX * K_DIM_MAX * sizeof(float);
        static float* scratch_buf  = nullptr;
        static size_t scratch_size = 0;
        if (scratch_size < state_bytes) {
            if (scratch_buf) sycl::free(scratch_buf, q);
            scratch_buf = sycl::malloc_device<float>(
                state_bytes / sizeof(float), q);
            scratch_size = state_bytes;
        }
        float* scratch_dev = scratch_buf;

        sycl::event kev = ie::ps(q, "dn_recurrence_rewrite",
            [&](sycl::handler& h) {
                h.depends_on(deps);
                sycl::local_accessor<float, 1> q_slm(K_DIM_MAX, h);
                sycl::local_accessor<float, 1> k_slm(K_DIM_MAX, h);
                sycl::local_accessor<float, 1> v_slm(K_DIM_MAX, h);
                sycl::local_accessor<float, 1> sc_slm(2, h);

                const uint32_t WG_ITEMS = K_DIM_MAX;
                h.parallel_for(
                    sycl::nd_range<2>({uint64_t(B) * n_v_heads, WG_ITEMS},
                                      {1,                       WG_ITEMS}),
                    [=](sycl::nd_item<2> it)
                    [[sycl::reqd_sub_group_size(16),
                      sycl::reqd_work_group_size(1, K_DIM_MAX)]] {
                        const uint32_t bh = uint32_t(it.get_group(0));
                        const uint32_t b  = bh / n_v_heads;
                        const uint32_t hh = bh % n_v_heads;
                        const uint32_t vv = uint32_t(it.get_local_id(1));

                        const uint64_t state_base =
                            (uint64_t(b) * n_v_heads + hh) *
                            K_DIM_MAX * K_DIM_MAX;

                        float S_col[K_DIM_MAX];
                        #pragma unroll
                        for (int kk = 0; kk < K_DIM_MAX; ++kk) {
                            S_col[kk] = state[state_base +
                                              uint64_t(kk) * K_DIM_MAX + vv];
                        }
                        sycl::group_barrier(it.get_group());

                        for (uint32_t t = 0; t < T; ++t) {
                            const uint64_t qkv_row =
                                (uint64_t(b) * T + t) * n_v_heads + hh;
                            q_slm[vv] = q_in[qkv_row * K_DIM_MAX + vv];
                            k_slm[vv] = k_in[qkv_row * K_DIM_MAX + vv];
                            v_slm[vv] = v_in[qkv_row * K_DIM_MAX + vv];
                            if (vv == 0) {
                                sc_slm[0] = sycl::native::exp(g_in[qkv_row]);
                                sc_slm[1] = beta_in[qkv_row];
                            }
                            sycl::group_barrier(it.get_group());

                            float k_priv[K_DIM_MAX];
                            float q_priv[K_DIM_MAX];
                            #pragma unroll
                            for (int kk = 0; kk < K_DIM_MAX; ++kk) {
                                k_priv[kk] = k_slm[kk];
                                q_priv[kk] = q_slm[kk];
                            }
                            const float alpha  = sc_slm[0];
                            const float beta_t = sc_slm[1];
                            const float v_t    = v_slm[vv];
                            sycl::group_barrier(it.get_group());

                            #pragma unroll
                            for (int kk = 0; kk < K_DIM_MAX; ++kk)
                                S_col[kk] *= alpha;

                            float kv_mem = 0.f;
                            #pragma unroll
                            for (int kk = 0; kk < K_DIM_MAX; ++kk)
                                kv_mem += S_col[kk] * k_priv[kk];

                            const float delta = (v_t - kv_mem) * beta_t;

                            #pragma unroll
                            for (int kk = 0; kk < K_DIM_MAX; ++kk)
                                S_col[kk] += k_priv[kk] * delta;

                            float out_v = 0.f;
                            #pragma unroll
                            for (int kk = 0; kk < K_DIM_MAX; ++kk)
                                out_v += S_col[kk] * q_priv[kk];

                            out[(uint64_t(b) * T + t) * n_v_heads *
                                K_DIM_MAX + hh * K_DIM_MAX + vv] = out_v;

                            sycl::group_barrier(it.get_group());
                        }

                        // Writeback to scratch (not state).
                        #pragma unroll
                        for (int kk = 0; kk < K_DIM_MAX; ++kk) {
                            scratch_dev[state_base +
                                        uint64_t(kk) * K_DIM_MAX + vv] =
                                S_col[kk];
                        }
                    });
            });
        return q.memcpy(state, scratch_dev, state_bytes, kev);
    }
#else
    return ie::ps(q, "dn_recurrence", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<float, 1> q_slm(K_DIM_MAX, h);
        sycl::local_accessor<float, 1> k_slm(K_DIM_MAX, h);
        sycl::local_accessor<float, 1> v_slm(K_DIM_MAX, h);
        sycl::local_accessor<float, 1> sc_slm(2,        h);

        const uint32_t WG_ITEMS = K_DIM_MAX;
        h.parallel_for(sycl::nd_range<2>({uint64_t(B) * n_v_heads, WG_ITEMS}, {1, WG_ITEMS}),
                       [=](sycl::nd_item<2> it) {
            const uint32_t bh   = uint32_t(it.get_group(0));
            const uint32_t b    = bh / n_v_heads;
            const uint32_t hh   = bh % n_v_heads;
            const uint32_t lid  = uint32_t(it.get_local_id(1));   // = vv
            const uint32_t vv   = lid;

            float S_col[K_DIM_MAX];
            const uint64_t state_base = (uint64_t(b) * n_v_heads + hh) * K_DIM_MAX * K_DIM_MAX;
            #pragma unroll
            for (int kk = 0; kk < K_DIM_MAX; ++kk) {
                S_col[kk] = state[state_base + uint64_t(kk) * K_DIM_MAX + vv];
            }

            for (uint32_t t = 0; t < T; ++t) {
                const uint64_t qkv_row = (uint64_t(b) * T + t) * n_v_heads + hh;
                q_slm[vv] = q_in[qkv_row * K_DIM_MAX + vv];
                k_slm[vv] = k_in[qkv_row * K_DIM_MAX + vv];
                v_slm[vv] = v_in[qkv_row * K_DIM_MAX + vv];
                if (vv == 0) {
                    sc_slm[0] = sycl::native::exp(g_in[qkv_row]);
                    sc_slm[1] = beta_in[qkv_row];
                }
                sycl::group_barrier(it.get_group());

                const float alpha  = sc_slm[0];
                const float beta_t = sc_slm[1];
                const float v_t    = v_slm[vv];

                // Step 9a: decay
                #pragma unroll
                for (int kk = 0; kk < K_DIM_MAX; ++kk) S_col[kk] *= alpha;

                // Step 9b: kv_mem = sum_kk S_col[kk] * k_t[kk]
                float kv_mem = 0.f;
                #pragma unroll
                for (int kk = 0; kk < K_DIM_MAX; ++kk) kv_mem += S_col[kk] * k_slm[kk];

                // Step 9c: δ = (v − kv_mem) · β
                const float delta = (v_t - kv_mem) * beta_t;

                // Step 9d: S_col[kk] += k_t[kk] · δ
                #pragma unroll
                for (int kk = 0; kk < K_DIM_MAX; ++kk) S_col[kk] += k_slm[kk] * delta;

                // Step 9e: out[vv] = sum_kk S_col[kk] * q_t[kk]
                float out_v = 0.f;
                #pragma unroll
                for (int kk = 0; kk < K_DIM_MAX; ++kk) out_v += S_col[kk] * q_slm[kk];

                out[(uint64_t(b) * T + t) * n_v_heads * K_DIM_MAX + hh * K_DIM_MAX + vv] = out_v;

                sycl::group_barrier(it.get_group());
            }

            // Write back S
            #pragma unroll
            for (int kk = 0; kk < K_DIM_MAX; ++kk) {
                state[state_base + uint64_t(kk) * K_DIM_MAX + vv] = S_col[kk];
            }
        });
    });
#endif  // IE_DN_RECURRENCE_REWRITE
}

}  // namespace ie
