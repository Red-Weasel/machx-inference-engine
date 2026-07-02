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
#include "ie/quant_blocks.hpp"

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

// v1.5-D2 (2026-06-10): gated_rms_norm + Q8_1 emission — feeds the int-dot
// ssm_out GEMV.  Each per-head row (hidden=128) is exactly 4 q8 blocks.
// Quantizes the rounded fp16 outputs (numerics = standalone quantize chain).
sycl::event gated_rms_norm_q8(sycl::queue& q,
                              const float* x, const sycl::half* z,
                              const sycl::half* weight,
                              sycl::half* y, void* q8_out,
                              uint32_t n_rows, uint32_t hidden, float eps,
                              const std::vector<sycl::event>& deps) {
    constexpr int SG = 16;
    auto* out8 = static_cast<block_q8_1x*>(q8_out);
    return ie::ps(q, "dn_gated_rms_q8", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<sycl::half, 1> vbuf(hidden, h);
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
                const sycl::half hv = sycl::half(w * xn * silu);
                yr[i]   = hv;
                vbuf[i] = hv;
            }
            sycl::group_barrier(it.get_group());

            const uint32_t n_blocks = hidden / 32;
            for (uint32_t j = lid; j < n_blocks; j += SG) {
                block_q8_1x* ob = out8 + uint64_t(row) * n_blocks + j;
                float amax = 0.f;
                for (uint32_t i = 0; i < 32; ++i)
                    amax = sycl::fmax(amax, sycl::fabs(float(vbuf[j * 32 + i])));
                const float d   = amax / 127.0f;
                const float inv = (amax > 0.f) ? 127.0f / amax : 0.f;
                int32_t qsum = 0;
                for (uint32_t i = 0; i < 32; ++i) {
                    const int32_t qi =
                        int32_t(sycl::round(float(vbuf[j * 32 + i]) * inv));
                    ob->qs[i] = int8_t(qi);
                    qsum += qi;
                }
                ob->d = d;
                ob->s = d * float(qsum);
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

// ===== dn_qkv_split_norm_fused =====
// v1.4 fusion (2026-06-09): replaces the 5-launch cluster
//   cast_qkv_split + 2× repeat_interleave_heads + 2× l2_norm_scale
// (150 submissions/token at decode).  Per token: 2·SKH SGs — SGs 0..SKH-1
// handle Q source heads, SKH..2·SKH-1 K source heads.  Each SG computes its
// head's L2 norm ONCE and writes both tiled copies (dst heads h and h+SKH
// share the source head, per the GGUF tiled-repeat convention), normalized
// with the q/k scale.  The V slice casts to fp32 across all lanes.  Math is
// identical to the unfused chain (duplicated heads shared the same norm).
sycl::event dn_qkv_split_norm_fused(sycl::queue& q,
                                    const sycl::half* src,      // [T, 2*KT+VT]
                                    float* q_out, float* k_out, // [T, 2*KT]
                                    float* v_out,               // [T, VT]
                                    uint32_t T, uint32_t skh, uint32_t shd,
                                    float qscale, float eps,
                                    const std::vector<sycl::event>& deps) {
    constexpr int SG = 16;
    const uint32_t KT = skh * shd;          // per-row Q (and K) width
    const uint32_t VT = 2 * KT;             // V width (SVH = 2*SKH heads)
    const uint32_t row_stride = 2 * KT + VT;
    const uint32_t WG = 2 * skh * SG;       // one SG per source head (q+k)

    return ie::ps(q, "dn_qkv_split_norm", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<2>({T, WG}, {1, WG}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint32_t t    = uint32_t(it.get_group(0));
            const uint32_t lid  = uint32_t(it.get_local_id(1));
            const uint32_t sgid = lid / SG;
            const uint32_t lane = lid % SG;
            auto sg = it.get_sub_group();

            const bool is_q   = sgid < skh;
            const uint32_t hh = is_q ? sgid : (sgid - skh);
            const sycl::half* x = src + uint64_t(t) * row_stride +
                                  (is_q ? 0 : KT) + hh * shd;
            float* dst = (is_q ? q_out : k_out) + uint64_t(t) * 2 * KT;

            float partial = 0.f;
            for (uint32_t i = lane; i < shd; i += SG) {
                const float v = float(x[i]);
                partial += v * v;
            }
            const float sum_sq =
                sycl::reduce_over_group(sg, partial, sycl::plus<float>());
            const float r = sycl::native::rsqrt(sum_sq + eps) *
                            (is_q ? qscale : 1.0f);
            for (uint32_t i = lane; i < shd; i += SG) {
                const float v = float(x[i]) * r;
                dst[hh * shd + i]         = v;   // tiled copy 1
                dst[(hh + skh) * shd + i] = v;   // tiled copy 2
            }

            // V cast: all lanes share the slice.
            const sycl::half* vsrc = src + uint64_t(t) * row_stride + 2 * KT;
            float* vdst = v_out + uint64_t(t) * VT;
            for (uint32_t i = lid; i < VT; i += WG) vdst[i] = float(vsrc[i]);
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
