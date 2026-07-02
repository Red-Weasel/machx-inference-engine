// src/ops/elementwise.cpp — RMSNorm, SiLU, SwiGLU, residual_add, partial-rotary RoPE.

#include "ie/kernel_profiler.hpp"
#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"

#include <sycl/sycl.hpp>

#include <cmath>

namespace ie {

namespace {

// One subgroup per warp, one workgroup per row. WG = SG (16 lanes) keeps
// reduction simple and per-row launch grid 1:1 with `n_rows`.
constexpr int kRmsWG = 16;

}  // namespace

sycl::event rms_norm(sycl::queue& q,
                     const sycl::half* x, const sycl::half* w,
                     sycl::half* y,
                     uint32_t n_rows, uint32_t hidden,
                     float eps,
                     const std::vector<sycl::event>& deps) {
    if (hidden <= 512) {
        constexpr int WG_SIZE = 32;
        return ie::ps(q, "rms_norm", [&](sycl::handler& h) {
            h.depends_on(deps);
            h.parallel_for(sycl::nd_range<1>(n_rows * WG_SIZE, WG_SIZE),
                           [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WG_SIZE)]] {
                const uint32_t row = uint32_t(it.get_group(0));
                const uint32_t lid = uint32_t(it.get_local_id(0));
                const sycl::half* xr = x + row * hidden;
                sycl::half*       yr = y + row * hidden;
                float sum = 0.f;
                for (uint32_t i = lid; i < hidden; i += WG_SIZE) {
                    const float v = float(xr[i]);
                    sum += v * v;
                }
                sum = sycl::reduce_over_group(it.get_sub_group(), sum, sycl::plus<float>());
                const float rsqrt = sycl::native::rsqrt(sum / float(hidden) + eps);
                for (uint32_t i = lid; i < hidden; i += WG_SIZE) {
                    const float v = float(xr[i]) * rsqrt * float(w[i]);
                    yr[i] = sycl::half(v);
                }
            });
        });
    }
    constexpr int WG_SIZE = 256;
    constexpr int SG_SIZE = 16;
    return ie::ps(q, "rms_norm", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(n_rows * WG_SIZE, WG_SIZE),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t row = uint32_t(it.get_group(0));
            const uint32_t lid = uint32_t(it.get_local_id(0));
            const sycl::half* xr = x + row * hidden;
            sycl::half*       yr = y + row * hidden;
            float sum = 0.f;
            for (uint32_t i = lid; i < hidden; i += WG_SIZE) {
                const float v = float(xr[i]);
                sum += v * v;
            }
            sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
            const float rsqrt = sycl::native::rsqrt(sum / float(hidden) + eps);
            for (uint32_t i = lid; i < hidden; i += WG_SIZE) {
                const float v = float(xr[i]) * rsqrt * float(w[i]);
                yr[i] = sycl::half(v);
            }
        });
    });
}

sycl::event rms_norm_one_plus_w(sycl::queue& q,
                                const sycl::half* x, const float* w,
                                sycl::half* y,
                                uint32_t n_rows, uint32_t hidden,
                                float eps,
                                const std::vector<sycl::event>& deps) {
    if (hidden <= 512) {
        constexpr int WG_SIZE = 32;
        return ie::ps(q, "rms_norm_1pw", [&](sycl::handler& h) {
            h.depends_on(deps);
            h.parallel_for(sycl::nd_range<1>(n_rows * WG_SIZE, WG_SIZE),
                           [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WG_SIZE)]] {
                const uint32_t row = uint32_t(it.get_group(0));
                const uint32_t lid = uint32_t(it.get_local_id(0));
                const sycl::half* xr = x + row * hidden;
                sycl::half*       yr = y + row * hidden;
                float sum = 0.f;
                for (uint32_t i = lid; i < hidden; i += WG_SIZE) {
                    const float v = float(xr[i]);
                    sum += v * v;
                }
                sum = sycl::reduce_over_group(it.get_sub_group(), sum, sycl::plus<float>());
                const float rsqrt = sycl::native::rsqrt(sum / float(hidden) + eps);
                for (uint32_t i = lid; i < hidden; i += WG_SIZE) {
                    const float v = float(xr[i]) * rsqrt * (1.0f + w[i]);
                    yr[i] = sycl::half(v);
                }
            });
        });
    }
    constexpr int WG_SIZE = 256;
    constexpr int SG_SIZE = 16;
    return ie::ps(q, "rms_norm_1pw", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(n_rows * WG_SIZE, WG_SIZE),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t row = uint32_t(it.get_group(0));
            const uint32_t lid = uint32_t(it.get_local_id(0));
            const sycl::half* xr = x + row * hidden;
            sycl::half*       yr = y + row * hidden;
            float sum = 0.f;
            for (uint32_t i = lid; i < hidden; i += WG_SIZE) {
                const float v = float(xr[i]);
                sum += v * v;
            }
            sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
            const float rsqrt = sycl::native::rsqrt(sum / float(hidden) + eps);
            for (uint32_t i = lid; i < hidden; i += WG_SIZE) {
                const float v = float(xr[i]) * rsqrt * (1.0f + w[i]);
                yr[i] = sycl::half(v);
            }
        });
    });
}

// v1.5-C (2026-06-10): rms_norm_f32w + Q8_1 emission in one launch.
// Pass 3 quantizes the ROUNDED fp16 outputs (same values the standalone
// quantize_q8_1-after-rms chain saw — numerics-aligned), so the per-token
// quantize launches in the DN / full-attn blocks disappear.  hidden must
// be a multiple of 32 and ≤ 8192 (SLM stage of the row).
sycl::event rms_norm_f32w_q8(sycl::queue& q,
                             const sycl::half* x, const float* w,
                             sycl::half* y, void* q8_out,
                             uint32_t n_rows, uint32_t hidden,
                             float eps,
                             const std::vector<sycl::event>& deps) {
    constexpr int WG_SIZE = 256;
    constexpr int SG_SIZE = 16;
    auto* out8 = static_cast<block_q8_1x*>(q8_out);
    return ie::ps(q, "rms_norm_q8", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<sycl::half, 1> vbuf(hidden, h);
        h.parallel_for(sycl::nd_range<1>(n_rows * WG_SIZE, WG_SIZE),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t row = uint32_t(it.get_group(0));
            const uint32_t lid = uint32_t(it.get_local_id(0));
            const sycl::half* xr = x + row * hidden;
            sycl::half*       yr = y + row * hidden;

            float sum = 0.f;
            for (uint32_t i = lid; i < hidden; i += WG_SIZE) {
                const float v = float(xr[i]);
                sum += v * v;
            }
            sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
            const float rsqrt = sycl::native::rsqrt(sum / float(hidden) + eps);
            for (uint32_t i = lid; i < hidden; i += WG_SIZE) {
                const sycl::half hv = sycl::half(float(xr[i]) * rsqrt * w[i]);
                yr[i]   = hv;
                vbuf[i] = hv;
            }
            sycl::group_barrier(it.get_group());

            // Q8_1 emission: one lane per 32-element block (mirrors
            // quantize_q8_1's math on the rounded halfs).
            const uint32_t n_blocks = hidden / 32;
            for (uint32_t j = lid; j < n_blocks; j += WG_SIZE) {
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

sycl::event rms_norm_f32w(sycl::queue& q,
                          const sycl::half* x, const float* w,
                          sycl::half* y,
                          uint32_t n_rows, uint32_t hidden,
                          float eps,
                          const std::vector<sycl::event>& deps) {
    // Pick WG_SIZE so each row gets enough lanes to actually fill an Xe-core.
    //   hidden=256 (per-head q/k norm): WG=32 — one SIMD32 SG, 8 elems/lane
    //   hidden=2048 (model norm):       WG=256 — 16 SGs, 8 elems/lane,
    //                                   full-WG reduce via SLM
    // The previous version used WG=SG=16 which left a single warp on the
    // Xe-core idling the rest, making this kernel launch-overhead-bound.
    if (hidden <= 512) {
        constexpr int WG_SIZE = 32;
        return ie::ps(q, "rms_norm_f32w", [&](sycl::handler& h) {
            h.depends_on(deps);
            h.parallel_for(sycl::nd_range<1>(n_rows * WG_SIZE, WG_SIZE),
                           [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WG_SIZE)]] {
                const uint32_t row = uint32_t(it.get_group(0));
                const uint32_t lid = uint32_t(it.get_local_id(0));
                const sycl::half* xr = x + row * hidden;
                sycl::half*       yr = y + row * hidden;
                float sum = 0.f;
                for (uint32_t i = lid; i < hidden; i += WG_SIZE) {
                    const float v = float(xr[i]);
                    sum += v * v;
                }
                sum = sycl::reduce_over_group(it.get_sub_group(), sum, sycl::plus<float>());
                const float rsqrt = sycl::native::rsqrt(sum / float(hidden) + eps);
                for (uint32_t i = lid; i < hidden; i += WG_SIZE) {
                    const float v = float(xr[i]) * rsqrt * w[i];
                    yr[i] = sycl::half(v);
                }
            });
        });
    }

    constexpr int WG_SIZE = 256;
    constexpr int SG_SIZE = 16;
    return ie::ps(q, "rms_norm_f32w", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(n_rows * WG_SIZE, WG_SIZE),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t row = uint32_t(it.get_group(0));
            const uint32_t lid = uint32_t(it.get_local_id(0));
            const sycl::half* xr = x + row * hidden;
            sycl::half*       yr = y + row * hidden;
            float sum = 0.f;
            for (uint32_t i = lid; i < hidden; i += WG_SIZE) {
                const float v = float(xr[i]);
                sum += v * v;
            }
            sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
            const float rsqrt = sycl::native::rsqrt(sum / float(hidden) + eps);
            for (uint32_t i = lid; i < hidden; i += WG_SIZE) {
                const float v = float(xr[i]) * rsqrt * w[i];
                yr[i] = sycl::half(v);
            }
        });
    });
}

// =====================================================================
// residual_add_rms_norm_fused — fused (residual_add, rms_norm_f32w)
// =====================================================================
//
// Replaces the common pair:
//   residual_add(q, ws_x, ws_block, ws_x, T*H);
//   rms_norm_f32w(q, ws_x, w_norm, ws_normed, T, H, eps);
// with a single launch.  Per row, one WG cooperatively:
//   1. Adds ws_block + ws_x → ws_x  (residual_add result)
//   2. While doing (1), accumulates sum-of-squares of the result
//   3. WG-reduces the sum, computes rsqrt
//   4. Writes ws_normed[i] = ws_x[i] * rsqrt * w_norm[i]
//
// Both outputs written: ws_x (for next residual_add) and ws_normed
// (for the next gemv).  Eliminates the rms_norm launch entirely — at
// ~29 µs/launch on Intel SYCL, that's ~1 ms/step at 40 layers per
// decode step (only the in-layer residual_attn → post_attn_norm pair
// is fused here; cross-layer residual_moe → next-attn_norm is left
// unfused for now).
//
// Reuses the same WG_SIZE=256 layout that won on rms_norm_f32w —
// per row, full-WG cooperative reduce via SLM.  No extra SLM beyond
// what reduce_over_group needs.
sycl::event residual_add_rms_norm_fused(sycl::queue& q,
                                        sycl::half* ws_x,
                                        const sycl::half* ws_block,
                                        const float* w_norm,
                                        sycl::half* ws_normed,
                                        uint32_t n_rows, uint32_t hidden,
                                        float eps,
                                        const std::vector<sycl::event>& deps) {
    constexpr int WG_SIZE = 256;
    constexpr int SG_SIZE = 16;
    return ie::ps(q, "res_rms_fused", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(n_rows * WG_SIZE, WG_SIZE),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t row = uint32_t(it.get_group(0));
            const uint32_t lid = uint32_t(it.get_local_id(0));
            sycl::half*       xr = ws_x      + row * hidden;
            const sycl::half* br = ws_block  + row * hidden;
            sycl::half*       nr = ws_normed + row * hidden;

            // Pass 1: residual_add + sum-of-squares accumulation.
            float sum = 0.f;
            for (uint32_t i = lid; i < hidden; i += WG_SIZE) {
                const float v = float(xr[i]) + float(br[i]);
                xr[i] = sycl::half(v);
                sum += v * v;
            }
            sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
            const float rsqrt = sycl::native::rsqrt(sum / float(hidden) + eps);

            // Pass 2: apply RMS normalization to write ws_normed.
            for (uint32_t i = lid; i < hidden; i += WG_SIZE) {
                const float v = float(xr[i]) * rsqrt * w_norm[i];
                nr[i] = sycl::half(v);
            }
        });
    });
}

sycl::event cast_fp16_to_fp32(sycl::queue& q, const sycl::half* x, float* y, uint64_t n,
                              const std::vector<sycl::event>& deps) {
    return ie::ps(q, "cast_f16_f32", [&](sycl::handler& h) {
        h.depends_on(deps);
        constexpr uint64_t WG = 256;
        const uint64_t global = ((n + WG - 1) / WG) * WG;
        h.parallel_for(sycl::nd_range<1>(global, WG),
                       [=](sycl::nd_item<1> it) {
            const uint64_t i = it.get_global_id(0);
            if (i < n) y[i] = float(x[i]);
        });
    });
}

sycl::event cast_qkv_split_fp16_to_fp32(sycl::queue& q,
                                        const sycl::half* src,
                                        float* q_dst, float* k_dst, float* v_dst,
                                        uint32_t T, uint32_t k_total, uint32_t v_total,
                                        const std::vector<sycl::event>& deps) {
    // Each row of `src` is laid out [Q (k_total), K (k_total), V (v_total)].
    // Total elements per row = 2*k_total + v_total = SI2.
    const uint32_t row_stride = 2 * k_total + v_total;
    const uint64_t total_q = uint64_t(T) * k_total;
    const uint64_t total_k = uint64_t(T) * k_total;
    const uint64_t total_v = uint64_t(T) * v_total;
    const uint64_t total = total_q + total_k + total_v;
    return ie::ps(q, "cast_qkv_split", [&](sycl::handler& h) {
        h.depends_on(deps);
        constexpr uint64_t WG = 256;
        const uint64_t global = ((total + WG - 1) / WG) * WG;
        h.parallel_for(sycl::nd_range<1>(global, WG),
                       [=](sycl::nd_item<1> it) {
            const uint64_t i = it.get_global_id(0);
            if (i >= total) return;
            // Map a flat index across (Q∥K∥V) into one of the three buffers.
            uint64_t local = i;
            if (local < total_q) {
                const uint32_t t = uint32_t(local / k_total);
                const uint32_t d = uint32_t(local % k_total);
                q_dst[uint64_t(t) * k_total + d] =
                    float(src[uint64_t(t) * row_stride + d]);
                return;
            }
            local -= total_q;
            if (local < total_k) {
                const uint32_t t = uint32_t(local / k_total);
                const uint32_t d = uint32_t(local % k_total);
                k_dst[uint64_t(t) * k_total + d] =
                    float(src[uint64_t(t) * row_stride + k_total + d]);
                return;
            }
            local -= total_k;
            // V
            const uint32_t t = uint32_t(local / v_total);
            const uint32_t d = uint32_t(local % v_total);
            v_dst[uint64_t(t) * v_total + d] =
                float(src[uint64_t(t) * row_stride + 2 * k_total + d]);
        });
    });
}

sycl::event cast_fp32_to_fp16(sycl::queue& q, const float* x, sycl::half* y, uint64_t n,
                              const std::vector<sycl::event>& deps) {
    return ie::ps(q, "cast_f32_f16", [&](sycl::handler& h) {
        h.depends_on(deps);
        constexpr uint64_t WG = 256;
        const uint64_t global = ((n + WG - 1) / WG) * WG;
        h.parallel_for(sycl::nd_range<1>(global, WG),
                       [=](sycl::nd_item<1> it) {
            const uint64_t i = it.get_global_id(0);
            if (i < n) y[i] = sycl::half(x[i]);
        });
    });
}

// TILED repeat: y[t, h_out, d] = x[t, h_out % n_in_heads, d].
// For n_in=16, repeat=2: heads laid out as [k0..k15, k0..k15] (NOT grouped
// [k0,k0,k1,k1,...]). This matches GGUF's V-head storage convention used by
// llama.cpp's converter for Qwen3.5MoE — see convert_hf_to_gguf.py
// `_LinearAttentionVReorderBase` and qwen35moe.cpp's `ggml_repeat_4d`.
sycl::event repeat_interleave_heads(sycl::queue& q,
                                    const float* x, float* y,
                                    uint32_t T, uint32_t n_in_heads,
                                    uint32_t head_dim, uint32_t repeat,
                                    bool interleave,
                                    const std::vector<sycl::event>& deps) {
    return ie::ps(q, "repeat_heads", [&](sycl::handler& h) {
        h.depends_on(deps);
        const uint32_t n_out_heads = n_in_heads * repeat;
        const uint64_t total = uint64_t(T) * n_out_heads * head_dim;
        constexpr uint64_t WG = 256;
        const uint64_t global = ((total + WG - 1) / WG) * WG;
        h.parallel_for(sycl::nd_range<1>(global, WG),
                       [=](sycl::nd_item<1> it) {
            const uint64_t idx = it.get_global_id(0);
            if (idx >= total) return;
            const uint64_t d  = idx % head_dim;
            const uint64_t h_out = (idx / head_dim) % n_out_heads;
            const uint64_t t  = idx / (n_out_heads * head_dim);
            const uint64_t h_in = interleave ? (h_out / repeat)      // qwen3next
                                             : (h_out % n_in_heads); // 27B tiled
            y[idx] = x[(t * n_in_heads + h_in) * head_dim + d];
        });
    });
}

sycl::event split_q_gate_per_head(sycl::queue& q,
                                  const sycl::half* qk,
                                  sycl::half* q_out, sycl::half* gate_out,
                                  uint32_t T, uint32_t n_heads, uint32_t head_dim,
                                  const std::vector<sycl::event>& deps) {
    return ie::ps(q, "split_q_gate", [&](sycl::handler& h) {
        h.depends_on(deps);
        const uint64_t total = uint64_t(T) * n_heads * head_dim;
        constexpr uint64_t WG = 256;
        const uint64_t global = ((total + WG - 1) / WG) * WG;
        h.parallel_for(sycl::nd_range<1>(global, WG),
                       [=](sycl::nd_item<1> it) {
            const uint64_t idx = it.get_global_id(0);
            if (idx >= total) return;
            const uint64_t d   = idx % head_dim;
            const uint64_t h   = (idx / head_dim) % n_heads;
            const uint64_t t   = idx / (n_heads * head_dim);
            // Source (qk): index in [T, n_heads, 2, head_dim] flat.
            const uint64_t src_q   = ((t * n_heads + h) * 2 + 0) * head_dim + d;
            const uint64_t src_g   = ((t * n_heads + h) * 2 + 1) * head_dim + d;
            q_out   [idx] = qk[src_q];
            gate_out[idx] = qk[src_g];
        });
    });
}

sycl::event swiglu(sycl::queue& q,
                   const sycl::half* gate, const sycl::half* up,
                   sycl::half* y, uint64_t n,
                   const std::vector<sycl::event>& deps) {
    return ie::ps(q, "swiglu", [&](sycl::handler& h) {
        h.depends_on(deps);
        constexpr uint64_t WG = 256;
        const uint64_t global = ((n + WG - 1) / WG) * WG;
        h.parallel_for(sycl::nd_range<1>(global, WG),
                       [=](sycl::nd_item<1> it) {
            const uint64_t i = it.get_global_id(0);
            if (i >= n) return;
            const float g = float(gate[i]);
            const float u = float(up[i]);
            const float s = g / (1.f + sycl::native::exp(-g));   // SiLU(gate)
            y[i] = sycl::half(s * u);
        });
    });
}

// gpt-oss clamped gated-SwiGLU (ggml SWIGLU_OAI, alpha=1.702, limit=7.0):
//   x = min(gate, limit);  z = clamp(up, -limit, limit)
//   out = (x * sigmoid(alpha*x)) * (z + 1)
// Bit-exact with ggml_compute_forward_swiglu_oai (ops.cpp:3360-3363): gate is
// clamped ABOVE only, up clamped BOTH sides, and the (z+1) shift on the up path.
sycl::event swiglu_oai(sycl::queue& q,
                       const sycl::half* gate, const sycl::half* up,
                       sycl::half* y, uint64_t n, float alpha, float limit,
                       const std::vector<sycl::event>& deps) {
    return ie::ps(q, "swiglu_oai", [&](sycl::handler& h) {
        h.depends_on(deps);
        constexpr uint64_t WG = 256;
        const uint64_t global = ((n + WG - 1) / WG) * WG;
        h.parallel_for(sycl::nd_range<1>(global, WG),
                       [=](sycl::nd_item<1> it) {
            const uint64_t i = it.get_global_id(0);
            if (i >= n) return;
            const float x = sycl::fmin(float(gate[i]), limit);
            const float z = sycl::fmax(sycl::fmin(float(up[i]), limit), -limit);
            const float glu = x / (1.f + sycl::native::exp(alpha * (-x)));
            y[i] = sycl::half(glu * (z + 1.f));
        });
    });
}

sycl::event geglu(sycl::queue& q,
                  const sycl::half* gate, const sycl::half* up,
                  sycl::half* y, uint64_t n,
                  const std::vector<sycl::event>& deps) {
    // Gemma GeGLU: y = gelu_tanh(gate) * up, where gelu_tanh is the tanh
    // approximation (PyTorch `gelu(approximate="tanh")` = HF `gelu_pytorch_tanh`):
    //   gelu(x) = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3))).
    // Mirrors swiglu's launch shape; fp32 math, fp16 store.
    return ie::ps(q, "geglu", [&](sycl::handler& h) {
        h.depends_on(deps);
        constexpr uint64_t WG = 256;
        constexpr float kSqrt2OverPi = 0.7978845608028654f;
        const uint64_t global = ((n + WG - 1) / WG) * WG;
        h.parallel_for(sycl::nd_range<1>(global, WG),
                       [=](sycl::nd_item<1> it) {
            const uint64_t i = it.get_global_id(0);
            if (i >= n) return;
            const float g = float(gate[i]);
            const float u = float(up[i]);
            const float inner = kSqrt2OverPi * (g + 0.044715f * g * g * g);
            const float gelu = 0.5f * g * (1.f + sycl::tanh(inner));
            y[i] = sycl::half(gelu * u);
        });
    });
}

sycl::event silu(sycl::queue& q,
                 const sycl::half* x, sycl::half* y, uint64_t n,
                 const std::vector<sycl::event>& deps) {
    return ie::ps(q, "silu", [&](sycl::handler& h) {
        h.depends_on(deps);
        constexpr uint64_t WG = 256;
        const uint64_t global = ((n + WG - 1) / WG) * WG;
        h.parallel_for(sycl::nd_range<1>(global, WG),
                       [=](sycl::nd_item<1> it) {
            const uint64_t i = it.get_global_id(0);
            if (i >= n) return;
            const float v = float(x[i]);
            y[i] = sycl::half(v / (1.f + sycl::native::exp(-v)));
        });
    });
}

sycl::event residual_add(sycl::queue& q,
                         const sycl::half* a, const sycl::half* b,
                         sycl::half* y, uint64_t n,
                         const std::vector<sycl::event>& deps) {
    return ie::ps(q, "residual_add", [&](sycl::handler& h) {
        h.depends_on(deps);
        constexpr uint64_t WG = 256;
        const uint64_t global = ((n + WG - 1) / WG) * WG;
        h.parallel_for(sycl::nd_range<1>(global, WG),
                       [=](sycl::nd_item<1> it) {
            const uint64_t i = it.get_global_id(0);
            if (i >= n) return;
            // Saturate to fp16's finite max. gpt-oss-120b's residual stream carries
            // massive activations that exceed 65504 at deep layers; an fp16 `inf`
            // here NaN-cascades the entire forward (the Harmony-chat "!" bug — the
            // inf poisons that position's KV, then every later position reads it).
            // For any model that stays in range this is an EXACT no-op (so PPL-neutral);
            // it only ever changes values that would otherwise overflow to inf.
            // (The fully-correct fix is an fp32 residual stream; this prevents the NaN
            // universally at zero cost.)
            float s = float(a[i]) + float(b[i]);
            s = sycl::fmin(sycl::fmax(s, -65504.0f), 65504.0f);
            y[i] = sycl::half(s);
        });
    });
}

sycl::event sigmoid_gate(sycl::queue& q,
                         const sycl::half* x, const sycl::half* g,
                         sycl::half* y, uint64_t n,
                         const std::vector<sycl::event>& deps) {
    return ie::ps(q, "sigmoid_gate", [&](sycl::handler& h) {
        h.depends_on(deps);
        constexpr uint64_t WG = 256;
        const uint64_t global = ((n + WG - 1) / WG) * WG;
        h.parallel_for(sycl::nd_range<1>(global, WG),
                       [=](sycl::nd_item<1> it) {
            const uint64_t i = it.get_global_id(0);
            if (i >= n) return;
            const float gi = float(g[i]);
            const float s  = 1.f / (1.f + sycl::native::exp(-gi));
            y[i] = sycl::half(float(x[i]) * s);
        });
    });
}

// Partial-rotary RoPE. Rotates the first `n_rotary` dims of each head, leaves
// the remaining (head_dim - n_rotary) dims untouched. Pair convention is
// "halved", matching transformers' Qwen3/Qwen3.6 modeling: pair (r, r + n_rot/2).
//
// theta(r) = base^(-2r / n_rotary)  for r in [0, n_rotary/2)
sycl::event rope_partial(sycl::queue& q,
                         const sycl::half* x, const int32_t* positions,
                         sycl::half* y,
                         uint32_t n_tokens, uint32_t n_heads,
                         uint32_t head_dim, uint32_t n_rotary,
                         float theta_base,
                         const std::vector<sycl::event>& deps) {
    return ie::ps(q, "rope", [&](sycl::handler& h) {
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

// P3a: partial RoPE with llama3 frequency factors. Identical to rope_partial
// except the per-pair frequency is divided by freq_factors[r] (ggml semantics,
// ggml-cpu/ops.cpp: theta /= freq_factors[i]). freq_factors == nullptr makes the
// math BIT-IDENTICAL to rope_partial (the dense forward passes null for qwen3),
// so the two could be deduplicated later. freq_factors length must be n_rotary/2.
sycl::event rope_partial_ff(sycl::queue& q,
                            const sycl::half* x, const int32_t* positions,
                            sycl::half* y,
                            uint32_t n_tokens, uint32_t n_heads,
                            uint32_t head_dim, uint32_t n_rotary,
                            float theta_base, const float* freq_factors,
                            const std::vector<sycl::event>& deps) {
    return ie::ps(q, "rope_ff", [&](sycl::handler& h) {
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

            if (d >= n_rotary) { yr[d] = xr[d]; return; }   // pass-through dim

            const uint32_t half = n_rotary / 2;
            if (d >= half) return;                          // pair owned by item d
            const uint32_t r = d;
            const int32_t pos = positions[token];
            float inv_freq =
                sycl::native::exp(-float(2u * r) / float(n_rotary)
                                  * sycl::log(theta_base));
            // llama3 scaling: divide the frequency by the per-pair factor.
            if (freq_factors) inv_freq /= freq_factors[r];
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

// YaRN RoPE (NEOX), bit-faithful to ggml rope_yarn (ggml-cpu/ops.cpp:5683-5721) +
// ggml_rope_yarn_corr_dims (ggml.c:4333). NTK-by-parts: per-dim theta blends
// extrapolation (theta_extrap, high-freq dims) with interpolation (freq_scale*
// theta_extrap, low-freq dims) via the corr-dim ramp, AND folds the YaRN magnitude
// scale mscale = attn_factor*(1+0.1*ln(1/freq_scale)) into cos/sin at EVERY position
// (this is the term plain rope_partial omits — it makes q·k 1/mscale^2 too small).
// With ext_factor==0 it reduces to plain scaled rope. NEOX pairing (r, r+n_rotary/2);
// x/y may alias (one item owns a full pair). gpt-oss: low=8, high=18, mscale=1.34657.
sycl::event rope_yarn(sycl::queue& q,
                      const sycl::half* x, const int32_t* positions,
                      sycl::half* y,
                      uint32_t n_tokens, uint32_t n_heads,
                      uint32_t head_dim, uint32_t n_rotary,
                      float freq_base, float freq_scale, uint32_t orig_ctx,
                      float ext_factor, float attn_factor,
                      float beta_fast, float beta_slow,
                      const std::vector<sycl::event>& deps) {
    // per-op YaRN constants (ggml.c:4333-4345; ops.cpp:5701), hoisted host-side.
    const float ndf = float(n_rotary);
    auto corr_dim = [&](float n_rot) {
        return ndf * std::log(float(orig_ctx) / (n_rot * 2.0f * float(M_PI)))
               / (2.0f * std::log(freq_base));
    };
    const float low  = std::max(0.0f,       std::floor(corr_dim(beta_fast)));   // gpt-oss: 8
    const float high = std::min(ndf - 1.0f, std::ceil (corr_dim(beta_slow)));   // gpt-oss: 18
    float mscale = attn_factor;
    if (ext_factor != 0.0f) mscale *= 1.0f + 0.1f * std::log(1.0f / freq_scale);
    // DIAGNOSTIC: IE_GPTOSS_ROPE_NOINTERP applies the YaRN mscale but skips the
    // freq interpolation (theta=theta_extrap) — isolates mscale vs interp on PPL.
    static const bool no_interp = std::getenv("IE_GPTOSS_ROPE_NOINTERP") != nullptr;
    const float ext_eff = no_interp ? 0.0f : ext_factor;

    return ie::ps(q, "rope_yarn", [&](sycl::handler& h) {
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
            const uint64_t base_idx = (uint64_t(token) * n_heads + head) * head_dim;
            const sycl::half* xr = x + base_idx;
            sycl::half*       yr = y + base_idx;

            if (d >= n_rotary) { yr[d] = xr[d]; return; }   // pass-through (none for gpt-oss)
            const uint32_t half = n_rotary / 2;
            if (d >= half) return;                          // one item owns pair (r, r+half)
            const uint32_t r = d;
            const int32_t pos = positions[token];

            // theta_extrap = pos * freq_base^(-2r/n_rotary)  (same form as rope_partial)
            const float inv_freq =
                sycl::native::exp(-float(2u * r) / float(n_rotary) * sycl::log(freq_base));
            const float theta_extrap = float(pos) * inv_freq;

            float theta = freq_scale * theta_extrap;        // theta_interp
            if (ext_eff != 0.0f) {
                const float yv   = (float(r) - low) / sycl::fmax(0.001f, high - low);
                const float ramp = 1.0f - sycl::fmin(1.0f, sycl::fmax(0.0f, yv));
                const float ramp_mix = ramp * ext_eff;
                theta = theta * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            } else {
                theta = theta_extrap;   // mscale-only diagnostic (no freq interp)
            }
            const float cs = sycl::cos(theta) * mscale;
            const float sn = sycl::sin(theta) * mscale;
            const float a  = float(xr[r]);
            const float b  = float(xr[r + half]);
            yr[r]        = sycl::half(a * cs - b * sn);
            yr[r + half] = sycl::half(a * sn + b * cs);
        });
    });
}

// y[t, n] += bias[n]  (column-broadcast). Flat over T*N elements.
sycl::event add_bias(sycl::queue& q, sycl::half* y, const float* bias,
                     uint32_t T, uint32_t N,
                     const std::vector<sycl::event>& deps) {
    const uint64_t total = uint64_t(T) * N;
    return ie::ps(q, "add_bias", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(total), [=](sycl::id<1> i) {
            const uint64_t idx = i[0];
            y[idx] = sycl::half(float(y[idx]) + bias[idx % N]);
        });
    });
}

}  // namespace ie
