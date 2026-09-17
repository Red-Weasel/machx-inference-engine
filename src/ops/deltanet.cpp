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
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
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
                           bool sigmoid_gate,
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
                const float sig = 1.f / (1.f + sycl::native::exp(-gz));
                const float act = sigmoid_gate ? sig : gz * sig;
                yr[i] = sycl::half(w * xn * act);
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
                              bool sigmoid_gate,
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
                const float sig = 1.f / (1.f + sycl::native::exp(-gz));
                const float act = sigmoid_gate ? sig : gz * sig;
                const sycl::half hv = sycl::half(w * xn * act);
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

// B70 tuning for DeltaNet preparation. Other shapes use the original kernels.
namespace dn_prepare_detail {

static sycl::event empty(sycl::queue &q, const std::vector<sycl::event> &deps) {
  return q.submit([&](sycl::handler &h) {
    h.depends_on(deps);
    h.single_task([] {});
  });
}

inline bool b70(sycl::queue &q) {
    struct Choice {
        sycl::device device;
        bool tuned;
    };
    // A layer-split engine alternates queues on the same host thread. Keep each
    // device's answer so switching GPUs does not repeat driver property queries.
    static thread_local std::vector<Choice> choices;
    const auto device = q.get_device();
    for (const auto& choice : choices)
        if (choice.device == device) return choice.tuned;
    const bool tuned =
        device.get_backend() == sycl::backend::ext_oneapi_level_zero &&
        device.is_gpu() &&
        device.get_info<sycl::info::device::vendor_id>() == 0x8086 &&
        device.get_info<sycl::info::device::name>().find("B70") != std::string::npos;
    choices.push_back({device, tuned});
    return tuned;
}

static sycl::event l2_128(sycl::queue &q, const float *x, float *y,
                          uint32_t rows, float scale, float eps,
                          const std::vector<sycl::event> &deps) {
  return ie::ps(q, "dn_l2_norm", [&](sycl::handler &h) {
    h.depends_on(deps);
    h.parallel_for(sycl::nd_range<1>(uint64_t(rows) * 16, 16),
                   [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                     const uint32_t row = uint32_t(it.get_group(0));
                     const uint32_t lane = uint32_t(it.get_local_id(0));
                     const float *xr = x + uint64_t(row) * 128;
                     float values[8];
                     float partial = 0.f;
                     for (uint32_t j = 0; j < 8; ++j) {
                       const float v = xr[lane + j * 16];
                       values[j] = v;
                       // The frozen runtime loop contracts its accumulation to
                       // FMA. Make contraction explicit before compile-time
                       // unrolling.
                       partial = sycl::fma(v, v, partial);
                     }
                     const float sum = sycl::reduce_over_group(
                         it.get_sub_group(), partial, sycl::plus<float>());
                     const float r = sycl::native::rsqrt(sum + eps) * scale;
                     for (uint32_t j = 0; j < 8; ++j)
                       y[uint64_t(row) * 128 + lane + j * 16] = values[j] * r;
                   });
  });
}

template <uint32_t HEADS>
static sycl::event qkv_128(sycl::queue &q, const sycl::half *src, float *q_out,
                           float *k_out, float *v_out, uint32_t tokens,
                           float qscale, float eps,
                           const std::vector<sycl::event> &deps) {
  constexpr uint32_t KT = HEADS * 128;
  constexpr uint32_t VT = 2 * KT;
  return ie::ps(q, "dn_qkv_split_norm", [&](sycl::handler &h) {
    h.depends_on(deps);
    h.parallel_for(sycl::nd_range<1>(uint64_t(tokens) * 2 * HEADS * 16, 16),
                   [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                     const uint32_t source_head = uint32_t(it.get_group(0));
                     const uint32_t lane = uint32_t(it.get_local_id(0));
                     const uint32_t t = source_head / (2 * HEADS);
                     const uint32_t s = source_head % (2 * HEADS);
                     const bool is_q = s < HEADS;
                     const uint32_t head = is_q ? s : s - HEADS;
                     const sycl::half *x = src + uint64_t(t) * 4 * KT +
                                           (is_q ? 0 : KT) + head * 128;
                     float *dst = (is_q ? q_out : k_out) + uint64_t(t) * VT;
                     float values[8];
                     float partial = 0.f;
                     for (uint32_t j = 0; j < 8; ++j) {
                       const float v = float(x[lane + j * 16]);
                       values[j] = v;
                       partial = sycl::fma(v, v, partial);
                     }
                     const float sum = sycl::reduce_over_group(
                         it.get_sub_group(), partial, sycl::plus<float>());
                     const float r =
                         sycl::native::rsqrt(sum + eps) * (is_q ? qscale : 1.f);
                     for (uint32_t j = 0; j < 8; ++j) {
                       const uint32_t i = lane + j * 16;
                       const float v = values[j] * r;
                       dst[head * 128 + i] = v;
                       dst[(head + HEADS) * 128 + i] = v;
                     }
                     // Each source-head subgroup casts one contiguous V head.
                     for (uint32_t j = 0; j < 8; ++j) {
                       const uint32_t i = lane + j * 16;
                       v_out[uint64_t(t) * VT + s * 128 + i] = float(
                           src[uint64_t(t) * 4 * KT + 2 * KT + s * 128 + i]);
                     }
                   });
  });
}

template <typename Input, uint32_t HEADS>
static sycl::event gb(sycl::queue &q, const Input *a, const Input *b,
                      const float *A_log, const float *dt_bias, float *g_out,
                      float *beta_out, uint32_t rows,
                      const std::vector<sycl::event> &deps) {
  const uint32_t total = rows * HEADS; // Dispatch bounds total to 32768.
  const char *label =
      std::is_same_v<Input, float> ? "dn_g_beta" : "dn_g_beta_h16";
  return ie::ps(q, label, [&](sycl::handler &h) {
    h.depends_on(deps);
    constexpr uint32_t WG = 64;
    h.parallel_for(sycl::nd_range<1>((uint64_t(total) + WG - 1) / WG * WG, WG),
                   [=](sycl::nd_item<1> it) {
                     const uint32_t i = uint32_t(it.get_global_id(0));
                     if (i >= total)
                       return;
                     const uint32_t head = i % HEADS;
                     const float a_h = float(a[i]) + dt_bias[head];
                     const float ax = sycl::fabs(a_h);
                     const float sp = sycl::fmax(a_h, 0.f) +
                                      sycl::log1p(sycl::native::exp(-ax));
                     g_out[i] = A_log[head] * sp;
                     beta_out[i] =
                         1.f / (1.f + sycl::native::exp(-float(b[i])));
                   });
  });
}

} // namespace dn_prepare_detail


// ===== l2_norm_scale =====
sycl::event l2_norm_scale(sycl::queue &q, const float *x, float *y,
                          uint32_t n_rows, uint32_t head_dim, float scale,
                          float eps, const std::vector<sycl::event> &deps) {
  if (!n_rows || !head_dim)
    return dn_prepare_detail::empty(q, deps);
  if (head_dim == 128 && n_rows <= 12288 && dn_prepare_detail::b70(q))
    return dn_prepare_detail::l2_128(q, x, y, n_rows, scale, eps, deps);

  constexpr int SG = 16;
  return ie::ps(q, "dn_l2_norm", [&](sycl::handler &h) {
    h.depends_on(deps);
    h.parallel_for(sycl::nd_range<1>(uint64_t(n_rows) * SG, SG),
                   [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                     const uint32_t row = uint32_t(it.get_group(0));
                     const uint32_t lid = uint32_t(it.get_local_id(0));
                     auto sg = it.get_sub_group();
                     const float *xr = x + uint64_t(row) * head_dim;
                     float *yr = y + uint64_t(row) * head_dim;

                     float partial = 0.f;
                     for (uint32_t i = lid; i < head_dim; i += SG) {
                       partial += xr[i] * xr[i];
                     }
                     const float sum_sq = sycl::reduce_over_group(
                         sg, partial, sycl::plus<float>());
                     const float r = sycl::native::rsqrt(sum_sq + eps) * scale;
                     for (uint32_t i = lid; i < head_dim; i += SG)
                       yr[i] = xr[i] * r;
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
sycl::event dn_qkv_split_norm_fused(sycl::queue &q,
                                    const sycl::half *src,      // [T, 2*KT+VT]
                                    float *q_out, float *k_out, // [T, 2*KT]
                                    float *v_out,               // [T, VT]
                                    uint32_t T, uint32_t skh, uint32_t shd,
                                    float qscale, float eps,
                                    const std::vector<sycl::event> &deps) {
  if (!T || !skh || !shd)
    return dn_prepare_detail::empty(q, deps);
  if (shd == 128 && T <= 256 && (skh == 16 || skh == 24) &&
      dn_prepare_detail::b70(q)) {
    if (skh == 16)
      return dn_prepare_detail::qkv_128<16>(q, src, q_out, k_out, v_out, T,
                                            qscale, eps, deps);
    return dn_prepare_detail::qkv_128<24>(q, src, q_out, k_out, v_out, T,
                                          qscale, eps, deps);
  }

  constexpr int SG = 16;
  const uint32_t KT = skh * shd; // per-row Q (and K) width
  const uint32_t VT = 2 * KT;    // V width (SVH = 2*SKH heads)
  const uint32_t row_stride = 2 * KT + VT;
  const uint32_t WG = 2 * skh * SG; // one SG per source head (q+k)

  return ie::ps(q, "dn_qkv_split_norm", [&](sycl::handler &h) {
    h.depends_on(deps);
    h.parallel_for(
        sycl::nd_range<2>({T, WG}, {1, WG}),
        [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
          const uint32_t t = uint32_t(it.get_group(0));
          const uint32_t lid = uint32_t(it.get_local_id(1));
          const uint32_t sgid = lid / SG;
          const uint32_t lane = lid % SG;
          auto sg = it.get_sub_group();

          const bool is_q = sgid < skh;
          const uint32_t hh = is_q ? sgid : (sgid - skh);
          const sycl::half *x =
              src + uint64_t(t) * row_stride + (is_q ? 0 : KT) + hh * shd;
          float *dst = (is_q ? q_out : k_out) + uint64_t(t) * 2 * KT;

          float partial = 0.f;
          for (uint32_t i = lane; i < shd; i += SG) {
            const float v = float(x[i]);
            partial += v * v;
          }
          const float sum_sq =
              sycl::reduce_over_group(sg, partial, sycl::plus<float>());
          const float r =
              sycl::native::rsqrt(sum_sq + eps) * (is_q ? qscale : 1.0f);
          for (uint32_t i = lane; i < shd; i += SG) {
            const float v = float(x[i]) * r;
            dst[hh * shd + i] = v;         // tiled copy 1
            dst[(hh + skh) * shd + i] = v; // tiled copy 2
          }

          // V cast: all lanes share the slice.
          const sycl::half *vsrc = src + uint64_t(t) * row_stride + 2 * KT;
          float *vdst = v_out + uint64_t(t) * VT;
          for (uint32_t i = lid; i < VT; i += WG)
            vdst[i] = float(vsrc[i]);
        });
  });
}

// ===== compute_g_beta =====
sycl::event compute_g_beta(sycl::queue &q, const float *a, const float *b,
                           const float *A_log, const float *dt_bias,
                           float *g_out, float *beta_out, uint32_t n_rows,
                           uint32_t n_heads,
                           const std::vector<sycl::event> &deps) {
  if (!n_rows || !n_heads)
    return dn_prepare_detail::empty(q, deps);
  if (n_rows <= 512 && (n_heads == 32 || n_heads == 48 || n_heads == 64) &&
      dn_prepare_detail::b70(q)) {
    if (n_heads == 32)
      return dn_prepare_detail::gb<float, 32>(q, a, b, A_log, dt_bias, g_out,
                                              beta_out, n_rows, deps);
    if (n_heads == 48)
      return dn_prepare_detail::gb<float, 48>(q, a, b, A_log, dt_bias, g_out,
                                              beta_out, n_rows, deps);
    if (n_heads == 64)
      return dn_prepare_detail::gb<float, 64>(q, a, b, A_log, dt_bias, g_out,
                                              beta_out, n_rows, deps);
  }

  return ie::ps(q, "dn_g_beta", [&](sycl::handler &hdl) {
    hdl.depends_on(deps);
    constexpr uint32_t WG = 64;
    const uint64_t total = uint64_t(n_rows) * n_heads;
    const uint64_t global = ((total + WG - 1) / WG) * WG;
    hdl.parallel_for(sycl::nd_range<1>(global, WG), [=](sycl::nd_item<1> it) {
      const uint64_t i = it.get_global_id(0);
      if (i >= total)
        return;
      const uint32_t h = uint32_t(i % n_heads);
      const float a_h = a[i] + dt_bias[h];
      const float ax = sycl::fabs(a_h);
      const float sp =
          sycl::fmax(a_h, 0.f) + sycl::log1p(sycl::native::exp(-ax));
      g_out[i] = A_log[h] * sp;
      beta_out[i] = 1.0f / (1.0f + sycl::native::exp(-b[i]));
    });
  });
}

// Fused fp16-input variant of compute_g_beta — saves two cast_fp16_to_fp32
// launches per DeltaNet layer on the decode hot path.
sycl::event compute_g_beta_h16(sycl::queue &q, const sycl::half *a_h16,
                               const sycl::half *b_h16, const float *A_log,
                               const float *dt_bias, float *g_out,
                               float *beta_out, uint32_t n_rows,
                               uint32_t n_heads,
                               const std::vector<sycl::event> &deps) {
  if (!n_rows || !n_heads)
    return dn_prepare_detail::empty(q, deps);
  if (n_rows <= 512 && (n_heads == 32 || n_heads == 48 || n_heads == 64) &&
      dn_prepare_detail::b70(q)) {
    if (n_heads == 32)
      return dn_prepare_detail::gb<sycl::half, 32>(
          q, a_h16, b_h16, A_log, dt_bias, g_out, beta_out, n_rows, deps);
    if (n_heads == 48)
      return dn_prepare_detail::gb<sycl::half, 48>(
          q, a_h16, b_h16, A_log, dt_bias, g_out, beta_out, n_rows, deps);
    if (n_heads == 64)
      return dn_prepare_detail::gb<sycl::half, 64>(
          q, a_h16, b_h16, A_log, dt_bias, g_out, beta_out, n_rows, deps);
  }

  return ie::ps(q, "dn_g_beta_h16", [&](sycl::handler &hdl) {
    hdl.depends_on(deps);
    constexpr uint32_t WG = 64;
    const uint64_t total = uint64_t(n_rows) * n_heads;
    const uint64_t global = ((total + WG - 1) / WG) * WG;
    hdl.parallel_for(sycl::nd_range<1>(global, WG), [=](sycl::nd_item<1> it) {
      const uint64_t i = it.get_global_id(0);
      if (i >= total)
        return;
      const uint32_t h = uint32_t(i % n_heads);
      const float a_h = float(a_h16[i]) + dt_bias[h];
      const float ax = sycl::fabs(a_h);
      const float sp =
          sycl::fmax(a_h, 0.f) + sycl::log1p(sycl::native::exp(-ax));
      g_out[i] = A_log[h] * sp;
      beta_out[i] = 1.0f / (1.0f + sycl::native::exp(-float(b_h16[i])));
    });
  });
}

// ===== deltanet_recurrence_rowshard — EXPERIMENT D1 (row-shard occupancy probe) =====
//
// Default OFF; selected at the deltanet_recurrence() dispatch site by env
// IE_DN_ROWSHARD=1.  Same math, same fp32 state, same buffers/signature as the
// shipped kernel; only the state-column layout across threads changes.
//
// WHAT IT PROBES — the shipped kernel launches nd_range<2>({B*n_v_heads,128},
//   {1,128}) → only 48 work-groups (B=1, 48 DeltaNet heads) on 24 Xe-cores, and
//   each LANE privately holds a full 128-float state COLUMN (512 B/lane) → GRF
//   pressure spills and resident-subgroup occupancy is starved (~0.95 TFLOP/s
//   effective on pp512 = latency/occupancy-bound, not compute-bound).
//   Hypothesis (PrismML fork gated_delta_net.cpp:56-166 — warp-owns-column with
//   rows_per_lane sharding + 2 warp-reduces/step): shard each column's 128
//   k-rows across a SUBGROUP (SG16, ROWS_PER_LANE=8) so per-lane state drops
//   512 B → 32 B, lifting occupancy while the arithmetic stays bit-for-bit the
//   same recurrence in fp32 (reduction ORDER changes — probe, PPL re-gated).
//
// GEOMETRY — one SG16 subgroup per (head, v-column vv); COLS_PER_WG=8 columns
//   share one 128-item WG (= 8 subgroups, mirrors the shipped 128-item WG) and
//   one SLM staging of q_t/k_t (staged once, reused by all 8 columns).  Lane
//   owns rows i = r*16 + lane, r∈[0,8) (Prism convention → coalesced k/q SLM
//   reads).  The two per-step reductions (kv_mem, out) become reduce_over_group
//   over the 16-lane subgroup.  Launch: nd_range<2>({B*n_v_heads*(128/8),128},
//   {1,128}) = 48*16 = 768 WGs / 6144 subgroups vs the shipped 48 WGs → 16× WGs,
//   comfortably ≥4× the subgroups-in-flight target.
//
// MATH — decay-first, identical to the shipped kernel:
//   S*=α; kv=⟨S,k⟩; δ=(v−kv)·β; S+=k·δ; out=⟨S,q⟩   (α=exp(g)), fp32 throughout.
//   State buffer (state[base + kk*128 + vv]) is read/written element-for-element
//   at the same addresses → interchangeable with the shipped kernel per-segment.
//
// KNOWN RISKS — (a) redundant q/k global reads = 128/COLS_PER_WG = 16× (each
//   column-block WG re-loads the head's q_t/k_t; expected L2-absorbed, and
//   COLS_PER_WG is the knob to trade redundancy vs WG size); (b) 2 group_barriers
//   per step, same as shipped (SLM stage + end-of-step), so barriers are NOT the
//   win — GRF/occupancy is; (c) SLM per WG = q_slm[128]+k_slm[128]+v_slm[8]+
//   sc_slm[2] ≈ 1.07 KiB (tiny → many WGs resident).
//
// FALSIFIER — if kprofile dn_recurrence ms at pp512 does not drop materially
//   (≥ ~1.3×) with IE_DN_ROWSHARD=1, row-shard occupancy is not the lever
//   (bound by the 2 barriers/step or redundant q/k traffic, not GRF) → escalate
//   to a chunked / matrix-form prefill kernel.
static sycl::event deltanet_recurrence_rowshard(
        sycl::queue& q,
        const float* q_in, const float* k_in, const float* v_in,
        const float* g_in, const float* beta_in,
        float* state,
        float* out,
        uint32_t B, uint32_t T,
        uint32_t n_v_heads, uint32_t /*k_head_dim*/, uint32_t /*v_head_dim*/,
        const std::vector<sycl::event>& deps) {
    constexpr int K_DIM_MAX      = 128;
    constexpr int SG             = 16;
    constexpr int ROWS_PER_LANE  = K_DIM_MAX / SG;   // 8 rows/lane
    constexpr int COLS_PER_WG    = 8;                 // subgroups (columns) per WG
    constexpr int WG_ITEMS       = COLS_PER_WG * SG;  // 128
    const uint32_t N_BLK         = K_DIM_MAX / COLS_PER_WG;  // 16 column-blocks/head

    return ie::ps(q, "dn_recurrence_rowshard", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<float, 1> q_slm(K_DIM_MAX,   h);
        sycl::local_accessor<float, 1> k_slm(K_DIM_MAX,   h);
        sycl::local_accessor<float, 1> v_slm(COLS_PER_WG, h);
        sycl::local_accessor<float, 1> sc_slm(2,          h);

        h.parallel_for(
            sycl::nd_range<2>({uint64_t(B) * n_v_heads * N_BLK, WG_ITEMS},
                              {1, WG_ITEMS}),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
                const uint32_t grp  = uint32_t(it.get_group(0));
                const uint32_t blk  = grp % N_BLK;                 // column-block
                const uint32_t bh   = grp / N_BLK;
                const uint32_t b    = bh / n_v_heads;
                const uint32_t hh   = bh % n_v_heads;

                const uint32_t lid  = uint32_t(it.get_local_id(1));
                const uint32_t colb = lid / SG;                    // column in block
                const uint32_t lane = lid % SG;                    // subgroup lane
                const uint32_t vv   = blk * COLS_PER_WG + colb;    // state column
                auto sg = it.get_sub_group();

                const uint64_t state_base =
                    (uint64_t(b) * n_v_heads + hh) * K_DIM_MAX * K_DIM_MAX;

                // Load this lane's row-shard of column vv: rows i = r*16 + lane.
                float s_shard[ROWS_PER_LANE];
                #pragma unroll
                for (int r = 0; r < ROWS_PER_LANE; ++r) {
                    const uint32_t i = uint32_t(r) * SG + lane;
                    s_shard[r] = state[state_base + uint64_t(i) * K_DIM_MAX + vv];
                }

                for (uint32_t t = 0; t < T; ++t) {
                    const uint64_t qkv_row = (uint64_t(b) * T + t) * n_v_heads + hh;
                    // Cooperative SLM stage: the 128 WG items load q_t/k_t once,
                    // shared by all COLS_PER_WG columns.
                    q_slm[lid] = q_in[qkv_row * K_DIM_MAX + lid];
                    k_slm[lid] = k_in[qkv_row * K_DIM_MAX + lid];
                    if (lid < COLS_PER_WG)
                        v_slm[lid] =
                            v_in[qkv_row * K_DIM_MAX + blk * COLS_PER_WG + lid];
                    if (lid == 0) {
                        sc_slm[0] = sycl::native::exp(g_in[qkv_row]);
                        sc_slm[1] = beta_in[qkv_row];
                    }
                    sycl::group_barrier(it.get_group());

                    const float alpha  = sc_slm[0];
                    const float beta_t = sc_slm[1];
                    const float v_t    = v_slm[colb];

                    // Step 9a: decay (matches shipped — applied before kv_mem).
                    #pragma unroll
                    for (int r = 0; r < ROWS_PER_LANE; ++r) s_shard[r] *= alpha;

                    // Step 9b: kv_mem = sum_kk S[kk][vv]·k[kk] — shard partial
                    // then one subgroup reduce over the 16 row-lanes.
                    float kv_part = 0.f;
                    #pragma unroll
                    for (int r = 0; r < ROWS_PER_LANE; ++r)
                        kv_part += s_shard[r] * k_slm[uint32_t(r) * SG + lane];
                    const float kv_mem =
                        sycl::reduce_over_group(sg, kv_part, sycl::plus<float>());

                    // Step 9c: δ = (v − kv_mem)·β
                    const float delta = (v_t - kv_mem) * beta_t;

                    // Step 9d + 9e fused: rank-1 update then out = sum_kk S·q
                    // (out uses the UPDATED shard, as in the shipped kernel).
                    float out_part = 0.f;
                    #pragma unroll
                    for (int r = 0; r < ROWS_PER_LANE; ++r) {
                        const uint32_t i = uint32_t(r) * SG + lane;
                        s_shard[r] += k_slm[i] * delta;
                        out_part   += s_shard[r] * q_slm[i];
                    }
                    const float out_v =
                        sycl::reduce_over_group(sg, out_part, sycl::plus<float>());

                    if (lane == 0) {
                        out[(uint64_t(b) * T + t) * n_v_heads * K_DIM_MAX +
                            hh * K_DIM_MAX + vv] = out_v;
                    }

                    sycl::group_barrier(it.get_group());
                }

                // Write back this lane's row-shard of column vv.
                #pragma unroll
                for (int r = 0; r < ROWS_PER_LANE; ++r) {
                    const uint32_t i = uint32_t(r) * SG + lane;
                    state[state_base + uint64_t(i) * K_DIM_MAX + vv] = s_shard[r];
                }
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

    // Experiment D1 dispatch — row-shard occupancy probe. Default OFF → falls
    // through to the shipped kernel below, byte-identical. Enable: IE_DN_ROWSHARD=1.
    static const bool use_rowshard = []{
        const char* e = std::getenv("IE_DN_ROWSHARD");
        return e && e[0] == '1';
    }();
    if (use_rowshard) {
        return deltanet_recurrence_rowshard(q, q_in, k_in, v_in, g_in, beta_in,
                                            state, out, B, T, n_v_heads,
                                            k_head_dim, v_head_dim, deps);
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

sycl::event deltanet_recurrence_ckpt(sycl::queue& q,
                                const float* q_in, const float* k_in, const float* v_in,
                                const float* g_in, const float* beta_in,
                                float* state,
                                float* out,
                                float* ckpt_out, uint64_t ckpt_step,
                                uint32_t T,
                                uint32_t n_v_heads,
                                uint32_t k_head_dim, uint32_t v_head_dim,
                                const std::vector<sycl::event>& deps) {
    constexpr int K_DIM_MAX = 128;
    if (k_head_dim != K_DIM_MAX || v_head_dim != K_DIM_MAX) {
        sycl::event e;
        return e;
    }
    return ie::ps(q, "dn_recurrence_ckpt", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<float, 1> q_slm(K_DIM_MAX, h);
        sycl::local_accessor<float, 1> k_slm(K_DIM_MAX, h);
        sycl::local_accessor<float, 1> v_slm(K_DIM_MAX, h);
        sycl::local_accessor<float, 1> sc_slm(2,        h);

        const uint32_t WG_ITEMS = K_DIM_MAX;
        h.parallel_for(sycl::nd_range<2>({uint64_t(n_v_heads), WG_ITEMS}, {1, WG_ITEMS}),
                       [=](sycl::nd_item<2> it) {
            const uint32_t hh   = uint32_t(it.get_group(0));
            const uint32_t lid  = uint32_t(it.get_local_id(1));   // = vv
            const uint32_t vv   = lid;

            float S_col[K_DIM_MAX];
            const uint64_t state_base = uint64_t(hh) * K_DIM_MAX * K_DIM_MAX;
            #pragma unroll
            for (int kk = 0; kk < K_DIM_MAX; ++kk) {
                S_col[kk] = state[state_base + uint64_t(kk) * K_DIM_MAX + vv];
            }

            for (uint32_t t = 0; t < T; ++t) {
                const uint64_t qkv_row = uint64_t(t) * n_v_heads + hh;
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

                #pragma unroll
                for (int kk = 0; kk < K_DIM_MAX; ++kk) S_col[kk] *= alpha;

                float kv_mem = 0.f;
                #pragma unroll
                for (int kk = 0; kk < K_DIM_MAX; ++kk) kv_mem += S_col[kk] * k_slm[kk];

                const float delta = (v_t - kv_mem) * beta_t;

                #pragma unroll
                for (int kk = 0; kk < K_DIM_MAX; ++kk) S_col[kk] += k_slm[kk] * delta;

                float out_v = 0.f;
                #pragma unroll
                for (int kk = 0; kk < K_DIM_MAX; ++kk) out_v += S_col[kk] * q_slm[kk];

                out[uint64_t(t) * n_v_heads * K_DIM_MAX + hh * K_DIM_MAX + vv] = out_v;

                // Post-step snapshot: equals the state the T=1-loop's memcpy captured.
                float* ck = ckpt_out + uint64_t(t) * ckpt_step + state_base;
                #pragma unroll
                for (int kk = 0; kk < K_DIM_MAX; ++kk)
                    ck[uint64_t(kk) * K_DIM_MAX + vv] = S_col[kk];

                sycl::group_barrier(it.get_group());
            }

            // Write back S
            #pragma unroll
            for (int kk = 0; kk < K_DIM_MAX; ++kk) {
                state[state_base + uint64_t(kk) * K_DIM_MAX + vv] = S_col[kk];
            }
        });
    });
}

// ===== KDA (Kimi Delta Attention — glm5next) ===============================
// Reference: llama.cpp PR 27754 build_kda_layer + build_delta_net (the KDA
// branch: per-KEY-CHANNEL decay broadcast over the value axis; GDA above is
// the g->ne[0]==1 special case of the same recurrence).

// kda_l2norm: ggml_l2_norm semantics EXACTLY — r = scale / max(sqrt(Σx²), eps)
// (eps is a FLOOR on the norm, NOT added under the sqrt — the existing
// l2_norm_scale computes rsqrt(Σ+eps) and is deliberately left untouched).
template<uint32_t HeadDim>
static sycl::event kda_l2norm_cached(sycl::queue& q,
                                    const float* x, float* y, uint32_t n_rows,
                                    float scale, float eps,
                                    const std::vector<sycl::event>& deps) {
    constexpr uint32_t SG = 16;
    return ie::ps(q, "kda_l2norm", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_rows) * SG, SG),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint32_t row = uint32_t(it.get_group(0));
            const uint32_t lane = uint32_t(it.get_local_id(0));
            const float* xr = x + uint64_t(row) * HeadDim;
            float values[HeadDim / SG];
            float partial = 0.f;
            #pragma unroll
            for (uint32_t j = 0; j < HeadDim / SG; ++j) {
                const float v = xr[j * SG + lane];
                values[j] = v;
                // Preserve the loop kernel's fused accumulation when unrolling.
                partial = sycl::fma(v, v, partial);
            }
            const float sum_sq = sycl::reduce_over_group(
                it.get_sub_group(), partial, sycl::plus<float>());
            const float r = scale / sycl::fmax(sycl::sqrt(sum_sq), eps);
            float* yr = y + uint64_t(row) * HeadDim;
            #pragma unroll
            for (uint32_t j = 0; j < HeadDim / SG; ++j)
                yr[j * SG + lane] = values[j] * r;
        });
    });
}

sycl::event kda_l2norm(sycl::queue& q,
                       const float* x, float* y,
                       uint32_t n_rows, uint32_t head_dim, float scale, float eps,
                       const std::vector<sycl::event>& deps) {
    // Constant widths avoid dynamic private-array indexing and a second read.
    // Restrict dispatch to the range validated on both B70 cards.
    if (n_rows > 0 && n_rows <= 32768) {
        if (head_dim == 64) return kda_l2norm_cached<64>(q, x, y, n_rows, scale, eps, deps);
        if (head_dim == 128) return kda_l2norm_cached<128>(q, x, y, n_rows, scale, eps, deps);
    }
    constexpr int SG = 16;
    return ie::ps(q, "kda_l2norm", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_rows) * SG, SG),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint32_t row = uint32_t(it.get_group(0));
            const uint32_t lid = uint32_t(it.get_local_id(0));
            auto sg = it.get_sub_group();
            const float* xr = x + uint64_t(row) * head_dim;
            float* yr = y + uint64_t(row) * head_dim;
            float partial = 0.f;
            for (uint32_t i = lid; i < head_dim; i += SG) partial += xr[i] * xr[i];
            const float sum_sq = sycl::reduce_over_group(sg, partial, sycl::plus<float>());
            const float r = scale / sycl::fmax(sycl::sqrt(sum_sq), eps);
            for (uint32_t i = lid; i < head_dim; i += SG) yr[i] = xr[i] * r;
        });
    });
}

// kda_gate: per-(head, channel) log-decay.
//   g = lower_bound * sigmoid(-(pre + dt_bias) * a[h])
// with a = GGUF ssm_a = -exp(A_log), so -pre*a = pre*exp(A_log) — the
// reference chain (mul by ssm_a -> sigmoid(-x) -> scale by lower_bound)
// verbatim. Output is the LOG decay; the recurrence exponentiates.
sycl::event kda_gate(sycl::queue& q,
                     const float* pre, const float* dt_bias, const float* a,
                     float* g_out,
                     uint32_t n_rows, uint32_t n_heads, uint32_t head_dim,
                     float lower_bound,
                     const std::vector<sycl::event>& deps) {
    const uint32_t D = n_heads * head_dim;
    // Small GLM chunks avoid emulated 64-bit modulo per element. The
    // bandwidth-bound large prefill path benchmarks better with the flat grid.
    if (n_rows > 0 && n_rows <= 128 && D == 8192) {
        return ie::ps(q, "kda_gate", [&](sycl::handler& h) {
            h.depends_on(deps);
            h.parallel_for(sycl::nd_range<2>({n_rows, D}, {1, 256}),
                           [=](sycl::nd_item<2> it) {
                const uint32_t c = uint32_t(it.get_global_id(1));
                const uint64_t i = uint64_t(it.get_global_id(0)) * D + c;
                const uint32_t hh = c / head_dim;
                const float x = -(pre[i] + dt_bias[c]) * a[hh];
                g_out[i] = lower_bound / (1.f + sycl::native::exp(-x));
            });
        });
    }
    return ie::ps(q, "kda_gate", [&](sycl::handler& h) {
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

// kda_recurrence: the deltanet_recurrence scan with a PER-KEY-CHANNEL decay
// vector — S[k, v] *= exp(g_t[k]) (broadcast over the value axis) instead of
// GDA's per-head scalar. Everything else (delta rule, layouts, the lane-owns-
// one-value-column shape, sequential t loop => bit-stable op order) is the
// shipped kernel's verbatim.
//   q, k, v : [B, T, n_heads, 128] f32 (post-l2norm; q post-scale 1/sqrt(128))
//   g       : [B, T, n_heads, 128] f32 LOG decay (kda_gate output)
//   beta    : [B, T, n_heads]      f32 (post-sigmoid)
//   state   : [B, n_heads, 128, 128] f32 in/out, S[key][value] value-fastest
//   out     : [B, T, n_heads, 128] f32
// The 128-element private state column benefits from the large register file
// at SIMD16 on B70. Keep the arithmetic and workgroup mapping unchanged.
// A functor property avoids the deprecated parallel_for property overload.
template <typename Body>
struct KdaLargeGrfKernel {
    Body body;
    void operator()(sycl::nd_item<2> it) const { body(it); }
    auto get(sycl::ext::oneapi::experimental::properties_tag) const {
        return sycl::ext::oneapi::experimental::properties{
            sycl::ext::intel::experimental::grf_size<256>,
            sycl::ext::oneapi::experimental::sub_group_size<16>};
    }
};

template <bool LARGE_GRF>
static sycl::event kda_recurrence_launch(sycl::queue& q,
                           const float* q_in, const float* k_in, const float* v_in,
                           const float* g_in, const float* beta_in,
                           float* state,
                           float* out,
                           uint32_t B, uint32_t T,
                           uint32_t n_heads, uint32_t k_head_dim, uint32_t v_head_dim,
                           const std::vector<sycl::event>& deps) {
    constexpr int K_DIM_MAX = 128;
    return ie::ps(q, "kda_recurrence", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<float, 1> q_slm(K_DIM_MAX, h);
        sycl::local_accessor<float, 1> k_slm(K_DIM_MAX, h);
        sycl::local_accessor<float, 1> v_slm(K_DIM_MAX, h);
        sycl::local_accessor<float, 1> a_slm(K_DIM_MAX, h);   // exp(g_t) per key
        sycl::local_accessor<float, 1> sc_slm(1, h);

        const uint32_t WG_ITEMS = K_DIM_MAX;
        auto body = [=](sycl::nd_item<2> it) {
            const uint32_t bh = uint32_t(it.get_group(0));
            const uint32_t b  = bh / n_heads;
            const uint32_t hh = bh % n_heads;
            const uint32_t vv = uint32_t(it.get_local_id(1));

            float S_col[K_DIM_MAX];
            const uint64_t state_base = (uint64_t(b) * n_heads + hh) * K_DIM_MAX * K_DIM_MAX;
            #pragma unroll
            for (int kk = 0; kk < K_DIM_MAX; ++kk)
                S_col[kk] = state[state_base + uint64_t(kk) * K_DIM_MAX + vv];

            for (uint32_t t = 0; t < T; ++t) {
                const uint64_t row = (uint64_t(b) * T + t) * n_heads + hh;
                q_slm[vv] = q_in[row * K_DIM_MAX + vv];
                k_slm[vv] = k_in[row * K_DIM_MAX + vv];
                v_slm[vv] = v_in[row * K_DIM_MAX + vv];
                a_slm[vv] = sycl::native::exp(g_in[row * K_DIM_MAX + vv]);
                if (vv == 0) sc_slm[0] = beta_in[row];
                sycl::group_barrier(it.get_group());

                const float beta_t = sc_slm[0];
                const float v_t    = v_slm[vv];

                // decay (per key channel) + kv_mem in one pass
                float kv_mem = 0.f;
                #pragma unroll
                for (int kk = 0; kk < K_DIM_MAX; ++kk) {
                    S_col[kk] *= a_slm[kk];
                    kv_mem += S_col[kk] * k_slm[kk];
                }
                const float delta = (v_t - kv_mem) * beta_t;
                float out_v = 0.f;
                #pragma unroll
                for (int kk = 0; kk < K_DIM_MAX; ++kk) {
                    S_col[kk] += k_slm[kk] * delta;
                    out_v += S_col[kk] * q_slm[kk];
                }
                out[(uint64_t(b) * T + t) * n_heads * K_DIM_MAX + hh * K_DIM_MAX + vv] = out_v;

                sycl::group_barrier(it.get_group());
            }

            #pragma unroll
            for (int kk = 0; kk < K_DIM_MAX; ++kk)
                state[state_base + uint64_t(kk) * K_DIM_MAX + vv] = S_col[kk];
        };
        const sycl::nd_range<2> range({uint64_t(B) * n_heads, WG_ITEMS}, {1, WG_ITEMS});
        if constexpr (LARGE_GRF)
            h.parallel_for(range, KdaLargeGrfKernel{body});
        else
            h.parallel_for(range, body);
    });
}

sycl::event kda_recurrence(sycl::queue& q,
                           const float* q_in, const float* k_in, const float* v_in,
                           const float* g_in, const float* beta_in,
                           float* state,
                           float* out,
                           uint32_t B, uint32_t T,
                           uint32_t n_heads, uint32_t k_head_dim, uint32_t v_head_dim,
                           const std::vector<sycl::event>& deps) {
    if (k_head_dim != 128 || v_head_dim != 128)
        throw std::invalid_argument("KDA recurrence requires 128-wide key and value heads");
    if (B == 0 || T == 0 || n_heads == 0) return q.ext_oneapi_submit_barrier(deps);
    // Cache the target check per submitting thread, but key it by device so a
    // thread alternating queues cannot apply an Intel-only property elsewhere.
    struct Choice { sycl::device device; bool large_grf; };
    static thread_local std::optional<Choice> choice;
    const auto device = q.get_device();
    if (!choice || choice->device != device) {
        const bool tuned = device.get_backend() == sycl::backend::ext_oneapi_level_zero &&
            device.is_gpu() && device.get_info<sycl::info::device::vendor_id>() == 0x8086 &&
            device.get_info<sycl::info::device::name>().find("B70") != std::string::npos;
        choice.emplace(Choice{device, tuned});
    }
    if (choice->large_grf)
        return kda_recurrence_launch<true>(q, q_in, k_in, v_in, g_in, beta_in,
            state, out, B, T, n_heads, k_head_dim, v_head_dim, deps);
    return kda_recurrence_launch<false>(q, q_in, k_in, v_in, g_in, beta_in,
        state, out, B, T, n_heads, k_head_dim, v_head_dim, deps);
}

}  // namespace ie
