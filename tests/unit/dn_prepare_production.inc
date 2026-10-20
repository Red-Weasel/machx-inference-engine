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
