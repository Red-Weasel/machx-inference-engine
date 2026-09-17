// Frozen reference is embedded so the production-linked test does not depend on
// results/.
#include "ie/kernel_profiler.hpp"
#include "ie/ops.hpp"
#include <optional>
#include <string>
#include <type_traits>
#ifdef DN_PREPARE_PRIVATE
namespace ie {
#include "../../results/kernel-seven-2026-09-10/dn/production.inc"
}
#endif

#include <sycl/sycl.hpp>
#include <vector>
namespace frozen {
sycl::event l2_norm_scale(sycl::queue &q, const float *x, float *y,
                          uint32_t n_rows, uint32_t head_dim, float scale,
                          float eps, const std::vector<sycl::event> &deps) {
  constexpr int SG = 16;
  return q.submit([&](sycl::handler &h) {
    h.depends_on(deps);
    h.parallel_for(sycl::nd_range<1>(uint64_t(n_rows) * SG, SG),
                   [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                     const uint32_t row = uint32_t(it.get_group(0));
                     const uint32_t lid = uint32_t(it.get_local_id(0));
                     auto sg = it.get_sub_group();
                     const float *xr = x + row * head_dim;
                     float *yr = y + row * head_dim;

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
  constexpr int SG = 16;
  const uint32_t KT = skh * shd; // per-row Q (and K) width
  const uint32_t VT = 2 * KT;    // V width (SVH = 2*SKH heads)
  const uint32_t row_stride = 2 * KT + VT;
  const uint32_t WG = 2 * skh * SG; // one SG per source head (q+k)

  return q.submit([&](sycl::handler &h) {
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
  return q.submit([&](sycl::handler &hdl) {
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
  return q.submit([&](sycl::handler &hdl) {
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
} // namespace frozen

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <thread>
using E = std::vector<sycl::event>;
using Launch = std::function<sycl::event(const E &)>;
struct Buf {
  sycl::queue &q;
  float *p;
  size_t n;
  Buf(sycl::queue &q_, size_t n_) : q(q_), n(n_) {
    p = sycl::malloc_shared<float>(n + 32, q);
    std::fill(p, p + n + 32, 12345.f);
  }
  ~Buf() { sycl::free(p, q); }
  float *d() { return p + 16; }
  void guard() {
    for (int i = 0; i < 16; i++)
      if (p[i] != 12345.f || p[n + 16 + i] != 12345.f)
        throw std::runtime_error("guard");
  }
};
struct Half {
  sycl::queue &q;
  sycl::half *p;
  size_t n;
  Half(sycl::queue &q_, size_t n_) : q(q_), n(n_) {
    p = sycl::malloc_shared<sycl::half>(n + 32, q);
    std::fill(p, p + n + 32, sycl::half(1234));
  }
  ~Half() { sycl::free(p, q); }
  sycl::half *d() { return p + 16; }
  void guard() {
    for (int i = 0; i < 16; i++)
      if (float(p[i]) != 1234.f || float(p[n + 16 + i]) != 1234.f)
        throw std::runtime_error("half guard");
  }
};
void exact(float *a, float *b, size_t n) {
  for (size_t i = 0; i < n; i++)
    if (std::memcmp(a + i, b + i, 4) && !(std::isnan(a[i]) && std::isnan(b[i])))
      throw std::runtime_error("frozen mismatch index=" + std::to_string(i) +
                               " baseline=" + std::to_string(a[i]) +
                               " candidate=" + std::to_string(b[i]));
}
void near(float a, double b) {
  if (std::isnan(a) && std::isnan(b))
    return;
  if (std::isinf(a) && std::isinf(b) && std::signbit(a) == std::signbit(b))
    return;
  if (!std::isfinite(a) ||
      std::abs(double(a) - b) > 3e-5 * std::max(1., std::abs(b)))
    throw std::runtime_error("oracle mismatch");
}
float input(size_t i) { return float(int((i * 199 + 17) % 977) - 488) / 137.f; }
E delayed(sycl::queue &q) {
  return {q.submit([&](sycl::handler &h) {
    h.host_task(
        [] { std::this_thread::sleep_for(std::chrono::milliseconds(3)); });
  })};
}
void bench(std::string tag, std::vector<std::pair<std::string, Launch>> &fs) {
  constexpr int rounds = 7, reps = 120;
  std::vector<std::vector<double>> times(fs.size());
  for (auto &f : fs)
    for (int i = 0; i < 8; i++)
      f.second({}).wait();
  for (int r = 0; r < rounds; r++)
    for (size_t jj = 0; jj < fs.size(); jj++) {
      size_t j = (jj + r) % fs.size();
      E ev;
      ev.reserve(reps);
      auto start = std::chrono::steady_clock::now();
      for (int i = 0; i < reps; i++)
        ev.push_back(fs[j].second(i ? E{ev.back()} : E{}));
      ev.back().wait_and_throw();
      double ns = 0;
      for (auto &e : ev)
        ns +=
            e.get_profiling_info<sycl::info::event_profiling::command_end>() -
            e.get_profiling_info<sycl::info::event_profiling::command_start>();
      times[j].push_back(ns / reps);
    }
  double base = 0;
  for (size_t j = 0; j < fs.size(); j++) {
    auto t = times[j];
    std::sort(t.begin(), t.end());
    if (!j)
      base = t[rounds / 2];
    std::cout << tag << "," << fs[j].first << "," << t[rounds / 2] << ","
              << t[0] << "," << t.back() << "," << base / t[rounds / 2] << "\n";
  }
}
void l2case(sycl::queue &q, int R, int D, bool timing) {
  size_t N = size_t(R) * D;
  Buf x(q, N), ref(q, N), out(q, N);
  for (size_t i = 0; i < N; i++)
    x.d()[i] = input(i);
  if (N >= D)
    std::fill(x.d(), x.d() + D, 0.f);
  if (N >= size_t(D) * 2 && D > 2) {
    x.d()[D] = 1e15f;
    x.d()[D + 1] = 1e-30f;
  }
  if (N >= size_t(D) * 3)
    for (int d = 0; d < D; d++)
      x.d()[size_t(D) * 2 + d] = 1e-5f;
  const float eps = 1e-6f, scale = .0883883476f;
  std::vector<std::pair<std::string, Launch>> fs;
  fs.push_back({"base", [&](const E &e) {
                  return frozen::l2_norm_scale(q, x.d(), ref.d(), R, D, scale,
                                               eps, e);
                }});
  fs.push_back({"selected", [&](const E &e) {
                  return ie::l2_norm_scale(q, x.d(), out.d(), R, D, scale, eps,
                                           e);
                }});
  fs[0].second({}).wait_and_throw();
  for (int r = 0; r < R; r++) {
    double sum = 0;
    for (int d = 0; d < D; d++)
      sum += double(x.d()[r * D + d]) * x.d()[r * D + d];
    for (int d = 0; d < D; d++)
      near(ref.d()[r * D + d], x.d()[r * D + d] / std::sqrt(sum + eps) * scale);
  }
  for (size_t j = 1; j < fs.size(); j++) {
    auto dep = delayed(q);
    auto fill = q.fill(out.d(), -777.f, N, dep);
    fs[j].second({fill}).wait_and_throw();
    q.wait_and_throw();
    exact(ref.d(), out.d(), N);
    out.guard();
  }
  if (!timing && R && D) {
    std::vector<float> save(x.d(), x.d() + N);
    ie::l2_norm_scale(q, x.d(), x.d(), R, D, scale, eps, {}).wait_and_throw();
    exact(ref.d(), x.d(), N);
    std::copy(save.begin(), save.end(), x.d());
  }
  x.guard();
  ref.guard();
  if (timing) {
    auto saved = ref.p;
    ref.p = out.p;
    bench("l2:" + std::to_string(R) + ":" + std::to_string(D), fs);
    ref.p = saved;
  }
}
void qkvcase(sycl::queue &q, int T, int H, int D, bool timing) {
  size_t KT = size_t(H) * D, N = size_t(T) * 2 * KT;
  Half x(q, N * 2);
  Buf qr(q, N), kr(q, N), vr(q, N), qo(q, N), ko(q, N), vo(q, N);
  for (size_t i = 0; i < N * 2; i++)
    x.d()[i] = sycl::half(input(i));
  if (T && H && D)
    for (int d = 0; d < D; d++)
      x.d()[d] = sycl::half(1e-5f);
  float scale = .0883883476f, eps = 1e-6f;
  std::vector<std::pair<std::string, Launch>> fs;
  fs.push_back({"base", [&](const E &e) {
                  return frozen::dn_qkv_split_norm_fused(
                      q, x.d(), qr.d(), kr.d(), vr.d(), T, H, D, scale, eps, e);
                }});
  fs.push_back({"selected", [&](const E &e) {
                  return ie::dn_qkv_split_norm_fused(
                      q, x.d(), qo.d(), ko.d(), vo.d(), T, H, D, scale, eps, e);
                }});
  if (H)
    fs[0].second({}).wait_and_throw();
  else
    q.ext_oneapi_submit_barrier().wait();
  for (int t = 0; t < T; t++)
    for (int s = 0; s < 2 * H; s++) {
      double sum = 0;
      for (int d = 0; d < D; d++) {
        double v = float(x.d()[t * 4 * KT + s * D + d]);
        sum += v * v;
      }
      for (int d = 0; d < D; d++) {
        float want = float(x.d()[t * 4 * KT + s * D + d]) /
                     std::sqrt(sum + eps) * (s < H ? scale : 1.f);
        auto *dst = s < H ? qr.d() : kr.d();
        near(dst[t * 2 * KT + (s % H) * D + d], want);
        near(dst[t * 2 * KT + KT + (s % H) * D + d], want);
        near(vr.d()[t * 2 * KT + s * D + d],
             float(x.d()[t * 4 * KT + 2 * KT + s * D + d]));
      }
    }
  for (size_t j = 1; j < fs.size(); j++) {
    auto dep = delayed(q);
    auto f1 = q.fill(qo.d(), -777.f, N, dep);
    auto f2 = q.fill(ko.d(), -777.f, N, dep);
    auto f3 = q.fill(vo.d(), -777.f, N, dep);
    fs[j].second({f1, f2, f3}).wait_and_throw();
    q.wait_and_throw();
    exact(qr.d(), qo.d(), N);
    exact(kr.d(), ko.d(), N);
    exact(vr.d(), vo.d(), N);
    qo.guard();
    ko.guard();
    vo.guard();
  }
  x.guard();
  if (timing) {
    auto sq = qr.p, sk = kr.p, sv = vr.p;
    qr.p = qo.p;
    kr.p = ko.p;
    vr.p = vo.p;
    bench("qkv:" + std::to_string(T) + ":" + std::to_string(H) + ":" +
              std::to_string(D),
          fs);
    qr.p = sq;
    kr.p = sk;
    vr.p = sv;
  }
}
template <typename Input>
void gbcase(sycl::queue &q, int R, int H, bool timing) {
  size_t N = size_t(R) * H;
  Buf af(q, N), bf(q, N), A(q, H), dt(q, H), gr(q, N), br(q, N), go(q, N),
      bo(q, N);
  Half ah(q, N), bh(q, N);
  for (size_t i = 0; i < N; i++) {
    float a = input(i) * 20, b = input(i + 57) * 30;
    af.d()[i] = a;
    bf.d()[i] = b;
    ah.d()[i] = sycl::half(a);
    bh.d()[i] = sycl::half(b);
  }
  for (int h = 0; h < H; h++) {
    A.d()[h] = -float(h + 1) / 5;
    dt.d()[h] = input(h);
  }
  const Input *a;
  const Input *b;
  if constexpr (std::is_same_v<Input, float>) {
    a = af.d();
    b = bf.d();
  } else {
    a = ah.d();
    b = bh.d();
  }
  std::vector<std::pair<std::string, Launch>> fs;
  fs.push_back({"base", [&](const E &e) {
                  if constexpr (std::is_same_v<Input, float>)
                    return frozen::compute_g_beta(q, a, b, A.d(), dt.d(),
                                                  gr.d(), br.d(), R, H, e);
                  else
                    return frozen::compute_g_beta_h16(q, a, b, A.d(), dt.d(),
                                                      gr.d(), br.d(), R, H, e);
                }});
  fs.push_back({"selected", [&](const E &e) {
                  if constexpr (std::is_same_v<Input, float>)
                    return ie::compute_g_beta(q, a, b, A.d(), dt.d(), go.d(),
                                              bo.d(), R, H, e);
                  else
                    return ie::compute_g_beta_h16(q, a, b, A.d(), dt.d(),
                                                  go.d(), bo.d(), R, H, e);
                }});
  fs[0].second({}).wait_and_throw();
  for (size_t i = 0; i < N; i++) {
    double v = float(float(a[i]) + dt.d()[i % H]);
    near(gr.d()[i],
         A.d()[i % H] * (std::max(v, 0.) + std::log1p(std::exp(-std::abs(v)))));
    near(br.d()[i], 1. / (1. + std::exp(-double(float(b[i])))));
  }
  for (size_t j = 1; j < fs.size(); j++) {
    auto dep = delayed(q);
    auto f1 = q.fill(go.d(), -777.f, N, dep);
    auto f2 = q.fill(bo.d(), -777.f, N, dep);
    fs[j].second({f1, f2}).wait_and_throw();
    q.wait_and_throw();
    exact(gr.d(), go.d(), N);
    exact(br.d(), bo.d(), N);
    go.guard();
    bo.guard();
  }
  if (!timing && std::is_same_v<Input, float> && N) {
    ie::compute_g_beta(q, af.d(), bf.d(), A.d(), dt.d(), af.d(), bf.d(), R, H,
                       {})
        .wait_and_throw();
    exact(gr.d(), af.d(), N);
    exact(br.d(), bf.d(), N);
  }
  ah.guard();
  bh.guard();
  af.guard();
  bf.guard();
  A.guard();
  dt.guard();
  if (timing) {
    auto sg = gr.p, sb = br.p;
    gr.p = go.p;
    br.p = bo.p;
    bench(std::string(std::is_same_v<Input, float> ? "gbf:" : "gbh:") +
              std::to_string(R) + ":" + std::to_string(H),
          fs);
    gr.p = sg;
    br.p = sb;
  }
}
void empty_dependencies(sycl::queue &q) {
  for (int which = 0; which < 9; ++which) {
    std::atomic<bool> done{false};
    auto producer = q.submit([&](sycl::handler &h) {
      h.host_task([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(3));
        done.store(true, std::memory_order_release);
      });
    });
    E deps{producer};
    sycl::event event;
    switch (which) {
    case 0:
      event = ie::l2_norm_scale(q, nullptr, nullptr, 0, 128, 1.f, 1e-6f, deps);
      break;
    case 1:
      event = ie::l2_norm_scale(q, nullptr, nullptr, 3, 0, 1.f, 1e-6f, deps);
      break;
    case 2:
      event = ie::dn_qkv_split_norm_fused(q, nullptr, nullptr, nullptr, nullptr,
                                          0, 16, 128, 1.f, 1e-6f, deps);
      break;
    case 3:
      event = ie::dn_qkv_split_norm_fused(q, nullptr, nullptr, nullptr, nullptr,
                                          3, 0, 128, 1.f, 1e-6f, deps);
      break;
    case 4:
      event = ie::compute_g_beta(q, nullptr, nullptr, nullptr, nullptr, nullptr,
                                 nullptr, 0, 48, deps);
      break;
    case 5:
      event = ie::compute_g_beta(q, nullptr, nullptr, nullptr, nullptr, nullptr,
                                 nullptr, 3, 0, deps);
      break;
    case 6:
      event = ie::compute_g_beta_h16(q, nullptr, nullptr, nullptr, nullptr,
                                     nullptr, nullptr, 0, 48, deps);
      break;
    case 7:
      event = ie::compute_g_beta_h16(q, nullptr, nullptr, nullptr, nullptr,
                                     nullptr, nullptr, 3, 0, deps);
      break;
    default:
      event = ie::dn_qkv_split_norm_fused(q, nullptr, nullptr, nullptr, nullptr,
                                          3, 16, 0, 1.f, 1e-6f, deps);
      break;
    }
    event.wait_and_throw();
    if (!done.load(std::memory_order_acquire)) {
      producer.wait();
      throw std::runtime_error("empty event dropped dependencies");
    }
  }
}
int main(int argc, char **argv) {
  try {
    std::string mode = argc > 1 ? argv[1] : "verify";
    int card = argc > 2 ? std::atoi(argv[2]) : -1;
    int index = 0;
    int ran = 0;
    for (auto &d : sycl::device::get_devices(sycl::info::device_type::gpu)) {
      auto name = d.get_info<sycl::info::device::name>();
      if (name.find("B70") == std::string::npos)
        continue;
      int current = index++;
      if (card >= 0 && current != card)
        continue;
      ++ran;
      sycl::queue q(d, sycl::property::queue::enable_profiling{});
      std::cout << "DEVICE," << current << "," << name << "\n" << std::flush;
      if (mode == "verify") {
        empty_dependencies(q);
        for (int R : {0, 1, 3, 17, 49, 12288, 12289})
          for (int D : {0, 1, 15, 16, 17, 63, 127, 128, 129})
            if (R < 100 || D == 128)
              l2case(q, R, D, false);
        for (int T : {0, 1, 3, 17, 256, 257})
          for (int H : {0, 1, 3, 16, 24})
            for (int D : {1, 17, 127, 128, 129})
              if (T < 100 || (D == 128 && (H == 16 || H == 24)))
                qkvcase(q, T, H, D, false);
        for (int R : {0, 1, 3, 17, 256, 512, 513})
          for (int H : {0, 1, 7, 32, 48, 64, 65}) {
            gbcase<float>(q, R, H, false);
            gbcase<sycl::half>(q, R, H, false);
          }
        std::cout << "PASS correctness card=" << current << "\n";
      } else if (mode == "l2")
        for (int R : {32, 48, 96, 768, 12288, 24576})
          l2case(q, R, 128, true);
      else if (mode == "qkv")
        for (int T : {1, 4, 16, 128, 256, 512})
          for (int H : {16, 24})
            qkvcase(q, T, H, 128, true);
      else if (mode == "gb")
        for (int R : {1, 4, 16, 128, 256, 512})
          for (int H : {32, 48, 64}) {
            gbcase<float>(q, R, H, true);
            gbcase<sycl::half>(q, R, H, true);
          }
    }
    if (ran == 0)
      throw std::runtime_error("no B70 matched the requested card selector");
  } catch (const std::exception &e) {
    std::cerr << "FAIL " << e.what() << "\n";
    return 1;
  }
}
