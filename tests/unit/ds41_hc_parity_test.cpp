// tests/unit/ds41_hc_parity_test.cpp — ds41_hc_mixes vs the engine's own ds4_hyper_connection.
//
// V4.1 needs `pre` as an output and needs the collapse to consume a pre from the PREVIOUS
// sublayer, which ds4_hyper_connection (four dispatch paths, collapse fused) does not provide.
// So src/ops/deepseek41_ops.cpp carries its own copy of the coefficient math, including the
// Sinkhorn. Duplicated numerics are how two implementations quietly diverge, so this test
// requires the copy to agree with the original on the shared outputs — `post` and `comb` — over
// random inputs at the real V4.1 shape.
//
// Bit-exactness is NOT required and would be the wrong bar: the two kernels reduce the RMS sum
// over hc_mult*hidden in different orders (different work-group tilings), and fp32 addition is
// not associative. The bar is fp32 reduction-order noise.
#include "ie/deepseek4_ops.hpp"
#include "ie/deepseek41_upload.hpp"
#include "ie/deepseek41.hpp"

#include <cstdlib>

#include <cmath>
#include <cstring>
#include <cstdio>
#include <random>
#include <vector>

int main() {
    sycl::device dev;
    bool found = false;
    for (const auto& p : sycl::platform::get_platforms())
        for (const auto& d : p.get_devices(sycl::info::device_type::gpu))
            if (!found && d.get_info<sycl::info::device::name>().find("Arc") != std::string::npos) {
                dev = d; found = true;
            }
    if (!found) { std::printf("ds41_hc_parity_test: no Arc GPU, SKIP\n"); return 77; }
    sycl::queue q(dev, sycl::property::queue::in_order{});

    // The real V4.1 shape: hc_mult 4, hidden 5120, 20 Sinkhorn iterations, hc_eps 1e-6.
    const uint32_t T = 8, HC = 4, H = 5120, ITERS = 20;
    const uint32_t MIX = (2 + HC) * HC, FLAT = HC * H;
    const float rms_eps = 1e-20f, hc_eps = 1e-6f;

    // The Phase 5 gate found the first version of this test could not tell 19, 20 or 21
    // Sinkhorn iterations apart, nor a dropped eps, nor the wrong pass order: its random
    // fn/base/scale produced a comb that was already doubly stochastic before the Sinkhorn
    // ran, so every variant agreed. Now: the REAL layer-0 hc tensors and a stream at the
    // real post-layer scale, and -- below -- the test proves it can discriminate or fails.
    const char* mdir = std::getenv("DS41_MODEL_DIR");
    const std::string model_dir = mdir ? mdir : std::string(std::getenv("HOME")) + "/models/DeepSeek-V4.1-Flash";
    ie::DeepSeek41Model model;
    if (const auto e = model.load(model_dir); !e.empty()) {
        std::printf("ds41_hc_parity_test: model not available (%s), SKIP\n", e.c_str()); return 77;
    }
    const auto& L0 = model.layers()[0];
    std::mt19937 rng(20260912);
    std::normal_distribution<float> n01(0.f, 1.f);
    std::vector<float> h_streams(size_t(T) * FLAT), h_fn(size_t(MIX) * FLAT), h_base(MIX), h_scale(3);
    for (auto& v : h_streams) v = n01(rng) * 0.5f;                  // post-layer scale
    std::memcpy(h_fn.data(),    L0.hc_attn_fn.w->data,    h_fn.size() * 4);
    std::memcpy(h_base.data(),  L0.hc_attn_base.w->data,  h_base.size() * 4);
    std::memcpy(h_scale.data(), L0.hc_attn_scale.w->data, h_scale.size() * 4);

    auto up = [&](const std::vector<float>& v) {
        float* d = sycl::malloc_device<float>(v.size(), q);
        q.memcpy(d, v.data(), v.size() * 4).wait();
        return d;
    };
    float* d_streams = up(h_streams);
    float* d_fn = up(h_fn);
    float* d_base = up(h_base);
    float* d_scale = up(h_scale);

    auto alloc = [&](size_t n) { return sycl::malloc_device<float>(n, q); };
    float *a_post = alloc(size_t(T) * HC), *a_comb = alloc(size_t(T) * HC * HC), *a_coll = alloc(size_t(T) * H);
    float *b_pre = alloc(size_t(T) * HC), *b_post = alloc(size_t(T) * HC), *b_comb = alloc(size_t(T) * HC * HC);

    ie::ds4_hyper_connection(q, d_streams, d_fn, d_base, d_scale, a_post, a_comb, a_coll,
                             T, H, HC, ITERS, rms_eps, hc_eps).wait();
    ie::ds41_hc_mixes(q, d_streams, d_fn, d_base, d_scale, b_pre, b_post, b_comb,
                      T, H, HC, ITERS, rms_eps, hc_eps).wait();

    auto down = [&](const float* d, size_t n) {
        std::vector<float> v(n);
        q.memcpy(v.data(), d, n * 4).wait();
        return v;
    };
    const auto ap = down(a_post, size_t(T) * HC), bp = down(b_post, size_t(T) * HC);
    const auto ac = down(a_comb, size_t(T) * HC * HC), bc = down(b_comb, size_t(T) * HC * HC);
    const auto pre = down(b_pre, size_t(T) * HC);

    auto worst = [](const std::vector<float>& a, const std::vector<float>& b) {
        double m = 0, s = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            m = std::max(m, std::fabs(double(a[i]) - double(b[i])));
            s = std::max(s, std::fabs(double(a[i])));
        }
        return s > 0 ? m / s : 0.0;
    };
    const double dp = worst(ap, bp), dc = worst(ac, bc);

    // The collapse: ds4_hyper_connection folds it in using its OWN pre, so running
    // ds41_hc_collapse with the pre ds41_hc_mixes produced must reproduce it. That pins `pre`
    // itself, which the V4 kernel never exposes.
    float* b_coll = alloc(size_t(T) * H);
    ie::ds41_hc_collapse(q, d_streams, b_pre, b_coll, T, H, HC).wait();
    const double dcoll = worst(down(a_coll, size_t(T) * H), down(b_coll, size_t(T) * H));

    std::printf("ds41_hc_parity_test: post %.3e  comb %.3e  collapse(pre) %.3e\n", dp, dc, dcoll);
    std::printf("  pre[0] = %.6f %.6f %.6f %.6f\n", pre[0], pre[1], pre[2], pre[3]);


    // Self-validation: the same copy at 19 and 21 iterations MUST disagree with the V4 kernel
    // at 20 by more than the bar, or this test cannot see an iteration-count defect and must
    // say so rather than pass.
    double d19 = 0, d21 = 0;
    for (uint32_t iters : {19u, 21u}) {
        float* c_alt = alloc(size_t(T) * HC * HC);
        ie::ds41_hc_mixes(q, d_streams, d_fn, d_base, d_scale, b_pre, b_post, c_alt,
                          T, H, HC, iters, rms_eps, hc_eps).wait();
        const double d = worst(ac, down(c_alt, size_t(T) * HC * HC));
        (iters == 19 ? d19 : d21) = d;
        sycl::free(c_alt, q);
    }
    std::printf("  discrimination: comb at 19 iters differs from V4@20 by %.3e, at 21 by %.3e\n", d19, d21);
    // Every device buffer is dead from here on. The Phase 6 gate found the first version freed
    // these BEFORE the self-validation above, which then read six freed USM pointers: the
    // printed discrimination was nondeterministic, and on the trivial input the guard exists to
    // reject it could exit 0. Frees come last.
    for (void* p : {(void*)d_streams,(void*)d_fn,(void*)d_base,(void*)d_scale,(void*)a_post,
                    (void*)a_comb,(void*)a_coll,(void*)b_pre,(void*)b_post,(void*)b_comb,(void*)b_coll})
        sycl::free(p, q);

    const double tol = 1e-5;
    if (d19 <= tol || d21 <= tol) {
        std::printf("ds41_hc_parity_test: FAIL — cannot distinguish 19/21 from 20 Sinkhorn iterations; "
                    "the input is too trivial for this test to guard anything\n");
        return 1;
    }
    if (dp > tol || dc > tol || dcoll > tol) {
        std::printf("ds41_hc_parity_test: FAIL — the V4.1 copy has diverged from ds4_hyper_connection\n");
        return 1;
    }
    std::printf("ds41_hc_parity_test: all OK (V4.1 mixes == V4 kernel within fp32 reduction noise)\n");
    return 0;
}
