// bench/ds41_topk_split_check.cpp -- Phase 39: the split top-k against the single kernel, bit for bit, and timed.
// Run with IE_DS41_TOPK_SPLIT=0 so ds4_indexer_topk is the single kernel; the split is called directly.
#include "ie/deepseek4_attn.hpp"
#include "ie/deepseek41_topk_split.hpp"
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    const int iters = argc > 1 ? std::atoi(argv[1]) : 50;
    sycl::device dev;
    for (auto& d : sycl::device::get_devices(sycl::info::device_type::gpu))
        if (d.get_info<sycl::info::device::name>().find("Arc") != std::string::npos) { dev = d; break; }
    sycl::queue q(dev, sycl::property::queue::in_order{});
    std::mt19937 rng(5); int bad = 0;
    for (uint32_t n_keys : {16384u, 32768u, 65536u, 114688u}) for (uint32_t T : {1u, 4u}) {
        const uint32_t K = 512, R = 2;
        std::vector<float> hs(size_t(T) * n_keys); std::vector<int32_t> hp(T);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& v : hs) v = nd(rng);
        // a few exact ties, and rows whose causal cutoff lands inside the key range (and one with fewer reachable keys than top_k)
        for (size_t i = 0; i + 1 < hs.size(); i += 977) hs[i + 1] = hs[i];
        for (uint32_t t = 0; t < T; ++t) hp[t] = (t == 1) ? int32_t(300 * R) : int32_t((n_keys - 1 - t * 37) * R + 1);
        float* ds = sycl::malloc_device<float>(hs.size(), q); int32_t* dp = sycl::malloc_device<int32_t>(T, q);
        int32_t* o1 = sycl::malloc_device<int32_t>(size_t(T) * K, q); int32_t* o2 = sycl::malloc_device<int32_t>(size_t(T) * K, q);
        q.memcpy(ds, hs.data(), hs.size() * 4); q.memcpy(dp, hp.data(), T * 4).wait();
        ie::ds4_indexer_topk(q, ds, dp, o1, T, n_keys, K, R).wait();
        ie::ds41_indexer_topk_split(q, ds, dp, o2, T, n_keys, K, R).wait();
        std::vector<int32_t> a(size_t(T) * K), b(a.size()); q.memcpy(a.data(), o1, a.size() * 4); q.memcpy(b.data(), o2, b.size() * 4).wait();
        size_t nd_ = 0; for (size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) ++nd_;
        auto tm = [&](auto&& fn) { for (int i = 0; i < 3; ++i) fn(); q.wait(); const auto t0 = std::chrono::steady_clock::now(); for (int i = 0; i < iters; ++i) fn(); q.wait(); return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters; };
        const double us1 = tm([&] { ie::ds4_indexer_topk(q, ds, dp, o1, T, n_keys, K, R); });
        const double us2 = tm([&] { ie::ds41_indexer_topk_split(q, ds, dp, o2, T, n_keys, K, R); });
        std::printf("n_keys %6u T %u: single %8.1f us  split %8.1f us  (%4.2fx)  %s\n", n_keys, T, us1, us2, us1 / us2, nd_ ? ("DIFFER: " + std::to_string(nd_) + " picks").c_str() : "identical picks");
        if (nd_) ++bad;
        sycl::free(ds, q); sycl::free(dp, q); sycl::free(o1, q); sycl::free(o2, q);
    }
    std::printf("%s\n", bad ? "SPLIT TOP-K CHECK: FAIL" : "SPLIT TOP-K CHECK: PASS");
    return bad ? 1 : 0;
}
