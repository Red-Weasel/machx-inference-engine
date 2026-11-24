// tools/ie_graph_bench.cpp — what a kernel launch costs on this driver when submitted one by one versus replayed from a
// recorded SYCL command graph (Inkling 2's ~1000 small launches per decode step, 2026-11-23).
//
//   ie-graph-bench [--card N] [--n KERNELS] [--reps R]
//
// A chain of N dependent small kernels (one work-group of 256 over a 2304-float row, like the step's norms / mixes) on an
// in-order queue: (a) submitted one by one, one host wait at the end; (b) recorded once into a command graph (queue
// recording), replayed R times, one host wait per replay. Also (c) a chain of N medium kernels (2304 x 16 work-items) to
// see whether replay still helps when kernels are not empty. Reports microseconds per kernel.
#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace sx = sycl::ext::oneapi::experimental;
using clk = std::chrono::steady_clock;

int main(int argc, char** argv) {
    unsigned card = 1, n = 1000, reps = 50;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--card") && i + 1 < argc) card = unsigned(std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "--n") && i + 1 < argc) n = unsigned(std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) reps = unsigned(std::atoi(argv[++i]));
    }
    std::vector<sycl::device> arcs;
    for (const auto& p : sycl::platform::get_platforms()) {
        if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        for (const auto& d : p.get_devices(sycl::info::device_type::gpu))
            if (d.get_info<sycl::info::device::name>().find("Arc") != std::string::npos) arcs.push_back(d);
    }
    if (card >= arcs.size()) { std::printf("no card %u\n", card); return 1; }
    const auto dev = arcs[card];
    std::printf("graph support: %s, limited: %s\n", dev.has(sycl::aspect::ext_oneapi_graph) ? "yes" : "no",
                dev.has(sycl::aspect::ext_oneapi_limited_graph) ? "yes" : "no");
    sycl::queue q(dev, sycl::property::queue::in_order{});
    const size_t H = 2304;
    float* a = sycl::malloc_device<float>(H * 16, q);
    float* b = sycl::malloc_device<float>(H * 16, q);
    q.memset(a, 0, H * 16 * 4).wait();
    q.memset(b, 0, H * 16 * 4).wait();
    auto small = [&](int k) {
        float* x = (k & 1) ? b : a;
        float* y = (k & 1) ? a : b;
        q.parallel_for(sycl::nd_range<1>(256, 256), [=](sycl::nd_item<1> it) {
            const size_t l = it.get_local_id(0);
            float s = 0.f;
            for (size_t i = l; i < H; i += 256) s += x[i] * x[i];
            s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
            const float f = 1.f / sycl::sqrt(s / H + 1e-6f);
            for (size_t i = l; i < H; i += 256) y[i] = x[i] * f * 0.5f + 1.f;
        });
    };
    auto medium = [&](int k) {
        float* x = (k & 1) ? b : a;
        float* y = (k & 1) ? a : b;
        q.parallel_for(sycl::range<1>(H * 16), [=](sycl::id<1> i) { y[i] = x[i] * 0.5f + 1.f; });
    };
    for (int pass = 0; pass < 2; ++pass) {
        const char* nm = pass ? "medium (36864 items)" : "small (1 WG of 256)";
        auto kern = [&](int k) { if (pass) medium(k); else small(k); };
        for (unsigned k = 0; k < n; ++k) kern(int(k));                 // warm-up / JIT
        q.wait();
        double best_sub = 1e30, sum_sub = 0;
        for (unsigned r = 0; r < 10; ++r) {
            const auto t = clk::now();
            for (unsigned k = 0; k < n; ++k) kern(int(k));
            q.wait();
            const double us = std::chrono::duration<double, std::micro>(clk::now() - t).count();
            best_sub = std::min(best_sub, us); sum_sub += us;
        }
        std::printf("%-22s submitted: %7.2f us/kernel (best %7.2f)\n", nm, sum_sub / 10 / n, best_sub / n);
        try {
            sx::command_graph<sx::graph_state::modifiable> g(q.get_context(), q.get_device());
            g.begin_recording(q);
            for (unsigned k = 0; k < n; ++k) kern(int(k));
            g.end_recording(q);
            auto t0 = clk::now();
            auto ex = g.finalize();
            const double fin = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
            q.ext_oneapi_graph(ex).wait();                              // first replay
            double best = 1e30, sum = 0;
            for (unsigned r = 0; r < reps; ++r) {
                const auto t = clk::now();
                q.ext_oneapi_graph(ex).wait();
                const double us = std::chrono::duration<double, std::micro>(clk::now() - t).count();
                best = std::min(best, us); sum += us;
            }
            std::printf("%-22s replayed:  %7.2f us/kernel (best %7.2f), finalize %.1f ms\n", nm, sum / reps / n, best / n, fin);
        } catch (const sycl::exception& e) {
            std::printf("%-22s graph: %s\n", nm, e.what());
        }
    }
    sycl::free(a, q); sycl::free(b, q);
    return 0;
}
