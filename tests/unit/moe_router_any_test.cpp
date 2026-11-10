// tests/unit/moe_router_any_test.cpp -- ie::moe_router on every router shape (2026-11-10, known_bugs 51).
// Until 2026-11-10 any shape but 256 experts / top-8 submitted an EMPTY task. Each shape here runs on the GPU and is checked
// against a host reference: fp32 dots of the fp16 input, softmax over all experts, top-k, renormalised, ascending ids.
// Indices must match wherever the reference's k-th and (k+1)-th probabilities are not within 1e-4 relative of each other
// (the device sums in another order); weights within 2e-3 (they are stored as fp16).
#include "ie/ops.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

namespace {
int g_fail = 0;
int run(sycl::queue& q, uint32_t E, uint32_t K, uint32_t H, uint32_t T, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nx(0.f, 1.f), nw(0.f, 0.05f);
    std::vector<sycl::half> x(uint64_t(T) * H);
    std::vector<float> W(uint64_t(E) * H);
    for (auto& v : x) v = sycl::half(nx(rng));
    for (auto& v : W) v = nw(rng);
    auto* dx = sycl::malloc_device<sycl::half>(x.size(), q);
    auto* dW = sycl::malloc_device<float>(W.size(), q);
    auto* di = sycl::malloc_device<int32_t>(uint64_t(T) * K, q);
    auto* dw = sycl::malloc_device<sycl::half>(uint64_t(T) * K, q);
    q.memcpy(dx, x.data(), x.size() * sizeof(sycl::half));
    q.memcpy(dW, W.data(), W.size() * sizeof(float));
    q.fill(di, int32_t(-1), uint64_t(T) * K);
    q.wait();
    ie::moe_router(q, dx, dW, di, dw, T, H, E, K).wait();
    std::vector<int32_t> gi(uint64_t(T) * K);
    std::vector<sycl::half> gw(uint64_t(T) * K);
    q.memcpy(gi.data(), di, gi.size() * sizeof(int32_t));
    q.memcpy(gw.data(), dw, gw.size() * sizeof(sycl::half));
    q.wait();
    sycl::free(dx, q); sycl::free(dW, q); sycl::free(di, q); sycl::free(dw, q);
    int bad = 0, loose = 0;
    for (uint32_t t = 0; t < T; ++t) {
        std::vector<double> l(E);
        for (uint32_t e = 0; e < E; ++e) {
            double s = 0;
            for (uint32_t h = 0; h < H; ++h) s += double(W[uint64_t(e) * H + h]) * double(float(x[uint64_t(t) * H + h]));
            l[e] = s;
        }
        const double m = *std::max_element(l.begin(), l.end());
        double sum = 0;
        for (auto& v : l) { v = std::exp(v - m); sum += v; }
        std::vector<uint32_t> ord(E);
        std::iota(ord.begin(), ord.end(), 0u);
        std::sort(ord.begin(), ord.end(), [&](uint32_t a, uint32_t b) { return l[a] > l[b]; });
        const bool tie = K < E && (l[ord[K - 1]] - l[ord[K]]) <= 1e-4 * l[ord[K - 1]];
        std::vector<uint32_t> top(ord.begin(), ord.begin() + K);
        std::sort(top.begin(), top.end());
        double ren = 0;
        for (uint32_t e : top) ren += l[e] / sum;
        if (tie) { ++loose; continue; }
        for (uint32_t j = 0; j < K; ++j) {
            const int32_t ge = gi[uint64_t(t) * K + j];
            const double rw = (l[top[j]] / sum) / ren, dwv = double(float(gw[uint64_t(t) * K + j]));
            if (ge != int32_t(top[j]) || std::fabs(rw - dwv) > 2e-3) {
                if (bad < 3) std::fprintf(stderr, "    E=%u K=%u H=%u T=%u token %u slot %u: got (%d, %.4f) want (%u, %.4f)\n", E, K, H, T, t, j, ge, dwv, top[j], rw);
                ++bad;
            }
        }
    }
    std::printf("  [%s] %4u experts top-%-2u hidden %4u, %u token(s)%s\n", bad ? "FAIL" : "OK", E, K, H, T,
                loose ? " (a near-tie row skipped)" : "");
    if (bad) ++g_fail;
    return bad;
}
}  // namespace

int main() {
    sycl::device dev; bool found = false;
    for (const auto& d : sycl::device::get_devices()) if (d.is_gpu()) { dev = d; found = true; break; }
    if (!found) { std::fputs("moe_router_any_test: no GPU -- SKIP\n", stderr); return 0; }
    sycl::queue q(dev);
    std::printf("moe_router_any_test on %s\n", dev.get_info<sycl::info::device::name>().c_str());
    uint32_t seed = 20261110;
    for (uint32_t T : {1u, 5u}) {
        run(q, 256, 8, 2048, T, seed++);   // the 35B's shape: the kernels that were always there
        run(q, 16, 4, 1024, T, seed++);    // the shapes that were an empty task
        run(q, 64, 6, 1536, T, seed++);
        run(q, 512, 10, 2560, T, seed++);
        run(q, 128, 8, 1000, T, seed++);   // a hidden size that is not a multiple of 64
        run(q, 7, 7, 256, T, seed++);      // every expert taken
        run(q, 33, 1, 96, T, seed++);      // top-1
    }
    bool threw = false;
    try { ie::moe_router(q, nullptr, nullptr, nullptr, nullptr, 1, 64, 64, 40).wait(); } catch (const std::exception&) { threw = true; }
    std::printf("  [%s] top-40 is refused with an exception, not answered\n", threw ? "OK" : "FAIL");
    if (!threw) ++g_fail;
    std::printf("moe_router_any_test: %s (%d failures)\n", g_fail ? "FAIL" : "OK", g_fail);
    return g_fail ? 1 : 0;
}
