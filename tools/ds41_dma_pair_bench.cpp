// tools/ds41_dma_pair_bench.cpp — ie-ds41-dma-pair-bench: the two cards' PCIe links H2D from pinned host memory, each alone and both at once,
// 4 GiB copies and expert-sized (17.9 MiB) copies (Phase 14 step 0 evidence, docs/deepseek41/39: 26 + 26 GB/s alone and together).
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>
int main() {
    std::vector<sycl::device> devs;
    for (const auto& p : sycl::platform::get_platforms()) { if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) continue; for (const auto& d : p.get_devices(sycl::info::device_type::gpu)) if (d.get_info<sycl::info::device::name>().find("Arc") != std::string::npos) devs.push_back(d); }
    if (devs.size() < 2) { std::printf("need two cards\n"); return 1; }
    const size_t gib = size_t(1) << 30, n = 4 * gib;
    std::vector<sycl::queue> qs; std::vector<uint8_t*> hsrc, ddst;
    for (auto& d : devs) { qs.emplace_back(sycl::context(d), d, sycl::property_list{sycl::property::queue::in_order{}}); }
    for (auto& q : qs) { auto* h = sycl::malloc_host<uint8_t>(n, q); std::memset(h, 1, n); hsrc.push_back(h); ddst.push_back(sycl::malloc_device<uint8_t>(n, q)); q.memcpy(ddst.back(), h, gib).wait(); }
    auto one = [&](size_t c, size_t bytes) { const auto t0 = std::chrono::steady_clock::now(); qs[c].memcpy(ddst[c], hsrc[c], bytes).wait(); return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); };
    for (size_t c = 0; c < 2; ++c) { const double ms = one(c, n); std::printf("card %zu alone: 4 GiB H2D in %.1f ms = %.1f GB/s\n", c, ms, n / (ms / 1000.0) / 1e9); }
    for (int rep = 0; rep < 2; ++rep) {
        double ms[2]; const auto t0 = std::chrono::steady_clock::now();
        std::thread t1([&] { ms[1] = one(1, n); }); ms[0] = one(0, n); t1.join();
        const double wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::printf("both at once: card 0 %.1f GB/s, card 1 %.1f GB/s, aggregate %.1f GB/s over the wall\n", n / (ms[0] / 1000.0) / 1e9, n / (ms[1] / 1000.0) / 1e9, 2.0 * n / (wall / 1000.0) / 1e9);
    }
    // expert-sized copies (17.9 MiB), 64 of them per card, alone and concurrently
    const size_t eb = 18800640; std::vector<double> t;
    auto many = [&](size_t c) { const auto t0 = std::chrono::steady_clock::now(); for (int i = 0; i < 64; ++i) qs[c].memcpy(ddst[c] + (size_t(i) * eb) % (n - eb), hsrc[c] + (size_t(i) * eb) % (n - eb), eb); qs[c].wait(); return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); };
    for (size_t c = 0; c < 2; ++c) { const double ms = many(c); std::printf("card %zu alone: 64 x 17.9 MiB in %.1f ms = %.1f GB/s\n", c, ms, 64.0 * eb / (ms / 1000.0) / 1e9); }
    { double ms[2]; std::thread t1([&] { ms[1] = many(1); }); ms[0] = many(0); t1.join(); std::printf("both at once, expert-sized: card 0 %.1f GB/s, card 1 %.1f GB/s\n", 64.0 * eb / (ms[0] / 1000.0) / 1e9, 64.0 * eb / (ms[1] / 1000.0) / 1e9); }
    for (size_t c = 0; c < 2; ++c) { sycl::free(hsrc[c], qs[c]); sycl::free(ddst[c], qs[c]); }
    return 0;
}
