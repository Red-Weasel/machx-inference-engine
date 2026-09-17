// FP32 routing is sensitive to accumulation order. Compare the production
// entrypoint to the frozen scalar-lane kernel, with real input dependencies.
#include "ie/glm5next.hpp"
#include <sycl/sycl.hpp>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

static sycl::event frozen_router(sycl::queue& q, const float* x, const float* w,
                            float* rl, uint32_t T, uint32_t H, uint32_t E,
                            const std::vector<sycl::event>& deps) {
    constexpr int SG = 32;
    return q.submit( [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<2>({T, uint64_t(E) * SG}, {1, SG}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint32_t t = uint32_t(it.get_group(0));
            const uint32_t e = uint32_t(it.get_group(1));
            const uint32_t l = uint32_t(it.get_local_id(1));
            auto sg = it.get_sub_group();
            const float* xr = x + uint64_t(t) * H;
            const float* wr = w + uint64_t(e) * H;
            float acc = 0.f;
            for (uint32_t k = l; k < H; k += SG) acc = sycl::fma(xr[k], wr[k], acc);
            acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
            if (l == 0) rl[uint64_t(t) * E + e] = acc;
        });
    });
}


using Router = sycl::event (*)(sycl::queue&, const float*, const float*, float*,
    uint32_t, uint32_t, uint32_t, const std::vector<sycl::event>&);

static void check(sycl::queue& q, uint32_t H, uint32_t E, uint32_t T,
                  unsigned banks, bool timing, int pattern = 0, bool offset = false) {
    const size_t nx = size_t(H) * T, nw = size_t(H) * E;
    const size_t ny = size_t(T) * E + 32;
    std::vector<float> hx(nx * banks), hw(nw * banks), ref(ny * banks), out(ny);
    std::mt19937 rng(8592 + T + H + E);
    for (auto& v : hx)
        v = pattern == 1 ? 0.f : pattern == 2 ? 1.f : (int(rng() % 2001) - 1000) / 997.f;
    for (auto& v : hw)
        v = pattern ? 1.f : (int(rng() % 2001) - 1000) / 991.f;
    auto* xa = sycl::malloc_device<float>(nx * banks + 1, q);
    auto* wa = sycl::malloc_device<float>(nw * banks + 1, q);
    auto* y = sycl::malloc_device<float>(ny, q);
    if (!xa || !wa || !y) throw std::runtime_error("router allocation");
    auto* x = xa + unsigned(offset);
    auto* w = wa + unsigned(offset);
    auto run = [&](Router fn, unsigned bank, const std::vector<sycl::event>& deps = {}) {
        return fn(q, x + nx * bank, w + nw * bank, y + 16, T, H, E, deps);
    };
    for (Router fn : {frozen_router, ie::glm5_router_logits}) {
        for (unsigned bank = 0; bank < banks; ++bank) {
            std::vector<sycl::event> deps{
                q.memcpy(x + nx * bank, hx.data() + nx * bank, nx * sizeof(float)),
                q.memcpy(w + nw * bank, hw.data() + nw * bank, nw * sizeof(float)),
                q.fill(y, -123.f, ny)};
            auto done = run(fn, bank, deps);
            auto* dest = fn == frozen_router ? ref.data() + ny * bank : out.data();
            // No queue-wide wait: the public event must cover every result.
            q.submit([&](sycl::handler& h) {
                h.depends_on(done);
                h.memcpy(dest, y, ny * sizeof(float));
            }).wait_and_throw();
            if (fn != frozen_router && std::memcmp(out.data(), ref.data() + ny * bank,
                                                  ny * sizeof(float)))
                throw std::runtime_error("router FP32/guard mismatch");
            if (pattern) {
                for (size_t i = 0; i < ny; ++i) {
                    const float expected = (i < 16 || i >= ny - 16) ? -123.f
                                         : pattern == 1 ? 0.f : float(H);
                    if (dest[i] != expected) throw std::runtime_error("router exact oracle");
                }
            }
        }
    }
    std::printf("EXACT,H%u,E%u,T%u,banks%u,pattern%d,offset%d\n",
                H, E, T, banks, pattern, offset);
    std::fflush(stdout);
    if (timing) {
        using Clock = std::chrono::steady_clock;
        const auto until = Clock::now() + std::chrono::milliseconds(500);
        do {
            for (unsigned bank = 0; bank < banks; ++bank) run(frozen_router, bank);
            q.wait_and_throw();
        } while (Clock::now() < until);
        const int reps = T >= 128 ? 8 : 64;
        auto measure = [&](Router fn) {
            std::vector<sycl::event> events;
            auto begin = Clock::now();
            for (int i = 0; i < reps; ++i) events.push_back(run(fn, i % banks));
            q.wait_and_throw();
            const double wall = std::chrono::duration<double, std::micro>(
                Clock::now() - begin).count() / reps;
            double gpu = 0;
            for (auto e : events)
                gpu += e.get_profiling_info<sycl::info::event_profiling::command_end>()
                     - e.get_profiling_info<sycl::info::event_profiling::command_start>();
            return std::pair{wall, gpu / (1000 * reps)};
        };
        std::vector<double> bw, aw, bg, ag;
        for (int i = 0; i < 7; ++i) {
            std::pair<double, double> before, after;
            if (i % 2) {
                after = measure(ie::glm5_router_logits); before = measure(frozen_router);
            } else {
                before = measure(frozen_router); after = measure(ie::glm5_router_logits);
            }
            bw.push_back(before.first); aw.push_back(after.first);
            bg.push_back(before.second); ag.push_back(after.second);
        }
        for (auto* samples : {&bw, &aw, &bg, &ag}) std::sort(samples->begin(), samples->end());
        std::printf("RESULT,production,%u,%u,%u,%u,%.3f,%.3f,%.4f,%.3f,%.3f,%.4f\n",
                    H, E, T, banks, bw[3], aw[3], bw[3] / aw[3], bg[3], ag[3], bg[3] / ag[3]);
        std::fflush(stdout);
    }
    sycl::free(xa, q); sycl::free(wa, q); sycl::free(y, q);
}

int main(int argc, char**) {
    std::vector<sycl::device> devices;
    for (auto d : sycl::device::get_devices(sycl::info::device_type::gpu))
        if (d.get_backend() == sycl::backend::ext_oneapi_level_zero &&
            d.get_info<sycl::info::device::name>().find("B70") != std::string::npos)
            devices.push_back(d);
    if (devices.empty()) return 77;
    sycl::queue q(devices.back(), {sycl::property::queue::in_order{},
                                  sycl::property::queue::enable_profiling{}});
    std::printf("Timing backend: Level Zero, PCI %s\n",
        q.get_device().get_info<sycl::ext::intel::info::device::pci_address>().c_str());
    for (unsigned T : {1u, 2u, 3u, 16u, 128u, 1024u}) {
        check(q, 4096, 288, T, 1, argc > 1);
        if (argc > 1 && T <= 16) check(q, 4096, 288, T, 32, true);
    }
    for (auto d : devices) {
        std::printf("Checking Level Zero PCI %s\n",
            d.get_info<sycl::ext::intel::info::device::pci_address>().c_str());
        sycl::queue out_of_order(d);
        check(out_of_order, 4096, 288, 2, 2, false);
        check(out_of_order, 4093, 19, 3, 2, false, 0, true);
        check(out_of_order, 33, 257, 16, 2, false);
        check(out_of_order, 4096, 288, 1, 2, false, 1);
        check(out_of_order, 4096, 288, 2, 2, false, 2);
    }
    std::puts("GATE PASSED");
}
