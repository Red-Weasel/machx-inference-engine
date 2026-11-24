// tools/ie_xchg_bench.cpp — what a card-to-card exchange costs between the two B70s, for deciding whether two cards can
// work on the same token of one conversation (Inkling 2, 2026-11-23).
//
//   ie-xchg-bench [--iters N] [--busy]
//
// For a 2304-float result (one sublayer's output) and an 8 x 2304 one (per-expert partials), measured as a dependent chain
// the way a split decode step would run: a small kernel on card A produces the result, it is pushed to card B (A's queue,
// one shared context: peer writes; peer reads are broken on this bridge), a kernel on card B consumes it and produces the
// answer, which is pushed back, and A's next kernel consumes that. Modes:
//   p2p-event : the chain linked by events only (no host wait inside a round trip)
//   p2p-host  : the host waits on each push before the next submit (what a two-thread loop with a host barrier does)
//   bounce    : through pinned host memory (card -> host, host -> card)
//   thread2   : two host threads, one per card, each submitting its kernel and its push, meeting at a spin barrier each
//               step (the synchronisation a two-thread step loop costs)
// --busy keeps a bandwidth-heavy kernel running on a second queue of each card during the measurements.
// Reports mean / p50 / p99 / max microseconds per round trip (two exchanges).
#include <sycl/sycl.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using clk = std::chrono::steady_clock;

static void report(const char* mode, size_t n, std::vector<double> us) {
    std::sort(us.begin(), us.end());
    double s = 0;
    for (double v : us) s += v;
    std::printf("%-10s %6zu floats: mean %7.1f us  p50 %7.1f  p99 %7.1f  max %7.1f  (round trip = 2 exchanges)\n", mode, n,
                s / us.size(), us[us.size() / 2], us[size_t(us.size() * 0.99)], us.back());
    std::fflush(stdout);
}

int main(int argc, char** argv) {
    int iters = 2000;
    bool busy = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--busy")) busy = true;
    }
    std::vector<sycl::device> gpus;
    for (const auto& p : sycl::platform::get_platforms()) {
        if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        for (const auto& d : p.get_devices(sycl::info::device_type::gpu))
            if (d.get_info<sycl::info::device::name>().find("Arc") != std::string::npos) gpus.push_back(d);
    }
    if (gpus.size() < 2) { std::printf("need two cards (%zu visible)\n", gpus.size()); return 1; }
    gpus.resize(2);
    bool p2p = false;
    try {
        p2p = gpus[0].ext_oneapi_can_access_peer(gpus[1], sycl::ext::oneapi::peer_access::access_supported) &&
              gpus[1].ext_oneapi_can_access_peer(gpus[0], sycl::ext::oneapi::peer_access::access_supported);
        if (p2p) { gpus[0].ext_oneapi_enable_peer_access(gpus[1]); gpus[1].ext_oneapi_enable_peer_access(gpus[0]); }
    } catch (const sycl::exception& e) { std::printf("peer access: %s\n", e.what()); p2p = false; }
    std::printf("peer access: %s\n", p2p ? "yes" : "NO (p2p modes skipped)");
    sycl::context ctx(gpus);
    sycl::queue q0(ctx, gpus[0], sycl::property::queue::in_order{}), q1(ctx, gpus[1], sycl::property::queue::in_order{});
    std::atomic<bool> stop{false};
    std::vector<std::thread> hogs;
    if (busy) {
        for (int c = 0; c < 2; ++c)
            hogs.emplace_back([&, c] {
                sycl::queue qb(ctx, gpus[c], sycl::property::queue::in_order{});
                const size_t N = size_t(256) << 20;                     // 1 GiB of floats
                float* a = sycl::malloc_device<float>(N, qb);
                float* b = sycl::malloc_device<float>(N, qb);
                qb.memset(a, 0, N * 4).wait();
                while (!stop.load()) {
                    qb.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) { b[i] = a[i] * 1.0001f; }).wait();
                }
                sycl::free(a, qb); sycl::free(b, qb);
            });
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    for (size_t n : {size_t(2304), size_t(8 * 2304)}) {
        float* a0 = sycl::malloc_device<float>(n, q0);                  // card 0's result
        float* in0 = sycl::malloc_device<float>(n, q0);                 // card 0's inbox
        float* a1 = sycl::malloc_device<float>(n, q1);
        float* in1 = sycl::malloc_device<float>(n, q1);
        float* h = sycl::malloc_host<float>(n, q0);
        q0.memset(a0, 0, n * 4).wait(); q1.memset(a1, 0, n * 4).wait();
        q0.memset(in0, 0, n * 4).wait(); q1.memset(in1, 0, n * 4).wait();
        auto k0 = [&](std::vector<sycl::event> dep) {
            return q0.submit([&](sycl::handler& hh) {
                hh.depends_on(dep);
                hh.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { a0[i] = in0[i] * 0.5f + 1.f; });
            });
        };
        auto k1 = [&](std::vector<sycl::event> dep) {
            return q1.submit([&](sycl::handler& hh) {
                hh.depends_on(dep);
                hh.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { a1[i] = in1[i] * 0.5f + 1.f; });
            });
        };
        std::vector<double> us;
        // baseline: the two kernels with no exchange (what the chain costs without the copies)
        for (int it = 0; it < iters; ++it) {
            const auto t = clk::now();
            k0({}).wait(); k1({}).wait();
            us.push_back(std::chrono::duration<double, std::micro>(clk::now() - t).count());
        }
        report("kernels", n, us);
        if (p2p) {
            us.clear();
            for (int it = 0; it < iters; ++it) {
                const auto t = clk::now();
                auto e0 = k0({});
                auto c0 = q0.submit([&](sycl::handler& hh) { hh.depends_on(e0); hh.memcpy(in1, a0, n * 4); });
                auto e1 = k1({c0});
                auto c1 = q1.submit([&](sycl::handler& hh) { hh.depends_on(e1); hh.memcpy(in0, a1, n * 4); });
                c1.wait();
                us.push_back(std::chrono::duration<double, std::micro>(clk::now() - t).count());
            }
            report("p2p-event", n, us);
            us.clear();
            for (int it = 0; it < iters; ++it) {
                const auto t = clk::now();
                k0({});
                q0.memcpy(in1, a0, n * 4).wait();
                k1({});
                q1.memcpy(in0, a1, n * 4).wait();
                us.push_back(std::chrono::duration<double, std::micro>(clk::now() - t).count());
            }
            report("p2p-host", n, us);
        }
        us.clear();
        for (int it = 0; it < iters; ++it) {
            const auto t = clk::now();
            k0({});
            q0.memcpy(h, a0, n * 4).wait();
            q1.memcpy(in1, h, n * 4);
            k1({});
            q1.memcpy(h, a1, n * 4).wait();
            q0.memcpy(in0, h, n * 4).wait();
            us.push_back(std::chrono::duration<double, std::micro>(clk::now() - t).count());
        }
        report("bounce", n, us);
        if (p2p) {                                                      // two threads, spin barrier per step
            std::atomic<int> arrive{0};
            std::atomic<int> gen{0};
            auto barrier = [&] {
                const int g = gen.load(std::memory_order_acquire);
                if (arrive.fetch_add(1, std::memory_order_acq_rel) == 1) { arrive.store(0); gen.fetch_add(1, std::memory_order_acq_rel); }
                else while (gen.load(std::memory_order_acquire) == g) {}
            };
            std::vector<double> st;
            std::thread t1([&] {
                for (int it = 0; it < iters; ++it) {
                    k1({});
                    q1.memcpy(in0, a1, n * 4).wait();               // push my half to card 0
                    barrier();
                }
            });
            for (int it = 0; it < iters; ++it) {
                const auto t = clk::now();
                k0({});
                q0.memcpy(in1, a0, n * 4).wait();                   // push my half to card 1
                barrier();
                st.push_back(std::chrono::duration<double, std::micro>(clk::now() - t).count());
            }
            t1.join();
            report("thread2", n, st);
        }
        sycl::free(a0, q0); sycl::free(in0, q0); sycl::free(a1, q1); sycl::free(in1, q1); sycl::free(h, q0);
    }
    stop.store(true);
    for (auto& t : hogs) t.join();
    return 0;
}
