// tools/qwen4exp_profile.cpp — Flash-Next decode/prefill profile (charter
// step 1: one clean serial capture before any optimization).
//
// Prints, for one decode step (and optionally a prefill chunk):
//   - wall time
//   - GPU-busy time (sum of ie::ps-tagged kernel events) + top buckets
//   - the wall - busy gap = host overhead (launch/sync/copy submission), the
//     number the kernel table cannot show.
// Untagged work (q.memcpy DMA, inline helper kernels) shows up only in the
// gap; a large gap with small buckets means the cost is copies/syncs, not
// kernels.
//
// usage: ie-qwen4exp-profile <model.gguf> [gpu] [ctx] [depth]
#include "ie/gguf.hpp"
#include "ie/kernel_profiler.hpp"
#include "ie/model_config.hpp"
#include "ie/qwen4exp.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

using namespace ie;

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <model.gguf> [gpu]\n", argv[0]); return 2; }
    const uint32_t ordinal = argc > 2 ? uint32_t(std::atoi(argv[2])) : 0;

    // Without IE_QUEUE_PROFILING=1 the queue carries no event times: the tool then prints the true wall times and the host
    // counters only (the second run the profile needs -- the profiling queue inflates wall).
    const bool kprof = std::getenv("IE_QUEUE_PROFILING") != nullptr;
    GgufReader g;
    if (auto e = g.open(argv[1]); !e.empty()) { std::fprintf(stderr, "open: %s\n", e.c_str()); return 1; }
    Qwen4ExpConfig cfg;
    if (auto e = read_qwen4exp_config(g, cfg); !e.empty()) { std::fprintf(stderr, "config: %s\n", e.c_str()); return 1; }
    DeviceAllocator alloc;
    if (auto e = alloc.init("B70", ordinal); !e.empty()) { std::fprintf(stderr, "gpu: %s\n", e.c_str()); return 1; }

    Qwen4ExpModel m;
    if (auto e = m.load(alloc, g, cfg); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
    // 2026-11-10: [ctx] [depth] -- the measurements below run at token depth `depth` (attention's cost grows with it; the
    // owner's sessions decode at 35-44K), the context sized to hold it. Defaults = the tool's old regime (ctx 8192, ~3.1K).
    const uint32_t ctx_arg = argc > 3 ? uint32_t(std::atoi(argv[3])) : 8192;
    const uint32_t depth_arg = argc > 4 ? uint32_t(std::atoi(argv[4])) : 0;
    if (auto e = m.init_runtime(ctx_arg, 1024); !e.empty()) { std::fprintf(stderr, "runtime: %s\n", e.c_str()); return 1; }

    // Warm: short prefill + one decode step (JIT, caches, page-ins).
    std::vector<int32_t> prompt(8);
    for (int i = 0; i < 8; ++i) prompt[i] = 3000 + i * 37;
    if (auto e = m.forward(prompt.data(), uint32_t(prompt.size()), 0, nullptr); !e.empty()) {
        std::fprintf(stderr, "prefill: %s\n", e.c_str()); return 1;
    }
    int32_t tok = 1000;
    uint32_t pos = uint32_t(prompt.size());
    m.forward(&tok, 1, pos, nullptr); ++pos;

    // ---- prefill chunk measurement + sparse-depth advance ----------------
    KernelProfiler prof2;
    for (int rep2 = 0; rep2 < 3; ++rep2) {
        std::vector<int32_t> chunk(1024);
        for (int i = 0; i < 1024; ++i) chunk[i] = 2000 + (i * 97) % 200000;
        if (kprof) { g_profiler = &prof2; prof2.begin_step(); }
        const auto tp = std::chrono::steady_clock::now();
        if (auto e = m.forward(chunk.data(), 1024, pos, nullptr); !e.empty()) {
            std::fprintf(stderr, "prefill256: %s\n", e.c_str()); return 1;
        }
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - tp).count();
        pos += 1024;
        auto st2 = kprof ? prof2.harvest() : decltype(prof2.harvest()){};
        g_profiler = nullptr;
        double busy2 = 0;
        for (auto& s2 : st2) busy2 += s2.total_ms();
        std::sort(st2.begin(), st2.end(),
                  [](const auto& a, const auto& b) { return a.total_ns > b.total_ns; });
        std::printf("== prefill T=1024 @pos %u: %.0f ms  (%.2f ms/token, %.1f tok/s) | GPU busy %.0f ms | gap %.0f ms ==\n",
                    pos - 1024, ms, ms / 1024.0, 1024000.0 / ms, busy2, ms - busy2);
        for (size_t i2 = 0; rep2 == 0 && i2 < st2.size() && i2 < 10; ++i2)
            std::printf("   %-22s %6u %10.2f\n", st2[i2].name.c_str(),
                        st2[i2].calls, st2[i2].total_ms());
    }

    // [depth]: read on (unprofiled) until the position reaches it
    while (depth_arg && pos + 1024 + 64 < depth_arg) {
        std::vector<int32_t> chunk(1024);
        for (int i = 0; i < 1024; ++i) chunk[i] = 2000 + int32_t((uint64_t(pos) * 131 + uint64_t(i) * 97) % 200000);
        if (auto e = m.forward(chunk.data(), 1024, pos, nullptr); !e.empty()) { std::fprintf(stderr, "advance: %s\n", e.c_str()); return 1; }
        pos += 1024;
    }
    if (depth_arg) std::printf("== advanced to position %u (ctx %u) ==\n", pos, ctx_arg);
    // Warm the expert cache to steady state before measuring.
    for (int wsteps = 0; wsteps < 24; ++wsteps) { m.forward(&tok, 1, pos, nullptr); ++pos; }
    m.ecache_hits = m.ecache_misses = 0;

    // ---- profiled decode steps -------------------------------------------
    KernelProfiler prof;
    for (int rep = 0; rep < 3; ++rep) {
        if (kprof) { g_profiler = &prof; prof.begin_step(); }
        const double rw0 = m.t_route_wait, rh0 = m.t_route_host, ms0 = m.t_moe_submit;
        const auto t0 = std::chrono::steady_clock::now();
        m.forward(&tok, 1, pos, nullptr);
        const double wall = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        auto stats = kprof ? prof.harvest() : decltype(prof.harvest()){};
        g_profiler = nullptr;
        ++pos;

        double busy = 0;
        for (auto& s : stats) busy += s.total_ms();
        std::sort(stats.begin(), stats.end(),
                  [](const auto& a, const auto& b) { return a.total_ns > b.total_ns; });
        std::printf("\n== decode step %d @pos %u: wall %.1f ms | GPU busy %.1f ms | host/copy gap %.1f ms | ecache hit %.0f%% ==\n",
                    rep, pos, wall, busy, wall - busy,
                    100.0 * m.ecache_hits / std::max<uint64_t>(1, m.ecache_hits + m.ecache_misses));
        std::printf("   route: D2H wait %.2f ms  host route %.2f ms  moe submit %.2f ms (per step)\n",
                    (m.t_route_wait - rw0) * 1e3, (m.t_route_host - rh0) * 1e3,
                    (m.t_moe_submit - ms0) * 1e3);
        std::printf("   %-22s %6s %10s %9s\n", "bucket", "calls", "total ms", "avg us");
        for (size_t i = 0; i < stats.size() && i < 14; ++i)
            std::printf("   %-22s %6u %10.2f %9.1f\n", stats[i].name.c_str(),
                        stats[i].calls, stats[i].total_ms(),
                        stats[i].avg_ms() * 1000.0);
    }
    // ---- 2026-11-10: a --spec verify forward of T rows against the plain step: what does each extra row cost, by kernel? ------
    // (At the owner's launch a 4-row round costs ~60-68 ms for <= 2.5 tokens against ~29-36 ms for a plain step.) The verify
    // forward computes every row with the decode kernels' bits; commit_verify(T) then adopts all T rows, so the next
    // measurement starts from a consistent state. GPU-busy buckets are the trustworthy numbers (IE_QUEUE_PROFILING inflates
    // wall); IE_PROFILE_VERIFY=0 skips this part.
    if (const char* pv = std::getenv("IE_PROFILE_VERIFY"); !(pv && *pv == '0')) {
        struct Host { double miss = 0, hit = 0, d2h = 0, route = 0, submit = 0; };
        Host h1, h2, h4;
        auto measure = [&](uint32_t T, int reps, std::vector<std::pair<std::string, double>>& out, double& busy_out, double& wall_out, Host& ho) -> std::string {
            std::vector<std::pair<std::string, double>> acc;
            double busy_sum = 0, wall_sum = 0;
            Host hs;
            for (int rep = 0; rep < reps + 1; ++rep) {   // (rep 0 warms the verify path: not counted)
                std::vector<int32_t> ids(T);
                for (uint32_t i = 0; i < T; ++i) ids[i] = 1000 + int32_t((pos * 31 + i * 977) % 150000);
                KernelProfiler pr;
                if (kprof) { g_profiler = &pr; pr.begin_step(); }
                const uint64_t eh0 = m.ecache_hits, em0 = m.ecache_misses;
                const double rw0 = m.t_route_wait, rh0 = m.t_route_host, ms0 = m.t_moe_submit;
                const auto t0 = std::chrono::steady_clock::now();
                std::string e;
                if (T == 1) e = m.forward(ids.data(), 1, pos, nullptr);
                else {
                    m.set_spec_verify(true);
                    e = m.forward(ids.data(), T, pos, nullptr);
                    m.set_spec_verify(false);
                    if (e.empty()) e = m.commit_verify(T);
                }
                const double wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                auto st = kprof ? pr.harvest() : decltype(pr.harvest()){};
                g_profiler = nullptr;
                if (!e.empty()) return e;
                pos += T;
                if (rep == 0) continue;
                wall_sum += wall;
                hs.miss += double(m.ecache_misses - em0); hs.hit += double(m.ecache_hits - eh0);
                hs.d2h += (m.t_route_wait - rw0) * 1e3; hs.route += (m.t_route_host - rh0) * 1e3; hs.submit += (m.t_moe_submit - ms0) * 1e3;
                for (auto& x : st) {
                    busy_sum += x.total_ms();
                    auto it = std::find_if(acc.begin(), acc.end(), [&](const auto& a) { return a.first == x.name; });
                    if (it == acc.end()) acc.emplace_back(x.name, x.total_ms()); else it->second += x.total_ms();
                }
            }
            for (auto& a : acc) a.second /= reps;
            out = acc; busy_out = busy_sum / reps; wall_out = wall_sum / reps;
            ho.miss = hs.miss / reps; ho.hit = hs.hit / reps; ho.d2h = hs.d2h / reps; ho.route = hs.route / reps; ho.submit = hs.submit / reps;
            return {};
        };
        std::vector<std::pair<std::string, double>> s1, s2, s4;
        double b1 = 0, b2 = 0, b4 = 0, w1 = 0, w2 = 0, w4 = 0;
        const int R = 6;
        if (auto e = measure(1, R, s1, b1, w1, h1); !e.empty()) { std::fprintf(stderr, "step: %s\n", e.c_str()); return 1; }
        if (auto e = measure(2, R, s2, b2, w2, h2); !e.empty()) { std::fprintf(stderr, "verify 2: %s\n", e.c_str()); return 1; }
        if (auto e = measure(4, R, s4, b4, w4, h4); !e.empty()) { std::fprintf(stderr, "verify 4: %s\n", e.c_str()); return 1; }
        auto get = [](const std::vector<std::pair<std::string, double>>& v, const std::string& n) {
            for (auto& a : v) if (a.first == n) return a.second;
            return 0.0;
        };
        std::vector<std::string> names;
        for (auto* v : {&s4, &s2, &s1}) for (auto& a : *v) if (std::find(names.begin(), names.end(), a.first) == names.end()) names.push_back(a.first);
        std::sort(names.begin(), names.end(), [&](const std::string& x, const std::string& y) { return get(s4, x) > get(s4, y); });
        std::printf("\n== spec verify against the plain step (mean of %d, GPU-busy ms a forward; the last column = (4 rows - 1 row) / 3) ==\n", R);
        std::printf("   %-26s %9s %9s %9s %12s\n", "bucket", "1 row", "2 rows", "4 rows", "per extra row");
        double shown = 0;
        for (size_t i = 0; i < names.size() && i < 22; ++i) {
            const double a = get(s1, names[i]), b = get(s2, names[i]), c = get(s4, names[i]);
            std::printf("   %-26s %9.2f %9.2f %9.2f %12.2f\n", names[i].c_str(), a, b, c, (c - a) / 3.0);
            shown += c;
        }
        std::printf("   %-26s %9.2f %9.2f %9.2f %12.2f\n", "TOTAL GPU busy", b1, b2, b4, (b4 - b1) / 3.0);
        std::printf("   %-26s %9.2f %9.2f %9.2f %12.2f   (inflated when IE_QUEUE_PROFILING=1)\n", "wall", w1, w2, w4, (w4 - w1) / 3.0);
        std::printf("   %-26s %9.1f %9.1f %9.1f %12.1f\n", "expert picks (hit+miss)", h1.hit + h1.miss, h2.hit + h2.miss, h4.hit + h4.miss, (h4.hit + h4.miss - h1.hit - h1.miss) / 3.0);
        std::printf("   %-26s %9.1f %9.1f %9.1f %12.1f\n", "expert cache misses", h1.miss, h2.miss, h4.miss, (h4.miss - h1.miss) / 3.0);
        std::printf("   %-26s %9.2f %9.2f %9.2f %12.2f\n", "host: route D2H wait ms", h1.d2h, h2.d2h, h4.d2h, (h4.d2h - h1.d2h) / 3.0);
        std::printf("   %-26s %9.2f %9.2f %9.2f %12.2f\n", "host: route ms", h1.route, h2.route, h4.route, (h4.route - h1.route) / 3.0);
        std::printf("   %-26s %9.2f %9.2f %9.2f %12.2f\n", "host: moe submit ms", h1.submit, h2.submit, h4.submit, (h4.submit - h1.submit) / 3.0);
        std::printf("   (measured at position ~%u, ctx %u)\n", pos, ctx_arg);
    }
    return 0;
}
