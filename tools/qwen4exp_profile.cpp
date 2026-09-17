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
// usage: ie-qwen4exp-profile <model.gguf> [gpu]
#include "ie/gguf.hpp"
#include "ie/kernel_profiler.hpp"
#include "ie/model_config.hpp"
#include "ie/qwen4exp.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace ie;

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <model.gguf> [gpu]\n", argv[0]); return 2; }
    const uint32_t ordinal = argc > 2 ? uint32_t(std::atoi(argv[2])) : 0;

    GgufReader g;
    if (auto e = g.open(argv[1]); !e.empty()) { std::fprintf(stderr, "open: %s\n", e.c_str()); return 1; }
    Qwen4ExpConfig cfg;
    if (auto e = read_qwen4exp_config(g, cfg); !e.empty()) { std::fprintf(stderr, "config: %s\n", e.c_str()); return 1; }
    DeviceAllocator alloc;
    if (auto e = alloc.init("B70", ordinal); !e.empty()) { std::fprintf(stderr, "gpu: %s\n", e.c_str()); return 1; }

    Qwen4ExpModel m;
    if (auto e = m.load(alloc, g, cfg); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
    if (auto e = m.init_runtime(8192, 1024); !e.empty()) { std::fprintf(stderr, "runtime: %s\n", e.c_str()); return 1; }

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
        g_profiler = &prof2;
        prof2.begin_step();
        const auto tp = std::chrono::steady_clock::now();
        if (auto e = m.forward(chunk.data(), 1024, pos, nullptr); !e.empty()) {
            std::fprintf(stderr, "prefill256: %s\n", e.c_str()); return 1;
        }
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - tp).count();
        pos += 1024;
        auto st2 = prof2.harvest();
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

    // Warm the expert cache to steady state before measuring.
    for (int wsteps = 0; wsteps < 24; ++wsteps) { m.forward(&tok, 1, pos, nullptr); ++pos; }
    m.ecache_hits = m.ecache_misses = 0;

    // ---- profiled decode steps -------------------------------------------
    KernelProfiler prof;
    for (int rep = 0; rep < 3; ++rep) {
        g_profiler = &prof;
        prof.begin_step();
        const double rw0 = m.t_route_wait, rh0 = m.t_route_host, ms0 = m.t_moe_submit;
        const auto t0 = std::chrono::steady_clock::now();
        m.forward(&tok, 1, pos, nullptr);
        const double wall = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        auto stats = prof.harvest();
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
    return 0;
}
