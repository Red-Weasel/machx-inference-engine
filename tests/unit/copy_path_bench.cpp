// tests/unit/copy_path_bench.cpp — a bench, not a CTest: per-call latency of the host <-> device copy paths at the sizes
// decode moves every token (ids, a hidden row, a logits row), 2026-10-20 (GitHub issue #2). Compares a direct copy from
// pageable memory (what the engine did), DeviceAllocator::h2d / d2h of pageable memory (the pinned stage) and of host
// USM (copied directly), a direct copy from host USM pinned in the copying queue's own context, and the same from
// another context's host USM; then copy_across. Needs one B70 (two for copy_across). Median of 200 calls after 50
// warm-up calls (a B70's first kernels and copies run slow).
//
// --engines [ordinal]: which engine each copy kind runs on. Each kind loops for 2 s on that card (default 1); the
// xe driver's per-client engine counters (/proc/self/fdinfo: drm-cycles-<class>, summed over this process's DRM
// fds) say which engine class did the work: bcs = copy engine, ccs = compute, rcs = render. (xpu-smi reports
// engine utilization as N/A on this stack.)
#include "ie/allocator.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dirent.h>
#include <unistd.h>

#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {
double median_us(const std::function<void()>& f) {
    for (int i = 0; i < 50; ++i) f();
    std::vector<double> t(200);
    for (double& v : t) {
        const auto a = std::chrono::steady_clock::now();
        f();
        v = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - a).count();
    }
    std::nth_element(t.begin(), t.begin() + t.size() / 2, t.end());
    return t[t.size() / 2];
}
}  // namespace

// drm-cycles-<class> summed over this process's DRM file descriptors (xe fdinfo).
std::map<std::string, uint64_t> drm_cycles() {
    std::map<std::string, uint64_t> out;
    if (DIR* dir = opendir("/proc/self/fdinfo")) {
        while (dirent* e = readdir(dir)) {
            if (e->d_name[0] == '.') continue;
            std::ifstream f(std::string("/proc/self/fdinfo/") + e->d_name);
            std::string line;
            while (std::getline(f, line))
                if (line.rfind("drm-cycles-", 0) == 0) {
                    const auto c = line.find(':');
                    out[line.substr(11, c - 11)] += std::stoull(line.substr(c + 1));
                }
        }
        closedir(dir);
    }
    return out;
}

// --engines: each copy kind in a loop for 2 s; the engine classes whose cycles grew did the work.
int engines(uint32_t ordinal) {
    ie::DeviceAllocator a;
    if (auto e = a.init("B70", ordinal); !e.empty()) { std::printf("SKIP: %s\n", e.c_str()); return 77; }
    constexpr size_t kBig = 64ull << 20, kSmall = 10240;
    void* d = a.malloc(kBig);
    std::vector<uint8_t> pageable(kBig, 1);
    auto* pinned = static_cast<uint8_t*>(a.malloc_host(kBig));
    if (!d || !pinned) { std::printf("alloc failed\n"); return 1; }
    sycl::queue& q = a.queue();
    struct Kind { const char* name; std::function<void()> f; const char* what = nullptr; size_t n = 0; };
    const Kind kinds[] = {
        {"idle", [] { std::this_thread::sleep_for(std::chrono::milliseconds(1)); }},
        {"h2d 64 MiB pageable", [&] { q.memcpy(d, pageable.data(), kBig).wait(); }},
        {"h2d 64 MiB pinned", [&] { q.memcpy(d, pinned, kBig).wait(); }},
        {"d2h 64 MiB pageable", [&] { q.memcpy(pageable.data(), d, kBig).wait(); }},
        {"d2h 64 MiB pinned", [&] { q.memcpy(pinned, d, kBig).wait(); }},
        {"d2h 10 KiB pageable", [&] { q.memcpy(pageable.data(), d, kSmall).wait(); }},
        {"d2h 10 KiB pinned", [&] { q.memcpy(pinned, d, kSmall).wait(); }},
        {"d2h 10 KiB a.d2h", [&] { a.d2h(pageable.data(), d, kSmall); }},   // the stage's kernel path
        {"h2d 10 KiB pageable", [&] { q.memcpy(d, pageable.data(), kSmall).wait(); }},
        {"h2d 10 KiB pinned", [&] { q.memcpy(d, pinned, kSmall).wait(); }},
        {"d2h 4 B pageable", [&] { q.memcpy(pageable.data(), d, 4).wait(); }},   // a sampled token
        {"d2h 4 B pinned", [&] { q.memcpy(pinned, d, 4).wait(); }},
        {"d2h 4 B a.d2h", [&] { a.d2h(pageable.data(), d, 4); }},
        {"h2d 4 B pageable", [&] { q.memcpy(d, pageable.data(), 4).wait(); }},
    };
    std::vector<Kind> all(std::begin(kinds), std::end(kinds));
    // the driver's CPU-copy limit: the largest size a pageable copy still runs with no engine
    for (size_t n : {size_t(64), size_t(256), size_t(1024), size_t(2048), size_t(4096), size_t(8192)})
        all.push_back({nullptr, [&, n] { q.memcpy(pageable.data(), d, n).wait(); }, "d2h pageable", n});
    for (size_t n : {size_t(65536), size_t(262144), size_t(524288), size_t(1) << 20, size_t(2) << 20, size_t(4) << 20})
        all.push_back({nullptr, [&, n] { q.memcpy(d, pageable.data(), n).wait(); }, "h2d pageable", n});
    for (const Kind& k : all) {
        const auto c0 = drm_cycles();
        const auto t0 = std::chrono::steady_clock::now();
        uint64_t n = 0;
        while (std::chrono::steady_clock::now() - t0 < std::chrono::seconds(2)) { k.f(); ++n; }
        const auto c1 = drm_cycles();
        const std::string name = k.name ? k.name : std::string(k.what) + " " + std::to_string(k.n) + " B";
        std::printf("%-22s %9llu calls %8.2f us/call  engine cycles:", name.c_str(), (unsigned long long)n, 2e6 / double(n));
        for (const auto& [cls, v] : c1) {
            const auto it = c0.find(cls);
            const uint64_t d = v - (it == c0.end() ? 0 : it->second);
            if (d) std::printf(" %s %llu", cls.c_str(), (unsigned long long)d);
        }
        std::printf("\n");
    }
    a.free(d);
    return 0;
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--engines") return engines(argc > 2 ? uint32_t(std::atoi(argv[2])) : 1);
    ie::DeviceFleet fleet;
    if (auto e = fleet.init(2, "B70", /*p2p=*/false, /*shared_ctx=*/false); !e.empty()) {
        std::printf("SKIP: %s\n", e.c_str());
        return 77;
    }
    ie::DeviceAllocator& a = fleet.dev(0);
    ie::DeviceAllocator other;   // another context on the same card (measured: its host USM copies at pinned speed)
    if (auto e = other.init("B70", 0); !e.empty()) { std::printf("SKIP: %s\n", e.c_str()); return 77; }
    constexpr size_t kMax = 1 << 20;
    void* d = a.malloc(kMax);
    std::vector<uint8_t> pageable(kMax, 1);
    auto* own = static_cast<uint8_t*>(a.malloc_host(kMax));
    auto* foreign = static_cast<uint8_t*>(other.malloc_host(kMax));
    if (!d || !own || !foreign) { std::printf("alloc failed\n"); return 1; }
    sycl::queue& q = a.queue();

    std::printf("%-10s %12s %12s %12s %12s %12s   (us per call, median)\n", "bytes", "pageable", "h2d/d2h", "h2d/d2h USM",
                "own pinned", "other ctx");
    for (size_t n : {size_t(4), size_t(10240), size_t(496640), size_t(kMax)}) {
        const double hp = median_us([&] { q.memcpy(d, pageable.data(), n).wait(); });
        const double hs = median_us([&] { a.h2d(d, pageable.data(), n); });
        const double hu = median_us([&] { a.h2d(d, foreign, n); });
        const double ho = median_us([&] { q.memcpy(d, own, n).wait(); });
        const double hf = median_us([&] { q.memcpy(d, foreign, n).wait(); });
        std::printf("h2d %-6zu %12.1f %12.1f %12.1f %12.1f %12.1f\n", n, hp, hs, hu, ho, hf);
        const double dp = median_us([&] { q.memcpy(pageable.data(), d, n).wait(); });
        const double ds = median_us([&] { a.d2h(pageable.data(), d, n); });
        const double du = median_us([&] { a.d2h(foreign, d, n); });
        const double dO = median_us([&] { q.memcpy(own, d, n).wait(); });
        const double df = median_us([&] { q.memcpy(foreign, d, n).wait(); });
        std::printf("d2h %-6zu %12.1f %12.1f %12.1f %12.1f %12.1f\n", n, dp, ds, du, dO, df);
    }

    if (fleet.size() >= 2) {
        void* d1 = fleet.dev(1).malloc(kMax);
        std::printf("\ncopy_across 0 -> 1 (us per call, median)\n");
        for (size_t n : {size_t(10240), size_t(496640)}) {
            const double old_way = median_us([&] {   // what copy_across did before: a fresh vector bounce
                std::vector<uint8_t> h(n);
                fleet.dev(0).queue().memcpy(h.data(), d, n).wait();
                fleet.dev(1).queue().memcpy(d1, h.data(), n).wait();
            });
            const double new_way = median_us([&] { fleet.copy_across(0, d1, 1, d, n); });
            std::printf("  %-8zu vector bounce %8.1f   staged %8.1f\n", n, old_way, new_way);
        }
        fleet.dev(1).free(d1);
    }
    a.free(d);
    return 0;
}
