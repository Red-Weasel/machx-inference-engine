// tests/unit/fill_overlap_bench.cpp — a bench, not a CTest (2026-11-10): does an expert-cache miss fill (host USM -> device,
// ~3 MiB, on a second in-order queue) run BESIDE a compute kernel, or do the two serialize on the card? And which engine
// runs the fill? The decode / verify MoE body waits for its fills before the routed experts' kernels; launching the cached
// experts' kernels first (IE_Q4E_HITS_FIRST) only pays if the answer is "beside".
//
// usage: fill_overlap_bench [ordinal]
// Prints, for n = 1, 3, 10 fills of 3 x 1 MiB: the fills alone, a compute kernel alone, fills -> barrier -> kernel (the
// serial order), and kernel beside the fills (kernel submitted before the barrier). "beside" near max(fills, kernel) =
// they overlap; near the sum = they serialize. Then the engine classes (xe fdinfo drm-cycles) of the fill loop.
#include "ie/allocator.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dirent.h>
#include <fstream>
#include <functional>
#include <map>
#include <string>
#include <thread>
#include <vector>

namespace {
double median_ms(const std::function<void()>& f, int n = 60) {
    for (int i = 0; i < 15; ++i) f();
    std::vector<double> t(n);
    for (double& v : t) {
        const auto a = std::chrono::steady_clock::now();
        f();
        v = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
    }
    std::nth_element(t.begin(), t.begin() + t.size() / 2, t.end());
    return t[t.size() / 2];
}
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
}  // namespace

int main(int argc, char** argv) {
    // --hog <MiB> <seconds> [ordinal]: hold <MiB> of device memory on the card for <seconds> (touched, so it is backed) --
    // another process's memory for the engine's VRAM guard test (Qwen4ExpModel::ecache_ensure_free).
    if (argc > 3 && std::string(argv[1]) == "--hog") {
        ie::DeviceAllocator h;
        if (auto e = h.init("B70", argc > 4 ? uint32_t(std::atoi(argv[4])) : 1); !e.empty()) { std::printf("SKIP: %s\n", e.c_str()); return 77; }
        const size_t mib = size_t(std::atol(argv[2]));
        std::vector<void*> held;
        for (size_t got = 0; got < mib; got += 64) { void* p = h.malloc(64ull << 20); if (!p) break; h.queue().memset(p, 1, 64ull << 20); held.push_back(p); }
        h.queue().wait();
        std::printf("hog: %zu MiB held for %s s\n", held.size() * 64, argv[3]); std::fflush(stdout);
        std::this_thread::sleep_for(std::chrono::seconds(std::atol(argv[3])));
        for (void* p : held) h.free(p);
        return 0;
    }
    ie::DeviceAllocator a;
    if (auto e = a.init("B70", argc > 1 ? uint32_t(std::atoi(argv[1])) : 1); !e.empty()) { std::printf("SKIP: %s\n", e.c_str()); return 77; }
    sycl::queue& q = a.queue();
    sycl::queue cq(q.get_context(), q.get_device(), sycl::property_list{sycl::property::queue::in_order{}});
    constexpr size_t kSlice = 1ull << 20, kSlot = 3 * kSlice, kSlots = 64, kBank = 256ull << 20, kWork = 256ull << 20;
    auto* cache = static_cast<uint8_t*>(a.malloc(kSlots * kSlot));
    auto* bank = static_cast<uint8_t*>(a.malloc_host(kBank));
    auto* work = static_cast<uint32_t*>(a.malloc(kWork));
    auto* out = static_cast<uint32_t*>(a.malloc(4096 * sizeof(uint32_t)));
    if (!cache || !bank || !work || !out) { std::printf("alloc failed\n"); return 1; }
    for (size_t i = 0; i < kBank; i += 4096) bank[i] = uint8_t(i >> 12);
    q.memset(work, 1, kWork).wait();
    q.memset(cache, 0, kSlots * kSlot).wait();
    // a memory-bound kernel like a group of expert GEMVs: 4096 work-items each summing `words` words of `work`
    auto kernel = [&](size_t words) {
        const uint32_t* w = work; uint32_t* o = out;
        return q.parallel_for(sycl::range<1>(4096), [=](sycl::id<1> i) {
            uint32_t s = 0;
            const uint32_t* p = w + i[0] * words;
            for (size_t k = 0; k < words; ++k) s += p[k];
            o[i[0]] = s;
        });
    };
    uint32_t next = 0;
    auto fills = [&](uint32_t n, bool one_copy) {   // n misses: 3 slices each from the pinned bank into the next slots
        for (uint32_t m = 0; m < n; ++m, ++next) {
            uint8_t* dst = cache + (next % kSlots) * kSlot;
            const uint8_t* src = bank + (size_t(next) * 7919 % 80) * kSlot;
            if (one_copy) cq.memcpy(dst, src, kSlot);
            else for (int s = 0; s < 3; ++s) cq.memcpy(dst + s * kSlice, src + s * kSlice, kSlice);
        }
    };
    auto barrier = [&] { q.ext_oneapi_submit_barrier({cq.ext_oneapi_submit_barrier()}); };
    auto tiny = [&] { uint32_t* o = out; return q.single_task([=] { o[0] += 1; }); };
    for (size_t words : {size_t(1024), size_t(4096), size_t(16384)}) {   // 16 / 64 / 256 MiB read
        const double k_ms = median_ms([&] { kernel(words).wait(); });
        std::printf("== compute kernel reading %zu MiB: %.3f ms alone ==\n", words * 4096 * 4 >> 20, k_ms);
        for (uint32_t n : {1u, 3u, 10u})
            for (bool one : {false, true}) {
                const double f_ms = median_ms([&] { fills(n, one); barrier(); tiny().wait(); });
                const double s_ms = median_ms([&] { fills(n, one); barrier(); kernel(words).wait(); });
                const double o_ms = median_ms([&] { fills(n, one); kernel(words); barrier(); tiny().wait(); });
                std::printf("   %2u fill(s) %s: alone %.3f ms (%.3f a fill) | fills then kernel %.3f | kernel beside the fills %.3f | "
                            "sum %.3f, max %.3f\n", n, one ? "of 1 x 3 MiB" : "of 3 x 1 MiB", f_ms, f_ms / n, s_ms, o_ms, f_ms + k_ms,
                            std::max(f_ms, k_ms));
            }
    }
    // ---- a decode / verify stage in miniature (2026-11-10 evening): 24 "layers", each = a router kernel, its result
    // copied to pinned host memory and waited for (the route wait), the job table uploaded, the cached experts' kernel
    // (`hit_words` of work), n fills, the barrier, the filled experts' small kernel, a reduce kernel. What do n fills a
    // layer ADD to the stage, against the best case max(0, fills - hits)? The variants move one thing at a time.
    {
        auto* h_route = static_cast<uint16_t*>(a.malloc_host(4096 * 2));
        auto* h_table = static_cast<int32_t*>(a.malloc_host(4096 * 4));
        auto* d_route = static_cast<uint16_t*>(a.malloc(4096 * 2));
        auto* d_table = static_cast<int32_t*>(a.malloc(4096 * 4));
        q.memset(d_route, 0, 4096 * 2).wait();
        for (int i = 0; i < 4096; ++i) h_table[i] = i;
        auto router = [&] { uint16_t* o = d_route; const uint32_t* w = work; return q.parallel_for(sycl::range<1>(2048), [=](sycl::id<1> i) { o[i[0]] = uint16_t(w[i[0]]); }); };
        auto small = [&](size_t words) {   // the filled experts' jobs: reads the slots just filled
            const uint8_t* c = cache; uint32_t* o = out + 2048;
            return q.parallel_for(sycl::range<1>(256), [=](sycl::id<1> i) {
                uint32_t s2 = 0; const uint8_t* p2 = c + i[0] * words;
                for (size_t k = 0; k < words; ++k) s2 += p2[k];
                o[i[0]] = s2;
            });
        };
        // mode bits: 1 = fills submitted BEFORE the hits' kernel, 2 = no route copy / wait (the host never waits),
        //            4 = no table upload, 8 = the hits' kernel AFTER the barrier (the old order)
        auto stage = [&](uint32_t n, size_t hit_words, unsigned mode) {
            for (int L = 0; L < 24; ++L) {
                router();
                if (!(mode & 2)) q.memcpy(h_route, d_route, 2048 * 2).wait();
                if (!(mode & 4)) q.memcpy(d_table, h_table, 512 * 4);
                if (mode & 8) { fills(n, false); if (n) barrier(); kernel(hit_words); small(4096); }
                else {
                    if (mode & 1) fills(n, false);
                    kernel(hit_words);
                    if (!(mode & 1)) fills(n, false);
                    if (n) barrier();
                    small(4096);
                }
                tiny();
            }
            q.wait();
        };
        for (size_t hit_words : {size_t(2048), size_t(8192)}) {   // ~32 / ~128 MiB read by the hits' kernel
            const double k_ms = median_ms([&] { kernel(hit_words).wait(); }, 40);
            std::printf("== a 24-layer stage in miniature; the hits' kernel alone %.3f ms ==\n", k_ms);
            struct V { const char* name; unsigned mode; };
            for (const V& v : {V{"hits first, fills after (the engine now)", 0u}, V{"hits first, fills before", 1u}, V{"old order (all behind the fills)", 8u},
                               V{"no route wait", 2u}, V{"no table upload", 4u}, V{"no route wait, no table", 6u}}) {
                const double t0 = median_ms([&] { stage(0, hit_words, v.mode); }, 20);
                const double t1 = median_ms([&] { stage(1, hit_words, v.mode); }, 20);
                const double t3 = median_ms([&] { stage(3, hit_words, v.mode); }, 20);
                std::printf("   %-42s 0 fills %.2f ms | 1 fill a layer %.2f (+%.3f a layer; best +%.3f) | 3 fills %.2f (+%.3f; best +%.3f)\n",
                            v.name, t0, t1, (t1 - t0) / 24, std::max(0.0, 0.118 - k_ms), t3, (t3 - t0) / 24, std::max(0.0, 0.354 - k_ms));
            }
        }
    }
    // which engine runs the fills
    for (bool one : {false, true}) {
        const auto c0 = drm_cycles();
        const auto t0 = std::chrono::steady_clock::now();
        while (std::chrono::steady_clock::now() - t0 < std::chrono::seconds(2)) { fills(10, one); cq.wait(); }
        const auto c1 = drm_cycles();
        std::printf("engine cycles of 2 s of fills %s:", one ? "of 1 x 3 MiB" : "of 3 x 1 MiB");
        for (auto& [k, v] : c1) { const auto it = c0.find(k); const uint64_t d = v - (it == c0.end() ? 0 : it->second); if (d) std::printf(" %s=%llu", k.c_str(), (unsigned long long)d); }
        std::printf("\n");
    }
    return 0;
}
