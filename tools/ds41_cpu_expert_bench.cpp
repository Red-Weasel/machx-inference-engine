// tools/ds41_cpu_expert_bench.cpp — the CPU MXFP4 expert kernel on real experts (Phase 13 step 2,
// docs/deepseek41/34): experts per ms on a core set, cold (rotating over more experts than the
// caches hold), and one expert against the GPU's own M = 1 expert path on the same bytes.
//   usage: ie-ds41-cpu-expert-bench <model> [n_experts=96] [threads=8,12,20]
//   pin the team with the environment (OMP_PLACES / OMP_PROC_BIND, or taskset) -- the kernel
//   only sizes it.
#include "ie/cpu_moe_mxfp4.hpp"
#include "ie/deepseek41.hpp"
#include "ie/deepseek41_experts.hpp"
#include "ie/deepseek4_experts.hpp"
#include "ie/expert_stream.hpp"

#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>

#include <chrono>
#include <dlfcn.h>
#include <sys/mman.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <time.h>
#include <vector>

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : std::string(std::getenv("HOME")) + "/models/DeepSeek-V4.1-Flash";
    const uint32_t n_exp = argc > 2 ? uint32_t(std::atoi(argv[2])) : 96u;
    std::vector<int> teams = {8, 12, 20};
    if (argc > 3) { teams.clear(); std::string s = argv[3]; size_t p = 0; while (p < s.size()) { size_t q = s.find(',', p); if (q == std::string::npos) q = s.size(); teams.push_back(std::atoi(s.substr(p, q - p).c_str())); p = q + 1; } }
    ie::DeepSeek41Model m; if (auto e = m.load(dir); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
    const auto& c = m.config(); const uint32_t H = c.dim, EF = c.moe_inter_dim;
    ie::Ds4SlotLayout lay; if (auto e = ie::ds41_slot_layout(H, EF, lay); !e.empty()) { std::fprintf(stderr, "layout: %s\n", e.c_str()); return 1; }
    std::printf("expert slot %.1f MiB (gate/up K %u N %u, down K %u N %u); packing %u experts of layer 0 (%.1f GiB)\n", lay.bytes / 1048576.0, H, EF, EF, H, n_exp, n_exp * lay.bytes / 1073741824.0);
    std::vector<uint8_t*> slots(n_exp);
    const auto& Lw = m.layers()[0];
    // IE_DS41_BENCH_PINNED=1: the slots in pinned USM host memory on card 0's context, as the tier's
    // arena is -- the CPU's read rate on that mapping is the question
    const bool pinned = std::getenv("IE_DS41_BENCH_PINNED") != nullptr;
    std::unique_ptr<sycl::queue> pq;
    if (pinned) {
        for (const auto& pl : sycl::platform::get_platforms()) { if (pl.get_backend() != sycl::backend::ext_oneapi_level_zero) continue; for (const auto& d : pl.get_devices(sycl::info::device_type::gpu)) if (d.get_info<sycl::info::device::name>().find("Arc") != std::string::npos && !pq) pq = std::make_unique<sycl::queue>(sycl::context(d), d); }
        if (!pq) { std::fprintf(stderr, "no Arc GPU for the pinned allocation\n"); return 1; }
        std::printf("slots in PINNED USM host memory\n");
    }
    // IE_DS41_BENCH_ARENA_GIB=N (with the pinned option): the slots spread over N GiB of pinned
    // arena in 4 GiB pieces (the tier's per-layer allocations), every page touched -- the tier's
    // shape, whose page tables no cache holds
    const size_t arena_gib = std::getenv("IE_DS41_BENCH_ARENA_GIB") ? size_t(std::atol(std::getenv("IE_DS41_BENCH_ARENA_GIB"))) : 0;
    std::vector<uint8_t*> arena;
    // IE_DS41_BENCH_IMPORT=1 (with the arena): the pieces are anonymous memory with MADV_HUGEPAGE
    // (THP), handed to the driver with zexDriverImportExternalPointer so the GPU can DMA from
    // them -- the page-table fix candidate for the tier's arena (docs/deepseek41/35 step 3)
    const bool import = std::getenv("IE_DS41_BENCH_IMPORT") != nullptr;
    ze_driver_handle_t zdrv = nullptr; ze_result_t (*ze_import)(ze_driver_handle_t, void*, size_t) = nullptr;
    if (pinned && arena_gib && import) {
        zdrv = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(pq->get_context().get_platform());
        void* lib = dlopen("libze_loader.so.1", RTLD_NOW);
        auto getext = lib ? reinterpret_cast<ze_result_t (*)(ze_driver_handle_t, const char*, void**)>(dlsym(lib, "zeDriverGetExtensionFunctionAddress")) : nullptr;
        if (!getext || getext(zdrv, "zexDriverImportExternalPointer", reinterpret_cast<void**>(&ze_import)) != ZE_RESULT_SUCCESS || !ze_import) {
            std::fprintf(stderr, "zexDriverImportExternalPointer is not available in this driver\n"); return 1; }
    }
    if (pinned && arena_gib) {
        for (size_t g = 0; g < arena_gib; g += 4) {
            const size_t sz = size_t(4) << 30; uint8_t* a = nullptr;
            if (import) {
                void* m = mmap(nullptr, sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                if (m == MAP_FAILED) { std::fprintf(stderr, "arena mmap failed at %zu GiB\n", g); return 1; }
                madvise(m, sz, MADV_HUGEPAGE); std::memset(m, 1, sz);
                if (const auto r = ze_import(zdrv, m, sz); r != ZE_RESULT_SUCCESS) { std::fprintf(stderr, "import failed at %zu GiB: 0x%x\n", g, unsigned(r)); return 1; }
                a = static_cast<uint8_t*>(m);
            } else {
                a = static_cast<uint8_t*>(sycl::malloc_host(sz, *pq));
                if (!a) { std::fprintf(stderr, "arena alloc failed at %zu GiB\n", g); return 1; }
                std::memset(a, 1, sz);
            }
            arena.push_back(a);
        }
        std::printf("slots spread over %zu x 4 GiB of %s arena, every page touched\n", arena.size(), import ? "imported THP" : "pinned");
        if (FILE* f = std::fopen("/proc/self/smaps_rollup", "r")) { char line[256]; while (std::fgets(line, sizeof line, f)) if (std::strstr(line, "AnonHugePages")) std::printf("  %s", line); std::fclose(f); }
    }
    for (uint32_t e = 0; e < n_exp; ++e) {
        void* p = nullptr;
        if (!arena.empty()) p = arena[e % arena.size()] + (size_t(e / arena.size()) * ((size_t(4) << 30) / (n_exp / arena.size() + 1))) / 4096 * 4096;
        else if (pinned) p = sycl::malloc_host(lay.bytes, *pq);
        else if (posix_memalign(&p, 64, lay.bytes) != 0) p = nullptr;
        if (!p) { std::fprintf(stderr, "alloc\n"); return 1; }
        slots[e] = static_cast<uint8_t*>(p);
        if (auto er = ie::ds41_slot_pack(lay, Lw.exp_w1[e], Lw.exp_w3[e], Lw.exp_w2[e], p); !er.empty()) { std::fprintf(stderr, "pack %u: %s\n", e, er.c_str()); return 1; }
    }
    std::mt19937 rng(3); std::vector<float> x(H); for (auto& v : x) v = float(int(rng() % 2001) - 1000) / 1000.f;
    std::vector<float> scratch(2 * EF), out(H);
    // IE_DS41_BENCH_GAP_US=N: the calling thread sleeps N us before every expert, as the tier's worker
    // does between layers (the team idles, the cores cool); only the experts' own time is counted
    const long gap_us = std::getenv("IE_DS41_BENCH_GAP_US") ? std::atol(std::getenv("IE_DS41_BENCH_GAP_US")) : 0;
    if (gap_us > 0) std::printf("a %ld us idle gap before every expert (the experts' own time counted)\n", gap_us);
    for (int nt : teams) {
        ie::cpu_expert_mxfp4(slots[0], lay, x.data(), scratch.data(), out.data(), c.swiglu_limit, nt);   // warm the team
        const uint32_t rounds = 3; double best = 1e9;
        for (uint32_t r = 0; r < rounds; ++r) {
            double work = 0;
            for (uint32_t e = 0; e < n_exp; ++e) {
                if (gap_us > 0) { const timespec ts{gap_us / 1000000, (gap_us % 1000000) * 1000}; nanosleep(&ts, nullptr); }
                const auto t0 = std::chrono::steady_clock::now();
                ie::cpu_expert_mxfp4(slots[e], lay, x.data(), scratch.data(), out.data(), c.swiglu_limit, nt);
                work += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            }
            best = std::min(best, work);
        }
        std::printf("threads %2d: %6.2f ms per expert = %5.2f experts/ms = %5.1f GB/s of slot bytes (best of %u rounds over %u experts)\n", nt, best / n_exp, n_exp / best, n_exp * lay.bytes / (best / 1000.0) / 1e9, rounds, n_exp);
    }

    // one expert against the GPU's own M = 1 path on the same slot bytes (the tier's ds4_gemm path
    // with Q8 activations, so the bar is the quantisation of x, ~1e-2; the CPU keeps x fp32)
    std::vector<sycl::device> devs;
    for (const auto& p : sycl::platform::get_platforms()) { if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) continue; for (const auto& d : p.get_devices(sycl::info::device_type::gpu)) if (d.get_info<sycl::info::device::name>().find("Arc") != std::string::npos) devs.push_back(d); }
    if (devs.empty()) { std::printf("(no GPU: the GPU comparison skipped)\n"); return 0; }
    sycl::queue q(sycl::context(devs[0]), devs[0], sycl::property_list{sycl::property::queue::in_order{}});
    uint8_t* d_slot = sycl::malloc_device<uint8_t>(lay.bytes, q); q.memcpy(d_slot, slots[0], lay.bytes).wait();
    if (!arena.empty()) {   // the DMA rate from the arena's memory vs a fresh pinned buffer, 1 GiB each
        const size_t gib = size_t(1) << 30; uint8_t* d_big = sycl::malloc_device<uint8_t>(gib, q);
        uint8_t* fresh = sycl::malloc_host<uint8_t>(gib, q); std::memset(fresh, 2, gib);
        for (int pass = 0; pass < 2; ++pass) for (const uint8_t* src : {static_cast<const uint8_t*>(arena[0]), static_cast<const uint8_t*>(fresh)}) {
            const auto t0 = std::chrono::steady_clock::now(); q.memcpy(d_big, src, gib).wait();
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (pass) std::printf("H2D 1 GiB from the %s: %.1f ms = %.1f GB/s\n", src == arena[0] ? (import ? "imported THP arena" : "pinned arena") : "fresh pinned buffer", ms, gib / (ms / 1000.0) / 1e9);
        }
        std::vector<uint8_t> back(4096); q.memcpy(back.data(), d_big, 4096).wait();
        std::printf("  (the arena's bytes read back on the GPU: %s)\n", back[0] == 2 ? "the fresh buffer's" : back[0] == 1 ? "intact" : "WRONG");
        sycl::free(d_big, q); sycl::free(fresh, q);
    }
    ie::DS4ExpertWorkspace ws; if (auto e = ie::ds4_expert_ws_alloc(q, H, EF, ws); !e.empty()) { std::fprintf(stderr, "ws: %s\n", e.c_str()); return 1; }
    float* d_x = sycl::malloc_device<float>(H, q); q.memcpy(d_x, x.data(), H * 4).wait();
    float* d_y = sycl::malloc_device<float>(H, q);
    ie::DS4ExpertBank bg = ie::ds4_slot_bank(lay.gate, d_slot), bu = ie::ds4_slot_bank(lay.up, d_slot), bd = ie::ds4_slot_bank(lay.down, d_slot);
    const int32_t idx[1] = {0}; const float w1[1] = {1.f};
    std::string err = ie::ds4_experts_forward(q, bg, bu, bd, d_x, idx, w1, d_y, 1, H, EF, 1, c.swiglu_limit, ws, true);
    if (!err.empty()) { std::printf("(GPU path: %s -- comparison skipped)\n", err.c_str()); return 0; }
    std::vector<float> gy(H); q.memcpy(gy.data(), d_y, H * 4).wait();
    ie::cpu_expert_mxfp4(slots[0], lay, x.data(), scratch.data(), out.data(), c.swiglu_limit, 8);
    double mx = 0, sc = 0; for (uint32_t i = 0; i < H; ++i) { mx = std::max(mx, std::fabs(double(out[i]) - gy[i])); sc = std::max(sc, std::fabs(double(gy[i]))); }
    std::printf("expert 0 of layer 0: CPU fp32 vs the GPU M = 1 path (Q8 activations): max rel %.3e over %u outputs\n", sc > 0 ? mx / sc : 0.0, H);
    ie::ds4_expert_ws_free(q, ws); sycl::free(d_slot, q); sycl::free(d_x, q); sycl::free(d_y, q);
    if (!arena.empty()) for (auto* a : arena) { if (import) munmap(a, size_t(4) << 30); else sycl::free(a, *pq); } else for (auto* p : slots) { if (pinned) sycl::free(p, *pq); else std::free(p); }
    return 0;
}
