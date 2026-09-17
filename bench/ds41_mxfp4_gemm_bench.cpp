// bench/ds41_mxfp4_gemm_bench.cpp -- Phase 40 (docs/deepseek41/80): the grouped M = 1 MXFP4 expert GEMM at the V4.1
// decode shapes (six single-row jobs per launch, like a layer's top-6), streamed from a 64-expert bank so no launch
// re-reads L2, with a hash of the outputs so IE_DS4_MXFP4_PIPE=1 and =0 can be compared bit for bit across runs.
//   usage: ds41-mxfp4-gemm-bench [iters=60]
#include "ie/deepseek4_experts.hpp"
#include "ie/quant_blocks.hpp"
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    const int iters = argc > 1 ? std::atoi(argv[1]) : 60;
    sycl::device dev;
    for (auto& d : sycl::device::get_devices(sycl::info::device_type::gpu))
        if (d.get_info<sycl::info::device::name>().find("Arc") != std::string::npos) { dev = d; break; }
    sycl::queue q(dev, sycl::property::queue::in_order{});
    std::mt19937 rng(0xD5E4);
    const uint32_t E = 64, JOBS = 6;
    struct Shape { const char* name; uint32_t K, N; };
    std::printf("device %s, %d launches per point, %u jobs (M = 1) per launch, IE_DS4_MXFP4_PIPE=%s\n", dev.get_info<sycl::info::device::name>().c_str(), iters, JOBS, std::getenv("IE_DS4_MXFP4_PIPE") ? std::getenv("IE_DS4_MXFP4_PIPE") : "(default)");
    for (const Shape& s : { Shape{"gate/up", 5120, 2304}, Shape{"down   ", 2304, 5120} }) {
        ie::DS4ExpertBank b{}; b.dtype = ie::DType::kMXFP4; b.K = s.K; b.N = s.N; b.E = E;
        const uint64_t qs = uint64_t(s.N) * (s.K / 2), ep = uint64_t(s.N) * (s.K / 32);
        b.mx_qs_stride = qs; b.mx_e_stride = ep;
        std::vector<uint8_t> hq(uint64_t(E) * qs), he(uint64_t(E) * ep);
        for (auto& v : hq) v = uint8_t(rng());
        for (auto& v : he) v = uint8_t(123 + rng() % 6);
        b.mx_qs = sycl::malloc_device<uint8_t>(hq.size(), q); b.mx_e = sycl::malloc_device<uint8_t>(he.size(), q);
        q.memcpy(b.mx_qs, hq.data(), hq.size()); q.memcpy(b.mx_e, he.data(), he.size()).wait();
        const uint32_t bpc = s.K / 32;
        std::vector<ie::block_q8_1x> hx(uint64_t(JOBS) * bpc);
        for (auto& blk : hx) { blk.d = 0.002f + float(rng() % 64) * 1e-5f; blk.s = 0.f; for (auto& c : blk.qs) c = int8_t(rng()); }
        auto* x = sycl::malloc_device<ie::block_q8_1x>(hx.size(), q); q.memcpy(x, hx.data(), hx.size() * sizeof(ie::block_q8_1x)).wait();
        auto* y = sycl::malloc_device<sycl::half>(uint64_t(JOBS) * s.N, q);
        ie::DS4GemmGroupWs gws{}; if (auto e = ie::ds4_gemm_group_ws_alloc(q, JOBS, gws); !e.empty()) { std::printf("ws: %s\n", e.c_str()); return 1; }
        uint32_t rot = 0;
        auto launch = [&] {
            std::vector<ie::DS4GemmJob> jobs(JOBS);
            for (uint32_t j = 0; j < JOBS; ++j) { jobs[j].bank = b; jobs[j].e = (rot * JOBS + j) % E; jobs[j].M = 1; jobs[j].x_q8 = x + uint64_t(j) * bpc; jobs[j].y = y + uint64_t(j) * s.N; }
            ++rot;
            if (auto e = ie::ds4_expert_gemm_q8_grouped(q, jobs.data(), JOBS, gws); !e.empty()) { std::printf("gemm: %s\n", e.c_str()); std::exit(1); }
        };
        for (int i = 0; i < 4; ++i) launch(); q.wait();
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) launch();
        q.wait();
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
        const double bytes = double(JOBS) * (qs + ep);
        // the hash: rotation reset so both PIPE settings hash the same expert set
        rot = 0; launch(); q.wait();
        std::vector<uint16_t> hy(uint64_t(JOBS) * s.N); q.memcpy(hy.data(), y, hy.size() * 2).wait();
        uint64_t hsh = 1469598103934665603ull; for (uint16_t v : hy) { hsh ^= v; hsh *= 1099511628211ull; }
        std::printf("  %s K %5u N %5u: %8.1f us per launch = %6.1f GB/s of expert bytes   (%.1f ms/token over 40 layers)   hash %016llx\n", s.name, s.K, s.N, us, bytes / us / 1e3, us * 40 / 1000.0, (unsigned long long)hsh);
        ie::ds4_gemm_group_ws_free(q, gws); sycl::free(b.mx_qs, q); sycl::free(b.mx_e, q); sycl::free(x, q); sycl::free(y, q);
    }
    return 0;
}
