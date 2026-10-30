// tools/q8_0_kernel_bench.cpp — Flash-Next expert-shape microbench for the
// Q8_0 grouped-tiles gate/up kernels: naive (gemv_q8_0_q8s_grouped_tiles)
// vs dp4a (gemv_q8_0_q8s_grouped_tiles_dp4a). Correctness diff first (the
// dp4a kernel regroups the per-lane partials, so expect small fp differences,
// not bit equality), then a T sweep at the bank shapes
// (gate/up K=2048 -> N=1536; the down shape K=1536 -> N=2048 uses the same
// kernel family with swapped operands).
//
// usage: ie-q8-0-kernel-bench [gpu]
#include "ie/allocator.hpp"
#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"
#include "ie/qwen4_quant.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace ie;

namespace {

struct Ctx {
    DeviceAllocator alloc;
    sycl::queue& q = alloc.queue();
};

double bench_once(Ctx& c, bool dp4a, const void* xq, const uint8_t* slot_base,
                  uint64_t slot_bytes, uint64_t gate_off, uint64_t up_off,
                  const int32_t* jobs, sycl::half* g, sycl::half* u,
                  uint32_t K, uint32_t N, uint32_t J) {
    auto run = [&] {
        if (dp4a)
            gemv_q8_0_q8s_grouped_tiles_dp4a(c.q, xq, slot_base, slot_bytes,
                                             gate_off, up_off, jobs, g, u, K, N, J, {});
        else
            gemv_q8_0_q8s_grouped_tiles(c.q, xq, slot_base, slot_bytes,
                                        gate_off, up_off, jobs, g, u, K, N, J, {});
    };
    for (int i = 0; i < 5; ++i) run();
    c.q.wait();
    const int iters = 30;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) run();
    c.q.wait();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / iters;
}

int run_shape(Ctx& c, uint32_t K, uint32_t N, const char* what) {
    const uint32_t J = 256;                       // tiles per call (one wave of expert jobs)
    const uint64_t slice_bytes = uint64_t(K / 32) * sizeof(block_q8_0) * N;
    const uint64_t gate_off = 0, up_off = slice_bytes;
    const uint64_t slot_bytes = 2 * slice_bytes;  // gate | up in one slot

    // 8 distinct expert slots (defeats L2 across jobs), sane fp16 headers.
    std::vector<uint8_t> hw(slot_bytes * 8);
    std::mt19937 rng(42);
    for (auto& b : hw) b = uint8_t(rng());
    for (size_t off = 0; off + sizeof(block_q8_0) <= hw.size(); off += sizeof(block_q8_0)) {
        const sycl::half d(0.002f + 0.01f * float(rng() & 0xFF) / 255.f);
        std::memcpy(hw.data() + off, &d, 2);
    }
    auto* dW = static_cast<uint8_t*>(c.alloc.malloc(hw.size()));
    c.q.memcpy(dW, hw.data(), hw.size()).wait();

    // Activation mega buffer: 16 q8_1s rows per tile job.
    const uint32_t blocks_per_col = K / 32;
    const uint32_t row_blocks = blocks_per_col;
    std::vector<uint8_t> hx(sizeof(block_q8_1s) * row_blocks * 16 * 8);
    for (auto& b : hx) b = uint8_t(rng());
    // Sane q8_1s headers: d, s0, s1 floats (offsets 0/4/8), pad at 12.
    for (size_t off = 0; off + sizeof(block_q8_1s) <= hx.size(); off += sizeof(block_q8_1s)) {
        const float d = 0.001f + 0.002f * float(rng() & 0xFF) / 255.f;
        float s0 = 0.f, s1 = 0.f;
        for (int i = 0; i < 16; ++i) {
            const int8_t q = int8_t(hx[off + 16 + i]);
            (i < 8 ? s0 : s1) += d * float(q);
        }
        // NOTE: q8_1s's s0/s1 are per-16-element half sums over the FIRST and
        // SECOND 16 qs bytes (see quant_blocks.hpp); the Q8_0 kernels never
        // read them, so exact values don't matter — fill plausibly anyway.
        std::memcpy(hx.data() + off + 0, &d, 4);
        std::memcpy(hx.data() + off + 4, &s0, 4);
        std::memcpy(hx.data() + off + 8, &s1, 4);
    }
    auto* dX = static_cast<uint8_t*>(c.alloc.malloc(hx.size()));
    c.q.memcpy(dX, hx.data(), hx.size()).wait();

    auto* dG = static_cast<sycl::half*>(c.alloc.malloc(uint64_t(16) * J * N * 2));
    auto* dU = static_cast<sycl::half*>(c.alloc.malloc(uint64_t(16) * J * N * 2));
    auto* dG2 = static_cast<sycl::half*>(c.alloc.malloc(uint64_t(16) * J * N * 2));
    auto* dU2 = static_cast<sycl::half*>(c.alloc.malloc(uint64_t(16) * J * N * 2));
    auto* dJobs = static_cast<int32_t*>(c.alloc.malloc(uint64_t(J) * 4 * 4));

    std::printf("\n[%s] shape [T,%u] x [%u,%u], slot %.2f MB, J=%u\n",
                what, K, K, N, slot_bytes / 1e6, J);
    std::printf("   T  naive ms   dp4a ms   speedup   W-GB/s(naive)  W-GB/s(dp4a)  maxdiff\n");
    for (uint32_t T : {1u, 2u, 4u, 8u, 16u}) {
        std::vector<int32_t> jobs(uint64_t(J) * 4);
        for (uint32_t j = 0; j < J; ++j) {
            jobs[j * 4 + 0] = int32_t((j * 7) % 8) * 16;   // x row base (8 distinct rows of 16)
            jobs[j * 4 + 1] = int32_t(T);
            jobs[j * 4 + 2] = int32_t(j % 8);              // slot
            jobs[j * 4 + 3] = int32_t(j * 16);             // y row base
        }
        c.q.memcpy(dJobs, jobs.data(), jobs.size() * 4).wait();

        const double ms_n = bench_once(c, false, dX, dW, slot_bytes, gate_off, up_off,
                                       dJobs, dG, dU, K, N, J);
        const double ms_d = bench_once(c, true, dX, dW, slot_bytes, gate_off, up_off,
                                       dJobs, dG2, dU2, K, N, J);

        std::vector<sycl::half> hg(16 * J * N), hu(16 * J * N), hg2(16 * J * N), hu2(16 * J * N);
        c.q.memcpy(hg.data(), dG, hg.size() * 2).wait();
        c.q.memcpy(hu.data(), dU, hu.size() * 2).wait();
        c.q.memcpy(hg2.data(), dG2, hg2.size() * 2).wait();
        c.q.memcpy(hu2.data(), dU2, hu2.size() * 2).wait();
        double mx = 0.0;
        double scale = 0.0;
        for (size_t i = 0; i < hg.size(); ++i) {
            mx = std::max(mx, double(std::abs(float(hg[i]) - float(hg2[i]))));
            mx = std::max(mx, double(std::abs(float(hu[i]) - float(hu2[i]))));
            scale = std::max(scale, double(std::abs(float(hg[i]))));
            scale = std::max(scale, double(std::abs(float(hu[i]))));
        }

        const uint64_t wbytes = uint64_t(J) * slot_bytes;   // gate+up weight bytes per call
        std::printf("  %2u  %8.3f  %8.3f  %7.2fx  %12.1f  %12.1f  %.2e%s\n",
                    T, ms_n, ms_d, ms_n / ms_d,
                    wbytes / ms_n / 1e6, wbytes / ms_d / 1e6, mx,
                    mx <= 5e-2 * scale + 1e-3 ? "" : "  << DIFF");
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const uint32_t ordinal = argc > 1 ? uint32_t(std::atoi(argv[1])) : 0;
    Ctx c;
    if (auto e = c.alloc.init("B70", ordinal); !e.empty()) { std::fprintf(stderr, "gpu: %s\n", e.c_str()); return 1; }
    std::printf("device: %s\n", c.q.get_device().get_info<sycl::info::device::name>().c_str());
    run_shape(c, 2048, 1536, "gate/up");
    run_shape(c, 1536, 2048, "down-ish");
    return 0;
}
