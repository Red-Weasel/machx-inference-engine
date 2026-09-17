// tools/q4k_kernel_bench.cpp — expert-shape microbench for the W4A8 int-dot
// kernels (2026-08-27): is gemv_q4_K_q8_batched ALU-bound in its T-loop at
// the MoE shapes ([T<=16, K=2560] x [K, N=640])? Sweep T at fixed shape:
// flat GB/s past small T = ALU-bound (the dp4a chain per weight byte scales
// with T) -> the fix class is int8 dpas, not more batching.
//
// usage: ie-q4k-bench [gpu]
#include "ie/allocator.hpp"
#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"
#include "ie/qwen4_quant.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace ie;

int main(int argc, char** argv) {
    const uint32_t ordinal = argc > 1 ? uint32_t(std::atoi(argv[1])) : 0;
    DeviceAllocator alloc;
    if (auto e = alloc.init("B70", ordinal); !e.empty()) { std::fprintf(stderr, "gpu: %s\n", e.c_str()); return 1; }
    sycl::queue& q = alloc.queue();

    const uint32_t K = 2560, N = 640, JOBS = 512;   // ~one MoE layer's experts
    const uint64_t wbytes = uint64_t(K / 256) * sizeof(block_q4_K) * N;
    const uint64_t q8row  = uint64_t(K / 32) * sizeof(block_q8_1x);

    // Synthetic weights: JOBS distinct expert slices (defeats L2 across jobs).
    std::vector<uint8_t> hw(wbytes * 8);   // 8 distinct slices rotated
    std::mt19937 rng(42);
    for (auto& b : hw) b = uint8_t(rng());
    // Sane block headers: random bytes make random-f16 d/dmin (Inf/NaN
    // poison both kernels and void the correctness diff). Overwrite with
    // small positive halves; scales/qs stay random.
    for (size_t off = 0; off + sizeof(block_q4_K) <= hw.size();
         off += sizeof(block_q4_K)) {
        const sycl::half d(0.002f + 0.01f * float(rng() & 0xFF) / 255.f);
        const sycl::half m(0.0005f + 0.002f * float(rng() & 0xFF) / 255.f);
        std::memcpy(hw.data() + off, &d, 2);
        std::memcpy(hw.data() + off + 2, &m, 2);
    }
    auto* dW = static_cast<uint8_t*>(alloc.malloc(hw.size()));
    q.memcpy(dW, hw.data(), hw.size()).wait();

    std::vector<uint8_t> hx(q8row * 16);
    for (auto& b : hx) b = uint8_t(rng());
    auto* dX = static_cast<uint8_t*>(alloc.malloc(hx.size()));
    q.memcpy(dX, hx.data(), hx.size()).wait();
    auto* dY = static_cast<sycl::half*>(alloc.malloc(uint64_t(16) * N * 2));

    std::printf("shape [T, %u] x [%u, %u], weight slice %.2f MB\n\n  T   ms/call   W-GB/s   rows-GB/s\n",
                K, K, N, wbytes / 1e6);
    for (uint32_t T : {1u, 2u, 4u, 8u, 16u}) {
        // warm
        gemv_q4_K_q8_batched(q, dX, dW, dY, K, N, T);
        q.wait();
        const int reps = 200;
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r)
            gemv_q4_K_q8_batched(q, dX, dW + (uint64_t(r & 7) * wbytes), dY, K, N, T);
        q.wait();
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count() / reps;
        std::printf(" %2u   %7.3f   %6.1f   %6.1f\n", T, ms,
                    wbytes / ms / 1e6, wbytes * T / ms / 1e6);
    }

    // Grouped-tiles at layer scale: JOBS jobs x T=16, distinct slots.
    {
        auto* jobs = static_cast<int32_t*>(alloc.malloc(uint64_t(JOBS) * 4 * 4));
        std::vector<int32_t> hj(JOBS * 4);
        for (uint32_t j = 0; j < JOBS; ++j) {
            hj[4 * j + 0] = 0;                    // x rows base
            hj[4 * j + 1] = 16;                   // rows
            hj[4 * j + 2] = int32_t(j & 7);       // slot (8 rotated slices)
            hj[4 * j + 3] = 0;                    // y base (overwrite, bench only)
        }
        q.memcpy(jobs, hj.data(), hj.size() * 4).wait();
        gemv_q4_K_q8_grouped_tiles(q, dX, dW, wbytes, 0, 0, jobs, dY, dY, K, N, JOBS);
        q.wait();
        const int reps = 20;
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r)
            gemv_q4_K_q8_grouped_tiles(q, dX, dW, wbytes, 0, 0, jobs, dY, dY, K, N, JOBS);
        q.wait();
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count() / reps;
        std::printf("\ngrouped-tiles %u jobs x T=16 (both mats): %.2f ms  "
                    "eff W-GB/s %.1f (8-slice L2 resident: upper bound)\n",
                    JOBS, ms, double(wbytes) * JOBS * 2 / ms / 1e6);
    }
    // s-form grouped tiles (block_q8_1s activations, half the inner ALU).
    {
        const uint64_t q8srow = uint64_t(K / 32) * sizeof(block_q8_1s);
        std::vector<uint8_t> hxs(q8srow * 16);
        for (auto& b : hxs) b = uint8_t(rng() & 0x3F);
        auto* dXs = static_cast<uint8_t*>(alloc.malloc(hxs.size()));
        q.memcpy(dXs, hxs.data(), hxs.size()).wait();
        auto* jobs = static_cast<int32_t*>(alloc.malloc(uint64_t(JOBS) * 4 * 4));
        std::vector<int32_t> hj(JOBS * 4);
        for (uint32_t j = 0; j < JOBS; ++j) {
            hj[4 * j + 0] = 0; hj[4 * j + 1] = 16;
            hj[4 * j + 2] = int32_t(j & 7); hj[4 * j + 3] = 0;
        }
        q.memcpy(jobs, hj.data(), hj.size() * 4).wait();
        gemv_q4_K_q8s_grouped_tiles(q, dXs, dW, wbytes, 0, 0, jobs, dY, dY, K, N, JOBS);
        q.wait();
        const int reps = 20;
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r)
            gemv_q4_K_q8s_grouped_tiles(q, dXs, dW, wbytes, 0, 0, jobs, dY, dY, K, N, JOBS);
        q.wait();
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count() / reps;
        std::printf("s-form grouped-tiles same shape:            %.2f ms  eff W-GB/s %.1f\n",
                    ms, double(wbytes) * JOBS * 2 / ms / 1e6);
    }
    // int8-DPAS grouped tiles (dpas-form fp order): correctness vs s-form
    // (rel err — fp order differs by design), then same-shape timing.
    {
        const uint64_t q8srow = uint64_t(K / 32) * sizeof(block_q8_1s);
        // REAL q8s activations (quantize a random f16 row set) so the
        // correctness diff is meaningful, not garbage-vs-garbage.
        std::vector<uint8_t> hxs(q8srow * 16, 0);
        {
            std::mt19937 r2(7);
            std::normal_distribution<float> nd(0.f, 1.f);
            for (uint32_t t = 0; t < 16; ++t)
                for (uint32_t b = 0; b < K / 32; ++b) {
                    auto* blk = reinterpret_cast<block_q8_1s*>(
                        hxs.data() + t * q8srow + b * sizeof(block_q8_1s));
                    float v[32], amax = 0.f;
                    for (int i = 0; i < 32; ++i) { v[i] = nd(r2); amax = std::max(amax, std::abs(v[i])); }
                    const float d = amax / 127.f;
                    const float id = d ? 1.f / d : 0.f;
                    float s0 = 0.f, s1 = 0.f;
                    for (int i = 0; i < 32; ++i) {
                        blk->qs[i] = int8_t(std::lround(v[i] * id));
                        (i < 16 ? s0 : s1) += d * float(blk->qs[i]);
                    }
                    blk->d = d; blk->s0 = s0; blk->s1 = s1;
                }
        }
        auto* dXs = static_cast<uint8_t*>(alloc.malloc(hxs.size()));
        q.memcpy(dXs, hxs.data(), hxs.size()).wait();
        auto* jobs = static_cast<int32_t*>(alloc.malloc(uint64_t(JOBS) * 4 * 4));
        std::vector<int32_t> hj(JOBS * 4);
        for (uint32_t j = 0; j < JOBS; ++j) {
            hj[4 * j + 0] = 0; hj[4 * j + 1] = 16;
            hj[4 * j + 2] = int32_t(j & 7); hj[4 * j + 3] = 0;
        }
        q.memcpy(jobs, hj.data(), hj.size() * 4).wait();
        auto* dYu = static_cast<sycl::half*>(alloc.malloc(uint64_t(16) * N * 2));
        auto* dYg2 = static_cast<sycl::half*>(alloc.malloc(uint64_t(16) * N * 2));
        auto* dYu2 = static_cast<sycl::half*>(alloc.malloc(uint64_t(16) * N * 2));
        // Reference: s-form on ONE job (slot 0), rows 16.
        std::vector<int32_t> h1(hj.begin(), hj.begin() + 4);
        auto* job1 = static_cast<int32_t*>(alloc.malloc(4 * 4));
        q.memcpy(job1, h1.data(), 16).wait();
        gemv_q4_K_q8s_grouped_tiles(q, dXs, dW, wbytes, 0, 0, job1, dY, dYu, K, N, 1);
        gemv_q4_K_q8d_grouped_tiles(q, dXs, dW, wbytes, 0, 0, job1, dYg2, dYu2, K, N, 1);
        q.wait();
        std::vector<sycl::half> a(16 * N), b2(16 * N), c(16 * N), d2(16 * N);
        q.memcpy(a.data(), dY, a.size() * 2).wait();
        q.memcpy(b2.data(), dYg2, b2.size() * 2).wait();
        q.memcpy(c.data(), dYu, c.size() * 2).wait();
        q.memcpy(d2.data(), dYu2, d2.size() * 2).wait();
        double num = 0, den = 0, num2 = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            const double x1 = float(a[i]), x2 = float(b2[i]);
            num += (x1 - x2) * (x1 - x2); den += x1 * x1;
            const double y1 = float(c[i]), y2 = float(d2[i]);
            num2 += (y1 - y2) * (y1 - y2);
        }
        std::printf("\ndpas vs s-form rel-L2: gate %.3e  up %.3e  (fp-order tol ~1e-3)\n",
                    std::sqrt(num / std::max(den, 1e-30)),
                    std::sqrt(num2 / std::max(den, 1e-30)));
        // Mixed-rows jobs (the in-model shape): rows 1..16, per-row diff.
        {
            std::vector<int32_t> hm;
            for (uint32_t r = 1; r <= 16; ++r) {
                hm.push_back(0); hm.push_back(int32_t(r));
                hm.push_back(int32_t(r & 7)); hm.push_back(0);
            }
            const uint32_t JM = uint32_t(hm.size() / 4);
            auto* jm = static_cast<int32_t*>(alloc.malloc(hm.size() * 4));
            q.memcpy(jm, hm.data(), hm.size() * 4).wait();
            q.memset(dY, 0, uint64_t(16) * N * 2);
            q.memset(dYg2, 0, uint64_t(16) * N * 2);
            q.wait();
            // Run jobs one at a time so overlapping y rows don't race; diff
            // after each.
            double worst = 0; uint32_t worst_rows = 0;
            for (uint32_t jj = 0; jj < JM; ++jj) {
                gemv_q4_K_q8s_grouped_tiles(q, dXs, dW, wbytes, 0, 0, jm + 4 * jj, dY, dYu, K, N, 1);
                gemv_q4_K_q8d_grouped_tiles(q, dXs, dW, wbytes, 0, 0, jm + 4 * jj, dYg2, dYu2, K, N, 1);
                q.wait();
                std::vector<sycl::half> ra(16 * N), rb(16 * N);
                q.memcpy(ra.data(), dY, ra.size() * 2).wait();
                q.memcpy(rb.data(), dYg2, rb.size() * 2).wait();
                const uint32_t rows = uint32_t(hm[4 * jj + 1]);
                const uint32_t yb   = uint32_t(hm[4 * jj + 3]);
                double n3 = 0, d3 = 0;
                for (uint32_t r = 0; r < rows; ++r)
                    for (uint32_t c = 0; c < N; ++c) {
                        const double x1 = float(ra[(yb + r) * N + c]);
                        const double x2 = float(rb[(yb + r) * N + c]);
                        n3 += (x1 - x2) * (x1 - x2); d3 += x1 * x1;
                    }
                const double rl = std::sqrt(n3 / std::max(d3, 1e-30));
                if (rl > worst) { worst = rl; worst_rows = rows; }
            }
            std::printf("mixed-rows dpas worst rel-L2: %.3e (at rows=%u)\n",
                        worst, worst_rows);
        }
        gemv_q4_K_q8d_grouped_tiles(q, dXs, dW, wbytes, 0, 0, jobs, dY, dYu, K, N, JOBS);
        q.wait();
        const int reps = 20;
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r)
            gemv_q4_K_q8d_grouped_tiles(q, dXs, dW, wbytes, 0, 0, jobs, dY, dYu, K, N, JOBS);
        q.wait();
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count() / reps;
        std::printf("int8-DPAS grouped-tiles same shape:         %.2f ms  eff W-GB/s %.1f\n",
                    ms, double(wbytes) * JOBS * 2 / ms / 1e6);
    }
    // Q5_1 down shape [16, 640] x [640, 2560]: pick-major grouped (current)
    // vs fp16-DPAS tiles.
    {
        const uint32_t Kd = 640, Nd = 2560;
        const uint64_t wb5 = uint64_t(Kd / 32) * sizeof(block_q5_1) * Nd;
        std::vector<uint8_t> hw5(wb5 * 8);
        for (auto& b : hw5) b = uint8_t(rng());
        for (size_t off = 0; off + sizeof(block_q5_1) <= hw5.size();
             off += sizeof(block_q5_1)) {
            const sycl::half d(0.002f + 0.01f * float(rng() & 0xFF) / 255.f);
            const sycl::half m(-0.02f + 0.04f * float(rng() & 0xFF) / 255.f);
            std::memcpy(hw5.data() + off, &d, 2);
            std::memcpy(hw5.data() + off + 2, &m, 2);
        }
        auto* dW5 = static_cast<uint8_t*>(alloc.malloc(hw5.size()));
        q.memcpy(dW5, hw5.data(), hw5.size()).wait();
        std::vector<sycl::half> hh(32 * Kd);   // 16 rows + 16 slack
        {
            std::mt19937 r3(9); std::normal_distribution<float> nd(0.f, 1.f);
            for (auto& v : hh) v = sycl::half(nd(r3));
        }
        auto* dH = static_cast<sycl::half*>(alloc.malloc(hh.size() * 2));
        q.memcpy(dH, hh.data(), hh.size() * 2).wait();
        auto* dRs = static_cast<float*>(alloc.malloc(uint64_t(32) * (Kd / 32) * 4));
        q51_block_rowsums(q, dH, 32, Kd, dRs);
        q.wait();
        auto* dY5  = static_cast<sycl::half*>(alloc.malloc(uint64_t(16) * Nd * 2));
        auto* dY5b = static_cast<sycl::half*>(alloc.malloc(uint64_t(16) * Nd * 2));
        auto* jobs = static_cast<int32_t*>(alloc.malloc(uint64_t(JOBS) * 4 * 4));
        std::vector<int32_t> hj(JOBS * 4);
        for (uint32_t j = 0; j < JOBS; ++j) {
            hj[4 * j + 0] = 0; hj[4 * j + 1] = 16;
            hj[4 * j + 2] = int32_t(j & 7); hj[4 * j + 3] = 0;
        }
        q.memcpy(jobs, hj.data(), hj.size() * 4).wait();
        // Correctness: 1 job vs the existing tiles kernel.
        auto* job1 = static_cast<int32_t*>(alloc.malloc(4 * 4));
        q.memcpy(job1, hj.data(), 16).wait();
        gemv_q5_1_grouped_tiles(q, dH, dW5, wb5, 0, job1, dY5, Kd, Nd, 1);
        gemv_q5_1_f16d_grouped_tiles(q, dH, dW5, wb5, 0, job1, dY5b, dRs, Kd, Nd, 1);
        q.wait();
        std::vector<sycl::half> a(16 * Nd), b2(16 * Nd);
        q.memcpy(a.data(), dY5, a.size() * 2).wait();
        q.memcpy(b2.data(), dY5b, b2.size() * 2).wait();
        double num = 0, den = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            const double x1 = float(a[i]), x2 = float(b2[i]);
            num += (x1 - x2) * (x1 - x2); den += x1 * x1;
        }
        std::printf("\nq5_1 f16-dpas vs grouped-tiles rel-L2: %.3e (f16-B rounding tol ~1e-3)\n",
                    std::sqrt(num / std::max(den, 1e-30)));
        auto bench5 = [&](const char* name, auto call) {
            call(); q.wait();
            const int reps = 20;
            const auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < reps; ++r) call();
            q.wait();
            const double ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count() / reps;
            std::printf("%s %.2f ms  eff W-GB/s %.1f\n", name, ms,
                        double(wb5) * JOBS / ms / 1e6);
        };
        bench5("q5_1 grouped-tiles (current):               ", [&] {
            gemv_q5_1_grouped_tiles(q, dH, dW5, wb5, 0, jobs, dY5, Kd, Nd, JOBS); });
        bench5("q5_1 fp16-DPAS tiles:                       ", [&] {
            gemv_q5_1_f16d_grouped_tiles(q, dH, dW5, wb5, 0, jobs, dY5b, dRs, Kd, Nd, JOBS); });
    }
    return 0;
}
