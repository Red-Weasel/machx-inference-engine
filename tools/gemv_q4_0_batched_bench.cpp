// tools/gemv_q4_0_batched_bench.cpp — spec-decode VERIFY amortization de-risk.
//
// The question that decides whether the gemma-4 MTP self-spec port is worth it:
// does a batched T=K verify forward amortize the BW-bound weight read on B70, or
// does B70's small-batch weakness make T=K cost ~K× a single decode?
//
// This times gemv_q4_0_soa_q8_batched (each Q4_0-SoA weight column read ONCE,
// dotted against T staged Q8_1 rows) at T=1,2,4,8 on a real 31B projection shape.
//   * total_ms(T)/total_ms(1) ≈ 1  → perfect amortization → spec port viable.
//   * total_ms(T)/total_ms(1) ≈ T  → no amortization (small-batch wall) → dead.
// Also verifies batched T=1 == the single-row gemv_q4_0_soa_q8 (lossless gate).
#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"
#include "ie/allocator.hpp"

#include <sycl/sycl.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
static double median(std::vector<double> v) {
    if (v.empty()) return 0; std::sort(v.begin(), v.end()); return v[v.size() / 2];
}

int main(int argc, char** argv) {
    const uint32_t K = (argc > 1) ? std::atoi(argv[1]) : 5376;    // 31B hidden
    const uint32_t N = (argc > 2) ? std::atoi(argv[2]) : 21504;   // 31B ffn
    const int runs = 9, inner = 20;
    const int Ts[] = {1, 2, 4, 8};

    ie::DeviceAllocator alloc;
    if (auto e = alloc.init(); !e.empty()) { std::fprintf(stderr, "alloc: %s\n", e.c_str()); return 1; }
    auto& q = alloc.queue();
    std::printf("gemv_q4_0_soa_q8_batched AMORTIZATION  K=%u N=%u  device=%s\n",
                K, N, q.get_device().get_info<sycl::info::device::name>().c_str());

    const uint32_t bpc = K / 32;
    // Random Q4_0 AoS weights [K,N] (N columns × bpc blocks), then SoA repack.
    std::mt19937 rng(1234);
    std::uniform_int_distribution<int> nib(0, 255);
    std::vector<ie::block_q4_0> aos(uint64_t(N) * bpc);
    for (auto& b : aos) {
        b.d = sycl::half(0.05f + 0.001f * (nib(rng) & 7));
        for (int i = 0; i < 16; ++i) b.qs[i] = uint8_t(nib(rng));
    }
    std::vector<uint8_t>  h_qs(uint64_t(N) * (K / 2));
    std::vector<uint16_t> h_d (uint64_t(N) * bpc);
    ie::repack_q4_0_to_soa(aos.data(), K, N, h_qs.data(), h_d.data());

    auto* d_qs = static_cast<uint8_t*>(alloc.malloc(h_qs.size()));
    auto* d_d  = static_cast<uint16_t*>(alloc.malloc(h_d.size() * sizeof(uint16_t)));
    q.memcpy(d_qs, h_qs.data(), h_qs.size()).wait();
    q.memcpy(d_d,  h_d.data(),  h_d.size() * sizeof(uint16_t)).wait();

    const int T_MAX = 8;
    std::vector<sycl::half> act(uint64_t(T_MAX) * K);
    std::uniform_real_distribution<float> uf(-1.f, 1.f);
    for (auto& v : act) v = sycl::half(uf(rng));
    auto* d_act = sycl::malloc_device<sycl::half>(act.size(), q);
    q.memcpy(d_act, act.data(), act.size() * sizeof(sycl::half)).wait();
    auto* d_q8 = alloc.malloc(uint64_t(T_MAX) * bpc * sizeof(ie::block_q8_1x));
    ie::quantize_q8_1(q, d_act, d_q8, T_MAX * K).wait();   // T_MAX contiguous rows
    auto* d_y = sycl::malloc_device<sycl::half>(uint64_t(T_MAX) * N, q);

    // Correctness: batched T=1 vs single-row gemv_q4_0_soa_q8.
    ie::gemv_q4_0_soa_q8(q, d_q8, d_qs, d_d, d_y, K, N).wait();
    std::vector<sycl::half> y_ref(N); q.memcpy(y_ref.data(), d_y, N * sizeof(sycl::half)).wait();
    ie::gemv_q4_0_soa_q8_batched(q, d_q8, d_qs, d_d, d_y, K, N, 1).wait();
    std::vector<sycl::half> y_bat(N); q.memcpy(y_bat.data(), d_y, N * sizeof(sycl::half)).wait();
    double max_abs = 0; for (uint32_t i = 0; i < N; ++i) max_abs = std::max(max_abs, double(std::fabs(float(y_ref[i]) - float(y_bat[i]))));
    std::printf("batched T=1 vs single-row: max_abs_diff=%.3g %s\n", max_abs, max_abs < 1e-2 ? "OK" : "MISMATCH");

    const double wbytes = double(N) * K * 0.5625;   // weights read once / call
    double t1_ms = 0;
    std::printf("\n  T   total_ms   ms/token   GB/s    vs_T1\n");
    for (int T : Ts) {
        for (int w = 0; w < 3; ++w) ie::gemv_q4_0_soa_q8_batched(q, d_q8, d_qs, d_d, d_y, K, N, T);
        q.wait();
        std::vector<double> ms;
        for (int r = 0; r < runs; ++r) {
            const double t0 = now_ms();
            for (int it = 0; it < inner; ++it)
                ie::gemv_q4_0_soa_q8_batched(q, d_q8, d_qs, d_d, d_y, K, N, T);
            q.wait();
            ms.push_back((now_ms() - t0) / inner);
        }
        const double m = median(ms);
        if (T == 1) t1_ms = m;
        std::printf("  %d   %8.4f   %8.4f   %6.1f   %.2fx\n",
                    T, m, m / T, wbytes / (m / 1000.0) / 1e9, m / t1_ms);
    }
    std::printf("\nVERDICT: total_ms(T=8)/total_ms(T=1) ~1 = amortizes (port viable); ~8 = small-batch wall.\n");

    sycl::free(d_act, q); sycl::free(d_y, q);
    return 0;
}
