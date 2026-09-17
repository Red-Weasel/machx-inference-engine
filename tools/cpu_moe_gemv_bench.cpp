// tools/cpu_moe_gemv_bench.cpp — DRAM miss-stream bench of SHIPPED cpu_moe_gemv.
//
// g++ -std=c++20 -O3 -march=native -pthread -I include \
//     src/ops/cpu_moe_gemv.cpp tools/cpu_moe_gemv_bench.cpp \
//     -o cpu_moe_gemv_bench
#include "ie/cpu_moe_gemv.hpp"
#include "ie/quant_blocks.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using ie::block_q4_K;
using ie::block_q5_K;
using ie::fp32_to_fp16;

static void fill_q4(block_q4_K* w, int n, std::mt19937& rng) {
    std::uniform_int_distribution<int> u8(0, 255), u6(1, 63);
    for (int i = 0; i < n; ++i) {
        w[i].d = fp32_to_fp16(0.02f);
        w[i].dmin = fp32_to_fp16(0.002f);
        for (int s = 0; s < 12; ++s) w[i].scales[s] = uint8_t(u6(rng) | ((u6(rng) & 3) << 6));
        for (int q = 0; q < 128; ++q) w[i].qs[q] = uint8_t(u8(rng));
    }
}
static void fill_q5(block_q5_K* w, int n, std::mt19937& rng) {
    std::uniform_int_distribution<int> u8(0, 255), u6(1, 63);
    for (int i = 0; i < n; ++i) {
        w[i].d = fp32_to_fp16(0.02f);
        w[i].dmin = fp32_to_fp16(0.002f);
        for (int s = 0; s < 12; ++s) w[i].scales[s] = uint8_t(u6(rng) | ((u6(rng) & 3) << 6));
        for (int q = 0; q < 32; ++q) w[i].qh[q] = uint8_t(u8(rng));
        for (int q = 0; q < 128; ++q) w[i].qs[q] = uint8_t(u8(rng));
    }
}

int main() {
    constexpr int H = 4096, EF = 2048, NEXP = 48;
    std::printf("cpu_moe_gemv_bench SHIPPED  H=%d EF=%d NEXP=%d\n", H, EF, NEXP);
    std::mt19937 rng(1);
    std::vector<block_q4_K> gate(size_t(NEXP) * EF * (H / 256));
    std::vector<block_q4_K> up(size_t(NEXP) * EF * (H / 256));
    std::vector<block_q5_K> down(size_t(NEXP) * H * (EF / 256));
    fill_q4(gate.data(), int(gate.size()), rng);
    fill_q4(up.data(), int(up.size()), rng);
    fill_q5(down.data(), int(down.size()), rng);
    std::vector<float> x(H), y(H);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (float& v : x) v = nd(rng);

    const uint64_t bytes = uint64_t(NEXP) *
        (uint64_t(EF) * (H / 256) * sizeof(block_q4_K) * 2 +
         uint64_t(H) * (EF / 256) * sizeof(block_q5_K));
    std::printf("working set %.1f MiB\n", bytes / 1048576.0);

    auto run = [&] {
        for (int e = 0; e < NEXP; ++e) {
            ie::cpu_moe_expert_q8(x.data(),
                &gate[size_t(e) * EF * (H / 256)],
                &up[size_t(e) * EF * (H / 256)],
                &down[size_t(e) * H * (EF / 256)],
                H, EF, 10.f, y.data());
        }
    };
    run();
    const auto t0 = std::chrono::steady_clock::now();
    constexpr int REPS = 4;
    for (int i = 0; i < REPS; ++i) run();
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const double ms = 1e3 * s / REPS;
    const double gbps = (double(bytes) * REPS / s) / 1e9;
    std::printf("48-exp q8 DRAM  %.3f ms  %.1f GB/s  %.3f ms/expert\n",
                ms, gbps, ms / NEXP);
    std::printf("vs PCIe 0.69 GB @ 26.5 GB/s ≈ 26 ms/tok  → CPU %.1f ms/tok\n", ms);
    const bool go = gbps >= 50.0 && (ms / NEXP) <= 0.30;
    std::printf("P1a gate: %s\n", go ? "GO" : "NO-GO");
    return go ? 0 : 1;
}
