// bench/bench_gemv_q4k.cpp — Phase 3 bandwidth gate.
//
// Measures effective input bandwidth for the W4A16 GEMV across a few
// representative Qwen3.6-35B-A3B projection sizes. The gate is ≥ 365 GB/s
// effective (60% of the B70's 608 GB/s peak).
//
//   ./bench_gemv_q4k [--iters 200]
//
// The kernel is launched on synthetic tensor shapes (random Q4_K bytes; values
// don't matter for bandwidth). Output bandwidth is dominated by W reads:
// `nbytes_W = N * (K/256) * 144`. A reads add up via L2 caching but are tiny
// relative to W. Gate uses W bytes alone.

#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"

#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr const char* kBold = "\033[1m";
constexpr const char* kReset = "\033[0m";
constexpr const char* kGreen = "\033[32m";
constexpr const char* kYell  = "\033[33m";

struct Shape { const char* label; uint32_t K; uint32_t N; };

void run_one(sycl::queue& q, const Shape& sh, int warmup, int iters,
             double gate_gbps) {
    const uint32_t K = sh.K;
    const uint32_t N = sh.N;
    const size_t blocks_per_col = K / 256;
    const size_t W_bytes = size_t(N) * blocks_per_col * sizeof(ie::block_q4_K);
    const size_t A_bytes = K * sizeof(sycl::half);
    const size_t Y_bytes = N * sizeof(sycl::half);

    auto* dA = sycl::malloc_device<sycl::half>(K, q);
    auto* dW = sycl::malloc_device(W_bytes, q);
    auto* dY = sycl::malloc_device<sycl::half>(N, q);

    {
        // Random fill, force d & dmin to small fp16 values to keep dot products finite.
        std::mt19937 rng(0xBEEF);
        std::vector<uint8_t> buf(W_bytes);
        std::uniform_int_distribution<int> bytes(0, 255);
        for (auto& b : buf) b = uint8_t(bytes(rng));
        const uint16_t small = 0x2c00;            // ~0.0625
        for (size_t i = 0; i < N * blocks_per_col; ++i) {
            uint16_t pat[2] = {small, small};
            std::memcpy(buf.data() + i * sizeof(ie::block_q4_K), pat, sizeof(pat));
        }
        q.memcpy(dW, buf.data(), W_bytes).wait();
        std::vector<sycl::half> ha(K, sycl::half(0.01f));
        q.memcpy(dA, ha.data(), A_bytes).wait();
    }

    static const std::vector<sycl::event> nodeps;
    for (int i = 0; i < warmup; ++i) ie::gemv_q4_K(q, dA, dW, dY, K, N, nodeps).wait();

    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();
    for (int i = 0; i < iters; ++i) ie::gemv_q4_K(q, dA, dW, dY, K, N, nodeps).wait();
    auto t1 = clk::now();
    const double sec = std::chrono::duration<double>(t1 - t0).count() / iters;

    const double gbps = W_bytes / sec / 1e9;
    const char* color = (gbps >= gate_gbps) ? kGreen : kYell;

    std::printf("  %s%-20s%s K=%-5u N=%-5u  W=%6.1f MiB  | %.3f ms/iter  %sBW=%6.1f GB/s%s",
                kBold, sh.label, kReset, K, N,
                W_bytes / double(1u << 20),
                sec * 1e3, color, gbps, kReset);
    if (gate_gbps > 0) std::printf("  (gate ≥ %.0f)", gate_gbps);
    std::putchar('\n');

    sycl::free(dA, q); sycl::free(dW, q); sycl::free(dY, q);
}

}  // namespace

int main(int argc, char** argv) {
    int iters = 200, warmup = 20;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
    }
    sycl::device dev;
    bool found = false;
    for (const auto& d : sycl::device::get_devices()) {
        if (d.is_gpu() && d.get_info<sycl::info::device::name>().find("0xe223") != std::string::npos) {
            dev = d; found = true; break;
        }
    }
    if (!found) for (const auto& d : sycl::device::get_devices()) if (d.is_gpu()) { dev = d; found = true; break; }
    if (!found) { std::fputs("no GPU\n", stderr); return 2; }
    sycl::queue q(dev, sycl::property::queue::enable_profiling{});

    std::printf("Device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
    std::printf("Iters: %d (+%d warmup)\n\n", iters, warmup);

    // Global warmup: absorb the one-off SYCL/L0 init that hits the first shape.
    {
        const uint32_t K = 2048, N = 1024;
        auto* dA = sycl::malloc_device<sycl::half>(K, q);
        auto* dW = sycl::malloc_device(size_t(N) * (K/256) * sizeof(ie::block_q4_K), q);
        auto* dY = sycl::malloc_device<sycl::half>(N, q);
        q.memset(dA, 0, K * sizeof(sycl::half)).wait();
        q.memset(dW, 0, size_t(N) * (K/256) * sizeof(ie::block_q4_K)).wait();
        for (int i = 0; i < 50; ++i) ie::gemv_q4_K(q, dA, dW, dY, K, N).wait();
        sycl::free(dA, q); sycl::free(dW, q); sycl::free(dY, q);
    }

    // Sizes drawn from real Qwen3.6-35B-A3B projections (research/04 + GGUF inspect):
    //   attn_q (full-attn):  K=2048, N=8192   (Q+gate fold-in)
    //   attn_output:         K=4096, N=2048
    //   attn_qkv (DeltaNet): K=2048, N=8192
    //   ssm_out:             K=4096, N=2048
    //   gate_proj per expert (decode rare path): K=2048, N=512
    //   down_proj per expert:                    K=512,  N=2048
    Shape shapes[] = {
        {"attn_q   2k×8k",   2048,  8192},
        {"attn_out 4k×2k",   4096,  2048},
        {"ssm_out  4k×2k",   4096,  2048},
        {"qkv      2k×8k",   2048,  8192},
        {"big      4k×4k",   4096,  4096},
        {"4k×8k             ", 4096,  8192},
    };
    std::printf("%sGEMV-Q4_K bandwidth — gate is the FIRST shape (attn_q)%s\n", kBold, kReset);
    for (size_t i = 0; i < sizeof(shapes)/sizeof(shapes[0]); ++i) {
        const double gate = (i == 0) ? 365.0 : 0.0;
        run_one(q, shapes[i], warmup, iters, gate);
    }
    return 0;
}
