// bench/bench_attention.cpp — Phase 4 perf snapshot.
//
// Captures the prefill GFLOPS metric and the per-decode-step latency for the
// naive `full_attention` at Qwen3.6 dimensions (n_q=16, n_kv=2, d_head=256).
// Not a perf gate — Phase 9's FA-2 work will revisit. This measurement is
// purely informational for the postmortem.

#include "ie/ops.hpp"

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

struct Case { const char* label; uint32_t T; uint32_t start_pos; };

void run_one(sycl::queue& q, const Case& c, int warmup, int iters) {
    constexpr uint32_t n_q = 16, n_kv = 2, d = 256;
    const uint32_t total_ctx = c.start_pos + c.T;
    const uint32_t max_ctx = std::max<uint32_t>(total_ctx, 32);

    auto* dq    = sycl::malloc_device<sycl::half>(c.T * n_q  * d, q);
    auto* dk_in = sycl::malloc_device<sycl::half>(c.T * n_kv * d, q);
    auto* dv_in = sycl::malloc_device<sycl::half>(c.T * n_kv * d, q);
    auto* dk    = sycl::malloc_device<sycl::half>(uint64_t(n_kv) * max_ctx * d, q);
    auto* dv    = sycl::malloc_device<sycl::half>(uint64_t(n_kv) * max_ctx * d, q);
    auto* dy    = sycl::malloc_device<sycl::half>(c.T * n_q  * d, q);
    q.memset(dq,    0, c.T * n_q  * d * sizeof(sycl::half)).wait();
    q.memset(dk_in, 0, c.T * n_kv * d * sizeof(sycl::half)).wait();
    q.memset(dv_in, 0, c.T * n_kv * d * sizeof(sycl::half)).wait();
    q.memset(dk,    0, uint64_t(n_kv) * max_ctx * d * sizeof(sycl::half)).wait();
    q.memset(dv,    0, uint64_t(n_kv) * max_ctx * d * sizeof(sycl::half)).wait();

    static const std::vector<sycl::event> nodeps;
    for (int i = 0; i < warmup; ++i)
        ie::full_attention(q, dq, dk_in, dv_in, dk, dv, dy,
                           c.T, c.start_pos, n_q, n_kv, d, max_ctx, nodeps).wait();

    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();
    for (int i = 0; i < iters; ++i)
        ie::full_attention(q, dq, dk_in, dv_in, dk, dv, dy,
                           c.T, c.start_pos, n_q, n_kv, d, max_ctx, nodeps).wait();
    auto t1 = clk::now();
    const double sec = std::chrono::duration<double>(t1 - t0).count() / iters;

    // Approximate FLOPs: per token attended-to position, 2*d*2 for QK + V
    // accumulation = 4d FLOPs per (q_head, k_position). Plus exp/softmax overhead
    // ~2 FLOPs per (q_head, k_position). Total ≈ 4d * n_q * sum_t (start_pos+t+1).
    uint64_t kv_pairs = 0;
    for (uint32_t t = 0; t < c.T; ++t) kv_pairs += (c.start_pos + t + 1);
    const double flops = 4.0 * d * n_q * double(kv_pairs);
    const double gflops = flops / sec / 1e9;

    std::printf("  %s%-26s%s T=%-4u start_pos=%-5u  | %.3f ms/iter  ~%.1f GFLOPS\n",
                kBold, c.label, kReset, c.T, c.start_pos, sec * 1e3, gflops);

    sycl::free(dq, q); sycl::free(dk_in, q); sycl::free(dv_in, q);
    sycl::free(dk, q); sycl::free(dv, q); sycl::free(dy, q);
}

}  // namespace

int main() {
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
    std::printf("Naive full_attention at Qwen3.6 dims (n_q=16, n_kv=2, d_head=256)\n\n");

    Case cases[] = {
        {"prefill T=1, ctx=0",     1,    0},
        {"prefill T=16, ctx=0",   16,    0},
        {"prefill T=256, ctx=0", 256,    0},
        {"prefill T=1024, ctx=0", 1024,  0},
        {"decode @ ctx=4k",        1, 4096},
        {"decode @ ctx=32k",       1, 32768},
    };
    for (const auto& c : cases) run_one(q, c, /*warmup=*/3, /*iters=*/20);
    return 0;
}
