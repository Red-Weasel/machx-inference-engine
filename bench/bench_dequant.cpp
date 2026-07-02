// bench/bench_dequant.cpp — Phase 2 throughput gate.
//
//   bench_dequant [--mb 256] [--iters 20]
//
// For each format, allocates a contiguous packed buffer of ~`mb` MiB worth of
// blocks, runs the dequant kernel `iters` times, and reports:
//   - input read bandwidth (bytes packed / time)
//   - output write bandwidth (bytes fp16 / time)
//
// Phase 2 gate: Q4_K output write bandwidth ≥ 350 GB/s.

#include "ie/dequant.hpp"
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

template <typename Block, typename DevFn>
void run_one(const char* label, size_t mb, int warmup, int iters,
             sycl::queue& q, DevFn dev_fn, size_t elems_per_block,
             std::mt19937& rng, double gate_gbps_out)
{
    const size_t bytes_target = mb * (1u << 20);
    const size_t n_blocks = bytes_target / sizeof(Block);
    const size_t packed_bytes = n_blocks * sizeof(Block);
    const size_t n_elem = n_blocks * elems_per_block;
    const size_t out_bytes = n_elem * sizeof(sycl::half);

    void* d_in = sycl::malloc_device(packed_bytes, q);
    sycl::half* d_out = sycl::malloc_device<sycl::half>(n_elem, q);

    // Random fill on host then memcpy. The actual byte values barely matter for
    // a bandwidth test as long as fp16 conversion doesn't yield NaN/Inf.
    {
        std::vector<uint8_t> buf(packed_bytes);
        std::uniform_int_distribution<int> bytes(0, 255);
        for (auto& b : buf) b = uint8_t(bytes(rng));
        // Force d/dmin to small fp16 values so output is finite. We patch only
        // the first 4 bytes of every block (fits both Q8_0's `d` and K-quant's
        // `d, dmin` pair). Q6_K has `d` at offset 208, so handle that too.
        const uint16_t small_pos = 0x3000;  // ~0.125f
        const uint16_t small_neg = 0xb000;  // ~-0.125f
        if constexpr (std::is_same_v<Block, ie::block_q6_K>) {
            for (size_t i = 0; i < n_blocks; ++i) {
                std::memcpy(buf.data() + i * sizeof(Block) + 208,
                            (i & 1) ? &small_neg : &small_pos, 2);
            }
        } else {
            for (size_t i = 0; i < n_blocks; ++i) {
                uint16_t pat[2] = {small_pos, small_neg};
                std::memcpy(buf.data() + i * sizeof(Block), pat, sizeof(pat));
            }
        }
        q.memcpy(d_in, buf.data(), packed_bytes).wait();
    }

    static const std::vector<sycl::event> kNoDeps;

    // Warmup
    for (int i = 0; i < warmup; ++i) dev_fn(q, d_in, d_out, n_elem, kNoDeps).wait();

    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();
    for (int i = 0; i < iters; ++i) {
        auto e = dev_fn(q, d_in, d_out, n_elem, kNoDeps);
        e.wait();
    }
    auto t1 = clk::now();
    const double sec = std::chrono::duration<double>(t1 - t0).count() / iters;

    const double in_gbps  = packed_bytes / sec / 1e9;
    const double out_gbps = out_bytes    / sec / 1e9;
    const char* gate_color = (out_gbps >= gate_gbps_out) ? kGreen : kYell;

    std::printf("  %s%-5s%s  %.1f MiB packed -> %.1f MiB fp16  | %.3f ms/iter  in=%.1f GB/s  %sout=%.1f GB/s%s",
                kBold, label, kReset,
                packed_bytes / double(1u << 20),
                out_bytes / double(1u << 20),
                sec * 1e3, in_gbps,
                gate_color, out_gbps, kReset);
    if (gate_gbps_out > 0) std::printf("  (gate ≥ %.0f GB/s)", gate_gbps_out);
    std::putchar('\n');

    sycl::free(d_in, q);
    sycl::free(d_out, q);
}

}  // namespace

int main(int argc, char** argv) {
    size_t mb = 256;
    int iters = 20, warmup = 5;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--mb" && i + 1 < argc) mb = std::atoi(argv[++i]);
        else if (a == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
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
    std::printf("Workload: ~%zu MiB packed per format, %d warmup + %d iters\n\n",
                mb, warmup, iters);

    std::mt19937 rng(0xBADC0DE);

    run_one<ie::block_q8_0>("Q8_0", mb, warmup, iters, q,
                            ie::dequant_q8_0, 32, rng, /*gate=*/0.0);
    run_one<ie::block_q4_K>("Q4_K", mb, warmup, iters, q,
                            ie::dequant_q4_K, 256, rng, /*gate=*/350.0);
    run_one<ie::block_q5_K>("Q5_K", mb, warmup, iters, q,
                            ie::dequant_q5_K, 256, rng, /*gate=*/0.0);
    run_one<ie::block_q6_K>("Q6_K", mb, warmup, iters, q,
                            ie::dequant_q6_K, 256, rng, /*gate=*/0.0);
    return 0;
}
