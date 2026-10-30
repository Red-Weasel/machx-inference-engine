// dequant_q6k_c32_bench (2026-10-28, a bench, not a ctest): the prefill expand Q6_K -> fp16 B^T[K, N] at Nanbeige4.3-33B's
// four projection shapes -- the AoS kernel (dequant_q6_K_to_Bt, the dense path before P3-A) vs the c32 tiled kernel
// (dequant_q6k_c32_to_Bt, the P3-A route). GB/s counts the 6.56 bpw read + the 2 B/element write. Random blocks, timing only.
#include "ie/dequant.hpp"
#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"

#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

int main() {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    struct Shape { uint32_t K, N; const char* name; };
    const Shape shapes[] = {{7168, 7168, "attn_q / attn_output"}, {7168, 22016, "ffn_gate / ffn_up"},
                            {22016, 7168, "ffn_down"}, {7168, 1024, "attn_k / attn_v"}};
    std::mt19937 g(1);
    for (const Shape& s : shapes) {
        const uint32_t K = s.K, N = s.N;
        const uint64_t nb = uint64_t(N) * (K / 256);
        std::vector<uint8_t> blocks(nb * sizeof(ie::block_q6_K));
        for (auto& b : blocks) b = uint8_t(g());
        for (uint64_t i = 0; i < nb; ++i) { const uint16_t d = 0x211f; std::memcpy(blocks.data() + i * sizeof(ie::block_q6_K) + 208, &d, 2); }   // d = ~0.01
        void* d_aos = sycl::malloc_device(blocks.size(), q);
        q.memcpy(d_aos, blocks.data(), blocks.size()).wait();
        std::vector<uint8_t> lo(uint64_t(N) * K / 2), hi(uint64_t(N) * K / 4);
        std::vector<int8_t> sc(uint64_t(N) * K / 16);
        std::vector<uint16_t> dd(uint64_t(N) * K / 256);
        ie::repack_q6_K_to_c32(blocks.data(), K, N, lo.data(), hi.data(), sc.data(), dd.data());
        auto* d_lo = sycl::malloc_device<uint8_t>(lo.size(), q);  q.memcpy(d_lo, lo.data(), lo.size()).wait();
        auto* d_hi = sycl::malloc_device<uint8_t>(hi.size(), q);  q.memcpy(d_hi, hi.data(), hi.size()).wait();
        auto* d_sc = sycl::malloc_device<int8_t>(sc.size(), q);   q.memcpy(d_sc, sc.data(), sc.size()).wait();
        auto* d_d  = sycl::malloc_device<uint16_t>(dd.size(), q); q.memcpy(d_d, dd.data(), dd.size() * 2).wait();
        auto* bt = sycl::malloc_device<sycl::half>(uint64_t(K) * N, q);
        const double bytes = double(K) * N * (210.0 / 256.0 + 2.0);
        auto time = [&](const char* tag, auto&& fn) {
            for (int i = 0; i < 2; ++i) fn().wait();
            q.wait();
            const auto t0 = std::chrono::steady_clock::now();
            const int iters = 10;
            for (int i = 0; i < iters; ++i) fn();
            q.wait();
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / iters;
            std::printf("  %-28s %8.3f ms  %7.1f GB/s\n", tag, ms, bytes / (ms * 1e6));
            return ms;
        };
        std::printf("%s  K=%u N=%u (%.0f MB fp16 out)\n", s.name, K, N, double(K) * N * 2 / 1e6);
        const double a = time("AoS dequant_q6_K_to_Bt", [&] { return ie::dequant_q6_K_to_Bt(q, d_aos, bt, K, N); });
        const double c = time("c32 dequant_q6k_c32_to_Bt", [&] { return ie::dequant_q6k_c32_to_Bt(q, d_lo, d_hi, d_sc, d_d, bt, K, N); });
        std::printf("  c32 / AoS = %.2fx\n", c / a);
        sycl::free(d_aos, q); sycl::free(d_lo, q); sycl::free(d_hi, q); sycl::free(d_sc, q); sycl::free(d_d, q); sycl::free(bt, q);
    }
    return 0;
}
