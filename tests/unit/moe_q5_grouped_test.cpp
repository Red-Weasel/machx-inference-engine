// tests/unit/moe_q5_grouped_test.cpp — grouped Q5_K down vs P solos.
// GPU1 (09:00.0). GLM-5.3 decode down: K=2048 N=4096 P=8.
#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"

#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

int main() {
    sycl::device dev;
    bool found = false;
    for (auto& d : sycl::device::get_devices(sycl::info::device_type::gpu)) {
        auto bdf = d.get_info<sycl::ext::intel::info::device::pci_address>();
        if (bdf.find("09:00.0") != std::string::npos) { dev = d; found = true; break; }
    }
    if (!found) {
        std::fputs("GPU1 09:00.0 not found\n", stderr);
        return 2;
    }
    sycl::queue q(dev, sycl::property::queue::in_order{});
    std::printf("Device: %s %s\n",
                dev.get_info<sycl::info::device::name>().c_str(),
                dev.get_info<sycl::ext::intel::info::device::pci_address>().c_str());

    constexpr uint32_t K = 2048, N = 4096, P = 8;
    const uint32_t bpc = K / 256;
    const uint64_t mat_bytes = uint64_t(N) * bpc * sizeof(ie::block_q5_K);
    const uint64_t slot_bytes = mat_bytes;
    const uint64_t down_off = 0;

    std::mt19937 rng(23);
    std::uniform_int_distribution<int> u8(0, 255);
    std::uniform_real_distribution<float> xf(-1.f, 1.f);
    std::vector<uint8_t> slots_h(P * slot_bytes);
    for (auto& b : slots_h) b = uint8_t(u8(rng));
    for (uint32_t p = 0; p < P; ++p) {
        auto* blk0 = reinterpret_cast<ie::block_q5_K*>(
            slots_h.data() + p * slot_bytes);
        for (uint32_t n = 0; n < N; ++n) {
            auto* blk = blk0 + uint64_t(n) * bpc;
            for (uint32_t b = 0; b < bpc; ++b) {
                blk[b].d = sycl::bit_cast<uint16_t>(sycl::half(0.05f));
                blk[b].dmin = sycl::bit_cast<uint16_t>(sycl::half(0.01f));
            }
        }
    }
    std::vector<sycl::half> x(uint64_t(P) * K);
    for (auto& v : x) v = sycl::half(xf(rng));
    std::vector<int32_t> slot_ids(P);
    for (uint32_t p = 0; p < P; ++p) slot_ids[p] = int32_t(p);

    auto* dslots = sycl::malloc_device<uint8_t>(slots_h.size(), q);
    auto* dx = sycl::malloc_device<sycl::half>(x.size(), q);
    auto* y1 = sycl::malloc_device<sycl::half>(P * N, q);
    auto* y2 = sycl::malloc_device<sycl::half>(P * N, q);
    q.memcpy(dslots, slots_h.data(), slots_h.size());
    q.memcpy(dx, x.data(), x.size() * 2).wait();

    for (uint32_t p = 0; p < P; ++p) {
        ie::gemv_q5_K(q, dx + uint64_t(p) * K,
                      dslots + uint64_t(p) * slot_bytes + down_off,
                      y1 + uint64_t(p) * N, K, N, {});
    }
    ie::moe_gemv_q5_K_down_grouped(q, dx, dslots, slot_bytes, down_off,
                                   slot_ids.data(), P, y2, K, N, {});
    q.wait();

    std::vector<sycl::half> h1(P * N), h2(P * N);
    q.memcpy(h1.data(), y1, P * N * 2);
    q.memcpy(h2.data(), y2, P * N * 2).wait();
    int mism = 0;
    for (uint32_t i = 0; i < P * N; ++i)
        if (sycl::bit_cast<uint16_t>(h1[i]) != sycl::bit_cast<uint16_t>(h2[i]))
            ++mism;
    std::printf("grouped vs %u gemv_q5_K: %d/%u mismatches\n", P, mism, P * N);

    auto time_ms = [&](auto fn) {
        for (int i = 0; i < 5; ++i) fn();
        q.wait();
        const auto t0 = std::chrono::steady_clock::now();
        constexpr int I = 20;
        for (int i = 0; i < I; ++i) fn();
        q.wait();
        return std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count() / I;
    };
    const double tsolo = time_ms([&] {
        for (uint32_t p = 0; p < P; ++p)
            ie::gemv_q5_K(q, dx + uint64_t(p) * K,
                          dslots + uint64_t(p) * slot_bytes + down_off,
                          y1 + uint64_t(p) * N, K, N, {});
    });
    const double tgrp = time_ms([&] {
        ie::moe_gemv_q5_K_down_grouped(q, dx, dslots, slot_bytes, down_off,
                                       slot_ids.data(), P, y2, K, N, {});
    });
    const double bytes = double(P) * double(mat_bytes);
    std::printf("bench P=%u K=%u N=%u  solo %.3f ms (%.1f GB/s)  grouped %.3f ms (%.1f GB/s)  %.2fx\n",
                P, K, N, tsolo, bytes / (tsolo * 1e6), tgrp, bytes / (tgrp * 1e6), tsolo / tgrp);

    sycl::free(dslots, q); sycl::free(dx, q); sycl::free(y1, q); sycl::free(y2, q);
    if (mism != 0) { std::puts("FAIL"); return 1; }
    std::puts("GATE PASSED");
    return 0;
}
