// tests/unit/gemm_q4_xmx_prefill_test.cpp — XMX vs rows at GLM-5.3 prefill Ms.
// GPU1 only. Drives shipped gemm_q4_K_xmx / gemv_q4_K_rows.
#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"

#include <sycl/sycl.hpp>

#include <chrono>
#include <cmath>
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
    if (!found) { std::fputs("GPU1 09:00.0 not found\n", stderr); return 2; }
    sycl::queue q(dev, sycl::property::queue::in_order{});
    std::printf("Device: %s %s\n",
                dev.get_info<sycl::info::device::name>().c_str(),
                dev.get_info<sycl::ext::intel::info::device::pci_address>().c_str());

    constexpr uint32_t K = 4096, N = 2048;
    const uint32_t bpc = K / 256;
    const uint64_t nbytes = uint64_t(N) * bpc * sizeof(ie::block_q4_K);
    std::mt19937 rng(31);
    std::uniform_int_distribution<int> u8(0, 255);
    std::uniform_real_distribution<float> xf(-1.f, 1.f);
    std::vector<uint8_t> Wh(nbytes);
    for (auto& b : Wh) b = uint8_t(u8(rng));
    auto* blk = reinterpret_cast<ie::block_q4_K*>(Wh.data());
    for (uint64_t i = 0; i < nbytes / sizeof(ie::block_q4_K); ++i) {
        blk[i].d = sycl::bit_cast<uint16_t>(sycl::half(0.05f));
        blk[i].dmin = sycl::bit_cast<uint16_t>(sycl::half(0.01f));
    }
    auto* dW = sycl::malloc_device<uint8_t>(nbytes, q);
    q.memcpy(dW, Wh.data(), nbytes).wait();

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

    int bad = 0;
    for (uint32_t M : {4u, 8u, 16u, 32u}) {
        std::vector<sycl::half> A(uint64_t(M) * K);
        for (auto& v : A) v = sycl::half(xf(rng));
        auto* dA = sycl::malloc_device<sycl::half>(A.size(), q);
        auto* yx = sycl::malloc_device<sycl::half>(uint64_t(M) * N, q);
        auto* yr = sycl::malloc_device<sycl::half>(uint64_t(M) * N, q);
        q.memcpy(dA, A.data(), A.size() * 2).wait();

        const double tr = time_ms([&] {
            ie::gemv_q4_K_rows(q, dA, dW, yr, K, N, M, {});
        });
        const double tx = time_ms([&] {
            if (M <= 16)
                ie::gemm_q4_K_xmx(q, dA, dW, yx, M, K, N, {});
            else {
                ie::gemm_q4_K_xmx(q, dA, dW, yx, 16, K, N, {});
                ie::gemm_q4_K_xmx(q, dA + 16ull * K, dW, yx + 16ull * N, M - 16, K, N, {});
            }
        });
        std::vector<sycl::half> hx(uint64_t(M) * N), hr(uint64_t(M) * N);
        q.memcpy(hx.data(), yx, uint64_t(M) * N * 2);
        q.memcpy(hr.data(), yr, uint64_t(M) * N * 2).wait();
        int nan = 0; float maxd = 0.f;
        for (uint64_t i = 0; i < uint64_t(M) * N; ++i) {
            const float a = float(hx[i]), b = float(hr[i]);
            if (!std::isfinite(a) || !std::isfinite(b)) ++nan;
            maxd = std::max(maxd, std::fabs(a - b));
        }
        const double bytes = double(nbytes);
        std::printf("M=%u  rows %.3f ms (%.0f GB/s)  xmx %.3f ms (%.0f GB/s)  %.2fx  max|d|=%.3g nan=%d\n",
                    M, tr, bytes / (tr * 1e6), tx, bytes / (tx * 1e6), tr / tx, maxd, nan);
        if (nan) ++bad;
        sycl::free(dA, q); sycl::free(yx, q); sycl::free(yr, q);
    }
    sycl::free(dW, q);
    if (bad) { std::puts("FAIL"); return 1; }
    std::puts("GATE PASSED");
    return 0;
}
