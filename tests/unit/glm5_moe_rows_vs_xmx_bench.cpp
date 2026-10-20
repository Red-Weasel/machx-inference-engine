// tests/unit/glm5_moe_rows_vs_xmx_bench.cpp — the GLM-5.3 prefill expert GEMM at the row counts a prefill chunk
// gives one expert (chunk 1024: ~28 rows on average, chunk 2048: ~57): the shipped multi-row GEMV
// (gemv_q4_K_rows, raw Q4_K) vs the dequant + XMX chain (dequant_q4_K_to_Bt + gemm_fp16, fp32 out) that
// IE_G5_XMX_ROWS routes experts to (src/model/glm5next.cpp). Gate/up shape K=4096 N=2048. GPU1 (09:00.0) only;
// not a CTest: a bench (2026-10-19).
#include "ie/dequant.hpp"
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
    std::printf("Device: %s %s\n", dev.get_info<sycl::info::device::name>().c_str(),
                dev.get_info<sycl::ext::intel::info::device::pci_address>().c_str());

    constexpr uint32_t K = 4096, N = 2048;
    const uint64_t nbytes = uint64_t(N) * (K / 256) * sizeof(ie::block_q4_K);
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
    auto* dB = sycl::malloc_device<sycl::half>(uint64_t(K) * N, q);
    q.memcpy(dW, Wh.data(), nbytes).wait();

    auto time_ms = [&](auto fn) {
        for (int i = 0; i < 5; ++i) fn();
        q.wait();
        const auto t0 = std::chrono::steady_clock::now();
        constexpr int I = 30;
        for (int i = 0; i < I; ++i) fn();
        q.wait();
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / I;
    };
    const double td = time_ms([&] { ie::dequant_q4_K_to_Bt(q, dW, dB, K, N, {}); });
    std::printf("dequant_q4_K_to_Bt alone: %.3f ms per expert matrix\n", td);

    int bad = 0;
    for (uint32_t M : {8u, 16u, 24u, 32u, 48u, 64u, 96u, 128u, 192u, 256u}) {
        std::vector<sycl::half> A(uint64_t(M) * K);
        for (auto& v : A) v = sycl::half(xf(rng));
        auto* dA = sycl::malloc_device<sycl::half>(A.size(), q);
        auto* yr = sycl::malloc_device<sycl::half>(uint64_t(M) * N, q);
        auto* yc = sycl::malloc_device<float>(uint64_t(M) * N, q);
        q.memcpy(dA, A.data(), A.size() * 2).wait();
        const double tr = time_ms([&] { ie::gemv_q4_K_rows(q, dA, dW, yr, K, N, M, {}); });
        const double tc = time_ms([&] {
            ie::dequant_q4_K_to_Bt(q, dW, dB, K, N, {});
            ie::gemm_fp16(q, dA, dB, yc, M, N, K, {});
        });
        std::vector<sycl::half> hr(uint64_t(M) * N);
        std::vector<float> hc(uint64_t(M) * N);
        q.memcpy(hr.data(), yr, hr.size() * 2);
        q.memcpy(hc.data(), yc, hc.size() * 4).wait();
        double maxd = 0, maxv = 0;
        int nan = 0;
        for (uint64_t i = 0; i < hc.size(); ++i) {
            const double a = double(float(hr[i])), b = double(hc[i]);
            if (!std::isfinite(a) || !std::isfinite(b)) { ++nan; continue; }
            maxd = std::max(maxd, std::fabs(a - b));
            maxv = std::max(maxv, std::fabs(b));
        }
        const double rel = maxv > 0 ? maxd / maxv : 0;
        if (nan || rel > 2e-3) ++bad;
        std::printf("M=%3u  rows %.3f ms  dequant+xmx %.3f ms  rows/xmx %.2fx  max|d|=%.3g (rel %.2g) nan=%d\n",
                    M, tr, tc, tr / tc, maxd, rel, nan);
        sycl::free(dA, q); sycl::free(yr, q); sycl::free(yc, q);
    }
    sycl::free(dW, q); sycl::free(dB, q);
    std::printf("%s\n", bad ? "MISMATCH" : "outputs agree (rel <= 2e-3)");
    return bad ? 1 : 0;
}
