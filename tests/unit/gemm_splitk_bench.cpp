// tests/unit/gemm_splitk_bench.cpp — gemm_fp16_splitk vs gemm_fp16 at the row counts a GLM-5.3 prefill chunk gives
// one expert (gate/up K 4096 N 2048, down K 2048 N 4096): time on warm clocks, and the error of each against an fp64
// host reference on sampled outputs (so "not bit-identical" is shown to be "as accurate"). GPU1 (09:00.0) only;
// not a CTest: a bench.
#include "ie/ops.hpp"

#include <sycl/sycl.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {
template <class F>
double time_ms(sycl::queue& q, F fn) {
    for (int i = 0; i < 10; ++i) fn();
    q.wait();
    const auto t0 = std::chrono::steady_clock::now();
    constexpr int I = 60;
    for (int i = 0; i < I; ++i) fn();
    q.wait();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / I;
}
}  // namespace

int main() {
    sycl::device dev;
    bool found = false;
    for (auto& d : sycl::device::get_devices(sycl::info::device_type::gpu)) {
        auto bdf = d.get_info<sycl::ext::intel::info::device::pci_address>();
        if (bdf.find("09:00.0") != std::string::npos) { dev = d; found = true; break; }
    }
    if (!found) { std::fputs("GPU1 09:00.0 not found\n", stderr); return 2; }
    sycl::queue q(dev, sycl::property::queue::in_order{});
    std::printf("Device: %s %s, SLM %llu B\n", dev.get_info<sycl::info::device::name>().c_str(),
                dev.get_info<sycl::ext::intel::info::device::pci_address>().c_str(),
                (unsigned long long)dev.get_info<sycl::info::device::local_mem_size>());

    std::mt19937 rng(20261020);
    std::uniform_real_distribution<float> xf(-1.f, 1.f), wf(-0.05f, 0.05f);
    constexpr uint32_t MMAX = 256;
    int bad = 0;
    struct Shape { uint32_t K, N; const char* what; };
    for (Shape s : {Shape{4096, 2048, "gate/up"}, Shape{2048, 4096, "down"}}) {
        std::vector<sycl::half> A(uint64_t(MMAX) * s.K), B(uint64_t(s.K) * s.N);
        for (auto& v : A) v = sycl::half(xf(rng));
        for (auto& v : B) v = sycl::half(wf(rng));
        auto* dA = sycl::malloc_device<sycl::half>(A.size(), q);
        auto* dB = sycl::malloc_device<sycl::half>(B.size(), q);
        auto* c1 = sycl::malloc_device<float>(uint64_t(MMAX + 8) * s.N, q);
        auto* c2 = sycl::malloc_device<float>(uint64_t(MMAX + 8) * s.N, q);
        q.memcpy(dA, A.data(), A.size() * 2);
        q.memcpy(dB, B.data(), B.size() * 2).wait();
        {
            const auto t0 = std::chrono::steady_clock::now();
            while (std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(500)) {
                ie::gemm_fp16(q, dA, dB, c1, 64, s.N, s.K, {});
                q.wait();
            }
        }
        for (uint32_t M : {16u, 24u, 32u, 48u, 64u, 96u, 128u, 192u, 256u}) {
            if (!ie::gemm_fp16_splitk_ok(M, s.N, s.K)) { std::printf("M=%u: not eligible\n", M); continue; }
            const double t1 = time_ms(q, [&] { ie::gemm_fp16(q, dA, dB, c1, M, s.N, s.K, {}); });
            const double t2 = time_ms(q, [&] { ie::gemm_fp16_splitk(q, dA, dB, c2, M, s.N, s.K, {}); });
            std::vector<float> h1(uint64_t(M) * s.N), h2(uint64_t(M) * s.N);
            q.memcpy(h1.data(), c1, h1.size() * 4);
            q.memcpy(h2.data(), c2, h2.size() * 4).wait();
            double e1 = 0, e2 = 0, mref = 0;
            for (uint32_t t = 0; t < 512; ++t) {   // sampled outputs vs fp64
                const uint32_t m = (t * 2654435761u) % M, n = (t * 40503u + 7) % s.N;
                double r = 0;
                for (uint32_t k = 0; k < s.K; ++k)
                    r += double(float(A[uint64_t(m) * s.K + k])) * double(float(B[uint64_t(k) * s.N + n]));
                mref = std::max(mref, std::fabs(r));
                e1 = std::max(e1, std::fabs(h1[uint64_t(m) * s.N + n] - r));
                e2 = std::max(e2, std::fabs(h2[uint64_t(m) * s.N + n] - r));
            }
            if (e2 > 4 * e1 + 1e-6 * mref) ++bad;
            const double gf = 2.0 * M * s.N * s.K / 1e9;
            std::printf("%s M=%3u: gemm_fp16 %.4f ms (%5.1f TF/s)  splitk %.4f ms (%5.1f TF/s)  x%.2f  "
                        "err vs fp64: %.2e / %.2e (max|ref| %.2f)\n",
                        s.what, M, t1, gf / t1, t2, gf / t2, t1 / t2, e1, e2, mref);
        }
        sycl::free(dA, q); sycl::free(dB, q); sycl::free(c1, q); sycl::free(c2, q);
    }
    std::printf("%s\n", bad ? "SPLITK LESS ACCURATE" : "splitk as accurate as gemm_fp16 (fp32 accumulate)");
    return bad ? 1 : 0;
}
