// tests/unit/ds41_xmx_ntail_test.cpp -- the fused MXFP4 XMX expert GEMM with a part-full last column tile (2026-10-20:
// V4.1 tensor parallel over 4 cards gives an expert part 576 = 4.5 x 128 columns).
//   1. N = 576 and N = 64 (only a part tile) against an independent host oracle
//   2. the full tiles of an N = 576 job are bit-identical to an N = 512 job over the same weight rows
//   3. nothing is written past the job's N columns (canary), and N % 16 != 0 is still refused
#include "ie/deepseek4_experts.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

static int g_fail = 0;
static void ok(const char* what, bool cond, const std::string& detail = {}) {
    std::printf("  [%s] %s%s%s\n", cond ? "PASS" : "FAIL", what, detail.empty() ? "" : " -- ", detail.c_str());
    if (!cond) ++g_fail;
}

int main() {
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order{});
    constexpr unsigned M = 37, K = 256, NMAX = 576, PAD = 64;
    constexpr float canary = -12345.f;
    uint8_t* qs = sycl::malloc_shared<uint8_t>(NMAX * K / 2, q);
    uint8_t* ep = sycl::malloc_shared<uint8_t>(NMAX * K / 32, q);
    sycl::half* x = sycl::malloc_shared<sycl::half>(M * K, q);
    float* y = sycl::malloc_shared<float>(M * NMAX + PAD, q);
    float* y512 = sycl::malloc_shared<float>(M * 512, q);
    for (unsigned i = 0; i < NMAX * K / 2; ++i) qs[i] = uint8_t((i * 73 + 19) % 256);
    for (unsigned i = 0; i < NMAX * K / 32; ++i) ep[i] = uint8_t(124 + i % 5);
    for (unsigned i = 0; i < M * K; ++i) x[i] = sycl::half(float(int(i % 17) - 8) / 16);
    const float mag[8] = {0, 1, 2, 3, 4, 6, 8, 12};
    auto oracle = [&](unsigned m, unsigned n) {
        double e = 0;
        for (unsigned k = 0; k < K; ++k) {
            const uint8_t byte = qs[n * K / 2 + (k / 32) * 16 + k % 16];
            const unsigned nib = k % 32 < 16 ? byte & 15 : byte >> 4;
            const float w = std::ldexp(mag[nib & 7] * (nib & 8 ? -1.f : 1.f), int(ep[n * K / 32 + k / 32]) - 128);
            e += double(float(x[m * K + k])) * w;
        }
        return e;
    };
    for (unsigned N : {576u, 64u}) {
        std::fill(y, y + M * NMAX + PAD, canary);
        ie::Ds4MxXmxJob J{qs, ep, x, y, M, K, N};
        ie::ds4_gemm_mxfp4_xmx(q, &J, 1).wait_and_throw();
        double worst = 0;
        for (unsigned m = 0; m < M; ++m) for (unsigned n = 0; n < N; ++n) worst = std::max(worst, std::fabs(double(y[m * N + n]) - oracle(m, n)));
        char buf[96]; std::snprintf(buf, sizeof buf, "N = %u: max |y - oracle| %.3g", N, worst);
        ok("matches the host oracle", worst < 1e-5, buf);
        bool clean = true; for (unsigned i = M * N; i < M * N + PAD; ++i) clean &= y[i] == canary;
        ok("nothing written past M x N", clean);
    }
    {   // 2. the full tiles do not change: N = 576's columns 0..511 == an N = 512 job's (the job's rows are its own)
        std::vector<float> a(M * 576);
        ie::Ds4MxXmxJob J{qs, ep, x, y, M, K, 576};
        ie::ds4_gemm_mxfp4_xmx(q, &J, 1).wait_and_throw();
        std::memcpy(a.data(), y, a.size() * 4);
        ie::Ds4MxXmxJob J2{qs, ep, x, y512, M, K, 512};
        ie::ds4_gemm_mxfp4_xmx(q, &J2, 1).wait_and_throw();
        bool same = true;
        for (unsigned m = 0; m < M; ++m) same &= std::memcmp(a.data() + m * 576, y512 + m * 512, 512 * 4) == 0;
        ok("full tiles bit-identical to an N = 512 job", same);
    }
    {   // 3. N % 16 != 0 refused
        bool threw = false;
        try { ie::Ds4MxXmxJob J{qs, ep, x, y, M, K, 72}; ie::ds4_gemm_mxfp4_xmx(q, &J, 1).wait_and_throw(); } catch (const std::invalid_argument&) { threw = true; }
        ok("N = 72 refused", threw);
    }
    for (void* p : {(void*)qs, (void*)ep, (void*)x, (void*)y, (void*)y512}) sycl::free(p, q);
    if (g_fail) { std::printf("ds41_xmx_ntail_test: %d FAILED\n", g_fail); return 1; }
    std::printf("ds41_xmx_ntail_test: PASS\n");
    return 0;
}
