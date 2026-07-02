// tools/ie_exl3_test.cpp — EXL3 GPU prototype: the Phase-B Task-2 DECISION GATE.
//
// Validates the two new SYCL kernels (gemv_exl3 + hadamard_transform) against the
// host oracle vectors (tests/data/exl3/onelayer.*, regenerate via
// tools/exl3/make_oracle.py), and times gemv_exl3 vs gemv_q4_K on the same [K,N].
//
//   Test A  decode GEMV : y = x @ W_rot   (gemv_exl3 vs host x@wrot.f16)
//   Test B  Hadamard    : had128 round-trip + vs host Sylvester WHT
//   Test C  full forward: had128(x⊙suh) @ W_rot, had128(acc)⊙svh  vs  x @ weight.f16
//   Speed   gemv_exl3 vs gemv_q4_K (same K,N)
//
// GATE: Test A cosine ≥ 0.9999  AND  gemv_exl3 within ~1.5× of gemv_q4_K @ T=1.
//
// usage: ie-exl3-test [--dir tests/data/exl3]

#include "ie/allocator.hpp"
#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace {

float h2f(uint16_t b) { sycl::half h; std::memcpy(&h, &b, 2); return float(h); }
uint16_t f2h(float f) { sycl::half h = sycl::half(f); uint16_t b; std::memcpy(&b, &h, 2); return b; }

std::vector<uint8_t> read_bytes(const std::string& p, bool& ok) {
    std::ifstream f(p, std::ios::binary);
    if (!f) { ok = false; return {}; }
    f.seekg(0, std::ios::end); std::streamsize n = f.tellg(); f.seekg(0, std::ios::beg);
    std::vector<uint8_t> b(size_t(n < 0 ? 0 : n));
    if (n > 0) f.read(reinterpret_cast<char*>(b.data()), n);
    ok = bool(f); return b;
}

std::vector<float> read_f16(const std::string& p, bool& ok) {
    auto b = read_bytes(p, ok);
    if (!ok) return {};
    const uint16_t* u = reinterpret_cast<const uint16_t*>(b.data());
    std::vector<float> v(b.size() / 2);
    for (size_t i = 0; i < v.size(); ++i) v[i] = h2f(u[i]);
    return v;
}

long meta_int(const std::string& js, const char* key) {
    auto p = js.find(std::string("\"") + key + "\"");
    if (p == std::string::npos) return -1;
    p = js.find(':', p); if (p == std::string::npos) return -1;
    return std::strtol(js.c_str() + p + 1, nullptr, 10);
}

double cosine(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); ++i) { dot += double(a[i]) * b[i]; na += double(a[i]) * a[i]; nb += double(b[i]) * b[i]; }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
}
double maxabs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0; for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(double(a[i]) - b[i])); return m;
}

void had128_host(const float* in, float* out) {
    float t[128]; std::memcpy(t, in, sizeof t);
    for (int len = 1; len < 128; len <<= 1)
        for (int i = 0; i < 128; i += 2 * len)
            for (int j = 0; j < len; ++j) {
                float a = t[i + j], b = t[i + j + len];
                t[i + j] = a + b; t[i + j + len] = a - b;
            }
    for (int i = 0; i < 128; ++i) out[i] = t[i] * 0.088388347648f;
}

}  // namespace

int main(int argc, char** argv) {
    std::string dir = "tests/data/exl3";
    for (int i = 1; i < argc - 1; ++i) if (std::string(argv[i]) == "--dir") dir = argv[i + 1];

    bool ok = true, ok2 = true, ok3 = true, ok4 = true, ok5 = true;
    auto trellis = read_bytes(dir + "/onelayer.trellis", ok);
    auto wrot   = read_f16(dir + "/onelayer.wrot.f16", ok2);
    auto weight = read_f16(dir + "/onelayer.weight.f16", ok3);
    auto suh    = read_f16(dir + "/onelayer.suh.f16", ok4);
    auto svh    = read_f16(dir + "/onelayer.svh.f16", ok5);
    std::ifstream mf(dir + "/onelayer.meta.json");
    if (!ok || !ok2 || !ok3 || !ok4 || !ok5 || !mf) {
        std::printf("ie-exl3-test: SKIP (vectors absent — regenerate via tools/exl3/make_oracle.py)\n");
        return 0;
    }
    std::string meta((std::istreambuf_iterator<char>(mf)), std::istreambuf_iterator<char>());
    const uint32_t K = uint32_t(meta_int(meta, "K"));
    const uint32_t N = uint32_t(meta_int(meta, "N"));
    const uint32_t bits = uint32_t(meta_int(meta, "bits"));
    std::printf("ie-exl3-test: K=%u N=%u bits=%u  (trellis %zu B, wrot %zu, weight %zu)\n",
                K, N, bits, trellis.size(), wrot.size(), weight.size());

    ie::DeviceFleet fleet;
    if (auto e = fleet.init(1); !e.empty()) { std::fprintf(stderr, "fleet init: %s\n", e.c_str()); return 1; }
    sycl::queue& q = fleet.dev(0).queue();
    std::printf("device: %s\n", fleet.dev(0).device().get_info<sycl::info::device::name>().c_str());

    // ---- device buffers ----
    auto* d_codes = static_cast<uint8_t*>(fleet.dev(0).malloc(trellis.size()));
    auto* d_x   = static_cast<sycl::half*>(fleet.dev(0).malloc(K * sizeof(sycl::half)));
    auto* d_xs  = static_cast<sycl::half*>(fleet.dev(0).malloc(K * sizeof(sycl::half)));
    auto* d_xh  = static_cast<sycl::half*>(fleet.dev(0).malloc(K * sizeof(sycl::half)));
    auto* d_y   = static_cast<sycl::half*>(fleet.dev(0).malloc(N * sizeof(sycl::half)));
    auto* d_acc = static_cast<sycl::half*>(fleet.dev(0).malloc(N * sizeof(sycl::half)));
    auto* d_acch= static_cast<sycl::half*>(fleet.dev(0).malloc(N * sizeof(sycl::half)));
    q.memcpy(d_codes, trellis.data(), trellis.size()).wait();

    // fixed pseudo-random activation
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x(K);
    std::vector<uint16_t> xh16(K), xs16(K);
    for (uint32_t k = 0; k < K; ++k) { x[k] = nd(rng); xh16[k] = f2h(x[k]); xs16[k] = f2h(x[k] * suh[k]); }
    q.memcpy(d_x,  xh16.data(), K * sizeof(uint16_t)).wait();
    q.memcpy(d_xs, xs16.data(), K * sizeof(uint16_t)).wait();

    int rc = 0;

    // ================= Test A — decode GEMV =================
    ie::gemv_exl3(q, d_x, d_codes, d_y, K, N, bits).wait();
    std::vector<uint16_t> y16(N); q.memcpy(y16.data(), d_y, N * sizeof(uint16_t)).wait();
    std::vector<float> yA_gpu(N), yA_host(N, 0.f);
    for (uint32_t n = 0; n < N; ++n) yA_gpu[n] = h2f(y16[n]);
    for (uint32_t k = 0; k < K; ++k) { float xk = x[k]; const float* wr = &wrot[size_t(k) * N]; for (uint32_t n = 0; n < N; ++n) yA_host[n] += xk * wr[n]; }
    double cosA = cosine(yA_gpu, yA_host), errA = maxabs_diff(yA_gpu, yA_host);
    std::printf("[A] decode GEMV   cosine=%.7f  maxerr=%.4e  %s\n", cosA, errA, cosA >= 0.9999 ? "PASS" : "FAIL");
    if (cosA < 0.9999) rc = 1;

    // ================= Test B — Hadamard =================
    {
        const uint32_t n = 512;  // 4 blocks
        auto* d_v = static_cast<sycl::half*>(fleet.dev(0).malloc(n * sizeof(sycl::half)));
        auto* d_hv = static_cast<sycl::half*>(fleet.dev(0).malloc(n * sizeof(sycl::half)));
        auto* d_hhv = static_cast<sycl::half*>(fleet.dev(0).malloc(n * sizeof(sycl::half)));
        std::vector<float> v(n); std::vector<uint16_t> v16(n);
        for (uint32_t i = 0; i < n; ++i) { v[i] = nd(rng); v16[i] = f2h(v[i]); }
        q.memcpy(d_v, v16.data(), n * sizeof(uint16_t)).wait();
        ie::hadamard_transform(q, d_v, d_hv, n).wait();
        ie::hadamard_transform(q, d_hv, d_hhv, n).wait();
        std::vector<uint16_t> hv16(n), hhv16(n);
        q.memcpy(hv16.data(), d_hv, n * sizeof(uint16_t)).wait();
        q.memcpy(hhv16.data(), d_hhv, n * sizeof(uint16_t)).wait();
        std::vector<float> hv_gpu(n), hhv_gpu(n), hv_host(n);
        for (uint32_t i = 0; i < n; ++i) { hv_gpu[i] = h2f(hv16[i]); hhv_gpu[i] = h2f(hhv16[i]); }
        for (uint32_t blk = 0; blk < n / 128; ++blk) had128_host(&v[blk * 128], &hv_host[blk * 128]);
        double cosH = cosine(hv_gpu, hv_host);
        double cosRT = cosine(hhv_gpu, v);
        std::printf("[B] hadamard      cosine(vs host WHT)=%.7f  round-trip cosine=%.7f  %s\n",
                    cosH, cosRT, (cosH >= 0.999 && cosRT >= 0.999) ? "PASS" : "FAIL");
        if (cosH < 0.999 || cosRT < 0.999) rc = 1;
        fleet.dev(0).free(d_v); fleet.dev(0).free(d_hv); fleet.dev(0).free(d_hhv);
    }

    // ================= Test C — full forward vs weight.f16 =================
    {
        ie::hadamard_transform(q, d_xs, d_xh, K).wait();
        ie::gemv_exl3(q, d_xh, d_codes, d_acc, K, N, bits).wait();
        ie::hadamard_transform(q, d_acc, d_acch, N).wait();
        std::vector<uint16_t> acch16(N); q.memcpy(acch16.data(), d_acch, N * sizeof(uint16_t)).wait();
        std::vector<float> yC_gpu(N), yC_host(N, 0.f);
        for (uint32_t n = 0; n < N; ++n) yC_gpu[n] = h2f(acch16[n]) * svh[n];
        for (uint32_t k = 0; k < K; ++k) { float xk = x[k]; const float* w = &weight[size_t(k) * N]; for (uint32_t n = 0; n < N; ++n) yC_host[n] += xk * w[n]; }
        double cosC = cosine(yC_gpu, yC_host), errC = maxabs_diff(yC_gpu, yC_host);
        std::printf("[C] full forward  cosine=%.7f  maxerr=%.4e  %s\n", cosC, errC, cosC >= 0.999 ? "PASS" : "FAIL");
        if (cosC < 0.999) rc = 1;
    }

    // ================= Speed — gemv_exl3 vs gemv_q4_K =================
    {
        const size_t q4k_bytes = size_t(N) * (K / 256) * sizeof(ie::block_q4_K);
        auto* d_q4k = static_cast<uint8_t*>(fleet.dev(0).malloc(q4k_bytes));
        std::vector<uint8_t> junk(q4k_bytes);
        for (size_t i = 0; i < q4k_bytes; ++i) junk[i] = uint8_t(i * 131 + 7);
        q.memcpy(d_q4k, junk.data(), q4k_bytes).wait();

        const int ITERS = 60, WARM = 10;
        auto bench = [&](auto launch) {
            std::vector<double> ms;
            for (int i = 0; i < ITERS; ++i) {
                auto t0 = std::chrono::high_resolution_clock::now();
                launch().wait();
                auto t1 = std::chrono::high_resolution_clock::now();
                if (i >= WARM) ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
            std::sort(ms.begin(), ms.end());
            return ms[ms.size() / 2];  // median
        };
        double t_exl3 = bench([&] { return ie::gemv_exl3(q, d_x, d_codes, d_y, K, N, bits); });
        double t_q4k  = bench([&] { return ie::gemv_q4_K(q, d_x, d_q4k, d_y, K, N); });
        std::printf("[S] gemv_exl3=%.4f ms  gemv_q4_K=%.4f ms  ratio=%.2fx  %s\n",
                    t_exl3, t_q4k, t_exl3 / t_q4k,
                    (t_exl3 / t_q4k <= 1.5) ? "PASS(<=1.5x)" : "OVER 1.5x");
        fleet.dev(0).free(d_q4k);
    }

    fleet.dev(0).free(d_codes); fleet.dev(0).free(d_x); fleet.dev(0).free(d_xs);
    fleet.dev(0).free(d_xh); fleet.dev(0).free(d_y); fleet.dev(0).free(d_acc); fleet.dev(0).free(d_acch);

    std::printf("ie-exl3-test: %s\n", rc == 0 ? "GATE PASS (cosine)" : "GATE FAIL");
    return rc;
}
