// tests/unit/gemv_rows_test.cpp — DSpark P2 (docs/deepseek41/55): the ROWS kernels. (1) gemv_fp8_rows at M = 1 is
// bit-identical to the shipped one-row table decode (gemv_fp8_e4m3_f16_variant(1)); (2) every row of gemv_fp8_rows at
// M = 2..8 is bit-identical to the M = 1 result for that row (the same arithmetic per row, the weights read once);
// (3) gemv_f16_rows: within 1e-5 of a double reference, every row at M = 2..8 bit-identical to M = 1, and the column
// group mode (the block-diagonal o_a) equals the per-group reference.
#include "ie/deepseek41_upload.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {
int g_fail = 0;
void check(bool ok, const char* w, const std::string& d = "") { std::printf("%s%s%s\n", ok ? "[ ok ] " : "[FAIL] ", w, d.empty() ? "" : ("  (" + d + ")").c_str()); if (!ok) ++g_fail; }
double e4m3_host(uint8_t b) { const int s = b >> 7, e = (b >> 3) & 0xF, m = b & 7; const double mag = e == 0 ? m * std::ldexp(1.0, -9) : (1.0 + m / 8.0) * std::ldexp(1.0, e - 7); return s ? -mag : mag; }
double e8m0_host(uint8_t b) { return std::ldexp(1.0, int(b) - 127); }
}  // namespace

int main() {
    std::vector<sycl::device> devs;
    for (const auto& p : sycl::platform::get_platforms()) { if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) continue; for (const auto& d : p.get_devices(sycl::info::device_type::gpu)) if (d.get_info<sycl::info::device::name>().find("Arc") != std::string::npos) devs.push_back(d); }
    if (devs.empty()) { std::printf("no Arc GPU: skipped\n"); return 77; }
    sycl::queue q(sycl::context(devs[0]), devs[0], sycl::property_list{sycl::property::queue::in_order{}});
    std::mt19937 rng(23);
    const uint32_t MMAX = 8;
    // ---- FP8: the model's shapes (a subset), M rows of activations
    const std::pair<uint32_t, uint32_t> shapes[] = {{1280, 5120}, {32768, 1280}, {512, 5120}, {2304, 5120}, {128, 5120}};
    for (auto [N, K] : shapes) {
        const uint32_t SK = K / 32, SN = N / 32;
        std::vector<uint8_t> w(size_t(N) * K), sc(size_t(SN) * SK);
        for (auto& b : w) { uint8_t v; do { v = uint8_t(rng() & 0xFF); } while ((v & 0x7F) == 0x7F); b = v; }
        for (auto& s : sc) s = uint8_t(118 + (rng() % 12));
        std::vector<sycl::half> x(size_t(MMAX) * K);
        for (auto& v : x) v = sycl::half(float(int(rng() % 2001) - 1000) / 1000.f);
        uint8_t* dw = sycl::malloc_device<uint8_t>(w.size(), q); uint8_t* dsc = sycl::malloc_device<uint8_t>(sc.size(), q);
        sycl::half* dx = sycl::malloc_device<sycl::half>(x.size(), q); float* dy1 = sycl::malloc_device<float>(N, q); float* dym = sycl::malloc_device<float>(size_t(MMAX) * N, q);
        q.memcpy(dw, w.data(), w.size()); q.memcpy(dsc, sc.data(), sc.size()); q.memcpy(dx, x.data(), x.size() * 2); q.wait();
        // (1) M = 1 rows vs the shipped table decode, per row of x
        std::vector<float> ref(size_t(MMAX) * N), got(size_t(MMAX) * N);
        for (uint32_t m = 0; m < MMAX; ++m) { ie::gemv_fp8_e4m3_f16_variant(q, dx + size_t(m) * K, dw, dsc, dy1, K, N, 1).wait(); q.memcpy(ref.data() + size_t(m) * N, dy1, N * 4).wait(); }
        ie::gemv_fp8_rows(q, dx, K, dw, dsc, dy1, 1, K, N).wait(); q.memcpy(got.data(), dy1, N * 4).wait();
        uint32_t nd = 0; for (uint32_t n = 0; n < N; ++n) if (got[n] != ref[n]) ++nd;
        char b1[96]; std::snprintf(b1, sizeof b1, "N %u K %u: %u of %u differ", N, K, nd, N);
        check(nd == 0, "FP8 rows at M = 1 bit-identical to the shipped one-row table decode", b1);
        // (2) M = 2..8: every row equals the M = 1 result for that row
        for (uint32_t M = 2; M <= MMAX; ++M) {
            ie::gemv_fp8_rows(q, dx, K, dw, dsc, dym, M, K, N).wait(); q.memcpy(got.data(), dym, size_t(M) * N * 4).wait();
            uint32_t ndm = 0; for (size_t i = 0; i < size_t(M) * N; ++i) if (got[i] != ref[i]) ++ndm;
            char b2[96]; std::snprintf(b2, sizeof b2, "N %u K %u M %u: %u of %zu differ", N, K, M, ndm, size_t(M) * N);
            check(ndm == 0, "FP8 rows: every row at M > 1 bit-identical to the one-row result", b2);
        }
        sycl::free(dw, q); sycl::free(dsc, q); sycl::free(dx, q); sycl::free(dy1, q); sycl::free(dym, q);
    }
    // ---- fp16 rows: a double reference, M-identity, and the column-group mode (o_a: 8 groups of [1024, 4096])
    {
        const uint32_t G = 8, OPG = 1024, IPG = 4096, N = G * OPG, K = IPG, QH = G * IPG;
        std::vector<sycl::half> w(size_t(N) * K), x(size_t(MMAX) * QH); std::vector<double> wd(w.size()), xd(x.size());
        for (size_t i = 0; i < w.size(); ++i) { const float v = float(int(rng() % 2001) - 1000) / 4000.f; w[i] = sycl::half(v); wd[i] = double(float(w[i])); }
        for (size_t i = 0; i < x.size(); ++i) { const float v = float(int(rng() % 2001) - 1000) / 1000.f; x[i] = sycl::half(v); xd[i] = double(float(x[i])); }
        sycl::half* dw = sycl::malloc_device<sycl::half>(w.size(), q); sycl::half* dx = sycl::malloc_device<sycl::half>(x.size(), q); float* dy = sycl::malloc_device<float>(size_t(MMAX) * N, q);
        q.memcpy(dw, w.data(), w.size() * 2); q.memcpy(dx, x.data(), x.size() * 2); q.wait();
        // the grouped reference in double: y[m][n] = sum_k w[n][k] * x[m][(n / OPG) * IPG + k]
        std::vector<double> refd(size_t(MMAX) * N);
        for (uint32_t m = 0; m < MMAX; ++m) for (uint32_t n = 0; n < N; ++n) { double a = 0; const size_t xo = size_t(m) * QH + size_t(n / OPG) * IPG; for (uint32_t k = 0; k < K; ++k) a += wd[size_t(n) * K + k] * xd[xo + k]; refd[size_t(m) * N + n] = a; }
        double smax = 0; for (double v : refd) smax = std::max(smax, std::fabs(v));
        std::vector<float> y1(size_t(MMAX) * N), ym(size_t(MMAX) * N);
        for (uint32_t m = 0; m < MMAX; ++m) { ie::gemv_f16_rows(q, dx + size_t(m) * QH, QH, dw, dy, 1, K, N, OPG).wait(); q.memcpy(y1.data() + size_t(m) * N, dy, N * 4).wait(); }
        double worst = 0; for (size_t i = 0; i < y1.size(); ++i) worst = std::max(worst, std::fabs(double(y1[i]) - refd[i]) / smax);
        char b3[96]; std::snprintf(b3, sizeof b3, "grouped o_a shape, M = 1 per row: %.2e of the max output", worst);
        check(worst < 1e-5, "fp16 rows (column groups) within 1e-5 of the double reference", b3);
        for (uint32_t M = 2; M <= MMAX; ++M) {
            ie::gemv_f16_rows(q, dx, QH, dw, dy, M, K, N, OPG).wait(); q.memcpy(ym.data(), dy, size_t(M) * N * 4).wait();
            uint32_t ndm = 0; for (size_t i = 0; i < size_t(M) * N; ++i) if (ym[i] != y1[i]) ++ndm;
            char b4[96]; std::snprintf(b4, sizeof b4, "M %u: %u of %zu differ", M, ndm, size_t(M) * N);
            check(ndm == 0, "fp16 rows: every row at M > 1 bit-identical to the one-row result", b4);
        }
        sycl::free(dw, q); sycl::free(dx, q); sycl::free(dy, q);
    }
    std::printf("gemv_rows_test: %s\n", g_fail ? "FAIL" : "all OK");
    return g_fail ? 1 : 0;
}
