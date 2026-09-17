// tests/unit/gemv_fp8_test.cpp — the FP8 (E4M3 + 32x32 E8M0) T = 1 GEMV (docs/deepseek41/43 step 1): on the
// model's dense shapes (the seven of docs/43 plus the indexer's q_b and the engram kv, gate 17 finding 2) with random bytes and scales, (1) the kernel's fp32 output is within 1e-5 of a
// double-precision dot over the same decoded weights (relative to the largest output), and (2) it agrees with the existing path --
// ds41_dense_dequant_f16 + gemm_nt_f16_onednn -- within 2e-3 (the two paths' summation orders differ).
#include "ie/deepseek41_upload.hpp"
#include "ie/ops.hpp"

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
    std::mt19937 rng(11);
    // the dense shapes the FP8 path runs (gate 17 finding 2: idx_wq_b and engram_wkv are converted and were untested; {8192, 4096} is o_a, fp16 today, Phase 19's shape)
    const std::pair<uint32_t, uint32_t> shapes[] = {{1280, 5120}, {32768, 1280}, {512, 5120}, {5120, 8192}, {8192, 4096}, {2304, 5120}, {5120, 2304}, {4096, 1280}, {25600, 6144}, {128, 5120}};   // {128, 5120}: the N <= 256 (4 columns per work-group) branch (gate 18 finding 4; N % 32 != 0 cannot occur: the scale grid is 32 x 32)
    for (auto [N, K] : shapes) {
        const uint32_t SK = K / 32, SN = N / 32;
        std::vector<uint8_t> w(size_t(N) * K), sc(size_t(SN) * SK);
        for (auto& b : w) { uint8_t v; do { v = uint8_t(rng() & 0xFF); } while ((v & 0x7F) == 0x7F); b = v; }   // no NaN codes
        for (auto& s : sc) s = uint8_t(118 + (rng() % 12));                                                    // scales 2^-9 .. 2^2
        std::vector<sycl::half> x(K); std::vector<double> xd(K);
        for (uint32_t k = 0; k < K; ++k) { const float v = float(int(rng() % 2001) - 1000) / 1000.f; x[k] = sycl::half(v); xd[k] = double(float(x[k])); }
        // the double reference over the decoded weights
        std::vector<double> ref(N);
        for (uint32_t n = 0; n < N; ++n) { double acc = 0; for (uint32_t k = 0; k < K; ++k) acc += e4m3_host(w[size_t(n) * K + k]) * e8m0_host(sc[size_t(n / 32) * SK + k / 32]) * xd[k]; ref[n] = acc; }
        uint8_t* dw = sycl::malloc_device<uint8_t>(w.size(), q); uint8_t* dsc = sycl::malloc_device<uint8_t>(sc.size(), q);
        sycl::half* dx = sycl::malloc_device<sycl::half>(K, q); float* dy = sycl::malloc_device<float>(N, q);
        q.memcpy(dw, w.data(), w.size()); q.memcpy(dsc, sc.data(), sc.size()); q.memcpy(dx, x.data(), K * 2); q.wait();
        ie::gemv_fp8_e4m3_f16(q, dx, dw, dsc, dy, K, N).wait();
        std::vector<float> y(N); q.memcpy(y.data(), dy, N * 4).wait();
        double worst_rel = 0, scale_max = 0; for (uint32_t n = 0; n < N; ++n) scale_max = std::max(scale_max, std::fabs(ref[n]));
        for (uint32_t n = 0; n < N; ++n) worst_rel = std::max(worst_rel, std::fabs(double(y[n]) - ref[n]) / scale_max);
        char b1[96]; std::snprintf(b1, sizeof b1, "N %u K %u: %.2e of the max output", N, K, worst_rel);
        check(worst_rel < 1e-5, "FP8 GEMV within 1e-5 of the double reference over the decoded weights (fp32 out)", b1);
        // docs/45 criterion 1: the packed decode is bit-identical to the scalar decode (same products, same order)
        { float* dy3 = sycl::malloc_device<float>(N, q); float* dy4 = sycl::malloc_device<float>(N, q);
          ie::gemv_fp8_e4m3_f16_scalar(q, dx, dw, dsc, dy3, K, N).wait();
          std::vector<float> y3(N), y4(N); q.memcpy(y3.data(), dy3, N * 4).wait();
          for (int dec = 1; dec <= 3; ++dec) {   // docs/46 term 3: every table variant (1 fp16 table, 2 fp32 table, 3 + the block's x staged) bit-identical to the scalar
              ie::gemv_fp8_e4m3_f16_variant(q, dx, dw, dsc, dy4, K, N, dec).wait(); q.memcpy(y4.data(), dy4, N * 4).wait();
              uint32_t ndiff = 0; double wd = 0; for (uint32_t n = 0; n < N; ++n) if (y3[n] != y4[n]) { ++ndiff; wd = std::max(wd, std::fabs(double(y3[n]) - y4[n]) / scale_max); }
              char b3[112]; std::snprintf(b3, sizeof b3, "variant %d: %u of %u outputs differ, worst %.2e of the max output", dec, ndiff, N, wd);
              check(ndiff == 0, "the table E4M3 decode is bit-identical to the scalar decode", b3);
          }
          sycl::free(dy3, q); sycl::free(dy4, q); }
        // the existing path: dequant to fp16, then the oneDNN fp16 GEMM at M = 1
        sycl::half* dwf = sycl::malloc_device<sycl::half>(w.size(), q); float* dy2 = sycl::malloc_device<float>(N, q);
        ie::ds41_dense_dequant_f16(q, dw, dsc, N, K, 32, 32, dwf).wait();
        ie::gemm_nt_f16_onednn(q, dx, dwf, dy2, 1, N, K).wait();
        std::vector<float> y2(N); q.memcpy(y2.data(), dy2, N * 4).wait();
        double worst2 = 0; for (uint32_t n = 0; n < N; ++n) worst2 = std::max(worst2, std::fabs(double(y[n]) - double(y2[n])) / scale_max);
        char b2[64]; std::snprintf(b2, sizeof b2, "rel %.2e", worst2);
        check(worst2 < 2e-3, "FP8 GEMV == dequant + oneDNN fp16 GEMM (M = 1) within 2e-3 of the max output", b2);
        sycl::free(dw, q); sycl::free(dsc, q); sycl::free(dx, q); sycl::free(dy, q); sycl::free(dwf, q); sycl::free(dy2, q);
    }
    // the weight decode equals the header's device decode: every byte through ds41_e4m3 on the host build vs the double formula
    { bool all = true; for (uint32_t b = 0; b < 256; ++b) { if ((b & 0x7F) == 0x7F) continue; all &= std::fabs(double(ie::ds41_e4m3(uint8_t(b))) - e4m3_host(uint8_t(b))) <= 1e-12; } check(all, "ds41_e4m3 equals the E4M3 formula on all 254 finite codes"); }
    std::printf("%s\n", g_fail ? "gemv_fp8_test: FAILURE(S)" : "gemv_fp8_test: all OK");
    return g_fail ? 1 : 0;
}
