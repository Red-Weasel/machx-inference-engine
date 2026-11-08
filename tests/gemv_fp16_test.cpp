// tests/gemv_fp16_test.cpp — correctness gate for the dense fp16 GEMV (P2 Task 2).
//
// y[n] = sum_k A[k] * W_kn[k*N + n]   (W_kn is [K, N] fp16 row-major — the
// load-time transpose of GGUF's [N, K] attn_v layout).
//
// CPU reference accumulates in double over the SAME fp16-rounded inputs, so
// the only expected divergence is the GPU's fp32 accumulation order plus the
// final fp16 store rounding (~5e-4 rel) → gate at max_rel < 2e-3 per plan.
// No GGUF required: pure random inputs, fixed seeds.

#include "ie/ops.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr const char* G = "\033[32m";
constexpr const char* R = "\033[31m";
constexpr const char* Z = "\033[0m";

sycl::queue make_queue() {
    sycl::device dev;
    bool found = false;
    for (const auto& d : sycl::device::get_devices()) {
        if (d.is_gpu() && d.get_info<sycl::info::device::name>().find("0xe223") != std::string::npos) {
            dev = d; found = true; break;
        }
    }
    if (!found) for (const auto& d : sycl::device::get_devices()) if (d.is_gpu()) { dev = d; found = true; break; }
    if (!found) { std::fputs("no GPU\n", stderr); std::exit(2); }
    return sycl::queue(dev, sycl::property::queue::enable_profiling{});
}

bool test_shape(sycl::queue& q, uint32_t K, uint32_t N, uint32_t seed,
                bool report_perf) {
    std::vector<sycl::half> hA(K);
    std::vector<sycl::half> hW(size_t(K) * N);
    std::vector<sycl::half> hY(N);
    {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
        for (auto& v : hA) v = sycl::half(dist(rng));
        for (auto& v : hW) v = sycl::half(dist(rng));
    }

    // CPU reference: double accumulate over the fp16-rounded inputs.
    std::vector<double> y_ref(N, 0.0);
    for (uint32_t k = 0; k < K; ++k) {
        const double a = double(float(hA[k]));
        const sycl::half* wrow = hW.data() + size_t(k) * N;
        for (uint32_t n = 0; n < N; ++n) y_ref[n] += a * double(float(wrow[n]));
    }

    auto* dA = sycl::malloc_device<sycl::half>(K, q);
    auto* dW = sycl::malloc_device<sycl::half>(size_t(K) * N, q);
    auto* dY = sycl::malloc_device<sycl::half>(N, q);
    q.memcpy(dA, hA.data(), K * sizeof(sycl::half)).wait();
    q.memcpy(dW, hW.data(), size_t(K) * N * sizeof(sycl::half)).wait();

    ie::gemv_fp16(q, dA, dW, dY, K, N).wait_and_throw();
    q.memcpy(hY.data(), dY, N * sizeof(sycl::half)).wait();

    // Diff. rel uses a 0.05 abs floor so near-zero outputs (fp16 spacing +
    // fp32 ordering noise dominates) don't blow up the ratio.
    float max_abs = 0.f;
    float max_rel = 0.f;
    uint32_t worst = 0;
    for (uint32_t n = 0; n < N; ++n) {
        const float a = float(y_ref[n]);
        const float b = float(hY[n]);
        const float d = std::fabs(a - b);
        const float r = d / std::fmax(std::fabs(a), 0.05f);
        if (r > max_rel) { max_rel = r; worst = n; }
        if (d > max_abs) max_abs = d;
    }
    const bool ok = max_rel < 2e-3f;
    std::printf("  %sgemv_fp16 K=%5u N=%5u%s  max_abs=%-9.5f max_rel=%-9.3g worst@%u (exp=%g got=%g)  %s\n",
                ok ? G : R, K, N, Z, max_abs, max_rel, worst,
                float(y_ref[worst]), float(hY[worst]), ok ? "OK" : "FAIL");

    if (report_perf) {
        // Decode streams a DIFFERENT attn_v tensor every call — W never sits
        // in L2 (18 MB on B580).  Rotate among enough W copies that the
        // working set exceeds L2, otherwise this measures cache bandwidth.
        constexpr int N_BUFS = 5;   // 5 × 8.4 MB = 42 MB >> 18 MB L2
        constexpr int REPS   = 50;
        std::vector<sycl::half*> dWs(N_BUFS);
        for (int b = 0; b < N_BUFS; ++b) {
            dWs[b] = sycl::malloc_device<sycl::half>(size_t(K) * N, q);
            q.memcpy(dWs[b], hW.data(), size_t(K) * N * sizeof(sycl::half)).wait();
        }
        for (int b = 0; b < N_BUFS; ++b)  // warmup
            ie::gemv_fp16(q, dA, dWs[b], dY, K, N).wait();
        uint64_t best_ns = UINT64_MAX, total_ns = 0;
        for (int i = 0; i < REPS; ++i) {
            auto e = ie::gemv_fp16(q, dA, dWs[i % N_BUFS], dY, K, N);
            e.wait();
            const uint64_t t0 = e.get_profiling_info<sycl::info::event_profiling::command_start>();
            const uint64_t t1 = e.get_profiling_info<sycl::info::event_profiling::command_end>();
            const uint64_t ns = (t1 > t0) ? (t1 - t0) : 0;
            best_ns = std::min(best_ns, ns);
            total_ns += ns;
        }
        for (int b = 0; b < N_BUFS; ++b) sycl::free(dWs[b], q);
        const double bytes = double(size_t(K) * N + K + N) * sizeof(sycl::half);
        const double gbps_best = bytes / double(best_ns);
        const double gbps_avg  = bytes * REPS / double(total_ns);
        std::printf("    perf (DRAM-stream): best %.1f us  avg %.1f us  -> %.0f GB/s best / %.0f GB/s avg (%.1f MB/call)\n",
                    double(best_ns) * 1e-3, double(total_ns) * 1e-3 / REPS,
                    gbps_best, gbps_avg, bytes * 1e-6);
    }

    sycl::free(dA, q); sycl::free(dW, q); sycl::free(dY, q);
    return ok;
}

// gemv_fp16_rows(R) must equal R gemv_fp16 calls BIT FOR BIT (the buckets
// replay the T=1 chain per row; R=1 IS gemv_fp16). perf: rows(R) vs R solo.
bool test_rows(sycl::queue& q, uint32_t K, uint32_t N, uint32_t seed, bool report_perf) {
    constexpr uint32_t RM = 8;
    std::vector<sycl::half> hX(size_t(RM) * K), hW(size_t(K) * N);
    {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
        for (auto& v : hX) v = sycl::half(dist(rng));
        for (auto& v : hW) v = sycl::half(dist(rng));
    }
    auto* dX = sycl::malloc_device<sycl::half>(size_t(RM) * K, q);
    auto* dW = sycl::malloc_device<sycl::half>(size_t(K) * N, q);
    auto* dS = sycl::malloc_device<sycl::half>(size_t(RM) * N, q);   // R solo calls
    auto* dR = sycl::malloc_device<sycl::half>(size_t(RM) * N, q);   // one rows call
    q.memcpy(dX, hX.data(), size_t(RM) * K * sizeof(sycl::half)).wait();
    q.memcpy(dW, hW.data(), size_t(K) * N * sizeof(sycl::half)).wait();
    for (uint32_t r = 0; r < RM; ++r) ie::gemv_fp16(q, dX + size_t(r) * K, dW, dS + size_t(r) * N, K, N);
    q.wait_and_throw();
    std::vector<uint16_t> hS(size_t(RM) * N), hR(size_t(RM) * N);
    q.memcpy(hS.data(), dS, hS.size() * 2).wait();
    bool ok = true;
    std::printf("  gemv_fp16_rows K=%5u N=%6u  bits differing from R gemv_fp16 calls:", K, N);
    for (uint32_t R : {1u, 2u, 3u, 4u, 8u}) {
        q.memset(dR, 0xff, size_t(RM) * N * 2).wait();
        ie::gemv_fp16_rows(q, dX, K, dW, dR, N, K, N, R).wait_and_throw();
        q.memcpy(hR.data(), dR, hR.size() * 2).wait();
        uint32_t d = 0;
        for (size_t i = 0; i < size_t(R) * N; ++i) d += hS[i] != hR[i];
        std::printf(" R=%u %u", R, d);
        ok = ok && d == 0;
    }
    std::printf("  %s%s%s\n", ok ? G : R, ok ? "OK" : "FAIL", Z);
    if (report_perf) {
        constexpr int N_BUFS = 16, REPS = 48;   // 16 copies of W: every call streams it from DRAM
        std::vector<sycl::half*> dWs(N_BUFS);
        for (auto& p : dWs) {
            p = sycl::malloc_device<sycl::half>(size_t(K) * N, q);
            q.memcpy(p, hW.data(), size_t(K) * N * sizeof(sycl::half)).wait();
        }
        auto ns_of = [](const sycl::event& e) {
            const uint64_t t0 = e.get_profiling_info<sycl::info::event_profiling::command_start>();
            const uint64_t t1 = e.get_profiling_info<sycl::info::event_profiling::command_end>();
            return t1 > t0 ? t1 - t0 : 0;
        };
        for (uint32_t R : {1u, 2u, 4u, 8u}) {
            uint64_t t_rows = 0, t_solo = 0;
            for (int i = -N_BUFS; i < REPS; ++i) {   // the first N_BUFS warm up
                const sycl::half* W = dWs[(i + N_BUFS) % N_BUFS];
                auto e = ie::gemv_fp16_rows(q, dX, K, W, dR, N, K, N, R);
                e.wait();
                if (i >= 0) t_rows += ns_of(e);
                for (uint32_t r = 0; r < R; ++r) {
                    auto s = ie::gemv_fp16(q, dX + size_t(r) * K, dWs[(i + N_BUFS + 7) % N_BUFS], dS + size_t(r) * N, K, N);
                    s.wait();
                    if (i >= 0) t_solo += ns_of(s);
                }
            }
            std::printf("    R=%u  rows %6.1f us   %u x gemv_fp16 %6.1f us\n", R, double(t_rows) * 1e-3 / REPS, R,
                        double(t_solo) * 1e-3 / REPS);
        }
        for (auto* p : dWs) sycl::free(p, q);
    }
    sycl::free(dX, q); sycl::free(dW, q); sycl::free(dS, q); sycl::free(dR, q);
    return ok;
}

}  // namespace

int main() {
    auto q = make_queue();
    std::printf("Device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    int fails = 0;
    // The real qwen3-8b attn_v decode shape (W_kn = [K=4096, N=1024]).
    if (!test_shape(q, 4096, 1024, 0xA53, /*report_perf=*/true)) ++fails;
    // Odd shapes per plan: no K/N alignment assumptions.
    if (!test_shape(q, 192, 7, 0xB17, false)) ++fails;
    // Odd N (vec fast path disabled) + K not a multiple of the SG slice.
    if (!test_shape(q, 333, 129, 0xC29, false)) ++fails;
    // K=1 / N=1 degenerate corners.
    if (!test_shape(q, 1, 1, 0xD31, false)) ++fails;
    // The rows buckets vs R solo calls, bit for bit (the Flash-Next router
    // [2560, 512] with perf, the qwen3 attn_v shape, odd / tiny / tail shapes).
    if (!test_rows(q, 2560, 512, 0xE41, /*report_perf=*/true)) ++fails;
    if (!test_rows(q, 4096, 1024, 0xA53, false)) ++fails;
    if (!test_rows(q, 2560, 4099, 0xE43, false)) ++fails;
    if (!test_rows(q, 333, 129, 0xC29, false)) ++fails;
    if (!test_rows(q, 192, 7, 0xB17, false)) ++fails;
    if (!test_rows(q, 1, 1, 0xD31, false)) ++fails;

    std::printf("\n%s\n", fails == 0 ? "ALL PASS" : "FAILURES");
    return fails;
}
