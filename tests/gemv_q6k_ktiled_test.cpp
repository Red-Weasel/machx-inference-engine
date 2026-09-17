// tests/gemv_q6k_ktiled_test.cpp — correctness gate for the K-tiled int-dot
// Q6_K GEMV (P3b Task 1, dense ffn_down decode).
//
// y[n] = sum_k x[k] * W[k, n]   where W is Q6_K-packed [K, N] column-major
// (each output column n owns K/256 contiguous super-blocks) and x is consumed
// as the Q8_1 activation stream the dense forward feeds the kernel.
//
// The kernel does integer-dot over the device-quantized Q8_1 of x, so the
// CPU reference uses the SAME device-produced q8 stream (copied back to host)
// to compute the exact int-dot math in double — the only expected divergence
// is the kernel's tile-wise fp32 accumulation order + the final fp16 store
// (~5e-4 rel). Gate: max_rel < 2e-3 per plan.
//
// We also cross-check at a FITTING shape (K=4096) against the existing
// whole-column gemv_q6_K_q8 to confirm the K-tiling produces equivalent
// results to the established kernel (same int-dot, only accumulation order
// differs). No GGUF: synthetic random Q6_K weights, fixed seeds.

#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"
#include "ie/dequant_ref.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdio>
#include <cstdint>
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
    for (const auto& d : sycl::device::get_devices())
        if (d.is_gpu()) { dev = d; found = true; break; }
    if (!found) { std::fputs("no GPU\n", stderr); std::exit(2); }
    return sycl::queue(dev, sycl::property::queue::enable_profiling{});
}

// Build one random block_q6_K (random ql/qh/scales, fp16 super-scale).
void random_q6_K(std::mt19937& rng, ie::block_q6_K* b) {
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_int_distribution<int> sc(-64, 63);   // int8 per-16 scale
    for (int i = 0; i < 128; ++i) b->ql[i] = uint8_t(byte(rng));
    for (int i = 0; i < 64;  ++i) b->qh[i] = uint8_t(byte(rng));
    for (int i = 0; i < 16;  ++i) b->scales[i] = int8_t(sc(rng));
    // super-scale ~ small positive fp16; reuse fp16 round-trip via float store.
    const float d = 0.001f + 0.01f * float(byte(rng)) / 255.0f;
    b->d = uint16_t(sycl::bit_cast<uint16_t>(sycl::half(d)));
}

bool test_shape(sycl::queue& q, uint32_t K, uint32_t N, uint32_t seed,
                bool cross_check_whole, bool report_perf) {
    const uint32_t n_blocks_col = K / 256;
    const uint32_t q8_blocks    = K / 32;

    // Random activation x[K] (fp16) and random Q6_K weights [K, N] column-major.
    std::vector<sycl::half> hX(K);
    std::vector<ie::block_q6_K> hW(size_t(n_blocks_col) * N);
    {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-0.6f, 0.6f);
        for (auto& v : hX) v = sycl::half(dist(rng));
        for (auto& b : hW) random_q6_K(rng, &b);
    }

    auto* dX  = sycl::malloc_device<sycl::half>(K, q);
    auto* dQ8 = sycl::malloc_device<ie::block_q8_1x>(q8_blocks, q);
    auto* dW  = sycl::malloc_device<ie::block_q6_K>(size_t(n_blocks_col) * N, q);
    auto* dY  = sycl::malloc_device<sycl::half>(N, q);
    q.memcpy(dX, hX.data(), K * sizeof(sycl::half)).wait();
    q.memcpy(dW, hW.data(),
             size_t(n_blocks_col) * N * sizeof(ie::block_q6_K)).wait();

    // Device-quantize x → Q8_1 (exactly the dense forward's path), then copy
    // the q8 stream back so the CPU reference consumes the identical quants.
    ie::quantize_q8_1(q, dX, dQ8, K).wait_and_throw();
    std::vector<ie::block_q8_1x> hQ8(q8_blocks);
    q.memcpy(hQ8.data(), dQ8, q8_blocks * sizeof(ie::block_q8_1x)).wait();

    // CPU reference: dequant each column's Q6_K to floats, dot with the q8
    // stream (q8.d * q8.qs[i]) in double.
    std::vector<double> y_ref(N, 0.0);
    std::vector<float> col(K);
    for (uint32_t n = 0; n < N; ++n) {
        ie::ref::dequant_q6_K_buffer(&hW[size_t(n) * n_blocks_col], K,
                                     col.data());
        double acc = 0.0;
        for (uint32_t b = 0; b < q8_blocks; ++b) {
            const double d8 = double(hQ8[b].d);
            for (int i = 0; i < 32; ++i)
                acc += double(col[b * 32 + i]) * d8 * double(hQ8[b].qs[i]);
        }
        y_ref[n] = acc;
    }

    ie::gemv_q6_K_q8_ktiled(q, dQ8, dW, dY, K, N).wait_and_throw();
    std::vector<sycl::half> hY(N);
    q.memcpy(hY.data(), dY, N * sizeof(sycl::half)).wait();

    float max_abs = 0.f, max_rel = 0.f;
    uint32_t worst = 0;
    for (uint32_t n = 0; n < N; ++n) {
        const float a = float(y_ref[n]);
        const float bb = float(hY[n]);
        const float d = std::fabs(a - bb);
        const float r = d / std::fmax(std::fabs(a), 0.05f);
        if (r > max_rel) { max_rel = r; worst = n; }
        if (d > max_abs) max_abs = d;
    }
    bool ok = max_rel < 2e-3f;
    std::printf("  %sktiled K=%5u N=%5u%s vs CPU  max_abs=%-9.5f max_rel=%-9.3g "
                "worst@%u (exp=%g got=%g)  %s\n",
                ok ? G : R, K, N, Z, max_abs, max_rel, worst,
                float(y_ref[worst]), float(hY[worst]), ok ? "OK" : "FAIL");

    // Cross-check vs the established whole-column int-dot kernel at fitting K.
    if (cross_check_whole) {
        auto* dY2 = sycl::malloc_device<sycl::half>(N, q);
        ie::gemv_q6_K_q8(q, dQ8, dW, dY2, K, N).wait_and_throw();
        std::vector<sycl::half> hY2(N);
        q.memcpy(hY2.data(), dY2, N * sizeof(sycl::half)).wait();
        float cmax_rel = 0.f; uint32_t cw = 0;
        for (uint32_t n = 0; n < N; ++n) {
            const float a = float(hY2[n]);
            const float bb = float(hY[n]);
            const float r = std::fabs(a - bb) / std::fmax(std::fabs(a), 0.05f);
            if (r > cmax_rel) { cmax_rel = r; cw = n; }
        }
        const bool cok = cmax_rel < 2e-3f;
        std::printf("  %sktiled K=%5u N=%5u%s vs whole-col gemv_q6_K_q8  "
                    "max_rel=%-9.3g worst@%u (%g vs %g)  %s\n",
                    cok ? G : R, K, N, Z, cmax_rel, cw,
                    float(hY2[cw]), float(hY[cw]), cok ? "OK" : "FAIL");
        ok = ok && cok;
        sycl::free(dY2, q);
    }

    if (report_perf) {
        // Rotate enough W copies that the working set exceeds L2 (18 MB B580):
        // K=12288,N=4096 Q6_K = ~98 MB/buf already, so 3 buffers suffice.
        constexpr int N_BUFS = 3, REPS = 30;
        std::vector<ie::block_q6_K*> dWs(N_BUFS);
        for (int b = 0; b < N_BUFS; ++b) {
            dWs[b] = sycl::malloc_device<ie::block_q6_K>(
                size_t(n_blocks_col) * N, q);
            q.memcpy(dWs[b], hW.data(),
                     size_t(n_blocks_col) * N * sizeof(ie::block_q6_K)).wait();
        }
        for (int b = 0; b < N_BUFS; ++b)
            ie::gemv_q6_K_q8_ktiled(q, dQ8, dWs[b], dY, K, N).wait();
        uint64_t best_ns = UINT64_MAX, total_ns = 0;
        for (int i = 0; i < REPS; ++i) {
            auto e = ie::gemv_q6_K_q8_ktiled(q, dQ8, dWs[i % N_BUFS], dY, K, N);
            e.wait();
            const uint64_t t0 = e.get_profiling_info<
                sycl::info::event_profiling::command_start>();
            const uint64_t t1 = e.get_profiling_info<
                sycl::info::event_profiling::command_end>();
            const uint64_t ns = (t1 > t0) ? (t1 - t0) : 0;
            best_ns = std::min(best_ns, ns);
            total_ns += ns;
        }
        for (int b = 0; b < N_BUFS; ++b) sycl::free(dWs[b], q);
        const double bytes =
            double(size_t(n_blocks_col) * N) * sizeof(ie::block_q6_K);
        const double gbps_best = bytes / double(best_ns);
        const double gbps_avg  = bytes * REPS / double(total_ns);
        std::printf("    perf (DRAM-stream): best %.1f us avg %.1f us -> "
                    "%.0f GB/s best / %.0f GB/s avg (%.1f MB weights)\n",
                    double(best_ns) * 1e-3, double(total_ns) * 1e-3 / REPS,
                    gbps_best, gbps_avg, bytes * 1e-6);
    }

    sycl::free(dX, q); sycl::free(dQ8, q); sycl::free(dW, q); sycl::free(dY, q);
    return ok;
}

}  // namespace

int main() {
    auto q = make_queue();
    std::printf("Device: %s\n",
                q.get_device().get_info<sycl::info::device::name>().c_str());

    int fails = 0;
    // The real qwen3-8b ffn_down decode shape: K=F=12288, N=H=4096.
    if (!test_shape(q, 12288, 4096, 0x6C01, /*cross=*/false, /*perf=*/true))
        ++fails;
    // Fitting shape K=4096: cross-check vs the whole-column int-dot kernel.
    if (!test_shape(q, 4096, 2048, 0x6C02, /*cross=*/true, /*perf=*/false))
        ++fails;
    // K not a multiple of K_TILE(2048) → exercises the partial last tile.
    if (!test_shape(q, 12288 + 256, 1024, 0x6C03, /*cross=*/false, false))
        ++fails;
    // Small fitting K + partial tile + odd N (partial last WG).
    if (!test_shape(q, 2304, 257, 0x6C04, /*cross=*/false, false)) ++fails;
    // K exactly one tile.
    if (!test_shape(q, 2048, 512, 0x6C05, /*cross=*/true, false)) ++fails;

    std::printf("\n%s\n", fails == 0 ? "ALL PASS" : "FAILURES");
    return fails;
}
