// tests/unit/gemv_iq_raw_test.cpp — parity-pin the raw-layout IQ3_XXS /
// IQ4_XS / Q3_K GEMVs and their _to_Bt dequants (gemv_iq_raw.cpp) against
// the host references in ie::ref (bit-exact vs ggml).
//   GEMV: y[n] vs fp64 dot over ref-dequanted column (tolerance = fp32
//         accumulation-order noise, rel 5e-3 + abs 1e-3).
//   Bt:   out[k*N+n] vs half(ref_flat[n*K+k]) (tolerance = one fp16 ulp
//         class via rel 2e-3 + abs 2e-4 — fma contraction may differ).
#include "ie/dequant_ref.hpp"
#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

sycl::queue make_queue() {
    sycl::device dev;
    bool found = false;
    for (const auto& d : sycl::device::get_devices())
        if (d.is_gpu()) { dev = d; found = true; break; }
    if (!found) { std::fputs("gemv_iq_raw_test: no GPU — SKIP\n", stderr); std::exit(0); }
    return sycl::queue(dev);
}

std::vector<uint8_t> rand_blocks(size_t nblk, size_t bytes, size_t d_off,
                                 std::mt19937& rng) {
    std::vector<uint8_t> buf(nblk * bytes);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> mag(0.01f, 0.5f);
    for (auto& b : buf) b = uint8_t(byte(rng));
    for (size_t i = 0; i < nblk; ++i) {
        const uint16_t d = ie::fp32_to_fp16(mag(rng));
        std::memcpy(buf.data() + i * bytes + d_off, &d, 2);
    }
    return buf;
}

using RefFn  = void (*)(const void*, size_t, float*);
using GemvFn = sycl::event (*)(sycl::queue&, const sycl::half*, const void*,
                               sycl::half*, uint32_t, uint32_t,
                               const std::vector<sycl::event>&);
using BtFn   = sycl::event (*)(sycl::queue&, const void*, sycl::half*,
                               uint32_t, uint32_t,
                               const std::vector<sycl::event>&);

int run_one(sycl::queue& q, const char* name, size_t blk_bytes, size_t d_off,
            uint32_t K, uint32_t N, RefFn ref, GemvFn gemv, BtFn bt,
            std::mt19937& rng) {
    const size_t nblk = size_t(K / 256) * N;
    auto W = rand_blocks(nblk, blk_bytes, d_off, rng);

    std::vector<sycl::half> A(K);
    std::uniform_real_distribution<float> av(-1.f, 1.f);
    for (auto& a : A) a = sycl::half(av(rng));

    // Host truth: per-column ref dequant + fp64 dot.
    std::vector<float> col(K);
    std::vector<double> truth(N);
    for (uint32_t n = 0; n < N; ++n) {
        ref(W.data() + size_t(n) * (K / 256) * blk_bytes, K, col.data());
        double s = 0.0;
        for (uint32_t k = 0; k < K; ++k) s += double(col[k]) * double(float(A[k]));
        truth[n] = s;
    }

    void* d_w = sycl::malloc_device(W.size(), q);
    sycl::half* d_a = sycl::malloc_device<sycl::half>(K, q);
    sycl::half* d_y = sycl::malloc_device<sycl::half>(N, q);
    sycl::half* d_bt = sycl::malloc_device<sycl::half>(uint64_t(K) * N, q);
    q.memcpy(d_w, W.data(), W.size()).wait();
    q.memcpy(d_a, A.data(), K * 2).wait();

    gemv(q, d_a, d_w, d_y, K, N, {}).wait();
    if (bt) bt(q, d_w, d_bt, K, N, {}).wait();

    std::vector<sycl::half> y(N), hbt(uint64_t(K) * N);
    q.memcpy(y.data(), d_y, N * 2).wait();
    q.memcpy(hbt.data(), d_bt, uint64_t(K) * N * 2).wait();
    sycl::free(d_w, q); sycl::free(d_a, q); sycl::free(d_y, q); sycl::free(d_bt, q);

    int bad = 0;
    for (uint32_t n = 0; n < N; ++n) {
        const double got = double(float(y[n])), want = truth[n];
        const double tol = 5e-3 * std::abs(want) + 1e-3 * std::sqrt(double(K));
        if (std::abs(got - want) > tol) {
            if (bad < 3)
                std::fprintf(stderr, "%s gemv n=%u: got %.6f want %.6f\n",
                             name, n, got, want);
            ++bad;
        }
    }
    // Bt vs ref flat (transpose identity, tolerant of fma contraction).
    int badbt = 0;
    for (uint32_t n = 0; bt && n < N && badbt < 5; ++n) {
        ref(W.data() + size_t(n) * (K / 256) * blk_bytes, K, col.data());
        for (uint32_t k = 0; k < K; ++k) {
            const float got = float(hbt[uint64_t(k) * N + n]);
            const float want = col[k];
            if (std::abs(got - want) > 2e-3f * std::abs(want) + 2e-4f) {
                if (badbt < 3)
                    std::fprintf(stderr, "%s bt k=%u n=%u: got %.6f want %.6f\n",
                                 name, k, n, got, want);
                ++badbt;
            }
        }
    }
    std::printf("  %-8s K=%u N=%u: gemv %s (%d bad), bt %s (%d bad)\n",
                name, K, N, bad ? "FAIL" : "OK", bad, badbt ? "FAIL" : "OK", badbt);
    return bad + badbt;
}

}  // namespace

int main() {
    sycl::queue q = make_queue();
    std::mt19937 rng(20260829);
    int fails = 0;
    for (uint32_t K : {1536u, 4096u}) {
        fails += run_one(q, "IQ4_XS", sizeof(ie::block_iq4_xs), 0, K, 64,
                         ie::ref::dequant_iq4_xs_buffer, ie::gemv_iq4_xs,
                         ie::dequant_iq4_xs_to_Bt, rng);
        fails += run_one(q, "Q3_K", sizeof(ie::block_q3_K), 108, K, 64,
                         ie::ref::dequant_q3_K_buffer, ie::gemv_q3_K,
                         ie::dequant_q3_K_to_Bt, rng);
        fails += run_one(q, "IQ3_XXS", sizeof(ie::block_iq3_xxs), 0, K, 64,
                         ie::ref::dequant_iq3_xxs_buffer, ie::gemv_iq3_xxs_raw,
                         ie::dequant_iq3_xxs_to_Bt, rng);
        // Hy4-preview expert dtypes.
        fails += run_one(q, "IQ2_XXS", sizeof(ie::block_iq2_xxs), 0, K, 64,
                         ie::ref::dequant_iq2_xxs_buffer, ie::gemv_iq2_xxs_raw,
                         ie::dequant_iq2_xxs_to_Bt, rng);
        fails += run_one(q, "STQ1_0", sizeof(ie::block_stq1_0), 40, K, 64,
                         ie::ref::dequant_stq1_0_buffer, ie::gemv_stq1_0,
                         ie::dequant_stq1_0_to_Bt, rng);
    }
    if (fails) { std::printf("gemv_iq_raw_test: FAIL (%d)\n", fails); return 1; }
    std::printf("gemv_iq_raw_test: all OK\n");
    return 0;
}
