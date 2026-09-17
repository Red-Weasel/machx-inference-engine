// tests/unit/dequant_bt_test.cpp — P3d. Validate the new Q5_K/Q8_0 → B^T [K,N]
// dequant kernels against the proven flat dequant via the transpose identity:
//   out_bt[k*N + n] == out_flat[n*K + k]
// (the flat dequant lays a [N columns × K/blk] weight as [N][K]; _to_Bt as [K][N]).
#include "ie/dequant.hpp"
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
    if (!found) { std::fputs("dequant_bt_test: no GPU — SKIP\n", stderr); std::exit(0); }
    return sycl::queue(dev);
}

// Fill `nblk` quant blocks of `bytes` each with random bytes; force the leading
// fp16 d (and dmin for K-quants) to a sane small magnitude (no NaN/Inf).
template <bool HasDmin>
std::vector<uint8_t> rand_blocks(size_t nblk, size_t bytes, std::mt19937& rng) {
    std::vector<uint8_t> buf(nblk * bytes);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> mag(0.01f, 0.5f);
    for (auto& b : buf) b = uint8_t(byte(rng));
    for (size_t i = 0; i < nblk; ++i) {
        uint16_t d = ie::fp32_to_fp16(mag(rng));
        std::memcpy(buf.data() + i * bytes, &d, 2);
        if (HasDmin) { uint16_t dm = ie::fp32_to_fp16(mag(rng)); std::memcpy(buf.data() + i * bytes + 2, &dm, 2); }
    }
    return buf;
}

// Run a (_to_Bt vs flat) pair and check the transpose identity. Returns #mismatch.
int check(sycl::queue& q, const std::vector<uint8_t>& W, size_t blk_bytes,
          uint32_t K, uint32_t N, const char* name,
          sycl::event (*bt)(sycl::queue&, const void*, sycl::half*, uint32_t, uint32_t, const std::vector<sycl::event>&),
          sycl::event (*flat)(sycl::queue&, const void*, sycl::half*, size_t, const std::vector<sycl::event>&)) {
    const uint64_t nelem = uint64_t(K) * N;
    void* d_in = sycl::malloc_device(W.size(), q);
    sycl::half* d_bt   = sycl::malloc_device<sycl::half>(nelem, q);
    sycl::half* d_flat = sycl::malloc_device<sycl::half>(nelem, q);
    q.memcpy(d_in, W.data(), W.size()).wait();
    bt(q, d_in, d_bt, K, N, {}).wait();
    flat(q, d_in, d_flat, nelem, {}).wait();
    std::vector<sycl::half> hbt(nelem), hflat(nelem);
    q.memcpy(hbt.data(), d_bt, nelem * 2).wait();
    q.memcpy(hflat.data(), d_flat, nelem * 2).wait();
    sycl::free(d_in, q); sycl::free(d_bt, q); sycl::free(d_flat, q);

    int mism = 0;
    for (uint32_t k = 0; k < K && mism < 5; ++k)
        for (uint32_t n = 0; n < N; ++n)
            if (float(hbt[uint64_t(k) * N + n]) != float(hflat[uint64_t(n) * K + k])) {
                if (mism < 3) std::fprintf(stderr, "%s mismatch k=%u n=%u: bt=%.5f flat=%.5f\n",
                    name, k, n, float(hbt[uint64_t(k)*N+n]), float(hflat[uint64_t(n)*K+k]));
                ++mism;
            }
    std::printf("  %-7s K=%u N=%u: %s\n", name, K, N, mism ? "FAIL" : "OK");
    (void)blk_bytes;
    return mism;
}

}  // namespace

int main() {
    sycl::queue q = make_queue();
    std::mt19937 rng(20260611);
    const uint32_t K = 512, N = 128;
    int fails = 0;

    // Q5_K (176 B, 256 elem/blk): K/256 blocks per column.
    {
        const auto W = rand_blocks<true>(size_t(N) * (K / 256), sizeof(ie::block_q5_K), rng);
        fails += check(q, W, sizeof(ie::block_q5_K), K, N, "Q5_K",
                       ie::dequant_q5_K_to_Bt, ie::dequant_q5_K);
    }
    // Q8_0 (34 B, 32 elem/blk): K/32 blocks per column.
    {
        const auto W = rand_blocks<false>(size_t(N) * (K / 32), sizeof(ie::block_q8_0), rng);
        fails += check(q, W, sizeof(ie::block_q8_0), K, N, "Q8_0",
                       ie::dequant_q8_0_to_Bt, ie::dequant_q8_0);
    }

    if (fails) { std::fprintf(stderr, "dequant_bt_test: %d mismatch\n", fails); return 1; }
    std::printf("dequant_bt_test: all OK (Q5_K + Q8_0 _to_Bt match flat dequant)\n");
    return 0;
}
