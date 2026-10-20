// tests/unit/dequant_bt_bench.cpp — the tiled Q4_K / Q5_K -> B^T fp16 dequants (src/ops/dequant_kernels.cpp,
// 2026-10-20) against a verbatim copy of the one-lane-per-(block, column) kernels they replaced: bitwise-identical
// output and time per matrix at GLM-5.3's expert shapes (gate/up K 4096 N 2048, down K 2048 N 4096). GPU1 (09:00.0)
// only; not a CTest: a bench.
#include "ie/dequant.hpp"
#include "ie/quant_blocks.hpp"

#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

float ref_h2f(uint16_t h) { return float(sycl::bit_cast<sycl::half>(h)); }

void ref_scale_min_k4(int j, const uint8_t* q, uint8_t& sc, uint8_t& m) {
    if (j < 4) { sc = q[j] & 0x3F; m = q[j + 4] & 0x3F; }
    else { sc = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4); }
}

// The replaced kernels, verbatim apart from names (src/ops/dequant_kernels.cpp before 2026-10-20).
sycl::event ref_q4k_bt(sycl::queue& q, const void* packed_in, sycl::half* out, uint32_t K, uint32_t N) {
    const uint32_t blocks_per_col = K / 256;
    auto* W = static_cast<const ie::block_q4_K*>(packed_in);
    return q.parallel_for(sycl::nd_range<2>({blocks_per_col, N}, {1, 64}), [=](sycl::nd_item<2> it) {
        const uint32_t b = uint32_t(it.get_group(0));
        const uint32_t n = uint32_t(it.get_global_id(1));
        const ie::block_q4_K& blk = W[uint64_t(n) * blocks_per_col + b];
        sycl::half* dst = out + uint64_t(b) * 256 * N + n;
        const float d_super = ref_h2f(blk.d);
        const float m_super = ref_h2f(blk.dmin);
        #pragma unroll
        for (int sub = 0; sub < 8; ++sub) {
            uint8_t s_raw, m_raw;
            ref_scale_min_k4(sub, blk.scales, s_raw, m_raw);
            const float scale = d_super * float(s_raw);
            const float bias  = m_super * float(m_raw);
            const int g = sub >> 1, hi_nib = sub & 1, qs_base = g * 32, k_base = g * 64 + hi_nib * 32;
            #pragma unroll
            for (int i = 0; i < 32; ++i) {
                const uint8_t qb = blk.qs[qs_base + i];
                const int q4 = hi_nib ? (qb >> 4) : (qb & 0x0F);
                dst[uint64_t(k_base + i) * N] = sycl::half(scale * float(q4) - bias);
            }
        }
    });
}

sycl::event ref_q5k_bt(sycl::queue& q, const void* packed_in, sycl::half* out, uint32_t K, uint32_t N) {
    const uint32_t blocks_per_col = K / 256;
    auto* W = static_cast<const ie::block_q5_K*>(packed_in);
    return q.parallel_for(sycl::nd_range<2>({blocks_per_col, N}, {1, 64}), [=](sycl::nd_item<2> it) {
        const uint32_t b = uint32_t(it.get_group(0));
        const uint32_t n = uint32_t(it.get_global_id(1));
        const ie::block_q5_K& blk = W[uint64_t(n) * blocks_per_col + b];
        sycl::half* dst = out + uint64_t(b) * 256 * N + n;
        const float d = ref_h2f(blk.d), dm = ref_h2f(blk.dmin);
        float sc[8], mn[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) { uint8_t s, m; ref_scale_min_k4(j, blk.scales, s, m); sc[j] = float(s); mn[j] = float(m); }
        #pragma unroll
        for (int k = 0; k < 128; ++k) {
            const int g = k / 32, l = k % 32;
            const uint8_t ql = blk.qs[k], qh = blk.qh[l];
            const int q5_lo = (ql & 0x0F) + ((qh & (1u << (2 * g + 0))) ? 16 : 0);
            const int q5_hi = (ql >>   4) + ((qh & (1u << (2 * g + 1))) ? 16 : 0);
            const float v_lo = d * sc[2 * g + 0] * float(q5_lo) - dm * mn[2 * g + 0];
            const float v_hi = d * sc[2 * g + 1] * float(q5_hi) - dm * mn[2 * g + 1];
            dst[uint64_t(g * 64 + l) * N] = sycl::half(v_lo);
            dst[uint64_t(g * 64 + l + 32) * N] = sycl::half(v_hi);
        }
    });
}

template <class F>
double time_ms(sycl::queue& q, F fn) {
    for (int i = 0; i < 5; ++i) fn();
    q.wait();
    const auto t0 = std::chrono::steady_clock::now();
    constexpr int I = 50;
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
    std::printf("Device: %s %s\n", dev.get_info<sycl::info::device::name>().c_str(),
                dev.get_info<sycl::ext::intel::info::device::pci_address>().c_str());

    std::mt19937 rng(20261020);
    std::uniform_int_distribution<int> u8(0, 255);
    std::uniform_real_distribution<float> mag(0.001f, 0.5f);
    int bad = 0;
    struct Shape { uint32_t K, N; const char* what; };
    for (int q5 = 0; q5 < 2; ++q5)
        for (Shape s : {Shape{4096, 2048, "gate/up"}, Shape{2048, 4096, "down"}}) {
            const size_t bb = q5 ? sizeof(ie::block_q5_K) : sizeof(ie::block_q4_K);
            const size_t nblk = size_t(s.N) * (s.K / 256);
            std::vector<uint8_t> W(nblk * bb);
            for (auto& b : W) b = uint8_t(u8(rng));
            for (size_t i = 0; i < nblk; ++i) {   // sane d / dmin (no NaN / Inf), random everything else
                const uint16_t d = sycl::bit_cast<uint16_t>(sycl::half(mag(rng)));
                const uint16_t m = sycl::bit_cast<uint16_t>(sycl::half(mag(rng)));
                std::memcpy(W.data() + i * bb, &d, 2);
                std::memcpy(W.data() + i * bb + 2, &m, 2);
            }
            const uint64_t ne = uint64_t(s.K) * s.N;
            auto* dW = sycl::malloc_device<uint8_t>(W.size(), q);
            auto* yo = sycl::malloc_device<sycl::half>(ne, q);
            auto* yn = sycl::malloc_device<sycl::half>(ne, q);
            q.memcpy(dW, W.data(), W.size()).wait();
            auto run_old = [&] { q5 ? ref_q5k_bt(q, dW, yo, s.K, s.N) : ref_q4k_bt(q, dW, yo, s.K, s.N); };
            auto run_new = [&] {
                q5 ? ie::dequant_q5_K_to_Bt(q, dW, yn, s.K, s.N, {}) : ie::dequant_q4_K_to_Bt(q, dW, yn, s.K, s.N, {});
            };
            {   // a cold B70 times its first kernels 2-6x slow: hold it busy 500 ms first
                const auto t0 = std::chrono::steady_clock::now();
                while (std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(500)) {
                    run_old();
                    q.wait();
                }
            }
            const double to = time_ms(q, run_old), tn = time_ms(q, run_new);
            std::vector<uint16_t> ho(ne), hn(ne);
            q.memcpy(ho.data(), yo, ne * 2);
            q.memcpy(hn.data(), yn, ne * 2).wait();
            uint64_t diff = 0;
            for (uint64_t i = 0; i < ne; ++i) diff += ho[i] != hn[i];
            if (diff) ++bad;
            const double mb = double(W.size() + ne * 2) / 1e6;
            std::printf("%s K=%u N=%u (%s): old %.3f ms  new %.3f ms  x%.2f  (%.0f GB/s)  %llu of %llu halves differ\n",
                        q5 ? "Q5_K" : "Q4_K", s.K, s.N, s.what, to, tn, to / tn, mb / tn,
                        (unsigned long long)diff, (unsigned long long)ne);
            sycl::free(dW, q); sycl::free(yo, q); sycl::free(yn, q);
        }
    std::printf("%s\n", bad ? "MISMATCH" : "bit-identical to the replaced kernels");
    return bad ? 1 : 0;
}
