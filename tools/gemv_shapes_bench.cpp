// tools/gemv_shapes_bench.cpp — achieved-bandwidth probe for the W8A8 SoA
// GEMV at the qwen4exp DECODE shapes (roofline step 1: is the kernel or the
// shape the bottleneck?). Synthetic weights; weight-stream GB/s per shape.
//
// usage: ie-gemv-shapes-bench [gpu]

#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"
#include "ie/dp4a.hpp"

#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace ie;


// ---- prototype variants (bench-local): N_PER_WG + split-K sweep ------------
// Split-K reassociates the fp32 accumulation (PPL-gate before production).
template <int NPW, int SK>
static sycl::event gemv_q8_proto(sycl::queue& q, const void* x_q8,
                                 const int8_t* qs_W, const uint16_t* d_W,
                                 sycl::half* y, uint32_t K, uint32_t N) {
    constexpr int SG_SIZE = 16;
    constexpr int WG_ITEMS = NPW * SG_SIZE * SK;
    const auto* X8 = static_cast<const block_q8_1x*>(x_q8);
    const uint32_t blocks_per_col = K / 32;
    const uint32_t n_wgs = (N + NPW - 1) / NPW;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> part(SK > 1 ? NPW * SK : 1, h);
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid  = uint32_t(it.get_local_id(0));
            const uint32_t sg   = lid / SG_SIZE;          // 0..NPW*SK-1
            const uint32_t lane = lid % SG_SIZE;
            const uint32_t col  = sg / SK;                // column within WG
            const uint32_t kseg = sg % SK;                // K segment
            const uint32_t n    = uint32_t(it.get_group(0)) * NPW + col;
            if (n >= N) return;
            const int8_t*    wcol = qs_W + uint64_t(n) * K;
            const uint16_t* dcol = d_W +
                                    uint64_t(n) * blocks_per_col;
            float acc = 0.f;
            for (uint32_t b = kseg * SG_SIZE + lane; b < blocks_per_col;
                 b += SG_SIZE * SK) {
                const uint32_t* wq = reinterpret_cast<const uint32_t*>(wcol + uint64_t(b) * 32);
                const uint32_t* xq = reinterpret_cast<const uint32_t*>(X8[b].qs);
                int32_t idot = 0;
                #pragma unroll
                for (int w = 0; w < 8; ++w)
                    idot = ie::dp4a_ss(int32_t(wq[w]), int32_t(xq[w]), idot);
                acc += float(sycl::bit_cast<sycl::half>(dcol[b])) * float(X8[b].d) * float(idot);
            }
            acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
            if constexpr (SK == 1) {
                if (lane == 0) y[n] = sycl::half(acc);
            } else {
                if (lane == 0) part[col * SK + kseg] = acc;
                it.barrier(sycl::access::fence_space::local_space);
                if (kseg == 0 && lane == 0) {
                    float s2 = 0.f;
                    for (int i = 0; i < SK; ++i) s2 += part[col * SK + i];
                    y[n] = sycl::half(s2);
                }
            }
        });
    });
}

template <int NPW, int SK>
static void bench_proto(sycl::queue& q, const void* x, const int8_t* qs,
                        const uint16_t* d, sycl::half* y, uint32_t K, uint32_t N,
                        uint64_t bytes) {
    for (int i = 0; i < 5; ++i) gemv_q8_proto<NPW, SK>(q, x, qs, d, y, K, N);
    q.wait();
    const int iters = 50;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) gemv_q8_proto<NPW, SK>(q, x, qs, d, y, K, N);
    q.wait();
    const double us = std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - t0).count() / iters;
    std::printf("    proto NPW=%-2d SK=%d  %8.1f us  %6.1f GB/s\n", NPW, SK, us,
                double(bytes) / us / 1e3);
}

int main(int argc, char** argv) {
    const uint32_t ordinal = argc > 1 ? uint32_t(std::atoi(argv[1])) : 0;
    sycl::queue q;
    {
        auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
        uint32_t n = 0;
        for (auto& d : devs) {
            if (d.get_info<sycl::info::device::name>().find("B70") == std::string::npos) continue;
            if (n++ == ordinal) { q = sycl::queue(d, sycl::property::queue::in_order{}); break; }
        }
    }
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    struct Shape { uint32_t K, N; const char* what; };
    const Shape shapes[] = {
        {2560, 10240, "dn qkv        (x36/tok)"},
        {2560, 6144,  "dn gate       (x36/tok)"},
        {6144, 2560,  "dn out        (x36/tok)"},
        {2560, 6144,  "attn qkv-ish  (x12/tok)"},
        {2560, 2560,  "small square"},
        {2560, 248320, "lm_head       (x1/tok)"},
        {12288, 4096, "reference big-K (533+ GB/s proven)"},
    };
    for (const Shape& sh : shapes) {
        const uint64_t wbytes = uint64_t(sh.N) * sh.K;             // int8 qs
        const uint64_t dbytes = uint64_t(sh.N) * (sh.K / 32) * 2;  // f16 scales
        auto* qs = sycl::malloc_device<int8_t>(wbytes, q);
        auto* d  = sycl::malloc_device<uint16_t>(dbytes / 2, q);
        auto* x  = sycl::malloc_device<uint8_t>((sh.K / 32) * sizeof(block_q8_1x), q);
        auto* y  = sycl::malloc_device<sycl::half>(sh.N, q);
        q.memset(qs, 1, wbytes).wait();
        q.memset(d, 0x3c, dbytes).wait();   // ~1.0 halves
        q.memset(x, 1, (sh.K / 32) * sizeof(block_q8_1x)).wait();
        // warmup
        for (int i = 0; i < 5; ++i) gemv_q8_0_soa_q8_g(q, x, qs, d, y, sh.K, sh.N);
        q.wait();
        const int iters = 50;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) gemv_q8_0_soa_q8_g(q, x, qs, d, y, sh.K, sh.N);
        q.wait();
        const double us = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - t0).count() / iters;
        const double gbs = double(wbytes + dbytes) / us / 1e3;
        std::printf("  K=%6u N=%6u  %-36s %8.1f us  %6.1f GB/s\n",
                    sh.K, sh.N, sh.what, us, gbs);
        if (sh.K <= 6144) {
            bench_proto<32, 2>(q, x, qs, d, y, sh.K, sh.N, wbytes + dbytes);
            bench_proto<16, 1>(q, x, qs, d, y, sh.K, sh.N, wbytes + dbytes);
            bench_proto<16, 2>(q, x, qs, d, y, sh.K, sh.N, wbytes + dbytes);
            bench_proto<8, 2>(q, x, qs, d, y, sh.K, sh.N, wbytes + dbytes);
            bench_proto<8, 4>(q, x, qs, d, y, sh.K, sh.N, wbytes + dbytes);
        }
        sycl::free(qs, q); sycl::free(d, q); sycl::free(x, q); sycl::free(y, q);
    }
    return 0;
}
