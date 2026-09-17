// bench/ds41_fp8_gemv_bench.cpp -- Phase 38 (docs/deepseek41/78): the FP8 GEMV's decode variants at the V4.1
// decode shapes against a raw-read floor of the same bytes, plus a bit-identity check of every variant against
// the scalar decode on random bytes (all 256 codes, the NaN codes included).
//   usage: ds41-fp8-gemv-bench [iters=100]
#include "ie/deepseek41_upload.hpp"
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    const int iters = argc > 1 ? std::atoi(argv[1]) : 100;
    sycl::device dev;
    for (auto& d : sycl::device::get_devices(sycl::info::device_type::gpu))
        if (d.get_info<sycl::info::device::name>().find("Arc") != std::string::npos) { dev = d; break; }
    sycl::queue q(dev, sycl::property::queue::in_order{});
    struct Shape { const char* name; uint32_t K, N; };
    const Shape shapes[] = { {"q_b  ", 1280, 32768}, {"o_b  ", 8192, 5120}, {"o_a  ", 4096, 8192}, {"sh_gate", 5120, 2304}, {"sh_down", 2304, 5120}, {"q_a  ", 5120, 1280} };
    std::mt19937 rng(11);
    std::printf("device %s, %d iters per point; GB/s of WEIGHT bytes (E4M3), floor = a plain read of the same bytes\n", dev.get_info<sycl::info::device::name>().c_str(), iters);
    std::printf("%-8s %7s | %8s | %8s %8s %8s %8s %8s | bit-identical to scalar?\n", "shape", "MB", "floor", "dec0", "dec1", "dec2", "dec3", "dec4");
    for (const auto& sh : shapes) {
        const uint32_t K = sh.K, N = sh.N; const size_t nw = size_t(N) * K, ns = size_t(N / 32) * (K / 32);
        std::vector<uint8_t> hw(nw), hs(ns); std::vector<sycl::half> hx(K);
        for (auto& v : hw) v = uint8_t(rng());                    // every code, NaN codes included
        for (auto& v : hs) v = uint8_t(120 + rng() % 16);         // scales 2^-7 .. 2^8
        for (auto& v : hx) v = sycl::half(float(int(rng() % 2001) - 1000) / 1000.f);
        // R copies of the matrix, rotated per launch, so >= 256 MB streams between two visits of the same bytes: the
        // real token reads each matrix once among ~2 GB of others, and a 12 MB matrix re-read 100 times sits in L2.
        const uint32_t R = uint32_t(std::max<size_t>(1, (size_t(256) << 20) / nw));
        uint8_t* dw = sycl::malloc_device<uint8_t>(nw * R, q); uint8_t* ds = sycl::malloc_device<uint8_t>(ns, q);
        sycl::half* dx = sycl::malloc_device<sycl::half>(K, q); float* dy = sycl::malloc_device<float>(size_t(N) * 7, q);
        for (uint32_t r = 0; r < R; ++r) q.memcpy(dw + size_t(r) * nw, hw.data(), nw);
        q.memcpy(ds, hs.data(), ns); q.memcpy(dx, hx.data(), K * 2).wait();
        uint32_t rot = 0;
        auto wsel = [&]() { const uint8_t* p = dw + size_t(rot % R) * nw; ++rot; return p; };
        auto gbs = [&](auto&& fn) { for (int i = 0; i < 5; ++i) fn(); q.wait(); const auto t0 = std::chrono::steady_clock::now(); for (int i = 0; i < iters; ++i) fn(); q.wait();
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / iters; return double(nw) / s / 1e9; };
        // the floor: 16-lane subgroups, each lane 32 contiguous bytes per iteration, the kernel's own access pattern, summed and stored
        float* dsum = sycl::malloc_device<float>(size_t(N), q);
        const double floor = gbs([&] { const uint8_t* wr = wsel(); q.parallel_for(sycl::nd_range<1>(size_t(N) * 16, 256), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const uint32_t n = uint32_t(it.get_global_id(0)) / 16, lane = uint32_t(it.get_local_id(0)) % 16; if (n >= N) return;
            const auto* w128 = reinterpret_cast<const sycl::vec<uint32_t, 4>*>(wr + uint64_t(n) * K); uint32_t acc = 0;
            for (uint32_t b = lane; b < K / 32; b += 16) { const auto a = w128[2 * b], c = w128[2 * b + 1]; acc += a[0] ^ a[1] ^ a[2] ^ a[3] ^ c[0] ^ c[1] ^ c[2] ^ c[3]; }
            acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<uint32_t>()); if (lane == 0) dsum[n] = float(acc); }); });
        double r[5]; std::vector<float> ref(N), got(N); std::string ident;
        for (int dec = 0; dec <= 4; ++dec) {
            float* y = dy + size_t(dec) * N;
            r[dec] = gbs([&] { ie::gemv_fp8_e4m3_f16_variant(q, dx, wsel(), ds, y, K, N, dec); });
            q.memcpy(dec == 0 ? ref.data() : got.data(), y, size_t(N) * 4).wait();
            if (dec > 0) { size_t nd = 0; for (uint32_t i = 0; i < N; ++i) if (sycl::bit_cast<uint32_t>(ref[i]) != sycl::bit_cast<uint32_t>(got[i])) ++nd; ident += (nd ? " dec" + std::to_string(dec) + ":" + std::to_string(nd) + "diff" : ""); }
        }
        std::printf("%-8s %7.1f | %8.0f | %8.0f %8.0f %8.0f %8.0f %8.0f | %s\n", sh.name, nw / 1e6, floor, r[0], r[1], r[2], r[3], r[4], ident.empty() ? "all identical" : ident.c_str());
        sycl::free(dw, q); sycl::free(ds, q); sycl::free(dx, q); sycl::free(dy, q); sycl::free(dsum, q);
    }
    return 0;
}
