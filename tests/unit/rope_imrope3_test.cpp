// tests/unit/rope_imrope3_test.cpp — P2 gate piece (docs/qwen4/16_vision_port.md §4).
// 1) equal T/H/W streams: rope_imrope3 must be BIT-identical to rope_partial
//    (text degeneration, docs/qwen4/14_mrope.md §3).
// 2) divergent streams: matches a host fp32 reference with owner(r) = r % 3.

#include "ie/ops.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

int main() {
    const uint32_t T = 7, HEADS = 4, HD = 256, ROT = 64;
    const float THETA = 1e7f;
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order{});

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-2.f, 2.f);
    const size_t n = size_t(T) * HEADS * HD;
    std::vector<sycl::half> hx(n);
    for (auto& v : hx) v = sycl::half(dist(rng));

    sycl::half* x  = sycl::malloc_device<sycl::half>(n, q);
    sycl::half* y1 = sycl::malloc_device<sycl::half>(n, q);
    sycl::half* y2 = sycl::malloc_device<sycl::half>(n, q);
    int32_t* pos1  = sycl::malloc_device<int32_t>(T, q);
    int32_t* pos3  = sycl::malloc_device<int32_t>(size_t(3) * T, q);
    q.memcpy(x, hx.data(), n * sizeof(sycl::half)).wait();

    int fails = 0;

    {   // ---- equal streams: bit-identity vs rope_partial ----
        std::vector<int32_t> p(T);
        for (uint32_t i = 0; i < T; ++i) p[i] = int32_t(100 + 7 * i);
        std::vector<int32_t> p3(size_t(3) * T);
        for (int s = 0; s < 3; ++s) std::memcpy(&p3[size_t(s) * T], p.data(), T * 4);
        q.memcpy(pos1, p.data(), T * 4).wait();
        q.memcpy(pos3, p3.data(), 3 * T * 4).wait();
        ie::rope_partial(q, x, pos1, y1, T, HEADS, HD, ROT, THETA).wait();
        ie::rope_imrope3(q, x, pos3, y2, T, HEADS, HD, ROT, THETA).wait();
        std::vector<sycl::half> a(n), b(n);
        q.memcpy(a.data(), y1, n * 2).wait();
        q.memcpy(b.data(), y2, n * 2).wait();
        int bad = 0;
        for (size_t i = 0; i < n; ++i)
            if (std::memcmp(&a[i], &b[i], 2) != 0) ++bad;
        std::printf("equal-streams bit-identity: %d/%zu mismatches\n", bad, n);
        if (bad) ++fails;
    }

    {   // ---- divergent streams vs host fp32 reference ----
        std::vector<int32_t> p3(size_t(3) * T);
        for (uint32_t i = 0; i < T; ++i) {
            p3[i] = int32_t(50);                 // T stream constant (image block)
            p3[T + i] = int32_t(50 + i / 3);     // H
            p3[2 * T + i] = int32_t(50 + i % 3); // W
        }
        q.memcpy(pos3, p3.data(), 3 * T * 4).wait();
        ie::rope_imrope3(q, x, pos3, y2, T, HEADS, HD, ROT, THETA).wait();
        std::vector<sycl::half> got(n);
        q.memcpy(got.data(), y2, n * 2).wait();
        double max_d = 0;
        for (uint32_t t = 0; t < T; ++t)
            for (uint32_t hd2 = 0; hd2 < HEADS; ++hd2)
                for (uint32_t r = 0; r < ROT / 2; ++r) {
                    const size_t base = (size_t(t) * HEADS + hd2) * HD;
                    const float inv = std::exp(-2.f * float(r) / float(ROT) * std::log(THETA));
                    const float ang = float(p3[size_t(r % 3) * T + t]) * inv;
                    const float a = float(hx[base + r]), b = float(hx[base + r + ROT / 2]);
                    const float e0 = a * std::cos(ang) - b * std::sin(ang);
                    const float e1 = a * std::sin(ang) + b * std::cos(ang);
                    max_d = std::max(max_d, std::fabs(double(float(got[base + r])) - e0));
                    max_d = std::max(max_d, std::fabs(double(float(got[base + r + ROT / 2])) - e1));
                }
        // pass-through dims must be copied
        int pt_bad = 0;
        for (uint32_t t = 0; t < T; ++t)
            for (uint32_t hd2 = 0; hd2 < HEADS; ++hd2)
                for (uint32_t d = ROT; d < HD; ++d) {
                    const size_t i = (size_t(t) * HEADS + hd2) * HD + d;
                    if (std::memcmp(&got[i], &hx[i], 2) != 0) ++pt_bad;
                }
        std::printf("divergent-streams: max|d|=%.3e vs host ref, passthrough bad=%d\n",
                    max_d, pt_bad);
        if (max_d > 2e-3 || pt_bad) ++fails;   // fp16 storage + native transcendentals
    }

    sycl::free(x, q); sycl::free(y1, q); sycl::free(y2, q);
    sycl::free(pos1, q); sycl::free(pos3, q);
    std::printf(fails ? "GATE FAILED\n" : "GATE PASSED\n");
    return fails ? 1 : 0;
}
