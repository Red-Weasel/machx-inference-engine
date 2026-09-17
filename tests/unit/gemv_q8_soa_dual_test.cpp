// Q8 single/dual batched projections must match independent T=1 leaves exactly.
// GPU1 (09:00.0) only. Also times GLM-5.3 decode shapes (SLM x vs 2× leaf).
#include "ie/ops.hpp"

#include <sycl/sycl.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static int check_dual(sycl::queue& q, uint32_t K, uint32_t N, uint32_t seed, uint32_t T = 1, unsigned offsets = 0) {
    const uint64_t count = uint64_t(T) * N;
    const uint32_t bpc = K / 32;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> i8(-128, 127);
    std::uniform_real_distribution<float> xf(-1.f, 1.f);

    std::vector<int8_t> qsA(uint64_t(N) * K), qsB(uint64_t(N) * K);
    std::vector<uint16_t> dA(uint64_t(N) * bpc), dB(uint64_t(N) * bpc);
    for (auto& v : qsA) v = int8_t(i8(rng));
    for (auto& v : qsB) v = int8_t(i8(rng));
    for (auto& v : dA) v = sycl::bit_cast<uint16_t>(sycl::half(0.001f * (1 + rng() % 32)));
    for (auto& v : dB) v = sycl::bit_cast<uint16_t>(sycl::half(0.001f * (1 + rng() % 32)));
    std::vector<sycl::half> x(uint64_t(T) * K);
    for (auto& v : x) v = sycl::half(xf(rng));

    auto* dqsA = sycl::malloc_device<int8_t>(qsA.size(), q);
    auto* dqsB = sycl::malloc_device<int8_t>(qsB.size(), q);
    auto* ddA  = sycl::malloc_device<uint16_t>(dA.size(), q);
    auto* ddB  = sycl::malloc_device<uint16_t>(dB.size(), q);
    auto* dx   = sycl::malloc_device<sycl::half>(x.size(), q);
    auto* yA1  = sycl::malloc_device<sycl::half>(count, q);
    auto* yB1  = sycl::malloc_device<sycl::half>(count, q);
    auto* yA2  = sycl::malloc_device<sycl::half>(count, q);
    auto* yB2  = sycl::malloc_device<sycl::half>(count, q);
    q.memcpy(dqsA, qsA.data(), qsA.size());
    q.memcpy(dqsB, qsB.data(), qsB.size());
    q.memcpy(ddA, dA.data(), dA.size() * 2);
    q.memcpy(ddB, dB.data(), dB.size() * 2);
    q.memcpy(dx, x.data(), x.size() * 2).wait();

    for (uint32_t row = 0; row < T; ++row) {
        ie::gemv_q8_0_soa_f16_g(q, dx + uint64_t(row) * K, dqsA, ddA,
                              yA1 + uint64_t(row) * N, K, N, {});
        ie::gemv_q8_0_soa_f16_g(q, dx + uint64_t(row) * K, dqsB, ddB,
                              yB1 + uint64_t(row) * N, K, N, {});
    }
    // Reference leaves retain aligned weights. Independently offset each
    // candidate input to exercise every alignment-dispatch condition.
    auto* shiftedA = offsets & 2 ? sycl::malloc_device<int8_t>(qsA.size() + 1, q) : nullptr;
    auto* shiftedB = offsets & 4 ? sycl::malloc_device<int8_t>(qsB.size() + 3, q) : nullptr;
    auto* shiftedX = offsets & 1 ? sycl::malloc_device<sycl::half>(x.size() + 1, q) : nullptr;
    if (shiftedA) q.memcpy(shiftedA + 1, qsA.data(), qsA.size());
    if (shiftedB) q.memcpy(shiftedB + 3, qsB.data(), qsB.size());
    if (shiftedX) q.memcpy(shiftedX + 1, x.data(), x.size() * 2);
    auto* candA = shiftedA ? shiftedA + 1 : dqsA;
    auto* candB = shiftedB ? shiftedB + 3 : dqsB;
    auto* candX = shiftedX ? shiftedX + 1 : dx;
    ie::gemv_q8_0_soa_f16_rows_dual(q, candX, candA, ddA, yA2, candB, ddB, yB2, K, N, T, {});
    q.wait();

    std::vector<sycl::half> hA1(count), hB1(count), hA2(count), hB2(count);
    q.memcpy(hA1.data(), yA1, count * 2);
    q.memcpy(hB1.data(), yB1, count * 2);
    q.memcpy(hA2.data(), yA2, count * 2);
    q.memcpy(hB2.data(), yB2, count * 2).wait();

    int mism = 0;
    for (uint64_t n = 0; n < count; ++n) {
        if (sycl::bit_cast<uint16_t>(hA1[n]) != sycl::bit_cast<uint16_t>(hA2[n]) ||
            sycl::bit_cast<uint16_t>(hB1[n]) != sycl::bit_cast<uint16_t>(hB2[n]))
            ++mism;
    }
    ie::gemv_q8_0_soa_f16_rows(q, candX, candA, ddA, yA2, K, N, T, {});
    q.memcpy(hA2.data(), yA2, count * 2).wait();
    for (uint64_t n = 0; n < count; ++n)
        if (sycl::bit_cast<uint16_t>(hA1[n]) != sycl::bit_cast<uint16_t>(hA2[n]))
            ++mism;
    std::printf("single/dual vs independent leaves K=%u N=%u T=%u offsets=%u: %d mismatches\n",
                K, N, T, offsets, mism);

    sycl::free(dqsA, q); sycl::free(dqsB, q);
    sycl::free(ddA, q); sycl::free(ddB, q);
    sycl::free(dx, q);
    if (shiftedA) sycl::free(shiftedA, q);
    if (shiftedB) sycl::free(shiftedB, q);
    if (shiftedX) sycl::free(shiftedX, q);
    sycl::free(yA1, q); sycl::free(yB1, q);
    sycl::free(yA2, q); sycl::free(yB2, q);
    return mism;
}

static void bench(sycl::queue& q, uint32_t K, uint32_t N) {
    const uint32_t bpc = K / 32;
    const uint64_t wbytes = uint64_t(N) * K + uint64_t(N) * bpc * 2;
    std::vector<int8_t> qs(uint64_t(N) * K, int8_t(3));
    std::vector<uint16_t> d(uint64_t(N) * bpc, sycl::bit_cast<uint16_t>(sycl::half(0.02f)));
    std::vector<sycl::half> x(K, sycl::half(0.1f));
    auto* dqsA = sycl::malloc_device<int8_t>(qs.size(), q);
    auto* dqsB = sycl::malloc_device<int8_t>(qs.size(), q);
    auto* ddA  = sycl::malloc_device<uint16_t>(d.size(), q);
    auto* ddB  = sycl::malloc_device<uint16_t>(d.size(), q);
    auto* dx   = sycl::malloc_device<sycl::half>(K, q);
    auto* yA   = sycl::malloc_device<sycl::half>(N, q);
    auto* yB   = sycl::malloc_device<sycl::half>(N, q);
    q.memcpy(dqsA, qs.data(), qs.size());
    q.memcpy(dqsB, qs.data(), qs.size());
    q.memcpy(ddA, d.data(), d.size() * 2);
    q.memcpy(ddB, d.data(), d.size() * 2);
    q.memcpy(dx, x.data(), K * 2).wait();

    auto time_ms = [&](auto fn) {
        for (int i = 0; i < 5; ++i) fn();
        q.wait();
        const auto t0 = std::chrono::steady_clock::now();
        constexpr int I = 20;
        for (int i = 0; i < I; ++i) fn();
        q.wait();
        const auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count() / I;
    };
    const double t2 = time_ms([&] {
        ie::gemv_q8_0_soa_f16_g(q, dx, dqsA, ddA, yA, K, N, {});
        ie::gemv_q8_0_soa_f16_g(q, dx, dqsB, ddB, yB, K, N, {});
    });
    const double td = time_ms([&] {
        ie::gemv_q8_0_soa_f16_rows_dual(q, dx, dqsA, ddA, yA, dqsB, ddB, yB, K, N, 1, {});
    });
    const double gbs2 = (2.0 * double(wbytes)) / (t2 * 1e6);
    const double gbsd = (2.0 * double(wbytes)) / (td * 1e6);
    std::printf("bench K=%u N=%u  two_g %.3f ms (%.1f GB/s)  dual %.3f ms (%.1f GB/s)  %.2fx\n",
                K, N, t2, gbs2, td, gbsd, t2 / td);

    sycl::free(dqsA, q); sycl::free(dqsB, q);
    sycl::free(ddA, q); sycl::free(ddB, q);
    sycl::free(dx, q); sycl::free(yA, q); sycl::free(yB, q);
}

// Readback waits ONLY for the returned event on an out-of-order queue.
// Integer-valued operands give an independent, exactly representable oracle.
static int check_completion(const sycl::device& dev,
                            uint32_t K = 4096, uint32_t N = 129, uint32_t T = 65) {
    sycl::queue q(dev);
    constexpr uint32_t guard = 16;
    auto* wa = sycl::malloc_device<int8_t>(uint64_t(K) * N + 1, q);
    auto* wb = sycl::malloc_device<int8_t>(uint64_t(K) * N + 1, q);
    auto* da = sycl::malloc_device<uint16_t>(uint64_t(K / 32) * N, q);
    auto* db = sycl::malloc_device<uint16_t>(uint64_t(K / 32) * N, q);
    auto* x = sycl::malloc_device<sycl::half>(uint64_t(T) * K + 1, q);
    const size_t count = uint64_t(T) * N;
    auto* a = sycl::malloc_device<sycl::half>(count + 2 * guard, q);
    auto* b = sycl::malloc_device<sycl::half>(count + 2 * guard, q);
    int bad = 0;
    for (unsigned offset : {0u, 1u}) for (bool dual : {false, true}) {
        std::vector<sycl::event> deps{
            q.fill(wa, int8_t(1), uint64_t(K) * N + 1),
            q.fill(wb, int8_t(-1), uint64_t(K) * N + 1),
            q.fill(da, sycl::bit_cast<uint16_t>(sycl::half(1)), uint64_t(K / 32) * N),
            q.fill(db, sycl::bit_cast<uint16_t>(sycl::half(1)), uint64_t(K / 32) * N),
            q.fill(x, sycl::half(1), uint64_t(T) * K + 1),
            q.fill(a, sycl::half(7), count + 2 * guard),
            q.fill(b, sycl::half(7), count + 2 * guard)};
        auto done = dual
            ? ie::gemv_q8_0_soa_f16_rows_dual(q, x + offset, wa + offset, da, a + guard,
                wb + offset, db, b + guard, K, N, T, deps)
            : ie::gemv_q8_0_soa_f16_rows(q, x + offset, wa + offset, da, a + guard, K, N, T, deps);
        for (unsigned which = 0; which < (dual ? 2u : 1u); ++which) {
            std::vector<sycl::half> got(count + 2 * guard);
            q.submit([&](sycl::handler& h) {
                h.depends_on(done); h.memcpy(got.data(), which ? b : a, got.size() * 2);
            }).wait_and_throw();
            for (size_t i = 0; i < got.size(); ++i) {
                float expected = (i < guard || i >= count + guard) ? 7.f : (which ? -float(K) : float(K));
                bad += float(got[i]) != expected;
            }
        }
        q.wait_and_throw(); // retire the unused b fill in the single case
    }
    sycl::free(wa, q); sycl::free(wb, q); sycl::free(da, q); sycl::free(db, q);
    sycl::free(x, q); sycl::free(a, q); sycl::free(b, q);
    std::printf("out-of-order K%u N%u T%u single/dual, aligned/unaligned: %d mismatches\n", K, N, T, bad);
    return bad;
}

int main() {
    sycl::device dev;
    bool found = false;
    for (auto& d : sycl::device::get_devices(sycl::info::device_type::gpu)) {
        auto bdf = d.get_info<sycl::ext::intel::info::device::pci_address>();
        if (d.get_backend() == sycl::backend::ext_oneapi_level_zero &&
            bdf.find("09:00.0") != std::string::npos) { dev = d; found = true; break; }
    }
    if (!found) {
        std::fputs("GPU1 09:00.0 not found\n", stderr);
        return 2;
    }
    sycl::queue q(dev, sycl::property::queue::in_order{});
    std::printf("Level Zero device: %s %s\n",
                dev.get_info<sycl::info::device::name>().c_str(),
                dev.get_info<sycl::ext::intel::info::device::pci_address>().c_str());

    int mism = 0;
    mism += check_completion(dev);
    for (uint32_t T : {2u, 3u, 4u, 16u, 128u})
        mism += check_completion(dev, 4096, 129, T);
    mism += check_completion(dev, 8192, 4096, 128);
    mism += check_completion(dev, 16384, 257, 128);
    mism += check_completion(dev, 4096, 2048, 512);
    mism += check_completion(dev, 4096, 129, 129);
    mism += check_completion(dev, 8192, 4096, 129);
    mism += check_completion(dev, 16384, 257, 129);
    mism += check_completion(dev, 4096, 2048, 513);
    mism += check_dual(q, 256, 64, 7);
    mism += check_dual(q, 4096, 64, 11);
    mism += check_dual(q, 4096, 2048, 13);   // shexp gate/up
    mism += check_dual(q, 128, 8192, 17);    // f_b / g_b K=128
    // Decode dispatch: narrow outputs, ragged fallback, and contraction cap.
    mism += check_dual(q, 4096, 256, 601);
    mism += check_dual(q, 4096, 257, 602);
    mism += check_dual(q, 512, 2049, 603);
    mism += check_dual(q, 12288, 257, 604);
    mism += check_dual(q, 12320, 257, 605);
    // Dispatch boundaries, multi-tile tails, and partial output subgroups.
    for (uint32_t T : {2u, 3u, 4u, 5u, 7u, 8u, 9u, 15u, 16u, 17u, 31u, 32u, 65u}) {
        mism += check_dual(q, 608, 19, 101 + T, T);
        mism += check_dual(q, 4096, 128, 211 + T, T);
    }
    for (unsigned offsets = 1; offsets < 8; ++offsets) {
        mism += check_dual(q, 608, 19, 400 + offsets, 17, offsets);
        mism += check_dual(q, 608, 19, 500 + offsets, 1, offsets);
    }
    // Combined-grid threshold, partial final tiles, and each grid ordering.
    for (uint32_t T : {127u, 128u, 129u, 257u, 1024u})
        mism += check_dual(q, 4096, 64, 700 + T, T);
    mism += check_dual(q, 8192, 4096, 801, 129);
    mism += check_dual(q, 12288, 4096, 802, 129);
    mism += check_dual(q, 16384, 257, 803, 129);
    mism += check_dual(q, 4096, 2048, 804, 513);
    mism += check_dual(q, 608, 19, 805, 129, 7);
    if (mism != 0) {
        std::puts("FAIL");
        return 1;
    }
    bench(q, 4096, 2048);
    bench(q, 4096, 4096);
    bench(q, 4096, 8192);
    bench(q, 2048, 4096);
    bench(q, 4096, 128);
    // Contraction-dispatch boundary. Same N, GB/s is the fair comparison
    // (weights-only accounting in bench()).
    bench(q, 12288, 4096);
    bench(q, 16384, 4096);
    std::puts("GATE PASSED");
    return 0;
}
