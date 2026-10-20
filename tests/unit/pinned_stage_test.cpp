// tests/unit/pinned_stage_test.cpp — DeviceAllocator::h2d / d2h (pageable host <-> device through the allocator's
// pinned stage, kStageBytes at a time) and DeviceFleet::copy_across's host bounce through the source card's stage
// (2026-10-20, GitHub issue #2). Byte-exact round trips at sizes around the stage size and past the 512 MiB
// single-copy threshold of intel/compute-runtime#948, two threads on one stage, malloc_host / free_host, and an
// allocator destroyed with its stage live. Needs one B70; the cross-card leg runs when two are visible.
#include "ie/allocator.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

namespace {
int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { std::printf("FAIL: " __VA_ARGS__); std::printf("\n"); ++fails; } } while (0)

void fill(std::vector<uint8_t>& v, uint32_t seed) {
    uint32_t x = seed * 2654435761u + 1;
    for (auto& b : v) { x ^= x << 13; x ^= x >> 17; x ^= x << 5; b = uint8_t(x); }
}

void round_trip(ie::DeviceAllocator& a, size_t n, uint32_t seed) {
    std::vector<uint8_t> src(n), back(n, 0);
    fill(src, seed);
    void* d = a.malloc(n);
    CHECK(d, "malloc %zu", n);
    if (!d) return;
    a.h2d(d, src.data(), n);
    a.d2h(back.data(), d, n);
    a.free(d);
    CHECK(std::memcmp(src.data(), back.data(), n) == 0, "round trip %zu bytes differs", n);
}
}  // namespace

int main() {
    ie::DeviceFleet fleet;
    if (auto e = fleet.init(2, "B70", /*p2p=*/false, /*shared_ctx=*/false); !e.empty()) {
        std::printf("SKIP: %s\n", e.c_str());
        return 77;
    }
    ie::DeviceAllocator& a = fleet.dev(0);
    constexpr size_t S = ie::DeviceAllocator::kStageBytes;
    for (size_t n : {size_t(1), size_t(4096), S - 1, S, S + 7, 3 * S + 12345, size_t(600) << 20})
        round_trip(a, n, uint32_t(n));
    std::printf("h2d / d2h round trips: 1 B .. 600 MiB\n");

    {   // the staged d2h kernel (<= kKernelD2hBytes): 4-aligned words, an unaligned source, odd sizes, the limit
        const size_t n = ie::DeviceAllocator::kKernelD2hBytes + 64;
        std::vector<uint8_t> src(n), back(n);
        fill(src, 5);
        auto* d = static_cast<uint8_t*>(a.malloc(n));
        CHECK(d, "malloc for the d2h kernel");
        if (d) {
            a.h2d(d, src.data(), n);
            for (auto [off, len] : {std::pair<size_t, size_t>{0, 4}, {0, 1000}, {1, 1001}, {3, 7}, {2, 4096},
                                    {0, ie::DeviceAllocator::kKernelD2hBytes}, {1, ie::DeviceAllocator::kKernelD2hBytes}}) {
                std::fill(back.begin(), back.end(), 0);
                a.d2h(back.data(), d + off, len);
                CHECK(std::memcmp(back.data(), src.data() + off, len) == 0, "d2h kernel off %zu len %zu", off, len);
            }
        }
        a.free(d);
        std::printf("d2h kernel: aligned, unaligned, odd sizes, the limit\n");
    }

    {   // two threads through one stage (its mutex), each its own device buffer
        const size_t n = 2 * S + 99;
        std::vector<uint8_t> s0(n), s1(n), b0(n), b1(n);
        fill(s0, 7); fill(s1, 8);
        void* d0 = a.malloc(n);
        void* d1 = a.malloc(n);
        CHECK(d0 && d1, "malloc for threads");
        if (d0 && d1) {
            std::thread t0([&] { for (int i = 0; i < 4; ++i) { a.h2d(d0, s0.data(), n); a.d2h(b0.data(), d0, n); } });
            std::thread t1([&] { for (int i = 0; i < 4; ++i) { a.h2d(d1, s1.data(), n); a.d2h(b1.data(), d1, n); } });
            t0.join(); t1.join();
            CHECK(std::memcmp(s0.data(), b0.data(), n) == 0 && std::memcmp(s1.data(), b1.data(), n) == 0,
                  "two threads on one stage");
        }
        a.free(d0); a.free(d1);
        std::printf("two threads on one stage\n");
    }

    {   // malloc_host: pinned in this context, usable as an async copy source; free_host gives it back
        const size_t n = 1 << 20;
        auto* h = static_cast<uint8_t*>(a.malloc_host(n));
        CHECK(h, "malloc_host");
        void* d = a.malloc(n);
        if (h && d) {
            for (size_t i = 0; i < n; ++i) h[i] = uint8_t(i * 31);
            a.queue().memcpy(d, h, n);   // async from pinned
            std::vector<uint8_t> back(n);
            a.d2h(back.data(), d, n);    // in-order queue: after the copy above
            bool ok = true;
            for (size_t i = 0; i < n; ++i) ok &= back[i] == uint8_t(i * 31);
            CHECK(ok, "async copy from malloc_host");
        }
        a.free(d);
        a.free_host(h);
        a.free_host(h);   // second free is a no-op (not tracked any more)
        std::printf("malloc_host / free_host\n");
    }

    {   // h2d_async: 3 x kRingSlots uploads with the source overwritten right after each call (the copy reads the
        // pinned slot, not the source), one over kRingSlotBytes (falls back to h2d), one from USM (direct)
        constexpr size_t R = ie::DeviceAllocator::kRingSlotBytes;
        constexpr uint32_t N = 3 * ie::DeviceAllocator::kRingSlots;
        const size_t n = 4096 + 12;
        auto* d = static_cast<uint8_t*>(a.malloc(N * n + 2 * R));
        CHECK(d, "malloc for h2d_async");
        if (d) {
            std::vector<uint8_t> src(2 * R);
            for (uint32_t i = 0; i < N; ++i) {
                std::fill(src.begin(), src.begin() + n, uint8_t(i + 1));
                a.h2d_async(d + size_t(i) * n, src.data(), n);
                std::fill(src.begin(), src.begin() + n, uint8_t(0xEE));   // must not reach the device
            }
            fill(src, 99);
            a.h2d_async(d + size_t(N) * n, src.data(), R + 1);            // over a slot: blocking h2d
            std::vector<uint8_t> big(src.begin(), src.begin() + R + 1);
            auto* u = static_cast<uint8_t*>(a.malloc_host(64));
            for (int i = 0; i < 64; ++i) u[i] = uint8_t(200 + i);
            a.h2d_async(d + size_t(N) * n + R + 1, u, 64).wait();        // USM: enqueued directly
            std::vector<uint8_t> back(N * n + R + 1 + 64);
            a.d2h(back.data(), d, back.size());
            bool ok = true;
            for (uint32_t i = 0; i < N; ++i)
                for (size_t k = 0; k < n; ++k) ok &= back[size_t(i) * n + k] == uint8_t(i + 1);
            CHECK(ok, "h2d_async ring uploads");
            CHECK(std::memcmp(back.data() + size_t(N) * n, big.data(), R + 1) == 0, "h2d_async over a slot");
            CHECK(std::memcmp(back.data() + size_t(N) * n + R + 1, u, 64) == 0, "h2d_async from USM");
            a.free_host(u);
        }
        a.free(d);
        std::printf("h2d_async: %u ring uploads, one over a slot, one from USM\n", N);
    }

    {   // h2d / d2h of USM host memory (another allocator's context) take the direct path, in kStageBytes pieces
        ie::DeviceAllocator other;
        if (other.init("B70", 0).empty()) {
            const size_t n = 2 * S + 5;
            auto* h = static_cast<uint8_t*>(other.malloc_host(n));
            void* d = a.malloc(n);
            CHECK(h && d, "alloc for the USM path");
            if (h && d) {
                for (size_t i = 0; i < n; ++i) h[i] = uint8_t(i * 7 + 3);
                a.h2d(d, h, n);
                std::memset(h, 0, n);
                a.d2h(h, d, n);
                bool ok = true;
                for (size_t i = 0; i < n; ++i) ok &= h[i] == uint8_t(i * 7 + 3);
                CHECK(ok, "h2d / d2h of another context's host USM");
            }
            a.free(d);
        }
        std::printf("h2d / d2h of USM host memory\n");
    }

    if (fleet.size() >= 2) {
        for (size_t n : {size_t(10240), S + 3, size_t(130) << 20}) {
            std::vector<uint8_t> src(n), back(n, 0);
            fill(src, uint32_t(n) + 1);
            void* d0 = fleet.dev(0).malloc(n);
            void* d1 = fleet.dev(1).malloc(n);
            CHECK(d0 && d1, "malloc for copy_across %zu", n);
            if (d0 && d1) {
                fleet.dev(0).h2d(d0, src.data(), n);
                fleet.copy_across(0, d1, 1, d0, n);
                fleet.dev(1).d2h(back.data(), d1, n);
                CHECK(std::memcmp(src.data(), back.data(), n) == 0, "copy_across 0->1 %zu bytes", n);
                std::fill(back.begin(), back.end(), 0);
                fleet.copy_across(1, d0, 0, d1, n);   // and back
                fleet.dev(0).d2h(back.data(), d0, n);
                CHECK(std::memcmp(src.data(), back.data(), n) == 0, "copy_across 1->0 %zu bytes", n);
            }
            fleet.dev(0).free(d0);
            fleet.dev(1).free(d1);
        }
        std::printf("copy_across both ways: 10 KiB .. 130 MiB\n");
    } else {
        std::printf("copy_across: one card visible, skipped\n");
    }

    {   // an allocator destroyed with its stage and a malloc_host block live frees them on its own queue
        ie::DeviceAllocator b;
        if (b.init("B70", 0).empty()) {
            std::vector<uint8_t> v(4096, 3), w(4096);
            void* d = b.malloc(4096);
            b.h2d(d, v.data(), v.size());
            b.d2h(w.data(), d, w.size());
            (void)b.malloc_host(1 << 16);
            b.free(d);
            CHECK(w == v, "scoped allocator round trip");
        }
    }
    std::printf("allocator destroyed with a live stage\n");

    std::printf("%s\n", fails ? "pinned_stage_test: FAILED" : "pinned_stage_test: all OK");
    return fails ? 1 : 0;
}
