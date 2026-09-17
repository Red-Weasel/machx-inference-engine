// Batched Q4/Q5 must equal independent rows and honor dependencies on an
// out-of-order queue. Canaries catch writes outside partial output tiles.
#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"
#include <cstdio>
#include <cstring>
#include <random>

using RowsFn = sycl::event (*)(sycl::queue&, const sycl::half*, const void*,
    sycl::half*, uint32_t, uint32_t, uint32_t, const std::vector<sycl::event>&);

template<class Block>
static bool check(sycl::queue& q, RowsFn fn, const char* name, uint32_t K, uint32_t N) {
    constexpr uint32_t max_rows = 65, guard = 16;
    std::mt19937 rng(53 + K + N);
    std::uniform_real_distribution<float> value(-1, 1);
    std::vector<Block> weights(uint64_t(K / 256) * N);
    for (auto& b : weights) {
        auto* bytes = reinterpret_cast<unsigned char*>(&b);
        for (size_t i = 0; i < sizeof(b); ++i) bytes[i] = uint8_t(rng());
        b.d = sycl::bit_cast<uint16_t>(sycl::half(.001f * (1 + rng() % 8)));
        b.dmin = sycl::bit_cast<uint16_t>(sycl::half(.001f * (1 + rng() % 8)));
    }
    std::vector<sycl::half> input(uint64_t(max_rows) * K);
    for (auto& v : input) v = sycl::half(value(rng));
    auto* w = sycl::malloc_device<Block>(weights.size(), q);
    auto* x = sycl::malloc_device<sycl::half>(input.size(), q);
    auto* ref = sycl::malloc_device<sycl::half>(uint64_t(max_rows) * N, q);
    auto* dst = sycl::malloc_device<sycl::half>(uint64_t(max_rows) * N + 2 * guard, q);
    auto ew = q.memcpy(w, weights.data(), weights.size() * sizeof(Block));
    auto ex = q.memcpy(x, input.data(), input.size() * 2);
    bool ok = true;
    for (uint32_t R : {1u, 2u, 3u, 4u, 5u, 7u, 8u, 9u, 15u, 16u, 17u, 31u, 32u, 33u, 63u, 64u, 65u}) {
        const size_t count = uint64_t(R) * N;
        std::vector<sycl::half> expected(count), actual(count + 2 * guard);
        auto clear = q.fill(dst, sycl::half(-123.f), actual.size());
        auto done = fn(q, x, w, dst + guard, K, N, R, {ew, ex, clear});
        // The returned event must cover every row tile, without q.wait().
        q.submit([&](sycl::handler& h) {
            h.depends_on(done);
            h.memcpy(actual.data(), dst, actual.size() * 2);
        }).wait_and_throw();
        std::vector<sycl::event> rows;
        for (uint32_t r = 0; r < R; ++r)
            rows.push_back(fn(q, x + uint64_t(r) * K, w, ref + uint64_t(r) * N, K, N, 1, {ew, ex}));
        q.submit([&](sycl::handler& h) {
            h.depends_on(rows);
            h.memcpy(expected.data(), ref, count * 2);
        }).wait_and_throw();
        if (std::memcmp(expected.data(), actual.data() + guard, count * 2)) {
            std::printf("FAIL %s K%u N%u R%u: row mismatch\n", name, K, N, R);
            ok = false;
        }
        for (size_t i = 0; i < guard; ++i)
            if (float(actual[i]) != -123.f || float(actual[guard + count + i]) != -123.f) {
                std::printf("FAIL %s K%u N%u R%u: output canary\n", name, K, N, R);
                ok = false;
                break;
            }
    }
    sycl::free(w, q); sycl::free(x, q); sycl::free(ref, q); sycl::free(dst, q);
    std::printf("%s %s K%u N%u: 17 row counts, dependency and canary gates\n", ok ? "PASS" : "FAIL", name, K, N);
    return ok;
}
int main() {
    std::vector<sycl::device> devices;
    for (auto d : sycl::device::get_devices(sycl::info::device_type::gpu))
        if (d.get_info<sycl::info::device::name>().find("B70") != std::string::npos) devices.push_back(d);
    if (devices.empty()) { std::fputs("No B70 available\n", stderr); return 2; }
    sycl::queue q(devices.back());  // deliberately out of order
    bool ok = true;
    for (auto [K,N] : {std::pair{256u, 1u}, std::pair{768u, 19u}, std::pair{4096u, 128u}}) {
        ok &= check<ie::block_q4_K>(q, ie::gemv_q4_K_rows, "Q4_K", K, N);
        ok &= check<ie::block_q5_K>(q, ie::gemv_q5_K_rows, "Q5_K", K, N);
    }
    return ok ? 0 : 1;
}
