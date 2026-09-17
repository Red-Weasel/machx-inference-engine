#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"
#include <sycl/sycl.hpp>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {
// Construct packed weights and the oracle together from logical quant values.
// Dyadic inputs/scales keep the nonsaturating sums exactly representable.
void check(sycl::queue& q, uint32_t K, uint32_t N, bool unaligned, bool saturate) {
    const size_t blocks = size_t(N) * (K / 256), stride = N + 32;
    std::vector<sycl::half> input(K), expected(stride, sycl::half(-123));
    std::vector<ie::block_q5_K> weights(blocks);
    for (uint32_t k = 0; k < K; ++k)
        input[k] = sycl::half(saturate ? 1.f : float(int((k * 13 + 7) % 17) - 8) / 8.f);
    for (uint32_t n = 0; n < N; ++n) {
        double sum = 0;
        for (uint32_t b = 0; b < K / 256; ++b) {
            auto& w = weights[size_t(n) * (K / 256) + b];
            std::array<uint8_t, 8> scales{}, mins{};
            const float d = saturate ? 16.f : 1.f / 16;
            const float dm = saturate ? 16.f : 1.f / 32;
            w.d = sycl::bit_cast<uint16_t>(sycl::half(d));
            w.dmin = sycl::bit_cast<uint16_t>(sycl::half(dm));
            for (unsigned j = 0; j < 8; ++j) {
                scales[j] = saturate ? (n % 2 ? 0 : 63) : (n * 7 + b * 3 + j * 11) % 64;
                mins[j] = saturate ? (n % 2 ? 63 : 0) : (n * 5 + b * 13 + j * 7) % 64;
            }
            for (unsigned j = 0; j < 4; ++j) {
                w.scales[j] = scales[j] | ((scales[j + 4] >> 4) << 6);
                w.scales[j + 4] = mins[j] | ((mins[j + 4] >> 4) << 6);
                w.scales[j + 8] = (scales[j + 4] & 15) | ((mins[j + 4] & 15) << 4);
            }
            for (unsigned i = 0; i < 256; ++i) {
                const unsigned quant = saturate ? 31 : (i * 7 + (i / 32) * 11 + n * 3 + b * 11) % 32;
                w.qs[(i / 64) * 32 + i % 32] |= (quant & 15) << ((i % 64 >= 32) ? 4 : 0);
                w.qh[i % 32] |= (quant >> 4) << (i / 32);
                const double value = double(d) * scales[i / 32] * quant - double(dm) * mins[i / 32];
                sum += double(float(input[b * 256 + i])) * value;
            }
        }
        expected[16 + n] = sycl::half(std::clamp(sum, -65504., 65504.));
    }
    auto* allocation = sycl::aligned_alloc_device<sycl::half>(64, K + 1, q);
    auto* packed = sycl::malloc_device<ie::block_q5_K>(weights.size(), q);
    auto* output = sycl::malloc_device<sycl::half>(expected.size(), q);
    if (!allocation || !packed || !output) throw std::runtime_error("USM allocation failed");
    auto* a = allocation + unsigned(unaligned);
    std::vector<sycl::event> deps{
        q.memcpy(a, input.data(), input.size() * sizeof(sycl::half)),
        q.memcpy(packed, weights.data(), weights.size() * sizeof(ie::block_q5_K)),
        q.fill(output, sycl::half(-123), expected.size())};
    auto done = ie::gemv_q5_K(q, a, packed, output + 16, K, N, deps);
    std::vector<sycl::half> actual(expected.size());
    q.submit([&](sycl::handler& h) {
        h.depends_on(done);
        h.memcpy(actual.data(), output, actual.size() * sizeof(sycl::half));
    }).wait_and_throw();
    sycl::free(allocation, q);
    sycl::free(packed, q);
    sycl::free(output, q);
    if (std::memcmp(actual.data(), expected.data(), expected.size() * sizeof(sycl::half))) {
        for (size_t i = 0; i < actual.size(); ++i) if (actual[i] != expected[i]) {
            std::fprintf(stderr, "K%u N%u offset%d saturation%d: output %zu expected %g got %g\n",
                         K, N, unaligned, saturate, i, float(expected[i]), float(actual[i]));
            break;
        }
        throw std::runtime_error("output or canary mismatch");
    }
    std::printf("EXACT K%u N%u offset%d saturation%d\n", K, N, unaligned, saturate);
}
}

int main() {
    try {
        auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
        if (devices.empty()) { std::puts("SKIP: no SYCL GPU"); return 77; }
        auto device = devices.back();
        for (auto d : devices)
            if (d.get_info<sycl::info::device::name>().find("B70") != std::string::npos) device = d;
        // Out-of-order by default: the returned event must cover the output and honor all uploads.
        sycl::queue q(device);
        for (auto [K, N] : {std::pair{2048u, 4096u}, {2048u, 4095u}, {2048u, 4097u},
                            {256u, 1024u}, {768u, 1025u}, {4096u, 2048u},
                            {8192u, 1024u}, {4096u, 1023u}, {8448u, 1024u},
                            {256u, 4097u}, {768u, 19u}})
            check(q, K, N, false, false);
        check(q, 2048, 4096, true, false);
        check(q, 2048, 4096, false, true);
        check(q, 2048, 4096, true, true);
        std::puts("ALL_EXACT");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
