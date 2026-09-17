// Q preparation equivalence across staged and direct-SLM dispatch, with
// finite/saturated fp16 operands, odd segmented KV boundaries, guards,
// producer dependencies, input/output aliasing, and mixed asynchronous reuse.
#include "ie/deepseek4_attn.hpp"
#include <sycl/sycl.hpp>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

int main() {
    sycl::queue q(sycl::gpu_selector_v); // Dependencies must work out of order.
    constexpr uint32_t H = 32, D = 512, R = 17, A = 131, B = 129, K = A + B;
    constexpr size_t G = 128;
    int failures = 0;
    auto check = [&](bool pass, const char* label) {
        std::printf("[%s] %s\n", pass ? "PASS" : "FAIL", label);
        failures += !pass;
    };
    std::mt19937 rng(914);
    std::uniform_real_distribution<float> random(-.5f, .5f);
    for (bool half_b : {false, true}) {
        auto alloc = [&](size_t n) { return sycl::malloc_shared<float>(n, q); };
        float* a = alloc(A * D);
        float* b32 = alloc((B + 64) * D);
        auto b16 = sycl::malloc_shared<sycl::half>((B + 64) * D, q);
        float* sinks = alloc(H);
        for (size_t i = 0; i < A * D; ++i) a[i] = random(rng);
        for (size_t i = 0; i < (B + 64) * D; ++i) {
            b32[i] = i < B * D ? random(rng) : 0.f;
            b16[i] = sycl::half(b32[i]);
        }
        for (uint32_t h = 0; h < H; ++h) sinks[h] = random(rng);
        ie::Ds4KvSegs seg{a, A, half_b ? static_cast<void*>(b16) : static_cast<void*>(b32), B, half_b};
        for (uint32_t T : {511u, 512u, 513u}) {
            const size_t n = size_t(T) * H * D, nr = size_t(R) * H * D;
            std::vector<float> source(n), mask(size_t(T) * K);
            for (size_t i = 0; i < nr; ++i) source[i] = random(rng);
            source[0] = 70000.f; source[1] = -70000.f;
            source[D] = 65504.f; source[D + 1] = -65504.f; source[D + 2] = 0x1p-24f;
            for (size_t i = nr; i < n; ++i) source[i] = source[i % nr];
            for (uint32_t t = 0; t < T; ++t)
                for (uint32_t i = 0; i < K; ++i)
                    mask[size_t(t) * K + i] = t % R == 3 ? -INFINITY
                        : ((i + t % R) % 5 == 0 ? (i % 11) * .01f : -INFINITY);
            const size_t overlap_offset = size_t(257) * H * D;
            float* x = alloc(n + overlap_offset); float* m = alloc(mask.size());
            float* ref = alloc(nr); float* out = alloc(n + 2 * G);
            std::copy(source.begin(), source.end(), x);
            std::copy(mask.begin(), mask.end(), m);
            std::fill(out, out + n + 2 * G, 12345.f);
            for (bool dense : {false, true}) {
                const float* mp = dense ? nullptr : m;
                const float* sp = dense ? nullptr : sinks;
                auto run = [&](uint32_t rows, float* output, bool prepared,
                               const std::vector<sycl::event>& deps = std::vector<sycl::event>{}) {
                    return ie::ds4_attention_xmx_segs(q, x, seg, mp, sp, output,
                        rows, H, D, 1.f / std::sqrt(float(D)), prepared, deps);
                };
                run(R, ref, false).wait_and_throw();
                // Poison Q, then restore it through a delayed external producer.
                // Reuse prepared KV so its conversion cannot accidentally supply
                // the missing direct-Q dependency if that edge is removed.
                std::fill(x, x + n, 0.f);
                auto delay = q.submit([&](sycl::handler& h) {
                    h.host_task([] { std::this_thread::sleep_for(std::chrono::milliseconds(20)); });
                });
                auto producer = q.submit([&](sycl::handler& h) {
                    h.depends_on(delay);
                    h.memcpy(x, source.data(), n * sizeof(float));
                });
                run(T, out + G, true, {producer}).wait_and_throw();
                auto matches = [&](const float* y) {
                    for (size_t i = 0; i < n; ++i)
                        if (!std::isfinite(y[i]) || std::memcmp(y + i, ref + i % nr, sizeof(float))) return false;
                    return true;
                };
                std::printf("T=%u b16=%d dense=%d ", T, half_b, dense);
                check(matches(out + G), "all rows match the 17-row staged oracle exactly");
                producer.wait_and_throw(); // Also drain the producer on a failing mutation.
                bool guard = true;
                for (size_t i = 0; i < G; ++i) guard &= out[i] == 12345.f && out[G + n + i] == 12345.f;
                check(guard, "output guard regions intact");
                const int repeats = half_b && T == 512 && !dense ? 1024 : 128;
                const auto start = std::chrono::steady_clock::now();
                auto e = run(T, out + G, false);
                for (int repeat = 0; repeat < repeats; ++repeat)
                    e = run(repeat % 2 ? T : R, out + G, true, {e});
                e.wait_and_throw();
                check(matches(out + G), "mixed asynchronous submissions preserve prepared KV");
                std::printf("stress_calls=%d elapsed_ms=%.3f\n", repeats + 1,
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
                run(T, x, false).wait_and_throw();
                check(matches(x), "Q/output exact alias matches staged oracle");
                std::copy(source.begin(), source.end(), x);
                run(T, x + overlap_offset, false).wait_and_throw();
                check(matches(x + overlap_offset), "shifted Q/output overlap matches staged oracle");
                std::copy(source.begin(), source.end(), x);
            }
            for (float* p : {x, m, ref, out}) sycl::free(p, q);
        }
        for (float* p : {a, b32, sinks}) sycl::free(p, q);
        sycl::free(b16, q);
    }
    std::printf("FAILURES=%d\n", failures);
    return failures ? 1 : 0;
}
