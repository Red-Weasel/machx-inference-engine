// A chunk and single-token calls must produce the same convolution outputs
// and history, including chunks shorter than the retained history.
#include "ie/ops.hpp"
#include <cstdio>
#include <cstring>
#include <random>

static bool check(sycl::queue& q, uint32_t kernel, uint32_t channels) {
    constexpr uint32_t tokens = 17, guard = 32;
    const size_t state_n = uint64_t(kernel - 1) * channels;
    std::mt19937 rng(5300 + kernel + channels);
    std::uniform_real_distribution<float> real(-.5f, .5f);
    std::vector<sycl::half> x(uint64_t(tokens) * channels), w(uint64_t(kernel) * channels);
    std::vector<sycl::half> init(state_n + 2 * guard, sycl::half(7));
    for (auto& v : x) v = sycl::half(real(rng));
    for (auto& v : w) v = sycl::half(real(rng));
    for (size_t i = 0; i < state_n; ++i) init[guard + i] = sycl::half(real(rng));
    auto* dx = sycl::malloc_device<sycl::half>(x.size(), q);
    auto* dw = sycl::malloc_device<sycl::half>(w.size(), q);
    auto* state = sycl::malloc_device<sycl::half>(init.size(), q);
    auto* y = sycl::malloc_device<sycl::half>(x.size() + 2 * guard, q);
    auto ex = q.memcpy(dx, x.data(), x.size() * 2);
    auto ew = q.memcpy(dw, w.data(), w.size() * 2);
    std::vector<sycl::half> reference;
    bool ok = true;
    for (uint32_t chunk : {1u, 2u, 3u, 4u, 8u, 17u}) {
        auto es = q.memcpy(state, init.data(), init.size() * 2);
        auto ey = q.fill(y, sycl::half(7), x.size() + 2 * guard);
        std::vector<sycl::event> deps{ex, ew, es, ey};
        for (uint32_t pos = 0; pos < tokens; pos += chunk) {
            auto e = ie::depthwise_conv1d_causal(q, dx + uint64_t(pos) * channels, dw,
                state + guard, y + guard + uint64_t(pos) * channels,
                std::min(chunk, tokens - pos), channels, kernel, deps);
            deps = {e};
        }
        std::vector<sycl::half> actual(x.size() + 2 * guard), history(init.size());
        q.submit([&](sycl::handler& h) { h.depends_on(deps); h.memcpy(actual.data(), y, actual.size() * 2); }).wait_and_throw();
        q.submit([&](sycl::handler& h) { h.depends_on(deps); h.memcpy(history.data(), state, history.size() * 2); }).wait_and_throw();
        if (chunk == 1) reference = actual;
        if (std::memcmp(reference.data(), actual.data(), actual.size() * 2)) {
            std::printf("FAIL kernel%u C%u chunk%u: output/chunk mismatch\n", kernel, channels, chunk); ok = false;
        }
        // Independent history oracle: suffix of [initial history, all input].
        for (size_t i = 0; i < state_n; ++i) {
            const size_t concatenated = x.size() + i;
            const sycl::half expected = concatenated < state_n
                ? init[guard + concatenated] : x[concatenated - state_n];
            if (std::memcmp(&expected, &history[guard + i], 2)) {
                std::printf("FAIL kernel%u C%u chunk%u: history[%zu]\n", kernel, channels, chunk, i); ok = false; break;
            }
        }
        for (size_t i = 0; i < guard; ++i)
            if (float(history[i]) != 7 || float(history[guard + state_n + i]) != 7 ||
                float(actual[i]) != 7 || float(actual[guard + x.size() + i]) != 7) {
                std::printf("FAIL kernel%u C%u chunk%u: canary\n", kernel, channels, chunk); ok = false; break;
            }
    }
    sycl::free(dx, q); sycl::free(dw, q); sycl::free(state, q); sycl::free(y, q);
    std::printf("%s kernel%u C%u: chunk equivalence, history, events, canaries\n", ok ? "PASS" : "FAIL", kernel, channels);
    return ok;
}
int main(int argc, char**) {
    std::vector<sycl::device> ds;
    for (auto d : sycl::device::get_devices(sycl::info::device_type::gpu))
        if (d.get_info<sycl::info::device::name>().find("B70") != std::string::npos) ds.push_back(d);
    if (ds.empty()) return 2;
    sycl::queue q(ds.back()); // deliberately out of order
    bool ok = true;
    for (uint32_t k : {1u, 2u, 4u, 7u, 65u}) {
        if (argc > 1 && k == 1) continue; // safe baseline reproduction, before kernel=1 fix
        for (uint32_t c : {1u, 65u, 4096u}) ok &= check(q, k, c);
    }
    return ok ? 0 : 1;
}
