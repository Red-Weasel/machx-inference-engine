#include "ie/ops.hpp"
#include "kda_norm_reference.hpp"
#ifdef KDA_PRIVATE
#include "candidate.hpp"
namespace impl = candidate;
#else
namespace impl = ie;
#endif
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <thread>

constexpr size_t guard = 17;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
struct Memory {
    sycl::queue& q;
    std::vector<void*> allocations;
    float* get(size_t n) {
        auto p = sycl::malloc_device<float>(n, q);
        if (!p) throw std::bad_alloc();
        allocations.push_back(p);
        return p;
    }
    ~Memory() { q.wait(); for (auto p : allocations) sycl::free(p, q); }
};
sycl::event copy(sycl::queue& q, float* dst, const float* src, size_t n,
                 const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& h) { h.depends_on(deps); h.memcpy(dst, src, n * sizeof(float)); });
}
void run(sycl::queue& q, unsigned rows, unsigned dim, bool alias, unsigned data) {
    const size_t n = size_t(rows) * dim, allocation = n + 2 * guard;
    Memory m{q, {}};
    auto x = m.get(allocation), old = m.get(allocation), now = m.get(allocation);
    std::vector<float> src(allocation, 7.f), initial(allocation, 7.f);
    for (size_t i = 0; i < n; ++i) {
        float v = float(int(i * 37 % 251) - 125) / 97;
        if (data == 1) v *= 1e-12f; // Exercise the norm floor, not epsilon addition.
        if (data == 2) v = i % 2 ? 0.f : -0.f;
        if (data == 3 && i % 137 == 0) v = i % 2 ? std::numeric_limits<float>::infinity()
                                                                : std::numeric_limits<float>::quiet_NaN();
        if (data == 4) v = std::ldexp(v, int(i * 19 % 81) - 40);
        if (data == 5) v = (i % 2 ? -1.f : 1.f) *
            (0.9f + float((i / dim) % 3) * 0.1f) * 1e-6f / std::sqrt(float(dim));
        src[guard + i] = v;
    }
    const float scale = data == 2 ? 0.f : (data == 1 ? -.125f : .125f), eps = 1e-6f;
    auto held = q.submit([](sycl::handler& h) {
        h.host_task([] { std::this_thread::sleep_for(std::chrono::milliseconds(2)); });
    });
    auto ex = copy(q, x, src.data(), allocation, {held});
    auto ea = copy(q, old, alias ? src.data() : initial.data(), allocation, {held});
    auto eb = copy(q, now, alias ? src.data() : initial.data(), allocation, {held});
    auto a = kda_norm_frozen::kda_l2norm(q, alias ? old + guard : x + guard, old + guard,
                                       rows, dim, scale, eps, {ex, ea});
    auto b = impl::kda_l2norm(q, alias ? now + guard : x + guard, now + guard,
                            rows, dim, scale, eps, {ex, eb});
    std::vector<float> av(allocation), bv(allocation);
    copy(q, av.data(), old, allocation, {a}).wait_and_throw();
    copy(q, bv.data(), now, allocation, {b}).wait_and_throw();
    for (size_t i = 0; i < allocation; ++i)
        require((std::isnan(av[i]) && std::isnan(bv[i])) || std::memcmp(&av[i], &bv[i], 4) == 0,
                "frozen KDA L2 output mismatch");
    for (size_t i = 0; i < guard; ++i)
        require(bv[i] == 7.f && bv[n + guard + i] == 7.f, "KDA L2 guard modified");
    if (data != 3) for (unsigned row = 0; row < rows; ++row) {
        double sum = 0;
        size_t offset = guard + size_t(row) * dim;
        for (unsigned d = 0; d < dim; ++d) sum += double(src[offset + d]) * src[offset + d];
        double r = scale / std::max(std::sqrt(sum), double(eps));
        for (unsigned d = 0; d < dim; ++d) {
            double expected = src[offset + d] * r;
            require(std::isfinite(bv[offset + d]) &&
                    std::abs(bv[offset + d] - expected) <= 1e-6 * std::max(1e-10, std::abs(expected)),
                    "independent KDA norm-floor oracle");
        }
    }
}
void empty(sycl::queue& q) {
    for (auto dims : {std::pair<unsigned, unsigned>{0, 128}, {3, 0}}) {
        std::atomic<bool> ready = false;
        auto e = q.submit([&](sycl::handler& h) { h.host_task([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(20)); ready.store(true);
        }); });
        auto done = impl::kda_l2norm(q, nullptr, nullptr, dims.first, dims.second, 1.f, 1e-6f, {e});
        done.wait_and_throw(); bool observed = ready.load(); e.wait_and_throw();
        require(observed, "empty KDA norm dropped dependency");
    }
}
int main() { try {
    sycl::queue q(sycl::gpu_selector_v); // Out-of-order queue exercises the public event contract.
    unsigned cases = 0;
    for (unsigned dim : {1, 63, 64, 65, 127, 128, 129})
        for (bool alias : {false, true}) for (unsigned data = 0; data < 6; ++data) {
            run(q, 17, dim, alias, data); ++cases;
        }
    for (unsigned dim : {64, 128}) for (unsigned rows : {1, 64, 128, 1024, 8192, 32768, 32769})
        for (bool alias : {false, true}) { run(q, rows, dim, alias, 0); ++cases; }
    empty(q);
    std::printf("PASS %u KDA norm cases: exact finite bits, norm-floor oracle, alias, guards, OOO and empty events\n", cases);
} catch (const std::exception& e) { std::fprintf(stderr, "FAIL %s\n", e.what()); return 1; } }
