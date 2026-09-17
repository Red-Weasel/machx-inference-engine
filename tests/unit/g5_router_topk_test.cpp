// Exact normal-path routing plus ragged-count and underflow reliability gates.
#include "ie/glm5next.hpp"
#include <sycl/sycl.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

sycl::event frozen(sycl::queue& q, const float* rl, const float* bias,
                          int32_t* top, float* topw, uint32_t NE, uint32_t TOPK,
                          float wscale, uint32_t T,
                          const std::vector<sycl::event>& deps) {
    constexpr int SG = 32;
    const uint32_t per = NE / SG;   // 9 at 288
    return q.submit( [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(uint64_t(T) * SG, SG),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint32_t t = uint32_t(it.get_group(0));
            auto sg = it.get_sub_group();
            const uint32_t lane = uint32_t(sg.get_local_id()[0]);
            const float* rlt = rl + uint64_t(t) * NE;
            float sc[16];
            for (uint32_t i = 0; i < per; ++i) {
                const float v = rlt[lane * per + i];
                sc[i] = 1.f / (1.f + sycl::exp(-v)) + bias[lane * per + i];
            }
            float wsum = 0.f;
            for (uint32_t k = 0; k < TOPK; ++k) {
                float bv = -INFINITY; uint32_t bi = 0;
                for (uint32_t i = 0; i < per; ++i) if (sc[i] > bv) { bv = sc[i]; bi = i; }
                const float m = sycl::reduce_over_group(sg, bv, sycl::maximum<float>());
                const uint32_t cand = (bv == m) ? (lane * per + bi) : NE;
                const uint32_t win  = sycl::reduce_over_group(sg, cand, sycl::minimum<uint32_t>());
                if (win / per == lane) sc[win - lane * per] = -INFINITY;
                const float wp = 1.f / (1.f + sycl::exp(-rlt[win]));
                wsum += wp;
                if (lane == 0) { top[uint64_t(t) * TOPK + k] = int32_t(win);
                                 topw[uint64_t(t) * TOPK + k] = wp; }
            }
            if (lane == 0)
                for (uint32_t k = 0; k < TOPK; ++k)
                    topw[uint64_t(t) * TOPK + k] = topw[uint64_t(t) * TOPK + k] / wsum * wscale;
        });
    });
}

static void check(sycl::queue& q, uint32_t E, uint32_t K, uint32_t T, int pattern) {
    const size_t nr = size_t(T) * E, ny = size_t(T) * K + 32;
    std::mt19937 rng(7721 + E + K + T);
    std::vector<float> logits(nr), bias(E), weights(ny), old_weights(ny);
    std::vector<int32_t> indices(ny), old_indices(ny);
    for (auto& v : logits) v = (int(rng() % 12001) - 6000) / 1000.f;
    for (auto& v : bias) v = (int(rng() % 2001) - 1000) / 10000.f;
    if (pattern) {
        std::fill(logits.begin(), logits.end(), 0.f);
        std::fill(bias.begin(), bias.end(), 0.f);
        if (pattern == 2)
            for (uint32_t t = 0; t < T; ++t) logits[size_t(t) * E + E - 1] = 20.f;
        if (pattern == 3)
            for (uint32_t e = 0; e < E; ++e) bias[e] = float(e);
        if (pattern == 4)
            for (size_t j = 0; j < nr; ++j) logits[j] = -1000.f - float(j % 3);
    }
    auto* r = sycl::malloc_device<float>(nr, q);
    auto* b = sycl::malloc_device<float>(E, q);
    auto* ids = sycl::malloc_device<int32_t>(ny, q);
    auto* w = sycl::malloc_device<float>(ny, q);
    if (!r || !b || !ids || !w) throw std::runtime_error("top-k allocation");
    auto invoke = [&](bool reference) {
        std::vector<sycl::event> deps{
            q.memcpy(r, logits.data(), nr * sizeof(float)),
            q.memcpy(b, bias.data(), E * sizeof(float)),
            q.fill(ids, int32_t(-123), ny), q.fill(w, -123.f, ny)};
        auto done = reference
            ? frozen(q, r, b, ids + 16, w + 16, E, K, 2.5f, T, deps)
            : ie::glm5_router_topk(q, r, b, ids + 16, w + 16, E, K, 2.5f, T, deps);
        q.submit([&](sycl::handler& h) {
            h.depends_on(done);
            h.memcpy(reference ? old_indices.data() : indices.data(), ids, ny * sizeof(int32_t));
        }).wait_and_throw();
        q.submit([&](sycl::handler& h) {
            h.depends_on(done);
            h.memcpy(reference ? old_weights.data() : weights.data(), w, ny * sizeof(float));
        }).wait_and_throw();
    };
    const bool old_supported = E >= 32 && E <= 512 && E % 32 == 0 && pattern != 4;
    if (old_supported) invoke(true);
    invoke(false);
    if (old_supported && (std::memcmp(indices.data(), old_indices.data(), ny * sizeof(int32_t)) ||
                          std::memcmp(weights.data(), old_weights.data(), ny * sizeof(float))))
        throw std::runtime_error("top-k normal path changed bits");
    for (size_t j = 0; j < ny; ++j)
        if ((j < 16 || j >= ny - 16) && (indices[j] != -123 || weights[j] != -123.f))
            throw std::runtime_error("top-k output guard");
    for (uint32_t t = 0; t < T; ++t) {
        std::vector<float> probability(E), score(E);
        std::vector<uint32_t> expected(E);
        std::iota(expected.begin(), expected.end(), 0u);
        for (uint32_t e = 0; e < E; ++e) {
            probability[e] = 1.f / (1.f + std::exp(-logits[size_t(t) * E + e]));
            score[e] = probability[e] + bias[e];
        }
        std::sort(expected.begin(), expected.end(), [&](uint32_t a, uint32_t b) {
            return score[a] > score[b] || (score[a] == score[b] && a < b);
        });
        float total = 0.f;
        for (uint32_t k = 0; k < K; ++k) total += probability[expected[k]];
        if (total == 0.f) {
            float largest = -INFINITY;
            for (uint32_t k = 0; k < K; ++k)
                largest = std::max(largest, logits[size_t(t) * E + expected[k]]);
            for (uint32_t k = 0; k < K; ++k) {
                probability[expected[k]] = std::exp(logits[size_t(t) * E + expected[k]] - largest);
                total += probability[expected[k]];
            }
        }
        float got_sum = 0.f;
        for (uint32_t k = 0; k < K; ++k) {
            const size_t j = 16 + size_t(t) * K + k;
            const float want = probability[expected[k]] / total * 2.5f;
            if (indices[j] != int32_t(expected[k]) || !std::isfinite(weights[j]) ||
                std::abs(weights[j] - want) > 3e-6f) {
                std::fprintf(stderr, "E%u K%u T%u pattern%d row%u k%u: id%d/%u w%.9g/%.9g\n",
                             E, K, T, pattern, t, k, indices[j], expected[k], weights[j], want);
                throw std::runtime_error("top-k independent oracle");
            }
            got_sum += weights[j];
        }
        if (std::abs(got_sum - 2.5f) > 1e-5f) throw std::runtime_error("top-k weight sum");
    }
    sycl::free(r, q); sycl::free(b, q); sycl::free(ids, q); sycl::free(w, q);
    std::printf("PASS E%u K%u T%u pattern%d exact_old%d\n", E, K, T, pattern, old_supported);
}

int main() {
    unsigned devices = 0;
    for (auto d : sycl::device::get_devices(sycl::info::device_type::gpu)) {
        if (d.get_backend() != sycl::backend::ext_oneapi_level_zero ||
            d.get_info<sycl::info::device::name>().find("B70") == std::string::npos) continue;
        ++devices;
        sycl::queue q(d); // Out of order: every input and result needs a real dependency.
        std::printf("Level Zero PCI %s\n",
                    d.get_info<sycl::ext::intel::info::device::pci_address>().c_str());
        for (uint32_t E : {1u, 19u, 31u, 32u, 33u, 255u, 256u, 257u, 287u, 288u,
                           289u, 511u, 512u, 513u, 544u, 1025u, 2048u})
            for (int pattern : {0, 1, 2, 3, 4}) check(q, E, std::min(E, 8u), 3, pattern);
        check(q, 33, 33, 2, 1);
        check(q, 288, 16, 128, 0);
        check(q, 288, 8, 1024, 0);
        for (auto counts : {std::pair{0u, 1u}, std::pair{8u, 0u}, std::pair{8u, 9u},
                            std::pair{0x80000000u, 1u}}) {
            bool threw = false;
            try { ie::glm5_router_topk(q, nullptr, nullptr, nullptr, nullptr,
                                      counts.first, counts.second, 2.5f, 1); }
            catch (const std::invalid_argument&) { threw = true; }
            if (!threw) throw std::runtime_error("top-k count validation");
        }
        auto* marker = sycl::malloc_device<int>(1, q);
        auto fill = q.fill(marker, 42, 1);
        auto empty = ie::glm5_router_topk(q, nullptr, nullptr, nullptr, nullptr, 288, 8, 2.5f, 0, {fill});
        int got = 0;
        q.submit([&](sycl::handler& h) { h.depends_on(empty); h.memcpy(&got, marker, sizeof(int)); }).wait_and_throw();
        sycl::free(marker, q);
        if (got != 42) throw std::runtime_error("empty top-k dependency");
    }
    if (!devices) return 77;
    std::puts("GATE PASSED");
}
