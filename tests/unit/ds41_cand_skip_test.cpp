// tests/unit/ds41_cand_skip_test.cpp -- V4.1 Phase 45 (docs/deepseek41/85): ds4_indexer_score_candidates must equal
// ds4_indexer_score followed by ds41_candidate_apply BIT FOR BIT, on the keytile shape (T >= 64) and the headlane shape
// (small T), with odd key counts (a tail key pair / a partial last block), rows with every block kept and rows with none.
#include "ie/deepseek4_attn.hpp"
#include "ie/deepseek41_candidate.hpp"
#include <sycl/sycl.hpp>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

int main() {
    sycl::device dev;
    bool found = false;
    for (const auto& p : sycl::platform::get_platforms()) {
        if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        for (const auto& d : p.get_devices(sycl::info::device_type::gpu))
            if (!found && d.get_info<sycl::info::device::name>().find("Arc") != std::string::npos) { dev = d; found = true; }
    }
    if (!found) { std::printf("no Arc GPU\n"); return 1; }
    sycl::queue q(dev, sycl::property::queue::in_order{});
    const uint32_t H = 32, D = 128, B = 8;
    struct Case { uint32_t T, n_keys; };
    const Case cases[] = {{1, 20000}, {1, 16385}, {7, 16391}, {64, 16389}, {256, 40001}, {130, 33}, {65, 131077}};
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.f);
    int fails = 0;
    for (const auto& c : cases) {
        const uint32_t NB = (c.n_keys + B - 1) / B;
        auto* qin  = sycl::aligned_alloc_shared<float>(64, size_t(c.T) * H * D, q);
        auto* keys = sycl::aligned_alloc_shared<float>(64, size_t(c.n_keys) * D, q);
        auto* wp   = sycl::aligned_alloc_shared<float>(64, size_t(c.T) * H, q);
        auto* keep = sycl::malloc_shared<uint8_t>(size_t(c.T) * NB, q);
        auto* ref  = sycl::aligned_alloc_shared<float>(64, size_t(c.T) * c.n_keys, q);
        auto* got  = sycl::aligned_alloc_shared<float>(64, size_t(c.T) * c.n_keys, q);
        for (size_t i = 0; i < size_t(c.T) * H * D; ++i) qin[i] = nd(rng);
        for (size_t i = 0; i < size_t(c.n_keys) * D; ++i) keys[i] = nd(rng);
        for (size_t i = 0; i < size_t(c.T) * H; ++i) wp[i] = nd(rng);
        for (uint32_t t = 0; t < c.T; ++t)
            for (uint32_t b = 0; b < NB; ++b)
                keep[size_t(t) * NB + b] = t == 0 ? 1 : t == 1 ? 0 : uint8_t((rng() % 10) < 3);
        const float s1 = 1.0f / std::sqrt(float(D)), s2 = 1.0f / std::sqrt(float(H));
        ie::ds4_indexer_score(q, qin, keys, wp, ref, c.T, H, D, c.n_keys, s1, s2).wait();
        ie::ds41_candidate_apply(q, keep, ref, c.T, c.n_keys, B).wait();
        std::memset(got, 0x5A, size_t(c.T) * c.n_keys * sizeof(float));      // every entry must be written
        ie::ds4_indexer_score_candidates(q, qin, keys, wp, got, c.T, H, D, c.n_keys, s1, s2, keep, B,
                                         -std::numeric_limits<float>::infinity()).wait();
        size_t diff = 0, first = SIZE_MAX;
        for (size_t i = 0; i < size_t(c.T) * c.n_keys; ++i)
            if (std::memcmp(&ref[i], &got[i], sizeof(float)) != 0) { ++diff; if (first == SIZE_MAX) first = i; }
        std::printf("  T %4u n_keys %6u: %s (%zu of %zu entries differ%s)\n", c.T, c.n_keys, diff ? "FAIL" : "ok  ", diff,
                    size_t(c.T) * c.n_keys, diff ? (", first at row " + std::to_string(first / c.n_keys) + " key " + std::to_string(first % c.n_keys)).c_str() : "");
        fails += diff != 0;
        for (void* p : {static_cast<void*>(qin), static_cast<void*>(keys), static_cast<void*>(wp), static_cast<void*>(keep), static_cast<void*>(ref), static_cast<void*>(got)}) sycl::free(p, q);
    }
    std::printf("%s\n", fails ? "ds41_cand_skip_test: FAIL" : "ds41_cand_skip_test: PASS -- bit-identical to score + apply on every shape");
    return fails ? 1 : 0;
}
