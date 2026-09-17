// tests/unit/dequant_ref_test.cpp — bit-exact known-answer test (KAT) for the
// host reference dequantizers added for the increment-2 universal-loader
// coverage: Q4_1, Q5_0, Q5_1, Q2_K, Q3_K.
//
// The fixtures in `dequant_fixtures.inc` were produced by ggml itself: each
// entry pairs the raw packed GGUF bytes with ggml's OWN dequantized fp32
// output ("golden"). We run our `ie::ref::dequant_*_buffer` over the same
// packed bytes and require the result to match golden[0..n).
//
// Comparison: strict bit-exact (memcmp of the 4-byte fp32 pattern). The golden
// values are hex-float literals (%a) — the exact, sign-of-zero-preserving
// serialization of ggml's output — so our dequant must reproduce every bit,
// including -0.0. ANY bit difference is a real mismatch and FAILS the test.
//
// CPU-only, header-only: no SYCL, no ie_core. `ctest` keys on the exit code —
// nonzero on ANY mismatch or unrecognized fixture name, zero only when all five
// types dequant bit-identically to ggml's golden.

#include "ie/dequant_ref.hpp"
#include "dequant_fixtures.inc"
#include "dequant_ref_test.iq3xxs.inc"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

// Dispatch a fixture's packed bytes to the matching reference dequantizer.
// Returns false if the name is not one of the five expected types.
bool dequant_by_name(const char* name, const void* packed, std::size_t n,
                     float* out) {
    const std::string type(name);
    if (type == "Q4_1") { ie::ref::dequant_q4_1_buffer(packed, n, out); return true; }
    if (type == "Q5_0") { ie::ref::dequant_q5_0_buffer(packed, n, out); return true; }
    if (type == "Q5_1") { ie::ref::dequant_q5_1_buffer(packed, n, out); return true; }
    if (type == "Q2_K") { ie::ref::dequant_q2_K_buffer(packed, n, out); return true; }
    if (type == "Q3_K") { ie::ref::dequant_q3_K_buffer(packed, n, out); return true; }
    if (type == "IQ4_NL") { ie::ref::dequant_iq4_nl_buffer(packed, n, out); return true; }
    if (type == "IQ4_XS") { ie::ref::dequant_iq4_xs_buffer(packed, n, out); return true; }
    // IQ3_XXS: "" = real DeepSeek-V4 ffn_gate_exps bytes, "_S" = ggml-quantized synthetic.
    if (type == "IQ3_XXS" || type == "IQ3_XXS_S") {
        ie::ref::dequant_iq3_xxs_buffer(packed, n, out); return true;
    }
    return false;
}

}  // namespace

int main() {
    int failures = 0;
    const std::size_t num_fixtures =
        sizeof(ie::test::kFixtures) / sizeof(ie::test::kFixtures[0]) +
        sizeof(ie::test::kIQ3XXSFixtures) / sizeof(ie::test::kIQ3XXSFixtures[0]);

    for (std::size_t f = 0; f < num_fixtures; ++f) {
        const std::size_t n_base =
            sizeof(ie::test::kFixtures) / sizeof(ie::test::kFixtures[0]);
        const ie::test::Fixture& fx = (f < n_base)
            ? ie::test::kFixtures[f]
            : ie::test::kIQ3XXSFixtures[f - n_base];

        std::vector<float> got(fx.n, 0.0f);
        if (!dequant_by_name(fx.name, fx.packed, fx.n, got.data())) {
            std::printf("  %-5s n=%-4zu  UNRECOGNIZED FIXTURE NAME  FAIL\n",
                        fx.name, fx.n);
            ++failures;
            continue;
        }

        std::size_t mismatches = 0;  // any bit difference -> failure
        for (std::size_t i = 0; i < fx.n; ++i) {
            if (std::memcmp(&got[i], &fx.golden[i], sizeof(float)) != 0)
                ++mismatches;
        }

        const bool ok = (mismatches == 0);
        std::printf("  %-5s n=%-4zu  mismatches=%-4zu  %s\n",
                    fx.name, fx.n, mismatches, ok ? "OK" : "FAIL");
        if (!ok) ++failures;
    }

    std::printf("dequant_ref KAT: %zu/%zu types OK\n",
                num_fixtures - static_cast<std::size_t>(failures), num_fixtures);
    return failures == 0 ? 0 : 1;
}
