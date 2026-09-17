// tests/unit/preflight_test.cpp — pure dtype-classification helper for the
// load-time preflight (Global Constraint #4 supported-set). Host-only, no GPU,
// no GGUF: is_loadable_weight_dtype is a pure whitelist predicate.
#undef NDEBUG
#include "ie/preflight.hpp"
#include "ie/dtype.hpp"
#include "ie/dequant_ref.hpp"   // IQ3_XXS false-loadable regression guard below
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <type_traits>

using ie::DType;
using ie::is_loadable_weight_dtype;

int main() {
    // Supported weight-dtype set — every one MUST classify loadable. The first
    // group has native/device fast paths; Q4_1/Q5_0/Q5_1/Q2_K/Q3_K load via the
    // host universal dequant→fp16 fallback (dense_dispatch upload_host_dequant_to_fp16).
    const DType supported[] = {
        DType::kF32, DType::kF16, DType::kBF16,
        DType::kQ4_0, DType::kQ8_0,
        DType::kQ4_K, DType::kQ5_K, DType::kQ6_K,
        DType::kMXFP4,
        DType::kQ4_1, DType::kQ5_0, DType::kQ5_1, DType::kQ2_K, DType::kQ3_K,
        DType::kIQ4_NL, DType::kIQ4_XS,
        DType::kEXL3,   // native trellis decode (gemv_exl3)
        // IQ3_XXS — the deepseek4 (DeepSeek-V4-Flash UD-Q3_K_XL) expert dtype;
        // host dequant via ie::ref::dequant_iq3_xxs_buffer (guarded below).
        DType::kIQ3_XXS,
    };
    for (DType d : supported) {
        assert(is_loadable_weight_dtype(d) &&
               "supported dtype must be loadable");
    }

    // Representative UNSUPPORTED spread — none has a load path today (no fast
    // kernel AND no host dequant). The remaining IQ*/TQ* trellis + Q8_1/Q8_K.
    const DType unsupported[] = {
        DType::kTQ2_0, DType::kQ8_1, DType::kQ8_K,
        DType::kIQ2_XXS, DType::kIQ2_XS,
        DType::kIQ1_S, DType::kIQ3_S,
        DType::kIQ2_S, DType::kIQ1_M, DType::kTQ1_0,
    };
    for (DType d : unsupported) {
        assert(!is_loadable_weight_dtype(d) &&
               "unsupported dtype must NOT be loadable");
    }

    // REGRESSION GUARD — the false-loadable claim. kIQ3_XXS was once put on the
    // whitelist while NO iq3_xxs dequantiser existed anywhere, so preflight
    // reported the DeepSeek-V4-Flash GGUF "loadable" for a load that could not
    // succeed. It is loadable only because that decode entry point now exists;
    // this pins the two together so deleting the dequantiser fails the build
    // here instead of silently restoring the false claim. Do NOT relax this — if
    // the dequantiser goes, kIQ3_XXS comes off the whitelist in the same edit.
    static_assert(std::is_same_v<decltype(&ie::ref::dequant_iq3_xxs_buffer),
                                 void (*)(const void*, std::size_t, float*)>,
                  "IQ3_XXS is loadable but its host dequantiser is gone");
    assert(is_loadable_weight_dtype(DType::kIQ3_XXS) &&
           "IQ3_XXS must stay loadable while its host dequantiser exists");

    // Integer INDEX dtypes are not weights and must never enter the weight
    // whitelist. deepseek4 ships three I32 hash-router tables (blk.0-2
    // ffn_gate_tid2eid); preflight stops them blocking the verdict by
    // classifying them as index tensors, NOT by calling them loadable weights.
    // Widening the whitelist here would assert an I32 weight is dequantisable.
    const DType index_dtypes[] = {
        DType::kI8, DType::kI16, DType::kI32, DType::kI64,
    };
    for (DType d : index_dtypes) {
        assert(!is_loadable_weight_dtype(d) &&
               "integer index dtype must NOT be a loadable weight dtype");
    }

    std::puts("preflight_test: OK");
    return 0;
}
