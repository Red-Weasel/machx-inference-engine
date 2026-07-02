// src/core/dtype.cpp — type-info table.

#include "ie/dtype.hpp"

#include <array>

namespace ie {

namespace {

// Block sizes / bytes-per-block from research/03 §2 and ggml-common.h.
//
//                   { dtype,         name,       block_size, bytes,  quantized? }
constexpr std::array<TypeInfo, static_cast<size_t>(DType::kCount)> kTable = {{
    { DType::kF32,     "F32",          1,    4,   false },
    { DType::kF16,     "F16",          1,    2,   false },
    { DType::kQ4_0,    "Q4_0",        32,   18,   true  },
    { DType::kQ4_1,    "Q4_1",        32,   20,   true  },
    { DType{4},        "<retired-4>",  0,    0,   false },
    { DType{5},        "<retired-5>",  0,    0,   false },
    { DType::kQ5_0,    "Q5_0",        32,   22,   true  },
    { DType::kQ5_1,    "Q5_1",        32,   24,   true  },
    { DType::kQ8_0,    "Q8_0",        32,   34,   true  },
    { DType::kQ8_1,    "Q8_1",        32,   36,   true  },
    { DType::kQ2_K,    "Q2_K",       256,   84,   true  },
    { DType::kQ3_K,    "Q3_K",       256,  110,   true  },
    { DType::kQ4_K,    "Q4_K",       256,  144,   true  },
    { DType::kQ5_K,    "Q5_K",       256,  176,   true  },
    { DType::kQ6_K,    "Q6_K",       256,  210,   true  },
    { DType::kQ8_K,    "Q8_K",       256,  292,   true  },
    { DType::kIQ2_XXS, "IQ2_XXS",    256,   66,   true  },
    { DType::kIQ2_XS,  "IQ2_XS",     256,   74,   true  },
    { DType::kIQ3_XXS, "IQ3_XXS",    256,   98,   true  },
    { DType::kIQ1_S,   "IQ1_S",      256,   50,   true  },
    { DType::kIQ4_NL,  "IQ4_NL",      32,   18,   true  },
    { DType::kIQ3_S,   "IQ3_S",      256,  110,   true  },
    { DType::kIQ2_S,   "IQ2_S",      256,   82,   true  },
    { DType::kIQ4_XS,  "IQ4_XS",     256,  136,   true  },
    { DType::kI8,      "I8",           1,    1,   false },
    { DType::kI16,     "I16",          1,    2,   false },
    { DType::kI32,     "I32",          1,    4,   false },
    { DType::kI64,     "I64",          1,    8,   false },
    { DType::kF64,     "F64",          1,    8,   false },
    { DType::kIQ1_M,   "IQ1_M",      256,   56,   true  },
    { DType::kBF16,    "BF16",         1,    2,   false },
    { DType{31},       "<reserved>",   0,    0,   false },
    { DType{32},       "<reserved>",   0,    0,   false },
    { DType{33},       "<reserved>",   0,    0,   false },
    { DType::kTQ1_0,   "TQ1_0",      256,   54,   true  },
    { DType::kTQ2_0,   "TQ2_0",      256,   66,   true  },
    { DType{36},       "<reserved>",   0,    0,   false },
    // EXL3 trellis: stored as int16 words (block_size 1, 2 bytes/elem) so GGUF's
    // bytes_for() size math is exact for a [16*bits, N/16, K/16] tensor.
    { DType::kEXL3,    "EXL3",         1,    2,   true  },
    { DType{38},       "<reserved>",   0,    0,   false },
    { DType::kMXFP4,   "MXFP4",       32,   17,   true  },
}};

}  // namespace

const TypeInfo* type_info(DType d) noexcept {
    auto i = static_cast<uint32_t>(d);
    if (i >= kTable.size()) return nullptr;
    const auto& ti = kTable[i];
    if (ti.block_bytes == 0) return nullptr;     // retired/reserved slot
    return &ti;
}

size_t bytes_for(DType d, size_t n_elements) noexcept {
    auto* ti = type_info(d);
    if (!ti) return 0;
    if (ti->block_size == 0) return 0;
    // Round up so callers asking about a slice that doesn't end on a block
    // boundary still get a sane answer; in practice GGUF always aligns.
    const size_t blocks = (n_elements + ti->block_size - 1) / ti->block_size;
    return blocks * ti->block_bytes;
}

}  // namespace ie
