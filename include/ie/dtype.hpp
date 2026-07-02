// include/ie/dtype.hpp — DType enum and type-info table.
//
// IDs match ggml_type from llama.cpp's ggml.h verbatim, so reading them out of
// a GGUF file is just a static_cast. Block sizes and bytes/block come from
// research/03_quant_formats.md §1.4 and §2.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace ie {

enum class DType : uint32_t {
    kF32      = 0,
    kF16      = 1,
    kQ4_0     = 2,
    kQ4_1     = 3,
    // 4, 5 retired (Q4_2, Q4_3) — never accept these
    kQ5_0     = 6,
    kQ5_1     = 7,
    kQ8_0     = 8,
    kQ8_1     = 9,
    kQ2_K     = 10,
    kQ3_K     = 11,
    kQ4_K     = 12,
    kQ5_K     = 13,
    kQ6_K     = 14,
    kQ8_K     = 15,
    kIQ2_XXS  = 16,
    kIQ2_XS   = 17,
    kIQ3_XXS  = 18,
    kIQ1_S    = 19,
    kIQ4_NL   = 20,
    kIQ3_S    = 21,
    kIQ2_S    = 22,
    kIQ4_XS   = 23,
    kI8       = 24,
    kI16      = 25,
    kI32      = 26,
    kI64      = 27,
    kF64      = 28,
    kIQ1_M    = 29,
    kBF16     = 30,
    // 31..33 reserved
    kTQ1_0    = 34,
    kTQ2_0    = 35,
    // 36, 38 reserved
    kEXL3     = 37,   // engine-custom: EXL3 (QTIP) trellis. NOT a ggml type — our
                      // importer owns this id (EXL3 ships as safetensors). The
                      // trellis is stored as a [16*bits, N/16, K/16] tensor at
                      // 2 bytes/elem (int16 words); bits = shape[0]/16. The suh/svh
                      // side-vectors ride as sibling F16 tensors (<base>.suh/.svh).
    kMXFP4    = 39,
    kCount    = 40,
};

struct TypeInfo {
    DType            dtype;
    std::string_view name;        // canonical short name, e.g. "Q4_K"
    uint32_t         block_size;  // elements per block (1 for non-quantized)
    uint32_t         block_bytes; // bytes per block
    bool             is_quantized;
};

// Returns null if dtype is unknown / retired (e.g. ids 4, 5).
const TypeInfo* type_info(DType d) noexcept;

// Convenience accessors (return 0 / "" if dtype is unknown).
inline std::string_view type_name(DType d) noexcept {
    if (auto* ti = type_info(d)) return ti->name;
    return "?";
}
inline uint32_t type_block_size(DType d) noexcept {
    if (auto* ti = type_info(d)) return ti->block_size;
    return 0;
}
inline uint32_t type_block_bytes(DType d) noexcept {
    if (auto* ti = type_info(d)) return ti->block_bytes;
    return 0;
}

// bytes for a tensor of this dtype with `n_elements` (last dim must be a
// multiple of block_size for quantized types — caller's responsibility).
size_t bytes_for(DType d, size_t n_elements) noexcept;

}  // namespace ie
