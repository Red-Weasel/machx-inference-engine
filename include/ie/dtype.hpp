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
    // 40 reserved
    kQ1_0     = 41,   // Prism ternary 1-bit (sign-only): 128 w/block, 18 B, 1.125 bpw
    kQ2_0     = 42,   // Prism ternary 2-bit: 128 w/block, 34 B, 2.125 bpw.
                      // block = fp16 d(=amax) + 32 B qs (2 bits/w, LSB-first,
                      // natural order); value = (code-1)*d ∈ {-d,0,+d,+2d}.
    kSTQ1_0   = 43,   // Tencent Hy4-preview scaled-ternary (hy4 llama.cpp patch
                      // ggml type 43): 256 w/block, 42 B, 1.3125 bpw. Groups of
                      // 4 weights stride-16 in 64-weight chunks; codebook-packed
                      // ternary, w = (q-1)*d. See quant_blocks.hpp.
    kCount    = 44,   // end of the dense id range

    // ---- ik_llama.cpp extended ggml types (ids 128+) ----------------------
    // These live far above kCount, so they are held in a sparse side-table
    // rather than padding the dense array with ~110 empty rows. Ids verified
    // against ik_llama.cpp ggml.h and against the GLM-5.2 GGUF's own tensor
    // offsets — see docs/glm52/KERNEL_VALIDATION.md §1.
    kQ6_0     = 133,  // 32 w/block, 26 B, 6.5 bpw:
                      // fp16 d + qh[8] (2 high bits) + qs[16] (nibbles)
    kIQ2_KT   = 153,  // 256 w/block, 68 B, 2.125 bpw + a 4-BYTE PER-ROW fp32
                      // scale. QTIP-style trellis: block = scales[4] + ql[64],
                      // each uint16 in ql seeds 8 weights through
                      // val *= 0xCBAC1FED; w = sum(6-bit lanes) - 126.

    // ---- safetensors element types (ids 200+, engine-custom) --------------
    // DeepSeek-V4.1-Flash ships as safetensors, not GGUF, and stores weights
    // and their scales as SEPARATE tensors rather than interleaved blocks. So
    // these name the element encoding only; the scale plane is a sibling tensor
    // (`<name>.scale`) and the block geometry lives in the two shapes. ggml has
    // no ids for them and never will, which is why they sit in the engine's own
    // range above ik's.
    kFP8_E4M3 = 200,  // 1 B/element. Dense weights, with an E8M0 scale per
                      // 32x32 block (verified on the shipped weights
                      // 2026-09-12: scale grid tiles the tensor exactly).
    kFP4_E2M1 = 201,  // 2 elements/byte, low nibble first along the LAST dim.
                      // Routed experts. The 16 codes decode to
                      // {0, .5, 1, 1.5, 2, 3, 4, 6} and their negations —
                      // byte-identical to this engine's MXFP4 nibble table
                      // (checked against the format spec on all 16 codes).
                      // The plane SHAPES also match DS4ExpertBank's mx_qs /
                      // mx_e, but the element ORDER inside a 32-element block
                      // does NOT: the engine's layout is gpt-oss interleaved
                      // (byte j holds elements j and j+16), this one is
                      // sequential (byte b holds 2b and 2b+1). Only elements 0
                      // and 31 coincide. ds41_expert_bank_upload permutes at
                      // upload; see docs/deepseek41/08.
    kE8M0     = 202,  // 1 B/element, value = 2^(byte - 127). Scale planes.
};

struct TypeInfo {
    DType            dtype;
    std::string_view name;        // canonical short name, e.g. "Q4_K"
    uint32_t         block_size;  // elements per block (1 for non-quantized)
    uint32_t         block_bytes; // bytes per block
    bool             is_quantized;
    // Extra bytes carried once per ROW, ahead of that row's blocks (ik's
    // `row_meta_size`). Zero for every ggml type except the KT trellis family,
    // which stores a per-row fp32 scale. bytes_for() accounts for it, and every
    // caller already asks for one row at a time, so nothing else changes.
    uint32_t         row_meta_size;
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
