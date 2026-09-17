// include/ie/fp8.hpp — OCP FP8/E8M0 element decoders for safetensors checkpoints.
//
// DeepSeek-V4.1-Flash stores dense weights as E4M3 bytes with a sibling E8M0 scale plane, so
// these two conversions sit under every dense tensor the model loads. Both are exact: E4M3 has
// 8 bits of information and fp32 has 24, and E8M0 is a pure power of two.
//
// Lives in a header because three call sites need the identical decode — the host probe, the
// unit test that checks all 256 codes against torch, and the device upload path — and a
// divergence between them would be a silent numerical bug rather than a link error.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

namespace ie {

// E4M3 in the OCP "fn" flavour: 1 sign, 4 exponent (bias 7), 3 mantissa. No infinities; the
// single NaN encoding is exponent all-ones with mantissa all-ones (0x7F / 0xFF). Exponent 0 is
// subnormal at 2^-6 * m/8. Verified against torch.float8_e4m3fn on all 256 codes.
inline float e4m3_to_f32(uint8_t b) {
    const uint32_t s = (b >> 7) & 1u, e = (b >> 3) & 0xFu, m = b & 0x7u;
    float v;
    if (e == 0) {
        v = float(m) * (1.0f / 8.0f) * 0.015625f;          // 2^-6
    } else if (e == 15 && m == 7) {
        v = std::nanf("");
    } else {
        const int ex = int(e) - 7;
        float scale = 1.0f;
        if (ex > 0) for (int i = 0; i < ex; ++i)  scale *= 2.0f;
        else        for (int i = 0; i < -ex; ++i) scale *= 0.5f;
        v = (1.0f + float(m) * (1.0f / 8.0f)) * scale;
    }
    // Negating through the sign bit rather than `-v` so that code 0x80 yields -0.0 exactly,
    // which the fp32 image must carry: a dequantised weight of -0.0 is not +0.0 downstream.
    return s ? -v : v;
}

// E8M0: the byte IS the exponent, value = 2^(b-127); 255 is NaN. b == 0 gives 2^-127, which is
// SUBNORMAL in fp32, so it is built as a bit pattern rather than by halving 2^-126: icpx
// defaults to -ffp-model=fast, which flushes denormal results to zero, and the arithmetic form
// silently returned 0.0 for the smallest scale. Constructing the bits sidesteps the FTZ
// entirely. (Found by tests/unit/fp8_decode_test.cpp; a reference check of the same function
// compiled at -O0 with gcc did NOT reproduce it, because gcc keeps denormals by default.)
// Verified against torch.float8_e8m0fnu on all 256 codes.
inline float e8m0_to_f32(uint8_t b) {
    if (b == 255) return std::nanf("");
    const int ex = int(b) - 127;
    uint32_t bits;
    if (ex >= -126)      bits = uint32_t(ex + 127) << 23;   // normal
    else if (ex >= -149) bits = uint32_t(1) << (ex + 149);  // subnormal: 2^-149 is bit 0
    else                 bits = 0;                          // underflows fp32 entirely
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

}  // namespace ie
