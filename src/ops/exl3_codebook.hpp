// src/ops/exl3_codebook.hpp — the three EXL3 procedural codebooks (exllamav3 master bd1f519, exllamav3_ext/quant/codebook.cuh
// decode_3inst<cb>), device side. A 16-bit trellis window -> one fp16 weight (widened to float), integer math and one fp16
// rounding each:
//   cb 0 "3INST" (the original):  x = x*89226354 + 64248484; x = (x & 0x8fff8fff) ^ 0x3b603b60; w = half(lo) + half(hi)
//   cb 1 "mcg":                   x *= 0xCBAC1FED;          the same lop3 and fp16 add
//   cb 2 "mul1" (turboderp's Qwen3.8-Flash-Next quants): x *= 0x83DCD12D; s = bytesum(x) + 0x6400 (dp4a, unsigned bytes),
//                                 read as fp16 bits (1024 + bytesum); w = fma(s, half(0x1eee), half(0xc931)) in fp16
// cb 0 keeps the expression the engine shipped with (decode_cb0_dev), now one template instance of three.
// The host twin is exl3_decode_code_host (src/core/exl3_decode.cpp); ie-exl3-codebook-test [F] checks all 65536 codes x 3.
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>

namespace ie {

template <int CB>
inline float exl3_decode_dev(uint32_t code) {
    if constexpr (CB == 2) {
        const uint32_t x = code * 0x83DCD12Du;
        const uint32_t s = (x & 0xffu) + ((x >> 8) & 0xffu) + ((x >> 16) & 0xffu) + (x >> 24) + 0x6400u;
        const sycl::half h = sycl::bit_cast<sycl::half>(uint16_t(s));
        const sycl::half k_inv = sycl::bit_cast<sycl::half>(uint16_t(0x1eee));    //  0.00677 = 1/147.7
        const sycl::half k_bias = sycl::bit_cast<sycl::half>(uint16_t(0xc931));   // -10.39
        return float(sycl::fma(h, k_inv, k_bias));
    } else {
        uint32_t x;
        if constexpr (CB == 1) x = code * 0xCBAC1FEDu;
        else                   x = code * 89226354u + 64248484u;
        x = (x & 0x8fff8fffu) ^ 0x3b603b60u;
        const sycl::half lo = sycl::bit_cast<sycl::half>(uint16_t(x & 0xffffu));
        const sycl::half hi = sycl::bit_cast<sycl::half>(uint16_t(x >> 16));
        return float(sycl::half(float(lo) + float(hi)));
    }
}

}  // namespace ie
