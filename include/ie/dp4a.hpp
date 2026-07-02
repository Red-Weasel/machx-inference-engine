// include/ie/dp4a.hpp — packed int8 dot-product helper.
//
// sycl/ext/oneapi/dot_product.hpp defines its functions without `inline`
// (multiple-definition link errors when included from >1 TU), so we carry
// our own.  The plain byte arithmetic below is the exact pattern IGC
// recognizes and lowers to the hardware integer-dot (idp4a) on Xe2 —
// verified numerically on BMG-G31 (2026-06-09 smoke test).

#pragma once

#include <cstdint>

namespace ie {

// (u8x4 · s8x4) + c — the Q4_K/Q6_K-quant × Q8_1-activation form.
inline int32_t dp4a_us(uint32_t pa, int32_t pb, int32_t c) {
    const uint32_t b = uint32_t(pb);
    return int32_t(pa & 0xFFu)         * int32_t(int8_t(b & 0xFFu)) +
           int32_t((pa >> 8) & 0xFFu)  * int32_t(int8_t((b >> 8) & 0xFFu)) +
           int32_t((pa >> 16) & 0xFFu) * int32_t(int8_t((b >> 16) & 0xFFu)) +
           int32_t((pa >> 24) & 0xFFu) * int32_t(int8_t((b >> 24) & 0xFFu)) + c;
}

// (s8x4 · s8x4) + c — the Q8_0-weight × Q8_1-activation form (both signed).
inline int32_t dp4a_ss(int32_t pa, int32_t pb, int32_t c) {
    const uint32_t a = uint32_t(pa), b = uint32_t(pb);
    return int32_t(int8_t(a & 0xFFu))         * int32_t(int8_t(b & 0xFFu)) +
           int32_t(int8_t((a >> 8) & 0xFFu))  * int32_t(int8_t((b >> 8) & 0xFFu)) +
           int32_t(int8_t((a >> 16) & 0xFFu)) * int32_t(int8_t((b >> 16) & 0xFFu)) +
           int32_t(int8_t((a >> 24) & 0xFFu)) * int32_t(int8_t((b >> 24) & 0xFFu)) + c;
}

}  // namespace ie
