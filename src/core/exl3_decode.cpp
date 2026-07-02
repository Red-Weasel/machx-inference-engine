// src/core/exl3_decode.cpp — EXL3 host reference decode (cb=0).
// Ported bit-for-bit from exllamav3 (file:line in docs/exl3_format_notes.md):
//   - bit extraction: exl3_dq.cuh:15-31  (tail-biting, two u32 funnel-shift)
//   - codebook cb=0:  codebook.cuh:25-35 (x*mul+add ; (x&m)^c ; f16(lo)+f16(hi))
//   - tile permute:   quantize.py:21-48  (lane order → row-major)

#include "ie/exl3.hpp"

#include <sycl/sycl.hpp>

#include <cstring>

namespace ie {
namespace {

// codebook.cuh:25-35, cb=0. The lop3 lut 0x6a = (a & b) ^ c (hand-verified truth
// table 0b01101010). Sum the two fp16 halves of the hashed word. Matches the numpy
// oracle: widen halves to fp32, sum, round to fp16.
inline float decode_cb0(uint32_t code) {
    uint32_t x = code * 89226354u + 64248484u;          // MCG hash (wraps mod 2^32)
    x = (x & 0x8fff8fffu) ^ 0x3b603b60u;                // lop3 0x6a
    uint16_t lob = uint16_t(x & 0xffffu);
    uint16_t hib = uint16_t(x >> 16);
    sycl::half lo, hi;
    std::memcpy(&lo, &lob, sizeof lo);
    std::memcpy(&hi, &hib, sizeof hi);
    sycl::half h = sycl::half(float(lo) + float(hi));   // fp32 sum → fp16 round
    return float(h);
}

// quantize.py:21-48 — tensor_core_perm: lane index t (0..255) → row-major (r*16+c).
void build_perm(int* perm) {
    for (int t = 0; t < 32; ++t) {
        const int r0 = (t % 4) * 2, r1 = r0 + 1, r2 = r0 + 8, r3 = r0 + 9;
        const int c0 = t / 4,       c1 = c0 + 8;
        perm[t * 8 + 0] = r0 * 16 + c0;
        perm[t * 8 + 1] = r1 * 16 + c0;
        perm[t * 8 + 2] = r2 * 16 + c0;
        perm[t * 8 + 3] = r3 * 16 + c0;
        perm[t * 8 + 4] = r0 * 16 + c1;
        perm[t * 8 + 5] = r1 * 16 + c1;
        perm[t * 8 + 6] = r2 * 16 + c1;
        perm[t * 8 + 7] = r3 * 16 + c1;
    }
}

}  // namespace

std::string exl3_decode_host(const Exl3Tensor& t, float* out) {
    if (t.cb != 0)                 return "exl3_decode_host: only cb=0 supported";
    if (t.bits < 1 || t.bits > 8)  return "exl3_decode_host: bits out of range";
    if (!out)                      return "exl3_decode_host: null out";
    const uint32_t bits = t.bits, TK = t.tile_k, TN = t.tile_n, N = t.N;
    const uint32_t w16 = 16u * bits;       // uint16 words per tile
    const uint32_t psz = bits * 256u / 32; // uint32 words per tile (tail-biting modulus)
    if (size_t(TK) * TN * w16 != t.trellis.size())
        return "exl3_decode_host: trellis size mismatch";
    if (TK * 16u != t.K || TN * 16u != t.N)
        return "exl3_decode_host: K/N vs tile dims mismatch";

    int perm[256];
    build_perm(perm);
    std::vector<uint32_t> words(psz);

    for (uint32_t ki = 0; ki < TK; ++ki) {
        for (uint32_t ni = 0; ni < TN; ++ni) {
            const uint16_t* tile = &t.trellis[(size_t(ki) * TN + ni) * w16];
            for (uint32_t j = 0; j < psz; ++j)             // pack u16 pairs → u32 (LE)
                words[j] = uint32_t(tile[2 * j]) | (uint32_t(tile[2 * j + 1]) << 16);
            for (uint32_t tt = 0; tt < 256; ++tt) {
                const int b0 = int(tt * bits + bits - 16 + 256 * bits);
                const int b1 = b0 + 16;
                const int i0 = b0 / 32;
                const int i1 = (b1 - 1) / 32;
                const int s0 = (i1 + 1) * 32 - b1;
                const uint32_t a = words[i0 % psz];
                const uint32_t b = words[i1 % psz];
                const uint64_t merged = (uint64_t(a) << 32) | b;   // fshift: (a<<32)|b
                const uint32_t code = uint32_t(merged >> s0) & 0xffffu;
                const float val = decode_cb0(code);
                const int p = perm[tt];
                out[size_t(ki * 16 + p / 16) * N + (ni * 16 + p % 16)] = val;
            }
        }
    }
    return {};
}

}  // namespace ie
