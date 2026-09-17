// deepseek4_q8_f32_bind_test — the layout contract the DeepSeek-V4 loader relies
// on when it materializes a Q8_0 tensor for an F32 role (2026-09-11: the
// ggml-org MXFP4 file stores the hyper-connection fns and the compressor APEs
// as Q8_0; `f32vec` / `gvec` dequantize the whole tensor as ONE row of n
// elements through ref::dequant_q8_0_buffer, the bit-exact reference).
//
// Proves, host-only, no GPU, no model:
//   1. a 2-D Q8_0 tensor [K, rows] (rows of K/32 blocks, 34 B each, contiguous)
//      dequantized as one row of K*rows elements equals the per-row
//      dequantization concatenated — the assumption behind the one-row call;
//   2. the reference formula on exactly representable scales: y = d * qs.
#include "ie/dequant_ref.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

int main() {
    constexpr uint32_t K = 64, rows = 3, nb = K / 32;   // 2 blocks per row
    // fp16 bit patterns of exactly representable scales: 1.0, 0.5, -2.0, 0.25, 4.0, -0.125
    const uint16_t d_bits[rows * nb] = {0x3C00, 0x3800, 0xC000, 0x3400, 0x4400, 0xB000};
    const float    d_val[rows * nb]  = {1.0f, 0.5f, -2.0f, 0.25f, 4.0f, -0.125f};
    std::vector<uint8_t> packed(size_t(rows) * nb * 34);
    for (uint32_t b = 0; b < rows * nb; ++b) {
        uint8_t* blk = packed.data() + size_t(b) * 34;
        std::memcpy(blk, &d_bits[b], 2);
        for (uint32_t i = 0; i < 32; ++i) blk[2 + i] = uint8_t(int8_t(int32_t(i) * 7 - 111 + int32_t(b)));
    }
    // 1. one-row dequant of the whole tensor vs per-row dequant
    std::vector<float> whole(size_t(K) * rows), per_row(size_t(K) * rows);
    ie::ref::dequant_q8_0_buffer(packed.data(), size_t(K) * rows, whole.data());
    for (uint32_t r = 0; r < rows; ++r)
        ie::ref::dequant_q8_0_buffer(packed.data() + size_t(r) * nb * 34, K, per_row.data() + size_t(r) * K);
    if (std::memcmp(whole.data(), per_row.data(), whole.size() * sizeof(float)) != 0) {
        std::printf("FAIL: one-row dequant of a [%u x %u] Q8_0 tensor differs from per-row dequant\n", K, rows);
        return 1;
    }
    // 2. the formula, bit-exact on representable scales
    for (uint32_t b = 0; b < rows * nb; ++b)
        for (uint32_t i = 0; i < 32; ++i) {
            const float want = d_val[b] * float(int8_t(int32_t(i) * 7 - 111 + int32_t(b)));
            const float got  = whole[size_t(b) * 32 + i];
            if (std::memcmp(&want, &got, 4) != 0) {
                std::printf("FAIL: block %u elem %u: want %g got %g\n", b, i, double(want), double(got));
                return 1;
            }
        }
    std::printf("deepseek4_q8_f32_bind_test: OK (%u blocks, %u elements)\n", rows * nb, K * rows);
    return 0;
}
