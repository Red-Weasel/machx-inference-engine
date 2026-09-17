// src/ops/gemv_iq_raw.cpp — RAW-layout W-A16 GEMVs + B^T dequants for the
// GLM-5.3-Flash UD-Q3_K_XL expert banks (gate/up IQ3_XXS + one Q3_K layer,
// down IQ4_XS). Unlike gemv_iq3_xxs.cpp (load-time SoA repack, DeepSeek's
// host-resident bank path), these consume the UNMODIFIED packed blocks
// straight out of an expert-cache slot — the glm5next fill path stays a raw
// byte copy for every dtype.
//
// Structure is gemv_q5k.cpp's lattice verbatim: 16-lane subgroup per output
// column (32 columns/WG), activation staged in SLM, per-block scale decode,
// fp32 accumulate, one subgroup reduce. Element mappings per format:
//   IQ4_XS  lane -> (sub-block sb = lane/2, half = lane&1): 16 nibbles of
//           qs[sb*16..+16), value = d*(ls-32) * kIQ4NLValues[nib].
//   Q3_K    lane == sub-block `is` (16 sub-blocks of 16 == 16 lanes):
//           q3 = ((qs >> shift) & 3) - (hmask bit ? 0 : 4), scale6[is]-32.
//   IQ3_XXS lane -> (sb, half): 4 grid words + 2 sign selectors of the
//           sub-block's aux32 (unaligned at +66 — assembled from bytes).
// The per-element math matches ie::ref::dequant_* bit-for-bit; parity tests
// pin each kernel against the reference on random blocks.
#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"

#include <sycl/sycl.hpp>
#include "ie/kernel_profiler.hpp"

namespace ie {

namespace {
inline float dev_fp16_to_fp32(uint16_t h) {
    return float(sycl::bit_cast<sycl::half>(h));
}
// Q3_K 6-bit scale unpack (the kmask shuffle) — scales6[is] for is in [0,16).
// Byte-exact port of ie::ref::dequant_q3_K's aux[] construction.
inline int q3k_scale6(const uint8_t* scales, int is) {
    constexpr uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
    uint32_t a0, a1, t;
    __builtin_memcpy(&a0, scales + 0, 4);
    __builtin_memcpy(&a1, scales + 4, 4);
    __builtin_memcpy(&t,  scales + 8, 4);
    uint32_t aux;
    switch (is >> 2) {
        case 0: aux = (a0 & kmask2) | (((t >> 0) & kmask1) << 4); break;
        case 1: aux = (a1 & kmask2) | (((t >> 2) & kmask1) << 4); break;
        case 2: aux = ((a0 >> 4) & kmask2) | (((t >> 4) & kmask1) << 4); break;
        default: aux = ((a1 >> 4) & kmask2) | (((t >> 6) & kmask1) << 4); break;
    }
    return int(int8_t((aux >> (8 * (is & 3))) & 0xFF));
}
}  // namespace

// ---------------------------------------------------------------------------
// IQ4_XS — W4(nl)A16 GEMV, raw 136-byte blocks.
// ---------------------------------------------------------------------------
sycl::event gemv_iq4_xs(sycl::queue& q,
                        const sycl::half* A, const void* W_packed,
                        sycl::half* y,
                        uint32_t K, uint32_t N,
                        const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE = 16;
    const int N_PER_WG = 32;
    const int WG_ITEMS = N_PER_WG * SG_SIZE;

    const auto* W = static_cast<const block_iq4_xs*>(W_packed);
    const uint32_t blocks_per_col = K / 256;
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "gemv_iq4xs", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<sycl::half, 1> A_slm(sycl::range<1>(K), h);

        h.parallel_for(sycl::nd_range<1>(n_wgs * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t wgid  = uint32_t(it.get_group(0));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t n     = wgid * N_PER_WG + sg_id;

            for (uint32_t i = lid; i < K; i += WG_ITEMS) A_slm[i] = A[i];
            sycl::group_barrier(it.get_group());

            if (n >= N) return;

            const int sb   = int(lane) >> 1;      // sub-block 0..7
            const int half = int(lane) & 1;       // low/high nibble
            const int shift = half ? 4 : 0;

            float acc = 0.f;
            const block_iq4_xs* col = &W[n * blocks_per_col];
            for (uint32_t b = 0; b < blocks_per_col; ++b) {
                const block_iq4_xs& blk = col[b];
                const int ls = ((blk.scales_l[sb >> 1] >> (4 * (sb & 1))) & 0xF) |
                               (((blk.scales_h >> (2 * sb)) & 3) << 4);
                const float dl = dev_fp16_to_fp32(blk.d) * float(ls - 32);

                const sycl::vec<uint8_t, 16> qs_v =
                    *reinterpret_cast<const sycl::vec<uint8_t, 16>*>(&blk.qs[sb * 16]);
                const sycl::vec<sycl::half, 16> a_v =
                    *reinterpret_cast<const sycl::vec<sycl::half, 16>*>(
                        &A_slm[b * 256 + sb * 32 + half * 16]);

                float sum = 0.f;
                #pragma unroll
                for (int i = 0; i < 16; ++i)
                    sum += float(a_v[i]) *
                           float(kIQ4NLValues[(qs_v[i] >> shift) & 0x0F]);
                acc += dl * sum;
            }

            acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
            if (lane == 0) y[n] = sycl::half(acc);
        });
    });
}

// ---------------------------------------------------------------------------
// Q3_K — W3A16 GEMV, raw 110-byte blocks. Lane == 16-element sub-block.
// ---------------------------------------------------------------------------
sycl::event gemv_q3_K(sycl::queue& q,
                      const sycl::half* A, const void* W_packed,
                      sycl::half* y,
                      uint32_t K, uint32_t N,
                      const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE = 16;
    const int N_PER_WG = 32;
    const int WG_ITEMS = N_PER_WG * SG_SIZE;

    const auto* W = static_cast<const block_q3_K*>(W_packed);
    const uint32_t blocks_per_col = K / 256;
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "gemv_q3k", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<sycl::half, 1> A_slm(sycl::range<1>(K), h);

        h.parallel_for(sycl::nd_range<1>(n_wgs * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t wgid  = uint32_t(it.get_group(0));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;   // == sub-block index `is`
            const uint32_t n     = wgid * N_PER_WG + sg_id;

            for (uint32_t i = lid; i < K; i += WG_ITEMS) A_slm[i] = A[i];
            sycl::group_barrier(it.get_group());

            if (n >= N) return;

            const int is     = int(lane);
            const int qs_off = (is >> 3) * 32 + (is & 1) * 16;
            const int shift  = 2 * ((is >> 1) & 3);
            const int hm_off = (is & 1) * 16;
            const uint8_t mbit = uint8_t(1u << (is >> 1));
            const int p0 = (is >> 3) * 128 + ((is >> 1) & 3) * 32 + (is & 1) * 16;

            float acc = 0.f;
            const block_q3_K* col = &W[n * blocks_per_col];
            for (uint32_t b = 0; b < blocks_per_col; ++b) {
                const block_q3_K& blk = col[b];
                const float dl = dev_fp16_to_fp32(blk.d) *
                                 float(q3k_scale6(blk.scales, is) - 32);

                const sycl::vec<uint8_t, 16> qs_v =
                    *reinterpret_cast<const sycl::vec<uint8_t, 16>*>(&blk.qs[qs_off]);
                const sycl::vec<uint8_t, 16> hm_v =
                    *reinterpret_cast<const sycl::vec<uint8_t, 16>*>(&blk.hmask[hm_off]);
                const sycl::vec<sycl::half, 16> a_v =
                    *reinterpret_cast<const sycl::vec<sycl::half, 16>*>(&A_slm[b * 256 + p0]);

                float sum = 0.f;
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const int q3 = int((qs_v[i] >> shift) & 3) -
                                   ((hm_v[i] & mbit) ? 0 : 4);
                    sum += float(a_v[i]) * float(q3);
                }
                acc += dl * sum;
            }

            acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
            if (lane == 0) y[n] = sycl::half(acc);
        });
    });
}

// ---------------------------------------------------------------------------
// IQ3_XXS — W3(grid)A16 GEMV, raw 98-byte blocks. The aux32 words sit at
// byte offset 66 (2 mod 4) — assembled from bytes, never a u32 load.
// ---------------------------------------------------------------------------
sycl::event gemv_iq3_xxs_raw(sycl::queue& q,
                             const sycl::half* A, const void* W_packed,
                             sycl::half* y,
                             uint32_t K, uint32_t N,
                             const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE = 16;
    const int N_PER_WG = 32;
    const int WG_ITEMS = N_PER_WG * SG_SIZE;

    const auto* W = static_cast<const uint8_t*>(W_packed);   // 98-byte stride
    const uint32_t blocks_per_col = K / 256;
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "gemv_iq3xxs_raw", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<sycl::half, 1> A_slm(sycl::range<1>(K), h);

        h.parallel_for(sycl::nd_range<1>(n_wgs * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t wgid  = uint32_t(it.get_group(0));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t n     = wgid * N_PER_WG + sg_id;

            for (uint32_t i = lid; i < K; i += WG_ITEMS) A_slm[i] = A[i];
            sycl::group_barrier(it.get_group());

            if (n >= N) return;

            const int sb   = int(lane) >> 1;   // sub-block 0..7
            const int half = int(lane) & 1;    // elems [half*16, half*16+16)

            float acc = 0.f;
            const uint8_t* col = W + uint64_t(n) * blocks_per_col * 98;
            for (uint32_t b = 0; b < blocks_per_col; ++b) {
                const uint8_t* blk = col + uint64_t(b) * 98;
                const uint16_t d16 = uint16_t(blk[0]) | (uint16_t(blk[1]) << 8);
                const uint8_t* gq  = blk + 2;          // 64 grid indices
                const uint8_t* scs = blk + 2 + 64;     // 8x u32, unaligned
                const uint32_t aux32 = uint32_t(scs[4 * sb + 0])        |
                                       (uint32_t(scs[4 * sb + 1]) << 8)  |
                                       (uint32_t(scs[4 * sb + 2]) << 16) |
                                       (uint32_t(scs[4 * sb + 3]) << 24);
                const float db = dev_fp16_to_fp32(d16) *
                                 (0.5f + float(aux32 >> 28)) * 0.5f;

                const sycl::half* a = &A_slm[b * 256 + sb * 32 + half * 16];
                float sum = 0.f;
                #pragma unroll
                for (int t = 0; t < 4; ++t) {
                    const uint32_t grid =
                        kIQ3XXSGrid[gq[sb * 8 + half * 4 + t]];
                    const int l_sel = half * 2 + (t >> 1);
                    const uint8_t signs = kSignsIQ2XS[(aux32 >> (7 * l_sel)) & 127];
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const float mag = float((grid >> (8 * j)) & 0xFF);
                        const float s =
                            (signs & kMaskIQ2XS[(t & 1) * 4 + j]) ? -1.f : 1.f;
                        sum += float(a[t * 4 + j]) * mag * s;
                    }
                }
                acc += db * sum;
            }

            acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
            if (lane == 0) y[n] = sycl::half(acc);
        });
    });
}

// ---------------------------------------------------------------------------
// B^T dequants (prefill emat chain): raw [N, K] packed -> fp16 [K, N].
// One work-item per (block, column), dequant_q5_K_to_Bt's shape verbatim.
// ---------------------------------------------------------------------------
sycl::event dequant_iq4_xs_to_Bt(sycl::queue& q, const void* packed_in,
                                 sycl::half* out, uint32_t K, uint32_t N,
                                 const std::vector<sycl::event>& deps) {
    const uint32_t blocks_per_col = K / 256;
    constexpr uint32_t WG = 64;
    auto* W = static_cast<const block_iq4_xs*>(packed_in);

    return ie::ps(q, "dequant_iq4xs_bt", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<2>({blocks_per_col, N}, {1, WG}),
                       [=](sycl::nd_item<2> it) {
            const uint32_t b = uint32_t(it.get_group(0));
            const uint32_t n = uint32_t(it.get_global_id(1));
            const block_iq4_xs& blk = W[uint64_t(n) * blocks_per_col + b];
            sycl::half* dst = out + uint64_t(b) * 256 * N + n;

            const float d = dev_fp16_to_fp32(blk.d);
            #pragma unroll
            for (int ib = 0; ib < 8; ++ib) {
                const int ls = ((blk.scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) |
                               (((blk.scales_h >> (2 * ib)) & 3) << 4);
                const float dl = d * float(ls - 32);
                for (int j = 0; j < 16; ++j) {
                    const uint8_t byte = blk.qs[ib * 16 + j];
                    dst[uint64_t(ib * 32 + j)      * N] =
                        sycl::half(dl * float(kIQ4NLValues[byte & 0x0F]));
                    dst[uint64_t(ib * 32 + j + 16) * N] =
                        sycl::half(dl * float(kIQ4NLValues[byte >> 4]));
                }
            }
        });
    });
}

sycl::event dequant_q3_K_to_Bt(sycl::queue& q, const void* packed_in,
                               sycl::half* out, uint32_t K, uint32_t N,
                               const std::vector<sycl::event>& deps) {
    const uint32_t blocks_per_col = K / 256;
    constexpr uint32_t WG = 64;
    auto* W = static_cast<const block_q3_K*>(packed_in);

    return ie::ps(q, "dequant_q3k_bt", [=](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<2>({blocks_per_col, N}, {1, WG}),
                       [=](sycl::nd_item<2> it) {
            const uint32_t b = uint32_t(it.get_group(0));
            const uint32_t n = uint32_t(it.get_global_id(1));
            const block_q3_K& blk = W[uint64_t(n) * blocks_per_col + b];
            sycl::half* dst = out + uint64_t(b) * 256 * N + n;

            const float d_all = dev_fp16_to_fp32(blk.d);
            for (int is = 0; is < 16; ++is) {
                const float dl = d_all * float(q3k_scale6(blk.scales, is) - 32);
                const int qs_off = (is >> 3) * 32 + (is & 1) * 16;
                const int shift  = 2 * ((is >> 1) & 3);
                const int hm_off = (is & 1) * 16;
                const uint8_t mbit = uint8_t(1u << (is >> 1));
                const int p0 = (is >> 3) * 128 + ((is >> 1) & 3) * 32 + (is & 1) * 16;
                for (int i = 0; i < 16; ++i) {
                    const int q3 = int((blk.qs[qs_off + i] >> shift) & 3) -
                                   ((blk.hmask[hm_off + i] & mbit) ? 0 : 4);
                    dst[uint64_t(p0 + i) * N] = sycl::half(dl * float(q3));
                }
            }
        });
    });
}

sycl::event dequant_iq3_xxs_to_Bt(sycl::queue& q, const void* packed_in,
                                  sycl::half* out, uint32_t K, uint32_t N,
                                  const std::vector<sycl::event>& deps) {
    const uint32_t blocks_per_col = K / 256;
    constexpr uint32_t WG = 64;
    auto* W = static_cast<const uint8_t*>(packed_in);   // 98-byte stride

    return ie::ps(q, "dequant_iq3xxs_bt", [=](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<2>({blocks_per_col, N}, {1, WG}),
                       [=](sycl::nd_item<2> it) {
            const uint32_t b = uint32_t(it.get_group(0));
            const uint32_t n = uint32_t(it.get_global_id(1));
            const uint8_t* blk =
                W + (uint64_t(n) * blocks_per_col + b) * 98;
            sycl::half* dst = out + uint64_t(b) * 256 * N + n;

            const uint16_t d16 = uint16_t(blk[0]) | (uint16_t(blk[1]) << 8);
            const float d = dev_fp16_to_fp32(d16);
            const uint8_t* gq  = blk + 2;
            const uint8_t* scs = blk + 2 + 64;
            for (int sb = 0; sb < 8; ++sb) {
                const uint32_t aux32 = uint32_t(scs[4 * sb + 0])        |
                                       (uint32_t(scs[4 * sb + 1]) << 8)  |
                                       (uint32_t(scs[4 * sb + 2]) << 16) |
                                       (uint32_t(scs[4 * sb + 3]) << 24);
                const float db = d * (0.5f + float(aux32 >> 28)) * 0.5f;
                for (int l = 0; l < 4; ++l) {
                    const uint8_t signs = kSignsIQ2XS[(aux32 >> (7 * l)) & 127];
                    const uint32_t g1 = kIQ3XXSGrid[gq[sb * 8 + 2 * l + 0]];
                    const uint32_t g2 = kIQ3XXSGrid[gq[sb * 8 + 2 * l + 1]];
                    for (int j = 0; j < 4; ++j) {
                        const int p = sb * 32 + l * 8;
                        dst[uint64_t(p + j)     * N] = sycl::half(
                            db * float((g1 >> (8 * j)) & 0xFF) *
                            ((signs & kMaskIQ2XS[j])     ? -1.f : 1.f));
                        dst[uint64_t(p + j + 4) * N] = sycl::half(
                            db * float((g2 >> (8 * j)) & 0xFF) *
                            ((signs & kMaskIQ2XS[j + 4]) ? -1.f : 1.f));
                    }
                }
            }
        });
    });
}

// ---------------------------------------------------------------------------
// IQ2_XXS — W2(grid)A16 GEMV, raw 66-byte blocks (Hy4-preview expert gate/up,
// 48 layers). Same lattice as gemv_iq3_xxs_raw: sub-block sb = lane/2, half
// picks elements [half*16, half*16+16) of the 32. Per sub-block the 4 u16 at
// qs[4*sb..4*sb+4) split into aux32[0] (four grid-index bytes, u64 grid — 8
// magnitudes per entry) and aux32[1] (four 7-bit sign selectors + 4-bit scale):
//   db = d * (0.5 + (aux32[1] >> 28)) * 0.25
// The 66-byte stride leaves qs 2 mod 4 — words assembled from bytes.
// Matches ie::ref::dequant_iq2_xxs bit-for-bit per element.
// ---------------------------------------------------------------------------
sycl::event gemv_iq2_xxs_raw(sycl::queue& q,
                             const sycl::half* A, const void* W_packed,
                             sycl::half* y,
                             uint32_t K, uint32_t N,
                             const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE = 16;
    const int N_PER_WG = 32;
    const int WG_ITEMS = N_PER_WG * SG_SIZE;

    const auto* W = static_cast<const uint8_t*>(W_packed);   // 66-byte stride
    const uint32_t blocks_per_col = K / 256;
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "gemv_iq2xxs_raw", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<sycl::half, 1> A_slm(sycl::range<1>(K), h);

        h.parallel_for(sycl::nd_range<1>(n_wgs * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t wgid  = uint32_t(it.get_group(0));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t n     = wgid * N_PER_WG + sg_id;

            for (uint32_t i = lid; i < K; i += WG_ITEMS) A_slm[i] = A[i];
            sycl::group_barrier(it.get_group());

            if (n >= N) return;

            const int sb   = int(lane) >> 1;   // sub-block 0..7
            const int half = int(lane) & 1;    // elems [half*16, half*16+16)

            float acc = 0.f;
            const uint8_t* col = W + uint64_t(n) * blocks_per_col * 66;
            for (uint32_t b = 0; b < blocks_per_col; ++b) {
                const uint8_t* blk = col + uint64_t(b) * 66;
                const uint16_t d16 = uint16_t(blk[0]) | (uint16_t(blk[1]) << 8);
                const uint8_t* qs  = blk + 2 + uint64_t(sb) * 8;  // this sub-block's 4 u16
                const uint32_t aux32 = uint32_t(qs[4])         |
                                       (uint32_t(qs[5]) << 8)  |
                                       (uint32_t(qs[6]) << 16) |
                                       (uint32_t(qs[7]) << 24);
                const float db = dev_fp16_to_fp32(d16) *
                                 (0.5f + float(aux32 >> 28)) * 0.25f;

                const sycl::half* a = &A_slm[b * 256 + sb * 32 + half * 16];
                float sum = 0.f;
                #pragma unroll
                for (int t = 0; t < 2; ++t) {                   // grids l = half*2 + t
                    const int l = half * 2 + t;
                    const uint64_t grid = kIQ2XXSGrid[qs[l]];
                    const uint8_t signs = kSignsIQ2XS[(aux32 >> (7 * l)) & 127];
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const float mag = float((grid >> (8 * j)) & 0xFF);
                        const float s = (signs & kMaskIQ2XS[j]) ? -1.f : 1.f;
                        sum += float(a[t * 8 + j]) * mag * s;
                    }
                }
                acc += db * sum;
            }

            acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
            if (lane == 0) y[n] = sycl::half(acc);
        });
    });
}

// ---------------------------------------------------------------------------
// STQ1_0 — scaled-ternary W1.3A16 GEMV, raw 42-byte blocks (Hy4-preview
// expert gate/up, 29 layers). Weight groups of 4 are STRIDE-16 inside each
// 64-element chunk (see quant_blocks.hpp), so the natural lane map is
// lane -> (chunk c = lane/4, lane p = lane&3): the lane owns the 16
// CONTIGUOUS activation elements [c*64 + p*16, +16), reading group gloc's
// code/sign once per element. w = ((codebook >> 2p) & 3) - 1 in {-1,0,+1},
// scaled by the block's single d. d sits at byte 40 (2 mod 4) — assembled
// from bytes. Matches ie::ref::dequant_stq1_0 bit-for-bit per element.
// ---------------------------------------------------------------------------
sycl::event gemv_stq1_0(sycl::queue& q,
                        const sycl::half* A, const void* W_packed,
                        sycl::half* y,
                        uint32_t K, uint32_t N,
                        const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE = 16;
    const int N_PER_WG = 32;
    const int WG_ITEMS = N_PER_WG * SG_SIZE;

    const auto* W = static_cast<const uint8_t*>(W_packed);   // 42-byte stride
    const uint32_t blocks_per_col = K / 256;
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "gemv_stq1_0", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<sycl::half, 1> A_slm(sycl::range<1>(K), h);

        h.parallel_for(sycl::nd_range<1>(n_wgs * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t wgid  = uint32_t(it.get_group(0));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t n     = wgid * N_PER_WG + sg_id;

            for (uint32_t i = lid; i < K; i += WG_ITEMS) A_slm[i] = A[i];
            sycl::group_barrier(it.get_group());

            if (n >= N) return;

            const int c = int(lane) >> 2;        // chunk 0..3
            const int p = int(lane) & 3;         // lane inside the group
            const int shift = 2 * p;

            float acc = 0.f;
            const uint8_t* col = W + uint64_t(n) * blocks_per_col * 42;
            for (uint32_t b = 0; b < blocks_per_col; ++b) {
                const uint8_t* blk  = col + uint64_t(b) * 42;
                const uint8_t* qs   = blk + uint64_t(c) * 8;    // 16 group codes
                const uint16_t sbits = uint16_t(blk[32 + 2 * c]) |
                                       (uint16_t(blk[32 + 2 * c + 1]) << 8);
                const uint16_t d16  = uint16_t(blk[40]) | (uint16_t(blk[41]) << 8);
                const float d = dev_fp16_to_fp32(d16);

                const sycl::half* a = &A_slm[b * 256 + c * 64 + p * 16];
                float sum = 0.f;
                #pragma unroll
                for (int g = 0; g < 16; ++g) {                  // gloc
                    const uint8_t code  = uint8_t((qs[g >> 1] >> (4 * (g & 1))) & 0x0F);
                    const uint8_t sign  = uint8_t((sbits >> g) & 1);
                    const uint8_t qpack = kSTQ1Codebook[(uint32_t(sign) << 4) | code];
                    const int qv = int((qpack >> shift) & 0x3) - 1;
                    sum += float(a[g]) * float(qv);
                }
                acc += d * sum;
            }

            acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
            if (lane == 0) y[n] = sycl::half(acc);
        });
    });
}

// Hy4-preview expert dtypes: B^T dequants for the prefill emat chain.
// Same (block, column) grid and fp32 math as the refs; fp16 store.
sycl::event dequant_iq2_xxs_to_Bt(sycl::queue& q, const void* packed_in,
                                  sycl::half* out, uint32_t K, uint32_t N,
                                  const std::vector<sycl::event>& deps) {
    const uint32_t blocks_per_col = K / 256;
    constexpr uint32_t WG = 64;
    auto* W = static_cast<const uint8_t*>(packed_in);   // 66-byte stride

    return ie::ps(q, "dequant_iq2xxs_bt", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<2>({blocks_per_col, N}, {1, WG}),
                       [=](sycl::nd_item<2> it) {
            const uint32_t b = uint32_t(it.get_group(0));
            const uint32_t n = uint32_t(it.get_global_id(1));
            const uint8_t* blk = W + (uint64_t(n) * blocks_per_col + b) * 66;
            sycl::half* dst = out + uint64_t(b) * 256 * N + n;

            const uint16_t d16 = uint16_t(blk[0]) | (uint16_t(blk[1]) << 8);
            const float d = dev_fp16_to_fp32(d16);
            for (int sb = 0; sb < 8; ++sb) {
                const uint8_t* qs = blk + 2 + uint64_t(sb) * 8;
                const uint32_t aux32 = uint32_t(qs[4])         |
                                       (uint32_t(qs[5]) << 8)  |
                                       (uint32_t(qs[6]) << 16) |
                                       (uint32_t(qs[7]) << 24);
                const float db = d * (0.5f + float(aux32 >> 28)) * 0.25f;
                for (int l = 0; l < 4; ++l) {
                    const uint64_t grid = kIQ2XXSGrid[qs[l]];
                    const uint8_t signs = kSignsIQ2XS[(aux32 >> (7 * l)) & 127];
                    for (int j = 0; j < 8; ++j) {
                        const float mag = float((grid >> (8 * j)) & 0xFF);
                        const float s = (signs & kMaskIQ2XS[j]) ? -1.f : 1.f;
                        dst[uint64_t(sb * 32 + l * 8 + j) * N] = sycl::half(db * mag * s);
                    }
                }
            }
        });
    });
}

sycl::event dequant_stq1_0_to_Bt(sycl::queue& q, const void* packed_in,
                                 sycl::half* out, uint32_t K, uint32_t N,
                                 const std::vector<sycl::event>& deps) {
    const uint32_t blocks_per_col = K / 256;
    constexpr uint32_t WG = 64;
    auto* W = static_cast<const uint8_t*>(packed_in);   // 42-byte stride

    return ie::ps(q, "dequant_stq1_bt", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<2>({blocks_per_col, N}, {1, WG}),
                       [=](sycl::nd_item<2> it) {
            const uint32_t b = uint32_t(it.get_group(0));
            const uint32_t n = uint32_t(it.get_global_id(1));
            const uint8_t* blk = W + (uint64_t(n) * blocks_per_col + b) * 42;
            sycl::half* dst = out + uint64_t(b) * 256 * N + n;

            const uint16_t d16 = uint16_t(blk[40]) | (uint16_t(blk[41]) << 8);
            const float d = dev_fp16_to_fp32(d16);
            for (int g = 0; g < 64; ++g) {
                const uint8_t code  = uint8_t((blk[g >> 1] >> (4 * (g & 1))) & 0x0F);
                const uint8_t sign  = uint8_t((blk[32 + (g >> 3)] >> (g % 8)) & 0x01);
                const uint8_t qpack = kSTQ1Codebook[(uint32_t(sign) << 4) | code];
                const int chunk = g / 16;
                const int gloc  = g % 16;
                for (int p = 0; p < 4; ++p) {
                    const int qv = (qpack >> (2 * p)) & 0x3;
                    dst[uint64_t(chunk * 64 + gloc + p * 16) * N] =
                        sycl::half(float(qv - 1) * d);
                }
            }
        });
    });
}

}  // namespace ie
