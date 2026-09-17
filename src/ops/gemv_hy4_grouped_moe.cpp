// src/ops/gemv_hy4_grouped_moe.cpp — grouped raw-slot MoE decode GEMVs for
// the Hy4-preview expert dtypes (gate/up STQ1_0 or IQ2_XXS, down IQ3_XXS or
// IQ4_XS), the gemv_qk_grouped_moe.cpp recipe applied to gemv_iq_raw.cpp's
// solo kernels.
//
// One launch covers all P (<= 8) routed experts of a layer at T==1: profiled
// 2026-09-01, the per-expert chain cost ~1,800 launches/token at ~36 GB/s
// effective (launch/occupancy-bound, 40% of decode GPU time). Per (expert,
// column) the SOLO kernel's lane lattice and fp32 fold run VERBATIM
// (gemv_stq1_0 / gemv_iq2_xxs_raw / gemv_iq3_xxs_raw / gemv_iq4_xs), so
// yg/yu/yd are bit-identical to the sequential per-expert path; the ordered
// reduce (moe_reduce_waccum) is shared with the Q4/Q5 grouped path.
//
// Inner-P layout: a WG owns a column tile, loads the activation into SLM once
// (gate/up: the single T=1 row; down: reloads per expert's own yg row) and
// walks the P experts. Slot ids ride by value (std::array<int32_t, 8>).
#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"

#include <sycl/sycl.hpp>
#include <array>
#include "ie/kernel_profiler.hpp"

namespace ie {
namespace {
inline float dev_fp16_to_fp32(uint16_t h) {
    return float(sycl::bit_cast<sycl::half>(h));
}
using Slots8 = std::array<int32_t, 8>;
constexpr int kSG = 16;
constexpr int kNPerWG = 32;
constexpr int kWGItems = kNPerWG * kSG;   // 512
}  // namespace

// ---------------------------------------------------------------------------
// STQ1_0 gate+up, grouped. Lane -> (chunk c = lane/4, group lane p = lane&3),
// 16 contiguous activations per lane per 256-block — gemv_stq1_0 verbatim,
// both matrices folded off one activation read.
// ---------------------------------------------------------------------------
sycl::event moe_gemv_stq1_0_gu_grouped(sycl::queue& q, const sycl::half* A,
                                       const uint8_t* slot_base, uint64_t slot_bytes,
                                       uint64_t gate_off, uint64_t up_off,
                                       const int32_t* slots, uint32_t P,
                                       sycl::half* yg, sycl::half* yu,
                                       uint32_t K, uint32_t N,
                                       const std::vector<sycl::event>& deps) {
    const uint32_t blocks_per_col = K / 256;
    const uint32_t n_wgs = (N + kNPerWG - 1) / kNPerWG;
    Slots8 sl{};
    for (uint32_t p = 0; p < P; ++p) sl[p] = slots[p];

    return ie::ps(q, "hy4_moe_gu_stq1", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<sycl::half, 1> A_slm(sycl::range<1>(K), h);
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * kWGItems, kWGItems),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t wgid  = uint32_t(it.get_group(0));
            const uint32_t sg_id = lid / kSG;
            const uint32_t lane  = lid % kSG;
            const uint32_t n     = wgid * kNPerWG + sg_id;

            for (uint32_t i = lid; i < K; i += kWGItems) A_slm[i] = A[i];
            sycl::group_barrier(it.get_group());
            if (n >= N) return;

            const int c = int(lane) >> 2;
            const int pl = int(lane) & 3;
            const int shift = 2 * pl;

            for (uint32_t p = 0; p < P; ++p) {
                const uint8_t* slot = slot_base + uint64_t(sl[p]) * slot_bytes;
                const uint8_t* gcol = slot + gate_off + uint64_t(n) * blocks_per_col * 42;
                const uint8_t* ucol = slot + up_off   + uint64_t(n) * blocks_per_col * 42;
                float gacc = 0.f, uacc = 0.f;
                for (uint32_t b = 0; b < blocks_per_col; ++b) {
                    const uint8_t* gb = gcol + uint64_t(b) * 42;
                    const uint8_t* ub = ucol + uint64_t(b) * 42;
                    const uint8_t* gqs = gb + uint64_t(c) * 8;
                    const uint8_t* uqs = ub + uint64_t(c) * 8;
                    const uint16_t gsb = uint16_t(gb[32 + 2 * c]) | (uint16_t(gb[32 + 2 * c + 1]) << 8);
                    const uint16_t usb = uint16_t(ub[32 + 2 * c]) | (uint16_t(ub[32 + 2 * c + 1]) << 8);
                    const float gd = dev_fp16_to_fp32(uint16_t(gb[40]) | (uint16_t(gb[41]) << 8));
                    const float ud = dev_fp16_to_fp32(uint16_t(ub[40]) | (uint16_t(ub[41]) << 8));
                    const sycl::half* a = &A_slm[b * 256 + c * 64 + pl * 16];
                    float gsum = 0.f, usum = 0.f;
                    #pragma unroll
                    for (int g = 0; g < 16; ++g) {
                        const float av = float(a[g]);
                        const uint8_t gcode = uint8_t((gqs[g >> 1] >> (4 * (g & 1))) & 0x0F);
                        const uint8_t ucode = uint8_t((uqs[g >> 1] >> (4 * (g & 1))) & 0x0F);
                        const uint8_t gsgn = uint8_t((gsb >> g) & 1);
                        const uint8_t usgn = uint8_t((usb >> g) & 1);
                        const int gq = int((kSTQ1Codebook[(uint32_t(gsgn) << 4) | gcode] >> shift) & 0x3) - 1;
                        const int uq = int((kSTQ1Codebook[(uint32_t(usgn) << 4) | ucode] >> shift) & 0x3) - 1;
                        gsum += av * float(gq);
                        usum += av * float(uq);
                    }
                    gacc += gd * gsum;
                    uacc += ud * usum;
                }
                gacc = sycl::reduce_over_group(it.get_sub_group(), gacc, sycl::plus<float>());
                uacc = sycl::reduce_over_group(it.get_sub_group(), uacc, sycl::plus<float>());
                if (lane == 0) {
                    yg[uint64_t(p) * N + n] = sycl::half(gacc);
                    yu[uint64_t(p) * N + n] = sycl::half(uacc);
                }
            }
        });
    });
}

// ---------------------------------------------------------------------------
// IQ2_XXS gate+up, grouped. Lane -> (sub-block sb = lane/2, half): 16 elems
// of the 32; two u64 grid entries per lane — gemv_iq2_xxs_raw verbatim.
// ---------------------------------------------------------------------------
sycl::event moe_gemv_iq2_xxs_gu_grouped(sycl::queue& q, const sycl::half* A,
                                        const uint8_t* slot_base, uint64_t slot_bytes,
                                        uint64_t gate_off, uint64_t up_off,
                                        const int32_t* slots, uint32_t P,
                                        sycl::half* yg, sycl::half* yu,
                                        uint32_t K, uint32_t N,
                                        const std::vector<sycl::event>& deps) {
    const uint32_t blocks_per_col = K / 256;
    const uint32_t n_wgs = (N + kNPerWG - 1) / kNPerWG;
    Slots8 sl{};
    for (uint32_t p = 0; p < P; ++p) sl[p] = slots[p];

    return ie::ps(q, "hy4_moe_gu_iq2", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<sycl::half, 1> A_slm(sycl::range<1>(K), h);
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * kWGItems, kWGItems),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t wgid  = uint32_t(it.get_group(0));
            const uint32_t sg_id = lid / kSG;
            const uint32_t lane  = lid % kSG;
            const uint32_t n     = wgid * kNPerWG + sg_id;

            for (uint32_t i = lid; i < K; i += kWGItems) A_slm[i] = A[i];
            sycl::group_barrier(it.get_group());
            if (n >= N) return;

            const int sb   = int(lane) >> 1;
            const int half = int(lane) & 1;

            auto fold = [&](const uint8_t* col, float& acc) {
                for (uint32_t b = 0; b < blocks_per_col; ++b) {
                    const uint8_t* blk = col + uint64_t(b) * 66;
                    const uint16_t d16 = uint16_t(blk[0]) | (uint16_t(blk[1]) << 8);
                    const uint8_t* qs  = blk + 2 + uint64_t(sb) * 8;
                    const uint32_t aux32 = uint32_t(qs[4])         |
                                           (uint32_t(qs[5]) << 8)  |
                                           (uint32_t(qs[6]) << 16) |
                                           (uint32_t(qs[7]) << 24);
                    const float db = dev_fp16_to_fp32(d16) *
                                     (0.5f + float(aux32 >> 28)) * 0.25f;
                    const sycl::half* a = &A_slm[b * 256 + sb * 32 + half * 16];
                    float sum = 0.f;
                    #pragma unroll
                    for (int t = 0; t < 2; ++t) {
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
            };

            for (uint32_t p = 0; p < P; ++p) {
                const uint8_t* slot = slot_base + uint64_t(sl[p]) * slot_bytes;
                float gacc = 0.f, uacc = 0.f;
                fold(slot + gate_off + uint64_t(n) * blocks_per_col * 66, gacc);
                fold(slot + up_off   + uint64_t(n) * blocks_per_col * 66, uacc);
                gacc = sycl::reduce_over_group(it.get_sub_group(), gacc, sycl::plus<float>());
                uacc = sycl::reduce_over_group(it.get_sub_group(), uacc, sycl::plus<float>());
                if (lane == 0) {
                    yg[uint64_t(p) * N + n] = sycl::half(gacc);
                    yu[uint64_t(p) * N + n] = sycl::half(uacc);
                }
            }
        });
    });
}

// ---------------------------------------------------------------------------
// IQ3_XXS down, grouped. A is per-expert (P rows of K); each p reloads SLM.
// Lane -> (sb, half), 4 grid words + 2 sign selectors — gemv_iq3_xxs_raw
// verbatim (aux32 assembled from bytes: 98-byte stride).
// ---------------------------------------------------------------------------
sycl::event moe_gemv_iq3_xxs_down_grouped(sycl::queue& q, const sycl::half* A,
                                          const uint8_t* slot_base, uint64_t slot_bytes,
                                          uint64_t down_off,
                                          const int32_t* slots, uint32_t P,
                                          sycl::half* y, uint32_t K, uint32_t N,
                                          const std::vector<sycl::event>& deps) {
    const uint32_t blocks_per_col = K / 256;
    const uint32_t n_wgs = (N + kNPerWG - 1) / kNPerWG;
    Slots8 sl{};
    for (uint32_t p = 0; p < P; ++p) sl[p] = slots[p];

    return ie::ps(q, "hy4_moe_dn_iq3", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<sycl::half, 1> A_slm(sycl::range<1>(K), h);
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * kWGItems, kWGItems),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t wgid  = uint32_t(it.get_group(0));
            const uint32_t sg_id = lid / kSG;
            const uint32_t lane  = lid % kSG;
            const uint32_t n     = wgid * kNPerWG + sg_id;
            const int sb   = int(lane) >> 1;
            const int half = int(lane) & 1;

            for (uint32_t p = 0; p < P; ++p) {
                const sycl::half* Ap = A + uint64_t(p) * K;
                for (uint32_t i = lid; i < K; i += kWGItems) A_slm[i] = Ap[i];
                sycl::group_barrier(it.get_group());
                if (n < N) {
                    const uint8_t* col = slot_base + uint64_t(sl[p]) * slot_bytes + down_off +
                                         uint64_t(n) * blocks_per_col * 98;
                    float acc = 0.f;
                    for (uint32_t b = 0; b < blocks_per_col; ++b) {
                        const uint8_t* blk = col + uint64_t(b) * 98;
                        const uint16_t d16 = uint16_t(blk[0]) | (uint16_t(blk[1]) << 8);
                        const uint8_t* gq  = blk + 2;
                        const uint8_t* scs = blk + 2 + 64;
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
                            const uint32_t grid = kIQ3XXSGrid[gq[sb * 8 + half * 4 + t]];
                            const int l_sel = half * 2 + (t >> 1);
                            const uint8_t signs = kSignsIQ2XS[(aux32 >> (7 * l_sel)) & 127];
                            #pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                const float mag = float((grid >> (8 * j)) & 0xFF);
                                const float s = (signs & kMaskIQ2XS[(t & 1) * 4 + j]) ? -1.f : 1.f;
                                sum += float(a[t * 4 + j]) * mag * s;
                            }
                        }
                        acc += db * sum;
                    }
                    acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
                    if (lane == 0) y[uint64_t(p) * N + n] = sycl::half(acc);
                }
                sycl::group_barrier(it.get_group());   // before the next expert's SLM reload
            }
        });
    });
}

// ---------------------------------------------------------------------------
// IQ4_XS down, grouped (blk 75-77). Lane -> (sb, half), 16 nibbles —
// gemv_iq4_xs verbatim; 136-byte blocks read through block_iq4_xs.
// ---------------------------------------------------------------------------
sycl::event moe_gemv_iq4_xs_down_grouped(sycl::queue& q, const sycl::half* A,
                                         const uint8_t* slot_base, uint64_t slot_bytes,
                                         uint64_t down_off,
                                         const int32_t* slots, uint32_t P,
                                         sycl::half* y, uint32_t K, uint32_t N,
                                         const std::vector<sycl::event>& deps) {
    const uint32_t blocks_per_col = K / 256;
    const uint32_t n_wgs = (N + kNPerWG - 1) / kNPerWG;
    Slots8 sl{};
    for (uint32_t p = 0; p < P; ++p) sl[p] = slots[p];

    return ie::ps(q, "hy4_moe_dn_iq4", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<sycl::half, 1> A_slm(sycl::range<1>(K), h);
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * kWGItems, kWGItems),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t wgid  = uint32_t(it.get_group(0));
            const uint32_t sg_id = lid / kSG;
            const uint32_t lane  = lid % kSG;
            const uint32_t n     = wgid * kNPerWG + sg_id;
            const int sb   = int(lane) >> 1;
            const int half = int(lane) & 1;
            const int shift = half ? 4 : 0;

            for (uint32_t p = 0; p < P; ++p) {
                const sycl::half* Ap = A + uint64_t(p) * K;
                for (uint32_t i = lid; i < K; i += kWGItems) A_slm[i] = Ap[i];
                sycl::group_barrier(it.get_group());
                if (n < N) {
                    const auto* W = reinterpret_cast<const block_iq4_xs*>(
                        slot_base + uint64_t(sl[p]) * slot_bytes + down_off);
                    const block_iq4_xs* col = &W[uint64_t(n) * blocks_per_col];
                    float acc = 0.f;
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
                            sum += float(a_v[i]) * float(kIQ4NLValues[(qs_v[i] >> shift) & 0x0F]);
                        acc += dl * sum;
                    }
                    acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
                    if (lane == 0) y[uint64_t(p) * N + n] = sycl::half(acc);
                }
                sycl::group_barrier(it.get_group());
            }
        });
    });
}

}  // namespace ie
