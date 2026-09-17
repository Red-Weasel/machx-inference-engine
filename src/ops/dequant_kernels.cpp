// src/ops/dequant_kernels.cpp — SYCL kernels for Q8_0/Q4_K/Q5_K/Q6_K dequant.
//
// Each kernel uses one workgroup per block (Q8_0 32-elem; K-quants 256-elem).
// Scales/mins are loaded cooperatively into SLM and reused by all WG items.

#include "ie/dequant.hpp"
#include "ie/quant_blocks.hpp"
#include "ie/quant_soa.hpp"

#include <sycl/sycl.hpp>
#include "ie/kernel_profiler.hpp"

namespace ie {

namespace {

// Device-side fp16 -> fp32 — same algorithm as ie::fp16_to_fp32, written so
// it inlines cleanly in a SYCL kernel.
inline float dev_fp16_to_fp32(uint16_t h) {
    const uint32_t s = uint32_t(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1fu;
    uint32_t m =  h        & 0x3ffu;
    if (e == 0) {
        if (m == 0) return sycl::bit_cast<float>(s);
        while ((m & 0x400u) == 0) { m <<= 1; e -= 1; }
        e += 1;
        m &= ~0x400u;
    } else if (e == 31) {
        const uint32_t r = s | 0x7f800000u | (m << 13);
        return sycl::bit_cast<float>(r);
    }
    e += (127 - 15);
    const uint32_t r = s | (e << 23) | (m << 13);
    return sycl::bit_cast<float>(r);
}

// 6-bit packed scale/min decode (research/03 §2.5). Same on host and device.
inline void get_scale_min_k4(int j, const uint8_t* q,
                             uint8_t& sc_out, uint8_t& m_out) {
    if (j < 4) {
        sc_out = q[j]     & 0x3F;
        m_out  = q[j + 4] & 0x3F;
    } else {
        sc_out = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        m_out  = (q[j + 4] >>   4) | ((q[j - 0] >> 6) << 4);
    }
}

}  // namespace

// ============================================================
//  Q8_0 — 32 elements per block, 34 bytes
//  WG = 256 items = 8 blocks per WG; each item dequants 1 element.
// ============================================================
sycl::event dequant_q8_0(sycl::queue& q, const void* packed_in,
                         sycl::half* out, size_t n_elements,
                         const std::vector<sycl::event>& deps) {
    const size_t n_blocks  = n_elements / 32;
    const size_t blocks_per_wg = 8;
    const size_t wg_items = blocks_per_wg * 32;            // 256
    const size_t n_wgs    = (n_blocks + blocks_per_wg - 1) / blocks_per_wg;

    auto* p_in = static_cast<const block_q8_0*>(packed_in);

    return ie::ps(q, "dequant_q8_0", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(n_wgs * wg_items, wg_items),
                       [=](sycl::nd_item<1> it) {
            const size_t lid     = it.get_local_id(0);
            const size_t gid_blk = it.get_group(0) * blocks_per_wg + lid / 32;
            const size_t lane    = lid % 32;
            if (gid_blk >= n_blocks) return;
            const auto& b = p_in[gid_blk];
            const float d = dev_fp16_to_fp32(b.d);
            out[gid_blk * 32 + lane] = sycl::half(d * float(b.qs[lane]));
        });
    });
}

// ============================================================
//  Q4_K — 256 elements per block, 144 bytes
//  WG = 64 items per super-block. Each item reads TWO qs bytes (4-byte aligned)
//  and produces FOUR output elements: two low-nibble outputs and two
//  high-nibble outputs at consecutive positions within the super-half.
// ============================================================
sycl::event dequant_q4_K(sycl::queue& q, const void* packed_in,
                         sycl::half* out, size_t n_elements,
                         const std::vector<sycl::event>& deps) {
    const size_t n_blocks = n_elements / 256;
    constexpr size_t WG = 64;
    auto* p_in = static_cast<const block_q4_K*>(packed_in);

    return ie::ps(q, "dequant_q4k", [&](sycl::handler& h) {
        h.depends_on(deps);

        sycl::local_accessor<float, 1> sc8 (sycl::range<1>(8),  h);
        sycl::local_accessor<float, 1> mn8 (sycl::range<1>(8),  h);
        sycl::local_accessor<float, 1> dpr (sycl::range<1>(2),  h);

        h.parallel_for(sycl::nd_range<1>(n_blocks * WG, WG),
                       [=](sycl::nd_item<1> it) {
            const size_t lid = it.get_local_id(0);
            const size_t bk  = it.get_group(0);
            const auto&  b   = p_in[bk];

            if (lid < 8) {
                uint8_t s, m;
                get_scale_min_k4(int(lid), b.scales, s, m);
                sc8[lid] = float(s);
                mn8[lid] = float(m);
            }
            if (lid == 0) {
                dpr[0] = dev_fp16_to_fp32(b.d);
                dpr[1] = dev_fp16_to_fp32(b.dmin);
            }
            sycl::group_barrier(it.get_group());

            // Each item handles bytes (2*lid, 2*lid+1), 0..127.
            const int k_lo = 2 * int(lid);
            const int k_hi = k_lo + 1;
            const int g    = k_lo / 32;            // 0..3 (same for both bytes)
            const int l_a  = k_lo % 32;            // even, 0..30
            const int l_b  = k_hi % 32;            // l_a + 1
            const uint8_t qa = b.qs[k_lo];
            const uint8_t qb = b.qs[k_hi];

            const int is_lo = 2 * g + 0;
            const int is_hi = 2 * g + 1;
            const float dlo = dpr[0] * sc8[is_lo];
            const float mlo = dpr[1] * mn8[is_lo];
            const float dhi = dpr[0] * sc8[is_hi];
            const float mhi = dpr[1] * mn8[is_hi];

            const float v_a_lo = dlo * float(qa & 0x0F) - mlo;
            const float v_b_lo = dlo * float(qb & 0x0F) - mlo;
            const float v_a_hi = dhi * float(qa >>   4) - mhi;
            const float v_b_hi = dhi * float(qb >>   4) - mhi;

            const size_t base = bk * 256 + g * 64;
            out[base + l_a]          = sycl::half(v_a_lo);
            out[base + l_b]          = sycl::half(v_b_lo);
            out[base + l_a + 32]     = sycl::half(v_a_hi);
            out[base + l_b + 32]     = sycl::half(v_b_hi);
        });
    });
}

// ============================================================
//  Q5_K — 256 elements per block, 176 bytes
//  WG = 128 items per super-block. Each item reads one qs byte and one qh
//  byte (shared by lane group), produces TWO output elements.
// ============================================================
sycl::event dequant_q5_K(sycl::queue& q, const void* packed_in,
                         sycl::half* out, size_t n_elements,
                         const std::vector<sycl::event>& deps) {
    const size_t n_blocks = n_elements / 256;
    constexpr size_t WG = 128;
    auto* p_in = static_cast<const block_q5_K*>(packed_in);

    return ie::ps(q, "dequant_q5k", [&](sycl::handler& h) {
        h.depends_on(deps);

        sycl::local_accessor<float, 1> sc8 (sycl::range<1>(8),  h);
        sycl::local_accessor<float, 1> mn8 (sycl::range<1>(8),  h);
        sycl::local_accessor<float, 1> dpr (sycl::range<1>(2),  h);

        h.parallel_for(sycl::nd_range<1>(n_blocks * WG, WG),
                       [=](sycl::nd_item<1> it) {
            const size_t lid = it.get_local_id(0);
            const size_t bk  = it.get_group(0);
            const auto&  b   = p_in[bk];

            if (lid < 8) {
                uint8_t s, m;
                get_scale_min_k4(int(lid), b.scales, s, m);
                sc8[lid] = float(s);
                mn8[lid] = float(m);
            }
            if (lid == 0) {
                dpr[0] = dev_fp16_to_fp32(b.d);
                dpr[1] = dev_fp16_to_fp32(b.dmin);
            }
            sycl::group_barrier(it.get_group());

            const int k = int(lid);             // 0..127
            const int g = k / 32;               // 0..3
            const int l = k % 32;               // 0..31
            const uint8_t ql = b.qs[k];         // packed low+high nibbles
            const uint8_t qh = b.qh[l];         // 5th bits, shared across g
            const uint8_t u_lo = uint8_t(1u << (2 * g + 0));
            const uint8_t u_hi = uint8_t(1u << (2 * g + 1));

            const int is_lo = 2 * g + 0;
            const int is_hi = 2 * g + 1;
            const float dlo = dpr[0] * sc8[is_lo];
            const float mlo = dpr[1] * mn8[is_lo];
            const float dhi = dpr[0] * sc8[is_hi];
            const float mhi = dpr[1] * mn8[is_hi];

            const int q5_lo = (ql & 0x0F) + ((qh & u_lo) ? 16 : 0);
            const int q5_hi = (ql >>   4) + ((qh & u_hi) ? 16 : 0);
            const float v_lo = dlo * float(q5_lo) - mlo;
            const float v_hi = dhi * float(q5_hi) - mhi;
            out[bk * 256 + g * 64 + l]      = sycl::half(v_lo);
            out[bk * 256 + g * 64 + l + 32] = sycl::half(v_hi);
        });
    });
}

// ============================================================
//  Q6_K — 256 elements per block, 210 bytes
//  WG = 64 items per super-block. Each item handles ONE (half, l) pair and
//  produces FOUR output elements at offsets 0, 32, 64, 96 within its half.
//  Reads: 2 ql bytes + 1 qh byte per item -> 4 fp16 writes.
// ============================================================
sycl::event dequant_q6_K(sycl::queue& q, const void* packed_in,
                         sycl::half* out, size_t n_elements,
                         const std::vector<sycl::event>& deps) {
    const size_t n_blocks = n_elements / 256;
    constexpr size_t WG = 64;
    auto* p_in = static_cast<const block_q6_K*>(packed_in);

    return ie::ps(q, "dequant_q6k", [&](sycl::handler& h) {
        h.depends_on(deps);

        sycl::local_accessor<float,1> dval(sycl::range<1>(1), h);
        sycl::local_accessor<float,1> sc16(sycl::range<1>(16), h);

        h.parallel_for(sycl::nd_range<1>(n_blocks * WG, WG),
                       [=](sycl::nd_item<1> it) {
            const size_t lid = it.get_local_id(0);
            const size_t bk  = it.get_group(0);
            const auto&  b   = p_in[bk];

            if (lid < 16) sc16[lid] = float(b.scales[lid]);
            if (lid == 0) dval[0]   = dev_fp16_to_fp32(b.d);
            sycl::group_barrier(it.get_group());

            const int half = int(lid) / 32;            // 0 or 1
            const int l    = int(lid) % 32;            // 0..31
            const int is   = l / 16;                   // 0 or 1
            const int ql_lo_off = half * 64 + l;       // ql[l] within half
            const int ql_hi_off = ql_lo_off + 32;      // ql[l + 32] within half
            const int qh_off    = half * 32 + l;
            const int sc_off    = half * 8;

            const uint8_t ql_lo = b.ql[ql_lo_off];
            const uint8_t ql_hi = b.ql[ql_hi_off];
            const uint8_t qh_b  = b.qh[qh_off];

            const int q1 = int8_t((ql_lo & 0x0F) | (((qh_b >> 0) & 3) << 4)) - 32;
            const int q2 = int8_t((ql_hi & 0x0F) | (((qh_b >> 2) & 3) << 4)) - 32;
            const int q3 = int8_t((ql_lo >>   4) | (((qh_b >> 4) & 3) << 4)) - 32;
            const int q4 = int8_t((ql_hi >>   4) | (((qh_b >> 6) & 3) << 4)) - 32;

            const float d = dval[0];
            const float v1 = d * sc16[sc_off + is + 0] * float(q1);
            const float v2 = d * sc16[sc_off + is + 2] * float(q2);
            const float v3 = d * sc16[sc_off + is + 4] * float(q3);
            const float v4 = d * sc16[sc_off + is + 6] * float(q4);

            const size_t base = bk * 256 + half * 128 + l;
            out[base + 0]  = sycl::half(v1);
            out[base + 32] = sycl::half(v2);
            out[base + 64] = sycl::half(v3);
            out[base + 96] = sycl::half(v4);
        });
    });
}

// ============================================================
//  Transposed full-matrix dequant → B^T fp16 [K, N] for gemm_fp16
//  (prefill fp16-GEMM path, docs/prefill_attack_plan_2026-06-09.md E1).
//
//  Each lane dequants ONE super-block (column n, k-block b) into registers
//  and writes its 256 values down column n of B^T (stride N). WG = 64
//  consecutive columns of the same k-block, so for each fixed k the 64
//  lanes' stores land on consecutive addresses — coalesced 128 B stores.
//  Reads stay scalar (144/210 B per lane) but reads are 4–7× smaller than
//  the fp16 writes, so writes dominate the traffic.
// ============================================================
sycl::event dequant_q4_K_to_Bt(sycl::queue& q, const void* packed_in,
                               sycl::half* out, uint32_t K, uint32_t N,
                               const std::vector<sycl::event>& deps) {
    const uint32_t blocks_per_col = K / 256;
    constexpr uint32_t WG = 64;
    auto* W = static_cast<const block_q4_K*>(packed_in);

    return ie::ps(q, "dequant_q4k_bt", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<2>({blocks_per_col, N}, {1, WG}),
                       [=](sycl::nd_item<2> it) {
            const uint32_t b = uint32_t(it.get_group(0));
            const uint32_t n = uint32_t(it.get_global_id(1));
            const block_q4_K& blk = W[uint64_t(n) * blocks_per_col + b];
            sycl::half* dst = out + uint64_t(b) * 256 * N + n;

            const float d_super = dev_fp16_to_fp32(blk.d);
            const float m_super = dev_fp16_to_fp32(blk.dmin);
            #pragma unroll
            for (int sub = 0; sub < 8; ++sub) {
                uint8_t s_raw, m_raw;
                get_scale_min_k4(sub, blk.scales, s_raw, m_raw);
                const float scale = d_super * float(s_raw);
                const float bias  = m_super * float(m_raw);
                const int g       = sub >> 1;
                const int hi_nib  = sub & 1;
                const int qs_base = g * 32;
                const int k_base  = g * 64 + hi_nib * 32;
                #pragma unroll
                for (int i = 0; i < 32; ++i) {
                    const uint8_t qb = blk.qs[qs_base + i];
                    const int q4 = hi_nib ? (qb >> 4) : (qb & 0x0F);
                    dst[uint64_t(k_base + i) * N] =
                        sycl::half(scale * float(q4) - bias);
                }
            }
        });
    });
}

// MXFP4 (gpt-oss): E8M0 *half-scaled* shared exponent. Bit-exact with llama's
// ggml_e8m0_to_fp32_half (e<2 → denormal patterns 0x00200000<<e; else 2^(e-128)
// via exp field e-1). File-local; inlines into the kernels below.
static inline float dev_e8m0_half(uint8_t e) {
    const uint32_t bits = (e < 2u) ? (0x00200000u << e) : (uint32_t(e - 1u) << 23);
    return sycl::bit_cast<float>(bits);
}

// MXFP4 → contiguous fp16 [n_elements]. 32 vals/block; qs[j] low nibble = elem j,
// high nibble = elem j+16 (one lane per qs byte → 2 outputs).
sycl::event dequant_mxfp4(sycl::queue& q, const void* packed_in,
                          sycl::half* out, size_t n_elements,
                          const std::vector<sycl::event>& deps) {
    const size_t n_blocks = n_elements / kQK_MXFP4;
    constexpr size_t blocks_per_wg = 8;
    constexpr size_t LANES = kQK_MXFP4 / 2;           // 16 (one qs byte/lane)
    const size_t wg_items = blocks_per_wg * LANES;    // 128
    const size_t n_wgs = (n_blocks + blocks_per_wg - 1) / blocks_per_wg;
    auto* p_in = static_cast<const block_mxfp4*>(packed_in);

    return ie::ps(q, "dequant_mxfp4", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(n_wgs * wg_items, wg_items),
                       [=](sycl::nd_item<1> it) {
            const size_t lid = it.get_local_id(0);
            const size_t blk = it.get_group(0) * blocks_per_wg + lid / LANES;
            const size_t j   = lid % LANES;           // 0..15
            if (blk >= n_blocks) return;
            const int lut[16] = {0, 1, 2, 3, 4, 6, 8, 12,
                                 0, -1, -2, -3, -4, -6, -8, -12};
            const auto& b = p_in[blk];
            const float d = dev_e8m0_half(b.e);
            const uint8_t qb = b.qs[j];
            out[blk * kQK_MXFP4 + j]      = sycl::half(d * float(lut[qb & 0x0F]));
            out[blk * kQK_MXFP4 + j + 16] = sycl::half(d * float(lut[qb >> 4]));
        });
    });
}

// MXFP4 → B^T fp16 [K,N] (out[k*N+n]) for the per-expert oneDNN MoE GEMM. Input
// is N columns, each with K/32 blocks along K. K%32==0, N%64==0 (same WG=64
// column mapping as dequant_q4_K_to_Bt). One item per (block, n) → 32 rows.
sycl::event dequant_mxfp4_to_Bt(sycl::queue& q, const void* packed_in,
                                sycl::half* out, uint32_t K, uint32_t N,
                                const std::vector<sycl::event>& deps) {
    const uint32_t blocks_per_col = K / kQK_MXFP4;
    constexpr uint32_t WG = 64;
    auto* W = static_cast<const block_mxfp4*>(packed_in);

    return ie::ps(q, "dequant_mxfp4_bt", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<2>({blocks_per_col, N}, {1, WG}),
                       [=](sycl::nd_item<2> it) {
            const uint32_t b = uint32_t(it.get_group(0));
            const uint32_t n = uint32_t(it.get_global_id(1));
            const int lut[16] = {0, 1, 2, 3, 4, 6, 8, 12,
                                 0, -1, -2, -3, -4, -6, -8, -12};
            const block_mxfp4& blk = W[uint64_t(n) * blocks_per_col + b];
            sycl::half* dst = out + uint64_t(b) * kQK_MXFP4 * N + n;
            const float d = dev_e8m0_half(blk.e);
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                const uint8_t qb = blk.qs[j];
                dst[uint64_t(j)      * N] = sycl::half(d * float(lut[qb & 0x0F]));
                dst[uint64_t(j + 16) * N] = sycl::half(d * float(lut[qb >> 4]));
            }
        });
    });
}

sycl::event dequant_q6_K_to_Bt(sycl::queue& q, const void* packed_in,
                               sycl::half* out, uint32_t K, uint32_t N,
                               const std::vector<sycl::event>& deps) {
    const uint32_t blocks_per_col = K / 256;
    constexpr uint32_t WG = 64;
    auto* W = static_cast<const block_q6_K*>(packed_in);

    return ie::ps(q, "dequant_q6k_bt", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<2>({blocks_per_col, N}, {1, WG}),
                       [=](sycl::nd_item<2> it) {
            const uint32_t b = uint32_t(it.get_group(0));
            const uint32_t n = uint32_t(it.get_global_id(1));
            const block_q6_K& blk = W[uint64_t(n) * blocks_per_col + b];
            sycl::half* dst = out + uint64_t(b) * 256 * N + n;

            const float d = dev_fp16_to_fp32(blk.d);
            #pragma unroll
            for (int hf = 0; hf < 2; ++hf) {
                #pragma unroll
                for (int l = 0; l < 32; ++l) {
                    const int is = l / 16;
                    const uint8_t ql_lo = blk.ql[hf * 64 + l];
                    const uint8_t ql_hi = blk.ql[hf * 64 + l + 32];
                    const uint8_t qh_b  = blk.qh[hf * 32 + l];
                    const int sc = hf * 8;
                    const int q1 = int8_t((ql_lo & 0x0F) | (((qh_b >> 0) & 3) << 4)) - 32;
                    const int q2 = int8_t((ql_hi & 0x0F) | (((qh_b >> 2) & 3) << 4)) - 32;
                    const int q3 = int8_t((ql_lo >>   4) | (((qh_b >> 4) & 3) << 4)) - 32;
                    const int q4 = int8_t((ql_hi >>   4) | (((qh_b >> 6) & 3) << 4)) - 32;
                    const int k_base = hf * 128 + l;
                    dst[uint64_t(k_base +  0) * N] = sycl::half(d * float(blk.scales[sc + is + 0]) * float(q1));
                    dst[uint64_t(k_base + 32) * N] = sycl::half(d * float(blk.scales[sc + is + 2]) * float(q2));
                    dst[uint64_t(k_base + 64) * N] = sycl::half(d * float(blk.scales[sc + is + 4]) * float(q3));
                    dst[uint64_t(k_base + 96) * N] = sycl::half(d * float(blk.scales[sc + is + 6]) * float(q4));
                }
            }
        });
    });
}

// Q6_K SoA (per-expert MoE repack) → B^T fp16 [K,N]. Same value math and same
// (block,column) work-item mapping as dequant_q6_K_to_Bt above, but reads the
// four de-interleaved planes produced by repack_moe_q6k_soa_host instead of the
// canonical block_q6_K AoS struct.
//
// `soa_bank_expert_slice` is the per-expert region base (= bank + e*expert_stride
// bytes — exactly the same base the AoS variant gets, because the SoA repack
// keeps every expert region at its AoS stride nb*210 and only reorders bytes
// WITHIN the region). The region holds nb_total = N*(K/256) Q6_K blocks. Plane
// offsets/strides (q6k_soa_offsets, quant_soa.hpp):
//   ql:     base + 0              , 128 B/block
//   qh:     base + nb_total*128   ,  64 B/block
//   scales: base + nb_total*192   ,  16 B/block (int8)
//   d:      base + nb_total*208   ,   2 B/block (raw fp16 bits)
// The AoS variant indexes block (n, b) as W[n*blocks_per_col + b]; the repack
// flattens with the SAME order (b ranges over the whole region n-major), so the
// flat block index here is bi = n*blocks_per_col + b. Because the per-block
// ql[128]/qh[64]/scales[16]/d bytes are copied verbatim (no requant, no intra-
// block reorder), the per-element reconstruction below is byte-for-byte the
// AoS dequant — output is bit-identical to dequant_q6_K_to_Bt on the same weight.
sycl::event dequant_q6_K_soa_to_Bt(sycl::queue& q, const void* soa_bank_expert_slice,
                                   sycl::half* out, uint32_t K, uint32_t N,
                                   const std::vector<sycl::event>& deps) {
    const uint32_t blocks_per_col = K / 256;
    const uint64_t nb_total = uint64_t(N) * blocks_per_col;
    constexpr uint32_t WG = 64;

    const q6k_soa_offsets off(nb_total);
    const auto* base   = static_cast<const uint8_t*>(soa_bank_expert_slice);
    const uint8_t* ql_plane = base + off.ql;
    const uint8_t* qh_plane = base + off.qh;
    const int8_t*  sc_plane = reinterpret_cast<const int8_t*>(base + off.scales);
    const uint8_t* d_plane  = base + off.d;

    return ie::ps(q, "dequant_q6k_soa_bt", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<2>({blocks_per_col, N}, {1, WG}),
                       [=](sycl::nd_item<2> it) {
            const uint32_t b = uint32_t(it.get_group(0));
            const uint32_t n = uint32_t(it.get_global_id(1));
            const uint64_t bi = uint64_t(n) * blocks_per_col + b;

            // Per-block plane slices (strides: ql 128, qh 64, scales 16, d 2).
            const uint8_t* ql = ql_plane + bi * 128u;
            const uint8_t* qh = qh_plane + bi * 64u;
            const int8_t*  scales = sc_plane + bi * 16u;
            const uint16_t d_bits = *reinterpret_cast<const uint16_t*>(d_plane + bi * 2u);

            sycl::half* dst = out + uint64_t(b) * 256 * N + n;
            const float d = dev_fp16_to_fp32(d_bits);
            #pragma unroll
            for (int hf = 0; hf < 2; ++hf) {
                #pragma unroll
                for (int l = 0; l < 32; ++l) {
                    const int is = l / 16;
                    const uint8_t ql_lo = ql[hf * 64 + l];
                    const uint8_t ql_hi = ql[hf * 64 + l + 32];
                    const uint8_t qh_b  = qh[hf * 32 + l];
                    const int sc = hf * 8;
                    const int q1 = int8_t((ql_lo & 0x0F) | (((qh_b >> 0) & 3) << 4)) - 32;
                    const int q2 = int8_t((ql_hi & 0x0F) | (((qh_b >> 2) & 3) << 4)) - 32;
                    const int q3 = int8_t((ql_lo >>   4) | (((qh_b >> 4) & 3) << 4)) - 32;
                    const int q4 = int8_t((ql_hi >>   4) | (((qh_b >> 6) & 3) << 4)) - 32;
                    const int k_base = hf * 128 + l;
                    dst[uint64_t(k_base +  0) * N] = sycl::half(d * float(scales[sc + is + 0]) * float(q1));
                    dst[uint64_t(k_base + 32) * N] = sycl::half(d * float(scales[sc + is + 2]) * float(q2));
                    dst[uint64_t(k_base + 64) * N] = sycl::half(d * float(scales[sc + is + 4]) * float(q3));
                    dst[uint64_t(k_base + 96) * N] = sycl::half(d * float(scales[sc + is + 6]) * float(q4));
                }
            }
        });
    });
}

// Q4_K SoA (per-expert MoE repack) → B^T fp16 [K,N]. Same value math and
// (block,column) work-item mapping as dequant_q4_K_to_Bt above, but reads the
// three de-interleaved planes produced by repack_moe_q4k_soa_host instead of the
// canonical block_q4_K AoS struct. Plane offsets/strides (q4k_soa_offsets):
//   qs:     base + 0            , 128 B/block
//   scales: base + nb_total*128 ,  12 B/block (PACKED 6-bit get_scale_min_k4 form)
//   dm:     base + nb_total*140 ,   4 B/block ({d, dmin} as raw fp16 bit pairs)
// The repack copies per-block bytes verbatim (no requant/intra-block reorder) and
// flattens n-major (bi = n*blocks_per_col + b), so the reconstruction below is
// byte-for-byte the AoS dequant — output is bit-identical to dequant_q4_K_to_Bt.
sycl::event dequant_q4_moe_soa_to_Bt(sycl::queue& q, const void* soa_bank_expert_slice,
                                     sycl::half* out, uint32_t K, uint32_t N,
                                     const std::vector<sycl::event>& deps) {
    const uint32_t blocks_per_col = K / 256;
    const uint64_t nb_total = uint64_t(N) * blocks_per_col;
    constexpr uint32_t WG = 64;

    const q4k_soa_offsets off(nb_total);
    const auto* base = static_cast<const uint8_t*>(soa_bank_expert_slice);
    const uint8_t* qs_plane = base + off.qs;        // 128 B/block
    const uint8_t* sc_plane = base + off.scales;    //  12 B/block (packed)
    const uint8_t* dm_plane = base + off.dm;        //   4 B/block ({d,dmin} fp16)

    return ie::ps(q, "dequant_q4moe_soa_bt", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<2>({blocks_per_col, N}, {1, WG}),
                       [=](sycl::nd_item<2> it) {
            const uint32_t b = uint32_t(it.get_group(0));
            const uint32_t n = uint32_t(it.get_global_id(1));
            const uint64_t bi = uint64_t(n) * blocks_per_col + b;

            const uint8_t* qs     = qs_plane + bi * 128u;
            const uint8_t* scales = sc_plane + bi * 12u;
            const uint16_t* dm    = reinterpret_cast<const uint16_t*>(dm_plane + bi * 4u);

            sycl::half* dst = out + uint64_t(b) * 256 * N + n;
            const float d_super = dev_fp16_to_fp32(dm[0]);   // d
            const float m_super = dev_fp16_to_fp32(dm[1]);   // dmin
            #pragma unroll
            for (int sub = 0; sub < 8; ++sub) {
                uint8_t s_raw, m_raw;
                get_scale_min_k4(sub, scales, s_raw, m_raw);
                const float scale = d_super * float(s_raw);
                const float bias  = m_super * float(m_raw);
                const int g       = sub >> 1;
                const int hi_nib  = sub & 1;
                const int qs_base = g * 32;
                const int k_base  = g * 64 + hi_nib * 32;
                #pragma unroll
                for (int i = 0; i < 32; ++i) {
                    const uint8_t qb = qs[qs_base + i];
                    const int q4 = hi_nib ? (qb >> 4) : (qb & 0x0F);
                    dst[uint64_t(k_base + i) * N] =
                        sycl::half(scale * float(q4) - bias);
                }
            }
        });
    });
}

// Q5_K → B^T fp16 [K,N]. One work-item per (block, column); unpacks the 256-elem
// block into the transposed output (out[k*N + n]). Mirrors dequant_q6_K_to_Bt.
sycl::event dequant_q5_K_to_Bt(sycl::queue& q, const void* packed_in,
                               sycl::half* out, uint32_t K, uint32_t N,
                               const std::vector<sycl::event>& deps) {
    const uint32_t blocks_per_col = K / 256;
    constexpr uint32_t WG = 64;
    auto* W = static_cast<const block_q5_K*>(packed_in);

    return ie::ps(q, "dequant_q5k_bt", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<2>({blocks_per_col, N}, {1, WG}),
                       [=](sycl::nd_item<2> it) {
            const uint32_t b = uint32_t(it.get_group(0));
            const uint32_t n = uint32_t(it.get_global_id(1));
            const block_q5_K& blk = W[uint64_t(n) * blocks_per_col + b];
            sycl::half* dst = out + uint64_t(b) * 256 * N + n;

            const float d  = dev_fp16_to_fp32(blk.d);
            const float dm = dev_fp16_to_fp32(blk.dmin);
            float sc[8], mn[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                uint8_t s, m;
                get_scale_min_k4(j, blk.scales, s, m);
                sc[j] = float(s); mn[j] = float(m);
            }
            #pragma unroll
            for (int k = 0; k < 128; ++k) {
                const int g = k / 32, l = k % 32;
                const uint8_t ql = blk.qs[k];
                const uint8_t qh = blk.qh[l];
                const int q5_lo = (ql & 0x0F) + ((qh & (1u << (2 * g + 0))) ? 16 : 0);
                const int q5_hi = (ql >>   4) + ((qh & (1u << (2 * g + 1))) ? 16 : 0);
                const float v_lo = d * sc[2 * g + 0] * float(q5_lo) - dm * mn[2 * g + 0];
                const float v_hi = d * sc[2 * g + 1] * float(q5_hi) - dm * mn[2 * g + 1];
                dst[uint64_t(g * 64 + l)      * N] = sycl::half(v_lo);
                dst[uint64_t(g * 64 + l + 32) * N] = sycl::half(v_hi);
            }
        });
    });
}

// Q8_0 → B^T fp16 [K,N]. 32-elem blocks; one work-item per (block, column).
sycl::event dequant_q8_0_to_Bt(sycl::queue& q, const void* packed_in,
                               sycl::half* out, uint32_t K, uint32_t N,
                               const std::vector<sycl::event>& deps) {
    const uint32_t blocks_per_col = K / 32;
    constexpr uint32_t WG = 64;
    auto* W = static_cast<const block_q8_0*>(packed_in);

    return ie::ps(q, "dequant_q8_0_bt", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<2>({blocks_per_col, N}, {1, WG}),
                       [=](sycl::nd_item<2> it) {
            const uint32_t b = uint32_t(it.get_group(0));
            const uint32_t n = uint32_t(it.get_global_id(1));
            const block_q8_0& blk = W[uint64_t(n) * blocks_per_col + b];
            sycl::half* dst = out + uint64_t(b) * 32 * N + n;
            const float d = dev_fp16_to_fp32(blk.d);
            #pragma unroll
            for (int l = 0; l < 32; ++l)
                dst[uint64_t(l) * N] = sycl::half(d * float(blk.qs[l]));
        });
    });
}

}  // namespace ie
