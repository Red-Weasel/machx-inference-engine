// src/ops/dequant_kernels.cpp — SYCL kernels for Q8_0/Q4_K/Q5_K/Q6_K dequant.
//
// Each kernel uses one workgroup per block (Q8_0 32-elem; K-quants 256-elem).
// Scales/mins are loaded cooperatively into SLM and reused by all WG items.

#include "ie/dequant.hpp"
#include "ie/quant_blocks.hpp"

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

}  // namespace ie
