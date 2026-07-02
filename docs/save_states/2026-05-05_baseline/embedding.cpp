// src/ops/embedding.cpp — Q6_K row dequant for embedding lookup.

#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"

#include <sycl/sycl.hpp>
#include "ie/kernel_profiler.hpp"

namespace ie {

namespace {

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

}  // namespace

// One WG per token, 64 lanes per WG. Each WG dequants `hidden / 256` super-blocks.
// hidden must be a multiple of 256 (true for Qwen3.6: 2048 / 256 = 8).
sycl::event embedding_lookup_q6k(sycl::queue& q,
                                 const int32_t* token_ids,
                                 const void* token_embd_q6k,
                                 sycl::half* y,
                                 uint32_t n_tokens, uint32_t hidden,
                                 const std::vector<sycl::event>& deps) {
    constexpr uint32_t WG = 64;
    const auto* W = static_cast<const block_q6_K*>(token_embd_q6k);
    const uint32_t blocks_per_token = hidden / 256;

    return ie::ps(q, "embed_q6k", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<2>({n_tokens, WG}, {1, WG}),
                       [=](sycl::nd_item<2> it) {
            const uint32_t t = uint32_t(it.get_global_id(0));
            const uint32_t lid = uint32_t(it.get_local_id(1));
            const int32_t  tok = token_ids[t];
            if (tok < 0) return;
            const block_q6_K* row = &W[uint64_t(tok) * blocks_per_token];

            // 64 lanes × 4 outputs each = 256 outputs per super-block.
            // Iterate super-blocks serially, parallelize within each.
            for (uint32_t b = 0; b < blocks_per_token; ++b) {
                const block_q6_K& blk = row[b];
                const float d = dev_fp16_to_fp32(blk.d);

                // Lane → (half, l) where half = lid/32, l = lid%32.
                const int half = int(lid) / 32;
                const int l    = int(lid) % 32;
                const int is   = l / 16;
                const int ql_lo_off = half * 64 + l;
                const int ql_hi_off = ql_lo_off + 32;
                const int qh_off    = half * 32 + l;
                const int sc_off    = half * 8;

                const uint8_t ql_lo = blk.ql[ql_lo_off];
                const uint8_t ql_hi = blk.ql[ql_hi_off];
                const uint8_t qh_b  = blk.qh[qh_off];

                const int q1 = int8_t((ql_lo & 0x0F) | (((qh_b >> 0) & 3) << 4)) - 32;
                const int q2 = int8_t((ql_hi & 0x0F) | (((qh_b >> 2) & 3) << 4)) - 32;
                const int q3 = int8_t((ql_lo >>   4) | (((qh_b >> 4) & 3) << 4)) - 32;
                const int q4 = int8_t((ql_hi >>   4) | (((qh_b >> 6) & 3) << 4)) - 32;

                const float v1 = d * float(blk.scales[sc_off + is + 0]) * float(q1);
                const float v2 = d * float(blk.scales[sc_off + is + 2]) * float(q2);
                const float v3 = d * float(blk.scales[sc_off + is + 4]) * float(q3);
                const float v4 = d * float(blk.scales[sc_off + is + 6]) * float(q4);

                const uint64_t base = uint64_t(t) * hidden + uint64_t(b) * 256
                                    + uint64_t(half) * 128 + uint64_t(l);
                y[base + 0]  = sycl::half(v1);
                y[base + 32] = sycl::half(v2);
                y[base + 64] = sycl::half(v3);
                y[base + 96] = sycl::half(v4);
            }
        });
    });
}

// Q4_K equivalent: 64 lanes per WG, 4 outputs per lane, hidden / 256 blocks per token.
sycl::event embedding_lookup_q4k(sycl::queue& q,
                                 const int32_t* token_ids,
                                 const void* token_embd_q4k,
                                 sycl::half* y,
                                 uint32_t n_tokens, uint32_t hidden,
                                 const std::vector<sycl::event>& deps) {
    constexpr uint32_t WG = 64;
    const auto* W = static_cast<const block_q4_K*>(token_embd_q4k);
    const uint32_t blocks_per_token = hidden / 256;

    return ie::ps(q, "embed_q4k", [&](sycl::handler& h) {
        h.depends_on(deps);

        sycl::local_accessor<float, 1> sc8 (sycl::range<1>(8),  h);
        sycl::local_accessor<float, 1> mn8 (sycl::range<1>(8),  h);
        sycl::local_accessor<float, 1> dpr (sycl::range<1>(2),  h);

        h.parallel_for(sycl::nd_range<2>({n_tokens, WG}, {1, WG}),
                       [=](sycl::nd_item<2> it) {
            const uint32_t t   = uint32_t(it.get_global_id(0));
            const uint32_t lid = uint32_t(it.get_local_id(1));
            const int32_t  tok = token_ids[t];
            if (tok < 0) return;
            const block_q4_K* row = &W[uint64_t(tok) * blocks_per_token];

            for (uint32_t b = 0; b < blocks_per_token; ++b) {
                const block_q4_K& blk = row[b];
                if (lid < 8) {
                    uint8_t s, m;
                    if (lid < 4) { s = blk.scales[lid] & 0x3F; m = blk.scales[lid + 4] & 0x3F; }
                    else {
                        s = (blk.scales[lid + 4] & 0x0F) | ((blk.scales[lid - 4] >> 6) << 4);
                        m = (blk.scales[lid + 4] >>   4) | ((blk.scales[lid    ] >> 6) << 4);
                    }
                    sc8[lid] = float(s);
                    mn8[lid] = float(m);
                }
                if (lid == 0) {
                    dpr[0] = dev_fp16_to_fp32(blk.d);
                    dpr[1] = dev_fp16_to_fp32(blk.dmin);
                }
                sycl::group_barrier(it.get_group());
                // Each lane handles 2 qs bytes producing 4 outputs.
                const int k_lo = 2 * int(lid);
                const int k_hi = k_lo + 1;
                const int g    = k_lo / 32;
                const int l_a  = k_lo % 32;
                const int l_b  = k_hi % 32;
                const uint8_t qa = blk.qs[k_lo];
                const uint8_t qb = blk.qs[k_hi];
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
                const uint64_t base = uint64_t(t) * hidden + uint64_t(b) * 256
                                    + uint64_t(g) * 64;
                y[base + l_a]      = sycl::half(v_a_lo);
                y[base + l_b]      = sycl::half(v_b_lo);
                y[base + l_a + 32] = sycl::half(v_a_hi);
                y[base + l_b + 32] = sycl::half(v_b_hi);
                sycl::group_barrier(it.get_group());
            }
        });
    });
}

}  // namespace ie
