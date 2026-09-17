// src/ops/gemv_q8dot.cpp — integer-dot (dp4a) decode GEMVs.  P1a, 2026-06-09.
//
// Why: llama.cpp SYCL master's decode lead (81 vs 66 tok/s on B70) comes from
// its MMVQ path — activations quantized once to Q8_1, then GEMV inner loops
// run as INT8 dot products (idp4a) against the quantized weights directly,
// with no per-element dequant.  This file brings that technique to the
// engine: `quantize_q8_1` (one tiny launch per unique activation vector) and
// `gemv_q4_K_q8` consuming it.
//
// Math (per 256-element Q4_K super-block, per 32-element sub-block j):
//   w[i] = d4·sc_j · q4[i] − dmin4·m_j
//   Σ_i w[i]·x[i] ≈ Σ_i w[i]·(d8_j·q8[i])
//                 = d4·sc_j · d8_j · (q4·q8)_j  −  dmin4·m_j · d8_j·Σq8_j
// The (q4·q8) integer dots run 4 lanes per dp4a; the Σq8 partial sums also
// come from dp4a against 0x01010101.  Per-LANE granularity here is 16
// elements (half a q8 block), so Σq8 is computed per-16 on the fly rather
// than using block_q8_1x::s (which spans 32).
//
// Numerics: activations round to int8 (~0.4% RMS).  This is the established
// llama.cpp decode path; the engine's PPL ≤ 6.57 gate decides acceptance.

#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"

#include <sycl/sycl.hpp>
#include <cstring>
#include <cstdlib>
#include "ie/dp4a.hpp"
#include "ie/kernel_profiler.hpp"

namespace ie {

namespace {

inline float dev_fp16_to_fp32(uint16_t h) {
    // Native half->float: exact conversion, bit-identical to the old
    // software path, one instruction instead of a branchy bit chain
    // (2026-08-28 — the sw convert was the top ALU cost of the
    // dequant-bound GEMV kernels; Q5_1 swap measured bvs-bit-exact and
    // spec2 round 95.4 -> 90.9 ms).
    return float(sycl::bit_cast<sycl::half>(h));
}

}  // namespace

// Quantize fp16 activations to block_q8_1s (split half-sums; MoE prefill).
// One SG (32 lanes) per block — same numerics as quantize_q8_1 for d/qs;
// only the sum bookkeeping differs (two per-16 partial sums).
sycl::event quantize_q8_1s(sycl::queue& q,
                           const sycl::half* x, void* out_q8,
                           uint64_t K,
                           const std::vector<sycl::event>& deps) {
    const uint64_t n_blocks = K / 32;
    auto* out = static_cast<block_q8_1s*>(out_q8);

    return ie::ps(q, "quant_q8_1s", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(n_blocks * 32, 32),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const uint64_t b    = it.get_group(0);
            const uint32_t lane = uint32_t(it.get_local_id(0));
            auto sg = it.get_sub_group();

            const float v = float(x[b * 32 + lane]);
            const float amax = sycl::reduce_over_group(
                sg, sycl::fabs(v), sycl::maximum<float>());
            const float d   = amax / 127.0f;
            const float inv = (amax > 0.f) ? 127.0f / amax : 0.f;
            const int32_t qi = int32_t(sycl::round(v * inv));
            out[b].qs[lane] = int8_t(qi);
            const int32_t q_lo = sycl::reduce_over_group(
                sg, lane < 16 ? qi : 0, sycl::plus<int32_t>());
            const int32_t q_hi = sycl::reduce_over_group(
                sg, lane < 16 ? 0 : qi, sycl::plus<int32_t>());
            if (lane == 0) {
                out[b].d  = d;
                out[b].s0 = d * float(q_lo);
                out[b].s1 = d * float(q_hi);
                out[b].pad = 0.f;
            }
        });
    });
}

// Quantize fp16 activations to block_q8_1x.  One SG (32 lanes) per block.
sycl::event quantize_q8_1(sycl::queue& q,
                          const sycl::half* x, void* out_q8,
                          uint32_t K,
                          const std::vector<sycl::event>& deps) {
    const uint32_t n_blocks = K / 32;
    auto* out = static_cast<block_q8_1x*>(out_q8);

    return ie::ps(q, "quant_q8_1", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_blocks) * 32, 32),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const uint32_t b    = uint32_t(it.get_group(0));
            const uint32_t lane = uint32_t(it.get_local_id(0));
            auto sg = it.get_sub_group();

            const float v = float(x[b * 32 + lane]);
            const float amax = sycl::reduce_over_group(
                sg, sycl::fabs(v), sycl::maximum<float>());
            const float d   = amax / 127.0f;
            const float inv = (amax > 0.f) ? 127.0f / amax : 0.f;
            const int32_t qi = int32_t(sycl::round(v * inv));
            out[b].qs[lane] = int8_t(qi);
            const int32_t qsum = sycl::reduce_over_group(
                sg, qi, sycl::plus<int32_t>());
            if (lane == 0) {
                out[b].d = d;
                out[b].s = d * float(qsum);
            }
        });
    });
}

// ===========================================================================
// REORDERED Q4_K decode (llama-SYCL's layout — the 52%-BW trick, 2026-06-25).
// llama de-interleaves the AoS block_q4_K array into 3 GLOBAL contiguous regions
// so the decode reads nibbles as a PURE contiguous stream (no 16-byte d/dmin/
// scales header gap every 128 B that breaks our AoS coalescing). Same total
// bytes (4.5 bpw), same dp4a math → PPL-safe. Layout (nblocks = N·K/256):
//   nibbles [nblocks·128 B] | scales [nblocks·12 B packed] | dm [nblocks·4 B half2]
// repack_q4_K_to_reorder: host de-interleave [N,K] AoS → the 3 regions.
void repack_q4_K_to_reorder(const void* W_blocks, uint32_t K, uint32_t N,
                            uint8_t* out) {
    const uint64_t bpc     = K / 256;
    const uint64_t nblocks = uint64_t(N) * bpc;
    const auto* blk = static_cast<const block_q4_K*>(W_blocks);
    uint8_t* nib = out;
    uint8_t* sc  = out + nblocks * 128;
    uint8_t* dm  = sc  + nblocks * 12;
    for (uint64_t ib = 0; ib < nblocks; ++ib) {
        std::memcpy(nib + ib * 128, blk[ib].qs,     128);
        std::memcpy(sc  + ib * 12,  blk[ib].scales,  12);
        std::memcpy(dm  + ib * 4,   &blk[ib].d,       4);   // d (half) + dmin (half)
    }
}

// REORDERED Q4_K int-dot W4A8 GEMV. Identical lane lattice + dp4a + fold to
// gemv_q4_K_q8 (validated), but reads the 3-region reordered layout → the nibble
// stream is pure-contiguous (128 B blocks back-to-back) for max coalescing.
template <int N_PER_WG>
static sycl::event gemv_q4_K_reorder_q8_impl(sycl::queue& q,
                         const void* x_q8, const void* W_reorder,
                         sycl::half* y,
                         uint32_t K, uint32_t N,
                         const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE  = 16;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;

    const auto* X8 = static_cast<const block_q8_1x*>(x_q8);
    const uint32_t blocks_per_col = K / 256;
    const uint32_t q8_blocks      = K / 32;
    const uint64_t nblocks        = uint64_t(N) * blocks_per_col;
    const uint8_t* nib_base = static_cast<const uint8_t*>(W_reorder);
    const uint8_t* sc_base  = nib_base + nblocks * 128;
    const uint8_t* dm_base  = sc_base  + nblocks * 12;
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "gemv_q4k_reorder", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<uint32_t, 1> q8s(q8_blocks * 8, h);
        sycl::local_accessor<float, 1>    q8d(q8_blocks, h);
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t wgid  = uint32_t(it.get_group(0));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t n     = wgid * N_PER_WG + sg_id;

            for (uint32_t i = lid; i < q8_blocks * 8; i += WG_ITEMS) {
                const uint32_t blk = i / 8, w = i % 8;
                q8s[i] = reinterpret_cast<const uint32_t*>(X8[blk].qs)[w];
            }
            for (uint32_t i = lid; i < q8_blocks; i += WG_ITEMS) q8d[i] = X8[i].d;
            sycl::group_barrier(it.get_group());
            if (n >= N) return;

            const int sub  = int(lane) >> 1;
            const int half = int(lane) & 1;
            const int g     = sub >> 1;
            const int hi_nib = sub & 1;
            const int qs_off = g * 32 + half * 16;
            const int out_off = g * 64 + hi_nib * 32 + half * 16;
            const int q4_shift = hi_nib ? 4 : 0;

            float acc = 0.f;
            for (uint32_t b = 0; b < blocks_per_col; ++b) {
                const uint64_t ib = uint64_t(n) * blocks_per_col + b;
                const uint8_t* qs_ptr = nib_base + ib * 128;       // pure nibble stream
                const uint8_t* sc_ptr = sc_base  + ib * 12;
                const uint32_t dmw = *reinterpret_cast<const uint32_t*>(dm_base + ib * 4);
                const auto hsc = [&](int j) { return sc_ptr[j]; };
                uint8_t s_raw, m_raw;
                if (sub < 4) {
                    s_raw = hsc(sub)     & 0x3F;
                    m_raw = hsc(sub + 4) & 0x3F;
                } else {
                    s_raw = (hsc(sub + 4) & 0x0F) | ((hsc(sub - 4) >> 6) << 4);
                    m_raw = (hsc(sub + 4) >>   4) | ((hsc(sub    ) >> 6) << 4);
                }
                const float d4  = dev_fp16_to_fp32(uint16_t(dmw & 0xFFFFu)) * float(s_raw);
                const float dm4 = dev_fp16_to_fp32(uint16_t(dmw >> 16))     * float(m_raw);

                const auto qv = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(qs_ptr + qs_off);
                const uint32_t q8_base = (uint32_t(b) * 256 + uint32_t(out_off));
                const uint32_t q8_blk  = q8_base / 32;
                const uint32_t q8_word = (q8_base % 32) / 4;

                int32_t idot = 0, isum = 0;
                #pragma unroll
                for (int w = 0; w < 4; ++w) {
                    const uint32_t nibw = (qv[w] >> q4_shift) & 0x0F0F0F0Fu;
                    const int32_t  q8w  = int32_t(q8s[q8_blk * 8 + q8_word + w]);
                    idot = ie::dp4a_us(nibw, q8w, idot);
                    isum = ie::dp4a_us(0x01010101u, q8w, isum);
                }
                const float d8 = q8d[q8_blk];
                acc += d4 * d8 * float(idot) - dm4 * d8 * float(isum);
            }
            acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
            if (lane == 0) y[n] = sycl::half(acc);
        });
    });
}

// Public entry. WG geometry (rows/WG) is an occupancy knob on the reorder layout —
// IE_Q4K_REORDER_NPWG selects 1/8/16/32 (default 32). MMV (1 row/WG) = many tiny
// high-occupancy WGs like llama's MMVQ; fat WG (32) = our lattice. Same math.
sycl::event gemv_q4_K_reorder_q8(sycl::queue& q,
                         const void* x_q8, const void* W_reorder,
                         sycl::half* y,
                         uint32_t K, uint32_t N,
                         const std::vector<sycl::event>& deps) {
    static const int npwg = [] {
        const char* e = std::getenv("IE_Q4K_REORDER_NPWG");
        return e ? std::atoi(e) : 32;
    }();
    switch (npwg) {
        case 1:  return gemv_q4_K_reorder_q8_impl<1 >(q, x_q8, W_reorder, y, K, N, deps);
        case 8:  return gemv_q4_K_reorder_q8_impl<8 >(q, x_q8, W_reorder, y, K, N, deps);
        case 16: return gemv_q4_K_reorder_q8_impl<16>(q, x_q8, W_reorder, y, K, N, deps);
        default: return gemv_q4_K_reorder_q8_impl<32>(q, x_q8, W_reorder, y, K, N, deps);
    }
}

// Integer-dot W4A8 GEMV.  Same WG shape as gemv_q4_K (32 SGs × 16 lanes,
// one SG per output column, lanes split K).  The Q8 activation blocks are
// staged once into SLM and reused by all 32 columns.
sycl::event gemv_q4_K_q8(sycl::queue& q,
                         const void* x_q8, const void* W_packed,
                         sycl::half* y,
                         uint32_t K, uint32_t N,
                         const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE  = 16;
    constexpr int N_PER_WG = 32;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;        // 512

    const auto* W  = static_cast<const block_q4_K*>(W_packed);
    const auto* X8 = static_cast<const block_q8_1x*>(x_q8);
    const uint32_t blocks_per_col = K / 256;
    const uint32_t q8_blocks      = K / 32;
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "gemv_q4k_q8", [&](sycl::handler& h) {
        h.depends_on(deps);
        // SLM copy of the Q8 activation stream: qs as packed uint32 +
        // per-block d (s unused at per-16 granularity).
        sycl::local_accessor<uint32_t, 1> q8s(q8_blocks * 8, h);   // 32 B/blk
        sycl::local_accessor<float, 1>    q8d(q8_blocks, h);

        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid    = uint32_t(it.get_local_id(0));
            const uint32_t wgid   = uint32_t(it.get_group(0));
            const uint32_t sg_id  = lid / SG_SIZE;
            const uint32_t lane   = lid % SG_SIZE;
            const uint32_t n      = wgid * N_PER_WG + sg_id;

            // Cooperative Q8 stage: qs words + d per block.
            for (uint32_t i = lid; i < q8_blocks * 8; i += WG_ITEMS) {
                const uint32_t blk = i / 8, w = i % 8;
                q8s[i] = reinterpret_cast<const uint32_t*>(X8[blk].qs)[w];
            }
            for (uint32_t i = lid; i < q8_blocks; i += WG_ITEMS) q8d[i] = X8[i].d;
            sycl::group_barrier(it.get_group());

            if (n >= N) return;

            // Lane lattice — identical to gemv_q4_K.
            const int sub  = int(lane) >> 1;
            const int half = int(lane) & 1;
            const int g     = sub >> 1;
            const int hi_nib = sub & 1;
            const int qs_off = g * 32 + half * 16;
            const int out_off = g * 64 + hi_nib * 32 + half * 16;
            const int q4_shift = hi_nib ? 4 : 0;

            float acc = 0.f;
            const block_q4_K* col_blocks = &W[uint64_t(n) * blocks_per_col];

            for (uint32_t b = 0; b < blocks_per_col; ++b) {
                const block_q4_K& blk = col_blocks[b];
                // Header (d, dmin, scales[12]) as one uint4 (E4 trick).
                const auto hdr = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(&blk);
                const auto hsc = [&](int j) {
                    return uint8_t(hdr[1 + (j >> 2)] >> ((j & 3) * 8));
                };
                uint8_t s_raw, m_raw;
                if (sub < 4) {
                    s_raw = hsc(sub)     & 0x3F;
                    m_raw = hsc(sub + 4) & 0x3F;
                } else {
                    s_raw = (hsc(sub + 4) & 0x0F) | ((hsc(sub - 4) >> 6) << 4);
                    m_raw = (hsc(sub + 4) >>   4) | ((hsc(sub    ) >> 6) << 4);
                }
                const float d4  = dev_fp16_to_fp32(uint16_t(hdr[0] & 0xFFFFu)) * float(s_raw);
                const float dm4 = dev_fp16_to_fp32(uint16_t(hdr[0] >> 16))     * float(m_raw);

                // 16 q4 bytes → 32 nibbles; this lane uses one nibble per
                // byte (q4_shift) → 16 quants, packed into 4 dp4a words.
                const auto qv = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(&blk.qs[qs_off]);
                // Per-element global k = b*256 + out_off + i.
                const uint32_t q8_base = (uint32_t(b) * 256 + uint32_t(out_off));
                const uint32_t q8_blk  = q8_base / 32;
                const uint32_t q8_word = (q8_base % 32) / 4;     // 0 or 4

                int32_t idot = 0, isum = 0;
                #pragma unroll
                for (int w = 0; w < 4; ++w) {
                    // Expand 4 nibbles of word w into a packed u8x4.
                    const uint32_t raw = qv[w];
                    const uint32_t nib = (raw >> q4_shift) & 0x0F0F0F0Fu;
                    const int32_t  q8w = int32_t(q8s[q8_blk * 8 + q8_word + w]);
                    idot = ie::dp4a_us(nib, q8w, idot);
                    isum = ie::dp4a_us(0x01010101u, q8w, isum);
                }
                const float d8 = q8d[q8_blk];
                acc += d4 * d8 * float(idot) - dm4 * d8 * float(isum);
            }

            acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
            if (lane == 0) y[n] = sycl::half(acc);
        });
    });
}

// -------------------------------------------------------------------------
// W4A8s family (2026-08-27): identical lattice to the _1x kernels but the
// activation stream is block_q8_1s — the per-16 half sums s0/s1 (stored as
// d*sum floats by quantize_q8_1s) REPLACE the dp4a-ones isum chain, halving
// the inner-loop ALU (the microbench showed the T-loop ALU-bound: weight
// throughput FELL 43.5 -> 19.3 GB/s from T=4 to T=16). All Q4_K x Q8 users
// (decode solo, verify grouped, prefill tiles) move together to the same
// s-form expression `acc += d4*(d8*idot) - dm4*s_half`, keeping the verify
// == decode contract internally consistent.
sycl::event gemv_q4_K_q8s(sycl::queue& q,
                         const void* x_q8s, const void* W_packed,
                         sycl::half* y,
                         uint32_t K, uint32_t N,
                         const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE  = 16;
    constexpr int N_PER_WG = 32;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;

    const auto* W  = static_cast<const block_q4_K*>(W_packed);
    const auto* X8 = static_cast<const block_q8_1s*>(x_q8s);
    const uint32_t blocks_per_col = K / 256;
    const uint32_t q8_blocks      = K / 32;
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "gemv_q4k_q8s", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t wgid  = uint32_t(it.get_group(0));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t n     = wgid * N_PER_WG + sg_id;
            if (n >= N) return;

            const int sub  = int(lane) >> 1;
            const int half = int(lane) & 1;
            const int g     = sub >> 1;
            const int hi_nib = sub & 1;
            const int qs_off = g * 32 + half * 16;
            const int out_off = g * 64 + hi_nib * 32 + half * 16;
            const int q4_shift = hi_nib ? 4 : 0;

            float acc = 0.f;
            const block_q4_K* col_blocks = &W[uint64_t(n) * blocks_per_col];

            for (uint32_t b = 0; b < blocks_per_col; ++b) {
                const block_q4_K& blk = col_blocks[b];
                const auto hdr = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(&blk);
                const auto hsc = [&](int j) {
                    return uint8_t(hdr[1 + (j >> 2)] >> ((j & 3) * 8));
                };
                uint8_t s_raw, m_raw;
                if (sub < 4) {
                    s_raw = hsc(sub)     & 0x3F;
                    m_raw = hsc(sub + 4) & 0x3F;
                } else {
                    s_raw = (hsc(sub + 4) & 0x0F) | ((hsc(sub - 4) >> 6) << 4);
                    m_raw = (hsc(sub + 4) >>   4) | ((hsc(sub    ) >> 6) << 4);
                }
                const float d4  = dev_fp16_to_fp32(uint16_t(hdr[0] & 0xFFFFu)) * float(s_raw);
                const float dm4 = dev_fp16_to_fp32(uint16_t(hdr[0] >> 16))     * float(m_raw);

                const auto qv = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(&blk.qs[qs_off]);
                const uint32_t q8_base = (uint32_t(b) * 256 + uint32_t(out_off));
                const uint32_t q8_blk  = q8_base / 32;
                const uint32_t q8_word = (q8_base % 32) / 4;

                const block_q8_1s& xb = X8[q8_blk];
                const uint32_t* xqw = reinterpret_cast<const uint32_t*>(xb.qs);
                int32_t idot = 0;
                #pragma unroll
                for (int w = 0; w < 4; ++w) {
                    const uint32_t nib = (qv[w] >> q4_shift) & 0x0F0F0F0Fu;
                    idot = ie::dp4a_us(nib, int32_t(xqw[q8_word + w]), idot);
                }
                const float sdot = xb.d * float(idot);
                acc += d4 * sdot - dm4 * (half ? xb.s1 : xb.s0);
            }
            acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
            if (lane == 0) y[n] = sycl::half(acc);
        });
    });
}

sycl::event gemv_q4_K_q8s_grouped_dual(sycl::queue& q,
                         const void* x_q8s,
                         const uint8_t* slot_base, uint64_t slot_bytes,
                         uint64_t gate_off, uint64_t up_off,
                         const int32_t* jobs,
                         sycl::half* g_out, sycl::half* u_out,
                         uint32_t K, uint32_t N, uint32_t P,
                         const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE  = 16;
    constexpr int N_PER_WG = 32;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;

    const auto* X8 = static_cast<const block_q8_1s*>(x_q8s);
    const uint32_t blocks_per_col = K / 256;
    const uint32_t q8_blocks      = K / 32;
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "gemv_q4k_q8s_grp", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(uint64_t(P) * 2 * n_wgs * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t gid   = uint32_t(it.get_group(0));
            const uint32_t pk    = gid / (2 * n_wgs);
            const uint32_t rem   = gid % (2 * n_wgs);
            const bool     is_up = rem >= n_wgs;
            const uint32_t wgid  = is_up ? rem - n_wgs : rem;
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t n     = wgid * N_PER_WG + sg_id;
            if (n >= N) return;

            const uint32_t vt   = uint32_t(jobs[pk * 3 + 0]);
            const uint32_t slot = uint32_t(jobs[pk * 3 + 1]);
            const block_q8_1s* xrow = X8 + uint64_t(vt) * q8_blocks;
            const auto* W = reinterpret_cast<const block_q4_K*>(
                slot_base + uint64_t(slot) * slot_bytes +
                (is_up ? up_off : gate_off));
            sycl::half* y = (is_up ? u_out : g_out) + uint64_t(pk) * N;

            const int sub  = int(lane) >> 1;
            const int half = int(lane) & 1;
            const int g     = sub >> 1;
            const int hi_nib = sub & 1;
            const int qs_off = g * 32 + half * 16;
            const int out_off = g * 64 + hi_nib * 32 + half * 16;
            const int q4_shift = hi_nib ? 4 : 0;

            float acc = 0.f;
            const block_q4_K* col_blocks = &W[uint64_t(n) * blocks_per_col];
            for (uint32_t b = 0; b < blocks_per_col; ++b) {
                const block_q4_K& blk = col_blocks[b];
                const auto hdr = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(&blk);
                const auto hsc = [&](int j) {
                    return uint8_t(hdr[1 + (j >> 2)] >> ((j & 3) * 8));
                };
                uint8_t s_raw, m_raw;
                if (sub < 4) {
                    s_raw = hsc(sub)     & 0x3F;
                    m_raw = hsc(sub + 4) & 0x3F;
                } else {
                    s_raw = (hsc(sub + 4) & 0x0F) | ((hsc(sub - 4) >> 6) << 4);
                    m_raw = (hsc(sub + 4) >>   4) | ((hsc(sub    ) >> 6) << 4);
                }
                const float d4  = dev_fp16_to_fp32(uint16_t(hdr[0] & 0xFFFFu)) * float(s_raw);
                const float dm4 = dev_fp16_to_fp32(uint16_t(hdr[0] >> 16))     * float(m_raw);
                const auto qv = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(&blk.qs[qs_off]);
                const uint32_t q8_base = (uint32_t(b) * 256 + uint32_t(out_off));
                const uint32_t q8_blk  = q8_base / 32;
                const uint32_t q8_word = (q8_base % 32) / 4;
                const block_q8_1s& xb = xrow[q8_blk];
                const uint32_t* xqw = reinterpret_cast<const uint32_t*>(xb.qs);
                int32_t idot = 0;
                #pragma unroll
                for (int w = 0; w < 4; ++w) {
                    const uint32_t nib = (qv[w] >> q4_shift) & 0x0F0F0F0Fu;
                    idot = ie::dp4a_us(nib, int32_t(xqw[q8_word + w]), idot);
                }
                const float sdot = xb.d * float(idot);
                acc += d4 * sdot - dm4 * (half ? xb.s1 : xb.s0);
            }
            acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
            if (lane == 0) y[n] = sycl::half(acc);
        });
    });
}

sycl::event gemv_q4_K_q8s_grouped_tiles(sycl::queue& q,
                         const void* x_q8s_mega,
                         const uint8_t* slot_base, uint64_t slot_bytes,
                         uint64_t gate_off, uint64_t up_off,
                         const int32_t* tile_jobs,
                         sycl::half* g_out, sycl::half* u_out,
                         uint32_t K, uint32_t N, uint32_t J,
                         const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE  = 16;
    constexpr int N_PER_WG = 16;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;
    constexpr int T_MAX     = 16;

    const auto* X8 = static_cast<const block_q8_1s*>(x_q8s_mega);
    const uint32_t blocks_per_col = K / 256;
    const uint32_t q8_blocks      = K / 32;
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "gemv_q4k_q8s_gtile", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(uint64_t(J) * 2 * n_wgs * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t gid   = uint32_t(it.get_group(0));
            const uint32_t j     = gid / (2 * n_wgs);
            const uint32_t rem   = gid % (2 * n_wgs);
            const bool     is_up = rem >= n_wgs;
            const uint32_t wgid  = is_up ? rem - n_wgs : rem;
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t n     = wgid * N_PER_WG + sg_id;
            if (n >= N) return;

            const uint32_t x0   = uint32_t(tile_jobs[j * 4 + 0]);
            const uint32_t T    = uint32_t(tile_jobs[j * 4 + 1]);
            const uint32_t slot = uint32_t(tile_jobs[j * 4 + 2]);
            const uint32_t y0   = uint32_t(tile_jobs[j * 4 + 3]);
            const block_q8_1s* Xj = X8 + uint64_t(x0) * q8_blocks;
            const auto* W = reinterpret_cast<const block_q4_K*>(
                slot_base + uint64_t(slot) * slot_bytes +
                (is_up ? up_off : gate_off));
            sycl::half* y = (is_up ? u_out : g_out) + uint64_t(y0) * N;

            const int sub  = int(lane) >> 1;
            const int half = int(lane) & 1;
            const int g     = sub >> 1;
            const int hi_nib = sub & 1;
            const int qs_off = g * 32 + half * 16;
            const int out_off = g * 64 + hi_nib * 32 + half * 16;
            const int q4_shift = hi_nib ? 4 : 0;

            float acc[T_MAX];
            #pragma unroll
            for (int t = 0; t < T_MAX; ++t) acc[t] = 0.f;

            const block_q4_K* col_blocks = &W[uint64_t(n) * blocks_per_col];
            for (uint32_t b = 0; b < blocks_per_col; ++b) {
                const block_q4_K& blk = col_blocks[b];
                const auto hdr = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(&blk);
                const auto hsc = [&](int jj) {
                    return uint8_t(hdr[1 + (jj >> 2)] >> ((jj & 3) * 8));
                };
                uint8_t s_raw, m_raw;
                if (sub < 4) {
                    s_raw = hsc(sub)     & 0x3F;
                    m_raw = hsc(sub + 4) & 0x3F;
                } else {
                    s_raw = (hsc(sub + 4) & 0x0F) | ((hsc(sub - 4) >> 6) << 4);
                    m_raw = (hsc(sub + 4) >>   4) | ((hsc(sub    ) >> 6) << 4);
                }
                const float d4  = dev_fp16_to_fp32(uint16_t(hdr[0] & 0xFFFFu)) * float(s_raw);
                const float dm4 = dev_fp16_to_fp32(uint16_t(hdr[0] >> 16))     * float(m_raw);

                const auto qv = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(&blk.qs[qs_off]);
                uint32_t nib[4];
                #pragma unroll
                for (int w = 0; w < 4; ++w)
                    nib[w] = (qv[w] >> q4_shift) & 0x0F0F0F0Fu;

                const uint32_t q8_base = (uint32_t(b) * 256 + uint32_t(out_off));
                const uint32_t q8_blk  = q8_base / 32;
                const uint32_t q8_word = (q8_base % 32) / 4;

                #pragma unroll
                for (int t = 0; t < T_MAX; ++t) {
                    if (uint32_t(t) >= T) break;
                    const block_q8_1s& xb = Xj[uint64_t(t) * q8_blocks + q8_blk];
                    const uint32_t* xqw = reinterpret_cast<const uint32_t*>(xb.qs);
                    int32_t idot = 0;
                    #pragma unroll
                    for (int w = 0; w < 4; ++w)
                        idot = ie::dp4a_us(nib[w], int32_t(xqw[q8_word + w]), idot);
                    const float sdot = xb.d * float(idot);
                    acc[t] += d4 * sdot - dm4 * (half ? xb.s1 : xb.s0);
                }
            }
            #pragma unroll
            for (int t = 0; t < T_MAX; ++t) {
                if (uint32_t(t) >= T) break;
                const float r = sycl::reduce_over_group(it.get_sub_group(), acc[t],
                                                        sycl::plus<float>());
                if (lane == 0) y[uint64_t(t) * N + n] = sycl::half(r);
            }
        });
    });
}

// -------------------------------------------------------------------------
// GROUPED-TILES integer-dot W4A8 GEMM for the PREFILL expert-major MoE
// (2026-08-27): ONE launch per layer covers every (expert, 16-row chunk,
// gate|up, column-tile) — the per-expert-chunk launches under-occupied the
// device on the in-order queue (10-40 WGs each; the XMX A/B lost 35% to the
// same effect). Math is gemv_q4_K_q8_batched's VERBATIM (same acc[T_MAX],
// same lattice, same block loop) so every output row is bit-identical to
// the chunked-launch path — a pure occupancy/launch-count change.
// tile_jobs = [J x 4] int32 {x_row0, rows, slot, y_row0}: rows <= 16 q8
// rows starting at x_row0 of the gathered mega stream; outputs land at
// y_row0 of g_out/u_out. Both mats computed per job (grid J x 2 x n_wgs).
sycl::event gemv_q4_K_q8_grouped_tiles(sycl::queue& q,
                         const void* x_q8_mega,
                         const uint8_t* slot_base, uint64_t slot_bytes,
                         uint64_t gate_off, uint64_t up_off,
                         const int32_t* tile_jobs,
                         sycl::half* g_out, sycl::half* u_out,
                         uint32_t K, uint32_t N, uint32_t J,
                         const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE  = 16;
    constexpr int N_PER_WG = 16;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;
    constexpr int T_MAX     = 16;

    const auto* X8 = static_cast<const block_q8_1x*>(x_q8_mega);
    const uint32_t blocks_per_col = K / 256;
    const uint32_t q8_blocks      = K / 32;
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "gemv_q4k_q8_gtile", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(uint64_t(J) * 2 * n_wgs * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t gid   = uint32_t(it.get_group(0));
            const uint32_t j     = gid / (2 * n_wgs);
            const uint32_t rem   = gid % (2 * n_wgs);
            const bool     is_up = rem >= n_wgs;
            const uint32_t wgid  = is_up ? rem - n_wgs : rem;
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t n     = wgid * N_PER_WG + sg_id;
            if (n >= N) return;

            const uint32_t x0   = uint32_t(tile_jobs[j * 4 + 0]);
            const uint32_t T    = uint32_t(tile_jobs[j * 4 + 1]);
            const uint32_t slot = uint32_t(tile_jobs[j * 4 + 2]);
            const uint32_t y0   = uint32_t(tile_jobs[j * 4 + 3]);
            const block_q8_1x* Xj = X8 + uint64_t(x0) * q8_blocks;
            const auto* W = reinterpret_cast<const block_q4_K*>(
                slot_base + uint64_t(slot) * slot_bytes +
                (is_up ? up_off : gate_off));
            sycl::half* y = (is_up ? u_out : g_out) + uint64_t(y0) * N;

            // Lane lattice — identical to gemv_q4_K_q8_batched.
            const int sub  = int(lane) >> 1;
            const int half = int(lane) & 1;
            const int g     = sub >> 1;
            const int hi_nib = sub & 1;
            const int qs_off = g * 32 + half * 16;
            const int out_off = g * 64 + hi_nib * 32 + half * 16;
            const int q4_shift = hi_nib ? 4 : 0;

            float acc[T_MAX];
            #pragma unroll
            for (int t = 0; t < T_MAX; ++t) acc[t] = 0.f;

            const block_q4_K* col_blocks = &W[uint64_t(n) * blocks_per_col];

            for (uint32_t b = 0; b < blocks_per_col; ++b) {
                const block_q4_K& blk = col_blocks[b];
                const auto hdr = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(&blk);
                const auto hsc = [&](int jj) {
                    return uint8_t(hdr[1 + (jj >> 2)] >> ((jj & 3) * 8));
                };
                uint8_t s_raw, m_raw;
                if (sub < 4) {
                    s_raw = hsc(sub)     & 0x3F;
                    m_raw = hsc(sub + 4) & 0x3F;
                } else {
                    s_raw = (hsc(sub + 4) & 0x0F) | ((hsc(sub - 4) >> 6) << 4);
                    m_raw = (hsc(sub + 4) >>   4) | ((hsc(sub    ) >> 6) << 4);
                }
                const float d4  = dev_fp16_to_fp32(uint16_t(hdr[0] & 0xFFFFu)) * float(s_raw);
                const float dm4 = dev_fp16_to_fp32(uint16_t(hdr[0] >> 16))     * float(m_raw);

                const auto qv = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(&blk.qs[qs_off]);
                uint32_t nib[4];
                #pragma unroll
                for (int w = 0; w < 4; ++w)
                    nib[w] = (qv[w] >> q4_shift) & 0x0F0F0F0Fu;

                const uint32_t q8_base = (uint32_t(b) * 256 + uint32_t(out_off));
                const uint32_t q8_blk  = q8_base / 32;
                const uint32_t q8_word = (q8_base % 32) / 4;

                #pragma unroll
                for (int t = 0; t < T_MAX; ++t) {
                    if (uint32_t(t) >= T) break;
                    const block_q8_1x& xb = Xj[uint64_t(t) * q8_blocks + q8_blk];
                    const uint32_t* xqw = reinterpret_cast<const uint32_t*>(xb.qs);
                    int32_t idot = 0, isum = 0;
                    #pragma unroll
                    for (int w = 0; w < 4; ++w) {
                        const int32_t q8w = int32_t(xqw[q8_word + w]);
                        idot = ie::dp4a_us(nib[w], q8w, idot);
                        isum = ie::dp4a_us(0x01010101u, q8w, isum);
                    }
                    acc[t] += d4 * xb.d * float(idot) - dm4 * xb.d * float(isum);
                }
            }

            #pragma unroll
            for (int t = 0; t < T_MAX; ++t) {
                if (uint32_t(t) >= T) break;
                const float r = sycl::reduce_over_group(it.get_sub_group(), acc[t],
                                                        sycl::plus<float>());
                if (lane == 0) y[uint64_t(t) * N + n] = sycl::half(r);
            }
        });
    });
}

// -------------------------------------------------------------------------
// GROUPED integer-dot W4A8 GEMV over an expert-pick job list (spec-verify
// MoE, 2026-08-27). One launch computes gate AND up for EVERY pick: group id
// decodes to (pick, mat, column-tile); within a (pick, mat, column) the solo
// gemv_q4_K_q8 body runs VERBATIM (same SLM Q8 stage of the pick's row, same
// lane lattice, same block loop, same subgroup reduce) -> every output is
// bit-identical to the solo call the decode body would have made. Exists to
// kill the verify launch storm (54 gemv + 27 gather launches/layer -> 1).
sycl::event gemv_q4_K_q8_grouped_dual(sycl::queue& q,
                         const void* x_q8,
                         const uint8_t* slot_base, uint64_t slot_bytes,
                         uint64_t gate_off, uint64_t up_off,
                         const int32_t* jobs,
                         sycl::half* g_out, sycl::half* u_out,
                         uint32_t K, uint32_t N, uint32_t P,
                         const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE  = 16;
    constexpr int N_PER_WG = 32;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;        // 512

    const auto* X8 = static_cast<const block_q8_1x*>(x_q8);
    const uint32_t blocks_per_col = K / 256;
    const uint32_t q8_blocks      = K / 32;
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "gemv_q4k_q8_grp", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<uint32_t, 1> q8s(q8_blocks * 8, h);
        sycl::local_accessor<float, 1>    q8d(q8_blocks, h);

        h.parallel_for(sycl::nd_range<1>(uint64_t(P) * 2 * n_wgs * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t gid   = uint32_t(it.get_group(0));
            const uint32_t p     = gid / (2 * n_wgs);
            const uint32_t rem   = gid % (2 * n_wgs);
            const bool     is_up = rem >= n_wgs;
            const uint32_t wgid  = is_up ? rem - n_wgs : rem;
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;

            const uint32_t vt   = uint32_t(jobs[p * 3 + 0]);
            const uint32_t slot = uint32_t(jobs[p * 3 + 1]);
            const block_q8_1x* xrow = X8 + uint64_t(vt) * q8_blocks;
            const auto* W = reinterpret_cast<const block_q4_K*>(
                slot_base + uint64_t(slot) * slot_bytes +
                (is_up ? up_off : gate_off));
            sycl::half* y = (is_up ? u_out : g_out) + uint64_t(p) * N;

            // Cooperative Q8 stage of THIS pick's activation row — same
            // layout/values the solo kernel stages.
            for (uint32_t i = lid; i < q8_blocks * 8; i += WG_ITEMS) {
                const uint32_t blk = i / 8, w = i % 8;
                q8s[i] = reinterpret_cast<const uint32_t*>(xrow[blk].qs)[w];
            }
            for (uint32_t i = lid; i < q8_blocks; i += WG_ITEMS) q8d[i] = xrow[i].d;
            sycl::group_barrier(it.get_group());

            const uint32_t n = wgid * N_PER_WG + sg_id;
            if (n >= N) return;

            // Lane lattice — identical to gemv_q4_K_q8.
            const int sub  = int(lane) >> 1;
            const int half = int(lane) & 1;
            const int g     = sub >> 1;
            const int hi_nib = sub & 1;
            const int qs_off = g * 32 + half * 16;
            const int out_off = g * 64 + hi_nib * 32 + half * 16;
            const int q4_shift = hi_nib ? 4 : 0;

            float acc = 0.f;
            const block_q4_K* col_blocks = &W[uint64_t(n) * blocks_per_col];

            for (uint32_t b = 0; b < blocks_per_col; ++b) {
                const block_q4_K& blk = col_blocks[b];
                const auto hdr = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(&blk);
                const auto hsc = [&](int j) {
                    return uint8_t(hdr[1 + (j >> 2)] >> ((j & 3) * 8));
                };
                uint8_t s_raw, m_raw;
                if (sub < 4) {
                    s_raw = hsc(sub)     & 0x3F;
                    m_raw = hsc(sub + 4) & 0x3F;
                } else {
                    s_raw = (hsc(sub + 4) & 0x0F) | ((hsc(sub - 4) >> 6) << 4);
                    m_raw = (hsc(sub + 4) >>   4) | ((hsc(sub    ) >> 6) << 4);
                }
                const float d4  = dev_fp16_to_fp32(uint16_t(hdr[0] & 0xFFFFu)) * float(s_raw);
                const float dm4 = dev_fp16_to_fp32(uint16_t(hdr[0] >> 16))     * float(m_raw);

                const auto qv = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(&blk.qs[qs_off]);
                const uint32_t q8_base = (uint32_t(b) * 256 + uint32_t(out_off));
                const uint32_t q8_blk  = q8_base / 32;
                const uint32_t q8_word = (q8_base % 32) / 4;     // 0 or 4

                int32_t idot = 0, isum = 0;
                #pragma unroll
                for (int w = 0; w < 4; ++w) {
                    const uint32_t raw = qv[w];
                    const uint32_t nib = (raw >> q4_shift) & 0x0F0F0F0Fu;
                    const int32_t  q8w = int32_t(q8s[q8_blk * 8 + q8_word + w]);
                    idot = ie::dp4a_us(nib, q8w, idot);
                    isum = ie::dp4a_us(0x01010101u, q8w, isum);
                }
                const float d8 = q8d[q8_blk];
                acc += d4 * d8 * float(idot) - dm4 * d8 * float(isum);
            }

            acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
            if (lane == 0) y[n] = sycl::half(acc);
        });
    });
}

// -------------------------------------------------------------------------
// BATCHED-T integer-dot W4A8 GEMV (spec-decode verify, 2026-06-21).
//
// Same lane lattice as gemv_q4_K_q8 (16 lanes/column, each lane owns 16 quants
// of a super-block split-K over the column's blocks) but keeps T accumulators:
// each weight nibble-word is reconstructed ONCE and dp4a'd against T activation
// rows.  Weight BW amortized over T = the spec-decode verify amortization; the
// per-row Q4_K prefill path instead dequant→XMX-restreams the weight per 16-row
// slice (~53 GB/s, T-independent), so a 4-token verify there costs ~16× a decode
// step.  Activations read from GLOBAL (T rows × K/32 blocks would blow the SLM
// budget at T≥8; the act is tiny + WG-shared → L2-resident).
//
// x_q8: T contiguous block_q8_1x streams (row t at block offset t*(K/32)), from
// quantize_q8_1 over the [T,K] fp16 activations.  Per-row numerics IDENTICAL to
// gemv_q4_K_q8 (same dp4a-ones offset) → keeps the spec loop bit-identical to
// the T==1 decode path (losslessness gate).  y is [T,N] (y[t*N+n]).
sycl::event gemv_q4_K_q8_batched(sycl::queue& q,
                         const void* x_q8, const void* W_packed,
                         sycl::half* y,
                         uint32_t K, uint32_t N, uint32_t T,
                         const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE  = 16;
    constexpr int N_PER_WG = 16;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;        // 512
    constexpr int T_MAX     = 16;                        // matches the wiring THRESH

    const auto* W  = static_cast<const block_q4_K*>(W_packed);
    const auto* X8 = static_cast<const block_q8_1x*>(x_q8);
    const uint32_t blocks_per_col = K / 256;
    const uint32_t q8_blocks      = K / 32;             // q8 blocks per row
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "gemv_q4k_q8_T", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid    = uint32_t(it.get_local_id(0));
            const uint32_t wgid   = uint32_t(it.get_group(0));
            const uint32_t sg_id  = lid / SG_SIZE;
            const uint32_t lane   = lid % SG_SIZE;
            const uint32_t n      = wgid * N_PER_WG + sg_id;
            if (n >= N) return;

            // Lane lattice — identical to gemv_q4_K_q8.
            const int sub  = int(lane) >> 1;
            const int half = int(lane) & 1;
            const int g     = sub >> 1;
            const int hi_nib = sub & 1;
            const int qs_off = g * 32 + half * 16;
            const int out_off = g * 64 + hi_nib * 32 + half * 16;
            const int q4_shift = hi_nib ? 4 : 0;

            float acc[T_MAX];
            #pragma unroll
            for (int t = 0; t < T_MAX; ++t) acc[t] = 0.f;

            const block_q4_K* col_blocks = &W[uint64_t(n) * blocks_per_col];

            for (uint32_t b = 0; b < blocks_per_col; ++b) {
                const block_q4_K& blk = col_blocks[b];
                const auto hdr = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(&blk);
                const auto hsc = [&](int j) {
                    return uint8_t(hdr[1 + (j >> 2)] >> ((j & 3) * 8));
                };
                uint8_t s_raw, m_raw;
                if (sub < 4) {
                    s_raw = hsc(sub)     & 0x3F;
                    m_raw = hsc(sub + 4) & 0x3F;
                } else {
                    s_raw = (hsc(sub + 4) & 0x0F) | ((hsc(sub - 4) >> 6) << 4);
                    m_raw = (hsc(sub + 4) >>   4) | ((hsc(sub    ) >> 6) << 4);
                }
                const float d4  = dev_fp16_to_fp32(uint16_t(hdr[0] & 0xFFFFu)) * float(s_raw);
                const float dm4 = dev_fp16_to_fp32(uint16_t(hdr[0] >> 16))     * float(m_raw);

                // Reconstruct this lane's 4 nibble-words ONCE (weight amortized).
                const auto qv = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(&blk.qs[qs_off]);
                uint32_t nib[4];
                #pragma unroll
                for (int w = 0; w < 4; ++w)
                    nib[w] = (qv[w] >> q4_shift) & 0x0F0F0F0Fu;

                const uint32_t q8_base = (uint32_t(b) * 256 + uint32_t(out_off));
                const uint32_t q8_blk  = q8_base / 32;
                const uint32_t q8_word = (q8_base % 32) / 4;     // 0 or 4

                #pragma unroll
                for (int t = 0; t < T_MAX; ++t) {
                    if (uint32_t(t) >= T) break;
                    const block_q8_1x& xb = X8[uint64_t(t) * q8_blocks + q8_blk];
                    const uint32_t* xqw = reinterpret_cast<const uint32_t*>(xb.qs);
                    int32_t idot = 0, isum = 0;
                    #pragma unroll
                    for (int w = 0; w < 4; ++w) {
                        const int32_t q8w = int32_t(xqw[q8_word + w]);
                        idot = ie::dp4a_us(nib[w], q8w, idot);
                        isum = ie::dp4a_us(0x01010101u, q8w, isum);
                    }
                    acc[t] += d4 * xb.d * float(idot) - dm4 * xb.d * float(isum);
                }
            }

            #pragma unroll
            for (int t = 0; t < T_MAX; ++t) {
                if (uint32_t(t) >= T) break;
                const float r = sycl::reduce_over_group(it.get_sub_group(), acc[t],
                                                        sycl::plus<float>());
                if (lane == 0) y[uint64_t(t) * N + n] = sycl::half(r);
            }
        });
    });
}

// gemv_q4_K_q8s_batched — spec-decode VERIFY kernel, isum-elimination variant.
// IDENTICAL idot (Σ q4·q8) to gemv_q4_K_q8_batched, but the Σq8 ("isum") term is
// READ from the precomputed per-16 half-block sums (block_q8_1s.s0/s1 = d·Σq8 over
// the lane's 16 elements, written by quantize_q8_1s) instead of recomputed by a
// redundant per-column dp4a.  Profiling found the T=4 verify is ALU-bound on dp4a;
// dropping the isum dp4a HALVES the inner dp4a count (4 idot + 4 isum → 4 idot),
// the only ALU lever that survives (XMX at M=4 was 40% SLOWER — dequant+SLM
// overhead + half-wasted TM=8 tile).  A NEW kernel (the int-dot batched gemv any
// winning model might share is untouched).  Activation must be quantize_q8_1s'd.
// Numerics: idot bit-identical; the bias dm4·xb.d·isum becomes dm4·(d·Σq8) — same
// value, last-bit assoc differs from the dp4a path, so this is lossless-GATED.
template <int T_MAX>
static sycl::event gemv_q4_K_q8s_batched_impl(sycl::queue& q,
                         const void* x_q8s, const void* W_packed,
                         sycl::half* y,
                         uint32_t K, uint32_t N, uint32_t T,
                         const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE  = 16;
    constexpr int N_PER_WG = 16;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;        // 512

    const auto* W  = static_cast<const block_q4_K*>(W_packed);
    const auto* X8 = static_cast<const block_q8_1s*>(x_q8s);
    const uint32_t blocks_per_col = K / 256;
    const uint32_t q8_blocks      = K / 32;             // q8 blocks per row
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "gemv_q4k_q8s_T", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid    = uint32_t(it.get_local_id(0));
            const uint32_t wgid   = uint32_t(it.get_group(0));
            const uint32_t sg_id  = lid / SG_SIZE;
            const uint32_t lane   = lid % SG_SIZE;
            const uint32_t n      = wgid * N_PER_WG + sg_id;
            if (n >= N) return;

            // Lane lattice — identical to gemv_q4_K_q8_batched.
            const int sub  = int(lane) >> 1;
            const int half = int(lane) & 1;            // 0 → first 16 (s0), 1 → last 16 (s1)
            const int g     = sub >> 1;
            const int hi_nib = sub & 1;
            const int qs_off = g * 32 + half * 16;
            const int out_off = g * 64 + hi_nib * 32 + half * 16;
            const int q4_shift = hi_nib ? 4 : 0;

            float acc[T_MAX];
            #pragma unroll
            for (int t = 0; t < T_MAX; ++t) acc[t] = 0.f;

            const block_q4_K* col_blocks = &W[uint64_t(n) * blocks_per_col];

            for (uint32_t b = 0; b < blocks_per_col; ++b) {
                const block_q4_K& blk = col_blocks[b];
                const auto hdr = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(&blk);
                const auto hsc = [&](int j) {
                    return uint8_t(hdr[1 + (j >> 2)] >> ((j & 3) * 8));
                };
                uint8_t s_raw, m_raw;
                if (sub < 4) {
                    s_raw = hsc(sub)     & 0x3F;
                    m_raw = hsc(sub + 4) & 0x3F;
                } else {
                    s_raw = (hsc(sub + 4) & 0x0F) | ((hsc(sub - 4) >> 6) << 4);
                    m_raw = (hsc(sub + 4) >>   4) | ((hsc(sub    ) >> 6) << 4);
                }
                const float d4  = dev_fp16_to_fp32(uint16_t(hdr[0] & 0xFFFFu)) * float(s_raw);
                const float dm4 = dev_fp16_to_fp32(uint16_t(hdr[0] >> 16))     * float(m_raw);

                const auto qv = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(&blk.qs[qs_off]);
                uint32_t nib[4];
                #pragma unroll
                for (int w = 0; w < 4; ++w)
                    nib[w] = (qv[w] >> q4_shift) & 0x0F0F0F0Fu;

                const uint32_t q8_base = (uint32_t(b) * 256 + uint32_t(out_off));
                const uint32_t q8_blk  = q8_base / 32;
                const uint32_t q8_word = (q8_base % 32) / 4;

                #pragma unroll
                for (int t = 0; t < T_MAX; ++t) {
                    if (uint32_t(t) >= T) break;
                    const block_q8_1s& xb = X8[uint64_t(t) * q8_blocks + q8_blk];
                    const uint32_t* xqw = reinterpret_cast<const uint32_t*>(xb.qs);
                    int32_t idot = 0;
                    // ONLY idot now (isum dp4a dropped — read the precomputed sum).
                    #pragma unroll
                    for (int w = 0; w < 4; ++w) {
                        const int32_t q8w = int32_t(xqw[q8_word + w]);
                        idot = ie::dp4a_us(nib[w], q8w, idot);
                    }
                    // bias = dm4·(d·Σq8 over the lane's 16 elems) = dm4·(half?s1:s0).
                    const float ssum = (half == 0) ? xb.s0 : xb.s1;
                    acc[t] += d4 * xb.d * float(idot) - dm4 * ssum;
                }
            }

            #pragma unroll
            for (int t = 0; t < T_MAX; ++t) {
                if (uint32_t(t) >= T) break;
                const float r = sycl::reduce_over_group(it.get_sub_group(), acc[t],
                                                        sycl::plus<float>());
                if (lane == 0) y[uint64_t(t) * N + n] = sycl::half(r);
            }
        });
    });
}

// Public entry. T-bucket templating: the [T_MAX] accumulator array is the kernel's
// dominant register pressure, so reserving 16 fp32/lane for a K=4 spec verify
// halves occupancy and throttles the weight read (the profiled bottleneck — this
// gemv is weight-BW-bound, not dp4a-bound). Specializing T_MAX to the smallest
// bucket ≥ T (acc[4] for K=4) frees registers → higher occupancy → faster read.
// Same math → lossless. Only this NEW verify kernel is templated; nothing shared.
sycl::event gemv_q4_K_q8s_batched(sycl::queue& q,
                         const void* x_q8s, const void* W_packed,
                         sycl::half* y,
                         uint32_t K, uint32_t N, uint32_t T,
                         const std::vector<sycl::event>& deps) {
    if (T <= 4)
        return gemv_q4_K_q8s_batched_impl<4 >(q, x_q8s, W_packed, y, K, N, T, deps);
    if (T <= 8)
        return gemv_q4_K_q8s_batched_impl<8 >(q, x_q8s, W_packed, y, K, N, T, deps);
    return gemv_q4_K_q8s_batched_impl<16>(q, x_q8s, W_packed, y, K, N, T, deps);
}

// Integer-dot W8A8 GEMV over a SoA-repacked Q8_0 weight (P3b Q6_K-repack prototype,
// 2026-06-14). The Q6_K ffn_down scalar fallback at K=12288 is the #1 decode cliff
// (140 GB/s, dequant-ALU-bound); repacking it to 8-bit at load kills the 6-bit
// unpack so the GEMV streams weights with only int8×int8 dp4a (no nibble/scale
// machinery). SoA layout (chosen here, not on-disk block_q8_0 whose qs sits at a
// 2-byte offset → misaligned uint32 loads): per output column n,
//   qs_W[n*K + k]                 — int8 quant, column-contiguous (4-aligned, K%4==0)
//   d_W[n*blocks_per_col + b]      — fp16 per-32-block scale
// Each SG (16 lanes) owns one column; lanes stride whole 32-elem blocks.
// Activation = block_q8_1x (quantize_q8_1), staged once in SLM per WG.
sycl::event gemv_q8_0_soa_q8(sycl::queue& q,
                             const void* x_q8, const int8_t* qs_W, const uint16_t* d_W,
                             sycl::half* y,
                             uint32_t K, uint32_t N,
                             const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE  = 16;
    constexpr int N_PER_WG = 16;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;        // 512

    const auto* X8 = static_cast<const block_q8_1x*>(x_q8);
    const uint32_t blocks_per_col = K / 32;
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "gemv_q8_0_soa", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<uint32_t, 1> q8s(blocks_per_col * 8, h);   // act qs words
        sycl::local_accessor<float, 1>    q8d(blocks_per_col, h);       // act d

        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t wgid  = uint32_t(it.get_group(0));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t n     = wgid * N_PER_WG + sg_id;

            for (uint32_t i = lid; i < blocks_per_col * 8; i += WG_ITEMS) {
                const uint32_t blk = i / 8, w = i % 8;
                q8s[i] = reinterpret_cast<const uint32_t*>(X8[blk].qs)[w];
            }
            for (uint32_t i = lid; i < blocks_per_col; i += WG_ITEMS) q8d[i] = X8[i].d;
            sycl::group_barrier(it.get_group());
            if (n >= N) return;

            const int8_t*     wcol = qs_W + uint64_t(n) * K;
            const gguf_half*  dcol = d_W  + uint64_t(n) * blocks_per_col;
            float acc = 0.f;
            for (uint32_t b = lane; b < blocks_per_col; b += SG_SIZE) {
                const uint32_t* wq = reinterpret_cast<const uint32_t*>(wcol + uint64_t(b) * 32);
                int32_t idot = 0;
                #pragma unroll
                for (int w = 0; w < 8; ++w)
                    idot = ie::dp4a_ss(int32_t(wq[w]), int32_t(q8s[b * 8 + w]), idot);
                acc += dev_fp16_to_fp32(dcol[b]) * q8d[b] * float(idot);
            }
            acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
            if (lane == 0) y[n] = sycl::half(acc);
        });
    });
}

// Q8_0-SoA -> F16 dequant into gemv_fp16 layout W[k*N+n]. One work-item per
// weight element, reads coalesced along each row's K (qs[n*K+k]); the strided
// F16 writes are fine for a 63 MB-max scratch fill amortized over a whole
// prefill chunk.
sycl::event dequant_q8_soa_f16(sycl::queue& q,
                               const int8_t* qs_W, const uint16_t* d_W,
                               sycl::half* out,
                               uint32_t K, uint32_t N,
                               const std::vector<sycl::event>& deps) {
    const uint32_t bpc = K / 32;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(uint64_t(N) * K), [=](sycl::id<1> i) {
            const uint32_t n = uint32_t(i / K), k = uint32_t(i % K);
            const sycl::half d =
                sycl::bit_cast<sycl::half>(d_W[uint64_t(n) * bpc + k / 32]);
            out[uint64_t(k) * N + n] =
                sycl::half(float(qs_W[i]) * float(d));
        });
    });
}

// W8A8 SoA GEMV — global-activation variant (NO SLM staging). Identical numerics to
// gemv_q8_0_soa_q8, but reads the (small, L2-resident, all-WGs-shared) activation
// directly from global instead of staging it in SLM. The SLM version's local buffer
// scales with K (blocks_per_col*8 uint32 ≈ 19.6 KB/WG at K=17408 = ffn_down), which
// starves occupancy on large-K projections; dropping it maximizes resident WGs →
// better weight-read bandwidth (the actual decode bottleneck for the dense Q8 27B,
// where the FFN is ~65% of decode). Same dp4a math → bit-identical output.
// Small-K tuned variant: 8 columns/WG, split-K=2 (two SGs share a column,
// SLM-combined). At the qwen4exp decode shapes (K=2560, DRAM-bound) the stock
// 32-col WG underfills the machine: shapes bench 2026-08-31 measured 202.5 ->
// 268.7 GB/s at K=2560,N=10240. Split-K REASSOCIATES the fp32 block-sum
// (kseg-major instead of flat) — last-bit different, PPL-gated like the v2
// kernel. Opt-in via IE_GEMV_SMALLK=1 inside the public entry below.
static sycl::event gemv_q8_0_soa_q8_g_smallk(sycl::queue& q,
                               const void* x_q8, const int8_t* qs_W, const uint16_t* d_W,
                               sycl::half* y,
                               uint32_t K, uint32_t N,
                               const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE = 16, NPW = 8, SK = 2;
    constexpr int WG_ITEMS = NPW * SG_SIZE * SK;
    const auto* X8 = static_cast<const block_q8_1x*>(x_q8);
    const uint32_t blocks_per_col = K / 32;
    const uint32_t n_wgs = (N + NPW - 1) / NPW;
    return ie::ps(q, "gemv_q8_0_soa_g", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<float, 1> part(NPW * SK, h);
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid  = uint32_t(it.get_local_id(0));
            const uint32_t sg   = lid / SG_SIZE;
            const uint32_t lane = lid % SG_SIZE;
            const uint32_t col  = sg / SK;
            const uint32_t kseg = sg % SK;
            const uint32_t n    = uint32_t(it.get_group(0)) * NPW + col;
            if (n >= N) return;
            const int8_t*    wcol = qs_W + uint64_t(n) * K;
            const gguf_half* dcol = d_W + uint64_t(n) * blocks_per_col;
            float acc = 0.f;
            for (uint32_t b = kseg * SG_SIZE + lane; b < blocks_per_col;
                 b += SG_SIZE * SK) {
                const uint32_t* wq = reinterpret_cast<const uint32_t*>(wcol + uint64_t(b) * 32);
                const uint32_t* xq = reinterpret_cast<const uint32_t*>(X8[b].qs);
                int32_t idot = 0;
                #pragma unroll
                for (int w = 0; w < 8; ++w)
                    idot = ie::dp4a_ss(int32_t(wq[w]), int32_t(xq[w]), idot);
                acc += dev_fp16_to_fp32(dcol[b]) * float(X8[b].d) * float(idot);
            }
            acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
            if (lane == 0) part[col * SK + kseg] = acc;
            it.barrier(sycl::access::fence_space::local_space);
            if (kseg == 0 && lane == 0)
                y[n] = sycl::half(part[col * SK + 0] + part[col * SK + 1]);
        });
    });
}

sycl::event gemv_q8_0_soa_q8_g(sycl::queue& q,
                               const void* x_q8, const int8_t* qs_W, const uint16_t* d_W,
                               sycl::half* y,
                               uint32_t K, uint32_t N,
                               const std::vector<sycl::event>& deps) {
    static const bool smallk = std::getenv("IE_GEMV_SMALLK") != nullptr;
    if (smallk && K <= 6144 && (K % 32) == 0)
        return gemv_q8_0_soa_q8_g_smallk(q, x_q8, qs_W, d_W, y, K, N, deps);
    constexpr int SG_SIZE  = 16;
    constexpr int N_PER_WG = 32;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;        // 512

    const auto* X8 = static_cast<const block_q8_1x*>(x_q8);
    const uint32_t blocks_per_col = K / 32;
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "gemv_q8_0_soa_g", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t wgid  = uint32_t(it.get_group(0));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t n     = wgid * N_PER_WG + sg_id;
            if (n >= N) return;

            const int8_t*     wcol = qs_W + uint64_t(n) * K;
            const gguf_half*  dcol = d_W  + uint64_t(n) * blocks_per_col;
            float acc = 0.f;
            for (uint32_t b = lane; b < blocks_per_col; b += SG_SIZE) {
                const uint32_t* wq = reinterpret_cast<const uint32_t*>(wcol + uint64_t(b) * 32);
                const uint32_t* xq = reinterpret_cast<const uint32_t*>(X8[b].qs);
                int32_t idot = 0;
                #pragma unroll
                for (int w = 0; w < 8; ++w)
                    idot = ie::dp4a_ss(int32_t(wq[w]), int32_t(xq[w]), idot);
                acc += dev_fp16_to_fp32(dcol[b]) * float(X8[b].d) * float(idot);
            }
            acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
            if (lane == 0) y[n] = sycl::half(acc);
        });
    });
}

// W8A16 SoA GEMV: F16 activation over the int8 SoA weight — the
// quality-neutral W8 leaf (the 2026-08-27 PPL decomposition: W8 costs
// +0.01%, A8 activation quantization costs the other +1%). Same weight
// stream as the int-dot kernel (still the bound at GEMV shapes); the inner
// dot runs fp32 FMA instead of dp4a. Block b partial = sum(qs_i * x_i) in
// fp32, then * d_b — one accumulation order, shared verbatim with the
// batched rows variant below (per-row bit-identity = the sv contract).
//
// SLM staging (K<=12288): every SG in a WG used to re-read all of x from
// global, so activation traffic was ~N·K·2 bytes = 2× the weight stream
// (2026-08-30 kprof: gemv_q8_soa_f16 = 44% of GLM-5.3 decode GPU time).
// Cooperative SLM load once per WG drops that to (N/32)·K·2. The FMA
// lattice is untouched. K>12288 (24 KiB) keeps the global path so Qwen-size
// ffn_down (K=17408) does not eat occupancy — the W8A8 SLM drop 2026-08-15.
template <bool kSlm, int NC = 32>
static sycl::event gemv_q8_0_soa_f16_g_t(sycl::queue& q,
                                const sycl::half* x, const int8_t* qs_W,
                                const uint16_t* d_W, sycl::half* y,
                                uint32_t K, uint32_t N,
                                const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE  = 16;
    constexpr int N_PER_WG = NC;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;

    const uint32_t blocks_per_col = K / 32;
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "gemv_q8_soa_f16", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<sycl::half, 1> xslm(sycl::range<1>(kSlm ? K : 1u), h);
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t wgid  = uint32_t(it.get_group(0));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t n     = wgid * N_PER_WG + sg_id;

            if constexpr (kSlm) {
                for (uint32_t i = lid; i < K; i += WG_ITEMS) xslm[i] = x[i];
                sycl::group_barrier(it.get_group());
            }
            if (n >= N) return;

            const int8_t*    wcol = qs_W + uint64_t(n) * K;
            const gguf_half* dcol = d_W  + uint64_t(n) * blocks_per_col;
            float acc = 0.f;
            for (uint32_t b = lane; b < blocks_per_col; b += SG_SIZE) {
                const sycl::vec<uint32_t, 4>* w128 =
                    reinterpret_cast<const sycl::vec<uint32_t, 4>*>(
                        wcol + uint64_t(b) * 32);
                const sycl::vec<sycl::half, 8>* xv;
                if constexpr (kSlm) {
                    xv = reinterpret_cast<const sycl::vec<sycl::half, 8>*>(
                        &xslm[uint64_t(b) * 32]);
                } else {
                    xv = reinterpret_cast<const sycl::vec<sycl::half, 8>*>(
                        x + uint64_t(b) * 32);
                }
                const sycl::vec<uint32_t, 4> wa = w128[0], wb2 = w128[1];
                int8_t wv[32];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const uint32_t v = wa[j], u = wb2[j];
                    wv[4 * j + 0]      = int8_t(uint8_t(v));
                    wv[4 * j + 1]      = int8_t(uint8_t(v >> 8));
                    wv[4 * j + 2]      = int8_t(uint8_t(v >> 16));
                    wv[4 * j + 3]      = int8_t(uint8_t(v >> 24));
                    wv[16 + 4 * j + 0] = int8_t(uint8_t(u));
                    wv[16 + 4 * j + 1] = int8_t(uint8_t(u >> 8));
                    wv[16 + 4 * j + 2] = int8_t(uint8_t(u >> 16));
                    wv[16 + 4 * j + 3] = int8_t(uint8_t(u >> 24));
                }
                sycl::vec<sycl::half, 8> xq[4];
                #pragma unroll
                for (int j = 0; j < 4; ++j) xq[j] = xv[j];
                float bacc = 0.f;
                #pragma unroll
                for (int i = 0; i < 32; ++i)
                    bacc += float(wv[i]) * float(xq[i >> 3][i & 7]);
                acc += dev_fp16_to_fp32(dcol[b]) * bacc;
            }
            acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
            if (lane == 0) y[n] = sycl::half(acc);
        });
    });
}

sycl::event gemv_q8_0_soa_f16_g(sycl::queue& q,
                                const sycl::half* x, const int8_t* qs_W,
                                const uint16_t* d_W, sycl::half* y,
                                uint32_t K, uint32_t N,
                                const std::vector<sycl::event>& deps) {
    if (K <= 12288u) {
        // B70 sweep: global activation reads avoid the copy barrier; smaller
        // groups improve these shapes. Preserve the larger ragged-N path.
        if (N <= 256u)
            return gemv_q8_0_soa_f16_g_t<false, 4>(q, x, qs_W, d_W, y, K, N, deps);
        if (N % 32u == 0)
            return gemv_q8_0_soa_f16_g_t<false, 16>(q, x, qs_W, d_W, y, K, N, deps);
        return gemv_q8_0_soa_f16_g_t<true>(q, x, qs_W, d_W, y, K, N, deps);
    }
    return gemv_q8_0_soa_f16_g_t<false>(q, x, qs_W, d_W, y, K, N, deps);
}

// T=1 fused dual: one SG owns column n of BOTH mats and shares x reads.
// Dispatch selects global or SLM staging. Per-mat FMA lattice is the leaf's
// (unpack then i=0..31 then *d) so yA[n]/yB[n] are bit-identical to two
// gemv_q8_0_soa_f16_g calls. The previous rows_dual T=1 path launched
// 2× the grid through rows_impl<4> (N_PER_WG=16, scalar loads) — launch
// fusion only, no x-traffic win, and slower than two vectorized leaves.
template <bool kSlm, int NC = 32>
static sycl::event gemv_q8_0_soa_f16_dual_g_t(sycl::queue& q,
                        const sycl::half* x,
                        const int8_t* qsA, const uint16_t* dA, sycl::half* yA,
                        const int8_t* qsB, const uint16_t* dB, sycl::half* yB,
                        uint32_t K, uint32_t N,
                        const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE  = 16;
    constexpr int N_PER_WG = NC;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;

    const uint32_t blocks_per_col = K / 32;
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "gemv_q8_soa_f16_d", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<sycl::half, 1> xslm(sycl::range<1>(kSlm ? K : 1u), h);
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t wgid  = uint32_t(it.get_group(0));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t n     = wgid * N_PER_WG + sg_id;

            if constexpr (kSlm) {
                for (uint32_t i = lid; i < K; i += WG_ITEMS) xslm[i] = x[i];
                sycl::group_barrier(it.get_group());
            }
            if (n >= N) return;

            const int8_t*    wA = qsA + uint64_t(n) * K;
            const int8_t*    wB = qsB + uint64_t(n) * K;
            const gguf_half* dAcol = dA + uint64_t(n) * blocks_per_col;
            const gguf_half* dBcol = dB + uint64_t(n) * blocks_per_col;
            float accA = 0.f, accB = 0.f;
            for (uint32_t b = lane; b < blocks_per_col; b += SG_SIZE) {
                const sycl::vec<sycl::half, 8>* xv;
                if constexpr (kSlm) {
                    xv = reinterpret_cast<const sycl::vec<sycl::half, 8>*>(
                        &xslm[uint64_t(b) * 32]);
                } else {
                    xv = reinterpret_cast<const sycl::vec<sycl::half, 8>*>(
                        x + uint64_t(b) * 32);
                }
                sycl::vec<sycl::half, 8> xq[4];
                #pragma unroll
                for (int j = 0; j < 4; ++j) xq[j] = xv[j];

                // A then B, sequential — wv live-range is one mat, same
                // register pressure as the leaf plus one extra acc.
                {
                    const sycl::vec<uint32_t, 4>* w128 =
                        reinterpret_cast<const sycl::vec<uint32_t, 4>*>(
                            wA + uint64_t(b) * 32);
                    const sycl::vec<uint32_t, 4> wa = w128[0], wb2 = w128[1];
                    int8_t wv[32];
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const uint32_t v = wa[j], u = wb2[j];
                        wv[4 * j + 0]      = int8_t(uint8_t(v));
                        wv[4 * j + 1]      = int8_t(uint8_t(v >> 8));
                        wv[4 * j + 2]      = int8_t(uint8_t(v >> 16));
                        wv[4 * j + 3]      = int8_t(uint8_t(v >> 24));
                        wv[16 + 4 * j + 0] = int8_t(uint8_t(u));
                        wv[16 + 4 * j + 1] = int8_t(uint8_t(u >> 8));
                        wv[16 + 4 * j + 2] = int8_t(uint8_t(u >> 16));
                        wv[16 + 4 * j + 3] = int8_t(uint8_t(u >> 24));
                    }
                    float bacc = 0.f;
                    #pragma unroll
                    for (int i = 0; i < 32; ++i)
                        bacc += float(wv[i]) * float(xq[i >> 3][i & 7]);
                    accA += dev_fp16_to_fp32(dAcol[b]) * bacc;
                }
                {
                    const sycl::vec<uint32_t, 4>* w128 =
                        reinterpret_cast<const sycl::vec<uint32_t, 4>*>(
                            wB + uint64_t(b) * 32);
                    const sycl::vec<uint32_t, 4> wa = w128[0], wb2 = w128[1];
                    int8_t wv[32];
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const uint32_t v = wa[j], u = wb2[j];
                        wv[4 * j + 0]      = int8_t(uint8_t(v));
                        wv[4 * j + 1]      = int8_t(uint8_t(v >> 8));
                        wv[4 * j + 2]      = int8_t(uint8_t(v >> 16));
                        wv[4 * j + 3]      = int8_t(uint8_t(v >> 24));
                        wv[16 + 4 * j + 0] = int8_t(uint8_t(u));
                        wv[16 + 4 * j + 1] = int8_t(uint8_t(u >> 8));
                        wv[16 + 4 * j + 2] = int8_t(uint8_t(u >> 16));
                        wv[16 + 4 * j + 3] = int8_t(uint8_t(u >> 24));
                    }
                    float bacc = 0.f;
                    #pragma unroll
                    for (int i = 0; i < 32; ++i)
                        bacc += float(wv[i]) * float(xq[i >> 3][i & 7]);
                    accB += dev_fp16_to_fp32(dBcol[b]) * bacc;
                }
            }
            accA = sycl::reduce_over_group(it.get_sub_group(), accA, sycl::plus<float>());
            accB = sycl::reduce_over_group(it.get_sub_group(), accB, sycl::plus<float>());
            if (lane == 0) {
                yA[n] = sycl::half(accA);
                yB[n] = sycl::half(accB);
            }
        });
    });
}

// Known alignment is essential here: plain vec::load from a half pointer
// lowers to sixteen d32 gathers per Q8 block on Xe2. Aligned vector reads
// permit wider memory messages without changing FP32 accumulation order.
// Keep vec::load for valid, element-aligned but non-vector-aligned inputs.
template <bool ALIGNED, typename V>
static inline sycl::vec<V, 16> q8_rows_load16(const V* p) {
    if constexpr (ALIGNED) {
        return *reinterpret_cast<const sycl::vec<V, 16>*>(p);
    } else {
        sycl::vec<V, 16> v;
        v.load(0, sycl::address_space_cast<sycl::access::address_space::global_space,
                                        sycl::access::decorated::no>(p));
        return v;
    }
}

// Batched W8A16 rows (spec-verify / small-T): every weight block is loaded
// once and applied to all T F16 activation rows. Per row: SAME lane->block
// partition (b = lane; b += 16), SAME within-block i-order, SAME
// block-partial*d accumulation, SAME subgroup reduce as gemv_q8_0_soa_f16_g
// -> row t is bit-identical to the T=1 leaf on that row. x = [T, K] rows,
// y = [T, N] rows. T > T_MAX is chunked INSIDE ("never trust callers to
// know a kernel's register cap").
// GRID: 0 = one token tile, 1 = token-major grid, 2 = weight-column-major grid.
// FULL is selected only when every launched tile contains T_MAX rows.
template <int T_MAX, bool ALIGNED, int GRID = 0, bool FULL = false>
static sycl::event gemv_q8_0_soa_f16_rows_impl(sycl::queue& q,
                        const sycl::half* x, const int8_t* qs_W,
                        const uint16_t* d_W, sycl::half* y,
                        uint32_t K, uint32_t N, uint32_t T,
                        const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE  = 16;
    constexpr int N_PER_WG = 16;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;

    const uint32_t blocks_per_col = K / 32;
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "gemv_q8_soa_f16_T", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * (GRID ? (T + T_MAX - 1) / T_MAX : 1) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            uint32_t wgid = uint32_t(it.get_group(0));
            uint32_t t0 = 0;
            if constexpr (GRID == 2) {
                const uint32_t tiles = (T + T_MAX - 1) / T_MAX;
                t0 = (wgid % tiles) * T_MAX;
                wgid /= tiles;
            } else if constexpr (GRID == 1) {
                t0 = (wgid / n_wgs) * T_MAX;
                wgid %= n_wgs;
            }
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t n     = wgid * N_PER_WG + sg_id;
            if (n >= N) return;

            const int8_t*    wcol = qs_W + uint64_t(n) * K;
            const gguf_half* dcol = d_W  + uint64_t(n) * blocks_per_col;
            float acc[T_MAX];
            #pragma unroll
            for (int t = 0; t < T_MAX; ++t) acc[t] = 0.f;
            const uint32_t tn = FULL ? uint32_t(T_MAX) : sycl::min(T - t0, uint32_t(T_MAX));
            for (uint32_t b = lane; b < blocks_per_col; b += SG_SIZE) {
                const int8_t* wb = wcol + uint64_t(b) * 32;
                const float   d  = dev_fp16_to_fp32(dcol[b]);
                // Hoist the int8->fp32 weight converts out of the t-loop —
                // same values, same per-row order (bit-exact vs the leaf),
                // converts amortized T x.
                const auto w0 = q8_rows_load16<ALIGNED>(wb);
                const auto w1 = q8_rows_load16<ALIGNED>(wb + 16);
                float wf[32];
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    wf[i] = float(w0[i]);
                    wf[i + 16] = float(w1[i]);
                }
                #pragma unroll (FULL ? T_MAX : 1)
                for (uint32_t t = 0; t < tn; ++t) {
                    const sycl::half* xb = x + uint64_t(t0 + t) * K + uint64_t(b) * 32;
                    const auto x0 = q8_rows_load16<ALIGNED>(xb);
                    const auto x1 = q8_rows_load16<ALIGNED>(xb + 16);
                    float bacc = 0.f;
                    #pragma unroll
                    for (int i = 0; i < 16; ++i)
                        bacc += wf[i] * float(x0[i]);
                    #pragma unroll
                    for (int i = 0; i < 16; ++i)
                        bacc += wf[i + 16] * float(x1[i]);
                    acc[t] += d * bacc;
                }
            }
            #pragma unroll (FULL ? T_MAX : 1)
            for (uint32_t t = 0; t < tn; ++t) {
                const float r = sycl::reduce_over_group(
                    it.get_sub_group(), acc[t], sycl::plus<float>());
                if (lane == 0) y[uint64_t(t0 + t) * N + n] = sycl::half(r);
            }
        });
    });
}

// DUAL rows variant: two same-shape [K,N] W8 mats (gate+up) against one
// shared F16 activation in ONE launch — first half of the WG range runs A,
// second half B, inner math verbatim from rows_impl per output ->
// bit-identical to two gemv_q8_0_soa_f16_rows calls. Halves the tiny-launch
// count on the SEF=640 shexp leaves (2026-08-27 profile: 80us/call,
// latency-bound).
// GRID: 0 = one token tile, 1 = token-major grid, 2 = weight-column-major grid.
// FULL is selected only when every launched tile contains T_MAX rows.
template <int T_MAX, bool ALIGNED, int GRID = 0, bool FULL = false>
static sycl::event gemv_q8_0_soa_f16_rows_dual_impl(sycl::queue& q,
                        const sycl::half* x,
                        const int8_t* qsA, const uint16_t* dA, sycl::half* yA,
                        const int8_t* qsB, const uint16_t* dB, sycl::half* yB,
                        uint32_t K, uint32_t N, uint32_t T,
                        const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE  = 16;
    constexpr int N_PER_WG = 16;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;

    const uint32_t blocks_per_col = K / 32;
    const uint32_t n_wgs_one = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "gemv_q8_soa_f16_Td", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(uint64_t(2 * n_wgs_one) * (GRID ? (T + T_MAX - 1) / T_MAX : 1) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            uint32_t wgid        = uint32_t(it.get_group(0));
            uint32_t t0 = 0;
            if constexpr (GRID == 2) {
                const uint32_t tiles = (T + T_MAX - 1) / T_MAX;
                t0 = (wgid % tiles) * T_MAX;
                wgid /= tiles;
            } else if constexpr (GRID == 1) {
                t0 = (wgid / (2 * n_wgs_one)) * T_MAX;
                wgid %= 2 * n_wgs_one;
            }
            const bool second    = wgid >= n_wgs_one;
            if (second) wgid -= n_wgs_one;
            const int8_t*    qs_W = second ? qsB : qsA;
            const gguf_half* d_W  = second ? dB : dA;
            sycl::half*      y    = second ? yB : yA;
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t n     = wgid * N_PER_WG + sg_id;
            if (n >= N) return;

            const int8_t*    wcol = qs_W + uint64_t(n) * K;
            const gguf_half* dcol = d_W  + uint64_t(n) * blocks_per_col;
            float acc[T_MAX];
            #pragma unroll
            for (int t = 0; t < T_MAX; ++t) acc[t] = 0.f;
            const uint32_t tn = FULL ? uint32_t(T_MAX) : sycl::min(T - t0, uint32_t(T_MAX));
            for (uint32_t b = lane; b < blocks_per_col; b += SG_SIZE) {
                const int8_t* wb = wcol + uint64_t(b) * 32;
                const float   d  = dev_fp16_to_fp32(dcol[b]);
                const auto w0 = q8_rows_load16<ALIGNED>(wb);
                const auto w1 = q8_rows_load16<ALIGNED>(wb + 16);
                float wf[32];
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    wf[i] = float(w0[i]);
                    wf[i + 16] = float(w1[i]);
                }
                #pragma unroll (FULL ? T_MAX : 1)
                for (uint32_t t = 0; t < tn; ++t) {
                    const sycl::half* xb = x + uint64_t(t0 + t) * K + uint64_t(b) * 32;
                    const auto x0 = q8_rows_load16<ALIGNED>(xb);
                    const auto x1 = q8_rows_load16<ALIGNED>(xb + 16);
                    float bacc = 0.f;
                    #pragma unroll
                    for (int i = 0; i < 16; ++i)
                        bacc += wf[i] * float(x0[i]);
                    #pragma unroll
                    for (int i = 0; i < 16; ++i)
                        bacc += wf[i + 16] * float(x1[i]);
                    acc[t] += d * bacc;
                }
            }
            #pragma unroll (FULL ? T_MAX : 1)
            for (uint32_t t = 0; t < tn; ++t) {
                const float r = sycl::reduce_over_group(
                    it.get_sub_group(), acc[t], sycl::plus<float>());
                if (lane == 0) y[uint64_t(t0 + t) * N + n] = sycl::half(r);
            }
        });
    });
}

sycl::event gemv_q8_0_soa_f16_rows_dual(sycl::queue& q,
                        const sycl::half* x,
                        const int8_t* qsA, const uint16_t* dA, sycl::half* yA,
                        const int8_t* qsB, const uint16_t* dB, sycl::half* yB,
                        uint32_t K, uint32_t N, uint32_t T,
                        const std::vector<sycl::event>& deps) {
    const bool aligned = (reinterpret_cast<uintptr_t>(x) % alignof(sycl::vec<sycl::half, 16>) == 0) &&
        (reinterpret_cast<uintptr_t>(qsA) % alignof(sycl::vec<int8_t, 16>) == 0) &&
        (reinterpret_cast<uintptr_t>(qsB) % alignof(sycl::vec<int8_t, 16>) == 0) && K % 32 == 0;
    if (T == 1 && aligned) {
        if (K <= 12288u) {
            if (N <= 256u)
                return gemv_q8_0_soa_f16_dual_g_t<false, 4>(
                    q, x, qsA, dA, yA, qsB, dB, yB, K, N, deps);
            if (N % 32u == 0)
                return gemv_q8_0_soa_f16_dual_g_t<false, 16>(
                    q, x, qsA, dA, yA, qsB, dB, yB, K, N, deps);
            return gemv_q8_0_soa_f16_dual_g_t<true>(
                q, x, qsA, dA, yA, qsB, dB, yB, K, N, deps);
        }
        return gemv_q8_0_soa_f16_dual_g_t<false>(
            q, x, qsA, dA, yA, qsB, dB, yB, K, N, deps);
    }
    // One event covers all full row tiles. Exact row counts expose independent
    // accumulators without changing FMA order; ragged/unaligned rows retain bounds.
    if (aligned && N > 0 && K >= 32) {
        if (T == 2)
            return gemv_q8_0_soa_f16_rows_dual_impl<2, true, 0, true>(
                q, x, qsA, dA, yA, qsB, dB, yB, K, N, T, deps);
        if (T == 3)
            return gemv_q8_0_soa_f16_rows_dual_impl<3, true, 0, true>(
                q, x, qsA, dA, yA, qsB, dB, yB, K, N, T, deps);
        if (T == 4)
            return gemv_q8_0_soa_f16_rows_dual_impl<4, true, 0, true>(
                q, x, qsA, dA, yA, qsB, dB, yB, K, N, T, deps);
        if (T == 16)
            return gemv_q8_0_soa_f16_rows_dual_impl<16, true, 0, true>(
                q, x, qsA, dA, yA, qsB, dB, yB, K, N, T, deps);
        if (T >= 128 && T % 16 == 0) {
            if (K == 4096 && N == 2048 && T >= 512)
                return gemv_q8_0_soa_f16_rows_dual_impl<16, true, 2, true>(
                    q, x, qsA, dA, yA, qsB, dB, yB, K, N, T, deps);
            return gemv_q8_0_soa_f16_rows_dual_impl<16, true, 1, true>(
                q, x, qsA, dA, yA, qsB, dB, yB, K, N, T, deps);
        }
    }
    if (aligned && T >= 128 && N > 0 && K >= 32) {
        if (K == 4096 && N == 2048 && T >= 512)
            return gemv_q8_0_soa_f16_rows_dual_impl<16, true, 2>(
                q, x, qsA, dA, yA, qsB, dB, yB, K, N, T, deps);
        return gemv_q8_0_soa_f16_rows_dual_impl<16, true, 1>(
            q, x, qsA, dA, yA, qsB, dB, yB, K, N, T, deps);
    }
    auto launch = [&]<bool ALIGNED>() {
        sycl::event ev;
        const bool join_tiles = T > 16 && !q.is_in_order();
        std::vector<sycl::event> tiles;
        if (join_tiles) tiles.reserve((T + 15u) / 16u);
        for (uint32_t t0 = 0; t0 < T; t0 += 16) {
            const uint32_t tn = std::min(16u, T - t0);
            const sycl::half* xr = x + uint64_t(t0) * K;
            if (tn <= 4)
                ev = gemv_q8_0_soa_f16_rows_dual_impl<4, ALIGNED>(q, xr, qsA, dA,
                    yA + uint64_t(t0) * N, qsB, dB, yB + uint64_t(t0) * N, K, N, tn, deps);
            else if (tn <= 8)
                ev = gemv_q8_0_soa_f16_rows_dual_impl<8, ALIGNED>(q, xr, qsA, dA,
                    yA + uint64_t(t0) * N, qsB, dB, yB + uint64_t(t0) * N, K, N, tn, deps);
            else
                ev = gemv_q8_0_soa_f16_rows_dual_impl<16, ALIGNED>(q, xr, qsA, dA,
                    yA + uint64_t(t0) * N, qsB, dB, yB + uint64_t(t0) * N, K, N, tn, deps);
            if (join_tiles) tiles.push_back(ev);
        }
        // On an out-of-order queue the short final tile may finish first.
        return join_tiles ? q.ext_oneapi_submit_barrier(tiles) : ev;
    };
    return aligned ? launch.template operator()<true>()
                   : launch.template operator()<false>();
}

sycl::event gemv_q8_0_soa_f16_rows(sycl::queue& q,
                        const sycl::half* x, const int8_t* qs_W,
                        const uint16_t* d_W, sycl::half* y,
                        uint32_t K, uint32_t N, uint32_t T,
                        const std::vector<sycl::event>& deps) {
    const bool aligned = (reinterpret_cast<uintptr_t>(x) % alignof(sycl::vec<sycl::half, 16>) == 0) &&
        (reinterpret_cast<uintptr_t>(qs_W) % alignof(sycl::vec<int8_t, 16>) == 0) && K % 32 == 0;
    // Exact row counts expose independent accumulators without changing FMA order.
    if (aligned && N > 0 && K >= 32) {
        if (T == 2)
            return gemv_q8_0_soa_f16_rows_impl<2, true, 0, true>(q, x, qs_W, d_W, y, K, N, T, deps);
        if (T == 3)
            return gemv_q8_0_soa_f16_rows_impl<3, true, 0, true>(q, x, qs_W, d_W, y, K, N, T, deps);
        if (T == 4)
            return gemv_q8_0_soa_f16_rows_impl<4, true, 0, true>(q, x, qs_W, d_W, y, K, N, T, deps);
        if (T == 16)
            return gemv_q8_0_soa_f16_rows_impl<16, true, 0, true>(q, x, qs_W, d_W, y, K, N, T, deps);
        if (T >= 128 && T % 16 == 0) {
            if (K >= 8192 && K <= 12288 && N == 4096)
                return gemv_q8_0_soa_f16_rows_impl<16, true, 2, true>(q, x, qs_W, d_W, y, K, N, T, deps);
            if (K > 12288)
                return gemv_q8_0_soa_f16_rows_impl<8, true, 1, true>(q, x, qs_W, d_W, y, K, N, T, deps);
            return gemv_q8_0_soa_f16_rows_impl<16, true, 1, true>(q, x, qs_W, d_W, y, K, N, T, deps);
        }
    }
    if (aligned && T >= 128 && N > 0 && K >= 32) {
        // Column-major ordering helps these contraction sizes; at larger K
        // its cache footprint regresses at T1024, so use a smaller token tile.
        if (K >= 8192 && K <= 12288 && N == 4096)
            return gemv_q8_0_soa_f16_rows_impl<16, true, 2>(q, x, qs_W, d_W, y, K, N, T, deps);
        if (K > 12288)
            return gemv_q8_0_soa_f16_rows_impl<8, true, 1>(q, x, qs_W, d_W, y, K, N, T, deps);
        return gemv_q8_0_soa_f16_rows_impl<16, true, 1>(q, x, qs_W, d_W, y, K, N, T, deps);
    }
    auto launch = [&]<bool ALIGNED>() {
        sycl::event ev;
        const bool join_tiles = T > 16 && !q.is_in_order();
        std::vector<sycl::event> tiles;
        if (join_tiles) tiles.reserve((T + 15u) / 16u);
        for (uint32_t t0 = 0; t0 < T; t0 += 16) {
            const uint32_t tn = std::min(16u, T - t0);
            const sycl::half* xr = x + uint64_t(t0) * K;
            sycl::half*       yr = y + uint64_t(t0) * N;
            if (tn == 1 && ALIGNED)
                ev = gemv_q8_0_soa_f16_g(q, xr, qs_W, d_W, yr, K, N, deps);
            else if (tn <= 4)
                ev = gemv_q8_0_soa_f16_rows_impl<4, ALIGNED>(q, xr, qs_W, d_W, yr, K, N, tn, deps);
            else if (tn <= 8)
                ev = gemv_q8_0_soa_f16_rows_impl<8, ALIGNED>(q, xr, qs_W, d_W, yr, K, N, tn, deps);
            else
                ev = gemv_q8_0_soa_f16_rows_impl<16, ALIGNED>(q, xr, qs_W, d_W, yr, K, N, tn, deps);
            if (join_tiles) tiles.push_back(ev);
        }
        // On an out-of-order queue the short final tile may finish first.
        return join_tiles ? q.ext_oneapi_submit_barrier(tiles) : ev;
    };
    return aligned ? launch.template operator()<true>()
                   : launch.template operator()<false>();
}

// gemv_q8_0_soa_q8_batched — spec-decode VERIFY kernel for the Q8_0-SoA split
// path (2026-08-15). T rows (2..16) share every weight load: v1's exact lane
// map (lane strides whole 32-elem blocks) and per-lane accumulation ORDER, but
// each loaded weight word is dp4a'd against all T activation rows → the weight
// stream (the decode bottleneck) is amortized T×. Activation is read from
// global (T rows of block_q8_1x, [t * K/32 + b]) like the proven Q4_K/Q6_K
// batched verify kernels — T×K SLM staging would blow the 64 KB budget at
// K=17408. T-bucket templating mirrors gemv_q4_K_q8s_batched (acc[T_MAX] is
// the register-pressure knob). NUMERICS: per-row operand set and per-lane
// block order identical to v1 → row t equals the T=1 v1 result; the spec loop
// is lossless-GATED on top (greedy equality).
template <int T_MAX, bool SLM>
static sycl::event gemv_q8_0_soa_q8_batched_impl(sycl::queue& q,
                         const void* x_q8, const int8_t* qs_W, const uint16_t* d_W,
                         sycl::half* y,
                         uint32_t K, uint32_t N, uint32_t T,
                         const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE  = 16;
    constexpr int N_PER_WG = 16;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;        // 512

    const auto* X8 = static_cast<const block_q8_1x*>(x_q8);
    const uint32_t blocks_per_col = K / 32;
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "gemv_q8_soa_T", [&](sycl::handler& h) {
        h.depends_on(deps);
        // SLM=true: stage the T activation streams (qs words + d floats) once per
        // WG — the T=1 kernel's documented 10×-over-global lesson, applied to the
        // batched verify. Same bytes, same per-lane order → bit-identical.
        sycl::local_accessor<uint32_t, 1> xs(SLM ? uint64_t(T) * blocks_per_col * 8 : 1, h);
        sycl::local_accessor<float, 1>    xd(SLM ? uint64_t(T) * blocks_per_col     : 1, h);
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t wgid  = uint32_t(it.get_group(0));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t n     = wgid * N_PER_WG + sg_id;
            if constexpr (SLM) {
                const uint32_t nb = T * blocks_per_col;
                for (uint32_t i = lid; i < nb * 8; i += WG_ITEMS) {
                    const uint32_t blk = i / 8, w = i % 8;
                    xs[i] = reinterpret_cast<const uint32_t*>(X8[blk].qs)[w];
                }
                for (uint32_t i = lid; i < nb; i += WG_ITEMS) xd[i] = X8[i].d;
                sycl::group_barrier(it.get_group());
            }
            if (n >= N) return;

            const int8_t*    wcol = qs_W + uint64_t(n) * K;
            const gguf_half* dcol = d_W  + uint64_t(n) * blocks_per_col;

            float acc[T_MAX];
            #pragma unroll
            for (int t = 0; t < T_MAX; ++t) acc[t] = 0.f;

            for (uint32_t b = lane; b < blocks_per_col; b += SG_SIZE) {
                const uint32_t* wq = reinterpret_cast<const uint32_t*>(wcol + uint64_t(b) * 32);
                uint32_t w8[8];
                #pragma unroll
                for (int w = 0; w < 8; ++w) w8[w] = wq[w];
                const float dw = dev_fp16_to_fp32(dcol[b]);
                #pragma unroll
                for (int t = 0; t < T_MAX; ++t) {
                    if (uint32_t(t) >= T) break;
                    const uint64_t xi = uint64_t(t) * blocks_per_col + b;
                    int32_t idot = 0;
                    if constexpr (SLM) {
                        #pragma unroll
                        for (int w = 0; w < 8; ++w)
                            idot = ie::dp4a_ss(int32_t(w8[w]), int32_t(xs[xi * 8 + w]), idot);
                        acc[t] += dw * xd[xi] * float(idot);
                    } else {
                        const block_q8_1x& xb = X8[xi];
                        const uint32_t* xqw = reinterpret_cast<const uint32_t*>(xb.qs);
                        #pragma unroll
                        for (int w = 0; w < 8; ++w)
                            idot = ie::dp4a_ss(int32_t(w8[w]), int32_t(xqw[w]), idot);
                        acc[t] += dw * float(xb.d) * float(idot);
                    }
                }
            }
            #pragma unroll
            for (int t = 0; t < T_MAX; ++t) {
                if (uint32_t(t) >= T) break;
                const float r = sycl::reduce_over_group(it.get_sub_group(), acc[t],
                                                        sycl::plus<float>());
                if (lane == 0) y[uint64_t(t) * N + n] = sycl::half(r);
            }
        });
    });
}

// SLM-vs-global choice: stage when the T activation streams fit the budget.
// Default 24 KB — measured 2026-08-16 (T=3): staging the small-K mats (K=5120:
// 17 KB) wins ~1.9 ms/rd, but staging ffn_down too (K=17408: 58.7 KB/WG) gives
// back ~0.4 ms (one WG/Xe-core starves the weight read). IE_Q8BAT_SLM_KB
// overrides for A/B; 0 disables staging.
static inline bool q8bat_use_slm(uint32_t K, uint32_t T) {
    static const uint32_t kb = []{
        const char* e = std::getenv("IE_Q8BAT_SLM_KB");
        return e ? uint32_t(std::atoi(e)) : 24u;
    }();
    return uint64_t(T) * (K / 32) * 36 <= uint64_t(kb) * 1024;
}

// Dual-weight twin of the batched impl: identical inner loop per output; the
// WG range's first half computes A, the second half B (uniform branch per WG).
template <int T_MAX, bool SLM>
static sycl::event gemv_q8_0_soa_q8_batched_dual_impl(sycl::queue& q,
                         const void* x_q8,
                         const int8_t* qsA, const uint16_t* dA, sycl::half* yA,
                         const int8_t* qsB, const uint16_t* dB, sycl::half* yB,
                         uint32_t K, uint32_t N, uint32_t T,
                         const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE  = 16;
    constexpr int N_PER_WG = 16;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;        // 512

    const auto* X8 = static_cast<const block_q8_1x*>(x_q8);
    const uint32_t blocks_per_col = K / 32;
    const uint32_t n_wgs_one = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "gemv_q8_soa_T2x", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<uint32_t, 1> xs(SLM ? uint64_t(T) * blocks_per_col * 8 : 1, h);
        sycl::local_accessor<float, 1>    xd(SLM ? uint64_t(T) * blocks_per_col     : 1, h);
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs_one) * 2 * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t wgid  = uint32_t(it.get_group(0));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const bool     isB   = wgid >= n_wgs_one;
            const uint32_t n     = (isB ? wgid - n_wgs_one : wgid) * N_PER_WG + sg_id;
            if constexpr (SLM) {
                const uint32_t nb = T * blocks_per_col;
                for (uint32_t i = lid; i < nb * 8; i += WG_ITEMS) {
                    const uint32_t blk = i / 8, w = i % 8;
                    xs[i] = reinterpret_cast<const uint32_t*>(X8[blk].qs)[w];
                }
                for (uint32_t i = lid; i < nb; i += WG_ITEMS) xd[i] = X8[i].d;
                sycl::group_barrier(it.get_group());
            }
            if (n >= N) return;
            const int8_t*   qs_W = isB ? qsB : qsA;
            const uint16_t* d_W  = isB ? dB  : dA;
            sycl::half*     y    = isB ? yB  : yA;

            const int8_t*    wcol = qs_W + uint64_t(n) * K;
            const gguf_half* dcol = d_W  + uint64_t(n) * blocks_per_col;

            float acc[T_MAX];
            #pragma unroll
            for (int t = 0; t < T_MAX; ++t) acc[t] = 0.f;

            for (uint32_t b = lane; b < blocks_per_col; b += SG_SIZE) {
                const uint32_t* wq = reinterpret_cast<const uint32_t*>(wcol + uint64_t(b) * 32);
                uint32_t w8[8];
                #pragma unroll
                for (int w = 0; w < 8; ++w) w8[w] = wq[w];
                const float dw = dev_fp16_to_fp32(dcol[b]);
                #pragma unroll
                for (int t = 0; t < T_MAX; ++t) {
                    if (uint32_t(t) >= T) break;
                    const uint64_t xi = uint64_t(t) * blocks_per_col + b;
                    int32_t idot = 0;
                    if constexpr (SLM) {
                        #pragma unroll
                        for (int w = 0; w < 8; ++w)
                            idot = ie::dp4a_ss(int32_t(w8[w]), int32_t(xs[xi * 8 + w]), idot);
                        acc[t] += dw * xd[xi] * float(idot);
                    } else {
                        const block_q8_1x& xb = X8[xi];
                        const uint32_t* xqw = reinterpret_cast<const uint32_t*>(xb.qs);
                        #pragma unroll
                        for (int w = 0; w < 8; ++w)
                            idot = ie::dp4a_ss(int32_t(w8[w]), int32_t(xqw[w]), idot);
                        acc[t] += dw * float(xb.d) * float(idot);
                    }
                }
            }
            #pragma unroll
            for (int t = 0; t < T_MAX; ++t) {
                if (uint32_t(t) >= T) break;
                const float r = sycl::reduce_over_group(it.get_sub_group(), acc[t],
                                                        sycl::plus<float>());
                if (lane == 0) y[uint64_t(t) * N + n] = sycl::half(r);
            }
        });
    });
}

sycl::event gemv_q8_0_soa_q8_batched_dual(sycl::queue& q, const void* x_q8,
                                     const int8_t* qsA, const uint16_t* dA,
                                     sycl::half* yA,
                                     const int8_t* qsB, const uint16_t* dB,
                                     sycl::half* yB,
                                     uint32_t K, uint32_t N, uint32_t T,
                                     const std::vector<sycl::event>& deps) {
    if (T > 16) {
        sycl::event e;
        const uint64_t row = uint64_t(K / 32) * sizeof(block_q8_1x);
        for (uint32_t t0 = 0; t0 < T; t0 += 16)
            e = gemv_q8_0_soa_q8_batched_dual(q,
                    static_cast<const uint8_t*>(x_q8) + uint64_t(t0) * row,
                    qsA, dA, yA + uint64_t(t0) * N,
                    qsB, dB, yB + uint64_t(t0) * N, K, N,
                    std::min(16u, T - t0),
                    t0 == 0 ? deps : std::vector<sycl::event>{});
        return e;
    }
    const bool slm = q8bat_use_slm(K, T);
    if (T == 2)  return slm ? gemv_q8_0_soa_q8_batched_dual_impl<2 , true>(q, x_q8, qsA, dA, yA, qsB, dB, yB, K, N, T, deps)
                            : gemv_q8_0_soa_q8_batched_dual_impl<2 , false>(q, x_q8, qsA, dA, yA, qsB, dB, yB, K, N, T, deps);
    if (T == 3)  return slm ? gemv_q8_0_soa_q8_batched_dual_impl<3 , true>(q, x_q8, qsA, dA, yA, qsB, dB, yB, K, N, T, deps)
                            : gemv_q8_0_soa_q8_batched_dual_impl<3 , false>(q, x_q8, qsA, dA, yA, qsB, dB, yB, K, N, T, deps);
    if (T <= 4)  return slm ? gemv_q8_0_soa_q8_batched_dual_impl<4 , true>(q, x_q8, qsA, dA, yA, qsB, dB, yB, K, N, T, deps)
                            : gemv_q8_0_soa_q8_batched_dual_impl<4 , false>(q, x_q8, qsA, dA, yA, qsB, dB, yB, K, N, T, deps);
    if (T <= 8)  return gemv_q8_0_soa_q8_batched_dual_impl<8 , false>(q, x_q8, qsA, dA, yA, qsB, dB, yB, K, N, T, deps);
    return gemv_q8_0_soa_q8_batched_dual_impl<16, false>(q, x_q8, qsA, dA, yA, qsB, dB, yB, K, N, T, deps);
}

sycl::event gemv_q8_0_soa_q8_batched(sycl::queue& q,
                                     const void* x_q8, const int8_t* qs_W,
                                     const uint16_t* d_W, sycl::half* y,
                                     uint32_t K, uint32_t N, uint32_t T,
                                     const std::vector<sycl::event>& deps) {
    // Defense in depth: T beyond the largest bucket LOOPS in 16-chunks — the
    // acc[16] overrun was silent corruption -> DEVICE_LOST -> machine crash
    // when a caller passed a prefill chunk (2026-08-27). Never trust callers
    // to know a kernel's register cap.
    if (T > 16) {
        sycl::event e;
        const uint64_t row = uint64_t(K / 32) * sizeof(block_q8_1x);
        for (uint32_t t0 = 0; t0 < T; t0 += 16)
            e = gemv_q8_0_soa_q8_batched(q,
                    static_cast<const uint8_t*>(x_q8) + uint64_t(t0) * row,
                    qs_W, d_W, y + uint64_t(t0) * N, K, N,
                    std::min(16u, T - t0),
                    t0 == 0 ? deps : std::vector<sycl::event>{});
        return e;
    }
    // Exact small-T buckets: T=3 (spec K=3) in the T_MAX=4 build wastes an acc
    // register + a dead unroll branch; the smallest bucket frees registers →
    // higher occupancy → faster weight read (same lesson as the Q4K dispatcher).
    const bool slm = q8bat_use_slm(K, T);
    if (T == 2)  return slm ? gemv_q8_0_soa_q8_batched_impl<2 , true>(q, x_q8, qs_W, d_W, y, K, N, T, deps)
                            : gemv_q8_0_soa_q8_batched_impl<2 , false>(q, x_q8, qs_W, d_W, y, K, N, T, deps);
    if (T == 3)  return slm ? gemv_q8_0_soa_q8_batched_impl<3 , true>(q, x_q8, qs_W, d_W, y, K, N, T, deps)
                            : gemv_q8_0_soa_q8_batched_impl<3 , false>(q, x_q8, qs_W, d_W, y, K, N, T, deps);
    if (T <= 4)  return slm ? gemv_q8_0_soa_q8_batched_impl<4 , true>(q, x_q8, qs_W, d_W, y, K, N, T, deps)
                            : gemv_q8_0_soa_q8_batched_impl<4 , false>(q, x_q8, qs_W, d_W, y, K, N, T, deps);
    if (T <= 8)  return gemv_q8_0_soa_q8_batched_impl<8 , false>(q, x_q8, qs_W, d_W, y, K, N, T, deps);
    return gemv_q8_0_soa_q8_batched_impl<16, false>(q, x_q8, qs_W, d_W, y, K, N, T, deps);
}

// gemv_q8_0_soa_q8_v2 — COALESCED-LOAD W8A8 SoA GEMV (2026-08-15).
//
// The Q8 sibling of gemv_q6_soa_q8_v2 / gemv_q4_K_soa_q8_v2. v1 (above) strides
// WHOLE 32-elem blocks (lane b += SG_SIZE): one subgroup instruction gathers 16
// words 32 B apart → a 512 B scatter across 8 half-used cache lines, and its SLM
// staging scales with K (~19.6 KB/WG at K=17408 = ffn_down) which starves
// occupancy exactly on the big FFN projections that dominate decode.
//
// LANE MAP (per pass = TWO consecutive 32-elem blocks = 64 B of int8 weights =
// 16 uint32 words): lane L owns word (L & 7) of block b0 + (L >> 3), i.e. the
// 16 lane loads are 16 CONSECUTIVE uint32 = one 64 B line, fully
// subgroup-coalesced. No SLM, no barrier — the q8 activation is read from
// global/L2 like the proven Q6/Q4K v2 kernels (the _g A/B only regressed
// because v1's scattered weight loads dominated; coalesced loads flip that).
//
// NUMERICS: identical per-element int-dot operands to v1 (same dp4a words, same
// d_w·d8 scaling); only the fp accumulation GROUPING differs — each lane sums
// half-blocks instead of whole blocks (last-bit reassoc). PPL gate must hold.
// Requires K % 64 == 0 (blocks_per_col even) — true for every qwen35 projection
// (5120/6144/12288/17408); falls back to v1 otherwise.
sycl::event gemv_q8_0_soa_q8_v2(sycl::queue& q,
                                const void* x_q8, const int8_t* qs_W, const uint16_t* d_W,
                                sycl::half* y,
                                uint32_t K, uint32_t N,
                                const std::vector<sycl::event>& deps) {
    if ((K / 32) & 1u)   // odd block count — keep the safe path
        return gemv_q8_0_soa_q8(q, x_q8, qs_W, d_W, y, K, N, deps);

    constexpr int SG_SIZE  = 16;
    constexpr int N_PER_WG = 16;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;        // 512

    const auto* X8 = static_cast<const block_q8_1x*>(x_q8);
    const uint32_t blocks_per_col = K / 32;
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "gemv_q8_0_soa_v2", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t wgid  = uint32_t(it.get_group(0));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t n     = wgid * N_PER_WG + sg_id;
            if (n >= N) return;

            const uint32_t*   wcol = reinterpret_cast<const uint32_t*>(qs_W + uint64_t(n) * K);
            const gguf_half*  dcol = d_W + uint64_t(n) * blocks_per_col;
            const uint32_t wsub = lane & 7u;    // word within the lane's block
            const uint32_t bsub = lane >> 3;    // which of the pass's two blocks

            float acc = 0.f;
            for (uint32_t b0 = 0; b0 < blocks_per_col; b0 += 2) {
                const uint32_t blk = b0 + bsub;
                const uint32_t wq  = wcol[b0 * 8 + lane];   // 16 consecutive words = 64 B/SG
                const block_q8_1x& xb = X8[blk];
                const uint32_t xq = reinterpret_cast<const uint32_t*>(xb.qs)[wsub];
                const int32_t idot = ie::dp4a_ss(int32_t(wq), int32_t(xq), 0);
                acc += dev_fp16_to_fp32(dcol[blk]) * float(xb.d) * float(idot);
            }
            acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
            if (lane == 0) y[n] = sycl::half(acc);
        });
    });
}

}  // namespace ie
