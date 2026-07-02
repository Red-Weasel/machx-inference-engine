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
#include "ie/dp4a.hpp"
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
    constexpr int N_PER_WG = 32;
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
    constexpr int N_PER_WG = 32;
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
    constexpr int N_PER_WG = 32;
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

// W8A8 SoA GEMV — global-activation variant (NO SLM staging). Identical numerics to
// gemv_q8_0_soa_q8, but reads the (small, L2-resident, all-WGs-shared) activation
// directly from global instead of staging it in SLM. The SLM version's local buffer
// scales with K (blocks_per_col*8 uint32 ≈ 19.6 KB/WG at K=17408 = ffn_down), which
// starves occupancy on large-K projections; dropping it maximizes resident WGs →
// better weight-read bandwidth (the actual decode bottleneck for the dense Q8 27B,
// where the FFN is ~65% of decode). Same dp4a math → bit-identical output.
sycl::event gemv_q8_0_soa_q8_g(sycl::queue& q,
                               const void* x_q8, const int8_t* qs_W, const uint16_t* d_W,
                               sycl::half* y,
                               uint32_t K, uint32_t N,
                               const std::vector<sycl::event>& deps) {
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

}  // namespace ie
