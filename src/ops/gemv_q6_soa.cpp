// src/ops/gemv_q6_soa.cpp — fast Q6_K decode GEMV via a load-time SoA-Q6 repack
// + int-dot (dp4a, W6A8) inner loop with split-K.  2026-06-20.
//
// WHY: the on-disk AoS block_q6_K (210 B, interleaved ql/qh/scales) forces the
// decode GEMV (gemv_q6_K_slm / _ktiled in gemv_q6k.cpp) into strided per-lane
// byte loads + a complex 8-quadrant lane lattice; at K=12288 (ffn_down) that
// floors it at ~9% of B70's 608 GB/s (the 27B Q6 single-GPU decode = 5 tok/s,
// 97% of which is the Q6_K matmul).  This file mirrors the proven Q8-SoA int-dot
// win (gemv_q8_0_soa_q8, ~80% BW) at 6.5 bpw: weights are repacked at load into
// per-column NATURAL-ORDER bit-plane streams (q6_lo 4-bit, q6_hi 2-bit, per-16
// int8 scales, fp16 super-scale), so the kernel does coalesced loads + a cheap
// 6-bit reconstruct + int8 dp4a.  Split-K: SG_SIZE lanes per column stride whole
// 256-elem super-blocks (the K-reduction the old per-column-serial kernel lacks).
//
// SoA-Q6 layout (per output column n, contiguous; mirrors qwen35_split's Q8-SoA
//   column-major-by-output-n discipline so a column-slice stays a pure byte cut):
//   q6_lo[n*(K/2) + ...]   uint8 — low nibble of element k, packed 2 elems/byte
//                          in natural k order: byte (k/2) holds elem 2j in bits
//                          0..3 and elem 2j+1 in bits 4..7.
//   q6_hi[n*(K/4) + ...]   uint8 — high 2 bits of element k, packed 4 elems/byte:
//                          byte (k/4) holds elem 4j+r in bits (2r)..(2r+1).
//   q6_sc[n*(K/16) + ...]  int8  — per-16-element signed scale (natural order).
//   q6_d [n*(K/256) + ...] fp16  — per-256 super-block scale.
// Footprint: K/2 + K/4 + K/16 + (K/256)*2 bytes/col = 0.5+0.25+0.0625+0.0078
//   ≈ 0.820 B/elem = ~6.56 bpw → the 22 GB Q6 GGUF stays ~22 GB (single-card).
//
// NUMERICS: int-dot W6A8 is NOT bit-exact vs fp16-dequant Q6_K (act rounds to
// int8 ~0.4% RMS; fp fold order differs).  Gated on the engine's PPL/cosine
// checks, exactly like every other int-dot decode kernel here.

#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"

#include <cstdlib>
#include <sycl/sycl.hpp>
#include "ie/dp4a.hpp"
#include "ie/kernel_profiler.hpp"

namespace ie {

namespace {

inline float soa_fp16_to_fp32(uint16_t h) {
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

// Host-side repack of a canonical block_q6_K weight [K (contiguous), N] into the
// SoA-Q6 bit-plane streams above. `W_blocks` = the GGUF tensor data (N columns,
// each K/256 contiguous block_q6_K). Outputs are caller-allocated host buffers:
//   q6_lo : N * (K/2) bytes, q6_hi : N * (K/4) bytes,
//   q6_sc : N * (K/16) int8,  q6_d  : N * (K/256) uint16 (raw fp16).
// Element order is NATURAL (k = 0..K-1), matching the q8 activation stream.
// Mirrors dequant_q6_K_row's de-interleave (ggml dequantize_row_q6_K), but keeps
// the value 6-bit (lo/hi split) instead of dequantizing to fp32.
void repack_q6_K_to_soa(const void* W_blocks, uint32_t K, uint32_t N,
                        uint8_t* q6_lo, uint8_t* q6_hi,
                        int8_t* q6_sc, uint16_t* q6_d) {
    const uint32_t bpc = K / 256;                         // q6_K blocks per column
    const auto* blocks = static_cast<const block_q6_K*>(W_blocks);
    for (uint64_t n = 0; n < N; ++n) {
        const block_q6_K* col = blocks + n * bpc;
        uint8_t*  lo_col = q6_lo + n * (uint64_t(K) / 2);
        uint8_t*  hi_col = q6_hi + n * (uint64_t(K) / 4);
        int8_t*   sc_col = q6_sc + n * (uint64_t(K) / 16);
        uint16_t* d_col  = q6_d  + n * (uint64_t(K) / 256);
        for (uint32_t b = 0; b < bpc; ++b) {
            const block_q6_K& blk = col[b];
            d_col[b] = *reinterpret_cast<const uint16_t*>(&blk.d);
            // Natural-order base for this super-block.
            const uint32_t kb = b * 256;
            // 16 per-16 scales (natural order): scale index is = (out)/16.
            // The ggml de-interleave below writes outputs 0..255; scale[out/16].
            for (int i = 0; i < 16; ++i) sc_col[(kb >> 4) + i] = blk.scales[i];
            // De-interleave the 8 quadrants into natural element order, emitting
            // the 6-bit value as (lo nibble, hi 2-bit) bit-planes. Mirror
            // dequant_q6_K_row exactly so the unsigned quant (q6u = lo|hi<<4,
            // value = q6u - 32) reconstructs identically.
            const uint8_t* ql = blk.ql;
            const uint8_t* qh = blk.qh;
            for (int half = 0; half < 256; half += 128) {
                for (int l = 0; l < 32; ++l) {
                    const uint8_t qhl = qh[l];
                    const uint8_t q1 = uint8_t( (ql[l +  0] & 0x0F) | (((qhl >> 0) & 3) << 4) );
                    const uint8_t q2 = uint8_t( (ql[l + 32] & 0x0F) | (((qhl >> 2) & 3) << 4) );
                    const uint8_t q3 = uint8_t( (ql[l +  0] >>  4 ) | (((qhl >> 4) & 3) << 4) );
                    const uint8_t q4 = uint8_t( (ql[l + 32] >>  4 ) | (((qhl >> 6) & 3) << 4) );
                    const uint32_t o0 = uint32_t(half) + l +  0;
                    const uint32_t o1 = uint32_t(half) + l + 32;
                    const uint32_t o2 = uint32_t(half) + l + 64;
                    const uint32_t o3 = uint32_t(half) + l + 96;
                    auto emit = [&](uint32_t out, uint8_t q6u) {
                        const uint32_t k = kb + out;
                        // low nibble → q6_lo[k/2], 2 elems/byte
                        const uint32_t loi = k >> 1;
                        const uint8_t  loshift = uint8_t((k & 1) * 4);
                        lo_col[loi] = uint8_t((lo_col[loi] & ~(0x0Fu << loshift)) |
                                              (uint32_t(q6u & 0x0F) << loshift));
                        // high 2 bits → q6_hi[k/4], 4 elems/byte
                        const uint32_t hii = k >> 2;
                        const uint8_t  hishift = uint8_t((k & 3) * 2);
                        hi_col[hii] = uint8_t((hi_col[hii] & ~(0x03u << hishift)) |
                                              (uint32_t((q6u >> 4) & 0x03) << hishift));
                    };
                    emit(o0, q1); emit(o1, q2); emit(o2, q3); emit(o3, q4);
                }
                ql += 64; qh += 32;
            }
        }
    }
}

// W6A8 GEMV over the SoA-Q6 streams.  Mirrors gemv_q8_0_soa_q8's occupancy
// structure (SG_SIZE=16, N_PER_WG=32, Q8_1 activation staged once in SLM per WG)
// — the proven ~80% BW template.  Split-K: the 16 lanes of an SG stride whole
// 256-elem super-blocks of one column (lane b += 16), each lane accumulates its
// super-blocks, then an SG reduce sums the column.  Per super-block: read the
// 4-bit lo plane (64 bytes = 16 uint32), the 2-bit hi plane (32 bytes = 8
// uint32), reconstruct 4 q6u/word, dp4a vs the staged q8, fold the per-16 int8
// scale × fp16 super-scale × act-scale, and subtract the 32-offset via the
// ones-dp4a (acc += d6·d8·(Σ q6u·q8 − 32·Σ q8)).
template <bool USE_SLM>
static sycl::event gemv_q6_soa_q8_impl(sycl::queue& q,
                           const void* x_q8,
                           const uint8_t* q6_lo, const uint8_t* q6_hi,
                           const int8_t* q6_sc, const uint16_t* q6_d,
                           sycl::half* y,
                           uint32_t K, uint32_t N,
                           const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE  = 16;
    constexpr int N_PER_WG = 32;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;        // 512

    const auto* X8 = static_cast<const block_q8_1x*>(x_q8);
    const uint32_t blocks_per_col = K / 256;            // super-blocks/col
    const uint32_t q8_blocks      = K / 32;             // q8 blocks (8 per super-block)
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "gemv_q6_soa", [&](sycl::handler& h) {
        h.depends_on(deps);
        // Stage the Q8 activation stream once per WG (shared by all 32 columns):
        // qs words (8 uint32/block) + per-block d.  K/32 blocks × 36 B ≈ 13.5 KiB
        // at K=12288 — within the 128 KiB WG SLM budget for any decode K.  The
        // global variant (USE_SLM=false) drops the staging+barrier and reads the
        // (small, L2-resident, WG-shared) activation directly — higher occupancy
        // on large-K projections (the gemv_q8_0_soa_q8_g finding).
        sycl::local_accessor<uint32_t, 1> q8s(USE_SLM ? q8_blocks * 8 : 1, h);
        sycl::local_accessor<float, 1>    q8d(USE_SLM ? q8_blocks     : 1, h);

        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t wgid  = uint32_t(it.get_group(0));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t n     = wgid * N_PER_WG + sg_id;

            if constexpr (USE_SLM) {
                for (uint32_t i = lid; i < q8_blocks * 8; i += WG_ITEMS) {
                    const uint32_t blk = i / 8, w = i % 8;
                    q8s[i] = reinterpret_cast<const uint32_t*>(X8[blk].qs)[w];
                }
                for (uint32_t i = lid; i < q8_blocks; i += WG_ITEMS) q8d[i] = X8[i].d;
                sycl::group_barrier(it.get_group());
            }
            if (n >= N) return;

            // Column-contiguous SoA streams (column-major by output n).
            const uint8_t*  lo_col = q6_lo + uint64_t(n) * (uint64_t(K) / 2);
            const uint8_t*  hi_col = q6_hi + uint64_t(n) * (uint64_t(K) / 4);
            const int8_t*   sc_col = q6_sc + uint64_t(n) * (uint64_t(K) / 16);
            const uint16_t* d_col  = q6_d  + uint64_t(n) * (uint64_t(K) / 256);

            float acc = 0.f;
            // Split-K: lane strides whole 256-elem super-blocks of this column.
            for (uint32_t b = lane; b < blocks_per_col; b += SG_SIZE) {
                const float d6 = soa_fp16_to_fp32(d_col[b]);
                // 4-bit lo plane: 128 bytes/super-block = 32 uint32 (256 elems).
                const uint32_t* lo_w =
                    reinterpret_cast<const uint32_t*>(lo_col + uint64_t(b) * 128);
                // 2-bit hi plane: 64 bytes/super-block = 16 uint32.
                const uint32_t* hi_w =
                    reinterpret_cast<const uint32_t*>(hi_col + uint64_t(b) * 64);
                const int8_t* sc = sc_col + uint64_t(b) * 16;
                const uint32_t q8_base = b * 8;     // 8 q8 blocks per super-block

                // 16 groups of 16 elements (each = one per-16 scale).  Within a
                // group, 4 dp4a words of 4 elems each.  lo plane: 2 elems/byte →
                // 8 bytes (2 uint32) per 16 elems.  hi plane: 4 elems/byte → 4
                // bytes (1 uint32) per 16 elems.  q8: 16 elems = 16 int8 = 4
                // uint32 (half of a 32-elem q8 block).
                #pragma unroll
                for (int g = 0; g < 16; ++g) {
                    const float dsc = d6 * float(sc[g]);
                    const uint32_t lo0 = lo_w[g * 2 + 0];
                    const uint32_t lo1 = lo_w[g * 2 + 1];
                    const uint32_t hi  = hi_w[g];
                    // q8: group g covers elems [g*16, g*16+16) = q8 block
                    // (q8_base + g/2), word offset (g&1)*4 .. +4.
                    const uint32_t q8_blk  = q8_base + (g >> 1);
                    const uint32_t q8_word = (g & 1) * 4;
                    int32_t idot = 0, isum = 0;
                    #pragma unroll
                    for (int w = 0; w < 4; ++w) {
                        // 4 consecutive elems of this word: indices 4w..4w+3.
                        // low nibbles: from lo0 (w<2) or lo1 (w>=2). Each byte
                        // holds 2 elems (lo bits = even, hi bits = odd).
                        const uint32_t lo_src = (w < 2) ? lo0 : lo1;
                        const uint32_t lo_byte_pair = (lo_src >> ((w & 1) * 16)) & 0xFFFFu;
                        // 2 bytes → 4 nibbles: e0=b0&F, e1=b0>>4, e2=b1&F, e3=b1>>4.
                        const uint32_t b0 = lo_byte_pair & 0xFFu;
                        const uint32_t b1 = (lo_byte_pair >> 8) & 0xFFu;
                        const uint32_t n0 = b0 & 0x0Fu, n1 = (b0 >> 4) & 0x0Fu;
                        const uint32_t n2 = b1 & 0x0Fu, n3 = (b1 >> 4) & 0x0Fu;
                        // high 2 bits: elems 4w..4w+3 → byte (4w..)/4 = word w of
                        // hi has 4 elems/byte; elem (4w+r) is in hi byte w, bits
                        // (2*((4w+r)&3)). Since 4w is a multiple of 4, (4w+r)&3==r.
                        const uint32_t hb = (hi >> (w * 8)) & 0xFFu;
                        const uint32_t h0 = (hb >> 0) & 0x3u;
                        const uint32_t h1 = (hb >> 2) & 0x3u;
                        const uint32_t h2 = (hb >> 4) & 0x3u;
                        const uint32_t h3 = (hb >> 6) & 0x3u;
                        const uint32_t pack =
                            (n0 | (h0 << 4))         |
                            ((n1 | (h1 << 4)) << 8)  |
                            ((n2 | (h2 << 4)) << 16) |
                            ((n3 | (h3 << 4)) << 24);
                        const int32_t q8w = USE_SLM
                            ? int32_t(q8s[q8_blk * 8 + q8_word + w])
                            : int32_t(reinterpret_cast<const uint32_t*>(X8[q8_blk].qs)[q8_word + w]);
                        idot = ie::dp4a_us(pack, q8w, idot);
                        isum = ie::dp4a_us(0x01010101u, q8w, isum);
                    }
                    const float q8d_v = USE_SLM ? q8d[q8_blk] : float(X8[q8_blk].d);
                    acc += dsc * q8d_v * (float(idot) - 32.0f * float(isum));
                }
            }

            acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
            if (lane == 0) y[n] = sycl::half(acc);
        });
    });
}

// Public entry. Default stages the q8 activation in SLM; IE_QWEN35_Q6_SOA_GMEM=1
// selects the global-activation variant (occupancy A/B, mirrors Q8-SoA's _g).
sycl::event gemv_q6_soa_q8(sycl::queue& q,
                           const void* x_q8,
                           const uint8_t* q6_lo, const uint8_t* q6_hi,
                           const int8_t* q6_sc, const uint16_t* q6_d,
                           sycl::half* y,
                           uint32_t K, uint32_t N,
                           const std::vector<sycl::event>& deps) {
    static const bool gmem = std::getenv("IE_QWEN35_Q6_SOA_GMEM") != nullptr;
    if (gmem)
        return gemv_q6_soa_q8_impl<false>(q, x_q8, q6_lo, q6_hi, q6_sc, q6_d, y, K, N, deps);
    return gemv_q6_soa_q8_impl<true>(q, x_q8, q6_lo, q6_hi, q6_sc, q6_d, y, K, N, deps);
}

// -------------------------------------------------------------------------
// BATCHED-T W6A8 GEMV over the SoA-Q6 streams (spec-decode verify, 2026-06-21).
//
// The single-row gemv_q6_soa_q8 reads each weight column once per decode step.
// For speculative-decode VERIFY we need to score T (2..16) draft positions in
// ONE forward — the classic spec-decode amortization is "read the weights once,
// dot against T activation rows".  The qwen35 prefill path (T≥2) instead
// dequant→gemm_fp16's the whole weight (T-independent ~53 GB/s restream): at
// T=4 that pays the hundreds-of-tokens prefill amortization for 4 tokens, so
// the verify forward(T=4) costs ~16× a decode step (the spec-decode killer).
//
// This kernel reads each SoA-Q6 weight COLUMN exactly once (same split-K as the
// single-row kernel) and keeps T int accumulators — one dp4a per (weight value,
// activation row).  Weight BW is amortized over T; the extra cost is T× the ALU
// (dp4a) + reading T activation rows (small, L2-resident).  Activations are
// read from GLOBAL (NOT staged in SLM): staging T rows of a K=17408 ffn_up is
// T×~17 KiB > the 128 KiB WG budget at T≥8, and the activation is tiny + WG-
// shared so it stays L2-resident.  acc[t] += d6·d8[t]·(Σ q6u·q8[t] − 32·Σ q8[t]).
//
// x_q8 layout: T contiguous block_q8_1x activation streams (row t at block
// offset t*(K/32)), produced by quantize_q8_1 over the [T,K] fp16 activations.
// Output y is [T,N] row-major (y[t*N + n]).  Per-row numerics IDENTICAL to the
// single-row gemv_q6_soa_q8 (same int-dot math, dp4a-ones offset) → switching
// verify to this keeps the spec loop bit-identical to the T==1 decode path
// (the losslessness gate).  (A block_q8_1s stored-half-sum variant that skips
// the ones-dp4a was tried 2026-06-21; it broke losslessness — argmax flips —
// for a negligible ~4 ms verify saving, so the dp4a-ones offset is kept.)
sycl::event gemv_q6_soa_q8_batched(sycl::queue& q,
                           const void* x_q8,
                           const uint8_t* q6_lo, const uint8_t* q6_hi,
                           const int8_t* q6_sc, const uint16_t* q6_d,
                           sycl::half* y,
                           uint32_t K, uint32_t N, uint32_t T,
                           const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE  = 16;
    constexpr int N_PER_WG = 32;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;        // 512
    constexpr int T_MAX     = 16;                        // matches the wiring THRESH

    const auto* X8 = static_cast<const block_q8_1x*>(x_q8);
    const uint32_t blocks_per_col = K / 256;            // super-blocks/col
    const uint32_t q8_blocks      = K / 32;             // q8 blocks per row (8/super-block)
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "gemv_q6_soa_T", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t wgid  = uint32_t(it.get_group(0));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t n     = wgid * N_PER_WG + sg_id;
            if (n >= N) return;

            const uint8_t*  lo_col = q6_lo + uint64_t(n) * (uint64_t(K) / 2);
            const uint8_t*  hi_col = q6_hi + uint64_t(n) * (uint64_t(K) / 4);
            const int8_t*   sc_col = q6_sc + uint64_t(n) * (uint64_t(K) / 16);
            const uint16_t* d_col  = q6_d  + uint64_t(n) * (uint64_t(K) / 256);

            float acc[T_MAX];
            #pragma unroll
            for (int t = 0; t < T_MAX; ++t) acc[t] = 0.f;

            // Split-K: lane strides whole 256-elem super-blocks of this column.
            for (uint32_t b = lane; b < blocks_per_col; b += SG_SIZE) {
                const float d6 = soa_fp16_to_fp32(d_col[b]);
                const uint32_t* lo_w =
                    reinterpret_cast<const uint32_t*>(lo_col + uint64_t(b) * 128);
                const uint32_t* hi_w =
                    reinterpret_cast<const uint32_t*>(hi_col + uint64_t(b) * 64);
                const int8_t* sc = sc_col + uint64_t(b) * 16;
                const uint32_t q8_base = b * 8;     // 8 q8 blocks per super-block

                #pragma unroll
                for (int g = 0; g < 16; ++g) {
                    const float dsc = d6 * float(sc[g]);
                    const uint32_t lo0 = lo_w[g * 2 + 0];
                    const uint32_t lo1 = lo_w[g * 2 + 1];
                    const uint32_t hi  = hi_w[g];
                    const uint32_t q8_blk  = q8_base + (g >> 1);
                    const uint32_t q8_word = (g & 1) * 4;
                    // Reconstruct the 4 packed-u8x4 weight words ONCE (weight
                    // read amortized over T rows).
                    uint32_t pack[4];
                    #pragma unroll
                    for (int w = 0; w < 4; ++w) {
                        const uint32_t lo_src = (w < 2) ? lo0 : lo1;
                        const uint32_t lo_byte_pair = (lo_src >> ((w & 1) * 16)) & 0xFFFFu;
                        const uint32_t b0 = lo_byte_pair & 0xFFu;
                        const uint32_t b1 = (lo_byte_pair >> 8) & 0xFFu;
                        const uint32_t n0 = b0 & 0x0Fu, n1 = (b0 >> 4) & 0x0Fu;
                        const uint32_t n2 = b1 & 0x0Fu, n3 = (b1 >> 4) & 0x0Fu;
                        const uint32_t hb = (hi >> (w * 8)) & 0xFFu;
                        const uint32_t h0 = (hb >> 0) & 0x3u;
                        const uint32_t h1 = (hb >> 2) & 0x3u;
                        const uint32_t h2 = (hb >> 4) & 0x3u;
                        const uint32_t h3 = (hb >> 6) & 0x3u;
                        pack[w] = (n0 | (h0 << 4))         |
                                  ((n1 | (h1 << 4)) << 8)  |
                                  ((n2 | (h2 << 4)) << 16) |
                                  ((n3 | (h3 << 4)) << 24);
                    }
                    // T activation rows: dp4a the (already-reconstructed) weight
                    // against each row's q8.  Row t's q8 blocks are at offset
                    // t*q8_blocks.
                    #pragma unroll
                    for (int t = 0; t < T_MAX; ++t) {
                        if (uint32_t(t) >= T) break;
                        const block_q8_1x& xb = X8[uint64_t(t) * q8_blocks + q8_blk];
                        const uint32_t* xqw = reinterpret_cast<const uint32_t*>(xb.qs);
                        int32_t idot = 0, isum = 0;
                        #pragma unroll
                        for (int w = 0; w < 4; ++w) {
                            const int32_t q8w = int32_t(xqw[q8_word + w]);
                            idot = ie::dp4a_us(pack[w], q8w, idot);
                            isum = ie::dp4a_us(0x01010101u, q8w, isum);
                        }
                        acc[t] += dsc * xb.d * (float(idot) - 32.0f * float(isum));
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

// Dequant the SoA-Q6 streams → fp16 Bt[K,N] (K contiguous, N columns), so the
// prefill path (gemm_fp16) consumes it exactly like the AoS dequant_q6_K_to_Bt
// output. One work-item per (n, super-block): reconstructs the block's 256
// values (value = (lo|hi<<4) - 32) × per-16 scale × super-scale.  Used only at
// T≥2 (prefill) — not bit-identical to the AoS gemm_q6_K (same value math, fp16
// store), and prefill stays chunked at T≤256.
sycl::event dequant_q6_soa_to_Bt(sycl::queue& q,
                                 const uint8_t* q6_lo, const uint8_t* q6_hi,
                                 const int8_t* q6_sc, const uint16_t* q6_d,
                                 sycl::half* Bt, uint32_t K, uint32_t N,
                                 const std::vector<sycl::event>& deps) {
    const uint32_t bpc = K / 256;
    return ie::ps(q, "dequant_q6_soa", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(uint64_t(N) * bpc), [=](sycl::id<1> idx) {
            const uint64_t lin = idx;
            const uint32_t n = uint32_t(lin / bpc);
            const uint32_t b = uint32_t(lin % bpc);
            const uint8_t*  lo = q6_lo + uint64_t(n) * (uint64_t(K) / 2)  + uint64_t(b) * 128;
            const uint8_t*  hi = q6_hi + uint64_t(n) * (uint64_t(K) / 4)  + uint64_t(b) * 64;
            const int8_t*   sc = q6_sc + uint64_t(n) * (uint64_t(K) / 16) + uint64_t(b) * 16;
            const float     d6 = soa_fp16_to_fp32(q6_d[uint64_t(n) * (uint64_t(K) / 256) + b]);
            const uint32_t  kb = b * 256;
            for (int e = 0; e < 256; ++e) {
                const uint32_t k = kb + uint32_t(e);
                const uint8_t lob = lo[e >> 1];
                const uint8_t hib = hi[e >> 2];
                const int q6u = int((lob >> ((e & 1) * 4)) & 0x0F) |
                                (int((hib >> ((e & 3) * 2)) & 0x03) << 4);
                const float val = d6 * float(sc[e >> 4]) * float(q6u - 32);
                // Bt is [K,N]: row k (= kb+e), column n.
                Bt[uint64_t(k) * N + n] = sycl::half(val);
            }
        });
    });
}

}  // namespace ie
