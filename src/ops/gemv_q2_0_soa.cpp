// src/ops/gemv_q2_0_soa.cpp — Prism ternary Q2_0 (2-bit) fast decode GEMV via a
// load-time SoA repack + coalesced int-dot (dp4a, W2A8) inner loop.  2026-07-17.
//
// Q2_0 (llama.cpp-prism ggml_type 42): 128 weights/block, 34 B = fp16 d (= amax
// over the block) + 32 B qs (2 bits/weight, NATURAL order, LSB-first within each
// byte — byte j/4 holds weight j at bit (j%4)*2).  code c ∈ {0,1,2,3} decodes to
// value (c-1)*d ∈ {-d, 0, +d, +2d}.  This is the SIMPLER cousin of Q6_K: one fp16
// scale per 128, NO 6-bit lo/hi split, NO per-16 int8 sub-scales, offset 1 (not
// 32).  The on-disk qs is already the natural-order 4-elems/byte plane the int-dot
// kernel wants, so the "repack" only separates d from qs and makes each column's
// qs contiguous.
//
// SoA-Q2 layout (per output column n, contiguous — column-major by output n):
//   q2_qs[n*(K/4) + b*32 + ...]  uint8 — 2-bit codes, 4 elems/byte, natural order.
//                                 As uint32: 8 words/block, word w = elems
//                                 [16w, 16w+16); byte r of that word = elems
//                                 [16w+4r, 16w+4r+4).
//   q2_d [n*(K/128) + b]         fp16  — per-128 block scale (raw fp16 bits).
// Footprint: K/4 + (K/128)*2 = 0.25 + 0.015625 B/elem ≈ 2.125 bpw (== on-disk).
//
// DECODE KERNEL (gemv_q2_0_soa_q8): the coalesced all-16-lane pattern of
// gemv_q6_soa_q8_v2, adapted to Q2_0's 128-elem blocks.  One subgroup (16 lanes)
// owns one output column; it processes TWO 128-elem blocks per step so each lane
// reads exactly ONE uint32 word (16 codes) — the 16 lanes read the 16 consecutive
// words of the 2-block (256-elem) span → a fully coalesced 64 B subgroup load,
// all 16 lanes always active.  Lane L (L<8 → block 2s, L≥8 → block 2s+1) owns
// word (lane&7) of its block = natural elems [16·(lane&7), +16).  Those 16 elems
// are exactly half a 32-elem Q8_1 activation block, so the lane's 4 dp4a words
// share one q8 block + one q8 scale.  Per element: value·act = (c-1)·d · d8·qa =
// d·d8·(Σ c·qa − Σ qa) = d·d8·(idot − isum); idot via dp4a over the codes, isum
// via ones-dp4a over q8 (the −1 offset).  Global activation (no SLM/barrier),
// native f16→f32 via bit_cast<half> — matching the winning Q4/Q6 v2 geometry.
//
// NUMERICS: int-dot W2A8 is NOT bit-exact vs fp16-dequant Q2_0 (activation rounds
// to int8; fp fold order differs) — same class as every other int-dot decode
// kernel here.  Gated on the engine's PPL/coherence checks.

#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"

#include <cstdlib>
#include <sycl/sycl.hpp>
#include "ie/dp4a.hpp"
#include "ie/kernel_profiler.hpp"

namespace ie {

// Host-side repack of a canonical block_q2_0 weight [K (contiguous), N] into the
// SoA-Q2 streams above. `W_blocks` = the GGUF tensor data (N columns, each K/128
// contiguous block_q2_0). Outputs are caller-allocated host buffers:
//   q2_qs : N * (K/4) bytes,  q2_d : N * (K/128) uint16 (raw fp16).
// The on-disk qs bytes are already NATURAL order (byte j/4 = weights j..j+3), so
// each block's 32 qs bytes are copied verbatim; only d is split out into its own
// per-column-contiguous plane.
void repack_q2_0_to_soa(const void* W_blocks, uint32_t K, uint32_t N,
                        uint8_t* q2_qs, uint16_t* q2_d) {
    const uint32_t bpc = K / 128;                         // q2_0 blocks per column
    const auto* blocks = static_cast<const block_q2_0*>(W_blocks);
    for (uint64_t n = 0; n < N; ++n) {
        const block_q2_0* col = blocks + n * bpc;
        uint8_t*  qs_col = q2_qs + n * (uint64_t(K) / 4);
        uint16_t* d_col  = q2_d  + n * (uint64_t(K) / 128);
        for (uint32_t b = 0; b < bpc; ++b) {
            d_col[b] = *reinterpret_cast<const uint16_t*>(&col[b].d);
            const uint8_t* src = col[b].qs;
            uint8_t* dst = qs_col + uint64_t(b) * 32;
            for (int i = 0; i < 32; ++i) dst[i] = src[i];   // natural order preserved
        }
    }
}

// W2A8 coalesced int-dot GEMV over the SoA-Q2 streams. One SG per output column,
// N_PER_WG SGs per WG, global activation. Two 128-elem blocks per subgroup-step.
template <int N_PER_WG = 32>
static sycl::event gemv_q2_0_soa_q8_impl(sycl::queue& q,
                           const void* x_q8,
                           const uint8_t* q2_qs, const uint16_t* q2_d,
                           sycl::half* y, uint32_t K, uint32_t N,
                           const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE  = 16;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;

    const auto* X8 = static_cast<const block_q8_1x*>(x_q8);
    const uint32_t blocks_per_col = K / 128;                 // 128-elem blocks/col
    const uint32_t words_per_col  = K / 16;                  // uint32 words of qs/col
    const uint32_t n_steps = (blocks_per_col + 1) / 2;       // 2 blocks/step
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "gemv_q2_soa", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t wgid  = uint32_t(it.get_group(0));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t n     = wgid * N_PER_WG + sg_id;
            if (n >= N) return;

            // Column-contiguous SoA streams (column-major by output n). q2_qs is
            // K/4-byte-aligned per column (K/4 % 4 == 0 for K % 16 == 0), so the
            // uint32 view gives words_per_col = K/16 words per column.
            const uint32_t* qs_col = reinterpret_cast<const uint32_t*>(
                q2_qs + uint64_t(n) * (uint64_t(K) / 4));
            const uint16_t* d_col  = q2_d + uint64_t(n) * (uint64_t(K) / 128);

            const uint32_t w7      = lane & 7u;         // word within the lane's block
            const uint32_t q8wbase = (w7 & 1u) * 4u;    // 0 or 4 (which half of q8 blk)

            float acc = 0.f;
            for (uint32_t s = 0; s < n_steps; ++s) {
                // lanes 0..7 → block 2s ; lanes 8..15 → block 2s+1.
                const uint32_t blk = 2u * s + (lane >> 3);
                if (blk >= blocks_per_col) continue;    // odd blocks_per_col guard

                const float d2 = float(sycl::bit_cast<sycl::half>(d_col[blk]));

                // Coalesced: lane L reads word (blk*8 + w7); across the SG that is
                // words {16s .. 16s+15} → one 64 B subgroup load.
                const uint32_t widx = blk * 8u + w7;
                const uint32_t qw = (widx < words_per_col) ? qs_col[widx] : 0u;

                // This lane's 16 codes cover natural elems [16w7, 16w7+16) of its
                // block → q8 block (blk*4 + w7/2), word base (w7&1)*4.
                const block_q8_1x& xb = X8[blk * 4u + (w7 >> 1)];
                const uint32_t* xqw = reinterpret_cast<const uint32_t*>(xb.qs);
                const float d8 = xb.d;

                int32_t idot = 0, isum = 0;
                #pragma unroll
                for (int w = 0; w < 4; ++w) {
                    // dp4a word w = byte w of qw = 4 codes (elems 16w7+4w..+3),
                    // LSB-first: c_r = (byte >> 2r) & 3.
                    const uint32_t byte_w = (qw >> (w * 8)) & 0xFFu;
                    const uint32_t c0 = (byte_w >> 0) & 0x3u;
                    const uint32_t c1 = (byte_w >> 2) & 0x3u;
                    const uint32_t c2 = (byte_w >> 4) & 0x3u;
                    const uint32_t c3 = (byte_w >> 6) & 0x3u;
                    const uint32_t pack = c0 | (c1 << 8) | (c2 << 16) | (c3 << 24);
                    const int32_t q8w = int32_t(xqw[q8wbase + w]);
                    idot = ie::dp4a_us(pack, q8w, idot);
                    isum = ie::dp4a_us(0x01010101u, q8w, isum);
                }
                acc += d2 * d8 * (float(idot) - float(isum));
            }

            acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
            if (lane == 0) y[n] = sycl::half(acc);
        });
    });
}

// #1b-ILP variant (docs/q2_0_optimization/05, experiment #1b): FOUR 128-elem
// blocks per subgroup-step — two uint32 words per lane = two independent
// load→unpack→dp4a chains in flight, so the second load hides the first
// chain's latency (the captured kprofile put the baseline at 230 GB/s = 38%
// of peak, latency-bound on its single dependent chain). Same SoA layout,
// lane map, and activation contract as the baseline. fp accumulation order
// differs (pairwise) — W2A8 was never bit-exact; gated by unit-test cos + PPL.
template <int N_PER_WG = 32>
static sycl::event gemv_q2_0_soa_q8_ilp2_impl(sycl::queue& q,
                           const void* x_q8,
                           const uint8_t* q2_qs, const uint16_t* q2_d,
                           sycl::half* y, uint32_t K, uint32_t N,
                           const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE  = 16;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;

    const auto* X8 = static_cast<const block_q8_1x*>(x_q8);
    const uint32_t blocks_per_col = K / 128;
    const uint32_t words_per_col  = K / 16;
    const uint32_t n_steps = (blocks_per_col + 3) / 4;       // 4 blocks/step
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "gemv_q2_soa", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t wgid  = uint32_t(it.get_group(0));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t n     = wgid * N_PER_WG + sg_id;
            if (n >= N) return;

            const uint32_t* qs_col = reinterpret_cast<const uint32_t*>(
                q2_qs + uint64_t(n) * (uint64_t(K) / 4));
            const uint16_t* d_col  = q2_d + uint64_t(n) * (uint64_t(K) / 128);

            const uint32_t w7      = lane & 7u;
            const uint32_t q8wbase = (w7 & 1u) * 4u;

            float acc = 0.f;
            for (uint32_t s = 0; s < n_steps; ++s) {
                // Pair A = blocks 4s + (lane>>3); pair B = A + 2. Guards
                // zero-fill (d = 0 kills the contribution) instead of
                // `continue`, so chain B never depends on chain A's branch.
                const uint32_t blkA = 4u * s + (lane >> 3);
                const uint32_t blkB = blkA + 2u;
                const bool vA = blkA < blocks_per_col;
                const bool vB = blkB < blocks_per_col;

                // Both subgroup loads issued up front — independent chains.
                const uint32_t widxA = blkA * 8u + w7;
                const uint32_t widxB = blkB * 8u + w7;
                const uint32_t qwA = (vA && widxA < words_per_col) ? qs_col[widxA] : 0u;
                const uint32_t qwB = (vB && widxB < words_per_col) ? qs_col[widxB] : 0u;
                const float dA = vA ? float(sycl::bit_cast<sycl::half>(d_col[blkA])) : 0.f;
                const float dB = vB ? float(sycl::bit_cast<sycl::half>(d_col[blkB])) : 0.f;

                const block_q8_1x& xa = X8[vA ? (blkA * 4u + (w7 >> 1)) : 0u];
                const block_q8_1x& xb = X8[vB ? (blkB * 4u + (w7 >> 1)) : 0u];
                const uint32_t* xqwA = reinterpret_cast<const uint32_t*>(xa.qs);
                const uint32_t* xqwB = reinterpret_cast<const uint32_t*>(xb.qs);
                const float d8A = xa.d;
                const float d8B = xb.d;

                int32_t idotA = 0, isumA = 0, idotB = 0, isumB = 0;
                #pragma unroll
                for (int w = 0; w < 4; ++w) {
                    const uint32_t byteA = (qwA >> (w * 8)) & 0xFFu;
                    const uint32_t packA = ((byteA >> 0) & 0x3u)
                                         | (((byteA >> 2) & 0x3u) << 8)
                                         | (((byteA >> 4) & 0x3u) << 16)
                                         | (((byteA >> 6) & 0x3u) << 24);
                    const int32_t q8wA = int32_t(xqwA[q8wbase + w]);
                    idotA = ie::dp4a_us(packA, q8wA, idotA);
                    isumA = ie::dp4a_us(0x01010101u, q8wA, isumA);

                    const uint32_t byteB = (qwB >> (w * 8)) & 0xFFu;
                    const uint32_t packB = ((byteB >> 0) & 0x3u)
                                         | (((byteB >> 2) & 0x3u) << 8)
                                         | (((byteB >> 4) & 0x3u) << 16)
                                         | (((byteB >> 6) & 0x3u) << 24);
                    const int32_t q8wB = int32_t(xqwB[q8wbase + w]);
                    idotB = ie::dp4a_us(packB, q8wB, idotB);
                    isumB = ie::dp4a_us(0x01010101u, q8wB, isumB);
                }
                acc += dA * d8A * (float(idotA) - float(isumA))
                     + dB * d8B * (float(idotB) - float(isumB));
            }

            acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
            if (lane == 0) y[n] = sycl::half(acc);
        });
    });
}

// #1c-vec variant (docs/q2_0_optimization/05, experiment #1c): WIDE LOADS.
// The shape microbench showed a uniform ~290 GB/s wall across all shapes —
// load-ISSUE-rate bound (4 B/lane/step scalar weight loads), not DRAM, not
// geometry. Here each lane loads one uint4 (16 B = 64 codes) per step: the 16
// lanes cover EIGHT 128-elem blocks per step via one 256 B coalesced load —
// 4x fewer weight-load instructions per byte. Bonus: each lane now spans two
// WHOLE 32-elem q8 blocks, so the deferred (-1) correction is EXACT and FREE
// via the precomputed block_q8_1x.s (contribution = d2*(d8*idot - s)) — the
// second ones-dp4a chain is gone entirely. Requires K % 512 == 0 (true for
// every Bonsai shape); dispatch falls back to the baseline otherwise.
template <int N_PER_WG = 32>
static sycl::event gemv_q2_0_soa_q8_vec_impl(sycl::queue& q,
                           const void* x_q8,
                           const uint8_t* q2_qs, const uint16_t* q2_d,
                           sycl::half* y, uint32_t K, uint32_t N,
                           const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE  = 16;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;

    const auto* X8 = static_cast<const block_q8_1x*>(x_q8);
    const uint32_t blocks_per_col = K / 128;
    const uint32_t n_steps = (blocks_per_col + 7) / 8;       // 8 blocks/step
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "gemv_q2_soa", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t wgid  = uint32_t(it.get_group(0));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t n     = wgid * N_PER_WG + sg_id;
            if (n >= N) return;

            // Column base is K/4 bytes; K % 512 == 0 keeps it uint4-aligned.
            const auto* qs4 = reinterpret_cast<const sycl::uint4*>(
                q2_qs + uint64_t(n) * (uint64_t(K) / 4));
            const uint16_t* d_col = q2_d + uint64_t(n) * (uint64_t(K) / 128);

            float acc = 0.f;
            for (uint32_t s = 0; s < n_steps; ++s) {
                // Lane L owns elems [s*1024 + 64L, +64): q2 block s*8 + L/2
                // (half L&1), q8 blocks blk*4 + (L&1)*2 and +1 — whole blocks.
                const uint32_t blk = s * 8u + (lane >> 1);
                const bool v = blk < blocks_per_col;
                const float d2 = v ? float(sycl::bit_cast<sycl::half>(d_col[v ? blk : 0u])) : 0.f;

                const sycl::uint4 qv = v ? qs4[s * 16u + lane]
                                         : sycl::uint4{0u, 0u, 0u, 0u};

                const uint32_t qb0 = blk * 4u + (lane & 1u) * 2u;
                const block_q8_1x& x0 = X8[v ? qb0 : 0u];
                const block_q8_1x& x1 = X8[v ? (qb0 + 1u) : 0u];
                const uint32_t* xq0 = reinterpret_cast<const uint32_t*>(x0.qs);
                const uint32_t* xq1 = reinterpret_cast<const uint32_t*>(x1.qs);

                int32_t id0 = 0, id1 = 0;
                const uint32_t qw01[2] = {qv.x(), qv.y()};
                const uint32_t qw23[2] = {qv.z(), qv.w()};
                #pragma unroll
                for (int h2 = 0; h2 < 2; ++h2) {
                    const uint32_t qwa = qw01[h2];
                    const uint32_t qwb = qw23[h2];
                    #pragma unroll
                    for (int w = 0; w < 4; ++w) {
                        const uint32_t ba = (qwa >> (w * 8)) & 0xFFu;
                        const uint32_t pa = ((ba >> 0) & 0x3u) | (((ba >> 2) & 0x3u) << 8)
                                          | (((ba >> 4) & 0x3u) << 16) | (((ba >> 6) & 0x3u) << 24);
                        id0 = ie::dp4a_us(pa, int32_t(xq0[h2 * 4 + w]), id0);

                        const uint32_t bb = (qwb >> (w * 8)) & 0xFFu;
                        const uint32_t pb = ((bb >> 0) & 0x3u) | (((bb >> 2) & 0x3u) << 8)
                                          | (((bb >> 4) & 0x3u) << 16) | (((bb >> 6) & 0x3u) << 24);
                        id1 = ie::dp4a_us(pb, int32_t(xq1[h2 * 4 + w]), id1);
                    }
                }
                // Exact deferred -1: sum(c-1)*d2*a = d2*(d8*idot - s) per whole
                // q8 block (x.s = d8 * sum(qs), computed at activation quant).
                acc += d2 * (x0.d * float(id0) - x0.s)
                     + d2 * (x1.d * float(id1) - x1.s);
            }

            acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
            if (lane == 0) y[n] = sycl::half(acc);
        });
    });
}

// Multi-bank wide-load GEMV: ≤3 banks sharing one activation, one launch. WGs
// span the concatenated column space (per-bank wgoff), same inner loop as the
// vec kernel. Requires K%512==0. docs/q2_0_optimization/05 (#1c multi-bank).
template <int N_PER_WG = 16>
static sycl::event gemv_q2_0_soa_q8_vec_multi_impl(sycl::queue& q,
                           const void* x_q8,
                           const uint8_t* const q2_qs[3],
                           const uint16_t* const q2_d[3],
                           sycl::half* const y[3],
                           const uint32_t N[3], int n_banks, uint32_t K,
                           const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE  = 16;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;

    const auto* X8 = static_cast<const block_q8_1x*>(x_q8);
    const uint32_t blocks_per_col = K / 128;
    const uint32_t n_steps = (blocks_per_col + 7) / 8;

    const uint8_t*  qsb[3] = {nullptr, nullptr, nullptr};
    const uint16_t* db [3] = {nullptr, nullptr, nullptr};
    sycl::half*     yb [3] = {nullptr, nullptr, nullptr};
    uint32_t        Nb [3] = {0, 0, 0};
    uint32_t        wgoff[4] = {0, 0, 0, 0};
    uint32_t total_wgs = 0;
    for (int b = 0; b < n_banks; ++b) {
        qsb[b] = q2_qs[b]; db[b] = q2_d[b]; yb[b] = y[b]; Nb[b] = N[b];
        wgoff[b] = total_wgs;
        total_wgs += (N[b] + N_PER_WG - 1) / N_PER_WG;
    }
    wgoff[n_banks] = total_wgs;
    const int nb = n_banks;

    return ie::ps(q, "gemv_q2_soa", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(uint64_t(total_wgs) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t wgid  = uint32_t(it.get_group(0));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;

            int bank = 0;
            #pragma unroll
            for (int b = 1; b < 3; ++b) if (b < nb && wgid >= wgoff[b]) bank = b;
            const uint32_t n = (wgid - wgoff[bank]) * N_PER_WG + sg_id;
            if (n >= Nb[bank]) return;

            const auto* qs4 = reinterpret_cast<const sycl::uint4*>(
                qsb[bank] + uint64_t(n) * (uint64_t(K) / 4));
            const uint16_t* d_col = db[bank] + uint64_t(n) * (uint64_t(K) / 128);

            float acc = 0.f;
            for (uint32_t s = 0; s < n_steps; ++s) {
                const uint32_t blk = s * 8u + (lane >> 1);
                const bool v = blk < blocks_per_col;
                const float d2 = v ? float(sycl::bit_cast<sycl::half>(d_col[v ? blk : 0u])) : 0.f;
                const sycl::uint4 qv = v ? qs4[s * 16u + lane] : sycl::uint4{0u, 0u, 0u, 0u};

                const uint32_t qb0 = blk * 4u + (lane & 1u) * 2u;
                const block_q8_1x& x0 = X8[v ? qb0 : 0u];
                const block_q8_1x& x1 = X8[v ? (qb0 + 1u) : 0u];
                const uint32_t* xq0 = reinterpret_cast<const uint32_t*>(x0.qs);
                const uint32_t* xq1 = reinterpret_cast<const uint32_t*>(x1.qs);

                int32_t id0 = 0, id1 = 0;
                const uint32_t qw01[2] = {qv.x(), qv.y()};
                const uint32_t qw23[2] = {qv.z(), qv.w()};
                #pragma unroll
                for (int h2 = 0; h2 < 2; ++h2) {
                    const uint32_t qwa = qw01[h2];
                    const uint32_t qwb = qw23[h2];
                    #pragma unroll
                    for (int w = 0; w < 4; ++w) {
                        const uint32_t ba = (qwa >> (w * 8)) & 0xFFu;
                        const uint32_t pa = ((ba >> 0) & 0x3u) | (((ba >> 2) & 0x3u) << 8)
                                          | (((ba >> 4) & 0x3u) << 16) | (((ba >> 6) & 0x3u) << 24);
                        id0 = ie::dp4a_us(pa, int32_t(xq0[h2 * 4 + w]), id0);
                        const uint32_t bb = (qwb >> (w * 8)) & 0xFFu;
                        const uint32_t pb = ((bb >> 0) & 0x3u) | (((bb >> 2) & 0x3u) << 8)
                                          | (((bb >> 4) & 0x3u) << 16) | (((bb >> 6) & 0x3u) << 24);
                        id1 = ie::dp4a_us(pb, int32_t(xq1[h2 * 4 + w]), id1);
                    }
                }
                acc += d2 * (x0.d * float(id0) - x0.s)
                     + d2 * (x1.d * float(id1) - x1.s);
            }
            acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
            if (lane == 0) yb[bank][n] = sycl::half(acc);
        });
    });
}

// BATCHED-T variant of the #1c-vec kernel (docs/q2_0_optimization/07, native
// W2A8 prefill) — v2, IN-KERNEL T-tiling: SAME SoA layout and wide-load lane
// geometry as gemv_q2_0_soa_q8_vec_impl (16 lanes, lane owns one uint4 = 64
// codes = 2 whole q8 blocks per step, blk = s*8 + (lane>>1), qb0 = blk*4 +
// (lane&1)*2), but ONE LAUNCH covers the whole [T,N] output. v1 sliced T on
// the host into one launch per T_TILE=16 — 16 launches per projection at
// T=256, each re-streaming the ENTIRE weight matrix from DRAM (the §4
// "restream trap"; measured 5.3× slower than the dequant→gemm path). v2 moves
// the T-slice loop inside the kernel (the moe_fused.cpp tk_base pattern): each
// subgroup owns one output column n and walks T in T_TILE row groups; the
// column's weights are re-WALKED per tk group but the per-subgroup footprint
// is only K/4 bytes (≤ 4.4 KB at Bonsai K) — L1/L2-resident after the tk==0
// pass, NOT a fresh DRAM stream per launch. The 2-bit→dp4a unpack ALU repeats
// per tk group (accepted trade). Per step the lane's 16 dp4a weight words are
// unpacked ONCE into p0/p1 and dotted against the tk group's ≤T_TILE staged
// Q8_1 rows (row t's blocks at t*(K/32)). Exact deferred-(−1) per row via the
// precomputed block_q8_1x.s: acc[t] += d2*(d8*idot − s), identical to the
// T==1 vec kernel. y is [T,N] row-major (y[t*N + n]). T remainder: tlim =
// min(T_TILE, T−tk) guards the X8 fetch, the accumulate, and the store; tlim
// is SG-uniform (function of tk and T only), so the guarded per-t
// reduce_over_group stays convergent. Requires K % 512 == 0 (callers guard).
template <int T_TILE, int N_PER_WG = 16>
static sycl::event gemv_q2_0_soa_q8_batched_impl(sycl::queue& q,
                           const void* x_q8,
                           const uint8_t* q2_qs, const uint16_t* q2_d,
                           sycl::half* y, uint32_t K, uint32_t N, uint32_t T,
                           const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE  = 16;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;

    const auto* X8 = static_cast<const block_q8_1x*>(x_q8);
    const uint32_t blocks_per_col = K / 128;
    const uint32_t n_steps   = (blocks_per_col + 7) / 8;     // 8 blocks/step
    const uint32_t q8_blocks = K / 32;                       // q8 blocks per act row
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "gemv_q2_soa_T", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t wgid  = uint32_t(it.get_group(0));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t n     = wgid * N_PER_WG + sg_id;
            if (n >= N) return;

            // Column base is K/4 bytes; K % 512 == 0 keeps it uint4-aligned.
            const auto* qs4 = reinterpret_cast<const sycl::uint4*>(
                q2_qs + uint64_t(n) * (uint64_t(K) / 4));
            const uint16_t* d_col = q2_d + uint64_t(n) * (uint64_t(K) / 128);

            // In-kernel T tiling (tk_base pattern): after the tk==0 pass the
            // column's K/4 weight bytes are cache-resident; later tk groups
            // re-walk them from L1/L2, not from DRAM.
            for (uint32_t tk = 0; tk < T; tk += T_TILE) {
                const uint32_t tlim = sycl::min(uint32_t(T_TILE), T - tk);

                float acc[T_TILE];
                #pragma unroll
                for (int t = 0; t < T_TILE; ++t) acc[t] = 0.f;

                for (uint32_t s = 0; s < n_steps; ++s) {
                    // Lane L owns elems [s*1024 + 64L, +64): q2 block s*8 + L/2
                    // (half L&1), q8 blocks blk*4 + (L&1)*2 and +1 — whole blocks.
                    const uint32_t blk = s * 8u + (lane >> 1);
                    const bool v = blk < blocks_per_col;
                    const float d2 = v ? float(sycl::bit_cast<sycl::half>(d_col[v ? blk : 0u])) : 0.f;

                    const sycl::uint4 qv = v ? qs4[s * 16u + lane]
                                             : sycl::uint4{0u, 0u, 0u, 0u};

                    const uint32_t qb0 = blk * 4u + (lane & 1u) * 2u;
                    const uint32_t xi0 = v ? qb0 : 0u;          // guarded q8 block idx
                    const uint32_t xi1 = v ? (qb0 + 1u) : 0u;   // (d2=0 kills !v terms)

                    // Unpack the lane's 16 dp4a weight words ONCE into registers —
                    // the weight expand amortizes over the tlim rows below.
                    uint32_t p0[8], p1[8];
                    const uint32_t qw01[2] = {qv.x(), qv.y()};
                    const uint32_t qw23[2] = {qv.z(), qv.w()};
                    #pragma unroll
                    for (int h2 = 0; h2 < 2; ++h2) {
                        const uint32_t qwa = qw01[h2];
                        const uint32_t qwb = qw23[h2];
                        #pragma unroll
                        for (int w = 0; w < 4; ++w) {
                            const uint32_t ba = (qwa >> (w * 8)) & 0xFFu;
                            p0[h2 * 4 + w] = ((ba >> 0) & 0x3u) | (((ba >> 2) & 0x3u) << 8)
                                           | (((ba >> 4) & 0x3u) << 16) | (((ba >> 6) & 0x3u) << 24);
                            const uint32_t bb = (qwb >> (w * 8)) & 0xFFu;
                            p1[h2 * 4 + w] = ((bb >> 0) & 0x3u) | (((bb >> 2) & 0x3u) << 8)
                                           | (((bb >> 4) & 0x3u) << 16) | (((bb >> 6) & 0x3u) << 24);
                        }
                    }

                    // This tk group's activation rows (tk+t, t < tlim): dp4a
                    // the already-unpacked weight words against each row's two
                    // whole q8 blocks.
                    #pragma unroll
                    for (int t = 0; t < T_TILE; ++t) {
                        if (uint32_t(t) < tlim) {
                            const block_q8_1x& x0 = X8[(uint64_t(tk) + t) * q8_blocks + xi0];
                            const block_q8_1x& x1 = X8[(uint64_t(tk) + t) * q8_blocks + xi1];
                            const uint32_t* xq0 = reinterpret_cast<const uint32_t*>(x0.qs);
                            const uint32_t* xq1 = reinterpret_cast<const uint32_t*>(x1.qs);
                            int32_t id0 = 0, id1 = 0;
                            #pragma unroll
                            for (int w = 0; w < 8; ++w) {
                                id0 = ie::dp4a_us(p0[w], int32_t(xq0[w]), id0);
                                id1 = ie::dp4a_us(p1[w], int32_t(xq1[w]), id1);
                            }
                            // Exact deferred -1 per whole q8 block (x.s = d8·Σqs).
                            acc[t] += d2 * (x0.d * float(id0) - x0.s)
                                    + d2 * (x1.d * float(id1) - x1.s);
                        }
                    }
                }

                // tlim is SG-uniform → the guarded reduce stays convergent.
                #pragma unroll
                for (int t = 0; t < T_TILE; ++t) {
                    if (uint32_t(t) < tlim) {
                        const float r = sycl::reduce_over_group(it.get_sub_group(), acc[t],
                                                                sycl::plus<float>());
                        if (lane == 0) y[(uint64_t(tk) + t) * N + n] = sycl::half(r);
                    }
                }
            }
        });
    });
}

// Sub-groups per WG for the Q2 decode GEMV. Default 16 since the 2026-07-17
// VEC-era A/B (+2% in-app, fixes the isolated attn_qkv [5120,10240] NPWG-32
// pathology 132→383 GB/s; docs/q2_0_optimization/05). IE_Q2_NPWG ∈
// {8,16,32,64} overrides. Read once.
static uint32_t q2_npwg() {
    static const uint32_t v = []{
        if (const char* e = std::getenv("IE_Q2_NPWG")) {
            int n = std::atoi(e); if (n==8||n==16||n==32||n==64) return uint32_t(n);
        }
        return 16u;
    }();
    return v;
}

// IE_Q2_ILP=1 selects the #1b two-chain variant (default 0 = shipped baseline;
// experiment gating per docs/q2_0_optimization/05). Read once.
static bool q2_ilp() {
    static const bool v = []{
        const char* e = std::getenv("IE_Q2_ILP");
        return e && std::atoi(e) == 1;
    }();
    return v;
}

// #1c wide-load variant: DEFAULT ON since the 2026-07-17 A/B (+23.5% decode,
// 26.9→33.2 tok/s, PPL held — docs/q2_0_optimization/05). IE_Q2_VEC=0 opts
// out (kill switch); K%512!=0 falls through to the baseline kernel.
static bool q2_vec() {
    static const bool v = []{
        const char* e = std::getenv("IE_Q2_VEC");
        return !e || std::atoi(e) != 0;
    }();
    return v;
}

// In-kernel T tile for the batched prefill kernel. Default 16 (acc[16] fp32 +
// p0/p1[8] u32 fits GRF without spill); IE_Q2_BATCH_TILE=8 opts down. Read once.
static int q2_batch_tile() {
    static const int v = []{
        if (const char* e = std::getenv("IE_Q2_BATCH_TILE")) {
            if (std::atoi(e) == 8) return 8;
        }
        return 16;
    }();
    return v;
}

sycl::event gemv_q2_0_soa_q8(sycl::queue& q,
                             const void* x_q8,
                             const uint8_t* q2_qs, const uint16_t* q2_d,
                             sycl::half* y, uint32_t K, uint32_t N,
                             const std::vector<sycl::event>& deps) {
    if (q2_vec() && (K % 512u) == 0u) {
        switch (q2_npwg()) {
            case 8:  return gemv_q2_0_soa_q8_vec_impl<8> (q, x_q8, q2_qs, q2_d, y, K, N, deps);
            case 16: return gemv_q2_0_soa_q8_vec_impl<16>(q, x_q8, q2_qs, q2_d, y, K, N, deps);
            case 64: return gemv_q2_0_soa_q8_vec_impl<64>(q, x_q8, q2_qs, q2_d, y, K, N, deps);
            default: return gemv_q2_0_soa_q8_vec_impl<32>(q, x_q8, q2_qs, q2_d, y, K, N, deps);
        }
    }
    if (q2_ilp()) {
        switch (q2_npwg()) {
            case 8:  return gemv_q2_0_soa_q8_ilp2_impl<8> (q, x_q8, q2_qs, q2_d, y, K, N, deps);
            case 16: return gemv_q2_0_soa_q8_ilp2_impl<16>(q, x_q8, q2_qs, q2_d, y, K, N, deps);
            case 64: return gemv_q2_0_soa_q8_ilp2_impl<64>(q, x_q8, q2_qs, q2_d, y, K, N, deps);
            default: return gemv_q2_0_soa_q8_ilp2_impl<32>(q, x_q8, q2_qs, q2_d, y, K, N, deps);
        }
    }
    switch (q2_npwg()) {
        case 8:  return gemv_q2_0_soa_q8_impl<8> (q, x_q8, q2_qs, q2_d, y, K, N, deps);
        case 16: return gemv_q2_0_soa_q8_impl<16>(q, x_q8, q2_qs, q2_d, y, K, N, deps);
        case 64: return gemv_q2_0_soa_q8_impl<64>(q, x_q8, q2_qs, q2_d, y, K, N, deps);
        default: return gemv_q2_0_soa_q8_impl<32>(q, x_q8, q2_qs, q2_d, y, K, N, deps);
    }
}

// N_PER_WG dispatch for the batched kernel (reuses the Q2 NPWG knob).
template <int T_TILE>
static sycl::event gemv_q2_batched_dispatch(sycl::queue& q, const void* x_q8,
                           const uint8_t* q2_qs, const uint16_t* q2_d,
                           sycl::half* y, uint32_t K, uint32_t N, uint32_t T,
                           const std::vector<sycl::event>& deps) {
    switch (q2_npwg()) {
        case 8:  return gemv_q2_0_soa_q8_batched_impl<T_TILE, 8> (q, x_q8, q2_qs, q2_d, y, K, N, T, deps);
        case 32: return gemv_q2_0_soa_q8_batched_impl<T_TILE, 32>(q, x_q8, q2_qs, q2_d, y, K, N, T, deps);
        case 64: return gemv_q2_0_soa_q8_batched_impl<T_TILE, 64>(q, x_q8, q2_qs, q2_d, y, K, N, T, deps);
        default: return gemv_q2_0_soa_q8_batched_impl<T_TILE, 16>(q, x_q8, q2_qs, q2_d, y, K, N, T, deps);
    }
}

// Public batched entry, v2: ONE kernel launch covers the whole [T,N] output —
// the T-slice loop lives INSIDE the kernel (tk groups of T_TILE, remainder
// handled by the in-kernel tlim clamp). v1's host-side slicing (one launch
// per T_TILE=16 slice, dep-chained) re-streamed the full weight matrix from
// DRAM per slice and measured 5.3× slower than dequant→gemm. T==1 passes
// through to the decode vec kernel unchanged.
// Requires K % 512 == 0 — callers guard and take the dequant path otherwise.
sycl::event gemv_q2_0_soa_q8_batched(sycl::queue& q,
                             const void* x_q8,
                             const uint8_t* q2_qs, const uint16_t* q2_d,
                             sycl::half* y, uint32_t K, uint32_t N, uint32_t T,
                             const std::vector<sycl::event>& deps) {
    if (T == 0) return {};
    if (T == 1) return gemv_q2_0_soa_q8(q, x_q8, q2_qs, q2_d, y, K, N, deps);
    if (q2_batch_tile() == 8)
        return gemv_q2_batched_dispatch<8> (q, x_q8, q2_qs, q2_d, y, K, N, T, deps);
    return gemv_q2_batched_dispatch<16>(q, x_q8, q2_qs, q2_d, y, K, N, T, deps);
}

sycl::event gemv_q2_0_soa_q8_multi(sycl::queue& q, const void* x_q8,
                                   const uint8_t* const q2_qs[3],
                                   const uint16_t* const q2_d[3],
                                   sycl::half* const y[3],
                                   const uint32_t N[3], int n_banks, uint32_t K,
                                   const std::vector<sycl::event>& deps) {
    switch (q2_npwg()) {
        case 8:  return gemv_q2_0_soa_q8_vec_multi_impl<8> (q, x_q8, q2_qs, q2_d, y, N, n_banks, K, deps);
        case 32: return gemv_q2_0_soa_q8_vec_multi_impl<32>(q, x_q8, q2_qs, q2_d, y, N, n_banks, K, deps);
        case 64: return gemv_q2_0_soa_q8_vec_multi_impl<64>(q, x_q8, q2_qs, q2_d, y, N, n_banks, K, deps);
        default: return gemv_q2_0_soa_q8_vec_multi_impl<16>(q, x_q8, q2_qs, q2_d, y, N, n_banks, K, deps);
    }
}

// Write-coalesced prefill dequant: DEFAULT ON since the 2026-07-18 A/B
// (+84% pp512, 449→823 tok/s; dequant 706→188 ms; PPL bit-exact). IE_Q2_DEQUANT_V2=0
// reverts to the scatter-write baseline (kill switch). Read once.
static bool q2_dequant_v2() {
    static const bool v = []{
        const char* e = std::getenv("IE_Q2_DEQUANT_V2");
        return !e || std::atoi(e) != 0;
    }();
    return v;
}

// V2 write-coalesced dequant (IE_Q2_DEQUANT_V2=1): same Bt[K,N] (n fastest)
// and the same per-element math as the baseline below — bit-exact output —
// but the item map is (block b, column n) with n the fastest-varying LOCAL
// dimension (the proven dequant_q4_K_to_Bt geometry). At every inner step the
// WG's 64 lanes store 64 CONSECUTIVE fp16 of one Bt row (128 B contiguous),
// where the baseline's linear (n-major) map scatters adjacent lanes 128·N·2 B
// apart — one cacheline transaction per 2 B store = the measured ~86 GB/s (82%
// of prefill GPU time). Reads are two uint4 (32 B) per item (the #1c-vec
// wide-load geometry); alignment holds because K % 128 == 0.
static sycl::event dequant_q2_0_soa_to_Bt_v2(sycl::queue& q,
                                   const uint8_t* q2_qs, const uint16_t* q2_d,
                                   sycl::half* Bt, uint32_t K, uint32_t N,
                                   const std::vector<sycl::event>& deps) {
    const uint32_t bpc = K / 128;
    constexpr uint32_t WG_N = 64;
    const uint32_t n_pad = ((N + WG_N - 1) / WG_N) * WG_N;
    return ie::ps(q, "dequant_q2_soa", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<2>({bpc, n_pad}, {1, WG_N}),
                       [=](sycl::nd_item<2> it) {
            const uint32_t b = uint32_t(it.get_group(0));
            const uint32_t n = uint32_t(it.get_global_id(1));
            if (n >= N) return;

            const auto* qs4 = reinterpret_cast<const sycl::uint4*>(
                q2_qs + uint64_t(n) * (uint64_t(K) / 4) + uint64_t(b) * 32u);
            const sycl::uint4 v0 = qs4[0], v1 = qs4[1];
            const uint32_t qw[8] = {v0.x(), v0.y(), v0.z(), v0.w(),
                                    v1.x(), v1.y(), v1.z(), v1.w()};
            const float d2 = float(sycl::bit_cast<sycl::half>(
                q2_d[uint64_t(n) * (uint64_t(K) / 128) + b]));

            sycl::half* dst = Bt + uint64_t(b) * 128u * N + n;
            #pragma unroll
            for (int w = 0; w < 8; ++w) {
                #pragma unroll
                for (int j = 0; j < 16; ++j) {
                    // Elem e = 16w + j; bit-in-word = 2j selects the SAME
                    // LSB-first bits the baseline takes from qs[e>>2].
                    const uint8_t code = uint8_t((qw[w] >> (2 * j)) & 0x3u);
                    const float   val  = (int(code) - 1) * d2;
                    dst[uint64_t(w * 16 + j) * N] = sycl::half(val);
                }
            }
        });
    });
}

// V3 experiment gate (IE_Q2_DEQUANT_V3=1, default OFF): read-AND-write-coalesced
// dequant via cooperative SLM staging. Wins over V2 when set. Read once.
static bool q2_dequant_v3() {
    static const bool v = []{
        const char* e = std::getenv("IE_Q2_DEQUANT_V3");
        return e && std::atoi(e) != 0;
    }();
    return v;
}

// V3 (IE_Q2_DEQUANT_V3=1): V2 coalesces the WRITES but its reads stay strided —
// at fixed block b the 64 lanes' 32 B qs reads sit K/4 bytes apart (64 scattered
// cachelines per WG step, measured ~105 GB/s). The only read-contiguous axis of
// the SoA layout is ALONG b within a column, so V3 tiles B_TILE=4 blocks × 64
// columns per WG and stages each column's B_TILE*32 B contiguous qs run into SLM
// with a linear cooperative load (off-within-run fastest: consecutive work-items
// read consecutive global uint4s → each 16-lane SG load = two 128 B contiguous
// segments; the gemm_q4k_xmx dequant_q4k_block_to_stripe staging idiom). After
// the barrier each lane dequants its column's B_TILE blocks from SLM with V2's
// exact per-element math and V2's n-fastest 128 B-contiguous store pattern —
// bit-exact output, only the load path changed. SLM = 64 cols × (B_TILE*2+1)
// uint4 = 9216 B/WG (+1 uint4/column pad → conflict-free SLM banks both ways).
// Grid = ceil(bpc/B_TILE) × N/64 WGs — 160 WGs on attn_k/v [5120,1024], enough
// for 24 Xe-cores. K % 128 == 0 assumed (same as V2); partial b-tiles guarded.
static sycl::event dequant_q2_0_soa_to_Bt_v3(sycl::queue& q,
                                   const uint8_t* q2_qs, const uint16_t* q2_d,
                                   sycl::half* Bt, uint32_t K, uint32_t N,
                                   const std::vector<sycl::event>& deps) {
    const uint32_t bpc = K / 128;
    constexpr uint32_t WG_N    = 64;             // columns per WG (= V2's WG)
    constexpr uint32_t B_TILE  = 4;              // 128-elem blocks per WG tile
    constexpr uint32_t VPC     = B_TILE * 2;     // uint4s per column run (32 B/blk)
    constexpr uint32_t VPC_PAD = VPC + 1;        // +1 uint4 pad → bank spread
    const uint32_t n_btiles = (bpc + B_TILE - 1) / B_TILE;
    const uint32_t n_pad = ((N + WG_N - 1) / WG_N) * WG_N;
    return ie::ps(q, "dequant_q2_soa", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<sycl::uint4, 1> smem(WG_N * VPC_PAD, h);
        h.parallel_for(sycl::nd_range<2>({n_btiles, n_pad}, {1, WG_N}),
                       [=](sycl::nd_item<2> it) {
            const uint32_t bt0 = uint32_t(it.get_group(0)) * B_TILE;
            const uint32_t c   = uint32_t(it.get_local_id(1));
            const uint32_t n0  = uint32_t(it.get_group(1)) * WG_N;
            // Valid blocks in this tile (WG-uniform; bt0 < bpc by grid size).
            const uint32_t nb  = sycl::min(B_TILE, bpc - bt0);

            // Cooperative coalesced stage: run offset varies FASTEST with the
            // linear id, so consecutive work-items read consecutive global
            // uint4s of one column's contiguous qs run. No early return before
            // the barrier — pad columns / pad blocks just skip their load.
            for (uint32_t i = c; i < WG_N * VPC; i += WG_N) {
                const uint32_t col = i / VPC;
                const uint32_t off = i % VPC;
                const uint32_t n   = n0 + col;
                if (n < N && off < nb * 2u) {
                    const auto* src4 = reinterpret_cast<const sycl::uint4*>(
                        q2_qs + uint64_t(n) * (uint64_t(K) / 4) + uint64_t(bt0) * 32u);
                    smem[col * VPC_PAD + off] = src4[off];
                }
            }
            sycl::group_barrier(it.get_group());

            const uint32_t n = n0 + c;
            if (n < N) {
                const uint16_t* d_col = q2_d + uint64_t(n) * (uint64_t(K) / 128);
                for (uint32_t bt = 0; bt < nb; ++bt) {
                    const uint32_t blk = bt0 + bt;
                    const sycl::uint4 v0 = smem[c * VPC_PAD + bt * 2u];
                    const sycl::uint4 v1 = smem[c * VPC_PAD + bt * 2u + 1u];
                    const uint32_t qw[8] = {v0.x(), v0.y(), v0.z(), v0.w(),
                                            v1.x(), v1.y(), v1.z(), v1.w()};
                    const float d2 = float(sycl::bit_cast<sycl::half>(d_col[blk]));

                    sycl::half* dst = Bt + uint64_t(blk) * 128u * N + n;
                    #pragma unroll
                    for (int w = 0; w < 8; ++w) {
                        #pragma unroll
                        for (int j = 0; j < 16; ++j) {
                            // Identical extraction to V2/baseline — bit-exact.
                            const uint8_t code = uint8_t((qw[w] >> (2 * j)) & 0x3u);
                            const float   val  = (int(code) - 1) * d2;
                            dst[uint64_t(w * 16 + j) * N] = sycl::half(val);
                        }
                    }
                }
            }
        });
    });
}

// Dequant the SoA-Q2 streams → fp16 Bt[K,N] (K contiguous, N columns), so the
// prefill path (gemm_fp16) consumes it exactly like dequant_q6_soa_to_Bt. One
// work-item per (n, block): reconstructs the block's 128 values (value =
// (code-1)*d). Used only at T≥2 (prefill).
sycl::event dequant_q2_0_soa_to_Bt(sycl::queue& q,
                                   const uint8_t* q2_qs, const uint16_t* q2_d,
                                   sycl::half* Bt, uint32_t K, uint32_t N,
                                   const std::vector<sycl::event>& deps) {
    if (q2_dequant_v3())
        return dequant_q2_0_soa_to_Bt_v3(q, q2_qs, q2_d, Bt, K, N, deps);
    if (q2_dequant_v2())
        return dequant_q2_0_soa_to_Bt_v2(q, q2_qs, q2_d, Bt, K, N, deps);
    const uint32_t bpc = K / 128;
    return ie::ps(q, "dequant_q2_soa", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(uint64_t(N) * bpc), [=](sycl::id<1> idx) {
            const uint64_t lin = idx;
            const uint32_t n = uint32_t(lin / bpc);
            const uint32_t b = uint32_t(lin % bpc);
            const uint8_t* qs = q2_qs + uint64_t(n) * (uint64_t(K) / 4) + uint64_t(b) * 32;
            const float    d2 = float(sycl::bit_cast<sycl::half>(
                                    q2_d[uint64_t(n) * (uint64_t(K) / 128) + b]));
            const uint32_t kb = b * 128;
            for (int e = 0; e < 128; ++e) {
                const uint8_t code = (qs[e >> 2] >> ((e & 3) * 2)) & 0x03u;
                const float   val  = (int(code) - 1) * d2;
                Bt[uint64_t(kb + uint32_t(e)) * N + n] = sycl::half(val);
            }
        });
    });
}

}  // namespace ie
