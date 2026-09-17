// src/ops/gemv_q4_soa.cpp — fast Q4_K decode GEMV via a load-time SoA-Q4 repack
// + int-dot (dp4a, W4A8) inner loop with split-K.  2026-06-22.
//
// WHY: the on-disk AoS block_q4_K (144 B: d/dmin/12-byte packed scales/128-byte
// nibbles) forces the decode GEMV (gemv_q4_K in gemv_q4k.cpp) into a complex
// (sub,half,g,hi_nib) lane lattice + per-lane byte loads + the 6-bit scale/min
// unpack on every block, and it dequants the weight to fp16 to multiply against
// an fp16 activation (W4A16, FLOAT macs).  llama.cpp's SYCL Q4_K decode wins on
// B70 (its ~64% vs our ~42% BW) primarily via a reordered Structure-of-Arrays
// weight layout (see reference_llama_sycl_q4k_decode.md).  This file mirrors the
// PROVEN Q6-SoA int-dot win (gemv_q6_soa.cpp, ~80% BW at 6.5 bpw) at 4.5 bpw:
// the Q4_K weight is repacked at load into per-column NATURAL-ORDER streams
// (q4_q nibbles, per-32 int8 s_raw/m_raw, per-256 fp16 d/dmin), so the kernel
// does coalesced loads + a cheap nibble unpack + int8 dp4a.  Split-K: SG_SIZE
// lanes per column stride whole 256-elem super-blocks (the K-reduction the old
// per-column-serial kernel lacks).
//
// SoA-Q4 layout (per output column n, contiguous; mirrors gemv_q6_soa's
//   column-major-by-output-n discipline so a column-slice stays a pure byte cut):
//   q4_q [n*(K/2)  + ...] uint8 — low nibble of element k, packed 2 elems/byte
//                          in natural k order: byte (k/2) holds elem 2j in bits
//                          0..3 and elem 2j+1 in bits 4..7.
//   q4_sc[n*(K/32) + ...] int8  — per-32-element 6-bit UNPACKED scale s_raw
//                          (natural sub-block order; 0..63 fits int8).
//   q4_mn[n*(K/32) + ...] int8  — per-32-element 6-bit UNPACKED min   m_raw.
//   q4_d [n*(K/256)+ ...] fp16  — per-256 super-block scale d.
//   q4_dmin[n*(K/256)+..] fp16  — per-256 super-block min   dmin.
// Footprint: K/2 + 2·(K/32) + 2·(K/256)·2 bytes/col = 0.5+0.0625+0.0078
//   ≈ 0.578 B/elem = ~4.625 bpw → the Q4_K GGUF stays ~its on-disk size.
//
// NUMERICS: int-dot W4A8 is NOT bit-exact vs the AoS gemv_q4_K W4A16 path (the
// activation rounds to int8 ~0.4% RMS; fp fold order differs).  But the WEIGHT
// value reconstructed here (value = d·s_raw·q4 − dmin·m_raw) is BIT-IDENTICAL to
// the AoS dequant — the repack is a pure layout move (ie-q4soa-test proves this
// host-side).  Gated on the engine's PPL/cosine checks, exactly like every other
// int-dot decode kernel here (gemv_q6_soa_q8, gemv_q4_0_soa_q8, ...).

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

// get_scale_min_k4 — decode the 6-bit packed (scale,min) for sub-block `sub`
// (0..7) from the 12-byte block_q4_K.scales table.  IDENTICAL to the unpack in
// gemv_q4k.cpp (and ggml's get_scale_min_k4): sub<4 reads a direct 6-bit field,
// sub>=4 stitches a 4-bit low + 2-bit high.  Outputs the raw 6-bit s/m (0..63).
inline void get_scale_min_k4_host(int sub, const uint8_t* scales,
                                  uint8_t& s_raw, uint8_t& m_raw) {
    if (sub < 4) {
        s_raw = scales[sub]     & 0x3F;
        m_raw = scales[sub + 4] & 0x3F;
    } else {
        s_raw = (scales[sub + 4] & 0x0F) | ((scales[sub - 4] >> 6) << 4);
        m_raw = (scales[sub + 4] >>   4) | ((scales[sub    ] >> 6) << 4);
    }
}

// Reconstruct the natural-order 4-bit quant of element `e` (0..255) within a
// block_q4_K, matching gemv_q4_K's (g, hi_nib, half) lattice exactly.
//   In gemv_q4_K: out_off = g*64 + hi_nib*32 + half*16, q4 read from
//   qs[qs_off + i] with qs_off = g*32 + half*16, nibble = low if hi_nib==0
//   else high, sub-block index = g*2 + hi_nib.
//   Inverting for natural e: g = e/64, r = e%64, hi_nib = r/32,
//   half = (r%32)/16, i = e%16 → qs index = g*32 + half*16 + i.
inline uint8_t q4k_nibble_natural(const uint8_t* qs, uint32_t e) {
    const uint32_t g      = e >> 6;          // 0..3
    const uint32_t r      = e & 63;          // 0..63
    const uint32_t hi_nib = r >> 5;          // 0 = low nibble, 1 = high nibble
    const uint32_t half   = (r & 31) >> 4;   // 0 or 1
    const uint32_t i      = e & 15;          // 0..15
    const uint8_t  byte   = qs[g * 32 + half * 16 + i];
    return hi_nib ? uint8_t((byte >> 4) & 0x0F) : uint8_t(byte & 0x0F);
}

}  // namespace

// Host-side repack of a canonical block_q4_K weight [K (contiguous), N] into the
// SoA-Q4 streams above. `W_blocks` = the GGUF tensor data (N columns, each
// K/256 contiguous block_q4_K). Outputs are caller-allocated host buffers:
//   q4_q : N * (K/2) bytes,  q4_sc/q4_mn : N * (K/32) int8,
//   q4_d / q4_dmin : N * (K/256) uint16 (raw fp16).
// Element order is NATURAL (k = 0..K-1), matching the q8 activation stream.
// Reconstructs the SAME value the AoS gemv_q4_K produces (value =
// d·s_raw·q4 − dmin·m_raw); the nibbles are kept 4-bit (no dequant).
//
// NOTE: q4_q is written 2 elems/byte; for even/odd elements the two nibble
// writes of a byte are disjoint, so unlike q6_lo this is NOT a read-modify-write
// across iterations (each byte's two halves are written exactly once).  We still
// build each byte explicitly so the natural order is unambiguous.
void repack_q4_K_to_soa(const void* W_blocks, uint32_t K, uint32_t N,
                        uint8_t* q4_q, int8_t* q4_sc, int8_t* q4_mn,
                        uint16_t* q4_d, uint16_t* q4_dmin) {
    const uint32_t bpc = K / 256;                         // q4_K blocks per column
    const auto* blocks = static_cast<const block_q4_K*>(W_blocks);
    for (uint64_t n = 0; n < N; ++n) {
        const block_q4_K* col = blocks + n * bpc;
        uint8_t*  q_col  = q4_q   + n * (uint64_t(K) / 2);
        int8_t*   sc_col = q4_sc  + n * (uint64_t(K) / 32);
        int8_t*   mn_col = q4_mn  + n * (uint64_t(K) / 32);
        uint16_t* d_col  = q4_d   + n * (uint64_t(K) / 256);
        uint16_t* dm_col = q4_dmin+ n * (uint64_t(K) / 256);
        for (uint32_t b = 0; b < bpc; ++b) {
            const block_q4_K& blk = col[b];
            d_col[b]  = *reinterpret_cast<const uint16_t*>(&blk.d);
            dm_col[b] = *reinterpret_cast<const uint16_t*>(&blk.dmin);
            const uint32_t kb = b * 256;                  // natural base for this super-block
            // 8 sub-blocks of 32 → unpack each (scale,min) once.
            for (int sub = 0; sub < 8; ++sub) {
                uint8_t s_raw, m_raw;
                get_scale_min_k4_host(sub, blk.scales, s_raw, m_raw);
                sc_col[(kb >> 5) + sub] = int8_t(s_raw);   // 0..63 fits int8
                mn_col[(kb >> 5) + sub] = int8_t(m_raw);
            }
            // Emit the 256 nibbles in natural order, 2 elems/byte.
            const uint8_t* qs = blk.qs;
            for (uint32_t e = 0; e < 256; ++e) {
                const uint8_t q4 = q4k_nibble_natural(qs, e);
                const uint32_t k    = kb + e;
                const uint32_t qi   = k >> 1;
                const uint8_t  shft = uint8_t((k & 1) * 4);
                if ((k & 1) == 0) q_col[qi] = q4;          // low nibble (first write)
                else              q_col[qi] = uint8_t(q_col[qi] | (uint32_t(q4) << shft));
            }
        }
    }
}

// W4A8 GEMV over the SoA-Q4 streams.  Mirrors gemv_q6_soa_q8's occupancy
// structure (SG_SIZE=16, N_PER_WG=32, Q8_1 activation staged once in SLM per WG)
// — the proven ~80% BW template.  Split-K: the 16 lanes of an SG stride whole
// 256-elem super-blocks of one column (lane b += 16), each lane accumulates its
// super-blocks, then an SG reduce sums the column.  Per 32-elem sub-block: read
// the 16-byte nibble plane (4 uint32), expand to 8 dp4a words, dp4a vs the
// staged q8, fold (d·s_raw)·idot − (dmin·m_raw)·isum, scale by the q8 block d.
template <bool USE_SLM, int N_PER_WG = 32>
static sycl::event gemv_q4_soa_q8_impl(sycl::queue& q,
                           const void* x_q8,
                           const uint8_t* q4_q,
                           const int8_t* q4_sc, const int8_t* q4_mn,
                           const uint16_t* q4_d, const uint16_t* q4_dmin,
                           sycl::half* y,
                           uint32_t K, uint32_t N,
                           const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE  = 16;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;        // N_PER_WG subgroups/WG

    const auto* X8 = static_cast<const block_q8_1x*>(x_q8);
    const uint32_t blocks_per_col = K / 256;            // super-blocks/col
    const uint32_t q8_blocks      = K / 32;             // q8 blocks (8 per super-block) = sub-blocks
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "gemv_q4_soa", [&](sycl::handler& h) {
        h.depends_on(deps);
        // Stage the Q8 activation stream once per WG (shared by all 32 columns):
        // qs words (8 uint32/block) + per-block d.  K/32 blocks × 36 B usable.
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
            const uint8_t* q_col  = q4_q   + uint64_t(n) * (uint64_t(K) / 2);
            const int8_t*  sc_col = q4_sc  + uint64_t(n) * (uint64_t(K) / 32);
            const int8_t*  mn_col = q4_mn  + uint64_t(n) * (uint64_t(K) / 32);
            const uint16_t* d_col  = q4_d   + uint64_t(n) * (uint64_t(K) / 256);
            const uint16_t* dm_col = q4_dmin+ uint64_t(n) * (uint64_t(K) / 256);

            float acc = 0.f;
            // Split-K: lane strides whole 256-elem super-blocks of this column.
            for (uint32_t sb = lane; sb < blocks_per_col; sb += SG_SIZE) {
                const float d    = soa_fp16_to_fp32(d_col[sb]);
                const float dmin = soa_fp16_to_fp32(dm_col[sb]);
                // 4-bit nibble plane: 128 bytes/super-block = 32 uint32 (256 elems).
                const uint32_t* q_w =
                    reinterpret_cast<const uint32_t*>(q_col + uint64_t(sb) * 128);
                const int8_t* sc = sc_col + uint64_t(sb) * 8;   // 8 sub-blocks/super-block
                const int8_t* mn = mn_col + uint64_t(sb) * 8;
                const uint32_t q8_base = sb * 8;                // 8 q8 blocks per super-block

                // 8 sub-blocks of 32 elements (each = one (s_raw, m_raw) pair).
                // Nibble plane: 32 elems = 16 bytes = 4 uint32 per sub-block.
                // q8: 32 elems = 32 int8 = 8 uint32 (exactly one q8 block).
                #pragma unroll
                for (int s = 0; s < 8; ++s) {
                    const float dsc = d    * float(sc[s]);
                    const float dmn = dmin * float(mn[s]);
                    const uint32_t q8_blk = q8_base + s;
                    // 4 nibble words of this sub-block.
                    const uint32_t w0 = q_w[s * 4 + 0];
                    const uint32_t w1 = q_w[s * 4 + 1];
                    const uint32_t w2 = q_w[s * 4 + 2];
                    const uint32_t w3 = q_w[s * 4 + 3];
                    int32_t idot = 0, isum = 0;
                    // Each nibble word packs 8 elems (2/byte). Expand to TWO
                    // dp4a words of 4 elems each (one nibble per byte, 0..15).
                    #pragma unroll
                    for (int wi = 0; wi < 4; ++wi) {
                        const uint32_t nw =
                            (wi == 0) ? w0 : (wi == 1) ? w1 : (wi == 2) ? w2 : w3;
                        // nw holds elems [8wi .. 8wi+7]: byte j → elems 2j (low),
                        // 2j+1 (high).  Build the two 4-elem dp4a words in natural
                        // element order (the same order the q8 stream stores).
                        const uint32_t b0 = (nw >>  0) & 0xFFu;
                        const uint32_t b1 = (nw >>  8) & 0xFFu;
                        const uint32_t b2 = (nw >> 16) & 0xFFu;
                        const uint32_t b3 = (nw >> 24) & 0xFFu;
                        // elems 8wi+0..3 = {b0&F, b0>>4, b1&F, b1>>4}
                        const uint32_t pack_lo =
                             (b0 & 0x0Fu)              |
                            (((b0 >> 4) & 0x0Fu) << 8) |
                             ((b1 & 0x0Fu)        << 16)|
                            (((b1 >> 4) & 0x0Fu) << 24);
                        // elems 8wi+4..7 = {b2&F, b2>>4, b3&F, b3>>4}
                        const uint32_t pack_hi =
                             (b2 & 0x0Fu)              |
                            (((b2 >> 4) & 0x0Fu) << 8) |
                             ((b3 & 0x0Fu)        << 16)|
                            (((b3 >> 4) & 0x0Fu) << 24);
                        // q8: sub-block covers elems [0,32); word wi*2 / wi*2+1.
                        const uint32_t q8_word_lo = uint32_t(wi) * 2;
                        const uint32_t q8_word_hi = q8_word_lo + 1;
                        const int32_t q8_lo = USE_SLM
                            ? int32_t(q8s[q8_blk * 8 + q8_word_lo])
                            : int32_t(reinterpret_cast<const uint32_t*>(X8[q8_blk].qs)[q8_word_lo]);
                        const int32_t q8_hi = USE_SLM
                            ? int32_t(q8s[q8_blk * 8 + q8_word_hi])
                            : int32_t(reinterpret_cast<const uint32_t*>(X8[q8_blk].qs)[q8_word_hi]);
                        idot = ie::dp4a_us(pack_lo, q8_lo, idot);
                        idot = ie::dp4a_us(pack_hi, q8_hi, idot);
                        isum = ie::dp4a_us(0x01010101u, q8_lo, isum);
                        isum = ie::dp4a_us(0x01010101u, q8_hi, isum);
                    }
                    const float q8d_v = USE_SLM ? q8d[q8_blk] : float(X8[q8_blk].d);
                    // a_i = q8d_v · q8_i ; Σ a_i·(d·s_raw·q4_i − dmin·m_raw)
                    //   = q8d_v · ( dsc·idot − dmn·isum ).
                    acc += q8d_v * (dsc * float(idot) - dmn * float(isum));
                }
            }

            acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
            if (lane == 0) y[n] = sycl::half(acc);
        });
    });
}

// Public entry. Geometry selects how the work-groups map columns → occupancy:
//   default        : N_PER_WG=32 fat WG (512 items), Q8 activation staged in SLM
//                     once per WG, shared across 32 columns.
//   IE_QWEN35_Q4_SOA_GMEM=1 : same fat WG but read activation from global (A/B).
//   IE_QWEN35_Q4_SOA_MMV=1  : llama.cpp geometry — ONE subgroup per WG (16 items,
//                     1 column), activation from global/L2. Many tiny WGs → high
//                     occupancy (attacks the measured 27%-occ / 85%-XVE-stall on
//                     the fat-WG path; this is how llama hits ~64% BW vs our 42%).
//                     Diagnosed 2026-06-22 by the 5-lens decode fan-out: the gap
//                     is workgroup geometry, not dtype (both W4A8) or layout.
sycl::event gemv_q4_soa_q8(sycl::queue& q,
                           const void* x_q8,
                           const uint8_t* q4_q,
                           const int8_t* q4_sc, const int8_t* q4_mn,
                           const uint16_t* q4_d, const uint16_t* q4_dmin,
                           sycl::half* y,
                           uint32_t K, uint32_t N,
                           const std::vector<sycl::event>& deps) {
    static const bool mmv  = std::getenv("IE_QWEN35_Q4_SOA_MMV")  != nullptr;
    static const bool gmem = std::getenv("IE_QWEN35_Q4_SOA_GMEM") != nullptr;
    if (mmv)   // one subgroup per WG, activation in global (no SLM staging)
        return gemv_q4_soa_q8_impl<false, 1>(q, x_q8, q4_q, q4_sc, q4_mn, q4_d, q4_dmin, y, K, N, deps);
    if (gmem)
        return gemv_q4_soa_q8_impl<false, 32>(q, x_q8, q4_q, q4_sc, q4_mn, q4_d, q4_dmin, y, K, N, deps);
    return gemv_q4_soa_q8_impl<true, 32>(q, x_q8, q4_q, q4_sc, q4_mn, q4_d, q4_dmin, y, K, N, deps);
}

// ===========================================================================
// gemv_q4_K_soa_q8_v2 — COALESCED-LOAD Q4_K decode GEMV (2026-07-17).
//
// The clean isolation of the llama-SYCL coalescing win, UNBUNDLED from SLM
// staging (which masked it in the split-K gemv_q4_soa_q8 A/Bs — PERF_LEDGER
// 2026-06-22 "SoA REGRESSED", diagnosed as bundled SLM-barrier + occupancy cost
// hiding the layout win). Reads the SAME natural-order SoA-Q4 streams produced by
// repack_q4_K_to_soa (bit-identical repack, ie-q4soa-test), but changes the
// LANE→WORD mapping so a subgroup's 16 lanes iterate the 32-word nibble plane of
// ONE super-block TOGETHER: lane L reads word L and word L+16 → two fully
// subgroup-COALESCED 64 B global loads (W32[b*32+lane], W32[b*32+16+lane]). The
// split-K variant instead had each lane stride whole 256-elem super-blocks →
// 128 B per-instruction lane scatter. THIS relayout is the isolated experiment.
//
// Three coupled numeric-preserving levers ship together (else the coalescing win
// is masked): (1) the coalesced load above; (2) NO SLM staging / NO barrier —
// the Q8 activation is read straight from global/L2 (mirrors the USE_SLM=false
// _g variant that won for Q6-SoA, gemv_q6_soa.cpp); (3) native f16→f32 via
// bit_cast<half> instead of the branchy denormal-loop soa_fp16_to_fp32.
//
// NUMERICS: keeps the ones-dp4a Σq8 ("isum") min term, so every per-element
// int-dot operand (q4 nibble, q8, s_raw/m_raw, d/dmin, q8-d) is IDENTICAL to the
// default gemv_q4_K_q8 — only the cross-lane fp accumulation GROUPING differs
// (last-bit reassoc). PPL must match the default. The per-16 stored-halfsum
// (isum-elimination, block_q8_1s.s0/s1) is a deliberate FOLLOW-ON (Stage 4), NOT
// done here, so this stage changes zero numerics vs today's default.
//
// LANE MAP (per super-block b: 32 words / 256 elems; natural order 2 elems/byte):
//   word L    → natural elems [8L,   8L+8)   → sub-block sub_a = L>>2  (0..3)
//   word L+16 → natural elems [8L+128,  +8)  → sub-block sub_b = sub_a+4 (4..7)
//   both      → q8 word offset 2*(L&3), +1   (identical for the two words)
// The 4 lanes {4s..4s+3} tile sub-block s (via word L) and s+4 (via word L+16),
// so the 16 lanes cover all 8 sub-blocks; the SG-reduce sums the column.
template <int N_PER_WG = 32>
static sycl::event gemv_q4_K_soa_q8_v2_impl(sycl::queue& q,
                           const void* x_q8,
                           const uint8_t* q4_q,
                           const int8_t* q4_sc, const int8_t* q4_mn,
                           const uint16_t* q4_d, const uint16_t* q4_dmin,
                           sycl::half* y,
                           uint32_t K, uint32_t N,
                           const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE  = 16;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;

    const auto* X8 = static_cast<const block_q8_1x*>(x_q8);
    const uint32_t blocks_per_col = K / 256;
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "gemv_q4k_soa_v2", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t wgid  = uint32_t(it.get_group(0));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t n     = wgid * N_PER_WG + sg_id;
            if (n >= N) return;

            // Per-column SoA streams (column-major by output n). q4_q is 4-byte
            // aligned per column (K/2 is a multiple of 128), so the uint32 view
            // gives 32 words per 128-byte super-block nibble plane.
            const uint32_t* q_col = reinterpret_cast<const uint32_t*>(
                q4_q + uint64_t(n) * (uint64_t(K) / 2));
            const int8_t*   sc_col = q4_sc  + uint64_t(n) * (uint64_t(K) / 32);
            const int8_t*   mn_col = q4_mn  + uint64_t(n) * (uint64_t(K) / 32);
            const uint16_t* d_col  = q4_d   + uint64_t(n) * (uint64_t(K) / 256);
            const uint16_t* dm_col = q4_dmin+ uint64_t(n) * (uint64_t(K) / 256);

            // This lane's two sub-blocks + the (shared) q8 uint32 word offset.
            const int sub_a = int(lane) >> 2;          // 0..3
            const int sub_b = sub_a + 4;               // 4..7
            const int q8w0  = (int(lane) & 3) * 2;     // 0/2/4/6
            const int q8w1  = q8w0 + 1;

            // Expand one nibble word (8 elems, 2/byte, natural order) into two
            // dp4a words of 4 int8 lanes each (byte j → elems 2j low, 2j+1 high).
            auto expand = [](uint32_t nw, uint32_t& plo, uint32_t& phi) {
                const uint32_t b0 = (nw >>  0) & 0xFFu, b1 = (nw >>  8) & 0xFFu;
                const uint32_t b2 = (nw >> 16) & 0xFFu, b3 = (nw >> 24) & 0xFFu;
                plo = (b0 & 0xFu) | (((b0 >> 4) & 0xFu) << 8) |
                      ((b1 & 0xFu) << 16) | (((b1 >> 4) & 0xFu) << 24);
                phi = (b2 & 0xFu) | (((b2 >> 4) & 0xFu) << 8) |
                      ((b3 & 0xFu) << 16) | (((b3 >> 4) & 0xFu) << 24);
            };

            float acc = 0.f;
            for (uint32_t b = 0; b < blocks_per_col; ++b) {
                const float d    = float(sycl::bit_cast<sycl::half>(d_col[b]));
                const float dmin = float(sycl::bit_cast<sycl::half>(dm_col[b]));
                const int8_t* sc = sc_col + uint64_t(b) * 8;
                const int8_t* mn = mn_col + uint64_t(b) * 8;

                // Two coalesced 64 B subgroup loads of the 128 B nibble plane.
                const uint32_t nwA = q_col[b * 32 + lane];
                const uint32_t nwB = q_col[b * 32 + 16 + lane];

                // q8 blocks for this lane's two sub-blocks (global/L2, no SLM).
                const uint32_t* xqa = reinterpret_cast<const uint32_t*>(X8[b * 8 + sub_a].qs);
                const uint32_t* xqb = reinterpret_cast<const uint32_t*>(X8[b * 8 + sub_b].qs);
                const float d8a = float(X8[b * 8 + sub_a].d);
                const float d8b = float(X8[b * 8 + sub_b].d);

                // Stage-4 isum-elimination: the Q4_K per-sub-block min bias is
                // dmin·m_raw·Σq8. block_q8_1x.s ALREADY stores d8·Σq8 (per-32,
                // exactly the Q4_K sub-block granularity — quantize_q8_1), so the
                // whole bias is dmin·m_raw·(d8·Σ32) with NO ones-dp4a. Add it once
                // per sub-block on the lane that owns q8 word 0 of that sub-block
                // (lane&3==0 of the 4 lanes tiling it); the SG-reduce folds it in.
                // Halves the inner dp4a (drops the two isum chains). Algebraically
                // identical to the ones-dp4a form — only fp grouping of the bias
                // differs (last-bit reassoc, same class as Stage 1).
                const bool min_lane = ((lane & 3) == 0);

                // --- sub-block A (word L, elems [8L, 8L+8)) ---
                {
                    uint32_t plo, phi; expand(nwA, plo, phi);
                    const int32_t x0 = int32_t(xqa[q8w0]), x1 = int32_t(xqa[q8w1]);
                    int32_t idot = ie::dp4a_us(plo, x0, 0);
                    idot = ie::dp4a_us(phi, x1, idot);
                    acc += d8a * d * float(sc[sub_a]) * float(idot);
                }
                // --- sub-block B (word L+16, elems [8L+128, +8)) ---
                {
                    uint32_t plo, phi; expand(nwB, plo, phi);
                    const int32_t x0 = int32_t(xqb[q8w0]), x1 = int32_t(xqb[q8w1]);
                    int32_t idot = ie::dp4a_us(plo, x0, 0);
                    idot = ie::dp4a_us(phi, x1, idot);
                    acc += d8b * d * float(sc[sub_b]) * float(idot);
                }
                // Stage-4 min bias for both sub-blocks, once, via the stored sum.
                if (min_lane) {
                    const float sA = float(X8[b * 8 + sub_a].s);   // d8a·Σ32 for sub_a
                    const float sB = float(X8[b * 8 + sub_b].s);   // d8b·Σ32 for sub_b
                    acc -= dmin * (float(mn[sub_a]) * sA + float(mn[sub_b]) * sB);
                }
            }
            acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
            if (lane == 0) y[n] = sycl::half(acc);
        });
    });
}

// Public entry. One SG per output column, N_PER_WG=32 SGs/WG, global activation
// (no SLM), matching the winning Q6-SoA _g launch geometry.
sycl::event gemv_q4_K_soa_q8_v2(sycl::queue& q,
                           const void* x_q8,
                           const uint8_t* q4_q,
                           const int8_t* q4_sc, const int8_t* q4_mn,
                           const uint16_t* q4_d, const uint16_t* q4_dmin,
                           sycl::half* y,
                           uint32_t K, uint32_t N,
                           const std::vector<sycl::event>& deps) {
    return gemv_q4_K_soa_q8_v2_impl<32>(q, x_q8, q4_q, q4_sc, q4_mn, q4_d, q4_dmin,
                                        y, K, N, deps);
}

// V2 write-coalesced prefill dequant gate (IE_Q4_DEQUANT_V2=1, DEFAULT OFF for
// a clean A/B vs the scatter-write baseline below). Read once.
static bool q4_dequant_v2() {
    static const bool v = []{
        const char* e = std::getenv("IE_Q4_DEQUANT_V2");
        return e && std::atoi(e) != 0;
    }();
    return v;
}

// V2 write-coalesced dequant (IE_Q4_DEQUANT_V2=1): same Bt[K,N] (n fastest) and
// the same per-element math as the baseline below — bit-exact output — but the
// item map is (super-block b, column n) with n the fastest-varying LOCAL
// dimension (the proven dequant_q2_0_soa_to_Bt_v2 geometry). At every inner
// step the WG's 64 lanes store 64 CONSECUTIVE fp16 of one Bt row (128 B
// contiguous), where the baseline's linear (n-major) map scatters adjacent
// lanes' 2 B stores 256·N·2 B apart — one cacheline transaction per store.
// Reads are wide uint4s (nibbles: 128 B/block = 8); alignment holds because
// K % 256 == 0 keeps every per-column block run 16 B-aligned.
static sycl::event dequant_q4_soa_to_Bt_v2(sycl::queue& q,
                                 const uint8_t* q4_q,
                                 const int8_t* q4_sc, const int8_t* q4_mn,
                                 const uint16_t* q4_d, const uint16_t* q4_dmin,
                                 sycl::half* Bt, uint32_t K, uint32_t N,
                                 const std::vector<sycl::event>& deps) {
    const uint32_t bpc = K / 256;
    constexpr uint32_t WG_N = 64;
    const uint32_t n_pad = ((N + WG_N - 1) / WG_N) * WG_N;
    return ie::ps(q, "dequant_q4_soa", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<2>({bpc, n_pad}, {1, WG_N}),
                       [=](sycl::nd_item<2> it) {
            const uint32_t sb = uint32_t(it.get_group(0));
            const uint32_t n  = uint32_t(it.get_global_id(1));
            if (n >= N) return;

            const auto* qd4 = reinterpret_cast<const sycl::uint4*>(
                q4_q + uint64_t(n) * (uint64_t(K) / 2) + uint64_t(sb) * 128u);
            const int8_t* sc = q4_sc + uint64_t(n) * (uint64_t(K) / 32) + uint64_t(sb) * 8u;
            const int8_t* mn = q4_mn + uint64_t(n) * (uint64_t(K) / 32) + uint64_t(sb) * 8u;
            const float   d  = soa_fp16_to_fp32(q4_d  [uint64_t(n) * (uint64_t(K) / 256) + sb]);
            const float   dm = soa_fp16_to_fp32(q4_dmin[uint64_t(n) * (uint64_t(K) / 256) + sb]);

            sycl::half* dst = Bt + uint64_t(sb) * 256u * N + n;
            for (int c = 0; c < 4; ++c) {          // 64 elems per chunk
                const sycl::uint4 v0 = qd4[c * 2], v1 = qd4[c * 2 + 1];
                const uint32_t qw[8] = {v0.x(), v0.y(), v0.z(), v0.w(),
                                        v1.x(), v1.y(), v1.z(), v1.w()};
                sycl::half* dstc = dst + uint64_t(c) * 64u * N;
                #pragma unroll
                for (int j = 0; j < 64; ++j) {
                    // Elem e = 64c + j. Byte (e>>1) read little-endian out of
                    // the words = the baseline's exact per-byte load; nibble
                    // select unchanged (e&1 == j&1); sub-block e>>5 == 2c+(j>>5).
                    const uint32_t byte = (qw[j >> 3] >> (((j >> 1) & 3) * 8)) & 0xFFu;
                    const int q4 = int((byte >> ((j & 1) * 4)) & 0x0F);
                    const int s  = c * 2 + (j >> 5);        // sub-block 0..7
                    const float val = d * float(sc[s]) * float(q4) - dm * float(mn[s]);
                    dstc[uint64_t(j) * N] = sycl::half(val);
                }
            }
        });
    });
}

// Dequant the SoA-Q4 streams → fp16 Bt[K,N] (K contiguous, N columns), so the
// prefill path (gemm_fp16) consumes it exactly like the AoS dequant output. One
// work-item per (n, super-block): reconstructs the block's 256 values
// (value = d·s_raw·q4 − dmin·m_raw).  Used only at T≥2 (prefill); same value
// math as the AoS path (fp16 store).
sycl::event dequant_q4_soa_to_Bt(sycl::queue& q,
                                 const uint8_t* q4_q,
                                 const int8_t* q4_sc, const int8_t* q4_mn,
                                 const uint16_t* q4_d, const uint16_t* q4_dmin,
                                 sycl::half* Bt, uint32_t K, uint32_t N,
                                 const std::vector<sycl::event>& deps) {
    if (q4_dequant_v2())
        return dequant_q4_soa_to_Bt_v2(q, q4_q, q4_sc, q4_mn, q4_d, q4_dmin,
                                       Bt, K, N, deps);
    const uint32_t bpc = K / 256;
    return ie::ps(q, "dequant_q4_soa", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(uint64_t(N) * bpc), [=](sycl::id<1> idx) {
            const uint64_t lin = idx;
            const uint32_t n  = uint32_t(lin / bpc);
            const uint32_t sb = uint32_t(lin % bpc);
            const uint8_t* qd  = q4_q   + uint64_t(n) * (uint64_t(K) / 2)  + uint64_t(sb) * 128;
            const int8_t*  sc  = q4_sc  + uint64_t(n) * (uint64_t(K) / 32) + uint64_t(sb) * 8;
            const int8_t*  mn  = q4_mn  + uint64_t(n) * (uint64_t(K) / 32) + uint64_t(sb) * 8;
            const float    d   = soa_fp16_to_fp32(q4_d  [uint64_t(n) * (uint64_t(K) / 256) + sb]);
            const float    dm  = soa_fp16_to_fp32(q4_dmin[uint64_t(n) * (uint64_t(K) / 256) + sb]);
            const uint32_t kb  = sb * 256;
            for (uint32_t e = 0; e < 256; ++e) {
                const uint32_t k   = kb + e;
                const uint8_t  byte = qd[e >> 1];
                const int q4 = int((byte >> ((e & 1) * 4)) & 0x0F);
                const int s  = int(e >> 5);                 // sub-block 0..7
                const float val = d * float(sc[s]) * float(q4) - dm * float(mn[s]);
                Bt[uint64_t(k) * N + n] = sycl::half(val);
            }
        });
    });
}

}  // namespace ie
