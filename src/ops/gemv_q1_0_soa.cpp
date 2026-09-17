// src/ops/gemv_q1_0_soa.cpp — Prism 1-bit Q1_0 fast decode GEMV via a load-time
// SoA repack + coalesced int-dot (dp4a, W1A8) inner loop.  2026-07-18.
//
// Q1_0 (llama.cpp-prism ggml_type 41): 128 weights/block, 18 B = fp16 d (= mean-
// abs over the block — NOT amax like Q2_0) + 16 B qs (1 bit/weight, NATURAL order,
// LSB-first within each byte — byte j/8 holds weight j at bit (j%8)).  bit b ∈
// {0,1} decodes to value (2b−1)·d ∈ {−d, +d} — sign-only, NO zero code.  This is
// the 1-bit cousin of Q2_0: same per-128 fp16 scale and same d plane, but half the
// qs bytes.  The on-disk qs is already the natural-order 8-elems/byte plane the
// int-dot kernel wants, so the "repack" only separates d from qs and makes each
// column's qs contiguous.
//
// SoA-Q1 layout (per output column n, contiguous — column-major by output n):
//   q1_qs[n*(K/8) + b*16 + ...]  uint8 — 1-bit signs, 8 elems/byte, natural order.
//                                 As uint32: 4 words/block, word w = elems
//                                 [32w, 32w+32); bit r of that word = elem 32w+r.
//   q1_d [n*(K/128) + b]         fp16  — per-128 block scale (raw fp16 bits).
// Footprint: K/8 + (K/128)*2 = 0.125 + 0.015625 B/elem = 1.125 bpw (== on-disk).
//
// DECODE KERNEL (gemv_q1_0_soa_q8): the #1c-vec wide-load geometry of
// gemv_q2_0_soa_q8_vec_impl, adapted to Q1_0's 1-bit blocks.  One subgroup (16
// lanes) owns one output column.  Each lane reads ONE uint4 (16 B = 128 codes =
// ONE whole q1 block = FOUR whole 32-elem Q8_1 activation blocks) per step, so the
// 16 lanes cover 16 whole blocks (2048 elems) via one 256 B coalesced subgroup
// load.  Lane map blk = s*16 + lane.  Because each lane spans WHOLE q8 blocks the
// deferred (−1) correction is EXACT and FREE via block_q8_1x.s: value = (2b−1)·d,
// so Σ value·act = d·(2·d8·idot − s) per whole q8 block, idot = dp4a over the
// {0,1} sign bytes.  The ×2 folds the +1 half of the sign; the −s subtracts the
// −1 half (s = d8·Σqs, precomputed at activation quant).  No ones-dp4a chain.
// Each qv u32 word = 32 codes → 8 dp4a words via 1-bit spreads (per-BYTE ALU
// doubles vs Q2 — the perf risk noted in the port map).  Global activation (no
// SLM/barrier), native f16→f32 via bit_cast<half>.  K % 128 == 0 keeps each
// column's uint4 base aligned (K/8 % 16 == K/128) — always true for a Q1_0 tensor;
// the scalar baseline below is the IE_Q1_VEC=0 kill switch.
//
// NUMERICS: int-dot W1A8 is NOT bit-exact vs fp16-dequant Q1_0 (activation rounds
// to int8; fp fold order differs) — same class as the Q2 int-dot decode kernel.
// The integer part (Σ bit·q8) is EXACT (sign bits are exactly representable), so
// the only error is the q8 activation rounding — a tighter tolerance than Q2's.
// Gated on the engine's PPL/coherence checks.

#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"

#include <cstdlib>
#include <sycl/sycl.hpp>
#include "ie/dp4a.hpp"
#include "ie/kernel_profiler.hpp"

namespace ie {

// Host-side repack of a canonical block_q1_0 weight [K (contiguous), N] into the
// SoA-Q1 streams above. `W_blocks` = the GGUF tensor data (N columns, each K/128
// contiguous block_q1_0). Outputs are caller-allocated host buffers:
//   q1_qs : N * (K/8) bytes,  q1_d : N * (K/128) uint16 (raw fp16).
// The on-disk qs bytes are already NATURAL order (byte j/8 = weights j..j+7), so
// each block's 16 qs bytes are copied verbatim; only d is split out into its own
// per-column-contiguous plane.
void repack_q1_0_to_soa(const void* W_blocks, uint32_t K, uint32_t N,
                        uint8_t* q1_qs, uint16_t* q1_d) {
    const uint32_t bpc = K / 128;                         // q1_0 blocks per column
    const auto* blocks = static_cast<const block_q1_0*>(W_blocks);
    for (uint64_t n = 0; n < N; ++n) {
        const block_q1_0* col = blocks + n * bpc;
        uint8_t*  qs_col = q1_qs + n * (uint64_t(K) / 8);
        uint16_t* d_col  = q1_d  + n * (uint64_t(K) / 128);
        for (uint32_t b = 0; b < bpc; ++b) {
            d_col[b] = *reinterpret_cast<const uint16_t*>(&col[b].d);
            const uint8_t* src = col[b].qs;
            uint8_t* dst = qs_col + uint64_t(b) * 16;
            for (int i = 0; i < 16; ++i) dst[i] = src[i];   // natural order preserved
        }
    }
}

// Dot one whole q8 block against one uint32 sign-word (32 codes). Unpacks the
// word into 8 dp4a words via 1-bit spreads (dp4a word w = codes [4w, 4w+4),
// LSB-first: c_r = (word >> (4w+r)) & 1). idot = Σ bit·q8.
static inline float q1_block_contrib(uint32_t word, const block_q8_1x& xb) {
    const uint32_t* xq = reinterpret_cast<const uint32_t*>(xb.qs);
    int32_t idot = 0;
    #pragma unroll
    for (int w = 0; w < 8; ++w) {
        const uint32_t b0 = (word >> (4 * w + 0)) & 1u;
        const uint32_t b1 = (word >> (4 * w + 1)) & 1u;
        const uint32_t b2 = (word >> (4 * w + 2)) & 1u;
        const uint32_t b3 = (word >> (4 * w + 3)) & 1u;
        const uint32_t pack = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
        idot = ie::dp4a_us(pack, int32_t(xq[w]), idot);
    }
    // Exact deferred −1 with the ×2 coefficient: value = (2b−1)·d1, so
    //   Σ value·act = d1·(2·d8·idot − s)  per whole q8 block  (x.s = d8·Σqs).
    // d1 is applied by the caller.
    return 2.f * xb.d * float(idot) - xb.s;
}

// #1c-vec wide-load W1A8 int-dot GEMV over the SoA-Q1 streams. One SG per output
// column, N_PER_WG SGs per WG, global activation. One whole 128-elem block per
// lane per step (one uint4 = 4 whole q8 blocks), 16 blocks per subgroup-step.
template <int N_PER_WG = 16>
static sycl::event gemv_q1_0_soa_q8_vec_impl(sycl::queue& q,
                           const void* x_q8,
                           const uint8_t* q1_qs, const uint16_t* q1_d,
                           sycl::half* y, uint32_t K, uint32_t N,
                           const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE  = 16;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;

    const auto* X8 = static_cast<const block_q8_1x*>(x_q8);
    const uint32_t blocks_per_col = K / 128;                 // 128-elem blocks/col
    const uint32_t n_steps = (blocks_per_col + 15) / 16;     // 16 blocks/step
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "gemv_q1_soa", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t wgid  = uint32_t(it.get_group(0));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t n     = wgid * N_PER_WG + sg_id;
            if (n >= N) return;

            // Column base is K/8 bytes; K % 128 == 0 keeps it uint4-aligned
            // (K/8 % 16 == K/128, integer). One uint4 = 16 B = one whole q1 block.
            const auto* qs4 = reinterpret_cast<const sycl::uint4*>(
                q1_qs + uint64_t(n) * (uint64_t(K) / 8));
            const uint16_t* d_col = q1_d + uint64_t(n) * (uint64_t(K) / 128);

            float acc = 0.f;
            for (uint32_t s = 0; s < n_steps; ++s) {
                // Lane L owns block blk = s*16 + L (128 codes = 4 whole q8 blocks).
                // Coalesced: across the SG that is uint4 indices {s*16 .. s*16+15}
                // → one 256 B subgroup load. Masked lanes (blk >= bpc on the last
                // step) get d1 = 0 → zero contribution, so the reduce stays whole.
                const uint32_t blk = s * 16u + lane;
                const bool v = blk < blocks_per_col;
                const float d1 = v ? float(sycl::bit_cast<sycl::half>(d_col[v ? blk : 0u])) : 0.f;

                const sycl::uint4 qv = v ? qs4[blk] : sycl::uint4{0u, 0u, 0u, 0u};
                const uint32_t qw[4] = {qv.x(), qv.y(), qv.z(), qv.w()};

                const uint32_t qb0 = blk * 4u;              // q8 blocks [qb0, qb0+4)
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    // u32 word i = 32 codes = one whole q8 block (elems 32i..32i+31).
                    const block_q8_1x& xi = X8[v ? (qb0 + uint32_t(i)) : 0u];
                    acc += d1 * q1_block_contrib(qw[i], xi);
                }
            }

            acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
            if (lane == 0) y[n] = sycl::half(acc);
        });
    });
}

// Scalar baseline (IE_Q1_VEC=0 kill switch / general fallback). Same lane map and
// per-block math as the vec kernel, but reads each block's 16 qs bytes as four
// scalar uint32 words instead of one wide uint4 — needs only 4-byte column-base
// alignment (K % 32 == 0), so it covers any Q1_0 shape the uint4 geometry can't.
template <int N_PER_WG = 16>
static sycl::event gemv_q1_0_soa_q8_scalar_impl(sycl::queue& q,
                           const void* x_q8,
                           const uint8_t* q1_qs, const uint16_t* q1_d,
                           sycl::half* y, uint32_t K, uint32_t N,
                           const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE  = 16;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;

    const auto* X8 = static_cast<const block_q8_1x*>(x_q8);
    const uint32_t blocks_per_col = K / 128;
    const uint32_t words_per_col  = K / 32;                  // uint32 words of qs/col
    const uint32_t n_steps = (blocks_per_col + 15) / 16;     // 16 blocks/step
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "gemv_q1_soa", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t wgid  = uint32_t(it.get_group(0));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t n     = wgid * N_PER_WG + sg_id;
            if (n >= N) return;

            const uint32_t* qs32 = reinterpret_cast<const uint32_t*>(
                q1_qs + uint64_t(n) * (uint64_t(K) / 8));
            const uint16_t* d_col = q1_d + uint64_t(n) * (uint64_t(K) / 128);

            float acc = 0.f;
            for (uint32_t s = 0; s < n_steps; ++s) {
                const uint32_t blk = s * 16u + lane;
                const bool v = blk < blocks_per_col;
                const float d1 = v ? float(sycl::bit_cast<sycl::half>(d_col[v ? blk : 0u])) : 0.f;

                const uint32_t wbase = blk * 4u;            // qs words [wbase, +4)
                const uint32_t qb0   = blk * 4u;            // q8 blocks [qb0, +4)
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const uint32_t widx = wbase + uint32_t(i);
                    const uint32_t word = (v && widx < words_per_col) ? qs32[widx] : 0u;
                    const block_q8_1x& xi = X8[v ? (qb0 + uint32_t(i)) : 0u];
                    acc += d1 * q1_block_contrib(word, xi);
                }
            }

            acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
            if (lane == 0) y[n] = sycl::half(acc);
        });
    });
}

// Sub-groups per WG for the Q1 decode GEMV. Default 16 to match the shipped Q2
// VEC default (docs/q2_0_optimization/05 A/B). IE_Q1_NPWG ∈ {8,16,32,64}
// overrides. Read once.
static uint32_t q1_npwg() {
    static const uint32_t v = []{
        if (const char* e = std::getenv("IE_Q1_NPWG")) {
            int n = std::atoi(e); if (n==8||n==16||n==32||n==64) return uint32_t(n);
        }
        return 16u;
    }();
    return v;
}

// #1c wide-load variant: DEFAULT ON (mirrors the shipped Q2 IE_Q2_VEC default).
// IE_Q1_VEC=0 opts out (kill switch → scalar baseline). K%128!=0 (impossible for
// a Q1_0 tensor) also falls through to the baseline. Read once.
static bool q1_vec() {
    static const bool v = []{
        const char* e = std::getenv("IE_Q1_VEC");
        return !e || std::atoi(e) != 0;
    }();
    return v;
}

sycl::event gemv_q1_0_soa_q8(sycl::queue& q,
                             const void* x_q8,
                             const uint8_t* q1_qs, const uint16_t* q1_d,
                             sycl::half* y, uint32_t K, uint32_t N,
                             const std::vector<sycl::event>& deps) {
    if (q1_vec() && (K % 128u) == 0u) {
        switch (q1_npwg()) {
            case 8:  return gemv_q1_0_soa_q8_vec_impl<8> (q, x_q8, q1_qs, q1_d, y, K, N, deps);
            case 32: return gemv_q1_0_soa_q8_vec_impl<32>(q, x_q8, q1_qs, q1_d, y, K, N, deps);
            case 64: return gemv_q1_0_soa_q8_vec_impl<64>(q, x_q8, q1_qs, q1_d, y, K, N, deps);
            default: return gemv_q1_0_soa_q8_vec_impl<16>(q, x_q8, q1_qs, q1_d, y, K, N, deps);
        }
    }
    switch (q1_npwg()) {
        case 8:  return gemv_q1_0_soa_q8_scalar_impl<8> (q, x_q8, q1_qs, q1_d, y, K, N, deps);
        case 32: return gemv_q1_0_soa_q8_scalar_impl<32>(q, x_q8, q1_qs, q1_d, y, K, N, deps);
        case 64: return gemv_q1_0_soa_q8_scalar_impl<64>(q, x_q8, q1_qs, q1_d, y, K, N, deps);
        default: return gemv_q1_0_soa_q8_scalar_impl<16>(q, x_q8, q1_qs, q1_d, y, K, N, deps);
    }
}

// V2 write-coalesced dequant: SoA-Q1 streams → fp16 Bt[K,N] (K contiguous, N
// columns — n fastest), the layout gemm_fp16 consumes for the T≥2 prefill path
// (like dequant_q2_0_soa_to_Bt_v2). Item map is (block b, column n) with n the
// fastest-varying LOCAL dim: at every step the WG's 64 lanes store 64 consecutive
// fp16 of one Bt row (128 B contiguous). One uint4 (16 B = a whole 128-code block)
// per work-item; alignment holds because K % 128 == 0.
sycl::event dequant_q1_0_soa_to_Bt(sycl::queue& q,
                                   const uint8_t* q1_qs, const uint16_t* q1_d,
                                   sycl::half* Bt, uint32_t K, uint32_t N,
                                   const std::vector<sycl::event>& deps) {
    const uint32_t bpc = K / 128;
    constexpr uint32_t WG_N = 64;
    const uint32_t n_pad = ((N + WG_N - 1) / WG_N) * WG_N;
    return ie::ps(q, "dequant_q1_soa", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<2>({bpc, n_pad}, {1, WG_N}),
                       [=](sycl::nd_item<2> it) {
            const uint32_t b = uint32_t(it.get_group(0));
            const uint32_t n = uint32_t(it.get_global_id(1));
            if (n >= N) return;

            // One uint4 = 16 B = a whole 128-code block of this column.
            const auto* qs4 = reinterpret_cast<const sycl::uint4*>(
                q1_qs + uint64_t(n) * (uint64_t(K) / 8) + uint64_t(b) * 16u);
            const sycl::uint4 v0 = qs4[0];
            const uint32_t qw[4] = {v0.x(), v0.y(), v0.z(), v0.w()};
            const float d1 = float(sycl::bit_cast<sycl::half>(
                q1_d[uint64_t(n) * (uint64_t(K) / 128) + b]));

            sycl::half* dst = Bt + uint64_t(b) * 128u * N + n;
            #pragma unroll
            for (int w = 0; w < 4; ++w) {
                #pragma unroll
                for (int j = 0; j < 32; ++j) {
                    // Elem e = 32w + j; bit-in-word = j selects the SAME LSB-first
                    // bit the reference takes from qs[e>>3].
                    const uint32_t bit = (qw[w] >> j) & 0x1u;
                    const float    val = (int(bit) * 2 - 1) * d1;
                    dst[uint64_t(w * 32 + j) * N] = sycl::half(val);
                }
            }
        });
    });
}

}  // namespace ie
