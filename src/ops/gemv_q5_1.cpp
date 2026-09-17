// src/ops/gemv_q5_1.cpp — W5A16 GEMV/GEMM with Q5_1 weights (qwen4exp).
//
// Q5_1 is the quant format of the Qwen3.8-Flash-Next routed-expert down bank
// (ffn_down_exps [640, 2560, 512]: K=640 = 20 blocks per output column).
// Block = 32 elements, 24 bytes: fp16 scale `d` + fp16 min `m` + 4-byte qh
// (5th bit of each quant, little-endian uint32) + 16 bytes of low nibbles.
// ggml layout (== dequant_ref.hpp dequant_q5_1):
//   xh0 = ((qh >> j)      << 4) & 0x10 ;  x0 = (qs[j] & 0x0F) | xh0
//   xh1 = ((qh >> (j+12))      ) & 0x10 ;  x1 = (qs[j] >>  4)  | xh1
//   w[j] = d*x0 + m ;  w[j+16] = d*x1 + m       (j in 0..15, unsigned + min)
//
// Weight layout matches gemv_q4_0/gemv_q4_K: W[K, N] column-major-packed —
// each output column n is K/32 consecutive blocks. y = A @ W, K % 32 == 0.
//
// Correctness-first kernels (mirror gemv_q4_0/gemm_q4_0's WG/SG shape): one
// subgroup per output column; lane L reads qs[L] and contributes elements L
// and L+16 of each block; fp32 accumulate; subgroup-reduce; lane 0 writes.
//
// Also here (host-side, load-time): convert_bf16_to_f16 (indexer projections)
// and convert_q5_k_to_f16_buffer (the 2 F16-resident Q5_K expert banks).

#include "ie/qwen4_quant.hpp"
#include "ie/dequant_ref.hpp"
#include "ie/quant_blocks.hpp"

#include <sycl/sycl.hpp>
#include "ie/kernel_profiler.hpp"

namespace ie {

// (The old software q51_fp16_to_fp32 helper is gone: the kernels use the
// native half->float conversion, which is exact and therefore bit-identical.)

sycl::event gemv_q5_1(sycl::queue& q,
                      const sycl::half* A, const void* W_packed,
                      sycl::half* y,
                      uint32_t K, uint32_t N,
                      const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE  = 16;
    constexpr int N_PER_WG = 16;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;       // 256
    const auto* W = static_cast<const block_q5_1*>(W_packed);
    const uint32_t blocks_per_col = K / 32;
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "gemv_q5_1", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<sycl::half, 1> A_slm(sycl::range<1>(K), h);
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t wgid  = uint32_t(it.get_group(0));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t n     = wgid * N_PER_WG + sg_id;

            for (uint32_t i = lid; i < K; i += WG_ITEMS) A_slm[i] = A[i];
            sycl::group_barrier(it.get_group());
            if (n >= N) return;

            float acc = 0.f;
            const block_q5_1* col = &W[uint64_t(n) * blocks_per_col];
            for (uint32_t b = 0; b < blocks_per_col; ++b) {
                const block_q5_1& blk = col[b];
                // Native half->float (exact conversion, bit-identical to the
                // old software path) — the branchy sw convert ran 2x per
                // block per lane and was the top ALU cost of this
                // dequant-bound kernel. qh via two u16 loads (alignas(2)
                // struct — a u32 load is not guaranteed aligned); the
                // assembled value is identical.
                const float d = float(sycl::bit_cast<sycl::half>(blk.d));
                const float m = float(sycl::bit_cast<sycl::half>(blk.m));
                const uint16_t* qh16 = reinterpret_cast<const uint16_t*>(blk.qh);
                const uint32_t qh = uint32_t(qh16[0]) | (uint32_t(qh16[1]) << 16);
                const uint8_t qb  = blk.qs[lane];
                const uint32_t xh0 = ((qh >> lane)        << 4) & 0x10u;
                const uint32_t xh1 = ((qh >> (lane + 12))     ) & 0x10u;
                const float w_lo = float((qb & 0x0Fu) | xh0) * d + m;
                const float w_hi = float((qb >>  4u ) | xh1) * d + m;
                acc += float(A_slm[b * 32 + lane])      * w_lo +
                       float(A_slm[b * 32 + lane + 16]) * w_hi;
            }
            acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
            if (lane == 0) y[n] = sycl::half(acc);
        });
    });
}

// GROUPED down-proj GEMV over an expert-pick job list (spec-verify MoE,
// 2026-08-27): one launch for ALL picks. Group id decodes to (pick,
// column-tile); within a (pick, column) the solo gemv_q5_1 body runs
// VERBATIM (same SLM A stage of the pick's h row, same lane roles, same
// block loop, same subgroup reduce) -> bit-identical to the per-pick solo
// calls (lossless contract). jobs = [P x 3] int32 {vt, slot, ycell}; pick p
// reads h_rows + p*K and the down slice at slot_base + slot*slot_bytes +
// down_off, writing ystage + ycell*y_stride.
sycl::event gemv_q5_1_grouped(sycl::queue& q,
                      const sycl::half* h_rows,
                      const uint8_t* slot_base, uint64_t slot_bytes,
                      uint64_t down_off,
                      const int32_t* jobs,
                      sycl::half* ystage, uint32_t y_stride,
                      uint32_t K, uint32_t N, uint32_t P,
                      const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE  = 16;
    constexpr int N_PER_WG = 16;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;       // 256
    const uint32_t blocks_per_col = K / 32;
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "gemv_q5_1_grp", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<sycl::half, 1> A_slm(sycl::range<1>(K), h);
        h.parallel_for(sycl::nd_range<1>(uint64_t(P) * n_wgs * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t gid   = uint32_t(it.get_group(0));
            const uint32_t p     = gid / n_wgs;
            const uint32_t wgid  = gid % n_wgs;
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t n     = wgid * N_PER_WG + sg_id;

            const uint32_t slot  = uint32_t(jobs[p * 3 + 1]);
            const uint32_t ycell = uint32_t(jobs[p * 3 + 2]);
            const sycl::half* A = h_rows + uint64_t(p) * K;
            const auto* W = reinterpret_cast<const block_q5_1*>(
                slot_base + uint64_t(slot) * slot_bytes + down_off);

            for (uint32_t i = lid; i < K; i += WG_ITEMS) A_slm[i] = A[i];
            sycl::group_barrier(it.get_group());
            if (n >= N) return;

            float acc = 0.f;
            const block_q5_1* col = &W[uint64_t(n) * blocks_per_col];
            for (uint32_t b = 0; b < blocks_per_col; ++b) {
                const block_q5_1& blk = col[b];
                // Native half->float (exact conversion, bit-identical to the
                // old software path) — the branchy sw convert ran 2x per
                // block per lane and was the top ALU cost of this
                // dequant-bound kernel. qh via two u16 loads (alignas(2)
                // struct — a u32 load is not guaranteed aligned); the
                // assembled value is identical.
                const float d = float(sycl::bit_cast<sycl::half>(blk.d));
                const float m = float(sycl::bit_cast<sycl::half>(blk.m));
                const uint16_t* qh16 = reinterpret_cast<const uint16_t*>(blk.qh);
                const uint32_t qh = uint32_t(qh16[0]) | (uint32_t(qh16[1]) << 16);
                const uint8_t qb  = blk.qs[lane];
                const uint32_t xh0 = ((qh >> lane)        << 4) & 0x10u;
                const uint32_t xh1 = ((qh >> (lane + 12))     ) & 0x10u;
                const float w_lo = float((qb & 0x0Fu) | xh0) * d + m;
                const float w_hi = float((qb >>  4u ) | xh1) * d + m;
                acc += float(A_slm[b * 32 + lane])      * w_lo +
                       float(A_slm[b * 32 + lane + 16]) * w_hi;
            }
            acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
            if (lane == 0) ystage[uint64_t(ycell) * y_stride + n] = sycl::half(acc);
        });
    });
}

// GROUPED-TILES down-proj (prefill expert-major MoE, 2026-08-27): one launch
// per layer covers every (expert, <=16-row chunk, column-tile). Per row the
// gemv_q5_1 expression runs in the same order -> bit-identical to the
// per-expert gemm_q5_1 path (whose per-row numerics equal gemv_q5_1's).
// Rows read from GLOBAL (the gathered mega h buffer; same values solo stages
// through SLM). tile_jobs = [J x 4] int32 {x_row0, rows, slot, y_row0}.
sycl::event gemv_q5_1_grouped_tiles(sycl::queue& q,
                      const sycl::half* h_mega,
                      const uint8_t* slot_base, uint64_t slot_bytes,
                      uint64_t down_off,
                      const int32_t* tile_jobs,
                      sycl::half* y_mega,
                      uint32_t K, uint32_t N, uint32_t J,
                      const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE  = 16;
    constexpr int N_PER_WG = 16;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;       // 256
    constexpr int T_MAX    = 16;
    const uint32_t blocks_per_col = K / 32;
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "gemv_q5_1_gtile", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(uint64_t(J) * n_wgs * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t gid   = uint32_t(it.get_group(0));
            const uint32_t j     = gid / n_wgs;
            const uint32_t wgid  = gid % n_wgs;
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t n     = wgid * N_PER_WG + sg_id;
            if (n >= N) return;

            const uint32_t x0   = uint32_t(tile_jobs[j * 4 + 0]);
            const uint32_t T    = uint32_t(tile_jobs[j * 4 + 1]);
            const uint32_t slot = uint32_t(tile_jobs[j * 4 + 2]);
            const uint32_t y0   = uint32_t(tile_jobs[j * 4 + 3]);
            const sycl::half* A0 = h_mega + uint64_t(x0) * K;
            const auto* W = reinterpret_cast<const block_q5_1*>(
                slot_base + uint64_t(slot) * slot_bytes + down_off);

            float acc[T_MAX];
            #pragma unroll
            for (int t = 0; t < T_MAX; ++t) acc[t] = 0.f;

            const block_q5_1* col = &W[uint64_t(n) * blocks_per_col];
            for (uint32_t b = 0; b < blocks_per_col; ++b) {
                const block_q5_1& blk = col[b];
                // Native half->float (exact conversion, bit-identical to the
                // old software path) — the branchy sw convert ran 2x per
                // block per lane and was the top ALU cost of this
                // dequant-bound kernel. qh via two u16 loads (alignas(2)
                // struct — a u32 load is not guaranteed aligned); the
                // assembled value is identical.
                const float d = float(sycl::bit_cast<sycl::half>(blk.d));
                const float m = float(sycl::bit_cast<sycl::half>(blk.m));
                const uint16_t* qh16 = reinterpret_cast<const uint16_t*>(blk.qh);
                const uint32_t qh = uint32_t(qh16[0]) | (uint32_t(qh16[1]) << 16);
                const uint8_t qb  = blk.qs[lane];
                const uint32_t xh0 = ((qh >> lane)        << 4) & 0x10u;
                const uint32_t xh1 = ((qh >> (lane + 12))     ) & 0x10u;
                const float w_lo = float((qb & 0x0Fu) | xh0) * d + m;
                const float w_hi = float((qb >>  4u ) | xh1) * d + m;
                #pragma unroll
                for (int t = 0; t < T_MAX; ++t) {
                    if (uint32_t(t) >= T) break;
                    const sycl::half* A = A0 + uint64_t(t) * K;
                    acc[t] += float(A[b * 32 + lane])      * w_lo +
                              float(A[b * 32 + lane + 16]) * w_hi;
                }
            }
            #pragma unroll
            for (int t = 0; t < T_MAX; ++t) {
                if (uint32_t(t) >= T) break;
                const float r = sycl::reduce_over_group(it.get_sub_group(), acc[t],
                                                        sycl::plus<float>());
                if (lane == 0) y_mega[uint64_t(y0 + uint32_t(t)) * N + n] = sycl::half(r);
            }
        });
    });
}

// Multi-row GEMM: y[M,N] = A[M,K] @ W[K,N]. Same Q5_1 layout. M rows per WG via
// per-lane M accumulators; weight block dequantized once per (block, column),
// reused across rows. Any M (tiled by M_TILE over grid dim 0).
sycl::event gemm_q5_1(sycl::queue& q,
                      const sycl::half* A, const void* W_packed,
                      sycl::half* y,
                      uint32_t M, uint32_t K, uint32_t N,
                      const std::vector<sycl::event>& deps) {
    if (M == 0) return {};
    constexpr int SG_SIZE  = 16;
    constexpr int N_PER_WG = 16;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;       // 256
    constexpr int M_TILE   = 16;
    const auto* W = static_cast<const block_q5_1*>(W_packed);
    const uint32_t blocks_per_col = K / 32;
    const uint32_t n_wgs   = (N + N_PER_WG - 1) / N_PER_WG;
    const uint32_t m_tiles = (M + M_TILE - 1) / M_TILE;
    const uint32_t slm_rows = std::min<uint32_t>(M, M_TILE);

    return ie::ps(q, "gemm_q5_1", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<sycl::half, 1> A_slm(uint64_t(slm_rows) * K, h);
        h.parallel_for(sycl::nd_range<2>({m_tiles, uint64_t(n_wgs) * WG_ITEMS},
                                         {1, WG_ITEMS}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(1));
            const uint32_t wgid  = uint32_t(it.get_group(1));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t n     = wgid * N_PER_WG + sg_id;
            const uint32_t m0    = uint32_t(it.get_group(0)) * M_TILE;
            const uint32_t Mc    = sycl::min(uint32_t(M_TILE), M - m0);

            const uint64_t a_total = uint64_t(Mc) * K;
            for (uint64_t i = lid; i < a_total; i += WG_ITEMS)
                A_slm[i] = A[uint64_t(m0) * K + i];
            sycl::group_barrier(it.get_group());
            if (n >= N) return;

            float acc[M_TILE];
            #pragma unroll
            for (int mm = 0; mm < M_TILE; ++mm) acc[mm] = 0.f;

            const block_q5_1* col = &W[uint64_t(n) * blocks_per_col];
            for (uint32_t b = 0; b < blocks_per_col; ++b) {
                const block_q5_1& blk = col[b];
                // Native half->float (exact conversion, bit-identical to the
                // old software path) — the branchy sw convert ran 2x per
                // block per lane and was the top ALU cost of this
                // dequant-bound kernel. qh via two u16 loads (alignas(2)
                // struct — a u32 load is not guaranteed aligned); the
                // assembled value is identical.
                const float d = float(sycl::bit_cast<sycl::half>(blk.d));
                const float m = float(sycl::bit_cast<sycl::half>(blk.m));
                const uint16_t* qh16 = reinterpret_cast<const uint16_t*>(blk.qh);
                const uint32_t qh = uint32_t(qh16[0]) | (uint32_t(qh16[1]) << 16);
                const uint8_t qb  = blk.qs[lane];
                const uint32_t xh0 = ((qh >> lane)        << 4) & 0x10u;
                const uint32_t xh1 = ((qh >> (lane + 12))     ) & 0x10u;
                const float w_lo = float((qb & 0x0Fu) | xh0) * d + m;
                const float w_hi = float((qb >>  4u ) | xh1) * d + m;
                #pragma unroll
                for (int mm = 0; mm < M_TILE; ++mm) {
                    if (uint32_t(mm) < Mc) {
                        const sycl::half* a = &A_slm[uint64_t(mm) * K + b * 32];
                        acc[mm] += float(a[lane]) * w_lo + float(a[lane + 16]) * w_hi;
                    }
                }
            }
            #pragma unroll
            for (int mm = 0; mm < M_TILE; ++mm) {
                if (uint32_t(mm) < Mc) {
                    const float r = sycl::reduce_over_group(it.get_sub_group(), acc[mm],
                                                            sycl::plus<float>());
                    if (lane == 0) y[uint64_t(m0 + mm) * N + n] = sycl::half(r);
                }
            }
        });
    });
}

// ---------------------------------------------------------------------------
// Host-side load-time conversions.
// ---------------------------------------------------------------------------

// bf16 → f16: promote (top 16 bits of fp32) then RTNE via the engine's
// canonical fp32_to_fp16. NaN in → NaN out (quietened). Bit stores through
// memcpy so host sycl::half conversion semantics never enter the path.
void convert_bf16_to_f16(const uint16_t* src, sycl::half* dst, size_t n) {
    static_assert(sizeof(sycl::half) == 2, "sycl::half must be 2 bytes");
    for (size_t i = 0; i < n; ++i) {
        const float f = std::bit_cast<float>(uint32_t(src[i]) << 16);
        const uint16_t h = fp32_to_fp16(f);
        __builtin_memcpy(&dst[i], &h, sizeof(h));
    }
}

// Q5_K → f16: ref::dequant_q5_K per 256-elem super-block, then RTNE to f16.
void convert_q5_k_to_f16_buffer(const void* packed, size_t n, sycl::half* out) {
    const auto* b = static_cast<const block_q5_K*>(packed);
    float tmp[kQK_K];
    for (size_t i = 0; i < n / kQK_K; ++i) {
        ref::dequant_q5_K(&b[i], tmp);
        for (int j = 0; j < kQK_K; ++j) {
            const uint16_t h = fp32_to_fp16(tmp[j]);
            __builtin_memcpy(&out[i * kQK_K + j], &h, sizeof(h));
        }
    }
}

}  // namespace ie
