// src/ops/gemv_q8_0.cpp — W8A16/W8A8 GEMV/GEMM with Q8_0 weights (qwen4exp).
//
// Q8_0 is the quant format of the Qwen3.8-Flash-Next routed-expert banks in
// the community GGUFs (Unsloth Q8_0, UD-Q6_K_XL): per 32 elements, 34 bytes —
// fp16 scale d + 32 int8. w[k] = d * qs[k] (symmetric — no min term). The
// banks stream RAW from the host mmap through the per-card expert cache (the
// load-time F16 fallback would be 1.88x the file in host RAM); these kernels
// dequant on the fly.
//
// Two activation families mirror the Q4_K/Q5_1 raw paths:
//   - block_q8_1s activations (gate/up int-dot, quantize_q8_1s): per lane the
//     int8 products of its 2 qs bytes; acc += dW * dX * idot in fp32; the
//     Q4_K min/sub-scale correction terms vanish (Q8_0 is symmetric);
//   - f16 activations (down): lane L covers elements L and L+16 of each
//     32-block, exactly gemv_q5_1's lattice.
//
// Weight layout: W[K, N] column-major-packed — each output column n is K/32
// consecutive block_q8_0. One subgroup per output column; fp32 accumulate;
// subgroup-reduce; lane 0 writes. Variants mirror gemv_q5_1.cpp /
// gemv_q8dot.cpp's scaffolding verbatim (solo / grouped / grouped_tiles /
// dual / batched / gemm) so the dispatch sites stay parallel.

#include "ie/qwen4_quant.hpp"
#include "ie/quant_blocks.hpp"

#include <sycl/sycl.hpp>
#include "ie/kernel_profiler.hpp"
#include "ie/dp4a.hpp"

namespace ie {

// ---------------------------------------------------------------------------
// x = f16 activations (the down bank).
// ---------------------------------------------------------------------------
sycl::event gemv_q8_0(sycl::queue& q,
                      const sycl::half* A, const void* W_packed,
                      sycl::half* y,
                      uint32_t K, uint32_t N,
                      const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE  = 16;
    constexpr int N_PER_WG = 16;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;       // 256
    const auto* W = static_cast<const block_q8_0*>(W_packed);
    const uint32_t blocks_per_col = K / 32;
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "gemv_q8_0", [&](sycl::handler& h) {
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
            const block_q8_0* col = &W[uint64_t(n) * blocks_per_col];
            for (uint32_t b = 0; b < blocks_per_col; ++b) {
                const block_q8_0& blk = col[b];
                const float d = float(sycl::bit_cast<sycl::half>(blk.d));
                acc += float(A_slm[b * 32 + lane])      * d * float(blk.qs[lane]) +
                       float(A_slm[b * 32 + lane + 16]) * d * float(blk.qs[lane + 16]);
            }
            acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
            if (lane == 0) y[n] = sycl::half(acc);
        });
    });
}

// Grouped expert-pick variant (spec-verify MoE): jobs = [P x 3] int32
// {vt, slot, ycell}; per (pick, column) the solo body runs verbatim.
sycl::event gemv_q8_0_grouped(sycl::queue& q,
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

    return ie::ps(q, "gemv_q8_0_grp", [&](sycl::handler& h) {
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
            const auto* W = reinterpret_cast<const block_q8_0*>(
                slot_base + uint64_t(slot) * slot_bytes + down_off);

            for (uint32_t i = lid; i < K; i += WG_ITEMS) A_slm[i] = A[i];
            sycl::group_barrier(it.get_group());
            if (n >= N) return;

            float acc = 0.f;
            const block_q8_0* col = &W[uint64_t(n) * blocks_per_col];
            for (uint32_t b = 0; b < blocks_per_col; ++b) {
                const block_q8_0& blk = col[b];
                const float d = float(sycl::bit_cast<sycl::half>(blk.d));
                acc += float(A_slm[b * 32 + lane])      * d * float(blk.qs[lane]) +
                       float(A_slm[b * 32 + lane + 16]) * d * float(blk.qs[lane + 16]);
            }
            acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
            if (lane == 0) ystage[uint64_t(ycell) * y_stride + n] = sycl::half(acc);
        });
    });
}

// Grouped-tiles down-proj (prefill expert-major MoE): tile_jobs = [J x 4]
// int32 {x_row0, rows, slot, y_row0}; per row the gemv_q8_0 body verbatim.
sycl::event gemv_q8_0_grouped_tiles(sycl::queue& q,
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

    return ie::ps(q, "gemv_q8_0_gtile", [&](sycl::handler& h) {
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
            const auto* W = reinterpret_cast<const block_q8_0*>(
                slot_base + uint64_t(slot) * slot_bytes + down_off);

            float acc[T_MAX];
            #pragma unroll
            for (int t = 0; t < T_MAX; ++t) acc[t] = 0.f;

            const block_q8_0* col = &W[uint64_t(n) * blocks_per_col];
            for (uint32_t b = 0; b < blocks_per_col; ++b) {
                const block_q8_0& blk = col[b];
                const float d = float(sycl::bit_cast<sycl::half>(blk.d));
                const float w_lo = d * float(blk.qs[lane]);
                const float w_hi = d * float(blk.qs[lane + 16]);
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

// Multi-row GEMM: y[M,N] = A[M,K] @ W[K,N] (chunked prefill down).
sycl::event gemm_q8_0(sycl::queue& q,
                      const sycl::half* A, const void* W_packed,
                      sycl::half* y,
                      uint32_t M, uint32_t K, uint32_t N,
                      const std::vector<sycl::event>& deps) {
    if (M == 0) return {};
    constexpr int SG_SIZE  = 16;
    constexpr int N_PER_WG = 16;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;       // 256
    constexpr int M_TILE   = 16;
    const auto* W = static_cast<const block_q8_0*>(W_packed);
    const uint32_t blocks_per_col = K / 32;
    const uint32_t n_wgs   = (N + N_PER_WG - 1) / N_PER_WG;
    const uint32_t m_tiles = (M + M_TILE - 1) / M_TILE;
    const uint32_t slm_rows = std::min<uint32_t>(M, M_TILE);

    return ie::ps(q, "gemm_q8_0", [&](sycl::handler& h) {
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

            const block_q8_0* col = &W[uint64_t(n) * blocks_per_col];
            for (uint32_t b = 0; b < blocks_per_col; ++b) {
                const block_q8_0& blk = col[b];
                const float d = float(sycl::bit_cast<sycl::half>(blk.d));
                const float w_lo = d * float(blk.qs[lane]);
                const float w_hi = d * float(blk.qs[lane + 16]);
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
// x = block_q8_1s activations (the gate/up banks, quantize_q8_1s): int-dot.
// Lane L covers qs bytes 2L and 2L+1 of each 32-block; acc += dW * dX * idot
// per block per lane in fp32; subgroup-reduce. Q8_0 is symmetric, so the
// Q4_K min/sub-scale correction terms vanish entirely.
// ---------------------------------------------------------------------------
sycl::event gemv_q8_0_q8s(sycl::queue& q,
                         const void* x_q8s, const void* W_packed,
                         sycl::half* y,
                         uint32_t K, uint32_t N,
                         const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE  = 16;
    constexpr int N_PER_WG = 16;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;       // 256

    const auto* W  = static_cast<const block_q8_0*>(W_packed);
    const auto* X8 = static_cast<const block_q8_1s*>(x_q8s);
    const uint32_t blocks_per_col = K / 32;
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "gemv_q8_0_q8s", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t wgid  = uint32_t(it.get_group(0));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t n     = wgid * N_PER_WG + sg_id;
            if (n >= N) return;

            float acc = 0.f;
            const block_q8_0* col = &W[uint64_t(n) * blocks_per_col];
            for (uint32_t b = 0; b < blocks_per_col; ++b) {
                const block_q8_0& wblk = col[b];
                const block_q8_1s& xb = X8[b];
                const float dW = float(sycl::bit_cast<sycl::half>(wblk.d));
                const int j0 = int(lane) * 2;
                const int32_t dot = int32_t(xb.qs[j0])     * int32_t(wblk.qs[j0]) +
                                    int32_t(xb.qs[j0 + 1]) * int32_t(wblk.qs[j0 + 1]);
                acc += dW * xb.d * float(dot);
            }
            acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
            if (lane == 0) y[n] = sycl::half(acc);
        });
    });
}

// Grouped dual gate/up over an expert-pick job list (spec-verify MoE):
// jobs = [P x 3] int32 {vt, slot, ycell}; x row vt of the q8_1s mega stream.
sycl::event gemv_q8_0_q8s_grouped_dual(sycl::queue& q,
                         const void* x_q8s,
                         const uint8_t* slot_base, uint64_t slot_bytes,
                         uint64_t gate_off, uint64_t up_off,
                         const int32_t* jobs,
                         sycl::half* g_out, sycl::half* u_out,
                         uint32_t K, uint32_t N, uint32_t P,
                         const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE  = 16;
    constexpr int N_PER_WG = 16;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;       // 256

    const auto* X8 = static_cast<const block_q8_1s*>(x_q8s);
    const uint32_t blocks_per_col = K / 32;
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "gemv_q8_0_q8s_grp", [&](sycl::handler& h) {
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
            const block_q8_1s* xrow = X8 + uint64_t(vt) * blocks_per_col;
            const auto* W = reinterpret_cast<const block_q8_0*>(
                slot_base + uint64_t(slot) * slot_bytes +
                (is_up ? up_off : gate_off));
            sycl::half* y = (is_up ? u_out : g_out) + uint64_t(pk) * N;

            float acc = 0.f;
            const block_q8_0* col = &W[uint64_t(n) * blocks_per_col];
            for (uint32_t b = 0; b < blocks_per_col; ++b) {
                const block_q8_0& wblk = col[b];
                const block_q8_1s& xb = xrow[b];
                const float dW = float(sycl::bit_cast<sycl::half>(wblk.d));
                const int j0 = int(lane) * 2;
                const int32_t dot = int32_t(xb.qs[j0])     * int32_t(wblk.qs[j0]) +
                                    int32_t(xb.qs[j0 + 1]) * int32_t(wblk.qs[j0 + 1]);
                acc += dW * xb.d * float(dot);
            }
            acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
            if (lane == 0) y[n] = sycl::half(acc);
        });
    });
}

// Grouped-tiles dual gate/up (prefill expert-major MoE + decode small_moe):
// tile_jobs = [J x 4] int32 {x_row0, rows, slot, y_row0}; T <= 16 rows per
// tile; x rows are consecutive q8_1s streams in the mega buffer.
sycl::event gemv_q8_0_q8s_grouped_tiles(sycl::queue& q,
                         const void* x_q8s_mega,
                         const uint8_t* slot_base, uint64_t slot_bytes,
                         uint64_t gate_off, uint64_t up_off,
                         const int32_t* tile_jobs,
                         sycl::half* g_out, sycl::half* u_out,
                         uint32_t K, uint32_t N, uint32_t J,
                         const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE  = 16;
    constexpr int N_PER_WG = 16;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;       // 256
    constexpr int T_MAX     = 16;

    const auto* X8 = static_cast<const block_q8_1s*>(x_q8s_mega);
    const uint32_t blocks_per_col = K / 32;
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "gemv_q8_0_q8s_gtile", [&](sycl::handler& h) {
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
            const block_q8_1s* Xj = X8 + uint64_t(x0) * blocks_per_col;
            const auto* W = reinterpret_cast<const block_q8_0*>(
                slot_base + uint64_t(slot) * slot_bytes +
                (is_up ? up_off : gate_off));
            sycl::half* y = (is_up ? u_out : g_out) + uint64_t(y0) * N;

            float acc[T_MAX];
            #pragma unroll
            for (int t = 0; t < T_MAX; ++t) acc[t] = 0.f;

            const block_q8_0* col = &W[uint64_t(n) * blocks_per_col];
            for (uint32_t b = 0; b < blocks_per_col; ++b) {
                const block_q8_0& wblk = col[b];
                const float dW = float(sycl::bit_cast<sycl::half>(wblk.d));
                const int j0 = int(lane) * 2;
                #pragma unroll
                for (int t = 0; t < T_MAX; ++t) {
                    if (uint32_t(t) >= T) break;
                    const block_q8_1s& xb = Xj[uint64_t(t) * blocks_per_col + b];
                    const int32_t dot = int32_t(xb.qs[j0])     * int32_t(wblk.qs[j0]) +
                                        int32_t(xb.qs[j0 + 1]) * int32_t(wblk.qs[j0 + 1]);
                    acc[t] += dW * xb.d * float(dot);
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

// Batched-T gate/up over STAGED q8_1s activation rows (chunked prefill):
// x_q8s_mega = T-row streams; per-row numerics identical to gemv_q8_0_q8s.

// dp4a variant of the tiles kernel (2026-10-27): the naive body above gives
// each lane 2 qs bytes of a 32-byte block, so every lane loads the whole
// 34-byte block and does 2 scalar imuls per token. Here a subgroup pass
// covers TWO Q8_0 blocks: lanes 0-7 own bytes [4l, 4l+4) of block p, lanes
// 8-15 the same window of block p+1 — each weight byte loaded exactly once,
// one dp4a_ss per lane per block, half the int ALU and dW loads. Per-lane
// partial sums regroup (4 products vs 2), so fp rounding differs slightly
// from the naive kernel — same gate, same tolerance class.
sycl::event gemv_q8_0_q8s_grouped_tiles_dp4a(sycl::queue& q,
                         const void* x_q8s_mega,
                         const uint8_t* slot_base, uint64_t slot_bytes,
                         uint64_t gate_off, uint64_t up_off,
                         const int32_t* tile_jobs,
                         sycl::half* g_out, sycl::half* u_out,
                         uint32_t K, uint32_t N, uint32_t J,
                         const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE  = 16;
    constexpr int N_PER_WG = 16;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;       // 256
    constexpr int T_MAX     = 16;

    const auto* X8 = static_cast<const block_q8_1s*>(x_q8s_mega);
    const uint32_t blocks_per_col = K / 32;
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "gemv_q8_0_q8s_gtile_dp4a", [&](sycl::handler& h) {
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
            const block_q8_1s* Xj = X8 + uint64_t(x0) * blocks_per_col;
            const auto* W = reinterpret_cast<const block_q8_0*>(
                slot_base + uint64_t(slot) * slot_bytes +
                (is_up ? up_off : gate_off));
            sycl::half* y = (is_up ? u_out : g_out) + uint64_t(y0) * N;

            const block_q8_0* col = &W[uint64_t(n) * blocks_per_col];
            if (T < 8) {
                // small tiles: the naive 2-byte lane mapping wins here
                // (2.3x at T=1, ie-q8-0-kernel-bench) — same body verbatim.
                float accS[T_MAX];
                #pragma unroll
                for (int t = 0; t < T_MAX; ++t) accS[t] = 0.f;
                const int j0 = int(lane) * 2;
                for (uint32_t b = 0; b < blocks_per_col; ++b) {
                    const block_q8_0& wblk = col[b];
                    const float dW = float(sycl::bit_cast<sycl::half>(wblk.d));
                    #pragma unroll
                    for (int t = 0; t < T_MAX; ++t) {
                        if (uint32_t(t) >= T) break;
                        const block_q8_1s& xb = Xj[uint64_t(t) * blocks_per_col + b];
                        const int32_t dot = int32_t(xb.qs[j0])     * int32_t(wblk.qs[j0]) +
                                            int32_t(xb.qs[j0 + 1]) * int32_t(wblk.qs[j0 + 1]);
                        accS[t] += dW * xb.d * float(dot);
                    }
                }
                #pragma unroll
                for (int t = 0; t < T_MAX; ++t) {
                    if (uint32_t(t) >= T) break;
                    const float r = sycl::reduce_over_group(it.get_sub_group(), accS[t],
                                                            sycl::plus<float>());
                    if (lane == 0) y[uint64_t(t) * N + n] = sycl::half(r);
                }
                return;
            }
            const uint32_t s  = lane & 7u;          // byte window within a block
            const uint32_t hb = lane >= 8u ? 1u : 0u; // which block of the pair
            float acc[T_MAX];
            #pragma unroll
            for (int t = 0; t < T_MAX; ++t) acc[t] = 0.f;

            uint32_t p = 0;
            for (; p + 1 < blocks_per_col; p += 2) {
                const uint32_t blk = p + hb;
                const block_q8_0& wblk = col[blk];
                const float dW = float(sycl::bit_cast<sycl::half>(wblk.d));
                int32_t wq; __builtin_memcpy(&wq, wblk.qs + 4 * s, 4);
                #pragma unroll
                for (int t = 0; t < T_MAX; ++t) {
                    if (uint32_t(t) >= T) break;
                    const block_q8_1s& xb = Xj[uint64_t(t) * blocks_per_col + blk];
                    int32_t xq; __builtin_memcpy(&xq, xb.qs + 4 * s, 4);
                    acc[t] += dW * xb.d * float(ie::dp4a_ss(wq, xq, 0));
                }
            }
            if (p < blocks_per_col && hb == 0) {          // odd tail, block pair incomplete
                const block_q8_0& wblk = col[p];
                const float dW = float(sycl::bit_cast<sycl::half>(wblk.d));
                int32_t wq; __builtin_memcpy(&wq, wblk.qs + 4 * s, 4);
                #pragma unroll
                for (int t = 0; t < T_MAX; ++t) {
                    if (uint32_t(t) >= T) break;
                    const block_q8_1s& xb = Xj[uint64_t(t) * blocks_per_col + p];
                    int32_t xq; __builtin_memcpy(&xq, xb.qs + 4 * s, 4);
                    acc[t] += dW * xb.d * float(ie::dp4a_ss(wq, xq, 0));
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

// Batched-T gate/up over STAGED q8_1x activation rows (chunked prefill):
// x_q8x = T contiguous block_q8_1x streams (row t at block offset t*(K/32));
// per-row numerics identical to gemv_q8_0_q8s. T <= 16.
sycl::event gemv_q8_0_q8x_batched(sycl::queue& q,
                         const void* x_q8x, const void* W_packed,
                         sycl::half* y,
                         uint32_t K, uint32_t N, uint32_t T,
                         const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE  = 16;
    constexpr int N_PER_WG = 16;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;        // 512
    constexpr int T_MAX     = 16;

    const auto* W  = static_cast<const block_q8_0*>(W_packed);
    const auto* X8 = static_cast<const block_q8_1x*>(x_q8x);
    const uint32_t blocks_per_col = K / 32;
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "gemv_q8_0_q8x_T", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t wgid  = uint32_t(it.get_group(0));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t n     = wgid * N_PER_WG + sg_id;
            if (n >= N) return;

            float acc[T_MAX];
            #pragma unroll
            for (int t = 0; t < T_MAX; ++t) acc[t] = 0.f;

            const block_q8_0* col = &W[uint64_t(n) * blocks_per_col];
            for (uint32_t b = 0; b < blocks_per_col; ++b) {
                const block_q8_0& wblk = col[b];
                const float dW = float(sycl::bit_cast<sycl::half>(wblk.d));
                const int j0 = int(lane) * 2;
                #pragma unroll
                for (int t = 0; t < T_MAX; ++t) {
                    if (uint32_t(t) >= T) break;
                    const block_q8_1x& xb = X8[uint64_t(t) * blocks_per_col + b];
                    const int32_t dot = int32_t(xb.qs[j0])     * int32_t(wblk.qs[j0]) +
                                        int32_t(xb.qs[j0 + 1]) * int32_t(wblk.qs[j0 + 1]);
                    acc[t] += dW * xb.d * float(dot);
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

// dp4a variant of gemv_q8_0_q8x_batched (0004): same lane remap — lanes 0-7
// own bytes [4s, 4s+4) of block p, lanes 8-15 of block p+1; one dp4a_ss per
// lane per block instead of two scalar imuls. 1.1-1.2x at the down shape
// (K=1536 -> N=2048) for T >= 8; slightly different fp rounding.
sycl::event gemv_q8_0_q8x_batched_dp4a(sycl::queue& q,
                         const void* x_q8x, const void* W_packed,
                         sycl::half* y,
                         uint32_t K, uint32_t N, uint32_t T,
                         const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE  = 16;
    constexpr int N_PER_WG = 16;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;
    constexpr int T_MAX     = 16;

    const auto* W  = static_cast<const block_q8_0*>(W_packed);
    const auto* X8 = static_cast<const block_q8_1x*>(x_q8x);
    const uint32_t blocks_per_col = K / 32;
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "gemv_q8_0_q8x_T_dp4a", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t wgid  = uint32_t(it.get_group(0));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t n     = wgid * N_PER_WG + sg_id;
            if (n >= N) return;

            const uint32_t s  = lane & 7u;
            const uint32_t hb = lane >= 8u ? 1u : 0u;
            float acc[T_MAX];
            #pragma unroll
            for (int t = 0; t < T_MAX; ++t) acc[t] = 0.f;

            const block_q8_0* col = &W[uint64_t(n) * blocks_per_col];
            uint32_t p = 0;
            for (; p + 1 < blocks_per_col; p += 2) {
                const uint32_t blk = p + hb;
                const block_q8_0& wblk = col[blk];
                const float dW = float(sycl::bit_cast<sycl::half>(wblk.d));
                int32_t wq; __builtin_memcpy(&wq, wblk.qs + 4 * s, 4);
                #pragma unroll
                for (int t = 0; t < T_MAX; ++t) {
                    if (uint32_t(t) >= T) break;
                    const block_q8_1x& xb = X8[uint64_t(t) * blocks_per_col + blk];
                    int32_t xq; __builtin_memcpy(&xq, xb.qs + 4 * s, 4);
                    acc[t] += dW * float(xb.d) * float(ie::dp4a_ss(wq, xq, 0));
                }
            }
            if (p < blocks_per_col && hb == 0) {
                const block_q8_0& wblk = col[p];
                const float dW = float(sycl::bit_cast<sycl::half>(wblk.d));
                int32_t wq; __builtin_memcpy(&wq, wblk.qs + 4 * s, 4);
                #pragma unroll
                for (int t = 0; t < T_MAX; ++t) {
                    if (uint32_t(t) >= T) break;
                    const block_q8_1x& xb = X8[uint64_t(t) * blocks_per_col + p];
                    int32_t xq; __builtin_memcpy(&xq, xb.qs + 4 * s, 4);
                    acc[t] += dW * float(xb.d) * float(ie::dp4a_ss(wq, xq, 0));
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

}  // namespace ie
