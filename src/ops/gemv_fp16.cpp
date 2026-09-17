// src/ops/gemv_fp16.cpp — dense fp16-weight GEMV (P2: qwen3 F16 attn_v).
//
// y[1, N] = A[1, K] @ W_kn[K, N]
// W_kn is plain fp16 row-major [K, N] — the load-time TRANSPOSE of GGUF's
// [N, K] storage (W_kn[k*N + n] = W_gguf[n*K + k]).  The transpose is what
// makes the decode read pattern coalesced: at a fixed k, consecutive lanes
// read consecutive n.
//
// WG layout (vs gemv_q4_K, where one SG owns one column and lanes split K —
// that mapping would stride column reads by N here, so it is inverted):
//   * 32 subgroups × 16 lanes = 512 work-items per WG.
//   * Each lane owns 2 consecutive output columns (one 4-byte half2 load per
//     k step → a contiguous 64 B line per SG per step).
//   * The 32 SGs split the K range (unroll 16 keeps many W loads in flight);
//     per-column partials meet in SLM and the first N_PER_WG work-items do
//     the final 32-way sum + fp16 store.
//   * A is loaded once into SLM (K halfs) and reused by all 32 K-slices.
//
// Tuning record (real shape [K=4096, N=1024]; DRAM-stream harness — W
// rotated over 5 buffers so 42 MB working set defeats the 18 MB L2):
//   16 SG/WG no-unroll 49 GB/s → unroll-8 175 → half4 lanes 157 (occupancy
//   loss) → 32 SG/WG half2 unroll-8 194 → unroll-16 ≈235 GB/s best /
//   ≈210 avg (35–38 µs) ✓.  unroll-24/32 regressed; 64 SG/WG (1024-item
//   WGs = a full Xe core's 64 threads) was bimodal (24 µs best, ~1 ms avg)
//   — rejected.
//
// Re-measured solo 2026-06-10 after concurrent-load taint (prior run had a
// 20 GB model pinned on the same 32 GB GPU, inflating latency figures):
//   Idle runs (3×): best 37.6/37.8/39.4 µs, avg ~40.1–40.2 µs
//                   → 223/222/213 GB/s best, 209/209/209 GB/s avg
//   Sustained (post ie-perplexity clock-warm, 1×): best 37.8 µs, avg 40.2 µs
//                   → 222 GB/s best, 209 GB/s avg
//   Summary: idle best 213–223 GB/s, sustained 222 GB/s; avg consistently
//   209 GB/s.  All runs clear the >200 GB/s bandwidth-bound bar — Task 2
//   acceptance stands.  Original "510/464 GB/s" headline was a taint artifact
//   (those figures were for a different kernel shape or metric); the 35–38 µs
//   latency range was real and is confirmed here.
//
// Decode use runs this 36×/token on [K=4096, N=1024] (8.4 MB) — purely
// bandwidth-bound; unit test reports measured GB/s (target > 200 GB/s).
// No K/N alignment constraints (odd N falls back to bounds-checked scalar
// loads).  SLM budget is K halfs + 4 KiB partials → K ≤ ~28K on BMG.

#include "ie/ops.hpp"

#include <sycl/sycl.hpp>
#include "ie/kernel_profiler.hpp"

namespace ie {

sycl::event gemv_fp16(sycl::queue& q,
                      const sycl::half* A, const sycl::half* W_kn,
                      sycl::half* y, uint32_t K, uint32_t N,
                      const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE       = 16;
    constexpr int SGS_PER_WG    = 32;
    constexpr int WG_ITEMS      = SGS_PER_WG * SG_SIZE;       // 512
    constexpr int COLS_PER_LANE = 2;
    constexpr int N_PER_WG      = SG_SIZE * COLS_PER_LANE;    // 32

    const uint32_t n_wgs   = (N + N_PER_WG - 1) / N_PER_WG;
    const uint32_t k_slice = (K + SGS_PER_WG - 1) / SGS_PER_WG;

    return ie::ps(q, "gemv_fp16", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<sycl::half, 1> A_slm(sycl::range<1>(K), h);
        // Per-SG partials: [sg][col_in_wg].
        sycl::local_accessor<float, 1> part(sycl::range<1>(SGS_PER_WG * N_PER_WG), h);

        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t wgid  = uint32_t(it.get_group(0));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;

            // Cooperative A load — once per WG, reused by all 32 K-slices.
            for (uint32_t i = lid; i < K; i += WG_ITEMS) A_slm[i] = A[i];
            sycl::group_barrier(it.get_group());

            // This lane's column pair and this SG's K-slice.
            const uint32_t n0 = wgid * N_PER_WG + lane * COLS_PER_LANE;
            const uint32_t k0 = sg_id * k_slice;
            const uint32_t k1 = sycl::min(k0 + k_slice, K);

            float acc[COLS_PER_LANE] = {0.f, 0.f};
            if ((N & 1u) == 0 && n0 + COLS_PER_LANE - 1 < N) {
                // Fast path: N even → k*N + n0 even → 4 B-aligned half2
                // loads; consecutive lanes cover one contiguous 64 B line per
                // k step.  Unroll → multiple W loads in flight per thread
                // (the loop is DRAM-latency-bound, not FLOP-bound).
                #pragma unroll 16
                for (uint32_t k = k0; k < k1; ++k) {
                    const float a = float(A_slm[k]);
                    const auto wv = *reinterpret_cast<const sycl::vec<sycl::half, 2>*>(
                        &W_kn[uint64_t(k) * N + n0]);
                    #pragma unroll
                    for (int c = 0; c < COLS_PER_LANE; ++c)
                        acc[c] += a * float(wv[c]);
                }
            } else {
                // Tail path: odd N or the last column pair — bounds-checked
                // scalars.
                for (uint32_t k = k0; k < k1; ++k) {
                    const float a = float(A_slm[k]);
                    const uint64_t row = uint64_t(k) * N;
                    #pragma unroll
                    for (int c = 0; c < COLS_PER_LANE; ++c)
                        if (n0 + c < N) acc[c] += a * float(W_kn[row + n0 + c]);
                }
            }
            #pragma unroll
            for (int c = 0; c < COLS_PER_LANE; ++c)
                part[sg_id * N_PER_WG + lane * COLS_PER_LANE + c] = acc[c];
            sycl::group_barrier(it.get_group());

            // Cross-SG reduce: first N_PER_WG items own one column each.
            if (lid < N_PER_WG) {
                const uint32_t n = wgid * N_PER_WG + lid;
                if (n < N) {
                    float s = 0.f;
                    #pragma unroll
                    for (int sg = 0; sg < SGS_PER_WG; ++sg)
                        s += part[uint32_t(sg) * N_PER_WG + lid];
                    y[n] = sycl::half(s);
                }
            }
        });
    });
}

// Leading-dimension variant: y[1,N] over the FIRST N columns of a [K, ld]
// row-major weight (draft lm_head vocab cap: BPE ids are ~frequency-ordered,
// so drafting over a prefix vocab only lowers accept rate — verify corrects,
// losslessness untouched). Math per column identical to gemv_fp16.
sycl::event gemv_fp16_ld(sycl::queue& q,
                         const sycl::half* A, const sycl::half* W_kn,
                         uint32_t ld, sycl::half* y, uint32_t K, uint32_t N,
                         const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE       = 16;
    constexpr int SGS_PER_WG    = 32;
    constexpr int WG_ITEMS      = SGS_PER_WG * SG_SIZE;
    constexpr int COLS_PER_LANE = 2;
    constexpr int N_PER_WG      = SG_SIZE * COLS_PER_LANE;

    const uint32_t n_wgs   = (N + N_PER_WG - 1) / N_PER_WG;
    const uint32_t k_slice = (K + SGS_PER_WG - 1) / SGS_PER_WG;

    return ie::ps(q, "gemv_fp16_ld", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<sycl::half, 1> A_slm(sycl::range<1>(K), h);
        sycl::local_accessor<float, 1> part(sycl::range<1>(SGS_PER_WG * N_PER_WG), h);
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t wgid  = uint32_t(it.get_group(0));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            for (uint32_t i = lid; i < K; i += WG_ITEMS) A_slm[i] = A[i];
            sycl::group_barrier(it.get_group());
            const uint32_t n0 = wgid * N_PER_WG + lane * COLS_PER_LANE;
            const uint32_t k0 = sg_id * k_slice;
            const uint32_t k1 = sycl::min(k0 + k_slice, K);
            float acc[COLS_PER_LANE] = {0.f, 0.f};
            for (uint32_t k = k0; k < k1; ++k) {
                const float a = float(A_slm[k]);
                const uint64_t row = uint64_t(k) * ld;
                #pragma unroll
                for (int c = 0; c < COLS_PER_LANE; ++c)
                    if (n0 + c < N) acc[c] += a * float(W_kn[row + n0 + c]);
            }
            #pragma unroll
            for (int c = 0; c < COLS_PER_LANE; ++c)
                part[sg_id * N_PER_WG + lane * COLS_PER_LANE + c] = acc[c];
            sycl::group_barrier(it.get_group());
            if (lid < N_PER_WG) {
                const uint32_t n = wgid * N_PER_WG + lid;
                if (n < N) {
                    float s = 0.f;
                    #pragma unroll
                    for (int sg = 0; sg < SGS_PER_WG; ++sg)
                        s += part[uint32_t(sg) * N_PER_WG + lid];
                    y[n] = sycl::half(s);
                }
            }
        });
    });
}

// Multi-row DUAL-weight variant for the spec-verify alpha/beta leaf
// (2026-08-16): R rows × two same-shape [K,N] F16 mats in ONE launch. Each
// (row, mat, wg) group runs the T=1 kernel body VERBATIM on its row/mat →
// every output is bit-identical to the corresponding per-row gemv_fp16 call
// (same K-slice split, same SLM partial layout, same 32-way cross-SG sum).
// Exists because the verify loop's lossless contract requires the per-row
// T=1 leaf (batched gemm reorders fp adds) — this keeps the math and kills
// the 2·R-launches-per-DN-layer overhead (N=64 → launch-bound).
sycl::event gemv_fp16_rows_dual(sycl::queue& q,
                                const sycl::half* X, uint32_t x_stride,
                                const sycl::half* Wa, sycl::half* Ya,
                                const sycl::half* Wb, sycl::half* Yb,
                                uint32_t y_stride,
                                uint32_t K, uint32_t N, uint32_t R,
                                const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE       = 16;
    constexpr int SGS_PER_WG    = 32;
    constexpr int WG_ITEMS      = SGS_PER_WG * SG_SIZE;       // 512
    constexpr int COLS_PER_LANE = 2;
    constexpr int N_PER_WG      = SG_SIZE * COLS_PER_LANE;    // 32

    const uint32_t n_wgs   = (N + N_PER_WG - 1) / N_PER_WG;
    const uint32_t k_slice = (K + SGS_PER_WG - 1) / SGS_PER_WG;

    return ie::ps(q, "gemv_fp16_rows2x", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<sycl::half, 1> A_slm(sycl::range<1>(K), h);
        sycl::local_accessor<float, 1> part(sycl::range<1>(SGS_PER_WG * N_PER_WG), h);

        h.parallel_for(sycl::nd_range<1>(uint64_t(R) * 2 * n_wgs * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t gid   = uint32_t(it.get_group(0));
            const uint32_t r     = gid / (2 * n_wgs);
            const uint32_t rem   = gid % (2 * n_wgs);
            const bool     isB   = rem >= n_wgs;
            const uint32_t wgid  = isB ? rem - n_wgs : rem;
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;

            const sycl::half* A    = X + uint64_t(r) * x_stride;
            const sycl::half* W_kn = isB ? Wb : Wa;
            sycl::half*       y    = (isB ? Yb : Ya) + uint64_t(r) * y_stride;

            for (uint32_t i = lid; i < K; i += WG_ITEMS) A_slm[i] = A[i];
            sycl::group_barrier(it.get_group());

            const uint32_t n0 = wgid * N_PER_WG + lane * COLS_PER_LANE;
            const uint32_t k0 = sg_id * k_slice;
            const uint32_t k1 = sycl::min(k0 + k_slice, K);

            float acc[COLS_PER_LANE] = {0.f, 0.f};
            if ((N & 1u) == 0 && n0 + COLS_PER_LANE - 1 < N) {
                #pragma unroll 16
                for (uint32_t k = k0; k < k1; ++k) {
                    const float a = float(A_slm[k]);
                    const auto wv = *reinterpret_cast<const sycl::vec<sycl::half, 2>*>(
                        &W_kn[uint64_t(k) * N + n0]);
                    #pragma unroll
                    for (int c = 0; c < COLS_PER_LANE; ++c)
                        acc[c] += a * float(wv[c]);
                }
            } else {
                for (uint32_t k = k0; k < k1; ++k) {
                    const float a = float(A_slm[k]);
                    const uint64_t row = uint64_t(k) * N;
                    #pragma unroll
                    for (int c = 0; c < COLS_PER_LANE; ++c)
                        if (n0 + c < N) acc[c] += a * float(W_kn[row + n0 + c]);
                }
            }
            #pragma unroll
            for (int c = 0; c < COLS_PER_LANE; ++c)
                part[sg_id * N_PER_WG + lane * COLS_PER_LANE + c] = acc[c];
            sycl::group_barrier(it.get_group());

            if (lid < N_PER_WG) {
                const uint32_t n = wgid * N_PER_WG + lid;
                if (n < N) {
                    float s = 0.f;
                    #pragma unroll
                    for (int sg = 0; sg < SGS_PER_WG; ++sg)
                        s += part[uint32_t(sg) * N_PER_WG + lid];
                    y[n] = sycl::half(s);
                }
            }
        });
    });
}

// Multi-row SHARED-weight variant for the spec-verify dense leaves
// (2026-08-27): R rows x one [K,N] F16 mat, W read ONCE per k step and
// applied to all R rows (the verify bandwidth amortization — rows_dual
// re-reads W per row and only kills launch overhead). Per-row accumulation
// order is the T=1 kernel's VERBATIM: same K-slice split, k ascending, same
// SLM partial layout, same 32-way cross-SG sum -> every row bit-identical to
// its gemv_fp16 call. Activations read from GLOBAL (R rows of K halfs are
// L2-resident; same VALUES the solo kernel stages through SLM).
// R buckets are templates: the first cut carried acc[8][2] for every R and
// measured 79 us avg vs the solo 34 us (register pressure); R_MAX-sized
// buckets restore the occupancy (spec verify runs R=K+1=4).
namespace {
template <int R_MAX>
sycl::event gemv_fp16_rows_impl(sycl::queue& q,
                                const sycl::half* X, uint32_t x_stride,
                                const sycl::half* W_kn, sycl::half* y,
                                uint32_t y_stride,
                                uint32_t K, uint32_t N, uint32_t R,
                                const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE       = 16;
    constexpr int SGS_PER_WG    = 32;
    constexpr int WG_ITEMS      = SGS_PER_WG * SG_SIZE;       // 512
    constexpr int COLS_PER_LANE = 2;
    constexpr int N_PER_WG      = SG_SIZE * COLS_PER_LANE;    // 32

    const uint32_t n_wgs   = (N + N_PER_WG - 1) / N_PER_WG;
    const uint32_t k_slice = (K + SGS_PER_WG - 1) / SGS_PER_WG;

    return ie::ps(q, "gemv_fp16_rowsN", [&](sycl::handler& h) {
        h.depends_on(deps);
        // Per-SG partials per row: [r][sg][col_in_wg].
        sycl::local_accessor<float, 1> part(
            sycl::range<1>(uint64_t(R_MAX) * SGS_PER_WG * N_PER_WG), h);

        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t wgid  = uint32_t(it.get_group(0));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;

            const uint32_t n0 = wgid * N_PER_WG + lane * COLS_PER_LANE;
            const uint32_t k0 = sg_id * k_slice;
            const uint32_t k1 = sycl::min(k0 + k_slice, K);

            float acc[R_MAX][COLS_PER_LANE];
            #pragma unroll
            for (int r = 0; r < R_MAX; ++r)
                for (int c = 0; c < COLS_PER_LANE; ++c) acc[r][c] = 0.f;

            if ((N & 1u) == 0 && n0 + COLS_PER_LANE - 1 < N) {
                #pragma unroll 16
                for (uint32_t k = k0; k < k1; ++k) {
                    const auto wv = *reinterpret_cast<const sycl::vec<sycl::half, 2>*>(
                        &W_kn[uint64_t(k) * N + n0]);
                    #pragma unroll
                    for (int r = 0; r < R_MAX; ++r) {
                        if (uint32_t(r) >= R) break;
                        const float a = float(X[uint64_t(r) * x_stride + k]);
                        #pragma unroll
                        for (int c = 0; c < COLS_PER_LANE; ++c)
                            acc[r][c] += a * float(wv[c]);
                    }
                }
            } else {
                for (uint32_t k = k0; k < k1; ++k) {
                    const uint64_t row = uint64_t(k) * N;
                    #pragma unroll
                    for (int r = 0; r < R_MAX; ++r) {
                        if (uint32_t(r) >= R) break;
                        const float a = float(X[uint64_t(r) * x_stride + k]);
                        #pragma unroll
                        for (int c = 0; c < COLS_PER_LANE; ++c)
                            if (n0 + c < N) acc[r][c] += a * float(W_kn[row + n0 + c]);
                    }
                }
            }
            #pragma unroll
            for (int r = 0; r < R_MAX; ++r) {
                if (uint32_t(r) >= R) break;
                for (int c = 0; c < COLS_PER_LANE; ++c)
                    part[(uint64_t(r) * SGS_PER_WG + sg_id) * N_PER_WG +
                         lane * COLS_PER_LANE + c] = acc[r][c];
            }
            sycl::group_barrier(it.get_group());

            if (lid < N_PER_WG) {
                const uint32_t n = wgid * N_PER_WG + lid;
                if (n < N) {
                    for (uint32_t r = 0; r < R; ++r) {
                        float s = 0.f;
                        #pragma unroll
                        for (int sg = 0; sg < SGS_PER_WG; ++sg)
                            s += part[(uint64_t(r) * SGS_PER_WG + uint32_t(sg)) *
                                      N_PER_WG + lid];
                        y[uint64_t(r) * y_stride + n] = sycl::half(s);
                    }
                }
            }
        });
    });
}
}  // namespace

// Per-row dot against ONE weight vector: y[r] = X[r,:] . w  — the N=1 GEMV
// shape (MoE shared-expert scalar gate), which the per-token loop launched
// 49K times per prefill chunk (~540 ms of pure submits, 2026-08-27 profile).
// One WG per row REPLICATES gemv_fp16(N=1)'s math verbatim: 32 K-slices,
// only lane 0 of each SG accumulates (the N=1 tail path's live lane), same
// SLM partial layout, same 32-way cross-SG sum -> bit-identical to R solo
// gemv_fp16(..., N=1) calls in ONE launch.
sycl::event gemv_fp16_dotrows(sycl::queue& q,
                              const sycl::half* X, uint32_t x_stride,
                              const sycl::half* w, sycl::half* y,
                              uint32_t K, uint32_t R,
                              const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE    = 16;
    constexpr int SGS_PER_WG = 32;
    constexpr int WG_ITEMS   = SGS_PER_WG * SG_SIZE;   // 512
    const uint32_t k_slice = (K + SGS_PER_WG - 1) / SGS_PER_WG;

    return ie::ps(q, "gemv_fp16_dotrows", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<float, 1> part(sycl::range<1>(SGS_PER_WG), h);
        h.parallel_for(sycl::nd_range<1>(uint64_t(R) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t r     = uint32_t(it.get_group(0));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const sycl::half* A = X + uint64_t(r) * x_stride;

            const uint32_t k0 = sg_id * k_slice;
            const uint32_t k1 = sycl::min(k0 + k_slice, K);
            float acc = 0.f;
            if (lane == 0)
                for (uint32_t k = k0; k < k1; ++k)
                    acc += float(A[k]) * float(w[k]);
            if (lane == 0) part[sg_id] = acc;
            sycl::group_barrier(it.get_group());
            if (lid == 0) {
                float s = 0.f;
                #pragma unroll
                for (int sg = 0; sg < SGS_PER_WG; ++sg) s += part[sg];
                y[r] = sycl::half(s);
            }
        });
    });
}

sycl::event gemv_fp16_rows(sycl::queue& q,
                           const sycl::half* X, uint32_t x_stride,
                           const sycl::half* W_kn, sycl::half* y,
                           uint32_t y_stride,
                           uint32_t K, uint32_t N, uint32_t R,
                           const std::vector<sycl::event>& deps) {
    // R beyond the largest bucket falls back to exact per-row leaves —
    // NEVER silently drop rows (the gemv_q4_K_q8 T_MAX lesson, 2026-08-27).
    if (R > 8) {
        sycl::event e;
        for (uint32_t r = 0; r < R; ++r)
            e = gemv_fp16(q, X + uint64_t(r) * x_stride, W_kn,
                          y + uint64_t(r) * y_stride, K, N, deps);
        return e;
    }
    if (R <= 2) return gemv_fp16_rows_impl<2>(q, X, x_stride, W_kn, y, y_stride, K, N, R, deps);
    if (R <= 4) return gemv_fp16_rows_impl<4>(q, X, x_stride, W_kn, y, y_stride, K, N, R, deps);
    return gemv_fp16_rows_impl<8>(q, X, x_stride, W_kn, y, y_stride, K, N, R, deps);
}

}  // namespace ie
