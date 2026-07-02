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

}  // namespace ie
