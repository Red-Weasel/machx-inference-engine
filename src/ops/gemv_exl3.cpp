// src/ops/gemv_exl3.cpp — EXL3 (QTIP-based, cb=0) trellis-decode GEMV.
//
// y[1, N] = A[1, K] @ W_rot[K, N]   where W_rot is decoded on the fly from the
// EXL3 tail-biting trellis. This is the DECODE-ONLY matmul (the Hadamard
// incoherence + suh/svh scales are applied to the activation / output around
// this kernel — see hadamard_transform + the forward). Mirrors the SG structure
// of gemv_q4_K. Ported bit-for-bit from the proven host decode
// (src/core/exl3_decode.cpp) + spec docs/exl3_format_notes.md §2,§3,§7.
//
// Tile geometry: W_rot is laid out in 16×16 tiles; tile (ki, ni) holds 256
// weights in tensor-core lane order. trellis bytes are [TK, TN, 16*bits] int16
// (TK=K/16, TN=N/16), so tile (ki,ni) begins at u32 word (ki*TN+ni)*psz with
// psz = bits*256/32 words.
//
// Parallel mapping (the Task-2 "risky" decision — resolved): one workgroup per
// TILE-COLUMN ni (TN workgroups), N_PER_WG=16 subgroups × SG_SIZE=16 lanes.
//   * subgroup sg_id = within-tile column c (0..15) → output column n = ni*16+c
//   * lane          = within-tile row    r (0..15)
// Each lane owns exactly ONE of the 16 weights in its column for every ki tile;
// its tensor-core lane index `my_tt` is constant across all ki (computed once
// from the inverse tile permutation). So each weight in the tensor is decoded
// exactly once (no redundancy), and the 16 lanes reduce over the K-rows of the
// tile. A is staged once in SLM and shared by all 16 columns of the WG.

#include "ie/ops.hpp"

#include <sycl/sycl.hpp>
#include "ie/kernel_profiler.hpp"

#include <algorithm>
#include <mutex>
#include <vector>

namespace ie {

namespace {

// codebook.cuh:25-35, cb=0 — identical arithmetic to the host decode_cb0:
// MCG hash, lop3 0x6a = (a&b)^c, then sum the two fp16 halves (rounded in fp16).
inline float decode_cb0_dev(uint32_t code) {
    uint32_t x = code * 89226354u + 64248484u;
    x = (x & 0x8fff8fffu) ^ 0x3b603b60u;
    const uint16_t lob = uint16_t(x & 0xffffu);
    const uint16_t hib = uint16_t(x >> 16);
    const sycl::half lo = sycl::bit_cast<sycl::half>(lob);
    const sycl::half hi = sycl::bit_cast<sycl::half>(hib);
    return float(sycl::half(float(lo) + float(hi)));
}

}  // namespace

sycl::event gemv_exl3(sycl::queue& q,
                      const sycl::half* A, const void* codes,
                      sycl::half* y,
                      uint32_t K, uint32_t N, uint32_t bits,
                      const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE  = 16;
    constexpr int N_PER_WG = 16;                 // one tile-column per WG
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE; // 256 = exactly one tile's lanes

    const uint32_t TN  = N / 16;
    const uint32_t TK  = K / 16;
    const uint32_t psz = bits * 256u / 32u;      // u32 words per tile
    const uint32_t n_wgs = TN;
    const auto* codes_u32 = static_cast<const uint32_t*>(codes);

    return ie::ps(q, "gemv_exl3", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<sycl::half, 1> A_slm(sycl::range<1>(K), h);
        sycl::local_accessor<int, 1>        inv(sycl::range<1>(256), h);

        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid = uint32_t(it.get_local_id(0));
            const uint32_t ni  = uint32_t(it.get_group(0));
            const uint32_t c   = lid / SG_SIZE;   // subgroup id → within-tile column
            const uint32_t r   = lid % SG_SIZE;   // lane        → within-tile row

            // Build the inverse tensor-core permutation (quantize.py:21-48) in SLM:
            // forward perm[t*8+kk] = (rr,cc); we store inv[rr*16+cc] = t*8+kk = lane index.
            {
                const int t  = int(lid) >> 3;     // 0..31
                const int kk = int(lid) & 7;      // 0..7
                const int r0 = (t & 3) * 2, r1 = r0 + 1, r2 = r0 + 8, r3 = r0 + 9;
                const int c0 = t >> 2,      c1 = c0 + 8;
                int rr, cc;
                switch (kk) {
                    case 0: rr = r0; cc = c0; break;
                    case 1: rr = r1; cc = c0; break;
                    case 2: rr = r2; cc = c0; break;
                    case 3: rr = r3; cc = c0; break;
                    case 4: rr = r0; cc = c1; break;
                    case 5: rr = r1; cc = c1; break;
                    case 6: rr = r2; cc = c1; break;
                    default: rr = r3; cc = c1; break;
                }
                inv[rr * 16 + cc] = int(lid);
            }

            for (uint32_t i = lid; i < K; i += WG_ITEMS) A_slm[i] = A[i];
            sycl::group_barrier(it.get_group());

            const uint32_t n = ni * 16 + c;

            // This lane's constant tensor-core code index, and its tail-biting
            // bit-field straddle (exl3_dq.cuh:15-31) — all loop-invariant in ki.
            const int my_tt = inv[int(r * 16 + c)];
            const int b0 = int(uint32_t(my_tt) * bits + bits - 16u + 256u * bits);
            const int i0 = b0 / 32;
            const int i1 = (b0 + 15) / 32;
            const int s0 = (i1 + 1) * 32 - (b0 + 16);
            const uint32_t i0w = uint32_t(i0) % psz;
            const uint32_t i1w = uint32_t(i1) % psz;

            // 4 independent decode+FMA chains (ILP): the per-weight decode is a
            // latency chain (MCG hash → lop3 → 2×fp16 widen → add), so serializing
            // 128 of them through one accumulator stalls the pipe. TK = K/16 is
            // always a multiple of 8 (K%128==0) → TK%4==0, no tail.
            constexpr uint32_t U = 8;
            float accv[U] = {};
            for (uint32_t ki = 0; ki < TK; ki += U) {
                #pragma unroll
                for (uint32_t u = 0; u < U; ++u) {
                    const uint32_t base = ((ki + u) * TN + ni) * psz;
                    const uint32_t a = codes_u32[base + i0w];
                    const uint32_t b = codes_u32[base + i1w];
                    const uint64_t merged = (uint64_t(a) << 32) | b;   // funnel-shift
                    const uint32_t code = uint32_t(merged >> s0) & 0xffffu;
                    accv[u] += float(A_slm[(ki + u) * 16 + r]) * decode_cb0_dev(code);
                }
            }
            float acc = 0.f;
            #pragma unroll
            for (uint32_t u = 0; u < U; ++u) acc += accv[u];

            acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
            if (r == 0) y[n] = sycl::half(acc);
        });
    });
}

// Row-batched EXL3 trellis-decode GEMV for the fused MoE. Identical decode math
// to gemv_exl3 (same tensor-core perm, tail-biting straddle, 8-way ILP), with a
// leading row dimension: group(0)=row r, group(1)=tile-column ni. Each row reads
// A[r,:K] and decodes the trellis of expert row_expert[r] (codes_base +
// e*expert_stride_bytes) into y[r,:N]. `bits` is uniform across a layer's experts.
sycl::event gemv_exl3_moe(sycl::queue& q,
                          const sycl::half* A, const void* codes_base,
                          uint64_t expert_stride_bytes,
                          const int32_t* row_expert,
                          sycl::half* y,
                          uint32_t K, uint32_t N, uint32_t R, uint32_t bits,
                          const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE  = 16;
    constexpr int N_PER_WG = 16;                 // one tile-column per WG
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE; // 256 = one tile's lanes

    const uint32_t TN  = N / 16;
    const uint32_t TK  = K / 16;
    const uint32_t psz = bits * 256u / 32u;      // u32 words per tile
    const auto* base8  = static_cast<const uint8_t*>(codes_base);

    return ie::ps(q, "gemv_exl3_moe", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<sycl::half, 1> A_slm(sycl::range<1>(K), h);
        sycl::local_accessor<int, 1>        inv(sycl::range<1>(256), h);

        h.parallel_for(sycl::nd_range<2>({uint64_t(R), uint64_t(TN) * WG_ITEMS},
                                          {1, WG_ITEMS}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t row = uint32_t(it.get_group(0));
            const uint32_t ni  = uint32_t(it.get_group(1));
            const uint32_t lid = uint32_t(it.get_local_id(1));
            const uint32_t c   = lid / SG_SIZE;   // subgroup id → within-tile column
            const uint32_t r   = lid % SG_SIZE;   // lane        → within-tile row

            // Inverse tensor-core permutation in SLM (identical to gemv_exl3).
            {
                const int t  = int(lid) >> 3;
                const int kk = int(lid) & 7;
                const int r0 = (t & 3) * 2, r1 = r0 + 1, r2 = r0 + 8, r3 = r0 + 9;
                const int c0 = t >> 2,      c1 = c0 + 8;
                int rr, cc;
                switch (kk) {
                    case 0: rr = r0; cc = c0; break;
                    case 1: rr = r1; cc = c0; break;
                    case 2: rr = r2; cc = c0; break;
                    case 3: rr = r3; cc = c0; break;
                    case 4: rr = r0; cc = c1; break;
                    case 5: rr = r1; cc = c1; break;
                    case 6: rr = r2; cc = c1; break;
                    default: rr = r3; cc = c1; break;
                }
                inv[rr * 16 + cc] = int(lid);
            }

            const sycl::half* Arow = A + uint64_t(row) * K;
            for (uint32_t i = lid; i < K; i += WG_ITEMS) A_slm[i] = Arow[i];
            sycl::group_barrier(it.get_group());

            // This row's expert → trellis base.
            const uint32_t e = uint32_t(row_expert[row]);
            const auto* codes_u32 = reinterpret_cast<const uint32_t*>(
                base8 + uint64_t(e) * expert_stride_bytes);

            const uint32_t n = ni * 16 + c;
            const int my_tt = inv[int(r * 16 + c)];
            const int b0 = int(uint32_t(my_tt) * bits + bits - 16u + 256u * bits);
            const int i0 = b0 / 32;
            const int i1 = (b0 + 15) / 32;
            const int s0 = (i1 + 1) * 32 - (b0 + 16);
            const uint32_t i0w = uint32_t(i0) % psz;
            const uint32_t i1w = uint32_t(i1) % psz;

            constexpr uint32_t U = 8;
            float accv[U] = {};
            for (uint32_t ki = 0; ki < TK; ki += U) {
                #pragma unroll
                for (uint32_t u = 0; u < U; ++u) {
                    const uint32_t bse = ((ki + u) * TN + ni) * psz;
                    const uint32_t a = codes_u32[bse + i0w];
                    const uint32_t b = codes_u32[bse + i1w];
                    const uint64_t merged = (uint64_t(a) << 32) | b;
                    const uint32_t code = uint32_t(merged >> s0) & 0xffffu;
                    accv[u] += float(A_slm[(ki + u) * 16 + r]) * decode_cb0_dev(code);
                }
            }
            float acc = 0.f;
            #pragma unroll
            for (uint32_t u = 0; u < U; ++u) acc += accv[u];

            acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
            if (r == 0) y[uint64_t(row) * N + n] = sycl::half(acc);
        });
    });
}

namespace {

// Per-device scratch for the EXL3 forward (xh[K_max] + acc[N_max]). Keyed by
// sycl::device so a multi-GPU fleet (the 80B) gets one set per card — same
// per-device discipline as the gemv_q_T scratch fix. Grows monotonically.
struct Exl3Scratch {
    sycl::device dev;
    sycl::queue  q;
    sycl::half*  xh = nullptr;  uint32_t xh_n = 0;   // single-row forward: had128(A⊙suh)
    sycl::half*  acc = nullptr; uint32_t acc_n = 0;  // single-row forward: xh@W_rot
    // Batched prefill (gemm_exl3) scratch, grown monotonically:
    sycl::half*  xhT  = nullptr; uint64_t xhT_n  = 0;  // [T*K] Hadamard-transformed activations
    sycl::half*  accT = nullptr; uint64_t accT_n = 0;  // [T*N] batched GEMM output
    int32_t*     zeros = nullptr; uint32_t zeros_n = 0; // [T] all-zero "expert" index → suh/svh broadcast
};
std::mutex g_exl3_mu;
std::vector<Exl3Scratch> g_exl3_scratch;

Exl3Scratch& scratch_for(sycl::queue& q) {
    const sycl::device d = q.get_device();
    for (auto& s : g_exl3_scratch) if (s.dev == d) return s;
    g_exl3_scratch.push_back(Exl3Scratch{d, q, nullptr, 0, nullptr, 0});
    return g_exl3_scratch.back();
}

}  // namespace

sycl::event gemv_exl3_forward(sycl::queue& q,
                              const sycl::half* A, const void* codes,
                              const sycl::half* suh, const sycl::half* svh,
                              sycl::half* y,
                              uint32_t K, uint32_t N, uint32_t bits,
                              const std::vector<sycl::event>& deps) {
    sycl::half *xh, *acc;
    {
        std::lock_guard<std::mutex> lk(g_exl3_mu);
        Exl3Scratch& s = scratch_for(q);
        if (K > s.xh_n)  { if (s.xh)  sycl::free(s.xh, s.q);  s.xh  = sycl::malloc_device<sycl::half>(K, q); s.xh_n = K; }
        if (N > s.acc_n) { if (s.acc) sycl::free(s.acc, s.q); s.acc = sycl::malloc_device<sycl::half>(N, q); s.acc_n = N; }
        xh = s.xh; acc = s.acc;
    }
    // EXL3 linear (exllamav3 forward, validated bit-exact vs weight.f16):
    //   xh  = had128(A ⊙ suh)
    //   acc = xh @ W_rot               (trellis decode-MAC)
    //   y   = had128(acc) ⊙ svh
    auto e1 = hadamard_transform(q, A, xh, K, /*pre=*/suh, /*post=*/nullptr, deps);
    auto e2 = gemv_exl3(q, xh, codes, acc, K, N, bits, {e1});
    return hadamard_transform(q, acc, y, N, /*pre=*/nullptr, /*post=*/svh, {e2});
}

// ============================================================================
// gemm_exl3 — BATCHED (T-row) EXL3 linear forward. PREFILL entry point.
// ============================================================================
//
// Semantics (per row t ∈ [0,T)): identical to gemv_exl3_forward —
//   xh_t  = had128(A[t] ⊙ suh)          (K-length, per-128 Sylvester WHT)
//   acc_t = xh_t @ W_rot                 (trellis decode-MAC over K)
//   y[t]  = had128(acc_t) ⊙ svh          (N-length WHT + output scale)
// A is [T,K] row-major, y is [T,N] row-major. The Hadamard/suh/svh wrap is the
// cheap part; the cost center is the K×N trellis decode-MAC (`gemv_exl3`).
//
// WHY BATCH (the whole point of this kernel): the EXL3 trellis decode is a pure
// ALU latency chain per weight — MCG hash → lop3 → 2×fp16 widen → add
// (decode_cb0_dev above). At T==1 (decode) that ALU cost is unavoidable and the
// kernel is ALU-bound, not memory-bound. At PREFILL (T=256/512) the shipped path
// loops the single-row forward T times, so it RE-DECODES the entire weight
// tensor T times — T× redundant ALU on the exact bottleneck. A batched GEMM
// decodes each tile's 256 weights ONCE and MACs the decoded values against all T
// activation rows. That turns the per-tile decode from O(T) to O(1), i.e. it
// amortizes the ALU-bound decode across the batch. This is the memory-stationary
// / decode-once-MAC-many transform (analogous to the expert-batched weight-
// stationary Q8_0 MoE prefill win: crown 510→963 tok/s).
//
// -----------------------------------------------------------------------------
// IMPLEMENTATION: decode-once / MAC-many batched core.
// -----------------------------------------------------------------------------
// gemm_exl3 composes three launches (per the forward, all T-batched):
//   1. Xh  = had128(A ⊙ suh)   — hadamard_transform_moe with an all-zero expert
//      index so the per-expert suh bank collapses to a single broadcast suh[K].
//   2. ACC = Xh @ W_rot        — gemm_exl3_core below (the new batched GEMM).
//   3. y   = had128(ACC) ⊙ svh — hadamard_transform_moe, zero index → svh[N].
// Only the middle GEMM is new; both Hadamard wraps already batch over rows.
//
// gemm_exl3_core is gemv_exl3's "one WG per tile-column ni" map, extended to
// carry a T-row batch with the decode HOISTED OUT of the row loop: each lane
// decodes ITS weight of tile (ki,ni) exactly once (constant my_tt / tail-biting
// straddle) and MACs that one decoded value against all T staged activation rows
// — turning per-tile decode from O(T) to O(1). Xh is staged in SLM in KC-wide
// K-chunks (SLM budget); T is split host-side into T_TILE passes so the per-lane
// fp32 accumulators acc[T_TILE] stay in registers (constant-indexed, unrolled).
// Correctness gate: batched ACC vs the per-row gemv_exl3_forward loop (the OFF
// path) — cosine ≥ 0.999999 (fp32 accumulation order differs from the per-row
// 8-way-ILP split, so not bit-identical, but numerically indistinguishable).

namespace {

// Batched EXL3 trellis-decode GEMM core: ACC[T,N] = Xh[T,K] @ W_rot[K,N].
// Decode-once/MAC-many. T is split into T_TILE passes host-side (one launch
// each); each launch writes ACC rows [t0, t0+tc).
sycl::event gemm_exl3_core(sycl::queue& q,
                           const sycl::half* Xh, const void* codes,
                           sycl::half* ACC,
                           uint32_t T, uint32_t K, uint32_t N, uint32_t bits,
                           const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE  = 16;
    constexpr int N_PER_WG = 16;                  // one tile-column per WG
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;  // 256 = one tile's lanes
    constexpr uint32_t T_TILE = 32;               // rows per launch (acc[] regs)
    constexpr uint32_t KC     = 512;              // K staged in SLM per chunk

    const uint32_t TN  = N / 16;
    const uint32_t psz = bits * 256u / 32u;       // u32 words per tile
    const auto* codes_u32 = static_cast<const uint32_t*>(codes);

    sycl::event last;
    for (uint32_t t0 = 0; t0 < T; t0 += T_TILE) {
        const uint32_t tc = (T - t0 < T_TILE) ? (T - t0) : T_TILE;
        const std::vector<sycl::event> d = (t0 == 0) ? deps
                                                     : std::vector<sycl::event>{};
        last = ie::ps(q, "gemm_exl3_core", [&](sycl::handler& h) {
            h.depends_on(d);
            // SLM: [T_TILE, KC] staged activations (constant stride KC so the
            // unrolled MAC indexes acc[] / Xh_slm by compile-time t) + inv[256].
            sycl::local_accessor<sycl::half, 1> Xh_slm(
                sycl::range<1>(size_t(T_TILE) * KC), h);
            sycl::local_accessor<int, 1> inv(sycl::range<1>(256), h);

            h.parallel_for(sycl::nd_range<1>(uint64_t(TN) * WG_ITEMS, WG_ITEMS),
                           [=](sycl::nd_item<1> it)
                               [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const uint32_t lid = uint32_t(it.get_local_id(0));
                const uint32_t ni  = uint32_t(it.get_group(0));
                const uint32_t c   = lid / SG_SIZE;   // subgroup → within-tile col
                const uint32_t r   = lid % SG_SIZE;   // lane     → within-tile row

                // Inverse tensor-core permutation in SLM (identical to gemv_exl3).
                {
                    const int t  = int(lid) >> 3;
                    const int kk = int(lid) & 7;
                    const int r0 = (t & 3) * 2, r1 = r0 + 1, r2 = r0 + 8, r3 = r0 + 9;
                    const int c0 = t >> 2,      c1 = c0 + 8;
                    int rr, cc;
                    switch (kk) {
                        case 0: rr = r0; cc = c0; break;
                        case 1: rr = r1; cc = c0; break;
                        case 2: rr = r2; cc = c0; break;
                        case 3: rr = r3; cc = c0; break;
                        case 4: rr = r0; cc = c1; break;
                        case 5: rr = r1; cc = c1; break;
                        case 6: rr = r2; cc = c1; break;
                        default: rr = r3; cc = c1; break;
                    }
                    inv[rr * 16 + cc] = int(lid);
                }
                sycl::group_barrier(it.get_group());  // inv ready before read

                // ki-invariant: this lane's tensor-core code index + straddle.
                const int my_tt = inv[int(r * 16 + c)];
                const int b0 = int(uint32_t(my_tt) * bits + bits - 16u + 256u * bits);
                const int i0 = b0 / 32;
                const int i1 = (b0 + 15) / 32;
                const int s0 = (i1 + 1) * 32 - (b0 + 16);
                const uint32_t i0w = uint32_t(i0) % psz;
                const uint32_t i1w = uint32_t(i1) % psz;

                float acc[T_TILE];
                #pragma unroll
                for (uint32_t t = 0; t < T_TILE; ++t) acc[t] = 0.f;

                for (uint32_t kc0 = 0; kc0 < K; kc0 += KC) {
                    const uint32_t KCc = (KC < K - kc0) ? KC : (K - kc0);
                    // Stage Xh[t0+t, kc0 + j] → Xh_slm[t*KC + j] (only valid rows).
                    for (uint32_t t = 0; t < tc; ++t)
                        for (uint32_t j = lid; j < KCc; j += WG_ITEMS)
                            Xh_slm[t * KC + j] = Xh[uint64_t(t0 + t) * K + kc0 + j];
                    sycl::group_barrier(it.get_group());

                    const uint32_t nkk = KCc / 16;    // ki tiles in this chunk
                    for (uint32_t kk = 0; kk < nkk; ++kk) {
                        const uint32_t ki   = kc0 / 16 + kk;
                        const uint32_t base = (ki * TN + ni) * psz;
                        const uint32_t a = codes_u32[base + i0w];
                        const uint32_t b = codes_u32[base + i1w];
                        const uint64_t merged = (uint64_t(a) << 32) | b;
                        const uint32_t code = uint32_t(merged >> s0) & 0xffffu;
                        const float wdec = decode_cb0_dev(code);   // ONCE per (lane,ki)
                        const uint32_t lk = kk * 16 + r;
                        // MAC the one decoded weight against all T_TILE rows;
                        // t≥tc is masked (acc[t] unused). Constant t → acc[] in regs.
                        #pragma unroll
                        for (uint32_t t = 0; t < T_TILE; ++t)
                            if (t < tc)
                                acc[t] += float(Xh_slm[t * KC + lk]) * wdec;
                    }
                    sycl::group_barrier(it.get_group());  // WAR before restage
                }

                const uint32_t n = ni * 16 + c;
                #pragma unroll
                for (uint32_t t = 0; t < T_TILE; ++t) {
                    const float s = sycl::reduce_over_group(
                        it.get_sub_group(), (t < tc) ? acc[t] : 0.f,
                        sycl::plus<float>());
                    if (r == 0 && t < tc)
                        ACC[uint64_t(t0 + t) * N + n] = sycl::half(s);
                }
            });
        });
    }
    return last;
}

}  // namespace

sycl::event gemm_exl3(sycl::queue& q,
                      const sycl::half* A, const void* codes,
                      const sycl::half* suh, const sycl::half* svh,
                      sycl::half* y,
                      uint32_t T, uint32_t K, uint32_t N, uint32_t bits,
                      const std::vector<sycl::event>& deps) {
    if (T == 0) return {};

    sycl::half *Xh, *ACC;
    int32_t*    zeros;
    bool        zeros_fresh = false;
    {
        std::lock_guard<std::mutex> lk(g_exl3_mu);
        Exl3Scratch& s = scratch_for(q);
        const uint64_t needXh  = uint64_t(T) * K;
        const uint64_t needAcc = uint64_t(T) * N;
        if (needXh > s.xhT_n) {
            if (s.xhT) sycl::free(s.xhT, s.q);
            s.xhT = sycl::malloc_device<sycl::half>(needXh, q); s.xhT_n = needXh;
        }
        if (needAcc > s.accT_n) {
            if (s.accT) sycl::free(s.accT, s.q);
            s.accT = sycl::malloc_device<sycl::half>(needAcc, q); s.accT_n = needAcc;
        }
        if (T > s.zeros_n) {
            if (s.zeros) sycl::free(s.zeros, s.q);
            s.zeros = sycl::malloc_device<int32_t>(T, q); s.zeros_n = T;
            zeros_fresh = true;   // memset once on (re)alloc; it stays zero after
        }
        Xh = s.xhT; ACC = s.accT; zeros = s.zeros;
    }

    std::vector<sycl::event> d = deps;
    if (zeros_fresh)
        d.push_back(q.memset(zeros, 0, size_t(T) * sizeof(int32_t)));

    // 1. Xh = had128(A ⊙ suh) per row (zero expert index → suh broadcast).
    auto e1 = hadamard_transform_moe(q, A, Xh, K, T, zeros,
                                     /*pre=*/suh, /*post=*/nullptr, d);
    // 2. ACC = Xh @ W_rot  (decode-once/MAC-many, T-chunked).
    auto e2 = gemm_exl3_core(q, Xh, codes, ACC, T, K, N, bits, {e1});
    // 3. y = had128(ACC) ⊙ svh per row.
    return hadamard_transform_moe(q, ACC, y, N, T, zeros,
                                  /*pre=*/nullptr, /*post=*/svh, {e2});
}

}  // namespace ie
