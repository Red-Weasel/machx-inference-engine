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
    sycl::half*  xh = nullptr;  uint32_t xh_n = 0;
    sycl::half*  acc = nullptr; uint32_t acc_n = 0;
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

}  // namespace ie
