// src/ops/gemv_q4k.cpp — W4A16 GEMV with Q4_K weights (decode-critical kernel).
//
// y[1, N] = A[1, K] @ W[K, N]
// W is packed Q4_K with the GGUF layout: each output column n consists of
// `K/256` consecutive super-blocks (research/03 §2.2). Total bytes per column
// = K/256 * 144.  Memory layout in increasing address order:
//
//   col 0 super-blocks [0..K/256-1]
//   col 1 super-blocks [0..K/256-1]
//   ...
//   col N-1 super-blocks [0..K/256-1]
//
// One workgroup produces N_PER_WG=16 output columns. Inside the WG:
//   * 16 subgroups × 16 lanes = 256 work-items.
//   * Each subgroup is responsible for one output column.
//   * The 16 lanes split the K dimension (16 of every 256-elem super-block per
//     lane), each accumulating an fp32 partial sum, then reduced across the
//     subgroup at the end.
//
// A is loaded once into SLM (≤ 16 KiB for typical hidden sizes) and reused
// across all 16 output columns of the WG.
//
// Phase 3 gate: ≥ 365 GB/s effective input bandwidth (60% of 608 GB/s peak).

#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"

#include <sycl/sycl.hpp>
#include "ie/kernel_profiler.hpp"

namespace ie {

namespace {

inline float dev_fp16_to_fp32(uint16_t h) {
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

sycl::event gemv_q4_K(sycl::queue& q,
                      const sycl::half* A, const void* W_packed,
                      sycl::half* y,
                      uint32_t K, uint32_t N,
                      const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE   = 16;
    // N_PER_WG=32 wins for big-N shapes (attn_qkv, ssm_out, lm_head)
    // where the gemv was launch-WG-overhead-dominated at N=16 stride.
    // Smaller N (E_ffn=512 inside MoE etc.) doesn't see a difference;
    // 32 is safe everywhere.
    const int N_PER_WG = 32;
    const int WG_ITEMS = N_PER_WG * SG_SIZE;        // 512

    const auto* W = static_cast<const block_q4_K*>(W_packed);
    const uint32_t blocks_per_col = K / 256;
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "gemv_q4k", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<sycl::half, 1> A_slm(sycl::range<1>(K), h);

        h.parallel_for(sycl::nd_range<1>(n_wgs * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid    = uint32_t(it.get_local_id(0));
            const uint32_t wgid   = uint32_t(it.get_group(0));
            const uint32_t sg_id  = lid / SG_SIZE;       // 0..N_PER_WG-1
            const uint32_t lane   = lid % SG_SIZE;       // 0..SG_SIZE-1
            const uint32_t n      = wgid * N_PER_WG + sg_id;

            // Cooperative A load.
            for (uint32_t i = lid; i < K; i += WG_ITEMS) A_slm[i] = A[i];
            sycl::group_barrier(it.get_group());

            if (n >= N) return;

            // Lane → (sub-block index, half-chunk index).
            //   sub  = lane / 2 ∈ [0, 8): which of 8 sub-blocks
            //   half = lane & 1 : low (0..15) or high (16..31) half within the
            //                     32-elem sub-block
            const int sub  = int(lane) >> 1;
            const int half = int(lane) & 1;
            const int g     = sub >> 1;          // 0..3 super-half
            const int hi_nib = sub & 1;          // 0 = low nibble, 1 = high nibble
            const int qs_off = g * 32 + half * 16;
            const int out_off = g * 64 + hi_nib * 32 + half * 16;

            float acc = 0.f;
            const block_q4_K* col_blocks = &W[n * blocks_per_col];
            const int q4_shift = hi_nib ? 4 : 0;

            for (uint32_t b = 0; b < blocks_per_col; ++b) {
                const block_q4_K& blk = col_blocks[b];

                // get_scale_min_k4(sub, scales[12]) -> (s_raw, m_raw)
                uint8_t s_raw, m_raw;
                if (sub < 4) {
                    s_raw = blk.scales[sub]     & 0x3F;
                    m_raw = blk.scales[sub + 4] & 0x3F;
                } else {
                    s_raw = (blk.scales[sub + 4] & 0x0F) |
                            ((blk.scales[sub - 4] >> 6) << 4);
                    m_raw = (blk.scales[sub + 4] >>   4) |
                            ((blk.scales[sub    ] >> 6) << 4);
                }
                const float d = dev_fp16_to_fp32(blk.d)    * float(s_raw);
                const float m = dev_fp16_to_fp32(blk.dmin) * float(m_raw);

                // Vectorized 16-byte loads — qs_off is a multiple of 16.
                const sycl::vec<uint8_t, 16> qs_v =
                    *reinterpret_cast<const sycl::vec<uint8_t, 16>*>(&blk.qs[qs_off]);
                const sycl::vec<sycl::half, 16> a_v =
                    *reinterpret_cast<const sycl::vec<sycl::half, 16>*>(&A_slm[b * 256 + out_off]);

                // Algebraic fold: a*(d*q4 - m) summed over 16 elements
                //   = d * sum_aq - m * sum_a
                // Two independent FMA chains pipeline better than the original
                // dependency chain through `acc`.
                float sum_aq = 0.f;
                float sum_a  = 0.f;
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const int q4 = (qs_v[i] >> q4_shift) & 0x0F;
                    const float a = float(a_v[i]);
                    sum_aq += a * float(q4);
                    sum_a  += a;
                }
                acc += d * sum_aq - m * sum_a;
            }

            // Subgroup-wide reduce; lane 0 writes the result.
            acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
            if (lane == 0) y[n] = sycl::half(acc);
        });
    });
}

// =====================================================================
// gemv_q4_K_dual — single-launch fused (alpha, beta) GEMV.
// =====================================================================
//
// Computes BOTH y_alpha[1, N] = A[1, K] @ W_alpha[K, N] and
//          y_beta [1, N] = A[1, K] @ W_beta [K, N] in one kernel launch.
//
// Decode-time use case: DeltaNet's ssm_alpha and ssm_beta projections
// share the same input row (ws_x_normed_) and the same shape (K=H=2048,
// N=SVH=32).  In the unfused path each layer issues two gemv_q_T calls
// → 60 launches per decode step across 30 DeltaNet layers.  Fusing
// them halves that to 30, attacking the launch-overhead component
// (~10 µs / launch on Intel SYCL) directly.
//
// Lane mapping is identical to gemv_q4_K (16 SGs × 16 lanes, sub/half/g
// decomposition); we just maintain two parallel FMA chains per lane
// (g_acc + b_acc) plus the shared sum_a.  A is loaded once into SLM and
// reused across both weight reads — same trick as
// moe_decode_gate_up_silu_q4k for the routed MoE.
sycl::event gemv_q4_K_dual(sycl::queue& q,
                           const sycl::half* A,
                           const void* W_alpha_packed,
                           const void* W_beta_packed,
                           sycl::half* y_alpha,
                           sycl::half* y_beta,
                           uint32_t K, uint32_t N,
                           const char* prof_name,
                           const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE   = 16;
    constexpr int N_PER_WG  = 32;
    constexpr int WG_ITEMS  = N_PER_WG * SG_SIZE;        // 512

    const auto* W_alpha = static_cast<const block_q4_K*>(W_alpha_packed);
    const auto* W_beta  = static_cast<const block_q4_K*>(W_beta_packed);
    const uint32_t blocks_per_col = K / 256;
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, prof_name, [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<sycl::half, 1> A_slm(sycl::range<1>(K), h);

        h.parallel_for(sycl::nd_range<1>(n_wgs * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid    = uint32_t(it.get_local_id(0));
            const uint32_t wgid   = uint32_t(it.get_group(0));
            const uint32_t sg_id  = lid / SG_SIZE;
            const uint32_t lane   = lid % SG_SIZE;
            const uint32_t n      = wgid * N_PER_WG + sg_id;

            // Cooperative A load — once per WG, shared across both projections.
            for (uint32_t i = lid; i < K; i += WG_ITEMS) A_slm[i] = A[i];
            sycl::group_barrier(it.get_group());

            if (n >= N) return;

            // Same Q4_K lane mapping as gemv_q4_K.
            const int sub  = int(lane) >> 1;
            const int half = int(lane) & 1;
            const int g     = sub >> 1;
            const int hi_nib = sub & 1;
            const int qs_off = g * 32 + half * 16;
            const int out_off = g * 64 + hi_nib * 32 + half * 16;
            const int q4_shift = hi_nib ? 4 : 0;

            float a_acc = 0.f;   // ssm_alpha output accumulator
            float b_acc = 0.f;   // ssm_beta  output accumulator

            const block_q4_K* alpha_blocks = &W_alpha[uint64_t(n) * blocks_per_col];
            const block_q4_K* beta_blocks  = &W_beta [uint64_t(n) * blocks_per_col];

            for (uint32_t b = 0; b < blocks_per_col; ++b) {
                const block_q4_K& ablk = alpha_blocks[b];
                const block_q4_K& bblk = beta_blocks [b];

                // get_scale_min_k4 — duplicated for both blocks.
                uint8_t as_raw, am_raw, bs_raw, bm_raw;
                if (sub < 4) {
                    as_raw = ablk.scales[sub]     & 0x3F;
                    am_raw = ablk.scales[sub + 4] & 0x3F;
                    bs_raw = bblk.scales[sub]     & 0x3F;
                    bm_raw = bblk.scales[sub + 4] & 0x3F;
                } else {
                    as_raw = (ablk.scales[sub + 4] & 0x0F) | ((ablk.scales[sub - 4] >> 6) << 4);
                    am_raw = (ablk.scales[sub + 4] >>   4) | ((ablk.scales[sub    ] >> 6) << 4);
                    bs_raw = (bblk.scales[sub + 4] & 0x0F) | ((bblk.scales[sub - 4] >> 6) << 4);
                    bm_raw = (bblk.scales[sub + 4] >>   4) | ((bblk.scales[sub    ] >> 6) << 4);
                }
                const float a_d = dev_fp16_to_fp32(ablk.d)    * float(as_raw);
                const float a_m = dev_fp16_to_fp32(ablk.dmin) * float(am_raw);
                const float b_d = dev_fp16_to_fp32(bblk.d)    * float(bs_raw);
                const float b_m = dev_fp16_to_fp32(bblk.dmin) * float(bm_raw);

                const sycl::vec<uint8_t, 16> a_qs_v =
                    *reinterpret_cast<const sycl::vec<uint8_t, 16>*>(&ablk.qs[qs_off]);
                const sycl::vec<uint8_t, 16> b_qs_v =
                    *reinterpret_cast<const sycl::vec<uint8_t, 16>*>(&bblk.qs[qs_off]);
                const sycl::vec<sycl::half, 16> a_v =
                    *reinterpret_cast<const sycl::vec<sycl::half, 16>*>(&A_slm[b * 256 + out_off]);

                // Same algebraic fold as gemv_q4_K, with two parallel FMA
                // chains.  Three independent accumulators keep the FMA pipe
                // busy and share the activation reads across both weights.
                float sum_a_aq = 0.f, sum_b_aq = 0.f, sum_a = 0.f;
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const int aq4 = (a_qs_v[i] >> q4_shift) & 0x0F;
                    const int bq4 = (b_qs_v[i] >> q4_shift) & 0x0F;
                    const float a = float(a_v[i]);
                    sum_a_aq += a * float(aq4);
                    sum_b_aq += a * float(bq4);
                    sum_a    += a;
                }
                a_acc += a_d * sum_a_aq - a_m * sum_a;
                b_acc += b_d * sum_b_aq - b_m * sum_a;
            }

            a_acc = sycl::reduce_over_group(it.get_sub_group(), a_acc, sycl::plus<float>());
            b_acc = sycl::reduce_over_group(it.get_sub_group(), b_acc, sycl::plus<float>());
            if (lane == 0) {
                y_alpha[n] = sycl::half(a_acc);
                y_beta [n] = sycl::half(b_acc);
            }
        });
    });
}

// =====================================================================
// v1.4 shared-expert decode fusion (2026-06-09).
// =====================================================================
// The T==1 shared-expert chain was 5 launches: shared_expert_gate (a
// 22 µs latency-bound single dot), dual gate/up GEMV, swiglu, down GEMV,
// scaled_add — ~1 ms/token across 40 layers.  Now 2 launches:
//   1. gemv_q4_K_shexp_gate_up — the dual GEMV with SiLU·up combined at
//      the write, plus ONE extra workgroup computing the sigmoid gate
//      scalar (reuses the same x).
//   2. gemv_q4_K_down_accum — the down GEMV with the scaled-accumulate
//      epilogue (y += sh_g · acc), removing the ws_eo_ round-trip.
sycl::event gemv_q4_K_shexp_gate_up(sycl::queue& q,
                                    const sycl::half* A,
                                    const void* W_gate_packed,
                                    const void* W_up_packed,
                                    const float* W_shg,
                                    sycl::half* y_h, sycl::half* sh_g_out,
                                    uint32_t K, uint32_t N,
                                    const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE   = 16;
    constexpr int N_PER_WG  = 32;
    constexpr int WG_ITEMS  = N_PER_WG * SG_SIZE;        // 512

    const auto* W_gate = static_cast<const block_q4_K*>(W_gate_packed);
    const auto* W_up   = static_cast<const block_q4_K*>(W_up_packed);
    const uint32_t blocks_per_col = K / 256;
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "gemv_q4k_shexp", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<sycl::half, 1> A_slm(sycl::range<1>(K), h);

        // +1 workgroup: the sigmoid gate-scalar dot.
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs + 1) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid    = uint32_t(it.get_local_id(0));
            const uint32_t wgid   = uint32_t(it.get_group(0));
            const uint32_t sg_id  = lid / SG_SIZE;
            const uint32_t lane   = lid % SG_SIZE;

            if (wgid == n_wgs) {
                // Gate-scalar WG: sigmoid(W_shg · x), full WG on the dot.
                float partial = 0.f;
                for (uint32_t i = lid; i < K; i += WG_ITEMS)
                    partial += float(A[i]) * W_shg[i];
                const float dot = sycl::reduce_over_group(
                    it.get_group(), partial, sycl::plus<float>());
                if (lid == 0)
                    sh_g_out[0] = sycl::half(1.0f / (1.0f + sycl::native::exp(-dot)));
                return;
            }

            const uint32_t n = wgid * N_PER_WG + sg_id;
            for (uint32_t i = lid; i < K; i += WG_ITEMS) A_slm[i] = A[i];
            sycl::group_barrier(it.get_group());
            if (n >= N) return;

            const int sub  = int(lane) >> 1;
            const int half = int(lane) & 1;
            const int g     = sub >> 1;
            const int hi_nib = sub & 1;
            const int qs_off = g * 32 + half * 16;
            const int out_off = g * 64 + hi_nib * 32 + half * 16;
            const int q4_shift = hi_nib ? 4 : 0;

            float g_acc = 0.f, u_acc = 0.f;
            const block_q4_K* gate_blocks = &W_gate[uint64_t(n) * blocks_per_col];
            const block_q4_K* up_blocks   = &W_up  [uint64_t(n) * blocks_per_col];

            for (uint32_t b = 0; b < blocks_per_col; ++b) {
                const block_q4_K& gblk = gate_blocks[b];
                const block_q4_K& ublk = up_blocks [b];
                uint8_t gs_raw, gm_raw, us_raw, um_raw;
                if (sub < 4) {
                    gs_raw = gblk.scales[sub]     & 0x3F;
                    gm_raw = gblk.scales[sub + 4] & 0x3F;
                    us_raw = ublk.scales[sub]     & 0x3F;
                    um_raw = ublk.scales[sub + 4] & 0x3F;
                } else {
                    gs_raw = (gblk.scales[sub + 4] & 0x0F) | ((gblk.scales[sub - 4] >> 6) << 4);
                    gm_raw = (gblk.scales[sub + 4] >>   4) | ((gblk.scales[sub    ] >> 6) << 4);
                    us_raw = (ublk.scales[sub + 4] & 0x0F) | ((ublk.scales[sub - 4] >> 6) << 4);
                    um_raw = (ublk.scales[sub + 4] >>   4) | ((ublk.scales[sub    ] >> 6) << 4);
                }
                const float g_d = dev_fp16_to_fp32(gblk.d)    * float(gs_raw);
                const float g_m = dev_fp16_to_fp32(gblk.dmin) * float(gm_raw);
                const float u_d = dev_fp16_to_fp32(ublk.d)    * float(us_raw);
                const float u_m = dev_fp16_to_fp32(ublk.dmin) * float(um_raw);

                const sycl::vec<uint8_t, 16> g_qs_v =
                    *reinterpret_cast<const sycl::vec<uint8_t, 16>*>(&gblk.qs[qs_off]);
                const sycl::vec<uint8_t, 16> u_qs_v =
                    *reinterpret_cast<const sycl::vec<uint8_t, 16>*>(&ublk.qs[qs_off]);
                const sycl::vec<sycl::half, 16> a_v =
                    *reinterpret_cast<const sycl::vec<sycl::half, 16>*>(&A_slm[b * 256 + out_off]);

                float sum_g_aq = 0.f, sum_u_aq = 0.f, sum_a = 0.f;
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const int gq4 = (g_qs_v[i] >> q4_shift) & 0x0F;
                    const int uq4 = (u_qs_v[i] >> q4_shift) & 0x0F;
                    const float a = float(a_v[i]);
                    sum_g_aq += a * float(gq4);
                    sum_u_aq += a * float(uq4);
                    sum_a    += a;
                }
                g_acc += g_d * sum_g_aq - g_m * sum_a;
                u_acc += u_d * sum_u_aq - u_m * sum_a;
            }

            g_acc = sycl::reduce_over_group(it.get_sub_group(), g_acc, sycl::plus<float>());
            u_acc = sycl::reduce_over_group(it.get_sub_group(), u_acc, sycl::plus<float>());
            if (lane == 0) {
                const float silu_g = g_acc / (1.0f + sycl::native::exp(-g_acc));
                y_h[n] = sycl::half(silu_g * u_acc);
            }
        });
    });
}

// Down GEMV with scaled-accumulate epilogue: y[n] += sh_g[0] · (W·h).
sycl::event gemv_q4_K_down_accum(sycl::queue& q,
                                 const sycl::half* A, const void* W_packed,
                                 const sycl::half* sh_g, sycl::half* y,
                                 uint32_t K, uint32_t N,
                                 const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE   = 16;
    constexpr int N_PER_WG  = 32;
    constexpr int WG_ITEMS  = N_PER_WG * SG_SIZE;

    const auto* W = static_cast<const block_q4_K*>(W_packed);
    const uint32_t blocks_per_col = K / 256;
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "gemv_q4k_down_acc", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<sycl::half, 1> A_slm(sycl::range<1>(K), h);

        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid    = uint32_t(it.get_local_id(0));
            const uint32_t wgid   = uint32_t(it.get_group(0));
            const uint32_t sg_id  = lid / SG_SIZE;
            const uint32_t lane   = lid % SG_SIZE;
            const uint32_t n      = wgid * N_PER_WG + sg_id;

            for (uint32_t i = lid; i < K; i += WG_ITEMS) A_slm[i] = A[i];
            sycl::group_barrier(it.get_group());
            if (n >= N) return;

            const int sub  = int(lane) >> 1;
            const int half = int(lane) & 1;
            const int g     = sub >> 1;
            const int hi_nib = sub & 1;
            const int qs_off = g * 32 + half * 16;
            const int out_off = g * 64 + hi_nib * 32 + half * 16;
            const int q4_shift = hi_nib ? 4 : 0;

            float acc = 0.f;
            const block_q4_K* col_blocks = &W[uint64_t(n) * blocks_per_col];

            for (uint32_t b = 0; b < blocks_per_col; ++b) {
                const block_q4_K& blk = col_blocks[b];
                uint8_t s_raw, m_raw;
                if (sub < 4) {
                    s_raw = blk.scales[sub]     & 0x3F;
                    m_raw = blk.scales[sub + 4] & 0x3F;
                } else {
                    s_raw = (blk.scales[sub + 4] & 0x0F) | ((blk.scales[sub - 4] >> 6) << 4);
                    m_raw = (blk.scales[sub + 4] >>   4) | ((blk.scales[sub    ] >> 6) << 4);
                }
                const float d = dev_fp16_to_fp32(blk.d)    * float(s_raw);
                const float m = dev_fp16_to_fp32(blk.dmin) * float(m_raw);

                const sycl::vec<uint8_t, 16> qs_v =
                    *reinterpret_cast<const sycl::vec<uint8_t, 16>*>(&blk.qs[qs_off]);
                const sycl::vec<sycl::half, 16> a_v =
                    *reinterpret_cast<const sycl::vec<sycl::half, 16>*>(&A_slm[b * 256 + out_off]);

                float sum_aq = 0.f, sum_a = 0.f;
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const int q4 = (qs_v[i] >> q4_shift) & 0x0F;
                    const float a = float(a_v[i]);
                    sum_aq += a * float(q4);
                    sum_a  += a;
                }
                acc += d * sum_aq - m * sum_a;
            }

            acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
            if (lane == 0)
                y[n] = sycl::half(float(y[n]) + float(sh_g[0]) * acc);
        });
    });
}

// v1.4: T==1 alpha/beta dual GEMV with compute_g_beta fused at the tail.
// The dual SSM GEMV is a single-WG latency-bound launch (N=32) chaining
// directly into the tiny g/beta kernel — one launch instead of two, and
// the alpha/beta fp16 intermediates never touch global memory.  To keep
// numerics IDENTICAL to the unfused chain, the accumulators round through
// fp16 exactly where the old path wrote/read ws_alpha/beta_fp16_.
sycl::event gemv_q4_K_dual_ssm_gbeta(sycl::queue& q,
                                     const sycl::half* A,
                                     const void* W_alpha_packed,
                                     const void* W_beta_packed,
                                     const float* A_log, const float* dt_bias,
                                     float* g_out, float* beta_out,
                                     uint32_t K, uint32_t N,
                                     const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE   = 16;
    constexpr int N_PER_WG  = 32;
    constexpr int WG_ITEMS  = N_PER_WG * SG_SIZE;

    const auto* W_alpha = static_cast<const block_q4_K*>(W_alpha_packed);
    const auto* W_beta  = static_cast<const block_q4_K*>(W_beta_packed);
    const uint32_t blocks_per_col = K / 256;
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "gemv_q4k_ssm_gb", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<sycl::half, 1> A_slm(sycl::range<1>(K), h);
        sycl::local_accessor<float, 1> sa(N_PER_WG, h);
        sycl::local_accessor<float, 1> sb(N_PER_WG, h);

        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid    = uint32_t(it.get_local_id(0));
            const uint32_t wgid   = uint32_t(it.get_group(0));
            const uint32_t sg_id  = lid / SG_SIZE;
            const uint32_t lane   = lid % SG_SIZE;
            const uint32_t n      = wgid * N_PER_WG + sg_id;

            for (uint32_t i = lid; i < K; i += WG_ITEMS) A_slm[i] = A[i];
            sycl::group_barrier(it.get_group());

            if (n < N) {
                const int sub  = int(lane) >> 1;
                const int half = int(lane) & 1;
                const int g     = sub >> 1;
                const int hi_nib = sub & 1;
                const int qs_off = g * 32 + half * 16;
                const int out_off = g * 64 + hi_nib * 32 + half * 16;
                const int q4_shift = hi_nib ? 4 : 0;

                float a_acc = 0.f, b_acc = 0.f;
                const block_q4_K* alpha_blocks = &W_alpha[uint64_t(n) * blocks_per_col];
                const block_q4_K* beta_blocks  = &W_beta [uint64_t(n) * blocks_per_col];

                for (uint32_t b = 0; b < blocks_per_col; ++b) {
                    const block_q4_K& ablk = alpha_blocks[b];
                    const block_q4_K& bblk = beta_blocks [b];
                    uint8_t as_raw, am_raw, bs_raw, bm_raw;
                    if (sub < 4) {
                        as_raw = ablk.scales[sub]     & 0x3F;
                        am_raw = ablk.scales[sub + 4] & 0x3F;
                        bs_raw = bblk.scales[sub]     & 0x3F;
                        bm_raw = bblk.scales[sub + 4] & 0x3F;
                    } else {
                        as_raw = (ablk.scales[sub + 4] & 0x0F) | ((ablk.scales[sub - 4] >> 6) << 4);
                        am_raw = (ablk.scales[sub + 4] >>   4) | ((ablk.scales[sub    ] >> 6) << 4);
                        bs_raw = (bblk.scales[sub + 4] & 0x0F) | ((bblk.scales[sub - 4] >> 6) << 4);
                        bm_raw = (bblk.scales[sub + 4] >>   4) | ((bblk.scales[sub    ] >> 6) << 4);
                    }
                    const float a_d = dev_fp16_to_fp32(ablk.d)    * float(as_raw);
                    const float a_m = dev_fp16_to_fp32(ablk.dmin) * float(am_raw);
                    const float b_d = dev_fp16_to_fp32(bblk.d)    * float(bs_raw);
                    const float b_m = dev_fp16_to_fp32(bblk.dmin) * float(bm_raw);

                    const sycl::vec<uint8_t, 16> a_qs_v =
                        *reinterpret_cast<const sycl::vec<uint8_t, 16>*>(&ablk.qs[qs_off]);
                    const sycl::vec<uint8_t, 16> b_qs_v =
                        *reinterpret_cast<const sycl::vec<uint8_t, 16>*>(&bblk.qs[qs_off]);
                    const sycl::vec<sycl::half, 16> a_v =
                        *reinterpret_cast<const sycl::vec<sycl::half, 16>*>(&A_slm[b * 256 + out_off]);

                    float sum_a_aq = 0.f, sum_b_aq = 0.f, sum_a = 0.f;
                    #pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        const int aq4 = (a_qs_v[i] >> q4_shift) & 0x0F;
                        const int bq4 = (b_qs_v[i] >> q4_shift) & 0x0F;
                        const float a = float(a_v[i]);
                        sum_a_aq += a * float(aq4);
                        sum_b_aq += a * float(bq4);
                        sum_a    += a;
                    }
                    a_acc += a_d * sum_a_aq - a_m * sum_a;
                    b_acc += b_d * sum_b_aq - b_m * sum_a;
                }

                a_acc = sycl::reduce_over_group(it.get_sub_group(), a_acc, sycl::plus<float>());
                b_acc = sycl::reduce_over_group(it.get_sub_group(), b_acc, sycl::plus<float>());
                if (lane == 0) {
                    // Round through fp16 exactly like the old global round-trip.
                    sa[sg_id] = float(sycl::half(a_acc));
                    sb[sg_id] = float(sycl::half(b_acc));
                }
            }
            sycl::group_barrier(it.get_group());

            // g/beta tail — same math as compute_g_beta_h16.
            const uint32_t hh = lid;
            if (hh < N) {
                const float a_h = sa[hh] + dt_bias[hh];
                const float ax  = sycl::fabs(a_h);
                const float sp  = sycl::fmax(a_h, 0.f) +
                                  sycl::log1p(sycl::native::exp(-ax));
                g_out[hh]    = A_log[hh] * sp;
                beta_out[hh] = 1.0f / (1.0f + sycl::native::exp(-sb[hh]));
            }
        });
    });
}

// Thin profile-named aliases — keep the bucketing distinct in the kernel
// monitor so the two fusion sites can be measured independently.
sycl::event gemv_q4_K_dual_ssm(sycl::queue& q,
                               const sycl::half* A,
                               const void* W_alpha, const void* W_beta,
                               sycl::half* y_alpha, sycl::half* y_beta,
                               uint32_t K, uint32_t N,
                               const std::vector<sycl::event>& deps) {
    return gemv_q4_K_dual(q, A, W_alpha, W_beta, y_alpha, y_beta,
                          K, N, "gemv_q4k_dual_ssm", deps);
}

sycl::event gemv_q4_K_dual_ffn(sycl::queue& q,
                               const sycl::half* A,
                               const void* W_gate, const void* W_up,
                               sycl::half* y_gate, sycl::half* y_up,
                               uint32_t K, uint32_t N,
                               const std::vector<sycl::event>& deps) {
    return gemv_q4_K_dual(q, A, W_gate, W_up, y_gate, y_up,
                          K, N, "gemv_q4k_dual_ffn", deps);
}

// =====================================================================
// gemm_q4_K — multi-row GEMM with Q4_K weights, M ≤ M_TILE per launch.
// =====================================================================
// y[M, N] = A[M, K] @ W[K, N]  (W = Q4_K-packed, same layout as gemv_q4_K)
//
// Why: prefill currently calls `gemv_q_T` which loops `gemv_q4_K` over T
// tokens, paying the full per-row weight read T times. With M tokens
// processed in ONE launch, the weight reads are amortized M× — same A
// SLM tile, same dequant, M output rows updated in parallel.
//
// Caller chunks M into M_TILE-sized passes. M_TILE=8 keeps the SLM
// budget sane (M_TILE × K halfs = 8 × 2048 × 2 = 32 KiB per WG) and
// gives ~8× weight-read amortization.
//
// WG layout matches gemv_q4_K (16 SGs × 16 lanes = 256 lanes), each SG
// produces one output column for ALL M rows. Per-lane state is M floats
// (one accumulator per row).
sycl::event gemm_q4_K(sycl::queue& q,
                      const sycl::half* A, const void* W_packed,
                      sycl::half* y,
                      uint32_t M, uint32_t K, uint32_t N,
                      const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE   = 16;
    constexpr int N_PER_WG  = 16;
    constexpr int WG_ITEMS  = N_PER_WG * SG_SIZE;
    constexpr int M_TILE    = 32;       // rows per WG; any M now allowed (M-tile grid dim).
    if (M == 0) return {};

    const auto* W = static_cast<const block_q4_K*>(W_packed);
    const uint32_t blocks_per_col = K / 256;
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;
    // E2b (docs/prefill_attack_plan_2026-06-09.md): M is tiled in-kernel via
    // grid dim 0 instead of the caller looping launches.  For the tiny-N
    // projections (alpha/beta, N=32) this turns 16 underoccupied launches
    // into one launch with 16× the workgroups.
    const uint32_t m_tiles = (M + M_TILE - 1) / M_TILE;
    const uint32_t slm_rows = std::min<uint32_t>(M, M_TILE);

    return ie::ps(q, "gemm_q4k", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<sycl::half, 1> A_slm(uint64_t(slm_rows) * K, h);

        h.parallel_for(sycl::nd_range<2>({m_tiles, uint64_t(n_wgs) * WG_ITEMS},
                                         {1, WG_ITEMS}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid    = uint32_t(it.get_local_id(1));
            const uint32_t wgid   = uint32_t(it.get_group(1));
            const uint32_t sg_id  = lid / SG_SIZE;
            const uint32_t lane   = lid % SG_SIZE;
            const uint32_t n      = wgid * N_PER_WG + sg_id;
            const uint32_t m0     = uint32_t(it.get_group(0)) * M_TILE;
            const uint32_t Mc     = sycl::min(uint32_t(M_TILE), M - m0);

            // Cooperative A load: Mc*K halfs starting at row m0.
            const uint64_t a_total = uint64_t(Mc) * K;
            for (uint64_t i = lid; i < a_total; i += WG_ITEMS)
                A_slm[i] = A[uint64_t(m0) * K + i];
            sycl::group_barrier(it.get_group());

            if (n >= N) return;

            // Lane → (sub, half) — same lattice as gemv_q4_K.
            const int sub  = int(lane) >> 1;
            const int half = int(lane) & 1;
            const int g     = sub >> 1;
            const int hi_nib = sub & 1;
            const int qs_off = g * 32 + half * 16;
            const int out_off = g * 64 + hi_nib * 32 + half * 16;

            float acc[M_TILE];
            #pragma unroll
            for (int mm = 0; mm < M_TILE; ++mm) acc[mm] = 0.f;

            const block_q4_K* col_blocks = &W[uint64_t(n) * blocks_per_col];

            for (uint32_t b = 0; b < blocks_per_col; ++b) {
                const block_q4_K& blk = col_blocks[b];
                uint8_t s_raw, m_raw;
                if (sub < 4) {
                    s_raw = blk.scales[sub]     & 0x3F;
                    m_raw = blk.scales[sub + 4] & 0x3F;
                } else {
                    s_raw = (blk.scales[sub + 4] & 0x0F) | ((blk.scales[sub - 4] >> 6) << 4);
                    m_raw = (blk.scales[sub + 4] >>   4) | ((blk.scales[sub    ] >> 6) << 4);
                }
                const float d = dev_fp16_to_fp32(blk.d)    * float(s_raw);
                const float dm = dev_fp16_to_fp32(blk.dmin) * float(m_raw);

                // E2b: one 16 B vector load instead of 16 byte loads, dequant
                // to registers once, then vec8 SLM reads of A.  One weight
                // read serves Mc rows; per-accumulator add order unchanged.
                const auto qv = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(&blk.qs[qs_off]);
                float wv[16];
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const uint8_t qb = uint8_t(qv[i >> 2] >> ((i & 3) * 8));
                    const int q4 = hi_nib ? (qb >> 4) : (qb & 0x0F);
                    wv[i] = d * float(q4) - dm;
                }
                const sycl::half* a_ptr =
                    A_slm.get_multi_ptr<sycl::access::decorated::no>().get();
                #pragma unroll
                for (int mm = 0; mm < M_TILE; ++mm) {
                    if (uint32_t(mm) < Mc) {
                        const auto* av = reinterpret_cast<const sycl::vec<sycl::half, 8>*>(
                            a_ptr + uint64_t(mm) * K + uint64_t(b) * 256 + out_off);
                        const auto a0 = av[0];
                        const auto a1 = av[1];
                        #pragma unroll
                        for (int i = 0; i < 8; ++i) acc[mm] += float(a0[i]) * wv[i];
                        #pragma unroll
                        for (int i = 0; i < 8; ++i) acc[mm] += float(a1[i]) * wv[i + 8];
                    }
                }
            }

            // SG-reduce one row at a time and write.
            #pragma unroll
            for (int mm = 0; mm < M_TILE; ++mm) {
                if (uint32_t(mm) < Mc) {
                    const float r = sycl::reduce_over_group(it.get_sub_group(),
                                                            acc[mm],
                                                            sycl::plus<float>());
                    if (lane == 0) y[uint64_t(m0 + mm) * N + n] = sycl::half(r);
                }
            }
        });
    });
}

}  // namespace ie
