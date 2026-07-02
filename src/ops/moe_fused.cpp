// src/ops/moe_fused.cpp — fused decode-time MoE kernels.
//
// Collapses the per-expert dispatch loop (8 experts × {gate, up, swiglu, down,
// scaled_add} = 40 launches) into 2 launches:
//
//   stage 1: moe_decode_gate_up_silu_q4k (Q4_K gate + Q4_K up + swiglu fused)
//   stage 2: moe_decode_down_q4k / _q6k  (Q4_K- or Q6_K-down + per-expert
//                                          scale + accumulate over experts)
//
// Together with the existing moe_router + (per-token) shared expert path,
// this drops a MoE layer from ~47 launches to ~7 (router + 2 stages + 4
// shared-expert ops). On Battlemage where each launch is ~30 µs, that's
// ~1.2 ms/layer × 30 MoE layers = ~36 ms/token of pure launch overhead.
//
// Both kernels read `topk_idx` (and `topk_w` for stage 2) directly from
// device memory — eliminates the two host memcpy roundtrips per layer that
// the scalar dispatch needed.
//
// Layout assumptions (Qwen3.6-35B-A3B):
//   H = 2048, E_ffn = 512, K_top = 8, E_total = 256
//   gate/up: [E_total, E_ffn, H], stored Q4_K-packed along H
//   down:    [E_total, H, E_ffn], stored Q4_K- or Q6_K-packed along E_ffn
// Per-expert byte strides are passed by the caller (model layer knows them).

#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"
#include "ie/quant_soa.hpp"

#include <sycl/sycl.hpp>
#include "ie/dp4a.hpp"
#include "ie/kernel_profiler.hpp"

// Prefill-crown (2026-06-10): every kernel below that reads expert weights is
// templated on SOA.  SOA=false reproduces the historical AoS addresses
// bit-for-bit (same loads, same math order); SOA=true reads the per-expert
// struct-of-arrays streams produced by the load-time repack in qwen36.cpp
// (see ie/quant_soa.hpp).  Dispatchers take a runtime `soa` flag so one
// binary serves order-controlled A/Bs via IE_NO_MOE_SOA=1.

namespace ie {

namespace {

// Same fp16→fp32 helper as gemv_q*k.cpp. Inlined here so this file compiles
// independently and gets the same inlining behavior.
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

// ============================================================================
// Stage 1: gate + up + swiglu, Q4_K weights, K_top experts in one launch
// ============================================================================
//
// WG layout (matches gemv_q4_K so register pressure / SLM footprint stays
// close to the validated baseline):
//   group(0) = k          (∈ [0, K_top))
//   group(1) = wg_n_chunk (∈ [0, E_ffn / N_PER_WG))
//   N_PER_WG = 16, SG_SIZE = 16, WG_ITEMS = 256
//
// Per WG:
//   - cooperatively load x[H] into A_slm (4 KiB)
//   - 16 sub-groups, each computes one output column n's gate_n and up_n
//   - lane parallelizes the K direction (32 super-blocks × 16 elements per
//     lane just like gemv_q4_K)
//   - reduce both sums, apply silu(gate)*up, lane 0 writes one fp16
template <bool SOA>
static sycl::event moe_decode_gate_up_silu_q4k_impl(sycl::queue& q,
                                        const sycl::half* x,
                                        const void* gate_W, const void* up_W,
                                        const int32_t* topk_idx,
                                        sycl::half* h_out,
                                        uint32_t H, uint32_t E_ffn, uint32_t K_top,
                                        uint64_t expert_stride_bytes,
                                        const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE   = 16;
    // N_PER_WG=32 — E_ffn=512 → 16 WGs/(K_top expert), 128 WGs total at K_top=8.
    // (Was 16 → 32 wgs/expert which over-saturated for an L1-bound kernel.)
    constexpr int N_PER_WG  = 32;
    constexpr int WG_ITEMS  = N_PER_WG * SG_SIZE;   // 512

    const uint32_t blocks_per_col = H / 256;
    const uint32_t wgs_n = (E_ffn + N_PER_WG - 1) / N_PER_WG;
    const uint64_t nb_e = uint64_t(blocks_per_col) * E_ffn;  // blocks per expert

    return ie::ps(q, "moe_dec_gate_q4k", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<sycl::half, 1> A_slm(sycl::range<1>(H), h);

        h.parallel_for(sycl::nd_range<2>({uint64_t(K_top), uint64_t(wgs_n) * WG_ITEMS},
                                          {1, WG_ITEMS}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t k     = uint32_t(it.get_group(0));
            const uint32_t wg_n  = uint32_t(it.get_group(1));
            const uint32_t lid   = uint32_t(it.get_local_id(1));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t n     = wg_n * N_PER_WG + sg_id;

            // Cooperative A load — once per WG.
            for (uint32_t i = lid; i < H; i += WG_ITEMS) A_slm[i] = x[i];
            sycl::group_barrier(it.get_group());

            if (n >= E_ffn) return;

            const int32_t expert_id = topk_idx[k];
            // Layout-generic per-expert weight views (AoS or SoA streams).
            const auto gv = q4k_wview<SOA>::at(gate_W, expert_stride_bytes,
                                               uint32_t(expert_id), nb_e);
            const auto uv = q4k_wview<SOA>::at(up_W, expert_stride_bytes,
                                               uint32_t(expert_id), nb_e);

            // Lane → (sub, half) — same mapping as gemv_q4_K.
            const int sub  = int(lane) >> 1;
            const int half = int(lane) & 1;
            const int g     = sub >> 1;
            const int hi_nib = sub & 1;
            const int qs_off = g * 32 + half * 16;
            const int out_off = g * 64 + hi_nib * 32 + half * 16;

            float g_acc = 0.f;
            float u_acc = 0.f;

            const uint64_t col0 = uint64_t(n) * blocks_per_col;

            for (uint32_t b = 0; b < blocks_per_col; ++b) {
                const uint64_t bi = col0 + b;
                const uint8_t* gsc = gv.sc_blk(bi);
                const uint8_t* usc = uv.sc_blk(bi);

                // get_scale_min_k4 — duplicated for both blocks (cheap).
                uint8_t gs_raw, gm_raw, us_raw, um_raw;
                if (sub < 4) {
                    gs_raw = gsc[sub]     & 0x3F;
                    gm_raw = gsc[sub + 4] & 0x3F;
                    us_raw = usc[sub]     & 0x3F;
                    um_raw = usc[sub + 4] & 0x3F;
                } else {
                    gs_raw = (gsc[sub + 4] & 0x0F) | ((gsc[sub - 4] >> 6) << 4);
                    gm_raw = (gsc[sub + 4] >>   4) | ((gsc[sub    ] >> 6) << 4);
                    us_raw = (usc[sub + 4] & 0x0F) | ((usc[sub - 4] >> 6) << 4);
                    um_raw = (usc[sub + 4] >>   4) | ((usc[sub    ] >> 6) << 4);
                }
                const float g_d = dev_fp16_to_fp32(gv.d_bits(bi))    * float(gs_raw);
                const float g_m = dev_fp16_to_fp32(gv.dmin_bits(bi)) * float(gm_raw);
                const float u_d = dev_fp16_to_fp32(uv.d_bits(bi))    * float(us_raw);
                const float u_m = dev_fp16_to_fp32(uv.dmin_bits(bi)) * float(um_raw);

                const sycl::half* a_chunk = &A_slm[b * 256 + out_off];
                const int q4_shift = hi_nib ? 4 : 0;
                const uint8_t* gqs = gv.qs_blk(bi) + qs_off;
                const uint8_t* uqs = uv.qs_blk(bi) + qs_off;
                // Algebraic fold (same trick as gemv_q4_K):
                //   acc += a*(d*q4 - m) per element
                //   = d*Σ(a*q4) − m*Σa per block.
                // Three independent accumulators keep the FMA pipe busy.
                float sum_g_aq = 0.f, sum_u_aq = 0.f, sum_a = 0.f;
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const uint8_t gqb = gqs[i];
                    const uint8_t uqb = uqs[i];
                    const int gq4 = (gqb >> q4_shift) & 0x0F;
                    const int uq4 = (uqb >> q4_shift) & 0x0F;
                    const float a = float(a_chunk[i]);
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
                // swiglu = silu(gate) * up = gate / (1 + exp(-gate)) * up
                const float silu_g = g_acc / (1.0f + sycl::native::exp(-g_acc));
                h_out[uint64_t(k) * E_ffn + n] = sycl::half(silu_g * u_acc);
            }
        });
    });
}

sycl::event moe_decode_gate_up_silu_q4k(sycl::queue& q,
                                        const sycl::half* x,
                                        const void* gate_W, const void* up_W,
                                        const int32_t* topk_idx,
                                        sycl::half* h_out,
                                        uint32_t H, uint32_t E_ffn, uint32_t K_top,
                                        uint64_t expert_stride_bytes, bool soa,
                                        const std::vector<sycl::event>& deps) {
    return soa ? moe_decode_gate_up_silu_q4k_impl<true >(q, x, gate_W, up_W, topk_idx,
                     h_out, H, E_ffn, K_top, expert_stride_bytes, deps)
               : moe_decode_gate_up_silu_q4k_impl<false>(q, x, gate_W, up_W, topk_idx,
                     h_out, H, E_ffn, K_top, expert_stride_bytes, deps);
}

// ============================================================================
// Stage 2: down + per-expert weight scale + accumulate, Q4_K weights
// ============================================================================
//
// Each WG owns one H-output chunk (16 outputs). Within a sub-group:
//   - lane parallelizes E_ffn into 16-element chunks over the K direction
//   - the K loop has shape [K_top experts × blocks_per_col blocks]
//   - the inner unroll is the same Q4_K dequant lattice as gemv_q4_K
//
// SLM load: h[K_top * E_ffn] ≈ 8 KiB total (8×512 halfs). Fits well within
// the per-WG SLM budget alongside any compiler scratch.
template <bool SOA>
static sycl::event moe_decode_down_q4k_impl(sycl::queue& q,
                                const sycl::half* h_in,
                                const void* down_W,
                                const int32_t* topk_idx,
                                const sycl::half* topk_w,
                                sycl::half* y_out,
                                uint32_t H, uint32_t E_ffn, uint32_t K_top,
                                uint64_t expert_stride_bytes,
                                const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE  = 16;
    constexpr int N_PER_WG = 32;     // matches q6k variant; H=2048 → 64 WGs.
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;

    const uint32_t blocks_per_col = E_ffn / 256;     // = 2 for E_ffn=512
    const uint32_t n_wgs = (H + N_PER_WG - 1) / N_PER_WG;
    const uint64_t h_total = uint64_t(K_top) * E_ffn;
    const uint64_t nb_e = uint64_t(blocks_per_col) * H;  // blocks per expert

    return ie::ps(q, "moe_dec_down_q4k", [&](sycl::handler& hdl) {
        hdl.depends_on(deps);
        sycl::local_accessor<sycl::half, 1> H_slm(sycl::range<1>(h_total), hdl);
        sycl::local_accessor<int32_t, 1>   idx_slm(sycl::range<1>(K_top),  hdl);
        sycl::local_accessor<sycl::half, 1> w_slm (sycl::range<1>(K_top),  hdl);

        hdl.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * WG_ITEMS, WG_ITEMS),
                         [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid    = uint32_t(it.get_local_id(0));
            const uint32_t wgid   = uint32_t(it.get_group(0));
            const uint32_t sg_id  = lid / SG_SIZE;
            const uint32_t lane   = lid % SG_SIZE;
            const uint32_t n      = wgid * N_PER_WG + sg_id;

            // Cooperative loads.
            for (uint64_t i = lid; i < h_total; i += WG_ITEMS) H_slm[i] = h_in[i];
            if (lid < K_top) { idx_slm[lid] = topk_idx[lid]; w_slm[lid] = topk_w[lid]; }
            sycl::group_barrier(it.get_group());

            if (n >= H) return;

            const int sub  = int(lane) >> 1;
            const int half = int(lane) & 1;
            const int g     = sub >> 1;
            const int hi_nib = sub & 1;
            const int qs_off = g * 32 + half * 16;
            const int out_off = g * 64 + hi_nib * 32 + half * 16;

            float total = 0.f;
            for (uint32_t k = 0; k < K_top; ++k) {
                const int32_t expert_id = idx_slm[k];
                const float weight_k    = float(w_slm[k]);
                const auto dv = q4k_wview<SOA>::at(down_W, expert_stride_bytes,
                                                   uint32_t(expert_id), nb_e);
                const uint64_t col0 = uint64_t(n) * blocks_per_col;
                const sycl::half* h_k = &H_slm[uint64_t(k) * E_ffn];

                float k_acc = 0.f;
                for (uint32_t b = 0; b < blocks_per_col; ++b) {
                    const uint64_t bi = col0 + b;
                    const uint8_t* sc = dv.sc_blk(bi);
                    uint8_t s_raw, m_raw;
                    if (sub < 4) {
                        s_raw = sc[sub]     & 0x3F;
                        m_raw = sc[sub + 4] & 0x3F;
                    } else {
                        s_raw = (sc[sub + 4] & 0x0F) | ((sc[sub - 4] >> 6) << 4);
                        m_raw = (sc[sub + 4] >>   4) | ((sc[sub    ] >> 6) << 4);
                    }
                    const float d = dev_fp16_to_fp32(dv.d_bits(bi))    * float(s_raw);
                    const float m = dev_fp16_to_fp32(dv.dmin_bits(bi)) * float(m_raw);
                    const sycl::half* h_chunk = &h_k[b * 256 + out_off];
                    const int q4_shift = hi_nib ? 4 : 0;
                    const uint8_t* qsp = dv.qs_blk(bi) + qs_off;
                    float sum_aq = 0.f, sum_a = 0.f;
                    #pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        const uint8_t qb = qsp[i];
                        const int q4 = (qb >> q4_shift) & 0x0F;
                        const float a = float(h_chunk[i]);
                        sum_aq += a * float(q4);
                        sum_a  += a;
                    }
                    k_acc += d * sum_aq - m * sum_a;
                }
                total += weight_k * k_acc;
            }

            total = sycl::reduce_over_group(it.get_sub_group(), total, sycl::plus<float>());
            if (lane == 0) y_out[n] = sycl::half(total);
        });
    });
}

sycl::event moe_decode_down_q4k(sycl::queue& q,
                                const sycl::half* h_in,
                                const void* down_W,
                                const int32_t* topk_idx,
                                const sycl::half* topk_w,
                                sycl::half* y_out,
                                uint32_t H, uint32_t E_ffn, uint32_t K_top,
                                uint64_t expert_stride_bytes, bool soa,
                                const std::vector<sycl::event>& deps) {
    return soa ? moe_decode_down_q4k_impl<true >(q, h_in, down_W, topk_idx, topk_w,
                     y_out, H, E_ffn, K_top, expert_stride_bytes, deps)
               : moe_decode_down_q4k_impl<false>(q, h_in, down_W, topk_idx, topk_w,
                     y_out, H, E_ffn, K_top, expert_stride_bytes, deps);
}

// ============================================================================
// Stage 2: same shape contract as _q4k, but Q6_K weights
// ============================================================================
//
// USE_SG_DEQUANT_T = true: replace the per-lane strided byte loads of
//   blk.ql[ql_off+i] / blk.qh[qh_off+i] (16 strided byte reads per inner
//   iter, 8 unique ql_offs / 4 unique qh_offs across the SG) with one
//   coalesced sub-group cooperative load — same pattern that won 34% on
//   gemv_q6_K's huge/med shapes.  Per super-block, 16 lanes form a single
//   64-byte block transaction for ql (×2), qh (×1), and scales (×1).
//   Then redistribute to each lane's offsets via select_from_group.
//
// Same lane-mapping observations as gemv_q6_K:
//   scale_off == lane      → no shuffle for scales
//   ql_off ∈ {0,16,..,112} → 8 unique values, ql_off/4 always multiple of 4
//   qh_off ∈ {0,16,32,48}  → 4 unique values
//   n is uniform across the SG → early-exit doesn't break SG semantics.
//
// Shuffle constants are loop-invariant in `k` (expert) and `b` (super-block)
// — only depend on `lane` — so they're hoisted above both loops.
template <bool USE_SG_DEQUANT_T, bool SOA>
static sycl::event moe_decode_down_q6k_impl(sycl::queue& q,
                                            const sycl::half* h_in,
                                            const void* down_W,
                                            const int32_t* topk_idx,
                                            const sycl::half* topk_w,
                                            sycl::half* y_out,
                                            uint32_t H, uint32_t E_ffn, uint32_t K_top,
                                            uint64_t expert_stride_bytes,
                                            const char* prof_name,
                                            const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE  = 16;
    // N_PER_WG=32 — H=2048 → 64 WGs (≈2.7 per Xe-core, healthy occupancy) and
    // halves the per-WG H_in SLM-load count vs N_PER_WG=16.
    constexpr int N_PER_WG = 32;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;

    const uint32_t blocks_per_col = E_ffn / 256;     // = 2 for E_ffn=512
    const uint32_t n_wgs = (H + N_PER_WG - 1) / N_PER_WG;
    const uint64_t h_total = uint64_t(K_top) * E_ffn;
    const uint64_t nb_e = uint64_t(blocks_per_col) * H;  // blocks per expert

    return ie::ps(q, prof_name, [&](sycl::handler& hdl) {
        hdl.depends_on(deps);
        sycl::local_accessor<sycl::half, 1> H_slm(sycl::range<1>(h_total), hdl);
        sycl::local_accessor<int32_t, 1>   idx_slm(sycl::range<1>(K_top),  hdl);
        sycl::local_accessor<sycl::half, 1> w_slm (sycl::range<1>(K_top),  hdl);

        hdl.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * WG_ITEMS, WG_ITEMS),
                         [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid    = uint32_t(it.get_local_id(0));
            const uint32_t wgid   = uint32_t(it.get_group(0));
            const uint32_t sg_id  = lid / SG_SIZE;
            const uint32_t lane   = lid % SG_SIZE;
            const uint32_t n      = wgid * N_PER_WG + sg_id;
            auto sg = it.get_sub_group();

            for (uint64_t i = lid; i < h_total; i += WG_ITEMS) H_slm[i] = h_in[i];
            if (lid < K_top) { idx_slm[lid] = topk_idx[lid]; w_slm[lid] = topk_w[lid]; }
            sycl::group_barrier(it.get_group());

            if (n >= H) return;

            // Same Q6_K lane mapping as gemv_q6_K.
            const int half_q   = int(lane) >> 3;
            const int sub      = (int(lane) >> 1) & 0x3;
            const int l_half   = int(lane) & 0x1;
            const int l_start  = l_half * 16;
            const int ql_off   = half_q * 64 + (sub & 1) * 32 + l_start;
            const int qh_off   = half_q * 32 + l_start;
            const int scale_off = half_q * 8 + sub * 2 + l_half;
            const int qh_shift = sub * 2;
            const bool high_nibble = (sub & 2) != 0;
            const int out_off  = half_q * 128 + sub * 32 + l_start;
            const int ql_shift = high_nibble ? 4 : 0;

            // SG-coop redistribution constants — only depend on `lane`.
            const int ql_word_base = ql_off >> 2;          // {0,4,8,12,16,20,24,28}
            const bool ql_from_hi  = ql_word_base >= 16;
            const int ql_lane_base = ql_word_base & 0xF;   // 0,4,8,12
            const int qh_lane_base = qh_off >> 2;          // 0,4,8,12

            float total = 0.f;
            for (uint32_t k = 0; k < K_top; ++k) {
                const int32_t expert_id = idx_slm[k];
                const float weight_k    = float(w_slm[k]);
                const auto dv = q6k_wview<SOA>::at(down_W, expert_stride_bytes,
                                                   uint32_t(expert_id), nb_e);
                const uint64_t col0 = uint64_t(n) * blocks_per_col;
                const sycl::half* h_k = &H_slm[uint64_t(k) * E_ffn];

                float k_acc = 0.f;
                for (uint32_t b = 0; b < blocks_per_col; ++b) {
                    const uint64_t bi = col0 + b;
                    const sycl::half* h_chunk = &h_k[b * 256 + out_off];

                    if constexpr (USE_SG_DEQUANT_T) {
                        // Coalesced cooperative load — 16 lanes form a single
                        // 64-byte block transaction per call.
                        const uint32_t* ql_w = reinterpret_cast<const uint32_t*>(dv.ql_blk(bi));
                        const uint32_t* qh_w = reinterpret_cast<const uint32_t*>(dv.qh_blk(bi));
                        const uint32_t my_ql_lo = ql_w[lane];
                        const uint32_t my_ql_hi = ql_w[lane + 16];
                        const uint32_t my_qh    = qh_w[lane];
                        // scale_off == lane → cooperative scales[lane] is each
                        // lane's own scale. No shuffle needed.
                        const int8_t my_scale = dv.sc_blk(bi)[lane];
                        const float d = dev_fp16_to_fp32(dv.d_bits(bi)) * float(my_scale);

                        // Redistribute. Same lo/hi pick rule as gemv_q6_K:
                        // shuffle BOTH halves and choose per-receiver, since
                        // sender lane has its own ql_from_hi.
                        uint32_t my_ql4[4], my_qh4[4];
                        #pragma unroll
                        for (int kk = 0; kk < 4; ++kk) {
                            const sycl::id<1> ql_src(ql_lane_base + kk);
                            const uint32_t lo_w = sycl::select_from_group(sg, my_ql_lo, ql_src);
                            const uint32_t hi_w = sycl::select_from_group(sg, my_ql_hi, ql_src);
                            my_ql4[kk] = ql_from_hi ? hi_w : lo_w;
                            my_qh4[kk] = sycl::select_from_group(
                                sg, my_qh, sycl::id<1>(qh_lane_base + kk));
                        }

                        // Dequant from registers — preserves the original
                        // a*d*(qsig - 32) algebra, matching the scalar path.
                        #pragma unroll
                        for (int i = 0; i < 16; ++i) {
                            const int word_idx = i >> 2;
                            const int sh       = (i & 3) << 3;
                            const uint8_t ql_b = uint8_t(my_ql4[word_idx] >> sh);
                            const uint8_t qh_b = uint8_t(my_qh4[word_idx] >> sh);
                            const uint8_t lo_hi = uint8_t((ql_b >> ql_shift) & 0x0F);
                            const int8_t  qsig  = int8_t(lo_hi | (((qh_b >> qh_shift) & 0x3) << 4)) - 32;
                            k_acc += float(h_chunk[i]) * d * float(qsig);
                        }
                    } else {
                        // Original strided per-lane byte loads.
                        const float d = dev_fp16_to_fp32(dv.d_bits(bi)) *
                                        float(dv.sc_blk(bi)[scale_off]);
                        const uint8_t* qlp = dv.ql_blk(bi) + ql_off;
                        const uint8_t* qhp = dv.qh_blk(bi) + qh_off;
                        #pragma unroll
                        for (int i = 0; i < 16; ++i) {
                            const uint8_t ql_b = qlp[i];
                            const uint8_t qh_b = qhp[i];
                            const uint8_t lo_hi = high_nibble ? uint8_t(ql_b >> 4) : uint8_t(ql_b & 0x0F);
                            const int8_t  qsig  = int8_t(lo_hi | (((qh_b >> qh_shift) & 0x3) << 4)) - 32;
                            k_acc += float(h_chunk[i]) * d * float(qsig);
                        }
                    }
                }
                total += weight_k * k_acc;
            }

            total = sycl::reduce_over_group(it.get_sub_group(), total, sycl::plus<float>());
            if (lane == 0) y_out[n] = sycl::half(total);
        });
    });
}

sycl::event moe_decode_down_q6k(sycl::queue& q,
                                const sycl::half* h_in,
                                const void* down_W,
                                const int32_t* topk_idx,
                                const sycl::half* topk_w,
                                sycl::half* y_out,
                                uint32_t H, uint32_t E_ffn, uint32_t K_top,
                                uint64_t expert_stride_bytes, bool soa,
                                const std::vector<sycl::event>& deps) {
    // Heavy decode shape: H=hidden=2048 (E_ffn=512, K_top=8 routed experts).
    // Always SG-coop on the heavy shape; fall back to the original loop for
    // any other shape (none in current model, but keeps the kernel general).
    if (H >= 2048) {
        return soa ? moe_decode_down_q6k_impl</*USE_SG_DEQUANT=*/true, true >(
                         q, h_in, down_W, topk_idx, topk_w, y_out,
                         H, E_ffn, K_top, expert_stride_bytes,
                         "moe_dec_down_q6k", deps)
                   : moe_decode_down_q6k_impl</*USE_SG_DEQUANT=*/true, false>(
                         q, h_in, down_W, topk_idx, topk_w, y_out,
                         H, E_ffn, K_top, expert_stride_bytes,
                         "moe_dec_down_q6k", deps);
    }
    return soa ? moe_decode_down_q6k_impl</*USE_SG_DEQUANT=*/false, true >(
                     q, h_in, down_W, topk_idx, topk_w, y_out,
                     H, E_ffn, K_top, expert_stride_bytes,
                     "moe_dec_down_q6k", deps)
               : moe_decode_down_q6k_impl</*USE_SG_DEQUANT=*/false, false>(
                     q, h_in, down_W, topk_idx, topk_w, y_out,
                     H, E_ffn, K_top, expert_stride_bytes,
                     "moe_dec_down_q6k", deps);
}

// ============================================================================
// Multi-expert prefill stage 1 (gate + up + swiglu)
// ============================================================================
//
// One WG per (expert e, n_chunk in E_ffn). 256 × 32 = 8192 WGs total at
// E_ffn=512, N_PER_WG=16. For each expert with n_tok>0, the WG iterates
// over M_TILE=8 token chunks, gathering the corresponding x rows from the
// global input via `sorted_token_idx`, and runs the same dequant lattice
// as gemm_q4_K to produce gate[M, n_chunk] and up[M, n_chunk] in parallel
// (sharing weight reads across M).  Then swiglu, write h_packed.
// =============================================================================
// P1b-2 (2026-06-10): integer-dot stage 1 — the llama.cpp MoE prefill edge.
// =============================================================================
// Their unitrace shows expert FFNs running as q8_1-activation int-dot kernels
// (mul_mat_vec_q4_K_q8_1_ncols + quantize_row_q8_1) — ~4× cheaper than our
// compute-bound fp32-FMA stage 1 (45% of fp32 roofline at M≈16/expert).
// This kernel is moe_prefill_gate_up_silu_q4k with the SLM activation tile
// and inner loop switched to Q8_1 + dp4a:
//   per (mm, block): 4 dp4a (gate) + 4 dp4a (up) + 4 dp4a (Σq8, shared)
//   vs 32 fp32 FMA + 16 half→float converts.
// x arrives pre-quantized (block_q8_1x stream over the E5 expert-sorted
// x_packed — rows are H-aligned so blocks never straddle rows).
template <bool SOA>
static sycl::event moe_prefill_gate_up_silu_q4k_q8_impl(sycl::queue& q,
                                            const void* xq8_packed,
                                            const void* gate_W, const void* up_W,
                                            const uint32_t* expert_offsets,
                                            sycl::half* h_packed,
                                            uint32_t E, uint32_t H, uint32_t E_ffn,
                                            uint64_t expert_stride_bytes,
                                            const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE   = 16;
    constexpr int N_PER_WG  = 32;
    constexpr int WG_ITEMS  = N_PER_WG * SG_SIZE;
    constexpr int M_TILE    = 16;
    constexpr uint32_t BW   = sizeof(block_q8_1s) / 4;   // 12 words / q8 block

    const uint32_t blocks_per_col = H / 256;             // 8
    const uint32_t q8_per_row     = H / 32;              // 64
    const uint32_t n_steps        = blocks_per_col / 2;  // 4 (needs H % 512 == 0)
    const uint32_t n_chunks = E_ffn / N_PER_WG;
    const uint64_t nb_e = uint64_t(blocks_per_col) * E_ffn;  // blocks per expert
    const auto* X8 = static_cast<const block_q8_1s*>(xq8_packed);

    return ie::ps(q, "moe_pfl_gate_q8", [&](sycl::handler& h) {
        h.depends_on(deps);
        // SLM: M_TILE rows of q8 blocks staged verbatim (48 KiB at H=2048).
        sycl::local_accessor<uint32_t, 1> hq8(
            uint64_t(M_TILE) * q8_per_row * BW, h);

        h.parallel_for(sycl::nd_range<2>({uint64_t(E) * n_chunks, WG_ITEMS},
                                          {1, WG_ITEMS}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t en = uint32_t(it.get_group(0));
            const uint32_t e  = en / n_chunks;
            const uint32_t nc = en % n_chunks;
            const uint32_t lid = uint32_t(it.get_local_id(1));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t n     = nc * N_PER_WG + sg_id;

            const uint32_t off_start = expert_offsets[e];
            const uint32_t off_end   = expert_offsets[e + 1];
            const uint32_t n_tok     = off_end - off_start;
            if (n_tok == 0) return;

            const auto gv = q4k_wview<SOA>::at(gate_W, expert_stride_bytes, e, nb_e);
            const auto uv = q4k_wview<SOA>::at(up_W,   expert_stride_bytes, e, nb_e);

            // Register lattice (stage-2 shape): lane <-> one q8 activation
            // block per step; a step spans two Q4_K blocks of the column.
            const int b_in   = int(lane) >> 3;
            const int sb     = int(lane) & 7;
            const int g      = sb >> 1;
            const int hi_nib = sb & 1;

            for (uint32_t tk_base = 0; tk_base < n_tok; tk_base += M_TILE) {
                const uint32_t M = sycl::min(uint32_t(M_TILE), n_tok - tk_base);

                // Stage M rows of q8 x-blocks (verbatim words, coalesced).
                const uint32_t* src = reinterpret_cast<const uint32_t*>(
                    X8 + (uint64_t(off_start) + tk_base) * q8_per_row);
                for (uint32_t i = lid; i < M * q8_per_row * BW; i += WG_ITEMS)
                    hq8[i] = src[i];
                sycl::group_barrier(it.get_group());

                if (n >= E_ffn) {
                    sycl::group_barrier(it.get_group());
                    continue;
                }

                float g_acc[M_TILE], u_acc[M_TILE];
                #pragma unroll
                for (int mm = 0; mm < M_TILE; ++mm) { g_acc[mm] = 0.f; u_acc[mm] = 0.f; }

                const uint32_t* base =
                    hq8.get_multi_ptr<sycl::access::decorated::no>().get();
                const uint64_t col0 = uint64_t(n) * blocks_per_col;

                for (uint32_t ss = 0; ss < n_steps; ++ss) {
                    const uint64_t bi = col0 + ss * 2 + b_in;

                    // Per-step register weights: lane's 32-elem sub-block of
                    // gate and up (one scale pair + 8 nib-masked words each).
                    const uint8_t* gsc = gv.sc_blk(bi);
                    const uint8_t* usc = uv.sc_blk(bi);
                    uint8_t gs_raw, gm_raw, us_raw, um_raw;
                    if (sb < 4) {
                        gs_raw = gsc[sb]     & 0x3F;
                        gm_raw = gsc[sb + 4] & 0x3F;
                        us_raw = usc[sb]     & 0x3F;
                        um_raw = usc[sb + 4] & 0x3F;
                    } else {
                        gs_raw = (gsc[sb + 4] & 0x0F) | ((gsc[sb - 4] >> 6) << 4);
                        gm_raw = (gsc[sb + 4] >>   4) | ((gsc[sb    ] >> 6) << 4);
                        us_raw = (usc[sb + 4] & 0x0F) | ((usc[sb - 4] >> 6) << 4);
                        um_raw = (usc[sb + 4] >>   4) | ((usc[sb    ] >> 6) << 4);
                    }
                    const float g_d  = dev_fp16_to_fp32(gv.d_bits(bi))    * float(gs_raw);
                    const float g_dm = dev_fp16_to_fp32(gv.dmin_bits(bi)) * float(gm_raw);
                    const float u_d  = dev_fp16_to_fp32(uv.d_bits(bi))    * float(us_raw);
                    const float u_dm = dev_fp16_to_fp32(uv.dmin_bits(bi)) * float(um_raw);

                    const auto gq0 = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(
                        gv.qs_blk(bi) + g * 32);
                    const auto gq1 = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(
                        gv.qs_blk(bi) + g * 32 + 16);
                    const auto uq0 = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(
                        uv.qs_blk(bi) + g * 32);
                    const auto uq1 = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(
                        uv.qs_blk(bi) + g * 32 + 16);
                    uint32_t gnib[8], unib[8];
                    #pragma unroll
                    for (int w = 0; w < 4; ++w) {
                        gnib[w]     = (hi_nib ? (gq0[w] >> 4) : gq0[w]) & 0x0F0F0F0Fu;
                        gnib[w + 4] = (hi_nib ? (gq1[w] >> 4) : gq1[w]) & 0x0F0F0F0Fu;
                        unib[w]     = (hi_nib ? (uq0[w] >> 4) : uq0[w]) & 0x0F0F0F0Fu;
                        unib[w + 4] = (hi_nib ? (uq1[w] >> 4) : uq1[w]) & 0x0F0F0F0Fu;
                    }

                    // Lane's q8 block within the row: step*16 + lane.
                    const uint32_t blk_in_row = ss * 16 + lane;
                    #pragma unroll
                    for (int mm = 0; mm < M_TILE; ++mm) {
                        if (uint32_t(mm) < M) {
                            const uint32_t* blkp =
                                base + (uint32_t(mm) * q8_per_row + blk_in_row) * BW;
                            const auto hdr = *reinterpret_cast<const sycl::vec<float, 4>*>(blkp);
                            const auto q0 = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(blkp + 4);
                            const auto q1 = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(blkp + 8);
                            int32_t idg = 0, idu = 0;
                            #pragma unroll
                            for (int w = 0; w < 4; ++w) {
                                idg = ie::dp4a_us(gnib[w],     int32_t(q0[w]), idg);
                                idu = ie::dp4a_us(unib[w],     int32_t(q0[w]), idu);
                            }
                            #pragma unroll
                            for (int w = 0; w < 4; ++w) {
                                idg = ie::dp4a_us(gnib[w + 4], int32_t(q1[w]), idg);
                                idu = ie::dp4a_us(unib[w + 4], int32_t(q1[w]), idu);
                            }
                            const float d8 = hdr[0];
                            const float sx = hdr[1] + hdr[2];
                            g_acc[mm] += g_d * (d8 * float(idg)) - g_dm * sx;
                            u_acc[mm] += u_d * (d8 * float(idu)) - u_dm * sx;
                        }
                    }
                }

                #pragma unroll
                for (int mm = 0; mm < M_TILE; ++mm) {
                    if (uint32_t(mm) < M) {
                        const float gr = sycl::reduce_over_group(it.get_sub_group(),
                                                                 g_acc[mm], sycl::plus<float>());
                        const float ur = sycl::reduce_over_group(it.get_sub_group(),
                                                                 u_acc[mm], sycl::plus<float>());
                        if (lane == 0) {
                            const float silu_g = gr / (1.0f + sycl::native::exp(-gr));
                            const uint64_t row = uint64_t(off_start + tk_base + mm);
                            h_packed[row * E_ffn + n] = sycl::half(silu_g * ur);
                        }
                    }
                }
                sycl::group_barrier(it.get_group());
            }
        });
    });
}

sycl::event moe_prefill_gate_up_silu_q4k_q8(sycl::queue& q,
                                            const void* xq8_packed,
                                            const void* gate_W, const void* up_W,
                                            const uint32_t* expert_offsets,
                                            sycl::half* h_packed,
                                            uint32_t E, uint32_t H, uint32_t E_ffn,
                                            uint64_t expert_stride_bytes, bool soa,
                                            const std::vector<sycl::event>& deps) {
    return soa ? moe_prefill_gate_up_silu_q4k_q8_impl<true >(q, xq8_packed, gate_W, up_W,
                     expert_offsets, h_packed, E, H, E_ffn, expert_stride_bytes, deps)
               : moe_prefill_gate_up_silu_q4k_q8_impl<false>(q, xq8_packed, gate_W, up_W,
                     expert_offsets, h_packed, E, H, E_ffn, expert_stride_bytes, deps);
}

template <bool SOA>
static sycl::event moe_prefill_gate_up_silu_q4k_impl(sycl::queue& q,
                                         const sycl::half* x_packed,
                                         const void* gate_W, const void* up_W,
                                         const uint32_t* expert_offsets,
                                         sycl::half* h_packed,
                                         uint32_t E, uint32_t H, uint32_t E_ffn,
                                         uint64_t expert_stride_bytes,
                                         const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE   = 16;
    constexpr int N_PER_WG  = 32;
    constexpr int WG_ITEMS  = N_PER_WG * SG_SIZE;
    // 2026-05-05: M_TILE 8 → 16. Doubles SLM A_slm to 64 KiB/WG (still
    // ≤ 192 KiB Xe-core budget). For T=512 prefill with K_top=8 across
    // E=256 experts, average n_tok/expert ≈ 16, so M_TILE=16 covers a
    // typical expert in one outer iteration (was 2 before), halving the
    // weight-read amortization window. Tried 32 — regressed (occupancy).
    constexpr int M_TILE    = 16;

    const uint32_t blocks_per_col = H / 256;
    const uint32_t n_chunks = E_ffn / N_PER_WG;     // 32 for E_ffn=512
    const uint64_t nb_e = uint64_t(blocks_per_col) * E_ffn;  // blocks per expert

    return ie::ps(q, "moe_pfl_gate_q4k", [&](sycl::handler& h) {
        h.depends_on(deps);
        // SLM: M_TILE tokens × H halfs (gathered x). 8 × 2048 × 2 = 32 KiB.
        sycl::local_accessor<sycl::half, 1> A_slm(uint64_t(M_TILE) * H, h);

        h.parallel_for(sycl::nd_range<2>({uint64_t(E) * n_chunks, WG_ITEMS},
                                          {1, WG_ITEMS}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t en = uint32_t(it.get_group(0));
            const uint32_t e  = en / n_chunks;
            const uint32_t nc = en % n_chunks;
            const uint32_t lid = uint32_t(it.get_local_id(1));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t n     = nc * N_PER_WG + sg_id;

            const uint32_t off_start = expert_offsets[e];
            const uint32_t off_end   = expert_offsets[e + 1];
            const uint32_t n_tok     = off_end - off_start;
            if (n_tok == 0) return;

            const auto gv = q4k_wview<SOA>::at(gate_W, expert_stride_bytes, e, nb_e);
            const auto uv = q4k_wview<SOA>::at(up_W,   expert_stride_bytes, e, nb_e);

            // Lane → (sub, half) — same lattice as gemm_q4_K.
            const int sub  = int(lane) >> 1;
            const int half = int(lane) & 1;
            const int g     = sub >> 1;
            const int hi_nib = sub & 1;
            const int qs_off = g * 32 + half * 16;
            const int out_off = g * 64 + hi_nib * 32 + half * 16;

            // Process n_tok in M_TILE-sized blocks.
            for (uint32_t tk_base = 0; tk_base < n_tok; tk_base += M_TILE) {
                const uint32_t M = sycl::min(uint32_t(M_TILE), n_tok - tk_base);

                // E5: rows were pre-gathered into expert-sorted x_packed by
                // moe_gather_rows (once per layer) — contiguous coalesced
                // reads here, no per-WG index indirection.
                const uint64_t row0 = (uint64_t(off_start) + tk_base) * H;
                for (uint64_t i = lid; i < uint64_t(M) * H; i += WG_ITEMS) {
                    A_slm[i] = x_packed[row0 + i];
                }
                sycl::group_barrier(it.get_group());

                if (n >= E_ffn) {
                    sycl::group_barrier(it.get_group());
                    continue;
                }

                float g_acc[M_TILE], u_acc[M_TILE];
                #pragma unroll
                for (int mm = 0; mm < M_TILE; ++mm) { g_acc[mm] = 0.f; u_acc[mm] = 0.f; }

                const uint64_t col0 = uint64_t(n) * blocks_per_col;

                for (uint32_t b = 0; b < blocks_per_col; ++b) {
                    const uint64_t bi = col0 + b;
                    const uint8_t* gsc = gv.sc_blk(bi);
                    const uint8_t* usc = uv.sc_blk(bi);
                    uint8_t gs_raw, gm_raw, us_raw, um_raw;
                    if (sub < 4) {
                        gs_raw = gsc[sub]     & 0x3F;
                        gm_raw = gsc[sub + 4] & 0x3F;
                        us_raw = usc[sub]     & 0x3F;
                        um_raw = usc[sub + 4] & 0x3F;
                    } else {
                        gs_raw = (gsc[sub + 4] & 0x0F) | ((gsc[sub - 4] >> 6) << 4);
                        gm_raw = (gsc[sub + 4] >>   4) | ((gsc[sub    ] >> 6) << 4);
                        us_raw = (usc[sub + 4] & 0x0F) | ((usc[sub - 4] >> 6) << 4);
                        um_raw = (usc[sub + 4] >>   4) | ((usc[sub    ] >> 6) << 4);
                    }
                    const float g_d  = dev_fp16_to_fp32(gv.d_bits(bi))    * float(gs_raw);
                    const float g_dm = dev_fp16_to_fp32(gv.dmin_bits(bi)) * float(gm_raw);
                    const float u_d  = dev_fp16_to_fp32(uv.d_bits(bi))    * float(us_raw);
                    const float u_dm = dev_fp16_to_fp32(uv.dmin_bits(bi)) * float(um_raw);

                    // E2 (docs/prefill_attack_plan_2026-06-09.md): one 16 B
                    // vector load per matrix instead of 16 byte loads
                    // (AoS block stride 144 B and SoA stream stride 128 B both
                    // keep the qs base 16 B-aligned), dequant into registers
                    // once, then vec8 SLM reads of A.
                    // Per-accumulator add order is unchanged → bit-identical.
                    const auto gqv = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(gv.qs_blk(bi) + qs_off);
                    const auto uqv = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(uv.qs_blk(bi) + qs_off);
                    float gw[16], uw[16];
                    #pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        const uint8_t gqb = uint8_t(gqv[i >> 2] >> ((i & 3) * 8));
                        const uint8_t uqb = uint8_t(uqv[i >> 2] >> ((i & 3) * 8));
                        const int gq4 = hi_nib ? (gqb >> 4) : (gqb & 0x0F);
                        const int uq4 = hi_nib ? (uqb >> 4) : (uqb & 0x0F);
                        gw[i] = g_d * float(gq4) - g_dm;
                        uw[i] = u_d * float(uq4) - u_dm;
                    }
                    const sycl::half* a_ptr =
                        A_slm.get_multi_ptr<sycl::access::decorated::no>().get();
                    #pragma unroll
                    for (int mm = 0; mm < M_TILE; ++mm) {
                        if (uint32_t(mm) < M) {
                            const auto* av = reinterpret_cast<const sycl::vec<sycl::half, 8>*>(
                                a_ptr + uint64_t(mm) * H + b * 256 + out_off);
                            const auto a0 = av[0];
                            const auto a1 = av[1];
                            #pragma unroll
                            for (int i = 0; i < 8; ++i) {
                                g_acc[mm] += float(a0[i]) * gw[i];
                                u_acc[mm] += float(a0[i]) * uw[i];
                            }
                            #pragma unroll
                            for (int i = 0; i < 8; ++i) {
                                g_acc[mm] += float(a1[i]) * gw[i + 8];
                                u_acc[mm] += float(a1[i]) * uw[i + 8];
                            }
                        }
                    }
                }

                // SG-reduce M rows; lane 0 writes silu(gate)*up to h_packed.
                #pragma unroll
                for (int mm = 0; mm < M_TILE; ++mm) {
                    if (uint32_t(mm) < M) {
                        const float gr = sycl::reduce_over_group(it.get_sub_group(),
                                                                 g_acc[mm], sycl::plus<float>());
                        const float ur = sycl::reduce_over_group(it.get_sub_group(),
                                                                 u_acc[mm], sycl::plus<float>());
                        if (lane == 0) {
                            const float silu_g = gr / (1.0f + sycl::native::exp(-gr));
                            const uint64_t row = uint64_t(off_start + tk_base + mm);
                            h_packed[row * E_ffn + n] = sycl::half(silu_g * ur);
                        }
                    }
                }
                sycl::group_barrier(it.get_group());  // before next M-tile gather
            }
        });
    });
}

sycl::event moe_prefill_gate_up_silu_q4k(sycl::queue& q,
                                         const sycl::half* x_packed,
                                         const void* gate_W, const void* up_W,
                                         const uint32_t* expert_offsets,
                                         sycl::half* h_packed,
                                         uint32_t E, uint32_t H, uint32_t E_ffn,
                                         uint64_t expert_stride_bytes, bool soa,
                                         const std::vector<sycl::event>& deps) {
    return soa ? moe_prefill_gate_up_silu_q4k_impl<true >(q, x_packed, gate_W, up_W,
                     expert_offsets, h_packed, E, H, E_ffn, expert_stride_bytes, deps)
               : moe_prefill_gate_up_silu_q4k_impl<false>(q, x_packed, gate_W, up_W,
                     expert_offsets, h_packed, E, H, E_ffn, expert_stride_bytes, deps);
}

// ============================================================================
// Multi-expert prefill stage 2 (down + scatter-add)
// ============================================================================
//
// One WG per (expert e, n_chunk in H).  Each WG processes its expert's full
// n_tok rows: dequant down weights (Q4_K or Q6_K), accumulate to a partial
// per-(token,expert) output, multiply by sorted_token_w, atomically add to
// y_out[token].  Race: multiple experts contributing to the same token's row
// — handled by atomic_ref relaxed adds.
namespace {
template <typename Block>
inline float dequant_q4k_dot_chunk(const Block& blk, int sub, int hi_nib,
                                   int qs_off, const sycl::half* a_chunk) {
    uint8_t s_raw, m_raw;
    if (sub < 4) {
        s_raw = blk.scales[sub]     & 0x3F;
        m_raw = blk.scales[sub + 4] & 0x3F;
    } else {
        s_raw = (blk.scales[sub + 4] & 0x0F) | ((blk.scales[sub - 4] >> 6) << 4);
        m_raw = (blk.scales[sub + 4] >>   4) | ((blk.scales[sub    ] >> 6) << 4);
    }
    const float d  = dev_fp16_to_fp32(blk.d)    * float(s_raw);
    const float dm = dev_fp16_to_fp32(blk.dmin) * float(m_raw);
    float acc = 0.f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const uint8_t qb = blk.qs[qs_off + i];
        const int q4 = hi_nib ? (qb >> 4) : (qb & 0x0F);
        acc += float(a_chunk[i]) * (d * float(q4) - dm);
    }
    return acc;
}
}  // namespace

sycl::event moe_prefill_down_q4k(sycl::queue& q,
                                 const sycl::half* h_packed,
                                 const void* down_W,
                                 const uint32_t* expert_offsets,
                                 const int32_t* sorted_token_idx,
                                 const sycl::half* sorted_token_w,
                                 float* y_fp32,
                                 uint32_t E, uint32_t H, uint32_t E_ffn,
                                 uint64_t expert_stride_bytes,
                                 const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE   = 16;
    constexpr int N_PER_WG  = 16;
    constexpr int WG_ITEMS  = N_PER_WG * SG_SIZE;
    const uint32_t blocks_per_col = E_ffn / 256;
    const uint32_t n_chunks = (H + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "moe_pfl_down_q4k", [&](sycl::handler& h) {
        h.depends_on(deps);

        h.parallel_for(sycl::nd_range<2>({uint64_t(E) * n_chunks, WG_ITEMS},
                                          {1, WG_ITEMS}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t en = uint32_t(it.get_group(0));
            const uint32_t e  = en / n_chunks;
            const uint32_t nc = en % n_chunks;
            const uint32_t lid = uint32_t(it.get_local_id(1));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t n     = nc * N_PER_WG + sg_id;

            const uint32_t off_start = expert_offsets[e];
            const uint32_t off_end   = expert_offsets[e + 1];
            const uint32_t n_tok     = off_end - off_start;
            if (n_tok == 0 || n >= H) return;

            const auto* down_e = reinterpret_cast<const block_q4_K*>(
                reinterpret_cast<const uint8_t*>(down_W) + uint64_t(e) * expert_stride_bytes);
            const block_q4_K* col_blocks = &down_e[uint64_t(n) * blocks_per_col];

            const int sub  = int(lane) >> 1;
            const int hi_nib = (sub >> 0) & 1;  // for stage-2 unused but keeps symmetry
            const int g     = sub >> 1;
            const int half_l = int(lane) & 1;
            const int qs_off = g * 32 + half_l * 16;
            const int out_off = g * 64 + hi_nib * 32 + half_l * 16;

            for (uint32_t tk = 0; tk < n_tok; ++tk) {
                const uint64_t row = uint64_t(off_start + tk);
                const sycl::half* h_row = h_packed + row * E_ffn;
                float acc = 0.f;
                for (uint32_t b = 0; b < blocks_per_col; ++b) {
                    const sycl::half* a_chunk = h_row + b * 256 + out_off;
                    acc += dequant_q4k_dot_chunk(col_blocks[b], sub, hi_nib, qs_off, a_chunk);
                }
                acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
                if (lane == 0) {
                    const int32_t tok = sorted_token_idx[row];
                    const float w     = float(sorted_token_w[row]);
                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        a_y(y_fp32[uint64_t(tok) * H + n]);
                    a_y.fetch_add(acc * w);
                }
            }
        });
    });
}

sycl::event moe_prefill_down_q6k(sycl::queue& q,
                                 const sycl::half* h_packed,
                                 const void* down_W,
                                 const uint32_t* expert_offsets,
                                 const int32_t* sorted_token_idx,
                                 const sycl::half* sorted_token_w,
                                 float* y_fp32,
                                 uint32_t E, uint32_t H, uint32_t E_ffn,
                                 uint64_t expert_stride_bytes,
                                 const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE   = 16;
    constexpr int N_PER_WG  = 16;
    constexpr int WG_ITEMS  = N_PER_WG * SG_SIZE;
    const uint32_t blocks_per_col = E_ffn / 256;
    const uint32_t n_chunks = (H + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "moe_pfl_down_q6k", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<2>({uint64_t(E) * n_chunks, WG_ITEMS},
                                          {1, WG_ITEMS}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t en = uint32_t(it.get_group(0));
            const uint32_t e  = en / n_chunks;
            const uint32_t nc = en % n_chunks;
            const uint32_t lid = uint32_t(it.get_local_id(1));
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t n     = nc * N_PER_WG + sg_id;

            const uint32_t off_start = expert_offsets[e];
            const uint32_t off_end   = expert_offsets[e + 1];
            const uint32_t n_tok     = off_end - off_start;
            if (n_tok == 0 || n >= H) return;

            const auto* down_e = reinterpret_cast<const block_q6_K*>(
                reinterpret_cast<const uint8_t*>(down_W) + uint64_t(e) * expert_stride_bytes);
            const block_q6_K* col_blocks = &down_e[uint64_t(n) * blocks_per_col];

            const int half_q   = int(lane) >> 3;
            const int sub      = (int(lane) >> 1) & 0x3;
            const int l_half   = int(lane) & 0x1;
            const int l_start  = l_half * 16;
            const int ql_off   = half_q * 64 + (sub & 1) * 32 + l_start;
            const int qh_off   = half_q * 32 + l_start;
            const int scale_off = half_q * 8 + sub * 2 + l_half;
            const int qh_shift = sub * 2;
            const bool high_nibble = (sub & 2) != 0;
            const int out_off  = half_q * 128 + sub * 32 + l_start;

            for (uint32_t tk = 0; tk < n_tok; ++tk) {
                const uint64_t row = uint64_t(off_start + tk);
                const sycl::half* h_row = h_packed + row * E_ffn;
                float acc = 0.f;
                for (uint32_t b = 0; b < blocks_per_col; ++b) {
                    const block_q6_K& blk = col_blocks[b];
                    const float d = dev_fp16_to_fp32(blk.d) * float(blk.scales[scale_off]);
                    const sycl::half* a_chunk = h_row + b * 256 + out_off;
                    #pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        const uint8_t ql_b = blk.ql[ql_off + i];
                        const uint8_t qh_b = blk.qh[qh_off + i];
                        const uint8_t lo_hi = high_nibble ? uint8_t(ql_b >> 4) : uint8_t(ql_b & 0x0F);
                        const int8_t  qsig  = int8_t(lo_hi | (((qh_b >> qh_shift) & 0x3) << 4)) - 32;
                        acc += float(a_chunk[i]) * d * float(qsig);
                    }
                }
                acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
                if (lane == 0) {
                    const int32_t tok = sorted_token_idx[row];
                    const float w     = float(sorted_token_w[row]);
                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        a_y(y_fp32[uint64_t(tok) * H + n]);
                    a_y.fetch_add(acc * w);
                }
            }
        });
    });
}

// =========================================================================
// Atomics-free multi-expert stage 2 + reduce
// =========================================================================
sycl::event moe_prefill_down_packed_q4k(sycl::queue& q,
                                        const sycl::half* h_packed,
                                        const void* down_W,
                                        const uint32_t* expert_offsets,
                                        sycl::half* out_packed,
                                        uint32_t E, uint32_t H, uint32_t E_ffn,
                                        uint64_t expert_stride_bytes,
                                        const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE   = 16;
    constexpr int N_PER_WG  = 32;
    constexpr int WG_ITEMS  = N_PER_WG * SG_SIZE;
    const uint32_t blocks_per_col = E_ffn / 256;
    const uint32_t n_chunks = (H + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "moe_pfl_down_pk4k", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<2>({uint64_t(E) * n_chunks, WG_ITEMS},
                                          {1, WG_ITEMS}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t en = uint32_t(it.get_group(0));
            const uint32_t e  = en / n_chunks;
            const uint32_t nc = en % n_chunks;
            const uint32_t lid = uint32_t(it.get_local_id(1));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t n     = nc * N_PER_WG + sg_id;

            const uint32_t off_start = expert_offsets[e];
            const uint32_t off_end   = expert_offsets[e + 1];
            const uint32_t n_tok     = off_end - off_start;
            if (n_tok == 0 || n >= H) return;

            const auto* down_e = reinterpret_cast<const block_q4_K*>(
                reinterpret_cast<const uint8_t*>(down_W) + uint64_t(e) * expert_stride_bytes);
            const block_q4_K* col_blocks = &down_e[uint64_t(n) * blocks_per_col];

            const int sub  = int(lane) >> 1;
            const int hi_nib = (sub >> 0) & 1;
            const int g     = sub >> 1;
            const int half_l = int(lane) & 1;
            const int qs_off = g * 32 + half_l * 16;
            const int out_off = g * 64 + hi_nib * 32 + half_l * 16;

            for (uint32_t tk = 0; tk < n_tok; ++tk) {
                const uint64_t row = uint64_t(off_start + tk);
                const sycl::half* h_row = h_packed + row * E_ffn;
                float acc = 0.f;
                for (uint32_t b = 0; b < blocks_per_col; ++b) {
                    const sycl::half* a_chunk = h_row + b * 256 + out_off;
                    acc += dequant_q4k_dot_chunk(col_blocks[b], sub, hi_nib, qs_off, a_chunk);
                }
                acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
                if (lane == 0) {
                    out_packed[row * H + n] = sycl::half(acc);
                }
            }
        });
    });
}

sycl::event moe_prefill_down_packed_q6k(sycl::queue& q,
                                        const sycl::half* h_packed,
                                        const void* down_W,
                                        const uint32_t* expert_offsets,
                                        sycl::half* out_packed,
                                        uint32_t E, uint32_t H, uint32_t E_ffn,
                                        uint64_t expert_stride_bytes,
                                        const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE   = 16;
    constexpr int N_PER_WG  = 32;
    constexpr int WG_ITEMS  = N_PER_WG * SG_SIZE;
    const uint32_t blocks_per_col = E_ffn / 256;
    const uint32_t n_chunks = (H + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "moe_pfl_down_pk6k", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<2>({uint64_t(E) * n_chunks, WG_ITEMS},
                                          {1, WG_ITEMS}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t en = uint32_t(it.get_group(0));
            const uint32_t e  = en / n_chunks;
            const uint32_t nc = en % n_chunks;
            const uint32_t lid = uint32_t(it.get_local_id(1));
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t n     = nc * N_PER_WG + sg_id;

            const uint32_t off_start = expert_offsets[e];
            const uint32_t off_end   = expert_offsets[e + 1];
            const uint32_t n_tok     = off_end - off_start;
            if (n_tok == 0 || n >= H) return;

            const auto* down_e = reinterpret_cast<const block_q6_K*>(
                reinterpret_cast<const uint8_t*>(down_W) + uint64_t(e) * expert_stride_bytes);
            const block_q6_K* col_blocks = &down_e[uint64_t(n) * blocks_per_col];

            const int half_q   = int(lane) >> 3;
            const int sub      = (int(lane) >> 1) & 0x3;
            const int l_half   = int(lane) & 0x1;
            const int l_start  = l_half * 16;
            const int ql_off   = half_q * 64 + (sub & 1) * 32 + l_start;
            const int qh_off   = half_q * 32 + l_start;
            const int scale_off = half_q * 8 + sub * 2 + l_half;
            const int qh_shift = sub * 2;
            const bool high_nibble = (sub & 2) != 0;
            const int out_off  = half_q * 128 + sub * 32 + l_start;

            for (uint32_t tk = 0; tk < n_tok; ++tk) {
                const uint64_t row = uint64_t(off_start + tk);
                const sycl::half* h_row = h_packed + row * E_ffn;
                float acc = 0.f;
                for (uint32_t b = 0; b < blocks_per_col; ++b) {
                    const block_q6_K& blk = col_blocks[b];
                    const float d = dev_fp16_to_fp32(blk.d) * float(blk.scales[scale_off]);
                    const sycl::half* a_chunk = h_row + b * 256 + out_off;
                    #pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        const uint8_t ql_b = blk.ql[ql_off + i];
                        const uint8_t qh_b = blk.qh[qh_off + i];
                        const uint8_t lo_hi = high_nibble ? uint8_t(ql_b >> 4) : uint8_t(ql_b & 0x0F);
                        const int8_t  qsig  = int8_t(lo_hi | (((qh_b >> qh_shift) & 0x3) << 4)) - 32;
                        acc += float(a_chunk[i]) * d * float(qsig);
                    }
                }
                acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
                if (lane == 0) {
                    out_packed[row * H + n] = sycl::half(acc);
                }
            }
        });
    });
}

// =========================================================================
// v2: M_TILE=8 SLM-amortized stage-2 down kernels.
// =========================================================================
// The v1 kernels above re-stream the expert's full Q4_K/Q6_K column from
// DRAM once per token in their `for tk` outer loop.  IGC asm dump (Xe2,
// 128 GRF) showed v1 had 21 load.ugm and zero load.slm per kernel — pure
// memory-bound with no amortization, against stage-1 gate_up_silu's 49
// load.ugm + 128 load.slm with M_TILE=8 amortization.  At T=692 with
// 256 experts and K_top=8, n_tok/expert ≈ 22 — every Q-block is read
// ~22× per launch where it could be read once.
//
// v2 mirrors moe_prefill_gate_up_silu_q4k's exact SLM-cooperative shape:
//   - SLM: M_TILE × E_ffn halfs of h_packed (8 × 512 × 2 = 8 KiB)
//   - Outer loop tiles n_tok in M_TILE-sized chunks
//   - Inner loop walks blocks_per_col ONCE per tile, dequant per block,
//     multiply against M rows from SLM
//   - One block-load per Q-block per tile (vs n_tok block-loads in v1)
template <bool SOA>
static sycl::event moe_prefill_down_packed_q4k_v2_impl(sycl::queue& q,
                                           const sycl::half* h_packed,
                                           const void* down_W,
                                           const uint32_t* expert_offsets,
                                           sycl::half* out_packed,
                                           uint32_t E, uint32_t H, uint32_t E_ffn,
                                           uint64_t expert_stride_bytes,
                                           const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE   = 16;
    constexpr int N_PER_WG  = 32;
    constexpr int WG_ITEMS  = N_PER_WG * SG_SIZE;
    constexpr int M_TILE    = 16;   // v1.2-C: 8 -> 16, doubles weight amortization (A_slm 24 KiB)
    const uint32_t blocks_per_col = E_ffn / 256;
    const uint32_t n_chunks = (H + N_PER_WG - 1) / N_PER_WG;
    const uint64_t nb_e = uint64_t(blocks_per_col) * H;  // blocks per expert

    return ie::ps(q, "moe_pfl_down_pk4k_v2", [&](sycl::handler& h) {
        h.depends_on(deps);
        // SLM: M_TILE packed-rows × E_ffn halfs.  At E_ffn=512 → 8 KiB/WG.
        sycl::local_accessor<sycl::half, 1> A_slm(uint64_t(M_TILE) * E_ffn, h);

        h.parallel_for(sycl::nd_range<2>({uint64_t(E) * n_chunks, WG_ITEMS},
                                          {1, WG_ITEMS}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t en = uint32_t(it.get_group(0));
            const uint32_t e  = en / n_chunks;
            const uint32_t nc = en % n_chunks;
            const uint32_t lid = uint32_t(it.get_local_id(1));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t n     = nc * N_PER_WG + sg_id;

            const uint32_t off_start = expert_offsets[e];
            const uint32_t off_end   = expert_offsets[e + 1];
            const uint32_t n_tok     = off_end - off_start;
            if (n_tok == 0) return;

            const auto dv = q4k_wview<SOA>::at(down_W, expert_stride_bytes, e, nb_e);

            // Lane → (sub, half_l) — same lattice as v1 down_packed_q4k.
            const int sub  = int(lane) >> 1;
            const int hi_nib = sub & 1;
            const int g     = sub >> 1;
            const int half_l = int(lane) & 1;
            const int qs_off = g * 32 + half_l * 16;
            const int out_off = g * 64 + hi_nib * 32 + half_l * 16;

            for (uint32_t tk_base = 0; tk_base < n_tok; tk_base += M_TILE) {
                const uint32_t M = sycl::min(uint32_t(M_TILE), n_tok - tk_base);

                // Cooperative SLM load of M packed-rows from h_packed.
                // Packed rows are contiguous: row r at h_packed[r * E_ffn].
                const uint64_t row_base = uint64_t(off_start + tk_base);
                for (uint64_t i = lid; i < uint64_t(M) * E_ffn; i += WG_ITEMS) {
                    const uint32_t mm = uint32_t(i / E_ffn);
                    const uint32_t d  = uint32_t(i % E_ffn);
                    A_slm[i] = h_packed[(row_base + mm) * E_ffn + d];
                }
                sycl::group_barrier(it.get_group());

                if (n >= H) {
                    sycl::group_barrier(it.get_group());
                    continue;
                }

                float acc[M_TILE];
                #pragma unroll
                for (int mm = 0; mm < M_TILE; ++mm) acc[mm] = 0.f;

                const uint64_t col0 = uint64_t(n) * blocks_per_col;

                for (uint32_t b = 0; b < blocks_per_col; ++b) {
                    const uint64_t bi = col0 + b;
                    const uint8_t* sc = dv.sc_blk(bi);
                    uint8_t s_raw, m_raw;
                    if (sub < 4) {
                        s_raw = sc[sub]     & 0x3F;
                        m_raw = sc[sub + 4] & 0x3F;
                    } else {
                        s_raw = (sc[sub + 4] & 0x0F) | ((sc[sub - 4] >> 6) << 4);
                        m_raw = (sc[sub + 4] >>   4) | ((sc[sub    ] >> 6) << 4);
                    }
                    const float d  = dev_fp16_to_fp32(dv.d_bits(bi))    * float(s_raw);
                    const float dm = dev_fp16_to_fp32(dv.dmin_bits(bi)) * float(m_raw);

                    // E2: vector weight load + register dequant + vec8 SLM
                    // reads (see gate_up_silu above).  Bit-identical math.
                    const auto qv = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(dv.qs_blk(bi) + qs_off);
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
                        if (uint32_t(mm) < M) {
                            const auto* av = reinterpret_cast<const sycl::vec<sycl::half, 8>*>(
                                a_ptr + uint64_t(mm) * E_ffn + b * 256 + out_off);
                            const auto a0 = av[0];
                            const auto a1 = av[1];
                            #pragma unroll
                            for (int i = 0; i < 8; ++i) acc[mm] += float(a0[i]) * wv[i];
                            #pragma unroll
                            for (int i = 0; i < 8; ++i) acc[mm] += float(a1[i]) * wv[i + 8];
                        }
                    }
                }

                // SG-reduce M rows; lane 0 writes to out_packed.
                #pragma unroll
                for (int mm = 0; mm < M_TILE; ++mm) {
                    if (uint32_t(mm) < M) {
                        const float r = sycl::reduce_over_group(it.get_sub_group(),
                                                                acc[mm], sycl::plus<float>());
                        if (lane == 0) {
                            out_packed[(row_base + mm) * H + n] = sycl::half(r);
                        }
                    }
                }
                sycl::group_barrier(it.get_group());
            }
        });
    });
}

sycl::event moe_prefill_down_packed_q4k_v2(sycl::queue& q,
                                           const sycl::half* h_packed,
                                           const void* down_W,
                                           const uint32_t* expert_offsets,
                                           sycl::half* out_packed,
                                           uint32_t E, uint32_t H, uint32_t E_ffn,
                                           uint64_t expert_stride_bytes, bool soa,
                                           const std::vector<sycl::event>& deps) {
    return soa ? moe_prefill_down_packed_q4k_v2_impl<true >(q, h_packed, down_W,
                     expert_offsets, out_packed, E, H, E_ffn, expert_stride_bytes, deps)
               : moe_prefill_down_packed_q4k_v2_impl<false>(q, h_packed, down_W,
                     expert_offsets, out_packed, E, H, E_ffn, expert_stride_bytes, deps);
}

template <bool SOA>
static sycl::event moe_prefill_down_packed_q6k_v2_impl(sycl::queue& q,
                                           const sycl::half* h_packed,
                                           const void* down_W,
                                           const uint32_t* expert_offsets,
                                           sycl::half* out_packed,
                                           uint32_t E, uint32_t H, uint32_t E_ffn,
                                           uint64_t expert_stride_bytes,
                                           const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE   = 16;
    constexpr int N_PER_WG  = 32;
    constexpr int WG_ITEMS  = N_PER_WG * SG_SIZE;
    constexpr int M_TILE    = 16;   // v1.2-C: 8 -> 16 (A_slm 24 KiB + 20 KiB W slab = 44 KiB/WG)
    const uint32_t blocks_per_col = E_ffn / 256;
    const uint32_t n_chunks = (H + N_PER_WG - 1) / N_PER_WG;
    const uint64_t nb_e = uint64_t(blocks_per_col) * H;  // blocks per expert

    return ie::ps(q, "moe_pfl_down_pk6k_v2", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<sycl::half, 1> A_slm(uint64_t(M_TILE) * E_ffn, h);
        // E2c: the WG's 32 columns × blocks_per_col Q6_K blocks are one
        // contiguous slab (32 × bpc × 210 B; 20 160 B at E_ffn=768 — always
        // 16 B-divisible).  Block stride 210 B is only 2 B-aligned, which
        // made direct vector loads impossible; staging the slab into SLM
        // with cooperative 16 B loads once per WG fixes that AND stops
        // re-reading weights from global per M-tile.
        sycl::local_accessor<uint8_t, 1> W_slm(
            uint64_t(N_PER_WG) * blocks_per_col * sizeof(block_q6_K), h);

        h.parallel_for(sycl::nd_range<2>({uint64_t(E) * n_chunks, WG_ITEMS},
                                          {1, WG_ITEMS}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t en = uint32_t(it.get_group(0));
            const uint32_t e  = en / n_chunks;
            const uint32_t nc = en % n_chunks;
            const uint32_t lid = uint32_t(it.get_local_id(1));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t n     = nc * N_PER_WG + sg_id;

            const uint32_t off_start = expert_offsets[e];
            const uint32_t off_end   = expert_offsets[e + 1];
            const uint32_t n_tok     = off_end - off_start;
            if (n_tok == 0) return;

            const auto dv = q6k_wview<SOA>::at(down_W, expert_stride_bytes, e, nb_e);

            // E2c: cooperative SLM staging of the WG's weight slab.
            // SB = slab capacity in blocks (full N_PER_WG chunk); the SLM
            // image keeps the source layout: one contiguous AoS slab, or the
            // four SoA streams packed back-to-back at capacity offsets.
            constexpr uint32_t SB_CAP = uint32_t(N_PER_WG);  // cols capacity
            const uint32_t SB = SB_CAP * blocks_per_col;     // slab blocks
            uint8_t* slm_b = W_slm.get_multi_ptr<sycl::access::decorated::no>().get();
            {
                // Clamp to the expert's real columns so a partial last chunk
                // (H % N_PER_WG != 0) never reads past the tensor.
                const uint32_t cols = sycl::min(uint32_t(N_PER_WG),
                                                H - nc * N_PER_WG);
                const uint32_t slab_blocks = cols * blocks_per_col;
                const uint64_t blk0 = uint64_t(nc) * N_PER_WG * blocks_per_col;
                // Cooperative copy: 16 B vector main + byte tail.  All SoA
                // stream offsets and the AoS slab offset are 16 B-aligned for
                // the production shapes; the tail loop covers any remainder.
                auto coop_copy = [&](const uint8_t* src, uint8_t* dst, uint32_t n_bytes) {
                    const auto* s4 = reinterpret_cast<const sycl::vec<uint32_t, 4>*>(src);
                    auto* d4 = reinterpret_cast<sycl::vec<uint32_t, 4>*>(dst);
                    const uint32_t n_vec = n_bytes / 16;
                    for (uint32_t i = lid; i < n_vec; i += WG_ITEMS) d4[i] = s4[i];
                    for (uint32_t i = n_vec * 16 + lid; i < n_bytes; i += WG_ITEMS)
                        dst[i] = src[i];
                };
                if constexpr (SOA) {
                    coop_copy(dv.ql_blk(blk0), slm_b,            slab_blocks * 128u);
                    coop_copy(dv.qh_blk(blk0), slm_b + SB * 128u, slab_blocks * 64u);
                    coop_copy(reinterpret_cast<const uint8_t*>(dv.sc_blk(blk0)),
                              slm_b + SB * 192u, slab_blocks * 16u);
                    coop_copy(dv.d + blk0 * 2u, slm_b + SB * 208u, slab_blocks * 2u);
                } else {
                    // AoS: the slab is one contiguous run of block_q6_K
                    // (dv.ql is the expert base in the AoS instantiation).
                    coop_copy(dv.ql + blk0 * sizeof(block_q6_K), slm_b,
                              slab_blocks * uint32_t(sizeof(block_q6_K)));
                }
            }
            sycl::group_barrier(it.get_group());

            // SLM-side view mirroring the staged layout (block index local
            // to the slab).
            const q6k_wview<SOA> wv = SOA
                ? q6k_wview<SOA>{ slm_b, slm_b + SB * 128u, slm_b + SB * 192u,
                                  slm_b + SB * 208u }
                : q6k_wview<SOA>{ slm_b, slm_b + 128u, slm_b + 192u, slm_b + 208u };

            // Lane → Q6_K lattice — same as v1 down_packed_q6k.
            const int half_q   = int(lane) >> 3;
            const int sub      = (int(lane) >> 1) & 0x3;
            const int l_half   = int(lane) & 0x1;
            const int l_start  = l_half * 16;
            const int ql_off   = half_q * 64 + (sub & 1) * 32 + l_start;
            const int qh_off   = half_q * 32 + l_start;
            const int scale_off = half_q * 8 + sub * 2 + l_half;
            const int qh_shift = sub * 2;
            const bool high_nibble = (sub & 2) != 0;
            const int out_off  = half_q * 128 + sub * 32 + l_start;

            for (uint32_t tk_base = 0; tk_base < n_tok; tk_base += M_TILE) {
                const uint32_t M = sycl::min(uint32_t(M_TILE), n_tok - tk_base);

                const uint64_t row_base = uint64_t(off_start + tk_base);
                for (uint64_t i = lid; i < uint64_t(M) * E_ffn; i += WG_ITEMS) {
                    const uint32_t mm = uint32_t(i / E_ffn);
                    const uint32_t d  = uint32_t(i % E_ffn);
                    A_slm[i] = h_packed[(row_base + mm) * E_ffn + d];
                }
                sycl::group_barrier(it.get_group());

                if (n >= H) {
                    sycl::group_barrier(it.get_group());
                    continue;
                }

                float acc[M_TILE];
                #pragma unroll
                for (int mm = 0; mm < M_TILE; ++mm) acc[mm] = 0.f;

                // E2c: read this column's blocks from the SLM-staged slab.
                const uint32_t blk_local0 = (n - nc * N_PER_WG) * blocks_per_col;

                for (uint32_t b = 0; b < blocks_per_col; ++b) {
                    const uint32_t bi = blk_local0 + b;
                    const float d = dev_fp16_to_fp32(wv.d_bits(bi)) *
                                    float(wv.sc_blk(bi)[scale_off]);

                    // E2: AoS block_q6_K is 210 B (only 2 B-aligned), so weight
                    // bytes load as uint16 pairs (8 loads instead of 16+16),
                    // dequant to registers, then vec8 SLM reads of A.
                    // Bit-identical math.  (SoA streams are better-aligned but
                    // keep the same read shape — addresses only.)
                    const auto* ql16 = reinterpret_cast<const uint16_t*>(wv.ql_blk(bi) + ql_off);
                    const auto* qh16 = reinterpret_cast<const uint16_t*>(wv.qh_blk(bi) + qh_off);
                    float wv[16];
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const uint16_t qlp = ql16[i];
                        const uint16_t qhp = qh16[i];
                        #pragma unroll
                        for (int p = 0; p < 2; ++p) {
                            const uint8_t ql_b = uint8_t(qlp >> (p * 8));
                            const uint8_t qh_b = uint8_t(qhp >> (p * 8));
                            const uint8_t lo_hi = high_nibble ? uint8_t(ql_b >> 4) : uint8_t(ql_b & 0x0F);
                            const int8_t  qsig  = int8_t(lo_hi | (((qh_b >> qh_shift) & 0x3) << 4)) - 32;
                            wv[i * 2 + p] = d * float(qsig);
                        }
                    }
                    const sycl::half* a_ptr =
                        A_slm.get_multi_ptr<sycl::access::decorated::no>().get();
                    #pragma unroll
                    for (int mm = 0; mm < M_TILE; ++mm) {
                        if (uint32_t(mm) < M) {
                            const auto* av = reinterpret_cast<const sycl::vec<sycl::half, 8>*>(
                                a_ptr + uint64_t(mm) * E_ffn + b * 256 + out_off);
                            const auto a0 = av[0];
                            const auto a1 = av[1];
                            #pragma unroll
                            for (int i = 0; i < 8; ++i) acc[mm] += float(a0[i]) * wv[i];
                            #pragma unroll
                            for (int i = 0; i < 8; ++i) acc[mm] += float(a1[i]) * wv[i + 8];
                        }
                    }
                }

                #pragma unroll
                for (int mm = 0; mm < M_TILE; ++mm) {
                    if (uint32_t(mm) < M) {
                        const float r = sycl::reduce_over_group(it.get_sub_group(),
                                                                acc[mm], sycl::plus<float>());
                        if (lane == 0) {
                            out_packed[(row_base + mm) * H + n] = sycl::half(r);
                        }
                    }
                }
                sycl::group_barrier(it.get_group());
            }
        });
    });
}

sycl::event moe_prefill_down_packed_q6k_v2(sycl::queue& q,
                                           const sycl::half* h_packed,
                                           const void* down_W,
                                           const uint32_t* expert_offsets,
                                           sycl::half* out_packed,
                                           uint32_t E, uint32_t H, uint32_t E_ffn,
                                           uint64_t expert_stride_bytes, bool soa,
                                           const std::vector<sycl::event>& deps) {
    return soa ? moe_prefill_down_packed_q6k_v2_impl<true >(q, h_packed, down_W,
                     expert_offsets, out_packed, E, H, E_ffn, expert_stride_bytes, deps)
               : moe_prefill_down_packed_q6k_v2_impl<false>(q, h_packed, down_W,
                     expert_offsets, out_packed, E, H, E_ffn, expert_stride_bytes, deps);
}

// =========================================================================
// Prefill-crown (2026-06-10): integer-dot stage-2 down kernels (IE_MOE_Q8).
// =========================================================================
// The fp16 v2 kernels above run at ~half of stage-1's ALU rate: a short
// K loop (2 blocks/column) leaves the per-tile overhead un-amortized, and
// every element costs an fp16->fp32 convert + FMA.  Here the SG covers the
// FULL K = E_ffn = 512 reduction at once: 16 lanes x 32 elements, one q8
// activation block per lane.  Expert weights for the SG's column are
// register-resident — loaded ONCE per (expert, column) and reused across
// every routed token row.  Per (column, row): 8 dp4a + a handful of fp ops
// against the fp16 path's 64 converts + 64 FMA.
//
// Activations arrive as a block_q8_1s stream over h_packed (quantize_q8_1s
// once per layer).  The split half-sums s0/s1 give exact corrections for
// both quant types: Q4_K needs the 32-block sum (s0+s1) for its min term,
// Q6_K needs per-16 sums for its signed-offset term (-32 per element)
// because its scales are per-16.
sycl::event moe_prefill_down_packed_q4k_q8(sycl::queue& q,
                                        const void* hq8_packed,
                                        const void* down_W,
                                        const uint32_t* expert_offsets,
                                        sycl::half* out_packed,
                                        uint32_t E, uint32_t H, uint32_t E_ffn,
                                        uint64_t expert_stride_bytes, bool soa,
                                        const std::vector<sycl::event>& deps);

template <bool SOA>
static sycl::event moe_prefill_down_packed_q4k_q8_impl(sycl::queue& q,
                                        const void* hq8_packed,
                                        const void* down_W,
                                        const uint32_t* expert_offsets,
                                        sycl::half* out_packed,
                                        uint32_t E, uint32_t H, uint32_t E_ffn,
                                        uint64_t expert_stride_bytes,
                                        const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE   = 16;
    constexpr int N_PER_WG  = 32;
    constexpr int WG_ITEMS  = N_PER_WG * SG_SIZE;
    constexpr int M_TILE    = 16;
    constexpr uint32_t BW   = sizeof(block_q8_1s) / 4;   // 12 words / q8 block
    const uint32_t blocks_per_col = E_ffn / 256;         // = 2
    const uint32_t q8_per_row     = E_ffn / 32;          // = 16 == SG_SIZE
    const uint32_t n_chunks = (H + N_PER_WG - 1) / N_PER_WG;
    const uint64_t nb_e = uint64_t(blocks_per_col) * H;
    const auto* X8 = static_cast<const block_q8_1s*>(hq8_packed);

    return ie::ps(q, "moe_pfl_down_q8_4k", [&](sycl::handler& h) {
        h.depends_on(deps);
        // SLM: M_TILE rows x 16 q8 blocks, staged verbatim (12 KiB).
        sycl::local_accessor<uint32_t, 1> hq8(
            uint64_t(M_TILE) * 16 * BW, h);

        h.parallel_for(sycl::nd_range<2>({uint64_t(E) * n_chunks, WG_ITEMS},
                                          {1, WG_ITEMS}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t en = uint32_t(it.get_group(0));
            const uint32_t e  = en / n_chunks;
            const uint32_t nc = en % n_chunks;
            const uint32_t lid = uint32_t(it.get_local_id(1));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t n     = nc * N_PER_WG + sg_id;

            const uint32_t off_start = expert_offsets[e];
            const uint32_t off_end   = expert_offsets[e + 1];
            const uint32_t n_tok     = off_end - off_start;
            if (n_tok == 0) return;

            const auto dv = q4k_wview<SOA>::at(down_W, expert_stride_bytes, e, nb_e);

            // Lane -> q8 block `lane` of the row = Q4_K block b_in, sub-block sb.
            const int b_in   = int(lane) >> 3;
            const int sb     = int(lane) & 7;
            const int g      = sb >> 1;
            const int hi_nib = sb & 1;

            // Register-resident column weights — loaded once per WG.
            uint32_t wnib[8];
            float dsub = 0.f, msub = 0.f;
            if (n < H) {
                const uint64_t bi = uint64_t(n) * blocks_per_col + b_in;
                const uint8_t* sc = dv.sc_blk(bi);
                uint8_t s_raw, m_raw;
                if (sb < 4) {
                    s_raw = sc[sb]     & 0x3F;
                    m_raw = sc[sb + 4] & 0x3F;
                } else {
                    s_raw = (sc[sb + 4] & 0x0F) | ((sc[sb - 4] >> 6) << 4);
                    m_raw = (sc[sb + 4] >>   4) | ((sc[sb    ] >> 6) << 4);
                }
                dsub = dev_fp16_to_fp32(dv.d_bits(bi))    * float(s_raw);
                msub = dev_fp16_to_fp32(dv.dmin_bits(bi)) * float(m_raw);
                const auto qv0 = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(
                    dv.qs_blk(bi) + g * 32);
                const auto qv1 = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(
                    dv.qs_blk(bi) + g * 32 + 16);
                #pragma unroll
                for (int w = 0; w < 4; ++w) {
                    wnib[w]     = (hi_nib ? (qv0[w] >> 4) : qv0[w]) & 0x0F0F0F0Fu;
                    wnib[w + 4] = (hi_nib ? (qv1[w] >> 4) : qv1[w]) & 0x0F0F0F0Fu;
                }
            }

            for (uint32_t tk_base = 0; tk_base < n_tok; tk_base += M_TILE) {
                const uint32_t M = sycl::min(uint32_t(M_TILE), n_tok - tk_base);

                // Stage M rows of q8 h-blocks (verbatim words, coalesced).
                const uint32_t* src = reinterpret_cast<const uint32_t*>(
                    X8 + (uint64_t(off_start) + tk_base) * q8_per_row);
                for (uint32_t i = lid; i < M * 16 * BW; i += WG_ITEMS)
                    hq8[i] = src[i];
                sycl::group_barrier(it.get_group());

                if (n >= H) {
                    sycl::group_barrier(it.get_group());
                    continue;
                }

                const uint32_t* base =
                    hq8.get_multi_ptr<sycl::access::decorated::no>().get();
                float acc[M_TILE];
                #pragma unroll
                for (int mm = 0; mm < M_TILE; ++mm) {
                    acc[mm] = 0.f;
                    if (uint32_t(mm) < M) {
                        const uint32_t* blkp =
                            base + (uint32_t(mm) * 16 + uint32_t(lane)) * BW;
                        const auto hdr = *reinterpret_cast<const sycl::vec<float, 4>*>(blkp);
                        const auto q0 = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(blkp + 4);
                        const auto q1 = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(blkp + 8);
                        int32_t idot = 0;
                        #pragma unroll
                        for (int w = 0; w < 4; ++w)
                            idot = ie::dp4a_us(wnib[w], int32_t(q0[w]), idot);
                        #pragma unroll
                        for (int w = 0; w < 4; ++w)
                            idot = ie::dp4a_us(wnib[w + 4], int32_t(q1[w]), idot);
                        acc[mm] = dsub * (hdr[0] * float(idot)) - msub * (hdr[1] + hdr[2]);
                    }
                }

                #pragma unroll
                for (int mm = 0; mm < M_TILE; ++mm) {
                    if (uint32_t(mm) < M) {
                        const float r = sycl::reduce_over_group(it.get_sub_group(),
                                                                acc[mm], sycl::plus<float>());
                        if (lane == 0) {
                            out_packed[(uint64_t(off_start) + tk_base + mm) * H + n] =
                                sycl::half(r);
                        }
                    }
                }
                sycl::group_barrier(it.get_group());
            }
        });
    });
}

sycl::event moe_prefill_down_packed_q4k_q8(sycl::queue& q,
                                        const void* hq8_packed,
                                        const void* down_W,
                                        const uint32_t* expert_offsets,
                                        sycl::half* out_packed,
                                        uint32_t E, uint32_t H, uint32_t E_ffn,
                                        uint64_t expert_stride_bytes, bool soa,
                                        const std::vector<sycl::event>& deps) {
    return soa ? moe_prefill_down_packed_q4k_q8_impl<true >(q, hq8_packed, down_W,
                     expert_offsets, out_packed, E, H, E_ffn, expert_stride_bytes, deps)
               : moe_prefill_down_packed_q4k_q8_impl<false>(q, hq8_packed, down_W,
                     expert_offsets, out_packed, E, H, E_ffn, expert_stride_bytes, deps);
}

template <bool SOA>
static sycl::event moe_prefill_down_packed_q6k_q8_impl(sycl::queue& q,
                                        const void* hq8_packed,
                                        const void* down_W,
                                        const uint32_t* expert_offsets,
                                        sycl::half* out_packed,
                                        uint32_t E, uint32_t H, uint32_t E_ffn,
                                        uint64_t expert_stride_bytes,
                                        const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE   = 16;
    constexpr int N_PER_WG  = 32;
    constexpr int WG_ITEMS  = N_PER_WG * SG_SIZE;
    constexpr int M_TILE    = 16;
    constexpr uint32_t BW   = sizeof(block_q8_1s) / 4;
    const uint32_t blocks_per_col = E_ffn / 256;         // = 2
    const uint32_t q8_per_row     = E_ffn / 32;          // = 16 == SG_SIZE
    const uint32_t n_chunks = (H + N_PER_WG - 1) / N_PER_WG;
    const uint64_t nb_e = uint64_t(blocks_per_col) * H;
    const auto* X8 = static_cast<const block_q8_1s*>(hq8_packed);

    return ie::ps(q, "moe_pfl_down_q8_6k", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<uint32_t, 1> hq8(
            uint64_t(M_TILE) * 16 * BW, h);

        h.parallel_for(sycl::nd_range<2>({uint64_t(E) * n_chunks, WG_ITEMS},
                                          {1, WG_ITEMS}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t en = uint32_t(it.get_group(0));
            const uint32_t e  = en / n_chunks;
            const uint32_t nc = en % n_chunks;
            const uint32_t lid = uint32_t(it.get_local_id(1));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t n     = nc * N_PER_WG + sg_id;

            const uint32_t off_start = expert_offsets[e];
            const uint32_t off_end   = expert_offsets[e + 1];
            const uint32_t n_tok     = off_end - off_start;
            if (n_tok == 0) return;

            const auto dv = q6k_wview<SOA>::at(down_W, expert_stride_bytes, e, nb_e);

            // Lane -> q8 block `lane`: Q6_K block b_in, 32-elem group sb.
            const int b_in   = int(lane) >> 3;
            const int sb     = int(lane) & 7;
            const int half_q = sb >> 2;
            const int sub    = sb & 3;
            const int ql_off = half_q * 64 + (sub & 1) * 32;
            const int qh_off = half_q * 32;
            const int qh_shift = sub * 2;
            const int ql_shift = (sub & 2) ? 4 : 0;

            // Register-resident column weights, assembled once per WG:
            // q6u byte = (ql nibble) | (qh 2-bit << 4), value in [0, 64).
            uint32_t w6[8];
            float d6 = 0.f, sc0 = 0.f, sc1 = 0.f;
            if (n < H) {
                const uint64_t bi = uint64_t(n) * blocks_per_col + b_in;
                const int8_t* scp = dv.sc_blk(bi);
                sc0 = float(scp[half_q * 8 + sub * 2]);
                sc1 = float(scp[half_q * 8 + sub * 2 + 1]);
                d6  = dev_fp16_to_fp32(dv.d_bits(bi));
                uint32_t qlw[8], qhw[8];
                if constexpr (SOA) {
                    // SoA streams are 32 B-aligned at these offsets.
                    const auto l0 = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(
                        dv.ql_blk(bi) + ql_off);
                    const auto l1 = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(
                        dv.ql_blk(bi) + ql_off + 16);
                    const auto h0 = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(
                        dv.qh_blk(bi) + qh_off);
                    const auto h1 = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(
                        dv.qh_blk(bi) + qh_off + 16);
                    #pragma unroll
                    for (int w = 0; w < 4; ++w) {
                        qlw[w] = l0[w]; qlw[w + 4] = l1[w];
                        qhw[w] = h0[w]; qhw[w + 4] = h1[w];
                    }
                } else {
                    // AoS block_q6_K is only 2 B-aligned: u16-pair assembly.
                    const auto* ql16 = reinterpret_cast<const uint16_t*>(dv.ql_blk(bi) + ql_off);
                    const auto* qh16 = reinterpret_cast<const uint16_t*>(dv.qh_blk(bi) + qh_off);
                    #pragma unroll
                    for (int w = 0; w < 8; ++w) {
                        qlw[w] = uint32_t(ql16[w * 2]) | (uint32_t(ql16[w * 2 + 1]) << 16);
                        qhw[w] = uint32_t(qh16[w * 2]) | (uint32_t(qh16[w * 2 + 1]) << 16);
                    }
                }
                #pragma unroll
                for (int w = 0; w < 8; ++w) {
                    w6[w] = ((qlw[w] >> ql_shift) & 0x0F0F0F0Fu) |
                            (((qhw[w] >> qh_shift) & 0x03030303u) << 4);
                }
            }

            for (uint32_t tk_base = 0; tk_base < n_tok; tk_base += M_TILE) {
                const uint32_t M = sycl::min(uint32_t(M_TILE), n_tok - tk_base);

                const uint32_t* src = reinterpret_cast<const uint32_t*>(
                    X8 + (uint64_t(off_start) + tk_base) * q8_per_row);
                for (uint32_t i = lid; i < M * 16 * BW; i += WG_ITEMS)
                    hq8[i] = src[i];
                sycl::group_barrier(it.get_group());

                if (n >= H) {
                    sycl::group_barrier(it.get_group());
                    continue;
                }

                const uint32_t* base =
                    hq8.get_multi_ptr<sycl::access::decorated::no>().get();
                float acc[M_TILE];
                #pragma unroll
                for (int mm = 0; mm < M_TILE; ++mm) {
                    acc[mm] = 0.f;
                    if (uint32_t(mm) < M) {
                        const uint32_t* blkp =
                            base + (uint32_t(mm) * 16 + uint32_t(lane)) * BW;
                        const auto hdr = *reinterpret_cast<const sycl::vec<float, 4>*>(blkp);
                        const auto q0 = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(blkp + 4);
                        const auto q1 = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(blkp + 8);
                        // Per-16 scales: separate dots for each half, exact
                        // signed-offset correction via the split sums.
                        int32_t id0 = 0, id1 = 0;
                        #pragma unroll
                        for (int w = 0; w < 4; ++w)
                            id0 = ie::dp4a_us(w6[w], int32_t(q0[w]), id0);
                        #pragma unroll
                        for (int w = 0; w < 4; ++w)
                            id1 = ie::dp4a_us(w6[w + 4], int32_t(q1[w]), id1);
                        const float d8 = hdr[0];
                        acc[mm] = d6 * (sc0 * (d8 * float(id0) - 32.f * hdr[1]) +
                                        sc1 * (d8 * float(id1) - 32.f * hdr[2]));
                    }
                }

                #pragma unroll
                for (int mm = 0; mm < M_TILE; ++mm) {
                    if (uint32_t(mm) < M) {
                        const float r = sycl::reduce_over_group(it.get_sub_group(),
                                                                acc[mm], sycl::plus<float>());
                        if (lane == 0) {
                            out_packed[(uint64_t(off_start) + tk_base + mm) * H + n] =
                                sycl::half(r);
                        }
                    }
                }
                sycl::group_barrier(it.get_group());
            }
        });
    });
}

sycl::event moe_prefill_down_packed_q6k_q8(sycl::queue& q,
                                        const void* hq8_packed,
                                        const void* down_W,
                                        const uint32_t* expert_offsets,
                                        sycl::half* out_packed,
                                        uint32_t E, uint32_t H, uint32_t E_ffn,
                                        uint64_t expert_stride_bytes, bool soa,
                                        const std::vector<sycl::event>& deps) {
    return soa ? moe_prefill_down_packed_q6k_q8_impl<true >(q, hq8_packed, down_W,
                     expert_offsets, out_packed, E, H, E_ffn, expert_stride_bytes, deps)
               : moe_prefill_down_packed_q6k_q8_impl<false>(q, hq8_packed, down_W,
                     expert_offsets, out_packed, E, H, E_ffn, expert_stride_bytes, deps);
}

// Reduce: y[t, h] += Σ_k weights[tk_to_packed[t,k]] · out_packed[tk_to_packed[t,k], h]
sycl::event moe_prefill_reduce(sycl::queue& q,
                               const sycl::half* out_packed,
                               const uint32_t* tk_to_packed,
                               const sycl::half* weights_packed,
                               sycl::half* y,
                               uint32_t T, uint32_t K_top, uint32_t H,
                               const std::vector<sycl::event>& deps) {
    constexpr uint32_t WG = 256;
    const uint64_t total = uint64_t(T) * H;
    const uint64_t global = ((total + WG - 1) / WG) * WG;
    return ie::ps(q, "moe_pfl_reduce", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(global, WG),
                       [=](sycl::nd_item<1> it) {
            const uint64_t i = it.get_global_id(0);
            if (i >= total) return;
            const uint32_t t = uint32_t(i / H);
            const uint32_t n = uint32_t(i % H);
            float acc = float(y[uint64_t(t) * H + n]);
            for (uint32_t k = 0; k < K_top; ++k) {
                const uint32_t row = tk_to_packed[t * K_top + k];
                const float w = float(weights_packed[row]);
                acc += w * float(out_packed[uint64_t(row) * H + n]);
            }
            y[uint64_t(t) * H + n] = sycl::half(acc);
        });
    });
}

sycl::event moe_prefill_merge_fp32_to_fp16(sycl::queue& q,
                                           const float* y_fp32,
                                           sycl::half* y_fp16,
                                           uint64_t n,
                                           const std::vector<sycl::event>& deps) {
    return ie::ps(q, "moe_pfl_merge", [&](sycl::handler& h) {
        h.depends_on(deps);
        constexpr uint64_t WG = 256;
        const uint64_t global = ((n + WG - 1) / WG) * WG;
        h.parallel_for(sycl::nd_range<1>(global, WG),
                       [=](sycl::nd_item<1> it) {
            const uint64_t i = it.get_global_id(0);
            if (i >= n) return;
            y_fp16[i] = sycl::half(float(y_fp16[i]) + y_fp32[i]);
        });
    });
}

}  // namespace ie
