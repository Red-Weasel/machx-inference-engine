// src/ops/gemv_qk_grouped_moe.cpp — grouped W4/W5/W6 A16 MoE decode GEMVs.
//
// One launch covers all P (<= 8) routed experts of a layer at T==1, replacing
// the per-expert launch chain (~5 launches x 8 experts x 42 layers/token).
// Per (expert, output column) the corresponding SOLO kernel's lane lattice and
// FP fold run VERBATIM — gemv_q4_K_dual for gate/up, gemv_q5_K (T=1 d/m fold)
// and gemv_q6_K_slm (the N>=2048 route) for down — so outputs are
// bit-identical to the sequential per-expert path. The reduce epilogue walks
// experts in the caller's order inside one fp32 register chain, reproducing
// the sequential k_scatter_add_h rounding exactly.
//
// Expert slot ids and routing weights ride as by-value kernel arguments
// (std::array<...,8>) — no device upload, no extra launch.

#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"

#include <sycl/sycl.hpp>
#include <array>
#include "ie/kernel_profiler.hpp"

namespace ie {

namespace {
inline float dev_fp16_to_fp32(uint16_t h) {
    return float(sycl::bit_cast<sycl::half>(h));
}
using Slots8 = std::array<int32_t, 8>;
using Wts8   = std::array<float, 8>;
}  // namespace

sycl::event moe_gemv_q4_K_gu_grouped(sycl::queue& q, const sycl::half* A,
                                     const uint8_t* slot_base, uint64_t slot_bytes,
                                     uint64_t gate_off, uint64_t up_off,
                                     const int32_t* slots, uint32_t P,
                                     sycl::half* yg, sycl::half* yu,
                                     uint32_t K, uint32_t N,
                                     const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE  = 16;
    constexpr int N_PER_WG = 32;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;      // 512
    const uint32_t blocks_per_col = K / 256;
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;
    Slots8 sl{};
    for (uint32_t p = 0; p < P; ++p) sl[p] = slots[p];

    // Inner-P: one WG owns a column-tile and walks all P experts after a
    // single SLM load of A. The old grid was P×n_wgs — every expert WG
    // reloaded the SAME decode activation (T=1 x16_). Per-(p,n) FMA lattice
    // is verbatim; experts complete in order so y[p] is bit-identical to
    // the P-wide grid and to sequential gemv_q4_K_dual.
    return ie::ps(q, "g5_moe_gu_grp", [&](sycl::handler& h) {
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

            const int sub  = int(lane) >> 1;
            const int half = int(lane) & 1;
            const int g      = sub >> 1;
            const int hi_nib = sub & 1;
            const int qs_off = g * 32 + half * 16;
            const int out_off = g * 64 + hi_nib * 32 + half * 16;
            const int q4_shift = hi_nib ? 4 : 0;

            for (uint32_t p = 0; p < P; ++p) {
                const uint8_t* slot = slot_base + uint64_t(sl[p]) * slot_bytes;
                const auto* W_alpha = reinterpret_cast<const block_q4_K*>(slot + gate_off);
                const auto* W_beta  = reinterpret_cast<const block_q4_K*>(slot + up_off);

                float a_acc = 0.f;
                float b_acc = 0.f;
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
                    // saturate, not inf — the sequential leaves (gemv_q4k.cpp) do;
                    // this path did not, so prefill (grouped) and decode could
                    // diverge on overflow and prefill could emit inf -> NaN
                    // (Astra review 2026-09-04). No-op below 65504.
                    yg[uint64_t(p) * N + n] = sycl::half(sycl::clamp(a_acc, -65504.f, 65504.f));
                    yu[uint64_t(p) * N + n] = sycl::half(sycl::clamp(b_acc, -65504.f, 65504.f));
                }
            }
        });
    });
}

sycl::event moe_gemv_q5_K_down_grouped(sycl::queue& q, const sycl::half* A,
                                       const uint8_t* slot_base, uint64_t slot_bytes,
                                       uint64_t down_off,
                                       const int32_t* slots, uint32_t P,
                                       sycl::half* y, uint32_t K, uint32_t N,
                                       const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE  = 16;
    constexpr int N_PER_WG = 32;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;
    const uint32_t blocks_per_col = K / 256;
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;
    Slots8 sl{};
    for (uint32_t p = 0; p < P; ++p) sl[p] = slots[p];

    // Inner-P: one WG owns a column-tile and walks all P experts. A is
    // per-expert (yg rows), so each p reloads SLM; the win is n_wgs WGs
    // doing real work instead of P×n_wgs launch/occupancy. Per-(p,n) fold
    // is verbatim vs gemv_q5_K / the P-wide grid.
    return ie::ps(q, "g5_moe_dn_grp5", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<sycl::half, 1> A_slm(sycl::range<1>(K), h);

        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t wgid  = uint32_t(it.get_group(0));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t n     = wgid * N_PER_WG + sg_id;

            const int sub  = int(lane) >> 1;
            const int half = int(lane) & 1;
            const int g      = sub >> 1;
            const int hi_nib = sub & 1;
            const int qs_off = g * 32 + half * 16;
            const int qh_off = half * 16;
            const int out_off = g * 64 + hi_nib * 32 + half * 16;
            const uint8_t ubit = uint8_t(1u << sub);
            const int q4_shift = hi_nib ? 4 : 0;

            for (uint32_t p = 0; p < P; ++p) {
                const sycl::half* Ap = A + uint64_t(p) * K;
                for (uint32_t i = lid; i < K; i += WG_ITEMS) A_slm[i] = Ap[i];
                sycl::group_barrier(it.get_group());

                if (n < N) {
                const auto* W = reinterpret_cast<const block_q5_K*>(
                    slot_base + uint64_t(sl[p]) * slot_bytes + down_off);

                float acc = 0.f;
                const block_q5_K* col_blocks = &W[n * blocks_per_col];

                for (uint32_t b = 0; b < blocks_per_col; ++b) {
                    const block_q5_K& blk = col_blocks[b];

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

                    const sycl::vec<uint8_t, 16> qs_v =
                        *reinterpret_cast<const sycl::vec<uint8_t, 16>*>(&blk.qs[qs_off]);
                    const sycl::vec<uint8_t, 16> qh_v =
                        *reinterpret_cast<const sycl::vec<uint8_t, 16>*>(&blk.qh[qh_off]);
                    const sycl::vec<sycl::half, 16> a_v =
                        *reinterpret_cast<const sycl::vec<sycl::half, 16>*>(&A_slm[b * 256 + out_off]);

                    float sum_aq = 0.f;
                    float sum_a  = 0.f;
                    #pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        const int q5 = ((qs_v[i] >> q4_shift) & 0x0F) +
                                       ((qh_v[i] & ubit) ? 16 : 0);
                        const float a = float(a_v[i]);
                        sum_aq += a * float(q5);
                        sum_a  += a;
                    }
                    acc += d * sum_aq - m * sum_a;
                }

                acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
                if (lane == 0) y[uint64_t(p) * N + n] = sycl::half(sycl::clamp(acc, -65504.f, 65504.f));   // saturate, not inf (see Q4 above)
                }
                // All SGs must finish reading A_slm before the next p overwrites it.
                sycl::group_barrier(it.get_group());
            }
        });
    });
}

// Q6_K grouped down copies gemv_q6_K_slm (the default N>=2048 route; the
// IE_NO_Q6K_SLM experimental fallback is a different FP fold and is NOT
// mirrored here — callers gate grouped separately).
sycl::event moe_gemv_q6_K_down_grouped(sycl::queue& q, const sycl::half* A,
                                       const uint8_t* slot_base, uint64_t slot_bytes,
                                       uint64_t down_off,
                                       const int32_t* slots, uint32_t P,
                                       sycl::half* y, uint32_t K, uint32_t N,
                                       const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE  = 16;
    constexpr int N_PER_WG = 16;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;      // 256
    const uint32_t blocks_per_col = K / 256;
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;
    const uint32_t slab_bytes =
        N_PER_WG * blocks_per_col * uint32_t(sizeof(block_q6_K));
    Slots8 sl{};
    for (uint32_t p = 0; p < P; ++p) sl[p] = slots[p];

    return ie::ps(q, "g5_moe_dn_grp6", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<uint8_t, 1> W_slm(slab_bytes, h);

        h.parallel_for(sycl::nd_range<1>(uint64_t(P) * n_wgs * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t gid   = uint32_t(it.get_group(0));
            const uint32_t p     = gid / n_wgs;
            const uint32_t wgid  = gid % n_wgs;
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t n0    = wgid * N_PER_WG;
            const uint32_t n     = n0 + sg_id;

            const auto* W = reinterpret_cast<const block_q6_K*>(
                slot_base + uint64_t(sl[p]) * slot_bytes + down_off);
            const sycl::half* Ap = A + uint64_t(p) * K;

            {
                const uint32_t cols = sycl::min(uint32_t(N_PER_WG), N - n0);
                const uint32_t n_bytes =
                    cols * blocks_per_col * uint32_t(sizeof(block_q6_K));
                const uint8_t* src_b = reinterpret_cast<const uint8_t*>(W) +
                    uint64_t(n0) * blocks_per_col * sizeof(block_q6_K);
                uint8_t* dst_b =
                    W_slm.get_multi_ptr<sycl::access::decorated::no>().get();
                const auto* src = reinterpret_cast<const sycl::vec<uint32_t, 4>*>(src_b);
                auto* dst = reinterpret_cast<sycl::vec<uint32_t, 4>*>(dst_b);
                const uint32_t n_vec = n_bytes / 16;
                for (uint32_t i = lid; i < n_vec; i += WG_ITEMS) dst[i] = src[i];
                for (uint32_t i = n_vec * 16 + lid; i < n_bytes; i += WG_ITEMS)
                    dst_b[i] = src_b[i];
            }
            sycl::group_barrier(it.get_group());
            if (n >= N) return;

            const int half   = int(lane) >> 3;
            const int sub    = (int(lane) >> 1) & 0x3;
            const int l_half = int(lane) & 0x1;
            const int l_start = l_half * 16;
            const int ql_off    = half * 64 + (sub & 1) * 32 + l_start;
            const int qh_off    = half * 32 + l_start;
            const int scale_off = half * 8  + sub * 2  + l_half;
            const int qh_shift  = sub * 2;
            const bool high_nibble = (sub & 2) != 0;
            const int out_off = half * 128 + sub * 32 + l_start;
            const int ql_shift = high_nibble ? 4 : 0;

            float acc = 0.f;
            const uint8_t* col_base =
                W_slm.get_multi_ptr<sycl::access::decorated::no>().get() +
                uint64_t(sg_id) * blocks_per_col * sizeof(block_q6_K);

            for (uint32_t b = 0; b < blocks_per_col; ++b) {
                const auto& blk = *reinterpret_cast<const block_q6_K*>(
                    col_base + uint64_t(b) * sizeof(block_q6_K));
                const sycl::half* a_chunk = &Ap[b * 256 + out_off];
                const float d =
                    dev_fp16_to_fp32(blk.d) * float(blk.scales[scale_off]);

                float sum_aq = 0.f, sum_a = 0.f;
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const uint8_t ql_b = blk.ql[ql_off + i];
                    const uint8_t qh_b = blk.qh[qh_off + i];
                    const int qu = int((ql_b >> ql_shift) & 0x0F) |
                                   (int((qh_b >> qh_shift) & 0x3) << 4);
                    const float a = float(a_chunk[i]);
                    sum_aq += a * float(qu);
                    sum_a  += a;
                }
                acc += d * (sum_aq - 32.0f * sum_a);
            }

            acc = sycl::reduce_over_group(it.get_sub_group(), acc,
                                          sycl::plus<float>());
            if (lane == 0) y[uint64_t(p) * N + n] = sycl::half(acc);
        });
    });
}

sycl::event moe_reduce_waccum(sycl::queue& q, const sycl::half* y,
                              const float* wts, uint32_t P,
                              float* acc, uint32_t H,
                              const std::vector<sycl::event>& deps) {
    Wts8 w{};
    for (uint32_t p = 0; p < P; ++p) w[p] = wts[p];
    return ie::ps(q, "g5_moe_red", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(H), [=](sycl::id<1> id) {
            const uint32_t d = uint32_t(id[0]);
            // fp32 register chain in expert order == the sequential
            // k_scatter_add_h launches' rounding, bit for bit.
            float t = acc[d];
            for (uint32_t p = 0; p < P; ++p)
                t += w[p] * float(y[uint64_t(p) * H + d]);
            acc[d] = t;
        });
    });
}


// ---------------------------------------------------------------------------
// PREFILL grouped-tiles (A16): one launch covers a WAVE of experts' multi-row
// GEMVs — jobs = [G x 3] i32 {pickA, rc, slot}: a <=16-row chunk of one
// expert. Per (chunk, column) the gemv_q4_K_rows / gemv_q5_K_rows wv-fold
// lattice runs VERBATIM (bit-identical outputs). row_idx (pick -> token row)
// fuses the gather: X is read at row_idx[pickA+r]; nullptr row_idx reads X
// contiguously at (pickA-pick0)+r (the down projection over wave-local
// swiglu rows). y rows are wave-local: (pickA-pick0)+r.
// ---------------------------------------------------------------------------
namespace {
template <typename BLK, bool Q5>
sycl::event rows_tiles_impl(sycl::queue& q, const char* prof,
                            const sycl::half* X, const int32_t* row_idx,
                            uint32_t pick0,
                            const uint8_t* slot_base, uint64_t slot_bytes,
                            uint64_t w_off, const int32_t* jobs,
                            sycl::half* y, uint32_t K, uint32_t N, uint32_t G,
                            const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE = 16;
    constexpr int N_PER_WG = 32;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;
    constexpr uint32_t RT = 16;
    const uint32_t blocks_per_col = K / 256;
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;
    return ie::ps(q, prof, [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(uint64_t(G) * n_wgs * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid  = uint32_t(it.get_local_id(0));
            const uint32_t gid  = uint32_t(it.get_group(0));
            const uint32_t g    = gid / n_wgs;
            const uint32_t n    = (gid % n_wgs) * N_PER_WG + lid / SG_SIZE;
            const uint32_t lane = lid % SG_SIZE;
            if (n >= N) return;
            const int32_t pickA = jobs[3u * g + 0];
            const uint32_t rc   = uint32_t(jobs[3u * g + 1]);
            const int32_t slot  = jobs[3u * g + 2];
            const auto* W = reinterpret_cast<const BLK*>(
                slot_base + uint64_t(slot) * slot_bytes + w_off);

            const int sub  = int(lane) >> 1;
            const int half = int(lane) & 1;
            const int g2     = sub >> 1;
            const int hi_nib = sub & 1;
            const int qs_off = g2 * 32 + half * 16;
            const int qh_off = half * 16;               // Q5 only
            const int out_off = g2 * 64 + hi_nib * 32 + half * 16;
            const uint8_t ubit = uint8_t(1u << sub);    // Q5 only
            const int q4_shift = hi_nib ? 4 : 0;

            float acc[RT];
            #pragma unroll
            for (uint32_t r = 0; r < RT; ++r) acc[r] = 0.f;
            const BLK* col = &W[uint64_t(n) * blocks_per_col];

            for (uint32_t b = 0; b < blocks_per_col; ++b) {
                const BLK& blk = col[b];
                uint8_t s_raw, m_raw;
                if (sub < 4) {
                    s_raw = blk.scales[sub]     & 0x3F;
                    m_raw = blk.scales[sub + 4] & 0x3F;
                } else {
                    s_raw = (blk.scales[sub + 4] & 0x0F) | ((blk.scales[sub - 4] >> 6) << 4);
                    m_raw = (blk.scales[sub + 4] >>   4) | ((blk.scales[sub    ] >> 6) << 4);
                }
                const float dsc = dev_fp16_to_fp32(blk.d)    * float(s_raw);
                const float msc = dev_fp16_to_fp32(blk.dmin) * float(m_raw);
                const sycl::vec<uint8_t, 16> qs_v =
                    *reinterpret_cast<const sycl::vec<uint8_t, 16>*>(&blk.qs[qs_off]);
                float wv[16];
                if constexpr (Q5) {
                    const sycl::vec<uint8_t, 16> qh_v =
                        *reinterpret_cast<const sycl::vec<uint8_t, 16>*>(&blk.qh[qh_off]);
                    #pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        const int q5v = ((qs_v[i] >> q4_shift) & 0x0F) +
                                        ((qh_v[i] & ubit) ? 16 : 0);
                        wv[i] = dsc * float(q5v) - msc;
                    }
                } else {
                    #pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        const int q4v = (qs_v[i] >> q4_shift) & 0x0F;
                        wv[i] = dsc * float(q4v) - msc;
                    }
                }
                const uint64_t a_off = uint64_t(b) * 256 + out_off;
                for (uint32_t r = 0; r < rc; ++r) {
                    const uint64_t arow = row_idx
                        ? uint64_t(row_idx[pickA + int32_t(r)])
                        : uint64_t(pickA - int32_t(pick0)) + r;
                    const auto a_v = *reinterpret_cast<const sycl::vec<sycl::half, 16>*>(
                        X + arow * K + a_off);
                    float s = 0.f;
                    #pragma unroll
                    for (int i = 0; i < 16; ++i) s += float(a_v[i]) * wv[i];
                    acc[r] += s;
                }
            }
            auto sg = it.get_sub_group();
            for (uint32_t r = 0; r < rc; ++r) {
                const float v = sycl::reduce_over_group(sg, acc[r], sycl::plus<float>());
                if (lane == 0)
                    y[(uint64_t(pickA - int32_t(pick0)) + r) * N + n] = sycl::half(v);
            }
        });
    });
}
}  // namespace

sycl::event moe_gemv_q4_K_rows_tiles(sycl::queue& q,
        const sycl::half* X, const int32_t* row_idx, uint32_t pick0,
        const uint8_t* slot_base, uint64_t slot_bytes, uint64_t w_off,
        const int32_t* jobs, sycl::half* y, uint32_t K, uint32_t N, uint32_t G,
        const std::vector<sycl::event>& deps) {
    return rows_tiles_impl<block_q4_K, false>(q, "g5_pp_q4_tiles", X, row_idx,
        pick0, slot_base, slot_bytes, w_off, jobs, y, K, N, G, deps);
}
sycl::event moe_gemv_q5_K_rows_tiles(sycl::queue& q,
        const sycl::half* X, const int32_t* row_idx, uint32_t pick0,
        const uint8_t* slot_base, uint64_t slot_bytes, uint64_t w_off,
        const int32_t* jobs, sycl::half* y, uint32_t K, uint32_t N, uint32_t G,
        const std::vector<sycl::event>& deps) {
    return rows_tiles_impl<block_q5_K, true>(q, "g5_pp_q5_tiles", X, row_idx,
        pick0, slot_base, slot_bytes, w_off, jobs, y, K, N, G, deps);
}
}  // namespace ie
