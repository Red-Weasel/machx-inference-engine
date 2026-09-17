// src/ops/gemv_q5k.cpp — W5A16 GEMV with Q5_K weights (glm5next expert down
// banks; one gate/up pair).
//
// gemv_q4_K's structure verbatim (16-lane SG per output column, cooperative
// A-SLM, algebraic d/m fold) with the Q5_K delta: each element carries a 5th
// bit in qh[32] — bit `sub` of qh[l], where l ∈ [0, 32) indexes the element
// within its 32-wide sub-block. The lane's 16-element chunk therefore reads
// 16 qh bytes at half*16 alongside the 16 qs bytes at g*32 + half*16.
#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"

#include <sycl/sycl.hpp>
#include "ie/kernel_profiler.hpp"

namespace ie {

namespace {
inline float dev_fp16_to_fp32(uint16_t h) {
    return float(sycl::bit_cast<sycl::half>(h));
}
}  // namespace

sycl::event gemv_q5_K(sycl::queue& q,
                      const sycl::half* A, const void* W_packed,
                      sycl::half* y,
                      uint32_t K, uint32_t N,
                      const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE = 16;
    const int N_PER_WG = 32;
    const int WG_ITEMS = N_PER_WG * SG_SIZE;

    const auto* W = static_cast<const block_q5_K*>(W_packed);
    const uint32_t blocks_per_col = K / 256;
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "gemv_q5k", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<sycl::half, 1> A_slm(sycl::range<1>(K), h);

        h.parallel_for(sycl::nd_range<1>(n_wgs * WG_ITEMS, WG_ITEMS),
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
            const int qh_off = half * 16;
            const int out_off = g * 64 + hi_nib * 32 + half * 16;
            const uint8_t ubit = uint8_t(1u << sub);

            float acc = 0.f;
            const block_q5_K* col_blocks = &W[n * blocks_per_col];
            const int q4_shift = hi_nib ? 4 : 0;

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
            if (lane == 0) y[n] = sycl::half(sycl::clamp(acc, -65504.f, 65504.f));   // saturate, not inf
        });
    });
}

// Multi-row variant: R rows (<= 16 per internal chunk) share every weight
// decode; A is read from GLOBAL (L2 carries the row reuse) so there is no
// SLM ceiling at K=4096. Same lane lattice and per-output math as the T=1
// kernel. y[r*N + n]. Built for the glm5next prefill MoE (replaces the
// dequant-materialize + gemm_fp16 chain: ~1/7th the bytes per expert).
sycl::event gemv_q5_K_rows(sycl::queue& q,
                           const sycl::half* A, const void* W_packed,
                           sycl::half* y,
                           uint32_t K, uint32_t N, uint32_t R,
                           const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE = 16;
    constexpr int N_PER_WG = 32;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;
    constexpr uint32_t RT = 16;
    const auto* W = static_cast<const block_q5_K*>(W_packed);
    const uint32_t blocks_per_col = K / 256;
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;

    sycl::event last;
    std::vector<sycl::event> d = deps;
    for (uint32_t r0 = 0; r0 < R; r0 += RT) {
        const uint32_t rc = std::min(RT, R - r0);
        last = ie::ps(q, "gemv_q5k_rows", [&](sycl::handler& h) {
            h.depends_on(d);
            h.parallel_for(sycl::nd_range<1>(n_wgs * WG_ITEMS, WG_ITEMS),
                           [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const uint32_t lid  = uint32_t(it.get_local_id(0));
                const uint32_t n    = uint32_t(it.get_group(0)) * N_PER_WG + lid / SG_SIZE;
                const uint32_t lane = lid % SG_SIZE;
                if (n >= N) return;
                const int sub  = int(lane) >> 1;
                const int half = int(lane) & 1;
                const int g      = sub >> 1;
                const int hi_nib = sub & 1;
                const int qs_off = g * 32 + half * 16;
                const int qh_off = half * 16;
                const int out_off = g * 64 + hi_nib * 32 + half * 16;
                const uint8_t ubit = uint8_t(1u << sub);
                const int q4_shift = hi_nib ? 4 : 0;

                float acc[RT];
                #pragma unroll
                for (uint32_t r = 0; r < RT; ++r) acc[r] = 0.f;
                const block_q5_K* col = &W[uint64_t(n) * blocks_per_col];
                const sycl::half* Ab = A + uint64_t(r0) * K;

                for (uint32_t b = 0; b < blocks_per_col; ++b) {
                    const block_q5_K& blk = col[b];
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
                    const sycl::vec<uint8_t, 16> qh_v =
                        *reinterpret_cast<const sycl::vec<uint8_t, 16>*>(&blk.qh[qh_off]);
                    float wv[16];
                    #pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        const int q5 = ((qs_v[i] >> q4_shift) & 0x0F) +
                                       ((qh_v[i] & ubit) ? 16 : 0);
                        wv[i] = dsc * float(q5) - msc;
                    }
                    const uint64_t a_off = uint64_t(b) * 256 + out_off;
                    for (uint32_t r = 0; r < rc; ++r) {
                        const auto a_v = *reinterpret_cast<const sycl::vec<sycl::half, 16>*>(
                            Ab + uint64_t(r) * K + a_off);
                        float s = 0.f;
                        #pragma unroll
                        for (int i = 0; i < 16; ++i) s += float(a_v[i]) * wv[i];
                        acc[r] += s;
                    }
                }
                auto sg = it.get_sub_group();
                for (uint32_t r = 0; r < rc; ++r) {
                    const float v = sycl::reduce_over_group(sg, acc[r], sycl::plus<float>());
                    if (lane == 0) y[(uint64_t(r0) + r) * N + n] = sycl::half(v);
                }
            });
        });
        d = {last};
    }
    return last;
}

}  // namespace ie
