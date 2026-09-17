// src/ops/moe_qwen3.cpp — qwen3moe-owned generalized int-dot MoE down kernels.
//
// Generalizes the crown's E_ffn==512-locked int-dot down kernels
// (moe_fused.cpp moe_prefill_down_packed_q{4,6}k_q8) to any E_ffn % 256 == 0,
// so Qwen3-Coder-30B's E_ffn=768 gets the same W4A8 int-dot path that beats the
// fp16-activation fallback (≈83% of qwen3moe prefill GPU time). The ONLY change
// vs the crown kernel is the K-direction tiling: instead of one q8 block per
// lane (q8_per_row==16==SG_SIZE), each lane walks q8 blocks {lane, lane+16, …} <
// q8_per_row and accumulates them, then the SG-reduce sums across lanes — so the
// full K=E_ffn reduction is covered for any q8_per_row. At E_ffn=512 this is
// n_blk_per_lane==1 and reduces bit-for-bit to the crown kernel.
//
// moe_fused.cpp is NOT edited (P2/P3 iron rule). dev_fp16_to_fp32 is copied
// (copy-not-hoist). ESIMD-safe — plain SLM + vec loads, no block2d/lsc paths.

#include "ie/moe_qwen3.hpp"

#include "ie/dp4a.hpp"
#include "ie/kernel_profiler.hpp"
#include "ie/quant_soa.hpp"

namespace ie {
namespace {

// Same fp16→fp32 helper as gemv_q*k.cpp / moe_fused.cpp (copy-not-hoist).
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

constexpr int SG_SIZE   = 16;
constexpr int N_PER_WG  = 32;
constexpr int WG_ITEMS  = N_PER_WG * SG_SIZE;
constexpr int M_TILE    = 16;
constexpr uint32_t BW   = sizeof(block_q8_1s) / 4;   // 12 words / q8 block
constexpr int MAX_BPL   = 4;                         // n_blk_per_lane cap (E_ffn ≤ 2048)

// ----------------------------- Q6_K down -----------------------------------
template <bool SOA>
sycl::event down_q6k_q8_gen_impl(sycl::queue& q,
                                 const void* hq8_packed, const void* down_W,
                                 const uint32_t* expert_offsets, sycl::half* out_packed,
                                 uint32_t E, uint32_t H, uint32_t E_ffn,
                                 uint64_t expert_stride_bytes,
                                 const std::vector<sycl::event>& deps) {
    const uint32_t blocks_per_col = E_ffn / 256;        // Q6_K blocks per H column
    const uint32_t q8_per_row     = E_ffn / 32;         // q8 activation blocks per token row
    const uint32_t n_blk_per_lane = (q8_per_row + SG_SIZE - 1) / SG_SIZE;
    const uint32_t n_chunks = (H + N_PER_WG - 1) / N_PER_WG;
    const uint64_t nb_e = uint64_t(blocks_per_col) * H; // blocks per expert
    const auto* X8 = static_cast<const block_q8_1s*>(hq8_packed);

    return ie::ps(q, "moe_pfl_down_q8_6k_gen", [&](sycl::handler& h) {
        h.depends_on(deps);
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
            const uint32_t n     = nc * N_PER_WG + sg_id;   // H output column

            const uint32_t off_start = expert_offsets[e];
            const uint32_t off_end   = expert_offsets[e + 1];
            const uint32_t n_tok     = off_end - off_start;
            if (n_tok == 0) return;

            const auto dv = q6k_wview<SOA>::at(down_W, expert_stride_bytes, e, nb_e);

            // Register-resident column weights for each q8 block this lane owns.
            uint32_t w6[MAX_BPL][8];
            float d6[MAX_BPL], sc0[MAX_BPL], sc1[MAX_BPL];
            uint32_t my_j[MAX_BPL];
            uint32_t my_nblk = 0;
            if (n < H) {
                for (uint32_t s = 0; s < n_blk_per_lane; ++s) {
                    const uint32_t j = lane + s * SG_SIZE;
                    if (j >= q8_per_row) break;
                    const int b_in   = int(j) >> 3;     // Q6_K block in column
                    const int sb     = int(j) & 7;      // 32-elem sub-group
                    const int half_q = sb >> 2;
                    const int sub    = sb & 3;
                    const int ql_off = half_q * 64 + (sub & 1) * 32;
                    const int qh_off = half_q * 32;
                    const int qh_shift = sub * 2;
                    const int ql_shift = (sub & 2) ? 4 : 0;
                    const uint64_t bi = uint64_t(n) * blocks_per_col + b_in;
                    const int8_t* scp = dv.sc_blk(bi);
                    sc0[my_nblk] = float(scp[half_q * 8 + sub * 2]);
                    sc1[my_nblk] = float(scp[half_q * 8 + sub * 2 + 1]);
                    d6[my_nblk]  = dev_fp16_to_fp32(dv.d_bits(bi));
                    uint32_t qlw[8], qhw[8];
                    if constexpr (SOA) {
                        const auto l0 = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(dv.ql_blk(bi) + ql_off);
                        const auto l1 = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(dv.ql_blk(bi) + ql_off + 16);
                        const auto h0 = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(dv.qh_blk(bi) + qh_off);
                        const auto h1 = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(dv.qh_blk(bi) + qh_off + 16);
                        #pragma unroll
                        for (int w = 0; w < 4; ++w) {
                            qlw[w] = l0[w]; qlw[w + 4] = l1[w];
                            qhw[w] = h0[w]; qhw[w + 4] = h1[w];
                        }
                    } else {
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
                        w6[my_nblk][w] = ((qlw[w] >> ql_shift) & 0x0F0F0F0Fu) |
                                         (((qhw[w] >> qh_shift) & 0x03030303u) << 4);
                    }
                    my_j[my_nblk] = j;
                    ++my_nblk;
                }
            }

            for (uint32_t tk_base = 0; tk_base < n_tok; tk_base += M_TILE) {
                const uint32_t M = sycl::min(uint32_t(M_TILE), n_tok - tk_base);

                // Stage M rows of q8 h-blocks (q8_per_row blocks/row, coalesced).
                const uint32_t* src = reinterpret_cast<const uint32_t*>(
                    X8 + (uint64_t(off_start) + tk_base) * q8_per_row);
                for (uint32_t i = lid; i < M * q8_per_row * BW; i += WG_ITEMS)
                    hq8[i] = src[i];
                sycl::group_barrier(it.get_group());

                if (n >= H) { sycl::group_barrier(it.get_group()); continue; }

                const uint32_t* base =
                    hq8.get_multi_ptr<sycl::access::decorated::no>().get();
                float acc[M_TILE];
                #pragma unroll
                for (int mm = 0; mm < M_TILE; ++mm) {
                    acc[mm] = 0.f;
                    if (uint32_t(mm) < M) {
                        for (uint32_t s = 0; s < my_nblk; ++s) {
                            const uint32_t* blkp =
                                base + (uint32_t(mm) * q8_per_row + my_j[s]) * BW;
                            const auto hdr = *reinterpret_cast<const sycl::vec<float, 4>*>(blkp);
                            const auto q0 = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(blkp + 4);
                            const auto q1 = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(blkp + 8);
                            int32_t id0 = 0, id1 = 0;
                            #pragma unroll
                            for (int w = 0; w < 4; ++w)
                                id0 = ie::dp4a_us(w6[s][w], int32_t(q0[w]), id0);
                            #pragma unroll
                            for (int w = 0; w < 4; ++w)
                                id1 = ie::dp4a_us(w6[s][w + 4], int32_t(q1[w]), id1);
                            const float d8 = hdr[0];
                            acc[mm] += d6[s] * (sc0[s] * (d8 * float(id0) - 32.f * hdr[1]) +
                                                sc1[s] * (d8 * float(id1) - 32.f * hdr[2]));
                        }
                    }
                }

                #pragma unroll
                for (int mm = 0; mm < M_TILE; ++mm) {
                    if (uint32_t(mm) < M) {
                        const float r = sycl::reduce_over_group(it.get_sub_group(),
                                                                acc[mm], sycl::plus<float>());
                        if (lane == 0)
                            out_packed[(uint64_t(off_start) + tk_base + mm) * H + n] = sycl::half(r);
                    }
                }
                sycl::group_barrier(it.get_group());
            }
        });
    });
}

// ----------------------------- Q4_K down -----------------------------------
template <bool SOA>
sycl::event down_q4k_q8_gen_impl(sycl::queue& q,
                                 const void* hq8_packed, const void* down_W,
                                 const uint32_t* expert_offsets, sycl::half* out_packed,
                                 uint32_t E, uint32_t H, uint32_t E_ffn,
                                 uint64_t expert_stride_bytes,
                                 const std::vector<sycl::event>& deps) {
    const uint32_t blocks_per_col = E_ffn / 256;
    const uint32_t q8_per_row     = E_ffn / 32;
    const uint32_t n_blk_per_lane = (q8_per_row + SG_SIZE - 1) / SG_SIZE;
    const uint32_t n_chunks = (H + N_PER_WG - 1) / N_PER_WG;
    const uint64_t nb_e = uint64_t(blocks_per_col) * H;
    const auto* X8 = static_cast<const block_q8_1s*>(hq8_packed);

    return ie::ps(q, "moe_pfl_down_q8_4k_gen", [&](sycl::handler& h) {
        h.depends_on(deps);
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

            const auto dv = q4k_wview<SOA>::at(down_W, expert_stride_bytes, e, nb_e);

            uint32_t wnib[MAX_BPL][8];
            float dsub[MAX_BPL], msub[MAX_BPL];
            uint32_t my_j[MAX_BPL];
            uint32_t my_nblk = 0;
            if (n < H) {
                for (uint32_t s = 0; s < n_blk_per_lane; ++s) {
                    const uint32_t j = lane + s * SG_SIZE;
                    if (j >= q8_per_row) break;
                    const int b_in   = int(j) >> 3;
                    const int sb     = int(j) & 7;
                    const int g      = sb >> 1;
                    const int hi_nib = sb & 1;
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
                    dsub[my_nblk] = dev_fp16_to_fp32(dv.d_bits(bi))    * float(s_raw);
                    msub[my_nblk] = dev_fp16_to_fp32(dv.dmin_bits(bi)) * float(m_raw);
                    const auto qv0 = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(dv.qs_blk(bi) + g * 32);
                    const auto qv1 = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(dv.qs_blk(bi) + g * 32 + 16);
                    #pragma unroll
                    for (int w = 0; w < 4; ++w) {
                        wnib[my_nblk][w]     = (hi_nib ? (qv0[w] >> 4) : qv0[w]) & 0x0F0F0F0Fu;
                        wnib[my_nblk][w + 4] = (hi_nib ? (qv1[w] >> 4) : qv1[w]) & 0x0F0F0F0Fu;
                    }
                    my_j[my_nblk] = j;
                    ++my_nblk;
                }
            }

            for (uint32_t tk_base = 0; tk_base < n_tok; tk_base += M_TILE) {
                const uint32_t M = sycl::min(uint32_t(M_TILE), n_tok - tk_base);

                const uint32_t* src = reinterpret_cast<const uint32_t*>(
                    X8 + (uint64_t(off_start) + tk_base) * q8_per_row);
                for (uint32_t i = lid; i < M * q8_per_row * BW; i += WG_ITEMS)
                    hq8[i] = src[i];
                sycl::group_barrier(it.get_group());

                if (n >= H) { sycl::group_barrier(it.get_group()); continue; }

                const uint32_t* base =
                    hq8.get_multi_ptr<sycl::access::decorated::no>().get();
                float acc[M_TILE];
                #pragma unroll
                for (int mm = 0; mm < M_TILE; ++mm) {
                    acc[mm] = 0.f;
                    if (uint32_t(mm) < M) {
                        for (uint32_t s = 0; s < my_nblk; ++s) {
                            const uint32_t* blkp =
                                base + (uint32_t(mm) * q8_per_row + my_j[s]) * BW;
                            const auto hdr = *reinterpret_cast<const sycl::vec<float, 4>*>(blkp);
                            const auto q0 = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(blkp + 4);
                            const auto q1 = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(blkp + 8);
                            int32_t idot = 0;
                            #pragma unroll
                            for (int w = 0; w < 4; ++w)
                                idot = ie::dp4a_us(wnib[s][w], int32_t(q0[w]), idot);
                            #pragma unroll
                            for (int w = 0; w < 4; ++w)
                                idot = ie::dp4a_us(wnib[s][w + 4], int32_t(q1[w]), idot);
                            acc[mm] += dsub[s] * (hdr[0] * float(idot)) - msub[s] * (hdr[1] + hdr[2]);
                        }
                    }
                }

                #pragma unroll
                for (int mm = 0; mm < M_TILE; ++mm) {
                    if (uint32_t(mm) < M) {
                        const float r = sycl::reduce_over_group(it.get_sub_group(),
                                                                acc[mm], sycl::plus<float>());
                        if (lane == 0)
                            out_packed[(uint64_t(off_start) + tk_base + mm) * H + n] = sycl::half(r);
                    }
                }
                sycl::group_barrier(it.get_group());
            }
        });
    });
}

// -------------------- Q6_K gate+up+silu (int-dot W6A8) ----------------------
// Mirror of moe_fused.cpp:moe_prefill_gate_up_silu_q4k_q8_impl — SAME two-bank
// outer schedule (n_steps = blocks_per_col/2 over the H contraction, lane →
// (b_in = lane>>3, sb = lane&7), q8 activation block blk_in_row = ss*16 + lane,
// SG-reduce → silu(gate)*up → h_packed[row*E_ffn + n]) — with the Q6_K per-block
// weight read + two-scale fold copied verbatim from down_q6k_q8_gen_impl above
// (the two int8 sub-scales sc0/sc1 per 32-elem sub-block, d6, and the w6 byte
// build ((ql>>ql_shift)&0x0F0F0F0F)|(((qh>>qh_shift)&0x03030303)<<4)). The gate
// and up banks are contracted over H (blocks_per_col = H/256; output col in
// E_ffn). Requires H % 512 == 0 (n_steps exact). x arrives pre-quantized as a
// block_q8_1s stream over the expert-sorted rows (quantize_q8_1s once/layer);
// hdr = {d8, s0, s1}, s0/s1 = d8·Σ of each 16-elem half. copy-not-hoist.
template <bool SOA>
sycl::event gate_up_silu_q6k_q8_impl(sycl::queue& q,
                                     const void* xq8_packed,
                                     const void* gate_W, const void* up_W,
                                     const uint32_t* expert_offsets,
                                     sycl::half* h_packed,
                                     uint32_t E, uint32_t H, uint32_t E_ffn,
                                     uint64_t expert_stride_bytes,
                                     const std::vector<sycl::event>& deps) {
    const uint32_t blocks_per_col = H / 256;             // Q6_K blocks per output col
    const uint32_t q8_per_row     = H / 32;              // q8 blocks per token row
    const uint32_t n_steps        = blocks_per_col / 2;  // needs H % 512 == 0
    const uint32_t n_chunks = E_ffn / N_PER_WG;
    const uint64_t nb_e = uint64_t(blocks_per_col) * E_ffn;  // Q6_K blocks / expert
    const auto* X8 = static_cast<const block_q8_1s*>(xq8_packed);

    return ie::ps(q, "moe_pfl_gate_q8_6k", [&](sycl::handler& h) {
        h.depends_on(deps);
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
            const uint32_t n     = nc * N_PER_WG + sg_id;   // E_ffn output column

            const uint32_t off_start = expert_offsets[e];
            const uint32_t off_end   = expert_offsets[e + 1];
            const uint32_t n_tok     = off_end - off_start;
            if (n_tok == 0) return;

            const auto gv = q6k_wview<SOA>::at(gate_W, expert_stride_bytes, e, nb_e);
            const auto uv = q6k_wview<SOA>::at(up_W,   expert_stride_bytes, e, nb_e);

            // Q6_K lane lattice (mirror down kernel): lane == q8-block within the
            // step; b_in picks which of the step's 2 Q6_K blocks, sb the 32-elem
            // sub-block, (half_q, sub) the ql/qh offsets + shifts + scale pair.
            const int b_in   = int(lane) >> 3;
            const int sb     = int(lane) & 7;
            const int half_q = sb >> 2;
            const int sub    = sb & 3;
            const int ql_off = half_q * 64 + (sub & 1) * 32;
            const int qh_off = half_q * 32;
            const int qh_shift = sub * 2;
            const int ql_shift = (sub & 2) ? 4 : 0;
            const int sc_idx   = half_q * 8 + sub * 2;

            for (uint32_t tk_base = 0; tk_base < n_tok; tk_base += M_TILE) {
                const uint32_t M = sycl::min(uint32_t(M_TILE), n_tok - tk_base);

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

                    const int8_t* gsc = gv.sc_blk(bi);
                    const int8_t* usc = uv.sc_blk(bi);
                    const float g_sc0 = float(gsc[sc_idx]);
                    const float g_sc1 = float(gsc[sc_idx + 1]);
                    const float u_sc0 = float(usc[sc_idx]);
                    const float u_sc1 = float(usc[sc_idx + 1]);
                    const float g_d6 = dev_fp16_to_fp32(gv.d_bits(bi));
                    const float u_d6 = dev_fp16_to_fp32(uv.d_bits(bi));

                    uint32_t gw6[8], uw6[8];
                    {
                        uint32_t gql[8], gqh[8], uql[8], uqh[8];
                        if constexpr (SOA) {
                            const auto gl0 = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(gv.ql_blk(bi) + ql_off);
                            const auto gl1 = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(gv.ql_blk(bi) + ql_off + 16);
                            const auto gh0 = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(gv.qh_blk(bi) + qh_off);
                            const auto gh1 = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(gv.qh_blk(bi) + qh_off + 16);
                            const auto ul0 = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(uv.ql_blk(bi) + ql_off);
                            const auto ul1 = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(uv.ql_blk(bi) + ql_off + 16);
                            const auto uh0 = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(uv.qh_blk(bi) + qh_off);
                            const auto uh1 = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(uv.qh_blk(bi) + qh_off + 16);
                            #pragma unroll
                            for (int w = 0; w < 4; ++w) {
                                gql[w] = gl0[w]; gql[w + 4] = gl1[w];
                                gqh[w] = gh0[w]; gqh[w + 4] = gh1[w];
                                uql[w] = ul0[w]; uql[w + 4] = ul1[w];
                                uqh[w] = uh0[w]; uqh[w + 4] = uh1[w];
                            }
                        } else {
                            const auto* gql16 = reinterpret_cast<const uint16_t*>(gv.ql_blk(bi) + ql_off);
                            const auto* gqh16 = reinterpret_cast<const uint16_t*>(gv.qh_blk(bi) + qh_off);
                            const auto* uql16 = reinterpret_cast<const uint16_t*>(uv.ql_blk(bi) + ql_off);
                            const auto* uqh16 = reinterpret_cast<const uint16_t*>(uv.qh_blk(bi) + qh_off);
                            #pragma unroll
                            for (int w = 0; w < 8; ++w) {
                                gql[w] = uint32_t(gql16[w * 2]) | (uint32_t(gql16[w * 2 + 1]) << 16);
                                gqh[w] = uint32_t(gqh16[w * 2]) | (uint32_t(gqh16[w * 2 + 1]) << 16);
                                uql[w] = uint32_t(uql16[w * 2]) | (uint32_t(uql16[w * 2 + 1]) << 16);
                                uqh[w] = uint32_t(uqh16[w * 2]) | (uint32_t(uqh16[w * 2 + 1]) << 16);
                            }
                        }
                        #pragma unroll
                        for (int w = 0; w < 8; ++w) {
                            gw6[w] = ((gql[w] >> ql_shift) & 0x0F0F0F0Fu) |
                                     (((gqh[w] >> qh_shift) & 0x03030303u) << 4);
                            uw6[w] = ((uql[w] >> ql_shift) & 0x0F0F0F0Fu) |
                                     (((uqh[w] >> qh_shift) & 0x03030303u) << 4);
                        }
                    }

                    // Lane's q8 activation block within the row (step*16 + lane).
                    const uint32_t blk_in_row = ss * 16 + lane;
                    #pragma unroll
                    for (int mm = 0; mm < M_TILE; ++mm) {
                        if (uint32_t(mm) < M) {
                            const uint32_t* blkp =
                                base + (uint32_t(mm) * q8_per_row + blk_in_row) * BW;
                            const auto hdr = *reinterpret_cast<const sycl::vec<float, 4>*>(blkp);
                            const auto q0 = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(blkp + 4);
                            const auto q1 = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(blkp + 8);
                            int32_t gid0 = 0, gid1 = 0, uid0 = 0, uid1 = 0;
                            #pragma unroll
                            for (int w = 0; w < 4; ++w) {
                                gid0 = ie::dp4a_us(gw6[w],     int32_t(q0[w]), gid0);
                                uid0 = ie::dp4a_us(uw6[w],     int32_t(q0[w]), uid0);
                            }
                            #pragma unroll
                            for (int w = 0; w < 4; ++w) {
                                gid1 = ie::dp4a_us(gw6[w + 4], int32_t(q1[w]), gid1);
                                uid1 = ie::dp4a_us(uw6[w + 4], int32_t(q1[w]), uid1);
                            }
                            const float d8 = hdr[0];
                            const float s0 = hdr[1], s1 = hdr[2];
                            g_acc[mm] += g_d6 * (g_sc0 * (d8 * float(gid0) - 32.f * s0) +
                                                 g_sc1 * (d8 * float(gid1) - 32.f * s1));
                            u_acc[mm] += u_d6 * (u_sc0 * (d8 * float(uid0) - 32.f * s0) +
                                                 u_sc1 * (d8 * float(uid1) - 32.f * s1));
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

}  // namespace

sycl::event moe_prefill_down_q6k_q8_gen(sycl::queue& q,
                                        const void* hq8_packed, const void* down_W,
                                        const uint32_t* expert_offsets, sycl::half* out_packed,
                                        uint32_t E, uint32_t H, uint32_t E_ffn,
                                        uint64_t expert_stride_bytes, bool soa,
                                        const std::vector<sycl::event>& deps) {
    return soa ? down_q6k_q8_gen_impl<true >(q, hq8_packed, down_W, expert_offsets,
                     out_packed, E, H, E_ffn, expert_stride_bytes, deps)
               : down_q6k_q8_gen_impl<false>(q, hq8_packed, down_W, expert_offsets,
                     out_packed, E, H, E_ffn, expert_stride_bytes, deps);
}

sycl::event moe_prefill_down_q4k_q8_gen(sycl::queue& q,
                                        const void* hq8_packed, const void* down_W,
                                        const uint32_t* expert_offsets, sycl::half* out_packed,
                                        uint32_t E, uint32_t H, uint32_t E_ffn,
                                        uint64_t expert_stride_bytes, bool soa,
                                        const std::vector<sycl::event>& deps) {
    return soa ? down_q4k_q8_gen_impl<true >(q, hq8_packed, down_W, expert_offsets,
                     out_packed, E, H, E_ffn, expert_stride_bytes, deps)
               : down_q4k_q8_gen_impl<false>(q, hq8_packed, down_W, expert_offsets,
                     out_packed, E, H, E_ffn, expert_stride_bytes, deps);
}

sycl::event moe_prefill_gate_up_silu_q6k_q8(sycl::queue& q,
                                            const void* xq8_packed,
                                            const void* gate_W, const void* up_W,
                                            const uint32_t* expert_offsets,
                                            sycl::half* h_packed,
                                            uint32_t E, uint32_t H, uint32_t E_ffn,
                                            uint64_t expert_stride_bytes, bool soa,
                                            const std::vector<sycl::event>& deps) {
    return soa ? gate_up_silu_q6k_q8_impl<true >(q, xq8_packed, gate_W, up_W,
                     expert_offsets, h_packed, E, H, E_ffn, expert_stride_bytes, deps)
               : gate_up_silu_q6k_q8_impl<false>(q, xq8_packed, gate_W, up_W,
                     expert_offsets, h_packed, E, H, E_ffn, expert_stride_bytes, deps);
}

}  // namespace ie
