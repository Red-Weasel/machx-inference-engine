// src/ops/moe_gemma4.cpp — Gemma 4 fused-MoE helper kernels (additive,
// gemma4-owned; the crown's moe_fused.cpp is untouched).
//
// M0 (batched-fp16): expert-sorted gather → ONE batched gemm_q4_0 per expert
// for gate_up and down → scatter (moe_prefill_reduce). The only new primitive
// is geglu_rows (row-wise GeGLU over the interleaved [n_rows, 2*EF] buffer).
//
// M1 (int-dot W4A8): moe_prefill_proj_q4_0_q8 — a SINGLE-launch int-dot
// projection y[TK,N] = q8(x)[TK,K] @ Q4_0 W[K,N] over ALL experts (indexed by
// expert_offsets), the Q4_0 analog of qwen3moe's moe_prefill_down_q4k_q8_gen.
// Used for BOTH gate_up (K=H, N=2*EF) and down (K=EF, N=H); geglu_rows runs
// between. Q4_0 is simpler than Q4_K: one fp16 scale per 32-elem block, no
// sub-scales/min, symmetric w = d*(nib-8), so for a q8_1s activation block
//   Σ d*(nib-8)*(d8*q) = d*(d8*Σ nib*q - 8*d8*Σ q) = d*(d8*idot - 8*(s0+s1)).
// ESIMD-safe (plain SLM + aligned vec loads on the 16B-aligned q8 stream;
// the 2-aligned Q4_0 weight is read byte-wise once per (expert,column)).

#include "ie/dp4a.hpp"
#include "ie/kernel_profiler.hpp"
#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"

#include <sycl/sycl.hpp>

namespace ie {

namespace {
// Same fp16→fp32 helper as gemv_q*k.cpp / moe_qwen3.cpp (copy-not-hoist).
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

constexpr int SG_SIZE  = 16;
constexpr int N_PER_WG = 32;
constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;        // 512
constexpr int M_TILE   = 8;                         // keeps SLM ≤ ~34 KB at K=H
constexpr uint32_t BW  = sizeof(block_q8_1s) / 4;   // 12 words / q8 block
constexpr int MAX_BPL  = 6;                          // K up to 6*16*32 = 3072 (≥ H)
}  // namespace

sycl::event geglu_rows(sycl::queue& q,
                       const sycl::half* gu_packed,   // [n_rows, 2*EF]
                       sycl::half* y,                 // [n_rows, EF]
                       uint32_t n_rows, uint32_t EF, bool swap,
                       const std::vector<sycl::event>& deps) {
    return ie::ps(q, "geglu_rows", [&](sycl::handler& h) {
        h.depends_on(deps);
        constexpr uint64_t WG = 256;
        constexpr float kSqrt2OverPi = 0.7978845608028654f;
        const uint64_t n = uint64_t(n_rows) * EF;
        const uint64_t global = ((n + WG - 1) / WG) * WG;
        const uint64_t ef = EF;
        const uint64_t gate_off = swap ? ef : 0;   // gate = second/first half
        const uint64_t up_off   = swap ? 0  : ef;
        h.parallel_for(sycl::nd_range<1>(global, WG),
                       [=](sycl::nd_item<1> it) {
            const uint64_t i = it.get_global_id(0);
            if (i >= n) return;
            const uint64_t r = i / ef;
            const uint64_t j = i % ef;
            const uint64_t base = r * (2 * ef);
            const float g = float(gu_packed[base + gate_off + j]);
            const float u = float(gu_packed[base + up_off + j]);
            const float inner = kSqrt2OverPi * (g + 0.044715f * g * g * g);
            const float gelu = 0.5f * g * (1.f + sycl::tanh(inner));
            y[i] = sycl::half(gelu * u);
        });
    });
}

// y[TK,N] = q8(x)[TK,K] @ Q4_0 W[K,N], per expert (expert_offsets[E+1]).
// W is column-packed: output column n's K weights are K/32 contiguous Q4_0
// blocks at byte offset n*(K/32)*18 within the expert's region (same layout
// gemv_q4_0 consumes). One WG per (expert, N-chunk of N_PER_WG columns).
sycl::event moe_prefill_proj_q4_0_q8(sycl::queue& q,
                                     const void* xq8_packed,   // block_q8_1s [TK, K/32]
                                     const void* W_q4_0,       // Q4_0 [K, N] per expert
                                     const uint32_t* expert_offsets,
                                     sycl::half* out_packed,    // [TK, N]
                                     uint32_t E, uint32_t K, uint32_t N,
                                     uint64_t expert_stride_bytes,
                                     const std::vector<sycl::event>& deps) {
    const uint32_t blocks_per_col = K / 32;
    const uint32_t q8_per_row     = K / 32;
    const uint32_t n_blk_per_lane = (q8_per_row + SG_SIZE - 1) / SG_SIZE;
    const uint32_t n_chunks = (N + N_PER_WG - 1) / N_PER_WG;
    const auto* X8 = static_cast<const block_q8_1s*>(xq8_packed);
    const auto* WB = static_cast<const uint8_t*>(W_q4_0);

    return ie::ps(q, "moe_pfl_proj_q4_0_q8", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<uint32_t, 1> sx(uint64_t(M_TILE) * q8_per_row * BW, h);

        h.parallel_for(sycl::nd_range<2>({uint64_t(E) * n_chunks, WG_ITEMS},
                                          {1, WG_ITEMS}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t en = uint32_t(it.get_group(0));
            const uint32_t e  = en / n_chunks;
            const uint32_t nc = en % n_chunks;
            const uint32_t lid = uint32_t(it.get_local_id(1));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t nn    = nc * N_PER_WG + sg_id;   // output column

            const uint32_t off_start = expert_offsets[e];
            const uint32_t n_tok     = expert_offsets[e + 1] - off_start;
            if (n_tok == 0) return;

            const uint8_t* w_e = WB + uint64_t(e) * expert_stride_bytes;

            // Register-resident lo/hi nibble words + scale for each block this
            // lane owns; assembled byte-wise (Q4_0 blocks are only 2-aligned).
            uint32_t wlo[MAX_BPL][4], whi[MAX_BPL][4];
            float    d4[MAX_BPL];
            uint32_t my_j[MAX_BPL];
            uint32_t my_nblk = 0;
            if (nn < N) {
                for (uint32_t s = 0; s < n_blk_per_lane; ++s) {
                    const uint32_t j = lane + s * SG_SIZE;
                    if (j >= q8_per_row) break;
                    const uint64_t bi = uint64_t(nn) * blocks_per_col + j;
                    const uint8_t* bp = w_e + bi * 18;       // block_q4_0 = 18 bytes
                    const uint16_t db = uint16_t(bp[0]) | (uint16_t(bp[1]) << 8);
                    d4[my_nblk] = dev_fp16_to_fp32(db);
                    const uint8_t* qs = bp + 2;
                    #pragma unroll
                    for (int w = 0; w < 4; ++w) {
                        const uint32_t qw = uint32_t(qs[4 * w])
                                          | (uint32_t(qs[4 * w + 1]) << 8)
                                          | (uint32_t(qs[4 * w + 2]) << 16)
                                          | (uint32_t(qs[4 * w + 3]) << 24);
                        wlo[my_nblk][w] = qw & 0x0F0F0F0Fu;
                        whi[my_nblk][w] = (qw >> 4) & 0x0F0F0F0Fu;
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
                    sx[i] = src[i];
                sycl::group_barrier(it.get_group());

                if (nn >= N) { sycl::group_barrier(it.get_group()); continue; }

                const uint32_t* base =
                    sx.get_multi_ptr<sycl::access::decorated::no>().get();
                float acc[M_TILE];
                #pragma unroll
                for (int mm = 0; mm < M_TILE; ++mm) {
                    acc[mm] = 0.f;
                    if (uint32_t(mm) < M) {
                        for (uint32_t s = 0; s < my_nblk; ++s) {
                            const uint32_t* blkp =
                                base + (uint32_t(mm) * q8_per_row + my_j[s]) * BW;
                            const auto hdr = *reinterpret_cast<const sycl::vec<float, 4>*>(blkp);
                            const auto q0  = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(blkp + 4);
                            const auto q1  = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(blkp + 8);
                            int32_t idot = 0;
                            #pragma unroll
                            for (int w = 0; w < 4; ++w)
                                idot = ie::dp4a_us(wlo[s][w], int32_t(q0[w]), idot);
                            #pragma unroll
                            for (int w = 0; w < 4; ++w)
                                idot = ie::dp4a_us(whi[s][w], int32_t(q1[w]), idot);
                            acc[mm] += d4[s] * (hdr[0] * float(idot) - 8.f * (hdr[1] + hdr[2]));
                        }
                    }
                }

                #pragma unroll
                for (int mm = 0; mm < M_TILE; ++mm) {
                    if (uint32_t(mm) < M) {
                        const float r = sycl::reduce_over_group(it.get_sub_group(),
                                                                acc[mm], sycl::plus<float>());
                        if (lane == 0)
                            out_packed[(uint64_t(off_start) + tk_base + mm) * N + nn] = sycl::half(r);
                    }
                }
                sycl::group_barrier(it.get_group());
            }
        });
    });
}

// SoA-Q4_0 variant of moe_prefill_proj_q4_0_q8: identical single-launch int-dot
// over all experts, but the weight is read from the SoA streams (per-expert
// contiguous nibble bank + fp16-scale bank) instead of the 18-byte AoS blocks —
// 16-byte lane-coalesced nibble reads (vs misaligned 18 B) raise BW, the win at
// both prefill (T-batched) and decode (T=1, the 8-active-expert reads). The
// expert banks store SoA INSTEAD of AoS (no memory doubling). Per expert the
// weight is [K,N] column-packed: column n's K/2 nibble bytes + K/32 fp16 scales
// are contiguous; block (col nn, k-block j) → nibbles at qs_e+(nn*bpc+j)*16,
// scale at d_e[nn*bpc+j]. Fold identical: d4*(d8*idot − 8*(s0+s1)).
sycl::event moe_prefill_proj_q4_0_soa_q8(sycl::queue& q,
                                         const void* xq8_packed,    // block_q8_1s [TK, K/32]
                                         const uint8_t* qs_bank,    // [E][N*(K/2)]
                                         const uint16_t* d_bank,    // [E][N*(K/32)]
                                         const uint32_t* expert_offsets,
                                         sycl::half* out_packed,    // [TK, N]
                                         uint32_t E, uint32_t K, uint32_t N,
                                         uint64_t qs_stride, uint64_t d_stride,
                                         const std::vector<sycl::event>& deps) {
    const uint32_t blocks_per_col = K / 32;
    const uint32_t q8_per_row     = K / 32;
    const uint32_t n_blk_per_lane = (q8_per_row + SG_SIZE - 1) / SG_SIZE;
    const uint32_t n_chunks = (N + N_PER_WG - 1) / N_PER_WG;
    const auto* X8 = static_cast<const block_q8_1s*>(xq8_packed);

    return ie::ps(q, "moe_pfl_proj_q4_0_soa", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<uint32_t, 1> sx(uint64_t(M_TILE) * q8_per_row * BW, h);

        h.parallel_for(sycl::nd_range<2>({uint64_t(E) * n_chunks, WG_ITEMS},
                                          {1, WG_ITEMS}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t en = uint32_t(it.get_group(0));
            const uint32_t e  = en / n_chunks;
            const uint32_t nc = en % n_chunks;
            const uint32_t lid = uint32_t(it.get_local_id(1));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t nn    = nc * N_PER_WG + sg_id;

            const uint32_t off_start = expert_offsets[e];
            const uint32_t n_tok     = expert_offsets[e + 1] - off_start;
            if (n_tok == 0) return;

            const uint8_t*  qs_e = qs_bank + uint64_t(e) * qs_stride;
            const uint16_t* d_e  = d_bank  + uint64_t(e) * d_stride;

            uint32_t wlo[MAX_BPL][4], whi[MAX_BPL][4];
            float    d4[MAX_BPL];
            uint32_t my_j[MAX_BPL];
            uint32_t my_nblk = 0;
            if (nn < N) {
                for (uint32_t s = 0; s < n_blk_per_lane; ++s) {
                    const uint32_t j = lane + s * SG_SIZE;
                    if (j >= q8_per_row) break;
                    const uint64_t bi = uint64_t(nn) * blocks_per_col + j;
                    d4[my_nblk] = dev_fp16_to_fp32(d_e[bi]);
                    const uint32_t* qw = reinterpret_cast<const uint32_t*>(qs_e + bi * 16);
                    #pragma unroll
                    for (int w = 0; w < 4; ++w) {
                        const uint32_t v = qw[w];
                        wlo[my_nblk][w] = v & 0x0F0F0F0Fu;
                        whi[my_nblk][w] = (v >> 4) & 0x0F0F0F0Fu;
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
                    sx[i] = src[i];
                sycl::group_barrier(it.get_group());

                if (nn >= N) { sycl::group_barrier(it.get_group()); continue; }

                const uint32_t* base =
                    sx.get_multi_ptr<sycl::access::decorated::no>().get();
                float acc[M_TILE];
                #pragma unroll
                for (int mm = 0; mm < M_TILE; ++mm) {
                    acc[mm] = 0.f;
                    if (uint32_t(mm) < M) {
                        for (uint32_t s = 0; s < my_nblk; ++s) {
                            const uint32_t* blkp =
                                base + (uint32_t(mm) * q8_per_row + my_j[s]) * BW;
                            const auto hdr = *reinterpret_cast<const sycl::vec<float, 4>*>(blkp);
                            const auto q0  = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(blkp + 4);
                            const auto q1  = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(blkp + 8);
                            int32_t idot = 0;
                            #pragma unroll
                            for (int w = 0; w < 4; ++w)
                                idot = ie::dp4a_us(wlo[s][w], int32_t(q0[w]), idot);
                            #pragma unroll
                            for (int w = 0; w < 4; ++w)
                                idot = ie::dp4a_us(whi[s][w], int32_t(q1[w]), idot);
                            acc[mm] += d4[s] * (hdr[0] * float(idot) - 8.f * (hdr[1] + hdr[2]));
                        }
                    }
                }
                #pragma unroll
                for (int mm = 0; mm < M_TILE; ++mm) {
                    if (uint32_t(mm) < M) {
                        const float r = sycl::reduce_over_group(it.get_sub_group(),
                                                                acc[mm], sycl::plus<float>());
                        if (lane == 0)
                            out_packed[(uint64_t(off_start) + tk_base + mm) * N + nn] = sycl::half(r);
                    }
                }
                sycl::group_barrier(it.get_group());
            }
        });
    });
}

}  // namespace ie
