// src/ops/moe_q8_down_v2.cpp — P4 B31: moe_prefill_down_q8 v2, the crown split's expert-batched Q8_0 down
// projection (IE_Q8_MOE_DOWN_V2=0 = the v1 kernel in moe_q8.cpp).
//
// Why (b30prof: card 0, T = 2042) v1 was the biggest prefill kernel: 33.6 % of the card, 2.03 TOPS against gate_up's
// 6.95 at twice gate_up's ops per FLOP. Its bmg-g31 ISA (the installed IGC 2.36.5, offline) issues ~270 instructions
// per routed row and subgroup for 8 dp4a (~34 per dp4a; gate_up ~9.6), and that ratio is the measured gap:
//   (1) ie::dp4a_ss's byte arithmetic over vector-loaded words is lowered through <4 x i8> extract/insert: 12 byte
//       movs per operand word, 192 per row (gate_up pays it for its activation words only, 96 per 16 dp4a);
//   (2) wq[PF_MAX_BPL][8] indexed by the runtime s < my_nblk lives in scratch: two private loads per row, and the
//       weight words are re-shuffled every row;
//   (3) the reduce + fold + store of every (row, column) costs ~29 instructions for 8 dp4a.
// v2: the native integer dot (no byte movs), gate_up's structure (the lane's weight block loaded into registers per
// tile, the runtime block loop outermost so each accumulator stays loop-carried) and V2_NB columns per subgroup (one
// SLM read of an activation block feeds V2_NB columns; one store per row).
//
// BIT-EXACT vs v1: per (row, column) every lane forms the same partial -- acc = 0, then acc += wd*d8*float(idot) over
// its blocks j = lane, lane+16, ... in ascending order (the integer dot is exact) -- the same reduce_over_group sums
// the 16 lanes and the same half(w * r) is stored. tests/unit/moe_q8_down_v2_test.cpp checks it bit for bit.
// ESIMD-safe: plain SLM + dp4a, no block2d / lsc paths.

#include "ie/moe_q8.hpp"

#include "ie/dp4a.hpp"
#include "ie/kernel_profiler.hpp"
#include "ie/quant_blocks.hpp"

#if defined(__SYCL_DEVICE_ONLY__) && !defined(IE_Q8_DOWN_V2_PORTABLE)
// IGC's integer-dot builtin (what its own OpenCL dot-product builtins lower to): GenISA.dp4a.ss(c, a, b, sat) on the
// packed words. IGC-only; -DIE_Q8_DOWN_V2_PORTABLE=1 builds the plain byte arithmetic for another SYCL device.
extern "C" SYCL_EXTERNAL int __builtin_IB_dp4a_ss(int c, int a, int b, bool sat);
#endif

namespace ie {
namespace {

// (s8x4 · s8x4) + c, exact: the same integer as ie::dp4a_ss.
inline int32_t v2_dp4a_ss(uint32_t a, uint32_t b, int32_t c) {
#if defined(__SYCL_DEVICE_ONLY__) && !defined(IE_Q8_DOWN_V2_PORTABLE)
    return __builtin_IB_dp4a_ss(c, int32_t(a), int32_t(b), false);
#else
    return ie::dp4a_ss(int32_t(a), int32_t(b), c);
#endif
}

// fp16→fp32 (copy-not-hoist, same as moe_q8.cpp).
inline float v2_fp16_to_fp32(uint16_t h) {
    const uint32_t s = uint32_t(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1fu;
    uint32_t m =  h        & 0x3ffu;
    if (e == 0) {
        if (m == 0) return sycl::bit_cast<float>(s);
        while ((m & 0x400u) == 0) { m <<= 1; e -= 1; }
        e += 1; m &= ~0x400u;
    } else if (e == 31) {
        return sycl::bit_cast<float>(s | 0x7f800000u | (m << 13));
    }
    e += (127 - 15);
    return sycl::bit_cast<float>(s | (e << 23) | (m << 13));
}

constexpr int      V2_SG  = 16;                          // lanes: K-blocks j = lane, lane+16, ... (as v1)
constexpr int      V2_SGS = 32;                          // subgroups per work-group
constexpr int      V2_NB  = 4;                           // output columns per subgroup
constexpr int      V2_TM  = 8;                           // routed rows per staged tile
constexpr int      V2_WG  = V2_SG * V2_SGS;              // 512
constexpr uint32_t V2_BW  = sizeof(block_q8_1x) / 4;     // 10 words / activation block: word 0 = d, 2..9 = qs

}  // namespace

sycl::event moe_prefill_down_q8_v2(sycl::queue& q, const void* hq8_packed,
                                   const int8_t* d_qs, const uint16_t* d_d,
                                   uint64_t qs_stride, uint64_t d_stride,
                                   const uint32_t* expert_offsets,
                                   const sycl::half* sorted_w,
                                   sycl::half* out_packed,
                                   uint32_t E, uint32_t H, uint32_t E_ffn,
                                   const std::vector<sycl::event>& deps) {
    const uint32_t q8_per_row     = E_ffn / 32;           // activation blocks / row == weight blocks / column
    const uint32_t n_blk_per_lane = (q8_per_row + V2_SG - 1) / V2_SG;
    const uint32_t cols_per_wg    = uint32_t(V2_SGS * V2_NB);
    const uint32_t n_chunks       = (H + cols_per_wg - 1) / cols_per_wg;
    const auto* X8 = static_cast<const block_q8_1x*>(hq8_packed);
    return ie::ps(q, "moe_pfl_down_q8_v2", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<uint32_t, 1> sh(uint64_t(V2_TM) * q8_per_row * V2_BW, h);
        h.parallel_for(sycl::nd_range<2>({uint64_t(E) * n_chunks, V2_WG}, {1, V2_WG}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(V2_SG)]] {
            const uint32_t en = uint32_t(it.get_group(0));
            const uint32_t e  = en / n_chunks;
            const uint32_t nc = en % n_chunks;
            const uint32_t lid  = uint32_t(it.get_local_id(1));
            const uint32_t sg = lid / V2_SG, lane = lid % V2_SG;
            const uint32_t n0 = nc * cols_per_wg + sg * V2_NB;      // this subgroup's columns n0 .. n0+V2_NB-1
            const uint32_t off_start = expert_offsets[e];
            const uint32_t n_tok     = expert_offsets[e + 1] - off_start;
            if (n_tok == 0) return;                                  // an empty expert: the whole work-group leaves

            const int8_t*   wcol = d_qs + e * qs_stride + uint64_t(n0) * E_ffn;
            const uint16_t* wsc  = d_d  + e * d_stride  + uint64_t(n0) * q8_per_row;
            const uint32_t* base = sh.get_multi_ptr<sycl::access::decorated::no>().get();

            for (uint32_t tk = 0; tk < n_tok; tk += V2_TM) {
                const uint32_t M = sycl::min(uint32_t(V2_TM), n_tok - tk);
                const uint32_t* src = reinterpret_cast<const uint32_t*>(
                    X8 + uint64_t(off_start + tk) * q8_per_row);
                for (uint32_t i = lid; i < M * q8_per_row * V2_BW; i += V2_WG) sh[i] = src[i];
                sycl::group_barrier(it.get_group());
                if (n0 < H) {                                        // subgroup-uniform
                    float acc[V2_NB][V2_TM];
                    #pragma unroll
                    for (int c = 0; c < V2_NB; ++c) {
                        #pragma unroll
                        for (int mm = 0; mm < V2_TM; ++mm) acc[c][mm] = 0.f;
                    }
                    for (uint32_t s = 0; s < n_blk_per_lane; ++s) {
                        const uint32_t j = lane + s * V2_SG;
                        if (j >= q8_per_row) break;
                        uint32_t wq[V2_NB][8];
                        float    wd[V2_NB];
                        #pragma unroll
                        for (int c = 0; c < V2_NB; ++c) {
                            // a column past H reads column n0 instead; it is never stored
                            const uint64_t cc = (n0 + uint32_t(c) < H) ? uint64_t(c) : 0u;
                            const uint32_t* wp = reinterpret_cast<const uint32_t*>(
                                wcol + cc * E_ffn + uint64_t(j) * 32);
                            #pragma unroll
                            for (int w = 0; w < 8; ++w) wq[c][w] = wp[w];
                            wd[c] = v2_fp16_to_fp32(wsc[cc * q8_per_row + j]);
                        }
                        #pragma unroll
                        for (int mm = 0; mm < V2_TM; ++mm) {
                            if (uint32_t(mm) < M) {
                                const uint32_t* blk = base + (uint32_t(mm) * q8_per_row + j) * V2_BW;
                                const float d8 = sycl::bit_cast<float>(blk[0]);
                                uint32_t xw[8];
                                #pragma unroll
                                for (int w = 0; w < 8; ++w) xw[w] = blk[2 + w];
                                #pragma unroll
                                for (int c = 0; c < V2_NB; ++c) {
                                    int32_t idot = 0;
                                    #pragma unroll
                                    for (int w = 0; w < 8; ++w) idot = v2_dp4a_ss(wq[c][w], xw[w], idot);
                                    acc[c][mm] += wd[c] * d8 * float(idot);
                                }
                            }
                        }
                    }
                    #pragma unroll
                    for (int mm = 0; mm < V2_TM; ++mm) {
                        if (uint32_t(mm) < M) {
                            float r_mine = 0.f;                      // lane c keeps column n0 + c's sum
                            #pragma unroll
                            for (int c = 0; c < V2_NB; ++c) {
                                const float r = sycl::reduce_over_group(it.get_sub_group(), acc[c][mm],
                                                                        sycl::plus<float>());
                                if (lane == uint32_t(c)) r_mine = r;
                            }
                            const uint64_t row = uint64_t(off_start + tk + mm);
                            if (lane < uint32_t(V2_NB) && n0 + lane < H)
                                out_packed[row * H + n0 + lane] = sycl::half(float(sorted_w[row]) * r_mine);
                        }
                    }
                }
                sycl::group_barrier(it.get_group());
            }
        });
    });
}

}  // namespace ie
