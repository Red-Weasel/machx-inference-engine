// src/ops/moe_q8.cpp — fully GPU-resident Q8_0-expert MoE kernels (crown split).
// See header. Int-dot (dp4a_ss) over the block_q8_1x activation stream, structure
// lifted VERBATIM from gemv_q8_0_soa_q8 (gemv_q8dot.cpp): SG owns one output column,
// lanes stride 32-elem blocks, dp4a over 8 words/block, acc = d_W·d_act·idot,
// SG-reduce. The ONLY additions vs that kernel: (1) the expert id is read per
// token-slot from device topk_idx (NO host pull), (2) gate+up are fused with silu,
// (3) down folds the topk weight. ESIMD-safe: plain SLM + dp4a, no block2d/lsc.
//
// Phase-1 layout: NOT the sorted-expert-offset grouped layout (that's the further
// perf lever); this is one WG per (token-slot, column-chunk) reading the expert from
// topk_idx. Fully on-GPU — removes the host loop. block_q8_1x activation is produced
// by per-row quantize_q8_1 enqueues (in-order queue, no host sync).

#include "ie/moe_q8.hpp"

#include "ie/dp4a.hpp"
#include "ie/kernel_profiler.hpp"
#include "ie/quant_blocks.hpp"

namespace ie {
namespace {

// fp16→fp32 (copy-not-hoist, same as gemv_q8dot.cpp / moe_qwen3.cpp).
inline float dev_fp16_to_fp32(uint16_t h) {
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

constexpr int SG_SIZE  = 16;
constexpr int N_PER_WG = 32;
constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;   // 512

}  // namespace

sycl::event moe_gate_up_silu_q8(sycl::queue& q, const void* x_q8,
                                const int8_t* g_qs, const uint16_t* g_d,
                                const int8_t* u_qs, const uint16_t* u_d,
                                uint64_t qs_stride, uint64_t d_stride,
                                const int32_t* topk_idx, sycl::half* h_out,
                                uint32_t T, uint32_t K, uint32_t H, uint32_t E_ffn) {
    const uint32_t bpc = H / 32;                          // act blocks per row
    const uint32_t col_wgs = (E_ffn + N_PER_WG - 1) / N_PER_WG;
    const uint64_t n_wgs = uint64_t(T) * K * col_wgs;
    const auto* X8base = static_cast<const block_q8_1x*>(x_q8);
    return ie::ps(q, "moe_gate_up_silu_q8", [&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> q8s(bpc * 8, h);
        sycl::local_accessor<float, 1>    q8d(bpc, h);
        h.parallel_for(sycl::nd_range<1>(n_wgs * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid  = uint32_t(it.get_local_id(0));
            const uint32_t wgid = uint32_t(it.get_group(0));
            const uint32_t tk = uint32_t(wgid / col_wgs), cw = uint32_t(wgid % col_wgs);
            const uint32_t t  = tk / K;
            const uint32_t sg = lid / SG_SIZE, lane = lid % SG_SIZE;
            const uint32_t n  = cw * N_PER_WG + sg;
            const block_q8_1x* X8 = X8base + uint64_t(t) * bpc;
            for (uint32_t i = lid; i < bpc * 8; i += WG_ITEMS) {
                const uint32_t b = i / 8, w = i % 8;
                q8s[i] = reinterpret_cast<const uint32_t*>(X8[b].qs)[w];
            }
            for (uint32_t i = lid; i < bpc; i += WG_ITEMS) q8d[i] = X8[i].d;
            sycl::group_barrier(it.get_group());
            if (n >= E_ffn) return;
            const uint32_t e = uint32_t(topk_idx[tk]);
            const int8_t*   gw = g_qs + e * qs_stride + uint64_t(n) * H;
            const int8_t*   uw = u_qs + e * qs_stride + uint64_t(n) * H;
            const uint16_t* gd = g_d  + e * d_stride  + uint64_t(n) * bpc;
            const uint16_t* ud = u_d  + e * d_stride  + uint64_t(n) * bpc;
            float ga = 0.f, ua = 0.f;
            for (uint32_t b = lane; b < bpc; b += SG_SIZE) {
                const uint32_t* gq = reinterpret_cast<const uint32_t*>(gw + uint64_t(b) * 32);
                const uint32_t* uq = reinterpret_cast<const uint32_t*>(uw + uint64_t(b) * 32);
                int32_t gi = 0, ui = 0;
                #pragma unroll
                for (int w = 0; w < 8; ++w) {
                    gi = ie::dp4a_ss(int32_t(gq[w]), int32_t(q8s[b * 8 + w]), gi);
                    ui = ie::dp4a_ss(int32_t(uq[w]), int32_t(q8s[b * 8 + w]), ui);
                }
                ga += dev_fp16_to_fp32(gd[b]) * q8d[b] * float(gi);
                ua += dev_fp16_to_fp32(ud[b]) * q8d[b] * float(ui);
            }
            ga = sycl::reduce_over_group(it.get_sub_group(), ga, sycl::plus<float>());
            ua = sycl::reduce_over_group(it.get_sub_group(), ua, sycl::plus<float>());
            if (lane == 0) {
                const float s = ga / (1.f + sycl::exp(-ga));   // silu(gate)
                h_out[uint64_t(tk) * E_ffn + n] = sycl::half(s * ua);
            }
        });
    });
}

sycl::event moe_down_q8(sycl::queue& q, const void* h_q8,
                        const int8_t* d_qs, const uint16_t* d_d,
                        uint64_t qs_stride, uint64_t d_stride,
                        const int32_t* topk_idx, const sycl::half* topk_w,
                        sycl::half* y_packed, uint32_t T, uint32_t K,
                        uint32_t E_ffn, uint32_t H) {
    const uint32_t bpc = E_ffn / 32;
    const uint32_t col_wgs = (H + N_PER_WG - 1) / N_PER_WG;
    const uint64_t n_wgs = uint64_t(T) * K * col_wgs;
    const auto* X8base = static_cast<const block_q8_1x*>(h_q8);
    return ie::ps(q, "moe_down_q8", [&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> q8s(bpc * 8, h);
        sycl::local_accessor<float, 1>    q8d(bpc, h);
        h.parallel_for(sycl::nd_range<1>(n_wgs * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid  = uint32_t(it.get_local_id(0));
            const uint32_t wgid = uint32_t(it.get_group(0));
            const uint32_t tk = uint32_t(wgid / col_wgs), cw = uint32_t(wgid % col_wgs);
            const uint32_t sg = lid / SG_SIZE, lane = lid % SG_SIZE;
            const uint32_t n  = cw * N_PER_WG + sg;
            const block_q8_1x* X8 = X8base + uint64_t(tk) * bpc;
            for (uint32_t i = lid; i < bpc * 8; i += WG_ITEMS) {
                const uint32_t b = i / 8, w = i % 8;
                q8s[i] = reinterpret_cast<const uint32_t*>(X8[b].qs)[w];
            }
            for (uint32_t i = lid; i < bpc; i += WG_ITEMS) q8d[i] = X8[i].d;
            sycl::group_barrier(it.get_group());
            if (n >= H) return;
            const uint32_t e = uint32_t(topk_idx[tk]);
            const int8_t*   ww = d_qs + e * qs_stride + uint64_t(n) * E_ffn;
            const uint16_t* wd = d_d  + e * d_stride  + uint64_t(n) * bpc;
            float acc = 0.f;
            for (uint32_t b = lane; b < bpc; b += SG_SIZE) {
                const uint32_t* wq = reinterpret_cast<const uint32_t*>(ww + uint64_t(b) * 32);
                int32_t idot = 0;
                #pragma unroll
                for (int w = 0; w < 8; ++w)
                    idot = ie::dp4a_ss(int32_t(wq[w]), int32_t(q8s[b * 8 + w]), idot);
                acc += dev_fp16_to_fp32(wd[b]) * q8d[b] * float(idot);
            }
            acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
            if (lane == 0)
                y_packed[uint64_t(tk) * H + n] = sycl::half(float(topk_w[tk]) * acc);
        });
    });
}

sycl::event moe_reduce_q8(sycl::queue& q, const sycl::half* y_packed, sycl::half* y,
                          uint32_t T, uint32_t K, uint32_t H) {
    return q.parallel_for(sycl::range<1>(uint64_t(T) * H), [=](sycl::id<1> i) {
        const uint32_t t = uint32_t(uint64_t(i) / H), n = uint32_t(uint64_t(i) % H);
        float s = 0.f;
        for (uint32_t k = 0; k < K; ++k)
            s += float(y_packed[(uint64_t(t) * K + k) * H + n]);
        y[i] = sycl::half(s);
    });
}

// ===========================================================================
// EXPERT-BATCHED Q8_0 MoE PREFILL (T>1) — weight-stationary GEMM analog.
//
// The decode kernels above are one WG per (token-slot, col-chunk): at T=512 the
// TK=T·K=4096 slots re-read each of the 256 experts' Q8 weights ~16× (TK/E),
// turning the MoE into a BW-replay of the weights and dominating prefill
// (~80%). These kernels invert the parallelism — one WG per (expert, col-chunk)
// that loops the expert's M routed token rows in PF_M_TILE tiles, staging the
// tile's block_q8_1x activation into SLM and reusing each weight column across
// the tile. Each expert's weights are read ~⌈n_tok/PF_M_TILE⌉ ≈ 1× instead.
//
// Numerics are BIT-IDENTICAL to moe_gate_up_silu_q8/moe_down_q8: same
// block_q8_1x activation (quantize_q8_1), same per-block dp4a_ss, same ascending
// lane→block accumulation order (lane, lane+16, …), same SG-reduce, and the
// routing weight is folded at down (half(w·acc)) exactly as the decode path —
// so a clean A/B holds PPL exactly. ESIMD-safe: plain SLM + dp4a, no block2d.
//
// Activation block stride in words (block_q8_1x = {f32 d; f32 s; i8 qs[32]}):
//   word 0 = d (block scale), words 2..9 = qs (32 int8). s (word 1) is unused.
namespace {
constexpr int      PF_M_TILE  = 16;
constexpr int      PF_MAX_BPL = 4;                      // ⌈(H/32)/SG_SIZE⌉ for H≤2048
constexpr uint32_t Q8X_BW     = sizeof(block_q8_1x) / 4;   // 10 words / activation block
}  // namespace

sycl::event moe_prefill_gate_up_silu_q8(sycl::queue& q, const void* xq8_packed,
                                        const int8_t* g_qs, const uint16_t* g_d,
                                        const int8_t* u_qs, const uint16_t* u_d,
                                        uint64_t qs_stride, uint64_t d_stride,
                                        const uint32_t* expert_offsets,
                                        sycl::half* h_packed,
                                        uint32_t E, uint32_t H, uint32_t E_ffn,
                                        const std::vector<sycl::event>& deps) {
    const uint32_t bpc        = H / 32;                       // weight blocks / col
    const uint32_t q8_per_row = H / 32;                       // activation blocks / row
    const uint32_t n_blk_per_lane = (q8_per_row + SG_SIZE - 1) / SG_SIZE;
    const uint32_t n_chunks   = (E_ffn + N_PER_WG - 1) / N_PER_WG;
    const auto* X8 = static_cast<const block_q8_1x*>(xq8_packed);
    return ie::ps(q, "moe_pfl_gate_up_q8", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<uint32_t, 1> sx(uint64_t(PF_M_TILE) * q8_per_row * Q8X_BW, h);
        h.parallel_for(sycl::nd_range<2>({uint64_t(E) * n_chunks, WG_ITEMS}, {1, WG_ITEMS}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t en = uint32_t(it.get_group(0));
            const uint32_t e  = en / n_chunks;
            const uint32_t nc = en % n_chunks;
            const uint32_t lid  = uint32_t(it.get_local_id(1));
            const uint32_t sg = lid / SG_SIZE, lane = lid % SG_SIZE;
            const uint32_t n  = nc * N_PER_WG + sg;
            const uint32_t off_start = expert_offsets[e];
            const uint32_t n_tok     = expert_offsets[e + 1] - off_start;
            if (n_tok == 0) return;

            // Column-n weight bases (read inside the tile loop; Q8_0 is cheap +
            // L2-hot, so streaming beats the register pressure of pre-loading
            // gate AND up for all n_blk_per_lane blocks).
            const int8_t*   gcol = g_qs + e * qs_stride + uint64_t(n) * H;
            const int8_t*   ucol = u_qs + e * qs_stride + uint64_t(n) * H;
            const uint16_t* gsc  = g_d  + e * d_stride  + uint64_t(n) * bpc;
            const uint16_t* usc  = u_d  + e * d_stride  + uint64_t(n) * bpc;

            for (uint32_t tk = 0; tk < n_tok; tk += PF_M_TILE) {
                const uint32_t M = sycl::min(uint32_t(PF_M_TILE), n_tok - tk);
                const uint32_t* src = reinterpret_cast<const uint32_t*>(
                    X8 + uint64_t(off_start + tk) * q8_per_row);
                for (uint32_t i = lid; i < M * q8_per_row * Q8X_BW; i += WG_ITEMS) sx[i] = src[i];
                sycl::group_barrier(it.get_group());
                if (n >= E_ffn) { sycl::group_barrier(it.get_group()); continue; }

                const uint32_t* base = sx.get_multi_ptr<sycl::access::decorated::no>().get();
                float gacc[PF_M_TILE], uacc[PF_M_TILE];
                #pragma unroll
                for (int mm = 0; mm < PF_M_TILE; ++mm) { gacc[mm] = 0.f; uacc[mm] = 0.f; }

                for (uint32_t s = 0; s < n_blk_per_lane; ++s) {
                    const uint32_t j = lane + s * SG_SIZE;
                    if (j >= q8_per_row) break;
                    const uint32_t* gp = reinterpret_cast<const uint32_t*>(gcol + uint64_t(j) * 32);
                    const uint32_t* up = reinterpret_cast<const uint32_t*>(ucol + uint64_t(j) * 32);
                    uint32_t gw[8], uw[8];
                    #pragma unroll
                    for (int w = 0; w < 8; ++w) { gw[w] = gp[w]; uw[w] = up[w]; }
                    const float gd = dev_fp16_to_fp32(gsc[j]);
                    const float ud = dev_fp16_to_fp32(usc[j]);
                    #pragma unroll
                    for (int mm = 0; mm < PF_M_TILE; ++mm) {
                        if (uint32_t(mm) < M) {
                            const uint32_t* blk = base + (uint32_t(mm) * q8_per_row + j) * Q8X_BW;
                            const float d8 = sycl::bit_cast<float>(blk[0]);
                            int32_t gi = 0, ui = 0;
                            #pragma unroll
                            for (int w = 0; w < 8; ++w) {
                                gi = ie::dp4a_ss(int32_t(gw[w]), int32_t(blk[2 + w]), gi);
                                ui = ie::dp4a_ss(int32_t(uw[w]), int32_t(blk[2 + w]), ui);
                            }
                            gacc[mm] += gd * d8 * float(gi);
                            uacc[mm] += ud * d8 * float(ui);
                        }
                    }
                }
                #pragma unroll
                for (int mm = 0; mm < PF_M_TILE; ++mm) {
                    if (uint32_t(mm) < M) {
                        const float gr = sycl::reduce_over_group(it.get_sub_group(), gacc[mm], sycl::plus<float>());
                        const float ur = sycl::reduce_over_group(it.get_sub_group(), uacc[mm], sycl::plus<float>());
                        if (lane == 0) {
                            const float s_ = gr / (1.f + sycl::exp(-gr));   // silu(gate)
                            h_packed[uint64_t(off_start + tk + mm) * E_ffn + n] = sycl::half(s_ * ur);
                        }
                    }
                }
                sycl::group_barrier(it.get_group());
            }
        });
    });
}

sycl::event moe_prefill_down_q8(sycl::queue& q, const void* hq8_packed,
                                const int8_t* d_qs, const uint16_t* d_d,
                                uint64_t qs_stride, uint64_t d_stride,
                                const uint32_t* expert_offsets,
                                const sycl::half* sorted_w,
                                sycl::half* out_packed,
                                uint32_t E, uint32_t H, uint32_t E_ffn,
                                const std::vector<sycl::event>& deps) {
    const uint32_t bpc        = E_ffn / 32;                  // weight blocks / col
    const uint32_t q8_per_row = E_ffn / 32;                  // activation blocks / row
    const uint32_t n_blk_per_lane = (q8_per_row + SG_SIZE - 1) / SG_SIZE;
    const uint32_t n_chunks   = (H + N_PER_WG - 1) / N_PER_WG;
    const auto* X8 = static_cast<const block_q8_1x*>(hq8_packed);
    return ie::ps(q, "moe_pfl_down_q8", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<uint32_t, 1> sh(uint64_t(PF_M_TILE) * q8_per_row * Q8X_BW, h);
        h.parallel_for(sycl::nd_range<2>({uint64_t(E) * n_chunks, WG_ITEMS}, {1, WG_ITEMS}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t en = uint32_t(it.get_group(0));
            const uint32_t e  = en / n_chunks;
            const uint32_t nc = en % n_chunks;
            const uint32_t lid  = uint32_t(it.get_local_id(1));
            const uint32_t sg = lid / SG_SIZE, lane = lid % SG_SIZE;
            const uint32_t n  = nc * N_PER_WG + sg;
            const uint32_t off_start = expert_offsets[e];
            const uint32_t n_tok     = expert_offsets[e + 1] - off_start;
            if (n_tok == 0) return;

            // Pre-load this lane's down-weight blocks for column n (E_ffn/32 ≤ 16
            // ⇒ n_blk_per_lane = 1 on the crown: each lane owns one block, read
            // once per expert and reused across every routed token row).
            uint32_t wq[PF_MAX_BPL][8];
            float    wd[PF_MAX_BPL];
            uint32_t my_j[PF_MAX_BPL];
            uint32_t my_nblk = 0;
            if (n < H) {
                const int8_t*   wcol = d_qs + e * qs_stride + uint64_t(n) * E_ffn;
                const uint16_t* wsc  = d_d  + e * d_stride  + uint64_t(n) * bpc;
                for (uint32_t s = 0; s < n_blk_per_lane; ++s) {
                    const uint32_t j = lane + s * SG_SIZE;
                    if (j >= q8_per_row) break;
                    const uint32_t* wp = reinterpret_cast<const uint32_t*>(wcol + uint64_t(j) * 32);
                    #pragma unroll
                    for (int w = 0; w < 8; ++w) wq[my_nblk][w] = wp[w];
                    wd[my_nblk]   = dev_fp16_to_fp32(wsc[j]);
                    my_j[my_nblk] = j;
                    ++my_nblk;
                }
            }

            for (uint32_t tk = 0; tk < n_tok; tk += PF_M_TILE) {
                const uint32_t M = sycl::min(uint32_t(PF_M_TILE), n_tok - tk);
                const uint32_t* src = reinterpret_cast<const uint32_t*>(
                    X8 + uint64_t(off_start + tk) * q8_per_row);
                for (uint32_t i = lid; i < M * q8_per_row * Q8X_BW; i += WG_ITEMS) sh[i] = src[i];
                sycl::group_barrier(it.get_group());
                if (n >= H) { sycl::group_barrier(it.get_group()); continue; }

                const uint32_t* base = sh.get_multi_ptr<sycl::access::decorated::no>().get();
                float acc[PF_M_TILE];
                #pragma unroll
                for (int mm = 0; mm < PF_M_TILE; ++mm) {
                    acc[mm] = 0.f;
                    if (uint32_t(mm) < M) {
                        for (uint32_t s = 0; s < my_nblk; ++s) {
                            const uint32_t* blk = base + (uint32_t(mm) * q8_per_row + my_j[s]) * Q8X_BW;
                            const float d8 = sycl::bit_cast<float>(blk[0]);
                            int32_t idot = 0;
                            #pragma unroll
                            for (int w = 0; w < 8; ++w)
                                idot = ie::dp4a_ss(int32_t(wq[s][w]), int32_t(blk[2 + w]), idot);
                            acc[mm] += wd[s] * d8 * float(idot);
                        }
                    }
                }
                #pragma unroll
                for (int mm = 0; mm < PF_M_TILE; ++mm) {
                    if (uint32_t(mm) < M) {
                        const float r = sycl::reduce_over_group(it.get_sub_group(), acc[mm], sycl::plus<float>());
                        if (lane == 0) {
                            const uint64_t row = uint64_t(off_start + tk + mm);
                            out_packed[row * H + n] = sycl::half(float(sorted_w[row]) * r);
                        }
                    }
                }
                sycl::group_barrier(it.get_group());
            }
        });
    });
}

sycl::event moe_prefill_reduce_sum(sycl::queue& q, const sycl::half* out_packed,
                                   const uint32_t* tk_to_packed, sycl::half* y,
                                   uint32_t T, uint32_t K, uint32_t H,
                                   const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(uint64_t(T) * H), [=](sycl::id<1> i) {
            const uint32_t t = uint32_t(uint64_t(i) / H), n = uint32_t(uint64_t(i) % H);
            float s = 0.f;
            for (uint32_t k = 0; k < K; ++k) {
                const uint32_t pos = tk_to_packed[uint64_t(t) * K + k];
                s += float(out_packed[uint64_t(pos) * H + n]);
            }
            y[i] = sycl::half(s);
        });
    });
}

}  // namespace ie
