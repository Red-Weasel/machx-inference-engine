// src/ops/prefill_gemm.cpp — P4 B48 (include/ie/prefill_gemm.hpp): full_attention_prefill_gemm, a prefill chunk's
// causal full attention as whole-matrix products, and q8_scales_kb_major.
//
// Why: the 27B's reading profile (2026-10-04, 8,133-token prompt, serial): its 16 attention layers took half the
// wall time and the attention kernels ran at ~0.7e12 multiply-adds/s, against ~50-70e12 for a matmul on the same
// card. The streaming kernels (attention.cpp) keep one subgroup per query row or tile and never reach the matrix
// engines' rate at head_dim 256; here both products are plain matmuls and only the softmax between them is ours.

#include "ie/prefill_gemm.hpp"
#include "ie/kernel_profiler.hpp"
#include "ie/ops.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <limits>

// State columns per subgroup in deltanet_scan_prefill (a power of two up to 16).
#ifndef IE_DN_SCAN_COLS
#define IE_DN_SCAN_COLS 4
#endif

namespace ie {

namespace {
// The body of both entry points. k_in / v_in null = the caller appended the chunk's rows already; sel_rows null = the
// causal mask alone, else token t attends only the n_sel_rows[t] keys listed in sel_rows + t * sel_stride.
sycl::event attn_gemm_impl(sycl::queue& q,
                           const sycl::half* q_in,
                           const sycl::half* k_in,
                           const sycl::half* v_in,
                           sycl::half* k_cache,
                           sycl::half* v_cache,
                           sycl::half* y,
                           uint32_t T,
                           uint32_t start_pos,
                           uint32_t n_q_heads,
                           uint32_t n_kv_heads,
                           uint32_t head_dim,
                           uint32_t max_ctx,
                           const int32_t* sel_rows, uint32_t sel_stride, const int32_t* n_sel_rows,
                           const AttnGemmScratch& scr,
                           const std::vector<sycl::event>& deps) {
    if (T == 0) return {};
    const uint32_t hd   = head_dim;
    const uint32_t gqa  = n_q_heads / n_kv_heads;
    const uint32_t ctx  = start_pos + T;
    // the rounded shapes (see the header): keys, tokens, rows of one kv head, rows of a block
    const uint32_t keys = std::min<uint32_t>((ctx + kAttnGemmKeyStep - 1) / kAttnGemmKeyStep * kAttnGemmKeyStep, max_ctx);
    const uint32_t Tp   = (T + kAttnGemmRowStep - 1) / kAttnGemmRowStep * kAttnGemmRowStep;
    const uint32_t rows_all = gqa * T;                                   // real query rows of one kv head
    const uint32_t rows_pad = gqa * Tp;
    const uint32_t rows_blk = attn_gemm_rows(rows_pad, keys, scr.sp_elems);
    sycl::half* const qh = scr.qh;
    float*      const s  = scr.s;
    sycl::half* const p  = scr.p;
    float*      const o  = scr.o;
    float*      const l  = scr.l;

    // 1. Append (k_in, v_in) at [start_pos, start_pos + T) — full_attention's append — and zero the value rows
    //    [ctx, keys) of every kv head (their weights are 0; 0 * a stale value must not be a NaN).
    if (k_in && v_in)
    ie::ps(q, "attn_gemm_append", [&](sycl::handler& h) {
        h.depends_on(deps);
        const uint32_t total = T * n_kv_heads * hd;
        constexpr uint32_t WG = 256;
        const uint32_t global = ((total + WG - 1) / WG) * WG;
        h.parallel_for(sycl::nd_range<1>(global, WG), [=](sycl::nd_item<1> it) {
            const uint32_t idx = uint32_t(it.get_global_id(0));
            if (idx >= total) return;
            const uint32_t d  = idx % hd;
            const uint32_t kv = (idx / hd) % n_kv_heads;
            const uint32_t t  = idx / (n_kv_heads * hd);
            const uint64_t in_off  = (uint64_t(t) * n_kv_heads + kv) * hd + d;
            const uint64_t out_off = (uint64_t(kv) * max_ctx + (start_pos + t)) * hd + d;
            k_cache[out_off] = k_in[in_off];
            v_cache[out_off] = v_in[in_off];
        });
    });
    if (keys > ctx) {
        ie::ps(q, "attn_gemm_vzero", [&](sycl::handler& h) {
            h.parallel_for(sycl::range<2>(uint64_t(n_kv_heads) * (keys - ctx), hd), [=](sycl::id<2> id) {
                const uint32_t i  = uint32_t(id[0]);
                const uint32_t kv = i / (keys - ctx);
                const uint32_t c  = ctx + i % (keys - ctx);
                v_cache[(uint64_t(kv) * max_ctx + c) * hd + uint32_t(id[1])] = sycl::half(0.f);
            });
        });
    }

    // 1b. The selected keys as one bit a (token, key): a token's row is cleared and set by one work-item.
    const bool use_mask = sel_rows != nullptr;
    const uint32_t W = (keys + 31u) / 32u;
    uint32_t* const mask = scr.mask;
    if (use_mask) {
        ie::ps(q, "attn_gemm_mask", [&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(T), [=](sycl::id<1> ti) {
                const uint32_t t = uint32_t(ti[0]);
                uint32_t* row = mask + uint64_t(t) * W;
                for (uint32_t w = 0; w < W; ++w) row[w] = 0u;
                const int32_t* sel = sel_rows + uint64_t(t) * sel_stride;
                const int32_t n = n_sel_rows[t];
                for (int32_t j = 0; j < n; ++j) {
                    const uint32_t c = uint32_t(sel[j]);
                    row[c >> 5] |= 1u << (c & 31u);
                }
            });
        });
    }

    // 2. q_in [T, n_q, hd] -> qh [n_q, T, hd]: the heads of one kv head become one contiguous [gqa * T, hd] operand.
    ie::ps(q, "attn_gemm_qperm", [&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>(uint64_t(T) * n_q_heads, hd), [=](sycl::id<2> id) {
            const uint32_t th = uint32_t(id[0]);
            const uint32_t d  = uint32_t(id[1]);
            const uint32_t t  = th / n_q_heads;
            const uint32_t hq = th % n_q_heads;
            qh[(uint64_t(hq) * T + t) * hd + d] = q_in[uint64_t(th) * hd + d];
        });
    });

    const float scale = 1.0f / sycl::sqrt(float(hd));
    sycl::event last;
    for (uint32_t g = 0; g < n_kv_heads; ++g) {
        const sycl::half* kc = k_cache + uint64_t(g) * max_ctx * hd;     // [keys, hd]
        const sycl::half* vc = v_cache + uint64_t(g) * max_ctx * hd;     // [keys, hd]
        for (uint32_t r0 = 0; r0 < rows_all; r0 += rows_blk) {
            const uint32_t R = std::min(rows_blk, rows_all - r0);        // real rows of this block
            const uint32_t M = std::min(rows_blk, rows_pad - r0);        // the matmuls' (rounded) rows
            // S [M, keys] = Q [M, hd] @ K^T
            gemm_attn_qk_onednn(q, qh + (uint64_t(g) * rows_all + r0) * hd, kc, s, M, keys, hd);

            // P = exp((S - rowmax) * scale) * 2^10 as fp16 over the causal keys, 0 elsewhere; L = the sum of the
            // rounded weights. One work-group a row; a lane takes every WGS-th key.
            ie::ps(q, "attn_gemm_softmax", [&](sycl::handler& h) {
                constexpr uint32_t WGS = 128;
                h.parallel_for(sycl::nd_range<2>({R, WGS}, {1, WGS}), [=](sycl::nd_item<2> it) {
                    const uint32_t r    = uint32_t(it.get_group(0));
                    const uint32_t lane = uint32_t(it.get_local_id(1));
                    const uint32_t t    = (r0 + r) % T;
                    const uint32_t n_ok = start_pos + t + 1;             // keys [0, n_ok)
                    const float* srow = s + uint64_t(r) * keys;
                    sycl::half*  prow = p + uint64_t(r) * keys;
                    const uint32_t* mrow = mask + uint64_t(use_mask ? t : 0u) * W;   // read only when use_mask
                    float m = -std::numeric_limits<float>::infinity();
                    for (uint32_t c = lane; c < n_ok; c += WGS)
                        if (!use_mask || ((mrow[c >> 5] >> (c & 31u)) & 1u)) m = sycl::fmax(m, srow[c]);
                    m = sycl::reduce_over_group(it.get_group(), m, sycl::maximum<float>());
                    float sum = 0.f;
                    for (uint32_t c = lane; c < keys; c += WGS) {
                        sycl::half w(0.f);
                        if (c < n_ok && (!use_mask || ((mrow[c >> 5] >> (c & 31u)) & 1u)))
                            w = sycl::half(sycl::exp((srow[c] - m) * scale) * 1024.f);
                        prow[c] = w;
                        sum += float(w);
                    }
                    sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
                    if (lane == 0) l[r] = sum;
                });
            });

            // O [M, hd] = P [M, keys] @ V [keys, hd]
            gemm_attn_pv_onednn(q, p, vc, o, M, hd, keys);

            // y [T, n_q, hd] = O / L
            last = ie::ps(q, "attn_gemm_norm", [&](sycl::handler& h) {
                h.parallel_for(sycl::range<2>(R, hd), [=](sycl::id<2> id) {
                    const uint32_t r  = uint32_t(id[0]);
                    const uint32_t d  = uint32_t(id[1]);
                    const uint32_t rg = r0 + r;
                    const uint32_t hq = g * gqa + rg / T;
                    const uint32_t t  = rg % T;
                    const float lv = l[r];
                    y[(uint64_t(t) * n_q_heads + hq) * hd + d] =
                        sycl::half(lv > 0.f ? o[uint64_t(r) * hd + d] / lv : 0.f);
                });
            });
        }
    }
    return last;
}
}  // namespace

sycl::event full_attention_prefill_gemm(sycl::queue& q,
                                        const sycl::half* q_in,
                                        const sycl::half* k_in,
                                        const sycl::half* v_in,
                                        sycl::half* k_cache,
                                        sycl::half* v_cache,
                                        sycl::half* y,
                                        uint32_t T,
                                        uint32_t start_pos,
                                        uint32_t n_q_heads,
                                        uint32_t n_kv_heads,
                                        uint32_t head_dim,
                                        uint32_t max_ctx,
                                        const AttnGemmScratch& scr,
                                        const std::vector<sycl::event>& deps) {
    return attn_gemm_impl(q, q_in, k_in, v_in, k_cache, v_cache, y, T, start_pos, n_q_heads, n_kv_heads, head_dim, max_ctx,
                          nullptr, 0, nullptr, scr, deps);
}

sycl::event full_attention_prefill_gemm_sel(sycl::queue& q,
                                            const sycl::half* q_in,
                                            sycl::half* k_cache,
                                            sycl::half* v_cache,
                                            sycl::half* y,
                                            uint32_t T,
                                            uint32_t start_pos,
                                            uint32_t n_q_heads,
                                            uint32_t n_kv_heads,
                                            uint32_t head_dim,
                                            uint32_t max_ctx,
                                            const int32_t* sel_rows, uint32_t sel_stride,
                                            const int32_t* n_sel_rows,
                                            const AttnGemmScratch& scr) {
    return attn_gemm_impl(q, q_in, nullptr, nullptr, k_cache, v_cache, y, T, start_pos, n_q_heads, n_kv_heads, head_dim,
                          max_ctx, sel_rows, sel_stride, n_sel_rows, scr, {});
}

sycl::event q8_scales_kb_major(sycl::queue& q, const uint16_t* d, sycl::half* dt, uint32_t N, uint32_t nb) {
    // A work-item moves 16 consecutive kb of one n, and the SIMD lanes run along n: the reads are 32 contiguous bytes
    // a lane, the writes 32 contiguous bytes a kb, so a 16 x 16 tile touches 32 cache lines (an element-wise transpose
    // touches one line per element on one of its two sides).
    constexpr uint32_t KT = 16;
    return ie::ps(q, "q8_scales_kb_major", [&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>((nb + KT - 1) / KT, N), [=](sycl::id<2> id) {
            const uint32_t kb0 = uint32_t(id[0]) * KT;
            const uint32_t n   = uint32_t(id[1]);
            const uint16_t* src = d + uint64_t(n) * nb + kb0;
            #pragma unroll
            for (uint32_t j = 0; j < KT; ++j)
                if (kb0 + j < nb) dt[uint64_t(kb0 + j) * N + n] = sycl::bit_cast<sycl::half>(src[j]);
        });
    });
}

sycl::event deltanet_scan_prefill(sycl::queue& q,
                                  const float* q_in, const float* k_in, const float* v_in,
                                  const float* g_in, const float* beta_in,
                                  float* state, float* out,
                                  uint32_t T, uint32_t n_v_heads) {
    // One subgroup a (head, block of C state columns): lane l holds rows {l, 16 + l, .., 112 + l} of each of its C
    // columns in registers (8 floats a column), so a step is 16 contiguous-block loads (k_t, q_t: shared by the C
    // columns), ~24 FMAs a column and two subgroup sums a column -- no SLM, no group barrier, no private array. Per
    // column the math is deltanet_recurrence's; the two dots are summed shard-first (fp32 rounding order only).
    constexpr int D  = 128;
    constexpr int SG = 16;
    constexpr int R  = D / SG;
    constexpr int C  = IE_DN_SCAN_COLS;
    return ie::ps(q, "dn_scan_prefill", [&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<2>({uint64_t(n_v_heads) * (D / C), SG}, {1, SG}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint32_t grp  = uint32_t(it.get_group(0));
            const uint32_t hh   = grp / (D / C);
            const uint32_t vv0  = (grp % (D / C)) * C;
            const uint32_t lane = uint32_t(it.get_local_id(1));
            auto sg = it.get_sub_group();
            const uint64_t state_base = uint64_t(hh) * D * D;

            float s[C][R];
            #pragma unroll
            for (int c = 0; c < C; ++c)
                #pragma unroll
                for (int r = 0; r < R; ++r) s[c][r] = state[state_base + uint64_t(r * SG + lane) * D + vv0 + c];

            for (uint32_t t = 0; t < T; ++t) {
                const uint64_t row = uint64_t(t) * n_v_heads + hh;
                const float* k_t = k_in + row * D + lane;
                const float* q_t = q_in + row * D + lane;
                const float alpha  = sycl::native::exp(g_in[row]);
                const float beta_t = beta_in[row];
                const float v_l    = v_in[row * D + vv0 + (lane % C)];   // lane c < C carries column c's v_t

                float kr[R];
                #pragma unroll
                for (int r = 0; r < R; ++r) kr[r] = k_t[r * SG];
                float delta[C];
                #pragma unroll
                for (int c = 0; c < C; ++c) {
                    float kv_part = 0.f;
                    #pragma unroll
                    for (int r = 0; r < R; ++r) {
                        s[c][r] *= alpha;
                        kv_part += s[c][r] * kr[r];
                    }
                    const float kv_mem = sycl::reduce_over_group(sg, kv_part, sycl::plus<float>());
                    const float v_t    = sycl::group_broadcast(sg, v_l, c);
                    delta[c] = (v_t - kv_mem) * beta_t;
                }
                float out_part[C];
                #pragma unroll
                for (int c = 0; c < C; ++c) out_part[c] = 0.f;
                #pragma unroll
                for (int r = 0; r < R; ++r) {
                    const float qr = q_t[r * SG];
                    #pragma unroll
                    for (int c = 0; c < C; ++c) {
                        s[c][r] += kr[r] * delta[c];
                        out_part[c] += s[c][r] * qr;
                    }
                }
                float out_l = 0.f;   // lane c < C ends up holding column c's output
                #pragma unroll
                for (int c = 0; c < C; ++c) {
                    const float o = sycl::reduce_over_group(sg, out_part[c], sycl::plus<float>());
                    if (lane == uint32_t(c)) out_l = o;
                }
                if (lane < uint32_t(C)) out[row * D + vv0 + lane] = out_l;
            }

            #pragma unroll
            for (int c = 0; c < C; ++c)
                #pragma unroll
                for (int r = 0; r < R; ++r) state[state_base + uint64_t(r * SG + lane) * D + vv0 + c] = s[c][r];
        });
    });
}

}  // namespace ie
