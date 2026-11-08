// src/ops/qwen4_bank.cpp -- A16 GEMV / grouped / tiles / batched kernels over qwen4exp expert banks in Q8_0, Q5_K, Q6_K,
// Q5_0 and Q4_0 (include/ie/qwen4_bank.hpp has the contracts). One column walk per kind feeds every variant through an
// `emit(k_lo, w_lo, k_hi, w_hi)` callback: the solo and grouped kernels read one activation row, the tiles and batched
// kernels read T rows, and per row the fp32 expression `acc += a_lo*w_lo + a_hi*w_hi` runs in the same order -> the
// variants are bit-identical per output (ie-qwen4-bank-test). The dequantisation is dequant_ref.hpp's, transcribed:
//   Q8_0: d*q                       Q4_0: (q-8)*d                     Q5_0: ((q | qh-bit) - 16)*d
//   Q5_K: d*sc*(q | qh-bit) - dmin*m (6-bit scale/min table)         Q6_K: d*sc16*(q6 - 32) (int8 scales per 16)
// Lane roles (16-wide subgroups): lane L owns elements L and L+16 of every 32-wide block (the Q5_1 kernel's split); in a
// 256-wide K-quant super-block it owns l = L and L+16 of each 32-wide half-chunk, chunks in ascending order.
#include "ie/qwen4_bank.hpp"
#include "ie/quant_blocks.hpp"
#include "ie/kernel_profiler.hpp"
#include "ie/dp4a.hpp"

#include <sycl/sycl.hpp>

namespace ie {

namespace {

constexpr int SG_SIZE  = 16;
constexpr int N_PER_WG = 16;
constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;   // 256
constexpr int T_MAX    = 16;

inline float h2f(uint16_t h) { return float(sycl::bit_cast<sycl::half>(h)); }
inline uint32_t u32le(const uint8_t* p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}
// ggml's get_scale_min_k4 (dequant_ref.hpp): the 6-bit scale and min of sub-block j from the 12-byte table.
inline void scale_min_k4(int j, const uint8_t* q, float& sc, float& m) {
    uint32_t s, mm;
    if (j < 4) { s = q[j] & 0x3Fu;                                   mm = q[j + 4] & 0x3Fu; }
    else       { s = (q[j + 4] & 0x0Fu) | ((q[j - 4] >> 6) << 4);   mm = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
    sc = float(s); m = float(mm);
}

template <BankQ Q> struct Blk;
template <> struct Blk<BankQ::kQ8_0> { using T = block_q8_0; static constexpr uint32_t kElems = 32; };
template <> struct Blk<BankQ::kQ5_0> { using T = block_q5_0; static constexpr uint32_t kElems = 32; };
template <> struct Blk<BankQ::kQ4_0> { using T = block_q4_0; static constexpr uint32_t kElems = 32; };
template <> struct Blk<BankQ::kQ5_K> { using T = block_q5_K; static constexpr uint32_t kElems = 256; };
template <> struct Blk<BankQ::kQ6_K> { using T = block_q6_K; static constexpr uint32_t kElems = 256; };

template <BankQ Q> inline uint64_t col_bytes_of(uint32_t K) {
    return uint64_t(K / Blk<Q>::kElems) * sizeof(typename Blk<Q>::T);
}

// The column walk (see the file header). `col` = the output column's first block.
template <BankQ Q, class Emit>
inline void walk_col(const uint8_t* col, uint32_t K, uint32_t lane, Emit&& emit) {
    const auto* W = reinterpret_cast<const typename Blk<Q>::T*>(col);
    const uint32_t nb = K / Blk<Q>::kElems;
    for (uint32_t b = 0; b < nb; ++b) {
        const auto& blk = W[b];
        if constexpr (Q == BankQ::kQ8_0) {
            const float d = h2f(blk.d);
            emit(b * 32 + lane, d * float(blk.qs[lane]), b * 32 + lane + 16, d * float(blk.qs[lane + 16]));
        } else if constexpr (Q == BankQ::kQ4_0) {
            const float d = h2f(blk.d);
            const uint8_t qb = blk.qs[lane];
            emit(b * 32 + lane,      float(int(qb & 0x0Fu) - 8) * d,
                 b * 32 + lane + 16, float(int(qb >> 4)    - 8) * d);
        } else if constexpr (Q == BankQ::kQ5_0) {
            const float d = h2f(blk.d);
            const uint32_t qh = u32le(blk.qh);
            const uint8_t qb = blk.qs[lane];
            const uint32_t xh0 = ((qh >> lane) << 4) & 0x10u, xh1 = (qh >> (lane + 12)) & 0x10u;
            emit(b * 32 + lane,      float(int((qb & 0x0Fu) | xh0) - 16) * d,
                 b * 32 + lane + 16, float(int((qb >> 4)    | xh1) - 16) * d);
        } else if constexpr (Q == BankQ::kQ5_K) {
            const float d = h2f(blk.d), dmin = h2f(blk.dmin);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float s1, m1, s2, m2;
                scale_min_k4(2 * j, blk.scales, s1, m1);
                scale_min_k4(2 * j + 1, blk.scales, s2, m2);
                const float d1 = d * s1, mn1 = dmin * m1, d2 = d * s2, mn2 = dmin * m2;
                const uint8_t u1 = uint8_t(1u << (2 * j)), u2 = uint8_t(2u << (2 * j));
                const uint8_t* ql = blk.qs + 32 * j;
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const uint32_t l = lane + 16 * h;
                    const float wlo = d1 * float((ql[l] & 0x0Fu) + ((blk.qh[l] & u1) ? 16u : 0u)) - mn1;
                    const float whi = d2 * float((ql[l] >> 4)    + ((blk.qh[l] & u2) ? 16u : 0u)) - mn2;
                    emit(b * 256 + j * 64 + l, wlo, b * 256 + j * 64 + 32 + l, whi);
                }
            }
        } else {   // Q6_K
            const float d = h2f(blk.d);
            #pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                const uint8_t* ql = blk.ql + 64 * hh;
                const uint8_t* qh = blk.qh + 32 * hh;
                const int8_t*  sc = blk.scales + 8 * hh;
                const uint32_t base = b * 256 + 128 * hh;
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const uint32_t l = lane + 16 * h;
                    const int is = int(l / 16);
                    const int q1 = int((ql[l]      & 0x0Fu) | (((qh[l] >> 0) & 3u) << 4)) - 32;
                    const int q2 = int((ql[l + 32] & 0x0Fu) | (((qh[l] >> 2) & 3u) << 4)) - 32;
                    const int q3 = int((ql[l]      >> 4)    | (((qh[l] >> 4) & 3u) << 4)) - 32;
                    const int q4 = int((ql[l + 32] >> 4)    | (((qh[l] >> 6) & 3u) << 4)) - 32;
                    emit(base + l,      d * float(sc[is])     * float(q1), base + l + 32, d * float(sc[is + 2]) * float(q2));
                    emit(base + l + 64, d * float(sc[is + 4]) * float(q3), base + l + 96, d * float(sc[is + 6]) * float(q4));
                }
            }
        }
    }
}

// One output column for T (<= 16) activation rows read from global memory: the tiles and batched kernels' body.
template <BankQ Q>
inline void rows_body(sycl::nd_item<1> it, uint32_t lane, uint32_t n, const uint8_t* col, uint32_t K,
                      const sycl::half* A0, uint32_t T, sycl::half* out, uint32_t N) {
    float acc[T_MAX];
    #pragma unroll
    for (int t = 0; t < T_MAX; ++t) acc[t] = 0.f;
    walk_col<Q>(col, K, lane, [&](uint32_t klo, float wlo, uint32_t khi, float whi) {
        #pragma unroll
        for (int t = 0; t < T_MAX; ++t)
            if (uint32_t(t) < T) {
                const sycl::half* a = A0 + uint64_t(t) * K;
                acc[t] += float(a[klo]) * wlo + float(a[khi]) * whi;
            }
    });
    for (uint32_t t = 0; t < T; ++t) {
        const float r = sycl::reduce_over_group(it.get_sub_group(), acc[t], sycl::plus<float>());
        if (lane == 0) out[uint64_t(t) * N + n] = sycl::half(r);
    }
}

template <BankQ Q>
sycl::event gemv_k(sycl::queue& q, const sycl::half* A, const void* Wv, sycl::half* y, uint32_t K, uint32_t N,
                   const std::vector<sycl::event>& deps) {
    const uint64_t cb = col_bytes_of<Q>(K);
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;
    const auto* W = static_cast<const uint8_t*>(Wv);
    return ie::ps(q, "q4e_bank_gemv", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<sycl::half, 1> A_slm(sycl::range<1>(K), h);
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid = uint32_t(it.get_local_id(0)), wgid = uint32_t(it.get_group(0));
            const uint32_t sg_id = lid / SG_SIZE, lane = lid % SG_SIZE;
            const uint32_t n = wgid * N_PER_WG + sg_id;
            for (uint32_t i = lid; i < K; i += WG_ITEMS) A_slm[i] = A[i];
            sycl::group_barrier(it.get_group());
            if (n >= N) return;
            float acc = 0.f;
            walk_col<Q>(W + uint64_t(n) * cb, K, lane, [&](uint32_t klo, float wlo, uint32_t khi, float whi) {
                acc += float(A_slm[klo]) * wlo + float(A_slm[khi]) * whi;
            });
            acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
            if (lane == 0) y[n] = sycl::half(acc);
        });
    });
}

template <BankQ Q>
sycl::event grouped_dual_k(sycl::queue& q, const sycl::half* x, const uint8_t* slot_base, uint64_t slot_bytes,
                           uint64_t gate_off, uint64_t up_off, const int32_t* jobs, sycl::half* g_out, sycl::half* u_out,
                           uint32_t K, uint32_t N, uint32_t P, const std::vector<sycl::event>& deps) {
    const uint64_t cb = col_bytes_of<Q>(K);
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;
    return ie::ps(q, "q4e_bank_grp_dual", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<sycl::half, 1> A_slm(sycl::range<1>(K), h);
        h.parallel_for(sycl::nd_range<1>(uint64_t(P) * 2 * n_wgs * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid = uint32_t(it.get_local_id(0)), gid = uint32_t(it.get_group(0));
            const uint32_t pk = gid / (2 * n_wgs), rem = gid % (2 * n_wgs);
            const bool is_up = rem >= n_wgs;
            const uint32_t wgid = is_up ? rem - n_wgs : rem;
            const uint32_t sg_id = lid / SG_SIZE, lane = lid % SG_SIZE;
            const uint32_t n = wgid * N_PER_WG + sg_id;
            const uint32_t vt = uint32_t(jobs[pk * 3 + 0]), slot = uint32_t(jobs[pk * 3 + 1]);
            const sycl::half* A = x + uint64_t(vt) * K;
            const uint8_t* W = slot_base + uint64_t(slot) * slot_bytes + (is_up ? up_off : gate_off);
            for (uint32_t i = lid; i < K; i += WG_ITEMS) A_slm[i] = A[i];
            sycl::group_barrier(it.get_group());
            if (n >= N) return;
            float acc = 0.f;
            walk_col<Q>(W + uint64_t(n) * cb, K, lane, [&](uint32_t klo, float wlo, uint32_t khi, float whi) {
                acc += float(A_slm[klo]) * wlo + float(A_slm[khi]) * whi;
            });
            acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
            if (lane == 0) (is_up ? u_out : g_out)[uint64_t(pk) * N + n] = sycl::half(acc);
        });
    });
}

// grouped_dual_k with swiglu fused into the epilogue (2026-11-08, the decode host gap, option 1): one work-group per
// (pick, 16-column strip) stages the activation row once and walks the gate and then the up column of each strip (each
// the dual kernel's exact chain); lane 0 writes h = silu(f16(g)) * f16(u) exactly as swiglu() computes it from the dual
// kernel's f16 outputs. Bit-identical to dual + swiglu (ie-qwen4-bank-test); one launch fewer a MoE layer.
template <BankQ Q>
sycl::event grouped_dual_swiglu_k(sycl::queue& q, const sycl::half* x, const uint8_t* slot_base, uint64_t slot_bytes,
                                  uint64_t gate_off, uint64_t up_off, const int32_t* jobs, sycl::half* h_out,
                                  uint32_t K, uint32_t N, uint32_t P, const std::vector<sycl::event>& deps) {
    const uint64_t cb = col_bytes_of<Q>(K);
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;
    return ie::ps(q, "q4e_bank_grp_dual_sw", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<sycl::half, 1> A_slm(sycl::range<1>(K), h);
        h.parallel_for(sycl::nd_range<1>(uint64_t(P) * n_wgs * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid = uint32_t(it.get_local_id(0)), gid = uint32_t(it.get_group(0));
            const uint32_t pk = gid / n_wgs, wgid = gid % n_wgs;
            const uint32_t sg_id = lid / SG_SIZE, lane = lid % SG_SIZE;
            const uint32_t n = wgid * N_PER_WG + sg_id;
            const uint32_t vt = uint32_t(jobs[pk * 3 + 0]), slot = uint32_t(jobs[pk * 3 + 1]);
            const sycl::half* A = x + uint64_t(vt) * K;
            const uint8_t* sb = slot_base + uint64_t(slot) * slot_bytes;
            for (uint32_t i = lid; i < K; i += WG_ITEMS) A_slm[i] = A[i];
            sycl::group_barrier(it.get_group());
            if (n >= N) return;
            float acc_g = 0.f, acc_u = 0.f;
            walk_col<Q>(sb + gate_off + uint64_t(n) * cb, K, lane, [&](uint32_t klo, float wlo, uint32_t khi, float whi) {
                acc_g += float(A_slm[klo]) * wlo + float(A_slm[khi]) * whi;
            });
            walk_col<Q>(sb + up_off + uint64_t(n) * cb, K, lane, [&](uint32_t klo, float wlo, uint32_t khi, float whi) {
                acc_u += float(A_slm[klo]) * wlo + float(A_slm[khi]) * whi;
            });
            acc_g = sycl::reduce_over_group(it.get_sub_group(), acc_g, sycl::plus<float>());
            acc_u = sycl::reduce_over_group(it.get_sub_group(), acc_u, sycl::plus<float>());
            if (lane == 0) {
                const float gf = float(sycl::half(acc_g)), uf = float(sycl::half(acc_u));
                const float s = gf / (1.f + sycl::native::exp(-gf));
                h_out[uint64_t(pk) * N + n] = sycl::half(s * uf);
            }
        });
    });
}

template <BankQ Q>
sycl::event grouped_k(sycl::queue& q, const sycl::half* h_rows, const uint8_t* slot_base, uint64_t slot_bytes, uint64_t off,
                      const int32_t* jobs, sycl::half* ystage, uint32_t y_stride, uint32_t K, uint32_t N, uint32_t P,
                      const std::vector<sycl::event>& deps) {
    const uint64_t cb = col_bytes_of<Q>(K);
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;
    return ie::ps(q, "q4e_bank_grp", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<sycl::half, 1> A_slm(sycl::range<1>(K), h);
        h.parallel_for(sycl::nd_range<1>(uint64_t(P) * n_wgs * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid = uint32_t(it.get_local_id(0)), gid = uint32_t(it.get_group(0));
            const uint32_t p = gid / n_wgs, wgid = gid % n_wgs;
            const uint32_t sg_id = lid / SG_SIZE, lane = lid % SG_SIZE;
            const uint32_t n = wgid * N_PER_WG + sg_id;
            const uint32_t slot = uint32_t(jobs[p * 3 + 1]), ycell = uint32_t(jobs[p * 3 + 2]);
            const sycl::half* A = h_rows + uint64_t(p) * K;
            const uint8_t* W = slot_base + uint64_t(slot) * slot_bytes + off;
            for (uint32_t i = lid; i < K; i += WG_ITEMS) A_slm[i] = A[i];
            sycl::group_barrier(it.get_group());
            if (n >= N) return;
            float acc = 0.f;
            walk_col<Q>(W + uint64_t(n) * cb, K, lane, [&](uint32_t klo, float wlo, uint32_t khi, float whi) {
                acc += float(A_slm[klo]) * wlo + float(A_slm[khi]) * whi;
            });
            acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
            if (lane == 0) ystage[uint64_t(ycell) * y_stride + n] = sycl::half(acc);
        });
    });
}

template <BankQ Q>
sycl::event tiles_k(sycl::queue& q, const sycl::half* A_mega, const uint8_t* slot_base, uint64_t slot_bytes,
                    uint64_t off0, uint64_t off1, const int32_t* tile_jobs, sycl::half* out0, sycl::half* out1,
                    uint32_t K, uint32_t N, uint32_t J, uint32_t mats, const std::vector<sycl::event>& deps) {
    const uint64_t cb = col_bytes_of<Q>(K);
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;
    return ie::ps(q, "q4e_bank_gtile", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(uint64_t(J) * mats * n_wgs * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid = uint32_t(it.get_local_id(0)), gid = uint32_t(it.get_group(0));
            const uint32_t j = gid / (mats * n_wgs), rem = gid % (mats * n_wgs);
            const uint32_t mat = rem / n_wgs, wgid = rem % n_wgs;
            const uint32_t sg_id = lid / SG_SIZE, lane = lid % SG_SIZE;
            const uint32_t n = wgid * N_PER_WG + sg_id;
            if (n >= N) return;
            const uint32_t x0 = uint32_t(tile_jobs[j * 4 + 0]);
            const uint32_t T  = sycl::min(uint32_t(tile_jobs[j * 4 + 1]), uint32_t(T_MAX));
            const uint32_t slot = uint32_t(tile_jobs[j * 4 + 2]);
            const uint32_t y0 = uint32_t(tile_jobs[j * 4 + 3]);
            const uint8_t* W = slot_base + uint64_t(slot) * slot_bytes + (mat ? off1 : off0);
            sycl::half* out = (mat ? out1 : out0) + uint64_t(y0) * N;
            rows_body<Q>(it, lane, n, W + uint64_t(n) * cb, K, A_mega + uint64_t(x0) * K, T, out, N);
        });
    });
}

template <BankQ Q>
sycl::event gemm_k(sycl::queue& q, const sycl::half* A, const void* Wv, sycl::half* y, uint32_t M, uint32_t K, uint32_t N,
                   const std::vector<sycl::event>& deps) {
    const uint64_t cb = col_bytes_of<Q>(K);
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;
    const uint32_t chunks = (M + T_MAX - 1) / T_MAX;
    const auto* W = static_cast<const uint8_t*>(Wv);
    return ie::ps(q, "q4e_bank_gemm", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(uint64_t(chunks) * n_wgs * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid = uint32_t(it.get_local_id(0)), gid = uint32_t(it.get_group(0));
            const uint32_t c = gid / n_wgs, wgid = gid % n_wgs;
            const uint32_t sg_id = lid / SG_SIZE, lane = lid % SG_SIZE;
            const uint32_t n = wgid * N_PER_WG + sg_id;
            if (n >= N) return;
            const uint32_t m0 = c * T_MAX, T = sycl::min(uint32_t(T_MAX), M - m0);
            rows_body<Q>(it, lane, n, W + uint64_t(n) * cb, K, A + uint64_t(m0) * K, T, y + uint64_t(m0) * N, N);
        });
    });
}

#define IE_BANK_DISPATCH(kind, CALL)                                   \
    switch (kind) {                                                    \
        case BankQ::kQ8_0: return CALL(BankQ::kQ8_0);                  \
        case BankQ::kQ5_K: return CALL(BankQ::kQ5_K);                  \
        case BankQ::kQ6_K: return CALL(BankQ::kQ6_K);                  \
        case BankQ::kQ5_0: return CALL(BankQ::kQ5_0);                  \
        case BankQ::kQ4_0: return CALL(BankQ::kQ4_0);                  \
    }                                                                  \
    return {};

}  // namespace

bool qwen4_bank_kind(DType dt, uint32_t K, BankQ& out) noexcept {
    switch (dt) {
        case DType::kQ8_0: out = BankQ::kQ8_0; return K % 32 == 0 && K > 0;
        case DType::kQ5_0: out = BankQ::kQ5_0; return K % 32 == 0 && K > 0;
        case DType::kQ4_0: out = BankQ::kQ4_0; return K % 32 == 0 && K > 0;
        case DType::kQ5_K: out = BankQ::kQ5_K; return K % 256 == 0 && K > 0;
        case DType::kQ6_K: out = BankQ::kQ6_K; return K % 256 == 0 && K > 0;
        default: return false;
    }
}

const char* qwen4_bank_name(BankQ q) noexcept {
    switch (q) {
        case BankQ::kQ8_0: return "Q8_0";
        case BankQ::kQ5_K: return "Q5_K";
        case BankQ::kQ6_K: return "Q6_K";
        case BankQ::kQ5_0: return "Q5_0";
        case BankQ::kQ4_0: return "Q4_0";
    }
    return "?";
}

uint64_t qwen4_bank_col_bytes(BankQ q, uint32_t K) noexcept {
    switch (q) {
        case BankQ::kQ8_0: return col_bytes_of<BankQ::kQ8_0>(K);
        case BankQ::kQ5_K: return col_bytes_of<BankQ::kQ5_K>(K);
        case BankQ::kQ6_K: return col_bytes_of<BankQ::kQ6_K>(K);
        case BankQ::kQ5_0: return col_bytes_of<BankQ::kQ5_0>(K);
        case BankQ::kQ4_0: return col_bytes_of<BankQ::kQ4_0>(K);
    }
    return 0;
}

sycl::event qwen4_bank_gemv(sycl::queue& q, BankQ kind, const sycl::half* A, const void* W, sycl::half* y,
                            uint32_t K, uint32_t N, const std::vector<sycl::event>& deps) {
#define IE_BANK_CALL(Q) gemv_k<Q>(q, A, W, y, K, N, deps)
    IE_BANK_DISPATCH(kind, IE_BANK_CALL)
#undef IE_BANK_CALL
}

sycl::event qwen4_bank_gemv_grouped_dual(sycl::queue& q, BankQ kind, const sycl::half* x, const uint8_t* slot_base,
                            uint64_t slot_bytes, uint64_t gate_off, uint64_t up_off, const int32_t* jobs,
                            sycl::half* g_out, sycl::half* u_out, uint32_t K, uint32_t N, uint32_t P,
                            const std::vector<sycl::event>& deps) {
#define IE_BANK_CALL(Q) grouped_dual_k<Q>(q, x, slot_base, slot_bytes, gate_off, up_off, jobs, g_out, u_out, K, N, P, deps)
    IE_BANK_DISPATCH(kind, IE_BANK_CALL)
#undef IE_BANK_CALL
}

sycl::event qwen4_bank_gemv_grouped_dual_swiglu(sycl::queue& q, BankQ kind, const sycl::half* x, const uint8_t* slot_base,
                            uint64_t slot_bytes, uint64_t gate_off, uint64_t up_off, const int32_t* jobs,
                            sycl::half* h_out, uint32_t K, uint32_t N, uint32_t P, const std::vector<sycl::event>& deps) {
#define IE_BANK_CALL(Q) grouped_dual_swiglu_k<Q>(q, x, slot_base, slot_bytes, gate_off, up_off, jobs, h_out, K, N, P, deps)
    IE_BANK_DISPATCH(kind, IE_BANK_CALL)
#undef IE_BANK_CALL
}

sycl::event qwen4_bank_gemv_grouped(sycl::queue& q, BankQ kind, const sycl::half* h_rows, const uint8_t* slot_base,
                            uint64_t slot_bytes, uint64_t off, const int32_t* jobs, sycl::half* ystage, uint32_t y_stride,
                            uint32_t K, uint32_t N, uint32_t P, const std::vector<sycl::event>& deps) {
#define IE_BANK_CALL(Q) grouped_k<Q>(q, h_rows, slot_base, slot_bytes, off, jobs, ystage, y_stride, K, N, P, deps)
    IE_BANK_DISPATCH(kind, IE_BANK_CALL)
#undef IE_BANK_CALL
}

sycl::event qwen4_bank_gemv_grouped_tiles(sycl::queue& q, BankQ kind, const sycl::half* A_mega, const uint8_t* slot_base,
                            uint64_t slot_bytes, uint64_t off0, uint64_t off1, const int32_t* tile_jobs,
                            sycl::half* out0, sycl::half* out1, uint32_t K, uint32_t N, uint32_t J, uint32_t mats,
                            const std::vector<sycl::event>& deps) {
    if (mats < 1 || mats > 2 || J == 0) return {};
#define IE_BANK_CALL(Q) tiles_k<Q>(q, A_mega, slot_base, slot_bytes, off0, off1, tile_jobs, out0, out1, K, N, J, mats, deps)
    IE_BANK_DISPATCH(kind, IE_BANK_CALL)
#undef IE_BANK_CALL
}

sycl::event qwen4_bank_gemm(sycl::queue& q, BankQ kind, const sycl::half* A, const void* W, sycl::half* y,
                            uint32_t M, uint32_t K, uint32_t N, const std::vector<sycl::event>& deps) {
    if (M == 0) return {};
#define IE_BANK_CALL(Q) gemm_k<Q>(q, A, W, y, M, K, N, deps)
    IE_BANK_DISPATCH(kind, IE_BANK_CALL)
#undef IE_BANK_CALL
}

// ---- int-dot (W5 / W6 x A8) variants of the K-quant gate/up kernels (2026-11-07) ------------------------------------------
// The A16 Q6_K grouped-dual kernel ran ALU-bound: 12.5 ms a decode step on the UD-Q6_K_XL at 103 GB/s, where the Q4_K
// int-dot kernel (gemv_q4_K_q8s_grouped_dual) runs 2.6 ms at 338 GB/s (docs/research/2026-11-07-flashnext-q6-profile.md).
// These take that kernel's grid (one 16-lane subgroup per output column, 32 columns a work-group) and its activations
// (block_q8_1s: 32 int8, d, the half-sums s0 / s1 = d * sum(q8) over elements 0-15 / 16-31):
//   Q5_K  lane -> sub-block j = lane >> 1 (32 elements, scale sc_j / min m_j), half h = lane & 1 (16 elements = 4 words):
//         q5 = (nibble | qh bit j << 4);  acc += (d*sc_j) * (d8 * dot(q5, q8)) - (dmin*m_j) * s_h
//   Q6_K  lane L -> the 16-element sub-block L (int8 scale sc[L]): half hh = L >> 3, group g = (L & 7) >> 1, is = L & 1;
//         q6 = (nibble | 2 qh bits << 4);  sum((q6 - 32) * q8) = dot(q6, q8) - 32 * sum(q8), and d8 * sum(q8) = s_is, so
//         acc += (d*sc_L) * (d8 * dot(q6, q8) - 32 * s_is).  block_q6_K is 210 bytes (2-byte aligned; odd blocks sit 2 bytes
//         off a dword), so the subgroup stages each block in SLM with coalesced dword loads from the column's dword stream
//         (a column is 4-aligned: K/256 x 210 bytes, 2100 at K = 2560) and each lane picks its 8 words out as halfword pairs
//         (byte-assembled words from global memory ran at 68 GB/s, slower than the A16 kernel; 2026-11-07 bench).
//         block_q5_K is 176 bytes with 16-aligned qs / qh offsets: vector loads straight from global memory.
// Not bit-identical to the A16 kernels (int8-rounded activations, the established Q4_K numerics); dual == solo per pick.
namespace {

// Q6_K, one output column against one Q8_1s row: this lane's 16-element sub-block of every 256-wide super-block. `stage` =
// this subgroup's 2 x 64 dwords of SLM. Block fields: ql at 0, qh at 128, scales at 192, d at 208 (210 bytes).
inline float q8dot_col_q6k(sycl::sub_group sg, uint32_t lane, const uint8_t* col, uint32_t K, const block_q8_1s* xrow,
                           uint32_t* stage) {
    float acc = 0.f;
    const uint32_t nb = K / 256;
    const uint32_t hh = lane >> 3, l8 = lane & 7, g = l8 >> 1, is = l8 & 1;
    const uint32_t ql_off = 64 * hh + 32 * (g & 1) + 16 * is, ql_shift = (g & 2) ? 4 : 0;
    const uint32_t qh_off = 128 + 32 * hh + 16 * is, qh_shift = 2 * g;
    const auto* col32 = reinterpret_cast<const uint32_t*>(col);
    for (uint32_t b = 0; b < nb; ++b) {
        const uint32_t byte0 = 210u * b, d0 = byte0 >> 2, par = byte0 & 3u;   // par = 0 or 2
        const uint32_t cnt = ((byte0 + 209u) >> 2) - d0 + 1;                  // the 53 or 54 dwords covering the block
        uint32_t* st = stage + (b & 1u) * 64;
        #pragma unroll
        for (int r = 0; r < 4; ++r) { const uint32_t i = lane + 16 * r; if (i < cnt) st[i] = col32[d0 + i]; }
        sycl::group_barrier(sg);
        const auto* st16 = reinterpret_cast<const uint16_t*>(st);
        const auto* st8  = reinterpret_cast<const uint8_t*>(st);
        auto w16 = [&](uint32_t off) { const uint32_t u = (par + off) >> 1; return uint32_t(st16[u]) | (uint32_t(st16[u + 1]) << 16); };
        const block_q8_1s& xb = xrow[b * 8 + hh * 4 + g];
        const uint32_t* xq = reinterpret_cast<const uint32_t*>(xb.qs) + 4 * is;
        int32_t idot = 0;
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            const uint32_t lo = (w16(ql_off + 4 * w) >> ql_shift) & 0x0F0F0F0Fu;
            const uint32_t hi = ((w16(qh_off + 4 * w) >> qh_shift) & 0x03030303u) << 4;
            idot = dp4a_us(lo | hi, int32_t(xq[w]), idot);
        }
        const float dsc = h2f(uint16_t(st16[(par + 208u) >> 1])) * float(int8_t(st8[par + 192u + lane]));
        acc += dsc * (xb.d * float(idot) - 32.f * (is ? xb.s1 : xb.s0));
    }
    return acc;
}

// One output column against one Q8_1s row: this lane's sub-block of every 256-wide super-block, fp32 partial.
template <BankQ Q>
inline float q8dot_col(sycl::sub_group sg, uint32_t lane, const uint8_t* col, uint32_t K, const block_q8_1s* xrow,
                       uint32_t* stage) {
    float acc = 0.f;
    const uint32_t nb = K / 256;
    if constexpr (Q == BankQ::kQ6_K) {
        acc = q8dot_col_q6k(sg, lane, col, K, xrow, stage);
    } else {   // Q5_K
        const uint32_t j = lane >> 1, hsel = lane & 1, j4 = j >> 1, shift = (j & 1) ? 4 : 0;
        const auto* W = reinterpret_cast<const block_q5_K*>(col);
        for (uint32_t b = 0; b < nb; ++b) {
            const block_q5_K& blk = W[b];
            const auto qv  = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(blk.qs + 32 * j4 + 16 * hsel);
            const auto qhv = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(blk.qh + 16 * hsel);
            const block_q8_1s& xb = xrow[b * 8 + j];
            const uint32_t* xq = reinterpret_cast<const uint32_t*>(xb.qs) + 4 * hsel;
            int32_t idot = 0;
            #pragma unroll
            for (int w = 0; w < 4; ++w) {
                const uint32_t lo = (qv[w] >> shift) & 0x0F0F0F0Fu;
                const uint32_t hb = ((qhv[w] >> j) & 0x01010101u) << 4;
                idot = dp4a_us(lo | hb, int32_t(xq[w]), idot);
            }
            float sc, m;
            scale_min_k4(int(j), blk.scales, sc, m);
            acc += (h2f(blk.d) * sc) * (xb.d * float(idot)) - (h2f(blk.dmin) * m) * (hsel ? xb.s1 : xb.s0);
        }
    }
    return acc;
}

// The tiles body: one output column against T (<= 16) Q8_1s rows (row t at xrows + t * q8_blocks). The lane's packed weight
// words of each super-block are built once and dotted against every row; per row the fp32 expression is the single-row
// kernel's, so each row is bit-identical to qwen4_bank_gemv_q8s on that row.
template <BankQ Q>
inline void q8dot_col_rows(sycl::sub_group sg, uint32_t lane, const uint8_t* col, uint32_t K, const block_q8_1s* xrows,
                           uint32_t q8_blocks, uint32_t T, uint32_t* stage, float* acc) {
    const uint32_t nb = K / 256;
    if constexpr (Q == BankQ::kQ6_K) {
        const uint32_t hh = lane >> 3, l8 = lane & 7, g = l8 >> 1, is = l8 & 1;
        const uint32_t ql_off = 64 * hh + 32 * (g & 1) + 16 * is, ql_shift = (g & 2) ? 4 : 0;
        const uint32_t qh_off = 128 + 32 * hh + 16 * is, qh_shift = 2 * g;
        const auto* col32 = reinterpret_cast<const uint32_t*>(col);
        for (uint32_t b = 0; b < nb; ++b) {
            const uint32_t byte0 = 210u * b, d0 = byte0 >> 2, par = byte0 & 3u;
            const uint32_t cnt = ((byte0 + 209u) >> 2) - d0 + 1;
            uint32_t* st = stage + (b & 1u) * 64;
            #pragma unroll
            for (int r = 0; r < 4; ++r) { const uint32_t i = lane + 16 * r; if (i < cnt) st[i] = col32[d0 + i]; }
            sycl::group_barrier(sg);
            const auto* st16 = reinterpret_cast<const uint16_t*>(st);
            const auto* st8  = reinterpret_cast<const uint8_t*>(st);
            auto w16 = [&](uint32_t off) { const uint32_t u = (par + off) >> 1; return uint32_t(st16[u]) | (uint32_t(st16[u + 1]) << 16); };
            uint32_t wq[4];
            #pragma unroll
            for (int w = 0; w < 4; ++w)
                wq[w] = ((w16(ql_off + 4 * w) >> ql_shift) & 0x0F0F0F0Fu) | (((w16(qh_off + 4 * w) >> qh_shift) & 0x03030303u) << 4);
            const float dsc = h2f(uint16_t(st16[(par + 208u) >> 1])) * float(int8_t(st8[par + 192u + lane]));
            const uint32_t xblk = b * 8 + hh * 4 + g;
            #pragma unroll
            for (int t = 0; t < T_MAX; ++t) {
                if (uint32_t(t) >= T) break;
                const block_q8_1s& xb = xrows[uint64_t(t) * q8_blocks + xblk];
                const uint32_t* xq = reinterpret_cast<const uint32_t*>(xb.qs) + 4 * is;
                int32_t idot = 0;
                #pragma unroll
                for (int w = 0; w < 4; ++w) idot = dp4a_us(wq[w], int32_t(xq[w]), idot);
                acc[t] += dsc * (xb.d * float(idot) - 32.f * (is ? xb.s1 : xb.s0));
            }
        }
    } else {   // Q5_K
        const uint32_t j = lane >> 1, hsel = lane & 1, j4 = j >> 1, shift = (j & 1) ? 4 : 0;
        const auto* W = reinterpret_cast<const block_q5_K*>(col);
        for (uint32_t b = 0; b < nb; ++b) {
            const block_q5_K& blk = W[b];
            const auto qv  = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(blk.qs + 32 * j4 + 16 * hsel);
            const auto qhv = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(blk.qh + 16 * hsel);
            uint32_t wq[4];
            #pragma unroll
            for (int w = 0; w < 4; ++w) wq[w] = ((qv[w] >> shift) & 0x0F0F0F0Fu) | (((qhv[w] >> j) & 0x01010101u) << 4);
            float sc, m;
            scale_min_k4(int(j), blk.scales, sc, m);
            const float dsc = h2f(blk.d) * sc, dmm = h2f(blk.dmin) * m;
            const uint32_t xblk = b * 8 + j;
            #pragma unroll
            for (int t = 0; t < T_MAX; ++t) {
                if (uint32_t(t) >= T) break;
                const block_q8_1s& xb = xrows[uint64_t(t) * q8_blocks + xblk];
                const uint32_t* xq = reinterpret_cast<const uint32_t*>(xb.qs) + 4 * hsel;
                int32_t idot = 0;
                #pragma unroll
                for (int w = 0; w < 4; ++w) idot = dp4a_us(wq[w], int32_t(xq[w]), idot);
                acc[t] += dsc * (xb.d * float(idot)) - dmm * (hsel ? xb.s1 : xb.s0);
            }
        }
    }
}

template <BankQ Q>
sycl::event grouped_tiles_q8s_k(sycl::queue& q, const void* x_q8s_mega, const uint8_t* slot_base, uint64_t slot_bytes,
                                uint64_t gate_off, uint64_t up_off, const int32_t* tile_jobs, sycl::half* g_out,
                                sycl::half* u_out, uint32_t K, uint32_t N, uint32_t J, const std::vector<sycl::event>& deps) {
    constexpr int NPW = 16, WGI = NPW * SG_SIZE;
    const uint64_t cb = col_bytes_of<Q>(K);
    const uint32_t n_wgs = (N + NPW - 1) / NPW, q8_blocks = K / 32;
    const auto* X8 = static_cast<const block_q8_1s*>(x_q8s_mega);
    return ie::ps(q, "q4e_bank_q8s_gtile", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<uint32_t, 1> stage(sycl::range<1>(NPW * 128), h);
        h.parallel_for(sycl::nd_range<1>(uint64_t(J) * 2 * n_wgs * WGI, WGI),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid = uint32_t(it.get_local_id(0)), gid = uint32_t(it.get_group(0));
            const uint32_t j = gid / (2 * n_wgs), rem = gid % (2 * n_wgs);
            const bool is_up = rem >= n_wgs;
            const uint32_t wgid = is_up ? rem - n_wgs : rem;
            const uint32_t sg_id = lid / SG_SIZE, lane = lid % SG_SIZE;
            const uint32_t n = wgid * NPW + sg_id;
            if (n >= N) return;
            const uint32_t x0 = uint32_t(tile_jobs[j * 4 + 0]), T = uint32_t(tile_jobs[j * 4 + 1]);
            const uint32_t slot = uint32_t(tile_jobs[j * 4 + 2]), y0 = uint32_t(tile_jobs[j * 4 + 3]);
            const uint8_t* W = slot_base + uint64_t(slot) * slot_bytes + (is_up ? up_off : gate_off);
            sycl::half* y = (is_up ? u_out : g_out) + uint64_t(y0) * N;
            uint32_t* st = stage.get_multi_ptr<sycl::access::decorated::no>().get() + sg_id * 128;
            float acc[T_MAX];
            #pragma unroll
            for (int t = 0; t < T_MAX; ++t) acc[t] = 0.f;
            q8dot_col_rows<Q>(it.get_sub_group(), lane, W + uint64_t(n) * cb, K, X8 + uint64_t(x0) * q8_blocks, q8_blocks, T, st, acc);
            #pragma unroll
            for (int t = 0; t < T_MAX; ++t) {
                if (uint32_t(t) >= T) break;
                const float r = sycl::reduce_over_group(it.get_sub_group(), acc[t], sycl::plus<float>());
                if (lane == 0) y[uint64_t(t) * N + n] = sycl::half(r);
            }
        });
    });
}

template <BankQ Q>
sycl::event gemv_q8s_k(sycl::queue& q, const void* x_q8s, const void* Wv, sycl::half* y, uint32_t K, uint32_t N,
                       const std::vector<sycl::event>& deps) {
    constexpr int NPW = 32, WGI = NPW * SG_SIZE;
    const uint64_t cb = col_bytes_of<Q>(K);
    const uint32_t n_wgs = (N + NPW - 1) / NPW;
    const auto* W = static_cast<const uint8_t*>(Wv);
    const auto* X8 = static_cast<const block_q8_1s*>(x_q8s);
    return ie::ps(q, "q4e_bank_q8s_gemv", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<uint32_t, 1> stage(sycl::range<1>(NPW * 128), h);   // 2 x 64 dwords a subgroup (Q6_K staging)
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * WGI, WGI),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid = uint32_t(it.get_local_id(0));
            const uint32_t sg_id = lid / SG_SIZE, lane = lid % SG_SIZE;
            const uint32_t n = uint32_t(it.get_group(0)) * NPW + sg_id;
            if (n >= N) return;
            uint32_t* st = stage.get_multi_ptr<sycl::access::decorated::no>().get() + sg_id * 128;
            float acc = q8dot_col<Q>(it.get_sub_group(), lane, W + uint64_t(n) * cb, K, X8, st);
            acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
            if (lane == 0) y[n] = sycl::half(acc);
        });
    });
}

template <BankQ Q>
sycl::event grouped_dual_q8s_k(sycl::queue& q, const void* x_q8s, const uint8_t* slot_base, uint64_t slot_bytes,
                               uint64_t gate_off, uint64_t up_off, const int32_t* jobs, sycl::half* g_out, sycl::half* u_out,
                               uint32_t K, uint32_t N, uint32_t P, const std::vector<sycl::event>& deps) {
    constexpr int NPW = 32, WGI = NPW * SG_SIZE;
    const uint64_t cb = col_bytes_of<Q>(K);
    const uint32_t n_wgs = (N + NPW - 1) / NPW, q8_blocks = K / 32;
    const auto* X8 = static_cast<const block_q8_1s*>(x_q8s);
    return ie::ps(q, "q4e_bank_q8s_grp_dual", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<uint32_t, 1> stage(sycl::range<1>(NPW * 128), h);   // 2 x 64 dwords a subgroup (Q6_K staging)
        h.parallel_for(sycl::nd_range<1>(uint64_t(P) * 2 * n_wgs * WGI, WGI),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid = uint32_t(it.get_local_id(0)), gid = uint32_t(it.get_group(0));
            const uint32_t pk = gid / (2 * n_wgs), rem = gid % (2 * n_wgs);
            const bool is_up = rem >= n_wgs;
            const uint32_t wgid = is_up ? rem - n_wgs : rem;
            const uint32_t sg_id = lid / SG_SIZE, lane = lid % SG_SIZE;
            const uint32_t n = wgid * NPW + sg_id;
            if (n >= N) return;
            const uint32_t vt = uint32_t(jobs[pk * 3 + 0]), slot = uint32_t(jobs[pk * 3 + 1]);
            const uint8_t* W = slot_base + uint64_t(slot) * slot_bytes + (is_up ? up_off : gate_off);
            uint32_t* st = stage.get_multi_ptr<sycl::access::decorated::no>().get() + sg_id * 128;
            float acc = q8dot_col<Q>(it.get_sub_group(), lane, W + uint64_t(n) * cb, K, X8 + uint64_t(vt) * q8_blocks, st);
            acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
            if (lane == 0) (is_up ? u_out : g_out)[uint64_t(pk) * N + n] = sycl::half(acc);
        });
    });
}

// grouped_dual_q8s_k with swiglu fused into the epilogue (2026-11-08): one work-group per (pick, 32-column strip), each
// subgroup the gate and then the up column through q8dot_col (each the dual kernel's exact chain), lane 0 writing
// h = silu(f16(g)) * f16(u) exactly as swiglu() computes it. Bit-identical to dual + swiglu (ie-qwen4-bank-test).
template <BankQ Q>
sycl::event grouped_dual_q8s_swiglu_k(sycl::queue& q, const void* x_q8s, const uint8_t* slot_base, uint64_t slot_bytes,
                                      uint64_t gate_off, uint64_t up_off, const int32_t* jobs, sycl::half* h_out,
                                      uint32_t K, uint32_t N, uint32_t P, const std::vector<sycl::event>& deps) {
    constexpr int NPW = 32, WGI = NPW * SG_SIZE;
    const uint64_t cb = col_bytes_of<Q>(K);
    const uint32_t n_wgs = (N + NPW - 1) / NPW, q8_blocks = K / 32;
    const auto* X8 = static_cast<const block_q8_1s*>(x_q8s);
    return ie::ps(q, "q4e_bank_q8s_grp_dual_sw", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<uint32_t, 1> stage(sycl::range<1>(NPW * 128), h);   // 2 x 64 dwords a subgroup (Q6_K staging)
        h.parallel_for(sycl::nd_range<1>(uint64_t(P) * n_wgs * WGI, WGI),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid = uint32_t(it.get_local_id(0)), gid = uint32_t(it.get_group(0));
            const uint32_t pk = gid / n_wgs, wgid = gid % n_wgs;
            const uint32_t sg_id = lid / SG_SIZE, lane = lid % SG_SIZE;
            const uint32_t n = wgid * NPW + sg_id;
            if (n >= N) return;
            const uint32_t vt = uint32_t(jobs[pk * 3 + 0]), slot = uint32_t(jobs[pk * 3 + 1]);
            const uint8_t* sb = slot_base + uint64_t(slot) * slot_bytes;
            const block_q8_1s* xrow = X8 + uint64_t(vt) * q8_blocks;
            uint32_t* st = stage.get_multi_ptr<sycl::access::decorated::no>().get() + sg_id * 128;
            float acc_g = q8dot_col<Q>(it.get_sub_group(), lane, sb + gate_off + uint64_t(n) * cb, K, xrow, st);
            float acc_u = q8dot_col<Q>(it.get_sub_group(), lane, sb + up_off   + uint64_t(n) * cb, K, xrow, st);
            acc_g = sycl::reduce_over_group(it.get_sub_group(), acc_g, sycl::plus<float>());
            acc_u = sycl::reduce_over_group(it.get_sub_group(), acc_u, sycl::plus<float>());
            if (lane == 0) {
                const float gf = float(sycl::half(acc_g)), uf = float(sycl::half(acc_u));
                const float s = gf / (1.f + sycl::native::exp(-gf));
                h_out[uint64_t(pk) * N + n] = sycl::half(s * uf);
            }
        });
    });
}

}  // namespace

bool qwen4_bank_has_q8dot(BankQ kind) noexcept { return kind == BankQ::kQ5_K || kind == BankQ::kQ6_K; }

sycl::event qwen4_bank_gemv_q8s(sycl::queue& q, BankQ kind, const void* x_q8s, const void* W, sycl::half* y,
                                uint32_t K, uint32_t N, const std::vector<sycl::event>& deps) {
    switch (kind) {
        case BankQ::kQ5_K: return gemv_q8s_k<BankQ::kQ5_K>(q, x_q8s, W, y, K, N, deps);
        case BankQ::kQ6_K: return gemv_q8s_k<BankQ::kQ6_K>(q, x_q8s, W, y, K, N, deps);
        default: return {};
    }
}

sycl::event qwen4_bank_gemv_grouped_tiles_q8s(sycl::queue& q, BankQ kind, const void* x_q8s_mega, const uint8_t* slot_base,
                                              uint64_t slot_bytes, uint64_t gate_off, uint64_t up_off, const int32_t* tile_jobs,
                                              sycl::half* g_out, sycl::half* u_out, uint32_t K, uint32_t N, uint32_t J,
                                              const std::vector<sycl::event>& deps) {
    switch (kind) {
        case BankQ::kQ5_K: return grouped_tiles_q8s_k<BankQ::kQ5_K>(q, x_q8s_mega, slot_base, slot_bytes, gate_off, up_off, tile_jobs, g_out, u_out, K, N, J, deps);
        case BankQ::kQ6_K: return grouped_tiles_q8s_k<BankQ::kQ6_K>(q, x_q8s_mega, slot_base, slot_bytes, gate_off, up_off, tile_jobs, g_out, u_out, K, N, J, deps);
        default: return {};
    }
}

sycl::event qwen4_bank_gemv_grouped_dual_q8s(sycl::queue& q, BankQ kind, const void* x_q8s, const uint8_t* slot_base,
                                             uint64_t slot_bytes, uint64_t gate_off, uint64_t up_off, const int32_t* jobs,
                                             sycl::half* g_out, sycl::half* u_out, uint32_t K, uint32_t N, uint32_t P,
                                             const std::vector<sycl::event>& deps) {
    switch (kind) {
        case BankQ::kQ5_K: return grouped_dual_q8s_k<BankQ::kQ5_K>(q, x_q8s, slot_base, slot_bytes, gate_off, up_off, jobs, g_out, u_out, K, N, P, deps);
        case BankQ::kQ6_K: return grouped_dual_q8s_k<BankQ::kQ6_K>(q, x_q8s, slot_base, slot_bytes, gate_off, up_off, jobs, g_out, u_out, K, N, P, deps);
        default: return {};
    }
}

sycl::event qwen4_bank_gemv_grouped_dual_q8s_swiglu(sycl::queue& q, BankQ kind, const void* x_q8s, const uint8_t* slot_base,
                                                    uint64_t slot_bytes, uint64_t gate_off, uint64_t up_off, const int32_t* jobs,
                                                    sycl::half* h_out, uint32_t K, uint32_t N, uint32_t P,
                                                    const std::vector<sycl::event>& deps) {
    switch (kind) {
        case BankQ::kQ5_K: return grouped_dual_q8s_swiglu_k<BankQ::kQ5_K>(q, x_q8s, slot_base, slot_bytes, gate_off, up_off, jobs, h_out, K, N, P, deps);
        case BankQ::kQ6_K: return grouped_dual_q8s_swiglu_k<BankQ::kQ6_K>(q, x_q8s, slot_base, slot_bytes, gate_off, up_off, jobs, h_out, K, N, P, deps);
        default: return {};
    }
}

}  // namespace ie
