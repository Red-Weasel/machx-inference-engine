// src/ops/gemv_q4_0.cpp — W4A16 GEMV/GEMM with Q4_0 weights.
//
// Q4_0 is the quant format of the Gemma 4 QAT GGUFs. Block = 32 elements, 18
// bytes: one FP16 scale `d` + 16 bytes of 4-bit quants. ggml nibble layout:
// qs[i] low nibble = weight i, high nibble = weight i+16 (i in 0..15), and
//   w[i]    = d * ((qs[i] & 0x0F) - 8)
//   w[i+16] = d * ((qs[i] >> 4)  - 8)   (symmetric, offset 8, no min).
//
// Weight layout matches gemv_q4_K: W[K, N] column-major-packed — each output
// column n is K/32 consecutive blocks. y[1,N] = A[1,K] @ W[K,N], K % 32 == 0.
//
// Correctness-first kernel (mirrors gemv_q4_K's WG/SG shape, simpler dequant):
// one subgroup per output column; lane L reads qs[L] and contributes elements
// L and L+16 of each block; fp32 accumulate; subgroup-reduce; lane 0 writes.

#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"
#include "ie/dp4a.hpp"
#include <cstdlib>

#include <sycl/sycl.hpp>
#include "ie/kernel_profiler.hpp"

namespace ie {

namespace {
inline float q40_fp16_to_fp32(uint16_t h) {
    const uint32_t s = uint32_t(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1fu;
    uint32_t m =  h        & 0x3ffu;
    if (e == 0) {
        if (m == 0) return sycl::bit_cast<float>(s);
        while ((m & 0x400u) == 0) { m <<= 1; e -= 1; }
        e += 1;
        m &= ~0x400u;
    } else if (e == 31) {
        return sycl::bit_cast<float>(s | 0x7f800000u | (m << 13));
    }
    e += (127 - 15);
    return sycl::bit_cast<float>(s | (e << 23) | (m << 13));
}
// Sub-groups per WG for the SoA-Q4_0 decode GEMVs. 64 measured best on 31B
// (vs 32/16/8); IE_GEMV_NPWG overrides. Read once (this is a per-call hot path).
inline uint32_t soa_npwg() {
    static const uint32_t v = []{
        if (const char* e = std::getenv("IE_GEMV_NPWG")) {
            int n = std::atoi(e); if (n==8||n==16||n==32||n==64) return uint32_t(n);
        }
        return 64u;
    }();
    return v;
}
}  // namespace

sycl::event gemv_q4_0(sycl::queue& q,
                      const sycl::half* A, const void* W_packed,
                      sycl::half* y,
                      uint32_t K, uint32_t N,
                      const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE  = 16;
    constexpr int N_PER_WG = 16;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;       // 256
    const auto* W = static_cast<const block_q4_0*>(W_packed);
    const uint32_t blocks_per_col = K / 32;
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "gemv_q4_0", [&](sycl::handler& h) {
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

            float acc = 0.f;
            const block_q4_0* col = &W[uint64_t(n) * blocks_per_col];
            for (uint32_t b = 0; b < blocks_per_col; ++b) {
                const block_q4_0& blk = col[b];
                const float d = q40_fp16_to_fp32(blk.d);
                const uint8_t qb = blk.qs[lane];
                const float a_lo = float(A_slm[b * 32 + lane]);
                const float a_hi = float(A_slm[b * 32 + lane + 16]);
                acc += d * (a_lo * float(int(qb & 0x0F) - 8) +
                            a_hi * float(int(qb >> 4)   - 8));
            }
            acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
            if (lane == 0) y[n] = sycl::half(acc);
        });
    });
}

// Multi-row GEMM: y[M,N] = A[M,K] @ W[K,N]. Same Q4_0 layout. M rows per WG via
// per-lane M accumulators; weight read once per (block, column), reused across
// rows. Caller may pass any M (tiled by M_TILE over grid dim 0).
sycl::event gemm_q4_0(sycl::queue& q,
                      const sycl::half* A, const void* W_packed,
                      sycl::half* y,
                      uint32_t M, uint32_t K, uint32_t N,
                      const std::vector<sycl::event>& deps) {
    if (M == 0) return {};
    constexpr int SG_SIZE  = 16;
    constexpr int N_PER_WG = 16;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;       // 256
    constexpr int M_TILE   = 16;
    const auto* W = static_cast<const block_q4_0*>(W_packed);
    const uint32_t blocks_per_col = K / 32;
    const uint32_t n_wgs   = (N + N_PER_WG - 1) / N_PER_WG;
    const uint32_t m_tiles = (M + M_TILE - 1) / M_TILE;
    const uint32_t slm_rows = std::min<uint32_t>(M, M_TILE);

    return ie::ps(q, "gemm_q4_0", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<sycl::half, 1> A_slm(uint64_t(slm_rows) * K, h);
        h.parallel_for(sycl::nd_range<2>({m_tiles, uint64_t(n_wgs) * WG_ITEMS},
                                         {1, WG_ITEMS}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(1));
            const uint32_t wgid  = uint32_t(it.get_group(1));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t n     = wgid * N_PER_WG + sg_id;
            const uint32_t m0    = uint32_t(it.get_group(0)) * M_TILE;
            const uint32_t Mc    = sycl::min(uint32_t(M_TILE), M - m0);

            const uint64_t a_total = uint64_t(Mc) * K;
            for (uint64_t i = lid; i < a_total; i += WG_ITEMS)
                A_slm[i] = A[uint64_t(m0) * K + i];
            sycl::group_barrier(it.get_group());
            if (n >= N) return;

            float acc[M_TILE];
            #pragma unroll
            for (int mm = 0; mm < M_TILE; ++mm) acc[mm] = 0.f;

            const block_q4_0* col = &W[uint64_t(n) * blocks_per_col];
            for (uint32_t b = 0; b < blocks_per_col; ++b) {
                const block_q4_0& blk = col[b];
                const float d = q40_fp16_to_fp32(blk.d);
                const uint8_t qb = blk.qs[lane];
                const float w_lo = d * float(int(qb & 0x0F) - 8);
                const float w_hi = d * float(int(qb >> 4)   - 8);
                #pragma unroll
                for (int mm = 0; mm < M_TILE; ++mm) {
                    if (uint32_t(mm) < Mc) {
                        const sycl::half* a = &A_slm[uint64_t(mm) * K + b * 32];
                        acc[mm] += float(a[lane]) * w_lo + float(a[lane + 16]) * w_hi;
                    }
                }
            }
            #pragma unroll
            for (int mm = 0; mm < M_TILE; ++mm) {
                if (uint32_t(mm) < Mc) {
                    const float r = sycl::reduce_over_group(it.get_sub_group(), acc[mm],
                                                            sycl::plus<float>());
                    if (lane == 0) y[uint64_t(m0 + mm) * N + n] = sycl::half(r);
                }
            }
        });
    });
}

// ---------------------------------------------------------------------------
// SoA-Q4_0 fast decode GEMV (W4A8 int-dot), mirroring gemv_q6_soa_q8 (~80% BW).
// The on-disk AoS block_q4_0 interleaves the fp16 scale every 18 B, flooring the
// decode GEMV at ~20% of B70's 608 GB/s. Repack at load into two contiguous
// per-column streams (nibbles + scales) so the kernel does coalesced loads +
// int8 dp4a. Q4_0 is simplest: one fp16 scale/32-block, w = d*(nib-8), and the
// q8 activation block stores s = d8*Σq8 so the −8 offset needs no dp4a-ones.
//   q4_qs[n*(K/2) + b*16 + i]  = block b's qs byte i (elem i low / elem i+16 hi)
//   q4_d [n*(K/32) + b]        = block b's fp16 scale (raw uint16)
// Footprint K/2 + (K/32)*2 = 0.5625 B/elem = 4.5 bpw (== Q4_0, no blowup).
void repack_q4_0_to_soa(const void* W_blocks, uint32_t K, uint32_t N,
                        uint8_t* q4_qs, uint16_t* q4_d) {
    const uint32_t bpc = K / 32;
    const auto* blocks = static_cast<const block_q4_0*>(W_blocks);
    for (uint64_t n = 0; n < N; ++n) {
        const block_q4_0* col = blocks + n * bpc;
        uint8_t*  qs_col = q4_qs + n * (uint64_t(K) / 2);
        uint16_t* d_col  = q4_d  + n * bpc;
        for (uint32_t b = 0; b < bpc; ++b) {
            d_col[b] = *reinterpret_cast<const uint16_t*>(&col[b].d);
            for (int i = 0; i < 16; ++i) qs_col[uint64_t(b) * 16 + i] = col[b].qs[i];
        }
    }
}

// y[1,N] = dequant(x_q8)[1,K] @ Q4_0-SoA W[K,N].  Split-K: the 16 lanes of an SG
// stride whole 32-elem blocks of one column; SG-reduce sums it.  Per block: read
// 16 qs bytes (4 uint32), split lo/hi nibbles → dp4a vs the staged q8 words; fold
// d4*(d8*idot − 8*s).  q8 staged once per WG (shared by all N_PER_WG columns).
sycl::event gemv_q4_0_soa_q8(sycl::queue& q,
                             const void* x_q8, const uint8_t* q4_qs,
                             const uint16_t* q4_d, sycl::half* y,
                             uint32_t K, uint32_t N,
                             const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE  = 16;
    // N_PER_WG = sub-groups per work-group. The q8 SLM staging is the occupancy
    // limiter; 64 measured best on 31B decode (vs 32/16/8). IE_GEMV_NPWG overrides.
    const uint32_t N_PER_WG = soa_npwg();
    const uint32_t WG_ITEMS = N_PER_WG * SG_SIZE;
    const auto* X8 = static_cast<const block_q8_1x*>(x_q8);
    const uint32_t blocks_per_col = K / 32;
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "gemv_q4_0_soa", [&](sycl::handler& h) {
        h.depends_on(deps);
        // Stage the Q8 activation once per WG: qs (8 uint32/block) + d + s.
        sycl::local_accessor<uint32_t, 1> q8s(blocks_per_col * 8, h);
        sycl::local_accessor<float, 1>    q8d(blocks_per_col, h);
        sycl::local_accessor<float, 1>    q8sm(blocks_per_col, h);
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t wgid  = uint32_t(it.get_group(0));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t n     = wgid * N_PER_WG + sg_id;

            for (uint32_t i = lid; i < blocks_per_col * 8; i += WG_ITEMS) {
                const uint32_t blk = i / 8, w = i % 8;
                q8s[i] = reinterpret_cast<const uint32_t*>(X8[blk].qs)[w];
            }
            for (uint32_t i = lid; i < blocks_per_col; i += WG_ITEMS) {
                q8d[i]  = X8[i].d;
                q8sm[i] = X8[i].s;
            }
            sycl::group_barrier(it.get_group());
            if (n >= N) return;

            const uint8_t*  qs_col = q4_qs + uint64_t(n) * (uint64_t(K) / 2);
            const uint16_t* d_col  = q4_d  + uint64_t(n) * blocks_per_col;

            float acc = 0.f;
            for (uint32_t b = lane; b < blocks_per_col; b += SG_SIZE) {
                const float d4 = q40_fp16_to_fp32(d_col[b]);
                const uint32_t* qs_w =
                    reinterpret_cast<const uint32_t*>(qs_col + uint64_t(b) * 16);
                const uint32_t* q8w = &q8s[b * 8];
                int32_t idot = 0;
                #pragma unroll
                for (int w = 0; w < 4; ++w) {
                    const uint32_t qw = qs_w[w];
                    const uint32_t lo = qw & 0x0F0F0F0Fu;          // elems 4w..4w+3
                    const uint32_t hi = (qw >> 4) & 0x0F0F0F0Fu;   // elems 4w+16..+19
                    idot = ie::dp4a_us(lo, int32_t(q8w[w]),     idot);
                    idot = ie::dp4a_us(hi, int32_t(q8w[w + 4]), idot);
                }
                acc += d4 * (q8d[b] * float(idot) - 8.0f * q8sm[b]);
            }
            acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
            if (lane == 0) y[n] = sycl::half(acc);
        });
    });
}

// Fused multi-bank decode GEMV — see ops.hpp. Up to 3 SoA-Q4_0 projections that
// share one quantized activation (same K) run in ONE launch: WGs tile the
// concatenated column space [0, ΣN_b), each WG resolves its bank from precomputed
// WG offsets. Staging + inner loop are identical to gemv_q4_0_soa_q8.
sycl::event gemv_q4_0_soa_q8_multi(sycl::queue& q,
                                   const void* x_q8,
                                   const uint8_t* const qs[3],
                                   const uint16_t* const d[3],
                                   sycl::half* const y[3],
                                   const uint32_t N[3], int n_banks,
                                   uint32_t K,
                                   const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE = 16;
    const uint32_t N_PER_WG = soa_npwg();
    const uint32_t WG_ITEMS = N_PER_WG * SG_SIZE;
    const auto* X8 = static_cast<const block_q8_1x*>(x_q8);
    const uint32_t blocks_per_col = K / 32;

    // Per-bank pointers/sizes + cumulative WG offsets, captured by value.
    const uint8_t*  qsb[3] = {nullptr,nullptr,nullptr};
    const uint16_t* db [3] = {nullptr,nullptr,nullptr};
    sycl::half*     yb [3] = {nullptr,nullptr,nullptr};
    uint32_t        Nb [3] = {0,0,0};
    uint32_t        wgoff[4] = {0,0,0,0};
    uint32_t total_wgs = 0;
    for (int b = 0; b < n_banks; ++b) {
        qsb[b] = qs[b]; db[b] = d[b]; yb[b] = y[b]; Nb[b] = N[b];
        wgoff[b] = total_wgs;
        total_wgs += (N[b] + N_PER_WG - 1) / N_PER_WG;
    }
    wgoff[n_banks] = total_wgs;
    const int nb = n_banks;

    return ie::ps(q, "gemv_q4_0_soa_multi", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<uint32_t, 1> q8s(blocks_per_col * 8, h);
        sycl::local_accessor<float, 1>    q8d(blocks_per_col, h);
        sycl::local_accessor<float, 1>    q8sm(blocks_per_col, h);
        h.parallel_for(sycl::nd_range<1>(uint64_t(total_wgs) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t wgid  = uint32_t(it.get_group(0));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;

            // Resolve this WG's bank (n_banks <= 3 → linear scan).
            int bank = 0;
            #pragma unroll
            for (int b = 1; b < 3; ++b) if (b < nb && wgid >= wgoff[b]) bank = b;
            const uint32_t n = (wgid - wgoff[bank]) * N_PER_WG + sg_id;

            for (uint32_t i = lid; i < blocks_per_col * 8; i += WG_ITEMS) {
                const uint32_t blk = i / 8, w = i % 8;
                q8s[i] = reinterpret_cast<const uint32_t*>(X8[blk].qs)[w];
            }
            for (uint32_t i = lid; i < blocks_per_col; i += WG_ITEMS) {
                q8d[i]  = X8[i].d;
                q8sm[i] = X8[i].s;
            }
            sycl::group_barrier(it.get_group());
            if (n >= Nb[bank]) return;

            const uint8_t*  qs_col = qsb[bank] + uint64_t(n) * (uint64_t(K) / 2);
            const uint16_t* d_col  = db[bank]  + uint64_t(n) * blocks_per_col;

            float acc = 0.f;
            for (uint32_t b = lane; b < blocks_per_col; b += SG_SIZE) {
                const float d4 = q40_fp16_to_fp32(d_col[b]);
                const uint32_t* qs_w =
                    reinterpret_cast<const uint32_t*>(qs_col + uint64_t(b) * 16);
                const uint32_t* q8w = &q8s[b * 8];
                int32_t idot = 0;
                #pragma unroll
                for (int w = 0; w < 4; ++w) {
                    const uint32_t qw = qs_w[w];
                    const uint32_t lo = qw & 0x0F0F0F0Fu;
                    const uint32_t hi = (qw >> 4) & 0x0F0F0F0Fu;
                    idot = ie::dp4a_us(lo, int32_t(q8w[w]),     idot);
                    idot = ie::dp4a_us(hi, int32_t(q8w[w + 4]), idot);
                }
                acc += d4 * (q8d[b] * float(idot) - 8.0f * q8sm[b]);
            }
            acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
            if (lane == 0) yb[bank][n] = sycl::half(acc);
        });
    });
}

// BATCHED-T variant of gemv_q4_0_soa_q8 (spec-decode VERIFY). Reads each Q4_0-SoA
// weight column ONCE and dots it against T staged Q8_1 activation rows → y[T,N]
// (y[t*N+n]). x_q8 = T contiguous block_q8_1x streams (row t at block offset
// t*(K/32)), from quantize_q8_1 over the [T,K] fp16 activations. Per-row numerics
// IDENTICAL to gemv_q4_0_soa_q8 (lossless gate). The weight BW is paid once for T
// outputs — this is what amortizes a T=K verify forward toward one decode's cost.
sycl::event gemv_q4_0_soa_q8_batched(sycl::queue& q,
                                     const void* x_q8, const uint8_t* q4_qs,
                                     const uint16_t* q4_d, sycl::half* y,
                                     uint32_t K, uint32_t N, uint32_t T,
                                     const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE = 16;
    constexpr int T_MAX   = 8;                  // spec verify: n_max draft + 1
    const uint32_t N_PER_WG = soa_npwg();
    const uint32_t WG_ITEMS = N_PER_WG * SG_SIZE;
    const auto* X8 = static_cast<const block_q8_1x*>(x_q8);
    const uint32_t blocks_per_col = K / 32;     // == q8 blocks per row
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "gemv_q4_0_soa_T", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t wgid  = uint32_t(it.get_group(0));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t n     = wgid * N_PER_WG + sg_id;
            if (n >= N) return;

            const uint8_t*  qs_col = q4_qs + uint64_t(n) * (uint64_t(K) / 2);
            const uint16_t* d_col  = q4_d  + uint64_t(n) * blocks_per_col;

            float acc[T_MAX];
            #pragma unroll
            for (int t = 0; t < T_MAX; ++t) acc[t] = 0.f;

            for (uint32_t b = lane; b < blocks_per_col; b += SG_SIZE) {
                const float d4 = q40_fp16_to_fp32(d_col[b]);
                // Weight nibbles read ONCE per column-block, reused across T rows.
                const uint32_t* qs_w =
                    reinterpret_cast<const uint32_t*>(qs_col + uint64_t(b) * 16);
                uint32_t lo[4], hi[4];
                #pragma unroll
                for (int w = 0; w < 4; ++w) {
                    lo[w] = qs_w[w] & 0x0F0F0F0Fu;
                    hi[w] = (qs_w[w] >> 4) & 0x0F0F0F0Fu;
                }
                #pragma unroll
                for (int t = 0; t < T_MAX; ++t) {
                    if (uint32_t(t) >= T) break;
                    const block_q8_1x& xb = X8[uint64_t(t) * blocks_per_col + b];
                    const uint32_t* q8w = reinterpret_cast<const uint32_t*>(xb.qs);
                    int32_t idot = 0;
                    #pragma unroll
                    for (int w = 0; w < 4; ++w) {
                        idot = ie::dp4a_us(lo[w], int32_t(q8w[w]),     idot);
                        idot = ie::dp4a_us(hi[w], int32_t(q8w[w + 4]), idot);
                    }
                    acc[t] += d4 * (xb.d * float(idot) - 8.0f * xb.s);
                }
            }
            #pragma unroll
            for (int t = 0; t < T_MAX; ++t) {
                if (uint32_t(t) >= T) break;
                const float r = sycl::reduce_over_group(it.get_sub_group(), acc[t],
                                                        sycl::plus<float>());
                if (lane == 0) y[uint64_t(t) * N + n] = sycl::half(r);
            }
        });
    });
}

// Transposing dequant: Q4_0 W[K,N] (column-packed, column n = K/32 contiguous
// blocks) → fp16 Bt[K,N] row-major (Bt[k*N+n]), the layout gemm_fp16/oneDNN
// consume. One work-item per (column n, 32-block); writes 32 fp16. Lets the
// projections run as a weight-stationary fp16 XMX GEMM (dequant once, O(KN),
// vs the dp4a kernel's per-call 4-bit reads) — the path that feeds oneDNN.
sycl::event dequant_q4_0_to_Bt(sycl::queue& q,
                               const void* W_packed, sycl::half* Bt,
                               uint32_t K, uint32_t N,
                               const std::vector<sycl::event>& deps) {
    const auto* W = static_cast<const block_q4_0*>(W_packed);
    const uint32_t blocks_per_col = K / 32;
    return ie::ps(q, "dequant_q4_0_Bt", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(uint64_t(N) * blocks_per_col),
                       [=](sycl::id<1> id) {
            const uint32_t n = uint32_t(id[0] / blocks_per_col);
            const uint32_t b = uint32_t(id[0] % blocks_per_col);
            const block_q4_0& blk = W[uint64_t(n) * blocks_per_col + b];
            const float d = q40_fp16_to_fp32(blk.d);
            const uint32_t k0 = b * 32;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                const uint8_t qb = blk.qs[i];
                Bt[uint64_t(k0 + i)      * N + n] = sycl::half(d * float(int(qb & 0x0F) - 8));
                Bt[uint64_t(k0 + i + 16) * N + n] = sycl::half(d * float(int(qb >> 4)   - 8));
            }
        });
    });
}

// SoA-stream variant of dequant_q4_0_to_Bt — bit-identical output, reads the
// per-column nibble (soa_qs) + fp16-scale (soa_d) streams from repack_q4_0_to_soa.
// Used by the SoA-only gemma4 prefill (no AoS copy resident) → onednn GEMM.
sycl::event dequant_q4_0_soa_to_Bt(sycl::queue& q,
                                   const uint8_t* soa_qs, const uint16_t* soa_d,
                                   sycl::half* Bt, uint32_t K, uint32_t N,
                                   const std::vector<sycl::event>& deps) {
    const uint32_t blocks_per_col = K / 32;
    return ie::ps(q, "dequant_q4_0_soa_Bt", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(uint64_t(N) * blocks_per_col),
                       [=](sycl::id<1> id) {
            const uint32_t n = uint32_t(id[0] / blocks_per_col);
            const uint32_t b = uint32_t(id[0] % blocks_per_col);
            const float d = q40_fp16_to_fp32(soa_d[uint64_t(n) * blocks_per_col + b]);
            const uint8_t* qs = soa_qs + uint64_t(n) * (uint64_t(K) / 2) + uint64_t(b) * 16;
            const uint32_t k0 = b * 32;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                const uint8_t qb = qs[i];
                Bt[uint64_t(k0 + i)      * N + n] = sycl::half(d * float(int(qb & 0x0F) - 8));
                Bt[uint64_t(k0 + i + 16) * N + n] = sycl::half(d * float(int(qb >> 4)   - 8));
            }
        });
    });
}

// int-dot W4A8 batched GEMM with split-K. y[M,N] = q8(x)[M,K] @ Q4_0 W[K,N].
// Mirrors moe_prefill_proj_q4_0_q8's dp4a fold (one fp16 scale per 32-block,
// symmetric w=d*(nib-8) → d4*(d8*idot - 8*(s0+s1))) but tiles the K dimension
// in chunks of TILE_BLK blocks so SLM staging + per-lane register blocks stay
// bounded for ANY K. One WG per (M-tile, N-chunk of N_PER_WG columns); each
// subgroup owns one output column; lane L owns K-blocks L,L+16,... of its tile.
sycl::event gemm_q4_0_q8(sycl::queue& q,
                         const void* xq8_packed, const void* W_q4_0,
                         sycl::half* y,
                         uint32_t M, uint32_t K, uint32_t N,
                         const std::vector<sycl::event>& deps) {
    if (M == 0) return {};
    constexpr int SG_SIZE  = 16;
    constexpr int N_PER_WG = 32;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;        // 512
    constexpr int M_TILE   = 8;
    constexpr int TILE_BLK = 96;                        // 96 blocks = 3072 K-elems
    constexpr int MAX_BPL  = TILE_BLK / SG_SIZE;        // 6 weight blocks / lane / tile
    constexpr uint32_t BW  = sizeof(block_q8_1s) / 4;   // 12 words / q8 block

    const uint32_t blocks_per_col = K / 32;             // == blocks per row
    const uint32_t n_chunks = (N + N_PER_WG - 1) / N_PER_WG;
    const uint32_t m_tiles  = (M + M_TILE - 1) / M_TILE;
    const auto* X8 = static_cast<const block_q8_1s*>(xq8_packed);
    const auto* WB = static_cast<const uint8_t*>(W_q4_0);

    return ie::ps(q, "gemm_q4_0_q8", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<uint32_t, 1> sx(uint64_t(M_TILE) * TILE_BLK * BW, h);

        h.parallel_for(sycl::nd_range<2>({uint64_t(m_tiles), uint64_t(n_chunks) * WG_ITEMS},
                                          {1, WG_ITEMS}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t m0   = uint32_t(it.get_group(0)) * M_TILE;
            const uint32_t Mc   = sycl::min(uint32_t(M_TILE), M - m0);
            const uint32_t nc   = uint32_t(it.get_group(1));
            const uint32_t lid  = uint32_t(it.get_local_id(1));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t nn    = nc * N_PER_WG + sg_id;   // output column

            float acc[M_TILE];
            #pragma unroll
            for (int mm = 0; mm < M_TILE; ++mm) acc[mm] = 0.f;

            for (uint32_t kb0 = 0; kb0 < blocks_per_col; kb0 += TILE_BLK) {
                const uint32_t nb = sycl::min(uint32_t(TILE_BLK), blocks_per_col - kb0);

                // This lane's weight blocks for the tile (≤ MAX_BPL).
                uint32_t wlo[MAX_BPL][4], whi[MAX_BPL][4];
                float    d4[MAX_BPL];
                uint32_t my_lb[MAX_BPL];
                uint32_t my_nblk = 0;
                if (nn < N) {
                    for (uint32_t lb = lane; lb < nb; lb += SG_SIZE) {
                        const uint64_t bi = uint64_t(nn) * blocks_per_col + kb0 + lb;
                        const uint8_t* bp = WB + bi * 18;        // block_q4_0 = 18 bytes
                        const uint16_t db = uint16_t(bp[0]) | (uint16_t(bp[1]) << 8);
                        d4[my_nblk] = q40_fp16_to_fp32(db);
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
                        my_lb[my_nblk] = lb;
                        ++my_nblk;
                    }
                }

                // Stage Mc rows × nb blocks of q8 activation for this tile.
                sycl::group_barrier(it.get_group());
                for (uint32_t i = lid; i < Mc * nb * BW; i += WG_ITEMS) {
                    const uint32_t row = i / (nb * BW);
                    const uint32_t off = i % (nb * BW);       // word within row-tile
                    const uint32_t* src = reinterpret_cast<const uint32_t*>(
                        X8 + (uint64_t(m0 + row) * blocks_per_col + kb0));
                    sx[i] = src[off];
                }
                sycl::group_barrier(it.get_group());

                if (nn < N) {
                    const uint32_t* base =
                        sx.get_multi_ptr<sycl::access::decorated::no>().get();
                    #pragma unroll
                    for (int mm = 0; mm < M_TILE; ++mm) {
                        if (uint32_t(mm) < Mc) {
                            for (uint32_t s = 0; s < my_nblk; ++s) {
                                const uint32_t* blkp =
                                    base + (uint32_t(mm) * nb + my_lb[s]) * BW;
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
                }
            }

            if (nn >= N) return;
            #pragma unroll
            for (int mm = 0; mm < M_TILE; ++mm) {
                if (uint32_t(mm) < Mc) {
                    const float r = sycl::reduce_over_group(it.get_sub_group(),
                                                            acc[mm], sycl::plus<float>());
                    if (lane == 0) y[(uint64_t(m0 + mm)) * N + nn] = sycl::half(r);
                }
            }
        });
    });
}

}  // namespace ie
