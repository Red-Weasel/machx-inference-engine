// src/ops/gemv_mxfp4.cpp — fused MXFP4 decode GEMV (gpt-oss MoE experts) over a
// load-time SoA repack of the experts.  2026-06-27.
//
// WHY: at decode (T=1) the gpt-oss MoE path dequantized each routed expert's FULL
// weight matrix [K,N] to fp16 (`dequant_mxfp4_to_Bt`, ~16.6 MB written/matrix) and
// ran an M=1 oneDNN fp16 GEMM that re-read it — ~10.8 GB of memory traffic/token vs
// a 1.27 GB packed-weight floor (the weights are only 4.25 bpw).  Profiled at 43.5%
// `dequant_mxfp4_bt` + a slice of `gemv_fp16`.  A fused GEMV reads the weights ONCE.
//
// THE LAYOUT TRAP: the native block_mxfp4 is 17 bytes (1 E8M0 + 16 nibble bytes), so
// the on-disk bank is byte-packed and EVERY cross-lane read is 17-strided / unaligned
// → a first cut over the native layout ran at ~64 GB/s (10% of peak).  Fix (the proven
// gemv_q6_soa/gemv_q4_soa recipe): a load-time SoA repack splits each expert into two
// aligned, coalesced planes —
//   qs_plane[n*(K/2) + b*16 + j]  = block(n,b).qs[j]   (16 nibble bytes/block, 16-aligned)
//   e_plane [n*(K/32) + b]        = block(n,b).e        (1 E8M0 exponent/block)
// — same 4.25 bpw total, no reorder, just de-interleaved e from qs.  16 lanes stride
// whole 32-elem blocks → consecutive lanes read consecutive 16-byte chunks (coalesced).
//
// NUMERICS:
//   gemv_mxfp4_soa_f16 : fp16 activation, BIT-FAITHFUL (value[k]=e8m0_half(e)·LUT[nib],
//     same as the dequant path).  The bit-faithful decode default + correctness ref.
//   gemv_mxfp4_soa_q8  : W4A8 int-dot (dp4a_ss) — x pre-quantized to block_q8_1x; the
//     ALU-efficient B70 decode lever (the proven Q6/Q4-SoA template).  PPL-gated.
//   dequant_mxfp4_soa_to_Bt : SoA → fp16 Bt[K,N] for the prefill (T≥2) oneDNN GEMM.
//
// The FP4 (E2M1) nibble→int magnitude is computed BRANCHLESS (no memory LUT — the
// canonical CUDA byte-permute LUT is software-emulated/slow on Intel Xe).

#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"

#include <sycl/sycl.hpp>
#include "ie/dp4a.hpp"
#include "ie/kernel_profiler.hpp"

namespace ie {

namespace {

// E8M0 *half-scaled* shared exponent → fp32 (bit-exact with ggml_e8m0_to_fp32_half).
inline float mxfp4_e8m0_half(uint8_t e) {
    const uint32_t bits = (e < 2u) ? (0x00200000u << e) : (uint32_t(e - 1u) << 23);
    return sycl::bit_cast<float>(bits);
}

// FP4 (E2M1) nibble → signed integer magnitude×2 ∈ [-12,12], BRANCHLESS (register
// ALU only).  Equivalent to LUT {0,1,2,3,4,6,8,12, 0,-1,…,-12}.  bit3=sign,
// bits[2:1]=exp, bit0=mantissa; mag = exp ? (2+mant)<<(exp-1) : mant.
inline int mxfp4_nibble_int(uint32_t nb) {
    const int exp  = int((nb >> 1) & 3u);
    const int mant = int(nb & 1u);
    const int mag  = exp ? ((2 + mant) << (exp - 1)) : mant;
    return (nb & 8u) ? -mag : mag;
}

// One qs uint32 (4 bytes = 4 low + 4 high nibbles) → two packed int8 dp4a words:
//   lo = decode(b0&F, b1&F, b2&F, b3&F)   (weights 4i..4i+3)
//   hi = decode(b0>>4, b1>>4, b2>>4, b3>>4) (weights 16+4i..16+4i+3)
inline void mxfp4_decode_word(uint32_t w, uint32_t& lo, uint32_t& hi) {
    lo = 0; hi = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t b = (w >> (i * 8)) & 0xFFu;
        const int vlo = mxfp4_nibble_int(b & 0x0Fu);
        const int vhi = mxfp4_nibble_int(b >> 4);
        lo |= (uint32_t(uint8_t(int8_t(vlo))) << (i * 8));
        hi |= (uint32_t(uint8_t(int8_t(vhi))) << (i * 8));
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// W4A8 int-dot decode GEMV over the SoA planes.  Mirrors gemv_q4_soa_q8's
// occupancy (SG_SIZE=16, N_PER_WG=32, Q8_1 activation staged once in SLM, split-K
// by whole 32-elem blocks).  Per block: idot = Σ dp4a_ss(decode(qs), q8); the per-32
// E8M0 scale d and the q8 block scale fold once: acc += d · q8d · idot.
// ---------------------------------------------------------------------------
template <bool USE_SLM>
static sycl::event gemv_mxfp4_soa_q8_impl(sycl::queue& q,
                           const void* x_q8,
                           const uint8_t* qs_plane, const uint8_t* e_plane,
                           sycl::half* y, uint32_t K, uint32_t N,
                           const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE  = 16;
    constexpr int N_PER_WG = 32;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;        // 512

    const auto* X8 = static_cast<const block_q8_1x*>(x_q8);
    const uint32_t bpc = K / kQK_MXFP4;                 // blocks/col = K/32 = q8 blocks
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "gemv_mxfp4", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<uint32_t, 1> q8s(USE_SLM ? bpc * 8 : 1, h);  // 8 u32/block
        sycl::local_accessor<float, 1>    q8d(USE_SLM ? bpc     : 1, h);

        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t wgid  = uint32_t(it.get_group(0));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t n     = wgid * N_PER_WG + sg_id;

            if constexpr (USE_SLM) {
                for (uint32_t i = lid; i < bpc * 8; i += WG_ITEMS)
                    q8s[i] = reinterpret_cast<const uint32_t*>(X8[i / 8].qs)[i % 8];
                for (uint32_t i = lid; i < bpc; i += WG_ITEMS) q8d[i] = X8[i].d;
                sycl::group_barrier(it.get_group());
            }
            if (n >= N) return;

            const uint8_t* qs_col = qs_plane + uint64_t(n) * (uint64_t(K) / 2);
            const uint8_t* e_col  = e_plane  + uint64_t(n) * bpc;

            float acc = 0.f;
            // Split-K: each lane strides whole 32-elem blocks of this column.
            for (uint32_t b = lane; b < bpc; b += SG_SIZE) {
                const float d = mxfp4_e8m0_half(e_col[b]);
                const uint32_t* qw =
                    reinterpret_cast<const uint32_t*>(qs_col + uint64_t(b) * 16);  // 4 u32, aligned
                int32_t idot = 0;
                #pragma unroll
                for (int wi = 0; wi < 4; ++wi) {
                    uint32_t wlo, whi;
                    mxfp4_decode_word(qw[wi], wlo, whi);
                    const int32_t q8_lo = USE_SLM ? int32_t(q8s[b * 8 + wi])
                                                  : int32_t(reinterpret_cast<const uint32_t*>(X8[b].qs)[wi]);
                    const int32_t q8_hi = USE_SLM ? int32_t(q8s[b * 8 + wi + 4])
                                                  : int32_t(reinterpret_cast<const uint32_t*>(X8[b].qs)[wi + 4]);
                    idot = ie::dp4a_ss(int32_t(wlo), q8_lo, idot);   // weights 4wi..4wi+3
                    idot = ie::dp4a_ss(int32_t(whi), q8_hi, idot);   // weights 16+4wi..+3
                }
                const float q8dv = USE_SLM ? q8d[b] : float(X8[b].d);
                acc += d * q8dv * float(idot);
            }

            acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
            if (lane == 0) y[n] = sycl::half(acc);
        });
    });
}

sycl::event gemv_mxfp4_soa_q8(sycl::queue& q,
                              const void* x_q8,
                              const uint8_t* qs_plane, const uint8_t* e_plane,
                              sycl::half* y, uint32_t K, uint32_t N,
                              const std::vector<sycl::event>& deps) {
    static const bool gmem = std::getenv("IE_GPTOSS_MXFP4_GMEM") != nullptr;
    if (gmem) return gemv_mxfp4_soa_q8_impl<false>(q, x_q8, qs_plane, e_plane, y, K, N, deps);
    return gemv_mxfp4_soa_q8_impl<true>(q, x_q8, qs_plane, e_plane, y, K, N, deps);
}

// ---------------------------------------------------------------------------
// FUSED gate+up decode GEMV (W4A8). gpt-oss MoE gate and up share the SAME
// quantized activation x (same K=H) and N=efc, differing only in weight planes —
// so ONE launch computes both: output columns [0,N)=gate, [N,2N)=up, staging x in
// SLM once (shared), and folds the per-column biases. Replaces 2×gemv + 2×add_bias
// (4 host launches) with 1 per expert — gpt-oss MoE decode is host-launch-bound.
// BIT-IDENTICAL to gemv_mxfp4_soa_q8 ×2 then add_bias (same dp4a/fold; bias added
// once post-reduce). One token (1 row) — the decode path only.
// ---------------------------------------------------------------------------
template <bool USE_SLM>
static sycl::event gemv_mxfp4_soa_q8_x2_impl(sycl::queue& q, const void* x_q8,
                           const uint8_t* gqs, const uint8_t* ge,
                           const uint8_t* uqs, const uint8_t* ue,
                           const float* gbias, const float* ubias,
                           sycl::half* gy, sycl::half* uy, uint32_t K, uint32_t N,
                           const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE  = 16;
    constexpr int N_PER_WG = 32;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;

    const auto* X8 = static_cast<const block_q8_1x*>(x_q8);
    const uint32_t bpc = K / kQK_MXFP4;
    const uint32_t N2  = 2u * N;
    const uint32_t n_wgs = (N2 + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "gemv_mxfp4_x2", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<uint32_t, 1> q8s(USE_SLM ? bpc * 8 : 1, h);
        sycl::local_accessor<float, 1>    q8d(USE_SLM ? bpc     : 1, h);

        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t wgid  = uint32_t(it.get_group(0));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t nn    = wgid * N_PER_WG + sg_id;     // 0..2N (gate then up)

            if constexpr (USE_SLM) {
                for (uint32_t i = lid; i < bpc * 8; i += WG_ITEMS)
                    q8s[i] = reinterpret_cast<const uint32_t*>(X8[i / 8].qs)[i % 8];
                for (uint32_t i = lid; i < bpc; i += WG_ITEMS) q8d[i] = X8[i].d;
                sycl::group_barrier(it.get_group());
            }
            if (nn >= N2) return;

            const bool     is_up  = nn >= N;
            const uint32_t n      = is_up ? (nn - N) : nn;
            const uint8_t* qs_col = (is_up ? uqs : gqs) + uint64_t(n) * (uint64_t(K) / 2);
            const uint8_t* e_col  = (is_up ? ue  : ge)  + uint64_t(n) * bpc;

            float acc = 0.f;
            for (uint32_t b = lane; b < bpc; b += SG_SIZE) {
                const float d = mxfp4_e8m0_half(e_col[b]);
                const uint32_t* qw =
                    reinterpret_cast<const uint32_t*>(qs_col + uint64_t(b) * 16);
                int32_t idot = 0;
                #pragma unroll
                for (int wi = 0; wi < 4; ++wi) {
                    uint32_t wlo, whi;
                    mxfp4_decode_word(qw[wi], wlo, whi);
                    const int32_t q8_lo = USE_SLM ? int32_t(q8s[b * 8 + wi])
                                                  : int32_t(reinterpret_cast<const uint32_t*>(X8[b].qs)[wi]);
                    const int32_t q8_hi = USE_SLM ? int32_t(q8s[b * 8 + wi + 4])
                                                  : int32_t(reinterpret_cast<const uint32_t*>(X8[b].qs)[wi + 4]);
                    idot = ie::dp4a_ss(int32_t(wlo), q8_lo, idot);
                    idot = ie::dp4a_ss(int32_t(whi), q8_hi, idot);
                }
                const float q8dv = USE_SLM ? q8d[b] : float(X8[b].d);
                acc += d * q8dv * float(idot);
            }

            acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
            if (lane == 0) {
                const float* bias = is_up ? ubias : gbias;
                if (bias) acc += bias[n];
                (is_up ? uy : gy)[n] = sycl::half(acc);
            }
        });
    });
}

sycl::event gemv_mxfp4_soa_q8_x2(sycl::queue& q, const void* x_q8,
                                 const uint8_t* gate_qs, const uint8_t* gate_e,
                                 const uint8_t* up_qs,   const uint8_t* up_e,
                                 const float* gate_bias, const float* up_bias,
                                 sycl::half* gate_y, sycl::half* up_y,
                                 uint32_t K, uint32_t N,
                                 const std::vector<sycl::event>& deps) {
    static const bool gmem = std::getenv("IE_GPTOSS_MXFP4_GMEM") != nullptr;
    if (gmem) return gemv_mxfp4_soa_q8_x2_impl<false>(q, x_q8, gate_qs, gate_e, up_qs, up_e,
                                                      gate_bias, up_bias, gate_y, up_y, K, N, deps);
    return gemv_mxfp4_soa_q8_x2_impl<true>(q, x_q8, gate_qs, gate_e, up_qs, up_e,
                                           gate_bias, up_bias, gate_y, up_y, K, N, deps);
}

// ---------------------------------------------------------------------------
// Bit-faithful fp16-activation decode GEMV over the SoA planes (same d·LUT values
// as the dequant path).  The safe default + the correctness reference for the
// int-dot variant.  fp16 macs over the branchless-decoded weights; A staged in SLM.
// ---------------------------------------------------------------------------
sycl::event gemv_mxfp4_soa_f16(sycl::queue& q,
                               const sycl::half* A,
                               const uint8_t* qs_plane, const uint8_t* e_plane,
                               sycl::half* y, uint32_t K, uint32_t N,
                               const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE  = 16;
    constexpr int N_PER_WG = 32;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;
    const uint32_t bpc = K / kQK_MXFP4;
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "gemv_mxfp4", [&](sycl::handler& h) {
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

            const uint8_t* qs_col = qs_plane + uint64_t(n) * (uint64_t(K) / 2);
            const uint8_t* e_col  = e_plane  + uint64_t(n) * bpc;

            float acc = 0.f;
            for (uint32_t b = lane; b < bpc; b += SG_SIZE) {
                const float d = mxfp4_e8m0_half(e_col[b]);
                const sycl::vec<uint8_t, 16> qs_v =
                    *reinterpret_cast<const sycl::vec<uint8_t, 16>*>(qs_col + uint64_t(b) * 16);
                const sycl::half* a = &A_slm[b * kQK_MXFP4];
                float sum = 0.f;
                #pragma unroll
                for (int j = 0; j < 16; ++j) {
                    const uint8_t qb = qs_v[j];
                    sum += float(a[j])      * float(mxfp4_nibble_int(qb & 0x0Fu));
                    sum += float(a[j + 16]) * float(mxfp4_nibble_int(qb >> 4));
                }
                acc += d * sum;
            }
            acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
            if (lane == 0) y[n] = sycl::half(acc);
        });
    });
}

// ---------------------------------------------------------------------------
// SoA → fp16 Bt[K,N] (out[k*N+n]) for the prefill (T≥2) per-expert oneDNN GEMM.
// Bit-identical values to dequant_mxfp4_to_Bt (the SoA repack is a pure split).
// One item per (block, n) → 32 rows.  K%32==0.  The dim-1 global is rounded UP to a
// multiple of WG (the `if (n >= N) return` tail guard skips the padding), so N need
// NOT be %64 — required for tensor-parallel where a column slice yields N=efc (e.g.
// EF=2880 → 1440 on 2 cards, 1440%64=32).  Round-up == N when N%64==0, so the
// single-GPU path (N=EF/N=H, both %64) is bit-identical (matches the sibling gemv).
// ---------------------------------------------------------------------------
sycl::event dequant_mxfp4_soa_to_Bt(sycl::queue& q,
                                    const uint8_t* qs_plane, const uint8_t* e_plane,
                                    sycl::half* out, uint32_t K, uint32_t N,
                                    const std::vector<sycl::event>& deps) {
    const uint32_t bpc = K / kQK_MXFP4;
    constexpr uint32_t WG = 64;
    const uint32_t N_pad = ((N + WG - 1) / WG) * WG;
    return ie::ps(q, "dequant_mxfp4_soa_bt", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<2>({bpc, N_pad}, {1, WG}),
                       [=](sycl::nd_item<2> it) {
            const uint32_t b = uint32_t(it.get_group(0));
            const uint32_t n = uint32_t(it.get_global_id(1));
            if (n >= N) return;
            const uint8_t* qs = qs_plane + uint64_t(n) * (uint64_t(K) / 2) + uint64_t(b) * 16;
            const float d = mxfp4_e8m0_half(e_plane[uint64_t(n) * bpc + b]);
            sycl::half* dst = out + uint64_t(b) * kQK_MXFP4 * N + n;
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                const uint8_t qb = qs[j];
                dst[uint64_t(j)      * N] = sycl::half(d * float(mxfp4_nibble_int(qb & 0x0Fu)));
                dst[uint64_t(j + 16) * N] = sycl::half(d * float(mxfp4_nibble_int(qb >> 4)));
            }
        });
    });
}

}  // namespace ie
