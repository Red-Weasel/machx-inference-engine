// src/ops/gemv_kq_c32.cpp — native Q6_K / Q5_K decode GEMV for the 2-card 27B split (P4 B21, 2026-09-28).
//
// WHY: the split's non-Q8_0 weights ran the AoS dense fallbacks (Q6_K: W6A16 gemv_q6_K / gemm_q6_K at M=1, the 140 GB/s
// cliff; Q5_K: dequanted to F16 at load, 2.9x the bytes), so a Q6_K or Q5_K_M 27B could not turn fewer bytes into speed.
// These kernels read the K-quant at its own size (6.56 / 5.63 bpw) with the Q8_0-SoA kernel's proven lane map.
//
// "c32" layout: a load-time repack of each weight COLUMN (output n; the Q8_0-SoA split's column-major discipline) into
// per-32-element chunks c (= one Q8_1 activation block), so one lane owns one chunk and the 16 lanes of a subgroup read 16
// consecutive chunks of one column -- coalesced wide loads, every lane active at every K (K % 256 == 0).
//   Both types, lo plane [n*(K/2) + 16c + j], j = 0..15: low nibble = bits 0..3 of element 32c+j, high nibble = bits 0..3
//     of element 32c+16+j. One uint4 per chunk; word i & 0x0F0F0F0F = elements 4i..4i+3, (word i >> 4) & 0x0F0F0F0F =
//     elements 16+4i..16+4i+3, byte-aligned for dp4a with the activation's word i / i+4.
//   Q6_K hi plane [n*(K/4) + 8c + 4h + b] (uint2 per chunk): bits 2s..2s+1 of byte b of half h = bits 4..5 of element
//     16h + 4s + b.  sc [n*(K/16) + 2c + g] int8 per 16 (ggml's scales, natural order). d [n*(K/256) + c/8] fp16.
//   Q5_K hi plane [n*(K/8) + 4c + b] (uint32 per chunk): bit s of byte b = bit 4 of element 4s + b.
//     sm [n*(K/32) + c] uint16 = 6-bit scale | 6-bit min << 8 (get_scale_min_k4 unpacked). dm [n*(K/256) + c/8] uint32 =
//     fp16 d | fp16 dmin << 16.
// The 6-/5-bit quants and every scale are the GGUF's own bits (a pure layout move; tests/unit/gemv_kq_c32_test checks the
// host dequant of the repack against ggml's dequant bit for bit).
//
// Math per lane chunk and activation row t (x = d8 * q8, block_q8_1x from quantize_q8_1):
//   Q6_K: it = sc0 * (Σ q6u·q8 − 32 Σ q8)[elements 0..15] + sc1 * (...)[16..31]  (exact int32), acc_t += d6 * d8 * it
//   Q5_K: idot = Σ q5·q8, acc_t += (d·sc) * d8 * idot − (dmin·m) * s8           (s8 = d8 Σ q8 from the activation block)
// then a subgroup sum per row. ONE kernel template serves every row count: T = 1 (decode) is the T_MAX = 1 instance of the
// code the rows buckets (T 2..16, the request lanes / spec verify) run, and a row's operands, per-lane chunk order and
// float expression do not depend on T -- so row t of a T-row call equals the T = 1 call on that row, bit for bit (the
// same-reduction-order rule; checked by the test on the device and by ie-q27-lanes-equiv on the model). This TU is built
// with -fp-model=precise (src/CMakeLists.txt) so the compiler may not re-associate the per-row float expression, and the
// fold is written as explicit fma: a contractable a*b+c (llvm.fmuladd) is fused or not at the backend's choice, which
// differed between the T_MAX = 1 and 16 instances on the CPU device (the first test run caught it).
#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"

#include <sycl/sycl.hpp>
#include "ie/dp4a.hpp"
#include "ie/kernel_profiler.hpp"

#include <algorithm>
#include <cstring>
#include <thread>
#include <vector>

namespace ie {

namespace {

inline float c32_h2f(uint16_t h) { return float(sycl::bit_cast<sycl::half>(h)); }

// ggml get_scale_min_k4: the 6-bit (scale, min) of sub-block j (0..7) from the 12-byte K-quant scales table.
inline void c32_scale_min(int j, const uint8_t* q, uint8_t& sc, uint8_t& m) {
    if (j < 4) { sc = q[j] & 63; m = q[j + 4] & 63; }
    else       { sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

// Split [0, N) columns over host threads (the repack is per column and the 27B's big matrices are tens of MB each).
template <class F> void c32_for_columns(uint32_t N, F&& f) {
    const uint32_t hw = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    const uint32_t nt = N >= 256 ? hw : 1u;
    std::vector<std::thread> th;
    for (uint32_t i = 0; i < nt; ++i) {
        const uint32_t a = uint32_t(uint64_t(N) * i / nt), b = uint32_t(uint64_t(N) * (i + 1) / nt);
        th.emplace_back([&f, a, b] { for (uint32_t n = a; n < b; ++n) f(n); });
    }
    for (auto& t : th) t.join();
}

// The lo byte pair layout shared by both types: q[32] (values 0..63 / 0..31) of one chunk -> 16 lo bytes.
inline void c32_put_lo(const uint8_t* q, uint8_t* lo) {
    for (int j = 0; j < 16; ++j) lo[j] = uint8_t((q[j] & 0xF) | ((q[16 + j] & 0xF) << 4));
}

}  // namespace

void repack_q6_K_to_c32(const void* W_blocks, uint32_t K, uint32_t N,
                        uint8_t* lo, uint8_t* hi, int8_t* sc, uint16_t* d) {
    const uint32_t bpc = K / 256;
    const auto* blocks = static_cast<const block_q6_K*>(W_blocks);
    c32_for_columns(N, [&](uint32_t n) {
        const block_q6_K* col = blocks + uint64_t(n) * bpc;
        uint8_t* lo_c = lo + uint64_t(n) * (K / 2);
        uint8_t* hi_c = hi + uint64_t(n) * (K / 4);
        int8_t*  sc_c = sc + uint64_t(n) * (K / 16);
        uint16_t* d_c = d  + uint64_t(n) * (K / 256);
        uint8_t q[256];
        for (uint32_t b = 0; b < bpc; ++b) {
            const block_q6_K& blk = col[b];
            std::memcpy(&d_c[b], &blk.d, 2);
            for (int i = 0; i < 16; ++i) sc_c[b * 16 + i] = blk.scales[i];
            // ggml dequantize_row_q6_K's de-interleave: the unsigned 6-bit quant (value = q - 32) in natural order
            const uint8_t* ql = blk.ql;
            const uint8_t* qh = blk.qh;
            for (int half = 0; half < 256; half += 128) {
                for (int l = 0; l < 32; ++l) {
                    q[half + l +  0] = uint8_t((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4));
                    q[half + l + 32] = uint8_t((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4));
                    q[half + l + 64] = uint8_t((ql[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4));
                    q[half + l + 96] = uint8_t((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4));
                }
                ql += 64; qh += 32;
            }
            for (int ci = 0; ci < 8; ++ci) {
                const uint32_t c = b * 8 + ci;
                const uint8_t* qc = q + ci * 32;
                c32_put_lo(qc, lo_c + uint64_t(c) * 16);
                uint8_t* hc = hi_c + uint64_t(c) * 8;
                for (int h = 0; h < 2; ++h)
                    for (int bb = 0; bb < 4; ++bb) {
                        uint8_t v = 0;
                        for (int s = 0; s < 4; ++s) v = uint8_t(v | (((qc[16 * h + 4 * s + bb] >> 4) & 3) << (2 * s)));
                        hc[4 * h + bb] = v;
                    }
            }
        }
    });
}

void repack_q5_K_to_c32(const void* W_blocks, uint32_t K, uint32_t N,
                        uint8_t* lo, uint32_t* hi, uint16_t* sm, uint32_t* dm) {
    const uint32_t bpc = K / 256;
    const auto* blocks = static_cast<const block_q5_K*>(W_blocks);
    c32_for_columns(N, [&](uint32_t n) {
        const block_q5_K* col = blocks + uint64_t(n) * bpc;
        uint8_t*  lo_c = lo + uint64_t(n) * (K / 2);
        uint32_t* hi_c = hi + uint64_t(n) * (K / 32);
        uint16_t* sm_c = sm + uint64_t(n) * (K / 32);
        uint32_t* dm_c = dm + uint64_t(n) * (K / 256);
        uint8_t q[256];
        for (uint32_t b = 0; b < bpc; ++b) {
            const block_q5_K& blk = col[b];
            uint16_t dd, dmin;
            std::memcpy(&dd, &blk.d, 2); std::memcpy(&dmin, &blk.dmin, 2);
            dm_c[b] = uint32_t(dd) | (uint32_t(dmin) << 16);
            // ggml dequantize_row_q5_K's order: 64-element groups j, low nibbles then high nibbles of ql[32j..32j+31],
            // the fifth bit from qh[l] bit 2j (low) / 2j+1 (high)
            for (int j = 0; j < 4; ++j)
                for (int l = 0; l < 32; ++l) {
                    const uint8_t v = blk.qs[32 * j + l];
                    q[64 * j + l]      = uint8_t((v & 0xF) | (((blk.qh[l] >> (2 * j))     & 1) << 4));
                    q[64 * j + 32 + l] = uint8_t((v >> 4)  | (((blk.qh[l] >> (2 * j + 1)) & 1) << 4));
                }
            for (int ci = 0; ci < 8; ++ci) {
                const uint32_t c = b * 8 + ci;
                const uint8_t* qc = q + ci * 32;
                c32_put_lo(qc, lo_c + uint64_t(c) * 16);
                uint32_t h = 0;
                for (int bb = 0; bb < 4; ++bb)
                    for (int s = 0; s < 8; ++s) h |= uint32_t((qc[4 * s + bb] >> 4) & 1) << (8 * bb + s);
                hi_c[c] = h;
                uint8_t s6, m6;
                c32_scale_min(ci, blk.scales, s6, m6);
                sm_c[c] = uint16_t(s6 | (uint16_t(m6) << 8));
            }
        }
    });
}

namespace {

constexpr int kC32Sg = 16;        // lanes per column

// One lane's decoded chunk c of weight column n (kind 6: lo uint4 + hi uint2 + sc int8 x2 + d fp16/256; kind 5: lo uint4 +
// hi uint32 + sm uint16 + dm uint32/256). Shared by the dense GEMV and the MoE kernels below, so every path folds a chunk
// with the SAME operations.
template <int KIND> struct C32Chunk {
    uint32_t P[8];            // the 32 quants as 8 dp4a words (bytes 0..63, so signed dp4a = the exact unsigned product)
    float    f0, f1;          // Q6: d6 (f1 unused). Q5: d*sc, dmin*m.
    int32_t  sc0, sc1;        // Q6 per-16 scales

    void load(const uint8_t* lo, const void* hi, const void* sc, const void* d, uint64_t n, uint32_t K, uint32_t c) {
        const sycl::uint4 L = reinterpret_cast<const sycl::uint4*>(lo + n * (K / 2))[c];
        const uint32_t lw[4] = {L.x(), L.y(), L.z(), L.w()};
        if constexpr (KIND == 6) {
            const sycl::uint2 H = reinterpret_cast<const sycl::uint2*>(static_cast<const uint8_t*>(hi) + n * (K / 4))[c];
            const uint16_t s2 = reinterpret_cast<const uint16_t*>(static_cast<const int8_t*>(sc) + n * (K / 16))[c];
            sc0 = int32_t(int8_t(s2 & 0xFF));
            sc1 = int32_t(int8_t(s2 >> 8));
            f0 = c32_h2f(static_cast<const uint16_t*>(d)[n * (K / 256) + c / 8]);
            f1 = 0.f;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                P[i]     = (lw[i] & 0x0F0F0F0Fu)        | (((H.x() >> (2 * i)) & 0x03030303u) << 4);
                P[i + 4] = ((lw[i] >> 4) & 0x0F0F0F0Fu) | (((H.y() >> (2 * i)) & 0x03030303u) << 4);
            }
        } else {
            const uint32_t H  = static_cast<const uint32_t*>(hi)[n * (K / 32) + c];
            const uint16_t s2 = static_cast<const uint16_t*>(sc)[n * (K / 32) + c];
            const uint32_t dmw = static_cast<const uint32_t*>(d)[n * (K / 256) + c / 8];
            sc0 = sc1 = 0;
            f0 = c32_h2f(uint16_t(dmw & 0xFFFF)) * float(s2 & 0xFF);
            f1 = c32_h2f(uint16_t(dmw >> 16)) * float(s2 >> 8);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                P[i]     = (lw[i] & 0x0F0F0F0Fu)        | (((H >> i) & 0x01010101u) << 4);
                P[i + 4] = ((lw[i] >> 4) & 0x0F0F0F0Fu) | (((H >> (i + 4)) & 0x01010101u) << 4);
            }
        }
    }
    // acc + this chunk . activation block (xw = its 8 qs words, d8 / s8 its scale and d8*sum). Explicit fma: a contractable
    // a*b+c is fused or not at the backend's choice per instance.
    float fold(const uint32_t* xw, float d8, float s8, float acc) const {
        if constexpr (KIND == 6) {
            int32_t i0 = 0, s0 = 0, i1 = 0, s1 = 0;
            #pragma unroll
            for (int w = 0; w < 4; ++w) {
                i0 = ie::dp4a_ss(int32_t(P[w]), int32_t(xw[w]), i0);
                s0 = ie::dp4a_ss(0x01010101, int32_t(xw[w]), s0);
                i1 = ie::dp4a_ss(int32_t(P[w + 4]), int32_t(xw[w + 4]), i1);
                s1 = ie::dp4a_ss(0x01010101, int32_t(xw[w + 4]), s1);
            }
            const int32_t iv = sc0 * (i0 - 32 * s0) + sc1 * (i1 - 32 * s1);
            return sycl::fma(f0 * d8, float(iv), acc);
        } else {
            // two independent dp4a chains (an integer sum: exact in any order), not one serial chain of 8
            int32_t id0 = 0, id1 = 0;
            #pragma unroll
            for (int w = 0; w < 4; ++w) {
                id0 = ie::dp4a_ss(int32_t(P[w]), int32_t(xw[w]), id0);
                id1 = ie::dp4a_ss(int32_t(P[w + 4]), int32_t(xw[w + 4]), id1);
            }
            return sycl::fma(-f1, s8, sycl::fma(f0 * d8, float(id0 + id1), acc));
        }
    }
};

// The activation block's 8 qs words.
inline void c32_xwords(const block_q8_1x& xb, uint32_t* xw) {
    #pragma unroll
    for (int w = 0; w < 8; ++w) xw[w] = reinterpret_cast<const uint32_t*>(xb.qs)[w];
}

// CPS columns per subgroup: a lane dots chunk c of CPS adjacent columns against ONE load of the activation block (the
// activation is re-read from L2 per column otherwise, ~1.8x the weight bytes at K = 5120), and keeps CPS columns' weight
// loads in flight. A column's chunks still fold in ascending c on its lane, so CPS does not change a result.
// The activation is read from L2 (block_q8_1x per chunk): staging it in SLM measured slower here (the barrier; B21 bench).
template <int KIND, int T_MAX, int CPS, int COLS = 16>
sycl::event c32_gemv_impl(sycl::queue& q, const void* x_q8, const uint8_t* lo, const void* hi, const void* sc,
                          const void* d, sycl::half* y, uint32_t K, uint32_t N, uint32_t T,
                          const std::vector<sycl::event>& deps) {
    constexpr int WG = kC32Sg * COLS;
    const auto* X8 = static_cast<const block_q8_1x*>(x_q8);
    const uint32_t nb = K / 32;                                   // chunks per column = q8 blocks per row
    const uint32_t n_wgs = (N + COLS * CPS - 1) / (COLS * CPS);
    return ie::ps(q, KIND == 6 ? "gemv_q6k_c32" : "gemv_q5k_c32", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * WG, WG),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kC32Sg)]] {
            const uint32_t lid  = uint32_t(it.get_local_id(0));
            const uint32_t lane = lid % kC32Sg;
            const uint32_t n0   = (uint32_t(it.get_group(0)) * COLS + lid / kC32Sg) * CPS;
            if (n0 >= N) return;
            uint32_t col[CPS];            // an out-of-range column reads column n0 (its result is not stored)
            #pragma unroll
            for (int j = 0; j < CPS; ++j) col[j] = n0 + j < N ? n0 + j : n0;

            float acc[CPS][T_MAX];
            #pragma unroll
            for (int j = 0; j < CPS; ++j)
                #pragma unroll
                for (int t = 0; t < T_MAX; ++t) acc[j][t] = 0.f;

            for (uint32_t c = lane; c < nb; c += kC32Sg) {
                C32Chunk<KIND> wc[CPS];
                #pragma unroll
                for (int j = 0; j < CPS; ++j) wc[j].load(lo, hi, sc, d, col[j], K, c);
                #pragma unroll
                for (int t = 0; t < T_MAX; ++t) {
                    if (uint32_t(t) >= T) break;
                    const block_q8_1x& xb = X8[uint64_t(t) * nb + c];
                    uint32_t xw[8];
                    c32_xwords(xb, xw);
                    #pragma unroll
                    for (int j = 0; j < CPS; ++j) acc[j][t] = wc[j].fold(xw, xb.d, xb.s, acc[j][t]);
                }
            }
            #pragma unroll
            for (int j = 0; j < CPS; ++j) {
                #pragma unroll
                for (int t = 0; t < T_MAX; ++t) {
                    if (uint32_t(t) >= T) break;
                    const float r = sycl::reduce_over_group(it.get_sub_group(), acc[j][t], sycl::plus<float>());
                    if (lane == 0 && n0 + j < N) y[uint64_t(t) * N + n0 + j] = sycl::half(r);
                }
            }
        });
    });
}

template <int KIND>
sycl::event c32_gemv(sycl::queue& q, const void* x_q8, const uint8_t* lo, const void* hi, const void* sc, const void* d,
                     sycl::half* y, uint32_t K, uint32_t N, uint32_t T, const std::vector<sycl::event>& deps) {
    if (T == 0) return {};
    if (T > 16) {   // rows above the largest bucket loop in 16-row slices (never overrun acc[])
        sycl::event e;
        const uint64_t row = uint64_t(K / 32) * sizeof(block_q8_1x);
        for (uint32_t t0 = 0; t0 < T; t0 += 16)
            e = c32_gemv<KIND>(q, static_cast<const uint8_t*>(x_q8) + uint64_t(t0) * row, lo, hi, sc, d,
                               y + uint64_t(t0) * N, K, N, std::min(16u, T - t0), t0 == 0 ? deps : std::vector<sycl::event>{});
        return e;
    }
    // columns a subgroup (measured, gemv_kq_c32_test --bench on the 27B shapes): T = 1 two at K <= 8192, four at the 27B's
    // K = 17408; rows T <= 8 two; T 9..16 one (the 16 rows' accumulators fill the registers).
    const int cps = T == 1 ? (K <= 8192 ? 2 : 4) : T <= 8 ? 2 : 1;
#define IE_C32_GO(TM, CP) return c32_gemv_impl<KIND, TM, CP>(q, x_q8, lo, hi, sc, d, y, K, N, T, deps)
    if (T == 1) { if (cps == 4) IE_C32_GO(1, 4); IE_C32_GO(1, 2); }
    if (T == 2) IE_C32_GO(2, 2);
    if (T == 3) IE_C32_GO(3, 2);
    if (T <= 4) IE_C32_GO(4, 2);
    if (T <= 8) IE_C32_GO(8, 2);
    IE_C32_GO(16, 1);
#undef IE_C32_GO
}

// Prefill: c32 streams -> fp16 Bt[K, N] (row k, column n; the layout gemv_q_T's F16 branch / oneDNN take). One work-item
// per (super-block b, column n) with n the fastest local dimension, so each of its 256 stores is one element of a 64-wide
// contiguous Bt row segment (dequant_q6_soa_to_Bt_v2's geometry). Values as ggml: Q6 d*sc*(q-32), Q5 (d*sc)*q - dmin*m.
template <int KIND>
sycl::event c32_dequant(sycl::queue& q, const uint8_t* lo, const void* hi, const void* sc, const void* d,
                        sycl::half* Bt, uint32_t K, uint32_t N, const std::vector<sycl::event>& deps) {
    constexpr uint32_t WG_N = 64;
    const uint32_t bpc = K / 256;
    const uint32_t n_pad = ((N + WG_N - 1) / WG_N) * WG_N;
    return ie::ps(q, KIND == 6 ? "dequant_q6k_c32" : "dequant_q5k_c32", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<2>({bpc, n_pad}, {1, WG_N}), [=](sycl::nd_item<2> it) {
            const uint32_t b = uint32_t(it.get_group(0));
            const uint32_t n = uint32_t(it.get_global_id(1));
            if (n >= N) return;
            const auto* lo_c = reinterpret_cast<const sycl::uint4*>(lo + uint64_t(n) * (K / 2));
            sycl::half* dst = Bt + uint64_t(b) * 256u * N + n;
            for (uint32_t ci = 0; ci < 8; ++ci) {
                const uint32_t c = b * 8 + ci;
                const sycl::uint4 L = lo_c[c];
                const uint32_t lw[4] = {L.x(), L.y(), L.z(), L.w()};
                sycl::half* dc = dst + uint64_t(ci) * 32u * N;
                if constexpr (KIND == 6) {
                    const sycl::uint2 H = reinterpret_cast<const sycl::uint2*>(
                        static_cast<const uint8_t*>(hi) + uint64_t(n) * (K / 4))[c];
                    const int8_t* s = static_cast<const int8_t*>(sc) + uint64_t(n) * (K / 16) + 2u * c;
                    const float d6 = c32_h2f(static_cast<const uint16_t*>(d)[uint64_t(n) * (K / 256) + b]);
                    #pragma unroll
                    for (int e = 0; e < 32; ++e) {
                        const int hh = e >> 4, r = e & 15, i = r >> 2, bb = r & 3;
                        const uint32_t nib = (lw[i] >> (8 * bb + 4 * hh)) & 0xFu;
                        const uint32_t h2 = ((hh ? H.y() : H.x()) >> (8 * bb + 2 * i)) & 3u;
                        const int qv = int(nib | (h2 << 4)) - 32;
                        dc[uint64_t(e) * N] = sycl::half(d6 * float(s[hh]) * float(qv));
                    }
                } else {
                    const uint32_t H  = static_cast<const uint32_t*>(hi)[uint64_t(n) * (K / 32) + c];
                    const uint16_t s2 = static_cast<const uint16_t*>(sc)[uint64_t(n) * (K / 32) + c];
                    const uint32_t dmw = static_cast<const uint32_t*>(d)[uint64_t(n) * (K / 256) + b];
                    const float d1 = c32_h2f(uint16_t(dmw & 0xFFFF)) * float(s2 & 0xFF);
                    const float m1 = c32_h2f(uint16_t(dmw >> 16)) * float(s2 >> 8);
                    #pragma unroll
                    for (int e = 0; e < 32; ++e) {
                        const int hh = e >> 4, r = e & 15, i = r >> 2, bb = r & 3;
                        const uint32_t nib = (lw[i] >> (8 * bb + 4 * hh)) & 0xFu;
                        const uint32_t h1 = (H >> (8 * bb + (e >> 2))) & 1u;
                        dc[uint64_t(e) * N] = sycl::half(d1 * float(nib | (h1 << 4)) - m1);
                    }
                }
            }
        });
    });
}

}  // namespace

sycl::event gemv_q6k_c32_q8(sycl::queue& q, const void* x_q8, const uint8_t* lo, const uint8_t* hi, const int8_t* sc,
                            const uint16_t* d, sycl::half* y, uint32_t K, uint32_t N, uint32_t T,
                            const std::vector<sycl::event>& deps) {
    return c32_gemv<6>(q, x_q8, lo, hi, sc, d, y, K, N, T, deps);
}

sycl::event gemv_q5k_c32_q8(sycl::queue& q, const void* x_q8, const uint8_t* lo, const uint32_t* hi, const uint16_t* sm,
                            const uint32_t* dm, sycl::half* y, uint32_t K, uint32_t N, uint32_t T,
                            const std::vector<sycl::event>& deps) {
    return c32_gemv<5>(q, x_q8, lo, hi, sm, dm, y, K, N, T, deps);
}

sycl::event dequant_q6k_c32_to_Bt(sycl::queue& q, const uint8_t* lo, const uint8_t* hi, const int8_t* sc,
                                  const uint16_t* d, sycl::half* Bt, uint32_t K, uint32_t N,
                                  const std::vector<sycl::event>& deps) {
    return c32_dequant<6>(q, lo, hi, sc, d, Bt, K, N, deps);
}

sycl::event dequant_q5k_c32_to_Bt(sycl::queue& q, const uint8_t* lo, const uint32_t* hi, const uint16_t* sm,
                                  const uint32_t* dm, sycl::half* Bt, uint32_t K, uint32_t N,
                                  const std::vector<sycl::event>& deps) {
    return c32_dequant<5>(q, lo, hi, sm, dm, Bt, K, N, deps);
}

// ============================================================================================================================
// P4 B21: the qwen35moe (35B-A3B crown split) expert kernels over c32 banks. An expert tensor [K, N, E] is expert-major
// contiguous, so its c32 repack over E*N columns puts expert e's column n at bank column e*N + n. Structure mirrors
// moe_q8.cpp (one subgroup per output column, the expert read per token-slot from topk_idx, gate+up fused with silu, down
// folding the routing weight) and every chunk goes through C32Chunk::fold -- the dense GEMV's arithmetic. The per-slot
// kernels serve T = 1 AND the B14 rows (T slots of the SAME kernel; a slot's result does not depend on T).
// ============================================================================================================================
namespace {

template <int KIND>
sycl::event c32_moe_gate_up(sycl::queue& q, const void* x_q8, const C32Bank& g, const C32Bank& u, const int32_t* topk_idx,
                            sycl::half* h_out, uint32_t T, uint32_t Kt, uint32_t H, uint32_t EFF) {
    constexpr int COLS = 16, WG = kC32Sg * COLS;
    const uint32_t nb = H / 32, col_wgs = (EFF + COLS - 1) / COLS;
    const uint64_t n_wgs = uint64_t(T) * Kt * col_wgs;
    const auto* X8 = static_cast<const block_q8_1x*>(x_q8);
    return ie::ps(q, KIND == 6 ? "moe_gate_up_q6k_c32" : "moe_gate_up_q5k_c32", [&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(n_wgs * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kC32Sg)]] {
            const uint32_t lid = uint32_t(it.get_local_id(0)), lane = lid % kC32Sg;
            const uint32_t tk = uint32_t(it.get_group(0) / col_wgs), cw = uint32_t(it.get_group(0) % col_wgs);
            const uint32_t n = cw * COLS + lid / kC32Sg;
            if (n >= EFF) return;
            const uint64_t col = uint64_t(uint32_t(topk_idx[tk])) * EFF + n;
            const block_q8_1x* xr = X8 + uint64_t(tk / Kt) * nb;
            float ga = 0.f, ua = 0.f;
            for (uint32_t c = lane; c < nb; c += kC32Sg) {
                C32Chunk<KIND> gc, uc;
                gc.load(g.lo, g.hi, g.sc, g.d, col, H, c);
                uc.load(u.lo, u.hi, u.sc, u.d, col, H, c);
                uint32_t xw[8];
                c32_xwords(xr[c], xw);
                ga = gc.fold(xw, xr[c].d, xr[c].s, ga);
                ua = uc.fold(xw, xr[c].d, xr[c].s, ua);
            }
            ga = sycl::reduce_over_group(it.get_sub_group(), ga, sycl::plus<float>());
            ua = sycl::reduce_over_group(it.get_sub_group(), ua, sycl::plus<float>());
            if (lane == 0) h_out[uint64_t(tk) * EFF + n] = sycl::half(ga / (1.f + sycl::exp(-ga)) * ua);
        });
    });
}

template <int KIND>
sycl::event c32_moe_down(sycl::queue& q, const void* h_q8, const C32Bank& dw, const int32_t* topk_idx,
                         const sycl::half* topk_w, sycl::half* y_packed, uint32_t T, uint32_t Kt, uint32_t EFF, uint32_t H) {
    constexpr int COLS = 16, WG = kC32Sg * COLS;
    const uint32_t nb = EFF / 32, col_wgs = (H + COLS - 1) / COLS;
    const uint64_t n_wgs = uint64_t(T) * Kt * col_wgs;
    const auto* X8 = static_cast<const block_q8_1x*>(h_q8);
    return ie::ps(q, KIND == 6 ? "moe_down_q6k_c32" : "moe_down_q5k_c32", [&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(n_wgs * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kC32Sg)]] {
            const uint32_t lid = uint32_t(it.get_local_id(0)), lane = lid % kC32Sg;
            const uint32_t tk = uint32_t(it.get_group(0) / col_wgs), cw = uint32_t(it.get_group(0) % col_wgs);
            const uint32_t n = cw * COLS + lid / kC32Sg;
            if (n >= H) return;
            const uint64_t col = uint64_t(uint32_t(topk_idx[tk])) * H + n;
            const block_q8_1x* xr = X8 + uint64_t(tk) * nb;
            float acc = 0.f;
            for (uint32_t c = lane; c < nb; c += kC32Sg) {
                C32Chunk<KIND> wc;
                wc.load(dw.lo, dw.hi, dw.sc, dw.d, col, EFF, c);
                uint32_t xw[8];
                c32_xwords(xr[c], xw);
                acc = wc.fold(xw, xr[c].d, xr[c].s, acc);
            }
            acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
            if (lane == 0) y_packed[uint64_t(tk) * H + n] = sycl::half(float(topk_w[tk]) * acc);
        });
    });
}

// Prefill (expert-batched): the routed rows sorted by expert (expert e owns rows eoff[e]..eoff[e+1]); one work-group per
// (expert, 16 columns) walks its rows in tiles of TILE, so an expert's weights are read about once per chunk instead of
// once per routed row. Outputs by sorted row, like moe_prefill_{gate_up_silu,down}_q8 (moe_prefill_reduce_sum after).
constexpr uint32_t kC32PfTile = 8;

template <int KIND>
sycl::event c32_moe_prefill_gate_up(sycl::queue& q, const void* xp_q8, const C32Bank& g, const C32Bank& u,
                                    const uint32_t* eoff, sycl::half* hp, uint32_t E, uint32_t H, uint32_t EFF) {
    constexpr int COLS = 16, WG = kC32Sg * COLS;
    const uint32_t nb = H / 32, col_wgs = (EFF + COLS - 1) / COLS;
    const auto* X8 = static_cast<const block_q8_1x*>(xp_q8);
    return ie::ps(q, KIND == 6 ? "moe_pfl_gate_up_q6k_c32" : "moe_pfl_gate_up_q5k_c32", [&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(uint64_t(E) * col_wgs * WG, WG),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kC32Sg)]] {
            const uint32_t lid = uint32_t(it.get_local_id(0)), lane = lid % kC32Sg;
            const uint32_t e = uint32_t(it.get_group(0) / col_wgs), cw = uint32_t(it.get_group(0) % col_wgs);
            const uint32_t n = cw * COLS + lid / kC32Sg;
            const uint32_t r_begin = eoff[e], r_end = eoff[e + 1];
            if (n >= EFF || r_begin == r_end) return;
            const uint64_t col = uint64_t(e) * EFF + n;
            for (uint32_t r0 = r_begin; r0 < r_end; r0 += kC32PfTile) {
                const uint32_t M = sycl::min(kC32PfTile, r_end - r0);
                float ga[kC32PfTile], ua[kC32PfTile];
                #pragma unroll
                for (uint32_t m = 0; m < kC32PfTile; ++m) { ga[m] = 0.f; ua[m] = 0.f; }
                for (uint32_t c = lane; c < nb; c += kC32Sg) {
                    C32Chunk<KIND> gc, uc;
                    gc.load(g.lo, g.hi, g.sc, g.d, col, H, c);
                    uc.load(u.lo, u.hi, u.sc, u.d, col, H, c);
                    #pragma unroll
                    for (uint32_t m = 0; m < kC32PfTile; ++m) {
                        if (m >= M) break;
                        const block_q8_1x& xb = X8[uint64_t(r0 + m) * nb + c];
                        uint32_t xw[8];
                        c32_xwords(xb, xw);
                        ga[m] = gc.fold(xw, xb.d, xb.s, ga[m]);
                        ua[m] = uc.fold(xw, xb.d, xb.s, ua[m]);
                    }
                }
                #pragma unroll
                for (uint32_t m = 0; m < kC32PfTile; ++m) {
                    if (m >= M) break;
                    const float gr = sycl::reduce_over_group(it.get_sub_group(), ga[m], sycl::plus<float>());
                    const float ur = sycl::reduce_over_group(it.get_sub_group(), ua[m], sycl::plus<float>());
                    if (lane == 0) hp[uint64_t(r0 + m) * EFF + n] = sycl::half(gr / (1.f + sycl::exp(-gr)) * ur);
                }
            }
        });
    });
}

template <int KIND>
sycl::event c32_moe_prefill_down(sycl::queue& q, const void* hq8, const C32Bank& dw, const uint32_t* eoff,
                                 const sycl::half* sorted_w, sycl::half* out_packed, uint32_t E, uint32_t H, uint32_t EFF) {
    constexpr int COLS = 16, WG = kC32Sg * COLS;
    const uint32_t nb = EFF / 32, col_wgs = (H + COLS - 1) / COLS;
    const auto* X8 = static_cast<const block_q8_1x*>(hq8);
    return ie::ps(q, KIND == 6 ? "moe_pfl_down_q6k_c32" : "moe_pfl_down_q5k_c32", [&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(uint64_t(E) * col_wgs * WG, WG),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kC32Sg)]] {
            const uint32_t lid = uint32_t(it.get_local_id(0)), lane = lid % kC32Sg;
            const uint32_t e = uint32_t(it.get_group(0) / col_wgs), cw = uint32_t(it.get_group(0) % col_wgs);
            const uint32_t n = cw * COLS + lid / kC32Sg;
            const uint32_t r_begin = eoff[e], r_end = eoff[e + 1];
            if (n >= H || r_begin == r_end) return;
            const uint64_t col = uint64_t(e) * H + n;
            for (uint32_t r0 = r_begin; r0 < r_end; r0 += kC32PfTile) {
                const uint32_t M = sycl::min(kC32PfTile, r_end - r0);
                float acc[kC32PfTile];
                #pragma unroll
                for (uint32_t m = 0; m < kC32PfTile; ++m) acc[m] = 0.f;
                for (uint32_t c = lane; c < nb; c += kC32Sg) {
                    C32Chunk<KIND> wc;
                    wc.load(dw.lo, dw.hi, dw.sc, dw.d, col, EFF, c);
                    #pragma unroll
                    for (uint32_t m = 0; m < kC32PfTile; ++m) {
                        if (m >= M) break;
                        const block_q8_1x& xb = X8[uint64_t(r0 + m) * nb + c];
                        uint32_t xw[8];
                        c32_xwords(xb, xw);
                        acc[m] = wc.fold(xw, xb.d, xb.s, acc[m]);
                    }
                }
                #pragma unroll
                for (uint32_t m = 0; m < kC32PfTile; ++m) {
                    if (m >= M) break;
                    const float r = sycl::reduce_over_group(it.get_sub_group(), acc[m], sycl::plus<float>());
                    if (lane == 0) out_packed[uint64_t(r0 + m) * H + n] = sycl::half(float(sorted_w[r0 + m]) * r);
                }
            }
        });
    });
}

}  // namespace

sycl::event moe_gate_up_silu_c32(sycl::queue& q, const void* x_q8, const C32Bank& g, const C32Bank& u,
                                 const int32_t* topk_idx, sycl::half* h_out, uint32_t T, uint32_t Kt, uint32_t H,
                                 uint32_t EFF) {
    return g.kind == 6 ? c32_moe_gate_up<6>(q, x_q8, g, u, topk_idx, h_out, T, Kt, H, EFF)
                       : c32_moe_gate_up<5>(q, x_q8, g, u, topk_idx, h_out, T, Kt, H, EFF);
}
sycl::event moe_down_c32(sycl::queue& q, const void* h_q8, const C32Bank& dw, const int32_t* topk_idx,
                         const sycl::half* topk_w, sycl::half* y_packed, uint32_t T, uint32_t Kt, uint32_t EFF, uint32_t H) {
    return dw.kind == 6 ? c32_moe_down<6>(q, h_q8, dw, topk_idx, topk_w, y_packed, T, Kt, EFF, H)
                        : c32_moe_down<5>(q, h_q8, dw, topk_idx, topk_w, y_packed, T, Kt, EFF, H);
}
sycl::event moe_prefill_gate_up_silu_c32(sycl::queue& q, const void* xp_q8, const C32Bank& g, const C32Bank& u,
                                         const uint32_t* eoff, sycl::half* hp, uint32_t E, uint32_t H, uint32_t EFF) {
    return g.kind == 6 ? c32_moe_prefill_gate_up<6>(q, xp_q8, g, u, eoff, hp, E, H, EFF)
                       : c32_moe_prefill_gate_up<5>(q, xp_q8, g, u, eoff, hp, E, H, EFF);
}
sycl::event moe_prefill_down_c32(sycl::queue& q, const void* hq8, const C32Bank& dw, const uint32_t* eoff,
                                 const sycl::half* sorted_w, sycl::half* out_packed, uint32_t E, uint32_t H, uint32_t EFF) {
    return dw.kind == 6 ? c32_moe_prefill_down<6>(q, hq8, dw, eoff, sorted_w, out_packed, E, H, EFF)
                        : c32_moe_prefill_down<5>(q, hq8, dw, eoff, sorted_w, out_packed, E, H, EFF);
}

}  // namespace ie
