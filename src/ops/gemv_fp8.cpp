// src/ops/gemv_fp8.cpp — the T = 1 GEMV over a dense matrix kept in the checkpoint's own FP8
// (E4M3 bytes [N, K], one E8M0 scale per 32 x 32 block [N/32, K/32]) with an fp16 activation
// (docs/deepseek41/43, Phase 17). The Q8_0 SoA f16 leaf's shape (gemv_q8dot.cpp): one 16-lane
// subgroup per output column n, each lane the 32-weight blocks b = lane, lane + 16, ...; two
// 128-bit weight loads per block, 32 fp32 FMAs against x, the block's scale applied once per
// block, fp32 accumulation, a subgroup reduction, an fp32 store (the oneDNN path's output type). The E4M3 / E8M0 decode is
// ds41_e4m3 / ds41_e8m0 (deepseek41_upload.hpp) -- the dense cache's load-time dequant, so the
// weight values are the same ones the fp16 path reads; only the summation order differs.
#include "ie/deepseek41_upload.hpp"
#include "ie/kernel_profiler.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace ie {

// The E4M3 decode, four ways, all bit-identical (the same products in the same order; the unit
// test checks every variant against the scalar):
//   0  the scalar decode of Phase 17 (ds41_e4m3 per weight: three field extractions, a subnormal
//      table, a select, int-to-float -- ~8 ops per weight, measured ALU-bound at ~325 GB/s).
//   1  Phase 17b (docs/deepseek41/45): a 256-entry fp16 table in local memory, filled once per
//      work-group from ds41_e4m3 (every E4M3 value is exact in fp16); one byte extraction and one
//      local load per weight. (The field-arithmetic pair decode tried first measured 2x SLOWER than
//      the scalar and is gone.)
//   2  Phase 18 (docs/deepseek41/46 term 3): the table as fp32 -- no half -> float conversion per weight.
//   3  ... and the block's 32 activations converted to fp32 once, before the FMAs, instead of once
//      per FMA (the same conversions, hoisted).
//   4  Phase 38 (docs/deepseek41/78): NO table. An E4M3 byte's low seven bits shifted left by 7, with the
//      sign at bit 15, IS the fp16 bit pattern of (value / 256) -- for normals (exponent field e -> fp16
//      exponent e, bias 15 vs 7 = 2^-8) AND for subnormals (E4M3's m * 2^-9 lands as fp16's m * 2^-17, the
//      same 2^-8, because the two formats' subnormal boundaries align). Two bytes of a 32-bit word relay
//      at once, then two half -> float conversions; the block's E8M0 scale is multiplied by 256 (exact).
//      Every product and every partial sum is the table variant's scaled by a power of two, so the
//      result is bit-identical (the unit check in bench/ds41_fp8_gemv_bench.cpp covers all 256 codes;
//      the two NaN codes take the scalar decode, which weights never hold). ~4.5 ops per weight and no
//      local-memory traffic against the table's one load per weight -- the SLM bank traffic was the
//      suspect for the 400 GB/s ceiling of variants 1-3 (docs/78).
template <int NC, int kDec>
static sycl::event gemv_fp8_e4m3_f16_t(sycl::queue& q, const sycl::half* x, const uint8_t* w, const uint8_t* scale,
                                       float* y, uint32_t K, uint32_t N, uint32_t cpg, const char* tag, const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE = 16, N_PER_WG = NC, WG_ITEMS = N_PER_WG * SG_SIZE;
    constexpr bool kLut16 = kDec == 1, kLut32 = kDec == 2 || kDec == 3, kStage = kDec >= 3, kRelay = kDec == 4;
    const uint32_t blocks_per_col = K / 32, SK = K / 32;
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;
    // `tag` (Phase 31 diagnostic): a per-call-site name so the profiler splits this kernel by SHAPE --
    // eight projections per layer share it and their efficiency differs by shape, not by decode variant.
    return ie::ps(q, tag ? tag : (kDec ? "gemv_fp8_e4m3_f16t" : "gemv_fp8_e4m3_f16"), [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<sycl::half, 1> lut16(sycl::range<1>(kLut16 ? 256u : 1u), h);
        sycl::local_accessor<float, 1> lut32(sycl::range<1>(kLut32 ? 256u : 1u), h);
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid = uint32_t(it.get_local_id(0)), wgid = uint32_t(it.get_group(0));
            const uint32_t sg_id = lid / SG_SIZE, lane = lid % SG_SIZE, n = wgid * N_PER_WG + sg_id;
            if constexpr (kLut16) {
                for (uint32_t i = lid; i < 256u; i += WG_ITEMS) lut16[i] = sycl::half(ds41_e4m3(uint8_t(i)));
                sycl::group_barrier(it.get_group());
            }
            if constexpr (kLut32) {
                for (uint32_t i = lid; i < 256u; i += WG_ITEMS) lut32[i] = ds41_e4m3(uint8_t(i));
                sycl::group_barrier(it.get_group());
            }
            if (n >= N) return;
            const uint8_t* wcol = w + uint64_t(n) * K;
            const uint8_t* srow = scale + uint64_t(n >> 5) * SK;
            const uint64_t xoff = uint64_t(n / cpg) * K;          // the block-diagonal group (cpg == N: no group, offset 0)
            auto dec = [&](uint32_t code) -> float {
                if constexpr (kLut16) return float(lut16[code]);
                else if constexpr (kLut32) return lut32[code];
                else return ds41_e4m3(uint8_t(code));
            };
            float acc = 0.f;
            for (uint32_t b = lane; b < blocks_per_col; b += SG_SIZE) {
                const auto* w128 = reinterpret_cast<const sycl::vec<uint32_t, 4>*>(wcol + uint64_t(b) * 32);
                const auto* xv = reinterpret_cast<const sycl::vec<sycl::half, 8>*>(x + xoff + uint64_t(b) * 32);
                const sycl::vec<uint32_t, 4> wa = w128[0], wb = w128[1];
                sycl::vec<sycl::half, 8> xq[4];
                #pragma unroll
                for (int j = 0; j < 4; ++j) xq[j] = xv[j];
                [[maybe_unused]] float xf[32];
                if constexpr (kStage) {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j)
                        #pragma unroll
                        for (int k = 0; k < 8; ++k) xf[8 * j + k] = float(xq[j][k]);
                }
                auto xat = [&](int i) -> float {
                    if constexpr (kStage) return xf[i];
                    else return float(xq[i >> 3][i & 7]);
                };
                float bacc = 0.f;
                if constexpr (kRelay) {
                    // Phase 38: bytes 0/2 and 1/3 of each word relaid to fp16 (value / 256) in one op each;
                    // a word holding a NaN code (low 7 bits all ones) takes the scalar decode instead.
                    auto relay4 = [&](uint32_t v, int i0) {
                        const uint32_t nan = ((v & 0x7F7F7F7Fu) + 0x01010101u) & 0x80808080u;
                        if (nan) {
                            bacc += (ds41_e4m3(uint8_t(v & 0xFFu)) * 0.00390625f)         * xat(i0 + 0);
                            bacc += (ds41_e4m3(uint8_t((v >> 8) & 0xFFu)) * 0.00390625f)  * xat(i0 + 1);
                            bacc += (ds41_e4m3(uint8_t((v >> 16) & 0xFFu)) * 0.00390625f) * xat(i0 + 2);
                            bacc += (ds41_e4m3(uint8_t(v >> 24)) * 0.00390625f)           * xat(i0 + 3);
                            return;
                        }
                        const uint32_t p02 = ((v & 0x007F007Fu) << 7) | ((v & 0x00800080u) << 8);
                        const uint32_t p13 = (((v >> 8) & 0x007F007Fu) << 7) | (((v >> 8) & 0x00800080u) << 8);
                        const sycl::half h0 = sycl::bit_cast<sycl::half>(uint16_t(p02 & 0xFFFFu));
                        const sycl::half h1 = sycl::bit_cast<sycl::half>(uint16_t(p13 & 0xFFFFu));
                        const sycl::half h2 = sycl::bit_cast<sycl::half>(uint16_t(p02 >> 16));
                        const sycl::half h3 = sycl::bit_cast<sycl::half>(uint16_t(p13 >> 16));
                        bacc += float(h0) * xat(i0 + 0);
                        bacc += float(h1) * xat(i0 + 1);
                        bacc += float(h2) * xat(i0 + 2);
                        bacc += float(h3) * xat(i0 + 3);
                    };
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) { relay4(wa[j], 4 * j); relay4(wb[j], 16 + 4 * j); }
                    acc += (ds41_e8m0(srow[b]) * 256.f) * bacc;
                    continue;
                }
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const uint32_t v = wa[j], u = wb[j];
                    bacc += dec(v & 0xFFu)         * xat(4 * j + 0);
                    bacc += dec((v >> 8) & 0xFFu)  * xat(4 * j + 1);
                    bacc += dec((v >> 16) & 0xFFu) * xat(4 * j + 2);
                    bacc += dec(v >> 24)           * xat(4 * j + 3);
                    bacc += dec(u & 0xFFu)         * xat(16 + 4 * j + 0);
                    bacc += dec((u >> 8) & 0xFFu)  * xat(16 + 4 * j + 1);
                    bacc += dec((u >> 16) & 0xFFu) * xat(16 + 4 * j + 2);
                    bacc += dec(u >> 24)           * xat(16 + 4 * j + 3);
                }
                acc += ds41_e8m0(srow[b]) * bacc;
            }
            acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
            if (lane == 0) y[n] = acc;
        });
    });
}

// Phase 31: NC -- columns per work-group -- is chosen by WORK-GROUP COUNT, not by N's divisibility.
// Measured per shape at ctx 2,048 (docs/deepseek41/72): efficiency against the 580 GB/s floor tracks
// parallelism, not size -- q_b at 2,048 work-groups reaches 71%, o_b (320) 72%, sh_gate/up (144) 48-49%,
// kv (32) 44%. The old rule gave every N%32==0 shape NC=16, starving the mid-N projections. NC changes
// only how columns are grouped into work-groups: each column's block order, FMAs and subgroup
// reduction are untouched, so this is BIT-IDENTICAL by construction. IE_DS41_FP8_NC=1|2|4|8|16 forces
// one value for A/B; 16 reproduces the old grouping for every N%32==0 shape.
static int fp8_nc_forced() {
    static const int v = [] { const char* e = std::getenv("IE_DS41_FP8_NC"); return e && *e ? std::atoi(e) : 0; }();
    return v;
}
template <int kDec>
static sycl::event gemv_fp8_dispatch(sycl::queue& q, const sycl::half* x, const uint8_t* w, const uint8_t* scale,
                                     float* y, uint32_t K, uint32_t N, uint32_t cpg, const char* tag, const std::vector<sycl::event>& deps) {
    const int f = fp8_nc_forced();
    const uint32_t nc = f > 0 ? uint32_t(f) : (N <= 1024u ? 2u : N <= 4096u ? 4u : N <= 8192u ? 8u : 16u);
    if (nc <= 1)  return gemv_fp8_e4m3_f16_t<1,  kDec>(q, x, w, scale, y, K, N, cpg, tag, deps);
    if (nc <= 2)  return gemv_fp8_e4m3_f16_t<2,  kDec>(q, x, w, scale, y, K, N, cpg, tag, deps);
    if (nc <= 4)  return gemv_fp8_e4m3_f16_t<4,  kDec>(q, x, w, scale, y, K, N, cpg, tag, deps);
    if (nc <= 8)  return gemv_fp8_e4m3_f16_t<8,  kDec>(q, x, w, scale, y, K, N, cpg, tag, deps);
    return gemv_fp8_e4m3_f16_t<16, kDec>(q, x, w, scale, y, K, N, cpg, tag, deps);
}

static sycl::event gemv_fp8_e4m3_f16_variant_tag(sycl::queue& q, const sycl::half* x, const uint8_t* w, const uint8_t* scale,
                                                 float* y, uint32_t K, uint32_t N, uint32_t cpg, int dec, const char* tag, const std::vector<sycl::event>& deps) {
    if (cpg == 0) cpg = N;
    // Phase 38 (docs/deepseek41/78): the default is chosen BY SHAPE -- the relay decode (4) from 2,048 columns up
    // (+13-20 % on every such shape, streamed from VRAM), the fp16 table (1) below, where q_a's 80 work-groups
    // cannot hide the relay's extra ALU work (-22 %). Both are bit-identical, so the choice is only about speed.
    // dec == -1 means "the default"; IE_DS41_FP8_PACKED forces one variant everywhere.
    if (dec < 0) dec = N >= 2048u ? 4 : 1;
    switch (dec) {
        case 1:  return gemv_fp8_dispatch<1>(q, x, w, scale, y, K, N, cpg, tag, deps);
        case 2:  return gemv_fp8_dispatch<2>(q, x, w, scale, y, K, N, cpg, tag, deps);
        case 3:  return gemv_fp8_dispatch<3>(q, x, w, scale, y, K, N, cpg, tag, deps);
        case 4:  return gemv_fp8_dispatch<4>(q, x, w, scale, y, K, N, cpg, tag, deps);
        default: return gemv_fp8_dispatch<0>(q, x, w, scale, y, K, N, cpg, tag, deps);
    }
}
sycl::event gemv_fp8_e4m3_f16_variant(sycl::queue& q, const sycl::half* x, const uint8_t* w, const uint8_t* scale,
                                      float* y, uint32_t K, uint32_t N, int dec, const std::vector<sycl::event>& deps) {
    return gemv_fp8_e4m3_f16_variant_tag(q, x, w, scale, y, K, N, N, dec, nullptr, deps);
}

// IE_DS41_FP8_PACKED: unset -> the per-shape default (Phase 38: relay from 2,048 columns, the fp16 table below);
// "0" -> the scalar decode; 1 / 2 / 3 -> that table variant; 4 -> the relay decode, everywhere.
// gate 17 PASS (docs/47): the fp16 table decode -- bit-identical to the scalar on every shape, 17.9 -> 14.7
// ms/token of device time; variants 2 / 3 measure the same and stay selectable. "0" restores the scalar decode.
constexpr int kDefaultDec = -1;
static int fp8_decode_variant() {
    static const int v = [] {
        const char* e = std::getenv("IE_DS41_FP8_PACKED");
        if (!e || !*e) return kDefaultDec;
        const int k = std::atoi(e);
        return k <= 0 ? 0 : (k > 4 ? 4 : k);
    }();
    return v;
}

sycl::event gemv_fp8_e4m3_f16(sycl::queue& q, const sycl::half* x, const uint8_t* w, const uint8_t* scale,
                              float* y, uint32_t K, uint32_t N, const std::vector<sycl::event>& deps) {
    return gemv_fp8_e4m3_f16_variant_tag(q, x, w, scale, y, K, N, N, fp8_decode_variant(), nullptr, deps);
}
sycl::event gemv_fp8_e4m3_f16_tagged(sycl::queue& q, const sycl::half* x, const uint8_t* w, const uint8_t* scale,
                                     float* y, uint32_t K, uint32_t N, const char* tag, const std::vector<sycl::event>& deps) {
    return gemv_fp8_e4m3_f16_variant_tag(q, x, w, scale, y, K, N, N, fp8_decode_variant(), tag, deps);
}
// Phase 32: the block-diagonal form. Column n reads the activation slice (n / cols_per_group) * K, exactly
// as gemv_f16_rows does for o_a; with cols_per_group == N it is the plain GEMV, bit for bit.
sycl::event gemv_fp8_e4m3_f16_grouped(sycl::queue& q, const sycl::half* x, const uint8_t* w, const uint8_t* scale,
                                      float* y, uint32_t K, uint32_t N, uint32_t cols_per_group, const char* tag,
                                      const std::vector<sycl::event>& deps) {
    return gemv_fp8_e4m3_f16_variant_tag(q, x, w, scale, y, K, N, cols_per_group, fp8_decode_variant(), tag, deps);
}

// the two decodes of docs/45 side by side, for the unit test's bit-identity check (criterion 1)
sycl::event gemv_fp8_e4m3_f16_scalar(sycl::queue& q, const sycl::half* x, const uint8_t* w, const uint8_t* scale,
                                     float* y, uint32_t K, uint32_t N, const std::vector<sycl::event>& deps) {
    return gemv_fp8_dispatch<0>(q, x, w, scale, y, K, N, N, nullptr, deps);
}
sycl::event gemv_fp8_e4m3_f16_packed(sycl::queue& q, const sycl::half* x, const uint8_t* w, const uint8_t* scale,
                                     float* y, uint32_t K, uint32_t N, const std::vector<sycl::event>& deps) {
    return gemv_fp8_dispatch<1>(q, x, w, scale, y, K, N, N, nullptr, deps);
}

}  // namespace ie

// ---------------------------------------------------------------------------------------------
// DSpark P2 (docs/deepseek41/55): the ROWS kernels -- the same GEMV for 1..8 activation rows, every row's
// arithmetic the M = 1 kernel's term for term (the same blocks in the same order per lane, the same FMA
// sequence per row, the same subgroup reduction), so a T-row decode step is bit-identical per row to T
// one-row steps, and the weights are read once for all rows. The fp16 variant takes an optional column
// group: column n reads the activation slice (n / cols_per_group) * K (the block-diagonal o_a).
namespace ie {

template <int NC, int M>
static sycl::event gemv_fp8_rows_t(sycl::queue& q, const sycl::half* x, uint32_t x_stride, const uint8_t* w, const uint8_t* scale,
                                   float* y, uint32_t K, uint32_t N, uint32_t cpg, const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE = 16, N_PER_WG = NC, WG_ITEMS = N_PER_WG * SG_SIZE;
    const uint32_t blocks_per_col = K / 32, SK = K / 32;
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;
    return ie::ps(q, "gemv_fp8_rows", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<sycl::half, 1> lut16(sycl::range<1>(256u), h);
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid = uint32_t(it.get_local_id(0)), wgid = uint32_t(it.get_group(0));
            const uint32_t sg_id = lid / SG_SIZE, lane = lid % SG_SIZE, n = wgid * N_PER_WG + sg_id;
            for (uint32_t i = lid; i < 256u; i += WG_ITEMS) lut16[i] = sycl::half(ds41_e4m3(uint8_t(i)));
            sycl::group_barrier(it.get_group());
            if (n >= N) return;
            const uint8_t* wcol = w + uint64_t(n) * K;
            const uint8_t* srow = scale + uint64_t(n >> 5) * SK;
            const uint64_t xoff = uint64_t(n / cpg) * K;          // the block-diagonal group (cpg == N: offset 0)
            float acc[M];
            #pragma unroll
            for (int m = 0; m < M; ++m) acc[m] = 0.f;
            for (uint32_t b = lane; b < blocks_per_col; b += SG_SIZE) {
                const auto* w128 = reinterpret_cast<const sycl::vec<uint32_t, 4>*>(wcol + uint64_t(b) * 32);
                const sycl::vec<uint32_t, 4> wa = w128[0], wb = w128[1];
                sycl::vec<sycl::half, 8> xq[M][4];
                #pragma unroll
                for (int m = 0; m < M; ++m) {
                    const auto* xv = reinterpret_cast<const sycl::vec<sycl::half, 8>*>(x + uint64_t(m) * x_stride + xoff + uint64_t(b) * 32);
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) xq[m][j] = xv[j];
                }
                float bacc[M];
                #pragma unroll
                for (int m = 0; m < M; ++m) bacc[m] = 0.f;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const uint32_t v = wa[j], u = wb[j];
                    const float d0 = float(lut16[v & 0xFFu]), d1 = float(lut16[(v >> 8) & 0xFFu]), d2 = float(lut16[(v >> 16) & 0xFFu]), d3 = float(lut16[v >> 24]);
                    const float e0 = float(lut16[u & 0xFFu]), e1 = float(lut16[(u >> 8) & 0xFFu]), e2 = float(lut16[(u >> 16) & 0xFFu]), e3 = float(lut16[u >> 24]);
                    #pragma unroll
                    for (int m = 0; m < M; ++m) {
                        bacc[m] += d0 * float(xq[m][(4 * j + 0) >> 3][(4 * j + 0) & 7]);
                        bacc[m] += d1 * float(xq[m][(4 * j + 1) >> 3][(4 * j + 1) & 7]);
                        bacc[m] += d2 * float(xq[m][(4 * j + 2) >> 3][(4 * j + 2) & 7]);
                        bacc[m] += d3 * float(xq[m][(4 * j + 3) >> 3][(4 * j + 3) & 7]);
                        bacc[m] += e0 * float(xq[m][(16 + 4 * j + 0) >> 3][(16 + 4 * j + 0) & 7]);
                        bacc[m] += e1 * float(xq[m][(16 + 4 * j + 1) >> 3][(16 + 4 * j + 1) & 7]);
                        bacc[m] += e2 * float(xq[m][(16 + 4 * j + 2) >> 3][(16 + 4 * j + 2) & 7]);
                        bacc[m] += e3 * float(xq[m][(16 + 4 * j + 3) >> 3][(16 + 4 * j + 3) & 7]);
                    }
                }
                const float sc = ds41_e8m0(srow[b]);
                #pragma unroll
                for (int m = 0; m < M; ++m) acc[m] += sc * bacc[m];
            }
            #pragma unroll
            for (int m = 0; m < M; ++m) {
                const float r = sycl::reduce_over_group(it.get_sub_group(), acc[m], sycl::plus<float>());
                if (lane == 0) y[uint64_t(m) * N + n] = r;
            }
        });
    });
}

template <int M>
static sycl::event gemv_fp8_rows_m(sycl::queue& q, const sycl::half* x, uint32_t x_stride, const uint8_t* w, const uint8_t* scale,
                                   float* y, uint32_t K, uint32_t N, uint32_t cpg, const std::vector<sycl::event>& deps) {
    if (N <= 256u) return gemv_fp8_rows_t<4, M>(q, x, x_stride, w, scale, y, K, N, cpg, deps);
    if (N % 32u == 0) return gemv_fp8_rows_t<16, M>(q, x, x_stride, w, scale, y, K, N, cpg, deps);
    return gemv_fp8_rows_t<8, M>(q, x, x_stride, w, scale, y, K, N, cpg, deps);
}

sycl::event gemv_fp8_rows(sycl::queue& q, const sycl::half* x, uint32_t x_stride, const uint8_t* w, const uint8_t* scale,
                          float* y, uint32_t M, uint32_t K, uint32_t N, const std::vector<sycl::event>& deps) {
    switch (M) {
        case 1: return gemv_fp8_rows_m<1>(q, x, x_stride, w, scale, y, K, N, N, deps);
        case 2: return gemv_fp8_rows_m<2>(q, x, x_stride, w, scale, y, K, N, N, deps);
        case 3: return gemv_fp8_rows_m<3>(q, x, x_stride, w, scale, y, K, N, N, deps);
        case 4: return gemv_fp8_rows_m<4>(q, x, x_stride, w, scale, y, K, N, N, deps);
        case 5: return gemv_fp8_rows_m<5>(q, x, x_stride, w, scale, y, K, N, N, deps);
        case 6: return gemv_fp8_rows_m<6>(q, x, x_stride, w, scale, y, K, N, N, deps);
        case 7: return gemv_fp8_rows_m<7>(q, x, x_stride, w, scale, y, K, N, N, deps);
        case 8: return gemv_fp8_rows_m<8>(q, x, x_stride, w, scale, y, K, N, N, deps);
        default: throw std::invalid_argument("gemv_fp8_rows: M must be 1..8");
    }
}
// Phase 32: the block-diagonal rows form (o_a at a 2..8-row decode step), column n reads x + (n / cols_per_group) * K.
sycl::event gemv_fp8_rows_grouped(sycl::queue& q, const sycl::half* x, uint32_t x_stride, const uint8_t* w, const uint8_t* scale,
                                  float* y, uint32_t M, uint32_t K, uint32_t N, uint32_t cols_per_group, const std::vector<sycl::event>& deps) {
    if (cols_per_group == 0) cols_per_group = N;
    switch (M) {
        case 1: return gemv_fp8_rows_m<1>(q, x, x_stride, w, scale, y, K, N, cols_per_group, deps);
        case 2: return gemv_fp8_rows_m<2>(q, x, x_stride, w, scale, y, K, N, cols_per_group, deps);
        case 3: return gemv_fp8_rows_m<3>(q, x, x_stride, w, scale, y, K, N, cols_per_group, deps);
        case 4: return gemv_fp8_rows_m<4>(q, x, x_stride, w, scale, y, K, N, cols_per_group, deps);
        case 5: return gemv_fp8_rows_m<5>(q, x, x_stride, w, scale, y, K, N, cols_per_group, deps);
        case 6: return gemv_fp8_rows_m<6>(q, x, x_stride, w, scale, y, K, N, cols_per_group, deps);
        case 7: return gemv_fp8_rows_m<7>(q, x, x_stride, w, scale, y, K, N, cols_per_group, deps);
        case 8: return gemv_fp8_rows_m<8>(q, x, x_stride, w, scale, y, K, N, cols_per_group, deps);
        default: throw std::invalid_argument("gemv_fp8_rows_grouped: M must be 1..8");
    }
}

// fp16 weights [N, K], fp16 activations [M, x_stride], fp32 out [M, N]; column n reads x + (n / cols_per_group) * K
template <int NC, int M>
static sycl::event gemv_f16_rows_t(sycl::queue& q, const sycl::half* x, uint32_t x_stride, const sycl::half* w, float* y,
                                   uint32_t K, uint32_t N, uint32_t cols_per_group, const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE = 16, N_PER_WG = NC, WG_ITEMS = N_PER_WG * SG_SIZE;
    const uint32_t blocks_per_col = K / 32;
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;
    return ie::ps(q, "gemv_f16_rows", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid = uint32_t(it.get_local_id(0)), wgid = uint32_t(it.get_group(0));
            const uint32_t sg_id = lid / SG_SIZE, lane = lid % SG_SIZE, n = wgid * N_PER_WG + sg_id;
            if (n >= N) return;
            const sycl::half* wcol = w + uint64_t(n) * K;
            const uint64_t xoff = uint64_t(n / cols_per_group) * K;
            float acc[M];
            #pragma unroll
            for (int m = 0; m < M; ++m) acc[m] = 0.f;
            for (uint32_t b = lane; b < blocks_per_col; b += SG_SIZE) {
                const auto* wv = reinterpret_cast<const sycl::vec<sycl::half, 8>*>(wcol + uint64_t(b) * 32);
                sycl::vec<sycl::half, 8> wq[4];
                #pragma unroll
                for (int j = 0; j < 4; ++j) wq[j] = wv[j];
                #pragma unroll
                for (int m = 0; m < M; ++m) {
                    const auto* xv = reinterpret_cast<const sycl::vec<sycl::half, 8>*>(x + uint64_t(m) * x_stride + xoff + uint64_t(b) * 32);
                    float bacc = 0.f;
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const sycl::vec<sycl::half, 8> xq = xv[j];
                        #pragma unroll
                        for (int k = 0; k < 8; ++k) bacc += float(wq[j][k]) * float(xq[k]);
                    }
                    acc[m] += bacc;
                }
            }
            #pragma unroll
            for (int m = 0; m < M; ++m) {
                const float r = sycl::reduce_over_group(it.get_sub_group(), acc[m], sycl::plus<float>());
                if (lane == 0) y[uint64_t(m) * N + n] = r;
            }
        });
    });
}

template <int M>
static sycl::event gemv_f16_rows_m(sycl::queue& q, const sycl::half* x, uint32_t x_stride, const sycl::half* w, float* y,
                                   uint32_t K, uint32_t N, uint32_t cpg, const std::vector<sycl::event>& deps) {
    if (N <= 256u) return gemv_f16_rows_t<4, M>(q, x, x_stride, w, y, K, N, cpg, deps);
    if (N % 32u == 0) return gemv_f16_rows_t<16, M>(q, x, x_stride, w, y, K, N, cpg, deps);
    return gemv_f16_rows_t<8, M>(q, x, x_stride, w, y, K, N, cpg, deps);
}

sycl::event gemv_f16_rows(sycl::queue& q, const sycl::half* x, uint32_t x_stride, const sycl::half* w, float* y,
                          uint32_t M, uint32_t K, uint32_t N, uint32_t cols_per_group, const std::vector<sycl::event>& deps) {
    if (cols_per_group == 0) cols_per_group = N;
    switch (M) {
        case 1: return gemv_f16_rows_m<1>(q, x, x_stride, w, y, K, N, cols_per_group, deps);
        case 2: return gemv_f16_rows_m<2>(q, x, x_stride, w, y, K, N, cols_per_group, deps);
        case 3: return gemv_f16_rows_m<3>(q, x, x_stride, w, y, K, N, cols_per_group, deps);
        case 4: return gemv_f16_rows_m<4>(q, x, x_stride, w, y, K, N, cols_per_group, deps);
        case 5: return gemv_f16_rows_m<5>(q, x, x_stride, w, y, K, N, cols_per_group, deps);
        case 6: return gemv_f16_rows_m<6>(q, x, x_stride, w, y, K, N, cols_per_group, deps);
        case 7: return gemv_f16_rows_m<7>(q, x, x_stride, w, y, K, N, cols_per_group, deps);
        case 8: return gemv_f16_rows_m<8>(q, x, x_stride, w, y, K, N, cols_per_group, deps);
        default: throw std::invalid_argument("gemv_f16_rows: M must be 1..8");
    }
}

}  // namespace ie
