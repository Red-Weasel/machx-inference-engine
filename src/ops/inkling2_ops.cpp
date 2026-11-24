// src/ops/inkling2_ops.cpp — Inkling 2 Mini primitive ops, phase 1: plain fp32 kernels written for correctness
// against docs/inkling2/SPEC.md, not for speed (one work-group per output row, no XMX, no fusion beyond what the
// formulas need). See include/ie/inkling2_ops.hpp for the contracts.
//
// Kernel lambdas live in functions of the named namespace ie (prefix i2_) — never in an anonymous namespace with a
// name another TU also uses (the SYCL kernel-name collision trap: two TUs' identically named anonymous helpers can
// resolve to one kernel).

#include "ie/inkling2_ops.hpp"

#include "ie/kernel_profiler.hpp"
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <cmath>

namespace ie {

namespace {
inline float i2r(float x, bool round) { return round ? i2_bf16_round(x) : x; }
}  // namespace

namespace {
struct I2LinPack {                                                      // up to 8 projections in one launch
    const float* x[8];
    const uint16_t* W[8];
    float* y[8];
    uint32_t N[8];
    uint32_t g0[9];                                                     // first work-group of each segment
    uint32_t n = 0;
};
// Few tokens (T <= 4): one 16-lane sub-group per output row, each lane loads 8 bf16 (16 bytes) per step and keeps
// one fp32 accumulator per token; 8 rows of one segment per work-group. Needs K % 8 == 0 and 16-byte aligned rows.
template <uint32_t TT>
sycl::event i2_linear_bf16_sg(sycl::queue& q, const I2LinPack& pk, uint32_t Tall, uint32_t K, bool round) {
    constexpr uint32_t SG = 16, RPG = 8;
    const uint32_t groups = pk.g0[pk.n];
    const uint32_t tiles = (Tall + TT - 1) / TT;                           // token tiles (prefill); consecutive groups
    const char* nm = Tall > 4 ? "i2_linear_bf16_sg_tiled" : groups > 8192 ? "i2_linear_bf16_sg_big" : groups <= 16 ? "i2_linear_bf16_sg_small" : K > 2304 ? "i2_linear_bf16_sg_bigK" : K < 256 ? "i2_linear_bf16_sg_smallK" : "i2_linear_bf16_sg";
    return ie::ps(q, nm, [&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(uint64_t(groups) * tiles * RPG * SG, RPG * SG),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const auto sg = it.get_sub_group();
            const uint32_t tile = uint32_t(it.get_group(0) % tiles);     // the same rows for all tiles in a row
            const uint32_t gid = uint32_t(it.get_group(0) / tiles);
            const uint32_t t0 = tile * TT, T = sycl::min(TT, Tall - t0);
            uint32_t sgi = 0;
            for (uint32_t i = 1; i < pk.n; ++i) if (gid >= pk.g0[i]) sgi = i;
            const uint32_t N = pk.N[sgi];
            const uint64_t n = uint64_t(gid - pk.g0[sgi]) * RPG + sg.get_group_linear_id();
            if (n >= N) return;
            const float* x = pk.x[sgi] + uint64_t(t0) * K;
            float* y = pk.y[sgi] + uint64_t(t0) * N;
            const uint32_t l = uint32_t(sg.get_local_linear_id());
            const sycl::uint4* w = reinterpret_cast<const sycl::uint4*>(pk.W[sgi] + n * K);
            float acc[TT];
            for (uint32_t t = 0; t < TT; ++t) acc[t] = 0.f;
            auto step = [&](const sycl::uint4 v, uint32_t c) {
                float wf[8];
                for (int j = 0; j < 4; ++j) {
                    const uint32_t u = v[j];
                    wf[2 * j] = sycl::bit_cast<float>(u << 16);
                    wf[2 * j + 1] = sycl::bit_cast<float>(u & 0xffff0000u);
                }
                for (uint32_t t = 0; t < TT; ++t) {
                    if (t >= T) break;
                    const sycl::float4* xr = reinterpret_cast<const sycl::float4*>(x + uint64_t(t) * K) + 2 * c;
                    const sycl::float4 a = xr[0], b = xr[1];
                    acc[t] += a[0] * wf[0] + a[1] * wf[1] + a[2] * wf[2] + a[3] * wf[3] +
                              b[0] * wf[4] + b[1] * wf[5] + b[2] * wf[6] + b[3] * wf[7];
                }
            };
            uint32_t c = l;
            for (; c + 3 * SG < K / 8; c += 4 * SG) {                    // four loads in flight per lane, same order
                const sycl::uint4 v0 = w[c], v1 = w[c + SG], v2 = w[c + 2 * SG], v3 = w[c + 3 * SG];
                step(v0, c); step(v1, c + SG); step(v2, c + 2 * SG); step(v3, c + 3 * SG);
            }
            for (; c < K / 8; c += SG) step(w[c], c);
            for (uint32_t t = 0; t < TT; ++t) {
                if (t >= T) break;
                const float s = sycl::reduce_over_group(sg, acc[t], sycl::plus<float>());
                if (l == 0) y[uint64_t(t) * N + n] = i2r(s, round);
            }
        });
    });
}
// Prefill (T > 4): token tiles of TT, RPS rows per sub-group so each x load serves RPS rows (x traffic, not the
// weights, limits a tiled fp32 GEMM here); every row's sum is formed in the same order as in i2_linear_bf16_sg (no
// unrolled-load difference: the order of the adds per lane is the chunk order either way).
template <uint32_t TT, uint32_t RPS>
sycl::event i2_linear_bf16_tiled(sycl::queue& q, I2LinPack pk, uint32_t Tall, uint32_t K, bool round) {
    constexpr uint32_t SG = 16, NSG = 8, RPG = NSG * RPS;
    pk.g0[0] = 0;
    for (uint32_t i = 0; i < pk.n; ++i) pk.g0[i + 1] = pk.g0[i] + (pk.N[i] + RPG - 1) / RPG;
    const uint32_t groups = pk.g0[pk.n];
    const uint32_t tiles = (Tall + TT - 1) / TT;
    const char* nm = groups * RPG > 65536 ? "i2_linear_bf16_tiled_big" : K > 2304 ? "i2_linear_bf16_tiled_bigK" : K < 256 ? "i2_linear_bf16_tiled_smallK" : "i2_linear_bf16_tiled";
    return ie::ps(q, nm, [&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(uint64_t(groups) * tiles * NSG * SG, NSG * SG),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const auto sg = it.get_sub_group();
            const uint32_t tile = uint32_t(it.get_group(0) % tiles);
            const uint32_t gid = uint32_t(it.get_group(0) / tiles);
            const uint32_t t0 = tile * TT, T = sycl::min(TT, Tall - t0);
            uint32_t sgi = 0;
            for (uint32_t i = 1; i < pk.n; ++i) if (gid >= pk.g0[i]) sgi = i;
            const uint32_t N = pk.N[sgi];
            const uint64_t n0 = uint64_t(gid - pk.g0[sgi]) * RPG + uint64_t(sg.get_group_linear_id()) * RPS;
            if (n0 >= N) return;
            const float* x = pk.x[sgi] + uint64_t(t0) * K;
            float* y = pk.y[sgi] + uint64_t(t0) * N;
            const uint32_t l = uint32_t(sg.get_local_linear_id());
            const uint16_t* Wb = pk.W[sgi];
            float acc[RPS][TT];
#pragma unroll
            for (uint32_t r = 0; r < RPS; ++r)
#pragma unroll
                for (uint32_t t = 0; t < TT; ++t) acc[r][t] = 0.f;
            for (uint32_t c = l; c < K / 8; c += SG) {
                float wf[RPS][8];
#pragma unroll
                for (uint32_t r = 0; r < RPS; ++r) {
                    const uint64_t nr = sycl::min(uint64_t(n0 + r), uint64_t(N - 1));      // a tail row repeats the last (unused)
                    const sycl::uint4 v = reinterpret_cast<const sycl::uint4*>(Wb + nr * K)[c];
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const uint32_t u = v[j];
                        wf[r][2 * j] = sycl::bit_cast<float>(u << 16);
                        wf[r][2 * j + 1] = sycl::bit_cast<float>(u & 0xffff0000u);
                    }
                }
#pragma unroll
                for (uint32_t t = 0; t < TT; ++t) {
                    if (t >= T) break;
                    const sycl::float4* xr = reinterpret_cast<const sycl::float4*>(x + uint64_t(t) * K) + 2 * c;
                    const sycl::float4 a = xr[0], b = xr[1];
#pragma unroll
                    for (uint32_t r = 0; r < RPS; ++r)
                        acc[r][t] += a[0] * wf[r][0] + a[1] * wf[r][1] + a[2] * wf[r][2] + a[3] * wf[r][3] +
                                     b[0] * wf[r][4] + b[1] * wf[r][5] + b[2] * wf[r][6] + b[3] * wf[r][7];
                }
            }
#pragma unroll
            for (uint32_t r = 0; r < RPS; ++r) {
                if (n0 + r >= N) break;
#pragma unroll
                for (uint32_t t = 0; t < TT; ++t) {
                    if (t >= T) break;
                    const float s2 = sycl::reduce_over_group(sg, acc[r][t], sycl::plus<float>());
                    if (l == 0) y[uint64_t(t) * N + n0 + r] = i2r(s2, round);
                }
            }
        });
    });
}
// Batched decoding rows (T 5..TT): the TT tokens' x chunk is loaded once per step and serves RPS rows (x, not the
// weights, is the traffic that limits the tiled kernel); each row's sum per token is formed in the same chunk order as
// i2_linear_bf16_sg (bit-identical per token whatever T).
template <typename Body>
struct I2BigGrf {                                                       // 256 registers per thread, 16-lane sub-groups
    Body body;
    void operator()(sycl::nd_item<1> it) const { body(it); }
    auto get(sycl::ext::oneapi::experimental::properties_tag) const {
        return sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>,
                                                           sycl::ext::oneapi::experimental::sub_group_size<16>};
    }
};
template <uint32_t TT, uint32_t RPS, bool BIG = false>
sycl::event i2_linear_bf16_xfirst(sycl::queue& q, I2LinPack pk, uint32_t Tall, uint32_t K, bool round) {
    constexpr uint32_t SG = 16, NSG = 8, RPG = NSG * RPS;
    pk.g0[0] = 0;
    for (uint32_t i = 0; i < pk.n; ++i) pk.g0[i + 1] = pk.g0[i] + (pk.N[i] + RPG - 1) / RPG;
    const uint32_t groups = pk.g0[pk.n];
    const uint32_t tiles = (Tall + TT - 1) / TT;
    return ie::ps(q, "i2_linear_bf16_xfirst", [&](sycl::handler& h) {
        auto body = [=](sycl::nd_item<1> it) {
            const auto sg = it.get_sub_group();
            const uint32_t tile = uint32_t(it.get_group(0) % tiles);
            const uint32_t gid = uint32_t(it.get_group(0) / tiles);
            const uint32_t t0 = tile * TT, T = sycl::min(TT, Tall - t0);
            uint32_t sgi = 0;
            for (uint32_t i = 1; i < pk.n; ++i) if (gid >= pk.g0[i]) sgi = i;
            const uint32_t N = pk.N[sgi];
            const uint64_t n0 = uint64_t(gid - pk.g0[sgi]) * RPG + uint64_t(sg.get_group_linear_id()) * RPS;
            if (n0 >= N) return;
            const float* x = pk.x[sgi] + uint64_t(t0) * K;
            float* y = pk.y[sgi] + uint64_t(t0) * N;
            const uint32_t l = uint32_t(sg.get_local_linear_id());
            const uint16_t* Wb = pk.W[sgi];
            float acc[RPS][TT];
#pragma unroll
            for (uint32_t r = 0; r < RPS; ++r)
#pragma unroll
                for (uint32_t t = 0; t < TT; ++t) acc[r][t] = 0.f;
            for (uint32_t c = l; c < K / 8; c += SG) {
                sycl::uint4 v[RPS];
#pragma unroll
                for (uint32_t r = 0; r < RPS; ++r) {
                    const uint64_t nr = sycl::min(uint64_t(n0 + r), uint64_t(N - 1));
                    v[r] = reinterpret_cast<const sycl::uint4*>(Wb + nr * K)[c];
                }
                sycl::float4 xa[TT], xb[TT];
#pragma unroll
                for (uint32_t t = 0; t < TT; ++t) {
                    const sycl::float4* xr = reinterpret_cast<const sycl::float4*>(x + uint64_t(sycl::min(t, T - 1)) * K) + 2 * c;
                    xa[t] = xr[0]; xb[t] = xr[1];
                }
#pragma unroll
                for (uint32_t r = 0; r < RPS; ++r) {
                    float wf[8];
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        wf[2 * j] = sycl::bit_cast<float>(v[r][j] << 16);
                        wf[2 * j + 1] = sycl::bit_cast<float>(v[r][j] & 0xffff0000u);
                    }
#pragma unroll
                    for (uint32_t t = 0; t < TT; ++t) {
                        const sycl::float4 a = xa[t], b = xb[t];
                        acc[r][t] += a[0] * wf[0] + a[1] * wf[1] + a[2] * wf[2] + a[3] * wf[3] +
                                     b[0] * wf[4] + b[1] * wf[5] + b[2] * wf[6] + b[3] * wf[7];
                    }
                }
            }
#pragma unroll
            for (uint32_t r = 0; r < RPS; ++r) {
                if (n0 + r >= N) break;
#pragma unroll
                for (uint32_t t = 0; t < TT; ++t) {
                    if (t >= T) break;
                    const float s2 = sycl::reduce_over_group(sg, acc[r][t], sycl::plus<float>());
                    if (l == 0) y[uint64_t(t) * N + n0 + r] = i2r(s2, round);
                }
            }
        };
        const sycl::nd_range<1> nr(uint64_t(groups) * tiles * NSG * SG, NSG * SG);
        if constexpr (BIG) h.parallel_for(nr, I2BigGrf<decltype(body)>{body});
        else h.parallel_for(nr, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] { body(it); });
    });
}
// Few rows (router, gate down-projection): one work-group per row, its 8 sub-groups take every 8th 16-byte chunk of
// the row and the 8 partial sums are added in sub-group order through local memory.
template <uint32_t TT>
sycl::event i2_linear_bf16_rows(sycl::queue& q, const I2LinPack& pk, uint32_t Tall, uint32_t K, bool round) {
    // Tall > TT (batched decoding): token tiles of TT, each formed exactly as alone.
    constexpr uint32_t SG = 16, NSG = 8;
    const uint32_t tiles = (Tall + TT - 1) / TT;
    uint32_t rows = 0;
    uint32_t r0[9];
    for (uint32_t i = 0; i < pk.n; ++i) { r0[i] = rows; rows += pk.N[i]; }
    r0[pk.n] = rows;
    const uint32_t nseg = pk.n;
    return ie::ps(q, "i2_linear_bf16_rows", [&](sycl::handler& h) {
        sycl::local_accessor<float, 1> red(sycl::range<1>(NSG * TT), h);
        h.parallel_for(sycl::nd_range<1>(uint64_t(rows) * tiles * NSG * SG, NSG * SG),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const auto sg = it.get_sub_group();
            const uint32_t row = uint32_t(it.get_group(0) / tiles), tile = uint32_t(it.get_group(0) % tiles);
            const uint32_t t0 = tile * TT, T = sycl::min(TT, Tall - t0);
            uint32_t si = 0;
            for (uint32_t i = 1; i < nseg; ++i) if (row >= r0[i]) si = i;
            const uint32_t N = pk.N[si];
            const uint64_t n = row - r0[si];
            const float* x = pk.x[si] + uint64_t(t0) * K;
            const uint32_t l = uint32_t(sg.get_local_linear_id()), sgid = uint32_t(sg.get_group_linear_id());
            const sycl::uint4* w = reinterpret_cast<const sycl::uint4*>(pk.W[si] + n * K);
            float acc[TT];
            for (uint32_t t = 0; t < TT; ++t) acc[t] = 0.f;
            for (uint32_t c = sgid * SG + l; c < K / 8; c += NSG * SG) {
                const sycl::uint4 v = w[c];
                float wf[8];
                for (int j = 0; j < 4; ++j) {
                    const uint32_t u = v[j];
                    wf[2 * j] = sycl::bit_cast<float>(u << 16);
                    wf[2 * j + 1] = sycl::bit_cast<float>(u & 0xffff0000u);
                }
                for (uint32_t t = 0; t < TT; ++t) {
                    if (t >= T) break;
                    const sycl::float4* xr = reinterpret_cast<const sycl::float4*>(x + uint64_t(t) * K) + 2 * c;
                    const sycl::float4 a = xr[0], b = xr[1];
                    acc[t] += a[0] * wf[0] + a[1] * wf[1] + a[2] * wf[2] + a[3] * wf[3] +
                              b[0] * wf[4] + b[1] * wf[5] + b[2] * wf[6] + b[3] * wf[7];
                }
            }
            for (uint32_t t = 0; t < TT; ++t) {
                if (t >= T) break;
                const float r = sycl::reduce_over_group(sg, acc[t], sycl::plus<float>());
                if (l == 0) red[sgid * TT + t] = r;
            }
            sycl::group_barrier(it.get_group());
            if (sgid == 0 && l < TT && l < T) {
                float r = 0.f;
                for (uint32_t j = 0; j < NSG; ++j) r += red[j * TT + l];
                pk.y[si][uint64_t(t0 + l) * N + n] = i2r(r, round);
            }
        });
    });
}
bool i2_lin_sg_ok(uint32_t T, uint32_t K, const void* W, const float* x) {
    static const bool generic_only = std::getenv("IE_INKLING2_LINEAR_GENERIC") != nullptr;   // A/B switch
    (void)T;
    return !generic_only && K % 8 == 0 && (reinterpret_cast<uintptr_t>(W) & 15) == 0 &&
           (reinterpret_cast<uintptr_t>(x) & 15) == 0;
}
}  // namespace

sycl::event i2_linear_bf16_multi(sycl::queue& q, uint32_t n, const I2LinSeg* seg, uint32_t T, uint32_t K, bool round,
                                 bool fast, bool rowwise) {
    bool ok = fast && n <= 8;
    for (uint32_t i = 0; ok && i < n; ++i) ok = i2_lin_sg_ok(T, K, seg[i].W, seg[i].x);
    if (!ok) {
        sycl::event ev;
        for (uint32_t i = 0; i < n; ++i) ev = i2_linear_bf16(q, seg[i].x, T, K, seg[i].W, seg[i].N, seg[i].y, round, false);
        return ev;
    }
    I2LinPack pk{};
    pk.n = n;
    pk.g0[0] = 0;
    for (uint32_t i = 0; i < n; ++i) {
        pk.x[i] = seg[i].x; pk.W[i] = seg[i].W; pk.y[i] = seg[i].y; pk.N[i] = seg[i].N;
        pk.g0[i + 1] = pk.g0[i] + (seg[i].N + 7) / 8;
    }
    static const bool no_rows = std::getenv("IE_INKLING2_LINEAR_NO_ROWS") != nullptr;          // A/B switch
    static const uint32_t rows_max = [] { const char* e = std::getenv("IE_INKLING2_ROWS_MAX"); return e ? uint32_t(std::atoi(e)) : 512u; }();
    if (!no_rows && (T <= 4 || rowwise) && pk.g0[n] * 8 < rows_max && K >= 1024)   // < 512 rows of a long K: a work-group per row
        return T == 1 ? i2_linear_bf16_rows<1>(q, pk, T, K, round) : i2_linear_bf16_rows<4>(q, pk, T, K, round);
    if (T == 1) return i2_linear_bf16_sg<1>(q, pk, T, K, round);
    if (T <= 4) return i2_linear_bf16_sg<4>(q, pk, T, K, round);
    static const uint32_t sgmax = [] { const char* e = std::getenv("IE_INKLING2_LINEAR_SGMAX"); return e ? uint32_t(std::atoi(e)) : 0u; }();
    static const uint32_t xf = [] { const char* e = std::getenv("IE_INKLING2_LINEAR_XF"); return e ? uint32_t(std::atoi(e)) : 10812u; }();
    if (xf == 68 && T <= 6) return i2_linear_bf16_xfirst<6, 8>(q, pk, T, K, round);
    if (xf == 1068 && T <= 6) return i2_linear_bf16_xfirst<6, 8, true>(q, pk, T, K, round);
    if (xf == 1088 && T <= 8) return i2_linear_bf16_xfirst<8, 8, true>(q, pk, T, K, round);
    if (xf == 10812 && T <= 8) return i2_linear_bf16_xfirst<8, 12, true>(q, pk, T, K, round);
    if (xf == 1128 && T <= 12) return i2_linear_bf16_xfirst<12, 8, true>(q, pk, T, K, round);
    if (xf == 1168 && T <= 16) return i2_linear_bf16_xfirst<16, 8, true>(q, pk, T, K, round);
    if (xf == 64 && T <= 6) return i2_linear_bf16_xfirst<6, 4>(q, pk, T, K, round);
    if (xf == 88 && T <= 8) return i2_linear_bf16_xfirst<8, 8>(q, pk, T, K, round);
    if (xf == 84 && T <= 8) return i2_linear_bf16_xfirst<8, 4>(q, pk, T, K, round);
    if (xf == 126 && T <= 12) return i2_linear_bf16_xfirst<12, 6>(q, pk, T, K, round);
    if (xf == 124 && T <= 12) return i2_linear_bf16_xfirst<12, 4>(q, pk, T, K, round);
    if (T <= 8 && sgmax >= 8) return i2_linear_bf16_sg<8>(q, pk, T, K, round);
    if (T <= 16 && sgmax >= 16) return i2_linear_bf16_sg<16>(q, pk, T, K, round);
    static const uint32_t rps = [] { const char* e = std::getenv("IE_INKLING2_PREFILL_RPS"); return e ? uint32_t(std::atoi(e)) : 4u; }();
    static const uint32_t ptt = [] { const char* e = std::getenv("IE_INKLING2_PREFILL_TT"); return e ? uint32_t(std::atoi(e)) : 8u; }();
    if (rps == 1) return ptt == 16 ? i2_linear_bf16_tiled<16, 1>(q, pk, T, K, round) : i2_linear_bf16_tiled<8, 1>(q, pk, T, K, round);
    if (rps == 4) return ptt == 16 ? i2_linear_bf16_tiled<16, 4>(q, pk, T, K, round) : i2_linear_bf16_tiled<8, 4>(q, pk, T, K, round);
    return ptt == 16 ? i2_linear_bf16_tiled<16, 2>(q, pk, T, K, round) : i2_linear_bf16_tiled<8, 2>(q, pk, T, K, round);   // prefill
}

sycl::event i2_linear_bf16(sycl::queue& q, const float* x, uint32_t T, uint32_t K, const uint16_t* W, uint32_t N,
                           float* y, bool round, bool fast, bool rowwise) {
    if (fast && i2_lin_sg_ok(T, K, W, x)) {
        const I2LinSeg sg1{x, W, N, y};
        return i2_linear_bf16_multi(q, 1, &sg1, T, K, round, true, rowwise);
    }
    constexpr uint32_t WG = 64;
    return ie::ps(q, "i2_linear_bf16", [&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(uint64_t(N) * WG, WG), [=](sycl::nd_item<1> it) {
            const uint64_t n = it.get_group(0);
            const uint32_t l = uint32_t(it.get_local_id(0));
            const uint16_t* w = W + n * K;
            for (uint32_t t = 0; t < T; ++t) {
                const float* xr = x + uint64_t(t) * K;
                float acc = 0.f;
                for (uint32_t k = l; k < K; k += WG) acc += xr[k] * i2_bf16_to_f32(w[k]);
                const float s = sycl::reduce_over_group(it.get_group(), acc, sycl::plus<float>());
                if (l == 0) y[uint64_t(t) * N + n] = i2r(s, round);
            }
        });
    });
}

sycl::event i2_rmsnorm(sycl::queue& q, const float* x, uint32_t rows, uint32_t dim, const float* w, float eps,
                       float* y, uint32_t y_stride, bool round) {
    constexpr uint32_t WG = 256;
    return ie::ps(q, "i2_rmsnorm", [&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(uint64_t(rows) * WG, WG), [=](sycl::nd_item<1> it) {
            const uint64_t r = it.get_group(0);
            const uint32_t l = uint32_t(it.get_local_id(0));
            const float* xr = x + r * dim;
            float ss = 0.f;
            for (uint32_t i = l; i < dim; i += WG) ss += xr[i] * xr[i];
            ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
            const float inv = 1.f / sycl::sqrt(ss / float(dim) + eps);
            float* yr = y + r * y_stride;
            for (uint32_t i = l; i < dim; i += WG) {
                const float v = xr[i] * inv;
                yr[i] = i2r(w ? w[i] * v : v, round);
            }
        });
    });
}

sycl::event i2_rmsnorm_multi(sycl::queue& q, uint32_t n, const I2NormSeg* seg, uint32_t rows, uint32_t dim, float eps,
                             uint32_t y_stride, bool round) {
    if (n == 0 || n > 8) {
        sycl::event ev;
        for (uint32_t i = 0; i < n; ++i) ev = i2_rmsnorm(q, seg[i].x, rows, dim, seg[i].w, eps, seg[i].y, y_stride, round);
        return ev;
    }
    constexpr uint32_t WG = 256;
    struct P { const float* x[8]; const float* w[8]; float* y[8]; } pk{};
    for (uint32_t i = 0; i < n; ++i) { pk.x[i] = seg[i].x; pk.w[i] = seg[i].w; pk.y[i] = seg[i].y; }
    return ie::ps(q, "i2_rmsnorm", [&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(uint64_t(rows) * n * WG, WG), [=](sycl::nd_item<1> it) {
            const uint64_t gid = it.get_group(0);
            const uint32_t si = uint32_t(gid % n);
            const uint64_t r = gid / n;
            const uint32_t l = uint32_t(it.get_local_id(0));
            const float* xr = pk.x[si] + r * dim;
            const float* w = pk.w[si];
            float ss = 0.f;
            for (uint32_t i = l; i < dim; i += WG) ss += xr[i] * xr[i];
            ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
            const float inv = 1.f / sycl::sqrt(ss / float(dim) + eps);
            float* yr = pk.y[si] + r * y_stride;
            for (uint32_t i = l; i < dim; i += WG) {
                const float v = xr[i] * inv;
                yr[i] = i2r(w ? w[i] * v : v, round);
            }
        });
    });
}

sycl::event i2_scale_l2clip(sycl::queue& q, const float* x, uint32_t rows, uint32_t dim, const float* scale,
                            uint32_t heads, float thr, float* y, bool round) {
    constexpr uint32_t WG = 64;
    return ie::ps(q, "i2_scale_l2clip", [&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(uint64_t(rows) * WG, WG), [=](sycl::nd_item<1> it) {
            const uint64_t r = it.get_group(0);
            const uint32_t l = uint32_t(it.get_local_id(0));
            const float* xr = x + r * dim;
            const float* sr = scale ? scale + (r % heads) * dim : nullptr;
            float ss = 0.f;
            for (uint32_t i = l; i < dim; i += WG) {
                const float v = sr ? xr[i] * sr[i] : xr[i];
                ss += v * v;
            }
            ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
            const float f = sycl::fmin(thr / (sycl::sqrt(ss) + 1e-6f), 1.f);
            // every work-item read its elements before the reduction (a group barrier); writes may now alias x
            float* yr = y + r * dim;
            for (uint32_t i = l; i < dim; i += WG) {
                const float v = sr ? xr[i] * sr[i] : xr[i];
                yr[i] = i2r(v * f, round);
            }
        });
    });
}

sycl::event i2_mul_token(sycl::queue& q, const float* x, uint32_t T, uint32_t len, const float* f, float* y, bool round) {
    const uint64_t n = uint64_t(T) * len;
    return ie::ps(q, "i2_mul_token", [&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
            const uint64_t i = id[0];
            y[i] = i2r(x[i] * f[i / len], round);
        });
    });
}

sycl::event i2_sconv_static(sycl::queue& q, const float* x, uint32_t T, uint32_t C, const float* w, float* hist,
                            uint32_t hist_rows, uint32_t pos0, float* y, bool round) {
    const float div = std::sqrt(5.0f);                                  // sqrt(K + 1), K = 4
    ie::ps(q, "i2_sconv_static", [&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(uint64_t(T) * C), [=](sycl::id<1> id) {
            const uint64_t i = id[0];
            const int t = int(i / C);
            const uint32_t c = uint32_t(i % C);
            float conv = 0.f;
            for (int j = 0; j < 4; ++j) {                               // tap j reads x[t - 3 + j]; tap 3 = current
                const int s = t - 3 + j;
                const int64_t p = int64_t(pos0) + s;
                const float xv = s >= 0 ? x[uint64_t(s) * C + c] : (p >= 0 ? hist[uint64_t(p % hist_rows) * C + c] : 0.f);
                conv += w[uint64_t(c) * 4 + j] * xv;
            }
            y[i] = i2r((conv + x[i]) / div, round);
        });
    });
    return i2_store_rows(q, x, T, C, hist, pos0, hist_rows);
}

sycl::event i2_sconv_dynamic(sycl::queue& q, const float* x, uint32_t T, uint32_t C, const float* dyn,
                             const float* static_w, uint32_t heads, float* hist, uint32_t hist_rows, uint32_t pos0,
                             float* y, bool round) {
    const uint32_t per_head = C / heads;
    ie::ps(q, "i2_sconv_dynamic", [&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(uint64_t(T) * C), [=](sycl::id<1> id) {
            const uint64_t i = id[0];
            const int t = int(i / C);
            const uint32_t c = uint32_t(i % C);
            const float* d = dyn + uint64_t(t) * heads * 4 + (c / per_head) * 4;
            float conv = 0.f;
            for (int lag = 0; lag < 4; ++lag) {                         // lag 0 = current token
                const int s = t - lag;
                const int64_t p = int64_t(pos0) + s;
                const float xv = s >= 0 ? x[uint64_t(s) * C + c] : (p >= 0 ? hist[uint64_t(p % hist_rows) * C + c] : 0.f);
                conv += (static_w[uint64_t(lag) * C + c] + d[lag]) * xv;
            }
            y[i] = i2r((conv + x[i]) / 3.f, round);                    // sqrt(2K + 1) = 3
        });
    });
    return i2_store_rows(q, x, T, C, hist, pos0, hist_rows);
}

sycl::event i2_attn_res(sycl::queue& q, const I2Sources& src, uint32_t n, uint32_t T, uint32_t H, const float* Q,
                        uint32_t nq, float eps, float* out, bool fast, float* scratch, bool rowwise) {
    constexpr uint32_t WG = 256;
    const float scale = float(1.0 / std::sqrt(double(H)));
    static const bool one_wg = std::getenv("IE_INKLING2_ATTNRES_ONE_WG") != nullptr;       // A/B switch
    if (fast && scratch && !one_wg && n <= 8 && nq <= 4 && (T <= 16 || rowwise)) {
        // Few tokens: the H columns are cut into NB slices. Kernel 1: one work-group per (token, slice) writes the
        // slice's sums of squares and raw dot products; kernel 2: one work-group per (token, slice) adds the NB
        // partials in slice order, forms the softmax weights and writes its slice of the output.
        constexpr uint32_t MS = 8, MQ = 4, NV = MS * (1 + MQ), NB = 16, W2 = 64;
        const uint32_t span = (H + NB - 1) / NB;
        ie::ps(q, "i2_attn_res_part", [&](sycl::handler& h) {
            sycl::local_accessor<float, 1> red(sycl::range<1>(NV * (W2 / 16)), h);
            h.parallel_for(sycl::nd_range<1>(uint64_t(T) * NB * W2, W2), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                const uint64_t t = it.get_group(0) / NB;
                const uint32_t blk = uint32_t(it.get_group(0) % NB);
                const uint32_t l = uint32_t(it.get_local_id(0));
                const auto sg = it.get_sub_group();
                const uint32_t sgid = uint32_t(sg.get_group_linear_id());
                const uint32_t i0 = blk * span, i1 = sycl::min(H, i0 + span);
                float ss[MS], dt[MQ][MS];
#pragma unroll
                for (uint32_t s2 = 0; s2 < MS; ++s2) {
                    ss[s2] = 0.f;
#pragma unroll
                    for (uint32_t qi = 0; qi < MQ; ++qi) dt[qi][s2] = 0.f;
                }
                for (uint32_t i = i0 + l; i < i1; i += W2) {
                    float qv[MQ];
#pragma unroll
                    for (uint32_t qi = 0; qi < MQ; ++qi) qv[qi] = qi < nq ? Q[uint64_t(qi) * H + i] : 0.f;
#pragma unroll
                    for (uint32_t s2 = 0; s2 < MS; ++s2) {
                        if (s2 < n) {
                            const float xs = src.p[s2][t * H + i];
                            ss[s2] += xs * xs;
#pragma unroll
                            for (uint32_t qi = 0; qi < MQ; ++qi) dt[qi][s2] += qv[qi] * xs;
                        }
                    }
                }
#pragma unroll
                for (uint32_t s2 = 0; s2 < MS; ++s2) {
                    if (s2 < n) {
                        const float r = sycl::reduce_over_group(sg, ss[s2], sycl::plus<float>());
                        if (sg.get_local_linear_id() == 0) red[s2 * (W2 / 16) + sgid] = r;
#pragma unroll
                        for (uint32_t qi = 0; qi < MQ; ++qi) {
                            if (qi < nq) {
                                const float r2 = sycl::reduce_over_group(sg, dt[qi][s2], sycl::plus<float>());
                                if (sg.get_local_linear_id() == 0) red[(MS + qi * MS + s2) * (W2 / 16) + sgid] = r2;
                            }
                        }
                    }
                }
                sycl::group_barrier(it.get_group());
                if (l < NV) {
                    float r = 0.f;
                    for (uint32_t j = 0; j < W2 / 16; ++j) r += red[l * (W2 / 16) + j];
                    scratch[(t * NB + blk) * NV + l] = r;
                }
            });
        });
        return ie::ps(q, "i2_attn_res_out", [&](sycl::handler& h) {
            h.parallel_for(sycl::nd_range<1>(uint64_t(T) * NB * W2, W2), [=](sycl::nd_item<1> it) {
                const uint64_t t = it.get_group(0) / NB;
                const uint32_t blk = uint32_t(it.get_group(0) % NB);
                const uint32_t l = uint32_t(it.get_local_id(0));
                float fin[NV];
#pragma unroll
                for (uint32_t v = 0; v < NV; ++v) fin[v] = 0.f;
                for (uint32_t b2 = 0; b2 < NB; ++b2)
#pragma unroll
                    for (uint32_t v = 0; v < NV; ++v) fin[v] += scratch[(t * NB + b2) * NV + v];
                float inv[MS], a[MQ][MS];
#pragma unroll
                for (uint32_t s2 = 0; s2 < MS; ++s2) inv[s2] = s2 < n ? 1.f / sycl::sqrt(fin[s2] / float(H) + eps) : 0.f;
#pragma unroll
                for (uint32_t qi = 0; qi < MQ; ++qi) {
                    float mx = -3.0e38f;
#pragma unroll
                    for (uint32_t s2 = 0; s2 < MS; ++s2) {
                        a[qi][s2] = s2 < n ? fin[MS + qi * MS + s2] * inv[s2] * scale : -3.0e38f;
                        mx = sycl::fmax(mx, a[qi][s2]);
                    }
                    float sum = 0.f;
#pragma unroll
                    for (uint32_t s2 = 0; s2 < MS; ++s2) { a[qi][s2] = s2 < n ? sycl::exp(a[qi][s2] - mx) : 0.f; sum += a[qi][s2]; }
#pragma unroll
                    for (uint32_t s2 = 0; s2 < MS; ++s2) a[qi][s2] = a[qi][s2] / sum * inv[s2];
                }
                const uint32_t i0 = blk * span, i1 = sycl::min(H, i0 + span);
                for (uint32_t i = i0 + l; i < i1; i += W2) {
                    float xs[MS];
#pragma unroll
                    for (uint32_t s2 = 0; s2 < MS; ++s2) xs[s2] = s2 < n ? src.p[s2][t * H + i] : 0.f;
#pragma unroll
                    for (uint32_t qi = 0; qi < MQ; ++qi) {
                        if (qi < nq) {
                            float o = 0.f;
#pragma unroll
                            for (uint32_t s2 = 0; s2 < MS; ++s2) if (s2 < n) o += a[qi][s2] * xs[s2];
                            out[(uint64_t(qi) * T + t) * H + i] = o;
                        }
                    }
                }
            });
        });
    }
    if (fast && n <= 8 && nq <= 4) {
        // One sweep: each lane accumulates every source's sum of squares and every (query, source) raw dot product,
        // one batched reduction through local memory, then dot * inv (inv taken out of the sum: fp32 order only).
        // Loops run to the compile-time maxima (8 sources, 4 queries) under guards so the arrays stay in registers.
        constexpr uint32_t MS = 8, MQ = 4, NV = MS * (1 + MQ), NSG = WG / 16;
        return ie::ps(q, "i2_attn_res_fast", [&](sycl::handler& h) {
            sycl::local_accessor<float, 1> red(sycl::range<1>(NV * NSG), h);
            sycl::local_accessor<float, 1> fin(sycl::range<1>(NV), h);
            h.parallel_for(sycl::nd_range<1>(uint64_t(T) * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                const uint64_t t = it.get_group(0);
                const uint32_t l = uint32_t(it.get_local_id(0));
                const auto sg = it.get_sub_group();
                const uint32_t sgid = uint32_t(sg.get_group_linear_id());
                float ss[MS], dt[MQ][MS];
#pragma unroll
                for (uint32_t s2 = 0; s2 < MS; ++s2) {
                    ss[s2] = 0.f;
#pragma unroll
                    for (uint32_t qi = 0; qi < MQ; ++qi) dt[qi][s2] = 0.f;
                }
                for (uint32_t i = l; i < H; i += WG) {
                    float qv[MQ];
#pragma unroll
                    for (uint32_t qi = 0; qi < MQ; ++qi) qv[qi] = qi < nq ? Q[uint64_t(qi) * H + i] : 0.f;
#pragma unroll
                    for (uint32_t s2 = 0; s2 < MS; ++s2) {
                        if (s2 < n) {
                            const float xs = src.p[s2][t * H + i];
                            ss[s2] += xs * xs;
#pragma unroll
                            for (uint32_t qi = 0; qi < MQ; ++qi) dt[qi][s2] += qv[qi] * xs;
                        }
                    }
                }
#pragma unroll
                for (uint32_t s2 = 0; s2 < MS; ++s2) {
                    if (s2 < n) {
                        const float r = sycl::reduce_over_group(sg, ss[s2], sycl::plus<float>());
                        if (sg.get_local_linear_id() == 0) red[s2 * NSG + sgid] = r;
#pragma unroll
                        for (uint32_t qi = 0; qi < MQ; ++qi) {
                            if (qi < nq) {
                                const float r2 = sycl::reduce_over_group(sg, dt[qi][s2], sycl::plus<float>());
                                if (sg.get_local_linear_id() == 0) red[(MS + qi * MS + s2) * NSG + sgid] = r2;
                            }
                        }
                    }
                }
                sycl::group_barrier(it.get_group());
                if (l < NV) {
                    float r = 0.f;
                    for (uint32_t j = 0; j < NSG; ++j) r += red[l * NSG + j];
                    fin[l] = r;
                }
                sycl::group_barrier(it.get_group());
                float inv[MS], a[MQ][MS];
#pragma unroll
                for (uint32_t s2 = 0; s2 < MS; ++s2) inv[s2] = s2 < n ? 1.f / sycl::sqrt(fin[s2] / float(H) + eps) : 0.f;
#pragma unroll
                for (uint32_t qi = 0; qi < MQ; ++qi) {
                    float mx = -3.0e38f;
#pragma unroll
                    for (uint32_t s2 = 0; s2 < MS; ++s2) {
                        a[qi][s2] = s2 < n ? fin[MS + qi * MS + s2] * inv[s2] * scale : -3.0e38f;
                        mx = sycl::fmax(mx, a[qi][s2]);
                    }
                    float sum = 0.f;
#pragma unroll
                    for (uint32_t s2 = 0; s2 < MS; ++s2) { a[qi][s2] = s2 < n ? sycl::exp(a[qi][s2] - mx) : 0.f; sum += a[qi][s2]; }
#pragma unroll
                    for (uint32_t s2 = 0; s2 < MS; ++s2) a[qi][s2] = a[qi][s2] / sum * inv[s2];
                }
                for (uint32_t i = l; i < H; i += WG) {
                    float xs[MS];
#pragma unroll
                    for (uint32_t s2 = 0; s2 < MS; ++s2) xs[s2] = s2 < n ? src.p[s2][t * H + i] : 0.f;
#pragma unroll
                    for (uint32_t qi = 0; qi < MQ; ++qi) {
                        if (qi < nq) {
                            float o = 0.f;
#pragma unroll
                            for (uint32_t s2 = 0; s2 < MS; ++s2) if (s2 < n) o += a[qi][s2] * xs[s2];
                            out[(uint64_t(qi) * T + t) * H + i] = o;
                        }
                    }
                }
            });
        });
    }
    return ie::ps(q, "i2_attn_res", [&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(uint64_t(T) * WG, WG), [=](sycl::nd_item<1> it) {
            const uint64_t t = it.get_group(0);
            const uint32_t l = uint32_t(it.get_local_id(0));
            float inv[8], a[4][8];
            for (uint32_t s = 0; s < n; ++s) {
                const float* xs = src.p[s] + t * H;
                float ss = 0.f;
                for (uint32_t i = l; i < H; i += WG) ss += xs[i] * xs[i];
                ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
                inv[s] = 1.f / sycl::sqrt(ss / float(H) + eps);
            }
            for (uint32_t qi = 0; qi < nq; ++qi)
                for (uint32_t s = 0; s < n; ++s) {
                    const float* xs = src.p[s] + t * H;
                    const float* qq = Q + uint64_t(qi) * H;
                    float d = 0.f;
                    for (uint32_t i = l; i < H; i += WG) d += qq[i] * (xs[i] * inv[s]);
                    a[qi][s] = sycl::reduce_over_group(it.get_group(), d, sycl::plus<float>()) * scale;
                }
            for (uint32_t qi = 0; qi < nq; ++qi) {                      // softmax over the n sources (fp32, no mask)
                float mx = a[qi][0];
                for (uint32_t s = 1; s < n; ++s) mx = sycl::fmax(mx, a[qi][s]);
                float sum = 0.f;
                for (uint32_t s = 0; s < n; ++s) { a[qi][s] = sycl::exp(a[qi][s] - mx); sum += a[qi][s]; }
                for (uint32_t s = 0; s < n; ++s) a[qi][s] /= sum;
            }
            for (uint32_t i = l; i < H; i += WG)
                for (uint32_t qi = 0; qi < nq; ++qi) {
                    float o = 0.f;
                    for (uint32_t s = 0; s < n; ++s) o += a[qi][s] * (src.p[s][t * H + i] * inv[s]);
                    out[(uint64_t(qi) * T + t) * H + i] = o;
                }
        });
    });
}

sycl::event i2_gate_combine(sycl::queue& q, const float* hh, const float* m, const float* f, const float* g, uint64_t n,
                            float* out) {
    return ie::ps(q, "i2_gate_combine", [&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
            const uint64_t i = id[0];
            out[i] = (1.f - sycl::tanh(f[i])) * hh[i] + sycl::tanh(g[i]) * m[i];
        });
    });
}

sycl::event i2_mul_sigmoid(sycl::queue& q, const float* x, const float* gate, uint64_t n, float* y, bool round) {
    return ie::ps(q, "i2_mul_sigmoid", [&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
            const uint64_t i = id[0];
            y[i] = i2r(x[i] * (1.f / (1.f + sycl::exp(-gate[i]))), round);
        });
    });
}

sycl::event i2_add(sycl::queue& q, const float* a, const float* b, uint64_t n, float* y, bool round) {
    return ie::ps(q, "i2_add", [&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) { y[id[0]] = i2r(a[id[0]] + b[id[0]], round); });
    });
}

sycl::event i2_ablate_add(sycl::queue& q, const float* a, const float* x, uint32_t rows, uint32_t dim, const float* dirs,
                          uint32_t k, float strength, float* y, bool keep_norm) {
    constexpr uint32_t WG = 256;
    const float s2 = 2.f * strength - strength * strength;             // |x'|^2 = |x|^2 - s2 * sum_j (x . r_j)^2
    if (k <= 4 && dim <= 2560) {                                      // the usual case: coefficients in registers
        return ie::ps(q, "i2_ablate_add_k4", [&](sycl::handler& h) {
            h.parallel_for(sycl::nd_range<1>(uint64_t(rows) * WG, WG), [=](sycl::nd_item<1> it) {
                const uint64_t r = it.get_group(0);
                const uint32_t l = uint32_t(it.get_local_id(0));
                const float* xr = x + r * dim;
                float xv[(2304 + WG - 1) / WG + 1];                     // this item's elements (dim <= 2560)
                float c[4] = {0.f, 0.f, 0.f, 0.f};
                const uint32_t n = (dim - l + WG - 1) / WG;
                for (uint32_t e = 0; e < n; ++e) xv[e] = xr[l + e * WG];
                float dd = 0.f;
                for (uint32_t j = 0; j < k; ++j) {
                    const float* d = dirs + uint64_t(j) * dim;
                    float s = 0.f;
                    for (uint32_t e = 0; e < n; ++e) s += xv[e] * d[l + e * WG];
                    const float dj = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());   // to every item
                    c[j] = strength * dj;
                    dd += dj * dj;
                }
                float scale = 1.f;
                if (keep_norm) {                                        // the write keeps its length (experiment)
                    float ss = 0.f;
                    for (uint32_t e = 0; e < n; ++e) ss += xv[e] * xv[e];
                    ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
                    scale = sycl::sqrt(ss / sycl::fmax(ss - s2 * dd, 1e-12f * ss + 1e-30f));
                }
                float* yr = y + r * dim;
                const float* ar = a ? a + r * dim : nullptr;
                for (uint32_t e = 0; e < n; ++e) {
                    const uint32_t i = l + e * WG;
                    float v = xv[e];
                    for (uint32_t j = 0; j < k; ++j) v -= c[j] * dirs[uint64_t(j) * dim + i];
                    if (keep_norm) v *= scale;
                    yr[i] = ar ? ar[i] + v : v;
                }
            });
        });
    }
    return ie::ps(q, "i2_ablate_add", [&](sycl::handler& h) {
        sycl::local_accessor<float, 1> coef(sycl::range<1>(kI2AblateMaxK + 1), h);
        h.parallel_for(sycl::nd_range<1>(uint64_t(rows) * WG, WG), [=](sycl::nd_item<1> it) {
            const uint64_t r = it.get_group(0);
            const uint32_t l = uint32_t(it.get_local_id(0));
            const float* xr = x + r * dim;
            float dd = 0.f;
            for (uint32_t j = 0; j < k; ++j) {                              // the k projections (each a group reduction)
                const float* d = dirs + uint64_t(j) * dim;
                float s = 0.f;
                for (uint32_t i = l; i < dim; i += WG) s += xr[i] * d[i];
                s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
                if (l == 0) coef[j] = strength * s;
                dd += s * s;
            }
            float scale = 1.f;
            if (keep_norm) {
                float ss = 0.f;
                for (uint32_t i = l; i < dim; i += WG) ss += xr[i] * xr[i];
                ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
                scale = sycl::sqrt(ss / sycl::fmax(ss - s2 * dd, 1e-12f * ss + 1e-30f));
            }
            sycl::group_barrier(it.get_group());
            float* yr = y + r * dim;
            const float* ar = a ? a + r * dim : nullptr;
            for (uint32_t i = l; i < dim; i += WG) {                        // each item writes only what it read itself
                float v = xr[i];
                for (uint32_t j = 0; j < k; ++j) v -= coef[j] * dirs[uint64_t(j) * dim + i];
                if (keep_norm) v *= scale;
                yr[i] = ar ? ar[i] + v : v;
            }
        });
    });
}

sycl::event i2_attn_prep(sycl::queue& q, const I2AttnPrep& p) {
    // One work-group of 128 per (token, unit): units 0..NH-1 = query head h (+ its relative-state head), then NKV key
    // heads, then NKV value heads. Each unit does what the separate kernels did for its head: scale + L2 clip (+ tau),
    // or the causal conv of its channels (static or dynamic) + the history store + clip (keys) / norm (values) + the
    // cache store. Only the reductions' order differs from the separate kernels (fp32 path only).
    constexpr uint32_t WG = 128;
    const uint32_t units = p.nh + 2 * p.nkv;
    const float div_s = std::sqrt(5.0f);
    return ie::ps(q, "i2_attn_prep", [&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(uint64_t(p.T) * units * WG, WG), [=](sycl::nd_item<1> it) {
            const uint32_t t = uint32_t(it.get_group(0) / units), u = uint32_t(it.get_group(0) % units);
            const uint32_t l = uint32_t(it.get_local_id(0));
            const uint32_t HD = p.hd, KD = p.nkv * HD;
            const bool rw = p.rows.pos != nullptr;                      // batched: row t = one token of its own slot
            const int tl = rw ? 0 : int(t);                             // t within its conversation's chunk
            const int64_t P0 = rw ? int64_t(p.rows.pos[t]) : int64_t(p.pos0);
            const uint64_t sl = rw ? uint64_t(p.rows.slot[t]) : 0;
            const bool act = l < HD;
            if (u < p.nh) {
                float* qr = p.qb + (uint64_t(t) * p.nh + u) * HD;
                const float v = act ? qr[l] * p.q_scale[uint64_t(u) * HD + l] : 0.f;
                const float ss = sycl::reduce_over_group(it.get_group(), v * v, sycl::plus<float>());
                const float f = sycl::fmin(p.qk_clip / (sycl::sqrt(ss) + 1e-6f), 1.f);
                if (act) qr[l] = p.glob ? (v * f) * p.tau[t] : v * f;
                float* rr = p.rb + (uint64_t(t) * p.nh + u) * p.d_rel;
                const bool ra = l < p.d_rel;
                const float rv = ra ? rr[l] * p.rel_scale[uint64_t(u) * p.d_rel + l] : 0.f;
                const float rs = sycl::reduce_over_group(it.get_group(), rv * rv, sycl::plus<float>());
                const float g = sycl::fmin(p.rel_clip / (sycl::sqrt(rs) + 1e-6f), 1.f);
                if (ra) rr[l] = p.glob ? (rv * g) * p.tau[t] : rv * g;
                return;
            }
            const bool isk = u < p.nh + p.nkv;
            const uint32_t kh = isk ? u - p.nh : u - p.nh - p.nkv;
            const float* x = (isk ? p.kp : p.vp) + uint64_t(int64_t(t) - tl) * KD;   // row 0 of this token's chunk
            float* hist = (isk ? p.ck : p.cv) + sl * p.c_stride;
            const uint32_t c = kh * HD + (act ? l : 0);
            const int64_t P = P0 + tl;
            float y = 0.f;
            if (act) {
                const float xc = x[uint64_t(tl) * KD + c];
                auto xv = [&](int sidx) -> float {                       // sidx relative to the chunk
                    const int64_t pp = P0 + sidx;
                    return sidx >= 0 ? x[uint64_t(sidx) * KD + c] : (pp >= 0 ? hist[uint64_t(pp % p.crow) * KD + c] : 0.f);
                };
                float conv = 0.f;
                if (p.glob) {
                    const float* d = (isk ? p.dyn_k : p.dyn_v) + uint64_t(t) * p.nkv * 4 + kh * 4;
                    const float* sw = isk ? p.k_conv : p.v_conv;
                    for (int lag = 0; lag < 4; ++lag) conv += (sw[uint64_t(lag) * KD + c] + d[lag]) * xv(tl - lag);
                    y = (conv + xc) / 3.f;
                } else {
                    const float* w = isk ? p.k_conv : p.v_conv;
                    for (int j = 0; j < 4; ++j) conv += w[uint64_t(c) * 4 + j] * xv(tl - 3 + j);
                    y = (conv + xc) / div_s;
                }
                hist[uint64_t(P % p.crow) * KD + c] = xc;
            }
            const float ss = sycl::reduce_over_group(it.get_group(), y * y, sycl::plus<float>());
            const float f = isk ? sycl::fmin(p.qk_clip / (sycl::sqrt(ss) + 1e-6f), 1.f) : 1.f / sycl::sqrt(ss / float(HD) + p.eps);
            if (act) {
                const uint64_t row = p.ring ? uint64_t(P) % p.ring : uint64_t(P);
                const uint64_t at = sl * p.kv_stride + row * KD + c;
                if (p.kv16)                                             // bf16 cache (round to nearest even)
                    reinterpret_cast<uint16_t*>(isk ? p.Kc : p.Vc)[at] = uint16_t(sycl::bit_cast<uint32_t>(i2_bf16_round(y * f)) >> 16);
                else (isk ? p.Kc : p.Vc)[at] = y * f;
            }
        });
    });
}

sycl::event i2_conv_add(sycl::queue& q, const float* x, uint32_t T, uint32_t C, const float* w, float* hist,
                        uint32_t hist_rows, uint32_t pos0, float* y, const float* a, float* res, I2Rows rows,
                        uint64_t h_stride, const float* addv) {
    const float div = std::sqrt(5.0f);
    return ie::ps(q, "i2_conv_add", [&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(uint64_t(T) * C), [=](sycl::id<1> id) {
            const uint64_t i = id[0];
            const int t = int(i / C);
            const uint32_t c = uint32_t(i % C);
            const bool rw = rows.pos != nullptr;
            const int tl = rw ? 0 : t;
            const int64_t P0 = rw ? int64_t(rows.pos[t]) : int64_t(pos0);
            float* hb = hist + (rw ? uint64_t(rows.slot[t]) * h_stride : 0);
            float conv = 0.f;
            for (int j = 0; j < 4; ++j) {
                const int s2 = tl - 3 + j;
                const int64_t p = P0 + s2;
                const float xv = s2 >= 0 ? x[uint64_t(t - tl + s2) * C + c] : (p >= 0 ? hb[uint64_t(p % hist_rows) * C + c] : 0.f);
                conv += w[uint64_t(c) * 4 + j] * xv;
            }
            const float v = (conv + x[i]) / div;
            hb[uint64_t((P0 + tl) % hist_rows) * C + c] = x[i];
            y[i] = v;
            const float wv = addv ? v + addv[c] : v;
            if (res) res[i] = a ? a[i] + wv : wv;
        });
    });
}

sycl::event i2_add_rows(sycl::queue& q, const float* x, uint32_t T, uint32_t H, const float* v, float* y) {
    return ie::ps(q, "i2_add_rows", [&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(uint64_t(T) * H), [=](sycl::id<1> id) { y[id[0]] = x[id[0]] + v[id[0] % H]; });
    });
}

sycl::event i2_moe_out(sycl::queue& q, const float* ey, const float* wts, uint32_t T, uint32_t k, uint32_t H,
                       const float* w, float* hist, uint32_t hist_rows, uint32_t pos0, float* y, const float* a, float* out,
                       I2Rows rows, uint64_t h_stride, const float* addv) {
    const float div = std::sqrt(5.0f);
    return ie::ps(q, "i2_moe_out", [&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(uint64_t(T) * H), [=](sycl::id<1> id) {
            const uint64_t i = id[0];
            const int t = int(i / H);
            const uint32_t c = uint32_t(i % H);
            const bool rw = rows.pos != nullptr;
            const int tl = rw ? 0 : t;
            const int64_t P0 = rw ? int64_t(rows.pos[t]) : int64_t(pos0);
            float* hb = hist + (rw ? uint64_t(rows.slot[t]) * h_stride : 0);
            auto comb = [&](int tt) {                                   // the expert combine of token tt, channel c
                float acc = 0.f;
                for (uint32_t j = 0; j < k; ++j) acc += wts[uint64_t(tt) * k + j] * ey[(uint64_t(tt) * k + j) * H + c];
                return acc;
            };
            const float xc = comb(t);
            float conv = 0.f;
            for (int j = 0; j < 4; ++j) {
                const int s2 = tl - 3 + j;
                const int64_t p = P0 + s2;
                const float xv = s2 == tl ? xc : s2 >= 0 ? comb(t - tl + s2) : (p >= 0 ? hb[uint64_t(p % hist_rows) * H + c] : 0.f);
                conv += w[uint64_t(c) * 4 + j] * xv;
            }
            const float v = (conv + xc) / div;
            hb[uint64_t((P0 + tl) % hist_rows) * H + c] = xc;
            y[i] = v;
            out[i] = a[i] + (addv ? v + addv[c] : v);
        });
    });
}

sycl::event i2_write_ablate(sycl::queue& q, const I2WriteAbl& p) {
    // One work-group of 256 per (token, 256-column block). Each forms its block of the write exactly as i2_conv_add /
    // i2_moe_out do (history store included), keeps it in y, and leaves its partial dot products with the directions
    // (and |x|^2) in p.part; the last block of a token to finish (a device-scope ticket in p.cnt, which it resets)
    // adds the partials in block order, then removes the directions from the whole token row and adds the residual --
    // the removal of i2_ablate_add with its reductions folded into the write's own launch.
    constexpr uint32_t WG = 256, MK = 4;
    const uint32_t H = p.H, NB = (H + WG - 1) / WG;
    const float div = std::sqrt(5.0f);
    const float s2c = 2.f * p.strength - p.strength * p.strength;
    return ie::ps(q, "i2_write_ablate", [&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> last(sycl::range<1>(1), h);
        h.parallel_for(sycl::nd_range<1>(uint64_t(p.T) * NB * WG, WG), [=](sycl::nd_item<1> it) {
            const uint32_t t = uint32_t(it.get_group(0) / NB), blk = uint32_t(it.get_group(0) % NB);
            const uint32_t l = uint32_t(it.get_local_id(0));
            const uint32_t c = blk * WG + l;
            const bool act = c < H;
            const bool rw = p.rows.pos != nullptr;
            const int tl = rw ? 0 : int(t);
            const int64_t P0 = rw ? int64_t(p.rows.pos[t]) : int64_t(p.pos0);
            float* hb = p.hist + (rw ? uint64_t(p.rows.slot[t]) * p.h_stride : 0);
            float v = 0.f;
            if (act) {
                auto comb = [&](int tt) {
                    float acc = 0.f;
                    for (uint32_t j = 0; j < p.k_top; ++j) acc += p.wts[uint64_t(tt) * p.k_top + j] * p.x[(uint64_t(tt) * p.k_top + j) * H + c];
                    return acc;
                };
                const float xc = p.moe ? comb(int(t)) : p.x[uint64_t(t) * H + c];
                float conv = 0.f;
                for (int j = 0; j < 4; ++j) {
                    const int s2 = tl - 3 + j;
                    const int64_t pp = P0 + s2;
                    const int gr = int(t) - tl + s2;                    // the batch row of that tap
                    const float xv = s2 == tl ? xc
                                   : s2 >= 0 ? (p.moe ? comb(gr) : p.x[uint64_t(gr) * H + c])
                                   : (pp >= 0 ? hb[uint64_t(pp % p.hist_rows) * H + c] : 0.f);
                    conv += p.w[uint64_t(c) * 4 + j] * xv;
                }
                v = (conv + xc) / div;
                hb[uint64_t((P0 + tl) % p.hist_rows) * H + c] = xc;
                p.y[uint64_t(t) * H + c] = v;
            }
            float* pt = p.part + (uint64_t(t) * NB + blk) * (MK + 1);
            for (uint32_t j = 0; j < p.kd; ++j) {
                const float r = sycl::reduce_over_group(it.get_group(), act ? v * p.dirs[uint64_t(j) * H + c] : 0.f, sycl::plus<float>());
                if (l == 0) pt[j] = r;
            }
            if (p.keep_norm) {
                const float r = sycl::reduce_over_group(it.get_group(), v * v, sycl::plus<float>());
                if (l == 0) pt[MK] = r;
            }
            sycl::group_barrier(it.get_group());
            if (l == 0) {
                sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::device);
                const uint32_t ticket = sycl::atomic_ref<uint32_t, sycl::memory_order::acq_rel, sycl::memory_scope::device,
                                                         sycl::access::address_space::global_space>(p.cnt[t]).fetch_add(1u);
                last[0] = ticket == NB - 1 ? 1u : 0u;
            }
            sycl::group_barrier(it.get_group());
            if (!last[0]) return;
            sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::device);
            float cf[MK] = {0.f, 0.f, 0.f, 0.f}, dd = 0.f, ss = 0.f;
            for (uint32_t j = 0; j < p.kd; ++j) {
                float dj = 0.f;
                for (uint32_t b2 = 0; b2 < NB; ++b2) dj += p.part[(uint64_t(t) * NB + b2) * (MK + 1) + j];
                cf[j] = p.strength * dj;
                dd += dj * dj;
            }
            float scale = 1.f;
            if (p.keep_norm) {
                for (uint32_t b2 = 0; b2 < NB; ++b2) ss += p.part[(uint64_t(t) * NB + b2) * (MK + 1) + MK];
                scale = sycl::sqrt(ss / sycl::fmax(ss - s2c * dd, 1e-12f * ss + 1e-30f));
            }
            for (uint32_t i = l; i < H; i += WG) {
                float vv = p.y[uint64_t(t) * H + i];
                for (uint32_t j = 0; j < p.kd; ++j) vv -= cf[j] * p.dirs[uint64_t(j) * H + i];
                if (p.keep_norm) vv *= scale;
                if (p.addv) vv += p.addv[i];
                p.res[uint64_t(t) * H + i] = p.a ? p.a[uint64_t(t) * H + i] + vv : vv;
            }
            if (l == 0) p.cnt[t] = 0u;
        });
    });
}

sycl::event i2_gate_combine_multi(sycl::queue& q, uint32_t n, const I2GateSeg* seg, uint64_t len) {
    struct P { const float* h[4]; const float* m[4]; const float* f[4]; const float* g[4]; float* o[4]; } pk{};
    for (uint32_t i = 0; i < n && i < 4; ++i) { pk.h[i] = seg[i].h; pk.m[i] = seg[i].m; pk.f[i] = seg[i].f; pk.g[i] = seg[i].g; pk.o[i] = seg[i].out; }
    return ie::ps(q, "i2_gate_combine", [&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(uint64_t(n) * len), [=](sycl::id<1> id) {
            const uint32_t sgi = uint32_t(id[0] / len);
            const uint64_t i = id[0] % len;
            pk.o[sgi][i] = (1.f - sycl::tanh(pk.f[sgi][i])) * pk.h[sgi][i] + sycl::tanh(pk.g[sgi][i]) * pk.m[sgi][i];
        });
    });
}

sycl::event i2_gate_down_norm(sycl::queue& q, uint32_t n, const I2GateDownSeg* seg, uint32_t T, uint32_t H, uint32_t R, float eps) {
    constexpr uint32_t SG = 16, NSG = 8, WG = SG * NSG, TT = 4;
    struct P { const float* h[4]; const float* m[4]; const uint16_t* W[4]; float* gd[4]; } pk{};
    for (uint32_t i = 0; i < n && i < 4; ++i) { pk.h[i] = seg[i].h; pk.m[i] = seg[i].m; pk.W[i] = seg[i].W; pk.gd[i] = seg[i].gd; }
    const uint32_t tiles = (T + TT - 1) / TT, K = 2 * H;
    return ie::ps(q, "i2_gate_down_norm", [&](sycl::handler& h) {
        sycl::local_accessor<float, 1> red(sycl::range<1>(NSG * TT * 2), h);
        h.parallel_for(sycl::nd_range<1>(uint64_t(n) * R * tiles * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const auto sg = it.get_sub_group();
            const uint64_t g = it.get_group(0);
            const uint32_t tile = uint32_t(g % tiles), row = uint32_t((g / tiles) % R), si = uint32_t(g / (uint64_t(tiles) * R));
            const uint32_t t0 = tile * TT, Tn = sycl::min(TT, T - t0);
            const uint32_t l = uint32_t(sg.get_local_linear_id()), sgid = uint32_t(sg.get_group_linear_id());
            const uint32_t lt = uint32_t(it.get_local_id(0));
            const float* hb = pk.h[si] + uint64_t(t0) * H;
            const float* mb = pk.m[si] + uint64_t(t0) * H;
            // the rms of each token's h and m rows (the whole work-group), then the row's dot over [h * ih, m * im]
            float ih[TT], im[TT];
#pragma unroll
            for (uint32_t t = 0; t < TT; ++t) {
                float a = 0.f, b = 0.f;
                if (t < Tn)
                    for (uint32_t i = lt; i < H; i += WG) { const float x = hb[uint64_t(t) * H + i], y = mb[uint64_t(t) * H + i]; a += x * x; b += y * y; }
                a = sycl::reduce_over_group(it.get_group(), a, sycl::plus<float>());
                b = sycl::reduce_over_group(it.get_group(), b, sycl::plus<float>());
                ih[t] = 1.f / sycl::sqrt(a / float(H) + eps);
                im[t] = 1.f / sycl::sqrt(b / float(H) + eps);
            }
            const sycl::uint4* w = reinterpret_cast<const sycl::uint4*>(pk.W[si] + uint64_t(row) * K);
            float acc[TT];
#pragma unroll
            for (uint32_t t = 0; t < TT; ++t) acc[t] = 0.f;
            for (uint32_t c = sgid * SG + l; c < K / 8; c += NSG * SG) {      // as i2_linear_bf16_rows
                const sycl::uint4 v = w[c];
                float wf[8];
#pragma unroll
                for (int j = 0; j < 4; ++j) { wf[2 * j] = sycl::bit_cast<float>(v[j] << 16); wf[2 * j + 1] = sycl::bit_cast<float>(v[j] & 0xffff0000u); }
                const uint32_t k0 = 8 * c;
                const bool lo = k0 < H;
#pragma unroll
                for (uint32_t t = 0; t < TT; ++t) {
                    if (t >= Tn) break;
                    const float* xr = lo ? hb + uint64_t(t) * H + k0 : mb + uint64_t(t) * H + (k0 - H);
                    const float f = lo ? ih[t] : im[t];
                    const sycl::float4 a = reinterpret_cast<const sycl::float4*>(xr)[0], b = reinterpret_cast<const sycl::float4*>(xr)[1];
                    acc[t] += (a[0] * f) * wf[0] + (a[1] * f) * wf[1] + (a[2] * f) * wf[2] + (a[3] * f) * wf[3] +
                              (b[0] * f) * wf[4] + (b[1] * f) * wf[5] + (b[2] * f) * wf[6] + (b[3] * f) * wf[7];
                }
            }
#pragma unroll
            for (uint32_t t = 0; t < TT; ++t) {
                const float r = sycl::reduce_over_group(sg, t < Tn ? acc[t] : 0.f, sycl::plus<float>());
                if (l == 0) red[sgid * TT + t] = r;
            }
            sycl::group_barrier(it.get_group());
            if (sgid == 0 && l < Tn) {
                float r = 0.f;
                for (uint32_t j = 0; j < NSG; ++j) r += red[j * TT + l];
                pk.gd[si][uint64_t(t0 + l) * R + row] = r;
            }
        });
    });
}

sycl::event i2_gate_up_combine(sycl::queue& q, uint32_t n, const I2GateUpSeg* seg, uint32_t T, uint32_t H, uint32_t R) {
    constexpr uint32_t SG = 16, NSG = 8;
    struct P { const float* gd[4]; const uint16_t* Wf[4]; const uint16_t* Wg[4]; const float* h[4]; const float* m[4]; float* o[4]; } pk{};
    for (uint32_t i = 0; i < n && i < 4; ++i) {
        pk.gd[i] = seg[i].gd; pk.Wf[i] = seg[i].Wf; pk.Wg[i] = seg[i].Wg; pk.h[i] = seg[i].h; pk.m[i] = seg[i].m; pk.o[i] = seg[i].out;
    }
    const uint32_t groups = (H + NSG - 1) / NSG;                        // 8 elements (one per sub-group) per work-group
    return ie::ps(q, "i2_gate_up_combine", [&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(uint64_t(n) * T * groups * NSG * SG, NSG * SG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const auto sg = it.get_sub_group();
            const uint64_t g = it.get_group(0);
            const uint32_t gi = uint32_t(g % groups), t = uint32_t((g / groups) % T), si = uint32_t(g / (uint64_t(groups) * T));
            const uint32_t e = gi * NSG + uint32_t(sg.get_group_linear_id());
            if (e >= H) return;
            const uint32_t l = uint32_t(sg.get_local_linear_id());
            const float* x = pk.gd[si] + uint64_t(t) * R;
            float af = 0.f, ag = 0.f;
            for (uint32_t c = l; c < R / 8; c += SG) {                  // as i2_linear_bf16_sg<1>: chunk order per lane
                const sycl::uint4 vf = reinterpret_cast<const sycl::uint4*>(pk.Wf[si] + uint64_t(e) * R)[c];
                const sycl::uint4 vg = reinterpret_cast<const sycl::uint4*>(pk.Wg[si] + uint64_t(e) * R)[c];
                const sycl::float4 a = reinterpret_cast<const sycl::float4*>(x)[2 * c], b = reinterpret_cast<const sycl::float4*>(x)[2 * c + 1];
                float f8[8], g8[8];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    f8[2 * j] = sycl::bit_cast<float>(vf[j] << 16); f8[2 * j + 1] = sycl::bit_cast<float>(vf[j] & 0xffff0000u);
                    g8[2 * j] = sycl::bit_cast<float>(vg[j] << 16); g8[2 * j + 1] = sycl::bit_cast<float>(vg[j] & 0xffff0000u);
                }
                af += a[0] * f8[0] + a[1] * f8[1] + a[2] * f8[2] + a[3] * f8[3] + b[0] * f8[4] + b[1] * f8[5] + b[2] * f8[6] + b[3] * f8[7];
                ag += a[0] * g8[0] + a[1] * g8[1] + a[2] * g8[2] + a[3] * g8[3] + b[0] * g8[4] + b[1] * g8[5] + b[2] * g8[6] + b[3] * g8[7];
            }
            const float f = sycl::reduce_over_group(sg, af, sycl::plus<float>());
            const float gg = sycl::reduce_over_group(sg, ag, sycl::plus<float>());
            if (l == 0) {
                const uint64_t i = uint64_t(t) * H + e;
                pk.o[si][i] = (1.f - sycl::tanh(f)) * pk.h[si][i] + sycl::tanh(gg) * pk.m[si][i];
            }
        });
    });
}

sycl::event i2_round(sycl::queue& q, float* x, uint64_t n) {
    return ie::ps(q, "i2_round", [&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) { x[id[0]] = i2_bf16_round(x[id[0]]); });
    });
}

sycl::event i2_softcap(sycl::queue& q, float* x, uint64_t n, float cap) {
    return ie::ps(q, "i2_softcap", [&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) { x[id[0]] = cap * sycl::tanh(x[id[0]] / cap); });
    });
}

sycl::event i2_attention(sycl::queue& q, const float* Q, const float* Kc, const float* Vc, uint32_t ring,
                         const float* rel, const float* proj, uint32_t d_rel, uint32_t extent,
                         const float* sink, float* o, uint32_t T, uint32_t pos0, uint32_t nh, uint32_t nkv,
                         uint32_t hd, uint32_t window, bool round) {
    const uint32_t WG = hd;                                             // one work-item per output dimension
    const uint32_t group = nh / nkv;
    const float scale = float(1.0 / std::sqrt(double(hd)));
    return ie::ps(q, "i2_attention", [&](sycl::handler& h) {
        sycl::local_accessor<float, 1> qs(sycl::range<1>(hd), h);      // this query
        sycl::local_accessor<float, 1> rs(sycl::range<1>(d_rel), h);    // its relative state
        sycl::local_accessor<float, 1> tile(sycl::range<1>(WG), h);     // probabilities of one tile of keys
        h.parallel_for(sycl::nd_range<1>(uint64_t(T) * nh * WG, WG), [=](sycl::nd_item<1> it) {
            const uint64_t g = it.get_group(0);
            const uint32_t t = uint32_t(g / nh), hh = uint32_t(g % nh), kvh = hh / group;
            const uint32_t l = uint32_t(it.get_local_id(0));
            const uint32_t P = pos0 + t;                                // this query's absolute position
            const uint32_t jlo = (window > 0 && P + 1 > window) ? P + 1 - window : 0;   // visible: P - j < window
            qs[l] = Q[g * hd + l];
            if (l < d_rel) rs[l] = rel[g * d_rel + l];
            sycl::group_barrier(it.get_group());
            auto score = [&](uint32_t j) {
                const uint64_t row = ring ? j % ring : j;
                const float* kv = Kc + (row * nkv + kvh) * hd;
                float d = 0.f;
                for (uint32_t c = 0; c < hd; ++c) d += qs[c] * kv[c];
                float s = i2r(d, round);
                s = i2r(s * scale, round);
                const uint32_t dist = P - j;
                float b = 0.f;
                if (dist < extent) {
                    for (uint32_t x = 0; x < d_rel; ++x) b += rs[x] * proj[uint64_t(x) * extent + dist];
                    b = i2r(b, round);                                  // the reference rounds relative_states @ proj
                }
                return i2r(s + b, round);
            };
            const float sk = sink[uint64_t(t) * nh + hh];
            float mx = sk;
            for (uint32_t j = jlo + l; j <= P; j += WG) mx = sycl::fmax(mx, score(j));
            mx = sycl::reduce_over_group(it.get_group(), mx, sycl::maximum<float>());
            float den = 0.f;
            for (uint32_t j = jlo + l; j <= P; j += WG) den += sycl::exp(score(j) - mx);
            den = sycl::reduce_over_group(it.get_group(), den, sycl::plus<float>()) + sycl::exp(sk - mx);
            float acc = 0.f;                                            // thread l = output dimension l
            for (uint32_t j0 = jlo; j0 <= P; j0 += WG) {
                const uint32_t j = j0 + l;
                tile[l] = j <= P ? i2r(sycl::exp(score(j) - mx) / den, round) : 0.f;
                sycl::group_barrier(it.get_group());
                const uint32_t n = sycl::min(WG, P - j0 + 1);
                for (uint32_t i = 0; i < n; ++i) {
                    const uint64_t row = ring ? (j0 + i) % ring : (j0 + i);
                    acc += tile[i] * Vc[(row * nkv + kvh) * hd + l];
                }
                sycl::group_barrier(it.get_group());
            }
            o[g * hd + l] = i2r(acc, round);
        });
    });
}

// One query row (decode, batched rows): at most this many splits (IE_INKLING2_ATTN_MAXSPLITS, 0 = no limit): past it the
// merge's serial walk over the splits costs more than the extra work-groups give (32K context: 250 splits of 128 keys).
static uint32_t i2_decode_max_splits() {
    static const uint32_t v = [] { const char* e = std::getenv("IE_INKLING2_ATTN_MAXSPLITS"); return e ? uint32_t(std::atoi(e)) : 16u; }();
    return v ? v : 0xffffffffu;
}

uint32_t i2_attention_splits(uint32_t T, uint32_t pos0, uint32_t nh, uint32_t hd, uint32_t window, uint64_t part_floats,
                             uint32_t min_keys) {
    const uint32_t max_keys = window ? sycl::min(window, pos0 + T) : pos0 + T;   // keys the widest query sees
    const uint32_t want = (max_keys + min_keys - 1) / min_keys;
    uint64_t cap = part_floats / (uint64_t(T) * nh * (hd + 2));
    if (T == 1) cap = std::min<uint64_t>(cap, i2_decode_max_splits());
    return uint32_t(std::max<uint64_t>(1, std::min<uint64_t>(want, cap)));
}

namespace {
// Split attention, tiled (v2): each 16-lane sub-group takes 16 keys at a time. Lane l holds dims 8l..8l+7 of the G query
// heads (two float4 / one uint4 of bf16 per K or V row), forms its partial dot with each of the 16 keys (plus the
// relative bias term of dimension l, as v1), then one butterfly transpose-reduce leaves key i's score in lane i (15
// shuffles per head instead of 16 reductions); one online-softmax update per tile (max over the 16, exp per lane), and
// the V rows weighted by the 16 probabilities. Same partials (max, sum, values per split) as v1; a different fp32 order.
template <uint32_t GG>
void i2_attn_split_v2(sycl::queue& q, const float* Q, const float* Kb, const float* Vb, uint32_t ring, const float* rel,
                      const float* proj, uint32_t d_rel, uint32_t extent, float* part, uint32_t T, uint32_t pos0,
                      uint32_t nh, uint32_t nkv, uint32_t hd, uint32_t window, uint32_t NS, uint32_t cap1, uint32_t mk,
                      I2Rows rows, uint64_t kv_stride, bool kv16, float scale) {
    constexpr uint32_t SG = 16, WG = 128, NSG = WG / SG, GMAX = 4;
    const uint32_t G = nh / nkv;
    ie::ps(q, "i2_attention_split_v2", [&](sycl::handler& h) {
        sycl::local_accessor<float, 1> lm(sycl::range<1>(NSG * GMAX * 2), h);
        sycl::local_accessor<float, 1> la(sycl::range<1>(NSG * GMAX * 128), h);
        h.parallel_for(sycl::nd_range<1>(uint64_t(T) * nkv * NS * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const uint64_t g = it.get_group(0);
            const uint32_t sp = uint32_t(g % NS), kvh = uint32_t((g / NS) % nkv), t = uint32_t(g / (uint64_t(NS) * nkv));
            const auto sgr = it.get_sub_group();
            const uint32_t l = uint32_t(sgr.get_local_linear_id()), sgid = uint32_t(sgr.get_group_linear_id());
            const bool rw = rows.pos != nullptr;
            const uint32_t P = rw ? uint32_t(rows.pos[t]) : pos0 + t;
            uint32_t nsr = NS;
            if (rw) {
                const uint32_t max_keys = window ? sycl::min(window, P + 1) : P + 1;
                nsr = sycl::max(1u, sycl::min((max_keys + mk - 1) / mk, cap1));
            }
            if (sp >= nsr) return;
            const uint64_t so = rw ? uint64_t(rows.slot[t]) * kv_stride : 0;
            const uint32_t jlo = (window > 0 && P + 1 > window) ? P + 1 - window : 0;
            const uint32_t n = P - jlo + 1;
            const uint32_t per = (n + nsr - 1) / nsr;
            const uint32_t a = jlo + sp * per, b = sycl::min(P + 1, a + per);
            const uint32_t d0 = 8 * l;
            const bool dl = d0 < hd;
            float qr[GG][8], acc[GG][8], rr[GG], m[GG], ls[GG];
#pragma unroll
            for (uint32_t gg = 0; gg < GG; ++gg) {
                const uint64_t hq = uint64_t(t) * nh + kvh * G + gg;
                const bool on = gg < G && dl;
#pragma unroll
                for (uint32_t i = 0; i < 8; ++i) { qr[gg][i] = on ? Q[hq * hd + d0 + i] * scale : 0.f; acc[gg][i] = 0.f; }
                rr[gg] = (gg < G && l < d_rel) ? rel[hq * d_rel + l] : 0.f;
                m[gg] = -3.0e38f;
                ls[gg] = 0.f;
            }
            auto row8 = [&](const float* base, uint32_t j, float* o) {        // 8 dims of key j's row (this lane's)
                const uint64_t row = ring ? j % ring : j;
                const uint64_t ro = so + (row * nkv + kvh) * hd + d0;
                if (!dl) { for (int i = 0; i < 8; ++i) o[i] = 0.f; return; }
                if (kv16) {
                    const sycl::uint4 u = *reinterpret_cast<const sycl::uint4*>(reinterpret_cast<const uint16_t*>(base) + ro);
#pragma unroll
                    for (int i = 0; i < 4; ++i) { o[2 * i] = sycl::bit_cast<float>(u[i] << 16); o[2 * i + 1] = sycl::bit_cast<float>(u[i] & 0xFFFF0000u); }
                } else {
                    const sycl::float4 x0 = reinterpret_cast<const sycl::float4*>(base + ro)[0];
                    const sycl::float4 x1 = reinterpret_cast<const sycl::float4*>(base + ro)[1];
                    o[0] = x0[0]; o[1] = x0[1]; o[2] = x0[2]; o[3] = x0[3]; o[4] = x1[0]; o[5] = x1[1]; o[6] = x1[2]; o[7] = x1[3];
                }
            };
            for (uint32_t j0 = a + sgid * SG; j0 < b; j0 += NSG * SG) {
                float d[GG][SG];
#pragma unroll
                for (uint32_t kk = 0; kk < SG; ++kk) {
                    const uint32_t j = sycl::min(j0 + kk, b - 1);
                    float kv[8];
                    row8(Kb, j, kv);
                    const uint32_t dist = P - j;
                    const float pj = (dist < extent && l < d_rel) ? proj[uint64_t(l) * extent + dist] : 0.f;
#pragma unroll
                    for (uint32_t gg = 0; gg < GG; ++gg) {
                        float x = 0.f;
#pragma unroll
                        for (uint32_t i = 0; i < 8; ++i) x += qr[gg][i] * kv[i];
                        d[gg][kk] = x + rr[gg] * pj;
                    }
                }
                // butterfly transpose-reduce: lane l ends with the full score of key j0 + l
#pragma unroll
                for (uint32_t gg = 0; gg < GG; ++gg) {
#pragma unroll
                    for (uint32_t w = 8; w >= 1; w >>= 1) {
                        const bool up = (l & w) != 0;
#pragma unroll
                        for (uint32_t i = 0; i < w; ++i) {
                            const float keep = up ? d[gg][i + w] : d[gg][i];
                            const float send = up ? d[gg][i] : d[gg][i + w];
                            d[gg][i] = keep + sycl::permute_group_by_xor(sgr, send, w);
                        }
                    }
                }
                const bool valid = j0 + l < b;
                float pv[GG];
#pragma unroll
                for (uint32_t gg = 0; gg < GG; ++gg) {
                    const float sc = valid ? d[gg][0] : -3.0e38f;
                    const float tm = sycl::reduce_over_group(sgr, sc, sycl::maximum<float>());
                    const float nm = sycl::fmax(m[gg], tm);
                    const float corr = sycl::exp(m[gg] - nm);
                    pv[gg] = valid ? sycl::exp(sc - nm) : 0.f;
                    ls[gg] = ls[gg] * corr + pv[gg];                     // lane-local sum (reduced at the end)
#pragma unroll
                    for (uint32_t i = 0; i < 8; ++i) acc[gg][i] *= corr;
                    m[gg] = nm;
                }
#pragma unroll
                for (uint32_t kk = 0; kk < SG; ++kk) {
                    if (j0 + kk >= b) break;
                    float vv[8];
                    row8(Vb, j0 + kk, vv);
#pragma unroll
                    for (uint32_t gg = 0; gg < GG; ++gg) {
                        const float p = sycl::select_from_group(sgr, pv[gg], kk);
#pragma unroll
                        for (uint32_t i = 0; i < 8; ++i) acc[gg][i] += p * vv[i];
                    }
                }
            }
#pragma unroll
            for (uint32_t gg = 0; gg < GG; ++gg) {
                if (gg >= G) break;
                const float L = sycl::reduce_over_group(sgr, ls[gg], sycl::plus<float>());
                if (l == 0) { lm[(sgid * GMAX + gg) * 2] = m[gg]; lm[(sgid * GMAX + gg) * 2 + 1] = L; }
                if (dl)
#pragma unroll
                    for (uint32_t i = 0; i < 8; ++i) la[(sgid * GMAX + gg) * 128 + d0 + i] = acc[gg][i];
            }
            sycl::group_barrier(it.get_group());
            const uint32_t lt = uint32_t(it.get_local_id(0));
            for (uint32_t gg = 0; gg < G; ++gg) {
                float M = -3.0e38f;
                for (uint32_t s2 = 0; s2 < NSG; ++s2)
                    if (lm[(s2 * GMAX + gg) * 2 + 1] > 0.f) M = sycl::fmax(M, lm[(s2 * GMAX + gg) * 2]);
                float L = 0.f, A = 0.f;
                for (uint32_t s2 = 0; s2 < NSG; ++s2) {
                    const float l2 = lm[(s2 * GMAX + gg) * 2 + 1];
                    if (!(l2 > 0.f)) continue;
                    const float f = sycl::exp(lm[(s2 * GMAX + gg) * 2] - M);
                    L += l2 * f;
                    if (lt < hd) A += la[(s2 * GMAX + gg) * 128 + lt] * f;
                }
                float* pp = part + ((uint64_t(t) * nh + kvh * G + gg) * NS + sp) * (hd + 2);
                if (lt < hd) pp[2 + lt] = A;
                if (lt == 0) { pp[0] = M; pp[1] = L; }
            }
        });
    });
}

// Query-tiled split attention (a prompt chunk, T >= 16 rows of one conversation): a work-group of NSG sub-groups takes
// NSG consecutive query rows of one KV head, sub-group s row t0 + s with all G query heads (lane l: dims 8l..8l+7, as v2).
// The K / V rows of 16 keys at a time are loaded ONCE into local memory for all NSG rows (v2 reads them per row: at depth
// the chunk's attention is a stream of the same keys 256 times); each row masks its own visible range [jlo, P] (causal,
// window). The keys of the tile's union range are split over NS splits; every (row, head, split) partial is written
// (L = 0 where the split holds none of the row's keys), so the merge is v2's.
template <uint32_t GG, uint32_t NSG>
void i2_attn_tile(sycl::queue& q, const float* Q, const float* Kb, const float* Vb, uint32_t ring, const float* rel,
                  const float* proj, uint32_t d_rel, uint32_t extent, float* part, uint32_t T, uint32_t pos0,
                  uint32_t nh, uint32_t nkv, uint32_t hd, uint32_t window, uint32_t NS, bool kv16, float scale) {
    constexpr uint32_t SG = 16, WG = NSG * SG, KT = 16;
    const uint32_t G = nh / nkv, ntile = (T + NSG - 1) / NSG;
    ie::ps(q, "i2_attention_tile", [&](sycl::handler& h) {
        sycl::local_accessor<float, 1> ks(sycl::range<1>(KT * 128), h);
        sycl::local_accessor<float, 1> vs(sycl::range<1>(KT * 128), h);
        h.parallel_for(sycl::nd_range<1>(uint64_t(ntile) * nkv * NS * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const uint64_t g = it.get_group(0);
            const uint32_t sp = uint32_t(g % NS), kvh = uint32_t((g / NS) % nkv), tt = uint32_t(g / (uint64_t(NS) * nkv));
            const auto sgr = it.get_sub_group();
            const uint32_t l = uint32_t(sgr.get_local_linear_id()), sgid = uint32_t(sgr.get_group_linear_id());
            const uint32_t lt = uint32_t(it.get_local_id(0));
            const uint32_t t0 = tt * NSG, tl = sycl::min(T, t0 + NSG) - 1;   // rows t0 .. tl
            const uint32_t t = t0 + sgid;
            const bool row_on = t < T;
            const uint32_t P = pos0 + sycl::min(t, tl);
            const uint32_t jlo = (window > 0 && P + 1 > window) ? P + 1 - window : 0;
            // the union range of the tile's rows and this split's share of it
            const uint32_t P0 = pos0 + t0, PL = pos0 + tl;
            const uint32_t ulo = (window > 0 && P0 + 1 > window) ? P0 + 1 - window : 0;
            const uint32_t un = PL - ulo + 1, per = (un + NS - 1) / NS;
            const uint32_t a = ulo + sp * per, b = sycl::min(PL + 1, a + per);
            const uint32_t d0 = 8 * l;
            float qr[GG][8], acc[GG][8], rr[GG], m[GG], ls[GG];
#pragma unroll
            for (uint32_t gg = 0; gg < GG; ++gg) {
                const uint64_t hq = uint64_t(sycl::min(t, T - 1)) * nh + kvh * G + gg;
                const bool on = gg < G && row_on;
#pragma unroll
                for (uint32_t i = 0; i < 8; ++i) { qr[gg][i] = on ? Q[hq * hd + d0 + i] * scale : 0.f; acc[gg][i] = 0.f; }
                rr[gg] = (on && l < d_rel) ? rel[hq * d_rel + l] : 0.f;
                m[gg] = -3.0e38f;
                ls[gg] = 0.f;
            }
            for (uint32_t j0 = a; j0 < b; j0 += KT) {
                sycl::group_barrier(it.get_group());
                for (uint32_t e = lt; e < KT * 32; e += WG) {           // K and V rows of keys j0 .. j0 + 15 (float4 each)
                    const uint32_t kk = e / 32, c4 = e % 32, j = sycl::min(j0 + kk, b - 1);
                    const uint64_t row = ring ? j % ring : j;
                    const uint64_t ro = (row * nkv + kvh) * hd + 4 * c4;
                    sycl::float4 kx, vx;
                    if (kv16) {
                        const sycl::uint2 ku = *reinterpret_cast<const sycl::uint2*>(reinterpret_cast<const uint16_t*>(Kb) + ro);
                        const sycl::uint2 vu = *reinterpret_cast<const sycl::uint2*>(reinterpret_cast<const uint16_t*>(Vb) + ro);
                        kx = sycl::float4(sycl::bit_cast<float>(ku[0] << 16), sycl::bit_cast<float>(ku[0] & 0xFFFF0000u),
                                          sycl::bit_cast<float>(ku[1] << 16), sycl::bit_cast<float>(ku[1] & 0xFFFF0000u));
                        vx = sycl::float4(sycl::bit_cast<float>(vu[0] << 16), sycl::bit_cast<float>(vu[0] & 0xFFFF0000u),
                                          sycl::bit_cast<float>(vu[1] << 16), sycl::bit_cast<float>(vu[1] & 0xFFFF0000u));
                    } else {
                        kx = *reinterpret_cast<const sycl::float4*>(Kb + ro);
                        vx = *reinterpret_cast<const sycl::float4*>(Vb + ro);
                    }
                    *reinterpret_cast<sycl::float4*>(&ks[kk * 128 + 4 * c4]) = kx;
                    *reinterpret_cast<sycl::float4*>(&vs[kk * 128 + 4 * c4]) = vx;
                }
                sycl::group_barrier(it.get_group());
                if (!row_on || j0 > P || j0 + KT <= jlo) continue;     // none of these keys is visible to this row
                float d[GG][SG];
#pragma unroll
                for (uint32_t kk = 0; kk < SG; ++kk) {
                    float kv[8];
#pragma unroll
                    for (uint32_t i = 0; i < 8; ++i) kv[i] = ks[kk * 128 + d0 + i];
                    const uint32_t j = j0 + kk, dist = P - j;
                    const float pj = (j <= P && dist < extent && l < d_rel) ? proj[uint64_t(l) * extent + dist] : 0.f;
#pragma unroll
                    for (uint32_t gg = 0; gg < GG; ++gg) {
                        float x = 0.f;
#pragma unroll
                        for (uint32_t i = 0; i < 8; ++i) x += qr[gg][i] * kv[i];
                        d[gg][kk] = x + rr[gg] * pj;
                    }
                }
#pragma unroll
                for (uint32_t gg = 0; gg < GG; ++gg) {
#pragma unroll
                    for (uint32_t w = 8; w >= 1; w >>= 1) {
                        const bool up = (l & w) != 0;
#pragma unroll
                        for (uint32_t i = 0; i < w; ++i) {
                            const float keep = up ? d[gg][i + w] : d[gg][i];
                            const float send = up ? d[gg][i] : d[gg][i + w];
                            d[gg][i] = keep + sycl::permute_group_by_xor(sgr, send, w);
                        }
                    }
                }
                const uint32_t jl = j0 + l;
                const bool valid = jl < b && jl >= jlo && jl <= P;
                float pv[GG];
#pragma unroll
                for (uint32_t gg = 0; gg < GG; ++gg) {
                    const float sc = valid ? d[gg][0] : -3.0e38f;
                    const float tm = sycl::reduce_over_group(sgr, sc, sycl::maximum<float>());
                    const float nm = sycl::fmax(m[gg], tm);
                    const float corr = sycl::exp(m[gg] - nm);
                    pv[gg] = valid ? sycl::exp(sc - nm) : 0.f;
                    ls[gg] = ls[gg] * corr + pv[gg];
#pragma unroll
                    for (uint32_t i = 0; i < 8; ++i) acc[gg][i] *= corr;
                    m[gg] = nm;
                }
#pragma unroll
                for (uint32_t kk = 0; kk < SG; ++kk) {
                    float vv[8];
#pragma unroll
                    for (uint32_t i = 0; i < 8; ++i) vv[i] = vs[kk * 128 + d0 + i];
#pragma unroll
                    for (uint32_t gg = 0; gg < GG; ++gg) {
                        const float p = sycl::select_from_group(sgr, pv[gg], kk);
#pragma unroll
                        for (uint32_t i = 0; i < 8; ++i) acc[gg][i] += p * vv[i];
                    }
                }
            }
            if (!row_on) return;
#pragma unroll
            for (uint32_t gg = 0; gg < GG; ++gg) {
                if (gg >= G) break;
                const float L = sycl::reduce_over_group(sgr, ls[gg], sycl::plus<float>());
                float* pp = part + ((uint64_t(t) * nh + kvh * G + gg) * NS + sp) * (hd + 2);
#pragma unroll
                for (uint32_t i = 0; i < 8; ++i) pp[2 + d0 + i] = acc[gg][i];
                if (l == 0) { pp[0] = m[gg]; pp[1] = L; }
            }
        });
    });
}
}  // namespace

sycl::event i2_attention_split(sycl::queue& q, const float* Q, const float* Kc, const float* Vc, uint32_t ring,
                               const float* rel, const float* proj, uint32_t d_rel, uint32_t extent,
                               const float* sink, float* o, uint32_t T, uint32_t pos0, uint32_t nh, uint32_t nkv,
                               uint32_t hd, uint32_t window, float* part, uint64_t part_floats, uint32_t min_keys,
                               const float* gate, I2Rows rows, uint64_t kv_stride, uint32_t ns_max, bool kv16) {
    constexpr uint32_t WG = 128;                                        // keys per tile = threads; hd <= 128
    constexpr uint32_t GMAX = 4;                                        // query heads per KV head handled together
    const uint32_t G = nh / nkv;
    const bool rwh = rows.pos != nullptr;
    // NS: the launch's split range (the partials' stride); a batched row has its own count nsr <= NS (as alone)
    const uint32_t NS = rwh ? ns_max : i2_attention_splits(T, pos0, nh, hd, window, part_floats, min_keys);
    const uint32_t cap1 = uint32_t(std::max<uint64_t>(1, std::min<uint64_t>({part_floats / (uint64_t(nh) * (hd + 2)), 0xffffffffull,
                                                                              uint64_t(i2_decode_max_splits())})));
    const uint32_t mk = min_keys;
    auto row_ns = [=](uint32_t P) {                                     // i2_attention_splits(1, P, ...)
        const uint32_t max_keys = window ? sycl::min(window, P + 1) : P + 1;
        const uint32_t want = (max_keys + mk - 1) / mk;
        return sycl::max(1u, sycl::min(want, cap1));
    };
    const float scale = float(1.0 / std::sqrt(double(hd)));
    static const bool old_kernel = std::getenv("IE_INKLING2_ATTN_SPLIT_V1") != nullptr;   // A/B switch
    if ((rwh || kv16) && d_rel > 16) throw std::runtime_error("i2_attention_split: batched rows / a bf16 cache need d_rel <= 16");
    const float* Kb = Kc;
    const float* Vb = Vc;
    static const bool v2 = [] { const char* e = std::getenv("IE_INKLING2_ATTN_V2"); return !e || *e != '0'; }();
    // one query row over few keys (a short context): v1's one key per sub-group has less fixed cost per split
    static const uint32_t v2_min = [] { const char* e = std::getenv("IE_INKLING2_ATTN_V2_MINKEYS"); return e ? uint32_t(std::atoi(e)) : 384u; }();
    const uint32_t vis = window ? std::min(window, pos0 + T) : pos0 + T;
    const bool few = !rwh && T == 1 && vis < v2_min;
    static const bool tile = [] { const char* e = std::getenv("IE_INKLING2_ATTN_TILE"); return !e || *e != '0'; }();
    if (tile && !rwh && T >= 16 && !old_kernel && d_rel <= 16 && hd == 128 && G <= 4 && G >= 3 &&
        (uint64_t(reinterpret_cast<uintptr_t>(Kc)) & 15) == 0) {
        if (G == 3) i2_attn_tile<3, 8>(q, Q, Kc, Vc, ring, rel, proj, d_rel, extent, part, T, pos0, nh, nkv, hd, window, NS, kv16, scale);
        else i2_attn_tile<4, 8>(q, Q, Kc, Vc, ring, rel, proj, d_rel, extent, part, T, pos0, nh, nkv, hd, window, NS, kv16, scale);
    } else
    if (v2 && !few && !old_kernel && d_rel <= 16 && hd == 128 && G <= 4 && (uint64_t(reinterpret_cast<uintptr_t>(Kc)) & 15) == 0) {
        if (G == 3) i2_attn_split_v2<3>(q, Q, Kc, Vc, ring, rel, proj, d_rel, extent, part, T, pos0, nh, nkv, hd, window, NS, cap1, mk, rows, kv_stride, kv16, scale);
        else i2_attn_split_v2<4>(q, Q, Kc, Vc, ring, rel, proj, d_rel, extent, part, T, pos0, nh, nkv, hd, window, NS, cap1, mk, rows, kv_stride, kv16, scale);
    } else if ((!old_kernel || rwh || kv16) && d_rel <= 16) {
        // One 16-lane sub-group per key: lane l holds dims l, l + 16, ... of the G query heads, the K and V rows are read
        // 64 contiguous bytes at a time, one sub-group reduction per head gives the score (the relative bias term of
        // dimension x rides in lane x), and each sub-group keeps its own online softmax over every 8th key of the
        // split; the 8 sub-groups are merged in local memory into the same partial (max, sum, values) as before.
        constexpr uint32_t SG = 16, NSG = WG / SG, DM = 128 / SG;
        ie::ps(q, "i2_attention_split_sg", [&](sycl::handler& h) {
            sycl::local_accessor<float, 1> lm(sycl::range<1>(NSG * GMAX * 2), h);
            sycl::local_accessor<float, 1> la(sycl::range<1>(NSG * GMAX * 128), h);
            h.parallel_for(sycl::nd_range<1>(uint64_t(T) * nkv * NS * WG, WG),
                           [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                const uint64_t g = it.get_group(0);
                const uint32_t sp = uint32_t(g % NS), kvh = uint32_t((g / NS) % nkv), t = uint32_t(g / (uint64_t(NS) * nkv));
                const auto sgr = it.get_sub_group();
                const uint32_t l = uint32_t(sgr.get_local_linear_id()), sgid = uint32_t(sgr.get_group_linear_id());
                const bool rw = rows.pos != nullptr;
                const uint32_t P = rw ? uint32_t(rows.pos[t]) : pos0 + t;
                const uint32_t nsr = rw ? row_ns(P) : NS;
                if (sp >= nsr) return;                                  // the whole work-group: beyond this row's splits
                const uint64_t so = rw ? uint64_t(rows.slot[t]) * kv_stride : 0;
                const float* Kc = Kb + so;
                const float* Vc = Vb + so;
                const uint32_t jlo = (window > 0 && P + 1 > window) ? P + 1 - window : 0;
                const uint32_t n = P - jlo + 1;
                const uint32_t per = (n + nsr - 1) / nsr;
                const uint32_t a = jlo + sp * per, b = sycl::min(P + 1, a + per);
                float qr[GMAX][DM], rr[GMAX], m[GMAX], ls[GMAX], acc[GMAX][DM];
#pragma unroll
                for (uint32_t gg = 0; gg < GMAX; ++gg) {
                    const uint64_t hq = uint64_t(t) * nh + kvh * G + gg;
#pragma unroll
                    for (uint32_t i = 0; i < DM; ++i) {
                        const uint32_t dd = l + SG * i;
                        qr[gg][i] = (gg < G && dd < hd) ? Q[hq * hd + dd] : 0.f;
                        acc[gg][i] = 0.f;
                    }
                    rr[gg] = (gg < G && l < d_rel) ? rel[hq * d_rel + l] : 0.f;
                    m[gg] = -3.0e38f;
                    ls[gg] = 0.f;
                }
                for (uint32_t j = a + sgid; j < b; j += NSG) {
                    const uint64_t row = ring ? j % ring : j;
                    const uint64_t ro = (row * nkv + kvh) * hd;
                    float kv[DM], vv[DM];
                    if (kv16) {                                         // bf16 cache: the same values, widened
                        const uint16_t* kr = reinterpret_cast<const uint16_t*>(Kb) + so + ro;
                        const uint16_t* vr = reinterpret_cast<const uint16_t*>(Vb) + so + ro;
#pragma unroll
                        for (uint32_t i = 0; i < DM; ++i) {
                            const uint32_t dd = l + SG * i;
                            kv[i] = dd < hd ? i2_bf16_to_f32(kr[dd]) : 0.f;
                            vv[i] = dd < hd ? i2_bf16_to_f32(vr[dd]) : 0.f;
                        }
                    } else {
                        const float* kr = Kc + ro;
                        const float* vr = Vc + ro;
#pragma unroll
                        for (uint32_t i = 0; i < DM; ++i) {
                            const uint32_t dd = l + SG * i;
                            kv[i] = dd < hd ? kr[dd] : 0.f;
                            vv[i] = dd < hd ? vr[dd] : 0.f;
                        }
                    }
                    const uint32_t dist = P - j;
                    const float pj = (dist < extent && l < d_rel) ? proj[uint64_t(l) * extent + dist] : 0.f;
#pragma unroll
                    for (uint32_t gg = 0; gg < GMAX; ++gg) {
                        if (gg >= G) break;
                        float d = 0.f;
#pragma unroll
                        for (uint32_t i = 0; i < DM; ++i) d += qr[gg][i] * kv[i];
                        const float sc = sycl::reduce_over_group(sgr, d * scale + rr[gg] * pj, sycl::plus<float>());
                        const float nm = sycl::fmax(m[gg], sc);
                        const float corr = sycl::exp(m[gg] - nm), p = sycl::exp(sc - nm);
                        ls[gg] = ls[gg] * corr + p;
#pragma unroll
                        for (uint32_t i = 0; i < DM; ++i) acc[gg][i] = acc[gg][i] * corr + p * vv[i];
                        m[gg] = nm;
                    }
                }
#pragma unroll
                for (uint32_t gg = 0; gg < GMAX; ++gg) {
                    if (gg >= G) break;
                    if (l == 0) { lm[(sgid * GMAX + gg) * 2] = m[gg]; lm[(sgid * GMAX + gg) * 2 + 1] = ls[gg]; }
#pragma unroll
                    for (uint32_t i = 0; i < DM; ++i) {
                        const uint32_t dd = l + SG * i;
                        if (dd < hd) la[(sgid * GMAX + gg) * 128 + dd] = acc[gg][i];
                    }
                }
                sycl::group_barrier(it.get_group());
                const uint32_t lt = uint32_t(it.get_local_id(0));
                for (uint32_t gg = 0; gg < G; ++gg) {
                    float M = -3.0e38f;
                    for (uint32_t s2 = 0; s2 < NSG; ++s2)
                        if (lm[(s2 * GMAX + gg) * 2 + 1] > 0.f) M = sycl::fmax(M, lm[(s2 * GMAX + gg) * 2]);
                    float L = 0.f, A = 0.f;
                    for (uint32_t s2 = 0; s2 < NSG; ++s2) {
                        const float l2 = lm[(s2 * GMAX + gg) * 2 + 1];
                        if (!(l2 > 0.f)) continue;
                        const float f = sycl::exp(lm[(s2 * GMAX + gg) * 2] - M);
                        L += l2 * f;
                        if (lt < hd) A += la[(s2 * GMAX + gg) * 128 + lt] * f;
                    }
                    float* pp = part + ((uint64_t(t) * nh + kvh * G + gg) * NS + sp) * (hd + 2);
                    if (lt < hd) pp[2 + lt] = A;
                    if (lt == 0) { pp[0] = M; pp[1] = L; }
                }
            });
        });
    } else
    ie::ps(q, "i2_attention_split", [&](sycl::handler& h) {
        sycl::local_accessor<float, 1> qs(sycl::range<1>(GMAX * 128), h);
        sycl::local_accessor<float, 1> rs(sycl::range<1>(GMAX * 32), h);
        sycl::local_accessor<float, 1> ps_(sycl::range<1>(GMAX * WG), h);   // tile probabilities per head
        h.parallel_for(sycl::nd_range<1>(uint64_t(T) * nkv * NS * WG, WG), [=](sycl::nd_item<1> it) {
            const uint64_t g = it.get_group(0);
            const uint32_t sp = uint32_t(g % NS), kvh = uint32_t((g / NS) % nkv), t = uint32_t(g / (uint64_t(NS) * nkv));
            const uint32_t l = uint32_t(it.get_local_id(0));
            const uint32_t P = pos0 + t;
            const uint32_t jlo = (window > 0 && P + 1 > window) ? P + 1 - window : 0;
            const uint32_t n = P - jlo + 1;                              // visible keys
            const uint32_t per = (n + NS - 1) / NS;
            const uint32_t a = jlo + sp * per, b = sycl::min(P + 1, a + per);   // this split: keys [a, b)
            for (uint32_t i = l; i < G * hd; i += WG) qs[i] = Q[(uint64_t(t) * nh + kvh * G) * hd + i];
            for (uint32_t i = l; i < G * d_rel; i += WG) rs[i] = rel[(uint64_t(t) * nh + kvh * G) * d_rel + i];
            sycl::group_barrier(it.get_group());
            float m[GMAX], lsum[GMAX], acc[GMAX];
            for (uint32_t gg = 0; gg < GMAX; ++gg) { m[gg] = -3.0e38f; lsum[gg] = 0.f; acc[gg] = 0.f; }
            for (uint32_t j0 = a; j0 < b; j0 += WG) {
                const uint32_t j = j0 + l;
                float sc[GMAX];
                if (j < b) {
                    const uint64_t row = ring ? j % ring : j;
                    const float* kv = Kc + (row * nkv + kvh) * hd;
                    for (uint32_t gg = 0; gg < G; ++gg) sc[gg] = 0.f;
                    for (uint32_t c = 0; c < hd; ++c) {
                        const float kc = kv[c];
                        for (uint32_t gg = 0; gg < G; ++gg) sc[gg] += qs[gg * hd + c] * kc;
                    }
                    const uint32_t dist = P - j;
                    for (uint32_t gg = 0; gg < G; ++gg) {
                        float bb = 0.f;
                        if (dist < extent)
                            for (uint32_t x = 0; x < d_rel; ++x) bb += rs[gg * d_rel + x] * proj[uint64_t(x) * extent + dist];
                        sc[gg] = sc[gg] * scale + bb;
                    }
                } else {
                    for (uint32_t gg = 0; gg < G; ++gg) sc[gg] = -3.0e38f;
                }
                for (uint32_t gg = 0; gg < G; ++gg) {
                    const float tm = sycl::reduce_over_group(it.get_group(), sc[gg], sycl::maximum<float>());
                    const float nm = sycl::fmax(m[gg], tm);
                    const float pj = j < b ? sycl::exp(sc[gg] - nm) : 0.f;
                    const float corr = sycl::exp(m[gg] - nm);
                    lsum[gg] = lsum[gg] * corr + sycl::reduce_over_group(it.get_group(), pj, sycl::plus<float>());
                    acc[gg] *= corr;
                    m[gg] = nm;
                    ps_[gg * WG + l] = pj;
                }
                sycl::group_barrier(it.get_group());
                const uint32_t cnt = sycl::min(WG, b - j0);
                if (l < hd)
                    for (uint32_t i = 0; i < cnt; ++i) {
                        const uint64_t row = ring ? (j0 + i) % ring : (j0 + i);
                        const float vv = Vc[(row * nkv + kvh) * hd + l];
                        for (uint32_t gg = 0; gg < G; ++gg) acc[gg] += ps_[gg * WG + i] * vv;
                    }
                sycl::group_barrier(it.get_group());
            }
            for (uint32_t gg = 0; gg < G; ++gg) {
                float* pp = part + ((uint64_t(t) * nh + kvh * G + gg) * NS + sp) * (hd + 2);
                if (l < hd) pp[2 + l] = acc[gg];
                if (l == 0) { pp[0] = m[gg]; pp[1] = lsum[gg]; }
            }
        });
    });
    return ie::ps(q, "i2_attention_merge", [&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(uint64_t(T) * nh * hd, hd), [=](sycl::nd_item<1> it) {
            const uint64_t th = it.get_group(0);                        // t * nh + head
            const uint32_t d = uint32_t(it.get_local_id(0));
            const float sk = sink[th];
            const float* pp = part + th * NS * (hd + 2);
            const uint32_t nsr = rows.pos ? row_ns(uint32_t(rows.pos[th / nh])) : NS;
            float M = sk;
            for (uint32_t s2 = 0; s2 < nsr; ++s2) if (pp[s2 * (hd + 2) + 1] > 0.f) M = sycl::fmax(M, pp[s2 * (hd + 2)]);
            float L = sycl::exp(sk - M), acc = 0.f;
            for (uint32_t s2 = 0; s2 < nsr; ++s2) {
                const float* e = pp + s2 * (hd + 2);
                if (!(e[1] > 0.f)) continue;                            // an empty split (fewer keys than splits)
                const float f = sycl::exp(e[0] - M);
                L += e[1] * f;
                acc += e[2 + d] * f;
            }
            const float ov = acc / L;
            o[th * hd + d] = gate ? ov * (1.f / (1.f + sycl::exp(-gate[th * hd + d]))) : ov;
        });
    });
}

sycl::event i2_store_rows(sycl::queue& q, const float* src, uint32_t T, uint32_t width, float* dst, uint32_t pos0,
                          uint32_t ring) {
    return ie::ps(q, "i2_store_rows", [&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(uint64_t(T) * width), [=](sycl::id<1> id) {
            const uint64_t t = id[0] / width, c = id[0] % width;
            const uint64_t pos = pos0 + t;
            dst[(ring ? pos % ring : pos) * width + c] = src[id[0]];
        });
    });
}

sycl::event i2_router(sycl::queue& q, const float* logits, const float* bias, uint32_t T, uint32_t E, uint32_t k,
                      float route_scale, const int32_t* forced, int32_t* used, float* w, int32_t* own, float* margin) {
    static const bool serial = std::getenv("IE_INKLING2_ROUTER_SERIAL") != nullptr;          // A/B switch
    static const bool wg_router = std::getenv("IE_INKLING2_ROUTER_WG") != nullptr;            // A/B switch
    if (!serial && !wg_router && E <= 256 && k <= 16) {
        // One 16-lane sub-group per token, lane l holding experts l, l + 16, ...: each of the k + 1 rounds takes the
        // largest value (sub-group max of the lanes' maxima) and the lowest id holding it (sub-group min) -- the same
        // picks and tie rule as the work-group version, with shuffles instead of work-group reductions.
        constexpr uint32_t SG = 16, PL = 16;                               // E <= 256
        return ie::ps(q, "i2_router_sg", [&](sycl::handler& h) {
            h.parallel_for(sycl::nd_range<1>(uint64_t(T) * SG, SG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                const auto sg = it.get_sub_group();
                const uint32_t t = uint32_t(it.get_group(0)), l = uint32_t(sg.get_local_linear_id());
                const float* lg = logits + uint64_t(t) * E;
                float v[PL];
                uint32_t taken = 0;
#pragma unroll
                for (uint32_t i = 0; i < PL; ++i) {
                    const uint32_t e = l + SG * i;
                    v[i] = e < E ? 1.f / (1.f + sycl::exp(-lg[e])) + bias[e] : -3.0e38f;
                    if (e >= E) taken |= 1u << i;
                }
                // round j's pick and value stay in lane j (no indexed private arrays: those go to scratch memory)
                int32_t myp = 0x7fffffff;
                float myv = -3.0e38f;
#pragma unroll
                for (uint32_t j = 0; j < 17; ++j) {
                    if (j > k || j >= E) break;
                    float lm = -3.0e38f;
                    int32_t li = 0x7fffffff;
#pragma unroll
                    for (uint32_t i = 0; i < PL; ++i)
                        if (!(taken & (1u << i)) && (li == 0x7fffffff || v[i] > lm)) { lm = v[i]; li = int32_t(l + SG * i); }
                    const float mv = sycl::reduce_over_group(sg, li == 0x7fffffff ? -3.0e38f : lm, sycl::maximum<float>());
                    const int32_t best = sycl::reduce_over_group(sg, (li != 0x7fffffff && lm == mv) ? li : 0x7fffffff, sycl::minimum<int32_t>());
                    if (l == j) { myp = best; myv = mv; }
                    if (uint32_t(best) % SG == l) taken |= 1u << (uint32_t(best) / SG);
                }
                if (k < E) {
                    const float a1 = sycl::select_from_group(sg, myv, k - 1), a2 = sycl::select_from_group(sg, myv, k);
                    if (l == 0) margin[t] = a1 - a2;
                }
                const bool in = l < k;
                // ascending ids: a lane's place = how many picks are smaller (distinct ids)
                uint32_t rk = 0, ru = 0;
                const int32_t u = in ? (forced ? forced[uint64_t(t) * k + l] : myp) : 0x7fffffff;
#pragma unroll
                for (uint32_t j = 0; j < 16; ++j) {
                    const int32_t pj = sycl::select_from_group(sg, myp, j), uj = sycl::select_from_group(sg, u, j);
                    if (j < k) { rk += pj < myp ? 1u : 0u; ru += (uj < u || (uj == u && j < l)) ? 1u : 0u; }
                }
                if (in) { own[uint64_t(t) * k + rk] = myp; used[uint64_t(t) * k + ru] = u; }
                const float scv = in ? 1.f / (1.f + sycl::exp(-lg[u])) : 0.f;
                float sum = 0.f;                                          // in ascending order, as the serial sum
#pragma unroll
                for (uint32_t r = 0; r < 16; ++r) {
                    if (r >= k) break;
                    sum += sycl::reduce_over_group(sg, (in && ru == r) ? scv : 0.f, sycl::plus<float>());
                }
                if (in) w[uint64_t(t) * k + ru] = scv / (sum + 1e-20f) * route_scale;
            });
        });
    }
    if (!serial && E <= 256 && k <= 16) {
        // One work-group per token, one lane per expert: k + 1 rounds of (max value, then the lowest id holding it),
        // the same picks and tie rule as the serial loop below; lane 0 then sorts and forms the weights.
        constexpr uint32_t WG = 256;
        return ie::ps(q, "i2_router_wg", [&](sycl::handler& h) {
            sycl::local_accessor<int32_t, 1> pk(sycl::range<1>(17), h);
            sycl::local_accessor<float, 1> pv(sycl::range<1>(17), h);
            h.parallel_for(sycl::nd_range<1>(uint64_t(T) * WG, WG), [=](sycl::nd_item<1> it) {
                const uint32_t t = uint32_t(it.get_group(0)), e = uint32_t(it.get_local_id(0));
                const float* lg = logits + uint64_t(t) * E;
                float v = -3.0e38f;
                if (e < E) v = 1.f / (1.f + sycl::exp(-lg[e])) + bias[e];
                bool taken = e >= E;
                for (uint32_t j = 0; j <= k && j < E; ++j) {
                    const float mv = sycl::reduce_over_group(it.get_group(), taken ? -3.0e38f : v, sycl::maximum<float>());
                    const int32_t cand = (!taken && v == mv) ? int32_t(e) : int32_t(WG);
                    const int32_t best = sycl::reduce_over_group(it.get_group(), cand, sycl::minimum<int32_t>());
                    if (e == 0) { pk[j] = best; pv[j] = mv; }
                    if (int32_t(e) == best) taken = true;
                }
                sycl::group_barrier(it.get_group());
                if (e != 0) return;
                int32_t pick[16];
                for (uint32_t j = 0; j < k; ++j) pick[j] = pk[j];
                if (k < E) margin[t] = pv[k - 1] - pv[k];
                for (uint32_t a = 1; a < k; ++a)
                    for (uint32_t b2 = a; b2 > 0 && pick[b2] < pick[b2 - 1]; --b2) { const int32_t x = pick[b2]; pick[b2] = pick[b2 - 1]; pick[b2 - 1] = x; }
                int32_t u[16];
                for (uint32_t j = 0; j < k; ++j) { own[uint64_t(t) * k + j] = pick[j]; u[j] = forced ? forced[uint64_t(t) * k + j] : pick[j]; }
                for (uint32_t a = 1; a < k; ++a)
                    for (uint32_t b2 = a; b2 > 0 && u[b2] < u[b2 - 1]; --b2) { const int32_t x = u[b2]; u[b2] = u[b2 - 1]; u[b2 - 1] = x; }
                float sc[16], sum = 0.f;
                for (uint32_t j = 0; j < k; ++j) { sc[j] = 1.f / (1.f + sycl::exp(-lg[u[j]])); sum += sc[j]; }
                for (uint32_t j = 0; j < k; ++j) {
                    used[uint64_t(t) * k + j] = u[j];
                    w[uint64_t(t) * k + j] = sc[j] / (sum + 1e-20f) * route_scale;
                }
            });
        });
    }
    return ie::ps(q, "i2_router", [&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(T), [=](sycl::id<1> id) {
            const uint32_t t = uint32_t(id[0]);
            const float* lg = logits + uint64_t(t) * E;
            uint32_t taken[8] = {0, 0, 0, 0, 0, 0, 0, 0};              // bitset over E <= 256
            int32_t pick[16];
            float kth = 0.f;
            for (uint32_t j = 0; j <= k && j < E; ++j) {                // k picks, then the (k+1)-th for the margin
                int best = -1;
                float bv = 0.f;
                for (uint32_t e = 0; e < E; ++e) {
                    if (taken[e >> 5] & (1u << (e & 31))) continue;
                    const float v = 1.f / (1.f + sycl::exp(-lg[e])) + bias[e];
                    if (best < 0 || v > bv) { best = int(e); bv = v; }  // strict >: lower id wins exact ties
                }
                if (j == k) { margin[t] = kth - bv; break; }
                taken[best >> 5] |= 1u << (best & 31);
                pick[j] = best;
                kth = bv;
            }
            for (uint32_t a = 1; a < k; ++a)                            // ascending ids
                for (uint32_t b = a; b > 0 && pick[b] < pick[b - 1]; --b) { const int32_t x = pick[b]; pick[b] = pick[b - 1]; pick[b - 1] = x; }
            int32_t u[16];
            for (uint32_t j = 0; j < k; ++j) { own[uint64_t(t) * k + j] = pick[j]; u[j] = forced ? forced[uint64_t(t) * k + j] : pick[j]; }
            for (uint32_t a = 1; a < k; ++a)
                for (uint32_t b = a; b > 0 && u[b] < u[b - 1]; --b) { const int32_t x = u[b]; u[b] = u[b - 1]; u[b - 1] = x; }
            float sc[16], sum = 0.f;
            for (uint32_t j = 0; j < k; ++j) { sc[j] = 1.f / (1.f + sycl::exp(-lg[u[j]])); sum += sc[j]; }
            for (uint32_t j = 0; j < k; ++j) {
                used[uint64_t(t) * k + j] = u[j];
                w[uint64_t(t) * k + j] = sc[j] / (sum + 1e-20f) * route_scale;
            }
        });
    });
}

namespace {
// 2 * |e2m1| for the 3 magnitude bits, packed 4 bits each: 0, 1, 2, 3, 4, 6, 8, 12; the sign bit goes into the float.
inline float i2_e2m1x2(uint32_t c) {
    const float v = float((0xC8643210u >> ((c & 7u) << 2)) & 15u);
    return sycl::bit_cast<float>(sycl::bit_cast<uint32_t>(v) | ((c & 8u) << 28));
}
// E4M3 without the NaN code (0x7F / 0xFF; a checkpoint containing it would give NaN outputs either way).
inline float i2_e4m3_fin(uint32_t b) {
    const uint32_t e = (b >> 3) & 15u, m = b & 7u;
    const float v = e ? sycl::bit_cast<float>(((b & 0x7Fu) << 20) + (120u << 23)) : float(m) * 0.001953125f;
    return sycl::bit_cast<float>(sycl::bit_cast<uint32_t>(v) | ((b & 0x80u) << 24));
}

// One 16-lane sub-group per (slot, output row), 8 rows of one slot per work-group (they share the input vector).
// fp32 path only (no bf16 rounding of the weights): the block scale and the per-expert factor are applied once per
// block / row instead of per element, so sum(x * ((c * s) * f)) becomes f * sum(s * sum(x * c)) -- the same products,
// a different fp32 rounding order. NVFP4: a lane loads 8 bytes = one 16-column scale block; MXFP8: 16 bytes, half of
// one 32-column exponent block.
// Two NVFP4 nibbles (j, j + 4 of the word u >> 4j) as fp16 pairs: magnitude e1 e0 m into fp16 bits 11..9, the sign to bit
// 15, so the half is the e2m1 value times 2^-14 exactly (e = 0: an fp16 subnormal). 6 integer ops and 2 conversions per
// pair. The sums differ from i2_e2m1x2's (2 x value) by the exact factor 2^-15, folded into the per-expert factor:
// the rounded results are bit-identical.
inline sycl::float2 i2_h2f(uint32_t h) { return sycl::bit_cast<sycl::vec<sycl::half, 2>>(h).convert<float>(); }
inline void i2_nv4_dec8(uint32_t u, float* w) {                         // w[j] = value(nibble j) * 2^-14, j = 0..7
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint32_t s = u >> (4 * j);
        const sycl::float2 f = i2_h2f(((s & 0x00070007u) << 9) | ((s & 0x00080008u) << 12));
        w[j] = f[0]; w[j + 4] = f[1];
    }
}
// Four E4M3 bytes as fp16: the 7 magnitude bits into fp16 bits 13..7, the sign to bit 15: the value times 2^-8 exactly
// (subnormals included); 0x7F reads as 480 as i2_e4m3_fin does. Folded back with 256 per row: bit-identical sums.
inline void i2_e4m3_dec4(uint32_t u, float* w) {                        // w[b] = value(byte b) * 2^-8, b = 0..3
    const sycl::float2 a = i2_h2f(((u & 0x007F007Fu) << 7) | ((u & 0x00800080u) << 8));
    const uint32_t v = u >> 8;
    const sycl::float2 c = i2_h2f(((v & 0x007F007Fu) << 7) | ((v & 0x00800080u) << 8));
    w[0] = a[0]; w[2] = a[1]; w[1] = c[0]; w[3] = c[1];
}

inline float i2_swiglu_cap(float g, float u, float gcap, float ucap) {   // as i2_expert_act
    return gcap * sycl::tanh(g / gcap) * (1.f / (1.f + sycl::exp(-g))) * ucap * sycl::tanh(u / ucap);
}
// NVFP4, flat: a sub-group streams its RPS rows (contiguous in memory) as one run of 16-byte chunks (two scale blocks
// each), lane l taking chunks l, l + 16, ... over the run (as the unembedding's kernel reads its rows): 16-byte loads,
// no idle lanes when RPS * cols / 32 is a multiple of 16. A chunk's row is lane-dependent: each lane adds its block
// sums into the row's accumulator by a select. A different fp32 order from i2_expert_matvec_sg (fp noise).
template <uint32_t RPS, uint32_t U, uint32_t CPR = 0>
sycl::event i2_expert_flat_nv4(sycl::queue& q, const I2ExpertBank& b, const float* x, bool x_per_slot, const int32_t* idx,
                               uint32_t T, uint32_t k, float* y, bool round, float* act, float gcap, float ucap) {
    constexpr uint32_t SG = 16, NSG = 8, RPG = NSG * RPS;
    const uint32_t rows = b.rows, cols = b.cols;
    const uint32_t rgroups = (rows + RPG - 1) / RPG;
    const uint32_t cpr = CPR ? CPR : cols / 32;                         // 16-byte chunks per row
    const uint32_t its = RPS * cpr / SG;
    const uint8_t* W = b.w;
    const uint8_t* S = b.s;
    const float* fac = b.fac;
    return ie::ps(q, "i2_expert_flat", [&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(uint64_t(T) * k * rgroups * NSG * SG, NSG * SG),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const auto sg = it.get_sub_group();
            const uint64_t g = it.get_group(0);
            const uint64_t slot = g / rgroups;
            const uint32_t rb = uint32_t(g % rgroups) * RPG + uint32_t(sg.get_group_linear_id()) * RPS;
            if (rb >= rows) return;
            const uint32_t t = uint32_t(slot / k);
            const uint32_t l = uint32_t(sg.get_local_linear_id());
            const uint64_t e = uint64_t(idx[slot]);
            const float* xv = x + (x_per_slot ? slot : uint64_t(t)) * cols;
            const uint64_t row0 = e * rows + rb;                       // RPS full rows (rows % RPS == 0)
            const sycl::uint4* Wr = reinterpret_cast<const sycl::uint4*>(W + row0 * (cols / 2));
            const uint16_t* Sr = reinterpret_cast<const uint16_t*>(S + row0 * (cols / 16));
            float acc[RPS];
#pragma unroll
            for (uint32_t r = 0; r < RPS; ++r) acc[r] = 0.f;
            for (uint32_t i0 = 0; i0 < its; i0 += U) {
                sycl::uint4 v[U];
                uint16_t sc[U];
#pragma unroll
                for (uint32_t u = 0; u < U; ++u) {
                    const uint32_t ci = l + SG * sycl::min(i0 + u, its - 1);
                    v[u] = Wr[ci];
                    sc[u] = Sr[ci];                                     // row-major chunks: scale pair index = chunk index
                }
#pragma unroll
                for (uint32_t u = 0; u < U; ++u) {
                    if (i0 + u >= its) break;
                    const uint32_t ci = l + SG * (i0 + u);
                    const uint32_t r = CPR ? ci / CPR : ci / cpr, cc = ci - r * cpr;
                    const sycl::float4* xb = reinterpret_cast<const sycl::float4*>(xv + 32 * cc);
                    float val = 0.f;
#pragma unroll
                    for (int hb = 0; hb < 2; ++hb) {
                        float w0[8], w1[8];
                        i2_nv4_dec8(v[u][2 * hb], w0); i2_nv4_dec8(v[u][2 * hb + 1], w1);
                        const sycl::float4 x0 = xb[4 * hb], x1 = xb[4 * hb + 1], x2 = xb[4 * hb + 2], x3 = xb[4 * hb + 3];
                        float p = 0.f;
                        p += x0[0] * w0[0] + x0[1] * w0[1] + x0[2] * w0[2] + x0[3] * w0[3] + x1[0] * w0[4] + x1[1] * w0[5] +
                             x1[2] * w0[6] + x1[3] * w0[7];
                        p += x2[0] * w1[0] + x2[1] * w1[1] + x2[2] * w1[2] + x2[3] * w1[3] + x3[0] * w1[4] + x3[1] * w1[5] +
                             x3[2] * w1[6] + x3[3] * w1[7];
                        val += p * i2_e4m3_fin((sc[u] >> (8 * hb)) & 255u);
                    }
#pragma unroll
                    for (uint32_t r2 = 0; r2 < RPS; ++r2) acc[r2] += r == r2 ? val : 0.f;
                }
            }
            const float f = 16384.f * fac[e];
#pragma unroll
            for (uint32_t r = 0; r < RPS; ++r) acc[r] *= f;
            if (act) {
#pragma unroll
                for (uint32_t r = 0; r < RPS; r += 2) {
                    const float g2 = sycl::reduce_over_group(sg, acc[r], sycl::plus<float>());
                    const float u2 = sycl::reduce_over_group(sg, acc[r + 1], sycl::plus<float>());
                    if (l == 0) act[slot * (rows / 2) + (rb + r) / 2] = i2_swiglu_cap(g2, u2, gcap, ucap);
                }
                return;
            }
#pragma unroll
            for (uint32_t r = 0; r < RPS; ++r) {
                const float s2 = sycl::reduce_over_group(sg, acc[r], sycl::plus<float>());
                if (l == 0) y[slot * rows + rb + r] = i2r(s2, round);
            }
        });
    });
}

template <int FMT, uint32_t RPS, bool WIDE = false, uint32_t NSG = 8, int DIAG = 0, int DEC = 0, uint32_t U = 1>
sycl::event i2_expert_matvec_sg(sycl::queue& q, const I2ExpertBank& b, const float* x, bool x_per_slot,
                                const int32_t* idx, uint32_t T, uint32_t k, float* y, bool round,
                                float* act = nullptr, float gcap = 0.f, float ucap = 0.f) {
    // RPS rows per sub-group share each x load (x is 8x the NVFP4 weight bytes); every row's sum is formed in the same
    // order as with one row per sub-group. DIAG (timing experiments only, wrong values): 1 = trivial nibble decode,
    // 2 = no x loads.
    constexpr uint32_t SG = 16, RPG = NSG * RPS;
    const uint32_t rows = b.rows, cols = b.cols;
    const uint32_t rgroups = (rows + RPG - 1) / RPG;
    const uint8_t* W = b.w;
    const uint8_t* S = b.s;
    const float* fac = b.fac;
    return ie::ps(q, "i2_expert_matvec_sg", [&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(uint64_t(T) * k * rgroups * NSG * SG, NSG * SG),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const auto sg = it.get_sub_group();
            const uint64_t g = it.get_group(0);
            const uint64_t slot = g / rgroups;                          // t * k + j
            const uint32_t rb = uint32_t(g % rgroups) * RPG + uint32_t(sg.get_group_linear_id()) * RPS;
            if (rb >= rows) return;
            const uint32_t t = uint32_t(slot / k);
            const uint32_t l = uint32_t(sg.get_local_linear_id());
            const uint64_t e = uint64_t(idx[slot]);
            const float* xv = x + (x_per_slot ? slot : uint64_t(t)) * cols;
            float acc[RPS];
#pragma unroll
            for (uint32_t r = 0; r < RPS; ++r) acc[r] = 0.f;
            if constexpr (DEC == 1) {                                   // fp16-bit decode (i2_nv4_dec8 / i2_e4m3_dec4)
                // U iterations' weight, scale and x loads are issued before any of their arithmetic (more bytes in
                // flight per lane); the sums are formed in the same order as one iteration at a time.
                constexpr uint32_t WB = FMT == 0 ? 2 : 4;                // uint32 words of weights per lane per row
                const uint32_t nblk = cols / 16;
                uint64_t rowv[RPS];
#pragma unroll
                for (uint32_t r = 0; r < RPS; ++r) rowv[r] = e * rows + sycl::min(rb + r, rows - 1);
                for (uint32_t c0 = l; c0 < nblk; c0 += SG * U) {
                    uint32_t v[U][RPS][WB];
                    uint8_t sb[U][RPS];
                    sycl::float4 xa[U][4];
#pragma unroll
                    for (uint32_t u = 0; u < U; ++u) {
                        const uint32_t c = c0 + SG * u;
                        const bool in = c < nblk;
                        const uint32_t cc = in ? c : c0;                 // a lane past the end re-reads its first block
#pragma unroll
                        for (uint32_t r = 0; r < RPS; ++r) {
                            if constexpr (FMT == 0) {
                                const sycl::uint2 w = reinterpret_cast<const sycl::uint2*>(W + rowv[r] * (cols / 2))[cc];
                                v[u][r][0] = w[0]; v[u][r][1] = w[1];
                                sb[u][r] = S[rowv[r] * (cols / 16) + cc];
                            } else {
                                const sycl::uint4 w = reinterpret_cast<const sycl::uint4*>(W + rowv[r] * cols)[cc];
                                v[u][r][0] = w[0]; v[u][r][1] = w[1]; v[u][r][2] = w[2]; v[u][r][3] = w[3];
                                sb[u][r] = S[rowv[r] * (cols / 32) + (cc >> 1)];
                            }
                        }
                        const sycl::float4* xb = reinterpret_cast<const sycl::float4*>(xv + 16 * cc);
                        xa[u][0] = xb[0]; xa[u][1] = xb[1]; xa[u][2] = xb[2]; xa[u][3] = xb[3];
                    }
#pragma unroll
                    for (uint32_t u = 0; u < U; ++u) {
                        if (c0 + SG * u >= nblk) break;
                        const sycl::float4 x0 = xa[u][0], x1 = xa[u][1], x2 = xa[u][2], x3 = xa[u][3];
#pragma unroll
                        for (uint32_t r = 0; r < RPS; ++r) {
                            float p = 0.f;
                            if constexpr (FMT == 0) {
                                float w0[8], w1[8];
                                i2_nv4_dec8(v[u][r][0], w0); i2_nv4_dec8(v[u][r][1], w1);
                                p += x0[0] * w0[0] + x0[1] * w0[1] + x0[2] * w0[2] + x0[3] * w0[3] + x1[0] * w0[4] + x1[1] * w0[5] +
                                     x1[2] * w0[6] + x1[3] * w0[7];
                                p += x2[0] * w1[0] + x2[1] * w1[1] + x2[2] * w1[2] + x2[3] * w1[3] + x3[0] * w1[4] + x3[1] * w1[5] +
                                     x3[2] * w1[6] + x3[3] * w1[7];
                                acc[r] += p * i2_e4m3_fin(sb[u][r]);
                            } else {
                                for (int j = 0; j < 4; ++j) {
                                    float w[4];
                                    i2_e4m3_dec4(v[u][r][j], w);
                                    p += xa[u][j][0] * w[0] + xa[u][j][1] * w[1] + xa[u][j][2] * w[2] + xa[u][j][3] * w[3];
                                }
                                acc[r] += p * i2_ue8m0(sb[u][r]);
                            }
                        }
                    }
                }
                const float f = FMT == 0 ? 16384.f * fac[e] : 256.f;    // 0.5 * 2^15 * fac; 2^8
#pragma unroll
                for (uint32_t r = 0; r < RPS; ++r) acc[r] *= f;
            } else if constexpr (FMT == 0 && WIDE) {                   // 16 bytes = two scale blocks per lane load
                for (uint32_t c = l; c < cols / 32; c += SG) {
                    const sycl::float4* xb = reinterpret_cast<const sycl::float4*>(xv + 32 * c);
#pragma unroll
                    for (uint32_t r = 0; r < RPS; ++r) {
                        if (rb + r >= rows) break;
                        const uint64_t row = e * rows + rb + r;
                        const sycl::uint4 v = reinterpret_cast<const sycl::uint4*>(W + row * (cols / 2))[c];
                        const uint16_t sc2 = reinterpret_cast<const uint16_t*>(S + row * (cols / 16))[c];
#pragma unroll
                        for (int hb = 0; hb < 2; ++hb) {
                            const uint32_t u0 = v[2 * hb], u1 = v[2 * hb + 1];
                            const sycl::float4 x0 = xb[4 * hb], x1 = xb[4 * hb + 1], x2 = xb[4 * hb + 2], x3 = xb[4 * hb + 3];
                            float p = 0.f;
                            p += x0[0] * i2_e2m1x2(u0) + x0[1] * i2_e2m1x2(u0 >> 4) + x0[2] * i2_e2m1x2(u0 >> 8) +
                                 x0[3] * i2_e2m1x2(u0 >> 12) + x1[0] * i2_e2m1x2(u0 >> 16) + x1[1] * i2_e2m1x2(u0 >> 20) +
                                 x1[2] * i2_e2m1x2(u0 >> 24) + x1[3] * i2_e2m1x2(u0 >> 28);
                            p += x2[0] * i2_e2m1x2(u1) + x2[1] * i2_e2m1x2(u1 >> 4) + x2[2] * i2_e2m1x2(u1 >> 8) +
                                 x2[3] * i2_e2m1x2(u1 >> 12) + x3[0] * i2_e2m1x2(u1 >> 16) + x3[1] * i2_e2m1x2(u1 >> 20) +
                                 x3[2] * i2_e2m1x2(u1 >> 24) + x3[3] * i2_e2m1x2(u1 >> 28);
                            acc[r] += p * i2_e4m3_fin((sc2 >> (8 * hb)) & 255u);
                        }
                    }
                }
                const float f = 0.5f * fac[e];
#pragma unroll
                for (uint32_t r = 0; r < RPS; ++r) acc[r] *= f;
            } else if constexpr (FMT == 0 && DIAG == 1) {
                for (uint32_t c = l; c < cols / 16; c += SG) {
                    const sycl::float4* xb = reinterpret_cast<const sycl::float4*>(xv + 16 * c);
                    const sycl::float4 x0 = xb[0], x1 = xb[1], x2 = xb[2], x3 = xb[3];
#pragma unroll
                    for (uint32_t r = 0; r < RPS; ++r) {
                        const uint64_t row = e * rows + sycl::min(rb + r, rows - 1);
                        const sycl::uint2 v = reinterpret_cast<const sycl::uint2*>(W + row * (cols / 2))[c];
                        const float a0 = float(v[0]), a1 = float(v[1]);
                        float p = (x0[0] + x0[1] + x0[2] + x0[3] + x1[0] + x1[1] + x1[2] + x1[3]) * a0 +
                                  (x2[0] + x2[1] + x2[2] + x2[3] + x3[0] + x3[1] + x3[2] + x3[3]) * a1;
                        acc[r] += p * float(S[row * (cols / 16) + c]);
                    }
                }
            } else if constexpr (FMT == 0 && DIAG == 2) {
                for (uint32_t c = l; c < cols / 16; c += SG) {
                    const float xs = float(c);
#pragma unroll
                    for (uint32_t r = 0; r < RPS; ++r) {
                        const uint64_t row = e * rows + sycl::min(rb + r, rows - 1);
                        const sycl::uint2 v = reinterpret_cast<const sycl::uint2*>(W + row * (cols / 2))[c];
                        const uint32_t u0 = v[0], u1 = v[1];
                        float p = 0.f;
                        for (int j = 0; j < 8; ++j) p += xs * i2_e2m1x2(u0 >> (4 * j)) + xs * i2_e2m1x2(u1 >> (4 * j));
                        acc[r] += p * i2_e4m3_fin(S[row * (cols / 16) + c]);
                    }
                }
            } else if constexpr (FMT == 0) {
                for (uint32_t c = l; c < cols / 16; c += SG) {
                    const sycl::float4* xb = reinterpret_cast<const sycl::float4*>(xv + 16 * c);
                    const sycl::float4 x0 = xb[0], x1 = xb[1], x2 = xb[2], x3 = xb[3];
#pragma unroll
                    for (uint32_t r = 0; r < RPS; ++r) {
                        if (rb + r >= rows) break;
                        const uint64_t row = e * rows + rb + r;
                        const sycl::uint2 v = reinterpret_cast<const sycl::uint2*>(W + row * (cols / 2))[c];
                        const uint32_t u0 = v[0], u1 = v[1];
                        float p = 0.f;
                        p += x0[0] * i2_e2m1x2(u0) + x0[1] * i2_e2m1x2(u0 >> 4) + x0[2] * i2_e2m1x2(u0 >> 8) +
                             x0[3] * i2_e2m1x2(u0 >> 12) + x1[0] * i2_e2m1x2(u0 >> 16) + x1[1] * i2_e2m1x2(u0 >> 20) +
                             x1[2] * i2_e2m1x2(u0 >> 24) + x1[3] * i2_e2m1x2(u0 >> 28);
                        p += x2[0] * i2_e2m1x2(u1) + x2[1] * i2_e2m1x2(u1 >> 4) + x2[2] * i2_e2m1x2(u1 >> 8) +
                             x2[3] * i2_e2m1x2(u1 >> 12) + x3[0] * i2_e2m1x2(u1 >> 16) + x3[1] * i2_e2m1x2(u1 >> 20) +
                             x3[2] * i2_e2m1x2(u1 >> 24) + x3[3] * i2_e2m1x2(u1 >> 28);
                        acc[r] += p * i2_e4m3_fin(S[row * (cols / 16) + c]);
                    }
                }
                const float f = 0.5f * fac[e];
#pragma unroll
                for (uint32_t r = 0; r < RPS; ++r) acc[r] *= f;
            } else {
                for (uint32_t c = l; c < cols / 16; c += SG) {
                    const sycl::float4* xb = reinterpret_cast<const sycl::float4*>(xv + 16 * c);
                    const sycl::float4 xa[4] = {xb[0], xb[1], xb[2], xb[3]};
#pragma unroll
                    for (uint32_t r = 0; r < RPS; ++r) {
                        if (rb + r >= rows) break;
                        const uint64_t row = e * rows + rb + r;
                        const sycl::uint4 v = reinterpret_cast<const sycl::uint4*>(W + row * cols)[c];
                        float p = 0.f;
                        for (int j = 0; j < 4; ++j) {
                            const uint32_t u = v[j];
                            p += xa[j][0] * i2_e4m3_fin(u & 255u) + xa[j][1] * i2_e4m3_fin((u >> 8) & 255u) +
                                 xa[j][2] * i2_e4m3_fin((u >> 16) & 255u) + xa[j][3] * i2_e4m3_fin(u >> 24);
                        }
                        acc[r] += p * i2_ue8m0(S[row * (cols / 32) + (c >> 1)]);
                    }
                }
            }
            if constexpr (RPS % 2 == 0) {
                if (act) {                                              // rows 2j / 2j + 1 = gate / up of unit j
#pragma unroll
                    for (uint32_t r = 0; r < RPS; r += 2) {
                        if (rb + r >= rows) break;
                        const float g2 = sycl::reduce_over_group(sg, acc[r], sycl::plus<float>());
                        const float u2 = sycl::reduce_over_group(sg, acc[r + 1], sycl::plus<float>());
                        if (l == 0) act[slot * (rows / 2) + (rb + r) / 2] = i2_swiglu_cap(g2, u2, gcap, ucap);
                    }
                    return;
                }
            }
#pragma unroll
            for (uint32_t r = 0; r < RPS; ++r) {
                if (rb + r >= rows) break;
                const float s2 = sycl::reduce_over_group(sg, acc[r], sycl::plus<float>());
                if (l == 0) y[slot * rows + rb + r] = i2r(s2, round);
            }
        });
    });
}
}  // namespace

sycl::event i2_moe_group(sycl::queue& q, const int32_t* used, uint32_t T, uint32_t k, uint32_t E, uint32_t tile,
                         int32_t* order, int32_t* tiles, int32_t* ntiles) {
    constexpr uint32_t WG = 256;
    const uint32_t TK = T * k;
    if (TK <= WG && E <= WG) {
        // Batched decoding (few slots): the per-expert offsets by group scans, each slot's place among its expert's
        // slots by counting the earlier ones (ascending slots per expert, as below); the same arrays, no serial walk.
        return ie::ps(q, "i2_moe_group_scan", [&](sycl::handler& h) {
            sycl::local_accessor<uint32_t, 1> cnt(sycl::range<1>(WG), h);
            sycl::local_accessor<uint32_t, 1> st(sycl::range<1>(WG), h);
            sycl::local_accessor<uint32_t, 1> ts(sycl::range<1>(WG), h);
            sycl::local_accessor<int32_t, 1> ul(sycl::range<1>(WG), h);
            h.parallel_for(sycl::nd_range<1>(WG, WG), [=](sycl::nd_item<1> it) {
                const auto grp = it.get_group();
                const uint32_t l = uint32_t(it.get_local_id(0));
                cnt[l] = 0;
                const int32_t e = l < TK ? used[l] : -1;
                ul[l] = e;
                sycl::group_barrier(grp);
                if (e >= 0)
                    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                                     sycl::access::address_space::local_space>(cnt[uint32_t(e)]).fetch_add(1u);
                sycl::group_barrier(grp);
                const uint32_t c = l < E ? cnt[l] : 0u;
                const uint32_t nt = (c + tile - 1) / tile;
                const uint32_t s0 = sycl::exclusive_scan_over_group(grp, c, sycl::plus<uint32_t>());
                const uint32_t t0 = sycl::exclusive_scan_over_group(grp, nt, sycl::plus<uint32_t>());
                const uint32_t tot = sycl::reduce_over_group(grp, nt, sycl::plus<uint32_t>());
                st[l] = s0; ts[l] = t0;
                if (l == 0) ntiles[0] = int32_t(tot);
                sycl::group_barrier(grp);
                if (e >= 0) {
                    uint32_t rank = 0;
                    for (uint32_t j = 0; j < l; ++j) rank += ul[j] == e ? 1u : 0u;
                    order[st[uint32_t(e)] + rank] = int32_t(l);
                }
                if (l < E && c > 0)
                    for (uint32_t j = 0; j * tile < c; ++j) {
                        int32_t* tp = tiles + uint64_t(t0 + j) * 3;
                        tp[0] = int32_t(l); tp[1] = int32_t(s0 + j * tile); tp[2] = int32_t(sycl::min(tile, c - j * tile));
                    }
            });
        });
    }
    return ie::ps(q, "i2_moe_group", [&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> cnt(sycl::range<1>(WG), h);
        sycl::local_accessor<uint32_t, 1> st(sycl::range<1>(WG), h);
        sycl::local_accessor<uint32_t, 1> ts(sycl::range<1>(WG + 1), h);
        h.parallel_for(sycl::nd_range<1>(WG, WG), [=](sycl::nd_item<1> it) {
            const uint32_t l = uint32_t(it.get_local_id(0));
            cnt[l] = 0;
            sycl::group_barrier(it.get_group());
            for (uint32_t s2 = l; s2 < TK; s2 += WG) {
                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                                 sycl::access::address_space::local_space>(cnt[uint32_t(used[s2])]).fetch_add(1u);
            }
            sycl::group_barrier(it.get_group());
            if (l == 0) {
                uint32_t a = 0, b2 = 0;
                for (uint32_t e = 0; e < E; ++e) { st[e] = a; ts[e] = b2; a += cnt[e]; b2 += (cnt[e] + tile - 1) / tile; }
                ts[E] = b2;
                ntiles[0] = int32_t(b2);
            }
            sycl::group_barrier(it.get_group());
            if (l < E && cnt[l] > 0) {
                uint32_t pos = st[l];
                for (uint32_t s2 = 0; s2 < TK; ++s2)
                    if (uint32_t(used[s2]) == l) order[pos++] = int32_t(s2);       // ascending slots per expert
                const uint32_t c = cnt[l];
                for (uint32_t j = 0; j * tile < c; ++j) {
                    int32_t* tp = tiles + uint64_t(ts[l] + j) * 3;
                    tp[0] = int32_t(l); tp[1] = int32_t(st[l] + j * tile); tp[2] = int32_t(sycl::min(tile, c - j * tile));
                }
            }
        });
    });
}

namespace {
template <int FMT, uint32_t TT, uint32_t RPS, int DEC = 1>
sycl::event i2_expert_grouped_sg(sycl::queue& q, const I2ExpertBank& b, const float* x, bool x_per_slot, uint32_t k,
                                 const int32_t* order, const int32_t* tiles, const int32_t* ntiles, uint32_t maxtiles,
                                 float* y, bool round, float* act = nullptr, float gcap = 0.f, float ucap = 0.f) {
    // Like i2_expert_matvec_sg (one row's sum formed per lane chunk in the same order, so each slot's value is
    // bit-identical to the few-token path), but a tile of up to TT slots routed to the same expert shares each weight
    // load and decode, and each sub-group takes RPS rows so each x load serves RPS rows.
    constexpr uint32_t SG = 16, NSG = 8, RPG = NSG * RPS;
    const uint32_t rows = b.rows, cols = b.cols;
    const uint32_t rgroups = (rows + RPG - 1) / RPG;
    const uint8_t* W = b.w;
    const uint8_t* S = b.s;
    const float* fac = b.fac;
    return ie::ps(q, "i2_expert_grouped", [&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(uint64_t(maxtiles) * rgroups * NSG * SG, NSG * SG),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const uint64_t g = it.get_group(0);
            const uint32_t ti = uint32_t(g / rgroups);
            if (ti >= uint32_t(ntiles[0])) return;
            const auto sg = it.get_sub_group();
            const uint32_t r0 = uint32_t(g % rgroups) * RPG + uint32_t(sg.get_group_linear_id()) * RPS;
            if (r0 >= rows) return;
            const uint32_t l = uint32_t(sg.get_local_linear_id());
            const uint64_t e = uint64_t(tiles[ti * 3]);
            const uint32_t o0 = uint32_t(tiles[ti * 3 + 1]), nt = uint32_t(tiles[ti * 3 + 2]);
            const float* xs[TT];
            uint32_t sl[TT];
            float acc[RPS][TT];
#pragma unroll
            for (uint32_t i = 0; i < TT; ++i) {
                sl[i] = i < nt ? uint32_t(order[o0 + i]) : 0u;
                xs[i] = x + (x_per_slot ? uint64_t(sl[i]) : uint64_t(sl[i] / k)) * cols;
#pragma unroll
                for (uint32_t r = 0; r < RPS; ++r) acc[r][i] = 0.f;
            }
            uint64_t rowv[RPS];
#pragma unroll
            for (uint32_t r = 0; r < RPS; ++r) rowv[r] = e * rows + sycl::min(r0 + r, rows - 1);   // a tail row repeats (unused)
            if constexpr (FMT == 0) {
                for (uint32_t c = l; c < cols / 16; c += SG) {
                    float w0[RPS][8], w1[RPS][8], sc[RPS];
#pragma unroll
                    for (uint32_t r = 0; r < RPS; ++r) {
                        const sycl::uint2 v = reinterpret_cast<const sycl::uint2*>(W + rowv[r] * (cols / 2))[c];
                        const uint32_t u0 = v[0], u1 = v[1];
                        if constexpr (DEC == 1) { i2_nv4_dec8(u0, w0[r]); i2_nv4_dec8(u1, w1[r]); }
                        else {
#pragma unroll
                            for (int j = 0; j < 8; ++j) { w0[r][j] = i2_e2m1x2(u0 >> (4 * j)); w1[r][j] = i2_e2m1x2(u1 >> (4 * j)); }
                        }
                        sc[r] = i2_e4m3_fin(S[rowv[r] * (cols / 16) + c]);
                    }
#pragma unroll
                    for (uint32_t i = 0; i < TT; ++i) {
                        if (i >= nt) break;
                        const sycl::float4* xb = reinterpret_cast<const sycl::float4*>(xs[i] + 16 * c);
                        const sycl::float4 x0 = xb[0], x1 = xb[1], x2 = xb[2], x3 = xb[3];
#pragma unroll
                        for (uint32_t r = 0; r < RPS; ++r) {
                            float p = 0.f;
                            p += x0[0] * w0[r][0] + x0[1] * w0[r][1] + x0[2] * w0[r][2] + x0[3] * w0[r][3] + x1[0] * w0[r][4] +
                                 x1[1] * w0[r][5] + x1[2] * w0[r][6] + x1[3] * w0[r][7];
                            p += x2[0] * w1[r][0] + x2[1] * w1[r][1] + x2[2] * w1[r][2] + x2[3] * w1[r][3] + x3[0] * w1[r][4] +
                                 x3[1] * w1[r][5] + x3[2] * w1[r][6] + x3[3] * w1[r][7];
                            acc[r][i] += p * sc[r];
                        }
                    }
                }
                const float f = DEC == 1 ? 16384.f * fac[e] : 0.5f * fac[e];
#pragma unroll
                for (uint32_t r = 0; r < RPS; ++r)
#pragma unroll
                    for (uint32_t i = 0; i < TT; ++i) acc[r][i] *= f;
            } else {
                for (uint32_t c = l; c < cols / 16; c += SG) {
                    float wv[RPS][16], sc[RPS];
#pragma unroll
                    for (uint32_t r = 0; r < RPS; ++r) {
                        const sycl::uint4 v = reinterpret_cast<const sycl::uint4*>(W + rowv[r] * cols)[c];
                        if constexpr (DEC == 1) {
#pragma unroll
                            for (int j = 0; j < 4; ++j) i2_e4m3_dec4(v[j], &wv[r][4 * j]);
                        } else {
#pragma unroll
                            for (int j = 0; j < 4; ++j)
#pragma unroll
                                for (int bb = 0; bb < 4; ++bb) wv[r][4 * j + bb] = i2_e4m3_fin((v[j] >> (8 * bb)) & 255u);
                        }
                        sc[r] = i2_ue8m0(S[rowv[r] * (cols / 32) + (c >> 1)]);
                    }
#pragma unroll
                    for (uint32_t i = 0; i < TT; ++i) {
                        if (i >= nt) break;
                        const sycl::float4* xb = reinterpret_cast<const sycl::float4*>(xs[i] + 16 * c);
                        const sycl::float4 xa[4] = {xb[0], xb[1], xb[2], xb[3]};
#pragma unroll
                        for (uint32_t r = 0; r < RPS; ++r) {
                            float p = 0.f;
                            for (int j = 0; j < 4; ++j)
                                p += xa[j][0] * wv[r][4 * j] + xa[j][1] * wv[r][4 * j + 1] + xa[j][2] * wv[r][4 * j + 2] +
                                     xa[j][3] * wv[r][4 * j + 3];
                            acc[r][i] += p * sc[r];
                        }
                    }
                }
                if constexpr (DEC == 1) {
#pragma unroll
                    for (uint32_t r = 0; r < RPS; ++r)
#pragma unroll
                        for (uint32_t i = 0; i < TT; ++i) acc[r][i] *= 256.f;
                }
            }
            if constexpr (RPS == 2) {
                if (act) {
#pragma unroll
                    for (uint32_t i = 0; i < TT; ++i) {
                        if (i >= nt) break;
                        const float g2 = sycl::reduce_over_group(sg, acc[0][i], sycl::plus<float>());
                        const float u2 = sycl::reduce_over_group(sg, acc[1][i], sycl::plus<float>());
                        if (l == 0) act[uint64_t(sl[i]) * (rows / 2) + r0 / 2] = i2_swiglu_cap(g2, u2, gcap, ucap);
                    }
                    return;
                }
            }
#pragma unroll
            for (uint32_t r = 0; r < RPS; ++r) {
                if (r0 + r >= rows) break;
#pragma unroll
                for (uint32_t i = 0; i < TT; ++i) {
                    if (i >= nt) break;
                    const float s2 = sycl::reduce_over_group(sg, acc[r][i], sycl::plus<float>());
                    if (l == 0) y[uint64_t(sl[i]) * rows + r0 + r] = i2r(s2, round);
                }
            }
        });
    });
}
}  // namespace

namespace {
// Prompt MoE on the matrix engines (2026-11-23): one work-group = one tile of up to 8 * MR slots routed to the same
// expert x 128 output rows; per 32-column step it stages the slots' inputs as fp16 (SPLIT: plus the fp16 remainder
// x - fp16(x), a second product pass, ~fp32 inputs) and the 128 rows' NVFP4 weights decoded to fp16 (e2m1 value x E4M3
// block scale: <= 8 significant bits, exact in fp16), then joint_matrix 8 x 16 x 16 products into fp32 accumulators;
// the per-expert factor at the store. y[slot][row] as i2_expert_grouped (fp32 sums, a different order).
// MXFP8 banks (FMT 1): E4M3 x 2^(e - 127) staged as fp16 times 2^10 (weights stay far inside fp16's normal range),
// the factor 2^-10 at the store.
template <uint32_t MR, bool SPLIT, int FMT = 0>
sycl::event i2_expert_xmx_nv4(sycl::queue& q, const I2ExpertBank& b, const float* x, bool x_per_slot, uint32_t k,
                              const int32_t* order, const int32_t* tiles, const int32_t* ntiles, uint32_t maxtiles,
                              float* y) {
    namespace mat = sycl::ext::oneapi::experimental::matrix;
    namespace imat = sycl::ext::intel::experimental::matrix;
    constexpr uint32_t SG = 16, BN = 128, BK = 32, TM = 8, TN = 16, TK = 16, MT = TM * MR, WG = 128;
    constexpr uint32_t VALS = MT * BK / WG, LPR = WG / MT;              // A values per lane, lanes per slot row
    static_assert(VALS % 4 == 0, "A staged as float4s");
    const uint32_t rows = b.rows, cols = b.cols, nb = rows / BN;
    const uint8_t* W = b.w;
    const uint8_t* S = b.s;
    const float* fac = b.fac;
    return ie::ps(q, "i2_expert_xmx", [&](sycl::handler& h) {
        sycl::local_accessor<sycl::half, 1> Ah(sycl::range<1>(MT * BK), h);
        sycl::local_accessor<sycl::half, 1> Al(sycl::range<1>(SPLIT ? MT * BK : 1), h);
        sycl::local_accessor<sycl::half, 1> Bs(sycl::range<1>(BK * BN), h);
        h.parallel_for(sycl::nd_range<1>(uint64_t(maxtiles) * nb * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const uint32_t g = uint32_t(it.get_group(0));
            const uint32_t ti = g / nb, n0 = (g % nb) * BN;
            if (ti >= uint32_t(ntiles[0])) return;
            const uint32_t lid = uint32_t(it.get_local_id(0)), sgid = lid / SG;
            auto sg = it.get_sub_group();
            const uint64_t e = uint64_t(tiles[ti * 3]);
            const uint32_t o0 = uint32_t(tiles[ti * 3 + 1]), nt = uint32_t(tiles[ti * 3 + 2]);
            const uint32_t ar = lid / LPR, ac = (lid % LPR) * VALS;
            const bool a_ok = ar < nt;
            const uint32_t aslot = a_ok ? uint32_t(order[o0 + ar]) : 0u;
            const float* xrow = x + (x_per_slot ? uint64_t(aslot) : uint64_t(aslot / k)) * cols;
            const uint64_t wr = e * rows + n0 + lid;
            const uint32_t* wrow = reinterpret_cast<const uint32_t*>(W + wr * (FMT == 0 ? cols / 2 : cols));
            const uint8_t* srow = S + wr * (FMT == 0 ? cols / 16 : cols / 32);
            mat::joint_matrix<sycl::sub_group, float, mat::use::accumulator, TM, TN> acc[MR];
#pragma unroll
            for (uint32_t mr = 0; mr < MR; ++mr) mat::joint_matrix_fill(sg, acc[mr], 0.0f);
            auto Ahp = Ah.template get_multi_ptr<sycl::access::decorated::no>();
            auto Alp = Al.template get_multi_ptr<sycl::access::decorated::no>();
            auto Bp = Bs.template get_multi_ptr<sycl::access::decorated::no>();
            for (uint32_t kb = 0; kb < cols / BK; ++kb) {
#pragma unroll
                for (uint32_t v4 = 0; v4 < VALS / 4; ++v4) {
                    const sycl::float4 xv = a_ok ? *reinterpret_cast<const sycl::float4*>(xrow + kb * BK + ac + 4 * v4) : sycl::float4(0.f);
#pragma unroll
                    for (uint32_t i = 0; i < 4; ++i) {
                        const sycl::half hi = sycl::half(xv[i]);
                        Ah[ar * BK + ac + 4 * v4 + i] = hi;
                        if constexpr (SPLIT) Al[ar * BK + ac + 4 * v4 + i] = sycl::half(xv[i] - float(hi));
                    }
                }
                if constexpr (FMT == 0) {
                    const sycl::uint4 wv = reinterpret_cast<const sycl::uint4*>(wrow)[kb];
                    const float s0 = i2_e4m3_fin(srow[2 * kb]) * 16384.f, s1 = i2_e4m3_fin(srow[2 * kb + 1]) * 16384.f;
#pragma unroll
                    for (uint32_t wi = 0; wi < 4; ++wi) {
                        float w8[8];
                        i2_nv4_dec8(wv[wi], w8);
                        const float sc = wi < 2 ? s0 : s1;
#pragma unroll
                        for (uint32_t j = 0; j < 8; ++j) Bs[(8 * wi + j) * BN + lid] = sycl::half(w8[j] * sc);
                    }
                } else {
                    const sycl::uint4 wa = reinterpret_cast<const sycl::uint4*>(wrow)[2 * kb];
                    const sycl::uint4 wb = reinterpret_cast<const sycl::uint4*>(wrow)[2 * kb + 1];
                    const float sc = i2_ue8m0(srow[kb]) * 262144.f;              // 2^8 (the decode) x 2^10
#pragma unroll
                    for (uint32_t wi = 0; wi < 8; ++wi) {
                        float w4[4];
                        i2_e4m3_dec4(wi < 4 ? wa[wi] : wb[wi - 4], w4);
#pragma unroll
                        for (uint32_t j = 0; j < 4; ++j) Bs[(4 * wi + j) * BN + lid] = sycl::half(w4[j] * sc);
                    }
                }
                sycl::group_barrier(it.get_group());
#pragma unroll
                for (uint32_t kk = 0; kk < BK; kk += TK) {
                    mat::joint_matrix<sycl::sub_group, sycl::half, mat::use::b, TK, TN, mat::layout::row_major> bt;
                    mat::joint_matrix_load(sg, bt, Bp + kk * BN + sgid * TN, BN);
#pragma unroll
                    for (uint32_t mr = 0; mr < MR; ++mr) {
                        if (mr * TM >= nt) continue;
                        mat::joint_matrix<sycl::sub_group, sycl::half, mat::use::a, TM, TK, mat::layout::row_major> at;
                        mat::joint_matrix_load(sg, at, Ahp + (mr * TM) * BK + kk, BK);
                        mat::joint_matrix_mad(sg, acc[mr], at, bt, acc[mr]);
                        if constexpr (SPLIT) {
                            mat::joint_matrix_load(sg, at, Alp + (mr * TM) * BK + kk, BK);
                            mat::joint_matrix_mad(sg, acc[mr], at, bt, acc[mr]);
                        }
                    }
                }
                sycl::group_barrier(it.get_group());
            }
            const float f = FMT == 0 ? fac[e] : 0.0009765625f;
            const uint32_t c0 = n0 + sgid * TN;
#pragma unroll
            for (uint32_t mr = 0; mr < MR; ++mr) {
                if (mr * TM >= nt) continue;
                const uint32_t rb = mr * TM;
                imat::joint_matrix_apply(sg, acc[mr], [&](float& v, size_t r, size_t c) {
                    if (rb + r < nt) y[uint64_t(order[o0 + rb + r]) * rows + c0 + c] = v * f;
                });
            }
        });
    });
}
}  // namespace

namespace {
inline uint32_t i2_fkey(float f) {                          // order-preserving float -> uint32
    const uint32_t u = sycl::bit_cast<uint32_t>(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
inline float i2_gumbel(uint64_t seed, uint32_t i) {        // splitmix64 of (seed, i) -> Gumbel(0, 1)
    uint64_t z = seed + 0x9E3779B97F4A7C15ull * (uint64_t(i) + 1);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    const float u = (float(uint32_t(z >> 40)) + 0.5f) * (1.0f / 16777216.0f);   // (0, 1)
    return -sycl::log(-sycl::log(u));
}
}  // namespace

sycl::event i2_sample_rows(sycl::queue& q, float* logits, uint32_t n, uint32_t V, const I2SampleRow* rows, const int32_t* recent,
                           const int32_t* pf_id, const float* pf_d, int32_t* cand, int32_t* out) {
    constexpr uint32_t WG = 512;
    return ie::ps(q, "i2_sample_rows", [&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> cnt(sycl::range<1>(1), h);
        h.parallel_for(sycl::nd_range<1>(uint64_t(n) * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const auto grp = it.get_group();
            const uint32_t r = uint32_t(it.get_group(0)), lid = uint32_t(it.get_local_id(0));
            const I2SampleRow P = rows[r];
            float* lg = logits + uint64_t(r) * V;
            int32_t* cd = cand + uint64_t(r) * V;
            if (lid == 0) {
                for (uint32_t k = 0; k < P.pf_n; ++k) {
                    const int32_t t = pf_id[P.pf_off + k];
                    if (t >= 0 && uint32_t(t) < V) lg[t] -= pf_d[P.pf_off + k];
                }
                if (P.rep != 1.f)
                    for (uint32_t k = 0; k < P.rec_n; ++k) {
                        const int32_t t = recent[P.rec_off + k];
                        if (t < 0 || uint32_t(t) >= V) continue;
                        const float v = lg[t];
                        lg[t] = v > 0 ? v / P.rep : v * P.rep;
                    }
                cnt[0] = 0;
            }
            sycl::group_barrier(grp, sycl::memory_scope::work_group);
            sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::work_group);
            float mx = -3.0e38f; uint32_t am = 0xFFFFFFFFu; bool bad = false;
            for (uint32_t i = lid; i < V; i += WG) {
                const float v = lg[i];
                if ((sycl::bit_cast<uint32_t>(v) & 0x7F800000u) == 0x7F800000u) bad = true;
                else if (v > mx) { mx = v; am = i; }
            }
            const float gmx = sycl::reduce_over_group(grp, mx, sycl::maximum<float>());
            const uint32_t gid = sycl::reduce_over_group(grp, mx == gmx ? am : 0xFFFFFFFFu, sycl::minimum<uint32_t>());
            const bool gbad = sycl::any_of_group(grp, bad);
            if (gbad || gid == 0xFFFFFFFFu) { if (lid == 0) out[r] = -1; return; }
            if (!(P.temp > 0.f)) { if (lid == 0) out[r] = int32_t(gid); return; }
            const float invT = 1.f / P.temp, mxs = gmx * invT;
            const uint32_t gkey = i2_fkey(gmx);
            uint32_t kt = 0;
            if (P.top_k && P.top_k < V) {                      // the largest key t with #(key >= t) >= k
                uint32_t lo = 0, hi = gkey;
                while (lo < hi) {
                    const uint32_t mid = lo + (hi - lo + 1) / 2;
                    uint32_t c = 0;
                    for (uint32_t i = lid; i < V; i += WG) c += i2_fkey(lg[i]) >= mid ? 1u : 0u;
                    c = sycl::reduce_over_group(grp, c, sycl::plus<uint32_t>());
                    if (c >= P.top_k) lo = mid; else hi = mid - 1;
                }
                kt = lo;
            }
            for (uint32_t i = lid; i < V; i += WG) {           // the candidates
                const float v = lg[i];
                if (i2_fkey(v) >= kt && v * invT - mxs > -25.f) {
                    const uint32_t at = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                                                         sycl::access::address_space::local_space>(cnt[0]).fetch_add(1u);
                    cd[at] = int32_t(i);
                }
            }
            sycl::group_barrier(grp, sycl::memory_scope::work_group);
            sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::work_group);
            const uint32_t nc = cnt[0];
            uint32_t kp = 0;
            if (P.top_p < 1.f) {
                float z = 0.f; uint32_t kmin = 0xFFFFFFFFu;
                for (uint32_t c = lid; c < nc; c += WG) { const float v = lg[cd[c]]; z += sycl::exp(v * invT - mxs); kmin = sycl::min(kmin, i2_fkey(v)); }
                z = sycl::reduce_over_group(grp, z, sycl::plus<float>());
                kmin = sycl::reduce_over_group(grp, kmin, sycl::minimum<uint32_t>());
                const float target = P.top_p * z;
                uint32_t lo = kmin, hi = gkey;                 // the largest key t with mass(key >= t) >= target
                while (lo < hi) {
                    const uint32_t mid = lo + (hi - lo + 1) / 2;
                    float sm = 0.f;
                    for (uint32_t c = lid; c < nc; c += WG) { const float v = lg[cd[c]]; if (i2_fkey(v) >= mid) sm += sycl::exp(v * invT - mxs); }
                    sm = sycl::reduce_over_group(grp, sm, sycl::plus<float>());
                    if (sm >= target) lo = mid; else hi = mid - 1;
                }
                kp = lo;
            }
            const float lnmp = P.min_p > 0.f ? sycl::log(P.min_p) : -3.0e38f;
            float best = -3.0e38f; uint32_t bi = 0xFFFFFFFFu;
            for (uint32_t c = lid; c < nc; c += WG) {
                const uint32_t i = uint32_t(cd[c]);
                const float v = lg[i];
                if (i2_fkey(v) < kp || v * invT - mxs < lnmp) continue;
                const float sc = v * invT + i2_gumbel(P.seed, i);
                if (sc > best || (sc == best && i < bi)) { best = sc; bi = i; }
            }
            const float gb = sycl::reduce_over_group(grp, best, sycl::maximum<float>());
            const uint32_t gi = sycl::reduce_over_group(grp, best == gb ? bi : 0xFFFFFFFFu, sycl::minimum<uint32_t>());
            if (lid == 0) out[r] = gi == 0xFFFFFFFFu ? int32_t(gid) : int32_t(gi);
        });
    });
}

bool i2_expert_xmx_ok(const I2ExpertBank& b) {
    return (b.fmt == 0 || b.fmt == 1) && b.rows % 128 == 0 && b.cols % 32 == 0 && (reinterpret_cast<uintptr_t>(b.w) & 15) == 0;
}

sycl::event i2_expert_xmx(sycl::queue& q, const I2ExpertBank& b, const float* x, bool x_per_slot, uint32_t k,
                          const int32_t* order, const int32_t* tiles, const int32_t* ntiles, uint32_t maxtiles,
                          uint32_t tile, bool split, float* y) {
#define I2X(MR, SP, F) i2_expert_xmx_nv4<MR, SP, F>(q, b, x, x_per_slot, k, order, tiles, ntiles, maxtiles, y)
#define I2XF(MR, SP) (b.fmt == 0 ? I2X(MR, SP, 0) : I2X(MR, SP, 1))
    if (tile == 32) return split ? I2XF(4, true) : I2XF(4, false);
    if (tile == 64) return split ? I2XF(8, true) : I2XF(8, false);
    return split ? I2XF(2, true) : I2XF(2, false);
#undef I2XF
#undef I2X
}

sycl::event i2_expert_w13_act(sycl::queue& q, const I2ExpertBank& b, const float* x, const int32_t* idx, uint32_t T,
                              uint32_t k, float* act, float gcap, float ucap) {
    static const bool wide = std::getenv("IE_INKLING2_EXPERT_WIDE") != nullptr;
    static const int nsg = [] { const char* e = std::getenv("IE_INKLING2_EXPERT_NSG"); return e ? std::atoi(e) : 8; }();
    static const int diag = [] { const char* e = std::getenv("IE_INKLING2_EXPERT_DIAG"); return e ? std::atoi(e) : 0; }();
    static const int dec = [] { const char* e = std::getenv("IE_INKLING2_EXPERT_DEC"); return e ? std::atoi(e) : 1; }();
    static const int r13 = [] { const char* e = std::getenv("IE_INKLING2_EXPERT_RPS13"); return e ? std::atoi(e) : 2; }();
    static const int u13 = [] { const char* e = std::getenv("IE_INKLING2_EXPERT_U13"); return e ? std::atoi(e) : 3; }();
    static const int flat = [] { const char* e = std::getenv("IE_INKLING2_EXPERT_FLAT"); return e ? std::atoi(e) : 0; }();
    if (flat && b.fmt == 0 && (2 * (b.cols / 32)) % 16 == 0 && b.rows % 2 == 0) {
        if (flat == 5 && b.cols == 2304) return i2_expert_flat_nv4<2, 3, 72>(q, b, x, false, idx, T, k, nullptr, false, act, gcap, ucap);
        if (flat == 6 && b.cols == 2304) return i2_expert_flat_nv4<2, 1, 72>(q, b, x, false, idx, T, k, nullptr, false, act, gcap, ucap);
        if (flat == 3) return i2_expert_flat_nv4<2, 3>(q, b, x, false, idx, T, k, nullptr, false, act, gcap, ucap);
        if (flat == 1) return i2_expert_flat_nv4<2, 1>(q, b, x, false, idx, T, k, nullptr, false, act, gcap, ucap);
        if (flat == 4) return i2_expert_flat_nv4<4, 3>(q, b, x, false, idx, T, k, nullptr, false, act, gcap, ucap);
        return i2_expert_flat_nv4<2, 2>(q, b, x, false, idx, T, k, nullptr, false, act, gcap, ucap);
    }
    if (dec == 1 && !wide && !diag && nsg == 8) {
#define I2_W13(R, UU) if (r13 == R && u13 == UU) return b.fmt == 0 \
            ? i2_expert_matvec_sg<0, R, false, 8, 0, 1, UU>(q, b, x, false, idx, T, k, nullptr, false, act, gcap, ucap) \
            : i2_expert_matvec_sg<1, R, false, 8, 0, 1, UU>(q, b, x, false, idx, T, k, nullptr, false, act, gcap, ucap);
        I2_W13(2, 1) I2_W13(2, 2) I2_W13(2, 3) I2_W13(2, 4) I2_W13(4, 1) I2_W13(4, 2) I2_W13(4, 3)
#undef I2_W13
        return b.fmt == 0 ? i2_expert_matvec_sg<0, 2, false, 8, 0, 1>(q, b, x, false, idx, T, k, nullptr, false, act, gcap, ucap)
                          : i2_expert_matvec_sg<1, 2, false, 8, 0, 1>(q, b, x, false, idx, T, k, nullptr, false, act, gcap, ucap);
    }
    if (wide && b.fmt == 0) return i2_expert_matvec_sg<0, 2, true>(q, b, x, false, idx, T, k, nullptr, false, act, gcap, ucap);
    if (b.fmt == 0 && diag == 1) return i2_expert_matvec_sg<0, 2, false, 8, 1>(q, b, x, false, idx, T, k, nullptr, false, act, gcap, ucap);
    if (b.fmt == 0 && diag == 2) return i2_expert_matvec_sg<0, 2, false, 8, 2>(q, b, x, false, idx, T, k, nullptr, false, act, gcap, ucap);
    if (nsg == 4) return b.fmt == 0 ? i2_expert_matvec_sg<0, 2, false, 4>(q, b, x, false, idx, T, k, nullptr, false, act, gcap, ucap)
                                    : i2_expert_matvec_sg<1, 2, false, 4>(q, b, x, false, idx, T, k, nullptr, false, act, gcap, ucap);
    if (nsg == 16) return b.fmt == 0 ? i2_expert_matvec_sg<0, 2, false, 16>(q, b, x, false, idx, T, k, nullptr, false, act, gcap, ucap)
                                     : i2_expert_matvec_sg<1, 2, false, 16>(q, b, x, false, idx, T, k, nullptr, false, act, gcap, ucap);
    return b.fmt == 0 ? i2_expert_matvec_sg<0, 2>(q, b, x, false, idx, T, k, nullptr, false, act, gcap, ucap)
                      : i2_expert_matvec_sg<1, 2>(q, b, x, false, idx, T, k, nullptr, false, act, gcap, ucap);
}
sycl::event i2_expert_grouped_w13_act(sycl::queue& q, const I2ExpertBank& b, const float* x, uint32_t k,
                                      const int32_t* order, const int32_t* tiles, const int32_t* ntiles, uint32_t maxtiles,
                                      float* act, float gcap, float ucap) {
    return b.fmt == 0 ? i2_expert_grouped_sg<0, kI2MoeTile, 2>(q, b, x, false, k, order, tiles, ntiles, maxtiles, nullptr, false, act, gcap, ucap)
                      : i2_expert_grouped_sg<1, kI2MoeTile, 2>(q, b, x, false, k, order, tiles, ntiles, maxtiles, nullptr, false, act, gcap, ucap);
}

sycl::event i2_expert_grouped(sycl::queue& q, const I2ExpertBank& b, const float* x, bool x_per_slot, uint32_t k,
                              const int32_t* order, const int32_t* tiles, const int32_t* ntiles, uint32_t maxtiles,
                              float* y, bool round) {
    static const uint32_t rps = [] { const char* e = std::getenv("IE_INKLING2_GROUPED_RPS"); return e ? uint32_t(std::atoi(e)) : 2u; }();
    if (rps == 1)
        return b.fmt == 0 ? i2_expert_grouped_sg<0, kI2MoeTile, 1>(q, b, x, x_per_slot, k, order, tiles, ntiles, maxtiles, y, round)
                          : i2_expert_grouped_sg<1, kI2MoeTile, 1>(q, b, x, x_per_slot, k, order, tiles, ntiles, maxtiles, y, round);
    if (rps == 4)
        return b.fmt == 0 ? i2_expert_grouped_sg<0, kI2MoeTile, 4>(q, b, x, x_per_slot, k, order, tiles, ntiles, maxtiles, y, round)
                          : i2_expert_grouped_sg<1, kI2MoeTile, 4>(q, b, x, x_per_slot, k, order, tiles, ntiles, maxtiles, y, round);
    return b.fmt == 0 ? i2_expert_grouped_sg<0, kI2MoeTile, 2>(q, b, x, x_per_slot, k, order, tiles, ntiles, maxtiles, y, round)
                      : i2_expert_grouped_sg<1, kI2MoeTile, 2>(q, b, x, x_per_slot, k, order, tiles, ntiles, maxtiles, y, round);
}

sycl::event i2_expert_matvec(sycl::queue& q, const I2ExpertBank& b, const float* x, bool x_per_slot,
                             const int32_t* idx, uint32_t T, uint32_t k, float* y, bool round_w, bool round, bool fast) {
    static const bool generic_only = std::getenv("IE_INKLING2_EXPERT_GENERIC") != nullptr;   // A/B switch
    if (fast && !round_w && !generic_only && b.cols % 32 == 0 && (reinterpret_cast<uintptr_t>(b.w) & 15) == 0 &&
        (reinterpret_cast<uintptr_t>(x) & 15) == 0)
    {
        static const uint32_t rps = [] { const char* e = std::getenv("IE_INKLING2_EXPERT_RPS"); return e ? uint32_t(std::atoi(e)) : 2u; }();
        static const bool wide = std::getenv("IE_INKLING2_EXPERT_WIDE") != nullptr;
        if (wide && b.fmt == 0 && b.cols % 32 == 0) return i2_expert_matvec_sg<0, 1, true>(q, b, x, x_per_slot, idx, T, k, y, round);
        static const int nsg = [] { const char* e = std::getenv("IE_INKLING2_EXPERT_NSG"); return e ? std::atoi(e) : 8; }();
        static const int diag = [] { const char* e = std::getenv("IE_INKLING2_EXPERT_DIAG"); return e ? std::atoi(e) : 0; }();
        static const int dec = [] { const char* e = std::getenv("IE_INKLING2_EXPERT_DEC"); return e ? std::atoi(e) : 1; }();
        static const uint32_t u2 = [] { const char* e = std::getenv("IE_INKLING2_EXPERT_U2"); return e ? uint32_t(std::atoi(e)) : 2u; }();
        static const int flat = [] { const char* e = std::getenv("IE_INKLING2_EXPERT_FLAT"); return e ? std::atoi(e) : 0; }();
        if (flat && b.fmt == 0 && (4 * (b.cols / 32)) % 16 == 0 && b.rows % 4 == 0) {
            if (flat == 5 && b.cols == 1152) return i2_expert_flat_nv4<4, 3, 36>(q, b, x, x_per_slot, idx, T, k, y, round, nullptr, 0.f, 0.f);
            if (flat == 6 && b.cols == 1152) return i2_expert_flat_nv4<4, 1, 36>(q, b, x, x_per_slot, idx, T, k, y, round, nullptr, 0.f, 0.f);
            if (flat == 3) return i2_expert_flat_nv4<4, 3>(q, b, x, x_per_slot, idx, T, k, y, round, nullptr, 0.f, 0.f);
            if (flat == 1) return i2_expert_flat_nv4<4, 1>(q, b, x, x_per_slot, idx, T, k, y, round, nullptr, 0.f, 0.f);
            if (flat == 4) return i2_expert_flat_nv4<8, 3>(q, b, x, x_per_slot, idx, T, k, y, round, nullptr, 0.f, 0.f);
            return i2_expert_flat_nv4<4, 2>(q, b, x, x_per_slot, idx, T, k, y, round, nullptr, 0.f, 0.f);
        }
        if (dec == 1 && !diag && nsg == 8) {
#define I2_W2(R, UU) if (rps == R && u2 == UU) return b.fmt == 0 \
                ? i2_expert_matvec_sg<0, R, false, 8, 0, 1, UU>(q, b, x, x_per_slot, idx, T, k, y, round) \
                : i2_expert_matvec_sg<1, R, false, 8, 0, 1, UU>(q, b, x, x_per_slot, idx, T, k, y, round);
            I2_W2(1, 1) I2_W2(1, 2) I2_W2(1, 3) I2_W2(1, 4) I2_W2(2, 1) I2_W2(2, 2) I2_W2(2, 3) I2_W2(4, 1) I2_W2(4, 2)
#undef I2_W2
            return b.fmt == 0 ? i2_expert_matvec_sg<0, 1, false, 8, 0, 1>(q, b, x, x_per_slot, idx, T, k, y, round)
                              : i2_expert_matvec_sg<1, 1, false, 8, 0, 1>(q, b, x, x_per_slot, idx, T, k, y, round);
        }
        if (rps == 1 && b.fmt == 0 && diag == 1) return i2_expert_matvec_sg<0, 1, false, 8, 1>(q, b, x, x_per_slot, idx, T, k, y, round);
        if (rps == 1 && b.fmt == 0 && diag == 2) return i2_expert_matvec_sg<0, 1, false, 8, 2>(q, b, x, x_per_slot, idx, T, k, y, round);
        if (rps == 1 && nsg == 4) return b.fmt == 0 ? i2_expert_matvec_sg<0, 1, false, 4>(q, b, x, x_per_slot, idx, T, k, y, round)
                                                    : i2_expert_matvec_sg<1, 1, false, 4>(q, b, x, x_per_slot, idx, T, k, y, round);
        if (rps == 1 && nsg == 16) return b.fmt == 0 ? i2_expert_matvec_sg<0, 1, false, 16>(q, b, x, x_per_slot, idx, T, k, y, round)
                                                     : i2_expert_matvec_sg<1, 1, false, 16>(q, b, x, x_per_slot, idx, T, k, y, round);
        if (rps == 1) return b.fmt == 0 ? i2_expert_matvec_sg<0, 1>(q, b, x, x_per_slot, idx, T, k, y, round)
                                        : i2_expert_matvec_sg<1, 1>(q, b, x, x_per_slot, idx, T, k, y, round);
        if (rps == 2) return b.fmt == 0 ? i2_expert_matvec_sg<0, 2>(q, b, x, x_per_slot, idx, T, k, y, round)
                                        : i2_expert_matvec_sg<1, 2>(q, b, x, x_per_slot, idx, T, k, y, round);
        return b.fmt == 0 ? i2_expert_matvec_sg<0, 4>(q, b, x, x_per_slot, idx, T, k, y, round)
                          : i2_expert_matvec_sg<1, 4>(q, b, x, x_per_slot, idx, T, k, y, round);
    }
    constexpr uint32_t WG = 64;
    const uint32_t rows = b.rows, cols = b.cols;
    const int fmt = b.fmt;
    const uint8_t* W = b.w;
    const uint8_t* S = b.s;
    const float* fac = b.fac;
    return ie::ps(q, "i2_expert_matvec", [&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(uint64_t(T) * k * rows * WG, WG), [=](sycl::nd_item<1> it) {
            const uint64_t g = it.get_group(0);
            const uint32_t r = uint32_t(g % rows);
            const uint64_t slot = g / rows;                             // t * k + j
            const uint32_t t = uint32_t(slot / k);
            const uint32_t l = uint32_t(it.get_local_id(0));
            const uint64_t e = uint64_t(idx[slot]);
            const float* xv = x + (x_per_slot ? slot : uint64_t(t)) * cols;
            float acc = 0.f;
            if (fmt == 0) {                                             // NVFP4: element 2i = low nibble of byte i
                const uint8_t* wr = W + (e * rows + r) * (cols / 2);
                const uint8_t* sr = S + (e * rows + r) * (cols / 16);
                const float f = fac[e];
                for (uint32_t bi = l; bi < cols / 2; bi += WG) {
                    const uint8_t byte = wr[bi];
                    const float sc = i2_e4m3(sr[bi >> 3]);              // one scale per 16 columns
                    float w0 = (i2_e2m1(byte & 15u) * sc) * f, w1 = (i2_e2m1(byte >> 4) * sc) * f;
                    if (round_w) { w0 = i2_bf16_round(w0); w1 = i2_bf16_round(w1); }
                    acc += w0 * xv[2 * bi] + w1 * xv[2 * bi + 1];
                }
            } else {                                                    // MXFP8: one exponent per 32 columns
                const uint8_t* wr = W + (e * rows + r) * cols;
                const uint8_t* sr = S + (e * rows + r) * (cols / 32);
                for (uint32_t c = l; c < cols; c += WG) acc += (i2_e4m3(wr[c]) * i2_ue8m0(sr[c >> 5])) * xv[c];
            }
            const float s = sycl::reduce_over_group(it.get_group(), acc, sycl::plus<float>());
            if (l == 0) y[slot * rows + r] = i2r(s, round);
        });
    });
}

sycl::event i2_expert_act(sycl::queue& q, const float* gu, uint32_t R, uint32_t I, float gcap, float ucap, float* act,
                          bool round) {
    return ie::ps(q, "i2_expert_act", [&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(uint64_t(R) * I), [=](sycl::id<1> id) {
            const uint64_t i = id[0];
            const uint64_t r = i / I, j = i % I;
            const float g = gu[r * 2 * I + 2 * j], u = gu[r * 2 * I + 2 * j + 1];
            const float v = gcap * sycl::tanh(g / gcap) * (1.f / (1.f + sycl::exp(-g))) * ucap * sycl::tanh(u / ucap);
            act[i] = i2r(v, round);
        });
    });
}

sycl::event i2_expert_combine(sycl::queue& q, const float* y, const float* w, uint32_t T, uint32_t k, uint32_t H,
                              float* out, bool round) {
    return ie::ps(q, "i2_expert_combine", [&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(uint64_t(T) * H), [=](sycl::id<1> id) {
            const uint64_t t = id[0] / H, i = id[0] % H;
            float acc = 0.f;
            for (uint32_t j = 0; j < k; ++j) acc += w[t * k + j] * y[(t * k + j) * H + i];
            out[id[0]] = i2r(acc, round);
        });
    });
}

}  // namespace ie
