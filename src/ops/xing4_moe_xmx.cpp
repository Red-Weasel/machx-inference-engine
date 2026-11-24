// src/ops/xing4_moe_xmx.cpp -- Xing4 MoE for prompt chunks on the XMX units, expert-grouped (include/ie/xing4_ops.hpp).
//
// A work-group computes a tile of 32 jobs (rows) x 64 outputs of one group (an expert, or the shared expert / a dense
// layer). Per 64-wide K step: the 32 rows' activations go to local memory (fp16), the weight tile's Q8_0 blocks are
// dequantized into local memory (a block a lane), and 16 sub-groups multiply 8 x 16 x 16 tiles. gate_up: the tile holds
// 64 gate and 64 up rows of the same outputs, each sub-group owning a gate tile and the matching up tile, so the epilogue
// writes h = silu(gate) * up. down: 32 rows x 128 outputs, written times the job's weight into per-job rows.
// The tile list (group, first row) comes from xg4_moe_tiles. Helpers are prefixed xg4mx_ (kernel-name collision trap).
#include "ie/xing4_ops.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <stdexcept>

namespace ie {
namespace xg4mx {
namespace mat  = sycl::ext::oneapi::experimental::matrix;
namespace imat = sycl::ext::intel::experimental::matrix;
constexpr int SG = 16, NSG = 16, WG = SG * NSG, TM = 8, TN = 16, TK = 16;
constexpr int MT = 32, KT = 64;                 // rows a tile, K a step
inline float h2f(uint16_t b) { return float(sycl::bit_cast<sycl::half>(b)); }
}  // namespace xg4mx

sycl::event xg4_moe_tiles(sycl::queue& q, const int32_t* off, uint32_t E, int32_t* tiles, uint32_t max_tiles) {
    return q.single_task<class Xg4MoeTilesK>([=]() {
        uint32_t n = 0;
        for (uint32_t g = 0; g <= E; ++g)
            for (int32_t r = off[g]; r < off[g + 1] && n < max_tiles; r += xg4mx::MT) { tiles[1 + 2 * n] = int32_t(g); tiles[2 + 2 * n] = r; ++n; }
        tiles[0] = int32_t(n);
    });
}

sycl::event xg4_moe_gate_up_xmx(sycl::queue& q, const sycl::half* x, const int8_t* g_qs, const uint16_t* g_d, const int8_t* u_qs,
                                const uint16_t* u_d, const int8_t* gs_qs, const uint16_t* gs_d, const int8_t* us_qs,
                                const uint16_t* us_d, const int32_t* off, const int32_t* list, const int32_t* tiles,
                                uint32_t max_tiles, float* h, uint32_t J, uint32_t per_tok, uint32_t N, uint32_t K,
                                uint32_t E, bool dense) {
    using namespace xg4mx;
    constexpr int NT = 64;                       // outputs a tile (64 gate + 64 up weight rows)
    if (N % NT || K % KT) throw std::runtime_error("xg4_moe_gate_up_xmx: N a multiple of 64 and K of 64");
    const uint32_t ncol = N / NT, nb = K / 32;
    const uint32_t n_row_tiles = dense ? (J + MT - 1) / MT : max_tiles;
    return q.submit([&](sycl::handler& hd) {
        sycl::local_accessor<sycl::half, 1> Xs(sycl::range<1>(MT * KT), hd);
        sycl::local_accessor<sycl::half, 1> Ws(sycl::range<1>(2 * NT * KT), hd);
        sycl::local_accessor<float, 1> Gs(sycl::range<1>(MT * NT), hd), Us(sycl::range<1>(MT * NT), hd);
        sycl::local_accessor<int32_t, 1> jrow(sycl::range<1>(MT), hd);
        hd.parallel_for<class Xg4MoeGuXmxK>(sycl::nd_range<2>({size_t(n_row_tiles), size_t(ncol) * WG}, {1, size_t(WG)}), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint32_t tix = uint32_t(it.get_group(0)), cb = uint32_t(it.get_group(1)), lid = uint32_t(it.get_local_id(1));
            auto sg = it.get_sub_group(); const uint32_t s = uint32_t(sg.get_group_linear_id()); auto grp = it.get_group();
            uint32_t g; int32_t r0, r1;
            if (dense) { g = 0; r0 = int32_t(tix * MT); r1 = int32_t(J); }
            else {
                if (int32_t(tix) >= tiles[0]) return;
                g = uint32_t(tiles[1 + 2 * tix]); r0 = tiles[2 + 2 * tix]; r1 = off[g + 1];
            }
            const int32_t nr = sycl::min(int32_t(MT), r1 - r0);
            if (lid < uint32_t(MT)) jrow[lid] = int32_t(lid) < nr ? (dense ? r0 + int32_t(lid) : list[r0 + int32_t(lid)]) : -1;
            const bool shared = dense || g == E;
            const size_t n0 = size_t(cb) * NT;
            const int8_t* gq = shared ? gs_qs : g_qs + size_t(g) * N * K; const int8_t* uq = shared ? us_qs : u_qs + size_t(g) * N * K;
            const uint16_t* gd = shared ? gs_d : g_d + size_t(g) * N * nb; const uint16_t* ud = shared ? us_d : u_d + size_t(g) * N * nb;
            const uint32_t mr = s % 4, nc = s / 4;     // this sub-group: row tile mr, gate tile nc and up tile nc (+64)
            mat::joint_matrix<sycl::sub_group, float, mat::use::accumulator, TM, TN> ag, au;
            mat::joint_matrix_fill(sg, ag, 0.0f); mat::joint_matrix_fill(sg, au, 0.0f);
            sycl::group_barrier(grp);
            for (uint32_t k0 = 0; k0 < K; k0 += KT) {
                {   // activations: 32 rows x 64, eight halves a lane
                    const uint32_t r = lid / 8, c8 = lid % 8;
                    const int32_t j = jrow[r];
                    sycl::vec<sycl::half, 8> v(sycl::half(0.f));
                    if (j >= 0) v = *reinterpret_cast<const sycl::vec<sycl::half, 8>*>(x + size_t(uint32_t(j) / per_tok) * K + k0 + c8 * 8);
                    *reinterpret_cast<sycl::vec<sycl::half, 8>*>(&Xs[r * KT + c8 * 8]) = v;
                }
                {   // weights: 128 rows (64 gate, 64 up) x 2 blocks, a block a lane
                    const uint32_t wr = lid / 2, bk = lid % 2;
                    const bool up = wr >= uint32_t(NT);
                    const size_t row = n0 + (up ? wr - NT : wr), blk = k0 / 32 + bk;
                    const int8_t* qp = (up ? uq : gq) + row * K + blk * 32;
                    const float dd = h2f((up ? ud : gd)[row * nb + blk]);
                    const sycl::vec<int32_t, 8> w = *reinterpret_cast<const sycl::vec<int32_t, 8>*>(qp);
                    sycl::half* o = &Ws[wr * KT + bk * 32];
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const int32_t a = w[i];
                        o[4 * i + 0] = sycl::half(dd * float(int8_t(a)));       o[4 * i + 1] = sycl::half(dd * float(int8_t(a >> 8)));
                        o[4 * i + 2] = sycl::half(dd * float(int8_t(a >> 16)));  o[4 * i + 3] = sycl::half(dd * float(int8_t(a >> 24)));
                    }
                }
                sycl::group_barrier(grp);
#pragma unroll
                for (int kk = 0; kk < KT; kk += TK) {
                    mat::joint_matrix<sycl::sub_group, sycl::half, mat::use::a, TM, TK, mat::layout::row_major> a;
                    mat::joint_matrix_load(sg, a, Xs.template get_multi_ptr<sycl::access::decorated::no>() + (mr * TM) * KT + kk, KT);
                    mat::joint_matrix<sycl::sub_group, sycl::half, mat::use::b, TK, TN, mat::layout::col_major> bg, bu;
                    mat::joint_matrix_load(sg, bg, Ws.template get_multi_ptr<sycl::access::decorated::no>() + (nc * TN) * KT + kk, KT);
                    mat::joint_matrix_load(sg, bu, Ws.template get_multi_ptr<sycl::access::decorated::no>() + (NT + nc * TN) * KT + kk, KT);
                    mat::joint_matrix_mad(sg, ag, a, bg, ag);
                    mat::joint_matrix_mad(sg, au, a, bu, au);
                }
                sycl::group_barrier(grp);
            }
            imat::joint_matrix_apply(sg, ag, [&](float& v, size_t r, size_t c) { Gs[(mr * TM + r) * NT + nc * TN + c] = v; });
            imat::joint_matrix_apply(sg, au, [&](float& v, size_t r, size_t c) { Us[(mr * TM + r) * NT + nc * TN + c] = v; });
            sycl::group_barrier(grp);
            for (uint32_t u = lid; u < uint32_t(MT) * NT; u += WG) {
                const uint32_t r = u / NT, c = u % NT; const int32_t j = jrow[r];
                if (j < 0) continue;
                const float gv = Gs[u], uv = Us[u];
                h[size_t(j) * N + n0 + c] = gv / (1.0f + sycl::exp(-gv)) * uv;   // fp32: xg4_h_pack scales a row that would overflow fp16
            }
        });
    });
}

sycl::event xg4_moe_down_xmx(sycl::queue& q, const sycl::half* hh, const float* hs, const int8_t* d_qs, const uint16_t* d_d, const int8_t* ds_qs,
                             const uint16_t* ds_d, const int32_t* off, const int32_t* list, const int32_t* tiles,
                             uint32_t max_tiles, const float* job_w, float* yj, uint32_t J, uint32_t N, uint32_t K, uint32_t E,
                             bool dense) {
    using namespace xg4mx;
    constexpr int NT = 128;                      // outputs a tile
    if (N % NT || K % KT) throw std::runtime_error("xg4_moe_down_xmx: N a multiple of 128 and K of 64");
    const uint32_t ncol = N / NT, nb = K / 32;
    const uint32_t n_row_tiles = dense ? (J + MT - 1) / MT : max_tiles;
    return q.submit([&](sycl::handler& hd) {
        sycl::local_accessor<sycl::half, 1> Xs(sycl::range<1>(MT * KT), hd);
        sycl::local_accessor<sycl::half, 1> Ws(sycl::range<1>(NT * KT), hd);
        sycl::local_accessor<int32_t, 1> jrow(sycl::range<1>(MT), hd);
        hd.parallel_for<class Xg4MoeDnXmxK>(sycl::nd_range<2>({size_t(n_row_tiles), size_t(ncol) * WG}, {1, size_t(WG)}), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint32_t tix = uint32_t(it.get_group(0)), cb = uint32_t(it.get_group(1)), lid = uint32_t(it.get_local_id(1));
            auto sg = it.get_sub_group(); const uint32_t s = uint32_t(sg.get_group_linear_id()); auto grp = it.get_group();
            uint32_t g; int32_t r0, r1;
            if (dense) { g = 0; r0 = int32_t(tix * MT); r1 = int32_t(J); }
            else {
                if (int32_t(tix) >= tiles[0]) return;
                g = uint32_t(tiles[1 + 2 * tix]); r0 = tiles[2 + 2 * tix]; r1 = off[g + 1];
            }
            const int32_t nr = sycl::min(int32_t(MT), r1 - r0);
            if (lid < uint32_t(MT)) jrow[lid] = int32_t(lid) < nr ? (dense ? r0 + int32_t(lid) : list[r0 + int32_t(lid)]) : -1;
            const bool shared = dense || g == E;
            const size_t n0 = size_t(cb) * NT;
            const int8_t* wq = shared ? ds_qs : d_qs + size_t(g) * N * K;
            const uint16_t* wd = shared ? ds_d : d_d + size_t(g) * N * nb;
            const uint32_t mr = s % 4, nc = s / 4;     // row tile mr, output tiles nc and nc + 4
            mat::joint_matrix<sycl::sub_group, float, mat::use::accumulator, TM, TN> a0, a1;
            mat::joint_matrix_fill(sg, a0, 0.0f); mat::joint_matrix_fill(sg, a1, 0.0f);
            sycl::group_barrier(grp);
            for (uint32_t k0 = 0; k0 < K; k0 += KT) {
                {
                    const uint32_t r = lid / 8, c8 = lid % 8;
                    const int32_t j = jrow[r];
                    sycl::vec<sycl::half, 8> v(sycl::half(0.f));
                    if (j >= 0) v = *reinterpret_cast<const sycl::vec<sycl::half, 8>*>(hh + size_t(j) * K + k0 + c8 * 8);
                    *reinterpret_cast<sycl::vec<sycl::half, 8>*>(&Xs[r * KT + c8 * 8]) = v;
                }
                {   // 128 rows x 2 blocks: one block a lane
                    const uint32_t wr = lid / 2, bk = lid % 2;
                    const size_t row = n0 + wr, blk = k0 / 32 + bk;
                    const float dd = h2f(wd[row * nb + blk]);
                    const sycl::vec<int32_t, 8> w = *reinterpret_cast<const sycl::vec<int32_t, 8>*>(wq + row * K + blk * 32);
                    sycl::half* o = &Ws[wr * KT + bk * 32];
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const int32_t a = w[i];
                        o[4 * i + 0] = sycl::half(dd * float(int8_t(a)));       o[4 * i + 1] = sycl::half(dd * float(int8_t(a >> 8)));
                        o[4 * i + 2] = sycl::half(dd * float(int8_t(a >> 16)));  o[4 * i + 3] = sycl::half(dd * float(int8_t(a >> 24)));
                    }
                }
                sycl::group_barrier(grp);
#pragma unroll
                for (int kk = 0; kk < KT; kk += TK) {
                    mat::joint_matrix<sycl::sub_group, sycl::half, mat::use::a, TM, TK, mat::layout::row_major> a;
                    mat::joint_matrix_load(sg, a, Xs.template get_multi_ptr<sycl::access::decorated::no>() + (mr * TM) * KT + kk, KT);
                    mat::joint_matrix<sycl::sub_group, sycl::half, mat::use::b, TK, TN, mat::layout::col_major> b0, b1;
                    mat::joint_matrix_load(sg, b0, Ws.template get_multi_ptr<sycl::access::decorated::no>() + (nc * TN) * KT + kk, KT);
                    mat::joint_matrix_load(sg, b1, Ws.template get_multi_ptr<sycl::access::decorated::no>() + ((nc + 4) * TN) * KT + kk, KT);
                    mat::joint_matrix_mad(sg, a0, a, b0, a0);
                    mat::joint_matrix_mad(sg, a1, a, b1, a1);
                }
                sycl::group_barrier(grp);
            }
            imat::joint_matrix_apply(sg, a0, [&](float& v, size_t r, size_t c) {
                const int32_t j = jrow[mr * TM + r];
                if (j >= 0) yj[size_t(j) * N + n0 + nc * TN + c] = (job_w && !dense ? job_w[j] : 1.f) * (hs ? hs[j] : 1.f) * v; });
            imat::joint_matrix_apply(sg, a1, [&](float& v, size_t r, size_t c) {
                const int32_t j = jrow[mr * TM + r];
                if (j >= 0) yj[size_t(j) * N + n0 + (nc + 4) * TN + c] = (job_w && !dense ? job_w[j] : 1.f) * (hs ? hs[j] : 1.f) * v; });
        });
    });
}

// Q8_0-SoA -> fp16 [N, K] row-major (the oneDNN GEMM's in-place W layout)
sycl::event xg4_q8_dequant(sycl::queue& q, const int8_t* qs, const uint16_t* d, sycl::half* w16, uint32_t N, uint32_t K) {
    const uint32_t nb = K / 32;
    return q.parallel_for<class Xg4Q8DequantK>(sycl::range<1>(size_t(N) * nb), [=](sycl::id<1> id) {
        const size_t b = id[0];
        const float dd = xg4mx::h2f(d[b]);
        const sycl::vec<int32_t, 8> w = *reinterpret_cast<const sycl::vec<int32_t, 8>*>(qs + b * 32);
        sycl::vec<sycl::half, 8> o;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int32_t a0 = w[2 * i], a1 = w[2 * i + 1];
            o[0] = sycl::half(dd * float(int8_t(a0))); o[1] = sycl::half(dd * float(int8_t(a0 >> 8)));
            o[2] = sycl::half(dd * float(int8_t(a0 >> 16))); o[3] = sycl::half(dd * float(int8_t(a0 >> 24)));
            o[4] = sycl::half(dd * float(int8_t(a1))); o[5] = sycl::half(dd * float(int8_t(a1 >> 8)));
            o[6] = sycl::half(dd * float(int8_t(a1 >> 16))); o[7] = sycl::half(dd * float(int8_t(a1 >> 24)));
            *reinterpret_cast<sycl::vec<sycl::half, 8>*>(w16 + b * 32 + 8 * i) = o;
        }
    });
}

}  // namespace ie
