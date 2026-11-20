// src/ops/hy4flash_moe_grp.cpp -- Hy4-flash prompt chunks: the routed experts expert-major on the XMX units.
//
// The chunk path used to run one matrix-vector product per (row, expert) pick: an expert picked by 64 rows of a chunk was
// read from VRAM 64 times. Here the picks are sorted by expert ("j" order: an expert's picks contiguous, its rows ascending)
// and a job is up to 32 picks of one expert: its weights are read once per job, dequantized a 32-wide K span at a time into
// local memory, and multiplied on the XMX units against the job's rows (fp16 x fp16 -> fp32).
//   gu:   hj [j, MI]  = silu(min(x W_g^T, lim)) * clamp(x W_u^T, -lim, lim)       (x = the pick's row of x16)
//   down: yj [j, LAT] = w_j * (hj W_d^T)
//   sum:  y [t, LAT]  = sum over the row's picks k = 0 .. TK-1 (pos[t * TK + k] >= 0) of yj[pos]  -- fixed order
// A job's weights live wherever job_base points: a resident slot of the layer's slab, or a staging slot.
// 8-bit experts only (int8 [N][K] + fp16 scale per 32, the Hy4fQ8Store slot layout).
// Kernels are prefixed hy4fgrp_ (the SYCL kernel-name collision trap).
#include "ie/hy4flash_q8.hpp"
#include "ie/kernel_profiler.hpp"

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>

#include <cstdint>

namespace ie {

namespace {
namespace mat  = sycl::ext::oneapi::experimental::matrix;
namespace imat = sycl::ext::intel::experimental::matrix;
constexpr int SG = 16, NSG = 4, WG = SG * NSG, TM = 8, TN = 16, TK = 16, MR = 4, JR = TM * MR /* 32 rows a job */, SPAN = 32;
inline float hy4fgrp_h2f(uint16_t h) { return float(sycl::bit_cast<sycl::half>(h)); }
}

sycl::event hy4f_moe_grp_gu(sycl::queue& q, const sycl::half* x16, const uint64_t* job_base, const int32_t* job_j0, const int32_t* job_n,
                            const int32_t* j_row, uint32_t n_jobs, Hy4fQ8Store::Offsets off, sycl::half* hj, uint32_t LAT, uint32_t MI, float lim) {
    const uint64_t o_up = off.up, o_gd = off.gate_d, o_ud = off.up_d;
    return ie::ps(q, "hy4fgrp_gu", [&](sycl::handler& h) {
        sycl::local_accessor<sycl::half, 1> As(sycl::range<1>(JR * SPAN), h);
        sycl::local_accessor<sycl::half, 1> Bg(sycl::range<1>(NSG * TN * SPAN), h), Bu(sycl::range<1>(NSG * TN * SPAN), h);
        sycl::local_accessor<float, 1> Og(sycl::range<1>(NSG * JR * TN), h), Ou(sycl::range<1>(NSG * JR * TN), h);
        h.parallel_for(sycl::nd_range<2>({size_t(n_jobs), size_t(MI / (TN * NSG)) * WG}, {1, size_t(WG)}), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
            auto sg = it.get_sub_group(); auto grp = it.get_group();
            const uint32_t job = uint32_t(it.get_group(0)), lid = uint32_t(it.get_local_id(1)), sgid = lid / SG, lane = lid % SG;
            const uint32_t n0 = (uint32_t(it.get_group(1)) * NSG + sgid) * TN;
            const uint8_t* base = reinterpret_cast<const uint8_t*>(job_base[job]);
            const int32_t j0 = job_j0[job], nr = job_n[job];
            const int8_t* wg = reinterpret_cast<const int8_t*>(base) + uint64_t(n0 + lane) * LAT;
            const int8_t* wu = reinterpret_cast<const int8_t*>(base + o_up) + uint64_t(n0 + lane) * LAT;
            const uint16_t* dg = reinterpret_cast<const uint16_t*>(base + o_gd) + uint64_t(n0 + lane) * (LAT / 32);
            const uint16_t* du = reinterpret_cast<const uint16_t*>(base + o_ud) + uint64_t(n0 + lane) * (LAT / 32);
            const uint32_t ar = lid / 2, ac = (lid % 2) * 16;                     // this lane's A staging: row ar, 16 halves at ac
            const sycl::half* xa = ar < uint32_t(nr) ? x16 + uint64_t(j_row[j0 + ar]) * LAT + ac : nullptr;
            mat::joint_matrix<sycl::sub_group, float, mat::use::accumulator, TM, TN> ag[MR], au[MR];
            #pragma unroll
            for (int m = 0; m < MR; ++m) { mat::joint_matrix_fill(sg, ag[m], 0.0f); mat::joint_matrix_fill(sg, au[m], 0.0f); }
            sycl::half* bg = &Bg[sgid * TN * SPAN]; sycl::half* bu = &Bu[sgid * TN * SPAN];
            for (uint32_t s = 0; s < LAT / SPAN; ++s) {
                #pragma unroll
                for (int t = 0; t < 16; ++t) As[ar * SPAN + ac + t] = xa ? xa[s * SPAN + t] : sycl::half(0.f);
                const float sgd = hy4fgrp_h2f(dg[s]), sud = hy4fgrp_h2f(du[s]);
                const sycl::vec<int8_t, 16>* pg = reinterpret_cast<const sycl::vec<int8_t, 16>*>(wg + s * SPAN);
                const sycl::vec<int8_t, 16>* pu = reinterpret_cast<const sycl::vec<int8_t, 16>*>(wu + s * SPAN);
                #pragma unroll
                for (int v = 0; v < 2; ++v) {
                    const sycl::vec<int8_t, 16> g = pg[v], u = pu[v];
                    #pragma unroll
                    for (int t = 0; t < 16; ++t) { bg[lane * SPAN + v * 16 + t] = sycl::half(float(g[t]) * sgd); bu[lane * SPAN + v * 16 + t] = sycl::half(float(u[t]) * sud); }
                }
                sycl::group_barrier(grp);
                #pragma unroll
                for (int kh = 0; kh < SPAN / TK; ++kh) {
                    mat::joint_matrix<sycl::sub_group, sycl::half, mat::use::b, TK, TN, mat::layout::col_major> tg, tu;
                    mat::joint_matrix_load(sg, tg, Bg.template get_multi_ptr<sycl::access::decorated::no>() + sgid * TN * SPAN + kh * TK, SPAN);
                    mat::joint_matrix_load(sg, tu, Bu.template get_multi_ptr<sycl::access::decorated::no>() + sgid * TN * SPAN + kh * TK, SPAN);
                    #pragma unroll
                    for (int m = 0; m < MR; ++m) {
                        mat::joint_matrix<sycl::sub_group, sycl::half, mat::use::a, TM, TK, mat::layout::row_major> ta;
                        mat::joint_matrix_load(sg, ta, As.template get_multi_ptr<sycl::access::decorated::no>() + (m * TM) * SPAN + kh * TK, SPAN);
                        mat::joint_matrix_mad(sg, ag[m], ta, tg, ag[m]);
                        mat::joint_matrix_mad(sg, au[m], ta, tu, au[m]);
                    }
                }
                sycl::group_barrier(grp);
            }
            float* og = &Og[sgid * JR * TN]; float* ou = &Ou[sgid * JR * TN];
            #pragma unroll
            for (int m = 0; m < MR; ++m) {
                imat::joint_matrix_apply(sg, ag[m], [&](float& v, size_t r, size_t c) { og[(m * TM + r) * TN + c] = v; });
                imat::joint_matrix_apply(sg, au[m], [&](float& v, size_t r, size_t c) { ou[(m * TM + r) * TN + c] = v; });
            }
            sycl::group_barrier(sg);
            for (uint32_t e = lane; e < uint32_t(JR * TN); e += SG) {
                const uint32_t r = e / TN, c = e % TN;
                if (r >= uint32_t(nr)) continue;
                const float g = sycl::fmin(og[e], lim), u = sycl::fmin(sycl::fmax(ou[e], -lim), lim);
                hj[uint64_t(j0 + r) * MI + n0 + c] = sycl::half(g / (1.0f + sycl::exp(-g)) * u);
            }
        });
    });
}

sycl::event hy4f_moe_grp_down(sycl::queue& q, const sycl::half* hj, const uint64_t* job_base, const int32_t* job_j0, const int32_t* job_n,
                              const float* j_w, uint32_t n_jobs, Hy4fQ8Store::Offsets off, float* yj, uint32_t LAT, uint32_t MI) {
    const uint64_t o_dn = off.down, o_dd = off.down_d;
    return ie::ps(q, "hy4fgrp_down", [&](sycl::handler& h) {
        sycl::local_accessor<sycl::half, 1> As(sycl::range<1>(JR * SPAN), h), Bd(sycl::range<1>(NSG * TN * SPAN), h);
        sycl::local_accessor<float, 1> Od(sycl::range<1>(NSG * JR * TN), h);
        h.parallel_for(sycl::nd_range<2>({size_t(n_jobs), size_t(LAT / (TN * NSG)) * WG}, {1, size_t(WG)}), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
            auto sg = it.get_sub_group(); auto grp = it.get_group();
            const uint32_t job = uint32_t(it.get_group(0)), lid = uint32_t(it.get_local_id(1)), sgid = lid / SG, lane = lid % SG;
            const uint32_t n0 = (uint32_t(it.get_group(1)) * NSG + sgid) * TN;
            const uint8_t* base = reinterpret_cast<const uint8_t*>(job_base[job]);
            const int32_t j0 = job_j0[job], nr = job_n[job];
            const int8_t* wd = reinterpret_cast<const int8_t*>(base + o_dn) + uint64_t(n0 + lane) * MI;
            const uint16_t* dd = reinterpret_cast<const uint16_t*>(base + o_dd) + uint64_t(n0 + lane) * (MI / 32);
            const uint32_t ar = lid / 2, ac = (lid % 2) * 16;
            const sycl::half* xa = ar < uint32_t(nr) ? hj + uint64_t(j0 + ar) * MI + ac : nullptr;
            mat::joint_matrix<sycl::sub_group, float, mat::use::accumulator, TM, TN> ad[MR];
            #pragma unroll
            for (int m = 0; m < MR; ++m) mat::joint_matrix_fill(sg, ad[m], 0.0f);
            sycl::half* bd = &Bd[sgid * TN * SPAN];
            for (uint32_t s = 0; s < MI / SPAN; ++s) {
                #pragma unroll
                for (int t = 0; t < 16; ++t) As[ar * SPAN + ac + t] = xa ? xa[s * SPAN + t] : sycl::half(0.f);
                const float sdd = hy4fgrp_h2f(dd[s]);
                const sycl::vec<int8_t, 16>* pd = reinterpret_cast<const sycl::vec<int8_t, 16>*>(wd + s * SPAN);
                #pragma unroll
                for (int v = 0; v < 2; ++v) {
                    const sycl::vec<int8_t, 16> d = pd[v];
                    #pragma unroll
                    for (int t = 0; t < 16; ++t) bd[lane * SPAN + v * 16 + t] = sycl::half(float(d[t]) * sdd);
                }
                sycl::group_barrier(grp);
                #pragma unroll
                for (int kh = 0; kh < SPAN / TK; ++kh) {
                    mat::joint_matrix<sycl::sub_group, sycl::half, mat::use::b, TK, TN, mat::layout::col_major> tb;
                    mat::joint_matrix_load(sg, tb, Bd.template get_multi_ptr<sycl::access::decorated::no>() + sgid * TN * SPAN + kh * TK, SPAN);
                    #pragma unroll
                    for (int m = 0; m < MR; ++m) {
                        mat::joint_matrix<sycl::sub_group, sycl::half, mat::use::a, TM, TK, mat::layout::row_major> ta;
                        mat::joint_matrix_load(sg, ta, As.template get_multi_ptr<sycl::access::decorated::no>() + (m * TM) * SPAN + kh * TK, SPAN);
                        mat::joint_matrix_mad(sg, ad[m], ta, tb, ad[m]);
                    }
                }
                sycl::group_barrier(grp);
            }
            float* od = &Od[sgid * JR * TN];
            #pragma unroll
            for (int m = 0; m < MR; ++m) imat::joint_matrix_apply(sg, ad[m], [&](float& v, size_t r, size_t c) { od[(m * TM + r) * TN + c] = v; });
            sycl::group_barrier(sg);
            for (uint32_t e = lane; e < uint32_t(JR * TN); e += SG) {
                const uint32_t r = e / TN, c = e % TN;
                if (r < uint32_t(nr)) yj[uint64_t(j0 + r) * LAT + n0 + c] = j_w[j0 + r] * od[e];
            }
        });
    });
}

sycl::event hy4f_moe_grp_sum(sycl::queue& q, const float* yj, const int32_t* pos, uint32_t T, uint32_t TKp, float* y, uint32_t LAT) {
    return ie::ps(q, "hy4fgrp_sum", [&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>(T, LAT), [=](sycl::id<2> i) {
            float a = 0.f;
            for (uint32_t k = 0; k < TKp; ++k) { const int32_t p = pos[i[0] * TKp + k]; if (p >= 0) a += yj[uint64_t(p) * LAT + i[1]]; }
            y[i[0] * LAT + i[1]] = a;
        });
    });
}

}  // namespace ie
