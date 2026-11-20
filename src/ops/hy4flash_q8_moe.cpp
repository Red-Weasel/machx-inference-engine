// src/ops/hy4flash_q8_moe.cpp — Hy4-flash: one decode token's routed experts in TWO launches (include/ie/hy4flash_q8.hpp).
//
// The bring-up path ran every picked expert as its own chain (gather, gate + up, SwiGLU, down, scatter: 7 launches x 16
// experts x 41 layers a token); at 3.5 tok/s the launches, not the weights, were the cost. Here the 16 experts of a token
// share a launch: kernel 1 computes h[j] = silu(min(gate_j . x, lim)) * clamp(up_j . x, +-lim) for every job j, kernel 2
// y = sum_j w_j * (down_j . h[j]). The inner loops are gemv_q8_0_soa_f16_dual_g_t<false, 16>'s (src/ops/gemv_q8dot.cpp):
// W8A16, 16 columns a work-group, a sub-group of 16 lanes striding the 32-weight blocks, fp32 accumulation, the block's
// fp16 scale applied per block -- only the job dimension is new. All planes must be 16-byte aligned (the store's are).
#include "ie/hy4flash_q8.hpp"
#include "ie/kernel_profiler.hpp"

#include <sycl/sycl.hpp>
#include <array>

namespace ie {

namespace {
constexpr int kJobsMax = 16;
inline float h2f(uint16_t h) { return float(sycl::bit_cast<sycl::half>(h)); }
// 32 int8 weights at w (16-byte aligned) against the 32 fp16 activations in xq
inline float dot32(const int8_t* w, const sycl::vec<sycl::half, 8>* xq) {
    const sycl::vec<uint32_t, 4>* w128 = reinterpret_cast<const sycl::vec<uint32_t, 4>*>(w);
    const sycl::vec<uint32_t, 4> wa = w128[0], wb = w128[1];
    int8_t wv[32];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint32_t v = wa[j], u = wb[j];
        wv[4 * j + 0] = int8_t(uint8_t(v)); wv[4 * j + 1] = int8_t(uint8_t(v >> 8)); wv[4 * j + 2] = int8_t(uint8_t(v >> 16)); wv[4 * j + 3] = int8_t(uint8_t(v >> 24));
        wv[16 + 4 * j + 0] = int8_t(uint8_t(u)); wv[16 + 4 * j + 1] = int8_t(uint8_t(u >> 8)); wv[16 + 4 * j + 2] = int8_t(uint8_t(u >> 16)); wv[16 + 4 * j + 3] = int8_t(uint8_t(u >> 24));
    }
    float acc = 0.f;
    #pragma unroll
    for (int i = 0; i < 32; ++i) acc += float(wv[i]) * float(xq[i >> 3][i & 7]);
    return acc;
}
// The 32 weights of block b of a row stored at B bits (Hy4fQFmt's layouts), against the 32 fp16 activations in xq.
// K = the row's weight count (where the 5-bit rows keep their high-bit words).
template <int B> inline float dotw(const int8_t* row, uint32_t b, uint32_t K, const sycl::vec<sycl::half, 8>* xq) {
    if constexpr (B == 8) return dot32(row + uint64_t(b) * 32, xq);
    else {
        const sycl::vec<uint32_t, 4> w = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(row + uint64_t(b) * 16);
        uint32_t hb = 0; int off = 8;
        if constexpr (B == 5) { hb = *reinterpret_cast<const uint32_t*>(row + (K >> 1) + uint64_t(b) * 4); off = 16; }
        float acc = 0.f;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            const uint32_t nib = (w[i >> 3] >> (4 * (i & 7))) & 15u;
            const int qv = int(nib | (((hb >> i) & 1u) << 4)) - off;
            acc += float(qv) * float(xq[i >> 3][i & 7]);
        }
        return acc;
    }
}
constexpr int SG = 16, NPW = 16, WGI = NPW * SG;

// ---- one token: gate, up and the clamped SwiGLU of every job (BG = the width gate / up are stored at) ----
template <int BG> sycl::event decode_gu(sycl::queue& q, const sycl::half* x16, const std::array<const int8_t*, kJobsMax>& GQ, const std::array<const int8_t*, kJobsMax>& UQ,
                                        const std::array<const uint16_t*, kJobsMax>& GD, const std::array<const uint16_t*, kJobsMax>& UD,
                                        uint32_t n_jobs, sycl::half* hj, uint32_t LAT, uint32_t MI, float lim) {
    const uint32_t kb = LAT / 32, wgs1 = (MI + NPW - 1) / NPW; const uint64_t rb = Hy4fQFmt::row_bytes(BG, LAT);
    return ie::ps(q, "hy4f_q8_moe_gu", [&](sycl::handler& h) {
        // 2-D range: the job is its own dimension (no run-time division or modulo in the kernel -- Strata's notes on this
        // compiler and card report `t % n` with a run-time n coming out wrong)
        h.parallel_for(sycl::nd_range<2>({size_t(n_jobs), size_t(wgs1) * WGI}, {1, size_t(WGI)}), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint32_t lid = uint32_t(it.get_local_id(1)), wg = uint32_t(it.get_group(1));
            const uint32_t job = uint32_t(it.get_global_id(0)), n = wg * NPW + lid / SG, lane = lid % SG;
            if (n >= MI) return;
            const int8_t* wa = GQ[job] + uint64_t(n) * rb; const int8_t* wb = UQ[job] + uint64_t(n) * rb;
            const uint16_t* da = GD[job] + uint64_t(n) * kb; const uint16_t* db = UD[job] + uint64_t(n) * kb;
            float ag = 0.f, au = 0.f;
            for (uint32_t b = lane; b < kb; b += SG) {
                const sycl::vec<sycl::half, 8>* xv = reinterpret_cast<const sycl::vec<sycl::half, 8>*>(x16 + uint64_t(b) * 32);
                sycl::vec<sycl::half, 8> xq[4];
                #pragma unroll
                for (int j = 0; j < 4; ++j) xq[j] = xv[j];
                ag += h2f(da[b]) * dotw<BG>(wa, b, LAT, xq);
                au += h2f(db[b]) * dotw<BG>(wb, b, LAT, xq);
            }
            ag = sycl::reduce_over_group(it.get_sub_group(), ag, sycl::plus<float>());
            au = sycl::reduce_over_group(it.get_sub_group(), au, sycl::plus<float>());
            if (lane == 0) {
                const float g = sycl::fmin(ag, lim), u = sycl::fmin(sycl::fmax(au, -lim), lim);
                hj[uint64_t(job) * MI + n] = sycl::half(g / (1.0f + sycl::exp(-g)) * u);
            }
        });
    });
}
// ---- one token: down of every job, weighted and summed ----
template <int BD> sycl::event decode_down(sycl::queue& q, const std::array<const int8_t*, kJobsMax>& DQ, const std::array<const uint16_t*, kJobsMax>& DD,
                                          const std::array<float, kJobsMax>& WJ, uint32_t n_jobs, const sycl::half* hj, float* y, uint32_t LAT, uint32_t MI) {
    const uint32_t kb2 = MI / 32, wgs2 = (LAT + NPW - 1) / NPW; const uint64_t rb = Hy4fQFmt::row_bytes(BD, MI);
    return ie::ps(q, "hy4f_q8_moe_down", [&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(uint64_t(wgs2) * WGI, WGI), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint32_t lid = uint32_t(it.get_local_id(0)), wg = uint32_t(it.get_group(0));
            const uint32_t n = wg * NPW + lid / SG, lane = lid % SG;
            if (n >= LAT) return;
            float acc = 0.f;
            for (uint32_t job = 0; job < n_jobs; ++job) {
                const int8_t* wd = DQ[job] + uint64_t(n) * rb; const uint16_t* dd = DD[job] + uint64_t(n) * kb2;
                const sycl::half* xh = hj + uint64_t(job) * MI;
                float a = 0.f;
                for (uint32_t b = lane; b < kb2; b += SG) {
                    const sycl::vec<sycl::half, 8>* xv = reinterpret_cast<const sycl::vec<sycl::half, 8>*>(xh + uint64_t(b) * 32);
                    sycl::vec<sycl::half, 8> xq[4];
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) xq[j] = xv[j];
                    a += h2f(dd[b]) * dotw<BD>(wd, b, MI, xq);
                }
                acc += WJ[job] * a;
            }
            acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
            if (lane == 0) y[n] = acc;
        });
    });
}
// ---- a chunk: the same two steps per (row, expert) job addressed through the layer's slab ----
template <int BG> void rows_gu(sycl::queue& q, const sycl::half* x16, const uint8_t* slab, uint64_t slot_bytes, uint64_t o_up, uint64_t o_gd, uint64_t o_ud,
                               const int32_t* job_slot, const int32_t* job_row, uint32_t n_jobs, sycl::half* hj, uint32_t LAT, uint32_t MI, float lim) {
    const uint32_t kb = LAT / 32, wgs1 = (MI + NPW - 1) / NPW; const uint64_t rb = Hy4fQFmt::row_bytes(BG, LAT);
    ie::ps(q, "hy4f_q8_moe_rows_gu", [&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<2>({size_t(n_jobs), size_t(wgs1) * WGI}, {1, size_t(WGI)}), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint32_t lid = uint32_t(it.get_local_id(1)), wg = uint32_t(it.get_group(1));
            const uint32_t job = uint32_t(it.get_global_id(0)), n = wg * NPW + lid / SG, lane = lid % SG;
            if (n >= MI) return;
            const uint8_t* sb = slab + uint64_t(job_slot[job]) * slot_bytes;
            const int8_t* wa = reinterpret_cast<const int8_t*>(sb) + uint64_t(n) * rb;
            const int8_t* wb = reinterpret_cast<const int8_t*>(sb + o_up) + uint64_t(n) * rb;
            const uint16_t* da = reinterpret_cast<const uint16_t*>(sb + o_gd) + uint64_t(n) * kb;
            const uint16_t* db = reinterpret_cast<const uint16_t*>(sb + o_ud) + uint64_t(n) * kb;
            const sycl::half* xr = x16 + uint64_t(job_row[job]) * LAT;
            float ag = 0.f, au = 0.f;
            for (uint32_t b = lane; b < kb; b += SG) {
                const sycl::vec<sycl::half, 8>* xv = reinterpret_cast<const sycl::vec<sycl::half, 8>*>(xr + uint64_t(b) * 32);
                sycl::vec<sycl::half, 8> xq[4];
                #pragma unroll
                for (int j = 0; j < 4; ++j) xq[j] = xv[j];
                ag += h2f(da[b]) * dotw<BG>(wa, b, LAT, xq);
                au += h2f(db[b]) * dotw<BG>(wb, b, LAT, xq);
            }
            ag = sycl::reduce_over_group(it.get_sub_group(), ag, sycl::plus<float>());
            au = sycl::reduce_over_group(it.get_sub_group(), au, sycl::plus<float>());
            if (lane == 0) {
                const float g = sycl::fmin(ag, lim), u = sycl::fmin(sycl::fmax(au, -lim), lim);
                hj[uint64_t(job) * MI + n] = sycl::half(g / (1.0f + sycl::exp(-g)) * u);
            }
        });
    });
}
template <int BD> void rows_down(sycl::queue& q, const uint8_t* slab, uint64_t slot_bytes, uint64_t o_down, uint64_t o_dd, const int32_t* job_slot, const float* job_w,
                                 uint32_t n_jobs, const sycl::half* hj, float* yj, uint32_t LAT, uint32_t MI) {
    const uint32_t kb2 = MI / 32, wgs2 = (LAT + NPW - 1) / NPW; const uint64_t rb = Hy4fQFmt::row_bytes(BD, MI);
    ie::ps(q, "hy4f_q8_moe_rows_down", [&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<2>({size_t(n_jobs), size_t(wgs2) * WGI}, {1, size_t(WGI)}), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint32_t lid = uint32_t(it.get_local_id(1)), wg = uint32_t(it.get_group(1));
            const uint32_t job = uint32_t(it.get_global_id(0)), n = wg * NPW + lid / SG, lane = lid % SG;
            if (n >= LAT) return;
            const uint8_t* sb = slab + uint64_t(job_slot[job]) * slot_bytes;
            const int8_t* wd = reinterpret_cast<const int8_t*>(sb + o_down) + uint64_t(n) * rb;
            const uint16_t* dd = reinterpret_cast<const uint16_t*>(sb + o_dd) + uint64_t(n) * kb2;
            const sycl::half* xh = hj + uint64_t(job) * MI;
            float a = 0.f;
            for (uint32_t b = lane; b < kb2; b += SG) {
                const sycl::vec<sycl::half, 8>* xv = reinterpret_cast<const sycl::vec<sycl::half, 8>*>(xh + uint64_t(b) * 32);
                sycl::vec<sycl::half, 8> xq[4];
                #pragma unroll
                for (int j = 0; j < 4; ++j) xq[j] = xv[j];
                a += h2f(dd[b]) * dotw<BD>(wd, b, MI, xq);
            }
            a = sycl::reduce_over_group(it.get_sub_group(), a, sycl::plus<float>());
            if (lane == 0) yj[uint64_t(job) * LAT + n] = job_w[job] * a;
        });
    });
}
// ---- a decode pass of a few rows (2 .. 8: a drafted block being verified): every resident expert any row picked is a
// job, read ONCE; job_mask says which rows picked it. hj [job, T, MI], yj [job, T, LAT] (unweighted). ----
constexpr int kMRows = 8;
template <int BG> void mrows_gu(sycl::queue& q, const sycl::half* x16, const uint8_t* slab, uint64_t slot_bytes, uint64_t o_up, uint64_t o_gd, uint64_t o_ud,
                                const int32_t* job_slot, const uint32_t* job_mask, uint32_t n_jobs, uint32_t T, sycl::half* hj, uint32_t LAT, uint32_t MI, float lim) {
    const uint32_t kb = LAT / 32, wgs1 = (MI + NPW - 1) / NPW; const uint64_t rb = Hy4fQFmt::row_bytes(BG, LAT);
    ie::ps(q, "hy4f_q8_moe_mrows_gu", [&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<2>({size_t(n_jobs), size_t(wgs1) * WGI}, {1, size_t(WGI)}), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint32_t lid = uint32_t(it.get_local_id(1)), wg = uint32_t(it.get_group(1));
            const uint32_t job = uint32_t(it.get_global_id(0)), n = wg * NPW + lid / SG, lane = lid % SG;
            if (n >= MI) return;
            const uint8_t* sb = slab + uint64_t(job_slot[job]) * slot_bytes; const uint32_t mask = job_mask[job];
            const int8_t* wa = reinterpret_cast<const int8_t*>(sb) + uint64_t(n) * rb;
            const int8_t* wb = reinterpret_cast<const int8_t*>(sb + o_up) + uint64_t(n) * rb;
            const uint16_t* da = reinterpret_cast<const uint16_t*>(sb + o_gd) + uint64_t(n) * kb;
            const uint16_t* db = reinterpret_cast<const uint16_t*>(sb + o_ud) + uint64_t(n) * kb;
            float ag[kMRows], au[kMRows];
            for (int t = 0; t < kMRows; ++t) { ag[t] = 0.f; au[t] = 0.f; }
            for (uint32_t b = lane; b < kb; b += SG) {
                const float sa = h2f(da[b]), sbv = h2f(db[b]);
                for (uint32_t t = 0; t < T; ++t) {
                    if (!((mask >> t) & 1u)) continue;
                    const sycl::vec<sycl::half, 8>* xv = reinterpret_cast<const sycl::vec<sycl::half, 8>*>(x16 + uint64_t(t) * LAT + uint64_t(b) * 32);
                    sycl::vec<sycl::half, 8> xq[4];
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) xq[j] = xv[j];
                    ag[t] += sa * dotw<BG>(wa, b, LAT, xq);
                    au[t] += sbv * dotw<BG>(wb, b, LAT, xq);
                }
            }
            for (uint32_t t = 0; t < T; ++t) {
                const float g0 = sycl::reduce_over_group(it.get_sub_group(), ag[t], sycl::plus<float>());
                const float u0 = sycl::reduce_over_group(it.get_sub_group(), au[t], sycl::plus<float>());
                if (lane == 0 && ((mask >> t) & 1u)) {
                    const float g = sycl::fmin(g0, lim), u = sycl::fmin(sycl::fmax(u0, -lim), lim);
                    hj[(uint64_t(job) * T + t) * MI + n] = sycl::half(g / (1.0f + sycl::exp(-g)) * u);
                }
            }
        });
    });
}
template <int BD> void mrows_down(sycl::queue& q, const uint8_t* slab, uint64_t slot_bytes, uint64_t o_down, uint64_t o_dd, const int32_t* job_slot, const uint32_t* job_mask,
                                  uint32_t n_jobs, uint32_t T, const sycl::half* hj, float* yj, uint32_t LAT, uint32_t MI) {
    const uint32_t kb2 = MI / 32, wgs2 = (LAT + NPW - 1) / NPW; const uint64_t rb = Hy4fQFmt::row_bytes(BD, MI);
    ie::ps(q, "hy4f_q8_moe_mrows_down", [&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<2>({size_t(n_jobs), size_t(wgs2) * WGI}, {1, size_t(WGI)}), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint32_t lid = uint32_t(it.get_local_id(1)), wg = uint32_t(it.get_group(1));
            const uint32_t job = uint32_t(it.get_global_id(0)), n = wg * NPW + lid / SG, lane = lid % SG;
            if (n >= LAT) return;
            const uint8_t* sb = slab + uint64_t(job_slot[job]) * slot_bytes; const uint32_t mask = job_mask[job];
            const int8_t* wd = reinterpret_cast<const int8_t*>(sb + o_down) + uint64_t(n) * rb;
            const uint16_t* dd = reinterpret_cast<const uint16_t*>(sb + o_dd) + uint64_t(n) * kb2;
            float a[kMRows];
            for (int t = 0; t < kMRows; ++t) a[t] = 0.f;
            for (uint32_t b = lane; b < kb2; b += SG) {
                const float sd = h2f(dd[b]);
                for (uint32_t t = 0; t < T; ++t) {
                    if (!((mask >> t) & 1u)) continue;
                    const sycl::vec<sycl::half, 8>* xv = reinterpret_cast<const sycl::vec<sycl::half, 8>*>(hj + (uint64_t(job) * T + t) * MI + uint64_t(b) * 32);
                    sycl::vec<sycl::half, 8> xq[4];
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) xq[j] = xv[j];
                    a[t] += sd * dotw<BD>(wd, b, MI, xq);
                }
            }
            for (uint32_t t = 0; t < T; ++t) {
                const float v = sycl::reduce_over_group(it.get_sub_group(), a[t], sycl::plus<float>());
                if (lane == 0 && ((mask >> t) & 1u)) yj[(uint64_t(job) * T + t) * LAT + n] = v;
            }
        });
    });
}
}  // namespace

sycl::event hy4f_q8_moe_mrows(sycl::queue& q, const sycl::half* x16, const uint8_t* slab, uint64_t slot_bytes, Hy4fQ8Store::Offsets off,
                              const int32_t* job_slot, const uint32_t* job_mask, const int32_t* row_jobs, const float* row_w, uint32_t n_jobs, uint32_t T, uint32_t TK,
                              sycl::half* hj, float* yj, float* y, uint32_t LAT, uint32_t MI, float lim, Hy4fQFmt fmt) {
    if (n_jobs) {
        if (fmt.gu == 8) mrows_gu<8>(q, x16, slab, slot_bytes, off.up, off.gate_d, off.up_d, job_slot, job_mask, n_jobs, T, hj, LAT, MI, lim);
        else if (fmt.gu == 5) mrows_gu<5>(q, x16, slab, slot_bytes, off.up, off.gate_d, off.up_d, job_slot, job_mask, n_jobs, T, hj, LAT, MI, lim);
        else mrows_gu<4>(q, x16, slab, slot_bytes, off.up, off.gate_d, off.up_d, job_slot, job_mask, n_jobs, T, hj, LAT, MI, lim);
        if (fmt.dn == 8) mrows_down<8>(q, slab, slot_bytes, off.down, off.down_d, job_slot, job_mask, n_jobs, T, hj, yj, LAT, MI);
        else if (fmt.dn == 5) mrows_down<5>(q, slab, slot_bytes, off.down, off.down_d, job_slot, job_mask, n_jobs, T, hj, yj, LAT, MI);
        else mrows_down<4>(q, slab, slot_bytes, off.down, off.down_d, job_slot, job_mask, n_jobs, T, hj, yj, LAT, MI);
    }
    // a row's resident picks summed in the order it picked them (row_jobs [T, TK]: the job of pick k, -1 = not on the card)
    return ie::ps(q, "hy4f_q8_moe_mrows_sum", [&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>(T, LAT), [=](sycl::id<2> i) {
            float a = 0.f;
            for (uint32_t k = 0; k < TK; ++k) {
                const int32_t j = row_jobs[i[0] * TK + k];
                if (j >= 0) a += row_w[i[0] * TK + k] * yj[(uint64_t(j) * T + i[0]) * LAT + i[1]];
            }
            y[i[0] * LAT + i[1]] = a;
        });
    });
}

// y [T, N] fp32 = x16 [T, K] . W^T for T <= 8 rows with the matrix read ONCE: the weight row is the unit and the rows are
// its inner loop (hy4f_q8_gemv_f32 reads the matrix once per row: a 4-row pass cost three single rows).
sycl::event hy4f_q8_gemv_rows_f32(sycl::queue& q, const sycl::half* x16, const int8_t* qs, const uint16_t* d, float* y,
                                  uint32_t K, uint32_t N, uint32_t T) {
    constexpr int SG = 16, NPW = 16, WGI = NPW * SG, MR = 8;
    const uint32_t kb = K / 32, wgs = (N + NPW - 1) / NPW;
    return ie::ps(q, "hy4f_q8_gemv_rows_f32", [&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(size_t(wgs) * WGI, size_t(WGI)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint32_t lid = uint32_t(it.get_local_id(0)), wg = uint32_t(it.get_group(0));
            const uint32_t n = wg * NPW + lid / SG, lane = lid % SG;
            if (n >= N) return;
            const int8_t* w = qs + uint64_t(n) * K; const uint16_t* dn = d + uint64_t(n) * kb;
            float a[MR];
            for (int t = 0; t < MR; ++t) a[t] = 0.f;
            for (uint32_t b = lane; b < kb; b += SG) {
                const float sc = h2f(dn[b]);
                for (uint32_t t = 0; t < T; ++t) {
                    const sycl::vec<sycl::half, 8>* xv = reinterpret_cast<const sycl::vec<sycl::half, 8>*>(x16 + uint64_t(t) * K + uint64_t(b) * 32);
                    sycl::vec<sycl::half, 8> xq[4];
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) xq[j] = xv[j];
                    a[t] += sc * dot32(w + uint64_t(b) * 32, xq);
                }
            }
            for (uint32_t t = 0; t < T; ++t) {
                const float v = sycl::reduce_over_group(it.get_sub_group(), a[t], sycl::plus<float>());
                if (lane == 0) y[uint64_t(t) * N + n] = v;
            }
        });
    });
}

sycl::event hy4f_q8_moe_decode(sycl::queue& q, const sycl::half* x16, const Hy4fQ8Store::Planes* pl, const float* w, uint32_t n_jobs,
                               sycl::half* hj, float* y, uint32_t LAT, uint32_t MI, float lim, Hy4fQFmt fmt) {
    std::array<const int8_t*, kJobsMax> GQ{}, UQ{}, DQ{}; std::array<const uint16_t*, kJobsMax> GD{}, UD{}, DD{}; std::array<float, kJobsMax> WJ{};
    for (uint32_t j = 0; j < n_jobs && j < uint32_t(kJobsMax); ++j) { GQ[j] = pl[j].gate_qs; UQ[j] = pl[j].up_qs; DQ[j] = pl[j].down_qs; GD[j] = pl[j].gate_d; UD[j] = pl[j].up_d; DD[j] = pl[j].down_d; WJ[j] = w[j]; }
    if (fmt.gu == 8) decode_gu<8>(q, x16, GQ, UQ, GD, UD, n_jobs, hj, LAT, MI, lim);
    else if (fmt.gu == 5) decode_gu<5>(q, x16, GQ, UQ, GD, UD, n_jobs, hj, LAT, MI, lim);
    else decode_gu<4>(q, x16, GQ, UQ, GD, UD, n_jobs, hj, LAT, MI, lim);
    if (fmt.dn == 8) return decode_down<8>(q, DQ, DD, WJ, n_jobs, hj, y, LAT, MI);
    if (fmt.dn == 5) return decode_down<5>(q, DQ, DD, WJ, n_jobs, hj, y, LAT, MI);
    return decode_down<4>(q, DQ, DD, WJ, n_jobs, hj, y, LAT, MI);
}

sycl::event hy4f_q8_moe_rows(sycl::queue& q, const sycl::half* x16, const uint8_t* slab, uint64_t slot_bytes, Hy4fQ8Store::Offsets off,
                             const int32_t* job_slot, const int32_t* job_row, const float* job_w, const int32_t* row_start,
                             uint32_t n_jobs, uint32_t T, sycl::half* hj, float* yj, float* y, uint32_t LAT, uint32_t MI, float lim, Hy4fQFmt fmt) {
    if (n_jobs) {
        if (fmt.gu == 8) rows_gu<8>(q, x16, slab, slot_bytes, off.up, off.gate_d, off.up_d, job_slot, job_row, n_jobs, hj, LAT, MI, lim);
        else if (fmt.gu == 5) rows_gu<5>(q, x16, slab, slot_bytes, off.up, off.gate_d, off.up_d, job_slot, job_row, n_jobs, hj, LAT, MI, lim);
        else rows_gu<4>(q, x16, slab, slot_bytes, off.up, off.gate_d, off.up_d, job_slot, job_row, n_jobs, hj, LAT, MI, lim);
        if (fmt.dn == 8) rows_down<8>(q, slab, slot_bytes, off.down, off.down_d, job_slot, job_w, n_jobs, hj, yj, LAT, MI);
        else if (fmt.dn == 5) rows_down<5>(q, slab, slot_bytes, off.down, off.down_d, job_slot, job_w, n_jobs, hj, yj, LAT, MI);
        else rows_down<4>(q, slab, slot_bytes, off.down, off.down_d, job_slot, job_w, n_jobs, hj, yj, LAT, MI);
    }
    // a row's jobs summed in their order
    return ie::ps(q, "hy4f_q8_moe_rows_sum", [&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>(T, LAT), [=](sycl::id<2> i) {
            float a = 0.f;
            for (int32_t j = row_start[i[0]]; j < row_start[i[0] + 1]; ++j) a += yj[uint64_t(j) * LAT + i[1]];
            y[i[0] * LAT + i[1]] = a;
        });
    });
}

// y [T, N] fp32 = x16 [T, K] . W^T for an 8-bit SoA matrix (qs [N][K], d fp16 bits [N][K / 32]): decode's form of the
// big non-expert maps (gate, o, q_b, the shared expert, the latent maps). The same inner loop as above; a row of x per
// index of the first range dimension, so T rows read the weights T times -- for T <= 4.
sycl::event hy4f_q8_gemv_f32(sycl::queue& q, const sycl::half* x16, const int8_t* qs, const uint16_t* d, float* y,
                             uint32_t K, uint32_t N, uint32_t T) {
    constexpr int SG = 16, NPW = 16, WGI = NPW * SG;
    const uint32_t kb = K / 32, wgs = (N + NPW - 1) / NPW;
    return ie::ps(q, "hy4f_q8_gemv_f32", [&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<2>({size_t(T), size_t(wgs) * WGI}, {1, size_t(WGI)}), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint32_t lid = uint32_t(it.get_local_id(1)), wg = uint32_t(it.get_group(1));
            const uint32_t t = uint32_t(it.get_global_id(0)), n = wg * NPW + lid / SG, lane = lid % SG;
            if (n >= N) return;
            const int8_t* w = qs + uint64_t(n) * K; const uint16_t* dn = d + uint64_t(n) * kb; const sycl::half* xr = x16 + uint64_t(t) * K;
            float a = 0.f;
            for (uint32_t b = lane; b < kb; b += SG) {
                const sycl::vec<sycl::half, 8>* xv = reinterpret_cast<const sycl::vec<sycl::half, 8>*>(xr + uint64_t(b) * 32);
                sycl::vec<sycl::half, 8> xq[4];
                #pragma unroll
                for (int j = 0; j < 4; ++j) xq[j] = xv[j];
                a += h2f(dn[b]) * dot32(w + uint64_t(b) * 32, xq);
            }
            a = sycl::reduce_over_group(it.get_sub_group(), a, sycl::plus<float>());
            if (lane == 0) y[uint64_t(t) * N + n] = a;
        });
    });
}

// w16 [N][K] = qs * d: the same matrix as fp16 for a many-row GEMM (prompt reading), into a scratch buffer
sycl::event hy4f_q8_dequant_f16(sycl::queue& q, const int8_t* qs, const uint16_t* d, sycl::half* w16, uint32_t N, uint32_t K) {
    const uint32_t kb = K / 32;
    return ie::ps(q, "hy4f_q8_dequant_f16", [&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>(N, kb), [=](sycl::id<2> i) {
            const uint64_t at = uint64_t(i[0]) * K + uint64_t(i[1]) * 32;
            const float s = h2f(d[uint64_t(i[0]) * kb + i[1]]);
            #pragma unroll
            for (int j = 0; j < 32; ++j) w16[at + j] = sycl::half(float(qs[at + j]) * s);
        });
    });
}

}  // namespace ie
