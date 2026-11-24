// src/ops/xing4_attn.cpp -- Xing4 MLA attention on the XMX units over a PAGED latent cache (64-key pages, a page table a
// lane): the tile bodies of hy4flash_attn_xmx.cpp (split-key flash decoding for decode rows, a running softmax for prompt
// rows), with the keys read where they lie in their pages instead of gathered (decode) and each row carrying its own
// lane and position (a batch of requests in one call). Keys 0 .. rpos[r] of lane rlane[r] are attended, causal. No sinks.
// Kernel-launching helpers are prefixed xg4dx_ (the SYCL kernel-name collision trap).
#include "ie/xing4_ops.hpp"
#include "ie/kernel_profiler.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <deque>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <algorithm>
#include <utility>

namespace ie {
namespace xg4dx {

namespace mat  = sycl::ext::oneapi::experimental::matrix;
namespace imat = sycl::ext::intel::experimental::matrix;

constexpr int   SG  = 16, NSG = 16, WG = SG * NSG;   // 256 lanes
constexpr int   HT  = 32;                            // heads per tile
constexpr int   DQ  = 576, DV = 512;                 // score width (latent + rotary), context width (latent)
constexpr int   TM  = 8, TN = 16, TK = 16;
constexpr int   DC  = DV / NSG, NC = DC / TN;        // 32 context columns a sub-group, 2 column tiles
constexpr int   MR  = HT / TM, CT = HT / TN;         // 4 head row tiles (P . V), 2 head column tiles (S)
constexpr int   LPH = WG / HT;                       // softmax: 8 lanes a head
constexpr int   CWG = 512;                           // combine: a work-group a (row, head), a dim a lane
constexpr float kNegInf = -std::numeric_limits<float>::infinity();

struct Ws {
    sycl::half* kgat = nullptr; uint64_t kcap = 0;   // [rows][NCOLP][DQ]
    float*      mgat = nullptr; uint64_t mcap = 0;   // [rows][NCOLP]
    float*      po   = nullptr; uint64_t pcap = 0;   // [rows][n_ht][NB][HT][DV]
    float*      pml  = nullptr; uint64_t lcap = 0;   // [rows][n_ht][NB][2][HT]
};
Ws& xg4dx_ws_for(sycl::queue& q) {
    static std::deque<std::pair<sycl::queue, Ws>> all;
    static std::mutex m;
    std::lock_guard<std::mutex> lk(m);
    for (auto& p : all) if (p.first == q) return p.second;
    all.emplace_back(q, Ws{});
    return all.back().second;
}
template <class V>
void xg4dx_grow(sycl::queue& q, V*& p, uint64_t& cap, uint64_t need, const char* what) {
    if (need <= cap) return;
    if (p) { q.wait(); sycl::free(p, q); p = nullptr; cap = 0; }
    p = sycl::malloc_device<V>(size_t(need), q);
    if (!p) throw std::runtime_error(std::string("xg4_attn_decode: could not allocate ") + what);
    cap = need;
}

}  // namespace xg4dx

sycl::event xg4_attn_decode(sycl::queue& q, const sycl::half* a16, const sycl::half* pool, const int32_t* tab, uint32_t maxp,
                            const int32_t* rlane, const int32_t* rpos, sycl::half* ctx16, uint32_t rows, uint32_t NH, float scaling,
                            uint32_t max_keys) {
    using namespace xg4dx;
    const float* sinks = nullptr; constexpr bool use_sink = false;
    if (NH == 0 || NH % HT) throw std::invalid_argument("xg4_attn_decode: heads must be a multiple of 32");
    // splits: enough work-groups to fill the card (~64), each walking at least 4 blocks of 32 keys with a running softmax
    const uint32_t max_blk = (max_keys + 31) / 32;
    uint32_t NB = std::max<uint32_t>(1, (64 + rows - 1) / rows);
    NB = std::min<uint32_t>(NB, std::max<uint32_t>(1, max_blk / 4));
    Ws& w = xg4dx_ws_for(q);
    const uint32_t n_ht0 = NH / HT;
    xg4dx_grow(q, w.po,  w.pcap, uint64_t(rows) * n_ht0 * NB * HT * DV, "the partial contexts");
    xg4dx_grow(q, w.pml, w.lcap, uint64_t(rows) * n_ht0 * NB * 2 * HT, "the partial maxima and sums");
    float* po = w.po; float* pml = w.pml;
    constexpr int KB = 32, RT = KB / TM, KS = NSG / (RT * CT), KPL = KB / LPH, DS = DQ / KS;
    static_assert(RT * CT * KS == NSG && (DQ / TK) % KS == 0, "tile plan");
    const uint32_t n_ht = NH / HT;
    using h8 = sycl::vec<sycl::half, 8>;
    auto ep = ie::ps(q, "xg4_attn_split", [&](sycl::handler& h) {
        sycl::local_accessor<sycl::half, 1> Kb(sycl::range<1>(size_t(KB) * DQ), h);
        sycl::local_accessor<float, 1>      St(sycl::range<1>(size_t(KS) * KB * HT), h);
        sycl::local_accessor<sycl::half, 1> P (sycl::range<1>(size_t(HT) * KB), h);
        sycl::local_accessor<float, 1>      red(sycl::range<1>(WG), h);
        sycl::local_accessor<float, 1>      mrun(sycl::range<1>(HT), h), lrun(sycl::range<1>(HT), h), alpha(sycl::range<1>(HT), h), mnew(sycl::range<1>(HT), h);
        h.parallel_for(sycl::nd_range<2>({size_t(rows) * NB, size_t(n_ht) * WG}, {1, size_t(WG)}), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint32_t tb = uint32_t(it.get_group(0)), t = tb / NB, spl = tb % NB, ht = uint32_t(it.get_group(1)), lid = uint32_t(it.get_local_id(1));
            auto sg = it.get_sub_group(); const uint32_t s = uint32_t(sg.get_group_linear_id()); auto grp = it.get_group();
            const int32_t klo = 0, khi = rpos[t];
            const int32_t* ptab = tab + uint64_t(rlane[t]) * maxp;
            const sycl::half* qtile = a16 + (uint64_t(t) * NH + uint64_t(ht) * HT) * DQ;
            if (lid < uint32_t(HT)) { mrun[lid] = kNegInf; lrun[lid] = 0.f; }
            mat::joint_matrix<sycl::sub_group, float, mat::use::accumulator, TM, TN> acc[MR][NC];
            #pragma unroll
            for (int mr = 0; mr < MR; ++mr)
                #pragma unroll
                for (int nc = 0; nc < NC; ++nc) mat::joint_matrix_fill(sg, acc[mr][nc], 0.0f);
            const uint32_t nb_row = khi < 0 ? 0u : uint32_t(khi) / KB + 1, per = (nb_row + NB - 1) / NB;
            const uint32_t blk0 = std::min(nb_row, spl * per), nblk = std::min(nb_row, blk0 + per);
            (void)klo;
            const uint32_t hh = lid % HT, jj = lid / HT;
            for (uint32_t blk = blk0; blk < nblk; ++blk) {
                // gather the block's keys (a masked or padded key reads as zeros)
                for (uint32_t u = lid; u < uint32_t(KB) * (DQ / 8); u += WG) {
                    const uint32_t r = u / (DQ / 8), d8 = u % (DQ / 8); const int32_t c = int32_t(blk * KB + r);
                    h8 v(sycl::half(0.f));
                    if (c >= klo && c <= khi) { const uint64_t k = uint64_t(ptab[c / int32_t(kXg4Page)]) * kXg4Page + uint64_t(c % int32_t(kXg4Page)); v = reinterpret_cast<const h8*>(pool + k * DQ)[d8]; }
                    reinterpret_cast<h8*>(&Kb[0] + uint64_t(r) * DQ)[d8] = v;
                }
                sycl::group_barrier(grp);
                {   // S^T [KB][HT] = Kblk . Q^T, as the decode kernel
                    const uint32_t tile = s % (RT * CT), ks = s / (RT * CT), rt = tile / CT, ct = tile % CT;
                    mat::joint_matrix<sycl::sub_group, float, mat::use::accumulator, TM, TN> sacc;
                    mat::joint_matrix_fill(sg, sacc, 0.0f);
                    #pragma unroll 4
                    for (int kk = int(ks) * DS; kk < int(ks + 1) * DS; kk += TK) {
                        mat::joint_matrix<sycl::sub_group, sycl::half, mat::use::a, TM, TK, mat::layout::row_major> a_tile;
                        mat::joint_matrix_load(sg, a_tile, Kb.template get_multi_ptr<sycl::access::decorated::no>() + uint64_t(rt * TM) * DQ + kk, DQ);
                        mat::joint_matrix<sycl::sub_group, sycl::half, mat::use::b, TK, TN, mat::layout::col_major> b_tile;
                        mat::joint_matrix_load(sg, b_tile,
                            sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(qtile + uint64_t(ct * TN) * DQ + kk), DQ);
                        mat::joint_matrix_mad(sg, sacc, a_tile, b_tile, sacc);
                    }
                    imat::joint_matrix_apply(sg, sacc, [&](float& v, size_t r, size_t c) { St[(size_t(ks) * KB + rt * TM + r) * HT + ct * TN + c] = v; });
                }
                sycl::group_barrier(grp);
                float sc[KPL]; float bm = kNegInf;
                #pragma unroll
                for (int r = 0; r < KPL; ++r) {
                    const uint32_t row = jj * KPL + uint32_t(r); const int32_t c = int32_t(blk * KB + row);
                    float v = 0.f;
                    #pragma unroll
                    for (int k2 = 0; k2 < KS; ++k2) v += St[(size_t(k2) * KB + row) * HT + hh];
                    const bool ok = c >= klo && c <= khi;
                    sc[r] = ok ? v * scaling : kNegInf;
                    bm = sycl::fmax(bm, sc[r]);
                }
                red[lid] = bm;
                sycl::group_barrier(grp);
                if (lid < uint32_t(HT)) {
                    float m8 = kNegInf; for (int j = 0; j < LPH; ++j) m8 = sycl::fmax(m8, red[j * HT + lid]);
                    const float mo = mrun[lid], mn = sycl::fmax(mo, m8);
                    mnew[lid] = mn; alpha[lid] = (mo == kNegInf) ? 0.f : sycl::exp(mo - mn);
                }
                sycl::group_barrier(grp);
                {
                    const float m_n = mnew[hh]; float ps = 0.f;
                    #pragma unroll
                    for (int r = 0; r < KPL; ++r) {
                        const uint32_t row = jj * KPL + uint32_t(r);
                        const float e = (sc[r] > kNegInf && m_n > kNegInf) ? sycl::exp(sc[r] - m_n) : 0.f;
                        P[hh * KB + row] = sycl::half(e); ps += e;
                    }
                    red[lid] = ps;
                }
                sycl::group_barrier(grp);
                if (lid < uint32_t(HT)) {
                    float l8 = 0.f; for (int j = 0; j < LPH; ++j) l8 += red[j * HT + lid];
                    lrun[lid] = lrun[lid] * alpha[lid] + l8; mrun[lid] = mnew[lid];
                }
                // rescale the running context, then add P . Vblk (this sub-group's 32 columns)
                #pragma unroll
                for (int mr = 0; mr < MR; ++mr)
                    #pragma unroll
                    for (int nc = 0; nc < NC; ++nc) imat::joint_matrix_apply(sg, acc[mr][nc], [&](float& v, size_t r, size_t) { v *= alpha[mr * TM + r]; });
                #pragma unroll
                for (int kk = 0; kk < KB; kk += TK) {
                    mat::joint_matrix<sycl::sub_group, sycl::half, mat::use::b, TK, TN, mat::layout::row_major> b_tile[NC];
                    #pragma unroll
                    for (int nc = 0; nc < NC; ++nc)
                        mat::joint_matrix_load(sg, b_tile[nc], Kb.template get_multi_ptr<sycl::access::decorated::no>() + uint64_t(kk) * DQ + s * DC + nc * TN, DQ);
                    #pragma unroll
                    for (int mr = 0; mr < MR; ++mr) {
                        mat::joint_matrix<sycl::sub_group, sycl::half, mat::use::a, TM, TK, mat::layout::row_major> a_tile;
                        mat::joint_matrix_load(sg, a_tile, P.template get_multi_ptr<sycl::access::decorated::no>() + (mr * TM) * KB + kk, KB);
                        #pragma unroll
                        for (int nc = 0; nc < NC; ++nc) mat::joint_matrix_mad(sg, acc[mr][nc], a_tile, b_tile[nc], acc[mr][nc]);
                    }
                }
                sycl::group_barrier(grp);                           // (Kb / P / St reused by the next block)
            }
            // the split's partial: running max / sum per head, the unnormalised context
            const uint64_t slot = (uint64_t(t) * n_ht + ht) * NB + spl;
            if (lid < uint32_t(HT)) { pml[slot * 2 * HT + lid] = mrun[lid]; pml[slot * 2 * HT + HT + lid] = lrun[lid]; }
            float* obase = po + slot * uint64_t(HT) * DV + s * DC;
            #pragma unroll
            for (int mr = 0; mr < MR; ++mr)
                #pragma unroll
                for (int nc = 0; nc < NC; ++nc)
                    imat::joint_matrix_apply(sg, acc[mr][nc], [&](float& v, size_t r, size_t c) { obase[uint64_t(mr * TM + r) * DV + nc * TN + c] = v; });
        });
    });
        // COMBINE: a work-group a (row, head); lane i takes dims i, i + 128, .. (coalesced across the lanes)
        return ie::ps(q, "xg4_attn_dx_combine", [&](sycl::handler& h) {
            h.depends_on(ep);
            sycl::local_accessor<float, 1> wsc(sycl::range<1>(size_t(NB) + 1), h);
            h.parallel_for(sycl::nd_range<1>(size_t(rows) * NH * CWG, CWG), [=](sycl::nd_item<1> it) {
                const uint32_t rh = uint32_t(it.get_group(0)), t = rh / NH, head = rh % NH, ht = head / HT, hh = head % HT;
                const uint32_t lid = uint32_t(it.get_local_id(0));
                const uint64_t base = (uint64_t(t) * n_ht + ht) * NB;
                // the blocks' maxima and sums read side by side (a lane a block), reduced over the work-group
                float m_l = kNegInf, l_l = 0.f;
                for (uint32_t b = lid; b < NB; b += CWG) m_l = sycl::fmax(m_l, pml[(base + b) * 2 * HT + hh]);
                const float gm = sycl::reduce_over_group(it.get_group(), m_l, sycl::maximum<float>());
                for (uint32_t b = lid; b < NB; b += CWG) {
                    const float m_b = pml[(base + b) * 2 * HT + hh];
                    const float w = (m_b == kNegInf) ? 0.f : sycl::exp(m_b - gm);
                    wsc[b] = w; l_l += w * pml[(base + b) * 2 * HT + HT + hh];
                }
                const float gl = sycl::reduce_over_group(it.get_group(), l_l, sycl::plus<float>());
                if (lid == 0) {
                    float corr = 1.f, ll = gl;
                    if (use_sink) {
                        const float sv = sinks[head], m_f = sycl::fmax(gm, sv);
                        corr = (gm == kNegInf) ? 0.f : sycl::exp(gm - m_f);
                        ll = gl * corr + sycl::exp(sv - m_f);
                    }
                    wsc[NB] = corr * ((ll > 0.f) ? (1.f / ll) : 0.f);
                }
                sycl::group_barrier(it.get_group());
                const float sc = wsc[NB];
                sycl::half* yb = ctx16 + uint64_t(rh) * DQ;
                for (uint32_t d = lid; d < uint32_t(DV); d += CWG) {          // four independent sums: the loads overlap
                    const float* pb = po + base * uint64_t(HT) * DV + uint64_t(hh) * DV + d;
                    const uint64_t st = uint64_t(HT) * DV;
                    float o0 = 0.f, o1 = 0.f, o2 = 0.f, o3 = 0.f; uint32_t b = 0;
                    // (a block with no allowed key wrote no partial sums: weight 0, its stale values never read into a sum)
                    auto term = [&](uint32_t bb) { const float w = wsc[bb]; return w != 0.f ? w * pb[bb * st] : 0.f; };
                    for (; b + 3 < NB; b += 4) { o0 += term(b); o1 += term(b + 1); o2 += term(b + 2); o3 += term(b + 3); }
                    for (; b < NB; ++b) o0 += term(b);
                    yb[d] = sycl::half(((o0 + o1) + (o2 + o3)) * sc);
                }
                for (uint32_t d = uint32_t(DV) + lid; d < uint32_t(DQ); d += CWG) yb[d] = sycl::half(0.f);   // the rotary dims
            });
        });
}

sycl::event xg4_attn_rows(sycl::queue& q, const sycl::half* a16, const sycl::half* pool, const int32_t* tab, uint32_t maxp,
                          const int32_t* rlane, const int32_t* rpos, sycl::half* ctx16, uint32_t rows, uint32_t NH, float scaling) {
    using namespace xg4dx;
    const float* sinks = nullptr; constexpr bool use_sink = false;
    if (NH == 0 || NH % HT) throw std::invalid_argument("xg4_attn_rows: heads must be a multiple of 32");
    constexpr int KB = 32, RT = KB / TM, KS = NSG / (RT * CT), KPL = KB / LPH, DS = DQ / KS;
    static_assert(RT * CT * KS == NSG && (DQ / TK) % KS == 0, "tile plan");
    const uint32_t n_ht = NH / HT;
    using h8 = sycl::vec<sycl::half, 8>;
    return ie::ps(q, "xg4_attn_rows", [&](sycl::handler& h) {
        sycl::local_accessor<sycl::half, 1> Kb(sycl::range<1>(size_t(KB) * DQ), h);
        sycl::local_accessor<float, 1>      St(sycl::range<1>(size_t(KS) * KB * HT), h);
        sycl::local_accessor<sycl::half, 1> P (sycl::range<1>(size_t(HT) * KB), h);
        sycl::local_accessor<float, 1>      red(sycl::range<1>(WG), h);
        sycl::local_accessor<float, 1>      mrun(sycl::range<1>(HT), h), lrun(sycl::range<1>(HT), h), alpha(sycl::range<1>(HT), h), mnew(sycl::range<1>(HT), h);
        h.parallel_for(sycl::nd_range<2>({size_t(rows), size_t(n_ht) * WG}, {1, size_t(WG)}), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint32_t t = uint32_t(it.get_group(0)), ht = uint32_t(it.get_group(1)), lid = uint32_t(it.get_local_id(1));
            auto sg = it.get_sub_group(); const uint32_t s = uint32_t(sg.get_group_linear_id()); auto grp = it.get_group();
            const int32_t klo = 0, khi = rpos[t];
            const int32_t* ptab = tab + uint64_t(rlane[t]) * maxp;
            const sycl::half* qtile = a16 + (uint64_t(t) * NH + uint64_t(ht) * HT) * DQ;
            if (lid < uint32_t(HT)) { mrun[lid] = kNegInf; lrun[lid] = 0.f; }
            mat::joint_matrix<sycl::sub_group, float, mat::use::accumulator, TM, TN> acc[MR][NC];
            #pragma unroll
            for (int mr = 0; mr < MR; ++mr)
                #pragma unroll
                for (int nc = 0; nc < NC; ++nc) mat::joint_matrix_fill(sg, acc[mr][nc], 0.0f);
            const uint32_t nblk = khi < 0 ? 0u : uint32_t(khi) / KB + 1, blk0 = klo > 0 ? uint32_t(klo) / KB : 0u;
            const uint32_t hh = lid % HT, jj = lid / HT;
            for (uint32_t blk = blk0; blk < nblk; ++blk) {
                // gather the block's keys (a masked or padded key reads as zeros)
                for (uint32_t u = lid; u < uint32_t(KB) * (DQ / 8); u += WG) {
                    const uint32_t r = u / (DQ / 8), d8 = u % (DQ / 8); const int32_t c = int32_t(blk * KB + r);
                    h8 v(sycl::half(0.f));
                    if (c >= klo && c <= khi) { const uint64_t k = uint64_t(ptab[c / int32_t(kXg4Page)]) * kXg4Page + uint64_t(c % int32_t(kXg4Page)); v = reinterpret_cast<const h8*>(pool + k * DQ)[d8]; }
                    reinterpret_cast<h8*>(&Kb[0] + uint64_t(r) * DQ)[d8] = v;
                }
                sycl::group_barrier(grp);
                {   // S^T [KB][HT] = Kblk . Q^T, as the decode kernel
                    const uint32_t tile = s % (RT * CT), ks = s / (RT * CT), rt = tile / CT, ct = tile % CT;
                    mat::joint_matrix<sycl::sub_group, float, mat::use::accumulator, TM, TN> sacc;
                    mat::joint_matrix_fill(sg, sacc, 0.0f);
                    #pragma unroll 4
                    for (int kk = int(ks) * DS; kk < int(ks + 1) * DS; kk += TK) {
                        mat::joint_matrix<sycl::sub_group, sycl::half, mat::use::a, TM, TK, mat::layout::row_major> a_tile;
                        mat::joint_matrix_load(sg, a_tile, Kb.template get_multi_ptr<sycl::access::decorated::no>() + uint64_t(rt * TM) * DQ + kk, DQ);
                        mat::joint_matrix<sycl::sub_group, sycl::half, mat::use::b, TK, TN, mat::layout::col_major> b_tile;
                        mat::joint_matrix_load(sg, b_tile,
                            sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(qtile + uint64_t(ct * TN) * DQ + kk), DQ);
                        mat::joint_matrix_mad(sg, sacc, a_tile, b_tile, sacc);
                    }
                    imat::joint_matrix_apply(sg, sacc, [&](float& v, size_t r, size_t c) { St[(size_t(ks) * KB + rt * TM + r) * HT + ct * TN + c] = v; });
                }
                sycl::group_barrier(grp);
                float sc[KPL]; float bm = kNegInf;
                #pragma unroll
                for (int r = 0; r < KPL; ++r) {
                    const uint32_t row = jj * KPL + uint32_t(r); const int32_t c = int32_t(blk * KB + row);
                    float v = 0.f;
                    #pragma unroll
                    for (int k2 = 0; k2 < KS; ++k2) v += St[(size_t(k2) * KB + row) * HT + hh];
                    const bool ok = c >= klo && c <= khi;
                    sc[r] = ok ? v * scaling : kNegInf;
                    bm = sycl::fmax(bm, sc[r]);
                }
                red[lid] = bm;
                sycl::group_barrier(grp);
                if (lid < uint32_t(HT)) {
                    float m8 = kNegInf; for (int j = 0; j < LPH; ++j) m8 = sycl::fmax(m8, red[j * HT + lid]);
                    const float mo = mrun[lid], mn = sycl::fmax(mo, m8);
                    mnew[lid] = mn; alpha[lid] = (mo == kNegInf) ? 0.f : sycl::exp(mo - mn);
                }
                sycl::group_barrier(grp);
                {
                    const float m_n = mnew[hh]; float ps = 0.f;
                    #pragma unroll
                    for (int r = 0; r < KPL; ++r) {
                        const uint32_t row = jj * KPL + uint32_t(r);
                        const float e = (sc[r] > kNegInf && m_n > kNegInf) ? sycl::exp(sc[r] - m_n) : 0.f;
                        P[hh * KB + row] = sycl::half(e); ps += e;
                    }
                    red[lid] = ps;
                }
                sycl::group_barrier(grp);
                if (lid < uint32_t(HT)) {
                    float l8 = 0.f; for (int j = 0; j < LPH; ++j) l8 += red[j * HT + lid];
                    lrun[lid] = lrun[lid] * alpha[lid] + l8; mrun[lid] = mnew[lid];
                }
                // rescale the running context, then add P . Vblk (this sub-group's 32 columns)
                #pragma unroll
                for (int mr = 0; mr < MR; ++mr)
                    #pragma unroll
                    for (int nc = 0; nc < NC; ++nc) imat::joint_matrix_apply(sg, acc[mr][nc], [&](float& v, size_t r, size_t) { v *= alpha[mr * TM + r]; });
                #pragma unroll
                for (int kk = 0; kk < KB; kk += TK) {
                    mat::joint_matrix<sycl::sub_group, sycl::half, mat::use::b, TK, TN, mat::layout::row_major> b_tile[NC];
                    #pragma unroll
                    for (int nc = 0; nc < NC; ++nc)
                        mat::joint_matrix_load(sg, b_tile[nc], Kb.template get_multi_ptr<sycl::access::decorated::no>() + uint64_t(kk) * DQ + s * DC + nc * TN, DQ);
                    #pragma unroll
                    for (int mr = 0; mr < MR; ++mr) {
                        mat::joint_matrix<sycl::sub_group, sycl::half, mat::use::a, TM, TK, mat::layout::row_major> a_tile;
                        mat::joint_matrix_load(sg, a_tile, P.template get_multi_ptr<sycl::access::decorated::no>() + (mr * TM) * KB + kk, KB);
                        #pragma unroll
                        for (int nc = 0; nc < NC; ++nc) mat::joint_matrix_mad(sg, acc[mr][nc], a_tile, b_tile[nc], acc[mr][nc]);
                    }
                }
                sycl::group_barrier(grp);                           // (Kb / P / St reused by the next block)
            }
            // the sink and the normalisation, per head: scale[hh] into alpha
            if (lid < uint32_t(HT)) {
                const float gm = mrun[lid], gl = lrun[lid];
                float corr = 1.f, ll = gl;
                if (use_sink) {
                    const float sv = sinks[ht * HT + lid], m_f = sycl::fmax(gm, sv);
                    corr = (gm == kNegInf) ? 0.f : sycl::exp(gm - m_f);
                    ll = gl * corr + sycl::exp(sv - m_f);
                }
                alpha[lid] = corr * ((ll > 0.f) ? (1.f / ll) : 0.f);
            }
            sycl::group_barrier(grp);
            sycl::half* yb = ctx16 + (uint64_t(t) * NH + uint64_t(ht) * HT) * DQ;
            #pragma unroll
            for (int mr = 0; mr < MR; ++mr)
                #pragma unroll
                for (int nc = 0; nc < NC; ++nc)
                    imat::joint_matrix_apply(sg, acc[mr][nc], [&](float& v, size_t r, size_t c) {
                        yb[uint64_t(mr * TM + r) * DQ + s * DC + nc * TN + c] = sycl::half(v * alpha[mr * TM + r]); });
            for (uint32_t u = lid; u < uint32_t(HT) * (DQ - DV); u += WG) yb[uint64_t(u / (DQ - DV)) * DQ + DV + u % (DQ - DV)] = sycl::half(0.f);
        });
    });
}

// Prompt rows, 4 at a time (2026-11-20): a work-group per GROUP of up to 4 rows of one lane (groups[2g] = first row,
// groups[2g + 1] = count), 512 lanes, the 4 rows x 32 heads = 128 query vectors sharing every key block -- a key block is
// read once for 4 rows, not once per row. S^T [32 keys][128 vectors]: 32 sub-groups, one 8 x 16 tile each over the full
// depth; P . V: each sub-group owns 16 of the 512 context columns for all 128 vectors (16 accumulators). Each row keeps its
// own causal limit (rpos). The single-row kernel above (IE_XG4_ATTN_OLD=1) is the reference.
sycl::event xg4_attn_rows4(sycl::queue& q, const sycl::half* a16, const sycl::half* pool, const int32_t* tab, uint32_t maxp,
                           const int32_t* rlane, const int32_t* rpos, const int32_t* groups, uint32_t n_groups, sycl::half* ctx16,
                           uint32_t NH, float scaling) {
    using namespace xg4dx;
    if (NH == 0 || NH % HT) throw std::invalid_argument("xg4_attn_rows4: heads must be a multiple of 32");
    constexpr int R = 4, NV = R * HT;                      // 128 query vectors
    constexpr int NSG4 = 32, WG4 = SG * NSG4;              // 512 lanes
    constexpr int KB = 32;
    constexpr int RT4 = KB / TM, CT4 = NV / TN;            // 4 key row tiles x 8 vector tiles = 32 tiles: one a sub-group
    constexpr int DC4 = DV / NSG4;                         // 16 context columns a sub-group
    constexpr int MR4 = NV / TM;                           // 16 vector row tiles (P . V)
    constexpr int LPV = WG4 / NV, KPL4 = KB / LPV;         // softmax: 4 lanes a vector, 8 keys a lane
    static_assert(RT4 * CT4 == NSG4 && DC4 == TN, "tile plan");
    const uint32_t n_ht = NH / HT;
    using h8 = sycl::vec<sycl::half, 8>;
    return ie::ps(q, "xg4_attn_rows4", [&](sycl::handler& h) {
        sycl::local_accessor<sycl::half, 1> Kb(sycl::range<1>(size_t(KB) * DQ), h);
        sycl::local_accessor<float, 1>      St(sycl::range<1>(size_t(KB) * NV), h);
        sycl::local_accessor<sycl::half, 1> P (sycl::range<1>(size_t(NV) * KB), h);
        sycl::local_accessor<float, 1>      red(sycl::range<1>(WG4), h);
        sycl::local_accessor<float, 1>      mrun(sycl::range<1>(NV), h), lrun(sycl::range<1>(NV), h), alpha(sycl::range<1>(NV), h), mnew(sycl::range<1>(NV), h);
        h.parallel_for(sycl::nd_range<2>({size_t(n_groups), size_t(n_ht) * WG4}, {1, size_t(WG4)}), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint32_t gi = uint32_t(it.get_group(0)), ht = uint32_t(it.get_group(1)), lid = uint32_t(it.get_local_id(1));
            auto sg = it.get_sub_group(); const uint32_t s = uint32_t(sg.get_group_linear_id()); auto grp = it.get_group();
            const int32_t r0 = groups[2 * gi], nr = groups[2 * gi + 1];
            const int32_t* ptab = tab + uint64_t(rlane[r0]) * maxp;
            int32_t lim[R]; int32_t khi = -1;
            #pragma unroll
            for (int i = 0; i < R; ++i) { lim[i] = i < nr ? rpos[r0 + i] : -1; khi = sycl::max(khi, lim[i]); }
            if (lid < uint32_t(NV)) { mrun[lid] = kNegInf; lrun[lid] = 0.f; }
            mat::joint_matrix<sycl::sub_group, float, mat::use::accumulator, TM, TN> acc[MR4];
            #pragma unroll
            for (int mr = 0; mr < MR4; ++mr) mat::joint_matrix_fill(sg, acc[mr], 0.0f);
            const uint32_t nblk = khi < 0 ? 0u : uint32_t(khi) / KB + 1;
            // this lane's softmax share: vector vv (row vv / 32, head vv % 32), keys jj * 8 ..
            const uint32_t vv = lid % NV, jj = lid / NV;
            const int32_t vlim = lim[vv / HT];
            // the S tile of this sub-group: key rows rt * 8, vectors ct * 16 (row ct / 2, heads (ct % 2) * 16)
            const uint32_t rt = s / CT4, ct = s % CT4;
            const int32_t qrow = int32_t(ct * TN) / HT;
            const sycl::half* qtile = a16 + (uint64_t(r0 + sycl::min(qrow, nr - 1)) * NH + uint64_t(ht) * HT + uint64_t((ct * TN) % HT)) * DQ;
            for (uint32_t blk = 0; blk < nblk; ++blk) {
                for (uint32_t u = lid; u < uint32_t(KB) * (DQ / 8); u += WG4) {
                    const uint32_t r = u / (DQ / 8), d8 = u % (DQ / 8); const int32_t c = int32_t(blk * KB + r);
                    h8 v(sycl::half(0.f));
                    if (c <= khi) { const uint64_t k = uint64_t(ptab[c / int32_t(kXg4Page)]) * kXg4Page + uint64_t(c % int32_t(kXg4Page)); v = reinterpret_cast<const h8*>(pool + k * DQ)[d8]; }
                    reinterpret_cast<h8*>(&Kb[0] + uint64_t(r) * DQ)[d8] = v;
                }
                sycl::group_barrier(grp);
                {   // S^T tile [8 keys][16 vectors] over the full 576
                    mat::joint_matrix<sycl::sub_group, float, mat::use::accumulator, TM, TN> sacc;
                    mat::joint_matrix_fill(sg, sacc, 0.0f);
                    #pragma unroll 4
                    for (int kk = 0; kk < DQ; kk += TK) {
                        mat::joint_matrix<sycl::sub_group, sycl::half, mat::use::a, TM, TK, mat::layout::row_major> a_tile;
                        mat::joint_matrix_load(sg, a_tile, Kb.template get_multi_ptr<sycl::access::decorated::no>() + uint64_t(rt * TM) * DQ + kk, DQ);
                        mat::joint_matrix<sycl::sub_group, sycl::half, mat::use::b, TK, TN, mat::layout::col_major> b_tile;
                        mat::joint_matrix_load(sg, b_tile, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(qtile + kk), DQ);
                        mat::joint_matrix_mad(sg, sacc, a_tile, b_tile, sacc);
                    }
                    imat::joint_matrix_apply(sg, sacc, [&](float& v, size_t r, size_t c) { St[(rt * TM + r) * NV + ct * TN + c] = v; });
                }
                sycl::group_barrier(grp);
                float sc[KPL4]; float bm = kNegInf;
                #pragma unroll
                for (int r = 0; r < KPL4; ++r) {
                    const uint32_t row = jj * KPL4 + uint32_t(r); const int32_t c = int32_t(blk * KB + row);
                    sc[r] = c <= vlim ? St[row * NV + vv] * scaling : kNegInf;
                    bm = sycl::fmax(bm, sc[r]);
                }
                red[lid] = bm;
                sycl::group_barrier(grp);
                if (lid < uint32_t(NV)) {
                    float m4 = kNegInf; for (int j = 0; j < LPV; ++j) m4 = sycl::fmax(m4, red[j * NV + lid]);
                    const float mo = mrun[lid], mn = sycl::fmax(mo, m4);
                    mnew[lid] = mn; alpha[lid] = (mo == kNegInf) ? (mn == kNegInf ? 1.f : 0.f) : sycl::exp(mo - mn);
                }
                sycl::group_barrier(grp);
                {
                    const float m_n = mnew[vv]; float ps = 0.f;
                    #pragma unroll
                    for (int r = 0; r < KPL4; ++r) {
                        const uint32_t row = jj * KPL4 + uint32_t(r);
                        const float e = (sc[r] > kNegInf && m_n > kNegInf) ? sycl::exp(sc[r] - m_n) : 0.f;
                        P[vv * KB + row] = sycl::half(e); ps += e;
                    }
                    red[lid] = ps;
                }
                sycl::group_barrier(grp);
                if (lid < uint32_t(NV)) {
                    float l4 = 0.f; for (int j = 0; j < LPV; ++j) l4 += red[j * NV + lid];
                    lrun[lid] = lrun[lid] * alpha[lid] + l4; mrun[lid] = mnew[lid];
                }
                #pragma unroll
                for (int mr = 0; mr < MR4; ++mr) imat::joint_matrix_apply(sg, acc[mr], [&](float& v, size_t r, size_t) { v *= alpha[mr * TM + r]; });
                #pragma unroll
                for (int kk = 0; kk < KB; kk += TK) {
                    mat::joint_matrix<sycl::sub_group, sycl::half, mat::use::b, TK, TN, mat::layout::row_major> b_tile;
                    mat::joint_matrix_load(sg, b_tile, Kb.template get_multi_ptr<sycl::access::decorated::no>() + uint64_t(kk) * DQ + s * DC4, DQ);
                    #pragma unroll
                    for (int mr = 0; mr < MR4; ++mr) {
                        mat::joint_matrix<sycl::sub_group, sycl::half, mat::use::a, TM, TK, mat::layout::row_major> a_tile;
                        mat::joint_matrix_load(sg, a_tile, P.template get_multi_ptr<sycl::access::decorated::no>() + (mr * TM) * KB + kk, KB);
                        mat::joint_matrix_mad(sg, acc[mr], a_tile, b_tile, acc[mr]);
                    }
                }
                sycl::group_barrier(grp);
            }
            if (lid < uint32_t(NV)) { const float gl = lrun[lid]; alpha[lid] = gl > 0.f ? 1.f / gl : 0.f; }
            sycl::group_barrier(grp);
            #pragma unroll
            for (int mr = 0; mr < MR4; ++mr)
                imat::joint_matrix_apply(sg, acc[mr], [&](float& v, size_t r, size_t c) {
                    const uint32_t vec = mr * TM + uint32_t(r), row = vec / HT, hd = vec % HT;
                    if (int32_t(row) < nr) ctx16[(uint64_t(r0 + int32_t(row)) * NH + uint64_t(ht) * HT + hd) * DQ + s * DC4 + c] = sycl::half(v * alpha[vec]); });
            for (uint32_t u = lid; u < uint32_t(NV) * (DQ - DV); u += WG4) {
                const uint32_t vec = u / (DQ - DV), row = vec / HT, hd = vec % HT;
                if (int32_t(row) < nr) ctx16[(uint64_t(r0 + int32_t(row)) * NH + uint64_t(ht) * HT + hd) * DQ + DV + u % (DQ - DV)] = sycl::half(0.f);
            }
        });
    });
}

}  // namespace ie
