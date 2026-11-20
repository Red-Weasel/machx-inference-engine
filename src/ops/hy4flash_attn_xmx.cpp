// src/ops/hy4flash_attn_xmx.cpp -- Hy4-flash decode attention on the XMX units: split-key flash decoding after
// deepseek41_attn_xmx.cpp (the same tile body; Hy4's keys are 576 wide for the scores and 512 for the context).
//
// The attention of a decode token is arithmetic, not memory: every head of a layer scores the same latent keys (MLA), so
// 32 heads x 576 dims x ~520 keys is ~10 M products for the scores and as many for the context, against ~0.6 MB of keys.
// The vector kernel (hy4f_attn_decode) did them at ~0.4 T products a second (51 us a layer-half, 2026-11-19); the
// matrix units do them as 8 x 16 x 16 tiles. Three launches:
//   1. GATHER: row r's column c = key c of the row's axis (or its c-th selected key) as fp16 [rows, NCOLP, 576], with a
//      mask (0 allowed, -inf not) -- columns outside [lo[r], hi[r]] and the padding to 64-key blocks are masked zeros;
//   2. PARTIAL per (row, 32-head tile, 64-key block): S^T = Kblk . Q^T over 576 dims, the block's softmax (max m_b, sum
//      l_b, P = exp(S - m_b) in fp16), O_b = P . Vblk over the 512 latent dims -> (m_b, l_b, O_b);
//   3. COMBINE per (row, tile): the blocks rescaled to the common max, the head's sink logit in the denominator, 1 / l.
// The context's last 64 dims (the keys' rotary part) are written as zeros: W_v is zero under them (attn_heads).
// fp16 operands, fp32 accumulation; P in fp16 as the three-launch form's p16 was. NOT bit-identical to the vector
// kernel; gated by the mini gate and the real-weights yardstick.
//
// Every kernel-launching helper is prefixed hy4fdx_ (the SYCL kernel-name collision trap).
#include "ie/hy4flash_q8.hpp"
#include "ie/kernel_profiler.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <deque>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace ie {
namespace hy4fdx {

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
Ws& hy4fdx_ws_for(sycl::queue& q) {
    static std::deque<std::pair<sycl::queue, Ws>> all;
    static std::mutex m;
    std::lock_guard<std::mutex> lk(m);
    for (auto& p : all) if (p.first == q) return p.second;
    all.emplace_back(q, Ws{});
    return all.back().second;
}
template <class V>
void hy4fdx_grow(sycl::queue& q, V*& p, uint64_t& cap, uint64_t need, const char* what) {
    if (need <= cap) return;
    if (p) { q.wait(); sycl::free(p, q); p = nullptr; cap = 0; }
    p = sycl::malloc_device<V>(size_t(need), q);
    if (!p) throw std::runtime_error(std::string("hy4f_attn_decode_xmx: could not allocate ") + what);
    cap = need;
}

}  // namespace hy4fdx

bool hy4f_attn_xmx_fits(uint32_t NH, uint32_t LW, uint32_t KL) {
    return LW == uint32_t(hy4fdx::DQ) && KL == uint32_t(hy4fdx::DV) && NH > 0 && NH % uint32_t(hy4fdx::HT) == 0;
}

sycl::event hy4f_attn_decode_xmx(sycl::queue& q, const sycl::half* a16, const sycl::half* keys, const int32_t* sel, uint32_t S,
                                 uint32_t n_axis, const int32_t* lo, const int32_t* hi, const float* sinks, sycl::half* ctx16,
                                 uint32_t rows, uint32_t NH, float scaling, bool use_sink) {
    using namespace hy4fdx;
    if (NH == 0 || NH % HT) throw std::invalid_argument("hy4f_attn_decode_xmx: heads must be a multiple of 32");
    const uint32_t ncol = sel ? S : n_axis;
    // ~32 blocks: 16 keys a block up to 512 keys, 32 up to 1,024, then 64 (a block is a work-group; 32 cores)
    const uint32_t KBc = ncol <= 512 ? 16u : ncol <= 1024 ? 32u : 64u;
    const uint32_t NB = (ncol + KBc - 1) / KBc, NCOLP = NB * KBc, n_ht = NH / HT;
    Ws& w = hy4fdx_ws_for(q);
    hy4fdx_grow(q, w.kgat, w.kcap, uint64_t(rows) * NCOLP * DQ, "the gathered keys");
    hy4fdx_grow(q, w.mgat, w.mcap, uint64_t(rows) * NCOLP, "the gathered mask");
    hy4fdx_grow(q, w.po,   w.pcap, uint64_t(rows) * n_ht * NB * HT * DV, "the partial contexts");
    hy4fdx_grow(q, w.pml,  w.lcap, uint64_t(rows) * n_ht * NB * 2 * HT, "the partial maxima and sums");
    sycl::half* kgat = w.kgat; float* mgat = w.mgat; float* po = w.po; float* pml = w.pml;
    using h8 = sycl::vec<sycl::half, 8>;

    // ---- 1. GATHER ----
    auto eg = ie::ps(q, "hy4f_attn_dx_gather", [&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<2>({size_t(rows) * NCOLP, size_t(DQ / 8)}, {1, size_t(DQ / 8)}), [=](sycl::nd_item<2> it) {
            const uint32_t rc = uint32_t(it.get_group(0)), r = rc / NCOLP, c = rc % NCOLP, d8 = uint32_t(it.get_local_id(1));
            const bool ok = c < ncol && int32_t(c) >= lo[r] && int32_t(c) <= hi[r];
            h8 v(sycl::half(0.f));
            if (ok) { const uint64_t k = sel ? uint64_t(sel[uint64_t(r) * S + c]) : uint64_t(c); v = reinterpret_cast<const h8*>(keys + k * DQ)[d8]; }
            reinterpret_cast<h8*>(kgat + uint64_t(rc) * DQ)[d8] = v;
            if (d8 == 0) mgat[rc] = ok ? 0.f : kNegInf;
        });
    });

    // ---- 2. PARTIAL (KB keys a block) and 3. COMBINE ----
    auto launch = [&](auto kb_tag) {
        constexpr int KB = decltype(kb_tag)::value;
        constexpr int RT = KB / TM;                       // key row tiles of the S step
        constexpr int KS = NSG / (RT * CT);               // the S step's 576 dims split this many ways (a sub-group each)
        constexpr int KPL = KB / LPH;                     // softmax keys a lane
        static_assert(RT * CT * KS == NSG && (DQ / TK) % KS == 0 && KB % TK == 0, "tile plan");
        auto ep = ie::ps(q, "hy4f_attn_dx_part", [&](sycl::handler& h) {
            h.depends_on(eg);
            sycl::local_accessor<float, 1>      St(sycl::range<1>(size_t(KS) * KB * HT), h);
            sycl::local_accessor<sycl::half, 1> P (sycl::range<1>(size_t(HT) * KB), h);
            sycl::local_accessor<float, 1>      mk(sycl::range<1>(KB), h);
            sycl::local_accessor<float, 1>      red(sycl::range<1>(WG), h);
            sycl::local_accessor<float, 1>      mb(sycl::range<1>(HT), h);
            h.parallel_for(sycl::nd_range<2>({size_t(rows) * NB, size_t(n_ht) * WG}, {1, size_t(WG)}), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
                const uint32_t tb = uint32_t(it.get_group(0)), t = tb / NB, blk = tb % NB;
                const uint32_t ht = uint32_t(it.get_group(1)), lid = uint32_t(it.get_local_id(1));
                auto sg = it.get_sub_group(); const uint32_t s = uint32_t(sg.get_group_linear_id());
                auto grp = it.get_group();
                const uint64_t slot = (uint64_t(t) * n_ht + ht) * NB + blk;
                if (lid < uint32_t(KB)) mk[lid] = mgat[uint64_t(t) * NCOLP + uint64_t(blk) * KB + lid];
                sycl::group_barrier(grp);
                const bool allowed = sycl::any_of_group(grp, lid < uint32_t(KB) && mk[lid] > kNegInf);
                if (!allowed) {
                    if (lid < uint32_t(HT)) { pml[slot * 2 * HT + lid] = kNegInf; pml[slot * 2 * HT + HT + lid] = 0.f; }
                    return;
                }
                // Q^T is read where it lies: a16's rows [head][576] are Q^T's columns (a column-major B tile)
                const sycl::half* qtile = a16 + (uint64_t(t) * NH + uint64_t(ht) * HT) * DQ;
                const sycl::half* kblk = kgat + (uint64_t(t) * NCOLP + uint64_t(blk) * KB) * DQ;
                {   // S^T [KB][HT] = Kblk . Q^T: RT x CT tiles, each summed over KS slices of the 576 dims
                    const uint32_t tile = s % (RT * CT), ks = s / (RT * CT), rt = tile / CT, ct = tile % CT;
                    constexpr int DS = DQ / KS;
                    mat::joint_matrix<sycl::sub_group, float, mat::use::accumulator, TM, TN> sacc;
                    mat::joint_matrix_fill(sg, sacc, 0.0f);
                    const sycl::half* kbase = kblk + uint64_t(rt * TM) * DQ;
                    #pragma unroll 4
                    for (int kk = int(ks) * DS; kk < int(ks + 1) * DS; kk += TK) {
                        mat::joint_matrix<sycl::sub_group, sycl::half, mat::use::a, TM, TK, mat::layout::row_major> a_tile;
                        mat::joint_matrix_load(sg, a_tile,
                            sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(kbase + kk), DQ);
                        mat::joint_matrix<sycl::sub_group, sycl::half, mat::use::b, TK, TN, mat::layout::col_major> b_tile;
                        mat::joint_matrix_load(sg, b_tile,
                            sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(qtile + uint64_t(ct * TN) * DQ + kk), DQ);
                        mat::joint_matrix_mad(sg, sacc, a_tile, b_tile, sacc);
                    }
                    imat::joint_matrix_apply(sg, sacc, [&](float& v, size_t r, size_t c) { St[(size_t(ks) * KB + rt * TM + r) * HT + ct * TN + c] = v; });
                }
                sycl::group_barrier(grp);
                const uint32_t hh = lid % HT, jj = lid / HT;          // the block softmax: LPH lanes a head, KPL keys each
                float sc[KPL]; float bm = kNegInf;
                #pragma unroll
                for (int r = 0; r < KPL; ++r) {
                    const uint32_t row = jj * KPL + uint32_t(r);
                    float v = 0.f;
                    #pragma unroll
                    for (int k2 = 0; k2 < KS; ++k2) v += St[(size_t(k2) * KB + row) * HT + hh];
                    sc[r] = (mk[row] > kNegInf) ? v * scaling : kNegInf;
                    bm = sycl::fmax(bm, sc[r]);
                }
                red[lid] = bm;
                sycl::group_barrier(grp);
                if (lid < uint32_t(HT)) { float m8 = kNegInf; for (int j = 0; j < LPH; ++j) m8 = sycl::fmax(m8, red[j * HT + lid]); mb[lid] = m8; }
                sycl::group_barrier(grp);
                {
                    const float m_n = mb[hh]; float ps = 0.f;
                    #pragma unroll
                    for (int r = 0; r < KPL; ++r) {
                        const uint32_t row = jj * KPL + uint32_t(r);
                        const float e = (sc[r] > kNegInf) ? sycl::exp(sc[r] - m_n) : 0.f;
                        P[hh * KB + row] = sycl::half(e);
                        ps += e;
                    }
                    red[lid] = ps;
                }
                sycl::group_barrier(grp);
                if (lid < uint32_t(HT)) {
                    float l8 = 0.f; for (int j = 0; j < LPH; ++j) l8 += red[j * HT + lid];
                    pml[slot * 2 * HT + lid] = mb[lid]; pml[slot * 2 * HT + HT + lid] = l8;
                }
                // O_b [HT][DV] = P [HT][KB] . Vblk [KB][DV]: this sub-group's 32 columns
                mat::joint_matrix<sycl::sub_group, float, mat::use::accumulator, TM, TN> acc[MR][NC];
                #pragma unroll
                for (int mr = 0; mr < MR; ++mr)
                    #pragma unroll
                    for (int nc = 0; nc < NC; ++nc) mat::joint_matrix_fill(sg, acc[mr][nc], 0.0f);
                #pragma unroll
                for (int kk = 0; kk < KB; kk += TK) {
                    mat::joint_matrix<sycl::sub_group, sycl::half, mat::use::b, TK, TN, mat::layout::row_major> b_tile[NC];
                    #pragma unroll
                    for (int nc = 0; nc < NC; ++nc)
                        mat::joint_matrix_load(sg, b_tile[nc],
                            sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(kblk + uint64_t(kk) * DQ + s * DC + nc * TN), DQ);
                    #pragma unroll
                    for (int mr = 0; mr < MR; ++mr) {
                        mat::joint_matrix<sycl::sub_group, sycl::half, mat::use::a, TM, TK, mat::layout::row_major> a_tile;
                        mat::joint_matrix_load(sg, a_tile, P.template get_multi_ptr<sycl::access::decorated::no>() + (mr * TM) * KB + kk, KB);
                        #pragma unroll
                        for (int nc = 0; nc < NC; ++nc) mat::joint_matrix_mad(sg, acc[mr][nc], a_tile, b_tile[nc], acc[mr][nc]);
                    }
                }
                float* obase = po + slot * uint64_t(HT) * DV + s * DC;
                #pragma unroll
                for (int mr = 0; mr < MR; ++mr)
                    #pragma unroll
                    for (int nc = 0; nc < NC; ++nc)
                        imat::joint_matrix_apply(sg, acc[mr][nc], [&](float& v, size_t r, size_t c) { obase[uint64_t(mr * TM + r) * DV + nc * TN + c] = v; });
            });
        });
        // COMBINE: a work-group a (row, head); lane i takes dims i, i + 128, .. (coalesced across the lanes)
        return ie::ps(q, "hy4f_attn_dx_combine", [&](sycl::handler& h) {
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
    };
    if (KBc == 16) return launch(std::integral_constant<int, 16>{});
    if (KBc == 32) return launch(std::integral_constant<int, 32>{});
    return launch(std::integral_constant<int, 64>{});
}


// Prompt rows: one work-group a (row, 32-head tile) walks ALL of the row's chosen keys (32 a block, gathered through sel
// into local memory) with a running softmax per head -- the accumulators rescaled in registers -- and applies the sink at
// the end as the combine above does. No workspace: a chunk's rows give the parallelism the decode kernel's key blocks give.
sycl::event hy4f_attn_rows_xmx(sycl::queue& q, const sycl::half* a16, const sycl::half* keys, const int32_t* sel, uint32_t S,
                               const int32_t* lo, const int32_t* hi, const float* sinks, sycl::half* ctx16,
                               uint32_t rows, uint32_t NH, float scaling, bool use_sink) {
    using namespace hy4fdx;
    if (NH == 0 || NH % HT) throw std::invalid_argument("hy4f_attn_rows_xmx: heads must be a multiple of 32");
    constexpr int KB = 32, RT = KB / TM, KS = NSG / (RT * CT), KPL = KB / LPH, DS = DQ / KS;
    static_assert(RT * CT * KS == NSG && (DQ / TK) % KS == 0, "tile plan");
    const uint32_t n_ht = NH / HT;
    using h8 = sycl::vec<sycl::half, 8>;
    return ie::ps(q, "hy4f_attn_rows_xmx", [&](sycl::handler& h) {
        sycl::local_accessor<sycl::half, 1> Kb(sycl::range<1>(size_t(KB) * DQ), h);
        sycl::local_accessor<float, 1>      St(sycl::range<1>(size_t(KS) * KB * HT), h);
        sycl::local_accessor<sycl::half, 1> P (sycl::range<1>(size_t(HT) * KB), h);
        sycl::local_accessor<float, 1>      red(sycl::range<1>(WG), h);
        sycl::local_accessor<float, 1>      mrun(sycl::range<1>(HT), h), lrun(sycl::range<1>(HT), h), alpha(sycl::range<1>(HT), h), mnew(sycl::range<1>(HT), h);
        h.parallel_for(sycl::nd_range<2>({size_t(rows), size_t(n_ht) * WG}, {1, size_t(WG)}), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint32_t t = uint32_t(it.get_group(0)), ht = uint32_t(it.get_group(1)), lid = uint32_t(it.get_local_id(1));
            auto sg = it.get_sub_group(); const uint32_t s = uint32_t(sg.get_group_linear_id()); auto grp = it.get_group();
            const int32_t klo = lo[t], khi = hi[t];
            const sycl::half* qtile = a16 + (uint64_t(t) * NH + uint64_t(ht) * HT) * DQ;
            const int32_t* srow = sel ? sel + uint64_t(t) * S : nullptr;      // (null: the keys are contiguous, key c = keys row c)
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
                    if (c >= klo && c <= khi) { const int32_t k = srow ? srow[c] : c; if (k >= 0) v = reinterpret_cast<const h8*>(keys + uint64_t(k) * DQ)[d8]; }
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
                    const bool ok = c >= klo && c <= khi && (!srow || srow[c] >= 0);
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

}  // namespace ie
