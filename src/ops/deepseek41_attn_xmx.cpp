// src/ops/deepseek41_attn_xmx.cpp -- Phase 34 (docs/deepseek41/74): XMX gathered flash-decoding for the V4.1
// decode attention. See the header for the shape of the three launches. The tile arithmetic (S^T = Kblk . Q^T,
// P = exp(S - m), O = P . Vblk with K == V) is the prefill kernel's (deepseek4_attn_xmx.cpp) block body, run ONCE
// per work-group instead of looped over every block: the loop was a serial ~10 us per block at decode (probe,
// docs/74), so the blocks run side by side and a combine folds them.
//
// EVERY kernel-launching helper here is prefixed ds41dx_ (the SYCL kernel-name collision trap).
#include "ie/deepseek41_attn_xmx.hpp"
#include "ie/kernel_profiler.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace ie {
namespace ds41dx {

namespace mat  = sycl::ext::oneapi::experimental::matrix;
namespace imat = sycl::ext::intel::experimental::matrix;

constexpr int   SG  = 16;
constexpr int   NSG = 16;          // sub-groups per work-group
constexpr int   WG  = SG * NSG;    // 256 lanes
constexpr int   HT  = 32;          // heads per tile
constexpr int   KB  = 64;          // keys per block
constexpr int   D   = 512;         // head_dim
constexpr int   TM  = 8, TN = 16, TK = 16;
constexpr int   DC  = D / NSG;     // 32 output columns per sub-group
constexpr int   NC  = DC / TN;     // 2 column tiles per sub-group
constexpr int   MR  = HT / TM;     // 4 row tiles (heads)
constexpr float kNegInf    = -std::numeric_limits<float>::infinity();
constexpr float kMaskedCut = -1e30f;

constexpr uint64_t kSlmBytes = uint64_t(D) * HT * 2 + uint64_t(KB) * HT * 4 + uint64_t(HT) * KB * 2
                             + uint64_t(KB) * 4 + uint64_t(WG) * 4 + uint64_t(HT) * 4 * 3;

struct Ws {
    sycl::half* kgat = nullptr; uint64_t kgat_cap = 0;   // [T][NCOLP][D]
    float*      mgat = nullptr; uint64_t mgat_cap = 0;   // [T][NCOLP]
    float*      po   = nullptr; uint64_t po_cap   = 0;   // [T][n_ht][NB][HT][D]
    float*      pml  = nullptr; uint64_t pml_cap  = 0;   // [T][n_ht][NB][2][HT]
};

Ws& ds41dx_ws_for(sycl::queue& q) {
    static std::deque<std::pair<sycl::queue, Ws>> all;
    static std::mutex m;
    std::lock_guard<std::mutex> lk(m);
    for (auto& p : all) if (p.first == q) return p.second;
    all.emplace_back(q, Ws{});
    return all.back().second;
}

template <class V>
static void ds41dx_grow(sycl::queue& q, V*& p, uint64_t& cap, uint64_t need, const char* what) {
    if (need <= cap) return;
    if (p) { q.wait(); sycl::free(p, q); p = nullptr; cap = 0; }
    p = sycl::malloc_device<V>(size_t(need), q);
    if (!p) throw std::runtime_error(std::string("ds41_attention_decode_xmx: could not allocate ") + what);
    cap = need;
}

inline sycl::half ds41dx_sat16(float v) {
    v = sycl::fmin(sycl::fmax(v, -65504.0f), 65504.0f);
    return sycl::half(v);
}

}  // namespace ds41dx

bool ds41_attention_decode_xmx_eligible(const sycl::device& dev, uint32_t T, uint32_t n_heads, uint32_t head_dim) {
    static const bool on = [] { const char* e = std::getenv("IE_DS41_ATTN_XMX"); return !(e && *e && std::string(e) == "0"); }();
    if (!on) return false;
    if (T == 0 || T > 8 || head_dim != uint32_t(ds41dx::D) || n_heads == 0 || n_heads % ds41dx::HT) return false;
    static const bool fits = [&dev] {
        const uint64_t slm = dev.get_info<sycl::info::device::local_mem_size>();
        const size_t   wg  = dev.get_info<sycl::info::device::max_work_group_size>();
        const bool ok = slm >= ds41dx::kSlmBytes && wg >= size_t(ds41dx::WG);
        if (!ok) std::fprintf(stderr, "[ds41-attn-dx] OFF: SLM %llu B (need %llu), max WG %zu (need %d)\n",
                              (unsigned long long)slm, (unsigned long long)ds41dx::kSlmBytes, wg, ds41dx::WG);
        return ok;
    }();
    return fits;
}

sycl::event ds41_attention_decode_xmx(sycl::queue& q,
                                      const float* q_in, const Ds4KvSegs& kv,
                                      const float* mask, const float* sinks,
                                      float* y,
                                      uint32_t T, uint32_t n_heads, uint32_t head_dim,
                                      float scaling,
                                      const std::vector<sycl::event>& deps) {
    using namespace ds41dx;
    if (head_dim != uint32_t(D) || n_heads % HT)
        throw std::invalid_argument("ds41_attention_decode_xmx: head_dim must be 512 and n_heads a multiple of 32");
    if (kv.n_b && !kv.b_picks)
        throw std::invalid_argument("ds41_attention_decode_xmx: a b segment needs the picks");
    const uint32_t n_a   = kv.n_a, n_kv = kv.n_a + kv.n_b;
    const uint32_t npk   = kv.n_b ? kv.n_picks : 0u;
    const uint32_t NCOL  = n_a + npk;                       // the gathered axis
    const uint32_t NB    = (NCOL + KB - 1) / KB;
    const uint32_t NCOLP = NB * KB;
    const uint32_t n_ht  = n_heads / HT;

    Ws& w = ds41dx_ws_for(q);
    ds41dx_grow(q, w.kgat, w.kgat_cap, uint64_t(T) * NCOLP * D, "the gathered K");
    ds41dx_grow(q, w.mgat, w.mgat_cap, uint64_t(T) * NCOLP, "the gathered mask");
    ds41dx_grow(q, w.po,   w.po_cap,   uint64_t(T) * n_ht * NB * HT * D, "the partial O");
    ds41dx_grow(q, w.pml,  w.pml_cap,  uint64_t(T) * n_ht * NB * 2 * HT, "the partial m/l");
    sycl::half* kgat = w.kgat; float* mgat = w.mgat; float* po = w.po; float* pml = w.pml;

    // ---- 1. GATHER: column c of row t is window slot c (c < n_a; the ring rule of Ds4KvSegs::a_new) or the
    //      pick c - n_a (ascending; a negative pick is a masked zero row), then zero masked pad to NCOLP ----
    const float* ka = kv.a; const float* ka_new = kv.a_new; const uint32_t a_base = kv.a_new_base, a_n = kv.a_new_n;
    const bool a_nocausal = kv.a_new_nocausal;
    const float* kb = static_cast<const float*>(kv.b); const bool kb_f16 = kv.b_f16;
    const sycl::half* kb16 = static_cast<const sycl::half*>(kv.b);
    const int32_t* picks = kv.b_picks; const uint32_t pitch = kv.n_picks;
    auto eg = ie::ps(q, "ds41_attn_dx_gather", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<2>({size_t(T) * NCOLP, size_t(D)}, {1, size_t(D)}), [=](sycl::nd_item<2> it) {
            const uint32_t tc = uint32_t(it.get_group(0)), d = uint32_t(it.get_local_id(1));
            const uint32_t t = tc / NCOLP, c = tc % NCOLP;
            const float* src = nullptr; const sycl::half* src16 = nullptr; float mv = kNegInf;
            if (c < n_a) {
                src = ka + size_t(c) * D;
                if (ka_new) { const uint32_t j = (c + n_a - a_base) % n_a; if (j < a_n && (j <= t || a_nocausal)) src = ka_new + size_t(j) * D; }
                mv = mask[size_t(t) * n_kv + c];
            } else if (c < NCOL) {
                const int32_t p = picks[size_t(t) * pitch + (c - n_a)];
                if (p >= 0 && n_a + uint32_t(p) < n_kv) {
                    if (kb_f16) src16 = kb16 + size_t(p) * D; else src = kb + size_t(p) * D;
                    mv = mask[size_t(t) * n_kv + n_a + uint32_t(p)];
                }
            }
            sycl::half v = sycl::half(0.f);
            if (src) v = ds41dx_sat16(src[d]); else if (src16) v = src16[d];
            kgat[size_t(tc) * D + d] = v;
            if (d == 0) mgat[tc] = mv;
        });
    });

    // ---- 2. PARTIAL: work-group = (row t, 64-key block blk) x 32-head tile ht. The prefill kernel's block body
    //      with m starting at -inf (so alpha = 0, O = P . V), writing (m_b, l_b, O_b) instead of running on ----
    auto ep = ie::ps(q, "ds41_attn_dx_part", [&](sycl::handler& h) {
        h.depends_on(eg);
        sycl::local_accessor<sycl::half, 1> Qt(sycl::range<1>(size_t(D) * HT), h);
        sycl::local_accessor<float, 1>      St(sycl::range<1>(size_t(KB) * HT), h);
        sycl::local_accessor<sycl::half, 1> P (sycl::range<1>(size_t(HT) * KB), h);
        sycl::local_accessor<float, 1>      mk(sycl::range<1>(KB), h);
        sycl::local_accessor<float, 1>      red(sycl::range<1>(WG), h);
        sycl::local_accessor<float, 1>      mb(sycl::range<1>(HT), h);
        sycl::local_accessor<float, 1>      lb(sycl::range<1>(HT), h);
        sycl::local_accessor<int, 1>        anyl(sycl::range<1>(1), h);
        h.parallel_for(sycl::nd_range<2>({size_t(T) * NB, size_t(n_ht) * WG}, {1, WG}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint32_t tb  = uint32_t(it.get_group(0)), t = tb / NB, blk = tb % NB;
            const uint32_t ht  = uint32_t(it.get_group(1));
            const uint32_t lid = uint32_t(it.get_local_id(1));
            auto sg = it.get_sub_group();
            const uint32_t s = uint32_t(sg.get_group_linear_id());
            auto grp = it.get_group();
            const uint64_t slot = (uint64_t(t) * n_ht + ht) * NB + blk;

            // the block's mask; a fully masked block writes m = -inf, l = 0 and no O (the combine skips it)
            if (lid < KB) mk[lid] = mgat[size_t(t) * NCOLP + blk * KB + lid];
            sycl::group_barrier(grp);
            const bool allowed = sycl::any_of_group(grp, lid < KB && mk[lid] > kMaskedCut);
            if (!allowed) {
                if (lid < HT) { pml[slot * 2 * HT + lid] = kNegInf; pml[slot * 2 * HT + HT + lid] = 0.f; }
                return;
            }
            {   // Q^T [D][HT] staged in fp16, saturating
                const uint64_t qoff = (uint64_t(t) * n_heads + uint64_t(ht) * HT) * D;
                for (uint32_t i = lid; i < uint32_t(HT) * D; i += WG) {
                    const uint32_t hh = i / D, k = i % D;
                    Qt[k * HT + hh] = ds41dx_sat16(q_in[qoff + i]);
                }
            }
            sycl::group_barrier(grp);
            const sycl::half* kblk = kgat + (size_t(t) * NCOLP + size_t(blk) * KB) * D;

            // S^T [KB][HT] = Kblk . Q^T: 8 row tiles x 2 head column tiles, one per sub-group
            {
                const uint32_t rt = s >> 1, ct = s & 1;
                mat::joint_matrix<sycl::sub_group, float, mat::use::accumulator, TM, TN> sacc;
                mat::joint_matrix_fill(sg, sacc, 0.0f);
                const sycl::half* kbase = kblk + uint64_t(rt * TM) * D;
                #pragma unroll 4
                for (int kk = 0; kk < D; kk += TK) {
                    mat::joint_matrix<sycl::sub_group, sycl::half, mat::use::a, TM, TK, mat::layout::row_major> a_tile;
                    mat::joint_matrix_load(sg, a_tile,
                        sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(kbase + kk), D);
                    mat::joint_matrix<sycl::sub_group, sycl::half, mat::use::b, TK, TN, mat::layout::row_major> b_tile;
                    mat::joint_matrix_load(sg, b_tile, Qt.template get_multi_ptr<sycl::access::decorated::no>() + kk * HT + ct * TN, HT);
                    mat::joint_matrix_mad(sg, sacc, a_tile, b_tile, sacc);
                }
                imat::joint_matrix_apply(sg, sacc, [&](float& v, size_t r, size_t c) { St[(rt * TM + r) * HT + ct * TN + c] = v; });
            }
            sycl::group_barrier(grp);

            // the block softmax per head: 8 lanes per head, 8 keys each; m_b = the block max, l_b = its sum
            const uint32_t hh = lid & (HT - 1);
            const uint32_t jj = lid >> 5;
            float sc[8]; float bm = kNegInf;
            #pragma unroll
            for (int r = 0; r < 8; ++r) {
                const uint32_t row = jj * 8 + r;
                const float mv = mk[row];
                sc[r] = (mv > kMaskedCut) ? St[row * HT + hh] * scaling + mv : kNegInf;
                bm = sycl::fmax(bm, sc[r]);
            }
            red[lid] = bm;
            sycl::group_barrier(grp);
            if (lid < HT) { float m8 = kNegInf; for (int j = 0; j < 8; ++j) m8 = sycl::fmax(m8, red[j * HT + lid]); mb[lid] = m8; }
            sycl::group_barrier(grp);
            {
                const float m_n = mb[hh]; float ps = 0.f;
                #pragma unroll
                for (int r = 0; r < 8; ++r) {
                    const uint32_t row = jj * 8 + r;
                    const float e = (sc[r] > kNegInf) ? sycl::exp(sc[r] - m_n) : 0.f;
                    P[hh * KB + row] = sycl::half(e);
                    ps += e;
                }
                red[lid] = ps;
            }
            sycl::group_barrier(grp);
            if (lid < HT) { float l8 = 0.f; for (int j = 0; j < 8; ++j) l8 += red[j * HT + lid]; lb[lid] = l8;
                            pml[slot * 2 * HT + lid] = mb[lid]; pml[slot * 2 * HT + HT + lid] = l8; }

            // O_b [HT][D] = P [HT][KB] . Vblk [KB][D]: this sub-group's 32 columns
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
                        sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(kblk + uint64_t(kk) * D + s * DC + nc * TN), D);
                #pragma unroll
                for (int mr = 0; mr < MR; ++mr) {
                    mat::joint_matrix<sycl::sub_group, sycl::half, mat::use::a, TM, TK, mat::layout::row_major> a_tile;
                    mat::joint_matrix_load(sg, a_tile, P.template get_multi_ptr<sycl::access::decorated::no>() + (mr * TM) * KB + kk, KB);
                    #pragma unroll
                    for (int nc = 0; nc < NC; ++nc) mat::joint_matrix_mad(sg, acc[mr][nc], a_tile, b_tile[nc], acc[mr][nc]);
                }
            }
            float* obase = po + slot * uint64_t(HT) * D + s * DC;
            #pragma unroll
            for (int mr = 0; mr < MR; ++mr)
                #pragma unroll
                for (int nc = 0; nc < NC; ++nc)
                    imat::joint_matrix_apply(sg, acc[mr][nc], [&](float& v, size_t r, size_t c) { obase[uint64_t(mr * TM + r) * D + nc * TN + c] = v; });
        });
    });

    // ---- 3. COMBINE per (row, tile): blocks folded in index order, then the sinks and 1/l as the fp32 kernel ----
    return ie::ps(q, "ds41_attn_dx_combine", [&](sycl::handler& h) {
        h.depends_on(ep);
        h.parallel_for(sycl::nd_range<2>({size_t(T), size_t(n_ht) * WG}, {1, WG}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint32_t t = uint32_t(it.get_group(0)), ht = uint32_t(it.get_group(1)), lid = uint32_t(it.get_local_id(1));
            const uint32_t hh = lid >> 3, d0 = (lid & 7) * (D / 8);          // 8 lanes per head, 64 dims each
            const uint64_t base = (uint64_t(t) * n_ht + ht) * NB;
            float gm = kNegInf;
            for (uint32_t b = 0; b < NB; ++b) gm = sycl::fmax(gm, pml[(base + b) * 2 * HT + hh]);
            float gl = 0.f; float o[D / 8];
            for (int i = 0; i < D / 8; ++i) o[i] = 0.f;
            for (uint32_t b = 0; b < NB; ++b) {
                const float m_b = pml[(base + b) * 2 * HT + hh];
                if (m_b == kNegInf) continue;
                const float wgt = sycl::exp(m_b - gm);
                gl += pml[(base + b) * 2 * HT + HT + hh] * wgt;
                const float* ob = po + (base + b) * uint64_t(HT) * D + uint64_t(hh) * D + d0;
                for (int i = 0; i < D / 8; ++i) o[i] += ob[i] * wgt;
            }
            float corr = 1.f, ll = gl;
            if (sinks) {
                const float sv  = sinks[ht * HT + hh];
                const float m_f = sycl::fmax(gm, sv);
                corr = (gm == kNegInf) ? 0.f : sycl::exp(gm - m_f);
                ll   = gl * corr + sycl::exp(sv - m_f);
            }
            const float inv = (ll > 0.f) ? (1.f / ll) : 0.f;
            const float sc = corr * inv;
            float* yb = y + (uint64_t(t) * n_heads + uint64_t(ht) * HT + hh) * D + d0;
            for (int i = 0; i < D / 8; ++i) yb[i] = o[i] * sc;
        });
    });
}

sycl::event ds41_attention_cont_xmx(sycl::queue& q, const float* q_in,
                                    const float* wkeys, uint32_t nkw, uint32_t nprev, uint32_t win, const float* mwin, uint32_t row0,
                                    const float* ckv, uint32_t nc, const int32_t* picks, uint32_t pitch,
                                    const float* sinks, float* y,
                                    uint32_t T, uint32_t n_heads, uint32_t head_dim, float scaling) {
    using namespace ds41dx;
    if (head_dim != uint32_t(D)) throw std::invalid_argument("ds41_attention_cont_xmx: head_dim must be 512");
    const uint32_t npk   = picks ? pitch : 0u;
    const uint32_t NCOL  = win + npk;
    const uint32_t NCOLP = (NCOL + KB - 1) / KB * KB;
    Ws& w = ds41dx_ws_for(q);
    ds41dx_grow(q, w.kgat, w.kgat_cap, uint64_t(T) * NCOLP * D, "the gathered K");
    ds41dx_grow(q, w.mgat, w.mgat_cap, uint64_t(T) * NCOLP, "the gathered mask");
    sycl::half* kgat = w.kgat; float* mgat = w.mgat;
    // column c < win: window key k0 + c (masked past the segment or where mwin closes it); c < NCOL: pick c - win
    // (a latent, bias 0); the pad to the block masked. One work-group per (row, column), D lanes: a key row's copy.
    auto eg = ie::ps(q, "ds41_attn_cont_gather", [&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<2>({size_t(T) * NCOLP, size_t(D)}, {1, size_t(D)}), [=](sycl::nd_item<2> it) {
            const uint32_t tc = uint32_t(it.get_group(0)), d = uint32_t(it.get_local_id(1));
            const uint32_t t = tc / NCOLP, c = tc % NCOLP, r = row0 + t;
            const float* src = nullptr; float mv = kNegInf;
            if (c < win) {
                const int64_t k = int64_t(r) + int64_t(nprev) - int64_t(win - 1) + int64_t(c);
                if (k >= 0 && k < int64_t(nkw)) {
                    const float m = mwin[size_t(r) * nkw + size_t(k)];
                    if (m > kMaskedCut) { src = wkeys + size_t(k) * D; mv = m; }
                }
            } else if (c < NCOL) {
                const int32_t p = picks[size_t(t) * pitch + (c - win)];
                if (p >= 0 && uint32_t(p) < nc) { src = ckv + size_t(p) * D; mv = 0.f; }
            }
            kgat[size_t(tc) * D + d] = src ? ds41dx_sat16(src[d]) : sycl::half(0.f);
            if (d == 0) mgat[tc] = mv;
        });
    });
    Ds4KvSegs segs; segs.a = nullptr; segs.n_a = NCOLP; segs.b = nullptr; segs.n_b = 0; segs.b_f16 = false;
    return ds4_attention_xmx_segs(q, q_in, segs, mgat, sinks, y, T, n_heads, head_dim, scaling, /*kv_prepared=*/false, {eg}, kgat);
}

}  // namespace ie
