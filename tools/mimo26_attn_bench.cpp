// tools/mimo26_attn_bench.cpp -- ie-mimo26-attn-bench: MiMo-V2.6's attention kernels timed in isolation on one card
// (P4, docs/mimo26/00_PORT_PLAN.md): mimo26_attention_decode at T = 1 and mimo26_attention_prefill_xmx at T = 2048, the
// full layers' geometry (GQA 64/4, K 192 / V 128 in 192-wide V rows, linear cache) and the SWA layers' (GQA 64/8, window
// 128, sinks, a 2176-slot ring), at several context positions. Random fp16 data; the median of --iters event-timed calls.
//   ie-mimo26-attn-bench [--iters N] [--ctx 4096,16384,32768] [--ablate]
// --ablate: the full layers' prefill compute at each --ctx with parts of the kernel skipped (px_ablate below).
#include "ie/mimo26_ops.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace {

// ---- ablation harness: a copy of mimo26_attention_prefill_xmx's compute kernel (no append) whose AB bits skip parts:
//      1 = the V staging, 2 = the S^T MMA, 4 = the online softmax, 8 = the P.V MMA; and variants: 16 = V staged with
//      16-byte loads, 32 = V tiles straight from global (no staging), 64 = K staged in SLM with 16-byte loads. Timing only.
namespace benchpx {
namespace mat  = sycl::ext::oneapi::experimental::matrix;
namespace imat = sycl::ext::intel::experimental::matrix;
constexpr uint32_t SG = 16, NSG = 16, WG = SG * NSG, HT = 64, KB = 64, DK = 192, DV = 128, TM = 8, TN = 16, TK = 16;
constexpr uint32_t KPL = KB * HT / WG;
constexpr float kNegInf = -std::numeric_limits<float>::infinity();
}  // namespace benchpx
template <int AB>
sycl::event px_ablate(sycl::queue& q, const sycl::half* Q, const sycl::half* K, const sycl::half* V,
                                         sycl::half* k_cache, sycl::half* v_cache, sycl::half* y,
                                         uint32_t T, uint32_t pos0, uint32_t n_q, uint32_t n_kv, uint32_t max_ctx,
                                         uint32_t window, const float* sinks, const std::vector<sycl::event>& deps,
                                         uint32_t v_stride, uint32_t ring) {
    using namespace benchpx;
    if (!v_stride) v_stride = DV;
    const uint32_t gqa = n_q / n_kv;
    if (!gqa || n_q % n_kv || HT % gqa) throw std::invalid_argument("mimo26_attention_prefill_xmx: n_q / n_kv must divide 64");
    if (ring % TM) throw std::invalid_argument("mimo26_attention_prefill_xmx: the ring must be a multiple of 8 slots");
    const uint32_t slots = ring ? ring : max_ctx;
    const uint32_t n_tt = HT / gqa, n_tb = (T + n_tt - 1) / n_tt;   // tokens per work-group, token blocks
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<sycl::half, 1> Qt(sycl::range<1>((AB & 128) ? 1 : size_t(DK) * HT), h);   // Q^T [DK][HT]
        sycl::local_accessor<float, 1>      St(sycl::range<1>(size_t(KB) * HT), h);   // S^T [KB][HT]
        sycl::local_accessor<sycl::half, 1> P (sycl::range<1>(size_t(HT) * KB), h);   // P [HT][KB]
        sycl::local_accessor<sycl::half, 1> Vs(sycl::range<1>((AB & 32) ? 1 : size_t(KB) * DV), h);   // the block's V [KB][DV]
        sycl::local_accessor<sycl::half, 1> Ks(sycl::range<1>((AB & 64) ? size_t(KB) * DK : 1), h);   // AB & 64: the block's K
        sycl::local_accessor<float, 1>      red(sycl::range<1>(WG), h);
        sycl::local_accessor<float, 1>      mrun(sycl::range<1>(HT), h), lrun(sycl::range<1>(HT), h);
        sycl::local_accessor<float, 1>      alpha(sycl::range<1>(HT), h), mnew(sycl::range<1>(HT), h);
        const float scale = 1.0f / sycl::sqrt(float(DK));
        const uint32_t vs = v_stride, R = ring;
        h.parallel_for(sycl::nd_range<2>({size_t(n_tb), size_t(n_kv) * WG}, {1, WG}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint32_t tb  = n_tb - 1 - uint32_t(it.get_group(0));   // the longest key ranges start first
            const uint32_t kvh = uint32_t(it.get_group(1));
            const uint32_t lid = uint32_t(it.get_local_id(1));
            auto sg = it.get_sub_group();
            auto grp = it.get_group();
            const uint32_t s  = uint32_t(sg.get_group_linear_id());
            const uint32_t t0 = tb * n_tt, t_last = sycl::min(t0 + n_tt, T) - 1;
            // row = tt * gqa + g: token t0 + tt, q head kvh * gqa + g
            if constexpr (!(AB & 128)) for (uint32_t i = lid; i < HT * DK; i += WG) {
                const uint32_t row = i / DK, k = i % DK, t = t0 + row / gqa;
                Qt[k * HT + row] = t < T ? Q[(size_t(t) * n_q + kvh * gqa + row % gqa) * DK + k] : sycl::half(0.f);
            }
            if (lid < HT) { mrun[lid] = kNegInf; lrun[lid] = 0.f; }
            // O [HT][DV]: sub-group s owns rows [32 (s >> 3), +32) (4 row tiles) x columns [16 (s & 7), +16)
            const uint32_t orow0 = (s >> 3) * 32, ocol = (s & 7) * TN;
            mat::joint_matrix<sycl::sub_group, float, mat::use::accumulator, TM, TN> acc[4];
            #pragma unroll
            for (int mr = 0; mr < 4; ++mr) mat::joint_matrix_fill(sg, acc[mr], 0.0f);
            // this lane's softmax row and keys: row hh, keys [KPL jj, KPL jj + KPL) of each block
            const uint32_t hh = lid & (HT - 1), jj = lid / HT;
            const uint32_t t_row = t0 + hh / gqa, p_row = pos0 + t_row;
            const bool row_live = t_row < T;
            const uint32_t lo_row = (window && p_row + 1 > window) ? p_row + 1 - window : 0u;   // first allowed key
            const uint32_t kend = pos0 + t_last + 1;
            const uint32_t p_first = pos0 + t0;
            const uint32_t kbeg = ((window && p_first + 1 > window) ? p_first + 1 - window : 0u) / KB * KB;
            sycl::group_barrier(grp);

            for (uint32_t n0 = kbeg; n0 < kend; n0 += KB) {
                // the block's V rows (past kend: zero -- a stale or NaN row times P = 0 must not reach O)
                if constexpr (!(AB & 1) && !(AB & 16) && !(AB & 32)) for (uint32_t i = lid; i < KB * DV; i += WG) {
                    const uint32_t r = i / DV, c = i % DV, key = n0 + r;
                    Vs[i] = key < kend ? v_cache[(size_t(kvh) * slots + (R ? key % R : key)) * vs + c] : sycl::half(0.f);
                }
                if constexpr ((AB & 16) && !(AB & 32)) for (uint32_t i = lid; i < KB * (DV / 8); i += WG) {   // 16-byte V staging
                    const uint32_t r = i / (DV / 8), c8 = i % (DV / 8), key = n0 + r;
                    sycl::vec<sycl::half, 8> v(sycl::half(0.f));
                    if (key < kend) v.load(0, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(v_cache + (size_t(kvh) * slots + (R ? key % R : key)) * vs + c8 * 8));
                    v.store(0, Vs.template get_multi_ptr<sycl::access::decorated::no>() + r * DV + c8 * 8);
                }
                if constexpr (AB & 64) for (uint32_t i = lid; i < KB * (DK / 8); i += WG) {   // 16-byte K staging
                    const uint32_t r = i / (DK / 8), c8 = i % (DK / 8), key = n0 + r;
                    sycl::vec<sycl::half, 8> v(sycl::half(0.f));
                    if (key < kend) v.load(0, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(k_cache + (size_t(kvh) * slots + (R ? key % R : key)) * DK + c8 * 8));
                    v.store(0, Ks.template get_multi_ptr<sycl::access::decorated::no>() + r * DK + c8 * 8);
                }
                if constexpr (AB & 64) sycl::group_barrier(grp);
                // S^T [KB][HT] = Kblk [KB][DK] . Q^T [DK][HT]: 8 key-row tiles x 4 row-column tiles; sub-group s takes key
                // tile s >> 1 and row-column tiles 2 (s & 1) + {0, 1}. K straight from the cache (rows past kend are masked;
                // in a ring an 8-row tile starts at a multiple of 8 and R is one, so it never wraps)
                {
                    const uint32_t rt = s >> 1, c0 = (s & 1) * 2;
                    mat::joint_matrix<sycl::sub_group, float, mat::use::accumulator, TM, TN> sacc[2];
                    mat::joint_matrix_fill(sg, sacc[0], 0.0f);
                    mat::joint_matrix_fill(sg, sacc[1], 0.0f);
                    if (!(AB & 2) && n0 + rt * TM < kend) {
                        const uint32_t key = n0 + rt * TM;
                        const sycl::half* kbase = k_cache + (size_t(kvh) * slots + (R ? key % R : key)) * DK;
                        #pragma unroll 4
                        for (uint32_t kk = 0; kk < DK; kk += TK) {
                            mat::joint_matrix<sycl::sub_group, sycl::half, mat::use::a, TM, TK, mat::layout::row_major> a_tile;
                            if constexpr (AB & 64) mat::joint_matrix_load(sg, a_tile, Ks.template get_multi_ptr<sycl::access::decorated::no>() + (rt * TM) * DK + kk, DK);
                            else mat::joint_matrix_load(sg, a_tile,
                                sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(kbase + kk), DK);
                            #pragma unroll
                            for (uint32_t j = 0; j < 2; ++j) {
                                if constexpr (AB & 128) {   // Q^T tile straight from global Q (gqa 16: column tile c = token t0 + c's 16 heads)
                                    mat::joint_matrix<sycl::sub_group, sycl::half, mat::use::b, TK, TN, mat::layout::col_major> bq;
                                    const sycl::half* qb = Q + (size_t(t0 + c0 + j) * n_q + kvh * gqa) * DK + kk;
                                    mat::joint_matrix_load(sg, bq, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(qb), DK);
                                    mat::joint_matrix_mad(sg, sacc[j], a_tile, bq, sacc[j]);
                                } else {
                                mat::joint_matrix<sycl::sub_group, sycl::half, mat::use::b, TK, TN, mat::layout::row_major> b_tile;
                                mat::joint_matrix_load(sg, b_tile, Qt.template get_multi_ptr<sycl::access::decorated::no>() + kk * HT + (c0 + j) * TN, HT);
                                mat::joint_matrix_mad(sg, sacc[j], a_tile, b_tile, sacc[j]);
                                }
                            }
                        }
                    }
                    #pragma unroll
                    for (uint32_t j = 0; j < 2; ++j)
                        imat::joint_matrix_apply(sg, sacc[j], [&](float& v, size_t r, size_t c) { St[(rt * TM + r) * HT + (c0 + j) * TN + c] = v; });
                }
                sycl::group_barrier(grp);
                // online softmax: 4 lanes per row, KPL keys each
                if constexpr (!(AB & 4)) {
                float sc[KPL]; float bm = kNegInf;
                #pragma unroll
                for (uint32_t r = 0; r < KPL; ++r) {
                    const uint32_t kl = jj * KPL + r, key = n0 + kl;
                    const bool ok = row_live && key <= p_row && key >= lo_row;
                    sc[r] = ok ? St[kl * HT + hh] * scale : kNegInf;
                    bm = sycl::fmax(bm, sc[r]);
                }
                red[lid] = bm;
                sycl::group_barrier(grp);
                if (lid < HT) {
                    float m4 = kNegInf;
                    #pragma unroll
                    for (uint32_t j = 0; j < WG / HT; ++j) m4 = sycl::fmax(m4, red[j * HT + lid]);
                    const float m_old = mrun[lid], m_n = sycl::fmax(m_old, m4);
                    mnew[lid] = m_n;
                    alpha[lid] = (m_old == kNegInf) ? 0.f : sycl::native::exp(m_old - m_n);
                }
                sycl::group_barrier(grp);
                {
                    const float m_n = mnew[hh]; float ps = 0.f;
                    #pragma unroll
                    for (uint32_t r = 0; r < KPL; ++r) {
                        const float e = sc[r] > kNegInf ? sycl::native::exp(sc[r] - m_n) : 0.f;
                        P[hh * KB + jj * KPL + r] = sycl::half(e);
                        ps += e;
                    }
                    red[lid] = ps;
                }
                sycl::group_barrier(grp);
                if (lid < HT) {
                    float l4 = 0.f;
                    #pragma unroll
                    for (uint32_t j = 0; j < WG / HT; ++j) l4 += red[j * HT + lid];
                    lrun[lid] = lrun[lid] * alpha[lid] + l4;
                    mrun[lid] = mnew[lid];
                }
                }
                // O = O * alpha + P [HT][KB] . Vblk [KB][DV]
                #pragma unroll
                for (int mr = 0; mr < 4; ++mr)
                    imat::joint_matrix_apply(sg, acc[mr], [&](float& v, size_t r, size_t) { v *= alpha[orow0 + mr * TM + r]; });
                if constexpr (!(AB & 8))
                #pragma unroll
                for (uint32_t kk = 0; kk < KB; kk += TK) {
                    mat::joint_matrix<sycl::sub_group, sycl::half, mat::use::b, TK, TN, mat::layout::row_major> b_tile;
                    if constexpr (AB & 32) mat::joint_matrix_load(sg, b_tile,
                        sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(v_cache + (size_t(kvh) * slots + n0 + kk) * vs + ocol), vs);
                    else mat::joint_matrix_load(sg, b_tile, Vs.template get_multi_ptr<sycl::access::decorated::no>() + kk * DV + ocol, DV);
                    #pragma unroll
                    for (int mr = 0; mr < 4; ++mr) {
                        mat::joint_matrix<sycl::sub_group, sycl::half, mat::use::a, TM, TK, mat::layout::row_major> a_tile;
                        mat::joint_matrix_load(sg, a_tile, P.template get_multi_ptr<sycl::access::decorated::no>() + (orow0 + mr * TM) * KB + kk, KB);
                        mat::joint_matrix_mad(sg, acc[mr], a_tile, b_tile, acc[mr]);
                    }
                }
                sycl::group_barrier(grp);
            }
            // the sink (a virtual key of logit sinks[head] and no value) and 1/l
            if (lid < HT) {
                const float m = mrun[lid], l = lrun[lid];
                float corr = 1.f, ll = l;
                if (sinks) {
                    const float sv = sinks[kvh * gqa + lid % gqa], m_f = sycl::fmax(m, sv);
                    corr = (m == kNegInf) ? 0.f : sycl::native::exp(m - m_f);
                    ll = l * corr + sycl::native::exp(sv - m_f);
                }
                alpha[lid] = ll > 0.f ? corr / ll : 0.f;
            }
            sycl::group_barrier(grp);
            #pragma unroll
            for (int mr = 0; mr < 4; ++mr)
                imat::joint_matrix_apply(sg, acc[mr], [&](float& v, size_t r, size_t c) {
                    const uint32_t row = orow0 + mr * TM + uint32_t(r), t = t0 + row / gqa;
                    if (t < T) y[(size_t(t) * n_q + kvh * gqa + row % gqa) * DV + ocol + c] = sycl::half(v * alpha[row]);
                });
        });
    });
}

}  // namespace

int main(int argc, char** argv) {
    uint32_t iters = 20; bool ablate = false;
    std::vector<uint32_t> ctxs = {4096, 16384, 32768};
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--iters" && i + 1 < argc) iters = uint32_t(std::atoi(argv[++i]));
        else if (a == "--ablate") ablate = true;
        else if (a == "--ctx" && i + 1 < argc) { ctxs.clear(); std::stringstream ss(argv[++i]); std::string t; while (std::getline(ss, t, ',')) ctxs.push_back(uint32_t(std::atol(t.c_str()))); }
        else { std::fprintf(stderr, "usage: ie-mimo26-attn-bench [--iters N] [--ctx a,b,c]\n"); return 2; }
    }
    sycl::device dev; bool found = false;
    for (const auto& p : sycl::platform::get_platforms()) {
        if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        for (const auto& d : p.get_devices(sycl::info::device_type::gpu))
            if (!found && d.get_info<sycl::info::device::name>().find("Arc") != std::string::npos) { dev = d; found = true; }
    }
    if (!found) { std::fprintf(stderr, "no Arc GPU\n"); return 1; }
    sycl::queue q(sycl::context(dev), dev, sycl::property_list{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}});
    const uint32_t n_q = 64, hd = 192, hdv = 128, TP = 2048, max_ctx = *std::max_element(ctxs.begin(), ctxs.end()) + TP;
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto fill = [&](sycl::half* p, size_t n) {
        std::vector<sycl::half> h(n); for (auto& v : h) v = sycl::half(nd(rng));
        q.memcpy(p, h.data(), n * 2).wait();
    };
    auto ms_of = [](const sycl::event& e) {
        return double(e.get_profiling_info<sycl::info::event_profiling::command_end>() - e.get_profiling_info<sycl::info::event_profiling::command_start>()) * 1e-6;
    };
    if (!ablate) std::printf("%-6s %-8s %8s %12s %12s\n", "layer", "kernel", "keys", "median ms", "min ms");
    if (ablate) {
        const uint32_t n_kv = 4, vs = hd;
        sycl::half* Q = sycl::malloc_device<sycl::half>(size_t(TP) * n_q * hd, q);
        sycl::half* kc = sycl::malloc_device<sycl::half>((size_t(n_kv) * max_ctx + 64) * hd, q);
        sycl::half* vc = sycl::malloc_device<sycl::half>(size_t(n_kv) * max_ctx * vs, q);
        sycl::half* y = sycl::malloc_device<sycl::half>(size_t(TP) * n_q * hdv, q);
        fill(Q, size_t(TP) * n_q * hd); fill(kc, (size_t(n_kv) * max_ctx + 64) * hd); fill(vc, size_t(n_kv) * max_ctx * vs);
        std::printf("full-layer prefill compute (T %u), ms per call by skipped parts (skip: 1 V staging, 2 S^T MMA, 4 softmax, 8 P.V MMA; variants: 16 V staged 16 B/lane, 32 V tiles from global, 64 K staged 16 B/lane, 128 Q tiles from global)\n", TP);
        for (uint32_t ctx : ctxs) {
            const uint32_t pos0 = ctx >= TP ? ctx - TP : 0u;
            std::printf("  keys %6u:", ctx);
            auto run = [&](auto tag) {
                constexpr int AB = decltype(tag)::value;
                std::vector<double> t;
                for (uint32_t it = 0; it < std::max(3u, iters / 4) + 1; ++it) {
                    auto e = px_ablate<AB>(q, Q, nullptr, nullptr, kc, vc, y, TP, pos0, n_q, n_kv, max_ctx, 0, nullptr, {}, vs, 0);
                    e.wait(); if (it >= 1) t.push_back(ms_of(e));
                }
                std::sort(t.begin(), t.end());
                std::printf("  [%2d] %8.2f", AB, t[t.size() / 2]); std::fflush(stdout);
            };
            run(std::integral_constant<int, 0>{}); run(std::integral_constant<int, 32>{}); run(std::integral_constant<int, 128>{});
            run(std::integral_constant<int, 160>{}); run(std::integral_constant<int, 164>{}); run(std::integral_constant<int, 162>{});
            run(std::integral_constant<int, 168>{});
            std::printf("\n");
        }
        return 0;
    }
    for (int swa = 0; swa < 2; ++swa) {
        const uint32_t n_kv = swa ? 8u : 4u, vs = swa ? hdv : hd, window = swa ? 128u : 0u, ring = swa ? 128u + TP : 0u;
        const uint32_t slots = ring ? ring : max_ctx;
        sycl::half* Q = sycl::malloc_device<sycl::half>(size_t(TP) * n_q * hd, q);
        sycl::half* K = sycl::malloc_device<sycl::half>(size_t(TP) * n_kv * hd, q);
        sycl::half* V = sycl::malloc_device<sycl::half>(size_t(TP) * n_kv * vs, q);
        sycl::half* kc = sycl::malloc_device<sycl::half>((size_t(n_kv) * slots + 64) * hd, q);
        sycl::half* vc = sycl::malloc_device<sycl::half>(size_t(n_kv) * slots * vs, q);
        sycl::half* y = sycl::malloc_device<sycl::half>(size_t(TP) * n_q * hdv, q);
        float* part = sycl::malloc_device<float>(size_t(8) * n_q * ie::mimo26_decode_max_splits() * (hdv + 2), q);
        float* sinks = sycl::malloc_device<float>(n_q, q);
        if (!Q || !K || !V || !kc || !vc || !y || !part || !sinks) { std::fprintf(stderr, "allocation failed\n"); return 1; }
        fill(Q, size_t(TP) * n_q * hd); fill(K, size_t(TP) * n_kv * hd); fill(V, size_t(TP) * n_kv * vs);
        fill(kc, (size_t(n_kv) * slots + 64) * hd); fill(vc, size_t(n_kv) * slots * vs);
        { std::vector<float> s(n_q, 1.f); q.memcpy(sinks, s.data(), n_q * 4).wait(); }
        const float* sk = swa ? sinks : nullptr;
        for (uint32_t ctx : ctxs) {
            std::vector<double> td, tp;
            for (uint32_t it = 0; it < iters + 2; ++it) {
                // the decode step's kernels: append + partials + merge (the partial and merge events are what ps returns last)
                const auto t0 = q.submit([](sycl::handler& h) { h.single_task([] {}); });
                auto e = ie::mimo26_attention_decode(q, Q, K, V, kc, vc, y, part, 1, ctx - 1, n_q, n_kv, hd, hdv, max_ctx, window, sk, {}, ring, vs);
                e.wait();
                const double ms = double(e.get_profiling_info<sycl::info::event_profiling::command_end>() - t0.get_profiling_info<sycl::info::event_profiling::command_end>()) * 1e-6;
                if (it >= 2) td.push_back(ms);
            }
            for (uint32_t it = 0; it < std::max(3u, iters / 4) + 1; ++it) {
                const uint32_t pos0 = ctx >= TP ? ctx - TP : 0u;
                auto e = ie::mimo26_attention_prefill_xmx(q, Q, K, V, kc, vc, y, TP, pos0, n_q, n_kv, max_ctx, window, sk, {}, vs, ring);
                e.wait();
                if (it >= 1) tp.push_back(ms_of(e));
            }
            std::sort(td.begin(), td.end()); std::sort(tp.begin(), tp.end());
            std::printf("%-6s %-8s %8u %12.4f %12.4f\n", swa ? "swa" : "full", "decode", ctx, td[td.size() / 2], td.front());
            std::printf("%-6s %-8s %8u %12.3f %12.3f   (T %u rows ending at key %u)\n", swa ? "swa" : "full", "prefill", ctx, tp[tp.size() / 2], tp.front(), TP, ctx);
        }
        for (void* p : {(void*)Q, (void*)K, (void*)V, (void*)kc, (void*)vc, (void*)y, (void*)part, (void*)sinks}) sycl::free(p, q);
    }
    return 0;
}
