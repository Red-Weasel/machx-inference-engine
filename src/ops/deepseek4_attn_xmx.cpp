// src/ops/deepseek4_attn_xmx.cpp — block-sparse XMX flash attention for the
// deepseek4 PREFILL path (docs/deepseek4/72 Phase K).
//
// WHAT IT COMPUTES — the same function as ds4_attention in deepseek4_attn.cpp
// (transformers modeling_deepseek_v4 attention with MLA-absorbed K == V, an
// additive mask with disallowed sentinels, and the sink column):
//   y[t][h] = softmax_i( q[t][h] . kv[i] * scaling + mask[t][i] ) . kv[i]
// with the running max / sum / sink / 1/l order of the fp32 kernel.  Masked
// entries (mask <= kMaskedCut) contribute exactly nothing, as there.
//
// HOW — the three facts the kernel is built on:
//   * K and V are the same latent row, shared by every head of the card, so
//     one token's 32 heads are one 32-row tile with ONE mask row.
//   * The sliding window is 128 tokens and the rest of the context arrives as
//     compressed entries, so per token only a few hundred of the n_kv keys are
//     allowed: a 64-key block whose 64 mask entries are all disallowed is
//     skipped outright (exact, the whole tile shares the mask row).
//   * XMX wants fp16 operands: KV is converted once per layer into a per-queue
//     workspace. Q uses a pre-pass for small calls; large calls apply the same
//     saturating conversion while transposing Q into SLM. Scores, softmax and
//     the output accumulate in fp32.  So this is NOT bit-identical to the fp32
//     kernel — it is gated by a derived tolerance and by the PPL band.
// Work-group = one (token, 32-head tile) = 16 sub-groups of 16.  Q^T [512][32]
// is staged transposed in SLM once; per key block: S^T [64][32] = Kblk . Q^T
// (one 8x16 tile per sub-group, K = 512), online softmax per head across the
// block (8 lanes per head), P [32][64] fp16 in SLM, O [32][512] fp32 in
// registers (sub-group s owns columns [32 s, 32 s + 32): 8 tiles), rescaled
// by alpha per row and accumulated as P . Vblk with Vblk read straight from
// the fp16 KV rows (row-major B operand, no transpose).
//
// EVERY kernel-launching helper here is prefixed ds4ax_ (the SYCL kernel-name
// collision trap: same-named helpers in ie::(anonymous) across two TUs run
// the wrong device code).
#include "ie/deepseek4_attn.hpp"
#include "ie/kernel_profiler.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <deque>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace ie {
namespace ds4ax {

namespace mat  = sycl::ext::oneapi::experimental::matrix;
namespace imat = sycl::ext::intel::experimental::matrix;

constexpr int   SG  = 16;
constexpr int   NSG = 16;          // sub-groups per work-group
constexpr int   WG  = SG * NSG;    // 256 lanes
constexpr int   HT  = 32;          // heads per tile
constexpr int   KB  = 64;          // keys per block
constexpr int   D   = 512;         // the head_dim this kernel is built for
constexpr int   TM  = 8, TN = 16, TK = 16;
constexpr int   DC  = D / NSG;     // 32 output columns per sub-group
constexpr int   NC  = DC / TN;     // 2 column tiles per sub-group
constexpr int   MR  = HT / TM;     // 4 row tiles (heads)
constexpr float kNegInf    = -std::numeric_limits<float>::infinity();
constexpr float kMaskedCut = -1e30f;   // the fp32 kernel's sentinel cut

// SLM the kernel declares, for the dispatch check.
constexpr uint64_t kSlmBytes = uint64_t(D) * HT * 2      // Q^T fp16
                             + uint64_t(KB) * HT * 4     // S^T fp32
                             + uint64_t(HT) * KB * 2     // P fp16
                             + uint64_t(KB) * 4          // block mask
                             + uint64_t(WG) * 4          // per-lane partials
                             + uint64_t(HT) * 4 * 4;     // m, l, alpha, m_new

struct Ws {
    sycl::half* q16      = nullptr;   uint64_t q16_cap  = 0;   // elements
    sycl::half* kv16     = nullptr;   uint64_t kv16_cap = 0;   // elements, padded to KB rows
};

Ws& ds4ax_ws_for(sycl::queue& q) {
    static std::deque<std::pair<sycl::queue, Ws>> all;   // deque: references stay valid
    static std::mutex m;
    std::lock_guard<std::mutex> lk(m);
    for (auto& p : all) if (p.first == q) return p.second;
    all.emplace_back(q, Ws{});
    return all.back().second;
}

std::string ds4ax_reserve(sycl::queue& q, Ws& w, uint64_t q_elems, uint64_t kv_elems) {
    if (q_elems > w.q16_cap) {
        if (w.q16) { q.wait(); sycl::free(w.q16, q); w.q16 = nullptr; w.q16_cap = 0; }
        w.q16 = sycl::malloc_device<sycl::half>(size_t(q_elems), q);
        if (!w.q16) return "ds4_attention_xmx: could not allocate the fp16 Q workspace";
        w.q16_cap = q_elems;
    }
    if (kv_elems > w.kv16_cap) {
        if (w.kv16) { q.wait(); sycl::free(w.kv16, q); w.kv16 = nullptr; w.kv16_cap = 0; }
        w.kv16 = sycl::malloc_device<sycl::half>(size_t(kv_elems), q);
        if (!w.kv16) return "ds4_attention_xmx: could not allocate the fp16 KV workspace";
        w.kv16_cap = kv_elems;
    }
    return {};
}

}  // namespace ds4ax

bool ds4_attention_xmx_eligible(const sycl::device& dev, uint32_t T, uint32_t n_heads,
                                uint32_t head_dim) {
    // DEFAULT since the Phase K gate (2026-09-02): IE_DS4_ATTN_XMX=0 restores
    // the fp32 kernels for the prefill shapes.
    static const bool on = [] {
        const char* e = std::getenv("IE_DS4_ATTN_XMX");
        return !(e && *e && std::string(e) == "0");
    }();
    if (!on) return false;
    if (T <= 16 || head_dim != uint32_t(ds4ax::D) || n_heads == 0 || n_heads % ds4ax::HT) return false;
    static const bool fits = [&dev] {
        const uint64_t slm = dev.get_info<sycl::info::device::local_mem_size>();
        const size_t   wg  = dev.get_info<sycl::info::device::max_work_group_size>();
        const bool ok = slm >= ds4ax::kSlmBytes && wg >= size_t(ds4ax::WG);
        if (!ok)
            std::fprintf(stderr, "[ds4-attn-xmx] OFF: SLM %llu B (need %llu), max WG %zu (need %d)\n",
                         (unsigned long long)slm, (unsigned long long)ds4ax::kSlmBytes, wg, ds4ax::WG);
        return ok;
    }();
    return fits;
}

uint64_t ds4_attention_xmx_ws_bytes(uint32_t T, uint32_t n_heads, uint32_t n_kv_max) noexcept {
    using namespace ds4ax;
    const uint64_t n_kv_pad = (uint64_t(n_kv_max) + KB - 1) / KB * KB;
    return (uint64_t(T) * n_heads * D + n_kv_pad * D) * sizeof(sycl::half);
}

std::string ds4_attention_xmx_reserve(sycl::queue& q, uint32_t T, uint32_t n_heads,
                                      uint32_t n_kv_max) {
    using namespace ds4ax;
    const uint64_t n_kv_pad = (uint64_t(n_kv_max) + KB - 1) / KB * KB;
    return ds4ax_reserve(q, ds4ax_ws_for(q), uint64_t(T) * n_heads * D, n_kv_pad * D);
}

sycl::event ds4_attention_xmx(sycl::queue& q,
                              const float* q_in, const float* kv,
                              const float* mask, const float* sinks,
                              float* y,
                              uint32_t T, uint32_t n_heads, uint32_t head_dim,
                              uint32_t n_kv, float scaling,
                              const std::vector<sycl::event>& deps) {
    return ds4_attention_xmx_segs(q, q_in, Ds4KvSegs{kv, n_kv, nullptr, 0u, false}, mask, sinks, y,
                                  T, n_heads, head_dim, scaling, /*kv_prepared=*/false, deps);
}

sycl::event ds4_attention_xmx_segs(sycl::queue& q,
                                   const float* q_in, Ds4KvSegs kv,
                                   const float* mask, const float* sinks,
                                   float* y,
                                   uint32_t T, uint32_t n_heads, uint32_t head_dim,
                                   float scaling, bool kv_prepared,
                                   const std::vector<sycl::event>& deps,
                                   const sycl::half* kv_rows) {
    using namespace ds4ax;
    if (kv_rows && (kv.b || kv.n_b || kv.n_a % KB))
        throw std::invalid_argument("ds4_attention_xmx: per-row keys take one segment of whole 64-key blocks");
    if (head_dim != uint32_t(D) || n_heads % HT)
        throw std::invalid_argument("ds4_attention_xmx: head_dim must be 512 and n_heads a multiple of 32");
    const uint32_t n_kv     = kv.n_a + kv.n_b;
    const uint32_t n_kv_pad = (n_kv + KB - 1) / KB * KB;
    // Rows STAGED in fp16.  With a b segment: segment a padded out to a whole
    // key block, the b rows that share that block copied in after it (they are
    // fp16 already); every block at or past n_stage is read straight from b at
    // row n0 - n_a — the cache pads b with 64 finite rows, so the last block's
    // over-read is in bounds and masked.  Without a b segment: the whole
    // padded kv axis, as before.  Block boundaries are the contiguous layout's
    // in both cases, so the online-softmax rounding is unchanged.
    // A fp32 b segment (IE_DS4_KV16=0) cannot be read by the XMX tiles in
    // place: it is converted into the staging like the old single buffer.
    const uint32_t n_stage  = (kv.n_b && kv.b_f16)
                                  ? std::min<uint32_t>((kv.n_a + KB - 1) / KB * KB, n_kv_pad)
                                  : n_kv_pad;
    Ws& w = ds4ax_ws_for(q);
    if (std::string e = ds4ax_reserve(q, w, uint64_t(T) * n_heads * D, uint64_t(n_stage) * D); !e.empty())
        throw std::runtime_error(e);
    sycl::half* q16  = w.q16;
    sycl::half* kv16 = w.kv16;

    // ---- pre-passes: fp32 -> fp16 (SATURATING: a value past fp16's range
    // clamps instead of becoming inf and poisoning the token — the GLM fp16
    // overflow incident is the precedent), the KV pad rows zeroed (their scores
    // are masked).  Profiled under the kernel's own name so --kprofile counts
    // them with it (Phase K gate finding 2).
    auto sat16 = [](float v) {
        v = sycl::fmin(sycl::fmax(v, -65504.0f), 65504.0f);
        return sycl::half(v);
    };
    // At large prefill sizes the Q conversion launch and intermediate global
    // read cost more than saturating values during the existing SLM transpose.
    // Keep the staged path for smaller calls and preserve the reserved workspace
    // contract, including callers that alternate small and large strips.
    bool direct_q = false;
    if (T >= 512) {
        // Staging snapshots all Q before any output writes. Preserve that
        // behavior for partial as well as exact Q/output overlap. Check sizes
        // and address differences without allowing extent/end arithmetic to wrap.
        const uint64_t row_bytes = uint64_t(n_heads) * D * sizeof(float);
        const uintptr_t limit = std::numeric_limits<uintptr_t>::max();
        const uintptr_t qb = reinterpret_cast<uintptr_t>(q_in);
        const uintptr_t yb = reinterpret_cast<uintptr_t>(y);
        if (row_bytes && uint64_t(T) <= uint64_t(limit) / row_bytes) {
            const uintptr_t bytes = uintptr_t(uint64_t(T) * row_bytes);
            direct_q = bytes <= limit - qb && bytes <= limit - yb
                && (qb <= yb ? bytes <= yb - qb : bytes <= qb - yb);
        }
    }
    sycl::event eq;
    if (!direct_q) eq = ie::ps(q, "ds4_attention_xmx_prep", [&](sycl::handler& h) {
        h.depends_on(deps);
        const uint64_t n = uint64_t(T) * n_heads * D;
        h.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> i) { q16[i] = sat16(q_in[i]); });
    });
    // The KV staging holds the two segments in the OLD contiguous layout
    // ([a rows | b rows], blocks of KB from column 0), so the block grouping —
    // and with it the online-softmax rounding — is exactly what the
    // single-buffer form produced.  `kv_prepared` says a previous query-row
    // strip of this same layer already staged these rows on this queue: the
    // conversion is per layer, not per strip.
    sycl::event ek;
    if (!kv_prepared && !kv_rows) {
        const float*      ka   = kv.a;
        const sycl::half* kb16 = kv.b_f16 ? static_cast<const sycl::half*>(kv.b) : nullptr;
        const float*      kb32 = kv.b_f16 ? nullptr : static_cast<const float*>(kv.b);
        const uint64_t    ea   = uint64_t(kv.n_a) * D;     // elements in segment a
        ek = ie::ps(q, "ds4_attention_xmx_prep", [&](sycl::handler& h) {
            h.depends_on(deps);
            const uint64_t n = uint64_t(n_stage) * D, live = uint64_t(n_kv) * D;
            h.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> i) {
                kv16[i] = i < ea   ? sat16(ka[i])
                        : i < live ? (kb16 ? kb16[i - ea] : sat16(kb32[i - ea]))
                                   : sycl::half(0.f);
            });
        });
    }
    const sycl::half* kvb     = static_cast<const sycl::half*>(kv.b);
    const uint32_t    n_a     = kv.n_a;
    const uint32_t    n_stage_c = n_stage;
    const sycl::half* kvr       = kv_rows;

    const uint32_t n_ht = n_heads / HT;
    auto submit_attention = [&]<bool DirectQ, bool Rows>() {
    return ie::ps(q, "ds4_attention_xmx", [&](sycl::handler& h) {
        if constexpr (DirectQ) h.depends_on(deps);
        else h.depends_on(eq);
        h.depends_on(ek);
        sycl::local_accessor<sycl::half, 1> Qt(sycl::range<1>(size_t(D) * HT), h);
        sycl::local_accessor<float, 1>      St(sycl::range<1>(size_t(KB) * HT), h);
        sycl::local_accessor<sycl::half, 1> P (sycl::range<1>(size_t(HT) * KB), h);
        sycl::local_accessor<float, 1>      mk(sycl::range<1>(KB), h);
        sycl::local_accessor<float, 1>      red(sycl::range<1>(WG), h);
        sycl::local_accessor<float, 1>      mrun(sycl::range<1>(HT), h);
        sycl::local_accessor<float, 1>      lrun(sycl::range<1>(HT), h);
        sycl::local_accessor<float, 1>      alpha(sycl::range<1>(HT), h);
        sycl::local_accessor<float, 1>      mnew(sycl::range<1>(HT), h);
        h.parallel_for(sycl::nd_range<2>({size_t(T), size_t(n_ht) * WG}, {1, WG}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint32_t t   = uint32_t(it.get_group(0));
            const uint32_t ht  = uint32_t(it.get_group(1));
            const uint32_t lid = uint32_t(it.get_local_id(1));
            auto sg = it.get_sub_group();
            const uint32_t s = uint32_t(sg.get_group_linear_id());   // 0..15
            auto grp = it.get_group();

            // ---- stage Q^T once, with identical saturating fp16 conversion on both paths ----
            {
                const uint64_t qoff = (uint64_t(t) * n_heads + uint64_t(ht) * HT) * D;
                for (uint32_t i = lid; i < uint32_t(HT) * D; i += WG) {
                    const uint32_t hh = i / D, k = i % D;
                    if constexpr (DirectQ)
                        Qt[k * HT + hh] = sat16(q_in[qoff + i]);
                    else {
                        const sycl::half* qbase = q16 + qoff;
                        Qt[k * HT + hh] = qbase[uint64_t(hh) * D + k];
                    }
                }
            }
            if (lid < HT) { mrun[lid] = kNegInf; lrun[lid] = 0.f; }

            mat::joint_matrix<sycl::sub_group, float, mat::use::accumulator, TM, TN> acc[MR][NC];
            #pragma unroll
            for (int mr = 0; mr < MR; ++mr)
                #pragma unroll
                for (int nc = 0; nc < NC; ++nc) mat::joint_matrix_fill(sg, acc[mr][nc], 0.0f);

            const float* mrow = mask ? mask + uint64_t(t) * n_kv : nullptr;
            const uint32_t hh = lid & (HT - 1);      // this lane's head for the softmax
            const uint32_t jj = lid >> 5;            // 0..7: rows [8 jj, 8 jj + 8) of the block
            sycl::group_barrier(grp);

            for (uint32_t n0 = 0; n0 < n_kv; n0 += KB) {
                // ---- the block's mask entries; skip the block if none is allowed ----
                if (lid < KB) {
                    const uint32_t i = n0 + lid;
                    mk[lid] = (i < n_kv) ? (mrow ? mrow[i] : 0.f) : kNegInf;
                }
                sycl::group_barrier(grp);
                const bool allowed = sycl::any_of_group(grp, lid < KB && mk[lid] > kMaskedCut);
                if (!allowed) { sycl::group_barrier(grp); continue; }
                // This block's 64 fp16 key rows: staged, or straight from segment b.
                // (Rows: the row's own gathered axis.)
                const sycl::half* kblk = Rows ? kvr + (uint64_t(t) * n_kv + n0) * D
                                       : (n0 < n_stage_c) ? kv16 + uint64_t(n0) * D
                                                          : kvb + uint64_t(n0 - n_a) * D;

                // ---- S^T [KB][HT] = Kblk [KB][D] . Q^T [D][HT]: 8 row tiles x 2 head
                //      column tiles = 16 tiles, one per sub-group (rt = s >> 1, ct = s & 1) ----
                {
                    const uint32_t rt = s >> 1, ct = s & 1;
                    mat::joint_matrix<sycl::sub_group, float, mat::use::accumulator, TM, TN> sacc;
                    mat::joint_matrix_fill(sg, sacc, 0.0f);
                    const sycl::half* kbase = kblk + uint64_t(rt * TM) * D;
                    #pragma unroll 4
                    for (int kk = 0; kk < D; kk += TK) {
                        mat::joint_matrix<sycl::sub_group, sycl::half, mat::use::a, TM, TK,
                                          mat::layout::row_major> a_tile;
                        mat::joint_matrix_load(sg, a_tile,
                            sycl::address_space_cast<sycl::access::address_space::global_space,
                                                     sycl::access::decorated::no>(kbase + kk),
                            /*stride=*/D);
                        mat::joint_matrix<sycl::sub_group, sycl::half, mat::use::b, TK, TN,
                                          mat::layout::row_major> b_tile;
                        mat::joint_matrix_load(sg, b_tile,
                            Qt.template get_multi_ptr<sycl::access::decorated::no>() + kk * HT + ct * TN,
                            /*stride=*/HT);
                        mat::joint_matrix_mad(sg, sacc, a_tile, b_tile, sacc);
                    }
                    imat::joint_matrix_apply(sg, sacc, [&](float& v, size_t r, size_t c) {
                        St[(rt * TM + r) * HT + ct * TN + c] = v;
                    });
                }
                sycl::group_barrier(grp);

                // ---- online softmax per head; 8 lanes per head, 8 keys each ----
                float sc[8];
                float bm = kNegInf;
                #pragma unroll
                for (int r = 0; r < 8; ++r) {
                    const uint32_t row = jj * 8 + r;
                    const float mv = mk[row];
                    sc[r] = (mv > kMaskedCut) ? St[row * HT + hh] * scaling + mv : kNegInf;
                    bm = sycl::fmax(bm, sc[r]);
                }
                red[lid] = bm;
                sycl::group_barrier(grp);
                if (lid < HT) {
                    float m8 = kNegInf;
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) m8 = sycl::fmax(m8, red[j * HT + lid]);
                    const float m_old = mrun[lid];
                    const float m_n   = sycl::fmax(m_old, m8);
                    mnew[lid]  = m_n;
                    alpha[lid] = (m_old == kNegInf) ? 0.f : sycl::exp(m_old - m_n);
                }
                sycl::group_barrier(grp);
                {
                    const float m_n = mnew[hh];
                    float ps = 0.f;
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
                if (lid < HT) {
                    float l8 = 0.f;
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) l8 += red[j * HT + lid];
                    lrun[lid] = lrun[lid] * alpha[lid] + l8;
                    mrun[lid] = mnew[lid];
                }
                // (alpha[] is read below; the barrier before the next block's
                //  writes to mk/St/P/red also covers it.)

                // ---- O = O * alpha (per head row) + P [HT][KB] . Vblk [KB][D]: this
                //      sub-group's 32 columns [32 s, 32 s + 32) ----
                #pragma unroll
                for (int mr = 0; mr < MR; ++mr) {
                    #pragma unroll
                    for (int nc = 0; nc < NC; ++nc)
                        imat::joint_matrix_apply(sg, acc[mr][nc], [&](float& v, size_t r, size_t) {
                            v *= alpha[mr * TM + r];
                        });
                }
                #pragma unroll
                for (int kk = 0; kk < KB; kk += TK) {
                    mat::joint_matrix<sycl::sub_group, sycl::half, mat::use::b, TK, TN,
                                      mat::layout::row_major> b_tile[NC];
                    #pragma unroll
                    for (int nc = 0; nc < NC; ++nc)
                        mat::joint_matrix_load(sg, b_tile[nc],
                            sycl::address_space_cast<sycl::access::address_space::global_space,
                                                     sycl::access::decorated::no>(
                                kblk + uint64_t(kk) * D + s * DC + nc * TN),
                            /*stride=*/D);
                    #pragma unroll
                    for (int mr = 0; mr < MR; ++mr) {
                        mat::joint_matrix<sycl::sub_group, sycl::half, mat::use::a, TM, TK,
                                          mat::layout::row_major> a_tile;
                        mat::joint_matrix_load(sg, a_tile,
                            P.template get_multi_ptr<sycl::access::decorated::no>() + (mr * TM) * KB + kk,
                            /*stride=*/KB);
                        #pragma unroll
                        for (int nc = 0; nc < NC; ++nc)
                            mat::joint_matrix_mad(sg, acc[mr][nc], a_tile, b_tile[nc], acc[mr][nc]);
                    }
                }
                sycl::group_barrier(grp);
            }

            // ---- finalize: the sink column and 1/l, in the fp32 kernel's order ----
            if (lid < HT) {
                const float m = mrun[lid], l = lrun[lid];
                float corr = 1.f, ll = l;
                if (sinks) {
                    const float sv  = sinks[ht * HT + lid];
                    const float m_f = sycl::fmax(m, sv);
                    corr = (m == kNegInf) ? 0.f : sycl::exp(m - m_f);
                    ll   = l * corr + sycl::exp(sv - m_f);
                }
                const float inv = (ll > 0.f) ? (1.f / ll) : 0.f;
                alpha[lid] = corr * inv;   // reuse: the per-head output scale
            }
            sycl::group_barrier(grp);
            float* ybase = y + (uint64_t(t) * n_heads + uint64_t(ht) * HT) * D + s * DC;
            #pragma unroll
            for (int mr = 0; mr < MR; ++mr) {
                #pragma unroll
                for (int nc = 0; nc < NC; ++nc)
                    imat::joint_matrix_apply(sg, acc[mr][nc], [&](float& v, size_t r, size_t c) {
                        ybase[uint64_t(mr * TM + r) * D + nc * TN + c] = v * alpha[mr * TM + r];
                    });
            }
        });
    });
    };
    if (kv_rows) return direct_q ? submit_attention.template operator()<true, true>()
                                 : submit_attention.template operator()<false, true>();
    return direct_q ? submit_attention.template operator()<true, false>()
                    : submit_attention.template operator()<false, false>();
}

}  // namespace ie
