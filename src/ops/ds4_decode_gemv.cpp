// src/ops/ds4_decode_gemv.cpp — see include/ie/ds4_decode_gemv.hpp for why.
//
// SHAPE OF THE KERNEL.  One output row is owned by SGS sub-groups of 16 lanes;
// a work-group is always 16 sub-groups (256 work-items) and therefore covers
// 16/SGS rows.  Each lane walks its row with 32-byte loads (V=16 halves) and a
// serial fp32 fma chain, the 16 lanes tree-reduce inside the sub-group with no
// barrier, and when SGS > 1 the SGS partials meet through 16 floats of SLM.
//
// WHY NOT THE EXISTING SCALAR KERNEL.  `dense_f16<TT>` at T=1 spends a whole
// 128-lane work-group per output row doing 2-byte scalar loads, then runs FOUR
// `reduce_over_group`s (TT=4) to publish ONE result.  Measured cache-cold at the
// real per-card decode shapes it is 1.6x-2.6x slower than this kernel.
//
// MEASUREMENTS (B70, M=1, cache-cold: >=200 MB of rotating weight buffers so
// nothing survives the 25.17 MB L2 between repetitions; us/call):
//
//   shape            K      N   MB | dense_f16 | oneDNN | oneDNN | this  | vs
//                                  |    TT=4   | +cast  |  mm    | GEMV  | oneDNN
//   q_a           4096   1024  8.4 |   28.66   | 21.13  | 17.03  | 15.70 | 1.35x
//   q_b           1024  16384 33.6 |  134.60   | 66.85  | 62.48  | 59.83 | 1.12x
//   kv            4096    512  4.2 |   15.38   | 15.55  | 11.37  |  8.75 | 1.78x
//   o_b           4096   4096 33.6 |  106.57   | 64.62  | 60.19  | 58.88 | 1.10x
//   idx_q_b       1024   8192 16.8 |   68.32   | 36.81  | 31.34  | 31.02 | 1.19x
//   idx_comp      4096    256  2.1 |   15.29   |  6.69  |  5.82  |  5.06 | 1.32x
//   idx_proj      4096     64  0.5 |   15.93   |  5.78  |  3.84  |  2.38 | 2.42x
//   shexp_down    1024   4096  8.4 |    -      |   -    |   -    |   -   |  (*)
//   grouped o_a   4096   4096 33.6 |  126.32   | 64.01  | 59.66  | 59.20 | 1.08x
//   lm_head       4096 129280 1059 | 3202.38   |1771.95 |1769.92 |1765.94| 1.00x
//
// The lm_head row is the honesty check on the whole table: 1.77 ms is also what
// the real model's profiler reports for ds4_dense_f16@lmhead, so the microbench
// reproduces the in-model number to 0.1%.  It is also why lm_head is the one
// shape with nothing to win — at 1059 MB it is pure DRAM streaming and both
// routes already run at 600 GB/s, the card's measured ceiling.
#include "ie/ds4_decode_gemv.hpp"

#include "ie/kernel_profiler.hpp"

#include <array>

namespace ie {

namespace {

constexpr int kSG   = 16;   // sub-group width (B70 supports 16 and 32)
constexpr int kWGSG = 16;   // sub-groups per work-group  -> 256 work-items

// Sub-groups cooperating on one output row.  At SGS=1 a row is one sub-group,
// so the launch has exactly N sub-groups; the card has 2048 hardware threads,
// and below N ~ 512 that leaves most of it idle.  Widening the row spreads the
// same bytes over more threads.  Measured cache-cold, us/call, K=4096:
//   N=  64:  SGS=1 7.77   SGS=4 2.99   SGS=8 2.38   SGS=16 2.61
//   N= 256:  SGS=1 7.62   SGS=4 5.06   SGS=8 5.16
//   N= 512:  SGS=1 8.75   SGS=4 8.81
//   N=1024:  SGS=1 15.70  SGS=4 16.28
// Above N=512 extra sub-groups per row only add reduction work, so the rule
// stops there rather than tracking occupancy further.
constexpr uint32_t sgs_for(uint32_t N) noexcept {
    return N >= 512 ? 1u : (N >= 128 ? 4u : 8u);
}

// OPG == 0 selects the dense form; OPG > 0 selects the block-diagonal form, in
// which row n reads activation block (n / OPG) and K carries IPG.
template <int SGS, int V>
sycl::event launch(sycl::queue& q, const char* name, const float* x, const sycl::half* w,
                   float* y, uint32_t K, uint32_t N, uint32_t OPG,
                   const std::vector<sycl::event>& deps) {
    constexpr int ROWS = kWGSG / SGS;
    constexpr int WG   = kWGSG * kSG;
    const uint32_t nwg  = (N + ROWS - 1) / ROWS;
    const uint32_t step = uint32_t(SGS) * kSG * V;
    return ie::ps(q, name, [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<float, 1> part(sycl::range<1>(kWGSG), h);
        h.parallel_for(sycl::nd_range<1>(size_t(nwg) * WG, WG),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
            auto sg = it.get_sub_group();
            const uint32_t sgid = uint32_t(sg.get_group_id()[0]);
            const uint32_t lane = uint32_t(sg.get_local_id()[0]);
            const uint32_t r    = sgid / SGS;          // which row of this work-group
            const uint32_t s    = sgid % SGS;          // which slice of that row
            const uint32_t n    = uint32_t(it.get_group(0)) * ROWS + r;

            float acc = 0.f;
            if (n < N) {
                const sycl::half* wr = w + uint64_t(n) * K;
                const float*      xr = OPG ? x + uint64_t(n / OPG) * K : x;
                const uint32_t  off0 = (s * kSG + lane) * V;
                // Vector body: the SGS*16 lanes of a row cover `step` contiguous
                // halves per iteration, lane l taking bytes [2*V*l, 2*V*(l+1)).
                for (uint32_t k0 = 0; k0 + step <= K; k0 += step) {
                    const uint32_t kk = k0 + off0;
                    const auto wv = *reinterpret_cast<const sycl::vec<sycl::half, V>*>(wr + kk);
                    const auto xv = *reinterpret_cast<const sycl::vec<float, V>*>(xr + kk);
                    #pragma unroll
                    for (int v = 0; v < V; ++v) acc = sycl::fma(float(wv[v]), xv[v], acc);
                }
                // Tail when K is not a whole number of steps.  Lane-strided, so
                // it is coalesced too; it just cannot use the wide load.
                for (uint32_t k = (K / step) * step + s * kSG + lane; k < K; k += uint32_t(SGS) * kSG)
                    acc = sycl::fma(float(wr[k]), xr[k], acc);
            }

            acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
            if (SGS == 1) {
                if (lane == 0 && n < N) y[n] = acc;
            } else {
                if (lane == 0) part[sgid] = acc;
                it.barrier(sycl::access::fence_space::local_space);
                const uint32_t lid = uint32_t(it.get_local_id(0));
                if (lid < uint32_t(ROWS)) {
                    const uint32_t nn = uint32_t(it.get_group(0)) * ROWS + lid;
                    if (nn < N) {
                        float t = 0.f;
                        #pragma unroll
                        for (int i = 0; i < SGS; ++i) t += part[lid * SGS + i];
                        y[nn] = t;
                    }
                }
            }
        });
    });
}

// The 32-byte load needs `w + n*K` 32-byte aligned for every n, which needs
// K % 16 == 0 (USM allocations are already over-aligned), and needs at least
// one whole step to exist.  Everything else takes V=1, which is the same kernel
// with scalar loads and is correct for any K.
template <int SGS>
sycl::event pick_v(sycl::queue& q, const char* name, const float* x, const sycl::half* w,
                   float* y, uint32_t K, uint32_t N, uint32_t OPG,
                   const std::vector<sycl::event>& deps) {
    if (K % 16 == 0 && K >= uint32_t(SGS) * kSG * 16)
        return launch<SGS, 16>(q, name, x, w, y, K, N, OPG, deps);
    return launch<SGS, 1>(q, name, x, w, y, K, N, OPG, deps);
}

sycl::event dispatch(sycl::queue& q, const char* name, const float* x, const sycl::half* w,
                     float* y, uint32_t K, uint32_t N, uint32_t OPG,
                     const std::vector<sycl::event>& deps) {
    if (!name) name = "ds4_decode_gemv";
    switch (sgs_for(N)) {
        case 1:  return pick_v<1>(q, name, x, w, y, K, N, OPG, deps);
        case 4:  return pick_v<4>(q, name, x, w, y, K, N, OPG, deps);
        default: return pick_v<8>(q, name, x, w, y, K, N, OPG, deps);
    }
}

}  // namespace

sycl::event ds4_decode_gemv_f16(sycl::queue& q, const char* prof_name,
                                const float* x, const sycl::half* w, float* y,
                                uint32_t K, uint32_t N,
                                const std::vector<sycl::event>& deps) {
    return dispatch(q, prof_name, x, w, y, K, N, /*OPG=*/0, deps);
}

sycl::event ds4_decode_gemv_grouped_f16(sycl::queue& q, const char* prof_name,
                                        const float* x, const sycl::half* w, float* y,
                                        uint32_t G, uint32_t IPG, uint32_t OPG,
                                        const std::vector<sycl::event>& deps) {
    return dispatch(q, prof_name, x, w, y, IPG, G * OPG, OPG, deps);
}

uint32_t ds4_decode_gemv_rounding_steps(uint32_t K, uint32_t N) noexcept {
    const uint32_t sgs = sgs_for(N);
    const uint32_t lanes = sgs * kSG;                 // lanes cooperating on a row
    const uint32_t per_lane = (K + lanes - 1) / lanes; // longest serial fma chain
    // + 4 for the 16-lane tree reduction, + (sgs - 1) for the serial cross-
    // sub-group fixup.  Each is one fp32 rounding.
    return per_lane + 4 + (sgs - 1);
}

// ===========================================================================
// Q8_0-SoA — same geometry, one 32-element block per lane per step
// ===========================================================================
//
// The block is the unit because it is the unit the scale applies to: a lane
// reads 32 int8 as two 16 B vectors, dots them with 32 fp32 activations, and
// folds the single fp16 scale in ONCE at the end.  That is 3 weight-load
// instructions per 32 elements against the fp16 kernel's 2 per 32 — 1.5x the
// issue rate for 0.53x the bytes, which is why it wins wherever the shape is
// bandwidth-bound and loses on the tiny ones (see the table below).
//
// MEASURED, B70, M=1, cache-cold (>=250 MB of rotating distinct weight buffers,
// batched submission so no per-call round trip is counted), us/call:
//
//   shape                  K      N   f16 MB  q8 MB | f16 us | q8 us | speed
//   q_a                 4096   1024     8.39   4.46 |  15.71 |  9.47 | 1.66x
//   q_b                 1024  16384    33.55  17.83 |  59.71 | 33.19 | 1.80x
//   kv                  4096    512     4.19   2.23 |   8.77 |  6.28 | 1.40x
//   o_a (grouped, G=4)  4096   4096    33.55  17.83 |  59.20 | 32.84 | 1.80x
//   o_b                 4096   4096    33.55  17.83 |  58.83 | 32.60 | 1.80x
//   sh_down             1024   4096     8.39   4.46 |  16.51 |  9.92 | 1.66x
//   idx_q_b             1024   8192    16.78   8.91 |  31.01 | 17.99 | 1.72x
//   idx_proj            4096     64     0.52   0.28 |   2.48 |  2.72 | 0.91x  <-- LOSES
//   idx_comp            4096    256     2.10   1.11 |   5.02 |  3.55 | 1.41x
//   lm_head             4096  64640   529.53 281.31 | 889.55 |478.64 | 1.86x
//
// The f16 column reproduces the shipped table at the top of this file to within
// 1% at every shape it shares (q_a 15.71 vs 15.70, o_b 58.83 vs 58.88, grouped
// o_a 59.20 vs 59.20), so the two are the same measurement, not two harnesses.
//
// idx_proj is the one shape where this layout LOSES, and it is not an artefact:
// at 0.5 MB the call is launch-bound, not bandwidth-bound, so the extra scale
// load is pure cost.  src/model/deepseek4.cpp excludes that tensor from
// requantisation for this reason (and for a second, independent one).
namespace {

template <int SGS>
sycl::event launch_q8(sycl::queue& q, const char* name, const float* x, const int8_t* qs,
                      const sycl::half* dsc, float* y, uint32_t K, uint32_t N, uint32_t OPG,
                      const std::vector<sycl::event>& deps) {
    constexpr int ROWS = kWGSG / SGS;
    constexpr int WG   = kWGSG * kSG;
    const uint32_t nwg   = (N + ROWS - 1) / ROWS;
    const uint32_t nb    = K / 32;
    const uint32_t lanes = uint32_t(SGS) * kSG;
    return ie::ps(q, name, [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<float, 1> part(sycl::range<1>(kWGSG), h);
        h.parallel_for(sycl::nd_range<1>(size_t(nwg) * WG, WG),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
            auto sg = it.get_sub_group();
            const uint32_t sgid = uint32_t(sg.get_group_id()[0]);
            const uint32_t lane = uint32_t(sg.get_local_id()[0]);
            const uint32_t r    = sgid / SGS;
            const uint32_t s    = sgid % SGS;
            const uint32_t n    = uint32_t(it.get_group(0)) * ROWS + r;

            float acc = 0.f;
            if (n < N) {
                const int8_t*     wr = qs  + uint64_t(n) * K;
                const sycl::half* dr = dsc + uint64_t(n) * nb;
                const float*      xr = OPG ? x + uint64_t(n / OPG) * K : x;
                // Row base is 16 B aligned (USM over-aligns, and K % 32 == 0),
                // and every block starts at a multiple of 32 within the row, so
                // both halves of the 16 B load pair are legally aligned.
                for (uint32_t b = s * kSG + lane; b < nb; b += lanes) {
                    const uint32_t k0 = b * 32;
                    const auto q0 = *reinterpret_cast<const sycl::vec<int8_t, 16>*>(wr + k0);
                    const auto q1 = *reinterpret_cast<const sycl::vec<int8_t, 16>*>(wr + k0 + 16);
                    const auto x0 = *reinterpret_cast<const sycl::vec<float, 16>*>(xr + k0);
                    const auto x1 = *reinterpret_cast<const sycl::vec<float, 16>*>(xr + k0 + 16);
                    float bs = 0.f;
                    #pragma unroll
                    for (int j = 0; j < 16; ++j) bs = sycl::fma(float(q0[j]), x0[j], bs);
                    #pragma unroll
                    for (int j = 0; j < 16; ++j) bs = sycl::fma(float(q1[j]), x1[j], bs);
                    acc = sycl::fma(float(dr[b]), bs, acc);
                }
            }

            acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
            if (SGS == 1) {
                if (lane == 0 && n < N) y[n] = acc;
            } else {
                if (lane == 0) part[sgid] = acc;
                it.barrier(sycl::access::fence_space::local_space);
                const uint32_t lid = uint32_t(it.get_local_id(0));
                if (lid < uint32_t(ROWS)) {
                    const uint32_t nn = uint32_t(it.get_group(0)) * ROWS + lid;
                    if (nn < N) {
                        float t = 0.f;
                        #pragma unroll
                        for (int i = 0; i < SGS; ++i) t += part[lid * SGS + i];
                        y[nn] = t;
                    }
                }
            }
        });
    });
}

sycl::event dispatch_q8(sycl::queue& q, const char* name, const float* x, const int8_t* qs,
                        const sycl::half* dsc, float* y, uint32_t K, uint32_t N, uint32_t OPG,
                        const std::vector<sycl::event>& deps) {
    if (!name) name = "ds4_decode_gemv_q8";
    switch (sgs_for(N)) {
        case 1:  return launch_q8<1>(q, name, x, qs, dsc, y, K, N, OPG, deps);
        case 4:  return launch_q8<4>(q, name, x, qs, dsc, y, K, N, OPG, deps);
        default: return launch_q8<8>(q, name, x, qs, dsc, y, K, N, OPG, deps);
    }
}

}  // namespace

sycl::event ds4_decode_gemv_q8(sycl::queue& q, const char* prof_name,
                               const float* x, const int8_t* qs, const sycl::half* d,
                               float* y, uint32_t K, uint32_t N,
                               const std::vector<sycl::event>& deps) {
    return dispatch_q8(q, prof_name, x, qs, d, y, K, N, /*OPG=*/0, deps);
}

sycl::event ds4_decode_gemv_grouped_q8(sycl::queue& q, const char* prof_name,
                                       const float* x, const int8_t* qs, const sycl::half* d,
                                       float* y, uint32_t G, uint32_t IPG, uint32_t OPG,
                                       const std::vector<sycl::event>& deps) {
    return dispatch_q8(q, prof_name, x, qs, d, y, IPG, G * OPG, OPG, deps);
}

uint32_t ds4_decode_gemv_q8_rounding_steps(uint32_t K, uint32_t N) noexcept {
    const uint32_t sgs   = sgs_for(N);
    const uint32_t lanes = sgs * kSG;
    const uint32_t nb    = K / 32;
    const uint32_t per_lane = (nb + lanes - 1) / lanes;   // blocks this lane owns
    // 32 serial fma inside the lane's first block, then one fma per block into
    // the row accumulator, then the 4-level sub-group tree and the (sgs - 1)
    // serial cross-sub-group fixup.
    return 32u + per_lane + 4u + (sgs - 1u);
}

// ===========================================================================
// MULTI-PROJECTION — the same two kernels, several weights per launch
// ===========================================================================
//
// See the contract on `ds4_decode_gemv_multi` in the header.  The one thing the
// implementation has to get right is that a projection's geometry must not
// change when it is fused: `sgs`, `rows = 16/sgs`, the V=16-vs-scalar choice,
// the loop bounds and the reduction shape are all recomputed here from exactly
// the same rules `sgs_for` / `pick_v` apply to a lone call, only at runtime
// instead of at compile time.  That is what makes the fusion bit-identical and
// it is what `deepseek4_decgemv_gate_test` checks by EQUALITY.
//
// The descriptor table is captured BY VALUE into the kernel argument block.
// Staging it in device memory would have cost a memcpy submission and given
// back a third of what the fusion buys.
namespace {

struct Ds4MSeg {
    const sycl::half* w;
    const int8_t*     qs;
    const sycl::half* qd;
    float*            y;
    uint32_t          N;
    uint32_t          sgs;
    uint32_t          wg0;   // first work-group index of this projection
    uint32_t          pad;
};

// The work-group -> projection lookup.  Written as an unrolled chain of
// predicated whole-struct copies rather than `segs[j]`, because a dynamic index
// into a private array spills it to scratch; this keeps it in registers.  It is
// work-group-uniform, so every barrier below is reached uniformly.
template <uint32_t P>
inline Ds4MSeg pick_seg(const std::array<Ds4MSeg, P>& segs, uint32_t nseg, uint32_t gwg) {
    Ds4MSeg s = segs[0];
    #pragma unroll
    for (uint32_t t = 1; t < P; ++t)
        if (t < nseg && gwg >= segs[t].wg0) s = segs[t];
    return s;
}

// fp16 form, UNIFORM geometry: every projection in the run shares (SGS, V), so
// both stay compile-time and the inner loop is `launch<SGS, V>`'s verbatim.
//
// WHY THIS EXISTS AND THE GENERIC ONE IS NOT ENOUGH.  The generic kernel below
// takes SGS and V as runtime values, which costs the compiler the trip count of
// the k-loop and with it some of the load pipelining.  Measured cache-cold on a
// B70 over the real per-card HCA attention set (q_a 1024, kv 512, comp_kv 512,
// comp_gate 512, all SGS=1, 21.0 MB): four separate launches 42.65 us
// (492 GB/s), one GENERIC launch 45.42 us (462 GB/s).  The 3 launches saved are
// worth ~5.0 us of host submit, so even that was a net win on the critical
// path, but there is no reason to pay it: the run is uniform, so specialise.
template <uint32_t P, int SGS, int V>
sycl::event launch_multi_f16_u(sycl::queue& q, const char* name, const float* x,
                               const std::array<Ds4MSeg, P>& segs, uint32_t nseg,
                               uint32_t total_wg, uint32_t K,
                               const std::vector<sycl::event>& deps) {
    constexpr int ROWS = kWGSG / SGS;
    constexpr int WG   = kWGSG * kSG;
    const uint32_t step = uint32_t(SGS) * kSG * V;
    return ie::ps(q, name, [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<float, 1> part(sycl::range<1>(kWGSG), h);
        h.parallel_for(sycl::nd_range<1>(size_t(total_wg) * WG, WG),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
            const uint32_t gwg = uint32_t(it.get_group(0));
            const Ds4MSeg  sg_ = pick_seg<P>(segs, nseg, gwg);
            const uint32_t lwg = gwg - sg_.wg0;
            const uint32_t N   = sg_.N;

            auto sg = it.get_sub_group();
            const uint32_t sgid = uint32_t(sg.get_group_id()[0]);
            const uint32_t lane = uint32_t(sg.get_local_id()[0]);
            const uint32_t r    = sgid / SGS;
            const uint32_t s    = sgid % SGS;
            const uint32_t n    = lwg * ROWS + r;

            float acc = 0.f;
            if (n < N) {
                const sycl::half* wr = sg_.w + uint64_t(n) * K;
                const uint32_t  off0 = (s * kSG + lane) * V;
                for (uint32_t k0 = 0; k0 + step <= K; k0 += step) {
                    const uint32_t kk = k0 + off0;
                    const auto wv = *reinterpret_cast<const sycl::vec<sycl::half, V>*>(wr + kk);
                    const auto xv = *reinterpret_cast<const sycl::vec<float, V>*>(x + kk);
                    #pragma unroll
                    for (int v = 0; v < V; ++v) acc = sycl::fma(float(wv[v]), xv[v], acc);
                }
                for (uint32_t k = (K / step) * step + s * kSG + lane; k < K; k += uint32_t(SGS) * kSG)
                    acc = sycl::fma(float(wr[k]), x[k], acc);
            }

            acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
            if (SGS == 1) {
                if (lane == 0 && n < N) sg_.y[n] = acc;
            } else {
                if (lane == 0) part[sgid] = acc;
                it.barrier(sycl::access::fence_space::local_space);
                const uint32_t lid = uint32_t(it.get_local_id(0));
                if (lid < uint32_t(ROWS)) {
                    const uint32_t nn = lwg * ROWS + lid;
                    if (nn < N) {
                        float t = 0.f;
                        #pragma unroll
                        for (int i = 0; i < SGS; ++i) t += part[lid * SGS + i];
                        sg_.y[nn] = t;
                    }
                }
            }
        });
    });
}

// fp16 form.  Body transcribed from `launch<SGS, V>` with SGS and V runtime.
template <uint32_t P>
sycl::event launch_multi_f16(sycl::queue& q, const char* name, const float* x,
                             const std::array<Ds4MSeg, P>& segs, uint32_t nseg,
                             uint32_t total_wg, uint32_t K,
                             const std::vector<sycl::event>& deps) {
    constexpr int WG = kWGSG * kSG;
    return ie::ps(q, name, [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<float, 1> part(sycl::range<1>(kWGSG), h);
        h.parallel_for(sycl::nd_range<1>(size_t(total_wg) * WG, WG),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
            const uint32_t gwg = uint32_t(it.get_group(0));
            const Ds4MSeg  s   = pick_seg<P>(segs, nseg, gwg);
            const uint32_t sgs  = s.sgs;
            const uint32_t rows = uint32_t(kWGSG) / sgs;
            const uint32_t lwg  = gwg - s.wg0;

            auto sg = it.get_sub_group();
            const uint32_t sgid = uint32_t(sg.get_group_id()[0]);
            const uint32_t lane = uint32_t(sg.get_local_id()[0]);
            const uint32_t r    = sgid / sgs;
            const uint32_t sl   = sgid % sgs;
            const uint32_t n    = lwg * rows + r;
            const uint32_t lanes = sgs * uint32_t(kSG);

            float acc = 0.f;
            if (n < s.N) {
                const sycl::half* wr = s.w + uint64_t(n) * K;
                // `pick_v`'s rule, evaluated at runtime.
                if (K % 16u == 0u && K >= lanes * 16u) {
                    const uint32_t step = lanes * 16u;
                    const uint32_t off0 = (sl * uint32_t(kSG) + lane) * 16u;
                    for (uint32_t k0 = 0; k0 + step <= K; k0 += step) {
                        const uint32_t kk = k0 + off0;
                        const auto wv = *reinterpret_cast<const sycl::vec<sycl::half, 16>*>(wr + kk);
                        const auto xv = *reinterpret_cast<const sycl::vec<float, 16>*>(x + kk);
                        #pragma unroll
                        for (int v = 0; v < 16; ++v) acc = sycl::fma(float(wv[v]), xv[v], acc);
                    }
                    for (uint32_t k = (K / step) * step + sl * uint32_t(kSG) + lane; k < K; k += lanes)
                        acc = sycl::fma(float(wr[k]), x[k], acc);
                } else {
                    const uint32_t step = lanes;
                    const uint32_t off0 = sl * uint32_t(kSG) + lane;
                    for (uint32_t k0 = 0; k0 + step <= K; k0 += step) {
                        const uint32_t kk = k0 + off0;
                        acc = sycl::fma(float(wr[kk]), x[kk], acc);
                    }
                    for (uint32_t k = (K / step) * step + off0; k < K; k += lanes)
                        acc = sycl::fma(float(wr[k]), x[k], acc);
                }
            }

            acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
            if (sgs == 1u) {
                if (lane == 0 && n < s.N) s.y[n] = acc;
            } else {
                if (lane == 0) part[sgid] = acc;
                it.barrier(sycl::access::fence_space::local_space);
                const uint32_t lid = uint32_t(it.get_local_id(0));
                if (lid < rows) {
                    const uint32_t nn = lwg * rows + lid;
                    if (nn < s.N) {
                        float t = 0.f;
                        for (uint32_t i = 0; i < sgs; ++i) t += part[lid * sgs + i];
                        s.y[nn] = t;
                    }
                }
            }
        });
    });
}

// Q8_0-SoA form.  Body transcribed from `launch_q8<SGS>` with SGS runtime.
template <uint32_t P>
sycl::event launch_multi_q8(sycl::queue& q, const char* name, const float* x,
                            const std::array<Ds4MSeg, P>& segs, uint32_t nseg,
                            uint32_t total_wg, uint32_t K,
                            const std::vector<sycl::event>& deps) {
    constexpr int WG = kWGSG * kSG;
    const uint32_t nb = K / 32;
    return ie::ps(q, name, [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<float, 1> part(sycl::range<1>(kWGSG), h);
        h.parallel_for(sycl::nd_range<1>(size_t(total_wg) * WG, WG),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
            const uint32_t gwg = uint32_t(it.get_group(0));
            const Ds4MSeg  s   = pick_seg<P>(segs, nseg, gwg);
            const uint32_t sgs  = s.sgs;
            const uint32_t rows = uint32_t(kWGSG) / sgs;
            const uint32_t lwg  = gwg - s.wg0;

            auto sg = it.get_sub_group();
            const uint32_t sgid = uint32_t(sg.get_group_id()[0]);
            const uint32_t lane = uint32_t(sg.get_local_id()[0]);
            const uint32_t r    = sgid / sgs;
            const uint32_t sl   = sgid % sgs;
            const uint32_t n    = lwg * rows + r;
            const uint32_t lanes = sgs * uint32_t(kSG);

            float acc = 0.f;
            if (n < s.N) {
                const int8_t*     wr = s.qs + uint64_t(n) * K;
                const sycl::half* dr = s.qd + uint64_t(n) * nb;
                for (uint32_t b = sl * uint32_t(kSG) + lane; b < nb; b += lanes) {
                    const uint32_t k0 = b * 32;
                    const auto q0 = *reinterpret_cast<const sycl::vec<int8_t, 16>*>(wr + k0);
                    const auto q1 = *reinterpret_cast<const sycl::vec<int8_t, 16>*>(wr + k0 + 16);
                    const auto x0 = *reinterpret_cast<const sycl::vec<float, 16>*>(x + k0);
                    const auto x1 = *reinterpret_cast<const sycl::vec<float, 16>*>(x + k0 + 16);
                    float bs = 0.f;
                    #pragma unroll
                    for (int j = 0; j < 16; ++j) bs = sycl::fma(float(q0[j]), x0[j], bs);
                    #pragma unroll
                    for (int j = 0; j < 16; ++j) bs = sycl::fma(float(q1[j]), x1[j], bs);
                    acc = sycl::fma(float(dr[b]), bs, acc);
                }
            }

            acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
            if (sgs == 1u) {
                if (lane == 0 && n < s.N) s.y[n] = acc;
            } else {
                if (lane == 0) part[sgid] = acc;
                it.barrier(sycl::access::fence_space::local_space);
                const uint32_t lid = uint32_t(it.get_local_id(0));
                if (lid < rows) {
                    const uint32_t nn = lwg * rows + lid;
                    if (nn < s.N) {
                        float t = 0.f;
                        for (uint32_t i = 0; i < sgs; ++i) t += part[lid * sgs + i];
                        s.y[nn] = t;
                    }
                }
            }
        });
    });
}

// THE WAVE SAWTOOTH, AND WHY THE PACKER IS NOT SIMPLY "FUSE EVERYTHING".
//
// This kernel's achieved bandwidth is not a function of how many bytes a launch
// reads; it is a function of how many WORK-GROUPS the launch has, and it saws
// with a period of 128.  Measured on a B70, K=4096, SGS=1, cache-cold over a
// 360 MB rotating pool, 9 interleaved reps, min-reduced (us and GB/s):
//
//   wg   16   32   48   64   80   96  112  128 | 144  160  176  192  208  224  240  256
//   GB/s 574  511  498  548  532  563  546  570| 451  480  513  538  549  559  568  571
//                                              ^ the cliff
//
// The card retires 128 work-groups per wave.  A launch of 144 or 160 pays for a
// whole second wave to run 16 or 32 groups in, and loses up to 21% of the card.
// The same total work split so that no launch straddles a wave boundary keeps
// the peak.  Verified against the real per-card HCA attention set (q_a 1024,
// kv 512, comp_kv 512, comp_gate 512 = 160 work-groups), every contiguous
// partition measured, 11 interleaved reps:
//
//   4 separate launches   submit 6.97   wall 42.45 us
//   1 fused launch        submit 2.05   wall 44.77 us   <- 160 wg, straddles
//   2 fused launches      submit 3.87   wall 39.64 us   <- 128 + 32, best
//
// So the packer maximises fused length FIRST and only cuts when the run would
// end just past a wave boundary — which is the one case where an extra launch
// (~1.79 us of host submit) buys more than it costs.  The CSA set totals
// exactly 256 work-groups, lands on a boundary, and is therefore NOT cut: it
// measures 54.52 us fused against 59.41 us as five launches, better on BOTH
// axes at once.
constexpr uint32_t kWaveWg = 128;

uint32_t seg_wg(const Ds4GemvProj& p) {
    const uint32_t rows = uint32_t(kWGSG) / sgs_for(p.N);
    return (p.N + rows - 1) / rows;
}

// How many of the `n` projections the first launch should take.  Returns `n`
// unless the whole run would straddle a wave boundary badly, in which case it
// returns the prefix that best fills whole waves.
uint32_t wave_cut(const Ds4GemvProj* p, uint32_t n) {
    if (n < 2) return n;
    uint32_t total = 0;
    for (uint32_t j = 0; j < n; ++j) total += seg_wg(p[j]);
    const uint32_t tail = total % kWaveWg;
    // On a boundary, or under one wave, or the tail already fills most of a
    // wave: nothing to gain, keep the single launch.
    if (total <= kWaveWg || tail == 0 || tail > kWaveWg / 2) return n;
    // Score a prefix by how full it leaves its last wave; a prefix that lands
    // exactly on a boundary scores highest.
    uint32_t best_p = n, best_s = tail;
    uint32_t cum = 0;
    for (uint32_t j = 0; j + 1 < n; ++j) {
        cum += seg_wg(p[j]);
        const uint32_t s = (cum % kWaveWg == 0) ? kWaveWg : (cum % kWaveWg);
        if (s > best_s) { best_s = s; best_p = j + 1; }
    }
    return best_p;
}

// One run of same-dtype projections -> one launch.
sycl::event submit_run(sycl::queue& q, const char* name, const float* x,
                       const Ds4GemvProj* projs, uint32_t n, uint32_t K, bool f16,
                       const std::vector<sycl::event>& deps) {
    std::array<Ds4MSeg, kDs4GemvMultiMax> segs{};
    uint32_t total = 0;
    for (uint32_t j = 0; j < n; ++j) {
        const uint32_t sgs  = sgs_for(projs[j].N);
        const uint32_t rows = uint32_t(kWGSG) / sgs;
        segs[j] = Ds4MSeg{projs[j].w, projs[j].qs, projs[j].qd, projs[j].y,
                          projs[j].N, sgs, total, 0u};
        total += (projs[j].N + rows - 1) / rows;
    }
    // Every unused slot must still route somewhere legal: `pick_seg` starts at
    // slot 0 and only advances into slots < nseg, so the tail is never read.
    for (uint32_t j = n; j < kDs4GemvMultiMax; ++j) segs[j] = segs[0];
    if (!f16) return launch_multi_q8<kDs4GemvMultiMax>(q, name, x, segs, n, total, K, deps);

    // Uniform geometry -> the compile-time kernel.  `vec` is `pick_v`'s rule;
    // it only depends on K and sgs, so a run that agrees on sgs agrees on it.
    bool uniform = true;
    for (uint32_t j = 1; j < n; ++j) uniform = uniform && segs[j].sgs == segs[0].sgs;
    if (uniform) {
        const uint32_t lanes = segs[0].sgs * uint32_t(kSG);
        const bool vec = (K % 16 == 0) && (K >= lanes * 16u);
        if (vec) switch (segs[0].sgs) {
            case 1: return launch_multi_f16_u<kDs4GemvMultiMax, 1, 16>(q, name, x, segs, n, total, K, deps);
            case 4: return launch_multi_f16_u<kDs4GemvMultiMax, 4, 16>(q, name, x, segs, n, total, K, deps);
            default: return launch_multi_f16_u<kDs4GemvMultiMax, 8, 16>(q, name, x, segs, n, total, K, deps);
        }
        switch (segs[0].sgs) {
            case 1: return launch_multi_f16_u<kDs4GemvMultiMax, 1, 1>(q, name, x, segs, n, total, K, deps);
            case 4: return launch_multi_f16_u<kDs4GemvMultiMax, 4, 1>(q, name, x, segs, n, total, K, deps);
            default: return launch_multi_f16_u<kDs4GemvMultiMax, 8, 1>(q, name, x, segs, n, total, K, deps);
        }
    }
    return launch_multi_f16<kDs4GemvMultiMax>(q, name, x, segs, n, total, K, deps);
}

}  // namespace

sycl::event ds4_decode_gemv_multi(sycl::queue& q, const char* prof_name,
                                  const float* x, const Ds4GemvProj* projs, uint32_t n_proj,
                                  uint32_t K, const std::vector<sycl::event>& deps) {
    if (!prof_name) prof_name = "ds4_decode_gemv_multi";
    // Degenerate cases go to the single-projection entry points verbatim, so a
    // caller can hand this a one-element list without paying anything for the
    // generality and without changing a single instruction of what runs.
    if (n_proj == 0) {
        return ie::ps(q, prof_name, [&](sycl::handler& h) {
            h.depends_on(deps);
            h.single_task([]() {});
        });
    }
    if (n_proj == 1) {
        return projs[0].w
                   ? ds4_decode_gemv_f16(q, prof_name, x, projs[0].w, projs[0].y, K,
                                         projs[0].N, deps)
                   : ds4_decode_gemv_q8(q, prof_name, x, projs[0].qs, projs[0].qd, projs[0].y,
                                        K, projs[0].N, deps);
    }
    // Runs of one dtype, at most kDs4GemvMultiMax long.  A homogeneous list —
    // which is what every real call site hands over — is one run and therefore
    // one launch, unless the wave rule below cuts it.
    sycl::event ev;
    bool first = true;
    uint32_t i = 0;
    while (i < n_proj) {
        const bool f16 = projs[i].w != nullptr;
        uint32_t j = i + 1;
        while (j < n_proj && (projs[j].w != nullptr) == f16 && (j - i) < kDs4GemvMultiMax) ++j;
        j = i + wave_cut(projs + i, j - i);
        const std::vector<sycl::event> d = first ? deps : std::vector<sycl::event>{ev};
        ev = submit_run(q, prof_name, x, projs + i, j - i, K, f16, d);
        first = false;
        i = j;
    }
    return ev;
}

sycl::event ds4_q8_soa_to_f16(sycl::queue& q, const int8_t* qs, const sycl::half* d,
                              sycl::half* out, uint32_t K, uint32_t N,
                              const std::vector<sycl::event>& deps) {
    const uint32_t nb = K / 32;
    constexpr uint32_t WG = 256;
    // 2-D, one row per group-1 index, rather than a flat 1-D index divided by K.
    // The flat form spent a 64-BIT INTEGER DIVISION per element to recover the
    // row, and Xe has no hardware 64-bit divide — IGC emulates it.  At the real
    // per-card set that is 3.4e9 emulated divisions per chunk on a kernel whose
    // entire job is to move 10.45 GB.  The row now comes from the launch
    // geometry, `k` is the fast axis so both the int8 read and the fp16 write
    // stay coalesced, and the scale index is a shift.
    //
    // The value written is unchanged, bit for bit: same fp16 scale, same int8,
    // same single fp32 multiply, same one rounding to fp16.  Only the address
    // arithmetic moved.
    const uint32_t kg = (K + WG - 1) / WG * WG;
    return ie::ps(q, "ds4_q8_soa_to_f16", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>(N, kg), sycl::range<2>(1, WG)),
                       [=](sycl::nd_item<2> it) {
            const uint32_t n = uint32_t(it.get_global_id(0));
            const uint32_t k = uint32_t(it.get_global_id(1));
            if (k >= K) return;
            const uint64_t i = uint64_t(n) * K + k;
            out[i] = sycl::half(float(d[uint64_t(n) * nb + (k >> 5)]) * float(qs[i]));
        });
    });
}

}  // namespace ie
