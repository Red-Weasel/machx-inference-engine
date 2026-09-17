// src/ops/hadamard.cpp — 128-point Walsh–Hadamard transform on activations.
//
// EXL3's incoherence step (QuaRot-style): weights are stored in a Hadamard-
// rotated basis, so at inference the *activation* is rotated instead. Fixed
// 128-point Sylvester (natural-order) Hadamard, normalized by 1/sqrt(128)
// (hadamard.cu:107, exl3_format_notes.md §7). Applied block-wise: the input of
// length `n` (a multiple of 128) is split into n/128 contiguous blocks and each
// is transformed independently.
//
// Matches make_oracle.py `sylvester_norm(128) @ block`: the radix-2 butterfly
// below computes the unnormalized Sylvester H·v exactly (verified N=2,4 against
// the np.block construction), then scales by 1/sqrt(128). H is symmetric and
// orthogonal under this norm, so applying twice is the identity (the round-trip
// validation).

#include "ie/ops.hpp"

#include <sycl/sycl.hpp>
#include "ie/kernel_profiler.hpp"

namespace ie {

sycl::event hadamard_transform(sycl::queue& q,
                               const sycl::half* x, sycl::half* y,
                               uint32_t n,
                               const sycl::half* pre_scale,
                               const sycl::half* post_scale,
                               const std::vector<sycl::event>& deps) {
    constexpr int H = 128;
    const float norm = 0.088388347648f;          // 1/sqrt(128)
    const uint32_t n_blocks = n / H;
    // EXL3 forward folds the per-feature scales in: pre = suh (applied to the
    // input BEFORE the transform → had128(x⊙suh)); post = svh (applied AFTER →
    // had128(acc)⊙svh). Both null → a plain orthogonal WHT (the validation path).

    return ie::ps(q, "hadamard128", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<float, 1> s(sycl::range<1>(H), h);

        h.parallel_for(sycl::nd_range<1>(uint64_t(n_blocks) * H, H),
                       [=](sycl::nd_item<1> it) {
            const uint32_t lid = uint32_t(it.get_local_id(0));   // 0..127
            const uint32_t blk = uint32_t(it.get_group(0));
            const uint64_t gidx = uint64_t(blk) * H + lid;

            float v = float(x[gidx]);
            if (pre_scale) v *= float(pre_scale[gidx]);
            s[lid] = v;
            sycl::group_barrier(it.get_group());

            // 7 radix-2 stages, natural (Sylvester) order. Pair partner = lid^len.
            // new[lo]=old[lo]+old[hi]; new[hi]=old[lo]-old[hi].
            for (uint32_t len = 1; len < H; len <<= 1) {
                const float a = s[lid];
                const float b = s[lid ^ len];
                sycl::group_barrier(it.get_group());   // all reads before any write
                s[lid] = (lid & len) ? (b - a) : (a + b);
                sycl::group_barrier(it.get_group());
            }

            float o = s[lid] * norm;
            if (post_scale) o *= float(post_scale[gidx]);
            y[gidx] = sycl::half(o);
        });
    });
}

// Row-batched variant for the fused EXL3 MoE: x/y are [R, n] row-major; row r
// uses per-expert scales from the suh/svh banks at expert row_expert[r]. The
// transform itself is identical (one WG per 128-block); the only change is the
// scale index (e*n + within-row offset) and the row-strided data index.
sycl::event hadamard_transform_moe(sycl::queue& q,
                                   const sycl::half* x, sycl::half* y,
                                   uint32_t n, uint32_t R,
                                   const int32_t* row_expert,
                                   const sycl::half* pre_base,
                                   const sycl::half* post_base,
                                   const std::vector<sycl::event>& deps) {
    constexpr int H = 128;
    const float norm = 0.088388347648f;          // 1/sqrt(128)
    const uint32_t bpr = n / H;                   // 128-blocks per row
    const uint64_t total_blocks = uint64_t(R) * bpr;

    return ie::ps(q, "hadamard128_moe", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<float, 1> s(sycl::range<1>(H), h);

        h.parallel_for(sycl::nd_range<1>(total_blocks * H, H),
                       [=](sycl::nd_item<1> it) {
            const uint32_t lid = uint32_t(it.get_local_id(0));   // 0..127
            const uint64_t blk = uint64_t(it.get_group(0));
            const uint32_t r   = uint32_t(blk / bpr);            // active row
            const uint32_t bir = uint32_t(blk % bpr);            // block within row
            const uint64_t off  = uint64_t(bir) * H + lid;       // within-row offset
            const uint64_t gidx = uint64_t(r) * n + off;         // data index ([R,n])
            const uint64_t sidx = uint64_t(row_expert[r]) * n + off;  // scale bank index

            float v = float(x[gidx]);
            if (pre_base) v *= float(pre_base[sidx]);
            s[lid] = v;
            sycl::group_barrier(it.get_group());

            for (uint32_t len = 1; len < H; len <<= 1) {
                const float a = s[lid];
                const float b = s[lid ^ len];
                sycl::group_barrier(it.get_group());
                s[lid] = (lid & len) ? (b - a) : (a + b);
                sycl::group_barrier(it.get_group());
            }

            float o = s[lid] * norm;
            if (post_base) o *= float(post_base[sidx]);
            y[gidx] = sycl::half(o);
        });
    });
}

}  // namespace ie
