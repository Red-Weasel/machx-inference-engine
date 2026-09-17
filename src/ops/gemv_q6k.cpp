// src/ops/gemv_q6k.cpp — W6A16 GEMV with Q6_K weights.
//
// Mirrors `gemv_q4_K`: 16 SGs/WG, each SG produces one output column over the
// full K dimension; 16 lanes split the K direction into 16-element chunks.
//
// Q6_K layout (research/03 §2.4):
//   ql[128]   low 4 bits of each value
//   qh[64]    high 2 bits, two values per byte
//   scales[16] int8  per-16-element signed scales
//   d   fp16  super-block scale (no dmin — values are signed, offset 32)
//
// Per super-block, 256 outputs split into 8 quadrants (half × sub):
//   half = 0 → outputs 0..127, ql[0..63], qh[0..31], scales[0..7]
//   half = 1 → outputs 128..255, ql[64..127], qh[32..63], scales[8..15]
//
// Within a half, 4 quadrants of 32 outputs each, indexed by `sub` ∈ {0..3}:
//   sub=0 (q1): ql[l]      &0xF | (qh[l]>>0)&3<<4 - 32, scale[is+0]   l in 0..31
//   sub=1 (q2): ql[l+32]   &0xF | (qh[l]>>2)&3<<4 - 32, scale[is+2]
//   sub=2 (q3): ql[l]      >>4   | (qh[l]>>4)&3<<4 - 32, scale[is+4]
//   sub=3 (q4): ql[l+32]   >>4   | (qh[l]>>6)&3<<4 - 32, scale[is+6]
//   where is = l / 16.

#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"

#include <cstdlib>
#include "ie/dp4a.hpp"
#include <sycl/sycl.hpp>
#include "ie/kernel_profiler.hpp"

namespace ie {

namespace {

inline float dev_fp16_to_fp32(uint16_t h) {
    const uint32_t s = uint32_t(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1fu;
    uint32_t m =  h        & 0x3ffu;
    if (e == 0) {
        if (m == 0) return sycl::bit_cast<float>(s);
        while ((m & 0x400u) == 0) { m <<= 1; e -= 1; }
        e += 1;
        m &= ~0x400u;
    } else if (e == 31) {
        const uint32_t r = s | 0x7f800000u | (m << 13);
        return sycl::bit_cast<float>(r);
    }
    e += (127 - 15);
    const uint32_t r = s | (e << 23) | (m << 13);
    return sycl::bit_cast<float>(r);
}

}  // namespace

// Body of gemv_q6_K parameterized on N_PER_WG so we can dispatch a small-N
// path (good occupancy at low N) and a large-N path (better A-SLM amortization
// at lm_head shapes where N > 4k).
//
// USE_SLM_T = true (default): cooperative A load into SLM, single read pattern
//   per super-block compute step.  Wins when many lanes within the WG re-read
//   the same A bytes (small N_PER_WG → A bandwidth amortized across columns).
// USE_SLM_T = false (Experiment A): read A directly from global with cache.
//   Removes the cooperative load, the barrier, and the SLM allocation.  When
//   each Xe-core has many WGs, all 2374 lm_head WGs re-stage the same 4 KB of
//   A into SLM redundantly; the L1/L2 path serves repeat reads of A for free
//   and the barrier overhead disappears.
//
// USE_SG_DEQUANT_T = true: per-SG cooperative block read of ql/qh/scales for
//   each super-block, redistributed via sub_group shuffles.  Replaces the 16
//   strided per-lane byte loads of ql + 16 of qh per inner iteration (uncoalesced
//   pattern: lanes have stride-16 offsets) with three coalesced sub-group reads
//   (32 uint32 of ql + 16 uint32 of qh + 16 byte scales).  Lane mapping in this
//   kernel has scale_off==lane (1:1 — no shuffle for scales), 8 unique ql_offs,
//   and 4 unique qh_offs; since `n` is uniform across an SG, the early-exit
//   path doesn't break SG-collective semantics.
template <int N_PER_WG_T, bool USE_SLM_T = true, bool USE_SG_DEQUANT_T = false>
static sycl::event gemv_q6_K_impl(sycl::queue& q,
                                  const sycl::half* A, const block_q6_K* W,
                                  sycl::half* y,
                                  uint32_t K, uint32_t N,
                                  const char* prof_name,
                                  const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE  = 16;
    constexpr int N_PER_WG = N_PER_WG_T;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;

    const uint32_t blocks_per_col = K / 256;
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, prof_name, [&](sycl::handler& h) {
        h.depends_on(deps);
        // Allocate a 1-element SLM placeholder when bypassing — sycl requires
        // the accessor object to exist if referenced; we just don't read from it.
        sycl::local_accessor<sycl::half, 1> A_slm(
            sycl::range<1>(USE_SLM_T ? K : 1), h);

        h.parallel_for(sycl::nd_range<1>(n_wgs * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid    = uint32_t(it.get_local_id(0));
            const uint32_t wgid   = uint32_t(it.get_group(0));
            const uint32_t sg_id  = lid / SG_SIZE;
            const uint32_t lane   = lid % SG_SIZE;
            const uint32_t n      = wgid * N_PER_WG + sg_id;
            auto sg = it.get_sub_group();

            if constexpr (USE_SLM_T) {
                // Cooperative A load.
                for (uint32_t i = lid; i < K; i += WG_ITEMS) A_slm[i] = A[i];
                sycl::group_barrier(it.get_group());
            }
            if (n >= N) return;

            // Lane → (half, sub, l_half).
            const int half   = int(lane) >> 3;             // 0 or 1
            const int sub    = (int(lane) >> 1) & 0x3;     // 0..3
            const int l_half = int(lane) & 0x1;            // 0 or 1
            const int l_start = l_half * 16;               // 0 or 16
            const int ql_off    = half * 64 + (sub & 1) * 32 + l_start;
            const int qh_off    = half * 32 + l_start;
            const int scale_off = half * 8  + sub * 2  + l_half;
            const int qh_shift  = sub * 2;                 // 0, 2, 4, 6
            const bool high_nibble = (sub & 2) != 0;       // sub ∈ {2,3}
            const int out_off = half * 128 + sub * 32 + l_start;
            const int ql_shift = high_nibble ? 4 : 0;

            // Per-lane shuffle topology — constant across super-blocks, hoist out.
            // ql_off ∈ {0,16,32,48,64,80,96,112}; ql_off/4 ∈ {0,4,8,12,16,20,24,28}.
            // 4 contiguous uint32s starting at word_idx, all on the same side
            // of the 16-word lo/hi boundary.
            const int ql_word_base = ql_off >> 2;
            const bool ql_from_hi  = ql_word_base >= 16;
            const int ql_lane_base = ql_word_base & 0xF;   // 0,4,8,12
            // qh_off ∈ {0,16,32,48}; qh_off/4 ∈ {0,4,8,12}.
            const int qh_lane_base = qh_off >> 2;          // 0,4,8,12

            float acc = 0.f;
            const block_q6_K* col_blocks = &W[n * blocks_per_col];

            for (uint32_t b = 0; b < blocks_per_col; ++b) {
                const block_q6_K& blk = col_blocks[b];
                // Pick A source — SLM (default) or global with L1/L2 cache.
                const sycl::half* a_chunk =
                    USE_SLM_T ? &A_slm[b * 256 + out_off]
                              : &A    [b * 256 + out_off];

                if constexpr (USE_SG_DEQUANT_T) {
                    // ── Sub-group cooperative dequant ─────────────────────────
                    // Coalesced load: lane k reads uint32 #k (and k+16) of ql,
                    // uint32 #k of qh, and byte #k of scales. The compiler/L1
                    // sees 16 contiguous lanes reading stride-4 (or stride-1
                    // for scales) and emits a single block transaction.
                    const uint32_t* ql_w = reinterpret_cast<const uint32_t*>(blk.ql);
                    const uint32_t* qh_w = reinterpret_cast<const uint32_t*>(blk.qh);
                    const uint32_t my_ql_lo = ql_w[lane];
                    const uint32_t my_ql_hi = ql_w[lane + 16];
                    const uint32_t my_qh    = qh_w[lane];
                    // scale_off == lane in this lane-mapping, so the cooperative
                    // scales[lane] read already gives each lane its own scale —
                    // no shuffle needed.
                    const int8_t my_scale = blk.scales[lane];
                    const float d = dev_fp16_to_fp32(blk.d) * float(my_scale);
                    const float bias = 32.0f * d;

                    // Redistribute: each lane needs 4 uint32s (16 bytes) of ql
                    // and 4 uint32s of qh starting at its ql_off/qh_off.
                    // NOTE: select_from_group(sg, x, src) returns the SOURCE
                    // lane's value of x. Since lanes pre-pick lo/hi differently,
                    // we must shuffle both lo and hi independently and choose
                    // per-receiver — otherwise lane 8 (wants hi from lane 0)
                    // ends up reading lane 0's lo. Costs 2× the shuffles but
                    // eliminates the cross-lane semantic mismatch.
                    uint32_t my_ql4[4], my_qh4[4];
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const sycl::id<1> ql_src(ql_lane_base + k);
                        const uint32_t lo_w = sycl::select_from_group(sg, my_ql_lo, ql_src);
                        const uint32_t hi_w = sycl::select_from_group(sg, my_ql_hi, ql_src);
                        my_ql4[k] = ql_from_hi ? hi_w : lo_w;
                        my_qh4[k] = sycl::select_from_group(
                            sg, my_qh, sycl::id<1>(qh_lane_base + k));
                    }

                    // Dequant from registers.
                    #pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        const int word_idx = i >> 2;
                        const int sh       = (i & 3) << 3;
                        const uint8_t ql_b = uint8_t(my_ql4[word_idx] >> sh);
                        const uint8_t qh_b = uint8_t(my_qh4[word_idx] >> sh);
                        const uint8_t lo_hi = uint8_t((ql_b >> ql_shift) & 0x0F);
                        const uint8_t hi    = uint8_t((qh_b >> qh_shift) & 0x3);
                        const float w_unsigned = float(int(lo_hi | (hi << 4)));
                        acc += float(a_chunk[i]) * (d * w_unsigned - bias);
                    }
                } else {
                    // ── Original per-lane scalar byte loads ───────────────────
                    const float d    = dev_fp16_to_fp32(blk.d) * float(blk.scales[scale_off]);
                    const float bias = 32.0f * d;

                    #pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        const uint8_t ql_b = blk.ql[ql_off + i];
                        const uint8_t qh_b = blk.qh[qh_off + i];
                        const uint8_t lo_hi = uint8_t((ql_b >> ql_shift) & 0x0F);
                        const uint8_t hi   = uint8_t((qh_b >> qh_shift) & 0x3);
                        const float w_unsigned = float(int(lo_hi | (hi << 4)));
                        acc += float(a_chunk[i]) * (d * w_unsigned - bias);
                    }
                }
            }

            acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
            if (lane == 0) y[n] = sycl::half(acc);
        });
    });
}

// ===========================================================================
// gemv_q6_K_slm — SLM-slab-staged GEMV for huge-N shapes (lm_head).
// ===========================================================================
// 2026-06-09 (v1.1 decode work): the SG-cooperative-dequant variant above is
// ALU-bound — 12 sub_group shuffles per block per SG dominate.  This variant
// removes shuffles entirely: a WG's 16 columns × blocks_per_col Q6_K blocks
// form ONE contiguous slab (16 × bpc × 210 B; 26.9 KiB at K=2048, always
// 16 B-divisible), staged into SLM once with coalesced vec4 loads.  Lanes
// then dequant with cheap SLM byte reads.  Also applies the same algebraic
// fold as gemv_q4_K: w = d·(q−32) ⇒ acc += d·(Σa·q − 32·Σa), removing a
// multiply from the inner loop.  NOT bit-identical to the old variant
// (different FP fold/order) — gated on PPL like every quant-kernel change.
// A is read from global (L1-served; same as the USE_SLM=false finding).
// Kill switch: IE_NO_Q6K_SLM=1 falls back to the SG-dequant variant.
static sycl::event gemv_q6_K_slm(sycl::queue& q,
                                 const sycl::half* A, const block_q6_K* W,
                                 sycl::half* y,
                                 uint32_t K, uint32_t N,
                                 const char* prof_name,
                                 const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE  = 16;
    constexpr int N_PER_WG = 16;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;     // 256
    const uint32_t blocks_per_col = K / 256;
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;
    const uint32_t slab_bytes =
        N_PER_WG * blocks_per_col * uint32_t(sizeof(block_q6_K));

    return ie::ps(q, prof_name, [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<uint8_t, 1> W_slm(slab_bytes, h);

        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t wgid  = uint32_t(it.get_group(0));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t n0    = wgid * N_PER_WG;
            const uint32_t n     = n0 + sg_id;

            // Stage the WG's column slab (clamped for a partial last WG).
            {
                const uint32_t cols = sycl::min(uint32_t(N_PER_WG), N - n0);
                const uint32_t n_bytes =
                    cols * blocks_per_col * uint32_t(sizeof(block_q6_K));
                const uint8_t* src_b = reinterpret_cast<const uint8_t*>(W) +
                    uint64_t(n0) * blocks_per_col * sizeof(block_q6_K);
                uint8_t* dst_b =
                    W_slm.get_multi_ptr<sycl::access::decorated::no>().get();
                const auto* src = reinterpret_cast<const sycl::vec<uint32_t, 4>*>(src_b);
                auto* dst = reinterpret_cast<sycl::vec<uint32_t, 4>*>(dst_b);
                const uint32_t n_vec = n_bytes / 16;
                for (uint32_t i = lid; i < n_vec; i += WG_ITEMS) dst[i] = src[i];
                for (uint32_t i = n_vec * 16 + lid; i < n_bytes; i += WG_ITEMS)
                    dst_b[i] = src_b[i];
            }
            sycl::group_barrier(it.get_group());
            if (n >= N) return;

            // Lane → (half, sub, l_half) — same lattice as gemv_q6_K_impl.
            const int half   = int(lane) >> 3;
            const int sub    = (int(lane) >> 1) & 0x3;
            const int l_half = int(lane) & 0x1;
            const int l_start = l_half * 16;
            const int ql_off    = half * 64 + (sub & 1) * 32 + l_start;
            const int qh_off    = half * 32 + l_start;
            const int scale_off = half * 8  + sub * 2  + l_half;
            const int qh_shift  = sub * 2;
            const bool high_nibble = (sub & 2) != 0;
            const int out_off = half * 128 + sub * 32 + l_start;
            const int ql_shift = high_nibble ? 4 : 0;

            float acc = 0.f;
            const uint8_t* col_base =
                W_slm.get_multi_ptr<sycl::access::decorated::no>().get() +
                uint64_t(sg_id) * blocks_per_col * sizeof(block_q6_K);

            for (uint32_t b = 0; b < blocks_per_col; ++b) {
                const auto& blk = *reinterpret_cast<const block_q6_K*>(
                    col_base + uint64_t(b) * sizeof(block_q6_K));
                const sycl::half* a_chunk = &A[b * 256 + out_off];
                const float d =
                    dev_fp16_to_fp32(blk.d) * float(blk.scales[scale_off]);

                float sum_aq = 0.f, sum_a = 0.f;
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const uint8_t ql_b = blk.ql[ql_off + i];
                    const uint8_t qh_b = blk.qh[qh_off + i];
                    const int qu = int((ql_b >> ql_shift) & 0x0F) |
                                   (int((qh_b >> qh_shift) & 0x3) << 4);
                    const float a = float(a_chunk[i]);
                    sum_aq += a * float(qu);
                    sum_a  += a;
                }
                acc += d * (sum_aq - 32.0f * sum_a);
            }

            acc = sycl::reduce_over_group(it.get_sub_group(), acc,
                                          sycl::plus<float>());
            if (lane == 0) y[n] = sycl::half(acc);
        });
    });
}

// ===========================================================================
// gemv_q6_K_slm_q8 — integer-dot (dp4a) variant of the SLM-slab-staged GEMV.
// ===========================================================================
// v1.5-D (2026-06-10): the staged fp16 kernel runs the lm_head at ~248 GB/s
// effective — ALU-bound on the 6-bit decode + fp chain (bandwidth floor for
// its 417 MB is ~1.0 ms; it takes 1.68).  With Q8_1 activations the inner
// loop becomes: build four u8x4 packs of q6u = ql|qh<<4 (no −32), dp4a
// against the staged q8 words, and fold per-16:
//   acc += d6·sc · d8 · (Σ q6u·q8 − 32·Σ q8)
// The −32 rides the ones-dp4a sum; sc is Q6_K's per-16 scale = exactly one
// lane's slice.  Weights stream once via the coalesced slab stage as before.
sycl::event gemv_q6_K_slm_q8(sycl::queue& q,
                             const void* x_q8, const block_q6_K* W,
                             sycl::half* y,
                             uint32_t K, uint32_t N,
                             const char* prof_name,
                             const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE  = 16;
    constexpr int N_PER_WG = 16;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;     // 256
    const uint32_t blocks_per_col = K / 256;
    const uint32_t q8_blocks      = K / 32;
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;
    const uint32_t slab_bytes =
        N_PER_WG * blocks_per_col * uint32_t(sizeof(block_q6_K));
    const auto* X8 = static_cast<const block_q8_1x*>(x_q8);

    return ie::ps(q, prof_name, [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<uint8_t, 1>  W_slm(slab_bytes, h);
        sycl::local_accessor<uint32_t, 1> q8s(q8_blocks * 8, h);
        sycl::local_accessor<float, 1>    q8d(q8_blocks, h);

        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t wgid  = uint32_t(it.get_group(0));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t n0    = wgid * N_PER_WG;
            const uint32_t n     = n0 + sg_id;

            // Stage the WG's weight slab (as in gemv_q6_K_slm).
            {
                const uint32_t cols = sycl::min(uint32_t(N_PER_WG), N - n0);
                const uint32_t n_bytes =
                    cols * blocks_per_col * uint32_t(sizeof(block_q6_K));
                const uint8_t* src_b = reinterpret_cast<const uint8_t*>(W) +
                    uint64_t(n0) * blocks_per_col * sizeof(block_q6_K);
                uint8_t* dst_b =
                    W_slm.get_multi_ptr<sycl::access::decorated::no>().get();
                const auto* src = reinterpret_cast<const sycl::vec<uint32_t, 4>*>(src_b);
                auto* dst = reinterpret_cast<sycl::vec<uint32_t, 4>*>(dst_b);
                const uint32_t n_vec = n_bytes / 16;
                for (uint32_t i = lid; i < n_vec; i += WG_ITEMS) dst[i] = src[i];
                for (uint32_t i = n_vec * 16 + lid; i < n_bytes; i += WG_ITEMS)
                    dst_b[i] = src_b[i];
            }
            // Stage the Q8 activation stream.
            for (uint32_t i = lid; i < q8_blocks * 8; i += WG_ITEMS) {
                const uint32_t blk = i / 8, w = i % 8;
                q8s[i] = reinterpret_cast<const uint32_t*>(X8[blk].qs)[w];
            }
            for (uint32_t i = lid; i < q8_blocks; i += WG_ITEMS) q8d[i] = X8[i].d;
            sycl::group_barrier(it.get_group());
            if (n >= N) return;

            // Lane lattice — same as gemv_q6_K_impl.
            const int half   = int(lane) >> 3;
            const int sub    = (int(lane) >> 1) & 0x3;
            const int l_half = int(lane) & 0x1;
            const int l_start = l_half * 16;
            const int ql_off    = half * 64 + (sub & 1) * 32 + l_start;
            const int qh_off    = half * 32 + l_start;
            const int scale_off = half * 8  + sub * 2  + l_half;
            const int qh_shift  = sub * 2;
            const bool high_nibble = (sub & 2) != 0;
            const int out_off = half * 128 + sub * 32 + l_start;
            const int ql_shift = high_nibble ? 4 : 0;

            float acc = 0.f;
            const uint8_t* col_base =
                W_slm.get_multi_ptr<sycl::access::decorated::no>().get() +
                uint64_t(sg_id) * blocks_per_col * sizeof(block_q6_K);

            for (uint32_t b = 0; b < blocks_per_col; ++b) {
                const auto& blk = *reinterpret_cast<const block_q6_K*>(
                    col_base + uint64_t(b) * sizeof(block_q6_K));
                const float d6 =
                    dev_fp16_to_fp32(blk.d) * float(blk.scales[scale_off]);

                // Build 4 packed-u8 words of q6u and dp4a against q8.
                const uint32_t k_base  = uint32_t(b) * 256 + uint32_t(out_off);
                const uint32_t q8_blk  = k_base / 32;
                const uint32_t q8_word = (k_base % 32) / 4;     // 0 or 4

                int32_t idot = 0, isum = 0;
                #pragma unroll
                for (int w = 0; w < 4; ++w) {
                    uint32_t pack = 0;
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int i = w * 4 + j;
                        const uint8_t ql_b = blk.ql[ql_off + i];
                        const uint8_t qh_b = blk.qh[qh_off + i];
                        const uint32_t q6u =
                            uint32_t((ql_b >> ql_shift) & 0x0F) |
                            (uint32_t((qh_b >> qh_shift) & 0x3) << 4);
                        pack |= q6u << (8 * j);
                    }
                    const int32_t q8w = int32_t(q8s[q8_blk * 8 + q8_word + w]);
                    idot = ie::dp4a_us(pack, q8w, idot);
                    isum = ie::dp4a_us(0x01010101u, q8w, isum);
                }
                acc += d6 * q8d[q8_blk] * (float(idot) - 32.0f * float(isum));
            }

            acc = sycl::reduce_over_group(it.get_sub_group(), acc,
                                          sycl::plus<float>());
            if (lane == 0) y[n] = sycl::half(acc);
        });
    });
}

// ===========================================================================
// gemv_q6_K_slm_q8_ktiled — int-dot Q6_K GEMV (dense ffn_down, big K).
// ===========================================================================
// P3b Task 1 (2026-06-10): the lm_head int-dot variant (gemv_q6_K_slm_q8)
// stages the WHOLE column's WEIGHT slab (16 cols × K/256 × 210 B) + the q8
// stream in SLM. At K=12288 (qwen3-8b ffn_down) that slab is 175104 B > the
// 128 KiB WG SLM budget → q6k_slm_gemv_q8_fits(12288) FALSE → the dense path
// fell back to scalar gemm_q6_K at M=1 (140 GB/s, the #1 decode bottleneck).
//
// DESIGN NOTE (the staging that matters): the plan's first cut tiled the
// WEIGHT slab into SLM in K-chunks — but for ffn_down each WG owns UNIQUE
// columns and the 743 MB of weights is read exactly once, so staging weights
// global→SLM→registers is pure overhead (+ a barrier per tile). Measured: it
// REGRESSED to ~70 GB/s (slower than the 140 GB/s cliff). The fast Q4_K decode
// GEMV (gemv_q4_K_q8, 321 GB/s, same int-dot stream) does NOT stage weights:
// it stages only the ACTIVATION in SLM (reused across the WG's N_PER_WG cols)
// and reads weight blocks DIRECTLY from global (streamed once via L1/L2). This
// kernel follows that pattern for Q6_K: stage the q8 activation stream in SLM
// (K/32 blocks × 36 B = 13.8 KiB at K=12288 « 128 KiB — fits for any K), read
// Q6_K weight blocks from global in the inner loop, N_PER_WG=32 for occupancy,
// ONE barrier. Int-dot math (build q6u packs, dp4a vs q8, fold per-block) is
// identical to gemv_q6_K_slm_q8. NOT bit-identical to gemm_q6_K (dp4a vs
// scalar dequant + fp32 order) — gated on PPL.
//
// SLM footprint (independent of N_PER_WG; activation only):
//   q8 qs words : (K/32) · 8 · 4   = K·1.0 B   → 12288  B at K=12288
//   q8 d        : (K/32) · 4       = K·0.125 B →  1536  B
//   total                          ≈ 13.5 KiB  (« 128 KiB, any K up to ~93k)
static sycl::event gemv_q6_K_slm_q8_ktiled(sycl::queue& q,
                                           const void* x_q8, const block_q6_K* W,
                                           sycl::half* y,
                                           uint32_t K, uint32_t N,
                                           const char* prof_name,
                                           const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE  = 16;
    constexpr int N_PER_WG = 32;                     // occupancy: match q4_K_q8
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;     // 512

    const uint32_t blocks_per_col = K / 256;
    const uint32_t q8_blocks      = K / 32;
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;
    const auto* X8 = static_cast<const block_q8_1x*>(x_q8);

    return ie::ps(q, prof_name, [&](sycl::handler& h) {
        h.depends_on(deps);
        // Stage ONLY the q8 activation stream — weights stream from global.
        sycl::local_accessor<uint32_t, 1> q8s(q8_blocks * 8, h);
        sycl::local_accessor<float, 1>    q8d(q8_blocks, h);

        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t wgid  = uint32_t(it.get_group(0));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t n     = wgid * N_PER_WG + sg_id;

            // Cooperative q8-activation load (once per WG, shared across cols).
            for (uint32_t i = lid; i < q8_blocks * 8; i += WG_ITEMS) {
                const uint32_t blk = i / 8, w = i % 8;
                q8s[i] = reinterpret_cast<const uint32_t*>(X8[blk].qs)[w];
            }
            for (uint32_t i = lid; i < q8_blocks; i += WG_ITEMS) q8d[i] = X8[i].d;
            sycl::group_barrier(it.get_group());
            if (n >= N) return;

            // Lane lattice — same as gemv_q6_K_impl / gemv_q6_K_slm_q8.
            const int half   = int(lane) >> 3;
            const int sub    = (int(lane) >> 1) & 0x3;
            const int l_half = int(lane) & 0x1;
            const int l_start = l_half * 16;
            const int ql_off    = half * 64 + (sub & 1) * 32 + l_start;
            const int qh_off    = half * 32 + l_start;
            const int scale_off = half * 8  + sub * 2  + l_half;
            const int qh_shift  = sub * 2;
            const bool high_nibble = (sub & 2) != 0;
            const int out_off = half * 128 + sub * 32 + l_start;
            const int ql_shift = high_nibble ? 4 : 0;

            float acc = 0.f;
            const block_q6_K* col_blocks = &W[uint64_t(n) * blocks_per_col];

            for (uint32_t b = 0; b < blocks_per_col; ++b) {
                const block_q6_K& blk = col_blocks[b];   // global, L1/L2 stream
                const float d6 =
                    dev_fp16_to_fp32(blk.d) * float(blk.scales[scale_off]);

                const uint32_t k_base  = b * 256 + uint32_t(out_off);
                const uint32_t q8_blk  = k_base / 32;
                const uint32_t q8_word = (k_base % 32) / 4;   // 0 or 4

                int32_t idot = 0, isum = 0;
                #pragma unroll
                for (int w = 0; w < 4; ++w) {
                    uint32_t pack = 0;
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int i = w * 4 + j;
                        const uint8_t ql_b = blk.ql[ql_off + i];
                        const uint8_t qh_b = blk.qh[qh_off + i];
                        const uint32_t q6u =
                            uint32_t((ql_b >> ql_shift) & 0x0F) |
                            (uint32_t((qh_b >> qh_shift) & 0x3) << 4);
                        pack |= q6u << (8 * j);
                    }
                    const int32_t q8w = int32_t(q8s[q8_blk * 8 + q8_word + w]);
                    idot = ie::dp4a_us(pack, q8w, idot);
                    isum = ie::dp4a_us(0x01010101u, q8w, isum);
                }
                acc += d6 * q8d[q8_blk] * (float(idot) - 32.0f * float(isum));
            }

            acc = sycl::reduce_over_group(it.get_sub_group(), acc,
                                          sycl::plus<float>());
            if (lane == 0) y[n] = sycl::half(acc);
        });
    });
}

// Public entry for the int-dot Q6_K decode GEMV (dense ffn_down, big-K Q6_K).
// Stages only the q8 activation in SLM (≈13.5 KiB at K=12288), so it is usable
// at K=12288 where gemv_q6_K_q8's whole-column WEIGHT slab overflows SLM.
sycl::event gemv_q6_K_q8_ktiled(sycl::queue& q,
                                const void* x_q8, const void* W_packed,
                                sycl::half* y,
                                uint32_t K, uint32_t N,
                                const std::vector<sycl::event>& deps) {
    return gemv_q6_K_slm_q8_ktiled(q, x_q8,
                                   static_cast<const block_q6_K*>(W_packed),
                                   y, K, N, "gemv_q6k_ktiled_q8", deps);
}

// Public entry for the int-dot staged GEMV (lm_head shape).
sycl::event gemv_q6_K_q8(sycl::queue& q,
                         const void* x_q8, const void* W_packed,
                         sycl::half* y,
                         uint32_t K, uint32_t N,
                         const std::vector<sycl::event>& deps) {
    return gemv_q6_K_slm_q8(q, x_q8,
                            static_cast<const block_q6_K*>(W_packed),
                            y, K, N, "gemv_q6k_huge_q8", deps);
}

sycl::event gemv_q6_K(sycl::queue& q,
                      const sycl::half* A, const void* W_packed,
                      sycl::half* y,
                      uint32_t K, uint32_t N,
                      const std::vector<sycl::event>& deps) {
    const auto* W = static_cast<const block_q6_K*>(W_packed);
    // Q6_K decode shapes:
    //   attn_v        K=2048  N=512    → small  (32 WGs at N_PER_WG=16)
    //   ssm_out       K=4096  N=2048   → mid    (64 WGs at N_PER_WG=32)
    //   lm_head       K=2048  N=152k   → huge   (2374 WGs at N_PER_WG=64)
    // Wider WG amortizes the per-WG A-SLM load over more output columns and
    // cuts launch grid count proportionally.
    // Per-shape profile names so the kernel monitor / bench breaks q6_K time
    // out by call shape (attn_v vs ssm_out vs lm_head). Single-line revert:
    // collapse all three back to "gemv_q6k".
    // Per-shape profile names so the kernel monitor / bench breaks q6_K time
    // out by call shape (attn_v vs ssm_out vs lm_head). Single-line revert:
    // collapse all three back to "gemv_q6k".
    if (N >= 32768) {
        // lm_head (N=248320). 2026-06-09: SLM-slab-staged variant (no
        // shuffles, algebraic fold) replaces the SG-cooperative-dequant
        // Experiment SG-1 variant, which remains the IE_NO_Q6K_SLM fallback.
        static const bool no_slm_stage = std::getenv("IE_NO_Q6K_SLM") != nullptr;
        if (!no_slm_stage)
            return gemv_q6_K_slm(q, A, W, y, K, N, "gemv_q6k_huge", deps);
        return gemv_q6_K_impl<64, /*USE_SLM=*/false, /*USE_SG_DEQUANT=*/true>(
            q, A, W, y, K, N, "gemv_q6k_huge", deps);
    }
    if (N >= 2048) {
        // ssm_out. 2026-06-09: route through the SLM-slab-staged variant
        // (same win as lm_head); IE_NO_Q6K_SLM falls back to SG-dequant.
        static const bool no_slm_stage_med = std::getenv("IE_NO_Q6K_SLM") != nullptr;
        if (!no_slm_stage_med)
            return gemv_q6_K_slm(q, A, W, y, K, N, "gemv_q6k_med", deps);
        return gemv_q6_K_impl<32, /*USE_SLM=*/false, /*USE_SG_DEQUANT=*/true>(
            q, A, W, y, K, N, "gemv_q6k_med", deps);
    }
    // attn_v (K=2048, N=512). Tried SLM bypass here too — flat within noise
    // (median 38.32 vs 38.31 over 5 prompts), so keep the SLM path: with only
    // 32 WGs total, SLM amortization across N_PER_WG=16 columns matters more
    // than at the lm_head/ssm_out shapes.
    return gemv_q6_K_impl<16>(q, A, W, y, K, N, "gemv_q6k_small", deps);
}

// =====================================================================
// gemm_q6_K — multi-row GEMM with Q6_K weights, M ≤ M_TILE per launch.
// =====================================================================
// y[M, N] = A[M, K] @ W[K, N]  (W = Q6_K-packed, same layout as gemv_q6_K)
//
// Mirrors gemm_q4_K's structure exactly:
//   - WG layout: 16 SGs × 16 lanes = 256 lanes (N_PER_WG=16)
//   - Each SG produces ONE output column for ALL M rows
//   - Per-lane state: M floats (one accumulator per row)
//   - SLM tile: M × K halfs (M_TILE=8 → 32 KiB at K=2048, 64 KiB at K=4096)
//   - Caller chunks T into M_TILE-sized passes (gemv_q_T does this)
//
// Q6_K dequant uses the same lane mapping as gemv_q6_K:
//   half = lane >> 3, sub = (lane >> 1) & 0x3, l_half = lane & 0x1
//   ql_off / qh_off / scale_off / qh_shift / out_off / ql_shift identical
//
// Why this matters for prefill: pre-this-kernel, gemv_q_T fell back to a
// per-row gemv_q6_K loop for T>1 — so 30 ssm_out + 14 attn_qkv DN + 6
// attn_v Q6_K weights all paid the full weight read T times.  With this
// kernel the weight read is amortized M× (M_TILE=8 → 8× fewer weight reads).
sycl::event gemm_q6_K(sycl::queue& q,
                      const sycl::half* A, const void* W_packed,
                      sycl::half* y,
                      uint32_t M, uint32_t K, uint32_t N,
                      const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE   = 16;
    constexpr int N_PER_WG  = 16;
    constexpr int WG_ITEMS  = N_PER_WG * SG_SIZE;
    constexpr int M_TILE    = 8;
    if (M == 0 || M > M_TILE) return {};

    const auto* W = static_cast<const block_q6_K*>(W_packed);
    const uint32_t blocks_per_col = K / 256;
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "gemm_q6k", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<sycl::half, 1> A_slm(uint64_t(M) * K, h);

        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid    = uint32_t(it.get_local_id(0));
            const uint32_t wgid   = uint32_t(it.get_group(0));
            const uint32_t sg_id  = lid / SG_SIZE;
            const uint32_t lane   = lid % SG_SIZE;
            const uint32_t n      = wgid * N_PER_WG + sg_id;

            // Cooperative A load: M*K halfs.
            const uint64_t a_total = uint64_t(M) * K;
            for (uint64_t i = lid; i < a_total; i += WG_ITEMS) A_slm[i] = A[i];
            sycl::group_barrier(it.get_group());

            if (n >= N) return;

            // Q6_K lane mapping — same as gemv_q6_K_impl.
            const int half     = int(lane) >> 3;
            const int sub      = (int(lane) >> 1) & 0x3;
            const int l_half   = int(lane) & 0x1;
            const int l_start  = l_half * 16;
            const int ql_off   = half * 64 + (sub & 1) * 32 + l_start;
            const int qh_off   = half * 32 + l_start;
            const int scale_off = half * 8 + sub * 2 + l_half;
            const int qh_shift = sub * 2;
            const bool high_nibble = (sub & 2) != 0;
            const int out_off  = half * 128 + sub * 32 + l_start;
            const int ql_shift = high_nibble ? 4 : 0;

            float acc[M_TILE];
            #pragma unroll
            for (int mm = 0; mm < M_TILE; ++mm) acc[mm] = 0.f;

            const block_q6_K* col_blocks = &W[uint64_t(n) * blocks_per_col];

            for (uint32_t b = 0; b < blocks_per_col; ++b) {
                const block_q6_K& blk = col_blocks[b];
                const float d    = dev_fp16_to_fp32(blk.d) * float(blk.scales[scale_off]);
                const float bias = 32.0f * d;

                // Dequant 16 weights this lane is responsible for, then
                // multiply by every M row's A.  Same amortization win:
                // one Q6_K weight read serves M rows.
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const uint8_t ql_b  = blk.ql[ql_off + i];
                    const uint8_t qh_b  = blk.qh[qh_off + i];
                    const uint8_t lo_hi = uint8_t((ql_b >> ql_shift) & 0x0F);
                    const uint8_t hi    = uint8_t((qh_b >> qh_shift) & 0x3);
                    const float w_unsigned = float(int(lo_hi | (hi << 4)));
                    const float w_val = d * w_unsigned - bias;

                    #pragma unroll
                    for (int mm = 0; mm < M_TILE; ++mm) {
                        if (uint32_t(mm) < M) {
                            const sycl::half a_h =
                                A_slm[uint64_t(mm) * K + uint64_t(b) * 256 + out_off + i];
                            acc[mm] += float(a_h) * w_val;
                        }
                    }
                }
            }

            // SG-reduce one row at a time and write.
            #pragma unroll
            for (int mm = 0; mm < M_TILE; ++mm) {
                if (uint32_t(mm) < M) {
                    const float r = sycl::reduce_over_group(it.get_sub_group(),
                                                            acc[mm],
                                                            sycl::plus<float>());
                    if (lane == 0) y[uint64_t(mm) * N + n] = sycl::half(r);
                }
            }
        });
    });
}

}  // namespace ie
