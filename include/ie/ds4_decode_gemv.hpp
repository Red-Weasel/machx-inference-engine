// include/ie/ds4_decode_gemv.hpp — M=1 GEMV for the DeepSeek-V4 fp16 dense
// projections.  Decode only.
//
// WHY THIS EXISTS.  `dense_w`/`grouped_w` route the fp16 dense projections
// through oneDNN at every T.  That was justified by a PREFILL measurement and
// never validated at T=1.  At T=1 these are not GEMMs at all: M=1 makes them
// pure streaming reductions whose entire cost is reading the weight once.  On a
// B70 (25.17 MB L2, 590 GB/s measured streaming read) the big shapes are already
// at the memory roofline no matter who runs them, so what separates the routes
// at decode is FIXED PER-CALL COST, and the model makes 1021 of these calls per
// token across both cards.
//
// oneDNN's fp16 matmul needs an fp16 A operand, so the oneDNN route pays a
// `cast_fp32_to_fp16` launch per projection on top of the matmul.  Measured
// cache-cold at the real per-card shapes that cast is ~4.2 us and the matmul
// itself carries ~1-3 us more than a purpose-built GEMV.  This kernel reads the
// fp32 activation directly, so one launch replaces two.
//
// LAYOUT — the one thing that must not be got wrong.  The weight is fp16
// [N, K] ROW-MAJOR: row n starts at w + n*K.  That is the layout `dense_f16`
// (src/model/deepseek4.cpp) reads and the layout `gemm_nt_f16_onednn` addresses
// with strides {1, K}; the buffer is shared with both, plus the packed-dtype
// branch.  It is NOT the [K, N] layout `gemv_fp16` (src/ops/gemv_fp16.cpp) uses
// for every other model in this repo — that path's loader transposes at upload
// and this one does not.  Nothing here transposes anything.
//
// ACCURACY.  The activation stays fp32 and accumulation is fp32, so this route
// is strictly more accurate than the oneDNN route it replaces, which rounds the
// activation to fp16 (relative error up to 2^-11) before multiplying.
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>
#include <vector>

namespace ie {

// y[n] = sum_k float(w[n*K + k]) * x[k],  n in [0, N)
//
// x: fp32 [K] device.  w: fp16 [N, K] device.  y: fp32 [N] device.
// `prof_name` is the name the kernel is pushed under for --kprofile (may be
// null).  Any K and N are accepted; K % 16 == 0 takes a 32-byte-per-lane
// vectorised path, anything else takes an equivalent scalar path.
sycl::event ds4_decode_gemv_f16(sycl::queue& q, const char* prof_name,
                                const float* x, const sycl::half* w, float* y,
                                uint32_t K, uint32_t N,
                                const std::vector<sycl::event>& deps = {});

// Block-diagonal sibling — DeepseekV4GroupedLinear at T=1.  Group g reads only
// its own slice of the activation:
//   y[g*OPG + o] = sum_h float(w[(g*OPG + o)*IPG + h]) * x[g*IPG + h]
// x: fp32 [G*IPG].  w: fp16 [G*OPG, IPG].  y: fp32 [G*OPG].
sycl::event ds4_decode_gemv_grouped_f16(sycl::queue& q, const char* prof_name,
                                        const float* x, const sycl::half* w, float* y,
                                        uint32_t G, uint32_t IPG, uint32_t OPG,
                                        const std::vector<sycl::event>& deps = {});

// Number of sequential fp32 roundings on the longest path through the kernel's
// reduction tree for a given (K, N): per-lane serial fma chain, then the 4-level
// sub-group tree, then the cross-sub-group fixup.  The gate derives its error
// bound from THIS, so the bound tracks the kernel's real reduction shape (which
// depends on the internal sub-groups-per-row choice) instead of being an
// independently tunable constant.
//   |y_kernel - y_exact| <= gamma_p * sum_k |w[n,k] * x[k]|,
//   gamma_p = p*u / (1 - p*u),  u = 2^-24,  p = ds4_decode_gemv_rounding_steps(K, N)
uint32_t ds4_decode_gemv_rounding_steps(uint32_t K, uint32_t N) noexcept;

// ---------------------------------------------------------------------------
// Q8_0-SoA siblings — the SAME kernel shape over a requantised weight
// ---------------------------------------------------------------------------
//
// WHY A SECOND LAYOUT AND NOT THE GGUF ONE.  `dense_packed` in
// src/model/deepseek4.cpp already decodes a verbatim Q8_0 GGUF row, and at M=1
// it is the WRONG kernel by a factor of two: a GGUF Q8_0 block is 34 B, so the
// int8 payload starts at `34*b + 2` and is never 4-, 8- or 16-byte aligned for
// any b.  A lane can only read it one byte at a time, which is 32 load
// instructions where the fp16 kernel issues one 32 B vector load, and the
// kernel goes load-ISSUE-rate bound instead of bandwidth bound.  Measured
// cache-cold on a B70 over the eleven real per-card decode shapes, summed:
// fp16 1223 us, verbatim-GGUF Q8_0 2520 us (2.06x SLOWER on 47% FEWER bytes),
// this SoA layout 701 us.  That is the whole reason this entry point exists.
//
// LAYOUT.  Two planes, both derived from one allocation:
//   qs   int8       [N][K]      row n at qs + n*K
//   d    fp16       [N][K/32]   row n at d + n*(K/32); d[n][b] scales
//                               qs[n][32b .. 32b+31]
// Dequantised value w[n,k] = float(d[n][k/32]) * float(qs[n][k]) — the same
// value `ie::ref::dequant_q8_0_buffer` produces from a GGUF block, just with the
// two fields stored in separate planes so the int8 run is 16 B aligned.
// K MUST be a whole number of 32-element blocks; the caller checks.
//
// ACCURACY.  Two effects, and they are different in kind:
//   1. the weight is 8-bit — a REAL, lossy change, bounded per element by
//      d/2 where d = amax/127 over its block.  This dominates and
//      deepseek4_dense_gate_test bounds it explicitly against the fp32/BF16
//      original rather than against the quantised bytes.
//   2. the reduction is fp32 with a different association than the fp16 kernel:
//      32 serial fma inside a block, then one fma per block into the row
//      accumulator.  ds4_decode_gemv_q8_rounding_steps reports its length.
// The activation is never rounded — it stays fp32 all the way through.
sycl::event ds4_decode_gemv_q8(sycl::queue& q, const char* prof_name,
                               const float* x, const int8_t* qs, const sycl::half* d,
                               float* y, uint32_t K, uint32_t N,
                               const std::vector<sycl::event>& deps = {});

// Block-diagonal sibling — DeepseekV4GroupedLinear at T=1 over the same layout.
//   y[g*OPG + o] = sum_h w[(g*OPG + o), h] * x[g*IPG + h]
sycl::event ds4_decode_gemv_grouped_q8(sycl::queue& q, const char* prof_name,
                                       const float* x, const int8_t* qs, const sycl::half* d,
                                       float* y, uint32_t G, uint32_t IPG, uint32_t OPG,
                                       const std::vector<sycl::event>& deps = {});

// Sequential fp32 roundings on the longest path of the Q8 kernel: the 32-long
// serial chain inside one block, then one rounding per block the lane owns,
// then the 4-level sub-group tree and the cross-sub-group fixup.  Larger than
// the fp16 kernel's count because the in-block chain is serial by construction.
uint32_t ds4_decode_gemv_q8_rounding_steps(uint32_t K, uint32_t N) noexcept;

// Expands a Q8_0-SoA weight back to fp16 [N, K] — the layout `dense_f16`,
// `gemm_nt_f16_onednn` and `gemm_bmm_nt_f16_onednn` all read.  This is what the
// T > 1 (prefill) path uses: the oneDNN route is worth 1.54-1.70x on prefill and
// materialising the weight keeps it, at the cost of one extra pass over the
// weight per projection per chunk.  `out` must hold K*N halves.
//
// WHAT THAT PASS COSTS, AND WHAT IT DOES NOT EXPLAIN.  Over the real per-card
// call set — 3.411e9 requantised elements per chunk, derived shape by shape in
// tools/ie_ds4_dense_prefill_bench.cpp and cross-checked against the 6.92 GB/card
// dense stream src/model/deepseek4.cpp records — this kernel reads 3.63 GB and
// writes 6.82 GB, i.e. 10.45 GB, which at this card's 450-590 GB/s streaming rate
// is 18-23 ms per card per chunk.
//
// Turning IE_DS4_DENSE_Q8 on costs pp512 +5,259 ms of wall (115.88 -> 52.91
// tok/s on one 512-token chunk).  23 ms is 229x too small to be that, so THE
// MATERIALISATION IS NOT THE REGRESSION, whatever the surrounding comments used
// to assert.  The two routes that ARE the right order of magnitude are both
// silent fallbacks in `dense_w`: the scalar `dense_f16` (which re-reads the
// activation once per output ROW: 6,986 GB from L2, ~3,500 ms) and the per-token
// GEMV loop (T passes over the weight: 1,857 GB, ~4,100 ms plus 308k launches).
// Both now announce themselves on stderr, both now have distinct profiler names,
// and `ie-ds4-dense-prefill-bench` measures all of them without the model.
sycl::event ds4_q8_soa_to_f16(sycl::queue& q, const int8_t* qs, const sycl::half* d,
                              sycl::half* out, uint32_t K, uint32_t N,
                              const std::vector<sycl::event>& deps = {});

// ---------------------------------------------------------------------------
// MULTI-PROJECTION GEMV — several [N_j, K] weights against ONE activation, in
// ONE kernel launch
// ---------------------------------------------------------------------------
//
// WHY.  Decode's cost is not the arithmetic, it is the SUBMISSION.  A kernel
// submit on an in-order Level Zero queue on this box costs 1.43 us with queue
// profiling off and 1.85 us with it on, CONSTANT from 1 to 65,536 work-items,
// and one decoded token issues thousands of them.  At the attention site the
// model spells seven separate projections that all read the SAME `ws_norm_`
// activation and all contract over the same K = hidden:
//
//     q_a  kv  comp_kv  comp_gate  idx_kv  idx_gate  idx_proj
//
// Nothing couples them — they write seven disjoint outputs — so seven launches
// is seven times the fixed cost for one activation read.  This entry point
// takes the list and issues one launch for it.
//
// BIT-IDENTICAL, NOT MERELY CLOSE.  Each projection keeps the geometry it would
// have had on its own: the same sub-groups-per-row `sgs_for(N_j)`, the same
// V=16-vs-scalar choice, the same rows-per-work-group, the same per-lane serial
// fma chain, the same 16-lane `reduce_over_group`, and the same cross-sub-group
// fixup.  The only thing that changes is that the per-projection work-group
// ranges are concatenated into one nd_range and each work-group looks its own
// projection up from a by-value descriptor table (no extra submission — the
// table travels in the kernel argument block, not in device memory).  The gate
// requires EXACT equality against the per-projection calls, not a tolerance.
//
// LAUNCH-vs-GPU-TIME TRADE.  The k-loop trip count and `step` become runtime
// values, which costs the compiler its compile-time trip count.  Removing a
// launch is worth ~1.43 us even if the kernel gets that much slower, so the
// kernel is allowed to lose up to (n_proj - 1) * 1.43 us and still win; the
// measured cost is reported in the file comment on the implementation.
struct Ds4GemvProj {
    // Exactly one form: `w` set (fp16 [N, K]) OR `qs`+`qd` set (Q8_0-SoA).
    const sycl::half* w  = nullptr;
    const int8_t*     qs = nullptr;
    const sycl::half* qd = nullptr;
    float*            y  = nullptr;   // fp32 [N]
    uint32_t          N  = 0;
};

// Most projections one launch can carry.  The descriptor table is a by-value
// kernel argument, so this bounds the argument block at 8 * 48 = 384 B; Level
// Zero guarantees at least 1 KB.  A longer list is split across launches.
inline constexpr uint32_t kDs4GemvMultiMax = 8;

// y_j = W_j x for every j in [0, n_proj), one launch per dtype run.
//
// Returns the event of the LAST launch.  Every projection must share `K` and
// `x`; the outputs must not overlap (they are written concurrently).  n_proj
// may exceed kDs4GemvMultiMax — the list is then chunked, which still costs
// strictly fewer launches than one call per projection.
sycl::event ds4_decode_gemv_multi(sycl::queue& q, const char* prof_name,
                                  const float* x, const Ds4GemvProj* projs, uint32_t n_proj,
                                  uint32_t K, const std::vector<sycl::event>& deps = {});

}  // namespace ie
