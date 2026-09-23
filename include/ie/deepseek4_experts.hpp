// include/ie/deepseek4_experts.hpp — DeepSeek-V4-Flash expert compute path
// (Phase 4): a device-side IQ3_XXS GEMV plus the mixed IQ3_XXS / MXFP4 expert
// dispatch and the routed-expert block that sits on top of them.
//
// SCOPE — HONEST.  This is the COMPUTE path only.  There is no streaming, no
// host-arena residency management, no prefetch, and no full forward here; the
// expert banks handed to these functions are already resident in device memory.
// Expert streaming is Phase 5.  `ds4_experts_forward` is decode-shaped: it
// issues one GEMV per (token, routed-slot), so at large T it re-reads an
// expert's weights once per routed slot instead of once per expert.  That is a
// real limitation and it is why the EXPERT-MAJOR BATCHED PATH at the bottom of
// this header exists; the decode-shaped block is kept because it is the
// bit-exact reference the batched path is gated against, and because at T == 1
// the two are the same amount of work.
//
// WHY IT EXISTS.  Before this file the only IQ3_XXS path in the engine was
// `ie::ref::dequant_iq3_xxs_buffer` — a host loop that expands the packed
// weights to fp32.  DeepSeek-V4-Flash's routed experts are 120.4 GB of IQ3_XXS
// (blk.26's gate/up are MXFP4 instead — Unsloth mixed precision), so a
// dequant-first path is not merely slow, it does not fit.
//
// THE DTYPE RULE.  Nothing here keys off the layer index or a per-model
// constant.  `ds4_expert_bank_upload` reads the GGUF tensor's OWN `dtype` and
// picks the repack; `ds4_expert_gemv` dispatches on the bank's recorded dtype.
// A layer whose gate/up are IQ3_XXS and whose down is MXFP4 (every layer except
// blk.26) and a layer that is MXFP4 throughout (blk.26) therefore go through
// the same call site with no branch in the caller.
#pragma once

#include "ie/dtype.hpp"
#include "ie/gguf.hpp"
#include "ie/qwen3moe_pack.hpp"   // MoePacking + build_moe_packing (the host counting sort)

#include <sycl/sycl.hpp>

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace ie {

// ===========================================================================
// Device IQ3_XXS GEMV over the SoA planes (src/ops/gemv_iq3_xxs.cpp)
// ===========================================================================
//
// Plane layout for a bank of E experts, each a [K, N] weight matrix stored
// column-major-by-output (`y[n] = Σ_k w(n,k) * x[k]`, the GGUF expert layout):
//
//   gp  uint8   grid indices        n stride K/4   ; expert stride N*(K/4)
//   ap  uint32  scale/sign words    n stride K/32  ; expert stride N*(K/32)
//   dp  uint16  fp16 super-scales   n stride K/256 ; expert stride N*(K/256)
//
// 2 + 64 + 32 == 98 bytes per 256-element block: the split is exact, so the SoA
// form is the SAME size as the native bank (3.0625 bpw preserved).
// Requires K % 256 == 0.  Both kernels write fp16, like every other GEMV here.

// Host-side repack of a native IQ3_XXS bank (E*N*(K/256) blocks laid out
// (expert, column, super-block)) into the three planes.  `gp`/`ap`/`dp` must be
// sized E*N*(K/4) bytes, E*N*(K/32) words and E*N*(K/256) halves.
void iq3_xxs_repack_soa(const void* src, uint32_t K, uint32_t N, uint32_t E,
                        uint8_t* gp, uint32_t* ap, uint16_t* dp);

// W3A8 int-dot (dp4a_ss).  `x_q8` is a block_q8_1x stream of K/32 blocks
// (quantize_q8_1).  The signed IQ3_XXS grid magnitudes are <= 62, so the
// weight side of the dot is EXACT in int8; the only approximation is the Q8_1
// activation quantiser, exactly as for the Q4/Q6/MXFP4 SoA kernels.
sycl::event gemv_iq3_xxs_soa_q8(sycl::queue& q, const void* x_q8,
                                const uint8_t* gp, const uint32_t* ap, const uint16_t* dp,
                                sycl::half* y, uint32_t K, uint32_t N,
                                const std::vector<sycl::event>& deps = {});

// Value-faithful fp16-activation variant (same db*grid*sign factors as
// ie::ref::dequant_iq3_xxs, fp32 accumulation).  Correctness reference.
sycl::event gemv_iq3_xxs_soa_f16(sycl::queue& q, const sycl::half* A,
                                 const uint8_t* gp, const uint32_t* ap, const uint16_t* dp,
                                 sycl::half* y, uint32_t K, uint32_t N,
                                 const std::vector<sycl::event>& deps = {});

// One column dequantised to fp32 on the device.  Values identical to
// ie::ref::dequant_iq3_xxs; used to prove the repack is lossless.
sycl::event dequant_iq3_xxs_soa_col(sycl::queue& q,
                                    const uint8_t* gp, const uint32_t* ap, const uint16_t* dp,
                                    float* out, uint32_t K, uint32_t N, uint32_t n,
                                    const std::vector<sycl::event>& deps = {});

// ===========================================================================
// Mixed-dtype expert bank
// ===========================================================================

// One layer's routed-expert weight bank (gate, up or down), resident on device
// in whatever SoA form its dtype needs.  `dtype` is copied verbatim from the
// GGUF tensor — never inferred from the role or the model.
struct DS4ExpertBank {
    DType    dtype = DType::kF32;   // kIQ3_XXS or kMXFP4
    uint32_t K = 0;                 // contraction dim (rows of the matrix)
    uint32_t N = 0;                 // output dim (columns)
    uint32_t E = 0;                 // experts resident in this bank

    // IQ3_XXS planes (null for MXFP4).
    uint8_t*  gp = nullptr;
    uint32_t* ap = nullptr;
    uint16_t* dp = nullptr;

    // MXFP4 planes (null for IQ3_XXS) — the gpt-oss layout, reused verbatim:
    //   mx_qs[n*(K/2) + b*16 + j]  nibble bytes,  mx_e[n*(K/32) + b]  E8M0.
    uint8_t*  mx_qs = nullptr;
    uint8_t*  mx_e  = nullptr;

    // Per-expert strides, in elements of the corresponding plane's type.
    uint64_t gp_stride = 0, ap_stride = 0, dp_stride = 0;
    uint64_t mx_qs_stride = 0, mx_e_stride = 0;
};

// Repack + upload the first `E_take` experts of `ti` (a [K, N, E] GGUF expert
// tensor).  Returns "" on success or a diagnostic.  Hard-errors on a dtype this
// path does not implement — it never silently falls back.
std::string ds4_expert_bank_upload(sycl::queue& q, const GgufTensorInfo& ti,
                                   uint32_t K, uint32_t N, uint32_t E_take,
                                   DS4ExpertBank& out);
void ds4_expert_bank_free(sycl::queue& q, DS4ExpertBank& b);

// y[N] (fp16) = W_e^T · x, dispatching on `b.dtype`.  THE CALLER HAS NO DTYPE
// BRANCH — that is the whole point of this function.
//   x_f16 == nullptr : W*A8 int-dot over the block_q8_1x stream `x_q8`
//                      (b.K/32 blocks) — the fast decode path.
//   x_f16 != nullptr : the value-faithful fp16-activation kernel over `x_f16`
//                      (b.K halves); `x_q8` is ignored.  Correctness reference.
sycl::event ds4_expert_gemv(sycl::queue& q, const DS4ExpertBank& b, uint32_t e,
                            const void* x_q8, const sycl::half* x_f16, sycl::half* y,
                            const std::vector<sycl::event>& deps = {});

// ===========================================================================
// Routed-expert block — DeepseekV4Experts.forward (modeling_deepseek_v4.py:993)
// ===========================================================================
//
//   final[t] = Σ_k topk_w[t,k] · down_{e}( silu(clamp(gate_{e}(x[t]), max=L))
//                                          * clamp(up_{e}(x[t]), -L, L) )
//   where e = topk_idx[t,k] and L = swiglu_limit (config: 10.0).
//
// The clamp is ASYMMETRIC — gate is bounded ABOVE only, up on both sides — and
// the routing weight is applied AFTER down, not before.  Both are transcribed
// from `DeepseekV4Experts._apply_gate` / `.forward`; the clamped SwiGLU itself
// reuses the Phase-2, parity-gated `ds4_swiglu_clamped`.
//
// `topk_idx` / `topk_w` are HOST arrays of T*top_k entries: the expert id has
// to reach the host anyway to offset the bank pointers (same as the gpt-oss and
// qwen3moe MoE paths).  `x` and `y` are device fp32 [T, H]; `y` is zeroed here.
// The queue must be in-order.

struct DS4ExpertWorkspace {
    uint32_t    H = 0, EF = 0;
    sycl::half* xh     = nullptr;   // [H]      fp16 activation
    void*       x_q8   = nullptr;   // [H/32]   block_q8_1x
    sycl::half* gate_h = nullptr;   // [EF]
    sycl::half* up_h   = nullptr;   // [EF]
    float*      gate_f = nullptr;   // [EF]
    float*      up_f   = nullptr;   // [EF]
    float*      h_f    = nullptr;   // [EF]
    sycl::half* h_h    = nullptr;   // [EF]
    void*       h_q8   = nullptr;   // [EF/32]  block_q8_1x
    sycl::half* y_h    = nullptr;   // [H]
};

std::string ds4_expert_ws_alloc(sycl::queue& q, uint32_t H, uint32_t EF,
                                DS4ExpertWorkspace& ws);
void ds4_expert_ws_free(sycl::queue& q, DS4ExpertWorkspace& ws);

// `f16_activation` selects the value-faithful fp16-activation GEMVs instead of
// the W*A8 int-dot ones.  Same block structure either way; it exists so the
// gate can separate "the block is wired correctly" from "the Q8_1 activation
// quantiser costs 0.4%".
std::string ds4_experts_forward(sycl::queue& q,
                                const DS4ExpertBank& gate,
                                const DS4ExpertBank& up,
                                const DS4ExpertBank& down,
                                const float* x, const int32_t* topk_idx,
                                const float* topk_w, float* y,
                                uint32_t T, uint32_t H, uint32_t EF, uint32_t top_k,
                                float swiglu_limit, DS4ExpertWorkspace& ws,
                                bool f16_activation = false);

// Host reference for the block above, transcribed line-by-line from
// DeepseekV4Experts.forward, over ALREADY-DEQUANTISED fp32 weights.  Kept in
// the public header because it is the cross-check the Phase 4 gate runs:
//   (a) fed the parity blobs' own fp32 expert weights it must reproduce the
//       reference output (proves the transcription), and
//   (b) fed the real model's weights dequantised by the bit-exact
//       ie::ref::dequant_*_buffer it is the target the device path must hit.
// gate_up is [E, 2*EF, H] (gate rows first, then up — the reference's fused
// `gate_up_proj`); down is [E, H, EF].  Accumulates in double.
void ds4_experts_forward_ref(const float* x, const float* gate_up, const float* down,
                             const int32_t* topk_idx, const float* topk_w, float* y,
                             uint32_t T, uint32_t H, uint32_t EF, uint32_t top_k,
                             uint32_t E, float swiglu_limit);

// ===========================================================================
// EXPERT-MAJOR BATCHED PATH — the T>1 form of the block above
// ===========================================================================
//
// WHY IT EXISTS.  `ds4_experts_forward` (and the equivalent loop inlined in
// DeepSeek4Runtime::layer_forward_pre) is DECODE-shaped: it submits 9 kernels
// per (token, routed-slot) and re-reads the routed expert's whole weight slice
// on every one of them.  At T=128 with top-6 of 256 that is 7,168 submissions
// per layer-card and, because 768 routed slots land on ~243 distinct experts,
// ~3.2 reads of every slice that is touched at all.
//
// The kernels below invert the loop.  Tokens are counting-sorted by expert with
// `build_moe_packing` — the SAME host packer gpt-oss (gptoss_tp.cpp:990) and
// qwen3moe already use — and each expert then runs ONE batched GEMM over all of
// the rows routed to it.  Submission count becomes O(distinct experts in the
// chunk) + 9, INDEPENDENT of T, and every weight slice is read exactly once.
//
// BIT-IDENTICAL, NOT MERELY CLOSE.  The batched GEMMs are the decode GEMVs with
// an M loop wrapped around the innermost dot: identical lane->sub-block split-K
// assignment, identical dp4a operand order, identical `db * q8d * float(idot)`
// fold, identical sub-group reduce, identical `sycl::half(acc)` store.  The only
// structural change is that a lane decodes its sub-block's weights ONCE and
// re-uses them across the M-tile instead of re-decoding per token.  The
// elementwise stages (cast / clamped SwiGLU / Q8_1) are blockwise and row-
// agnostic, so running them over the concatenated packed rows is the same
// arithmetic; the final scatter accumulates in ASCENDING kslot order, the same
// order the per-token `accum_f16` chain used.  deepseek4_experts_test §7 asserts
// EXACT equality against the token-major path — no tolerance.
//
// WHAT IT DELIBERATELY IS NOT.  It is not a dequant-to-fp16 + oneDNN GEMM (the
// gpt-oss prefill shape).  That would change the numerics, and on this hardware
// the small-M XMX regime is the known-corrupting one (gptoss_tp.cpp:703).  This
// keeps the proven W*A8 int-dot arithmetic and only fixes the loop order.

// y[M, N] (fp16, row-major) = W_e^T · x for M packed activation rows.
// `x_q8` is M contiguous block_q8_1x rows of b.K/32 blocks each.  Dispatches on
// `b.dtype` exactly like `ds4_expert_gemv`, and reduces to it when M == 1.
sycl::event ds4_expert_gemm_q8(sycl::queue& q, const DS4ExpertBank& b, uint32_t e,
                               uint32_t M, const void* x_q8, sycl::half* y,
                               const std::vector<sycl::event>& deps = {});

// ===========================================================================
// GROUPED EXPERT GEMM — every occupied expert of a chunk in ONE launch
// ===========================================================================
//
// WHY.  The expert-major path above already reads each weight slice once, but
// it still submits one kernel per (expert, projection).  On a pp512 prefill of
// DeepSeek-V4-Flash that is 73,698 launches, and the problem is not the
// submission cost (measured 1.4 us/launch, 5% of the time) — it is that ONE
// expert's gate at N=1024 fills only 32 work-groups on a 32-Xe-core B70, so the
// machine runs one work-group per core with nothing to hide memory latency
// behind.  Folding the whole chunk's experts into a single 2-D launch (grid
// dim 0 = job, dim 1 = column tile) raises that to thousands of work-groups.
//
// MEASURED, one layer-chunk of the real shape (H=4096, EFc=1024, 256 experts,
// 1536 routed rows, MXFP4, B70 card 1): 19.97 ms over 768 launches -> 14.30 ms
// over 2.  Combined with the column tiling in the kernels themselves the same
// work is 9.55 ms.  The kernels are NOT bandwidth-bound: a pure streaming read
// of the same arena runs at 599 GB/s and this reaches 179 GB/s.
//
// BIT-IDENTICAL.  A job carries the same plane pointers, the same row range and
// the same (K, N) the single-expert entry point would have used, and the kernel
// body is the same body — only the work-group -> (job, column-tile) mapping
// changed.  Every output element still accumulates over the identical
// lane -> sub-block assignment and the identical sub-group reduce.
// deepseek4_expertgemm_gate_test asserts EXACT equality, no tolerance.

// One single-expert GEMM inside a grouped launch.  `bank`/`e` name the weight
// slice exactly as `ds4_expert_gemm_q8` does — including the dtype, which is
// read from THIS bank and never from a per-model constant, so a group may mix
// IQ3_XXS and MXFP4 experts freely (it is then split into one launch per
// dtype).
struct DS4GemmJob {
    DS4ExpertBank bank{};
    uint32_t      e     = 0;         // expert index inside `bank`
    uint32_t      M     = 0;         // rows of `x_q8` this expert consumes
    const void*   x_q8  = nullptr;   // M rows of bank.K/32 block_q8_1x
    sycl::half*   y     = nullptr;   // [M, bank.N] fp16, row-major
    // First PACKED ROW this expert owns.  Unused by the int-dot route (whose
    // x_q8 already points at the row base); the XMX route needs it to index the
    // shared f16 activation block.  Defaulted so every existing site is
    // unaffected.
    uint32_t      row0  = 0;
    // MXFP4 banks only: when set, the rows land here as fp32 [M, bank.N] and `y` is not written (MiMo-V2.6's unclamped
    // SwiGLU experts can exceed fp16's range, docs/mimo26). Defaulted null: every existing site writes fp16, bit for bit.
    float*        y32   = nullptr;
};

// Device-side descriptor staging for the grouped launch.
//
// The ring exists because the queue is in-order but the HOST is not: the
// descriptor memcpy of launch i may still be in flight when the host starts
// filling the staging buffer for launch i+1.  Reusing a slot therefore waits on
// that slot's own previous copy — after `kDs4GemmRing` intervening launches
// that wait is already satisfied and costs nothing.
struct DS4GemmGroupWs {
    uint32_t cap = 0;          // jobs per ring slot
    void*    host = nullptr;   // pinned staging, kDs4GemmRing * cap descriptors
    void*    dev  = nullptr;   // device copy, same size
    uint32_t cursor = 0;
    std::vector<sycl::event> slot_ev;
};
constexpr uint32_t kDs4GemmRing = 4;

std::string ds4_gemm_group_ws_alloc(sycl::queue& q, uint32_t max_jobs,
                                    DS4GemmGroupWs& gws);
void ds4_gemm_group_ws_free(sycl::queue& q, DS4GemmGroupWs& gws);

// Issues `n_jobs` single-expert GEMMs in as few launches as possible: one per
// distinct (dtype, K, N) present, further split only if a bucket exceeds
// `gws.cap`.  Returns "" or a diagnostic; never silently drops a job.
//
// A launch in which EVERY job has M == 1 — which is exactly single-token decode
// — takes a separate pair of kernels; see `ds4_expert_gemm_m1_ncols`.  Nothing
// else about the call changes, and the results are bit-identical either way.
std::string ds4_expert_gemm_q8_grouped(sycl::queue& q, const DS4GemmJob* jobs,
                                       uint32_t n_jobs, DS4GemmGroupWs& gws);

// Float-output XMX descriptor. This intentionally replaces the old wrapper's
// DS4GemmJob parameter: callers must supply a float allocation in y, never cast
// half storage. DS4GemmJob continues to describe the int-dot fp16 output.
struct DS4XmxGemmJob {
    DS4ExpertBank bank{};
    uint32_t e = 0;
    uint32_t M = 0;
    uint32_t row0 = 0;             // first row in shared x_f16
    float* y = nullptr;           // [M, bank.N] fp32, row-major
};

// XMX ROUTE (opt-in, $IE_DS4_EXPERT_XMX=1) — oneDNN f16 matmul over
// a per-expert dequantised weight instead of the int-dot kernel.
//
// WHY IT IS OPT-IN AND WHAT CHANGED.  XMX was BANNED on expert GEMMs by the
// gpt-oss 120b corruption (gptoss_tp.cpp:703, June 2026: small-M f16 matmul on
// BMG garbled every packed row except row 0 -> PPL 42 vs 12.91).  Re-probed
// 2026-08-09 on the current driver/oneDNN 2026.0: 160 trials across M in
// {1,2,3,4,6,8,16,24}, worst relative error 1.4e-3 = pure f16 rounding, ZERO
// gross errors on row 0 or any later row — the corruption does not reproduce.
// Measured ceiling at the real shapes: 11.5-32 TFLOP/s vs the int-dot kernel's
// ~3, i.e. the only remaining lever on a surface that is 66%% of prefill.
//
// IT IS A NUMERICS CHANGE, and that is why it stays behind a flag until a PPL
// gate says otherwise: the int-dot path is W4A8 (activations quantised to
// int8), this path is W4A16 with an f16 accumulate chain.  The WEIGHT side is
// exact either way — every MXFP4 value is representable in f16.
// Requires an in-order queue and serialized host calls on that queue. All live
// jobs require K=H_in divisible by 32, N divisible by 128, and valid bank slices.
// Empty jobs are ignored, and zero work submits nothing. No 16-byte alignment
// is required for x_f16 or the packed bytes. The caller owns allocation extents.
// `w16_scratch` must hold max(K*N) halves only for IE_DS4_EXPERT_XMX_FUSED=0;
// the default fused route accepts null scratch. The oneDNN fallback stages
// operands that lack 16-byte alignment and waits before freeing those copies.
// IQ3_XXS jobs are REFUSED (no dequant kernel here); the caller keeps the
// int-dot route for them.
std::string ds4_expert_gemm_xmx_grouped(sycl::queue& q, const DS4XmxGemmJob* jobs,
                                        uint32_t n_jobs, const sycl::half* x_f16,
                                        uint32_t H_in, sycl::half* w16_scratch,
                                        uint64_t w16_capacity);

// Is the XMX expert route enabled ($IE_DS4_EXPERT_XMX=1, default OFF)?
bool ds4_expert_xmx_on();

// Whether a launch of `n_jobs` jobs at output width `N` will use the
// COLUMN-TILED kernel (one sub-group owning 4 weight columns) or the untiled
// one.  Tiling divides the work-group count by 4, which is free on a wide grid
// and a 2x loss on a narrow one, so it is chosen from the resulting grid width.
//
// Exposed for one reason: the gate has to be able to ASSERT it exercised both
// kernels instead of silently testing the same one twice.
bool ds4_expert_gemm_column_tiled(uint32_t n_jobs, uint32_t N);

// ===========================================================================
// SINGLE-TOKEN DECODE — the M == 1 kernels
// ===========================================================================
//
// At T=1 the routed-expert GEMM is 6 experts x ONE token row per layer, so the
// M>1 body's register blocking is not merely unhelpful, it is a cost: its row
// bound is a runtime value, so eight predicated row bodies are emitted and
// acc[NC][8] stays live when one row is in flight.  A launch whose jobs ALL
// have M == 1 therefore takes a body with the row loop removed, and picks its
// column width from the resulting grid instead of from the two-way boolean
// above (which can only express NC=1 and NC=4, and at the real decode shapes
// picks the wrong one for both projections).
//
// MEASURED, real decode shapes on a B70, weights rotated over a 408 MB pool so
// nothing survives the 24 MB L2, clock pinned, min of 5 interleaved passes:
//
//   MXFP4  gate+up  12 jobs K=4096 N=1024  0.0826 -> 0.0486 ms  (1.70x)
//   MXFP4  down      6 jobs K=1024 N=4096  0.0374 -> 0.0278 ms  (1.35x)
//   IQ3    gate+up  12 jobs K=4096 N=1024  0.0711 -> 0.0488 ms  (1.46x)
//
// A pure streaming read of the same MXFP4 arena runs at 619 GB/s and the tuned
// kernel reaches 550, so the decode expert GEMM is now within 13% of the memory
// system rather than spending its issue slots on nibble decode.
//
// BIT-IDENTICAL, as with every other step here: the lane -> sub-block
// assignment, the dp4a operand order, the `db * q8d * float(idot)` fold and the
// 16-lane sub-group reduce are all unchanged, and an output element's value is
// a function of only those.  deepseek4_expertgemm_gate_test §6 asserts EXACT
// equality against the untouched reference GEMVs, no tolerance.
//
// Returns the column width (1, 2, 4 or 8) an all-M==1 launch of `n_jobs` at
// output width `N` will use.  Exposed for the same reason
// `ds4_expert_gemm_column_tiled` is: the gate must be able to assert it
// exercised more than one of the four widths.
uint32_t ds4_expert_gemm_m1_ncols(uint32_t n_jobs, uint32_t N);

// xp[m, :] = fp16(x[rows[m], :]) — the expert-sorted gather fused with the
// fp32->fp16 cast the token-major path did per token.  Same halves either way.
sycl::event ds4_expert_gather_cast(sycl::queue& q, const float* x, const int32_t* rows,
                                   sycl::half* xp, uint32_t M, uint32_t H,
                                   const std::vector<sycl::event>& deps = {});

// y[t, :] += Σ_{kslot=0..top_k-1} w_packed[p] · float(yp[p, :]),  p = tk2p[t*top_k+kslot].
// ACCUMULATES (the caller owns zeroing), and walks kslot ASCENDING so the fp32
// summation order matches the token-major `accum_f16` chain exactly.
// f32-source variant (the XMX `down` route's oneDNN result, unrounded).
sycl::event ds4_expert_scatter_accum_f32(sycl::queue& q, const float* yp,
                                         const int32_t* tk2p, const float* w_packed,
                                         float* y, uint32_t T, uint32_t top_k, uint32_t H,
                                         const std::vector<sycl::event>& deps = {});

sycl::event ds4_expert_scatter_accum(sycl::queue& q, const sycl::half* yp,
                                     const int32_t* tk2p, const float* w_packed,
                                     float* y, uint32_t T, uint32_t top_k, uint32_t H,
                                     const std::vector<sycl::event>& deps = {});

// The largest token batch the expert-major path packs at once.  It bounds the
// packed workspace (rows = cap * top_k, and the two [rows, H] fp16 buffers are
// the big ones: at H=7168 / top-6 that is 2 x 21 MB).  The runtime's streaming
// chunk loop splits on the SMALLER of this and the layer's free slot count, so
// raising it never overruns the buffers — it only changes how much re-reading
// of a hot expert the chunking still forces.
// Tokens per expert-major range.  EVERY range re-streams the layer's expert
// union, so this cap is the prefill expert-DMA amortization factor: at 256 the
// measured pp512 streamed 285.9 acquire-ids/layer/card (two ranges' unions)
// and a 1024-token chunk moved 108.9 GB/card — 2.3x the 47.4 GB union floor.
// Raised 256 -> 1024 on 2026-08-09: a 1024-token chunk becomes ONE range
// (union streamed once), the batch workspace grows 4x (~370 MB/card at the
// V4-Flash shapes — now counted against the expert-arena budget via
// ds4_expert_batch_ws_bytes), and the taller per-expert GEMMs help oneDNN.
// The range planner ALSO bounds a range by the streaming-slot budget, so a
// larger cap degrades gracefully where slots bind first.
constexpr uint32_t kDs4BatchTokenCap = 2048;   // docs/deepseek4/72 Phase N: was 1024

// Packed workspace for one batch of at most `max_tokens` tokens.  Every buffer
// is sized for max_tokens*top_k rows; `EF` is the WIDEST intermediate slice the
// caller will ask for (under expert-TP the per-call width is this card's slice,
// which is smaller, and the buffers are simply used strided by that width).
struct DS4ExpertBatchWs {
    uint32_t max_tokens = 0, top_k = 0, H = 0, EF = 0, rows = 0;
    sycl::half* xp    = nullptr;   // [rows, H]
    void*       xp_q8 = nullptr;   // [rows * H/32]  block_q8_1x
    sycl::half* g_h   = nullptr;   // [rows, EF]
    sycl::half* u_h   = nullptr;
    float*      g_f   = nullptr;
    float*      u_f   = nullptr;
    float*      h_f   = nullptr;
    sycl::half* h_h   = nullptr;
    void*       h_q8  = nullptr;   // [rows * EF/32] block_q8_1x
    sycl::half* yp    = nullptr;   // [rows, H]
    int32_t*    row_tok = nullptr; // [rows]  token id of each packed row
    int32_t*    tk2p    = nullptr; // [rows]  (t*top_k+kslot) -> packed row
    float*      w_pk    = nullptr; // [rows]  routing weight of each packed row
    // Host scratch, reused across chunks so the steady state allocates nothing.
    MoePacking  pk{};
    std::vector<std::vector<std::pair<uint32_t, float>>> routes;
    // Descriptor staging for `ds4_expert_gemm_q8_grouped`.  Lives here so the
    // routed-expert block — and the runtime's streaming twin of it, which holds
    // exactly one of these per model — allocates it once.
    DS4GemmGroupWs           grp{};
    std::vector<DS4GemmJob>  jobs;
    // XMX route only ($IE_DS4_EXPERT_XMX): one dequantised f16 weight, reused
    // per job. max(K*N) over the expert projections; null when the route is off.
    sycl::half* w16     = nullptr;
    uint64_t    w16_cap = 0;
    // f32 landing for the XMX `down` projection (oneDNN writes f32; the
    // scatter consumes f16), cast back into `yp` per group.
    float*      y_f32   = nullptr;
};

// `max_jobs` bounds ONE grouped launch: gate+up for every occupied expert of a
// chunk, i.e. 2*n_experts.  A larger group is split into several launches
// rather than refused, so the default is a performance knob, not a limit.
// The device bytes ds4_expert_batch_ws_alloc will allocate for these shapes
// (excluding the small ds4_gemm_group_ws jobs arrays).  The residency planner
// subtracts this BEFORE sizing the expert arena — the workspace is allocated
// after the arena budget is derived, so an unaccounted growth here would eat
// the load-bearing 2 GB margin instead.  Keep in lockstep with the alloc.
// MXFP4 planes -> f16 W (column-contiguous, the gemm_nt_f16_onednn B layout).
sycl::event ds4_dequant_mxfp4_w16(sycl::queue& q,
                                  const uint8_t* qs_plane, const uint8_t* e_plane,
                                  sycl::half* w16, uint32_t K, uint32_t N,
                                  const std::vector<sycl::event>& deps = {});

// Fused MXFP4 -> XMX GEMM (docs/deepseek4/72 Phase E): y[M,N] (fp32, row
// stride N) = x[M,K] (fp16, row stride K) . W[N,K]^T with W read straight from
// the MXFP4 planes (`qs` [N][K/2] nibbles, `e` [N][K/32] e8m0) and dequantised
// into the SLM B tile — the same exact fp16 values ds4_dequant_mxfp4_w16
// materialises, fp32 accumulation, never a full-width weight in memory.  One
// launch over all `n_jobs` (K % 32 == 0, N % 128 == 0, any M; rows >= M are
// never written).  Throws std::invalid_argument on a shape it cannot carry.
// Job descriptors are copied before return; operand storage must remain live
// until the returned event completes. Both queue orderings and concurrent host
// submissions are supported. Ring reuse may wait for a prior descriptor copy.
// Empty work preserves deps and never rewrites an outstanding descriptor slot.
struct Ds4MxXmxJob {
    const uint8_t*    qs = nullptr;
    const uint8_t*    e  = nullptr;
    const sycl::half* x  = nullptr;
    float*            y  = nullptr;
    uint32_t M = 0, K = 0, N = 0;
};
sycl::event ds4_gemm_mxfp4_xmx(sycl::queue& q, const Ds4MxXmxJob* jobs, uint32_t n_jobs,
                               const std::vector<sycl::event>& deps = {});

uint64_t ds4_expert_batch_ws_bytes(uint32_t max_tokens, uint32_t top_k,
                                   uint32_t H, uint32_t EF) noexcept;

std::string ds4_expert_batch_ws_alloc(sycl::queue& q, uint32_t max_tokens, uint32_t top_k,
                                      uint32_t H, uint32_t EF, DS4ExpertBatchWs& ws,
                                      uint32_t max_jobs = 1024);
void ds4_expert_batch_ws_free(sycl::queue& q, DS4ExpertBatchWs& ws);

// Resolves an expert id to the three single-expert bank views for THIS card.
// Returns false when the id has no bank (streamed-but-not-resident), which the
// block turns into a hard error rather than a skipped term.  It is a callback
// because the two callers hold their experts very differently: the test holds a
// contiguous E-expert bank, the runtime holds one slot pointer per expert in the
// streaming cache.  Both go through the identical batched code below.
using Ds4ExpertBankFn = std::function<bool(uint32_t e, DS4ExpertBank& gate,
                                           DS4ExpertBank& up, DS4ExpertBank& down)>;

// The expert-major routed-expert block.  `EFc` is THIS CARD's intermediate slice
// width (== gate.N of the banks the resolver returns), not the model's EF.
// `y` is ACCUMULATED into, [T, H] fp32; the caller zeroes it.  In-order queue.
std::string ds4_experts_forward_batched(sycl::queue& q, const Ds4ExpertBankFn& banks,
                                        const float* x, const int32_t* topk_idx,
                                        const float* topk_w, float* y,
                                        uint32_t T, uint32_t H, uint32_t EFc,
                                        uint32_t top_k, uint32_t n_experts,
                                        float swiglu_limit, DS4ExpertBatchWs& ws);

}  // namespace ie
