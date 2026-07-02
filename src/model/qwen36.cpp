// src/model/qwen36.cpp — Qwen3.6-35B-A3B model loader + 40-layer forward pass.
//
// Cross-references: research/04 (model spec), research/05 (DeltaNet), research/06 (MoE).
// This file is purely orchestration over the kernels built in phases 1-7.

#include "ie/qwen36.hpp"
#include "ie/qwen35_dense.hpp"   // Qwen35SpecCheckpoint (generic DeltaNet snapshot, reused)

#include "ie/dequant.hpp"
#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"
#include "ie/quant_soa.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Compile-time switch — fuse ssm_alpha + ssm_beta projections (same shape,
// same input) into a single gemv_q4_K_dual launch per DeltaNet layer.
// Halves the per-decode-step launch count for these calls (60 → 30) and
// shares the cooperative A-SLM load across both projections.
#ifndef IE_FUSE_SSM_AB
#define IE_FUSE_SSM_AB 1
#endif

// Compile-time switch — fuse the shared-expert ffn_gate_shexp + ffn_up_shexp
// projections (same shape K=H N=E_ffn, same input) into a single
// gemv_q4_K_dual_ffn launch per layer.  Halves the per-decode-step launch
// count for these calls (80 → 40 across 40 layers) and shares the
// cooperative A-SLM load — same lever that won on ssm_alpha+beta.
#ifndef IE_FUSE_FFN_GATE_UP
#define IE_FUSE_FFN_GATE_UP 1
#endif

// Compile-time switch — fuse residual_add + rms_norm_f32w (post_attn_norm
// site only for now) into a single launch.  Eliminates one rms_norm
// launch per layer (40/step) at ~29 µs each = ~1.16 ms/step pure
// launch-overhead saving.
#ifndef IE_FUSE_RES_RMS
#define IE_FUSE_RES_RMS 1
#endif

// Compile-time switch — route the multi-expert prefill stage-2 down
// projection through the M_TILE=8 SLM-amortized v2 kernels.  v1 had no
// SLM staging and re-streamed each Q-block from DRAM once per token; v2
// mirrors moe_prefill_gate_up_silu_q4k's tile pattern.  Audit projected
// +15-25% prefill at pp512.  Default ON; build with -DIE_ENABLE_MOE_DOWN_TILE=0
// to fall back to v1 for A/B testing.
#ifndef IE_ENABLE_MOE_DOWN_TILE
#define IE_ENABLE_MOE_DOWN_TILE 1
#endif

// Compile-time switch — route Q4_K T>1 prefill GEMM through the
// gemm_q4_K_esimd kernel (cooperative SG block reads + vec<u8,16> Q4_K
// dequant) instead of gemm_q4_K_xmx.  Default OFF until pp512 gate
// (>= 200 tok/s) is verified.  Override with -DIE_ENABLE_GEMM_Q4K_ESIMD=1
// to enable.  Same shape constraints as the XMX variant (N%64==0,
// K%256==0); the dispatch flips between the two implementations.
#ifndef IE_ENABLE_GEMM_Q4K_ESIMD
#define IE_ENABLE_GEMM_Q4K_ESIMD 0
#endif

// Compile-time switch — route T>1 full-attention prefill through the
// tiled FA-2 kernel `full_attention_fa2_prefill` instead of the naive
// per-token-WG `full_attention`.  Default OFF until the gate is met
// (pp512 ≥ 165 tok/s = +10 over Step-1-OFF baseline of 155).  Override
// with -DIE_ENABLE_FA2_PREFILL_TILED=1 to enable.  Same correctness
// contract: appends (k_in, v_in) for the new T tokens to cache, then
// computes attention with causal mask over [0, start_pos+T).
#ifndef IE_ENABLE_FA2_PREFILL_TILED
#define IE_ENABLE_FA2_PREFILL_TILED 0
#endif

// Function-like so the v2 kernels get the SoA-layout flag while the v1
// fallbacks (AoS-only; selected with IE_ENABLE_MOE_DOWN_TILE=0, which also
// disables the load-time repack below) silently drop it.
#if IE_ENABLE_MOE_DOWN_TILE
#  define IE_MOE_DN_PK4K(q, h, w, off, out, E, H, Ef, sd, soa) \
       moe_prefill_down_packed_q4k_v2(q, h, w, off, out, E, H, Ef, sd, soa)
#  define IE_MOE_DN_PK6K(q, h, w, off, out, E, H, Ef, sd, soa) \
       moe_prefill_down_packed_q6k_v2(q, h, w, off, out, E, H, Ef, sd, soa)
#else
#  define IE_MOE_DN_PK4K(q, h, w, off, out, E, H, Ef, sd, soa) \
       moe_prefill_down_packed_q4k(q, h, w, off, out, E, H, Ef, sd)
#  define IE_MOE_DN_PK6K(q, h, w, off, out, E, H, Ef, sd, soa) \
       moe_prefill_down_packed_q6k(q, h, w, off, out, E, H, Ef, sd)
#endif

namespace ie {

// Forward decls — defined in src/ops/gemv_q4k.cpp.  Profile-named aliases
// keep the kernel monitor's buckets distinct so each fusion site can be
// measured independently.
sycl::event gemv_q4_K_dual_ssm(sycl::queue& q,
                               const sycl::half* A,
                               const void* W_alpha, const void* W_beta,
                               sycl::half* y_alpha, sycl::half* y_beta,
                               uint32_t K, uint32_t N,
                               const std::vector<sycl::event>& deps = {});
sycl::event gemv_q4_K_dual_ffn(sycl::queue& q,
                               const sycl::half* A,
                               const void* W_gate, const void* W_up,
                               sycl::half* y_gate, sycl::half* y_up,
                               uint32_t K, uint32_t N,
                               const std::vector<sycl::event>& deps = {});

// Forward decl — gemm_q4_K_esimd, ESIMD-style cooperative-load Q4_K GEMM
// drop-in for gemm_q4_K_xmx.  Defined in src/ops/gemm_q4k_esimd.cpp.
sycl::event gemm_q4_K_esimd(sycl::queue& q,
                             const sycl::half* A, const void* W_packed,
                             sycl::half* y,
                             uint32_t M, uint32_t K, uint32_t N,
                             const std::vector<sycl::event>& deps = {});

// Forward decl — tiled FA-2 prefill (T>1).  Drop-in replacement for
// `full_attention` (naive per-token-WG) when IE_ENABLE_FA2_PREFILL_TILED
// is set.  Defined in src/ops/attention.cpp.
sycl::event full_attention_fa2_prefill(sycl::queue& q,
                                       const sycl::half* q_in,
                                       const sycl::half* k_in,
                                       const sycl::half* v_in,
                                       sycl::half* k_cache,
                                       sycl::half* v_cache,
                                       sycl::half* y,
                                       uint32_t T,
                                       uint32_t start_pos,
                                       uint32_t n_q_heads,
                                       uint32_t n_kv_heads,
                                       uint32_t head_dim,
                                       uint32_t max_ctx,
                                       const std::vector<sycl::event>& deps = {});

// Forward decls — v2 stage-2 down kernels with M_TILE=8 SLM amortization.
// Defined in src/ops/moe_fused.cpp.  Same signature as the v1 packed
// variants in include/ie/ops.hpp; selected by IE_ENABLE_MOE_DOWN_TILE.
sycl::event moe_prefill_down_packed_q4k_v2(sycl::queue& q,
                                           const sycl::half* h_packed,
                                           const void* down_W,
                                           const uint32_t* expert_offsets,
                                           sycl::half* out_packed,
                                           uint32_t E, uint32_t H, uint32_t E_ffn,
                                           uint64_t expert_stride_bytes, bool soa,
                                           const std::vector<sycl::event>& deps = {});
sycl::event moe_prefill_down_packed_q6k_v2(sycl::queue& q,
                                           const sycl::half* h_packed,
                                           const void* down_W,
                                           const uint32_t* expert_offsets,
                                           sycl::half* out_packed,
                                           uint32_t E, uint32_t H, uint32_t E_ffn,
                                           uint64_t expert_stride_bytes, bool soa,
                                           const std::vector<sycl::event>& deps = {});

// Forward decl — defined in src/ops/elementwise.cpp.
sycl::event residual_add_rms_norm_fused(sycl::queue& q,
                                        sycl::half* ws_x,
                                        const sycl::half* ws_block,
                                        const float* w_norm,
                                        sycl::half* ws_normed,
                                        uint32_t n_rows, uint32_t hidden,
                                        float eps,
                                        const std::vector<sycl::event>& deps = {});


namespace {

// Helper: load a tensor from GGUF onto device, return device pointer.
// `expected` may be DType::kCount to skip the dtype check (any).
template <typename T>
T* upload(DeviceAllocator& alloc, const GgufTensorInfo* t,
          std::vector<void*>& owned, std::string& err,
          DType expected = DType::kCount) {
    if (!t) { err = "tensor not found"; return nullptr; }
    if (expected != DType::kCount && t->dtype != expected) {
        err = std::string("tensor dtype mismatch: ") +
              std::string(type_name(t->dtype)) + " expected " +
              std::string(type_name(expected));
        return nullptr;
    }
    void* d = alloc.malloc(t->nbytes);
    if (!d) { err = "malloc failed"; return nullptr; }
    alloc.queue().memcpy(d, t->data, t->nbytes).wait();
    owned.push_back(d);
    return static_cast<T*>(d);
}

// Quantized tensor loader: accepts either Q4_K or Q6_K, returns a QuantPtr.
QuantPtr upload_quant(DeviceAllocator& alloc, const GgufTensorInfo* t,
                      std::vector<void*>& owned, std::string& err) {
    QuantPtr out;
    if (!t) { err = "tensor not found"; return out; }
    if (t->dtype == DType::kF16) {
        // F16 weight (e.g. high-fidelity fine-tuned projections). The GGUF stores
        // it [N,K] row-major, but gemv_fp16 (the F16 branch of gemv_q) consumes
        // [K,N] — so host-transpose on upload. Faithful mirror of the dense path
        // (src/model/dense_dispatch.hpp upload_quant). ADDITIVE: a normal crown
        // GGUF has no F16 projection tensors, so this branch never fires for it.
        const uint64_t K = t->shape[0];   // ggml: shape[0] = contiguous dim
        const uint64_t N = t->shape[1];
        if (t->n_dims != 2 || K == 0 || N == 0 ||
            t->nbytes != K * N * sizeof(uint16_t)) {
            err = std::string("f16 tensor '") + std::string(t->name) +
                  "': unexpected geometry";
            return out;
        }
        std::vector<uint16_t> staging(K * N);
        const auto* src = reinterpret_cast<const uint16_t*>(t->data);
        for (uint64_t n = 0; n < N; ++n)
            for (uint64_t k = 0; k < K; ++k)
                staging[k * N + n] = src[n * K + k];
        void* d = alloc.malloc(t->nbytes);
        if (!d) { err = "malloc failed"; return out; }
        alloc.queue().memcpy(d, staging.data(), t->nbytes).wait();
        owned.push_back(d);
        out.p = d;
        out.dt = DType::kF16;
        return out;
    }
    if (t->dtype == DType::kQ8_0 && t->n_dims == 2) {
        // Q8_0 weight (near-lossless fine-tuned projections). The crown has no
        // raw-Q8_0 GEMV, so dequant to fp16 [K,N] at load → it then rides the
        // gemv_fp16 dispatch above. Faithful mirror of the dense _auto path
        // (src/model/dense_dispatch.hpp upload_dequant_to_fp16). ADDITIVE: a
        // normal crown GGUF has no Q8_0 projections. The n_dims==2 guard keeps the
        // 3-D MoE expert banks (consumed RAW by the fused kernel) on the raw path
        // below — they are never Q8_0 in practice, but this fails safe regardless.
        const uint32_t K = uint32_t(t->shape[0]);
        const uint32_t N = uint32_t(t->shape[1]);
        if (K == 0 || N == 0) { err = "q8_0 dequant: zero dim"; return out; }
        void* packed = alloc.malloc(t->nbytes);
        if (!packed) { err = "malloc failed (q8_0 packed)"; return out; }
        alloc.queue().memcpy(packed, t->data, t->nbytes).wait();
        auto* d = static_cast<sycl::half*>(
            alloc.malloc(uint64_t(K) * N * sizeof(sycl::half)));
        if (!d) { alloc.free(packed); err = "malloc failed (q8_0 fp16)"; return out; }
        dequant_q8_0_to_Bt(alloc.queue(), packed, d, K, N).wait();
        alloc.free(packed);
        owned.push_back(d);
        out.p = d;
        out.dt = DType::kF16;
        return out;
    }
    if (t->dtype != DType::kQ4_K && t->dtype != DType::kQ6_K &&
        t->dtype != DType::kQ8_0 && t->dtype != DType::kQ5_K) {
        err = std::string("tensor dtype unsupported: ") +
              std::string(type_name(t->dtype));
        return out;
    }
    void* d = alloc.malloc(t->nbytes);
    if (!d) { err = "malloc failed"; return out; }
    alloc.queue().memcpy(d, t->data, t->nbytes).wait();
    owned.push_back(d);
    out.p = d;
    out.dt = t->dtype;
    return out;
}

// Prefill-crown (2026-06-10): MoE expert tensor loader with per-expert SoA
// repack (ie/quant_soa.hpp).  Repacks on the host from the mmap'd GGUF data
// into `staging`, then uploads — same bits moved, PPL-free by construction.
// `staging` is caller-owned so its capacity is reused across the 120 expert
// tensors instead of re-allocating ~144 MiB per tensor.
QuantPtr upload_quant_soa(DeviceAllocator& alloc, const GgufTensorInfo* t,
                          uint32_t n_experts, std::vector<uint8_t>& staging,
                          std::vector<void*>& owned, std::string& err) {
    QuantPtr out;
    if (!t) { err = "tensor not found"; return out; }
    const size_t bs = (t->dtype == DType::kQ4_K) ? sizeof(block_q4_K)
                    : (t->dtype == DType::kQ6_K) ? sizeof(block_q6_K) : 0;
    if (bs == 0 || n_experts == 0 ||
        t->nbytes % (uint64_t(n_experts) * bs) != 0) {
        err = "soa repack: unexpected tensor dtype/geometry";
        return out;
    }
    const uint64_t nb_e = t->nbytes / n_experts / bs;
    staging.resize(t->nbytes);
    if (t->dtype == DType::kQ4_K) {
        repack_moe_q4k_soa_host(static_cast<const uint8_t*>(t->data),
                                staging.data(), n_experts, nb_e);
    } else {
        repack_moe_q6k_soa_host(static_cast<const uint8_t*>(t->data),
                                staging.data(), n_experts, nb_e);
    }
    void* d = alloc.malloc(t->nbytes);
    if (!d) { err = "malloc failed"; return out; }
    alloc.queue().memcpy(d, staging.data(), t->nbytes).wait();
    owned.push_back(d);
    out.p = d;
    out.dt = t->dtype;
    return out;
}

// Dispatch GEMV by quant dtype.
sycl::event gemv_q(sycl::queue& q, const sycl::half* A, QuantPtr W,
                   sycl::half* y, uint32_t K, uint32_t N,
                   const std::vector<sycl::event>& deps = {}) {
    if (W.dt == DType::kQ4_K) return gemv_q4_K(q, A, W.p, y, K, N, deps);
    if (W.dt == DType::kQ6_K) return gemv_q6_K(q, A, W.p, y, K, N, deps);
    // F16 weight (transposed [K,N] at load): reuse the existing dense fp16 GEMV.
    // gemv_q_T (T>1 fallback loop) and gemv_q_i8 both delegate here, so this one
    // line also covers prefill and the int-dot decode fallback for F16 weights.
    if (W.dt == DType::kF16)
        return gemv_fp16(q, A, static_cast<const sycl::half*>(W.p), y, K, N, deps);
    return {};
}

// P1a (2026-06-09): integer-dot decode dispatch.  `x_q8` (when non-null)
// holds quantize_q8_1 of the SAME vector as `x` — caller quantizes once per
// unique activation vector and reuses it across that vector's GEMVs.  Q4_K
// routes through the dp4a GEMV; other dtypes fall back to the fp16 path.
// Opt-in while being gated: IE_Q8_DECODE=1.
inline bool q8_decode_enabled() {
    // v1.5-C: default ON (quality-verified; quant rides the attn_norm
    // launch).  Opt out with IE_NO_Q8_DECODE=1.
    static const bool off = std::getenv("IE_NO_Q8_DECODE") != nullptr;
    return !off;
}
sycl::event gemv_q_i8(sycl::queue& q, const sycl::half* x, const void* x_q8,
                      QuantPtr W, sycl::half* y, uint32_t K, uint32_t N) {
    if (x_q8 && W.dt == DType::kQ4_K)
        return gemv_q4_K_q8(q, x_q8, W.p, y, K, N);
    return gemv_q(q, x, W, y, K, N);
}
// Multi-token: dispatch the best multi-row kernel we have for this dtype.
// For Q4_K we use gemm_q4_K with M_TILE=8 chunks (amortizes the weight-read
// cost 8× across rows). For other dtypes (Q6_K) we fall back to a serial
// gemv_q loop until those grow a multi-row variant.
sycl::event gemv_q_T(sycl::queue& q, const sycl::half* A, QuantPtr W,
                     sycl::half* y, uint32_t K, uint32_t N, uint32_t T,
                     const std::vector<sycl::event>& deps = {}) {
    if (T == 0) return {};
    if (T == 1) return gemv_q(q, A, W, y, K, N, deps);
    // E1 (docs/prefill_attack_plan_2026-06-09.md): at prefill sizes, dequant
    // the whole weight matrix to an fp16 scratch once and run the dense
    // gemm_fp16 instead of the fused Q4_K/Q6_K XMX path.  The fused path
    // re-streams quant blocks per 16-row slice at ~53 GB/s effective; the
    // dequant+dense route matches Intel's B70 finding in llama.cpp PR #22147.
    // Kill switch for A/B benching: IE_NO_DEQ_FP16=1.
    static const bool deq_fp16_disabled = std::getenv("IE_NO_DEQ_FP16") != nullptr;
    if (!deq_fp16_disabled && T >= 64 &&
        (W.dt == DType::kQ4_K || W.dt == DType::kQ6_K) &&
        (N % 64 == 0) && (K % 256 == 0)) {
        // Process-lifetime device scratch, grown on demand.  Host is
        // single-threaded and the queue is in-order, so reuse across
        // back-to-back projections is race-free.
        static sycl::half* bt_scr = nullptr;  static uint64_t bt_cap = 0;
        static float*      c_scr  = nullptr;  static uint64_t c_cap  = 0;
        const uint64_t bt_need = uint64_t(K) * N;
        // +8 rows: gemm_fp16's joint_matrix_store writes full TM=8 row tiles
        // whose origin is < M, so the last tile may overrun by up to 7 rows.
        const uint64_t c_need = (uint64_t(T) + 8) * N;
        if (bt_need > bt_cap) {
            if (bt_scr) sycl::free(bt_scr, q);
            bt_scr = sycl::malloc_device<sycl::half>(bt_need, q);
            bt_cap = bt_scr ? bt_need : 0;
        }
        if (c_need > c_cap) {
            if (c_scr) sycl::free(c_scr, q);
            c_scr = sycl::malloc_device<float>(c_need, q);
            c_cap = c_scr ? c_need : 0;
        }
        if (bt_scr && c_scr) {
            sycl::event e = (W.dt == DType::kQ4_K)
                ? dequant_q4_K_to_Bt(q, W.p, bt_scr, K, N, deps)
                : dequant_q6_K_to_Bt(q, W.p, bt_scr, K, N, deps);
            // P1b (2026-06-10): oneDNN matmul tried — measured SLOWER than
            // our gemm_fp16 pipeline at these shapes (929-931 vs 955 pp512
            // first pass; see docs/benchmark_matrix_2026-06-09.md).  Kept as
            // opt-IN (IE_ONEDNN=1) for future study; llama.cpp's prefill
            // lead is not explained by the GEMM library alone.
            static const bool use_onednn = std::getenv("IE_ONEDNN") != nullptr;
            if (use_onednn)
                return gemm_fp16_onednn(q, A, bt_scr, y, T, N, K, {e});
            e = gemm_fp16(q, A, bt_scr, c_scr, T, N, K, {e});
            return cast_fp32_to_fp16(q, c_scr, y, uint64_t(T) * N, {e});
        }
        // scratch allocation failed → fall through to the fused path
    }
    if (W.dt == DType::kQ4_K) {
        // M_TILE for the XMX path — bumped 8 → 16 (2026-05-05) when
        // gemm_q4_K_xmx grew M_GROUPS_MAX=2 (TM=8 stacked twice).  Each
        // 16-row launch reuses the same B_smem dequant for both 8-row
        // groups, so weight-read amortization doubles vs M_TILE=8.
        // Scalar fallback (gemm_q4_K) still has M_TILE=8 (its own cap).
        constexpr uint32_t M_TILE_XMX = 16;
        static const bool xmx_disabled = std::getenv("IE_NO_XMX") != nullptr;
        const bool use_xmx = !xmx_disabled && (N % 64 == 0) && (K % 256 == 0);
        // E2b: scalar gemm_q4_K now tiles M in-kernel (grid dim 0) — one
        // launch for the whole T instead of T/32 underoccupied launches.
        if (!use_xmx) return gemm_q4_K(q, A, W.p, y, T, K, N, deps);
        sycl::event last;
        for (uint32_t m = 0; m < T; m += M_TILE_XMX) {
            const uint32_t mc = std::min(M_TILE_XMX, T - m);
            const sycl::half* a_t = A + uint64_t(m) * K;
            sycl::half*       y_t = y + uint64_t(m) * N;
#if IE_ENABLE_GEMM_Q4K_ESIMD
            last = (m == 0) ? gemm_q4_K_esimd(q, a_t, W.p, y_t, mc, K, N, deps)
                            : gemm_q4_K_esimd(q, a_t, W.p, y_t, mc, K, N);
#else
            last = (m == 0) ? gemm_q4_K_xmx(q, a_t, W.p, y_t, mc, K, N, deps)
                            : gemm_q4_K_xmx(q, a_t, W.p, y_t, mc, K, N);
#endif
        }
        return last;
    }
    if (W.dt == DType::kQ6_K) {
        constexpr uint32_t M_TILE_XMX    = 16;
        constexpr uint32_t M_TILE_SCALAR = 32;
        static const bool xmx_disabled = std::getenv("IE_NO_XMX") != nullptr;
        const bool use_xmx = !xmx_disabled && (N % 64 == 0) && (K % 256 == 0);
        const uint32_t M_TILE = use_xmx ? M_TILE_XMX : M_TILE_SCALAR;
        sycl::event last;
        for (uint32_t m = 0; m < T; m += M_TILE) {
            const uint32_t mc = std::min(M_TILE, T - m);
            const sycl::half* a_t = A + uint64_t(m) * K;
            sycl::half*       y_t = y + uint64_t(m) * N;
            if (use_xmx) {
                last = (m == 0) ? gemm_q6_K_xmx(q, a_t, W.p, y_t, mc, K, N, deps)
                                : gemm_q6_K_xmx(q, a_t, W.p, y_t, mc, K, N);
            } else {
                last = (m == 0) ? gemm_q6_K(q, a_t, W.p, y_t, mc, K, N, deps)
                                : gemm_q6_K(q, a_t, W.p, y_t, mc, K, N);
            }
        }
        return last;
    }
    // Other dtype fallback: scalar per-row loop.
    sycl::event last;
    for (uint32_t t = 0; t < T; ++t) {
        const sycl::half* a_t = A + uint64_t(t) * K;
        sycl::half*       y_t = y + uint64_t(t) * N;
        last = (t == 0) ? gemv_q(q, a_t, W, y_t, K, N, deps)
                        : gemv_q(q, a_t, W, y_t, K, N);
    }
    return last;
}
inline size_t bytes_per_block_for(DType dt) {
    switch (dt) {
        case DType::kQ4_K: return sizeof(block_q4_K);
        case DType::kQ6_K: return sizeof(block_q6_K);
        default: return 0;
    }
}

}  // namespace

QwenModel::~QwenModel() {
    if (alloc_) for (void* p : owned_) alloc_->free(p);
}

std::string QwenModel::load(DeviceAllocator& alloc, const GgufReader& g, const QwenConfig& cfg) {
    alloc_ = &alloc;
    cfg_ = cfg;

    char buf[64];
    auto T = [&](const char* name) {
        return g.find_tensor(name);
    };
    auto Tlayer = [&](int L, const char* name) {
        std::snprintf(buf, sizeof(buf), "blk.%d.%s", L, name);
        return g.find_tensor(buf);
    };

    std::string err;

    // Top-level — token_embd and output may be either Q4_K or Q6_K.
    {
        const auto* ti = T("token_embd.weight");
        if (!ti) return "token_embd: not found";
        if (ti->dtype != DType::kQ4_K && ti->dtype != DType::kQ6_K &&
            ti->dtype != DType::kQ8_0) {   // Q8_0: unsloth MTP GGUF embd
            return "token_embd: unsupported dtype";
        }
        token_embd_dtype_ = ti->dtype;
        token_embd_ = upload<void>(alloc, ti, owned_, err, ti->dtype);
        if (!err.empty()) return "token_embd: " + err;
    }
    output_norm_ = upload<float>(alloc, T("output_norm.weight"), owned_, err, DType::kF32);
    if (!err.empty()) return "output_norm: " + err;
    {
        const auto* ti = T("output.weight");
        if (!ti) return "output: not found";
        if (ti->dtype != DType::kQ4_K && ti->dtype != DType::kQ6_K) {
            return "output: unsupported dtype";
        }
        output_dtype_ = ti->dtype;
        output_ = upload<void>(alloc, ti, owned_, err, ti->dtype);
        if (!err.empty()) return "output: " + err;
    }

    // Prefill-crown: decide the MoE expert-tensor layout globally before any
    // upload.  All layers must carry the fused-path dtypes (gate/up Q4_K,
    // down Q4_K/Q6_K) — they always do for Q4_K_M — otherwise fall back to
    // AoS everywhere so the non-fused dispatch paths stay correct.
    // IE_NO_MOE_SOA=1 forces AoS (order-controlled A/B without rebuild).
    // IE_ENABLE_MOE_DOWN_TILE=0 builds keep AoS too (v1 down kernels are
    // AoS-only).
    moe_exps_soa_ = IE_ENABLE_MOE_DOWN_TILE &&
                    std::getenv("IE_NO_MOE_SOA") == nullptr;
    for (uint32_t L = 0; L < cfg.n_layers && moe_exps_soa_; ++L) {
        const auto* tg = Tlayer(L, "ffn_gate_exps.weight");
        const auto* tu = Tlayer(L, "ffn_up_exps.weight");
        const auto* td = Tlayer(L, "ffn_down_exps.weight");
        if (!tg || !tu || !td ||
            tg->dtype != DType::kQ4_K || tu->dtype != DType::kQ4_K ||
            (td->dtype != DType::kQ4_K && td->dtype != DType::kQ6_K)) {
            moe_exps_soa_ = false;
        }
    }
    std::vector<uint8_t> soa_staging;   // host repack scratch, reused per tensor

    // Per-layer
    layers_.assign(cfg.n_layers, {});
    for (uint32_t L = 0; L < cfg.n_layers; ++L) {
        auto& w = layers_[L];
        const bool full_attn = (L % cfg.full_attn_interval == cfg.full_attn_interval - 1);

        // Always present
        w.attn_norm        = upload<float>(alloc, Tlayer(L, "attn_norm.weight"),       owned_, err, DType::kF32);
        if (!err.empty()) return std::string("layer ") + std::to_string(L) + " attn_norm: " + err;
        w.post_attn_norm   = upload<float>(alloc, Tlayer(L, "post_attention_norm.weight"), owned_, err, DType::kF32);
        if (!err.empty()) return std::string("layer ") + std::to_string(L) + " post_attn_norm: " + err;

        auto LQ = [&](const char* name) -> QuantPtr {
            return upload_quant(alloc, Tlayer(L, name), owned_, err);
        };

        // MoE FFN (every layer)
        w.ffn_gate_inp        = upload<float>(alloc, Tlayer(L, "ffn_gate_inp.weight"),       owned_, err, DType::kF32);
        if (!err.empty()) return std::string("layer ") + std::to_string(L) + " ffn_gate_inp: " + err;
        w.ffn_gate_inp_shexp  = upload<float>(alloc, Tlayer(L, "ffn_gate_inp_shexp.weight"), owned_, err, DType::kF32);
        if (!err.empty()) return std::string("layer ") + std::to_string(L) + " ffn_gate_inp_shexp: " + err;
        // Expert banks: per-expert SoA repack when enabled (see prescan above).
        auto LQE = [&](const char* name) -> QuantPtr {
            if (moe_exps_soa_)
                return upload_quant_soa(alloc, Tlayer(L, name), cfg.n_experts,
                                        soa_staging, owned_, err);
            return upload_quant(alloc, Tlayer(L, name), owned_, err);
        };
        w.ffn_gate_exps       = LQE("ffn_gate_exps.weight");     if (!err.empty()) return std::string("layer ") + std::to_string(L) + " ffn_gate_exps: " + err;
        w.ffn_up_exps         = LQE("ffn_up_exps.weight");       if (!err.empty()) return std::string("layer ") + std::to_string(L) + " ffn_up_exps: " + err;
        w.ffn_down_exps       = LQE("ffn_down_exps.weight");     if (!err.empty()) return std::string("layer ") + std::to_string(L) + " ffn_down_exps: " + err;
        w.ffn_gate_shexp      = LQ("ffn_gate_shexp.weight");     if (!err.empty()) return std::string("layer ") + std::to_string(L) + " ffn_gate_shexp: " + err;
        w.ffn_up_shexp        = LQ("ffn_up_shexp.weight");       if (!err.empty()) return std::string("layer ") + std::to_string(L) + " ffn_up_shexp: " + err;
        w.ffn_down_shexp      = LQ("ffn_down_shexp.weight");     if (!err.empty()) return std::string("layer ") + std::to_string(L) + " ffn_down_shexp: " + err;

        if (full_attn) {
            w.attn_q       = LQ("attn_q.weight");      if (!err.empty()) return std::string("layer ") + std::to_string(L) + " attn_q: " + err;
            w.attn_k       = LQ("attn_k.weight");      if (!err.empty()) return std::string("layer ") + std::to_string(L) + " attn_k: " + err;
            w.attn_v       = LQ("attn_v.weight");      if (!err.empty()) return std::string("layer ") + std::to_string(L) + " attn_v: " + err;
            w.attn_output  = LQ("attn_output.weight"); if (!err.empty()) return std::string("layer ") + std::to_string(L) + " attn_output: " + err;
            w.attn_q_norm  = upload<float>(alloc, Tlayer(L, "attn_q_norm.weight"), owned_, err, DType::kF32);
            if (!err.empty()) return std::string("layer ") + std::to_string(L) + " attn_q_norm: " + err;
            w.attn_k_norm  = upload<float>(alloc, Tlayer(L, "attn_k_norm.weight"), owned_, err, DType::kF32);
            if (!err.empty()) return std::string("layer ") + std::to_string(L) + " attn_k_norm: " + err;
        } else {
            w.attn_qkv     = LQ("attn_qkv.weight");    if (!err.empty()) return std::string("layer ") + std::to_string(L) + " attn_qkv: " + err;
            w.attn_gate    = LQ("attn_gate.weight");   if (!err.empty()) return std::string("layer ") + std::to_string(L) + " attn_gate: " + err;
            w.ssm_a        = upload<float>(alloc, Tlayer(L, "ssm_a"),              owned_, err, DType::kF32);
            if (!err.empty()) return std::string("layer ") + std::to_string(L) + " ssm_a: " + err;
            w.ssm_alpha    = LQ("ssm_alpha.weight");   if (!err.empty()) return std::string("layer ") + std::to_string(L) + " ssm_alpha: " + err;
            w.ssm_beta     = LQ("ssm_beta.weight");    if (!err.empty()) return std::string("layer ") + std::to_string(L) + " ssm_beta: " + err;
            w.ssm_conv1d   = upload<float>(alloc, Tlayer(L, "ssm_conv1d.weight"),  owned_, err, DType::kF32);
            if (!err.empty()) return std::string("layer ") + std::to_string(L) + " ssm_conv1d: " + err;
            w.ssm_dt_bias  = upload<float>(alloc, Tlayer(L, "ssm_dt.bias"),        owned_, err, DType::kF32);
            if (!err.empty()) return std::string("layer ") + std::to_string(L) + " ssm_dt_bias: " + err;
            w.ssm_norm     = upload<float>(alloc, Tlayer(L, "ssm_norm.weight"),    owned_, err, DType::kF32);
            if (!err.empty()) return std::string("layer ") + std::to_string(L) + " ssm_norm: " + err;
            // FP16 casts of weights that the kernels expect in FP16.
            {
                const auto* ti = Tlayer(L, "ssm_conv1d.weight");
                const uint64_t n = ti->shape[0] * ti->shape[1];
                w.ssm_conv1d_fp16 = static_cast<sycl::half*>(alloc.malloc(n * sizeof(sycl::half)));
                if (!w.ssm_conv1d_fp16) return "ssm_conv1d_fp16 alloc";
                owned_.push_back(w.ssm_conv1d_fp16);
                cast_fp32_to_fp16(alloc.queue(), w.ssm_conv1d, w.ssm_conv1d_fp16, n).wait();
            }
            {
                const uint64_t n = cfg.ssm_head_dim;
                w.ssm_norm_fp16 = static_cast<sycl::half*>(alloc.malloc(n * sizeof(sycl::half)));
                if (!w.ssm_norm_fp16) return "ssm_norm_fp16 alloc";
                owned_.push_back(w.ssm_norm_fp16);
                cast_fp32_to_fp16(alloc.queue(), w.ssm_norm, w.ssm_norm_fp16, n).wait();
            }
            w.ssm_out      = LQ("ssm_out.weight");     if (!err.empty()) return std::string("layer ") + std::to_string(L) + " ssm_out: " + err;
        }
    }
    return {};
}

std::string QwenModel::ensure_workspace(uint32_t max_T) {
    if (max_T == 0 || !alloc_) return "alloc not initialized";
    if (max_T <= ws_T_) return {};

    // Free old (re-add to owned_ for cleanup at dtor).
    auto alloc_half = [&](size_t n) {
        auto* p = static_cast<sycl::half*>(alloc_->malloc(n * sizeof(sycl::half)));
        if (p) owned_.push_back(p);
        return p;
    };
    auto alloc_f32  = [&](size_t n) {
        auto* p = static_cast<float*>(alloc_->malloc(n * sizeof(float)));
        if (p) owned_.push_back(p);
        return p;
    };
    auto alloc_i32  = [&](size_t n) {
        auto* p = static_cast<int32_t*>(alloc_->malloc(n * sizeof(int32_t)));
        if (p) owned_.push_back(p);
        return p;
    };

    const uint32_t H   = cfg_.hidden;
    const uint32_t N_q = cfg_.n_q_heads * cfg_.head_dim;     // 4096
    const uint32_t N_qg = N_q * 2;                            // 8192 (Q+gate fold)
    const uint32_t N_kv = cfg_.n_kv_heads * cfg_.head_dim;   // 512
    const uint32_t SI  = cfg_.ssm_inner;                      // 4096
    const uint32_t SI2 = SI * 2;                              // 8192 (Q|K|V fused for DeltaNet — actually Q+K+V = 2048+2048+4096 = 8192)
    const uint32_t SVH = cfg_.ssm_n_v_heads;                  // 32
    const uint32_t SHD = cfg_.ssm_head_dim;                   // 128
    const uint32_t E_ffn = cfg_.expert_ffn;                   // 512

    ws_x_              = alloc_half(max_T * H);
    ws_x_residual_     = alloc_half(max_T * H);
    ws_x_normed_       = alloc_half(max_T * H);
    ws_q_full_         = alloc_half(max_T * N_qg);
    ws_q_split_        = alloc_half(max_T * N_q);
    ws_gate_split_     = alloc_half(max_T * N_q);
    ws_k_              = alloc_half(max_T * N_kv);
    ws_v_              = alloc_half(max_T * N_kv);
    ws_attn_out_       = alloc_half(max_T * N_q);
    ws_attn_block_     = alloc_half(max_T * H);
    ws_positions_      = alloc_i32 (max_T);

    ws_qkv_            = alloc_half(max_T * SI2);
    ws_qkv_silu_       = alloc_half(max_T * SI2);
    ws_alpha_fp16_     = alloc_half(max_T * SVH);
    ws_beta_fp16_      = alloc_half(max_T * SVH);
    ws_z_fp16_         = alloc_half(max_T * SI);
    ws_q_fp32_         = alloc_f32 (max_T * SVH * SHD);
    ws_k_fp32_         = alloc_f32 (max_T * SVH * SHD);
    ws_v_fp32_         = alloc_f32 (max_T * SVH * SHD);
    ws_q_fp32_pre_     = alloc_f32 (max_T * cfg_.ssm_n_k_heads * SHD);
    ws_k_fp32_pre_     = alloc_f32 (max_T * cfg_.ssm_n_k_heads * SHD);
    ws_a_fp32_         = alloc_f32 (max_T * SVH);
    ws_b_fp32_         = alloc_f32 (max_T * SVH);
    ws_g_fp32_         = alloc_f32 (max_T * SVH);
    ws_beta_fp32_      = alloc_f32 (max_T * SVH);
    ws_recurrence_out_ = alloc_f32 (max_T * SVH * SHD);
    ws_gated_norm_     = alloc_half(max_T * SI);

    ws_topk_idx_       = alloc_i32 (max_T * cfg_.experts_topk);
    ws_topk_w_         = alloc_half(max_T * cfg_.experts_topk);
    ws_gate_o_         = alloc_half(max_T * E_ffn);
    ws_up_o_           = alloc_half(max_T * E_ffn);
    ws_h_o_            = alloc_half(max_T * E_ffn);
    ws_h_routed_       = alloc_half(uint64_t(cfg_.experts_topk) * E_ffn);
    ws_moe_x_packed_   = alloc_half(max_T * H);
    ws_moe_token_idx_  = alloc_i32 (max_T * cfg_.experts_topk);
    ws_moe_token_w_    = alloc_half(max_T * cfg_.experts_topk);
    {
        auto* p = static_cast<uint32_t*>(alloc_->malloc(uint64_t(cfg_.n_experts + 1) * sizeof(uint32_t)));
        if (p) owned_.push_back(p);
        ws_moe_expert_offsets_ = p;
    }
    // Cap MoE-prefill scratch at 4k tokens (typical prefill chunk; bigger
    // prefills fall through to the fused-per-token path). The large-M long-ctx
    // lever raises this (IE_QWEN36_MOE_PREFILL_MAXT) so a big-T prefill chunk
    // runs the FUSED multi-expert path at large rows/expert (M=T·K/E) instead
    // of the per-token serial fallback — the regime where the MoE GEMM amortizes
    // the 256-expert weight read once over many rows. [2026-06-26]
    // The long-ctx oneDNN-MoE lever (1.48× prefill, decode-safe) is DEFAULT-ON for
    // long-ctx configs (max_T ≥ 8192) — opt-out `IE_QWEN36_NO_MOE_ONEDNN`. Short-ctx
    // configs (max_T < 8192) CANNOT engage it (the prefill chunk is capped at max_T,
    // below the T≥4096 oneDNN gate's useful regime) so they allocate nothing extra
    // and behave EXACTLY as before — the +~0.7 GB scratch/buffers only land when a
    // long context is configured. Explicit `IE_QWEN36_MOE_ONEDNN` force-enables even
    // below 8192 (for testing); `IE_QWEN36_MOE_PREFILL_MAXT` still overrides the cap.
    const bool moe_onednn_optout = std::getenv("IE_QWEN36_NO_MOE_ONEDNN") != nullptr;
    const bool moe_onednn_force  = std::getenv("IE_QWEN36_MOE_ONEDNN") != nullptr;
    const bool moe_onednn_on = !moe_onednn_optout && (moe_onednn_force || max_T >= 8192u);
    uint32_t moe_scratch_cap = moe_onednn_on ? 8192u : 4096u;
    if (const char* e = std::getenv("IE_QWEN36_MOE_PREFILL_MAXT")) {
        int v = std::atoi(e); if (v >= 256) moe_scratch_cap = uint32_t(v);
    }
    const uint32_t max_T_moe = std::min<uint32_t>(max_T, moe_scratch_cap);
    ws_T_moe_           = max_T_moe;
    ws_moe_h_packed_    = alloc_half(uint64_t(max_T_moe) * cfg_.experts_topk * E_ffn);
    ws_moe_out_packed_  = alloc_half(uint64_t(max_T_moe) * cfg_.experts_topk * H);
    // E5: expert-sorted gathered activations for prefill stage 1.
    ws_moe_xp_          = alloc_half(uint64_t(max_T_moe) * cfg_.experts_topk * H);
    // oneDNN large-M MoE-prefill scratch (only when the lever is on): a separate
    // up-projection output (the int-dot path fuses gate+up+silu in one kernel; the
    // oneDNN path needs gate and up materialized for swiglu) + a per-expert fp16
    // Bt dequant target reused across experts. [2026-06-26]
    if (moe_onednn_on) {
        ws_moe_up_packed_ = alloc_half(uint64_t(max_T_moe) * cfg_.experts_topk * E_ffn);
        ws_moe_btf16_     = alloc_half(uint64_t(H) * E_ffn);
    }
    // P1b-2 / prefill-crown: Q8_1s stream over ws_moe_xp_ (48 B per 32
    // elems, split half-sums) + Q8_1s stream over the stage-1 output
    // h_packed for the int-dot stage-2 down kernels.
    {
        void* p = alloc_->malloc(uint64_t(max_T_moe) * cfg_.experts_topk *
                                 (H / 32) * sizeof(block_q8_1s));
        if (p) owned_.push_back(p);
        ws_moe_xp_q8_ = p;
    }
    {
        void* p = alloc_->malloc(uint64_t(max_T_moe) * cfg_.experts_topk *
                                 (E_ffn / 32) * sizeof(block_q8_1s));
        if (p) owned_.push_back(p);
        ws_moe_h_q8_ = p;
    }
    // P1a: Q8_1 activation scratch for integer-dot decode GEMVs.  Sized for
    // the largest decode GEMV input (K = SI = 4096 → 128 blocks).
    {
        void* p = alloc_->malloc((uint64_t(std::max(H, SI)) / 32) *
                                 sizeof(block_q8_1x));
        if (p) owned_.push_back(p);
        ws_q8_ = p;
    }
    {
        auto* p = static_cast<uint32_t*>(alloc_->malloc(
            uint64_t(max_T_moe) * cfg_.experts_topk * sizeof(uint32_t)));
        if (p) owned_.push_back(p);
        ws_moe_tk_to_packed_ = p;
    }
    ws_moe_y_fp32_     = alloc_f32 (uint64_t(max_T_moe) * H);  // unused in new path
    ws_eo_             = alloc_half(max_T * H);
    ws_sh_g_           = alloc_half(max_T);

    if (!ws_x_ || !ws_q_full_ || !ws_qkv_ || !ws_recurrence_out_ || !ws_gate_o_) {
        return "workspace allocation failed";
    }
    ws_T_ = max_T;
    return {};
}

void QwenModel::set_profile(bool b) {
    profile_enabled_ = b;
    if (b) profile_totals_.clear();
}

void QwenModel::clear_profile() {
    profile_totals_.clear();
}

void QwenModel::dump_profile(std::FILE* out) const {
    if (profile_totals_.empty()) {
        std::fprintf(out, "(profile empty)\n");
        return;
    }
    // Aggregate by section name.
    std::vector<std::pair<std::string, double>> sorted = profile_totals_;
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b){ return a.second > b.second; });
    double total = 0.0;
    for (auto& p : sorted) total += p.second;
    std::fprintf(out, "  per-section ms (sorted desc, total = %.3f ms):\n", total);
    for (auto& [name, ms] : sorted) {
        std::fprintf(out, "    %-40s  %8.3f ms  (%5.1f%%)\n",
                    name.c_str(), ms, 100.0 * ms / total);
    }
}


std::string QwenModel::ensure_attn_partials(uint32_t max_ctx) {
    if (!alloc_) return "alloc not initialized";
    if (max_ctx <= ws_attn_partials_ctx_) return {};
    // Conservative: size for the smallest Bc the FA-2 kernel could use.
    // Currently 64 is the floor; if the kernel grows to Bc=128 the partials
    // only get smaller, so this is always safe.
    constexpr uint32_t Bc_floor = 64;
    const uint32_t n_chunks_max = (max_ctx + Bc_floor - 1) / Bc_floor;
    const uint64_t n_floats =
        uint64_t(n_chunks_max) * cfg_.n_q_heads * (cfg_.head_dim + 2);
    auto* p = static_cast<float*>(alloc_->malloc(n_floats * sizeof(float)));
    if (!p) return "attn_partials alloc failed";
    owned_.push_back(p);
    ws_attn_partials_     = p;
    ws_attn_partials_ctx_ = max_ctx;
    return {};
}

void QwenModel::set_partial_logits_capture(std::vector<uint32_t> cut_layers,
                                           std::vector<sycl::half*> per_cut_out_logits) {
    if (cut_layers.size() != per_cut_out_logits.size()) {
        std::fprintf(stderr,
                     "set_partial_logits_capture: cut_layers.size()=%zu != "
                     "per_cut_out_logits.size()=%zu\n",
                     cut_layers.size(), per_cut_out_logits.size());
        return;
    }
    // Sort cuts ascending; reorder out_logits to match.
    std::vector<size_t> order(cut_layers.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](size_t a, size_t b){ return cut_layers[a] < cut_layers[b]; });
    std::vector<uint32_t>     sc(cut_layers.size());
    std::vector<sycl::half*>  so(cut_layers.size());
    for (size_t i = 0; i < order.size(); ++i) {
        sc[i] = cut_layers[order[i]];
        so[i] = per_cut_out_logits[order[i]];
    }
    partial_cuts_       = std::move(sc);
    partial_out_logits_ = std::move(so);
    // ws_partial_normed_ allocated lazily on first forward() (needs alloc_).
}

void QwenModel::clear_partial_logits_capture() {
    partial_cuts_.clear();
    partial_out_logits_.clear();
}

sycl::event QwenModel::forward(sycl::queue& q,
                               const int32_t* input_ids, uint32_t T, uint32_t start_pos,
                               KvCache& kv, DeltaNetState& dn,
                               sycl::half* out_logits,
                               sycl::half* all_logits,
                               sycl::half* hidden_pre_norm,
                               Qwen35SpecCheckpoint* ckpt) {
    if (T == 0) return {};
    // Spec-decode checkpoint mode: per-position DeltaNet snapshots for rollback-
    // free commit. Sized for ≥ T positions. Default-off → normal path untouched.
    const bool ckpt_mode = (ckpt != nullptr && ckpt->K >= T &&
                            ckpt->ckpt_state && ckpt->ckpt_conv);

    if (ws_T_ < T) {
        auto e = ensure_workspace(T);
        if (!e.empty()) {
            std::fprintf(stderr, "ensure_workspace: %s\n", e.c_str());
            return {};
        }
    }

    // Lazy-alloc partial-logits scratch (PR #1, observational; vocab*1 fp16).
    if (!partial_cuts_.empty() && ws_partial_normed_ == nullptr && alloc_) {
        auto* p = static_cast<sycl::half*>(alloc_->malloc(cfg_.hidden * sizeof(sycl::half)));
        if (p) { owned_.push_back(p); ws_partial_normed_ = p; }
    }

    const uint32_t H   = cfg_.hidden;
    const uint32_t H_d = cfg_.head_dim;
    const uint32_t N_q = cfg_.n_q_heads * H_d;          // 4096
    const uint32_t N_qg = N_q * 2;                       // 8192
    const uint32_t N_kv = cfg_.n_kv_heads * H_d;         // 512
    const uint32_t SI  = cfg_.ssm_inner;                 // 4096
    const uint32_t SVH = cfg_.ssm_n_v_heads;             // 32
    const uint32_t SHD = cfg_.ssm_head_dim;              // 128
    const uint32_t SKH = cfg_.ssm_n_k_heads;             // 16
    const uint32_t E_ffn = cfg_.expert_ffn;              // 512
    const uint32_t E   = cfg_.n_experts;                 // 256
    const uint32_t K_top = cfg_.experts_topk;            // 8

    // (per-expert byte strides are derived per-layer from each tensor's dtype below)

    // Materialize positions on device (int32 array [T] = start_pos..start_pos+T-1).
    // v1.5-B: at decode this was a host-BLOCKING memcpy+wait every token —
    // the .wait() drained the whole in-order queue before the host could
    // submit the next token's kernels.  T==1 writes the scalar with a fill
    // (no host buffer, no sync); T>1 keeps the synchronous upload (once per
    // prefill chunk, amortized).
    if (T == 1) {
        q.fill(ws_positions_, int32_t(start_pos), 1);
    } else {
        std::vector<int32_t> pos(T);
        for (uint32_t t = 0; t < T; ++t) pos[t] = int32_t(start_pos + t);
        q.memcpy(ws_positions_, pos.data(), T * sizeof(int32_t)).wait();
    }

    // 1. Embedding lookup → ws_x_
    if (token_embd_dtype_ == DType::kQ4_K) {
        embedding_lookup_q4k(q, input_ids, token_embd_, ws_x_, T, H);
    } else if (token_embd_dtype_ == DType::kQ8_0) {
        embedding_lookup_q8_0(q, input_ids, token_embd_, ws_x_, T, H);
    } else {
        embedding_lookup_q6k(q, input_ids, token_embd_, ws_x_, T, H);
    }

    // ---- DEBUG: optional residual-stream norm tracing ----
    auto trace_norm = [&](const char* tag) {
        if (!std::getenv("IE_TRACE")) return;
        std::vector<sycl::half> hh(uint64_t(T) * H);
        q.memcpy(hh.data(), ws_x_, hh.size() * sizeof(sycl::half)).wait();
        for (uint32_t t = 0; t < T; ++t) {
            double ss = 0.0;
            float fmax = 0.f, fmin = 0.f;
            int   nans = 0;
            for (uint32_t i = 0; i < H; ++i) {
                const float v = float(hh[t * H + i]);
                if (!std::isfinite(v)) ++nans;
                else { ss += double(v) * v; if (v > fmax) fmax = v; if (v < fmin) fmin = v; }
            }
            std::fprintf(stderr, "  [%s t=%u] L2=%.3f min=%.3f max=%.3f nans=%d\n",
                         tag, t, std::sqrt(ss), fmin, fmax, nans);
        }
    };

    // ---- Per-layer activation dump (opt-in via dump_prefix_) ----
    auto dump_buf = [&](const sycl::half* buf, uint64_t n_elem,
                        const char* tag_fmt, int idx) {
        if (dump_prefix_.empty()) return;
        std::vector<sycl::half> hh(n_elem);
        q.memcpy(hh.data(), buf, n_elem * sizeof(sycl::half)).wait();
        std::vector<float> ff(n_elem);
        for (size_t i = 0; i < n_elem; ++i) ff[i] = float(hh[i]);
        char path[1024];
        std::snprintf(path, sizeof(path), "%s_%s%02d.bin", dump_prefix_.c_str(), tag_fmt, idx);
        if (FILE* fp = std::fopen(path, "wb")) {
            std::fwrite(ff.data(), sizeof(float), n_elem, fp);
            std::fclose(fp);
        }
        std::snprintf(path, sizeof(path), "%s_%s%02d.meta", dump_prefix_.c_str(), tag_fmt, idx);
        if (FILE* fp = std::fopen(path, "w")) {
            std::fprintf(fp, "%u %u\n", T, H);
            std::fclose(fp);
        }
    };
    // Convenience: dump residual stream (ws_x_) as L<NN>.bin
    auto dump_residual = [&](int slot_NN) { dump_buf(ws_x_, uint64_t(T) * H, "L", slot_NN); };
    // Dump attn block output (the contribution before residual_add): A<NN>.bin
    auto dump_attn = [&](int layer) { dump_buf(ws_attn_block_, uint64_t(T) * H, "A", layer); };
    auto trace_buf = [&](const char* tag, const sycl::half* buf, uint64_t n) {
        if (!std::getenv("IE_TRACE")) return;
        std::vector<sycl::half> hh(n);
        q.memcpy(hh.data(), buf, n * sizeof(sycl::half)).wait();
        double ss = 0.0; float fmax = 0.f, fmin = 0.f; int nans = 0;
        for (uint64_t i = 0; i < n; ++i) {
            const float v = float(hh[i]);
            if (!std::isfinite(v)) ++nans;
            else { ss += double(v) * v; if (v > fmax) fmax = v; if (v < fmin) fmin = v; }
        }
        std::fprintf(stderr, "  [%s n=%llu] L2=%.3f min=%.3f max=%.3f nans=%d\n",
                     tag, (unsigned long long)n, std::sqrt(ss), fmin, fmax, nans);
    };
    auto trace_buf_f32 = [&](const char* tag, const float* buf, uint64_t n) {
        if (!std::getenv("IE_TRACE")) return;
        std::vector<float> hh(n);
        q.memcpy(hh.data(), buf, n * sizeof(float)).wait();
        double ss = 0.0; float fmax = 0.f, fmin = 0.f; int nans = 0;
        for (uint64_t i = 0; i < n; ++i) {
            const float v = hh[i];
            if (!std::isfinite(v)) ++nans;
            else { ss += double(v) * v; if (v > fmax) fmax = v; if (v < fmin) fmin = v; }
        }
        std::fprintf(stderr, "  [%s n=%llu] L2=%.3f min=%.3f max=%.3f nans=%d\n",
                     tag, (unsigned long long)n, std::sqrt(ss), fmin, fmax, nans);
    };
    trace_norm("embed");
    dump_residual(0);  // L00 = post-embedding residual

    // ---- Profiling helper.  When profile_enabled_, q.wait() before AND
    // after each section and accumulate by name.  Loses pipelining (so
    // wall-clock will be slower than non-profiled runs) but each section's
    // ms reading is exactly the kernel-level GPU time. ----
    auto profile_section = [&](const char* name, auto&& fn) {
        if (!profile_enabled_) {
            fn();
            return;
        }
        q.wait();
        const double t0 = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        fn();
        q.wait();
        const double t1 = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        // Aggregate by name (linear search; few sections).
        for (auto& p : profile_totals_) {
            if (p.first == name) { p.second += (t1 - t0); return; }
        }
        profile_totals_.emplace_back(name, t1 - t0);
    };

    // 2. 40 layers (clamped via IE_MAX_LAYER for bisection)
    uint32_t n_run = cfg_.n_layers;
    if (auto* s = std::getenv("IE_MAX_LAYER")) n_run = std::min<uint32_t>(n_run, std::atoi(s));
    for (uint32_t L = 0; L < n_run; ++L) {
        const auto& w = layers_[L];
        const bool full_attn = (L % cfg_.full_attn_interval == cfg_.full_attn_interval - 1);

        // 2a. attn_norm (1+w) → ws_x_normed_ (+ Q8_1 emission at decode —
        // v1.5-C: the int-dot GEMVs' activation quant rides this launch).
        const bool x_q8_ready = (T == 1) && q8_decode_enabled() && ws_q8_;
        profile_section("attn_norm", [&]{
            if (x_q8_ready)
                rms_norm_f32w_q8(q, ws_x_, w.attn_norm, ws_x_normed_, ws_q8_,
                                 T, H, cfg_.rms_eps);
            else
                rms_norm_f32w(q, ws_x_, w.attn_norm, ws_x_normed_, T, H, cfg_.rms_eps);
        });

        if (full_attn) {
            profile_section("attn_full_block", [&]{
            const uint32_t full_attn_idx = L / cfg_.full_attn_interval;
            // Within-layer activation dump for the chunking-equivalence
            // bisect (2026-05-02).  Gated to L == 3 (first full-attn layer
            // where divergence first appears per ie-validate-chunking).
            // Files: ${prefix}_<tag>03.bin, [T, per_token_width] fp32.
            // Per-token widths recorded in ${prefix}_<tag>03.meta as "T W".
            const bool dump_intermediate = (!dump_prefix_.empty()) && (L == 3);
            auto dump_int = [&](const sycl::half* buf, uint32_t per_token_w,
                                const char* tag) {
                if (!dump_intermediate) return;
                const uint64_t n_elem = uint64_t(T) * per_token_w;
                std::vector<sycl::half> hh(n_elem);
                q.memcpy(hh.data(), buf, n_elem * sizeof(sycl::half)).wait();
                std::vector<float> ff(n_elem);
                for (size_t i = 0; i < n_elem; ++i) ff[i] = float(hh[i]);
                char path[1024];
                std::snprintf(path, sizeof(path), "%s_%s03.bin",
                              dump_prefix_.c_str(), tag);
                if (FILE* fp = std::fopen(path, "wb")) {
                    std::fwrite(ff.data(), sizeof(float), n_elem, fp);
                    std::fclose(fp);
                }
                std::snprintf(path, sizeof(path), "%s_%s03.meta",
                              dump_prefix_.c_str(), tag);
                if (FILE* fp = std::fopen(path, "w")) {
                    std::fprintf(fp, "%u %u\n", T, per_token_w);
                    std::fclose(fp);
                }
            };

            // P1a/v1.5-C: x_normed's Q8 blocks come from the fused attn_norm
            // launch (non-Q4_K dtypes fall back inside gemv_q_i8).
            const void* xq8 = x_q8_ready ? static_cast<const void*>(ws_q8_) : nullptr;

            // Q proj (Q + gate per-head fold-in): hidden → 8192
            if (xq8) gemv_q_i8(q, ws_x_normed_, xq8, w.attn_q, ws_q_full_, H, N_qg);
            else     gemv_q_T(q, ws_x_normed_, w.attn_q, ws_q_full_, H, N_qg, T);
            dump_int(ws_q_full_, N_qg, "I1qproj_");
            // Split Q | gate per head — HF chunks `[..., n_heads, 2*head_dim]` along last dim.
            split_q_gate_per_head(q, ws_q_full_, ws_q_split_, ws_gate_split_,
                                  T, cfg_.n_q_heads, H_d);
            dump_int(ws_q_split_,    N_q, "I2qsplt_");

            // K, V projs
            if (xq8) {
                gemv_q_i8(q, ws_x_normed_, xq8, w.attn_k, ws_k_, H, N_kv);
                gemv_q_i8(q, ws_x_normed_, xq8, w.attn_v, ws_v_, H, N_kv);
            } else {
                gemv_q_T(q, ws_x_normed_, w.attn_k, ws_k_, H, N_kv, T);
                gemv_q_T(q, ws_x_normed_, w.attn_v, ws_v_, H, N_kv, T);
            }

            // QK-Norm: per-head RMSNorm, head_dim=256, FP32 weight
            rms_norm_f32w(q, ws_q_split_, w.attn_q_norm, ws_q_split_, T * cfg_.n_q_heads, H_d, cfg_.rms_eps);
            rms_norm_f32w(q, ws_k_,        w.attn_k_norm, ws_k_,        T * cfg_.n_kv_heads, H_d, cfg_.rms_eps);
            dump_int(ws_q_split_,    N_q, "I3qnorm_");
            dump_int(ws_k_,         N_kv, "I4knorm_");

            // Partial RoPE on Q and K (first 64 dims only)
            rope_partial(q, ws_q_split_, ws_positions_, ws_q_split_,
                         T, cfg_.n_q_heads, H_d, cfg_.rope_dim, cfg_.rope_theta);
            rope_partial(q, ws_k_, ws_positions_, ws_k_,
                         T, cfg_.n_kv_heads, H_d, cfg_.rope_dim, cfg_.rope_theta);
            dump_int(ws_q_split_,    N_q, "I5qrope_");
            dump_int(ws_k_,         N_kv, "I6krope_");

            // SDPA — kv-cache slice for layer full_attn_idx.
            // T=1 → FA-2 split-K (with INT8 path if the cache is INT8).
            // T>1 prefill → naive path, fp16 cache.
            const uint64_t per_layer = uint64_t(cfg_.n_kv_heads) * kv.config().max_ctx * H_d;
            sycl::half* k_cache = kv.k_ptr() + per_layer * full_attn_idx;
            sycl::half* v_cache = kv.v_ptr() + per_layer * full_attn_idx;
            if (T == 1 && ws_attn_partials_) {
                // Use INT8 path only if the INT8 cache is fully populated up to
                // start_pos — i.e. every prior position was quantized.
                if (kv.is_int8() && kv.k_int8_ptr() &&
                    start_pos == kv.int8_length(full_attn_idx)) {
                    const uint64_t int8_per_layer = uint64_t(cfg_.n_kv_heads) * kv.config().max_ctx * H_d;
                    const uint64_t scale_per_layer = uint64_t(cfg_.n_kv_heads) * kv.config().max_ctx;
                    int8_t*     k_i8 = kv.k_int8_ptr()   + int8_per_layer * full_attn_idx;
                    int8_t*     v_i8 = kv.v_int8_ptr()   + int8_per_layer * full_attn_idx;
                    sycl::half* k_sc = kv.k_scales_ptr() + scale_per_layer * full_attn_idx;
                    sycl::half* v_sc = kv.v_scales_ptr() + scale_per_layer * full_attn_idx;
                    full_attention_fa2_decode_int8(q, ws_q_split_, ws_k_, ws_v_,
                                                   k_i8, v_i8, k_sc, v_sc,
                                                   nullptr, nullptr,
                                                   ws_attn_out_,
                                                   ws_attn_partials_,
                                                   start_pos,
                                                   cfg_.n_q_heads, cfg_.n_kv_heads, H_d,
                                                   kv.config().max_ctx);
                    // Inline quantize-on-write just wrote position start_pos.
                    kv.set_int8_length(full_attn_idx, start_pos + 1);
                } else {
                    full_attention_fa2_decode(q, ws_q_split_, ws_k_, ws_v_,
                                              k_cache, v_cache, ws_attn_out_,
                                              ws_attn_partials_,
                                              start_pos,
                                              cfg_.n_q_heads, cfg_.n_kv_heads, H_d,
                                              kv.config().max_ctx);
                }
            } else {
                // Long-ctx full-attn prefill. Naive full_attention re-reads the whole
                // KV cache T× → collapses at 16K (the worst DeltaNet long-ctx loss,
                // ~0.20× vs llama). Route the head_dim-256 full-attn layers through the
                // proven Gemma wide-tile kernel (KV once per Br query-tile, appends k/v
                // internally, full causal window=0) for the collapsing long-ctx region.
                // Numerically equivalent to naive at hd256 (same 1/sqrt(HD) post-dot
                // scale on the same unscaled ws_q_split_; argmax-bit-identical on Gemma;
                // identical to the 27B port `07ab8ef`). GATED ctx≥minctx (default 6144)
                // so the verified short/mid prefill WIN (1.14×) is UNTOUCHED, and the
                // crown 6.4527 PPL gate is decode-path (T==1) so this prefill change
                // cannot affect it. Opt-out IE_QWEN36_NO_FA2_TILE; tune
                // IE_QWEN36_FA2_TILE_MINCTX. [2026-06-26]
                static const bool no_tile = std::getenv("IE_QWEN36_NO_FA2_TILE") != nullptr;
                static const uint32_t tile_minctx = []() -> uint32_t {
                    const char* e = std::getenv("IE_QWEN36_FA2_TILE_MINCTX");
                    if (!e) return 6144u;
                    int v = std::atoi(e); return v > 0 ? uint32_t(v) : 6144u;
                }();
                if (!no_tile && H_d == 256 && (start_pos + T) >= tile_minctx) {
                    full_attention_fa2_prefill_tile_gemma(
                        q, ws_q_split_, ws_k_, ws_v_, k_cache, v_cache, ws_attn_out_,
                        T, start_pos, cfg_.n_q_heads, cfg_.n_kv_heads, H_d,
                        kv.config().max_ctx, 0 /*window: full causal*/);
                } else
#if IE_ENABLE_FA2_PREFILL_TILED
                full_attention_fa2_prefill(q, ws_q_split_, ws_k_, ws_v_,
                                           k_cache, v_cache, ws_attn_out_,
                                           T, start_pos, cfg_.n_q_heads,
                                           cfg_.n_kv_heads, H_d,
                                           kv.config().max_ctx);
#else
                full_attention(q, ws_q_split_, ws_k_, ws_v_, k_cache, v_cache, ws_attn_out_,
                               T, start_pos, cfg_.n_q_heads, cfg_.n_kv_heads, H_d,
                               kv.config().max_ctx);
#endif
                // Post-quantize fp16 prefill rows into the INT8 shadow so that
                // the next T=1 decode step can use the INT8 path.
                if (kv.is_int8()) {
                    kv.quantize_to_int8(q, full_attn_idx, start_pos, T);
                    // int8_lengths_[full_attn_idx] updated to start_pos+T inside.
                }
            }
            kv.set_length(full_attn_idx, start_pos + T);
            dump_int(ws_attn_out_,   N_q, "I7attno_");

            // Output gate: ws_attn_out_ *= sigmoid(gate)
            sigmoid_gate(q, ws_attn_out_, ws_gate_split_, ws_attn_out_,
                         uint64_t(T) * N_q);
            dump_int(ws_attn_out_,   N_q, "I8sigga_");

            // O proj: [n_q*head=4096] → [hidden=2048]
            // P1a: separate input vector → its own quant (ws_q8_ reuse is
            // safe — the in-order queue serializes past the q/k/v consumers).
            if (T == 1 && q8_decode_enabled() && ws_q8_ &&
                w.attn_output.dt == DType::kQ4_K) {
                quantize_q8_1(q, ws_attn_out_, ws_q8_, N_q);
                gemv_q_i8(q, ws_attn_out_, ws_q8_, w.attn_output, ws_attn_block_, N_q, H);
            } else {
                gemv_q_T(q, ws_attn_out_, w.attn_output, ws_attn_block_, N_q, H, T);
            }
            // (post-O-proj is captured by the existing dump_attn(L) below
            // via the A03.bin file — no separate intermediate dump needed.)
            });  // end profile_section attn_full_block
        } else {
            profile_section("attn_dn_block", [&]{
            const uint32_t dn_idx = L - (L / cfg_.full_attn_interval);   // 0..30
            const uint32_t SI2 = SI * 2;

            // P1a/v1.5-C: x_normed's Q8 blocks were emitted by the fused
            // attn_norm launch; qkv + attn_gate run integer-dot for free.
            const void* xq8 = x_q8_ready ? static_cast<const void*>(ws_q8_) : nullptr;

            profile_section("dn_qkv_proj", [&]{
                if (xq8) gemv_q_i8(q, ws_x_normed_, xq8, w.attn_qkv, ws_qkv_, H, SI2);
                else     gemv_q_T(q, ws_x_normed_, w.attn_qkv, ws_qkv_, H, SI2, T);
            });

            sycl::half* conv_state_layer = dn.conv_state_ptr() +
                uint64_t(dn_idx) * dn.conv_elems_per_layer();
            profile_section("dn_conv1d_silu", [&]{
                if (ckpt_mode) {
                    // T single-token streaming steps; snapshot conv state AFTER
                    // each position. Output byte-identical to the T-batched call
                    // (conv is causal + streaming). conv width = SI2 per position.
                    const uint64_t ce = dn.conv_elems_per_layer();
                    for (uint32_t s = 0; s < T; ++s) {
                        depthwise_conv1d_causal(q, ws_qkv_ + uint64_t(s) * SI2,
                                                w.ssm_conv1d_fp16, conv_state_layer,
                                                ws_qkv_silu_ + uint64_t(s) * SI2,
                                                /*T=*/1, SI2, cfg_.ssm_conv_kernel);
                        q.memcpy(ckpt->ckpt_conv +
                                     (uint64_t(s) * ckpt->n_lin + dn_idx) * ce,
                                 conv_state_layer, ce * sizeof(sycl::half));
                    }
                } else {
                    depthwise_conv1d_causal(q, ws_qkv_, w.ssm_conv1d_fp16,
                                            conv_state_layer, ws_qkv_silu_,
                                            T, SI2, cfg_.ssm_conv_kernel);
                }
            });

            profile_section("dn_alpha_beta_gate", [&]{
            bool gbeta_done = false;
#if IE_FUSE_SSM_AB
                // Fused alpha + beta when both are Q4_K (always true in this
                // model) and T==1 (decode).  T>1 prefill keeps the original
                // 2-call path since it goes through gemm_q4_K, not gemv_q4_K.
                // v1.4: the T==1 path also folds compute_g_beta into the same
                // launch (numerics preserved via an explicit fp16 round).
                if (T == 1 &&
                    w.ssm_alpha.dt == DType::kQ4_K &&
                    w.ssm_beta.dt  == DType::kQ4_K && SVH <= 32) {
                    gemv_q4_K_dual_ssm_gbeta(q, ws_x_normed_,
                                             w.ssm_alpha.p, w.ssm_beta.p,
                                             w.ssm_a, w.ssm_dt_bias,
                                             ws_g_fp32_, ws_beta_fp32_,
                                             H, SVH);
                    gbeta_done = true;
                } else {
                    gemv_q_T(q, ws_x_normed_, w.ssm_alpha, ws_alpha_fp16_, H, SVH, T);
                    gemv_q_T(q, ws_x_normed_, w.ssm_beta,  ws_beta_fp16_,  H, SVH, T);
                }
#else
                gemv_q_T(q, ws_x_normed_, w.ssm_alpha, ws_alpha_fp16_, H, SVH, T);
                gemv_q_T(q, ws_x_normed_, w.ssm_beta,  ws_beta_fp16_,  H, SVH, T);
#endif
                if (xq8) gemv_q_i8(q, ws_x_normed_, xq8, w.attn_gate, ws_z_fp16_, H, SI);
                else     gemv_q_T(q, ws_x_normed_, w.attn_gate, ws_z_fp16_, H, SI, T);
                if (!gbeta_done)
                    compute_g_beta_h16(q, ws_alpha_fp16_, ws_beta_fp16_,
                                   w.ssm_a, w.ssm_dt_bias,
                                   ws_g_fp32_, ws_beta_fp32_, T, SVH);
            });

            profile_section("dn_qkv_split_norm", [&]{
                // v1.4: fused 5-launch cluster (cast/split + 2× tiled repeat
                // + 2× per-head L2 norm) into one kernel — math identical.
                const float qscale = 1.0f / sycl::sqrt(float(SHD));
                dn_qkv_split_norm_fused(q, ws_qkv_silu_,
                                        ws_q_fp32_, ws_k_fp32_, ws_v_fp32_,
                                        T, SKH, SHD, qscale, 1e-6f);
            });

            const uint64_t state_per_layer = uint64_t(SVH) * SHD * SHD;
            float* state_layer = dn.state_ptr() + state_per_layer * dn_idx;

            // Step 17 capture (2026-05-03): dump deltanet_recurrence INPUTS
            // per DN layer when dump_prefix_ is set.  Used by the validator
            // to compare C1 vs C2 inputs at the iter where SSM state first
            // diverges — answers "are inputs deterministic but kernel is not?".
            // Files: ${prefix}_DN{q,k,v,g,b}_<dn_idx>.bin (T * width fp32).
            if (!dump_prefix_.empty()) {
                auto dump_dn_in = [&](const float* buf, uint32_t per_token_w,
                                       const char* tag) {
                    const uint64_t n_elem = uint64_t(T) * per_token_w;
                    std::vector<float> ff(n_elem);
                    q.memcpy(ff.data(), buf, n_elem * sizeof(float)).wait();
                    char path[1024];
                    std::snprintf(path, sizeof(path), "%s_%s_%02u.bin",
                                  dump_prefix_.c_str(), tag, dn_idx);
                    if (FILE* fp = std::fopen(path, "wb")) {
                        std::fwrite(ff.data(), sizeof(float), n_elem, fp);
                        std::fclose(fp);
                    }
                };
                dump_dn_in(ws_q_fp32_,    SVH * SHD, "DNq");
                dump_dn_in(ws_k_fp32_,    SVH * SHD, "DNk");
                dump_dn_in(ws_v_fp32_,    SVH * SHD, "DNv");
                dump_dn_in(ws_g_fp32_,    SVH,       "DNg");
                dump_dn_in(ws_beta_fp32_, SVH,       "DNb");
            }

            profile_section("dn_recurrence", [&]{
                if (ckpt_mode) {
                    // T single-token recurrence steps; snapshot recurrent state
                    // AFTER each position. K×(T=1) == 1×(T=K) (sequential scan over
                    // `state_layer`) → byte-identical, rollback-free commit.
                    const uint64_t se = dn.state_elems_per_layer();
                    const uint64_t qkv_stride = uint64_t(SVH) * SHD;   // per-position
                    const uint64_t gb_stride  = uint64_t(SVH);
                    for (uint32_t s = 0; s < T; ++s) {
                        deltanet_recurrence(q,
                            ws_q_fp32_ + s * qkv_stride, ws_k_fp32_ + s * qkv_stride,
                            ws_v_fp32_ + s * qkv_stride, ws_g_fp32_ + s * gb_stride,
                            ws_beta_fp32_ + s * gb_stride, state_layer,
                            ws_recurrence_out_ + s * qkv_stride, /*B=*/1, /*T=*/1,
                            SVH, SHD, SHD);
                        q.memcpy(ckpt->ckpt_state +
                                     (uint64_t(s) * ckpt->n_lin + dn_idx) * se,
                                 state_layer, se * sizeof(float));
                    }
                } else {
                    // §1 BMG DeltaNet-recurrence HW-bug cap: each recurrence LAUNCH
                    // must stay ≤512 steps (the validated-safe regime — longer launches
                    // risk the stochastic state[] divergence AND the Triton-#6658
                    // DEVICE_LOST-from-long-recurrence hazard, both growing with launch
                    // length). When forward() receives a big-T prefill chunk (the
                    // large-M MoE long-ctx lever), sub-chunk the scan HERE: the
                    // recurrence is a sequential state-carrying scan, so splitting
                    // [0:T] into ≤512 segments that thread `state_layer` through is
                    // bit-identical to one call (modulo the very §1 non-determinism
                    // this AVOIDS by keeping launches short). NO-OP at T≤512 (one
                    // iteration == today's behavior; default unchanged). [2026-06-26]
                    static const uint32_t DN_RECUR_CHUNK = []() -> uint32_t {
                        const char* e = std::getenv("IE_QWEN36_DN_RECUR_CHUNK");
                        if (!e) return 512u;
                        int v = std::atoi(e);
                        return (v >= 1 && v <= 512) ? uint32_t(v) : 512u;
                    }();
                    const uint64_t qkv_stride = uint64_t(SVH) * SHD;  // per-position
                    const uint64_t gb_stride  = uint64_t(SVH);
                    for (uint32_t off = 0; off < T; off += DN_RECUR_CHUNK) {
                        const uint32_t n = std::min<uint32_t>(DN_RECUR_CHUNK, T - off);
                        deltanet_recurrence(q,
                            ws_q_fp32_ + off * qkv_stride, ws_k_fp32_ + off * qkv_stride,
                            ws_v_fp32_ + off * qkv_stride, ws_g_fp32_ + off * gb_stride,
                            ws_beta_fp32_ + off * gb_stride, state_layer,
                            ws_recurrence_out_ + off * qkv_stride, /*B=*/1, n,
                            SVH, SHD, SHD);
                    }
                }
            });

            const bool ssm_out_q8 = (T == 1) && q8_decode_enabled() && ws_q8_ &&
                                    w.ssm_out.dt == DType::kQ6_K;
            profile_section("dn_gated_norm", [&]{
                if (ssm_out_q8)
                    gated_rms_norm_q8(q, ws_recurrence_out_, ws_z_fp16_,
                                      w.ssm_norm_fp16, ws_gated_norm_, ws_q8_,
                                      T * SVH, SHD, cfg_.rms_eps);
                else
                    gated_rms_norm(q, ws_recurrence_out_, ws_z_fp16_, w.ssm_norm_fp16,
                                   ws_gated_norm_, T * SVH, SHD, cfg_.rms_eps);
            });

            profile_section("dn_ssm_out", [&]{
                if (ssm_out_q8)
                    gemv_q6_K_q8(q, ws_q8_, w.ssm_out.p, ws_attn_block_, SI, H);
                else
                    gemv_q_T(q, ws_gated_norm_, w.ssm_out, ws_attn_block_, SI, H, T);
            });
            });  // end profile_section attn_dn_block
        }

        dump_attn(int(L));
#if IE_FUSE_RES_RMS
        // 2b+2c fused: residual_attn → post_attn_norm in one launch.
        // Eliminates the rms_norm launch (~29 µs) per layer × 40 = ~1.16 ms/step.
        profile_section("res_rms_attn", [&]{
            residual_add_rms_norm_fused(q, ws_x_, ws_attn_block_, w.post_attn_norm,
                                        ws_x_normed_, T, H, cfg_.rms_eps);
        });
#else
        // 2b. residual: ws_x_ += ws_attn_block_
        profile_section("residual_attn", [&]{
            residual_add(q, ws_x_, ws_attn_block_, ws_x_, uint64_t(T) * H);
        });

        // 2c. post_attention_norm (1+w) → ws_x_normed_
        profile_section("post_attn_norm", [&]{
            rms_norm_f32w(q, ws_x_, w.post_attn_norm, ws_x_normed_, T, H, cfg_.rms_eps);
        });
#endif

        // 2d. MoE — route + dispatch + shared expert all rolled up.  The
        // dispatch block has 4 branches (decode, multi-expert prefill,
        // unfused prefill, unusual dtypes) which makes per-section wrap
        // awkward; we time the big rollup and name it after the dominant
        // path that ran.
        const double _moe_t0 =
            profile_enabled_
                ? (q.wait(),
                   std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now().time_since_epoch()).count())
                : 0.0;
        moe_router(q, ws_x_normed_, w.ffn_gate_inp, ws_topk_idx_, ws_topk_w_,
                   T, H, E, K_top);

        // Per-expert byte strides depend on this layer's dtype.
        const size_t s_g = uint64_t(H) * E_ffn / 256 * bytes_per_block_for(w.ffn_gate_exps.dt);
        const size_t s_u = uint64_t(H) * E_ffn / 256 * bytes_per_block_for(w.ffn_up_exps.dt);
        const size_t s_d = uint64_t(E_ffn) * H / 256 * bytes_per_block_for(w.ffn_down_exps.dt);

        // ---- Fused MoE path: stage1 (gate+up+silu) + stage2 (down+accum) ----
        // Active for the daily-driver Q4_K_M layout where gate/up are Q4_K
        // and down is Q4_K or Q6_K. For T>1 (prefill) we run the fused
        // pair per-token (T launches per stage instead of 40 per token in
        // the scalar path). Falls back to the scalar loop for unusual
        // dtype mixes.
        const bool can_fuse_moe =
            w.ffn_gate_exps.dt == DType::kQ4_K &&
            w.ffn_up_exps.dt   == DType::kQ4_K &&
            (w.ffn_down_exps.dt == DType::kQ4_K || w.ffn_down_exps.dt == DType::kQ6_K);

        // Fused multi-expert prefill ceiling. Default 2048: above it the int-dot
        // multi-expert kernel re-streams all 256 experts' weights per call and the
        // per-token path won on data locality (small-M regime). The large-M lever
        // raises this (IE_QWEN36_MOE_FUSED_CEIL) so a big-T chunk takes the fused
        // path at M=T·K/E ≫ 16, where the single weight-read amortizes. Bounded by
        // the scratch cap (T ≤ ws_T_moe_). [2026-06-26]
        // Fused-path ceiling: 16384 when the oneDNN lever is active for this load
        // (signalled by the buffers being allocated = ensure_workspace turned it on),
        // else the historical 2048 (above which the per-token path won at small-M).
        // Env still overrides. Per-call (cheap) since it reads the per-instance buffer.
        static const char* moe_fceil_env = std::getenv("IE_QWEN36_MOE_FUSED_CEIL");
        const uint32_t moe_fused_ceil =
            moe_fceil_env ? (std::atoi(moe_fceil_env) >= 64 ? uint32_t(std::atoi(moe_fceil_env)) : 2048u)
                          : (ws_moe_btf16_ ? 16384u : 2048u);

        if (can_fuse_moe && T == 1) {
            // T=1 fused decode (each expert has at most 1 token).
            int32_t* idx_t  = ws_topk_idx_;
            sycl::half* w_t = ws_topk_w_;
            profile_section("moe_decode_gate_up_silu", [&]{
                moe_decode_gate_up_silu_q4k(q, ws_x_normed_,
                                            w.ffn_gate_exps.p, w.ffn_up_exps.p,
                                            idx_t, ws_h_routed_,
                                            H, E_ffn, K_top, s_g, moe_exps_soa_);
            });
            profile_section("moe_decode_down", [&]{
                if (w.ffn_down_exps.dt == DType::kQ4_K) {
                    moe_decode_down_q4k(q, ws_h_routed_, w.ffn_down_exps.p,
                                        idx_t, w_t, ws_attn_block_,
                                        H, E_ffn, K_top, s_d, moe_exps_soa_);
                } else {
                    moe_decode_down_q6k(q, ws_h_routed_, w.ffn_down_exps.p,
                                        idx_t, w_t, ws_attn_block_,
                                        H, E_ffn, K_top, s_d, moe_exps_soa_);
                }
            });
        } else if (can_fuse_moe && T >= 64 && T < moe_fused_ceil && T <= ws_T_moe_ &&
                   ws_moe_h_packed_ && ws_moe_out_packed_ && ws_moe_xp_ &&
                   ws_moe_expert_offsets_ && ws_moe_tk_to_packed_) {
            // Sweet spot 64 ≤ T < 2048:
            //   T < 64:    setup overhead (sort + 4 H2D + 8k WGs mostly empty) > work
            //   T ≥ 2048:  fused-per-token wins on data locality (each token's
            //              working set stays in cache; multi-expert touches all
            //              256 experts' weights spread across DRAM)
            // Multi-expert single-launch prefill (item 10), atomics-free
            // variant. Stage 1 packs gate+up+silu for all 256 experts in
            // ONE launch.  Stage 2 packs down for all 256 experts in ONE
            // launch (each WG owns its (expert, h_chunk), no race).
            // Reduce kernel sums weighted contributions per token.
            //
            // Sort tokens by expert id (host counting sort) and ALSO build
            // the inverse map tk_to_packed[t*K_top + k_slot] → packed_row
            // so the reduce kernel can look up the K_top contributions
            // for each token.
            // Pull topk to host (only THIS read must block — subsequent
            // H2D writes ride the in-order queue without per-call .wait()).
            std::vector<int32_t>     hidx(T * K_top);
            std::vector<sycl::half>  htw(T * K_top);
            q.memcpy(hidx.data(), ws_topk_idx_, hidx.size() * sizeof(int32_t));
            q.memcpy(htw.data(),  ws_topk_w_,   htw.size()  * sizeof(sycl::half)).wait();
            q.memset(ws_attn_block_, 0, uint64_t(T) * H * sizeof(sycl::half));

            const uint32_t TOTAL = T * K_top;
            std::vector<uint32_t> off(E + 1, 0);
            for (uint32_t i = 0; i < TOTAL; ++i) ++off[uint32_t(hidx[i]) + 1];
            for (uint32_t e = 0; e < E; ++e) off[e + 1] += off[e];
            std::vector<int32_t>    sorted_idx(TOTAL);
            std::vector<sycl::half> sorted_w(TOTAL);
            std::vector<uint32_t>   tk2pk(TOTAL);
            std::vector<uint32_t>   cursor(E, 0);
            for (uint32_t t = 0; t < T; ++t) {
                for (uint32_t kk = 0; kk < K_top; ++kk) {
                    const uint32_t e   = uint32_t(hidx[t * K_top + kk]);
                    const uint32_t pos = off[e] + cursor[e]++;
                    sorted_idx[pos]    = int32_t(t);
                    sorted_w[pos]      = htw[t * K_top + kk];
                    tk2pk[t * K_top + kk] = pos;
                }
            }

            // H2D writes — queue without .wait(), in-order queue handles deps.
            // Caller of qwen36.forward holds host buffers' lifetime via the
            // outer .wait() on the returned event.  But these vectors go out
            // of scope at end of this block — we must wait BEFORE leaving
            // scope, OR persist them.  Wait once after queueing all 4.
            q.memcpy(ws_moe_expert_offsets_, off.data(),       (E + 1) * sizeof(uint32_t));
            q.memcpy(ws_moe_token_idx_,      sorted_idx.data(), TOTAL * sizeof(int32_t));
            q.memcpy(ws_moe_token_w_,        sorted_w.data(),   TOTAL * sizeof(sycl::half));
            q.memcpy(ws_moe_tk_to_packed_,   tk2pk.data(),      TOTAL * sizeof(uint32_t)).wait();

            // E5: gather all TOTAL rows into expert-sorted ws_moe_xp_ ONCE.
            // Before, every n-chunk WG in stage 1 re-gathered the same rows
            // through sorted_token_idx — E_ffn/32 = 24× redundant scattered
            // reads of x per layer.
            moe_gather_rows(q, ws_x_normed_, ws_moe_token_idx_,
                            ws_moe_xp_, TOTAL, H);
            // P1b-2: quantize the packed rows once per layer for the
            // int-dot stage 1 (rows are H-aligned: blocks never straddle).
            // Prefill-crown (2026-06-10): int-dot MoE prefill is default ON.
            // Same-hour showdown: pp512 1147/1139/1147 vs llama.cpp SYCL
            // master 1064.25 +- 7.66 (+7.6%); PPL 6.52 held.  The win is
            // SoA streams + full-K register lattices + s0/s1 corrections
            // (docs/prefill_crown_plan.md, executed).  Opt out with
            // IE_NO_MOE_Q8=1.  Shape guard: stage-1 lattice needs H%512==0
            // (stage-2 additionally checks E_ffn==512 at its dispatch).
            // --- oneDNN large-M MoE branch (the long-ctx prefill lever) ---
            // The int-dot fused kernels re-stream each expert's weights ~linearly
            // with that expert's row count → catastrophic at large rows/expert (M):
            // measured ~LINEAR (Exp1) and a REAL-prompt REGRESSION at big chunks
            // (Exp2). A blocked GEMM reads each expert's weights ONCE and reuses
            // them across its M rows, so it WINS once M ≥ ~256 (T ≳ 8K here) — the
            // same lever that put Coder/Gemma prefill ahead. Per-expert: dequant
            // Q4_K/Q6_K → fp16 Bt, then gemm_fp16_onednn over the EXISTING expert-
            // sorted buffers; output feeds the SAME moe_prefill_reduce. Works on the
            // DEFAULT SoA experts (SoA-aware dq_bt below) so decode is untouched.
            // Single-card only (oneDNN ctx binds card 0 — crown is 1-card). The lever
            // is enabled at LOAD (DEFAULT-ON for max_ctx≥8192; opt-out
            // IE_QWEN36_NO_MOE_ONEDNN) — signalled here by the buffers being present.
            // T-gate IE_QWEN36_MOE_ONEDNN_MINT (def 4096). [2026-06-26]
            static const uint32_t moe_onednn_minT = []() -> uint32_t {
                const char* e = std::getenv("IE_QWEN36_MOE_ONEDNN_MINT");
                if (!e) return 4096u; int v = std::atoi(e); return v >= 1 ? uint32_t(v) : 4096u;
            }();
            const bool use_onednn = T >= moe_onednn_minT &&
                                    ws_moe_btf16_ && ws_moe_up_packed_;
            if (use_onednn) {
                // Per-expert weight → fp16 Bt dequant, SoA-AWARE so the DEFAULT SoA
                // expert layout is kept (decode stays on the fast SoA int-dot path —
                // NO AoS decode regression). Crown's Q4_K MoE SoA keeps packed 6-bit
                // scales → its own dequant_q4_moe_soa_to_Bt; Q6_K down → the shared
                // dequant_q6_K_soa_to_Bt. AoS fall-through if moe_exps_soa_ is off.
                auto dq_bt = [&](const void* p_e, DType dt, uint32_t Kd, uint32_t Nd)
                             -> sycl::event {
                    if (dt == DType::kQ4_K)
                        return moe_exps_soa_
                            ? dequant_q4_moe_soa_to_Bt(q, p_e, ws_moe_btf16_, Kd, Nd, {})
                            : dequant_q4_K_to_Bt(q, p_e, ws_moe_btf16_, Kd, Nd, {});
                    return moe_exps_soa_
                        ? dequant_q6_K_soa_to_Bt(q, p_e, ws_moe_btf16_, Kd, Nd, {})
                        : dequant_q6_K_to_Bt(q, p_e, ws_moe_btf16_, Kd, Nd, {});
                };
                // Stage 1: gate + up (separate GEMMs), then swiglu over all rows.
                for (uint32_t e = 0; e < E; ++e) {
                    const uint32_t o = off[e]; const uint32_t n_e = off[e + 1] - o;
                    if (n_e == 0) continue;
                    const sycl::half* x_e = ws_moe_xp_ + uint64_t(o) * H;
                    auto dg = dq_bt((const uint8_t*)w.ffn_gate_exps.p + uint64_t(e) * s_g,
                                    w.ffn_gate_exps.dt, H, E_ffn);
                    gemm_fp16_onednn(q, x_e, ws_moe_btf16_,
                        ws_moe_h_packed_ + uint64_t(o) * E_ffn, n_e, E_ffn, H, {dg});
                    auto du = dq_bt((const uint8_t*)w.ffn_up_exps.p + uint64_t(e) * s_u,
                                    w.ffn_up_exps.dt, H, E_ffn);
                    gemm_fp16_onednn(q, x_e, ws_moe_btf16_,
                        ws_moe_up_packed_ + uint64_t(o) * E_ffn, n_e, E_ffn, H, {du});
                }
                swiglu(q, ws_moe_h_packed_, ws_moe_up_packed_, ws_moe_h_packed_,
                       uint64_t(TOTAL) * E_ffn);
                // Stage 2: down (per-expert GEMM, into the expert-sorted out buffer).
                for (uint32_t e = 0; e < E; ++e) {
                    const uint32_t o = off[e]; const uint32_t n_e = off[e + 1] - o;
                    if (n_e == 0) continue;
                    const sycl::half* h_e = ws_moe_h_packed_ + uint64_t(o) * E_ffn;
                    auto dd = dq_bt((const uint8_t*)w.ffn_down_exps.p + uint64_t(e) * s_d,
                                    w.ffn_down_exps.dt, E_ffn, H);
                    gemm_fp16_onednn(q, h_e, ws_moe_btf16_,
                        ws_moe_out_packed_ + uint64_t(o) * H, n_e, H, E_ffn, {dd});
                }
            } else {
            static const bool moe_q8 = std::getenv("IE_NO_MOE_Q8") == nullptr;
            const bool q8_path = moe_q8 && ws_moe_xp_q8_ && (H % 512 == 0);
            if (q8_path)
                quantize_q8_1s(q, ws_moe_xp_, ws_moe_xp_q8_,
                               uint64_t(TOTAL) * H);

            // Stage 1: gate + up + silu (all experts in one launch).
            // XMX variant tried (8 SGs / WG, 4 gate + 4 up) — still slower
            // than the scalar version (~ -10-15%).  Likely the SLM round-trip
            // for dequanted Q4_K weights costs more than the mat_mad savings,
            // since scalar inlines dequant directly into the multiply.  The
            // multi-expert pattern (many small WGs, fragmented work per
            // expert) compounds this.  Kept in gemm_q4k_xmx.cpp for a future
            // dedicated rewrite — possibly with cl_intel_subgroup_2d_block_io
            // for the Q4_K reads to compress the dequant cost.
            // v1.3 experiment: XMX stage-1 (rewritten 2026-06-09), opt-in
            // via IE_MOE_XMX=1 while it's being validated against the
            // scalar kernel.
            static const bool moe_xmx = std::getenv("IE_MOE_XMX") != nullptr;
            if (moe_xmx && moe_exps_soa_) {
                // The XMX experiment kernel is AoS-only; with the SoA repack
                // active it would read garbage.  Warn once and fall through.
                static bool warned = (std::fprintf(stderr,
                    "[qwen36] IE_MOE_XMX ignored: expert tensors are SoA-"
                    "repacked (set IE_NO_MOE_SOA=1 to use the XMX path)\n"), true);
                (void)warned;
            }
            if (moe_xmx && !moe_exps_soa_) {
                moe_prefill_gate_up_silu_q4k_xmx(q, ws_moe_xp_,
                                                 w.ffn_gate_exps.p, w.ffn_up_exps.p,
                                                 ws_moe_expert_offsets_,
                                                 ws_moe_h_packed_,
                                                 E, H, E_ffn, s_g);
            } else if (q8_path) {
                moe_prefill_gate_up_silu_q4k_q8(q, ws_moe_xp_q8_,
                                                w.ffn_gate_exps.p, w.ffn_up_exps.p,
                                                ws_moe_expert_offsets_,
                                                ws_moe_h_packed_,
                                                E, H, E_ffn, s_g, moe_exps_soa_);
            } else {
                moe_prefill_gate_up_silu_q4k(q, ws_moe_xp_,
                                             w.ffn_gate_exps.p, w.ffn_up_exps.p,
                                             ws_moe_expert_offsets_,
                                             ws_moe_h_packed_,
                                             E, H, E_ffn, s_g, moe_exps_soa_);
            }
            // Stage 2: down (all experts in one launch, atomics-free).
            // IE_MOE_Q8 + E_ffn==512: int-dot kernels over a q8 stream of
            // h_packed (quantized once per layer; the SG covers the full
            // K=512 reduction so weights are register-resident per column).
            // Otherwise the fp16 kernel selected by IE_ENABLE_MOE_DOWN_TILE.
            if (q8_path && ws_moe_h_q8_ && E_ffn == 512) {
                quantize_q8_1s(q, ws_moe_h_packed_, ws_moe_h_q8_,
                               uint64_t(TOTAL) * E_ffn);
                if (w.ffn_down_exps.dt == DType::kQ4_K) {
                    moe_prefill_down_packed_q4k_q8(q, ws_moe_h_q8_,
                                                   w.ffn_down_exps.p,
                                                   ws_moe_expert_offsets_,
                                                   ws_moe_out_packed_,
                                                   E, H, E_ffn, s_d, moe_exps_soa_);
                } else {
                    moe_prefill_down_packed_q6k_q8(q, ws_moe_h_q8_,
                                                   w.ffn_down_exps.p,
                                                   ws_moe_expert_offsets_,
                                                   ws_moe_out_packed_,
                                                   E, H, E_ffn, s_d, moe_exps_soa_);
                }
            } else if (w.ffn_down_exps.dt == DType::kQ4_K) {
                IE_MOE_DN_PK4K(q, ws_moe_h_packed_, w.ffn_down_exps.p,
                               ws_moe_expert_offsets_,
                               ws_moe_out_packed_,
                               E, H, E_ffn, s_d, moe_exps_soa_);
            } else {
                IE_MOE_DN_PK6K(q, ws_moe_h_packed_, w.ffn_down_exps.p,
                               ws_moe_expert_offsets_,
                               ws_moe_out_packed_,
                               E, H, E_ffn, s_d, moe_exps_soa_);
            }
            }  // end else: int-dot fused stage1+stage2 (vs the oneDNN branch)
            // Reduce: per (token, h), sum weighted K_top contributions into
            // ws_attn_block_ (shared expert adds on top after).
            moe_prefill_reduce(q, ws_moe_out_packed_, ws_moe_tk_to_packed_,
                               ws_moe_token_w_, ws_attn_block_,
                               T, K_top, H);
        } else if (can_fuse_moe && T > 1) {
            // Fused per-token loop (T>=2048 — multi-expert atomics dominate
            // beyond that point).
            for (uint32_t t = 0; t < T; ++t) {
                sycl::half* x_t = ws_x_normed_ + uint64_t(t) * H;
                sycl::half* y_t = ws_attn_block_ + uint64_t(t) * H;
                int32_t* idx_t  = ws_topk_idx_ + uint64_t(t) * K_top;
                sycl::half* w_t = ws_topk_w_   + uint64_t(t) * K_top;
                moe_decode_gate_up_silu_q4k(q, x_t,
                                            w.ffn_gate_exps.p, w.ffn_up_exps.p,
                                            idx_t, ws_h_routed_,
                                            H, E_ffn, K_top, s_g, moe_exps_soa_);
                if (w.ffn_down_exps.dt == DType::kQ4_K) {
                    moe_decode_down_q4k(q, ws_h_routed_, w.ffn_down_exps.p,
                                        idx_t, w_t, y_t,
                                        H, E_ffn, K_top, s_d, moe_exps_soa_);
                } else {
                    moe_decode_down_q6k(q, ws_h_routed_, w.ffn_down_exps.p,
                                        idx_t, w_t, y_t,
                                        H, E_ffn, K_top, s_d, moe_exps_soa_);
                }
            }
        } else if (false) {
            // (placeholder for old scatter-gather path)
            std::vector<int32_t>     hidx(T * K_top);
            std::vector<sycl::half>  htw(T * K_top);
            q.memcpy(hidx.data(), ws_topk_idx_, hidx.size() * sizeof(int32_t)).wait();
            q.memcpy(htw.data(),  ws_topk_w_,   htw.size()  * sizeof(sycl::half)).wait();
            q.memset(ws_attn_block_, 0, uint64_t(T) * H * sizeof(sycl::half)).wait();

            // Counting sort: O(T*K_top + E). Produces:
            //   off[e]      = expert e's start offset in sorted_idx/sorted_w
            //   off[e+1] - off[e] = n_tok routed to expert e
            //   sorted_idx[i] = token id for the i-th (e, t) pair, expert-sorted
            //   sorted_w[i]   = corresponding router weight
            const uint32_t TOTAL = T * K_top;
            std::vector<uint32_t> count(E + 1, 0);
            for (uint32_t i = 0; i < TOTAL; ++i) ++count[uint32_t(hidx[i]) + 1];
            for (uint32_t e = 0; e < E; ++e) count[e + 1] += count[e];
            std::vector<uint32_t> off = count;          // 0..E (offsets)
            std::vector<int32_t>    sorted_idx(TOTAL);
            std::vector<sycl::half> sorted_w(TOTAL);
            std::vector<uint32_t> cursor(E, 0);
            for (uint32_t t = 0; t < T; ++t) {
                for (uint32_t kk = 0; kk < K_top; ++kk) {
                    const uint32_t e = uint32_t(hidx[t * K_top + kk]);
                    const uint32_t pos = off[e] + cursor[e]++;
                    sorted_idx[pos] = int32_t(t);
                    sorted_w[pos]   = htw[t * K_top + kk];
                }
            }
            // Single H2D for the whole layer.
            q.memcpy(ws_moe_token_idx_, sorted_idx.data(), TOTAL * sizeof(int32_t)).wait();
            q.memcpy(ws_moe_token_w_,   sorted_w.data(),   TOTAL * sizeof(sycl::half)).wait();

            // Per non-empty expert: gather → gate/up/silu/down → scatter.
            for (uint32_t e = 0; e < E; ++e) {
                const uint32_t n_tok = off[e + 1] - off[e];
                if (n_tok == 0) continue;
                const uint32_t base = off[e];
                int32_t*    idx_dev = ws_moe_token_idx_ + base;
                sycl::half* w_dev   = ws_moe_token_w_   + base;

                moe_gather_rows(q, ws_x_normed_, idx_dev,
                                ws_moe_x_packed_, n_tok, H);

                QuantPtr Wg{ static_cast<uint8_t*>(w.ffn_gate_exps.p) + e * s_g,
                             w.ffn_gate_exps.dt };
                QuantPtr Wu{ static_cast<uint8_t*>(w.ffn_up_exps.p)   + e * s_u,
                             w.ffn_up_exps.dt };
                QuantPtr Wd{ static_cast<uint8_t*>(w.ffn_down_exps.p) + e * s_d,
                             w.ffn_down_exps.dt };

                gemv_q_T(q, ws_moe_x_packed_, Wg, ws_gate_o_, H, E_ffn, n_tok);
                gemv_q_T(q, ws_moe_x_packed_, Wu, ws_up_o_,   H, E_ffn, n_tok);
                swiglu(q, ws_gate_o_, ws_up_o_, ws_h_o_, uint64_t(n_tok) * E_ffn);
                gemv_q_T(q, ws_h_o_, Wd, ws_eo_, E_ffn, H, n_tok);

                moe_scatter_add(q, ws_eo_, idx_dev, w_dev,
                                ws_attn_block_, n_tok, H);
            }
        } else {
            // Scalar fallback for unusual dtype mixes. Host-roundtrip topk
            // + 5 launches per expert per token (40 launches/token/layer).
            std::vector<int32_t>     hidx(T * K_top);
            std::vector<sycl::half>  htw(T * K_top);
            q.memcpy(hidx.data(), ws_topk_idx_, hidx.size() * sizeof(int32_t)).wait();
            q.memcpy(htw.data(),  ws_topk_w_,   htw.size()  * sizeof(sycl::half)).wait();
            q.memset(ws_attn_block_, 0, uint64_t(T) * H * sizeof(sycl::half)).wait();

            auto run_expert = [&](sycl::half* x_t, sycl::half* y_t, uint32_t e, sycl::half scale) {
                QuantPtr Wg{ static_cast<uint8_t*>(w.ffn_gate_exps.p) + e * s_g, w.ffn_gate_exps.dt };
                QuantPtr Wu{ static_cast<uint8_t*>(w.ffn_up_exps.p)   + e * s_u, w.ffn_up_exps.dt };
                QuantPtr Wd{ static_cast<uint8_t*>(w.ffn_down_exps.p) + e * s_d, w.ffn_down_exps.dt };
                gemv_q(q, x_t, Wg, ws_gate_o_, H, E_ffn);
                gemv_q(q, x_t, Wu, ws_up_o_,   H, E_ffn);
                swiglu(q, ws_gate_o_, ws_up_o_, ws_h_o_, E_ffn);
                gemv_q(q, ws_h_o_, Wd, ws_eo_, E_ffn, H);
                scaled_add(q, ws_eo_, scale, y_t, H);
            };

            for (uint32_t t = 0; t < T; ++t) {
                sycl::half* x_t = ws_x_normed_   + uint64_t(t) * H;
                sycl::half* y_t = ws_attn_block_ + uint64_t(t) * H;
                for (uint32_t kk = 0; kk < K_top; ++kk) {
                    run_expert(x_t, y_t, uint32_t(hidx[t * K_top + kk]),
                               htw[t * K_top + kk]);
                }
            }
        }

        // Close moe_routed timing window before shared expert.
        if (profile_enabled_) {
            q.wait();
            const double t_now = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            const char* moe_label = (T == 1) ? "moe_routed_decode" :
                                    (T < 2048) ? "moe_routed_multiexpert" :
                                                  "moe_routed_pertoken";
            bool added = false;
            for (auto& p : profile_totals_) {
                if (p.first == moe_label) { p.second += (t_now - _moe_t0); added = true; break; }
            }
            if (!added) profile_totals_.emplace_back(moe_label, t_now - _moe_t0);
        }

        // Shared expert (applied once per token; per-token gate scalar stays
        // on device → no host roundtrip during decode).
        profile_section("moe_shared", [&]{
        // v1.4: T==1 all-Q4_K fast path — 5 launches (gate-scalar dot, dual
        // gate/up, swiglu, down, scaled_add) fused into 2.
        const bool shexp_fused =
            (T == 1) &&
            w.ffn_gate_shexp.dt == DType::kQ4_K &&
            w.ffn_up_shexp.dt   == DType::kQ4_K &&
            w.ffn_down_shexp.dt == DType::kQ4_K;
        if (!shexp_fused)
            shared_expert_gate(q, ws_x_normed_, w.ffn_gate_inp_shexp, ws_sh_g_, T, H);
        if (shexp_fused) {
            gemv_q4_K_shexp_gate_up(q, ws_x_normed_,
                                    w.ffn_gate_shexp.p, w.ffn_up_shexp.p,
                                    w.ffn_gate_inp_shexp,
                                    ws_h_o_, ws_sh_g_, H, E_ffn);
            gemv_q4_K_down_accum(q, ws_h_o_, w.ffn_down_shexp.p,
                                 ws_sh_g_, ws_attn_block_, E_ffn, H);
        } else if (T == 1) {
#if IE_FUSE_FFN_GATE_UP
            // Fused shared-expert gate+up when both are Q4_K (always true in
            // this model).  T==1 only — T>1 prefill keeps the unfused path
            // since it goes through gemm_q4_K, not gemv.
            if (w.ffn_gate_shexp.dt == DType::kQ4_K &&
                w.ffn_up_shexp.dt   == DType::kQ4_K) {
                gemv_q4_K_dual_ffn(q, ws_x_normed_,
                                   w.ffn_gate_shexp.p, w.ffn_up_shexp.p,
                                   ws_gate_o_, ws_up_o_,
                                   H, E_ffn);
            } else {
                gemv_q(q, ws_x_normed_, w.ffn_gate_shexp, ws_gate_o_, H, E_ffn);
                gemv_q(q, ws_x_normed_, w.ffn_up_shexp,   ws_up_o_,   H, E_ffn);
            }
            swiglu(q, ws_gate_o_, ws_up_o_, ws_h_o_, E_ffn);
            gemv_q(q, ws_h_o_, w.ffn_down_shexp, ws_eo_, E_ffn, H);
            scaled_add_dev_scalar(q, ws_eo_, ws_sh_g_, ws_attn_block_, H);
#else
            gemv_q(q, ws_x_normed_, w.ffn_gate_shexp, ws_gate_o_, H, E_ffn);
            gemv_q(q, ws_x_normed_, w.ffn_up_shexp,   ws_up_o_,   H, E_ffn);
            swiglu(q, ws_gate_o_, ws_up_o_, ws_h_o_, E_ffn);
            gemv_q(q, ws_h_o_, w.ffn_down_shexp, ws_eo_, E_ffn, H);
            scaled_add_dev_scalar(q, ws_eo_, ws_sh_g_, ws_attn_block_, H);
#endif
        } else {
            // 2026-05-05: T>1 prefill — batch through gemv_q_T (now uses
            // M_TILE=16 XMX) and a per-token scaled_add helper.  Replaces the
            // 4-launch-per-token loop (≈ 4×T launches per layer) with 4
            // device-side launches total per layer, eliminating the host
            // memcpy + serialization on ws_sh_g_.
            gemv_q_T(q, ws_x_normed_, w.ffn_gate_shexp, ws_gate_o_, H, E_ffn, T);
            gemv_q_T(q, ws_x_normed_, w.ffn_up_shexp,   ws_up_o_,   H, E_ffn, T);
            swiglu(q, ws_gate_o_, ws_up_o_, ws_h_o_, uint64_t(T) * E_ffn);
            gemv_q_T(q, ws_h_o_, w.ffn_down_shexp, ws_eo_, E_ffn, H, T);
            scaled_add_per_token_row(q, ws_eo_, ws_sh_g_, ws_attn_block_, T, H);
        }
        });  // end profile_section moe_shared

        // 2e. residual: ws_x_ += ws_attn_block_
        profile_section("residual_moe", [&]{
            residual_add(q, ws_x_, ws_attn_block_, ws_x_, uint64_t(T) * H);
        });

        if (std::getenv("IE_TRACE")) {
            char buf[16]; std::snprintf(buf, sizeof(buf), "L=%u", L);
            trace_norm(buf);
        }
        dump_residual(int(L) + 1);  // L01..L40 = post-layer-N residual

        // PR #1 — partial-stack logit capture for self-speculative decoding
        // viability study.  Purely observational: rms_norm + lm_head on the
        // last token's residual into a side buffer.  Does not modify ws_x_,
        // KV cache, or DeltaNet state.  No-op when partial_cuts_ empty.
        if (!partial_cuts_.empty() && ws_partial_normed_) {
            const uint32_t cut_layer_id = L + 1;  // 1..n_layers
            auto it = std::lower_bound(partial_cuts_.begin(), partial_cuts_.end(), cut_layer_id);
            if (it != partial_cuts_.end() && *it == cut_layer_id) {
                const size_t idx = size_t(it - partial_cuts_.begin());
                sycl::half* last_resid = ws_x_ + uint64_t(T - 1) * H;
                rms_norm_f32w(q, last_resid, output_norm_,
                              ws_partial_normed_, /*T=*/1, H, cfg_.rms_eps);
                QuantPtr lm_head{output_, output_dtype_};
                gemv_q(q, ws_partial_normed_, lm_head,
                       partial_out_logits_[idx], H, cfg_.vocab);
            }
        }
    }

    // Spec-decode: export the pre-output_norm residual (h_i for the native MTP
    // head) BEFORE the final norm consumes ws_x_. ws_x_ holds the final residual.
    if (hidden_pre_norm)
        q.memcpy(hidden_pre_norm, ws_x_, uint64_t(T) * H * sizeof(sycl::half));

    // 3. Final norm (1+w bake-in: stored weight is `1+w`, formula is plain weight*x_normed)
    profile_section("final_norm", [&]{
        // v1.5-D: at decode, emit Q8_1 of the final normed vector in the
        // same launch — feeds the int-dot lm_head.
        if (T == 1 && q8_decode_enabled() && ws_q8_)
            rms_norm_f32w_q8(q, ws_x_, output_norm_, ws_x_normed_, ws_q8_,
                             T, H, cfg_.rms_eps);
        else
            rms_norm_f32w(q, ws_x_, output_norm_, ws_x_normed_, T, H, cfg_.rms_eps);
    });
    if (!dump_prefix_.empty()) {
        // Dump final-norm output (uses ws_x_normed_, not ws_x_).
        std::vector<sycl::half> hh(uint64_t(T) * H);
        q.memcpy(hh.data(), ws_x_normed_, hh.size() * sizeof(sycl::half)).wait();
        std::vector<float> ff(hh.size());
        for (size_t i = 0; i < hh.size(); ++i) ff[i] = float(hh[i]);
        char path[1024];
        std::snprintf(path, sizeof(path), "%s_L41.bin", dump_prefix_.c_str());
        if (FILE* fp = std::fopen(path, "wb")) {
            std::fwrite(ff.data(), sizeof(float), ff.size(), fp);
            std::fclose(fp);
        }
        std::snprintf(path, sizeof(path), "%s_L41.meta", dump_prefix_.c_str());
        if (FILE* fp = std::fopen(path, "w")) { std::fprintf(fp, "%u %u\n", T, H); std::fclose(fp); }
    }

    // Spec-decode verify: lm_head on ALL T positions → [T, vocab] (the draft
    // tokens at every position are checked), then mirror the last into out_logits.
    if (all_logits) {
        QuantPtr lm_all{output_, output_dtype_};
        gemv_q_T(q, ws_x_normed_, lm_all, all_logits, H, cfg_.vocab, T);
        return q.memcpy(out_logits, all_logits + uint64_t(T - 1) * cfg_.vocab,
                        uint64_t(cfg_.vocab) * sizeof(sycl::half));
    }

    // 4. lm_head GEMV — only the LAST token's logits are needed.
    //    Big shape (K=2048, N=vocab≈248k); XMX path wins here on Q4_K.
    sycl::half* last_x = ws_x_normed_ + uint64_t(T - 1) * H;
    QuantPtr lm_head{output_, output_dtype_};
    // lm_head is M=1 (single token).  XMX path was tried and consistently
    // ~5% slower — for M=1 there's no row-amortization, and the XMX SLM
    // dequant round-trip is pure overhead vs the scalar gemv_q's inline
    // dequant.  Scalar wins here.
    auto run_lmhead = [&]() -> sycl::event {
        // v1.5-D: int-dot staged path — the fp16 staged kernel is ALU-bound
        // at this shape (1.68 ms vs the ~1.0 ms bandwidth floor for 417 MB).
        if (T == 1 && q8_decode_enabled() && ws_q8_ && cfg_.vocab >= 32768) {
            if (lm_head.dt == DType::kQ6_K)
                return gemv_q6_K_q8(q, ws_q8_, lm_head.p, out_logits, H, cfg_.vocab);
            if (lm_head.dt == DType::kQ4_K)   // turbo GGUF variant
                return gemv_q4_K_q8(q, ws_q8_, lm_head.p, out_logits, H, cfg_.vocab);
        }
        return gemv_q(q, last_x, lm_head, out_logits, H, cfg_.vocab);
    };
    if (profile_enabled_) {
        q.wait();
        const double t0 = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        auto e = run_lmhead();
        e.wait();
        const double t1 = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        bool added = false;
        for (auto& p : profile_totals_) {
            if (p.first == "lm_head") { p.second += (t1 - t0); added = true; break; }
        }
        if (!added) profile_totals_.emplace_back("lm_head", t1 - t0);
        return e;
    }
    return run_lmhead();
}

// ===========================================================================
// MTP/NextN draft head (blk.<n_layers>) — load + draft forward + spec_generate.
// Mirrors the 27B MtpHead (qwen35_dense.cpp) but the head FFN is the Crown's MoE
// decode block. Draft need NOT be bit-exact — losslessness comes from the verify
// forward (hooks already present) + per-position DeltaNet commit; the shared
// expert is SKIPPED in the draft (only lowers acceptance, never breaks lossless).
// ===========================================================================
namespace {
sycl::half* mtp_dequant_any_to_fp16(DeviceAllocator& alloc, const GgufTensorInfo* t,
                                    std::vector<void*>& owned, std::string& err) {
    if (!t) { err = "tensor not found"; return nullptr; }
    if (t->n_dims != 2) { err = "mtp weight: expected 2-D"; return nullptr; }
    const uint32_t K = uint32_t(t->shape[0]);
    const uint32_t N = uint32_t(t->shape[1]);
    if (t->dtype == DType::kF16) { err = "mtp weight: F16 not supported"; return nullptr; }
    void* packed = alloc.malloc(t->nbytes);
    if (!packed) { err = "malloc packed"; return nullptr; }
    alloc.queue().memcpy(packed, t->data, t->nbytes).wait();
    auto* d = static_cast<sycl::half*>(alloc.malloc(uint64_t(K) * N * sizeof(sycl::half)));
    if (!d) { alloc.free(packed); err = "malloc fp16"; return nullptr; }
    sycl::event e;
    switch (t->dtype) {
        case DType::kQ4_K: e = dequant_q4_K_to_Bt(alloc.queue(), packed, d, K, N); break;
        case DType::kQ6_K: e = dequant_q6_K_to_Bt(alloc.queue(), packed, d, K, N); break;
        case DType::kQ5_K: e = dequant_q5_K_to_Bt(alloc.queue(), packed, d, K, N); break;
        case DType::kQ8_0: e = dequant_q8_0_to_Bt(alloc.queue(), packed, d, K, N); break;
        default: alloc.free(packed); alloc.free(d);
                 err = std::string("mtp weight: unsupported dtype ") + std::string(type_name(t->dtype));
                 return nullptr;
    }
    e.wait(); alloc.free(packed); owned.push_back(d); return d;
}
int mtp_argmax_row(const sycl::half* row, uint32_t vocab) {
    float best = float(row[0]); int arg = 0;
    for (uint32_t v = 1; v < vocab; ++v) { float val = float(row[v]); if (val > best) { best = val; arg = int(v); } }
    return arg;
}
}  // namespace

std::string QwenModel::load_mtp_head(const GgufReader& g, uint32_t max_ctx) {
    if (mtp_.loaded) return {};
    if (!alloc_) return "load_mtp_head: model not loaded";
    auto& a = *alloc_; auto& q = a.queue(); auto& m = mtp_;
    m.max_ctx = max_ctx;
    const uint32_t H = cfg_.hidden, HD = cfg_.head_dim;
    const uint32_t n_q = cfg_.n_q_heads, n_kv = cfg_.n_kv_heads;
    const uint32_t N_q = n_q * HD, N_qg = N_q * 2, N_kv = n_kv * HD;
    const uint32_t vocab = cfg_.vocab, HEAD = cfg_.n_layers;   // blk.40
    std::string err; char buf[64];
    auto MT = [&](const char* n) { std::snprintf(buf, sizeof(buf), "blk.%u.%s", HEAD, n); return g.find_tensor(buf); };
    auto F32 = [&](const char* n) -> float* { auto* p = upload<float>(a, MT(n), m.owned, err, DType::kF32); if (!err.empty()) err = std::string(n) + ": " + err; return p; };
    auto F16 = [&](const char* n) -> sycl::half* { auto* p = mtp_dequant_any_to_fp16(a, MT(n), m.owned, err); if (!err.empty()) err = std::string(n) + ": " + err; return p; };
    m.enorm = F32("nextn.enorm.weight");                  if (!err.empty()) return err;
    m.hnorm = F32("nextn.hnorm.weight");                  if (!err.empty()) return err;
    m.shead_norm = F32("nextn.shared_head_norm.weight");  if (!err.empty()) return err;
    m.eh_proj = F16("nextn.eh_proj.weight");              if (!err.empty()) return err;
    m.attn_norm = F32("attn_norm.weight");                if (!err.empty()) return err;
    m.post_attn_norm = F32("post_attention_norm.weight"); if (!err.empty()) return err;
    m.attn_q_norm = F32("attn_q_norm.weight");            if (!err.empty()) return err;
    m.attn_k_norm = F32("attn_k_norm.weight");            if (!err.empty()) return err;
    m.w_attn_q = F16("attn_q.weight");                    if (!err.empty()) return err;
    m.w_attn_k = F16("attn_k.weight");                    if (!err.empty()) return err;
    m.w_attn_v = F16("attn_v.weight");                    if (!err.empty()) return err;
    m.w_attn_out = F16("attn_output.weight");             if (!err.empty()) return err;
    // Router gate (F32, or BF16 -> F32 cast for the unsloth UD GGUF).
    {
        const auto* t = MT("ffn_gate_inp.weight");
        if (!t) return "ffn_gate_inp.weight not found (MTP)";
        if (t->dtype == DType::kF32) {
            m.ffn_gate_inp = upload<float>(a, t, m.owned, err, DType::kF32);
            if (!err.empty()) return "ffn_gate_inp: " + err;
        } else if (t->dtype == DType::kBF16) {
            const uint64_t ne = t->nbytes / 2;
            auto* src = static_cast<uint16_t*>(a.malloc(t->nbytes));
            auto* dst = static_cast<float*>(a.malloc(ne * sizeof(float)));
            if (!src || !dst) return "ffn_gate_inp: bf16 cast malloc";
            q.memcpy(src, t->data, t->nbytes).wait();
            q.parallel_for(sycl::range<1>(ne), [=](sycl::id<1> i) {
                dst[uint64_t(i)] = sycl::bit_cast<float>(uint32_t(src[uint64_t(i)]) << 16);
            }).wait();
            a.free(src); m.owned.push_back(dst); m.ffn_gate_inp = dst;
        } else {
            return "ffn_gate_inp: unsupported dtype";
        }
    }
    // Experts -> fp16 banks (per-expert dequant; handles Q4_K / Q5_K / Q6_K — the
    // unsloth UD GGUF mixes Q4_K gate/up + Q5_K/Q6_K down which moe_decode can't read).
    // gate/up expert = [K=H, N=E_ffn]; down expert = [K=E_ffn, N=H].
    auto load_experts = [&](const char* n) -> sycl::half* {
        const auto* t = MT(n);
        if (!t) { err = std::string(n) + ": not found"; return nullptr; }
        const uint32_t K = uint32_t(t->shape[0]);
        const uint32_t N = uint32_t(t->shape[1]);
        const uint32_t E = cfg_.n_experts;
        const uint64_t per_e_bytes = t->nbytes / E;
        const uint64_t per_e_elems = uint64_t(K) * N;
        void* packed = a.malloc(t->nbytes);
        if (!packed) { err = "experts packed malloc"; return nullptr; }
        q.memcpy(packed, t->data, t->nbytes).wait();
        auto* bank = static_cast<sycl::half*>(a.malloc(uint64_t(E) * per_e_elems * sizeof(sycl::half)));
        if (!bank) { a.free(packed); err = "experts fp16 malloc"; return nullptr; }
        for (uint32_t e = 0; e < E; ++e) {
            const void* pe = static_cast<const uint8_t*>(packed) + uint64_t(e) * per_e_bytes;
            sycl::half* be = bank + uint64_t(e) * per_e_elems;
            sycl::event ev;
            if      (t->dtype == DType::kQ4_K) ev = dequant_q4_K_to_Bt(q, pe, be, K, N);
            else if (t->dtype == DType::kQ5_K) ev = dequant_q5_K_to_Bt(q, pe, be, K, N);
            else if (t->dtype == DType::kQ6_K) ev = dequant_q6_K_to_Bt(q, pe, be, K, N);
            else { a.free(packed); a.free(bank); err = std::string(n) + ": expert dtype unsupported"; return nullptr; }
            ev.wait();
        }
        a.free(packed); m.owned.push_back(bank); return bank;
    };
    m.exp_gate = load_experts("ffn_gate_exps.weight"); if (!err.empty()) return err;
    m.exp_up   = load_experts("ffn_up_exps.weight");   if (!err.empty()) return err;
    m.exp_down = load_experts("ffn_down_exps.weight"); if (!err.empty()) return err;
    { const auto* o = g.find_tensor("output.weight"); if (!o) return "output.weight not found (MTP)";
      m.w_lm_head = mtp_dequant_any_to_fp16(a, o, m.owned, err); if (!err.empty()) return "output.weight: " + err; }
    { const auto* te = g.find_tensor("token_embd.weight"); if (!te) return "token_embd not found (MTP)";
      m.te_dtype = te->dtype; m.te_dev = a.malloc(te->nbytes); if (!m.te_dev) return "token_embd malloc";
      q.memcpy(m.te_dev, te->data, te->nbytes).wait(); m.owned.push_back(m.te_dev); }
    auto A = [&](uint64_t n) -> sycl::half* { auto* p = sycl::malloc_device<sycl::half>(n, q); if (p) m.owned.push_back(p); return p; };
    m.d_e=A(H); m.d_hn=A(H); m.d_en=A(H); m.d_cat=A(2u*H); m.d_x=A(H); m.d_xn=A(H);
    m.d_qg=A(N_qg); m.d_q=A(N_q); m.d_gate=A(N_q); m.d_k=A(N_kv); m.d_v=A(N_kv);
    m.d_ao=A(N_q); m.d_blk=A(H); m.d_moe=A(H); m.d_logits1=A(vocab);
    { const uint32_t E_ffn = cfg_.expert_ffn; m.d_fg=A(E_ffn); m.d_fu=A(E_ffn); m.d_fh=A(E_ffn); m.d_tmp=A(H); }
    m.d_pos1 = sycl::malloc_device<int32_t>(1, q); if (m.d_pos1) m.owned.push_back(m.d_pos1);
    m.d_tok1 = sycl::malloc_device<int32_t>(1, q); if (m.d_tok1) m.owned.push_back(m.d_tok1);
    m.kc = A(uint64_t(n_kv) * max_ctx * HD); m.vc = A(uint64_t(n_kv) * max_ctx * HD);
    if (!m.d_logits1 || !m.kc || !m.vc || !m.d_pos1 || !m.d_tok1) return "MTP head scratch alloc failed";
    m.loaded = true;
    return {};
}

void QwenModel::mtp_head_forward(sycl::queue& q, const sycl::half* h_src, int32_t e_tok, int32_t pos) {
    auto& m = mtp_;
    const uint32_t H = cfg_.hidden, HD = cfg_.head_dim;
    const uint32_t n_q = cfg_.n_q_heads, n_kv = cfg_.n_kv_heads;
    const uint32_t N_q = n_q * HD, N_qg = N_q * 2, N_kv = n_kv * HD;
    const uint32_t E = cfg_.n_experts, E_ffn = cfg_.expert_ffn, K_top = cfg_.experts_topk;
    const uint32_t vocab = cfg_.vocab; const float eps = cfg_.rms_eps;
    // build_x: embed(e_tok) + enorm/hnorm/cat/eh_proj. hnorm(h_src) FIRST (reads
    // h_src=prev d_x before eh_proj overwrites d_x — no separate snapshot needed).
    q.memcpy(m.d_tok1, &e_tok, sizeof(int32_t));
    if      (m.te_dtype == DType::kQ4_K) embedding_lookup_q4k(q, m.d_tok1, m.te_dev, m.d_e, 1, H);
    else if (m.te_dtype == DType::kQ6_K) embedding_lookup_q6k(q, m.d_tok1, m.te_dev, m.d_e, 1, H);
    else                                 embedding_lookup_q8_0(q, m.d_tok1, m.te_dev, m.d_e, 1, H);
    rms_norm_f32w(q, h_src, m.hnorm, m.d_hn, 1, H, eps);
    rms_norm_f32w(q, m.d_e, m.enorm, m.d_en, 1, H, eps);
    { auto en = m.d_en; auto hn = m.d_hn; auto cat = m.d_cat; const uint32_t Hl = H;
      q.parallel_for(sycl::range<1>(uint64_t(H)), [=](sycl::id<1> i) {
          cat[uint64_t(i)] = en[uint64_t(i)]; cat[uint64_t(i) + Hl] = hn[uint64_t(i)]; }); }
    gemv_fp16(q, m.d_cat, m.eh_proj, m.d_x, 2u * H, H);
    // full-attn block (mirror crown decode: q/k/v + QK-norm + partial RoPE + SDPA + out-gate)
    q.memcpy(m.d_pos1, &pos, sizeof(int32_t));
    rms_norm_f32w(q, m.d_x, m.attn_norm, m.d_xn, 1, H, eps);
    gemv_fp16(q, m.d_xn, m.w_attn_q, m.d_qg, H, N_qg);
    split_q_gate_per_head(q, m.d_qg, m.d_q, m.d_gate, 1, n_q, HD);
    gemv_fp16(q, m.d_xn, m.w_attn_k, m.d_k, H, N_kv);
    gemv_fp16(q, m.d_xn, m.w_attn_v, m.d_v, H, N_kv);
    rms_norm_f32w(q, m.d_q, m.attn_q_norm, m.d_q, n_q,  HD, eps);
    rms_norm_f32w(q, m.d_k, m.attn_k_norm, m.d_k, n_kv, HD, eps);
    rope_partial(q, m.d_q, m.d_pos1, m.d_q, 1, n_q,  HD, cfg_.rope_dim, cfg_.rope_theta);
    rope_partial(q, m.d_k, m.d_pos1, m.d_k, 1, n_kv, HD, cfg_.rope_dim, cfg_.rope_theta);
    full_attention(q, m.d_q, m.d_k, m.d_v, m.kc, m.vc, m.d_ao,
                   1, uint32_t(pos), n_q, n_kv, HD, m.max_ctx);
    sigmoid_gate(q, m.d_ao, m.d_gate, m.d_ao, uint64_t(N_q));
    gemv_fp16(q, m.d_ao, m.w_attn_out, m.d_blk, N_q, H);
    residual_add(q, m.d_x, m.d_blk, m.d_x, uint64_t(H));
    // MoE FFN (fp16 per-expert loop; shared expert skipped for the draft).
    rms_norm_f32w(q, m.d_x, m.post_attn_norm, m.d_xn, 1, H, eps);
    moe_router(q, m.d_xn, m.ffn_gate_inp, ws_topk_idx_, ws_topk_w_, 1, H, E, K_top);
    std::vector<int32_t> idx(K_top); std::vector<sycl::half> wts(K_top);
    q.memcpy(idx.data(), ws_topk_idx_, K_top * sizeof(int32_t)).wait();
    q.memcpy(wts.data(), ws_topk_w_,   K_top * sizeof(sycl::half)).wait();
    q.memset(m.d_moe, 0, uint64_t(H) * sizeof(sycl::half)).wait();
    for (uint32_t k = 0; k < K_top; ++k) {
        const uint32_t e = uint32_t(idx[k]);
        const sycl::half* ge = m.exp_gate + uint64_t(e) * H * E_ffn;
        const sycl::half* ue = m.exp_up   + uint64_t(e) * H * E_ffn;
        const sycl::half* de = m.exp_down + uint64_t(e) * E_ffn * H;
        gemv_fp16(q, m.d_xn, ge, m.d_fg, H, E_ffn);
        gemv_fp16(q, m.d_xn, ue, m.d_fu, H, E_ffn);
        swiglu(q, m.d_fg, m.d_fu, m.d_fh, E_ffn);
        gemv_fp16(q, m.d_fh, de, m.d_tmp, E_ffn, H);
        scaled_add(q, m.d_tmp, wts[k], m.d_moe, H);
    }
    residual_add(q, m.d_x, m.d_moe, m.d_x, uint64_t(H));
    rms_norm_f32w(q, m.d_x, m.shead_norm, m.d_xn, 1, H, eps);
    gemv_fp16(q, m.d_xn, m.w_lm_head, m.d_logits1, H, vocab);
}

void QwenModel::mtp_draft(sycl::queue& q, const sycl::half* h_last, int32_t tn,
                          uint32_t p_base, uint32_t K, std::vector<int32_t>& out) {
    out.clear();
    const uint32_t vocab = cfg_.vocab;
    int32_t e_tok = tn; const sycl::half* h_src = h_last;
    std::vector<sycl::half> row(vocab);
    for (uint32_t j = 0; j < K; ++j) {
        mtp_head_forward(q, h_src, e_tok, int32_t(p_base + j));
        q.wait();
        q.memcpy(row.data(), mtp_.d_logits1, uint64_t(vocab) * sizeof(sycl::half)).wait();
        int32_t gtok = mtp_argmax_row(row.data(), vocab);
        out.push_back(gtok);
        h_src = mtp_.d_x; e_tok = gtok;
    }
}

std::string QwenModel::spec_generate(sycl::queue& q, KvCache& kv, DeltaNetState& dn,
                                     sycl::half* h_last, int32_t tn,
                                     uint32_t start_pos, uint32_t max_new,
                                     uint32_t K, const SpecEmit& emit) {
    if (!mtp_.loaded) return "spec_generate: MTP head not loaded";
    if (K == 0) return "spec_generate: K == 0";
    const uint32_t H = cfg_.hidden, vocab = cfg_.vocab;
    const uint32_t L_full = cfg_.n_layers / cfg_.full_attn_interval;
    if (ws_T_ < K) { auto e = ensure_workspace(K); if (!e.empty()) return "spec ws: " + e; }
    Qwen35SpecCheckpoint ckpt;
    if (auto e = ckpt.init(*alloc_, dn, K); !e.empty()) return "spec ckpt: " + e;
    auto* d_all    = sycl::malloc_device<sycl::half>(uint64_t(K) * vocab, q);
    auto* d_hid    = sycl::malloc_device<sycl::half>(uint64_t(K) * H, q);
    auto* d_logits = sycl::malloc_device<sycl::half>(vocab, q);
    auto* d_ids    = sycl::malloc_device<int32_t>(K, q);
    if (!d_all || !d_hid || !d_logits || !d_ids) {
        if (d_all) sycl::free(d_all, q); if (d_hid) sycl::free(d_hid, q);
        if (d_logits) sycl::free(d_logits, q); if (d_ids) sycl::free(d_ids, q);
        return "spec_generate: scratch alloc failed";
    }
    std::vector<sycl::half> Lrow(uint64_t(K) * vocab), Hrow(uint64_t(K) * H);
    std::string ret; uint32_t emitted = 0, p = start_pos; bool first_round = true, abort = false;
    uint64_t spec_rounds = 0, spec_total_acc = 0;
    while (emitted < max_new && !abort) {
        std::vector<int32_t> drafted;
        mtp_draft(q, h_last, tn, /*p_base=*/0, K, drafted);
        std::vector<uint32_t> kv_len_snap(L_full);
        for (uint32_t l = 0; l < L_full; ++l) kv_len_snap[l] = kv.length(l);
        std::vector<int32_t> vin(K); vin[0] = tn;
        for (uint32_t j = 1; j < K; ++j) vin[j] = drafted[j - 1];
        q.memcpy(d_ids, vin.data(), K * sizeof(int32_t)).wait();
        forward(q, d_ids, K, p, kv, dn, d_logits, d_all, d_hid, &ckpt).wait();
        q.memcpy(Lrow.data(), d_all, uint64_t(K) * vocab * sizeof(sycl::half)).wait();
        q.memcpy(Hrow.data(), d_hid, uint64_t(K) * H * sizeof(sycl::half)).wait();
        std::vector<int32_t> targ(K);
        for (uint32_t j = 0; j < K; ++j) targ[j] = mtp_argmax_row(Lrow.data() + uint64_t(j) * vocab, vocab);
        uint32_t n = 0;
        for (uint32_t j = 1; j < K; ++j) { if (drafted[j - 1] == targ[j - 1]) ++n; else break; }
        const int32_t bonus = targ[n];
        const uint32_t accepted = n + 1;
        if (accepted < K) {
            if (auto e = ckpt.commit_to_n(q, dn, accepted); !e.empty()) { ret = "spec commit: " + e; abort = true; }
            for (uint32_t l = 0; l < L_full; ++l) kv.set_length(l, kv_len_snap[l] + accepted);
        }
        auto do_emit = [&](int32_t id) -> bool {
            if (emitted >= max_new) return false;
            ++emitted; if (!emit(id)) { abort = true; return false; } return true;
        };
        if (!abort && first_round) { (void)do_emit(tn); }
        for (uint32_t j = 0; j < n && !abort && emitted < max_new; ++j) do_emit(drafted[j]);
        if (!abort && emitted < max_new) do_emit(bonus);
        tn = bonus;
        q.memcpy(h_last, d_hid + uint64_t(accepted - 1) * H, uint64_t(H) * sizeof(sycl::half)).wait();
        p += accepted; first_round = false;
        ++spec_rounds; spec_total_acc += accepted;
    }
    std::fprintf(stderr, "[spec] mean accept = %.3f / round (K=%u, %llu rounds)\n",
                 spec_rounds ? double(spec_total_acc) / double(spec_rounds) : 0.0,
                 K, (unsigned long long)spec_rounds);
    sycl::free(d_all, q); sycl::free(d_hid, q); sycl::free(d_logits, q); sycl::free(d_ids, q);
    return ret;
}

}  // namespace ie
