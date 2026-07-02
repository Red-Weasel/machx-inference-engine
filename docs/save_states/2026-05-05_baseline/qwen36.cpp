// src/model/qwen36.cpp — Qwen3.6-35B-A3B model loader + 40-layer forward pass.
//
// Cross-references: research/04 (model spec), research/05 (DeltaNet), research/06 (MoE).
// This file is purely orchestration over the kernels built in phases 1-7.

#include "ie/qwen36.hpp"

#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"

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

#if IE_ENABLE_MOE_DOWN_TILE
#  define IE_MOE_DN_PK4K moe_prefill_down_packed_q4k_v2
#  define IE_MOE_DN_PK6K moe_prefill_down_packed_q6k_v2
#else
#  define IE_MOE_DN_PK4K moe_prefill_down_packed_q4k
#  define IE_MOE_DN_PK6K moe_prefill_down_packed_q6k
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
                                           uint64_t expert_stride_bytes,
                                           const std::vector<sycl::event>& deps = {});
sycl::event moe_prefill_down_packed_q6k_v2(sycl::queue& q,
                                           const sycl::half* h_packed,
                                           const void* down_W,
                                           const uint32_t* expert_offsets,
                                           sycl::half* out_packed,
                                           uint32_t E, uint32_t H, uint32_t E_ffn,
                                           uint64_t expert_stride_bytes,
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

// Dispatch GEMV by quant dtype.
sycl::event gemv_q(sycl::queue& q, const sycl::half* A, QuantPtr W,
                   sycl::half* y, uint32_t K, uint32_t N,
                   const std::vector<sycl::event>& deps = {}) {
    if (W.dt == DType::kQ4_K) return gemv_q4_K(q, A, W.p, y, K, N, deps);
    if (W.dt == DType::kQ6_K) return gemv_q6_K(q, A, W.p, y, K, N, deps);
    return {};
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
    if (W.dt == DType::kQ4_K) {
        constexpr uint32_t M_TILE = 8;
        // Use the joint_matrix XMX variant when shape constraints are met
        // (N % 64 == 0, K % 256 == 0).  Falls back to scalar otherwise.
        // Set IE_NO_XMX=1 to disable for A/B testing.
        // IE_ENABLE_GEMM_Q4K_ESIMD=1 routes through gemm_q4_K_esimd
        // (cooperative SG block-read variant) instead of gemm_q4_K_xmx.
        static const bool xmx_disabled = std::getenv("IE_NO_XMX") != nullptr;
        const bool use_xmx = !xmx_disabled && (N % 64 == 0) && (K % 256 == 0);
        sycl::event last;
        for (uint32_t m = 0; m < T; m += M_TILE) {
            const uint32_t mc = std::min(M_TILE, T - m);
            const sycl::half* a_t = A + uint64_t(m) * K;
            sycl::half*       y_t = y + uint64_t(m) * N;
            if (use_xmx) {
#if IE_ENABLE_GEMM_Q4K_ESIMD
                last = (m == 0) ? gemm_q4_K_esimd(q, a_t, W.p, y_t, mc, K, N, deps)
                                : gemm_q4_K_esimd(q, a_t, W.p, y_t, mc, K, N);
#else
                last = (m == 0) ? gemm_q4_K_xmx(q, a_t, W.p, y_t, mc, K, N, deps)
                                : gemm_q4_K_xmx(q, a_t, W.p, y_t, mc, K, N);
#endif
            } else {
                last = (m == 0) ? gemm_q4_K(q, a_t, W.p, y_t, mc, K, N, deps)
                                : gemm_q4_K(q, a_t, W.p, y_t, mc, K, N);
            }
        }
        return last;
    }
    if (W.dt == DType::kQ6_K) {
        constexpr uint32_t M_TILE = 8;
        // Mirror Q4_K dispatch: prefer XMX joint_matrix when shape allows
        // (N % 64 == 0, K % 256 == 0), fall back to scalar gemm_q6_K
        // otherwise.  IE_NO_XMX=1 forces the scalar path for A/B testing.
        static const bool xmx_disabled = std::getenv("IE_NO_XMX") != nullptr;
        const bool use_xmx = !xmx_disabled && (N % 64 == 0) && (K % 256 == 0);
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
        if (ti->dtype != DType::kQ4_K && ti->dtype != DType::kQ6_K) {
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
        w.ffn_gate_exps       = LQ("ffn_gate_exps.weight");      if (!err.empty()) return std::string("layer ") + std::to_string(L) + " ffn_gate_exps: " + err;
        w.ffn_up_exps         = LQ("ffn_up_exps.weight");        if (!err.empty()) return std::string("layer ") + std::to_string(L) + " ffn_up_exps: " + err;
        w.ffn_down_exps       = LQ("ffn_down_exps.weight");      if (!err.empty()) return std::string("layer ") + std::to_string(L) + " ffn_down_exps: " + err;
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
    // prefills fall through to the fused-per-token path).
    const uint32_t max_T_moe = std::min<uint32_t>(max_T, 4096u);
    ws_T_moe_           = max_T_moe;
    ws_moe_h_packed_    = alloc_half(uint64_t(max_T_moe) * cfg_.experts_topk * E_ffn);
    ws_moe_out_packed_  = alloc_half(uint64_t(max_T_moe) * cfg_.experts_topk * H);
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

sycl::event QwenModel::forward(sycl::queue& q,
                               const int32_t* input_ids, uint32_t T, uint32_t start_pos,
                               KvCache& kv, DeltaNetState& dn,
                               sycl::half* out_logits) {
    if (T == 0) return {};

    if (ws_T_ < T) {
        auto e = ensure_workspace(T);
        if (!e.empty()) {
            std::fprintf(stderr, "ensure_workspace: %s\n", e.c_str());
            return {};
        }
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
    {
        std::vector<int32_t> pos(T);
        for (uint32_t t = 0; t < T; ++t) pos[t] = int32_t(start_pos + t);
        q.memcpy(ws_positions_, pos.data(), T * sizeof(int32_t)).wait();
    }

    // 1. Embedding lookup → ws_x_
    if (token_embd_dtype_ == DType::kQ4_K) {
        embedding_lookup_q4k(q, input_ids, token_embd_, ws_x_, T, H);
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

        // 2a. attn_norm (1+w) → ws_x_normed_
        profile_section("attn_norm", [&]{
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

            // Q proj (Q + gate per-head fold-in): hidden → 8192
            gemv_q_T(q, ws_x_normed_, w.attn_q, ws_q_full_, H, N_qg, T);
            dump_int(ws_q_full_, N_qg, "I1qproj_");
            // Split Q | gate per head — HF chunks `[..., n_heads, 2*head_dim]` along last dim.
            split_q_gate_per_head(q, ws_q_full_, ws_q_split_, ws_gate_split_,
                                  T, cfg_.n_q_heads, H_d);
            dump_int(ws_q_split_,    N_q, "I2qsplt_");

            // K, V projs
            gemv_q_T(q, ws_x_normed_, w.attn_k, ws_k_, H, N_kv, T);
            gemv_q_T(q, ws_x_normed_, w.attn_v, ws_v_, H, N_kv, T);

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
            gemv_q_T(q, ws_attn_out_, w.attn_output, ws_attn_block_, N_q, H, T);
            // (post-O-proj is captured by the existing dump_attn(L) below
            // via the A03.bin file — no separate intermediate dump needed.)
            });  // end profile_section attn_full_block
        } else {
            profile_section("attn_dn_block", [&]{
            const uint32_t dn_idx = L - (L / cfg_.full_attn_interval);   // 0..30
            const uint32_t SI2 = SI * 2;

            profile_section("dn_qkv_proj", [&]{
                gemv_q_T(q, ws_x_normed_, w.attn_qkv, ws_qkv_, H, SI2, T);
            });

            sycl::half* conv_state_layer = dn.conv_state_ptr() +
                uint64_t(dn_idx) * dn.conv_elems_per_layer();
            profile_section("dn_conv1d_silu", [&]{
                depthwise_conv1d_causal(q, ws_qkv_, w.ssm_conv1d_fp16,
                                        conv_state_layer, ws_qkv_silu_,
                                        T, SI2, cfg_.ssm_conv_kernel);
            });

            profile_section("dn_alpha_beta_gate", [&]{
#if IE_FUSE_SSM_AB
                // Fused alpha + beta when both are Q4_K (always true in this
                // model) and T==1 (decode).  T>1 prefill keeps the original
                // 2-call path since it goes through gemm_q4_K, not gemv_q4_K.
                if (T == 1 &&
                    w.ssm_alpha.dt == DType::kQ4_K &&
                    w.ssm_beta.dt  == DType::kQ4_K) {
                    gemv_q4_K_dual_ssm(q, ws_x_normed_,
                                       w.ssm_alpha.p, w.ssm_beta.p,
                                       ws_alpha_fp16_, ws_beta_fp16_,
                                       H, SVH);
                } else {
                    gemv_q_T(q, ws_x_normed_, w.ssm_alpha, ws_alpha_fp16_, H, SVH, T);
                    gemv_q_T(q, ws_x_normed_, w.ssm_beta,  ws_beta_fp16_,  H, SVH, T);
                }
#else
                gemv_q_T(q, ws_x_normed_, w.ssm_alpha, ws_alpha_fp16_, H, SVH, T);
                gemv_q_T(q, ws_x_normed_, w.ssm_beta,  ws_beta_fp16_,  H, SVH, T);
#endif
                gemv_q_T(q, ws_x_normed_, w.attn_gate, ws_z_fp16_, H, SI, T);
                compute_g_beta_h16(q, ws_alpha_fp16_, ws_beta_fp16_,
                                   w.ssm_a, w.ssm_dt_bias,
                                   ws_g_fp32_, ws_beta_fp32_, T, SVH);
            });

            profile_section("dn_qkv_split_norm", [&]{
                cast_qkv_split_fp16_to_fp32(q, ws_qkv_silu_,
                                            ws_q_fp32_pre_, ws_k_fp32_pre_, ws_v_fp32_,
                                            T, SKH * SHD, SVH * SHD);
                repeat_interleave_heads(q, ws_q_fp32_pre_, ws_q_fp32_, T, SKH, SHD, /*repeat=*/2);
                repeat_interleave_heads(q, ws_k_fp32_pre_, ws_k_fp32_, T, SKH, SHD, /*repeat=*/2);
                const float qscale = 1.0f / sycl::sqrt(float(SHD));
                l2_norm_scale(q, ws_q_fp32_, ws_q_fp32_, T * SVH, SHD, qscale, 1e-6f);
                l2_norm_scale(q, ws_k_fp32_, ws_k_fp32_, T * SVH, SHD, 1.0f,    1e-6f);
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
                deltanet_recurrence(q, ws_q_fp32_, ws_k_fp32_, ws_v_fp32_,
                                    ws_g_fp32_, ws_beta_fp32_, state_layer,
                                    ws_recurrence_out_, /*B=*/1, T, SVH, SHD, SHD);
            });

            profile_section("dn_gated_norm", [&]{
                gated_rms_norm(q, ws_recurrence_out_, ws_z_fp16_, w.ssm_norm_fp16,
                               ws_gated_norm_, T * SVH, SHD, cfg_.rms_eps);
            });

            profile_section("dn_ssm_out", [&]{
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

        if (can_fuse_moe && T == 1) {
            // T=1 fused decode (each expert has at most 1 token).
            int32_t* idx_t  = ws_topk_idx_;
            sycl::half* w_t = ws_topk_w_;
            profile_section("moe_decode_gate_up_silu", [&]{
                moe_decode_gate_up_silu_q4k(q, ws_x_normed_,
                                            w.ffn_gate_exps.p, w.ffn_up_exps.p,
                                            idx_t, ws_h_routed_,
                                            H, E_ffn, K_top, s_g);
            });
            profile_section("moe_decode_down", [&]{
                if (w.ffn_down_exps.dt == DType::kQ4_K) {
                    moe_decode_down_q4k(q, ws_h_routed_, w.ffn_down_exps.p,
                                        idx_t, w_t, ws_attn_block_,
                                        H, E_ffn, K_top, s_d);
                } else {
                    moe_decode_down_q6k(q, ws_h_routed_, w.ffn_down_exps.p,
                                        idx_t, w_t, ws_attn_block_,
                                        H, E_ffn, K_top, s_d);
                }
            });
        } else if (can_fuse_moe && T >= 64 && T < 2048 && T <= ws_T_moe_ &&
                   ws_moe_h_packed_ && ws_moe_out_packed_ &&
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

            // Stage 1: gate + up + silu (all experts in one launch).
            // XMX variant tried (8 SGs / WG, 4 gate + 4 up) — still slower
            // than the scalar version (~ -10-15%).  Likely the SLM round-trip
            // for dequanted Q4_K weights costs more than the mat_mad savings,
            // since scalar inlines dequant directly into the multiply.  The
            // multi-expert pattern (many small WGs, fragmented work per
            // expert) compounds this.  Kept in gemm_q4k_xmx.cpp for a future
            // dedicated rewrite — possibly with cl_intel_subgroup_2d_block_io
            // for the Q4_K reads to compress the dequant cost.
            moe_prefill_gate_up_silu_q4k(q, ws_x_normed_,
                                         w.ffn_gate_exps.p, w.ffn_up_exps.p,
                                         ws_moe_expert_offsets_,
                                         ws_moe_token_idx_,
                                         ws_moe_h_packed_,
                                         E, H, E_ffn, s_g);
            // Stage 2: down (all experts in one launch, atomics-free).
            // Kernel selected by IE_ENABLE_MOE_DOWN_TILE (v2 = M_TILE=8 SLM
            // amortized; v1 = original, no amortization).
            if (w.ffn_down_exps.dt == DType::kQ4_K) {
                IE_MOE_DN_PK4K(q, ws_moe_h_packed_, w.ffn_down_exps.p,
                               ws_moe_expert_offsets_,
                               ws_moe_out_packed_,
                               E, H, E_ffn, s_d);
            } else {
                IE_MOE_DN_PK6K(q, ws_moe_h_packed_, w.ffn_down_exps.p,
                               ws_moe_expert_offsets_,
                               ws_moe_out_packed_,
                               E, H, E_ffn, s_d);
            }
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
                                            H, E_ffn, K_top, s_g);
                if (w.ffn_down_exps.dt == DType::kQ4_K) {
                    moe_decode_down_q4k(q, ws_h_routed_, w.ffn_down_exps.p,
                                        idx_t, w_t, y_t,
                                        H, E_ffn, K_top, s_d);
                } else {
                    moe_decode_down_q6k(q, ws_h_routed_, w.ffn_down_exps.p,
                                        idx_t, w_t, y_t,
                                        H, E_ffn, K_top, s_d);
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
        shared_expert_gate(q, ws_x_normed_, w.ffn_gate_inp_shexp, ws_sh_g_, T, H);
        if (T == 1) {
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
#else
            gemv_q(q, ws_x_normed_, w.ffn_gate_shexp, ws_gate_o_, H, E_ffn);
            gemv_q(q, ws_x_normed_, w.ffn_up_shexp,   ws_up_o_,   H, E_ffn);
#endif
            swiglu(q, ws_gate_o_, ws_up_o_, ws_h_o_, E_ffn);
            gemv_q(q, ws_h_o_, w.ffn_down_shexp, ws_eo_, E_ffn, H);
            scaled_add_dev_scalar(q, ws_eo_, ws_sh_g_, ws_attn_block_, H);
        } else {
            std::vector<sycl::half> hsh_g(T);
            q.memcpy(hsh_g.data(), ws_sh_g_, T * sizeof(sycl::half)).wait();
            for (uint32_t t = 0; t < T; ++t) {
                sycl::half* x_t = ws_x_normed_ + uint64_t(t) * H;
                gemv_q(q, x_t, w.ffn_gate_shexp, ws_gate_o_, H, E_ffn);
                gemv_q(q, x_t, w.ffn_up_shexp,   ws_up_o_,   H, E_ffn);
                swiglu(q, ws_gate_o_, ws_up_o_, ws_h_o_, E_ffn);
                gemv_q(q, ws_h_o_, w.ffn_down_shexp, ws_eo_, E_ffn, H);
                scaled_add(q, ws_eo_, hsh_g[t],
                           ws_attn_block_ + uint64_t(t) * H, H);
            }
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
    }

    // 3. Final norm (1+w bake-in: stored weight is `1+w`, formula is plain weight*x_normed)
    profile_section("final_norm", [&]{
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

    // 4. lm_head GEMV — only the LAST token's logits are needed.
    //    Big shape (K=2048, N=vocab≈248k); XMX path wins here on Q4_K.
    sycl::half* last_x = ws_x_normed_ + uint64_t(T - 1) * H;
    QuantPtr lm_head{output_, output_dtype_};
    // lm_head is M=1 (single token).  XMX path was tried and consistently
    // ~5% slower — for M=1 there's no row-amortization, and the XMX SLM
    // dequant round-trip is pure overhead vs the scalar gemv_q's inline
    // dequant.  Scalar wins here.
    auto run_lmhead = [&]() -> sycl::event {
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

}  // namespace ie
