// src/model/dense_dispatch.hpp — P2 internal: upload + GEMV dispatch helpers
// for the dense-model path (DenseModel).
//
// COPIED from src/model/qwen36.cpp's anonymous namespace (lines 168-392 at
// commit b2def5c): upload<T>, upload_quant (renamed upload_quant_dense),
// gemv_q, q8_decode_enabled, gemv_q_i8, gemv_q_T.  Deliberate duplication
// per the P2 iron rule — the crown file (qwen36.cpp) stays byte-identical;
// a dedup refactor is allowed only AFTER P2 ships, with a full crown A/B.
//
// TWO deliberate changes vs the qwen36.cpp originals:
//   1. gemv_q / gemv_q_T grow a DType::kF16 branch — gemv_fp16 at T==1,
//      gemm_fp16 + cast_fp32_to_fp16 at T>1 (B is the load-time transposed
//      [K,N] copy; fp32 C scratch exactly like the E1 dequant path).
//   2. HARD-FAIL dtype policy: upload_quant_dense accepts ONLY
//      {Q4_K, Q6_K, F16}.  Anything else (Q8_0, Q5_K, Q4_0, ...) returns a
//      load error naming the tensor and dtype.  This kills qwen36.cpp's
//      silent-garbage failure mode (upload accepted Q8_0/Q5_K but gemv_q
//      returned an empty event for them) for the new path.
//
// F16 upload performs a host transpose: GGUF stores F16 weights [N, K]
// row-major (ggml: shape[0] = K = contiguous dim); we store [K, N] so that
// gemv_fp16 lanes read consecutive n (coalesced) and gemm_fp16 consumes the
// buffer directly as B.
//
// This header is internal to src/model/dense_transformer.cpp.

#pragma once

#include "ie/dense_transformer.hpp"
#include "ie/dequant.hpp"
#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"   // block_q8_1x (batched-verify scratch sizing)

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace ie {

// Forward decl — defined in src/ops/elementwise.cpp (same forward-decl
// pattern as qwen36.cpp:158-165; hoisted here for the dense forward).
sycl::event residual_add_rms_norm_fused(sycl::queue& q,
                                        sycl::half* ws_x,
                                        const sycl::half* ws_block,
                                        const float* w_norm,
                                        sycl::half* ws_normed,
                                        uint32_t n_rows, uint32_t hidden,
                                        float eps,
                                        const std::vector<sycl::event>& deps = {});

namespace dense {

// Granite-3.x value-gated scalar multiply (in-place): x[i] *= c, in fp32 then
// stored back fp16.  Dense-path-only helper — the crown applies no such global
// scale.  Every call site is guarded by a `!= default-multiplier` check, so for
// all non-Granite models this kernel is never launched (byte-identical path).
inline sycl::event scale_inplace(sycl::queue& q, sycl::half* x, float c,
                                 uint64_t n) {
    return q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        x[i] = sycl::half(float(x[i]) * c);
    });
}

// Helper: load a tensor from GGUF onto device, return device pointer.
// `expected` may be DType::kCount to skip the dtype check (any).
// [copied: qwen36.cpp:172-188]
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

// Weight loader for the dense path.  [copied: qwen36.cpp:191-208, with the
// hard-fail dtype policy + F16 transpose-upload changes documented above]
inline DenseQuantPtr upload_quant_dense(DeviceAllocator& alloc,
                                        const GgufTensorInfo* t,
                                        std::vector<void*>& owned,
                                        std::string& err) {
    DenseQuantPtr out;
    if (!t) { err = "tensor not found"; return out; }
    if (t->dtype == DType::kF16) {
        // Host transpose [N, K] → [K, N] into a staging buffer, upload the
        // transposed copy only.  ~8 MB per qwen3-8b attn_v tensor.
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
    if (t->dtype != DType::kQ4_K && t->dtype != DType::kQ6_K) {
        err = std::string("tensor '") + std::string(t->name) +
              "' dtype unsupported by dense path: " +
              std::string(type_name(t->dtype)) +
              " (supported: Q4_K, Q6_K, F16) — refusing to load";
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

// EXL3 weight loader: uploads the packed trellis (kEXL3) verbatim + its two F16
// side-vectors suh[K]/svh[N]. The kernel (gemv_exl3_forward) decodes natively at
// inference — nothing is dequantized here. bits = trellis.shape[0]/16 (the trellis
// is [16*bits, N/16, K/16]; see DType::kEXL3). EXL3 weights are HF-natural NEOX
// order, so q/k are NEVER row-un-permuted (the caller skips the llama permute).
inline DenseQuantPtr upload_exl3(DeviceAllocator& alloc,
                                 const GgufTensorInfo* trellis,
                                 const GgufTensorInfo* suh,
                                 const GgufTensorInfo* svh,
                                 std::vector<void*>& owned, std::string& err) {
    DenseQuantPtr out;
    if (!trellis || !suh || !svh) { err = "exl3: missing trellis/suh/svh sibling"; return out; }
    if (trellis->dtype != DType::kEXL3) { err = "exl3: trellis dtype not kEXL3"; return out; }
    auto up = [&](const GgufTensorInfo* ti) -> void* {
        void* d = alloc.malloc(ti->nbytes);
        if (!d) { err = "exl3: malloc failed"; return nullptr; }
        alloc.queue().memcpy(d, ti->data, ti->nbytes).wait();
        owned.push_back(d);
        return d;
    };
    void* dt = up(trellis); if (!err.empty()) return out;
    void* ds = up(suh);     if (!err.empty()) return out;
    void* dv = up(svh);     if (!err.empty()) return out;
    out.p = dt; out.dt = DType::kEXL3; out.suh = ds; out.svh = dv;
    out.bits = uint32_t(trellis->shape[0] / 16);
    return out;
}

// Q5_K/Q8_0 → device F16 [K,N] (transposed dequant), consumable by the F16
// branch of gemv_q / gemv_q_T. Copied from qwen3moe.cpp:30-53 (copy-not-hoist;
// the qwen3moe/qwen35 copies stay — P2/P3 iron rule). SIBLING of
// upload_quant_dense: that function's Q4_K/Q6_K/F16 hard-fail policy is
// UNCHANGED; only loaders that opt in call the _auto path below.
inline DenseQuantPtr upload_dequant_to_fp16(DeviceAllocator& alloc,
                                            const GgufTensorInfo* t,
                                            std::vector<void*>& owned,
                                            std::string& err) {
    DenseQuantPtr out;
    if (!t) { err = "tensor not found"; return out; }
    if (t->n_dims != 2) { err = "dequant_to_fp16: expected 2-D weight"; return out; }
    const uint32_t K = uint32_t(t->shape[0]), N = uint32_t(t->shape[1]);
    if (K == 0 || N == 0) { err = "dequant_to_fp16: zero dim"; return out; }
    void* packed = alloc.malloc(t->nbytes);
    if (!packed) { err = "malloc failed (packed)"; return out; }
    alloc.queue().memcpy(packed, t->data, t->nbytes).wait();
    auto* d = static_cast<sycl::half*>(alloc.malloc(uint64_t(K) * N * sizeof(sycl::half)));
    if (!d) { alloc.free(packed); err = "malloc failed (fp16)"; return out; }
    sycl::event e;
    if      (t->dtype == DType::kQ5_K) e = dequant_q5_K_to_Bt(alloc.queue(), packed, d, K, N);
    else if (t->dtype == DType::kQ8_0) e = dequant_q8_0_to_Bt(alloc.queue(), packed, d, K, N);
    else { alloc.free(packed); alloc.free(d);
           err = std::string("dequant_to_fp16: unsupported ") + std::string(type_name(t->dtype));
           return out; }
    e.wait();
    alloc.free(packed);
    owned.push_back(d);
    out.p = d; out.dt = DType::kF16;
    return out;
}
// Auto upload: Q5_K/Q8_0 → fp16 dequant; Q4_K/Q6_K/F16 ride upload_quant_dense.
// Sibling of upload_quant_dense (NOT a widening of it) — opt-in only, so the
// crown/validated dense loads that keep calling upload_quant_dense are
// byte-identical. Unblocks any Q5_K/Q8_0 dense GGUF.
inline DenseQuantPtr upload_quant_dense_auto(DeviceAllocator& alloc,
                                             const GgufTensorInfo* t,
                                             std::vector<void*>& owned,
                                             std::string& err) {
    if (!t) { err = "tensor not found"; return {}; }
    if (t->dtype == DType::kQ5_K || t->dtype == DType::kQ8_0)
        return upload_dequant_to_fp16(alloc, t, owned, err);
    return upload_quant_dense(alloc, t, owned, err);
}

// Phi-unblock Task 2: slice output-rows [n0, n0+n_slice) out of a packed [K,N]
// GGUF tensor and upload via the auto path (Q5_K/Q8_0 dequant-to-fp16;
// Q4_K/Q6_K/F16 ride upload_quant_dense). row_bytes = nbytes / N — each output
// row is a contiguous superblock span (same invariant as
// upload_quant_dense_permuted), so a row-range slice is a pure byte copy, NO
// requant. The synthetic GgufTensorInfo describes the slice [K, n_slice].
// Used for the fused attn_qkv V-slice and the fused ffn_up gate/up slices (no
// permute); the permuted variant below handles the Q/K slices.
inline DenseQuantPtr upload_quant_dense_row_slice(
        DeviceAllocator& alloc, const GgufTensorInfo* t,
        uint64_t n0, uint64_t n_slice,
        std::vector<void*>& owned, std::string& err) {
    DenseQuantPtr out;
    if (!t) { err = "tensor not found"; return out; }
    const uint64_t K = t->shape[0], N = t->shape[1];
    if (t->n_dims != 2 || K == 0 || N == 0 || n0 + n_slice > N) {
        err = std::string("row_slice '") + std::string(t->name) + "': bad geometry";
        return out;
    }
    // F16 is [N,K] row-major in the GGUF (per upload_quant_dense's transpose
    // comment) → a fused F16 weight cannot be sliced by a flat byte offset the
    // same way. No phi3 fused tensor is F16 (attn_qkv=Q5_K, ffn_up=Q4_K), so
    // refuse rather than slice incorrectly.
    if (t->dtype == DType::kF16) {
        err = std::string("row_slice '") + std::string(t->name) +
              "': F16 fused slice unsupported";
        return out;
    }
    const uint64_t row_bytes = t->nbytes / N;
    if (row_bytes * N != t->nbytes) {
        err = std::string("row_slice '") + std::string(t->name) + "': nbytes % N != 0";
        return out;
    }
    // Slice-view: same dtype/K, the contiguous run of n_slice output rows.
    GgufTensorInfo s = *t;
    s.data     = t->data + n0 * row_bytes;
    s.nbytes   = n_slice * row_bytes;
    s.shape[1] = n_slice;
    return upload_quant_dense_auto(alloc, &s, owned, err);
}

// Forward decl — defined just below; the permuted slicer delegates Q4_K/Q6_K
// slices to it.
inline DenseQuantPtr upload_quant_dense_permuted(DeviceAllocator& alloc,
                                                 const GgufTensorInfo* t,
                                                 std::vector<void*>& owned,
                                                 std::string& err,
                                                 const std::vector<uint32_t>& perm);

// Phi-unblock Task 2: permuted row-slice — slice output-rows [n0,n0+n_slice)
// then apply the Llama NEOX Q/K un-permute (perm has size n_slice). Used for the
// Q and K slices of the fused attn_qkv (V is never permuted → use the plain
// slicer above). An EMPTY perm means "slice only, no permute" (non-llama arch).
//
// THE BIGGEST-RISK PATH (plan Self-Review #1): the row-shuffle must happen on
// the QUANT superblock-rows BEFORE dequant, never on a flat fp16 buffer.
//   - Q4_K/Q6_K: each output row is a contiguous (K/256)*block_bytes span, so we
//     hand a slice-view to upload_quant_dense_permuted, which byte-shuffles rows
//     then uploads the quant (no dequant). This is the EXACT same row-shuffle the
//     split-tensor llama path uses on attn_q/attn_k (dense_transformer.cpp:146-151).
//   - Q5_K/Q8_0: upload_quant_dense_permuted hard-fails these. But the row is
//     still a contiguous superblock span (Q5_K = 176 B/superblock, K/256 per row),
//     so we byte-shuffle the slice's rows into a host staging buffer FIRST, point
//     a GgufTensorInfo at it, then upload_quant_dense_auto (which dequants the
//     ALREADY-un-permuted superblocks to fp16). Same byte-exact shuffle, just
//     before the dequant instead of inside the quant uploader.
inline DenseQuantPtr upload_quant_dense_row_slice_permuted(
        DeviceAllocator& alloc, const GgufTensorInfo* t,
        uint64_t n0, uint64_t n_slice,
        std::vector<void*>& owned, std::string& err,
        const std::vector<uint32_t>& perm) {
    DenseQuantPtr out;
    if (!t) { err = "tensor not found"; return out; }
    const uint64_t K = t->shape[0], N = t->shape[1];
    if (t->n_dims != 2 || K == 0 || N == 0 || n0 + n_slice > N) {
        err = std::string("row_slice_perm '") + std::string(t->name) + "': bad geometry";
        return out;
    }
    if (!perm.empty() && perm.size() != n_slice) {
        err = std::string("row_slice_perm '") + std::string(t->name) +
              "': perm size " + std::to_string(perm.size()) +
              " != n_slice " + std::to_string(n_slice);
        return out;
    }
    // No permute requested → identical to the plain slicer.
    if (perm.empty())
        return upload_quant_dense_row_slice(alloc, t, n0, n_slice, owned, err);

    if (t->dtype == DType::kF16) {
        err = std::string("row_slice_perm '") + std::string(t->name) +
              "': F16 fused slice unsupported";
        return out;
    }
    const uint64_t row_bytes = t->nbytes / N;
    if (row_bytes * N != t->nbytes) {
        err = std::string("row_slice_perm '") + std::string(t->name) + "': nbytes % N != 0";
        return out;
    }
    // Slice-view onto the contiguous run of n_slice output rows.
    GgufTensorInfo s = *t;
    s.data     = t->data + n0 * row_bytes;
    s.nbytes   = n_slice * row_bytes;
    s.shape[1] = n_slice;

    // Q4_K/Q6_K: the existing permuted uploader byte-shuffles the rows of the
    // slice view and uploads the quant directly (no dequant). Identical to the
    // validated split-tensor attn_q/attn_k path.
    if (s.dtype == DType::kQ4_K || s.dtype == DType::kQ6_K)
        return upload_quant_dense_permuted(alloc, &s, owned, err, perm);

    // Q5_K/Q8_0: byte-shuffle the superblock-rows in a host staging buffer
    // (dest row i ← src row perm[i]), then dequant-to-fp16 the un-permuted
    // superblocks via upload_quant_dense_auto.
    if (s.dtype == DType::kQ5_K || s.dtype == DType::kQ8_0) {
        std::vector<uint8_t> staging(s.nbytes);
        const uint8_t* src = s.data;
        for (uint64_t i = 0; i < n_slice; ++i)
            std::memcpy(staging.data() + i * row_bytes,
                        src + uint64_t(perm[i]) * row_bytes, row_bytes);
        GgufTensorInfo sp = s;
        sp.data = staging.data();   // host buffer; upload_quant_dense_auto memcpys it before staging goes out of scope
        return upload_quant_dense_auto(alloc, &sp, owned, err);
    }
    err = std::string("row_slice_perm '") + std::string(t->name) +
          "': dtype " + std::string(type_name(s.dtype)) + " unsupported";
    return out;
}

// P3a: row-permuted weight upload — the Llama Q/K un-permute. Reorders the N
// output rows (dest row i ← src row perm[i]) in a host staging copy, then
// uploads. For Q4_K/Q6_K each output row is a contiguous (K/256)×block_bytes
// byte span (ggml lays out the K superblocks contiguously per output column),
// so the shuffle is byte-exact — no requant. F16 applies perm to the output
// index during the existing [N,K]→[K,N] transpose. `perm` must have size N.
// Used only for attn_q/attn_k when arch == kLlama3 (see dense_transformer.cpp).
inline DenseQuantPtr upload_quant_dense_permuted(DeviceAllocator& alloc,
                                                 const GgufTensorInfo* t,
                                                 std::vector<void*>& owned,
                                                 std::string& err,
                                                 const std::vector<uint32_t>& perm) {
    DenseQuantPtr out;
    if (!t) { err = "tensor not found"; return out; }
    const uint64_t K = t->shape[0];   // ggml: shape[0] = contiguous (K)
    const uint64_t N = t->shape[1];
    if (t->n_dims != 2 || K == 0 || N == 0 || perm.size() != N) {
        err = std::string("permute upload '") + std::string(t->name) +
              "': bad geometry or perm size";
        return out;
    }
    if (t->dtype == DType::kF16) {
        if (t->nbytes != K * N * sizeof(uint16_t)) {
            err = std::string("permute upload f16 '") + std::string(t->name) + "': geometry";
            return out;
        }
        std::vector<uint16_t> staging(K * N);
        const auto* src = reinterpret_cast<const uint16_t*>(t->data);
        for (uint64_t n = 0; n < N; ++n) {
            const uint64_t s = perm[n];
            for (uint64_t k = 0; k < K; ++k)
                staging[k * N + n] = src[s * K + k];   // permuted + transposed [K,N]
        }
        void* d = alloc.malloc(t->nbytes);
        if (!d) { err = "malloc failed"; return out; }
        alloc.queue().memcpy(d, staging.data(), t->nbytes).wait();
        owned.push_back(d);
        out.p = d; out.dt = DType::kF16;
        return out;
    }
    if (t->dtype != DType::kQ4_K && t->dtype != DType::kQ6_K) {
        err = std::string("permute upload '") + std::string(t->name) +
              "': dtype " + std::string(type_name(t->dtype)) +
              " unsupported (Q4_K/Q6_K/F16)";
        return out;
    }
    const uint64_t row_bytes = t->nbytes / N;
    if (row_bytes * N != t->nbytes) {
        err = std::string("permute upload '") + std::string(t->name) +
              "': nbytes not divisible by N rows";
        return out;
    }
    std::vector<uint8_t> staging(t->nbytes);
    const auto* src = reinterpret_cast<const uint8_t*>(t->data);
    for (uint64_t i = 0; i < N; ++i)
        std::memcpy(staging.data() + i * row_bytes, src + perm[i] * row_bytes, row_bytes);
    void* d = alloc.malloc(t->nbytes);
    if (!d) { err = "malloc failed"; return out; }
    alloc.queue().memcpy(d, staging.data(), t->nbytes).wait();
    owned.push_back(d);
    out.p = d; out.dt = t->dtype;
    return out;
}

// P2 Task 4 (dense-only addition): the Q6_K SLM-slab GEMV variants
// (gemv_q6_K_slm / gemv_q6_K_slm_q8, taken for N >= 2048) stage a
// N_PER_WG(16) × (K/256) × 210 B weight slab in SLM.  At K = 12288 (qwen3
// ffn_down) that is 157.5 KiB > the 128 KiB WG SLM limit → the launch fails
// with UR_RESULT_ERROR_OUT_OF_RESOURCES.  Crown shapes (K ≤ 4096 → ≤ 52.5
// KiB) never hit this, so the kernel files stay untouched (iron rule 1);
// the dense path routes big-K Q6_K decode GEMVs through the scalar
// gemm_q6_K at M=1 instead (its SLM is M×K halfs = 24 KiB at K=12288).
inline bool q6k_slm_gemv_fits(uint32_t K) {
    constexpr uint32_t kWgSlmBytes = 128u * 1024u;
    return 16u * (K / 256u) * 210u <= kWgSlmBytes;       // sizeof(block_q6_K)=210
}
inline bool q6k_slm_gemv_q8_fits(uint32_t K) {
    constexpr uint32_t kWgSlmBytes = 128u * 1024u;
    // slab + staged q8 stream (qs words + d per 32-elem block)
    return 16u * (K / 256u) * 210u + (K / 32u) * (32u + 4u) <= kWgSlmBytes;
}

// P3b Task 1: int-dot Q6_K decode GEMV for the big-K ffn_down shape (K=12288),
// where the whole-column SLM slab does NOT fit so the dense path fell back to
// scalar gemm_q6_K at M=1 (140 GB/s, the profiled #1 decode bottleneck).
//
// STATUS (2026-06-10): the int-dot port is correctness-verified but MEASURES
// SLOWER than the scalar cliff it was meant to replace (~75 GB/s vs 140 GB/s),
// so it is kept DEFAULT-OFF. Root cause: the SAME int-dot Q6_K math hits
// 226 GB/s at the lm_head shape (K=4096, N=151936 → ~9.5k WGs) but only
// ~75 GB/s here (K=12288, N=4096 → 128 WGs) — the per-byte Q6_K dequant is
// ALU-bound and the big-K/small-N aspect ratio leaves too few output columns
// to hide the 48-super-block serial K-reduction. dp4a does not help a
// dequant-bound kernel; beating the cliff needs split-K cross-WG reduction or
// a repacked Q6_K layout, both out of Task-1 scope (see the plan BLOCKED note).
// Opt IN for A/B with IE_Q6K_KTILED=1; default keeps the proven gemm_q6_K path.
inline bool q6k_ktiled_enabled() {
    static const bool on = std::getenv("IE_Q6K_KTILED") != nullptr;
    return on;
}

// Per-model opt-in to oneDNN matmul on the prefill dequant→gemm path. The
// qwen35 dense-hybrid sets this true (oneDNN is ~1.65x faster at its shapes and
// it has no bit-exact gate); crown/qwen3-dense leave it false to preserve their
// bit-exact PPL gates (gemm_fp16). Process-global — one model loaded at a time.
inline bool& prefer_onednn() { static bool v = false; return v; }

// Dispatch GEMV by quant dtype.  [copied: qwen36.cpp:246-252, + F16 branch
// + the big-K Q6_K reroute documented above]
inline sycl::event gemv_q(sycl::queue& q, const sycl::half* A, DenseQuantPtr W,
                          sycl::half* y, uint32_t K, uint32_t N,
                          const std::vector<sycl::event>& deps = {}) {
    if (W.dt == DType::kQ4_K) return gemv_q4_K(q, A, W.p, y, K, N, deps);
    if (W.dt == DType::kQ6_K) {
        if (N >= 2048 && !q6k_slm_gemv_fits(K))
            return gemm_q6_K(q, A, W.p, y, /*M=*/1, K, N, deps);
        return gemv_q6_K(q, A, W.p, y, K, N, deps);
    }
    if (W.dt == DType::kF16)
        return gemv_fp16(q, A, static_cast<const sycl::half*>(W.p), y, K, N,
                         deps);
    if (W.dt == DType::kEXL3)
        return gemv_exl3_forward(q, A, W.p,
                                 static_cast<const sycl::half*>(W.suh),
                                 static_cast<const sycl::half*>(W.svh),
                                 y, K, N, W.bits, deps);
    return {};  // unreachable: upload_quant_dense hard-fails other dtypes
}

// P1a (2026-06-09): integer-dot decode dispatch.  `x_q8` (when non-null)
// holds quantize_q8_1 of the SAME vector as `x` — caller quantizes once per
// unique activation vector and reuses it across that vector's GEMVs.  Q4_K
// routes through the dp4a GEMV; other dtypes fall back to the fp16 path.
// [copied: qwen36.cpp:259-270]
inline bool q8_decode_enabled() {
    // v1.5-C: default ON (quality-verified; quant rides the attn_norm
    // launch).  Opt out with IE_NO_Q8_DECODE=1.
    static const bool off = std::getenv("IE_NO_Q8_DECODE") != nullptr;
    return !off;
}
inline sycl::event gemv_q_i8(sycl::queue& q, const sycl::half* x,
                             const void* x_q8, DenseQuantPtr W, sycl::half* y,
                             uint32_t K, uint32_t N) {
    if (x_q8 && W.dt == DType::kQ4_K)
        return gemv_q4_K_q8(q, x_q8, W.p, y, K, N);
    return gemv_q(q, x, W, y, K, N);
}

// Multi-token: dispatch the best multi-row kernel we have for this dtype.
// For Q4_K we use gemm_q4_K with M_TILE=8 chunks (amortizes the weight-read
// cost 8× across rows). For other dtypes (Q6_K) we fall back to a serial
// gemv_q loop until those grow a multi-row variant.
// [copied: qwen36.cpp:275-383, + F16 branch; the ESIMD prefill switch is
//  omitted — iron rule 4 says never touch those paths, the dense path
//  always uses gemm_q4_K_xmx]
// P-B (multi-GPU): the gemv_q_T prefill scratch (dequant→gemm) must be PER
// DEVICE — a process-global static buffer allocated on the first device and then
// touched by a kernel on a second device faults it (UR_RESULT_ERROR_DEVICE_LOST).
// Keyed by the queue's device; host runs one generation at a time so no lock is
// needed. Single-GPU is unchanged (one entry, same grow-on-demand behavior).
struct GemvPrefillScratch {
    sycl::half* bt = nullptr; uint64_t bt_cap = 0;   // [K,N] dequanted weight
    float*      c  = nullptr; uint64_t c_cap  = 0;   // [T+8,N] fp32 gemm output
    void*       q8 = nullptr; uint64_t q8_cap = 0;   // [T*K/32] block_q8_1x (batched verify)
};
inline GemvPrefillScratch& gemv_prefill_scratch(sycl::queue& q) {
    // Heap-allocated entries → stable addresses across cache growth. Leaked at
    // process exit (process-lifetime scratch, like the statics it replaces).
    static std::vector<std::pair<sycl::device, GemvPrefillScratch*>> cache;
    const sycl::device d = q.get_device();
    for (auto& e : cache) if (e.first == d) return *e.second;
    cache.emplace_back(d, new GemvPrefillScratch{});
    return *cache.back().second;
}

inline sycl::event gemv_q_T(sycl::queue& q, const sycl::half* A,
                            DenseQuantPtr W, sycl::half* y,
                            uint32_t K, uint32_t N, uint32_t T,
                            const std::vector<sycl::event>& deps = {}) {
    if (T == 0) return {};
    if (T == 1) return gemv_q(q, A, W, y, K, N, deps);
    // Spec-decode VERIFY (small T): batched int-dot W4A8 — read each Q4_K column
    // ONCE, dot against T staged Q8_1 activation rows.  Amortizes weight BW over
    // T instead of the T-independent XMX/dequant restream the prefill path uses
    // (a 4-token verify there costs ~16× a decode step). Only fires for small T
    // (2..16), which normal prefill never produces (T=256/512) and decode is
    // T==1 → the validated prefill/decode paths are byte-identical. Per-device
    // Q8_1 scratch sized [T*K/32]. Opt-out IE_QWEN35_NO_BATCHED_VERIFY=1.
    {
        static const bool batched_off =
            std::getenv("IE_QWEN35_NO_BATCHED_VERIFY") != nullptr;
        if (!batched_off && W.dt == DType::kQ4_K && T >= 2 && T <= 16 &&
            (K % 32 == 0)) {
            // ISUM-elimination variant (IE_QWEN35_VERIFY_ISUM): quantize to
            // block_q8_1s (carries per-16 half-block sums s0/s1) and run the new
            // gemv_q4_K_q8s_batched, which drops the redundant per-column isum
            // dp4a — ~half the inner dp4a, the only ALU lever that survives at
            // T=4 (XMX was slower). Lossless-gated; opt-in while A/B-ing.
            // Default ON (strictly better: ~+9% spec, lossless, new kernel that
            // touches nothing shared). Opt-out IE_QWEN35_NO_VERIFY_ISUM for A/B.
            static const bool use_isum =
                std::getenv("IE_QWEN35_NO_VERIFY_ISUM") == nullptr;
            GemvPrefillScratch& scr = gemv_prefill_scratch(q);
            const uint64_t blk_sz =
                use_isum ? sizeof(block_q8_1s) : sizeof(block_q8_1x);
            const uint64_t q8_need = (uint64_t(T) * K / 32) * blk_sz;
            if (q8_need > scr.q8_cap) {
                if (scr.q8) sycl::free(scr.q8, q);
                scr.q8 = sycl::malloc_device(q8_need, q);
                scr.q8_cap = scr.q8 ? q8_need : 0;
            }
            if (scr.q8) {
                if (use_isum) {
                    sycl::event qe =
                        quantize_q8_1s(q, A, scr.q8, uint64_t(T) * K, deps);
                    return gemv_q4_K_q8s_batched(q, scr.q8, W.p, y, K, N, T, {qe});
                }
                sycl::event qe =
                    quantize_q8_1(q, A, scr.q8, uint32_t(uint64_t(T) * K), deps);
                return gemv_q4_K_q8_batched(q, scr.q8, W.p, y, K, N, T, {qe});
            }
            // scratch alloc failed → fall through to the prefill path
        }
    }
    if (W.dt == DType::kF16) {
        // F16 weights were transposed to [K, N] at load — gemm_fp16 consumes
        // them DIRECTLY as B.  fp32 C scratch + cast, like the E1 path.
        static const bool gemm_ok_env = std::getenv("IE_NO_XMX") == nullptr;
        const bool gemm_ok = gemm_ok_env && (N % 64 == 0) && (K % 256 == 0);
        if (gemm_ok && prefer_onednn()) {
            // qwen35: F16 weights (dequanted Q5_K/Q8_0, alpha/beta) go straight
            // through oneDNN too — it outputs fp16 directly (no c_scr/cast).
            return gemm_fp16_onednn(q, A, static_cast<const sycl::half*>(W.p),
                                    y, T, N, K, deps);
        }
        if (gemm_ok) {
            GemvPrefillScratch& scr = gemv_prefill_scratch(q);   // per-device
            // +8 rows: gemm_fp16's joint_matrix_store writes full TM=8 row
            // tiles whose origin is < M, so the last tile may overrun.
            const uint64_t c_need = (uint64_t(T) + 8) * N;
            if (c_need > scr.c_cap) {
                if (scr.c) sycl::free(scr.c, q);
                scr.c = sycl::malloc_device<float>(c_need, q);
                scr.c_cap = scr.c ? c_need : 0;
            }
            if (scr.c) {
                sycl::event e = gemm_fp16(
                    q, A, static_cast<const sycl::half*>(W.p), scr.c, T, N, K,
                    deps);
                return cast_fp32_to_fp16(q, scr.c, y, uint64_t(T) * N, {e});
            }
            // scratch allocation failed → fall through to the serial loop
        }
        sycl::event last;
        for (uint32_t t = 0; t < T; ++t) {
            const sycl::half* a_t = A + uint64_t(t) * K;
            sycl::half*       y_t = y + uint64_t(t) * N;
            last = (t == 0)
                ? gemv_fp16(q, a_t, static_cast<const sycl::half*>(W.p), y_t,
                            K, N, deps)
                : gemv_fp16(q, a_t, static_cast<const sycl::half*>(W.p), y_t,
                            K, N);
        }
        return last;
    }
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
        // Per-device device scratch (multi-GPU safe), grown on demand. Host is
        // single-threaded and the queue is in-order, so reuse across back-to-back
        // projections is race-free.
        GemvPrefillScratch& scr = gemv_prefill_scratch(q);
        const uint64_t bt_need = uint64_t(K) * N;
        // +8 rows: gemm_fp16's joint_matrix_store writes full TM=8 row tiles
        // whose origin is < M, so the last tile may overrun by up to 7 rows.
        const uint64_t c_need = (uint64_t(T) + 8) * N;
        if (bt_need > scr.bt_cap) {
            if (scr.bt) sycl::free(scr.bt, q);
            scr.bt = sycl::malloc_device<sycl::half>(bt_need, q);
            scr.bt_cap = scr.bt ? bt_need : 0;
        }
        if (c_need > scr.c_cap) {
            if (scr.c) sycl::free(scr.c, q);
            scr.c = sycl::malloc_device<float>(c_need, q);
            scr.c_cap = scr.c ? c_need : 0;
        }
        if (scr.bt && scr.c) {
            sycl::event e = (W.dt == DType::kQ4_K)
                ? dequant_q4_K_to_Bt(q, W.p, scr.bt, K, N, deps)
                : dequant_q6_K_to_Bt(q, W.p, scr.bt, K, N, deps);
            // P1b (2026-06-10): oneDNN matmul kept opt-IN (IE_ONEDNN=1),
            // same as the qwen36.cpp original.
            static const bool onednn_env = std::getenv("IE_ONEDNN") != nullptr;
            if (onednn_env || prefer_onednn())
                return gemm_fp16_onednn(q, A, scr.bt, y, T, N, K, {e});
            e = gemm_fp16(q, A, scr.bt, scr.c, T, N, K, {e});
            return cast_fp32_to_fp16(q, scr.c, y, uint64_t(T) * N, {e});
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
            last = (m == 0) ? gemm_q4_K_xmx(q, a_t, W.p, y_t, mc, K, N, deps)
                            : gemm_q4_K_xmx(q, a_t, W.p, y_t, mc, K, N);
        }
        return last;
    }
    if (W.dt == DType::kQ6_K) {
        constexpr uint32_t M_TILE_XMX    = 16;
        constexpr uint32_t M_TILE_SCALAR = 32;
        static const bool xmx_disabled = std::getenv("IE_NO_XMX") != nullptr;
        const bool use_xmx = !xmx_disabled && (N % 64 == 0) && (K % 256 == 0);
        // Scalar gemm_q6_K stages an M×K fp16 A-tile in SLM — clamp M so the
        // tile fits the 128 KiB WG budget at big K (qwen3 ffn_down K=12288
        // would need 768 KiB at M=32).  XMX tiling is unaffected.
        const uint32_t m_scalar_fit = std::max<uint32_t>(
            1u, (128u * 1024u) / (uint32_t(sizeof(sycl::half)) * K));
        // gemm_q6_K's internal M_TILE cap is 8 (src/ops/gemv_q6k.cpp:531)
        constexpr uint32_t M_CAP_Q6K_SCALAR = 8u;
        const uint32_t M_TILE =
            use_xmx ? M_TILE_XMX : std::min({M_TILE_SCALAR, m_scalar_fit, M_CAP_Q6K_SCALAR});
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

}  // namespace dense
}  // namespace ie
