// src/model/qwen35_dense.cpp — P3d: Qwen3.6-27B (`qwen35` dense-hybrid) loader
// + workspace. The hybrid forward lands in Task 3B/3C.
//
// Mirrors DenseModel::load's structure (src/model/dense_transformer.cpp) but
// branches per layer on cfg.recurrent_layer(il): linear (gated-DeltaNet) layers
// load the ssm_* / fused attn_qkv tensors; full-attention layers load the joint
// Q|gate attn_q + attn_k/v/o. Q5_K (attn_k/attn_output) and Q8_0 (ssm_out) are
// dequanted to F16 [K,N] at load (landed dequant_q5_K_to_Bt / dequant_q8_0_to_Bt)
// so they ride the dense path's F16 GEMV branch — no new GEMV kernel (plan R5).
//
// IRON RULE: crown (qwen36.cpp) is never edited; we reuse the P2 dense helpers
// (dense_dispatch.hpp) and the src/ops/* leaf functions.

#include "ie/qwen35_dense.hpp"

#include "ie/dequant.hpp"
#include "ie/dspark_drafter.hpp"
#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"

#include "dense_dispatch.hpp"

#include <algorithm>
#include <functional>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

namespace ie {

// Forward decl — tiled FlashAttention-2 prefill/verify (T>1). Drop-in for the
// naive `full_attention` (same signature + correctness); reads the KV cache ONCE
// per tile instead of T× (the naive path's cost). Defined in src/ops/attention.cpp.
// Wired here for the spec-decode VERIFY (T=K) where naive attn was 20% of the
// verify forward; default-ON, opt-out IE_QWEN35_NO_FA2_PREFILL for an A/B.
sycl::event full_attention_fa2_prefill(sycl::queue& q,
                                       const sycl::half* q_in, const sycl::half* k_in,
                                       const sycl::half* v_in, sycl::half* k_cache,
                                       sycl::half* v_cache, sycl::half* y,
                                       uint32_t T, uint32_t start_pos,
                                       uint32_t n_q_heads, uint32_t n_kv_heads,
                                       uint32_t head_dim, uint32_t max_ctx,
                                       const std::vector<sycl::event>& deps = {});

namespace {

// Spec-decode VERIFY: max draft batch T routed through the batched-T int-dot
// GEMV (gemv_q{4_K,6_soa}_q8_batched) instead of the dequant→gemm prefill path.
// Must match the T_MAX in those kernels (the activation scratch is sized for it).
constexpr uint32_t kBatchedVerifyMax = 16;

// IE_Q2_BATCHED (opt-IN, default OFF → forward byte-identical): native W2A8
// batched int-dot PREFILL for SoA-Q2 — routes T≥2 through
// gemv_q2_0_soa_q8_batched (weight column read once, dp4a vs T staged Q8_1
// rows) instead of the T-independent dequant→fp16→gemm restream
// (docs/q2_0_optimization/07). Read once.
bool q2_batched_prefill_on() {
    static const bool v = std::getenv("IE_Q2_BATCHED") != nullptr;
    return v;
}

// IE_Q2_XMX_PREFILL (opt-IN, default OFF → forward byte-identical): fused-dequant
// XMX/DPAS PREFILL for SoA-Q2 — routes T≥2 through gemm_q2_0_soa_xmx (cooperative
// Q2→fp16 SLM stripe + joint_matrix fp16 DPAS, internal M-tiling) instead of the
// dequant_q2_soa→fp16 Bt→gemm_fp16 path that materializes a 53.8 GB/pass fp16
// weight in DRAM (docs/q2_0_optimization/07). Guarded on N%64==0 && K%256==0.
// Read once.
bool q2_xmx_prefill_on() {
    static const bool v = std::getenv("IE_Q2_XMX_PREFILL") != nullptr;
    return v;
}

// IE_Q1_NATIVE: route Q1_0 (Prism 1-bit) tensors through the NATIVE W1A8 SoA
// stack (repack_q1_0_to_soa + gemv_q1_0_soa_q8 / dequant_q1_0_soa_to_Bt +
// embedding_lookup_q1_0) instead of the Phase-A load-time Q1→Q2 expansion.
// DEFAULT ON since the 2026-07-18 A/B: decode/prefill FLAT vs expansion
// (33.8 vs 33.9 tg — kernel is element-rate-bound, so halved bytes don't pay)
// but ~1.7 GB less weight VRAM (qs plane K/8 vs K/4) and ~3x faster load
// (1.1 s vs 3.5 s); PPL 12.2077 vs 12.2055 = fp-reduction noise. IE_Q1_NATIVE=0
// reverts to the exact Phase-A expansion (kill switch). Read once.
bool q1_native_on() {
    static const bool v = []{
        const char* e = std::getenv("IE_Q1_NATIVE");
        return !e || std::atoi(e) != 0;
    }();
    return v;
}

// Dequant a Q5_K / Q8_0 GGUF weight [in=K, out=N] (ggml shape[0]=K contiguous)
// to a device F16 [K,N] buffer via the transposed dequant kernels, so the
// result is consumable by gemv_fp16 / gemm_fp16 exactly like a load-time
// transposed F16 weight. Frees the transient packed upload. Returns kF16.
DenseQuantPtr upload_dequant_to_fp16(DeviceAllocator& alloc,
                                     const GgufTensorInfo* t,
                                     std::vector<void*>& owned,
                                     std::string& err) {
    DenseQuantPtr out;
    if (!t) { err = "tensor not found"; return out; }
    if (t->n_dims != 2) { err = "dequant_to_fp16: expected 2-D weight"; return out; }
    const uint32_t K = static_cast<uint32_t>(t->shape[0]);
    const uint32_t N = static_cast<uint32_t>(t->shape[1]);
    if (K == 0 || N == 0) { err = "dequant_to_fp16: zero dim"; return out; }

    void* packed = alloc.malloc(t->nbytes);
    if (!packed) { err = "malloc failed (packed)"; return out; }
    alloc.queue().memcpy(packed, t->data, t->nbytes).wait();

    auto* d = static_cast<sycl::half*>(
        alloc.malloc(static_cast<uint64_t>(K) * N * sizeof(sycl::half)));
    if (!d) { alloc.free(packed); err = "malloc failed (fp16)"; return out; }

    sycl::event e;
    if (t->dtype == DType::kQ5_K) {
        e = dequant_q5_K_to_Bt(alloc.queue(), packed, d, K, N);
    } else if (t->dtype == DType::kQ8_0) {
        e = dequant_q8_0_to_Bt(alloc.queue(), packed, d, K, N);
    } else {
        alloc.free(packed); alloc.free(d);
        err = std::string("dequant_to_fp16: unsupported dtype ") +
              std::string(type_name(t->dtype)) + " for '" +
              std::string(t->name) + "'";
        return out;
    }
    e.wait();
    alloc.free(packed);          // transient — not owned
    owned.push_back(d);
    out.p = d;
    out.dt = DType::kF16;
    return out;
}

// Load any matrix weight by its ACTUAL dtype (bartowski Q4_K_M mixes dtypes
// per layer — e.g. ssm_out is Q8_0 in some layers, Q4_K in others). Q4_K/Q6_K/
// F16 ride the dense int-dot/F16 GEMV path; Q5_K/Q8_0 (no int-dot GEMV) are
// dequanted to F16 [K,N]. The resulting DenseQuantPtr.dt drives the forward's
// gemv_q dispatch, so correctness is preserved whatever the per-layer dtype.
DenseQuantPtr upload_weight_auto(DeviceAllocator& alloc, const GgufTensorInfo* t,
                                 std::vector<void*>& owned, std::string& err) {
    if (!t) { err = "tensor not found"; return {}; }
    if (t->dtype == DType::kQ5_K || t->dtype == DType::kQ8_0)
        return upload_dequant_to_fp16(alloc, t, owned, err);
    return dense::upload_quant_dense(alloc, t, owned, err);  // Q4_K/Q6_K/F16 or hard-fail
}

// Host-side canonical Q6_K row dequant — mirrors ggml's dequantize_row_q6_K,
// producing weights in NATURAL element order. `blocks` = first of K/256 q6_K
// blocks for one row; `out` receives K floats (K % 256 == 0). Load-time only
// (small ssm proj tensors in *-Q6_K GGUFs), not a hot path.
static void dequant_q6_K_row(const block_q6_K* blocks, float* out, uint64_t K) {
    const uint64_t nb = K / kQK_K;
    for (uint64_t i = 0; i < nb; ++i) {
        const float    d  = fp16_to_fp32(blocks[i].d);
        const uint8_t* ql = blocks[i].ql;
        const uint8_t* qh = blocks[i].qh;
        const int8_t*  sc = blocks[i].scales;
        float*         y  = out + i * kQK_K;
        for (int half = 0; half < kQK_K; half += 128) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int q1 = int((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int q2 = int((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int q3 = int((ql[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int q4 = int((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * float(sc[is + 0]) * float(q1);
                y[l + 32] = d * float(sc[is + 2]) * float(q2);
                y[l + 64] = d * float(sc[is + 4]) * float(q3);
                y[l + 96] = d * float(sc[is + 6]) * float(q4);
            }
            y += 128; ql += 64; qh += 32; sc += 8;
        }
    }
}

// Natural-order Q4_K row dequant (ggml dequantize_row_q4_K) for the Q4_K-quantized
// ssm_alpha/ssm_beta projections in *-Q4_K_M GGUFs (this dense loader previously only
// took F32/Q8_0/Q6_K). Mirrors the crown-split fix (000adb6). Load-time only; K % 256 == 0.
static void dequant_q4_K_row(const block_q4_K* blocks, float* out, uint64_t K) {
    const uint64_t nb = K / kQK_K;
    for (uint64_t i = 0; i < nb; ++i) {
        const float d    = fp16_to_fp32(blocks[i].d);
        const float dmin = fp16_to_fp32(blocks[i].dmin);
        const uint8_t* sc = blocks[i].scales;   // 12B packed 6-bit scales/mins
        const uint8_t* q  = blocks[i].qs;        // 128B 4-bit quants
        float* y = out + i * kQK_K;
        for (int is = 0, j = 0; j < int(kQK_K); j += 64, is += 2) {
            // get_scale_min_k4 for sub-blocks `is` and `is+1`
            uint8_t s1, m1, s2, m2;
            if (is < 4) { s1 = sc[is] & 63; m1 = sc[is + 4] & 63; }
            else { s1 = (sc[is + 4] & 0xF) | ((sc[is - 4] >> 6) << 4);
                   m1 = (sc[is + 4] >> 4)  | ((sc[is    ] >> 6) << 4); }
            const int k2 = is + 1;
            if (k2 < 4) { s2 = sc[k2] & 63; m2 = sc[k2 + 4] & 63; }
            else { s2 = (sc[k2 + 4] & 0xF) | ((sc[k2 - 4] >> 6) << 4);
                   m2 = (sc[k2 + 4] >> 4)  | ((sc[k2    ] >> 6) << 4); }
            const float d1 = d * float(s1), b1 = dmin * float(m1);
            const float d2 = d * float(s2), b2 = dmin * float(m2);
            for (int l = 0; l < 32; ++l) y[l]      = d1 * float(q[l] & 0xF) - b1;
            for (int l = 0; l < 32; ++l) y[l + 32] = d2 * float(q[l] >> 4)  - b2;
            y += 64; q += 32;
        }
    }
}

// The ssm_alpha / ssm_beta projections are F32 [K=5120, N=48] in the GGUF
// (ggml shape[0]=K contiguous → [N,K] row-major). Transpose to [K,Npad] (cols
// ≥N zero-padded) and cast to fp16 so they ride the BATCHED gemm_fp16 path
// (gemm needs N a multiple of the SG tile to avoid OOB weight reads — N=48 is
// not, so pad to 64). Without the pad the F16 GEMV falls to a per-token serial
// loop (512 launches × proj × layer = 49k launches, 38% of prefill).
DenseQuantPtr upload_f32_proj_fp16(DeviceAllocator& alloc, const GgufTensorInfo* t,
                                   std::vector<void*>& owned, std::string& err,
                                   uint32_t Npad) {
    DenseQuantPtr out;
    if (!t) { err = "tensor not found"; return out; }
    if (t->n_dims != 2 || (t->dtype != DType::kF32 && t->dtype != DType::kQ8_0 &&
                           t->dtype != DType::kQ6_K && t->dtype != DType::kQ4_K &&
                           t->dtype != DType::kQ2_0 && t->dtype != DType::kQ1_0)) {
        err = "ssm proj: expected F32, Q8_0, Q6_K, Q4_K, Q2_0, or Q1_0 2-D"; return out;
    }
    const uint64_t K = t->shape[0];   // 5120 (in)
    const uint64_t N = t->shape[1];   // 48   (out)
    if (Npad < N) Npad = uint32_t(N);
    std::vector<sycl::half> staging(K * Npad, sycl::half(0.0f));   // zero-padded cols ≥N
    if (t->dtype == DType::kF32) {
        const float* src = reinterpret_cast<const float*>(t->data);
        for (uint64_t n = 0; n < N; ++n)
            for (uint64_t k = 0; k < K; ++k)
                staging[k * Npad + n] = sycl::half(src[n * K + k]);   // [N,K] → [K,Npad]
    } else if (t->dtype == DType::kQ8_0) {  // dequant on the fly (some quants store ssm proj Q8_0)
        const uint64_t bpr = K / 32;   // Q8_0 blocks per row (K % 32 == 0)
        const auto* blocks = reinterpret_cast<const block_q8_0*>(t->data);
        for (uint64_t n = 0; n < N; ++n)
            for (uint64_t k = 0; k < K; ++k) {
                const block_q8_0& b = blocks[n * bpr + (k >> 5)];
                staging[k * Npad + n] = sycl::half(fp16_to_fp32(b.d) * float(b.qs[k & 31]));
            }
    } else if (t->dtype == DType::kQ6_K) {  // Q6_K: K-quant ssm proj (e.g. *-Q6_K GGUFs). Dequant per row, then transpose.
        if (K % kQK_K != 0) { err = "ssm proj Q6_K: K not a multiple of 256"; return out; }
        const uint64_t bpr = K / kQK_K;   // q6_K super-blocks per row
        const auto* blocks = reinterpret_cast<const block_q6_K*>(t->data);
        std::vector<float> row(K);
        for (uint64_t n = 0; n < N; ++n) {
            dequant_q6_K_row(blocks + n * bpr, row.data(), K);
            for (uint64_t k = 0; k < K; ++k)
                staging[k * Npad + n] = sycl::half(row[k]);   // [N,K] → [K,Npad]
        }
    } else if (t->dtype == DType::kQ4_K) {  // Q4_K: K-quant ssm proj (e.g. *-Q4_K_M GGUFs). Dequant per row, then transpose.
        if (K % kQK_K != 0) { err = "ssm proj Q4_K: K not a multiple of 256"; return out; }
        const uint64_t bpr = K / kQK_K;   // q4_K super-blocks per row
        const auto* blocks = reinterpret_cast<const block_q4_K*>(t->data);
        std::vector<float> row(K);
        for (uint64_t n = 0; n < N; ++n) {
            dequant_q4_K_row(blocks + n * bpr, row.data(), K);
            for (uint64_t k = 0; k < K; ++k)
                staging[k * Npad + n] = sycl::half(row[k]);   // [N,K] → [K,Npad]
        }
    } else if (t->dtype == DType::kQ2_0) {  // Prism ternary 2-bit ssm proj. value = (code-1)*d, inline.
        if (K % 128 != 0) { err = "ssm proj Q2_0: K not a multiple of 128"; return out; }
        const uint64_t bpr = K / 128;   // q2_0 blocks per row
        const auto* blocks = reinterpret_cast<const block_q2_0*>(t->data);
        for (uint64_t n = 0; n < N; ++n)
            for (uint64_t k = 0; k < K; ++k) {
                const block_q2_0& b = blocks[n * bpr + (k >> 7)];   // k / 128
                const uint32_t j = uint32_t(k & 127);
                const uint8_t code = (b.qs[j >> 2] >> ((j & 3) * 2)) & 0x03;
                staging[k * Npad + n] = sycl::half((int(code) - 1) * fp16_to_fp32(b.d));
            }
    } else {  // Q1_0: Prism 1-bit ssm proj. value = (2*bit-1)*d, inline.
        if (K % 128 != 0) { err = "ssm proj Q1_0: K not a multiple of 128"; return out; }
        const uint64_t bpr = K / 128;   // q1_0 blocks per row
        const auto* blocks = reinterpret_cast<const block_q1_0*>(t->data);
        for (uint64_t n = 0; n < N; ++n)
            for (uint64_t k = 0; k < K; ++k) {
                const block_q1_0& b = blocks[n * bpr + (k >> 7)];   // k / 128
                const uint32_t j = uint32_t(k & 127);
                const int bit = (b.qs[j >> 3] >> (j & 7)) & 1;
                staging[k * Npad + n] = sycl::half(float(2 * bit - 1) * fp16_to_fp32(b.d));
            }
    }
    void* d = alloc.malloc(K * Npad * sizeof(sycl::half));
    if (!d) { err = "malloc failed (ssm proj)"; return out; }
    alloc.queue().memcpy(d, staging.data(), K * Npad * sizeof(sycl::half)).wait();
    owned.push_back(d);
    out.p = d; out.dt = DType::kF16;
    return out;
}

// Repack a canonical Q6_K GGUF weight [K,N] into device SoA-Q6 streams
// (q6_lo/q6_hi/q6_sc/q6_d). Host de-interleave via repack_q6_K_to_soa, then
// upload each stream. ~6.5 bpw — no inflation vs the 210 B AoS block. Returns a
// Q6SoaW with all four device pointers (pushed to `owned`) or sets `err`.
static Q6SoaW upload_q6_soa(DeviceAllocator& alloc, const GgufTensorInfo* t,
                            std::vector<void*>& owned, std::string& err) {
    Q6SoaW w;
    if (!t) { err = "tensor not found"; return w; }
    if (t->dtype != DType::kQ6_K) { err = "upload_q6_soa: not Q6_K"; return w; }
    if (t->n_dims != 2) { err = "upload_q6_soa: expected 2-D"; return w; }
    const uint32_t K = uint32_t(t->shape[0]);   // contiguous dim
    const uint32_t N = uint32_t(t->shape[1]);
    if (K == 0 || N == 0 || K % kQK_K != 0) { err = "upload_q6_soa: bad K/N"; return w; }
    const uint64_t lo_n = uint64_t(N) * (K / 2);
    const uint64_t hi_n = uint64_t(N) * (K / 4);
    const uint64_t sc_n = uint64_t(N) * (K / 16);
    const uint64_t d_n  = uint64_t(N) * (K / 256);
    // Host staging (zero-initialized — the bit-plane writes are read-modify-write).
    std::vector<uint8_t>  h_lo(lo_n, 0), h_hi(hi_n, 0);
    std::vector<int8_t>   h_sc(sc_n, 0);
    std::vector<uint16_t> h_d(d_n, 0);
    repack_q6_K_to_soa(t->data, K, N, h_lo.data(), h_hi.data(), h_sc.data(), h_d.data());
    auto* d_lo = static_cast<uint8_t*>(alloc.malloc(lo_n));
    auto* d_hi = static_cast<uint8_t*>(alloc.malloc(hi_n));
    auto* d_sc = static_cast<int8_t*>(alloc.malloc(sc_n));
    auto* d_d  = static_cast<uint16_t*>(alloc.malloc(d_n * sizeof(uint16_t)));
    if (!d_lo || !d_hi || !d_sc || !d_d) { err = "upload_q6_soa: malloc failed"; return w; }
    owned.push_back(d_lo); owned.push_back(d_hi);
    owned.push_back(d_sc); owned.push_back(d_d);
    alloc.queue().memcpy(d_lo, h_lo.data(), lo_n).wait();
    alloc.queue().memcpy(d_hi, h_hi.data(), hi_n).wait();
    alloc.queue().memcpy(d_sc, h_sc.data(), sc_n).wait();
    alloc.queue().memcpy(d_d,  h_d.data(),  d_n * sizeof(uint16_t)).wait();
    w.lo = d_lo; w.hi = d_hi; w.sc = d_sc; w.d = d_d; w.K = K; w.N = N;
    return w;
}

// Q2 weight-buffer allocator (TLB experiment, opt-in IE_Q2_HUGEPAGE). The Q2_0
// decode GEMV streams ~6.8 GB of distinct weight bytes/token; profiling suggests
// ~2 ms/token of page-walk cost over many 4 KB pages. When IE_Q2_HUGEPAGE is set,
// allocate 2 MB-aligned device USM AND round the size up to a 2 MB multiple —
// both are needed to encourage the driver to back the range with 2 MB pages
// (512x fewer pages). Extra tail bytes are never read. sycl::free handles both
// malloc_device and aligned_alloc_device, so the free path is unchanged.
static void* q2_alloc_device(DeviceAllocator& alloc, size_t nbytes) {
    if (std::getenv("IE_Q2_HUGEPAGE")) {
        constexpr size_t kHuge = 2ull * 1024 * 1024;   // 2 MB
        const size_t rounded = (nbytes + kHuge - 1) / kHuge * kHuge;
        return sycl::aligned_alloc_device(kHuge, rounded, alloc.queue());
    }
    return alloc.malloc(nbytes);   // default: byte-identical to before
}

// Expand Q1_0 blocks (1-bit signs) into equivalent block_q2_0 bytes: code
// c = 2*bit ∈ {0,2} with the SAME d gives (c-1)*d = ±d — mathematically EXACT.
// Phase A of Q1_0 support: the 1-bit Bonsai GGUF rides the entire validated
// Q2_0 SoA stack (kernels, prefill dequant, embedding, tests) unmodified, at
// Q2-sized VRAM (34 vs 18 B/block). Native 1-bit kernels are a follow-on.
// 2 output bytes per input byte: bit i of a nibble → crumb '10' at bits
// (2i+1,2i), i.e. value 2.
static std::vector<uint8_t> expand_q1_to_q2_blocks(const void* src,
                                                   uint64_t n_blocks) {
    static const uint8_t kSpread[16] = {
        0x00, 0x02, 0x08, 0x0A, 0x20, 0x22, 0x28, 0x2A,
        0x80, 0x82, 0x88, 0x8A, 0xA0, 0xA2, 0xA8, 0xAA};
    const auto* in = static_cast<const block_q1_0*>(src);
    std::vector<uint8_t> out(n_blocks * sizeof(block_q2_0));
    auto* ob = reinterpret_cast<block_q2_0*>(out.data());
    for (uint64_t b = 0; b < n_blocks; ++b) {
        ob[b].d = in[b].d;
        for (int j = 0; j < 16; ++j) {
            const uint8_t v = in[b].qs[j];
            ob[b].qs[2 * j]     = kSpread[v & 0x0F];
            ob[b].qs[2 * j + 1] = kSpread[v >> 4];
        }
    }
    return out;
}

// Repack a canonical Q2_0 (Prism ternary 2-bit) GGUF weight [K,N] into device
// SoA-Q2 streams (q2_qs 2-bit codes + per-128 fp16 scale). Host de-interleave via
// repack_q2_0_to_soa, then upload. ~2.125 bpw — no inflation vs the 34 B AoS
// block. Returns a Q2SoaW (device pointers pushed to `owned`) or sets `err`.
static Q2SoaW upload_q2_soa(DeviceAllocator& alloc, const GgufTensorInfo* t,
                            std::vector<void*>& owned, std::string& err) {
    Q2SoaW w;
    if (!t) { err = "tensor not found"; return w; }
    if (t->dtype != DType::kQ2_0 && t->dtype != DType::kQ1_0) {
        err = "upload_q2_soa: not Q2_0/Q1_0"; return w;
    }
    if (t->n_dims != 2) { err = "upload_q2_soa: expected 2-D"; return w; }
    const uint32_t K = uint32_t(t->shape[0]);   // contiguous dim
    const uint32_t N = uint32_t(t->shape[1]);
    if (K == 0 || N == 0 || K % 128 != 0) { err = "upload_q2_soa: K % 128 != 0"; return w; }

    // NATIVE Q1_0 (Phase B, IE_Q1_NATIVE): repack the 1-bit blocks straight into
    // the SoA-Q1 qs plane (K/8 bytes/col vs Q2's K/4) — no expansion. The d plane
    // is identical to Q2's. dtype=kQ1_0 routes the dispatch to the W1A8 kernels.
    if (t->dtype == DType::kQ1_0 && q1_native_on()) {
        const uint64_t qs_n = uint64_t(N) * (K / 8);    // bytes (half of Q2)
        const uint64_t d_n  = uint64_t(N) * (K / 128);  // uint16
        std::vector<uint8_t>  h_qs(qs_n, 0);
        std::vector<uint16_t> h_d(d_n, 0);
        repack_q1_0_to_soa(t->data, K, N, h_qs.data(), h_d.data());
        auto* d_qs = static_cast<uint8_t*>(q2_alloc_device(alloc, qs_n));
        auto* d_d  = static_cast<uint16_t*>(q2_alloc_device(alloc, d_n * sizeof(uint16_t)));
        if (!d_qs || !d_d) { err = "upload_q2_soa: malloc failed (Q1 native)"; return w; }
        owned.push_back(d_qs); owned.push_back(d_d);
        alloc.queue().memcpy(d_qs, h_qs.data(), qs_n).wait();
        alloc.queue().memcpy(d_d,  h_d.data(),  d_n * sizeof(uint16_t)).wait();
        w.qs = d_qs; w.d = d_d; w.K = K; w.N = N; w.dtype = DType::kQ1_0;
        return w;
    }

    const uint64_t qs_n = uint64_t(N) * (K / 4);        // bytes
    const uint64_t d_n  = uint64_t(N) * (K / 128);      // uint16
    std::vector<uint8_t>  h_qs(qs_n, 0);
    std::vector<uint16_t> h_d(d_n, 0);
    const void* blocks = t->data;
    std::vector<uint8_t> expanded;   // Q1_0 → exact Q2_0 codes {0,2} (Phase A)
    if (t->dtype == DType::kQ1_0) {
        expanded = expand_q1_to_q2_blocks(t->data, uint64_t(N) * (K / 128));
        blocks = expanded.data();
    }
    repack_q2_0_to_soa(blocks, K, N, h_qs.data(), h_d.data());
    auto* d_qs = static_cast<uint8_t*>(q2_alloc_device(alloc, qs_n));
    auto* d_d  = static_cast<uint16_t*>(q2_alloc_device(alloc, d_n * sizeof(uint16_t)));
    if (!d_qs || !d_d) { err = "upload_q2_soa: malloc failed"; return w; }
    owned.push_back(d_qs); owned.push_back(d_d);
    alloc.queue().memcpy(d_qs, h_qs.data(), qs_n).wait();
    alloc.queue().memcpy(d_d,  h_d.data(),  d_n * sizeof(uint16_t)).wait();
    w.qs = d_qs; w.d = d_d; w.K = K; w.N = N;   // dtype stays kQ2_0 (Phase A)
    return w;
}

// Repack a canonical Q4_K GGUF weight [K,N] into device SoA-Q4 streams
// (q4_q/q4_sc/q4_mn/q4_d/q4_dmin). Host de-interleave via repack_q4_K_to_soa,
// then upload each stream. ~4.625 bpw — no inflation vs the 144 B AoS block.
// Mirrors upload_q6_soa exactly for the Q4_K fast-decode path. Returns a Q4SoaW
// with all five device pointers (pushed to `owned`) or sets `err`.
static Q4SoaW upload_q4_soa(DeviceAllocator& alloc, const GgufTensorInfo* t,
                            std::vector<void*>& owned, std::string& err) {
    Q4SoaW w;
    if (!t) { err = "tensor not found"; return w; }
    if (t->dtype != DType::kQ4_K) { err = "upload_q4_soa: not Q4_K"; return w; }
    if (t->n_dims != 2) { err = "upload_q4_soa: expected 2-D"; return w; }
    const uint32_t K = uint32_t(t->shape[0]);   // contiguous dim
    const uint32_t N = uint32_t(t->shape[1]);
    if (K == 0 || N == 0 || K % kQK_K != 0) { err = "upload_q4_soa: bad K/N"; return w; }
    const uint64_t q_n  = uint64_t(N) * (K / 2);    // 4-bit nibbles, 2 elems/byte
    const uint64_t sc_n = uint64_t(N) * (K / 32);   // per-32 int8 s_raw
    const uint64_t mn_n = uint64_t(N) * (K / 32);   // per-32 int8 m_raw
    const uint64_t d_n  = uint64_t(N) * (K / 256);  // per-256 fp16 d
    const uint64_t dm_n = uint64_t(N) * (K / 256);  // per-256 fp16 dmin
    // Host staging (q4_q's two-nibble bytes are each written once; no RMW, but
    // zero-init keeps any K-edge bytes well-defined — mirrors upload_q6_soa).
    std::vector<uint8_t>  h_q(q_n, 0);
    std::vector<int8_t>   h_sc(sc_n, 0), h_mn(mn_n, 0);
    std::vector<uint16_t> h_d(d_n, 0), h_dm(dm_n, 0);
    repack_q4_K_to_soa(t->data, K, N, h_q.data(), h_sc.data(), h_mn.data(),
                       h_d.data(), h_dm.data());
    auto* d_q  = static_cast<uint8_t*>(alloc.malloc(q_n));
    auto* d_sc = static_cast<int8_t*>(alloc.malloc(sc_n));
    auto* d_mn = static_cast<int8_t*>(alloc.malloc(mn_n));
    auto* d_d  = static_cast<uint16_t*>(alloc.malloc(d_n  * sizeof(uint16_t)));
    auto* d_dm = static_cast<uint16_t*>(alloc.malloc(dm_n * sizeof(uint16_t)));
    if (!d_q || !d_sc || !d_mn || !d_d || !d_dm) { err = "upload_q4_soa: malloc failed"; return w; }
    owned.push_back(d_q);  owned.push_back(d_sc); owned.push_back(d_mn);
    owned.push_back(d_d);  owned.push_back(d_dm);
    alloc.queue().memcpy(d_q,  h_q.data(),  q_n).wait();
    alloc.queue().memcpy(d_sc, h_sc.data(), sc_n).wait();
    alloc.queue().memcpy(d_mn, h_mn.data(), mn_n).wait();
    alloc.queue().memcpy(d_d,  h_d.data(),  d_n  * sizeof(uint16_t)).wait();
    alloc.queue().memcpy(d_dm, h_dm.data(), dm_n * sizeof(uint16_t)).wait();
    w.q = d_q; w.sc = d_sc; w.mn = d_mn; w.d = d_d; w.dmin = d_dm; w.K = K; w.N = N;
    return w;
}

// Upload a Q4_K weight in llama's REORDERED layout (3 global regions
// nibbles|scales|dm) for gemv_q4_K_reorder_q8. Same bytes as AoS (pure relocate).
static void* upload_q4_reorder(DeviceAllocator& alloc, const GgufTensorInfo* t,
                               std::vector<void*>& owned, std::string& err) {
    if (!t || t->dtype != DType::kQ4_K || t->n_dims != 2) {
        err = "upload_q4_reorder: not Q4_K 2-D"; return nullptr;
    }
    const uint32_t K = uint32_t(t->shape[0]);
    const uint32_t N = uint32_t(t->shape[1]);
    if (K == 0 || N == 0 || K % kQK_K != 0) { err = "upload_q4_reorder: bad K/N"; return nullptr; }
    const uint64_t bytes = uint64_t(N) * (K / 256) * 144;   // nblocks * 144 (= AoS size)
    std::vector<uint8_t> h(bytes);
    repack_q4_K_to_reorder(t->data, K, N, h.data());
    void* d = alloc.malloc(bytes);
    if (!d) { err = "upload_q4_reorder: malloc failed"; return nullptr; }
    owned.push_back(d);
    alloc.queue().memcpy(d, h.data(), bytes).wait();
    return d;
}

// Repack a NATIVE Q8_0 weight [K(contig), N] → SoA: column-contiguous int8 qs
// [N*K] + per-32-block raw-fp16 d [N*(K/32)]. Bit-exact (no requant). Consumed by
// gemv_q8_0_soa_q8 (decode, ~80% BW) and dequant_q8_0_soa_to_Bt (prefill). Mirrors
// qwen35_split's build_split Q8_0 branch, for the single-GPU decode path.
static Q8SoaW upload_q8_soa(DeviceAllocator& alloc, const GgufTensorInfo* t,
                            std::vector<void*>& owned, std::string& err) {
    Q8SoaW w;
    if (!t) { err = "tensor not found"; return w; }
    if (t->dtype != DType::kQ8_0) { err = "upload_q8_soa: not Q8_0"; return w; }
    if (t->n_dims != 2) { err = "upload_q8_soa: expected 2-D"; return w; }
    const uint32_t K = uint32_t(t->shape[0]), N = uint32_t(t->shape[1]);
    if (K == 0 || N == 0 || K % 32 != 0) { err = "upload_q8_soa: bad K/N"; return w; }
    const uint32_t bpc = K / 32;
    const auto* blocks = reinterpret_cast<const block_q8_0*>(t->data);
    std::vector<int8_t>   qs(uint64_t(N) * K);
    std::vector<uint16_t> dd(uint64_t(N) * bpc);
    for (uint64_t n = 0; n < N; ++n)
        for (uint32_t b = 0; b < bpc; ++b) {
            const block_q8_0& blk = blocks[n * bpc + b];
            dd[n * bpc + b] = *reinterpret_cast<const uint16_t*>(&blk.d);
            for (int i = 0; i < 32; ++i)
                qs[n * uint64_t(K) + uint64_t(b) * 32 + i] = blk.qs[i];
        }
    auto* dqs = static_cast<int8_t*>(alloc.malloc(qs.size()));
    auto* ddd = static_cast<uint16_t*>(alloc.malloc(dd.size() * sizeof(uint16_t)));
    if (!dqs || !ddd) { err = "upload_q8_soa: malloc failed"; return w; }
    owned.push_back(dqs); owned.push_back(ddd);
    alloc.queue().memcpy(dqs, qs.data(), qs.size()).wait();
    alloc.queue().memcpy(ddd, dd.data(), dd.size() * sizeof(uint16_t)).wait();
    w.qs = dqs; w.d = ddd; w.K = K; w.N = N;
    return w;
}

// Requantize a Q5_K weight [K(contig),N] → SoA-Q8_0 (int8 qs col-contiguous +
// per-32 fp16 d). Q5_K reads as 16-bit on the F16-expand path (~3× native bytes);
// Q8_0 is 8.5 bpw at the ~80%-BW int-dot lane = ~half the decode traffic. Q8 (8-bit
// per-32) represents the dequantized Q5_K values near-losslessly (Q8 step < Q5
// step). Dequant on device (dequant_q5_K_to_Bt → fp16 Bt[k*N+n]) then host-quantize
// per column. 2026-06-22 GEMV grind #2.
static Q8SoaW upload_requant_q5k_q8_soa(DeviceAllocator& alloc, const GgufTensorInfo* t,
                                        std::vector<void*>& owned, std::string& err) {
    Q8SoaW w;
    if (!t) { err = "tensor not found"; return w; }
    if (t->dtype != DType::kQ5_K) { err = "requant_q5k: not Q5_K"; return w; }
    if (t->n_dims != 2) { err = "requant_q5k: expected 2-D"; return w; }
    const uint32_t K = uint32_t(t->shape[0]), N = uint32_t(t->shape[1]);
    if (K == 0 || N == 0 || K % 32 != 0) { err = "requant_q5k: bad K/N"; return w; }
    void* packed = alloc.malloc(t->nbytes);
    if (!packed) { err = "requant_q5k: malloc packed"; return w; }
    alloc.queue().memcpy(packed, t->data, t->nbytes).wait();
    auto* d_bt = static_cast<sycl::half*>(alloc.malloc(uint64_t(K) * N * sizeof(sycl::half)));
    if (!d_bt) { alloc.free(packed); err = "requant_q5k: malloc bt"; return w; }
    dequant_q5_K_to_Bt(alloc.queue(), packed, d_bt, K, N).wait();   // Bt[k*N+n] = w[k,n]
    std::vector<sycl::half> h_bt(uint64_t(K) * N);
    alloc.queue().memcpy(h_bt.data(), d_bt, uint64_t(K) * N * sizeof(sycl::half)).wait();
    alloc.free(packed); alloc.free(d_bt);
    const uint32_t bpc = K / 32;
    std::vector<int8_t>   qs(uint64_t(N) * K);
    std::vector<uint16_t> dd(uint64_t(N) * bpc);
    for (uint64_t n = 0; n < N; ++n)
        for (uint32_t b = 0; b < bpc; ++b) {
            float amax = 0.f;
            for (int i = 0; i < 32; ++i) {
                const float v = float(h_bt[uint64_t(b * 32 + i) * N + n]);
                amax = std::max(amax, std::fabs(v));
            }
            const float d  = amax / 127.f;
            const float id = (d > 0.f) ? 1.f / d : 0.f;
            const sycl::half dh = sycl::half(d);
            dd[n * bpc + b] = *reinterpret_cast<const uint16_t*>(&dh);
            for (int i = 0; i < 32; ++i) {
                const float v = float(h_bt[uint64_t(b * 32 + i) * N + n]);
                int q = int(std::lround(v * id));
                q = q < -127 ? -127 : (q > 127 ? 127 : q);
                qs[n * uint64_t(K) + uint64_t(b) * 32 + i] = int8_t(q);
            }
        }
    auto* dqs = static_cast<int8_t*>(alloc.malloc(qs.size()));
    auto* ddd = static_cast<uint16_t*>(alloc.malloc(dd.size() * sizeof(uint16_t)));
    if (!dqs || !ddd) { err = "requant_q5k: malloc soa"; return w; }
    owned.push_back(dqs); owned.push_back(ddd);
    alloc.queue().memcpy(dqs, qs.data(), qs.size()).wait();
    alloc.queue().memcpy(ddd, dd.data(), dd.size() * sizeof(uint16_t)).wait();
    w.qs = dqs; w.d = ddd; w.K = K; w.N = N;
    return w;
}

// Dequant a SoA-Q8_0 weight → fp16 Bt[K,N] (row-major, B for gemm_fp16). Prefill
// path (T≥2) so the packed Q8 weight rides the existing batched gemm. Copy of the
// split's helper (copy-not-hoist).
inline sycl::event dequant_q8_0_soa_to_Bt(sycl::queue& q, const int8_t* qs,
                                          const uint16_t* d, sycl::half* Bt,
                                          uint32_t K, uint32_t N) {
    const uint32_t bpc = K / 32;
    return q.parallel_for(sycl::range<2>(N, K), [=](sycl::id<2> id) {
        const uint32_t n = uint32_t(id[0]), k = uint32_t(id[1]);
        const float dv = float(sycl::bit_cast<sycl::half>(d[uint64_t(n) * bpc + (k >> 5)]));
        Bt[uint64_t(k) * N + n] = sycl::half(float(qs[uint64_t(n) * uint64_t(K) + k]) * dv);
    });
}

// Strided extract: dst[t,h] = src[t, h] for h < nh, src row stride = src_stride.
// Compacts the N-padded alpha/beta gemm output [T, src_stride] → [T, nh].
inline sycl::event extract_cols(sycl::queue& q, const sycl::half* src,
                                sycl::half* dst, uint32_t T, uint32_t nh,
                                uint32_t src_stride) {
    return q.parallel_for(sycl::range<1>(uint64_t(T) * nh), [=](sycl::id<1> i) {
        const uint32_t t = uint32_t(i) / nh, h = uint32_t(i) % nh;
        dst[uint64_t(t) * nh + h] = src[uint64_t(t) * src_stride + h];
    });
}

}  // namespace

Qwen35DenseModel::~Qwen35DenseModel() {
    if (alloc_) {
        if (q6soa_bt_) sycl::free(q6soa_bt_, alloc_->queue());
        if (q6soa_c_)  sycl::free(q6soa_c_,  alloc_->queue());
        for (void* p : owned_) alloc_->free(p);
    }
}

std::string Qwen35DenseModel::load(DeviceAllocator& alloc, const GgufReader& g,
                                   const Qwen35Config& cfg) {
    alloc_ = &alloc;
    cfg_ = cfg;

    // qwen35 prefill uses oneDNN matmul by default — ~1.65x faster at its shapes
    // (no bit-exact gate; PPL verified unchanged at 5.34). Crown/dense are
    // unaffected (they never set this; their bit-exact gemm_fp16 path stands).
    // Kill switch IE_QWEN35_NO_ONEDNN=1 forces the bit-exact gemm_fp16 path —
    // used for per-layer cosine parity vs the llama.cpp oracle (oneDNN is not
    // bit-identical to llama's GEMM, so it floors the achievable rel_fro).
    dense::prefer_onednn() = (std::getenv("IE_QWEN35_NO_ONEDNN") == nullptr);

    // Fast Q6_K decode path (SoA-Q6 repack + gemv_q6_soa_q8 int-dot). Default-ON
    // for Q6_K dense weights; opt out with IE_QWEN35_NO_Q6_SOA=1 (falls back to
    // the AoS Q6_K GEMV — the cosine/PPL oracle). The AoS path also remains the
    // path for the IE_QWEN35_NO_ONEDNN bit-exact branch is unaffected.
    q6_soa_on_ = (std::getenv("IE_QWEN35_NO_Q6_SOA") == nullptr);
    // Fast Q4_K decode path (SoA-Q4 repack + gemv_q4_soa_q8 int-dot W4A8). Opt-IN
    // ONLY (IE_QWEN35_Q4_SOA set → on), default OFF — so the default forward is
    // byte-identical to the AoS Q4_K GEMV and the crown PPL gate (6.4527) is
    // untouched until the maintainer GPU-validates the int-dot numerics.
    q4_soa_on_ = (std::getenv("IE_QWEN35_Q4_SOA") != nullptr);
    batched_verify_on_ = (std::getenv("IE_QWEN35_NO_BATCHED_VERIFY") == nullptr);
    fa2_prefill_on_    = (std::getenv("IE_QWEN35_NO_FA2_PREFILL") == nullptr);
    // Quant-hoist (decode T==1): compute each norm's Q8_1 stream once and reuse it
    // across all that norm's SoA-GEMV consumers, eliminating the redundant per-GEMV
    // quantize_q8_1 launches. Opt-IN ONLY (default OFF → default forward byte-
    // identical: dispatchers quantize internally, the plain norm runs). The hoisted
    // stream is bit-identical — fused norm+quant rounds the SAME fp16 outputs.
    quant_hoist_on_ = (std::getenv("IE_QWEN35_QUANT_HOIST") != nullptr);

    const DenseConfig& d = cfg.dense;
    if (d.n_layers == 0 || d.hidden == 0 || d.n_q_heads == 0 ||
        d.n_kv_heads == 0 || d.ffn == 0)            return "qwen35 config: zero dimension";
    if (d.vocab == 0)                                return "qwen35 config: vocab == 0";
    if (d.head_dim < 16 || (d.head_dim & (d.head_dim - 1)) != 0)
        return "qwen35 config: head_dim must be power-of-two >= 16, got " +
               std::to_string(d.head_dim);
    if (d.hidden % 256 != 0)                         return "qwen35 config: hidden % 256 != 0";
    if (d.ffn % 256 != 0)                            return "qwen35 config: ffn % 256 != 0";
    if (cfg.ssm_inner == 0 || cfg.ssm_n_v_heads == 0 || cfg.ssm_n_k_heads == 0 ||
        cfg.ssm_state == 0 || cfg.ssm_conv_kernel == 0)
        return "qwen35 config: missing ssm dims";

    // R1: conv_channels computed DIRECTLY — never SI*2 (that crown identity is
    // false at the 27B's 48v/16k geometry). Assert the split closes.
    const uint32_t d_inner    = cfg.ssm_inner;                              // 6144
    const uint32_t conv_ch    = d_inner + 2u * cfg.ssm_n_k_heads * cfg.ssm_state;  // 10240
    if (conv_ch != static_cast<uint32_t>(d_inner + 2u * cfg.ssm_n_k_heads * cfg.ssm_state))
        return "qwen35 config: conv_channels overflow";

    char buf[64];
    auto T = [&](const char* name) { return g.find_tensor(name); };
    auto Tlayer = [&](uint32_t L, const char* name) {
        std::snprintf(buf, sizeof(buf), "blk.%u.%s", L, name);
        return g.find_tensor(buf);
    };
    std::string err;

    // --- top-level ---
    {
        const auto* ti = T("token_embd.weight");
        if (!ti) return "token_embd: not found";
        if (ti->dtype != DType::kQ4_K && ti->dtype != DType::kQ6_K &&
            ti->dtype != DType::kQ8_0 && ti->dtype != DType::kQ2_0 &&
            ti->dtype != DType::kQ1_0)
            return std::string("token_embd: unsupported dtype ") +
                   std::string(type_name(ti->dtype)) + " (need Q4_K/Q6_K/Q8_0/Q2_0/Q1_0)";
        if (ti->dtype == DType::kQ1_0 && q1_native_on()) {
            // Native Q1_0 (Phase B): upload the raw 1-bit block_q1_0 table and use
            // embedding_lookup_q1_0 — no Q1→Q2 expansion (half the qs bytes).
            if (ti->n_dims != 2 || ti->shape[0] % 128 != 0)
                return "token_embd: bad Q1_0 shape";
            token_embd_dtype_ = DType::kQ1_0;
            token_embd_ = dense::upload<void>(alloc, ti, owned_, err, DType::kQ1_0);
            if (!err.empty()) return "token_embd: " + err;
        } else if (ti->dtype == DType::kQ1_0) {
            // Phase A: expand 1-bit blocks to exact Q2_0 host-side, upload the
            // expanded table, and ride the Q2_0 embedding lookup unchanged.
            if (ti->n_dims != 2 || ti->shape[0] % 128 != 0)
                return "token_embd: bad Q1_0 shape";
            const uint64_t nb = (uint64_t(ti->shape[0]) / 128) * ti->shape[1];
            std::vector<uint8_t> q2 = expand_q1_to_q2_blocks(ti->data, nb);
            void* d = alloc.malloc(q2.size());
            if (!d) return "token_embd: malloc failed (Q1_0 expand)";
            owned_.push_back(d);
            alloc.queue().memcpy(d, q2.data(), q2.size()).wait();
            token_embd_ = d;
            token_embd_dtype_ = DType::kQ2_0;   // logical: expanded blocks are Q2_0
        } else {
            token_embd_dtype_ = ti->dtype;
            token_embd_ = dense::upload<void>(alloc, ti, owned_, err, ti->dtype);
            if (!err.empty()) return "token_embd: " + err;
        }
    }
    output_norm_ = dense::upload<float>(alloc, T("output_norm.weight"),
                                        owned_, err, DType::kF32);
    if (!err.empty()) return "output_norm: " + err;
    {
        // qwen35-27B is NOT tied: output.weight is a separate Q6_K lm_head.
        const auto* ti = T("output.weight");
        if (!ti) { output_ = token_embd_; output_dtype_ = token_embd_dtype_; }
        else if (ti->dtype == DType::kQ2_0 || ti->dtype == DType::kQ1_0) {
            // Prism ternary/1-bit lm_head → SoA-Q2 (native W2A8 int-dot decode;
            // Q1_0 expands to exact Q2_0 codes inside upload_q2_soa).
            output_q2soa_ = upload_q2_soa(alloc, ti, owned_, err);
            if (!err.empty()) return "output: " + err;
            output_dtype_ = DType::kQ2_0;   // logical dtype; SoA streams hold it
        }
        else if (ti->dtype == DType::kQ8_0 || ti->dtype == DType::kQ5_K) {
            // gemv_q has no Q8_0/Q5_K path → dequant the lm_head to F16 at load
            // (upload_weight_auto), then gemv_q runs the F16 branch.
            DenseQuantPtr o = upload_weight_auto(alloc, ti, owned_, err);
            if (!err.empty()) return "output: " + err;
            output_ = o.p; output_dtype_ = o.dt;   // o.dt == kF16
        }
        else {
            if (ti->dtype != DType::kQ4_K && ti->dtype != DType::kQ6_K)
                return std::string("output: unsupported dtype ") +
                       std::string(type_name(ti->dtype)) + " (need Q4_K/Q6_K/Q8_0)";
            if (q6_soa_on_ && ti->dtype == DType::kQ6_K) {
                output_soa_ = upload_q6_soa(alloc, ti, owned_, err);
                if (!err.empty()) return "output: " + err;
                output_dtype_ = DType::kQ6_K;   // logical dtype; SoA streams hold it
            } else {
                output_dtype_ = ti->dtype;
                output_ = dense::upload<void>(alloc, ti, owned_, err, ti->dtype);
                if (!err.empty()) return "output: " + err;
            }
        }
    }

    // --- per transformer layer (skip the trailing NextN/MTP layer) ---
    const uint32_t n_layers = cfg.n_transformer_layers();   // 64 (excludes blk.64)
    layers_.assign(n_layers, {});
    for (uint32_t L = 0; L < n_layers; ++L) {
        auto& w = layers_[L];
        w.is_linear = cfg.recurrent_layer(L);
        auto layer_err = [&](const char* what) {
            return "layer " + std::to_string(L) + " " + what + ": " + err;
        };
        auto LQ = [&](const char* name) -> DenseQuantPtr {
            return upload_weight_auto(alloc, Tlayer(L, name), owned_, err);
        };
        // Load a matrix weight, preferring the SoA-Q6 fast-decode repack when the
        // GGUF tensor is Q6_K and the gate is on. On the SoA branch the AoS Q6_K
        // is NOT uploaded (no memory doubling); `dq` is left null and `soa` holds
        // the streams. Otherwise `dq` holds the normal DenseQuantPtr (soa null).
        auto LQsoa = [&](const char* name, DenseQuantPtr& dq, Q6SoaW& soa,
                         Q4SoaW* soa4 = nullptr, Q2SoaW* soa2 = nullptr) {
            const GgufTensorInfo* ti = Tlayer(L, name);
            if (soa2 && ti && (ti->dtype == DType::kQ2_0 ||
                               ti->dtype == DType::kQ1_0)) {
                // Prism ternary 2-bit: always SoA-Q2 (no AoS Q2_0 GEMV exists).
                // Q1_0 rides the same path via exact expansion (Phase A).
                *soa2 = upload_q2_soa(alloc, ti, owned_, err);
                dq = DenseQuantPtr{};   // AoS not loaded
            } else if (q6_soa_on_ && ti && ti->dtype == DType::kQ6_K) {
                soa = upload_q6_soa(alloc, ti, owned_, err);
                dq = DenseQuantPtr{};   // AoS not loaded
            } else if (soa4 && q4_soa_on_ && ti && ti->dtype == DType::kQ4_K) {
                // Opt-IN SoA-Q4 fast-decode repack (mirrors the Q6 branch). AoS
                // Q4_K NOT loaded (no memory doubling); the decode/prefill call
                // site routes through gemv_q4soa_T when the mirror is active.
                *soa4 = upload_q4_soa(alloc, ti, owned_, err);
                dq = DenseQuantPtr{};   // AoS not loaded
            } else {
                dq = upload_weight_auto(alloc, ti, owned_, err);
                // Reordered Q4_K (llama-SYCL 3-region global SoA) for the fast
                // decode kernel — repacked alongside the AoS (env-gated A/B).
                static const bool q4k_reorder =
                    std::getenv("IE_QWEN35_Q4K_REORDER") != nullptr;
                if (q4k_reorder && dq.p && dq.dt == DType::kQ4_K && ti)
                    dq.reorder = upload_q4_reorder(alloc, ti, owned_, err);
            }
        };
        auto F32 = [&](const char* name, float*& dst, const char* what) -> std::string {
            dst = dense::upload<float>(alloc, Tlayer(L, name), owned_, err, DType::kF32);
            if (!err.empty()) return layer_err(what);
            return {};
        };

        // shared: pre-attn norm, pre-FFN (post-attention) norm, dense SwiGLU
        if (auto m = F32("attn_norm.weight", w.attn_norm, "attn_norm"); !m.empty()) return m;
        if (auto m = F32("post_attention_norm.weight", w.post_attn_norm, "post_attention_norm"); !m.empty()) return m;

        if (w.is_linear) {
            // gated-DeltaNet layer
            LQsoa("attn_qkv.weight",  w.attn_qkv,  w.attn_qkv_soa,  &w.attn_qkv_q4soa,  &w.attn_qkv_q2soa);  if (!err.empty()) return layer_err("attn_qkv");
            LQsoa("attn_gate.weight", w.attn_gate, w.attn_gate_soa, &w.attn_gate_q4soa, &w.attn_gate_q2soa); if (!err.empty()) return layer_err("attn_gate");
            if (auto m = F32("ssm_a", w.ssm_a, "ssm_a"); !m.empty()) return m;
            const uint32_t svh_pad = ((cfg.ssm_n_v_heads + 63u) / 64u) * 64u;   // 48 → 64
            w.ssm_alpha = upload_f32_proj_fp16(alloc, Tlayer(L, "ssm_alpha.weight"), owned_, err, svh_pad);
            if (!err.empty()) return layer_err("ssm_alpha");
            w.ssm_beta  = upload_f32_proj_fp16(alloc, Tlayer(L, "ssm_beta.weight"), owned_, err, svh_pad);
            if (!err.empty()) return layer_err("ssm_beta");
            if (auto m = F32("ssm_dt.bias", w.ssm_dt_bias, "ssm_dt.bias"); !m.empty()) return m;
            // ssm_conv1d [conv_kernel, conv_ch] + ssm_norm [v_head_dim] → cast fp16
            {
                const auto* ti = Tlayer(L, "ssm_conv1d.weight");
                if (auto m = F32("ssm_conv1d.weight", w.ssm_conv1d, "ssm_conv1d"); !m.empty()) return m;
                const uint64_t n = ti->nbytes / sizeof(float);
                w.ssm_conv1d_fp16 = static_cast<sycl::half*>(alloc.malloc(n * sizeof(sycl::half)));
                if (!w.ssm_conv1d_fp16) return layer_err("ssm_conv1d fp16 malloc");
                owned_.push_back(w.ssm_conv1d_fp16);
                cast_fp32_to_fp16(alloc.queue(), w.ssm_conv1d, w.ssm_conv1d_fp16, n).wait();
            }
            {
                const auto* ti = Tlayer(L, "ssm_norm.weight");
                if (auto m = F32("ssm_norm.weight", w.ssm_norm, "ssm_norm"); !m.empty()) return m;
                const uint64_t n = ti->nbytes / sizeof(float);
                w.ssm_norm_fp16 = static_cast<sycl::half*>(alloc.malloc(n * sizeof(sycl::half)));
                if (!w.ssm_norm_fp16) return layer_err("ssm_norm fp16 malloc");
                owned_.push_back(w.ssm_norm_fp16);
                cast_fp32_to_fp16(alloc.queue(), w.ssm_norm, w.ssm_norm_fp16, n).wait();
            }
            // ssm_out is Q8_0/Q4_K/Q6_K per layer. Q6_K → SoA-Q6 (via LQsoa);
            // Q8_0 → SoA-Q8 fast lane (decode int-dot ~80% BW, vs F16-expand 2×
            // bytes) — the GEMV-grind win; else (Q4_K) → LQsoa AoS/auto path.
            {
                const GgufTensorInfo* so = Tlayer(L, "ssm_out.weight");
                if (q6_soa_on_ && so && so->dtype == DType::kQ8_0) {
                    w.ssm_out_q8 = upload_q8_soa(alloc, so, owned_, err);
                    w.ssm_out = DenseQuantPtr{};   // F16 copy not loaded
                } else {
                    LQsoa("ssm_out.weight", w.ssm_out, w.ssm_out_soa, &w.ssm_out_q4soa, &w.ssm_out_q2soa);
                }
            }
            if (!err.empty()) return layer_err("ssm_out");
        } else {
            // gated full-attention layer
            LQsoa("attn_q.weight", w.attn_q, w.attn_q_soa, &w.attn_q_q4soa, &w.attn_q_q2soa);  // joint Q|gate (Q6_K typ.)
            if (!err.empty()) return layer_err("attn_q");
            // attn_k/attn_output are Q5_K (read as F16 = ~3× native bytes on the
            // expand path). Requantize → SoA-Q8 (~half bytes, ~80%-BW int-dot lane,
            // near-lossless). Non-Q5_K → the existing auto path. GEMV grind #2.
            {
                const GgufTensorInfo* tk = Tlayer(L, "attn_k.weight");
                if (tk && (tk->dtype == DType::kQ2_0 || tk->dtype == DType::kQ1_0)) {
                    w.attn_k_q2soa = upload_q2_soa(alloc, tk, owned_, err);
                } else if (q6_soa_on_ && tk && tk->dtype == DType::kQ5_K) {
                    w.attn_k_q8 = upload_requant_q5k_q8_soa(alloc, tk, owned_, err);
                } else {
                    w.attn_k = LQ("attn_k.weight");
                }
            }
            if (!err.empty()) return layer_err("attn_k");
            {
                const GgufTensorInfo* tv = Tlayer(L, "attn_v.weight");
                if (tv && (tv->dtype == DType::kQ2_0 || tv->dtype == DType::kQ1_0)) {
                    w.attn_v_q2soa = upload_q2_soa(alloc, tv, owned_, err);
                } else if (q4_soa_on_ && tv && tv->dtype == DType::kQ4_K) {
                    w.attn_v_q4soa = upload_q4_soa(alloc, tv, owned_, err);
                } else {
                    w.attn_v = LQ("attn_v.weight");
                }
            }
            if (!err.empty()) return layer_err("attn_v");
            {
                const GgufTensorInfo* to = Tlayer(L, "attn_output.weight");
                if (to && (to->dtype == DType::kQ2_0 || to->dtype == DType::kQ1_0)) {
                    w.attn_output_q2soa = upload_q2_soa(alloc, to, owned_, err);
                } else if (q6_soa_on_ && to && to->dtype == DType::kQ5_K) {
                    w.attn_output_q8 = upload_requant_q5k_q8_soa(alloc, to, owned_, err);
                } else {
                    w.attn_output = LQ("attn_output.weight");
                }
            }
            if (!err.empty()) return layer_err("attn_output");
            if (auto m = F32("attn_q_norm.weight", w.attn_q_norm, "attn_q_norm"); !m.empty()) return m;
            if (auto m = F32("attn_k_norm.weight", w.attn_k_norm, "attn_k_norm"); !m.empty()) return m;
        }

        LQsoa("ffn_gate.weight", w.ffn_gate, w.ffn_gate_soa, &w.ffn_gate_q4soa, &w.ffn_gate_q2soa); if (!err.empty()) return layer_err("ffn_gate");
        LQsoa("ffn_up.weight",   w.ffn_up,   w.ffn_up_soa,   &w.ffn_up_q4soa,   &w.ffn_up_q2soa);   if (!err.empty()) return layer_err("ffn_up");
        LQsoa("ffn_down.weight", w.ffn_down, w.ffn_down_soa, &w.ffn_down_q4soa, &w.ffn_down_q2soa); if (!err.empty()) return layer_err("ffn_down");
    }
    return {};
}

std::string Qwen35DenseModel::ensure_workspace(uint32_t max_T) {
    if (max_T == 0 || !alloc_) return "alloc not initialized";
    if (max_T <= ws_T_) return {};

    auto ah = [&](size_t n) {
        auto* p = static_cast<sycl::half*>(alloc_->malloc(n * sizeof(sycl::half)));
        if (p) owned_.push_back(p);
        return p;
    };
    auto af = [&](size_t n) {
        auto* p = static_cast<float*>(alloc_->malloc(n * sizeof(float)));
        if (p) owned_.push_back(p);
        return p;
    };

    const DenseConfig& d = cfg_.dense;
    const uint64_t Tt   = max_T;
    const uint32_t H    = d.hidden;                          // 5120
    const uint32_t N_q  = d.n_q_heads  * d.head_dim;         // 6144
    const uint32_t N_qg = N_q * 2u;                          // 12288 (joint Q|gate)
    const uint32_t N_kv = d.n_kv_heads * d.head_dim;         // 1024
    const uint32_t F    = d.ffn;                             // 17408
    const uint32_t d_in = cfg_.ssm_inner;                    // 6144
    const uint32_t cvc  = d_in + 2u * cfg_.ssm_n_k_heads * cfg_.ssm_state;  // 10240
    const uint32_t Vd   = cfg_.ssm_n_v_heads * cfg_.ssm_state;  // 48*128 = 6144 (== d_inner)
    const uint32_t Nv   = cfg_.ssm_n_v_heads;                // 48

    ws_x_          = ah(Tt * H);
    ws_x_normed_   = ah(Tt * H);
    ws_attn_block_ = ah(Tt * H);
    // full-attn
    ws_qg_         = ah(Tt * N_qg);
    ws_q_          = ah(Tt * N_q);
    ws_gate_       = ah(Tt * N_q);
    ws_k_          = ah(Tt * N_kv);
    ws_v_          = ah(Tt * N_kv);
    ws_attn_out_   = ah(Tt * N_q);
    // DeltaNet
    ws_qkv_        = ah(Tt * cvc);
    ws_conv_       = ah(Tt * cvc);
    ws_dn_z_       = ah(Tt * d_in);
    ws_qpre_       = af(Tt * Vd);
    ws_kpre_       = af(Tt * Vd);
    ws_vpre_       = af(Tt * Vd);
    ws_g_          = af(Tt * Nv);
    ws_beta_       = af(Tt * Nv);
    ws_dn_out_     = af(Tt * Vd);
    ws_qrep_       = af(Tt * Vd);
    ws_krep_       = af(Tt * Vd);
    ws_alpha_h_    = ah(Tt * Nv);
    ws_beta_h_     = ah(Tt * Nv);
    const uint32_t Nvp = ((Nv + 63u) / 64u) * 64u;   // 48 → 64 (gemm N-padding)
    ws_alpha64_    = ah(Tt * Nvp);
    ws_beta64_     = ah(Tt * Nvp);
    // FFN
    ws_ffn_gate_   = ah(Tt * F);
    ws_ffn_up_     = ah(Tt * F);
    ws_ffn_h_      = ah(Tt * F);

    ws_positions_  = static_cast<int32_t*>(alloc_->malloc(Tt * sizeof(int32_t)));
    if (ws_positions_) owned_.push_back(ws_positions_);
    {
        // Q8_1 decode scratch: largest decode GEMV input K = max(H, F, conv_ch).
        // Sized for kBatchedVerifyMax rows so the small-T spec-decode verify can
        // stage T activation rows (row t at block offset t*(Kmax/32)); the T==1
        // decode path uses only the first row. ~16×544×40 B ≈ 348 KiB — trivial.
        const uint32_t Kmax = std::max(std::max(H, F), cvc);
        // block_q8_1s (48 B, batched verify) > block_q8_1x (40 B, T==1 decode) —
        // size by the larger so the same buffer serves both quant formats.
        const uint64_t blk_bytes =
            std::max(sizeof(block_q8_1x), sizeof(block_q8_1s));
        // IE_Q2_BATCHED native W2A8 prefill stages max_T activation rows (not
        // just the spec-verify 16) — grow the row count only when that gate is
        // on so the default allocation stays byte-identical.
        const uint32_t q8_rows = q2_batched_prefill_on()
            ? std::max<uint32_t>(kBatchedVerifyMax, max_T) : kBatchedVerifyMax;
        void* p = alloc_->malloc(uint64_t(q8_rows) * (uint64_t(Kmax) / 32) *
                                 blk_bytes);
        if (p) owned_.push_back(p);
        ws_q8_ = p;
    }
    {
        // Quant-hoist persistent Q8_1 streams (decode T==1). One stream per norm
        // that feeds ≥2 SoA-GEMV consumers reading the SAME vector: pre-attn norm
        // (H) and pre-FFN norm (H). block_q8_1x (40 B) × (H/32) ≈ 6.4 KiB each —
        // trivial. Allocated unconditionally; only written/read when the gate is on.
        const uint64_t q8_bytes = (uint64_t(H) / 32) * sizeof(block_q8_1x);
        void* pa = alloc_->malloc(q8_bytes);
        void* pf = alloc_->malloc(q8_bytes);
        if (pa) owned_.push_back(pa);
        if (pf) owned_.push_back(pf);
        ws_attn_norm_q8_ = pa;
        ws_ffn_in_q8_    = pf;
    }

    if (!ws_x_ || !ws_x_normed_ || !ws_attn_block_ || !ws_qg_ || !ws_q_ ||
        !ws_gate_ || !ws_k_ || !ws_v_ || !ws_attn_out_ || !ws_qkv_ || !ws_conv_ ||
        !ws_dn_z_ || !ws_qpre_ || !ws_kpre_ || !ws_vpre_ || !ws_g_ || !ws_beta_ ||
        !ws_dn_out_ || !ws_qrep_ || !ws_krep_ || !ws_alpha_h_ || !ws_beta_h_ ||
        !ws_alpha64_ || !ws_beta64_ ||
        !ws_ffn_gate_ || !ws_ffn_up_ || !ws_ffn_h_ ||
        !ws_positions_ || !ws_q8_ || !ws_attn_norm_q8_ || !ws_ffn_in_q8_) {
        return "qwen35 workspace allocation failed";
    }
    ws_T_ = max_T;
    return {};
}

sycl::event Qwen35DenseModel::gemv_q6soa_T(sycl::queue& q, const sycl::half* A,
                                           const Q6SoaW& w, sycl::half* y,
                                           uint32_t T, const void* x_q8_pre) {
    const uint32_t K = w.K, N = w.N;
    if (T == 0) return {};
    if (T == 1) {
        // Decode: quantize the single activation vector to Q8_1 once, int-dot.
        // Quant-hoist: if a pre-quantized Q8_1 stream was supplied (same vector,
        // bit-identical numerics), skip the redundant quantize_q8_1 launch.
        if (x_q8_pre) return gemv_q6_soa_q8(q, x_q8_pre, w.lo, w.hi, w.sc, w.d, y, K, N);
        sycl::event qe = quantize_q8_1(q, A, ws_q8_, K);
        return gemv_q6_soa_q8(q, ws_q8_, w.lo, w.hi, w.sc, w.d, y, K, N, {qe});
    }
    // Spec-decode VERIFY (small T): batched int-dot — read each weight column
    // ONCE, dot against T staged Q8_1 activation rows.  Amortizes weight BW over
    // T (the spec-decode amortization) instead of the T-independent dequant→gemm
    // restream.  Quantize all [T,K] rows in one launch (K%32==0 → row t's blocks
    // land at offset t*(K/32)).  Opt-out IE_QWEN35_NO_BATCHED_VERIFY=1.
    if (T >= 2 && T <= kBatchedVerifyMax && batched_verify_on_) {
        sycl::event qe = quantize_q8_1(q, A, ws_q8_, uint32_t(uint64_t(T) * K));
        return gemv_q6_soa_q8_batched(q, ws_q8_, w.lo, w.hi, w.sc, w.d, y, K, N, T, {qe});
    }
    // Prefill (T≥2): dequant SoA → fp16 Bt[K,N] scratch, then gemm_fp16.
    // Per-device scratch reused across projections (single GPU, one gen at once).
    const uint64_t bt_need = uint64_t(K) * N;
    const uint64_t c_need  = (uint64_t(T) + 8) * N;     // +8: gemm_fp16 TM=8 overrun
    if (bt_need > q6soa_bt_cap_) {
        if (q6soa_bt_) sycl::free(q6soa_bt_, q);
        q6soa_bt_ = sycl::malloc_device<sycl::half>(bt_need, q);
        q6soa_bt_cap_ = q6soa_bt_ ? bt_need : 0;
    }
    if (c_need > q6soa_c_cap_) {
        if (q6soa_c_) sycl::free(q6soa_c_, q);
        q6soa_c_ = sycl::malloc_device<float>(c_need, q);
        q6soa_c_cap_ = q6soa_c_ ? c_need : 0;
    }
    if (q6soa_bt_ && q6soa_c_) {
        sycl::event de = dequant_q6_soa_to_Bt(q, w.lo, w.hi, w.sc, w.d,
                                              q6soa_bt_, K, N);
        if (dense::prefer_onednn() && (N % 64 == 0) && (K % 256 == 0))
            return gemm_fp16_onednn(q, A, q6soa_bt_, y, T, N, K, {de});
        sycl::event ge = gemm_fp16(q, A, q6soa_bt_, q6soa_c_, T, N, K, {de});
        return cast_fp32_to_fp16(q, q6soa_c_, y, uint64_t(T) * N, {ge});
    }
    return {};   // scratch alloc failed — should not happen on the single GPU
}

sycl::event Qwen35DenseModel::gemv_q8soa_T(sycl::queue& q, const sycl::half* A,
                                           const Q8SoaW& w, sycl::half* y,
                                           uint32_t T, const void* x_q8_pre) {
    const uint32_t K = w.K, N = w.N;
    if (T == 0) return {};
    if (T == 1) {
        // Decode: quantize the activation to Q8_1 once, int-dot W8A8 (~80% BW —
        // half the bytes of the F16-expanded path this replaces).
        // Quant-hoist: reuse a pre-quantized Q8_1 stream when supplied.
        if (x_q8_pre) return gemv_q8_0_soa_q8(q, x_q8_pre, w.qs, w.d, y, K, N);
        sycl::event qe = quantize_q8_1(q, A, ws_q8_, K);
        return gemv_q8_0_soa_q8(q, ws_q8_, w.qs, w.d, y, K, N, {qe});
    }
    // T≥2 (spec verify + prefill): no batched Q8 int-dot kernel yet → dequant the
    // SoA-Q8 weight to fp16 Bt[K,N] once, then gemm_fp16 (reuses the Q6 prefill
    // scratch — single GPU, one gen at a time).
    const uint64_t bt_need = uint64_t(K) * N;
    const uint64_t c_need  = (uint64_t(T) + 8) * N;     // +8: gemm_fp16 TM=8 overrun
    if (bt_need > q6soa_bt_cap_) {
        if (q6soa_bt_) sycl::free(q6soa_bt_, q);
        q6soa_bt_ = sycl::malloc_device<sycl::half>(bt_need, q);
        q6soa_bt_cap_ = q6soa_bt_ ? bt_need : 0;
    }
    if (c_need > q6soa_c_cap_) {
        if (q6soa_c_) sycl::free(q6soa_c_, q);
        q6soa_c_ = sycl::malloc_device<float>(c_need, q);
        q6soa_c_cap_ = q6soa_c_ ? c_need : 0;
    }
    if (q6soa_bt_ && q6soa_c_) {
        sycl::event de = dequant_q8_0_soa_to_Bt(q, w.qs, w.d, q6soa_bt_, K, N);
        if (dense::prefer_onednn() && (N % 64 == 0) && (K % 256 == 0))
            return gemm_fp16_onednn(q, A, q6soa_bt_, y, T, N, K, {de});
        sycl::event ge = gemm_fp16(q, A, q6soa_bt_, q6soa_c_, T, N, K, {de});
        return cast_fp32_to_fp16(q, q6soa_c_, y, uint64_t(T) * N, {ge});
    }
    return {};
}

sycl::event Qwen35DenseModel::gemv_q4soa_T(sycl::queue& q, const sycl::half* A,
                                           const Q4SoaW& w, sycl::half* y,
                                           uint32_t T, const void* x_q8_pre) {
    const uint32_t K = w.K, N = w.N;
    if (T == 0) return {};
    if (T == 1) {
        // Decode: quantize the single activation vector to Q8_1 once, int-dot W4A8.
        // Quant-hoist: reuse a pre-quantized Q8_1 stream when supplied.
        if (x_q8_pre) return gemv_q4_soa_q8(q, x_q8_pre, w.q, w.sc, w.mn, w.d, w.dmin, y, K, N);
        sycl::event qe = quantize_q8_1(q, A, ws_q8_, K);
        return gemv_q4_soa_q8(q, ws_q8_, w.q, w.sc, w.mn, w.d, w.dmin, y, K, N, {qe});
    }
    // T≥2 (spec verify + prefill): no batched SoA-Q4 int-dot kernel yet → dequant
    // the SoA-Q4 weight to fp16 Bt[K,N] once, then gemm_fp16 (reuses the Q6/Q8
    // prefill scratch — single GPU, one gen at a time). Mirrors gemv_q8soa_T.
    const uint64_t bt_need = uint64_t(K) * N;
    const uint64_t c_need  = (uint64_t(T) + 8) * N;     // +8: gemm_fp16 TM=8 overrun
    if (bt_need > q6soa_bt_cap_) {
        if (q6soa_bt_) sycl::free(q6soa_bt_, q);
        q6soa_bt_ = sycl::malloc_device<sycl::half>(bt_need, q);
        q6soa_bt_cap_ = q6soa_bt_ ? bt_need : 0;
    }
    if (c_need > q6soa_c_cap_) {
        if (q6soa_c_) sycl::free(q6soa_c_, q);
        q6soa_c_ = sycl::malloc_device<float>(c_need, q);
        q6soa_c_cap_ = q6soa_c_ ? c_need : 0;
    }
    if (q6soa_bt_ && q6soa_c_) {
        sycl::event de = dequant_q4_soa_to_Bt(q, w.q, w.sc, w.mn, w.d, w.dmin,
                                              q6soa_bt_, K, N);
        if (dense::prefer_onednn() && (N % 64 == 0) && (K % 256 == 0))
            return gemm_fp16_onednn(q, A, q6soa_bt_, y, T, N, K, {de});
        sycl::event ge = gemm_fp16(q, A, q6soa_bt_, q6soa_c_, T, N, K, {de});
        return cast_fp32_to_fp16(q, q6soa_c_, y, uint64_t(T) * N, {ge});
    }
    return {};
}

sycl::event Qwen35DenseModel::gemv_q2soa_T(sycl::queue& q, const sycl::half* A,
                                           const Q2SoaW& w, sycl::half* y,
                                           uint32_t T, const void* x_q8_pre) {
    const uint32_t K = w.K, N = w.N;
    if (T == 0) return {};
    const bool is_q1 = (w.dtype == DType::kQ1_0);   // native Q1_0 (Phase B) streams
    if (T == 1) {
        // Decode: quantize the single activation vector to Q8_1 once, int-dot
        // W2A8 (Q2) or W1A8 (native Q1). Quant-hoist: reuse a pre-quantized Q8_1
        // stream when supplied.
        const void* xq = x_q8_pre;
        std::vector<sycl::event> d;
        if (!xq) { sycl::event qe = quantize_q8_1(q, A, ws_q8_, K); xq = ws_q8_; d = {qe}; }
        if (is_q1) return gemv_q1_0_soa_q8(q, xq, w.qs, w.d, y, K, N, d);
        return gemv_q2_0_soa_q8(q, xq, w.qs, w.d, y, K, N, d);
    }
    // Native Q1_0 T≥2 (prefill / spec verify): the Q2 batched/XMX opt-in kernels
    // don't understand the 1-bit qs plane, so route straight to the dequant→gemm
    // path (dequant_q1_0_soa_to_Bt below). No native W1A8 batched kernel yet.
    if (is_q1) {
        const uint64_t bt_need = uint64_t(K) * N;
        const uint64_t c_need  = (uint64_t(T) + 8) * N;     // +8: gemm_fp16 TM=8 overrun
        if (bt_need > q6soa_bt_cap_) {
            if (q6soa_bt_) sycl::free(q6soa_bt_, q);
            q6soa_bt_ = sycl::malloc_device<sycl::half>(bt_need, q);
            q6soa_bt_cap_ = q6soa_bt_ ? bt_need : 0;
        }
        if (c_need > q6soa_c_cap_) {
            if (q6soa_c_) sycl::free(q6soa_c_, q);
            q6soa_c_ = sycl::malloc_device<float>(c_need, q);
            q6soa_c_cap_ = q6soa_c_ ? c_need : 0;
        }
        if (q6soa_bt_ && q6soa_c_) {
            sycl::event de = dequant_q1_0_soa_to_Bt(q, w.qs, w.d, q6soa_bt_, K, N);
            if (dense::prefer_onednn() && (N % 64 == 0) && (K % 256 == 0))
                return gemm_fp16_onednn(q, A, q6soa_bt_, y, T, N, K, {de});
            sycl::event ge = gemm_fp16(q, A, q6soa_bt_, q6soa_c_, T, N, K, {de});
            return cast_fp32_to_fp16(q, q6soa_c_, y, uint64_t(T) * N, {ge});
        }
        return {};
    }
    // Native W2A8 batched int-dot prefill (IE_Q2_BATCHED opt-IN, default OFF →
    // byte-identical dequant→gemm below): read each SoA-Q2 weight column ONCE,
    // dp4a vs T staged Q8_1 rows (docs/q2_0_optimization/07) — deletes the
    // T-independent 15× weight round-trip of dequant_q2_soa→gemm_fp16. Same
    // one-launch [T,K] activation quant as the Q6 batched-verify branch (row t
    // at block offset t*(K/32)); ws_q8_ is sized for ws_T_ ≥ T rows when the
    // gate is on (see ensure_workspace). K%512: wide-load kernel geometry (all
    // Bonsai shapes pass; others fall through to the dequant path).
    if (q2_batched_prefill_on() && T >= 2 && (K % 512u) == 0u && T <= ws_T_) {
        sycl::event qe = quantize_q8_1(q, A, ws_q8_, uint32_t(uint64_t(T) * K));
        return gemv_q2_0_soa_q8_batched(q, ws_q8_, w.qs, w.d, y, K, N, T, {qe});
    }
    // Fused-dequant XMX/DPAS prefill (IE_Q2_XMX_PREFILL opt-IN, default OFF):
    // cooperatively expand this WG's 64 Q2 columns → fp16 SLM stripe ONCE per
    // K-block, then fp16 joint_matrix DPAS against the T-chunk (internal
    // M-tiling) — never materializes the fp16 Bt[K,N] the dequant→gemm path
    // writes+reads (53.8 GB/pass). y[T,N] fp16 written directly; no Bt/C scratch
    // needed. Guarded on the tile dims (N%64==0, K%256==0 — Bonsai FFN/attn all
    // pass); else falls through to the dequant→gemm path below.
    if (q2_xmx_prefill_on() && T >= 2 && (N % 64u) == 0u && (K % 256u) == 0u) {
        return gemm_q2_0_soa_xmx(q, A, w.qs, w.d, y, T, N, K);
    }
    // T≥2 (spec verify + prefill) default: dequant the SoA-Q2 weight to fp16
    // Bt[K,N] once, then gemm_fp16 (reuses the Q6/Q8/Q4 prefill scratch —
    // single GPU, one gen at a time). Mirrors gemv_q4soa_T.
    const uint64_t bt_need = uint64_t(K) * N;
    const uint64_t c_need  = (uint64_t(T) + 8) * N;     // +8: gemm_fp16 TM=8 overrun
    if (bt_need > q6soa_bt_cap_) {
        if (q6soa_bt_) sycl::free(q6soa_bt_, q);
        q6soa_bt_ = sycl::malloc_device<sycl::half>(bt_need, q);
        q6soa_bt_cap_ = q6soa_bt_ ? bt_need : 0;
    }
    if (c_need > q6soa_c_cap_) {
        if (q6soa_c_) sycl::free(q6soa_c_, q);
        q6soa_c_ = sycl::malloc_device<float>(c_need, q);
        q6soa_c_cap_ = q6soa_c_ ? c_need : 0;
    }
    if (q6soa_bt_ && q6soa_c_) {
        sycl::event de = dequant_q2_0_soa_to_Bt(q, w.qs, w.d, q6soa_bt_, K, N);
        if (dense::prefer_onednn() && (N % 64 == 0) && (K % 256 == 0))
            return gemm_fp16_onednn(q, A, q6soa_bt_, y, T, N, K, {de});
        sycl::event ge = gemm_fp16(q, A, q6soa_bt_, q6soa_c_, T, N, K, {de});
        return cast_fp32_to_fp16(q, q6soa_c_, y, uint64_t(T) * N, {ge});
    }
    return {};
}

std::string Qwen35DenseModel::ensure_attn_partials(uint32_t max_ctx) {
    if (!alloc_) return "alloc not initialized";
    if (max_ctx <= ws_attn_partials_ctx_) return {};
    constexpr uint32_t Bc_floor = 64;
    const uint32_t n_chunks_max = (max_ctx + Bc_floor - 1) / Bc_floor;
    const uint64_t n_floats =
        uint64_t(n_chunks_max) * cfg_.dense.n_q_heads * (cfg_.dense.head_dim + 2);
    auto* p = static_cast<float*>(alloc_->malloc(n_floats * sizeof(float)));
    if (!p) return "qwen35 attn_partials alloc failed";
    owned_.push_back(p);
    ws_attn_partials_     = p;
    ws_attn_partials_ctx_ = max_ctx;
    return {};
}

// ===========================================================================
// Qwen35SpecCheckpoint — per-position DeltaNet state snapshots (spec-decode).
// ===========================================================================
std::string Qwen35SpecCheckpoint::init(DeviceAllocator& a,
                                       const DeltaNetState& dn, uint32_t K_) {
    const uint32_t nl = dn.config().n_layers_linear;
    const uint64_t se = dn.state_elems_per_layer();
    const uint64_t ce = dn.conv_elems_per_layer();
    if (alloc == &a && K == K_ && n_lin == nl && state_elems == se &&
        conv_elems == ce && ckpt_state && ckpt_conv)
        return {};   // already sized
    free_storage();
    alloc = &a; K = K_; n_lin = nl; state_elems = se; conv_elems = ce;
    if (K == 0 || nl == 0 || se == 0 || ce == 0) return "checkpoint zero dim";
    const uint64_t s_bytes = uint64_t(K) * nl * se * sizeof(float);
    const uint64_t c_bytes = uint64_t(K) * nl * ce * sizeof(sycl::half);
    ckpt_state = static_cast<float*>(a.malloc(s_bytes));
    ckpt_conv  = static_cast<sycl::half*>(a.malloc(c_bytes));
    if (!ckpt_state || !ckpt_conv) { free_storage(); return "checkpoint alloc failed"; }
    return {};
}

void Qwen35SpecCheckpoint::free_storage() noexcept {
    if (alloc) {
        if (ckpt_state) alloc->free(ckpt_state);
        if (ckpt_conv)  alloc->free(ckpt_conv);
    }
    ckpt_state = nullptr; ckpt_conv = nullptr;
}

std::string Qwen35SpecCheckpoint::commit_to_n(sycl::queue& q, DeltaNetState& dn,
                                              uint32_t accepted) const {
    if (!ckpt_state || !ckpt_conv) return "commit: checkpoint not ready";
    if (accepted == 0 || accepted > K) return "commit: accepted out of range";
    if (dn.config().n_layers_linear != n_lin ||
        dn.state_elems_per_layer() != state_elems ||
        dn.conv_elems_per_layer()  != conv_elems)
        return "commit: dn geometry mismatch";
    // Snapshot taken AFTER verify-position (accepted-1) → slice s = accepted-1.
    const uint32_t s = accepted - 1;
    const uint64_t s_off = uint64_t(s) * n_lin * state_elems;
    const uint64_t c_off = uint64_t(s) * n_lin * conv_elems;
    q.memcpy(dn.state_ptr(),      ckpt_state + s_off,
             uint64_t(n_lin) * state_elems * sizeof(float));
    q.memcpy(dn.conv_state_ptr(), ckpt_conv  + c_off,
             uint64_t(n_lin) * conv_elems * sizeof(sycl::half));
    q.wait();
    return {};
}

// ===========================================================================
// MtpHead — native MTP/NextN draft head (loaded only when --spec). Lifted from
// the validated tools/ie_qwen35_spec.cpp; matrices dequanted to F16 [K,N] so
// they ride gemv_fp16 (the draft is all M=1 GEMVs).
// ===========================================================================
namespace {

// Dequant ANY supported matrix dtype (Q4_K/Q6_K/Q5_K/Q8_0/F16) → device F16
// [K,N] (transposed), so the MTP head can run every matmul through gemv_fp16.
// Host-side AoS block_q8_0 → SoA (qs[n*K+k] int8 col-contig + fp16 d per 32-block)
// upload for an MTP weight. One-shot at load; ~340 MB total across the block.
// Returns {nullptr,nullptr} (no error) when the source is not Q8_0.
MtpHead::SoaW mtp_upload_q8_soa(DeviceAllocator& alloc, const GgufTensorInfo* t,
                                std::vector<void*>& owned, std::string& err) {
    MtpHead::SoaW w{};
    if (!t || t->n_dims != 2 || t->dtype != DType::kQ8_0) return w;
    const uint32_t K = uint32_t(t->shape[0]);
    const uint32_t N = uint32_t(t->shape[1]);
    if (K % 32) return w;
    const uint32_t bpc = K / 32;
    struct BQ8 { uint16_t d; int8_t qs[32]; };
    static_assert(sizeof(BQ8) == 34, "block_q8_0 host view");
    const BQ8* src = reinterpret_cast<const BQ8*>(t->data);
    std::vector<int8_t>   hq((uint64_t)N * K);
    std::vector<uint16_t> hd((uint64_t)N * bpc);
    for (uint32_t n = 0; n < N; ++n)
        for (uint32_t b = 0; b < bpc; ++b) {
            const BQ8& blk = src[(uint64_t)n * bpc + b];
            hd[(uint64_t)n * bpc + b] = blk.d;
            std::memcpy(hq.data() + (uint64_t)n * K + (uint64_t)b * 32, blk.qs, 32);
        }
    auto* dq = static_cast<int8_t*>(alloc.malloc(hq.size()));
    auto* dd = static_cast<uint16_t*>(alloc.malloc(hd.size() * sizeof(uint16_t)));
    if (!dq || !dd) { if (dq) alloc.free(dq); if (dd) alloc.free(dd);
                      err = "mtp soa malloc"; return w; }
    alloc.queue().memcpy(dq, hq.data(), hq.size()).wait();
    alloc.queue().memcpy(dd, hd.data(), hd.size() * sizeof(uint16_t)).wait();
    owned.push_back(dq); owned.push_back(dd);
    w.qs = dq; w.d = dd;
    return w;
}

sycl::half* mtp_dequant_any_to_fp16(DeviceAllocator& alloc,
                                    const GgufTensorInfo* t,
                                    std::vector<void*>& owned, std::string& err) {
    if (!t) { err = "tensor not found"; return nullptr; }
    if (t->n_dims != 2) { err = "mtp weight: expected 2-D"; return nullptr; }
    const uint32_t K = uint32_t(t->shape[0]);
    const uint32_t N = uint32_t(t->shape[1]);
    if (t->dtype == DType::kF16) {
        // already F16 [N,K] row-major → upload + transpose. Simplest: route it
        // through the same dequant-less copy path the dense loader has — but the
        // MTP head expects [K,N]. F16 lm_heads are rare here; hard-fail to keep
        // the path honest if it ever appears (none in the bartowski 27B GGUF).
        err = "mtp weight: F16 not supported (expected quantized)"; return nullptr;
    }
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
        default:
            alloc.free(packed); alloc.free(d);
            err = std::string("mtp weight: unsupported dtype ") +
                  std::string(type_name(t->dtype));
            return nullptr;
    }
    e.wait();
    alloc.free(packed);          // transient
    owned.push_back(d);
    return d;
}

int mtp_argmax_row(const sycl::half* row, uint32_t vocab) {
    float best = float(row[0]); int arg = 0;
    for (uint32_t v = 1; v < vocab; ++v) {
        float val = float(row[v]);
        if (val > best) { best = val; arg = int(v); }
    }
    return arg;
}

sycl::event mtp_argmax_device(sycl::queue& q, const sycl::half* row,
                              uint32_t vocab, int32_t* out) {
    constexpr uint32_t WG = 256;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> best_val(WG, h);
        sycl::local_accessor<uint32_t, 1> best_idx(WG, h);
        h.parallel_for(sycl::nd_range<1>(WG, WG), [=](sycl::nd_item<1> it) {
            const uint32_t lid = uint32_t(it.get_local_id(0));

            // The host scan initializes from row[0]. A NaN there can never be
            // displaced by strict `>`, so preserve that corner case directly.
            if (sycl::isnan(float(row[0]))) {
                if (lid == 0) *out = 0;
                return;
            }

            float bv = -std::numeric_limits<float>::infinity();
            uint32_t bi = UINT32_MAX;
            for (uint32_t v = lid; v < vocab; v += WG) {
                const float x = float(row[v]);
                // Later NaNs fail the host scan's strict `x > best` test. For
                // equal finite values, retain the first (lowest) vocab index.
                if (x > bv || (x == bv && v < bi)) {
                    bv = x;
                    bi = v;
                }
            }
            best_val[lid] = bv;
            best_idx[lid] = bi;
            it.barrier(sycl::access::fence_space::local_space);
            for (uint32_t stride = WG / 2; stride != 0; stride >>= 1) {
                if (lid < stride) {
                    const float rv = best_val[lid + stride];
                    const uint32_t ri = best_idx[lid + stride];
                    if (rv > best_val[lid] ||
                        (rv == best_val[lid] && ri < best_idx[lid])) {
                        best_val[lid] = rv;
                        best_idx[lid] = ri;
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);
            }
            if (lid == 0) *out = int32_t(best_idx[0]);
        });
    });
}

}  // namespace

MtpHead::~MtpHead() {
    if (alloc) for (void* p : owned) alloc->free(p);
}

std::string MtpHead::load(DeviceAllocator& a, const GgufReader& g,
                          const Qwen35Config& cfg, uint32_t max_ctx_) {
    if (loaded) return {};
    alloc = &a;
    const DenseConfig& dc = cfg.dense;
    H = dc.hidden; HD = dc.head_dim;
    N_q = dc.n_q_heads * HD; N_qg = N_q * 2u; N_kv = dc.n_kv_heads * HD;
    F = dc.ffn; rope_n = dc.rope_dim; vocab = dc.vocab;
    eps = dc.rms_eps; rope_theta = dc.rope_theta;
    n_q_heads = dc.n_q_heads; n_kv_heads = dc.n_kv_heads;
    mtp_blk = cfg.n_transformer_layers();   // 64
    max_ctx = max_ctx_;
    // Default ON at 64k. The draft head's full-vocab GEMV was ~80% of draft cost
    // (1.27 GB/step over 248,320 rows) to produce a GUESS the target re-checks,
    // so narrowing it is lossless by construction — a token outside the prefix is
    // never proposed, costing acceptance, never correctness. GGUF vocabularies
    // are frequency-ordered, so a prefix is a good subset for English/code and a
    // poor one for CJK (ids 100k+): 64k is the safe default, 32768 measured
    // ~3.8% faster on English/code. IE_QWEN35_DRAFT_VOCAB=0 restores full vocab.
    draft_vocab = std::min<uint32_t>(65536u, vocab);
    if (const char* dv = std::getenv("IE_QWEN35_DRAFT_VOCAB")) {
        const long v = std::strtol(dv, nullptr, 10);
        draft_vocab = (v > 0 && uint32_t(v) < vocab) ? uint32_t(v) : 0;
        if (v > 0 && uint32_t(v) < vocab) {
            std::fprintf(stderr, "[mtp] draft vocabulary limited to %u of %u "
                                 "(head bytes %.0f%%; emitted text unaffected)\n",
                         draft_vocab, vocab, 100.0 * double(draft_vocab) / double(vocab));
        }
    }

    auto& q = a.queue();
    char buf[64];
    std::string err;
    auto MT = [&](const char* name) {
        std::snprintf(buf, sizeof(buf), "blk.%u.%s", mtp_blk, name);
        return g.find_tensor(buf);
    };
    auto reqf16 = [&](const char* n) -> sycl::half* {
        auto* p = mtp_dequant_any_to_fp16(a, MT(n), owned, err);
        if (!err.empty()) err = std::string("blk.") + std::to_string(mtp_blk) +
                                 "." + n + ": " + err;
        return p;
    };
    // Q8_0-SoA-or-F16: keep the SoA int8 (half the draft GEMV bytes) when the
    // source is Q8_0; otherwise fall back to the dequanted-F16 copy. Exactly one
    // of (soa.qs, *f16) is set per weight; run_layer branches per site.
    auto req_soa_or_f16 = [&](const char* n, MtpHead::SoaW& soa) -> sycl::half* {
        soa = mtp_upload_q8_soa(a, MT(n), owned, err);
        if (!err.empty()) { err = std::string("blk.") + std::to_string(mtp_blk) +
                                  "." + n + ": " + err; return nullptr; }
        if (soa.qs) return nullptr;          // SoA path active — no F16 needed
        return reqf16(n);
    };
    auto reqf32 = [&](const char* n) -> float* {
        const auto* t = MT(n);
        if (!t) { err = std::string(n) + ": not found"; return nullptr; }
        if (t->dtype != DType::kF32) { err = std::string(n) + ": expected F32"; return nullptr; }
        void* d = a.malloc(t->nbytes);
        if (!d) { err = std::string(n) + ": malloc"; return nullptr; }
        q.memcpy(d, t->data, t->nbytes).wait();
        owned.push_back(d);
        return static_cast<float*>(d);
    };

    eh_proj    = req_soa_or_f16("nextn.eh_proj.weight", s_eh);   if (!err.empty()) return err;
    enorm      = reqf32("nextn.enorm.weight");            if (!err.empty()) return err;
    hnorm      = reqf32("nextn.hnorm.weight");            if (!err.empty()) return err;
    shead_norm = reqf32("nextn.shared_head_norm.weight"); if (!err.empty()) return err;
    w_attn_q   = req_soa_or_f16("attn_q.weight", s_q);    if (!err.empty()) return err;
    w_attn_k   = req_soa_or_f16("attn_k.weight", s_k);    if (!err.empty()) return err;
    w_attn_v   = req_soa_or_f16("attn_v.weight", s_v);    if (!err.empty()) return err;
    w_attn_out = req_soa_or_f16("attn_output.weight", s_ao); if (!err.empty()) return err;
    attn_norm      = reqf32("attn_norm.weight");          if (!err.empty()) return err;
    post_attn_norm = reqf32("post_attention_norm.weight");if (!err.empty()) return err;
    attn_q_norm    = reqf32("attn_q_norm.weight");        if (!err.empty()) return err;
    attn_k_norm    = reqf32("attn_k_norm.weight");        if (!err.empty()) return err;
    w_ffn_gate = req_soa_or_f16("ffn_gate.weight", s_fg); if (!err.empty()) return err;
    w_ffn_up   = req_soa_or_f16("ffn_up.weight", s_fu);   if (!err.empty()) return err;
    w_ffn_down = req_soa_or_f16("ffn_down.weight", s_fd); if (!err.empty()) return err;

    // shared lm_head = output.weight (top-level, not under blk.64)
    {
        const auto* o = g.find_tensor("output.weight");
        if (!o) return "output.weight: not found (MTP shared head)";
        w_lm_head = mtp_dequant_any_to_fp16(a, o, owned, err);
        if (!err.empty()) return "output.weight: " + err;
    }
    // token_embd (independent upload for the lookup)
    {
        const auto* te = g.find_tensor("token_embd.weight");
        if (!te) return "token_embd.weight: not found (MTP)";
        te_dtype = te->dtype;
        te_dev = a.malloc(te->nbytes);
        if (!te_dev) return "token_embd: malloc (MTP)";
        q.memcpy(te_dev, te->data, te->nbytes).wait();
        owned.push_back(te_dev);
    }

    // scratch
    auto A = [&](uint64_t n) -> sycl::half* {
        auto* p = sycl::malloc_device<sycl::half>(n, q);
        if (p) owned.push_back(p);
        return p;
    };
    d_h=A(H); d_e=A(H); d_hn=A(H); d_en=A(H); d_cat=A(2u*H); d_x=A(H);
    d_xn=A(H); d_qg=A(N_qg); d_q=A(N_q); d_gate=A(N_q); d_k=A(N_kv);
    d_v=A(N_kv); d_ao=A(N_q); d_blk=A(H); d_fg=A(F); d_fu=A(F); d_fh=A(F);
    d_logits1=A(vocab);
    d_pos1 = sycl::malloc_device<int32_t>(1, q); if (d_pos1) owned.push_back(d_pos1);
    d_tok1 = sycl::malloc_device<int32_t>(1, q); if (d_tok1) owned.push_back(d_tok1);
    d_draft_ids = sycl::malloc_device<int32_t>(16, q);
    if (d_draft_ids) owned.push_back(d_draft_ids);
    d_act_q8 = sycl::malloc_device<unsigned char>((uint64_t(F) / 32) * sizeof(block_q8_1x), q);
    if (d_act_q8) owned.push_back(d_act_q8);
    mtp_kc = A(uint64_t(n_kv_heads) * max_ctx * HD);
    mtp_vc = A(uint64_t(n_kv_heads) * max_ctx * HD);
    if (!d_logits1 || !mtp_kc || !mtp_vc || !d_pos1 || !d_tok1 || !d_draft_ids)
        return "MTP head scratch alloc failed";

    loaded = true;
    return {};
}

void MtpHead::embed(sycl::queue& q, int32_t tok) {
    q.memcpy(d_tok1, &tok, sizeof(int32_t));
    embed_device(q, d_tok1);
}

void MtpHead::embed_device(sycl::queue& q, const int32_t* tok) {
    if (te_dtype == DType::kQ4_K)      embedding_lookup_q4k(q, tok, te_dev, d_e, 1, H);
    else if (te_dtype == DType::kQ6_K) embedding_lookup_q6k(q, tok, te_dev, d_e, 1, H);
    else                               embedding_lookup_q8_0(q, tok, te_dev, d_e, 1, H);
}

void MtpHead::build_x(sycl::queue& q, const sycl::half* h_src, int32_t e_tok) {
    embed(q, e_tok);
    q.memcpy(d_h, h_src, uint64_t(H) * sizeof(sycl::half));
    rms_norm_f32w(q, d_e, enorm, d_en, 1, H, eps);
    rms_norm_f32w(q, d_h, hnorm, d_hn, 1, H, eps);
    q.parallel_for(sycl::range<1>(uint64_t(H)),
                   [=, en=d_en, hn=d_hn, cat=d_cat, Hl=H](sycl::id<1> i) {
        cat[uint64_t(i)]      = en[uint64_t(i)];
        cat[uint64_t(i) + Hl] = hn[uint64_t(i)];
    });
    mg(q, d_cat, eh_proj, s_eh, d_x, 2u * H, H);
}

void MtpHead::build_x_device(sycl::queue& q, const sycl::half* h_src,
                             const int32_t* e_tok) {
    embed_device(q, e_tok);
    q.memcpy(d_h, h_src, uint64_t(H) * sizeof(sycl::half));
    rms_norm_f32w(q, d_e, enorm, d_en, 1, H, eps);
    rms_norm_f32w(q, d_h, hnorm, d_hn, 1, H, eps);
    q.parallel_for(sycl::range<1>(uint64_t(H)),
                   [=, en=d_en, hn=d_hn, cat=d_cat, Hl=H](sycl::id<1> i) {
        cat[uint64_t(i)]      = en[uint64_t(i)];
        cat[uint64_t(i) + Hl] = hn[uint64_t(i)];
    });
    mg(q, d_cat, eh_proj, s_eh, d_x, 2u * H, H);
}

void MtpHead::mg(sycl::queue& q, const sycl::half* in, const sycl::half* f16w,
                 const SoaW& soa, sycl::half* out, uint32_t K, uint32_t N) {
    if (soa.qs && d_act_q8) {
        quantize_q8_1(q, in, d_act_q8, K);
        gemv_q8_0_soa_q8(q, d_act_q8, soa.qs, soa.d, out, K, N);
    } else {
        gemv_fp16(q, in, f16w, out, K, N);
    }
}

void MtpHead::run_layer(sycl::queue& q, int32_t pos) {
    q.memcpy(d_pos1, &pos, sizeof(int32_t));
    rms_norm_f32w(q, d_x, attn_norm, d_xn, 1, H, eps);
    mg(q, d_xn, w_attn_q, s_q, d_qg, H, N_qg);
    split_q_gate_per_head(q, d_qg, d_q, d_gate, 1, n_q_heads, HD);
    mg(q, d_xn, w_attn_k, s_k, d_k, H, N_kv);
    mg(q, d_xn, w_attn_v, s_v, d_v, H, N_kv);
    rms_norm_f32w(q, d_q, attn_q_norm, d_q, n_q_heads,  HD, eps);
    rms_norm_f32w(q, d_k, attn_k_norm, d_k, n_kv_heads, HD, eps);
    rope_partial(q, d_q, d_pos1, d_q, 1, n_q_heads,  HD, rope_n, rope_theta);
    rope_partial(q, d_k, d_pos1, d_k, 1, n_kv_heads, HD, rope_n, rope_theta);
    full_attention(q, d_q, d_k, d_v, mtp_kc, mtp_vc, d_ao,
                   1, uint32_t(pos), n_q_heads, n_kv_heads, HD, max_ctx);
    sigmoid_gate(q, d_ao, d_gate, d_ao, uint64_t(N_q));
    mg(q, d_ao, w_attn_out, s_ao, d_blk, N_q, H);
    residual_add(q, d_x, d_blk, d_x, uint64_t(H));
    rms_norm_f32w(q, d_x, post_attn_norm, d_xn, 1, H, eps);
    mg(q, d_xn, w_ffn_gate, s_fg, d_fg, H, F);
    mg(q, d_xn, w_ffn_up,   s_fu, d_fu, H, F);
    swiglu(q, d_fg, d_fu, d_fh, uint64_t(F));
    mg(q, d_fh, w_ffn_down, s_fd, d_blk, F, H);
    residual_add(q, d_x, d_blk, d_x, uint64_t(H));
    rms_norm_f32w(q, d_x, shead_norm, d_xn, 1, H, eps);
    // The draft head may run over a PREFIX of the vocabulary. The Q8_0-SoA rows
    // are contiguous (qs_W + n*K), so a prefix touches exactly that many bytes.
    // This cannot change the emitted text: a draft is only a guess, and the
    // target's own full-vocab argmax decides every accepted token — a token
    // outside the prefix is simply never guessed, costing acceptance, not
    // correctness. GGUF vocabularies are frequency-ordered (ids 0-255 bytes,
    // ~1k common English, 100k+ CJK, 200k+ other scripts), so a prefix is a
    // principled subset for English/code and a poor one for CJK.
    const uint32_t dv = draft_vocab ? std::min(draft_vocab, vocab) : vocab;
    if (lm_q8_qs && d_act_q8) {
        // Borrowed Q8_0-SoA lm_head: half the head bytes vs the F16 copy — the
        // dominant draft cost. Draft-side only; verify still decides acceptance.
        quantize_q8_1(q, d_xn, d_act_q8, H);
        gemv_q8_0_soa_q8(q, d_act_q8, lm_q8_qs, lm_q8_d, d_logits1, H, dv);
    } else {
        gemv_fp16(q, d_xn, w_lm_head, d_logits1, H, dv);
    }
}

sycl::event MtpHead::argmax_row_device(sycl::queue& q, const sycl::half* row,
                                       uint32_t vocab, int32_t* d_out) {
    return mtp_argmax_device(q, row, vocab, d_out);
}

void MtpHead::draft(sycl::queue& q, const sycl::half* h_last, int32_t tn,
                    uint32_t p_base, uint32_t K, std::vector<int32_t>& out) {
    out.clear();
    int32_t e_tok = tn;
    const sycl::half* h_src = h_last;
    std::vector<sycl::half> row(vocab);
    static const bool tree_probe = std::getenv("IE_SPEC_TREE_PROBE") != nullptr;
    for (uint32_t j = 0; j < K; ++j) {
        build_x(q, h_src, e_tok);
        run_layer(q, int32_t(p_base + j));
        q.wait();
        q.memcpy(row.data(), d_logits1, uint64_t(vocab) * sizeof(sycl::half)).wait();
        int32_t g = mtp_argmax_row(row.data(), vocab);
        if (tree_probe && j == 0) {
            // Measurement probe (tree-spec sizing): the draft's SECOND choice at
            // step 0 — the token a top-2 tree branch would also verify.
            float best2 = -std::numeric_limits<float>::infinity(); int32_t a2 = -1;
            for (uint32_t v = 0; v < vocab; ++v) {
                if (int32_t(v) == g) continue;
                const float x = float(row[v]);
                if (x > best2) { best2 = x; a2 = int32_t(v); }
            }
            top2_last = a2;
        }
        out.push_back(g);
        h_src = d_x;     // d_x overwritten next build_x, but build_x reads h_src
                         // into d_h FIRST (memcpy) before touching d_x — safe.
        e_tok = g;
    }
}

void MtpHead::draft_device_argmax(sycl::queue& q, const sycl::half* h_last,
                                  int32_t tn, uint32_t p_base, uint32_t K,
                                  std::vector<int32_t>& out) {
    out.clear();
    if (K == 0 || K > 16) return;
    q.memcpy(d_tok1, &tn, sizeof(int32_t));
    const sycl::half* h_src = h_last;
    for (uint32_t j = 0; j < K; ++j) {
        const int32_t* e_tok = j == 0 ? d_tok1 : d_draft_ids + (j - 1);
        build_x_device(q, h_src, e_tok);
        run_layer(q, int32_t(p_base + j));
        mtp_argmax_device(q, d_logits1,
                          draft_vocab ? std::min(draft_vocab, vocab) : vocab,
                          d_draft_ids + j);
        h_src = d_x;     // build_x_device copies d_x to d_h before overwriting it
    }
    out.resize(K);
    q.memcpy(out.data(), d_draft_ids, uint64_t(K) * sizeof(int32_t)).wait();
}

std::string Qwen35DenseModel::load_mtp_head(const GgufReader& g, uint32_t max_ctx) {
    if (!alloc_) return "load_mtp_head: model not loaded";
    return mtp_.load(*alloc_, g, cfg_, max_ctx);
}

// ---------------------------------------------------------------------------
// spec_loop_ — shared GREEDY draft→verify→accept→commit→emit loop behind BOTH
// spec paths (MTP self-speculative + dspark separate-draft). Lossless vs plain
// greedy: the target argmax at each verify row decides acceptance, so the
// emitted stream is byte-identical no matter who drafted. `verify_len` = T per
// round (K for MTP, block_size+1 for dspark). Provider seams:
//   draft_fn(tn, p, drafted) — fill the round's draft block (>= verify_len-1)
//   cond_fn(accepted, p)     — update the provider's next-round conditioning
// forward() exports the provider's conditioning source: `hidden_out` (MTP's
// h_last) or `taps_out` (dspark's tap feed); the other is null.
// ---------------------------------------------------------------------------
std::string Qwen35DenseModel::spec_loop_(
        sycl::queue& q, KvCache& kv, DeltaNetState& dn,
        int32_t tn, uint32_t start_pos, uint32_t max_new, uint32_t verify_len,
        sycl::half* hidden_out, sycl::half* taps_out,
        const std::function<std::string(int32_t, uint32_t, std::vector<int32_t>&)>& draft_fn,
        const std::function<std::string(uint32_t, uint32_t)>& cond_fn,
        const SpecEmit& emit, SpecStats* stats) {
    if (verify_len < 2) return "spec_loop: verify_len < 2";
    const uint32_t vocab = cfg_.dense.vocab;
    const uint32_t L_full = cfg_.n_transformer_layers() / cfg_.full_attn_interval;
    const uint32_t V = verify_len;

    if (auto e = ensure_workspace(V); !e.empty()) return "spec ws: " + e;

    Qwen35SpecCheckpoint ckpt;
    if (auto e = ckpt.init(*alloc_, dn, V); !e.empty()) return "spec ckpt: " + e;

    auto* d_all    = sycl::malloc_device<sycl::half>(uint64_t(V) * vocab, q);
    auto* d_logits = sycl::malloc_device<sycl::half>(vocab, q);
    auto* d_ids    = sycl::malloc_device<int32_t>(V, q);
    if (!d_all || !d_logits || !d_ids) {
        if (d_all) sycl::free(d_all, q);
        if (d_logits) sycl::free(d_logits, q);
        if (d_ids) sycl::free(d_ids, q);
        return "spec_loop: scratch alloc failed";
    }
    std::vector<sycl::half> Lrow(uint64_t(V) * vocab);

    std::string ret;
    uint32_t emitted = 0;
    uint32_t p = start_pos;            // committed prefix length (= next write pos)
    bool first_round = true;
    bool abort = false;

    while (emitted < max_new && !abort) {
        // 1. DRAFT (provider-specific: MTP head or dspark drafter).
        std::vector<int32_t> drafted;
        if (auto e = draft_fn(tn, p, drafted); !e.empty()) { ret = e; break; }
        if (drafted.size() + 1 < V) { ret = "spec_loop: draft too short"; break; }

        // 2. VERIFY: target forward(T=V, start_pos=p) in CHECKPOINT MODE. Exports
        //    the provider's conditioning source (hidden_out or taps_out).
        std::vector<uint32_t> kv_len_snap(L_full);
        for (uint32_t l = 0; l < L_full; ++l) kv_len_snap[l] = kv.length(l);

        std::vector<int32_t> vin(V);
        vin[0] = tn;
        for (uint32_t j = 1; j < V; ++j) vin[j] = drafted[j - 1];
        q.memcpy(d_ids, vin.data(), V * sizeof(int32_t)).wait();
        forward(q, d_ids, V, p, kv, dn, d_logits,
                /*all_logits=*/d_all, /*hidden_pre_norm=*/hidden_out,
                /*ckpt=*/&ckpt, /*taps_out=*/taps_out).wait();
        q.memcpy(Lrow.data(), d_all, uint64_t(V) * vocab * sizeof(sycl::half)).wait();

        std::vector<int32_t> targ(V);
        for (uint32_t j = 0; j < V; ++j)
            targ[j] = mtp_argmax_row(Lrow.data() + uint64_t(j) * vocab, vocab);

        uint32_t n = 0;   // drafted tokens accepted (g_1..g_n)
        for (uint32_t j = 1; j < V; ++j) {
            if (drafted[j - 1] == targ[j - 1]) ++n; else break;
        }
        const int32_t bonus = targ[n];
        const uint32_t accepted = n + 1;   // input rows kept (tn + g_1..g_n)

        if (stats) {
            stats->rounds  += 1;
            stats->drafted += (V - 1);
            stats->accepted += n;
            stats->bonus   += 1;
        }

        // 3. COMMIT state to p+accepted — NO re-forward (per-position ckpt).
        if (accepted < V) {
            if (auto e = ckpt.commit_to_n(q, dn, accepted); !e.empty()) {
                ret = "spec commit: " + e; abort = true;
            }
            for (uint32_t l = 0; l < L_full; ++l)
                kv.set_length(l, kv_len_snap[l] + accepted);
        }
        // else: all V verify rows accepted → dn/kv already at p+V = p+accepted.

        // 4. EMIT committed tokens IN ORDER. The very first round must also emit
        //    tn (it came from prompt prefill); every later round's tn was already
        //    emitted as the previous round's bonus.
        auto do_emit = [&](int32_t id) -> bool {
            if (emitted >= max_new) return false;
            ++emitted;
            if (!emit(id)) { abort = true; return false; }
            return true;
        };
        if (!abort && first_round) { if (!do_emit(tn)) {} }
        for (uint32_t j = 0; j < n && !abort && emitted < max_new; ++j) do_emit(drafted[j]);
        if (!abort && emitted < max_new) do_emit(bonus);

        // 5. next round: tn' = bonus; provider refreshes conditioning from the
        //    accepted prefix (rejected tail discarded → rewind-correct), then
        //    advance the committed position.
        tn = bonus;
        if (auto e = cond_fn(accepted, p); !e.empty()) { ret = e; abort = true; }
        p += accepted;
        first_round = false;
    }

    sycl::free(d_all, q);
    sycl::free(d_logits, q);
    sycl::free(d_ids, q);
    if (stats && stats->rounds)
        stats->tau = double(stats->accepted + stats->bonus) / double(stats->rounds);
    return ret;
}

// ---------------------------------------------------------------------------
// spec_generate — MTP self-speculative GREEDY decode (lifted from the validated
// tools/ie_qwen35_spec.cpp spec_greedy lambda). Lossless vs plain greedy. The
// caller has prefilled the prompt into kv/dn and supplies (h_last, tn) for the
// last prompt position; we emit committed tokens IN ORDER through `emit`.
// ---------------------------------------------------------------------------
std::string Qwen35DenseModel::spec_generate(sycl::queue& q, KvCache& kv,
                                            DeltaNetState& dn,
                                            sycl::half* h_last, int32_t tn,
                                            uint32_t start_pos, uint32_t max_new,
                                            uint32_t K, const SpecEmit& emit) {
    if (!mtp_.loaded) return "spec_generate: MTP head not loaded";
    if (K == 0) return "spec_generate: K == 0";
    const uint32_t H = cfg_.dense.hidden;

    // MTP draft window (verify T=K over [tn, g_1..g_{K-1}]); h_last conditioning.
    auto* d_hid = sycl::malloc_device<sycl::half>(uint64_t(K) * H, q);
    if (!d_hid) return "spec_generate: hidden alloc failed";

    auto draft_fn = [&](int32_t tn_r, uint32_t /*p*/, std::vector<int32_t>& drafted)
                        -> std::string {
        mtp_.draft(q, h_last, tn_r, /*p_base=*/0, K, drafted);
        return {};
    };
    auto cond_fn = [&](uint32_t accepted, uint32_t /*p*/) -> std::string {
        q.memcpy(h_last, d_hid + uint64_t(accepted - 1) * H,
                 uint64_t(H) * sizeof(sycl::half)).wait();
        return {};
    };

    std::string ret = spec_loop_(q, kv, dn, tn, start_pos, max_new, /*verify_len=*/K,
                                 /*hidden_out=*/d_hid, /*taps_out=*/nullptr,
                                 draft_fn, cond_fn, emit);
    sycl::free(d_hid, q);
    return ret;
}

// ---------------------------------------------------------------------------
// spec_generate_dspark — separate-draft GREEDY spec decode driven by the
// target-conditioned DsparkDrafter (M2). Reuses spec_loop_'s lossless verify/
// accept/commit/emit; the only differences are the draft source, the verify
// length (block_size+1), and the conditioning (cumulative target taps instead
// of h_last). The M1 drafter recomputes its whole context each round, so `feat`
// carries the taps of EVERY committed position (0..p-1) and grows in place from
// the accepted prefix — a persistent drafter KV is the M3 perf lever.
// ---------------------------------------------------------------------------
std::string Qwen35DenseModel::spec_generate_dspark(
        sycl::queue& q, KvCache& kv, DeltaNetState& dn,
        DsparkDrafter& drafter,
        std::vector<float>& feat, std::vector<int32_t>& pos,
        int32_t tn, uint32_t start_pos, uint32_t max_new, const SpecEmit& emit,
        SpecStats* stats) {
    if (!drafter.loaded) return "spec_generate_dspark: drafter not loaded";
    const uint32_t bs   = drafter.block_size;
    if (bs == 0) return "spec_generate_dspark: block_size == 0";
    const uint32_t nc   = drafter.n_capture;
    const uint32_t ncap = drafter.n_embd_cap;     // nc * H
    const uint32_t H    = cfg_.dense.hidden;
    const uint32_t V    = bs + 1;                  // verify length
    if (ncap != nc * H) return "spec_generate_dspark: n_embd_cap != n_capture*hidden";

    // Verify-forward tap export target: [n_capture, V, H] fp16 (capture-outer).
    auto* d_taps = sycl::malloc_device<sycl::half>(uint64_t(nc) * V * H, q);
    if (!d_taps) return "spec_generate_dspark: taps alloc failed";
    std::vector<sycl::half> taps_host(uint64_t(nc) * V * H);

    // per-round drafter I/O scratch (resized inside draft_block).
    std::vector<float>   out_hidden, out_logits;
    std::vector<int32_t> out_ids;

    auto draft_fn = [&](int32_t tn_r, uint32_t p, std::vector<int32_t>& drafted)
                        -> std::string {
        if (pos.size() != size_t(p))
            return "dspark draft: conditioning rows != p (tap bookkeeping desync)";
        std::vector<int32_t> dtok(bs), dpos(bs);
        dtok[0] = tn_r;                                       // anchor
        for (uint32_t k = 1; k < bs; ++k) dtok[k] = drafter.mask_token_id;
        for (uint32_t k = 0; k < bs; ++k) dpos[k] = int32_t(p + k);
        if (auto e = drafter.draft_block(feat.data(), pos.data(), int32_t(p),
                                         dtok.data(), dpos.data(),
                                         out_hidden, out_logits, out_ids); !e.empty())
            return "dspark draft: " + e;
        drafted = out_ids;                                   // bs draft ids
        return {};
    };

    auto cond_fn = [&](uint32_t accepted, uint32_t p) -> std::string {
        // Append the ACCEPTED verify rows' taps (positions p..p+accepted-1) to the
        // cumulative conditioning; the rejected tail is never read → rewind-correct
        // (only committed positions ever condition the next round). Device taps are
        // [n_capture, V, H]; reorder into feat's per-row [n_capture*H] and F16→F32.
        q.memcpy(taps_host.data(), d_taps,
                 uint64_t(nc) * V * H * sizeof(sycl::half)).wait();
        const size_t base = pos.size();
        feat.resize((base + accepted) * size_t(ncap));
        pos.resize(base + accepted);
        for (uint32_t r = 0; r < accepted; ++r) {
            float* dst = feat.data() + (base + r) * size_t(ncap);
            for (uint32_t c = 0; c < nc; ++c) {
                const sycl::half* src = taps_host.data() + (uint64_t(c) * V + r) * H;
                float* dcol = dst + size_t(c) * H;
                for (uint32_t h = 0; h < H; ++h) dcol[h] = float(src[h]);
            }
            pos[base + r] = int32_t(p + r);
        }
        return {};
    };

    std::string ret = spec_loop_(q, kv, dn, tn, start_pos, max_new, /*verify_len=*/V,
                                 /*hidden_out=*/nullptr, /*taps_out=*/d_taps,
                                 draft_fn, cond_fn, emit, stats);
    sycl::free(d_taps, q);
    return ret;
}

// P3d Task 3B: hybrid forward — full-attention + dense-MLP path. The linear
// (gated-DeltaNet) layers are STUBBED to a zero attention contribution here
// (clearly marked); the real DeltaNet recurrence lands in Task 3C, validated
// tensor-by-tensor against the oracle (docs/qwen35_27b_oracle_dataflow.md).
//
// Verification checkpoints for Task 4 (flagged, not yet oracle-confirmed):
//   * NORM convention — uses rms_norm_f32w (standard). If the oracle shows a
//     (1+w) offset, switch the qwen35 norms to rms_norm_one_plus_w.
//   * PARTIAL RoPE — n_rotary = dense.rope_dim (expect 64 = 0.25·head_dim).
//   * sigmoid_gate direction (attn_out · σ(gate)) and the joint Q|gate split.
sycl::event Qwen35DenseModel::forward(sycl::queue& q,
                                      const int32_t* input_ids, uint32_t T,
                                      uint32_t start_pos,
                                      KvCache& kv, DeltaNetState& dn,
                                      sycl::half* out_logits,
                                      sycl::half* all_logits,
                                      sycl::half* hidden_pre_norm,
                                      Qwen35SpecCheckpoint* ckpt,
                                      sycl::half* taps_out) {
    if (T == 0) return {};
    // Checkpoint mode requires per-position snapshots sized for ≥ T positions.
    const bool ckpt_mode = (ckpt != nullptr && ckpt->K >= T &&
                            ckpt->ckpt_state && ckpt->ckpt_conv);
    if (ws_T_ < T) {
        auto e = ensure_workspace(T);
        if (!e.empty()) {
            std::fprintf(stderr, "Qwen35DenseModel::ensure_workspace: %s\n", e.c_str());
            return {};
        }
    }

    const DenseConfig& dc = cfg_.dense;
    const uint32_t H    = dc.hidden;                    // 5120
    const uint32_t HD   = dc.head_dim;                  // 256
    const uint32_t N_q  = dc.n_q_heads  * HD;           // 6144
    const uint32_t N_qg = N_q * 2u;                     // 12288 (joint Q|gate)
    const uint32_t N_kv = dc.n_kv_heads * HD;           // 1024
    const uint32_t F    = dc.ffn;                       // 17408
    const uint32_t rope_n = dc.rope_dim;                // 64 (partial) — Task-4 check
    const float    eps  = dc.rms_eps;
    const uint32_t interval = cfg_.full_attn_interval;  // 4
    const uint32_t n_layers = cfg_.n_transformer_layers();
    const uint64_t per_layer_kv = uint64_t(dc.n_kv_heads) * kv.config().max_ctx * HD;

    // positions [start_pos .. start_pos+T-1]
    if (T == 1) {
        q.fill(ws_positions_, int32_t(start_pos), 1);
    } else {
        std::vector<int32_t> pos(T);
        for (uint32_t t = 0; t < T; ++t) pos[t] = int32_t(start_pos + t);
        q.memcpy(ws_positions_, pos.data(), T * sizeof(int32_t)).wait();
    }

    // embedding → ws_x_
    if (token_embd_dtype_ == DType::kQ4_K)
        embedding_lookup_q4k(q, input_ids, token_embd_, ws_x_, T, H);
    else if (token_embd_dtype_ == DType::kQ8_0)
        embedding_lookup_q8_0(q, input_ids, token_embd_, ws_x_, T, H);
    else if (token_embd_dtype_ == DType::kQ2_0)
        embedding_lookup_q2_0(q, input_ids, token_embd_, ws_x_, T, H);
    else if (token_embd_dtype_ == DType::kQ1_0)
        embedding_lookup_q1_0(q, input_ids, token_embd_, ws_x_, T, H);
    else
        embedding_lookup_q6k(q, input_ids, token_embd_, ws_x_, T, H);

    // Optional per-layer residual dump (Task 4 oracle diff). Same naming as the
    // crown/dense path so tools/diff_layers.sh works unchanged.
    auto dump_residual = [&](int idx) {
        if (dump_prefix_.empty()) return;
        const uint64_t n = uint64_t(T) * H;
        std::vector<sycl::half> hh(n);
        q.memcpy(hh.data(), ws_x_, n * sizeof(sycl::half)).wait();
        std::vector<float> ff(n);
        for (size_t i = 0; i < n; ++i) ff[i] = float(hh[i]);
        char path[1024];
        std::snprintf(path, sizeof(path), "%s_L%02d.bin", dump_prefix_.c_str(), idx);
        if (FILE* fp = std::fopen(path, "wb")) { std::fwrite(ff.data(), 4, n, fp); std::fclose(fp); }
        std::snprintf(path, sizeof(path), "%s_L%02d.meta", dump_prefix_.c_str(), idx);
        if (FILE* fp = std::fopen(path, "w"))  { std::fprintf(fp, "%u %u\n", T, H); std::fclose(fp); }
    };
    dump_residual(0);

    for (uint32_t L = 0; L < n_layers; ++L) {
        const auto& w = layers_[L];

        // pre-attn norm. Quant-hoist (decode T==1): fused norm+Q8_1 emission —
        // ws_attn_norm_q8_ holds the bit-identical Q8_1 stream the SoA-GEMV
        // consumers (attn_qkv/attn_gate or attn_q/k/v) would otherwise each
        // re-quantize. OFF path is byte-identical (plain rms_norm_f32w).
        const void* attn_norm_q8 = nullptr;
        if (quant_hoist_on_ && T == 1) {
            rms_norm_f32w_q8(q, ws_x_, w.attn_norm, ws_x_normed_,
                             ws_attn_norm_q8_, T, H, eps);
            attn_norm_q8 = ws_attn_norm_q8_;
        } else {
            rms_norm_f32w(q, ws_x_, w.attn_norm, ws_x_normed_, T, H, eps);
        }

        if (w.is_linear) {
            // ---- gated-DeltaNet linear layer — UNFUSED chain (landmines R1/R2/R4) ----
            const uint32_t dn_idx  = L - L / interval;          // 0,1,2,3,... linear index
            const uint32_t SKH = cfg_.ssm_n_k_heads;            // 16
            const uint32_t SVH = cfg_.ssm_n_v_heads;            // 48
            const uint32_t SHD = cfg_.ssm_state;                // 128 (head dim)
            const uint32_t SI  = cfg_.ssm_inner;                // 6144 (d_inner)
            const uint32_t conv_ch = SI + 2u * SKH * SHD;       // 10240 (R1: not SI*2)
            const uint32_t kw  = SKH * SHD;                     // 2048 (q/k pre-repeat width)
            const uint32_t rep = SVH / SKH;                     // 3 (16→48)
            const float qscale = 1.0f / sycl::sqrt(float(SHD)); // q L2 scale

            // attn_qkv projection → conv input [conv_ch]
            if (w.attn_qkv_q2soa.active())
                gemv_q2soa_T(q, ws_x_normed_, w.attn_qkv_q2soa, ws_qkv_, T, attn_norm_q8);
            else if (w.attn_qkv_soa.active())
                gemv_q6soa_T(q, ws_x_normed_, w.attn_qkv_soa, ws_qkv_, T, attn_norm_q8);
            else if (w.attn_qkv_q4soa.active())
                gemv_q4soa_T(q, ws_x_normed_, w.attn_qkv_q4soa, ws_qkv_, T, attn_norm_q8);
            else
                dense::gemv_q_T(q, ws_x_normed_, w.attn_qkv, ws_qkv_, H, conv_ch, T);
            // causal depthwise conv1d (fuses SiLU) → ws_conv_
            sycl::half* conv_state = dn.conv_state_ptr() +
                                     uint64_t(dn_idx) * dn.conv_elems_per_layer();
            if (ckpt_mode) {
                // Decompose into T single-token streaming steps, snapshotting the
                // conv state AFTER each position. Output ws_conv_ ends byte-
                // identical to the T-batched call (conv is causal + streaming).
                const uint64_t ce = dn.conv_elems_per_layer();
                for (uint32_t s = 0; s < T; ++s) {
                    depthwise_conv1d_causal(q, ws_qkv_ + uint64_t(s) * conv_ch,
                                            w.ssm_conv1d_fp16, conv_state,
                                            ws_conv_ + uint64_t(s) * conv_ch,
                                            /*T=*/1, conv_ch, cfg_.ssm_conv_kernel);
                    q.memcpy(ckpt->ckpt_conv +
                                 (uint64_t(s) * ckpt->n_lin + dn_idx) * ce,
                             conv_state, ce * sizeof(sycl::half));
                }
            } else {
                depthwise_conv1d_causal(q, ws_qkv_, w.ssm_conv1d_fp16, conv_state, ws_conv_,
                                        T, conv_ch, cfg_.ssm_conv_kernel);
            }
            // split (q|k|v) + cast→fp32: q,k width kw=2048; v width SI=6144 (no l2norm)
            cast_qkv_split_fp16_to_fp32(q, ws_conv_, ws_qpre_, ws_kpre_, ws_vpre_, T, kw, SI);
            // per-head L2-norm over 16 heads × 128 (q with qscale, k with 1.0)
            l2_norm_scale(q, ws_qpre_, ws_qpre_, T * SKH, SHD, qscale, 1e-6f);
            l2_norm_scale(q, ws_kpre_, ws_kpre_, T * SKH, SHD, 1.0f,   1e-6f);
            // repeat q,k heads 16→48 (block-tile, R4)
            repeat_interleave_heads(q, ws_qpre_, ws_qrep_, T, SKH, SHD, rep);
            repeat_interleave_heads(q, ws_kpre_, ws_krep_, T, SKH, SHD, rep);
            // g = -exp(ssm_a)·softplus(alpha+dt_bias);  beta = sigmoid(beta_proj).
            // Projections are N-padded to 64 → take the BATCHED gemm (one launch)
            // instead of a per-token serial gemv loop; compact [T,64]→[T,SVH].
            const uint32_t SVHp = ((SVH + 63u) / 64u) * 64u;   // 64
            dense::gemv_q_T(q, ws_x_normed_, w.ssm_alpha, ws_alpha64_, H, SVHp, T);
            dense::gemv_q_T(q, ws_x_normed_, w.ssm_beta,  ws_beta64_,  H, SVHp, T);
            extract_cols(q, ws_alpha64_, ws_alpha_h_, T, SVH, SVHp);
            extract_cols(q, ws_beta64_,  ws_beta_h_,  T, SVH, SVHp);
            compute_g_beta_h16(q, ws_alpha_h_, ws_beta_h_, w.ssm_a, w.ssm_dt_bias,
                               ws_g_, ws_beta_, T, SVH);
            // gated delta-rule recurrence (state carried per linear layer)
            float* state_layer = dn.state_ptr() +
                                 uint64_t(dn_idx) * dn.state_elems_per_layer();
            if (ckpt_mode) {
                // T single-token recurrence steps; snapshot recurrent state AFTER
                // each position. K×(T=1) == 1×(T=K) (sequential scan over `state`).
                const uint64_t se = dn.state_elems_per_layer();
                const uint64_t qkv_stride = uint64_t(SVH) * SHD;   // per-position
                const uint64_t gb_stride  = uint64_t(SVH);
                for (uint32_t s = 0; s < T; ++s) {
                    deltanet_recurrence(q,
                        ws_qrep_ + s * qkv_stride, ws_krep_ + s * qkv_stride,
                        ws_vpre_ + s * qkv_stride, ws_g_ + s * gb_stride,
                        ws_beta_ + s * gb_stride, state_layer,
                        ws_dn_out_ + s * qkv_stride, /*B=*/1, /*T=*/1,
                        SVH, SHD, SHD);
                    q.memcpy(ckpt->ckpt_state +
                                 (uint64_t(s) * ckpt->n_lin + dn_idx) * se,
                             state_layer, se * sizeof(float));
                }
            } else {
                deltanet_recurrence(q, ws_qrep_, ws_krep_, ws_vpre_, ws_g_, ws_beta_,
                                    state_layer, ws_dn_out_, /*B=*/1, T, SVH, SHD, SHD);
            }
            // gated RMS-norm with z = attn_gate · x_normed (reuse ws_qkv_ as out)
            if (w.attn_gate_q2soa.active())
                gemv_q2soa_T(q, ws_x_normed_, w.attn_gate_q2soa, ws_dn_z_, T, attn_norm_q8);
            else if (w.attn_gate_soa.active())
                gemv_q6soa_T(q, ws_x_normed_, w.attn_gate_soa, ws_dn_z_, T, attn_norm_q8);
            else if (w.attn_gate_q4soa.active())
                gemv_q4soa_T(q, ws_x_normed_, w.attn_gate_q4soa, ws_dn_z_, T, attn_norm_q8);
            else
                dense::gemv_q_T(q, ws_x_normed_, w.attn_gate, ws_dn_z_, H, SI, T);
            gated_rms_norm(q, ws_dn_out_, ws_dn_z_, w.ssm_norm_fp16, ws_qkv_,
                           T * SVH, SHD, eps);
            // output projection → ws_attn_block_ (the linear attn contribution)
            if (w.ssm_out_q2soa.active())
                gemv_q2soa_T(q, ws_qkv_, w.ssm_out_q2soa, ws_attn_block_, T);
            else if (w.ssm_out_q8.active())
                gemv_q8soa_T(q, ws_qkv_, w.ssm_out_q8, ws_attn_block_, T);
            else if (w.ssm_out_soa.active())
                gemv_q6soa_T(q, ws_qkv_, w.ssm_out_soa, ws_attn_block_, T);
            else if (w.ssm_out_q4soa.active())
                gemv_q4soa_T(q, ws_qkv_, w.ssm_out_q4soa, ws_attn_block_, T);
            else
                dense::gemv_q_T(q, ws_qkv_, w.ssm_out, ws_attn_block_, SI, H, T);
        } else {
            // ---- gated full-attention ----
            const uint32_t full_idx = L / interval;
            // Q2 multi-bank fusion (IE_Q2_MULTI, decode): attn_q|gate + attn_k +
            // attn_v share the pre-attn-norm activation and have identical K=H →
            // one fused GEMV launch (k/v are the starved small-N banks).
            // Bit-identical: each output column is an independent reduction.
            static const bool q2_multi = std::getenv("IE_Q2_MULTI") != nullptr;
            bool fa_fused = false;
            if (q2_multi && T == 1 && w.attn_q_q2soa.active() &&
                w.attn_k_q2soa.active() && w.attn_v_q2soa.active() &&
                w.attn_q_q2soa.dtype == DType::kQ2_0 &&
                w.attn_k_q2soa.dtype == DType::kQ2_0 &&
                w.attn_v_q2soa.dtype == DType::kQ2_0) {   // Q1-native → per-projection
                const void* fin = attn_norm_q8;
                sycl::event qe;
                std::vector<sycl::event> d;
                if (!fin) { qe = quantize_q8_1(q, ws_x_normed_, ws_attn_norm_q8_, H); fin = ws_attn_norm_q8_; d = {qe}; }
                const uint8_t*  qs[3] = {w.attn_q_q2soa.qs, w.attn_k_q2soa.qs, w.attn_v_q2soa.qs};
                const uint16_t* dd[3] = {w.attn_q_q2soa.d,  w.attn_k_q2soa.d,  w.attn_v_q2soa.d};
                sycl::half*     yy[3] = {ws_qg_, ws_k_, ws_v_};
                const uint32_t  NN[3] = {w.attn_q_q2soa.N, w.attn_k_q2soa.N, w.attn_v_q2soa.N};
                gemv_q2_0_soa_q8_multi(q, fin, qs, dd, yy, NN, 3, w.attn_q_q2soa.K, d);
                fa_fused = true;
            }
            // joint Q|gate projection → split per head into Q and the σ-gate
            if (fa_fused) { /* q|gate already computed */ }
            else if (w.attn_q_q2soa.active())
                gemv_q2soa_T(q, ws_x_normed_, w.attn_q_q2soa, ws_qg_, T, attn_norm_q8);
            else if (w.attn_q_soa.active())
                gemv_q6soa_T(q, ws_x_normed_, w.attn_q_soa, ws_qg_, T, attn_norm_q8);
            else if (w.attn_q_q4soa.active())
                gemv_q4soa_T(q, ws_x_normed_, w.attn_q_q4soa, ws_qg_, T, attn_norm_q8);
            else
                dense::gemv_q_T(q, ws_x_normed_, w.attn_q, ws_qg_, H, N_qg, T);
            split_q_gate_per_head(q, ws_qg_, ws_q_, ws_gate_, T, dc.n_q_heads, HD);
            // K, V (attn_k Q5_K → SoA-Q8 fast lane when requantized; Q2_0 → SoA-Q2)
            if (fa_fused) { /* k already computed */ }
            else if (w.attn_k_q2soa.active())
                gemv_q2soa_T(q, ws_x_normed_, w.attn_k_q2soa, ws_k_, T, attn_norm_q8);
            else if (w.attn_k_q8.active())
                gemv_q8soa_T(q, ws_x_normed_, w.attn_k_q8, ws_k_, T, attn_norm_q8);
            else
                dense::gemv_q_T(q, ws_x_normed_, w.attn_k, ws_k_, H, N_kv, T);
            if (fa_fused) { /* v already computed */ }
            else if (w.attn_v_q2soa.active())
                gemv_q2soa_T(q, ws_x_normed_, w.attn_v_q2soa, ws_v_, T, attn_norm_q8);
            else if (w.attn_v_q4soa.active())
                gemv_q4soa_T(q, ws_x_normed_, w.attn_v_q4soa, ws_v_, T, attn_norm_q8);
            else
                dense::gemv_q_T(q, ws_x_normed_, w.attn_v, ws_v_, H, N_kv, T);
            // per-head Q/K RMS-norm
            rms_norm_f32w(q, ws_q_, w.attn_q_norm, ws_q_, T * dc.n_q_heads,  HD, eps);
            rms_norm_f32w(q, ws_k_, w.attn_k_norm, ws_k_, T * dc.n_kv_heads, HD, eps);
            // partial RoPE (n_rot 64)
            rope_partial(q, ws_q_, ws_positions_, ws_q_, T, dc.n_q_heads,  HD, rope_n, dc.rope_theta);
            rope_partial(q, ws_k_, ws_positions_, ws_k_, T, dc.n_kv_heads, HD, rope_n, dc.rope_theta);
            // SDPA (KV slice index = full_idx; only full layers are cached)
            sycl::half* kc = kv.k_ptr() + per_layer_kv * full_idx;
            sycl::half* vc = kv.v_ptr() + per_layer_kv * full_idx;
            if (T == 1 && ws_attn_partials_) {
                // DECODE (T==1). INT8-KV path (--int8-kv): halves the long-ctx KV
                // read (16 full layers × 4 kv_h × 256 hd fp16 = 4.3 GB/token @16K)
                // when the int8 shadow is populated up to start_pos on this
                // full-attn slot; else fp16 FA-2. Mirrors the validated
                // qwen3moe.cpp / qwen3next.cpp wiring (int8 kernel inline-quantizes
                // this position's K/V; fp16 shadow skipped — nullptr — same as the
                // siblings). Gated by kv.is_int8() (--int8-kv) → default fp16 path
                // byte-identical.
                if (kv.is_int8() && kv.k_int8_ptr() &&
                    start_pos == kv.int8_length(full_idx)) {
                    const uint64_t i8pl = uint64_t(dc.n_kv_heads) * kv.config().max_ctx * HD;
                    const uint64_t scpl = uint64_t(dc.n_kv_heads) * kv.config().max_ctx;
                    full_attention_fa2_decode_int8(
                        q, ws_q_, ws_k_, ws_v_,
                        kv.k_int8_ptr()   + i8pl * full_idx,
                        kv.v_int8_ptr()   + i8pl * full_idx,
                        kv.k_scales_ptr() + scpl * full_idx,
                        kv.v_scales_ptr() + scpl * full_idx,
                        nullptr, nullptr,
                        ws_attn_out_, ws_attn_partials_,
                        start_pos, dc.n_q_heads, dc.n_kv_heads,
                        HD, kv.config().max_ctx);
                    kv.set_int8_length(full_idx, start_pos + 1);   // inline-quantized this pos
                } else {
                    // IE_Q35_FA2_VEC: llama fattn-vec decode port (lane = D-slice,
                    // narrow width-8 reduces OFF the serial softmax chain — the
                    // structural long-ctx fix, attention.cpp:1263). Same buffers +
                    // partials format as v1 → drop-in; q_vals[] bumped to [64] for
                    // hd256. DEFAULT ON since the 2026-07-18 A/B: long-ctx 16K
                    // decode 24.78 vs 16.83 tok/s (+47%), short-ctx no regression
                    // (33.9/33.6 vs 33.7/33.4), Bonsai PPL 13.0942 vs 13.0767
                    // (held). IE_Q35_FA2_VEC=0 opts out (kill switch → v1). No
                    // int8-vec kernel exists yet, so int8 (above) takes precedence.
                    static const bool dec_vec = []{
                        const char* e = std::getenv("IE_Q35_FA2_VEC");
                        return !e || std::atoi(e) != 0;
                    }();
                    if (dec_vec)
                        full_attention_fa2_decode_vec(q, ws_q_, ws_k_, ws_v_, kc, vc,
                                                      ws_attn_out_, ws_attn_partials_,
                                                      start_pos, dc.n_q_heads, dc.n_kv_heads,
                                                      HD, kv.config().max_ctx);
                    else
                        full_attention_fa2_decode(q, ws_q_, ws_k_, ws_v_, kc, vc,
                                                  ws_attn_out_, ws_attn_partials_,
                                                  start_pos, dc.n_q_heads, dc.n_kv_heads,
                                                  HD, kv.config().max_ctx);
                }
            } else if (fa2_prefill_on_ && ws_attn_partials_ && T <= kBatchedVerifyMax) {
                // Spec-decode VERIFY (small T, long KV): the tiled FA-2 *prefill*
                // kernel starves the GPU at T=4 (one Br-chunk, serial KV tiles —
                // measured 4× SLOWER than naive). The naive path instead reads the
                // whole KV cache T× (its cost). Best of both: LOOP the KV-stationary
                // split-K fa2_DECODE over the T verify positions — each appends its
                // K/V at start_pos+t and attends [0,start_pos+t] causally, reading
                // KV once per position. Bit-identical to running T sequential T==1
                // decode steps → spec verify==decode LOSSLESS by construction.
                for (uint32_t t = 0; t < T; ++t)
                    full_attention_fa2_decode(
                        q, ws_q_ + uint64_t(t) * N_q, ws_k_ + uint64_t(t) * N_kv,
                        ws_v_ + uint64_t(t) * N_kv, kc, vc,
                        ws_attn_out_ + uint64_t(t) * N_q, ws_attn_partials_,
                        start_pos + t, dc.n_q_heads, dc.n_kv_heads, HD,
                        kv.config().max_ctx);
            } else {
                // Long-ctx full-attn prefill. Naive full_attention re-reads the
                // whole KV cache T× → collapses at 16K (137 s, pp 119 vs llama 231).
                // Route the head_dim-256 full-attn layers through the proven Gemma
                // wide-tile kernel (reads KV once per Br query-tile, appends k/v
                // internally, full causal window=0). Numerically equivalent to naive
                // at hd256 (argmax-bit-identical on Gemma; same 1/sqrt(HD) post-dot
                // scale on the same unscaled Q here). GATED to ctx ≥ minctx (default
                // 6144) so the verified ≤4K naive WIN (pp4096 577 = 2.07× vs llama)
                // is untouched — only the collapsing 8K–16K region switches. Opt-out
                // IE_QWEN35_NO_FA2_TILE; tune IE_QWEN35_FA2_TILE_MINCTX. [2026-06-26]
                static const bool no_tile = std::getenv("IE_QWEN35_NO_FA2_TILE") != nullptr;
                static const uint32_t tile_minctx = []() -> uint32_t {
                    const char* e = std::getenv("IE_QWEN35_FA2_TILE_MINCTX");
                    if (!e) return 6144u;
                    int v = std::atoi(e); return v > 0 ? uint32_t(v) : 6144u;
                }();
                if (!no_tile && HD == 256 && (start_pos + T) >= tile_minctx) {
                    full_attention_fa2_prefill_tile_gemma(
                        q, ws_q_, ws_k_, ws_v_, kc, vc, ws_attn_out_, T, start_pos,
                        dc.n_q_heads, dc.n_kv_heads, HD, kv.config().max_ctx,
                        0 /*window: full causal*/);
                } else {
                    full_attention(q, ws_q_, ws_k_, ws_v_, kc, vc, ws_attn_out_,
                                   T, start_pos, dc.n_q_heads, dc.n_kv_heads,
                                   HD, kv.config().max_ctx);
                }
            }
            // INT8-KV: post-quantize the fp16 rows appended by the T>1 branches
            // (spec-verify loop + BOTH prefill leaves) into the int8 shadow so
            // the next T==1 decode takes the int8 path. Sets int8_lengths_[full_idx]
            // host-side (in-order queue). Mirrors qwen3next.cpp. No-op unless
            // --int8-kv.
            if (T > 1 && kv.is_int8())
                kv.quantize_to_int8(q, full_idx, start_pos, T);
            // post-attention sigmoid gate: attn_out · σ(gate)
            sigmoid_gate(q, ws_attn_out_, ws_gate_, ws_attn_out_, uint64_t(T) * N_q);
            // output projection → ws_attn_block_
            if (w.attn_output_q2soa.active())
                gemv_q2soa_T(q, ws_attn_out_, w.attn_output_q2soa, ws_attn_block_, T);
            else if (w.attn_output_q8.active())
                gemv_q8soa_T(q, ws_attn_out_, w.attn_output_q8, ws_attn_block_, T);
            else
                dense::gemv_q_T(q, ws_attn_out_, w.attn_output, ws_attn_block_, N_q, H, T);
        }

        // residual + pre-FFN (post-attention) norm — FFN adds to the
        // pre-post-norm tensor (qwen35 residual order, landmine). This fused op
        // also writes ws_x_ (the residual), so it can't be the rms_norm_f32w_q8
        // variant; instead, quant-hoist (decode T==1) runs ONE explicit
        // quantize_q8_1 of the normed result and feeds it to ffn_gate + ffn_up
        // (which would otherwise each re-quantize the SAME vector). Net: −1 quant
        // launch. Bit-identical (same rounded-fp16 normed inputs).
        residual_add_rms_norm_fused(q, ws_x_, ws_attn_block_, w.post_attn_norm,
                                    ws_x_normed_, T, H, eps);
        // Reordered Q4_K decode (IE_QWEN35_Q4K_REORDER): the FFN gate+up share the
        // hoisted ffn_in_q8 → route to gemv_q4_K_reorder_q8 (llama's pure-contiguous
        // nibble layout = the BW win). Force the hoist so the q8 exists.
        static const bool q4k_reorder = std::getenv("IE_QWEN35_Q4K_REORDER") != nullptr;
        const void* ffn_in_q8 = nullptr;
        if ((quant_hoist_on_ || q4k_reorder) && T == 1) {
            quantize_q8_1(q, ws_x_normed_, ws_ffn_in_q8_, H);
            ffn_in_q8 = ws_ffn_in_q8_;
        }
        const bool ffn_ro = q4k_reorder && T == 1 && ffn_in_q8;
        // Q2 multi-bank fusion (IE_Q2_MULTI, decode): ffn_gate + ffn_up share the
        // FFN-input activation and have identical K=H → one quantize_q8_1 + one
        // fused GEMV launch (docs/q2_0_optimization/05 #1c). Bit-identical: each
        // output column is an independent reduction. Default off until the A/B.
        static const bool q2_multi = std::getenv("IE_Q2_MULTI") != nullptr;
        bool ffn_fused = false;
        if (q2_multi && T == 1 && w.ffn_gate_q2soa.active() && w.ffn_up_q2soa.active() &&
            w.ffn_gate_q2soa.dtype == DType::kQ2_0 &&
            w.ffn_up_q2soa.dtype == DType::kQ2_0) {   // Q1-native → per-projection
            const void* fin = ffn_in_q8;
            sycl::event qe;
            std::vector<sycl::event> d;
            if (!fin) { qe = quantize_q8_1(q, ws_x_normed_, ws_ffn_in_q8_, H); fin = ws_ffn_in_q8_; d = {qe}; }
            const uint8_t*  qs[3] = {w.ffn_gate_q2soa.qs, w.ffn_up_q2soa.qs, nullptr};
            const uint16_t* dd[3] = {w.ffn_gate_q2soa.d,  w.ffn_up_q2soa.d,  nullptr};
            sycl::half*     yy[3] = {ws_ffn_gate_, ws_ffn_up_, nullptr};
            const uint32_t  NN[3] = {w.ffn_gate_q2soa.N, w.ffn_up_q2soa.N, 0};
            gemv_q2_0_soa_q8_multi(q, fin, qs, dd, yy, NN, 2, w.ffn_gate_q2soa.K, d);
            ffn_fused = true;
        }
        // dense SwiGLU MLP
        if (ffn_fused) { /* gate+up already computed */ }
        else if (w.ffn_gate_q2soa.active())
            gemv_q2soa_T(q, ws_x_normed_, w.ffn_gate_q2soa, ws_ffn_gate_, T, ffn_in_q8);
        else if (w.ffn_gate_soa.active())
            gemv_q6soa_T(q, ws_x_normed_, w.ffn_gate_soa, ws_ffn_gate_, T, ffn_in_q8);
        else if (w.ffn_gate_q4soa.active())
            gemv_q4soa_T(q, ws_x_normed_, w.ffn_gate_q4soa, ws_ffn_gate_, T, ffn_in_q8);
        else if (ffn_ro && w.ffn_gate.reorder)
            gemv_q4_K_reorder_q8(q, ffn_in_q8, w.ffn_gate.reorder, ws_ffn_gate_, H, F);
        else
            dense::gemv_q_T(q, ws_x_normed_, w.ffn_gate, ws_ffn_gate_, H, F, T);
        if (ffn_fused) { /* handled above */ }
        else if (w.ffn_up_q2soa.active())
            gemv_q2soa_T(q, ws_x_normed_, w.ffn_up_q2soa, ws_ffn_up_, T, ffn_in_q8);
        else if (w.ffn_up_soa.active())
            gemv_q6soa_T(q, ws_x_normed_, w.ffn_up_soa, ws_ffn_up_, T, ffn_in_q8);
        else if (w.ffn_up_q4soa.active())
            gemv_q4soa_T(q, ws_x_normed_, w.ffn_up_q4soa, ws_ffn_up_, T, ffn_in_q8);
        else if (ffn_ro && w.ffn_up.reorder)
            gemv_q4_K_reorder_q8(q, ffn_in_q8, w.ffn_up.reorder, ws_ffn_up_, H, F);
        else
            dense::gemv_q_T(q, ws_x_normed_, w.ffn_up,   ws_ffn_up_,   H, F, T);
        swiglu(q, ws_ffn_gate_, ws_ffn_up_, ws_ffn_h_, uint64_t(T) * F);
        if (w.ffn_down_q2soa.active())
            gemv_q2soa_T(q, ws_ffn_h_, w.ffn_down_q2soa, ws_attn_block_, T);
        else if (w.ffn_down_soa.active())
            gemv_q6soa_T(q, ws_ffn_h_, w.ffn_down_soa, ws_attn_block_, T);
        else if (w.ffn_down_q4soa.active())
            gemv_q4soa_T(q, ws_ffn_h_, w.ffn_down_q4soa, ws_attn_block_, T);
        else if (ffn_ro && w.ffn_down.reorder) {
            // ffn_down (K=17408, biggest Q4_K) — quantize the swiglu output once
            // (ws_q8_ is sized for max(H,F) and free in the layer loop), reorder kernel.
            quantize_q8_1(q, ws_ffn_h_, ws_q8_, F);
            gemv_q4_K_reorder_q8(q, ws_q8_, w.ffn_down.reorder, ws_attn_block_, F, H);
        }
        else
            dense::gemv_q_T(q, ws_ffn_h_, w.ffn_down, ws_attn_block_, F, H, T);
        residual_add(q, ws_x_, ws_attn_block_, ws_x_, uint64_t(T) * H);

        dump_residual(int(L) + 1);

        // dspark 5-tap export (M2): copy this layer's OUTPUT residual (l_out) into
        // taps_out at its capture slot. Layout [n_capture, T, H] (capture-outer →
        // one device→device memcpy per captured layer). Default-off (taps_out null
        // OR tap_layers_ empty) → byte-identical to HEAD.
        if (taps_out && !tap_layers_.empty()) {
            for (uint32_t c = 0; c < tap_layers_.size(); ++c) {
                if (uint32_t(tap_layers_[c]) == L) {
                    q.memcpy(taps_out + uint64_t(c) * T * H, ws_x_,
                             uint64_t(T) * H * sizeof(sycl::half));
                }
            }
        }
    }

    // Spec-decode: export the pre-output_norm residual (h_i for the MTP head)
    // BEFORE the final norm overwrites ws_x_normed_ (ws_x_ itself is untouched).
    if (hidden_pre_norm)
        q.memcpy(hidden_pre_norm, ws_x_, uint64_t(T) * H * sizeof(sycl::half));

    // final norm + lm_head. Default: last token only (single-vector GEMV).
    // Spec-decode verify (all_logits != null): lm_head on ALL T → [T, vocab].
    rms_norm_f32w(q, ws_x_, output_norm_, ws_x_normed_, T, H, eps);
    DenseQuantPtr lm{output_, output_dtype_};
    if (all_logits) {
        if (output_q2soa_.active())
            gemv_q2soa_T(q, ws_x_normed_, output_q2soa_, all_logits, T);
        else if (output_soa_.active())
            gemv_q6soa_T(q, ws_x_normed_, output_soa_, all_logits, T);
        else
            dense::gemv_q_T(q, ws_x_normed_, lm, all_logits, H, dc.vocab, T);
        // Mirror the last position into out_logits (caller's single-vector view).
        // In-order queue (this forward chains all ops by ordering) → the memcpy
        // sees the completed all_logits GEMV; its event gates the result.
        return q.memcpy(out_logits, all_logits + uint64_t(T - 1) * dc.vocab,
                        uint64_t(dc.vocab) * sizeof(sycl::half));
    }
    const sycl::half* last = ws_x_normed_ + uint64_t(T - 1) * H;
    if (output_q2soa_.active())
        return gemv_q2soa_T(q, last, output_q2soa_, out_logits, /*T=*/1);
    if (output_soa_.active())
        return gemv_q6soa_T(q, last, output_soa_, out_logits, /*T=*/1);
    return dense::gemv_q(q, last, lm, out_logits, H, dc.vocab);
}

}  // namespace ie
