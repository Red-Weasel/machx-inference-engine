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
#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"

#include "dense_dispatch.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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
                           t->dtype != DType::kQ6_K)) {
        err = "ssm proj: expected F32, Q8_0, or Q6_K 2-D"; return out;
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
    } else {  // Q6_K: K-quant ssm proj (e.g. *-Q6_K GGUFs). Dequant per row, then transpose.
        if (K % kQK_K != 0) { err = "ssm proj Q6_K: K not a multiple of 256"; return out; }
        const uint64_t bpr = K / kQK_K;   // q6_K super-blocks per row
        const auto* blocks = reinterpret_cast<const block_q6_K*>(t->data);
        std::vector<float> row(K);
        for (uint64_t n = 0; n < N; ++n) {
            dequant_q6_K_row(blocks + n * bpr, row.data(), K);
            for (uint64_t k = 0; k < K; ++k)
                staging[k * Npad + n] = sycl::half(row[k]);   // [N,K] → [K,Npad]
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
            ti->dtype != DType::kQ8_0)
            return std::string("token_embd: unsupported dtype ") +
                   std::string(type_name(ti->dtype)) + " (need Q4_K/Q6_K/Q8_0)";
        token_embd_dtype_ = ti->dtype;
        token_embd_ = dense::upload<void>(alloc, ti, owned_, err, ti->dtype);
        if (!err.empty()) return "token_embd: " + err;
    }
    output_norm_ = dense::upload<float>(alloc, T("output_norm.weight"),
                                        owned_, err, DType::kF32);
    if (!err.empty()) return "output_norm: " + err;
    {
        // qwen35-27B is NOT tied: output.weight is a separate Q6_K lm_head.
        const auto* ti = T("output.weight");
        if (!ti) { output_ = token_embd_; output_dtype_ = token_embd_dtype_; }
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
                         Q4SoaW* soa4 = nullptr) {
            const GgufTensorInfo* ti = Tlayer(L, name);
            if (q6_soa_on_ && ti && ti->dtype == DType::kQ6_K) {
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
            LQsoa("attn_qkv.weight",  w.attn_qkv,  w.attn_qkv_soa,  &w.attn_qkv_q4soa);  if (!err.empty()) return layer_err("attn_qkv");
            LQsoa("attn_gate.weight", w.attn_gate, w.attn_gate_soa, &w.attn_gate_q4soa); if (!err.empty()) return layer_err("attn_gate");
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
                    LQsoa("ssm_out.weight", w.ssm_out, w.ssm_out_soa, &w.ssm_out_q4soa);
                }
            }
            if (!err.empty()) return layer_err("ssm_out");
        } else {
            // gated full-attention layer
            LQsoa("attn_q.weight", w.attn_q, w.attn_q_soa, &w.attn_q_q4soa);  // joint Q|gate (Q6_K typ.)
            if (!err.empty()) return layer_err("attn_q");
            // attn_k/attn_output are Q5_K (read as F16 = ~3× native bytes on the
            // expand path). Requantize → SoA-Q8 (~half bytes, ~80%-BW int-dot lane,
            // near-lossless). Non-Q5_K → the existing auto path. GEMV grind #2.
            {
                const GgufTensorInfo* tk = Tlayer(L, "attn_k.weight");
                if (q6_soa_on_ && tk && tk->dtype == DType::kQ5_K) {
                    w.attn_k_q8 = upload_requant_q5k_q8_soa(alloc, tk, owned_, err);
                } else {
                    w.attn_k = LQ("attn_k.weight");
                }
            }
            if (!err.empty()) return layer_err("attn_k");
            {
                const GgufTensorInfo* tv = Tlayer(L, "attn_v.weight");
                if (q4_soa_on_ && tv && tv->dtype == DType::kQ4_K) {
                    w.attn_v_q4soa = upload_q4_soa(alloc, tv, owned_, err);
                } else {
                    w.attn_v = LQ("attn_v.weight");
                }
            }
            if (!err.empty()) return layer_err("attn_v");
            {
                const GgufTensorInfo* to = Tlayer(L, "attn_output.weight");
                if (q6_soa_on_ && to && to->dtype == DType::kQ5_K) {
                    w.attn_output_q8 = upload_requant_q5k_q8_soa(alloc, to, owned_, err);
                } else {
                    w.attn_output = LQ("attn_output.weight");
                }
            }
            if (!err.empty()) return layer_err("attn_output");
            if (auto m = F32("attn_q_norm.weight", w.attn_q_norm, "attn_q_norm"); !m.empty()) return m;
            if (auto m = F32("attn_k_norm.weight", w.attn_k_norm, "attn_k_norm"); !m.empty()) return m;
        }

        LQsoa("ffn_gate.weight", w.ffn_gate, w.ffn_gate_soa, &w.ffn_gate_q4soa); if (!err.empty()) return layer_err("ffn_gate");
        LQsoa("ffn_up.weight",   w.ffn_up,   w.ffn_up_soa,   &w.ffn_up_q4soa);   if (!err.empty()) return layer_err("ffn_up");
        LQsoa("ffn_down.weight", w.ffn_down, w.ffn_down_soa, &w.ffn_down_q4soa); if (!err.empty()) return layer_err("ffn_down");
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
        void* p = alloc_->malloc(uint64_t(kBatchedVerifyMax) * (uint64_t(Kmax) / 32) *
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

    eh_proj    = reqf16("nextn.eh_proj.weight");          if (!err.empty()) return err;
    enorm      = reqf32("nextn.enorm.weight");            if (!err.empty()) return err;
    hnorm      = reqf32("nextn.hnorm.weight");            if (!err.empty()) return err;
    shead_norm = reqf32("nextn.shared_head_norm.weight"); if (!err.empty()) return err;
    w_attn_q   = reqf16("attn_q.weight");                 if (!err.empty()) return err;
    w_attn_k   = reqf16("attn_k.weight");                 if (!err.empty()) return err;
    w_attn_v   = reqf16("attn_v.weight");                 if (!err.empty()) return err;
    w_attn_out = reqf16("attn_output.weight");            if (!err.empty()) return err;
    attn_norm      = reqf32("attn_norm.weight");          if (!err.empty()) return err;
    post_attn_norm = reqf32("post_attention_norm.weight");if (!err.empty()) return err;
    attn_q_norm    = reqf32("attn_q_norm.weight");        if (!err.empty()) return err;
    attn_k_norm    = reqf32("attn_k_norm.weight");        if (!err.empty()) return err;
    w_ffn_gate = reqf16("ffn_gate.weight");               if (!err.empty()) return err;
    w_ffn_up   = reqf16("ffn_up.weight");                 if (!err.empty()) return err;
    w_ffn_down = reqf16("ffn_down.weight");               if (!err.empty()) return err;

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
    mtp_kc = A(uint64_t(n_kv_heads) * max_ctx * HD);
    mtp_vc = A(uint64_t(n_kv_heads) * max_ctx * HD);
    if (!d_logits1 || !mtp_kc || !mtp_vc || !d_pos1 || !d_tok1)
        return "MTP head scratch alloc failed";

    loaded = true;
    return {};
}

void MtpHead::embed(sycl::queue& q, int32_t tok) {
    q.memcpy(d_tok1, &tok, sizeof(int32_t));
    if (te_dtype == DType::kQ4_K)      embedding_lookup_q4k(q, d_tok1, te_dev, d_e, 1, H);
    else if (te_dtype == DType::kQ6_K) embedding_lookup_q6k(q, d_tok1, te_dev, d_e, 1, H);
    else                               embedding_lookup_q8_0(q, d_tok1, te_dev, d_e, 1, H);
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
    gemv_fp16(q, d_cat, eh_proj, d_x, 2u * H, H);
}

void MtpHead::run_layer(sycl::queue& q, int32_t pos) {
    q.memcpy(d_pos1, &pos, sizeof(int32_t));
    rms_norm_f32w(q, d_x, attn_norm, d_xn, 1, H, eps);
    gemv_fp16(q, d_xn, w_attn_q, d_qg, H, N_qg);
    split_q_gate_per_head(q, d_qg, d_q, d_gate, 1, n_q_heads, HD);
    gemv_fp16(q, d_xn, w_attn_k, d_k, H, N_kv);
    gemv_fp16(q, d_xn, w_attn_v, d_v, H, N_kv);
    rms_norm_f32w(q, d_q, attn_q_norm, d_q, n_q_heads,  HD, eps);
    rms_norm_f32w(q, d_k, attn_k_norm, d_k, n_kv_heads, HD, eps);
    rope_partial(q, d_q, d_pos1, d_q, 1, n_q_heads,  HD, rope_n, rope_theta);
    rope_partial(q, d_k, d_pos1, d_k, 1, n_kv_heads, HD, rope_n, rope_theta);
    full_attention(q, d_q, d_k, d_v, mtp_kc, mtp_vc, d_ao,
                   1, uint32_t(pos), n_q_heads, n_kv_heads, HD, max_ctx);
    sigmoid_gate(q, d_ao, d_gate, d_ao, uint64_t(N_q));
    gemv_fp16(q, d_ao, w_attn_out, d_blk, N_q, H);
    residual_add(q, d_x, d_blk, d_x, uint64_t(H));
    rms_norm_f32w(q, d_x, post_attn_norm, d_xn, 1, H, eps);
    gemv_fp16(q, d_xn, w_ffn_gate, d_fg, H, F);
    gemv_fp16(q, d_xn, w_ffn_up,   d_fu, H, F);
    swiglu(q, d_fg, d_fu, d_fh, uint64_t(F));
    gemv_fp16(q, d_fh, w_ffn_down, d_blk, F, H);
    residual_add(q, d_x, d_blk, d_x, uint64_t(H));
    rms_norm_f32w(q, d_x, shead_norm, d_xn, 1, H, eps);
    gemv_fp16(q, d_xn, w_lm_head, d_logits1, H, vocab);
}

void MtpHead::draft(sycl::queue& q, const sycl::half* h_last, int32_t tn,
                    uint32_t p_base, uint32_t K, std::vector<int32_t>& out) {
    out.clear();
    int32_t e_tok = tn;
    const sycl::half* h_src = h_last;
    std::vector<sycl::half> row(vocab);
    for (uint32_t j = 0; j < K; ++j) {
        build_x(q, h_src, e_tok);
        run_layer(q, int32_t(p_base + j));
        q.wait();
        q.memcpy(row.data(), d_logits1, uint64_t(vocab) * sizeof(sycl::half)).wait();
        int32_t g = mtp_argmax_row(row.data(), vocab);
        out.push_back(g);
        h_src = d_x;     // d_x overwritten next build_x, but build_x reads h_src
                         // into d_h FIRST (memcpy) before touching d_x — safe.
        e_tok = g;
    }
}

std::string Qwen35DenseModel::load_mtp_head(const GgufReader& g, uint32_t max_ctx) {
    if (!alloc_) return "load_mtp_head: model not loaded";
    return mtp_.load(*alloc_, g, cfg_, max_ctx);
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
    const uint32_t vocab = cfg_.dense.vocab;
    const uint32_t L_full = cfg_.n_transformer_layers() / cfg_.full_attn_interval;

    if (auto e = ensure_workspace(K); !e.empty()) return "spec ws: " + e;

    Qwen35SpecCheckpoint ckpt;
    if (auto e = ckpt.init(*alloc_, dn, K); !e.empty()) return "spec ckpt: " + e;

    auto* d_all   = sycl::malloc_device<sycl::half>(uint64_t(K) * vocab, q);
    auto* d_hid   = sycl::malloc_device<sycl::half>(uint64_t(K) * H, q);
    auto* d_logits= sycl::malloc_device<sycl::half>(vocab, q);
    auto* d_ids   = sycl::malloc_device<int32_t>(K, q);
    if (!d_all || !d_hid || !d_logits || !d_ids) {
        if (d_all) sycl::free(d_all, q); if (d_hid) sycl::free(d_hid, q);
        if (d_logits) sycl::free(d_logits, q); if (d_ids) sycl::free(d_ids, q);
        return "spec_generate: scratch alloc failed";
    }
    std::vector<sycl::half> Lrow(uint64_t(K) * vocab), Hrow(uint64_t(K) * H);

    std::string ret;
    uint32_t emitted = 0;
    uint32_t p = start_pos;            // committed prefix length (= next write pos)
    bool first_round = true;
    bool abort = false;

    while (emitted < max_new && !abort) {
        // 1. DRAFT K tokens (fresh per-round MTP KV window from pos 0).
        std::vector<int32_t> drafted;
        mtp_.draft(q, h_last, tn, /*p_base=*/0, K, drafted);

        // 2. VERIFY: target forward(T=K, start_pos=p) in CHECKPOINT MODE.
        std::vector<uint32_t> kv_len_snap(L_full);
        for (uint32_t l = 0; l < L_full; ++l) kv_len_snap[l] = kv.length(l);

        std::vector<int32_t> vin(K);
        vin[0] = tn;
        for (uint32_t j = 1; j < K; ++j) vin[j] = drafted[j - 1];
        q.memcpy(d_ids, vin.data(), K * sizeof(int32_t)).wait();
        forward(q, d_ids, K, p, kv, dn, d_logits,
                /*all_logits=*/d_all, /*hidden_pre_norm=*/d_hid, /*ckpt=*/&ckpt).wait();
        q.memcpy(Lrow.data(), d_all, uint64_t(K) * vocab * sizeof(sycl::half)).wait();
        q.memcpy(Hrow.data(), d_hid, uint64_t(K) * H * sizeof(sycl::half)).wait();

        std::vector<int32_t> targ(K);
        for (uint32_t j = 0; j < K; ++j)
            targ[j] = mtp_argmax_row(Lrow.data() + uint64_t(j) * vocab, vocab);

        uint32_t n = 0;   // drafted tokens accepted (g_1..g_n)
        for (uint32_t j = 1; j < K; ++j) {
            if (drafted[j - 1] == targ[j - 1]) ++n; else break;
        }
        const int32_t bonus = targ[n];
        const uint32_t accepted = n + 1;   // input rows kept (tn + g_1..g_n)

        // 3. COMMIT state to p+accepted — NO re-forward (per-position ckpt).
        if (accepted < K) {
            if (auto e = ckpt.commit_to_n(q, dn, accepted); !e.empty()) {
                ret = "spec commit: " + e; abort = true;
            }
            for (uint32_t l = 0; l < L_full; ++l)
                kv.set_length(l, kv_len_snap[l] + accepted);
        }
        // else: all K verify rows accepted → dn/kv already at p+K = p+accepted.

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

        // next round: tn' = bonus, h_last' = pre-output_norm hidden @ accepted-1.
        tn = bonus;
        q.memcpy(h_last, d_hid + uint64_t(accepted - 1) * H,
                 uint64_t(H) * sizeof(sycl::half)).wait();
        p += accepted;
        first_round = false;
    }

    sycl::free(d_all, q); sycl::free(d_hid, q);
    sycl::free(d_logits, q); sycl::free(d_ids, q);
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
                                      Qwen35SpecCheckpoint* ckpt) {
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
            if (w.attn_qkv_soa.active())
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
            if (w.attn_gate_soa.active())
                gemv_q6soa_T(q, ws_x_normed_, w.attn_gate_soa, ws_dn_z_, T, attn_norm_q8);
            else if (w.attn_gate_q4soa.active())
                gemv_q4soa_T(q, ws_x_normed_, w.attn_gate_q4soa, ws_dn_z_, T, attn_norm_q8);
            else
                dense::gemv_q_T(q, ws_x_normed_, w.attn_gate, ws_dn_z_, H, SI, T);
            gated_rms_norm(q, ws_dn_out_, ws_dn_z_, w.ssm_norm_fp16, ws_qkv_,
                           T * SVH, SHD, eps);
            // output projection → ws_attn_block_ (the linear attn contribution)
            if (w.ssm_out_q8.active())
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
            // joint Q|gate projection → split per head into Q and the σ-gate
            if (w.attn_q_soa.active())
                gemv_q6soa_T(q, ws_x_normed_, w.attn_q_soa, ws_qg_, T, attn_norm_q8);
            else if (w.attn_q_q4soa.active())
                gemv_q4soa_T(q, ws_x_normed_, w.attn_q_q4soa, ws_qg_, T, attn_norm_q8);
            else
                dense::gemv_q_T(q, ws_x_normed_, w.attn_q, ws_qg_, H, N_qg, T);
            split_q_gate_per_head(q, ws_qg_, ws_q_, ws_gate_, T, dc.n_q_heads, HD);
            // K, V (attn_k Q5_K → SoA-Q8 fast lane when requantized)
            if (w.attn_k_q8.active())
                gemv_q8soa_T(q, ws_x_normed_, w.attn_k_q8, ws_k_, T, attn_norm_q8);
            else
                dense::gemv_q_T(q, ws_x_normed_, w.attn_k, ws_k_, H, N_kv, T);
            if (w.attn_v_q4soa.active())
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
                full_attention_fa2_decode(q, ws_q_, ws_k_, ws_v_, kc, vc,
                                          ws_attn_out_, ws_attn_partials_,
                                          start_pos, dc.n_q_heads, dc.n_kv_heads,
                                          HD, kv.config().max_ctx);
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
            // post-attention sigmoid gate: attn_out · σ(gate)
            sigmoid_gate(q, ws_attn_out_, ws_gate_, ws_attn_out_, uint64_t(T) * N_q);
            // output projection → ws_attn_block_
            if (w.attn_output_q8.active())
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
        // dense SwiGLU MLP
        if (w.ffn_gate_soa.active())
            gemv_q6soa_T(q, ws_x_normed_, w.ffn_gate_soa, ws_ffn_gate_, T, ffn_in_q8);
        else if (w.ffn_gate_q4soa.active())
            gemv_q4soa_T(q, ws_x_normed_, w.ffn_gate_q4soa, ws_ffn_gate_, T, ffn_in_q8);
        else if (ffn_ro && w.ffn_gate.reorder)
            gemv_q4_K_reorder_q8(q, ffn_in_q8, w.ffn_gate.reorder, ws_ffn_gate_, H, F);
        else
            dense::gemv_q_T(q, ws_x_normed_, w.ffn_gate, ws_ffn_gate_, H, F, T);
        if (w.ffn_up_soa.active())
            gemv_q6soa_T(q, ws_x_normed_, w.ffn_up_soa, ws_ffn_up_, T, ffn_in_q8);
        else if (w.ffn_up_q4soa.active())
            gemv_q4soa_T(q, ws_x_normed_, w.ffn_up_q4soa, ws_ffn_up_, T, ffn_in_q8);
        else if (ffn_ro && w.ffn_up.reorder)
            gemv_q4_K_reorder_q8(q, ffn_in_q8, w.ffn_up.reorder, ws_ffn_up_, H, F);
        else
            dense::gemv_q_T(q, ws_x_normed_, w.ffn_up,   ws_ffn_up_,   H, F, T);
        swiglu(q, ws_ffn_gate_, ws_ffn_up_, ws_ffn_h_, uint64_t(T) * F);
        if (w.ffn_down_soa.active())
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
        if (output_soa_.active())
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
    if (output_soa_.active())
        return gemv_q6soa_T(q, last, output_soa_, out_logits, /*T=*/1);
    return dense::gemv_q(q, last, lm, out_logits, H, dc.vocab);
}

}  // namespace ie
