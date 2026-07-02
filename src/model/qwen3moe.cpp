// src/model/qwen3moe.cpp — Qwen3 standard MoE forward (Qwen3-Coder-30B-A3B).
// Additive — new file; the crown (qwen36.cpp) MoE kernels are untouched. The
// attention block mirrors DenseModel (QK-norm + partial RoPE + GQA); the FFN is
// an UNFUSED top-k MoE (router on host, expert GEMVs on device) for correctness.
// Optimization (batched/grouped expert GEMM) is a follow-up.

#include "ie/qwen3moe.hpp"

#include "ie/dequant.hpp"
#include "ie/moe_qwen3.hpp"
#include "ie/ops.hpp"
#include "ie/qwen3moe_pack.hpp"
#include "ie/quant_blocks.hpp"
#include "ie/quant_soa.hpp"

#include "dense_dispatch.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace ie {
namespace {
// Dequant a Q5_K/Q8_0 weight [K,N] → device F16 [K,N] (transposed dequant
// kernels), consumable by gemv_fp16/gemm_fp16. Copied from qwen35_dense.cpp:
// the Q4_K_M GGUF mixes per-layer dtypes (some attn_k/output are Q5_K/Q8_0).
DenseQuantPtr upload_dequant_to_fp16(DeviceAllocator& alloc, const GgufTensorInfo* t,
                                     std::vector<void*>& owned, std::string& err) {
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
// Load a matrix weight by its actual dtype: Q4_K/Q6_K/F16 ride the dense GEMV
// path; Q5_K/Q8_0 are dequanted to F16 [K,N].
DenseQuantPtr upload_weight_auto(DeviceAllocator& alloc, const GgufTensorInfo* t,
                                 std::vector<void*>& owned, std::string& err) {
    if (!t) { err = "tensor not found"; return {}; }
    if (t->dtype == DType::kQ5_K || t->dtype == DType::kQ8_0)
        return upload_dequant_to_fp16(alloc, t, owned, err);
    return dense::upload_quant_dense(alloc, t, owned, err);
}

// Repack an AoS expert-bank tensor (Q4_K/Q6_K, [n_experts, ...]) to per-expert
// SoA streams (ie/quant_soa.hpp), so the int-dot expert GEMVs read coalesced
// plane-contiguous bytes — the same lever the crown uses (qwen36.cpp:266) to
// hit ~80% BW on the MoE experts.  Same bits, reordered within each expert's
// per-expert block → PPL-free by construction.  `staging` is caller-owned and
// reused across the (gate/up/down × n_layers) expert tensors.  Per-expert byte
// stride is unchanged (nbytes/n_experts) — the SoA reorder is intra-expert.
void* upload_expert_soa(DeviceAllocator& alloc, const GgufTensorInfo* t,
                        uint32_t n_experts, std::vector<uint8_t>& staging,
                        std::vector<void*>& owned, std::string& err) {
    if (!t) { err = "tensor not found"; return nullptr; }
    const size_t bs = (t->dtype == DType::kQ4_K) ? sizeof(block_q4_K)
                    : (t->dtype == DType::kQ6_K) ? sizeof(block_q6_K) : 0;
    if (bs == 0 || n_experts == 0 ||
        t->nbytes % (uint64_t(n_experts) * bs) != 0) {
        err = "expert soa repack: unexpected dtype/geometry"; return nullptr;
    }
    const uint64_t nb_e = t->nbytes / n_experts / bs;
    staging.resize(t->nbytes);
    if (t->dtype == DType::kQ4_K)
        repack_moe_q4k_soa_host(static_cast<const uint8_t*>(t->data),
                                staging.data(), n_experts, nb_e);
    else
        repack_moe_q6k_soa_host(static_cast<const uint8_t*>(t->data),
                                staging.data(), n_experts, nb_e);
    void* d = alloc.malloc(t->nbytes);
    if (!d) { err = "expert soa: malloc failed"; return nullptr; }
    alloc.queue().memcpy(d, staging.data(), t->nbytes).wait();
    owned.push_back(d);
    return d;
}
}  // namespace

Qwen3MoeModel::~Qwen3MoeModel() { free_all(); }

void Qwen3MoeModel::free_all() {
    if (!alloc_) return;
    for (void* p : owned_) alloc_->free(p);
    owned_.clear();
}

std::string Qwen3MoeModel::ensure_workspace(uint32_t max_T) {
    if (max_T <= ws_T_) return {};
    auto ah = [&](uint64_t n) {
        auto* p = static_cast<sycl::half*>(alloc_->malloc(n * sizeof(sycl::half)));
        if (p) owned_.push_back(p);
        return p;
    };
    const auto& d = cfg_.dense;
    const uint32_t H = d.hidden, HD = d.head_dim, EF = cfg_.expert_ffn;
    const uint32_t N_q = d.n_q_heads * HD, N_kv = d.n_kv_heads * HD;
    const uint64_t T = max_T;
    ws_x_        = ah(T * H);
    ws_xn_       = ah(T * H);
    ws_q_        = ah(T * N_q);
    ws_k_        = ah(T * N_kv);
    ws_v_        = ah(T * N_kv);
    ws_attn_out_ = ah(T * N_q);
    ws_block_    = ah(T * H);
    ws_gate_     = ah(EF);
    ws_up_       = ah(EF);
    ws_eh_       = ah(EF);
    ws_eout_     = ah(H);
    ws_acc_      = ah(H);
    // --- fused MoE FFN staging buffers (no behaviour change; nothing reads these yet) ---
    const uint32_t Kc = cfg_.n_experts_used, Ec = cfg_.n_experts;
    const uint64_t TK = uint64_t(T) * Kc;
    ws_xp_packed_  = ah(TK * H);
    ws_h_packed_   = ah(TK * EF);
    ws_out_packed_ = ah(TK * H);
    // int-dot q8 activation streams (block_q8_1s = 48 B / 32 elems, split sums)
    ws_xp_q8_ = alloc_->malloc(TK * (H / 32) * sizeof(block_q8_1s));
    if (ws_xp_q8_) owned_.push_back(ws_xp_q8_);
    ws_h_q8_  = alloc_->malloc(TK * (EF / 32) * sizeof(block_q8_1s));
    if (ws_h_q8_) owned_.push_back(ws_h_q8_);
    ws_ffn_y_      = ah(T * H);
    ws_sorted_w_       = ah(TK);
    ws_weights_packed_ = ah(TK);
    auto ai = [&](uint64_t n) {
        auto* p = static_cast<int32_t*>(alloc_->malloc(n * sizeof(int32_t)));
        if (p) owned_.push_back(p);
        return p;
    };
    ws_expert_offsets_ = static_cast<uint32_t*>(alloc_->malloc((Ec + 1) * sizeof(uint32_t)));
    if (ws_expert_offsets_) owned_.push_back(ws_expert_offsets_);
    ws_sorted_idx_   = ai(TK);
    ws_tk_to_packed_ = ai(TK);
    ws_topk_idx_     = ai(Kc);
    ws_topk_w_       = ah(Kc);
    ws_router_logits_ = ah(uint64_t(T) * Ec);   // [T*E] device prefill logits
    host_router_logits_.resize(Ec);
    host_logits_.resize(uint64_t(T) * Ec);
    ws_T_moe_ = max_T;
    ws_pos_      = static_cast<int32_t*>(alloc_->malloc(T * sizeof(int32_t)));
    if (ws_pos_) owned_.push_back(ws_pos_);
    if (!ws_x_ || !ws_xn_ || !ws_q_ || !ws_k_ || !ws_v_ || !ws_attn_out_ || !ws_block_ ||
        !ws_gate_ || !ws_up_ || !ws_eh_ || !ws_eout_ || !ws_acc_ || !ws_pos_ ||
        !ws_xp_packed_ || !ws_h_packed_ || !ws_out_packed_ || !ws_ffn_y_ ||
        !ws_sorted_w_ || !ws_weights_packed_ || !ws_expert_offsets_ ||
        !ws_sorted_idx_ || !ws_tk_to_packed_ || !ws_topk_idx_ || !ws_topk_w_ ||
        !ws_router_logits_ || !ws_xp_q8_ || !ws_h_q8_)
        return "qwen3moe workspace alloc failed";
    host_xn_.resize(T * H);
    ws_T_ = max_T;
    return {};
}

// IE_QWEN3MOE_MOE_XMX scratch (default-OFF path; allocated lazily on first use
// so the int-dot path's default memory footprint is unchanged). Per-expert
// single-expert GEMM scratch: each gemm_fp16 writes its output at row 0 of a
// buffer sized to round_up(TK,8) rows, so the full-tile row-tail spill
// (gemm_fp16 CONTRACT: writes up to round_up(M,8)×round_up(N,16)) lands in
// unused rows past n_tok_e — never into another expert's slice. The exact-sized
// per-expert results are then copied into the globally-packed ws_h_packed_ /
// ws_out_packed_ (contiguous, no spill).
std::string Qwen3MoeModel::ensure_moe_xmx_scratch(uint32_t T) {
    const auto& d = cfg_.dense;
    const uint32_t H = d.hidden, EF = cfg_.expert_ffn, K = cfg_.n_experts_used;
    const uint32_t TK = T * K;
    const uint32_t TK8 = ((TK + 7) / 8) * 8;          // round_up(TK,8) — ragged-M pad
    if (TK8 <= ws_xmx_cap_) return {};
    auto ah = [&](uint64_t n) {
        auto* p = static_cast<sycl::half*>(alloc_->malloc(n * sizeof(sycl::half)));
        if (p) owned_.push_back(p);
        return p;
    };
    const uint64_t maxN = (EF > H) ? EF : H;          // widest GEMM output (= EF here)
    // B^T dequant scratch must hold both gate/up (K=H × N=EF) and down (K=EF × N=H);
    // both are H*EF elements → size to H*EF.
    ws_xmx_wbt_  = ah(uint64_t(H) * EF);
    ws_xmx_cf32_ = static_cast<float*>(alloc_->malloc(uint64_t(TK8) * maxN * sizeof(float)));
    if (ws_xmx_cf32_) owned_.push_back(ws_xmx_cf32_);
    ws_xmx_gate_ = ah(uint64_t(TK8) * EF);
    ws_xmx_up_   = ah(uint64_t(TK8) * EF);
    ws_xmx_h_    = ah(uint64_t(TK8) * EF);
    ws_xmx_out_  = ah(uint64_t(TK8) * H);
    if (!ws_xmx_wbt_ || !ws_xmx_cf32_ || !ws_xmx_gate_ || !ws_xmx_up_ ||
        !ws_xmx_h_ || !ws_xmx_out_)
        return "qwen3moe MoE-XMX scratch alloc failed";
    ws_xmx_cap_ = TK8;
    return {};
}

// FA-2 split-K decode partials scratch. Sizing mirrors qwen35_dense.cpp:358 /
// dense_transformer.cpp:230: Bc-chunked along ctx (Bc_floor=64), one slot per
// (chunk, q_head) of (head_dim + 2) floats (the +2 = online-softmax m_local,
// l_local). qwen3moe attn dims: n_q_heads=32, head_dim=128, all-full-attn.
std::string Qwen3MoeModel::ensure_attn_partials(uint32_t max_ctx) {
    if (!alloc_) return "alloc not initialized";
    if (max_ctx <= ws_attn_partials_ctx_) return {};
    constexpr uint32_t Bc_floor = 64;
    const uint32_t n_chunks_max = (max_ctx + Bc_floor - 1) / Bc_floor;
    const uint64_t n_floats =
        uint64_t(n_chunks_max) * cfg_.dense.n_q_heads * (cfg_.dense.head_dim + 2);
    auto* p = static_cast<float*>(alloc_->malloc(n_floats * sizeof(float)));
    if (!p) return "qwen3moe attn_partials alloc failed";
    owned_.push_back(p);
    ws_attn_partials_     = p;
    ws_attn_partials_ctx_ = max_ctx;
    return {};
}

std::string Qwen3MoeModel::load(DeviceAllocator& alloc, const GgufReader& g,
                                const Qwen3MoeConfig& cfg) {
    alloc_ = &alloc;
    cfg_ = cfg;

    // Tier-2 prefill investigation (docs/superpowers/specs/2026-06-12-decode-
    // attention-lever-scoping.md): oneDNN was the 27B's prefill lever, but a
    // warm-plateau A/B (2026-06-12) showed it does NOT help qwen3moe — and the
    // kernel profile says why: at pp512 the attention q/k/v/o projection GEMMs
    // (the only thing dense::gemv_q_T + prefer_onednn() would route through) are
    // just ~3% of prefill; ~84% is the MoE *expert* GEMMs (moe_pfl_down_pk6k
    // 69%, _pk4k 15%, _gate_q4k 12% — moe_fused.cpp), which run their own fused
    // path untouched by prefer_onednn(). Setting the flag was PPL-neutral (11.98)
    // but pp512 was unchanged (~88 in-house vs ~89-92 oneDNN, within ±noise), so
    // it is NOT enabled here (no win, avoids dead surface). The real qwen3moe
    // prefill lever is the Q6_K expert down-projection (Tier-3, crown-owned
    // moe_fused.cpp = forbidden; needs an additive qwen3moe SoA/repacked-Q6_K
    // kernel, deferred). H=2048 (vs the 27B's 5120) is why oneDNN's big-K edge
    // does not appear. Enable for an A/B with IE_QWEN3MOE_ONEDNN=1.
    if (std::getenv("IE_QWEN3MOE_ONEDNN") != nullptr) dense::prefer_onednn() = true;

    const auto& d = cfg.dense;
    const uint32_t E = cfg_.n_experts;
    char buf[64];
    auto T = [&](const char* n) { return g.find_tensor(n); };
    auto Tl = [&](uint32_t L, const char* n) {
        std::snprintf(buf, sizeof(buf), "blk.%u.%s", L, n); return g.find_tensor(buf);
    };
    std::string err;

    {
        const auto* ti = T("token_embd.weight");
        if (!ti) return "token_embd: not found";
        if (ti->dtype != DType::kQ4_K && ti->dtype != DType::kQ6_K) return "token_embd: unsupported dtype";
        token_embd_dtype_ = ti->dtype;
        token_embd_ = dense::upload<void>(alloc, ti, owned_, err, ti->dtype);
        if (!err.empty()) return "token_embd: " + err;
    }
    output_norm_ = dense::upload<float>(alloc, T("output_norm.weight"), owned_, err, DType::kF32);
    if (!err.empty()) return "output_norm: " + err;
    {
        const auto* ti = T("output.weight");
        if (!ti) { output_ = token_embd_; output_dtype_ = token_embd_dtype_; }   // tied
        else {
            if (ti->dtype != DType::kQ4_K && ti->dtype != DType::kQ6_K) return "output: unsupported dtype";
            output_dtype_ = ti->dtype;
            output_ = dense::upload<void>(alloc, ti, owned_, err, ti->dtype);
            if (!err.empty()) return "output: " + err;
        }
    }

    // Per-expert SoA repack (coalesced int-dot expert reads, the crown's lever).
    // Default-ON; opt-out IE_NO_MOE_SOA=1 for an order-controlled A/B.  The
    // int-dot kernels (gate_up_silu_q8 / down_q*k_q8_gen) consume the SoA layout
    // via q{4,6}k_wview<true>; the fp16/packed fallbacks need AoS, so SoA only
    // applies on the q8 decode/prefill path.
    // SoA is ONLY consumed by the q8 int-dot kernels (gate_up_silu_q{4,6}k_q8 +
    // down_q*k_q8_gen).  The fp16/packed fallbacks (NO_Q8 / FP16_DECODE /
    // UNFUSED) and incompatible geometry need AoS, so disable SoA there to keep
    // those paths correct.  Coder-30B (H=2048, EF=768) satisfies all of these.
    // Q6_K gate/up models (bartowski Q6_K of Qwen3-30B-A3B, e.g. Tongyi-Deep-
    // Research) now take the int-dot fused path too (moe_prefill_gate_up_silu_
    // q6k_q8), so their Q6_K gate/up banks ARE SoA-repacked.  INVARIANT: whenever
    // moe_exps_soa_ is on, the `unfused` guard in forward() takes the fused path
    // (the SoA preconditions here — no NO_Q8/FP16_DECODE/UNFUSED, H%512==0,
    // EF%256==0 — exactly match `gateup_fusable` there), so the AoS reference /
    // fp16 paths (which read AoS) are only reached when SoA is OFF.  Per-tensor
    // SoA stays dtype-gated below (Q6_K only; Q4_K AoS is already coalesced).
    // Only fuse gate/up dtypes we have int-dot kernels for (Q4_K, Q6_K); any
    // other gate/up dtype forces SoA off → whole-MoE AoS reference path.
    const auto* gate0 = T("blk.0.ffn_gate_exps.weight");
    const auto* up0   = T("blk.0.ffn_up_exps.weight");
    const bool gateup_soa_ok =
        (!gate0 || gate0->dtype == DType::kQ4_K || gate0->dtype == DType::kQ6_K) &&
        (!up0   || up0->dtype   == DType::kQ4_K || up0->dtype   == DType::kQ6_K);
    moe_exps_soa_ = std::getenv("IE_NO_MOE_SOA")            == nullptr
                 && std::getenv("IE_QWEN3MOE_NO_Q8")        == nullptr
                 && std::getenv("IE_QWEN3MOE_FP16_DECODE")  == nullptr
                 && std::getenv("IE_QWEN3MOE_UNFUSED")      == nullptr
                 && (d.hidden       % 512 == 0)
                 && (cfg_.expert_ffn % 256 == 0)
                 && gateup_soa_ok;
    std::vector<uint8_t> soa_staging;   // reused across gate/up/down × n_layers

    layers_.assign(d.n_layers, {});
    for (uint32_t L = 0; L < d.n_layers; ++L) {
        Layer& w = layers_[L];
        auto LQ = [&](const char* n) { return upload_weight_auto(alloc, Tl(L, n), owned_, err); };
        w.attn_norm = dense::upload<float>(alloc, Tl(L, "attn_norm.weight"), owned_, err, DType::kF32);
        if (!err.empty()) return "attn_norm: " + err;
        w.ffn_norm  = dense::upload<float>(alloc, Tl(L, "ffn_norm.weight"),  owned_, err, DType::kF32);
        if (!err.empty()) return "ffn_norm: " + err;
        w.attn_q = LQ("attn_q.weight"); if (!err.empty()) return "attn_q: " + err;
        w.attn_k = LQ("attn_k.weight"); if (!err.empty()) return "attn_k: " + err;
        w.attn_v = LQ("attn_v.weight"); if (!err.empty()) return "attn_v: " + err;
        w.attn_output = LQ("attn_output.weight"); if (!err.empty()) return "attn_output: " + err;
        if (const auto* qn = Tl(L, "attn_q_norm.weight")) {
            w.attn_q_norm = dense::upload<float>(alloc, qn, owned_, err, DType::kF32);
            if (!err.empty()) return "attn_q_norm: " + err;
        }
        if (const auto* kn = Tl(L, "attn_k_norm.weight")) {
            w.attn_k_norm = dense::upload<float>(alloc, kn, owned_, err, DType::kF32);
            if (!err.empty()) return "attn_k_norm: " + err;
        }
        // router weights → host (F32 [E, hidden], row e = e*hidden contiguous)
        {
            const auto* ti = Tl(L, "ffn_gate_inp.weight");
            if (!ti) return "ffn_gate_inp: not found (layer " + std::to_string(L) + ")";
            if (ti->dtype != DType::kF32) return "ffn_gate_inp: expected F32";
            if (ti->nbytes != uint64_t(E) * d.hidden * sizeof(float))
                return "ffn_gate_inp: unexpected size";
            w.router_w.resize(uint64_t(E) * d.hidden);
            std::memcpy(w.router_w.data(), ti->data, ti->nbytes);
            // ALSO upload transposed F16 [hidden, E] for the GPU-gemm router:
            // dev[h*E + e] = router_w[e*H + h]. router logits then run as
            // gemv_q_T(ws_xn, router_w_dev{F16}, ..., H, E, T) — one gemm vs a
            // single-threaded host dot loop (~66% of prefill wall, 2026-06-12).
            const uint32_t Hn = d.hidden;
            std::vector<sycl::half> rt(uint64_t(Hn) * E);
            for (uint32_t e = 0; e < E; ++e)
                for (uint32_t h = 0; h < Hn; ++h)
                    rt[uint64_t(h) * E + e] = sycl::half(w.router_w[uint64_t(e) * Hn + h]);
            auto* rd = static_cast<sycl::half*>(alloc.malloc(uint64_t(Hn) * E * sizeof(sycl::half)));
            if (!rd) return "ffn_gate_inp: router_w_dev alloc failed (layer " + std::to_string(L) + ")";
            alloc.queue().memcpy(rd, rt.data(), uint64_t(Hn) * E * sizeof(sycl::half)).wait();
            owned_.push_back(rd);
            w.router_w_dev = rd;
        }
        // stacked expert weights → device raw; per-expert byte stride = nbytes/E
        auto up_exps = [&](const char* nm, void*& dst, DType& dt, uint64_t& stride,
                           bool& soa) -> std::string {
            const auto* ti = Tl(L, nm);
            if (!ti) return std::string(nm) + ": not found";
            if (ti->dtype != DType::kQ4_K && ti->dtype != DType::kQ6_K)
                return std::string(nm) + ": unsupported expert dtype";
            if (ti->nbytes % E) return std::string(nm) + ": nbytes not divisible by expert_count";
            // SoA only pays for Q6_K experts (see header note); Q4_K stays AoS.
            soa = moe_exps_soa_ && ti->dtype == DType::kQ6_K;
            if (soa) {
                dst = upload_expert_soa(alloc, ti, E, soa_staging, owned_, err);
                if (!err.empty()) return std::string(nm) + ": " + err;
            } else {
                dst = dense::upload<void>(alloc, ti, owned_, err, ti->dtype);
                if (!err.empty()) return std::string(nm) + ": " + err;
            }
            dt = ti->dtype; stride = ti->nbytes / E;   // intra-expert reorder → stride unchanged
            return {};
        };
        if (auto m = up_exps("ffn_gate_exps.weight", w.gate_exps, w.gate_dt, w.gate_stride, w.gate_soa); !m.empty()) return m;
        if (auto m = up_exps("ffn_up_exps.weight",   w.up_exps,   w.up_dt,   w.up_stride,   w.up_soa);   !m.empty()) return m;
        if (auto m = up_exps("ffn_down_exps.weight", w.down_exps, w.down_dt, w.down_stride, w.down_soa); !m.empty()) return m;
    }
    return {};
}

namespace {
// Host top-k softmax router for one token. logits[e] = router_w[e]·x (F32);
// softmax over all E; pick top-k; renormalize the k weights to sum 1; sort the
// k by ascending expert index (HF iteration order). Matches Qwen3 norm_topk_prob.
// Softmax-after-topk over precomputed logits[E]: softmax all E, take top-k by
// prob, renormalize the k to sum 1, sort ascending by expert index (HF order).
// Split out of route_token so prefill can feed GPU-gemm logits (the dot product
// was a single-threaded host loop = ~66% of prefill wall-clock).
void route_from_logits(const float* logit, uint32_t E, uint32_t k,
                       std::vector<std::pair<uint32_t, float>>& out) {
    float m = logit[0];
    for (uint32_t e = 1; e < E; ++e) m = std::max(m, logit[e]);
    double sum = 0.0;
    std::vector<float> prob(E);
    for (uint32_t e = 0; e < E; ++e) { prob[e] = std::exp(logit[e] - m); sum += prob[e]; }
    for (uint32_t e = 0; e < E; ++e) prob[e] = float(prob[e] / sum);
    std::vector<uint32_t> idx(E);
    for (uint32_t e = 0; e < E; ++e) idx[e] = e;
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](uint32_t a, uint32_t b) { return prob[a] > prob[b]; });
    float wsum = 0.f;
    for (uint32_t j = 0; j < k; ++j) wsum += prob[idx[j]];
    out.clear();
    for (uint32_t j = 0; j < k; ++j) out.emplace_back(idx[j], prob[idx[j]] / wsum);
    std::sort(out.begin(), out.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
}
void route_token(const float* router_w, const sycl::half* xn, uint32_t E, uint32_t H,
                 uint32_t k, std::vector<std::pair<uint32_t, float>>& out) {
    std::vector<float> logit(E);
    for (uint32_t e = 0; e < E; ++e) {
        const float* row = router_w + uint64_t(e) * H;
        float acc = 0.f;
        for (uint32_t h = 0; h < H; ++h) acc += row[h] * float(xn[h]);
        logit[e] = acc;
    }
    route_from_logits(logit.data(), E, k, out);
}
}  // namespace

// Fused MoE FFN over T>1 tokens. Reads post-ffn-norm activations from ws_xn_
// (already on device), routes on host, packs by expert, runs the crown's
// parameterized fused kernels, leaves the per-token MoE output in ws_ffn_y_.
void Qwen3MoeModel::moe_ffn_fused_prefill(sycl::queue& q, const Layer& w, uint32_t T) {
    const auto& d = cfg_.dense;
    const uint32_t H = d.hidden, EF = cfg_.expert_ffn, E = cfg_.n_experts, K = cfg_.n_experts_used;

    // Diagnostic (env IE_QWEN3MOE_PROFILE_HOST): accumulate pure host-side
    // routing wall-time across all layers; print the running total each call.
    static const bool prof_host = std::getenv("IE_QWEN3MOE_PROFILE_HOST") != nullptr;
    // Opt-out: IE_QWEN3MOE_HOST_ROUTER=1 restores the legacy host dot loop (A/B).
    static const bool host_router = std::getenv("IE_QWEN3MOE_HOST_ROUTER") != nullptr;
    static double s_route_ms = 0.0;
    using clk = std::chrono::steady_clock;

    // 1) route every token. router logits = ws_xn[T,H] @ router_w_devᵀ → [T,E].
    //    The dot products run as ONE GPU gemm (gemv_q_T, F16 weight) instead of a
    //    single-threaded host loop that was ~66% of prefill wall-clock; only the
    //    cheap per-token softmax/top-k (E entries) stays on host.
    std::vector<std::vector<std::pair<uint32_t, float>>> routes(T);
    std::vector<std::pair<uint32_t, float>> topk;
    const uint32_t TK = T * K;   // shared scope: the kernels below index by TK
    MoePacking pk;               // filled by the host path; stays empty on-device
    // On-device decode routing (IE_QWEN3MOE_ONDEVICE_ROUTE): router GEMV (GPU) →
    // device softmax/top-K → device packing build → gather, NO host round-trip.
    // Kills the per-layer D→H/H→D stalls + the q.wait() drain so the whole token's
    // layers enqueue and the GPU runs ahead (llama's decode model). Lossless vs the
    // host route_from_logits up to fp32 near-tie flips → PPL-gated.
    static const bool ondevice_route = std::getenv("IE_QWEN3MOE_ONDEVICE_ROUTE") != nullptr;
    if (ondevice_route && T == 1) {
        dense::gemv_q_T(q, ws_xn_, DenseQuantPtr{w.router_w_dev, DType::kF16},
                        ws_router_logits_, H, E, 1);
        moe_topk_from_logits(q, ws_router_logits_, ws_topk_idx_, ws_topk_w_, E, K);
        moe_build_pack_decode(q, ws_topk_idx_, ws_topk_w_, ws_expert_offsets_,
                              ws_sorted_idx_, ws_tk_to_packed_, ws_weights_packed_, E, K);
        moe_gather_rows(q, ws_xn_, ws_sorted_idx_, ws_xp_packed_, K, H);
    } else {
    if (host_router) {
        q.memcpy(host_xn_.data(), ws_xn_, uint64_t(T) * H * sizeof(sycl::half)).wait();
        const auto t_route0 = clk::now();
        for (uint32_t t = 0; t < T; ++t) {
            route_token(w.router_w.data(), host_xn_.data() + uint64_t(t) * H, E, H, K, topk);
            routes[t] = topk;
        }
        if (prof_host) {
            s_route_ms += std::chrono::duration<double, std::milli>(clk::now() - t_route0).count();
            std::fprintf(stderr, "[q3moe-host] cumulative route(host): %.1f ms\n", s_route_ms);
        }
    } else {
        dense::gemv_q_T(q, ws_xn_, DenseQuantPtr{w.router_w_dev, DType::kF16},
                        ws_router_logits_, H, E, T);
        std::vector<sycl::half> hl(uint64_t(T) * E);
        q.memcpy(hl.data(), ws_router_logits_, uint64_t(T) * E * sizeof(sycl::half)).wait();
        const auto t_route0 = clk::now();
        for (uint64_t i = 0; i < uint64_t(T) * E; ++i) host_logits_[i] = float(hl[i]);
        for (uint32_t t = 0; t < T; ++t) {
            route_from_logits(host_logits_.data() + uint64_t(t) * E, E, K, topk);
            routes[t] = topk;
        }
        if (prof_host) {
            s_route_ms += std::chrono::duration<double, std::milli>(clk::now() - t_route0).count();
            std::fprintf(stderr, "[q3moe-host] cumulative route(gpu-logits, host-softmax): %.1f ms\n", s_route_ms);
        }
    }

    // 2) build the expert-sorted packing on host, upload the device arrays.
    build_moe_packing(routes, E, K, pk);
    q.memcpy(ws_expert_offsets_, pk.expert_offsets.data(), (E + 1) * sizeof(uint32_t));
    q.memcpy(ws_sorted_idx_,     pk.sorted_idx.data(),     TK * sizeof(int32_t));
    q.memcpy(ws_tk_to_packed_,   pk.tk_to_packed.data(),   TK * sizeof(int32_t));
    std::vector<sycl::half> wpk(TK);
    for (uint32_t i = 0; i < TK; ++i) wpk[i] = sycl::half(pk.weights_packed[i]);
    q.memcpy(ws_weights_packed_, wpk.data(), TK * sizeof(sycl::half)).wait();

    // 3) gather expert-sorted input rows: ws_xp_packed_[i] = ws_xn_[sorted_idx[i]].
    moe_gather_rows(q, ws_xn_, ws_sorted_idx_, ws_xp_packed_, TK, H);
    }  // end host-routing path (on-device T==1 path handled above)

    // ===================================================================
    // IE_QWEN3MOE_MOE_XMX: fp16-on-XMX per-expert dense GEMM prefill path.
    // Mirrors llama-SYCL's policy of switching quantized MoE GEMM to a
    // dequant-to-fp16 + fp16 matrix-engine GEMM above ~32 tokens/expert
    // (MMQ_MAX_BATCH_SIZE=32). Default-OFF; the int-dot W4A8 path below is
    // unchanged when the env is unset. Requires AoS (non-SoA) expert banks —
    // dequant_q{4,6}_K_to_Bt consume canonical block_q{4,6}_K, not the SoA
    // repack. Set IE_NO_MOE_SOA=1 alongside this to force AoS banks.
    // ===================================================================
    // MoE expert GEMMs on XMX (gemm_fp16) — wins at large batch (1.3-1.46x @ T>=4K)
    // but the per-expert dequant-per-forward fixed cost REGRESSES small batch
    // (0.55x @2K, n_e too small to amortize). Gate on T: XMX only when tokens/expert
    // is large enough. Crossover ~T=3K; default 4096 (known win), env-tunable.
    // DEFAULT-ON (opt-out IE_QWEN3MOE_NO_MOE_XMX): validated 1.3-1.43x prefill at
    // T>=4096, byte-identical to the AoS path, crown bit-exact, +0.17% NLL, decode
    // SoA-Q6K preserved (SoA-aware dequant). Falls back to int-dot below minT.
    static const bool moe_xmx = std::getenv("IE_QWEN3MOE_NO_MOE_XMX") == nullptr;
    static const uint32_t moe_xmx_minT = [] {
        const char* s = std::getenv("IE_QWEN3MOE_MOE_XMX_MINT");
        return s ? uint32_t(std::atoi(s)) : 4096u;
    }();
    // Per-bank layout admissibility: the dequant lambda handles Q6_K from BOTH
    // SoA (dequant_q6_K_soa_to_Bt) and AoS, but Q4_K only from AoS. So a bank is
    // admissible iff it is Q6_K, OR it is Q4_K AND not SoA. This lets the XMX
    // path engage with the DEFAULT SoA-repacked Q6_K down bank (the only SoA
    // bank in practice) while still rejecting any hypothetical SoA-Q4 bank.
    auto bank_ok = [](DType dt, bool soa) {
        if (dt == DType::kQ6_K) return true;            // SoA or AoS both handled
        if (dt == DType::kQ4_K) return !soa;            // Q4_K SoA dequant N/A
        return false;
    };
    if (moe_xmx && T >= moe_xmx_minT &&
        bank_ok(w.gate_dt, w.gate_soa) &&
        bank_ok(w.up_dt,   w.up_soa)   &&
        bank_ok(w.down_dt, w.down_soa) &&
        (H % 256 == 0) && (EF % 256 == 0)) {
        if (auto e = ensure_moe_xmx_scratch(T); !e.empty()) {
            std::fprintf(stderr, "Qwen3MoeModel::ensure_moe_xmx_scratch: %s\n", e.c_str());
        } else {
            moe_xmx_prefill(q, w, pk, T);
            // 6) reduce per token (shared with the int-dot path; see the memset
            //    rationale below — zero ws_ffn_y_ for ALL T, the reduce accumulates).
            q.memset(ws_ffn_y_, 0, uint64_t(T) * H * sizeof(sycl::half));
            moe_prefill_reduce(q, ws_out_packed_,
                               reinterpret_cast<const uint32_t*>(ws_tk_to_packed_),
                               ws_weights_packed_, ws_ffn_y_, T, K, H);
            return;
        }
    }

    // int-dot MoE prefill (crown's W4A8 technique). Default ON; opt out with
    // IE_QWEN3MOE_NO_Q8=1 (fp16-activation fallback) for A/B. soa=false: raw
    // stacked [E,*,*/256] layout, expert stride = nbytes/E.
    static const bool moe_q8 = std::getenv("IE_QWEN3MOE_NO_Q8") == nullptr;
    const bool q8_path = moe_q8 && (H % 512 == 0);

    // 4) stage 1: gate+up+silu per expert. The crown's int-dot q8 gate/up kernel
    //    is already E_ffn-parameterized (contraction is over H; E_ffn is just the
    //    output column with an n>=E_ffn guard), so it serves Coder's E_ffn=768
    //    unchanged once activations are q8-staged.
    if (q8_path) {
        quantize_q8_1s(q, ws_xp_packed_, ws_xp_q8_, uint64_t(TK) * H);
        // Q6_K gate/up (bartowski Q6_K of Qwen3-30B-A3B) → the qwen3moe-owned
        // W6A8 int-dot kernel; Q4_K gate/up (Coder-30B) → the crown kernel. Both
        // consume the same q8 activation stream + expert_offsets, write silu·mul
        // into ws_h_packed_. Q6_K only ever reaches here on the q8 path (the
        // `unfused` guard routes Q6_K to the reference otherwise).
        if (w.gate_dt == DType::kQ6_K)
            moe_prefill_gate_up_silu_q6k_q8(q, ws_xp_q8_, w.gate_exps, w.up_exps,
                                            ws_expert_offsets_, ws_h_packed_,
                                            E, H, EF, w.gate_stride, w.gate_soa);
        else
            moe_prefill_gate_up_silu_q4k_q8(q, ws_xp_q8_, w.gate_exps, w.up_exps,
                                            ws_expert_offsets_, ws_h_packed_,
                                            E, H, EF, w.gate_stride, w.gate_soa);
    } else {
        moe_prefill_gate_up_silu_q4k(q, ws_xp_packed_, w.gate_exps, w.up_exps,
                                     ws_expert_offsets_, ws_h_packed_,
                                     E, H, EF, w.gate_stride, /*soa=*/false);
    }

    // 5) stage 2: down per expert -> out_packed [TK, H]. The crown's int-dot down
    //    kernels are E_ffn==512-locked; the qwen3moe-owned _gen kernels generalize
    //    the same W4A8 path to any E_ffn%256==0 (Coder E_ffn=768). Needs a q8
    //    stream of h_packed; falls back to fp16 if not on the q8 path / E_ffn%256.
    const bool down_q8 = q8_path && (EF % 256 == 0);
    if (down_q8) quantize_q8_1s(q, ws_h_packed_, ws_h_q8_, uint64_t(TK) * EF);
    if (w.down_dt == DType::kQ6_K) {
        if (down_q8)
            moe_prefill_down_q6k_q8_gen(q, ws_h_q8_, w.down_exps, ws_expert_offsets_,
                                        ws_out_packed_, E, H, EF, w.down_stride, w.down_soa);
        else
            moe_prefill_down_packed_q6k(q, ws_h_packed_, w.down_exps, ws_expert_offsets_,
                                        ws_out_packed_, E, H, EF, w.down_stride);
    } else {
        if (down_q8)
            moe_prefill_down_q4k_q8_gen(q, ws_h_q8_, w.down_exps, ws_expert_offsets_,
                                        ws_out_packed_, E, H, EF, w.down_stride, w.down_soa);
        else
            moe_prefill_down_packed_q4k(q, ws_h_packed_, w.down_exps, ws_expert_offsets_,
                                        ws_out_packed_, E, H, EF, w.down_stride);
    }

    // 6) reduce per token: ws_ffn_y_[t] = Σ_kslot weight * out_packed[tk_to_packed[t,kslot]].
    //    tk_to_packed holds non-negative packed-row indices; reinterpret int32->uint32
    //    (bit-identical for values in [0,TK)) to match the kernel's uint32 signature.
    // Zero ws_ffn_y_ before EVERY reduce (all T). moe_prefill_reduce ACCUMULATES
    // (acc = y[t*H+n] as its base — moe_fused.cpp), and ws_ffn_y_ persists across
    // the 48 layers, so without this the reduce adds each layer's MoE onto the
    // prior layer's stale output → residual corruption that compounds to garbage.
    // This was previously T==1-only; the T>1 path was WRONGLY assumed to "overwrite
    // every slot" (the reduce reads y as its base, it does not overwrite). T>1
    // prefill was never correctness-checked at the last position — ie-perplexity
    // defaults to streaming T=1 (prefill_chunk=0), so PPL used the correct T==1
    // path and never exercised T>1. Verified: with this memset the fused T>1
    // prefill last-position logits match the unfused reference (cos≈1) for both
    // Coder (Q4_K gate/up) and Tongyi (Q6_K gate/up); without it, cos≈0.36/-0.02.
    q.memset(ws_ffn_y_, 0, uint64_t(T) * H * sizeof(sycl::half));
    moe_prefill_reduce(q, ws_out_packed_,
                       reinterpret_cast<const uint32_t*>(ws_tk_to_packed_),
                       ws_weights_packed_, ws_ffn_y_, T, K, H);
}

// IE_QWEN3MOE_MOE_XMX per-expert fp16-on-XMX prefill MoE.  For each expert e
// whose token slice is [off_e, off_{e+1}) of the expert-sorted packed rows:
//   gate = gemm_fp16( tokens[n_e×H], dequant(W_gate_e)[H×EF] ) -> fp16
//   up   = gemm_fp16( tokens[n_e×H], dequant(W_up_e)[H×EF]   ) -> fp16
//   h    = silu(gate) * up                                      -> [n_e×EF]
//   out  = gemm_fp16( h[n_e×EF],  dequant(W_down_e)[EF×H] )     -> fp16 [n_e×H]
// `out` is written UNWEIGHTED into ws_out_packed_[off_e..]; the routing weight
// is applied by the shared moe_prefill_reduce (same as the int-dot path — NO
// scatter-add / weighting here).  Each per-expert gemm_fp16 writes at row 0 of a
// round_up(TK,8)-row scratch, so the full-tile row-tail spill lands in unused
// pad rows (never another expert's data); the exact n_e×N result is then copied
// into the globally-packed buffers (contiguous, no spill).
//
// THRESH: experts with n_e <= THRESH are tiny-M; gemm_fp16 still produces the
// CORRECT result for them (it just under-utilizes XMX), so for simplicity we run
// the fp16 GEMM for every n_e>0 expert here rather than mixing with the int-dot
// kernels (which run over ALL experts at once via expert_offsets and cannot be
// cleanly composed per-expert). At prefill T>=512 nearly all experts exceed 32.
void Qwen3MoeModel::moe_xmx_prefill(sycl::queue& q, const Layer& w,
                                    const MoePacking& pk, uint32_t T) {
    const auto& d = cfg_.dense;
    const uint32_t H = d.hidden, EF = cfg_.expert_ffn, E = cfg_.n_experts;
    (void)T;
    constexpr uint32_t THRESH = 32;  // llama MMQ_MAX_BATCH_SIZE (documented; see note)
    (void)THRESH;

    auto dequant_bt = [&](void* bank, DType dt, bool soa, uint32_t e, uint64_t stride,
                          uint32_t Kd, uint32_t Nd, const std::vector<sycl::event>& dep) {
        const void* Wslice = static_cast<const char*>(bank) + uint64_t(e) * stride;
        if (dt == DType::kQ4_K) {
            // gate/up are Q4_K and always AoS on this path (the SoA repack only
            // touches Q6_K down banks); guard anyway.
            (void)soa;
            return dequant_q4_K_to_Bt(q, Wslice, ws_xmx_wbt_, Kd, Nd, dep);
        }
        // Q6_K: SoA banks (default load) vs AoS (IE_NO_MOE_SOA=1). Both produce
        // bit-identical fp16 Bt; only the input plane layout differs.
        return soa
            ? dequant_q6_K_soa_to_Bt(q, Wslice, ws_xmx_wbt_, Kd, Nd, dep)
            : dequant_q6_K_to_Bt(q, Wslice, ws_xmx_wbt_, Kd, Nd, dep);
    };

    for (uint32_t e = 0; e < E; ++e) {
        const uint32_t off0 = pk.expert_offsets[e];
        const uint32_t off1 = pk.expert_offsets[e + 1];
        const uint32_t n_e  = off1 - off0;
        if (n_e == 0) continue;
        const sycl::half* x_e = ws_xp_packed_ + uint64_t(off0) * H;   // [n_e, H]

        // --- gate GEMM: [n_e×H] @ dequant(W_gate)[H×EF] -> ws_xmx_gate_[n_e×EF]
        // NOTE: in-order queue → no per-op .wait() needed; the host enqueues all
        // per-expert ops ahead so the GPU never idles between them (was ~30k host
        // syncs/prefill). The shared scratch (wbt/cf32/gate/up/h/out) is reused per
        // expert; in-order execution serializes it correctly (WAR/RAW satisfied by
        // submit order). Event deps {de}/{ge} are kept (harmless on in-order).
        // GEMMs via oneDNN (gemm_fp16_onednn) — tuned XMX matmul, ~1.65x our
        // gemm_fp16 at these shapes, AND outputs fp16 DIRECTLY (no fp32 cf32
        // scratch + cast_fp32_to_fp16 per GEMM). Same path llama uses for its
        // MoE prefill GEMMs. Single-card only (oneDNN multi-card = DEVICE_LOST,
        // but qwen3moe is single-GPU).
        {
            auto de = dequant_bt(w.gate_exps, w.gate_dt, w.gate_soa, e, w.gate_stride, H, EF, {});
            gemm_fp16_onednn(q, x_e, ws_xmx_wbt_, ws_xmx_gate_, n_e, EF, H, {de});
        }
        // --- up GEMM -> ws_xmx_up_[n_e×EF]
        {
            auto de = dequant_bt(w.up_exps, w.up_dt, w.up_soa, e, w.up_stride, H, EF, {});
            gemm_fp16_onednn(q, x_e, ws_xmx_wbt_, ws_xmx_up_, n_e, EF, H, {de});
        }
        // --- SiLU·mul: h = silu(gate) * up  -> ws_xmx_h_[n_e×EF]
        swiglu(q, ws_xmx_gate_, ws_xmx_up_, ws_xmx_h_, uint64_t(n_e) * EF);
        // --- down GEMM: [n_e×EF] @ dequant(W_down)[EF×H] -> ws_xmx_out_[n_e×H]
        {
            auto de = dequant_bt(w.down_exps, w.down_dt, w.down_soa, e, w.down_stride, EF, H, {});
            gemm_fp16_onednn(q, ws_xmx_h_, ws_xmx_wbt_, ws_xmx_out_, n_e, H, EF, {de});
        }
        // --- copy the exact n_e×H result into the globally-packed out buffer
        //     (UNWEIGHTED — moe_prefill_reduce applies the routing weight).
        q.memcpy(ws_out_packed_ + uint64_t(off0) * H, ws_xmx_out_,
                 uint64_t(n_e) * H * sizeof(sycl::half));
    }
}

// Fused MoE FFN for a single token (decode, T==1). Routes the one token on
// host, uploads topk idx/weights, then runs the two fused decode kernels
// (gate+up+silu, then down+weight+sum-over-K) — 2 launches/layer vs K*4 GEMVs.
void Qwen3MoeModel::moe_ffn_decode(sycl::queue& q, const Layer& w) {
    const auto& d = cfg_.dense;
    const uint32_t H = d.hidden, EF = cfg_.expert_ffn, E = cfg_.n_experts, K = cfg_.n_experts_used;
    q.memcpy(host_xn_.data(), ws_xn_, uint64_t(H) * sizeof(sycl::half)).wait();
    std::vector<std::pair<uint32_t, float>> topk;
    route_token(w.router_w.data(), host_xn_.data(), E, H, K, topk);
    std::vector<int32_t>    idx(K);
    std::vector<sycl::half> wt(K);
    for (uint32_t k = 0; k < K; ++k) { idx[k] = int32_t(topk[k].first); wt[k] = sycl::half(topk[k].second); }
    q.memcpy(ws_topk_idx_, idx.data(), K * sizeof(int32_t));
    q.memcpy(ws_topk_w_,   wt.data(),  K * sizeof(sycl::half)).wait();

    // stage 1: gate+up+silu for K experts -> ws_h_packed_ [K, E_ffn].
    moe_decode_gate_up_silu_q4k(q, ws_xn_, w.gate_exps, w.up_exps, ws_topk_idx_,
                                ws_h_packed_, H, EF, K, w.gate_stride, /*soa=*/false);
    // stage 2: down + weight + sum over K -> ws_ffn_y_ [H] (kernel WRITES; zero first).
    q.memset(ws_ffn_y_, 0, uint64_t(H) * sizeof(sycl::half));
    if (w.down_dt == DType::kQ6_K)
        moe_decode_down_q6k(q, ws_h_packed_, w.down_exps, ws_topk_idx_, ws_topk_w_,
                            ws_ffn_y_, H, EF, K, w.down_stride, /*soa=*/false);
    else
        moe_decode_down_q4k(q, ws_h_packed_, w.down_exps, ws_topk_idx_, ws_topk_w_,
                            ws_ffn_y_, H, EF, K, w.down_stride, /*soa=*/false);
}

sycl::event Qwen3MoeModel::forward(sycl::queue& q, const int32_t* input_ids, uint32_t T,
                                   uint32_t start_pos, KvCache& kv, sycl::half* out_logits) {
    const auto& d = cfg_.dense;
    const uint32_t H = d.hidden, HD = d.head_dim, EF = cfg_.expert_ffn, E = cfg_.n_experts, K = cfg_.n_experts_used;
    const uint32_t N_q = d.n_q_heads * HD, N_kv = d.n_kv_heads * HD;
    const float eps = d.rms_eps;

    // Lazily size the FA-2 decode partials for this ctx so the T==1 fast path is
    // available even when the caller (engine / test tool) did not pre-allocate
    // via ensure_attn_partials. No-op once sized; prefill (T>1) never needs it.
    if (T == 1 && ws_attn_partials_ctx_ < kv.config().max_ctx) {
        if (auto e = ensure_attn_partials(kv.config().max_ctx); !e.empty())
            std::fprintf(stderr, "Qwen3MoeModel::ensure_attn_partials: %s\n", e.c_str());
    }

    std::vector<int32_t> pos(T);
    for (uint32_t t = 0; t < T; ++t) pos[t] = int32_t(start_pos + t);
    q.memcpy(ws_pos_, pos.data(), T * sizeof(int32_t)).wait();

    // embedding
    int32_t* d_ids = static_cast<int32_t*>(alloc_->malloc(T * sizeof(int32_t)));
    q.memcpy(d_ids, input_ids, T * sizeof(int32_t)).wait();
    if (token_embd_dtype_ == DType::kQ4_K) embedding_lookup_q4k(q, d_ids, token_embd_, ws_x_, T, H);
    else                                    embedding_lookup_q6k(q, d_ids, token_embd_, ws_x_, T, H);
    q.wait();
    alloc_->free(d_ids);

    std::vector<std::pair<uint32_t, float>> topk;
    for (uint32_t L = 0; L < d.n_layers; ++L) {
        const Layer& w = layers_[L];
        // --- attention ---
        rms_norm_f32w(q, ws_x_, w.attn_norm, ws_xn_, T, H, eps);
        dense::gemv_q_T(q, ws_xn_, w.attn_q, ws_q_, H, N_q,  T);
        dense::gemv_q_T(q, ws_xn_, w.attn_k, ws_k_, H, N_kv, T);
        dense::gemv_q_T(q, ws_xn_, w.attn_v, ws_v_, H, N_kv, T);
        if (w.attn_q_norm) rms_norm_f32w(q, ws_q_, w.attn_q_norm, ws_q_, T * d.n_q_heads,  HD, eps);
        if (w.attn_k_norm) rms_norm_f32w(q, ws_k_, w.attn_k_norm, ws_k_, T * d.n_kv_heads, HD, eps);
        rope_partial(q, ws_q_, ws_pos_, ws_q_, T, d.n_q_heads,  HD, d.rope_dim, d.rope_theta);
        rope_partial(q, ws_k_, ws_pos_, ws_k_, T, d.n_kv_heads, HD, d.rope_dim, d.rope_theta);
        const uint64_t per_layer = uint64_t(d.n_kv_heads) * kv.config().max_ctx * HD;
        sycl::half* kc = kv.k_ptr() + per_layer * L;
        sycl::half* vc = kv.v_ptr() + per_layer * L;
        // DECODE (T==1) → FA-2 split-K (coalesced KV, no gqa-redundant re-reads);
        // PREFILL (T>1) stays on naive full_attention (oneDNN prefill is Tier-2,
        // not this change). Partials sized lazily for this ctx (no-op if callers
        // pre-allocated via ensure_attn_partials). Mirrors qwen35_dense.cpp:514.
        // PREFILL attention: query-row-block FA-2 v2 above the L2 crossover.
        // Measured (Coder Q4_K_M, 1xB70, 2026-06-22): v2 vs naive prefill =
        // 0.95x @512 / 1.04x @4096 / 2.05x @16384 — naive's redundant KV reads
        // are L2-served up to ~4K (parity) but spill to HBM past it, where v2's
        // SLM tiling halves prefill (317s->146s @16K). Default-ON for T>=2048
        // (the crossover); naive below it (no redundant-HBM to cut). Opt-out
        // IE_NO_FA2_PREFILL_V2; IE_FA2_PREFILL_V2 forces v2 at any T (A/B bench).
        static const bool v2_force = std::getenv("IE_FA2_PREFILL_V2") != nullptr;
        static const bool v2_off   = std::getenv("IE_NO_FA2_PREFILL_V2") != nullptr;
        const bool use_v2 = !v2_off && (v2_force || T >= 2048);
        if (T == 1 && ws_attn_partials_) {
            // DECODE. INT8-KV path (halves KV bandwidth → long-ctx decode win,
            // Coder is pure full-attn so KV dominates decode past a few-K ctx)
            // when the cache is INT8 and fully populated to start_pos; else fp16.
            // Mirrors crown qwen36.cpp:1037. Gated by kv.is_int8() (opts.int8_kv /
            // --int8-kv) → default fp16 path byte-identical.
            if (kv.is_int8() && kv.k_int8_ptr() &&
                start_pos == kv.int8_length(L)) {
                const uint64_t i8_per_layer = uint64_t(d.n_kv_heads) * kv.config().max_ctx * HD;
                const uint64_t sc_per_layer = uint64_t(d.n_kv_heads) * kv.config().max_ctx;
                int8_t*     k_i8 = kv.k_int8_ptr()   + i8_per_layer * L;
                int8_t*     v_i8 = kv.v_int8_ptr()   + i8_per_layer * L;
                sycl::half* k_sc = kv.k_scales_ptr() + sc_per_layer * L;
                sycl::half* v_sc = kv.v_scales_ptr() + sc_per_layer * L;
                full_attention_fa2_decode_int8(q, ws_q_, ws_k_, ws_v_,
                                               k_i8, v_i8, k_sc, v_sc,
                                               nullptr, nullptr,
                                               ws_attn_out_, ws_attn_partials_,
                                               start_pos, d.n_q_heads, d.n_kv_heads,
                                               HD, kv.config().max_ctx);
                kv.set_int8_length(L, start_pos + 1);   // inline-quantized this pos
            } else {
                // Decode-attn variants — gated A/B. IE_FA2_VEC = ground-up llama
                // fattn-vec port (the structural fix for the 3.85x long-ctx gap);
                // IE_FA2_DECODE_V2 = the (flat) split-K defer scaffold.
                static const bool dec_vec = std::getenv("IE_FA2_VEC") != nullptr;
                static const bool dec_v2  = std::getenv("IE_FA2_DECODE_V2") != nullptr;
                if (dec_vec)
                    full_attention_fa2_decode_vec(q, ws_q_, ws_k_, ws_v_, kc, vc,
                                                  ws_attn_out_, ws_attn_partials_,
                                                  start_pos, d.n_q_heads, d.n_kv_heads,
                                                  HD, kv.config().max_ctx);
                else if (dec_v2)
                    full_attention_fa2_decode_v2(q, ws_q_, ws_k_, ws_v_, kc, vc,
                                                 ws_attn_out_, ws_attn_partials_,
                                                 start_pos, d.n_q_heads, d.n_kv_heads,
                                                 HD, kv.config().max_ctx);
                else
                    full_attention_fa2_decode(q, ws_q_, ws_k_, ws_v_, kc, vc,
                                              ws_attn_out_, ws_attn_partials_,
                                              start_pos, d.n_q_heads, d.n_kv_heads,
                                              HD, kv.config().max_ctx);
            }
        } else {
            // PREFILL (T>1): fp16 cache, then post-quantize into the INT8 shadow
            // so the next T=1 decode step can take the INT8 path.
            // DEFAULT prefill attention: the ported llama SIMD "tile" flash-attn-2
            // kernel — faithful register-tiled inner loop (full-lane KQ/softmax/VKQ,
            // Q-resident). Measured vs v2 (Coder 1×B70): 4K 2.0× / 8K 2.4× / 16K 3.1×
            // (grows with ctx). Coherent, crown 6.4527 bit-exact (qwen3moe-gated).
            // Opt out to v2 via IE_NO_FA2_PREFILL_TILE. (tile internally falls through
            // to v2 for head_dim != 128.) IE_FA2_PREFILL_XMX = experimental XMX A/B.
            static const bool xmx_force = std::getenv("IE_FA2_PREFILL_XMX") != nullptr;
            static const bool no_tile   = std::getenv("IE_NO_FA2_PREFILL_TILE") != nullptr;
            // IE_FA2_TILE_GQA selects the GQA head-packing tile variant (default
            // OFF → shipped tile kernel unchanged). It shares ONE K/V SLM tile
            // across G=4 GQA-sibling q-heads of the same kv_head, killing up to
            // gqa× redundant K/V reads + per-head SLM-stage barriers. Numerically
            // identical to the tile kernel. Falls through to v2 if gqa%4!=0 or
            // hd!=128 (e.g. non-GQA arches).
            static const bool tile_gqa = std::getenv("IE_FA2_TILE_GQA") != nullptr;
            // IE_FA2_TILE_REGTILE selects the bounded register-staging KQ-dot
            // tile variant (default OFF → shipped tile kernel unchanged). It
            // restructures only the KQ inner dot to llama-SYCL's bounded
            // chunked staging (CPE=4 half2 chunks + 2 parity accs per dot) to
            // break the single-accumulator carried dependency without a
            // register spill. Numerically matches the tile kernel within fp16
            // tol. Falls through to v2 if hd!=128.
            static const bool tile_regtile = std::getenv("IE_FA2_TILE_REGTILE") != nullptr;
            // The generalized "wide" tile (REGDOT + SMALLBC occupancy) — DEFAULT ON:
            // Coder prefill +6% (4K 1324/8K 921/16K 565 vs hd128-tile 1248/867/535),
            // flips 8K vs llama from parity to WIN (921 vs 849). crown PPL 6.45
            // UNCHANGED (numerically equiv at HD=128). Opt-out IE_QWEN3MOE_NO_WIDE_TILE.
            static const bool wide_tile     = std::getenv("IE_QWEN3MOE_NO_WIDE_TILE") == nullptr;
            if (xmx_force && T > 1) {
                full_attention_fa2_prefill_xmx(q, ws_q_, ws_k_, ws_v_, kc, vc, ws_attn_out_,
                                               T, start_pos, d.n_q_heads, d.n_kv_heads,
                                               HD, kv.config().max_ctx);
            } else if (tile_regtile && !no_tile && T > 1) {
                full_attention_fa2_prefill_tile_regtile(q, ws_q_, ws_k_, ws_v_, kc, vc, ws_attn_out_,
                                                        T, start_pos, d.n_q_heads, d.n_kv_heads,
                                                        HD, kv.config().max_ctx);
            } else if (tile_gqa && !no_tile && T > 1) {
                full_attention_fa2_prefill_tile_gqa(q, ws_q_, ws_k_, ws_v_, kc, vc, ws_attn_out_,
                                                    T, start_pos, d.n_q_heads, d.n_kv_heads,
                                                    HD, kv.config().max_ctx);
            } else if (wide_tile && !no_tile && T > 1) {
                // A/B: the generalized "wide" tile kernel at HD=128 (+ REGDOT default,
                // + SMALLBC occupancy via IE_GEMMA4_NO_TILE_SMALLBC off). Tests whether
                // the Gemma occupancy/regdot levers lift Coder's attention-bound MoE
                // prefill (8K parity / 16K behind llama). Env IE_QWEN3MOE_WIDE_TILE.
                full_attention_fa2_prefill_tile_gemma(q, ws_q_, ws_k_, ws_v_, kc, vc, ws_attn_out_,
                                                      T, start_pos, d.n_q_heads, d.n_kv_heads,
                                                      HD, kv.config().max_ctx);
            } else if (!no_tile && T > 1) {
                full_attention_fa2_prefill_tile(q, ws_q_, ws_k_, ws_v_, kc, vc, ws_attn_out_,
                                                T, start_pos, d.n_q_heads, d.n_kv_heads,
                                                HD, kv.config().max_ctx);
            } else if (use_v2) {
                full_attention_fa2_prefill_v2(q, ws_q_, ws_k_, ws_v_, kc, vc, ws_attn_out_,
                                              T, start_pos, d.n_q_heads, d.n_kv_heads,
                                              HD, kv.config().max_ctx);
            } else {
                full_attention(q, ws_q_, ws_k_, ws_v_, kc, vc, ws_attn_out_, T, start_pos,
                               d.n_q_heads, d.n_kv_heads, HD, kv.config().max_ctx);
            }
            if (kv.is_int8()) kv.quantize_to_int8(q, L, start_pos, T);
        }
        kv.set_length(L, start_pos + T);
        dense::gemv_q_T(q, ws_attn_out_, w.attn_output, ws_block_, N_q, H, T);

        // --- MoE FFN ---
        // OPT-IN fusion of the attn-out residual_add + ffn_norm into ONE pass:
        // 1 fewer kernel launch + one fewer HBM write/read-back of ws_x per layer.
        // Measured +2-3% decode (55.3 vs 54.2 tok/s, new-old-new). NOT default
        // because the fused 2-pass fp32 accumulation is not bit-identical to the
        // split residual_add→rms_norm_f32w (FMA/rounding differences inherent to
        // the single-kernel form) → Coder PPL 11.98→12.02 (+0.33%). The DEFAULT
        // path stays the lossless split. Enable: IE_QWEN3MOE_RES_RMS_FUSED. The
        // residual decode gap to llama is commodity intra-kernel efficiency, not
        // launch count (see docs/authority/coder.md §8). [2026-06-26]
        static const bool fuse_res_rms = std::getenv("IE_QWEN3MOE_RES_RMS_FUSED") != nullptr;
        if (fuse_res_rms) {
            residual_add_rms_norm_fused(q, ws_x_, ws_block_, w.ffn_norm, ws_xn_, T, H, eps);
        } else {
            residual_add(q, ws_x_, ws_block_, ws_x_, uint64_t(T) * H);
            rms_norm_f32w(q, ws_x_, w.ffn_norm, ws_xn_, T, H, eps);
        }
        static const bool unfused_env = std::getenv("IE_QWEN3MOE_UNFUSED") != nullptr;
        static const bool no_q8_env    = std::getenv("IE_QWEN3MOE_NO_Q8") != nullptr;
        // Gate/up fusability by dtype. Q4_K gate/up fuse on BOTH the int-dot q8
        // path and the fp16-activation fallback. Q6_K gate/up (bartowski Q6_K of
        // Qwen3-30B-A3B, e.g. Tongyi-DeepResearch) fuse ONLY on the int-dot W6A8
        // path (moe_prefill_gate_up_silu_q6k_q8) — there is no fp16-activation
        // Q6_K gate/up kernel — so a Q6_K gate/up model needs the q8 path
        // (H%512==0 and NOT IE_QWEN3MOE_NO_Q8); otherwise it routes to the
        // dtype-agnostic per-expert gemv reference (dense::gemv_q, AoS banks).
        // This condition matches the load-time moe_exps_soa_ preconditions, so the
        // INVARIANT holds: SoA-on ⟺ fused path (the reference reads AoS only when
        // SoA is off). Any other gate/up dtype (no int-dot kernel) → unfused.
        const bool gateup_q4k = (w.gate_dt == DType::kQ4_K && w.up_dt == DType::kQ4_K);
        const bool gateup_q6k = (w.gate_dt == DType::kQ6_K && w.up_dt == DType::kQ6_K);
        const bool q8_avail   = !no_q8_env && (H % 512 == 0);
        const bool unfused = unfused_env || !(gateup_q4k || (gateup_q6k && q8_avail));
        // T==1 decode routes through the int-dot W4A8 fused path by DEFAULT (task #6,
        // 2026-06-14): pack the single token's K experts as K rows → gather +
        // gate_up_silu_q8 + down_q8_gen + reduce. At qwen3moe's top-8 / E_ffn=768 shape
        // this is +28% faster than the crown fp16-activation moe_ffn_decode (37.5→48
        // tok/s, heat-robust F/D/F/D), PPL-identical — the crown decode kernels were
        // tuned for top-2; the fused _gen kernels pack top-8 far better. (Unlocked by the
        // ws_ffn_y_ memset in moe_ffn_fused_prefill — without it the reduce accumulated
        // onto stale state → garbage; ie-bench's "win" was timing that garbage, caught
        // only by the PPL/chat gate.) Opt out to fp16 decode via IE_QWEN3MOE_FP16_DECODE=1.
        static const bool fp16_decode = std::getenv("IE_QWEN3MOE_FP16_DECODE") != nullptr;
        static const bool ondevice_route_d = std::getenv("IE_QWEN3MOE_ONDEVICE_ROUTE") != nullptr;
        // fp16-decode opt-out (moe_ffn_decode) is Q4_K-only, so Q6_K gate/up always
        // takes the int-dot fused path at T==1 too (correct on AoS banks when SoA
        // is off). The `else if` below is therefore reached only for Q4_K gate/up.
        if (!unfused && (T > 1 || !fp16_decode || gateup_q6k)) {
            // fused (int-dot) path — prefill AND, by default, T==1 decode
            moe_ffn_fused_prefill(q, w, T);
            residual_add(q, ws_x_, ws_ffn_y_, ws_x_, uint64_t(T) * H);
            // On-device routing keeps the next layer's router on-GPU, so the
            // per-layer host sync is unnecessary — the token's 48 layers enqueue
            // and the GPU runs ahead. The final logits read syncs before sampling.
            if (!(ondevice_route_d && T == 1)) q.wait();
        } else if (!unfused) {   // T == 1, fp16 decode opt-out (Q4_K gate/up only)
            moe_ffn_decode(q, w);
            residual_add(q, ws_x_, ws_ffn_y_, ws_x_, uint64_t(H));
            q.wait();   // host router for the next layer needs this layer finished
        } else {
            // unfused reference path (parity oracle; also T==1 until Task 4)
            q.memcpy(host_xn_.data(), ws_xn_, uint64_t(T) * H * sizeof(sycl::half)).wait();
            for (uint32_t t = 0; t < T; ++t) {
                route_token(w.router_w.data(), host_xn_.data() + uint64_t(t) * H, E, H, K, topk);
                sycl::half* xn_t = ws_xn_ + uint64_t(t) * H;
                q.memset(ws_acc_, 0, uint64_t(H) * sizeof(sycl::half));
                for (const auto& [e, wt] : topk) {
                    DenseQuantPtr gate{static_cast<char*>(w.gate_exps) + e * w.gate_stride, w.gate_dt};
                    DenseQuantPtr up  {static_cast<char*>(w.up_exps)   + e * w.up_stride,   w.up_dt};
                    DenseQuantPtr down{static_cast<char*>(w.down_exps) + e * w.down_stride, w.down_dt};
                    dense::gemv_q(q, xn_t, gate, ws_gate_, H, EF);
                    dense::gemv_q(q, xn_t, up,   ws_up_,   H, EF);
                    swiglu(q, ws_gate_, ws_up_, ws_eh_, EF);
                    dense::gemv_q(q, ws_eh_, down, ws_eout_, EF, H);
                    scaled_add(q, ws_eout_, sycl::half(wt), ws_acc_, H);
                }
                residual_add(q, ws_x_ + uint64_t(t) * H, ws_acc_, ws_x_ + uint64_t(t) * H, H);
            }
            q.wait();   // host router for the next layer needs this layer finished
        }
    }

    // final norm + lm_head
    rms_norm_f32w(q, ws_x_, output_norm_, ws_xn_, T, H, eps);
    const sycl::half* last = ws_xn_ + uint64_t(T - 1) * H;
    DenseQuantPtr lm{output_, output_dtype_};
    return dense::gemv_q(q, last, lm, out_logits, H, d.vocab);
}

}  // namespace ie
