// src/model/gptoss.cpp — gpt-oss (OpenAI MoE) forward implementation.
// Structural fork of qwen3moe (top-k MoE + dense GQA) with the gpt-oss deltas:
// q/k/v/o + router + expert BIASES, per-head attention SINKS, alternating
// sliding-window (even layers), MXFP4 experts (oneDNN-only), clamped gated
// SwiGLU (OAI), and NO QK-norm. See docs/gptoss_arch_spec_2026-06-26.md.
#include "ie/gptoss.hpp"

#include "ie/dequant.hpp"        // dequant_mxfp4_to_Bt (per-expert fp16 Bt)
#include "ie/ops.hpp"            // rms_norm_f32w, add_bias, rope_partial, FA kernels, swiglu, embedding_lookup_*
#include "ie/quant_blocks.hpp"   // block_mxfp4

#include "dense_dispatch.hpp"    // dense::upload<T>, dense::upload_quant_dense_auto

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

namespace ie {

namespace {
// gpt-oss router = SOFTMAX_WEIGHT (llama-graph.cpp:1502-1578): logits (with router
// bias already added) → top-k by raw logit → softmax over the k selected logits.
// This is mathematically identical to "softmax over all E → top-k → renormalize
// by the sum of the k probs" (the global normalization cancels), which is what we
// compute here (more numerically stable: subtracts the global max). Output is
// sorted ascending by expert id (HF / build_moe_packing iteration order).
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
}  // namespace

GptOssModel::~GptOssModel() { free_all(); }

void GptOssModel::free_all() {
    if (!alloc_) return;
    for (void* p : owned_) alloc_->free(p);
    owned_.clear();
}

// ---------------------------------------------------------------------------
// load
// ---------------------------------------------------------------------------
std::string GptOssModel::load(DeviceAllocator& alloc, const GgufReader& g,
                              const GptOssConfig& cfg) {
    alloc_ = &alloc;
    cfg_   = cfg;

    const auto& d = cfg.dense;
    const uint32_t E = cfg_.n_experts;
    char buf[64];
    std::string err;
    auto T  = [&](const char* n) { return g.find_tensor(n); };
    auto Tl = [&](uint32_t L, const char* n) {
        std::snprintf(buf, sizeof(buf), "blk.%u.%s", L, n);
        return g.find_tensor(buf);
    };
    // load a required F32 vector (norms / biases / sinks) to device.
    auto F32 = [&](const GgufTensorInfo* ti, const char* nm, float*& dst) -> std::string {
        if (!ti) return std::string(nm) + ": not found";
        if (ti->dtype != DType::kF32) return std::string(nm) + ": expected F32";
        dst = dense::upload<float>(alloc, ti, owned_, err, DType::kF32);
        return err.empty() ? std::string{} : std::string(nm) + ": " + err;
    };

    // ---- globals ----
    {
        const auto* ti = T("token_embd.weight");
        if (!ti) return "token_embd: not found";
        // gpt-oss ships token_embd as Q8_0 (embedding_lookup_q8_0 consumes it raw).
        if (ti->dtype != DType::kQ8_0 && ti->dtype != DType::kQ4_K && ti->dtype != DType::kQ6_K)
            return "token_embd: unsupported dtype (want Q8_0/Q4_K/Q6_K)";
        token_embd_dtype_ = ti->dtype;
        token_embd_ = dense::upload<void>(alloc, ti, owned_, err, ti->dtype);
        if (!err.empty()) return "token_embd: " + err;
    }
    output_norm_ = dense::upload<float>(alloc, T("output_norm.weight"), owned_, err, DType::kF32);
    if (!err.empty()) return "output_norm: " + err;
    {
        // lm_head (Q8_0, separate/untied).  DEFAULT = Q8_0 SoA repack → W8A8 int-dot
        // gemv (half the fp16 read; the lm_head is the single biggest decode gemv at
        // 201k vocab).  Opt-out IE_GPTOSS_NO_Q8_LMHEAD → the F16 path (A/B baseline).
        const auto* ti = T("output.weight");
        if (!ti) return "output.weight: not found";
        static const bool q8_lmhead = std::getenv("IE_GPTOSS_NO_Q8_LMHEAD") == nullptr;
        if (q8_lmhead && ti->dtype == DType::kQ8_0 && ti->n_dims == 2) {
            const uint32_t K = uint32_t(ti->shape[0]), N = uint32_t(ti->shape[1]);
            if (K % 32) return "output: K not %32";
            const uint32_t bpc = K / 32;
            const auto* blocks = reinterpret_cast<const block_q8_0*>(ti->data);
            std::vector<int8_t>   qs(uint64_t(N) * K);
            std::vector<uint16_t> dd(uint64_t(N) * bpc);
            for (uint64_t n = 0; n < N; ++n)
                for (uint32_t b = 0; b < bpc; ++b) {
                    const block_q8_0& blk = blocks[n * bpc + b];
                    dd[n * bpc + b] = *reinterpret_cast<const uint16_t*>(&blk.d);
                    for (int i = 0; i < 32; ++i)
                        qs[n * uint64_t(K) + uint64_t(b) * 32 + i] = blk.qs[i];
                }
            lmhead_qs_ = static_cast<int8_t*>(alloc.malloc(qs.size()));
            lmhead_d_  = static_cast<uint16_t*>(alloc.malloc(dd.size() * sizeof(uint16_t)));
            if (!lmhead_qs_ || !lmhead_d_) return "output: SoA alloc failed";
            owned_.push_back(lmhead_qs_); owned_.push_back(lmhead_d_);
            alloc.queue().memcpy(lmhead_qs_, qs.data(), qs.size());
            alloc.queue().memcpy(lmhead_d_, dd.data(), dd.size() * sizeof(uint16_t)).wait();
            lmhead_K_ = K; lmhead_N_ = N;
        } else {
            output_ = dense::upload_quant_dense_auto(alloc, ti, owned_, err);
            if (!err.empty()) return "output: " + err;
        }
    }

    layers_.assign(d.n_layers, {});
    for (uint32_t L = 0; L < d.n_layers; ++L) {
        Layer& w = layers_[L];
        w.is_swa = (L % 2 == 0);   // gpt-oss: even layers windowed (set_swa_pattern(2), dense_first=false)

        // norms
        if (auto m = F32(Tl(L, "attn_norm.weight"), "attn_norm", w.attn_norm); !m.empty()) return m;
        if (auto m = F32(Tl(L, "post_attention_norm.weight"), "post_attention_norm", w.post_attn_norm); !m.empty()) return m;

        // attention projections (Q8_0 → F16) + biases (F32)
        w.attn_q = dense::upload_quant_dense_auto(alloc, Tl(L, "attn_q.weight"), owned_, err);
        if (!err.empty()) return "attn_q: " + err;
        w.attn_k = dense::upload_quant_dense_auto(alloc, Tl(L, "attn_k.weight"), owned_, err);
        if (!err.empty()) return "attn_k: " + err;
        w.attn_v = dense::upload_quant_dense_auto(alloc, Tl(L, "attn_v.weight"), owned_, err);
        if (!err.empty()) return "attn_v: " + err;
        w.attn_output = dense::upload_quant_dense_auto(alloc, Tl(L, "attn_output.weight"), owned_, err);
        if (!err.empty()) return "attn_output: " + err;
        if (auto m = F32(Tl(L, "attn_q.bias"),      "attn_q.bias",      w.attn_q_bias); !m.empty()) return m;
        if (auto m = F32(Tl(L, "attn_k.bias"),      "attn_k.bias",      w.attn_k_bias); !m.empty()) return m;
        if (auto m = F32(Tl(L, "attn_v.bias"),      "attn_v.bias",      w.attn_v_bias); !m.empty()) return m;
        if (auto m = F32(Tl(L, "attn_output.bias"), "attn_output.bias", w.attn_o_bias); !m.empty()) return m;
        if (auto m = F32(Tl(L, "attn_sinks.weight"),"attn_sinks",       w.attn_sinks);  !m.empty()) return m;

        // router: host F32 [E,hidden] + transposed F16 [hidden,E] device + bias F32[E]
        {
            const auto* ti = Tl(L, "ffn_gate_inp.weight");
            if (!ti) return "ffn_gate_inp: not found (layer " + std::to_string(L) + ")";
            if (ti->dtype != DType::kF32) return "ffn_gate_inp: expected F32";
            if (ti->nbytes != uint64_t(E) * d.hidden * sizeof(float))
                return "ffn_gate_inp: unexpected size";
            w.router_w.resize(uint64_t(E) * d.hidden);
            std::memcpy(w.router_w.data(), ti->data, ti->nbytes);
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
        if (auto m = F32(Tl(L, "ffn_gate_inp.bias"), "ffn_gate_inp.bias", w.router_bias); !m.empty()) return m;

        // experts (MXFP4) → load-time SoA repack: split each 17-byte block into an
        // aligned nibble plane (qs, 16 B/block) + an E8M0 exponent plane (e, 1 B/block).
        // Same 4.25 bpw total; coalesced/aligned reads for the decode int-dot GEMV +
        // the prefill SoA dequant.  Per-expert strides: qs = N*(K/2), e = N*(K/32).
        auto repack_exps = [&](const char* nm, uint32_t Kdim, uint32_t Ndim,
                               uint8_t*& qs_dev, uint8_t*& e_dev,
                               uint64_t& qs_stride, uint64_t& e_stride) -> std::string {
            const auto* ti = Tl(L, nm);
            if (!ti) return std::string(nm) + ": not found";
            if (ti->dtype != DType::kMXFP4)
                return std::string(nm) + ": gpt-oss experts must be MXFP4";
            const uint32_t Bpc = Kdim / kQK_MXFP4;                 // blocks per column
            const uint64_t bpe = uint64_t(Ndim) * Bpc;            // blocks per expert
            if (ti->nbytes != uint64_t(E) * bpe * sizeof(block_mxfp4))
                return std::string(nm) + ": size mismatch (expect E*N*K/32 blocks)";
            qs_stride = uint64_t(Ndim) * (uint64_t(Kdim) / 2);    // bytes/expert (qs)
            e_stride  = bpe;                                       // bytes/expert (e)
            std::vector<uint8_t> qs_host(uint64_t(E) * qs_stride);
            std::vector<uint8_t> e_host(uint64_t(E) * e_stride);
            const auto* src = reinterpret_cast<const block_mxfp4*>(ti->data);
            for (uint64_t i = 0; i < uint64_t(E) * bpe; ++i) {
                const uint64_t ex = i / bpe, within = i % bpe;
                std::memcpy(qs_host.data() + ex * qs_stride + within * 16, src[i].qs, 16);
                e_host[ex * e_stride + within] = src[i].e;
            }
            qs_dev = static_cast<uint8_t*>(alloc.malloc(qs_host.size()));
            e_dev  = static_cast<uint8_t*>(alloc.malloc(e_host.size()));
            if (!qs_dev || !e_dev) return std::string(nm) + ": SoA alloc failed";
            owned_.push_back(qs_dev); owned_.push_back(e_dev);
            alloc.queue().memcpy(qs_dev, qs_host.data(), qs_host.size());
            alloc.queue().memcpy(e_dev,  e_host.data(),  e_host.size()).wait();
            return {};
        };
        if (auto m = repack_exps("ffn_gate_exps.weight", d.hidden, cfg_.expert_ffn,
                                 w.gate_qs, w.gate_e, w.gate_qs_stride, w.gate_e_stride); !m.empty()) return m;
        if (auto m = repack_exps("ffn_up_exps.weight",   d.hidden, cfg_.expert_ffn,
                                 w.up_qs, w.up_e, w.up_qs_stride, w.up_e_stride); !m.empty()) return m;
        if (auto m = repack_exps("ffn_down_exps.weight", cfg_.expert_ffn, d.hidden,
                                 w.down_qs, w.down_e, w.down_qs_stride, w.down_e_stride); !m.empty()) return m;
        // per-expert biases (F32): gate/up [E_ffn,E], down [hidden,E]
        if (auto m = F32(Tl(L, "ffn_gate_exps.bias"), "ffn_gate_exps.bias", w.gate_bias); !m.empty()) return m;
        if (auto m = F32(Tl(L, "ffn_up_exps.bias"),   "ffn_up_exps.bias",   w.up_bias);   !m.empty()) return m;
        if (auto m = F32(Tl(L, "ffn_down_exps.bias"), "ffn_down_exps.bias", w.down_bias); !m.empty()) return m;
    }
    return {};
}

// ---------------------------------------------------------------------------
// workspace
// ---------------------------------------------------------------------------
std::string GptOssModel::ensure_workspace(uint32_t max_T) {
    if (max_T <= ws_T_) return {};
    const auto& d = cfg_.dense;
    const uint32_t H   = d.hidden;
    const uint32_t HD  = d.head_dim;
    const uint32_t N_q = d.n_q_heads * HD;
    const uint32_t N_kv= d.n_kv_heads * HD;
    const uint32_t EF  = cfg_.expert_ffn;
    const uint32_t E   = cfg_.n_experts;
    const uint32_t K   = cfg_.n_experts_used;
    const uint64_t TK  = uint64_t(max_T) * K;

    auto ah = [&](uint64_t n) -> sycl::half* {
        auto* p = static_cast<sycl::half*>(alloc_->malloc(n * sizeof(sycl::half)));
        if (p) owned_.push_back(p);
        return p;
    };
    ws_x_         = ah(uint64_t(max_T) * H);
    ws_xn_        = ah(uint64_t(max_T) * H);
    ws_block_     = ah(uint64_t(max_T) * H);
    ws_q_         = ah(uint64_t(max_T) * N_q);
    ws_attn_out_  = ah(uint64_t(max_T) * N_q);
    ws_k_         = ah(uint64_t(max_T) * N_kv);
    ws_v_         = ah(uint64_t(max_T) * N_kv);
    ws_pos_       = static_cast<int32_t*>(alloc_->malloc(uint64_t(max_T) * sizeof(int32_t)));
    if (ws_pos_) owned_.push_back(ws_pos_);

    ws_moe_logits_ = ah(uint64_t(max_T) * E);
    ws_moe_xp_     = ah(TK * H);
    ws_moe_h_      = ah(TK * EF);
    ws_moe_h2_     = ah(TK * EF);
    ws_moe_out_    = ah(TK * H);
    ws_moe_y_      = ah(uint64_t(max_T) * H);
    ws_moe_wpk_    = ah(TK);
    ws_moe_btf16_  = ah(uint64_t(H) * EF);
    // decode (T=1) int-dot activation scratch (block_q8_1x: 32 elems/40 B).  Sized by
    // H/EF (T-independent) → allocate once.  gate/up share ws_q8_x_; down uses ws_q8_h_.
    if (!ws_q8_x_) {
        ws_q8_x_ = alloc_->malloc(uint64_t(H  / 32) * sizeof(block_q8_1x));
        ws_q8_h_ = alloc_->malloc(uint64_t(EF / 32) * sizeof(block_q8_1x));
        if (ws_q8_x_) owned_.push_back(ws_q8_x_);
        if (ws_q8_h_) owned_.push_back(ws_q8_h_);
    }
    ws_moe_offsets_ = static_cast<uint32_t*>(alloc_->malloc(uint64_t(E + 1) * sizeof(uint32_t)));
    if (ws_moe_offsets_) owned_.push_back(ws_moe_offsets_);
    ws_moe_sorted_ = static_cast<int32_t*>(alloc_->malloc(TK * sizeof(int32_t)));
    if (ws_moe_sorted_) owned_.push_back(ws_moe_sorted_);
    ws_moe_tk2pk_  = static_cast<int32_t*>(alloc_->malloc(TK * sizeof(int32_t)));
    if (ws_moe_tk2pk_) owned_.push_back(ws_moe_tk2pk_);

    if (!ws_x_ || !ws_xn_ || !ws_block_ || !ws_q_ || !ws_attn_out_ || !ws_k_ || !ws_v_ ||
        !ws_pos_ || !ws_moe_logits_ || !ws_moe_xp_ || !ws_moe_h_ || !ws_moe_h2_ ||
        !ws_moe_out_ || !ws_moe_y_ || !ws_moe_wpk_ || !ws_moe_btf16_ ||
        !ws_q8_x_ || !ws_q8_h_ ||
        !ws_moe_offsets_ || !ws_moe_sorted_ || !ws_moe_tk2pk_)
        return "gptoss workspace alloc failed";

    host_logits_.resize(uint64_t(max_T) * E);
    host_routes_.resize(max_T);
    ws_T_ = max_T;
    ws_T_moe_ = max_T;
    return {};
}

std::string GptOssModel::ensure_attn_partials(uint32_t max_ctx) {
    if (max_ctx <= ws_attn_partials_ctx_) return {};
    const auto& d = cfg_.dense;
    // [n_chunks_max, n_q_heads, head_dim+2] FP32. MUST use Bc_floor=64 (the FA-2
    // decode kernel's Bc) — the combine writes up to n_super_chunks (≤ TARGET_SUPER=64
    // ≤ n_chunks@Bc64) partials; a /256 sizing under-allocates 4× and overflows at
    // smaller --ctx. Matches qwen3moe::ensure_attn_partials exactly.
    constexpr uint32_t Bc_floor = 64;
    const uint32_t n_chunks = (max_ctx + Bc_floor - 1) / Bc_floor;
    const uint64_t n = uint64_t(n_chunks) * d.n_q_heads * (d.head_dim + 2);
    ws_attn_partials_ = static_cast<float*>(alloc_->malloc(n * sizeof(float)));
    if (!ws_attn_partials_) return "gptoss attn_partials alloc failed";
    owned_.push_back(ws_attn_partials_);
    ws_attn_partials_ctx_ = max_ctx;
    return {};
}

// ---------------------------------------------------------------------------
// MoE FFN (top-4, MXFP4 experts). Unified per-expert oneDNN path for all T —
// MXFP4 has no int-dot kernel, so experts always dequant→fp16→GEMM. Reads
// post-norm activations from ws_xn_, leaves the per-token output in ws_moe_y_.
// ---------------------------------------------------------------------------
void GptOssModel::moe_ffn(sycl::queue& q, const Layer& w, uint32_t T) {
    const auto& d = cfg_.dense;
    const uint32_t H = d.hidden, EF = cfg_.expert_ffn, E = cfg_.n_experts, K = cfg_.n_experts_used;
    const uint32_t TK = T * K;
    // Decode (T==1): replace the per-expert dequant_mxfp4_to_Bt + M=1 oneDNN GEMM
    // (which materialized fp16 and re-read it — ~10.8 GB/token) with a fused MXFP4
    // GEMV over the SoA planes that reads the 4.25-bpw weights once.  At T==1 each
    // active expert has exactly one row (n_e==1), which the M=1 gemv requires.
    //   default        : W4A8 int-dot (gemv_mxfp4_soa_q8) — the perf lever (PPL-gated).
    //   IE_GPTOSS_MXFP4_F16 : bit-faithful fp16-act (gemv_mxfp4_soa_f16) — correctness ref.
    //   IE_GPTOSS_NO_MXFP4_GEMV : old SoA-dequant + oneDNN path (the A/B baseline).
    static const bool mxfp4_gemv = std::getenv("IE_GPTOSS_NO_MXFP4_GEMV") == nullptr;
    static const bool mxfp4_f16  = std::getenv("IE_GPTOSS_MXFP4_F16") != nullptr;
    const bool decode_gemv = mxfp4_gemv && T == 1;

    // 1. router logits = ws_xn_ @ router_wᵀ  (+ router bias, before top-k).
    // PREFILL (T>1): batch via oneDNN — like the attn projections, dense::gemv_q_T's
    // gemm branch is gated N%64==0 AND K%256==0, both of which the router (N=E=32,
    // K=H=2880) fails → it ran as a per-TOKEN gemv (49k calls = 54% of prefill after
    // the q/k/v/o fix). oneDNN handles N=32/K=2880. Decode (T==1) keeps the gemv.
    static const bool router_gemm = std::getenv("IE_GPTOSS_NO_PROJ_GEMM") == nullptr;
    if (router_gemm && T > 1)
        gemm_fp16_onednn(q, ws_xn_, w.router_w_dev, ws_moe_logits_, T, E, H, {});
    else
        dense::gemv_q_T(q, ws_xn_, DenseQuantPtr{w.router_w_dev, DType::kF16},
                        ws_moe_logits_, H, E, T);
    add_bias(q, ws_moe_logits_, w.router_bias, T, E);
    std::vector<sycl::half> hl(uint64_t(T) * E);
    q.memcpy(hl.data(), ws_moe_logits_, uint64_t(T) * E * sizeof(sycl::half)).wait();
    for (uint64_t i = 0; i < uint64_t(T) * E; ++i) host_logits_[i] = float(hl[i]);
    // size to EXACTLY T — build_moe_packing uses routes.size() as the token count,
    // so leftover stale routes from a prior (larger-T) call would corrupt the
    // packing (the decode-after-prefill garbage bug).
    host_routes_.resize(T);
    for (uint32_t t = 0; t < T; ++t)
        route_from_logits(host_logits_.data() + uint64_t(t) * E, E, K, host_routes_[t]);

    // 2. host counting-sort packing + upload device arrays
    build_moe_packing(host_routes_, E, K, pk_);
    q.memcpy(ws_moe_sorted_, pk_.sorted_idx.data(),    TK * sizeof(int32_t));
    q.memcpy(ws_moe_tk2pk_,  pk_.tk_to_packed.data(),  TK * sizeof(int32_t));
    std::vector<sycl::half> wpk(TK);
    for (uint32_t i = 0; i < TK; ++i) wpk[i] = sycl::half(pk_.weights_packed[i]);
    q.memcpy(ws_moe_wpk_, wpk.data(), TK * sizeof(sycl::half)).wait();

    // 3. gather expert-sorted input rows: ws_moe_xp_[i] = ws_xn_[sorted_idx[i]]
    moe_gather_rows(q, ws_xn_, ws_moe_sorted_, ws_moe_xp_, TK, H);

    const auto& off = pk_.expert_offsets;   // host [E+1]
    // At T==1 every active expert's input row is the SAME post-norm token (the gather
    // copies ws_xn_[0] into all K packed rows) → quantize it to Q8 ONCE for all
    // gate/up GEMVs (the int-dot decode path).
    if (decode_gemv && !mxfp4_f16) quantize_q8_1(q, ws_xn_, ws_q8_x_, H);
    // 4. per-expert gate + up.
    for (uint32_t e = 0; e < E; ++e) {
        const uint32_t o = off[e], n_e = off[e + 1] - o;
        if (n_e == 0) continue;
        const sycl::half* x_e = ws_moe_xp_ + uint64_t(o) * H;
        sycl::half* g_o = ws_moe_h_  + uint64_t(o) * EF;
        sycl::half* u_o = ws_moe_h2_ + uint64_t(o) * EF;
        const uint8_t* gqs = w.gate_qs + uint64_t(e) * w.gate_qs_stride;
        const uint8_t* ge  = w.gate_e  + uint64_t(e) * w.gate_e_stride;
        const uint8_t* uqs = w.up_qs   + uint64_t(e) * w.up_qs_stride;
        const uint8_t* ue  = w.up_e    + uint64_t(e) * w.up_e_stride;
        if (decode_gemv) {              // n_e==1: fused MXFP4 GEMV
            if (mxfp4_f16) {
                gemv_mxfp4_soa_f16(q, x_e, gqs, ge, g_o, H, EF, {});
                gemv_mxfp4_soa_f16(q, x_e, uqs, ue, u_o, H, EF, {});
            } else {
                gemv_mxfp4_soa_q8(q, ws_q8_x_, gqs, ge, g_o, H, EF, {});
                gemv_mxfp4_soa_q8(q, ws_q8_x_, uqs, ue, u_o, H, EF, {});
            }
            add_bias(q, g_o, w.gate_bias + uint64_t(e) * EF, n_e, EF);
            add_bias(q, u_o, w.up_bias   + uint64_t(e) * EF, n_e, EF);
            continue;
        }
        auto dg = dequant_mxfp4_soa_to_Bt(q, gqs, ge, ws_moe_btf16_, H, EF, {});
        gemm_fp16_onednn(q, x_e, ws_moe_btf16_, g_o, n_e, EF, H, {dg});
        add_bias(q, g_o, w.gate_bias + uint64_t(e) * EF, n_e, EF);
        auto du = dequant_mxfp4_soa_to_Bt(q, uqs, ue, ws_moe_btf16_, H, EF, {});
        gemm_fp16_onednn(q, x_e, ws_moe_btf16_, u_o, n_e, EF, H, {du});
        add_bias(q, u_o, w.up_bias + uint64_t(e) * EF, n_e, EF);
    }
    // 5. clamped gated-SwiGLU (OAI): gate=ws_moe_h_, up=ws_moe_h2_ → ws_moe_h_.
    swiglu_oai(q, ws_moe_h_, ws_moe_h2_, ws_moe_h_, uint64_t(TK) * EF, 1.702f, 7.0f);
    // 6. per-expert down + bias (bias is part of the per-expert output → weighted by
    //    the routing weight in the reduce, matching llama weight_before_ffn=false).
    for (uint32_t e = 0; e < E; ++e) {
        const uint32_t o = off[e], n_e = off[e + 1] - o;
        if (n_e == 0) continue;
        const sycl::half* h_e = ws_moe_h_ + uint64_t(o) * EF;
        sycl::half* d_o = ws_moe_out_ + uint64_t(o) * H;
        const uint8_t* dqs = w.down_qs + uint64_t(e) * w.down_qs_stride;
        const uint8_t* de  = w.down_e  + uint64_t(e) * w.down_e_stride;
        if (decode_gemv) {
            if (mxfp4_f16) {
                gemv_mxfp4_soa_f16(q, h_e, dqs, de, d_o, EF, H, {});
            } else {
                // h_e differs per expert → quantize each before the down GEMV.
                quantize_q8_1(q, h_e, ws_q8_h_, EF);
                gemv_mxfp4_soa_q8(q, ws_q8_h_, dqs, de, d_o, EF, H, {});
            }
            add_bias(q, d_o, w.down_bias + uint64_t(e) * H, n_e, H);
            continue;
        }
        auto dd = dequant_mxfp4_soa_to_Bt(q, dqs, de, ws_moe_btf16_, EF, H, {});
        gemm_fp16_onednn(q, h_e, ws_moe_btf16_, d_o, n_e, H, EF, {dd});
        add_bias(q, d_o, w.down_bias + uint64_t(e) * H, n_e, H);
    }
    // 7. weighted reduce per token → ws_moe_y_ [T,H]
    q.memset(ws_moe_y_, 0, uint64_t(T) * H * sizeof(sycl::half));
    moe_prefill_reduce(q, ws_moe_out_,
                       reinterpret_cast<const uint32_t*>(ws_moe_tk2pk_),
                       ws_moe_wpk_, ws_moe_y_, T, K, H);
    q.wait();
}

// ---------------------------------------------------------------------------
// forward — embedding → n × { attn(sinks+SWA) → +res → MoE → +res } → norm → lm_head
// ---------------------------------------------------------------------------
sycl::event GptOssModel::forward(sycl::queue& q, const int32_t* input_ids, uint32_t T,
                                 uint32_t start_pos, KvCache& kv, sycl::half* out_logits) {
    const auto& d = cfg_.dense;
    const uint32_t H = d.hidden, HD = d.head_dim;
    const uint32_t N_q = d.n_q_heads * HD, N_kv = d.n_kv_heads * HD;
    const float eps = d.rms_eps;
    static const bool swa_on = std::getenv("IE_GPTOSS_NO_SWA_WINDOW") == nullptr;

    // positions
    std::vector<int32_t> pos(T);
    for (uint32_t t = 0; t < T; ++t) pos[t] = int32_t(start_pos + t);
    q.memcpy(ws_pos_, pos.data(), T * sizeof(int32_t)).wait();

    // embedding (Q8_0)
    int32_t* d_ids = static_cast<int32_t*>(alloc_->malloc(T * sizeof(int32_t)));
    q.memcpy(d_ids, input_ids, T * sizeof(int32_t)).wait();
    embedding_lookup_q8_0(q, d_ids, token_embd_, ws_x_, T, H);
    q.wait();
    alloc_->free(d_ids);

    // Projection helper: at PREFILL (T>1) the fp16 attn weights must batch through
    // gemm_fp16_onednn — NOT dense::gemv_q_T, whose batched branch is gated K%256==0,
    // which gpt-oss's H=2880 (2880%256=64) FAILS → it fell back to a per-token gemv
    // (196k calls = 88% of prefill). oneDNN handles K=2880 fine (the MoE path already
    // does). T==1 (decode) keeps the gemv. (gemv_q_T already gemm-batches K%256==0
    // weights; this just covers the non-256 case it skips.)
    static const bool proj_gemm = std::getenv("IE_GPTOSS_NO_PROJ_GEMM") == nullptr;
    auto proj = [&](const sycl::half* A, const DenseQuantPtr& W, sycl::half* y,
                    uint32_t Kp, uint32_t Np) {
        if (proj_gemm && T > 1 && W.dt == DType::kF16 && (Np % 64 == 0))
            gemm_fp16_onednn(q, A, static_cast<const sycl::half*>(W.p), y, T, Np, Kp, {});
        else
            dense::gemv_q_T(q, A, W, y, Kp, Np, T);
    };

    for (uint32_t L = 0; L < d.n_layers; ++L) {
        const Layer& w = layers_[L];
        // --- attention ---
        rms_norm_f32w(q, ws_x_, w.attn_norm, ws_xn_, T, H, eps);
        proj(ws_xn_, w.attn_q, ws_q_, H, N_q);
        proj(ws_xn_, w.attn_k, ws_k_, H, N_kv);
        proj(ws_xn_, w.attn_v, ws_v_, H, N_kv);
        add_bias(q, ws_q_, w.attn_q_bias, T, N_q);
        add_bias(q, ws_k_, w.attn_k_bias, T, N_kv);
        add_bias(q, ws_v_, w.attn_v_bias, T, N_kv);
        // RoPE. DEFAULT = plain rope base 150000 — MEASURED best ≤4096 ctx (streaming
        // PPL 17.9 vs 30.5 full-YaRN / 22.1 mscale-only on the 586-tok corpus): the
        // model's correct-base plain rope is already good at short ctx, and the
        // factor-32 YaRN aggressively perturbs the low-freq dims + adds the 1.34657
        // mscale, both of which RAISE short-ctx PPL. YaRN (rope_yarn, ggml-exact, the
        // deployed rope) becomes essential only >4096 where plain-rope extrapolation
        // fails — opt-in IE_GPTOSS_YARN until long-ctx validation. (llama needs YaRN in
        // its chunked eval because --rope-scaling none breaks it: 447 vs 54.7.)
        static const bool use_yarn = std::getenv("IE_GPTOSS_YARN") != nullptr;
        if (use_yarn) {
            rope_yarn(q, ws_q_, ws_pos_, ws_q_, T, d.n_q_heads,  HD, d.rope_dim,
                      d.rope_theta, cfg_.rope_freq_scale, cfg_.rope_orig_ctx,
                      cfg_.rope_ext_factor, /*attn_factor=*/1.0f, /*beta_fast=*/32.0f, /*beta_slow=*/1.0f);
            rope_yarn(q, ws_k_, ws_pos_, ws_k_, T, d.n_kv_heads, HD, d.rope_dim,
                      d.rope_theta, cfg_.rope_freq_scale, cfg_.rope_orig_ctx,
                      cfg_.rope_ext_factor, 1.0f, 32.0f, 1.0f);
        } else {
            rope_partial(q, ws_q_, ws_pos_, ws_q_, T, d.n_q_heads,  HD, d.rope_dim, d.rope_theta);
            rope_partial(q, ws_k_, ws_pos_, ws_k_, T, d.n_kv_heads, HD, d.rope_dim, d.rope_theta);
        }
        const uint64_t per_layer = uint64_t(d.n_kv_heads) * kv.config().max_ctx * HD;
        sycl::half* kc = kv.k_ptr() + per_layer * L;
        sycl::half* vc = kv.v_ptr() + per_layer * L;
        const uint32_t window = (swa_on && w.is_swa) ? cfg_.sliding_window : 0u;
        // Decode (T==1) on a FULL-attention layer (window==0) → split-K FA-2 with
        // the per-head sink (the long-ctx decode lever: attn was 62.7% @2K). The
        // windowed (even) layers stay on the bounded naive path (already ~16× cheaper
        // — only 128 keys). Prefill (T>1) stays naive for now. Needs the partials
        // scratch (tools wire ensure_attn_partials; engine wires it at load).
        static const bool fa2_decode_on  = std::getenv("IE_GPTOSS_NO_FA2_DECODE") == nullptr;
        static const bool fa2_prefill_on = std::getenv("IE_GPTOSS_NO_FA2_PREFILL") == nullptr;
        if (T == 1 && fa2_decode_on && window == 0 && ws_attn_partials_) {
            full_attention_fa2_decode_gptoss(q, ws_q_, ws_k_, ws_v_, kc, vc, ws_attn_out_,
                                             ws_attn_partials_, start_pos, d.n_q_heads,
                                             d.n_kv_heads, HD, kv.config().max_ctx,
                                             w.attn_sinks, {});
        } else if (T > 1 && fa2_prefill_on) {
            // PREFILL wide-tile (hd64 + sink). Handles both full (window=0) and SWA
            // (window=128) layers — KV read once per query-tile vs naive O(T·ctx).
            full_attention_gptoss_prefill_tile(q, ws_q_, ws_k_, ws_v_, kc, vc, ws_attn_out_,
                                               T, start_pos, d.n_q_heads, d.n_kv_heads,
                                               kv.config().max_ctx, window, w.attn_sinks, {});
        } else {
            full_attention_gptoss(q, ws_q_, ws_k_, ws_v_, kc, vc, ws_attn_out_,
                                  T, start_pos, d.n_q_heads, d.n_kv_heads, HD,
                                  kv.config().max_ctx, window, w.attn_sinks);
        }
        kv.set_length(L, start_pos + T);
        proj(ws_attn_out_, w.attn_output, ws_block_, N_q, H);
        add_bias(q, ws_block_, w.attn_o_bias, T, H);
        residual_add(q, ws_x_, ws_block_, ws_x_, uint64_t(T) * H);

        // --- MoE FFN ---
        rms_norm_f32w(q, ws_x_, w.post_attn_norm, ws_xn_, T, H, eps);
        moe_ffn(q, w, T);
        residual_add(q, ws_x_, ws_moe_y_, ws_x_, uint64_t(T) * H);
        q.wait();
    }

    // final norm + lm_head (last token only — next-token logits)
    rms_norm_f32w(q, ws_x_, output_norm_, ws_xn_, T, H, eps);
    const sycl::half* last = ws_xn_ + uint64_t(T - 1) * H;
    if (lmhead_K_) {
        // W8A8 int-dot lm_head: quantize the final hidden once → gemv_q8_0_soa_q8.
        quantize_q8_1(q, last, ws_q8_x_, H);
        return gemv_q8_0_soa_q8(q, ws_q8_x_, lmhead_qs_, lmhead_d_, out_logits, H, d.vocab);
    }
    return dense::gemv_q(q, last, output_, out_logits, H, d.vocab);
}

}  // namespace ie
