// src/model/gemma4.cpp — Google Gemma 4 (`gemma4`) loader + forward.
// See include/ie/gemma4.hpp + docs/superpowers/plans/2026-06-13-gemma4-arch.md.
// Additive: Q4_0 weights via gemv_q4_0/gemm_q4_0; the crown/dense paths untouched.

#include "ie/gemma4.hpp"

#include "dense_dispatch.hpp"
#include "ie/gemma4_assistant.hpp"  // read_gemma4_assistant_config (MTP head)
#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"    // block_q8_0/q8_1x/q8_1s
#include "ie/qwen3moe_pack.hpp"   // build_moe_packing / MoePacking (fused-MoE)

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace ie {

namespace {
// Softmax-after-topk over logits[E]: softmax all E, take top-k by prob,
// renormalize the k to sum 1, sort ascending by expert index. (Gemma 4 router.)
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

Gemma4Model::~Gemma4Model() {
    if (!alloc_) return;
    for (void* p : owned_) alloc_->free(p);
    owned_.clear();
}

std::string Gemma4Model::load(DeviceAllocator& alloc, const GgufReader& g,
                              const GemmaConfig& cfg) {
    alloc_ = &alloc;
    cfg_   = cfg;
    const uint32_t L = cfg.n_layers;
    char buf[80];
    std::string err;

    auto Tt = [&](const char* n) { return g.find_tensor(n); };
    auto Tl = [&](uint32_t l, const char* n) {
        std::snprintf(buf, sizeof(buf), "blk.%u.%s", l, n);
        return g.find_tensor(buf);
    };
    auto up_f32 = [&](const GgufTensorInfo* t, const char* what) -> float* {
        auto* p = dense::upload<float>(alloc, t, owned_, err, DType::kF32);
        if (!err.empty()) err = std::string(what) + ": " + err;
        return p;
    };
    // Raw Q4_0 weight → DenseQuantPtr. SoA-ONLY by default: repack each dense Q4_0
    // weight into SoA streams (nibbles + fp16 scales, built from the HOST GGUF —
    // repack_q4_0_to_soa reads t->data, no device AoS needed) and do NOT upload the
    // AoS device copy. Decode = gemv_q4_0_soa_q8; prefill = dequant_q4_0_soa_to_Bt
    // → onednn GEMM (bit-identical to the AoS dequant). This halves resident
    // projection VRAM (e.g. 31B dense ~34→~18 GB → fits a 24 GB single GPU).
    //   IE_GEMMA4_NO_Q4_SOA : AoS-only (old gemv_q4_0 decode + AoS prefill), A/B.
    //   IE_GEMMA4_KEEP_AOS  : keep BOTH copies (enables the dp4a/xmx A/B prefill
    //                         paths that read W.p), at the old 2× memory cost.
    // Invariant maintained: every weight has W.p OR W.soa_qs (never neither).
    const bool q4_soa_on = std::getenv("IE_GEMMA4_NO_Q4_SOA") == nullptr;
    const bool keep_aos  = std::getenv("IE_GEMMA4_KEEP_AOS")  != nullptr;
    auto up_q40 = [&](const GgufTensorInfo* t, const char* what) -> DenseQuantPtr {
        if (!t) { err = std::string(what) + ": not found"; return {}; }
        if (t->dtype != DType::kQ4_0) { err = std::string(what) + ": expected Q4_0"; return {}; }
        DenseQuantPtr dq{nullptr, DType::kQ4_0};
        bool soa_built = false;
        if (q4_soa_on && t->n_dims == 2) {
            const uint32_t K = uint32_t(t->shape[0]), N = uint32_t(t->shape[1]);
            if (K % 32 == 0) {
                std::vector<uint8_t>  h_qs(uint64_t(N) * (K / 2));
                std::vector<uint16_t> h_d (uint64_t(N) * (K / 32));
                repack_q4_0_to_soa(t->data, K, N, h_qs.data(), h_d.data());
                auto* d_qs = alloc.malloc(h_qs.size());
                auto* d_d  = alloc.malloc(h_d.size() * sizeof(uint16_t));
                if (!d_qs || !d_d) { err = std::string(what) + ": SoA alloc failed"; return dq; }
                owned_.push_back(d_qs); owned_.push_back(d_d);
                alloc.queue().memcpy(d_qs, h_qs.data(), h_qs.size()).wait();
                alloc.queue().memcpy(d_d,  h_d.data(),  h_d.size() * sizeof(uint16_t)).wait();
                dq.soa_qs = d_qs; dq.soa_d = d_d;
                soa_built = true;
            }
        }
        // Upload the AoS device copy only when SoA wasn't built (non-2D / K%32≠0)
        // or explicitly requested for A/B (IE_GEMMA4_KEEP_AOS).
        if (!soa_built || keep_aos) {
            dq.p = dense::upload<void>(alloc, t, owned_, err, DType::kQ4_0);
            if (!err.empty()) { err = std::string(what) + ": " + err; return {}; }
        }
        return dq;
    };

    // --- top level ---
    {
        const auto* ti = Tt("token_embd.weight");
        if (!ti) return "token_embd: not found";
        if (ti->dtype != DType::kQ6_K && ti->dtype != DType::kQ4_K)
            return "token_embd: expected Q6_K/Q4_K";
        token_embd_dtype_ = ti->dtype;
        token_embd_ = dense::upload<void>(alloc, ti, owned_, err, ti->dtype);
        if (!err.empty()) return "token_embd: " + err;
        // Fast lm_head decode: repack the tied Q6_K token_embd into SoA-Q6 streams
        // for gemv_q6_soa_q8 (~57% BW vs the AoS gemv_q6_K ~20%). K=hidden,
        // N=vocab. Embedding lookup keeps the AoS copy. Opt-out IE_GEMMA4_NO_Q4_SOA.
        if (q4_soa_on && ti->dtype == DType::kQ6_K && ti->n_dims == 2) {
            const uint32_t K = uint32_t(ti->shape[0]), N = uint32_t(ti->shape[1]);
            if (K % 256 == 0) {
                std::vector<uint8_t> h_lo(uint64_t(N) * (K / 2), 0), h_hi(uint64_t(N) * (K / 4), 0);
                std::vector<int8_t>  h_sc(uint64_t(N) * (K / 16));
                std::vector<uint16_t> h_d(uint64_t(N) * (K / 256));
                repack_q6_K_to_soa(ti->data, K, N, h_lo.data(), h_hi.data(), h_sc.data(), h_d.data());
                lmh_q6_lo_ = static_cast<uint8_t*>(alloc.malloc(h_lo.size()));
                lmh_q6_hi_ = static_cast<uint8_t*>(alloc.malloc(h_hi.size()));
                lmh_q6_sc_ = static_cast<int8_t*>(alloc.malloc(h_sc.size()));
                lmh_q6_d_  = static_cast<uint16_t*>(alloc.malloc(h_d.size() * sizeof(uint16_t)));
                if (!lmh_q6_lo_ || !lmh_q6_hi_ || !lmh_q6_sc_ || !lmh_q6_d_) return "lm_head SoA-Q6 alloc failed";
                owned_.push_back(lmh_q6_lo_); owned_.push_back(lmh_q6_hi_);
                owned_.push_back(lmh_q6_sc_); owned_.push_back(lmh_q6_d_);
                alloc.queue().memcpy(lmh_q6_lo_, h_lo.data(), h_lo.size()).wait();
                alloc.queue().memcpy(lmh_q6_hi_, h_hi.data(), h_hi.size()).wait();
                alloc.queue().memcpy(lmh_q6_sc_, h_sc.data(), h_sc.size()).wait();
                alloc.queue().memcpy(lmh_q6_d_,  h_d.data(),  h_d.size() * sizeof(uint16_t)).wait();
            }
        }
    }
    output_norm_ = up_f32(Tt("output_norm.weight"), "output_norm");
    if (!err.empty()) return err;
    if (const auto* rf = Tt("rope_freqs.weight")) {
        rope_freqs_ = up_f32(rf, "rope_freqs");
        if (!err.empty()) return err;
    }

    layers_.assign(L, {});
    max_head_dim_ = 0; max_n_kv_ = 0;
    const uint32_t E  = cfg.n_experts;
    const uint32_t EF = cfg.expert_ffn;

    for (uint32_t l = 0; l < L; ++l) {
        Layer& w = layers_[l];
        w.is_swa     = cfg.is_swa[l] != 0;
        w.head_dim   = cfg.head_dim(l);
        w.n_kv       = cfg.n_kv_heads[l];
        w.n_rot      = cfg.n_rot(l);
        w.rope_theta = cfg.rope_theta(l);
        max_head_dim_ = std::max(max_head_dim_, w.head_dim);
        max_n_kv_     = std::max(max_n_kv_, w.n_kv);

        // norms (plain-w; load as-is)
        w.attn_norm      = up_f32(Tl(l, "attn_norm.weight"), "attn_norm");      if (!err.empty()) return err;
        w.attn_q_norm    = up_f32(Tl(l, "attn_q_norm.weight"), "attn_q_norm");  if (!err.empty()) return err;
        w.attn_k_norm    = up_f32(Tl(l, "attn_k_norm.weight"), "attn_k_norm");  if (!err.empty()) return err;
        w.post_attn_norm = up_f32(Tl(l, "post_attention_norm.weight"), "post_attention_norm"); if (!err.empty()) return err;
        w.ffn_norm       = up_f32(Tl(l, "ffn_norm.weight"), "ffn_norm");        if (!err.empty()) return err;
        w.post_ffw_norm  = up_f32(Tl(l, "post_ffw_norm.weight"), "post_ffw_norm"); if (!err.empty()) return err;

        // attention projections (Q4_0). attn_v absent on global layers → V=K.
        w.attn_q = up_q40(Tl(l, "attn_q.weight"), "attn_q"); if (!err.empty()) return err;
        w.attn_k = up_q40(Tl(l, "attn_k.weight"), "attn_k"); if (!err.empty()) return err;
        if (const auto* tv = Tl(l, "attn_v.weight")) {
            w.attn_v = up_q40(tv, "attn_v"); if (!err.empty()) return err;
            w.has_v = true;
        } else {
            w.has_v = false;  // reuse K projection as V
        }
        w.attn_output = up_q40(Tl(l, "attn_output.weight"), "attn_output"); if (!err.empty()) return err;

        // shared dense GeGLU FFN (every layer)
        w.ffn_gate = up_q40(Tl(l, "ffn_gate.weight"), "ffn_gate"); if (!err.empty()) return err;
        w.ffn_up   = up_q40(Tl(l, "ffn_up.weight"),   "ffn_up");   if (!err.empty()) return err;
        w.ffn_down = up_q40(Tl(l, "ffn_down.weight"), "ffn_down"); if (!err.empty()) return err;

        // per-layer output scalar (optional; host F32 [1])
        if (const auto* os = Tl(l, "layer_output_scale.weight")) {
            if (os->dtype != DType::kF32 || os->nbytes < sizeof(float)) return "layer_output_scale: bad";
            std::memcpy(&w.layer_out_scale_val, os->data, sizeof(float));
        }

        // MoE dual-path (when the layer carries a router)
        const auto* gi = Tl(l, "ffn_gate_inp.weight");
        if (cfg.is_moe && gi) {
            w.is_moe = true;
            if (gi->dtype != DType::kF32) return "ffn_gate_inp: expected F32";
            if (gi->nbytes != uint64_t(E) * cfg.hidden * sizeof(float))
                return "ffn_gate_inp: unexpected size";
            w.router_w.resize(uint64_t(E) * cfg.hidden);
            std::memcpy(w.router_w.data(), gi->data, gi->nbytes);
            if (const auto* gis = Tl(l, "ffn_gate_inp.scale")) {
                if (gis->dtype != DType::kF32 || gis->nbytes != uint64_t(cfg.hidden) * sizeof(float))
                    return "ffn_gate_inp.scale: bad";
                w.router_in_scale_h.resize(cfg.hidden);
                std::memcpy(w.router_in_scale_h.data(), gis->data, gis->nbytes);
            }
            // GPU router buffers: router_w as F16 [E,H] (gemv_q_T weight) + the
            // per-channel router_in_scale (all-ones when the GGUF omits it) as F32.
            {
                // TRANSPOSED [H,E] for gemv_q_T: dev[h*E+e] = router_w[e*H+h]
                // (matches qwen3moe.cpp:248). Natural [E,H] gives garbage logits.
                std::vector<sycl::half> rw(uint64_t(cfg.hidden) * E);
                for (uint32_t e = 0; e < E; ++e)
                    for (uint32_t h = 0; h < cfg.hidden; ++h)
                        rw[uint64_t(h) * E + e] = sycl::half(w.router_w[uint64_t(e) * cfg.hidden + h]);
                w.router_w_dev = static_cast<sycl::half*>(alloc.malloc(rw.size() * sizeof(sycl::half)));
                if (!w.router_w_dev) return "router_w_dev alloc failed";
                owned_.push_back(w.router_w_dev);
                alloc.queue().memcpy(w.router_w_dev, rw.data(), rw.size() * sizeof(sycl::half)).wait();

                std::vector<float> ris(cfg.hidden, 1.0f);
                if (!w.router_in_scale_h.empty()) ris = w.router_in_scale_h;
                w.router_in_scale_dev = static_cast<float*>(alloc.malloc(uint64_t(cfg.hidden) * sizeof(float)));
                if (!w.router_in_scale_dev) return "router_in_scale_dev alloc failed";
                owned_.push_back(w.router_in_scale_dev);
                alloc.queue().memcpy(w.router_in_scale_dev, ris.data(), uint64_t(cfg.hidden) * sizeof(float)).wait();
            }
            w.pre_ffw_norm_2  = up_f32(Tl(l, "pre_ffw_norm_2.weight"),  "pre_ffw_norm_2");  if (!err.empty()) return err;
            w.post_ffw_norm_1 = up_f32(Tl(l, "post_ffw_norm_1.weight"), "post_ffw_norm_1"); if (!err.empty()) return err;
            w.post_ffw_norm_2 = up_f32(Tl(l, "post_ffw_norm_2.weight"), "post_ffw_norm_2"); if (!err.empty()) return err;

            // SoA-Q4_0 expert banks (IE_GEMMA4_MOE_SOA): repack each expert's
            // [K,N] AoS → contiguous nibble + fp16-scale streams; store SoA
            // INSTEAD of AoS (no doubling). Serves prefill + decode via
            // moe_prefill_proj_q4_0_soa_q8. Per expert stride = N*(K/2) qs, N*(K/32) d.
            // SoA expert banks DEFAULT-ON, but REQUIRE fused MoE (the unfused per-token
            // path reads the AoS gate_up_exps which is null on the SoA path). So opting
            // out of fused (IE_GEMMA4_NO_FUSED_MOE) also disables SoA → AoS banks built.
            static const bool moe_soa = std::getenv("IE_GEMMA4_NO_MOE_SOA") == nullptr &&
                                        std::getenv("IE_GEMMA4_NO_FUSED_MOE") == nullptr;
            auto up_exps_soa = [&](const GgufTensorInfo* t, uint32_t K, uint32_t N,
                                   uint8_t*& qs_dev, uint16_t*& d_dev,
                                   uint64_t& qs_stride, uint64_t& d_stride) -> bool {
                const uint64_t e_stride = t->nbytes / E;        // AoS bytes/expert
                qs_stride = uint64_t(N) * (K / 2);
                d_stride  = uint64_t(N) * (K / 32);
                std::vector<uint8_t>  h_qs(uint64_t(E) * qs_stride);
                std::vector<uint16_t> h_d (uint64_t(E) * d_stride);
                for (uint32_t e = 0; e < E; ++e)
                    repack_q4_0_to_soa(t->data + uint64_t(e) * e_stride, K, N,
                                       h_qs.data() + uint64_t(e) * qs_stride,
                                       h_d.data()  + uint64_t(e) * d_stride);
                qs_dev = static_cast<uint8_t*>(alloc.malloc(h_qs.size()));
                d_dev  = static_cast<uint16_t*>(alloc.malloc(h_d.size() * sizeof(uint16_t)));
                if (!qs_dev || !d_dev) { err = "MoE SoA alloc failed"; return false; }
                owned_.push_back(qs_dev); owned_.push_back(d_dev);
                alloc.queue().memcpy(qs_dev, h_qs.data(), h_qs.size()).wait();
                alloc.queue().memcpy(d_dev,  h_d.data(),  h_d.size() * sizeof(uint16_t)).wait();
                return true;
            };

            const auto* gu = Tl(l, "ffn_gate_up_exps.weight");
            if (!gu) return "ffn_gate_up_exps: not found";
            if (gu->dtype != DType::kQ4_0) return "ffn_gate_up_exps: expected Q4_0";
            if (gu->nbytes % E) return "ffn_gate_up_exps: nbytes % expert_count";
            const auto* de = Tl(l, "ffn_down_exps.weight");
            if (!de) return "ffn_down_exps: not found";
            if (de->dtype != DType::kQ4_0) return "ffn_down_exps: expected Q4_0";
            if (de->nbytes % E) return "ffn_down_exps: nbytes % expert_count";

            if (moe_soa) {
                if (!up_exps_soa(gu, cfg.hidden, 2 * EF, w.gu_qs, w.gu_d, w.gu_qs_stride, w.gu_d_stride))
                    return "ffn_gate_up_exps SoA: " + err;
                if (!up_exps_soa(de, EF, cfg.hidden, w.dn_qs, w.dn_d, w.dn_qs_stride, w.dn_d_stride))
                    return "ffn_down_exps SoA: " + err;
            } else {
                w.gate_up_exps  = dense::upload<void>(alloc, gu, owned_, err, DType::kQ4_0);
                if (!err.empty()) return "ffn_gate_up_exps: " + err;
                w.gate_up_stride = gu->nbytes / E;
                w.down_exps  = dense::upload<void>(alloc, de, owned_, err, DType::kQ4_0);
                if (!err.empty()) return "ffn_down_exps: " + err;
                w.down_stride = de->nbytes / E;
            }

            if (const auto* ds = Tl(l, "ffn_down_exps.scale")) {
                if (ds->dtype != DType::kF32 || ds->nbytes != uint64_t(E) * sizeof(float))
                    return "ffn_down_exps.scale: bad";
                w.down_scale_h.resize(E);
                std::memcpy(w.down_scale_h.data(), ds->data, ds->nbytes);
            }
            (void)EF;
        }
    }
    return {};
}

std::string Gemma4Model::ensure_workspace(uint32_t max_T) {
    if (max_T <= ws_T_ && ws_x_) return {};
    auto ah = [&](uint64_t n) {
        auto* p = static_cast<sycl::half*>(alloc_->malloc(n * sizeof(sycl::half)));
        if (p) owned_.push_back(p);
        return p;
    };
    const uint32_t H = cfg_.hidden;
    const uint32_t HDX = max_head_dim_;                 // 512
    const uint32_t NQX = cfg_.n_q_heads * HDX;          // 16*512 = 8192
    const uint32_t NKX = max_n_kv_ * HDX;               // <= 8*512 (sized generously)
    const uint32_t EF  = cfg_.expert_ffn ? cfg_.expert_ffn : cfg_.ffn;
    const uint64_t T   = max_T;

    ws_x_         = ah(T * H);
    ws_xn_        = ah(T * H);
    ws_q_         = ah(T * NQX);
    ws_k_         = ah(T * NKX);
    ws_v_         = ah(T * NKX);
    ws_attn_out_  = ah(T * NQX);
    ws_block_     = ah(T * H);
    ws_attn_out2_ = ah(T * H);
    ws_gate_      = ah(T * std::max(cfg_.ffn, EF));
    ws_up_        = ah(T * std::max(cfg_.ffn, EF));
    ws_h_         = ah(T * std::max(cfg_.ffn, EF));
    ws_moe_y_     = ah(T * H);
    ws_shared_y_  = ah(T * H);
    ws_rin_       = ah(T * H);                         // GPU-router input
    ws_router_logits_ = ah(T * std::max<uint32_t>(cfg_.n_experts, 1u));  // [T,E]
    // per-(token,expert) MoE scratch (single token)
    ws_e_gateup_  = ah(uint64_t(2) * EF);
    ws_e_h_       = ah(EF);
    ws_e_out_     = ah(H);

    // int-dot W4A8 dense/attn projections (IE_GEMMA4_INTDOT_PROJ). Always
    // allocated (cheap, additive) so 26B-MoE and 31B-dense both get it. The
    // staging stream holds the quantized projection INPUT [T, Kmax/32]; Kmax is
    // the largest projection-input width (NQX for o-proj, H for q/k/v/gate/up,
    // ffn for down). Consumed by the split-K gemm_q4_0_q8.
    {
        const uint64_t Kmax = std::max<uint64_t>({H, NQX, cfg_.ffn, EF});
        const uint64_t q8b  = sizeof(block_q8_1s);
        ws_projq8_ = alloc_->malloc(T * (Kmax / 32) * q8b);
        if (ws_projq8_) owned_.push_back(ws_projq8_);
        if (!ws_projq8_) return "gemma4 int-dot proj workspace alloc failed";
        // block_q8_1x activation stream for SoA-Q4_0 decode (single row, max K).
        ws_q8_ = alloc_->malloc((Kmax / 32) * sizeof(block_q8_1x));
        if (ws_q8_) owned_.push_back(ws_q8_);
        if (!ws_q8_) return "gemma4 ws_q8 alloc failed";
        // T-row block_q8_1x stream for the batched-verify int-dot GEMV.
        ws_q8b_ = alloc_->malloc(T * (Kmax / 32) * sizeof(block_q8_1x));
        if (ws_q8b_) owned_.push_back(ws_q8b_);
        if (!ws_q8b_) return "gemma4 ws_q8b alloc failed";
    }

    // batched fused-MoE packing buffers (only when MoE + IE_GEMMA4_FUSED_MOE):
    // T*K expert-sorted routed rows for gather → per-expert gemm_q4_0 → reduce.
    static const bool fused_moe = std::getenv("IE_GEMMA4_NO_FUSED_MOE") == nullptr;
    if (fused_moe && cfg_.n_experts_used > 0) {
        const uint64_t TK = T * cfg_.n_experts_used;
        ws_xp_packed_      = ah(TK * H);
        ws_gu_packed_      = ah(TK * 2 * EF);
        ws_h_packed_       = ah(TK * EF);
        ws_out_packed_     = ah(TK * H);
        ws_weights_packed_ = ah(TK);
        ws_sorted_idx_   = static_cast<int32_t*>(alloc_->malloc(TK * sizeof(int32_t)));
        if (ws_sorted_idx_) owned_.push_back(ws_sorted_idx_);
        ws_tk_to_packed_ = static_cast<int32_t*>(alloc_->malloc(TK * sizeof(int32_t)));
        if (ws_tk_to_packed_) owned_.push_back(ws_tk_to_packed_);
        // int-dot W4A8: q8_1s activation streams + device expert_offsets.
        const uint64_t q8b = sizeof(block_q8_1s);                 // 48
        ws_xq8_ = alloc_->malloc(TK * (H / 32) * q8b);
        if (ws_xq8_) owned_.push_back(ws_xq8_);
        ws_hq8_ = alloc_->malloc(TK * (EF / 32) * q8b);
        if (ws_hq8_) owned_.push_back(ws_hq8_);
        ws_expert_offsets_ = static_cast<uint32_t*>(
            alloc_->malloc(uint64_t(cfg_.n_experts + 1) * sizeof(uint32_t)));
        if (ws_expert_offsets_) owned_.push_back(ws_expert_offsets_);
        if (!ws_xp_packed_ || !ws_gu_packed_ || !ws_h_packed_ || !ws_out_packed_ ||
            !ws_weights_packed_ || !ws_sorted_idx_ || !ws_tk_to_packed_ ||
            !ws_xq8_ || !ws_hq8_ || !ws_expert_offsets_)
            return "gemma4 fused-MoE workspace alloc failed";
    }

    if (!ws_x_ || !ws_xn_ || !ws_q_ || !ws_k_ || !ws_v_ || !ws_attn_out_ ||
        !ws_block_ || !ws_attn_out2_ || !ws_gate_ || !ws_up_ || !ws_h_ ||
        !ws_moe_y_ || !ws_shared_y_ || !ws_rin_ || !ws_router_logits_ ||
        !ws_e_gateup_ || !ws_e_h_ || !ws_e_out_)
        return "gemma4 workspace alloc failed";

    // all-ones weight buffer for the weightless V RMS-norm
    if (!ones_hd_) {
        ones_hd_ = static_cast<float*>(alloc_->malloc(uint64_t(HDX) * sizeof(float)));
        if (!ones_hd_) return "gemma4 ones_hd alloc failed";
        owned_.push_back(ones_hd_);
        std::vector<float> ones(HDX, 1.0f);
        alloc_->queue().memcpy(ones_hd_, ones.data(), uint64_t(HDX) * sizeof(float)).wait();
    }
    if (max_T > ws_positions_cap_) {
        ws_positions_ = static_cast<int32_t*>(alloc_->malloc(uint64_t(max_T) * sizeof(int32_t)));
        if (!ws_positions_) return "gemma4 positions alloc failed";
        owned_.push_back(ws_positions_);
        ws_positions_cap_ = max_T;
    }
    ws_T_ = max_T;
    return {};
}

std::string Gemma4Model::ensure_kv(uint32_t max_ctx) {
    if (max_ctx <= kv_ctx_ && !kcache_.empty()) return {};
    kcache_.assign(cfg_.n_layers, nullptr);
    vcache_.assign(cfg_.n_layers, nullptr);
    for (uint32_t l = 0; l < cfg_.n_layers; ++l) {
        const uint64_t n = uint64_t(layers_[l].n_kv) * layers_[l].head_dim * max_ctx;
        auto* k = static_cast<sycl::half*>(alloc_->malloc(n * sizeof(sycl::half)));
        auto* v = static_cast<sycl::half*>(alloc_->malloc(n * sizeof(sycl::half)));
        if (!k || !v) return "gemma4 kv alloc failed";
        owned_.push_back(k); owned_.push_back(v);
        kcache_[l] = k; vcache_[l] = v;
    }
    kv_ctx_ = max_ctx;
    return {};
}

namespace {
// logits = tanh(logits / cap) * cap  (Gemma 4 final-logit soft-cap; monotonic).
sycl::event softcap_inplace(sycl::queue& q, sycl::half* y, uint32_t n, float cap) {
    return q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        y[i] = sycl::half(sycl::tanh(float(y[i]) / cap) * cap);
    });
}
}  // namespace

std::string Gemma4Model::ensure_attn_partials(uint32_t max_ctx) {
    if (max_ctx <= ws_attn_partials_ctx_) return {};
    // 2 fp32 partials (m,l) + accumulator per (q_head, ctx) — sized by max geometry.
    const uint64_t n = uint64_t(cfg_.n_q_heads) * max_head_dim_ * 2;
    auto* p = static_cast<sycl::half*>(alloc_->malloc(n * sizeof(sycl::half)));
    if (!p) return "gemma4 attn_partials alloc failed";
    owned_.push_back(p);
    ws_attn_partials_ = p;
    ws_attn_partials_ctx_ = max_ctx;
    return {};
}

// Weight-stationary oneDNN fp16 MoE GEMM (prefill, large-M). See gemma4.hpp.
// Mirrors Qwen3MoeModel::moe_xmx_prefill, adapted for Gemma's SoA-Q4_0 banks
// (the dense path already proves dequant_q4_0_soa_to_Bt + gemm_fp16_onednn are
// bit-identical to AoS on these banks; same repack_q4_0_to_soa builds both).
void Gemma4Model::moe_onednn_proj(sycl::queue& q, const sycl::half* in_packed,
                                  const uint8_t* qs, const uint16_t* d,
                                  uint64_t qs_stride, uint64_t d_stride,
                                  sycl::half* out_packed, const uint32_t* expert_offsets,
                                  uint32_t Kd, uint32_t Nd) {
    const uint32_t E = cfg_.n_experts;
    for (uint32_t e = 0; e < E; ++e) {
        const uint32_t off = expert_offsets[e];
        const uint32_t n_e = expert_offsets[e + 1] - off;
        if (n_e == 0) continue;
        // dequant this expert's SoA-Q4_0 weight [Kd,Nd] → fp16 Bt (ws_btf16_).
        auto de = dequant_q4_0_soa_to_Bt(q, qs + e * qs_stride, d + e * d_stride,
                                         ws_btf16_, Kd, Nd, {});
        // fp16 oneDNN GEMM: in_e[n_e,Kd] @ Bt[Kd,Nd] → out_e[n_e,Nd]. In-order
        // queue serializes the per-expert reuse of ws_btf16_ (WAR/RAW); {de} dep
        // kept (harmless). Output UNWEIGHTED (moe_prefill_reduce weights later).
        gemm_fp16_onednn(q, in_packed + uint64_t(off) * Kd, ws_btf16_,
                         out_packed + uint64_t(off) * Nd, n_e, Nd, Kd, {de});
    }
}

sycl::event Gemma4Model::forward(sycl::queue& q, const int32_t* input_ids, uint32_t T,
                                 uint32_t start_pos, KvCache& kv, sycl::half* out_logits,
                                 sycl::half* hidden_pre_norm, sycl::half* all_logits) {
    (void)kv;  // Gemma self-manages per-layer KV (kcache_/vcache_).
    const uint32_t H   = cfg_.hidden;
    const float    eps = cfg_.rms_eps;
    const uint32_t nq  = cfg_.n_q_heads;
    const uint32_t E   = cfg_.n_experts;
    const uint32_t K   = cfg_.n_experts_used;
    const uint32_t EF  = cfg_.expert_ffn;

    // positions
    {
        std::vector<int32_t> pos(T);
        for (uint32_t t = 0; t < T; ++t) pos[t] = int32_t(start_pos + t);
        q.memcpy(ws_positions_, pos.data(), T * sizeof(int32_t)).wait();
    }

    // 1. embedding → ws_x_, scaled by sqrt(H)
    if (token_embd_dtype_ == DType::kQ6_K)
        embedding_lookup_q6k(q, input_ids, token_embd_, ws_x_, T, H);
    else
        embedding_lookup_q4k(q, input_ids, token_embd_, ws_x_, T, H);
    dense::scale_inplace(q, ws_x_, std::sqrt(float(H)), uint64_t(T) * H);
    {
        static const bool dbg0 = std::getenv("IE_GEMMA4_DBG") != nullptr;
        if (dbg0) {
            std::vector<sycl::half> he(H);
            q.memcpy(he.data(), ws_x_ + uint64_t(T - 1) * H, H * sizeof(sycl::half)).wait();
            float amax = 0.f; double asum = 0.0;
            for (uint32_t i = 0; i < H; ++i) { float v = float(he[i]); amax = std::max(amax, std::fabs(v)); asum += std::fabs(v); }
            std::fprintf(stderr, "[DBG EMB] tok=%d H=%u sqrtH=%.3f  absmax=%.4g mean|.|=%.4g\n",
                         int(T >= 1 ? -1 : 0), H, std::sqrt(float(H)), amax, asum / H);
        }
    }

    // Projection dispatch. DECODE (T==1) and large-K prefill projections use the
    // per-token gemv (stages only one K-row into SLM). PREFILL (T>1) batches into
    // one gemm_q4_0 when its SLM staging — min(T,16) rows × K halves — fits the
    // work-group local-memory budget; the global o-proj / ffn_down have K large
    // enough to overflow, so they fall back to the gemv loop. gemm vs gemv is the
    // same int-dot math (per-32-block fold over K); PPL is bit-stable (validated
    // via ie-gemma4-ppl --reprefill, gemm vs IE_GEMMA4_NO_GEMM_PROJ=1). Opt-out
    // for A/B. Decode is unaffected (T==1 always takes the gemv branch).
    static const bool no_gemm_proj = std::getenv("IE_GEMMA4_NO_GEMM_PROJ") != nullptr;
    // int-dot W4A8 projections (the 7.2×-prefill lever): route the dense/attn
    // Q4_0 projections through the same single-launch int-dot kernel that made
    // the MoE fast (moe_prefill_proj_q4_0_q8, run E=1 over [0,T]). Replaces the
    // fp16-dequant gemm_q4_0 / per-token gemv with a dp4a W4A8 fold. The current
    // kernel caps K at 3072 (MAX_BPL) — q/k/v/gate/up/down on 26B (K=H=2816) fit;
    // the large-K o-proj + 31B fall back below (split-K kernel = the follow-on).
    // Fast-config projections are DEFAULT-ON (proven crown-bit-exact); opt out for
    // A/B with IE_GEMMA4_NO_INTDOT_PROJ.
    static const bool intdot_proj = std::getenv("IE_GEMMA4_NO_INTDOT_PROJ") == nullptr;
    const uint64_t slm_budget =
        q.get_device().get_info<sycl::info::device::local_mem_size>();
    static const bool no_q4_soa = std::getenv("IE_GEMMA4_NO_Q4_SOA") != nullptr;
    auto proj = [&](const DenseQuantPtr& W, const sycl::half* in, sycl::half* out,
                    uint32_t Kdim, uint32_t N) {
        // DECODE (T==1): fast SoA-Q4_0 W4A8 int-dot GEMV (~80% BW) vs the AoS
        // gemv_q4_0 (~20% BW). quantize the activation once, then split-K int-dot.
        if (T == 1 && !no_q4_soa && W.soa_qs && W.dt == DType::kQ4_0 && Kdim % 32 == 0) {
            sycl::event qe = quantize_q8_1(q, in, ws_q8_, Kdim);
            gemv_q4_0_soa_q8(q, ws_q8_, static_cast<const uint8_t*>(W.soa_qs),
                             static_cast<const uint16_t*>(W.soa_d), out, Kdim, N, {qe});
            return;
        }
        // VERIFY (small T, 2..16): the MTP spec-decode verify forward. Route the
        // SoA-Q4_0 projections through the batched int-dot DECODE GEMV — each
        // weight column read ONCE, dotted vs all T staged Q8_1 rows. This is the
        // decode-class kernel the prefill GEMM (oneDNN dequant) can't match at
        // tiny T (the de-risk: T=4 verify ≈ 1× a single decode). Per-row
        // bit-identical to the T==1 gemv_q4_0_soa_q8. Opt-out for A/B.
        static const bool no_batched_verify = std::getenv("IE_GEMMA4_NO_BATCHED_VERIFY") != nullptr;
        if (!no_batched_verify && !no_q4_soa && T >= 2 && T <= 16 && W.soa_qs &&
            W.dt == DType::kQ4_0 && Kdim % 32 == 0 && ws_q8b_) {
            sycl::event qe = quantize_q8_1(q, in, ws_q8b_, uint64_t(T) * Kdim);
            gemv_q4_0_soa_q8_batched(q, ws_q8b_, static_cast<const uint8_t*>(W.soa_qs),
                                     static_cast<const uint16_t*>(W.soa_d), out,
                                     Kdim, N, T, {qe});
            return;
        }
        // Prefill-only (T>1): at T==1 the batched kernels + the extra quantize
        // are SLOWER than the decode-tuned per-token gemv_q4_0 (−16% tg). Decode
        // keeps gemv. Default prefill path = the split-K dp4a int-dot
        // gemm_q4_0_q8 (handles any K, reads 4-bit weights directly). MEASURED:
        // dp4a 336 > naive XMX 237 tok/s @T=477 — the per-M-tile XMX kernel
        // (gemm_q4_0_xmx) re-dequants Q4→fp16 every 16 rows, and that expansion
        // costs more than the matrix engine saves. XMX kept opt-in (IE_GEMMA4_XMX)
        // as a building block for a future weight-stationary GEMM.
        // Weight-stationary fp16 path (IE_GEMMA4_ONEDNN): dequant the Q4_0 weight
        // to fp16 ONCE (O(KN)) then a tuned fp16 XMX/oneDNN GEMM — no per-M-tile
        // re-dequant. The likely beat-llama lever for large-M prefill.
        static const bool onednn_proj = std::getenv("IE_GEMMA4_NO_ONEDNN") == nullptr;
        static const bool xmx_proj    = std::getenv("IE_GEMMA4_XMX") != nullptr;
        if (intdot_proj && T > 1 && W.dt == DType::kQ4_0 && onednn_proj &&
            Kdim % 32 == 0) {
            const uint64_t need = uint64_t(Kdim) * N;
            if (need > ws_btf16_cap_) {
                ws_btf16_ = static_cast<sycl::half*>(alloc_->malloc(need * sizeof(sycl::half)));
                if (ws_btf16_) { owned_.push_back(ws_btf16_); ws_btf16_cap_ = need; }
            }
            if (ws_btf16_) {
                // Dequant from the AoS copy if resident, else from the SoA streams
                // (SoA-only default) — bit-identical fp16, same onednn GEMM.
                if (W.p) dequant_q4_0_to_Bt(q, W.p, ws_btf16_, Kdim, N);
                else     dequant_q4_0_soa_to_Bt(q, static_cast<const uint8_t*>(W.soa_qs),
                                                static_cast<const uint16_t*>(W.soa_d),
                                                ws_btf16_, Kdim, N);
                gemm_fp16_onednn(q, in, ws_btf16_, out, T, N, Kdim);
                return;
            }
        }
        // SoA-only prefill fallback (no AoS copy resident, onednn unavailable):
        // per-token SoA int-dot gemv. Correct but unbatched — onednn normally fires.
        if (T > 1 && !W.p && W.soa_qs && W.dt == DType::kQ4_0 && Kdim % 32 == 0) {
            for (uint32_t t = 0; t < T; ++t) {
                sycl::event qe = quantize_q8_1(q, in + uint64_t(t) * Kdim, ws_q8_, Kdim);
                gemv_q4_0_soa_q8(q, ws_q8_, static_cast<const uint8_t*>(W.soa_qs),
                                 static_cast<const uint16_t*>(W.soa_d),
                                 out + uint64_t(t) * N, Kdim, N, {qe});
            }
            return;
        }
        if (intdot_proj && T > 1 && W.dt == DType::kQ4_0) {
            if (xmx_proj && N % 64 == 0 && Kdim % 256 == 0) {
                for (uint32_t m = 0; m < T; m += 16) {
                    const uint32_t mc = std::min<uint32_t>(16, T - m);
                    gemm_q4_0_xmx(q, in + uint64_t(m) * Kdim, W.p,
                                  out + uint64_t(m) * N, mc, Kdim, N);
                }
                return;
            }
            if (ws_projq8_ && Kdim % 32 == 0) {
                quantize_q8_1s(q, in, ws_projq8_, uint64_t(T) * Kdim);
                gemm_q4_0_q8(q, ws_projq8_, W.p, out, T, Kdim, N);
                return;
            }
        }
        const uint64_t gemm_slm = uint64_t(std::min<uint32_t>(T, 16)) * Kdim * sizeof(sycl::half);
        if (T > 1 && !no_gemm_proj && gemm_slm + 2048 <= slm_budget) {
            gemm_q4_0(q, in, W.p, out, T, Kdim, N);
            return;
        }
        for (uint32_t t = 0; t < T; ++t)
            gemv_q4_0(q, in + uint64_t(t) * Kdim, W.p, out + uint64_t(t) * N, Kdim, N);
    };

    // Fused decode: project up to 3 SoA-Q4_0 weights sharing input `in` (K=Kdim)
    // in ONE quant + ONE multi-bank gemv. Used for q/k/v (shared attn_norm) and
    // gate/up (shared ffn_norm): fewer launches + a full grid for the tiny GQA
    // k/v columns. Returns false (no-op) if any weight isn't on the SoA fast
    // path or T>1 — caller falls back to per-weight proj().
    static const bool no_fuse = std::getenv("IE_GEMMA4_NO_FUSE_QKV") != nullptr;
    auto fuse_decode = [&](const DenseQuantPtr* const Ws[3], sycl::half* const outs[3],
                           const uint32_t Ns[3], int n, const sycl::half* in,
                           uint32_t Kdim) -> bool {
        if (T != 1 || no_q4_soa || no_fuse) return false;
        const uint8_t* qs[3]; const uint16_t* d[3]; sycl::half* y[3]; uint32_t N3[3];
        for (int b = 0; b < n; ++b) {
            const DenseQuantPtr& W = *Ws[b];
            if (!(W.soa_qs && W.dt == DType::kQ4_0 && Kdim % 32 == 0)) return false;
            qs[b] = static_cast<const uint8_t*>(W.soa_qs);
            d[b]  = static_cast<const uint16_t*>(W.soa_d);
            y[b]  = outs[b]; N3[b] = Ns[b];
        }
        sycl::event qe = quantize_q8_1(q, in, ws_q8_, Kdim);
        gemv_q4_0_soa_q8_multi(q, ws_q8_, qs, d, y, N3, n, Kdim, {qe});
        return true;
    };

    for (uint32_t L = 0; L < cfg_.n_layers; ++L) {
        const Layer& w = layers_[L];
        const uint32_t HD = w.head_dim, NKV = w.n_kv;
        const uint32_t NQ = nq * HD, NK = NKV * HD;

        // ---- attention ----
        rms_norm_f32w(q, ws_x_, w.attn_norm, ws_xn_, T, H, eps);
        {
            const DenseQuantPtr* const Ws[3] = {&w.attn_q, &w.attn_k, w.has_v ? &w.attn_v : nullptr};
            sycl::half* const outs[3] = {ws_q_, ws_k_, ws_v_};
            const uint32_t Ns[3] = {NQ, NK, NK};
            const int n = w.has_v ? 3 : 2;
            if (!fuse_decode(Ws, outs, Ns, n, ws_xn_, H)) {
                proj(w.attn_q, ws_xn_, ws_q_, H, NQ);
                proj(w.attn_k, ws_xn_, ws_k_, H, NK);
                if (w.has_v) proj(w.attn_v, ws_xn_, ws_v_, H, NK);
            }
        }
        if (!w.has_v) q.memcpy(ws_v_, ws_k_, uint64_t(T) * NK * sizeof(sycl::half));  // V = raw K proj
        rms_norm_f32w(q, ws_q_, w.attn_q_norm, ws_q_, T * nq,  HD, eps);   // QK-norm (per-head)
        rms_norm_f32w(q, ws_k_, w.attn_k_norm, ws_k_, T * NKV, HD, eps);
        rms_norm_f32w(q, ws_v_, ones_hd_,      ws_v_, T * NKV, HD, eps);   // weightless V-norm
        dense::scale_inplace(q, ws_q_, std::sqrt(float(HD)), uint64_t(T) * NQ);  // net attn scale 1.0
        if (!w.is_swa && rope_freqs_) {
            rope_partial_ff(q, ws_q_, ws_positions_, ws_q_, T, nq,  HD, w.n_rot, w.rope_theta, rope_freqs_);
            rope_partial_ff(q, ws_k_, ws_positions_, ws_k_, T, NKV, HD, w.n_rot, w.rope_theta, rope_freqs_);
        } else {
            rope_partial(q, ws_q_, ws_positions_, ws_q_, T, nq,  HD, w.n_rot, w.rope_theta);
            rope_partial(q, ws_k_, ws_positions_, ws_k_, T, NKV, HD, w.n_rot, w.rope_theta);
        }
        // Prefill attention. full_attention_gemma is the NAIVE O(T²) path (one
        // query/WG, each re-reads all KV → collapses at large T: 8K was 248 vs
        // llama 758). For head_dim ≤ 256 (Gemma's SWA layers, the majority) route
        // through the proven query-row-block v2 kernel (Br=16 rows/WG share a
        // cooperatively-loaded KV SLM tile → ~Br× fewer redundant KV reads, 2.05×
        // vs naive @16K, parity ≤4K). NUMERICALLY IDENTICAL to full_attention_gemma
        // (same 1/sqrt(HD) scale with Q pre-scaled = net 1.0, online softmax, fp32
        // accum — v2 only reorganizes the KV read). v2 caps at head_dim 256
        // (q_vals[16], dpl=HD/16); the global layers (HD=512 → dpl=32) overflow it,
        // so they stay naive. T-GATED: v2's SLM-staging overhead REGRESSES small T
        // (4K 624 vs naive 729 = 0.86x; L2 absorbs the redundant KV reads), only
        // past-L2 does the ~Br-x read reduction win (6K ~+5-13%, 8K 297 vs 203 =
        // +46%). Crossover in (4096,6144] -> default minT=6144, env IE_GEMMA4_FA2_V2_MINT.
        // Opt-out IE_GEMMA4_NO_FA2_V2. Decode (T==1) stays naive.
        // PREFERRED: the "wide" TILE kernel (full_attention_fa2_prefill_tile_gemma)
        // — llama's flash-attn-tile structure generalized to head_dim 256/512:
        // per-lane COMPLETE KQ dots (NO per-key subgroup reduce, the v2 floor) +
        // half2 coalesced K/V SLM tiles → lifts the v2 8K wall AND handles the
        // global-512 layers (v2 can't). Same net-1.0 scale (Q pre-scaled), so it's
        // numerically consistent with naive/v2. Opt-out IE_GEMMA4_NO_FA2_TILE →
        // falls back to v2 (256 only). Decode (T==1) stays naive.
        static const bool fa2_tile = std::getenv("IE_GEMMA4_NO_FA2_TILE") == nullptr;
        static const bool fa2_v2   = std::getenv("IE_GEMMA4_NO_FA2_V2") == nullptr;
        // The TILE kernel does NOT regress at small T (unlike v2) — it's faster at
        // EVERY tested length (4K 1281 vs v2/naive 497 = 2.58×; 2K +24%; 8K/16K win),
        // so it gets a LOW gate (2048). v2 (a fallback only) keeps its higher gate
        // (6144) since its SLM overhead regresses below the L2 crossover.
        static const uint32_t tile_minT = [] {
            const char* s = std::getenv("IE_GEMMA4_TILE_MINT");
            return s ? uint32_t(std::atoi(s)) : 2048u;
        }();
        static const uint32_t v2_minT = [] {
            const char* s = std::getenv("IE_GEMMA4_FA2_V2_MINT");
            return s ? uint32_t(std::atoi(s)) : 6144u;
        }();
        if (fa2_tile && T >= tile_minT && (HD == 256 || HD == 512)) {
            // Sliding-window for SWA layers — DEFAULT ON (the 16K lever AND a
            // correctness FIX: Gemma's SWA layers attend only to the last
            // `sliding_window`=1024 tokens; full causal was an O(T²) approximation
            // that DIVERGED from llama at T>window). Measured: 16K 460→1131 (2.4×,
            // beats llama 658), 8K 879→1564 (1.78×). Output CORRECT + sharper
            // (adversarial-reviewed mask; coherence: windowed answers "Dartmouth"
            // top-1 vs full-causal's echo). Global layers stay 0 (full causal).
            // Opt-out IE_GEMMA4_NO_SWA_WINDOW.
            static const bool swa_window = std::getenv("IE_GEMMA4_NO_SWA_WINDOW") == nullptr;
            const uint32_t window = (swa_window && w.is_swa && cfg_.sliding_window)
                                    ? cfg_.sliding_window : 0u;
            full_attention_fa2_prefill_tile_gemma(q, ws_q_, ws_k_, ws_v_, kcache_[L],
                                                  vcache_[L], ws_attn_out_, T, start_pos,
                                                  nq, NKV, HD, kv_ctx_, window);
        } else if (fa2_v2 && T >= v2_minT && HD <= 256) {
            full_attention_fa2_prefill_v2(q, ws_q_, ws_k_, ws_v_, kcache_[L], vcache_[L],
                                          ws_attn_out_, T, start_pos, nq, NKV, HD, kv_ctx_);
        } else {
            full_attention_gemma(q, ws_q_, ws_k_, ws_v_, kcache_[L], vcache_[L], ws_attn_out_,
                                  T, start_pos, nq, NKV, HD, kv_ctx_);
        }
        proj(w.attn_output, ws_attn_out_, ws_block_, NQ, H);              // o-proj
        rms_norm_f32w(q, ws_block_, w.post_attn_norm, ws_block_, T, H, eps);
        residual_add(q, ws_x_, ws_block_, ws_attn_out2_, uint64_t(T) * H); // attn_out = x + post_norm(o)
        // In-order queue already orders kernels; these per-layer host syncs are
        // leftover from op-by-op debugging. IE_GEMMA4_NO_LAYER_WAIT drops them to
        // kill ~3 host round-trips/layer of bubble (decode lever).
        static const bool no_layer_wait = std::getenv("IE_GEMMA4_NO_LAYER_WAIT") != nullptr;
        if (!no_layer_wait) q.wait();

        // ---- shared dense GeGLU FFN (on attn_out) ----
        rms_norm_f32w(q, ws_attn_out2_, w.ffn_norm, ws_xn_, T, H, eps);
        {
            const DenseQuantPtr* const Ws[3] = {&w.ffn_gate, &w.ffn_up, nullptr};
            sycl::half* const outs[3] = {ws_gate_, ws_up_, nullptr};
            const uint32_t Ns[3] = {cfg_.ffn, cfg_.ffn, 0};
            if (!fuse_decode(Ws, outs, Ns, 2, ws_xn_, H)) {
                proj(w.ffn_gate, ws_xn_, ws_gate_, H, cfg_.ffn);
                proj(w.ffn_up,   ws_xn_, ws_up_,   H, cfg_.ffn);
            }
        }
        geglu(q, ws_gate_, ws_up_, ws_h_, uint64_t(T) * cfg_.ffn);
        proj(w.ffn_down, ws_h_, ws_shared_y_, cfg_.ffn, H);
        if (w.is_moe) rms_norm_f32w(q, ws_shared_y_, w.post_ffw_norm_1, ws_shared_y_, T, H, eps);

        static const bool no_experts  = std::getenv("IE_GEMMA4_NO_EXPERTS") != nullptr;
        static const bool swap_gateup = std::getenv("IE_GEMMA4_SWAP_GATEUP") != nullptr;
        static const bool expert_sync = std::getenv("IE_GEMMA4_EXPERT_SYNC") != nullptr;
        if (w.is_moe && !no_experts) {
            // ---- routed experts ----
            rms_norm_f32w(q, ws_attn_out2_, w.pre_ffw_norm_2, ws_xn_, T, H, eps);  // expert input
            q.memset(ws_moe_y_, 0, uint64_t(T) * H * sizeof(sycl::half)).wait();
            const float inv_sqrt_h = 1.0f / std::sqrt(float(H));

            // Router logits: GPU (default) or host (IE_GEMMA4_HOST_ROUTER, A/B).
            // The GPU path replaces the per-token T×E×H host dot loop (single-
            // threaded, serialized with the device — the qwen3moe lever) with
            // rin = weightless_rms(attn_out)·router_in_scale·(1/√H) on device, then
            // ONE gemv_q_T logits[T,E] = rin @ router_w_devᵀ; only the cheap top-k
            // stays on host. Not bit-identical (fp16 rms/gemm vs host fp64 dot) but
            // PPL bit-stable + same experts (validated, ie-gemma4-ppl --reprefill).
            static const bool host_router = std::getenv("IE_GEMMA4_HOST_ROUTER") != nullptr;
            std::vector<std::vector<std::pair<uint32_t, float>>> all_routed(T);
            if (!host_router && w.router_w_dev) {
                rms_norm_f32w(q, ws_attn_out2_, w.router_in_scale_dev, ws_rin_, T, H, eps);
                dense::scale_inplace(q, ws_rin_, inv_sqrt_h, uint64_t(T) * H);
                // MTP spec-decode losslessness: the batched gemv_q_T(T) gives a
                // (tiny) different per-token logit than the T==1 decode path, which
                // can flip a near-tie top-k expert pick → the verify forward then
                // diverges from plain greedy (the router is the ONLY non-bit-exact
                // op vs decode — rms/scale are per-row exact, the expert GEMMs are
                // per-row int-dot). For the small-T VERIFY range, compute the router
                // PER-TOKEN (T==1 calls) so it is byte-identical to decode → strict
                // lossless. Prefill (large T) keeps the single batched call (faster,
                // determinism irrelevant). Opt-out IE_GEMMA4_NO_ROUTER_PERTOK.
                static const bool no_router_pertok = std::getenv("IE_GEMMA4_NO_ROUTER_PERTOK") != nullptr;
                if (!no_router_pertok && T > 1 && T <= 16) {
                    for (uint32_t t = 0; t < T; ++t)
                        dense::gemv_q_T(q, ws_rin_ + uint64_t(t) * H,
                                        DenseQuantPtr{w.router_w_dev, DType::kF16},
                                        ws_router_logits_ + uint64_t(t) * E, H, E, 1);
                } else {
                    dense::gemv_q_T(q, ws_rin_, DenseQuantPtr{w.router_w_dev, DType::kF16},
                                    ws_router_logits_, H, E, T);
                }
                std::vector<sycl::half> hl(uint64_t(T) * E);
                q.memcpy(hl.data(), ws_router_logits_, uint64_t(T) * E * sizeof(sycl::half)).wait();
                std::vector<float> logit(E);
                for (uint32_t t = 0; t < T; ++t) {
                    for (uint32_t e = 0; e < E; ++e) logit[e] = float(hl[uint64_t(t) * E + e]);
                    route_from_logits(logit.data(), E, K, all_routed[t]);
                }
            } else {
                std::vector<sycl::half> ao(uint64_t(T) * H);
                q.memcpy(ao.data(), ws_attn_out2_, uint64_t(T) * H * sizeof(sycl::half)).wait();
                std::vector<float> rin(H), logit(E);
                for (uint32_t t = 0; t < T; ++t) {
                    double ss = 0.0;
                    for (uint32_t h = 0; h < H; ++h) { float v = float(ao[uint64_t(t) * H + h]); ss += double(v) * v; }
                    const float rms = 1.0f / std::sqrt(float(ss / H) + eps);
                    for (uint32_t h = 0; h < H; ++h)
                        rin[h] = float(ao[uint64_t(t) * H + h]) * rms * inv_sqrt_h *
                                 (w.router_in_scale_h.empty() ? 1.0f : w.router_in_scale_h[h]);
                    for (uint32_t e = 0; e < E; ++e) {
                        const float* row = w.router_w.data() + uint64_t(e) * H;
                        float acc = 0.f;
                        for (uint32_t h = 0; h < H; ++h) acc += row[h] * rin[h];
                        logit[e] = acc;
                    }
                    route_from_logits(logit.data(), E, K, all_routed[t]);
                }
            }

            static const bool fused_moe = std::getenv("IE_GEMMA4_NO_FUSED_MOE") == nullptr;
            if (fused_moe && ws_xp_packed_) {
                // ---- batched fused-MoE: gather-by-expert → per-expert batched
                // gemm_q4_0 (gate_up, down) → scatter-reduce. Replaces the
                // T×K×2 single-token GEMV launches with E×2 batched GEMMs that
                // amortize each expert's weight read across all its routed
                // tokens. Bit-faithful math (same Q4_0 GEMM + same gelu-tanh).
                const uint32_t TK = T * K;
                MoePacking pk;
                build_moe_packing(all_routed, E, K, pk);
                // Fold the per-expert ffn_down scale into the per-row routing
                // weight that moe_prefill_reduce applies on scatter.
                std::vector<sycl::half> wpacked(TK);
                for (uint32_t e = 0; e < E; ++e) {
                    const float ds = w.down_scale_h.empty() ? 1.0f : w.down_scale_h[e];
                    for (uint32_t i = pk.expert_offsets[e]; i < pk.expert_offsets[e + 1]; ++i)
                        wpacked[i] = sycl::half(pk.weights_packed[i] * ds);
                }
                q.memcpy(ws_sorted_idx_, pk.sorted_idx.data(), uint64_t(TK) * sizeof(int32_t));
                q.memcpy(ws_tk_to_packed_, pk.tk_to_packed.data(), uint64_t(TK) * sizeof(int32_t));
                q.memcpy(ws_weights_packed_, wpacked.data(), uint64_t(TK) * sizeof(sycl::half));
                // Gather the expert input rows into expert-sorted order.
                moe_gather_rows(q, ws_xn_, ws_sorted_idx_, ws_xp_packed_, TK, H);
                static const bool no_q8 = std::getenv("IE_GEMMA4_NO_Q8") != nullptr;
                if (!no_q8) {
                    // ---- M1: int-dot W4A8 — single-launch over all experts ----
                    q.memcpy(ws_expert_offsets_, pk.expert_offsets.data(),
                             uint64_t(E + 1) * sizeof(uint32_t));
                    // gate_up XMX/DPAS GEMM — MEASURED A LOSS (632→420 tok/s):
                    // the matrix engine starves at ~30 rows/expert (small-M), where
                    // the dp4a int-dot kernel wins (opposite of the M=477 dense
                    // projections, where XMX wins). Kept opt-in (IE_GEMMA4_MOE_XMX)
                    // — only useful at large batch/T where per-expert M grows.
                    static const bool moe_xmx = std::getenv("IE_GEMMA4_MOE_XMX") != nullptr;
                    if (w.gu_qs) {
                        // ---- oneDNN weight-stationary fp16 MoE GEMM (large-M prefill) ----
                        // At prefill T>=4096 the per-expert M = T*K/E = T/16 >= 256
                        // (Coder's winning oneDNN regime); our gemm_fp16 / int-dot
                        // were 18-29% of B70 peak — oneDNN (~1.65×, fp16-direct, no
                        // q8 quantize) put Coder MoE prefill AHEAD of llama. The old
                        // "MoE-XMX is a loss" note was T~512 (~30 rows/expert, small-M)
                        // AND used gemm_fp16, not oneDNN. Below minT the int-dot SoA
                        // path wins (small-M), so T-gate. Opt-out IE_GEMMA4_NO_MOE_XMX.
                        static const bool moe_xmx_onednn =
                            std::getenv("IE_GEMMA4_NO_MOE_XMX") == nullptr;
                        static const uint32_t moe_xmx_minT = [] {
                            const char* s = std::getenv("IE_GEMMA4_MOE_XMX_MINT");
                            return s ? uint32_t(std::atoi(s)) : 4096u;
                        }();
                        const uint64_t bt_need = uint64_t(H) * (2 * EF);  // gate_up Bt (the larger)
                        bool xmx_ok = false;
                        if (moe_xmx_onednn && T >= moe_xmx_minT &&
                            (H % 32 == 0) && (EF % 32 == 0)) {
                            if (bt_need > ws_btf16_cap_) {
                                ws_btf16_ = static_cast<sycl::half*>(
                                    alloc_->malloc(bt_need * sizeof(sycl::half)));
                                if (ws_btf16_) { owned_.push_back(ws_btf16_); ws_btf16_cap_ = bt_need; }
                            }
                            xmx_ok = ws_btf16_ && ws_btf16_cap_ >= bt_need;
                        }
                        if (xmx_ok) {
                            moe_onednn_proj(q, ws_xp_packed_, w.gu_qs, w.gu_d,
                                            w.gu_qs_stride, w.gu_d_stride, ws_gu_packed_,
                                            pk.expert_offsets.data(), H, 2 * EF);
                            geglu_rows(q, ws_gu_packed_, ws_h_packed_, TK, EF, swap_gateup);
                            moe_onednn_proj(q, ws_h_packed_, w.dn_qs, w.dn_d,
                                            w.dn_qs_stride, w.dn_d_stride, ws_out_packed_,
                                            pk.expert_offsets.data(), EF, H);
                        } else {
                        // ---- SoA-Q4_0 int-dot (lane-coalesced; prefill + decode) ----
                        quantize_q8_1s(q, ws_xp_packed_, ws_xq8_, uint64_t(TK) * H);
                        moe_prefill_proj_q4_0_soa_q8(q, ws_xq8_, w.gu_qs, w.gu_d, ws_expert_offsets_,
                                                     ws_gu_packed_, E, H, 2 * EF,
                                                     w.gu_qs_stride, w.gu_d_stride);
                        geglu_rows(q, ws_gu_packed_, ws_h_packed_, TK, EF, swap_gateup);
                        quantize_q8_1s(q, ws_h_packed_, ws_hq8_, uint64_t(TK) * EF);
                        moe_prefill_proj_q4_0_soa_q8(q, ws_hq8_, w.dn_qs, w.dn_d, ws_expert_offsets_,
                                                     ws_out_packed_, E, EF, H,
                                                     w.dn_qs_stride, w.dn_d_stride);
                        }
                    } else {
                    if (moe_xmx && (H % 256 == 0) && ((2 * EF) % 64 == 0)) {
                        moe_prefill_proj_q4_0_xmx(q, ws_xp_packed_, w.gate_up_exps,
                                                  ws_expert_offsets_, ws_gu_packed_,
                                                  E, H, 2 * EF, w.gate_up_stride);
                    } else {
                        quantize_q8_1s(q, ws_xp_packed_, ws_xq8_, uint64_t(TK) * H);
                        moe_prefill_proj_q4_0_q8(q, ws_xq8_, w.gate_up_exps, ws_expert_offsets_,
                                                 ws_gu_packed_, E, H, 2 * EF, w.gate_up_stride);
                    }
                    geglu_rows(q, ws_gu_packed_, ws_h_packed_, TK, EF, swap_gateup);
                    quantize_q8_1s(q, ws_h_packed_, ws_hq8_, uint64_t(TK) * EF);
                    moe_prefill_proj_q4_0_q8(q, ws_hq8_, w.down_exps, ws_expert_offsets_,
                                             ws_out_packed_, E, EF, H, w.down_stride);
                    }
                } else {
                    // ---- M0: batched fp16 — per-expert gemm_q4_0 ----
                    for (uint32_t e = 0; e < E; ++e) {
                        const uint32_t off = pk.expert_offsets[e];
                        const uint32_t cnt = pk.expert_offsets[e + 1] - off;
                        if (cnt == 0) continue;
                        const auto* gu = static_cast<const uint8_t*>(w.gate_up_exps) + uint64_t(e) * w.gate_up_stride;
                        gemm_q4_0(q, ws_xp_packed_ + uint64_t(off) * H, gu,
                                  ws_gu_packed_ + uint64_t(off) * 2 * EF, cnt, H, 2 * EF);
                    }
                    geglu_rows(q, ws_gu_packed_, ws_h_packed_, TK, EF, swap_gateup);
                    for (uint32_t e = 0; e < E; ++e) {
                        const uint32_t off = pk.expert_offsets[e];
                        const uint32_t cnt = pk.expert_offsets[e + 1] - off;
                        if (cnt == 0) continue;
                        const auto* dn = static_cast<const uint8_t*>(w.down_exps) + uint64_t(e) * w.down_stride;
                        gemm_q4_0(q, ws_h_packed_ + uint64_t(off) * EF, dn,
                                  ws_out_packed_ + uint64_t(off) * H, cnt, EF, H);
                    }
                }
                // Scatter-reduce into ws_moe_y_ (zeroed above) with routing weights.
                // tk_to_packed holds non-negative packed-row indices; reinterpret
                // int32->uint32 (matches qwen3moe.cpp).
                moe_prefill_reduce(q, ws_out_packed_,
                                   reinterpret_cast<const uint32_t*>(ws_tk_to_packed_),
                                   ws_weights_packed_, ws_moe_y_, T, K, H);
            } else {
                // expert dispatch (unchanged math): per token, per routed expert.
                for (uint32_t t = 0; t < T; ++t) {
                    const sycl::half* e_in = ws_xn_ + uint64_t(t) * H;
                    for (const auto& [e, wgt] : all_routed[t]) {
                        const auto* gu = static_cast<const uint8_t*>(w.gate_up_exps) + uint64_t(e) * w.gate_up_stride;
                        const auto* dn = static_cast<const uint8_t*>(w.down_exps)    + uint64_t(e) * w.down_stride;
                        gemv_q4_0(q, e_in, gu, ws_e_gateup_, H, 2 * EF);                 // [gate|up]
                        if (swap_gateup)
                            geglu(q, ws_e_gateup_ + EF, ws_e_gateup_, ws_e_h_, EF);      // gelu(up)*gate
                        else
                            geglu(q, ws_e_gateup_, ws_e_gateup_ + EF, ws_e_h_, EF);      // gelu(gate)*up
                        gemv_q4_0(q, ws_e_h_, dn, ws_e_out_, EF, H);                     // down
                        const float coef = wgt * (w.down_scale_h.empty() ? 1.0f : w.down_scale_h[e]);
                        scaled_add(q, ws_e_out_, sycl::half(coef), ws_moe_y_ + uint64_t(t) * H, H);
                        if (expert_sync) q.wait();   // default off — in-order queue serializes
                    }
                }
            }
            rms_norm_f32w(q, ws_moe_y_, w.post_ffw_norm_2, ws_moe_y_, T, H, eps);
            residual_add(q, ws_shared_y_, ws_moe_y_, ws_block_, uint64_t(T) * H);   // shared + experts
        } else {
            q.memcpy(ws_block_, ws_shared_y_, uint64_t(T) * H * sizeof(sycl::half)).wait();
        }
        if (!no_layer_wait) q.wait();

        // ---- outer post-FFN norm + residual + per-layer output scale ----
        static const bool no_layerscale = std::getenv("IE_GEMMA4_NO_LAYERSCALE") != nullptr;
        rms_norm_f32w(q, ws_block_, w.post_ffw_norm, ws_block_, T, H, eps);
        residual_add(q, ws_attn_out2_, ws_block_, ws_x_, uint64_t(T) * H);
        if (!no_layerscale && w.layer_out_scale_val != 1.0f)
            dense::scale_inplace(q, ws_x_, w.layer_out_scale_val, uint64_t(T) * H);
        if (!no_layer_wait) q.wait();

        // ---- env-gated per-layer probe (IE_GEMMA4_DBG): absmax/mean of the
        // layer output (last token) to localize where a forward diverges. ----
        static const bool dbg = std::getenv("IE_GEMMA4_DBG") != nullptr;
        if (dbg) {
            std::vector<sycl::half> hx(H);
            q.memcpy(hx.data(), ws_x_ + uint64_t(T - 1) * H, H * sizeof(sycl::half)).wait();
            float amax = 0.f; double asum = 0.0; uint32_t nnan = 0;
            for (uint32_t i = 0; i < H; ++i) {
                float v = float(hx[i]);
                if (std::isnan(v) || std::isinf(v)) ++nnan;
                amax = std::max(amax, std::fabs(v)); asum += std::fabs(v);
            }
            std::fprintf(stderr, "[DBG L%2u] swa=%d HD=%u nkv=%u nrot=%u theta=%.0f  absmax=%.4g mean|.|=%.4g nan/inf=%u\n",
                         L, int(w.is_swa), w.head_dim, w.n_kv, w.n_rot, w.rope_theta,
                         amax, asum / H, nnan);
        }
    }

    // MTP self-spec: hand the pre-output_norm residual stream (the head's inp_h)
    // to the caller BEFORE output_norm consumes it. [T,H] device→device copy.
    if (hidden_pre_norm)
        q.memcpy(hidden_pre_norm, ws_x_, uint64_t(T) * H * sizeof(sycl::half));

    // final norm → lm_head (tied Q6_K) → soft-cap
    rms_norm_f32w(q, ws_x_, output_norm_, ws_xn_, T, H, eps);
    {
        static const bool dbgf = std::getenv("IE_GEMMA4_DBG") != nullptr;
        if (dbgf) {
            std::vector<sycl::half> hf(H);
            q.memcpy(hf.data(), ws_xn_ + uint64_t(T - 1) * H, H * sizeof(sycl::half)).wait();
            float amax = 0.f; double asum = 0.0;
            for (uint32_t i = 0; i < H; ++i) { float v = float(hf[i]); amax = std::max(amax, std::fabs(v)); asum += std::fabs(v); }
            std::fprintf(stderr, "[DBG FINALNORM] absmax=%.4g mean|.|=%.4g (output_norm applied)\n", amax, asum / H);
        }
    }
    sycl::half* last = ws_xn_ + uint64_t(T - 1) * H;
    sycl::event ev;
    if (lmh_q6_lo_ && !no_q4_soa) {       // fast SoA-Q6 lm_head (W6A8 int-dot)
        sycl::event qe = quantize_q8_1(q, last, ws_q8_, H);
        ev = gemv_q6_soa_q8(q, ws_q8_, lmh_q6_lo_, lmh_q6_hi_, lmh_q6_sc_, lmh_q6_d_,
                            out_logits, H, cfg_.vocab, {qe});
    } else {
        ev = gemv_q6_K(q, last, token_embd_, out_logits, H, cfg_.vocab);
    }
    static const bool no_softcap = std::getenv("IE_GEMMA4_NO_SOFTCAP") != nullptr;
    if (!no_softcap && cfg_.final_logit_softcap > 0.f)
        ev = softcap_inplace(q, out_logits, cfg_.vocab, cfg_.final_logit_softcap);

    // MTP spec-decode VERIFY: per-position logits for ALL T rows (ws_xn_ already
    // holds output_norm(x) for every row). The tied lm_head is huge (~1.1 GB), so
    // the BATCHED SoA-Q6 int-dot GEMV reads it ONCE and dots vs all T staged Q8_1
    // rows — vs a per-row loop that re-reads the weight T× (the dominant cost that
    // made higher K unprofitable). Per-row bit-identical to the single-row path.
    if (all_logits) {
        if (lmh_q6_lo_ && !no_q4_soa) {
            sycl::event qe = quantize_q8_1(q, ws_xn_, ws_q8b_, uint64_t(T) * H);
            ev = gemv_q6_soa_q8_batched(q, ws_q8b_, lmh_q6_lo_, lmh_q6_hi_, lmh_q6_sc_,
                                        lmh_q6_d_, all_logits, H, cfg_.vocab, T, {qe});
        } else {
            for (uint32_t t = 0; t < T; ++t)
                ev = gemv_q6_K(q, ws_xn_ + uint64_t(t) * H, token_embd_,
                               all_logits + uint64_t(t) * cfg_.vocab, H, cfg_.vocab);
        }
        if (!no_softcap && cfg_.final_logit_softcap > 0.f)
            ev = softcap_inplace(q, all_logits, uint64_t(T) * cfg_.vocab, cfg_.final_logit_softcap);
    }
    return ev;
}

// =====================================================================
// MTP self-speculative decode (gemma4-assistant draft head)
// =====================================================================
// Lifted from the validated tools/ie_gemma4_spec.cpp. The head's Q-only attention
// SHARES this model's KV (read-only, last 2 layers); rollback is implicit (the
// verify forward writes KV by absolute position → the next forward overwrites
// stale draft keys). Strictly token-lossless vs plain greedy.

std::string Gemma4Model::load_mtp_head(const GgufReader& gh, uint32_t max_ctx) {
    (void)max_ctx;  // head writes no KV (reads target KV[0,p)); no own cache
    GemmaAssistantConfig hc;
    if (auto e = read_gemma4_assistant_config(gh, hc); !e.empty()) return "mtp head: " + e;
    Mtp& m = mtp_;
    m.H = hc.hidden; m.BB = hc.n_embd_backbone; m.F = hc.ffn;
    m.vocab = hc.vocab; m.n_q = hc.n_q_heads; m.eps = hc.rms_eps;
    if (m.vocab != cfg_.vocab) return "mtp head: vocab mismatch";
    if (m.BB != cfg_.hidden)   return "mtp head: backbone != target hidden";

    std::string err; char buf[96];
    auto HT  = [&](const char* n){ return gh.find_tensor(n); };
    auto HTl = [&](uint32_t l, const char* n){ std::snprintf(buf,sizeof(buf),"blk.%u.%s",l,n); return gh.find_tensor(buf); };
    auto upf32 = [&](const GgufTensorInfo* t, const char* what) -> float* {
        if (!t) { err = std::string(what)+": not found"; return nullptr; }
        auto* p = dense::upload<float>(*alloc_, t, owned_, err, DType::kF32);
        if (!err.empty()) err = std::string(what)+": "+err;
        return p;
    };
    auto upq8 = [&](const GgufTensorInfo* t, const char* what) -> MtpW {
        MtpW w{};
        if (!t) { err = std::string(what)+": not found"; return w; }
        if (t->dtype != DType::kQ8_0) { err = std::string(what)+": expected Q8_0"; return w; }
        w.K = uint32_t(t->shape[0]); w.N = uint32_t(t->shape[1]);
        if (w.K % 32 != 0) { err = std::string(what)+": K%32"; return w; }
        const uint32_t K=w.K, N=w.N, bpc=K/32;
        const auto* blocks = reinterpret_cast<const block_q8_0*>(t->data);
        std::vector<int8_t>   qs(uint64_t(N)*K);
        std::vector<uint16_t> dd(uint64_t(N)*bpc);
        for (uint64_t n=0;n<N;++n) for (uint32_t b=0;b<bpc;++b){
            const block_q8_0& blk = blocks[n*bpc+b];
            dd[n*bpc+b] = *reinterpret_cast<const uint16_t*>(&blk.d);
            for (int i=0;i<32;++i) qs[n*K+uint64_t(b)*32+i] = blk.qs[i];
        }
        w.qs = static_cast<int8_t*>(alloc_->malloc(qs.size()));
        w.d  = static_cast<uint16_t*>(alloc_->malloc(dd.size()*sizeof(uint16_t)));
        if (!w.qs || !w.d) { err = std::string(what)+": alloc"; return w; }
        owned_.push_back(w.qs); owned_.push_back(w.d);
        alloc_->queue().memcpy(w.qs, qs.data(), qs.size()).wait();
        alloc_->queue().memcpy(w.d,  dd.data(), dd.size()*sizeof(uint16_t)).wait();
        m.max_K = std::max(m.max_K, w.K);
        return w;
    };

    m.max_K = 0;
    m.pre_proj  = upq8(HT("nextn.pre_projection.weight"),  "pre_proj");  if (!err.empty()) return err;
    m.post_proj = upq8(HT("nextn.post_projection.weight"), "post_proj"); if (!err.empty()) return err;
    m.tok_embd  = upq8(HT("token_embd.weight"),            "tok_embd");  if (!err.empty()) return err;
    m.output_norm = upf32(HT("output_norm.weight"),        "output_norm"); if (!err.empty()) return err;
    float* rope_freqs_shared = nullptr;
    if (const auto* rf = HT("rope_freqs.weight")) { rope_freqs_shared = upf32(rf,"rope_freqs"); if (!err.empty()) return err; }

    m.L.resize(hc.n_layers); m.max_Nq = 0;
    for (uint32_t l=0;l<hc.n_layers;++l) {
        MtpLayer& w = m.L[l];
        w.is_swa=hc.is_swa[l]!=0; w.head_dim=hc.head_dim[l]; w.n_kv=hc.n_kv_heads[l];
        w.n_rot=hc.n_rot[l]; w.theta=hc.rope_theta[l];
        m.max_Nq = std::max(m.max_Nq, m.n_q*w.head_dim);
        w.attn_norm      = upf32(HTl(l,"attn_norm.weight"),"attn_norm");                if (!err.empty()) return err;
        w.attn_q_norm    = upf32(HTl(l,"attn_q_norm.weight"),"attn_q_norm");            if (!err.empty()) return err;
        w.post_attn_norm = upf32(HTl(l,"post_attention_norm.weight"),"post_attn");      if (!err.empty()) return err;
        w.ffn_norm       = upf32(HTl(l,"ffn_norm.weight"),"ffn_norm");                  if (!err.empty()) return err;
        w.post_ffw_norm  = upf32(HTl(l,"post_ffw_norm.weight"),"post_ffw_norm");        if (!err.empty()) return err;
        if (!w.is_swa) w.rope_freqs = rope_freqs_shared;
        if (const auto* os = HTl(l,"layer_output_scale.weight")) w.out_scale = *reinterpret_cast<const float*>(os->data);
        w.wq=upq8(HTl(l,"attn_q.weight"),"attn_q");        if (!err.empty()) return err;
        w.wo=upq8(HTl(l,"attn_output.weight"),"attn_output"); if (!err.empty()) return err;
        w.fg=upq8(HTl(l,"ffn_gate.weight"),"ffn_gate");    if (!err.empty()) return err;
        w.fu=upq8(HTl(l,"ffn_up.weight"),"ffn_up");        if (!err.empty()) return err;
        w.fd=upq8(HTl(l,"ffn_down.weight"),"ffn_down");    if (!err.empty()) return err;
    }
    auto A=[&](uint64_t n)->sycl::half*{ auto* p=static_cast<sycl::half*>(alloc_->malloc(n*sizeof(sycl::half))); if(p) owned_.push_back(p); return p; };
    m.xemb=A(m.BB); m.xh=A(2u*m.BB); m.inph=A(m.BB); m.cur=A(m.H); m.curn=A(m.H);
    m.qbuf=A(m.max_Nq); m.ao=A(m.max_Nq); m.blk=A(m.H); m.attnout=A(m.H);
    m.fgb=A(m.F); m.fub=A(m.F); m.fhb=A(m.F); m.hnext=A(m.BB); m.logits=A(m.vocab);
    m.actq8 = alloc_->malloc((m.max_K/32)*sizeof(block_q8_1x)); if (m.actq8) owned_.push_back(m.actq8);
    m.pos   = static_cast<int32_t*>(alloc_->malloc(sizeof(int32_t))); if (m.pos) owned_.push_back(m.pos);
    m.tokid = static_cast<int32_t*>(alloc_->malloc(sizeof(int32_t))); if (m.tokid) owned_.push_back(m.tokid);
    if (!m.logits || !m.actq8 || !m.pos || !m.tokid) return "mtp head: scratch alloc failed";
    m.loaded = true;
    std::printf("[gemma4-mtp] head loaded: %zu layers, H=%u BB=%u F=%u (SoA-Q8 int-dot draft)\n",
                m.L.size(), m.H, m.BB, m.F);
    return {};
}

void Gemma4Model::mtp_run(sycl::queue& q, const sycl::half* inp_h, int32_t tok, uint32_t p) {
    Mtp& m = mtp_;
    const uint32_t H=m.H, BB=m.BB, F=m.F;
    auto gv=[&](const sycl::half* in, const MtpW& W, sycl::half* y){
        sycl::event qe = quantize_q8_1(q, in, m.actq8, W.K);
        gemv_q8_0_soa_q8(q, m.actq8, W.qs, W.d, y, W.K, W.N, {qe});
    };
    auto scl=[&](sycl::half* x, float c, uint64_t n){
        q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i){ x[i]=sycl::half(float(x[i])*c); });
    };
    // x = pre_proj( concat( target_embd(tok)*sqrt(BB), inp_h ) )
    q.memcpy(m.tokid, &tok, sizeof(int32_t));
    if (token_embd_dtype_==DType::kQ6_K) embedding_lookup_q6k(q, m.tokid, token_embd_, m.xemb, 1, BB);
    else                                 embedding_lookup_q4k(q, m.tokid, token_embd_, m.xemb, 1, BB);
    scl(m.xemb, std::sqrt(float(BB)), BB);
    q.memcpy(m.xh,      m.xemb, uint64_t(BB)*sizeof(sycl::half));
    q.memcpy(m.xh+BB,   inp_h,  uint64_t(BB)*sizeof(sycl::half));
    gv(m.xh, m.pre_proj, m.cur);
    const int32_t posv=int32_t(p); q.memcpy(m.pos, &posv, sizeof(int32_t));
    for (uint32_t il=0; il<m.L.size(); ++il) {
        const MtpLayer& w = m.L[il];
        const uint32_t HD=w.head_dim, Nq=m.n_q*HD;
        const uint32_t Lsh = w.is_swa ? (cfg_.n_layers-2) : (cfg_.n_layers-1);
        rms_norm_f32w(q, m.cur, w.attn_norm, m.curn, 1, H, m.eps);
        gv(m.curn, w.wq, m.qbuf);
        rms_norm_f32w(q, m.qbuf, w.attn_q_norm, m.qbuf, m.n_q, HD, m.eps);
        scl(m.qbuf, std::sqrt(float(HD)), uint64_t(Nq));
        if (!w.is_swa && w.rope_freqs)
            rope_partial_ff(q, m.qbuf, m.pos, m.qbuf, 1, m.n_q, HD, w.n_rot, w.theta, w.rope_freqs);
        else
            rope_partial(q, m.qbuf, m.pos, m.qbuf, 1, m.n_q, HD, w.n_rot, w.theta);
        read_attention_gemma(q, m.qbuf, kcache_[Lsh], vcache_[Lsh], m.ao, p, m.n_q, w.n_kv, HD, kv_ctx_);
        gv(m.ao, w.wo, m.blk);
        rms_norm_f32w(q, m.blk, w.post_attn_norm, m.blk, 1, H, m.eps);
        residual_add(q, m.cur, m.blk, m.attnout, uint64_t(H));
        rms_norm_f32w(q, m.attnout, w.ffn_norm, m.curn, 1, H, m.eps);
        gv(m.curn, w.fg, m.fgb);
        gv(m.curn, w.fu, m.fub);
        geglu(q, m.fgb, m.fub, m.fhb, uint64_t(F));
        gv(m.fhb, w.fd, m.blk);
        rms_norm_f32w(q, m.blk, w.post_ffw_norm, m.blk, 1, H, m.eps);
        residual_add(q, m.attnout, m.blk, m.cur, uint64_t(H));
        if (w.out_scale != 1.0f) scl(m.cur, w.out_scale, uint64_t(H));
    }
    rms_norm_f32w(q, m.cur, m.output_norm, m.curn, 1, H, m.eps);
    gv(m.curn, m.tok_embd,  m.logits);   // draft logits
    gv(m.curn, m.post_proj, m.hnext);    // recurrence hidden
}

std::string Gemma4Model::spec_generate(sycl::queue& q, sycl::half* d_hlast, int32_t tn,
                                       uint32_t start_pos, uint32_t max_new, uint32_t K,
                                       const std::function<bool(int32_t)>& emit) {
    if (!mtp_.loaded) return "mtp head not loaded";
    Mtp& m = mtp_;
    const uint32_t V = cfg_.vocab, BB = m.BB;
    if (K < 1) K = 1;
    auto* d_ids    = static_cast<int32_t*>(alloc_->malloc(uint64_t(K)*sizeof(int32_t)));
    auto* d_all    = static_cast<sycl::half*>(alloc_->malloc(uint64_t(K)*V*sizeof(sycl::half)));
    auto* d_hid    = static_cast<sycl::half*>(alloc_->malloc(uint64_t(K)*BB*sizeof(sycl::half)));
    auto* d_lastlog= static_cast<sycl::half*>(alloc_->malloc(uint64_t(V)*sizeof(sycl::half)));
    auto cleanup = [&]{ if(d_ids)alloc_->free(d_ids); if(d_all)alloc_->free(d_all); if(d_hid)alloc_->free(d_hid); if(d_lastlog)alloc_->free(d_lastlog); };
    if (!d_ids || !d_all || !d_hid || !d_lastlog) { cleanup(); return "spec scratch alloc failed"; }

    std::vector<sycl::half> Lrow(uint64_t(K)*V), row(V);
    std::vector<int32_t> drafted, vin(K), targ(K);
    KvCache dummy;  // forward ignores it (Gemma self-manages KV)
    auto argmax = [&](const sycl::half* r)->int32_t{ float b=float(r[0]); int a=0; for(uint32_t v=1;v<V;++v){float x=float(r[v]); if(x>b){b=x;a=int(v);}} return a; };

    uint32_t p = start_pos, produced = 0; bool stop = false, first = true;
    auto put = [&](int32_t t)->bool{
        if (produced >= max_new) { stop = true; return false; }
        if (!emit(t))            { stop = true; return false; }
        ++produced; return true;
    };
    while (produced < max_new && !stop) {
        // ---- draft K (head reads target KV[0,p), fixed pos p) ----
        drafted.clear();
        q.memcpy(m.inph, d_hlast, uint64_t(BB)*sizeof(sycl::half)).wait();
        int32_t tok = tn;
        for (uint32_t j=0;j<K;++j) {
            mtp_run(q, m.inph, tok, p); q.wait();
            q.memcpy(row.data(), m.logits, uint64_t(V)*sizeof(sycl::half)).wait();
            int32_t g = argmax(row.data());
            drafted.push_back(g);
            q.memcpy(m.inph, m.hnext, uint64_t(BB)*sizeof(sycl::half)).wait();
            tok = g;
        }
        // ---- verify forward(T=K, start_pos=p) on [tn, g_1..g_{K-1}] ----
        vin[0]=tn; for (uint32_t j=1;j<K;++j) vin[j]=drafted[j-1];
        q.memcpy(d_ids, vin.data(), K*sizeof(int32_t)).wait();
        forward(q, d_ids, K, p, dummy, d_lastlog, /*hidden_pre_norm=*/d_hid, /*all_logits=*/d_all).wait();
        q.memcpy(Lrow.data(), d_all, uint64_t(K)*V*sizeof(sycl::half)).wait();
        for (uint32_t j=0;j<K;++j) targ[j]=argmax(Lrow.data()+uint64_t(j)*V);
        uint32_t n=0; for (uint32_t j=1;j<K;++j){ if(drafted[j-1]==targ[j-1]) ++n; else break; }
        const int32_t bonus = targ[n];
        const uint32_t accepted = n + 1;          // tn + g_1..g_n
        // ---- emit committed tokens IN ORDER (round0 emits tn too; later rounds
        //      had tn == previous bonus already emitted) ----
        if (first) { first = false; if (!put(tn)) break; }
        for (uint32_t j=0;j<n && !stop;++j) put(drafted[j]);
        if (!stop) put(bonus);
        // ---- advance (gemma KV needs NO rollback: next forward at p+accepted
        //      overwrites stale draft keys) ----
        tn = bonus;
        q.memcpy(d_hlast, d_hid + uint64_t(accepted-1)*BB, uint64_t(BB)*sizeof(sycl::half)).wait();
        p += accepted;
    }
    cleanup();
    return {};
}

}  // namespace ie
