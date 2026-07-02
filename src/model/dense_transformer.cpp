// src/model/dense_transformer.cpp — P2: DenseModel (Qwen3-dense family)
// loader + workspace.  Forward pass lands in P2 Task 4.
//
// Mirrors QwenModel::load's structure (src/model/qwen36.cpp) with the dense
// tensor names; dispatch helpers live in dense_dispatch.hpp (copied from
// qwen36.cpp per the P2 iron rule — the crown file is never edited).

#include "ie/dense_transformer.hpp"

#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"
#include "ie/dequant_ref.hpp"   // dequant_q6_K_buffer (Q6_K→Q8_0 repack)

#include "dense_dispatch.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace ie {

std::vector<uint32_t> llama_qk_unpermute_rows(uint32_t n_heads, uint32_t head_dim) {
    std::vector<uint32_t> perm(uint64_t(n_heads) * head_dim);
    const uint32_t half = head_dim / 2u;
    for (uint32_t h = 0; h < n_heads; ++h)
        for (uint32_t k = 0; k < half; ++k)
            for (uint32_t b = 0; b < 2u; ++b)
                perm[uint64_t(h) * head_dim + b * half + k] = h * head_dim + 2u * k + b;
    return perm;
}

DenseModel::~DenseModel() {
    if (alloc_) for (void* p : owned_) alloc_->free(p);
}

std::string DenseModel::load(DeviceAllocator& alloc, const GgufReader& g,
                             const DenseConfig& cfg) {
    alloc_ = &alloc;
    cfg_ = cfg;

    // Load-time validation — fail fast, no silent paths.
    if (cfg.n_layers == 0 || cfg.hidden == 0 || cfg.n_q_heads == 0 ||
        cfg.n_kv_heads == 0 || cfg.ffn == 0) {
        return "dense config: zero dimension";
    }
    if (cfg.vocab == 0) return "dense config: vocab == 0";
    if (cfg.head_dim < 16 || (cfg.head_dim & (cfg.head_dim - 1)) != 0) {
        return "dense config: head_dim must be a power of two >= 16 "
               "(FA-2 hd_mask requirement), got " + std::to_string(cfg.head_dim);
    }
    if (cfg.hidden % 256 != 0) {
        return "dense config: hidden % 256 != 0 (Q4_K/Q6_K GEMV needs "
               "K % 256 == 0), got " + std::to_string(cfg.hidden);
    }
    if (cfg.ffn % 256 != 0) {
        return "dense config: feed_forward_length % 256 != 0 (down-proj "
               "GEMV needs K % 256 == 0), got " + std::to_string(cfg.ffn);
    }

    char buf[64];
    auto T = [&](const char* name) { return g.find_tensor(name); };
    auto Tlayer = [&](uint32_t L, const char* name) {
        std::snprintf(buf, sizeof(buf), "blk.%u.%s", L, name);
        return g.find_tensor(buf);
    };

    std::string err;

    // Top-level — token_embd may be Q4_K, Q6_K, or F16 (embedding_lookup kernels
    // exist for those; F16 is the EXL3 path — token_embd ships F16, kept faithful).
    {
        const auto* ti = T("token_embd.weight");
        if (!ti) return "token_embd: not found";
        if (ti->dtype != DType::kQ4_K && ti->dtype != DType::kQ6_K &&
            ti->dtype != DType::kF16) {
            return std::string("token_embd: unsupported dtype ") +
                   std::string(type_name(ti->dtype)) +
                   " (supported: Q4_K, Q6_K, F16)";
        }
        token_embd_dtype_ = ti->dtype;
        token_embd_ = dense::upload<void>(alloc, ti, owned_, err, ti->dtype);
        if (!err.empty()) return "token_embd: " + err;
    }
    output_norm_ = dense::upload<float>(alloc, T("output_norm.weight"),
                                        owned_, err, DType::kF32);
    if (!err.empty()) return "output_norm: " + err;
    {
        // 0.6B–4B Qwen3-dense GGUFs tie embeddings: output.weight is absent
        // and the lm_head IS token_embd (layouts identical — each output
        // column's superblocks are contiguous in both).
        const auto* ti = T("output.weight");
        if (!ti) {
            output_ = token_embd_;
            output_dtype_ = token_embd_dtype_;
        } else if (ti->dtype == DType::kEXL3) {
            // Native EXL3 lm_head (e.g. turboderp ships it as a separate 6bpw
            // trellis even when tie_word_embeddings=true). Decoded natively in the
            // logits GEMV (gemv_exl3_forward) — not dequantized.
            DenseQuantPtr o = dense::upload_exl3(alloc, ti, T("output.suh"),
                                                 T("output.svh"), owned_, err);
            if (!err.empty()) return "output(exl3): " + err;
            output_ = o.p; output_dtype_ = DType::kEXL3;
            output_suh_ = o.suh; output_svh_ = o.svh; output_bits_ = o.bits;
        } else {
            if (ti->dtype != DType::kQ4_K && ti->dtype != DType::kQ6_K) {
                return std::string("output: unsupported dtype ") +
                       std::string(type_name(ti->dtype)) +
                       " (supported: Q4_K, Q6_K, EXL3)";
            }
            output_dtype_ = ti->dtype;
            output_ = dense::upload<void>(alloc, ti, owned_, err, ti->dtype);
            if (!err.empty()) return "output: " + err;
        }
    }

    // P3a: llama3 RoPE frequency factors. OPTIONAL — absent for qwen3 and for
    // llama-2/3.0 GGUFs (→ nullptr = no scaling). When present, must hold
    // exactly rope_dim/2 factors (one per NEOX rotary pair).
    if (const auto* ti = T("rope_freqs.weight")) {
        rope_freqs_ = dense::upload<float>(alloc, ti, owned_, err, DType::kF32);
        if (!err.empty()) return "rope_freqs: " + err;
        const uint64_t n = ti->nbytes / sizeof(float);
        if (n != cfg.rope_dim / 2u)
            return "rope_freqs: count " + std::to_string(n) +
                   " != rope_dim/2 " + std::to_string(cfg.rope_dim / 2u);
    }

    // P3a: Llama stores attn_q/attn_k rows permuted (2,hd/2)→(hd/2,2) for
    // NORM-pairing RoPE; we un-permute at load so our NEOX rope_partial(_ff)
    // is exactly correct. perm[dest] = src (see plan §permute derivation):
    //   dest row h*hd + b*(hd/2) + k  ←  src row h*hd + 2k + b.
    // qwen3/qwen2 leave perm empty (no reorder). Built once (shape-invariant).
    const bool llama_unpermute = (cfg.arch == ModelArch::kLlama3);
    const std::vector<uint32_t> perm_q = llama_unpermute ? llama_qk_unpermute_rows(cfg.n_q_heads,  cfg.head_dim) : std::vector<uint32_t>{};
    const std::vector<uint32_t> perm_k = llama_unpermute ? llama_qk_unpermute_rows(cfg.n_kv_heads, cfg.head_dim) : std::vector<uint32_t>{};

    // Per-layer — every layer is full-attention + dense SwiGLU FFN.
    layers_.assign(cfg.n_layers, {});
    for (uint32_t L = 0; L < cfg.n_layers; ++L) {
        auto& w = layers_[L];
        auto layer_err = [&](const char* what) {
            return "layer " + std::to_string(L) + " " + what + ": " + err;
        };
        auto LQ = [&](const char* name) -> DenseQuantPtr {
            const auto* ti = Tlayer(L, name);
            // EXL3: route the trellis + its suh/svh siblings to upload_exl3 (native
            // decode at inference). Names: blk.L.<role>.{weight,suh,svh}.
            if (ti && ti->dtype == DType::kEXL3) {
                std::string base(name);
                const std::string sfx = ".weight";
                if (base.size() > sfx.size() &&
                    base.compare(base.size() - sfx.size(), sfx.size(), sfx) == 0)
                    base.resize(base.size() - sfx.size());
                return dense::upload_exl3(alloc, ti,
                    Tlayer(L, (base + ".suh").c_str()),
                    Tlayer(L, (base + ".svh").c_str()), owned_, err);
            }
            // _auto so a STANDALONE Q5_K/Q8_0 dense weight (e.g. a Q5_K_M quant)
            // also loads; Q4_K/Q6_K/F16 forward to upload_quant_dense unchanged
            // (byte-identical for every existing dense GGUF).
            return dense::upload_quant_dense_auto(alloc, ti, owned_, err);
        };
        // EXL3 weights are HF-natural NEOX order → never row-un-permuted (like phi3).
        auto is_exl3 = [&](const char* name) {
            const auto* ti = Tlayer(L, name);
            return ti && ti->dtype == DType::kEXL3;
        };

        w.attn_norm = dense::upload<float>(alloc, Tlayer(L, "attn_norm.weight"),
                                           owned_, err, DType::kF32);
        if (!err.empty()) return layer_err("attn_norm");
        w.ffn_norm  = dense::upload<float>(alloc, Tlayer(L, "ffn_norm.weight"),
                                           owned_, err, DType::kF32);
        if (!err.empty()) return layer_err("ffn_norm");

        // Phi-unblock Task 2: phi3 ships a FUSED attn_qkv [K, Nq+Nk+Nk] (Q‖K‖V).
        // Trigger ONLY when attn_q is absent AND attn_qkv is present → every
        // existing dense GGUF (separate attn_q) takes the else-branch unchanged.
        const bool fused_qkv = !Tlayer(L, "attn_q.weight") && Tlayer(L, "attn_qkv.weight");
        if (fused_qkv) {
            const auto* qkv = Tlayer(L, "attn_qkv.weight");
            const uint64_t Nq = uint64_t(cfg.n_q_heads)  * cfg.head_dim;   // 40*128 = 5120
            const uint64_t Nk = uint64_t(cfg.n_kv_heads) * cfg.head_dim;   // 10*128 = 1280
            if (qkv->n_dims != 2 || qkv->shape[1] != Nq + 2 * Nk) {
                err = "attn_qkv width " +
                      std::to_string(qkv->n_dims == 2 ? qkv->shape[1] : 0) +
                      " != Nq+2Nk " + std::to_string(Nq + 2 * Nk);
                return layer_err("attn_qkv");
            }
            // phi3 (LLM_ARCH_PHI3) stores Q/K in NATURAL NEOX order — unlike
            // genuine Llama-3, llama.cpp's phi3 converter does NOT interleave-
            // permute Q/K (phi3 rope_type == NEOX, no convert-time permute). Our
            // NEOX rope_partial consumes natural order directly, so the fused
            // QKV slices must NOT be un-permuted. (Empirically proven: with the
            // un-permute applied, token-0 — RoPE pos 0, rotation=identity — was
            // bit-exact while every ROTATED token diverged → a pure rope-pairing
            // error, the un-permute signature.) Pass an empty perm = slice only.
            const std::vector<uint32_t> qkv_perm{};   // phi3: no Q/K permute
            w.attn_q = dense::upload_quant_dense_row_slice_permuted(
                alloc, qkv, 0,       Nq, owned_, err, qkv_perm);  if (!err.empty()) return layer_err("attn_q(qkv)");
            w.attn_k = dense::upload_quant_dense_row_slice_permuted(
                alloc, qkv, Nq,      Nk, owned_, err, qkv_perm);  if (!err.empty()) return layer_err("attn_k(qkv)");
            w.attn_v = dense::upload_quant_dense_row_slice(
                alloc, qkv, Nq + Nk, Nk, owned_, err);          if (!err.empty()) return layer_err("attn_v(qkv)");
        } else {
            // attn_q/attn_k: row-un-permuted for llama (perm_* non-empty), plain otherwise.
            w.attn_q = (llama_unpermute && !is_exl3("attn_q.weight"))
                ? dense::upload_quant_dense_permuted(alloc, Tlayer(L, "attn_q.weight"), owned_, err, perm_q)
                : LQ("attn_q.weight");                if (!err.empty()) return layer_err("attn_q");
            w.attn_k = (llama_unpermute && !is_exl3("attn_k.weight"))
                ? dense::upload_quant_dense_permuted(alloc, Tlayer(L, "attn_k.weight"), owned_, err, perm_k)
                : LQ("attn_k.weight");                if (!err.empty()) return layer_err("attn_k");
            w.attn_v  = LQ("attn_v.weight");          if (!err.empty()) return layer_err("attn_v");
        }
        w.attn_output = LQ("attn_output.weight"); if (!err.empty()) return layer_err("attn_output");

        // Q/K-norm: present for Qwen3, ABSENT for Qwen2 — load if present.
        auto opt_f32 = [&](const char* name, float*& dst, const char* what) -> std::string {
            if (const auto* ti = Tlayer(L, name)) {
                dst = dense::upload<float>(alloc, ti, owned_, err, DType::kF32);
                if (!err.empty()) return layer_err(what);
            }
            return {};
        };
        if (auto m = opt_f32("attn_q_norm.weight", w.attn_q_norm, "attn_q_norm"); !m.empty()) return m;
        if (auto m = opt_f32("attn_k_norm.weight", w.attn_k_norm, "attn_k_norm"); !m.empty()) return m;
        // Attention QKV bias: present for Qwen2, ABSENT for Qwen3.
        if (auto m = opt_f32("attn_q.bias", w.attn_q_bias, "attn_q.bias"); !m.empty()) return m;
        if (auto m = opt_f32("attn_k.bias", w.attn_k_bias, "attn_k.bias"); !m.empty()) return m;
        if (auto m = opt_f32("attn_v.bias", w.attn_v_bias, "attn_v.bias"); !m.empty()) return m;

        // Phi-unblock Task 2: phi3 ships a FUSED ffn_up [K, 2*ffn] (gate‖up).
        // Trigger ONLY when ffn_gate is absent, ffn_up is present, AND its width
        // is exactly 2*ffn → a normal ffn_up (width = ffn) never trips this, and
        // every existing dense GGUF (separate ffn_gate) takes the else-branch.
        const auto* ffn_up_t = Tlayer(L, "ffn_up.weight");
        const bool fused_gate_up = !Tlayer(L, "ffn_gate.weight") && ffn_up_t &&
                                   ffn_up_t->n_dims == 2 &&
                                   ffn_up_t->shape[1] == 2ull * cfg.ffn;
        if (fused_gate_up) {
            // gate = rows [0, ffn), up = rows [ffn, 2*ffn). Neither is permuted.
            w.ffn_gate = dense::upload_quant_dense_row_slice(
                alloc, ffn_up_t, 0,        cfg.ffn, owned_, err); if (!err.empty()) return layer_err("ffn_gate(up)");
            w.ffn_up   = dense::upload_quant_dense_row_slice(
                alloc, ffn_up_t, cfg.ffn,  cfg.ffn, owned_, err); if (!err.empty()) return layer_err("ffn_up(up)");
        } else {
            w.ffn_gate = LQ("ffn_gate.weight"); if (!err.empty()) return layer_err("ffn_gate");
            w.ffn_up   = LQ("ffn_up.weight");   if (!err.empty()) return layer_err("ffn_up");
        }
        w.ffn_down = LQ("ffn_down.weight"); if (!err.empty()) return layer_err("ffn_down");

        // Q6_K-repack decode optimization (IE_DENSE_Q6K_REPACK): repack a Q6_K
        // ffn_down into a SoA Q8_0 copy (int8 qs col-major + fp16 d/32-block) so the
        // T==1 decode GEMV streams int8 with no 6-bit unpack — 2.96× vs the scalar
        // gemm_q6_K cliff, ≈equal/better accuracy (8-bit>6-bit), +30% VRAM on this
        // tensor (kept ALONGSIDE the Q6_K, which prefill still uses). Gate proven by
        // ie-q8-0-repack-bench. **Default ON (2026-06-15)** after the family-wide
        // PPL pass (Llama-3.1/3.2, Granite-3.3, Phi-4, Mistral-24B, Codestral-22B
        // all within ±0.22% NLL; decode +19%→~2.9×). Opt-out IE_DENSE_NO_Q6K_REPACK=1
        // restores the fp16 ffn_down decode (the bit-exact NLL 2.940491 leak-canary).
        static const bool no_repack = std::getenv("IE_DENSE_NO_Q6K_REPACK") != nullptr;
        if (!no_repack && w.ffn_down.dt == DType::kQ6_K) {
            const auto* dt = Tlayer(L, "ffn_down.weight");
            const uint32_t Kd = cfg.ffn, Nd = cfg.hidden;   // ffn_down: K=ffn, N=hidden
            if (Kd % 256 == 0) {
                const auto* blocks = reinterpret_cast<const block_q6_K*>(dt->data);
                const uint32_t q6bpc = Kd / 256, bpc = Kd / 32;
                std::vector<int8_t>   qs(uint64_t(Nd) * Kd);
                std::vector<uint16_t> dd(uint64_t(Nd) * bpc);
                std::vector<float>    wf(Kd);
                for (uint32_t n = 0; n < Nd; ++n) {
                    ref::dequant_q6_K_buffer(blocks + uint64_t(n) * q6bpc, Kd, wf.data());
                    for (uint32_t b = 0; b < bpc; ++b) {
                        float amax = 0;
                        for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(wf[b * 32 + i]));
                        const float d = amax / 127.f, inv = amax > 0 ? 127.f / amax : 0.f;
                        dd[uint64_t(n) * bpc + b] = fp32_to_fp16(d);
                        for (int i = 0; i < 32; ++i)
                            qs[uint64_t(n) * Kd + uint64_t(b) * 32 + i] =
                                int8_t(std::lround(wf[b * 32 + i] * inv));
                    }
                }
                auto* d_qs = static_cast<int8_t*>(alloc.malloc(qs.size()));
                auto* d_dd = static_cast<uint16_t*>(alloc.malloc(dd.size() * sizeof(uint16_t)));
                if (!d_qs || !d_dd) return layer_err("ffn_down Q8_0 repack alloc");
                owned_.push_back(d_qs); owned_.push_back(d_dd);
                alloc.queue().memcpy(d_qs, qs.data(), qs.size()).wait();
                alloc.queue().memcpy(d_dd, dd.data(), dd.size() * sizeof(uint16_t)).wait();
                w.ffn_down_q8_qs = d_qs;
                w.ffn_down_q8_d  = d_dd;
            }
        }
    }
    return {};
}

std::string DenseModel::ensure_workspace(uint32_t max_T) {
    if (max_T == 0 || !alloc_) return "alloc not initialized";
    if (max_T <= ws_T_) return {};

    // Old buffers stay in owned_ (freed at dtor) — same growth pattern as
    // QwenModel::ensure_workspace.
    auto alloc_half = [&](size_t n) {
        auto* p = static_cast<sycl::half*>(alloc_->malloc(n * sizeof(sycl::half)));
        if (p) owned_.push_back(p);
        return p;
    };
    auto alloc_i32 = [&](size_t n) {
        auto* p = static_cast<int32_t*>(alloc_->malloc(n * sizeof(int32_t)));
        if (p) owned_.push_back(p);
        return p;
    };

    const uint32_t H    = cfg_.hidden;                       // 4096
    const uint32_t N_q  = cfg_.n_q_heads  * cfg_.head_dim;   // 4096
    const uint32_t N_kv = cfg_.n_kv_heads * cfg_.head_dim;   // 1024
    const uint32_t F    = cfg_.ffn;                          // 12288

    ws_x_          = alloc_half(uint64_t(max_T) * H);
    ws_x_normed_   = alloc_half(uint64_t(max_T) * H);
    ws_q_          = alloc_half(uint64_t(max_T) * N_q);
    ws_k_          = alloc_half(uint64_t(max_T) * N_kv);
    ws_v_          = alloc_half(uint64_t(max_T) * N_kv);
    ws_attn_out_   = alloc_half(uint64_t(max_T) * N_q);
    ws_attn_block_ = alloc_half(uint64_t(max_T) * H);
    ws_gate_       = alloc_half(uint64_t(max_T) * F);   // ≈ 19 MiB total for
    ws_up_         = alloc_half(uint64_t(max_T) * F);   //   the 3 FFN buffers
    ws_h_          = alloc_half(uint64_t(max_T) * F);   //   at max_T = 256
    ws_positions_  = alloc_i32(max_T);
    {
        // Q8_1 activation scratch for the integer-dot decode GEMVs.  Sized
        // for the largest decode GEMV input: K = max(H, F) — F = 12288 is
        // the down-proj input, bigger than QwenModel's max(H, SI).  Both H
        // and F are multiples of 256 (validated at load) so /32 is exact.
        void* p = alloc_->malloc((uint64_t(std::max(H, F)) / 32) *
                                 sizeof(block_q8_1x));
        if (p) owned_.push_back(p);
        ws_q8_ = p;
    }

    if (!ws_x_ || !ws_x_normed_ || !ws_q_ || !ws_k_ || !ws_v_ ||
        !ws_attn_out_ || !ws_attn_block_ || !ws_gate_ || !ws_up_ || !ws_h_ ||
        !ws_positions_ || !ws_q8_) {
        return "workspace allocation failed";
    }
    ws_T_ = max_T;
    return {};
}

std::string DenseModel::ensure_attn_partials(uint32_t max_ctx) {
    if (!alloc_) return "alloc not initialized";
    if (max_ctx <= ws_attn_partials_ctx_) return {};
    // Conservative: size for the smallest Bc the FA-2 kernel could use
    // (same rule as QwenModel::ensure_attn_partials, qwen36.cpp:687-703).
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

// P2 Task 4: the Qwen3-dense forward pass.  Structurally mirrors
// QwenModel::forward's full-attention branch + a dense SwiGLU FFN
// (src/model/qwen36.cpp:736-1665 at commit 6df7cf5), with the dense deltas
// from the P2 plan's architecture table:
//   * every layer is full-attention — the KV cache slice index is L
//     (qwen36 uses L / full_attn_interval);
//   * NO attention output gate — attn_q projects straight to [T, n_q*hd]
//     (no 2x fold, no split_q_gate_per_head, no sigmoid_gate);
//   * FULL RoPE — rope_partial with n_rotary == rope_dim == head_dim;
//   * pre-FFN norm tensor is ffn_norm (same op + (1+w)-baked semantics as
//     qwen36's post_attention_norm);
//   * dense SwiGLU FFN (gate/up/swiglu/down) replaces the whole MoE block.
sycl::event DenseModel::forward(sycl::queue& q,
                                const int32_t* input_ids, uint32_t T,
                                uint32_t start_pos,
                                KvCache& kv, sycl::half* out_logits) {
    if (T == 0) return {};
    if (ws_T_ < T) {
        auto e = ensure_workspace(T);
        if (!e.empty()) {
            std::fprintf(stderr, "DenseModel::ensure_workspace: %s\n",
                         e.c_str());
            return {};
        }
    }

    const uint32_t H    = cfg_.hidden;                       // 4096
    const uint32_t HD   = cfg_.head_dim;                     // 128
    const uint32_t N_q  = cfg_.n_q_heads  * HD;              // 4096
    const uint32_t N_kv = cfg_.n_kv_heads * HD;              // 1024
    const uint32_t F    = cfg_.ffn;                          // 12288

    // Materialize positions on device (int32 [T] = start_pos..start_pos+T-1).
    // Same T==1 fill / T>1 memcpy split as qwen36.cpp:776-782 — the fill
    // avoids a host-blocking .wait() on every decode token.
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
    } else if (token_embd_dtype_ == DType::kF16) {
        embedding_lookup_f16(q, static_cast<const sycl::half*>(token_embd_),
                             input_ids, ws_x_, T, H);
    } else {
        embedding_lookup_q6k(q, input_ids, token_embd_, ws_x_, T, H);
    }
    // Granite: h = embedding_multiplier · Embed(tokens).  Value-gated (1.0 →
    // identity for every non-Granite model).  Applied before the L00 dump so
    // the oracle compare sees the scaled embedding.
    if (cfg_.embedding_multiplier != 1.0f)
        dense::scale_inplace(q, ws_x_, cfg_.embedding_multiplier, uint64_t(T) * H);

    // ---- Per-layer activation dump (opt-in via dump_prefix_) ----
    // Same file naming as QwenModel so tools/diff_layers.sh works unchanged
    // (Task 7 oracle): ${prefix}_L<NN>.bin fp32 + .meta carrying "T H".
    // NN = 00 (embed) .. n_layers (post-layer residuals) .. n_layers+1
    // (final-norm output) — MAX_SLOT=37 for qwen3-8b.
    auto dump_buf = [&](const sycl::half* buf, uint64_t n_elem, int idx) {
        if (dump_prefix_.empty()) return;
        std::vector<sycl::half> hh(n_elem);
        q.memcpy(hh.data(), buf, n_elem * sizeof(sycl::half)).wait();
        std::vector<float> ff(n_elem);
        for (size_t i = 0; i < n_elem; ++i) ff[i] = float(hh[i]);
        char path[1024];
        std::snprintf(path, sizeof(path), "%s_L%02d.bin", dump_prefix_.c_str(), idx);
        if (FILE* fp = std::fopen(path, "wb")) {
            std::fwrite(ff.data(), sizeof(float), n_elem, fp);
            std::fclose(fp);
        }
        std::snprintf(path, sizeof(path), "%s_L%02d.meta", dump_prefix_.c_str(), idx);
        if (FILE* fp = std::fopen(path, "w")) {
            std::fprintf(fp, "%u %u\n", T, H);
            std::fclose(fp);
        }
    };
    auto dump_residual = [&](int slot) { dump_buf(ws_x_, uint64_t(T) * H, slot); };
    dump_residual(0);  // L00 = post-embedding residual

    // ---- DEBUG: IE_DENSE_SYNC=1 → host-sync after each section, printing
    // the section tag first, so a device-side launch failure names its
    // kernel (the last "..." without "ok" is the culprit). Zero cost when
    // the env is unset.  Throws (wait_and_throw) so callers see the error.
    static const bool dbg_sync = std::getenv("IE_DENSE_SYNC") != nullptr;
    auto sync_dbg = [&](uint32_t L, const char* tag) {
        if (!dbg_sync) return;
        std::fprintf(stderr, "  [dense-sync L=%u] %-12s ...", L, tag);
        q.wait_and_throw();
        std::fprintf(stderr, " ok\n");
    };
    sync_dbg(0, "embedding");

    // ---- DEBUG: IE_DENSE_HASH=1 → after each section, host-sync and print
    // an FNV-1a hash of the section's output buffer(s).  Used by the
    // 2026-06-10 nondeterminism bisect (ie-det-probe --test model) to find
    // the first op whose output differs across in-process repetitions.
    // Zero cost when the env is unset.
    static const bool dbg_hash = std::getenv("IE_DENSE_HASH") != nullptr;
    auto hash_dbg = [&](uint32_t L, const char* tag, const void* buf,
                        uint64_t nbytes) {
        if (!dbg_hash || !buf) return;
        q.wait();
        std::vector<uint8_t> hb(nbytes);
        q.memcpy(hb.data(), buf, nbytes).wait();
        uint64_t hh = 1469598103934665603ull;
        for (uint64_t i = 0; i < nbytes; ++i) {
            hh ^= hb[i]; hh *= 1099511628211ull;
        }
        std::fprintf(stderr, "[dh] pos=%u L=%u %s %016llx\n",
                     start_pos, L, tag, (unsigned long long)hh);
    };
    hash_dbg(0, "embedding", ws_x_, uint64_t(T) * H * 2);

    // 2. n_layers × { attn → residual+norm → SwiGLU FFN → residual }
    for (uint32_t L = 0; L < cfg_.n_layers; ++L) {
        const auto& w = layers_[L];

        // 2a. attn_norm (+ fused Q8_1 emission at decode — the int-dot
        // GEMVs' activation quant rides this launch; qwen36.cpp:895-902).
        const bool x_q8_ready = (T == 1) && dense::q8_decode_enabled() && ws_q8_;
        if (x_q8_ready)
            rms_norm_f32w_q8(q, ws_x_, w.attn_norm, ws_x_normed_, ws_q8_,
                             T, H, cfg_.rms_eps);
        else
            rms_norm_f32w(q, ws_x_, w.attn_norm, ws_x_normed_, T, H, cfg_.rms_eps);
        sync_dbg(L, "attn_norm");
        hash_dbg(L, "attn_norm", ws_x_normed_, uint64_t(T) * H * 2);
        const void* xq8 = x_q8_ready ? static_cast<const void*>(ws_q8_) : nullptr;

        // 2b. Q/K/V projections — plain (NO gate fold, NO per-head split).
        // attn_v is F16 in the Ollama qwen3-8b GGUF: gemv_q_i8 falls back to
        // gemv_q → gemv_fp16 for it; gemv_q_T takes the gemm_fp16 branch.
        if (xq8) {
            dense::gemv_q_i8(q, ws_x_normed_, xq8, w.attn_q, ws_q_, H, N_q);
            dense::gemv_q_i8(q, ws_x_normed_, xq8, w.attn_k, ws_k_, H, N_kv);
            dense::gemv_q_i8(q, ws_x_normed_, xq8, w.attn_v, ws_v_, H, N_kv);
        } else {
            dense::gemv_q_T(q, ws_x_normed_, w.attn_q, ws_q_, H, N_q, T);
            dense::gemv_q_T(q, ws_x_normed_, w.attn_k, ws_k_, H, N_kv, T);
            dense::gemv_q_T(q, ws_x_normed_, w.attn_v, ws_v_, H, N_kv, T);
        }
        // Qwen2 attention bias (added after the projection). Absent for Qwen3.
        if (w.attn_q_bias) add_bias(q, ws_q_, w.attn_q_bias, T, N_q);
        if (w.attn_k_bias) add_bias(q, ws_k_, w.attn_k_bias, T, N_kv);
        if (w.attn_v_bias) add_bias(q, ws_v_, w.attn_v_bias, T, N_kv);
        sync_dbg(L, "qkv_proj");
        hash_dbg(L, "q_proj", ws_q_, uint64_t(T) * N_q * 2);
        hash_dbg(L, "k_proj", ws_k_, uint64_t(T) * N_kv * 2);
        hash_dbg(L, "v_proj", ws_v_, uint64_t(T) * N_kv * 2);

        // 2c. QK-norm: per-head RMSNorm (Qwen3). ABSENT for Qwen2 → skipped.
        if (w.attn_q_norm)
            rms_norm_f32w(q, ws_q_, w.attn_q_norm, ws_q_, T * cfg_.n_q_heads,  HD, cfg_.rms_eps);
        if (w.attn_k_norm)
            rms_norm_f32w(q, ws_k_, w.attn_k_norm, ws_k_, T * cfg_.n_kv_heads, HD, cfg_.rms_eps);
        sync_dbg(L, "qk_norm");
        hash_dbg(L, "qk_norm_q", ws_q_, uint64_t(T) * N_q * 2);
        hash_dbg(L, "qk_norm_k", ws_k_, uint64_t(T) * N_kv * 2);

        // 2d. FULL RoPE — rope_dim == head_dim (the d >= n_rotary
        // pass-through branch is never taken; NEOX half-split pairing
        // matches qwen3).
        // P3a: rope_freqs_ (llama3 scaling) → rope_partial_ff; else plain rope.
        if (rope_freqs_) {
            rope_partial_ff(q, ws_q_, ws_positions_, ws_q_,
                            T, cfg_.n_q_heads,  HD, cfg_.rope_dim, cfg_.rope_theta, rope_freqs_);
            rope_partial_ff(q, ws_k_, ws_positions_, ws_k_,
                            T, cfg_.n_kv_heads, HD, cfg_.rope_dim, cfg_.rope_theta, rope_freqs_);
        } else {
            rope_partial(q, ws_q_, ws_positions_, ws_q_,
                         T, cfg_.n_q_heads,  HD, cfg_.rope_dim, cfg_.rope_theta);
            rope_partial(q, ws_k_, ws_positions_, ws_k_,
                         T, cfg_.n_kv_heads, HD, cfg_.rope_dim, cfg_.rope_theta);
        }
        sync_dbg(L, "rope");
        hash_dbg(L, "rope_q", ws_q_, uint64_t(T) * N_q * 2);
        hash_dbg(L, "rope_k", ws_k_, uint64_t(T) * N_kv * 2);

        // Granite: the SDPA softmax scale is `attention_multiplier`, not the
        // default 1/sqrt(head_dim).  The shared attention kernels apply
        // 1/sqrt(HD) internally (crown-shared — signature untouchable), so we
        // pre-scale Q by attention_multiplier·sqrt(HD): the net Q·K scale then
        // equals attention_multiplier.  Value-gated (attn_mult==0 → default).
        if (cfg_.attention_multiplier != 0.0f)
            dense::scale_inplace(q, ws_q_,
                                 cfg_.attention_multiplier * std::sqrt(float(HD)),
                                 uint64_t(T) * N_q);

        // 2e. SDPA — KV cache slice index is L (every layer is full-attn).
        // Dispatch copied from qwen36.cpp:972-1026 with full_attn_idx → L.
        const uint64_t per_layer = uint64_t(cfg_.n_kv_heads) * kv.config().max_ctx * HD;
        sycl::half* k_cache = kv.k_ptr() + per_layer * L;
        sycl::half* v_cache = kv.v_ptr() + per_layer * L;
        if (T == 1 && ws_attn_partials_) {
            // INT8 path only if the INT8 cache is fully populated up to
            // start_pos — i.e. every prior position was quantized.
            if (kv.is_int8() && kv.k_int8_ptr() &&
                start_pos == kv.int8_length(L)) {
                const uint64_t int8_per_layer  = uint64_t(cfg_.n_kv_heads) * kv.config().max_ctx * HD;
                const uint64_t scale_per_layer = uint64_t(cfg_.n_kv_heads) * kv.config().max_ctx;
                int8_t*     k_i8 = kv.k_int8_ptr()   + int8_per_layer * L;
                int8_t*     v_i8 = kv.v_int8_ptr()   + int8_per_layer * L;
                sycl::half* k_sc = kv.k_scales_ptr() + scale_per_layer * L;
                sycl::half* v_sc = kv.v_scales_ptr() + scale_per_layer * L;
                full_attention_fa2_decode_int8(q, ws_q_, ws_k_, ws_v_,
                                               k_i8, v_i8, k_sc, v_sc,
                                               nullptr, nullptr,
                                               ws_attn_out_,
                                               ws_attn_partials_,
                                               start_pos,
                                               cfg_.n_q_heads, cfg_.n_kv_heads, HD,
                                               kv.config().max_ctx);
                // Inline quantize-on-write just wrote position start_pos.
                kv.set_int8_length(L, start_pos + 1);
            } else {
                full_attention_fa2_decode(q, ws_q_, ws_k_, ws_v_,
                                          k_cache, v_cache, ws_attn_out_,
                                          ws_attn_partials_,
                                          start_pos,
                                          cfg_.n_q_heads, cfg_.n_kv_heads, HD,
                                          kv.config().max_ctx);
            }
        } else {
            // T>1 prefill: naive path, fp16 cache. NOTE: prefill-v2 (query-row-block
            // FA-2) was A/B'd here on granite-8B 2026-06-23 and is FLAT — dense
            // prefill is MLP-BOUND (granite MLP ~1.0e14 MACs vs attn ~4.4e13 @16K),
            // unlike MoE (Coder) where the sparse MLP lets attention dominate and v2
            // gives 2.05x. So prefill-v2 is an MoE-only lever; dense stays naive.
            full_attention(q, ws_q_, ws_k_, ws_v_, k_cache, v_cache,
                           ws_attn_out_, T, start_pos,
                           cfg_.n_q_heads, cfg_.n_kv_heads, HD,
                           kv.config().max_ctx);
            // Post-quantize fp16 prefill rows into the INT8 shadow so the
            // next T=1 decode step can use the INT8 path.
            if (kv.is_int8()) kv.quantize_to_int8(q, L, start_pos, T);
        }
        kv.set_length(L, start_pos + T);
        sync_dbg(L, "attention");
        hash_dbg(L, "attn_out", ws_attn_out_, uint64_t(T) * N_q * 2);

        // 2f. NO sigmoid_gate.  O-proj straight: [T,N_q] → ws_attn_block_
        // [T,H] (q8 decode variant as qwen36.cpp:1037-1043 — separate input
        // vector → its own quant; ws_q8_ reuse is safe on the in-order queue).
        if (T == 1 && dense::q8_decode_enabled() && ws_q8_ &&
            w.attn_output.dt == DType::kQ4_K) {
            quantize_q8_1(q, ws_attn_out_, ws_q8_, N_q);
            dense::gemv_q_i8(q, ws_attn_out_, ws_q8_, w.attn_output,
                             ws_attn_block_, N_q, H);
        } else {
            dense::gemv_q_T(q, ws_attn_out_, w.attn_output, ws_attn_block_,
                            N_q, H, T);
        }
        sync_dbg(L, "o_proj");
        hash_dbg(L, "o_proj", ws_attn_block_, uint64_t(T) * H * 2);

        // Granite: residual = residual + residual_multiplier · attn_sublayer.
        // Value-gated (1.0 → identity).  Scales the attn sub-block output
        // (ws_attn_block_) before the fused residual-add below.
        if (cfg_.residual_multiplier != 1.0f)
            dense::scale_inplace(q, ws_attn_block_, cfg_.residual_multiplier,
                                 uint64_t(T) * H);

        // 2g. Fused residual_attn → ffn_norm in one launch
        // (qwen36.cpp:1171-1174; decl hoisted into dense_dispatch.hpp).
        residual_add_rms_norm_fused(q, ws_x_, ws_attn_block_, w.ffn_norm,
                                    ws_x_normed_, T, H, cfg_.rms_eps);
        sync_dbg(L, "res_rms");
        hash_dbg(L, "res_rms_x", ws_x_, uint64_t(T) * H * 2);
        hash_dbg(L, "res_rms_n", ws_x_normed_, uint64_t(T) * H * 2);

        // 2h. Dense SwiGLU FFN (replaces ALL of qwen36's MoE block).
        // Decode: quantize the FFN input once (it is NOT the attn_norm
        // output — step 2g produced a fresh vector), reuse across gate+up.
        const bool ffn_q8 = (T == 1) && dense::q8_decode_enabled() && ws_q8_;
        if (ffn_q8) {
            quantize_q8_1(q, ws_x_normed_, ws_q8_, H);
            dense::gemv_q_i8(q, ws_x_normed_, ws_q8_, w.ffn_gate, ws_gate_, H, F);
            dense::gemv_q_i8(q, ws_x_normed_, ws_q8_, w.ffn_up,   ws_up_,   H, F);
        } else {
            dense::gemv_q_T(q, ws_x_normed_, w.ffn_gate, ws_gate_, H, F, T);
            dense::gemv_q_T(q, ws_x_normed_, w.ffn_up,   ws_up_,   H, F, T);
        }
        sync_dbg(L, "ffn_gate_up");
        hash_dbg(L, "ffn_gate", ws_gate_, uint64_t(T) * F * 2);
        hash_dbg(L, "ffn_up", ws_up_, uint64_t(T) * F * 2);
        swiglu(q, ws_gate_, ws_up_, ws_h_, uint64_t(T) * F);
        sync_dbg(L, "swiglu");
        hash_dbg(L, "swiglu", ws_h_, uint64_t(T) * F * 2);
        // Down proj: ws_h_ [T,F] → ws_attn_block_ [T,H]  (K=F=12288 % 256 ✓).
        // ws_q8_ is sized max(H,F)/32 blocks, so re-quantizing it with the
        // F-sized vector is in-bounds; in-order queue serializes the reuse.
        // Q6_K at K=F=12288 exceeds the whole-column SLM-slab kernels' budget
        // (both gemv_q6_K_q8 and the fp16 staged variant), so the dense path
        // falls back to scalar gemm_q6_K at M=1 (the profiled #1 decode cliff).
        // P3b Task 1 added an int-dot port (gemv_q6_K_q8_ktiled) for this shape;
        // it is correctness-verified but measures SLOWER than the scalar cliff
        // (~75 vs 140 GB/s — dequant-ALU-bound at big-K/small-N, see
        // dense_dispatch.hpp), so it is DEFAULT-OFF. Opt IN with IE_Q6K_KTILED=1.
        // Q6_K-repack decode (IE_DENSE_Q6K_REPACK): T==1 routes the Q6_K ffn_down
        // through its SoA Q8_0 copy — int8 dp4a, no 6-bit unpack, 2.96× vs scalar.
        const bool down_q8_0_repack = ffn_q8 && T == 1 && w.ffn_down_q8_qs != nullptr;
        const bool down_q6k_ktiled = ffn_q8 && !down_q8_0_repack &&
            w.ffn_down.dt == DType::kQ6_K &&
            !dense::q6k_slm_gemv_q8_fits(F) && dense::q6k_ktiled_enabled();
        const bool down_q8 = ffn_q8 && !down_q8_0_repack &&
            (w.ffn_down.dt == DType::kQ4_K ||
             (w.ffn_down.dt == DType::kQ6_K && dense::q6k_slm_gemv_q8_fits(F)));
        if (down_q8_0_repack) {
            quantize_q8_1(q, ws_h_, ws_q8_, F);
            gemv_q8_0_soa_q8(q, ws_q8_, w.ffn_down_q8_qs, w.ffn_down_q8_d, ws_attn_block_, F, H);
        } else if (down_q6k_ktiled) {
            quantize_q8_1(q, ws_h_, ws_q8_, F);
            gemv_q6_K_q8_ktiled(q, ws_q8_, w.ffn_down.p, ws_attn_block_, F, H);
        } else if (down_q8) {
            quantize_q8_1(q, ws_h_, ws_q8_, F);
            if (w.ffn_down.dt == DType::kQ4_K)
                gemv_q4_K_q8(q, ws_q8_, w.ffn_down.p, ws_attn_block_, F, H);
            else
                gemv_q6_K_q8(q, ws_q8_, w.ffn_down.p, ws_attn_block_, F, H);
        } else {
            dense::gemv_q_T(q, ws_h_, w.ffn_down, ws_attn_block_, F, H, T);
        }

        // 2i. residual: ws_x_ += ws_attn_block_
        sync_dbg(L, "ffn_down");
        hash_dbg(L, "ffn_down", ws_attn_block_, uint64_t(T) * H * 2);
        // Granite: residual = residual + residual_multiplier · ffn_sublayer
        // (value-gated; same multiplier as the attn sub-block above).
        if (cfg_.residual_multiplier != 1.0f)
            dense::scale_inplace(q, ws_attn_block_, cfg_.residual_multiplier,
                                 uint64_t(T) * H);
        residual_add(q, ws_x_, ws_attn_block_, ws_x_, uint64_t(T) * H);
        sync_dbg(L, "residual");
        hash_dbg(L, "residual", ws_x_, uint64_t(T) * H * 2);
        dump_residual(int(L) + 1);  // L01..L<n> = post-layer-L residual
    }

    // 3. Final norm ((1+w)-baked weight; at decode, emit Q8_1 of the normed
    // vector in the same launch — feeds the int-dot lm_head;
    // qwen36.cpp:1604-1613).
    const bool head_q8 = (T == 1) && dense::q8_decode_enabled() && ws_q8_;
    if (head_q8)
        rms_norm_f32w_q8(q, ws_x_, output_norm_, ws_x_normed_, ws_q8_,
                         T, H, cfg_.rms_eps);
    else
        rms_norm_f32w(q, ws_x_, output_norm_, ws_x_normed_, T, H, cfg_.rms_eps);
    dump_buf(ws_x_normed_, uint64_t(T) * H, int(cfg_.n_layers) + 1);

    // 4. lm_head GEMV — only the LAST token's logits are needed.
    // Scalar gemv wins at M=1 over XMX (qwen36.cpp:1630-1648); the int-dot
    // staged kernels take over at decode for big vocabs.  An F16 lm_head
    // (not present in qwen3 GGUFs) would route through gemv_fp16 in gemv_q.
    sycl::half* last_x = ws_x_normed_ + uint64_t(T - 1) * H;
    DenseQuantPtr lm_head{output_, output_dtype_};
    lm_head.suh = output_suh_; lm_head.svh = output_svh_; lm_head.bits = output_bits_;
    sycl::event head_ev;
    if (head_q8 && cfg_.vocab >= 32768 &&
        lm_head.dt == DType::kQ6_K && dense::q6k_slm_gemv_q8_fits(H))
        head_ev = gemv_q6_K_q8(q, ws_q8_, lm_head.p, out_logits, H, cfg_.vocab);
    else if (head_q8 && cfg_.vocab >= 32768 && lm_head.dt == DType::kQ4_K)
        head_ev = gemv_q4_K_q8(q, ws_q8_, lm_head.p, out_logits, H, cfg_.vocab);
    else
        head_ev = dense::gemv_q(q, last_x, lm_head, out_logits, H, cfg_.vocab);

    // Granite: logits /= logits_scaling (value-gated; 1.0 → identity).  Only
    // the last token's logits [vocab] were written; scale them in place.
    if (cfg_.logits_scaling != 1.0f)
        return dense::scale_inplace(q, out_logits, 1.0f / cfg_.logits_scaling,
                                    cfg_.vocab);
    return head_ev;
}

}  // namespace ie
