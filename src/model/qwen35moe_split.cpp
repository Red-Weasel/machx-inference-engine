// src/model/qwen35moe_split.cpp — Qwen3.6-35B-A3B crown (`kQwen35Moe`) multi-GPU
// layer-split for all-Q8_0 GGUFs. See header. ADDITIVE: mirrors the 27B
// Qwen35SplitModel orchestration (qwen35_split.cpp) with the crown's per-layer math
// (qwen36.cpp) lifted into a device-by-device loop; the FFN is the crown MoE run as
// a Phase-1 per-active-expert Q8_0 int-dot loop. The single-GPU QwenModel is NEVER
// edited (PPL 6.4527 gate safe).
//
// PHASE-1 SCOPE: correctness-first, reuse-only. The MoE decode/prefill is a per-expert
// loop over the existing single-matrix gemv_q8_0_soa_q8 / dequant+gemm — no grouped
// Q8_0-expert kernel yet (Phase 2). Decode is host/comm-bound on a 2-card layer-split,
// so the launch-heavy loop is tolerable for validation.

#include "ie/qwen35moe_split.hpp"

#include "ie/dequant.hpp"
#include "ie/moe_q8.hpp"
#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"

#include "dense_dispatch.hpp"   // dense::upload<T>, dense::upload_quant_dense_auto, dense::gemv_q_T

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace ie {
namespace {

// IE_Q35MOE_TIMING diagnostic clock (per-section prefill/decode attribution).
inline double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// COPY of qwen35_split.cpp:upload_f32_proj_fp16 (ssm_alpha/ssm_beta → [K,Npad] fp16).
// copy-not-hoist discipline.
DenseQuantPtr upload_f32_proj_fp16(DeviceAllocator& alloc, const GgufTensorInfo* t,
                                   std::vector<void*>& owned, std::string& err,
                                   uint32_t Npad) {
    DenseQuantPtr out;
    if (!t) { err = "tensor not found"; return out; }
    if (t->n_dims != 2 || (t->dtype != DType::kF32 && t->dtype != DType::kQ8_0)) {
        err = "ssm proj: expected F32 or Q8_0 2-D"; return out;
    }
    const uint64_t K = t->shape[0];
    const uint64_t N = t->shape[1];
    if (Npad < N) Npad = uint32_t(N);
    std::vector<sycl::half> staging(K * Npad, sycl::half(0.0f));
    if (t->dtype == DType::kF32) {
        const float* src = reinterpret_cast<const float*>(t->data);
        for (uint64_t n = 0; n < N; ++n)
            for (uint64_t k = 0; k < K; ++k)
                staging[k * Npad + n] = sycl::half(src[n * K + k]);
    } else {  // Q8_0
        const uint64_t bpr = K / 32;
        const auto* blocks = reinterpret_cast<const block_q8_0*>(t->data);
        for (uint64_t n = 0; n < N; ++n)
            for (uint64_t k = 0; k < K; ++k) {
                const block_q8_0& b = blocks[n * bpr + (k >> 5)];
                staging[k * Npad + n] = sycl::half(fp16_to_fp32(b.d) * float(b.qs[k & 31]));
            }
    }
    void* d = alloc.malloc(K * Npad * sizeof(sycl::half));
    if (!d) { err = "malloc failed (ssm proj)"; return out; }
    alloc.queue().memcpy(d, staging.data(), K * Npad * sizeof(sycl::half)).wait();
    owned.push_back(d);
    out.p = d; out.dt = DType::kF16;
    return out;
}

// COPY of qwen35_split.cpp:extract_cols.
inline sycl::event extract_cols(sycl::queue& q, const sycl::half* src, sycl::half* dst,
                                uint32_t T, uint32_t nh, uint32_t src_stride) {
    return q.parallel_for(sycl::range<1>(uint64_t(T) * nh), [=](sycl::id<1> i) {
        const uint32_t t = uint32_t(i) / nh, h = uint32_t(i) % nh;
        dst[uint64_t(t) * nh + h] = src[uint64_t(t) * src_stride + h];
    });
}

// COPY of qwen35_split.cpp:dequant_q8_0_soa_to_Bt.
inline sycl::event dequant_q8_0_soa_to_Bt(sycl::queue& q, const int8_t* qs,
                                          const uint16_t* d, sycl::half* Bt,
                                          uint32_t K, uint32_t N) {
    const uint32_t bpc = K / 32;
    return q.parallel_for(sycl::range<2>(N, K), [=](sycl::id<2> id) {
        const uint32_t n = uint32_t(id[0]), k = uint32_t(id[1]);
        const float dv = float(sycl::bit_cast<sycl::half>(d[uint64_t(n) * bpc + (k >> 5)]));
        Bt[uint64_t(k) * N + n] = sycl::half(float(qs[uint64_t(n) * K + k]) * dv);
    });
}

// Repack a Q8_0 AoS [N,K] block matrix (n-major rows of K/32 block_q8_0) → SoA
// (qs[n*K+k] int8 + d[n*(K/32)] fp16). Operates on the e-th expert slice. Mirrors
// qwen35_split.cpp:build_split's Q8_0 branch, but writing into pre-allocated dst.
// blocks = pointer to this expert's first block_q8_0; N,K = matrix dims.
static void repack_q8_0_soa(const block_q8_0* blocks, int8_t* qs, uint16_t* dd,
                            uint32_t N, uint32_t K) {
    const uint32_t bpc = K / 32;
    for (uint64_t n = 0; n < N; ++n)
        for (uint32_t b = 0; b < bpc; ++b) {
            const block_q8_0& blk = blocks[n * bpc + b];
            dd[n * bpc + b] = *reinterpret_cast<const uint16_t*>(&blk.d);
            for (int i = 0; i < 32; ++i)
                qs[n * K + uint64_t(b) * 32 + i] = blk.qs[i];
        }
}

}  // namespace

Qwen35MoeSplitModel::~Qwen35MoeSplitModel() { free_all(); }

void Qwen35MoeSplitModel::free_all() {
    if (!fleet_) return;
    for (uint32_t d = 0; d < ws_.size(); ++d) free_ws(d);
    ws_.clear();
    for (uint32_t d = 0; d < owned_.size(); ++d)
        for (void* p : owned_[d]) if (p) fleet_->dev(d).free(p);
    owned_.clear();
    for (uint32_t d = 0; d < prefill_bt_.size(); ++d)
        if (prefill_bt_[d]) fleet_->dev(d).free(prefill_bt_[d]);
    prefill_bt_.clear(); prefill_bt_cap_.clear(); act_q8_.clear();
    for (auto& s : dn_) s.free_storage();
    for (auto& c : kv_) c.free_storage();
}

std::string Qwen35MoeSplitModel::load(DeviceFleet& fleet, const LayerPlan& plan,
                                      const GgufReader& g, const QwenConfig& cfg,
                                      uint32_t max_ctx) {
    fleet_ = &fleet; plan_ = plan; cfg_ = cfg;
    n_dev_ = plan.n_dev();
    if (n_dev_ == 0 || plan.dev_of_layer.empty()) return "qwen35moe_split: empty plan";

    // Config = the crown's QwenConfig defaults (same as the single-GPU QwenModel,
    // which this fine-tune already loads config-wise). [VERIFY if a fine-tune ever
    // changes vocab/expert_count — then read the gguf arch keys here.]
    dense::prefer_onednn() = false;

    owned_.assign(n_dev_, {});
    dev_bytes_.assign(n_dev_, 0);

    char buf[96];
    auto Ttop = [&](const char* n) { return g.find_tensor(n); };
    auto Tl   = [&](uint32_t L, const char* n) {
        std::snprintf(buf, sizeof(buf), "blk.%u.%s", L, n); return g.find_tensor(buf);
    };
    std::string err;

    // build a SplitW (2-D Q8_0 → SoA; else fp fallback). COPY of qwen35_split.cpp.
    auto build_split = [](DeviceAllocator& a, const GgufTensorInfo* t,
                          std::vector<void*>& own, std::string& e, uint64_t& bytes) -> SplitW {
        SplitW w{};
        if (!t) { e = "tensor not found"; return w; }
        if (t->n_dims != 2) { e = "split weight: expected 2-D"; return w; }
        w.K = uint32_t(t->shape[0]); w.N = uint32_t(t->shape[1]);
        if (t->dtype == DType::kQ8_0) {
            const uint32_t K = w.K, N = w.N, bpc = K / 32;
            if (K % 32 != 0) { e = "Q8_0 SoA: K % 32 != 0"; return w; }
            std::vector<int8_t>   qs(uint64_t(N) * K);
            std::vector<uint16_t> dd(uint64_t(N) * bpc);
            repack_q8_0_soa(reinterpret_cast<const block_q8_0*>(t->data), qs.data(), dd.data(), N, K);
            auto* dqs = static_cast<int8_t*>(a.malloc(qs.size()));
            auto* ddd = static_cast<uint16_t*>(a.malloc(dd.size() * sizeof(uint16_t)));
            if (!dqs || !ddd) { e = "Q8_0 SoA alloc"; return w; }
            own.push_back(dqs); own.push_back(ddd);
            a.queue().memcpy(dqs, qs.data(), qs.size()).wait();
            a.queue().memcpy(ddd, dd.data(), dd.size() * sizeof(uint16_t)).wait();
            w.q8_qs = dqs; w.q8_d = ddd;
            bytes += qs.size() + dd.size() * sizeof(uint16_t);
        } else {
            w.fp = dense::upload_quant_dense_auto(a, t, own, e);
            if (e.empty()) bytes += t->nbytes;
        }
        return w;
    };

    // build an ExpertsW from a 3-D Q8_0 expert tensor. ggml MoE expert tensors are
    // [in, out, E] (shape[0]=K=in, shape[1]=N=out, shape[2]=E); per-expert data is
    // expert-major contiguous (expert e at block-offset e*N*(K/32)). Repack each
    // expert's [N,K] AoS slice → one contiguous SoA buffer.  [VERIFY shape order
    // against gguf vs qwen36.cpp expert loading — flagged.]
    auto build_experts = [](DeviceAllocator& a, const GgufTensorInfo* t,
                            std::vector<void*>& own, std::string& e, uint64_t& bytes) -> ExpertsW {
        ExpertsW w{};
        if (!t) { e = "experts tensor not found"; return w; }
        if (t->n_dims != 3) { e = "experts: expected 3-D"; return w; }
        if (t->dtype != DType::kQ8_0) { e = "experts: expected Q8_0"; return w; }
        w.K = uint32_t(t->shape[0]); w.N = uint32_t(t->shape[1]); w.E = uint32_t(t->shape[2]);
        const uint32_t K = w.K, N = w.N, E = w.E, bpc = K / 32;
        if (K % 32 != 0) { e = "experts Q8_0: K % 32 != 0"; return w; }
        w.qs_stride = uint64_t(N) * K;
        w.d_stride  = uint64_t(N) * bpc;
        std::vector<int8_t>   qs(uint64_t(E) * w.qs_stride);
        std::vector<uint16_t> dd(uint64_t(E) * w.d_stride);
        const uint64_t blocks_per_expert = uint64_t(N) * bpc;
        const auto* base = reinterpret_cast<const block_q8_0*>(t->data);
        for (uint32_t ex = 0; ex < E; ++ex)
            repack_q8_0_soa(base + ex * blocks_per_expert,
                            qs.data() + ex * w.qs_stride, dd.data() + ex * w.d_stride, N, K);
        auto* dqs = static_cast<int8_t*>(a.malloc(qs.size()));
        auto* ddd = static_cast<uint16_t*>(a.malloc(dd.size() * sizeof(uint16_t)));
        if (!dqs || !ddd) { e = "experts SoA alloc"; return w; }
        own.push_back(dqs); own.push_back(ddd);
        a.queue().memcpy(dqs, qs.data(), qs.size()).wait();
        a.queue().memcpy(ddd, dd.data(), dd.size() * sizeof(uint16_t)).wait();
        w.qs = dqs; w.d = ddd;
        bytes += qs.size() + dd.size() * sizeof(uint16_t);
        return w;
    };

    // token_embd → embed_dev.
    {
        DeviceAllocator& ea = fleet.dev(plan.embed_dev);
        const auto* ti = Ttop("token_embd.weight");
        if (!ti) return "token_embd: not found";
        if (ti->dtype != DType::kQ4_K && ti->dtype != DType::kQ6_K && ti->dtype != DType::kQ8_0)
            return std::string("token_embd: unsupported dtype ") +
                   std::string(type_name(ti->dtype)) + " (need Q4_K/Q6_K/Q8_0)";
        token_embd_dtype_ = ti->dtype;
        token_embd_ = dense::upload<void>(ea, ti, owned_[plan.embed_dev], err, ti->dtype);
        if (!err.empty()) return "token_embd: " + err;
        dev_bytes_[plan.embed_dev] += ti->nbytes;
    }
    // output_norm + lm_head → head_dev.
    {
        DeviceAllocator& ha = fleet.dev(plan.head_dev);
        output_norm_ = dense::upload<float>(ha, Ttop("output_norm.weight"),
                                            owned_[plan.head_dev], err, DType::kF32);
        if (!err.empty()) return "output_norm: " + err;
        const auto* ti = Ttop("output.weight");
        if (!ti) ti = Ttop("token_embd.weight");
        output_ = build_split(ha, ti, owned_[plan.head_dev], err, dev_bytes_[plan.head_dev]);
        if (!err.empty()) return "output: " + err;
    }

    const uint32_t n_layers = cfg_.n_layers;   // 40
    layers_.assign(n_layers, {});
    dn_local_.assign(n_layers, 0);
    kv_local_.assign(n_layers, 0);

    for (uint32_t L = 0; L < n_layers; ++L) {
        const uint32_t dev = plan.dev_of_layer[L];
        DeviceAllocator& a = fleet.dev(dev);
        auto& own = owned_[dev];
        LayerW& w = layers_[L];
        // crown: full-attn every full_attn_interval-th layer (i % 4 == 3); else DeltaNet.
        w.is_linear = ((L + 1) % cfg_.full_attn_interval) != 0;

        auto LW = [&](const char* n) -> SplitW {
            return build_split(a, Tl(L, n), own, err, dev_bytes_[dev]);
        };
        auto EW = [&](const char* n) -> ExpertsW {
            return build_experts(a, Tl(L, n), own, err, dev_bytes_[dev]);
        };
        auto F32 = [&](const char* n, float*& dst) -> std::string {
            const auto* ti = Tl(L, n);
            dst = dense::upload<float>(a, ti, own, err, DType::kF32);
            if (!err.empty()) return std::string(n) + ": " + err;
            if (ti) dev_bytes_[dev] += ti->nbytes;
            return {};
        };
        auto le = [&](const char* what) {
            return "layer " + std::to_string(L) + " " + what + ": " + err;
        };

        if (auto m = F32("attn_norm.weight", w.attn_norm); !m.empty())
            return "layer " + std::to_string(L) + " " + m;
        if (auto m = F32("post_attention_norm.weight", w.post_attn_norm); !m.empty())
            return "layer " + std::to_string(L) + " " + m;

        if (w.is_linear) {
            w.attn_qkv  = LW("attn_qkv.weight");  if (!err.empty()) return le("attn_qkv");
            w.attn_gate = LW("attn_gate.weight"); if (!err.empty()) return le("attn_gate");
            if (auto m = F32("ssm_a", w.ssm_a); !m.empty()) return "layer " + std::to_string(L) + " " + m;
            const uint32_t svh_pad = ((cfg_.ssm_n_v_heads + 63u) / 64u) * 64u;
            w.ssm_alpha = upload_f32_proj_fp16(a, Tl(L, "ssm_alpha.weight"), own, err, svh_pad);
            if (!err.empty()) return le("ssm_alpha");
            w.ssm_beta  = upload_f32_proj_fp16(a, Tl(L, "ssm_beta.weight"), own, err, svh_pad);
            if (!err.empty()) return le("ssm_beta");
            if (auto m = F32("ssm_dt.bias", w.ssm_dt_bias); !m.empty()) return "layer " + std::to_string(L) + " " + m;
            {
                const auto* ti = Tl(L, "ssm_conv1d.weight");
                if (auto m = F32("ssm_conv1d.weight", w.ssm_conv1d); !m.empty()) return "layer " + std::to_string(L) + " " + m;
                const uint64_t n = ti->nbytes / sizeof(float);
                w.ssm_conv1d_fp16 = static_cast<sycl::half*>(a.malloc(n * sizeof(sycl::half)));
                if (!w.ssm_conv1d_fp16) return le("ssm_conv1d fp16 malloc");
                own.push_back(w.ssm_conv1d_fp16); dev_bytes_[dev] += n * sizeof(sycl::half);
                cast_fp32_to_fp16(a.queue(), w.ssm_conv1d, w.ssm_conv1d_fp16, n).wait();
            }
            {
                const auto* ti = Tl(L, "ssm_norm.weight");
                if (auto m = F32("ssm_norm.weight", w.ssm_norm); !m.empty()) return "layer " + std::to_string(L) + " " + m;
                const uint64_t n = ti->nbytes / sizeof(float);
                w.ssm_norm_fp16 = static_cast<sycl::half*>(a.malloc(n * sizeof(sycl::half)));
                if (!w.ssm_norm_fp16) return le("ssm_norm fp16 malloc");
                own.push_back(w.ssm_norm_fp16); dev_bytes_[dev] += n * sizeof(sycl::half);
                cast_fp32_to_fp16(a.queue(), w.ssm_norm, w.ssm_norm_fp16, n).wait();
            }
            w.ssm_out = LW("ssm_out.weight");   if (!err.empty()) return le("ssm_out");
        } else {
            w.attn_q      = LW("attn_q.weight");      if (!err.empty()) return le("attn_q");
            w.attn_k      = LW("attn_k.weight");      if (!err.empty()) return le("attn_k");
            w.attn_v      = LW("attn_v.weight");      if (!err.empty()) return le("attn_v");
            w.attn_output = LW("attn_output.weight"); if (!err.empty()) return le("attn_output");
            if (auto m = F32("attn_q_norm.weight", w.attn_q_norm); !m.empty()) return "layer " + std::to_string(L) + " " + m;
            if (auto m = F32("attn_k_norm.weight", w.attn_k_norm); !m.empty()) return "layer " + std::to_string(L) + " " + m;
        }

        // MoE FFN — router + experts + shared expert (every layer).
        if (auto m = F32("ffn_gate_inp.weight", w.ffn_gate_inp); !m.empty()) return "layer " + std::to_string(L) + " " + m;
        w.exp_gate = EW("ffn_gate_exps.weight"); if (!err.empty()) return le("ffn_gate_exps");
        w.exp_up   = EW("ffn_up_exps.weight");   if (!err.empty()) return le("ffn_up_exps");
        w.exp_down = EW("ffn_down_exps.weight"); if (!err.empty()) return le("ffn_down_exps");
        // shared expert (optional — present on the crown).
        if (Tl(L, "ffn_gate_inp_shexp.weight")) {
            if (auto m = F32("ffn_gate_inp_shexp.weight", w.ffn_gate_inp_shexp); !m.empty()) return "layer " + std::to_string(L) + " " + m;
            w.sh_gate = LW("ffn_gate_shexp.weight"); if (!err.empty()) return le("ffn_gate_shexp");
            w.sh_up   = LW("ffn_up_shexp.weight");   if (!err.empty()) return le("ffn_up_shexp");
            w.sh_down = LW("ffn_down_shexp.weight"); if (!err.empty()) return le("ffn_down_shexp");
        }
    }

    // Per-card hybrid caches.
    std::vector<uint32_t> n_lin(n_dev_, 0), n_full(n_dev_, 0);
    for (uint32_t L = 0; L < n_layers; ++L) {
        const uint32_t dev = plan.dev_of_layer[L];
        if (layers_[L].is_linear) dn_local_[L] = n_lin[dev]++;
        else                      kv_local_[L] = n_full[dev]++;
    }
    dn_.resize(n_dev_); kv_.resize(n_dev_);
    for (uint32_t dev = 0; dev < n_dev_; ++dev) {
        if (n_lin[dev]) {
            DeltaNetStateConfig dc{};
            dc.n_layers_linear = n_lin[dev];
            dc.n_v_heads       = cfg_.ssm_n_v_heads;       // 32
            dc.v_head_dim      = cfg_.ssm_head_dim;        // 128
            dc.k_head_dim      = cfg_.ssm_head_dim;        // 128
            dc.conv_channels   = cfg_.ssm_inner + 2u * cfg_.ssm_n_k_heads * cfg_.ssm_head_dim;  // 8192
            dc.conv_kernel     = cfg_.ssm_conv_kernel;     // 4
            if (auto m = dn_[dev].init(fleet.dev(dev), dc); !m.empty())
                return "dn cache dev " + std::to_string(dev) + ": " + m;
        }
        if (n_full[dev]) {
            KvCacheConfig kc{};
            kc.n_layers_full = n_full[dev];
            kc.n_kv_heads    = cfg_.n_kv_heads;            // 2
            kc.max_ctx       = max_ctx;
            kc.head_dim      = cfg_.head_dim;              // 256
            if (auto m = kv_[dev].init(fleet.dev(dev), kc); !m.empty())
                return "kv cache dev " + std::to_string(dev) + ": " + m;
        }
    }

    // Per-card int-dot activation scratch + prefill dequant scratch.
    const uint32_t Kmax = std::max(std::max(cfg_.hidden, cfg_.expert_ffn),
                                   std::max(cfg_.ssm_inner, cfg_.n_q_heads * cfg_.head_dim));
    act_q8_.assign(n_dev_, nullptr);
    prefill_bt_.assign(n_dev_, nullptr);
    prefill_bt_cap_.assign(n_dev_, 0);
    for (uint32_t dev = 0; dev < n_dev_; ++dev) {
        void* p = fleet.dev(dev).malloc((uint64_t(Kmax) / 32) * sizeof(block_q8_1x));
        if (!p) return "act_q8 alloc dev " + std::to_string(dev);
        owned_[dev].push_back(p);
        act_q8_[dev] = p;
    }
    for (uint32_t dev = 0; dev < n_dev_; ++dev)
        std::fprintf(stderr, "[qwen35moe_split] card %u weights: %.2f GB\n",
                     dev, double(dev_bytes_[dev]) / 1e9);
    return {};
}

void Qwen35MoeSplitModel::free_ws(uint32_t dev) {
    if (!fleet_ || dev >= ws_.size()) return;
    Workspace& w = ws_[dev];
    auto& alloc = fleet_->dev(dev);
    for (void* p : {static_cast<void*>(w.x), static_cast<void*>(w.x_normed),
                    static_cast<void*>(w.attn_block), static_cast<void*>(w.positions),
                    static_cast<void*>(w.qg), static_cast<void*>(w.q), static_cast<void*>(w.gate),
                    static_cast<void*>(w.k), static_cast<void*>(w.v), static_cast<void*>(w.attn_out),
                    static_cast<void*>(w.attn_partials),
                    static_cast<void*>(w.dn_qkv), static_cast<void*>(w.dn_conv), static_cast<void*>(w.dn_z),
                    static_cast<void*>(w.dn_qpre), static_cast<void*>(w.dn_kpre), static_cast<void*>(w.dn_vpre),
                    static_cast<void*>(w.dn_g), static_cast<void*>(w.dn_beta), static_cast<void*>(w.dn_out),
                    static_cast<void*>(w.dn_qrep), static_cast<void*>(w.dn_krep),
                    static_cast<void*>(w.dn_alpha_h), static_cast<void*>(w.dn_beta_h),
                    static_cast<void*>(w.dn_alpha64), static_cast<void*>(w.dn_beta64),
                    static_cast<void*>(w.topk_idx), static_cast<void*>(w.topk_w),
                    static_cast<void*>(w.gate_o), static_cast<void*>(w.up_o), static_cast<void*>(w.ffn_h),
                    static_cast<void*>(w.moe_y), static_cast<void*>(w.eo),
                    static_cast<void*>(w.moe_hp), static_cast<void*>(w.moe_yp),
                    w.x_q8, w.h_q8,
                    static_cast<void*>(w.moe_xp), w.xp_q8,
                    static_cast<void*>(w.moe_eoff), static_cast<void*>(w.moe_sidx),
                    static_cast<void*>(w.moe_sw), static_cast<void*>(w.moe_tk2pk)})
        if (p) alloc.free(p);
    w = Workspace{};
}

std::string Qwen35MoeSplitModel::ensure_ws(uint32_t dev, uint32_t max_T) {
    Workspace& w = ws_[dev];
    if (max_T <= w.T) return {};
    if (w.T != 0) free_ws(dev);
    auto& alloc = fleet_->dev(dev);
    auto ah = [&](size_t n) { return static_cast<sycl::half*>(alloc.malloc(n * sizeof(sycl::half))); };
    auto af = [&](size_t n) { return static_cast<float*>(alloc.malloc(n * sizeof(float))); };

    const uint64_t T   = max_T;
    const uint32_t H   = cfg_.hidden;                       // 2048
    const uint32_t N_q = cfg_.n_q_heads * cfg_.head_dim;    // 4096
    const uint32_t N_qg = N_q * 2u;                         // 8192
    const uint32_t N_kv = cfg_.n_kv_heads * cfg_.head_dim;  // 512
    const uint32_t EFF = cfg_.expert_ffn;                  // 512
    const uint32_t SI  = cfg_.ssm_inner;                    // 4096
    const uint32_t cvc = SI + 2u * cfg_.ssm_n_k_heads * cfg_.ssm_head_dim;  // 8192
    const uint32_t Vd  = cfg_.ssm_n_v_heads * cfg_.ssm_head_dim;            // 4096
    const uint32_t Nv  = cfg_.ssm_n_v_heads;               // 32
    const uint32_t Nvp = ((Nv + 63u) / 64u) * 64u;         // 64
    const uint32_t Kt  = cfg_.experts_topk;                // 8
    const uint64_t TK  = T * Kt;

    w.x = ah(T*H); w.x_normed = ah(T*H); w.attn_block = ah(T*H);
    w.positions = static_cast<int32_t*>(alloc.malloc(T * sizeof(int32_t)));
    w.qg = ah(T*N_qg); w.q = ah(T*N_q); w.gate = ah(T*N_q);
    w.k = ah(T*N_kv); w.v = ah(T*N_kv); w.attn_out = ah(T*N_q);
    w.dn_qkv = ah(T*cvc); w.dn_conv = ah(T*cvc); w.dn_z = ah(T*SI);
    w.dn_qpre = af(T*Vd); w.dn_kpre = af(T*Vd); w.dn_vpre = af(T*Vd);
    w.dn_g = af(T*Nv); w.dn_beta = af(T*Nv); w.dn_out = af(T*Vd);
    w.dn_qrep = af(T*Vd); w.dn_krep = af(T*Vd);
    w.dn_alpha_h = ah(T*Nv); w.dn_beta_h = ah(T*Nv);
    w.dn_alpha64 = ah(T*Nvp); w.dn_beta64 = ah(T*Nvp);
    w.topk_idx = static_cast<int32_t*>(alloc.malloc(TK * sizeof(int32_t)));
    w.topk_w = ah(TK); w.gate_o = ah(T*EFF); w.up_o = ah(T*EFF); w.ffn_h = ah(T*EFF);
    w.moe_y = ah(T*H); w.eo = ah(T*H);
    w.moe_hp = ah(TK*EFF); w.moe_yp = ah(TK*H);
    w.x_q8 = alloc.malloc(uint64_t(T) * (H / 32) * sizeof(block_q8_1x));
    w.h_q8 = alloc.malloc(TK * (EFF / 32) * sizeof(block_q8_1x));
    // Expert-batched prefill scratch (sized to maxTK; tiny at decode T=1).
    const uint32_t E = cfg_.n_experts;                     // 256
    w.moe_xp = ah(TK*H);
    w.xp_q8 = alloc.malloc(TK * (H / 32) * sizeof(block_q8_1x));
    w.moe_eoff = static_cast<uint32_t*>(alloc.malloc((uint64_t(E) + 1) * sizeof(uint32_t)));
    w.moe_sidx = static_cast<int32_t*>(alloc.malloc(TK * sizeof(int32_t)));
    w.moe_sw = ah(TK);
    w.moe_tk2pk = static_cast<uint32_t*>(alloc.malloc(TK * sizeof(uint32_t)));

    if (!w.x || !w.x_normed || !w.attn_block || !w.positions || !w.qg || !w.q ||
        !w.gate || !w.k || !w.v || !w.attn_out || !w.dn_qkv || !w.dn_conv || !w.dn_z ||
        !w.dn_qpre || !w.dn_kpre || !w.dn_vpre || !w.dn_g || !w.dn_beta || !w.dn_out ||
        !w.dn_qrep || !w.dn_krep || !w.dn_alpha_h || !w.dn_beta_h || !w.dn_alpha64 ||
        !w.dn_beta64 || !w.topk_idx || !w.topk_w || !w.gate_o || !w.up_o || !w.ffn_h ||
        !w.moe_y || !w.eo || !w.moe_hp || !w.moe_yp || !w.x_q8 || !w.h_q8 ||
        !w.moe_xp || !w.xp_q8 || !w.moe_eoff || !w.moe_sidx || !w.moe_sw || !w.moe_tk2pk)
        return "qwen35moe_split workspace alloc failed on dev " + std::to_string(dev);

    if (kv_[dev].ready()) {
        const uint32_t max_ctx = kv_[dev].config().max_ctx;
        constexpr uint32_t Bc_floor = 64;
        const uint32_t n_chunks_max = (max_ctx + Bc_floor - 1) / Bc_floor;
        const uint64_t n_floats = uint64_t(n_chunks_max) * cfg_.n_q_heads * (cfg_.head_dim + 2);
        w.attn_partials = static_cast<float*>(alloc.malloc(n_floats * sizeof(float)));
        if (!w.attn_partials) return "qwen35moe_split attn_partials alloc failed on dev " + std::to_string(dev);
        w.partials_ctx = max_ctx;
    }
    w.T = max_T;
    return {};
}

sycl::event Qwen35MoeSplitModel::sgemv(uint32_t dev, const sycl::half* A, const SplitW& w,
                                       sycl::half* out, uint32_t K, uint32_t N, uint32_t T) {
    auto& alloc = fleet_->dev(dev);
    auto& q = alloc.queue();
    if (!w.q8_qs) return dense::gemv_q_T(q, A, w.fp, out, K, N, T);
    if (T == 1) {
        quantize_q8_1(q, A, act_q8_[dev], K);
        return gemv_q8_0_soa_q8(q, act_q8_[dev], w.q8_qs, w.q8_d, out, K, N);
    }
    const uint64_t need = uint64_t(K) * N;
    if (need > prefill_bt_cap_[dev]) {
        if (prefill_bt_[dev]) alloc.free(prefill_bt_[dev]);
        prefill_bt_[dev] = static_cast<sycl::half*>(alloc.malloc(need * sizeof(sycl::half)));
        prefill_bt_cap_[dev] = prefill_bt_[dev] ? need : 0;
    }
    if (!prefill_bt_[dev]) { std::fprintf(stderr, "qwen35moe_split: prefill_bt alloc failed\n"); return {}; }
    dequant_q8_0_soa_to_Bt(q, w.q8_qs, w.q8_d, prefill_bt_[dev], K, N);
    return dense::gemv_q_T(q, A, DenseQuantPtr{prefill_bt_[dev], DType::kF16}, out, K, N, T);
}

// Per-expert Q8_0 GEMV (one expert slice e of an ExpertsW). T tokens of A[T,K] →
// out[T,N]. Decode (T==1) int-dot; prefill dequant-to-fp16 + gemm.
sycl::event Qwen35MoeSplitModel::egemv(uint32_t dev, const sycl::half* A, const ExpertsW& w,
                                       uint32_t e, sycl::half* out, uint32_t T) {
    auto& alloc = fleet_->dev(dev);
    auto& q = alloc.queue();
    const int8_t*   qs = w.qs + uint64_t(e) * w.qs_stride;
    const uint16_t* dd = w.d  + uint64_t(e) * w.d_stride;
    if (T == 1) {
        quantize_q8_1(q, A, act_q8_[dev], w.K);
        return gemv_q8_0_soa_q8(q, act_q8_[dev], qs, dd, out, w.K, w.N);
    }
    const uint64_t need = uint64_t(w.K) * w.N;
    if (need > prefill_bt_cap_[dev]) {
        if (prefill_bt_[dev]) alloc.free(prefill_bt_[dev]);
        prefill_bt_[dev] = static_cast<sycl::half*>(alloc.malloc(need * sizeof(sycl::half)));
        prefill_bt_cap_[dev] = prefill_bt_[dev] ? need : 0;
    }
    if (!prefill_bt_[dev]) return {};
    dequant_q8_0_soa_to_Bt(q, qs, dd, prefill_bt_[dev], w.K, w.N);
    return dense::gemv_q_T(q, A, DenseQuantPtr{prefill_bt_[dev], DType::kF16}, out, w.K, w.N, T);
}

std::string Qwen35MoeSplitModel::forward(const int32_t* input_ids, uint32_t T,
                                         uint32_t start_pos, bool reset_kv,
                                         sycl::half* out_logits_host) {
    if (T == 0) return "T == 0";
    const uint32_t H    = cfg_.hidden;                  // 2048
    const uint32_t HD   = cfg_.head_dim;                // 256
    const uint32_t N_q  = cfg_.n_q_heads  * HD;         // 4096
    const uint32_t N_qg = N_q * 2u;                     // 8192
    const uint32_t N_kv = cfg_.n_kv_heads * HD;         // 512
    const uint32_t V    = cfg_.vocab;
    const uint32_t EFF  = cfg_.expert_ffn;             // 512
    const uint32_t E    = cfg_.n_experts;               // 256
    const uint32_t Kt   = cfg_.experts_topk;            // 8
    const uint32_t rope_n = cfg_.rope_dim;              // 64
    const float    eps  = cfg_.rms_eps;
    const uint32_t n_layers = cfg_.n_layers;
    const uint32_t n_kv = cfg_.n_kv_heads;

    if (ws_.size() != n_dev_) ws_.assign(n_dev_, {});
    for (uint32_t dev = 0; dev < n_dev_; ++dev)
        if (auto m = ensure_ws(dev, T); !m.empty()) return m;

    std::vector<int32_t> pos(T);
    for (uint32_t t = 0; t < T; ++t) pos[t] = int32_t(start_pos + t);
    for (uint32_t dev = 0; dev < n_dev_; ++dev) {
        if (ws_[dev].positions)
            fleet_->dev(dev).queue().memcpy(ws_[dev].positions, pos.data(), T * sizeof(int32_t)).wait();
        if (reset_kv) {
            if (dn_[dev].ready()) dn_[dev].reset(fleet_->dev(dev).queue());
            if (kv_[dev].ready()) kv_[dev].reset();
        }
    }

    // embedding → ws_[embed_dev].x
    {
        auto& alloc = fleet_->dev(plan_.embed_dev);
        auto& q = alloc.queue();
        int32_t* d_ids = static_cast<int32_t*>(alloc.malloc(T * sizeof(int32_t)));
        if (!d_ids) return "d_ids alloc failed";
        q.memcpy(d_ids, input_ids, T * sizeof(int32_t)).wait();
        if (token_embd_dtype_ == DType::kQ4_K)
            embedding_lookup_q4k(q, d_ids, token_embd_, ws_[plan_.embed_dev].x, T, H);
        else if (token_embd_dtype_ == DType::kQ8_0)
            embedding_lookup_q8_0(q, d_ids, token_embd_, ws_[plan_.embed_dev].x, T, H);
        else
            embedding_lookup_q6k(q, d_ids, token_embd_, ws_[plan_.embed_dev].x, T, H);
        q.wait();
        alloc.free(d_ids);
    }

    // IE_Q35MOE_TIMING: per-section wall-clock (adds q.wait barriers → relative
    // attribution only, not absolute tok/s). Answers "is prefill attn- or MoE-bound?".
    const bool dbg_timing = std::getenv("IE_Q35MOE_TIMING") != nullptr;
    double t_attn = 0.0, t_moe = 0.0;
    for (uint32_t dev = 0; dev < n_dev_; ++dev) {
        Workspace& w = ws_[dev];
        auto& q = fleet_->dev(dev).queue();
        if (dev > 0) {
            // Drain the SOURCE card's queue before the cross-card x copy. The GPU-resident
            // MoE removed the per-layer host-pull that used to drain it implicitly, so without
            // this the copy reads card (dev-1)'s x while its kernels are still writing → a
            // cross-card race that faults the device (UR_RESULT_ERROR_DEVICE_LOST). One barrier
            // per card boundary (n_dev-1 total) — compute stays fully on the GPU.
            fleet_->dev(dev - 1).queue().wait();
            fleet_->copy_across(dev - 1, w.x, dev, ws_[dev - 1].x, uint64_t(T) * H * sizeof(sycl::half));
        }
        const uint64_t per_layer_kv =
            uint64_t(n_kv) * (kv_[dev].ready() ? kv_[dev].config().max_ctx : 0u) * HD;

        for (uint32_t L = 0; L < n_layers; ++L) {
            if (plan_.dev_of_layer[L] != dev) continue;
            const LayerW& w_l = layers_[L];
            double _ta = 0.0; if (dbg_timing) { q.wait(); _ta = now_ms(); }
            rms_norm_f32w(q, w.x, w_l.attn_norm, w.x_normed, T, H, eps);

            if (w_l.is_linear) {
                const uint32_t SKH = cfg_.ssm_n_k_heads;          // 16
                const uint32_t SVH = cfg_.ssm_n_v_heads;          // 32
                const uint32_t SHD = cfg_.ssm_head_dim;           // 128
                const uint32_t SI  = cfg_.ssm_inner;              // 4096
                const uint32_t conv_ch = SI + 2u * SKH * SHD;     // 8192
                const uint32_t kw  = SKH * SHD;                   // 2048
                const uint32_t rep = SVH / SKH;                   // 2 (16→32) — crown TILE
                const float qscale = 1.0f / sycl::sqrt(float(SHD));

                sgemv(dev, w.x_normed, w_l.attn_qkv, w.dn_qkv, H, conv_ch, T);
                sycl::half* conv_state = dn_[dev].conv_state_ptr() +
                    uint64_t(dn_local_[L]) * dn_[dev].conv_elems_per_layer();
                depthwise_conv1d_causal(q, w.dn_qkv, w_l.ssm_conv1d_fp16, conv_state,
                                        w.dn_conv, T, conv_ch, cfg_.ssm_conv_kernel);
                cast_qkv_split_fp16_to_fp32(q, w.dn_conv, w.dn_qpre, w.dn_kpre, w.dn_vpre, T, kw, SI);
                l2_norm_scale(q, w.dn_qpre, w.dn_qpre, T * SKH, SHD, qscale, 1e-6f);
                l2_norm_scale(q, w.dn_kpre, w.dn_kpre, T * SKH, SHD, 1.0f,   1e-6f);
                repeat_interleave_heads(q, w.dn_qpre, w.dn_qrep, T, SKH, SHD, rep);
                repeat_interleave_heads(q, w.dn_kpre, w.dn_krep, T, SKH, SHD, rep);
                const uint32_t SVHp = ((SVH + 63u) / 64u) * 64u;
                dense::gemv_q_T(q, w.x_normed, w_l.ssm_alpha, w.dn_alpha64, H, SVHp, T);
                dense::gemv_q_T(q, w.x_normed, w_l.ssm_beta,  w.dn_beta64,  H, SVHp, T);
                extract_cols(q, w.dn_alpha64, w.dn_alpha_h, T, SVH, SVHp);
                extract_cols(q, w.dn_beta64,  w.dn_beta_h,  T, SVH, SVHp);
                compute_g_beta_h16(q, w.dn_alpha_h, w.dn_beta_h, w_l.ssm_a, w_l.ssm_dt_bias,
                                   w.dn_g, w.dn_beta, T, SVH);
                float* state_layer = dn_[dev].state_ptr() +
                    uint64_t(dn_local_[L]) * dn_[dev].state_elems_per_layer();
                deltanet_recurrence(q, w.dn_qrep, w.dn_krep, w.dn_vpre, w.dn_g, w.dn_beta,
                                    state_layer, w.dn_out, /*B=*/1, T, SVH, SHD, SHD);
                sgemv(dev, w.x_normed, w_l.attn_gate, w.dn_z, H, SI, T);
                gated_rms_norm(q, w.dn_out, w.dn_z, w_l.ssm_norm_fp16, w.dn_qkv, T * SVH, SHD, eps);
                sgemv(dev, w.dn_qkv, w_l.ssm_out, w.attn_block, SI, H, T);
            } else {
                sgemv(dev, w.x_normed, w_l.attn_q, w.qg, H, N_qg, T);
                split_q_gate_per_head(q, w.qg, w.q, w.gate, T, cfg_.n_q_heads, HD);
                sgemv(dev, w.x_normed, w_l.attn_k, w.k, H, N_kv, T);
                sgemv(dev, w.x_normed, w_l.attn_v, w.v, H, N_kv, T);
                rms_norm_f32w(q, w.q, w_l.attn_q_norm, w.q, T * cfg_.n_q_heads,  HD, eps);
                rms_norm_f32w(q, w.k, w_l.attn_k_norm, w.k, T * cfg_.n_kv_heads, HD, eps);
                rope_partial(q, w.q, w.positions, w.q, T, cfg_.n_q_heads,  HD, rope_n, cfg_.rope_theta);
                rope_partial(q, w.k, w.positions, w.k, T, cfg_.n_kv_heads, HD, rope_n, cfg_.rope_theta);
                sycl::half* kc = kv_[dev].k_ptr() + per_layer_kv * kv_local_[L];
                sycl::half* vc = kv_[dev].v_ptr() + per_layer_kv * kv_local_[L];
                const uint32_t max_ctx = kv_[dev].config().max_ctx;
                if (T == 1 && w.attn_partials) {
                    full_attention_fa2_decode(q, w.q, w.k, w.v, kc, vc, w.attn_out,
                                              w.attn_partials, start_pos,
                                              cfg_.n_q_heads, cfg_.n_kv_heads, HD, max_ctx);
                } else {
                    full_attention(q, w.q, w.k, w.v, kc, vc, w.attn_out, T, start_pos,
                                   cfg_.n_q_heads, cfg_.n_kv_heads, HD, max_ctx);
                }
                kv_[dev].set_length(kv_local_[L], start_pos + T);
                sigmoid_gate(q, w.attn_out, w.gate, w.attn_out, uint64_t(T) * N_q);
                sgemv(dev, w.attn_out, w_l.attn_output, w.attn_block, N_q, H, T);
            }

            // residual + post-attn norm → MoE.
            residual_add_rms_norm_fused(q, w.x, w.attn_block, w_l.post_attn_norm, w.x_normed, T, H, eps);
            if (dbg_timing) { q.wait(); t_attn += now_ms() - _ta; _ta = now_ms(); }
            if (std::getenv("IE_MOE_DBG")) { std::fprintf(stderr,"[L%u T%u dev%u attn-done is_lin=%d]\n",L,T,dev,(int)w_l.is_linear); q.wait(); }

            // ---- MoE — FULLY GPU-RESIDENT (no host pull, no per-expert host loop):
            // router → quantize x → gate_up_silu_q8 → quantize h → down_q8 → reduce.
            // topk_idx/topk_w stay on device; the grouped kernels read the expert id
            // per token-slot from device memory. ----
            moe_router(q, w.x_normed, w_l.ffn_gate_inp, w.topk_idx, w.topk_w, T, H, E, Kt);
            if (std::getenv("IE_MOE_DBG") && dev > 0) {   // card-1 router input + topk sanity
                std::vector<sycl::half> _xn(H);
                q.memcpy(_xn.data(), w.x_normed, H * sizeof(sycl::half)).wait();
                int _bad = 0; float _mx = 0.f;
                for (auto hh : _xn) { float v = float(hh); if (v != v || v > 1e30f || v < -1e30f) _bad++; float a = v < 0 ? -v : v; if (a > _mx) _mx = a; }
                std::vector<int32_t> _ti(uint64_t(T) * Kt);
                q.memcpy(_ti.data(), w.topk_idx, _ti.size() * sizeof(int32_t)).wait();
                int _tmn = 1 << 30, _tmx = -(1 << 30); for (int32_t v : _ti) { if (v < _tmn) _tmn = v; if (v > _tmx) _tmx = v; }
                std::fprintf(stderr, "[card1 L%u T%u] x_normed bad=%d maxabs=%.1f | topk min=%d max=%d (E=%u)\n", L, T, _bad, _mx, _tmn, _tmx, E);
            }
            // The DECODE (T==1) per-token-slot kernels re-stream every expert's Q8
            // weights ~TK/E times — fine at T=1 (host/comm-bound) but ~80% of pp512
            // prefill. T>1 routes through the EXPERT-BATCHED (weight-stationary) GEMM
            // analog: host counting-sort of the TK slots by expert → expert-sorted
            // gather → batched gate_up/down reading each weight column once. Numerics
            // are bit-identical (same block_q8_1x activation, same dp4a/lane order,
            // weight folded at down) so PPL is exact. Opt out: IE_Q35MOE_NO_PREFILL_GEMM.
            static const bool no_prefill_gemm = std::getenv("IE_Q35MOE_NO_PREFILL_GEMM") != nullptr;
            if (T > 1 && !no_prefill_gemm) {
                const uint32_t TOTAL = T * Kt;
                // Pull topk to host (the only blocking read), counting-sort by expert,
                // push the partition + gather index + reduce map back (mirror qwen36.cpp).
                std::vector<int32_t>    hidx(TOTAL);
                std::vector<sycl::half> htw(TOTAL);
                q.memcpy(hidx.data(), w.topk_idx, TOTAL * sizeof(int32_t));
                q.memcpy(htw.data(),  w.topk_w,   TOTAL * sizeof(sycl::half)).wait();
                std::vector<uint32_t> off(E + 1, 0);
                for (uint32_t i = 0; i < TOTAL; ++i) ++off[uint32_t(hidx[i]) + 1];
                for (uint32_t ee = 0; ee < E; ++ee) off[ee + 1] += off[ee];
                std::vector<int32_t>    sidx(TOTAL);
                std::vector<sycl::half> sw(TOTAL);
                std::vector<uint32_t>   tk2pk(TOTAL);
                std::vector<uint32_t>   cursor(E, 0);
                for (uint32_t t = 0; t < T; ++t)
                    for (uint32_t kk = 0; kk < Kt; ++kk) {
                        const uint32_t ex  = uint32_t(hidx[t * Kt + kk]);
                        const uint32_t pos = off[ex] + cursor[ex]++;
                        sidx[pos]            = int32_t(t);
                        sw[pos]              = htw[t * Kt + kk];
                        tk2pk[t * Kt + kk]   = pos;
                    }
                q.memcpy(w.moe_eoff,  off.data(),   (E + 1) * sizeof(uint32_t));
                q.memcpy(w.moe_sidx,  sidx.data(),  TOTAL * sizeof(int32_t));
                q.memcpy(w.moe_sw,    sw.data(),    TOTAL * sizeof(sycl::half));
                q.memcpy(w.moe_tk2pk, tk2pk.data(), TOTAL * sizeof(uint32_t)).wait();

                moe_gather_rows(q, w.x_normed, w.moe_sidx, w.moe_xp, TOTAL, H);
                quantize_q8_1(q, w.moe_xp, static_cast<block_q8_1x*>(w.xp_q8), uint32_t(TOTAL) * H);
                moe_prefill_gate_up_silu_q8(q, w.xp_q8, w_l.exp_gate.qs, w_l.exp_gate.d,
                                            w_l.exp_up.qs, w_l.exp_up.d, w_l.exp_gate.qs_stride,
                                            w_l.exp_gate.d_stride, w.moe_eoff, w.moe_hp, E, H, EFF);
                quantize_q8_1(q, w.moe_hp, static_cast<block_q8_1x*>(w.h_q8), uint32_t(TOTAL) * EFF);
                moe_prefill_down_q8(q, w.h_q8, w_l.exp_down.qs, w_l.exp_down.d,
                                    w_l.exp_down.qs_stride, w_l.exp_down.d_stride, w.moe_eoff,
                                    w.moe_sw, w.moe_yp, E, H, EFF);
                moe_prefill_reduce_sum(q, w.moe_yp, w.moe_tk2pk, w.moe_y, T, Kt, H);
            } else {
            // DECODE (and IE_Q35MOE_NO_PREFILL_GEMM A/B fallback) — per-token-slot
            // grouped kernels. Activation → block_q8_1x in ONE batched quantize.
            // H % 32 == 0 so the per-32 blocks are row-independent → quantizing all
            // T·H elements at once is identical to a per-row loop but a SINGLE enqueue
            // (the per-row loop overran the device command queue → DEVICE_LOST).
            quantize_q8_1(q, w.x_normed, static_cast<block_q8_1x*>(w.x_q8), uint32_t(T) * H);
            moe_gate_up_silu_q8(q, w.x_q8, w_l.exp_gate.qs, w_l.exp_gate.d,
                                w_l.exp_up.qs, w_l.exp_up.d, w_l.exp_gate.qs_stride,
                                w_l.exp_gate.d_stride, w.topk_idx, w.moe_hp, T, Kt, H, EFF);
            quantize_q8_1(q, w.moe_hp, static_cast<block_q8_1x*>(w.h_q8), uint32_t(T) * Kt * EFF);
            moe_down_q8(q, w.h_q8, w_l.exp_down.qs, w_l.exp_down.d, w_l.exp_down.qs_stride,
                        w_l.exp_down.d_stride, w.topk_idx, w.topk_w, w.moe_yp, T, Kt, EFF, H);
            moe_reduce_q8(q, w.moe_yp, w.moe_y, T, Kt, H);   // moe_y = Σ_k weighted experts
            }
            // shared expert (always-on, sigmoid-gated).
            if (w_l.ffn_gate_inp_shexp) {
                sgemv(dev, w.x_normed, w_l.sh_gate, w.gate_o, H, EFF, T);
                sgemv(dev, w.x_normed, w_l.sh_up,   w.up_o,   H, EFF, T);
                swiglu(q, w.gate_o, w.up_o, w.ffn_h, uint64_t(T) * EFF);
                sgemv(dev, w.ffn_h, w_l.sh_down, w.eo, EFF, H, T);
                // shared-expert gate g[t]=sigmoid(ffn_gate_inp_shexp · x_normed[t]);
                // moe_y += eo*g. [VERIFY gate semantics vs qwen36.cpp:1775-1829.]
                {
                    float* dng = w.dn_g; const float* gw = w_l.ffn_gate_inp_shexp;
                    const sycl::half* xn = w.x_normed; const sycl::half* eo = w.eo;
                    sycl::half* my = w.moe_y; const uint32_t HH = H;
                    q.parallel_for(sycl::range<1>(T), [=](sycl::id<1> ti) {
                        const uint32_t t = uint32_t(ti); float gg = 0.f;
                        for (uint32_t h = 0; h < HH; ++h) gg += gw[h] * float(xn[uint64_t(t) * HH + h]);
                        dng[t] = 1.f / (1.f + sycl::exp(-gg));
                    });
                    q.parallel_for(sycl::range<1>(uint64_t(T) * HH), [=](sycl::id<1> i) {
                        const uint32_t t = uint32_t(uint64_t(i) / HH);
                        my[i] = sycl::half(float(my[i]) + float(eo[i]) * dng[t]);
                    });
                }
            }
            residual_add(q, w.x, w.moe_y, w.x, uint64_t(T) * H);
            if (dbg_timing) { q.wait(); t_moe += now_ms() - _ta; }
            if (std::getenv("IE_MOE_DBG")) { std::fprintf(stderr,"[L%u T%u dev%u moe-done]\n",L,T,dev); q.wait(); }
        }
        q.wait();
    }
    if (dbg_timing)
        std::fprintf(stderr, "[q35moe-timing T=%u] attn=%.2f ms  moe=%.2f ms  (moe %.0f%%)\n",
                     T, t_attn, t_moe, 100.0 * t_moe / (t_attn + t_moe + 1e-9));

    // final norm + lm_head on head_dev.
    {
        const uint32_t hd = plan_.head_dev;
        Workspace& w = ws_[hd];
        auto& alloc = fleet_->dev(hd);
        auto& q = alloc.queue();
        rms_norm_f32w(q, w.x, output_norm_, w.x_normed, T, H, eps);
        const sycl::half* last = w.x_normed + uint64_t(T - 1) * H;
        sycl::half* d_logits = static_cast<sycl::half*>(alloc.malloc(uint64_t(V) * sizeof(sycl::half)));
        if (!d_logits) return "logits alloc failed";
        sgemv(hd, last, output_, d_logits, H, V, 1).wait();
        q.memcpy(out_logits_host, d_logits, uint64_t(V) * sizeof(sycl::half)).wait();
        alloc.free(d_logits);
    }
    return {};
}

}  // namespace ie
