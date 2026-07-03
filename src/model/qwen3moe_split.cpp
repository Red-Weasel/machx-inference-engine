// src/model/qwen3moe_split.cpp — Qwen3 standard MoE (`kQwen3Moe`) multi-GPU
// layer-split. See header. ADDITIVE: mirrors the crown Qwen35MoeSplitModel
// orchestration (qwen35moe_split.cpp) — device-by-device forward, per-card
// weights + KV, one residual copy per card boundary — with the qwen3moe
// per-layer math (qwen3moe.cpp) lifted into the loop:
//   * attention = standard Qwen3 QK-norm GQA full-attn (ungated), dense::gemv_q_T
//   * FFN       = top-k MoE, host-routed, int-dot W4A8/W6A8 fused kernels
//                 (Approach B: Q4_K_M / Q6_K experts, NOT all-Q8_0)
// The single-GPU Qwen3MoeModel is NEVER edited (its PPL gate stays put).

#include "ie/qwen3moe_split.hpp"

#include "ie/dequant.hpp"
#include "ie/moe_qwen3.hpp"          // moe_prefill_gate_up_silu_q6k_q8, *_down_q{4,6}k_q8_gen
#include "ie/ops.hpp"
#include "ie/qwen3moe_pack.hpp"      // MoePacking, build_moe_packing
#include "ie/quant_blocks.hpp"
#include "ie/quant_soa.hpp"          // repack_moe_q{4,6}k_soa_host
#include "ie/quantize.hpp"           // quantize_row_q6_K (BF16 dynamic-quant experts → Q6_K)

#include "dense_dispatch.hpp"        // dense::upload<T>, dense::upload_quant_dense, dense::gemv_q(_T)

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace ie {
namespace {

// Requantize a non-Q4_K/Q6_K 3-D expert tensor (BF16/F16/F32) to Q6_K on the host,
// to match the split's native Q6_K expert kernel + SoA path. Unsloth "dynamic"/XL
// qwen3moe quants keep a few sensitive expert layers at BF16 (higher precision than
// Q6_K); this makes them load. Tensor is [K, N, E]; K must be a multiple of 256.
std::string requant_exps_to_q6k(const GgufTensorInfo* t, std::vector<block_q6_K>& out) {
    const uint64_t K = t->shape[0], N = t->shape[1], E = t->shape[2];
    if (K % 256 != 0) return "requant->Q6_K: expert K % 256 != 0";
    const uint64_t nsb = K / 256;                          // Q6_K superblocks per row (QK_K=256)
    out.assign(E * N * nsb, block_q6_K{});
    const auto* raw16 = reinterpret_cast<const uint16_t*>(t->data);
    const auto* raw32 = reinterpret_cast<const float*>(t->data);
    const DType dt = t->dtype;
    std::vector<float> rowf(K);
    for (uint64_t e = 0; e < E; ++e)
        for (uint64_t n = 0; n < N; ++n) {
            const uint64_t off = (e * N + n) * K;
            for (uint64_t k = 0; k < K; ++k) {
                if      (dt == DType::kF32) rowf[k] = raw32[off + k];
                else if (dt == DType::kF16) rowf[k] = fp16_to_fp32(raw16[off + k]);
                else { uint32_t b = uint32_t(raw16[off + k]) << 16; float f;
                       std::memcpy(&f, &b, sizeof(f)); rowf[k] = f; }   // BF16 = top 16 bits of fp32
            }
            quantize_row_q6_K(rowf.data(), out.data() + (e * N + n) * nsb, int64_t(K));
        }
    return {};
}

// COPY of qwen3moe.cpp:upload_dequant_to_fp16 (Q5_K/Q8_0 → device F16 [K,N]).
// copy-not-hoist discipline (the single-GPU path is never edited).
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

// COPY of qwen3moe.cpp:upload_weight_auto (Q4_K/Q6_K/F16 ride dense GEMV; Q5_K/Q8_0
// dequant to F16). Attention projections consume this.
DenseQuantPtr upload_weight_auto(DeviceAllocator& alloc, const GgufTensorInfo* t,
                                 std::vector<void*>& owned, std::string& err) {
    if (!t) { err = "tensor not found"; return {}; }
    if (t->dtype == DType::kQ5_K || t->dtype == DType::kQ8_0)
        return upload_dequant_to_fp16(alloc, t, owned, err);
    return dense::upload_quant_dense(alloc, t, owned, err);
}

// COPY of qwen3moe.cpp:upload_expert_soa. Repack an AoS expert-bank tensor
// (Q4_K/Q6_K, [n_experts, ...]) to per-expert SoA streams (intra-expert reorder →
// per-expert byte stride unchanged). `staging` is caller-owned, reused across the
// (gate/up/down × n_layers) tensors of ONE card.
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

// COPY of qwen3moe.cpp:route_from_logits. Softmax-after-topk over precomputed
// logits[E]: softmax all E, take top-k by prob, renormalize the k to sum 1, sort
// ascending by expert index (HF norm_topk_prob order). Matches the single-GPU
// DEFAULT router numerics exactly (GPU-gemm logits + this host softmax/top-K).
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

Qwen3MoeSplitModel::~Qwen3MoeSplitModel() { free_all(); }

void Qwen3MoeSplitModel::free_all() {
    if (!fleet_) return;
    for (uint32_t d = 0; d < ws_.size(); ++d) free_ws(d);
    ws_.clear();
    for (uint32_t d = 0; d < owned_.size(); ++d)
        for (void* p : owned_[d]) if (p) fleet_->dev(d).free(p);
    owned_.clear();
    for (auto& c : kv_) c.free_storage();
    kv_.clear();
}

std::string Qwen3MoeSplitModel::load(DeviceFleet& fleet, const LayerPlan& plan,
                                     const GgufReader& g, const Qwen3MoeConfig& cfg,
                                     uint32_t max_ctx) {
    fleet_ = &fleet; plan_ = plan; cfg_ = cfg;
    n_dev_ = plan.n_dev();
    if (n_dev_ == 0 || plan.dev_of_layer.empty()) return "qwen3moe_split: empty plan";

    // oneDNN OFF on the split (mirror crown; the int-dot MoE never uses it, and
    // multi-card oneDNN is a DEVICE_LOST landmine unless routed per-device).
    dense::prefer_onednn() = false;

    const auto& d = cfg_.dense;
    const uint32_t H = d.hidden, E = cfg_.n_experts, EF = cfg_.expert_ffn;

    // Approach B q8 int-dot MoE preconditions (static, config-only). Fail LOUDLY
    // at load rather than silently mis-route: the fused gate/up kernel needs
    // H%512==0 and the _gen down kernel needs E_ffn%256==0.
    if (H % 512 != 0)  return "qwen3moe_split: hidden % 512 != 0 (int-dot MoE unsupported)";
    if (EF % 256 != 0) return "qwen3moe_split: expert_ffn % 256 != 0 (int-dot MoE unsupported)";

    owned_.assign(n_dev_, {});
    dev_bytes_.assign(n_dev_, 0);

    char buf[96];
    auto Ttop = [&](const char* n) { return g.find_tensor(n); };
    auto Tl   = [&](uint32_t L, const char* n) {
        std::snprintf(buf, sizeof(buf), "blk.%u.%s", L, n); return g.find_tensor(buf);
    };
    std::string err;

    // Global SoA decision (per-tensor gated to Q6_K below; Q4_K stays AoS —
    // already coalesced). Mirrors qwen3moe.cpp load. Opt-out IE_NO_MOE_SOA.
    const auto* gate0 = Ttop("blk.0.ffn_gate_exps.weight");
    const auto* up0   = Ttop("blk.0.ffn_up_exps.weight");
    const bool gateup_soa_ok =
        (!gate0 || gate0->dtype == DType::kQ4_K || gate0->dtype == DType::kQ6_K) &&
        (!up0   || up0->dtype   == DType::kQ4_K || up0->dtype   == DType::kQ6_K);
    moe_exps_soa_ = std::getenv("IE_NO_MOE_SOA") == nullptr && gateup_soa_ok;

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
        if (!ti) { output_ = token_embd_; output_dtype_ = token_embd_dtype_; }   // tied
        else {
            if (ti->dtype != DType::kQ4_K && ti->dtype != DType::kQ6_K)
                return "output: unsupported dtype (need Q4_K/Q6_K)";
            output_dtype_ = ti->dtype;
            output_ = dense::upload<void>(ha, ti, owned_[plan.head_dev], err, ti->dtype);
            if (!err.empty()) return "output: " + err;
            dev_bytes_[plan.head_dev] += ti->nbytes;
        }
    }

    const uint32_t n_layers = d.n_layers;
    layers_.assign(n_layers, {});
    kv_local_.assign(n_layers, 0);
    std::vector<std::vector<uint8_t>> soa_staging(n_dev_);   // per-card reuse buffer

    for (uint32_t L = 0; L < n_layers; ++L) {
        const uint32_t dev = plan.dev_of_layer[L];
        DeviceAllocator& a = fleet.dev(dev);
        auto& own = owned_[dev];
        LayerW& w = layers_[L];

        auto F32 = [&](const char* n, float*& dst) -> std::string {
            const auto* ti = Tl(L, n);
            dst = dense::upload<float>(a, ti, own, err, DType::kF32);
            if (!err.empty()) return std::string(n) + ": " + err;
            if (ti) dev_bytes_[dev] += ti->nbytes;
            return {};
        };
        auto LQ = [&](const char* n) -> DenseQuantPtr {
            const auto* ti = Tl(L, n);
            DenseQuantPtr p = upload_weight_auto(a, ti, own, err);
            if (err.empty() && ti) dev_bytes_[dev] += ti->nbytes;
            return p;
        };
        auto le = [&](const char* what) {
            return "layer " + std::to_string(L) + " " + what + ": " + err;
        };

        if (auto m = F32("attn_norm.weight", w.attn_norm); !m.empty())
            return "layer " + std::to_string(L) + " " + m;
        if (auto m = F32("ffn_norm.weight", w.ffn_norm); !m.empty())
            return "layer " + std::to_string(L) + " " + m;

        w.attn_q = LQ("attn_q.weight");      if (!err.empty()) return le("attn_q");
        w.attn_k = LQ("attn_k.weight");      if (!err.empty()) return le("attn_k");
        w.attn_v = LQ("attn_v.weight");      if (!err.empty()) return le("attn_v");
        w.attn_output = LQ("attn_output.weight"); if (!err.empty()) return le("attn_output");
        if (Tl(L, "attn_q_norm.weight"))
            if (auto m = F32("attn_q_norm.weight", w.attn_q_norm); !m.empty())
                return "layer " + std::to_string(L) + " " + m;
        if (Tl(L, "attn_k_norm.weight"))
            if (auto m = F32("attn_k_norm.weight", w.attn_k_norm); !m.empty())
                return "layer " + std::to_string(L) + " " + m;

        // MoE router: F32 [E, H] → device F16 [H, E] transposed for gemv_q_T.
        {
            const auto* ti = Tl(L, "ffn_gate_inp.weight");
            if (!ti) return le("ffn_gate_inp (not found)");
            if (ti->dtype != DType::kF32) return le("ffn_gate_inp: expected F32");
            if (ti->nbytes != uint64_t(E) * H * sizeof(float))
                return le("ffn_gate_inp: unexpected size");
            const float* rw = reinterpret_cast<const float*>(ti->data);
            std::vector<sycl::half> rt(uint64_t(H) * E);
            for (uint32_t e = 0; e < E; ++e)
                for (uint32_t h = 0; h < H; ++h)
                    rt[uint64_t(h) * E + e] = sycl::half(rw[uint64_t(e) * H + h]);
            auto* rd = static_cast<sycl::half*>(a.malloc(uint64_t(H) * E * sizeof(sycl::half)));
            if (!rd) return le("router_w_dev alloc failed");
            a.queue().memcpy(rd, rt.data(), uint64_t(H) * E * sizeof(sycl::half)).wait();
            own.push_back(rd);
            w.router_w_dev = rd;
            dev_bytes_[dev] += uint64_t(H) * E * sizeof(sycl::half);
        }

        // MoE experts: raw stacked device banks, per-expert stride = nbytes/E.
        // SoA (intra-expert reorder) only for Q6_K; Q4_K stays AoS.
        auto up_exps = [&](const char* nm, void*& dst, DType& dt, uint64_t& stride,
                           bool& soa) -> std::string {
            const auto* ti = Tl(L, nm);
            if (!ti) return std::string(nm) + ": not found";
            std::vector<block_q6_K> requant;   // holds requantized data if the tensor isn't Q4_K/Q6_K
            GgufTensorInfo synth;
            if (ti->dtype != DType::kQ4_K && ti->dtype != DType::kQ6_K) {
                if (ti->dtype != DType::kBF16 && ti->dtype != DType::kF16 &&
                    ti->dtype != DType::kF32)
                    return std::string(nm) + ": unsupported expert dtype " +
                           std::string(type_name(ti->dtype)) +
                           " (need Q4_K/Q6_K, or BF16/F16/F32 for dynamic quants)";
                if (auto m = requant_exps_to_q6k(ti, requant); !m.empty())
                    return std::string(nm) + ": " + m;   // unsloth dynamic/XL: dequant→Q6_K at load
                synth = *ti;
                synth.dtype  = DType::kQ6_K;
                synth.data   = reinterpret_cast<const uint8_t*>(requant.data());
                synth.nbytes = requant.size() * sizeof(block_q6_K);
                ti = &synth;
            }
            if (ti->nbytes % E) return std::string(nm) + ": nbytes not divisible by expert_count";
            soa = moe_exps_soa_ && ti->dtype == DType::kQ6_K;
            if (soa) {
                dst = upload_expert_soa(a, ti, E, soa_staging[dev], own, err);
                if (!err.empty()) return std::string(nm) + ": " + err;
            } else {
                dst = dense::upload<void>(a, ti, own, err, ti->dtype);
                if (!err.empty()) return std::string(nm) + ": " + err;
            }
            dt = ti->dtype; stride = ti->nbytes / E;   // intra-expert reorder → stride unchanged
            dev_bytes_[dev] += ti->nbytes;
            return {};
        };
        if (auto m = up_exps("ffn_gate_exps.weight", w.gate_exps, w.gate_dt, w.gate_stride, w.gate_soa); !m.empty()) return "layer " + std::to_string(L) + " " + m;
        if (auto m = up_exps("ffn_up_exps.weight",   w.up_exps,   w.up_dt,   w.up_stride,   w.up_soa);   !m.empty()) return "layer " + std::to_string(L) + " " + m;
        if (auto m = up_exps("ffn_down_exps.weight", w.down_exps, w.down_dt, w.down_stride, w.down_soa); !m.empty()) return "layer " + std::to_string(L) + " " + m;
        // The fused gate/up kernel selects on gate_dt and passes ONE stride/soa for
        // both banks → gate and up must share dtype (they always do in practice).
        if (w.gate_dt != w.up_dt) return "layer " + std::to_string(L) + " gate/up expert dtype mismatch (unsupported)";
    }

    // Per-card full-attn KV. qwen3moe is ALL full-attention → each card's KV holds
    // every layer assigned to it; kv_local_[L] is the local layer index on its card.
    std::vector<uint32_t> n_full(n_dev_, 0);
    for (uint32_t L = 0; L < n_layers; ++L)
        kv_local_[L] = n_full[plan.dev_of_layer[L]]++;
    kv_.resize(n_dev_);
    for (uint32_t dev = 0; dev < n_dev_; ++dev) {
        if (!n_full[dev]) continue;
        KvCacheConfig kc{};
        kc.n_layers_full = n_full[dev];
        kc.n_kv_heads    = d.n_kv_heads;      // read from GGUF (NOT hardcoded)
        kc.max_ctx       = max_ctx;
        kc.head_dim      = d.head_dim;
        if (auto m = kv_[dev].init(fleet.dev(dev), kc); !m.empty())
            return "kv cache dev " + std::to_string(dev) + ": " + m;
    }

    for (uint32_t dev = 0; dev < n_dev_; ++dev)
        std::fprintf(stderr, "[qwen3moe_split] card %u weights: %.2f GB\n",
                     dev, double(dev_bytes_[dev]) / 1e9);
    return {};
}

void Qwen3MoeSplitModel::free_ws(uint32_t dev) {
    if (!fleet_ || dev >= ws_.size()) return;
    Workspace& w = ws_[dev];
    auto& alloc = fleet_->dev(dev);
    for (void* p : {static_cast<void*>(w.x), static_cast<void*>(w.x_normed),
                    static_cast<void*>(w.q), static_cast<void*>(w.k), static_cast<void*>(w.v),
                    static_cast<void*>(w.attn_out), static_cast<void*>(w.attn_block),
                    static_cast<void*>(w.positions), static_cast<void*>(w.attn_partials),
                    static_cast<void*>(w.router_logits),
                    static_cast<void*>(w.xp_packed), w.xp_q8,
                    static_cast<void*>(w.h_packed), w.h_q8,
                    static_cast<void*>(w.out_packed), static_cast<void*>(w.ffn_y),
                    static_cast<void*>(w.expert_offsets), static_cast<void*>(w.sorted_idx),
                    static_cast<void*>(w.tk_to_packed), static_cast<void*>(w.weights_packed)})
        if (p) alloc.free(p);
    w = Workspace{};
}

std::string Qwen3MoeSplitModel::ensure_ws(uint32_t dev, uint32_t max_T) {
    Workspace& w = ws_[dev];
    if (max_T <= w.T) return {};
    if (w.T != 0) free_ws(dev);
    auto& alloc = fleet_->dev(dev);
    auto ah = [&](uint64_t n) { return static_cast<sycl::half*>(alloc.malloc(n * sizeof(sycl::half))); };

    const auto& d = cfg_.dense;
    const uint64_t T   = max_T;
    const uint32_t H   = d.hidden;
    const uint32_t HD  = d.head_dim;
    const uint32_t N_q = d.n_q_heads  * HD;
    const uint32_t N_kv= d.n_kv_heads * HD;
    const uint32_t EF  = cfg_.expert_ffn;
    const uint32_t E   = cfg_.n_experts;
    const uint32_t K   = cfg_.n_experts_used;
    const uint64_t TK  = T * K;

    w.x = ah(T*H); w.x_normed = ah(T*H);
    w.q = ah(T*N_q); w.k = ah(T*N_kv); w.v = ah(T*N_kv);
    w.attn_out = ah(T*N_q); w.attn_block = ah(T*H);
    w.positions = static_cast<int32_t*>(alloc.malloc(T * sizeof(int32_t)));
    w.router_logits = ah(T*E);
    w.xp_packed  = ah(TK*H);
    w.xp_q8      = alloc.malloc(TK * (H / 32) * sizeof(block_q8_1s));
    w.h_packed   = ah(TK*EF);
    w.h_q8       = alloc.malloc(TK * (EF / 32) * sizeof(block_q8_1s));
    w.out_packed = ah(TK*H);
    w.ffn_y      = ah(T*H);
    w.expert_offsets = static_cast<uint32_t*>(alloc.malloc((uint64_t(E) + 1) * sizeof(uint32_t)));
    w.sorted_idx     = static_cast<int32_t*>(alloc.malloc(TK * sizeof(int32_t)));
    w.tk_to_packed   = static_cast<int32_t*>(alloc.malloc(TK * sizeof(int32_t)));
    w.weights_packed = ah(TK);

    if (!w.x || !w.x_normed || !w.q || !w.k || !w.v || !w.attn_out || !w.attn_block ||
        !w.positions || !w.router_logits || !w.xp_packed || !w.xp_q8 || !w.h_packed ||
        !w.h_q8 || !w.out_packed || !w.ffn_y || !w.expert_offsets || !w.sorted_idx ||
        !w.tk_to_packed || !w.weights_packed)
        return "qwen3moe_split workspace alloc failed on dev " + std::to_string(dev);

    // FA-2 split-K decode partials, sized for this card's KV ctx. Mirror
    // qwen35moe_split.cpp / qwen3moe.cpp:ensure_attn_partials.
    if (kv_[dev].ready()) {
        const uint32_t max_ctx = kv_[dev].config().max_ctx;
        constexpr uint32_t Bc_floor = 64;
        const uint32_t n_chunks_max = (max_ctx + Bc_floor - 1) / Bc_floor;
        const uint64_t n_floats = uint64_t(n_chunks_max) * d.n_q_heads * (d.head_dim + 2);
        w.attn_partials = static_cast<float*>(alloc.malloc(n_floats * sizeof(float)));
        if (!w.attn_partials) return "qwen3moe_split attn_partials alloc failed on dev " + std::to_string(dev);
        w.partials_ctx = max_ctx;
    }
    w.T = max_T;
    return {};
}

// Inlined qwen3moe fused-MoE sequence for one layer on one card. Router = GPU
// gemv (F16 transposed router) + host softmax/top-K (single-GPU DEFAULT numerics),
// then expert-sorted int-dot W4A8/W6A8. Leaves the per-token MoE output in
// ws_[dev].ffn_y. Runs for ALL T (prefill AND T==1 decode), like the single-GPU
// default (Qwen3MoeModel::forward routes qwen3moe through moe_ffn_fused_prefill for
// every T when gate/up is Q4_K/Q6_K on the q8 path).
void Qwen3MoeSplitModel::moe_ffn(uint32_t dev, const LayerW& w, uint32_t T) {
    auto& q = fleet_->dev(dev).queue();
    Workspace& ws = ws_[dev];
    const auto& d = cfg_.dense;
    const uint32_t H = d.hidden, EF = cfg_.expert_ffn, E = cfg_.n_experts, K = cfg_.n_experts_used;
    const uint32_t TK = T * K;

    // 1) router logits = x_normed[T,H] @ router_w_devᵀ → [T,E] (GPU gemm), pull to
    //    host, softmax/top-K per token (cheap; E entries). Matches single-GPU default.
    dense::gemv_q_T(q, ws.x_normed, DenseQuantPtr{w.router_w_dev, DType::kF16},
                    ws.router_logits, H, E, T);
    std::vector<sycl::half> hl(uint64_t(T) * E);
    q.memcpy(hl.data(), ws.router_logits, uint64_t(T) * E * sizeof(sycl::half)).wait();
    if (host_logits_.size() < uint64_t(T) * E) host_logits_.resize(uint64_t(T) * E);
    for (uint64_t i = 0; i < uint64_t(T) * E; ++i) host_logits_[i] = float(hl[i]);
    std::vector<std::vector<std::pair<uint32_t, float>>> routes(T);
    std::vector<std::pair<uint32_t, float>> topk;
    for (uint32_t t = 0; t < T; ++t) {
        route_from_logits(host_logits_.data() + uint64_t(t) * E, E, K, topk);
        routes[t] = topk;
    }

    // 2) build the expert-sorted packing on host, upload the device arrays.
    MoePacking pk;
    build_moe_packing(routes, E, K, pk);
    q.memcpy(ws.expert_offsets, pk.expert_offsets.data(), (uint64_t(E) + 1) * sizeof(uint32_t));
    q.memcpy(ws.sorted_idx,     pk.sorted_idx.data(),     uint64_t(TK) * sizeof(int32_t));
    q.memcpy(ws.tk_to_packed,   pk.tk_to_packed.data(),   uint64_t(TK) * sizeof(int32_t));
    std::vector<sycl::half> wpk(TK);
    for (uint32_t i = 0; i < TK; ++i) wpk[i] = sycl::half(pk.weights_packed[i]);
    q.memcpy(ws.weights_packed, wpk.data(), uint64_t(TK) * sizeof(sycl::half)).wait();

    // 3) gather expert-sorted input rows.
    moe_gather_rows(q, ws.x_normed, ws.sorted_idx, ws.xp_packed, TK, H);

    // 4) stage 1: gate+up+silu per expert (int-dot W4A8 / W6A8).
    quantize_q8_1s(q, ws.xp_packed, ws.xp_q8, uint64_t(TK) * H);
    if (w.gate_dt == DType::kQ6_K)
        moe_prefill_gate_up_silu_q6k_q8(q, ws.xp_q8, w.gate_exps, w.up_exps,
                                        ws.expert_offsets, ws.h_packed,
                                        E, H, EF, w.gate_stride, w.gate_soa);
    else
        moe_prefill_gate_up_silu_q4k_q8(q, ws.xp_q8, w.gate_exps, w.up_exps,
                                        ws.expert_offsets, ws.h_packed,
                                        E, H, EF, w.gate_stride, w.gate_soa);

    // 5) stage 2: down per expert → out_packed [TK, H] (int-dot, E_ffn-general _gen).
    quantize_q8_1s(q, ws.h_packed, ws.h_q8, uint64_t(TK) * EF);
    if (w.down_dt == DType::kQ6_K)
        moe_prefill_down_q6k_q8_gen(q, ws.h_q8, w.down_exps, ws.expert_offsets,
                                    ws.out_packed, E, H, EF, w.down_stride, w.down_soa);
    else
        moe_prefill_down_q4k_q8_gen(q, ws.h_q8, w.down_exps, ws.expert_offsets,
                                    ws.out_packed, E, H, EF, w.down_stride, w.down_soa);

    // 6) reduce per token. CRITICAL: moe_prefill_reduce ACCUMULATES onto ffn_y
    //    (acc = y[t*H+n] base), and ffn_y persists across this card's layers, so
    //    ffn_y MUST be zeroed before EVERY reduce for ALL T — not just T==1. The
    //    old T==1-only memset piles each layer's MoE onto the prior layer's stale
    //    output at T>1 → corrupted hidden states (PPL-streaming-T==1 won't catch it).
    q.memset(ws.ffn_y, 0, uint64_t(T) * H * sizeof(sycl::half));
    moe_prefill_reduce(q, ws.out_packed,
                       reinterpret_cast<const uint32_t*>(ws.tk_to_packed),
                       ws.weights_packed, ws.ffn_y, T, K, H);
}

std::string Qwen3MoeSplitModel::forward(const int32_t* input_ids, uint32_t T,
                                        uint32_t start_pos, bool reset_kv,
                                        sycl::half* out_logits_host) {
    if (T == 0) return "T == 0";
    const auto& d = cfg_.dense;
    const uint32_t H    = d.hidden;
    const uint32_t HD   = d.head_dim;
    const uint32_t N_q  = d.n_q_heads  * HD;
    const uint32_t N_kv = d.n_kv_heads * HD;
    const uint32_t V    = d.vocab;
    const uint32_t n_kv = d.n_kv_heads;
    const float    eps  = d.rms_eps;
    const uint32_t n_layers = d.n_layers;

    if (ws_.size() != n_dev_) ws_.assign(n_dev_, {});
    for (uint32_t dev = 0; dev < n_dev_; ++dev)
        if (auto m = ensure_ws(dev, T); !m.empty()) return m;

    std::vector<int32_t> pos(T);
    for (uint32_t t = 0; t < T; ++t) pos[t] = int32_t(start_pos + t);
    for (uint32_t dev = 0; dev < n_dev_; ++dev) {
        if (ws_[dev].positions)
            fleet_->dev(dev).queue().memcpy(ws_[dev].positions, pos.data(), T * sizeof(int32_t)).wait();
        if (reset_kv && kv_[dev].ready()) kv_[dev].reset();
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

    for (uint32_t dev = 0; dev < n_dev_; ++dev) {
        Workspace& w = ws_[dev];
        auto& q = fleet_->dev(dev).queue();
        if (dev > 0) {
            // Drain the SOURCE card BEFORE the cross-card x copy (the MoE host-pull
            // no longer implicitly drains the whole card's queue), else the copy
            // races card (dev-1)'s in-flight kernels → UR_RESULT_ERROR_DEVICE_LOST.
            fleet_->dev(dev - 1).queue().wait();
            fleet_->copy_across(dev - 1, w.x, dev, ws_[dev - 1].x,
                                uint64_t(T) * H * sizeof(sycl::half));
        }
        const uint64_t per_layer_kv =
            uint64_t(n_kv) * (kv_[dev].ready() ? kv_[dev].config().max_ctx : 0u) * HD;

        for (uint32_t L = 0; L < n_layers; ++L) {
            if (plan_.dev_of_layer[L] != dev) continue;
            const LayerW& w_l = layers_[L];

            // ---- attention (standard Qwen3 QK-norm GQA, ungated) ----
            rms_norm_f32w(q, w.x, w_l.attn_norm, w.x_normed, T, H, eps);
            dense::gemv_q_T(q, w.x_normed, w_l.attn_q, w.q, H, N_q,  T);
            dense::gemv_q_T(q, w.x_normed, w_l.attn_k, w.k, H, N_kv, T);
            dense::gemv_q_T(q, w.x_normed, w_l.attn_v, w.v, H, N_kv, T);
            if (w_l.attn_q_norm) rms_norm_f32w(q, w.q, w_l.attn_q_norm, w.q, T * d.n_q_heads,  HD, eps);
            if (w_l.attn_k_norm) rms_norm_f32w(q, w.k, w_l.attn_k_norm, w.k, T * d.n_kv_heads, HD, eps);
            rope_partial(q, w.q, w.positions, w.q, T, d.n_q_heads,  HD, d.rope_dim, d.rope_theta);
            rope_partial(q, w.k, w.positions, w.k, T, d.n_kv_heads, HD, d.rope_dim, d.rope_theta);
            sycl::half* kc = kv_[dev].k_ptr() + per_layer_kv * kv_local_[L];
            sycl::half* vc = kv_[dev].v_ptr() + per_layer_kv * kv_local_[L];
            const uint32_t max_ctx = kv_[dev].config().max_ctx;
            if (T == 1 && w.attn_partials) {
                full_attention_fa2_decode(q, w.q, w.k, w.v, kc, vc, w.attn_out,
                                          w.attn_partials, start_pos,
                                          d.n_q_heads, d.n_kv_heads, HD, max_ctx);
            } else {
                full_attention(q, w.q, w.k, w.v, kc, vc, w.attn_out, T, start_pos,
                               d.n_q_heads, d.n_kv_heads, HD, max_ctx);
            }
            kv_[dev].set_length(kv_local_[L], start_pos + T);
            dense::gemv_q_T(q, w.attn_out, w_l.attn_output, w.attn_block, N_q, H, T);

            // ---- residual + ffn norm → MoE → residual ----
            residual_add(q, w.x, w.attn_block, w.x, uint64_t(T) * H);
            rms_norm_f32w(q, w.x, w_l.ffn_norm, w.x_normed, T, H, eps);
            moe_ffn(dev, w_l, T);                    // → w.ffn_y
            residual_add(q, w.x, w.ffn_y, w.x, uint64_t(T) * H);
        }
        q.wait();
    }

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
        dense::gemv_q(q, last, DenseQuantPtr{output_, output_dtype_}, d_logits, H, V).wait();
        q.memcpy(out_logits_host, d_logits, uint64_t(V) * sizeof(sycl::half)).wait();
        alloc.free(d_logits);
    }
    return {};
}

}  // namespace ie
