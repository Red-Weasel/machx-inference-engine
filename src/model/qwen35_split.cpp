// src/model/qwen35_split.cpp — Qwen3.6-27B (`kQwen35Dense`) multi-GPU layer-split.
// See header. ADDITIVE: mirrors the 80B fleet scaffold (qwen3next.cpp) with the
// validated single-GPU 27B per-layer math (qwen35_dense.cpp) lifted verbatim into
// a device-by-device loop. The single-GPU Qwen35DenseModel is NEVER edited.

#include "ie/qwen35_split.hpp"

#include "ie/dequant.hpp"
#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"

#include "dense_dispatch.hpp"   // dense::upload<T>, dense::upload_quant_dense_auto, dense::gemv_q/gemv_q_T

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace ie {
namespace {

// Host-side canonical Q6_K row dequant — mirrors ggml's dequantize_row_q6_K
// (natural element order). COPY of qwen35_dense.cpp:dequant_q6_K_row
// (copy-not-hoist discipline). Load-time only; K % 256 == 0.
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

// ssm_alpha / ssm_beta projections are F32 (or Q8_0) [K=5120, N=48] in the GGUF
// (ggml shape[0]=K → [N,K] row-major). Transpose to [K,Npad] (cols ≥N zero-padded,
// Npad=64) + cast fp16 so they ride the BATCHED gemm_fp16 path. COPY of
// qwen35_dense.cpp:upload_f32_proj_fp16 (copy-not-hoist discipline).
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
    } else if (t->dtype == DType::kQ8_0) {  // dequant on the fly
        const uint64_t bpr = K / 32;
        const auto* blocks = reinterpret_cast<const block_q8_0*>(t->data);
        for (uint64_t n = 0; n < N; ++n)
            for (uint64_t k = 0; k < K; ++k) {
                const block_q8_0& b = blocks[n * bpr + (k >> 5)];
                staging[k * Npad + n] = sycl::half(fp16_to_fp32(b.d) * float(b.qs[k & 31]));
            }
    } else {  // Q6_K: K-quant ssm proj (e.g. *-Q6_K GGUFs). Dequant per row, then transpose.
        if (K % kQK_K != 0) { err = "ssm proj Q6_K: K not a multiple of 256"; return out; }
        const uint64_t bpr = K / kQK_K;
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

// Strided extract: dst[t,h]=src[t,h], src row stride=src_stride. COPY of
// qwen35_dense.cpp:extract_cols (compacts the N-padded [T,64] gemm out → [T,48]).
inline sycl::event extract_cols(sycl::queue& q, const sycl::half* src, sycl::half* dst,
                                uint32_t T, uint32_t nh, uint32_t src_stride) {
    return q.parallel_for(sycl::range<1>(uint64_t(T) * nh), [=](sycl::id<1> i) {
        const uint32_t t = uint32_t(i) / nh, h = uint32_t(i) % nh;
        dst[uint64_t(t) * nh + h] = src[uint64_t(t) * src_stride + h];
    });
}

// Dequant a SoA-Q8_0 weight (qs[n*K+k] int8, d[n*(K/32)+b] fp16) → fp16 Bt[K,N]
// (row-major, B for gemm_fp16: Bt[k*N+n]). Used on prefill so the packed weight
// rides the existing batched gemm. q is the owning card's queue.
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

}  // namespace

Qwen35SplitModel::~Qwen35SplitModel() { free_all(); }

void Qwen35SplitModel::free_all() {
    if (!fleet_) return;
    for (uint32_t d = 0; d < ws_.size(); ++d) free_ws(d);
    ws_.clear();
    for (uint32_t d = 0; d < owned_.size(); ++d)
        for (void* p : owned_[d]) if (p) fleet_->dev(d).free(p);
    owned_.clear();
    for (uint32_t d = 0; d < prefill_bt_.size(); ++d)   // not in owned_ (re-alloc on grow)
        if (prefill_bt_[d]) fleet_->dev(d).free(prefill_bt_[d]);
    prefill_bt_.clear(); prefill_bt_cap_.clear(); act_q8_.clear();
    for (auto& s : dn_) s.free_storage();
    for (auto& c : kv_) c.free_storage();
}

std::string Qwen35SplitModel::load(DeviceFleet& fleet, const LayerPlan& plan,
                                   const GgufReader& g, const Qwen35Config& cfg,
                                   uint32_t max_ctx) {
    fleet_ = &fleet; plan_ = plan; cfg_ = cfg;
    n_dev_ = plan.n_dev();
    const DenseConfig& d = cfg.dense;

    if (n_dev_ == 0 || plan.dev_of_layer.empty()) return "qwen35split: empty plan";
    if (d.hidden == 0 || d.vocab == 0 || d.n_q_heads == 0 || d.n_kv_heads == 0 ||
        d.ffn == 0)                                return "qwen35split: zero dim";
    if (cfg.ssm_inner == 0 || cfg.ssm_n_v_heads == 0 || cfg.ssm_n_k_heads == 0 ||
        cfg.ssm_state == 0 || cfg.ssm_conv_kernel == 0)
        return "qwen35split: missing ssm dims";

    // oneDNN stays DISABLED on the split path — but the DEVICE_LOST landmine is
    // now FIXED: gemm_onednn.cpp:ctx_for keeps one engine/stream PER device
    // (Step 1, ie-onednn-multidev-test), so a card-1 GEMM no longer routes its
    // USM through a card-0-bound stream. What's still missing is the MoE
    // batch-decouple (Step 2): at the T≤512 chunk the 80B MoE is small-M
    // (~rows/expert ≈ T·top_k/E), the regime where oneDNN loses to int-dot, so
    // enabling it here alone buys nothing. Decouple the MoE chunk from the
    // DeltaNet recurrence (as crown does) → large M → then flip this on.
    dense::prefer_onednn() = false;

    owned_.assign(n_dev_, {});
    dev_bytes_.assign(n_dev_, 0);

    char buf[64];
    auto Ttop = [&](const char* n) { return g.find_tensor(n); };
    auto Tl   = [&](uint32_t L, const char* n) {
        std::snprintf(buf, sizeof(buf), "blk.%u.%s", L, n); return g.find_tensor(buf);
    };
    std::string err;

    // Phase 2: build a SplitW from a GGUF matrix weight. Q8_0 → de-interleave AoS
    // block_q8_0 → SoA (int8 qs col-contiguous + raw fp16 d/32-block), kept PACKED
    // (no F16 doubling, bit-exact). Else (Q4_K/Q6_K packed, Q5_K→F16, F16) → `fp`
    // fallback via the existing auto-upload. `bytes` accumulates the chosen device
    // residency. SplitW is the model's nested type → this lambda lives in load().
    auto build_split = [](DeviceAllocator& a, const GgufTensorInfo* t,
                          std::vector<void*>& own, std::string& e, uint64_t& bytes) -> SplitW {
        SplitW w{};
        if (!t) { e = "tensor not found"; return w; }
        if (t->n_dims != 2) { e = "split weight: expected 2-D"; return w; }
        w.K = uint32_t(t->shape[0]); w.N = uint32_t(t->shape[1]);
        if (t->dtype == DType::kQ8_0) {
            const uint32_t K = w.K, N = w.N, bpc = K / 32;
            if (K % 32 != 0) { e = "Q8_0 SoA: K % 32 != 0"; return w; }
            const auto* blocks = reinterpret_cast<const block_q8_0*>(t->data);
            std::vector<int8_t>   qs(uint64_t(N) * K);
            std::vector<uint16_t> dd(uint64_t(N) * bpc);
            for (uint64_t n = 0; n < N; ++n)
                for (uint32_t b = 0; b < bpc; ++b) {
                    const block_q8_0& blk = blocks[n * bpc + b];
                    dd[n * bpc + b] = *reinterpret_cast<const uint16_t*>(&blk.d);
                    for (int i = 0; i < 32; ++i)
                        qs[n * K + uint64_t(b) * 32 + i] = blk.qs[i];
                }
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

    // token_embd → embed_dev (Q4_K/Q6_K/Q8_0 — the cyber model is Q8_0).
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
        // lm_head is always a single-vector GEMV (last token) → Q8_0-SoA int-dot
        // when packed. Untied = output.weight; tied = replicate token_embd onto
        // head_dev (a device ptr can't be read from another card).
        const auto* ti = Ttop("output.weight");
        if (!ti) ti = Ttop("token_embd.weight");
        output_ = build_split(ha, ti, owned_[plan.head_dev], err, dev_bytes_[plan.head_dev]);
        if (!err.empty()) return "output: " + err;
    }

    const uint32_t n_layers = cfg.n_transformer_layers();   // 64 (skip trailing NextN/MTP)
    layers_.assign(n_layers, {});
    dn_local_.assign(n_layers, 0);
    kv_local_.assign(n_layers, 0);

    for (uint32_t L = 0; L < n_layers; ++L) {
        const uint32_t dev = plan.dev_of_layer[L];
        DeviceAllocator& a = fleet.dev(dev);
        auto& own = owned_[dev];
        LayerW& w = layers_[L];
        w.is_linear = cfg.recurrent_layer(L);

        auto LW = [&](const char* n) -> SplitW {
            return build_split(a, Tl(L, n), own, err, dev_bytes_[dev]);
        };
        auto F32 = [&](const char* n, float*& dst) -> std::string {
            const auto* ti = Tl(L, n);
            dst = dense::upload<float>(a, ti, own, err, DType::kF32);
            if (!err.empty()) return std::string(n) + ": " + err;
            if (ti) dev_bytes_[dev] += ti->nbytes;
            return {};
        };
        auto layer_err = [&](const char* what) {
            return "layer " + std::to_string(L) + " " + what + ": " + err;
        };

        if (auto m = F32("attn_norm.weight", w.attn_norm); !m.empty())
            return "layer " + std::to_string(L) + " " + m;
        if (auto m = F32("post_attention_norm.weight", w.post_attn_norm); !m.empty())
            return "layer " + std::to_string(L) + " " + m;

        if (w.is_linear) {
            w.attn_qkv  = LW("attn_qkv.weight");  if (!err.empty()) return layer_err("attn_qkv");
            w.attn_gate = LW("attn_gate.weight"); if (!err.empty()) return layer_err("attn_gate");
            if (auto m = F32("ssm_a", w.ssm_a); !m.empty()) return "layer " + std::to_string(L) + " " + m;
            const uint32_t svh_pad = ((cfg.ssm_n_v_heads + 63u) / 64u) * 64u;   // 48 → 64
            w.ssm_alpha = upload_f32_proj_fp16(a, Tl(L, "ssm_alpha.weight"), own, err, svh_pad);
            if (!err.empty()) return layer_err("ssm_alpha");
            w.ssm_beta  = upload_f32_proj_fp16(a, Tl(L, "ssm_beta.weight"), own, err, svh_pad);
            if (!err.empty()) return layer_err("ssm_beta");
            if (auto m = F32("ssm_dt.bias", w.ssm_dt_bias); !m.empty()) return "layer " + std::to_string(L) + " " + m;
            {   // ssm_conv1d [4, conv_ch] → cast fp16
                const auto* ti = Tl(L, "ssm_conv1d.weight");
                if (auto m = F32("ssm_conv1d.weight", w.ssm_conv1d); !m.empty()) return "layer " + std::to_string(L) + " " + m;
                const uint64_t n = ti->nbytes / sizeof(float);
                w.ssm_conv1d_fp16 = static_cast<sycl::half*>(a.malloc(n * sizeof(sycl::half)));
                if (!w.ssm_conv1d_fp16) return layer_err("ssm_conv1d fp16 malloc");
                own.push_back(w.ssm_conv1d_fp16); dev_bytes_[dev] += n * sizeof(sycl::half);
                cast_fp32_to_fp16(a.queue(), w.ssm_conv1d, w.ssm_conv1d_fp16, n).wait();
            }
            {   // ssm_norm [v_head_dim] → cast fp16
                const auto* ti = Tl(L, "ssm_norm.weight");
                if (auto m = F32("ssm_norm.weight", w.ssm_norm); !m.empty()) return "layer " + std::to_string(L) + " " + m;
                const uint64_t n = ti->nbytes / sizeof(float);
                w.ssm_norm_fp16 = static_cast<sycl::half*>(a.malloc(n * sizeof(sycl::half)));
                if (!w.ssm_norm_fp16) return layer_err("ssm_norm fp16 malloc");
                own.push_back(w.ssm_norm_fp16); dev_bytes_[dev] += n * sizeof(sycl::half);
                cast_fp32_to_fp16(a.queue(), w.ssm_norm, w.ssm_norm_fp16, n).wait();
            }
            w.ssm_out = LW("ssm_out.weight");   if (!err.empty()) return layer_err("ssm_out");
        } else {
            w.attn_q      = LW("attn_q.weight");      if (!err.empty()) return layer_err("attn_q");
            w.attn_k      = LW("attn_k.weight");      if (!err.empty()) return layer_err("attn_k");
            w.attn_v      = LW("attn_v.weight");      if (!err.empty()) return layer_err("attn_v");
            w.attn_output = LW("attn_output.weight"); if (!err.empty()) return layer_err("attn_output");
            if (auto m = F32("attn_q_norm.weight", w.attn_q_norm); !m.empty()) return "layer " + std::to_string(L) + " " + m;
            if (auto m = F32("attn_k_norm.weight", w.attn_k_norm); !m.empty()) return "layer " + std::to_string(L) + " " + m;
        }

        w.ffn_gate = LW("ffn_gate.weight"); if (!err.empty()) return layer_err("ffn_gate");
        w.ffn_up   = LW("ffn_up.weight");   if (!err.empty()) return layer_err("ffn_up");
        w.ffn_down = LW("ffn_down.weight"); if (!err.empty()) return layer_err("ffn_down");
    }

    // Per-card hybrid caches: count each card's linear/full-attn layers, assign
    // card-local indices, init one DeltaNetState + one KvCache per card.
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
            dc.n_v_heads       = cfg.ssm_n_v_heads;       // 48
            dc.v_head_dim      = cfg.ssm_state;           // 128
            dc.k_head_dim      = cfg.ssm_state;           // 128
            dc.conv_channels   = cfg.ssm_inner + 2u * cfg.ssm_n_k_heads * cfg.ssm_state;  // 10240
            dc.conv_kernel     = cfg.ssm_conv_kernel;     // 4
            if (auto m = dn_[dev].init(fleet.dev(dev), dc); !m.empty())
                return "dn cache dev " + std::to_string(dev) + ": " + m;
        }
        if (n_full[dev]) {
            KvCacheConfig kc{};
            kc.n_layers_full = n_full[dev];
            kc.n_kv_heads    = d.n_kv_heads;              // 4
            kc.max_ctx       = max_ctx;
            kc.head_dim      = d.head_dim;                // 256
            if (auto m = kv_[dev].init(fleet.dev(dev), kc); !m.empty())
                return "kv cache dev " + std::to_string(dev) + ": " + m;
        }
    }

    // Phase-2 per-card scratch (allocated once; T-independent). act_q8 = the int-dot
    // decode activation (block_q8_1x [Kmax/32]); prefill_bt grows lazily in sgemv to
    // the largest weight. Kmax = max projection input dim across the model.
    const uint32_t Kmax = std::max(std::max(d.hidden, d.ffn),
                                   std::max(cfg.ssm_inner, d.n_q_heads * d.head_dim));
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
        std::fprintf(stderr, "[qwen35split] card %u weights: %.2f GB\n",
                     dev, double(dev_bytes_[dev]) / 1e9);
    return {};
}

void Qwen35SplitModel::free_ws(uint32_t dev) {
    if (!fleet_ || dev >= ws_.size()) return;
    Workspace& w = ws_[dev];
    auto& alloc = fleet_->dev(dev);
    for (void* p : {static_cast<void*>(w.x), static_cast<void*>(w.x_normed),
                    static_cast<void*>(w.attn_block), static_cast<void*>(w.positions),
                    static_cast<void*>(w.qg), static_cast<void*>(w.q),
                    static_cast<void*>(w.gate), static_cast<void*>(w.k),
                    static_cast<void*>(w.v), static_cast<void*>(w.attn_out),
                    static_cast<void*>(w.attn_partials),
                    static_cast<void*>(w.dn_qkv), static_cast<void*>(w.dn_conv),
                    static_cast<void*>(w.dn_z), static_cast<void*>(w.dn_qpre),
                    static_cast<void*>(w.dn_kpre), static_cast<void*>(w.dn_vpre),
                    static_cast<void*>(w.dn_g), static_cast<void*>(w.dn_beta),
                    static_cast<void*>(w.dn_out), static_cast<void*>(w.dn_qrep),
                    static_cast<void*>(w.dn_krep), static_cast<void*>(w.dn_alpha_h),
                    static_cast<void*>(w.dn_beta_h), static_cast<void*>(w.dn_alpha64),
                    static_cast<void*>(w.dn_beta64),
                    static_cast<void*>(w.ffn_gate), static_cast<void*>(w.ffn_up),
                    static_cast<void*>(w.ffn_h)})
        if (p) alloc.free(p);
    w = Workspace{};
}

std::string Qwen35SplitModel::ensure_ws(uint32_t dev, uint32_t max_T) {
    Workspace& w = ws_[dev];
    if (max_T <= w.T) return {};
    if (w.T != 0) free_ws(dev);
    auto& alloc = fleet_->dev(dev);
    auto ah = [&](size_t n) { return static_cast<sycl::half*>(alloc.malloc(n * sizeof(sycl::half))); };
    auto af = [&](size_t n) { return static_cast<float*>(alloc.malloc(n * sizeof(float))); };

    const DenseConfig& d = cfg_.dense;
    const uint64_t T   = max_T;
    const uint32_t H   = d.hidden;                          // 5120
    const uint32_t N_q = d.n_q_heads * d.head_dim;          // 6144
    const uint32_t N_qg = N_q * 2u;                         // 12288
    const uint32_t N_kv = d.n_kv_heads * d.head_dim;        // 1024
    const uint32_t F   = d.ffn;                             // 17408
    const uint32_t SI  = cfg_.ssm_inner;                    // 6144
    const uint32_t cvc = SI + 2u * cfg_.ssm_n_k_heads * cfg_.ssm_state;  // 10240
    const uint32_t Vd  = cfg_.ssm_n_v_heads * cfg_.ssm_state;            // 6144
    const uint32_t Nv  = cfg_.ssm_n_v_heads;               // 48
    const uint32_t Nvp = ((Nv + 63u) / 64u) * 64u;         // 64

    w.x          = ah(T * H);
    w.x_normed   = ah(T * H);
    w.attn_block = ah(T * H);
    w.positions  = static_cast<int32_t*>(alloc.malloc(T * sizeof(int32_t)));
    w.qg         = ah(T * N_qg);
    w.q          = ah(T * N_q);
    w.gate       = ah(T * N_q);
    w.k          = ah(T * N_kv);
    w.v          = ah(T * N_kv);
    w.attn_out   = ah(T * N_q);
    w.dn_qkv     = ah(T * cvc);
    w.dn_conv    = ah(T * cvc);
    w.dn_z       = ah(T * SI);
    w.dn_qpre    = af(T * Vd);
    w.dn_kpre    = af(T * Vd);
    w.dn_vpre    = af(T * Vd);
    w.dn_g       = af(T * Nv);
    w.dn_beta    = af(T * Nv);
    w.dn_out     = af(T * Vd);
    w.dn_qrep    = af(T * Vd);
    w.dn_krep    = af(T * Vd);
    w.dn_alpha_h = ah(T * Nv);
    w.dn_beta_h  = ah(T * Nv);
    w.dn_alpha64 = ah(T * Nvp);
    w.dn_beta64  = ah(T * Nvp);
    w.ffn_gate   = ah(T * F);
    w.ffn_up     = ah(T * F);
    w.ffn_h      = ah(T * F);

    if (!w.x || !w.x_normed || !w.attn_block || !w.positions || !w.qg || !w.q ||
        !w.gate || !w.k || !w.v || !w.attn_out || !w.dn_qkv || !w.dn_conv ||
        !w.dn_z || !w.dn_qpre || !w.dn_kpre || !w.dn_vpre || !w.dn_g || !w.dn_beta ||
        !w.dn_out || !w.dn_qrep || !w.dn_krep || !w.dn_alpha_h || !w.dn_beta_h ||
        !w.dn_alpha64 || !w.dn_beta64 || !w.ffn_gate || !w.ffn_up || !w.ffn_h)
        return "qwen35split workspace alloc failed on dev " + std::to_string(dev);

    // FA-2 decode partials for cards holding ≥1 full-attn layer (have a KV cache).
    if (kv_[dev].ready()) {
        const uint32_t max_ctx = kv_[dev].config().max_ctx;
        constexpr uint32_t Bc_floor = 64;
        const uint32_t n_chunks_max = (max_ctx + Bc_floor - 1) / Bc_floor;
        const uint64_t n_floats = uint64_t(n_chunks_max) * d.n_q_heads * (d.head_dim + 2);
        w.attn_partials = static_cast<float*>(alloc.malloc(n_floats * sizeof(float)));
        if (!w.attn_partials)
            return "qwen35split attn_partials alloc failed on dev " + std::to_string(dev);
        w.partials_ctx = max_ctx;
    }
    w.T = max_T;
    return {};
}

// Q8_0-SoA aware GEMV. Decode (T==1): quantize activation → int-dot W8A8 over the
// packed int8 weight (≈½ the F16 bandwidth → decode win). Prefill (T>1): dequant
// the SoA weight → fp16 scratch (grown to the largest weight, reused) → gemm_fp16.
// Non-Q8_0 weights (Q4_K/Q6_K packed, Q5_K/F16) → the existing dense path.
sycl::event Qwen35SplitModel::sgemv(uint32_t dev, const sycl::half* A, const SplitW& w,
                                    sycl::half* out, uint32_t K, uint32_t N, uint32_t T) {
    auto& alloc = fleet_->dev(dev);
    auto& q = alloc.queue();
    if (!w.q8_qs)                                   // non-Q8_0 fallback
        return dense::gemv_q_T(q, A, w.fp, out, K, N, T);
    if (T == 1) {                                   // decode: int-dot W8A8
        quantize_q8_1(q, A, act_q8_[dev], K);
        // SLM-staged activation is the DEFAULT and the faster path: the 32 columns
        // per WG share ONE staged copy of the (reused) activation. The no-SLM `_g`
        // variant (IE_QWEN35_SOA_GMEM=1) was A/B-tested SLOWER (~10×) — each column
        // re-reads the activation from L2, and that traffic outweighs the occupancy
        // gain. Kept opt-in only so the negative result isn't re-discovered.
        static const bool soa_gmem = std::getenv("IE_QWEN35_SOA_GMEM") != nullptr;
        if (soa_gmem)
            return gemv_q8_0_soa_q8_g(q, act_q8_[dev], w.q8_qs, w.q8_d, out, K, N);
        return gemv_q8_0_soa_q8(q, act_q8_[dev], w.q8_qs, w.q8_d, out, K, N);
    }
    // prefill: dequant SoA → fp16 scratch, then the batched gemm via gemv_q_T(F16).
    const uint64_t need = uint64_t(K) * N;
    if (need > prefill_bt_cap_[dev]) {
        if (prefill_bt_[dev]) alloc.free(prefill_bt_[dev]);
        prefill_bt_[dev] = static_cast<sycl::half*>(alloc.malloc(need * sizeof(sycl::half)));
        prefill_bt_cap_[dev] = prefill_bt_[dev] ? need : 0;
    }
    if (!prefill_bt_[dev]) { std::fprintf(stderr, "qwen35split: prefill_bt alloc failed\n"); return {}; }
    dequant_q8_0_soa_to_Bt(q, w.q8_qs, w.q8_d, prefill_bt_[dev], K, N);
    return dense::gemv_q_T(q, A, DenseQuantPtr{prefill_bt_[dev], DType::kF16}, out, K, N, T);
}

std::string Qwen35SplitModel::forward(const int32_t* input_ids, uint32_t T,
                                      uint32_t start_pos, bool reset_kv,
                                      sycl::half* out_logits_host) {
    if (T == 0) return "T == 0";
    const DenseConfig& dc = cfg_.dense;
    const uint32_t H    = dc.hidden;                    // 5120
    const uint32_t HD   = dc.head_dim;                  // 256
    const uint32_t N_q  = dc.n_q_heads  * HD;           // 6144
    const uint32_t N_qg = N_q * 2u;                     // 12288
    const uint32_t N_kv = dc.n_kv_heads * HD;           // 1024
    const uint32_t V    = dc.vocab;
    const uint32_t F    = dc.ffn;                       // 17408
    const uint32_t rope_n = dc.rope_dim;                // 64 (partial)
    const float    eps  = dc.rms_eps;
    const uint32_t n_layers = cfg_.n_transformer_layers();
    const uint32_t n_kv = dc.n_kv_heads;

    // env-gated decode phase profiler (IE_QWEN35_PROFILE=1). Wait-bracketed →
    // absolute total runs slower than prod; read it as a RELATIVE breakdown.
    static const bool prof = std::getenv("IE_QWEN35_PROFILE") != nullptr;
    using pclk = std::chrono::steady_clock;
    static double pf_dn = 0, pf_attn = 0, pf_ffn = 0, pf_head = 0, pf_other = 0, pf_total = 0;
    static uint64_t pf_tok = 0, pf_calls = 0;
    auto pms = [](pclk::time_point a, pclk::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    pclk::time_point pf_t0 = pclk::now();
    pclk::time_point pf_last;
    auto mark = [&](sycl::queue& qq, double& acc) {
        if (!prof) return;
        qq.wait();
        auto now = pclk::now();
        acc += pms(pf_last, now);
        pf_last = now;
    };

    if (ws_.size() != n_dev_) ws_.assign(n_dev_, {});
    for (uint32_t dev = 0; dev < n_dev_; ++dev)
        if (auto m = ensure_ws(dev, T); !m.empty()) return m;

    // positions on every card; reset per-card hybrid state on a fresh sequence.
    std::vector<int32_t> pos(T);
    for (uint32_t t = 0; t < T; ++t) pos[t] = int32_t(start_pos + t);
    for (uint32_t dev = 0; dev < n_dev_; ++dev) {
        if (ws_[dev].positions)
            fleet_->dev(dev).queue().memcpy(ws_[dev].positions, pos.data(),
                                            T * sizeof(int32_t)).wait();
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

    // device-by-device: copy residual in from prior card, run this card's layers.
    for (uint32_t dev = 0; dev < n_dev_; ++dev) {
        Workspace& w = ws_[dev];
        auto& q = fleet_->dev(dev).queue();
        if (dev > 0)   // residual hand-off at the card boundary (ONE copy)
            fleet_->copy_across(dev - 1, w.x, dev, ws_[dev - 1].x,
                                uint64_t(T) * H * sizeof(sycl::half));
        const uint64_t per_layer_kv =
            uint64_t(n_kv) * (kv_[dev].ready() ? kv_[dev].config().max_ctx : 0u) * HD;
        if (prof) { q.wait(); pf_last = pclk::now(); }

        for (uint32_t L = 0; L < n_layers; ++L) {
            if (plan_.dev_of_layer[L] != dev) continue;
            const LayerW& w_l = layers_[L];
            // pre-attn norm
            rms_norm_f32w(q, w.x, w_l.attn_norm, w.x_normed, T, H, eps);

            if (w_l.is_linear) {
                // ---- gated-DeltaNet (27B conventions: separate ssm_alpha/beta,
                // tile repeat 16→48). Lifted verbatim from qwen35_dense.cpp. ----
                const uint32_t SKH = cfg_.ssm_n_k_heads;          // 16
                const uint32_t SVH = cfg_.ssm_n_v_heads;          // 48
                const uint32_t SHD = cfg_.ssm_state;              // 128
                const uint32_t SI  = cfg_.ssm_inner;              // 6144
                const uint32_t conv_ch = SI + 2u * SKH * SHD;     // 10240
                const uint32_t kw  = SKH * SHD;                   // 2048
                const uint32_t rep = SVH / SKH;                   // 3 (16→48)
                const float qscale = 1.0f / sycl::sqrt(float(SHD));

                sgemv(dev, w.x_normed, w_l.attn_qkv, w.dn_qkv, H, conv_ch, T);
                sycl::half* conv_state = dn_[dev].conv_state_ptr() +
                    uint64_t(dn_local_[L]) * dn_[dev].conv_elems_per_layer();
                depthwise_conv1d_causal(q, w.dn_qkv, w_l.ssm_conv1d_fp16, conv_state,
                                        w.dn_conv, T, conv_ch, cfg_.ssm_conv_kernel);
                cast_qkv_split_fp16_to_fp32(q, w.dn_conv, w.dn_qpre, w.dn_kpre, w.dn_vpre, T, kw, SI);
                l2_norm_scale(q, w.dn_qpre, w.dn_qpre, T * SKH, SHD, qscale, 1e-6f);
                l2_norm_scale(q, w.dn_kpre, w.dn_kpre, T * SKH, SHD, 1.0f,   1e-6f);
                // TILE repeat 16→48 (interleave=false, the VALIDATED 27B convention)
                repeat_interleave_heads(q, w.dn_qpre, w.dn_qrep, T, SKH, SHD, rep);
                repeat_interleave_heads(q, w.dn_kpre, w.dn_krep, T, SKH, SHD, rep);
                // separate N-padded ssm_alpha/ssm_beta projections → batched gemm → compact
                const uint32_t SVHp = ((SVH + 63u) / 64u) * 64u;   // 64
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
                // gated RMS-norm with z = attn_gate · x_normed (reuse dn_qkv as out)
                sgemv(dev, w.x_normed, w_l.attn_gate, w.dn_z, H, SI, T);
                gated_rms_norm(q, w.dn_out, w.dn_z, w_l.ssm_norm_fp16, w.dn_qkv,
                               T * SVH, SHD, eps);
                sgemv(dev, w.dn_qkv, w_l.ssm_out, w.attn_block, SI, H, T);
            } else {
                // ---- gated full-attention (lifted verbatim from qwen35_dense.cpp) ----
                sgemv(dev, w.x_normed, w_l.attn_q, w.qg, H, N_qg, T);
                split_q_gate_per_head(q, w.qg, w.q, w.gate, T, dc.n_q_heads, HD);
                sgemv(dev, w.x_normed, w_l.attn_k, w.k, H, N_kv, T);
                sgemv(dev, w.x_normed, w_l.attn_v, w.v, H, N_kv, T);
                rms_norm_f32w(q, w.q, w_l.attn_q_norm, w.q, T * dc.n_q_heads,  HD, eps);
                rms_norm_f32w(q, w.k, w_l.attn_k_norm, w.k, T * dc.n_kv_heads, HD, eps);
                rope_partial(q, w.q, w.positions, w.q, T, dc.n_q_heads,  HD, rope_n, dc.rope_theta);
                rope_partial(q, w.k, w.positions, w.k, T, dc.n_kv_heads, HD, rope_n, dc.rope_theta);
                sycl::half* kc = kv_[dev].k_ptr() + per_layer_kv * kv_local_[L];
                sycl::half* vc = kv_[dev].v_ptr() + per_layer_kv * kv_local_[L];
                const uint32_t max_ctx = kv_[dev].config().max_ctx;
                if (T == 1 && w.attn_partials) {
                    full_attention_fa2_decode(q, w.q, w.k, w.v, kc, vc, w.attn_out,
                                              w.attn_partials, start_pos,
                                              dc.n_q_heads, dc.n_kv_heads, HD, max_ctx);
                } else {
                    full_attention(q, w.q, w.k, w.v, kc, vc, w.attn_out, T, start_pos,
                                   dc.n_q_heads, dc.n_kv_heads, HD, max_ctx);
                }
                kv_[dev].set_length(kv_local_[L], start_pos + T);
                sigmoid_gate(q, w.attn_out, w.gate, w.attn_out, uint64_t(T) * N_q);
                sgemv(dev, w.attn_out, w_l.attn_output, w.attn_block, N_q, H, T);
            }
            mark(q, w_l.is_linear ? pf_dn : pf_attn);   // pre-norm + DeltaNet/attn block

            // residual + pre-FFN (post-attention) norm (27B fused order) → dense SwiGLU
            residual_add_rms_norm_fused(q, w.x, w.attn_block, w_l.post_attn_norm,
                                        w.x_normed, T, H, eps);
            sgemv(dev, w.x_normed, w_l.ffn_gate, w.ffn_gate, H, F, T);
            sgemv(dev, w.x_normed, w_l.ffn_up,   w.ffn_up,   H, F, T);
            swiglu(q, w.ffn_gate, w.ffn_up, w.ffn_h, uint64_t(T) * F);
            sgemv(dev, w.ffn_h, w_l.ffn_down, w.attn_block, F, H, T);
            residual_add(q, w.x, w.attn_block, w.x, uint64_t(T) * H);
            mark(q, pf_ffn);   // post-norm + dense SwiGLU FFN
        }
        q.wait();   // finish this card before the boundary copy reads its ws.x
    }

    // final norm + lm_head on head_dev → last token's logits → host.
    {
        const uint32_t hd = plan_.head_dev;
        Workspace& w = ws_[hd];
        auto& alloc = fleet_->dev(hd);
        auto& q = alloc.queue();
        if (prof) { q.wait(); pf_last = pclk::now(); }
        rms_norm_f32w(q, w.x, output_norm_, w.x_normed, T, H, eps);
        const sycl::half* last = w.x_normed + uint64_t(T - 1) * H;
        sycl::half* d_logits = static_cast<sycl::half*>(alloc.malloc(uint64_t(V) * sizeof(sycl::half)));
        if (!d_logits) return "logits alloc failed";
        // lm_head is a single-vector GEMV → sgemv with T=1 (Q8_0-SoA int-dot when packed).
        sgemv(hd, last, output_, d_logits, H, V, 1).wait();
        q.memcpy(out_logits_host, d_logits, uint64_t(V) * sizeof(sycl::half)).wait();
        alloc.free(d_logits);
        mark(q, pf_head);   // final norm + lm_head + logits bounce
    }

    if (prof) {
        pf_total += pms(pf_t0, pclk::now());
        pf_tok += T; pf_calls++;
        const double summed = pf_dn + pf_attn + pf_ffn + pf_head;
        pf_other = pf_total - summed;
        if (pf_calls % 32 == 0) {
            auto pct = [&](double x) { return pf_total > 0 ? 100.0 * x / pf_total : 0.0; };
            std::fprintf(stderr,
                "\n[qwen35split PROFILE] calls=%llu tok=%llu (wait-bracketed; relative)\n"
                "  total       %8.2f ms  (%.2f ms/tok = %.1f tok/s instrumented)\n"
                "  DeltaNet    %8.2f ms  %5.1f%%\n"
                "  full-attn   %8.2f ms  %5.1f%%\n"
                "  dense FFN   %8.2f ms  %5.1f%%\n"
                "  head+bounce %8.2f ms  %5.1f%%\n"
                "  other       %8.2f ms  %5.1f%%\n",
                (unsigned long long)pf_calls, (unsigned long long)pf_tok,
                pf_total, pf_total / double(pf_tok), 1000.0 * double(pf_tok) / pf_total,
                pf_dn, pct(pf_dn), pf_attn, pct(pf_attn), pf_ffn, pct(pf_ffn),
                pf_head, pct(pf_head), pf_other, pct(pf_other));
        }
    }
    return {};
}

}  // namespace ie
