// src/model/qwen3next.cpp — see header. Step 1: loader. Step 2.1 (F1): forward scaffold.
#include "ie/qwen3next.hpp"

#include "dense_dispatch.hpp"   // dense::upload<T>, dense::gemv_q/gemv_q_T (lives here)
#include "ie/ops.hpp"           // cast_fp32_to_fp16, rms_norm_f32w, residual_add, embedding_lookup_*
#include "ie/moe_qwen3.hpp"     // moe_prefill_down_q{4,6}k_q8_gen (E_ffn%256 int-dot down)
#include "ie/dequant.hpp"       // dequant_q{4,6}_K_to_Bt (per-expert fp16 Bt for oneDNN MoE)
#include "ie/qwen3moe_pack.hpp" // MoePacking + build_moe_packing (host counting-sort)
#include "ie/quant_blocks.hpp"  // block_q8_1s (int-dot activation stream)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

namespace ie {
namespace {

// File-local sibling of qwen35_dense.cpp:79 / qwen3moe.cpp:58 upload_weight_auto
// (those are anonymous-namespace and not visible across TUs; the header comment
// in qwen3next.hpp is aspirational). Identical behavior: Q5_K/Q8_0 → dequant to
// device F16 [K,N]; Q4_K/Q6_K/F16 ride upload_quant_dense (or hard-fail). Both
// land in dense_dispatch.hpp (upload_quant_dense_auto), so this just forwards.
inline DenseQuantPtr upload_weight_auto(DeviceAllocator& alloc,
                                        const GgufTensorInfo* t,
                                        std::vector<void*>& owned,
                                        std::string& err) {
    return dense::upload_quant_dense_auto(alloc, t, owned, err);
}

// De-interleave a fused in_proj_qkvz projection (EXL3-only — the qkvz trellis is one
// tensor, so the split is on the activation). The raw HF output is grouped per K-head:
//   [head h: q(qd) | k(qd) | v(vd) | z(vd)] for h in 0..nkh,   group = 2*qd + 2*vd.
// Produce dn_qkv = [q_all(nkh*qd) | k_all(nkh*qd) | v_all(nkh*vd)] (= conv input, the
// contiguous order llama.cpp's attn_qkv emits, which cast_qkv_split + the conv weight
// expect) and dn_z = z_all(nkh*vd) (v-head order, matching the recurrence output).
// qd = key_head_dim (=SHD); vd = (n_v/n_k)*value_head_dim (=rep*SHD). In-order queue →
// completes before the conv reads dn_qkv. Mirror llama.cpp qwen.py:320-331.
inline void slice_qkvz(sycl::queue& q, const sycl::half* qkvz, sycl::half* qkv,
                       sycl::half* z, uint32_t T, uint32_t nkh, uint32_t qd, uint32_t vd) {
    const uint32_t group = 2u * qd + 2u * vd;        // per-k-head [q|k|v|z]
    const uint32_t qkv_w = nkh * (2u * qd + vd);     // [q_all|k_all|v_all] = conv_ch
    const uint32_t W     = nkh * group;              // 12288
    const uint32_t qoff  = nkh * qd;                 // k_all starts here
    const uint32_t voff  = 2u * nkh * qd;            // v_all starts here
    const uint32_t zw    = nkh * vd;                 // z_all width
    q.parallel_for(sycl::range<2>(T, nkh), [=](sycl::id<2> id) {
        const uint32_t t = uint32_t(id[0]), h = uint32_t(id[1]);
        const sycl::half* s = qkvz + uint64_t(t) * W + uint64_t(h) * group;  // this head's chunk
        sycl::half* qr = qkv + uint64_t(t) * qkv_w;
        sycl::half* zr = z   + uint64_t(t) * zw;
        for (uint32_t i = 0; i < qd; ++i) qr[h * qd + i]        = s[i];                 // q→q_all
        for (uint32_t i = 0; i < qd; ++i) qr[qoff + h * qd + i] = s[qd + i];            // k→k_all
        for (uint32_t i = 0; i < vd; ++i) qr[voff + h * vd + i] = s[2u * qd + i];       // v→v_all
        for (uint32_t i = 0; i < vd; ++i) zr[h * vd + i]        = s[2u * qd + vd + i];  // z→z_all
    });
}

// Host top-k softmax router for one token's precomputed logits[E] — bit-for-bit
// copy of qwen3moe.cpp's route_from_logits (copy-not-hoist; that one is in an
// anonymous namespace in another TU): softmax over all E, take top-k by prob,
// renormalize the k to sum 1, sort ascending by expert index (HF norm_topk_prob
// + HF iteration order). The GPU-gemm router feeds these logits.
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

Qwen3NextModel::~Qwen3NextModel() { free_all(); }

void Qwen3NextModel::free_all() {
    if (!fleet_) return;
    // Workspace buffers live outside owned_ (so ensure_ws can re-alloc on grow
    // without leaking); free them per card here.
    for (uint32_t d = 0; d < ws_.size(); ++d) free_ws(d);
    ws_.clear();
    for (uint32_t d = 0; d < owned_.size(); ++d)
        for (void* p : owned_[d]) if (p) fleet_->dev(d).free(p);
    owned_.clear();
    // DeltaNetState / KvCache teardown is named free_storage() (destructor-safe,
    // idempotent); the no-arg free() the stub sketched does not exist on these.
    for (auto& s : dn_) s.free_storage();
    for (auto& c : kv_) c.free_storage();
}

std::string Qwen3NextModel::load(DeviceFleet& fleet, const LayerPlan& plan,
                                 const GgufReader& g, const Qwen3NextConfig& cfg,
                                 uint32_t max_ctx) {
    fleet_ = &fleet; plan_ = plan; cfg_ = cfg;
    n_dev_ = plan.n_dev();
    const Qwen35Config& h = cfg.hybrid;     // hybrid attn dims + DeltaNet ssm
    const DenseConfig&  d = h.dense;

    if (n_dev_ == 0 || plan.dev_of_layer.empty()) return "qwen3next: empty plan";
    if (d.hidden == 0 || d.vocab == 0)            return "qwen3next: zero dim";
    if (cfg.n_experts == 0 || cfg.expert_ffn == 0) return "qwen3next: missing MoE dims";

    owned_.assign(n_dev_, {});
    dev_bytes_.assign(n_dev_, 0);

    char buf[64];
    auto Ttop = [&](const char* n) { return g.find_tensor(n); };
    auto Tl   = [&](uint32_t L, const char* n) {
        std::snprintf(buf, sizeof(buf), "blk.%u.%s", L, n); return g.find_tensor(buf);
    };
    std::string err;

    // token_embd → embed_dev; output_norm + lm_head → head_dev. (mirror
    // qwen35_dense.cpp:173-199, but on the chosen card's allocator.)
    {
        DeviceAllocator& ea = fleet.dev(plan.embed_dev);
        const auto* ti = Ttop("token_embd.weight");
        if (!ti) return "token_embd: not found";
        if (ti->dtype != DType::kQ4_K && ti->dtype != DType::kQ6_K && ti->dtype != DType::kF16)
            return "token_embd: dtype";   // EXL3 GGUF keeps token_embd plain F16
        token_embd_dtype_ = ti->dtype;
        token_embd_ = dense::upload<void>(ea, ti, owned_[plan.embed_dev], err, ti->dtype);
        if (!err.empty()) return "token_embd: " + err;
        dev_bytes_[plan.embed_dev] += ti->nbytes;
    }
    {
        DeviceAllocator& ha = fleet.dev(plan.head_dev);
        output_norm_ = dense::upload<float>(ha, Ttop("output_norm.weight"),
                                            owned_[plan.head_dev], err, DType::kF32);
        if (!err.empty()) return "output_norm: " + err;
        const auto* ti = Ttop("output.weight");
        if (!ti) {  // tied lm_head
            if (plan.head_dev == plan.embed_dev) {
                // same card → alias the embedding (no extra alloc), as qwen35_dense.cpp:190 does
                output_ = token_embd_; output_dtype_ = token_embd_dtype_;
            } else {
                // cross-card: the lm_head GEMV runs on head_dev but token_embd lives on
                // embed_dev — a device pointer can't be read from another card, so replicate.
                const auto* te = Ttop("token_embd.weight");
                output_dtype_ = te->dtype;
                output_ = dense::upload<void>(ha, te, owned_[plan.head_dev], err, te->dtype);
                if (!err.empty()) return "output(tied): " + err;
                dev_bytes_[plan.head_dev] += te->nbytes;
            }
        } else if (ti->dtype == DType::kEXL3) {  // EXL3 lm_head: trellis + suh/svh, decoded by gemv_q
            DenseQuantPtr o = dense::upload_exl3(ha, ti, Ttop("output.suh"), Ttop("output.svh"),
                                                 owned_[plan.head_dev], err);
            if (!err.empty()) return "output: " + err;
            output_ = o.p; output_dtype_ = DType::kEXL3;
            output_suh_ = o.suh; output_svh_ = o.svh; output_bits_ = o.bits;
            dev_bytes_[plan.head_dev] += ti->nbytes;
        } else {
            if (ti->dtype != DType::kQ4_K && ti->dtype != DType::kQ6_K) return "output: dtype";
            output_dtype_ = ti->dtype;
            output_ = dense::upload<void>(ha, ti, owned_[plan.head_dev], err, ti->dtype);
            if (!err.empty()) return "output: " + err;
            dev_bytes_[plan.head_dev] += ti->nbytes;
        }
    }

    const uint32_t n_layers = h.n_transformer_layers();   // 48 (no NextN/MTP)
    layers_.assign(n_layers, {});
    dn_local_.assign(n_layers, 0);
    kv_local_.assign(n_layers, 0);

    for (uint32_t L = 0; L < n_layers; ++L) {
        const uint32_t dev = plan.dev_of_layer[L];
        DeviceAllocator& a = fleet.dev(dev);
        auto& own = owned_[dev];
        LayerW& w = layers_[L];
        w.is_linear = h.recurrent_layer(L);

        auto LQ  = [&](const char* n) { return upload_weight_auto(a, Tl(L, n), own, err); };
        // EXL3-aware linear: a kEXL3 ".weight" uploads its trellis + .suh/.svh siblings via
        // upload_exl3 (decoded natively by gemv_q_T's kEXL3 branch); otherwise the quant/F16 path.
        auto LX  = [&](const char* n) -> DenseQuantPtr {
            const auto* ti = Tl(L, n);
            if (ti && ti->dtype == DType::kEXL3) {
                std::string root(n); root.resize(root.size() - 7);   // strip ".weight"
                return dense::upload_exl3(a, ti, Tl(L, (root + ".suh").c_str()),
                                          Tl(L, (root + ".svh").c_str()), own, err);
            }
            return upload_weight_auto(a, ti, own, err);
        };
        auto F32 = [&](const char* n, float*& dst) -> std::string {
            const auto* ti = Tl(L, n);
            dst = dense::upload<float>(a, ti, own, err, DType::kF32);
            if (!err.empty()) return std::string(n) + ": " + err;
            if (ti) dev_bytes_[dev] += ti->nbytes;
            return {};
        };
        auto byteq = [&](const char* n) {  // track bytes for a quant upload
            const auto* ti = Tl(L, n); if (ti) dev_bytes_[dev] += ti->nbytes;
        };

        if (auto m = F32("attn_norm.weight", w.attn_norm); !m.empty())
            return "layer " + std::to_string(L) + " " + m;
        if (auto m = F32("post_attention_norm.weight", w.post_attn_norm); !m.empty())
            return "layer " + std::to_string(L) + " " + m;

        if (w.is_linear) {   // gated DeltaNet
            w.attn_qkv  = LX("attn_qkv.weight");  byteq("attn_qkv.weight");  if (!err.empty()) return "layer " + std::to_string(L) + " attn_qkv: " + err;
            // attn_gate is absent on the EXL3 path: in_proj_qkvz is one fused trellis (N=12288), so
            // the z-gate is sliced from the qkvz projection in the forward (a trellis can't be split).
            if (Tl(L, "attn_gate.weight")) {
                w.attn_gate = LX("attn_gate.weight"); byteq("attn_gate.weight"); if (!err.empty()) return "layer " + std::to_string(L) + " attn_gate: " + err;
            }
            w.ssm_ba    = LQ("ssm_ba.weight");    byteq("ssm_ba.weight");    if (!err.empty()) return "layer " + std::to_string(L) + " ssm_ba: " + err;
            w.ssm_out   = LX("ssm_out.weight");   byteq("ssm_out.weight");   if (!err.empty()) return "layer " + std::to_string(L) + " ssm_out: " + err;
            if (auto m = F32("ssm_a", w.ssm_a); !m.empty())             return "layer " + std::to_string(L) + " " + m;
            if (auto m = F32("ssm_dt.bias", w.ssm_dt_bias); !m.empty()) return "layer " + std::to_string(L) + " " + m;
            // ssm_conv1d [4, 8192] + ssm_norm [128] → also cast to fp16 (mirror qwen35_dense.cpp:235-252)
            {
                const auto* ti = Tl(L, "ssm_conv1d.weight");
                if (auto m = F32("ssm_conv1d.weight", w.ssm_conv1d); !m.empty()) return "layer " + std::to_string(L) + " " + m;
                const uint64_t n = ti->nbytes / sizeof(float);
                w.ssm_conv1d_fp16 = static_cast<sycl::half*>(a.malloc(n * sizeof(sycl::half)));
                if (!w.ssm_conv1d_fp16) return "layer " + std::to_string(L) + " ssm_conv1d fp16 malloc";
                own.push_back(w.ssm_conv1d_fp16); dev_bytes_[dev] += n * sizeof(sycl::half);
                cast_fp32_to_fp16(a.queue(), w.ssm_conv1d, w.ssm_conv1d_fp16, n).wait();
            }
            {
                const auto* ti = Tl(L, "ssm_norm.weight");
                if (auto m = F32("ssm_norm.weight", w.ssm_norm); !m.empty()) return "layer " + std::to_string(L) + " " + m;
                const uint64_t n = ti->nbytes / sizeof(float);
                w.ssm_norm_fp16 = static_cast<sycl::half*>(a.malloc(n * sizeof(sycl::half)));
                if (!w.ssm_norm_fp16) return "layer " + std::to_string(L) + " ssm_norm fp16 malloc";
                own.push_back(w.ssm_norm_fp16); dev_bytes_[dev] += n * sizeof(sycl::half);
                cast_fp32_to_fp16(a.queue(), w.ssm_norm, w.ssm_norm_fp16, n).wait();
            }
        } else {             // gated full-attn
            w.attn_q = LX("attn_q.weight"); byteq("attn_q.weight"); if (!err.empty()) return "layer " + std::to_string(L) + " attn_q: " + err;
            w.attn_k = LX("attn_k.weight"); byteq("attn_k.weight"); if (!err.empty()) return "layer " + std::to_string(L) + " attn_k: " + err;
            w.attn_v = LX("attn_v.weight"); byteq("attn_v.weight"); if (!err.empty()) return "layer " + std::to_string(L) + " attn_v: " + err;
            w.attn_output = LX("attn_output.weight"); byteq("attn_output.weight"); if (!err.empty()) return "layer " + std::to_string(L) + " attn_output: " + err;
            if (auto m = F32("attn_q_norm.weight", w.attn_q_norm); !m.empty()) return "layer " + std::to_string(L) + " " + m;
            if (auto m = F32("attn_k_norm.weight", w.attn_k_norm); !m.empty()) return "layer " + std::to_string(L) + " " + m;
        }

        const uint32_t E = cfg.n_experts;
        // router: host F32 [E,H] + transposed device F16 [H,E] (GPU-gemm router lever)
        {
            const auto* ti = Tl(L, "ffn_gate_inp.weight");
            if (!ti) return "layer " + std::to_string(L) + " ffn_gate_inp: not found";
            if (ti->dtype != DType::kF32) return "layer " + std::to_string(L) + " ffn_gate_inp: expected F32";
            if (ti->nbytes != uint64_t(E) * d.hidden * sizeof(float))
                return "layer " + std::to_string(L) + " ffn_gate_inp: size";
            w.router_w.resize(uint64_t(E) * d.hidden);
            std::memcpy(w.router_w.data(), ti->data, ti->nbytes);
            const uint32_t Hn = d.hidden;
            std::vector<sycl::half> rt(uint64_t(Hn) * E);
            for (uint32_t e = 0; e < E; ++e)
                for (uint32_t hh = 0; hh < Hn; ++hh)
                    rt[uint64_t(hh) * E + e] = sycl::half(w.router_w[uint64_t(e) * Hn + hh]);
            auto* rd = static_cast<sycl::half*>(a.malloc(uint64_t(Hn) * E * sizeof(sycl::half)));
            if (!rd) return "layer " + std::to_string(L) + " router_w_dev alloc";
            a.queue().memcpy(rd, rt.data(), uint64_t(Hn) * E * sizeof(sycl::half)).wait();
            own.push_back(rd); w.router_w_dev = rd; dev_bytes_[dev] += uint64_t(Hn) * E * sizeof(sycl::half);
        }
        // stacked experts: device raw, per-expert byte stride = nbytes/E (mirror qwen3moe.cpp:255-269).
        // gate/up Q4_K + down Q6_K stay PACKED on device (no dequant), so ti->nbytes is the true residency.
        // raw verbatim device upload (bytes as-is) — for the EXL3 trellis/suh/svh expert banks.
        auto raw_up = [&](const GgufTensorInfo* ti) -> void* {
            void* d = a.malloc(ti->nbytes);
            if (!d) { err = "expert bank malloc"; return nullptr; }
            a.queue().memcpy(d, ti->data, ti->nbytes).wait();
            own.push_back(d);
            return d;
        };
        auto up_exps = [&](const char* nm, void*& dst, DType& dt, uint64_t& stride,
                           void*& suh, void*& svh, uint32_t& bits) -> std::string {
            const auto* ti = Tl(L, nm);
            if (!ti) return std::string(nm) + ": not found";
            if (ti->dtype == DType::kEXL3) {   // fused EXL3 bank: trellis [16b,N/16,K/16,E] + suh/svh
                std::string root(nm); root.resize(root.size() - 7);   // strip ".weight"
                const auto* su = Tl(L, (root + ".suh").c_str());
                const auto* sv = Tl(L, (root + ".svh").c_str());
                if (!su || !sv) return std::string(nm) + ": missing EXL3 suh/svh bank";
                if (ti->nbytes % E || su->nbytes % E || sv->nbytes % E)
                    return std::string(nm) + ": bank nbytes % E";
                dst = raw_up(ti); suh = raw_up(su); svh = raw_up(sv);
                if (!err.empty()) return std::string(nm) + ": " + err;
                dt = DType::kEXL3; stride = ti->nbytes / E;
                bits = uint32_t(ti->shape[0] / 16);   // bank ne0 = 16*bits
                dev_bytes_[dev] += ti->nbytes + su->nbytes + sv->nbytes;
                return {};
            }
            if (ti->dtype != DType::kQ4_K && ti->dtype != DType::kQ6_K) return std::string(nm) + ": expert dtype";
            if (ti->nbytes % E) return std::string(nm) + ": nbytes % E";
            dst = dense::upload<void>(a, ti, own, err, ti->dtype);
            if (!err.empty()) return std::string(nm) + ": " + err;
            dt = ti->dtype; stride = ti->nbytes / E; dev_bytes_[dev] += ti->nbytes;
            return {};
        };
        if (auto m = up_exps("ffn_gate_exps.weight", w.gate_exps, w.gate_dt, w.gate_stride, w.gate_suh, w.gate_svh, w.gate_bits); !m.empty()) return "layer " + std::to_string(L) + " " + m;
        if (auto m = up_exps("ffn_up_exps.weight",   w.up_exps,   w.up_dt,   w.up_stride,   w.up_suh,   w.up_svh,   w.up_bits);   !m.empty()) return "layer " + std::to_string(L) + " " + m;
        if (auto m = up_exps("ffn_down_exps.weight", w.down_exps, w.down_dt, w.down_stride, w.down_suh, w.down_svh, w.down_bits); !m.empty()) return "layer " + std::to_string(L) + " " + m;

        // shared expert (always-on; names per docs/qwen3next_80b_dataflow.md §Per-layer tensors).
        // shexp quant weights are Q8_0 → upload_weight_auto dequants them to fp16, so dev_bytes_
        // (which adds ti->nbytes = packed size here) slightly UNDER-counts their true residency —
        // acceptable for the informational LOAD-OK report; the dominant MoE experts above are exact.
        if (cfg.shared_expert_ffn != 0) {
            w.ffn_gate_inp_shexp = dense::upload<float>(a, Tl(L, "ffn_gate_inp_shexp.weight"), own, err, DType::kF32);
            if (!err.empty()) return "layer " + std::to_string(L) + " ffn_gate_inp_shexp: " + err;
            if (const auto* ti = Tl(L, "ffn_gate_inp_shexp.weight")) dev_bytes_[dev] += ti->nbytes;
            w.gate_shexp = LX("ffn_gate_shexp.weight"); if (!err.empty()) return "layer " + std::to_string(L) + " ffn_gate_shexp: " + err;
            w.up_shexp   = LX("ffn_up_shexp.weight");   if (!err.empty()) return "layer " + std::to_string(L) + " ffn_up_shexp: " + err;
            w.down_shexp = LX("ffn_down_shexp.weight"); if (!err.empty()) return "layer " + std::to_string(L) + " ffn_down_shexp: " + err;
            for (const char* nm : {"ffn_gate_shexp.weight","ffn_up_shexp.weight","ffn_down_shexp.weight"})
                if (const auto* ti = Tl(L, nm)) dev_bytes_[dev] += ti->nbytes;
        }
    }

    // Per-card hybrid caches. Count each card's linear (DeltaNet) and full-attn
    // layers, assign card-local cache indices, then init one DeltaNetState +
    // one KvCache per card sized to that card's layer mix. The DeltaNet/KV
    // config fields mirror the validated single-GPU 27B `qwen35` path
    // (ie_perplexity.cpp:314-323 is_q35 branch — same Qwen35Config `hybrid`):
    //   v_head_dim = k_head_dim = ssm_state; conv_channels = ssm_inner +
    //   2*ssm_n_k_heads*ssm_state; conv_kernel = ssm_conv_kernel.
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
            dc.n_v_heads       = h.ssm_n_v_heads;
            dc.v_head_dim      = h.ssm_state;
            dc.k_head_dim      = h.ssm_state;
            dc.conv_channels   = h.ssm_inner + 2u * h.ssm_n_k_heads * h.ssm_state;
            dc.conv_kernel     = h.ssm_conv_kernel;
            if (auto m = dn_[dev].init(fleet.dev(dev), dc); !m.empty())
                return "dn cache dev " + std::to_string(dev) + ": " + m;
        }
        if (n_full[dev]) {
            KvCacheConfig kc{};
            kc.n_layers_full = n_full[dev];
            kc.n_kv_heads    = d.n_kv_heads;
            kc.max_ctx       = max_ctx;    // load gate default 2048; Engine sets opts.max_ctx
            kc.head_dim      = d.head_dim;
            if (auto m = kv_[dev].init(fleet.dev(dev), kc); !m.empty())
                return "kv cache dev " + std::to_string(dev) + ": " + m;
        }
    }
    return {};
}

// ---- F1 forward scaffold / F2 full-attn ------------------------------------
// Free a card's workspace buffers. These are tracked in the Workspace struct
// (NOT owned_) so a T-grow can release the old buffers without leaking and
// without scanning owned_; free_all calls this for every card too.
void Qwen3NextModel::free_ws(uint32_t dev) {
    if (!fleet_ || dev >= ws_.size()) return;
    Workspace& w = ws_[dev];
    auto& alloc = fleet_->dev(dev);
    for (void* p : {static_cast<void*>(w.x), static_cast<void*>(w.x_normed),
                    static_cast<void*>(w.attn_block), static_cast<void*>(w.qg),
                    static_cast<void*>(w.q), static_cast<void*>(w.gate),
                    static_cast<void*>(w.k), static_cast<void*>(w.v),
                    static_cast<void*>(w.attn_out),
                    static_cast<void*>(w.attn_partials),
                    static_cast<void*>(w.positions),
                    static_cast<void*>(w.dn_qkv), static_cast<void*>(w.dn_conv),
                    static_cast<void*>(w.dn_qpre), static_cast<void*>(w.dn_kpre),
                    static_cast<void*>(w.dn_vpre), static_cast<void*>(w.dn_qrep),
                    static_cast<void*>(w.dn_krep), static_cast<void*>(w.dn_ba),
                    static_cast<void*>(w.dn_alpha), static_cast<void*>(w.dn_beta_h),
                    static_cast<void*>(w.dn_g), static_cast<void*>(w.dn_beta),
                    static_cast<void*>(w.dn_out), static_cast<void*>(w.dn_z),
                    static_cast<void*>(w.dn_gn),
                    static_cast<void*>(w.moe_logits), static_cast<void*>(w.moe_xp),
                    static_cast<void*>(w.moe_h), static_cast<void*>(w.moe_out),
                    static_cast<void*>(w.moe_y), static_cast<void*>(w.moe_wpk),
                    static_cast<void*>(w.moe_xp_q8), static_cast<void*>(w.moe_h_q8),
                    static_cast<void*>(w.moe_offsets),
                    static_cast<void*>(w.moe_sorted_idx),
                    static_cast<void*>(w.moe_tk_to_packed),
                    static_cast<void*>(w.moe_topk_idx), static_cast<void*>(w.moe_topk_w),
                    static_cast<void*>(w.sh_gate), static_cast<void*>(w.sh_up),
                    static_cast<void*>(w.sh_h), static_cast<void*>(w.sh_eo),
                    static_cast<void*>(w.sh_g),
                    static_cast<void*>(w.dn_qkvz), static_cast<void*>(w.ex_gate),
                    static_cast<void*>(w.ex_up), static_cast<void*>(w.ex_h),
                    static_cast<void*>(w.ex_down), static_cast<void*>(w.ex_w),
                    static_cast<void*>(w.moe_btf16)})
        if (p) alloc.free(p);
    w = Workspace{};   // null everything, T=0
}

// Per-card scratch. H = cfg_.hybrid.dense.hidden (2048). Re-allocs if T grew —
// FREES the previous buffers first (no leak). Buffers live in the Workspace
// struct, not owned_, so the free here is targeted. Mirror DenseModelSplit::
// ensure_ws + the F2 gated-attn buffers.
std::string Qwen3NextModel::ensure_ws(uint32_t dev, uint32_t max_T) {
    Workspace& w = ws_[dev];
    if (max_T <= w.T) return {};
    if (w.T != 0) free_ws(dev);   // F1-fix #1: release the smaller buffers
    auto& alloc = fleet_->dev(dev);
    auto ah = [&](size_t n) {
        return static_cast<sycl::half*>(alloc.malloc(n * sizeof(sycl::half)));
    };
    auto af = [&](size_t n) {
        return static_cast<float*>(alloc.malloc(n * sizeof(float)));
    };
    const uint32_t H = cfg_.hybrid.dense.hidden;
    const uint64_t T = max_T;
    // qwen3next gated full-attn dims (docs/qwen3next_80b_dataflow.md §GatedAttn)
    const uint32_t n_q = cfg_.hybrid.dense.n_q_heads;   // 16
    const uint32_t HD  = cfg_.hybrid.dense.head_dim;    // 256
    const uint32_t N_q  = n_q * HD;                      // 4096
    const uint32_t N_qg = 2u * N_q;                      // 8192 (Q|gate)
    const uint32_t N_kv = cfg_.hybrid.dense.n_kv_heads * HD;  // 512
    w.x          = ah(T * H);
    w.x_normed   = ah(T * H);
    w.attn_block = ah(T * H);
    w.qg         = ah(T * N_qg);
    w.q          = ah(T * N_q);
    w.gate       = ah(T * N_q);
    w.k          = ah(T * N_kv);
    w.v          = ah(T * N_kv);
    w.attn_out   = ah(T * N_q);
    w.positions  = static_cast<int32_t*>(alloc.malloc(T * sizeof(int32_t)));
    // F3 gated-DeltaNet scratch (qwen3next ssm dims).
    const uint32_t SKH     = cfg_.hybrid.ssm_n_k_heads;            // 16
    const uint32_t SVH     = cfg_.hybrid.ssm_n_v_heads;            // 32
    const uint32_t SHD     = cfg_.hybrid.ssm_state;                // 128
    const uint32_t SI      = cfg_.hybrid.ssm_inner;                // 4096
    const uint32_t conv_ch = SI + 2u * SKH * SHD;                  // 8192
    const uint32_t kw      = SKH * SHD;                            // 2048 (q/k pre-repeat)
    const uint32_t qkrep   = (SVH / SKH) * kw;                     // 2*2048 = 4096
    w.dn_qkv    = ah(T * conv_ch);   // [T,8192]
    w.dn_conv   = ah(T * conv_ch);   // [T,8192]
    w.dn_qpre   = af(T * kw);        // [T,2048] fp32
    w.dn_kpre   = af(T * kw);        // [T,2048] fp32
    w.dn_vpre   = af(T * SI);        // [T,4096] fp32
    w.dn_qrep   = af(T * qkrep);     // [T,4096] fp32
    w.dn_krep   = af(T * qkrep);     // [T,4096] fp32
    w.dn_ba     = ah(T * 2u * SVH);  // [T,64]  = β|α
    w.dn_alpha  = ah(T * SVH);       // [T,32]
    w.dn_beta_h = ah(T * SVH);       // [T,32]
    w.dn_g      = af(T * SVH);       // [T,32] fp32
    w.dn_beta   = af(T * SVH);       // [T,32] fp32
    w.dn_out    = af(T * SI);        // [T,4096] fp32
    w.dn_z      = ah(T * SI);        // [T,4096] z-gate
    w.dn_gn     = ah(T * SI);        // [T,4096] gated_rms_norm out
    // F4 MoE + shared-expert scratch (mirror qwen3moe::ensure_workspace fused-MoE
    // staging; dims = qwen3next: E=512, K=10, E_ffn=512). TK = T·K packed rows.
    const uint32_t E   = cfg_.n_experts;            // 512
    const uint32_t K   = cfg_.n_experts_used;       // 10
    const uint32_t EF  = cfg_.expert_ffn;           // 512
    const uint64_t TK  = T * uint64_t(K);
    w.moe_logits      = ah(T * uint64_t(E));        // [T,E]
    w.moe_xp          = ah(TK * H);                 // [T*K,H]
    w.moe_h           = ah(TK * EF);                // [T*K,E_ffn]
    w.moe_out         = ah(TK * H);                 // [T*K,H]
    w.moe_y           = ah(T * H);                  // [T,H]
    w.moe_wpk         = ah(TK);                     // [T*K]
    w.moe_xp_q8       = alloc.malloc(TK * (H  / 32) * sizeof(block_q8_1s));
    w.moe_h_q8        = alloc.malloc(TK * (EF / 32) * sizeof(block_q8_1s));
    w.moe_offsets     = static_cast<uint32_t*>(alloc.malloc((E + 1) * sizeof(uint32_t)));
    w.moe_sorted_idx  = static_cast<int32_t*>(alloc.malloc(TK * sizeof(int32_t)));
    w.moe_tk_to_packed= static_cast<int32_t*>(alloc.malloc(TK * sizeof(int32_t)));
    w.moe_topk_idx    = static_cast<int32_t*>(alloc.malloc(uint64_t(K) * sizeof(int32_t)));
    w.moe_topk_w      = ah(K);
    w.sh_gate         = ah(T * EF);
    w.sh_up           = ah(T * EF);
    w.sh_h            = ah(T * EF);
    w.sh_eo           = ah(T * H);
    w.sh_g            = ah(T);
    // EXL3-only scratch (small; always allocated — the Q4_K path simply ignores them).
    w.dn_qkvz         = ah(T * (conv_ch + SI));   // [T,12288] fused in_proj_qkvz
    w.ex_gate         = ah(EF);                    // one-expert working buffers
    w.ex_up           = ah(EF);
    w.ex_h            = ah(EF);
    w.ex_down         = ah(H);
    w.ex_w            = ah(TK);                     // all routing weights (1 upload/layer)
    w.moe_h2          = ah(TK * EF);                // [T*K,E_ffn] batched up-proj (EXL3 fused MoE / oneDNN up)
    w.moe_rowtok      = static_cast<int32_t*>(alloc.malloc(TK * sizeof(int32_t)));
    w.moe_btf16       = ah(uint64_t(H) * EF);       // [H,E_ffn] per-expert oneDNN dequant Bt (reused)
    if (!w.x || !w.x_normed || !w.attn_block || !w.qg || !w.q || !w.gate ||
        !w.k || !w.v || !w.attn_out || !w.positions ||
        !w.dn_qkv || !w.dn_conv || !w.dn_qpre || !w.dn_kpre || !w.dn_vpre ||
        !w.dn_qrep || !w.dn_krep || !w.dn_ba || !w.dn_alpha || !w.dn_beta_h ||
        !w.dn_g || !w.dn_beta || !w.dn_out || !w.dn_z || !w.dn_gn ||
        !w.moe_logits || !w.moe_xp || !w.moe_h || !w.moe_out || !w.moe_y ||
        !w.moe_wpk || !w.moe_xp_q8 || !w.moe_h_q8 || !w.moe_offsets ||
        !w.moe_sorted_idx || !w.moe_tk_to_packed || !w.moe_topk_idx ||
        !w.moe_topk_w || !w.sh_gate || !w.sh_up || !w.sh_h || !w.sh_eo || !w.sh_g ||
        !w.dn_qkvz || !w.ex_gate || !w.ex_up || !w.ex_h || !w.ex_down || !w.ex_w ||
        !w.moe_h2 || !w.moe_rowtok || !w.moe_btf16)
        return "qwen3next workspace alloc failed on dev " + std::to_string(dev);
    // FA-2 decode partials: only for cards that hold ≥1 full-attn layer (have a
    // KV cache). Sized to this card's KV max_ctx; layout mirrors qwen3moe
    // ensure_attn_partials (n_chunks × n_q_heads × (head_dim+2) fp32).
    if (kv_[dev].ready()) {
        const uint32_t max_ctx = kv_[dev].config().max_ctx;
        constexpr uint32_t Bc_floor = 64;
        const uint32_t n_chunks_max = (max_ctx + Bc_floor - 1) / Bc_floor;
        const uint64_t n_floats = uint64_t(n_chunks_max) * n_q * (HD + 2);
        w.attn_partials = static_cast<float*>(alloc.malloc(n_floats * sizeof(float)));
        if (!w.attn_partials)
            return "qwen3next attn_partials alloc failed on dev " + std::to_string(dev);
        w.partials_ctx = max_ctx;
    }
    w.T = max_T;
    return {};
}

std::string Qwen3NextModel::forward(const int32_t* input_ids, uint32_t T,
                                    uint32_t start_pos, bool reset_kv,
                                    sycl::half* out_logits_host) {
    if (T == 0) return "T == 0";
    const Qwen35Config& h = cfg_.hybrid;
    const DenseConfig&  d = h.dense;
    const uint32_t H   = d.hidden;
    const uint32_t V   = d.vocab;
    const float    eps = d.rms_eps;

    // ---- env-gated decode/prefill phase profiler (IE_QWEN3NEXT_PROFILE=1) ----
    // Wait-bracketed: each mark() drains the card's queue then attributes the
    // elapsed wall-time to a phase. The added drains serialize more than prod,
    // so the ABSOLUTE total is slower than the bench tok/s — read it as a
    // RELATIVE breakdown (where does the time go), plus the pure-CPU host cost
    // (pf_hostcpu, measured with NO added wait → real router/packing overhead).
    static const bool prof = std::getenv("IE_QWEN3NEXT_PROFILE") != nullptr;
    using pf_clock = std::chrono::steady_clock;
    static double pf_attn = 0, pf_router = 0, pf_moe = 0, pf_shexp = 0;
    static double pf_hostcpu = 0, pf_other = 0, pf_total = 0;
    static uint64_t pf_tok = 0, pf_calls = 0;
    auto pf_ms = [](pf_clock::time_point a, pf_clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    auto pf_t_total = pf_clock::now();
    pf_clock::time_point pf_last;
    auto mark = [&](sycl::queue& qq, double& acc) {
        if (!prof) return;
        qq.wait();
        auto now = pf_clock::now();
        acc += pf_ms(pf_last, now);
        pf_last = now;
    };
    const uint32_t n_layers = h.n_transformer_layers();
    // gated full-attn dims (docs/qwen3next_80b_dataflow.md §GatedAttn)
    const uint32_t n_q   = d.n_q_heads;          // 16
    const uint32_t n_kv  = d.n_kv_heads;         // 2
    const uint32_t HD    = d.head_dim;           // 256
    const uint32_t N_q   = n_q * HD;             // 4096
    const uint32_t N_qg  = 2u * N_q;             // 8192 (Q|gate joint)
    const uint32_t N_kv  = n_kv * HD;            // 512
    const uint32_t rope_n   = d.rope_dim;        // 64  (n_rot; NEOX split-halves)
    const float    rope_th  = d.rope_theta;      // 1e7
    // MoE + shared-expert dims (docs/qwen3next_80b_dataflow.md §MoE)
    const uint32_t E   = cfg_.n_experts;         // 512
    const uint32_t K   = cfg_.n_experts_used;    // 10
    const uint32_t EF  = cfg_.expert_ffn;        // 512 (= the int-dot base case)

    if (ws_.size() != n_dev_) ws_.assign(n_dev_, {});

    // F1-fix #2: workspace on EVERY card (mirror DenseModelSplit::forward —
    // drops the gapless / layer-owning-only assumption; embed/head cards are
    // covered too).
    for (uint32_t dev = 0; dev < n_dev_; ++dev)
        if (auto m = ensure_ws(dev, T); !m.empty()) return m;

    // positions [start_pos..] on every card; reset per-card hybrid state if
    // asked. KV is positional → start_pos handles in-cache placement, but a
    // FRESH sequence must clear stale lengths (F1-fix #3, mirror dense_split).
    std::vector<int32_t> pos(T);
    for (uint32_t t = 0; t < T; ++t) pos[t] = int32_t(start_pos + t);
    for (uint32_t dev = 0; dev < n_dev_; ++dev) {
        if (ws_[dev].positions)
            fleet_->dev(dev).queue().memcpy(ws_[dev].positions, pos.data(),
                                            T * sizeof(int32_t)).wait();
        if (reset_kv) {
            if (dn_[dev].ready()) dn_[dev].reset(fleet_->dev(dev).queue());
            if (kv_[dev].ready()) kv_[dev].reset();   // F1-fix #3: clear KV lengths
        }
    }

    // embedding → ws_[embed_dev].x (token_embd is Q4_K or Q6_K, keyed on dtype).
    {
        auto& alloc = fleet_->dev(plan_.embed_dev);
        auto& q = alloc.queue();
        int32_t* d_ids = static_cast<int32_t*>(alloc.malloc(T * sizeof(int32_t)));
        if (!d_ids) return "d_ids alloc failed";
        q.memcpy(d_ids, input_ids, T * sizeof(int32_t)).wait();
        if (token_embd_dtype_ == DType::kQ4_K)
            embedding_lookup_q4k(q, d_ids, token_embd_, ws_[plan_.embed_dev].x, T, H);
        else if (token_embd_dtype_ == DType::kF16)   // EXL3: token_embd is plain F16 (embd arg first)
            embedding_lookup_f16(q, static_cast<const sycl::half*>(token_embd_), d_ids,
                                 ws_[plan_.embed_dev].x, T, H);
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
        if (prof) { q.wait(); pf_last = pf_clock::now(); }
        for (uint32_t L = 0; L < n_layers; ++L) {
            if (plan_.dev_of_layer[L] != dev) continue;
            const LayerW& lw = layers_[L];
            // pre-attn norm
            rms_norm_f32w(q, w.x, lw.attn_norm, w.x_normed, T, H, eps);
            if (!lw.is_linear) {
                // ---- F2: gated full-attention (mirror qwen35_dense full-attn
                // branch; NEOX partial RoPE like qwen3moe — the ONLY rope op,
                // split-halves form, NOT M-RoPE). KV slot is this card's local
                // index kv_local_[L]; only full-attn layers hold a KV slot. ----
                // joint Q|gate projection → split per head into Q and σ-gate
                dense::gemv_q_T(q, w.x_normed, lw.attn_q, w.qg, H, N_qg, T);
                split_q_gate_per_head(q, w.qg, w.q, w.gate, T, n_q, HD);
                // K, V
                dense::gemv_q_T(q, w.x_normed, lw.attn_k, w.k, H, N_kv, T);
                dense::gemv_q_T(q, w.x_normed, lw.attn_v, w.v, H, N_kv, T);
                // per-head Q/K RMS-norm over head_dim (F32 norm weight)
                rms_norm_f32w(q, w.q, lw.attn_q_norm, w.q, T * n_q,  HD, eps);
                rms_norm_f32w(q, w.k, lw.attn_k_norm, w.k, T * n_kv, HD, eps);
                // NEOX partial RoPE (n_rot 64, θ1e7) — same op qwen3moe uses
                rope_partial(q, w.q, w.positions, w.q, T, n_q,  HD, rope_n, rope_th);
                rope_partial(q, w.k, w.positions, w.k, T, n_kv, HD, rope_n, rope_th);
                // SDPA with this layer's KV slice (card-local kv_local_[L])
                sycl::half* kc = kv_[dev].k_ptr() + per_layer_kv * kv_local_[L];
                sycl::half* vc = kv_[dev].v_ptr() + per_layer_kv * kv_local_[L];
                const uint32_t max_ctx = kv_[dev].config().max_ctx;
                if (T == 1 && w.attn_partials) {
                    full_attention_fa2_decode(q, w.q, w.k, w.v, kc, vc,
                                              w.attn_out, w.attn_partials,
                                              start_pos, n_q, n_kv, HD, max_ctx);
                } else {
                    // Long-ctx full-attn prefill: route the head_dim-256 full-attn
                    // layers through the proven Gemma wide-tile kernel (KV once per Br
                    // query-tile, appends k/v internally, full causal window=0) for the
                    // collapsing long-ctx region — same lever as 27B (`07ab8ef`) / crown
                    // (`bf73093`). Numerically equivalent to naive at hd256 (same
                    // 1/sqrt(HD) post-dot scale on the same unscaled w.q; argmax-bit-
                    // identical on Gemma). Plain SIMD kernel (no oneDNN/ESIMD/block2d) →
                    // multi-card safe. GATED ctx≥minctx (default 6144). Opt-out
                    // IE_QWEN3NEXT_NO_FA2_TILE; tune IE_QWEN3NEXT_FA2_TILE_MINCTX. NOTE:
                    // 80B prefill is also MoE-GEMM-bound (like crown) → this recovers the
                    // attention portion only; the MoE-prefill GEMM is the residual. [2026-06-26]
                    static const bool no_tile = std::getenv("IE_QWEN3NEXT_NO_FA2_TILE") != nullptr;
                    static const uint32_t tile_minctx = []() -> uint32_t {
                        const char* e = std::getenv("IE_QWEN3NEXT_FA2_TILE_MINCTX");
                        if (!e) return 6144u;
                        int v = std::atoi(e); return v > 0 ? uint32_t(v) : 6144u;
                    }();
                    if (!no_tile && HD == 256 && (start_pos + T) >= tile_minctx) {
                        full_attention_fa2_prefill_tile_gemma(
                            q, w.q, w.k, w.v, kc, vc, w.attn_out, T, start_pos,
                            n_q, n_kv, HD, max_ctx, 0 /*window: full causal*/);
                    } else {
                        full_attention(q, w.q, w.k, w.v, kc, vc, w.attn_out,
                                       T, start_pos, n_q, n_kv, HD, max_ctx);
                    }
                }
                kv_[dev].set_length(kv_local_[L], start_pos + T);
                // post-attention sigmoid gate: attn_out · σ(gate)
                sigmoid_gate(q, w.attn_out, w.gate, w.attn_out, uint64_t(T) * N_q);
                // output projection → attn_block (the attn contribution)
                dense::gemv_q_T(q, w.attn_out, lw.attn_output, w.attn_block, N_q, H, T);
            } else {
                // ---- F3: gated DeltaNet (linear-attention). Mirror qwen35_dense
                // DeltaNet branch with the qwen3next deltas: n_v=32 (27B was 48),
                // conv_ch=8192, ssm_inner=4096, and a FUSED ssm_ba [T,64]=β|α
                // (the 27B had separate N-padded ssm_alpha/ssm_beta projections). ----
                const uint32_t SKH     = h.ssm_n_k_heads;          // 16
                const uint32_t SVH     = h.ssm_n_v_heads;          // 32 (n_v)
                const uint32_t SHD     = h.ssm_state;              // 128 (head dim)
                const uint32_t SI      = h.ssm_inner;              // 4096 (d_inner)
                const uint32_t conv_ch = SI + 2u * SKH * SHD;      // 8192
                const uint32_t kw      = SKH * SHD;                // 2048 (q/k pre-repeat)
                const uint32_t rep     = SVH / SKH;                // 2 (16→32)
                const float qscale = 1.0f / sycl::sqrt(float(SHD));

                // attn_qkv projection → conv input [conv_ch=8192]
                if (lw.attn_qkv.dt == DType::kEXL3) {
                    // EXL3: in_proj_qkvz is ONE fused trellis [H→conv_ch+SI=12288]; project into
                    // dn_qkvz then slice [0:conv_ch]→dn_qkv (conv path), [conv_ch:]→dn_z (z-gate).
                    dense::gemv_q_T(q, w.x_normed, lw.attn_qkv, w.dn_qkvz, H, conv_ch + SI, T);
                    slice_qkvz(q, w.dn_qkvz, w.dn_qkv, w.dn_z, T, SKH, SHD, rep * SHD);
                } else {
                    dense::gemv_q_T(q, w.x_normed, lw.attn_qkv, w.dn_qkv, H, conv_ch, T);
                }
                // causal depthwise conv1d (fuses silu) — per-layer streaming conv_state
                sycl::half* conv_state = dn_[dev].conv_state_ptr() +
                    uint64_t(dn_local_[L]) * dn_[dev].conv_elems_per_layer();
                depthwise_conv1d_causal(q, w.dn_qkv, lw.ssm_conv1d_fp16, conv_state,
                                        w.dn_conv, T, conv_ch, h.ssm_conv_kernel);
                // split (q|k|v) + cast→fp32: q,k width kw=2048; v width SI=4096
                cast_qkv_split_fp16_to_fp32(q, w.dn_conv, w.dn_qpre, w.dn_kpre,
                                            w.dn_vpre, T, kw, SI);
                // per-head L2-norm over 16 heads × 128 (q with qscale, k with 1.0)
                l2_norm_scale(q, w.dn_qpre, w.dn_qpre, T * SKH, SHD, qscale, 1e-6f);
                l2_norm_scale(q, w.dn_kpre, w.dn_kpre, T * SKH, SHD, 1.0f,   1e-6f);
                // repeat q,k heads 16→32 (⚠ 27B was 16→48)
                // qwen3next uses ggml-REPEAT INTERLEAVE (k-head h → v-heads
                // rep*h..rep*h+rep-1), NOT the 27B's tiled convention.
                repeat_interleave_heads(q, w.dn_qpre, w.dn_qrep, T, SKH, SHD, rep, /*interleave=*/true);
                repeat_interleave_heads(q, w.dn_kpre, w.dn_krep, T, SKH, SHD, rep, /*interleave=*/true);
                // FUSED ssm_ba projection → [T,64]. The split is PER-K-HEAD
                // INTERLEAVED, not two contiguous halves: the oracle reshapes
                // ssm_ba to {4, SKH=16} and views a(α)=[0:2], b(β)=[2:4] within
                // each group's 4 → layout [T, SKH, 2(α|β), SVH/SKH]. A contiguous
                // [β(32)|α(32)] slice scrambles heads (was the L0 dn_alpha/beta
                // divergence: 216/269 vs oracle a/b 118/367, same 484.92 total).
                // split_q_gate_per_head reads exactly [T, n_heads, 2, head_dim]
                // → q_out=first(α), gate_out=second(β).
                // per-group layout is [β(2)|α(2)] (beta FIRST, alpha second), so
                // beta_h=q_out(first), alpha=gate_out(second). Verified vs oracle:
                // dn_beta_h→b=366.76, dn_alpha→a=118.16 (a→softplus/g, b→sigmoid/β).
                dense::gemv_q_T(q, w.x_normed, lw.ssm_ba, w.dn_ba, H, 2u * SVH, T);
                split_q_gate_per_head(q, w.dn_ba, w.dn_beta_h, w.dn_alpha, T, SKH, SVH / SKH);
                // g = -exp(ssm_a)·softplus(α+dt_bias);  β = sigmoid(β_proj).
                // (alpha first, beta second — matches qwen35_dense's call order.)
                compute_g_beta_h16(q, w.dn_alpha, w.dn_beta_h, lw.ssm_a, lw.ssm_dt_bias,
                                   w.dn_g, w.dn_beta, T, SVH);
                // gated delta-rule recurrence (state carried per linear layer)
                float* state_layer = dn_[dev].state_ptr() +
                    uint64_t(dn_local_[L]) * dn_[dev].state_elems_per_layer();
                // §1 BMG DeltaNet-recurrence HW-bug cap: each recurrence LAUNCH must
                // stay ≤512 steps (the validated-safe regime — longer launches risk the
                // stochastic state[] divergence AND the Triton-#6658 long-recurrence
                // DEVICE_LOST hazard, both growing with launch length). When forward()
                // gets a big-T prefill chunk (the large-M MoE long-ctx lever, Step 2 of
                // the 80B-prefill unblock), sub-chunk the scan HERE: the recurrence is a
                // sequential state-carrying scan, so splitting [0:T] into ≤512 segments
                // that thread `state_layer` through is bit-identical to one call (and
                // AVOIDS the §1 non-determinism by keeping launches short). NO-OP at
                // T≤512 (one iteration == prior behavior; default chunk unchanged).
                // Mirrors qwen36.cpp's proven crown loop. [2026-06-26]
                static const uint32_t DN_RECUR_CHUNK = []() -> uint32_t {
                    const char* e = std::getenv("IE_QWEN3NEXT_DN_RECUR_CHUNK");
                    if (!e) return 512u;
                    int v = std::atoi(e);
                    return (v >= 1 && v <= 512) ? uint32_t(v) : 512u;
                }();
                const uint64_t dn_qkv_stride = uint64_t(SVH) * SHD;  // per-position
                const uint64_t dn_gb_stride  = uint64_t(SVH);
                for (uint32_t off = 0; off < T; off += DN_RECUR_CHUNK) {
                    const uint32_t n = std::min<uint32_t>(DN_RECUR_CHUNK, T - off);
                    deltanet_recurrence(q,
                        w.dn_qrep + off * dn_qkv_stride, w.dn_krep + off * dn_qkv_stride,
                        w.dn_vpre + off * dn_qkv_stride, w.dn_g + off * dn_gb_stride,
                        w.dn_beta + off * dn_gb_stride, state_layer,
                        w.dn_out + off * dn_qkv_stride, /*B=*/1, n, SVH, SHD, SHD);
                }
                // gated RMS-norm with z = sigmoid-gate from attn_gate · x_normed
                // z-gate: EXL3 already filled dn_z via the qkvz slice (no separate attn_gate).
                if (lw.attn_qkv.dt != DType::kEXL3)
                    dense::gemv_q_T(q, w.x_normed, lw.attn_gate, w.dn_z, H, SI, T);
                gated_rms_norm(q, w.dn_out, w.dn_z, lw.ssm_norm_fp16, w.dn_gn,
                               T * SVH, SHD, eps);
                // output projection → attn_block (the linear-attn contribution)
                dense::gemv_q_T(q, w.dn_gn, lw.ssm_out, w.attn_block, SI, H, T);
            }
            residual_add(q, w.x, w.attn_block, w.x, uint64_t(T) * H);
            mark(q, pf_attn);   // pre-attn norm + full-attn/DeltaNet + residual
            // post-attn (pre-FFN) norm → w.x_normed = rms(post_attention_norm)
            rms_norm_f32w(q, w.x, lw.post_attn_norm, w.x_normed, T, H, eps);

            // ---- F4: MoE int-dot (E_ffn=512) + shared expert (every layer) ----
            // Port of qwen3moe::moe_ffn_fused_prefill (T>1) / moe_ffn_decode (T==1):
            //   GPU-gemm router → host softmax+top-K+renorm → counting-sort packer
            //   → gather → gate_up_silu_q8 → down_q6k_q8_gen (E_ffn=512) → reduce.
            // The shared expert (qwen36.cpp:1516-1572) runs every layer in parallel
            // and is scaled-added onto the MoE output by its per-token σ-gate. The
            // combined FFN output lands in w.moe_y, then residual_add into w.x.
            {
                // 1) router logits via ONE GPU gemm: [T,H]@router_wᵀ{F16} → [T,E].
                dense::gemv_q_T(q, w.x_normed,
                                DenseQuantPtr{lw.router_w_dev, DType::kF16},
                                w.moe_logits, H, E, T);
                std::vector<sycl::half> hl(uint64_t(T) * E);
                q.memcpy(hl.data(), w.moe_logits, uint64_t(T) * E * sizeof(sycl::half)).wait();
                mark(q, pf_router);   // router gemm + logits memcpy + GPU drain (the sync)
                auto pf_h0 = pf_clock::now();
                std::vector<float> flog(uint64_t(T) * E);
                for (uint64_t i = 0; i < uint64_t(T) * E; ++i) flog[i] = float(hl[i]);
                // host softmax + top-K + renorm per token (HF norm_topk_prob order).
                std::vector<std::vector<std::pair<uint32_t, float>>> routes(T);
                std::vector<std::pair<uint32_t, float>> topk;
                for (uint32_t t = 0; t < T; ++t) {
                    route_from_logits(flog.data() + uint64_t(t) * E, E, K, topk);
                    routes[t] = topk;
                }
                if (prof) { auto n = pf_clock::now(); pf_hostcpu += pf_ms(pf_h0, n); pf_last = n; }

                // T==1 decode stays on the fp16 moe_decode_* path (below). The qwen3moe
                // #6 lever (route T==1 through the fused int-dot path) was MEASURED here
                // and is +CORRECT but ~40% SLOWER for qwen3next (decode 30 vs 50 tok/s,
                // PPL 4.7255≈4.7282): the 80B is split across 2 CARDS, so the fused path's
                // per-layer host packing + gather + cross-card coordination dominates at
                // T==1 — unlike single-card qwen3moe where it wins. So fp16 decode stays
                // the default (it's already 1.40× ahead of llama at 50 tok/s).
                if (lw.gate_dt == DType::kEXL3) {
                    // EXL3 MoE. Two paths, same result (diff-validated):
                    //  • FUSED (default): batched Hadamard + trellis-decode over all R=T·K
                    //    active rows → 11 launches/layer regardless of T,K (was ~11·T·K).
                    //    Reuses the proven gemv_exl3_forward 3-step (had_in⊙suh → trellis →
                    //    had_out⊙svh) batched with a per-row expert gather. suh/svh are
                    //    PER-EXPERT so the Hadamards stay separate launches (can't hoist),
                    //    but the K·T·3 trellis gemvs collapse into 3 batched launches.
                    //  • SLOW (IE_EXL3_MOE_SLOW=1): the original per-(token,expert) loop, kept
                    //    as the correctness oracle. Per (token t, active expert e):
                    //      h = swiglu(gemv_exl3(x_t,gate_e), gemv_exl3(x_t,up_e));  [E_ffn]
                    //      out = gemv_exl3(h,down_e);  moe_y[t] += wgt·out.         [H]
                    //    Per-expert offsets: trellis +e*stride; suh +e*K, svh +e*N
                    //    (gate/up K=H,N=EF; down K=EF,N=H).
                    static const bool moe_slow = std::getenv("IE_EXL3_MOE_SLOW") != nullptr;
                    if (!moe_slow) {
                        const uint32_t R = T * K;                 // active rows (token,expert)
                        std::vector<int32_t>    rexp(R), rtok(R), ridn(R);
                        std::vector<sycl::half> rw(R);
                        for (uint32_t t = 0; t < T; ++t)
                            for (uint32_t k = 0; k < K; ++k) {
                                const uint32_t r = t * K + k;
                                rexp[r] = int32_t(routes[t][k].first);   // row → expert
                                rtok[r] = int32_t(t);                    // row → token (x gather)
                                ridn[r] = int32_t(r);                    // identity reduce map
                                rw[r]   = sycl::half(routes[t][k].second);
                            }
                        q.memcpy(w.moe_sorted_idx,   rexp.data(), R * sizeof(int32_t));
                        q.memcpy(w.moe_rowtok,       rtok.data(), R * sizeof(int32_t));
                        q.memcpy(w.moe_tk_to_packed, ridn.data(), R * sizeof(int32_t));
                        q.memcpy(w.moe_wpk,          rw.data(),   R * sizeof(sycl::half)).wait();
                        const int32_t* rxe = w.moe_sorted_idx;
                        auto* gsu = static_cast<const sycl::half*>(lw.gate_suh);
                        auto* gsv = static_cast<const sycl::half*>(lw.gate_svh);
                        auto* usu = static_cast<const sycl::half*>(lw.up_suh);
                        auto* usv = static_cast<const sycl::half*>(lw.up_svh);
                        auto* dsu = static_cast<const sycl::half*>(lw.down_suh);
                        auto* dsv = static_cast<const sycl::half*>(lw.down_svh);
                        // gather active-row inputs: moe_xp[r] = x_normed[token(r)].
                        moe_gather_rows(q, w.x_normed, w.moe_rowtok, w.moe_xp, R, H);
                        // gate: had_in(x⊙suh)→moe_out ; trellis→moe_h ; had_out(·)⊙svh→moe_h
                        hadamard_transform_moe(q, w.moe_xp, w.moe_out, H, R, rxe, gsu, nullptr);
                        gemv_exl3_moe(q, w.moe_out, lw.gate_exps, lw.gate_stride, rxe, w.moe_h, H, EF, R, lw.gate_bits);
                        hadamard_transform_moe(q, w.moe_h, w.moe_h, EF, R, rxe, nullptr, gsv);
                        // up:   had_in(x⊙suh)→moe_out ; trellis→moe_h2 ; had_out(·)⊙svh→moe_h2
                        hadamard_transform_moe(q, w.moe_xp, w.moe_out, H, R, rxe, usu, nullptr);
                        gemv_exl3_moe(q, w.moe_out, lw.up_exps, lw.up_stride, rxe, w.moe_h2, H, EF, R, lw.up_bits);
                        hadamard_transform_moe(q, w.moe_h2, w.moe_h2, EF, R, rxe, nullptr, usv);
                        // swiglu(gate,up) → moe_h (in-place on the gate buffer)
                        swiglu(q, w.moe_h, w.moe_h2, w.moe_h, uint64_t(R) * EF);
                        // down: had_in(h⊙suh)→moe_h ; trellis→moe_out ; had_out(·)⊙svh→moe_out
                        hadamard_transform_moe(q, w.moe_h, w.moe_h, EF, R, rxe, dsu, nullptr);
                        gemv_exl3_moe(q, w.moe_h, lw.down_exps, lw.down_stride, rxe, w.moe_out, EF, H, R, lw.down_bits);
                        hadamard_transform_moe(q, w.moe_out, w.moe_out, H, R, rxe, nullptr, dsv);
                        // reduce: moe_y[t] = Σ_k wgt·moe_out[t*K+k] (ACCUMULATES → zero first)
                        q.memset(w.moe_y, 0, uint64_t(T) * H * sizeof(sycl::half));
                        moe_prefill_reduce(q, w.moe_out,
                                           reinterpret_cast<const uint32_t*>(w.moe_tk_to_packed),
                                           w.moe_wpk, w.moe_y, T, K, H);
                    } else {
                    std::vector<sycl::half> aw; aw.reserve(uint64_t(T) * K);
                    std::vector<uint32_t>   woff(T + 1, 0);
                    for (uint32_t t = 0; t < T; ++t) {
                        for (auto& pr : routes[t]) aw.push_back(sycl::half(pr.second));
                        woff[t + 1] = uint32_t(aw.size());
                    }
                    if (!aw.empty()) q.memcpy(w.ex_w, aw.data(), aw.size() * sizeof(sycl::half)).wait();
                    q.memset(w.moe_y, 0, uint64_t(T) * H * sizeof(sycl::half));
                    auto* gtr = static_cast<const uint8_t*>(lw.gate_exps);
                    auto* utr = static_cast<const uint8_t*>(lw.up_exps);
                    auto* dtr = static_cast<const uint8_t*>(lw.down_exps);
                    auto* gsu = static_cast<const sycl::half*>(lw.gate_suh);
                    auto* gsv = static_cast<const sycl::half*>(lw.gate_svh);
                    auto* usu = static_cast<const sycl::half*>(lw.up_suh);
                    auto* usv = static_cast<const sycl::half*>(lw.up_svh);
                    auto* dsu = static_cast<const sycl::half*>(lw.down_suh);
                    auto* dsv = static_cast<const sycl::half*>(lw.down_svh);
                    for (uint32_t t = 0; t < T; ++t) {
                        const sycl::half* xt = w.x_normed + uint64_t(t) * H;
                        uint32_t wi = woff[t];
                        for (auto& pr : routes[t]) {
                            const uint64_t e = pr.first;
                            gemv_exl3_forward(q, xt, gtr + e * lw.gate_stride, gsu + e * H, gsv + e * EF,
                                              w.ex_gate, H, EF, lw.gate_bits, {});
                            gemv_exl3_forward(q, xt, utr + e * lw.up_stride, usu + e * H, usv + e * EF,
                                              w.ex_up, H, EF, lw.up_bits, {});
                            swiglu(q, w.ex_gate, w.ex_up, w.ex_h, EF);
                            gemv_exl3_forward(q, w.ex_h, dtr + e * lw.down_stride, dsu + e * EF, dsv + e * H,
                                              w.ex_down, EF, H, lw.down_bits, {});
                            scaled_add_per_token_row(q, w.ex_down, w.ex_w + wi, w.moe_y + uint64_t(t) * H, 1, H);
                            ++wi;
                        }
                    }
                    }  // end else (IE_EXL3_MOE_SLOW)
                } else if (T > 1) {
                    // 2) host counting-sort packer → upload device arrays.
                    MoePacking pk;
                    build_moe_packing(routes, E, K, pk);
                    const uint32_t TK = T * K;
                    q.memcpy(w.moe_offsets,      pk.expert_offsets.data(), (E + 1) * sizeof(uint32_t));
                    q.memcpy(w.moe_sorted_idx,   pk.sorted_idx.data(),     TK * sizeof(int32_t));
                    q.memcpy(w.moe_tk_to_packed, pk.tk_to_packed.data(),   TK * sizeof(int32_t));
                    std::vector<sycl::half> wpk(TK);
                    for (uint32_t i = 0; i < TK; ++i) wpk[i] = sycl::half(pk.weights_packed[i]);
                    q.memcpy(w.moe_wpk, wpk.data(), TK * sizeof(sycl::half)).wait();

                    // 3) gather expert-sorted input rows: moe_xp[i] = x_normed[sorted_idx[i]].
                    moe_gather_rows(q, w.x_normed, w.moe_sorted_idx, w.moe_xp, TK, H);
                    // 4+5) stages 1+2: gate+up+silu then down → expert-sorted moe_out.
                    // Two compute paths, BOTH leaving w.moe_out[T*K,H] for the shared
                    // reduce below: (A) oneDNN large-M GEMM (the long-ctx lever, Step 2)
                    // and (B) the default int-dot W4A8.
                    // oneDNN gate: per-expert dequant Q4_K/Q6_K → fp16 Bt, then
                    // gemm_fp16_onednn over the expert-sorted rows. It AMORTIZES the
                    // dequant across M rows/expert (M ≈ T·K/E ≈ T·0.0195), so it only
                    // wins at large T: clean-box A/B = 1.53× over int-dot at M≈160 (T=8K).
                    // Per-card SAFE: gemm_fp16_onednn(q,…) resolves q's device engine via
                    // the per-device ctx map (Step 1), so each card runs its own engine —
                    // no card-0 DEVICE_LOST. DEFAULT-ON above the T-gate
                    // IE_QWEN3NEXT_MOE_ONEDNN_MINT (def 6144 → M≈120, inside the confirmed
                    // -winning regime); opt-out IE_QWEN3NEXT_NO_MOE_ONEDNN. The engine feeds
                    // a big chunk only for long-ctx (max_ctx≥8192) so short prompts never
                    // reach the gate. gate→moe_h, up→moe_h2, down→moe_out (AoS experts).
                    static const bool moe_onednn_off = std::getenv("IE_QWEN3NEXT_NO_MOE_ONEDNN") != nullptr;
                    static const uint32_t moe_onednn_minT = []() -> uint32_t {
                        const char* e = std::getenv("IE_QWEN3NEXT_MOE_ONEDNN_MINT");
                        if (!e) return 6144u;
                        int v = std::atoi(e);
                        return v >= 1 ? uint32_t(v) : 6144u;
                    }();
                    if (!moe_onednn_off && T >= moe_onednn_minT && w.moe_btf16) {
                        const std::vector<uint32_t>& off = pk.expert_offsets;   // host, [E+1]
                        // Stage 1: per-expert gate + up GEMMs (Kd=H, Nd=EF, both Q4_K).
                        for (uint32_t e = 0; e < E; ++e) {
                            const uint32_t o = off[e]; const uint32_t n_e = off[e + 1] - o;
                            if (n_e == 0) continue;
                            const sycl::half* x_e = w.moe_xp + uint64_t(o) * H;
                            auto dg = dequant_q4_K_to_Bt(q,
                                (const uint8_t*)lw.gate_exps + uint64_t(e) * lw.gate_stride,
                                w.moe_btf16, H, EF, {});
                            gemm_fp16_onednn(q, x_e, w.moe_btf16,
                                w.moe_h + uint64_t(o) * EF, n_e, EF, H, {dg});
                            auto du = dequant_q4_K_to_Bt(q,
                                (const uint8_t*)lw.up_exps + uint64_t(e) * lw.up_stride,
                                w.moe_btf16, H, EF, {});
                            gemm_fp16_onednn(q, x_e, w.moe_btf16,
                                w.moe_h2 + uint64_t(o) * EF, n_e, EF, H, {du});
                        }
                        swiglu(q, w.moe_h, w.moe_h2, w.moe_h, uint64_t(TK) * EF);
                        // Stage 2: per-expert down GEMM (Kd=EF, Nd=H), dtype per layer
                        // (Q6_K/Q4_K mixed — wrong kernel → garbage scales → NaN).
                        for (uint32_t e = 0; e < E; ++e) {
                            const uint32_t o = off[e]; const uint32_t n_e = off[e + 1] - o;
                            if (n_e == 0) continue;
                            const sycl::half* h_e = w.moe_h + uint64_t(o) * EF;
                            const void* dptr = (const uint8_t*)lw.down_exps + uint64_t(e) * lw.down_stride;
                            auto dd = (lw.down_dt == DType::kQ6_K)
                                ? dequant_q6_K_to_Bt(q, dptr, w.moe_btf16, EF, H, {})
                                : dequant_q4_K_to_Bt(q, dptr, w.moe_btf16, EF, H, {});
                            gemm_fp16_onednn(q, h_e, w.moe_btf16,
                                w.moe_out + uint64_t(o) * H, n_e, H, EF, {dd});
                        }
                    } else {
                    // 4) stage 1: gate+up+silu (int-dot W4A8; H=2048%512==0).
                    quantize_q8_1s(q, w.moe_xp, w.moe_xp_q8, uint64_t(TK) * H);
                    moe_prefill_gate_up_silu_q4k_q8(q, w.moe_xp_q8, lw.gate_exps, lw.up_exps,
                                                    w.moe_offsets, w.moe_h,
                                                    E, H, EF, lw.gate_stride, /*soa=*/false);
                    // 5) stage 2: down. The Q4_K_M GGUF mixes ffn_down precision per
                    // layer (Q6_K on blk.0-5/8/11/.../41-47, Q4_K on the rest — verified
                    // via ie-inspect), so the down kernel MUST dispatch on lw.down_dt
                    // (mirror qwen3moe.cpp). Calling the Q6_K kernel on a Q4_K tensor
                    // misreads the blocks (144B vs 210B) → garbage scales → 65504 NaN
                    // (this was the L6+ blow-up). Default = int-dot W4A8 (validated " Paris",
                    // pp512 224→566 tok/s = 2.5× vs fp16); opt OUT via IE_QWEN3NEXT_NO_Q8=1.
                    static const bool down_q8 = std::getenv("IE_QWEN3NEXT_NO_Q8") == nullptr;
                    if (down_q8) quantize_q8_1s(q, w.moe_h, w.moe_h_q8, uint64_t(TK) * EF);
                    if (lw.down_dt == DType::kQ6_K) {
                        if (down_q8)
                            moe_prefill_down_q6k_q8_gen(q, w.moe_h_q8, lw.down_exps, w.moe_offsets,
                                                        w.moe_out, E, H, EF, lw.down_stride, /*soa=*/false);
                        else
                            moe_prefill_down_packed_q6k(q, w.moe_h, lw.down_exps, w.moe_offsets,
                                                        w.moe_out, E, H, EF, lw.down_stride);
                    } else {
                        if (down_q8)
                            moe_prefill_down_q4k_q8_gen(q, w.moe_h_q8, lw.down_exps, w.moe_offsets,
                                                        w.moe_out, E, H, EF, lw.down_stride, /*soa=*/false);
                        else
                            moe_prefill_down_packed_q4k(q, w.moe_h, lw.down_exps, w.moe_offsets,
                                                        w.moe_out, E, H, EF, lw.down_stride);
                    }
                    }
                    // 6) reduce per token → moe_y[t] = Σ_kslot weight·out[tk_to_packed[t,kslot]].
                    // moe_prefill_reduce ACCUMULATES (acc = y[...]; acc += …) — qwen3moe
                    // relies on the shared expert writing y first, but qwen3next adds the
                    // shexp AFTER the reduce, so moe_y MUST be zeroed here or it carries the
                    // previous layer's value (L0 ok / L1+ inflated — bug #4).
                    q.memset(w.moe_y, 0, uint64_t(T) * H * sizeof(sycl::half));
                    moe_prefill_reduce(q, w.moe_out,
                                       reinterpret_cast<const uint32_t*>(w.moe_tk_to_packed),
                                       w.moe_wpk, w.moe_y, T, K, H);
                } else {
                    // T==1 decode: upload top-K idx/weights, 2-launch fused pair.
                    std::vector<int32_t>    idx(K);
                    std::vector<sycl::half> wt(K);
                    for (uint32_t k = 0; k < K; ++k) {
                        idx[k] = int32_t(routes[0][k].first);
                        wt[k]  = sycl::half(routes[0][k].second);
                    }
                    q.memcpy(w.moe_topk_idx, idx.data(), K * sizeof(int32_t));
                    q.memcpy(w.moe_topk_w,   wt.data(),  K * sizeof(sycl::half)).wait();
                    moe_decode_gate_up_silu_q4k(q, w.x_normed, lw.gate_exps, lw.up_exps,
                                                w.moe_topk_idx, w.moe_h, H, EF, K,
                                                lw.gate_stride, /*soa=*/false);
                    q.memset(w.moe_y, 0, uint64_t(H) * sizeof(sycl::half));
                    // down dispatches on lw.down_dt — the GGUF mixes Q6_K/Q4_K per layer
                    // (see the prefill branch above); a fixed Q6_K kernel NaNs the Q4_K layers.
                    if (lw.down_dt == DType::kQ6_K)
                        moe_decode_down_q6k(q, w.moe_h, lw.down_exps, w.moe_topk_idx, w.moe_topk_w,
                                            w.moe_y, H, EF, K, lw.down_stride, /*soa=*/false);
                    else
                        moe_decode_down_q4k(q, w.moe_h, lw.down_exps, w.moe_topk_idx, w.moe_topk_w,
                                            w.moe_y, H, EF, K, lw.down_stride, /*soa=*/false);
                }

                mark(q, pf_moe);   // MoE compute (router→pack→gate/up/down→reduce)
                // ---- shared expert (always on; qwen36.cpp:1516-1572) ----
                // g_sh = σ(x_normed · ffn_gate_inp_shexp) per token; y_sh =
                // (silu(x_normed@gate_shexp) ⊙ (x_normed@up_shexp)) @ down_shexp;
                // moe_y += g_sh · y_sh (per-token σ scalar, scaled_add_per_token_row).
                if (cfg_.shared_expert_ffn != 0) {
                    shared_expert_gate(q, w.x_normed, lw.ffn_gate_inp_shexp, w.sh_g, T, H);
                    dense::gemv_q_T(q, w.x_normed, lw.gate_shexp, w.sh_gate, H, EF, T);
                    dense::gemv_q_T(q, w.x_normed, lw.up_shexp,   w.sh_up,   H, EF, T);
                    swiglu(q, w.sh_gate, w.sh_up, w.sh_h, uint64_t(T) * EF);
                    dense::gemv_q_T(q, w.sh_h, lw.down_shexp, w.sh_eo, EF, H, T);
                    scaled_add_per_token_row(q, w.sh_eo, w.sh_g, w.moe_y, T, H);
                }

                // 7) residual: x += FFN output (MoE + shared expert).
                residual_add(q, w.x, w.moe_y, w.x, uint64_t(T) * H);
                mark(q, pf_shexp);   // shared expert + final residual
            }
        }
        q.wait();   // finish this card before the boundary copy reads its ws.x
    }

    // final norm + lm_head on head_dev → last token's logits → host.
    {
        const uint32_t hd = plan_.head_dev;
        Workspace& w = ws_[hd];
        auto& alloc = fleet_->dev(hd);
        auto& q = alloc.queue();
        rms_norm_f32w(q, w.x, output_norm_, w.x_normed, T, H, eps);
        const sycl::half* last = w.x_normed + uint64_t(T - 1) * H;
        sycl::half* d_logits = static_cast<sycl::half*>(alloc.malloc(uint64_t(V) * sizeof(sycl::half)));
        if (!d_logits) return "logits alloc failed";
        DenseQuantPtr lm{output_, output_dtype_, output_suh_, output_svh_, output_bits_};
        dense::gemv_q(q, last, lm, d_logits, H, V).wait();
        q.memcpy(out_logits_host, d_logits, uint64_t(V) * sizeof(sycl::half)).wait();
        alloc.free(d_logits);
    }

    if (prof) {
        pf_total += pf_ms(pf_t_total, pf_clock::now());
        pf_tok += T; pf_calls++;
        // lm_head + embed + boundary copies + instrumentation gaps land in "other".
        const double summed = pf_attn + pf_router + pf_hostcpu + pf_moe + pf_shexp;
        pf_other = pf_total - summed;
        if (pf_calls % 16 == 0) {
            auto pct = [&](double x) { return pf_total > 0 ? 100.0 * x / pf_total : 0.0; };
            std::fprintf(stderr,
                "\n[qwen3next PROFILE] calls=%llu tok=%llu  (wait-bracketed; relative breakdown)\n"
                "  total        %8.2f ms  (%.2f ms/tok = %.1f tok/s instrumented)\n"
                "  attn/deltanet%8.2f ms  %5.1f%%\n"
                "  router gemm+sync %5.2f ms %5.1f%%   <- GPU drain at the host bounce\n"
                "  router HOST cpu  %5.2f ms %5.1f%%   <- pure CPU softmax/top-K (no wait)\n"
                "  MoE compute  %8.2f ms  %5.1f%%\n"
                "  shared expert%8.2f ms  %5.1f%%\n"
                "  other(embed/lm_head/boundary) %5.2f ms %5.1f%%\n",
                (unsigned long long)pf_calls, (unsigned long long)pf_tok,
                pf_total, pf_total / double(pf_tok), 1000.0 * double(pf_tok) / pf_total,
                pf_attn, pct(pf_attn), pf_router, pct(pf_router),
                pf_hostcpu, pct(pf_hostcpu), pf_moe, pct(pf_moe),
                pf_shexp, pct(pf_shexp), pf_other, pct(pf_other));
        }
    }
    return {};
}

}  // namespace ie
