// src/model/qwen4exp.cpp — Qwen3.8-Flash-Next (`qwen4exp`) loader (v0).
//
// Placement contract in include/ie/qwen4exp.hpp; op ground truth in
// docs/qwen4/1*.md. This translation unit is LOAD ONLY — the forward lands on
// top of it (campaign P2) so the placement can gate first (alpha-omega).

#include "ie/qwen4exp.hpp"
#include <chrono>

#include "ie/dequant_ref.hpp"
#include "ie/dtype.hpp"
#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"
#include "ie/qwen4_hc.hpp"
#include "ie/deepseek4_attn.hpp"
#include "ie/kernel_profiler.hpp"
#include "ie/qwen4_quant.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <utility>

namespace ie {

namespace {

inline float bf16_to_fp32(uint16_t v) {
    uint32_t u = uint32_t(v) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

// Dequantize one whole GGUF tensor to fp32, GGUF element order. Returns error
// text on unsupported dtype. Covers every dtype the real UD-Q4_K_XL file uses
// for DEVICE-bound tensors (F32/F16/BF16/Q8_0) plus the K-quants/Q5_1 so a
// differently-quantized qwen4exp GGUF still loads.
std::string dequant_tensor_fp32(const GgufTensorInfo* t, std::vector<float>& out) {
    uint64_t n = 1;
    for (uint32_t d = 0; d < t->n_dims; ++d) n *= t->shape[d];
    out.resize(n);
    switch (t->dtype) {
        case DType::kF32:
            std::memcpy(out.data(), t->data, n * sizeof(float));
            return {};
        case DType::kF16: {
            const auto* s = reinterpret_cast<const uint16_t*>(t->data);
            for (uint64_t i = 0; i < n; ++i)
                out[i] = float(sycl::bit_cast<sycl::half>(s[i]));
            return {};
        }
        case DType::kBF16: {
            const auto* s = reinterpret_cast<const uint16_t*>(t->data);
            for (uint64_t i = 0; i < n; ++i) out[i] = bf16_to_fp32(s[i]);
            return {};
        }
        case DType::kQ8_0:  ref::dequant_q8_0_buffer(t->data, n, out.data()); return {};
        case DType::kQ4_K:  ref::dequant_q4_K_buffer(t->data, n, out.data()); return {};
        case DType::kQ5_K:  ref::dequant_q5_K_buffer(t->data, n, out.data()); return {};
        case DType::kQ6_K:  ref::dequant_q6_K_buffer(t->data, n, out.data()); return {};
        case DType::kQ5_1:  ref::dequant_q5_1_buffer(t->data, n, out.data()); return {};
        default:
            return std::string("qwen4exp load: unsupported device-tensor dtype ") +
                   std::string(type_name(t->dtype));
    }
}

}  // namespace

Qwen4ExpModel::~Qwen4ExpModel() { free_all(); }

void Qwen4ExpModel::free_all() {
    if (!alloc_) return;
    for (void* p : owned_) if (p) alloc_->free(p);
    owned_.clear();
    layers_.clear();
    alloc_ = nullptr;
}

std::string Qwen4ExpModel::load(DeviceAllocator& alloc, const GgufReader& g,
                                const Qwen4ExpConfig& cfg, uint64_t vram_budget_bytes,
                                uint32_t layer_lo, uint32_t layer_hi) {
    alloc_ = &alloc;
    cfg_   = cfg;
    layers_.assign(cfg.n_layers, {});
    layer_lo_ = std::min(layer_lo, cfg.n_layers);
    layer_hi_ = std::min(layer_hi, cfg.n_layers);
    if (layer_lo_ >= layer_hi_) return "qwen4exp load: empty layer range";
    if (vram_budget_bytes == 0) vram_budget_bytes = 28ull << 30;

    char buf[80];
    auto Tl = [&](uint32_t L, const char* n) -> const GgufTensorInfo* {
        std::snprintf(buf, sizeof(buf), "blk.%u.%s", L, n);
        return g.find_tensor(buf);
    };
    std::string err;

    // ---- pass 1: bind every tensor + project device residency BEFORE any
    // upload, so the VRAM guard fires with zero bytes allocated. -------------
    // Layout classes (see the header comment on Qwen4ExpLayer):
    //  kNative   — GGUF element order as-is (HC mats, token_embd, 1-D vectors)
    //  kTransKN  — transposed to [K, N] row-major for gemv_fp16
    //  kSsmPad64 — qwen35 ssm alpha/beta [K, 64]-padded transpose
    enum class Lay { kNative, kTransKN, kSsmPad64, kConvKMajor, kQ8Soa,
                     kQ8Requant };
    struct Job {
        const GgufTensorInfo* t;
        void**  dst_h;     // exactly one of dst_h / dst_f / dst_q8 is set
        float** dst_f;
        Qwen4ExpLayer::Q8W* dst_q8 = nullptr;
        Lay     lay = Lay::kNative;
    };
    std::vector<Job> jobs;
    uint64_t projected = 0;

    // Q8_0 tensors go SoA-native (no F16 copy); other dtypes fall back to
    // the F16 job so any qwen4exp quant still loads.
    // Q8-SoA residency defaults ON since the 2026-08-27 PPL gate certified
    // the trade (streaming +0.42% / chunked −0.90% vs F16; spec lossless +
    // bvs bit-exact under Q8 — the old 1.4e-2 flag was q8 ACTIVATIONS on a
    // parity stage tol, not corpus quality). IE_Q4E_DENSE_Q8=0 opts out.
    const bool q8_dense = qwen4exp_dense_q8();
    auto want_q8 = [&](const GgufTensorInfo* t, const char* name,
                       Qwen4ExpLayer::Q8W* q8dst, sycl::half** h_fallback,
                       Lay fb_lay) -> bool {
        if (!t) { err = std::string("qwen4exp load: missing tensor ") + name; return false; }
        if (!q8_dense) {
            uint64_t n = 1;
            for (uint32_t d2 = 0; d2 < t->n_dims; ++d2) n *= t->shape[d2];
            projected += n * sizeof(sycl::half);
            placement_.push_back({std::string(t->name), "device", "F16", n * 2});
            jobs.push_back({t, reinterpret_cast<void**>(h_fallback), nullptr, nullptr, fb_lay});
            return true;
        }
        // Q8_0 on disk repacks byte-faithfully; any other dtype REQUANTIZES
        // at load (dequant -> per-32-block Q8_0 -> SoA) so the whole dense
        // set shares the int-dot decode/verify path regardless of the file's
        // per-tensor quant mix (2026-08-27 profile: the F16 remainder was
        // 15 of 73 GPU-ms per verify forward).
        const uint64_t K = t->shape[0], N = t->shape[1];
        // qs + the two scale planes (n-major d for decode, KB-major dt for
        // the oneDNN s8 prefill GEMM).
        const uint64_t bytes = N * K + 2 * N * (K / 32) * 2;
        projected += bytes;
        const bool native = t->dtype == DType::kQ8_0;
        placement_.push_back({std::string(t->name), "device",
                              native ? "Q8soa" : "Q8rq", bytes});
        jobs.push_back({t, nullptr, nullptr, q8dst,
                        native ? Lay::kQ8Soa : Lay::kQ8Requant});
        return true;
    };
    auto want = [&](const GgufTensorInfo* t, const char* name,
                    sycl::half** h, float** f, Lay lay = Lay::kNative) -> bool {
        if (!t) { err = std::string("qwen4exp load: missing tensor ") + name; return false; }
        uint64_t n = 1;
        for (uint32_t d = 0; d < t->n_dims; ++d) n *= t->shape[d];
        if (lay == Lay::kSsmPad64) n = t->shape[0] * 64;   // [K,48] -> [K,64] pad
        const uint64_t bytes = n * (h ? sizeof(sycl::half) : sizeof(float));
        projected += bytes;
        placement_.push_back({std::string(t->name), "device", h ? "F16" : "F32", bytes});
        jobs.push_back({t, reinterpret_cast<void**>(h), f, nullptr, lay});
        return true;
    };
    auto host_bank = [&](uint32_t L, const char* n,
                         const GgufTensorInfo** dst) -> bool {
        const GgufTensorInfo* t = Tl(L, n);
        if (!t) { err = std::string("qwen4exp load: missing expert bank blk.") +
                        std::to_string(L) + "." + n; return false; }
        *dst = t;
        host_bytes_ += t->nbytes;
        placement_.push_back({std::string(t->name), "host", type_name(t->dtype).data(), t->nbytes});
        return true;
    };

    // Globals by role: head-of-pipe owns the embedding (+ PLE below when it
    // owns the PLE block); tail-of-pipe owns the final merge + lm_head.
    if (layer_lo_ == 0) {
        if (!want(g.find_tensor("token_embd.weight"), "token_embd.weight", &token_embd, nullptr)) return err;
    }
    if (layer_hi_ == cfg.n_layers) {
        if (!want_q8(g.find_tensor("output.weight"), "output.weight", &lmh_q8,
                     &lm_head, Lay::kTransKN)) return err;
        if (!want(g.find_tensor("output_hc_up.weight"), "output_hc_up.weight", &out_hc_up, nullptr)) return err;
        if (!want(g.find_tensor("output_hc_down.weight"), "output_hc_down.weight", &out_hc_down, nullptr)) return err;
        if (!want(g.find_tensor("output_hc_norm.weight"), "output_hc_norm.weight", nullptr, &out_hc_norm)) return err;
    }
    bool own_ple = false;
    for (int32_t l : cfg.ple_layers)
        if (uint32_t(l) >= layer_lo_ && uint32_t(l) < layer_hi_) own_ple = true;
    if (cfg.has_ple() && own_ple) {
        ple_table = g.find_tensor("per_layer_token_embd.weight");
        if (!ple_table) return "qwen4exp load: missing per_layer_token_embd.weight";
        host_bytes_ += ple_table->nbytes;
        placement_.push_back({std::string(ple_table->name), "host",
                              type_name(ple_table->dtype).data(), ple_table->nbytes});
    }

    for (uint32_t L = layer_lo_; L < layer_hi_; ++L) {
        Qwen4ExpLayer& w = layers_[L];
        // HC — every block.
        if (!want(Tl(L, "hc_attn_up.weight"),   "hc_attn_up",   &w.hc_attn_up,   nullptr)) return err;
        if (!want(Tl(L, "hc_attn_down.weight"), "hc_attn_down", &w.hc_attn_down, nullptr)) return err;
        if (!want(Tl(L, "hc_attn_norm.weight"), "hc_attn_norm", nullptr, &w.hc_attn_norm)) return err;
        if (!want(Tl(L, "hc_attn_inject.weight"), "hc_attn_inject", nullptr, &w.hc_attn_inject)) return err;
        if (!want(Tl(L, "hc_ffn_up.weight"),   "hc_ffn_up",   &w.hc_ffn_up,   nullptr)) return err;
        if (!want(Tl(L, "hc_ffn_down.weight"), "hc_ffn_down", &w.hc_ffn_down, nullptr)) return err;
        if (!want(Tl(L, "hc_ffn_norm.weight"), "hc_ffn_norm", nullptr, &w.hc_ffn_norm)) return err;
        if (!want(Tl(L, "hc_ffn_inject.weight"), "hc_ffn_inject", nullptr, &w.hc_ffn_inject)) return err;
        // MoE device side — every block.
        if (!want(Tl(L, "ffn_gate_inp.weight"), "ffn_gate_inp", &w.ffn_gate_inp, nullptr, Lay::kTransKN)) return err;
        if (!want(Tl(L, "ffn_gate_inp_shexp.weight"), "ffn_gate_inp_shexp", &w.ffn_gate_inp_shexp, nullptr)) return err;
        if (!want_q8(Tl(L, "ffn_gate_shexp.weight"), "ffn_gate_shexp", &w.shg_q8,
                     &w.ffn_gate_shexp, Lay::kTransKN)) return err;
        if (!want_q8(Tl(L, "ffn_up_shexp.weight"),   "ffn_up_shexp",   &w.shu_q8,
                     &w.ffn_up_shexp,   Lay::kTransKN)) return err;
        if (!want_q8(Tl(L, "ffn_down_shexp.weight"), "ffn_down_shexp", &w.shd_q8,
                     &w.ffn_down_shexp, Lay::kTransKN)) return err;
        // Routed banks — host.
        if (!host_bank(L, "ffn_gate_exps.weight", &w.gate_exps)) return err;
        if (!host_bank(L, "ffn_up_exps.weight",   &w.up_exps))   return err;
        if (!host_bank(L, "ffn_down_exps.weight", &w.down_exps)) return err;

        if (cfg.is_full_attn(L)) {
            if (!want_q8(Tl(L, "attn_q.weight"),   "attn_q",      &w.q_q8,
                         &w.attn_q, Lay::kTransKN)) return err;
            if (!want(Tl(L, "attn_k.weight"),      "attn_k",      &w.attn_k,      nullptr, Lay::kTransKN)) return err;
            if (!want(Tl(L, "attn_v.weight"),      "attn_v",      &w.attn_v,      nullptr, Lay::kTransKN)) return err;
            if (!want_q8(Tl(L, "attn_output.weight"), "attn_output", &w.attnout_q8,
                         &w.attn_output, Lay::kTransKN)) return err;
            if (!want(Tl(L, "attn_q_norm.weight"), "attn_q_norm", nullptr, &w.attn_q_norm)) return err;
            if (!want(Tl(L, "attn_k_norm.weight"), "attn_k_norm", nullptr, &w.attn_k_norm)) return err;
            if (!want(Tl(L, "indexer.q_proj.weight"), "indexer.q_proj", &w.idx_q_proj, nullptr, Lay::kTransKN)) return err;
            if (!want(Tl(L, "indexer.k_proj.weight"), "indexer.k_proj", &w.idx_k_proj, nullptr, Lay::kTransKN)) return err;
            if (!want(Tl(L, "indexer.q_norm.weight"), "indexer.q_norm", nullptr, &w.idx_q_norm)) return err;
            if (!want(Tl(L, "indexer.k_norm.weight"), "indexer.k_norm", nullptr, &w.idx_k_norm)) return err;
        } else {
            if (!want_q8(Tl(L, "attn_qkv.weight"), "attn_qkv", &w.qkv_q8,
                         &w.attn_qkv, Lay::kTransKN)) return err;
            if (!want_q8(Tl(L, "attn_gate.weight"), "attn_gate", &w.gate_q8,
                         &w.attn_gate, Lay::kTransKN)) return err;
            if (!want(Tl(L, "ssm_alpha.weight"), "ssm_alpha", &w.ssm_alpha, nullptr, Lay::kSsmPad64)) return err;
            if (!want(Tl(L, "ssm_beta.weight"),  "ssm_beta",  &w.ssm_beta,  nullptr, Lay::kSsmPad64)) return err;
            if (!want(Tl(L, "ssm_a"),            "ssm_a",     nullptr, &w.ssm_a)) return err;
            if (!want(Tl(L, "ssm_dt.bias"),      "ssm_dt.bias", nullptr, &w.ssm_dt)) return err;
            if (!want(Tl(L, "ssm_conv1d.weight"), "ssm_conv1d", &w.ssm_conv, nullptr)) return err;
            if (!want(Tl(L, "ssm_norm.weight"),  "ssm_norm",  &w.ssm_norm, nullptr)) return err;
            if (!want_q8(Tl(L, "ssm_out.weight"), "ssm_out", &w.out_q8,
                         &w.ssm_out, Lay::kTransKN)) return err;
        }
        if (cfg.is_ple_layer(L)) {
            if (!want(Tl(L, "ple_key.weight"),   "ple_key",   &w.ple_key,   nullptr, Lay::kTransKN)) return err;
            if (!want(Tl(L, "ple_value.weight"), "ple_value", &w.ple_value, nullptr, Lay::kTransKN)) return err;
            if (!want(Tl(L, "ple_conv1d.weight"), "ple_conv1d", nullptr, &w.ple_conv, Lay::kConvKMajor)) return err;
            if (!want(Tl(L, "ple_norm_query.weight"), "ple_norm_query", nullptr, &w.ple_norm_query)) return err;
            if (!want(Tl(L, "ple_norm_key.weight"),   "ple_norm_key",   nullptr, &w.ple_norm_key)) return err;
            if (!want(Tl(L, "ple_norm_conv.weight"),  "ple_norm_conv",  nullptr, &w.ple_norm_conv)) return err;
        }
    }

    // ---- VRAM guard: projected weights only (KV/workspaces come later and
    // small); refuse before the first byte moves. ----------------------------
    if (projected > vram_budget_bytes && std::getenv("IE_ALLOW_OOM") == nullptr) {
        std::snprintf(buf, sizeof(buf), "%.2f GiB > budget %.2f GiB",
                      projected / 1073741824.0, vram_budget_bytes / 1073741824.0);
        return std::string("qwen4exp load: projected device weights ") + buf +
               " (IE_ALLOW_OOM=1 to attempt anyway)";
    }

    // ---- pass 2: dequant + upload. -----------------------------------------
    std::vector<float>      f32;
    std::vector<sycl::half> h16;
    sycl::queue& q = alloc.queue();
    // Shared Q8-SoA upload (kQ8Soa + kQ8Requant): qs + n-major d, plus the
    // KB-major dt plane the oneDNN s8 prefill GEMM needs (see Q8W).
    auto upload_q8 = [&](const std::vector<int8_t>& qs,
                         const std::vector<uint16_t>& dd,
                         uint64_t N, uint32_t bpc,
                         Qwen4ExpLayer::Q8W* dst) -> std::string {
        std::vector<uint16_t> dt(dd.size());
        for (uint64_t nn = 0; nn < N; ++nn)
            for (uint32_t b = 0; b < bpc; ++b)
                dt[uint64_t(b) * N + nn] = dd[nn * bpc + b];
        auto* dqs = static_cast<int8_t*>(alloc.malloc(qs.size()));
        auto* ddd = static_cast<uint16_t*>(alloc.malloc(dd.size() * 2));
        auto* ddt = static_cast<uint16_t*>(alloc.malloc(dt.size() * 2));
        if (!dqs || !ddd || !ddt) return "qwen4exp load: q8 soa alloc failed";
        q.memcpy(dqs, qs.data(), qs.size()).wait();
        q.memcpy(ddd, dd.data(), dd.size() * 2).wait();
        q.memcpy(ddt, dt.data(), dt.size() * 2).wait();
        owned_.push_back(dqs); owned_.push_back(ddd); owned_.push_back(ddt);
        dst->qs = dqs; dst->d = ddd; dst->dt = ddt;
        dev_bytes_ += qs.size() + dd.size() * 2 + dt.size() * 2;
        return {};
    };
    for (const Job& j : jobs) {
        if (j.lay == Lay::kQ8Soa) {
            // COPY of qwen35_split build_split's Q8_0 SoA repack (layout must
            // match gemv_q8_0_soa_q8_*): qs[n*K+b*32+i], d[n*bpc+b].
            const uint32_t K = uint32_t(j.t->shape[0]), N = uint32_t(j.t->shape[1]);
            const uint32_t bpc = K / 32;
            const auto* blocks = reinterpret_cast<const block_q8_0*>(j.t->data);
            std::vector<int8_t>   qs(uint64_t(N) * K);
            std::vector<uint16_t> dd(uint64_t(N) * bpc);
            for (uint64_t nn = 0; nn < N; ++nn)
                for (uint32_t b = 0; b < bpc; ++b) {
                    const block_q8_0& blk = blocks[nn * bpc + b];
                    dd[nn * bpc + b] = *reinterpret_cast<const uint16_t*>(&blk.d);
                    for (int i2 = 0; i2 < 32; ++i2)
                        qs[nn * K + uint64_t(b) * 32 + i2] = blk.qs[i2];
                }
            if (auto e = upload_q8(qs, dd, N, bpc, j.dst_q8); !e.empty()) return e;
            continue;
        }
        if (j.lay == Lay::kQ8Requant) {
            // Requantize-at-load: dequant whatever the file stores, then
            // per-32-block Q8_0 in the same SoA layout as kQ8Soa. The block
            // is quantized against the HALF-rounded d (the value the kernels
            // dequant with), so the stored pair round-trips maximally.
            if (auto e = dequant_tensor_fp32(j.t, f32); !e.empty()) return e;
            const uint32_t K = uint32_t(j.t->shape[0]), N = uint32_t(j.t->shape[1]);
            const uint32_t bpc = K / 32;
            std::vector<int8_t>   qs(uint64_t(N) * K);
            std::vector<uint16_t> dd(uint64_t(N) * bpc);
            for (uint64_t nn = 0; nn < N; ++nn)
                for (uint32_t b = 0; b < bpc; ++b) {
                    const float* xb = f32.data() + nn * K + uint64_t(b) * 32;
                    float amax = 0.f;
                    for (int i2 = 0; i2 < 32; ++i2)
                        amax = std::max(amax, std::abs(xb[i2]));
                    const sycl::half dh(amax / 127.f);
                    const float df = float(dh);
                    const float id = df != 0.f ? 1.f / df : 0.f;
                    dd[nn * bpc + b] = *reinterpret_cast<const uint16_t*>(&dh);
                    for (int i2 = 0; i2 < 32; ++i2) {
                        const float v = std::round(xb[i2] * id);
                        qs[nn * K + uint64_t(b) * 32 + i2] =
                            int8_t(std::max(-127.f, std::min(127.f, v)));
                    }
                }
            if (auto e = upload_q8(qs, dd, N, bpc, j.dst_q8); !e.empty()) return e;
            continue;
        }
        if (auto e = dequant_tensor_fp32(j.t, f32); !e.empty()) return e;
        uint64_t n = f32.size();
        void* d = nullptr;
        if (j.dst_f) {                       // F32 tensor
            if (j.lay == Lay::kConvKMajor && j.t->n_dims == 2) {
                // GGUF [K_conv, C] stores (k, c) at c*K+k; the qwen4_ple conv
                // kernel contract is k-major W[k*C + c] — transpose here.
                const uint64_t Kc = j.t->shape[0], C = j.t->shape[1];
                std::vector<float> tr(n);
                for (uint64_t c2 = 0; c2 < C; ++c2)
                    for (uint64_t k = 0; k < Kc; ++k)
                        tr[k * C + c2] = f32[c2 * Kc + k];
                f32.swap(tr);
            }
            d = alloc.malloc(n * sizeof(float));
            if (!d) return "qwen4exp load: device malloc failed (f32)";
            q.memcpy(d, f32.data(), n * sizeof(float)).wait();
            *j.dst_f = static_cast<float*>(d);
            dev_bytes_ += n * sizeof(float);
        } else if (j.lay == Lay::kSsmPad64) {  // ssm alpha/beta: [N=48,K] -> [K,64]
            const uint64_t K = j.t->shape[0], N = j.t->shape[1];
            h16.assign(K * 64, sycl::half(0.f));
            for (uint64_t nn = 0; nn < N; ++nn)
                for (uint64_t k = 0; k < K; ++k)
                    h16[k * 64 + nn] = sycl::half(f32[nn * K + k]);
            d = alloc.malloc(K * 64 * sizeof(sycl::half));
            if (!d) return "qwen4exp load: device malloc failed (ssm proj)";
            q.memcpy(d, h16.data(), K * 64 * sizeof(sycl::half)).wait();
            *j.dst_h = d;
            dev_bytes_ += K * 64 * sizeof(sycl::half);
        } else if (j.lay == Lay::kTransKN && j.t->n_dims == 2) {
            // gemv_fp16 layout: W[k*N+n] from GGUF's W[n*K+k].
            const uint64_t K = j.t->shape[0], N = j.t->shape[1];
            h16.resize(n);
            for (uint64_t nn = 0; nn < N; ++nn)
                for (uint64_t k = 0; k < K; ++k)
                    h16[k * N + nn] = sycl::half(f32[nn * K + k]);
            d = alloc.malloc(n * sizeof(sycl::half));
            if (!d) return "qwen4exp load: device malloc failed (f16)";
            q.memcpy(d, h16.data(), n * sizeof(sycl::half)).wait();
            *j.dst_h = d;
            dev_bytes_ += n * sizeof(sycl::half);
        } else {                             // kNative F16 (HC mats, embd, 1-D)
            h16.resize(n);
            for (uint64_t i = 0; i < n; ++i) h16[i] = sycl::half(f32[i]);
            d = alloc.malloc(n * sizeof(sycl::half));
            if (!d) return "qwen4exp load: device malloc failed (f16)";
            q.memcpy(d, h16.data(), n * sizeof(sycl::half)).wait();
            *j.dst_h = d;
            dev_bytes_ += n * sizeof(sycl::half);
        }
        owned_.push_back(d);
    }
    return {};
}

}  // namespace ie

// ============================ runtime (v0) =================================
// Correctness-first single-GPU forward. Op chain per docs/qwen4/1*.md:
// wide-residual hyper-connections everywhere (no plain norms), DeltaNet via
// the certified qwen35 kernels (sigmoid output gate), EXACT dense attention
// (== QSA while ctx <= top_k + ratio - 1), softmax-top10 MoE with host-
// streamed expert slices, PLE at its ple.layers block.

namespace ie {

namespace {

// Dense-W8 leaf family: A8 int-dot (default) vs A16 (IE_Q4E_A16=1 — F16
// activation, quality-neutral W8: streaming PPL 6.689 == the F16 baseline;
// the +1% A8 cost is activation quantization). A16 default was tried
// 2026-08-27 and pulled back the same day: its naive leaves cost 8-20
// ms/round and dropped the chat-domain decode to 24 t/s, under the
// certified 30 floor (A8 band 30.1-35.6). Flip the default once the A16
// claw-back (DN-pair dual, occupancy at N=640, vectorized x loads) closes
// the gap. Both families' batched rows are per-row bit-identical to their
// own T=1 leaf, so the sv lossless contract holds under either.
static bool dense_a16() {
    static const bool v = [] {
        const char* e = std::getenv("IE_Q4E_A16");
        return e && e[0] == '1';
    }();
    return v;
}

// Opt-out for the T>16 HC-mix gemm path (falls back to mix v2 per-token).
static bool hc_prefill_gemm() {
    static const bool off = std::getenv("IE_Q4E_NO_HC_PREFILL") != nullptr;
    return !off;
}

// COPY of qwen35_split.cpp:extract_cols (copy-not-hoist discipline).
inline sycl::event extract_cols(sycl::queue& q, const sycl::half* src, sycl::half* dst,
                                uint32_t T, uint32_t nh, uint32_t src_stride) {
    return q.parallel_for(sycl::range<1>(uint64_t(T) * nh), [=](sycl::id<1> i) {
        const uint32_t t = uint32_t(i) / nh, h = uint32_t(i) % nh;
        dst[uint64_t(t) * nh + h] = src[uint64_t(t) * src_stride + h];
    });
}

// COPY of qwen3next.cpp:route_from_logits (copy-not-hoist): softmax over all
// E fp32, top-k by prob, renormalize to sum 1, sort ascending by expert id.
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

// emb[t, :] = table[ids[t], :]  (F16 rows, GGUF-native token_embd layout).
inline sycl::event gather_embed_rows(sycl::queue& q, const sycl::half* table,
                                     const int32_t* ids, sycl::half* emb,
                                     uint32_t T, uint32_t H) {
    return q.parallel_for(sycl::range<1>(uint64_t(T) * H), [=](sycl::id<1> i) {
        const uint32_t t = uint32_t(i) / H, h = uint32_t(i) % H;
        emb[uint64_t(t) * H + h] = table[uint64_t(ids[t]) * H + h];
    });
}

// acc[i] += w * y[i]   (routed-expert accumulation, fp32 accumulator).
inline sycl::event axpy_f16_to_f32(sycl::queue& q, float* acc, const sycl::half* y,
                                   float w, uint32_t n) {
    return q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        acc[i] += w * float(y[i]);
    });
}

// acc[i] += sigmoid(s[0]) * y[i]  (always-on shared expert, scalar gate).
inline sycl::event sigmoid_scalar_axpy(sycl::queue& q, float* acc, const sycl::half* y,
                                       const sycl::half* s, uint32_t n) {
    return q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        const float sv = float(s[0]);
        acc[i] += (1.f / (1.f + sycl::exp(-sv))) * float(y[i]);
    });
}

inline sycl::event cast_f32_to_f16(sycl::queue& q, const float* x, sycl::half* y,
                                   uint32_t n) {
    return q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        y[i] = sycl::half(x[i]);
    });
}

// ---- QSA indexer kernels (spec docs/qwen4/10_qsa_sparse_attention.md) -----

// Pool 4 raw keys (fp32 mean) -> RMS norm (gamma pre-folded) -> partial rope
// (64 of 128, NEOX half-split pairs (r, r+32), theta) at the block's
// FIRST-token position b*4. Order pool -> norm -> rope is the spec's.
inline sycl::event qsa_pool_norm_rope(sycl::queue& q, const sycl::half* raw,
                                      const float* gamma, float* out,
                                      uint32_t b0, uint32_t nb,
                                      float theta, float eps) {
    return q.parallel_for(sycl::range<1>(nb), [=](sycl::id<1> i) {
        const uint32_t b = b0 + uint32_t(i);
        float v[128];
        for (int d = 0; d < 128; ++d) {
            float sm = 0.f;
            for (int j = 0; j < 4; ++j)
                sm += float(raw[(uint64_t(b) * 4 + j) * 128 + d]);
            v[d] = sm * 0.25f;
        }
        float ss = 0.f;
        for (int d = 0; d < 128; ++d) ss += v[d] * v[d];
        const float r = sycl::rsqrt(ss / 128.f + eps);
        for (int d = 0; d < 128; ++d) v[d] = v[d] * r * gamma[d];
        const float pos = float(b * 4);
        for (int rr = 0; rr < 32; ++rr) {
            const float ang = pos * sycl::exp(-2.f * float(rr) / 64.f * sycl::log(theta));
            const float c = sycl::cos(ang), sn = sycl::sin(ang);
            const float a = v[rr], bb = v[rr + 32];
            v[rr]      = a * c - bb * sn;
            v[rr + 32] = a * sn + bb * c;
        }
        float* o = out + uint64_t(b) * 128;
        for (int d = 0; d < 128; ++d) o[d] = v[d];
    });
}

// scores[b] = sum_h ReLU(q_h . pk_b), fp32. The 1/sqrt(128) is dropped —
// monotonic, rank-invariant (the llama.cpp PR does the same).
inline sycl::event qsa_score(sycl::queue& q, const sycl::half* qh,
                             const float* pk, float* scores, uint32_t nb) {
    return q.parallel_for(sycl::range<1>(nb), [=](sycl::id<1> i) {
        const float* k = pk + uint64_t(i) * 128;
        float s = 0.f;
        for (int h = 0; h < 4; ++h) {
            float d = 0.f;
            for (int j = 0; j < 128; ++j) d += float(qh[h * 128 + j]) * k[j];
            s += sycl::fmax(d, 0.f);
        }
        scores[i] = s;
    });
}

// Chunk-batched QSA selection (2026-08-27): the per-token loop submitted
// score+topk+expand per token (3 x T launches/layer; a 1024-chunk past
// dense_ok paid ~61K submits/chunk across the 12 full-attn layers). These
// run the SAME per-token math over the whole chunk in ONE launch each —
// outputs bit-identical, pure submit-count change.
inline sycl::event qsa_score_batched(sycl::queue& q, const sycl::half* qh_rows,
                                     uint32_t qh_stride, const float* pk,
                                     float* scores, uint32_t s_stride,
                                     const int32_t* pos, uint32_t nb_max,
                                     uint32_t T) {
    return q.parallel_for(sycl::range<2>(T, nb_max), [=](sycl::id<2> id) {
        const uint32_t t = uint32_t(id[0]), i = uint32_t(id[1]);
        const uint32_t nv = uint32_t(pos[t] + 1) / 4;
        if (i >= nv) return;
        const sycl::half* qh = qh_rows + uint64_t(t) * qh_stride;
        const float* k = pk + uint64_t(i) * 128;
        float s = 0.f;
        for (int h = 0; h < 4; ++h) {
            float d = 0.f;
            for (int j = 0; j < 128; ++j) d += float(qh[h * 128 + j]) * k[j];
            s += sycl::fmax(d, 0.f);
        }
        scores[uint64_t(t) * s_stride + i] = s;
    });
}

inline sycl::event qsa_expand_sel_batched(sycl::queue& q,
                                          const int32_t* blk_sel_rows,
                                          uint32_t bs_stride, uint32_t max_blk,
                                          uint32_t kblk_thresh,
                                          const int32_t* pos,
                                          int32_t* sel_rows, uint32_t sel_stride,
                                          int32_t* n_sel_rows, uint32_t T) {
    return q.parallel_for(sycl::range<1>(T), [=](sycl::id<1> ti) {
        const uint32_t t = uint32_t(ti);
        const int32_t p = pos[t];
        const int32_t n_vis = (p + 1) / 4;
        int32_t* sel_out = sel_rows + uint64_t(t) * sel_stride;
        int32_t n = 0;
        if (uint32_t(n_vis) > kblk_thresh) {
            const int32_t* bs = blk_sel_rows + uint64_t(t) * bs_stride;
            for (uint32_t j = 0; j < max_blk; ++j) {
                const int32_t b = bs[j];
                if (b < 0) continue;
                for (int k = 0; k < 4; ++k) sel_out[n++] = b * 4 + k;
            }
        } else {
            for (int32_t b = 0; b < n_vis; ++b)
                for (int k = 0; k < 4; ++k) sel_out[n++] = b * 4 + k;
        }
        for (int32_t p2 = n_vis * 4; p2 <= p; ++p2) sel_out[n++] = p2;
        n_sel_rows[t] = n;
    });
}

// MTP fusion helpers. pre_fc_norm_hidden is ONE ungrouped RMSNorm over the
// full hc*H (10240) carried vector — vLLM mtp.py builds
// GemmaRMSNorm(hidden_size * hc_count) and flattens before applying;
// SGLang qwen4_exp_mtp.py comments "ONE GemmaRMSNorm over the full 10240".
// The per-2560-branch variant this shipped with is NOT the reference
// function whenever branch magnitudes differ (the chained draft steps,
// where accept k1 measurably broke). Gamma indexes the full width either
// way; fp32 in -> f16 out.
inline sycl::event mtp_group_norm_f16(sycl::queue& q, const float* x,
                                      const float* gamma, sycl::half* y,
                                      uint32_t T, uint32_t H, uint32_t hc,
                                      float eps) {
    constexpr int SG = 16;
    return q.parallel_for(sycl::nd_range<1>(uint64_t(T) * SG, SG),
                          [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const uint32_t t   = uint32_t(it.get_group(0));
        const uint32_t lid = uint32_t(it.get_local_id(0));
        auto sg = it.get_sub_group();
        const uint64_t D = uint64_t(hc) * H;
        const float* xr = x + t * D;
        float p = 0.f;
        for (uint64_t i = lid; i < D; i += SG) p += xr[i] * xr[i];
        const float ss = sycl::reduce_over_group(sg, p, sycl::plus<float>());
        const float rs = sycl::rsqrt(ss / float(D) + eps);
        sycl::half* yr = y + t * D;
        for (uint64_t i = lid; i < D; i += SG)
            yr[i] = sycl::half(xr[i] * rs * gamma[i]);
    });
}
// wide[t,s,:] = xh[(t*hc+s),:] + xe[t,:]
inline sycl::event mtp_fuse_add(sycl::queue& q, const sycl::half* xh,
                                const sycl::half* xe, float* wide,
                                uint32_t T, uint32_t H, uint32_t hc) {
    return q.parallel_for(sycl::range<1>(uint64_t(T) * hc * H), [=](sycl::id<1> i) {
        const uint32_t h2 = uint32_t(i % H);
        const uint32_t s  = uint32_t((i / H) % hc);
        const uint32_t t  = uint32_t(i / (uint64_t(H) * hc));
        wide[i] = float(xh[(uint64_t(t) * hc + s) * H + h2]) + float(xe[uint64_t(t) * H + h2]);
    });
}

// Prefill MoE expert-major helpers: gather rows by token index, scatter-
// accumulate weighted expert outputs back to token rows (fp32 accumulator).
inline sycl::event gather_bytes_rows(sycl::queue& q, const uint8_t* src,
                                     uint8_t* dst, const int32_t* idx,
                                     uint32_t n_rows, uint32_t row_bytes) {
    return q.parallel_for(sycl::range<1>(uint64_t(n_rows) * row_bytes),
                          [=](sycl::id<1> i) {
        const uint32_t j = uint32_t(i / row_bytes), b = uint32_t(i % row_bytes);
        dst[uint64_t(j) * row_bytes + b] =
            src[uint64_t(uint32_t(idx[j])) * row_bytes + b];
    });
}
inline sycl::event scatter_axpy_rows(sycl::queue& q, float* acc,
                                     const sycl::half* y, const int32_t* idx,
                                     const float* wgt, uint32_t n_rows, uint32_t H) {
    return q.parallel_for(sycl::range<1>(uint64_t(n_rows) * H), [=](sycl::id<1> i) {
        const uint32_t j = uint32_t(i / H), d = uint32_t(i % H);
        acc[uint64_t(uint32_t(idx[j])) * H + d] += wgt[j] * float(y[uint64_t(j) * H + d]);
    });
}
// acc[t,:] += sigmoid(s[t]) * y[t,:]  (batched shared expert).
inline sycl::event sigmoid_rows_axpy(sycl::queue& q, float* acc, const sycl::half* y,
                                     const sycl::half* sgate, uint32_t T, uint32_t H) {
    return q.parallel_for(sycl::range<1>(uint64_t(T) * H), [=](sycl::id<1> i) {
        const uint32_t t = uint32_t(i / H);
        const float sv = float(sgate[t]);
        acc[i] += (1.f / (1.f + sycl::exp(-sv))) * float(y[i]);
    });
}

// ---- fully-device QSA selection chain (no host round-trips) --------------
// Expand the top-k BLOCK ids (desc order, -1 sentinels) into a token-index
// list: all 4 tokens of each selected block (complete blocks only contain
// tokens <= pos, self included — the reference attends the same set), plus
// the always-visible tail [n_vis*4, pos]. Single tiny serial kernel (the
// list is <= 2052 entries); *n_sel_out is written DEVICE-side so the
// downstream gather/attend need no host sync.
inline sycl::event qsa_expand_sel(sycl::queue& q, const int32_t* blk_sel,
                                  uint32_t max_blk, int32_t pos,
                                  int32_t* sel_out, int32_t* n_sel_out) {
    return q.single_task([=]() {
        int32_t n = 0;
        const int32_t n_vis = (pos + 1) / 4;
        if (blk_sel) {
            for (uint32_t j = 0; j < max_blk; ++j) {
                const int32_t b = blk_sel[j];
                if (b < 0) continue;
                for (int32_t k = 0; k < 4; ++k) sel_out[n++] = b * 4 + k;
            }
        } else {   // no selection needed: every complete block is visible
            for (int32_t b = 0; b < n_vis; ++b)
                for (int32_t k = 0; k < 4; ++k) sel_out[n++] = b * 4 + k;
        }
        for (int32_t p2 = n_vis * 4; p2 <= pos; ++p2) sel_out[n++] = p2;
        *n_sel_out = n;
    });
}

// Gather with a DEVICE-side count (upper-bound launch, guard on *n_sel).
inline sycl::event qsa_gather_n(sycl::queue& q, const sycl::half* kc,
                                const sycl::half* vc, sycl::half* gk,
                                sycl::half* gv, const int32_t* sel,
                                const int32_t* n_sel, uint32_t cap,
                                uint32_t n_kv, uint32_t hd,
                                uint32_t src_ctx, uint32_t dst_ctx) {
    return q.parallel_for(sycl::range<1>(uint64_t(cap) * n_kv * hd),
                          [=](sycl::id<1> i) {
        const uint32_t d = uint32_t(i) % hd;
        const uint32_t j = (uint32_t(i) / hd) % cap;
        const uint32_t h = uint32_t(i) / (uint64_t(hd) * cap);
        if (int32_t(j) >= *n_sel) return;
        gk[(uint64_t(h) * dst_ctx + j) * hd + d] =
            kc[(uint64_t(h) * src_ctx + uint32_t(sel[j])) * hd + d];
        gv[(uint64_t(h) * dst_ctx + j) * hd + d] =
            vc[(uint64_t(h) * src_ctx + uint32_t(sel[j])) * hd + d];
    });
}

// Split-K one-query attention over gathered rows: pass 1 computes per
// (head, key-tile) partial online-softmax accumulators (m, l, acc[hd]);
// pass 2 merges tiles per head. Occupancy: n_q*TILES WGs instead of n_q.
inline sycl::event qsa_attend_splitk(sycl::queue& q, const sycl::half* qrow,
                                     const sycl::half* gk, const sycl::half* gv,
                                     const int32_t* n_sel, sycl::half* out,
                                     float* part /* [n_q, TILES, hd+2] */,
                                     uint32_t n_q, uint32_t n_kv, uint32_t hd,
                                     uint32_t cap) {
    constexpr int SG = 16;
    constexpr uint32_t TILES = 16;
    const float scale = 1.0f / sycl::sqrt(float(hd));
    auto e1 = ie::ps(q, "qsa_att_part", [&](sycl::handler& h2) {
        h2.parallel_for(sycl::nd_range<1>(uint64_t(n_q) * TILES * SG, SG),
                        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint32_t g2 = uint32_t(it.get_group(0));
            const uint32_t h = g2 / TILES, tile = g2 % TILES;
            const uint32_t lid = uint32_t(it.get_local_id(0));
            auto sg = it.get_sub_group();
            const uint32_t kvh = h / (n_q / n_kv);
            const sycl::half* qh = qrow + uint64_t(h) * hd;
            const int32_t n = *n_sel;
            const int32_t per = (n + int32_t(TILES) - 1) / int32_t(TILES);
            const int32_t j0 = int32_t(tile) * per;
            const int32_t j1 = sycl::min(n, j0 + per);
            float m = -1e30f, l = 0.f;
            float acc[16];
            for (int d = 0; d < 16; ++d) acc[d] = 0.f;
            for (int32_t j = j0; j < j1; ++j) {
                const sycl::half* kr = gk + (uint64_t(kvh) * cap + j) * hd;
                float p = 0.f;
                for (uint32_t d = lid; d < hd; d += SG)
                    p += float(qh[d]) * float(kr[d]);
                const float sc = sycl::reduce_over_group(sg, p, sycl::plus<float>()) * scale;
                const float m2 = sycl::fmax(m, sc);
                const float corr = sycl::exp(m - m2);
                const float w = sycl::exp(sc - m2);
                const sycl::half* vr = gv + (uint64_t(kvh) * cap + j) * hd;
                for (uint32_t d = lid, di = 0; d < hd; d += SG, ++di)
                    acc[di] = acc[di] * corr + w * float(vr[d]);
                l = l * corr + w;
                m = m2;
            }
            float* pr = part + (uint64_t(h) * TILES + tile) * (hd + 2);
            for (uint32_t d = lid, di = 0; d < hd; d += SG, ++di)
                pr[d] = acc[di];
            if (lid == 0) { pr[hd] = m; pr[hd + 1] = l; }
        });
    });
    return ie::ps(q, "qsa_att_red", [&](sycl::handler& h2) {
        h2.depends_on({e1});
        h2.parallel_for(sycl::nd_range<1>(uint64_t(n_q) * SG, SG),
                        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint32_t h = uint32_t(it.get_group(0));
            const uint32_t lid = uint32_t(it.get_local_id(0));
            float M = -1e30f;
            for (uint32_t t2 = 0; t2 < TILES; ++t2)
                M = sycl::fmax(M, part[(uint64_t(h) * TILES + t2) * (hd + 2) + hd]);
            float L = 0.f;
            for (uint32_t t2 = 0; t2 < TILES; ++t2) {
                const float* pr = part + (uint64_t(h) * TILES + t2) * (hd + 2);
                L += sycl::exp(pr[hd] - M) * pr[hd + 1];
            }
            const float inv = L > 0.f ? 1.f / L : 0.f;
            for (uint32_t d = lid; d < hd; d += SG) {
                float s2 = 0.f;
                for (uint32_t t2 = 0; t2 < TILES; ++t2) {
                    const float* pr = part + (uint64_t(h) * TILES + t2) * (hd + 2);
                    s2 += sycl::exp(pr[hd] - M) * pr[d];
                }
                out[uint64_t(h) * hd + d] = sycl::half(s2 * inv);
            }
        });
    });
}

// FUSED gather+attend, token-grouped (2026-08-27): reads the KV cache
// through sel[j] directly (the gather was a pure copy -> identical values,
// identical j order -> bit-identical outputs) and carries a token dimension
// so a group of Tg tokens runs in ONE part+reduce launch pair (the per-token
// gather+attend pair was 2 x T launches/layer in QSA prefill). part scratch
// is per-token within the group: part[(t*n_q + h)*TILES + tile][hd+2].
inline sycl::event qsa_attend_fused_grp(sycl::queue& q, const sycl::half* qrows,
                                        const sycl::half* kc, const sycl::half* vc,
                                        const int32_t* sel_rows, uint32_t sel_stride,
                                        const int32_t* n_sel_rows,
                                        sycl::half* out_rows,
                                        float* part, uint32_t n_q, uint32_t n_kv,
                                        uint32_t hd, uint32_t max_ctx, uint32_t Tg) {
    constexpr int SG = 16;
    constexpr uint32_t TILES = 16;
    const float scale = 1.0f / sycl::sqrt(float(hd));
    auto e1 = ie::ps(q, "qsa_att_part", [&](sycl::handler& h2) {
        h2.parallel_for(sycl::nd_range<1>(uint64_t(Tg) * n_q * TILES * SG, SG),
                        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint32_t g2 = uint32_t(it.get_group(0));
            const uint32_t t  = g2 / (n_q * TILES);
            const uint32_t rem = g2 % (n_q * TILES);
            const uint32_t h = rem / TILES, tile = rem % TILES;
            const uint32_t lid = uint32_t(it.get_local_id(0));
            auto sg = it.get_sub_group();
            const uint32_t kvh = h / (n_q / n_kv);
            const sycl::half* qh = qrows + uint64_t(t) * n_q * hd + uint64_t(h) * hd;
            const int32_t* sel = sel_rows + uint64_t(t) * sel_stride;
            const int32_t n = n_sel_rows[t];
            const int32_t per = (n + int32_t(TILES) - 1) / int32_t(TILES);
            const int32_t j0 = int32_t(tile) * per;
            const int32_t j1 = sycl::min(n, j0 + per);
            float m = -1e30f, l = 0.f;
            float acc[16];
            for (int d = 0; d < 16; ++d) acc[d] = 0.f;
            for (int32_t j = j0; j < j1; ++j) {
                const sycl::half* kr =
                    kc + (uint64_t(kvh) * max_ctx + uint32_t(sel[j])) * hd;
                float p = 0.f;
                for (uint32_t d = lid; d < hd; d += SG)
                    p += float(qh[d]) * float(kr[d]);
                const float sc = sycl::reduce_over_group(sg, p, sycl::plus<float>()) * scale;
                const float m2 = sycl::fmax(m, sc);
                const float corr = sycl::exp(m - m2);
                const float w = sycl::exp(sc - m2);
                const sycl::half* vr =
                    vc + (uint64_t(kvh) * max_ctx + uint32_t(sel[j])) * hd;
                for (uint32_t d = lid, di = 0; d < hd; d += SG, ++di)
                    acc[di] = acc[di] * corr + w * float(vr[d]);
                l = l * corr + w;
                m = m2;
            }
            float* pr = part + ((uint64_t(t) * n_q + h) * TILES + tile) * (hd + 2);
            for (uint32_t d = lid, di = 0; d < hd; d += SG, ++di)
                pr[d] = acc[di];
            if (lid == 0) { pr[hd] = m; pr[hd + 1] = l; }
        });
    });
    return ie::ps(q, "qsa_att_red", [&](sycl::handler& h2) {
        h2.depends_on({e1});
        h2.parallel_for(sycl::nd_range<1>(uint64_t(Tg) * n_q * SG, SG),
                        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint32_t g2 = uint32_t(it.get_group(0));
            const uint32_t t = g2 / n_q, h = g2 % n_q;
            const uint32_t lid = uint32_t(it.get_local_id(0));
            const float* pbase = part + (uint64_t(t) * n_q + h) * TILES * (hd + 2);
            float M = -1e30f;
            for (uint32_t t2 = 0; t2 < TILES; ++t2)
                M = sycl::fmax(M, pbase[t2 * (hd + 2) + hd]);
            float L = 0.f;
            for (uint32_t t2 = 0; t2 < TILES; ++t2) {
                const float* pr = pbase + t2 * (hd + 2);
                L += sycl::exp(pr[hd] - M) * pr[hd + 1];
            }
            const float inv = L > 0.f ? 1.f / L : 0.f;
            sycl::half* out = out_rows + uint64_t(t) * n_q * hd;
            for (uint32_t d = lid; d < hd; d += SG) {
                float s2 = 0.f;
                for (uint32_t t2 = 0; t2 < TILES; ++t2) {
                    const float* pr = pbase + t2 * (hd + 2);
                    s2 += sycl::exp(pr[hd] - M) * pr[d];
                }
                out[uint64_t(h) * hd + d] = sycl::half(s2 * inv);
            }
        });
    });
}

// One-query attention over the gathered rows (length read from device).
// No mask, no append: every gathered position is causally legal by
// construction (self included). One WG of 16 lanes per Q head; online
// softmax in fp32; GQA kv head = h / (n_q / n_kv). Correctness-first shape.
inline sycl::event qsa_attend_1row(sycl::queue& q, const sycl::half* qrow,
                                   const sycl::half* gk, const sycl::half* gv,
                                   const int32_t* n_sel, sycl::half* out,
                                   uint32_t n_q, uint32_t n_kv, uint32_t hd,
                                   uint32_t cap) {
    constexpr int SG = 16;
    const float scale = 1.0f / sycl::sqrt(float(hd));
    return ie::ps(q, "qsa_attend1", [&](sycl::handler& h2) {
        h2.parallel_for(sycl::nd_range<1>(uint64_t(n_q) * SG, SG),
                        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint32_t h = uint32_t(it.get_group(0));
            const uint32_t lid = uint32_t(it.get_local_id(0));
            auto sg = it.get_sub_group();
            const uint32_t kvh = h / (n_q / n_kv);
            const sycl::half* qh = qrow + uint64_t(h) * hd;
            const int32_t n = *n_sel;
            float m = -1e30f, l = 0.f;
            float acc[16];
            for (int d = 0; d < 16; ++d) acc[d] = 0.f;
            for (int32_t j = 0; j < n; ++j) {
                const sycl::half* kr = gk + (uint64_t(kvh) * cap + j) * hd;
                float p = 0.f;
                for (uint32_t d = lid; d < hd; d += SG)
                    p += float(qh[d]) * float(kr[d]);
                const float sc = sycl::reduce_over_group(sg, p, sycl::plus<float>()) * scale;
                const float m2 = sycl::fmax(m, sc);
                const float corr = sycl::exp(m - m2);
                const float w = sycl::exp(sc - m2);
                const sycl::half* vr = gv + (uint64_t(kvh) * cap + j) * hd;
                for (uint32_t d = lid, di = 0; d < hd; d += SG, ++di)
                    acc[di] = acc[di] * corr + w * float(vr[d]);
                l = l * corr + w;
                m = m2;
            }
            const float inv = l > 0.f ? 1.f / l : 0.f;
            for (uint32_t d = lid, di = 0; d < hd; d += SG, ++di)
                out[uint64_t(h) * hd + d] = sycl::half(acc[di] * inv);
        });
    });
}

// gathered[h, j, :] = cache[h, sel[j], :]  (both [n_kv, ctx, hd] head-major).
inline sycl::event qsa_gather(sycl::queue& q, const sycl::half* kc, const sycl::half* vc,
                              sycl::half* gk, sycl::half* gv, const int32_t* sel,
                              uint32_t n_sel, uint32_t n_kv, uint32_t hd,
                              uint32_t src_ctx, uint32_t dst_ctx) {
    return q.parallel_for(sycl::range<1>(uint64_t(n_sel) * n_kv * hd), [=](sycl::id<1> i) {
        const uint32_t d = uint32_t(i) % hd;
        const uint32_t j = (uint32_t(i) / hd) % n_sel;
        const uint32_t h = uint32_t(i) / (uint64_t(hd) * n_sel);
        gk[(uint64_t(h) * dst_ctx + j) * hd + d] =
            kc[(uint64_t(h) * src_ctx + uint32_t(sel[j])) * hd + d];
        gv[(uint64_t(h) * dst_ctx + j) * hd + d] =
            vc[(uint64_t(h) * src_ctx + uint32_t(sel[j])) * hd + d];
    });
}

}  // namespace

// Pinned (USM-host) pages are unevictable and unswappable: allocating past
// what the box can spare livelocks the whole machine instead of failing
// (2026-08-27 freeze — journald died mid "Under memory pressure").  Refuse a
// pin unless /proc/meminfo says the bytes fit with a 32 GiB headroom.
static bool ram_can_pin(uint64_t nbytes) {
    std::FILE* f = std::fopen("/proc/meminfo", "r");
    if (!f) return true;  // no meminfo, no opinion
    uint64_t avail_kb = 0;
    char line[128];
    while (std::fgets(line, sizeof line, f))
        if (std::sscanf(line, "MemAvailable: %lu kB", &avail_kb) == 1) break;
    std::fclose(f);
    if (!avail_kb) return true;
    constexpr uint64_t headroom = 32ull << 30;
    return avail_kb * 1024ull > nbytes + headroom;
}

std::string Qwen4ExpModel::prepare_bank_views() {
    // Bank donor (replicated multi-GPU topologies): reuse another instance's
    // prepared views — pinned host USM is context-scoped, and the fleet's
    // shared sycl::context makes it legal on every card. Avoids double-
    // pinning ~66 GB (the 2026-08-27 freeze class). Donor must cover this
    // instance's layer range and have run its own init_runtime first.
    if (bank_donor_) {
        for (uint32_t L = layer_lo_; L < layer_hi_; ++L) {
            const Qwen4ExpLayer& dl = bank_donor_->layers_[L];
            if (!dl.gate_bv.K && layers_[L].gate_exps)
                return "qwen4exp: bank donor has no views for layer " +
                       std::to_string(L) + " (init donor first)";
            layers_[L].gate_bv = dl.gate_bv;
            layers_[L].up_bv   = dl.up_bv;
            layers_[L].down_bv = dl.down_bv;
        }
        return {};
    }
    // Fast paths stream the raw mmap; anything else pre-dequants to a host
    // F16 [E][K,N]-transposed copy for gemv_fp16 (the 2 Q5_K gate/up banks
    // and 5 Q8_0 down banks on the real UD file).
    auto view = [&](const GgufTensorInfo* t, bool is_down,
                    Qwen4ExpLayer::BankView& bv) -> std::string {
        const uint32_t K = uint32_t(t->shape[0]), N = uint32_t(t->shape[1]);
        const uint32_t E = uint32_t(t->shape[2]);
        bv.K = K; bv.N = N;
        const bool fast = (!is_down && t->dtype == DType::kQ4_K) ||
                          ( is_down && t->dtype == DType::kQ5_1);
        // PINNED host copies by default: q.memcpy from mmap (non-USM) memory
        // takes the driver's slow per-call staging path — measured as the
        // process drowning in enqueueUSMMemcpy submissions (2026-08-27,
        // gdb stack). USM-host sources DMA asynchronously (the deepseek4
        // pinned-arena lesson). +66.5 GB host RAM; IE_Q4E_NO_PIN_BANKS=1
        // reverts to mmap-backed streaming on low-RAM boxes.
        static const bool pin_banks =
            std::getenv("IE_Q4E_NO_PIN_BANKS") == nullptr;
        if (fast) {
            if (pin_banks) {
                if (!ram_can_pin(t->nbytes))
                    return "qwen4exp: refusing bank pin, MemAvailable too low "
                           "(set IE_Q4E_NO_PIN_BANKS=1 to stream from mmap)";
                void* pin = sycl::malloc_host(t->nbytes, alloc_->queue());
                if (!pin) return "qwen4exp: bank pin alloc failed";
                std::memcpy(pin, t->data, t->nbytes);
                owned_.push_back(pin);
                bv.raw = static_cast<const uint8_t*>(pin);
            } else {
                bv.raw = static_cast<const uint8_t*>(t->data);
            }
            bv.dt  = t->dtype;
            bv.slice_bytes = t->nbytes / E;
            return {};
        }
        // Universal fallback: dequant whole bank -> F16 [E][K*N] transposed.
        std::vector<float> f32;
        if (auto e = dequant_tensor_fp32(t, f32); !e.empty()) return e;
        sycl::half* dst;
        if (pin_banks) {
            const uint64_t need = uint64_t(E) * K * N * sizeof(sycl::half);
            if (!ram_can_pin(need))
                return "qwen4exp: refusing f16 bank pin, MemAvailable too low "
                       "(set IE_Q4E_NO_PIN_BANKS=1 to stream from mmap)";
            dst = static_cast<sycl::half*>(
                sycl::malloc_host(need, alloc_->queue()));
            if (!dst) return "qwen4exp: f16 bank pin alloc failed";
            owned_.push_back(dst);
        } else {
            f16_banks_.emplace_back(uint64_t(E) * K * N);
            dst = f16_banks_.back().data();
        }
        for (uint32_t e = 0; e < E; ++e) {
            const float* src = f32.data() + uint64_t(e) * K * N;   // [N, K] rows
            sycl::half*  d   = dst + uint64_t(e) * K * N;          // [K, N]
            for (uint32_t nn = 0; nn < N; ++nn)
                for (uint32_t k = 0; k < K; ++k)
                    d[uint64_t(k) * N + nn] = sycl::half(src[uint64_t(nn) * K + k]);
        }
        bv.f16 = dst;
        bv.dt  = DType::kF16;
        bv.slice_bytes = uint64_t(K) * N * sizeof(sycl::half);
        return {};
    };
    for (uint32_t L = layer_lo_; L < layer_hi_; ++L) {
        Qwen4ExpLayer& w = layers_[L];
        if (auto e = view(w.gate_exps, false, w.gate_bv); !e.empty()) return e;
        if (auto e = view(w.up_exps,   false, w.up_bv);   !e.empty()) return e;
        if (auto e = view(w.down_exps, true,  w.down_bv); !e.empty()) return e;
    }
    return {};
}

std::string Qwen4ExpModel::init_runtime(uint32_t max_ctx, uint32_t max_chunk) {
    if (!alloc_) return "qwen4exp runtime: load() first";
    max_ctx_ = max_ctx; max_chunk_ = max_chunk;
    const uint32_t H = cfg_.hidden, MT = max_chunk;
    const uint32_t D = cfg_.hc_count * H;                     // 10240
    const uint32_t SI = cfg_.ssm_inner;                       // 6144
    const uint32_t CC = SI + 2u * cfg_.ssm_k_heads * cfg_.ssm_state;  // 10240

    if (auto e = prepare_bank_views(); !e.empty()) return e;

    full_idx_.assign(cfg_.n_layers, -1);
    lin_idx_.assign(cfg_.n_layers, -1);
    int32_t nf = 0, nl = 0;
    for (uint32_t L = layer_lo_; L < layer_hi_; ++L)
        if (cfg_.is_full_attn(L)) full_idx_[L] = nf++; else lin_idx_[L] = nl++;
    if (nf == 0) nf = 1;   // KvCache/DeltaNetState reject zero layers
    if (nl == 0) nl = 1;

    KvCacheConfig kc{uint32_t(nf), cfg_.n_kv_heads, max_ctx, cfg_.head_dim, false};
    if (auto e = kv_.init(*alloc_, kc); !e.empty()) return e;
    DeltaNetStateConfig dc{uint32_t(nl), cfg_.ssm_v_heads, cfg_.ssm_state,
                           cfg_.ssm_state, CC, cfg_.ssm_conv_kernel};
    if (auto e = dn_.init(*alloc_, dc); !e.empty()) return e;

    auto af = [&](uint64_t n) { void* p = alloc_->malloc(n * 4); owned_.push_back(p); return static_cast<float*>(p); };
    auto ah = [&](uint64_t n) { void* p = alloc_->malloc(n * 2); owned_.push_back(p); return static_cast<sycl::half*>(p); };
    auto ai = [&](uint64_t n) { void* p = alloc_->malloc(n * 4); owned_.push_back(p); return static_cast<int32_t*>(p); };

    d_tokens_ = ai(MT); d_pos_ = ai(MT); d_pos3_ = ai(uint64_t(3) * MT);
    emb_   = ah(uint64_t(MT) * H);
    wide_  = af(uint64_t(MT) * D);
    wide_bank0_ = wide_;
    xn_ws_ = af(uint64_t(MT) * D);
    lo_ws_ = af(uint64_t(MT) * cfg_.hc_low_rank);
    hc_x16_  = ah(uint64_t(MT) * D);
    hc_lo16_ = ah(uint64_t(MT) * cfg_.hc_low_rank);
    {   // [K, N]-transposed HC-weight copies for the T>16 prefill mix:
        // gemm_fp16 wants B in [K, N]; the [N, K] originals stay for v2's
        // per-token GEMV. In-place NT alternatives were rejected measured:
        // oneDNN nt = bit-nondeterministic (k-slice atomics), the hand nt
        // kernel = 3x slow (uncoalesced B staging re-read per M-block).
        // ~26 MB/layer, before the ecache sizing so the adaptive fit sees it.
        auto& tq = alloc_->queue();
        auto tpose = [&](const sycl::half* src, uint32_t N2, uint32_t K2)
            -> sycl::half* {
            if (!src) return nullptr;
            auto* dst = ah(uint64_t(N2) * K2);
            tq.parallel_for(sycl::range<1>(uint64_t(N2) * K2),
                            [=](sycl::id<1> i) {
                const uint32_t n2 = uint32_t(i / K2), k2 = uint32_t(i % K2);
                dst[uint64_t(k2) * N2 + n2] = src[i];
            });
            return dst;
        };
        const uint32_t r = cfg_.hc_low_rank;
        for (uint32_t L = layer_lo_; L < layer_hi_; ++L) {
            Qwen4ExpLayer& wl = layers_[L];
            wl.hc_attn_down_t = tpose(wl.hc_attn_down, r, D);
            wl.hc_attn_up_t   = tpose(wl.hc_attn_up, D, r);
            wl.hc_ffn_down_t  = tpose(wl.hc_ffn_down, r, D);
            wl.hc_ffn_up_t    = tpose(wl.hc_ffn_up, D, r);
        }
        if (mtp_.loaded) {
            mtp_.lw.hc_attn_down_t = tpose(mtp_.lw.hc_attn_down, r, D);
            mtp_.lw.hc_attn_up_t   = tpose(mtp_.lw.hc_attn_up, D, r);
            mtp_.lw.hc_ffn_down_t  = tpose(mtp_.lw.hc_ffn_down, r, D);
            mtp_.lw.hc_ffn_up_t    = tpose(mtp_.lw.hc_ffn_up, D, r);
        }
        out_hc_down_t = tpose(out_hc_down, r, D);
        out_hc_up_t   = tpose(out_hc_up, D, r);
        tq.wait();
    }
    mixed_ = ah(uint64_t(MT) * H);
    inj_   = af(uint64_t(MT) * cfg_.hc_count);
    blockout_ = ah(uint64_t(MT) * H);
    dn_qkv_ = ah(uint64_t(MT) * CC); dn_conv_ = ah(uint64_t(MT) * CC);
    dn_qpre_ = af(uint64_t(MT) * 2048); dn_kpre_ = af(uint64_t(MT) * 2048);
    dn_vpre_ = af(uint64_t(MT) * SI);
    dn_qrep_ = af(uint64_t(MT) * SI); dn_krep_ = af(uint64_t(MT) * SI);
    dn_ab64_ = ah(uint64_t(MT) * 128);
    dn_a48_ = ah(uint64_t(MT) * cfg_.ssm_v_heads); dn_b48_ = ah(uint64_t(MT) * cfg_.ssm_v_heads);
    dn_g_ = af(uint64_t(MT) * cfg_.ssm_v_heads); dn_beta_ = af(uint64_t(MT) * cfg_.ssm_v_heads);
    dn_out_ = af(uint64_t(MT) * SI);
    dn_z_ = ah(uint64_t(MT) * SI); dn_gn_ = ah(uint64_t(MT) * SI);
    const uint32_t Nq = cfg_.n_q_heads * cfg_.head_dim;          // 6144
    qg_ = ah(uint64_t(MT) * Nq * 2);
    aq_ = ah(uint64_t(MT) * Nq); agate_ = ah(uint64_t(MT) * Nq);
    ak_ = ah(uint64_t(MT) * cfg_.n_kv_heads * cfg_.head_dim);
    av_ = ah(uint64_t(MT) * cfg_.n_kv_heads * cfg_.head_dim);
    attn_out_ = ah(uint64_t(MT) * Nq);
    router_out_ = ah(cfg_.n_experts);
    act_q8_ = alloc_->malloc((SI / 32) * sizeof(block_q8_1x)); owned_.push_back(act_q8_);
    moe_g_ = ah(cfg_.expert_ffn); moe_u_ = ah(cfg_.expert_ffn); moe_h_ = ah(cfg_.expert_ffn);
    moe_y_ = ah(H);
    sh_g_ = ah(cfg_.shared_expert_ffn); sh_u_ = ah(cfg_.shared_expert_ffn);
    sh_h_ = ah(cfg_.shared_expert_ffn); sh_y_ = ah(H); sh_s_ = ah(1);
    moe_acc_ = af(H);
    {   // MoE VRAM expert cache: C slots/layer within IE_Q4E_ECACHE_GB
        // (default 10 GiB). C is derived from the summed per-layer slot sizes.
        uint64_t sum_slot = 0;
        for (uint32_t L = layer_lo_; L < layer_hi_; ++L)
            sum_slot += layers_[L].gate_bv.slice_bytes +
                        layers_[L].up_bv.slice_bytes +
                        layers_[L].down_bv.slice_bytes;
        // ADAPTIVE default: fill the card to a reserve line (2.5 GiB for
        // driver/display + KV growth + workspaces) instead of a fixed size —
        // per-card budgets then track what load() actually left free
        // (founder 2026-08-27: the cards should be loaded, hit rate is the
        // game). IE_Q4E_ECACHE_GB still overrides.
        const uint64_t gmem =
            alloc_->device().get_info<sycl::info::device::global_mem_size>();
        // 4 GiB reserve: 2.5 proved too thin — the driver near-OOM-migrates
        // and L0 spin-stalls when a tool's real workspaces exceed the 1 GiB
        // estimate (2026-08-27, run2 reference section at ~30/32 GiB).
        const uint64_t reserve = 4096ull << 20;
        const uint64_t used = dev_bytes_ + (1536ull << 20); // weights + ws estimate
        uint64_t budget = gmem > used + reserve ? gmem - used - reserve
                                                : 4ull << 30;
        if (const char* e = std::getenv("IE_Q4E_ECACHE_GB"))
            if (int v = std::atoi(e); v > 0) {
                // CLAMP to the adaptive fit: an override above what the card
                // can actually hold oversubscribes VRAM, and sustained
                // eviction pressure wedges the GuC command channel (the
                // 2026-08-27 17:41 desktop crash — IE_Q4E_ECACHE_GB=26 on a
                // 32 GiB card with ~9 GiB resident: TLB-invalidation storms
                // then "Invalid G2H release"). Never trust the env to know
                // the card's headroom.
                const uint64_t want = uint64_t(v) << 30;
                if (want > budget)
                    std::fprintf(stderr,
                        "[qwen4exp] IE_Q4E_ECACHE_GB=%d exceeds the safe fit "
                        "(%.1f GiB free after weights+reserve) — clamped\n",
                        v, budget / 1073741824.0);
                budget = std::min(want, budget);
            }
        ecache_slots_ = uint32_t(std::min<uint64_t>(
            cfg_.n_experts, std::max<uint64_t>(4, budget / sum_slot)));
        ecache_.resize(cfg_.n_layers);
        for (uint32_t L = layer_lo_; L < layer_hi_; ++L) {
            const Qwen4ExpLayer& w = layers_[L];
            ECache& ec = ecache_[L];
            ec.gate_off = 0;
            ec.up_off   = w.gate_bv.slice_bytes;
            ec.down_off = ec.up_off + w.up_bv.slice_bytes;
            ec.slot_bytes = ec.down_off + w.down_bv.slice_bytes;
            ec.base = alloc_->malloc(uint64_t(ecache_slots_) * ec.slot_bytes);
            if (!ec.base) return "qwen4exp runtime: expert-cache malloc failed";
            owned_.push_back(ec.base);
            ec.slot_of.assign(cfg_.n_experts, -1);
            ec.expert_in.assign(ecache_slots_, -1);
            ec.last_use.assign(ecache_slots_, 0);
        }
        std::fprintf(stderr, "[qwen4exp] expert cache: %u slots/layer, %.2f GiB\n",
                     ecache_slots_,
                     double(ecache_slots_) * sum_slot / 1073741824.0);
        copyq_ = std::make_unique<sycl::queue>(
            alloc_->queue().get_context(), alloc_->queue().get_device(),
            sycl::property_list{sycl::property::queue::in_order{}});
    }
    bool own_ple = false;
    for (int32_t l : cfg_.ple_layers)
        if (uint32_t(l) >= layer_lo_ && uint32_t(l) < layer_hi_) own_ple = true;
    if (cfg_.has_ple() && own_ple) {
        e16_ = ah(uint64_t(MT) * H);
        ple_key_out_ = ah(uint64_t(MT) * D);
        ple_v_out_   = ah(uint64_t(MT) * H);
        ple_conv_state_ = af(uint64_t(kPleStateRows) * kPleSI);
        ple_ws_ = af(qwen4_ple_workspace_floats(MT));
    }
    if (cfg_.indexer_top_k) {
        const uint32_t nf2 = uint32_t(nf);
        qsa_sel_cap_ = cfg_.indexer_top_k + 4;   // 512 blocks x4 + tail(3) + self
        idx_kcache_ = ah(uint64_t(nf2) * max_ctx * cfg_.indexer_head_dim);
        blk_keys_   = af(uint64_t(nf2) * (max_ctx / 4) * cfg_.indexer_head_dim);
        idx_q_      = ah(uint64_t(MT) * cfg_.indexer_n_heads * cfg_.indexer_head_dim);
        idx_kraw_   = ah(uint64_t(MT) * cfg_.indexer_head_dim);
        qsa_scores_ = af(max_ctx / 4);
        qsa_sel_    = ai(qsa_sel_cap_);
        qsa_blk_sel_ = ai(cfg_.indexer_top_k / 4);
        qsa_scoresT_  = af(uint64_t(MT) * (max_ctx / 4));
        qsa_blk_selT_ = ai(uint64_t(MT) * (cfg_.indexer_top_k / 4));
        qsa_selT_     = ai(uint64_t(MT) * qsa_sel_cap_);
        qsa_nselT_    = ai(MT);
        qsa_partG_    = af(64ull * cfg_.n_q_heads * 16 * (cfg_.head_dim + 2));
        qsa_nsel_    = ai(1);
        qsa_part_    = af(uint64_t(cfg_.n_q_heads) * 16 * (cfg_.head_dim + 2));
        qsa_gk_ = ah(uint64_t(cfg_.n_kv_heads) * qsa_sel_cap_ * cfg_.head_dim);
        qsa_gv_ = ah(uint64_t(cfg_.n_kv_heads) * qsa_sel_cap_ * cfg_.head_dim);
        blk_done_.assign(nf2, 0);
    }
    if (layer_hi_ == cfg_.n_layers) logits_ = ah(cfg_.vocab);
    if (mtp_.loaded) {
        KvCacheConfig mkc{1u, cfg_.n_kv_heads, max_ctx, cfg_.head_dim, false};
        if (auto e = mtp_.kv.init(*alloc_, mkc); !e.empty()) return "mtp kv: " + e;
        mtp_.idx_kcache = ah(uint64_t(max_ctx) * cfg_.indexer_head_dim);
        mtp_.blk_keys   = af(uint64_t(max_ctx / 4) * cfg_.indexer_head_dim);
        mtp_wide_  = af(uint64_t(MT) * cfg_.hc_count * H);
        mtp_x16_   = ah(uint64_t(MT) * cfg_.hc_count * H);
        mtp_e16_   = ah(uint64_t(MT) * H);
        mtp_carry_ = af(uint64_t(cfg_.hc_count) * H);
    }
    gemm_c_     = af(uint64_t(MT) * 12288);
    // T>16 dense-Q8 prefill: dequant-once scratch for the largest dense mat
    // (attn_q [H, 12288]); the batched int-dot would re-read the weight
    // ceil(T/16) times per chunk (the pp 143 -> 123 regression).
    if (qwen4exp_dense_q8()) dq_ws_ = ah(uint64_t(H) * 12288);
    router_all_ = ah(uint64_t(MT) * cfg_.n_experts);
    act_q8T_    = alloc_->malloc(uint64_t(MT) * (SI / 32) * sizeof(block_q8_1x));
    owned_.push_back(act_q8T_);
    moe_accT_   = af(uint64_t(MT) * H);
    // Mega (pick-major) sizing: the grouped prefill path stages EVERY pick's
    // gate/up/h/down rows at once (MT x n_experts_used rows, one launch per
    // stage per layer) — ~+95 MB device vs the per-expert chunk sizing.
    const uint64_t PICKS = uint64_t(MT) * cfg_.n_experts_used;
    moe_gT_ = ah(PICKS * cfg_.expert_ffn);
    moe_uT_ = ah(PICKS * cfg_.expert_ffn);
    // +16 rows of slack: the dpas down kernel's M-group-1 tile loads a full
    // 8 rows past the last job's rows.
    moe_hT_ = ah((PICKS + 16) * cfg_.expert_ffn);
    moe_rsum_ = af((PICKS + 16) * (cfg_.expert_ffn / 32));
    moe_yT_ = ah(PICKS * H);
    moe_tok_idx_ = ai(PICKS);
    moe_tok_w_   = af(PICKS);
    // Worst-case grouped-prefill tiles: every pick row chunked at 16 plus a
    // partial tile per expert (the wave-flush path takes ANY union size, so
    // a fixed cap would overflow once MT * n_experts_used / 16 outgrows it).
    const uint64_t TILES = PICKS / 16 + cfg_.n_experts + 16;
    moe_tiles_   = ai(TILES * 4);
    moe_cells_   = ai(PICKS + MT);
    {
        auto host_alloc = [&](uint64_t bytes) {
            void* p = sycl::malloc_host(bytes, alloc_->queue());
            owned_.push_back(p);
            if (p) std::memset(p, 0, bytes);   // match the zero-init the vectors had
            return p;
        };
        h_tok_idx_ = static_cast<int32_t*>(host_alloc(PICKS * sizeof(int32_t)));
        h_tok_w_   = static_cast<float*>(host_alloc(PICKS * sizeof(float)));
        h_router16_ = static_cast<sycl::half*>(
            host_alloc(uint64_t(16) * cfg_.n_experts * sizeof(sycl::half)));
    }
    h_tiles_.resize(TILES * 4);
    h_cells_.resize(PICKS + MT);
    xg_q8_ = alloc_->malloc(PICKS * (H / 32) * sizeof(block_q8_1x));
    owned_.push_back(xg_q8_);
    act_q8s_  = alloc_->malloc(uint64_t(SI / 32) * sizeof(block_q8_1s));
    act_q8sT_ = alloc_->malloc(uint64_t(MT) * (SI / 32) * sizeof(block_q8_1s));
    xg_q8s_   = alloc_->malloc(PICKS * (H / 32) * sizeof(block_q8_1s));
    owned_.push_back(act_q8s_); owned_.push_back(act_q8sT_); owned_.push_back(xg_q8s_);
    xg16_  = ah(PICKS * H);
    sgateT_ = ah(MT);
    reset_state();
    return {};
}

std::string Qwen4ExpModel::load_mtp(const GgufReader& mg,
                                    const GgufReader* main_g) {
    // Bespoke bind of the converter's mtp-head.gguf (scripts/mtp_to_gguf.py).
    // All tensors are F16/F32 already; 2-D projections transpose to the
    // gemv_fp16 [K, N] layout, HC mats stay GGUF-native (kernel contract),
    // expert banks go device-resident F16 with per-expert [K, N] transpose.
    sycl::queue& q = alloc_->queue();
    if (!token_embd && main_g) {
        // Tail-of-pipeline stage: pull token_embd for the MTP embed path.
        // Dequant->F16 upload (UD files quantize the embedding table); the
        // head stage's loader does the same normalization.
        const GgufTensorInfo* te = main_g->find_tensor("token_embd.weight");
        if (!te) return "mtp: main gguf missing token_embd.weight";
        std::vector<float> tf;
        if (auto e = dequant_tensor_fp32(te, tf); !e.empty()) return e;
        std::vector<sycl::half> th(tf.size());
        for (size_t i = 0; i < tf.size(); ++i) th[i] = sycl::half(tf[i]);
        void* d = alloc_->malloc(th.size() * 2);
        if (!d) return "mtp: token_embd malloc";
        q.memcpy(d, th.data(), th.size() * 2).wait();
        owned_.push_back(d);
        token_embd = static_cast<sycl::half*>(d);
        dev_bytes_ += th.size() * 2;
    }
    auto find = [&](const char* n) { return mg.find_tensor(n); };
    std::vector<float> f32;
    auto up_f32 = [&](const char* n, float** dst) -> std::string {
        const GgufTensorInfo* t = find(n);
        if (!t) return std::string("mtp: missing ") + n;
        if (auto e = dequant_tensor_fp32(t, f32); !e.empty()) return e;
        void* d = alloc_->malloc(f32.size() * 4);
        if (!d) return "mtp: malloc f32";
        q.memcpy(d, f32.data(), f32.size() * 4).wait();
        owned_.push_back(d);
        *dst = static_cast<float*>(d);
        dev_bytes_ += f32.size() * 4;
        return {};
    };
    auto up_f16 = [&](const char* n, sycl::half** dst, bool trans_kn) -> std::string {
        const GgufTensorInfo* t = find(n);
        if (!t) return std::string("mtp: missing ") + n;
        if (auto e = dequant_tensor_fp32(t, f32); !e.empty()) return e;
        std::vector<sycl::half> h16(f32.size());
        if (trans_kn && t->n_dims == 2) {
            const uint64_t K = t->shape[0], N = t->shape[1];
            for (uint64_t nn = 0; nn < N; ++nn)
                for (uint64_t k = 0; k < K; ++k)
                    h16[k * N + nn] = sycl::half(f32[nn * K + k]);
        } else {
            for (size_t i = 0; i < f32.size(); ++i) h16[i] = sycl::half(f32[i]);
        }
        void* d = alloc_->malloc(h16.size() * 2);
        if (!d) return "mtp: malloc f16";
        q.memcpy(d, h16.data(), h16.size() * 2).wait();
        owned_.push_back(d);
        *dst = static_cast<sycl::half*>(d);
        dev_bytes_ += h16.size() * 2;
        return {};
    };
    Mtp& m = mtp_;
    std::string e;
    if (!(e = up_f32("mtp.pre_fc_norm_hidden.weight", &m.pre_fc_norm_hidden)).empty()) return e;
    if (!(e = up_f32("mtp.pre_fc_norm_embedding.weight", &m.pre_fc_norm_embed)).empty()) return e;
    if (!(e = up_f16("mtp.fc_hidden.weight", &m.fc_hidden, true)).empty()) return e;
    if (!(e = up_f16("mtp.fc_embedding.weight", &m.fc_embed, true)).empty()) return e;
    Qwen4ExpLayer& w = m.lw;
    if (!(e = up_f16("mtp.attn_q.weight", &w.attn_q, true)).empty()) return e;
    if (!(e = up_f16("mtp.attn_k.weight", &w.attn_k, true)).empty()) return e;
    if (!(e = up_f16("mtp.attn_v.weight", &w.attn_v, true)).empty()) return e;
    if (!(e = up_f16("mtp.attn_output.weight", &w.attn_output, true)).empty()) return e;
    if (!(e = up_f32("mtp.attn_q_norm.weight", &w.attn_q_norm)).empty()) return e;
    if (!(e = up_f32("mtp.attn_k_norm.weight", &w.attn_k_norm)).empty()) return e;
    if (!(e = up_f16("mtp.indexer.q_proj.weight", &w.idx_q_proj, true)).empty()) return e;
    if (!(e = up_f16("mtp.indexer.k_proj.weight", &w.idx_k_proj, true)).empty()) return e;
    if (!(e = up_f32("mtp.indexer.q_norm.weight", &w.idx_q_norm)).empty()) return e;
    if (!(e = up_f32("mtp.indexer.k_norm.weight", &w.idx_k_norm)).empty()) return e;
    for (const char* side : {"attn", "ffn"}) {
        char b1[64], b2[64], b3[64], b4[64];
        std::snprintf(b1, 64, "mtp.hc_%s_norm.weight", side);
        std::snprintf(b2, 64, "mtp.hc_%s_down.weight", side);
        std::snprintf(b3, 64, "mtp.hc_%s_up.weight", side);
        std::snprintf(b4, 64, "mtp.hc_%s_inject.weight", side);
        const bool at = side[0] == 'a';
        if (!(e = up_f32(b1, at ? &w.hc_attn_norm : &w.hc_ffn_norm)).empty()) return e;
        if (!(e = up_f16(b2, at ? &w.hc_attn_down : &w.hc_ffn_down, false)).empty()) return e;
        if (!(e = up_f16(b3, at ? &w.hc_attn_up : &w.hc_ffn_up, false)).empty()) return e;
        if (!(e = up_f32(b4, at ? &w.hc_attn_inject : &w.hc_ffn_inject)).empty()) return e;
    }
    // Router downcast to F16 [K,N] like the backbone.
    if (!(e = up_f16("mtp.ffn_gate_inp.weight", &w.ffn_gate_inp, true)).empty()) return e;
    if (!(e = up_f16("mtp.ffn_gate_inp_shexp.weight", &w.ffn_gate_inp_shexp, false)).empty()) return e;
    if (!(e = up_f16("mtp.ffn_gate_shexp.weight", &w.ffn_gate_shexp, true)).empty()) return e;
    if (!(e = up_f16("mtp.ffn_up_shexp.weight", &w.ffn_up_shexp, true)).empty()) return e;
    if (!(e = up_f16("mtp.ffn_down_shexp.weight", &w.ffn_down_shexp, true)).empty()) return e;
    // Expert banks: device-resident F16, per-expert [K, N] transpose so the
    // cache-slot compute path's F16 branch (gemv_fp16) consumes them.
    auto up_bank = [&](const char* n, Qwen4ExpLayer::BankView& bv) -> std::string {
        const GgufTensorInfo* t = find(n);
        if (!t) return std::string("mtp: missing ") + n;
        if (auto e2 = dequant_tensor_fp32(t, f32); !e2.empty()) return e2;
        const uint32_t K = uint32_t(t->shape[0]), N = uint32_t(t->shape[1]);
        const uint32_t E = uint32_t(t->shape[2]);
        std::vector<sycl::half> h16(f32.size());
        for (uint32_t ex = 0; ex < E; ++ex) {
            const float* src = f32.data() + uint64_t(ex) * K * N;
            sycl::half*  d2  = h16.data() + uint64_t(ex) * K * N;
            for (uint64_t nn = 0; nn < N; ++nn)
                for (uint64_t k = 0; k < K; ++k)
                    d2[k * N + nn] = sycl::half(src[nn * K + k]);
        }
        void* d = alloc_->malloc(h16.size() * 2);
        if (!d) return "mtp: bank malloc";
        q.memcpy(d, h16.data(), h16.size() * 2).wait();
        owned_.push_back(d);
        bv.f16 = static_cast<const sycl::half*>(d);   // DEVICE pointer (resident)
        bv.raw = nullptr;
        bv.dt = DType::kF16;
        bv.K = K; bv.N = N;
        bv.slice_bytes = uint64_t(K) * N * 2;
        dev_bytes_ += h16.size() * 2;
        return {};
    };
    if (!(e = up_bank("mtp.ffn_gate_exps.weight", w.gate_bv)).empty()) return e;
    if (!(e = up_bank("mtp.ffn_up_exps.weight", w.up_bv)).empty()) return e;
    if (!(e = up_bank("mtp.ffn_down_exps.weight", w.down_bv)).empty()) return e;
    if (!(e = up_f32("mtp.out_hc_norm.weight", &m.out_hc_norm)).empty()) return e;
    if (!(e = up_f16("mtp.out_hc_down.weight", &m.out_hc_down, false)).empty()) return e;
    if (!(e = up_f16("mtp.out_hc_up.weight", &m.out_hc_up, false)).empty()) return e;
    m.loaded = true;
    return {};
}

std::string Qwen4ExpModel::logits_rows(uint32_t T, sycl::half* out) {
    if (!lm_head && !lmh_q8.qs) return "logits_rows: no lm_head on this stage";
    sycl::queue& q = alloc_->queue();
    const uint32_t V = cfg_.vocab, H = cfg_.hidden;
    // T is tiny (K_spec+1); rows-batched with the T=1 leaf's per-row order —
    // one ~1.3 GB lm_head read for all rows instead of one per row, each row
    // still bit-identical to its solo gemv (lossless contract). Under
    // IE_Q4E_DENSE_Q8 the reference lane's lm_head is the int-dot path, so
    // verify rows must take the (row-exact batched) int-dot kernel too.
    static const bool use_q8_lmh = qwen4exp_dense_q8();
    if (use_q8_lmh && lmh_q8.qs && dense_a16()) {
        // ANY T: the rows kernel chunks at 16 internally.
        gemv_q8_0_soa_f16_rows(q, mixed_, lmh_q8.qs, lmh_q8.d, out, H, V, T);
    } else if (use_q8_lmh && lmh_q8.qs) {
        // ANY T via 16-chunks (kernel T_MAX): under q8 residency the F16
        // lm_head pointer is NULL, so this branch must cover every T (the
        // 2026-08-27 DEVICE_LOST: T=512 fell to gemv_fp16_rows(nullptr)).
        quantize_q8_1(q, mixed_, act_q8T_, T * H);
        for (uint32_t t0 = 0; t0 < T; t0 += 16) {
            const uint32_t tn = std::min(16u, T - t0);
            gemv_q8_0_soa_q8_batched(q,
                static_cast<const uint8_t*>(act_q8T_) +
                    uint64_t(t0) * (H / 32) * sizeof(block_q8_1x),
                lmh_q8.qs, lmh_q8.d, out + uint64_t(t0) * V, H, V, tn);
        }
    } else {
        // 8-row groups: one lm_head read per group instead of per row
        // (row-exact rows kernel), any T.
        for (uint32_t t0 = 0; t0 < T; t0 += 8) {
            const uint32_t tn = std::min(8u, T - t0);
            if (tn >= 2)
                gemv_fp16_rows(q, mixed_ + uint64_t(t0) * H, H, lm_head,
                               out + uint64_t(t0) * V, V, H, V, tn);
            else
                gemv_fp16(q, mixed_ + uint64_t(t0) * H, lm_head,
                          out + uint64_t(t0) * V, H, V);
        }
    }
    q.wait();
    return {};
}

// ---- MTP draft head (speculative decode) ---------------------------------

// One full QSA decoder layer over mtp_wide_ [T,4,H] at absolute positions
// [start_pos, start_pos+T). Reuses the model's scratch buffers (never live at
// the same time as a backbone forward) and the same op kernels; the expert
// bank is DEVICE-resident F16, so the MoE loop reads slices directly.
std::string Qwen4ExpModel::mtp_layer_fwd(uint32_t T, uint32_t start_pos) {
    sycl::queue& q = alloc_->queue();
    const Qwen4ExpLayer& w = mtp_.lw;
    const uint32_t H = cfg_.hidden, HD = cfg_.head_dim;
    const uint32_t NQ = cfg_.n_q_heads * HD, NKV = cfg_.n_kv_heads * HD;
    const uint32_t EF = cfg_.expert_ffn, SEF = cfg_.shared_expert_ffn;
    const float eps = cfg_.rms_eps;
    {   // positions buffer for this range
        std::vector<int32_t> pos(T);
        for (uint32_t t = 0; t < T; ++t) pos[t] = int32_t(start_pos + t);
        q.memcpy(d_pos_, pos.data(), T * sizeof(int32_t)).wait();
    }
    auto proj = [&](const sycl::half* X, const sycl::half* W, sycl::half* Y,
                    uint32_t K, uint32_t N) {
        if (T == 1) { gemv_fp16(q, X, W, Y, K, N); return; }
        gemm_fp16(q, X, W, gemm_c_, T, N, K);
        cast_fp32_to_fp16(q, gemm_c_, Y, uint64_t(T) * N);
    };
    // hc attn mix
    if (T > 16 && hc_prefill_gemm() && w.hc_attn_down_t)
        qwen4_hc_mix_prefill(q, mtp_wide_, w.hc_attn_norm, w.hc_attn_down_t,
                    w.hc_attn_up_t, w.hc_attn_inject, xn_ws_, hc_x16_,
                    hc_lo16_, gemm_c_, mixed_, inj_,
                    T, H, cfg_.hc_count, cfg_.hc_low_rank, eps);
    else
        qwen4_hc_mix_v2(q, mtp_wide_, w.hc_attn_norm, w.hc_attn_down, w.hc_attn_up,
                    w.hc_attn_inject, xn_ws_, lo_ws_, mixed_, inj_,
                    T, H, cfg_.hc_count, cfg_.hc_low_rank, eps);
    // attention (own KV/idx caches)
    proj(mixed_, w.attn_q, qg_, H, NQ * 2);
    proj(mixed_, w.attn_k, ak_, H, NKV);
    proj(mixed_, w.attn_v, av_, H, NKV);
    split_q_gate_per_head(q, qg_, aq_, agate_, T, cfg_.n_q_heads, HD);
    rms_norm_f32w(q, aq_, w.attn_q_norm, aq_, T * cfg_.n_q_heads, HD, eps);
    rms_norm_f32w(q, ak_, w.attn_k_norm, ak_, T * cfg_.n_kv_heads, HD, eps);
    rope_partial(q, aq_, d_pos_, aq_, T, cfg_.n_q_heads, HD, cfg_.rope_dim, cfg_.rope_theta);
    rope_partial(q, ak_, d_pos_, ak_, T, cfg_.n_kv_heads, HD, cfg_.rope_dim, cfg_.rope_theta);
    sycl::half* kc = mtp_.kv.k_ptr();
    sycl::half* vc = mtp_.kv.v_ptr();
    const uint32_t dense_ok = cfg_.indexer_top_k ? cfg_.indexer_top_k + 3 : UINT32_MAX;
    // indexer maintenance (side cache + block keys)
    if (cfg_.indexer_top_k) {
        const uint32_t IHD = cfg_.indexer_head_dim;
        const uint32_t IQW = cfg_.indexer_n_heads * IHD;
        proj(mixed_, w.idx_q_proj, idx_q_, H, IQW);
        proj(mixed_, w.idx_k_proj, idx_kraw_, H, IHD);
        rms_norm_f32w(q, idx_q_, w.idx_q_norm, idx_q_, T * cfg_.indexer_n_heads, IHD, eps);
        rope_partial(q, idx_q_, d_pos_, idx_q_, T, cfg_.indexer_n_heads, IHD,
                     cfg_.rope_dim, cfg_.rope_theta);
        if (start_pos < mtp_.blk_done * 4) mtp_.blk_done = start_pos / 4;   // rewind
        q.memcpy(mtp_.idx_kcache + uint64_t(start_pos) * IHD, idx_kraw_,
                 uint64_t(T) * IHD * sizeof(sycl::half));
        const uint32_t nb_tot = (start_pos + T) / 4;
        if (nb_tot > mtp_.blk_done) {
            qsa_pool_norm_rope(q, mtp_.idx_kcache, w.idx_k_norm, mtp_.blk_keys,
                               mtp_.blk_done, nb_tot - mtp_.blk_done,
                               cfg_.rope_theta, eps);
            mtp_.blk_done = nb_tot;
        }
    }
    if (start_pos + T <= dense_ok) {
        if (T > 1)
            full_attention_fa2_prefill_tile_gemma(q, aq_, ak_, av_, kc, vc,
                attn_out_, T, start_pos, cfg_.n_q_heads, cfg_.n_kv_heads, HD,
                max_ctx_, 0);
        else
            full_attention(q, aq_, ak_, av_, kc, vc, attn_out_, T, start_pos,
                           cfg_.n_q_heads, cfg_.n_kv_heads, HD, max_ctx_);
    } else {
        mtp_.kv.append(q, 0, start_pos, T, ak_, av_);
        const uint32_t IQW = cfg_.indexer_n_heads * cfg_.indexer_head_dim;
        const uint32_t kblk = cfg_.indexer_top_k / 4;
        if (T > 1) {
            const uint32_t nb_max = (start_pos + T) / 4;
            qsa_score_batched(q, idx_q_, IQW, mtp_.blk_keys,
                              qsa_scoresT_, max_ctx_ / 4, d_pos_, nb_max, T);
            ds4_indexer_topk(q, qsa_scoresT_, d_pos_, qsa_blk_selT_,
                             T, max_ctx_ / 4, kblk, 4);
            qsa_expand_sel_batched(q, qsa_blk_selT_, kblk, kblk, kblk,
                                   d_pos_, qsa_selT_, qsa_sel_cap_,
                                   qsa_nselT_, T);
            for (uint32_t g0 = 0; g0 < T; g0 += 64) {
                const uint32_t gT = std::min(64u, T - g0);
                qsa_attend_fused_grp(q, aq_ + uint64_t(g0) * NQ, kc, vc,
                    qsa_selT_ + uint64_t(g0) * qsa_sel_cap_, qsa_sel_cap_,
                    qsa_nselT_ + g0, attn_out_ + uint64_t(g0) * NQ, qsa_partG_,
                    cfg_.n_q_heads, cfg_.n_kv_heads, HD, max_ctx_, gT);
            }
        } else
        for (uint32_t t = 0; t < T; ++t) {
            const uint32_t pos = start_pos + t;
            const uint32_t n_vis = (pos + 1) / 4;
            const bool select = n_vis > kblk;
            if (select) {
                qsa_score(q, idx_q_ + uint64_t(t) * IQW, mtp_.blk_keys,
                          qsa_scores_, n_vis);
                ds4_indexer_topk(q, qsa_scores_, d_pos_ + t, qsa_blk_sel_,
                                 1, n_vis, kblk, 4);
            }
            qsa_expand_sel(q, select ? qsa_blk_sel_ : nullptr, kblk,
                           int32_t(pos), qsa_sel_, qsa_nsel_);
            qsa_gather_n(q, kc, vc, qsa_gk_, qsa_gv_, qsa_sel_, qsa_nsel_,
                         qsa_sel_cap_, cfg_.n_kv_heads, HD, max_ctx_, qsa_sel_cap_);
            qsa_attend_splitk(q, aq_ + uint64_t(t) * NQ, qsa_gk_, qsa_gv_,
                              qsa_nsel_, attn_out_ + uint64_t(t) * NQ, qsa_part_,
                              cfg_.n_q_heads, cfg_.n_kv_heads, HD, qsa_sel_cap_);
        }
    }
    mtp_.kv.set_length(0, start_pos + T);
    sigmoid_gate(q, attn_out_, agate_, attn_out_, uint64_t(T) * NQ);
    proj(attn_out_, w.attn_output, blockout_, NQ, H);
    qwen4_hc_combine(q, mtp_wide_, blockout_, inj_, T, H, cfg_.hc_count);
    // hc ffn mix + MoE on the resident F16 banks
    if (T > 16 && hc_prefill_gemm() && w.hc_ffn_down_t)
        qwen4_hc_mix_prefill(q, mtp_wide_, w.hc_ffn_norm, w.hc_ffn_down_t,
                    w.hc_ffn_up_t, w.hc_ffn_inject, xn_ws_, hc_x16_,
                    hc_lo16_, gemm_c_, mixed_, inj_,
                    T, H, cfg_.hc_count, cfg_.hc_low_rank, eps);
    else
        qwen4_hc_mix_v2(q, mtp_wide_, w.hc_ffn_norm, w.hc_ffn_down, w.hc_ffn_up,
                    w.hc_ffn_inject, xn_ws_, lo_ws_, mixed_, inj_,
                    T, H, cfg_.hc_count, cfg_.hc_low_rank, eps);
    {
        gemm_fp16(q, mixed_, w.ffn_gate_inp, gemm_c_, T, cfg_.n_experts, H);
        std::vector<float> rall(uint64_t(T) * cfg_.n_experts);
        q.memcpy(rall.data(), gemm_c_, rall.size() * 4).wait();
        std::vector<std::pair<uint32_t, float>> tk;
        q.memset(moe_accT_, 0, uint64_t(T) * H * sizeof(float));
        for (uint32_t t = 0; t < T; ++t) {
            route_from_logits(rall.data() + uint64_t(t) * cfg_.n_experts,
                              cfg_.n_experts, cfg_.n_experts_used, tk);
            const sycl::half* x = mixed_ + uint64_t(t) * H;
            for (const auto& [ex, wgt] : tk) {
                const sycl::half* gp = w.gate_bv.f16 + uint64_t(ex) * H * EF;
                const sycl::half* up = w.up_bv.f16 + uint64_t(ex) * H * EF;
                const sycl::half* dp = w.down_bv.f16 + uint64_t(ex) * EF * H;
                gemv_fp16(q, x, gp, moe_g_, H, EF);
                gemv_fp16(q, x, up, moe_u_, H, EF);
                swiglu(q, moe_g_, moe_u_, moe_h_, EF);
                gemv_fp16(q, moe_h_, dp, moe_y_, EF, H);
                axpy_f16_to_f32(q, moe_accT_ + uint64_t(t) * H, moe_y_, wgt, H);
            }
            gemv_fp16(q, x, w.ffn_gate_shexp, sh_g_, H, SEF);
            gemv_fp16(q, x, w.ffn_up_shexp,   sh_u_, H, SEF);
            swiglu(q, sh_g_, sh_u_, sh_h_, SEF);
            gemv_fp16(q, sh_h_, w.ffn_down_shexp, sh_y_, SEF, H);
            gemv_fp16(q, x, w.ffn_gate_inp_shexp, sh_s_, H, 1);
            sigmoid_scalar_axpy(q, moe_accT_ + uint64_t(t) * H, sh_y_, sh_s_, H);
        }
        cast_f32_to_f16(q, moe_accT_, blockout_, uint64_t(T) * H);
    }
    qwen4_hc_combine(q, mtp_wide_, blockout_, inj_, T, H, cfg_.hc_count);
    return {};
}

std::string Qwen4ExpModel::mtp_ingest(const int32_t* tokens_host, uint32_t T,
                                      uint32_t start_pos) {
    if (!mtp_.loaded || T == 0) return {};
    sycl::queue& q = alloc_->queue();
    const uint32_t H = cfg_.hidden, hc = cfg_.hc_count;
    const float eps = cfg_.rms_eps;
    // Rows: optional carry (prev chunk's last position, fused with tokens[0]),
    // then positions start_pos..start_pos+T-2 fused with tokens[1..T).
    const uint32_t nc = mtp_have_carry_ ? 1u : 0u;
    const uint32_t nr = nc + (T - 1);
    if (nr == 0) {
        q.memcpy(mtp_carry_, wide_ + uint64_t(T - 1) * hc * H,
                 uint64_t(hc) * H * 4).wait();
        mtp_carry_pos_ = start_pos + T - 1;
        mtp_have_carry_ = true;
        return {};
    }
    // Assemble the raw wide inputs into mtp_wide_ (carry + chunk rows).
    if (nc)
        q.memcpy(mtp_wide_, mtp_carry_, uint64_t(hc) * H * 4);
    if (T > 1)
        q.memcpy(mtp_wide_ + uint64_t(nc) * hc * H, wide_,
                 uint64_t(T - 1) * hc * H * 4);
    q.wait();
    // Fusion: xh = fc_hidden(GN(wide)) per stream; xe = fc_embed(RMS(embed)).
    mtp_group_norm_f16(q, mtp_wide_, mtp_.pre_fc_norm_hidden, mtp_x16_, nr, H, hc, eps);
    gemm_fp16(q, mtp_x16_, mtp_.fc_hidden, gemm_c_, nr * hc, H, H);
    cast_fp32_to_fp16(q, gemm_c_, mtp_x16_, uint64_t(nr) * hc * H);
    {   // embed rows for tokens[0..nr): shifted stream
        std::vector<int32_t> et(nr);
        for (uint32_t i = 0; i < nr; ++i) et[i] = tokens_host[i + (nc ? 0 : 1)];
        q.memcpy(d_tokens_, et.data(), nr * sizeof(int32_t)).wait();
        gather_embed_rows(q, token_embd, d_tokens_, emb_, nr, H);
        rms_norm_f32w(q, emb_, mtp_.pre_fc_norm_embed, emb_, nr, H, eps);
        gemm_fp16(q, emb_, mtp_.fc_embed, gemm_c_, nr, H, H);
        cast_fp32_to_fp16(q, gemm_c_, mtp_e16_, uint64_t(nr) * H);
    }
    mtp_fuse_add(q, mtp_x16_, mtp_e16_, mtp_wide_, nr, H, hc);
    const uint32_t first_pos = nc ? mtp_carry_pos_ : start_pos;
    if (auto e = mtp_layer_fwd(nr, first_pos); !e.empty()) return e;
    // Carry the chunk's last row for the next chunk / the draft step.
    q.memcpy(mtp_carry_, wide_ + uint64_t(T - 1) * hc * H,
             uint64_t(hc) * H * 4).wait();
    mtp_carry_pos_ = start_pos + T - 1;
    mtp_have_carry_ = true;
    return {};
}

std::string Qwen4ExpModel::mtp_draft(int32_t t_next, uint32_t pos, uint32_t K,
                                     int32_t* draft) {
    if (!mtp_.loaded || !mtp_have_carry_) return "mtp: no context";
    sycl::queue& q = alloc_->queue();
    const uint32_t H = cfg_.hidden, hc = cfg_.hc_count;
    const float eps = cfg_.rms_eps;
    int32_t tok = t_next;
    // Draft step k runs the MTP layer at row position (carry_pos + k).
    for (uint32_t k = 0; k < K; ++k) {
        if (k == 0)
            q.memcpy(mtp_wide_, mtp_carry_, uint64_t(hc) * H * 4).wait();
        // else: mtp_wide_ already holds the previous step's layer OUTPUT wide.
        mtp_group_norm_f16(q, mtp_wide_, mtp_.pre_fc_norm_hidden, mtp_x16_, 1, H, hc, eps);
        gemm_fp16(q, mtp_x16_, mtp_.fc_hidden, gemm_c_, hc, H, H);
        cast_fp32_to_fp16(q, gemm_c_, mtp_x16_, uint64_t(hc) * H);
        q.memcpy(d_tokens_, &tok, sizeof(int32_t)).wait();
        gather_embed_rows(q, token_embd, d_tokens_, emb_, 1, H);
        rms_norm_f32w(q, emb_, mtp_.pre_fc_norm_embed, emb_, 1, H, eps);
        gemv_fp16(q, emb_, mtp_.fc_embed, mtp_e16_, H, H);
        mtp_fuse_add(q, mtp_x16_, mtp_e16_, mtp_wide_, 1, H, hc);
        if (auto e = mtp_layer_fwd(1, mtp_carry_pos_ + k); !e.empty()) return e;
        // Prediction head: GR-read mixer -> shared lm_head -> greedy token.
        qwen4_hc_mix_v2(q, mtp_wide_, mtp_.out_hc_norm, mtp_.out_hc_down,
                        mtp_.out_hc_up, nullptr, xn_ws_, lo_ws_, mixed_, nullptr,
                        1, H, hc, cfg_.hc_low_rank, eps);
        if (lmh_q8.qs && qwen4exp_dense_q8() && dense_a16()) {
            gemv_q8_0_soa_f16_g(q, mixed_, lmh_q8.qs, lmh_q8.d, logits_,
                                H, cfg_.vocab);
        } else if (lmh_q8.qs && qwen4exp_dense_q8()) {
            quantize_q8_1(q, mixed_, act_q8_, H);
            gemv_q8_0_soa_q8_g(q, act_q8_, lmh_q8.qs, lmh_q8.d, logits_, H, cfg_.vocab);
        } else {
            // Draft vocab cap (IE_Q4E_DRAFT_VCAP, default 65536): drafts
            // over a prefix vocab; misses only lower accept rate.
            // Default OFF: the 64K cap halved draft cost but collapsed
            // accept 1.78 -> 1.56 (k0 87 -> 80) — this model's outputs lean
            // on high token ids (74455/79790 argmaxes observed). Opt-in env.
            static const uint32_t vcap = [] {
                const char* e = std::getenv("IE_Q4E_DRAFT_VCAP");
                return e ? uint32_t(std::atoi(e)) : 0u;
            }();
            const uint32_t nv = vcap && vcap < cfg_.vocab ? vcap : cfg_.vocab;
            gemv_fp16_ld(q, mixed_, lm_head, cfg_.vocab, logits_, H, nv);
            sample_argmax(q, logits_, d_tokens_ + (max_chunk_ - 1), nv);
            int32_t am = 0;
            q.memcpy(&am, d_tokens_ + (max_chunk_ - 1), sizeof(int32_t)).wait();
            draft[k] = am;
            tok = am;
            continue;
        }
        // Device argmax + 4-byte D2H (was a 500 KB full-vocab D2H + host
        // scan per step). Draft-only: a tie-break difference vs the host
        // reference argmax can only change a PROPOSAL, never the output.
        int32_t am = 0;
        sample_argmax(q, logits_, d_tokens_ + (max_chunk_ - 1), cfg_.vocab);
        q.memcpy(&am, d_tokens_ + (max_chunk_ - 1), sizeof(int32_t)).wait();
        draft[k] = am;
        tok = am;
    }
    // Rewind the speculative MTP rows: the verify forward re-ingests the
    // accepted positions properly (position-indexed caches overwrite).
    mtp_.kv.set_length(0, mtp_carry_pos_);
    if (mtp_.blk_done * 4 > mtp_carry_pos_) mtp_.blk_done = mtp_carry_pos_ / 4;
    return {};
}

std::string Qwen4ExpModel::stash_slot(SlotState& s, uint32_t depth) {
    if (mtp_.loaded) return "qwen4exp stash: MTP state not stashable (no spec+parallel)";
    sycl::queue& q = alloc_->queue();
    s.depth = 0;
    // KV: pack rows [0, depth) per (full layer, kv head) — live cache strided
    // by max_ctx, stash packed by depth (the 27B HostSlotStash copy shape).
    const KvCacheConfig& kc = kv_.config();
    if (kc.use_int8) return "qwen4exp stash: int8-kv unsupported";
    {
        const uint64_t slice  = uint64_t(depth) * kc.head_dim * sizeof(sycl::half);
        const uint64_t stride = uint64_t(kc.max_ctx) * kc.head_dim;   // halves
        const uint64_t nsl    = uint64_t(kc.n_layers_full) * kc.n_kv_heads;
        s.kv_k.resize(nsl * slice);
        s.kv_v.resize(nsl * slice);
        for (uint64_t sl = 0; sl < nsl; ++sl) {
            q.memcpy(s.kv_k.data() + sl * slice, kv_.k_ptr() + sl * stride, slice);
            q.memcpy(s.kv_v.data() + sl * slice, kv_.v_ptr() + sl * stride, slice);
        }
    }
    // QSA indexer caches (flat [nf][max_ctx or max_ctx/4][idx_hd]).
    if (idx_kcache_) {
        const uint32_t nf = uint32_t(blk_done_.size());
        const uint32_t ihd = cfg_.indexer_head_dim;
        const uint64_t kslice  = uint64_t(depth) * ihd * sizeof(sycl::half);
        const uint64_t kstride = uint64_t(max_ctx_) * ihd;             // halves
        const uint32_t nblk    = (depth + 3) / 4;
        const uint64_t bslice  = uint64_t(nblk) * ihd * sizeof(float);
        const uint64_t bstride = uint64_t(max_ctx_ / 4) * ihd;        // floats
        s.idxk.resize(uint64_t(nf) * kslice);
        s.blkk.resize(uint64_t(nf) * bslice);
        for (uint32_t l = 0; l < nf; ++l) {
            q.memcpy(s.idxk.data() + uint64_t(l) * kslice,
                     idx_kcache_ + uint64_t(l) * kstride, kslice);
            q.memcpy(s.blkk.data() + uint64_t(l) * bslice,
                     blk_keys_ + uint64_t(l) * bstride, bslice);
        }
    }
    // DeltaNet + PLE (the snapshot() walk, into host buffers).
    {
        const uint64_t sb = dn_.state_elems_per_layer() *
                            dn_.config().n_layers_linear * sizeof(float);
        const uint64_t cb = dn_.conv_elems_per_layer() *
                            dn_.config().n_layers_linear * sizeof(sycl::half);
        s.dns.resize(sb);
        s.dnc.resize(cb);
        q.memcpy(s.dns.data(), dn_.state_ptr(), sb);
        q.memcpy(s.dnc.data(), dn_.conv_state_ptr(), cb);
    }
    if (ple_conv_state_) {
        const uint64_t pb = uint64_t(kPleStateRows) * kPleSI * sizeof(float);
        s.ple.resize(pb);
        q.memcpy(s.ple.data(), ple_conv_state_, pb);
    }
    q.wait();   // D2H must land before the caller releases the turn gate
    s.hist = ple_hist_;
    s.blk_done = blk_done_;
    s.depth = depth;
    return {};
}

std::string Qwen4ExpModel::unstash_slot(const SlotState& s) {
    if (!s.depth) return "qwen4exp unstash: empty slot";
    sycl::queue& q = alloc_->queue();
    const KvCacheConfig& kc = kv_.config();
    {
        const uint64_t slice  = uint64_t(s.depth) * kc.head_dim * sizeof(sycl::half);
        const uint64_t stride = uint64_t(kc.max_ctx) * kc.head_dim;
        const uint64_t nsl    = uint64_t(kc.n_layers_full) * kc.n_kv_heads;
        for (uint64_t sl = 0; sl < nsl; ++sl) {
            q.memcpy(kv_.k_ptr() + sl * stride, s.kv_k.data() + sl * slice, slice);
            q.memcpy(kv_.v_ptr() + sl * stride, s.kv_v.data() + sl * slice, slice);
        }
    }
    if (idx_kcache_ && !s.idxk.empty()) {
        const uint32_t nf = uint32_t(blk_done_.size());
        const uint32_t ihd = cfg_.indexer_head_dim;
        const uint64_t kslice  = uint64_t(s.depth) * ihd * sizeof(sycl::half);
        const uint64_t kstride = uint64_t(max_ctx_) * ihd;
        const uint32_t nblk    = (s.depth + 3) / 4;
        const uint64_t bslice  = uint64_t(nblk) * ihd * sizeof(float);
        const uint64_t bstride = uint64_t(max_ctx_ / 4) * ihd;
        for (uint32_t l = 0; l < nf; ++l) {
            q.memcpy(idx_kcache_ + uint64_t(l) * kstride,
                     s.idxk.data() + uint64_t(l) * kslice, kslice);
            q.memcpy(blk_keys_ + uint64_t(l) * bstride,
                     s.blkk.data() + uint64_t(l) * bslice, bslice);
        }
    }
    {
        const uint64_t sb = dn_.state_elems_per_layer() *
                            dn_.config().n_layers_linear * sizeof(float);
        const uint64_t cb = dn_.conv_elems_per_layer() *
                            dn_.config().n_layers_linear * sizeof(sycl::half);
        q.memcpy(dn_.state_ptr(), s.dns.data(), sb);
        q.memcpy(dn_.conv_state_ptr(), s.dnc.data(), cb);
    }
    if (ple_conv_state_ && !s.ple.empty())
        q.memcpy(ple_conv_state_, s.ple.data(),
                 uint64_t(kPleStateRows) * kPleSI * sizeof(float));
    q.wait();
    ple_hist_ = s.hist;
    blk_done_ = s.blk_done;
    for (uint32_t L = layer_lo_; L < layer_hi_; ++L)
        if (full_idx_[L] >= 0) kv_.set_length(uint32_t(full_idx_[L]), s.depth);
    return {};
}

std::string Qwen4ExpModel::snapshot(uint32_t depth) {
    sycl::queue& q = alloc_->queue();
    const uint64_t se = dn_.state_elems_per_layer() * dn_.config().n_layers_linear;
    const uint64_t ce = dn_.conv_elems_per_layer() * dn_.config().n_layers_linear;
    if (!snap_dn_state_) {
        snap_dn_state_ = static_cast<float*>(alloc_->malloc(se * sizeof(float)));
        snap_dn_conv_  = static_cast<sycl::half*>(alloc_->malloc(ce * sizeof(sycl::half)));
        if (!snap_dn_state_ || !snap_dn_conv_) return "qwen4exp snapshot: alloc failed";
        owned_.push_back(snap_dn_state_);
        owned_.push_back(snap_dn_conv_);
        if (ple_conv_state_) {
            snap_ple_conv_ = static_cast<float*>(
                alloc_->malloc(uint64_t(kPleStateRows) * kPleSI * sizeof(float)));
            if (!snap_ple_conv_) return "qwen4exp snapshot: ple alloc failed";
            owned_.push_back(snap_ple_conv_);
        }
    }
    q.memcpy(snap_dn_state_, dn_.state_ptr(), se * sizeof(float));
    q.memcpy(snap_dn_conv_, dn_.conv_state_ptr(), ce * sizeof(sycl::half));
    if (ple_conv_state_)
        q.memcpy(snap_ple_conv_, ple_conv_state_,
                 uint64_t(kPleStateRows) * kPleSI * sizeof(float));
    q.wait();
    snap_ple_hist_ = ple_hist_;
    snap_blk_done_ = blk_done_;
    snap_depth_    = depth;
    return {};
}

std::string Qwen4ExpModel::restore() {
    if (!snap_depth_) return "qwen4exp restore: no snapshot";
    sycl::queue& q = alloc_->queue();
    const uint64_t se = dn_.state_elems_per_layer() * dn_.config().n_layers_linear;
    const uint64_t ce = dn_.conv_elems_per_layer() * dn_.config().n_layers_linear;
    q.memcpy(dn_.state_ptr(), snap_dn_state_, se * sizeof(float));
    q.memcpy(dn_.conv_state_ptr(), snap_dn_conv_, ce * sizeof(sycl::half));
    if (ple_conv_state_ && snap_ple_conv_)
        q.memcpy(ple_conv_state_, snap_ple_conv_,
                 uint64_t(kPleStateRows) * kPleSI * sizeof(float));
    q.wait();
    ple_hist_ = snap_ple_hist_;
    blk_done_ = snap_blk_done_;
    // KV + QSA idx caches are position-indexed: rewinding is just the depth
    // markers (rows beyond the snapshot depth get overwritten on refill).
    for (uint32_t L = layer_lo_; L < layer_hi_; ++L)
        if (full_idx_[L] >= 0) kv_.set_length(uint32_t(full_idx_[L]), snap_depth_);
    return {};
}

std::string Qwen4ExpModel::commit_verify(uint32_t n_commit) {
    if (!vck_T_) return "commit_verify: no verify forward captured";
    if (n_commit == 0 || n_commit > vck_T_) return "commit_verify: bad n_commit";
    sycl::queue& q = alloc_->queue();
    const uint64_t se_all = dn_.state_elems_per_layer() *
                            dn_.config().n_layers_linear;
    const uint64_t ce_all = dn_.conv_elems_per_layer() *
                            dn_.config().n_layers_linear;
    const uint64_t s = n_commit - 1;   // slab s = state after verify row s
    q.memcpy(dn_.state_ptr(), vck_dn_state_ + s * se_all, se_all * sizeof(float));
    q.memcpy(dn_.conv_state_ptr(), vck_dn_conv_ + s * ce_all,
             ce_all * sizeof(sycl::half));
    if (ple_conv_state_ && vck_ple_conv_)
        q.memcpy(ple_conv_state_,
                 vck_ple_conv_ + s * uint64_t(kPleStateRows) * kPleSI,
                 uint64_t(kPleStateRows) * kPleSI * sizeof(float));
    q.wait();
    if (cfg_.has_ple()) {
        // Adopt the history AFTER the committed row; the conv_state handle
        // is slot-owned, not part of the checkpoint.
        ple_hist_.h1 = vck_ple_hist_[s].h1;
        ple_hist_.h2 = vck_ple_hist_[s].h2;
    }
    // KV + pooled-key markers rewind exactly like restore(): rows past the
    // committed depth get overwritten on the next append.
    const uint32_t depth = vck_pos_ + n_commit;
    for (uint32_t L = layer_lo_; L < layer_hi_; ++L)
        if (full_idx_[L] >= 0) kv_.set_length(uint32_t(full_idx_[L]), depth);
    for (auto& bd : blk_done_)
        if (bd * 4 > depth) bd = depth / 4;
    vck_T_ = 0;
    return {};
}

std::string Qwen4ExpModel::alloc_wide_bank2() {
    if (wide_alt_) return {};
    if (!wide_bank0_) return "qwen4exp wide_bank2: init_runtime first";
    void* p = alloc_->malloc(uint64_t(max_chunk_) * cfg_.hc_count * cfg_.hidden *
                             sizeof(float));
    if (!p) return "qwen4exp wide_bank2: device alloc failed";
    owned_.push_back(p);
    wide_alt_ = static_cast<float*>(p);
    return {};
}

std::string Qwen4ExpModel::set_vision(const float* rows_f32, uint32_t t0,
                                      uint32_t n_rows) {
    if (!rows_f32 || !n_rows) return "qwen4exp set_vision: empty rows";
    if (t0 + n_rows > max_ctx_) return "qwen4exp set_vision: span past max_ctx";
    const uint32_t H = cfg_.hidden;
    const size_t row0 = vis_rows_.size() / H;
    vis_rows_.resize(vis_rows_.size() + size_t(n_rows) * H);
    for (size_t i = 0; i < size_t(n_rows) * H; ++i)
        vis_rows_[row0 * H + i] = sycl::half(rows_f32[i]);
    vis_spans_.push_back({t0, n_rows, uint32_t(row0)});
    return {};
}

void Qwen4ExpModel::set_mrope(const int32_t* pos3, uint32_t n_total,
                              int32_t delta) {
    mrope3_.assign(pos3, pos3 + size_t(3) * n_total);
    mrope_n_ = n_total;
    mrope_delta_ = delta;
}

void Qwen4ExpModel::reset_state() {
    sycl::queue& q = alloc_->queue();
    vis_spans_.clear();
    vis_rows_.clear();
    mrope_n_ = 0;
    mrope_delta_ = 0;
    kv_.reset();
    dn_.reset(q);
    std::fill(blk_done_.begin(), blk_done_.end(), 0u);
    snap_depth_ = 0;
    if (mtp_.loaded) {
        mtp_.kv.reset();
        mtp_.blk_done = 0;
        mtp_have_carry_ = false;
    }
    if (cfg_.has_ple() && ple_conv_state_) {
        ple_hist_.reset(cfg_.ple_eos_token);
        ple_hist_.conv_state = ple_conv_state_;
        q.memset(ple_conv_state_, 0, uint64_t(kPleStateRows) * kPleSI * 4).wait();
    }
}

void Qwen4ExpModel::run_block(uint32_t L, uint32_t T, uint32_t start_pos,
                              const int32_t* tokens_host, ParityCapture* cap) {
    sycl::queue& q = alloc_->queue();
    const uint32_t H = cfg_.hidden, D = cfg_.hc_count * H;
    const uint32_t SI = cfg_.ssm_inner, SKH = cfg_.ssm_k_heads, SVH = cfg_.ssm_v_heads;
    const uint32_t SHD = cfg_.ssm_state, CC = SI + 2u * SKH * SHD, KW = SKH * SHD;
    const uint32_t HD = cfg_.head_dim, NQ = cfg_.n_q_heads * HD, NKV = cfg_.n_kv_heads * HD;
    const uint32_t EF = cfg_.expert_ffn, SEF = cfg_.shared_expert_ffn;
    const float eps = cfg_.rms_eps;
    const float qscale = 1.0f / std::sqrt(float(SHD));
    const uint64_t per_layer_kv = uint64_t(cfg_.n_kv_heads) * max_ctx_ * HD;
    (void)D; (void)KW;
    std::vector<sycl::half> rout16(cfg_.n_experts);
    std::vector<float>      rout32(cfg_.n_experts);
    std::vector<std::pair<uint32_t, float>> topk;
    const Qwen4ExpLayer& w = layers_[L];
    auto grab = [&](sycl::half* dst, const sycl::half* src) {
        if (dst) { q.wait(); q.memcpy(dst, src, uint64_t(T) * H * 2).wait(); }
    };
    // Batched projection: gemv at T==1, gemm_fp16 (+fp32->f16 cast) at T>1.
    // oneDNN measured SLOWER on these tall-skinny projection GEMMs (chunk-2
    // prefill 11.6 s plain vs 13.6 s oneDNN, 2026-08-27 A/B) — default OFF,
    // IE_Q4E_ONEDNN=1 re-enables for future shape regimes.
    static const bool use_onednn = std::getenv("IE_Q4E_ONEDNN") != nullptr;
    // Spec-verify: the batch must reproduce T sequential T=1 steps BIT-EXACTLY
    // (the verify rows ARE the greedy reference chain). gemm's fp reduction
    // order + the fp32->f16 recast both differ from the decode gemv — per-row
    // gemv leaves are required (the qwen35_split.cpp lossless lesson).
    const bool sv = spec_verify_ && T <= 16;   // T==1 keeps decode kernels
                                               // but still checkpoints (the
                                               // overlapped row-0 pass).
    auto proj = [&](const sycl::half* X, const sycl::half* W, sycl::half* Y,
                    uint32_t K, uint32_t N) {
        if (T == 1) { gemv_fp16(q, X, W, Y, K, N); return; }
        if (sv) {
            gemv_fp16_rows(q, X, K, W, Y, N, K, N, T);
            return;
        }
        if (use_onednn) { gemm_fp16_onednn(q, X, W, Y, T, N, K); return; }
        gemm_fp16(q, X, W, gemm_c_, T, N, K);
        cast_fp32_to_fp16(q, gemm_c_, Y, uint64_t(T) * N);
    };
    // Dense-Q8 projection dispatch: T=1 decode leaf, T<=16 the row-exact
    // batched int-dot (per-row bit-identical to the T=1 leaf — the sv
    // lossless contract), T>16 dequant-once-to-scratch + GEMM so the weight
    // streams once per chunk instead of ceil(T/16) times (the pp
    // 143 -> 123 regression when dense-Q8 first defaulted on).
    auto q8_proj = [&](const sycl::half* X, const Qwen4ExpLayer::Q8W& Wq,
                       sycl::half* Y, uint32_t K, uint32_t N) {
        if (T == 1) {
            if (dense_a16()) { gemv_q8_0_soa_f16_g(q, X, Wq.qs, Wq.d, Y, K, N); return; }
            quantize_q8_1(q, X, act_q8_, K);
            gemv_q8_0_soa_q8_g(q, act_q8_, Wq.qs, Wq.d, Y, K, N);
        } else if (T <= 16) {
            if (dense_a16()) { gemv_q8_0_soa_f16_rows(q, X, Wq.qs, Wq.d, Y, K, N, T); return; }
            quantize_q8_1(q, X, act_q8T_, T * K);
            gemv_q8_0_soa_q8_batched(q, act_q8T_, Wq.qs, Wq.d, Y, K, N, T);
        } else {
            // T>16 prefill: oneDNN s8 weight-decompression GEMM reads the
            // int8 SoA plane IN PLACE against the KB-major dt scale plane —
            // per-shape 4.4-7.6x over the dequant-scratch route in the
            // 2026-08-27 bench (jit:gemm:any, rel-L2 <= 1.1e-5) BUT
            // bit-NONDETERMINISTIC run-to-run (k-slice atomics; bisected:
            // chunk-512 PPL NLL swung +-0.5% through this path) and e2e pp
            // NEUTRAL (the won GPU ms sat inside the host/upload gap). The
            // scorer instruments and run2 bit-eq demand deterministic
            // prefill, so it is OPT-IN (IE_Q4E_DNN_PREFILL=1) until a
            // deterministic s8 GEMM replaces it; default = dequant-once
            // scratch + gemm_fp16 (deterministic).
            static const bool dnn_pre =
                std::getenv("IE_Q4E_DNN_PREFILL") != nullptr;
            sycl::event ev;
            if (dnn_pre && Wq.dt && onednn_available() &&
                gemm_nt_s8_onednn(q, X, Wq.qs,
                                  reinterpret_cast<const sycl::half*>(Wq.dt),
                                  gemm_c_, T, N, K, 32, {}, &ev)) {
                cast_fp32_to_fp16(q, gemm_c_, Y, uint64_t(T) * N);
                return;
            }
            dequant_q8_soa_f16(q, Wq.qs, Wq.d, dq_ws_, K, N);
            gemm_fp16(q, X, dq_ws_, gemm_c_, T, N, K);
            cast_fp32_to_fp16(q, gemm_c_, Y, uint64_t(T) * N);
        }
    };
    // Resolve one expert's cache slot; submits miss uploads (LRU eviction).
    // The upload queue is a parameter so the prefill wave loop can land
    // refills on copyq_ (overlapped with compute); decode/verify stay on q.
    ECache& ec = ecache_[L];
    auto ensure_slot_on = [&](sycl::queue& uq, uint32_t e) -> uint8_t* {
        int32_t sl = ec.slot_of[e];
        if (sl >= 0) ++ecache_hits; else ++ecache_misses;
        if (sl < 0) {
            uint32_t v = 0; uint64_t best = UINT64_MAX;
            for (uint32_t c2 = 0; c2 < ecache_slots_; ++c2)
                if (ec.last_use[c2] < best) { best = ec.last_use[c2]; v = c2; }
            if (ec.expert_in[v] >= 0) ec.slot_of[ec.expert_in[v]] = -1;
            ec.expert_in[v] = int32_t(e);
            ec.slot_of[e]   = int16_t(v);
            sl = int32_t(v);
            uint8_t* dst = static_cast<uint8_t*>(ec.base) + uint64_t(v) * ec.slot_bytes;
            auto srcp = [&](const Qwen4ExpLayer::BankView& bv) {
                return bv.raw ? bv.raw + uint64_t(e) * bv.slice_bytes
                              : reinterpret_cast<const uint8_t*>(bv.f16) +
                                    uint64_t(e) * bv.slice_bytes;
            };
            uq.memcpy(dst + ec.gate_off, srcp(w.gate_bv), w.gate_bv.slice_bytes);
            uq.memcpy(dst + ec.up_off,   srcp(w.up_bv),   w.up_bv.slice_bytes);
            uq.memcpy(dst + ec.down_off, srcp(w.down_bv), w.down_bv.slice_bytes);
        }
        ec.last_use[uint32_t(sl)] = ++ecache_clock_;
        return static_cast<uint8_t*>(ec.base) + uint64_t(sl) * ec.slot_bytes;
    };
    auto ensure_slot = [&](uint32_t e) -> uint8_t* {
        return ensure_slot_on(q, e);
    };
    // ---- small-T expert-major MoE: decode T=1 + spec-verify T<=16 ---------
    // THE dpas migration (2026-08-27, handoff queue #2): decode and verify
    // share ONE expert-major grouped path over the prefill-certified dpas
    // tile kernels. Per-row math is row-position-independent (bvs gates it:
    // batched T=N must bit-match N x T=1), and the fp32 accumulation adds
    // each row's picks in k order (cells built k-major) — the decode chain.
    // BOTH users run this or BOTH revert (IE_Q4E_MOE_SFORM=1): flipping one
    // alone would break the verify==decode lossless contract. Expert-major
    // tiles also read each hit expert's slices ONCE (T=4: ~40 picks share
    // ~27 experts = ~1.5x less weight traffic than the pick-major s-form),
    // and this box's ie-q4k-bench has the kernels at 2.4x (gate/up dual)
    // and 3.9x (q5_1 down) over the s-form/pick tiles.
    // tk2[vt] = the row's routed (expert, weight) picks, k-ascending.
    // acc_out [T2, H] fp32 is OVERWRITTEN by the ordered reduce.
    auto small_moe = [&](uint32_t T2,
                         const std::vector<std::vector<std::pair<uint32_t, float>>>& tk2,
                         float* acc_out) {
        const uint32_t NU = cfg_.n_experts_used;
        const bool gate_raw = w.gate_bv.raw != nullptr;
        const bool down_raw = w.down_bv.raw != nullptr;
        // Flatten picks and order expert-major (stable -> (vt,k) within e).
        struct SPick { uint32_t e, vt, k; float wgt; };
        std::vector<SPick> sp;
        sp.reserve(uint64_t(T2) * NU);
        for (uint32_t vt = 0; vt < T2; ++vt)
            for (uint32_t k = 0; k < uint32_t(tk2[vt].size()); ++k)
                sp.push_back({tk2[vt][k].first, vt, k, tk2[vt][k].second});
        std::stable_sort(sp.begin(), sp.end(),
                         [](const SPick& a, const SPick& b) { return a.e < b.e; });
        const uint32_t P = uint32_t(sp.size());
        // Compact rows c = expert-major position; tables:
        //   h_tok_idx_[c]           = vt   (activation gather source row)
        //   h_cells_[vt*NU + k]     = c    (k-ordered reduce indirection)
        //   h_cells_[T2*NU + vt]    = nk   (per-row pick count)
        //   h_tok_w_[vt*NU + k]     = wgt
        //   h_tiles_[j]             = {c0, n_e, slot, c0} per distinct expert
        uint32_t J = 0;
        for (uint32_t c = 0; c < P; ) {
            uint32_t c2 = c;
            while (c2 < P && sp[c2].e == sp[c].e) ++c2;
            const int32_t si = int32_t(
                (ensure_slot(sp[c].e) - static_cast<uint8_t*>(ec.base)) /
                ec.slot_bytes);
            h_tiles_[4u * J + 0] = int32_t(c);
            h_tiles_[4u * J + 1] = int32_t(c2 - c);
            h_tiles_[4u * J + 2] = si;
            h_tiles_[4u * J + 3] = int32_t(c);
            ++J;
            c = c2;
        }
        for (uint32_t c = 0; c < P; ++c) {
            h_tok_idx_[c] = int32_t(sp[c].vt);
            h_cells_[uint64_t(sp[c].vt) * NU + sp[c].k] = int32_t(c);
            h_tok_w_[uint64_t(sp[c].vt) * NU + sp[c].k] = sp[c].wgt;
        }
        for (uint32_t vt = 0; vt < T2; ++vt)
            h_cells_[uint64_t(T2) * NU + vt] = int32_t(tk2[vt].size());
        q.memcpy(moe_tok_idx_, h_tok_idx_, P * sizeof(int32_t));
        q.memcpy(moe_tok_w_, h_tok_w_,
                 uint64_t(T2) * NU * sizeof(float));
        q.memcpy(moe_tiles_, h_tiles_.data(), 4u * J * sizeof(int32_t));
        q.memcpy(moe_cells_, h_cells_.data(),
                 (uint64_t(T2) * NU + T2) * sizeof(int32_t));
        // Kernel family: s-form tiles by default (same weight-dedup, and
        // the s-form lattice is the fast one at 1-4-row tiles — the dpas
        // family was FALSIFIED here, kept for A/B via IE_Q4E_MOE_DPAS=1).
        static const bool moe_dpas_k =
            std::getenv("IE_Q4E_MOE_DPAS") != nullptr;
        if (gate_raw) {
            quantize_q8_1s(q, mixed_, act_q8sT_, T2 * H);
            const uint32_t q8srow = (H / 32) * uint32_t(sizeof(block_q8_1s));
            gather_bytes_rows(q, static_cast<const uint8_t*>(act_q8sT_),
                              static_cast<uint8_t*>(xg_q8s_), moe_tok_idx_,
                              P, q8srow);
            if (moe_dpas_k)
                gemv_q4_K_q8d_grouped_tiles(q, xg_q8s_,
                    static_cast<const uint8_t*>(ec.base), ec.slot_bytes,
                    ec.gate_off, ec.up_off, moe_tiles_,
                    moe_gT_, moe_uT_, H, EF, J);
            else
                gemv_q4_K_q8s_grouped_tiles(q, xg_q8s_,
                    static_cast<const uint8_t*>(ec.base), ec.slot_bytes,
                    ec.gate_off, ec.up_off, moe_tiles_,
                    moe_gT_, moe_uT_, H, EF, J);
        } else {
            gather_bytes_rows(q, reinterpret_cast<const uint8_t*>(mixed_),
                              reinterpret_cast<uint8_t*>(xg16_),
                              moe_tok_idx_, P, H * 2);
            gemm_f16_grouped_tiles(q, xg16_,
                static_cast<const uint8_t*>(ec.base), ec.slot_bytes,
                ec.gate_off, ec.up_off, moe_tiles_,
                moe_gT_, moe_uT_, H, EF, J, 2);
        }
        swiglu(q, moe_gT_, moe_uT_, moe_hT_, uint64_t(P) * EF);
        if (down_raw && moe_dpas_k) {
            q51_block_rowsums(q, moe_hT_, P, EF, moe_rsum_);
            gemv_q5_1_f16d_grouped_tiles(q, moe_hT_,
                static_cast<const uint8_t*>(ec.base), ec.slot_bytes,
                ec.down_off, moe_tiles_, moe_yT_, moe_rsum_, EF, H, J);
        } else if (down_raw) {
            gemv_q5_1_grouped_tiles(q, moe_hT_,
                static_cast<const uint8_t*>(ec.base), ec.slot_bytes,
                ec.down_off, moe_tiles_, moe_yT_, EF, H, J);
        } else {
            gemm_f16_grouped_tiles(q, moe_hT_,
                static_cast<const uint8_t*>(ec.base), ec.slot_bytes,
                ec.down_off, 0, moe_tiles_, moe_yT_, nullptr, EF, H, J, 1);
        }
        {   // Ordered reduce: each row's picks in k order via the cells
            // indirection — the decode body's exact fp32 add chain.
            const int32_t* cells = moe_cells_;
            const int32_t* cnt = moe_cells_ + uint64_t(T2) * NU;
            const float* wtab = moe_tok_w_;
            const sycl::half* ys = moe_yT_;
            float* acc = acc_out;
            const uint32_t NU_ = NU, H_ = H;
            q.parallel_for(sycl::range<1>(uint64_t(T2) * H), [=](sycl::id<1> i) {
                const uint32_t vt = uint32_t(i / H_);
                const uint32_t hh = uint32_t(i % H_);
                float a = 0.f;
                const int32_t nk = cnt[vt];
                for (int32_t k = 0; k < nk; ++k) {
                    const uint32_t c = uint32_t(cells[uint64_t(vt) * NU_ + uint32_t(k)]);
                    a += wtab[uint64_t(vt) * NU_ + uint32_t(k)] *
                         float(ys[uint64_t(c) * H_ + hh]);
                }
                acc[i] = a;
            });
        }
    };

        // ---- PLE (runs FIRST in its block, reading/writing the wide state) --
        if (cfg_.is_ple_layer(L)) {
            PleHashConsts pc{};
            for (int i = 0; i < 3; ++i) pc.M[i] = cfg_.ple_layer_multipliers[i];
            for (int h = 0; h < 16; ++h) { pc.V[h] = cfg_.ple_head_vocab_sizes[h];
                                           pc.O[h] = cfg_.ple_head_offsets[h]; }
            pc.eos = cfg_.ple_eos_token;
            std::vector<uint64_t> rows(uint64_t(T) * 16);
            if (sv && vck_cap_ >= T) {
                // Per-row hash (documented chunk-identical) so the history
                // AFTER each verify row is checkpointed for commit_verify.
                for (uint32_t s = 0; s < T; ++s) {
                    qwen4_ple_hash(pc, tokens_host + s, 1, ple_hist_,
                                   rows.data() + uint64_t(s) * 16);
                    vck_ple_hist_[vck_base_ + s] = ple_hist_;
                }
            } else {
                qwen4_ple_hash(pc, tokens_host, T, ple_hist_, rows.data());
            }
            std::vector<float> E(uint64_t(T) * H);
            qwen4_ple_gather(ple_table->data, rows.data(), T, E.data());
            std::vector<sycl::half> E16(E.size());
            for (size_t i = 0; i < E.size(); ++i) E16[i] = sycl::half(E[i]);
            q.memcpy(e16_, E16.data(), E16.size() * 2).wait();
            proj(e16_, w.ple_key, ple_key_out_, H, D);
            proj(e16_, w.ple_value, ple_v_out_, H, H);
            qwen4_ple_layer(q, ple_key_out_, ple_v_out_, wide_,
                            w.ple_norm_key, w.ple_norm_query, w.ple_norm_conv,
                            w.ple_conv, ple_conv_state_,
                            ple_ws_, T, eps);
            if (sv && vck_ple_conv_ && vck_cap_ >= T) {
                // Per-position conv-state snapshot: the state after row s is
                // a pure index remap of (u, pre-chunk state_in), both still
                // live in ple_ws_ — same remap the layer's own state-update
                // kernel applies for the LAST position (qwen4_ple.cpp:194).
                const float* u  = ple_ws_;
                const float* si = ple_ws_ + uint64_t(T) * kPleSI * 2 +
                                  uint64_t(T) * kPleStreams;
                float* ck = vck_ple_conv_ +
                    uint64_t(vck_base_) * kPleStateRows * kPleSI;
                constexpr uint32_t SR = kPleStateRows, PS = kPleSI;
                q.submit([&](sycl::handler& h) {
                    h.parallel_for(sycl::range<3>(T, SR, PS), [=](sycl::id<3> id) {
                        const uint32_t s = uint32_t(id[0]);
                        const uint32_t r = uint32_t(id[1]);
                        const uint32_t c = uint32_t(id[2]);
                        const int32_t p = int32_t(s) - int32_t(SR - 1) + int32_t(r);
                        const float v = (p >= 0)
                            ? u[uint64_t(uint32_t(p)) * PS + c]
                            : si[uint64_t(uint32_t(p + int32_t(SR))) * PS + c];
                        ck[(uint64_t(s) * SR + r) * PS + c] = v;
                    });
                });
            }
        }

        // ---- attn site: HC mix -> token mixer -> HC combine ----------------
        // T > 16 routes the low-rank projections through gemm_fp16 (see
        // qwen4_hc_mix_prefill) — v2 re-streamed both weight mats per token
        // (40% of prefill GPU busy). Decode/verify stay on v2 (lossless).
        if (T > 16 && hc_prefill_gemm() && w.hc_attn_down_t)
            qwen4_hc_mix_prefill(q, wide_, w.hc_attn_norm, w.hc_attn_down_t,
                     w.hc_attn_up_t, w.hc_attn_inject, xn_ws_, hc_x16_,
                     hc_lo16_, gemm_c_, mixed_, inj_,
                     T, H, cfg_.hc_count, cfg_.hc_low_rank, eps);
        else
            qwen4_hc_mix_v2(q, wide_, w.hc_attn_norm, w.hc_attn_down, w.hc_attn_up,
                     w.hc_attn_inject, xn_ws_, lo_ws_, mixed_, inj_,
                     T, H, cfg_.hc_count, cfg_.hc_low_rank, eps);
        if (cap) grab(cap->mixed_attn, mixed_);

        if (lin_idx_[L] >= 0) {
            // Gated DeltaNet — the qwen35 chain, sigmoid output gate.
            static const bool use_q8 = qwen4exp_dense_q8();
            if (use_q8 && w.qkv_q8.qs) {
                if (dense_a16() && T <= 16) {
                    gemv_q8_0_soa_f16_rows(q, mixed_, w.qkv_q8.qs, w.qkv_q8.d,
                                           dn_qkv_, H, CC, T);
                } else if (T == 1) {
                    quantize_q8_1(q, mixed_, act_q8_, H);
                    gemv_q8_0_soa_q8_g(q, act_q8_, w.qkv_q8.qs, w.qkv_q8.d,
                                       dn_qkv_, H, CC);
                } else if (T <= 16) {
                    quantize_q8_1(q, mixed_, act_q8T_, T * H);
                    // T_MAX=16 accumulators in the batched kernel — chunk
                    // (the gemv_q4_K_q8_batched lesson; T=512 DEVICE_LOST
                    // 2026-08-27 when this env path first ran at chunk size).
                    for (uint32_t t0 = 0; t0 < T; t0 += 16) {
                        const uint32_t tn = std::min(16u, T - t0);
                        gemv_q8_0_soa_q8_batched(q,
                            static_cast<const uint8_t*>(act_q8T_) +
                                uint64_t(t0) * (H / 32) * sizeof(block_q8_1x),
                            w.qkv_q8.qs, w.qkv_q8.d,
                            dn_qkv_ + uint64_t(t0) * CC, H, CC, tn);
                    }
                } else {
                    q8_proj(mixed_, w.qkv_q8, dn_qkv_, H, CC);
                }
            } else {
                proj(mixed_, w.attn_qkv, dn_qkv_, H, CC);
            }
            sycl::half* cst = dn_.conv_state_ptr() +
                uint64_t(lin_idx_[L]) * dn_.conv_elems_per_layer();
            if (sv && vck_dn_conv_ && vck_cap_ >= T) {
                // Per-position conv-state snapshot BEFORE the in-place conv:
                // the state after row s is an index remap of (pre-state,
                // dn_qkv_) — byte-identical to what per-step T=1 convs would
                // leave (the qwen35_split.cpp verify snapshot, ported).
                const uint32_t km1 = cfg_.ssm_conv_kernel - 1;
                const uint32_t ch  = CC;
                const sycl::half* xin = dn_qkv_;
                const sycl::half* pst = cst;
                const uint64_t ce = dn_.conv_elems_per_layer();
                const uint64_t ck_step =
                    uint64_t(dn_.config().n_layers_linear) * ce;
                sycl::half* ck_base = vck_dn_conv_ + vck_base_ * ck_step +
                                      uint64_t(lin_idx_[L]) * ce;
                q.submit([&](sycl::handler& h) {
                    h.parallel_for(sycl::range<3>(T, km1, ch), [=](sycl::id<3> id) {
                        const uint32_t s = uint32_t(id[0]);
                        const uint32_t p = uint32_t(id[1]);
                        const uint32_t c = uint32_t(id[2]);
                        const int32_t i = int32_t(s) - int32_t(km1 - 1) + int32_t(p);
                        const sycl::half v = (i >= 0)
                            ? xin[uint64_t(uint32_t(i)) * ch + c]
                            : pst[uint64_t(uint32_t(i + int32_t(km1))) * ch + c];
                        ck_base[uint64_t(s) * ck_step + uint64_t(p) * ch + c] = v;
                    });
                });
            }
            depthwise_conv1d_causal(q, dn_qkv_, w.ssm_conv, cst, dn_conv_,
                                    T, CC, cfg_.ssm_conv_kernel);
            cast_qkv_split_fp16_to_fp32(q, dn_conv_, dn_qpre_, dn_kpre_, dn_vpre_,
                                        T, KW, SI);
            l2_norm_scale(q, dn_qpre_, dn_qpre_, T * SKH, SHD, qscale, 1e-6f);
            l2_norm_scale(q, dn_kpre_, dn_kpre_, T * SKH, SHD, 1.0f, 1e-6f);
            repeat_interleave_heads(q, dn_qpre_, dn_qrep_, T, SKH, SHD, SVH / SKH);
            repeat_interleave_heads(q, dn_kpre_, dn_krep_, T, SKH, SHD, SVH / SKH);
            gemv_fp16_rows_dual(q, mixed_, H, w.ssm_alpha, dn_ab64_,
                                w.ssm_beta, dn_ab64_ + 64, 128, H, 64, T);
            extract_cols(q, dn_ab64_, dn_a48_, T, SVH, 128);
            extract_cols(q, dn_ab64_ + 64, dn_b48_, T, SVH, 128);
            compute_g_beta_h16(q, dn_a48_, dn_b48_, w.ssm_a, w.ssm_dt,
                               dn_g_, dn_beta_, T, SVH);
            float* st = dn_.state_ptr() +
                uint64_t(lin_idx_[L]) * dn_.state_elems_per_layer();
            if (sv && vck_dn_state_ && vck_cap_ >= T) {
                const uint64_t se = dn_.state_elems_per_layer();
                const uint64_t se_step =
                    uint64_t(dn_.config().n_layers_linear) * se;
                deltanet_recurrence_ckpt(q, dn_qrep_, dn_krep_, dn_vpre_,
                    dn_g_, dn_beta_, st, dn_out_,
                    vck_dn_state_ + vck_base_ * se_step +
                        uint64_t(lin_idx_[L]) * se,
                    se_step, T, SVH, SHD, SHD);
            } else {
                deltanet_recurrence(q, dn_qrep_, dn_krep_, dn_vpre_, dn_g_, dn_beta_,
                                    st, dn_out_, 1, T, SVH, SHD, SHD);
            }
            if (use_q8 && w.gate_q8.qs) {   // reuses the activation quantized for qkv
                if (dense_a16() && T <= 16)
                    gemv_q8_0_soa_f16_rows(q, mixed_, w.gate_q8.qs, w.gate_q8.d,
                                           dn_z_, H, SI, T);
                else if (T == 1)
                    gemv_q8_0_soa_q8_g(q, act_q8_, w.gate_q8.qs, w.gate_q8.d,
                                       dn_z_, H, SI);
                else if (T <= 16)
                    for (uint32_t t0 = 0; t0 < T; t0 += 16)
                        gemv_q8_0_soa_q8_batched(q,
                            static_cast<const uint8_t*>(act_q8T_) +
                                uint64_t(t0) * (H / 32) * sizeof(block_q8_1x),
                            w.gate_q8.qs, w.gate_q8.d,
                            dn_z_ + uint64_t(t0) * SI, H, SI,
                            std::min(16u, T - t0));
                else
                    q8_proj(mixed_, w.gate_q8, dn_z_, H, SI);
            } else {
                proj(mixed_, w.attn_gate, dn_z_, H, SI);
            }
            gated_rms_norm(q, dn_out_, dn_z_, w.ssm_norm, dn_gn_,
                           T * SVH, SHD, eps, /*sigmoid_gate=*/true);
            if (use_q8 && w.out_q8.qs) {
                if (dense_a16() && T <= 16) {
                    gemv_q8_0_soa_f16_rows(q, dn_gn_, w.out_q8.qs, w.out_q8.d,
                                           blockout_, SI, H, T);
                } else if (T == 1) {
                    quantize_q8_1(q, dn_gn_, act_q8_, SI);
                    gemv_q8_0_soa_q8_g(q, act_q8_, w.out_q8.qs, w.out_q8.d,
                                       blockout_, SI, H);
                } else if (T <= 16) {
                    quantize_q8_1(q, dn_gn_, act_q8T_, T * SI);
                    for (uint32_t t0 = 0; t0 < T; t0 += 16)
                        gemv_q8_0_soa_q8_batched(q,
                            static_cast<const uint8_t*>(act_q8T_) +
                                uint64_t(t0) * (SI / 32) * sizeof(block_q8_1x),
                            w.out_q8.qs, w.out_q8.d,
                            blockout_ + uint64_t(t0) * H, SI, H,
                            std::min(16u, T - t0));
                } else {
                    q8_proj(dn_gn_, w.out_q8, blockout_, SI, H);
                }
            } else {
                proj(dn_gn_, w.ssm_out, blockout_, SI, H);
            }
        } else {
            // Full attention, gate fused in attn_q; dense == QSA in this regime.
            if (w.q_q8.qs) q8_proj(mixed_, w.q_q8, qg_, H, NQ * 2);
            else           proj(mixed_, w.attn_q, qg_, H, NQ * 2);
            proj(mixed_, w.attn_k, ak_, H, NKV);
            proj(mixed_, w.attn_v, av_, H, NKV);
            split_q_gate_per_head(q, qg_, aq_, agate_, T, cfg_.n_q_heads, HD);
            rms_norm_f32w(q, aq_, w.attn_q_norm, aq_, T * cfg_.n_q_heads, HD, eps);
            rms_norm_f32w(q, ak_, w.attn_k_norm, ak_, T * cfg_.n_kv_heads, HD, eps);
            if (mrope_n_) {   // vision: 3-stream interleaved M-RoPE (bit-eq to
                              // rope_partial when the streams are equal)
                rope_imrope3(q, aq_, d_pos3_, aq_, T, cfg_.n_q_heads, HD,
                             cfg_.rope_dim, cfg_.rope_theta);
                rope_imrope3(q, ak_, d_pos3_, ak_, T, cfg_.n_kv_heads, HD,
                             cfg_.rope_dim, cfg_.rope_theta);
            } else {
                rope_partial(q, aq_, d_pos_, aq_, T, cfg_.n_q_heads, HD,
                             cfg_.rope_dim, cfg_.rope_theta);
                rope_partial(q, ak_, d_pos_, ak_, T, cfg_.n_kv_heads, HD,
                             cfg_.rope_dim, cfg_.rope_theta);
            }
            const uint32_t li = uint32_t(full_idx_[L]);
            sycl::half* kc = kv_.k_ptr() + per_layer_kv * li;
            sycl::half* vc = kv_.v_ptr() + per_layer_kv * li;
            // Dense == QSA exactly while every query's visible set fits the
            // budget: n_ctx <= top_k + ratio - 1 (2051 on the real file).
            const uint32_t dense_ok =
                cfg_.indexer_top_k ? cfg_.indexer_top_k + 3 : UINT32_MAX;
            // ---- QSA indexer maintenance (every chunk, cheap): raw keys are
            // cached un-normed/un-roped; complete blocks pool ONCE. ----------
            if (cfg_.indexer_top_k) {
                const uint32_t IHD = cfg_.indexer_head_dim;
                const uint32_t IQW = cfg_.indexer_n_heads * IHD;
                proj(mixed_, w.idx_q_proj, idx_q_, H, IQW);
                proj(mixed_, w.idx_k_proj, idx_kraw_, H, IHD);
                rms_norm_f32w(q, idx_q_, w.idx_q_norm, idx_q_,
                              T * cfg_.indexer_n_heads, IHD, eps);
                if (mrope_n_)   // indexer q ropes with the SAME mrope streams as
                                // attention (HF :1416 → :793, one cos/sin table)
                    rope_imrope3(q, idx_q_, d_pos3_, idx_q_, T,
                                 cfg_.indexer_n_heads, IHD, cfg_.rope_dim,
                                 cfg_.rope_theta);
                else
                    rope_partial(q, idx_q_, d_pos_, idx_q_, T,
                                 cfg_.indexer_n_heads, IHD, cfg_.rope_dim,
                                 cfg_.rope_theta);
                q.memcpy(idx_kcache_ + (uint64_t(li) * max_ctx_ + start_pos) * IHD,
                         idx_kraw_, uint64_t(T) * IHD * sizeof(sycl::half));
                const uint32_t nb_tot = (start_pos + T) / 4;
                if (nb_tot > blk_done_[li]) {
                    qsa_pool_norm_rope(q, idx_kcache_ + uint64_t(li) * max_ctx_ * IHD,
                                       w.idx_k_norm,
                                       blk_keys_ + uint64_t(li) * (max_ctx_ / 4) * IHD,
                                       blk_done_[li], nb_tot - blk_done_[li],
                                       cfg_.rope_theta, eps);
                    blk_done_[li] = nb_tot;
                }
            }
            if (start_pos + T <= dense_ok) {
                if (sv)
                    // Spec-verify: LOOP the decode kernel over the T rows —
                    // bit-identical to T sequential T=1 steps (the FA2 tile's
                    // online-softmax accumulation order is NOT; same lesson
                    // as qwen35_split.cpp's verify branch).
                    for (uint32_t t = 0; t < T; ++t)
                        full_attention(q, aq_ + uint64_t(t) * NQ,
                                       ak_ + uint64_t(t) * NKV,
                                       av_ + uint64_t(t) * NKV, kc, vc,
                                       attn_out_ + uint64_t(t) * NQ, 1,
                                       start_pos + t, cfg_.n_q_heads,
                                       cfg_.n_kv_heads, HD, max_ctx_);
                else if (T > 1)
                    // hd256 wide-tile FA2 (qwen35-certified argmax-identical);
                    // naive re-reads the whole KV per query row — measured
                    // 63 s for one 1024-chunk at pos 1033.
                    full_attention_fa2_prefill_tile_gemma(
                        q, aq_, ak_, av_, kc, vc, attn_out_, T, start_pos,
                        cfg_.n_q_heads, cfg_.n_kv_heads, HD, max_ctx_,
                        /*window=*/0);
                else
                    full_attention(q, aq_, ak_, av_, kc, vc, attn_out_, T,
                                   start_pos, cfg_.n_q_heads, cfg_.n_kv_heads,
                                   HD, max_ctx_);
            } else {
                // Per-token QSA: score visible complete blocks, top-512 by
                // prob (ties -> lower block id, deterministic), expand x4 +
                // always-visible tail, gather, and attend the selected set
                // with the existing kernel (current row appended into the
                // gathered scratch at n_sel).
                // Fully-DEVICE per-token selection chain: score -> radix
                // top-k -> expand(+tail, self incl.) -> gather -> attend.
                // No host round-trips: the host loop only SUBMITS, so long-ctx
                // prefill stays GPU-bound (the v0 host top-k serialized the
                // GPUs at every token — measured as a pegged CPU core and idle
                // GPUs, 2026-08-27).
                kv_.append(q, li, start_pos, T, ak_, av_);
                const uint32_t IQW = cfg_.indexer_n_heads * cfg_.indexer_head_dim;
                const uint32_t kblk = cfg_.indexer_top_k / 4;              // 512
                if (T > 1) {
                    // Chunk-batched selection: 3 launches for the whole chunk
                    // (was 3 x T), bit-identical per token. Gather+attend stay
                    // per token (their scratch is single-token sized; grouping
                    // them is the next lever if the profile still shows QSA).
                    const uint32_t nb_max = (start_pos + T) / 4;
                    qsa_score_batched(q, idx_q_, IQW,
                        blk_keys_ + uint64_t(li) * (max_ctx_ / 4) * cfg_.indexer_head_dim,
                        qsa_scoresT_, max_ctx_ / 4, d_pos_, nb_max, T);
                    ds4_indexer_topk(q, qsa_scoresT_, d_pos_, qsa_blk_selT_,
                                     T, max_ctx_ / 4, kblk, /*compress_rate=*/4);
                    qsa_expand_sel_batched(q, qsa_blk_selT_, kblk, kblk, kblk,
                                           d_pos_, qsa_selT_, qsa_sel_cap_,
                                           qsa_nselT_, T);
                    for (uint32_t g0 = 0; g0 < T; g0 += 64) {
                        const uint32_t gT = std::min(64u, T - g0);
                        qsa_attend_fused_grp(q, aq_ + uint64_t(g0) * NQ, kc, vc,
                            qsa_selT_ + uint64_t(g0) * qsa_sel_cap_,
                            qsa_sel_cap_, qsa_nselT_ + g0,
                            attn_out_ + uint64_t(g0) * NQ, qsa_partG_,
                            cfg_.n_q_heads, cfg_.n_kv_heads, HD, max_ctx_, gT);
                    }
                } else
                for (uint32_t t = 0; t < T; ++t) {
                    const uint32_t pos = start_pos + t;
                    const uint32_t n_vis = (pos + 1) / 4;      // complete blocks
                    const bool select = n_vis > kblk;
                    if (select) {
                        qsa_score(q, idx_q_ + uint64_t(t) * IQW,
                                  blk_keys_ + uint64_t(li) * (max_ctx_ / 4) * cfg_.indexer_head_dim,
                                  qsa_scores_, n_vis);
                        ds4_indexer_topk(q, qsa_scores_, d_pos_ + t, qsa_blk_sel_,
                                         1, n_vis, kblk, /*compress_rate=*/4);
                    }
                    qsa_expand_sel(q, select ? qsa_blk_sel_ : nullptr, kblk,
                                   int32_t(pos), qsa_sel_, qsa_nsel_);
                    qsa_gather_n(q, kc, vc, qsa_gk_, qsa_gv_, qsa_sel_, qsa_nsel_,
                                 qsa_sel_cap_, cfg_.n_kv_heads, HD, max_ctx_,
                                 qsa_sel_cap_);
                    qsa_attend_splitk(q, aq_ + uint64_t(t) * NQ, qsa_gk_, qsa_gv_,
                                      qsa_nsel_, attn_out_ + uint64_t(t) * NQ,
                                      qsa_part_,
                                      cfg_.n_q_heads, cfg_.n_kv_heads, HD,
                                      qsa_sel_cap_);
                }
            }
            kv_.set_length(li, start_pos + T);
            sigmoid_gate(q, attn_out_, agate_, attn_out_, uint64_t(T) * NQ);
            if (w.attnout_q8.qs) q8_proj(attn_out_, w.attnout_q8, blockout_, NQ, H);
            else                 proj(attn_out_, w.attn_output, blockout_, NQ, H);
        }
        if (cap) grab(cap->mixer_out, blockout_);
        {   // Diagnostic: IE_Q4E_DUMP_BLK=<file> appends blockout_ [T,H] f16 per stage
            static const char* dump_blk = std::getenv("IE_Q4E_DUMP_BLK");
            if (dump_blk) {
                q.wait();
                std::vector<sycl::half> hb(uint64_t(T) * H);
                q.memcpy(hb.data(), blockout_, hb.size() * 2).wait();
                if (FILE* f = std::fopen(dump_blk, "ab")) {
                    std::fwrite(hb.data(), 2, hb.size(), f); std::fclose(f);
                }
            }
        }
        qwen4_hc_combine(q, wide_, blockout_, inj_, T, H, cfg_.hc_count);

        // ---- ffn site: HC mix -> MoE (+ shared expert) -> HC combine -------
        if (T > 16 && hc_prefill_gemm() && w.hc_ffn_down_t)
            qwen4_hc_mix_prefill(q, wide_, w.hc_ffn_norm, w.hc_ffn_down_t,
                     w.hc_ffn_up_t, w.hc_ffn_inject, xn_ws_, hc_x16_,
                     hc_lo16_, gemm_c_, mixed_, inj_,
                     T, H, cfg_.hc_count, cfg_.hc_low_rank, eps);
        else
            qwen4_hc_mix_v2(q, wide_, w.hc_ffn_norm, w.hc_ffn_down, w.hc_ffn_up,
                     w.hc_ffn_inject, xn_ws_, lo_ws_, mixed_, inj_,
                     T, H, cfg_.hc_count, cfg_.hc_low_rank, eps);
        if (cap) grab(cap->mixed_ffn, mixed_);
        // T=1 decode rides the spec-verify grouped MoE body (ONE launch each
        // for gate+up / swiglu / down / reduce instead of ~5 per pick). The
        // verify body is certified row-exact to the per-pick chain (501/501
        // lossless, 2026-08-29); reroute gated by greedy-text A/B: byte-eq
        // output, 19.99 -> 23.82 tok/s, q4k launches 940 -> 47 (2026-08-31).
        // Default ON; IE_Q4E_DECODE_GROUPED=0 restores the per-pick body.
        static const bool decode_grouped = [] {
            const char* v = std::getenv("IE_Q4E_DECODE_GROUPED");
            return !(v && v[0] == '0');
        }();
        const bool dg1 = decode_grouped && T == 1 && !cap;
        if (T == 1 && !dg1) {
            // ---- decode: per-token routed experts from the VRAM cache ------
            const sycl::half* x = mixed_;
            gemv_fp16(q, x, w.ffn_gate_inp, router_out_, H, cfg_.n_experts);
            q.memcpy(rout16.data(), router_out_, cfg_.n_experts * 2).wait();
            for (uint32_t e = 0; e < cfg_.n_experts; ++e) rout32[e] = float(rout16[e]);
            route_from_logits(rout32.data(), cfg_.n_experts, cfg_.n_experts_used, topk);
            // small_moe (expert-major small-T MoE) is OPT-IN: BOTH kernel
            // families were FALSIFIED as verify/decode defaults on spec2
            // A/B (2026-08-28 00:5x): dpas tiles round 95.5 -> 101.9 ms;
            // s-form tiles round 95.4 -> 100.5-101.3 (verifyA +4) — at
            // 1-4-row jobs the expert-major table has FEWER WGs (~27 jobs
            // vs ~40 picks) and the weight dedup saves nothing (L2 already
            // absorbs duplicate expert reads). The pick-major s-form
            // grouped kernels are locally optimal at these shapes; the
            // remaining down-kernel headroom is INSIDE gemv_q5_1_grouped
            // (19.1 ms/round at ~110 GB/s, Q5_1 unpack ALU-bound).
            // IE_Q4E_MOE_EXPMAJOR=1 re-enables (plus IE_Q4E_MOE_DPAS=1 for
            // the dpas family) — flips BOTH decode and verify (bvs binds).
            static const bool moe_expmajor =
                std::getenv("IE_Q4E_MOE_EXPMAJOR") != nullptr;
            if (moe_expmajor) {
                std::vector<std::vector<std::pair<uint32_t, float>>> tk1(1);
                tk1[0] = topk;
                small_moe(1, tk1, moe_acc_);
            } else {
            q.memset(moe_acc_, 0, H * sizeof(float));
            if (w.gate_bv.raw) quantize_q8_1s(q, x, act_q8s_, H);
            struct Pick { float wgt; uint8_t* slot; };
            std::vector<Pick> picks;
            picks.reserve(topk.size());
            for (const auto& [e, wgt] : topk) picks.push_back({wgt, ensure_slot(e)});
            for (const Pick& pk : picks) {
                const void* gp = pk.slot + ec.gate_off;
                const void* up = pk.slot + ec.up_off;
                const void* dp = pk.slot + ec.down_off;
                if (w.gate_bv.raw) {
                    gemv_q4_K_q8s(q, act_q8s_, gp, moe_g_, H, EF);
                    gemv_q4_K_q8s(q, act_q8s_, up, moe_u_, H, EF);
                } else {
                    gemv_fp16(q, x, static_cast<const sycl::half*>(gp), moe_g_, H, EF);
                    gemv_fp16(q, x, static_cast<const sycl::half*>(up), moe_u_, H, EF);
                }
                swiglu(q, moe_g_, moe_u_, moe_h_, EF);
                if (w.down_bv.raw)
                    gemv_q5_1(q, moe_h_, dp, moe_y_, EF, H);
                else
                    gemv_fp16(q, moe_h_, static_cast<const sycl::half*>(dp), moe_y_, EF, H);
                axpy_f16_to_f32(q, moe_acc_, moe_y_, pk.wgt, H);
            }
            }
            if (w.shg_q8.qs && dense_a16()) {
                gemv_q8_0_soa_f16_rows_dual(q, x, w.shg_q8.qs, w.shg_q8.d, sh_g_,
                                            w.shu_q8.qs, w.shu_q8.d, sh_u_,
                                            H, SEF, 1);
                swiglu(q, sh_g_, sh_u_, sh_h_, SEF);
                gemv_q8_0_soa_f16_g(q, sh_h_, w.shd_q8.qs, w.shd_q8.d, sh_y_, SEF, H);
            } else if (w.shg_q8.qs) {
                quantize_q8_1(q, x, act_q8_, H);
                gemv_q8_0_soa_q8_g(q, act_q8_, w.shg_q8.qs, w.shg_q8.d, sh_g_, H, SEF);
                gemv_q8_0_soa_q8_g(q, act_q8_, w.shu_q8.qs, w.shu_q8.d, sh_u_, H, SEF);
                swiglu(q, sh_g_, sh_u_, sh_h_, SEF);
                quantize_q8_1(q, sh_h_, act_q8_, SEF);
                gemv_q8_0_soa_q8_g(q, act_q8_, w.shd_q8.qs, w.shd_q8.d, sh_y_, SEF, H);
            } else {
                gemv_fp16(q, x, w.ffn_gate_shexp, sh_g_, H, SEF);
                gemv_fp16(q, x, w.ffn_up_shexp,   sh_u_, H, SEF);
                swiglu(q, sh_g_, sh_u_, sh_h_, SEF);
                gemv_fp16(q, sh_h_, w.ffn_down_shexp, sh_y_, SEF, H);
            }
            gemv_fp16(q, x, w.ffn_gate_inp_shexp, sh_s_, H, 1);
            sigmoid_scalar_axpy(q, moe_acc_, sh_y_, sh_s_, H);
            cast_f32_to_f16(q, moe_acc_, blockout_, H);
        } else if (sv || dg1) {
            // ---- spec-verify MoE: expert-major weight reads, decode math ---
            // (also the T=1 grouped-decode body — see dg1 above)
            // Row-exact contract: (1) route from the f16-rounded per-row gemv
            // logits exactly as decode does (prefill's fp32-gemm routing flips
            // ~0.8% of near-tie picks — lossless-fatal); (2) expert kernels
            // are the per-row-exact batched variants; (3) fp32 accumulation
            // replays each row's top-k axpy sequence IN ORDER, shared expert
            // last — the same add order as the decode body.
            const uint32_t NU = cfg_.n_experts_used;
            gemv_fp16_rows(q, mixed_, H, w.ffn_gate_inp, router_all_,
                           cfg_.n_experts, H, cfg_.n_experts, T);
            sycl::half* r16 = h_router16_;   // pinned D2H target (no staging)
            const auto trw0 = std::chrono::steady_clock::now();
            // (Spin-polling this wait instead of blocking was A/B'd 2026-08-31:
            // 23.80 == 23.80 tok/s — wake-up latency is NOT the decode bubble.)
            q.memcpy(r16, router_all_, uint64_t(T) * cfg_.n_experts * 2).wait();
            const auto trw1 = std::chrono::steady_clock::now();
            t_route_wait += std::chrono::duration<double>(trw1 - trw0).count();
            std::vector<std::vector<std::pair<uint32_t, float>>> tk(T);
            for (uint32_t vt = 0; vt < T; ++vt) {
                for (uint32_t e = 0; e < cfg_.n_experts; ++e)
                    rout32[e] = float(r16[uint64_t(vt) * cfg_.n_experts + e]);
                route_from_logits(rout32.data(), cfg_.n_experts, NU, tk[vt]);
            }
            t_route_host += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - trw1).count();
            // Per-piece bank dispatch, mirroring the decode body: the real
            // UD file MIXES banks per layer (2 gate/up banks and 5 down banks
            // are F16-dequantized at load; raw == nullptr there). Coupling
            // them into one flag read raw Q4_K bytes as F16 on the mixed
            // layers -> the 2026-08-27 NaN regression.
            const bool gate_q8  = w.gate_bv.raw != nullptr;
            const bool down_q51 = w.down_bv.raw != nullptr;
            const auto tms0 = std::chrono::steady_clock::now();
            // Shared expert FIRST: its kernels occupy the compute queue while
            // the routed experts' miss fills stream on copyq_ below (the
            // router wait above drained q, so no in-flight reader races).
            // In-order q keeps the moe_gT_/uT_/hT_ scratch reuse safe.
            if (w.shg_q8.qs && dense_a16()) {
                // Row-exact A16 (same leaves as the decode body — the rows
                // kernel is per-row bit-identical to the T=1 leaf).
                gemv_q8_0_soa_f16_rows_dual(q, mixed_,
                                            w.shg_q8.qs, w.shg_q8.d, moe_gT_,
                                            w.shu_q8.qs, w.shu_q8.d, moe_uT_,
                                            H, SEF, T);
                swiglu(q, moe_gT_, moe_uT_, moe_hT_, uint64_t(T) * SEF);
                gemv_q8_0_soa_f16_rows(q, moe_hT_, w.shd_q8.qs, w.shd_q8.d,
                                       xg16_, SEF, H, T);
            } else if (w.shg_q8.qs) {
                // Row-exact int-dot (same leaves as the decode body — the
                // batched kernel is per-row bit-identical to the T=1 _g).
                quantize_q8_1(q, mixed_, act_q8T_, T * H);
                gemv_q8_0_soa_q8_batched(q, act_q8T_, w.shg_q8.qs, w.shg_q8.d,
                                         moe_gT_, H, SEF, T);
                gemv_q8_0_soa_q8_batched(q, act_q8T_, w.shu_q8.qs, w.shu_q8.d,
                                         moe_uT_, H, SEF, T);
                swiglu(q, moe_gT_, moe_uT_, moe_hT_, uint64_t(T) * SEF);
                quantize_q8_1(q, moe_hT_, act_q8T_, T * SEF);
                gemv_q8_0_soa_q8_batched(q, act_q8T_, w.shd_q8.qs, w.shd_q8.d,
                                         xg16_, SEF, H, T);
            } else {
                gemv_fp16_rows(q, mixed_, H, w.ffn_gate_shexp, moe_gT_, SEF, H, SEF, T);
                gemv_fp16_rows(q, mixed_, H, w.ffn_up_shexp,   moe_uT_, SEF, H, SEF, T);
                swiglu(q, moe_gT_, moe_uT_, moe_hT_, uint64_t(T) * SEF);
                gemv_fp16_rows(q, moe_hT_, SEF, w.ffn_down_shexp, xg16_, H, SEF, H, T);
            }
            gemv_fp16_dotrows(q, mixed_, H, w.ffn_gate_inp_shexp, sgateT_, H, T);
            {
                // Grouped path: ONE launch each for gate+up, swiglu, down,
                // and the ordered reduce — the verify launch storm was ~113
                // launches/layer and half the verify wall was host submit
                // time (2026-08-27 profile).
                if (gate_q8) quantize_q8_1s(q, mixed_, act_q8sT_, T * H);
                // Slots resolved per expert in (vt, k) first-use order, then
                // the pick-major job list references them.
                // Same bulk-resolve invariant as the grouped prefill path:
                // the batch's distinct (owned) experts must fit the cache
                // simultaneously or later resolves evict earlier jobs'
                // slots. K=3 verify unions (<=41) fit every observed config;
                // fail fast rather than corrupt if a K/cache combo doesn't.
                {
                    std::vector<char> seen(cfg_.n_experts, 0);
                    uint32_t uni = 0;
                    for (uint32_t vt = 0; vt < T; ++vt)
                        for (const auto& pr : tk[vt]) {
                            const uint32_t e = pr.first;
                            if (e >= tp_e_lo_ && e < tp_e_hi_ && !seen[e]) {
                                seen[e] = 1; ++uni;
                            }
                        }
                    if (uni > ecache_slots_) {
                        block_err_ = "qwen4exp verify: expert union " +
                            std::to_string(uni) + " > ecache slots " +
                            std::to_string(ecache_slots_) +
                            " (layer " + std::to_string(L) +
                            ") — raise IE_Q4E_ECACHE_GB or lower K";
                        return;
                    }
                }
                static const bool moe_expmajor2 =
                    std::getenv("IE_Q4E_MOE_EXPMAJOR") != nullptr;
                if (moe_expmajor2 && !tp_sync_) {
                    // OPT-IN dpas expert-major verify MoE (falsified as the
                    // default — see the decode-side comment; kept as the A/B
                    // instrument for future small-M kernel work). Same
                    // small_moe as decode T=1 (one flag flips both). The
                    // spec3 expert-parallel path (tp_sync_) always keeps the
                    // pick-major s-form kernels below: its exchange protocol
                    // scatters peer cells into moe_yT_ at (vt,k)-cell rows,
                    // which the compact-row layout would break.
                    small_moe(T, tk, moe_accT_);
                } else {
                const uint64_t misses_before = ecache_misses;
                std::vector<int16_t> slot_of(cfg_.n_experts, -1);
                if (tp_sync_ && tp_cell_owner_.size() < uint64_t(T) * NU)
                    tp_cell_owner_.assign(uint64_t(T) * NU, 0);
                uint32_t P = 0;
                for (uint32_t vt = 0; vt < T; ++vt) {
                    h_tok_idx_[3u * 16 * NU + vt] = int32_t(tk[vt].size());
                    for (uint32_t k = 0; k < uint32_t(tk[vt].size()); ++k) {
                        const uint32_t e = tk[vt][k].first;
                        h_tok_w_[vt * NU + k]  = tk[vt][k].second;
                        const bool own = e >= tp_e_lo_ && e < tp_e_hi_;
                        if (tp_sync_) tp_cell_owner_[vt * NU + k] = own ? 1 : 0;
                        // Verify expert-parallel: only owned experts get a
                        // job; peer cells arrive via tp_sync_ below.
                        if (!own) continue;
                        if (slot_of[e] < 0)
                            slot_of[e] = int16_t(
                                (ensure_slot_on(copyq_ ? *copyq_ : q, e) -
                                 static_cast<uint8_t*>(ec.base)) /
                                ec.slot_bytes);
                        h_tok_idx_[3u * P + 0] = int32_t(vt);
                        h_tok_idx_[3u * P + 1] = int32_t(slot_of[e]);
                        h_tok_idx_[3u * P + 2] = int32_t(vt * NU + k);
                        ++P;
                    }
                }
                if (misses_before != ecache_misses && copyq_) {
                    // Order every subsequent q submission behind the copyq_
                    // fills (cross-queue; q drained at the router wait, so
                    // this is the only ordering the fills need).
                    q.ext_oneapi_submit_barrier(
                        {copyq_->ext_oneapi_submit_barrier()});
                }
                q.memcpy(moe_tok_idx_, h_tok_idx_,
                         (3u * 16 * NU + T) * sizeof(int32_t));
                q.memcpy(moe_tok_w_, h_tok_w_,
                         uint64_t(T) * NU * sizeof(float));
                static const bool no_g1  = std::getenv("IE_Q4E_SV_NO_G1") != nullptr;
                static const bool no_g2  = std::getenv("IE_Q4E_SV_NO_G2") != nullptr;
                static const bool no_red = std::getenv("IE_Q4E_SV_NO_RED") != nullptr;
                if (!gate_q8) {
                    // F16 gate/up banks (pre-dequantized at load): per-pick
                    // decode leaves on the slot's F16 slices.
                    for (uint32_t pk = 0; pk < P; ++pk) {
                        const uint32_t vt = uint32_t(h_tok_idx_[3u * pk + 0]);
                        const uint8_t* slot = static_cast<uint8_t*>(ec.base) +
                            uint64_t(h_tok_idx_[3u * pk + 1]) * ec.slot_bytes;
                        const sycl::half* x = mixed_ + uint64_t(vt) * H;
                        gemv_fp16(q, x,
                                  reinterpret_cast<const sycl::half*>(slot + ec.gate_off),
                                  moe_gT_ + uint64_t(pk) * EF, H, EF);
                        gemv_fp16(q, x,
                                  reinterpret_cast<const sycl::half*>(slot + ec.up_off),
                                  moe_uT_ + uint64_t(pk) * EF, H, EF);
                    }
                } else if (no_g1) {
                    // Bisect alt: per-pick solo gate/up (decode kernels).
                    for (uint32_t pk = 0; pk < P; ++pk) {
                        const uint32_t vt = uint32_t(h_tok_idx_[3u * pk + 0]);
                        const uint8_t* slot = static_cast<uint8_t*>(ec.base) +
                            uint64_t(h_tok_idx_[3u * pk + 1]) * ec.slot_bytes;
                        const uint8_t* xq = static_cast<const uint8_t*>(act_q8sT_) +
                            uint64_t(vt) * (H / 32) * sizeof(block_q8_1s);
                        gemv_q4_K_q8s(q, xq, slot + ec.gate_off,
                                      moe_gT_ + uint64_t(pk) * EF, H, EF);
                        gemv_q4_K_q8s(q, xq, slot + ec.up_off,
                                      moe_uT_ + uint64_t(pk) * EF, H, EF);
                    }
                } else {
                    gemv_q4_K_q8s_grouped_dual(q, act_q8sT_,
                        static_cast<const uint8_t*>(ec.base), ec.slot_bytes,
                        ec.gate_off, ec.up_off, moe_tok_idx_,
                        moe_gT_, moe_uT_, H, EF, P);
                }
                swiglu(q, moe_gT_, moe_uT_, moe_hT_, uint64_t(P) * EF);
                if (!down_q51) {
                    for (uint32_t pk = 0; pk < P; ++pk) {
                        const uint8_t* slot = static_cast<uint8_t*>(ec.base) +
                            uint64_t(h_tok_idx_[3u * pk + 1]) * ec.slot_bytes;
                        gemv_fp16(q, moe_hT_ + uint64_t(pk) * EF,
                                  reinterpret_cast<const sycl::half*>(slot + ec.down_off),
                                  moe_yT_ + uint64_t(h_tok_idx_[3u * pk + 2]) * H,
                                  EF, H);
                    }
                } else if (no_g2) {
                    for (uint32_t pk = 0; pk < P; ++pk) {
                        const uint8_t* slot = static_cast<uint8_t*>(ec.base) +
                            uint64_t(h_tok_idx_[3u * pk + 1]) * ec.slot_bytes;
                        gemv_q5_1(q, moe_hT_ + uint64_t(pk) * EF,
                                  slot + ec.down_off,
                                  moe_yT_ + uint64_t(h_tok_idx_[3u * pk + 2]) * H,
                                  EF, H);
                    }
                } else {
                    gemv_q5_1_grouped(q, moe_hT_,
                        static_cast<const uint8_t*>(ec.base), ec.slot_bytes,
                        ec.down_off, moe_tok_idx_, moe_yT_, H, EF, H, P);
                }
                if (tp_sync_) tp_sync_(L);
                // Ordered reduce: each row's top-k axpy sequence in k order —
                // the decode body's exact fp32 add chain, one launch.
                if (no_red) {
                    q.memset(moe_accT_, 0, uint64_t(T) * H * sizeof(float));
                    for (uint32_t vt = 0; vt < T; ++vt)
                        for (uint32_t k = 0; k < uint32_t(tk[vt].size()); ++k)
                            axpy_f16_to_f32(q, moe_accT_ + uint64_t(vt) * H,
                                            moe_yT_ + (uint64_t(vt) * NU + k) * H,
                                            tk[vt][k].second, H);
                } else {
                    const float* wtab = moe_tok_w_;
                    const int32_t* cnt = moe_tok_idx_ + 3u * 16 * NU;
                    const sycl::half* ys = moe_yT_;
                    float* acc = moe_accT_;
                    const uint32_t NU_ = NU, H_ = H;
                    q.parallel_for(sycl::range<1>(uint64_t(T) * H), [=](sycl::id<1> i) {
                        const uint32_t vt = uint32_t(i / H_);
                        const uint32_t hh = uint32_t(i % H_);
                        float a = 0.f;
                        const int32_t nk = cnt[vt];
                        for (int32_t k = 0; k < nk; ++k)
                            a += wtab[vt * NU_ + uint32_t(k)] *
                                 float(ys[(uint64_t(vt) * NU_ + uint32_t(k)) * H_ + hh]);
                        acc[i] = a;
                    });
                }
                }
            }
            t_moe_submit += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - tms0).count();
            for (uint32_t vt = 0; vt < T; ++vt)
                sigmoid_scalar_axpy(q, moe_accT_ + uint64_t(vt) * H,
                                    xg16_ + uint64_t(vt) * H, sgateT_ + vt, H);
            cast_f32_to_f16(q, moe_accT_, blockout_, uint64_t(T) * H);
        } else {
            // ---- prefill: EXPERT-MAJOR batching — each hit expert's slices
            // are read once per chunk and applied to all its tokens with the
            // batched kernels (gemv_q4_K_q8_batched / gemm_q5_1 / gemm_fp16).
            // Router logits stay FP32 end-to-end here: gemm_fp16's C is fp32
            // already, and routing from f16-rounded logits flips ~0.8% of
            // top-10 picks at near-ties (T=2600 parity: 20/2600 rows off by
            // one expert). D2H the fp32 C directly.
            gemm_fp16(q, mixed_, w.ffn_gate_inp, gemm_c_, T, cfg_.n_experts, H);
            std::vector<float> rall(uint64_t(T) * cfg_.n_experts);
            q.memcpy(rall.data(), gemm_c_, rall.size() * sizeof(float)).wait();
            // Route all T tokens; the softmax+top-k loop is host CPU time
            // (~5 us/token x T x 48 layers), so it runs parallel-for over
            // tokens into per-token slots, then buckets by expert serially
            // (bucketing must stay ordered for determinism).
            std::vector<std::vector<std::pair<int32_t, float>>> by_e(cfg_.n_experts);
            std::vector<std::vector<std::pair<uint32_t, float>>> tk(T);
            {
                const uint32_t nw =
                    (T >= 64 && !std::getenv("IE_Q4E_ROUTE_SERIAL")) ? 8u : 1u;
                std::vector<std::future<void>> fs;
                for (uint32_t w2 = 0; w2 < nw; ++w2) {
                    const uint32_t t0 = T * w2 / nw, t1 = T * (w2 + 1) / nw;
                    fs.push_back(std::async(std::launch::async, [&, t0, t1]() {
                        for (uint32_t t = t0; t < t1; ++t)
                            route_from_logits(rall.data() + uint64_t(t) * cfg_.n_experts,
                                              cfg_.n_experts,
                                              cfg_.n_experts_used, tk[t]);
                    }));
                }
                for (auto& f : fs) f.get();
            }
            for (uint32_t t = 0; t < T; ++t)
                for (const auto& [e, wgt] : tk[t]) by_e[e].push_back({int32_t(t), wgt});
            const bool q8p = w.gate_bv.raw != nullptr;
            q.memset(moe_accT_, 0, uint64_t(T) * H * sizeof(float));
            // One host->device shot for all token index/weight lists.
            struct EJob { uint32_t e, off, n; uint8_t* slot; };
            std::vector<EJob> jobs2;
            uint32_t off = 0;
            for (uint32_t e = 0; e < cfg_.n_experts; ++e) {
                if (by_e[e].empty()) continue;
                jobs2.push_back({e, off, uint32_t(by_e[e].size()), nullptr});
                off += uint32_t(by_e[e].size());
            }
            off = 0;
            for (auto& jb : jobs2) {
                for (uint32_t k2 = 0; k2 < jb.n; ++k2) {
                    h_tok_idx_[off + k2] = by_e[jb.e][k2].first;
                    h_tok_w_[off + k2]   = by_e[jb.e][k2].second;
                }
                jb.off = off;
                off += jb.n;
            }
            q.memcpy(moe_tok_idx_, h_tok_idx_, off * sizeof(int32_t));
            q.memcpy(moe_tok_w_,   h_tok_w_,   off * sizeof(float));
            // NOTE: slot resolution must interleave with kernel launches so
            // eviction never invalidates a slot a pending launch reads: the
            // per-expert path resolves+launches job by job; the grouped path
            // resolves+launches WAVES of <= ecache_slots_ jobs (below).
            const uint32_t q8row = (H / 32) * uint32_t(sizeof(block_q8_1x));
            static const bool grouped_pre =
                std::getenv("IE_Q4E_NO_PREFILL_GROUPED") == nullptr;
            // PER-PIECE grouped dispatch (2026-08-27): gate/up and down pick
            // their kernels independently — raw banks take the int-dot/dpas
            // tiles, F16-dequantized banks (the UD file's 2 Q5_K gate/up +
            // 5 Q8_0 down banks) take gemm_f16_grouped_tiles over the same
            // job table. Before this, ANY f16 piece sent the whole layer to
            // the per-expert path (~7K submits + 210 ms busy per 1024-chunk).
            const bool gate_raw = q8p;
            const bool down_raw = w.down_bv.raw != nullptr;
            const bool f16tiles_ok = (H % 128 == 0) && (EF % 128 == 0);
            if (grouped_pre && (gate_raw || f16tiles_ok) &&
                (down_raw || f16tiles_ok)) {
                // GROUPED prefill MoE (2026-08-27): ONE launch per stage per
                // WAVE over a tile-job table — the per-expert chunk launches
                // under-occupied the device (10-40 WGs each on the in-order
                // queue; the XMX A/B lost 35% to the same effect). Kernel
                // math is the chunked path's verbatim; the ordered per-token
                // reduce replays the e-ascending scatter order -> outputs
                // bit-identical to the per-expert path.
                //
                // WAVE-FLUSH (2026-08-27): resolving the WHOLE job table
                // before one launch is wrong once the chunk's expert union
                // exceeds the slot count — later resolves evict slots earlier
                // jobs point into (chunked-PPL corruption 16.79 vs 6.29 at
                // 512-chunks on 113 slots). Segmenting the e-ascending table
                // into waves of <= ecache_slots_ jobs is eviction-safe: each
                // in-wave resolve marks its slot MRU, so an in-wave miss can
                // only evict a slot OUTSIDE the wave, and wave i+1's refills
                // queue (in-order) after wave i's kernels. H2D traffic equals
                // the per-expert path's (same experts, same LRU, same order).
                {
                    std::vector<uint32_t> nk(T, 0);
                    uint32_t c2 = 0;
                    for (uint32_t e = 0; e < cfg_.n_experts; ++e)
                        for (const auto& pr : by_e[e]) {
                            const uint32_t t2 = uint32_t(pr.first);
                            h_cells_[uint64_t(t2) * cfg_.n_experts_used + nk[t2]] =
                                int32_t(c2);
                            ++nk[t2]; ++c2;
                        }
                    for (uint32_t t2 = 0; t2 < T; ++t2)
                        h_cells_[uint64_t(T) * cfg_.n_experts_used + t2] =
                            int32_t(nk[t2]);
                }
                q.memcpy(moe_cells_, h_cells_.data(),
                         (uint64_t(T) * cfg_.n_experts_used + T) * sizeof(int32_t));
                static const bool xmx_grp =
                    std::getenv("IE_Q4E_PREFILL_XMX") != nullptr;
                static const bool no_dpas =
                    std::getenv("IE_Q4E_NO_DPAS") != nullptr;
                static const bool no_dpas5 =
                    std::getenv("IE_Q4E_NO_DPAS") != nullptr ||
                    std::getenv("IE_Q4E_NO_DPAS_DOWN") != nullptr;
                const bool use_xmx = gate_raw && xmx_grp &&
                                     EF % 64 == 0 && H % 256 == 0;
                if (use_xmx || !gate_raw) {
                    // F16 activation rows: the env'd q4k-xmx path and the
                    // f16-bank gate/up piece both consume xg16_.
                    gather_bytes_rows(q, reinterpret_cast<const uint8_t*>(mixed_),
                                      reinterpret_cast<uint8_t*>(xg16_),
                                      moe_tok_idx_, off, H * 2);
                } else {
                    quantize_q8_1s(q, mixed_, act_q8sT_, T * H);
                    const uint32_t q8srow = (H / 32) * uint32_t(sizeof(block_q8_1s));
                    gather_bytes_rows(q, static_cast<const uint8_t*>(act_q8sT_),
                                      static_cast<uint8_t*>(xg_q8s_), moe_tok_idx_,
                                      off, q8srow);
                }
                // COPY-QUEUE PREFETCH (2026-08-27): expert refills land on
                // copyq_ so wave i+1's H2D overlaps wave i's kernels (probe:
                // uploads run ~161 us/miss ~ PCIe-serial on the critical
                // path — 25% of a 4096-chunk's wall even after big-chunk).
                // Safety: waves shrink to HALF the cache, so wave i+1's LRU
                // evictions can only take slots last touched in waves <= i-1
                // ... concretely: waves i and i-1 together fill the cache and
                // hold the MRU stamps, so wave i+1 evicts only slots from
                // waves <= i-2 (or earlier chunks) — its refills gate on wave
                // i-1's kernel-done event. Waves 0/1 gate on a q-fence
                // covering every prior q reader (previous chunks, decode,
                // verify). Each wave's kernels barrier on the wave's own
                // refills, so q stays ordered after cq and later q-side slot
                // users can never race a pending copy. IE_Q4E_NO_COPYQ=1
                // restores same-queue uploads and full-cache waves.
                static const bool no_copyq =
                    std::getenv("IE_Q4E_NO_COPYQ") != nullptr;
                const bool prefetch = !no_copyq && copyq_ != nullptr;
                const uint32_t wave_max = prefetch
                    ? std::max(1u, ecache_slots_ / 2) : ecache_slots_;
                sycl::queue& cq = prefetch ? *copyq_ : q;
                sycl::event qfence, kdone[2];
                if (prefetch) qfence = q.ext_oneapi_submit_barrier();
                uint32_t J = 0, wi = 0;
                for (size_t wj0 = 0; wj0 < jobs2.size(); ) {
                    const size_t wj1 =
                        std::min(jobs2.size(), wj0 + size_t(wave_max));
                    const uint32_t J0 = J;
                    const uint32_t row0 = jobs2[wj0].off;
                    const uint32_t row1 = jobs2[wj1 - 1].off + jobs2[wj1 - 1].n;
                    if (prefetch)
                        cq.ext_oneapi_submit_barrier(
                            {wi < 2 ? qfence : kdone[wi & 1]});
                    for (size_t ji = wj0; ji < wj1; ++ji) {
                        EJob& jb = jobs2[ji];
                        jb.slot = ensure_slot_on(cq, jb.e);
                        const int32_t si = int32_t(
                            (jb.slot - static_cast<uint8_t*>(ec.base)) / ec.slot_bytes);
                        for (uint32_t j0 = 0; j0 < jb.n; j0 += 16) {
                            h_tiles_[4u * J + 0] = int32_t(jb.off + j0);
                            h_tiles_[4u * J + 1] = int32_t(std::min(16u, jb.n - j0));
                            h_tiles_[4u * J + 2] = si;
                            h_tiles_[4u * J + 3] = int32_t(jb.off + j0);
                            ++J;
                        }
                    }
                    // Disjoint host/device tile slices per wave: wave i+1's
                    // build never touches the slice wave i's async copy reads.
                    q.memcpy(moe_tiles_ + 4u * J0, h_tiles_.data() + 4u * J0,
                             4u * (J - J0) * sizeof(int32_t));
                    // Fence this wave's kernels behind its own cq refills.
                    if (prefetch)
                        q.ext_oneapi_submit_barrier(
                            {cq.ext_oneapi_submit_barrier()});
                    const int32_t* wt = moe_tiles_ + 4u * J0;
                    const uint32_t Jw = J - J0;
                    if (!gate_raw) {
                        // F16 gate/up banks: grouped XMX gemm straight off
                        // the slot slices (dual pass).
                        gemm_f16_grouped_tiles(q, xg16_,
                            static_cast<const uint8_t*>(ec.base), ec.slot_bytes,
                            ec.gate_off, ec.up_off, wt,
                            moe_gT_, moe_uT_, H, EF, Jw, 2);
                    } else if (use_xmx) {
                        gemm_q4_K_xmx_grouped(q, xg16_,
                            static_cast<const uint8_t*>(ec.base), ec.slot_bytes,
                            ec.gate_off, ec.up_off, wt,
                            moe_gT_, moe_uT_, H, EF, Jw);
                    } else if (no_dpas) {
                        gemv_q4_K_q8s_grouped_tiles(q, xg_q8s_,
                            static_cast<const uint8_t*>(ec.base), ec.slot_bytes,
                            ec.gate_off, ec.up_off, wt,
                            moe_gT_, moe_uT_, H, EF, Jw);
                    } else {
                        // int8-DPAS grouped MoE (2026-08-27, ie-q4k-bench):
                        // gate/up 2.3x, down 3.2x over the s-form/pick tiles
                        // (66 -> 151 and 37 -> 121 eff GB/s). dpas-form fp
                        // order; prefill numerics gated by chunked PPL +
                        // run2 bit-eq. IE_Q4E_NO_DPAS=1 restores the s-form.
                        gemv_q4_K_q8d_grouped_tiles(q, xg_q8s_,
                            static_cast<const uint8_t*>(ec.base), ec.slot_bytes,
                            ec.gate_off, ec.up_off, wt,
                            moe_gT_, moe_uT_, H, EF, Jw);
                    }
                    swiglu(q, moe_gT_ + uint64_t(row0) * EF,
                           moe_uT_ + uint64_t(row0) * EF,
                           moe_hT_ + uint64_t(row0) * EF,
                           uint64_t(row1 - row0) * EF);
                    sycl::event kev;   // wave's last slot-reader (kdone gate)
                    if (!down_raw) {
                        // F16 down bank: single-mat grouped XMX gemm.
                        kev = gemm_f16_grouped_tiles(q, moe_hT_,
                            static_cast<const uint8_t*>(ec.base), ec.slot_bytes,
                            ec.down_off, 0, wt, moe_yT_, nullptr, EF, H, Jw, 1);
                    } else if (no_dpas5) {
                        kev = gemv_q5_1_grouped_tiles(q, moe_hT_,
                            static_cast<const uint8_t*>(ec.base), ec.slot_bytes,
                            ec.down_off, wt, moe_yT_, EF, H, Jw);
                    } else {
                        q51_block_rowsums(q, moe_hT_ + uint64_t(row0) * EF,
                                          row1 - row0, EF,
                                          moe_rsum_ + uint64_t(row0) * (EF / 32));
                        kev = gemv_q5_1_f16d_grouped_tiles(q, moe_hT_,
                            static_cast<const uint8_t*>(ec.base), ec.slot_bytes,
                            ec.down_off, wt, moe_yT_, moe_rsum_, EF, H, Jw);
                    }
                    kdone[wi & 1] = kev;
                    ++wi;
                    wj0 = wj1;
                }
                {
                    const int32_t* cells = moe_cells_;
                    const int32_t* nk = moe_cells_ + uint64_t(T) * cfg_.n_experts_used;
                    const float* wtab = moe_tok_w_;
                    const sycl::half* ys = moe_yT_;
                    float* acc = moe_accT_;
                    const uint32_t NU_ = cfg_.n_experts_used, H_ = H;
                    q.parallel_for(sycl::range<1>(uint64_t(T) * H), [=](sycl::id<1> i) {
                        const uint32_t t2 = uint32_t(i / H_);
                        const uint32_t hh = uint32_t(i % H_);
                        float a = 0.f;
                        const int32_t n2 = nk[t2];
                        for (int32_t k = 0; k < n2; ++k) {
                            const uint32_t c = uint32_t(cells[uint64_t(t2) * NU_ + uint32_t(k)]);
                            a += wtab[c] * float(ys[uint64_t(c) * H_ + hh]);
                        }
                        acc[i] = a;
                    });
                }
            } else {
            if (q8p) quantize_q8_1(q, mixed_, act_q8T_, T * H);
            for (EJob& jb : jobs2) {
                jb.slot = ensure_slot(jb.e);
                const int32_t* idx = moe_tok_idx_ + jb.off;
                const float*   wv  = moe_tok_w_ + jb.off;
                const void* gp = jb.slot + ec.gate_off;
                const void* up = jb.slot + ec.up_off;
                const void* dp = jb.slot + ec.down_off;
                static const bool xmx_pre =
                    std::getenv("IE_Q4E_PREFILL_XMX") != nullptr;
                if (q8p && xmx_pre && EF % 64 == 0 && H % 256 == 0) {
                    // A/B (2026-08-27): dpas joint_matrix GEMM on the raw
                    // Q4_K slices, F16 activations — the qwen36 prefill
                    // pattern (M_TILE 16, weight dequant amortized in SLM).
                    // Different numerics than the int-dot default (f16 A vs
                    // q8): prefill-only, PPL/parity-gated before default-on.
                    gather_bytes_rows(q, reinterpret_cast<const uint8_t*>(mixed_),
                                      reinterpret_cast<uint8_t*>(xg16_), idx, jb.n, H * 2);
                    for (uint32_t j0 = 0; j0 < jb.n; j0 += 16) {
                        const uint32_t jn = std::min(16u, jb.n - j0);
                        const sycl::half* xa = xg16_ + uint64_t(j0) * H;
                        gemm_q4_K_xmx(q, xa, gp, moe_gT_ + uint64_t(j0) * EF,
                                      jn, H, EF);
                        gemm_q4_K_xmx(q, xa, up, moe_uT_ + uint64_t(j0) * EF,
                                      jn, H, EF);
                    }
                } else if (q8p) {
                    gather_bytes_rows(q, static_cast<const uint8_t*>(act_q8T_),
                                      static_cast<uint8_t*>(xg_q8_), idx, jb.n, q8row);
                    // gemv_q4_K_q8_batched has a HARD T_MAX=16 (acc[16]
                    // registers — gemv_q8dot.cpp:384; beyond it the kernel
                    // silently corrupts: caught by the T=2600 QSA parity gate
                    // 2026-08-27). Chunk the gathered rows at 16.
                    for (uint32_t j0 = 0; j0 < jb.n; j0 += 16) {
                        const uint32_t jn = std::min(16u, jb.n - j0);
                        const uint8_t* xo = static_cast<const uint8_t*>(xg_q8_) +
                                            uint64_t(j0) * q8row;
                        gemv_q4_K_q8_batched(q, xo, gp,
                                             moe_gT_ + uint64_t(j0) * EF, H, EF, jn);
                        gemv_q4_K_q8_batched(q, xo, up,
                                             moe_uT_ + uint64_t(j0) * EF, H, EF, jn);
                    }
                } else {
                    gather_bytes_rows(q, reinterpret_cast<const uint8_t*>(mixed_),
                                      reinterpret_cast<uint8_t*>(xg16_), idx, jb.n, H * 2);
                    gemm_fp16(q, xg16_, static_cast<const sycl::half*>(gp), gemm_c_, jb.n, EF, H);
                    cast_fp32_to_fp16(q, gemm_c_, moe_gT_, uint64_t(jb.n) * EF);
                    gemm_fp16(q, xg16_, static_cast<const sycl::half*>(up), gemm_c_, jb.n, EF, H);
                    cast_fp32_to_fp16(q, gemm_c_, moe_uT_, uint64_t(jb.n) * EF);
                }
                swiglu(q, moe_gT_, moe_uT_, moe_hT_, uint64_t(jb.n) * EF);
                if (w.down_bv.raw) {
                    if (std::getenv("IE_Q4E_DBG_Q51CHUNK")) {
                        for (uint32_t j0 = 0; j0 < jb.n; j0 += 16)
                            gemm_q5_1(q, moe_hT_ + uint64_t(j0) * EF, dp,
                                      moe_yT_ + uint64_t(j0) * H,
                                      std::min(16u, jb.n - j0), EF, H);
                    } else {
                        gemm_q5_1(q, moe_hT_, dp, moe_yT_, jb.n, EF, H);
                    }
                } else {
                    gemm_fp16(q, moe_hT_, static_cast<const sycl::half*>(dp), gemm_c_, jb.n, H, EF);
                    cast_fp32_to_fp16(q, gemm_c_, moe_yT_, uint64_t(jb.n) * H);
                }
                scatter_axpy_rows(q, moe_accT_, moe_yT_, idx, wv, jb.n, H);
            }
            }
            // Shared expert, batched (SEF == EF on the real file; the moe_*T_
            // buffers are sized EF so this holds — guarded).
            if (SEF <= EF) {
                if (w.shg_q8.qs) {
                    q8_proj(mixed_, w.shg_q8, moe_gT_, H, SEF);
                    q8_proj(mixed_, w.shu_q8, moe_uT_, H, SEF);
                    swiglu(q, moe_gT_, moe_uT_, moe_hT_, uint64_t(T) * SEF);
                    q8_proj(moe_hT_, w.shd_q8, moe_yT_, SEF, H);
                } else {
                    proj(mixed_, w.ffn_gate_shexp, moe_gT_, H, SEF);
                    proj(mixed_, w.ffn_up_shexp,   moe_uT_, H, SEF);
                    swiglu(q, moe_gT_, moe_uT_, moe_hT_, uint64_t(T) * SEF);
                    proj(moe_hT_, w.ffn_down_shexp, moe_yT_, SEF, H);
                }
                // N=1 stays on gemv_fp16: the decode-certified path (gemm at
                // N=1 is an untested tile edge; parity flagged the batched form).
                gemv_fp16_dotrows(q, mixed_, H, w.ffn_gate_inp_shexp,
                                  sgateT_, H, T);
                sigmoid_rows_axpy(q, moe_accT_, moe_yT_, sgateT_, T, H);
            } else {
                for (uint32_t t = 0; t < T; ++t) {
                    const sycl::half* x = mixed_ + uint64_t(t) * H;
                    if (w.shg_q8.qs && dense_a16()) {
                        gemv_q8_0_soa_f16_g(q, x, w.shg_q8.qs, w.shg_q8.d, sh_g_, H, SEF);
                        gemv_q8_0_soa_f16_g(q, x, w.shu_q8.qs, w.shu_q8.d, sh_u_, H, SEF);
                        swiglu(q, sh_g_, sh_u_, sh_h_, SEF);
                        gemv_q8_0_soa_f16_g(q, sh_h_, w.shd_q8.qs, w.shd_q8.d, sh_y_, SEF, H);
                    } else if (w.shg_q8.qs) {
                        quantize_q8_1(q, x, act_q8_, H);
                        gemv_q8_0_soa_q8_g(q, act_q8_, w.shg_q8.qs, w.shg_q8.d, sh_g_, H, SEF);
                        gemv_q8_0_soa_q8_g(q, act_q8_, w.shu_q8.qs, w.shu_q8.d, sh_u_, H, SEF);
                        swiglu(q, sh_g_, sh_u_, sh_h_, SEF);
                        quantize_q8_1(q, sh_h_, act_q8_, SEF);
                        gemv_q8_0_soa_q8_g(q, act_q8_, w.shd_q8.qs, w.shd_q8.d, sh_y_, SEF, H);
                    } else {
                        gemv_fp16(q, x, w.ffn_gate_shexp, sh_g_, H, SEF);
                        gemv_fp16(q, x, w.ffn_up_shexp,   sh_u_, H, SEF);
                        swiglu(q, sh_g_, sh_u_, sh_h_, SEF);
                        gemv_fp16(q, sh_h_, w.ffn_down_shexp, sh_y_, SEF, H);
                    }
                    gemv_fp16(q, x, w.ffn_gate_inp_shexp, sh_s_, H, 1);
                    sigmoid_scalar_axpy(q, moe_accT_ + uint64_t(t) * H, sh_y_, sh_s_, H);
                }
            }
            cast_f32_to_f16(q, moe_accT_, blockout_, uint64_t(T) * H);
        }
        if (cap) grab(cap->moe_out, blockout_);
        {   // Diagnostic: IE_Q4E_DUMP_BLK=<file> appends blockout_ [T,H] f16 per stage
            static const char* dump_blk = std::getenv("IE_Q4E_DUMP_BLK");
            if (dump_blk) {
                q.wait();
                std::vector<sycl::half> hb(uint64_t(T) * H);
                q.memcpy(hb.data(), blockout_, hb.size() * 2).wait();
                if (FILE* f = std::fopen(dump_blk, "ab")) {
                    std::fwrite(hb.data(), 2, hb.size(), f); std::fclose(f);
                }
            }
        }
        qwen4_hc_combine(q, wide_, blockout_, inj_, T, H, cfg_.hc_count);
}

std::string Qwen4ExpModel::run_block_parity(uint32_t L, const float* wide_in_host,
                                             uint32_t T, const int32_t* tokens_host,
                                             float* wide_out_host, ParityCapture* cap) {
    if (T == 0 || T > max_chunk_) return "qwen4exp parity: bad T";
    if (L >= cfg_.n_layers) return "qwen4exp parity: bad layer";
    if (cfg_.is_ple_layer(L) && !tokens_host)
        return "qwen4exp parity: PLE block needs token ids";
    sycl::queue& q = alloc_->queue();
    const uint64_t n = uint64_t(T) * cfg_.hc_count * cfg_.hidden;
    q.memcpy(wide_, wide_in_host, n * sizeof(float));
    {   // positions 0..T-1 (fresh-cache oracle convention)
        std::vector<int32_t> pos(T);
        for (uint32_t t = 0; t < T; ++t) pos[t] = int32_t(t);
        q.memcpy(d_pos_, pos.data(), T * sizeof(int32_t)).wait();
    }
    run_block(L, T, /*start_pos=*/0, tokens_host, cap);
    if (!block_err_.empty()) {
        std::string e; e.swap(block_err_);
        return e;
    }
    q.wait();
    q.memcpy(wide_out_host, wide_, n * sizeof(float)).wait();
    return {};
}

std::string Qwen4ExpModel::forward(const int32_t* tokens_host, uint32_t T,
                                   uint32_t start_pos, sycl::half* logits_out) {
    if (T == 0 || T > max_chunk_) return "qwen4exp forward: bad T";
    if (start_pos + T > max_ctx_)
        return "qwen4exp forward: ctx " + std::to_string(start_pos + T) +
               " > max_ctx " + std::to_string(max_ctx_);
    sycl::queue& q = alloc_->queue();
    const uint32_t H = cfg_.hidden, D = cfg_.hc_count * H;
    const uint32_t SI = cfg_.ssm_inner, SKH = cfg_.ssm_k_heads, SVH = cfg_.ssm_v_heads;
    const uint32_t SHD = cfg_.ssm_state, CC = SI + 2u * SKH * SHD, KW = SKH * SHD;
    const uint32_t HD = cfg_.head_dim, NQ = cfg_.n_q_heads * HD, NKV = cfg_.n_kv_heads * HD;
    const uint32_t EF = cfg_.expert_ffn, SEF = cfg_.shared_expert_ffn;
    const float eps = cfg_.rms_eps;
    const float qscale = 1.0f / std::sqrt(float(SHD));
    const uint64_t per_layer_kv = uint64_t(cfg_.n_kv_heads) * max_ctx_ * HD;

    return forward_range(tokens_host, T, start_pos, nullptr, nullptr, logits_out);
}

std::string Qwen4ExpModel::forward_range(const int32_t* tokens_host, uint32_t T,
                                         uint32_t start_pos,
                                         const float* wide_in_host,
                                         float* wide_out_host,
                                         sycl::half* logits_out,
                                         bool wide_in_device) {
    if (T == 0 || T > max_chunk_) return "qwen4exp forward: bad T";
    // Refuse past max_ctx HERE, not only in forward(): run2 drove this entry
    // straight past a 2051-slot KV into unmapped VAs — the 2026-08-27 03:08
    // xe pagefault flood ("Fault response: Unsuccessful -ENOENT").
    if (start_pos + T > max_ctx_)
        return "qwen4exp forward_range: ctx " + std::to_string(start_pos + T) +
               " > max_ctx " + std::to_string(max_ctx_);
    sycl::queue& q = alloc_->queue();
    const uint32_t H = cfg_.hidden;
    const float eps = cfg_.rms_eps;
    if (tokens_host) {
        q.memcpy(d_tokens_, tokens_host, T * sizeof(int32_t));
    }
    {
        std::vector<int32_t> pos(T);
        for (uint32_t t = 0; t < T; ++t) pos[t] = int32_t(start_pos + t);
        q.memcpy(d_pos_, pos.data(), T * sizeof(int32_t)).wait();
    }
    if (mrope_n_) {
        // Vision M-RoPE streams for this chunk ([3, T] stream-major). Prompt
        // positions come from the staged table; decode continuation ropes at
        // (linear + delta) on all streams (14_mrope.md §4.2). d_pos_ above
        // stays linear — the QSA mask/topk consumers require token indices.
        std::vector<int32_t> p3(size_t(3) * T);
        for (uint32_t s = 0; s < 3; ++s)
            for (uint32_t t = 0; t < T; ++t) {
                const uint32_t ap = start_pos + t;
                p3[size_t(s) * T + t] = ap < mrope_n_
                    ? mrope3_[size_t(s) * mrope_n_ + ap]
                    : int32_t(ap) + mrope_delta_;
            }
        q.memcpy(d_pos3_, p3.data(), p3.size() * sizeof(int32_t)).wait();
    }
    if (wide_in_device) {
        // wide_ already holds this range's input — the previous stage
        // PUSHED it over P2P (see set_wide_peer). Nothing to stage.
    } else if (wide_in_host) {
        q.memcpy(wide_, wide_in_host,
                 uint64_t(T) * cfg_.hc_count * H * sizeof(float)).wait();
    } else {
        if (layer_lo_ != 0 || !token_embd) return "qwen4exp: head stage needs layer 0";
        gather_embed_rows(q, token_embd, d_tokens_, emb_, T, H);
        for (const VisSpan& vs : vis_spans_) {
            // Vision splice: staged ViT rows overwrite the gathered embed rows
            // where this chunk overlaps [t0, t0+n). In-order queue ⇒ ordered
            // after the gather, before the stream expand.
            const uint32_t lo = std::max(start_pos, vs.t0);
            const uint32_t hi = std::min(start_pos + T, vs.t0 + vs.n);
            if (lo < hi)
                q.memcpy(emb_ + uint64_t(lo - start_pos) * H,
                         vis_rows_.data() +
                             (uint64_t(vs.row0) + (lo - vs.t0)) * H,
                         uint64_t(hi - lo) * H * sizeof(sycl::half));
        }
        qwen4_hc_expand_streams(q, emb_, wide_, T, H, cfg_.hc_count);
    }

    if (spec_verify_ && T >= 1 && T <= 16) {
        // A verify forward that EXTENDS the previous one (start == pos + T)
        // appends its per-position checkpoints after the existing slabs —
        // T=1 (overlapped with the draft) + T=K chunk-splits are bit-exact
        // (every sv op is per-row or streaming), and commit_verify() then
        // sees one uniform [vck_T_] slab range.
        const bool extends = vck_T_ > 0 && start_pos == vck_pos_ + vck_T_ &&
                             vck_T_ + T <= 16;
        const uint32_t need = extends ? vck_T_ + T : T;
        if (need > 8) return "qwen4exp: verify T beyond checkpoint cap 8";
        if (vck_cap_ < 8) {
            // One fixed 8-slab allocation (covers K<=7): a grow-realloc
            // would strand the append case's earlier rows.
            constexpr uint32_t CAP = 8;
            const uint64_t se_all = dn_.state_elems_per_layer() *
                                    dn_.config().n_layers_linear;
            const uint64_t ce_all = dn_.conv_elems_per_layer() *
                                    dn_.config().n_layers_linear;
            vck_dn_state_ = static_cast<float*>(
                alloc_->malloc(uint64_t(CAP) * se_all * sizeof(float)));
            vck_dn_conv_ = static_cast<sycl::half*>(
                alloc_->malloc(uint64_t(CAP) * ce_all * sizeof(sycl::half)));
            if (!vck_dn_state_ || !vck_dn_conv_)
                return "qwen4exp: verify checkpoint alloc failed";
            owned_.push_back(vck_dn_state_);
            owned_.push_back(vck_dn_conv_);
            if (ple_conv_state_) {
                vck_ple_conv_ = static_cast<float*>(alloc_->malloc(
                    uint64_t(CAP) * kPleStateRows * kPleSI * sizeof(float)));
                if (!vck_ple_conv_) return "qwen4exp: verify ple ckpt alloc failed";
                owned_.push_back(vck_ple_conv_);
            }
            vck_cap_ = CAP;
        }
        if (extends) {
            vck_base_ = vck_T_;
            vck_T_ += T;
        } else {
            vck_base_ = 0;
            vck_T_ = T;
            vck_pos_ = start_pos;
        }
    }

    for (uint32_t L = layer_lo_; L < layer_hi_; ++L) {
        run_block(L, T, start_pos, tokens_host, nullptr);
        if (!block_err_.empty()) {
            std::string e; e.swap(block_err_);
            return e;
        }
        // Diagnostic: IE_Q4E_DUMP_WIDE=<file> appends the post-block wide
        // state [T, hc, H] f32 per layer (cross-process first-divergence hunt).
        static const char* dump_wide = std::getenv("IE_Q4E_DUMP_WIDE");
        if (dump_wide) {
            q.wait();
            std::vector<float> hw(uint64_t(T) * cfg_.hc_count * H);
            q.memcpy(hw.data(), wide_, hw.size() * sizeof(float)).wait();
            if (FILE* f = std::fopen(dump_wide, "ab")) {
                std::fwrite(hw.data(), sizeof(float), hw.size(), f);
                std::fclose(f);
            }
        }
    }

    if (wide_out_host) {
        q.wait();
        if (wide_peer_) {
            // P2P PUSH from this device's queue into the peer stage's
            // wide_ — replaces the D2H+H2D host bounce. The host-side
            // wait keeps the cross-queue ordering the bounce provided.
            q.memcpy(wide_peer_, wide_,
                     uint64_t(T) * cfg_.hc_count * H * sizeof(float)).wait();
            return {};
        }
        q.memcpy(wide_out_host, wide_,
                 uint64_t(T) * cfg_.hc_count * H * sizeof(float)).wait();
        return {};
    }
    if (layer_hi_ != cfg_.n_layers || (!lm_head && !lmh_q8.qs))
        return "qwen4exp: tail stage needs the last layer";
    // Final HC merge (the model's output "norm") + lm_head on the last token.
    if (T > 16 && hc_prefill_gemm() && out_hc_down_t)
        qwen4_hc_mix_prefill(q, wide_, out_hc_norm, out_hc_down_t, out_hc_up_t,
                 nullptr, xn_ws_, hc_x16_, hc_lo16_, gemm_c_, mixed_, nullptr,
                 T, H, cfg_.hc_count, cfg_.hc_low_rank, eps);
    else
        qwen4_hc_mix_v2(q, wide_, out_hc_norm, out_hc_down, out_hc_up, nullptr,
                 xn_ws_, lo_ws_, mixed_, nullptr, T, H, cfg_.hc_count,
                 cfg_.hc_low_rank, eps);
    static const bool use_q8_lmh = qwen4exp_dense_q8();
    if (use_q8_lmh && lmh_q8.qs && dense_a16()) {
        gemv_q8_0_soa_f16_g(q, mixed_ + uint64_t(T - 1) * H, lmh_q8.qs,
                            lmh_q8.d, logits_out ? logits_out : logits_,
                            H, cfg_.vocab);
    } else if (use_q8_lmh && lmh_q8.qs) {
        quantize_q8_1(q, mixed_ + uint64_t(T - 1) * H, act_q8_, H);
        gemv_q8_0_soa_q8_g(q, act_q8_, lmh_q8.qs, lmh_q8.d,
                           logits_out ? logits_out : logits_, H, cfg_.vocab);
    } else {
        gemv_fp16(q, mixed_ + uint64_t(T - 1) * H, lm_head,
                  logits_out ? logits_out : logits_, H, cfg_.vocab);
    }
    q.wait();
    return {};
}

}  // namespace ie
