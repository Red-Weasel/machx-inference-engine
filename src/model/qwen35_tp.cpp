// src/model/qwen35_tp.cpp — Qwen3.6-27B (`kQwen35Dense`) TENSOR-PARALLEL decode.
// See header. ADDITIVE glue over Qwen35SplitModel's per-layer hybrid math (lifted
// verbatim) + DenseModelTP's all-reduce/overlap. The single-GPU/split models and
// the crown are NEVER edited.
//
// PHASE 0: FFN-slice TP only. attn + DeltaNet run REPLICATED (full weights) on
// every card from the replicated x → bit-identical residual; only the dense FFN is
// sharded (gate/up column, down row) with ONE all-reduce/layer. This isolates the
// all-reduce cost on this no-P2P board as the GO/NO-GO gate.

#include "ie/qwen35_tp.hpp"

#include "ie/dequant.hpp"
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

// COPY of qwen35_split.cpp:dequant_q6_K_row (copy-not-hoist discipline).
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

// COPY of qwen35_split.cpp:upload_f32_proj_fp16 (copy-not-hoist).
DenseQuantPtr upload_f32_proj_fp16(DeviceAllocator& alloc, const GgufTensorInfo* t,
                                   std::vector<void*>& owned, std::string& err,
                                   uint32_t Npad) {
    DenseQuantPtr out;
    if (!t) { err = "tensor not found"; return out; }
    if (t->n_dims != 2 || (t->dtype != DType::kF32 && t->dtype != DType::kQ8_0 &&
                           t->dtype != DType::kQ6_K)) {
        err = "ssm proj: expected F32, Q8_0, or Q6_K 2-D"; return out;
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
    } else if (t->dtype == DType::kQ8_0) {
        const uint64_t bpr = K / 32;
        const auto* blocks = reinterpret_cast<const block_q8_0*>(t->data);
        for (uint64_t n = 0; n < N; ++n)
            for (uint64_t k = 0; k < K; ++k) {
                const block_q8_0& b = blocks[n * bpr + (k >> 5)];
                staging[k * Npad + n] = sycl::half(fp16_to_fp32(b.d) * float(b.qs[k & 31]));
            }
    } else {
        if (K % kQK_K != 0) { err = "ssm proj Q6_K: K not a multiple of 256"; return out; }
        const uint64_t bpr = K / kQK_K;
        const auto* blocks = reinterpret_cast<const block_q6_K*>(t->data);
        std::vector<float> row(K);
        for (uint64_t n = 0; n < N; ++n) {
            dequant_q6_K_row(blocks + n * bpr, row.data(), K);
            for (uint64_t k = 0; k < K; ++k)
                staging[k * Npad + n] = sycl::half(row[k]);
        }
    }
    void* d = alloc.malloc(K * Npad * sizeof(sycl::half));
    if (!d) { err = "malloc failed (ssm proj)"; return out; }
    alloc.queue().memcpy(d, staging.data(), K * Npad * sizeof(sycl::half)).wait();
    owned.push_back(d);
    out.p = d; out.dt = DType::kF16;
    return out;
}

// COPY of qwen35_split.cpp:extract_cols (copy-not-hoist).
inline sycl::event extract_cols(sycl::queue& q, const sycl::half* src, sycl::half* dst,
                                uint32_t T, uint32_t nh, uint32_t src_stride) {
    return q.parallel_for(sycl::range<1>(uint64_t(T) * nh), [=](sycl::id<1> i) {
        const uint32_t t = uint32_t(i) / nh, h = uint32_t(i) % nh;
        dst[uint64_t(t) * nh + h] = src[uint64_t(t) * src_stride + h];
    });
}

// COPY of qwen35_split.cpp:dequant_q8_0_soa_to_Bt (copy-not-hoist).
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

Qwen35TpModel::~Qwen35TpModel() { free_all(); }

void Qwen35TpModel::free_all() {
    if (!fleet_) return;
    free_spec_replay();
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

void Qwen35TpModel::free_spec_replay() noexcept {
    if (!fleet_) return;
    for (uint32_t c = 0; c < spec_replay_.size(); ++c) {
        auto& a = fleet_->dev(c);
        SpecReplay& r = spec_replay_[c];
        for (void* p : {static_cast<void*>(r.pre_state),
                        static_cast<void*>(r.pre_conv),
                        static_cast<void*>(r.qkv),
                        static_cast<void*>(r.qrep),
                        static_cast<void*>(r.krep),
                        static_cast<void*>(r.vpre),
                        static_cast<void*>(r.g),
                        static_cast<void*>(r.beta)})
            if (p) a.free(p);
    }
    spec_replay_.clear();
}

std::string Qwen35TpModel::ensure_spec_replay(uint32_t K) {
    if (K == 0 || K > 16) return "spec replay K out of range";
    if (spec_replay_.size() == n_dev_) {
        bool ready = true;
        for (const auto& r : spec_replay_)
            ready = ready && r.K >= K && r.pre_state && r.pre_conv && r.qkv &&
                    r.qrep && r.krep && r.vpre && r.g && r.beta;
        if (ready) return {};
    }
    free_spec_replay();
    spec_replay_.resize(n_dev_);

    const uint32_t conv_ch = cfg_.ssm_inner +
        2u * cfg_.ssm_n_k_heads * cfg_.ssm_state;
    const uint32_t Vd = cfg_.ssm_n_v_heads * cfg_.ssm_state;
    const uint32_t Nv = cfg_.ssm_n_v_heads;
    for (uint32_t c = 0; c < n_dev_; ++c) {
        auto& a = fleet_->dev(c);
        SpecReplay& r = spec_replay_[c];
        r.K = K;
        r.n_lin = dn_[c].config().n_layers_linear;
        const uint64_t state_n = uint64_t(r.n_lin) * dn_[c].state_elems_per_layer();
        const uint64_t conv_n = uint64_t(r.n_lin) * dn_[c].conv_elems_per_layer();
        const uint64_t qkv_n = uint64_t(r.n_lin) * K * conv_ch;
        const uint64_t vec_n = uint64_t(r.n_lin) * K * Vd;
        const uint64_t scalar_n = uint64_t(r.n_lin) * K * Nv;
        r.pre_state = static_cast<float*>(a.malloc(state_n * sizeof(float)));
        r.pre_conv = static_cast<sycl::half*>(a.malloc(conv_n * sizeof(sycl::half)));
        r.qkv = static_cast<sycl::half*>(a.malloc(qkv_n * sizeof(sycl::half)));
        r.qrep = static_cast<float*>(a.malloc(vec_n * sizeof(float)));
        r.krep = static_cast<float*>(a.malloc(vec_n * sizeof(float)));
        r.vpre = static_cast<float*>(a.malloc(vec_n * sizeof(float)));
        r.g = static_cast<float*>(a.malloc(scalar_n * sizeof(float)));
        r.beta = static_cast<float*>(a.malloc(scalar_n * sizeof(float)));
        if (!r.pre_state || !r.pre_conv || !r.qkv || !r.qrep || !r.krep ||
            !r.vpre || !r.g || !r.beta) {
            free_spec_replay();
            return "spec replay alloc failed on dev " + std::to_string(c);
        }
    }
    return {};
}

std::string Qwen35TpModel::snapshot_spec_state() {
    if (spec_replay_.size() != n_dev_) return "spec replay not initialized";
    for (uint32_t c = 0; c < n_dev_; ++c) {
        const SpecReplay& r = spec_replay_[c];
        if (!dn_[c].ready() || r.n_lin != dn_[c].config().n_layers_linear)
            return "spec replay geometry mismatch on dev " + std::to_string(c);
        auto& q = fleet_->dev(c).queue();
        q.memcpy(r.pre_state, dn_[c].state_ptr(),
                 uint64_t(r.n_lin) * dn_[c].state_elems_per_layer() * sizeof(float));
        q.memcpy(r.pre_conv, dn_[c].conv_state_ptr(),
                 uint64_t(r.n_lin) * dn_[c].conv_elems_per_layer() *
                     sizeof(sycl::half));
    }
    for (uint32_t c = 0; c < n_dev_; ++c) fleet_->dev(c).queue().wait();
    return {};
}

std::string Qwen35TpModel::replay_spec_deltanet(uint32_t accepted) {
    if (spec_replay_.size() != n_dev_) return "spec replay not initialized";
    const uint32_t conv_ch = cfg_.ssm_inner +
        2u * cfg_.ssm_n_k_heads * cfg_.ssm_state;
    const uint32_t SVH = cfg_.ssm_n_v_heads;
    const uint32_t SHD = cfg_.ssm_state;
    const uint32_t Vd = SVH * SHD;

    for (uint32_t c = 0; c < n_dev_; ++c) {
        SpecReplay& r = spec_replay_[c];
        if (accepted == 0 || accepted >= r.K)
            return "spec replay accepted out of range";
        auto& q = fleet_->dev(c).queue();
        q.memcpy(dn_[c].state_ptr(), r.pre_state,
                 uint64_t(r.n_lin) * dn_[c].state_elems_per_layer() * sizeof(float));
        q.memcpy(dn_[c].conv_state_ptr(), r.pre_conv,
                 uint64_t(r.n_lin) * dn_[c].conv_elems_per_layer() *
                     sizeof(sycl::half));

        uint32_t dn_local = 0;
        for (uint32_t L = 0; L < cfg_.n_transformer_layers(); ++L) {
            const LayerW& wl = dlayers_[c][L];
            if (!wl.is_linear) continue;
            Workspace& w = ws_[c];
            const uint64_t row = uint64_t(dn_local) * r.K;
            sycl::half* conv_state = dn_[c].conv_state_ptr() +
                uint64_t(dn_local) * dn_[c].conv_elems_per_layer();
            depthwise_conv1d_causal(q, r.qkv + row * conv_ch,
                                    wl.ssm_conv1d_fp16, conv_state, w.dn_conv,
                                    accepted, conv_ch, cfg_.ssm_conv_kernel);
            float* state = dn_[c].state_ptr() +
                uint64_t(dn_local) * dn_[c].state_elems_per_layer();
            deltanet_recurrence(q, r.qrep + row * Vd, r.krep + row * Vd,
                                r.vpre + row * Vd, r.g + row * SVH,
                                r.beta + row * SVH, state, w.dn_out,
                                /*B=*/1, accepted, SVH, SHD, SHD);
            ++dn_local;
        }
    }
    for (uint32_t c = 0; c < n_dev_; ++c) fleet_->dev(c).queue().wait();
    return {};
}

std::string Qwen35TpModel::load(DeviceFleet& fleet, const GgufReader& g,
                                const Qwen35Config& cfg, uint32_t max_ctx) {
    fleet_ = &fleet; cfg_ = cfg;
    n_dev_ = fleet.size();
    const DenseConfig& d = cfg.dense;

    if (n_dev_ == 0) return "qwen35tp: empty fleet";
    if (d.hidden == 0 || d.vocab == 0 || d.n_q_heads == 0 || d.n_kv_heads == 0 ||
        d.ffn == 0)                                return "qwen35tp: zero dim";
    if (cfg.ssm_inner == 0 || cfg.ssm_n_v_heads == 0 || cfg.ssm_n_k_heads == 0 ||
        cfg.ssm_state == 0 || cfg.ssm_conv_kernel == 0)
        return "qwen35tp: missing ssm dims";
    if (d.ffn % 256) return "qwen35tp: ffn not 256-aligned";

    // oneDNN ON for TP prefill (2026-08-16). The old disable rationale was
    // MoE-era ("small per-expert M") — this dense path now feeds M=pf_chunk
    // (3072) gemms, exactly the shapes where oneDNN is ~1.65× the custom gemm
    // (same lesson as split, which is also multi-card and defaults ON). The
    // DEVICE_LOST landmine is gone (gemm_onednn.cpp:ctx_for keeps one engine
    // per device). Decode/verify never route here (int-dot T==1 / batched
    // T<=16). Same kill switch as split for byte-gate pinning.
    dense::prefer_onednn() = std::getenv("IE_QWEN35_SPLIT_NO_ONEDNN") == nullptr;

    owned_.assign(n_dev_, {});
    dev_bytes_.assign(n_dev_, 0);

    // FFN superblock split: card c owns a 256-aligned contiguous F range. The
    // gate/up column-split point == the ffn_down row-split point (both = f0/fc).
    shard_.assign(n_dev_, {});
    {
        const uint32_t nsb = d.ffn / 256;
        uint32_t f_acc = 0;
        for (uint32_t c = 0; c < n_dev_; ++c) {
            const uint32_t my_sb = nsb / n_dev_ + (c < (nsb % n_dev_) ? 1u : 0u);
            shard_[c].f0 = f_acc; shard_[c].fc = my_sb * 256; f_acc += shard_[c].fc;
        }
        if (f_acc != d.ffn) return "qwen35tp: ffn superblock split did not cover F";
    }

    char buf[64];
    auto Ttop = [&](const char* n) { return g.find_tensor(n); };
    auto Tl   = [&](uint32_t L, const char* n) {
        std::snprintf(buf, sizeof(buf), "blk.%u.%s", L, n); return g.find_tensor(buf);
    };
    std::string err;

    // build_split: replicate (full) a GGUF matrix weight onto card `c`. Q8_0 →
    // de-interleave AoS block_q8_0 → SoA (PACKED, bit-exact). Else → fp fallback.
    // COPY of qwen35_split.cpp:build_split.
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

    // build_col: COLUMN slice [n0,n0+nc) of a [N,K] weight onto card c. Q8_0 → take
    // output rows [n0,n0+nc) of the SoA (qs is [N*K] row=output → contiguous row
    // range; d is [N*(K/32)]). Else → re-pack a sliced AoS Q8_0... but the FFN of
    // the cyber model is Q8_0 so we only need the Q8_0 path here; non-Q8_0 errors.
    auto build_col = [](DeviceAllocator& a, const GgufTensorInfo* t,
                        std::vector<void*>& own, std::string& e, uint64_t& bytes,
                        uint32_t n0, uint32_t nc) -> SplitW {
        SplitW w{};
        if (!t) { e = "col tensor not found"; return w; }
        if (t->n_dims != 2) { e = "col weight: expected 2-D"; return w; }
        const uint32_t K = uint32_t(t->shape[0]), N = uint32_t(t->shape[1]);
        if (uint64_t(n0) + nc > N) { e = "col slice: bad geometry"; return w; }
        w.K = K; w.N = nc;
        if (t->dtype == DType::kQ8_0) {
            const uint32_t bpc = K / 32;
            if (K % 32 != 0) { e = "Q8_0 col SoA: K % 32 != 0"; return w; }
            const auto* blocks = reinterpret_cast<const block_q8_0*>(t->data);
            std::vector<int8_t>   qs(uint64_t(nc) * K);
            std::vector<uint16_t> dd(uint64_t(nc) * bpc);
            for (uint32_t j = 0; j < nc; ++j) {
                const uint32_t n = n0 + j;
                for (uint32_t b = 0; b < bpc; ++b) {
                    const block_q8_0& blk = blocks[uint64_t(n) * bpc + b];
                    dd[uint64_t(j) * bpc + b] = *reinterpret_cast<const uint16_t*>(&blk.d);
                    for (int i = 0; i < 32; ++i)
                        qs[uint64_t(j) * K + uint64_t(b) * 32 + i] = blk.qs[i];
                }
            }
            auto* dqs = static_cast<int8_t*>(a.malloc(qs.size()));
            auto* ddd = static_cast<uint16_t*>(a.malloc(dd.size() * sizeof(uint16_t)));
            if (!dqs || !ddd) { e = "Q8_0 col alloc"; return w; }
            own.push_back(dqs); own.push_back(ddd);
            a.queue().memcpy(dqs, qs.data(), qs.size()).wait();
            a.queue().memcpy(ddd, dd.data(), dd.size() * sizeof(uint16_t)).wait();
            w.q8_qs = dqs; w.q8_d = ddd;
            bytes += qs.size() + dd.size() * sizeof(uint16_t);
        } else {
            e = std::string("col slice: dtype ") + std::string(type_name(t->dtype)) +
                " unsupported (Phase-0 FFN shard needs Q8_0)";
        }
        return w;
    };

    // build_row: ROW slice [k0,k0+kc) of a [N,K] weight onto card c. Q8_0 → take the
    // K-sub-range of each output row (qs[n*K+k], d[n*(K/32)+b]); kc must be 32-aligned
    // (Q8_0 block). Produces a PARTIAL [T,N] → all-reduce.
    auto build_row = [](DeviceAllocator& a, const GgufTensorInfo* t,
                        std::vector<void*>& own, std::string& e, uint64_t& bytes,
                        uint32_t k0, uint32_t kc) -> SplitW {
        SplitW w{};
        if (!t) { e = "row tensor not found"; return w; }
        if (t->n_dims != 2) { e = "row weight: expected 2-D"; return w; }
        const uint32_t K = uint32_t(t->shape[0]), N = uint32_t(t->shape[1]);
        if (uint64_t(k0) + kc > K) { e = "row slice: bad geometry"; return w; }
        if ((k0 % 32) || (kc % 32)) { e = "row slice: not 32-aligned"; return w; }
        w.K = kc; w.N = N;
        if (t->dtype == DType::kQ8_0) {
            const uint32_t bpc = K / 32, bk0 = k0 / 32, nbk = kc / 32;
            const auto* blocks = reinterpret_cast<const block_q8_0*>(t->data);
            std::vector<int8_t>   qs(uint64_t(N) * kc);
            std::vector<uint16_t> dd(uint64_t(N) * nbk);
            for (uint32_t n = 0; n < N; ++n)
                for (uint32_t bb = 0; bb < nbk; ++bb) {
                    const block_q8_0& blk = blocks[uint64_t(n) * bpc + (bk0 + bb)];
                    dd[uint64_t(n) * nbk + bb] = *reinterpret_cast<const uint16_t*>(&blk.d);
                    for (int i = 0; i < 32; ++i)
                        qs[uint64_t(n) * kc + uint64_t(bb) * 32 + i] = blk.qs[i];
                }
            auto* dqs = static_cast<int8_t*>(a.malloc(qs.size()));
            auto* ddd = static_cast<uint16_t*>(a.malloc(dd.size() * sizeof(uint16_t)));
            if (!dqs || !ddd) { e = "Q8_0 row alloc"; return w; }
            own.push_back(dqs); own.push_back(ddd);
            a.queue().memcpy(dqs, qs.data(), qs.size()).wait();
            a.queue().memcpy(ddd, dd.data(), dd.size() * sizeof(uint16_t)).wait();
            w.q8_qs = dqs; w.q8_d = ddd;
            bytes += qs.size() + dd.size() * sizeof(uint16_t);
        } else {
            e = std::string("row slice: dtype ") + std::string(type_name(t->dtype)) +
                " unsupported (Phase-0 FFN shard needs Q8_0)";
        }
        return w;
    };

    // token_embd + output_norm + lm_head on card 0.
    {
        DeviceAllocator& ea = fleet.dev(0);
        const auto* ti = Ttop("token_embd.weight");
        if (!ti) return "token_embd: not found";
        if (ti->dtype != DType::kQ4_K && ti->dtype != DType::kQ6_K && ti->dtype != DType::kQ8_0)
            return std::string("token_embd: unsupported dtype ") +
                   std::string(type_name(ti->dtype));
        token_embd_dtype_ = ti->dtype;
        token_embd_ = dense::upload<void>(ea, ti, owned_[0], err, ti->dtype);
        if (!err.empty()) return "token_embd: " + err;
        dev_bytes_[0] += ti->nbytes;
    }
    {
        DeviceAllocator& ha = fleet.dev(0);
        output_norm_ = dense::upload<float>(ha, Ttop("output_norm.weight"),
                                            owned_[0], err, DType::kF32);
        if (!err.empty()) return "output_norm: " + err;
        const auto* ti = Ttop("output.weight");
        if (!ti) ti = Ttop("token_embd.weight");
        output_ = build_split(ha, ti, owned_[0], err, dev_bytes_[0]);
        if (!err.empty()) return "output: " + err;
    }

    const uint32_t n_layers = cfg.n_transformer_layers();
    dlayers_.assign(n_dev_, std::vector<LayerW>(n_layers));

    // Per-card weight residency. attn/DeltaNet REPLICATED full (build_split on each
    // card's allocator); FFN sharded (build_col gate/up, build_row down).
    for (uint32_t c = 0; c < n_dev_; ++c) {
        DeviceAllocator& a = fleet.dev(c);
        auto& own = owned_[c];
        const Shard& s = shard_[c];
        for (uint32_t L = 0; L < n_layers; ++L) {
            LayerW& w = dlayers_[c][L];
            w.is_linear = cfg.recurrent_layer(L);

            auto LW = [&](const char* n) -> SplitW {
                return build_split(a, Tl(L, n), own, err, dev_bytes_[c]);
            };
            auto F32 = [&](const char* n, float*& dst) -> std::string {
                const auto* ti = Tl(L, n);
                dst = dense::upload<float>(a, ti, own, err, DType::kF32);
                if (!err.empty()) return std::string(n) + ": " + err;
                if (ti) dev_bytes_[c] += ti->nbytes;
                return {};
            };
            auto layer_err = [&](const char* what) {
                return "card " + std::to_string(c) + " layer " + std::to_string(L) +
                       " " + what + ": " + err;
            };

            if (auto m = F32("attn_norm.weight", w.attn_norm); !m.empty())
                return layer_err(m.c_str());
            if (auto m = F32("post_attention_norm.weight", w.post_attn_norm); !m.empty())
                return layer_err(m.c_str());

            if (w.is_linear) {
                w.attn_qkv  = LW("attn_qkv.weight");  if (!err.empty()) return layer_err("attn_qkv");
                w.attn_gate = LW("attn_gate.weight"); if (!err.empty()) return layer_err("attn_gate");
                if (auto m = F32("ssm_a", w.ssm_a); !m.empty()) return layer_err(m.c_str());
                const uint32_t svh_pad = ((cfg.ssm_n_v_heads + 63u) / 64u) * 64u;
                w.ssm_alpha = upload_f32_proj_fp16(a, Tl(L, "ssm_alpha.weight"), own, err, svh_pad);
                if (!err.empty()) return layer_err("ssm_alpha");
                w.ssm_beta  = upload_f32_proj_fp16(a, Tl(L, "ssm_beta.weight"), own, err, svh_pad);
                if (!err.empty()) return layer_err("ssm_beta");
                if (auto m = F32("ssm_dt.bias", w.ssm_dt_bias); !m.empty()) return layer_err(m.c_str());
                {   // ssm_conv1d → fp16
                    const auto* ti = Tl(L, "ssm_conv1d.weight");
                    if (auto m = F32("ssm_conv1d.weight", w.ssm_conv1d); !m.empty()) return layer_err(m.c_str());
                    const uint64_t n = ti->nbytes / sizeof(float);
                    w.ssm_conv1d_fp16 = static_cast<sycl::half*>(a.malloc(n * sizeof(sycl::half)));
                    if (!w.ssm_conv1d_fp16) return layer_err("ssm_conv1d fp16 malloc");
                    own.push_back(w.ssm_conv1d_fp16); dev_bytes_[c] += n * sizeof(sycl::half);
                    cast_fp32_to_fp16(a.queue(), w.ssm_conv1d, w.ssm_conv1d_fp16, n).wait();
                }
                {   // ssm_norm → fp16
                    const auto* ti = Tl(L, "ssm_norm.weight");
                    if (auto m = F32("ssm_norm.weight", w.ssm_norm); !m.empty()) return layer_err(m.c_str());
                    const uint64_t n = ti->nbytes / sizeof(float);
                    w.ssm_norm_fp16 = static_cast<sycl::half*>(a.malloc(n * sizeof(sycl::half)));
                    if (!w.ssm_norm_fp16) return layer_err("ssm_norm fp16 malloc");
                    own.push_back(w.ssm_norm_fp16); dev_bytes_[c] += n * sizeof(sycl::half);
                    cast_fp32_to_fp16(a.queue(), w.ssm_norm, w.ssm_norm_fp16, n).wait();
                }
                w.ssm_out = LW("ssm_out.weight");   if (!err.empty()) return layer_err("ssm_out");
            } else {
                w.attn_q      = LW("attn_q.weight");      if (!err.empty()) return layer_err("attn_q");
                w.attn_k      = LW("attn_k.weight");      if (!err.empty()) return layer_err("attn_k");
                w.attn_v      = LW("attn_v.weight");      if (!err.empty()) return layer_err("attn_v");
                w.attn_output = LW("attn_output.weight"); if (!err.empty()) return layer_err("attn_output");
                if (auto m = F32("attn_q_norm.weight", w.attn_q_norm); !m.empty()) return layer_err(m.c_str());
                if (auto m = F32("attn_k_norm.weight", w.attn_k_norm); !m.empty()) return layer_err(m.c_str());
            }

            // FFN SHARD: gate/up column [f0,f0+fc); down row [f0,f0+fc).
            w.ffn_gate = build_col(a, Tl(L, "ffn_gate.weight"), own, err, dev_bytes_[c], s.f0, s.fc);
            if (!err.empty()) return layer_err("ffn_gate");
            w.ffn_up   = build_col(a, Tl(L, "ffn_up.weight"),   own, err, dev_bytes_[c], s.f0, s.fc);
            if (!err.empty()) return layer_err("ffn_up");
            w.ffn_down = build_row(a, Tl(L, "ffn_down.weight"),  own, err, dev_bytes_[c], s.f0, s.fc);
            if (!err.empty()) return layer_err("ffn_down");
        }
    }

    // Per-card hybrid caches: Phase 0 every card runs every layer's attn+DeltaNet
    // replicated, so each card holds ALL linear + ALL full-attn layers.
    std::vector<uint32_t> n_lin(1, 0), n_full(1, 0);
    for (uint32_t L = 0; L < n_layers; ++L) {
        if (cfg.recurrent_layer(L)) n_lin[0]++; else n_full[0]++;
    }
    dn_.resize(n_dev_); kv_.resize(n_dev_);
    for (uint32_t c = 0; c < n_dev_; ++c) {
        if (n_lin[0]) {
            DeltaNetStateConfig dc{};
            dc.n_layers_linear = n_lin[0];
            dc.n_v_heads       = cfg.ssm_n_v_heads;       // 48
            dc.v_head_dim      = cfg.ssm_state;           // 128
            dc.k_head_dim      = cfg.ssm_state;           // 128
            dc.conv_channels   = cfg.ssm_inner + 2u * cfg.ssm_n_k_heads * cfg.ssm_state;  // 10240
            dc.conv_kernel     = cfg.ssm_conv_kernel;     // 4
            if (auto m = dn_[c].init(fleet.dev(c), dc); !m.empty())
                return "dn cache dev " + std::to_string(c) + ": " + m;
        }
        if (n_full[0]) {
            KvCacheConfig kc{};
            kc.n_layers_full = n_full[0];
            kc.n_kv_heads    = d.n_kv_heads;              // 4
            kc.max_ctx       = max_ctx;
            kc.head_dim      = d.head_dim;                // 256
            if (auto m = kv_[c].init(fleet.dev(c), kc); !m.empty())
                return "kv cache dev " + std::to_string(c) + ": " + m;
        }
    }

    // Per-card scratch: act_q8 + prefill_bt (lazy). Kmax over all projections.
    const uint32_t Kmax = std::max(std::max(d.hidden, d.ffn),
                                   std::max(cfg.ssm_inner, d.n_q_heads * d.head_dim));
    act_q8_.assign(n_dev_, nullptr);
    prefill_bt_.assign(n_dev_, nullptr);
    prefill_bt_cap_.assign(n_dev_, 0);
    for (uint32_t c = 0; c < n_dev_; ++c) {
        // Row 0 serves plain decode; rows 0..15 serve batched spec verification.
        void* p = fleet.dev(c).malloc((uint64_t(Kmax) / 32) * sizeof(block_q8_1x) * 16u);
        if (!p) return "act_q8 alloc dev " + std::to_string(c);
        owned_[c].push_back(p);
        act_q8_[c] = p;
    }
    for (uint32_t c = 0; c < n_dev_; ++c)
        std::fprintf(stderr, "[qwen35tp] card %u weights: %.2f GB (f0=%u fc=%u)\n",
                     c, double(dev_bytes_[c]) / 1e9, shard_[c].f0, shard_[c].fc);
    return {};
}

void Qwen35TpModel::free_ws(uint32_t dev) {
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
                    static_cast<void*>(w.ffn_h), static_cast<void*>(w.ffn_part)})
        if (p) alloc.free(p);
    w = Workspace{};
}

std::string Qwen35TpModel::ensure_ws(uint32_t dev, uint32_t max_T) {
    Workspace& w = ws_[dev];
    if (max_T <= w.T) return {};
    if (w.T != 0) free_ws(dev);
    auto& alloc = fleet_->dev(dev);
    auto ah = [&](size_t n) { return static_cast<sycl::half*>(alloc.malloc(n * sizeof(sycl::half))); };
    auto af = [&](size_t n) { return static_cast<float*>(alloc.malloc(n * sizeof(float))); };

    const DenseConfig& d = cfg_.dense;
    const uint64_t T   = max_T;
    const uint32_t H   = d.hidden;
    const uint32_t N_q = d.n_q_heads * d.head_dim;
    const uint32_t N_qg = N_q * 2u;
    const uint32_t N_kv = d.n_kv_heads * d.head_dim;
    const uint32_t SI  = cfg_.ssm_inner;
    const uint32_t cvc = SI + 2u * cfg_.ssm_n_k_heads * cfg_.ssm_state;
    const uint32_t Vd  = cfg_.ssm_n_v_heads * cfg_.ssm_state;
    const uint32_t Nv  = cfg_.ssm_n_v_heads;
    const uint32_t Nvp = ((Nv + 63u) / 64u) * 64u;
    const uint32_t fc  = shard_[dev].fc;   // this card's FFN slice

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
    w.ffn_gate   = ah(T * fc);
    w.ffn_up     = ah(T * fc);
    w.ffn_h      = ah(T * fc);
    w.ffn_part   = ah(T * H);   // ffn_down partial → all-reduce

    if (!w.x || !w.x_normed || !w.attn_block || !w.positions || !w.qg || !w.q ||
        !w.gate || !w.k || !w.v || !w.attn_out || !w.dn_qkv || !w.dn_conv ||
        !w.dn_z || !w.dn_qpre || !w.dn_kpre || !w.dn_vpre || !w.dn_g || !w.dn_beta ||
        !w.dn_out || !w.dn_qrep || !w.dn_krep || !w.dn_alpha_h || !w.dn_beta_h ||
        !w.dn_alpha64 || !w.dn_beta64 || !w.ffn_gate || !w.ffn_up || !w.ffn_h ||
        !w.ffn_part)
        return "qwen35tp workspace alloc failed on dev " + std::to_string(dev);

    if (kv_[dev].ready()) {
        const uint32_t max_ctx = kv_[dev].config().max_ctx;
        constexpr uint32_t Bc_floor = 64;
        const uint32_t n_chunks_max = (max_ctx + Bc_floor - 1) / Bc_floor;
        const uint64_t n_floats = uint64_t(n_chunks_max) * d.n_q_heads * (d.head_dim + 2);
        w.attn_partials = static_cast<float*>(alloc.malloc(n_floats * sizeof(float)));
        if (!w.attn_partials)
            return "qwen35tp attn_partials alloc failed on dev " + std::to_string(dev);
        w.partials_ctx = max_ctx;
    }
    w.T = max_T;
    return {};
}

// COPY of Qwen35SplitModel::sgemv (copy-not-hoist). Decode int-dot W8A8 over the
// packed SoA weight; prefill dequant-to-fp16 + gemm; non-Q8_0 → dense path.
sycl::event Qwen35TpModel::sgemv(uint32_t dev, const sycl::half* A, const SplitW& w,
                                 sycl::half* out, uint32_t K, uint32_t N, uint32_t T) {
    auto& alloc = fleet_->dev(dev);
    auto& q = alloc.queue();
    if (!w.q8_qs)
        return dense::gemv_q_T(q, A, w.fp, out, K, N, T);
    if (T == 1) {
        quantize_q8_1(q, A, act_q8_[dev], K);
        static const bool soa_gmem = std::getenv("IE_QWEN35_SOA_GMEM") != nullptr;
        if (soa_gmem)
            return gemv_q8_0_soa_q8_g(q, act_q8_[dev], w.q8_qs, w.q8_d, out, K, N);
        return gemv_q8_0_soa_q8(q, act_q8_[dev], w.q8_qs, w.q8_d, out, K, N);
    }
    if (spec_verify_gemv_ && T >= 2 && T <= 16) {
        quantize_q8_1(q, A, act_q8_[dev], uint32_t(uint64_t(T) * K));
        return gemv_q8_0_soa_q8_batched(q, act_q8_[dev], w.q8_qs, w.q8_d,
                                        out, K, N, T);
    }
    const uint64_t need = uint64_t(K) * N;
    if (need > prefill_bt_cap_[dev]) {
        if (prefill_bt_[dev]) alloc.free(prefill_bt_[dev]);
        prefill_bt_[dev] = static_cast<sycl::half*>(alloc.malloc(need * sizeof(sycl::half)));
        prefill_bt_cap_[dev] = prefill_bt_[dev] ? need : 0;
    }
    if (!prefill_bt_[dev]) { std::fprintf(stderr, "qwen35tp: prefill_bt alloc failed\n"); return {}; }
    dequant_q8_0_soa_to_Bt(q, w.q8_qs, w.q8_d, prefill_bt_[dev], K, N);
    return dense::gemv_q_T(q, A, DenseQuantPtr{prefill_bt_[dev], DType::kF16}, out, K, N, T);
}

std::string Qwen35TpModel::forward(const int32_t* input_ids, uint32_t T,
                                   uint32_t start_pos, bool reset_kv,
                                   sycl::half* out_logits_host,
                                   sycl::half* all_logits_dev,
                                   sycl::half* hidden_pre_norm_dev) {
    if (T == 0) return "T == 0";
    const bool replay_mode = spec_verify_gemv_ && spec_replay_.size() == n_dev_;
    if (replay_mode)
        for (uint32_t c = 0; c < n_dev_; ++c)
            if (spec_replay_[c].K < T)
                return "spec replay too small on dev " + std::to_string(c);
    const DenseConfig& dc = cfg_.dense;
    const uint32_t H    = dc.hidden;
    const uint32_t HD   = dc.head_dim;
    const uint32_t N_q  = dc.n_q_heads  * HD;
    const uint32_t N_qg = N_q * 2u;
    const uint32_t N_kv = dc.n_kv_heads * HD;
    const uint32_t V    = dc.vocab;
    const uint32_t rope_n = dc.rope_dim;
    const float    eps  = dc.rms_eps;
    const uint32_t n_layers = cfg_.n_transformer_layers();
    const uint32_t n_kv = dc.n_kv_heads;

    // env-gated decode profiler (IE_QWEN35_TP_PROFILE=1).
    static const bool prof = std::getenv("IE_QWEN35_TP_PROFILE") != nullptr;
    using pclk = std::chrono::steady_clock;
    static double pf_dn = 0, pf_attn = 0, pf_ffn = 0, pf_ar = 0, pf_head = 0, pf_total = 0;
    static uint64_t pf_tok = 0, pf_calls = 0;
    auto pms = [](pclk::time_point a, pclk::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    pclk::time_point pf_t0 = pclk::now();
    pclk::time_point pf_last;
    auto mark_all = [&](double& acc) {
        if (!prof) return;
        for (uint32_t c = 0; c < n_dev_; ++c) fleet_->dev(c).queue().wait();
        auto now = pclk::now();
        acc += pms(pf_last, now);
        pf_last = now;
    };
    const bool spec_prof = spec_verify_gemv_ && spec_profile_.enabled;
    pclk::time_point spec_last;
    auto spec_mark = [&](double& acc) {
        if (!spec_prof) return;
        for (uint32_t c = 0; c < n_dev_; ++c) fleet_->dev(c).queue().wait();
        const auto now = pclk::now();
        acc += pms(spec_last, now);
        spec_last = now;
    };

    if (ws_.size() != n_dev_) ws_.assign(n_dev_, {});
    for (uint32_t c = 0; c < n_dev_; ++c)
        if (auto m = ensure_ws(c, T); !m.empty()) return m;

    std::vector<int32_t> pos(T);
    for (uint32_t t = 0; t < T; ++t) pos[t] = int32_t(start_pos + t);
    for (uint32_t c = 0; c < n_dev_; ++c) {
        if (ws_[c].positions)
            fleet_->dev(c).queue().memcpy(ws_[c].positions, pos.data(), T * sizeof(int32_t)).wait();
        if (reset_kv) {
            if (dn_[c].ready()) dn_[c].reset(fleet_->dev(c).queue());
            if (kv_[c].ready()) kv_[c].reset();
        }
    }

    // embedding on card 0 → ws_[0].x, broadcast to all cards (replicated x).
    {
        auto& alloc = fleet_->dev(0);
        auto& q = alloc.queue();
        int32_t* d_ids = static_cast<int32_t*>(alloc.malloc(T * sizeof(int32_t)));
        if (!d_ids) return "d_ids alloc failed";
        q.memcpy(d_ids, input_ids, T * sizeof(int32_t)).wait();
        if (token_embd_dtype_ == DType::kQ4_K)
            embedding_lookup_q4k(q, d_ids, token_embd_, ws_[0].x, T, H);
        else if (token_embd_dtype_ == DType::kQ8_0)
            embedding_lookup_q8_0(q, d_ids, token_embd_, ws_[0].x, T, H);
        else
            embedding_lookup_q6k(q, d_ids, token_embd_, ws_[0].x, T, H);
        q.wait();
        alloc.free(d_ids);
    }
    for (uint32_t c = 1; c < n_dev_; ++c)
        fleet_->copy_across(0, ws_[c].x, c, ws_[0].x, uint64_t(T) * H * sizeof(sycl::half));

    // all-reduce buffer list (ffn_part on every card).
    std::vector<sycl::half*> ffn_part_bufs(n_dev_);
    for (uint32_t c = 0; c < n_dev_; ++c) ffn_part_bufs[c] = ws_[c].ffn_part;

    if (prof) { for (uint32_t c = 0; c < n_dev_; ++c) fleet_->dev(c).queue().wait(); pf_last = pclk::now(); }
    if (spec_prof) {
        for (uint32_t c = 0; c < n_dev_; ++c) fleet_->dev(c).queue().wait();
        spec_last = pclk::now();
    }

    for (uint32_t L = 0; L < n_layers; ++L) {
        // ---- attn / DeltaNet REPLICATED on every card (no all-reduce). Each card
        // runs the identical block on its replicated x → attn_block bit-identical.
        for (uint32_t c = 0; c < n_dev_; ++c) {
            Workspace& w = ws_[c];
            auto& q = fleet_->dev(c).queue();
            const LayerW& w_l = dlayers_[c][L];
            const uint64_t per_layer_kv =
                uint64_t(n_kv) * (kv_[c].ready() ? kv_[c].config().max_ctx : 0u) * HD;
            // local hybrid index == global (every card holds every layer in order)
            uint32_t dn_local = 0, kv_local = 0;
            for (uint32_t LL = 0; LL < L; ++LL) {
                if (cfg_.recurrent_layer(LL)) dn_local++; else kv_local++;
            }

            rms_norm_f32w(q, w.x, w_l.attn_norm, w.x_normed, T, H, eps);

            if (w_l.is_linear) {
                const uint32_t SKH = cfg_.ssm_n_k_heads;
                const uint32_t SVH = cfg_.ssm_n_v_heads;
                const uint32_t SHD = cfg_.ssm_state;
                const uint32_t SI  = cfg_.ssm_inner;
                const uint32_t conv_ch = SI + 2u * SKH * SHD;
                const uint32_t kw  = SKH * SHD;
                const uint32_t rep = SVH / SKH;
                const float qscale = 1.0f / sycl::sqrt(float(SHD));

                SpecReplay* replay = replay_mode ? &spec_replay_[c] : nullptr;
                const uint64_t replay_row = replay
                    ? uint64_t(dn_local) * replay->K : 0;
                sycl::half* dn_qkv = replay
                    ? replay->qkv + replay_row * conv_ch : w.dn_qkv;
                float* dn_qrep = replay
                    ? replay->qrep + replay_row * uint64_t(SVH) * SHD : w.dn_qrep;
                float* dn_krep = replay
                    ? replay->krep + replay_row * uint64_t(SVH) * SHD : w.dn_krep;
                float* dn_vpre = replay
                    ? replay->vpre + replay_row * uint64_t(SVH) * SHD : w.dn_vpre;
                float* dn_g = replay
                    ? replay->g + replay_row * SVH : w.dn_g;
                float* dn_beta = replay
                    ? replay->beta + replay_row * SVH : w.dn_beta;

                sgemv(c, w.x_normed, w_l.attn_qkv, dn_qkv, H, conv_ch, T);
                sycl::half* conv_state = dn_[c].conv_state_ptr() +
                    uint64_t(dn_local) * dn_[c].conv_elems_per_layer();
                depthwise_conv1d_causal(q, dn_qkv, w_l.ssm_conv1d_fp16,
                                        conv_state, w.dn_conv, T, conv_ch,
                                        cfg_.ssm_conv_kernel);
                cast_qkv_split_fp16_to_fp32(q, w.dn_conv, w.dn_qpre,
                                            w.dn_kpre, dn_vpre, T, kw, SI);
                l2_norm_scale(q, w.dn_qpre, w.dn_qpre, T * SKH, SHD, qscale, 1e-6f);
                l2_norm_scale(q, w.dn_kpre, w.dn_kpre, T * SKH, SHD, 1.0f,   1e-6f);
                repeat_interleave_heads(q, w.dn_qpre, dn_qrep, T, SKH, SHD, rep);
                repeat_interleave_heads(q, w.dn_kpre, dn_krep, T, SKH, SHD, rep);
                const uint32_t SVHp = ((SVH + 63u) / 64u) * 64u;
                if (spec_verify_gemv_ && T >= 2 && T <= 16 &&
                    w_l.ssm_alpha.dt == DType::kF16 && w_l.ssm_beta.dt == DType::kF16) {
                    // Per-row math REQUIRED for losslessness; the fused kernel runs
                    // the T=1 F16 leaf verbatim per (row, mat) in ONE launch
                    // (split's 2026-08-16 fusion, ported).
                    gemv_fp16_rows_dual(q, w.x_normed, H,
                        static_cast<const sycl::half*>(w_l.ssm_alpha.p), w.dn_alpha64,
                        static_cast<const sycl::half*>(w_l.ssm_beta.p),  w.dn_beta64,
                        SVHp, H, SVHp, T);
                } else if (spec_verify_gemv_ && T >= 2 && T <= 16) {
                    // Preserve the exact T=1 reduction order used by plain TP.
                    for (uint32_t t = 0; t < T; ++t) {
                        dense::gemv_q_T(q, w.x_normed + uint64_t(t) * H,
                                        w_l.ssm_alpha,
                                        w.dn_alpha64 + uint64_t(t) * SVHp,
                                        H, SVHp, 1);
                        dense::gemv_q_T(q, w.x_normed + uint64_t(t) * H,
                                        w_l.ssm_beta,
                                        w.dn_beta64 + uint64_t(t) * SVHp,
                                        H, SVHp, 1);
                    }
                } else {
                    dense::gemv_q_T(q, w.x_normed, w_l.ssm_alpha,
                                    w.dn_alpha64, H, SVHp, T);
                    dense::gemv_q_T(q, w.x_normed, w_l.ssm_beta,
                                    w.dn_beta64, H, SVHp, T);
                }
                extract_cols(q, w.dn_alpha64, w.dn_alpha_h, T, SVH, SVHp);
                extract_cols(q, w.dn_beta64,  w.dn_beta_h,  T, SVH, SVHp);
                compute_g_beta_h16(q, w.dn_alpha_h, w.dn_beta_h, w_l.ssm_a, w_l.ssm_dt_bias,
                                   dn_g, dn_beta, T, SVH);
                float* state_layer = dn_[c].state_ptr() +
                    uint64_t(dn_local) * dn_[c].state_elems_per_layer();
                deltanet_recurrence(q, dn_qrep, dn_krep, dn_vpre, dn_g, dn_beta,
                                    state_layer, w.dn_out,
                                    /*B=*/1, T, SVH, SHD, SHD);
                sgemv(c, w.x_normed, w_l.attn_gate, w.dn_z, H, SI, T);
                gated_rms_norm(q, w.dn_out, w.dn_z, w_l.ssm_norm_fp16, w.dn_qkv,
                               T * SVH, SHD, eps);
                sgemv(c, w.dn_qkv, w_l.ssm_out, w.attn_block, SI, H, T);
            } else {
                if (spec_verify_gemv_ && T >= 2 && T <= 16 &&
                    w_l.attn_q.q8_qs && w_l.attn_k.q8_qs && w_l.attn_v.q8_qs) {
                    // Verify: ONE quantize for q/k/v + dual k+v launch (split's
                    // 2026-08-16 fusion, ported). Same kernels/order → bit-identical.
                    quantize_q8_1(q, w.x_normed, act_q8_[c],
                                  uint32_t(uint64_t(T) * H));
                    gemv_q8_0_soa_q8_batched(q, act_q8_[c],
                        w_l.attn_q.q8_qs, w_l.attn_q.q8_d, w.qg, H, N_qg, T);
                    split_q_gate_per_head(q, w.qg, w.q, w.gate, T, dc.n_q_heads, HD);
                    gemv_q8_0_soa_q8_batched_dual(q, act_q8_[c],
                        w_l.attn_k.q8_qs, w_l.attn_k.q8_d, w.k,
                        w_l.attn_v.q8_qs, w_l.attn_v.q8_d, w.v, H, N_kv, T);
                } else {
                    sgemv(c, w.x_normed, w_l.attn_q, w.qg, H, N_qg, T);
                    split_q_gate_per_head(q, w.qg, w.q, w.gate, T, dc.n_q_heads, HD);
                    sgemv(c, w.x_normed, w_l.attn_k, w.k, H, N_kv, T);
                    sgemv(c, w.x_normed, w_l.attn_v, w.v, H, N_kv, T);
                }
                rms_norm_f32w(q, w.q, w_l.attn_q_norm, w.q, T * dc.n_q_heads,  HD, eps);
                rms_norm_f32w(q, w.k, w_l.attn_k_norm, w.k, T * dc.n_kv_heads, HD, eps);
                rope_partial(q, w.q, w.positions, w.q, T, dc.n_q_heads,  HD, rope_n, dc.rope_theta);
                rope_partial(q, w.k, w.positions, w.k, T, dc.n_kv_heads, HD, rope_n, dc.rope_theta);
                sycl::half* kc = kv_[c].k_ptr() + per_layer_kv * kv_local;
                sycl::half* vc = kv_[c].v_ptr() + per_layer_kv * kv_local;
                const uint32_t max_ctx = kv_[c].config().max_ctx;
                if (T == 1 && w.attn_partials) {
                    full_attention_fa2_decode(q, w.q, w.k, w.v, kc, vc, w.attn_out,
                                              w.attn_partials, start_pos,
                                              dc.n_q_heads, dc.n_kv_heads, HD, max_ctx);
                } else if (spec_verify_gemv_ && T <= 16 && w.attn_partials) {
                    // Verify with the exact sequential decode leaf. The batched
                    // prefill attention reduction order is not lossless-greedy.
                    for (uint32_t t = 0; t < T; ++t)
                        full_attention_fa2_decode(
                            q, w.q + uint64_t(t) * N_q,
                            w.k + uint64_t(t) * N_kv,
                            w.v + uint64_t(t) * N_kv, kc, vc,
                            w.attn_out + uint64_t(t) * N_q, w.attn_partials,
                            start_pos + t, dc.n_q_heads, dc.n_kv_heads,
                            HD, max_ctx);
                } else {
                    // Long-ctx prefill: fa2 wide-tile at ctx >= minctx (split/dense
                    // port — argmax-bit-identical at hd256; the naive O(T²) leaf
                    // re-reads the whole KV T× and was the split's long-prompt wall
                    // before the same gate). Opt-out IE_QWEN35_NO_FA2_TILE.
                    static const bool no_tile =
                        std::getenv("IE_QWEN35_NO_FA2_TILE") != nullptr;
                    // TP default minctx = 1024 (2026-08-16: the naive leaf's
                    // intra-chunk quadratic was 32% of prefill at chunk 2048 —
                    // tile@2K lifted 395→500 tok/s; text byte-identical at 2K
                    // and 9K vs the naive reference). Split keeps its 6144.
                    static const uint32_t tile_minctx = []() -> uint32_t {
                        const char* e = std::getenv("IE_QWEN35_FA2_TILE_MINCTX");
                        if (!e) return 1024u;
                        int v = std::atoi(e); return v > 0 ? uint32_t(v) : 1024u;
                    }();
                    if (!no_tile && HD == 256 && (start_pos + T) >= tile_minctx) {
                        full_attention_fa2_prefill_tile_gemma(
                            q, w.q, w.k, w.v, kc, vc, w.attn_out, T, start_pos,
                            dc.n_q_heads, dc.n_kv_heads, HD, max_ctx,
                            0 /*window: full causal*/);
                    } else {
                        full_attention(q, w.q, w.k, w.v, kc, vc, w.attn_out, T,
                                       start_pos, dc.n_q_heads, dc.n_kv_heads, HD,
                                       max_ctx);
                    }
                }
                kv_[c].set_length(kv_local, start_pos + T);
                sigmoid_gate(q, w.attn_out, w.gate, w.attn_out, uint64_t(T) * N_q);
                sgemv(c, w.attn_out, w_l.attn_output, w.attn_block, N_q, H, T);
            }
        }
        spec_mark(dlayers_[0][L].is_linear ? spec_profile_.dn_ms
                                            : spec_profile_.attn_ms);
        mark_all(dlayers_[0][L].is_linear ? pf_dn : pf_attn);

        // ---- dense FFN SHARDED (gate/up column, down row → partial) ----
        for (uint32_t c = 0; c < n_dev_; ++c) {
            Workspace& w = ws_[c];
            auto& q = fleet_->dev(c).queue();
            const LayerW& w_l = dlayers_[c][L];
            const uint32_t fc = shard_[c].fc;
            // residual + post-attention norm (fused) → x_normed (replicated, since
            // attn_block bit-identical on both cards → x and x_normed stay in sync).
            residual_add_rms_norm_fused(q, w.x, w.attn_block, w_l.post_attn_norm,
                                        w.x_normed, T, H, eps);
            if (spec_verify_gemv_ && T >= 2 && T <= 16 &&
                w_l.ffn_gate.q8_qs && w_l.ffn_up.q8_qs) {
                // Verify: gate+up in ONE dual launch off ONE quantize (split's
                // 2026-08-16 fusion, ported; both mats are the [H, fc] slice).
                quantize_q8_1(q, w.x_normed, act_q8_[c],
                              uint32_t(uint64_t(T) * H));
                gemv_q8_0_soa_q8_batched_dual(q, act_q8_[c],
                    w_l.ffn_gate.q8_qs, w_l.ffn_gate.q8_d, w.ffn_gate,
                    w_l.ffn_up.q8_qs,   w_l.ffn_up.q8_d,   w.ffn_up,
                    H, fc, T);
            } else {
                sgemv(c, w.x_normed, w_l.ffn_gate, w.ffn_gate, H, fc, T);
                sgemv(c, w.x_normed, w_l.ffn_up,   w.ffn_up,   H, fc, T);
            }
            swiglu(q, w.ffn_gate, w.ffn_up, w.ffn_h, uint64_t(T) * fc);
            sgemv(c, w.ffn_h, w_l.ffn_down, w.ffn_part, fc, H, T);   // PARTIAL [T,H]
        }
        mark_all(pf_ffn);

        fleet_->all_reduce_sum_fp16(ffn_part_bufs, uint64_t(T) * H);   // full FFN on every card
        mark_all(pf_ar);

        for (uint32_t c = 0; c < n_dev_; ++c) {
            Workspace& w = ws_[c];
            residual_add(fleet_->dev(c).queue(), w.x, w.ffn_part, w.x, uint64_t(T) * H);
        }
        spec_mark(spec_profile_.ffn_ar_ms);
    }
    for (uint32_t c = 0; c < n_dev_; ++c) fleet_->dev(c).queue().wait();

    // final norm + lm_head on card 0 (x replicated → card 0 holds full hidden).
    {
        Workspace& w = ws_[0];
        auto& alloc = fleet_->dev(0);
        auto& q = alloc.queue();
        if (prof) { q.wait(); pf_last = pclk::now(); }
        if (hidden_pre_norm_dev)
            q.memcpy(hidden_pre_norm_dev, w.x,
                     uint64_t(T) * H * sizeof(sycl::half));
        rms_norm_f32w(q, w.x, output_norm_, w.x_normed, T, H, eps);
        if (all_logits_dev) {
            sgemv(0, w.x_normed, output_, all_logits_dev, H, V, T).wait();
            q.memcpy(out_logits_host,
                     all_logits_dev + uint64_t(T - 1) * V,
                     uint64_t(V) * sizeof(sycl::half)).wait();
        } else {
            const sycl::half* last = w.x_normed + uint64_t(T - 1) * H;
            sycl::half* d_logits = static_cast<sycl::half*>(
                alloc.malloc(uint64_t(V) * sizeof(sycl::half)));
            if (!d_logits) return "logits alloc failed";
            sgemv(0, last, output_, d_logits, H, V, 1).wait();
            q.memcpy(out_logits_host, d_logits,
                     uint64_t(V) * sizeof(sycl::half)).wait();
            alloc.free(d_logits);
        }
        if (prof) { q.wait(); auto now = pclk::now(); pf_head += pms(pf_last, now); pf_last = now; }
    }

    if (prof) {
        pf_total += pms(pf_t0, pclk::now());
        pf_tok += T; pf_calls++;
        const double summed = pf_dn + pf_attn + pf_ffn + pf_ar + pf_head;
        const double pf_other = pf_total - summed;
        if (pf_calls % 32 == 0) {
            auto pct = [&](double x) { return pf_total > 0 ? 100.0 * x / pf_total : 0.0; };
            std::fprintf(stderr,
                "\n[qwen35tp PROFILE] calls=%llu tok=%llu (wait-bracketed; relative)\n"
                "  total       %8.2f ms  (%.2f ms/tok = %.1f tok/s instrumented)\n"
                "  DeltaNet    %8.2f ms  %5.1f%%\n"
                "  full-attn   %8.2f ms  %5.1f%%\n"
                "  dense FFN   %8.2f ms  %5.1f%%\n"
                "  all-reduce  %8.2f ms  %5.1f%%\n"
                "  head+bounce %8.2f ms  %5.1f%%\n"
                "  other       %8.2f ms  %5.1f%%\n",
                (unsigned long long)pf_calls, (unsigned long long)pf_tok,
                pf_total, pf_total / double(pf_tok), 1000.0 * double(pf_tok) / pf_total,
                pf_dn, pct(pf_dn), pf_attn, pct(pf_attn), pf_ffn, pct(pf_ffn),
                pf_ar, pct(pf_ar), pf_head, pct(pf_head), pf_other, pct(pf_other));
        }
    }
    return {};
}

std::string Qwen35TpModel::load_mtp_head(const GgufReader& g, uint32_t max_ctx) {
    if (!fleet_) return "load_mtp_head: model not loaded";
    if (auto e = mtp_.load(fleet_->dev(0), g, cfg_, max_ctx); !e.empty()) return e;
    // Borrow the model's Q8_0-SoA lm_head (card 0 = the MTP head's device):
    // halves the draft's dominant per-step head-GEMV bytes.
    if (output_.q8_qs) mtp_.use_soa_lm_head(output_.q8_qs, output_.q8_d);
    return {};
}

namespace {
inline int32_t tp_argmax_row(const sycl::half* row, uint32_t n) {
    int32_t best_i = 0;
    float best = float(row[0]);
    for (uint32_t i = 1; i < n; ++i) {
        const float x = float(row[i]);
        if (x > best) { best = x; best_i = int32_t(i); }
    }
    return best_i;
}
}  // namespace

std::string Qwen35TpModel::spec_generate(const int32_t* ids, uint32_t P0,
                                         uint32_t pf_chunk, uint32_t max_new,
                                         uint32_t K, const SpecEmit& emit,
                                         double* prefill_ms_out) {
    if (!mtp_.loaded) return "spec_generate: MTP head not loaded";
    if (K < 2 || K > 16) return "spec_generate: K must be in [2,16]";
    if (P0 == 0) return "spec_generate: empty prompt";
    if (pf_chunk == 0) pf_chunk = 256;
    if (std::getenv("IE_QWEN35_SPEC_PF_DEBUG"))
        std::fprintf(stderr, "[spec-tp] prefill P0=%u pf_chunk=%u -> %u chunk(s)\n",
                     P0, pf_chunk, (P0 + pf_chunk - 1) / pf_chunk);

    const uint32_t H = cfg_.dense.hidden;
    const uint32_t vocab = cfg_.dense.vocab;
    const uint32_t V = K;
    auto& a0 = fleet_->dev(0);
    auto& q0 = a0.queue();

    // Attention and DeltaNet are replicated in Phase-0 TP. Each card keeps one
    // pre-round DeltaNet snapshot and the exact verify recurrence inputs needed
    // for a conv+recurrence-only partial-accept replay.
    if (auto e = ensure_spec_replay(V); !e.empty()) return "spec replay: " + e;

    const uint32_t hid_rows = std::max<uint32_t>(V, pf_chunk);
    auto* d_hid = static_cast<sycl::half*>(
        a0.malloc(uint64_t(hid_rows) * H * sizeof(sycl::half)));
    auto* d_all = static_cast<sycl::half*>(
        a0.malloc(uint64_t(V) * vocab * sizeof(sycl::half)));
    auto* h_last = static_cast<sycl::half*>(
        a0.malloc(uint64_t(H) * sizeof(sycl::half)));
    auto free_scratch = [&]() {
        if (d_hid) a0.free(d_hid);
        if (d_all) a0.free(d_all);
        if (h_last) a0.free(h_last);
    };
    if (!d_hid || !d_all || !h_last) {
        free_scratch();
        return "spec scratch alloc failed";
    }

    std::vector<sycl::half> row(vocab);
    std::vector<sycl::half> Lrow(uint64_t(V) * vocab);

    const auto t_pf0 = std::chrono::steady_clock::now();
    uint32_t pos = 0;
    bool first = true;
    while (pos < P0) {
        const uint32_t n = std::min<uint32_t>(pf_chunk, P0 - pos);
        if (auto e = forward(ids + pos, n, pos, first, row.data(),
                             /*all_logits=*/nullptr, /*hidden=*/d_hid); !e.empty()) {
            free_scratch();
            return "spec prefill: " + e;
        }
        q0.memcpy(h_last, d_hid + uint64_t(n - 1) * H,
                  uint64_t(H) * sizeof(sycl::half)).wait();
        pos += n;
        first = false;
    }
    if (prefill_ms_out)
        *prefill_ms_out = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_pf0).count();
    int32_t tn = tp_argmax_row(row.data(), vocab);

    // IE_SPEC_DEBUG: dump conditioning + first-round draft/target ids (TP vs split diff).
    static const bool spec_dbg = std::getenv("IE_SPEC_DEBUG") != nullptr;
    if (spec_dbg) {
        std::vector<sycl::half> hh(H);
        q0.memcpy(hh.data(), h_last, H * sizeof(sycl::half)).wait();
        double s = 0; for (uint32_t i = 0; i < H; ++i) s += double(float(hh[i]));
        std::fprintf(stderr, "[spec-dbg tp] tn=%d h_last[0..3]=%.6f %.6f %.6f %.6f sum=%.4f\n",
                     tn, float(hh[0]), float(hh[1]), float(hh[2]), float(hh[3]), s);
    }

    spec_profile_ = SpecProfile{};
    spec_profile_.enabled = std::getenv("IE_QWEN35_PROFILE") != nullptr;
    spec_profile_.K = K;
    spec_verify_gemv_ = true;
    std::string ret;
    uint32_t emitted = 0;
    uint32_t p = P0;
    bool first_round = true;
    bool abort = false;
    std::vector<int32_t> drafted, vin(V), targ(V);
    std::vector<std::vector<uint32_t>> kvsnap(n_dev_);
    for (uint32_t c = 0; c < n_dev_; ++c)
        if (kv_[c].ready())
            kvsnap[c].resize(kv_[c].config().n_layers_full);

    while (emitted < max_new && !abort) {
        drafted.clear();
        std::chrono::steady_clock::time_point t_draft;
        if (spec_profile_.enabled) {
            q0.wait();
            t_draft = std::chrono::steady_clock::now();
        }
        mtp_.draft_device_argmax(q0, h_last, tn, /*p_base=*/0, V, drafted);
        if (spec_profile_.enabled) {
            q0.wait();
            spec_profile_.draft_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t_draft).count();
        }
        if (drafted.size() + 1 < V) {
            ret = "spec: draft too short";
            break;
        }

        std::chrono::steady_clock::time_point t_dn_ckpt;
        if (spec_profile_.enabled)
            t_dn_ckpt = std::chrono::steady_clock::now();
        if (auto e = snapshot_spec_state(); !e.empty()) {
            ret = "spec snapshot: " + e;
            break;
        }
        if (spec_profile_.enabled)
            spec_profile_.dn_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t_dn_ckpt).count();

        for (uint32_t c = 0; c < n_dev_; ++c)
            for (uint32_t l = 0; l < uint32_t(kvsnap[c].size()); ++l)
                kvsnap[c][l] = kv_[c].length(l);
        vin[0] = tn;
        for (uint32_t j = 1; j < V; ++j) vin[j] = drafted[j - 1];
        if (auto e = forward(vin.data(), V, p, /*reset_kv=*/false, row.data(),
                             d_all, d_hid); !e.empty()) {
            ret = "spec verify: " + e;
            break;
        }
        q0.memcpy(Lrow.data(), d_all,
                  uint64_t(V) * vocab * sizeof(sycl::half)).wait();
        for (uint32_t j = 0; j < V; ++j)
            targ[j] = tp_argmax_row(Lrow.data() + uint64_t(j) * vocab, vocab);
        uint32_t n = 0;
        for (uint32_t j = 1; j < V; ++j) {
            if (drafted[j - 1] == targ[j - 1]) ++n;
            else break;
        }
        const int32_t bonus = targ[n];
        const uint32_t accepted = n + 1;
        if (spec_dbg && p < P0 + 8)
            std::fprintf(stderr, "[spec-dbg tp] p=%u tn=%d drafted=[%d,%d] targ=[%d,%d,%d] acc=%u\n",
                         p, vin[0], V > 1 ? drafted[0] : -1, V > 2 ? drafted[1] : -1,
                         targ[0], V > 1 ? targ[1] : -1, V > 2 ? targ[2] : -1, n);

        if (accepted < V) {
            const auto t_replay = std::chrono::steady_clock::now();
            if (auto e = replay_spec_deltanet(accepted); !e.empty()) {
                ret = "spec replay: " + e;
                abort = true;
            }
            if (spec_profile_.enabled)
                spec_profile_.dn_ms += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t_replay).count();
            for (uint32_t c = 0; c < n_dev_ && !abort; ++c)
                for (uint32_t l = 0; l < uint32_t(kvsnap[c].size()); ++l)
                    kv_[c].set_length(l, kvsnap[c][l] + accepted);
        }
        if (abort) break;

        if (spec_profile_.enabled) {
            ++spec_profile_.rounds;
            spec_profile_.drafted += V - 1;
            spec_profile_.accepted += n;
            ++spec_profile_.bonus;
            if (spec_profile_.rounds == 16) {
                constexpr double forwards = 16.0;
                const double tau = double(spec_profile_.accepted +
                                          spec_profile_.bonus) /
                                   double(spec_profile_.rounds);
                std::fprintf(stderr,
                    "\n[qwen35tp PROFILE] forwards=16 K=%u "
                    "(wait-bracketed; window)\n"
                    "  draft                    %9.2f ms  %8.2f ms/fwd\n"
                    "  verify-DeltaNet+ckpt      %9.2f ms  %8.2f ms/fwd\n"
                    "  verify-attn               %9.2f ms  %8.2f ms/fwd\n"
                    "  verify-FFN+all-reduce     %9.2f ms  %8.2f ms/fwd\n"
                    "  acceptance rounds=%llu drafted=%llu accepted=%llu "
                    "bonus=%llu tau=%.3f\n",
                    spec_profile_.K,
                    spec_profile_.draft_ms, spec_profile_.draft_ms / forwards,
                    spec_profile_.dn_ms, spec_profile_.dn_ms / forwards,
                    spec_profile_.attn_ms, spec_profile_.attn_ms / forwards,
                    spec_profile_.ffn_ar_ms, spec_profile_.ffn_ar_ms / forwards,
                    (unsigned long long)spec_profile_.rounds,
                    (unsigned long long)spec_profile_.drafted,
                    (unsigned long long)spec_profile_.accepted,
                    (unsigned long long)spec_profile_.bonus, tau);
                spec_profile_.rounds = 0;
                spec_profile_.drafted = 0;
                spec_profile_.accepted = 0;
                spec_profile_.bonus = 0;
                spec_profile_.draft_ms = 0.0;
                spec_profile_.dn_ms = 0.0;
                spec_profile_.attn_ms = 0.0;
                spec_profile_.ffn_ar_ms = 0.0;
            }
        }

        auto do_emit = [&](int32_t id) -> bool {
            if (emitted >= max_new) return false;
            ++emitted;
            if (!emit(id)) { abort = true; return false; }
            return true;
        };
        if (!abort && first_round) do_emit(tn);
        for (uint32_t j = 0; j < n && !abort && emitted < max_new; ++j)
            do_emit(drafted[j]);
        if (!abort && emitted < max_new) do_emit(bonus);

        tn = bonus;
        q0.memcpy(h_last, d_hid + uint64_t(accepted - 1) * H,
                  uint64_t(H) * sizeof(sycl::half)).wait();
        p += accepted;
        first_round = false;
    }

    spec_verify_gemv_ = false;
    free_scratch();
    return ret;
}

}  // namespace ie
