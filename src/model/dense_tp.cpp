// src/model/dense_tp.cpp — TP: multi-GPU tensor-parallel dense forward.
// ADDITIVE — never touches DenseModel / DenseModelSplit. Re-uses dense_dispatch.hpp
// upload + GEMV helpers and the src/ops leaf functions. Each layer is split
// Megatron-style (column q/k/v/gate-up, row o-proj/ffn-down) across all cards;
// the per-card body mirrors DenseModel::forward, with two host-bounce all-reduces
// per layer. NOT bit-exact vs single-GPU (reduction order differs).
// Spec: docs/superpowers/specs/2026-06-12-tensor-parallel-decode-design.md.

#include "ie/dense_tp.hpp"

#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"

#include "dense_dispatch.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace ie {
namespace {

// Column-parallel slice: own output rows [n0, n0+ncount) of a ggml [N,K] weight
// (logical [N_out, K_in], shape[0]=K contiguous). Q4_K/Q6_K rows are each a
// (K/256)*block_bytes contiguous span → a row range is a contiguous byte range
// (no re-pack). F16 is stored transposed [K,N] in the dense path → build
// [K,ncount] by strided gather. Used for q/k/v (output heads) + gate/up (F).
DenseQuantPtr up_col(DeviceAllocator& alloc, const GgufTensorInfo* t,
                     std::vector<void*>& owned, std::string& err,
                     uint32_t n0, uint32_t ncount) {
    DenseQuantPtr out;
    if (!t) { err = "col tensor not found"; return out; }
    const uint64_t K = t->shape[0], N = t->shape[1];
    if (t->n_dims != 2 || uint64_t(n0) + ncount > N) { err = "col slice: bad geometry"; return out; }
    if (t->dtype == DType::kF16) {
        if (t->nbytes != K * N * sizeof(uint16_t)) { err = "col f16: geometry"; return out; }
        std::vector<uint16_t> stg(K * ncount);
        const auto* src = reinterpret_cast<const uint16_t*>(t->data);
        for (uint64_t k = 0; k < K; ++k)
            for (uint32_t j = 0; j < ncount; ++j)
                stg[k * ncount + j] = src[(uint64_t(n0) + j) * K + k];   // transposed [K,ncount]
        void* d = alloc.malloc(stg.size() * sizeof(uint16_t));
        if (!d) { err = "col f16 malloc"; return out; }
        alloc.queue().memcpy(d, stg.data(), stg.size() * sizeof(uint16_t)).wait();
        owned.push_back(d); out.p = d; out.dt = DType::kF16; return out;
    }
    if (t->dtype != DType::kQ4_K && t->dtype != DType::kQ6_K) {
        err = std::string("col slice '") + std::string(t->name) + "': dtype " +
              std::string(type_name(t->dtype)) + " unsupported (Q4_K/Q6_K/F16)";
        return out;
    }
    const uint64_t row_bytes = t->nbytes / N;
    if (row_bytes * N != t->nbytes) { err = "col slice: nbytes % N"; return out; }
    // 2026-10-28 (P3-B): a Q6_K slice as the c32 streams (the per-layer int-dot kernels through the shared dispatch)
    if (t->dtype == DType::kQ6_K &&
        dense::upload_q6k_c32(alloc, reinterpret_cast<const uint8_t*>(t->data) + uint64_t(n0) * row_bytes, uint32_t(K), ncount, out, owned))
        return out;
    void* d = alloc.malloc(uint64_t(ncount) * row_bytes);
    if (!d) { err = "col malloc"; return out; }
    alloc.queue().memcpy(d, reinterpret_cast<const uint8_t*>(t->data) + uint64_t(n0) * row_bytes,
                         uint64_t(ncount) * row_bytes).wait();
    owned.push_back(d); out.p = d; out.dt = t->dtype; return out;
}

// Row-parallel re-pack: own input-K range [k0, k0+klen) of a ggml [N,K] weight.
// For each of N output rows, copy superblocks [k0/256, (k0+klen)/256) (contiguous
// within a row). Q4_K/Q6_K only (k0, klen, K all %256). Used for o-proj (K=N_q)
// + ffn_down (K=F).
DenseQuantPtr up_row(DeviceAllocator& alloc, const GgufTensorInfo* t,
                     std::vector<void*>& owned, std::string& err,
                     uint32_t k0, uint32_t klen) {
    DenseQuantPtr out;
    if (!t) { err = "row tensor not found"; return out; }
    const uint64_t K = t->shape[0], N = t->shape[1];
    if (t->n_dims != 2 || uint64_t(k0) + klen > K) { err = "row slice: bad geometry"; return out; }
    if (t->dtype != DType::kQ4_K && t->dtype != DType::kQ6_K) {
        err = std::string("row slice '") + std::string(t->name) + "': dtype " +
              std::string(type_name(t->dtype)) + " unsupported (TP row-parallel needs Q4_K/Q6_K)";
        return out;
    }
    if ((k0 % 256) || (klen % 256) || (K % 256)) { err = "row slice: not 256-aligned"; return out; }
    const uint64_t row_bytes = t->nbytes / N;
    const uint64_t sb = row_bytes / (K / 256);          // bytes per 256-superblock
    if (sb * (K / 256) != row_bytes) { err = "row slice: superblock size mismatch"; return out; }
    const uint64_t sb0 = k0 / 256, nsb = klen / 256, new_row = nsb * sb;
    std::vector<uint8_t> stg(uint64_t(N) * new_row);
    const auto* src = reinterpret_cast<const uint8_t*>(t->data);
    for (uint64_t r = 0; r < N; ++r)
        std::memcpy(stg.data() + r * new_row, src + r * row_bytes + sb0 * sb, new_row);
    // 2026-10-28 (P3-B): the sliced rows [N x klen] as the c32 streams
    if (t->dtype == DType::kQ6_K && dense::upload_q6k_c32(alloc, stg.data(), klen, uint32_t(N), out, owned)) return out;
    void* d = alloc.malloc(stg.size());
    if (!d) { err = "row malloc"; return out; }
    alloc.queue().memcpy(d, stg.data(), stg.size()).wait();
    owned.push_back(d); out.p = d; out.dt = t->dtype; return out;
}

// Column-parallel slice WITH the llama Q/K inverse row-permute composed in (one
// host gather: dest row j ← src row perm[n0+j]). The permute is block-diagonal
// per head (it only reorders head_dim pairing), so a contiguous head range maps
// within itself → the column split + permute compose. Q4_K/Q6_K (llama q/k are
// never F16). `perm` is the full N-row permutation from llama_qk_unpermute_rows.
DenseQuantPtr up_col_perm(DeviceAllocator& alloc, const GgufTensorInfo* t,
                          std::vector<void*>& owned, std::string& err,
                          uint32_t n0, uint32_t ncount, const std::vector<uint32_t>& perm) {
    DenseQuantPtr out;
    if (!t) { err = "col-perm tensor not found"; return out; }
    const uint64_t K = t->shape[0], N = t->shape[1];
    if (t->n_dims != 2 || uint64_t(n0) + ncount > N || perm.size() != N) {
        err = "col-perm: bad geometry or perm size"; return out;
    }
    if (t->dtype != DType::kQ4_K && t->dtype != DType::kQ6_K) {
        err = std::string("col-perm '") + std::string(t->name) + "': dtype " +
              std::string(type_name(t->dtype)) + " unsupported (llama q/k expect Q4_K/Q6_K)";
        return out;
    }
    const uint64_t row_bytes = t->nbytes / N;
    if (row_bytes * N != t->nbytes) { err = "col-perm: nbytes % N"; return out; }
    std::vector<uint8_t> stg(uint64_t(ncount) * row_bytes);
    const auto* src = reinterpret_cast<const uint8_t*>(t->data);
    for (uint32_t j = 0; j < ncount; ++j)
        std::memcpy(stg.data() + uint64_t(j) * row_bytes,
                    src + uint64_t(perm[n0 + j]) * row_bytes, row_bytes);
    // 2026-10-28 (P3-B): the row-shuffled slice [ncount x K] as the c32 streams
    if (t->dtype == DType::kQ6_K && dense::upload_q6k_c32(alloc, stg.data(), uint32_t(K), ncount, out, owned)) return out;
    void* d = alloc.malloc(stg.size());
    if (!d) { err = "col-perm malloc"; return out; }
    alloc.queue().memcpy(d, stg.data(), stg.size()).wait();
    owned.push_back(d); out.p = d; out.dt = t->dtype; return out;
}

// Slice a [n] F32 bias vector → [n0, n0+ncount) on device. Optional (null t → null).
float* up_bias(DeviceAllocator& alloc, const GgufTensorInfo* t,
               std::vector<void*>& owned, std::string& err,
               uint32_t n0, uint32_t ncount) {
    if (!t) return nullptr;
    if (t->dtype != DType::kF32) { err = "bias: expected F32"; return nullptr; }
    void* d = alloc.malloc(uint64_t(ncount) * sizeof(float));
    if (!d) { err = "bias malloc"; return nullptr; }
    alloc.queue().memcpy(d, reinterpret_cast<const float*>(t->data) + n0,
                         uint64_t(ncount) * sizeof(float)).wait();
    owned.push_back(d); return static_cast<float*>(d);
}

}  // namespace

DenseModelTP::~DenseModelTP() { free_all(); }

void DenseModelTP::free_all() {
    if (!fleet_) return;
    for (uint32_t d = 0; d < owned_.size(); ++d)
        for (void* p : owned_[d]) fleet_->dev(d).free(p);
    owned_.clear();
}

std::string DenseModelTP::ensure_ws(uint32_t dev, uint32_t max_T) {
    Workspace& w = ws_[dev];
    if (max_T <= w.T) return {};
    auto& alloc = fleet_->dev(dev);
    auto ah = [&](size_t n) {
        auto* p = static_cast<sycl::half*>(alloc.malloc(n * sizeof(sycl::half)));
        if (p) owned_[dev].push_back(p);
        return p;
    };
    const Shard& s = shard_[dev];
    const uint32_t H = cfg_.hidden;
    const uint64_t T = max_T;
    w.x          = ah(T * H);
    w.x_normed   = ah(T * H);
    w.q          = ah(T * s.Nq);
    w.k          = ah(T * s.Nkv);
    w.v          = ah(T * s.Nkv);
    w.attn_out   = ah(T * s.Nq);
    w.attn_block = ah(T * H);
    w.gate       = ah(T * s.fc);
    w.up         = ah(T * s.fc);
    w.h          = ah(T * s.fc);
    w.positions  = static_cast<int32_t*>(alloc.malloc(T * sizeof(int32_t)));
    if (w.positions) owned_[dev].push_back(w.positions);
    w.ids        = static_cast<int32_t*>(alloc.malloc(T * sizeof(int32_t)));   // 2026-10-28 (P1): no per-call malloc
    if (w.ids) owned_[dev].push_back(w.ids);
    if (!w.q8) {   // 2026-10-28 (P1 fix): the decode step's Q8_1 activation scratch, sized for this card's largest GEMV input
                   // (H for q / k / v / gate / up, Nq for the o-proj, fc for ffn_down; all multiples of 256, so / 32 is exact).
                   // Null = the fp16 GEMV path (DenseModel's rule).
        const uint64_t kmax = std::max<uint64_t>({uint64_t(H), uint64_t(s.Nq), uint64_t(s.fc)});
        w.q8 = alloc.malloc((kmax / 32) * sizeof(block_q8_1x));
        if (w.q8) owned_[dev].push_back(w.q8);
    }
    if (!w.x || !w.x_normed || !w.q || !w.k || !w.v || !w.attn_out ||
        !w.attn_block || !w.gate || !w.up || !w.h || !w.positions || !w.ids)
        return "tp workspace alloc failed on dev " + std::to_string(dev);
    // 2026-10-28 (P3-B step 2): the decode attention ran the naive prefill kernel at T = 1 on this path -- ~50 us per
    // context token per step against the one card's ~3 us (fa2 decode): 11.5 tok/s at 901 tokens, 3.3 at 5261. The fa2
    // decode kernel needs its split-K partials: n_chunks(max_ctx, 64-key floor) x this card's q heads x (HD + 2) floats,
    // sized once from the KV's max context (the rule of DenseModel::ensure_attn_partials); a failed alloc keeps the
    // naive kernel.
    if (!w.attn_partials && dev < kv_.size() && kv_[dev].config().max_ctx) {
        constexpr uint32_t Bc_floor = 64;
        const uint32_t n_chunks = (kv_[dev].config().max_ctx + Bc_floor - 1) / Bc_floor;
        const uint64_t n_floats = uint64_t(n_chunks) * s.nq * (cfg_.head_dim + 2);
        auto* p = static_cast<float*>(alloc.malloc(n_floats * sizeof(float)));
        if (p) { owned_[dev].push_back(p); w.attn_partials = p; }
        static const bool naive = std::getenv("IE_DENSE_TP_NAIVE_ATTN") != nullptr;   // the A/B control for the gate
        std::fprintf(stderr, "[tp] card %u decode attention: %s (partials %.1f MB for %u q heads at ctx %u)\n", dev,
                     p && !naive ? "fa2 decode" : naive ? "naive (IE_DENSE_TP_NAIVE_ATTN)" : "naive (partials alloc failed)",
                     double(n_floats) * 4 / 1e6, s.nq, kv_[dev].config().max_ctx);
    }
    w.T = max_T;
    return {};
}

std::string DenseModelTP::ensure_workspace(uint32_t max_T) {
    if (!fleet_) return "tp: not loaded";
    for (uint32_t d = 0; d < n_dev_; ++d)
        if (auto m = ensure_ws(d, std::max<uint32_t>(max_T, 1u)); !m.empty()) return m;
    return {};
}

std::string DenseModelTP::load(DeviceFleet& fleet, const GgufReader& g,
                               const DenseConfig& cfg, uint32_t max_ctx) {
    fleet_ = &fleet;
    cfg_   = cfg;
    n_dev_ = fleet.size();
    if (n_dev_ == 0) return "tp: empty fleet";
    const uint32_t HD = cfg.head_dim, NQ = cfg.n_q_heads,
                   NKV = cfg.n_kv_heads, F = cfg.ffn;
    // llama: q/k GGUF rows are NORM-permuted by convert_hf_to_gguf; invert at load
    // (same as DenseModel/DenseModelSplit) so our NEOX rope is correct, composed
    // with the column split. The permute is per-head → it survives the head split.
    const bool unpermute = (cfg.arch == ModelArch::kLlama3);
    const std::vector<uint32_t> perm_q = unpermute ? llama_qk_unpermute_rows(NQ,  HD) : std::vector<uint32_t>{};
    const std::vector<uint32_t> perm_k = unpermute ? llama_qk_unpermute_rows(NKV, HD) : std::vector<uint32_t>{};
    if (cfg.n_layers == 0 || HD < 16 || (HD & (HD - 1)))
        return "tp: bad config";
    if (NQ % n_dev_ || NKV % n_dev_)
        return "tp: n_q_heads/n_kv_heads not divisible by n_dev";
    if (F % 256)
        return "tp: ffn not 256-aligned";

    owned_.assign(n_dev_, {});
    ws_.assign(n_dev_, {});
    kv_.resize(n_dev_);
    shard_.assign(n_dev_, {});
    layers_.assign(n_dev_, {});
    rope_freqs_per_dev_.assign(n_dev_, nullptr);
    dev_bytes_.assign(n_dev_, 0);

    // Per-card Megatron bookkeeping. q/kv heads split evenly; ffn split into
    // 256-superblock contiguous ranges (so the gate/up column split point ==
    // the ffn_down row split point and both land on a superblock boundary).
    const uint32_t nq_c = NQ / n_dev_, nkv_c = NKV / n_dev_;
    const uint32_t nsb = F / 256;
    uint32_t f_acc = 0;
    for (uint32_t c = 0; c < n_dev_; ++c) {
        Shard& s = shard_[c];
        s.q_head0 = c * nq_c; s.nq = nq_c;
        s.kv_head0 = c * nkv_c; s.nkv = nkv_c;
        s.Nq = nq_c * HD; s.Nkv = nkv_c * HD;
        const uint32_t my_sb = nsb / n_dev_ + (c < (nsb % n_dev_) ? 1u : 0u);
        s.f0 = f_acc; s.fc = my_sb * 256; f_acc += s.fc;
        // o-proj row split = this card's q-head columns → must be 256-aligned.
        if ((uint64_t(s.q_head0) * HD) % 256 || uint64_t(s.Nq) % 256)
            return "tp: o-proj K-slice (n_q*head_dim/n_dev) not 256-aligned";
    }
    if (f_acc != F) return "tp: ffn superblock split did not cover F";

    char buf[64];
    auto Top = [&](const char* name) { return g.find_tensor(name); };
    auto Tl = [&](uint32_t L, const char* name) {
        std::snprintf(buf, sizeof(buf), "blk.%u.%s", L, name);
        return g.find_tensor(buf);
    };
    std::string err;

    // top-level: token_embd + output_norm + output (lm_head) all on card 0.
    {
        const auto* ti = Top("token_embd.weight");
        if (!ti) return "token_embd: not found";
        if (ti->dtype != DType::kQ4_K && ti->dtype != DType::kQ6_K)
            return "token_embd: unsupported dtype";
        token_embd_dtype_ = ti->dtype;
        token_embd_ = dense::upload<void>(fleet.dev(0), ti, owned_[0], err, ti->dtype);
        if (!err.empty()) return "token_embd: " + err;
        dev_bytes_[0] += ti->nbytes;
    }
    output_norm_ = dense::upload<float>(fleet.dev(0), Top("output_norm.weight"),
                                        owned_[0], err, DType::kF32);
    if (!err.empty()) return "output_norm: " + err;
    {
        const auto* ti = Top("output.weight");
        if (!ti) { output_ = token_embd_; output_dtype_ = token_embd_dtype_; }   // tied — on card 0
        else {
            if (ti->dtype != DType::kQ4_K && ti->dtype != DType::kQ6_K)
                return "output: unsupported dtype";
            output_dtype_ = ti->dtype;
            output_ = dense::upload<void>(fleet.dev(0), ti, owned_[0], err, ti->dtype);
            if (!err.empty()) return "output: " + err;
            dev_bytes_[0] += ti->nbytes;
        }
    }
    // 2026-10-28 (P1 fix): the lm_head's SoA-Q6 streams for the int-dot decode kernel (DenseModel's universal Q6_K lm_head
    // rule, IE_Q6K_SOA_V2 default ON; any skip leaves the AoS kernel). On card 0, where the head runs.
    if (output_dtype_ == DType::kQ6_K) {
        const GgufTensorInfo* lmw = Top("output.weight");
        if (!lmw) lmw = Top("token_embd.weight");   // tied
        DenseQuantPtr tmp;
        dense::maybe_upload_q6k_soa_v2(fleet.dev(0), lmw, tmp, owned_[0]);
        out_soa_q6_lo_ = tmp.soa_q6_lo; out_soa_q6_hi_ = tmp.soa_q6_hi; out_soa_q6_sc_ = tmp.soa_q6_sc; out_soa_q6_d_ = tmp.soa_q6_d;
    }
    if (const auto* ti = Top("rope_freqs.weight")) {
        const uint64_t n = ti->nbytes / sizeof(float);
        if (n != cfg.rope_dim / 2u) return "rope_freqs: count mismatch";
        for (uint32_t c = 0; c < n_dev_; ++c) {
            rope_freqs_per_dev_[c] = dense::upload<float>(fleet.dev(c), ti, owned_[c], err, DType::kF32);
            if (!err.empty()) return "rope_freqs dev " + std::to_string(c) + ": " + err;
        }
    }

    for (uint32_t c = 0; c < n_dev_; ++c) {
        auto& alloc = fleet.dev(c);
        auto& own   = owned_[c];
        const Shard& s = shard_[c];
        layers_[c].assign(cfg.n_layers, {});
        for (uint32_t L = 0; L < cfg.n_layers; ++L) {
            TpLayer& w = layers_[c][L];
            // replicated norms
            w.attn_norm = dense::upload<float>(alloc, Tl(L, "attn_norm.weight"), own, err, DType::kF32);
            if (!err.empty()) return "attn_norm: " + err;
            w.ffn_norm  = dense::upload<float>(alloc, Tl(L, "ffn_norm.weight"),  own, err, DType::kF32);
            if (!err.empty()) return "ffn_norm: " + err;
            if (const auto* qn = Tl(L, "attn_q_norm.weight")) {
                w.attn_q_norm = dense::upload<float>(alloc, qn, own, err, DType::kF32);
                if (!err.empty()) return "attn_q_norm: " + err;
            }
            if (const auto* kn = Tl(L, "attn_k_norm.weight")) {
                w.attn_k_norm = dense::upload<float>(alloc, kn, own, err, DType::kF32);
                if (!err.empty()) return "attn_k_norm: " + err;
            }
            // column-parallel q/k/v (output-head slice); llama q/k un-permuted first
            w.attn_q = unpermute ? up_col_perm(alloc, Tl(L, "attn_q.weight"), own, err, s.q_head0 * HD,  s.Nq,  perm_q)
                                 : up_col(alloc, Tl(L, "attn_q.weight"), own, err, s.q_head0 * HD,  s.Nq);  if (!err.empty()) return "attn_q: " + err;
            w.attn_k = unpermute ? up_col_perm(alloc, Tl(L, "attn_k.weight"), own, err, s.kv_head0 * HD, s.Nkv, perm_k)
                                 : up_col(alloc, Tl(L, "attn_k.weight"), own, err, s.kv_head0 * HD, s.Nkv); if (!err.empty()) return "attn_k: " + err;
            w.attn_v = up_col(alloc, Tl(L, "attn_v.weight"), own, err, s.kv_head0 * HD, s.Nkv); if (!err.empty()) return "attn_v: " + err;
            w.attn_q_bias = up_bias(alloc, Tl(L, "attn_q.bias"), own, err, s.q_head0 * HD,  s.Nq);  if (!err.empty()) return "attn_q.bias: " + err;
            w.attn_k_bias = up_bias(alloc, Tl(L, "attn_k.bias"), own, err, s.kv_head0 * HD, s.Nkv); if (!err.empty()) return "attn_k.bias: " + err;
            w.attn_v_bias = up_bias(alloc, Tl(L, "attn_v.bias"), own, err, s.kv_head0 * HD, s.Nkv); if (!err.empty()) return "attn_v.bias: " + err;
            // row-parallel o-proj (input-K = q-head columns)
            w.attn_output = up_row(alloc, Tl(L, "attn_output.weight"), own, err, s.q_head0 * HD, s.Nq); if (!err.empty()) return "attn_output: " + err;
            // column-parallel gate/up (output F slice)
            w.ffn_gate = up_col(alloc, Tl(L, "ffn_gate.weight"), own, err, s.f0, s.fc); if (!err.empty()) return "ffn_gate: " + err;
            w.ffn_up   = up_col(alloc, Tl(L, "ffn_up.weight"),   own, err, s.f0, s.fc); if (!err.empty()) return "ffn_up: " + err;
            // row-parallel ffn_down (input-K = F slice)
            w.ffn_down = up_row(alloc, Tl(L, "ffn_down.weight"), own, err, s.f0, s.fc); if (!err.empty()) return "ffn_down: " + err;
            // weight-byte tally (sliced): the 7 matmuls dominate
            for (const auto& dq : {w.attn_q, w.attn_k, w.attn_v, w.attn_output, w.ffn_gate, w.ffn_up, w.ffn_down}) {
                (void)dq;  // exact sliced bytes are alloc-tracked; approximate via full/n_dev below
            }
        }
        // approximate per-card weight bytes = total layer matmul bytes / n_dev
        for (uint32_t L = 0; L < cfg.n_layers; ++L)
            for (const char* nm : {"attn_q.weight","attn_k.weight","attn_v.weight",
                                   "attn_output.weight","ffn_gate.weight","ffn_up.weight","ffn_down.weight"})
                if (const auto* ti = Tl(L, nm)) dev_bytes_[c] += ti->nbytes / n_dev_;

        KvCacheConfig kc{};
        kc.n_layers_full = cfg.n_layers;   // TP: every card holds every layer (split)
        kc.n_kv_heads    = s.nkv;          // this card's kv-head slice
        kc.max_ctx       = max_ctx;
        kc.head_dim      = HD;
        if (auto m = kv_[c].init(fleet.dev(c), kc); !m.empty())
            return "kv dev " + std::to_string(c) + ": " + m;
    }
    // 2026-10-28 (P1): the lm_head's output scratch once (forward allocated and freed it on every call)
    d_logits_ = static_cast<sycl::half*>(fleet.dev(0).malloc(uint64_t(cfg.vocab) * sizeof(sycl::half)));
    if (!d_logits_) return "tp: logits scratch alloc failed";
    owned_[0].push_back(d_logits_);
    return {};
}

std::string DenseModelTP::forward(const int32_t* input_ids, uint32_t T,
                                  uint32_t start_pos, bool reset_kv,
                                  sycl::half* out_logits_host, sycl::half* all_logits_dev) {
    if (T == 0) return "T == 0";
    const uint32_t H = cfg_.hidden, HD = cfg_.head_dim;
    const uint32_t rope_n = cfg_.rope_dim;
    const float eps = cfg_.rms_eps;

    for (uint32_t d = 0; d < n_dev_; ++d)
        if (auto m = ensure_ws(d, T); !m.empty()) return m;
    // 2026-10-28 (P1 fix): the decode step's launch sequence is DenseModel::forward's (dense_transformer.cpp), per card --
    // the Q8_1 activation quant fused into the norm and shared by q / k / v, one quant for gate / up, the int-dot ffn_down
    // and lm_head (the SoA streams), two host waits a step (the ids copy in, the logits out) instead of four -- so one
    // card on this class computes what DenseModel computes, launch for launch (identical numerics there, byte-identical
    // greedy text with the prompt cache off; two cards differ by the reduction order, as before). The P1
    // gate measured the re-quantizing GEMVs, the AoS W6A16 lm_head over the 166,912-row vocabulary and two host waits a
    // step at -11 % decode against DenseModel, with near-tie text drift.
    const bool dec = (T == 1) && dense::q8_decode_enabled();

    std::vector<int32_t> pos;
    if (T > 1) { pos.resize(T); for (uint32_t t = 0; t < T; ++t) pos[t] = int32_t(start_pos + t); }
    for (uint32_t d = 0; d < n_dev_; ++d) {
        if (T == 1) fleet_->dev(d).queue().fill(ws_[d].positions, int32_t(start_pos), 1);   // no host wait at decode
        else        fleet_->dev(d).queue().memcpy(ws_[d].positions, pos.data(), T * sizeof(int32_t)).wait();
        if (reset_kv) kv_[d].reset();
    }

    // embedding on card 0 → ws_[0].x, then broadcast the replicated x to all cards.
    {
        auto& q = fleet_->dev(0).queue();
        int32_t* d_ids = ws_[0].ids;   // 2026-10-28 (P1): the workspace's copy (sized with it), no per-call malloc
        q.memcpy(d_ids, input_ids, T * sizeof(int32_t)).wait();   // (the caller's host buffer: copied before this returns)
        if (token_embd_dtype_ == DType::kQ4_K) embedding_lookup_q4k(q, d_ids, token_embd_, ws_[0].x, T, H);
        else                                    embedding_lookup_q6k(q, d_ids, token_embd_, ws_[0].x, T, H);
        if (n_dev_ > 1) q.wait();   // the cross-card copy below reads ws_[0].x; on one card the in-order queue orders it
    }
    for (uint32_t c = 1; c < n_dev_; ++c)
        fleet_->copy_across(0, ws_[c].x, c, ws_[0].x, uint64_t(T) * H * sizeof(sycl::half));

    // per-layer split forward. Submit each card's pre-reduce work to its OWN
    // in-order queue without waiting; the all-reduce's per-queue gather forces
    // completion → the cards overlap. Two all-reduces/layer (o-proj + ffn_down).
    std::vector<sycl::half*> attn_block_bufs(n_dev_);
    for (uint32_t c = 0; c < n_dev_; ++c) attn_block_bufs[c] = ws_[c].attn_block;

    for (uint32_t L = 0; L < cfg_.n_layers; ++L) {
        // ---- attention half (each card: norm → qkv → attn → o-proj partial) ----
        for (uint32_t c = 0; c < n_dev_; ++c) {
            Workspace& w = ws_[c];
            const Shard& s = shard_[c];
            const TpLayer& lw = layers_[c][L];
            auto& q = fleet_->dev(c).queue();
            const float* ff = rope_freqs_per_dev_[c];
            void* q8 = dec ? w.q8 : nullptr;   // the decode step's Q8_1 scratch (null: the fp16 GEMV path)
            if (q8) rms_norm_f32w_q8(q, w.x, lw.attn_norm, w.x_normed, q8, 1, H, eps);   // the norm + the quant, one launch
            else    rms_norm_f32w(q, w.x, lw.attn_norm, w.x_normed, T, H, eps);
            if (q8) {
                dense::gemv_q_i8(q, w.x_normed, q8, lw.attn_q, w.q, H, s.Nq);
                dense::gemv_q_i8(q, w.x_normed, q8, lw.attn_k, w.k, H, s.Nkv);
                dense::gemv_q_i8(q, w.x_normed, q8, lw.attn_v, w.v, H, s.Nkv);
            } else {
                dense::gemv_q_T(q, w.x_normed, lw.attn_q, w.q, H, s.Nq,  T);
                dense::gemv_q_T(q, w.x_normed, lw.attn_k, w.k, H, s.Nkv, T);
                dense::gemv_q_T(q, w.x_normed, lw.attn_v, w.v, H, s.Nkv, T);
            }
            if (lw.attn_q_bias) add_bias(q, w.q, lw.attn_q_bias, T, s.Nq);
            if (lw.attn_k_bias) add_bias(q, w.k, lw.attn_k_bias, T, s.Nkv);
            if (lw.attn_v_bias) add_bias(q, w.v, lw.attn_v_bias, T, s.Nkv);
            if (lw.attn_q_norm) rms_norm_f32w(q, w.q, lw.attn_q_norm, w.q, T * s.nq,  HD, eps);
            if (lw.attn_k_norm) rms_norm_f32w(q, w.k, lw.attn_k_norm, w.k, T * s.nkv, HD, eps);
            if (ff) {
                rope_partial_ff(q, w.q, w.positions, w.q, T, s.nq,  HD, rope_n, cfg_.rope_theta, ff);
                rope_partial_ff(q, w.k, w.positions, w.k, T, s.nkv, HD, rope_n, cfg_.rope_theta, ff);
            } else {
                rope_partial(q, w.q, w.positions, w.q, T, s.nq,  HD, rope_n, cfg_.rope_theta);
                rope_partial(q, w.k, w.positions, w.k, T, s.nkv, HD, rope_n, cfg_.rope_theta);
            }
            const uint64_t per_layer = uint64_t(s.nkv) * kv_[c].config().max_ctx * HD;
            sycl::half* kc = kv_[c].k_ptr() + per_layer * L;
            sycl::half* vc = kv_[c].v_ptr() + per_layer * L;
            static const bool naive_attn = std::getenv("IE_DENSE_TP_NAIVE_ATTN") != nullptr;   // the before/after control
            if (T == 1 && w.attn_partials && !naive_attn)   // 2026-10-28 (P3-B step 2): the decode step on the fa2 kernel, as one card
                full_attention_fa2_decode(q, w.q, w.k, w.v, kc, vc, w.attn_out, w.attn_partials, start_pos,
                                          s.nq, s.nkv, HD, kv_[c].config().max_ctx);
            else
                full_attention(q, w.q, w.k, w.v, kc, vc, w.attn_out, T, start_pos,
                               s.nq, s.nkv, HD, kv_[c].config().max_ctx);
            kv_[c].set_length(L, start_pos + T);
            // o-proj: this card's attn_out × its o-proj row-slice → PARTIAL [T,H] (DenseModel's rule: a Q4_K o-proj takes its
            // own quant + the int-dot GEMV; a c32 Q6_K one goes through gemv_q_T -> gemv_q's c32 branch)
            if (q8 && lw.attn_output.dt == DType::kQ4_K) {
                quantize_q8_1(q, w.attn_out, q8, s.Nq);
                dense::gemv_q_i8(q, w.attn_out, q8, lw.attn_output, w.attn_block, s.Nq, H);
            } else
                dense::gemv_q_T(q, w.attn_out, lw.attn_output, w.attn_block, s.Nq, H, T);
        }
        fleet_->all_reduce_sum_fp16(attn_block_bufs, uint64_t(T) * H);   // full attn on every card

        // ---- ffn half (each card: residual+norm → gate/up/swiglu → ffn_down partial) ----
        for (uint32_t c = 0; c < n_dev_; ++c) {
            Workspace& w = ws_[c];
            const Shard& s = shard_[c];
            const TpLayer& lw = layers_[c][L];
            auto& q = fleet_->dev(c).queue();
            residual_add_rms_norm_fused(q, w.x, w.attn_block, lw.ffn_norm, w.x_normed, T, H, eps);
            void* q8 = dec ? w.q8 : nullptr;
            if (q8) {   // one quant of the FFN input for gate and up (DenseModel's 2h)
                quantize_q8_1(q, w.x_normed, q8, H);
                dense::gemv_q_i8(q, w.x_normed, q8, lw.ffn_gate, w.gate, H, s.fc);
                dense::gemv_q_i8(q, w.x_normed, q8, lw.ffn_up,   w.up,   H, s.fc);
            } else {
                dense::gemv_q_T(q, w.x_normed, lw.ffn_gate, w.gate, H, s.fc, T);
                dense::gemv_q_T(q, w.x_normed, lw.ffn_up,   w.up,   H, s.fc, T);
            }
            swiglu(q, w.gate, w.up, w.h, uint64_t(T) * s.fc);
            // ffn_down PARTIAL [T,H]: the c32 or Q4_K int-dot kernel on the quantized h (DenseModel's down_c32 / down_q8), an AoS
            // Q6_K that fits the SLM slab through gemv_q6_K_q8, else the T-row dispatch
            if (q8 && (lw.ffn_down.kq_lo || lw.ffn_down.dt == DType::kQ4_K)) {
                quantize_q8_1(q, w.h, q8, s.fc);
                dense::gemv_q_i8(q, w.h, q8, lw.ffn_down, w.attn_block, s.fc, H);
            } else if (q8 && lw.ffn_down.dt == DType::kQ6_K && dense::q6k_slm_gemv_q8_fits(s.fc)) {
                quantize_q8_1(q, w.h, q8, s.fc);
                gemv_q6_K_q8(q, q8, lw.ffn_down.p, w.attn_block, s.fc, H);
            } else
                dense::gemv_q_T(q, w.h, lw.ffn_down, w.attn_block, s.fc, H, T);
        }
        fleet_->all_reduce_sum_fp16(attn_block_bufs, uint64_t(T) * H);   // full ffn on every card

        for (uint32_t c = 0; c < n_dev_; ++c) {
            Workspace& w = ws_[c];
            residual_add(fleet_->dev(c).queue(), w.x, w.attn_block, w.x, uint64_t(T) * H);
        }
    }

    // final norm + lm_head on card 0 (x is replicated → card 0 holds the full hidden).
    {
        Workspace& w = ws_[0];
        auto& q = fleet_->dev(0).queue();
        if (all_logits_dev) {   // 2026-10-28 (P2): a lookup round's rows -- every row's logits through the batched SoA int-dot head
            if (!ls_.q8_rows || T > ls_.rows) return "all-rows logits: a round of " + std::to_string(T) + " rows, init for " + std::to_string(ls_.rows);
            rms_norm_f32w(q, w.x, output_norm_, w.x_normed, T, H, eps);
            quantize_q8_1(q, w.x_normed, ls_.q8_rows, T * H);   // row t's blocks land at t * (H / 32)
            gemv_q6_soa_q8_batched(q, ls_.q8_rows, static_cast<const uint8_t*>(out_soa_q6_lo_), static_cast<const uint8_t*>(out_soa_q6_hi_),
                                   static_cast<const int8_t*>(out_soa_q6_sc_), static_cast<const uint16_t*>(out_soa_q6_d_),
                                   all_logits_dev, H, cfg_.vocab, T);
            if (out_logits_host) q.memcpy(out_logits_host, all_logits_dev + uint64_t(T - 1) * cfg_.vocab, uint64_t(cfg_.vocab) * sizeof(sycl::half)).wait();
            else q.wait();
            return {};
        }
        void* q8 = dec ? w.q8 : nullptr;
        if (q8) rms_norm_f32w_q8(q, w.x, output_norm_, w.x_normed, q8, 1, H, eps);
        else    rms_norm_f32w(q, w.x, output_norm_, w.x_normed, T, H, eps);
        const sycl::half* last = w.x_normed + uint64_t(T - 1) * H;
        DenseQuantPtr lm{output_, output_dtype_};
        // the lm_head at decode: DenseModel's chain -- the SoA int-dot kernel over the load's repack, else the staged int-dot
        // Q6_K / Q4_K kernels, else the fp16 GEMV (which the prefill's last row always takes)
        if (q8 && cfg_.vocab >= 32768 && lm.dt == DType::kQ6_K && dense::q6k_soa_v2_enabled() && out_soa_q6_lo_)
            gemv_q6_soa_q8_v2(q, q8, static_cast<const uint8_t*>(out_soa_q6_lo_), static_cast<const uint8_t*>(out_soa_q6_hi_),
                              static_cast<const int8_t*>(out_soa_q6_sc_), static_cast<const uint16_t*>(out_soa_q6_d_),
                              d_logits_, H, cfg_.vocab);
        else if (q8 && cfg_.vocab >= 32768 && lm.dt == DType::kQ6_K && dense::q6k_slm_gemv_q8_fits(H))
            gemv_q6_K_q8(q, q8, lm.p, d_logits_, H, cfg_.vocab);
        else if (q8 && cfg_.vocab >= 32768 && lm.dt == DType::kQ4_K)
            gemv_q4_K_q8(q, q8, lm.p, d_logits_, H, cfg_.vocab);
        else
            dense::gemv_q(q, last, lm, d_logits_, H, cfg_.vocab);   // (2026-10-28 P1: the load's scratch)
        q.memcpy(out_logits_host, d_logits_, uint64_t(cfg_.vocab) * sizeof(sycl::half)).wait();   // the step's one host wait
    }
    return {};
}

// ===========================================================================
// 2026-10-28 (serving campaign P2): prompt-lookup drafts at --parallel 1 (the header's note). The decisions are
// DenseLookupPolicy's (dense_lookup.hpp, host-tested); this is the device side: the round's forward with every row's
// logits kept, a served row's copy out, and the KV lengths back for rows not followed.
// ===========================================================================
std::string DenseModelTP::look_step_init(const DenseLookupParams& p) {
    if (ls_.pol) return {};
    if (!fleet_ || n_dev_ == 0 || kv_.empty()) return "look_step: not loaded";
    if (output_dtype_ != DType::kQ6_K || !out_soa_q6_lo_ || !dense::q6k_soa_v2_enabled())
        return "lookup needs a Q6_K lm_head with its SoA repack (IE_Q6K_SOA_V2 on)";
    if (!dense::q8_decode_enabled() || ws_.empty() || !ws_[0].q8) return "lookup needs the Q8 decode path (IE_NO_Q8_DECODE unset)";
    DenseLookupParams pp = p;
    pp.rows = std::min<uint32_t>(std::max<uint32_t>(pp.rows, 8u), kLookRowsMax);   // the batched head takes 16 rows
    pp.strong_rows = std::min(pp.strong_rows, pp.rows);
    auto& alloc = fleet_->dev(0);
    const uint32_t H = cfg_.hidden, V = cfg_.vocab;
    ls_.d_all   = static_cast<sycl::half*>(alloc.malloc(uint64_t(pp.rows) * V * sizeof(sycl::half)));
    ls_.q8_rows = alloc.malloc(uint64_t(pp.rows) * (H / 32) * sizeof(block_q8_1x));
    if (!ls_.d_all || !ls_.q8_rows) return "look_step: scratch alloc failed";
    owned_[0].push_back(ls_.d_all);
    owned_[0].push_back(ls_.q8_rows);
    if (auto m = ensure_workspace(pp.rows); !m.empty()) return m;
    ls_.rows = pp.rows;
    ls_.pol = std::make_unique<DenseLookupPolicy>(pp);
    return {};
}

void DenseModelTP::look_step_context(const int32_t* ids, uint32_t n) {
    if (ls_.pol) ls_.pol->context(std::vector<int32_t>(ids, ids + n));
}

void DenseModelTP::look_step_reset() {
    if (!ls_.pol) return;
    static const bool stats = std::getenv("IE_DENSE_LOOKUP_STATS") != nullptr;
    const DenseLookupPolicy::Stats& s = ls_.pol->stats();
    if (stats && (s.rounds || s.plain))
        std::fprintf(stderr, "[dense lookup] rounds=%llu drafted=%llu accepted=%llu rounds-not-followed=%llu plain=%llu "
                     "(%.3f tokens a forward); ms: rounds %.1f, plain %.1f\n",
                     (unsigned long long)s.rounds, (unsigned long long)s.drafted, (unsigned long long)s.hits,
                     (unsigned long long)s.misses, (unsigned long long)s.plain,
                     double(s.rounds + s.hits + s.plain) / double(std::max<uint64_t>(1, s.rounds + s.plain)),
                     ls_.ms_round, ls_.ms_plain);
    ls_.pol->reset();
    ls_.ms_round = ls_.ms_plain = 0;
}

void DenseModelTP::rollback_to(uint32_t pos) {
    for (uint32_t d = 0; d < n_dev_; ++d)
        for (uint32_t l = 0; l < cfg_.n_layers; ++l) kv_[d].set_length(l, pos);
}

std::string DenseModelTP::look_step_flush() {
    if (!ls_.pol) return {};
    uint32_t to = 0;
    if (ls_.pol->flush(to)) rollback_to(to);
    return {};
}

std::string DenseModelTP::look_step(int32_t token, uint32_t pos, sycl::half* logits_host) {
    if (!ls_.pol) return "look_step: not initialised";
    const uint64_t row_bytes = uint64_t(cfg_.vocab) * sizeof(sycl::half);
    const DenseLookupPolicy::Decision d = ls_.pol->step(token, pos, max_ctx());
    if (d.rollback) rollback_to(d.rollback_to);   // the open round's rows the sampler did not follow
    switch (d.action) {
    case DenseLookupPolicy::Action::kServeRow:      // this position's logits are a row of the open round
        fleet_->dev(0).d2h(logits_host, ls_.d_all + uint64_t(d.row) * cfg_.vocab, row_bytes);
        return {};
    case DenseLookupPolicy::Action::kPlain: {
        const auto t0 = std::chrono::steady_clock::now();
        const std::string e = forward(&token, 1, pos, /*reset_kv=*/pos == 0, logits_host);
        ls_.ms_plain += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        return e;
    }
    case DenseLookupPolicy::Action::kRound: {
        const auto t0 = std::chrono::steady_clock::now();
        const uint32_t Vr = uint32_t(d.vin.size());
        if (const std::string e = forward(d.vin.data(), Vr, pos, /*reset_kv=*/false, nullptr, ls_.d_all); !e.empty()) {
            rollback_to(pos);   // whatever the failed round wrote is not read
            return "look_step round: " + e;
        }
        fleet_->dev(0).d2h(logits_host, ls_.d_all, row_bytes);   // row 0 = this position
        ls_.pol->round_started(pos, d.vin);
        ls_.ms_round += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        return {};
    }
    }
    return {};
}

}  // namespace ie
