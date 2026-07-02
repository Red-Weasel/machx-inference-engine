// src/model/qwen3moe_tp.cpp — Qwen3 standard MoE (`kQwen3Moe`) TENSOR-PARALLEL.
// See header. ADDITIVE: reuses Qwen3MoeSplitModel's per-card int-dot MoE machinery
// (moe_prefill_gate_up_silu_q{4,6}k_q8 → *_down_q{4,6}k_q8_gen → moe_prefill_reduce)
// and GptOssTpModel's host-bounce all-reduce skeleton, with:
//   * ATTENTION head-sharded (q/k/v column-sliced quantized, o-proj row-sliced fp16,
//     per-head QK-norm + rope over the card's heads, HALF KV/card) → PARTIAL [T,H]
//     → all-reduce #1. Standard Qwen3 GQA — NO sinks/SWA/biases.
//   * MoE expert-sharded (card owns experts [ef0_e, ef0_e+efn); router replicated;
//     the int-dot kernels run over ONLY the card's experts via a sliced
//     expert_offsets base + local expert count + the card's expert banks;
//     out_packed memset → off-card packed rows stay 0 → PARTIAL) → all-reduce #2.
// The single-GPU Qwen3MoeModel and the layer-split Qwen3MoeSplitModel are NEVER
// edited (their PPL gates stay put). NOT bit-exact vs single-GPU (reduction order).

#include "ie/qwen3moe_tp.hpp"

#include "ie/dequant.hpp"            // dequant_q{4,5,6}_K_to_Bt, dequant_q8_0_to_Bt
#include "ie/moe_qwen3.hpp"         // moe_prefill_gate_up_silu_q6k_q8, *_down_q{4,6}k_q8_gen
#include "ie/ops.hpp"               // rms_norm, rope_partial, FA kernels, moe_gather_rows,
                                    // quantize_q8_1s, moe_prefill_gate_up_silu_q4k_q8, moe_prefill_reduce
#include "ie/qwen3moe_pack.hpp"     // MoePacking, build_moe_packing
#include "ie/quant_blocks.hpp"      // block_q4_K, block_q6_K, block_q8_1s
#include "ie/quant_soa.hpp"         // repack_moe_q{4,6}k_soa_host

#include "dense_dispatch.hpp"       // dense::upload<T>, dense::gemv_q(_T), dense::prefer_onednn

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

// COPY of qwen3moe.cpp/qwen3moe_split.cpp:route_from_logits (copy-not-hoist).
// Softmax-after-topk over precomputed logits[E]; renormalize the top-k; sort
// ascending by expert id. Matches the single-GPU DEFAULT router numerics.
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

// EXPERT-SHARD bank slice: keep experts [ef0_e, ef0_e+efn) of a stacked expert
// bank (Q4_K AoS / Q6_K per-expert-SoA). Experts are stored contiguously with a
// per-expert byte stride = nbytes/E, so the card's slice is a contiguous byte
// range [ef0_e*stride, (ef0_e+efn)*stride). For Q6_K the SoA repack is intra-expert
// (stride unchanged), so we repack exactly the efn sliced experts from the source
// slice base. The kernel indexes the returned bank by the card-LOCAL expert index
// (0..efn-1) → the sliced expert_offsets base aligns global packed rows to it.
std::string upload_expert_bank_slice(DeviceAllocator& alloc, const GgufTensorInfo* ti,
                                     uint32_t E_total, uint32_t ef0_e, uint32_t efn,
                                     bool moe_soa, std::vector<uint8_t>& staging,
                                     std::vector<void*>& own, void*& dst,
                                     DType& dt, uint64_t& stride, bool& soa) {
    if (!ti) return "expert bank: not found";
    if (ti->dtype != DType::kQ4_K && ti->dtype != DType::kQ6_K)
        return "expert bank: unsupported dtype (Q4_K/Q6_K)";
    const size_t bs = (ti->dtype == DType::kQ4_K) ? sizeof(block_q4_K) : sizeof(block_q6_K);
    if (E_total == 0 || ti->nbytes % (uint64_t(E_total) * bs) != 0)
        return "expert bank: unexpected geometry";
    if (uint64_t(ef0_e) + efn > E_total) return "expert bank: bad expert slice";
    const uint64_t nb_e   = ti->nbytes / E_total / bs;     // Q-blocks per expert
    stride                = ti->nbytes / E_total;          // per-expert byte stride (unchanged)
    const uint64_t slice  = uint64_t(efn) * stride;
    const uint8_t* src    = static_cast<const uint8_t*>(ti->data) + uint64_t(ef0_e) * stride;
    soa = moe_soa && ti->dtype == DType::kQ6_K;
    void* d = alloc.malloc(slice);
    if (!d) return "expert bank: malloc failed";
    own.push_back(d);
    if (soa) {
        staging.resize(slice);
        repack_moe_q6k_soa_host(src, staging.data(), efn, nb_e);   // intra-expert reorder
        alloc.queue().memcpy(d, staging.data(), slice).wait();
    } else {
        alloc.queue().memcpy(d, src, slice).wait();               // AoS: contiguous byte copy
    }
    dst = d; dt = ti->dtype;
    return {};
}

// Attention q/k/v COLUMN slice (head-shard). The weight [K=H, N] is stored
// output-major (N output cols, each K/blocksize blocks), so keeping output cols
// [n0, n0+nc) is a CONTIGUOUS byte range regardless of quant → the PACKED slice is a
// plain memcpy. Q4_K/Q6_K are dense::gemv_q's NATIVE dtypes → return the sliced bank
// as-is (fast path). Q5_K/Q8_0 are NOT native (gemv_q dequants them elsewhere, via
// upload_weight_auto) → dequant the sliced [K,nc] to fp16 Bt (dequant only the head-
// slice, not the full weight). F16 → host transpose-slice. dense::gemv_q_T consumes
// fp16 identically (dt=kF16). Robust to any attn q/k/v dtype in the target GGUFs
// (bartowski Q6_K mixes Q8_0/Q5_K attn tensors; Coder Q4_K_M mixes Q4_K/Q6_K).
DenseQuantPtr upload_attn_col_quant(DeviceAllocator& a, const GgufTensorInfo* ti,
                                    uint32_t n0, uint32_t nc, std::vector<void*>& own,
                                    std::string& err, uint64_t& bytes) {
    DenseQuantPtr out;
    if (!ti) { err = "attn col: not found"; return out; }
    if (ti->n_dims != 2) { err = "attn col: expected 2-D"; return out; }
    const uint32_t K = uint32_t(ti->shape[0]), N = uint32_t(ti->shape[1]);
    if (uint64_t(n0) + nc > N) { err = "attn col: bad slice"; return out; }

    // F16: on-disk [N,K] out[n*K+k] → sliced Bt [K,nc] out[k*nc+j] = src[(n0+j)*K+k].
    if (ti->dtype == DType::kF16) {
        std::vector<sycl::half> hb(uint64_t(K) * nc);
        const auto* src = reinterpret_cast<const sycl::half*>(ti->data);
        for (uint32_t j = 0; j < nc; ++j)
            for (uint32_t k = 0; k < K; ++k) hb[uint64_t(k) * nc + j] = src[uint64_t(n0 + j) * K + k];
        void* d = a.malloc(hb.size() * sizeof(sycl::half));
        if (!d) { err = "attn col: F16 alloc"; return out; }
        own.push_back(d);
        a.queue().memcpy(d, hb.data(), hb.size() * sizeof(sycl::half)).wait();
        bytes += hb.size() * sizeof(sycl::half);
        out.p = d; out.dt = DType::kF16; return out;
    }

    // Quantized output-major: (elems/block, block bytes) for the contiguous byte-slice.
    uint32_t epb = 0; size_t blkb = 0;
    switch (ti->dtype) {
        case DType::kQ4_K: epb = 256; blkb = sizeof(block_q4_K); break;
        case DType::kQ6_K: epb = 256; blkb = sizeof(block_q6_K); break;
        case DType::kQ5_K: epb = 256; blkb = sizeof(block_q5_K); break;
        case DType::kQ8_0: epb = 32;  blkb = sizeof(block_q8_0); break;
        default: err = "attn col: dtype unsupported (Q4_K/Q6_K/Q5_K/Q8_0/F16)"; return out;
    }
    if (K % epb != 0) { err = "attn col: K not block-aligned"; return out; }
    const uint64_t row_bytes = uint64_t(K / epb) * blkb;   // bytes per output col
    if (ti->nbytes != uint64_t(N) * row_bytes) { err = "attn col: size mismatch"; return out; }
    const uint64_t off = uint64_t(n0) * row_bytes, slice = uint64_t(nc) * row_bytes;

    // NATIVE fast path (gemv_q reads Q4_K/Q6_K directly): keep the sliced quantized bank.
    if (ti->dtype == DType::kQ4_K || ti->dtype == DType::kQ6_K) {
        void* d = a.malloc(slice);
        if (!d) { err = "attn col: alloc"; return out; }
        own.push_back(d);
        a.queue().memcpy(d, static_cast<const uint8_t*>(ti->data) + off, slice).wait();
        bytes += slice;
        out.p = d; out.dt = ti->dtype; return out;
    }

    // Q5_K / Q8_0 (non-native): byte-slice the packed weight → dequant the slice → fp16
    // Bt [K,nc]. dequant_q{5_K,8_0}_to_Bt read the packed as nc output cols × K/blk
    // blocks → out[k*nc+j], exactly the sliced Bt gemv_q_T (F16) wants.
    void* packed = a.malloc(slice);
    if (!packed) { err = "attn col: packed alloc"; return out; }
    a.queue().memcpy(packed, static_cast<const uint8_t*>(ti->data) + off, slice).wait();
    auto* Bt = static_cast<sycl::half*>(a.malloc(uint64_t(K) * nc * sizeof(sycl::half)));
    if (!Bt) { a.free(packed); err = "attn col: Bt alloc"; return out; }
    sycl::event ev;
    if (ti->dtype == DType::kQ5_K) {
        if (nc % 64) { a.free(packed); a.free(Bt); err = "attn col: Q5_K slice width not %64"; return out; }
        ev = dequant_q5_K_to_Bt(a.queue(), packed, Bt, K, nc);
    } else {   // Q8_0 (K % 32 == 0 already ensured above)
        ev = dequant_q8_0_to_Bt(a.queue(), packed, Bt, K, nc);
    }
    ev.wait();
    a.free(packed);
    own.push_back(Bt);
    bytes += uint64_t(K) * nc * sizeof(sycl::half);
    out.p = Bt; out.dt = DType::kF16; return out;
}

// Attention o-proj ROW slice (contraction split, head-shard). The full weight
// [K=N_q, N=H] is dequantized to fp16 Bt [K,N] (out[k*N+n]); keeping input rows
// [k0, k0+kc) is then a CONTIGUOUS device copy [k0*N, (k0+kc)*N) → the card's
// [kc, H] fp16 slice → dense::gemv_q_T (F16) → PARTIAL [T,H] the all-reduce sums.
DenseQuantPtr upload_attn_output_row_f16(DeviceAllocator& a, const GgufTensorInfo* ti,
                                         uint32_t k0, uint32_t kc, std::vector<void*>& own,
                                         std::string& err, uint64_t& bytes) {
    DenseQuantPtr out;
    if (!ti) { err = "attn o row: not found"; return out; }
    if (ti->n_dims != 2) { err = "attn o row: expected 2-D"; return out; }
    const uint32_t K = uint32_t(ti->shape[0]), N = uint32_t(ti->shape[1]);
    if (uint64_t(k0) + kc > K) { err = "attn o row: bad slice"; return out; }
    // upload packed → dequant full → Bt [K,N] fp16 → row-slice contiguous.
    void* packed = a.malloc(ti->nbytes);
    if (!packed) { err = "attn o row: packed alloc"; return out; }
    a.queue().memcpy(packed, ti->data, ti->nbytes).wait();
    auto* Bt = static_cast<sycl::half*>(a.malloc(uint64_t(K) * N * sizeof(sycl::half)));
    if (!Bt) { a.free(packed); err = "attn o row: Bt alloc"; return out; }
    sycl::event ev;
    if (ti->dtype == DType::kQ4_K) {           // K%256==0, N%64==0
        if (K % 256 || N % 64) { a.free(packed); a.free(Bt); err = "attn o row: Q4_K dims"; return out; }
        ev = dequant_q4_K_to_Bt(a.queue(), packed, Bt, K, N);
    } else if (ti->dtype == DType::kQ6_K) {
        if (K % 256 || N % 64) { a.free(packed); a.free(Bt); err = "attn o row: Q6_K dims"; return out; }
        ev = dequant_q6_K_to_Bt(a.queue(), packed, Bt, K, N);
    } else if (ti->dtype == DType::kQ5_K) {
        if (K % 256 || N % 64) { a.free(packed); a.free(Bt); err = "attn o row: Q5_K dims"; return out; }
        ev = dequant_q5_K_to_Bt(a.queue(), packed, Bt, K, N);
    } else if (ti->dtype == DType::kQ8_0) {
        if (K % 32) { a.free(packed); a.free(Bt); err = "attn o row: Q8_0 dims"; return out; }
        ev = dequant_q8_0_to_Bt(a.queue(), packed, Bt, K, N);
    } else if (ti->dtype == DType::kF16) {     // on-disk [N,K] out[n*K+k] → Bt out[k*N+n]
        std::vector<sycl::half> hb(uint64_t(K) * N);
        const auto* srch = reinterpret_cast<const sycl::half*>(ti->data);
        for (uint32_t n = 0; n < N; ++n)
            for (uint32_t k = 0; k < K; ++k) hb[uint64_t(k) * N + n] = srch[uint64_t(n) * K + k];
        a.queue().memcpy(Bt, hb.data(), hb.size() * sizeof(sycl::half)).wait();
    } else { a.free(packed); a.free(Bt); err = "attn o row: dtype unsupported"; return out; }
    if (ti->dtype != DType::kF16) ev.wait();
    a.free(packed);
    void* d = a.malloc(uint64_t(kc) * N * sizeof(sycl::half));
    if (!d) { a.free(Bt); err = "attn o row: slice alloc"; return out; }
    own.push_back(d);
    a.queue().memcpy(d, Bt + uint64_t(k0) * N, uint64_t(kc) * N * sizeof(sycl::half)).wait();
    a.free(Bt);
    bytes += uint64_t(kc) * N * sizeof(sycl::half);
    out.p = d; out.dt = DType::kF16; return out;
}

}  // namespace

Qwen3MoeTpModel::~Qwen3MoeTpModel() { free_all(); }

void Qwen3MoeTpModel::free_all() {
    if (!fleet_) return;
    for (uint32_t d = 0; d < ws_.size(); ++d) free_ws(d);
    ws_.clear();
    for (uint32_t d = 0; d < owned_.size(); ++d)
        for (void* p : owned_[d]) if (p) fleet_->dev(d).free(p);
    owned_.clear();
    for (auto& c : kv_) c.free_storage();
    kv_.clear();
}

std::string Qwen3MoeTpModel::load(DeviceFleet& fleet, const GgufReader& g,
                                  const Qwen3MoeConfig& cfg, uint32_t max_ctx) {
    fleet_ = &fleet; cfg_ = cfg;
    n_dev_ = fleet.size();
    if (n_dev_ == 0) return "qwen3moe-tp: empty fleet";

    // oneDNN OFF: the int-dot MoE never uses it and attention rides dense::gemv_q_T;
    // multi-card oneDNN is a DEVICE_LOST landmine unless routed per-device.
    dense::prefer_onednn() = false;

    const auto& d = cfg_.dense;
    const uint32_t H = d.hidden, E = cfg_.n_experts, EF = cfg_.expert_ffn, HD = d.head_dim;
    if (H == 0 || d.vocab == 0 || d.n_q_heads == 0 || d.n_kv_heads == 0 || EF == 0 || E == 0)
        return "qwen3moe-tp: zero dim";
    // int-dot MoE preconditions (fail loudly, like the layer-split).
    if (H % 512 != 0)  return "qwen3moe-tp: hidden % 512 != 0 (int-dot MoE unsupported)";
    if (EF % 256 != 0) return "qwen3moe-tp: expert_ffn % 256 != 0 (int-dot MoE unsupported)";
    // Head-shard needs q/kv heads divisible by the fleet size (the GQA ratio nq/nkv
    // is then preserved = the global ratio → correct grouping on each card).
    if (d.n_q_heads % n_dev_ || d.n_kv_heads % n_dev_)
        return "qwen3moe-tp: n_q_heads / n_kv_heads not divisible by fleet size (head-shard)";

    // Per-card shards: contiguous expert range (even split + remainder) + head range.
    shard_.assign(n_dev_, {});
    {
        uint32_t acc = 0;
        for (uint32_t c = 0; c < n_dev_; ++c) {
            const uint32_t my_e = E / n_dev_ + (c < (E % n_dev_) ? 1u : 0u);
            shard_[c].ef0_e = acc; shard_[c].efn = my_e; acc += my_e;
        }
        if (acc != E) return "qwen3moe-tp: expert split did not cover E";
    }
    for (uint32_t c = 0; c < n_dev_; ++c) {
        Shard& s = shard_[c];
        const uint32_t nqc = d.n_q_heads / n_dev_, nkvc = d.n_kv_heads / n_dev_;
        s.q_head0 = c * nqc; s.nq = nqc; s.kv_head0 = c * nkvc; s.nkv = nkvc;
        s.Nq = s.nq * HD; s.Nkv = s.nkv * HD;
    }

    owned_.assign(n_dev_, {});
    dev_bytes_.assign(n_dev_, 0);

    // Global SoA decision (Q6_K experts → per-expert SoA; Q4_K stays AoS). Opt-out
    // IE_NO_MOE_SOA (mirrors the layer-split).
    const auto* gate0 = g.find_tensor("blk.0.ffn_gate_exps.weight");
    const auto* up0   = g.find_tensor("blk.0.ffn_up_exps.weight");
    const bool gateup_soa_ok =
        (!gate0 || gate0->dtype == DType::kQ4_K || gate0->dtype == DType::kQ6_K) &&
        (!up0   || up0->dtype   == DType::kQ4_K || up0->dtype   == DType::kQ6_K);
    const bool moe_soa = std::getenv("IE_NO_MOE_SOA") == nullptr && gateup_soa_ok;

    char buf[96];
    auto Ttop = [&](const char* n) { return g.find_tensor(n); };
    auto Tl   = [&](uint32_t L, const char* n) {
        std::snprintf(buf, sizeof(buf), "blk.%u.%s", L, n); return g.find_tensor(buf);
    };
    std::string err;

    // ---- globals on card 0 (embedding + final norm + lm_head; x replicated) ----
    {
        DeviceAllocator& ea = fleet.dev(0);
        const auto* ti = Ttop("token_embd.weight");
        if (!ti) return "token_embd: not found";
        if (ti->dtype != DType::kQ4_K && ti->dtype != DType::kQ6_K && ti->dtype != DType::kQ8_0)
            return std::string("token_embd: unsupported dtype ") +
                   std::string(type_name(ti->dtype)) + " (need Q4_K/Q6_K/Q8_0)";
        token_embd_dtype_ = ti->dtype;
        token_embd_ = dense::upload<void>(ea, ti, owned_[0], err, ti->dtype);
        if (!err.empty()) return "token_embd: " + err;
        dev_bytes_[0] += ti->nbytes;

        output_norm_ = dense::upload<float>(ea, Ttop("output_norm.weight"), owned_[0], err, DType::kF32);
        if (!err.empty()) return "output_norm: " + err;

        const auto* lo = Ttop("output.weight");
        if (!lo) { output_ = token_embd_; output_dtype_ = token_embd_dtype_; }   // tied
        else {
            if (lo->dtype != DType::kQ4_K && lo->dtype != DType::kQ6_K)
                return "output: unsupported dtype (need Q4_K/Q6_K)";
            output_dtype_ = lo->dtype;
            output_ = dense::upload<void>(ea, lo, owned_[0], err, lo->dtype);
            if (!err.empty()) return "output: " + err;
            dev_bytes_[0] += lo->nbytes;
        }
    }

    const uint32_t n_layers = d.n_layers;
    dlayers_.assign(n_dev_, std::vector<LayerW>(n_layers));
    std::vector<std::vector<uint8_t>> soa_staging(n_dev_);   // per-card reuse buffer

    for (uint32_t c = 0; c < n_dev_; ++c) {
        DeviceAllocator& a = fleet.dev(c);
        auto& own = owned_[c];
        const Shard& s = shard_[c];
        for (uint32_t L = 0; L < n_layers; ++L) {
            LayerW& w = dlayers_[c][L];
            auto lerr = [&](const std::string& m) {
                return "card " + std::to_string(c) + " L" + std::to_string(L) + " " + m;
            };
            auto F32 = [&](const char* nm, float*& dst, bool required) -> std::string {
                const auto* ti = Tl(L, nm);
                if (!ti) { if (required) return std::string(nm) + ": not found"; dst = nullptr; return {}; }
                dst = dense::upload<float>(a, ti, own, err, DType::kF32);
                if (!err.empty()) return std::string(nm) + ": " + err;
                dev_bytes_[c] += ti->nbytes;
                return {};
            };

            // norms + QK-norm (all replicated full).
            if (auto m = F32("attn_norm.weight", w.attn_norm, true);  !m.empty()) return lerr(m);
            if (auto m = F32("ffn_norm.weight",  w.ffn_norm,  true);  !m.empty()) return lerr(m);
            if (auto m = F32("attn_q_norm.weight", w.attn_q_norm, false); !m.empty()) return lerr(m);
            if (auto m = F32("attn_k_norm.weight", w.attn_k_norm, false); !m.empty()) return lerr(m);

            // attention head-shard: q/k/v COLUMN-slice (quantized), o-proj ROW-slice (fp16).
            w.attn_q = upload_attn_col_quant(a, Tl(L, "attn_q.weight"), s.q_head0 * HD, s.Nq, own, err, dev_bytes_[c]);
            if (!err.empty()) return lerr("attn_q: " + err);
            w.attn_k = upload_attn_col_quant(a, Tl(L, "attn_k.weight"), s.kv_head0 * HD, s.Nkv, own, err, dev_bytes_[c]);
            if (!err.empty()) return lerr("attn_k: " + err);
            w.attn_v = upload_attn_col_quant(a, Tl(L, "attn_v.weight"), s.kv_head0 * HD, s.Nkv, own, err, dev_bytes_[c]);
            if (!err.empty()) return lerr("attn_v: " + err);
            w.attn_output = upload_attn_output_row_f16(a, Tl(L, "attn_output.weight"), s.q_head0 * HD, s.Nq, own, err, dev_bytes_[c]);
            if (!err.empty()) return lerr("attn_output: " + err);

            // MoE router — replicated full: F32 [E,H] → device F16 [H,E] transposed.
            {
                const auto* ti = Tl(L, "ffn_gate_inp.weight");
                if (!ti) return lerr("ffn_gate_inp: not found");
                if (ti->dtype != DType::kF32) return lerr("ffn_gate_inp: expected F32");
                if (ti->nbytes != uint64_t(E) * H * sizeof(float)) return lerr("ffn_gate_inp: size");
                const float* rw = reinterpret_cast<const float*>(ti->data);
                std::vector<sycl::half> rt(uint64_t(H) * E);
                for (uint32_t e = 0; e < E; ++e)
                    for (uint32_t h = 0; h < H; ++h)
                        rt[uint64_t(h) * E + e] = sycl::half(rw[uint64_t(e) * H + h]);
                auto* rd = static_cast<sycl::half*>(a.malloc(uint64_t(H) * E * sizeof(sycl::half)));
                if (!rd) return lerr("router_w_dev alloc");
                a.queue().memcpy(rd, rt.data(), uint64_t(H) * E * sizeof(sycl::half)).wait();
                own.push_back(rd); w.router_w_dev = rd;
                dev_bytes_[c] += uint64_t(H) * E * sizeof(sycl::half);
            }

            // MoE experts — the card's [ef0_e, ef0_e+efn) slice.
            if (auto m = upload_expert_bank_slice(a, Tl(L, "ffn_gate_exps.weight"), E, s.ef0_e, s.efn,
                                                  moe_soa, soa_staging[c], own, w.gate_exps,
                                                  w.gate_dt, w.gate_stride, w.gate_soa); !m.empty())
                return lerr("gate exps: " + m);
            dev_bytes_[c] += uint64_t(s.efn) * w.gate_stride;
            if (auto m = upload_expert_bank_slice(a, Tl(L, "ffn_up_exps.weight"), E, s.ef0_e, s.efn,
                                                  moe_soa, soa_staging[c], own, w.up_exps,
                                                  w.up_dt, w.up_stride, w.up_soa); !m.empty())
                return lerr("up exps: " + m);
            dev_bytes_[c] += uint64_t(s.efn) * w.up_stride;
            if (auto m = upload_expert_bank_slice(a, Tl(L, "ffn_down_exps.weight"), E, s.ef0_e, s.efn,
                                                  moe_soa, soa_staging[c], own, w.down_exps,
                                                  w.down_dt, w.down_stride, w.down_soa); !m.empty())
                return lerr("down exps: " + m);
            dev_bytes_[c] += uint64_t(s.efn) * w.down_stride;
            if (w.gate_dt != w.up_dt) return lerr("gate/up expert dtype mismatch (unsupported)");
        }
    }

    // Per-card KV: HEAD-SHARDED (nkv kv-heads/card → HALF KV/card). Every layer
    // full-attn (qwen3moe has no DeltaNet); TP → all layers live on each card.
    kv_.resize(n_dev_);
    for (uint32_t c = 0; c < n_dev_; ++c) {
        KvCacheConfig kc{};
        kc.n_layers_full = n_layers;
        kc.n_kv_heads    = shard_[c].nkv;   // HALVED
        kc.max_ctx       = max_ctx;
        kc.head_dim      = HD;
        if (auto m = kv_[c].init(fleet.dev(c), kc); !m.empty())
            return "kv dev " + std::to_string(c) + ": " + m;
    }

    ws_.assign(n_dev_, {});
    for (uint32_t c = 0; c < n_dev_; ++c)
        std::fprintf(stderr, "[qwen3moe-tp] card %u weights: %.2f GB (experts %u..%u, "
                     "attn q-heads %u..%u kv-heads %u..%u)\n",
                     c, double(dev_bytes_[c]) / 1e9, shard_[c].ef0_e, shard_[c].ef0_e + shard_[c].efn,
                     shard_[c].q_head0, shard_[c].q_head0 + shard_[c].nq,
                     shard_[c].kv_head0, shard_[c].kv_head0 + shard_[c].nkv);
    return {};
}

void Qwen3MoeTpModel::free_ws(uint32_t dev) {
    if (!fleet_ || dev >= ws_.size()) return;
    Workspace& w = ws_[dev];
    auto& a = fleet_->dev(dev);
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
        if (p) a.free(p);
    w = Workspace{};
}

std::string Qwen3MoeTpModel::ensure_ws(uint32_t dev, uint32_t max_T) {
    Workspace& w = ws_[dev];
    if (max_T <= w.T) return {};
    if (w.T != 0) free_ws(dev);
    auto& a = fleet_->dev(dev);
    auto ah = [&](uint64_t n) { return static_cast<sycl::half*>(a.malloc(n * sizeof(sycl::half))); };

    const auto& d = cfg_.dense;
    const uint32_t H = d.hidden, HD = d.head_dim;
    const uint32_t N_q = shard_[dev].Nq, N_kv = shard_[dev].Nkv;   // sharded heads
    const uint32_t nq  = shard_[dev].nq;
    const uint32_t EF = cfg_.expert_ffn, E = cfg_.n_experts, K = cfg_.n_experts_used;
    const uint64_t T = max_T, TK = uint64_t(max_T) * K;

    w.x = ah(T * H); w.x_normed = ah(T * H);
    w.q = ah(T * N_q); w.k = ah(T * N_kv); w.v = ah(T * N_kv);
    w.attn_out = ah(T * N_q); w.attn_block = ah(T * H);
    w.positions = static_cast<int32_t*>(a.malloc(T * sizeof(int32_t)));
    w.router_logits = ah(T * E);
    w.xp_packed  = ah(TK * H);
    w.xp_q8      = a.malloc(TK * (H / 32) * sizeof(block_q8_1s));
    w.h_packed   = ah(TK * EF);
    w.h_q8       = a.malloc(TK * (EF / 32) * sizeof(block_q8_1s));
    w.out_packed = ah(TK * H);
    w.ffn_y      = ah(T * H);
    w.expert_offsets = static_cast<uint32_t*>(a.malloc((uint64_t(E) + 1) * sizeof(uint32_t)));
    w.sorted_idx     = static_cast<int32_t*>(a.malloc(TK * sizeof(int32_t)));
    w.tk_to_packed   = static_cast<int32_t*>(a.malloc(TK * sizeof(int32_t)));
    w.weights_packed = ah(TK);

    if (!w.x || !w.x_normed || !w.q || !w.k || !w.v || !w.attn_out || !w.attn_block ||
        !w.positions || !w.router_logits || !w.xp_packed || !w.xp_q8 || !w.h_packed ||
        !w.h_q8 || !w.out_packed || !w.ffn_y || !w.expert_offsets || !w.sorted_idx ||
        !w.tk_to_packed || !w.weights_packed)
        return "qwen3moe-tp workspace alloc failed on dev " + std::to_string(dev);

    // One-time zero of the MoE staging that carries OFF-card packed rows across
    // layers (this card only writes its experts' rows): h_packed off-card rows are
    // never written → keep them finite for the (never-read) quantize; out_packed is
    // ALSO memset every layer (partial-reduce correctness), this is belt-and-braces.
    a.queue().memset(w.h_packed,   0, TK * EF * sizeof(sycl::half));
    a.queue().memset(w.out_packed, 0, TK * H  * sizeof(sycl::half)).wait();

    // FA-2 split-K decode partials, sized for this card's KV ctx + SHARDED q-heads.
    if (kv_[dev].ready()) {
        const uint32_t mc = kv_[dev].config().max_ctx;
        constexpr uint32_t Bc_floor = 64;
        const uint32_t n_chunks = (mc + Bc_floor - 1) / Bc_floor;
        const uint64_t n_floats = uint64_t(n_chunks) * nq * (HD + 2);
        w.attn_partials = static_cast<float*>(a.malloc(n_floats * sizeof(float)));
        if (!w.attn_partials) return "qwen3moe-tp attn_partials alloc failed on dev " + std::to_string(dev);
        w.partials_ctx = mc;
    }
    w.T = max_T;
    return {};
}

// EXPERT-SHARD MoE FFN for one layer on one card → ws.ffn_y (PARTIAL). Router
// replicated (this card computes the FULL top-K packing from its synced x_normed,
// identical across cards); the int-dot kernels run over ONLY the card's experts.
void Qwen3MoeTpModel::moe_ffn_card(uint32_t dev, const LayerW& w, uint32_t T) {
    auto& q = fleet_->dev(dev).queue();
    Workspace& ws = ws_[dev];
    const auto& d = cfg_.dense;
    const uint32_t H = d.hidden, EF = cfg_.expert_ffn, E = cfg_.n_experts, K = cfg_.n_experts_used;
    const uint32_t TK = T * K;
    const uint32_t ef0_e = shard_[dev].ef0_e, efn = shard_[dev].efn;

    // 1) FULL router logits = x_normed @ router_wᵀ → [T,E] (replicated weight) →
    //    host softmax/top-K per token → FULL packing (identical on every card since
    //    x_normed is synced by the two per-layer all-reduces).
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
    MoePacking pk;
    build_moe_packing(routes, E, K, pk);
    q.memcpy(ws.expert_offsets, pk.expert_offsets.data(), (uint64_t(E) + 1) * sizeof(uint32_t));
    q.memcpy(ws.sorted_idx,     pk.sorted_idx.data(),     uint64_t(TK) * sizeof(int32_t));
    q.memcpy(ws.tk_to_packed,   pk.tk_to_packed.data(),   uint64_t(TK) * sizeof(int32_t));
    std::vector<sycl::half> wpk(TK);
    for (uint32_t i = 0; i < TK; ++i) wpk[i] = sycl::half(pk.weights_packed[i]);
    q.memcpy(ws.weights_packed, wpk.data(), uint64_t(TK) * sizeof(sycl::half)).wait();

    // 2) gather the FULL expert-sorted rows (off-card rows are harmless — the card's
    //    experts form a contiguous packed sub-range that its kernels alone touch).
    moe_gather_rows(q, ws.x_normed, ws.sorted_idx, ws.xp_packed, TK, H);

    // 3) stage 1: gate+up+silu over ONLY the card's experts. E_param = efn; the
    //    expert_offsets base is shifted by ef0_e so kernel-local expert e ∈ [0,efn)
    //    reads global packed rows for global expert (ef0_e+e), aligning the card's
    //    LOCAL expert banks (indexed 0..efn-1) with the global packed layout.
    quantize_q8_1s(q, ws.xp_packed, ws.xp_q8, uint64_t(TK) * H);
    if (w.gate_dt == DType::kQ6_K)
        moe_prefill_gate_up_silu_q6k_q8(q, ws.xp_q8, w.gate_exps, w.up_exps,
                                        ws.expert_offsets + ef0_e, ws.h_packed,
                                        efn, H, EF, w.gate_stride, w.gate_soa);
    else
        moe_prefill_gate_up_silu_q4k_q8(q, ws.xp_q8, w.gate_exps, w.up_exps,
                                        ws.expert_offsets + ef0_e, ws.h_packed,
                                        efn, H, EF, w.gate_stride, w.gate_soa);

    // 4) stage 2: down over the card's experts. CRITICAL: memset out_packed so the
    //    OFF-card packed rows (never written by this card) stay 0 → the weighted
    //    reduce adds w·0 for a token's off-card kslots → ffn_y is a clean PARTIAL.
    quantize_q8_1s(q, ws.h_packed, ws.h_q8, uint64_t(TK) * EF);
    q.memset(ws.out_packed, 0, uint64_t(TK) * H * sizeof(sycl::half));
    if (w.down_dt == DType::kQ6_K)
        moe_prefill_down_q6k_q8_gen(q, ws.h_q8, w.down_exps, ws.expert_offsets + ef0_e,
                                    ws.out_packed, efn, H, EF, w.down_stride, w.down_soa);
    else
        moe_prefill_down_q4k_q8_gen(q, ws.h_q8, w.down_exps, ws.expert_offsets + ef0_e,
                                    ws.out_packed, efn, H, EF, w.down_stride, w.down_soa);

    // 5) weighted reduce per token → PARTIAL ffn_y [T,H] (all-reduce #2 sums cards).
    //    ffn_y zeroed every layer (moe_prefill_reduce accumulates onto it).
    q.memset(ws.ffn_y, 0, uint64_t(T) * H * sizeof(sycl::half));
    moe_prefill_reduce(q, ws.out_packed,
                       reinterpret_cast<const uint32_t*>(ws.tk_to_packed),
                       ws.weights_packed, ws.ffn_y, T, K, H);
}

std::string Qwen3MoeTpModel::forward(const int32_t* input_ids, uint32_t T,
                                     uint32_t start_pos, bool reset_kv,
                                     sycl::half* out_logits_host) {
    if (T == 0) return "T == 0";
    const auto& d = cfg_.dense;
    const uint32_t H = d.hidden, HD = d.head_dim;
    const uint32_t E = cfg_.n_experts, V = d.vocab;
    const float eps = d.rms_eps;
    const uint32_t n_layers = d.n_layers;
    (void)E;

    static const bool fa2_decode_on  = std::getenv("IE_QWEN3MOE_TP_NO_FA2_DECODE")  == nullptr;
    static const bool fa2_prefill_on = std::getenv("IE_QWEN3MOE_TP_NO_FA2_PREFILL") == nullptr;

    if (ws_.size() != n_dev_) ws_.assign(n_dev_, {});
    for (uint32_t c = 0; c < n_dev_; ++c)
        if (auto m = ensure_ws(c, T); !m.empty()) return m;

    std::vector<int32_t> pos(T);
    for (uint32_t t = 0; t < T; ++t) pos[t] = int32_t(start_pos + t);
    for (uint32_t c = 0; c < n_dev_; ++c) {
        fleet_->dev(c).queue().memcpy(ws_[c].positions, pos.data(), T * sizeof(int32_t)).wait();
        if (reset_kv) kv_[c].reset();
    }

    // embedding on card 0 → ws[0].x; broadcast replicated x to every card.
    {
        auto& a = fleet_->dev(0);
        auto& q = a.queue();
        int32_t* d_ids = static_cast<int32_t*>(a.malloc(T * sizeof(int32_t)));
        if (!d_ids) return "d_ids alloc failed";
        q.memcpy(d_ids, input_ids, T * sizeof(int32_t)).wait();
        if (token_embd_dtype_ == DType::kQ4_K)      embedding_lookup_q4k(q, d_ids, token_embd_, ws_[0].x, T, H);
        else if (token_embd_dtype_ == DType::kQ8_0) embedding_lookup_q8_0(q, d_ids, token_embd_, ws_[0].x, T, H);
        else                                         embedding_lookup_q6k(q, d_ids, token_embd_, ws_[0].x, T, H);
        q.wait();
        a.free(d_ids);
    }
    for (uint32_t c = 1; c < n_dev_; ++c)
        fleet_->copy_across(0, ws_[c].x, c, ws_[0].x, uint64_t(T) * H * sizeof(sycl::half));

    std::vector<sycl::half*> moe_y_bufs(n_dev_), attn_block_bufs(n_dev_);
    for (uint32_t c = 0; c < n_dev_; ++c) { moe_y_bufs[c] = ws_[c].ffn_y; attn_block_bufs[c] = ws_[c].attn_block; }

    for (uint32_t L = 0; L < n_layers; ++L) {
        // ===== ATTENTION head-shard → o-proj PARTIAL [T,H] → all-reduce #1 =====
        for (uint32_t c = 0; c < n_dev_; ++c) {
            Workspace& ws = ws_[c];
            const LayerW& w = dlayers_[c][L];
            const Shard& s = shard_[c];
            auto& q = fleet_->dev(c).queue();
            const uint32_t sNq = s.Nq, sNkv = s.Nkv, snq = s.nq, snkv = s.nkv;

            rms_norm_f32w(q, ws.x, w.attn_norm, ws.x_normed, T, H, eps);
            dense::gemv_q_T(q, ws.x_normed, w.attn_q, ws.q, H, sNq,  T);
            dense::gemv_q_T(q, ws.x_normed, w.attn_k, ws.k, H, sNkv, T);
            dense::gemv_q_T(q, ws.x_normed, w.attn_v, ws.v, H, sNkv, T);
            if (w.attn_q_norm) rms_norm_f32w(q, ws.q, w.attn_q_norm, ws.q, T * snq,  HD, eps);
            if (w.attn_k_norm) rms_norm_f32w(q, ws.k, w.attn_k_norm, ws.k, T * snkv, HD, eps);
            rope_partial(q, ws.q, ws.positions, ws.q, T, snq,  HD, d.rope_dim, d.rope_theta);
            rope_partial(q, ws.k, ws.positions, ws.k, T, snkv, HD, d.rope_dim, d.rope_theta);
            const uint64_t per_layer = uint64_t(snkv) * kv_[c].config().max_ctx * HD;
            sycl::half* kc = kv_[c].k_ptr() + per_layer * L;
            sycl::half* vc = kv_[c].v_ptr() + per_layer * L;
            const uint32_t mc = kv_[c].config().max_ctx;
            if (T == 1 && fa2_decode_on && ws.attn_partials) {
                full_attention_fa2_decode(q, ws.q, ws.k, ws.v, kc, vc, ws.attn_out,
                                          ws.attn_partials, start_pos, snq, snkv, HD, mc);
            } else if (T > 1 && fa2_prefill_on) {
                full_attention_fa2_prefill_tile(q, ws.q, ws.k, ws.v, kc, vc, ws.attn_out,
                                                T, start_pos, snq, snkv, HD, mc);
            } else {
                full_attention(q, ws.q, ws.k, ws.v, kc, vc, ws.attn_out, T, start_pos,
                               snq, snkv, HD, mc);
            }
            kv_[c].set_length(L, start_pos + T);
            dense::gemv_q_T(q, ws.attn_out, w.attn_output, ws.attn_block, sNq, H, T);  // PARTIAL [T,H]
        }
        fleet_->all_reduce_sum_fp16(attn_block_bufs, uint64_t(T) * H);
        for (uint32_t c = 0; c < n_dev_; ++c) {
            auto& q = fleet_->dev(c).queue();
            residual_add(q, ws_[c].x, ws_[c].attn_block, ws_[c].x, uint64_t(T) * H);
            rms_norm_f32w(q, ws_[c].x, dlayers_[c][L].ffn_norm, ws_[c].x_normed, T, H, eps);
        }

        // ===== MoE expert-shard → PARTIAL [T,H] → all-reduce #2 =====
        for (uint32_t c = 0; c < n_dev_; ++c) moe_ffn_card(c, dlayers_[c][L], T);
        fleet_->all_reduce_sum_fp16(moe_y_bufs, uint64_t(T) * H);
        for (uint32_t c = 0; c < n_dev_; ++c)
            residual_add(fleet_->dev(c).queue(), ws_[c].x, ws_[c].ffn_y, ws_[c].x, uint64_t(T) * H);
    }
    for (uint32_t c = 0; c < n_dev_; ++c) fleet_->dev(c).queue().wait();

    // final norm + lm_head on card 0 (x replicated).
    {
        Workspace& ws = ws_[0];
        auto& a = fleet_->dev(0);
        auto& q = a.queue();
        rms_norm_f32w(q, ws.x, output_norm_, ws.x_normed, T, H, eps);
        const sycl::half* last = ws.x_normed + uint64_t(T - 1) * H;
        sycl::half* d_logits = static_cast<sycl::half*>(a.malloc(uint64_t(V) * sizeof(sycl::half)));
        if (!d_logits) return "logits alloc failed";
        dense::gemv_q(q, last, DenseQuantPtr{output_, output_dtype_}, d_logits, H, V).wait();
        q.memcpy(out_logits_host, d_logits, uint64_t(V) * sizeof(sycl::half)).wait();
        a.free(d_logits);
    }
    return {};
}

}  // namespace ie
