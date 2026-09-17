// src/model/deepseek4.cpp — DeepSeek-V4-Flash (`deepseek4`) weight binding.
//
// PHASE 1: host-side binding + validation only. See include/ie/deepseek4.hpp
// for the scope statement — there is no forward here, and adding a stub one
// would be a lie about what runs.
//
// The contract this file enforces is two-way:
//   forward  — every tensor the config implies MUST exist, with the exact dtype
//              and shape the config predicts, or load() fails naming it;
//   backward — every tensor in the FILE must have been consumed by the forward
//              pass, or load() fails naming it. Together these make "1328 of
//              1328 bound" a proof rather than a count that happens to match.
#include "ie/deepseek4.hpp"

#include "ie/allocator.hpp"

#include "ie/deepseek4_attn.hpp"
#include "ie/deepseek4_ops.hpp"
#include "ie/dequant_ref.hpp"
#include "ie/ds4_decode_gemv.hpp"
#include "ie/kernel_profiler.hpp"
#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <initializer_list>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ie {

// $DS4_EXPERT_TRACE record layout: the four fixed columns (forward sequence
// number, token id, chunk width T, layer) followed by the K chosen expert ids.
// The cap is on INTS, i.e. 16 MB of host RAM at the widest, which at K=6 and 43
// layers is ~9 700 decode tokens — far more than any capture needs and small
// enough that leaving the switch on by accident cannot matter.
constexpr uint32_t kDs4TraceFields  = 4;
constexpr size_t   kDs4TraceMaxInts = 4u << 20;

namespace {

std::string dims_str(const uint64_t* s, uint32_t n) {
    std::string out = "[";
    char buf[24];
    for (uint32_t i = 0; i < n; ++i) {
        std::snprintf(buf, sizeof(buf), "%llu", (unsigned long long)s[i]);
        if (i) out += ", ";
        out += buf;
    }
    return out + "]";
}

std::string blk(uint32_t L, const char* suffix) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "blk.%u.%s", L, suffix);
    return buf;
}

// One binder over one reader. Records the names it consumed so load() can prove
// the file holds nothing it did not bind.
class Binder {
public:
    explicit Binder(const GgufReader& g) : g_(g) {}

    // `want == DType::kCount` means "accept whatever type_id the tensor itself
    // carries" — used for the routed experts, whose dtype varies PER TENSOR
    // (blk.26 gate/up are MXFP4 while every other layer's are IQ3_XXS). The
    // byte-size check below is then computed from that same per-tensor dtype,
    // so a role-wide dtype assumption cannot survive it.
    std::string bind(const GgufTensorInfo*& dst, const std::string& name, DType want,
                     std::initializer_list<uint64_t> shape) {
        const GgufTensorInfo* ti = g_.find_tensor(name);
        if (!ti) return name + ": not found";
        if (want != DType::kCount && ti->dtype != want)
            return name + ": expected dtype " + std::string(type_name(want)) +
                   ", file has " + std::string(type_name(ti->dtype));
        if (ti->n_dims != shape.size())
            return name + ": expected " + std::to_string(shape.size()) + " dims, file has " +
                   std::to_string(ti->n_dims) + " " + dims_str(ti->shape.data(), ti->n_dims);
        uint32_t i = 0;
        for (uint64_t want_d : shape) {
            if (ti->shape[i] != want_d)
                return name + ": expected shape " + dims_str(shape.begin(), uint32_t(shape.size())) +
                       ", file has " + dims_str(ti->shape.data(), ti->n_dims);
            ++i;
        }
        // Per-tensor dtype dispatch: the block geometry that must divide the row
        // length, and the byte size, both come from THIS tensor's type_id.
        const uint32_t bs = type_block_size(ti->dtype);
        if (bs > 1 && ti->shape[0] % bs)
            return name + ": row length " + std::to_string(ti->shape[0]) + " not a multiple of the " +
                   std::string(type_name(ti->dtype)) + " block size " + std::to_string(bs);
        uint64_t hi = 1;
        for (uint32_t d = 1; d < ti->n_dims; ++d) hi *= ti->shape[d];
        const uint64_t want_bytes = uint64_t(bytes_for(ti->dtype, size_t(ti->shape[0]))) * hi;
        if (want_bytes != ti->nbytes)
            return name + ": " + std::string(type_name(ti->dtype)) + " size mismatch (expect " +
                   std::to_string(want_bytes) + " B, file says " + std::to_string(ti->nbytes) + " B)";
        if (!ti->data) return name + ": no mmap backing (tensor unresolved in every shard)";
        seen_.insert(ti->name);
        dst = ti;
        return {};
    }

    // Binds an always-resident 2-D weight at whatever dtype the FILE declares,
    // gated on `ds4_dense_uploadable`.  This is the same per-tensor `type_id`
    // discipline the routed experts already used, applied to the dense set —
    // which is what lets ONE loader bind both UD-Q3_K_XL (Q8_0 / Q6_K here) and
    // UD-Q8_K_XL (BF16 here).  It is a whitelist, not a wildcard: a dtype this
    // engine cannot upload still fails by name, and the shape and byte-size
    // checks below are unchanged and still run against the file's own dtype.
    std::string bind_dense(const GgufTensorInfo*& dst, const std::string& name,
                           std::initializer_list<uint64_t> shape) {
        const GgufTensorInfo* ti = g_.find_tensor(name);
        if (!ti) return name + ": not found";
        if (!ds4_dense_uploadable(ti->dtype))
            return name + ": dtype " + std::string(type_name(ti->dtype)) +
                   " has no always-resident upload path in this engine";
        return bind(dst, name, ti->dtype, shape);
    }

    // Binds a role the ENGINE consumes as F32 from any source dtype the F32
    // row upload (`dequant_rows`) converts.  The NativePreserved file stores
    // the hyper-connection fns and the compressor APEs as F32; ggml-org's
    // MXFP4 file stores the same five roles as Q8_0 (2026-09-11).  Shape and
    // byte-size checks run against the file's own dtype, and `bind`'s block
    // geometry check covers Q8_0's K % 32.  Every other F32 role (norms,
    // biases, sinks, hc base/scale) stays strict: no file quantizes them.
    std::string bind_f32_src(const GgufTensorInfo*& dst, const std::string& name,
                             std::initializer_list<uint64_t> shape) {
        const GgufTensorInfo* ti = g_.find_tensor(name);
        if (!ti) return name + ": not found";
        switch (ti->dtype) {
            case DType::kF32: case DType::kF16: case DType::kBF16:
            case DType::kQ8_0: case DType::kQ6_K: case DType::kQ4_K: break;
            default:
                return name + ": dtype " + std::string(type_name(ti->dtype)) +
                       " cannot be dequantized to F32 at load";
        }
        return bind(dst, name, ti->dtype, shape);
    }

    // The negative half of the layer-variation contract.
    std::string require_absent(const std::string& name, const char* why) {
        if (g_.find_tensor(name)) return name + ": present, but " + why;
        return {};
    }

    const std::unordered_set<std::string_view>& seen() const noexcept { return seen_; }

private:
    const GgufReader& g_;
    std::unordered_set<std::string_view> seen_;
};

}  // namespace

// The GGUF records only a numeric ratio per layer; it does NOT say which of the
// reference's two compressor blocks a layer runs. The reference does, though:
// CSA (paper §2.3.1) compresses every m=4 tokens and projects to 2*head_dim,
// HCA (§2.3.2) compresses every m'=128 and projects to head_dim — "heavily
// compressed" is by construction the LARGER rate. So the two distinct nonzero
// ratios in the schedule are classified smallest→CSA, largest→HCA rather than
// hardcoding 4 and 128. A schedule with more or fewer than two distinct nonzero
// ratios cannot be classified this way and is rejected instead of guessed at.
std::string DeepSeek4Model::layer_kind(const DeepSeek4Config& cfg, uint32_t layer,
                                       DeepSeek4LayerKind& out) {
    out = {};
    if (layer >= cfg.n_layers) return "layer_kind: layer index out of range";
    if (cfg.compress_ratios.size() < cfg.n_layers)
        return "compress_ratios: only " + std::to_string(cfg.compress_ratios.size()) +
               " entries for " + std::to_string(cfg.n_layers) + " layers";
    uint32_t csa = 0, hca = 0, distinct = 0;
    for (uint32_t L = 0; L < cfg.n_layers; ++L) {
        const int32_t r = cfg.compress_ratios[L];
        if (r < 0) return "compress_ratios[" + std::to_string(L) + "]: negative";
        if (r == 0) continue;
        const auto u = uint32_t(r);
        if (u == csa || u == hca) continue;
        if (distinct == 0)      { csa = u; ++distinct; }
        else if (distinct == 1) { hca = u; ++distinct; if (hca < csa) std::swap(csa, hca); }
        else return "compress_ratios: more than two distinct nonzero ratios — cannot map "
                    "ratio→(CSA|HCA) block type without the reference config";
    }
    if (distinct != 2)
        return "compress_ratios: " + std::to_string(distinct) +
               " distinct nonzero ratio(s), expected exactly two (CSA + HCA)";

    const auto r = uint32_t(cfg.compress_ratios[layer]);
    out.compress_ratio = r;
    out.hash_router    = (layer < cfg.hash_layer_count);
    if (r == 0) return {};                       // layers 0-1: no long-range compressor
    out.has_compressor = true;
    if (r == csa) { out.has_indexer = true; out.compressor_dim = 2u * cfg.head_dim; }
    else          {                         out.compressor_dim = cfg.head_dim; }
    return {};
}

std::string DeepSeek4Model::load(const GgufReader& g, const DeepSeek4Config& cfg) {
    cfg_     = cfg;
    globals_ = {};
    layers_.clear();
    n_bound_ = 0;

    if (cfg_.n_layers == 0 || cfg_.hidden == 0 || cfg_.head_dim == 0 || cfg_.n_experts == 0)
        return "deepseek4: config not populated (call read_deepseek4_config first)";

    const uint32_t H    = cfg_.hidden;
    const uint32_t HD   = cfg_.head_dim;
    const uint32_t E    = cfg_.n_experts;
    const uint32_t IHD  = cfg_.indexer_head_dim;
    const uint32_t MIX  = (2u + cfg_.hc_count) * cfg_.hc_count;   // 24 — see 23_hyper_connections_solved.md
    const uint32_t HCH  = cfg_.hc_count * H;                      // 16384

    Binder b(g);
    std::string e;

    // ---- non-layer ----
    // token_embd and output are bind_dense: Q3 stores them Q8_0 / Q6_K, Q8 stores
    // both BF16.  Their dtype is read, not predicted.
    if (e = b.bind_dense(globals_.token_embd, "token_embd.weight",                   {H, cfg_.vocab});      !e.empty()) return e;
    if (e = b.bind_dense(globals_.output,     "output.weight",                       {H, cfg_.vocab});      !e.empty()) return e;
    if (e = b.bind(globals_.output_norm,     "output_norm.weight",     DType::kF32,  {H});                  !e.empty()) return e;
    if (e = b.bind_f32_src(globals_.output_hc_fn, "output_hc_fn.weight",              {HCH, cfg_.hc_count}); !e.empty()) return e;
    if (e = b.bind(globals_.output_hc_base,  "output_hc_base.weight",  DType::kF32,  {cfg_.hc_count});      !e.empty()) return e;
    if (e = b.bind(globals_.output_hc_scale, "output_hc_scale.weight", DType::kF32,  {1});                  !e.empty()) return e;

    // ---- layers ----
    layers_.assign(cfg_.n_layers, {});
    for (uint32_t L = 0; L < cfg_.n_layers; ++L) {
        DeepSeek4Layer& w = layers_[L];
        if (e = layer_kind(cfg_, L, w.kind); !e.empty()) return e;

        // attention (every layer)
        if (e = b.bind(w.attn_norm,      blk(L, "attn_norm.weight"),      DType::kF32,  {H});                          !e.empty()) return e;
        if (e = b.bind_dense(w.attn_q_a, blk(L, "attn_q_a.weight"),                     {H, cfg_.q_lora_rank});        !e.empty()) return e;
        if (e = b.bind(w.attn_q_a_norm,  blk(L, "attn_q_a_norm.weight"),  DType::kF32,  {cfg_.q_lora_rank});           !e.empty()) return e;
        if (e = b.bind_dense(w.attn_q_b, blk(L, "attn_q_b.weight"),                     {cfg_.q_lora_rank, cfg_.n_q_heads * HD}); !e.empty()) return e;
        if (e = b.bind_dense(w.attn_kv,  blk(L, "attn_kv.weight"),                      {H, HD});                      !e.empty()) return e;
        if (e = b.bind(w.attn_kv_a_norm, blk(L, "attn_kv_a_norm.weight"), DType::kF32,  {HD});                         !e.empty()) return e;
        if (e = b.bind(w.attn_sinks,     blk(L, "attn_sinks.weight"),     DType::kF32,  {cfg_.n_q_heads});             !e.empty()) return e;
        // Grouped output projection: rank o_lora_rank per group, o_groups groups.
        const uint64_t o_rank = uint64_t(cfg_.o_lora_rank) * cfg_.o_groups;
        if (e = b.bind_dense(w.attn_output_a, blk(L, "attn_output_a.weight"),           {H, o_rank});                  !e.empty()) return e;
        if (e = b.bind_dense(w.attn_output_b, blk(L, "attn_output_b.weight"),           {o_rank, H});                  !e.empty()) return e;

        // long-range compressor — present iff compress_ratios[L] != 0
        if (w.kind.has_compressor) {
            const uint32_t CD = w.kind.compressor_dim;
            if (e = b.bind_dense(w.compressor_kv,   blk(L, "attn_compressor_kv.weight"),   {H, CD});                       !e.empty()) return e;
            if (e = b.bind_dense(w.compressor_gate, blk(L, "attn_compressor_gate.weight"), {H, CD});                       !e.empty()) return e;
            if (e = b.bind_f32_src(w.compressor_ape, blk(L, "attn_compressor_ape.weight"),      {CD, w.kind.compress_ratio});   !e.empty()) return e;
            if (e = b.bind(w.compressor_norm, blk(L, "attn_compressor_norm.weight"), DType::kF32,  {HD});                          !e.empty()) return e;
        } else {
            for (const char* nm : {"attn_compressor_kv.weight", "attn_compressor_gate.weight",
                                   "attn_compressor_ape.weight", "attn_compressor_norm.weight"})
                if (e = b.require_absent(blk(L, nm), "compress_ratios says this layer has no compressor"); !e.empty()) return e;
        }

        // lightning indexer — present iff this is a CSA (fine-grained) layer
        if (w.kind.has_indexer) {
            // indexer.proj is the quietest of the lot: F32 in Q3, BF16 in Q8. It
            // is a dense upload like the rest, so it is bound the same way.
            if (e = b.bind_dense(w.indexer_q_b,             blk(L, "indexer.attn_q_b.weight"),        {cfg_.q_lora_rank, cfg_.indexer_n_heads * IHD}); !e.empty()) return e;
            if (e = b.bind_dense(w.indexer_proj,            blk(L, "indexer.proj.weight"),            {H, cfg_.indexer_n_heads});                      !e.empty()) return e;
            if (e = b.bind_dense(w.indexer_compressor_kv,   blk(L, "indexer_compressor_kv.weight"),   {H, 2u * IHD});                                  !e.empty()) return e;
            if (e = b.bind_dense(w.indexer_compressor_gate, blk(L, "indexer_compressor_gate.weight"), {H, 2u * IHD});                                  !e.empty()) return e;
            if (e = b.bind_f32_src(w.indexer_compressor_ape, blk(L, "indexer_compressor_ape.weight"),      {2u * IHD, w.kind.compress_ratio});              !e.empty()) return e;
            if (e = b.bind(w.indexer_compressor_norm, blk(L, "indexer_compressor_norm.weight"), DType::kF32,  {IHD});                                          !e.empty()) return e;
        } else {
            for (const char* nm : {"indexer.attn_q_b.weight", "indexer.proj.weight",
                                   "indexer_compressor_kv.weight", "indexer_compressor_gate.weight",
                                   "indexer_compressor_ape.weight", "indexer_compressor_norm.weight"})
                if (e = b.require_absent(blk(L, nm), "this is not a fine-grained (CSA) layer"); !e.empty()) return e;
        }

        // MoE. The three *_exps tensors are bound with want=kCount so the dtype
        // comes from each tensor's own type_id — the only way blk.26's MXFP4
        // gate/up bind alongside 42 layers of IQ3_XXS.
        if (e = b.bind(w.ffn_norm,       blk(L, "ffn_norm.weight"),       DType::kF32,   {H});                         !e.empty()) return e;
        if (e = b.bind_dense(w.ffn_gate_inp, blk(L, "ffn_gate_inp.weight"),               {H, E});                      !e.empty()) return e;
        if (e = b.bind(w.ffn_gate_exps,  blk(L, "ffn_gate_exps.weight"),  DType::kCount, {H, cfg_.expert_ffn, E});     !e.empty()) return e;
        if (e = b.bind(w.ffn_up_exps,    blk(L, "ffn_up_exps.weight"),    DType::kCount, {H, cfg_.expert_ffn, E});     !e.empty()) return e;
        if (e = b.bind(w.ffn_down_exps,  blk(L, "ffn_down_exps.weight"),  DType::kCount, {cfg_.expert_ffn, H, E});     !e.empty()) return e;
        for (const GgufTensorInfo* ti : {w.ffn_gate_exps, w.ffn_up_exps, w.ffn_down_exps}) {
            // Per-tensor gate on the expert dtype. The whitelist is exactly what
            // this engine can turn back into floats TODAY (host dequant refs);
            // it is applied to each tensor's own type_id, never to a role.
            switch (ti->dtype) {
                case DType::kIQ3_XXS:
                case DType::kMXFP4:
                    break;
                default:
                    return std::string(ti->name) + ": expert dtype " +
                           std::string(type_name(ti->dtype)) + " has no dequant path in this engine";
            }
        }
        if (e = b.bind_dense(w.ffn_gate_shexp, blk(L, "ffn_gate_shexp.weight"),         {H, cfg_.expert_ffn});        !e.empty()) return e;
        if (e = b.bind_dense(w.ffn_up_shexp,   blk(L, "ffn_up_shexp.weight"),           {H, cfg_.expert_ffn});        !e.empty()) return e;
        if (e = b.bind_dense(w.ffn_down_shexp, blk(L, "ffn_down_shexp.weight"),         {cfg_.expert_ffn, H});        !e.empty()) return e;

        // router: hash table on the first hash_layer_count layers, learned bias after
        if (w.kind.hash_router) {
            if (e = b.bind(w.ffn_gate_tid2eid, blk(L, "ffn_gate_tid2eid.weight"), DType::kI32, {cfg_.n_experts_used, cfg_.vocab}); !e.empty()) return e;
            if (e = b.require_absent(blk(L, "exp_probs_b.bias"), "hash-router layers carry no learned router bias"); !e.empty()) return e;
        } else {
            if (e = b.bind(w.exp_probs_b, blk(L, "exp_probs_b.bias"), DType::kF32, {E}); !e.empty()) return e;
            if (e = b.require_absent(blk(L, "ffn_gate_tid2eid.weight"), "only the first hash_layer_count layers hash-route"); !e.empty()) return e;
        }
        // Vision-Exp GGUFs carry `exp_probs_b_vl.bias` on EVERY layer (the hash
        // layers too — image tokens never hash-route; they top-k over
        // `ffn_gate_inp` + this bias). Text-only files have none. Optional, but
        // if present it must be the F32 [E] the reference declares.
        if (g.find_tensor(blk(L, "exp_probs_b_vl.bias"))) {
            if (e = b.bind(w.exp_probs_b_vl, blk(L, "exp_probs_b_vl.bias"), DType::kF32, {E}); !e.empty()) return e;
        }

        // hyper-connections
        if (e = b.bind_f32_src(w.hc_attn_fn, blk(L, "hc_attn_fn.weight"),           {HCH, MIX}); !e.empty()) return e;
        if (e = b.bind(w.hc_attn_base,  blk(L, "hc_attn_base.weight"),  DType::kF32, {MIX});      !e.empty()) return e;
        if (e = b.bind(w.hc_attn_scale, blk(L, "hc_attn_scale.weight"), DType::kF32, {3});        !e.empty()) return e;
        if (e = b.bind_f32_src(w.hc_ffn_fn,  blk(L, "hc_ffn_fn.weight"),            {HCH, MIX}); !e.empty()) return e;
        if (e = b.bind(w.hc_ffn_base,   blk(L, "hc_ffn_base.weight"),   DType::kF32, {MIX});      !e.empty()) return e;
        if (e = b.bind(w.hc_ffn_scale,  blk(L, "hc_ffn_scale.weight"),  DType::kF32, {3});        !e.empty()) return e;
    }

    // ---- backward half of the contract: nothing in the file left over ----
    for (const GgufTensorInfo& ti : g.tensors())
        if (!b.seen().count(ti.name))
            return std::string(ti.name) + ": present in the GGUF but bound by nothing — the "
                   "deepseek4 tensor set in this loader is incomplete";
    if (b.seen().size() != g.n_tensors())
        return "bound " + std::to_string(b.seen().size()) + " distinct names but the file lists " +
               std::to_string(g.n_tensors()) + " tensors (duplicate names?)";

    n_bound_ = b.seen().size();
    return {};
}


// ===========================================================================
// PHASE 5 — the always-resident packing policy
// ===========================================================================
//
// Q8_0 and Q6_K are packed because `q8_0_at` / `q6_K_at` below decode one
// element of a GGUF row in registers, bit-exactly against `ie::ref`.  BF16 and
// F16 are NOT listed: they are already 2 B/element, so packing them would save
// nothing and would only add a kernel path.  F32 expands to fp16 (a halving).
// Everything else has no device decoder here and expands.
uint64_t ds4_dense_row_bytes(DType dt, uint64_t K) noexcept {
    if (dt == DType::kQ8_0) return (K / 32) * 34;
    if (dt == DType::kQ6_K) return (K / 256) * 210;
    return 0;
}

bool ds4_dense_keeps_packed(DType dt, uint64_t K) noexcept {
    if (dt == DType::kQ8_0) return K != 0 && (K % 32) == 0;
    if (dt == DType::kQ6_K) return K != 0 && (K % 256) == 0;
    return false;
}

uint64_t ds4_dense_device_bytes(DType dt, uint64_t K, uint64_t N) noexcept {
    if (ds4_dense_keeps_packed(dt, K)) return ds4_dense_row_bytes(dt, K) * N;
    return K * N * 2;   // fp16 expansion
}

// BF16 IS DELIBERATELY NOT PACKED, and that is still true: a LOSSLESS packed
// BF16 saves exactly ZERO bytes over the fp16 expansion (BF16 is already
// 2 B/element, so `ds4_dense_device_bytes` returns K*N*2 either way) while
// costing a third `packed_at` specialisation and a shift plus bitcast per
// element in the innermost loop.  The expansion is a widening of the mantissa
// (BF16's 8 explicit bits into fp16's 11), so it loses no precision; what it
// narrows is the EXPONENT, and `upload_dense` hard-errors on any weight that
// would overflow fp16 rather than uploading an inf.
//
// What that argument does NOT cover — and what `ds4_dense_requantises` below
// does — is the LOSSY option.  Q8_0 is 1.0625 B/element, so it is the one thing
// that actually moves fewer bytes, and it is a different question from packing
// because the answer has to be paid for in accuracy rather than in kernels.
// The two policies are separate functions for exactly that reason: this one is
// "can the file's bytes be used as they are", the other is "may this tensor be
// made smaller than the file".

// A weight is requantised iff BOTH halves of the policy agree: the dtype and row
// length allow it (here) and the CALL SITE allows it (Ds4DenseQuant).
//
// BF16 ONLY.  It is the one dtype in either shipped quant's always-resident set
// that has no packed path and therefore no way to cost less than 2 B/element.
// F32 and F16 also reach `upload_dense` (UD-Q3_K_XL stores `indexer.proj` as
// F32) but they are a handful of small tensors, and the ONE tensor that would
// qualify is `indexer.proj`, which the call-site policy excludes anyway.
// Widening this predicate would buy bytes that do not show and spend risk that
// does.
// DEFAULT OFF as of 2026-08-03, on real-model evidence that contradicts the
// microbenchmark this path was built from.  Set IE_DS4_DENSE_Q8=1 to re-enable.
//
// The isolated kernel measurement was sound and reproducible: Q8_0-SoA moves
// 1.88x fewer bytes and runs 1.725x faster than fp16 across the eleven real
// per-card decode shapes, cache-cold.  On the real model, both axes REGRESSED:
//
//                        fp16 (shipped)   Q8_0-SoA
//   prefill pp512          116.85           55.67    -52%
//   decode  tg4096          16.70           14.52    -13%
//   decode  @ slots 128        —            13.12
//
// Measured back to back, same build, same machine, uncontended, chunk 1024.
// Raising slots made it worse, not better, so the extra 3.737 GB/card returned
// to the expert arena did not pay for it either.
//
// The prefill half is understood in direction if not in size: at T > 1 the
// weight has to be materialised back to fp16 for the oneDNN route, and that
// extra pass was ESTIMATED at under 0.4% of pp512. It is evidently far larger.
// The decode regression is NOT understood — the kernel is faster in isolation
// and moves fewer bytes, so something about the real residency, the arena
// pressure, or the materialisation is not represented in the harness.
//
// Everything else the work produced is kept and still gated: the SoA layout, the
// requantiser, `ds4_decode_gemv_q8`, and deepseek4_dense_gate_test §4/§5/§8
// (which prove both the kernel and the quantisation cost). Re-enabling is one
// environment variable, so whoever explains the gap does not have to rebuild it.
//
// Do NOT re-derive this from the microbenchmark. It has been measured twice.
// DEFAULT ON as of 2026-08-03, after a teacher-forced perplexity gate on TWO
// corpora. Set IE_DS4_DENSE_Q8=0 to take the fp16 dense weights instead.
//
// PERPLEXITY (tools/ie-ds4-ppl, --mode stream, fixed tokens, teacher-forced —
// immune to the run-to-run non-determinism documented in docs/deepseek4/50):
//
//   corpus              tokens  context   fp16      Q8       delta
//   built-in prose         589  short     5.2694   5.3139   +0.84%
//   wikitext-2 test       3071  >2048     2.4286   2.4225   -0.25%
//
// The wikitext row is the one that counts: at 589 tokens a ratio-4 CSA layer has
// only ~147 compressed entries against a top_k of 512, so `ds4_indexer_topk`
// admits everything and never SELECTS. Only past ~2048 tokens does the discrete
// path engage — and there Q8 measures marginally BETTER, not worse.
//
// The run-to-run control is exact: re-running the fp16 arm reproduces 511/511
// tokens bit-identically, delta 0.0000%. So these deltas are signal.
//
// WHAT THIS COSTS, STATED HONESTLY. It is not free per token. At long context 35
// of 3071 tokens move by more than 1 nat (max 4.21) — the indexer selecting a
// different set of compressed entries. That is symmetric (1704 worse, 1367
// better) and does not accumulate: mean |delta| 0.077 nats, median 0.0014. An
// earlier reading rejected this change on a single >1-nat token at short context,
// on the inference that only a discrete flip could move a token that far.
// Excluding the indexer compressor tensors was then measured BYTE-IDENTICAL, which
// falsified the inference — at 589 tokens the indexer is not even selecting.
//
// WHAT IT BUYS. Real model, 2x B70, per-layer residency, ctx 4096, decode 512:
// 19.55 -> 26.48 tok/s (+35%). Two mechanisms, and the second is the larger:
// the Q8 GEMV is 1.66-1.80x on the real decode shapes, AND the resident dense set
// shrinks 9.375 -> 5.638 GB, which the expert-arena derivation (global_mem -
// resident - 2 GB) converts into 79 -> 92 expert slots per layer. Cache hit rate
// 0.772 -> 0.876, decode expert DMA 785 -> 427 MB/token.
//
// KNOWN COST, NOT YET FIXED: prefill regresses 115.88 -> 52.91 tok/s (three
// measurements, 52.91 / 55.67 / 56.97 — the effect is reproducible).
//
// THE STATED CAUSE IS FALSIFIED. It was "at T > 1 the weight is materialised
// back to fp16 for the oneDNN route". That pass moves 10.45 GB per card per
// chunk over the real per-card call set (3.411e9 requantised elements; the
// derivation is in tools/ie_ds4_dense_prefill_bench.cpp and reproduces the
// 6.92 GB/card figure this file records elsewhere), which at 450-590 GB/s is
// 18-23 ms. The regression is +5,259 ms of pp512 wall. The materialisation model
// is 229x too small; it is not what runs.
//
// Two routes ARE the right order of magnitude, and both are SILENT FALLBACKS
// reachable from `dense_w`'s requantised branch on a null scratch pointer:
//   scalar dense_f16<16>  re-reads the ACTIVATION once per output row —
//                         T*4*sum(N*K) = 6,986 GB out of L2, ~3,500 ms
//   per-token GEMV loop   T passes over the weight — 1,857 GB, ~4,100 ms
//                         plus 308,224 launches
// Reading the code cannot separate them, and NEITHER COULD THE PROFILE: the
// scalar kernel and the oneDNN matmul it falls back from were submitted under
// the SAME profiler name, and none of the `ds4_dense_q8@…` names was in
// ie_ds4_bench's bucket table at all. Both are fixed; both fallbacks now print
// one line on stderr; `ie-ds4-dense-prefill-bench` measures every route at the
// real shapes with no model loaded. Until that measurement exists, the cause is
// NARROWED, not known.
//
// The no-materialise route ($IE_DS4_DENSE_S8=1, oneDNN weight decompression) is
// implemented and default OFF pending that same measurement.
bool ds4_dense_requantises(DType dt, uint64_t K) noexcept {
    static const bool on = [] {
        const char* e = std::getenv("IE_DS4_DENSE_Q8");
        return !(e && *e) || std::string(e) != "0";
    }();
    // BF16 (Unsloth 0731), F16 and F32 (the Vision-Exp source-preserving GGUF:
    // FP8 dense expanded to F16, BF16 embeddings/head expanded to F32) all have
    // no packed path and all go through the same fp32 row source, so the
    // requantiser is dtype-blind past this gate.
    //
    // Q8_0 sources (ggml-org's MXFP4 file stores every dense projection so,
    // 2026-09-11) take this route too, ahead of their packed path: measured on
    // that file the packed kernels gave pp 66 / tg 18.9 tok/s at 4096 against
    // ~650 / ~25 on this route with the F16-dense file.  The detour is exact:
    // `dequant_rows` reproduces d*qs bit-for-bit and the requantiser recovers
    // d (amax/127 with the block's +-127 element) and every qs, so the SoA
    // image carries the file's own values.  Q6_K keeps its packed path (a
    // re-quantisation would be lossy).  IE_DS4_DENSE_Q8=0 restores packed.
    const bool wide = dt == DType::kBF16 || dt == DType::kF16 || dt == DType::kF32 ||
                      dt == DType::kQ8_0;
    return on && wide && K != 0 && (K % 32) == 0;
}

// int8 plane + the fp16 scale plane TWICE: n-major [N][K/32] for the decode
// GEMV and the materialiser, kb-major [K/32][N] for oneDNN's s8
// weight-decompression matmul (which reads attr scales in plain layout and
// ignores the strides on the scales md — measured, 2026-08-08).
// K*N + 2 * N*(K/32)*2 = 1.125 B/element.
uint64_t ds4_dense_requant_bytes(uint64_t K, uint64_t N) noexcept {
    return K * N + 2 * (N * (K / 32) * 2);
}

// ggml's block quantiser, with the reciprocal taken from the fp16-ROUNDED scale
// (see the header): `d16` is what the kernel will multiply by, so quantising
// against anything else would put a rounding in the error budget that buys
// nothing.  A block that is entirely zero gets d = 0 and q = 0, which decodes
// back to exactly zero.
void ds4_requantise_q8_soa(const float* src, uint32_t K, uint32_t nrows,
                           int8_t* qs, sycl::half* d) noexcept {
    const uint32_t nb = K / 32;
    for (uint32_t r = 0; r < nrows; ++r) {
        const float* row = src + size_t(r) * K;
        for (uint32_t b = 0; b < nb; ++b) {
            const float* blk = row + size_t(b) * 32;
            float amax = 0.f;
            for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(blk[j]));
            const sycl::half d16 = sycl::half(amax / 127.f);
            const float df = float(d16);
            d[size_t(r) * nb + b] = d16;
            const float id = df > 0.f ? 1.f / df : 0.f;
            int8_t* out = qs + size_t(r) * K + size_t(b) * 32;
            for (int j = 0; j < 32; ++j) {
                // Rounding the scale DOWN can push |w|/d16 to 127.06, so the
                // clamp is load-bearing, not defensive decoration.
                const long v = std::lround(blk[j] * id);
                out[j] = int8_t(std::min<long>(127, std::max<long>(-127, v)));
            }
        }
    }
}

// The dtypes `upload_dense` can consume: the two that stay packed, plus the ones
// `dequant_rows` (below) can expand to fp16.  MUST be kept in step with that
// switch — a dtype listed here without a `dequant_rows` case would bind and then
// fail mid-upload, hours into a load, instead of failing at bind time.
// The dense-weight placement decision, as a pure function of the file's dtype,
// the full and per-card contraction lengths, and the call site's policy.  It
// lives here, and `upload_dense` calls it, so the rule can be exercised without
// a GPU, a queue or a model — `deepseek4_dense_route_test` is the only thing
// that reads it besides the uploader.
//
// The `K` / `Ks` asymmetry is deliberate and matches the branches below: whether
// a dtype HAS a packed form is a property of the whole row (`K`), while whether
// this card's column slice is a whole number of quantisation blocks is a property
// of the slice (`Ks`).
Ds4DenseRoute ds4_dense_route(DType dt, uint64_t K, uint64_t Ks, Ds4DenseQuant quant) noexcept {
    if (quant == Ds4DenseQuant::kAuto && ds4_dense_requantises(dt, Ks))
        return Ds4DenseRoute::kRequantised;
    if (ds4_dense_keeps_packed(dt, K) && quant != Ds4DenseQuant::kWideF16)
        return Ds4DenseRoute::kPacked;
    return Ds4DenseRoute::kFp16;
}

bool ds4_dense_uploadable(DType dt) noexcept {
    switch (dt) {
        case DType::kF32:
        case DType::kF16:
        case DType::kBF16:
        case DType::kQ8_0:
        case DType::kQ6_K:
        case DType::kQ4_K:
            return true;
        default:
            return false;
    }
}

// ===========================================================================
// PHASE 5 — device primitives owned by this file
// ===========================================================================
namespace {

// ---------------------------------------------------------------------------
// Packed-weight element readers.  Each is the corresponding `ie::ref` dequant
// (include/ie/dequant_ref.hpp, bit-exact vs ggml, gated by dequant_ref_test)
// re-expressed for ONE element, with the multiplications left in the reference's
// order so the result is bit-identical rather than merely close.
// `row` points at the first byte of the weight row; `k` indexes the row.
// ---------------------------------------------------------------------------

// The fp16 fields inside these blocks sit at odd multiples of 2 from the block
// base, so they are 2 B aligned but not 4 B — assemble them from bytes rather
// than relying on the pointer cast being legal.
inline float ds4_half_at(const uint8_t* p) {
    const uint16_t bits = uint16_t(p[0]) | uint16_t(uint16_t(p[1]) << 8);
    return float(sycl::bit_cast<sycl::half>(bits));
}

// Q8_0: 32 elements / 34 B = { fp16 d; int8 qs[32]; }.  ref: y[i] = d * qs[i].
inline float q8_0_at(const uint8_t* row, uint32_t k) {
    const uint8_t* b = row + uint64_t(k >> 5) * 34;
    return ds4_half_at(b) * float(static_cast<int8_t>(b[2 + (k & 31)]));
}

// Q6_K: 256 elements / 210 B = { u8 ql[128]; u8 qh[64]; i8 scales[16]; fp16 d; }.
// ref (dequant_ref.hpp:78-96) walks the block as two 128-element halves, each a
// 4 x 32 lane grid: lane l of sub-group g reads ql[l + 32*(g&1)] (low nibble for
// g<2, high nibble otherwise), the 2-bit extension at qh[l] >> 2g, and scale
// sc[2g + l/16].  This is that indexing solved for k instead of iterated.
inline float q6_K_at(const uint8_t* row, uint32_t k) {
    const uint8_t* b  = row + uint64_t(k >> 8) * 210;
    const uint32_t j  = k & 255u;
    const uint32_t hf = j >> 7;          // which 128-element half
    const uint32_t g  = (j & 127u) >> 5; // 0..3 — which of the half's four lanes-of-32
    const uint32_t l  = j & 31u;
    const uint8_t* ql = b + hf * 64;
    const uint8_t* qh = b + 128 + hf * 32;
    const auto*    sc = reinterpret_cast<const int8_t*>(b + 192) + hf * 8;
    const uint8_t  lo = ql[l + ((g & 1u) ? 32u : 0u)];
    const int32_t  qn = int32_t(static_cast<int8_t>(((g < 2u) ? (lo & 0x0Fu) : (lo >> 4)) |
                                                    (((qh[l] >> (2u * g)) & 3u) << 4))) - 32;
    return ds4_half_at(b + 208) * float(sc[(g << 1) + (l >> 4)]) * float(qn);
}

// One element of a packed row, dispatched on a COMPILE-TIME dtype so the decode
// stays branch-free inside the k-loop.
template <DType DT>
inline float packed_at(const uint8_t* row, uint32_t k) {
    if constexpr (DT == DType::kQ8_0) return q8_0_at(row, k);
    else                              return q6_K_at(row, k);
}

// ---------------------------------------------------------------------------
// CALL-SITE ATTRIBUTION FOR THE SHARED DENSE DISPATCHERS
// ---------------------------------------------------------------------------
//
// The four kernels below are the ONLY dense/grouped GEMV in this model, and
// three completely different parts of the forward pass go through them: the
// attention projections (q_a/q_b/kv/o_a/o_b, the compressors, the indexer), the
// SHARED expert (gate/up/down), and the lm_head.  `KernelProfiler` keys on the
// name a submission was pushed with, so without this the single row
// "ds4_dense_packed" is the sum of all three and a profile cannot say whether a
// decode step is attention-bound or MoE-bound — which is the first question a
// benchmark has to answer.
//
// The tag selects a STRING LITERAL and nothing else.  It changes no shape, no
// dependency and no arithmetic; with the profiler off (`ie::g_profiler ==
// nullptr`, the inference default) the name is never even read.  Every returned
// pointer has static storage duration, which `KernelProfiler::Entry` requires.
enum class Ds4Site : int { kNone = 0, kAttn = 1, kShExp = 2, kLmHead = 3 };
thread_local Ds4Site g_ds4_site = Ds4Site::kNone;

// Sets the tag for a lexical block and restores whatever was there before, so a
// nested scope (there are none today) could not leak.
struct Ds4SiteScope {
    Ds4Site prev;
    explicit Ds4SiteScope(Ds4Site s) noexcept : prev(g_ds4_site) { g_ds4_site = s; }
    ~Ds4SiteScope() noexcept { g_ds4_site = prev; }
    Ds4SiteScope(const Ds4SiteScope&)            = delete;
    Ds4SiteScope& operator=(const Ds4SiteScope&) = delete;
};

const char* const kNmDenseF16[4] = {
    "ds4_dense_f16", "ds4_dense_f16@attn", "ds4_dense_f16@shexp", "ds4_dense_f16@lmhead"};
// THE SCALAR KERNELS GET THEIR OWN NAMES, and the fact that they did not is why
// the Q8 prefill regression went undiagnosed.  `dense_f16<TT>` and the oneDNN
// matmul it is the fallback for were pushed under the SAME profiler name, so
// `--kprofile-prefill` could not tell a run that took the tuned route from a run
// that silently took the scalar one — and those two differ by a factor of ~8 in
// wall time on this model.  A profile that cannot distinguish the fast path from
// its own fallback is not attribution, it is a total.
const char* const kNmDenseF16Scalar[4] = {
    "ds4_dense_f16_scalar", "ds4_dense_f16_scalar@attn", "ds4_dense_f16_scalar@shexp",
    "ds4_dense_f16_scalar@lmhead"};
const char* const kNmGroupedF16Scalar[4] = {
    "ds4_grouped_f16_scalar", "ds4_grouped_f16_scalar@attn", "ds4_grouped_f16_scalar@shexp",
    "ds4_grouped_f16_scalar@lmhead"};
const char* const kNmDensePacked[4] = {
    "ds4_dense_packed", "ds4_dense_packed@attn", "ds4_dense_packed@shexp",
    "ds4_dense_packed@lmhead"};
const char* const kNmGroupedF16[4] = {
    "ds4_grouped_f16", "ds4_grouped_f16@attn", "ds4_grouped_f16@shexp",
    "ds4_grouped_f16@lmhead"};
const char* const kNmGroupedPacked[4] = {
    "ds4_grouped_packed", "ds4_grouped_packed@attn", "ds4_grouped_packed@shexp",
    "ds4_grouped_packed@lmhead"};
// The requantised (Q8_0-SoA) decode GEMVs get their OWN names rather than
// borrowing the fp16 ones: a profile that could not tell "this projection is
// 8-bit now" from "this projection is still fp16" could not attribute the byte
// saving this route exists for.
const char* const kNmDenseQ8[4] = {
    "ds4_dense_q8", "ds4_dense_q8@attn", "ds4_dense_q8@shexp", "ds4_dense_q8@lmhead"};
const char* const kNmGroupedQ8[4] = {
    "ds4_grouped_q8", "ds4_grouped_q8@attn", "ds4_grouped_q8@shexp",
    "ds4_grouped_q8@lmhead"};

inline const char* ds4_kname(const char* const (&tbl)[4]) noexcept {
    return tbl[int(g_ds4_site)];
}

// y[t, n] = Σ_k x[t, k] · w[n*K + k].  ONE kernel for every always-resident
// dense projection in the model: fp32 activation, fp16 weight, fp32 accumulate.
//
// The weight is read once per output row per t-tile of TT, so the weight
// traffic of a T-token call is ceil(T/TT) full passes over the weight.  TT is a
// template parameter and `ds4_tt_for` picks it from T: decode (T <= kTTSmall)
// keeps the original tile of 4 and is unchanged instruction-for-instruction,
// while prefill takes the wide tile and reads the weight ceil(T/kTTLarge)
// times instead of ceil(T/4).
//
// Widening TT is BIT-IDENTICAL, not merely close: for a fixed output (t, n) the
// lane visits exactly the same k sequence (k = lid, lid+WG, ...) in the same
// order and finishes with the same reduce_over_group over the same WG.  TT only
// changes how many independent accumulators one work-group carries, never the
// summation order of any one of them.
template <uint32_t TT>
sycl::event dense_f16(sycl::queue& q, const float* x, const sycl::half* w, float* y,
                      uint32_t T, uint32_t K, uint32_t N,
                      const std::vector<sycl::event>& deps = {}) {
    constexpr uint32_t WG = 128;
    const uint32_t ntile = (T + TT - 1) / TT;
    return ie::ps(q, ds4_kname(kNmDenseF16Scalar), [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>(uint64_t(N) * WG, ntile),
                                         sycl::range<2>(WG, 1)),
                       [=](sycl::nd_item<2> it) {
            const uint32_t n   = uint32_t(it.get_group(0));
            const uint32_t tb  = uint32_t(it.get_group(1)) * TT;
            const uint32_t lid = uint32_t(it.get_local_id(0));
            const sycl::half* wr = w + uint64_t(n) * K;
            float acc[TT] = {};
            for (uint32_t k = lid; k < K; k += WG) {
                const float wv = float(wr[k]);
                #pragma unroll
                for (uint32_t j = 0; j < TT; ++j) {
                    const uint32_t t = tb + j;
                    if (t < T) acc[j] = sycl::fma(wv, x[uint64_t(t) * K + k], acc[j]);
                }
            }
            auto grp = it.get_group();
            for (uint32_t j = 0; j < TT; ++j) {
                const float s = sycl::reduce_over_group(grp, acc[j], sycl::plus<float>());
                if (lid == 0 && tb + j < T) y[uint64_t(tb + j) * N + n] = s;
            }
        });
    });
}

// DeepseekV4GroupedLinear over an fp16 weight (modeling_deepseek_v4.py:303):
//   y[t, g, o] = Σ_h x[t, g, h] · w[(g*OPG + o) * IPG + h]
// Block-diagonal: group g never sees group g'≠g's input.
template <uint32_t TT>
sycl::event grouped_f16(sycl::queue& q, const float* x, const sycl::half* w, float* y,
                        uint32_t T, uint32_t G, uint32_t IPG, uint32_t OPG,
                        const std::vector<sycl::event>& deps = {}) {
    constexpr uint32_t WG = 128;
    const uint32_t N     = G * OPG;
    const uint32_t ntile = (T + TT - 1) / TT;
    return ie::ps(q, ds4_kname(kNmGroupedF16Scalar), [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>(uint64_t(N) * WG, ntile),
                                         sycl::range<2>(WG, 1)),
                       [=](sycl::nd_item<2> it) {
            const uint32_t n   = uint32_t(it.get_group(0));
            const uint32_t g   = n / OPG;
            const uint32_t tb  = uint32_t(it.get_group(1)) * TT;
            const uint32_t lid = uint32_t(it.get_local_id(0));
            const sycl::half* wr = w + uint64_t(n) * IPG;
            float acc[TT] = {};
            for (uint32_t k = lid; k < IPG; k += WG) {
                const float wv = float(wr[k]);
                #pragma unroll
                for (uint32_t j = 0; j < TT; ++j) {
                    const uint32_t t = tb + j;
                    if (t < T)
                        acc[j] = sycl::fma(wv, x[uint64_t(t) * G * IPG + uint64_t(g) * IPG + k],
                                           acc[j]);
                }
            }
            auto grp = it.get_group();
            for (uint32_t j = 0; j < TT; ++j) {
                const float s = sycl::reduce_over_group(grp, acc[j], sycl::plus<float>());
                if (lid == 0 && tb + j < T) y[uint64_t(tb + j) * N + n] = s;
            }
        });
    });
}

// The packed siblings of dense_f16 / grouped_f16.  Structurally identical — same
// tiling, same fp32 accumulate, same reduce — with the fp16 load replaced by an
// element decode, so T == 1 (decode) and T > 1 (prefill) are the same kernel and
// no separate GEMM is needed.  `row_bytes` is the packed stride between rows,
// which is NOT derivable from K alone once the dtype varies.
template <DType DT, uint32_t TT>
sycl::event dense_packed(sycl::queue& q, const float* x, const uint8_t* w, float* y,
                         uint32_t T, uint32_t K, uint32_t N, uint64_t row_bytes,
                         const std::vector<sycl::event>& deps = {}) {
    constexpr uint32_t WG = 128;
    const uint32_t ntile = (T + TT - 1) / TT;
    return ie::ps(q, ds4_kname(kNmDensePacked), [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>(uint64_t(N) * WG, ntile),
                                         sycl::range<2>(WG, 1)),
                       [=](sycl::nd_item<2> it) {
            const uint32_t n   = uint32_t(it.get_group(0));
            const uint32_t tb  = uint32_t(it.get_group(1)) * TT;
            const uint32_t lid = uint32_t(it.get_local_id(0));
            const uint8_t* wr  = w + uint64_t(n) * row_bytes;
            float acc[TT] = {};
            for (uint32_t k = lid; k < K; k += WG) {
                const float wv = packed_at<DT>(wr, k);
                #pragma unroll
                for (uint32_t j = 0; j < TT; ++j) {
                    const uint32_t t = tb + j;
                    if (t < T) acc[j] = sycl::fma(wv, x[uint64_t(t) * K + k], acc[j]);
                }
            }
            auto grp = it.get_group();
            for (uint32_t j = 0; j < TT; ++j) {
                const float s = sycl::reduce_over_group(grp, acc[j], sycl::plus<float>());
                if (lid == 0 && tb + j < T) y[uint64_t(tb + j) * N + n] = s;
            }
        });
    });
}

template <DType DT, uint32_t TT>
sycl::event grouped_packed(sycl::queue& q, const float* x, const uint8_t* w, float* y,
                           uint32_t T, uint32_t G, uint32_t IPG, uint32_t OPG,
                           uint64_t row_bytes, const std::vector<sycl::event>& deps = {}) {
    constexpr uint32_t WG = 128;
    const uint32_t N     = G * OPG;
    const uint32_t ntile = (T + TT - 1) / TT;
    return ie::ps(q, ds4_kname(kNmGroupedPacked), [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>(uint64_t(N) * WG, ntile),
                                         sycl::range<2>(WG, 1)),
                       [=](sycl::nd_item<2> it) {
            const uint32_t n   = uint32_t(it.get_group(0));
            const uint32_t g   = n / OPG;
            const uint32_t tb  = uint32_t(it.get_group(1)) * TT;
            const uint32_t lid = uint32_t(it.get_local_id(0));
            const uint8_t* wr  = w + uint64_t(n) * row_bytes;
            float acc[TT] = {};
            for (uint32_t k = lid; k < IPG; k += WG) {
                const float wv = packed_at<DT>(wr, k);
                #pragma unroll
                for (uint32_t j = 0; j < TT; ++j) {
                    const uint32_t t = tb + j;
                    if (t < T)
                        acc[j] = sycl::fma(wv, x[uint64_t(t) * G * IPG + uint64_t(g) * IPG + k],
                                           acc[j]);
                }
            }
            auto grp = it.get_group();
            for (uint32_t j = 0; j < TT; ++j) {
                const float s = sycl::reduce_over_group(grp, acc[j], sycl::plus<float>());
                if (lid == 0 && tb + j < T) y[uint64_t(tb + j) * N + n] = s;
            }
        });
    });
}

// Token-tile selection.  kTTSmall is the historic tile and is what decode and
// any short chunk still run, so the decode path is untouched; kTTLarge is the
// prefill tile.  A T=512 chunk goes from 128 passes over each weight to 32.
//
// kTTLarge is 16 and NOT larger for two measured reasons (sweep over the five
// real per-card shapes at T=512, B70): 16 is the fastest — 32 and 64 are ~2x
// SLOWER than 4 because acc[TT] stops fitting the GRF and spills — and 16 is
// the largest tile that stays BIT-IDENTICAL to 4.  At 32/64 the spill makes the
// compiler restructure the accumulation and the result moves by ~1.5e-7
// relative (1-2 ULP); at 16 the observed mismatch count is exactly 0 on every
// shape at both T=1 and T=512.  Bit-identity here is a gate, not a nicety:
// deepseek4_residency_test asserts n_cards==1 through the orchestrator is
// bit-identical to plain single-card.
constexpr uint32_t kTTSmall = 4;
constexpr uint32_t kTTLarge = 16;

// ---------------------------------------------------------------------------
// oneDNN route for the fp16 dense projections
// ---------------------------------------------------------------------------
// `dense_f16` and `grouped_f16` above are correct — deepseek4_dense_gate_test
// holds them to a derived fp32 bound with >=14x headroom — but they are scalar
// `sycl::fma` with no XMX, and `dense_f16` re-reads the whole weight
// ceil(T/TT) times, so its arithmetic intensity never amortises in T.  That is
// why prefill measured FLAT in T (pp128 43.56, pp512 49.43, pp2048 45.79) while
// these three buckets carried 65.9% of prefill GPU time.
//
// oneDNN reads the SAME [N, K] weight buffer with no transpose and no copy (see
// gemm_nt_f16_onednn).  What it costs is precision on the ACTIVATION: oneDNN's
// fast matmul path requires an fp16 A operand.  Feeding it fp32 directly is
// accepted by oneDNN but dispatches to a decompression reference kernel measured
// at 200-1600 ms per projection — 25x SLOWER than the kernel it replaces — so
// fp16 activations are not a shortcut here, they are the entry price.
//
// The precision cost is bounded and stated: one fp16 rounding of each activation,
// relative error <= 2^-11, measured at 2.0e-4 scaled against the exact double
// reference. deepseek4_dense_gate_test runs a SECOND time under IE_DS4_ONEDNN=1
// and holds this route to that bound explicitly, so the two routes are gated
// separately and nobody can mistake this one for bit-parity.
//
// Weights are untouched: they stay fp16 [N, K], which is what keeps the decode
// GEMV and the packed-dtype branch working off the same buffer.
// DEFAULT ON, on measured evidence.  Real model, UD-Q8_K_XL, 2x B70, ctx 4096,
// identical run configuration A/B'd back to back (2026-08-03):
//
//   prefill pp128   43.46 -> 67.10 tok/s   1.54x
//   prefill pp512   48.71 -> 82.82 tok/s   1.70x
//   prefill pp2048  45.72 -> 75.24 tok/s   1.65x
//   decode  tg4096   3.79 ->  4.00 tok/s   1.05x
//
// The kernel-level win is larger than the wall-clock win and the difference is
// understood, not hand-waved: pp512 kprof.gpu_ms fell 14,517 -> 5,942 ms (2.44x)
// while dma_busy_frac ROSE 0.476 -> 0.797.  Expert streaming, not compute, is now
// what bounds prefill.  Optimising these kernels further is no longer the lever.
//
// Decode barely moves (1.05x) and that is expected: the microbenchmark suggested
// 11-43x at T=1, but it kept an 8 MB weight resident in L2 across its repetitions
// at an implied 840 GB/s — above this card's memory bandwidth.  Across 43 real
// layers nothing stays cached, so decode remains bandwidth-bound.
//
// Set IE_DS4_ONEDNN=0 to take the fp32-accumulate kernels instead; both routes
// are gated by deepseek4_dense_gate_test under their own derived tolerances.
bool ds4_onednn_on() {
    static const bool on = [] {
        const char* e = std::getenv("IE_DS4_ONEDNN");
        const bool want = !(e && *e) || std::string(e) != "0";
        return want && ie::onednn_available();
    }();
    return on;
}

// OPT-IN.  The T > 1 route over a REQUANTISED weight hands the Q8_0-SoA planes
// straight to oneDNN's weight-decompression matmul instead of expanding them to
// fp16 first.  Default OFF: it changes the numerics slightly (nothing is rounded
// to fp16 between the stored int8 and the multiply), and oneDNN will happily
// build a REFERENCE primitive for an operand combination it has no kernel for.
// Measure with `ie-ds4-dense-prefill-bench` — it prints oneDNN's own
// `impl_info_str` and gates the result against a double reference — and re-run
// `tools/ie-ds4-ppl` before considering this a default.
bool ds4_dense_s8_on() {
    static const bool on = [] {
        const char* e = std::getenv("IE_DS4_DENSE_S8");
        return e && *e && std::string(e) != "0" && ie::onednn_available();
    }();
    return on;
}

// Decode (T == 1) routes the fp16 dense projections to a purpose-built GEMV
// instead of oneDNN.  Kill switch mirrors IE_DS4_ONEDNN so the old routing can
// be restored without a rebuild.  See include/ie/ds4_decode_gemv.hpp.
bool ds4_decgemv_on() {
    static const bool on = [] {
        const char* e = std::getenv("IE_DS4_DECGEMV");
        return !(e && *e) || std::string(e) != "0";
    }();
    return on;
}

// Every performance fallback in this file is a CLIFF, not a slope: each one
// replaces a route that reads a weight once with a route that reads it once per
// token or once per output row.  They were all silent.  That is how a 2.2x
// prefill regression got attributed to a materialisation pass whose own
// arithmetic says 23 ms/card/chunk.  One line per distinct message, on stderr,
// where the benchmark harness already captures it.
void ds4_warn_once(const std::string& msg) {
    static std::mutex               mu;
    static std::unordered_set<std::string> seen;
    std::lock_guard<std::mutex> lk(mu);
    if (seen.insert(msg).second)
        std::fprintf(stderr, "[ds4] SLOW PATH: %s\n", msg.c_str());
}

// One fp16 WEIGHT staging buffer per device, for the T > 1 route over a
// requantised weight.
//
// WHY MATERIALISE AT ALL.  The packed branch below is `dense_packed`, a scalar
// element-decode kernel, and at prefill it is the wrong kernel by a lot: oneDNN
// is worth 1.54-1.70x on real pp128/pp512/pp2048 and 2.44x on kernel time, and
// forcing a requantised weight down the scalar path would hand all of that back.
// Expanding Q8_0-SoA to the fp16 [N, K] that oneDNN already reads costs ONE
// extra pass over the weight per projection per chunk — about 19 ms/card/chunk
// against a measured 5,942 ms pp512, under 0.4% — and leaves the entire T > 1
// route, including its numerics, byte for byte what it was.
//
// It is a separate buffer from `ds4_a16_scratch` (activations) because both are
// live at once inside one `dense_w` call.  Sized by the largest per-card dense
// projection actually submitted at T > 1: `q_b`/`o_b`/`o_a` at 16.78 M elements
// = 33.6 MB.
//
// THE LM HEAD IS NOT RESERVED HERE AND THE OLD COMMENT SAYING IT COULD NOT BE
// WAS WRONG.  `forward_epilogue` passes `TL = last_only ? 1 : T`, so the head is
// a T == 1 call under generation and a T-token call under teacher-forced
// scoring (tools/ie-ds4-ppl, which needs a logit row per position).  The LOADER
// therefore reserves only the layer high-water — 33.6 MB — and the head grows
// this buffer on demand if and when a scoring run asks for it.  Reserving the
// head's 529 MB per card at load, which is what the code did, spent 495 MB of
// VRAM that generation never touches, and spent it WITHOUT adding to
// `resident_bytes_`, so the expert-arena derivation
// (`global_mem - resident - 2 GB`) did not know it was gone.  That margin is
// load-bearing: a run has already died at 32.6 GB of ~32.5 GB available.
//
// CONCURRENCY.  Two cards prepare and run on two host threads
// (`DeepSeek4TpRuntime::load`'s fan_out, and `run_cards` at every layer), and
// both reach this function.  `operator[]` on a shared std::unordered_map INSERTS
// — during the concurrent load both threads insert, which can rehash under the
// other thread's live reference.  The mutex is not decoration; without it the
// map is a data race on the load path of every two-card run.
sycl::half* ds4_scratch_grow(sycl::queue& q, uint64_t n, const char* what,
                             std::unordered_map<sycl::device, std::pair<sycl::half*, uint64_t>>& bufs,
                             std::mutex& mu) {
    std::lock_guard<std::mutex> lk(mu);
    std::pair<sycl::half*, uint64_t>& b = bufs[q.get_device()];
    if (b.second < n) {
        if (b.first) { q.wait(); sycl::free(b.first, q); b.first = nullptr; b.second = 0; }
        b.first  = sycl::malloc_device<sycl::half>(n, q);
        b.second = b.first ? n : 0;
        // A null here is not a slow path, it is a 100x one: every caller of this
        // function falls back to a route that re-reads the weight once per token
        // or once per token-tile.  It cost this project a day — prefill halved
        // and the regression was attributed to a materialisation pass that the
        // arithmetic says is 23 ms.  It says so now.
        // Once per message, not once per call: this is reached from every dense
        // projection of every layer, so an unconditional print would bury the
        // benchmark output it needs to appear in.
        if (!b.first)
            ds4_warn_once(std::string("could not allocate the ") + what +
                          " staging buffer (" + std::to_string(n * sizeof(sycl::half)) +
                          " B) — the T>1 dense route falls back to a per-token or "
                          "per-output-row kernel. This is a VRAM shortage, not a "
                          "numerical failure.");
    }
    return b.first;
}

sycl::half* ds4_w16_scratch(sycl::queue& q, uint64_t n) {
    static std::unordered_map<sycl::device, std::pair<sycl::half*, uint64_t>> bufs;
    static std::mutex mu;
    return ds4_scratch_grow(q, n, "fp16 weight", bufs, mu);
}

// One fp16 activation staging buffer per device, grown on demand and reused.
// Per-device rather than per-runtime because the test seams have no runtime; the
// forward path and the gate therefore exercise the identical code.  Growth is
// rare (the largest projection wins within the first layer) and waits before
// freeing, since in-flight work may still reference the old pointer.  Same
// locking argument as above.
sycl::half* ds4_a16_scratch(sycl::queue& q, uint64_t n) {
    static std::unordered_map<sycl::device, std::pair<sycl::half*, uint64_t>> bufs;
    static std::mutex mu;
    return ds4_scratch_grow(q, n, "fp16 activation", bufs, mu);
}

// oneDNN submits internally, so ie::ps() cannot wrap it.  Push the returned
// event under the same site-tagged name the scalar kernel would have used, or
// --kprofile-prefill goes blind on 66% of the pass exactly when it is needed to
// judge what is left.
inline sycl::event ds4_pe(const char* name, sycl::event e) {
    if (ie::g_profiler) [[unlikely]] ie::g_profiler->push(name, e);
    return e;
}

// The two dispatchers every call site goes through.  Shape comes from the
// Ds4Dense itself so a call site can no longer pass a K/N that disagrees with
// the weight it names.
// The fp16 tail of `dense_w`, hoisted so that a weight MATERIALISED from the
// Q8_0-SoA form takes the identical route — same decode-GEMV/oneDNN/scalar
// selection, same TT, same profiler name.  Hoisting rather than duplicating is
// what keeps "prefill over a requantised weight" and "prefill over an fp16
// weight" from being two code paths that can drift.
sycl::event dense_f16_route(sycl::queue& q, const float* x, const sycl::half* w, float* y,
                            uint32_t T, uint32_t K, uint32_t N,
                            const std::vector<sycl::event>& deps) {
    if (T == 1 && ds4_decgemv_on())
        return ds4_decode_gemv_f16(q, ds4_kname(kNmDenseF16), x, w, y, K, N, deps);
    if (ds4_onednn_on()) {
        const uint64_t an = uint64_t(T) * K;
        if (sycl::half* a16 = ds4_a16_scratch(q, an)) {
            auto e = cast_fp32_to_fp16(q, x, a16, an, deps);
            return ds4_pe(ds4_kname(kNmDenseF16), gemm_nt_f16_onednn(q, a16, w, y, T, N, K, {e}));
        }
        // Scratch allocation failed — fall through to the kernel that needs none
        // rather than proceeding with a null operand.  `ds4_scratch_grow` has
        // already said so on stderr; what follows is not a mild slowdown.
        // `dense_f16` re-reads the ACTIVATION once per output row, so at the real
        // per-card set it moves T*4*sum(N*K) = 6,986 GB per chunk out of L2
        // against oneDNN's 6.9 GB out of VRAM.  It is the difference between a
        // 4.4 s prefill and a 9.7 s one.
        ds4_warn_once("dense: oneDNN route unavailable at T>1 (no fp16 activation "
                      "buffer) — falling back to the scalar dense_f16 kernel, which "
                      "re-reads the activation once per output row");
    }
    return T <= kTTSmall ? dense_f16<kTTSmall>(q, x, w, y, T, K, N, deps)
                         : dense_f16<kTTLarge>(q, x, w, y, T, K, N, deps);
}

sycl::event dense_w(sycl::queue& q, const float* x, const Ds4Dense& d, float* y, uint32_t T,
                    const std::vector<sycl::event>& deps = {}) {
    if (d.p) {
        const uint64_t rb = ds4_dense_row_bytes(d.dt, d.K);
        if (d.dt == DType::kQ8_0)
            return T <= kTTSmall
                       ? dense_packed<DType::kQ8_0, kTTSmall>(q, x, d.p, y, T, d.K, d.N, rb, deps)
                       : dense_packed<DType::kQ8_0, kTTLarge>(q, x, d.p, y, T, d.K, d.N, rb, deps);
        return T <= kTTSmall
                   ? dense_packed<DType::kQ6_K, kTTSmall>(q, x, d.p, y, T, d.K, d.N, rb, deps)
                   : dense_packed<DType::kQ6_K, kTTLarge>(q, x, d.p, y, T, d.K, d.N, rb, deps);
    }
    // REQUANTISED (Q8_0-SoA).  Decode is the whole point — it reads 1.0625
    // B/element instead of 2 and measures 1.72x faster over the real per-card
    // call set — so T == 1 goes straight to the Q8 GEMV.  Everything else
    // materialises fp16 and rejoins the route above, because that route's oneDNN
    // step is worth 1.54-1.70x on prefill and a scalar packed kernel is not.
    if (d.q8) {
        if (T == 1 && ds4_decgemv_on())
            return ds4_decode_gemv_q8(q, ds4_kname(kNmDenseQ8), x, d.q8, d.q8d, y, d.K, d.N,
                                      deps);
        // NO-MATERIALISE ROUTE (opt-in, $IE_DS4_DENSE_S8=1).  oneDNN reads the
        // int8 plane and the fp16 scale plane where they lie — the same "describe
        // the buffer with strides" trick the fp16 route already uses, applied to
        // both planes — so the extra pass over every weight per chunk disappears
        // instead of being made cheaper.  It is DEFAULT OFF because it is a
        // numerical change (oneDNN decompresses and multiplies rather than
        // reading a value already rounded to fp16) and because oneDNN ACCEPTS
        // operand combinations it has no jitted kernel for and silently
        // dispatches a reference one.  `ie-ds4-dense-prefill-bench` prints the
        // implementation oneDNN chose and gates the numerics against a double
        // reference with three negative controls; turn this on when that says so,
        // not before.
        if (T > 1 && ds4_dense_s8_on() && d.q8dt) {
            sycl::event ev;
            const uint64_t an = uint64_t(T) * d.K;
            if (sycl::half* a16 = ds4_a16_scratch(q, an)) {
                auto e = cast_fp32_to_fp16(q, x, a16, an, deps);
                // d.q8dt, NOT d.q8d: oneDNN reads the attr scales in plain
                // kb-major layout.  The n-major plane here was the entire
                // numerics failure this route shipped with.
                if (gemm_nt_s8_onednn(q, a16, d.q8, d.q8dt, y, T, d.N, d.K, 32, {e}, &ev))
                    return ds4_pe(ds4_kname(kNmDenseQ8), ev);
                // No primitive for this shape.  `e` is already submitted and the
                // routes below re-derive `a16` themselves, so it is a dependency,
                // not a leak.
                ds4_warn_once("dense: IE_DS4_DENSE_S8=1 but oneDNN built no s8 "
                              "weight-decompression primitive for some shape — that "
                              "shape falls back to materialise+oneDNN");
            }
        }
        if (sycl::half* w16 = ds4_w16_scratch(q, uint64_t(d.K) * d.N)) {
            auto e = ds4_q8_soa_to_f16(q, d.q8, d.q8d, w16, d.K, d.N, deps);
            return dense_f16_route(q, x, w16, y, T, d.K, d.N, {e});
        }
        // No fp16 form could be allocated and there is no fp16 weight to fall
        // back to (`d.w` is null whenever `d.q8` is set).  Run the GEMV once per
        // token rather than dispatch a null operand: correct, and catastrophic.
        //
        // "Unreachable in practice because load_prepare reserves this buffer" is
        // what the comment used to say, and it is an ASSUMPTION, not a check —
        // the reservation is per-weight and the buffer is shared and grown, so a
        // later grow that fails leaves every subsequent call here.  T = 512 means
        // 512 launches per projection each re-reading the whole weight: 1,857 GB
        // per card per chunk against the 3.9 GB the intended route reads.  If
        // this line ever runs, it is the whole regression, and it now says so.
        ds4_warn_once("dense: no fp16 weight staging buffer at T>1 over a requantised "
                      "weight — running ONE GEMV PER TOKEN, which re-reads the entire "
                      "weight T times. Expect prefill to collapse.");
        sycl::event ev;
        for (uint32_t t = 0; t < T; ++t)
            ev = ds4_decode_gemv_q8(q, ds4_kname(kNmDenseQ8), x + uint64_t(t) * d.K, d.q8,
                                    d.q8d, y + uint64_t(t) * d.N, d.K, d.N,
                                    t == 0 ? deps : std::vector<sycl::event>{});
        return ev;
    }
    // T == 1 is a GEMV, not a GEMM.  M=1 makes the entire cost reading the weight
    // once, and oneDNN's fp16 matmul additionally needs an fp16 A operand, so its
    // route pays a cast_fp32_to_fp16 LAUNCH per projection — 510 of them per token
    // per card.  Replaying the exact per-token call sequence cache-cold on a B70:
    // oneDNN 15.44 ms/card, this GEMV 12.78 ms/card, dense_f16<4> 26.89 ms/card,
    // against an 11.73 ms/card streaming-bandwidth floor.  It is also strictly
    // MORE accurate, because the activation is never rounded to fp16.
    // Gated by deepseek4_decgemv_gate_test.
    //
    // AFTER the packed and requantised branches and BEFORE oneDNN, and every part
    // of that is load-bearing: `d.w` is null whenever `d.p` or `d.q8` is set, so
    // those weights must have returned already, and the oneDNN route inside
    // `dense_f16_route` is the one the T == 1 GEMV replaces.
    return dense_f16_route(q, x, d.w, y, T, d.K, d.N, deps);
}

// Decode-only: several dense projections that share ONE activation and ONE K, in
// ONE launch.  See include/ie/ds4_decode_gemv.hpp for why, and
// deepseek4_decgemv_gate_test's multi-projection section for the proof that it is
// BIT-IDENTICAL to the per-projection `dense_w` calls it replaces (each
// projection keeps its own sgs/V/rows and therefore its own reduction shape).
//
// Returns false having submitted NOTHING for any set it cannot carry, so a call
// site can fall back to the loop it replaced without a second eligibility test.
// Refused: T != 1, kill switch off, fewer than two projections, a packed-GGUF
// weight (`d.p` has no [N, K] form), a disagreeing K, or an over-long list.
bool dense_multi_w(sycl::queue& q, const float* x, const Ds4Dense* const* ds,
                   float* const* ys, uint32_t n, uint32_t T) {
    if (T != 1 || !ds4_decgemv_on() || n < 2 || n > kDs4GemvMultiMax) return false;
    const uint32_t K = ds[0]->K;
    Ds4GemvProj pr[kDs4GemvMultiMax];
    for (uint32_t j = 0; j < n; ++j) {
        const Ds4Dense& d = *ds[j];
        if (d.p || d.K != K || (!d.w && !d.q8)) return false;
        pr[j] = Ds4GemvProj{};
        if (d.q8) { pr[j].qs = d.q8; pr[j].qd = d.q8d; }
        else      { pr[j].w  = d.w; }
        pr[j].y = ys[j];
        pr[j].N = d.N;
    }
    ds4_decode_gemv_multi(q, ds4_kname(ds[0]->q8 ? kNmDenseQ8 : kNmDenseF16),
                          x, pr, n, K);
    return true;
}

// `dense_f16_route`'s block-diagonal sibling, hoisted for the same reason.
sycl::event grouped_f16_route(sycl::queue& q, const float* x, const sycl::half* w, float* y,
                              uint32_t T, uint32_t G, uint32_t IPG, uint32_t OPG,
                              const std::vector<sycl::event>& deps) {
    if (T == 1 && ds4_decgemv_on())
        return ds4_decode_gemv_grouped_f16(q, ds4_kname(kNmGroupedF16), x, w, y,
                                           G, IPG, OPG, deps);
    if (ds4_onednn_on()) {
        const uint64_t an = uint64_t(T) * G * IPG;
        if (sycl::half* a16 = ds4_a16_scratch(q, an)) {
            auto e = cast_fp32_to_fp16(q, x, a16, an, deps);
            return ds4_pe(ds4_kname(kNmGroupedF16),
                          gemm_bmm_nt_f16_onednn(q, a16, w, y, T, G, IPG, OPG, {e}));
        }
        // The batched matmul is worth 39.11 -> 0.305 ms at the real o_a shape,
        // i.e. 128x, precisely because it stops re-reading the [T, G*IPG]
        // activation slab once per output row.  Losing it silently is not an
        // option.
        ds4_warn_once("grouped: oneDNN batched route unavailable at T>1 (no fp16 "
                      "activation buffer) — falling back to grouped_f16, measured "
                      "128x slower at the real o_a shape");
    }
    return grouped_f16<kTTSmall>(q, x, w, y, T, G, IPG, OPG, deps);
}

// The grouped output projection walks rows of length IPG; `attn_output_a` is
// [K = n_q_heads*head_dim/o_groups... ] — i.e. IPG == d.K for this one weight,
// which is why the packed stride is taken from d.K (the file's row length) while
// the loop bound stays IPG.
sycl::event grouped_w(sycl::queue& q, const float* x, const Ds4Dense& d, float* y,
                      uint32_t T, uint32_t G, uint32_t IPG, uint32_t OPG,
                      const std::vector<sycl::event>& deps = {}) {
    // NOTE: the grouped projection deliberately stays on kTTSmall.  Widening the
    // tile is what dense_w does and it is bit-identical here too, but it buys
    // NOTHING measurable: at the real per-card o-projection shape (G=4,
    // IPG=4096, OPG=1024, T=512) TT=4 and TT=16 measured 46.0 ms and 46.5 ms.
    // The reason is that this kernel is activation-bound, not weight-bound —
    // its activation slab is [T, G*IPG] = 33.5 MB at T=512, far past L2, so the
    // N-fold activation re-read (which no token tile shrinks) dominates, while
    // dense_w's activation is [T, K] = 8.4 MB and stays cached so its weight
    // traffic is what shows.  Changing it here would be an unmeasured change.
    if (d.p) {
        const uint64_t rb = ds4_dense_row_bytes(d.dt, d.K);
        if (d.dt == DType::kQ8_0)
            return grouped_packed<DType::kQ8_0, kTTSmall>(q, x, d.p, y, T, G, IPG, OPG, rb, deps);
        return grouped_packed<DType::kQ6_K, kTTSmall>(q, x, d.p, y, T, G, IPG, OPG, rb, deps);
    }
    // The activation-bound diagnosis above is exactly what a batched matmul
    // fixes: oneDNN reads the [T, G*IPG] slab once per group tile instead of
    // once per output row, so the term that made the token tile useless is the
    // term that goes away.  Measured 39.11 ms -> 0.305 ms at the real shape.
    //
    // Same reasoning as dense_w at T == 1: the batched matmul's win is an
    // activation-reuse win that only exists for T > 1.  At T = 1 there is one
    // activation vector and the cost is the weight stream, which this GEMV runs
    // at 59.20 us vs oneDNN's 64.01 us at the real per-card o_a shape.
    //
    // The requantised form is the same story with 47% fewer bytes: 32.84 us at
    // that shape.  T > 1 materialises fp16 and rejoins the route below, which is
    // where the batched matmul lives.
    if (d.q8) {
        if (T == 1 && ds4_decgemv_on())
            return ds4_decode_gemv_grouped_q8(q, ds4_kname(kNmGroupedQ8), x, d.q8, d.q8d, y,
                                              G, IPG, OPG, deps);
        if (sycl::half* w16 = ds4_w16_scratch(q, uint64_t(d.K) * d.N)) {
            auto e = ds4_q8_soa_to_f16(q, d.q8, d.q8d, w16, d.K, d.N, deps);
            return grouped_f16_route(q, x, w16, y, T, G, IPG, OPG, {e});
        }
        // See dense_w: correct, and catastrophic.  x is [T, G*IPG] and y is
        // [T, G*OPG].  "Unreachable once load_prepare has reserved the buffer" is
        // an assumption about a SHARED, GROWN buffer, not a check.
        ds4_warn_once("grouped: no fp16 weight staging buffer at T>1 over a requantised "
                      "weight — running ONE GEMV PER TOKEN. Expect prefill to collapse.");
        sycl::event ev;
        for (uint32_t t = 0; t < T; ++t)
            ev = ds4_decode_gemv_grouped_q8(q, ds4_kname(kNmGroupedQ8),
                                            x + uint64_t(t) * G * IPG, d.q8, d.q8d,
                                            y + uint64_t(t) * G * OPG, G, IPG, OPG,
                                            t == 0 ? deps : std::vector<sycl::event>{});
        return ev;
    }
    return grouped_f16_route(q, x, d.w, y, T, G, IPG, OPG, deps);
}

// streams[t, h, :] = src[t, :]   (DeepseekV4Model: `inputs_embeds.unsqueeze(2)
// .expand(-1, -1, hc_mult, -1)` — every stream starts as a COPY of the
// embedding, modeling_deepseek_v4.py:1320).
sycl::event expand_streams(sycl::queue& q, const float* src, float* streams,
                           uint32_t T, uint32_t H, uint32_t hc,
                           const std::vector<sycl::event>& deps = {}) {
    const uint64_t n = uint64_t(T) * hc * H;
    constexpr uint32_t WG = 256;
    return ie::ps(q, "ds4_expand_streams", [&](sycl::handler& h_) {
        h_.depends_on(deps);
        h_.parallel_for(sycl::nd_range<1>((n + WG - 1) / WG * WG, WG), [=](sycl::nd_item<1> it) {
            const uint64_t i = it.get_global_id(0);
            if (i >= n) return;
            const uint64_t t = i / (uint64_t(hc) * H);
            const uint64_t d = i % H;
            streams[i] = src[t * H + d];
        });
    });
}

sycl::event gather_embd(sycl::queue& q, const int32_t* ids, const sycl::half* table,
                        float* y, uint32_t T, uint32_t H,
                        const std::vector<sycl::event>& deps = {}) {
    const uint64_t n = uint64_t(T) * H;
    constexpr uint32_t WG = 256;
    return ie::ps(q, "ds4_gather_embd", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>((n + WG - 1) / WG * WG, WG), [=](sycl::nd_item<1> it) {
            const uint64_t i = it.get_global_id(0);
            if (i >= n) return;
            y[i] = float(table[uint64_t(ids[i / H]) * H + (i % H)]);
        });
    });
}

// Same gather over a table that stayed in its packed GGUF form.
template <DType DT>
sycl::event gather_embd_packed(sycl::queue& q, const int32_t* ids, const uint8_t* table,
                               float* y, uint32_t T, uint32_t H, uint64_t row_bytes,
                               const std::vector<sycl::event>& deps = {}) {
    const uint64_t n = uint64_t(T) * H;
    constexpr uint32_t WG = 256;
    return ie::ps(q, "ds4_gather_embd_packed", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>((n + WG - 1) / WG * WG, WG), [=](sycl::nd_item<1> it) {
            const uint64_t i = it.get_global_id(0);
            if (i >= n) return;
            y[i] = packed_at<DT>(table + uint64_t(ids[i / H]) * row_bytes, uint32_t(i % H));
        });
    });
}

sycl::event gather_embd_w(sycl::queue& q, const int32_t* ids, const Ds4Dense& d,
                          float* y, uint32_t T, uint32_t H,
                          const std::vector<sycl::event>& deps = {}) {
    if (d.p) {
        const uint64_t rb = ds4_dense_row_bytes(d.dt, d.K);
        if (d.dt == DType::kQ8_0)
            return gather_embd_packed<DType::kQ8_0>(q, ids, d.p, y, T, H, rb, deps);
        return gather_embd_packed<DType::kQ6_K>(q, ids, d.p, y, T, H, rb, deps);
    }
    return gather_embd(q, ids, d.w, y, T, H, deps);
}

sycl::event add_f32(sycl::queue& q, const float* a, const float* b, float* y, uint64_t n,
                    const std::vector<sycl::event>& deps = {}) {
    constexpr uint32_t WG = 256;
    return ie::ps(q, "ds4_add_f32", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>((n + WG - 1) / WG * WG, WG), [=](sycl::nd_item<1> it) {
            const uint64_t i = it.get_global_id(0);
            if (i < n) y[i] = a[i] + b[i];
        });
    });
}

// y[i] += w * float(x[i]) — the routed-expert accumulate.  fp32 accumulation of
// the fp16 down-projection output; the routing weight is applied HERE, after
// down, exactly as `DeepseekV4Experts.forward` does.
sycl::event accum_f16(sycl::queue& q, const sycl::half* x, float w, float* y, uint32_t n,
                      const std::vector<sycl::event>& deps = {}) {
    constexpr uint32_t WG = 256;
    return ie::ps(q, "ds4_accum_f16", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>((n + WG - 1) / WG * WG, WG), [=](sycl::nd_item<1> it) {
            const uint64_t i = it.get_global_id(0);
            if (i < n) y[i] += w * float(x[i]);
        });
    });
}

// Concatenates the sliding mask [T, n_sl] and the compressed block bias
// [T, n_c] onto one [T, n_sl + n_c] additive mask — the reference's
// `torch.cat([attention_mask, block_bias], dim=-1)` (modeling:842).
sycl::event concat_mask(sycl::queue& q, const float* sl, const float* bias, float* out,
                        uint32_t T, uint32_t n_sl, uint32_t n_c,
                        const std::vector<sycl::event>& deps = {}) {
    const uint32_t n_kv = n_sl + n_c;
    const uint64_t n    = uint64_t(T) * n_kv;
    constexpr uint32_t WG = 256;
    return ie::ps(q, "ds4_concat_mask", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>((n + WG - 1) / WG * WG, WG), [=](sycl::nd_item<1> it) {
            const uint64_t i = it.get_global_id(0);
            if (i >= n) return;
            const uint32_t t = uint32_t(i / n_kv), k = uint32_t(i % n_kv);
            out[i] = (k < n_sl) ? sl[uint64_t(t) * n_sl + k]
                                : bias[uint64_t(t) * n_c + (k - n_sl)];
        });
    });
}

}  // namespace

// ---------------------------------------------------------------------------
// DeepseekV4HyperHead (modeling_deepseek_v4.py:955)
// ---------------------------------------------------------------------------
sycl::event ds4_hyper_head(sycl::queue& q,
                           const float* x, const float* hc_fn, const float* hc_base,
                           const float* hc_scale, float* y,
                           uint32_t n_tokens, uint32_t hidden, uint32_t hc_mult,
                           float rms_eps, float hc_eps,
                           const std::vector<sycl::event>& deps) {
    constexpr uint32_t WG = 256;
    const uint32_t flat = hc_mult * hidden;
    return ie::ps(q, "ds4_hyper_head", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<float, 1> pre(sycl::range<1>(16), h);
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_tokens) * WG, WG), [=](sycl::nd_item<1> it) {
            const uint32_t t   = uint32_t(it.get_group(0));
            const uint32_t lid = uint32_t(it.get_local_id(0));
            auto grp = it.get_group();
            const float* xs = x + uint64_t(t) * flat;

            // input_norm = DeepseekV4UnweightedRMSNorm over the flattened streams.
            float ss = 0.f;
            for (uint32_t i = lid; i < flat; i += WG) ss = sycl::fma(xs[i], xs[i], ss);
            ss = sycl::reduce_over_group(grp, ss, sycl::plus<float>());
            const float inv = sycl::rsqrt(ss / float(flat) + rms_eps);

            const float sc = hc_scale[0];
            for (uint32_t m = 0; m < hc_mult; ++m) {
                const float* fr = hc_fn + uint64_t(m) * flat;
                float acc = 0.f;
                for (uint32_t i = lid; i < flat; i += WG) acc = sycl::fma(fr[i], xs[i] * inv, acc);
                acc = sycl::reduce_over_group(grp, acc, sycl::plus<float>());
                if (lid == 0) pre[m] = 1.f / (1.f + sycl::exp(-(acc * sc + hc_base[m]))) + hc_eps;
            }
            sycl::group_barrier(grp);

            float* ys = y + uint64_t(t) * hidden;
            for (uint32_t d = lid; d < hidden; d += WG) {
                float s = 0.f;
                for (uint32_t m = 0; m < hc_mult; ++m) s = sycl::fma(pre[m], xs[uint64_t(m) * hidden + d], s);
                ys[d] = s;
            }
        });
    });
}

// ---------------------------------------------------------------------------
// DecoderLayer residual mix (modeling_deepseek_v4.py:1135 / 1141)
// ---------------------------------------------------------------------------
sycl::event ds4_hc_mix(sycl::queue& q,
                       const float* streams, const float* post, const float* comb,
                       const float* sub, float* out,
                       uint32_t n_tokens, uint32_t hidden, uint32_t hc_mult,
                       const std::vector<sycl::event>& deps) {
    const uint64_t rows = uint64_t(n_tokens) * hc_mult;
    constexpr uint64_t max_elements = std::numeric_limits<size_t>::max() / sizeof(float);
    if ((hidden && rows > max_elements / hidden)
        || (hc_mult && rows > max_elements / hc_mult))
        throw std::invalid_argument("ds4_hc_mix: tensor byte span exceeds addressable storage");
    const uint64_t n = rows * hidden;
    constexpr uint32_t WG = 256;
    if (hc_mult == 4) {
        return ie::ps(q, "ds4_hc_mix", [&](sycl::handler& h) {
            h.depends_on(deps);
            const size_t columns = (size_t(hidden) + WG - 1) / WG * WG;
            h.parallel_for(sycl::nd_range<2>({n_tokens, columns}, {1, WG}),
                           [=](sycl::nd_item<2> it) {
                const size_t t = it.get_group(0), d = it.get_global_id(1);
                if (d >= hidden) return;
                float sums[4] = {};
                #pragma unroll
                for (uint32_t j = 0; j < 4; ++j) {
                    const float x = streams[(t * 4 + j) * hidden + d];
                    #pragma unroll
                    for (uint32_t hh = 0; hh < 4; ++hh)
                        sums[hh] = sycl::fma(comb[t * 16 + j * 4 + hh], x, sums[hh]);
                }
                const float value = sub[t * hidden + d];
                // All four original streams have been consumed before any
                // write. One work-item owns this feature across every stream,
                // so exact in-place operation cannot race another work-item.
                #pragma unroll
                for (uint32_t hh = 0; hh < 4; ++hh)
                    out[(t * 4 + hh) * hidden + d] =
                        sycl::fma(post[t * 4 + hh], value, sums[hh]);
            });
        });
    }
    if (out == streams && n != 0) {
        // The generic kernel assigns different output streams to different
        // work-items. Snapshot via a separate output to honor exact aliasing.
        // This cold fallback waits for copy-back before freeing its temporary;
        // the model's disjoint buffers and hc=4 path remain asynchronous.
        const auto context = q.get_context();
        float* temporary = sycl::malloc_device<float>(size_t(n), q);
        if (!temporary) throw std::bad_alloc();
        sycl::event last;
        try {
            last = ds4_hc_mix(q, streams, post, comb, sub, temporary,
                              n_tokens, hidden, hc_mult, deps);
            last = q.submit([&](sycl::handler& h) {
                h.depends_on(last);
                h.memcpy(out, temporary, size_t(n) * sizeof(float));
            });
            last.wait_and_throw();
        } catch (...) {
            const auto error = std::current_exception();
            try { last.wait(); } catch (...) {}
            try { sycl::free(temporary, context); } catch (...) {}
            std::rethrow_exception(error);
        }
        sycl::free(temporary, context);
        return last;
    }
    return ie::ps(q, "ds4_hc_mix", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>((n + WG - 1) / WG * WG, WG), [=](sycl::nd_item<1> it) {
            const uint64_t i = it.get_global_id(0);
            if (i >= n) return;
            const uint64_t d  = i % hidden;
            const uint64_t hh = (i / hidden) % hc_mult;
            const uint64_t t  = i / (uint64_t(hc_mult) * hidden);
            // comb is consumed TRANSPOSED: Σ_j comb[t, j, h] · streams[t, j, d].
            float s = 0.f;
            for (uint32_t j = 0; j < hc_mult; ++j)
                s = sycl::fma(comb[t * hc_mult * hc_mult + uint64_t(j) * hc_mult + hh],
                              streams[t * hc_mult * hidden + uint64_t(j) * hidden + d], s);
            out[i] = sycl::fma(post[t * hc_mult + hh], sub[t * hidden + d], s);
        });
    });
}

// ===========================================================================
// DeepSeek4Runtime
// ===========================================================================
namespace {

// Dequantises one GGUF row-major [K, N] tensor (N rows of K elements) into a
// host fp32 scratch, row block by row block.  Reads THIS tensor's own dtype.
std::string dequant_rows(const GgufTensorInfo& ti, uint32_t K, uint64_t row0, uint32_t nrows,
                         std::vector<float>& out) {
    const size_t row_bytes = bytes_for(ti.dtype, K);
    const auto*  base      = static_cast<const uint8_t*>(ti.data) + row0 * row_bytes;
    out.resize(size_t(nrows) * K);
    for (uint32_t r = 0; r < nrows; ++r) {
        const void* src = base + size_t(r) * row_bytes;
        float*      dst = out.data() + size_t(r) * K;
        switch (ti.dtype) {
            case DType::kF32:  std::memcpy(dst, src, size_t(K) * 4); break;
            case DType::kF16:
                for (uint32_t i = 0; i < K; ++i)
                    dst[i] = fp16_to_fp32(static_cast<const uint16_t*>(src)[i]);
                break;
            case DType::kBF16:
                for (uint32_t i = 0; i < K; ++i) {
                    const uint32_t b = uint32_t(static_cast<const uint16_t*>(src)[i]) << 16;
                    float f; std::memcpy(&f, &b, 4); dst[i] = f;
                }
                break;
            case DType::kQ8_0: ref::dequant_q8_0_buffer(src, K, dst); break;
            case DType::kQ6_K: ref::dequant_q6_K_buffer(src, K, dst); break;
            case DType::kQ4_K: ref::dequant_q4_K_buffer(src, K, dst); break;
            case DType::kMXFP4:
            case DType::kIQ3_XXS:
            default:
                return std::string(ti.name) + ": dtype " + std::string(type_name(ti.dtype)) +
                       " has no always-resident dequant path (expert dtypes stream instead)";
        }
    }
    return {};
}

}  // namespace

// Allocates and fills the two Q8_0-SoA planes for THIS CARD's [Ks, Ns] slice of
// a [K, N] weight, pulling rows from `next_rows(r0, nr, out)` in chunks.
//
// `next_rows` yields `nr` FULL rows of K fp32 values starting at this card's row
// r0 — i.e. it owns the row (N) slicing, and this function owns the column (K)
// slicing.  The split is not cosmetic: a column slice has to be gathered out of
// the full row BEFORE the block quantiser sees it, because a Q8_0 scale covers
// 32 consecutive elements OF THIS CARD'S ROW, and quantising the whole row and
// then cutting would hand this card blocks whose scale was set by elements
// another card holds.
std::string ds4_build_q8_soa(sycl::queue& q, uint32_t K, uint32_t K0, uint32_t Ks, uint32_t Ns,
                             Ds4Dense& d, uint64_t& bytes,
                             const std::function<std::string(uint32_t, uint32_t,
                                                             std::vector<float>&)>& next_rows) {
    if (!ds4_dense_requantises(DType::kBF16, Ks))
        return "K slice " + std::to_string(Ks) +
               " is not a whole number of 32-element Q8_0 blocks";
    const uint64_t want = ds4_dense_requant_bytes(Ks, Ns);
    d.dt = DType::kQ8_0;
    d.K = Ks; d.N = Ns;
    d.q8 = static_cast<int8_t*>(sycl::malloc_device(want, q));
    if (!d.q8) return "malloc_device failed (" + std::to_string(want) + " B, requantised)";
    // Both scale planes live in the SAME allocation, immediately after the int8
    // plane: first the n-major [Ns][nbs] plane the decode GEMV reads, then the
    // kb-major [nbs][Ns] plane oneDNN's s8 route reads.  Ks*Ns is even (Ks is a
    // multiple of 32), so the half planes are 2 B aligned, which is all a half
    // needs; only `d.q8` is ever freed.
    d.q8d  = reinterpret_cast<sycl::half*>(d.q8 + uint64_t(Ks) * Ns);
    d.q8dt = d.q8d + uint64_t(Ns) * (Ks / 32);
    bytes = want;

    std::vector<float>      f32, slice;
    std::vector<int8_t>     qs;
    std::vector<sycl::half> sc;
    const uint32_t nbs   = Ks / 32;
    // The kb-major plane needs every row's scales before any of it can be laid
    // out, so the per-chunk scales are also accumulated here and transposed
    // once after the loop.  Ns*nbs halves = 1/32 of the int8 plane — at the
    // largest per-card dense projection that is ~1 MB of host RAM.
    std::vector<sycl::half> sc_all(size_t(Ns) * nbs);
    const uint32_t chunk = std::max(1u, uint32_t((16u << 20) / std::max(1u, K)));
    for (uint32_t r0 = 0; r0 < Ns; r0 += chunk) {
        const uint32_t nr = std::min(chunk, Ns - r0);
        if (std::string s = next_rows(r0, nr, f32); !s.empty()) return s;
        if (f32.size() < size_t(nr) * K)
            return "row source produced " + std::to_string(f32.size()) + " floats for " +
                   std::to_string(nr) + " rows of " + std::to_string(K);
        const float* rows = f32.data();
        if (Ks != K) {
            slice.resize(size_t(nr) * Ks);
            for (uint32_t i = 0; i < nr; ++i)
                std::memcpy(slice.data() + size_t(i) * Ks, f32.data() + size_t(i) * K + K0,
                            size_t(Ks) * sizeof(float));
            rows = slice.data();
        }
        qs.resize(size_t(nr) * Ks);
        sc.resize(size_t(nr) * nbs);
        ds4_requantise_q8_soa(rows, Ks, nr, qs.data(), sc.data());
        q.memcpy(d.q8 + uint64_t(r0) * Ks, qs.data(), qs.size()).wait();
        q.memcpy(d.q8d + uint64_t(r0) * nbs, sc.data(), sc.size() * sizeof(sycl::half)).wait();
        std::memcpy(sc_all.data() + size_t(r0) * nbs, sc.data(),
                    sc.size() * sizeof(sycl::half));
    }
    std::vector<sycl::half> sct(size_t(Ns) * nbs);
    for (uint32_t n = 0; n < Ns; ++n)
        for (uint32_t b = 0; b < nbs; ++b)
            sct[size_t(b) * Ns + n] = sc_all[size_t(n) * nbs + b];
    q.memcpy(d.q8dt, sct.data(), sct.size() * sizeof(sycl::half)).wait();
    return {};
}

// ---- test seams: the real dispatchers, on caller-supplied packed bytes -----
std::string ds4_dense_packed_gemv(sycl::queue& q, const float* x, const void* w, DType dt,
                                  float* y, uint32_t T, uint32_t K, uint32_t N) {
    if (!ds4_dense_keeps_packed(dt, K))
        return std::string("ds4_dense_packed_gemv: ") + std::string(type_name(dt)) +
               " with K=" + std::to_string(K) + " is not a packed-residency dtype";
    Ds4Dense d;
    d.p  = static_cast<uint8_t*>(const_cast<void*>(w));
    d.dt = dt; d.K = K; d.N = N;
    dense_w(q, x, d, y, T).wait();
    return {};
}

std::string ds4_grouped_packed_gemv(sycl::queue& q, const float* x, const void* w, DType dt,
                                    float* y, uint32_t T, uint32_t G, uint32_t IPG,
                                    uint32_t OPG) {
    if (!ds4_dense_keeps_packed(dt, IPG))
        return std::string("ds4_grouped_packed_gemv: ") + std::string(type_name(dt)) +
               " with IPG=" + std::to_string(IPG) + " is not a packed-residency dtype";
    Ds4Dense d;
    d.p  = static_cast<uint8_t*>(const_cast<void*>(w));
    d.dt = dt; d.K = IPG; d.N = G * OPG;
    grouped_w(q, x, d, y, T, G, IPG, OPG).wait();
    return {};
}

// The fp16-branch seams.  `w` non-null and `p` null is exactly what upload_dense
// leaves behind for a BF16 weight, so these reach the same dispatcher — and
// therefore the same TT selection and the same oneDNN routing — that forward()
// uses.  No shape is inferred: the caller states K/N and the dispatcher reads
// them back off the Ds4Dense.
std::string ds4_dense_f16_gemv(sycl::queue& q, const float* x, const sycl::half* w,
                               float* y, uint32_t T, uint32_t K, uint32_t N) {
    if (!w) return "ds4_dense_f16_gemv: null weight";
    Ds4Dense d;
    d.w = const_cast<sycl::half*>(w);
    d.dt = DType::kF16; d.K = K; d.N = N;
    dense_w(q, x, d, y, T).wait();
    return {};
}

std::string ds4_grouped_f16_gemv(sycl::queue& q, const float* x, const sycl::half* w,
                                 float* y, uint32_t T, uint32_t G, uint32_t IPG,
                                 uint32_t OPG) {
    if (!w) return "ds4_grouped_f16_gemv: null weight";
    Ds4Dense d;
    d.w = const_cast<sycl::half*>(w);
    d.dt = DType::kF16; d.K = IPG; d.N = G * OPG;
    grouped_w(q, x, d, y, T, G, IPG, OPG).wait();
    return {};
}

std::string ds4_dense_q8_gemv(sycl::queue& q, const float* x, const int8_t* qs,
                              const sycl::half* dsc, float* y, uint32_t T, uint32_t K,
                              uint32_t N) {
    if (!qs || !dsc) return "ds4_dense_q8_gemv: null weight plane";
    if (K == 0 || K % 32) return "ds4_dense_q8_gemv: K=" + std::to_string(K) +
                                 " is not a whole number of 32-element Q8_0 blocks";
    Ds4Dense d;
    d.q8 = const_cast<int8_t*>(qs);
    d.q8d = const_cast<sycl::half*>(dsc);
    d.dt = DType::kQ8_0; d.K = K; d.N = N;
    dense_w(q, x, d, y, T).wait();
    return {};
}

std::string ds4_grouped_q8_gemv(sycl::queue& q, const float* x, const int8_t* qs,
                                const sycl::half* dsc, float* y, uint32_t T, uint32_t G,
                                uint32_t IPG, uint32_t OPG) {
    if (!qs || !dsc) return "ds4_grouped_q8_gemv: null weight plane";
    if (IPG == 0 || IPG % 32) return "ds4_grouped_q8_gemv: IPG=" + std::to_string(IPG) +
                                     " is not a whole number of 32-element Q8_0 blocks";
    Ds4Dense d;
    d.q8 = const_cast<int8_t*>(qs);
    d.q8d = const_cast<sycl::half*>(dsc);
    d.dt = DType::kQ8_0; d.K = IPG; d.N = G * OPG;
    grouped_w(q, x, d, y, T, G, IPG, OPG).wait();
    return {};
}

// Uploads one always-resident 2-D weight, PACKED when its dtype has an
// element-wise device decoder here and expanded to fp16 otherwise
// (ds4_dense_keeps_packed).  Shared by upload_layer and the two globals so the
// two cannot drift apart — they did, and that is how token_embd/output ended up
// costing 2.118 GB of fp16 for 0.997 GB of file bytes.
std::string DeepSeek4Runtime::upload_dense(const GgufTensorInfo* ti, Ds4Dense& d,
                                           Ds4DenseSplit split, Ds4DenseQuant quant,
                                           bool stage_at_load) {
    if (!ti) return "upload_dense: null tensor";
    if (ti->n_dims != 2) return std::string(ti->name) + ": expected a 2-D weight";
    // The whole tensor is about to be read front to back, either by one memcpy
    // or by dequant_rows in ascending row order.  Say so: without this the
    // reader's blanket MADV_RANDOM makes it one 4 kB fault at a time (see
    // ds4_stream_advise in expert_stream.hpp).
    ds4_stream_advise(ti->data, ti->nbytes);
    const uint32_t K = uint32_t(ti->shape[0]), N = uint32_t(ti->shape[1]);

    // ---- resolve this card's slice of [K, N] ------------------------------
    // `nc`/`cc` are 1/0 for a mirrored weight, so everything below is the
    // original code path byte for byte when nothing is split.
    const uint32_t nc = split == Ds4DenseSplit::kMirror ? 1u : aslice_.n_cards;
    const uint32_t cc = split == Ds4DenseSplit::kMirror ? 0u : aslice_.card;
    uint32_t Ks = K, Ns = N, K0 = 0, N0 = 0;
    if (split == Ds4DenseSplit::kRows) {
        if (nc == 0 || N % nc)
            return std::string(ti->name) + ": output dimension " + std::to_string(N) +
                   " is not divisible by " + std::to_string(nc) +
                   " cards — refusing to row-split it unevenly";
        Ns = N / nc; N0 = cc * Ns;
    } else if (split == Ds4DenseSplit::kCols) {
        if (nc == 0 || K % nc)
            return std::string(ti->name) + ": contraction dimension " + std::to_string(K) +
                   " is not divisible by " + std::to_string(nc) +
                   " cards — refusing to column-split it unevenly";
        Ks = K / nc; K0 = cc * Ks;
        // A COLUMN slice cuts INSIDE a row, so for a packed dtype it has to land
        // on a quantisation block boundary or the decoder would address a
        // half-block.  Rows are whole in every case, which is why kRows carries
        // no such check.
        if (ds4_dense_keeps_packed(ti->dtype, K) && !ds4_dense_keeps_packed(ti->dtype, Ks))
            return std::string(ti->name) + ": a " + std::to_string(nc) +
                   "-way column split of K=" + std::to_string(K) + " gives " +
                   std::to_string(Ks) + " elements per card, which is not a whole number of " +
                   std::string(type_name(ti->dtype)) +
                   " blocks — the split would land mid-block and the decode would be garbage";
    }
    d.K = Ks; d.N = Ns;

    // The requantised SoA route wins over the packed one when both apply (Q8_0
    // sources, 2026-09-11 — see ds4_dense_requantises); `quant` other than
    // kAuto and IE_DS4_DENSE_Q8=0 keep the packed path.
    const Ds4DenseRoute route = ds4_dense_route(ti->dtype, K, Ks, quant);
    const bool soa = route == Ds4DenseRoute::kRequantised;
    if (route == Ds4DenseRoute::kPacked) {
        // The packed path indexes rows as `base + n * row_bytes`, so the file's
        // own byte count must be exactly N whole rows.  A mismatch means the
        // tensor is not the row-major [N][K] this decoder assumes; refuse rather
        // than read a shifted row.
        const uint64_t rb  = ds4_dense_row_bytes(ti->dtype, K);
        const uint64_t rbs = ds4_dense_row_bytes(ti->dtype, Ks);   // this card's stride
        const uint64_t off = ds4_dense_row_bytes(ti->dtype, K0);   // byte offset of element K0
        if (ti->nbytes != rb * N)
            return std::string(ti->name) + ": packed size " + std::to_string(ti->nbytes) +
                   " B is not " + std::to_string(N) + " rows of " + std::to_string(rb) + " B";
        const uint64_t want = rbs * Ns;
        d.dt = ti->dtype;
        d.p  = static_cast<uint8_t*>(sycl::malloc_device(want, *q_));
        if (!d.p) return std::string(ti->name) + ": malloc_device failed (" +
                         std::to_string(want) + " B, packed)";
        resident_bytes_ += want;
        resident_saved_ += uint64_t(Ks) * Ns * 2 - want;
        if (rbs == rb) {
            // Mirrored or row-split: this card's rows are one contiguous run.
            q_->memcpy(d.p, static_cast<const uint8_t*>(ti->data) + uint64_t(N0) * rb, want)
                .wait();
        } else {
            // Column-split: gather the in-row sub-range of every row, in chunks,
            // so one H2D covers many rows instead of one per row.
            std::vector<uint8_t> stage;
            const uint32_t chunk = std::max(1u, uint32_t((16u << 20) / std::max<uint64_t>(rbs, 1)));
            for (uint32_t r0 = 0; r0 < Ns; r0 += chunk) {
                const uint32_t nr = std::min(chunk, Ns - r0);
                stage.resize(size_t(nr) * rbs);
                for (uint32_t i = 0; i < nr; ++i)
                    std::memcpy(stage.data() + size_t(i) * rbs,
                                static_cast<const uint8_t*>(ti->data) + uint64_t(r0 + i) * rb + off,
                                size_t(rbs));
                q_->memcpy(d.p + uint64_t(r0) * rbs, stage.data(), stage.size()).wait();
            }
        }
        return {};
    }
    // ---- requantise to Q8_0-SoA ------------------------------------------
    // Reached only for a dtype with no packed path (BF16) whose CALL SITE opted
    // in.  `Ks` is this card's row length, so a column split is checked against
    // the SLICE — a 2-way split of K=8192 gives 4096, still a whole number of
    // 32-element blocks, but the check is made, not assumed.
    if (soa) {
        uint64_t spent = 0;
        // The row source is a lambda so that this plumbing — the chunking, the
        // column gather, the two plane offsets — is ONE piece of code that the
        // gate drives directly (deepseek4_dense_gate_test §8).  Nothing on this
        // box loads a GGUF with BF16 dense tensors, so without the seam this
        // branch would ship with its arithmetic checked by nothing.
        if (std::string s = ds4_build_q8_soa(
                *q_, K, K0, Ks, Ns, d, spent,
                [&](uint32_t r0, uint32_t nr, std::vector<float>& out) {
                    return dequant_rows(*ti, K, N0 + r0, nr, out);
                });
            !s.empty())
            return std::string(ti->name) + ": " + s;
        resident_bytes_ += spent;
        resident_saved_ += uint64_t(Ks) * Ns * 2 - spent;
        // Every requantised weight the forward pass submits at T > 1 needs an
        // fp16 form.  Reserve the high-water mark HERE, while a failure is still
        // a named load error, so the forward pass can never be the place that
        // discovers it — except for the one weight that generation never submits
        // at T > 1 (see `stage_at_load`).
        if (!stage_at_load) return {};
        if (!ds4_w16_scratch(*q_, uint64_t(Ks) * Ns))
            return std::string(ti->name) + ": could not reserve the " +
                   std::to_string(uint64_t(Ks) * Ns * 2) +
                   " B fp16 staging buffer the T>1 route needs for a requantised weight";
        return {};
    }

    d.dt = DType::kF16;
    d.w  = sycl::malloc_device<sycl::half>(uint64_t(Ks) * Ns, *q_);
    if (!d.w) return std::string(ti->name) + ": malloc_device failed (" +
                     std::to_string(uint64_t(Ks) * Ns * 2) + " B)";
    resident_bytes_ += uint64_t(Ks) * Ns * 2;
    std::vector<float>      f32;
    std::vector<sycl::half> f16;
    const uint32_t chunk = std::max(1u, uint32_t((16u << 20) / std::max(1u, K)));
    for (uint32_t r0 = 0; r0 < Ns; r0 += chunk) {
        const uint32_t nr = std::min(chunk, Ns - r0);
        // Rows [N0+r0, N0+r0+nr) of the FILE; K0 selects the in-row range.  Both
        // are 0 unless this card holds a slice.
        if (std::string s = dequant_rows(*ti, K, N0 + r0, nr, f32); !s.empty()) return s;
        f16.resize(size_t(nr) * Ks);
        for (size_t i = 0; i < f16.size(); ++i) {
            const float v = f32[(i / Ks) * K + K0 + (i % Ks)];
            // The one thing fp16 cannot represent that BF16 and F32 can: an
            // exponent past 65504.  Such a weight would become inf and poison
            // every dot product it entered, so the load stops here naming the
            // tensor and the element rather than uploading it.  (Underflow is
            // NOT an error: a weight below fp16's 6.1e-5 normal minimum rounds
            // toward zero with absolute error under 6e-8, which is negligible
            // against the activation it multiplies.)
            if (std::isfinite(v) && std::fabs(v) > 65504.0f) {
                char buf[96];
                std::snprintf(buf, sizeof(buf), "%g", double(v));
                return std::string(ti->name) + ": element " +
                       std::to_string((uint64_t(N0 + r0) + i / Ks) * K + K0 + (i % Ks)) + " = " +
                       buf + " overflows fp16 (max 65504) — this weight cannot be expanded to fp16";
            }
            f16[i] = sycl::half(v);
        }
        q_->memcpy(d.w + uint64_t(r0) * Ks, f16.data(), f16.size() * sizeof(sycl::half)).wait();
    }
    return {};
}

std::string DeepSeek4Runtime::upload_layer(const DeepSeek4Layer& w, Ds4LayerRT& rt) {
    std::string e;
    auto dense = [&](const GgufTensorInfo* ti, Ds4Dense& d) -> std::string {
        return upload_dense(ti, d);
    };
    // The two split axes, resolved once.  Both collapse to kMirror when this
    // card holds the whole non-expert set, so the six calls below are the
    // original ones verbatim in that configuration.
    const Ds4DenseSplit R = aslice_.split ? Ds4DenseSplit::kRows : Ds4DenseSplit::kMirror;
    const Ds4DenseSplit C = aslice_.split ? Ds4DenseSplit::kCols : Ds4DenseSplit::kMirror;
    auto f32vec = [&](const GgufTensorInfo* ti, float*& p, uint64_t n) -> std::string {
        if (!ti) return "upload_layer: null tensor";
        p = sycl::malloc_device<float>(n, *q_);
        if (!p) return std::string(ti->name) + ": malloc_device failed";
        resident_bytes_ += n * 4;
        if (ti->dtype != DType::kF32) {
            // A role bound through `bind_f32_src` from a quantized file (the
            // ggml-org MXFP4 file stores the hc fns and compressor APEs as
            // Q8_0): dequantize on the host with the same bit-exact reference
            // the F32 row upload uses, then upload the F32 image.  The tensor
            // is contiguous rows of K/32 blocks, so it is one row of n here.
            std::vector<float> tmp;
            if (std::string e = dequant_rows(*ti, uint32_t(n), 0, 1, tmp); !e.empty()) return e;
            q_->memcpy(p, tmp.data(), n * 4).wait();
            return {};
        }
        ds4_stream_advise(ti->data, n * 4);
        q_->memcpy(p, ti->data, n * 4).wait();
        return {};
    };
    // A per-head F32 vector, of which this card takes only its own heads.
    // `attn_sinks` is the one such tensor; 256 B whole, so this is about
    // INDEXING (ds4_attention reads sinks[h] for h in [0, nhc)), not bytes.
    auto f32slice = [&](const GgufTensorInfo* ti, float*& p, uint64_t off,
                        uint64_t n) -> std::string {
        if (!ti) return "upload_layer: null tensor";
        p = sycl::malloc_device<float>(n, *q_);
        if (!p) return std::string(ti->name) + ": malloc_device failed";
        resident_bytes_ += n * 4;
        ds4_stream_advise(ti->data, (off + n) * 4);
        q_->memcpy(p, reinterpret_cast<const float*>(ti->data) + off, n * 4).wait();
        return {};
    };

    rt.kind = w.kind;
    if (e = f32vec(w.attn_norm, rt.attn_norm, cfg_.hidden);           !e.empty()) return e;
    if (e = dense(w.attn_q_a, rt.q_a);                                !e.empty()) return e;
    if (e = f32vec(w.attn_q_a_norm, rt.q_a_norm, cfg_.q_lora_rank);   !e.empty()) return e;
    // ROW-SPLIT: card c's query heads.  The input (`ws_qres_`, post-q_a_norm) is
    // mirrored and whole, so this is a slice of the output and nothing is summed
    // across cards.
    if (e = upload_dense(w.attn_q_b, rt.q_b, R);                      !e.empty()) return e;
    // MIRRORED, and it is the tensor with no splittable axis at all:
    // `head_count_kv == 1`, so there is a single shared K/V head that every
    // query head on every card attends to in full.  Splitting its 512-wide
    // output would split the q·k contraction itself, which would need a
    // [T, n_kv] partial reduced per layer — unbounded in context.
    if (e = dense(w.attn_kv, rt.kv);                                  !e.empty()) return e;
    if (e = f32vec(w.attn_kv_a_norm, rt.kv_a_norm, cfg_.head_dim);    !e.empty()) return e;
    if (e = f32slice(w.attn_sinks, rt.sinks, aslice_.nh0, aslice_.nhc); !e.empty()) return e;
    // ROW-SPLIT by OUTPUT GROUP.  Group g of the grouped projection reads
    // attention heads [g*NH/G, (g+1)*NH/G); load_prepare has already checked
    // that this card's groups read exactly this card's heads, so `ws_attn_` is
    // the whole input this weight needs.
    if (e = upload_dense(w.attn_output_a, rt.o_a, R);                 !e.empty()) return e;
    // COLUMN-SPLIT — the one new partial.  `attn_output_b` contracts over all
    // G*o_lora_rank group outputs; this card holds only its own groups', so its
    // result is a partial sum and `layer_forward_pre` ends there.
    if (e = upload_dense(w.attn_output_b, rt.o_b, C);                 !e.empty()) return e;

    if (w.kind.has_compressor) {
        const uint32_t CD = w.kind.compressor_dim;
        if (e = dense(w.compressor_kv, rt.comp_kv);                                       !e.empty()) return e;
        if (e = dense(w.compressor_gate, rt.comp_gate);                                   !e.empty()) return e;
        if (e = f32vec(w.compressor_ape, rt.comp_ape, uint64_t(CD) * w.kind.compress_ratio); !e.empty()) return e;
        if (e = f32vec(w.compressor_norm, rt.comp_norm, cfg_.head_dim);                   !e.empty()) return e;
    }
    if (w.kind.has_indexer) {
        const uint32_t IHD = cfg_.indexer_head_dim;
        if (e = dense(w.indexer_q_b, rt.idx_q_b);                                          !e.empty()) return e;
        // `indexer.proj` IS NOT REQUANTISED, for two independent reasons and it
        // only takes one.
        //
        // (1) NO WIN.  It is [hidden, indexer_n_heads] = 4096x64, 0.5 MB — the
        //     smallest dense weight in the model and the only shape where the
        //     Q8 GEMV measured SLOWER than the fp16 one (2.72 us vs 2.48 us,
        //     cache-cold, B70).  At that size the call is launch-bound, not
        //     bandwidth-bound, so the extra scale load is pure cost.  Across the
        //     21 indexer layers the whole tensor is 11 MB of 7,453 MB: 0.15% of
        //     the byte budget, for a measured regression.
        // (2) IT WEIGHTS A DISCRETE SELECTION.  Its output is the indexer's
        //     per-head weight in `ds4_indexer_score`, which `ds4_indexer_topk`
        //     then thresholds to pick the top-512 compressed entries.  Like the
        //     router — which is why `ffn_gate_inp` is dequantised to fp32 below
        //     — a perturbed score can change WHICH entries are attended, not
        //     just by how much, and that is a class of error no per-element
        //     bound describes.  The other three indexer tensors feed the same
        //     score, so this is a mitigation and not a proof; it is the one
        //     place where the mitigation is free.
        //     (2026-09-12: `kWideF16`, not `kKeepWide`, and NOT for the reason
        //     above.  `kKeepWide` resolves to the file's dtype, so on the ggml-org
        //     Q8_0-dense GGUF these three landed on the per-element `dense_packed`
        //     kernel: 935.7 ms of a 4360.7 ms two-card prefill GPU budget over 126
        //     calls, 7.43 ms/call against 0.24 for `dense_f16`.  Moving them is a
        //     PERFORMANCE change, +16.6% prefill, and on a Q8_0 source it costs a
        //     little precision rather than adding any — `dense_packed` decoded
        //     d*qs exactly in fp32, the fp16 image rounds to 11 bits.  The
        //     mitigation argument above still holds for an F16/F32-source file,
        //     which is what the abliterated GGUF is; it does NOT justify the
        //     change on a Q8_0 source.  See Ds4DenseQuant and docs/deepseek4/82.)
        if (e = upload_dense(w.indexer_proj, rt.idx_proj, Ds4DenseSplit::kMirror,
                             Ds4DenseQuant::kWideF16);                                     !e.empty()) return e;
        // ...and these two join it, for the SAME structural reason and NOT on the
        // strength of a measurement — read on before citing this as evidence.
        //
        // They are the only other requantised tensors feeding `ds4_indexer_score`
        // -> `ds4_indexer_topk`, and they are cheap to keep wide: 2*IHD = 256
        // columns, ~11 MB across the 21 indexer layers against a ~7,453 MB budget,
        // i.e. ~0.15% of the byte budget. Removing a whole class of
        // discrete-selection error for 0.15% is worth it on structure alone.
        //
        // WHAT WAS ACTUALLY MEASURED, including the part that refuted my reasoning.
        // Teacher-forced PPL over 511 tokens, Q8 dense vs fp16 dense: 5.4576 ->
        // 5.5061 (+0.889%), with ONE token moving 1.373 nats. The run-to-run
        // control is 0.0000% with 511/511 tokens bit-identical, so those are real.
        // I hypothesised the 1.373-nat token was this indexer selection flipping
        // and excluded these two tensors to fix it. **The result was byte-identical:
        // +0.8894%, same 1.373443 max, same single token.** So it is NOT the
        // indexer, and the exclusion bought nothing measurable at that length.
        //
        // The reason is that the test could not have caught it: a ratio-4 CSA layer
        // emits ctx/4 compressed entries, so at 512 tokens there are ~128 entries
        // against a top_k of 512 and `ds4_indexer_topk` admits everything — it never
        // SELECTS. The discrete risk only exists past ~2048 tokens of context and
        // remains UNTESTED. Do not read the identical result as evidence that
        // requantising these is safe; read it as evidence the experiment was blind.
        // The 4096-token pair is the one that would settle it.
        if (e = upload_dense(w.indexer_compressor_kv, rt.idx_kv, Ds4DenseSplit::kMirror,
                             Ds4DenseQuant::kWideF16);                                     !e.empty()) return e;
        if (e = upload_dense(w.indexer_compressor_gate, rt.idx_gate, Ds4DenseSplit::kMirror,
                             Ds4DenseQuant::kWideF16);                                     !e.empty()) return e;
        if (e = f32vec(w.indexer_compressor_ape, rt.idx_ape,
                       uint64_t(2 * IHD) * w.kind.compress_ratio);                         !e.empty()) return e;
        if (e = f32vec(w.indexer_compressor_norm, rt.idx_norm, IHD);                       !e.empty()) return e;
    }

    if (e = f32vec(w.ffn_norm, rt.ffn_norm, cfg_.hidden); !e.empty()) return e;
    {   // router weight: BF16 [hidden, n_experts] -> fp32 [n_experts, hidden]
        const uint64_t n = uint64_t(cfg_.n_experts) * cfg_.hidden;
        rt.router_w = sycl::malloc_device<float>(n, *q_);
        if (!rt.router_w) return "ffn_gate_inp: malloc_device failed";
        resident_bytes_ += n * 4;
        std::vector<float> h;
        ds4_stream_advise(w.ffn_gate_inp->data, w.ffn_gate_inp->nbytes);
        if (e = dequant_rows(*w.ffn_gate_inp, cfg_.hidden, 0, cfg_.n_experts, h); !e.empty()) return e;
        q_->memcpy(rt.router_w, h.data(), n * 4).wait();
    }
    if (w.kind.hash_router) {
        const uint64_t n = uint64_t(cfg_.vocab) * cfg_.n_experts_used;
        rt.tid2eid = sycl::malloc_device<int32_t>(n, *q_);
        if (!rt.tid2eid) return "ffn_gate_tid2eid: malloc_device failed";
        resident_bytes_ += n * 4;
        ds4_stream_advise(w.ffn_gate_tid2eid->data, n * 4);
        q_->memcpy(rt.tid2eid, w.ffn_gate_tid2eid->data, n * 4).wait();
    } else {
        if (e = f32vec(w.exp_probs_b, rt.router_bias, cfg_.n_experts); !e.empty()) return e;
    }
    if (w.exp_probs_b_vl) {
        if (e = f32vec(w.exp_probs_b_vl, rt.router_bias_vl, cfg_.n_experts); !e.empty()) return e;
    }
    // The shared expert, split on the SAME intermediate axis as the routed ones
    // and for the same reason: gate/up are a slice of the output (exact), down
    // is a partial.  Its partial is added into `ws_moe_` BEFORE the routed-expert
    // reduction, so it rides a reduction that already existed and adds none.
    if (e = upload_dense(w.ffn_gate_shexp, rt.sh_gate, R); !e.empty()) return e;
    if (e = upload_dense(w.ffn_up_shexp,   rt.sh_up,   R); !e.empty()) return e;
    if (e = upload_dense(w.ffn_down_shexp, rt.sh_down, C); !e.empty()) return e;

    const uint32_t MIX = (2u + cfg_.hc_count) * cfg_.hc_count;
    const uint64_t HCH = uint64_t(cfg_.hc_count) * cfg_.hidden;
    if (e = f32vec(w.hc_attn_fn,    rt.hc_attn_fn,    HCH * MIX); !e.empty()) return e;
    if (e = f32vec(w.hc_attn_base,  rt.hc_attn_base,  MIX);       !e.empty()) return e;
    if (e = f32vec(w.hc_attn_scale, rt.hc_attn_scale, 3);         !e.empty()) return e;
    if (e = f32vec(w.hc_ffn_fn,     rt.hc_ffn_fn,     HCH * MIX); !e.empty()) return e;
    if (e = f32vec(w.hc_ffn_base,   rt.hc_ffn_base,   MIX);       !e.empty()) return e;
    if (e = f32vec(w.hc_ffn_scale,  rt.hc_ffn_scale,  3);         !e.empty()) return e;
    return {};
}

// The kv-axis capacity every kv-shaped workspace is cut to, in ONE place.
//
// Attention sees `sliding_window-1 + max_seq` ring rows plus every compressed
// entry emitted so far.  The slowest compressor emits one entry per
// `compress_ratio` tokens, so max_context/2 bounds the compressed list.  Sized
// generously; layer_forward_pre refuses by name rather than overrunning if the
// bound is ever reached.
//
// IT IS A FUNCTION BECAUSE IT HAD BEEN WRITTEN TWICE — once to size the buffers
// and once to check them — and two copies of a sizing formula is exactly the
// drift that turns a refusal into an out-of-bounds write the day one copy is
// edited.  Every caller now asks the same question of the same code.
uint32_t DeepSeek4Runtime::ws_kv_capacity() const noexcept {
    return ws_sliding_capacity() + ws_compressed_capacity();
}

// n_sl is the sliding ring's retained rows plus this forward's T new ones:
// at most sliding_window - 1 + max_seq (Ds4LayerCache::update_sliding).
uint32_t DeepSeek4Runtime::ws_sliding_capacity() const noexcept {
    return cfg_.sliding_window + opt_.max_seq;
}

// n_c (and the indexer's n_ik, which must equal it) is cumulative_length /
// compress_ratio, and the smallest ratio in the schedule is the one that emits
// the most entries: max_context / min_ratio + 1 is what Ds4LayerCache::reserve
// allocates for it.  The 64 extra rows are the XMX kernel's key-block pad, so a
// block that straddles the last live entry never reads past a workspace.
// Before docs/deepseek4/73 this was max(max_context / 2, max_seq) — twice the
// entries any layer can hold — and at 1M context that factor was 8 GB of
// workspace per card by itself.
uint32_t DeepSeek4Runtime::ws_compressed_capacity() const noexcept {
    uint32_t min_ratio = 0;
    for (int32_t r : cfg_.compress_ratios)
        if (r > 0 && (min_ratio == 0 || uint32_t(r) < min_ratio)) min_ratio = uint32_t(r);
    if (min_ratio == 0) return 64u;                     // no compressor anywhere
    return opt_.max_context / min_ratio + 1u + 64u;
}

uint32_t DeepSeek4Runtime::ws_strip_rows() const noexcept { return ws_strip_; }

std::string DeepSeek4Runtime::alloc_workspace() {
    const uint32_t T   = opt_.max_seq;
    const uint32_t H   = cfg_.hidden;
    const uint32_t hc  = cfg_.hc_count;
    const uint32_t HD  = cfg_.head_dim;
    const uint32_t NH  = cfg_.n_q_heads;
    const uint32_t QR  = cfg_.q_lora_rank;
    const uint32_t IHD = cfg_.indexer_head_dim;
    const uint32_t IH  = cfg_.indexer_n_heads;
    // THIS CARD's extents, not the model's: under non-expert TP the query/
    // attention/group workspaces hold only the heads and groups this card owns,
    // and the shared-expert scratch only its slice of the intermediate.  All
    // three equal the model's extents when the set is mirrored.
    const uint32_t NHc = aslice_.nhc ? aslice_.nhc : NH;
    const uint32_t EF  = aslice_.efc ? aslice_.efc : cfg_.expert_ffn;
    const uint32_t OR  = cfg_.o_lora_rank * (aslice_.gc ? aslice_.gc : cfg_.o_groups);
    const uint32_t KVMAX = ws_kv_capacity();
    const uint32_t KVC   = ws_compressed_capacity();
    const uint32_t NSL   = ws_sliding_capacity();

    // QUERY-ROW STRIPS (docs/deepseek4/73 Phase 1).  The three workspaces that
    // are [rows, kv] — the indexer scores, the block bias and the concatenated
    // mask — used to be [max_seq, KVMAX] fp32 each, 4.2 GB per card at
    // --ctx 250000 with the 2048-token chunk and 16 GB at 1M, and they were not
    // in the arena derivation.  layer_forward_pre now walks the T query rows of
    // a chunk in strips of `ws_strip_` rows (every kernel involved does one
    // query row per work-group, so a strip boundary changes nothing), and the
    // three are [ws_strip_, kv].  The strip is the largest multiple of 32 whose
    // three rows fit $IE_DS4_STRIP_MB (default 512 MB), balanced so the strips
    // of a full chunk are equal-sized, and it is the whole chunk whenever that
    // fits — at --ctx 65536 and below the chunk is one strip, i.e. exactly the
    // pre-Phase-1 launch sequence.
    {
        uint64_t budget = 512ull << 20;
        if (const char* e = std::getenv("IE_DS4_STRIP_MB"); e && *e)
            budget = uint64_t(std::max(1.0, std::atof(e)) * 1048576.0);
        const uint64_t row_bytes = (uint64_t(KVC) + KVMAX + KVC) * sizeof(float);
        uint32_t rows = uint32_t(std::min<uint64_t>(T, budget / row_bytes));
        if (rows < T) {
            rows = std::max(rows, 32u) / 32u * 32u;
            const uint32_t n_strips = (T + rows - 1u) / rows;
            rows = ((T + n_strips - 1u) / n_strips + 31u) / 32u * 32u;
        }
        ws_strip_ = std::max(1u, std::min(rows, T));
    }

    ws_bytes_ = 0;
    auto A = [&](float*& p, uint64_t n) {
        p = sycl::malloc_device<float>(n, *q_);
        if (p) ws_bytes_ += n * sizeof(float);
        return p != nullptr;
    };
    auto I = [&](int32_t*& p, uint64_t n) {
        p = sycl::malloc_device<int32_t>(n, *q_);
        if (p) ws_bytes_ += n * sizeof(int32_t);
        return p != nullptr;
    };
    const uint32_t TS = ws_strip_;
    const bool ok =
        A(ws_streams_, uint64_t(T) * hc * H) && A(ws_streams2_, uint64_t(T) * hc * H) &&
        A(ws_post_, uint64_t(T) * hc) && A(ws_comb_, uint64_t(T) * hc * hc) &&
        A(ws_coll_, uint64_t(T) * H) && A(ws_norm_, uint64_t(T) * H) &&
        A(ws_qres_, uint64_t(T) * QR) && A(ws_q_, uint64_t(T) * NHc * HD) &&
        // `ws_kv_` is the kv_new staging (T rows) and nothing else now: the
        // attention reads the sliding buffer and the compressed cache in place
        // (Ds4KvSegs), so the [KVMAX, HD] concatenation it used to hold — and
        // the per-forward copy of the whole compressed cache into it — is gone.
        A(ws_kv_, uint64_t(T) * HD) && A(ws_attn_, uint64_t(T) * NHc * HD) &&
        A(ws_grp_, uint64_t(T) * OR) && A(ws_sub_, uint64_t(T) * H) &&
        // The sliding causal mask is [T, n_sl] and n_sl <= NSL: it never spanned
        // the compressed entries, it was merely allocated as if it did.
        A(ws_mask_, uint64_t(T) * NSL) &&
        A(ws_bias_, uint64_t(TS) * KVC) &&
        A(ws_maskc_, uint64_t(TS) * KVMAX) &&
        I(ws_imgmask_, T) && I(ws_vleft_, T) && I(ws_vright_, T) &&
        A(ws_cos_, uint64_t(T) * cfg_.rope_dim) && A(ws_sin_, uint64_t(T) * cfg_.rope_dim) &&
        A(ws_ccos_, uint64_t(T) * cfg_.rope_dim) && A(ws_csin_, uint64_t(T) * cfg_.rope_dim) &&
        // The compressed-entry RoPE tables are filled for THIS forward's n_win
        // new windows only (n_win <= T / ratio), never for the whole kv axis.
        A(ws_kcos_, uint64_t(T) * cfg_.rope_dim) && A(ws_ksin_, uint64_t(T) * cfg_.rope_dim) &&
        A(ws_cmp_, uint64_t(T) * 2 * std::max(HD, IHD) * 2) &&
        A(ws_pool_, uint64_t(T) * std::max(HD, IHD)) &&
        A(ws_iq_, uint64_t(T) * IH * IHD) && A(ws_iw_, uint64_t(T) * IH) &&
        A(ws_iscore_, uint64_t(TS) * KVC) &&
        A(ws_rlogit_, uint64_t(T) * cfg_.n_experts) &&
        A(ws_rw_, uint64_t(T) * cfg_.n_experts_used) &&
        A(ws_moe_, uint64_t(T) * H * 2) &&
        A(ws_shg_, uint64_t(T) * EF) && A(ws_shu_, uint64_t(T) * EF) &&
        // ONE ROW, not T.  `forward_epilogue` writes [TL, vocab] where
        // TL = last_only ? 1 : T, and `last_only` defaults to true and is passed
        // true at the ONLY call site in the engine (src/engine/engine.cpp:736).
        // Sizing this by `max_seq` therefore reserved 129 280 * 4 B per token of
        // prefill chunk to write one row into: 132 MB/card at --max-seq 256 and
        // 529 MB/card at 1024, which is VRAM the expert cache could have had.
        // A caller that does pass last_only=false grows it below rather than
        // overrunning it — the buffer is sized by what is asked for, never by
        // what might be.
        A(ws_logits_, cfg_.vocab) &&
        I(ws_pos_, T) && I(ws_spos_, T) && I(ws_cpos_, T) &&
        I(ws_ridx_, uint64_t(T) * cfg_.n_experts_used) &&
        I(ws_topk_, uint64_t(T) * std::max(cfg_.indexer_top_k, 1u)) && I(ws_ids_, T);
    if (!ok) return "DeepSeek4Runtime: workspace malloc_device failed";
    ws_T_ = T;
    ws_logits_T_ = 1;

    if (std::string e = ds4_expert_ws_alloc(*q_, H, EF, xws_); !e.empty()) return e;
    // Expert-major batch workspace.  Sized for at most kDs4BatchTokenCap tokens
    // (the chunk loop in layer_forward_pre bounds every chunk by the same cap),
    // and for the FULL EF — under expert-TP the per-call width is this card's
    // smaller slice, which simply uses the buffers strided by that width.
    if (std::string e = ds4_expert_batch_ws_alloc(*q_, std::min(T, kDs4BatchTokenCap),
                                                  cfg_.n_experts_used, H, EF, bws_);
        !e.empty()) return e;
    ws_bytes_ += ds4_expert_batch_ws_bytes(std::min(T, kDs4BatchTokenCap),
                                           cfg_.n_experts_used, H, cfg_.expert_ffn);
    // The XMX prefill attention stages Q and the SLIDING rows in fp16 on this
    // queue, lazily at its first call (fp16 compressed rows are read from the
    // cache in place; fp32 ones — IE_DS4_KV16=0 — are staged too, so then the
    // whole kv capacity is reserved).  Q is staged for the WHOLE chunk: the
    // sliding-only layers 0-1 are not stripped (gate finding 1).  Take it now,
    // so the arena derivation subtracts a real number.
    if (ds4_attention_xmx_eligible(q_->get_device(), T, NHc, HD)) {
        const uint32_t stage_max = kv16_values_ ? NSL : KVMAX;
        if (std::string e = ds4_attention_xmx_reserve(*q_, T, NHc, stage_max); !e.empty()) return e;
        ws_bytes_ += ds4_attention_xmx_ws_bytes(T, NHc, stage_max);
    }
    std::fprintf(stderr,
                 "[ds4] workspaces: %.3f GB (attention/indexer strips of %u of %u query rows;"
                 " kv capacity %u = %u sliding + %u compressed)\n",
                 double(ws_bytes_) / 1e9, TS, T, KVMAX, NSL, KVC);

    // Two RoPE frequency tables — ONE PER ROPE-TYPE LABEL, not per layer.
    Ds4RopeConfig main_cfg;
    main_cfg.theta = cfg_.rope_theta;
    inv_main_ = ds4_rope_inv_freq(main_cfg, cfg_.rope_dim);
    Ds4RopeConfig comp_cfg;
    comp_cfg.theta            = cfg_.compress_rope_theta;
    comp_cfg.yarn             = true;
    comp_cfg.factor           = cfg_.rope_freq_scale > 0.f ? 1.f / cfg_.rope_freq_scale : 16.f;
    comp_cfg.beta_fast        = cfg_.yarn_beta_fast;
    comp_cfg.beta_slow        = cfg_.yarn_beta_slow;
    comp_cfg.original_max_pos = cfg_.rope_orig_ctx;
    comp_cfg.attention_factor = 1.f;   // forced to 1.0 by the V4 config
    inv_comp_ = ds4_rope_inv_freq(comp_cfg, cfg_.rope_dim);
    inv_main_dev_ = sycl::malloc_device<float>(inv_main_.size(), *q_);
    inv_comp_dev_ = sycl::malloc_device<float>(inv_comp_.size(), *q_);
    if (!inv_main_dev_ || !inv_comp_dev_) return "DeepSeek4Runtime: rope table alloc failed";
    q_->memcpy(inv_main_dev_, inv_main_.data(), inv_main_.size() * 4);
    q_->memcpy(inv_comp_dev_, inv_comp_.data(), inv_comp_.size() * 4).wait();
    return {};
}

std::string DeepSeek4Runtime::load_prepare(const DeepSeek4Model& m, const Ds4Options& opt) {
    release();
    m_   = &m;
    cfg_ = m.config();
    opt_ = opt;
    // The env vars FORCE the token-chunked reference on; they never turn it off,
    // so a caller that asked for it in the options still gets it.
    if (std::getenv("IE_DS4_MOE_TOKEN_CHUNK") || std::getenv("IE_DS4_MOE_TOKEN_MAJOR"))
        opt_.moe_token_chunk = true;
    // Slot count: the API field wins, then the env var — same precedence as
    // every other residency knob.  A value of 0 from either means "the default".
    if (opt_.seq_slots == 0 || opt_.seq_slots == 1)
        if (const char* s = std::getenv("DS4_SEQ_SLOTS")) {
            const uint32_t n = uint32_t(std::max(0, std::atoi(s)));
            if (n > 1) {
                opt_.seq_slots = n;
                std::fprintf(stderr, "[ds4] %u sequence slots from $DS4_SEQ_SLOTS\n", n);
            }
        }

    // Checked FIRST, not when the slice is derived: `ds4_expert_slice` catches
    // card >= n_cards too, but only after the whole always-resident set has been
    // uploaded — minutes of work thrown away to report a one-line option error.
    if (opt_.card >= (opt_.n_cards ? opt_.n_cards : 1u))
        return "DeepSeek4Runtime::load: card " + std::to_string(opt_.card) +
               " is outside [0," + std::to_string(opt_.n_cards ? opt_.n_cards : 1u) +
               ") — Ds4Options::card names which slice of every expert this runtime holds";

    // GPU selection: the caller names the ordinal; both cards are enumerated and
    // the choice is reported, so a run never silently lands on a busy card.
    std::vector<sycl::device> gpus;
    for (const auto& d : sycl::device::get_devices())
        if (d.is_gpu() && d.get_backend() == sycl::backend::ext_oneapi_level_zero) gpus.push_back(d);
    if (gpus.empty()) return "DeepSeek4Runtime: no Level Zero GPU";
    if (opt_.device_ordinal >= gpus.size())
        return "DeepSeek4Runtime: device ordinal " + std::to_string(opt_.device_ordinal) +
               " but only " + std::to_string(gpus.size()) + " GPUs";
    // enable_profiling is OPT-IN ($IE_QUEUE_PROFILING=1), matching every other
    // queue in the engine (src/core/allocator.cpp).  MEASURED on this box with a
    // 4000-kernel chain on an in-order Level-Zero queue: 1.43 us/submit without
    // it, 1.85 us/submit with — +0.42 us on EVERY launch, and +0.65 us on every
    // submit-then-wait round trip (5.11 -> 5.85 us).  Decode submits thousands
    // of kernels per token per card and the chain's wall equals its submission
    // time, so that surcharge is charged straight to the token.
    //
    // IT IS NOT FREE TO TURN OFF: `--kprofile` reads command_start/command_end
    // off these events, which a non-profiled queue does not have.  A profiling
    // run must therefore set $IE_QUEUE_PROFILING=1, and the banner below says
    // which mode this process is in so a profile can never be silently taken
    // against a queue that cannot produce one.
    const bool qprof = std::getenv("IE_QUEUE_PROFILING") != nullptr;
    if (qprof)
        // Explicit single-device context: sycl::queue(device) would bind to the platform
    // default context (all visible devices) and every VRAM allocation would then be
    // mirrored resident in host RAM on a two-card box (src/core/allocator.cpp init()).
    q_store_.emplace_back(sycl::context(gpus[opt_.device_ordinal]), gpus[opt_.device_ordinal],
                              sycl::property_list{sycl::property::queue::in_order(),
                                                  sycl::property::queue::enable_profiling()});
    else
        q_store_.emplace_back(sycl::context(gpus[opt_.device_ordinal]), gpus[opt_.device_ordinal],
                              sycl::property_list{sycl::property::queue::in_order()});
    q_ = &q_store_.back();
    q_profiling_ = qprof;
    std::fprintf(stderr, "[ds4] device %u: %s (queue profiling %s%s)\n", opt_.device_ordinal,
                 q_->get_device().get_info<sycl::info::device::name>().c_str(),
                 qprof ? "ON" : "OFF",
                 qprof ? "" : " — set IE_QUEUE_PROFILING=1 for --kprofile");

    const uint32_t L = cfg_.n_layers;
    rt_.assign(L, Ds4LayerRT{});
    // One SET of per-layer caches per sequence slot, over ONE set of per-layer
    // transient scratch.  At the default seq_slots == 1 this is one set of
    // caches sharing one scratch, which allocates and copies exactly what the
    // pre-slot code did.
    seq_slots_ = std::max(1u, opt_.seq_slots);
    seq_       = 0;
    scratch_.clear();
    scratch_.resize(L);
    for (uint32_t l = 0; l < L; ++l) scratch_[l].bind(*q_);
    caches_.clear();
    caches_.resize(size_t(seq_slots_) * L);
    // Pre-size every cache to max_context so the forward path never grows a
    // buffer: the geometric growth is malloc + blocking copy + free per entry
    // per layer, and the accumulated churn (not any kernel — GPU time was
    // 46 ms in the same step) collapsed decode to a uniform 5.06 s/token at
    // depth 12288 and fed the xe watchdog DEVICE_LOST at 16-18 K (2026-08-08).
    // The bytes land BEFORE the expert-arena budget below, which now accounts
    // them.  IE_DS4_PRESIZE=0 restores lazy growth (A/B and escape hatch).
    // $IE_DS4_KV16 — the compressed-entry storage width (docs/deepseek4/73 Phase 2 gate):
    //   values (default)  fp16 K=V rows, fp32 indexer keys: prefill bit-identical to the fp32 cache
    //   1                 both fp16: another ~1.3 GB/card at 1M; near-tie top-k picks flip
    //   0                 all fp32: the pre-Phase-2 cache (13.76 kB/token/card)
    kv16_values_ = true; kv16_keys_ = false;
    if (const char* e = std::getenv("IE_DS4_KV16"); e && *e) {
        const std::string v(e);
        if      (v == "0")      { kv16_values_ = false; kv16_keys_ = false; }
        else if (v == "1")      { kv16_values_ = true;  kv16_keys_ = true;  }
        else if (v != "values")
            std::fprintf(stderr, "[ds4] IE_DS4_KV16=%s not understood (values|1|0); using values\n", e);
    }
    std::fprintf(stderr, "[ds4] compressed-entry cache width: values %s, indexer keys %s (IE_DS4_KV16)\n",
                 kv16_values_ ? "fp16" : "fp32", kv16_keys_ ? "fp16" : "fp32");
    uint64_t kv_presize_ctx = opt_.max_context;
    if (const char* e = std::getenv("IE_DS4_PRESIZE"); e && *e && std::string(e) == "0")
        kv_presize_ctx = 0;
    uint64_t kv_reserved = 0;
    for (uint32_t s = 0; s < seq_slots_; ++s)
        for (uint32_t l = 0; l < L; ++l) {
            Ds4CacheConfig cc;
            cc.compress_ratio = m.layers()[l].kind.compress_ratio;
            cc.head_dim       = cfg_.head_dim;
            cc.index_head_dim = cfg_.indexer_head_dim;
            cc.sliding_window = cfg_.sliding_window;
            cc.f16_values     = kv16_values_;
            cc.f16_keys       = kv16_keys_;
            Ds4LayerCache& c = caches_[size_t(s) * L + l];
            if (std::string e = c.init(*q_, cc, &scratch_[l]); !e.empty())
                return "slot " + std::to_string(s) + " layer " + std::to_string(l) +
                       " cache: " + e;
            if (kv_presize_ctx)
                if (std::string e = c.reserve(kv_presize_ctx); !e.empty())
                    return "slot " + std::to_string(s) + " layer " + std::to_string(l) +
                           " cache reserve at ctx " + std::to_string(kv_presize_ctx) +
                           ": " + e + " — lower --ctx, or IE_DS4_PRESIZE=0 to "
                           "accept lazy growth and its long-context stalls";
            kv_reserved += c.allocated_bytes();
        }
    if (kv_presize_ctx)
        std::fprintf(stderr,
                     "[ds4] layer caches pre-sized: %.3f GB at ctx %llu across %u slot(s) "
                     "(never grown mid-decode; IE_DS4_PRESIZE=0 for lazy growth)\n",
                     double(kv_reserved) / 1e9, (unsigned long long)kv_presize_ctx,
                     seq_slots_);
    if (seq_slots_ > 1)
        std::fprintf(stderr, "[ds4] %u sequence slots (interleaved; one forward still"
                             " advances one slot)\n", seq_slots_);

    // ---- this card's share of the ALWAYS-RESIDENT set ----
    //
    // Resolved BEFORE a byte is uploaded, because every refusal below is knowable
    // from the config alone and the alternative is failing partway through a
    // multi-GB upload.  All three extents come from ONE split of the head axis;
    // the identity that ties them together is checked, not assumed.
    {
        const uint32_t nc = opt_.n_cards ? opt_.n_cards : 1u;
        aslice_ = Ds4AttnSlice{};
        aslice_.n_cards = nc;
        aslice_.card    = opt_.card;
        aslice_.nh0 = 0; aslice_.nhc = cfg_.n_q_heads;
        aslice_.g0  = 0; aslice_.gc  = cfg_.o_groups;
        aslice_.ef0 = 0; aslice_.efc = cfg_.expert_ffn;
        aslice_.split = nc > 1 && opt_.split_non_expert;
        if (aslice_.split) {
            if (cfg_.o_groups == 0 || cfg_.n_q_heads % cfg_.o_groups)
                return "non-expert TP: " + std::to_string(cfg_.n_q_heads) +
                       " query heads do not divide into " + std::to_string(cfg_.o_groups) +
                       " output groups — attn_output_a's group axis cannot be aligned with the"
                       " head axis, so a head split would make a group read another card's"
                       " attention output";
            if (cfg_.n_q_heads % nc || cfg_.o_groups % nc)
                return "non-expert TP: " + std::to_string(cfg_.n_q_heads) + " query heads / " +
                       std::to_string(cfg_.o_groups) + " output groups do not both divide by " +
                       std::to_string(nc) +
                       " cards — refusing to split the attention unevenly (set"
                       " Ds4Options::split_non_expert = false to mirror it instead)";
            if (cfg_.expert_ffn % nc)
                return "non-expert TP: shared-expert intermediate " +
                       std::to_string(cfg_.expert_ffn) + " does not divide by " +
                       std::to_string(nc) + " cards";
            aslice_.nhc = cfg_.n_q_heads / nc;  aslice_.nh0 = opt_.card * aslice_.nhc;
            aslice_.gc  = cfg_.o_groups  / nc;  aslice_.g0  = opt_.card * aslice_.gc;
            aslice_.efc = cfg_.expert_ffn / nc; aslice_.ef0 = opt_.card * aslice_.efc;
            // THE IDENTITY THE WHOLE SPLIT RESTS ON.  attn_output_a's group g
            // consumes heads [g*HPG, (g+1)*HPG); this card owns groups
            // [g0, g0+gc), so the heads it must own are [g0*HPG, (g0+gc)*HPG).
            // If that is not exactly [nh0, nh0+nhc) the grouped GEMV would index
            // a head this card never computed.
            const uint32_t HPG = cfg_.n_q_heads / cfg_.o_groups;
            if (aslice_.g0 * HPG != aslice_.nh0 || aslice_.gc * HPG != aslice_.nhc)
                return "non-expert TP: card " + std::to_string(opt_.card) + " owns groups [" +
                       std::to_string(aslice_.g0) + "," + std::to_string(aslice_.g0 + aslice_.gc) +
                       ") which read heads [" + std::to_string(aslice_.g0 * HPG) + "," +
                       std::to_string((aslice_.g0 + aslice_.gc) * HPG) + ") but it owns heads [" +
                       std::to_string(aslice_.nh0) + "," +
                       std::to_string(aslice_.nh0 + aslice_.nhc) + ") — the two axes disagree";
            std::fprintf(stderr,
                         "[ds4] non-expert TP: card %u of %u holds query heads [%u,%u),"
                         " output groups [%u,%u), shared-expert columns [%u,%u)\n",
                         opt_.card, nc, aslice_.nh0, aslice_.nh0 + aslice_.nhc,
                         aslice_.g0, aslice_.g0 + aslice_.gc,
                         aslice_.ef0, aslice_.ef0 + aslice_.efc);
        }
    }

    // ---- always-resident weights ----
    const auto& g = m.globals();
    // `token_embd` IS NOT REQUANTISED.  It is not a GEMV: `gather_embd_w` reads
    // ONE row per token, 8 KB, so shrinking it moves no decode bandwidth at all
    // — the entire lever this policy exists for is absent here.  Converting it
    // would need a third kernel (a Q8_0-SoA gather) purely to change the
    // embeddings numerically for zero throughput.  Its 1.06 GB of fp16 IS a real
    // VRAM opportunity, and one this change deliberately does not claim.
    if (std::string e = upload_dense(g.token_embd, embd_, Ds4DenseSplit::kMirror,
                                     Ds4DenseQuant::kKeepWide); !e.empty()) return e;
    // The LM head IS requantised.  It is the single largest dense weight (529 MB
    // per card after the row split) and the biggest per-call win in the set
    // (889.55 -> 478.64 us, 1.86x).  Its output is logits, so it is the most
    // argmax-sensitive tensor in the model — but UD-Q3_K_XL ships this very
    // tensor as Q6_K, i.e. SIX bits, so Q8_0 here is strictly more precise than
    // a configuration that is already in production.  What it is not is
    // bit-identical to the fp16 route: like every other tensor converted here it
    // moves the logits, and a near-tie greedy pick can therefore flip.
    //
    // `stage_at_load = false`: this is the ONE requantised weight generation
    // never submits at T > 1 (`forward_epilogue` passes TL = 1 unless a caller
    // asked for every position's logits, which only teacher-forced scoring
    // does).  Reserving its fp16 staging buffer here would take 529 MB per card
    // out of the expert arena for a route generation does not run — and take it
    // WITHOUT charging resident_bytes_, so the arena derivation would not see
    // it.  ie-ds4-ppl grows the buffer on its first scoring epilogue instead.
    // The split argument below is the DEFAULT (kMirror) written out, because the
    // fourth argument has to be reached positionally.  It is not a change: the
    // LM head is mirrored, both cards hold all 4096 x 129280 of it, and the
    // staging buffer that reservation would have claimed is 529,530,880 halves
    // = 1.059 GB per card, against a per-layer high-water of 33.6 MB.
    if (std::string e = upload_dense(g.output, lm_, Ds4DenseSplit::kMirror,
                                     Ds4DenseQuant::kAuto, /*stage_at_load=*/false);
        !e.empty())
        return e;
    auto gvec = [&](const GgufTensorInfo* ti, float*& p, uint64_t n) -> std::string {
        p = sycl::malloc_device<float>(n, *q_);
        if (!p) return std::string(ti->name) + ": malloc_device failed";
        resident_bytes_ += n * 4;
        if (ti->dtype != DType::kF32) {   // output_hc_fn is Q8_0 in the ggml-org file; see f32vec
            std::vector<float> tmp;
            if (std::string e = dequant_rows(*ti, uint32_t(n), 0, 1, tmp); !e.empty()) return e;
            q_->memcpy(p, tmp.data(), n * 4).wait();
            return {};
        }
        ds4_stream_advise(ti->data, n * 4);
        q_->memcpy(p, ti->data, n * 4).wait();
        return {};
    };
    if (std::string e = gvec(g.output_norm, onorm_, cfg_.hidden); !e.empty()) return e;
    if (std::string e = gvec(g.output_hc_fn, hh_fn_,
                             uint64_t(cfg_.hc_count) * cfg_.hc_count * cfg_.hidden); !e.empty()) return e;
    if (std::string e = gvec(g.output_hc_base, hh_base_, cfg_.hc_count); !e.empty()) return e;
    if (std::string e = gvec(g.output_hc_scale, hh_scale_, 1); !e.empty()) return e;

    for (uint32_t l = 0; l < L; ++l) {
        if (std::string e = upload_layer(m.layers()[l], rt_[l]); !e.empty())
            return "layer " + std::to_string(l) + ": " + e;
        // With n_cards > 1 this runs on one thread PER CARD, so the `\r` form
        // would have the cards overwriting each other's line.  Label the card and
        // terminate the line instead; stdio locks per FILE, so each line is
        // whole.  Same information, one line per card per report.
        if ((l % 4) == 0 || l + 1 == L) {
            if ((opt_.n_cards ? opt_.n_cards : 1u) > 1)
                std::fprintf(stderr, "[ds4:c%u] resident weights: layer %u/%u  %.2f GB\n",
                             opt_.card, l + 1, L, double(resident_bytes_) / 1e9);
            else
                std::fprintf(stderr, "[ds4] resident weights: layer %u/%u  %.2f GB\r",
                             l + 1, L, double(resident_bytes_) / 1e9);
        }
    }
    std::fprintf(stderr,
                 "\n[ds4] always-resident set uploaded: %.3f GB"
                 " (packed residency saved %.3f GB vs dequantising it all to fp16;"
                 " that VRAM goes to the expert arena)\n",
                 double(resident_bytes_) / 1e9, double(resident_saved_) / 1e9);

    // Workspaces BEFORE the arena derivation, so `ws_bytes_` is what is really
    // on the card (docs/deepseek4/73 Phase 1) rather than a copy of the sizing
    // formulas — the ctx-scaled attention/indexer buffers were never in the
    // derivation at all.  Nothing here depends on the residency plan.
    if (std::string e = alloc_workspace(); !e.empty()) return e;

    // ---- routed experts: slot layout, pinned arena, device cache ----
    // Under hidden-dim expert-TP (docs/deepseek4/33 §2.1) this card binds only
    // its slice of every expert.  The slice is derived PER LAYER, from that
    // layer's own three tensors, because the dtypes vary across layers (blk.26's
    // gate/up are MXFP4 where its siblings are IQ3_XXS) and the block size a
    // slice has to land on comes from the dtype, not from the model.
    const uint32_t n_cards = opt_.n_cards ? opt_.n_cards : 1;
    std::vector<uint64_t> slot_bytes(L, 0), whole_bytes(L, 0);
    for (uint32_t l = 0; l < L; ++l) {
        const DeepSeek4Layer& w = m.layers()[l];
        Ds4ExpertSlice sl;
        if (std::string e = ds4_expert_slice(*w.ffn_gate_exps, *w.ffn_up_exps, *w.ffn_down_exps,
                                             cfg_.expert_ffn, n_cards, opt_.card, sl); !e.empty())
            return "layer " + std::to_string(l) + " expert slice: " + e;
        if (l == 0) slice_ = sl;
        else if (sl.ef0 != slice_.ef0 || sl.efc != slice_.efc)
            return "layer " + std::to_string(l) + ": expert slice disagrees with layer 0";
        if (std::string e = ds4_slot_layout_tp(*w.ffn_gate_exps, *w.ffn_up_exps, *w.ffn_down_exps,
                                               cfg_.hidden, cfg_.expert_ffn, sl, rt_[l].slot);
            !e.empty())
            return "layer " + std::to_string(l) + " slot layout: " + e;
        Ds4SlotLayout whole;
        if (std::string e = ds4_slot_layout(*w.ffn_gate_exps, *w.ffn_up_exps, *w.ffn_down_exps,
                                            cfg_.hidden, cfg_.expert_ffn, whole); !e.empty())
            return "layer " + std::to_string(l) + " whole-expert slot layout: " + e;
        slot_bytes[l]  = rt_[l].slot.bytes;
        whole_bytes[l] = whole.bytes;
    }
    pinned_layers_ = opt_.pin_layers ? std::min(opt_.pin_layers, L) : L;
    std::vector<uint64_t> pin_bytes(slot_bytes.begin(), slot_bytes.begin() + pinned_layers_);
    std::vector<uint64_t> pin_whole(whole_bytes.begin(), whole_bytes.begin() + pinned_layers_);

    // VRAM budget for the cache: what the device reports minus the resident set,
    // with a 2 GB margin for the workspace and driver overhead.
    // Precedence, most explicit first: the API field, then $DS4_CACHE_MB, then
    // the derivation.  The env fallback exists for the same reason
    // $DS4_PIN_CAP_GB and $DS4_STREAM_SLOTS do: `EngineOptions` carries no
    // equivalent, so without it a served engine cannot reach a configuration the
    // bench can (src/engine/engine.cpp, "Options with no EngineOptions
    // equivalent … resolve through the DS4_* env vars the runtime already
    // reads").
    uint64_t budget = opt_.expert_cache_bytes;
    if (!budget)
        if (const char* s = std::getenv("DS4_CACHE_MB")) {
            budget = uint64_t(std::max(0.0, std::atof(s)) * 1e6);
            std::fprintf(stderr, "[ds4] expert cache budget %.3f GB from $DS4_CACHE_MB\n",
                         double(budget) / 1e9);
        }
    if (!budget) {
        const uint64_t total = q_->get_device().get_info<sycl::info::device::global_mem_size>();
        // `kv_reserved` is real VRAM the pre-sized layer caches above already
        // hold; before 2026-08-08 the caches grew INTO the arena's leftovers
        // mid-decode instead, which is the churn documented at the pre-size
        // site.  Counting it here is what makes the pre-size safe at any --ctx.
        // `ws_bytes_` is every byte alloc_workspace() put on the card just
        // above — the attention/indexer workspaces, the expert-major batch
        // workspace and the XMX fp16 staging.  Until docs/deepseek4/73 only the
        // batch workspace was counted (as an estimate), and the [max_seq, kv]
        // attention buffers — 4.2 GB/card at --ctx 250000 with the 2048-token
        // chunk — came out of the 2 GiB margin: the first thing the xe driver
        // evicted was the expert arena, and decode fell from 25 to 0.8 tok/s.
        const uint64_t margin = 2ull << 30;
        const uint64_t used   = resident_bytes_ + kv_reserved + ws_bytes_ + margin +
                                opt_.vram_reserved_extra;
        budget = total > used ? total - used : 0;
        std::fprintf(stderr,
                     "[ds4] VRAM plan: %.3f GB total - %.3f resident - %.3f layer caches (ctx %u)"
                     " - %.3f workspaces - %.3f margin%s = %.3f GB expert arena%s\n",
                     double(total) / 1e9, double(resident_bytes_) / 1e9, double(kv_reserved) / 1e9,
                     unsigned(kv_presize_ctx), double(ws_bytes_) / 1e9, double(margin) / 1e9,
                     opt_.vram_reserved_extra ? " - vision" : "", double(budget) / 1e9,
                     total > used ? "" : " (OVERSUBSCRIBED: lower --ctx)");
    }
    // Same precedence for the slot count.  0 keeps the derivation from `budget`.
    uint32_t slots = opt_.slots_per_layer;
    if (!slots)
        if (const char* s = std::getenv("DS4_SLOTS")) {
            slots = uint32_t(std::max(0, std::atoi(s)));
            std::fprintf(stderr, "[ds4] %u slots/layer from $DS4_SLOTS\n", slots);
        }

    // ---- the tiered residency rule (docs/deepseek4/33 §2.2) ----
    // An expert that lives permanently in the VRAM arena gets NO pinned host
    // copy.  The plan is computed BEFORE either arena exists, from the real
    // per-layer slot sizes, so the pinned bytes are known — and capped — before
    // a single page is locked.
    // Cap precedence, most explicit first: the API field, then $DS4_PIN_CAP_GB
    // (the escape hatch), then a cap DERIVED FROM LIVE MEMORY, and only if
    // /proc/meminfo cannot be read does the old constant apply.  The derived cap
    // is the one that can see the page cache — the variable that made the first
    // full-pin run an oomd event (docs/deepseek4/40, "Standing hazard").
    uint64_t pin_cap = opt_.host_pin_cap_bytes;
    if (!pin_cap) {
        if (const char* s = std::getenv("DS4_PIN_CAP_GB")) {
            pin_cap = uint64_t(std::max(0.0, std::atof(s)) * 1e9);
            std::fprintf(stderr, "[ds4] pinned-host cap %.3f GB from $DS4_PIN_CAP_GB (override)\n",
                         double(pin_cap) / 1e9);
        } else {
            std::string why;
            bool        meminfo_ok = false;
            pin_cap = ds4_host_pin_cap_live(why, &meminfo_ok);
            if (meminfo_ok) {
                // Use the derived number EVEN WHEN IT IS ZERO.  Zero here does
                // not mean "unknown", it means "this box has less free than the
                // reserve"; substituting the 84 GB constant would raise the cap
                // on a machine that is already starved, which is precisely
                // backwards.  A zero cap makes the plan refuse, loudly.
                std::fprintf(stderr, "[ds4] pinned-host cap %.3f GB derived live: %s\n",
                             double(pin_cap) / 1e9, why.c_str());
            } else {
                pin_cap = kDs4HostPinCapDefault;
                std::fprintf(stderr,
                             "[ds4] pinned-host cap: %s; falling back to the %.3f GB constant\n",
                             why.c_str(), double(pin_cap) / 1e9);
            }
        }
    }
    uint32_t min_stream = opt_.min_stream_slots;
    if (!min_stream) {
        // Default = EVERY slot streaming (no static tier; every expert keeps a
        // pinned host copy).  Measured 2026-09-01 with LRU eviction, 2x B70,
        // ctx 4096, ie-ds4-bench: streaming 12 (the old kDs4MinStreamSlots) ->
        // 22.0 tok/s decode @ hit 0.73; 24 -> 23.8/25.8; 48 -> 25.6/26.6;
        // all 89 -> 28.2 @ hit 0.88 (+35%).  Prefill pays: pp512 222 -> 188
        // tok/s (every expert is now streamed at prefill; the follow-up is the
        // cross-layer double-buffered prefill, docs/deepseek4/71 lever 2).
        // Host RAM: 256 experts/layer pinned = 73.6 GB per card (fits the live
        // pin cap on the 256 GB box).  $DS4_STREAM_SLOTS=<n> restores a static
        // tier (n evictable slots, the rest static, no host copy).
        const char* s = std::getenv("DS4_STREAM_SLOTS");
        min_stream = s ? uint32_t(std::max(0, std::atoi(s))) : cfg_.n_experts;
    }
    // One token routes to n_experts_used experts; if fewer than that can stream,
    // a single token could not be served even in principle.
    min_stream = std::max(min_stream, cfg_.n_experts_used);
    // Prefill fetch banks (docs/deepseek4/72 Phase A) are paid for HERE, out of
    // the same VRAM budget and before the slot count is derived — never out of
    // the 2 GB margin.  They are dropped, and the plan redone on the full budget,
    // when the budget after them would keep fewer than 2 x bank_slots streaming:
    // a bank larger than the working set it protects is pointless, and one that
    // costs the working set is worse.  Without banks the grouped prefill path is
    // the old one.  $DS4_PREFILL_BANKS=<slots per bank> sizes them (0 = off).
    uint32_t bank_slots = kDs4PrefillBankSlots;
    if (const char* s = std::getenv("DS4_PREFILL_BANKS"))
        bank_slots = uint32_t(std::max(0, std::atoi(s)));
    const uint64_t bank_bytes =
        uint64_t(2) * bank_slots *
        (pin_bytes.empty() ? 0 : *std::max_element(pin_bytes.begin(), pin_bytes.end()));
    // `pin_cap` is the whole box's pinned-host budget; the TP planner divides it
    // by n_cards, because both cards' arenas come out of the same RAM.
    auto plan = [&](uint64_t b) {
        return ds4_plan_residency_tp(pin_bytes, pin_whole, cfg_.n_experts, n_cards,
                                     b, slots, pin_cap, min_stream, plan_);
    };
    bool planned = false;
    if (!budget) bank_slots = 0;   // nothing to pay for them with (explicit slots only)
    if (bank_slots) {
        const std::string e = plan(budget > bank_bytes ? budget - bank_bytes : 0);
        if (e.empty() && plan_.card.stream_slots >= 2u * bank_slots) {
            planned = true;
        } else {
            std::fprintf(stderr,
                         "[ds4] prefill fetch banks OFF: %.3f GB of banks would leave %u streaming"
                         " slots/layer%s%s — prefill runs the plain grouped path\n",
                         double(bank_bytes) / 1e9, plan_.card.stream_slots,
                         e.empty() ? "" : "; plan: ", e.c_str());
            bank_slots = 0;
        }
    }
    if (!planned)
        if (std::string e = plan(budget); !e.empty())
            return e;
    if (n_cards > 1)
        std::fprintf(stderr,
                     "[ds4] expert-TP: card %u of %u holds intermediate columns/rows [%u,%u) of"
                     " every expert — %.3f MB/slot vs %.3f MB whole\n",
                     opt_.card, n_cards, slice_.ef0, slice_.ef0 + slice_.efc,
                     double(plan_.card_slot_total) / double(pinned_layers_) / 1e6,
                     double(plan_.whole_slot_total) / double(pinned_layers_) / 1e6);
    std::fprintf(stderr,
                 "[ds4] residency: %u slots/layer = %u static (VRAM only, host copy never allocated)"
                 " + %u streaming; %.3f GB VRAM experts; %u of %u experts/layer pinned = %.3f GB host"
                 " (this card; %.3f GB over %u cards, cap %.3f GB)\n",
                 plan_.card.slots_per_layer, plan_.card.static_slots, plan_.card.stream_slots,
                 double(plan_.card.vram_bytes) / 1e9, plan_.card.pinned_experts, cfg_.n_experts,
                 double(plan_.card.host_bytes) / 1e9, double(plan_.host_bytes_total) / 1e9,
                 n_cards, double(pin_cap) / 1e9);

    // ---- residency priority: WHICH experts get the static VRAM slots ----
    // Empty = index order, which is what this used to be hard-wired to.  A
    // supplied order must be a full permutation: a partial one would leave some
    // expert with neither a VRAM home nor a host copy, and the only place that
    // shows up is a refusal mid-decode.  See Ds4Options::expert_priority for what
    // the trained balancing bias does and does not say about the default.
    // Precedence, most explicit first: the API field, then a measured ranking
    // file (the option, then $DS4_EXPERT_PRIORITY_FILE), then index order.  The
    // file is READ HERE and not merely accepted — a malformed or foreign-model
    // ranking is a hard, named error, never a silent fall-through to the
    // default, because "the file I supplied was quietly ignored" is exactly the
    // failure a measurement loop cannot detect from the outside.
    // Precedence gains one rung at the top: the PER-LAYER field, then the single
    // permutation, then the file (which itself accepts either form), then index
    // order.  Everything below works in per-layer terms and the single-permutation
    // paths simply replicate, so there is one residency code path and not two.
    prio_layer_.clear();
    prio_from_ = PriorityOrigin::kIndex;
    prio_src_.clear();
    if (!opt_.expert_priority_layers.empty()) {
        if (opt_.expert_priority_layers.size() != L)
            return "Ds4Options::expert_priority_layers has " +
                   std::to_string(opt_.expert_priority_layers.size()) + " entries but the model has " +
                   std::to_string(L) + " layers — it must name every layer, not a prefix";
        prio_layer_ = opt_.expert_priority_layers;
        prio_from_  = PriorityOrigin::kOptionField;
        prio_src_   = "Ds4Options::expert_priority_layers";
    } else if (!opt_.expert_priority.empty()) {
        prio_layer_.assign(L, opt_.expert_priority);
        prio_from_ = PriorityOrigin::kOptionField;
        prio_src_  = "Ds4Options::expert_priority";
    } else {
        std::string pf = opt_.expert_priority_file;
        const char* env = std::getenv("DS4_EXPERT_PRIORITY_FILE");
        if (pf.empty() && env && *env) pf = env;
        if (!pf.empty()) {
            if (std::string e = ds4_expert_priority_read_layers(pf, cfg_.n_experts, L, prio_layer_);
                !e.empty())
                return e;
            prio_from_ = PriorityOrigin::kFile;
            prio_src_  = pf;
        }
    }
    if (prio_layer_.empty()) {
        std::vector<uint32_t> ident(cfg_.n_experts);
        for (uint32_t i = 0; i < cfg_.n_experts; ++i) ident[i] = i;
        prio_layer_.assign(L, std::move(ident));
    } else {
        // Validated PER LAYER.  The file reader already checks what it read, but
        // the API fields have not been through it, and a partial or duplicated
        // order leaves some expert with neither a VRAM home nor a host copy —
        // which surfaces only as a refusal mid-decode.
        for (uint32_t l = 0; l < L; ++l) {
            const std::vector<uint32_t>& o = prio_layer_[l];
            if (o.size() != cfg_.n_experts)
                return "residency order for layer " + std::to_string(l) + " has " +
                       std::to_string(o.size()) + " entries but the model has " +
                       std::to_string(cfg_.n_experts) +
                       " experts — it must be a full permutation, not a prefix";
            std::vector<bool> seen(cfg_.n_experts, false);
            for (size_t i = 0; i < o.size(); ++i) {
                if (o[i] >= cfg_.n_experts)
                    return "residency order for layer " + std::to_string(l) + "[" +
                           std::to_string(i) + "] = " + std::to_string(o[i]) + " is outside [0," +
                           std::to_string(cfg_.n_experts) + ")";
                if (seen[o[i]])
                    return "residency order for layer " + std::to_string(l) + " lists expert " +
                           std::to_string(o[i]) + " twice (at position " + std::to_string(i) +
                           ") — it must be a permutation";
                seen[o[i]] = true;
            }
        }
    }
    prio_ = prio_layer_[0];
    // prio_rank_[L][e] = e's static VRAM slot IN LAYER L, or -1 if it streams.
    prio_rank_.assign(uint64_t(L) * cfg_.n_experts, -1);
    // The experts that did NOT win a static slot are exactly the ones needing a
    // pinned host copy, in priority order — and that is now a per-layer list.
    std::vector<std::vector<uint32_t>> pinned_ids(L);
    for (uint32_t l = 0; l < L; ++l) {
        const std::vector<uint32_t>& o = prio_layer_[l];
        for (uint32_t i = 0; i < plan_.card.static_slots && i < o.size(); ++i)
            prio_rank_[uint64_t(l) * cfg_.n_experts + o[i]] = int32_t(i);
        pinned_ids[l].assign(
            o.begin() + std::min<size_t>(plan_.card.static_slots, o.size()),
            o.begin() + std::min<size_t>(size_t(plan_.card.static_slots) + plan_.card.pinned_experts,
                                         o.size()));
    }
    // Do the layers actually disagree?  Worth stating, because "I supplied a
    // per-layer ranking" and "the load used one" are different claims and the
    // difference is invisible from the outside otherwise.
    bool per_layer_differs = false;
    for (uint32_t l = 1; l < L && !per_layer_differs; ++l)
        per_layer_differs = prio_layer_[l] != prio_layer_[0];
    if (prio_from_ == PriorityOrigin::kIndex)
        std::fprintf(stderr,
                     "[ds4] residency order: index order (default; no Ds4Options::expert_priority,"
                     " no $DS4_EXPERT_PRIORITY_FILE)\n");
    else
        std::fprintf(stderr,
                     "[ds4] residency order: %s (%s); layer 0's static slots go to experts"
                     " %u,%u,%u... (index order would be 0,1,2...)\n",
                     prio_src_.c_str(),
                     per_layer_differs ? "PER-LAYER, the layers differ" : "one order for all layers",
                     prio_[0], prio_.size() > 1 ? prio_[1] : 0, prio_.size() > 2 ? prio_[2] : 0);

    // ---- optional expert-selection profiling ----
    // The measurement side of the same loop: counting is off unless asked for,
    // and $DS4_EXPERT_PROFILE additionally names the file release() dumps to.
    profile_path_.clear();
    bool want_profile = opt_.profile_experts;
    if (const char* pp = std::getenv("DS4_EXPERT_PROFILE"); pp && *pp) {
        want_profile  = true;
        profile_path_ = pp;
    }
    if (want_profile) {
        if (std::string e = profile_.init(L, cfg_.n_experts); !e.empty()) return e;
        std::fprintf(stderr,
                     "[ds4] expert-selection profiling ON (%u layers x %u experts)%s%s\n",
                     L, cfg_.n_experts, profile_path_.empty() ? "" : "; will dump to ",
                     profile_path_.c_str());
    }

    // ---- optional TEMPORAL routing trace (see the header) ----
    // Card 0 only, and it does not imply the histogram: the two are independent
    // switches so a run can ask for either without paying for the other.
    trace_path_.clear();
    trace_buf_.clear();
    trace_seq_ = trace_dropped_ = 0;
    if (const char* ap = std::getenv("DS4_ACT_DUMP"); ap && *ap && opt_.card == 0 && !act_dump_) {
        act_dump_ = std::fopen(ap, "wb");
        if (act_dump_) {
            const uint32_t hdr[4] = {0x31544341u /*'ACT1'*/, cfg_.n_layers, cfg_.hidden, 2u};
            std::fwrite(hdr, 4, 4, act_dump_);
            h_act_.assign(cfg_.hidden, 0.f);
            std::fprintf(stderr, "[ds4] activation DUMP on (card 0, T==1): %s\n", ap);
        }
    }
    if (const char* tp = std::getenv("DS4_EXPERT_TRACE"); tp && *tp && opt_.card == 0) {
        trace_path_ = tp;
        trace_buf_.reserve(kDs4TraceMaxInts / 8);
        std::fprintf(stderr,
                     "[ds4] routing TRACE on (card 0); up to %llu records to %s\n",
                     (unsigned long long)(kDs4TraceMaxInts / (kDs4TraceFields + cfg_.n_experts_used)),
                     trace_path_.c_str());
    }

    // The arena covers only the PINNED layers ($DS4_PIN_LAYERS may cut it short),
    // so it gets that prefix of the per-layer sets — sliced here rather than at
    // the point they were built, because `prio_rank_` describes every layer.
    if (std::string e = arena_.init_set_per_layer(
            *q_, pin_bytes, cfg_.n_experts,
            std::vector<std::vector<uint32_t>>(pinned_ids.begin(),
                                               pinned_ids.begin() + pinned_layers_));
        !e.empty())
        return e;
    std::fprintf(stderr, "[ds4] pinned host arena: %.3f GB in %zu segments (largest %.3f GB, cap %.3f GB)\n",
                 double(arena_.total_bytes()) / 1e9, arena_.n_segments(),
                 double(arena_.max_segment_bytes()) / 1e9, double(kDs4MaxAllocBytes) / 1e9);

    if (std::string e = cache_.init(*q_, arena_, plan_.card.slots_per_layer, 0,
                                    plan_.card.static_slots, bank_slots);
        !e.empty())
        return e;
    std::fprintf(stderr, "[ds4] expert cache: %u slots/layer, %.3f GB VRAM (%.1f%% residency)\n",
                 cache_.slots_per_layer(), double(cache_.device_bytes()) / 1e9,
                 100.0 * cache_.slots_per_layer() / cfg_.n_experts);

    // The staging buffer the routed-expert walk packs statically-resident experts
    // through: one slot's worth, reused for every one of them, so their host
    // bytes are DROPPED the moment `install_static` returns.  It belongs to THIS
    // runtime and no other card ever writes through it — that is the whole
    // synchronisation story of the orchestrator's interleaved drive.
    uint64_t widest = 0;
    for (uint64_t b : pin_bytes) widest = std::max(widest, b);
    pack_stage_.assign(static_cast<size_t>(widest), 0);
    packed_bytes_ = 0;
    return {};
}

void DeepSeek4Runtime::load_advise_layer(uint32_t l) noexcept {
    // Undo the reader's MADV_RANDOM over this layer's expert tensors before they
    // are walked in expert order.  Readahead is a property of the mapping, not a
    // queued request, so ONE call per layer covers the whole pass — the kernel
    // then reads ahead of the pack cursor by itself.
    if (!m_ || l >= pinned_layers_) return;
    const DeepSeek4Layer& w = m_->layers()[l];
    ds4_stream_advise_experts(*w.ffn_gate_exps, *w.ffn_up_exps, *w.ffn_down_exps);
}

std::string DeepSeek4Runtime::load_pack_expert(uint32_t l, uint32_t e) {
    // These are internal-API preconditions, but the caller is the TP orchestrator
    // driving two cards through one loop, so getting them wrong would silently
    // pack one card's bytes with the other's geometry.  Checked, not assumed.
    if (!m_) return "DeepSeek4Runtime::load_pack_expert: no model bound (load_prepare not run)";
    if (l >= pinned_layers_)
        return "DeepSeek4Runtime::load_pack_expert: layer " + std::to_string(l) +
               " is outside the pinned range [0," + std::to_string(pinned_layers_) + ")";
    if (e >= cfg_.n_experts)
        return "DeepSeek4Runtime::load_pack_expert: expert " + std::to_string(e) +
               " is outside [0," + std::to_string(cfg_.n_experts) + ")";
    if (pack_stage_.size() < rt_[l].slot.bytes)
        return "DeepSeek4Runtime::load_pack_expert: staging buffer is " +
               std::to_string(pack_stage_.size()) + " B but layer " + std::to_string(l) +
               "'s slot needs " + std::to_string(rt_[l].slot.bytes) + " B";

    const DeepSeek4Layer& w = m_->layers()[l];
    // PER LAYER: the same expert id may be statically resident in one layer and
    // stream in the next, which is the whole point of a per-layer residency order.
    const int32_t vslot = prio_rank_[uint64_t(l) * cfg_.n_experts + e];   // >= 0 => static slot
    const bool    stat  = vslot >= 0;
    void* dst = stat ? static_cast<void*>(pack_stage_.data()) : arena_.slot(l, e);
    if (!dst)
        return "layer " + std::to_string(l) + " expert " + std::to_string(e) +
               ": no arena slot — the residency plan and the arena disagree";
    if (std::string s = ds4_slot_pack(rt_[l].slot, *w.ffn_gate_exps, *w.ffn_up_exps,
                                      *w.ffn_down_exps, e, dst); !s.empty())
        return s;
    if (stat)
        if (std::string s = cache_.install_static(l, e, uint32_t(vslot), pack_stage_.data());
            !s.empty()) return s;
    packed_bytes_ += rt_[l].slot.bytes;
    return {};
}

std::string DeepSeek4Runtime::load_finish() {
    // The staging buffer has done its job; holding a slot's worth of host RAM for
    // the life of the runtime would be exactly the double-counting the tiered
    // residency rule exists to prevent.
    pack_stage_.clear();
    pack_stage_.shrink_to_fit();
    // (alloc_workspace() ran in load_prepare, ahead of the arena derivation.)
    // PINNED, because the router readback in layer_forward_mid is a device write
    // into these and a pageable destination puts that on the driver's staged
    // path.  A failed pin is a hard error: there is no pageable form of these
    // buffers the readback's contract allows.
    const size_t rk = size_t(opt_.max_seq) * cfg_.n_experts_used;
    h_ridx_ = sycl::malloc_host<int32_t>(rk, *q_);
    h_rw_   = sycl::malloc_host<float>(rk, *q_);
    if (!h_ridx_ || !h_rw_)
        return "DeepSeek4Runtime: could not pin " + std::to_string(rk * 8) +
               " B for the per-layer router readback";
    std::memset(h_ridx_, 0, rk * sizeof(int32_t));
    std::memset(h_rw_, 0, rk * sizeof(float));
    h_slots_.assign(size_t(opt_.max_seq) * cfg_.n_experts_used, 0);
    h_spos_.assign(size_t(opt_.max_seq), 0);
    // Vision-Exp row splice source: one forward's worth of [max_seq, hidden] f32, pinned.
    h_vis_rows_ = sycl::malloc_host<float>(size_t(opt_.max_seq) * cfg_.hidden, *q_);
    if (!h_vis_rows_)
        return "DeepSeek4Runtime: could not pin " +
               std::to_string(size_t(opt_.max_seq) * cfg_.hidden * 4) + " B for the vision row splice";
    h_vis_.assign(size_t(3) * opt_.max_seq, 0);
    return {};
}

std::string DeepSeek4Runtime::load(const DeepSeek4Model& m, const Ds4Options& opt) {
    if (std::string e = load_prepare(m, opt); !e.empty()) return e;
    // One pass over each layer's expert tensors, in ASCENDING expert order, so
    // the mmap is read forwards exactly once.  Expert order is independent of
    // residency priority: priority decides WHERE an expert's bytes land (a static
    // VRAM slot vs the pinned host arena), never the order they are READ in.
    const auto t0 = std::chrono::steady_clock::now();
    for (uint32_t l = 0; l < pinned_layers_; ++l) {
        load_advise_layer(l);
        for (uint32_t e = 0; e < cfg_.n_experts; ++e)
            if (std::string s = load_pack_expert(l, e); !s.empty()) return s;
        const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const double done = double(l + 1) * rt_[l].slot.bytes * cfg_.n_experts;
        std::fprintf(stderr, "[ds4] expert pool: layer %u/%u  %.2f GB read  %.1f s  %.1f MB/s\r",
                     l + 1, pinned_layers_, done / 1e9, el, done / 1e6 / std::max(el, 1e-9));
    }
    std::fprintf(stderr, "\n");
    return load_finish();
}

void DeepSeek4Runtime::accumulate_cache(uint32_t T) noexcept {
    const Ds4ExpertCache::Stats& s = cache_.stats();
    // T == 1 IS the decode regime, and it is the only honest way to separate the
    // two: a chunk touches most of the pool once, a decode step touches six.
    CacheTotals& t = (T == 1) ? ctot_decode_ : ctot_prefill_;
    t.acquires    += s.hits + s.misses + s.unavailable;
    t.static_hits += s.static_hits;
    t.stream_hits += s.stream_hits;
    t.misses      += s.misses;
    t.unavailable += s.unavailable;
    t.bytes       += s.bytes_fetched;
    ++t.forwards;
}

void DeepSeek4Runtime::release() noexcept {
    // The lifetime cache breakdown, before anything is torn down.  OFF unless
    // asked for: it is a diagnostic, and an unconditional line per card per run
    // is noise in every run that is not investigating residency.
    if (std::getenv("DS4_CACHE_BREAKDOWN")) {
        auto line = [&](const char* what, const CacheTotals& t) {
            if (!t.forwards) return;
            const double n = double(t.acquires ? t.acquires : 1);
            std::fprintf(stderr,
                         "[ds4] cache %s (card %u): %llu forwards, %llu acquires -> "
                         "static %.4f  stream %.4f  miss %.4f  (hit %.4f), %llu unavailable, "
                         "%.1f MB/forward\n",
                         what, opt_.card, (unsigned long long)t.forwards,
                         (unsigned long long)t.acquires, double(t.static_hits) / n,
                         double(t.stream_hits) / n, double(t.misses) / n,
                         double(t.static_hits + t.stream_hits) / n,
                         (unsigned long long)t.unavailable,
                         double(t.bytes) / 1e6 / double(t.forwards));
        };
        line("DECODE (T==1)", ctot_decode_);
        line("PREFILL(T>1) ", ctot_prefill_);
    }
    // Dump the measured profile before anything is torn down.  This is what
    // makes $DS4_EXPERT_PROFILE a complete loop on its own: run once with it
    // set, then point $DS4_EXPERT_PRIORITY_FILE at the same file next time.
    // A write failure is reported and never silently swallowed, but it cannot
    // fail a release — the inference already happened.
    if (!profile_path_.empty() && profile_.active()) {
        if (const std::string e = profile_.write(profile_path_); !e.empty())
            std::fprintf(stderr, "[ds4] expert profile NOT written: %s\n", e.c_str());
        else
            std::fprintf(stderr, "[ds4] expert profile written to %s (%llu selections)\n",
                         profile_path_.c_str(), (unsigned long long)profile_.selections());
        profile_path_.clear();   // release() is idempotent; the dump must be too
    }
    // The temporal trace, same contract: written before teardown, a failure is
    // reported and never swallowed, and never fails a release.
    if (!trace_path_.empty()) {
        const size_t rec = kDs4TraceFields + cfg_.n_experts_used;
        std::FILE* f = std::fopen(trace_path_.c_str(), "w");
        if (!f) {
            std::fprintf(stderr, "[ds4] routing trace NOT written: cannot open %s\n",
                         trace_path_.c_str());
        } else {
            std::fprintf(f, "# ds4 routing trace: seq tok T layer e0..e%u\n",
                         cfg_.n_experts_used - 1);
            std::fprintf(f, "# n_layers=%u n_experts=%u n_experts_used=%u hash_layers=%u"
                           " records=%zu dropped=%llu\n",
                         uint32_t(rt_.size()), cfg_.n_experts, cfg_.n_experts_used,
                         cfg_.hash_layer_count, trace_buf_.size() / rec,
                         (unsigned long long)trace_dropped_);
            for (size_t i = 0; i + rec <= trace_buf_.size(); i += rec) {
                for (size_t j = 0; j < rec; ++j)
                    std::fprintf(f, j + 1 == rec ? "%d\n" : "%d ", trace_buf_[i + j]);
            }
            const bool ok = std::fclose(f) == 0;
            std::fprintf(stderr, "[ds4] routing trace %s to %s (%zu records, %llu dropped)\n",
                         ok ? "written" : "PARTIALLY written", trace_path_.c_str(),
                         trace_buf_.size() / rec, (unsigned long long)trace_dropped_);
        }
        trace_path_.clear();
        trace_buf_.clear();
        trace_buf_.shrink_to_fit();
    }
    if (q_) {
        auto F = [&](void* p) { if (p) sycl::free(p, *q_); };
        for (Ds4LayerRT& r : rt_) {
            for (Ds4Dense* d : {&r.q_a, &r.q_b, &r.kv, &r.o_a, &r.o_b, &r.comp_kv, &r.comp_gate,
                                &r.idx_q_b, &r.idx_proj, &r.idx_kv, &r.idx_gate,
                                &r.sh_gate, &r.sh_up, &r.sh_down})
                // `q8d` points INSIDE `q8`'s allocation and must not be freed.
                { F(d->w); F(d->p); F(d->q8); }
            for (float* p : {r.attn_norm, r.q_a_norm, r.kv_a_norm, r.sinks, r.comp_ape,
                             r.comp_norm, r.idx_ape, r.idx_norm, r.ffn_norm, r.router_w,
                             r.router_bias, r.router_bias_vl, r.hc_attn_fn, r.hc_attn_base, r.hc_attn_scale,
                             r.hc_ffn_fn, r.hc_ffn_base, r.hc_ffn_scale}) F(p);
            F(r.tid2eid);
        }
        for (Ds4Dense* d : {&embd_, &lm_}) { F(d->w); F(d->p); F(d->q8); }
        F(onorm_); F(hh_fn_); F(hh_base_); F(hh_scale_);
        for (float* p : {ws_streams_, ws_streams2_, ws_post_, ws_comb_, ws_coll_, ws_norm_,
                         ws_qres_, ws_q_, ws_kv_, ws_attn_, ws_grp_, ws_sub_, ws_mask_,
                         ws_bias_, ws_maskc_, ws_cos_, ws_sin_, ws_ccos_, ws_csin_,
                         ws_kcos_, ws_ksin_, ws_cmp_, ws_pool_, ws_iq_, ws_iw_, ws_iscore_,
                         ws_rlogit_, ws_rw_, ws_moe_, ws_shg_, ws_shu_, ws_logits_}) F(p);
        for (int32_t* p : {ws_pos_, ws_spos_, ws_cpos_, ws_ridx_, ws_topk_, ws_ids_}) F(p);
        // The router readback's pinned destinations.  The kernel that writes
        // them is always drained before this point (layer_forward_mid's own
        // `drain`), so freeing them here is ordered against nothing outstanding.
        F(h_ridx_); F(h_rw_); F(h_vis_rows_);
        if (act_dump_) { std::fclose(act_dump_); act_dump_ = nullptr; }
        for (int32_t* p : {ws_imgmask_, ws_vleft_, ws_vright_}) F(p);
        F(inv_main_dev_); F(inv_comp_dev_);
        ds4_expert_ws_free(*q_, xws_);
        ds4_expert_batch_ws_free(*q_, bws_);
    }
    cache_.free_storage();
    arena_.free_storage();
    // Caches first, scratch second: a cache's free_storage() deliberately does
    // NOT touch a shared scratch (other slots are still on it), so the scratch
    // has to be released here — and before `q_store_` goes, since both free
    // through that queue.
    caches_.clear();
    scratch_.clear();
    seq_slots_ = 1;
    seq_       = 0;
    rt_.clear();
    embd_ = Ds4Dense{}; lm_ = Ds4Dense{};
    onorm_ = hh_fn_ = hh_base_ = hh_scale_ = nullptr;
    ws_streams_ = ws_streams2_ = ws_post_ = ws_comb_ = ws_coll_ = ws_norm_ = nullptr;
    ws_qres_ = ws_q_ = ws_kv_ = ws_attn_ = ws_grp_ = ws_sub_ = nullptr;
    ws_mask_ = ws_bias_ = ws_maskc_ = nullptr;
    ws_cos_ = ws_sin_ = ws_ccos_ = ws_csin_ = ws_kcos_ = ws_ksin_ = nullptr;
    ws_cmp_ = ws_pool_ = ws_iq_ = ws_iw_ = ws_iscore_ = nullptr;
    ws_rlogit_ = ws_rw_ = ws_moe_ = ws_shg_ = ws_shu_ = ws_logits_ = nullptr;
    ws_pos_ = ws_spos_ = ws_cpos_ = ws_ridx_ = ws_topk_ = ws_ids_ = nullptr;
    h_ridx_ = nullptr; h_rw_ = nullptr; h_vis_rows_ = nullptr;
    ws_imgmask_ = ws_vleft_ = ws_vright_ = nullptr; vis_chunk_ = false;
    inv_main_dev_ = inv_comp_dev_ = nullptr;
    q_ = nullptr;
    q_store_.clear();
    plan_ = Ds4TpResidencyPlan{};
    slice_ = Ds4ExpertSlice{};
    resident_bytes_ = 0; resident_saved_ = 0; pinned_layers_ = 0; ws_T_ = 0;
    ws_strip_ = 0; ws_bytes_ = 0;
    ws_logits_T_ = 0;
    q_profiling_ = false;
    pack_stage_.clear();
    pack_stage_.shrink_to_fit();
    packed_bytes_ = 0;
}

void DeepSeek4Runtime::act_dump(uint32_t L, uint32_t site, const float* dev_row) {
    // Probe only: one synchronous D2H per (layer, site) per token.
    (void)L; (void)site;
    q_->memcpy(h_act_.data(), dev_row, h_act_.size() * sizeof(float)).wait();
    std::fwrite(h_act_.data(), sizeof(float), h_act_.size(), act_dump_);
}

void DeepSeek4Runtime::reset_context() noexcept {
    for (Ds4LayerCache& c : caches_) c.reset();
}

std::string DeepSeek4Runtime::select_seq(uint32_t slot) noexcept {
    if (slot >= seq_slots_)
        return "DeepSeek4Runtime::select_seq: slot " + std::to_string(slot) +
               " but the runtime was loaded with " + std::to_string(seq_slots_) +
               " (raise Ds4Options::seq_slots or $DS4_SEQ_SLOTS)";
    seq_ = slot;
    return {};
}

std::string DeepSeek4Runtime::reset_seq(uint32_t slot) noexcept {
    if (slot >= seq_slots_)
        return "DeepSeek4Runtime::reset_seq: slot " + std::to_string(slot) +
               " but the runtime was loaded with " + std::to_string(seq_slots_);
    const size_t L = rt_.size();
    for (size_t l = 0; l < L; ++l) caches_[size_t(slot) * L + l].reset();
    return {};
}

// ---------------------------------------------------------------------------
// Prompt cache — ONE snapshot of the ACTIVE slot's layer caches.
//
// `reset_context()` two dozen lines above is the whole argument for why this is
// sufficient: it clears the layer caches and touches nothing else, so the layer
// caches ARE this runtime's per-context state.
// ---------------------------------------------------------------------------
void DeepSeek4Runtime::free_snapshot() noexcept {
    snap_.clear();          // each Ds4LayerCache releases its own device storage
    snap_depth_ = 0;
}

std::string DeepSeek4Runtime::snapshot_context() {
    if (!q_ || caches_.empty()) return "DeepSeek4Runtime::snapshot_context: not loaded";
    const size_t L = rt_.size();
    // FREE FIRST, for two reasons: the old snapshot's VRAM becomes available to
    // the new one (fewer refusals at the margin), and a failure below leaves NO
    // snapshot rather than a STALE one — which the caller's longest-prefix match
    // would otherwise restore as if it were current.
    free_snapshot();
    snap_.resize(L);
    for (size_t l = 0; l < L; ++l) {
        const Ds4LayerCache& live = caches_[size_t(seq_) * L + l];
        // No scratch: a snapshot never runs a forward, so binding the shared
        // transient half would only give it a pointer it can never use.
        std::string e = snap_[l].init(*q_, live.config(), /*scratch=*/nullptr);
        if (e.empty()) e = live.snapshot_to(snap_[l]);
        if (!e.empty()) {
            free_snapshot();
            return "ds4 snapshot: layer " + std::to_string(l) + ": " + e;
        }
    }
    snap_depth_ = uint32_t(caches_[size_t(seq_) * L].cumulative_length());
    return {};
}

uint32_t DeepSeek4Runtime::restore_context() {
    const size_t L = rt_.size();
    if (snap_depth_ == 0 || snap_.size() != L) return 0;
    for (size_t l = 0; l < L; ++l) {
        Ds4LayerCache& live = caches_[size_t(seq_) * L + l];
        if (std::string e = live.restore_from(snap_[l]); !e.empty()) {
            // A PARTIAL restore is the failure that does not announce itself:
            // layers [0,l) hold the cached conversation and the rest hold
            // whatever was there before.  Clear the whole slot so the caller's
            // pos0 == 0 path re-prefills honestly.
            std::fprintf(stderr, "[ds4-cache] restore failed at layer %zu (%s); full prefill\n",
                         l, e.c_str());
            for (size_t k = 0; k < L; ++k) caches_[size_t(seq_) * L + k].reset();
            return 0;
        }
    }
    return snap_depth_;
}

uint64_t DeepSeek4Runtime::snapshot_bytes() const noexcept {
    uint64_t n = 0;
    for (const Ds4LayerCache& c : snap_) n += c.allocated_bytes();
    return n;
}

std::string DeepSeek4Runtime::snapshot_to_host(Ds4HostSnapshot& out) {
    if (!q_ || caches_.empty()) return "DeepSeek4Runtime::snapshot_to_host: not loaded";
    const size_t L = rt_.size();
    out.layers.resize(L);
    out.depth = 0;
    for (size_t l = 0; l < L; ++l) {
        const Ds4LayerCache& live = caches_[size_t(seq_) * L + l];
        if (std::string e = live.export_host(out.layers[l]); !e.empty()) {
            q_->wait();
            return "ds4 host snapshot: layer " + std::to_string(l) + ": " + e;
        }
    }
    q_->wait();   // every layer's D2H has landed in the pinned buffers
    out.depth = uint32_t(caches_[size_t(seq_) * L].cumulative_length());
    return {};
}

uint32_t DeepSeek4Runtime::restore_from_host(const Ds4HostSnapshot& in) {
    const size_t L = rt_.size();
    if (!q_ || in.depth == 0 || in.layers.size() != L) return 0;
    for (size_t l = 0; l < L; ++l) {
        Ds4LayerCache& live = caches_[size_t(seq_) * L + l];
        if (std::string e = live.import_host(in.layers[l]); !e.empty()) {
            std::fprintf(stderr, "[ds4-cache] host restore failed at layer %zu (%s); full prefill\n",
                         l, e.c_str());
            for (size_t k = 0; k < L; ++k) caches_[size_t(seq_) * L + k].reset();
            return 0;
        }
    }
    return in.depth;
}

// The four accessors below are ACTIVE-SLOT figures.  With one slot that is the
// whole cache, so every existing caller (the bench's kv_state_bytes_e2 line and
// its context-scaling table) reports exactly what it reported before.
uint64_t DeepSeek4Runtime::kv_state_bytes(uint32_t elem_bytes) const noexcept {
    return seq_state_bytes(seq_, elem_bytes);
}

uint64_t DeepSeek4Runtime::kv_growing_bytes(uint32_t elem_bytes) const noexcept {
    uint64_t n = 0;
    const size_t L = rt_.size();
    for (size_t l = 0; l < L; ++l)
        n += caches_[size_t(seq_) * L + l].growing_elements() * elem_bytes;
    return n;
}

// WHOLE-RUNTIME, unlike the three above: every slot's per-sequence buffers PLUS
// the shared scratch.  It has to be, to keep meaning what it meant — before the
// transient/persistent split every cache owned its own scratch and this total
// included it, so a one-slot runtime must still report the same number or an
// `ie-ds4-bench` comparison against any earlier run silently shifts.
uint64_t DeepSeek4Runtime::kv_allocated_bytes() const noexcept {
    uint64_t n = cache_scratch_bytes();
    for (const Ds4LayerCache& c : caches_) n += c.allocated_bytes();
    return n;
}

uint64_t DeepSeek4Runtime::seq_state_bytes(uint32_t slot, uint32_t elem_bytes) const noexcept {
    if (slot >= seq_slots_) return 0;
    uint64_t n = 0;
    const size_t L = rt_.size();
    for (size_t l = 0; l < L; ++l) n += caches_[size_t(slot) * L + l].state_bytes(elem_bytes);
    return n;
}

uint64_t DeepSeek4Runtime::seq_allocated_bytes(uint32_t slot) const noexcept {
    if (slot >= seq_slots_) return 0;
    uint64_t n = 0;
    const size_t L = rt_.size();
    for (size_t l = 0; l < L; ++l) n += caches_[size_t(slot) * L + l].allocated_bytes();
    return n;
}

uint64_t DeepSeek4Runtime::cache_scratch_bytes() const noexcept {
    uint64_t n = 0;
    for (const Ds4CacheScratch& s : scratch_) n += s.allocated_bytes();
    return n;
}

uint64_t DeepSeek4Runtime::context_length() const noexcept {
    return caches_.empty() ? 0 : layer_cache(0).cumulative_length();
}

// The one place a forward blocks on this card's queue.  Every drain below goes
// through it so that "how long was this card stuck waiting for its own GPU, and
// at which of the four barriers" is a measurement and not an estimate.  The
// timing is two steady_clock reads, ~40 ns, against a barrier that costs
// microseconds — it cannot move the number it reports by anything readable.
void DeepSeek4Runtime::drain(double& acc, uint64_t& n) {
    const auto t0 = std::chrono::steady_clock::now();
    q_->wait();
    acc += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    ++n;
}

std::string DeepSeek4Runtime::layer_forward_pre(uint32_t L, uint32_t T, uint32_t pos0) {
    sycl::queue& q  = *q_;
    Ds4LayerRT&  rt = rt_[L];
    // The ACTIVE sequence slot's layer-L cache.  Identical to `caches_[L]` when
    // there is one slot, which is the default.
    Ds4LayerCache& cache = layer_cache(L);
    const uint32_t H = cfg_.hidden, hc = cfg_.hc_count, HD = cfg_.head_dim;
    const uint32_t NH = cfg_.n_q_heads, QR = cfg_.q_lora_rank;
    // THIS CARD's query heads and output groups.  Equal to the model's when the
    // non-expert set is mirrored, so every call below is unchanged in that case.
    const uint32_t NHc = aslice_.nhc, GC = aslice_.gc;
    const uint32_t RD = cfg_.rope_dim;
    const float eps = cfg_.rms_eps;
    // Everything this function submits through dense_w/grouped_w is an ATTENTION
    // projection — the routed experts below go through ds4_expert_gemv, which
    // carries its own kernel names.  Profiler bookkeeping only; see Ds4SiteScope.
    const Ds4SiteScope site(Ds4Site::kAttn);

    // ---- attention-site hyper-connection ----
    // The attn_norm is folded into the hyper-connection's tail: `ws_coll_` has no
    // other consumer, and the fused form is bit-identical to the two calls it
    // replaces (deepseek4_ops_gate_test §hc_norm) in one fewer launch — measured
    // 9.25 -> 7.24 us per site at T=1, so it is cheaper on BOTH axes.
    ds4_hyper_connection_norm(q, ws_streams_, rt.hc_attn_fn, rt.hc_attn_base,
                              rt.hc_attn_scale, rt.attn_norm,
                              ws_post_, ws_comb_, ws_coll_, ws_norm_, T, H, hc,
                              cfg_.hc_sinkhorn_iters, eps, cfg_.hc_eps, eps);

    // Sliding layers use the "main" table, CSA/HCA the yarn "compress" one.
    const bool compressed_layer = rt.kind.has_compressor;
    const float* cosT = compressed_layer ? ws_ccos_ : ws_cos_;
    const float* sinT = compressed_layer ? ws_csin_ : ws_sin_;

    // ---- q path ----
    dense_w(q, ws_norm_, rt.q_a, ws_qres_, T);
    ds4_rms_norm(q, ws_qres_, rt.q_a_norm, ws_qres_, T, QR, eps);
    dense_w(q, ws_qres_, rt.q_b, ws_q_, T);
    if (act_dump_ && T == 1) act_dump(L, 0, ws_norm_);
    // NHc, not NH: `rt.q_b` produced only this card's heads.  The per-head
    // RMSNorm and RoPE are elementwise in the head axis, so running them over a
    // slice is bit-identical to running them over the whole and taking the slice.
    ds4_unweighted_rms_norm(q, ws_q_, ws_q_, T * NHc, HD, eps);
    ds4_rope_apply(q, ws_q_, cosT, sinT, ws_q_, T, NHc, HD, RD, +1.f);

    // ---- shared-KV path (K == V, single head) ----
    const uint64_t base_abs = cache.cumulative_length() - cache.sliding_len();
    float* kv_new = ws_kv_;                       // staged, then handed to the cache
    dense_w(q, ws_norm_, rt.kv, kv_new, T);
    ds4_rms_norm(q, kv_new, rt.kv_a_norm, kv_new, T, HD, eps);
    ds4_rope_apply(q, kv_new, cosT, sinT, kv_new, T, 1, HD, RD, +1.f);
    // NO q.wait() before the cache call, here or at any of the five sites like
    // it below.  `update_sliding` consumes `kv_new` only through memcpys ON THIS
    // QUEUE, which is in-order, so it already runs after the rope above; the
    // wait bought a host round trip and nothing else.  What it did NOT protect
    // is the host-side `sycl::free` inside the cache's geometric-growth realloc,
    // and that stays safe for a different reason: `caches_[L]` is touched only
    // by layer L, and layer L-1 ended in `layer_forward_post`'s drain, so no
    // command reading THIS layer's buffers is ever in flight when they are
    // reallocated.  That drain is therefore load-bearing and is kept.
    uint32_t n_sl = 0;
    const float* sliding = cache.update_sliding(kv_new, T, &n_sl);

    // ---- long-range compressor ----
    uint32_t n_c = 0;
    const void* compressed = nullptr;   // fp16 or fp32 in the cache (IE_DS4_KV16)
    if (rt.kind.has_compressor) {
        const uint32_t CD    = rt.kind.compressor_dim;
        const uint32_t ratio = rt.kind.compress_ratio;
        float* ckv   = ws_cmp_;
        float* cgate = ws_cmp_ + uint64_t(T) * CD;
        dense_w(q, ws_norm_, rt.comp_kv,   ckv,   T);
        dense_w(q, ws_norm_, rt.comp_gate, cgate, T);
        const Ds4CompressChunk ch =
            cache.store_compression_weights(Ds4CacheEntry::Compressor, ckv, cgate, T);
        const uint32_t n_win = ch.n_tokens / ratio;
        if (n_win) {
            Ds4OverlapSlice ov{};
            if (rt.kind.has_indexer)   // CSA only: the Ca/Cb overlap scheme
                ov = cache.update_overlap_state(Ds4CacheEntry::Compressor, ch.kv, ch.gate, n_win);
            ds4_compress_pool(q, ch.kv, ch.gate, rt.comp_ape,
                              ov.valid ? ov.kv : nullptr, ov.valid ? ov.gate : nullptr,
                              ws_pool_, n_win, ratio, HD, rt.kind.has_indexer);
            ds4_rms_norm(q, ws_pool_, rt.comp_norm, ws_pool_, n_win, HD, eps);
            const std::vector<int32_t> cp =
                ds4_compress_positions(n_win, ratio, ch.first_window_position);
            // KEPT.  `cp` is a stack-local that dies at the closing brace, and
            // the source of an outstanding H2D must outlive it.
            q.memcpy(ws_cpos_, cp.data(), cp.size() * 4);
            drain(waits_.other, waits_.n_other);
            ds4_rope_cos_sin(q, inv_comp_dev_, ws_cpos_, ws_kcos_, ws_ksin_, n_win, RD / 2, 1.f);
            ds4_rope_apply(q, ws_pool_, ws_kcos_, ws_ksin_, ws_pool_, n_win, 1, HD, RD, +1.f);
        }
        compressed = cache.update_compressor_states(Ds4CacheEntry::Compressor, ws_pool_, n_win, &n_c);
    }

    // ---- the kv-length bound, CHECKED BEFORE THE FIRST WRITE CUT TO IT ----
    // This used to sit below the indexer, eighteen lines after
    // `ds4_indexer_score` had already written [T, n_ik] into `ws_iscore_` — a
    // buffer allocated [max_seq, ws_kv_capacity()].  The bound was correct in
    // VALUE and useless in ORDER: an over-long context overran the workspace and
    // only then reported that it was going to.  Both operands (`n_sl` from the
    // sliding ring, `n_c` from the compressor) are already final here, which is
    // the earliest point the question can be asked at all.
    //
    // DIAGNOSTIC ORDER CHANGED, BEHAVIOUR DID NOT: a context that trips this
    // AND the indexer/compressor drift check below now reports this one first.
    // Both are hard refusals of the same forward pass.
    const uint32_t kvmax = ws_kv_capacity();
    const uint32_t kvc   = ws_compressed_capacity();   // the block bias / score strips
    const uint32_t n_kv  = n_sl + n_c;
    if (n_kv > kvmax || n_c > kvc)
        return "layer " + std::to_string(L) + ": kv length " + std::to_string(n_kv) +
               " (" + std::to_string(n_c) + " compressed) exceeds the workspace sizing " +
               std::to_string(kvmax) + " (" + std::to_string(kvc) +
               " compressed) (raise Ds4Options::max_context, currently " +
               std::to_string(opt_.max_context) + ")";

    // ---- lightning indexer (CSA layers only) ----
    // Entered on EVERY forward call for a CSA layer, even when no compressed
    // entry can be emitted yet.  The indexer keeps its own window buffer in the
    // cache, and skipping the call would leave that buffer missing this call's
    // tokens while the outer compressor's buffer advanced — the two would drift
    // apart permanently and every later window would pool the wrong tokens.
    uint32_t topk_used = 0;
    uint32_t n_ik = 0;
    if (rt.kind.has_indexer) {
        const uint32_t IHD = cfg_.indexer_head_dim, IH = cfg_.indexer_n_heads;
        const uint32_t ratio = rt.kind.compress_ratio;
        float* ikv   = ws_cmp_;
        float* igate = ws_cmp_ + uint64_t(T) * 2 * IHD;
        {   // ONE launch: both read ws_norm_ with the same K and write into
            // ws_cmp_ disjointly, so nothing orders them against each other.
            const Ds4Dense* mds[2] = {&rt.idx_kv, &rt.idx_gate};
            float*          mys[2] = {ikv, igate};
            if (!dense_multi_w(q, ws_norm_, mds, mys, 2, T)) {
                dense_w(q, ws_norm_, rt.idx_kv,   ikv,   T);
                dense_w(q, ws_norm_, rt.idx_gate, igate, T);
            }
        }
        const Ds4CompressChunk ich =
            cache.store_compression_weights(Ds4CacheEntry::Indexer, ikv, igate, T);
        const uint32_t n_win = ich.n_tokens / ratio;
        if (n_win) {
            const Ds4OverlapSlice ov =
                cache.update_overlap_state(Ds4CacheEntry::Indexer, ich.kv, ich.gate, n_win);
            ds4_compress_pool(q, ich.kv, ich.gate, rt.idx_ape,
                              ov.valid ? ov.kv : nullptr, ov.valid ? ov.gate : nullptr,
                              ws_pool_, n_win, ratio, IHD, true);
            ds4_rms_norm(q, ws_pool_, rt.idx_norm, ws_pool_, n_win, IHD, eps);
            const std::vector<int32_t> cp =
                ds4_compress_positions(n_win, ratio, ich.first_window_position);
            // KEPT, for the same reason as its twin in the compressor above.
            q.memcpy(ws_cpos_, cp.data(), cp.size() * 4);
            drain(waits_.other, waits_.n_other);
            ds4_rope_cos_sin(q, inv_comp_dev_, ws_cpos_, ws_kcos_, ws_ksin_, n_win, RD / 2, 1.f);
            ds4_rope_apply(q, ws_pool_, ws_kcos_, ws_ksin_, ws_pool_, n_win, 1, IHD, RD, +1.f);
        }
        const void* ikeys =
            cache.update_compressor_states(Ds4CacheEntry::Indexer, ws_pool_, n_win, &n_ik);
        const bool  keys_f16 = cache.compressed_f16(Ds4CacheEntry::Indexer);
        if (n_ik) {
            // The indexer's OWN bound, and it is not implied by the one above.
            // `ds4_indexer_score` writes [T, n_ik] into `ws_iscore_`, and n_ik
            // comes from a different window state machine than n_c did — the
            // two are required to agree, but that is checked BELOW, so relying
            // on it here would make the guard depend on the thing it guards.
            if (n_ik > kvc)
                return "layer " + std::to_string(L) + ": the indexer holds " +
                       std::to_string(n_ik) + " compressed entries, past the workspace sizing " +
                       std::to_string(kvc) +
                       " (raise Ds4Options::max_context, currently " +
                       std::to_string(opt_.max_context) + ")";
            // Queries: the indexer's own q_b over q_residual, RoPE'd at the
            // model's positions with the SAME (compress) theta as its keys.
            dense_w(q, ws_qres_, rt.idx_q_b, ws_iq_, T);
            ds4_rope_apply(q, ws_iq_, ws_ccos_, ws_csin_, ws_iq_, T, IH, IHD, RD, +1.f);
            dense_w(q, ws_norm_, rt.idx_proj, ws_iw_, T);
            topk_used = std::min(cfg_.indexer_top_k, n_ik);
            // STRIPPED over query rows (docs/deepseek4/73 Phase 1): `ws_iscore_`
            // is [ws_strip_, n_ik], not [T, n_ik] — at 1M context the latter was
            // 8 GB per card.  Both kernels do one query row per work-group and
            // index rows from the pointers they are handed, so a strip boundary
            // changes nothing numerically; `ws_topk_` rows are strided by
            // `topk_used`, which is ds4_indexer_topk's own row stride.
            for (uint32_t t0 = 0; t0 < T; t0 += ws_strip_) {
                const uint32_t ts = std::min(ws_strip_, T - t0);
                if (keys_f16)
                    ds4_indexer_score(q, ws_iq_ + uint64_t(t0) * IH * IHD,
                                      static_cast<const sycl::half*>(ikeys),
                                      ws_iw_ + uint64_t(t0) * IH, ws_iscore_, ts, IH, IHD, n_ik,
                                      1.f / std::sqrt(float(IHD)), 1.f / std::sqrt(float(IH)));
                else
                    ds4_indexer_score(q, ws_iq_ + uint64_t(t0) * IH * IHD,
                                      static_cast<const float*>(ikeys),
                                      ws_iw_ + uint64_t(t0) * IH, ws_iscore_, ts, IH, IHD, n_ik,
                                      1.f / std::sqrt(float(IHD)), 1.f / std::sqrt(float(IH)));
                ds4_indexer_topk(q, ws_iscore_, ws_pos_ + t0, ws_topk_ + uint64_t(t0) * topk_used,
                                 ts, n_ik, cfg_.indexer_top_k, ratio);
            }
        }
        // No drain here either: `ws_topk_` is consumed by ds4_block_bias_topk on
        // this same in-order queue, and everything between is host arithmetic on
        // values (`n_ik`, `n_c`, `topk_used`) that are already on the host.
    }
    if (rt.kind.has_indexer && n_ik != n_c)
        return "layer " + std::to_string(L) + ": indexer emitted " + std::to_string(n_ik) +
               " entries but the compressor emitted " + std::to_string(n_c) +
               " — the two window state machines have drifted apart";

    // ---- mask ----
    // `n_kv` and `kvmax` were computed and checked above, before the first write
    // cut to them.
    // The sliding cache is a RING: column k of `sliding` is absolute position
    // base_abs + k, so the causal comparison uses positions[t] - base_abs.
    {
        // Staged through a MEMBER buffer, not a stack-local, so the H2D does not
        // have to be waited for: the source has to outlive the copy, and this
        // one does.  The next host write to it is layer L+1's, which cannot
        // happen until `layer_forward_post` has drained layer L.
        for (uint32_t t = 0; t < T; ++t)
            h_spos_[t] = int32_t(int64_t(pos0 + t) - int64_t(base_abs));
        q.memcpy(ws_spos_, h_spos_.data(), uint64_t(T) * 4);
    }
    if (vis_chunk_)
        ds4_sliding_causal_mask_vis(q, ws_spos_, ws_vleft_, ws_vright_, ws_mask_, T, n_sl,
                                    cfg_.sliding_window);
    else
        ds4_sliding_causal_mask(q, ws_spos_, ws_mask_, T, n_sl, cfg_.sliding_window);
    // kv = cat([sliding, compressed], dim=kv) — READ IN PLACE (Ds4KvSegs,
    // docs/deepseek4/73 Phase 1).  This used to be two device memcpys into
    // `ws_kv_` on every forward, the second of them the ENTIRE compressed
    // cache: ~1.1 GB per decoded token at 100K context, 10.7 GB at 1M.  The
    // kernels walk the same column order over the two segments, so nothing
    // numeric moved.  Both pointers are cache-owned device USM that the next
    // forward of THIS layer is the first thing to touch again.
    //
    // Every head this card owns attends over the SAME shared K/V — `head_count_kv`
    // is 1, so the segments, the mask and the compressed entries are whole here
    // even when the query heads are split.  `rt.sinks` is already this card's
    // slice, so ds4_attention's sinks[h] indexes [0, NHc) and needs no offset.
    const Ds4KvSegs segs{sliding, n_sl, compressed, n_c,
                         cache.compressed_f16(Ds4CacheEntry::Compressor)};
    const float     attn_scale = 1.f / std::sqrt(float(HD));
    if (n_c) {
        // STRIPPED over query rows: `ws_bias_` is [ws_strip_, n_c] and
        // `ws_maskc_` [ws_strip_, n_kv].  One query row per work-group in every
        // kernel below, so the strip boundary is invisible to the arithmetic;
        // the XMX route stages the fp16 kv axis on the first strip and reuses
        // it (`xmx_kv_prepared`) on the rest — per layer, not per strip.
        for (uint32_t t0 = 0; t0 < T; t0 += ws_strip_) {
            const uint32_t ts = std::min(ws_strip_, T - t0);
            if (rt.kind.has_indexer) {
                ds4_block_bias_topk(q, ws_topk_ + uint64_t(t0) * topk_used, ws_bias_, ts, n_c,
                                    topk_used);
            } else if (T == 1) {
                // Reference: HCA returns block_bias = None when seq_len == 1, and the
                // attention then F.pads the mask with 0.0 — every compressed entry is
                // visible.  Reproducing this exactly matters: a dense causal bias here
                // would hide the newest entries during decode.
                q.memset(ws_bias_, 0, uint64_t(ts) * n_c * sizeof(float));
            } else {
                ds4_block_bias_dense(q, ws_pos_ + t0, ws_bias_, ts, n_c, rt.kind.compress_ratio);
            }
            concat_mask(q, ws_mask_ + uint64_t(t0) * n_sl, ws_bias_, ws_maskc_, ts, n_sl, n_c);
            ds4_attention_segs(q, ws_q_ + uint64_t(t0) * NHc * HD, segs, ws_maskc_, rt.sinks,
                               ws_attn_ + uint64_t(t0) * NHc * HD, ts, NHc, HD, attn_scale,
                               /*xmx_kv_prepared=*/t0 != 0);
        }
    } else {
        // Sliding-only layers: the [T, n_sl] causal mask IS the attention mask.
        ds4_attention_segs(q, ws_q_, segs, ws_mask_, rt.sinks, ws_attn_, T, NHc, HD, attn_scale);
    }
    // K == V, so V carried RoPE on its trailing slice: undo it with the
    // conjugate rotation at the QUERY position before the heads are mixed.
    ds4_rope_apply(q, ws_attn_, cosT, sinT, ws_attn_, T, NHc, HD, RD, -1.f);
    // GC groups, not o_groups.  The INPUTS PER GROUP are unchanged — a group
    // still reads NH*HD/o_groups values — because splitting the head axis
    // removes whole groups, never part of one.
    grouped_w(q, ws_attn_, rt.o_a, ws_grp_, T, GC,
                NH * HD / cfg_.o_groups, cfg_.o_lora_rank);
    dense_w(q, ws_grp_, rt.o_b, ws_sub_, T);
    // END OF PHASE 0.  With the non-expert set split, `ws_sub_` is a PARTIAL sum
    // over this card's output groups and the orchestrator reduces it before
    // `layer_forward_mid` runs.  Mirrored, it is already whole and no reduction
    // is issued — which is the entire difference in comms cost between the two
    // configurations.
    return {};
}

// ---------------------------------------------------------------------------
// The routed-expert block, EXPERT-GROUPED.  See the contract on the declaration
// in include/ie/deepseek4.hpp — in particular why this is bit-identical to one
// whole-batch `ds4_experts_forward_batched` call, and why it can never issue
// more acquires, H2D bytes or GEMM launches than the token-chunked predecessor.
//
// `t0`/`nt` are a token range of the CURRENT layer's batch; `EFc` is this card's
// intermediate slice width.  `ws_moe_` is accumulated into and is NOT zeroed
// here — the caller owns that, exactly as `ds4_experts_forward_batched` does.
// ---------------------------------------------------------------------------
// IE_DS4_HOST_TRACE=1: where the HOST's time goes inside moe_expert_grouped for
// prefill-sized calls (nt >= 64), averaged per call and printed every 43 calls
// (one chunk of layers).  Diagnostic for docs/deepseek4/72 Phase F.
namespace {
struct Ds4HostTrace {
    bool on = std::getenv("IE_DS4_HOST_TRACE") != nullptr;
    double pack = 0, plan = 0, issue = 0, dep = 0, jobs = 0, gu = 0, act = 0, dn = 0, drain = 0, spec = 0, scatter = 0, total = 0;
    uint64_t calls = 0, groups = 0;
    static double now() {
        return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    void report() {
        const double k = 1e3 / double(calls);
        std::fprintf(stderr,
                     "[ds4-host] moe_expert_grouped avg per call (ms): total %.2f | pack %.2f plan %.2f | "
                     "issue %.2f dep %.2f jobs %.2f gu-submit %.2f act %.2f down %.2f drain %.2f spec %.2f | scatter %.2f | "
                     "%.1f groups/call\n",
                     total * k, pack * k, plan * k, issue * k, dep * k, jobs * k, gu * k, act * k, dn * k, drain * k,
                     spec * k, scatter * k, double(groups) / double(calls));
    }
};
Ds4HostTrace g_host_trace;
}  // namespace

std::string DeepSeek4Runtime::moe_expert_grouped(uint32_t L, uint32_t t0, uint32_t nt,
                                                 uint32_t EFc, float swiglu_limit) {
    Ds4HostTrace& HT = g_host_trace;
    const bool ht = HT.on && nt >= 64;
    double ht_t = ht ? Ds4HostTrace::now() : 0, ht_start = ht_t;
    auto lap = [&](double& acc) { if (ht) { const double n2 = Ds4HostTrace::now(); acc += n2 - ht_t; ht_t = n2; } };
    sycl::queue&      q  = *q_;
    const Ds4LayerRT& rt = rt_[L];
    const uint32_t    H  = cfg_.hidden;
    const uint32_t    K  = cfg_.n_experts_used;
    const uint32_t    E  = cfg_.n_experts;
    if (nt == 0) return {};
    if (nt > bws_.max_tokens || K != bws_.top_k || H != bws_.H || EFc > bws_.EF)
        return "layer " + std::to_string(L) + ": batch workspace too small (nt=" +
               std::to_string(nt) + "/" + std::to_string(bws_.max_tokens) + ", top_k=" +
               std::to_string(K) + "/" + std::to_string(bws_.top_k) + ", H=" +
               std::to_string(H) + "/" + std::to_string(bws_.H) + ", EFc=" +
               std::to_string(EFc) + "/" + std::to_string(bws_.EF) + ")";
    if (EFc % 32 != 0)
        return "layer " + std::to_string(L) + ": EFc=" + std::to_string(EFc) + " must be % 32";

    // ---- host counting sort over the WHOLE range: rows grouped by expert ----
    const int32_t* idx = h_ridx_ + uint64_t(t0) * K;
    const float*   rw  = h_rw_   + uint64_t(t0) * K;
    bws_.routes.resize(nt);
    for (uint32_t t = 0; t < nt; ++t) {
        bws_.routes[t].clear();
        bws_.routes[t].reserve(K);
        for (uint32_t k = 0; k < K; ++k) {
            const int32_t e = idx[uint64_t(t) * K + k];
            if (e < 0 || uint32_t(e) >= E)
                return "layer " + std::to_string(L) + ": expert id " + std::to_string(e) +
                       " outside [0," + std::to_string(E) + ")";
            bws_.routes[t].push_back({uint32_t(e), rw[uint64_t(t) * K + k]});
        }
    }
    build_moe_packing(bws_.routes, E, K, bws_.pk);
    const uint32_t TK = nt * K;

    q.memcpy(bws_.row_tok, bws_.pk.sorted_idx.data(),     TK * sizeof(int32_t));
    q.memcpy(bws_.tk2p,    bws_.pk.tk_to_packed.data(),   TK * sizeof(int32_t));
    q.memcpy(bws_.w_pk,    bws_.pk.weights_packed.data(), TK * sizeof(float));

    // ---- one activation pass for the WHOLE range -----------------------------
    ds4_expert_gather_cast(q, ws_norm_ + uint64_t(t0) * H, bws_.row_tok, bws_.xp, TK, H);
    quantize_q8_1(q, bws_.xp, bws_.xp_q8, TK * H);

    // The occupied experts, ASCENDING — which is also the packed row order, so a
    // consecutive run of them is a contiguous run of rows.
    const std::vector<uint32_t>& off = bws_.pk.expert_offsets;
    std::vector<uint32_t> occ;
    occ.reserve(E);
    for (uint32_t e = 0; e < E; ++e)
        if (off[e + 1] > off[e]) occ.push_back(e);

    lap(HT.pack);
    auto* xq8 = static_cast<block_q8_1x*>(bws_.xp_q8);
    auto* hq8 = static_cast<block_q8_1x*>(bws_.h_q8);
    const uint32_t stream = cache_.stream_slots();

    // ---- THE FETCH PIPELINE ------------------------------------------------
    // Group g+1's H2D is issued BEFORE group g's GEMMs run, so the copy engine
    // and the EUs overlap instead of alternating.  Two groups are therefore live
    // at once, which is why a group may claim at most HALF the streaming
    // partition and why the fetch has to go through `acquire_pipelined` — plain
    // `acquire` clears the in-use marks every call and would hand group g+1 a
    // slot group g's GEMMs are still reading.
    //
    // `pipelined_group_cap() == 0` means the streaming partition is too small to
    // double-buffer (fewer than 2 evictable slots); that falls back to the exact
    // serial shape, cap and all, rather than to a cap of zero.
    // Prefill BANKS (docs/deepseek4/72 Phase A): with banks the group cap is
    // the bank size, a miss fills a bank slot, a hit reads its LRU slot with no
    // recency bump, and the LRU is never evicted by a prefill sweep.
    // Banks are for PREFILL-scale sweeps only: a decode step (nt small) must
    // keep evicting into the LRU or the cache never learns the conversation's
    // working set — measured 2026-09-01: routing decode through the banks gave
    // 10.5 tok/s at hit 0.00 instead of 27.9 at 0.87.
    const uint32_t bank  = nt >= kDs4BankMinTokens ? cache_.bank_slots() : 0u;
    const uint32_t gcap0 = cache_.pipelined_group_cap();
    const bool     pipe  = gcap0 != 0 || bank != 0;
    const uint32_t gcap  = bank ? bank : gcap0;
    std::vector<uint32_t> gbnd;
    // HITS FIRST on the bank path (docs/deepseek4/72 Phase J).  In expert-id
    // order every layer's first group carried ~18 misses, and no expert GEMM
    // could start until that group's ~4.5 ms of DMA had landed — measured as
    // 36.8 ms of drain per layer against 26.5 ms of DMA.  The resident experts
    // (~45% at prefill) need no transfer, so they go first as ONE bank-free
    // group whose GEMMs run while the first miss group streams, and the misses
    // follow in bank-sized groups.  Bytes cannot change: the scatter runs once
    // over the whole range in kslot order, and the groups only pick when each
    // expert's rows are computed.  Decode (bank == 0) keeps the id order and
    // its LRU touch order.  IE_DS4_HITS_FIRST=0 restores the id order.
    static const int hits_mode = [] {   // 0 = id order, 1 = hits first + early issue, 2 = hits first only (bisection knob)
        const char* v = std::getenv("IE_DS4_HITS_FIRST");
        return (v && *v) ? std::atoi(v) : 1;
    }();
    const bool hits_first = hits_mode != 0;
    uint32_t n_hits_group = 0;
    if (bank && hits_first) {
        std::vector<uint32_t> ordered;
        n_hits_group = ds4_order_hits_first(cache_, L, occ.data(), occ.size(), gcap, ordered, gbnd);
        occ.swap(ordered);
    } else {
        ds4_plan_expert_groups(cache_, L, occ.data(), occ.size(), pipe ? gcap : stream, gbnd);
    }
    const size_t NG = gbnd.empty() ? 0 : gbnd.size() - 1;
    lap(HT.plan);

    // Per GROUP, not per bank: with the hits group in front, group 2's fetch is
    // issued while group 0 is still being computed (below), so two groups of
    // the same bank parity are live on the host at once.  The bank itself is
    // still g & 1 — group 0 never writes its bank, which is what makes that
    // early issue safe.
    std::vector<std::vector<int32_t>>  gids(NG);
    std::vector<std::vector<uint32_t>> gslot(NG);
    std::vector<sycl::event>           gev(NG);
    std::string           gerr;
    auto issue = [&](size_t g) {
        const uint32_t s = uint32_t(g & 1);
        gids[g].assign(occ.begin() + std::ptrdiff_t(gbnd[g]),
                       occ.begin() + std::ptrdiff_t(gbnd[g + 1]));
        gslot[g].assign(gids[g].size(), kDs4NoSlot);
        gev[g] = bank ? cache_.acquire_prefill(L, gids[g].data(), uint32_t(gids[g].size()),
                                               gslot[g].data(), s)
              : pipe  ? cache_.acquire_pipelined(L, gids[g].data(), uint32_t(gids[g].size()),
                                                 gslot[g].data())
                      : cache_.acquire(L, gids[g].data(), uint32_t(gids[g].size()), gslot[g].data());
        // An expert with neither a VRAM home nor a pinned host copy cannot be
        // computed.  There is no substitute for it that is not a fabricated
        // output, so this is a hard error rather than a skipped term.  It is
        // recorded and returned at the top of the loop body rather than thrown,
        // because the fetch for group g+1 is issued from inside group g.
        for (size_t i = 0; i < gslot[g].size(); ++i)
            if (gslot[g][i] == kDs4NoSlot && gerr.empty())
                gerr = "layer " + std::to_string(L) + ": expert " + std::to_string(gids[g][i]) +
                       " is neither VRAM-resident (" + std::to_string(cache_.static_slots()) +
                       " static slots, filled in residency-priority order) nor host-pinned (" +
                       std::to_string(arena_.n_pinned_experts()) + " experts pinned, lowest id " +
                       std::to_string(arena_.first_pinned_expert()) +
                       ") — refusing to fabricate an expert output";
    };

    // Set by any group that took the XMX `down` route: its result is f32 in
    // `y_f32`, so the scatter must read that instead of the fp16 `yp`.  The
    // route is all-or-nothing across a chunk's groups (the predicate depends
    // only on layer-level dtypes), so one flag is exact, not a heuristic.
    bool xmx_down_used = false;
    size_t issued = 0;
    auto issue_upto = [&](size_t upto) { while (issued < upto && issued < NG) issue(issued++); };
    if (pipe && NG) issue_upto(1);
    lap(HT.issue);
    for (size_t g = 0; g < NG; ++g) {
        if (!pipe) issue_upto(g + 1);
        else {
            issue_upto(g + 2);     // <-- group g+1's fetch overlaps the compute below
            // Phase J: group 0 is the hits group — no transfer, its bank never
            // written — so bank 0 is free and group 2's fetch goes out now too:
            // the transfer queue never idles between group 1 and group 2.  From
            // here on the two-bank schedule is the usual one (group g+1 issued
            // at iteration g, after group g-1's GEMMs drained).
            if (g == 0 && n_hits_group && hits_mode == 1) issue_upto(3);
        }
        if (!gerr.empty()) return gerr;
        lap(HT.issue);
        const sycl::event fetched = gev[g];
        // The compute queue does not start reading a slot before its H2D lands.
        // This single dependency IS the DMA/compute overlap mechanism.
        q.submit([&](sycl::handler& h) { h.depends_on(fetched); h.single_task([](){}); });
        lap(HT.dep);

        // ---- gate + up, ONE LAUNCH for the whole group -----------------------
        // Was one GEMM per expert per projection.  Merging them WITHIN a group
        // is what fixes grid starvation: a single expert's gate at N=EFc fills
        // exactly one work-group per Xe core, leaving nothing resident to hide
        // memory latency behind.  Measured 19.97 -> 9.13 ms per layer-chunk
        // (85.7 -> 187 GB/s) and 768 -> 2 launches, bit-identical.
        //
        // Grouping does NOT cross the `q.wait()` eviction barrier below — the
        // job list is rebuilt per group — so slot lifetime is unchanged.
        bws_.jobs.clear();
        for (size_t i = 0; i < gids[g].size(); ++i) {
            void* base = cache_.slot_ptr(L, gslot[g][i]);
            const DS4ExpertBank gb = ds4_slot_bank(rt.slot.gate, base);
            const DS4ExpertBank ub = ds4_slot_bank(rt.slot.up,   base);
            const DS4ExpertBank db = ds4_slot_bank(rt.slot.down, base);
            if (gb.K != H || gb.N != EFc || ub.K != H || ub.N != EFc ||
                db.K != EFc || db.N != H)
                return "layer " + std::to_string(L) + ": expert " + std::to_string(gids[g][i]) +
                       " bank shape mismatch (gate [" + std::to_string(gb.K) + "," +
                       std::to_string(gb.N) + "], down [" + std::to_string(db.K) + "," +
                       std::to_string(db.N) + "] vs H=" + std::to_string(H) + " EFc=" +
                       std::to_string(EFc) + ")";
            const uint32_t o = off[gids[g][i]], n_e = off[gids[g][i] + 1] - o;
            const void*    xe = xq8 + uint64_t(o) * (H / 32);
            bws_.jobs.push_back({gb, 0, n_e, xe, bws_.g_h + uint64_t(o) * EFc, o});
            bws_.jobs.push_back({ub, 0, n_e, xe, bws_.u_h + uint64_t(o) * EFc, o});
        }
        // XMX ROUTE (opt-in, $IE_DS4_EXPERT_XMX=1, PREFILL ONLY).  oneDNN f16
        // matmul over a dequantised weight instead of the W4A8 int-dot kernel:
        // measured 11.5-32 TFLOP/s against the int-dot kernel's ~3 ceiling, and
        // 4 ORDERS OF MAGNITUDE more accurate (int-dot quantises the activation
        // to int8; this does not) — the XMX gate measures max abs error 1.7e-4
        // vs 9.8 at the real gate/up shape.  The June ban (gptoss_tp.cpp:703)
        // was re-probed 2026-08-09 on the current stack: 160 trials, M in
        // [1,24], zero gross errors — the small-M corruption does not
        // reproduce.  Decode (T == 1) is deliberately NOT routed here: it is
        // DMA-bound, already has its own kernel, and staying on int-dot keeps
        // every decode number in this document bit-comparable.
        // Only gate/up move: they write f32 DIRECTLY into g_f/u_f (the buffers
        // the fp16 path would cast into), so the route also deletes two casts
        // per group.  `down` stays on int-dot — its f16 output feeds the
        // scatter unchanged, and moving it would need a new R*H f32 landing.
        // Decided BEFORE the SwiGLU chain because the chain's last stage
        // (quantize_q8_1) exists only for the int-dot down route.
        // DOWN STAYS INT-DOT — measured twice, both ways (2026-08-09):
        //   with the fp16 cast-back : pp512 127.9  pp2048 170.8
        //   with the f32 scatter    : pp512 128.6  pp2048 172.9
        //   int-dot down (shipped)  : pp512 151.1  pp2048 180.9
        // Deleting the cast AND the quantize (ds4_expert_scatter_accum_f32 was
        // built for exactly that) recovered only ~2 points, so the loss is not
        // the epilogue — it is the DEQUANT ITSELF: materialising K*N halves
        // WRITES ~5x the bytes the int-dot kernel READS of the packed weight.
        // gate/up survive that because they additionally delete two R*EF casts
        // per group; `down` has no such bonus.  Opt-in instrument only.
        //
        // 2026-09-10 RE-MEASURED on the FUSED v1 kernel (the numbers above were
        // the dequant+oneDNN route; the only fused-down number in
        // docs/deepseek4/72 was taken on the falsified v2 kernel). Two A/B pairs
        // on the Vision-Exp GGUF, 2 cards, ie-ds4-bench as the 09-09 campaign
        // ran it (--pp 128,512,4096 --ctx 128,4096 --decode 32 --warmup 8):
        //   pp512  int-dot {271.3, 257.7} -> fused {278.9, 285.3} tok/s (+6.7% median)
        //   pp4096 int-dot {576.7, 539.3} -> fused {613.3, 568.0} tok/s (+5.9% median)
        //   tg128 unchanged (32.7-32.9); tg4096 26.5/26.2 -> 24.6/24.6 with a LOWER
        //   expert-cache hit rate (0.837 -> 0.794): T == 1 never enters this route;
        //   the prefill's numerics shift which experts the bench's fixed synthetic
        //   sequence routes to, and its decode inherits that cache.
        //   wikitext batch PPL 4.1565 -> 4.1393 (511 tok; better, as gate/up were:
        //   no int8 activation quantisation), nonfinite 0.
        // DEFAULT ON since then. IE_DS4_EXPERT_XMX_DOWN=0 restores int-dot down.
        static const bool xmx_dn_env = [] {
            const char* e = std::getenv("IE_DS4_EXPERT_XMX_DOWN");
            return !(e && *e == '0');
        }();
        const bool xmx_gu = ds4_expert_xmx_on() && nt > 1 &&
                            rt.slot.gate.dt == DType::kMXFP4 &&
                            rt.slot.up.dt == DType::kMXFP4;
        const bool xmx_dn_planned = xmx_gu && xmx_dn_env &&
                                    rt.slot.down.dt == DType::kMXFP4 && bws_.y_f32;
        lap(HT.jobs);
        if (xmx_gu) {
            // `row0` was recorded when the job was built; the gate/up pair
            // alternates, so even indices are gate.  Deriving either from
            // POINTER COMPARISON across the two separate allocations was a real
            // bug (UB, and DEVICE_LOST on the mini fixture where the allocator
            // ordered u_h below g_h) — caught by deepseek4_sched_gate_test.
            static thread_local std::vector<DS4XmxGemmJob> xmx_jobs;
            xmx_jobs.clear();
            for (size_t ji = 0; ji < bws_.jobs.size(); ++ji) {
                const auto& J = bws_.jobs[ji];
                float* dst = ((ji & 1) ? bws_.u_f : bws_.g_f) + uint64_t(J.row0) * EFc;
                xmx_jobs.push_back({J.bank, J.e, J.M, J.row0, dst});
            }
            if (std::string e = ds4_expert_gemm_xmx_grouped(
                    q, xmx_jobs.data(), uint32_t(xmx_jobs.size()), bws_.xp, H,
                    bws_.w16, bws_.w16_cap); !e.empty())
                return "layer " + std::to_string(L) + ": " + e;
        } else if (std::string e = ds4_expert_gemm_q8_grouped(
                q, bws_.jobs.data(), uint32_t(bws_.jobs.size()), bws_.grp); !e.empty())
            return "layer " + std::to_string(L) + ": " + e;

        // ---- the SwiGLU chain over the packed row range [off[first], off[last+1]) ------
        // Every stage below is elementwise or blockwise over a row length that
        // is a multiple of 32, and `r0`/`nr` are whole rows, so this produces the
        // same bits as one pass over the whole batch would.  Under the hits-first
        // order (Phase J) a group's experts are not contiguous in id space, so
        // the range also covers OTHER groups' rows: their h is recomputed from
        // their own g/u by their own group before their down GEMM reads it, so
        // bytes are unaffected — the cost is redundant elementwise work (~1 ms
        // per layer at T = 1024; a job-list act kernel would remove it).
        lap(HT.gu);
        const uint32_t r0 = off[gids[g].front()];
        const uint32_t nr = off[gids[g].back() + 1] - r0;
        const uint64_t HN = uint64_t(nr) * EFc;
        if (!xmx_gu) {
            // int-dot gate/up produced fp16 in g_h/u_h: the fused fp16 SwiGLU
            // (exactly gated against the four-launch chain, deepseek4_ops.hpp)
            // writes h_h in ONE launch instead of cast+cast+swiglu+cast — three
            // launches fewer per group, 258 fewer per decode token over 2 cards.
            ds4_swiglu_clamped_h(q, bws_.g_h + uint64_t(r0) * EFc, bws_.u_h + uint64_t(r0) * EFc,
                                 bws_.h_h + uint64_t(r0) * EFc, HN, swiglu_limit);
        } else {   // the XMX route already produced f32 in g_f/u_f
            ds4_swiglu_clamped_to_f16(q, bws_.g_f + uint64_t(r0) * EFc, bws_.u_f + uint64_t(r0) * EFc,
                                      bws_.h_h + uint64_t(r0) * EFc, HN, swiglu_limit);
        }
        if (!xmx_dn_planned)   // the XMX down route consumes h_h (fp16) directly
            quantize_q8_1(q, bws_.h_h + uint64_t(r0) * EFc, hq8 + uint64_t(r0) * (EFc / 32),
                          uint32_t(HN));

        // ---- down, ONE LAUNCH -----------------------------------------------
        lap(HT.act);
        bws_.jobs.clear();
        for (size_t i = 0; i < gids[g].size(); ++i) {
            const DS4ExpertBank db = ds4_slot_bank(rt.slot.down, cache_.slot_ptr(L, gslot[g][i]));
            const uint32_t o = off[gids[g][i]], n_e = off[gids[g][i] + 1] - o;
            bws_.jobs.push_back({db, 0, n_e, hq8 + uint64_t(o) * (EFc / 32),
                                 bws_.yp + uint64_t(o) * H, o});
        }
        // `down` takes the same XMX route when it is on: oneDNN lands f32 in
        // `y_f32`, which is cast back into `yp` for the scatter (one extra
        // elementwise pass over the group's rows, against a 4-10x GEMM).
        const bool xmx_dn = xmx_dn_planned;
        if (xmx_dn) {
            static thread_local std::vector<DS4XmxGemmJob> xmx_jobs;
            xmx_jobs.clear();
            for (const auto& J : bws_.jobs)
                xmx_jobs.push_back({J.bank, J.e, J.M, J.row0, bws_.y_f32 + uint64_t(J.row0) * H});
            if (std::string e = ds4_expert_gemm_xmx_grouped(
                    q, xmx_jobs.data(), uint32_t(xmx_jobs.size()), bws_.h_h, EFc,
                    bws_.w16, bws_.w16_cap); !e.empty())
                return "layer " + std::to_string(L) + ": " + e;
            // NO cast back to fp16: the scatter reads f32 directly
            // (ds4_expert_scatter_accum_f32).  Deleting that R*H pass — and the
            // quantize_q8_1 the int-dot route needs — is what makes the XMX
            // down route viable at all; with the cast it measured a net LOSS.
            xmx_down_used = true;
        } else if (std::string e = ds4_expert_gemm_q8_grouped(
                q, bws_.jobs.data(), uint32_t(bws_.jobs.size()), bws_.grp); !e.empty())
            return "layer " + std::to_string(L) + ": " + e;

        // KEPT, AND IT IS THE EVICTION-SAFETY BARRIER.  The next group's
        // `acquire` writes an evicted slot through the TRANSFER queue, which this
        // in-order compute queue does not order against; the only thing standing
        // between a slot being overwritten and a GEMM still reading it is this
        // drain between groups.  Removing it corrupts an expert's weights
        // silently.
        lap(HT.dn);
        drain(waits_.moe, waits_.n_moe);
        lap(HT.drain);
        // Idle-time-only speculation for the NEXT layer, using this layer's first
        // token's ids as the predictor.  Declined outright unless the transfer
        // queue is empty, so it can never delay a demand fetch.
        if (spec_this_pass_ && L + 1 < pinned_layers_ && stream)
            cache_.speculate(L + 1, h_ridx_, std::min<uint32_t>(K, stream));
        lap(HT.spec);
        if (ht) ++HT.groups;
    }

    // ---- routing weight AFTER down, then the fp32 accumulate ----------------
    // ONE call over the whole range, ascending kslot, exactly as the whole-batch
    // path does — this is the only cross-row reduction and grouping never splits
    // it.
    if (xmx_down_used)
        ds4_expert_scatter_accum_f32(q, bws_.y_f32, bws_.tk2p, bws_.w_pk,
                                     ws_moe_ + uint64_t(t0) * H, nt, K, H);
    else
    ds4_expert_scatter_accum(q, bws_.yp, bws_.tk2p, bws_.w_pk, ws_moe_ + uint64_t(t0) * H,
                             nt, K, H);
    lap(HT.scatter);
    if (ht) {
        HT.total += Ds4HostTrace::now() - ht_start;
        if (++HT.calls % 43 == 0) HT.report();
    }
    return {};
}

std::string DeepSeek4Runtime::layer_forward_mid(uint32_t L, uint32_t T) {
    sycl::queue& q  = *q_;
    Ds4LayerRT&  rt = rt_[L];
    const uint32_t H = cfg_.hidden, hc = cfg_.hc_count;
    const float eps = cfg_.rms_eps;
    const Ds4SiteScope site(Ds4Site::kAttn);

    ds4_hc_mix(q, ws_streams_, ws_post_, ws_comb_, ws_sub_, ws_streams2_, T, H, hc);

    // ---- FFN-site hyper-connection ----
    // Same fusion as the attention site above.
    ds4_hyper_connection_norm(q, ws_streams2_, rt.hc_ffn_fn, rt.hc_ffn_base,
                              rt.hc_ffn_scale, rt.ffn_norm,
                              ws_post_, ws_comb_, ws_coll_, ws_norm_, T, H, hc,
                              cfg_.hc_sinkhorn_iters, eps, cfg_.hc_eps, eps);
    if (act_dump_ && T == 1) act_dump(L, 1, ws_norm_);

    // ---- routing ----
    const uint32_t K = cfg_.n_experts_used;
    if (vis_chunk_) {
        // Image rows route with the Vision-Exp bias on EVERY layer (hash layers
        // included); text rows exactly as below.  Text-only files have no such
        // bias, and an image request on one is a named refusal, not a guess.
        if (!rt.router_bias_vl)
            return "layer " + std::to_string(L) + ": image rows in this chunk but the GGUF carries no "
                   "exp_probs_b_vl.bias — not a Vision-Exp file";
        if (rt.kind.hash_router)
            ds4_router_hash_vl(q, ws_norm_, rt.router_w, rt.tid2eid, ws_ids_,
                               rt.router_bias_vl, ws_imgmask_,
                               ws_rlogit_, ws_rw_, ws_ridx_, T, H, cfg_.n_experts, K,
                               cfg_.expert_weights_scale);
        else
            ds4_router_topk_vl(q, ws_norm_, rt.router_w, rt.router_bias,
                               rt.router_bias_vl, ws_imgmask_,
                               ws_rlogit_, ws_rw_, ws_ridx_, T, H, cfg_.n_experts, K,
                               cfg_.expert_weights_scale);
    } else if (rt.kind.hash_router)
        ds4_router_hash(q, ws_norm_, rt.router_w, rt.tid2eid, ws_ids_,
                        ws_rlogit_, ws_rw_, ws_ridx_, T, H, cfg_.n_experts, K,
                        cfg_.expert_weights_scale);
    else
        ds4_router_topk(q, ws_norm_, rt.router_w, rt.router_bias,
                        ws_rlogit_, ws_rw_, ws_ridx_, T, H, cfg_.n_experts, K,
                        cfg_.expert_weights_scale);
    // KEPT, and it is the one wait in this function the whole design rests on:
    // the chunking below, the residency questions, the profile and the TP
    // lockstep check are all HOST reads of what the router decided.
    // THE DESTINATION IS PINNED, AND THE HOP IS STILL A `memcpy`.  BOTH HALVES
    // OF THAT WERE MEASURED, AND THE SECOND HALF IS A FALSIFICATION.
    //
    // This readback has the same shape that made the cross-card reduction slow:
    // a small device-to-host hop on the compute queue followed immediately by a
    // blocking drain, on the decode critical path, once per layer.  The
    // reduction's fix was to move the hop off the copy engine, and in isolation
    // that works here too — T*K = 8 elements, drained, four forms rotated
    // through the first position, one expert-sized burst per readback:
    //
    //   stream    | memcpy->pageable | kernel->pageable | memcpy->pinned | kernel->pinned
    //   ----------+------------------+------------------+----------------+---------------
    //    idle     |     27.15 us     |     12.79 us     |    17.28 us    |    12.39 us
    //   2.0 GB/s  |     34.01 us     |    217.68 us     |    26.96 us    |    13.76 us
    //  24.5 GB/s  |    173.23 us     |      6.68 us     |   166.19 us    |     6.96 us
    //
    // The kernel column is 24x faster under a saturating stream and it was
    // REJECTED anyway, because in the whole forward it breaks something the
    // microbenchmark cannot see.  With the two builds interleaved run-for-run
    // under identical load, deepseek4_residency_test §15h — every layer segment
    // of a two-card decode must have card 1's interval intersect card 0's —
    // went 13/14 passes with this `memcpy` and 3/14 with the kernel.  Losing
    // card overlap idles a 32 GB GPU for part of every segment, which is worth
    // far more than 7 ms of readback.  The mechanism was not chased down; the
    // measurement was repeated and believed.
    //
    // WHAT IS KEPT is the destination: PINNED rather than std::vector, which
    // takes the readback off the driver's staged pageable path for 7-10 us per
    // layer in every regime above and leaves §15h alone.
    q.memcpy(h_ridx_, ws_ridx_, uint64_t(T) * K * 4);
    q.memcpy(h_rw_, ws_rw_, uint64_t(T) * K * 4);
    drain(waits_.router, waits_.n_router);

    // The measurement, taken where the router's decision first exists on the
    // host and BEFORE any residency question is asked of it — so the counts
    // describe what the model routed to, never what this configuration happened
    // to be able to serve.  A no-op unless profiling was turned on.
    profile_.record(L, h_ridx_, T * K, /*decode=*/T == 1);

    // ...and the ORDERED form of the same decision, for the policy question the
    // histogram cannot answer.  Same site, same reason: this is what the router
    // chose, before residency has been consulted about it.
    if (!trace_path_.empty()) {
        const size_t rec = kDs4TraceFields + K;
        for (uint32_t t = 0; t < T; ++t) {
            if (trace_buf_.size() + rec > kDs4TraceMaxInts) { ++trace_dropped_; continue; }
            trace_buf_.push_back(int32_t(trace_seq_));
            trace_buf_.push_back(t < trace_ids_.size() ? trace_ids_[t] : -1);
            trace_buf_.push_back(int32_t(T));
            trace_buf_.push_back(int32_t(L));
            for (uint32_t k = 0; k < K; ++k) trace_buf_.push_back(h_ridx_[t * K + k]);
        }
    }

    if (L >= pinned_layers_)
        return "layer " + std::to_string(L) + ": routed experts were never pinned (pin_layers=" +
               std::to_string(pinned_layers_) + ") — refusing to fabricate an expert output";

    // ---- routed experts, over the streamed slot cache ----
    //
    // =====================================================================
    // THE PREFILL ROOFLINE.  Read this before optimising prefill.
    // =====================================================================
    //
    // At a 512-token chunk the routed set is THE WHOLE POOL, and that single
    // fact fixes the ceiling.  With expert_used_count = 6 of expert_count = 256
    // (deepseek4.expert_used_count, docs/deepseek4/14_moe.md §352):
    //
    //   P(one expert is never routed to by any of 512 tokens)
    //       = (1 - 6/256)^512 = e^-12.16 = 5.2e-6
    //   expected experts missed per layer = 256 * 5.2e-6 = 0.0013
    //
    // So a pp512 chunk touches all 256 experts of all 43 layers.  Two things
    // follow, and the second is the one that matters:
    //
    // 1. NO RESIDENCY ORDERING CAN HELP PREFILL.  If every expert is selected,
    //    the hit rate is exactly the resident FRACTION no matter which experts
    //    are resident.  A prefill-ranked priority file is therefore not an
    //    untried lever, it is an impossible one.  Measured independently and
    //    recorded in docs/deepseek4/50: "demand hit rate equals static residency
    //    to 1e-9".  Prefill hit rate 0.30-0.35 is not a tuning failure; it IS
    //    the arena fraction.
    //
    // 2. THE WALL IS ARENA-SIZE-BOUND, AND 350 tok/s IS NOT REACHABLE.
    //    Under hidden-dim expert-TP each card holds half of every expert, so
    //    with a P GB pool each card must stream P/2 * (1 - f) per chunk, where
    //    f = arena_per_card / (P/2) and arena = VRAM - resident - 2 GB.
    //
    //      measured inputs: P = 142-147 GB (UD-Q8_K_XL expert tensors)
    //                       VRAM 32.5 GB/card usable (a run died at 32.6)
    //                       H2D 24.3 GB/s/card = 92% of this box's ceiling,
    //                          50.15 GB/s aggregate = 97.7% — already saturated
    //                       resident dense 9.375 GB (fp16) / 5.638 GB (Q8)
    //
    //      Q8 dense   arena 24.86 GB  f 0.338-0.351  ->  1.89-2.00 s  256-270 tok/s
    //      fp16 dense arena 21.13 GB  f 0.287-0.298  ->  2.04-2.16 s  237-251 tok/s
    //      zero resident weights, zero margin (physically impossible):
    //                 arena 32.50 GB  f 0.442-0.458  ->  1.62-1.69 s  303-316 tok/s
    //
    //    350 tok/s means a 1.463 s wall, which needs f >= 0.50-0.52, i.e. an
    //    expert arena of 35.4-38.0 GB on a 32.5 GB card.  It is not close, and
    //    the conclusion does not depend on which end of the 142-147 GB pool
    //    estimate is right.
    //
    //    The only levers that move that ceiling are: fewer expert bytes (the
    //    pool is already ~4.2 bits/parameter), a faster host link (at 92-97.7%
    //    of ceiling), more VRAM, or not needing every expert — which is a
    //    quality decision, not an engineering one.
    //
    // WHAT IS STILL WORTH TAKING.  pp512 at 115.88 tok/s is a 4.418 s wall
    // against a 2.04-2.16 s DMA floor, so roughly HALF of prefill is currently
    // exposed non-DMA time.  Closing that is worth up to ~237 tok/s and is where
    // the remaining prefill work actually is.  Making expert DMA faster is not.
    // Loop order.  DEFAULT is expert-major: the chunk's (token, slot) pairs are
    // counting-sorted by expert and each expert runs ONE batched GEMM over its
    // rows.  $IE_DS4_MOE_TOKEN_MAJOR restores the original decode-shaped loop —
    // it is the bit-exact reference the batched path is gated against
    // (deepseek4_experts_test §7), not a fallback that hides a failure.
    static const bool moe_token_major = std::getenv("IE_DS4_MOE_TOKEN_MAJOR") != nullptr;
    // `Ds4Options::moe_token_chunk` (or $IE_DS4_MOE_TOKEN_CHUNK, folded into it
    // at load) restores the TOKEN-CHUNKED residency loop — the predecessor of
    // `moe_expert_grouped`, in which the streaming partition bounded the BATCH
    // rather than the expert group.  It is kept for the same reason
    // $IE_DS4_MOE_TOKEN_MAJOR is: it is the reference the grouped path is gated
    // bit-for-bit against (deepseek4_residency_test §15j), not a fallback that
    // hides a failure.  Token-major implies it, because the token-major loop IS
    // the per-chunk one.
    const bool moe_token_chunk = moe_token_major || opt_.moe_token_chunk;
    const float limit = L < cfg_.swiglu_clamp_exp.size() ? cfg_.swiglu_clamp_exp[L] : 10.f;
    // Not waited for: `ws_moe_` is next touched by the expert accumulation on
    // this same in-order queue.  The chunk building between here and there is
    // pure host arithmetic over `h_ridx_`, and `acquire`'s H2D goes to the slot
    // arena on the transfer queue, never to `ws_moe_`.
    q.memset(ws_moe_, 0, uint64_t(T) * H * sizeof(float));
    const uint32_t stream = cache_.stream_slots();
    // EXPERT-GROUPED (default).  The batch is bounded ONLY by the packed
    // workspace; residency is satisfied by grouping the EXPERTS, so the streaming
    // partition no longer decides how many tokens may be batched together.
    if (!moe_token_chunk) {
        for (uint32_t b0 = 0; b0 < T; b0 += bws_.max_tokens) {
            const uint32_t nb = std::min<uint32_t>(bws_.max_tokens, T - b0);
            if (std::string e = moe_expert_grouped(L, b0, nb, rt.slot.gate.N, limit);
                !e.empty())
                return e;
        }
    } else
    // TOKEN-CHUNKED (opt-in, $IE_DS4_MOE_TOKEN_CHUNK / $IE_DS4_MOE_TOKEN_MAJOR).
    for (uint32_t t0 = 0; t0 < T; ) {
        // Chunk tokens so the union of their STREAMED experts fits the layer's
        // streaming partition; between chunks the compute queue is drained,
        // which is what makes an eviction safe (no slot is overwritten while a
        // GEMV still reads it).  Statically resident experts are never evicted,
        // so they cost no slot and must not be counted against this bound.
        uint32_t t1 = t0, n_stream = 0;
        std::vector<int32_t> uniq;
        while (t1 < T) {
            std::vector<int32_t> add;
            uint32_t add_stream = 0;
            for (uint32_t k = 0; k < K; ++k) {
                const int32_t e = h_ridx_[uint64_t(t1) * K + k];
                if (std::find(uniq.begin(), uniq.end(), e) == uniq.end() &&
                    std::find(add.begin(), add.end(), e) == add.end()) {
                    add.push_back(e);
                    if (!cache_.is_static(L, uint32_t(e))) ++add_stream;
                }
            }
            if (n_stream + add_stream > stream && t1 > t0) break;
            uniq.insert(uniq.end(), add.begin(), add.end());
            n_stream += add_stream;
            ++t1;
            // Second bound: the expert-major packed workspace holds at most
            // kDs4BatchTokenCap tokens (bws_ was sized for exactly that).  The
            // token-major path does not need it, but splitting identically in
            // both modes keeps the two paths comparable chunk for chunk.
            if (t1 - t0 >= bws_.max_tokens) break;
        }
        if (n_stream > stream)
            return "layer " + std::to_string(L) + ": one token needs " + std::to_string(n_stream) +
                   " streamed experts but the cache holds " + std::to_string(stream) +
                   " streaming slots/layer (of " + std::to_string(cache_.slots_per_layer()) + ")";
        std::vector<uint32_t> uslot(uniq.size());
        const sycl::event fetched = cache_.acquire(L, uniq.data(), uint32_t(uniq.size()), uslot.data());
        // An expert with neither a VRAM home nor a pinned host copy cannot be
        // computed.  There is no substitute for it that is not a fabricated
        // output, so this is a hard error rather than a skipped term.
        for (size_t i = 0; i < uslot.size(); ++i)
            if (uslot[i] == kDs4NoSlot)
                return "layer " + std::to_string(L) + ": expert " + std::to_string(uniq[i]) +
                       " is neither VRAM-resident (" + std::to_string(cache_.static_slots()) +
                       " static slots, filled in residency-priority order) nor host-pinned (" +
                       std::to_string(arena_.n_pinned_experts()) + " experts pinned, lowest id " +
                       std::to_string(arena_.first_pinned_expert()) +
                       ") — refusing to fabricate an expert output";
        // The compute queue does not start reading a slot before its H2D lands.
        // This single dependency IS the DMA/compute overlap mechanism.
        q.submit([&](sycl::handler& h) { h.depends_on(fetched); h.single_task([](){}); });

        // The intermediate width is this CARD's slice, not the model's: gate/up
        // produce `gate.N` values and down contracts over `down.K`.  Equal to EF
        // when n_cards == 1.
        const uint32_t EFc = rt.slot.gate.N;
        // A slot pointer per routed expert id — the only thing that differs
        // between this caller and the test's contiguous bank.
        auto resolve = [&](uint32_t e, DS4ExpertBank& gb, DS4ExpertBank& ub,
                           DS4ExpertBank& db) {
            const auto it = std::find(uniq.begin(), uniq.end(), int32_t(e));
            if (it == uniq.end()) return false;
            void* base = cache_.slot_ptr(L, uslot[size_t(it - uniq.begin())]);
            gb = ds4_slot_bank(rt.slot.gate, base);
            ub = ds4_slot_bank(rt.slot.up,   base);
            db = ds4_slot_bank(rt.slot.down, base);
            return true;
        };

        if (!moe_token_major) {
            // EXPERT-MAJOR (default).  One batched GEMM per distinct expert over
            // all of the chunk's rows routed to it: O(distinct experts) kernels
            // instead of 9 per (token, slot), and each expert's weight slice is
            // read ONCE per chunk instead of once per routed token.
            if (std::string e = ds4_experts_forward_batched(
                    q, resolve, ws_norm_ + uint64_t(t0) * H, h_ridx_ + uint64_t(t0) * K,
                    h_rw_ + uint64_t(t0) * K, ws_moe_ + uint64_t(t0) * H,
                    t1 - t0, H, EFc, K, cfg_.n_experts, limit, bws_);
                !e.empty())
                return "layer " + std::to_string(L) + ": " + e;
        } else {
            // TOKEN-MAJOR (opt-in, $IE_DS4_MOE_TOKEN_MAJOR).  The decode-shaped
            // loop the expert-major path replaced.  Kept because it is the
            // reference the batched path is gated bit-for-bit against, and
            // because it is the fallback if a batched GEMM ever misbehaves.
            for (uint32_t t = t0; t < t1; ++t) {
                cast_fp32_to_fp16(q, ws_norm_ + uint64_t(t) * H, xws_.xh, H);
                quantize_q8_1(q, xws_.xh, xws_.x_q8, H);
                for (uint32_t k = 0; k < K; ++k) {
                    const int32_t e = h_ridx_[uint64_t(t) * K + k];
                    DS4ExpertBank gb, ub, db;
                    if (!resolve(uint32_t(e), gb, ub, db))
                        return "layer " + std::to_string(L) + ": expert " + std::to_string(e) +
                               " was not acquired for this chunk";
                    ds4_expert_gemv(q, gb, 0, xws_.x_q8, nullptr, xws_.gate_h);
                    ds4_expert_gemv(q, ub, 0, xws_.x_q8, nullptr, xws_.up_h);
                    cast_fp16_to_fp32(q, xws_.gate_h, xws_.gate_f, EFc);
                    cast_fp16_to_fp32(q, xws_.up_h,   xws_.up_f,   EFc);
                    ds4_swiglu_clamped(q, xws_.gate_f, xws_.up_f, xws_.h_f, EFc, limit);
                    cast_fp32_to_fp16(q, xws_.h_f, xws_.h_h, EFc);
                    quantize_q8_1(q, xws_.h_h, xws_.h_q8, EFc);
                    ds4_expert_gemv(q, db, 0, xws_.h_q8, nullptr, xws_.y_h);
                    accum_f16(q, xws_.y_h, h_rw_[uint64_t(t) * K + k],
                              ws_moe_ + uint64_t(t) * H, H);
                }
            }
        }
        // KEPT, AND IT IS THE EVICTION-SAFETY BARRIER.  `acquire` above writes
        // an evicted slot through the TRANSFER queue, which this in-order
        // compute queue does not order against; the only thing standing between
        // a slot being overwritten and a GEMV still reading it is this drain
        // between chunks.  Removing it corrupts an expert's weights silently.
        drain(waits_.moe, waits_.n_moe);
        // Idle-time-only speculation for the NEXT layer, using this token's ids
        // as the predictor (Rank 3).  It is declined outright unless the transfer
        // queue is empty, so it can never delay a demand fetch.
        if (spec_this_pass_ && L + 1 < pinned_layers_ && stream)
            cache_.speculate(L + 1, h_ridx_, std::min<uint32_t>(K, stream));
        t0 = t1;
    }
    // ---- shared expert (DeepseekV4MLP) — HERE ONLY WHEN IT IS SLICED ----
    //
    // WHY THE BRANCH IS LOAD-BEARING AND NOT A STYLE CHOICE.  Under the split
    // this card's `ffn_down_shexp` contracts over its own intermediate columns,
    // so its output is a PARTIAL of exactly the shape `ws_moe_` already is.
    // Computing it here puts it in `ds4_shared_out(T)`, the buffer immediately
    // after `ws_moe_`, so the single reduction the orchestrator already issues
    // covers both — the shared expert costs ZERO extra round trips.
    //
    // MIRRORED, its output is the WHOLE shared expert on EVERY card, and
    // reducing that would sum it n_cards times — a plausible-looking wrong
    // answer, not a rounding.  So it is computed AFTER the reduction instead, in
    // `layer_forward_post`, exactly where it always was.  n_cards == 1 takes that
    // same branch, which is what keeps the single-card path unchanged to the bit.
    if (aslice_.split) {
        const float shlimit =
            L < cfg_.swiglu_clamp_shexp.size() ? cfg_.swiglu_clamp_shexp[L] : 10.f;
        // The three dense_w below are the SHARED EXPERT, not attention.  Profiler
        // bookkeeping only; see Ds4SiteScope.
        const Ds4SiteScope shsite(Ds4Site::kShExp);
        {   // ONE launch: sh_gate and sh_up read the same activation with the
            // same K and write disjoint outputs.  `sh_down` reads ws_shg_, a
            // different activation, so it stays a separate call below.
            const Ds4Dense* mds[2] = {&rt.sh_gate, &rt.sh_up};
            float*          mys[2] = {ws_shg_, ws_shu_};
            if (!dense_multi_w(q, ws_norm_, mds, mys, 2, T)) {
                dense_w(q, ws_norm_, rt.sh_gate, ws_shg_, T);
                dense_w(q, ws_norm_, rt.sh_up,   ws_shu_, T);
            }
        }
        ds4_swiglu_clamped(q, ws_shg_, ws_shu_, ws_shg_, uint64_t(T) * aslice_.efc, shlimit);
        dense_w(q, ws_shg_, rt.sh_down, ds4_shared_out(T), T);
    }

    // `ws_moe_` now holds this CARD's routed-expert sum and NOTHING else — that
    // is `moe_accumulator()`'s contract and it holds in every configuration.
    // Under expert-TP it is a PARTIAL (see the class comment on
    // DeepSeek4TpRuntime), and so is `ds4_shared_out(T)` beside it when the
    // shared expert is sliced; the caller reduces both before
    // layer_forward_post runs.
    return {};
}

std::string DeepSeek4Runtime::layer_forward_post(uint32_t L, uint32_t T) {
    sycl::queue& q  = *q_;
    Ds4LayerRT&  rt = rt_[L];
    const uint32_t H = cfg_.hidden, hc = cfg_.hc_count, EF = cfg_.expert_ffn;

    // ---- shared expert (DeepseekV4MLP) + routed sum ----
    // Computed HERE when it is MIRRORED — every single-card run, and every
    // two-card run with `split_non_expert` off.  Every card computes all of it,
    // so it must land AFTER the cross-card reduction or it would be counted
    // n_cards times.  When it is SLICED, `layer_forward_mid` computed its partial
    // before the seam and the reduction has already made it whole, so only the
    // add below is left.
    //
    // THE ADD AND THE MIX ARE THEREFORE IDENTICAL IN BOTH CONFIGURATIONS — same
    // operands, same order, same destination.  That is deliberate: it is what
    // keeps the mirrored two-card path bit-identical to the single-card path
    // here, and it is why the branch covers only WHERE the shared expert is
    // computed and never WHAT is done with it.
    if (!aslice_.split) {
        const float shlimit =
            L < cfg_.swiglu_clamp_shexp.size() ? cfg_.swiglu_clamp_shexp[L] : 10.f;
        // The three dense_w below are the SHARED EXPERT, not attention.  Profiler
        // bookkeeping only; see Ds4SiteScope.
        const Ds4SiteScope site(Ds4Site::kShExp);
        {   // ONE launch: sh_gate and sh_up read the same activation with the
            // same K and write disjoint outputs.  `sh_down` reads ws_shg_, a
            // different activation, so it stays a separate call below.
            const Ds4Dense* mds[2] = {&rt.sh_gate, &rt.sh_up};
            float*          mys[2] = {ws_shg_, ws_shu_};
            if (!dense_multi_w(q, ws_norm_, mds, mys, 2, T)) {
                dense_w(q, ws_norm_, rt.sh_gate, ws_shg_, T);
                dense_w(q, ws_norm_, rt.sh_up,   ws_shu_, T);
            }
        }
        ds4_swiglu_clamped(q, ws_shg_, ws_shu_, ws_shg_, uint64_t(T) * EF, shlimit);
        dense_w(q, ws_shg_, rt.sh_down, ds4_shared_out(T), T);
    }
    add_f32(q, ws_moe_, ds4_shared_out(T), ws_sub_, uint64_t(T) * H);

    ds4_hc_mix(q, ws_streams2_, ws_post_, ws_comb_, ws_sub_, ws_streams_, T, H, hc);
    // KEPT — THE ONE PER-LAYER DRAIN, and every removed wait above depends on
    // it.  It is what lets the next layer's `Ds4LayerCache::grow` free a device
    // buffer without asking whether a kernel is still reading it, and it is the
    // point at which this card's `h_spos_` staging becomes rewritable.  Under
    // the sequential drive it was also the barrier that idled the other GPU;
    // now the other card is running its own layer across it, so it costs a host
    // round trip on this thread and nothing on the critical path of the pair.
    drain(waits_.post, waits_.n_post);
    return {};
}

std::string DeepSeek4Runtime::layer_forward(uint32_t L, uint32_t T, uint32_t pos0,
                                            const int32_t* input_ids) {
    (void)input_ids;
    if (std::string e = layer_forward_pre(L, T, pos0); !e.empty()) return e;
    if (std::string e = layer_forward_mid(L, T);       !e.empty()) return e;
    return layer_forward_post(L, T);
}

std::string DeepSeek4Runtime::forward_prologue(const int32_t* input_ids, uint32_t T,
                                               uint32_t pos0) {
    sycl::queue& q = *q_;
    const uint32_t H = cfg_.hidden, hc = cfg_.hc_count;
    // REFUSED, HERE, BEFORE A SINGLE EVENT IS COLLECTED.  `ie::g_profiler` is
    // set by a caller that intends to read command_start/command_end off every
    // event this pass submits, and a queue built without enable_profiling does
    // not carry them — `KernelProfiler::harvest()` would throw out of a call
    // that has no catch above it and take the process down.  Since profiling
    // became opt-in ($IE_QUEUE_PROFILING, see load) that is a reachable
    // combination, so it is answered with a named error, which every caller
    // already handles: ie-ds4-bench reports "profiled step failed: ..." and
    // carries on.  A crash would also have been loud; this is loud AND useful.
    if (g_profiler && !q_profiling_)
        return "DeepSeek4Runtime::forward: the kernel profiler is on but this card's queue was"
               " created WITHOUT enable_profiling, so its events carry no timestamps and"
               " harvesting them is undefined.  Queue profiling is opt-in because it costs"
               " ~0.42 us of host time on EVERY kernel launch (1.43 -> 1.85 us/submit measured)"
               " and decode submits thousands per token.  Re-run with IE_QUEUE_PROFILING=1.";
    cache_.reset_stats();
    // The host-wait breakdown describes ONE forward, so it is cleared where the
    // forward begins — here, on every path, single-card and TP alike, because
    // this is the one function both drives call first.
    waits_ = HostWaits{};

    // The routing trace needs the token ids, and this is the only place they
    // exist on the host.  Copied rather than aliased: `input_ids` belongs to the
    // caller and layer_forward_mid runs long after this returns.  A no-op unless
    // $DS4_EXPERT_TRACE asked for a trace.
    if (!trace_path_.empty()) {
        trace_ids_.assign(input_ids, input_ids + T);
        ++trace_seq_;
    }

    q.memcpy(ws_ids_, input_ids, uint64_t(T) * 4);
    {
        std::vector<int32_t> p(T);
        for (uint32_t t = 0; t < T; ++t) p[t] = int32_t(pos0 + t);
        q.memcpy(ws_pos_, p.data(), p.size() * 4);
        drain(waits_.other, waits_.n_other);
    }
    ds4_rope_cos_sin(q, inv_main_dev_, ws_pos_, ws_cos_, ws_sin_, T, cfg_.rope_dim / 2, 1.f);
    ds4_rope_cos_sin(q, inv_comp_dev_, ws_pos_, ws_ccos_, ws_csin_, T, cfg_.rope_dim / 2, 1.f);

    gather_embd_w(q, ws_ids_, embd_, ws_coll_, T, H);
    // Vision-Exp: image rows overwrite the gathered embeddings BEFORE the
    // hyper-connection streams are expanded (reference merge_image_embeddings
    // writes into h right after the embedding lookup).
    vis_chunk_ = false;
    if (!vis_spans_.empty())
        if (std::string e = vision_prologue(T, pos0); !e.empty()) return e;
    expand_streams(q, ws_coll_, ws_streams_, T, H, hc);
    drain(waits_.other, waits_.n_other);
    return {};
}

std::string DeepSeek4Runtime::set_vision(const float* rows, uint32_t t0, uint32_t n, uint32_t start_off) {
    if (!rows || n == 0) return "DeepSeek4Runtime::set_vision: empty block";
    if (start_off >= n) return "DeepSeek4Runtime::set_vision: start_off past the block";
    if (n > opt_.max_seq)
        return "DeepSeek4Runtime::set_vision: image block of " + std::to_string(n) +
               " rows exceeds max_seq " + std::to_string(opt_.max_seq) +
               " (one forward() must hold the whole block; raise --prefill-chunk)";
    VisSpan s;
    s.t0 = t0; s.n = n; s.start_off = start_off;
    s.rows.assign(rows, rows + size_t(n) * cfg_.hidden);
    vis_spans_.push_back(std::move(s));
    return {};
}

// Splice the staged rows of every block inside [pos0, pos0+T) into ws_coll_
// and build the per-token image mask / window-widening arrays for this chunk.
// Reference: model.py forward() + get_image_visible(): left = distance from
// IMAGE_START (clamped to max-1), right = distance to IMAGE_END (clamped to
// max), valid from IMAGE_START to IMAGE_END inclusive; image_mask covers every
// pseudo-id row, compress pads included.
std::string DeepSeek4Runtime::vision_prologue(uint32_t T, uint32_t pos0) {
    constexpr int64_t kMaxImg = 384;   // vision_max_n_token (Vision-Exp config)
    sycl::queue& q = *q_;
    const uint32_t H = cfg_.hidden;
    const int64_t win = int64_t(cfg_.sliding_window);
    const uint64_t c0 = pos0, c1 = uint64_t(pos0) + T;
    if (!ws_imgmask_ || !h_vis_rows_) return "deepseek4: vision workspace not allocated";
    bool any = false;
    int32_t* mask  = h_vis_.data();
    int32_t* left  = mask + T;
    int32_t* right = left + T;
    for (const VisSpan& s : vis_spans_) {
        const uint64_t s0 = s.t0, s1 = uint64_t(s.t0) + s.n;
        if (s1 <= c0 || s0 >= c1) continue;
        if (s0 < c0 || s1 > c1)
            return "deepseek4: image block [" + std::to_string(s0) + ", " + std::to_string(s1) +
                   ") straddles the forward chunk [" + std::to_string(c0) + ", " + std::to_string(c1) +
                   ") — an image block must be prefilled in one forward() call";
        if (!any) { std::fill_n(mask, size_t(3) * T, 0); any = true; }
        std::memcpy(h_vis_rows_ + (s0 - c0) * H, s.rows.data(), size_t(s.n) * H * sizeof(float));
        q.memcpy(ws_coll_ + (s0 - c0) * H, h_vis_rows_ + (s0 - c0) * H, size_t(s.n) * H * sizeof(float));
        for (uint64_t t = s0; t < s1; ++t) mask[t - c0] = 1;
        const uint64_t st = s0 + s.start_off, en = s1 - 1;
        for (uint64_t t = st; t <= en; ++t) {
            const int64_t l = std::min<int64_t>(int64_t(t - st), kMaxImg - 1);
            const int64_t r = std::min<int64_t>(int64_t(en - t), kMaxImg);
            left[t - c0]  = int32_t(std::max<int64_t>(l - (win - 1), 0));
            right[t - c0] = int32_t(r);
        }
    }
    if (!any) return {};
    q.memcpy(ws_imgmask_, mask,  size_t(T) * 4);
    q.memcpy(ws_vleft_,   left,  size_t(T) * 4);
    q.memcpy(ws_vright_,  right, size_t(T) * 4);
    vis_chunk_ = true;
    return {};
}

std::string DeepSeek4Runtime::forward_epilogue(uint32_t T, float* logits, bool last_only) {
    sycl::queue& q = *q_;
    const uint32_t H = cfg_.hidden, hc = cfg_.hc_count;
    ds4_hyper_head(q, ws_streams_, hh_fn_, hh_base_, hh_scale_, ws_coll_, T, H, hc,
                   cfg_.rms_eps, cfg_.hc_eps);
    ds4_rms_norm(q, ws_coll_, onorm_, ws_norm_, T, H, cfg_.rms_eps);
    const uint32_t TL = last_only ? 1u : T;
    // `ws_logits_` holds one row unless a caller has asked for more.  Growing it
    // here — rather than reserving max_seq rows at load — is what keeps the
    // buffer proportional to what is USED.  The drain before the free is
    // load-bearing: the previous call's D2H out of this buffer may still be in
    // flight, and `sycl::free` is a host call that does not wait for it.
    if (TL > ws_logits_T_) {
        q.wait();
        if (ws_logits_) sycl::free(ws_logits_, q);
        ws_logits_T_ = 0;
        ws_logits_   = sycl::malloc_device<float>(uint64_t(TL) * cfg_.vocab, q);
        if (!ws_logits_)
            return "DeepSeek4Runtime::forward_epilogue: could not allocate " +
                   std::to_string(uint64_t(TL) * cfg_.vocab * 4) + " B for [" +
                   std::to_string(TL) + ", " + std::to_string(cfg_.vocab) +
                   "] logits (last_only=false over T=" + std::to_string(T) + ")";
        ws_logits_T_ = TL;
    }
    // The lm_head is [vocab, hidden] — 163k rows on the real model, so it is a
    // decode cost in its own right and must not be filed under attention.
    // Profiler bookkeeping only; see Ds4SiteScope.
    const Ds4SiteScope site(Ds4Site::kLmHead);
    dense_w(q, ws_norm_ + uint64_t(last_only ? T - 1 : 0) * H, lm_, ws_logits_, TL);
    q.memcpy(logits, ws_logits_, uint64_t(TL) * cfg_.vocab * 4);
    drain(waits_.other, waits_.n_other);
    cache_.collect_dma_time();
    return {};
}

std::string DeepSeek4Runtime::forward(const int32_t* input_ids, uint32_t T, uint32_t pos0,
                                      float* logits, bool last_only) {
    if (!q_) return "DeepSeek4Runtime::forward: not loaded";
    // Hidden-dim expert-TP: this card's `down` GEMV contracts over only its
    // slice of the intermediate dimension, so ws_moe_ holds a PARTIAL sum that
    // is meaningless until the other cards' partials are added to it
    // (ds4_tp_reduce_host).  Nothing here drives the other cards, so the only
    // honest thing to return is an error.  Emitting the half-sum would be a
    // plausible-looking wrong answer, which is worse than a refusal.
    if (slice_.n_cards > 1)
        return "DeepSeek4Runtime::forward: this runtime holds intermediate columns [" +
               std::to_string(slice_.ef0) + "," + std::to_string(slice_.ef0 + slice_.efc) +
               ") of every expert (card " + std::to_string(slice_.card) + " of " +
               std::to_string(slice_.n_cards) +
               "); its routed-expert output is a PARTIAL sum that must be reduced across cards"
               " through host memory (ds4_tp_reduce_host — P2P is unavailable between these"
               " devices).  A single runtime cannot reach the other cards, so the only honest"
               " answer is a refusal — drive a sliced load through DeepSeek4TpRuntime, which"
               " performs that reduction.  Refusing to return a half-summed MoE output.";
    if (T == 0 || T > ws_T_)
        return "DeepSeek4Runtime::forward: T=" + std::to_string(T) + " outside [1," +
               std::to_string(ws_T_) + "]";
    const auto wall0 = std::chrono::steady_clock::now();

    // First-chunk-only speculation — FALSIFIED 2026-08-09 and therefore
    // default OFF, kept as an opt-in instrument (IE_DS4_PREFETCH_FIRST=1).
    // The 2026-08-03 basis (single cold chunk 72.15 -> 89.22 with prefetch)
    // does not reproduce on today's engine: A/B same binary, priority-ranked
    // residency, pp512 109.62 ON vs 109.35 OFF, pp2048 102.85 vs 102.57 —
    // FLAT.  The idle transfer-queue windows that speculation exploited at
    // pp512~72 no longer exist at pp512~110.  `--prefetch` still forces
    // speculation everywhere, unchanged.
    static const bool spec_first = [] {
        const char* e = std::getenv("IE_DS4_PREFETCH_FIRST");
        return e && *e && std::string(e) != "0";
    }();
    spec_this_pass_ = opt_.prefetch || (spec_first && T > 1 && pos0 == 0);

    if (std::string e = forward_prologue(input_ids, T, pos0); !e.empty()) return e;
    for (uint32_t L = 0; L < cfg_.n_layers; ++L)
        if (std::string e = layer_forward(L, T, pos0, input_ids); !e.empty()) return e;
    if (std::string e = forward_epilogue(T, logits, last_only); !e.empty()) return e;

    const auto& s = cache_.stats();
    timing_.wall_seconds  = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();
    timing_.dma_seconds   = s.dma_seconds;
    timing_.bytes_fetched = s.bytes_fetched;
    timing_.hits          = s.hits;
    timing_.misses        = s.misses;
    accumulate_cache(T);
    // Same $IE_DS4_TP_TRACE line as the two-card drive, for the one-card one:
    // there is no cross-card term to report, so it prints the part that exists.
    // A single-card decode that is submission-bound looks identical to a
    // two-card one that is, and the fix is the same, so the measurement must be
    // available on both paths.
    if (std::getenv("IE_DS4_TP_TRACE"))
        std::fprintf(stderr,
                     "[ds4-1c] T=%u wall=%.3f | wait=%.3f"
                     "(rt=%.3f/%llu moe=%.3f/%llu post=%.3f/%llu oth=%.3f/%llu)"
                     " submit+host=%.3f\n",
                     T, timing_.wall_seconds * 1e3, waits_.total() * 1e3,
                     waits_.router * 1e3, (unsigned long long)waits_.n_router,
                     waits_.moe * 1e3, (unsigned long long)waits_.n_moe,
                     waits_.post * 1e3, (unsigned long long)waits_.n_post,
                     waits_.other * 1e3, (unsigned long long)waits_.n_other,
                     (timing_.wall_seconds - waits_.total()) * 1e3);
    return {};
}

// ===========================================================================
// DeepSeek4TpRuntime — the two-card orchestrator
// ===========================================================================

// One card's host thread.  It owns nothing but its own job slot: the runtime it
// drives, the queue it submits to, the workspaces it writes and the caches it
// mutates all belong to that card's DeepSeek4Runtime, and no other thread
// touches them between the post and the join.  The mutex therefore guards only
// the handshake, never the model state — the happens-before it establishes is
// what publishes the card's writes to the thread that reads them next.
struct DeepSeek4TpRuntime::Worker {
    std::thread                  th;
    std::mutex                   m;
    std::condition_variable      cv;
    std::function<std::string()> job;
    std::chrono::steady_clock::time_point base{};
    // THE HANDSHAKE FLAGS ARE ATOMIC SO THEY CAN BE READ WITHOUT THE LOCK.
    // The mutex and condition variable are still the blocking path and still the
    // only thing that guards `job`, `base`, `err` and the timestamps; these three
    // exist so that the SPIN below can watch for a transition without taking a
    // lock the other thread may be holding.  Each is written with release and
    // read with acquire, so an observer that sees the flag also sees every write
    // the writer made before setting it — the same happens-before the mutex gave,
    // established a second way for the path that does not take the mutex.
    std::atomic<bool> has_job{false};
    std::atomic<bool> done{true};
    std::atomic<bool> quit{false};
    std::string err;
    double      begin_s = 0.0, end_s = 0.0;

    // HOW LONG TO SPIN BEFORE BLOCKING, AND WHY THIS NUMBER.
    //
    // Blocking on a condition variable costs a futex sleep and a futex wake.
    // MEASURED on this box with the exact handshake below: 1.7-2.7 us median
    // post-to-pickup when both threads stay hot, and 75-100 us when the worker's
    // core has had time to idle — which is what deepseek4_residency_test 15h
    // caught the moment the queue-profiling change made the segments short
    // enough for that latency to matter.  Spinning instead: 0.12 us median,
    // 0.15 us p95, in every regime measured.
    //
    // The window has to cover the longest gap the waiter should tolerate before
    // it is cheaper to sleep.  For the WORKER that gap is the time between
    // finishing one segment and being handed the next — the cross-card reduction
    // and the host bookkeeping around it, measured at 12-28 us per reduction.
    // For the DRIVER it is the imbalance between two cards running mirrored work
    // on identical devices, which the same trace puts at single-digit us.  60 us
    // covers both with room, and bounds the waste: a thread that spins the whole
    // window and then blocks has burned it once, and only when its sibling was
    // genuinely slower than any measured case.
    //
    // ONLY THE FIRST 5 us ARE A BUSY SPIN.  This box runs several agents at a
    // load average around ten on twenty cores, and a thread that holds a core for
    // 60 us while another process has work to do is taking more than it saves.
    // After `kPauseSeconds` the loop yields instead, which keeps the futex sleep
    // and its wake off the critical path while letting the scheduler run someone
    // else.  MEASURED post-to-pickup, same handshake, 4000 iterations each:
    //     condvar only    1.90-2.55 us median,  3.3-6.1 us p95
    //     pause only      0.13-0.17 us median,  0.18-0.21 us p95
    //     pause + yield   0.20-0.23 us median,  0.29-0.32 us p95
    // The hybrid gives up 0.06 us against the pure spin and keeps 1.7-2.3 us of
    // the 1.9-2.5 us the condvar was costing, so it is chosen on the strength of
    // what it does NOT take rather than on a measured latency difference.
    static constexpr double kSpinSeconds  = 60e-6;
    static constexpr double kPauseSeconds = 5e-6;

    // Waits for `p()` without sleeping, for at most the window.  Returns what it
    // last saw, so a false return means "give up and block", never "not ready".
    template <typename Pred>
    static bool spin_until(Pred p) {
        const auto t0 = std::chrono::steady_clock::now();
        for (;;) {
            if (p()) return true;
            const double el =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (el >= kSpinSeconds) return p();
            if (el >= kPauseSeconds) {
                std::this_thread::yield();
            } else {
#if defined(__x86_64__) || defined(__i386__)
                __builtin_ia32_pause();
#endif
            }
        }
    }

    void start() {
        th = std::thread([this] {
            for (;;) {
                // Watch for the next job without sleeping.  If it arrives inside
                // the window the futex round trip never happens; if it does not,
                // the wait below is exactly what it always was.
                spin_until([this] {
                    return has_job.load(std::memory_order_acquire) ||
                           quit.load(std::memory_order_acquire);
                });
                std::function<std::string()> j;
                std::chrono::steady_clock::time_point b;
                {
                    std::unique_lock<std::mutex> lk(m);
                    cv.wait(lk, [&] { return has_job.load(std::memory_order_acquire) ||
                                             quit.load(std::memory_order_acquire); });
                    if (quit.load(std::memory_order_acquire)) return;
                    j       = std::move(job);
                    b       = base;
                    has_job.store(false, std::memory_order_relaxed);
                }
                const auto  t0 = std::chrono::steady_clock::now();
                std::string r  = j();
                const auto  t1 = std::chrono::steady_clock::now();
                {
                    std::lock_guard<std::mutex> lk(m);
                    err     = std::move(r);
                    begin_s = std::chrono::duration<double>(t0 - b).count();
                    end_s   = std::chrono::duration<double>(t1 - b).count();
                    // RELEASE, and it is the last store in the critical section:
                    // a driver that observes it by spinning — without ever taking
                    // `m` — must still see `err` and the timestamps above it.
                    done.store(true, std::memory_order_release);
                }
                cv.notify_all();
            }
        });
    }
    void stop() noexcept {
        if (!th.joinable()) return;
        {
            std::lock_guard<std::mutex> lk(m);
            quit.store(true, std::memory_order_release);
        }
        cv.notify_all();
        th.join();
    }
    void post(std::function<std::string()> j,
              const std::chrono::steady_clock::time_point& b) {
        {
            std::lock_guard<std::mutex> lk(m);
            job  = std::move(j);
            base = b;
            err.clear();
            done.store(false, std::memory_order_relaxed);
            // RELEASE, last: a worker that picks this up by spinning has not
            // taken `m` yet, and must still see `job` and `base`.
            has_job.store(true, std::memory_order_release);
        }
        cv.notify_one();
    }
    // Blocks until the posted job has finished.  Called for EVERY posted job,
    // on every path including the failing ones, so a worker is never left
    // running while the driver unwinds.
    //
    // The spin is the fast path and the acquire-load is what publishes the job's
    // writes — `err`, the timestamps, and everything the card's runtime touched —
    // to this thread; the mutex path below publishes them the way it always did.
    // Both are real happens-before edges, so the caller's reads are safe on
    // whichever path it took.
    void join_job() {
        if (spin_until([this] { return done.load(std::memory_order_acquire); })) return;
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk, [&] { return done.load(std::memory_order_acquire); });
    }
};

DeepSeek4TpRuntime::DeepSeek4TpRuntime()  = default;
DeepSeek4TpRuntime::~DeepSeek4TpRuntime() { release(); }

std::string DeepSeek4TpRuntime::run_cards(uint32_t seg,
                                          const std::chrono::steady_clock::time_point& t0,
                                          const std::function<std::string(uint32_t)>& fn,
                                          uint32_t* out_card) {
    const uint32_t n = uint32_t(rts_.size());
    const auto secs  = [&](const std::chrono::steady_clock::time_point& t) {
        return std::chrono::duration<double>(t - t0).count();
    };
    // A card's work must not be able to throw out of a worker thread — there is
    // no catch above it there, so it would call std::terminate and take the
    // process down with the diagnostic still on the stack.  Card 0 runs on the
    // CALLING thread and would not have that problem, but it gets the same
    // treatment so that "which card ran where" stays an implementation detail
    // and cannot change what a failure looks like.
    const auto guarded = [&fn](uint32_t c) -> std::string {
        try {
            return fn(c);
        } catch (const sycl::exception& e) {
            return std::string("threw sycl::exception: ") + e.what();
        } catch (const std::exception& e) {
            return std::string("threw: ") + e.what();
        } catch (...) {
            return "threw a non-std exception";
        }
    };

    std::vector<std::string> errs(n);
    std::vector<double>      b(n, 0.0), e(n, 0.0);

    if (ftrace_.sequential) {
        // The profiler is on: its entry list is an unsynchronised std::vector,
        // so the cards take turns.  Nothing else about the drive changes.
        for (uint32_t c = 0; c < n; ++c) {
            b[c]    = secs(std::chrono::steady_clock::now());
            errs[c] = guarded(c);
            e[c]    = secs(std::chrono::steady_clock::now());
        }
    } else {
        for (uint32_t c = 1; c < n; ++c)
            workers_[c - 1]->post([&, c] { return guarded(c); }, t0);
        // Card 0 on this thread, WHILE the others run.
        b[0]    = secs(std::chrono::steady_clock::now());
        errs[0] = guarded(0);
        e[0]    = secs(std::chrono::steady_clock::now());
        // Join every worker before looking at any result — an early return on
        // card 0's error would otherwise leave card 1 writing into a workspace
        // the caller is about to tear down.
        for (uint32_t c = 1; c < n; ++c) {
            Worker& w = *workers_[c - 1];
            w.join_job();
            errs[c] = w.err;
            b[c]    = w.begin_s;
            e[c]    = w.end_s;
        }
    }

    if (seg != UINT32_MAX && ftrace_.begin.size() >= uint64_t(seg + 1) * n)
        for (uint32_t c = 0; c < n; ++c) {
            ftrace_.begin[uint64_t(seg) * n + c] = b[c];
            ftrace_.end[uint64_t(seg) * n + c]   = e[c];
        }

    for (uint32_t c = 0; c < n; ++c)
        if (!errs[c].empty()) { if (out_card) *out_card = c; return errs[c]; }
    return {};
}

std::string DeepSeek4TpRuntime::load(const DeepSeek4Model& m, const Ds4TpOptions& opt) {
    release();
    // A failure anywhere below leaves NOTHING half-built.  The cards are now
    // constructed up front and prepared in parallel, so a card that fails has
    // siblings holding device allocations and tens of GB of pinned host pages,
    // and the caller's only signal is the returned string.
    struct Cleanup {
        DeepSeek4TpRuntime* self;
        bool                armed = true;
        ~Cleanup() { if (armed) self->release(); }
    } cleanup{this};
    if (opt.n_cards == 0) return "DeepSeek4TpRuntime::load: n_cards == 0";
    // The three per-card fields belong to this class.  Overwriting a value the
    // caller set would be a silent reinterpretation of their configuration, and
    // the failure mode it hides — two runtimes both loading card 0's slice —
    // produces plausible-looking logits that are simply wrong.
    if (opt.base.n_cards != 1 || opt.base.card != 0)
        return "DeepSeek4TpRuntime::load: Ds4TpOptions::base.n_cards/base.card are set PER CARD by"
               " the orchestrator and must be left at 1/0 (got n_cards=" +
               std::to_string(opt.base.n_cards) + ", card=" + std::to_string(opt.base.card) +
               "); use Ds4TpOptions::n_cards instead";
    if (opt.base.device_ordinal != 0)
        return "DeepSeek4TpRuntime::load: Ds4TpOptions::base.device_ordinal is set PER CARD by the"
               " orchestrator and must be left at 0 (got " +
               std::to_string(opt.base.device_ordinal) +
               "); name the devices in Ds4TpOptions::device_ordinals";

    std::vector<sycl::device> gpus;
    for (const auto& d : sycl::device::get_devices())
        if (d.is_gpu() && d.get_backend() == sycl::backend::ext_oneapi_level_zero) gpus.push_back(d);
    if (gpus.empty()) return "DeepSeek4TpRuntime::load: no Level Zero GPU";

    ordinals_ = opt.device_ordinals;
    if (ordinals_.empty())
        for (uint32_t c = 0; c < opt.n_cards; ++c) ordinals_.push_back(c);
    if (ordinals_.size() != opt.n_cards) {
        const std::string e = "DeepSeek4TpRuntime::load: " + std::to_string(ordinals_.size()) +
                              " device ordinals for " + std::to_string(opt.n_cards) + " cards";
        ordinals_.clear();
        return e;
    }
    for (uint32_t c = 0; c < opt.n_cards; ++c)
        if (ordinals_[c] >= gpus.size()) {
            const std::string e = "DeepSeek4TpRuntime::load: card " + std::to_string(c) +
                                  " asks for GPU " + std::to_string(ordinals_[c]) +
                                  " but only " + std::to_string(gpus.size()) + " are present";
            ordinals_.clear();
            return e;
        }
    for (uint32_t c = 0; c < opt.n_cards; ++c)
        for (uint32_t d = c + 1; d < opt.n_cards; ++d)
            if (ordinals_[c] == ordinals_[d] && !opt.same_device_rehearsal) {
                const std::string e =
                    "DeepSeek4TpRuntime::load: cards " + std::to_string(c) + " and " +
                    std::to_string(d) + " are both bound to GPU " + std::to_string(ordinals_[c]) +
                    ".  Each card sizes its expert arena from that device's whole global_mem_size,"
                    " so two on one device would plan a residency neither can honour.  Set"
                    " Ds4TpOptions::same_device_rehearsal explicitly if that is what you want;"
                    " numbers measured under it are not two-card numbers.";
                ordinals_.clear();
                return e;
            }
    if (opt.same_device_rehearsal)
        std::fprintf(stderr,
                     "[ds4-tp] SAME-DEVICE REHEARSAL: card->GPU map may repeat a device.  The"
                     " arithmetic and the staging are real; the residency plan is NOT.\n");

    // ---- PHASE A: every card prepares itself, CONCURRENTLY ----
    //
    // `load_prepare` is everything a card can do alone: its queue, its mirrored
    // always-resident set, its residency plan, its pinned arena, its VRAM cache.
    // It touches no routed-expert byte.  Running the cards one after the other is
    // what put 31.9 GB on card 0 and 0 on card 1 while host RAM filled with card
    // 0's arena; one thread per card makes both devices climb together.
    //
    // It also removes a real inconsistency, not just a delay: the pinned-host cap
    // is DERIVED FROM LIVE /proc/meminfo, so under the sequential order card 1
    // read that file AFTER card 0 had already pinned tens of GB and therefore
    // planned against a smaller cap than card 0 did.  Both cards now read it
    // before either allocates.
    //
    // Nothing is shared and written: each thread owns its own runtime, its own
    // SYCL context, its own arena.  What IS shared is read-only — the bound model
    // and the mmap behind it — plus `madvise` calls on that mapping, which the
    // kernel serialises and which are idempotent for MADV_NORMAL.
    //
    // SYCL calls can throw; an exception escaping a std::thread is std::terminate,
    // so each thread converts one into this class's ordinary string error.
    rts_.resize(opt.n_cards);
    for (uint32_t c = 0; c < opt.n_cards; ++c) rts_[c] = std::make_unique<DeepSeek4Runtime>();
    std::vector<std::string> errs(opt.n_cards);
    auto fan_out = [&](const char* what, auto&& body) -> std::string {
        for (auto& e : errs) e.clear();
        if (opt.n_cards == 1) {
            // One card: run it on THIS thread.  A single-card orchestrator run is
            // asserted bit-identical to the plain single-card path, and adding a
            // thread it does not need would be a difference for its own sake.
            try { errs[0] = body(0u); }
            catch (const std::exception& ex) { errs[0] = std::string("threw: ") + ex.what(); }
            catch (...) { errs[0] = "threw a non-std exception"; }
        } else {
            std::vector<std::thread> th;
            th.reserve(opt.n_cards);
            for (uint32_t c = 0; c < opt.n_cards; ++c)
                th.emplace_back([&, c] {
                    try { errs[c] = body(c); }
                    catch (const std::exception& ex) { errs[c] = std::string("threw: ") + ex.what(); }
                    catch (...) { errs[c] = "threw a non-std exception"; }
                });
            // EVERY thread is joined before the first error is reported.  Returning
            // out from under a running thread would leave it writing into runtimes
            // this function is about to destroy.
            for (auto& t : th) t.join();
        }
        for (uint32_t c = 0; c < opt.n_cards; ++c)
            if (!errs[c].empty())
                return "DeepSeek4TpRuntime::load: " + std::string(what) + ": card " +
                       std::to_string(c) + ": " + errs[c];
        return {};
    };

    for (uint32_t c = 0; c < opt.n_cards; ++c)
        std::fprintf(stderr, "[ds4-tp] card %u of %u -> GPU %u\n", c, opt.n_cards, ordinals_[c]);
    trace_ = Ds4TpLoadTrace{};
    trace_.prepare_begin.assign(opt.n_cards, 0.0);
    trace_.prepare_end.assign(opt.n_cards, 0.0);
    const auto tprep = std::chrono::steady_clock::now();
    auto since = [&tprep] {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - tprep).count();
    };
    if (std::string e = fan_out("prepare", [&](uint32_t c) {
            Ds4Options o     = opt.base;
            o.n_cards        = opt.n_cards;
            o.card           = c;
            o.device_ordinal = ordinals_[c];
            trace_.prepare_begin[c] = since();
            std::string pe = rts_[c]->load_prepare(m, o);
            trace_.prepare_end[c]   = since();
            return pe;
        });
        !e.empty())
        return e;
    trace_.prepare_seconds = since();

    for (uint32_t c = 0; c < opt.n_cards; ++c) {
        // The binding is VERIFIED, not trusted: `device_ordinal` is an index into
        // an enumeration the runtime performs itself, so this proves the two
        // enumerations agree and that card c really did land on GPU ordinals_[c].
        if (rts_[c]->queue().get_device() != gpus[ordinals_[c]])
            return "DeepSeek4TpRuntime::load: card " + std::to_string(c) +
                   " was told GPU " + std::to_string(ordinals_[c]) + " (" +
                   gpus[ordinals_[c]].get_info<sycl::info::device::name>() +
                   ") but its queue landed on a different device (" +
                   rts_[c]->queue().get_device().get_info<sycl::info::device::name>() + ")";
    }

    // ---- which cross-card transfer path this run is on, stated ONCE ----
    //
    // `ds4_tp_reduce_host` stages every cross-card partial through pinned host
    // memory, and the reason on file for that is a peer-access measurement taken
    // on one board.  This checks it on THIS board, with the cards this run
    // actually bound, before the hour-long expert walk begins.  It changes
    // nothing: the host path is the only one implemented, and it is correct
    // whichever way the probe goes.  What it buys is that a run which pays for
    // host staging says so — a two-card forward that is slower than a P2P board
    // would be is then explicable from the log rather than a mystery.
    //
    // Peer access is refused between cards that sit under different PCIe root
    // ports with no shared upstream bridge; the kernel names that case exactly
    // ("do not share an upstream bridge or whitelisted host bridge") and there
    // is nothing this process can do about it but take the host path.
    if (opt.n_cards > 1) {
        std::vector<sycl::device> bound;
        bound.reserve(opt.n_cards);
        for (uint32_t c = 0; c < opt.n_cards; ++c) bound.push_back(rts_[c]->queue().get_device());
        (void)gpu_p2p_available(bound, "ds4-tp");
    }
    // The sliced prefill-size reducer (docs/deepseek4/72 Phase I): auxiliary
    // queues on these cards and the summing threads.  Decode sizes still take
    // ds4_tp_reduce_host inside it; the P2P form was measured and lost.
    {
        std::vector<sycl::queue*> rq(opt.n_cards);
        for (uint32_t c = 0; c < opt.n_cards; ++c) rq[c] = &rts_[c]->queue();
        if (std::string e = reducer_.init(rq); !e.empty()) return "DeepSeek4TpRuntime::load: " + e;
        std::fprintf(stderr, "[ds4-tp] cross-card reduction: %s\n",
                     reducer_.sliced() ? "sliced host path at prefill sizes (IE_DS4_REDUCE_SLICED=0 for the memcpy form)"
                                       : "memcpy host path");
    }

    // The slices must PARTITION the intermediate dimension: contiguous, in card
    // order, covering [0, expert_ffn) exactly once.  A gap loses weights and an
    // overlap double-counts them, and either shows up only as a wrong logit.
    //
    // Checked HERE, between the prepare and the file walk, and not after the whole
    // load as it used to be: the walk is the hour of work, and a split that does
    // not tile is knowable before a single expert byte is read.
    const uint32_t EF = m.config().expert_ffn;
    uint32_t covered = 0;
    for (uint32_t c = 0; c < opt.n_cards; ++c) {
        const Ds4ExpertSlice& s = rts_[c]->expert_slice();
        if (s.n_cards != opt.n_cards || s.card != c)
            return "DeepSeek4TpRuntime::load: card " + std::to_string(c) +
                   " loaded slice (card " + std::to_string(s.card) + " of " +
                   std::to_string(s.n_cards) + ") — the orchestrator asked for card " +
                   std::to_string(c) + " of " + std::to_string(opt.n_cards);
        if (s.ef0 != covered)
            return "DeepSeek4TpRuntime::load: card " + std::to_string(c) + " starts at " +
                   std::to_string(s.ef0) + " but the previous cards covered [0," +
                   std::to_string(covered) + ") — the slices do not tile the intermediate dimension";
        covered += s.efc;
    }
    if (covered != EF)
        return "DeepSeek4TpRuntime::load: the cards cover [0," + std::to_string(covered) +
               ") of an intermediate dimension of " + std::to_string(EF) +
               " — the split loses or duplicates weights";

    // ---- PHASE B: ONE walk over the routed-expert tensors, feeding every card ----
    //
    // WHY THIS IS THE BIG ONE.  Under hidden-dim expert-TP the cards hold the two
    // halves of the SAME expert.  For `ffn_gate_exps`/`ffn_up_exps` those halves
    // are adjacent runs of source columns, but `ffn_down_exps` is sliced along its
    // CONTRACTION dimension, so each card takes half of EVERY one of its columns —
    // a stride far below a page.  Both cards therefore touch every page of
    // `down`, and walking the file once per card read all of it twice, an hour
    // and 120 GB apart, which no page cache on a 128 GB box can bridge.
    //
    // Packing card 0's slice of expert e and then card 1's slice of the SAME
    // expert e keeps the reuse distance at one expert (single-digit MB), so the
    // pages the first pack faulted in are still resident for the second.  The
    // file is streamed forwards exactly once and consumed n_cards times.
    //
    // THE SYNCHRONISATION STORY, IN FULL: there is none to get wrong, by
    // construction.  This walk is single-threaded; the only buffer each pack
    // writes through is `pack_stage_`, which is a member of the card's OWN
    // runtime; and `Ds4ExpertCache::install_static` is blocking, so a card's
    // staging bytes are consumed before the next card's pack begins.  No USM
    // pointer ever crosses a context.  The alternative — one shared read buffer
    // handed to two threads — would need a handoff this does not have to specify.
    const uint32_t PL = rts_[0]->layers_pinned();
    const uint32_t NE = m.config().n_experts;
    for (uint32_t c = 1; c < opt.n_cards; ++c)
        if (rts_[c]->layers_pinned() != PL)
            return "DeepSeek4TpRuntime::load: card " + std::to_string(c) + " pinned " +
                   std::to_string(rts_[c]->layers_pinned()) + " layers but card 0 pinned " +
                   std::to_string(PL) +
                   " — the cards would not be walking the same file region";
    trace_.packed_bytes.assign(opt.n_cards, 0);
    const auto twalk = std::chrono::steady_clock::now();
    uint64_t run = 0, best_run = 0;
    uint32_t last_card = UINT32_MAX;
    for (uint32_t l = 0; l < PL; ++l) {
        for (uint32_t c = 0; c < opt.n_cards; ++c) rts_[c]->load_advise_layer(l);
        for (uint32_t e = 0; e < NE; ++e)
            for (uint32_t c = 0; c < opt.n_cards; ++c) {
                if (std::string s = rts_[c]->load_pack_expert(l, e); !s.empty())
                    return "DeepSeek4TpRuntime::load: card " + std::to_string(c) + ": " + s;
                if (c == last_card) ++run;
                else { run = 1; last_card = c; }
                best_run = std::max(best_run, run);
            }
        const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - twalk).count();
        Ds4TpLoadTrace::Sample sm;
        sm.seconds = el;
        sm.per_card.resize(opt.n_cards);
        uint64_t tot = 0;
        for (uint32_t c = 0; c < opt.n_cards; ++c) {
            sm.per_card[c] = rts_[c]->packed_expert_bytes();
            tot += sm.per_card[c];
        }
        trace_.timeline.push_back(std::move(sm));
        std::fprintf(stderr,
                     "[ds4-tp] expert pool: layer %u/%u  %.2f GB packed across %u cards"
                     "  %.1f s  %.1f MB/s\r",
                     l + 1, PL, double(tot) / 1e9, opt.n_cards, el,
                     double(tot) / 1e6 / std::max(el, 1e-9));
    }
    std::fprintf(stderr, "\n");
    trace_.pack_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - twalk).count();
    trace_.max_same_card_run = best_run;
    for (uint32_t c = 0; c < opt.n_cards; ++c)
        trace_.packed_bytes[c] = rts_[c]->packed_expert_bytes();

    // ---- PHASE C: per-card workspace, again concurrently ----
    if (std::string e = fan_out("finish", [&](uint32_t c) { return rts_[c]->load_finish(); });
        !e.empty())
        return e;

    // ---- the forward-pass workers ----
    // One per card ABOVE card 0 (card 0 runs on whichever thread calls
    // forward()).  Started here rather than inside forward() because a thread
    // creation per card per phase per layer would be 61*2*(n-1) of them for
    // every decoded token on the real model.
    for (uint32_t c = 1; c < opt.n_cards; ++c) {
        workers_.push_back(std::make_unique<Worker>());
        workers_.back()->start();
    }
    cleanup.armed = false;
    return {};
}

void DeepSeek4TpRuntime::release() noexcept {
    // The reducer's auxiliary queues are bound to these cards; a reload must
    // rebind them (init refuses a second call otherwise — gate I/J finding 3).
    reducer_.reset();
    // The workers first: they hold references to the runtimes below, so nothing
    // may be torn down while one could still be mid-job.
    for (auto& w : workers_) if (w) w->stop();
    workers_.clear();
    // Pinned staging first: each buffer belongs to its card's context, which
    // dies with the runtime.  The reduction's H2D out of these buffers is no
    // longer host-waited (see ds4_tp_reduce_host), so the LAST one may still be
    // in flight — draining the card's queue is what makes freeing its source
    // legal, and it is the same rule ensure_stage() follows.
    //
    // THE DRAIN CAN THROW, AND THIS FUNCTION IS `noexcept`.  When the Level-Zero
    // runtime has declared a device lost, `wait()` fails with a sycl::exception;
    // thrown out of a noexcept function that is std::terminate, and the process
    // dies with `terminate called after throwing an instance of
    // 'sycl::_V1::exception'` while the error that actually mattered — the one
    // `forward()` was about to return — never reaches the caller.  Freeing the
    // staging is still the right thing to attempt on a lost device, and a second
    // failure there is reported the same way rather than compounding it.
    for (size_t c = 0; c < stages_.size(); ++c)
        if (stages_[c] && c < rts_.size()) {
            try {
                rts_[c]->queue().wait();
                sycl::free(stages_[c], rts_[c]->queue());
            } catch (const sycl::exception& e) {
                std::fprintf(stderr,
                             "[ds4-tp] release: card %zu staging teardown failed: %s\n", c,
                             e.what());
            }
        }
    stages_.clear();
    stage_n_ = 0;
    rts_.clear();
    ordinals_.clear();
    reductions_ = 0;
    reduce_s_   = 0.0;
    trace_      = Ds4TpLoadTrace{};
    ftrace_     = Ds4TpForwardTrace{};
}

// Checked on EVERY card before ANY card is changed.  The cards are loaded with
// the same options so they cannot really disagree about the slot count, but
// "cannot really" is not a reason to leave half the cards switched when the
// last one refuses — that state is exactly the cross-sequence contamination
// this whole mechanism exists to make impossible.
std::string DeepSeek4TpRuntime::select_seq(uint32_t slot) noexcept {
    if (rts_.empty()) return "DeepSeek4TpRuntime::select_seq: not loaded";
    for (uint32_t c = 0; c < rts_.size(); ++c)
        if (slot >= rts_[c]->seq_slots())
            return "card " + std::to_string(c) + ": slot " + std::to_string(slot) +
                   " but that card was loaded with " +
                   std::to_string(rts_[c]->seq_slots()) + " sequence slots";
    for (auto& rt : rts_) (void)rt->select_seq(slot);
    return {};
}

std::string DeepSeek4TpRuntime::reset_seq(uint32_t slot) noexcept {
    if (rts_.empty()) return "DeepSeek4TpRuntime::reset_seq: not loaded";
    for (uint32_t c = 0; c < rts_.size(); ++c)
        if (slot >= rts_[c]->seq_slots())
            return "card " + std::to_string(c) + ": slot " + std::to_string(slot) +
                   " but that card was loaded with " +
                   std::to_string(rts_[c]->seq_slots()) + " sequence slots";
    for (auto& rt : rts_) (void)rt->reset_seq(slot);
    return {};
}

void DeepSeek4TpRuntime::reset_context() noexcept {
    for (auto& rt : rts_) rt->reset_context();
}

std::string DeepSeek4TpRuntime::set_vision(const float* rows, uint32_t t0, uint32_t n, uint32_t start_off) {
    for (auto& rt : rts_)
        if (std::string e = rt->set_vision(rows, t0, n, start_off); !e.empty()) return e;
    return {};
}

void DeepSeek4TpRuntime::clear_vision() noexcept {
    for (auto& rt : rts_) rt->clear_vision();
}

// Prompt cache across the cards.  The non-expert set is SPLIT, so no card holds
// the whole attention state and a snapshot that existed on some cards but not
// others would restore the conversation on part of the heads and a blank one on
// the rest — fluent text, wrong answer.  Both directions are all-or-nothing.
void DeepSeek4TpRuntime::free_snapshot() noexcept {
    for (auto& rt : rts_) rt->free_snapshot();
}

std::string DeepSeek4TpRuntime::snapshot_context() {
    if (rts_.empty()) return "DeepSeek4TpRuntime::snapshot_context: not loaded";
    for (size_t c = 0; c < rts_.size(); ++c)
        if (std::string e = rts_[c]->snapshot_context(); !e.empty()) {
            free_snapshot();
            return "card " + std::to_string(c) + ": " + e;
        }
    return {};
}

uint32_t DeepSeek4TpRuntime::restore_context() {
    if (rts_.empty()) return 0;
    uint32_t depth = 0;
    for (size_t c = 0; c < rts_.size(); ++c) {
        const uint32_t d = rts_[c]->restore_context();
        if (c == 0) depth = d;
        if (d == 0 || d != depth) {
            for (auto& rt : rts_) rt->reset_context();
            free_snapshot();
            return 0;
        }
    }
    return depth;
}

uint32_t DeepSeek4TpRuntime::snapshot_depth() const noexcept {
    return rts_.empty() ? 0 : rts_[0]->snapshot_depth();
}

uint64_t DeepSeek4TpRuntime::snapshot_bytes() const noexcept {
    uint64_t n = 0;
    for (const auto& rt : rts_) n += rt->snapshot_bytes();
    return n;
}

std::string DeepSeek4TpRuntime::snapshot_to_host(std::vector<Ds4HostSnapshot>& per_card) {
    if (rts_.empty()) return "DeepSeek4TpRuntime::snapshot_to_host: not loaded";
    per_card.resize(rts_.size());
    for (size_t c = 0; c < rts_.size(); ++c)
        if (std::string e = rts_[c]->snapshot_to_host(per_card[c]); !e.empty()) {
            for (auto& hs : per_card) hs.depth = 0;
            return "card " + std::to_string(c) + ": " + e;
        }
    return {};
}

uint32_t DeepSeek4TpRuntime::restore_from_host(const std::vector<Ds4HostSnapshot>& per_card) {
    if (rts_.empty() || per_card.size() != rts_.size()) return 0;
    uint32_t depth = 0;
    for (size_t c = 0; c < rts_.size(); ++c) {
        const uint32_t d = rts_[c]->restore_from_host(per_card[c]);
        if (c == 0) depth = d;
        if (d == 0 || d != depth) {   // a card failed or the cards disagree: every card starts clean
            for (auto& rt : rts_) rt->reset_context();
            return 0;
        }
    }
    return depth;
}

std::string DeepSeek4TpRuntime::ensure_stage(uint64_t n) {
    if (n <= stage_n_ && stages_.size() == rts_.size()) return {};
    // GROWTH ONLY — this path runs when a larger T is seen, never per layer.
    // The previous forward's LAST reduction pushed H2D out of these buffers and
    // did not host-wait for it, so the copy may still be reading the pages this
    // is about to unpin.  Drain the card's queue first; `sycl::free` is not
    // ordered against anything the queue is doing.
    for (size_t c = 0; c < stages_.size(); ++c)
        if (stages_[c] && c < rts_.size()) {
            rts_[c]->queue().wait();
            sycl::free(stages_[c], rts_[c]->queue());
        }
    stages_.assign(rts_.size(), nullptr);
    stage_n_ = 0;
    for (size_t c = 0; c < rts_.size(); ++c) {
        // PINNED, and in card c's OWN context — see ds4_tp_reduce_host.  A failed
        // pin is a hard error; there is no pageable form of this buffer that the
        // rest of the subsystem's rules would allow.
        stages_[c] = sycl::malloc_host<float>(size_t(n), rts_[c]->queue());
        if (!stages_[c])
            return "DeepSeek4TpRuntime: could not pin " + std::to_string(n * 4) +
                   " B of host staging for card " + std::to_string(c) +
                   "'s cross-card reduction";
    }
    stage_n_ = n;
    return {};
}

// $IE_DS4_TP_TRACE=1 — one line per forward, in ms, on stderr.
//
// WHY THIS EXISTS.  A decode step that is GPU-bound and one that is
// submission-bound have the same wall and the same kernel time; the only thing
// that tells them apart is where the HOST was while the wall ran.  Every number
// below is measured on the orchestrator's own clock and they tile `wall` by
// construction (see Ds4TpForwardTrace), so nothing can hide in a residual that
// is not printed: `gap` IS the residual, and it is printed.
//
//   pro/epi  prologue and epilogue, outside the layer loop
//   span     wall spent inside run_cards = Σ (last card out − first card in)
//   red      wall spent in the cross-card reductions, and how many
//   gap      wall − pro − span − red − epi: fork/join, lockstep check, bookkeeping
//   busy     per card, Σ of its own segment durations (≤ span when overlapped)
//   wait     per card, of `busy`, the part spent BLOCKED on its own queue,
//            split by barrier with the call count at each
// `busy − wait` is therefore the host time that card spent SUBMITTING, which is
// the quantity a launch-count reduction moves and a faster kernel does not.
static void ds4_trace_forward(const Ds4TpForwardTrace& f) {
    static const bool on = std::getenv("IE_DS4_TP_TRACE") != nullptr;
    if (!on) return;
    const double ms = 1e3;
    const double gap = f.wall_seconds - f.prologue_seconds - f.span_seconds -
                       f.reduce_seconds - f.epilogue_seconds;
    std::string s;
    char b[512];
    std::snprintf(b, sizeof(b),
                  "[ds4-tp] T=%u cards=%u%s wall=%.3f | pro=%.3f span=%.3f"
                  "(pre=%.3f mid=%.3f post=%.3f) red=%.3f/%llu epi=%.3f gap=%.3f",
                  f.tokens, f.n_cards, f.sequential ? " SEQUENTIAL" : "", f.wall_seconds * ms,
                  f.prologue_seconds * ms, f.span_seconds * ms, f.phase_span[0] * ms,
                  f.phase_span[1] * ms, f.phase_span[2] * ms, f.reduce_seconds * ms,
                  (unsigned long long)f.reductions, f.epilogue_seconds * ms, gap * ms);
    s = b;
    for (uint32_t c = 0; c < f.n_cards && c < f.card_busy.size() && c < f.card_waits.size(); ++c) {
        const DeepSeek4Runtime::HostWaits& w = f.card_waits[c];
        std::snprintf(b, sizeof(b),
                      " | c%u busy=%.3f submit=%.3f wait=%.3f"
                      "(rt=%.3f/%llu moe=%.3f/%llu post=%.3f/%llu oth=%.3f/%llu)",
                      c, f.card_busy[c] * ms, (f.card_busy[c] - w.total()) * ms, w.total() * ms,
                      w.router * ms, (unsigned long long)w.n_router,
                      w.moe * ms, (unsigned long long)w.n_moe,
                      w.post * ms, (unsigned long long)w.n_post,
                      w.other * ms, (unsigned long long)w.n_other);
        s += b;
    }
    std::snprintf(b, sizeof(b), " | ovl=%.3f in %llu/%llu seg\n", f.overlap_seconds * ms,
                  (unsigned long long)f.overlapped, (unsigned long long)f.total);
    s += b;
    std::fputs(s.c_str(), stderr);
}

// $IE_DS4_REDUCE_HASH=<path>: one line per cross-card reduction — chunk position,
// T, layer, site (1 = attention output, 2 = routed/shared experts) and a 64-bit
// hash of the reduced buffer as the host summed it.  Two runs' files diffed
// give the first layer and site whose bytes differ (docs/deepseek4/72, the
// batch-path nondeterminism hunt).  Off unless set; costs ~1 ms per reduction.
static void ds4_reduce_hash_note(uint32_t pos0, uint32_t T, uint32_t L, uint32_t site,
                                 const float* stage, uint64_t n) {
    static FILE* f = [] {
        const char* p = std::getenv("IE_DS4_REDUCE_HASH");
        return (p && *p) ? std::fopen(p, "a") : nullptr;
    }();
    if (!f) return;
    uint64_t h = 1469598103934665603ull;
    const uint64_t* w = reinterpret_cast<const uint64_t*>(stage);
    for (uint64_t i = 0; i < n / 2; ++i) { h ^= w[i]; h *= 1099511628211ull; }
    if (n & 1) { uint32_t last; std::memcpy(&last, stage + n - 1, 4); h ^= last; h *= 1099511628211ull; }
    std::fprintf(f, "pos %u T %u L %u site %u %016llx\n", pos0, T, L, site, (unsigned long long)h);
    std::fflush(f);
}

std::string DeepSeek4TpRuntime::forward(const int32_t* input_ids, uint32_t T, uint32_t pos0,
                                        float* logits, bool last_only) {
    if (rts_.empty()) return "DeepSeek4TpRuntime::forward: not loaded";
    const uint32_t n = uint32_t(rts_.size());
    const DeepSeek4Config& cfg = rts_[0]->cfg_;
    const uint32_t H = cfg.hidden, K = cfg.n_experts_used;
    if (T == 0 || T > rts_[0]->ws_T_)
        return "DeepSeek4TpRuntime::forward: T=" + std::to_string(T) + " outside [1," +
               std::to_string(rts_[0]->ws_T_) + "]";
    const uint64_t moe_n = uint64_t(T) * H;
    // Staging must cover the LARGEST reduction any layer will issue: 2*moe_n
    // when the shared expert is sliced alongside the routed ones.
    if (n > 1)
        if (std::string e = ensure_stage(rts_[0]->attn_slice().split ? 2 * moe_n : moe_n);
            !e.empty())
            return e;

    const auto wall0 = std::chrono::steady_clock::now();

    // ---- the concurrency trace, reset for this call ----
    // `ie::g_profiler` is a process-global KernelProfiler whose entry list is a
    // bare std::vector: every ie::ps() submission push_back()s into it.  Two
    // cards submitting at once would race it, so a profiled forward takes turns
    // instead — stated here, recorded in the trace, and never inferred from a
    // suspiciously tidy overlap number.
    ftrace_            = Ds4TpForwardTrace{};
    ftrace_.n_cards    = n;
    ftrace_.sequential = (g_profiler != nullptr);
    ftrace_.total      = uint64_t(cfg.n_layers) * 3u;
    ftrace_.begin.assign(ftrace_.total * n, 0.0);
    ftrace_.end.assign(ftrace_.total * n, 0.0);
    ftrace_.tokens = T;

    uint32_t bad = 0;
    if (std::string e = run_cards(UINT32_MAX, wall0,
                                  [&](uint32_t c) {
                                      return rts_[c]->forward_prologue(input_ids, T, pos0);
                                  },
                                  &bad);
        !e.empty())
        return "card " + std::to_string(bad) + ": " + e;
    ftrace_.prologue_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();

    std::vector<sycl::queue*> qs(n);
    std::vector<float*>       parts(n);
    for (uint32_t c = 0; c < n; ++c) qs[c] = &rts_[c]->queue();

    // Does the attention output need reducing?  Only when the non-expert set was
    // TP-split; mirrored, every card already computed the whole of it and a
    // reduction would double it.  Read off card 0's resolved slice rather than
    // from the options, so it describes what the load ACTUALLY did.
    const bool split_attn = n > 1 && rts_[0]->attn_slice().split;
    for (uint32_t c = 1; c < n; ++c)
        if (rts_[c]->attn_slice().split != split_attn)
            return "DeepSeek4TpRuntime::forward: card " + std::to_string(c) +
                   " loaded with a different non-expert split than card 0 — the cards would be"
                   " reducing buffers that do not mean the same thing.  Refusing.";

    for (uint32_t L = 0; L < cfg.n_layers; ++L) {
        // ---- every card up to its ATTENTION-OUTPUT partial, AT THE SAME TIME ----
        // Card 0 on this thread, cards 1..n-1 on their own; run_cards joins them
        // all before it returns, on the failing path too.  The segment ends where
        // the data actually joins — the cross-card reduction below — and nowhere
        // earlier.
        if (std::string e = run_cards(
                uint32_t(L) * 3u + 0u, wall0,
                [&](uint32_t c) { return rts_[c]->layer_forward_pre(L, T, pos0); }, &bad);
            !e.empty())
            return "card " + std::to_string(bad) + ", layer " + std::to_string(L) + ": " + e;

        // ---- REDUCTION 1: the attention output ----
        // `attn_output_b` contracted over this card's output groups only, so
        // `ws_sub_` is a partial.  Everything downstream of it — the residual
        // mix, the FFN-site hyper-connection's 20 Sinkhorn iterations, the
        // RMSNorm and the ROUTER — is nonlinear, so the sum has to be whole
        // before any of it runs.  There is no way to defer this into the MoE
        // reduction below, and that is exactly why splitting the attention costs
        // a second round trip per layer.
        if (split_attn) {
            for (uint32_t c = 0; c < n; ++c) parts[c] = rts_[c]->ws_sub_;
            const auto a0 = std::chrono::steady_clock::now();
            if (std::string e = reducer_.reduce(qs, parts, stages_, moe_n); !e.empty())
                return "DeepSeek4TpRuntime::forward: layer " + std::to_string(L) +
                       " (attention output): " + e;
            ds4_reduce_hash_note(pos0, T, L, 1, stages_[0], moe_n);
            const double a1 =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - a0).count();
            reduce_s_ += a1;
            ftrace_.reduce_seconds += a1;
            ++reductions_;
            ++ftrace_.reductions;
        }

        // ---- every card up to its routed-expert partial ----
        if (std::string e = run_cards(
                uint32_t(L) * 3u + 1u, wall0,
                [&](uint32_t c) { return rts_[c]->layer_forward_mid(L, T); }, &bad);
            !e.empty())
            return "card " + std::to_string(bad) + ", layer " + std::to_string(L) + ": " + e;

        if (n > 1) {
            // ---- the lockstep invariant, checked rather than assumed ----
            // Hidden-dim TP is only a partition if every card ran the SAME
            // experts with the SAME weights.  Both arrays came off the device in
            // layer_forward_mid, so this costs no transfer.
            //
            // IT IS A STRONGER CHECK NOW, NOT A WEAKER ONE.  With the attention
            // split, the cards' activations diverge between the two reductions —
            // each holds a partial — and reconverge only because REDUCTION 1
            // handed every card the identical summed buffer.  The router runs
            // downstream of that, so equal ids AND bit-equal routing weights are
            // a checksum of the attention reduction as well as of the mirrored
            // chain it always covered.
            const int32_t* id0 = rts_[0]->h_ridx_;
            const float*   rw0 = rts_[0]->h_rw_;
            for (uint32_t c = 1; c < n; ++c) {
                const int32_t* idc = rts_[c]->h_ridx_;
                const float*   rwc = rts_[c]->h_rw_;
                for (uint64_t i = 0; i < uint64_t(T) * K; ++i) {
                    if (idc[i] != id0[i])
                        return "DeepSeek4TpRuntime::forward: layer " + std::to_string(L) +
                               ": card " + std::to_string(c) + " routed token " +
                               std::to_string(i / K) + " slot " + std::to_string(i % K) +
                               " to expert " + std::to_string(idc[i]) + " but card 0 chose " +
                               std::to_string(id0[i]) +
                               ".  Hidden-dim expert-TP is only a partition if every card runs the"
                               " SAME experts; summing partials of different experts would be a"
                               " plausible-looking wrong answer.  Refusing.";
                    if (rwc[i] != rw0[i]) {
                        // Printed to FULL precision on purpose: the values that
                        // reach here differ in the last bits, and %f would show
                        // the same number twice and make the diagnostic useless.
                        char w[80];
                        std::snprintf(w, sizeof(w), "%.9g but card 0 by %.9g",
                                      double(rwc[i]), double(rw0[i]));
                        return "DeepSeek4TpRuntime::forward: layer " + std::to_string(L) +
                               ": card " + std::to_string(c) + " weighted token " +
                               std::to_string(i / K) + " slot " + std::to_string(i % K) +
                               " by " + w +
                               ".  The non-expert weights are mirrored, so the cards' activations"
                               " must be bit-identical; a difference here means they have drifted"
                               " and the partials no longer belong to the same forward pass."
                               "  Refusing.";
                    }
                }
            }

            // ---- REDUCTION 2: the routed experts, AND the shared expert ----
            // `ds4_shared_out(T)` sits immediately after `ws_moe_` in one
            // allocation, so when the shared expert is sliced too both partials
            // are one contiguous 2*T*hidden run and ONE round trip carries them.
            // Mirrored, the shared expert is whole on every card and must NOT be
            // reduced — reducing it would double it — so only `ws_moe_` goes.
            for (uint32_t c = 0; c < n; ++c) parts[c] = rts_[c]->ws_moe_;
            const auto r0 = std::chrono::steady_clock::now();
            if (std::string e = reducer_.reduce(qs, parts, stages_,
                                                split_attn ? 2 * moe_n : moe_n);
                !e.empty())
                return "DeepSeek4TpRuntime::forward: layer " + std::to_string(L) + ": " + e;
            ds4_reduce_hash_note(pos0, T, L, 2, stages_[0], split_attn ? 2 * moe_n : moe_n);
            const double r1 =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - r0).count();
            reduce_s_ += r1;
            ftrace_.reduce_seconds += r1;
            ++reductions_;
            ++ftrace_.reductions;
        }

        // ---- the FFN-site hyper-connection mix, on the reduced sum ----
        if (std::string e =
                run_cards(uint32_t(L) * 3u + 2u, wall0,
                          [&](uint32_t c) { return rts_[c]->layer_forward_post(L, T); }, &bad);
            !e.empty())
            return "card " + std::to_string(bad) + ", layer " + std::to_string(L) + ": " + e;
    }

    // ---- what the drive actually did, computed from what it recorded ----
    // A segment counts as OVERLAPPED only if every card's interval intersects
    // every other card's, and `overlap_seconds` is the n-way intersection.  Both
    // are exactly zero under a sequential drive: card c+1's begin is then after
    // card c's end, because card c's segment ends in a q.wait().
    ftrace_.card_busy.assign(n, 0.0);
    for (uint64_t s = 0; s < ftrace_.total; ++s) {
        double lo = ftrace_.begin[s * n], hi = ftrace_.end[s * n];
        // The SPAN of a segment is the wall the drive spent in it: from the
        // first card that entered to the last that left.  Σ span is therefore
        // the part of `wall` that run_cards is answerable for, and it is the
        // union of what the cards did, never the sum.
        double first = ftrace_.begin[s * n], last = ftrace_.end[s * n];
        for (uint32_t c = 0; c < n; ++c) {
            lo = std::max(lo, ftrace_.begin[s * n + c]);
            hi = std::min(hi, ftrace_.end[s * n + c]);
            first = std::min(first, ftrace_.begin[s * n + c]);
            last  = std::max(last,  ftrace_.end[s * n + c]);
            const double d = ftrace_.end[s * n + c] - ftrace_.begin[s * n + c];
            ftrace_.busy_seconds += d;
            ftrace_.card_busy[c] += d;
        }
        const double span = last - first;
        ftrace_.span_seconds += span;
        ftrace_.phase_span[s % 3u] += span;
        if (n > 1 && hi > lo) { ++ftrace_.overlapped; ftrace_.overlap_seconds += hi - lo; }
    }

    // The head is mirrored, so every card would produce the same logits; only
    // card 0's are computed.  The other cards' `ws_coll_`/`ws_norm_`/`ws_logits_`
    // are scratch that the next layer 0 overwrites, so skipping their head leaves
    // them in an equivalent state — `ws_streams_`, which carries the residual
    // across calls, is written by layer_forward_post and untouched here.
    const auto epi0 = std::chrono::steady_clock::now();
    if (std::string e = rts_[0]->forward_epilogue(T, logits, last_only); !e.empty())
        return "card 0: " + e;
    for (uint32_t c = 1; c < n; ++c) rts_[c]->cache().collect_dma_time();
    ftrace_.epilogue_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - epi0).count();

    const double wall =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();
    for (uint32_t c = 0; c < n; ++c) {
        const auto& s = rts_[c]->cache().stats();
        rts_[c]->timing_.wall_seconds  = wall;
        rts_[c]->timing_.dma_seconds   = s.dma_seconds;
        rts_[c]->timing_.bytes_fetched = s.bytes_fetched;
        rts_[c]->timing_.hits          = s.hits;
        rts_[c]->timing_.misses        = s.misses;
        rts_[c]->accumulate_cache(T);
    }
    ftrace_.wall_seconds = wall;
    ftrace_.card_waits.resize(n);
    for (uint32_t c = 0; c < n; ++c) ftrace_.card_waits[c] = rts_[c]->last_host_waits();
    ds4_trace_forward(ftrace_);
    return {};
}

}  // namespace ie
