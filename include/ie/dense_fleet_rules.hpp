// include/ie/dense_fleet_rules.hpp -- which one-card dense GGUFs take the fleet class (DenseModelTP on a one-card fleet).
//
// 2026-10-28 (serving campaign P1, docs/research/2026-10-28-nanbeige43-serving-campaign.md): the dense serving features
// (the prompt cache, then the lookup drafts and the request lanes) are built on DenseModelTP, so a one-card load goes
// there too when the class can load the file. DenseModelTP::load (src/model/dense_tp.cpp) takes: column-parallel q / k /
// v / gate / up in Q4_K, Q6_K or F16 -- except a llama file's q and k, which are row-permuted at load and so Q4_K / Q6_K
// only (up_col_perm); row-parallel o-proj and ffn_down in Q4_K / Q6_K (up_row); token_embd and output in Q4_K / Q6_K.
// Anything else (a Q8_0 or Q5_K projection, which DenseModel expands to F16 on one card) keeps the previous one-card
// path. Every layer is checked: the loader reads them all, and a mixed-quant file would otherwise fail at load instead
// of falling back. Host-tested in tests/unit/dense_fleet_rules_test.cpp.
#pragma once

#include "ie/dtype.hpp"
#include "ie/gguf.hpp"

#include <cstdint>
#include <cstdio>

namespace ie {

struct DenseFleetDtypes {
    DType q, k, v, o, gate, up, down;   // one layer's projections
    DType embd, output;                 // token_embd and the lm_head (tied: the same)
    bool  llama;                        // llama arch: q / k go through the row permute (Q4_K / Q6_K only)
};

inline bool dense_fleet_dtypes_ok(const DenseFleetDtypes& d) {
    const auto kq  = [](DType t) { return t == DType::kQ4_K || t == DType::kQ6_K; };
    const auto col = [&](DType t) { return kq(t) || t == DType::kF16; };
    if (!kq(d.o) || !kq(d.down) || !kq(d.embd) || !kq(d.output)) return false;
    if (!col(d.v) || !col(d.gate) || !col(d.up)) return false;
    return d.llama ? (kq(d.q) && kq(d.k)) : (col(d.q) && col(d.k));
}

// The geometry DenseModelTP::load requires on one card: the o-proj's K slice (n_q_heads * head_dim) and the FFN width on
// 256-superblock boundaries (up_row re-packs whole superblocks). (The P1 gate: a misaligned file would otherwise fail the
// TP load instead of falling back; a K-quant file implies K % 256, so this is belt and braces.)
inline bool dense_fleet_geometry_ok(uint32_t n_q_dim, uint32_t ffn) {
    return n_q_dim && ffn && (n_q_dim % 256) == 0 && (ffn % 256) == 0;
}

// The file's tensors against the rules above, every layer. A missing tensor counts as unloadable (DType::kCount).
inline bool dense_fleet_loads(const GgufReader& g, uint32_t n_layers, bool llama, uint32_t n_q_dim, uint32_t ffn) {
    if (!dense_fleet_geometry_ok(n_q_dim, ffn)) return false;
    const auto dt = [&](const char* name) { const auto* t = g.find_tensor(name); return t ? t->dtype : DType::kCount; };
    const DType embd = dt("token_embd.weight");
    const auto* out = g.find_tensor("output.weight");
    const DType output = out ? out->dtype : embd;
    char buf[64];
    const auto dl = [&](uint32_t L, const char* name) { std::snprintf(buf, sizeof buf, "blk.%u.%s", L, name); return dt(buf); };
    for (uint32_t L = 0; L < n_layers; ++L) {
        const DenseFleetDtypes d{dl(L, "attn_q.weight"), dl(L, "attn_k.weight"), dl(L, "attn_v.weight"),
                                 dl(L, "attn_output.weight"), dl(L, "ffn_gate.weight"), dl(L, "ffn_up.weight"),
                                 dl(L, "ffn_down.weight"), embd, output, llama};
        if (!dense_fleet_dtypes_ok(d)) return false;
    }
    return n_layers > 0;
}

}  // namespace ie
