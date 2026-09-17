// tools/qwen3next_config_test.cpp — Qwen3-Next-80B Task 0 gate.
//
// Validates detect_arch + read_qwen3next_config against a real `qwen3next`
// GGUF: parses every config field and prints them for an eyeball check vs
// `ie-inspect --kv`. No model load, no forward (those land in later tasks).
#include "ie/gguf.hpp"
#include "ie/model_config.hpp"

#include <cstdio>
#include <string>

namespace {
const char* dt_name(ie::DType d) {
    switch (d) {
        case ie::DType::kF32:  return "F32";
        case ie::DType::kF16:  return "F16";
        case ie::DType::kQ4_K: return "Q4_K";
        case ie::DType::kQ5_K: return "Q5_K";
        case ie::DType::kQ6_K: return "Q6_K";
        case ie::DType::kQ8_0: return "Q8_0";
        default: return "?";
    }
}
// Full 48-layer tensor-inventory + dataflow-dim validation. Proves the loader's
// complete tensor map (names/dtypes/shapes for both layer kinds + MoE + shared
// expert) against the real GGUF, before the forward is written. Returns # fails.
int check_inventory(const ie::GgufReader& g, const ie::Qwen3NextConfig& c) {
    const auto& d = c.hybrid.dense;
    const uint32_t conv_ch = c.hybrid.ssm_inner + 2u * c.hybrid.ssm_n_k_heads * c.hybrid.ssm_state;
    const uint32_t q_gate  = 2u * d.n_q_heads * d.head_dim;     // joint Q|gate width
    const uint32_t ba      = 2u * c.hybrid.ssm_n_v_heads;       // fused beta|alpha width
    int fails = 0;
    char buf[64];
    auto chk = [&](uint32_t L, const char* nm, bool want_f32, int64_t want_dim1) {
        std::snprintf(buf, sizeof(buf), "blk.%u.%s", L, nm);
        const auto* t = g.find_tensor(buf);
        if (!t) { std::printf("    ✗ L%u %-22s MISSING\n", L, nm); ++fails; return; }
        if (want_f32 && t->dtype != ie::DType::kF32) {
            std::printf("    ✗ L%u %-22s dtype %s != F32\n", L, nm, dt_name(t->dtype)); ++fails; return; }
        if (want_dim1 >= 0 && t->n_dims >= 2 && int64_t(t->shape[1]) != want_dim1) {
            std::printf("    ✗ L%u %-22s shape[1] %llu != %lld\n", L, nm,
                        (unsigned long long)t->shape[1], (long long)want_dim1); ++fails; return; }
    };
    std::printf("\n[inventory] conv_channels=%u  joint Q|gate=%u  fused ba=%u  (asserting these)\n",
                conv_ch, q_gate, ba);
    for (uint32_t L = 0; L < c.hybrid.n_transformer_layers(); ++L) {
        const bool dn = c.hybrid.recurrent_layer(L);
        // shared (both kinds)
        chk(L, "attn_norm.weight", true, -1);
        chk(L, "post_attention_norm.weight", true, -1);
        // MoE + shared expert (every layer)
        chk(L, "ffn_gate_inp.weight", true, -1);          // router [hidden, n_experts]
        chk(L, "ffn_gate_exps.weight", false, -1);
        chk(L, "ffn_up_exps.weight",   false, -1);
        chk(L, "ffn_down_exps.weight", false, -1);
        chk(L, "ffn_gate_inp_shexp.weight", true, -1);    // shared-expert sigmoid gate
        chk(L, "ffn_gate_shexp.weight", false, -1);
        chk(L, "ffn_up_shexp.weight",   false, -1);
        chk(L, "ffn_down_shexp.weight", false, -1);
        if (dn) {
            chk(L, "attn_qkv.weight",  false, conv_ch);   // fused q|k|v conv input
            chk(L, "attn_gate.weight", false, c.hybrid.ssm_inner);
            chk(L, "ssm_a",            true,  -1);
            chk(L, "ssm_ba.weight",    false, ba);        // fused beta|alpha
            chk(L, "ssm_conv1d.weight",true,  conv_ch);
            chk(L, "ssm_dt.bias",      true,  -1);
            chk(L, "ssm_norm.weight",  true,  -1);
            chk(L, "ssm_out.weight",   false, d.hidden);
        } else {
            chk(L, "attn_q.weight",      false, q_gate);  // joint Q|gate
            chk(L, "attn_k.weight",      false, -1);
            chk(L, "attn_v.weight",      false, -1);
            chk(L, "attn_q_norm.weight", true, -1);
            chk(L, "attn_k_norm.weight", true, -1);
            chk(L, "attn_output.weight", false, d.hidden);
        }
    }
    // top-level
    if (!g.find_tensor("token_embd.weight")) { std::printf("    ✗ token_embd MISSING\n"); ++fails; }
    if (!g.find_tensor("output_norm.weight")) { std::printf("    ✗ output_norm MISSING\n"); ++fails; }
    return fails;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <qwen3next.gguf>\n", argv[0]); return 2; }
    ie::GgufReader g;
    if (auto e = g.open(argv[1]); !e.empty()) { std::fprintf(stderr, "open: %s\n", e.c_str()); return 1; }

    const auto arch = ie::detect_arch(g);
    if (arch != ie::ModelArch::kQwen3Next) {
        std::fprintf(stderr, "detect_arch != kQwen3Next (got %d)\n", int(arch));
        return 1;
    }
    ie::Qwen3NextConfig c;
    if (auto e = ie::read_qwen3next_config(g, c); !e.empty()) {
        std::fprintf(stderr, "read_qwen3next_config: %s\n", e.c_str());
        return 1;
    }
    const auto& d = c.hybrid.dense;
    const auto& h = c.hybrid;
    std::printf("CONFIG OK  arch=qwen3next\n");
    std::printf("  layers              : %u  (transformer %u, nextn %u)\n",
                d.n_layers, h.n_transformer_layers(), h.nextn_predict_layers);
    std::printf("  hidden / vocab      : %u / %u\n", d.hidden, d.vocab);
    std::printf("  attn heads (q/kv)   : %u / %u   head_dim %u\n",
                d.n_q_heads, d.n_kv_heads, d.head_dim);
    std::printf("  rope dim / theta    : %u / %.0f   eps %.1e\n",
                d.rope_dim, double(d.rope_theta), double(d.rms_eps));
    std::printf("  full_attn_interval  : %u   →  %u DeltaNet + %u full-attn layers\n",
                h.full_attn_interval,
                h.n_transformer_layers() - h.n_transformer_layers() / h.full_attn_interval,
                h.n_transformer_layers() / h.full_attn_interval);
    std::printf("  DeltaNet ssm        : inner %u, state %u, k_heads %u, v_heads %u, "
                "v_head_dim %u, conv %u\n",
                h.ssm_inner, h.ssm_state, h.ssm_n_k_heads, h.ssm_n_v_heads,
                h.ssm_v_head_dim, h.ssm_conv_kernel);
    std::printf("  MoE                 : %u experts / %u used, E_ffn %u, shared E_ffn %u\n",
                c.n_experts, c.n_experts_used, c.expert_ffn, c.shared_expert_ffn);
    // Spot the layer pattern (first 8 layers).
    std::printf("  layer kinds [0..7]  : ");
    for (uint32_t L = 0; L < 8 && L < h.n_transformer_layers(); ++L)
        std::printf("%s ", h.recurrent_layer(L) ? "DN" : "ATTN");
    std::printf("\n");

    const int fails = check_inventory(g, c);
    if (fails == 0)
        std::printf("\n[inventory] ALL %u layers OK — every tensor present, dtypes + "
                    "dataflow dims (conv_ch/Q|gate/ba/inner/hidden) match.\n",
                    h.n_transformer_layers());
    else
        std::printf("\n[inventory] %d FAILURES — tensor map incomplete.\n", fails);
    return fails == 0 ? 0 : 1;
}
