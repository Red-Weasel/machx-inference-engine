// P1 gate: does the REAL Laguna-S-2.1 GGUF carry exactly the tensors the port
// plan assumes, at the shapes the config implies — INCLUDING the per-layer head
// count?  Headers only; no weight bytes.  SKIPs when the model is absent.
//
// THE CENTRAL ASSERTION.  `attention.head_count` is a 48-entry ARRAY (48 on
// full-attention layers, 72 on sliding) and the ATTENTION TENSOR SHAPES TRACK
// IT: attn_q is [hidden, heads*head_dim], attn_gate is [hidden, heads],
// attn_output is [heads*head_dim, hidden].  A loader that used one head count
// for the model would allocate 3/4 of its layers wrong — this gate is what
// makes that impossible to ship.
#undef NDEBUG
#include "ie/gguf.hpp"
#include "ie/model_config.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {
// Model path comes from $IE_LAGUNA_GGUF — these gates need the real Laguna-S-2.1
// GGUF, which is far too large to ship. Unset (or unreachable) means the
// test SKIPS cleanly, so a checkout without the model still goes green.
const char* shard1_path() { return std::getenv("IE_LAGUNA_GGUF"); }
std::string dims_str(const ie::GgufTensorInfo& t) {
    std::string s = "[";
    for (uint32_t i = 0; i < t.n_dims; ++i) s += (i ? ", " : "") + std::to_string(t.shape[i]);
    return s + "]";
}
}  // namespace

int main() {
    const char* path = shard1_path();
    if (!path || !*path) {
        std::puts("laguna_manifest_test: SKIPPED (set IE_LAGUNA_GGUF to the model's shard-1 path)");
        return 0;
    }
    ie::GgufReader g;
    if (!g.open(path).empty()) {
        std::puts("laguna_manifest_test: SKIPPED (model not present)");
        return 0;
    }
    ie::LagunaConfig c;
    const std::string e = ie::read_laguna_config(g, c);
    if (!e.empty()) { std::printf("FAIL: config: %s\n", e.c_str()); return 1; }

    int fails = 0;
    auto need = [&](const std::string& n, std::vector<uint64_t> want) {
        const auto* t = g.find_tensor(n);
        if (!t) { std::printf("  MISSING %s\n", n.c_str()); ++fails; return; }
        bool ok = t->n_dims == want.size();
        for (size_t i = 0; ok && i < want.size(); ++i) ok = t->shape[i] == want[i];
        if (!ok) {
            std::printf("  SHAPE   %-30s got %s want [", n.c_str(), dims_str(*t).c_str());
            for (size_t i = 0; i < want.size(); ++i)
                std::printf("%s%llu", i ? ", " : "", (unsigned long long)want[i]);
            std::puts("]");
            ++fails;
        }
    };
    const uint32_t H = c.hidden, HD = c.head_dim;
    const uint64_t kv_dim = uint64_t(c.n_kv_heads) * HD;
    std::printf("layers=%u H=%u kv_heads=%u head_dim=%u experts=%u top-%u dense=%u\n",
                c.n_layers, H, c.n_kv_heads, HD, c.n_experts, c.n_experts_used, c.dense_layers);

    need("token_embd.weight", {H, c.vocab});
    need("output.weight", {H, c.vocab});
    need("output_norm.weight", {H});
    // Laguna has NO token_embd_norm (Inkling does) and NO attention-sink
    // tensor (`swa_attention_sink_enabled` is off in this build) — asserted as
    // ABSENT so a future file that adds them is caught rather than ignored.
    if (g.find_tensor("token_embd_norm.weight")) {
        std::puts("  UNEXPECTED token_embd_norm.weight — layout assumption changed"); ++fails;
    }

    uint32_t n_full = 0, n_slide = 0, n_dense = 0, n_moe = 0;
    for (uint32_t L = 0; L < c.n_layers; ++L) {
        const std::string b = "blk." + std::to_string(L) + ".";
        const uint64_t heads = c.heads_for(L);
        const uint64_t q_dim = heads * HD;
        (c.is_sliding(L) ? n_slide : n_full)++;
        need(b + "attn_norm.weight", {H});
        need(b + "attn_q.weight", {H, q_dim});
        need(b + "attn_k.weight", {H, kv_dim});
        need(b + "attn_v.weight", {H, kv_dim});
        need(b + "attn_output.weight", {q_dim, H});
        need(b + "attn_q_norm.weight", {HD});
        need(b + "attn_k_norm.weight", {HD});
        // Per-head attention output gating: ONE gate per head, hence `heads`.
        need(b + "attn_gate.weight", {H, heads});
        need(b + "ffn_norm.weight", {H});
        if (g.find_tensor(b + "sink.weight") || g.find_tensor(b + "attn_sinks.weight")) {
            std::printf("  UNEXPECTED sink tensor at layer %u — sinks were disabled in "
                        "this build; the attention path would need them\n", L);
            ++fails;
        }
        const bool moe = g.find_tensor(b + "ffn_gate_exps.weight") != nullptr;
        if (moe) {
            ++n_moe;
            // 256 wide, NOT 256+shared: unlike Inkling, Laguna's shared expert
            // is NOT routed through the gate.  That difference decides how the
            // MoE block is wired, so it is asserted.
            need(b + "ffn_gate_inp.weight", {H, c.n_experts});
            need(b + "exp_probs_b.bias", {c.n_experts});
            need(b + "ffn_gate_exps.weight", {H, c.expert_ffn, c.n_experts});
            need(b + "ffn_up_exps.weight", {H, c.expert_ffn, c.n_experts});
            need(b + "ffn_down_exps.weight", {c.expert_ffn, H, c.n_experts});
            need(b + "ffn_gate_shexp.weight", {H, c.shared_expert_ffn});
            need(b + "ffn_up_shexp.weight", {H, c.shared_expert_ffn});
            need(b + "ffn_down_shexp.weight", {c.shared_expert_ffn, H});
        } else {
            ++n_dense;
            need(b + "ffn_gate.weight", {H, c.ffn_dense});
            need(b + "ffn_up.weight", {H, c.ffn_dense});
            need(b + "ffn_down.weight", {c.ffn_dense, H});
        }
    }
    std::printf("full-attn %u (48 heads) / sliding %u (72 heads); dense %u (config %u), MoE %u\n",
                n_full, n_slide, n_dense, c.dense_layers, n_moe);
    if (n_dense != c.dense_layers) { std::puts("  MISMATCH dense layer count"); ++fails; }
    if (n_full != 12 || n_slide != 36) { std::puts("  MISMATCH full/sliding split"); ++fails; }
    if (fails) { std::printf("laguna_manifest_test: FAILED (%d)\n", fails); return 1; }
    std::puts("laguna_manifest_test: OK — every tensor present, and the attention shapes "
              "track the PER-LAYER head count");
    return 0;
}
