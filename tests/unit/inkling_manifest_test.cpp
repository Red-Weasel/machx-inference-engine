// P1 gate: does the REAL Inkling-Small GGUF carry exactly the tensors the port
// plan assumes, with the shapes the config implies?  Opens shard 1 (the reader
// pulls in its siblings) and reads HEADERS ONLY — no weight bytes, so this is
// seconds, not the 163 GB the model actually is.
//
// WHY THIS EXISTS BEFORE THE LOADER.  Every surprise found here is a surprise
// the loader would otherwise hit halfway through a multi-GB upload.  The
// deepseek4 port learned that the expensive way; this is the cheap way.
// SKIPs cleanly when the Expansion drive is unmounted.
#undef NDEBUG
#include "ie/gguf.hpp"
#include "ie/model_config.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace {
// Model path comes from $IE_INKLING_GGUF — these gates need the real Inkling-Small
// GGUF, which is far too large to ship. Unset (or unreachable) means the
// test SKIPS cleanly, so a checkout without the model still goes green.
const char* shard1_path() { return std::getenv("IE_INKLING_GGUF"); }

// GGUF stores dims fastest-varying FIRST.  shape[0] is the ROW LENGTH.
std::string dims_str(const ie::GgufTensorInfo& t) {
    std::string s = "[";
    for (uint32_t i = 0; i < t.n_dims; ++i)
        s += (i ? ", " : "") + std::to_string(t.shape[i]);
    return s + "]";
}
}  // namespace

int main() {
    const char* path = shard1_path();
    if (!path || !*path) {
        std::puts("inkling_manifest_test: SKIPPED (set IE_INKLING_GGUF to the model's shard-1 path)");
        return 0;
    }
    ie::GgufReader g;
    if (!g.open(path).empty()) {
        std::puts("inkling_manifest_test: SKIPPED (model not reachable)");
        return 0;
    }
    ie::InklingConfig c;
    const std::string e = ie::read_inkling_config(g, c);
    if (!e.empty()) { std::printf("FAIL: config: %s\n", e.c_str()); return 1; }

    int fails = 0;
    auto need = [&](const std::string& name, std::vector<uint64_t> want) {
        const auto* t = g.find_tensor(name);
        if (!t) { std::printf("  MISSING %s\n", name.c_str()); ++fails; return; }
        bool ok = t->n_dims == want.size();
        for (size_t i = 0; ok && i < want.size(); ++i) ok = t->shape[i] == want[i];
        if (!ok) {
            std::printf("  SHAPE   %-34s got %s want [", name.c_str(), dims_str(*t).c_str());
            for (size_t i = 0; i < want.size(); ++i)
                std::printf("%s%llu", i ? ", " : "", (unsigned long long)want[i]);
            std::puts("]");
            ++fails;
        }
    };

    const uint32_t H = c.hidden, HD = c.head_dim, NH = c.n_q_heads;
    std::printf("layers=%u H=%u heads=%u head_dim=%u experts=%u top-%u shared=%u\n",
                c.n_layers, H, NH, HD, c.n_experts, c.n_experts_used, c.n_shared_experts);

    // Non-layer tensors.  `token_embd_norm` is NOT universal across arches —
    // its presence here is part of the contract, so it is asserted, not assumed.
    need("token_embd.weight", {H, c.vocab});
    need("token_embd_norm.weight", {H});
    need("output.weight", {H, c.vocab});
    need("output_norm.weight", {H});

    // Per-layer.  Two shapes of layer: the leading `dense_layers` carry a plain
    // MLP, the rest carry the MoE bank — the port plan's claim, checked.
    uint32_t n_dense = 0, n_moe = 0;
    for (uint32_t L = 0; L < c.n_layers; ++L) {
        const std::string b = "blk." + std::to_string(L) + ".";
        const uint32_t kv_dim = uint32_t(c.n_kv_heads[L]) * HD;
        need(b + "attn_norm.weight", {H});
        need(b + "attn_q.weight", {H, uint64_t(NH) * HD});
        need(b + "attn_k.weight", {H, kv_dim});
        need(b + "attn_v.weight", {H, kv_dim});
        need(b + "attn_output.weight", {uint64_t(NH) * HD, H});
        need(b + "attn_q_norm.weight", {HD});
        need(b + "attn_k_norm.weight", {HD});
        need(b + "ffn_norm.weight", {H});
        // The novel pieces this session built ops for.
        need(b + "attn_r.weight", {H, uint64_t(NH) * c.d_rel});
        need(b + "attn_rel_proj.weight", {c.rel_extent_for(L), c.d_rel});
        // [K, dim] as GGUF prints it = ne[0]=K taps (row length), ne[1]=dim
        // channels, i.e. CHANNEL-MAJOR with a channel's taps contiguous.  This
        // is the assertion that caught `ink_shortconv` indexing [K][C].
        need(b + "shortconv_attn.weight", {c.shortconv_kernel, uint64_t(NH) * HD});
        need(b + "shortconv_k.weight", {c.shortconv_kernel, kv_dim});
        need(b + "shortconv_v.weight", {c.shortconv_kernel, kv_dim});
        need(b + "shortconv_mlp.weight", {c.shortconv_kernel, H});
        const bool is_moe = g.find_tensor(b + "ffn_gate_exps.weight") != nullptr;
        if (is_moe) {
            ++n_moe;
            need(b + "ffn_gate_inp.weight", {H, uint64_t(c.n_experts) + c.n_shared_experts});
            need(b + "exp_probs_b.bias", {c.n_experts});
            need(b + "ffn_gscale.weight", {1});
            need(b + "ffn_gate_exps.weight", {H, c.expert_ffn, c.n_experts});
            need(b + "ffn_up_exps.weight", {H, c.expert_ffn, c.n_experts});
            need(b + "ffn_down_exps.weight", {c.expert_ffn, H, c.n_experts});
            need(b + "ffn_gate_shexp.weight", {H, c.expert_ffn, c.n_shared_experts});
            need(b + "ffn_up_shexp.weight", {H, c.expert_ffn, c.n_shared_experts});
            need(b + "ffn_down_shexp.weight", {c.expert_ffn, H, c.n_shared_experts});
        } else {
            ++n_dense;
            need(b + "ffn_gate.weight", {H, c.ffn_dense});
            need(b + "ffn_up.weight", {H, c.ffn_dense});
            need(b + "ffn_down.weight", {c.ffn_dense, H});
        }
    }
    std::printf("dense layers %u (config dense_block_count %u), MoE layers %u\n",
                n_dense, c.dense_layers, n_moe);
    if (n_dense != c.dense_layers) {
        std::printf("  MISMATCH: %u dense layers found, config says %u\n", n_dense, c.dense_layers);
        ++fails;
    }
    if (n_dense + n_moe != c.n_layers) { std::puts("  MISMATCH: layer count"); ++fails; }


    if (fails) { std::printf("inkling_manifest_test: FAILED (%d)\n", fails); return 1; }
    std::puts("inkling_manifest_test: OK — every tensor the port plan assumes is present "
              "with the config-implied shape");
    return 0;
}
