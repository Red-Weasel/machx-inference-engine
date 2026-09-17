// tools/glm5next_config_test.cpp — GLM-5.3-Flash (`glm5next`) config gate (P0).
//
// Verifies that the engine reads the real 6-shard 185.98 GiB GGUF correctly
// BEFORE any forward exists: arch detection, every hparam, the per-layer
// attention map, and the per-block tensor schedule (KDA vs MLA vs dense vs
// MoE vs MTP — and the hyper-connection coverage, which SKIPS the MTP block).
//
// Every expectation below is checked against the file itself, not hard-coded —
// except a small set of published invariants (46 blocks, 288 experts, …)
// which ARE hard-coded on purpose so a wrong or swapped file fails loudly.
//
// usage: ie-glm5next-config-test <model-00001-of-00006.gguf>
#include "ie/gguf.hpp"
#include "ie/model_config.hpp"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using namespace ie;

namespace {

int g_fail = 0;

void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? "\033[32mOK\033[0m" : "\033[31mFAIL\033[0m", what);
    if (!ok) ++g_fail;
}

template <class T>
void check_eq(T got, T want, const char* what) {
    const bool ok = (got == want);
    std::printf("  [%s] %-46s got %-12lld want %lld\n", ok ? "\033[32mOK\033[0m" : "\033[31mFAIL\033[0m",
                what, (long long)got, (long long)want);
    if (!ok) ++g_fail;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]); return 2; }

    GgufReader g;
    if (auto e = g.open(argv[1]); !e.empty()) {
        std::fprintf(stderr, "open: %s\n", e.c_str());
        return 1;
    }
    std::printf("\n\033[1mGLM-5.3-Flash config gate\033[0m  %s\n", argv[1]);
    std::printf("  %" PRIu64 " tensors\n\n", (uint64_t)g.tensors().size());

    std::printf("\033[1m1. arch detection\033[0m\n");
    const auto arch = detect_arch(g);
    check(arch == ModelArch::kGlm5Next, "general.architecture == \"glm5next\" -> kGlm5Next");
    if (arch != ModelArch::kGlm5Next) return 1;

    Glm5NextConfig c;
    if (auto e = read_glm5next_config(g, c); !e.empty()) {
        std::fprintf(stderr, "read_glm5next_config: %s\n", e.c_str());
        return 1;
    }

    std::printf("\n\033[1m2. hyper-parameters\033[0m\n");
    check_eq<uint32_t>(c.n_layers, 46, "block_count");
    check_eq<uint32_t>(c.hidden, 4096, "embedding_length");
    check_eq<uint32_t>(c.n_q_heads, 64, "attention.head_count");
    check_eq<uint32_t>(c.ffn, 12288, "feed_forward_length (dense blocks)");
    check_eq<uint32_t>(c.vocab, 154880, "vocab (tokenizer.ggml.tokens)");
    check_eq<uint32_t>(c.ctx_train, 1048576, "context_length");
    check_eq<uint32_t>(c.q_lora_rank, 1536, "attention.q_lora_rank");
    check_eq<uint32_t>(c.kv_lora_rank, 512, "attention.kv_lora_rank");
    check_eq<uint32_t>(c.key_len, 512, "attention.key_length");
    check_eq<uint32_t>(c.value_len, 512, "attention.value_length");
    check_eq<uint32_t>(c.key_len_mla, 256, "attention.key_length_mla");
    check_eq<uint32_t>(c.value_len_mla, 256, "attention.value_length_mla");
    check_eq<uint32_t>(c.kda_head_dim, 128, "kda.head_dim");
    check(c.kda_gate_lower_bound == -5.0f, "kda.gate_lower_bound == -5.0");
    check_eq<uint32_t>(c.conv_kernel, 4, "ssm.conv_kernel");
    check_eq<uint32_t>(c.indexer_n_heads, 32, "indexer.head_count");
    check_eq<uint32_t>(c.indexer_head_dim, 128, "indexer.key_length");
    check_eq<uint32_t>(c.indexer_top_k, 2048, "indexer.top_k");
    check_eq<uint32_t>(c.indexer_kpool, 4, "indexer.kpool");
    check_eq<uint32_t>(c.hc_count, 4, "hyper_connection.count");
    check_eq<uint32_t>(c.hc_sinkhorn_iters, 20, "hyper_connection.sinkhorn_iterations");
    check_eq<uint32_t>(c.leading_dense, 3, "leading_dense_block_count");
    check_eq<uint32_t>(c.nextn_predict_layers, 1, "nextn_predict_layers");
    check_eq<uint32_t>(c.n_experts, 288, "expert_count");
    check_eq<uint32_t>(c.n_experts_used, 8, "expert_used_count");
    check_eq<uint32_t>(c.n_shared_experts, 1, "expert_shared_count");
    check_eq<uint32_t>(c.expert_ffn, 2048, "expert_feed_forward_length");
    check_eq<uint32_t>(c.shexp_ffn, 2048, "expert_shared_feed_forward_length");
    check_eq<uint32_t>(c.expert_gating_func, 2, "expert_gating_func (2 = sigmoid)");
    check(c.expert_weights_norm, "expert_weights_norm == true");
    check(c.expert_weights_scale == 2.5f, "expert_weights_scale == 2.5");
    check(c.rms_eps > 9e-6f && c.rms_eps < 1.1e-5f, "layer_norm_rms_epsilon ~ 1e-5");
    check(c.ln_eps > 9e-7f && c.ln_eps < 1.1e-6f, "layer_norm_epsilon ~ 1e-6 (indexer k)");

    std::printf("\n\033[1m3. per-layer attention map + derived identities\033[0m\n");
    check_eq<size_t>(c.attn_kind.size(), 46, "attention.head_count_kv array length");
    check_eq<uint32_t>(c.n_transformer_layers(), 45, "transformer blocks (excl. MTP)");
    // Published layout: full attention every 4th block (il%4==3) plus the MTP
    // block 45; everything else is a KDA linear layer.
    uint32_t full_n = 0, map_mismatch = 0;
    for (uint32_t il = 0; il < c.attn_kind.size(); ++il) {
        const bool want_full = (il % 4 == 3) || il == 45;
        if (c.is_full_attn(il)) ++full_n;
        if (c.is_full_attn(il) != want_full) ++map_mismatch;
    }
    check_eq<uint32_t>(full_n, 12, "full-attn layers (11 + MTP)");
    check_eq<uint32_t>(map_mismatch, 0, "map matches il%4==3 | il==45 pattern");
    // NOPE identity: no rope anywhere, so the cache key IS the MLA latent.
    check(c.key_len == c.kv_lora_rank, "key_length == kv_lora_rank (NOPE, no rope dims)");
    check_eq<uint32_t>(c.n_q_heads * c.kda_head_dim, 8192, "KDA inner width (64 x 128)");
    bool clamp_ok = c.swiglu_clamp_exp.size() == 46 && c.swiglu_clamp_shexp.size() == 46;
    for (uint32_t il = 0; clamp_ok && il < 46; ++il)
        clamp_ok = c.swiglu_clamp_exp[il] == 10.0f && c.swiglu_clamp_shexp[il] == 10.0f;
    check(clamp_ok, "swiglu_clamp_{exp,shexp}[46] all == 10.0");

    // -- 4. tensor schedule, read from the file's own tensor table -----------
    std::printf("\n\033[1m4. per-block tensor schedule\033[0m\n");
    std::map<uint32_t, std::vector<std::string>> by_block;
    uint64_t bytes_total = 0;
    for (const auto& t : g.tensors()) {
        bytes_total += t.nbytes;
        unsigned il = 0;
        if (std::sscanf(std::string(t.name).c_str(), "blk.%u.", &il) == 1) {
            std::string n(t.name);
            by_block[il].push_back(n.substr(n.find('.', 4) + 1));
        }
    }
    check_eq<size_t>(by_block.size(), 46, "blocks present in tensor table");

    uint32_t kda_seen = 0, mla_seen = 0, idx_seen = 0, dense_seen = 0,
             moe_seen = 0, shexp_seen = 0, hc_seen = 0, kind_mismatch = 0;
    for (const auto& [il, names] : by_block) {
        bool has_ssm = false, has_qa = false, has_idx = false,
             has_dense_ffn = false, has_exps = false, has_shexp = false, has_hc = false;
        for (const auto& n : names) {
            if (n == "ssm_beta.weight")         has_ssm = true;
            if (n == "attn_q_a.weight")         has_qa = true;
            if (n == "indexer.attn_k.weight")   has_idx = true;
            if (n == "ffn_gate.weight")         has_dense_ffn = true;
            if (n == "ffn_gate_exps.weight")    has_exps = true;
            if (n == "ffn_gate_shexp.weight")   has_shexp = true;
            if (n == "hc_attn_fn.weight")       has_hc = true;
        }
        kda_seen += has_ssm; mla_seen += has_qa; idx_seen += has_idx;
        dense_seen += has_dense_ffn; moe_seen += has_exps;
        shexp_seen += has_shexp; hc_seen += has_hc;
        // The config predicate must agree with the file, block by block:
        // full-attn blocks carry the MLA stack + indexer and NO ssm tensors.
        if (c.is_full_attn(il) != has_qa || c.is_full_attn(il) != has_idx ||
            c.is_full_attn(il) == has_ssm) {
            ++kind_mismatch;
            std::printf("      block %2u: is_full_attn=%d but qa=%d idx=%d ssm=%d\n",
                        il, int(c.is_full_attn(il)), int(has_qa), int(has_idx), int(has_ssm));
        }
    }
    check_eq<uint32_t>(kda_seen, 34, "blocks with KDA (ssm_*) tensors");
    check_eq<uint32_t>(mla_seen, 12, "blocks with the MLA lora stack");
    check_eq<uint32_t>(idx_seen, 12, "blocks with indexer tensors");
    check_eq<uint32_t>(kind_mismatch, 0, "is_full_attn() predicate matches the file");
    check_eq<uint32_t>(dense_seen, 3, "blocks with a dense FFN");
    check_eq<uint32_t>(moe_seen, 43, "blocks with a routed-expert bank");
    check_eq<uint32_t>(shexp_seen, 43, "blocks with a shared expert");
    // Hyper-connections cover every TRANSFORMER block but NOT the MTP block —
    // the loader must not demand hc_* on blk.45.
    check_eq<uint32_t>(hc_seen, 45, "blocks with hyper-connection tensors (no MTP)");

    // MTP block 45: full attention + full expert bank + the nextn projections.
    {
        const auto& n45 = by_block[45];
        bool eh = false, en = false, hn = false, shn = false, exps = false, attn = false;
        for (const auto& n : n45) {
            if (n == "nextn.eh_proj.weight")          eh = true;
            if (n == "nextn.enorm.weight")            en = true;
            if (n == "nextn.hnorm.weight")            hn = true;
            if (n == "nextn.shared_head_norm.weight") shn = true;
            if (n == "ffn_gate_exps.weight")          exps = true;
            if (n == "attn_q_b.weight")               attn = true;
        }
        check(eh && en && hn && shn, "MTP block 45 carries nextn.{eh_proj,enorm,hnorm,shared_head_norm}");
        check(exps && attn, "MTP block 45 carries its OWN expert bank + MLA stack");
    }
    // The MTP head reuses the main embedding and lm_head.
    {
        bool emb_tok = false, shared_head = false;
        for (const auto& t : g.tensors()) {
            if (t.name.find("nextn.embed_tokens") != std::string_view::npos) emb_tok = true;
            if (t.name.find("nextn.shared_head_head") != std::string_view::npos) shared_head = true;
        }
        check(!emb_tok && !shared_head,
              "nextn.{embed_tokens,shared_head_head} ABSENT -> MTP reuses token_embd/output");
    }
    // Indexer numerics contract from the reference: k_norm is a LayerNorm WITH
    // bias, and the tiny per-head proj ships in fp32 (bf16 flips near-tie
    // rankings). Check both straight off the tensor table.
    {
        const auto* knb = g.find_tensor("blk.3.indexer.k_norm.bias");
        const auto* prj = g.find_tensor("blk.3.indexer.proj.weight");
        check(knb != nullptr, "indexer.k_norm.bias present (LayerNorm WITH bias)");
        check(prj && prj->dtype == DType::kF32, "indexer.proj.weight is F32");
    }

    // -- 5. size cross-check --------------------------------------------------
    std::printf("\n\033[1m5. size cross-check\033[0m\n");
    const double gib = bytes_total / 1073741824.0;
    std::printf("      total %.3f GiB (UD-Q4_K_XL, dynamic per-layer quants)\n", gib);
    check(gib > 185.8 && gib < 186.2, "total size within 185.8-186.2 GiB");

    std::printf("\n\033[1m%s\033[0m  (%d failure%s)\n\n",
                g_fail ? "\033[31mGATE FAILED\033[0m" : "\033[32mGATE PASSED\033[0m",
                g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
