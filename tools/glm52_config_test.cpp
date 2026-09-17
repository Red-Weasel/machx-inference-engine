// tools/glm52_config_test.cpp — GLM-5.2 (`glm-dsa`) config + tensor-layout gate.
//
// Verifies that the engine reads the real 196.77 GiB GGUF correctly BEFORE any
// forward exists: arch detection, every hparam, the derived MLA identities, the
// IQ2_KT per-row scale accounting, and the per-block tensor schedule (which
// blocks are dense / MoE / MTP / carry a DSA indexer).
//
// Every expectation below is checked against the file itself, not hard-coded —
// except a small set of published invariants (block_count 79, 256 experts, …)
// which ARE hard-coded on purpose so a wrong or swapped file fails loudly.
//
// usage: ie-glm52-config-test <model.gguf>
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
    std::printf("\n\033[1mGLM-5.2 config gate\033[0m  %s\n", argv[1]);
    std::printf("  %" PRIu64 " tensors\n\n", (uint64_t)g.tensors().size());

    std::printf("\033[1m1. arch detection\033[0m\n");
    const auto arch = detect_arch(g);
    check(arch == ModelArch::kGlmDsa, "general.architecture == \"glm-dsa\" -> kGlmDsa");
    if (arch != ModelArch::kGlmDsa) return 1;

    GlmDsaConfig c;
    if (auto e = read_glmdsa_config(g, c); !e.empty()) {
        std::fprintf(stderr, "read_glmdsa_config: %s\n", e.c_str());
        return 1;
    }

    std::printf("\n\033[1m2. hyper-parameters\033[0m\n");
    check_eq<uint32_t>(c.n_layers, 79, "block_count");
    check_eq<uint32_t>(c.hidden, 6144, "embedding_length");
    check_eq<uint32_t>(c.n_q_heads, 64, "attention.head_count");
    check_eq<uint32_t>(c.n_kv_heads, 1, "attention.head_count_kv (MLA latent)");
    check_eq<uint32_t>(c.head_dim, 576, "attention.key_length");
    check_eq<uint32_t>(c.v_head_dim, 512, "attention.value_length");
    check_eq<uint32_t>(c.q_lora_rank, 2048, "attention.q_lora_rank");
    check_eq<uint32_t>(c.kv_lora_rank, 512, "attention.kv_lora_rank");
    check_eq<uint32_t>(c.key_len_mla, 256, "attention.key_length_mla");
    check_eq<uint32_t>(c.value_len_mla, 256, "attention.value_length_mla");
    check_eq<uint32_t>(c.rope_dim, 64, "rope.dimension_count");
    check_eq<uint32_t>(c.ffn, 12288, "feed_forward_length (dense blocks)");
    check_eq<uint32_t>(c.vocab, 154880, "vocab");
    check_eq<uint32_t>(c.ctx_train, 1048576, "context_length");
    check_eq<uint32_t>(c.leading_dense, 3, "leading_dense_block_count");
    check_eq<uint32_t>(c.nextn_predict_layers, 1, "nextn_predict_layers");
    check_eq<uint32_t>(c.n_experts, 256, "expert_count");
    check_eq<uint32_t>(c.n_experts_used, 8, "expert_used_count");
    check_eq<uint32_t>(c.n_shared_experts, 1, "expert_shared_count");
    check_eq<uint32_t>(c.expert_ffn, 2048, "expert_feed_forward_length");
    check_eq<uint32_t>(c.expert_gating_func, 2, "expert_gating_func (2 = sigmoid)");
    check_eq<uint32_t>(c.expert_group_count, 1, "expert_group_count (flat top-k)");
    check(c.expert_weights_norm, "expert_weights_norm == true");
    check(c.expert_weights_scale == 2.5f, "expert_weights_scale == 2.5");
    check(c.rope_theta == 8e6f, "rope.freq_base == 8e6");
    check_eq<uint32_t>(c.indexer_n_heads, 32, "indexer.head_count");
    check_eq<uint32_t>(c.indexer_head_dim, 128, "indexer.key_length");
    check_eq<uint32_t>(c.indexer_top_k, 2048, "indexer.top_k");

    std::printf("\n\033[1m3. derived MLA identities\033[0m\n");
    check(c.head_dim == c.kv_lora_rank + c.rope_dim,
          "key_length == kv_lora_rank + rope_dim (576 = 512 + 64)");
    check(c.key_len_mla == 192 + c.rope_dim,
          "key_length_mla == qk_nope(192) + qk_rope(64)");
    check_eq<uint32_t>(c.n_transformer_layers(), 78, "transformer blocks (excl. MTP)");

    // -- 4. tensor schedule, read from the file's own tensor table -----------
    std::printf("\n\033[1m4. per-block tensor schedule\033[0m\n");
    std::map<uint32_t, std::vector<std::string>> by_block;
    size_t n_iq2kt = 0, n_q6_0 = 0;
    uint64_t bytes_iq2kt = 0, bytes_q6_0 = 0, bytes_total = 0;
    for (const auto& t : g.tensors()) {
        bytes_total += t.nbytes;
        if (t.dtype == DType::kIQ2_KT) { ++n_iq2kt; bytes_iq2kt += t.nbytes; }
        if (t.dtype == DType::kQ6_0)   { ++n_q6_0;  bytes_q6_0  += t.nbytes; }
        unsigned il = 0;
        if (std::sscanf(std::string(t.name).c_str(), "blk.%u.", &il) == 1) {
            std::string n(t.name);
            by_block[il].push_back(n.substr(n.find('.', 4) + 1));
        }
    }
    check_eq<size_t>(by_block.size(), 79, "blocks present in tensor table");

    uint32_t dense_seen = 0, moe_seen = 0, idx_seen = 0, idx_mismatch = 0;
    for (const auto& [il, names] : by_block) {
        bool has_exps = false, has_dense_ffn = false, has_idx = false;
        for (const auto& n : names) {
            if (n == "ffn_gate_exps.weight") has_exps = true;
            if (n == "ffn_gate.weight")      has_dense_ffn = true;
            if (n == "indexer.attn_k.weight") has_idx = true;
        }
        if (has_dense_ffn) ++dense_seen;
        if (has_exps)      ++moe_seen;
        if (has_idx)       ++idx_seen;
        // The predicate in GlmDsaConfig must agree with the file, block by block.
        if (has_idx != c.has_indexer(il)) {
            ++idx_mismatch;
            std::printf("      block %2u: file has_indexer=%d, predicate=%d\n",
                        il, int(has_idx), int(c.has_indexer(il)));
        }
    }
    check_eq<uint32_t>(dense_seen, 3, "blocks with a dense FFN");
    check_eq<uint32_t>(moe_seen, 76, "blocks with a routed-expert bank");
    check_eq<uint32_t>(idx_seen, 22, "blocks with DSA indexer tensors");
    check_eq<uint32_t>(idx_mismatch, 0, "has_indexer() predicate matches the file");

    // MTP block: full attention + full expert bank + the nextn projections.
    {
        const auto& n78 = by_block[78];
        bool eh = false, en = false, hn = false, shn = false, exps = false, attn = false;
        for (const auto& n : n78) {
            if (n == "nextn.eh_proj.weight")          eh = true;
            if (n == "nextn.enorm.weight")            en = true;
            if (n == "nextn.hnorm.weight")            hn = true;
            if (n == "nextn.shared_head_norm.weight") shn = true;
            if (n == "ffn_gate_exps.weight")          exps = true;
            if (n == "attn_q_b.weight")               attn = true;
        }
        check(eh && en && hn && shn, "MTP block 78 carries nextn.{eh_proj,enorm,hnorm,shared_head_norm}");
        check(exps && attn, "MTP block 78 carries its OWN expert bank + MLA stack");
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

    // -- 5. quantisation accounting -----------------------------------------
    std::printf("\n\033[1m5. quantisation + IQ2_KT row-scale accounting\033[0m\n");
    check_eq<size_t>(n_iq2kt, 228, "IQ2_KT tensors (76 blocks x 3 expert mats)");
    check_eq<size_t>(n_q6_0, 736, "Q6_0 tensors");
    std::printf("      IQ2_KT %.3f GiB | Q6_0 %.3f GiB | total %.3f GiB\n",
                bytes_iq2kt / 1073741824.0, bytes_q6_0 / 1073741824.0,
                bytes_total / 1073741824.0);

    // IQ2_KT carries a 4-byte fp32 scale PER ROW on top of the 68-B blocks.
    // Recompute the expected surplus independently and require the loader to
    // have charged exactly that much — this is what makes our total match the
    // model card's 196.756 GiB instead of undercounting by 0.74 GiB.
    uint64_t rows = 0, blocks_only = 0;
    for (const auto& t : g.tensors()) {
        if (t.dtype != DType::kIQ2_KT) continue;
        uint64_t hi = 1;
        for (uint32_t d = 1; d < t.n_dims; ++d) hi *= t.shape[d];
        rows += hi;
        blocks_only += (t.shape[0] / 256) * 68 * hi;
    }
    check_eq<uint64_t>(bytes_iq2kt - blocks_only, rows * 4,
                       "IQ2_KT surplus over block bytes == 4 B per row");
    std::printf("      %" PRIu64 " expert rows x 4 B = %.3f GiB of row scales\n",
                rows, rows * 4 / 1073741824.0);

    // Published size, as a whole-file cross-check.
    const double gib = bytes_total / 1073741824.0;
    check(gib > 196.6 && gib < 196.9, "total size within 196.6-196.9 GiB (card says 196.756)");

    std::printf("\n\033[1m%s\033[0m  (%d failure%s)\n\n",
                g_fail ? "\033[31mGATE FAILED\033[0m" : "\033[32mGATE PASSED\033[0m",
                g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
