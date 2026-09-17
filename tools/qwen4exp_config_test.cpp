// tools/qwen4exp_config_test.cpp — Qwen3.8-Flash-Next (`qwen4exp`) config +
// tensor-layout gate (campaign P0, docs/qwen4-flash-next-port-campaign.md).
//
// Verifies the engine reads the real 4-shard UD-Q4_K_XL GGUF correctly BEFORE
// any forward exists: arch detection, every hparam, the derived hybrid-layer
// schedule, and an EXACT tensor-map equality check — every one of the file's
// tensors must match a config-derived (name, shape) expectation and every
// expectation must be present in the file, in both directions. That is
// stronger than the glm52 gate's per-pattern counting: a single mis-shaped,
// missing, or extra tensor fails.
//
// Published invariants (48 blocks, 512 experts, ...) are hard-coded on purpose
// so a wrong or swapped file fails loudly. Quant dtypes are NOT pinned: the
// UD ("unsloth dynamic") scheme varies dtype per layer, and other quants of
// the same arch must pass this gate too — the dtype histogram is printed for
// dequant-coverage planning instead.
//
// usage: ie-qwen4exp-config-test <model-00001-of-0000N.gguf>
#include "ie/dtype.hpp"
#include "ie/gguf.hpp"
#include "ie/model_config.hpp"

#include <cinttypes>
#include <cstdio>
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
    std::printf("\n\033[1mQwen3.8-Flash-Next config gate\033[0m  %s\n", argv[1]);
    std::printf("  %" PRIu64 " tensors\n\n", (uint64_t)g.tensors().size());

    std::printf("\033[1m1. arch detection\033[0m\n");
    const auto arch = detect_arch(g);
    check(arch == ModelArch::kQwen4Exp, "general.architecture == \"qwen4exp\" -> kQwen4Exp");
    if (arch != ModelArch::kQwen4Exp) return 1;

    Qwen4ExpConfig c;
    if (auto e = read_qwen4exp_config(g, c); !e.empty()) {
        std::fprintf(stderr, "read_qwen4exp_config: %s\n", e.c_str());
        return 1;
    }

    std::printf("\n\033[1m2. hyper-parameters\033[0m\n");
    check_eq<uint32_t>(c.n_layers, 48, "block_count");
    check_eq<uint32_t>(c.hidden, 2560, "embedding_length");
    check_eq<uint32_t>(c.vocab, 248320, "vocab (tokenizer.ggml.tokens)");
    check_eq<uint32_t>(c.ctx_train, 262144, "context_length");
    check_eq<uint32_t>(c.n_q_heads, 24, "attention.head_count");
    check_eq<uint32_t>(c.n_kv_heads, 2, "attention.head_count_kv");
    check_eq<uint32_t>(c.head_dim, 256, "attention.key_length (== value_length)");
    check_eq<uint32_t>(c.full_attention_interval, 4, "full_attention_interval");
    check_eq<uint32_t>(c.rope_dim, 64, "rope.dimension_count (partial 0.25)");
    check(c.rope_theta == 1e7f, "rope.freq_base == 1e7");
    check(c.rope_sections == std::vector<int32_t>({11, 11, 10, 0}),
          "rope.dimension_sections == [11,11,10,0] (interleaved mrope)");
    check_eq<uint32_t>(c.indexer_n_heads, 4, "indexer.head_count");
    check_eq<uint32_t>(c.indexer_head_dim, 128, "indexer.key_length");
    check_eq<uint32_t>(c.indexer_top_k, 2048, "indexer.top_k");
    check_eq<size_t>(c.compress_ratios.size(), 48, "attention.compress_ratios entries");
    check_eq<uint32_t>(c.ssm_state, 128, "ssm.state_size (DeltaNet head_dim)");
    check_eq<uint32_t>(c.ssm_conv_kernel, 4, "ssm.conv_kernel");
    check_eq<uint32_t>(c.ssm_k_heads, 16, "ssm.group_count (K heads)");
    check_eq<uint32_t>(c.ssm_v_heads, 48, "ssm.time_step_rank (V heads)");
    check_eq<uint32_t>(c.ssm_inner, 6144, "ssm.inner_size");
    check_eq<uint32_t>(c.hc_count, 4, "hyper_connection.count");
    check_eq<uint32_t>(c.hc_low_rank, 320, "hyper_connection.low_rank");
    check_eq<uint32_t>(c.n_experts, 512, "expert_count");
    check_eq<uint32_t>(c.n_experts_used, 10, "expert_used_count");
    check_eq<uint32_t>(c.expert_ffn, 640, "expert_feed_forward_length");
    check_eq<uint32_t>(c.shared_expert_ffn, 640, "expert_shared_feed_forward_length");
    check(c.ple_layers == std::vector<int32_t>({1}),
          "ple.layers == [1] (0-based; HF ple_layer_ids [2] is 1-based)");
    check_eq<uint32_t>(c.ple_ngram, 3, "ple.ngram_size");
    check_eq<uint32_t>(c.ple_heads_per_ngram, 8, "ple.heads_per_ngram");
    check_eq<uint32_t>(c.ple_conv_kernel, 4, "ple.conv_kernel");
    check_eq<uint32_t>(c.ple_embed_per_input, 160, "embedding_length_per_layer_input");
    check(c.rms_eps == 1e-6f, "attention.layer_norm_rms_epsilon == 1e-6");

    std::printf("\n\033[1m3. derived identities\033[0m\n");
    check_eq<uint32_t>(c.n_full_attn_layers(), 12, "full-attention blocks (il %% 4 == 3)");
    check_eq<uint32_t>(c.n_linear_layers(), 36, "Gated-DeltaNet blocks");
    check(c.is_full_attn(3) && c.is_full_attn(47) && !c.is_full_attn(0) && !c.is_full_attn(46),
          "is_full_attn() spot checks (3, 47 yes; 0, 46 no)");
    check_eq<uint32_t>(c.ple_embed_dim(), 2560,
                       "ple_embed_dim = 160 x 8 heads x 2 tables");
    check_eq<uint64_t>(c.ple_total_vocab(), 320001446ull,
                       "ple total hash rows (16 prime vocabs)");
    check(c.ssm_inner == c.ssm_v_heads * c.ssm_state,
          "ssm.inner_size == V-heads * state (6144 = 48*128)");
    // DeltaNet dims identical to Qwen3.8-27B -> qwen35 kernel reuse holds.
    check(c.ssm_k_heads == 16 && c.ssm_v_heads == 48 && c.ssm_state == 128 && c.ssm_conv_kernel == 4,
          "DeltaNet geometry identical to Qwen3.8-27B (kernel reuse)");

    // -- 4. exact tensor map: config-derived expectations, both directions ---
    std::printf("\n\033[1m4. exact tensor-map equality\033[0m\n");
    const uint64_t hc4  = uint64_t(c.hc_count) * c.hidden;              // 10240
    const uint64_t qkvw = 2ull * c.ssm_k_heads * c.ssm_state + c.ssm_inner;  // 10240
    std::map<std::string, std::vector<uint64_t>> want;
    want["token_embd.weight"]     = {c.hidden, c.vocab};
    want["output.weight"]         = {c.hidden, c.vocab};
    want["output_hc_up.weight"]   = {c.hc_low_rank, hc4};
    want["output_hc_down.weight"] = {hc4, c.hc_low_rank};
    want["output_hc_norm.weight"] = {hc4};
    // per_layer_token_embd is special-cased below (row padding).
    for (uint32_t il = 0; il < c.n_layers; ++il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        for (const char* side : {"hc_attn_", "hc_ffn_"}) {
            want[p + side + "up.weight"]     = {c.hc_low_rank, hc4};
            want[p + side + "down.weight"]   = {hc4, c.hc_low_rank};
            want[p + side + "norm.weight"]   = {hc4};
            want[p + side + "inject.weight"] = {hc4, c.hc_count};
        }
        want[p + "ffn_gate_inp.weight"]       = {c.hidden, c.n_experts};
        want[p + "ffn_gate_exps.weight"]      = {c.hidden, c.expert_ffn, c.n_experts};
        want[p + "ffn_up_exps.weight"]        = {c.hidden, c.expert_ffn, c.n_experts};
        want[p + "ffn_down_exps.weight"]      = {c.expert_ffn, c.hidden, c.n_experts};
        want[p + "ffn_gate_shexp.weight"]     = {c.hidden, c.shared_expert_ffn};
        want[p + "ffn_up_shexp.weight"]       = {c.hidden, c.shared_expert_ffn};
        want[p + "ffn_down_shexp.weight"]     = {c.shared_expert_ffn, c.hidden};
        want[p + "ffn_gate_inp_shexp.weight"] = {c.hidden};
        if (c.is_full_attn(il)) {
            want[p + "attn_q.weight"]      = {c.hidden, uint64_t(c.n_q_heads) * c.head_dim * 2};  // gate fused
            want[p + "attn_k.weight"]      = {c.hidden, uint64_t(c.n_kv_heads) * c.head_dim};
            want[p + "attn_v.weight"]      = {c.hidden, uint64_t(c.n_kv_heads) * c.head_dim};
            want[p + "attn_output.weight"] = {uint64_t(c.n_q_heads) * c.head_dim, c.hidden};
            want[p + "attn_q_norm.weight"] = {c.head_dim};
            want[p + "attn_k_norm.weight"] = {c.head_dim};
            want[p + "indexer.q_proj.weight"] = {c.hidden, uint64_t(c.indexer_n_heads) * c.indexer_head_dim};
            want[p + "indexer.k_proj.weight"] = {c.hidden, c.indexer_head_dim};
            want[p + "indexer.q_norm.weight"] = {c.indexer_head_dim};
            want[p + "indexer.k_norm.weight"] = {c.indexer_head_dim};
        } else {
            want[p + "attn_qkv.weight"]  = {c.hidden, qkvw};
            want[p + "attn_gate.weight"] = {c.hidden, c.ssm_inner};
            want[p + "ssm_alpha.weight"] = {c.hidden, c.ssm_v_heads};
            want[p + "ssm_beta.weight"]  = {c.hidden, c.ssm_v_heads};
            want[p + "ssm_a"]            = {c.ssm_v_heads};   // no .weight suffix in the file
            want[p + "ssm_dt.bias"]      = {c.ssm_v_heads};
            want[p + "ssm_conv1d.weight"] = {c.ssm_conv_kernel, qkvw};
            want[p + "ssm_norm.weight"]  = {c.ssm_state};
            want[p + "ssm_out.weight"]   = {c.ssm_inner, c.hidden};
        }
        if (c.is_ple_layer(il)) {
            // Shapes per the HF reference (Qwen4ExpTextPLELayer): key_proj is
            // ple_embed_dim -> hc_count*hidden (per-stream keys), value_proj
            // is ple_embed_dim -> hidden; conv + all three norms live in the
            // hc_count*hidden space.
            want[p + "ple_key.weight"]        = {c.ple_embed_dim(), hc4};
            want[p + "ple_value.weight"]      = {c.ple_embed_dim(), c.hidden};
            want[p + "ple_conv1d.weight"]     = {c.ple_conv_kernel, hc4};
            want[p + "ple_norm_query.weight"] = {hc4};
            want[p + "ple_norm_key.weight"]   = {hc4};
            want[p + "ple_norm_conv.weight"]  = {hc4};
        }
    }

    size_t unexpected = 0, mismatched = 0, mtp_tensors = 0;
    uint64_t bytes_total = 0;
    std::map<DType, std::pair<size_t, uint64_t>> by_dtype;
    bool ple_table_seen = false, ple_table_ok = false;
    uint64_t ple_table_width = 0;
    for (const auto& t : g.tensors()) {
        bytes_total += t.nbytes;
        auto& slot = by_dtype[t.dtype];
        ++slot.first;
        slot.second += t.nbytes;
        const std::string name(t.name);
        if (name.find("nextn") != std::string::npos || name.find("mtp") != std::string::npos)
            ++mtp_tensors;
        if (name == "per_layer_token_embd.weight") {
            // Row count is ple_total_vocab padded up to a multiple of 128
            // (HF make_ngram_vocab_size_divisible_by; no GGUF key).
            ple_table_seen  = true;
            ple_table_width = t.n_dims == 2 ? t.shape[1] : 0;
            const uint64_t tot = c.ple_total_vocab();
            ple_table_ok = t.n_dims == 2 && t.shape[0] == c.ple_embed_per_input &&
                           ple_table_width >= tot && ple_table_width - tot < 128 &&
                           ple_table_width % 128 == 0;
            continue;
        }
        auto it = want.find(name);
        if (it == want.end()) {
            if (++unexpected <= 10) std::printf("      unexpected tensor: %s\n", name.c_str());
            continue;
        }
        const auto& exp = it->second;
        bool ok = t.n_dims == exp.size();
        for (size_t d = 0; ok && d < exp.size(); ++d) ok = t.shape[d] == exp[d];
        if (!ok) {
            if (++mismatched <= 10) {
                std::printf("      shape mismatch: %s got [", name.c_str());
                for (uint32_t d = 0; d < t.n_dims; ++d)
                    std::printf("%s%" PRIu64, d ? ", " : "", t.shape[d]);
                std::printf("] want [");
                for (size_t d = 0; d < exp.size(); ++d)
                    std::printf("%s%" PRIu64, d ? ", " : "", exp[d]);
                std::printf("]\n");
            }
        }
        want.erase(it);
    }
    check_eq<size_t>(unexpected, 0, "tensors with no config-derived expectation");
    check_eq<size_t>(mismatched, 0, "tensors with a mismatched shape");
    {
        size_t shown = 0;
        for (const auto& [name, shape] : want) {
            if (++shown > 10) break;
            std::printf("      missing tensor: %s\n", name.c_str());
        }
    }
    check_eq<size_t>(want.size(), 0, "expected tensors missing from the file");
    check(ple_table_seen, "per_layer_token_embd.weight present (has_ple)");
    check(ple_table_ok, "PLE table [160, align128(320001446) = 320001536]");

    std::printf("\n\033[1m5. schedule + counts + MTP\033[0m\n");
    check_eq<size_t>(g.tensors().size(), 1224, "total tensors (36x25 + 12x26 + 6 PLE + 6 global)");
    if (const auto* kv = g.find_kv("split.tensors.count"))
        check_eq<uint64_t>((uint64_t)g.tensors().size(), kv->as_uint(),
                           "tensors resolved == split.tensors.count");
    // The convert script sets no_mtp=True: MTP weights are never exported, so
    // native-MTP spec decode CANNOT come from this GGUF (campaign P0 finding).
    check_eq<size_t>(mtp_tensors, 0, "nextn/mtp tensors (converter no_mtp=True)");

    std::printf("\n\033[1m6. size + dtype coverage (informational)\033[0m\n");
    const double gib = bytes_total / 1073741824.0;
    check(gib > 103.0 && gib < 104.5, "total size within 103.0-104.5 GiB (card: 111 GB)");
    std::printf("      total %.3f GiB across %zu dtypes:\n", gib, by_dtype.size());
    for (const auto& [dt, cnt] : by_dtype)
        std::printf("        %-8s %4zu tensors  %9.3f GiB\n",
                    std::string(type_name(dt)).c_str(), cnt.first,
                    cnt.second / 1073741824.0);

    std::printf("\n\033[1m%s\033[0m  (%d failure%s)\n\n",
                g_fail ? "\033[31mGATE FAILED\033[0m" : "\033[32mGATE PASSED\033[0m",
                g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
