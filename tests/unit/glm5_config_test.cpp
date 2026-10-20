// read_glm5next_config under both arch names. Unsloth's 2026-10-05 metadata rewrite of
// GLM-5.3-Flash-GGUF renamed `glm5next` (and every `glm5next.` key) to llama.cpp's `glm5-next`;
// the engine refused the re-downloaded file ("unsupported architecture: glm5-next") until the
// reader followed the file's own key prefix. Synthetic files carry the real file's values;
// $IE_GLM53_GGUF (the shard-1 path) also checks the real metadata, SKIPPED when unset.
#undef NDEBUG
#include "ie/gguf_writer.hpp"
#include "ie/gguf.hpp"
#include "ie/model_config.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unistd.h>
#include <vector>

namespace {
// GLM-5.3-Flash UD-Q4_K_XL shard-1 values (2026-10-19 dump); every key keyed off `arch`.
std::string write_glm(const std::string& path, const std::string& arch, const std::string& key_prefix,
                      uint32_t vocab_size = 0) {   // 0: no vocab_size key (the reader then trusts the tokenizer)
    const std::string P = key_prefix + ".";
    std::vector<int32_t> kinds(46, 0);
    for (int i = 0; i < 46; ++i) kinds[i] = (i % 4 == 3) ? 1 : 0;   // a hybrid map; the reader only sizes it
    ie::GgufWriter w;
    w.kv_string("general.architecture", arch);
    w.kv_u32(P + "block_count", 46);
    w.kv_u32(P + "context_length", 1048576);
    w.kv_u32(P + "embedding_length", 4096);
    w.kv_u32(P + "feed_forward_length", 12288);
    w.kv_u32(P + "attention.head_count", 64);
    w.kv_f32(P + "attention.layer_norm_rms_epsilon", 1e-5f);
    w.kv_f32(P + "attention.layer_norm_epsilon", 1e-6f);
    w.kv_i32_array(P + "attention.head_count_kv", kinds);
    w.kv_u32(P + "attention.q_lora_rank", 1536);
    w.kv_u32(P + "attention.kv_lora_rank", 512);
    w.kv_u32(P + "rope.dimension_count", 0);
    w.kv_u32(P + "attention.key_length", 512);
    w.kv_u32(P + "attention.value_length", 512);
    w.kv_u32(P + "attention.key_length_mla", 256);
    w.kv_u32(P + "attention.value_length_mla", 256);
    w.kv_u32(P + "ssm.conv_kernel", 4);
    w.kv_u32(P + "kda.head_dim", 128);
    w.kv_f32(P + "kda.gate_lower_bound", -5.0f);
    w.kv_u32(P + "attention.indexer.head_count", 32);
    w.kv_u32(P + "attention.indexer.key_length", 128);
    w.kv_u32(P + "attention.indexer.top_k", 2048);
    w.kv_u32(P + "attention.indexer.kpool", 4);
    w.kv_u32(P + "hyper_connection.count", 4);
    w.kv_u32(P + "hyper_connection.sinkhorn_iterations", 20);
    w.kv_f32(P + "hyper_connection.epsilon", 1e-6f);
    w.kv_u32(P + "leading_dense_block_count", 3);
    w.kv_u32(P + "expert_count", 288);
    w.kv_u32(P + "expert_used_count", 8);
    w.kv_u32(P + "expert_group_count", 1);
    w.kv_u32(P + "expert_gating_func", 2);
    w.kv_u32(P + "expert_feed_forward_length", 2048);
    w.kv_u32(P + "expert_shared_feed_forward_length", 2048);
    w.kv_u32(P + "expert_shared_count", 1);
    w.kv_f32(P + "expert_weights_scale", 2.5f);
    w.kv_bool(P + "expert_weights_norm", true);
    w.kv_f32_array(P + "swiglu_clamp_exp", std::vector<float>(46, 10.f));
    w.kv_f32_array(P + "swiglu_clamp_shexp", std::vector<float>(46, 10.f));
    w.kv_u32(P + "nextn_predict_layers", 1);
    if (vocab_size) w.kv_u32(P + "vocab_size", vocab_size);
    w.kv_string_array("tokenizer.ggml.tokens", {"a", "b", "c"});
    return w.write(path);
}

void check_values(const ie::Glm5NextConfig& c) {
    assert(c.n_layers == 46 && c.hidden == 4096 && c.n_q_heads == 64 && c.ffn == 12288);
    assert(c.attn_kind.size() == 46);
    assert(c.q_lora_rank == 1536 && c.kv_lora_rank == 512 && c.key_len == 512 && c.key_len_mla == 256);
    assert(c.kda_head_dim == 128 && c.conv_kernel == 4 && c.kda_gate_lower_bound == -5.0f);
    assert(c.indexer_n_heads == 32 && c.indexer_head_dim == 128 && c.indexer_top_k == 2048 && c.indexer_kpool == 4);
    assert(c.hc_count == 4 && c.hc_sinkhorn_iters == 20);
    assert(c.leading_dense == 3 && c.n_experts == 288 && c.n_experts_used == 8 && c.expert_ffn == 2048);
    assert(c.n_shared_experts == 1 && c.shexp_ffn == 2048 && c.expert_gating_func == 2);
    assert(std::fabs(c.expert_weights_scale - 2.5f) < 1e-6f && c.expert_weights_norm);
    assert(c.swiglu_clamp_exp.size() == 46 && c.swiglu_clamp_shexp.size() == 46);
    assert(c.nextn_predict_layers == 1);
}
}  // namespace

int main() {
    const std::string path = "/tmp/ie-glm5-config-" + std::to_string(getpid()) + ".gguf";
    for (const char* arch : {"glm5next", "glm5-next"}) {
        assert(write_glm(path, arch, arch).empty());
        ie::GgufReader g;
        assert(g.open(path).empty());
        assert(ie::detect_arch(g) == ie::ModelArch::kGlm5Next);
        ie::Glm5NextConfig c;
        const std::string e = ie::read_glm5next_config(g, c);
        if (!e.empty()) { std::fprintf(stderr, "FAIL (%s): %s\n", arch, e.c_str()); return 1; }
        check_values(c);
    }
    // vocab_size is checked against the tokenizer under either name (3 tokens in the synthetic file).
    for (const char* arch : {"glm5next", "glm5-next"}) {
        for (uint32_t vs : {3u, 4u}) {
            assert(write_glm(path, arch, arch, vs).empty());
            ie::GgufReader g;
            assert(g.open(path).empty());
            ie::Glm5NextConfig c;
            assert(ie::read_glm5next_config(g, c).empty() == (vs == 3));
        }
    }
    // Keys follow the file's OWN arch name: a `glm5-next` file with `glm5next.` keys is malformed.
    {
        assert(write_glm(path, "glm5-next", "glm5next").empty());
        ie::GgufReader g;
        assert(g.open(path).empty());
        ie::Glm5NextConfig c;
        assert(!ie::read_glm5next_config(g, c).empty());
    }
    std::remove(path.c_str());

    if (const char* real = std::getenv("IE_GLM53_GGUF"); real && *real) {
        ie::GgufReader g;
        if (const auto e = g.open(real); !e.empty()) { std::fprintf(stderr, "FAIL open %s: %s\n", real, e.c_str()); return 1; }
        assert(ie::detect_arch(g) == ie::ModelArch::kGlm5Next);
        ie::Glm5NextConfig c;
        if (const auto e = ie::read_glm5next_config(g, c); !e.empty()) { std::fprintf(stderr, "FAIL real: %s\n", e.c_str()); return 1; }
        check_values(c);
        assert(c.vocab == 154880);
        std::printf("glm5_config_test: real file OK (%s)\n", real);
    } else {
        std::puts("glm5_config_test: real-file check SKIPPED (set IE_GLM53_GGUF to the shard-1 path)");
    }
    std::puts("glm5_config_test: OK");
    return 0;
}
