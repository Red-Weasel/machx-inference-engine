// tests/hf_import_test.cpp — P3e Task 2 (Steps A,B). Host-only: HF config.json
// parsing (real Qwen3-4B-AWQ fixture) + HF→GGUF tensor-name mapping.
#undef NDEBUG
#include "ie/hf_import.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <string>

// Verbatim Qwen/Qwen3-4B-AWQ config.json (the AWQ checkpoint we import).
static const char* kQwen34bAwqConfig = R"JSON({
  "architectures": ["Qwen3ForCausalLM"],
  "attention_bias": false,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "head_dim": 128,
  "hidden_size": 2560,
  "intermediate_size": 9728,
  "max_position_embeddings": 40960,
  "model_type": "qwen3",
  "num_attention_heads": 32,
  "num_hidden_layers": 36,
  "num_key_value_heads": 8,
  "quantization_config": {
    "bits": 4, "group_size": 128, "quant_method": "awq",
    "version": "gemm", "zero_point": true
  },
  "rms_norm_eps": 1e-06,
  "rope_theta": 1000000,
  "tie_word_embeddings": true,
  "torch_dtype": "float16",
  "vocab_size": 151936
})JSON";

static const char* kQwen3NextExl3Config = R"JSON({
  "model_type": "qwen3_next",
  "num_hidden_layers": 48,
  "hidden_size": 2048,
  "intermediate_size": 5120,
  "num_attention_heads": 16,
  "num_key_value_heads": 2,
  "head_dim": 256,
  "vocab_size": 151936,
  "max_position_embeddings": 262144,
  "rms_norm_eps": 1e-06,
  "rope_theta": 10000000,
  "quantization_config": {
    "quant_method": "exl3", "bits": 4.51, "head_bits": 6
  }
})JSON";

// Qwen3-DENSE EXL3 (not Next): head_dim is explicit and ≠ hidden/n_heads, and
// model_type is the plain "qwen3" the extended dense EXL3 importer now accepts.
static const char* kQwen3DenseExl3Config = R"JSON({
  "model_type": "qwen3",
  "num_hidden_layers": 36,
  "hidden_size": 2560,
  "intermediate_size": 9728,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "head_dim": 128,
  "vocab_size": 151936,
  "max_position_embeddings": 40960,
  "rms_norm_eps": 1e-06,
  "rope_theta": 1000000,
  "tie_word_embeddings": true,
  "quantization_config": {
    "quant_method": "exl3", "bits": 3.0, "head_bits": 6
  }
})JSON";

int main() {
    // ---- Step A: parse_hf_config ----
    ie::HfModelMeta m;
    const std::string err = ie::parse_hf_config(kQwen34bAwqConfig, m);
    if (!err.empty()) { std::fprintf(stderr, "parse_hf_config: %s\n", err.c_str()); return 1; }

    assert(m.arch == "qwen3");
    assert(m.cfg.n_layers   == 36);
    assert(m.cfg.hidden     == 2560);
    assert(m.cfg.n_q_heads  == 32);
    assert(m.cfg.n_kv_heads == 8);
    assert(m.cfg.head_dim   == 128);            // explicit (≠ hidden/n_q_heads = 80)
    assert(m.cfg.rope_dim   == 128);            // full rotary
    assert(m.cfg.ffn        == 9728);
    assert(m.cfg.vocab      == 151936);
    assert(m.cfg.ctx_train  == 40960);
    assert(std::fabs(m.cfg.rope_theta - 1e6f) <= 1.f);
    assert(std::fabs(m.cfg.rms_eps - 1e-6f) <= 1e-9f);
    assert(m.tie_word_embeddings == true);
    assert(m.attention_bias == false);
    assert(m.torch_dtype == "float16");
    assert(m.quantized == true);
    assert(m.quant.quant_method == "awq" && m.quant.group_size == 128 &&
           m.quant.version == "gemm" && m.quant.bits == 4);

    ie::HfModelMeta exl3;
    assert(ie::parse_hf_config(kQwen3NextExl3Config, exl3).empty());
    assert(exl3.arch == "qwen3_next");
    assert(exl3.quantized && exl3.quant.quant_method == "exl3");
    assert(exl3.cfg.hidden == 2048 && exl3.cfg.vocab == 151936);

    // Qwen3-dense EXL3: the dense EXL3 importer accepts model_type "qwen3" and
    // emits general.architecture="qwen3" (→ kQwen3Dense; NEOX rope, presence-gated
    // qk-norm). head_dim=128 is explicit (≠ hidden/n_q_heads = 80).
    ie::HfModelMeta dexl3;
    assert(ie::parse_hf_config(kQwen3DenseExl3Config, dexl3).empty());
    assert(dexl3.arch == "qwen3");
    assert(dexl3.quantized && dexl3.quant.quant_method == "exl3");
    assert(dexl3.cfg.head_dim == 128 && dexl3.cfg.rope_dim == 128);
    assert(dexl3.cfg.n_layers == 36 && dexl3.cfg.vocab == 151936);

    // Missing required field must error clearly.
    ie::HfModelMeta bad;
    assert(!ie::parse_hf_config(R"({"model_type":"qwen3"})", bad).empty());

    // ---- Step B: hf_to_gguf_tensor_name ----
    auto map = ie::hf_to_gguf_tensor_name;
    assert(map("model.embed_tokens.weight")               == "token_embd.weight");
    assert(map("model.norm.weight")                       == "output_norm.weight");
    assert(map("lm_head.weight")                          == "output.weight");
    assert(map("model.layers.0.self_attn.q_proj.qweight") == "blk.0.attn_q.qweight");
    assert(map("model.layers.5.self_attn.k_proj.scales")  == "blk.5.attn_k.scales");
    assert(map("model.layers.7.self_attn.v_proj.qzeros")  == "blk.7.attn_v.qzeros");
    assert(map("model.layers.3.self_attn.o_proj.weight")  == "blk.3.attn_output.weight");
    assert(map("model.layers.2.self_attn.q_norm.weight")  == "blk.2.attn_q_norm.weight");
    assert(map("model.layers.2.self_attn.k_norm.weight")  == "blk.2.attn_k_norm.weight");
    assert(map("model.layers.9.mlp.gate_proj.qweight")    == "blk.9.ffn_gate.qweight");
    assert(map("model.layers.9.mlp.up_proj.qweight")      == "blk.9.ffn_up.qweight");
    assert(map("model.layers.9.mlp.down_proj.weight")     == "blk.9.ffn_down.weight");
    assert(map("model.layers.0.input_layernorm.weight")   == "blk.0.attn_norm.weight");
    assert(map("model.layers.0.post_attention_layernorm.weight") == "blk.0.ffn_norm.weight");
    assert(map("model.layers.35.mlp.down_proj.g_idx")     == "blk.35.ffn_down.g_idx");
    assert(map("model.unknown.thing").empty());           // unrecognized → ""

    std::printf("hf_import_test: all OK (config parse + name map)\n");
    return 0;
}
