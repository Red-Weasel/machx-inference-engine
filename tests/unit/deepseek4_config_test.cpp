// tests/unit/deepseek4_config_test.cpp — Milestone 0: detect_arch +
// read_deepseek4_config against the REAL DeepSeek-V4-Flash-0731 GGUF.
// Opens shard 1 of 4 ONLY — that shard is metadata-only (0 tensors, 5 MB), so
// this is a fast mmap header parse with no tensor data touched and no GPU.
// Skips-with-warning (exit 0) if the model is absent, mirroring
// tests/model_config_test.cpp so CI on other boxes stays green.
#undef NDEBUG  // build is Release (-DNDEBUG); asserts must stay live here
#include "ie/model_config.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <string>

namespace {

// Absolute path (external drive, not under $HOME unlike the .seal models).
constexpr const char* kDs4Shard1 =
    "${IE_MODELS_DIR}/DeepSeek-V4-Flash-0731-GGUF/UD-Q3_K_XL/"
    "DeepSeek-V4-Flash-0731-UD-Q3_K_XL-00001-of-00004.gguf";

}  // namespace

int main() {
    ie::GgufReader g;
    const std::string err = g.open(kDs4Shard1);
    if (!err.empty()) {
        std::fprintf(stderr, "deepseek4_config_test: SKIP (cannot open %s: %s)\n",
                     kDs4Shard1, err.c_str());
        std::puts("deepseek4_config_test: SKIPPED");
        return 0;
    }

    assert(ie::detect_arch(g) == ie::ModelArch::kDeepSeek4);

    ie::DeepSeek4Config c;
    const std::string cerr = ie::read_deepseek4_config(g, c);
    if (!cerr.empty()) {
        std::fprintf(stderr, "read_deepseek4_config failed: %s\n", cerr.c_str());
        return 1;
    }

    // Dims.
    assert(c.n_layers   == 43);
    assert(c.hidden     == 4096);
    assert(c.n_q_heads  == 64);
    assert(c.n_kv_heads == 1);
    assert(c.head_dim   == 512);
    assert(c.v_head_dim == 512);
    assert(c.vocab      == 129280);
    assert(c.ctx_train  == 1048576);
    assert(std::fabs(c.rms_eps - 1e-6f) <= 1e-9f);

    // MLA low-rank projections + SWA window.
    assert(c.q_lora_rank    == 1024);
    assert(c.o_lora_rank    == 1024);
    assert(c.o_groups       == 8);
    assert(c.sliding_window == 128);

    // Lightning indexer.
    assert(c.indexer_n_heads  == 64);
    assert(c.indexer_head_dim == 128);
    assert(c.indexer_top_k    == 512);

    // Array-valued KVs. compress_ratios is 46 entries for 43 layers — the
    // surplus is REAL and unexplained, so the test pins the observed length
    // rather than asserting the "obvious" n_layers.
    assert(c.compress_ratios.size() == 46);
    assert(c.compress_ratios[0] == 0);
    assert(c.compress_ratios[1] == 0);
    assert(c.compress_ratios[2] == 4);
    assert(c.compress_ratios[3] == 128);
    assert(c.compress_ratios[43] == 0);
    assert(c.compress_ratios[45] == 0);
    assert(std::fabs(c.compress_rope_theta - 160000.f) <= 1.f);

    // YaRN RoPE.
    assert(c.rope_dim == 64);
    assert(std::fabs(c.rope_theta - 10000.f) <= 1.f);
    assert(std::fabs(c.rope_freq_scale - 1.0f / 16.0f) <= 1e-6f);
    assert(c.rope_orig_ctx == 65536);
    assert(c.rope_ext_factor == 1.0f);          // scaling.type == "yarn"
    assert(std::fabs(c.yarn_beta_fast - 32.f) <= 1e-6f);
    assert(std::fabs(c.yarn_beta_slow - 1.f)  <= 1e-6f);

    // MoE.
    assert(c.n_experts        == 256);
    assert(c.n_experts_used   == 6);
    assert(c.n_shared_experts == 1);
    assert(c.expert_ffn       == 2048);
    assert(c.expert_gating_func == 4);          // outside llama.cpp's enum; kept raw
    assert(std::fabs(c.expert_weights_scale - 1.5f) <= 1e-6f);
    assert(c.expert_weights_norm == true);

    // Per-layer SwiGLU clamps: length == n_layers, 10.0 everywhere.
    assert(c.swiglu_clamp_exp.size()   == c.n_layers);
    assert(c.swiglu_clamp_shexp.size() == c.n_layers);
    for (uint32_t i = 0; i < c.n_layers; ++i) {
        assert(std::fabs(c.swiglu_clamp_exp[i]   - 10.f) <= 1e-6f);
        assert(std::fabs(c.swiglu_clamp_shexp[i] - 10.f) <= 1e-6f);
    }

    // Hyper-connections + hash routing.
    assert(c.hc_count          == 4);
    assert(c.hc_sinkhorn_iters == 20);
    assert(std::fabs(c.hc_eps - 1e-6f) <= 1e-9f);
    assert(c.hash_layer_count  == 3);

    // A wrong prefix must not silently succeed anywhere else: the reader is
    // deepseek4-only, so re-reading a NON-deepseek4 GGUF is not testable here.
    // Instead pin the negative that IS reachable: this file is not dense.
    assert(!ie::is_dense_arch(ie::detect_arch(g)));

    std::puts("deepseek4_config_test: OK");
    return 0;
}
