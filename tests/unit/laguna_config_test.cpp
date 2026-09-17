// Gate for read_laguna_config against the REAL Laguna-S-2.1 GGUF metadata
// shard.  Values are the 2026-08-09 ground-truth dump recorded in
// docs/laguna/00_PORT_PLAN.md.  SKIPs when the model is not present.
//
// THE POINT OF THIS GATE is the per-layer map: the file carries NO
// sliding-window-pattern key, so a loader learns a layer's type ONLY from the
// 48-entry `attention.head_count` array (48 heads = full attention, 72 =
// sliding).  Getting that wrong silently gives 3/4 of the layers the wrong rope
// table and the wrong head count.
#undef NDEBUG
#include "ie/model_config.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {
// Model path comes from $IE_LAGUNA_GGUF — these gates need the real Laguna-S-2.1
// GGUF, which is far too large to ship. Unset (or unreachable) means the
// test SKIPS cleanly, so a checkout without the model still goes green.
const char* shard1_path() { return std::getenv("IE_LAGUNA_GGUF"); }
}

int main() {
    const char* path = shard1_path();
    if (!path || !*path) {
        std::puts("laguna_config_test: SKIPPED (set IE_LAGUNA_GGUF to the model's shard-1 path)");
        return 0;
    }
    ie::GgufReader g;
    if (!g.open(path).empty()) {
        std::puts("laguna_config_test: SKIPPED (model not present)");
        return 0;
    }
    assert(ie::detect_arch(g) == ie::ModelArch::kLaguna);
    ie::LagunaConfig c;
    const std::string e = ie::read_laguna_config(g, c);
    if (!e.empty()) { std::fprintf(stderr, "FAIL: %s\n", e.c_str()); return 1; }

    assert(c.n_layers == 48);
    assert(c.hidden == 3072);
    assert(c.ffn_dense == 12288);
    assert(c.n_kv_heads == 8);
    assert(c.head_dim == 128 && c.v_head_dim == 128);
    assert(c.vocab == 100352);
    assert(c.sliding_window == 512);
    assert(c.dense_layers == 1);
    assert(c.n_experts == 256 && c.n_experts_used == 10);
    assert(c.expert_ffn == 1024 && c.shared_expert_ffn == 1024);
    assert(c.expert_gating_func == 2);
    assert(c.expert_weights_norm);
    assert(std::fabs(c.expert_weights_scale - 2.5f) < 1e-6f);

    // Two rope configs.  Full: YaRN, theta 500000, 64 of 128 dims rotate.
    // Sliding: plain, theta 10000, all 128 dims rotate.
    assert(std::fabs(c.rope_theta_full - 500000.f) < 1.f);
    assert(std::fabs(c.rope_theta_swa - 10000.f) < 1.f);
    assert(c.rope_dim_full == 64 && c.rope_dim_swa == 128);
    assert(std::fabs(c.yarn_factor - 32.f) < 1e-6f);
    assert(c.yarn_orig_ctx == 8192);
    assert(std::fabs(c.yarn_beta_fast - 32.f) < 1e-6f);
    assert(std::fabs(c.yarn_beta_slow - 1.f) < 1e-6f);

    // The per-layer map, which is the whole reason this gate exists.
    assert(c.n_q_heads.size() == 48);
    uint32_t n_full = 0, n_slide = 0;
    for (uint32_t L = 0; L < c.n_layers; ++L) {
        const uint32_t hh = c.heads_for(L);
        assert(hh == 48 || hh == 72);
        if (c.is_sliding(L)) {
            ++n_slide;
            assert(hh == 72);
            assert(c.rope_dim_for(L) == 128);
            assert(std::fabs(c.rope_theta_for(L) - 10000.f) < 1.f);
        } else {
            ++n_full;
            assert(hh == 48);
            assert(c.rope_dim_for(L) == 64);
            assert(std::fabs(c.rope_theta_for(L) - 500000.f) < 1.f);
        }
    }
    // 1:3 full:sliding — full attention on every 4th layer starting at 0.
    assert(n_full == 12 && n_slide == 36);
    for (uint32_t L = 0; L < c.n_layers; ++L)
        assert(c.is_sliding(L) == (L % 4 != 0));
    std::printf("  48 layers: %u full-attention (48 heads, YaRN, 64/128 rotary) / "
                "%u sliding (72 heads, plain rope, 128/128)\n", n_full, n_slide);
    std::puts("laguna_config_test: OK — every key and the per-layer type map match the file");
    return 0;
}
