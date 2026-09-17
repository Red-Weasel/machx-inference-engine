// Gate for read_inkling_config against the REAL Inkling-Small metadata shard
// (5 MB, shard 1 of 5 — safe to open).  Values asserted are the 2026-08-09
// ground-truth dump recorded in docs/inkling/00_PORT_PLAN.md.  SKIPs cleanly
// when the Expansion drive is unmounted.
#undef NDEBUG
#include "ie/model_config.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {
// Model path comes from $IE_INKLING_GGUF — these gates need the real Inkling-Small
// GGUF, which is far too large to ship. Unset (or unreachable) means the
// test SKIPS cleanly, so a checkout without the model still goes green.
const char* shard1_path() { return std::getenv("IE_INKLING_GGUF"); }
}

int main() {
    const char* path = shard1_path();
    if (!path || !*path) {
        std::puts("inkling_config_test: SKIPPED (set IE_INKLING_GGUF to the model's shard-1 path)");
        return 0;
    }
    ie::GgufReader g;
    const std::string err = g.open(path);
    if (!err.empty()) {
        std::fprintf(stderr, "inkling_config_test: SKIP (cannot open %s: %s)\n",
                     path, err.c_str());
        std::puts("inkling_config_test: SKIPPED");
        return 0;
    }
    assert(ie::detect_arch(g) == ie::ModelArch::kInkling);
    ie::InklingConfig c;
    const std::string e = ie::read_inkling_config(g, c);
    if (!e.empty()) { std::fprintf(stderr, "FAIL: %s\n", e.c_str()); return 1; }
    assert(c.n_layers == 42);
    assert(c.hidden == 4096);
    assert(c.ffn_dense == 16384);
    assert(c.n_q_heads == 32 && c.head_dim == 128 && c.v_head_dim == 128);
    assert(c.n_kv_heads.size() == 42);
    assert(c.n_experts == 256 && c.n_experts_used == 6 && c.n_shared_experts == 2);
    assert(c.expert_ffn == 2048);
    assert(std::fabs(c.expert_weights_scale - 8.0f) < 1e-6f);
    assert(c.expert_gating_func == 2);
    assert(c.dense_layers == 2);
    assert(c.sliding_window == 512 && c.swa_pattern.size() == 42);
    assert(c.d_rel == 16 && c.rel_extent == 1024 && c.rel_extent_swa == 512);
    assert(c.shortconv_kernel == 4);
    assert(c.vocab == 201024 && c.vocab_unpadded == 200058);
    assert(c.ctx_train == 1048576);
    // Per-layer relative extent.  Verified against the REAL `attn_rel_proj`
    // row lengths (ne[0]) scanned from all 5 shards on 2026-08-09: 512 on 35
    // layers, 1024 on layers 5, 11, 17, 23, 29, 35, 41.  This is the check that
    // stops a loader from applying one extent to every layer.
    const uint32_t full_layers[] = {5, 11, 17, 23, 29, 35, 41};
    uint32_t n_full = 0, n_swa = 0;
    for (uint32_t L = 0; L < c.n_layers; ++L) {
        const uint32_t e = c.rel_extent_for(L);
        assert(e == 512 || e == 1024);
        if (e == 1024) ++n_full; else ++n_swa;
    }
    assert(n_full == 7 && n_swa == 35);
    for (uint32_t L : full_layers) {
        assert(c.rel_extent_for(L) == 1024);
        assert(!c.is_sliding(L));
    }
    assert(c.rel_extent_for(0) == 512 && c.is_sliding(0));
    std::printf("  per-layer rel_extent: %u full-attention (1024) / %u sliding (512)\n",
                n_full, n_swa);
    std::puts("inkling_config_test: OK — every geometry key matches the ground-truth dump");
    return 0;
}
