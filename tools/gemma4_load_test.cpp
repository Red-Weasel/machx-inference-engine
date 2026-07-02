// tools/gemma4_load_test.cpp — Gemma 4 loader gate. Reads the config (prints the
// per-layer head geometry to verify the swa/global pattern), then loads the
// model on one B70 via Gemma4Model. Success = config parses + all Q4_0/Q6_K
// tensors ingest with the variable geometry + fused experts + scales.
#include "ie/gemma4.hpp"
#include "ie/gguf.hpp"
#include "ie/model_config.hpp"
#include "ie/allocator.hpp"

#include <cstdio>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <gemma4.gguf>\n", argv[0]); return 2; }

    ie::GgufReader g;
    if (auto e = g.open(argv[1]); !e.empty()) { std::fprintf(stderr, "gguf open: %s\n", e.c_str()); return 1; }

    ie::GemmaConfig cfg;
    if (auto e = ie::read_gemma4_config(g, cfg); !e.empty()) { std::fprintf(stderr, "config: %s\n", e.c_str()); return 1; }

    std::printf("config: layers=%u hidden=%u q_heads=%u vocab=%u eps=%g swa_window=%u softcap=%g\n"
                "  head_dim swa=%u global=%u | n_rot swa=%u global=%u | theta swa=%g global=%g\n"
                "  moe=%d experts=%u/%u expert_ffn=%u shared_ffn=%u\n",
                cfg.n_layers, cfg.hidden, cfg.n_q_heads, cfg.vocab, cfg.rms_eps,
                cfg.sliding_window, cfg.final_logit_softcap,
                cfg.head_dim_swa, cfg.head_dim_global, cfg.n_rot_swa, cfg.n_rot_global,
                cfg.rope_theta_swa, cfg.rope_theta_global,
                int(cfg.is_moe), cfg.n_experts_used, cfg.n_experts, cfg.expert_ffn, cfg.ffn);
    for (uint32_t l = 0; l < cfg.n_layers; ++l)
        std::printf("  L%02u %s  head_dim=%u  n_kv=%u  n_rot=%u  theta=%g\n",
                    l, cfg.is_swa[l] ? "swa" : "GLOBAL",
                    cfg.head_dim(l), cfg.n_kv_heads[l], cfg.n_rot(l), cfg.rope_theta(l));

    ie::DeviceAllocator alloc;
    if (auto e = alloc.init(); !e.empty()) { std::fprintf(stderr, "alloc: %s\n", e.c_str()); return 1; }

    ie::Gemma4Model m;
    if (auto e = m.load(alloc, g, cfg); !e.empty()) { std::fprintf(stderr, "LOAD FAILED: %s\n", e.c_str()); return 1; }

    std::printf("\nLOAD OK — Gemma 4 ingested on one B70.\n");
    return 0;
}
