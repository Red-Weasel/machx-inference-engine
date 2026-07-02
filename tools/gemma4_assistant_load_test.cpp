// tools/gemma4_assistant_load_test.cpp — load the gemma4-assistant MTP head GGUF
// and dump its parsed config + confirm all tensors upload. Piece 2 validation.
#include "ie/gemma4_assistant.hpp"
#include "ie/gguf.hpp"
#include "ie/allocator.hpp"

#include <cstdio>

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <mtp-head.gguf>\n", argv[0]); return 2; }
    ie::GgufReader g;
    if (auto e = g.open(argv[1]); !e.empty()) { std::fprintf(stderr, "gguf: %s\n", e.c_str()); return 1; }

    ie::GemmaAssistantConfig cfg;
    if (auto e = ie::read_gemma4_assistant_config(g, cfg); !e.empty()) {
        std::fprintf(stderr, "config: %s\n", e.c_str()); return 1;
    }
    std::printf("config: layers=%u hidden=%u backbone=%u ffn=%u vocab=%u q_heads=%u eps=%.1e swa_win=%u\n",
                cfg.n_layers, cfg.hidden, cfg.n_embd_backbone, cfg.ffn, cfg.vocab,
                cfg.n_q_heads, cfg.rms_eps, cfg.sliding_window);
    for (uint32_t l = 0; l < cfg.n_layers; ++l)
        std::printf("  L%u: swa=%u head_dim=%u n_kv=%u n_rot=%u theta=%.0f\n",
                    l, cfg.is_swa[l], cfg.head_dim[l], cfg.n_kv_heads[l], cfg.n_rot[l], cfg.rope_theta[l]);

    ie::DeviceAllocator alloc;
    if (auto e = alloc.init(); !e.empty()) { std::fprintf(stderr, "alloc: %s\n", e.c_str()); return 1; }
    ie::Gemma4AssistantHead head;
    if (auto e = head.load(alloc, g, cfg); !e.empty()) { std::fprintf(stderr, "LOAD: %s\n", e.c_str()); return 1; }
    std::printf("LOAD OK — head loaded=%d\n", head.loaded());
    return 0;
}
