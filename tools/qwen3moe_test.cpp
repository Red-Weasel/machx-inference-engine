// tools/qwen3moe_test.cpp — load a qwen3moe GGUF (Qwen3-Coder-30B-A3B) and greedy-generate.
// usage: ie-qwen3moe-test <gguf> [prompt] [n_gen]
#include "ie/allocator.hpp"
#include "ie/gguf.hpp"
#include "ie/kv_cache.hpp"
#include "ie/model_config.hpp"
#include "ie/qwen3moe.hpp"
#include "ie/tokenizer.hpp"

#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <gguf> [prompt] [n_gen]\n", argv[0]); return 2; }
    const std::string gguf = argv[1];
    const std::string prompt = argc > 2 ? argv[2] : "The capital of France is";
    const int n_gen = argc > 3 ? std::atoi(argv[3]) : 24;

    ie::GgufReader g;
    if (auto e = g.open(gguf); !e.empty()) { std::fprintf(stderr, "gguf: %s\n", e.c_str()); return 1; }
    if (ie::detect_arch(g) != ie::ModelArch::kQwen3Moe) { std::fprintf(stderr, "not qwen3moe\n"); return 1; }
    ie::Tokenizer tok;
    if (auto e = tok.load_from_gguf(g); !e.empty()) { std::fprintf(stderr, "tok: %s\n", e.c_str()); return 1; }
    if (prompt == "TOKCHECK") {   // tokenizer-only diagnostic (no model load)
        const std::string chatml = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n";
        auto ids = tok.encode(chatml, /*allow_special=*/true);
        std::printf("encode('<chatml>') -> %zu tokens:", ids.size());
        for (int id : ids) std::printf(" %d", id);
        std::printf("\n  find('<|im_start|>')=%d find('<|im_end|>')=%d eos=%d add_bos=%d bos=%d\n",
                    tok.find_token("<|im_start|>"), tok.find_token("<|im_end|>"),
                    tok.eos_token_id(), int(tok.add_bos_token()), tok.bos_token_id());
        std::printf("  is_special(im_start)=%d is_special(im_end)=%d\n",
                    int(tok.is_special(tok.find_token("<|im_start|>"))),
                    int(tok.is_special(tok.find_token("<|im_end|>"))));
        return 0;
    }
    ie::Qwen3MoeConfig cfg;
    if (auto e = ie::read_qwen3moe_config(g, cfg); !e.empty()) { std::fprintf(stderr, "cfg: %s\n", e.c_str()); return 1; }
    std::printf("cfg: %u layers, hidden %u, %u q / %u kv heads, %u experts (%u used), expert_ffn %u, vocab %u\n",
                cfg.dense.n_layers, cfg.dense.hidden, cfg.dense.n_q_heads, cfg.dense.n_kv_heads,
                cfg.n_experts, cfg.n_experts_used, cfg.expert_ffn, cfg.dense.vocab);

    ie::DeviceAllocator alloc;
    if (auto e = alloc.init(); !e.empty()) { std::fprintf(stderr, "alloc: %s\n", e.c_str()); return 1; }
    auto& q = alloc.queue();

    ie::Qwen3MoeModel m;
    const double t0 = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    if (auto e = m.load(alloc, g, cfg); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
    const double t1 = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    std::printf("loaded in %.1fs\n", t1 - t0);

    ie::KvCache kv;
    if (auto e = kv.init(alloc, ie::KvCacheConfig{cfg.dense.n_layers, cfg.dense.n_kv_heads, 4096, cfg.dense.head_dim}); !e.empty())
        { std::fprintf(stderr, "kv: %s\n", e.c_str()); return 1; }
    kv.reset();

    auto ids = tok.encode(prompt, /*allow_special=*/true);
    const uint32_t T = uint32_t(ids.size());
    if (auto e = m.ensure_workspace(std::max<uint32_t>(T, 1)); !e.empty()) { std::fprintf(stderr, "ws: %s\n", e.c_str()); return 1; }

    auto* d_log = sycl::malloc_device<sycl::half>(cfg.dense.vocab, q);
    std::vector<sycl::half> hl(cfg.dense.vocab);
    auto argmax = [&]{ int b = 0; float bv = -1e30f; for (uint32_t i = 0; i < cfg.dense.vocab; ++i){ float f = float(hl[i]); if (f > bv){bv=f;b=int(i);} } return b; };

    // ---- Numerical diagnostic (IE_DIAG_LOGITS_OUT=<file>) --------------------
    // Runs each prompt token as a TEACHER-FORCED T==1 decode step (identical
    // inputs regardless of fused/unfused), dumping the full logit vector per
    // step. Compare two runs (default vs IE_QWEN3MOE_UNFUSED=1) to isolate the
    // fused decode kernel's numeric error from greedy-trajectory divergence.
    // Format: [uint32 n_steps][uint32 vocab][float32 logits[n_steps*vocab]].
    if (const char* diag_out = std::getenv("IE_DIAG_LOGITS_OUT")) {
        FILE* f = std::fopen(diag_out, "wb");
        if (!f) { std::fprintf(stderr, "diag: cannot open %s\n", diag_out); return 1; }
        const uint32_t vocab = cfg.dense.vocab;
        std::vector<float> lf(vocab);
        const bool prefill_mode = std::getenv("IE_DIAG_PREFILL") != nullptr;
        if (prefill_mode) {
            // Exercise the T>1 PREFILL MoE path (multiple tokens/expert, M_TILE>1):
            // one forward over the whole prompt, dump the last-position logits.
            const uint32_t nsteps = 1;
            std::fwrite(&nsteps, sizeof(uint32_t), 1, f);
            std::fwrite(&vocab, sizeof(uint32_t), 1, f);
            kv.reset();
            m.forward(q, ids.data(), T, 0, kv, d_log).wait();
            q.memcpy(hl.data(), d_log, vocab * sizeof(sycl::half)).wait();
            for (uint32_t i = 0; i < vocab; ++i) lf[i] = float(hl[i]);
            std::fwrite(lf.data(), sizeof(float), vocab, f);
            std::fprintf(stderr, "[diag-prefill] T=%u last-pos argmax %d (%.3f)\n",
                         T, argmax(), float(hl[argmax()]));
        } else {
            // Teacher-forced T==1 DECODE steps (identical inputs fused vs unfused).
            std::fwrite(&T, sizeof(uint32_t), 1, f);
            std::fwrite(&vocab, sizeof(uint32_t), 1, f);
            kv.reset();
            for (uint32_t p = 0; p < T; ++p) {
                int tk = ids[p];
                m.forward(q, &tk, 1, p, kv, d_log).wait();
                q.memcpy(hl.data(), d_log, vocab * sizeof(sycl::half)).wait();
                for (uint32_t i = 0; i < vocab; ++i) lf[i] = float(hl[i]);
                std::fwrite(lf.data(), sizeof(float), vocab, f);
                std::fprintf(stderr, "[diag] step %u (tok %d) -> argmax %d (%.3f)\n",
                             p, tk, argmax(), float(hl[argmax()]));
            }
        }
        std::fclose(f);
        std::printf("[diag] wrote logits to %s (%s mode)\n",
                    diag_out, prefill_mode ? "prefill" : "decode");
        sycl::free(d_log, q);
        return 0;
    }

    m.forward(q, ids.data(), T, 0, kv, d_log).wait();
    q.memcpy(hl.data(), d_log, cfg.dense.vocab * sizeof(sycl::half)).wait();
    std::vector<int> out; out.push_back(argmax());
    const double t2 = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    uint32_t pos = T;
    for (int s = 1; s < n_gen; ++s) {
        int tk = out.back();
        if (tk == tok.eos_token_id()) break;
        m.forward(q, &tk, 1, pos, kv, d_log).wait();
        q.memcpy(hl.data(), d_log, cfg.dense.vocab * sizeof(sycl::half)).wait();
        out.push_back(argmax()); ++pos;
    }
    const double t3 = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    std::printf("prompt: '%s'\noutput: '%s'\n", prompt.c_str(), tok.decode(out, true, {}).c_str());
    std::printf("[decode %.2f tok/s over %d tokens]\n", out.size() > 1 ? (out.size() - 1) / (t3 - t2) : 0.0, int(out.size()));
    sycl::free(d_log, q);
    return 0;
}
