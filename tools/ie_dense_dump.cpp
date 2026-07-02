// tools/ie_dense_dump.cpp — P2 Task 7: engine-side residual dumper for the
// dense (qwen3) path.  Counterpart of ie-llama-dump: loads a qwen3-dense
// GGUF, prefills one prompt with per-layer dumps enabled
// (DenseModel::set_dump_prefix), and prints last-token argmax + top-5 for
// the greedy-parity check.
//
// Output files: <prefix>_L00..L<n_layers+1>.bin (fp32 [T,H]) + .meta,
// same convention as ie-debug/ie-llama-dump → tools/diff_layers.sh works
// with max_slot = n_layers + 1 (37 for qwen3-8b).
//
// With -n N it also greedily decodes N tokens after the prefill (argmax,
// temp 0) and prints the token ids + decoded text — the engine side of the
// Task 7 greedy-parity check vs `llama-cli --temp 0`.
//
// Usage:
//   ie-dense-dump --gguf <path> --dump <prefix> [-p "prompt"] [--ctx N] [-n N]

#include "ie/allocator.hpp"
#include "ie/dense_transformer.hpp"
#include "ie/gguf.hpp"
#include "ie/kv_cache.hpp"
#include "ie/model_config.hpp"
#include "ie/ops.hpp"
#include "ie/tokenizer.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    std::string gguf_path = std::string(std::getenv("HOME") ? std::getenv("HOME") : "") +
                            "/.seal/models/Qwen3-8B-Q4_K_M.gguf";
    std::string prompt = "The capital of France is";
    std::string dump_prefix;
    uint32_t max_ctx = 256;
    uint32_t n_decode = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--gguf" && i + 1 < argc) gguf_path   = argv[++i];
        else if (a == "-p"     && i + 1 < argc) prompt      = argv[++i];
        else if (a == "--dump" && i + 1 < argc) dump_prefix = argv[++i];
        else if (a == "--ctx"  && i + 1 < argc) max_ctx     = std::atoi(argv[++i]);
        else if (a == "-n"     && i + 1 < argc) n_decode    = std::atoi(argv[++i]);
    }
    if (dump_prefix.empty() && n_decode == 0) {
        std::fprintf(stderr,
                     "usage: %s --gguf <gguf> [--dump <prefix>] [-p \"prompt\"] "
                     "[--ctx N] [-n greedy_tokens]\n",
                     argv[0]);
        return 1;
    }

    ie::GgufReader g;
    if (auto err = g.open(gguf_path); !err.empty()) {
        std::fprintf(stderr, "gguf open: %s\n", err.c_str()); return 1;
    }
    if (!ie::is_dense_arch(ie::detect_arch(g))) {
        std::fprintf(stderr, "%s is not a dense-path GGUF (qwen3/qwen2/llama)\n", gguf_path.c_str());
        return 1;
    }
    ie::DenseConfig cfg;
    if (auto err = ie::read_dense_config_auto(g, cfg); !err.empty()) {
        std::fprintf(stderr, "read_dense_config: %s\n", err.c_str()); return 1;
    }
    std::printf("ie-dense-dump: n_layers=%u hidden=%u q/kv=%u/%u hd=%u ffn=%u "
                "vocab=%u rope_dim=%u theta=%g eps=%g\n",
                cfg.n_layers, cfg.hidden, cfg.n_q_heads, cfg.n_kv_heads,
                cfg.head_dim, cfg.ffn, cfg.vocab, cfg.rope_dim,
                double(cfg.rope_theta), double(cfg.rms_eps));

    ie::Tokenizer tok;
    if (auto err = tok.load_from_gguf(g); !err.empty()) {
        std::fprintf(stderr, "tokenizer: %s\n", err.c_str()); return 1;
    }

    ie::DeviceAllocator alloc;
    if (auto err = alloc.init(); !err.empty()) {
        std::fprintf(stderr, "alloc: %s\n", err.c_str()); return 1;
    }
    auto& q = alloc.queue();

    ie::DenseModel model;
    if (auto err = model.load(alloc, g, cfg); !err.empty()) {
        std::fprintf(stderr, "model.load: %s\n", err.c_str()); return 1;
    }
    if (!dump_prefix.empty()) model.set_dump_prefix(dump_prefix);

    ie::KvCache kv;
    if (auto err = kv.init(alloc, ie::KvCacheConfig{cfg.n_layers, cfg.n_kv_heads,
                                                    max_ctx, cfg.head_dim});
        !err.empty()) {
        std::fprintf(stderr, "kv.init: %s\n", err.c_str()); return 1;
    }

    auto ids = tok.encode(prompt, /*allow_special=*/false);
    if (ids.empty()) { std::fprintf(stderr, "empty prompt encode\n"); return 1; }
    std::printf("prompt='%s' tokens=", prompt.c_str());
    for (auto t : ids) std::printf("%d ", int(t));
    std::printf("\n");

    if (auto err = model.ensure_workspace(uint32_t(ids.size())); !err.empty()) {
        std::fprintf(stderr, "ensure_workspace: %s\n", err.c_str()); return 1;
    }
    if (auto err = model.ensure_attn_partials(max_ctx); !err.empty()) {
        std::fprintf(stderr, "ensure_attn_partials: %s\n", err.c_str()); return 1;
    }

    auto* d_ids    = sycl::malloc_device<int32_t>(ids.size(), q);
    auto* d_logits = sycl::malloc_device<sycl::half>(cfg.vocab, q);
    if (!d_ids || !d_logits) { std::fprintf(stderr, "device alloc failed\n"); return 1; }
    q.memcpy(d_ids, ids.data(), ids.size() * sizeof(int32_t)).wait();

    try {
        model.forward(q, d_ids, uint32_t(ids.size()), /*start_pos=*/0, kv,
                      d_logits).wait_and_throw();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "SYCL exception: %s\n", e.what());
        std::_Exit(1);
    }

    // Last-token argmax + top-5 (greedy-parity evidence).
    std::vector<sycl::half> h(cfg.vocab);
    q.memcpy(h.data(), d_logits, cfg.vocab * sizeof(sycl::half)).wait();
    std::vector<int32_t> idx(cfg.vocab);
    for (uint32_t i = 0; i < cfg.vocab; ++i) idx[i] = int32_t(i);
    std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                      [&](int32_t a, int32_t b) { return float(h[a]) > float(h[b]); });
    std::printf("argmax: id=%d logit=%.2f\n", idx[0], double(float(h[idx[0]])));
    for (int r = 0; r < 5; ++r) {
        auto s = tok.token_str(idx[r]);
        std::printf("  top%d id=%-7d logit=%8.3f '%.*s'\n", r + 1, idx[r],
                    double(float(h[idx[r]])), int(s.size()), s.data());
    }
    if (!dump_prefix.empty())
        std::printf("dumped slots L00..L%02u to %s_LNN.bin\n",
                    cfg.n_layers + 1, dump_prefix.c_str());

    // Greedy decode loop (-n N): argmax-feed, dumps off (they'd be one row
    // per step), print ids then the decoded continuation.
    if (n_decode > 0) {
        model.set_dump_prefix("");
        std::vector<int32_t> picks;
        int32_t cur = idx[0];
        uint32_t pos = uint32_t(ids.size());
        try {
            for (uint32_t s = 0; s < n_decode; ++s) {
                picks.push_back(cur);
                if (s + 1 == n_decode) break;
                q.memcpy(d_ids, &cur, sizeof(int32_t)).wait();
                model.forward(q, d_ids, /*T=*/1, pos++, kv, d_logits)
                     .wait_and_throw();
                q.memcpy(h.data(), d_logits, cfg.vocab * sizeof(sycl::half)).wait();
                int32_t best = 0;
                float   bv   = float(h[0]);
                for (uint32_t i = 1; i < cfg.vocab; ++i)
                    if (float(h[i]) > bv) { bv = float(h[i]); best = int32_t(i); }
                cur = best;
            }
        } catch (const sycl::exception& e) {
            std::fprintf(stderr, "SYCL exception (decode): %s\n", e.what());
            std::_Exit(1);
        }
        std::printf("greedy ids:");
        for (auto t : picks) std::printf(" %d", t);
        std::printf("\ngreedy text: '%s'\n",
                    tok.decode(std::span<const int32_t>(picks), false).c_str());
    }

    sycl::free(d_ids, q);
    sycl::free(d_logits, q);
    return 0;
}
