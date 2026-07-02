// tools/ie_debug.cpp — long-running debug REPL.
//
// Loads model + caches ONCE, then accepts commands on stdin:
//
//   P <prompt>            — encode, prefill, dump activations (if --dump set),
//                            print argmax + top-5 of last token.
//   R                     — reset KV + DeltaNet caches.
//   Q                     — quit.
//   EOF                   — quit.
//
// Discipline: never run in parallel with another model-loading process.
// One copy of weights on the GPU at a time.

#include "ie/allocator.hpp"
#include "ie/deltanet_state.hpp"
#include "ie/gguf.hpp"
#include "ie/kv_cache.hpp"
#include "ie/ops.hpp"
#include "ie/qwen36.hpp"
#include "ie/tokenizer.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr const char* G = "\033[32m";
constexpr const char* Y = "\033[33m";
constexpr const char* R = "\033[31m";
constexpr const char* B = "\033[1m";
constexpr const char* Z = "\033[0m";

double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

}  // namespace

int main(int argc, char** argv) {
    std::string gguf_path = "/home/weezy/.seal/models/Qwen3.6-35B-A3B-Q4_K_M.gguf";
    std::string dump_prefix;            // empty = no dump
    uint32_t    max_ctx  = 512;
    uint32_t    max_T_ws = 64;          // initial workspace size for prefill
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--gguf"   && i + 1 < argc) gguf_path   = argv[++i];
        else if (a == "--dump"   && i + 1 < argc) dump_prefix = argv[++i];
        else if (a == "--ctx"    && i + 1 < argc) max_ctx     = std::atoi(argv[++i]);
        else if (a == "--max-T"  && i + 1 < argc) max_T_ws    = std::atoi(argv[++i]);
    }

    std::printf("%sie-debug%s — long-running REPL\n", B, Z);
    std::printf("  gguf : %s\n", gguf_path.c_str());
    std::printf("  dump : %s\n", dump_prefix.empty() ? "(off)" : dump_prefix.c_str());
    std::printf("  ctx  : %u\n", max_ctx);

    // 1. Open GGUF + tokenizer
    ie::GgufReader g;
    if (auto err = g.open(gguf_path); !err.empty()) {
        std::fprintf(stderr, "%sgguf open: %s%s\n", R, err.c_str(), Z);
        return 1;
    }
    ie::Tokenizer tok;
    if (auto err = tok.load_from_gguf(g); !err.empty()) {
        std::fprintf(stderr, "%stokenizer: %s%s\n", R, err.c_str(), Z);
        return 1;
    }

    // 2. Device + allocator + model
    ie::DeviceAllocator alloc;
    if (auto err = alloc.init(); !err.empty()) {
        std::fprintf(stderr, "%salloc: %s%s\n", R, err.c_str(), Z);
        return 1;
    }
    auto& q = alloc.queue();
    std::printf("  device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    const double t0 = now_ms();
    std::printf("  loading model (~22 GB to device)...\n");
    ie::QwenConfig cfg;
    ie::QwenModel model;
    if (auto err = model.load(alloc, g, cfg); !err.empty()) {
        std::fprintf(stderr, "%smodel.load: %s%s\n", R, err.c_str(), Z);
        return 1;
    }
    if (!dump_prefix.empty()) model.set_dump_prefix(dump_prefix);
    std::printf("  loaded in %.1f s\n", (now_ms() - t0) / 1000.0);

    // 3. Caches
    const uint32_t L_full = cfg.n_layers / cfg.full_attn_interval;
    const uint32_t L_lin  = cfg.n_layers - L_full;
    ie::KvCache kv;
    if (auto err = kv.init(alloc, ie::KvCacheConfig{L_full, cfg.n_kv_heads, max_ctx, cfg.head_dim});
        !err.empty()) { std::fprintf(stderr, "kv: %s\n", err.c_str()); return 1; }
    ie::DeltaNetState dn;
    if (auto err = dn.init(alloc, ie::DeltaNetStateConfig{
            L_lin, cfg.ssm_n_v_heads, cfg.ssm_head_dim, cfg.ssm_head_dim,
            cfg.ssm_inner * 2, cfg.ssm_conv_kernel}); !err.empty()) {
        std::fprintf(stderr, "dn: %s\n", err.c_str()); return 1;
    }
    dn.reset(q);
    if (auto err = model.ensure_workspace(max_T_ws); !err.empty()) {
        std::fprintf(stderr, "ws: %s\n", err.c_str()); return 1;
    }
    if (auto err = model.ensure_attn_partials(max_ctx); !err.empty()) {
        std::fprintf(stderr, "attn_partials: %s\n", err.c_str()); return 1;
    }

    // 4. Persistent device buffers reused across prompts.
    auto* d_logits = sycl::malloc_device<sycl::half>(cfg.vocab, q);
    auto* d_pick   = sycl::malloc_device<int32_t>(1, q);
    auto* d_ids_buf = sycl::malloc_device<int32_t>(max_ctx, q);

    uint32_t cur_pos = 0;          // position counter for KV/DN state
    int      cmd_idx = 0;          // command index (used in dump filenames)

    std::printf("\n%sready. Commands: P <prompt>, R, Q, EOF%s\n", G, Z);

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        const char op = line[0];
        if (op == 'Q' || op == 'q') break;

        if (op == 'R' || op == 'r') {
            kv.reset();
            dn.reset(q);
            cur_pos = 0;
            std::printf("  caches reset.\n");
            continue;
        }

        if (op != 'P' && op != 'p') {
            std::printf("  unknown op %c\n", op);
            continue;
        }
        std::string prompt = line.substr(1);
        // strip leading space
        size_t s = 0; while (s < prompt.size() && (prompt[s] == ' ' || prompt[s] == '\t')) ++s;
        prompt = prompt.substr(s);
        if (prompt.empty()) { std::printf("  empty prompt\n"); continue; }

        // Adjust dump prefix per-command if --dump set: append "_cmd<idx>"
        std::string per_cmd_prefix;
        if (!dump_prefix.empty()) {
            per_cmd_prefix = dump_prefix + "_cmd" + std::to_string(cmd_idx);
            model.set_dump_prefix(per_cmd_prefix);
        }

        const auto ids = tok.encode(prompt, /*allow_special=*/true);
        if (ids.empty()) { std::printf("  empty encoding\n"); continue; }
        if (cur_pos + ids.size() > max_ctx) {
            std::printf("  %sprompt too long for ctx=%u (have %u + %zu)%s\n",
                        Y, max_ctx, cur_pos, ids.size(), Z);
            continue;
        }
        if (auto e = model.ensure_workspace(uint32_t(ids.size())); !e.empty()) {
            std::printf("  ws: %s\n", e.c_str());
            continue;
        }
        q.memcpy(d_ids_buf, ids.data(), ids.size() * sizeof(int32_t)).wait();

        const double t = now_ms();
        model.forward(q, d_ids_buf, uint32_t(ids.size()), cur_pos, kv, dn, d_logits).wait();
        const double dt = now_ms() - t;
        cur_pos += uint32_t(ids.size());

        // Sample
        ie::sample_argmax(q, d_logits, d_pick, cfg.vocab).wait();
        int32_t pick;
        q.memcpy(&pick, d_pick, sizeof(int32_t)).wait();
        std::string out = tok.decode(std::vector<int32_t>{pick}, false);

        // top-5 from logits
        std::vector<sycl::half> hl(cfg.vocab);
        q.memcpy(hl.data(), d_logits, cfg.vocab * sizeof(sycl::half)).wait();
        struct Pair { int32_t id; float v; };
        std::vector<Pair> top; top.reserve(cfg.vocab);
        for (uint32_t i = 0; i < cfg.vocab; ++i) top.push_back({int32_t(i), float(hl[i])});
        std::partial_sort(top.begin(), top.begin() + 5, top.end(),
                          [](const Pair& a, const Pair& b){ return a.v > b.v; });

        std::printf("  cmd%d: %zu toks, %.1f ms (%.1f tok/s) → argmax=%d '%s'\n",
                    cmd_idx, ids.size(), dt, ids.size()/(dt/1000.0), pick, out.c_str());
        std::printf("     top-5:");
        for (int i = 0; i < 5; ++i) {
            const auto t = tok.decode(std::vector<int32_t>{top[i].id}, false);
            std::string clean;
            for (char c : t) clean += (c >= 0x20 && c <= 0x7E) ? c : '?';
            std::printf("  %d='%s'(%.2f)", top[i].id, clean.c_str(), top[i].v);
        }
        std::putchar('\n');
        if (!per_cmd_prefix.empty()) {
            std::printf("     dumped: %s_L00..L41.bin\n", per_cmd_prefix.c_str());
        }
        ++cmd_idx;
        std::fflush(stdout);
    }

    sycl::free(d_logits, q); sycl::free(d_pick, q); sycl::free(d_ids_buf, q);
    std::printf("bye.\n");
    return 0;
}
