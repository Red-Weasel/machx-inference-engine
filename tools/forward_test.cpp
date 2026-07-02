// tools/forward_test.cpp — Phase 8 integration: load model, run forward, sample.
//
//   ie-forward [--gguf path] [--prompt "..."] [--max-new N] [--ctx N]
//
// Default: prompts "Hello, world", runs prefill, decodes argmax of last-token
// logits. Verifies the forward pipeline by:
//   1. Loading all 22 GB of Q4_K_M weights to device (slow first time).
//   2. Encoding the prompt via our tokenizer.
//   3. One-shot prefill of all prompt tokens.
//   4. Sampling argmax of next-token logits.
//   5. Decoding the picked token to text and printing the result.

#include "ie/allocator.hpp"
#include "ie/deltanet_state.hpp"
#include "ie/gguf.hpp"
#include "ie/kv_cache.hpp"
#include "ie/ops.hpp"
#include "ie/qwen36.hpp"
#include "ie/tokenizer.hpp"

#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

constexpr const char* G = "\033[32m";
constexpr const char* R = "\033[31m";
constexpr const char* Y = "\033[33m";
constexpr const char* B = "\033[1m";
constexpr const char* Z = "\033[0m";

double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

}  // namespace

int main(int argc, char** argv) {
    std::string gguf_path = "/home/weezy/.seal/models/Qwen3.6-35B-A3B-Q4_K_M.gguf";
    std::string prompt = "Hello, world";
    bool         use_chat = false;
    int          max_new = 1;
    uint32_t     max_ctx = 256;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--gguf"   && i + 1 < argc) gguf_path = argv[++i];
        else if (a == "--prompt" && i + 1 < argc) prompt    = argv[++i];
        else if (a == "--chat") use_chat = true;
        else if (a == "--max-new" && i + 1 < argc) max_new  = std::atoi(argv[++i]);
        else if (a == "--ctx"     && i + 1 < argc) max_ctx  = std::atoi(argv[++i]);
    }

    std::printf("%sPhase 8 — End-to-end forward%s\n", B, Z);

    // 1. Open GGUF + tokenizer
    ie::GgufReader g;
    if (auto err = g.open(gguf_path); !err.empty()) {
        std::fprintf(stderr, "%sgguf open: %s%s\n", R, err.c_str(), Z);
        return 1;
    }
    std::printf("  GGUF opened: %s (%llu tensors)\n", gguf_path.c_str(),
                (unsigned long long)g.n_tensors());

    ie::Tokenizer tok;
    if (auto err = tok.load_from_gguf(g); !err.empty()) {
        std::fprintf(stderr, "%stokenizer: %s%s\n", R, err.c_str(), Z);
        return 1;
    }
    std::printf("  Tokenizer ready (vocab=%u)\n", tok.vocab_size());

    // 2. Encode prompt
    std::string final_prompt = prompt;
    if (use_chat) {
        std::vector<ie::ChatTurn> turns = {{"user", prompt}};
        final_prompt = ie::build_chatml_prompt(turns, /*add_generation_prompt=*/true,
                                               /*enable_thinking=*/false);
    }
    auto ids = tok.encode(final_prompt, /*allow_special=*/use_chat);
    std::printf("  Prompt: '%s' → %zu tokens\n", final_prompt.c_str(), ids.size());
    if (ids.empty()) { std::fprintf(stderr, "empty prompt\n"); return 2; }

    // 3. Setup device + allocator
    ie::DeviceAllocator alloc;
    if (auto err = alloc.init(); !err.empty()) {
        std::fprintf(stderr, "%salloc: %s%s\n", R, err.c_str(), Z);
        return 1;
    }
    auto& q = alloc.queue();
    std::printf("  Device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    // 4. Load model
    ie::QwenConfig cfg;   // defaults match Qwen3.6-35B-A3B
    ie::QwenModel model;
    const double t_load_start = now_ms();
    std::printf("  Loading weights to device (this can take a moment for ~22 GB)...\n");
    if (auto err = model.load(alloc, g, cfg); !err.empty()) {
        std::fprintf(stderr, "%smodel.load: %s%s\n", R, err.c_str(), Z);
        return 1;
    }
    const double t_load = now_ms() - t_load_start;
    std::printf("  Model loaded in %.1f s\n", t_load / 1000.0);

    // 5. Allocate caches sized for max_ctx
    const uint32_t L_full = cfg.n_layers / cfg.full_attn_interval;     // 10
    const uint32_t L_lin  = cfg.n_layers - L_full;                      // 30
    ie::KvCache kv;
    if (auto err = kv.init(alloc, ie::KvCacheConfig{L_full, cfg.n_kv_heads, max_ctx, cfg.head_dim});
        !err.empty()) {
        std::fprintf(stderr, "%sKvCache: %s%s\n", R, err.c_str(), Z);
        return 1;
    }
    ie::DeltaNetState dn;
    if (auto err = dn.init(alloc, ie::DeltaNetStateConfig{
            L_lin, cfg.ssm_n_v_heads, cfg.ssm_head_dim, cfg.ssm_head_dim,
            cfg.ssm_inner * 2, cfg.ssm_conv_kernel}); !err.empty()) {
        std::fprintf(stderr, "%sDeltaNetState: %s%s\n", R, err.c_str(), Z);
        return 1;
    }
    dn.reset(q);
    std::printf("  Caches ready: KV=%llu MiB, DN-state=%llu MiB\n",
                (unsigned long long)((kv.bytes_per_layer() * L_full) / (1u << 20)),
                (unsigned long long)((uint64_t(L_lin) * dn.state_elems_per_layer() * 4) / (1u << 20)));

    if (auto err = model.ensure_workspace(uint32_t(ids.size()) + max_new + 1); !err.empty()) {
        std::fprintf(stderr, "%sensure_workspace: %s%s\n", R, err.c_str(), Z);
        return 1;
    }

    // 6. Upload input ids
    auto* d_ids    = sycl::malloc_device<int32_t>(ids.size(), q);
    auto* d_logits = sycl::malloc_device<sycl::half>(cfg.vocab, q);
    auto* d_pick   = sycl::malloc_device<int32_t>(1, q);
    q.memcpy(d_ids, ids.data(), ids.size() * sizeof(int32_t)).wait();

    // 7. Prefill
    const double t0 = now_ms();
    std::printf("  Prefill (%zu tokens)...\n", ids.size());
    model.forward(q, d_ids, uint32_t(ids.size()), /*start_pos=*/0, kv, dn, d_logits).wait();
    const double t_prefill = now_ms() - t0;
    std::printf("  Prefill done: %.1f ms (%.1f tok/s)\n",
                t_prefill, ids.size() / (t_prefill / 1000.0));

    // 8. Argmax + decode
    ie::sample_argmax(q, d_logits, d_pick, cfg.vocab).wait();
    int32_t pick;
    q.memcpy(&pick, d_pick, sizeof(int32_t)).wait();
    std::string s = tok.decode(std::vector<int32_t>{pick}, /*skip_special=*/false);
    std::printf("  Argmax token: id=%d  text='%s'\n", pick, s.c_str());

    // Top-5 (host-side over fp16 logits) for diagnostics
    std::vector<sycl::half> hlog(cfg.vocab);
    q.memcpy(hlog.data(), d_logits, cfg.vocab * sizeof(sycl::half)).wait();
    struct Pair { int32_t id; float v; };
    std::vector<Pair> top;
    top.reserve(cfg.vocab);
    for (uint32_t i = 0; i < cfg.vocab; ++i) top.push_back({int32_t(i), float(hlog[i])});
    std::partial_sort(top.begin(), top.begin() + 5, top.end(),
                      [](const Pair& a, const Pair& b){ return a.v > b.v; });
    std::printf("  Top-5:");
    for (int i = 0; i < 5; ++i) {
        const auto t = tok.decode(std::vector<int32_t>{top[i].id}, false);
        std::string clean;
        for (char c : t) clean += (c >= 0x20 && c <= 0x7E) ? c : '?';
        std::printf("  %d='%s'(%.2f)", top[i].id, clean.c_str(), top[i].v);
    }
    std::putchar('\n');

    // 9. Optional generation loop
    if (max_new > 1) {
        std::printf("  Decoding %d more tokens (greedy)...\n", max_new - 1);
        std::string out_text = s;
        std::vector<int32_t> generated{pick};
        uint32_t pos = uint32_t(ids.size());
        for (int n = 1; n < max_new; ++n) {
            q.memcpy(d_ids, &pick, sizeof(int32_t)).wait();
            model.forward(q, d_ids, /*T=*/1, /*start_pos=*/pos, kv, dn, d_logits).wait();
            ie::sample_argmax(q, d_logits, d_pick, cfg.vocab).wait();
            q.memcpy(&pick, d_pick, sizeof(int32_t)).wait();
            ++pos;
            generated.push_back(pick);
            if (tok.is_special(pick) && pick == tok.eos_token_id()) break;
        }
        const auto whole = tok.decode(std::span<const int32_t>(generated), false);
        std::printf("  Generated: %s'%s'%s\n", G, whole.c_str(), Z);
    }

    sycl::free(d_ids, q); sycl::free(d_logits, q); sycl::free(d_pick, q);
    return 0;
}
