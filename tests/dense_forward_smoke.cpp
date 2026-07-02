// tests/dense_forward_smoke.cpp — P2 Task 4: GPU smoke for DenseModel::forward.
//
// Loads the qwen3-8b GGUF (skip-with-warning if absent, CI-safe), prefills a
// short prompt (T=8-ish) and greedily decodes 4 more tokens.  Asserts:
//   * load / forward report no error and throw no SYCL exception,
//   * every step's logits are all finite (no NaN/Inf),
//   * argmax is a plausible token id (in range, never id 0 / never the same
//     degenerate id on every step).
// Correctness vs llama.cpp (layer parity, greedy parity, PPL) is Task 7 —
// this gate only proves the forward runs clean end to end.

#include "ie/allocator.hpp"
#include "ie/dense_transformer.hpp"
#include "ie/gguf.hpp"
#include "ie/kv_cache.hpp"
#include "ie/model_config.hpp"
#include "ie/ops.hpp"
#include "ie/tokenizer.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

constexpr const char* G = "\033[32m";
constexpr const char* R = "\033[31m";
constexpr const char* Y = "\033[33m";
constexpr const char* Z = "\033[0m";

std::string default_gguf() {
    if (const char* env = std::getenv("IE_QWEN3_GGUF")) return env;
    const char* home = std::getenv("HOME");
    return std::string(home ? home : "") + "/.seal/models/Qwen3-8B-Q4_K_M.gguf";
}

// Returns the number of non-finite logits; fills argmax id/value.
uint32_t scan_logits(sycl::queue& q, const sycl::half* d_logits, uint32_t vocab,
                     int32_t& argmax_id, float& argmax_v) {
    std::vector<sycl::half> h(vocab);
    q.memcpy(h.data(), d_logits, vocab * sizeof(sycl::half)).wait();
    uint32_t bad = 0;
    argmax_id = -1;
    argmax_v = -1e30f;
    for (uint32_t i = 0; i < vocab; ++i) {
        const float v = float(h[i]);
        if (!std::isfinite(v)) { ++bad; continue; }
        if (v > argmax_v) { argmax_v = v; argmax_id = int32_t(i); }
    }
    return bad;
}

}  // namespace

int main() {
    setvbuf(stdout, nullptr, _IOLBF, 0);  // line-buffered: keep prints in
                                          // order with stderr on failure
    const std::string gguf_path = default_gguf();

    ie::GgufReader g;
    if (auto err = g.open(gguf_path); !err.empty()) {
        std::printf("%sSKIP%s dense_forward_smoke: cannot open %s (%s)\n",
                    Y, Z, gguf_path.c_str(), err.c_str());
        return 0;  // CI-safe skip: model not on this box
    }
    if (ie::detect_arch(g) != ie::ModelArch::kQwen3Dense) {
        std::fprintf(stderr, "%sFAIL%s %s is not a qwen3-dense GGUF\n",
                     R, Z, gguf_path.c_str());
        return 1;
    }
    ie::DenseConfig cfg;
    if (auto err = ie::read_dense_config(g, "qwen3", cfg); !err.empty()) {
        std::fprintf(stderr, "%sFAIL%s read_dense_config: %s\n", R, Z, err.c_str());
        return 1;
    }
    std::printf("dense_forward_smoke: %s\n  n_layers=%u hidden=%u q/kv=%u/%u "
                "hd=%u ffn=%u vocab=%u rope_dim=%u theta=%g\n",
                gguf_path.c_str(), cfg.n_layers, cfg.hidden, cfg.n_q_heads,
                cfg.n_kv_heads, cfg.head_dim, cfg.ffn, cfg.vocab, cfg.rope_dim,
                double(cfg.rope_theta));

    ie::Tokenizer tok;
    if (auto err = tok.load_from_gguf(g); !err.empty()) {
        std::fprintf(stderr, "%sFAIL%s tokenizer: %s\n", R, Z, err.c_str());
        return 1;
    }

    ie::DeviceAllocator alloc;
    if (auto err = alloc.init(); !err.empty()) {
        std::printf("%sSKIP%s dense_forward_smoke: no device (%s)\n",
                    Y, Z, err.c_str());
        return 0;
    }
    auto& q = alloc.queue();

    ie::DenseModel model;
    if (auto err = model.load(alloc, g, cfg); !err.empty()) {
        std::fprintf(stderr, "%sFAIL%s model.load: %s\n", R, Z, err.c_str());
        return 1;
    }
    std::printf("  model loaded\n");

    constexpr uint32_t kMaxCtx = 256;
    ie::KvCache kv;
    if (auto err = kv.init(alloc, ie::KvCacheConfig{cfg.n_layers, cfg.n_kv_heads,
                                                    kMaxCtx, cfg.head_dim});
        !err.empty()) {
        std::fprintf(stderr, "%sFAIL%s kv.init: %s\n", R, Z, err.c_str());
        return 1;
    }

    const std::string prompt = "The capital of France is";
    auto ids = tok.encode(prompt, /*allow_special=*/false);
    if (ids.empty()) { std::fprintf(stderr, "%sFAIL%s empty prompt encode\n", R, Z); return 1; }
    std::printf("  prompt '%s' -> %zu tokens\n", prompt.c_str(), ids.size());

    if (auto err = model.ensure_workspace(uint32_t(ids.size())); !err.empty()) {
        std::fprintf(stderr, "%sFAIL%s ensure_workspace: %s\n", R, Z, err.c_str());
        return 1;
    }
    if (auto err = model.ensure_attn_partials(kMaxCtx); !err.empty()) {
        std::fprintf(stderr, "%sFAIL%s ensure_attn_partials: %s\n", R, Z, err.c_str());
        return 1;
    }

    auto* d_ids    = sycl::malloc_device<int32_t>(ids.size(), q);
    auto* d_logits = sycl::malloc_device<sycl::half>(cfg.vocab, q);
    if (!d_ids || !d_logits) { std::fprintf(stderr, "%sFAIL%s device alloc\n", R, Z); return 1; }
    q.memcpy(d_ids, ids.data(), ids.size() * sizeof(int32_t)).wait();

    bool ok = true;
    std::vector<int32_t> picks;
    try {
        // Prefill all prompt tokens, then 4 greedy decode steps.
        model.forward(q, d_ids, uint32_t(ids.size()), /*start_pos=*/0, kv,
                      d_logits).wait_and_throw();
        uint32_t pos = uint32_t(ids.size());
        for (int step = 0; step < 5; ++step) {
            int32_t pick; float v;
            const uint32_t bad = scan_logits(q, d_logits, cfg.vocab, pick, v);
            const char* what = step == 0 ? "prefill" : "decode";
            const bool step_ok = (bad == 0) && pick > 0 && uint32_t(pick) < cfg.vocab;
            ok = ok && step_ok;
            auto txt = tok.decode(std::vector<int32_t>{pick}, false);
            std::string clean;
            for (char c : txt) clean += (c >= 0x20 && c <= 0x7E) ? c : '?';
            std::printf("  %s%s step %d%s: argmax id=%d '%s' logit=%.3f "
                        "non-finite=%u  %s\n",
                        step_ok ? G : R, what, step, Z, pick, clean.c_str(),
                        double(v), bad, step_ok ? "OK" : "FAIL");
            picks.push_back(pick);
            if (step == 4) break;
            q.memcpy(d_ids, &pick, sizeof(int32_t)).wait();
            model.forward(q, d_ids, /*T=*/1, pos, kv, d_logits).wait_and_throw();
            ++pos;
        }
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "%sFAIL%s SYCL exception: %s\n", R, Z, e.what());
        std::fflush(stdout);
        // The queue/context is wedged after a device-side failure — model
        // dtor / sycl::free would hang. Hard-exit with the failure code.
        std::_Exit(1);
    }

    // Degenerate-output guard: 5 identical picks = garbage forward.
    if (ok && picks.size() == 5) {
        bool all_same = true;
        for (auto p : picks) all_same = all_same && (p == picks[0]);
        if (all_same) {
            std::fprintf(stderr, "%sFAIL%s all 5 argmax picks identical (id=%d)"
                         " — degenerate forward\n", R, Z, picks[0]);
            ok = false;
        }
    }
    const auto whole = tok.decode(std::span<const int32_t>(picks), false);
    std::printf("  greedy continuation: '%s'\n", whole.c_str());

    sycl::free(d_ids, q); sycl::free(d_logits, q);
    std::printf("%s\n", ok ? "dense_forward_smoke: ALL OK" : "dense_forward_smoke: FAILED");
    return ok ? 0 : 1;
}
