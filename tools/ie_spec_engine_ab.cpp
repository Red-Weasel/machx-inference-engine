// tools/ie_spec_engine_ab.cpp — IN-ENGINE lossless + speedup A/B for --spec.
//
// Loads ie::Engine TWICE (serialized — one GPU at a time): once plain, once with
// EngineOptions.spec=true, and runs generate() at temperature 0 (greedy) on the
// same RAW prompt. Diffs the emitted text token-for-token and reports tok/s for
// both. This exercises the PRODUCTION Engine::generate spec path (not the
// standalone ie-qwen35-spec tool).
//
// usage: ie-spec-engine-ab --gguf <27b.gguf> [--prompt <text>] [--ntok N] [--K K]
#include "ie/engine.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    std::string gguf =
        "/home/weezy/models/bartowski/Qwen3.6-27B-GGUF/Qwen_Qwen3.6-27B-Q4_K_M.gguf";
    std::string prompt =
        "You are a cybersecurity expert. Explain in detail how a buffer overflow "
        "vulnerability works and how an attacker can exploit it to gain control.";
    uint32_t ntok = 64, K = 4;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--gguf"   && i + 1 < argc) gguf   = argv[++i];
        else if (a == "--prompt" && i + 1 < argc) prompt = argv[++i];
        else if (a == "--ntok"   && i + 1 < argc) ntok   = std::atoi(argv[++i]);
        else if (a == "--K"      && i + 1 < argc) K      = std::atoi(argv[++i]);
    }

    auto run = [&](bool spec, std::vector<int32_t>& ids_out, double& tps,
                   std::string& text) -> int {
        ie::EngineOptions opts;
        opts.max_ctx = 4096;
        opts.spec = spec;
        opts.spec_k = K;
        opts.prompt_cache = false;
        std::string err;
        auto eng = ie::Engine::load(gguf, opts, err);
        if (!eng) { std::fprintf(stderr, "load (%s): %s\n", spec ? "spec" : "plain", err.c_str()); return 1; }
        ie::SamplingParams sp;
        sp.temperature = 0.0f;     // greedy → spec engages + deterministic A/B
        sp.max_tokens = ntok;
        // warm-up
        eng->generate(prompt, sp, {});
        // measured
        text.clear();
        ie::GenerateResult res = eng->generate(prompt, sp, [&](std::string_view t) {
            text.append(t); return true;
        });
        // re-encode the produced text to compare token streams robustly
        ids_out = eng->tokenizer().encode(text, /*allow_special=*/false);
        tps = res.decode_ms > 0 ? 1000.0 * double(res.completion_tokens) / res.decode_ms : 0.0;
        std::printf("  [%s] completion=%u  prefill=%.1f ms  decode=%.1f ms  -> %.2f tok/s  finish=%s\n",
                    spec ? "spec " : "plain", res.completion_tokens, res.prefill_ms,
                    res.decode_ms, tps, res.finish_reason.c_str());
        return 0;
    };

    std::printf("ie-spec-engine-ab  ntok=%u  K=%u\n  gguf=%s\n", ntok, K, gguf.c_str());

    std::vector<int32_t> plain_ids, spec_ids;
    std::string plain_txt, spec_txt;
    double plain_tps = 0, spec_tps = 0;

    std::printf("\n[plain greedy]\n");
    if (run(false, plain_ids, plain_tps, plain_txt)) return 1;
    std::printf("\n[spec greedy]\n");
    if (run(true,  spec_ids,  spec_tps,  spec_txt)) return 1;

    bool lossless = (plain_txt == spec_txt);
    std::printf("\n=== RESULT ===\n");
    std::printf("  LOSSLESS (text identical) : %s\n", lossless ? "YES" : "NO");
    if (!lossless) {
        size_t n = std::min(plain_ids.size(), spec_ids.size());
        size_t fd = n;
        for (size_t i = 0; i < n; ++i) if (plain_ids[i] != spec_ids[i]) { fd = i; break; }
        std::printf("  token streams: plain=%zu spec=%zu  first diff @ %zu\n",
                    plain_ids.size(), spec_ids.size(), fd);
        std::printf("  --- plain text ---\n%s\n", plain_txt.c_str());
        std::printf("  --- spec text ----\n%s\n", spec_txt.c_str());
    }
    std::printf("  plain  : %.2f tok/s\n", plain_tps);
    std::printf("  spec   : %.2f tok/s\n", spec_tps);
    std::printf("  net    : %.2fx\n", plain_tps > 0 ? spec_tps / plain_tps : 0.0);
    return lossless ? 0 : 2;
}
