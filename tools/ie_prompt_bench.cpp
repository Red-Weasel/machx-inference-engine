// ie-prompt-bench — the REAL publish harness (OURS side).
//
// Drives the product path (ie::Engine — every arch, real tokenizer, chat-free
// raw-prompt prefill, optional spec-decode, auto multi-GPU) over the fixed
// 5-prompt suite in benchmark_prompts/. For each prompt it measures the SAME
// two axes llama-cli reports, the SAME way:
//   prefill tok/s = prompt_tokens     / prefill_ms
//   decode  tok/s = completion_tokens / decode_ms
// Greedy (temperature 0 → argmax; spec is lossless-greedy when --spec). Prompt
// cache is FORCED OFF so every run re-prefills (honest pp; no cache-skip).
// 3 runs, run 1 discarded (JIT/first-load warmup), median of the rest reported
// per prompt + the median across the 5. Identical protocol to the llama side
// (llama-cli -f <prompt> -n <N> --temp 0); see tools/bench5.sh.
//
// usage: ie-prompt-bench --gguf <model> [--gpus 0] [--spec] [--spec-k K]
//        [--spec-head <mtp.gguf>] [--decode 128] [--runs 3] [--warmup 1]
//        [--max-ctx 16384] [--prompts-dir benchmark_prompts]

#include "ie/engine.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace {
std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::ostringstream ss; ss << f.rdbuf();
    return ss.str();
}
double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return (n & 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}
}  // namespace

int main(int argc, char** argv) {
    std::string gguf, spec_head, prompts_dir = "benchmark_prompts";
    ie::EngineOptions opts;
    opts.n_gpus       = 0;       // auto: sense VRAM, pick 1 vs N (80B → 2 cards)
    opts.max_ctx      = 16384;
    opts.prompt_cache = false;   // CRITICAL: re-prefill every run (honest pp)
    uint32_t decode_n = 128, runs = 3, warmup = 1;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--gguf"        && i + 1 < argc) gguf        = argv[++i];
        else if (a == "--gpus"        && i + 1 < argc) opts.n_gpus = uint32_t(std::atoi(argv[++i]));
        else if (a == "--spec")                        opts.spec   = true;
        else if (a == "--spec-k"      && i + 1 < argc) opts.spec_k = uint32_t(std::atoi(argv[++i]));
        else if (a == "--spec-head"   && i + 1 < argc) { spec_head = argv[++i]; opts.spec_head = spec_head; }
        else if (a == "--decode"      && i + 1 < argc) decode_n    = uint32_t(std::atoi(argv[++i]));
        else if (a == "--runs"        && i + 1 < argc) runs        = uint32_t(std::atoi(argv[++i]));
        else if (a == "--warmup"      && i + 1 < argc) warmup      = uint32_t(std::atoi(argv[++i]));
        else if (a == "--max-ctx"     && i + 1 < argc) opts.max_ctx = uint32_t(std::atoi(argv[++i]));
        else if (a == "--prompts-dir" && i + 1 < argc) prompts_dir = argv[++i];
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
    }
    if (gguf.empty()) { std::fprintf(stderr, "need --gguf\n"); return 2; }
    if (runs <= warmup) { std::fprintf(stderr, "runs must exceed warmup\n"); return 2; }

    // Collect + sort the prompt files (01_.. .. 05_..).
    std::vector<std::string> files;
    for (auto& e : std::filesystem::directory_iterator(prompts_dir))
        if (e.path().extension() == ".txt") files.push_back(e.path().string());
    std::sort(files.begin(), files.end());
    if (files.empty()) { std::fprintf(stderr, "no .txt prompts in %s\n", prompts_dir.c_str()); return 2; }

    std::string err;
    auto eng = ie::Engine::load(gguf, opts, err);
    if (!eng) { std::fprintf(stderr, "load: %s\n", err.c_str()); return 1; }

    std::printf("ie-prompt-bench | gguf=%s\n", gguf.c_str());
    std::printf("  config: gpus=%u(0=auto) spec=%s K=%u decode=%u runs=%u(warmup %u discarded) max_ctx=%u prompt_cache=OFF greedy(temp0)\n",
                opts.n_gpus, opts.spec ? "ON" : "off", opts.spec_k, decode_n, runs, warmup, opts.max_ctx);
    std::printf("  %-22s %10s %10s %8s %8s\n", "prompt", "pp_tok/s", "tg_tok/s", "n_pp", "n_tg");

    ie::SamplingParams sp;
    sp.temperature = 0.0f;        // greedy / argmax (spec lossless-greedy when --spec)
    sp.max_tokens  = decode_n;
    sp.ignore_eos  = true;        // FIXED decode budget: never stop on eos (== llama
                                  // --ignore-eos) so tg is measured over exactly decode_n
                                  // tokens for every prompt (short prompts won't 0 out).

    std::vector<double> pp_med, tg_med;
    for (const auto& path : files) {
        const std::string text = read_file(path);
        std::vector<double> pps, tgs;
        for (uint32_t r = 0; r < runs; ++r) {
            ie::GenerateResult res = eng->generate(text, sp);
            if (r < warmup) continue;          // discard JIT/first-load runs
            const double pp = res.prefill_ms > 0 ? res.prompt_tokens     / (res.prefill_ms / 1000.0) : 0.0;
            const double tg = res.decode_ms  > 0 ? res.completion_tokens / (res.decode_ms  / 1000.0) : 0.0;
            pps.push_back(pp); tgs.push_back(tg);
        }
        const double pp = median(pps), tg = median(tgs);
        pp_med.push_back(pp); tg_med.push_back(tg);
        const std::string name = std::filesystem::path(path).filename().string();
        // n_pp/n_tg from the last run (deterministic across runs at temp 0).
        std::printf("  %-22s %10.1f %10.2f\n", name.c_str(), pp, tg);
    }
    std::printf("  %-22s %10.1f %10.2f   <- MEDIAN across 5 prompts\n",
                "MEDIAN", median(pp_med), median(tg_med));
    std::printf("# TSV\tmedian_pp\tmedian_tg\t%.2f\t%.3f\n", median(pp_med), median(tg_med));
    return 0;
}
