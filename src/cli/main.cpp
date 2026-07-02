// ie — run | serve | bench | import.  Thin front-end over ie::Engine.
#include "ie/engine.hpp"
#include "ie/openai_server.hpp"
#include "ie/hf_import.hpp"
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {
const char* USAGE =
  "usage: ie <run|serve> <model.gguf> [--ctx N] [--port P] [--host H] [--gpus N] [--spec]\n"
  "                                (--spec = MTP self-speculative GREEDY decode for the\n"
  "                                 Qwen3.6-27B (kQwen35Dense, single-GPU); always LOSSLESS.\n"
  "                                 Faster only when the verify forward is cheap: ~1.1-1.2x on\n"
  "                                 Q8_0-quant MTP heads, but it REGRESSES on Q4_K_M (the Q4_K\n"
  "                                 verify is dequant-bound; see docs/authority/qwen35-27b.md).\n"
  "                                 Optional --spec-k K (default 4))\n"
  "                                (--gpus N>1 = split across N GPUs: tensor-parallel\n"
  "                                 for dense archs, layer-split for Qwen3-Next-80B —\n"
  "                                 runs models too big for one card, e.g. Qwen2.5-72B\n"
  "                                 or Qwen3-Next-80B across 2x B70)\n"
  "       ie pull   <name|hf-repo> [file.gguf]   (fetch a GGUF; `ie pull --list` for names)\n"
  "       ie bench  <model.gguf>   (runs ie-bench from the sibling tools/ build)\n"
  "       ie import <hf_dir> <out.gguf> <tokenizer_ref.gguf>\n"
  "                                (convert an AWQ safetensors checkpoint to GGUF;\n"
  "                                 the ref supplies tokenizer KVs of the family)\n";

std::string model_id_from(const std::string& path) {
    const auto s = path.find_last_of("/\\");
    const auto base = path.substr(s == std::string::npos ? 0 : s + 1);
    const auto e = base.rfind(".gguf");
    return e == std::string::npos ? base : base.substr(0, e);
}

// ie lands in build/src/, ie-bench in build/tools/.  Prefer the sibling
// build path (resolved via the invoked argv[0]); fall back to PATH.
std::string find_ie_bench(const char* argv0) {
    const std::string self(argv0);
    const auto slash = self.find_last_of("/\\");
    if (slash != std::string::npos) {
        const auto cand = self.substr(0, slash + 1) + "../tools/ie-bench";
        std::error_code ec;
        if (std::filesystem::exists(cand, ec)) return cand;
    }
    return "ie-bench";   // hope it's on PATH
}

// Locate the ie-pull helper: next to `ie` (Docker/install layout), then the dev
// tree (build/src/ie → scripts/ie-pull), else PATH.
std::string find_ie_pull(const char* argv0) {
    const std::string self(argv0);
    const auto slash = self.find_last_of("/\\");
    if (slash != std::string::npos) {
        const std::string dir = self.substr(0, slash + 1);
        std::error_code ec;
        for (const char* rel : {"ie-pull", "../../scripts/ie-pull", "../scripts/ie-pull"}) {
            if (std::filesystem::exists(dir + rel, ec)) return dir + rel;
        }
    }
    return "ie-pull";   // hope it's on PATH
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fputs(USAGE, stderr); return 2; }
    const std::string cmd = argv[1];

    // ie import <hf_dir> <out.gguf> <tokenizer_ref.gguf> — AWQ→GGUF, no engine load.
    if (cmd == "import") {
        if (argc < 5) { std::fputs(USAGE, stderr); return 2; }
        std::string log;
        const std::string e = ie::import_awq_to_gguf(argv[2], argv[3], argv[4], &log);
        std::fputs(log.c_str(), stdout);
        if (!e.empty()) { std::fprintf(stderr, "import failed: %s\n", e.c_str()); return 1; }
        std::printf("OK: %s\n", argv[3]);
        return 0;
    }

    // ie pull <name|repo> [file] — fetch a GGUF (delegates to the ie-pull helper).
    if (cmd == "pull") {
        std::string c = "\"" + find_ie_pull(argv[0]) + "\"";
        for (int i = 2; i < argc; ++i) { c += " \""; c += argv[i]; c += "\""; }
        return std::system(c.c_str());
    }

    if (argc < 3) { std::fputs(USAGE, stderr); return 2; }
    const std::string model_path = argv[2];
    ie::EngineOptions opts;
    opts.n_gpus = 0;                        // 0 = auto: sense VRAM, pick 1 vs N GPUs.
                                            // --gpus N forces a specific count.
    std::string host = "127.0.0.1";
    int port = 11435;                       // one above ollama's 11434
    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--ctx"  && i + 1 < argc) opts.max_ctx = std::atoi(argv[++i]);
        else if (a == "--port" && i + 1 < argc) port = std::atoi(argv[++i]);
        else if (a == "--host" && i + 1 < argc) host = argv[++i];
        else if (a == "--gpus" && i + 1 < argc) opts.n_gpus = uint32_t(std::atoi(argv[++i]));
        else if (a == "--int8-kv")               opts.int8_kv = true;
        else if (a == "--spec")                 opts.spec = true;
        else if (a == "--spec-k" && i + 1 < argc) opts.spec_k = uint32_t(std::atoi(argv[++i]));
        else if (a == "--spec-head" && i + 1 < argc) opts.spec_head = argv[++i];
    }
    if (cmd == "bench") {
        const std::string c =
            "\"" + find_ie_bench(argv[0]) + "\" --gguf \"" + model_path + "\"";
        return std::system(c.c_str());
    }
    if (cmd != "run" && cmd != "serve") { std::fputs(USAGE, stderr); return 2; }

    std::string err;
    auto eng = ie::Engine::load(model_path, opts, err);
    if (!eng) { std::fprintf(stderr, "load failed: %s\n", err.c_str()); return 1; }

    if (cmd == "serve")
        return ie::run_openai_server(*eng, model_id_from(model_path), host, port);

    // cmd == "run"
    std::vector<ie::ChatTurn> turns;
    std::printf("ie chat — /reset clears history, /quit exits\n");
    std::string line;
    while (std::printf("> "), std::fflush(stdout), std::getline(std::cin, line)) {
        if (line == "/quit") break;
        if (line == "/reset") { turns.clear(); continue; }
        if (line.empty()) continue;
        turns.push_back({"user", line});
        ie::SamplingParams sp;
        // Interactive chat: don't truncate replies at the library default
        // (512); run until EOS or the ctx budget, like the server's
        // omitted-max_tokens semantics.
        sp.max_tokens = ie::kMaxTokensUnlimited;
        // --spec is GREEDY-only (the lossless guarantee holds for argmax decode);
        // force temperature 0 so the spec path actually engages instead of silently
        // falling back to sampled decode.
        if (opts.spec) sp.temperature = 0.0f;
        std::string out;
        ie::GenerateResult res = eng->chat(turns, sp, [&](std::string_view t) {
            std::fwrite(t.data(), 1, t.size(), stdout);
            std::fflush(stdout);
            out.append(t);
            return true;
        }, /*enable_thinking=*/false);
        std::printf("\n");
        if (const char* p = std::getenv("IE_PERF"); p && res.decode_ms > 0)
            std::fprintf(stderr,
                "[perf] prefill %u tok %.1f ms (%.1f tok/s) | decode %u tok %.1f ms = %.1f tok/s\n",
                res.prompt_tokens, res.prefill_ms,
                1000.0 * double(res.prompt_tokens) / res.prefill_ms,
                res.completion_tokens, res.decode_ms,
                1000.0 * double(res.completion_tokens) / res.decode_ms);
        turns.push_back({"assistant", out});
    }
    return 0;
}
