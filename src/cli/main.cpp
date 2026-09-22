// ie — run | serve | bench | import.  Thin front-end over ie::Engine.
#include "ie/engine.hpp"
#include "ie/openai_server.hpp"
#include "ie/hf_import.hpp"
#include "ie/preflight.hpp"
#include "ie/gguf.hpp"
#include "ie/serve_options.hpp"
#include "ie/server_capabilities.hpp"
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <malloc.h>
#include <string>
#include <vector>

namespace {
const char* USAGE =
  "usage: ie <run|serve> <model.gguf> [--ctx N] [--port P] [--host H] [--gpus N]\n"
  "                                [--prefill-chunk N] [--spec] [--parallel N] [--temp F]\n"
  "                                (--parallel N = 1..4 concurrent generations; >1\n"
  "                                 interleaves requests on the 27B 2-GPU split path\n"
  "                                 via host-staged slot switching, other archs run\n"
  "                                 whole generations in FIFO turns. Default 1.\n"
  "                                 Not yet compatible with --int8-kv.)\n"
  "                                (--max-queue N = requests allowed to WAIT for a generation\n"
  "                                 slot; beyond that the server answers HTTP 429 at once.\n"
  "                                 Default 8. /health reports inflight/queued.)\n"
  "                                (--prefill-chunk = tokens per prefill forward, default 256.\n"
  "                                 On DeepSeek-V4 this is the largest T any forward submits;\n"
  "                                 1024 measured pp512 72 -> 105 tok/s BUT needs a lowered\n"
  "                                 expert-slot count, which is still bench-only, or it OOMs.)\n"
  "                                (--spec = MTP self-speculative GREEDY decode for the\n"
  "                                 Qwen3.6-27B (kQwen35Dense, single-GPU); always LOSSLESS.\n"
  "                                 Faster only when the verify forward is cheap: ~1.1-1.2x on\n"
  "                                 Q8_0-quant MTP heads, but it REGRESSES on Q4_K_M (the Q4_K\n"
  "                                 verify is dequant-bound; see docs/authority/qwen35-27b.md).\n"
  "                                 Optional --spec-k K (default 4))\n"
  "                                (--spec-draft <dspark.gguf> = separate-draft spec decode\n"
  "                                 for the Qwen3.6-27B (kQwen35Dense, single-GPU) using a\n"
  "                                 target-conditioned dspark drafter; always LOSSLESS greedy)\n"
  "                                (--gpus N>1 = split across N GPUs: tensor-parallel\n"
  "                                 for dense archs, layer-split for Qwen3-Next-80B —\n"
  "                                 runs models too big for one card, e.g. Qwen2.5-72B\n"
  "                                 or Qwen3-Next-80B across 2x B70)\n"
  "                                (DeepSeek-V4-Flash: pass the …-00001-of-NNNNN.gguf\n"
  "                                 shard — the engine pulls in the siblings. --gpus\n"
  "                                 defaults to EVERY card (expert tensor-parallel);\n"
  "                                 routed experts stream from pinned host RAM, so the\n"
  "                                 file being far larger than VRAM is expected)\n"
  "       Sampling defaults (requests may override):\n"
  "         --temp 0..2 --top-k 0..1024 --top-p (0,1] --min-p 0..1\n"
  "         --repeat-penalty (0,10] --repeat-last-n 0..512\n"
  "         --presence-penalty -2..2 --frequency-penalty -2..2\n"
  "         --seed N (0=random) --max-tokens N (0=until context limit)\n"
  "         --stop STRING (repeat up to 4) --thinking on|off\n"
  "         --reasoning-effort LEVEL (choices depend on model; see capabilities)\n"
  "       Load controls: --threads 1..1024 --no-prompt-cache --slot-ctx N\n"
  "       top-k 0 uses the sampler ceiling of 1024; history penalties share\n"
  "       repeat-last-n over prompt+output (0 disables all history penalties).\n"
  "       ie capabilities [model.gguf] (JSON, no GPU load)\n"
  "       ie pull   <name|hf-repo> [file.gguf]   (fetch a GGUF; `ie pull --list` for names)\n"
  "       ie bench  <model.gguf>   (runs ie-bench from the sibling tools/ build)\n"
  "       ie preflight <model.gguf> [--ctx N] [--gpus N] [--int8-kv]\n"
  "                                (metadata-only: instant load/VRAM verdict\n"
  "                                 BEFORE any upload; exit 0=will load, 3=will not)\n"
  "       ie import <hf_dir> <out.gguf> <tokenizer_ref.gguf>\n"
  "                                (convert an AWQ/GPTQ/EXL3 checkpoint to GGUF;\n"
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

    // ie serve: pin glibc's mmap threshold at 256 KiB. Left dynamic it climbs to the largest
    // block freed so far, after which every request's big host buffers come from the heap and
    // their freed space stays resident: DS4.1 100-token requests grew RssAnon 42-54 MiB each
    // (free-in-arenas +38-40), with the pin 16-18 (+1-3); prefill and decode unchanged.
    // An explicit GLIBC_TUNABLES wins.
    if (cmd == "serve" && !std::getenv("GLIBC_TUNABLES")) mallopt(M_MMAP_THRESHOLD, 256 * 1024);

    if (cmd == "capabilities") {
        if (argc > 3) { std::fputs("usage: ie capabilities [model.gguf]\n", stderr); return 2; }
        try { std::cout << ie::server_capabilities_json(argc == 3 ? argv[2] : "") << '\n'; }
        catch (const std::exception& e) { std::fprintf(stderr, "capabilities: %s\n", e.what()); return 2; }
        return 0;
    }

    // ie import <hf_dir> <out.gguf> <tokenizer_ref.gguf> — no engine load.
    if (cmd == "import") {
        if (argc < 5) { std::fputs(USAGE, stderr); return 2; }
        std::string log;
        const std::string e = ie::import_hf_to_gguf(argv[2], argv[3], argv[4], &log);
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
    ie::LaunchOptions launch;
    try { launch = ie::parse_launch_options(std::vector<std::string>(argv + 3, argv + argc)); }
    catch (const std::exception& e) {
        std::fprintf(stderr, "invalid options: %s\n", e.what()); return 2;
    }
    auto& opts = launch.engine;
    const auto& host = launch.host;
    const int port = launch.port;
    ie::oai::configure_server_defaults(launch.defaults);
    if (opts.cpu_threads) {
        const auto n = std::to_string(opts.cpu_threads);
        setenv("OMP_NUM_THREADS", n.c_str(), 1);
        setenv("IE_CPU_THREADS", n.c_str(), 1);
    }
    if (cmd == "bench") {
        const std::string c =
            "\"" + find_ie_bench(argv[0]) + "\" --gguf \"" + model_path + "\"";
        return std::system(c.c_str());
    }

    // ie preflight <model.gguf> — metadata-only capability report. Opens the
    // GgufReader like Engine::load does, then reports without any upload/kernel.
    // Exit 0 if the model will load, 3 if a blocking dtype/arch/VRAM issue found.
    if (cmd == "preflight") {
        ie::GgufReader g;
        if (auto m = g.open(model_path); !m.empty()) {
            std::fprintf(stderr, "preflight: cannot open gguf: %s\n", m.c_str());
            return 1;
        }
        ie::PreflightReport rep =
            ie::preflight_model(g, opts.max_ctx, opts.int8_kv, opts.n_gpus);
        std::fputs(rep.text.c_str(), stdout);
        // deepseek4 is the one arch whose VRAM verdict above is meaningless. The
        // shared planner sums every GGUF tensor against per-card VRAM — 150.7 GB
        // vs 28.7 GB, so it always answers WILL NOT LOAD — but this arch is built
        // to keep only its always-resident set plus an expert slot cache in VRAM
        // and stream the 256 routed experts from a pinned host arena. The real
        // fit check is ds4_plan_residency_tp, inside the runtime's own load,
        // which refuses an over-cap plan by name. Say so, and do NOT hand back
        // exit 3 (= "will not load"), which is simply the wrong answer here.
        if (ie::detect_arch(g) == ie::ModelArch::kDeepSeek4) {
            std::fputs(
              "\nNOTE (deepseek4): IGNORE the VRAM verdict above. This arch does not hold its\n"
              "  weights in VRAM: only the always-resident set plus a routed-expert slot cache\n"
              "  is device-side, and the experts stream from pinned host RAM. The shared\n"
              "  planner cannot model that, so it compares the whole file against one card and\n"
              "  always says WILL NOT LOAD. The binding check is done at load by\n"
              "  ds4_plan_residency_tp, which refuses an over-cap plan loudly. Exit code is 0\n"
              "  because this preflight has NOT determined whether the model fits.\n", stdout);
            return 0;
        }
        return rep.will_load ? 0 : 3;
    }

    if (cmd != "run" && cmd != "serve") { std::fputs(USAGE, stderr); return 2; }

    std::string err;
    err=ie::server_reasoning_error(model_path,launch.defaults.reasoning_effort);
    if(!err.empty()){std::fprintf(stderr,"invalid model setting: %s\n",err.c_str());return 2;}
    auto eng = ie::Engine::load(model_path, opts, err);
    if (!eng) { std::fprintf(stderr, "load failed: %s\n", err.c_str()); return 1; }

    if (cmd == "serve")
        return ie::run_openai_server(*eng, model_id_from(model_path), host, port,
                                     launch.max_queue);

    // cmd == "run"
    std::vector<ie::ChatTurn> turns;
    std::printf("ie chat — /reset clears history, /quit exits\n");
    std::string line;
    while (std::printf("> "), std::fflush(stdout), std::getline(std::cin, line)) {
        if (line == "/quit") break;
        if (line == "/reset") { turns.clear(); continue; }
        if (line.empty()) continue;
        turns.push_back({"user", line});
        ie::SamplingParams sp = launch.defaults.sampling;
        // --spec is GREEDY-only (the lossless guarantee holds for argmax decode);
        // force temperature 0 so the spec path actually engages instead of silently
        // falling back to sampled decode.
        if (opts.spec || !opts.spec_draft.empty()) sp.temperature = 0.0f;
        std::string out, pending;
        bool stopped = false;
        size_t hold = 0;
        for (const auto& stop : launch.defaults.stop) hold = std::max(hold, stop.size() - 1);
        auto emit = [&](size_t count) {
            std::fwrite(pending.data(), 1, count, stdout);
            std::fflush(stdout);
            out.append(pending, 0, count);
            pending.erase(0, count);
        };
        ie::GenerateResult res = eng->chat(turns, sp, [&](std::string_view t) {
            pending.append(t);
            size_t cut = std::string::npos;
            for (const auto& stop : launch.defaults.stop) cut = std::min(cut, pending.find(stop));
            if (cut != std::string::npos) { emit(cut); stopped = true; return false; }
            if (pending.size() > hold) emit(pending.size() - hold);
            return true;
        }, launch.defaults.enable_thinking, {}, launch.defaults.reasoning_effort);
        if (!stopped) emit(pending.size());
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
