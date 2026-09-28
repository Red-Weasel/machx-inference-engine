// ie — run | serve | bench | import.  Thin front-end over ie::Engine.
#include "ie/engine.hpp"
#include "ie/openai_server.hpp"
#include "ie/hf_import.hpp"
#include "ie/preflight.hpp"
#include "ie/gguf.hpp"
#include "ie/serve_options.hpp"
#include "ie/serve_config.hpp"
#include "ie/card_select.hpp"
#include "ie/card_lock.hpp"
#include "ie/supervisor.hpp"
#include "ie/cli_help.hpp"
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
  "                                [--vram-reserve-gib F]\n"
  "                                (--parallel N = 1..16 concurrent generations, default 1;\n"
  "                                 what >1 does depends on the arch: `ie serve --help`.\n"
  "                                 Not yet compatible with --int8-kv.)\n"
  "                                (--max-queue N = requests allowed to WAIT for a generation\n"
  "                                 slot; beyond that the server answers HTTP 429 at once.\n"
  "                                 Default 8. /health reports inflight/queued.)\n"
  "                                (--prefill-chunk = tokens per prefill forward, default 256.\n"
  "                                 --vram-reserve-gib = GiB the auto expert tier leaves free per card\n"
  "                                 (mimo_v2, deepseek41; default 1.5; raise it for very long contexts).\n"
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

    // Help: printed before anything is parsed or loaded. `ie pull --help` stays with the ie-pull helper.
    if (cmd == "--help" || cmd == "-h" || cmd == "help") {
        if (argc < 3) { std::fputs(ie::cli_help_overview(), stdout); return 0; }
        const std::string topic = argv[2];
        const std::string h = ie::cli_help(topic == "--help" || topic == "-h" ? "help" : topic);
        if (h.empty()) {
            std::fprintf(stderr, "ie help: unknown command '%s'\n\n%s", argv[2], ie::cli_help_overview());
            return 2;
        }
        std::fputs(h.c_str(), stdout);
        return 0;
    }
    if (cmd != "pull" && !ie::cli_help(cmd).empty() &&
        ie::cli_wants_help(std::vector<std::string>(argv + 2, argv + argc))) {
        std::fputs(ie::cli_help(cmd).c_str(), stdout);
        return 0;
    }

    // ie cards -- the GPUs --cards numbers (Level Zero only; nothing is loaded).
    if (cmd == "cards") {
        if (argc > 2) { std::fputs("usage: ie cards\n", stderr); return 2; }
        std::vector<ie::LevelZeroGpu> gpus;
        if (auto e = ie::enumerate_level_zero_gpus(gpus); !e.empty()) {
            std::fprintf(stderr, "cards: %s\n", e.c_str()); return 1;
        }
        const auto cards = ie::cards_of(gpus);
        for (size_t i = 0; i < cards.size(); ++i)
            std::printf("card %zu  %s  %s  (level_zero:%u)\n", i, cards[i].pci.c_str(), cards[i].name.c_str(), cards[i].index);
        for (const auto& g : gpus)
            if (g.integrated)
                std::printf("-       %s  %s  (level_zero:%u, integrated: not a card)\n", g.pci.c_str(), g.name.c_str(), g.index);
        return 0;
    }

    // ie supervise --config <layout.json> [--host H] [--port P]: every server of the layout as its own `ie serve`
    // child behind one routing endpoint (ie/supervisor.hpp, docs/serve_config.md "Supervisor").
    if (cmd == "supervise") {
        std::string config_path, host;
        int port = 0;
        std::vector<ie::sup::ServerSpec> specs;
        ie::sup::Options so;
        try {
            for (int i = 2; i < argc; ++i) {
                const std::string a = argv[i];
                if (i + 1 >= argc) throw std::runtime_error(a + " requires a value (usage: ie supervise --config <layout.json> [--host H] [--port P])");
                const std::string v = argv[++i];
                if (a == "--config") config_path = v;
                else if (a == "--host") host = v;
                else if (a == "--port") {
                    const auto opts = ie::parse_launch_options({"--port", v});   // the same range check as serve
                    port = opts.port;
                } else throw std::runtime_error("unknown option: " + a + " (usage: ie supervise --config <layout.json> [--host H] [--port P])");
            }
            if (config_path.empty()) throw std::runtime_error("--config <layout.json> is required");
            const std::string abs_config = std::filesystem::absolute(config_path).string();
            const ie::ServeConfig layout = ie::load_serve_config(abs_config);
            std::error_code ec;
            const std::string self = std::filesystem::read_symlink("/proc/self/exe", ec).string();
            if (self.empty()) throw std::runtime_error("cannot resolve this executable (/proc/self/exe)");
            so.host = !host.empty() ? host : !layout.front_host.empty() ? layout.front_host : "127.0.0.1";
            so.port = port ? port : layout.front_port ? layout.front_port : 11435;
            for (size_t i = 0; i < layout.servers.size(); ++i) {
                const auto& s = layout.servers[i];
                const std::string where = "servers[" + std::to_string(i) + "]";
                if (s.name.empty()) throw std::runtime_error(where + ": `ie supervise` needs a \"name\" for every server");
                if (s.model.empty()) throw std::runtime_error(where + " (" + s.name + "): \"model\" is required");
                if (s.launch.cards.empty()) throw std::runtime_error(where + " (" + s.name + "): `ie supervise` needs \"cards\" for every server");
                const std::string& h = s.launch.host;
                ie::sup::ServerSpec sp;
                sp.connect_host.clear();
                // Reached over loopback (its /admin/shutdown answers loopback only): a wildcard bind through
                // 127.0.0.1 / ::1, a loopback bind at its own address.
                for (const auto& a : ie::bind_addresses(h)) {
                    if (a == "0.0.0.0") sp.connect_host = "127.0.0.1";
                    else if (a == "::") sp.connect_host = "::1";
                    else if (a.rfind("127.", 0) == 0 || a == "::1") sp.connect_host = a;
                    else continue;
                    break;
                }
                if (sp.connect_host.empty())
                    throw std::runtime_error(where + " (" + s.name + "): host " + h + " is not reachable over loopback; "
                                             "the supervisor stops each server through its loopback-only /admin/shutdown");
                if (s.launch.port == so.port && ie::binds_overlap(so.host, h))
                    throw std::runtime_error(where + " (" + s.name + "): port " + std::to_string(so.port) +
                                             " is the front endpoint's port; give the server its own");
                sp.name = s.name;
                sp.model_id = ie::model_id_from_path(s.model);
                sp.port = s.launch.port;
                sp.cards = s.launch.cards;
                sp.argv = {self, "serve", "--config", abs_config, "--server", s.name};
                sp.restart_on_failure = s.restart_on_failure;
                sp.slots = s.launch.engine.parallel + s.launch.max_queue;
                if (s.name == layout.default_server) so.default_index = int(i);
                specs.push_back(std::move(sp));
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "ie supervise: %s\n", e.what()); return 2;
        }
        return ie::sup::run_supervisor(specs, so);
    }

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

    // --config <layout.json> (docs/serve_config.md): the layout's flags, then the command line's. Without --config
    // the arguments are parsed exactly as before.
    const std::vector<std::string> rest(argv + 2, argv + argc);
    bool has_config = false;
    for (size_t i = 0; i < rest.size(); ++i) {
        if (rest[i] == "--config") { has_config = true; break; }
        if (rest[i].rfind("--", 0) == 0 && !ie::launch_flag_is_boolean(rest[i])) ++i;   // skip the flag's value
    }
    std::string model_path;
    std::string served_name;   // the layout's "name": what /v1/models reports (only when a layout names the server)
    ie::LaunchOptions launch;
    if (!has_config) {
        if (argc < 3) { std::fputs(USAGE, stderr); return 2; }
        model_path = argv[2];
        try { launch = ie::parse_launch_options(std::vector<std::string>(argv + 3, argv + argc)); }
        catch (const std::exception& e) {
            std::fprintf(stderr, "invalid options: %s\n", e.what()); return 2;
        }
    } else {
        try {
            std::string config_path, server_name;
            std::vector<std::string> cli;
            size_t i = 0;
            if (rest[0].rfind("-", 0) != 0) model_path = rest[i++];   // an explicit <model> overrides the layout's
            for (; i < rest.size(); ++i) {
                if (rest[i] == "--config") {
                    if (i + 1 >= rest.size()) throw std::runtime_error("--config requires a value");
                    if (!config_path.empty()) throw std::runtime_error("--config given twice");
                    config_path = rest[++i];
                    continue;
                }
                if (rest[i] == "--server") {   // one server of a multi-server layout (how `ie supervise` starts each)
                    if (i + 1 >= rest.size()) throw std::runtime_error("--server requires a value");
                    server_name = rest[++i];
                    continue;
                }
                cli.push_back(rest[i]);
                if (rest[i].rfind("--", 0) == 0 && !ie::launch_flag_is_boolean(rest[i]) && i + 1 < rest.size())
                    cli.push_back(rest[++i]);
            }
            const ie::ServeConfig layout = ie::load_serve_config(config_path);
            const ie::ServeConfigServer& srv =
                server_name.empty() ? ie::single_server(layout) : ie::named_server(layout, server_name);
            served_name = srv.name;
            if (model_path.empty()) model_path = srv.model;
            if (model_path.empty())
                throw std::runtime_error("no model: pass <model> or set \"model\" in " + config_path);
            const auto env_set = ie::apply_config_env(srv.env);   // before parsing: the IE_SERVE_* defaults are read there
            launch = ie::parse_launch_options(ie::merge_launch_args(srv.args, cli));
            std::string names;
            for (const auto& n : env_set) names += " " + n;
            std::fprintf(stderr, "[ie] layout %s: server \"%s\", model %s%s%s\n", config_path.c_str(),
                         srv.name.c_str(), model_path.c_str(), names.empty() ? "" : ", env", names.c_str());
        } catch (const std::exception& e) {
            std::fprintf(stderr, "invalid options: %s\n", e.what()); return 2;
        }
    }
    // --cards: pin this process to the chosen physical cards before anything starts the SYCL runtime (ie/card_select.hpp).
    // serve/run also take the cards' locks (ie/card_lock.hpp): one engine process per card.
    ie::CardLocks card_locks;
    if (!launch.cards.empty() && (cmd == "serve" || cmd == "run")) {
        if (auto e = ie::acquire_card_locks(launch.cards, card_locks); !e.empty()) {
            std::fprintf(stderr, "%s\n", e.c_str()); return 2;
        }
    }
    if (!launch.cards.empty()) {
        std::vector<std::string> pcis;
        std::string e = ie::apply_card_selection(launch.cards, pcis);
        if (e.empty()) e = ie::verify_card_selection(pcis);
        if (!e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 2; }
    }
    auto& opts = launch.engine;
    const auto& host = launch.host;
    const int port = launch.port;
    ie::oai::configure_server_defaults(launch.defaults);
    if (launch.vram_reserve_gib >= 0) {   // the auto expert tiers read their headroom from the environment at load
        const auto g = std::to_string(launch.vram_reserve_gib);
        setenv("IE_MIMO26_VRAM_RESERVE_GIB", g.c_str(), 1);
        setenv("IE_DS41_VRAM_RESERVE_GIB", g.c_str(), 1);
    }
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
        return ie::run_openai_server(*eng, served_name.empty() ? ie::model_id_from_path(model_path) : served_name,
                                     host, port, launch.max_queue,
                                     served_name.empty() ? std::string() : ie::model_id_from_path(model_path));

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
