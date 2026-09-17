// tools/qwen3next_bench.cpp — Qwen3-Next-80B throughput on the fixed 5-prompt
// suite (the qwen3next analogue of ie-bench-suite: same benchmark_prompts/01-05
// files, same ChatML build, same chunked-prefill + timed-greedy-decode protocol,
// so the numbers are consistent with the crown's suite). Per category: prefill
// tok/s + decode tok/s (median over --runs). Prefill is CHUNKED at <=256 (DeltaNet
// T<=256 hardware workaround). int-dot W4A8 down is ON by default (IE_QWEN3NEXT_NO_Q8=1
// for the fp16 A/B). RUN ONLY with the cards free + serialized.
//   usage: ie-qwen3next-bench <gguf> <n_gpus> [--prompts-dir benchmark_prompts]
//                             [--decode 128] [--runs 3] [--warmup 3]
#include "ie/qwen3next.hpp"
#include "ie/gguf.hpp"
#include "ie/model_config.hpp"
#include "ie/tokenizer.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
static std::string read_file(const std::string& p) {
    std::ifstream f(p); if (!f) return {}; std::stringstream ss; ss << f.rdbuf(); return ss.str();
}
static double median(std::vector<double> v) {
    if (v.empty()) return 0; std::sort(v.begin(), v.end()); return v[v.size() / 2];
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <gguf> <n_gpus> [--prompts-dir D] [--decode N] [--runs N] [--warmup N]\n", argv[0]);
        return 2;
    }
    const std::string gguf = argv[1];
    const uint32_t n_gpus = uint32_t(std::stoul(argv[2]));
    std::string pdir = "benchmark_prompts";
    uint32_t decode_n = 128, n_runs = 3, warmup = 3;
    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--prompts-dir" && i + 1 < argc) pdir     = argv[++i];
        else if (a == "--decode"      && i + 1 < argc) decode_n = std::atoi(argv[++i]);
        else if (a == "--runs"        && i + 1 < argc) n_runs   = std::atoi(argv[++i]);
        else if (a == "--warmup"      && i + 1 < argc) warmup   = std::atoi(argv[++i]);
    }
    // DeltaNet prefill chunk. Default 256 (the historical HW-bug cap); qwen3next
    // is empirically CLEAN at 512 (25/25 bit-identical, no collapse — see
    // docs/known_bugs.md note + MASTER_DEV_PLAN §7), which closes the pp512 gap
    // vs llama. Override for A/B via IE_QWEN3NEXT_PREFILL_CHUNK.
    uint32_t CHUNK = 256;
    // MAX_CTX: default 2048 for the short suite; override via IE_QWEN3NEXT_BENCH_MAXCTX
    // to bench long-context prefill (e.g. the hd256 tile lever, which gates at ctx≥6144).
    uint32_t MAX_CTX = 2048;
    if (const char* m = std::getenv("IE_QWEN3NEXT_BENCH_MAXCTX")) { int v = std::atoi(m); if (v >= 256) MAX_CTX = uint32_t(v); }
    if (const char* c = std::getenv("IE_QWEN3NEXT_PREFILL_CHUNK")) {
        const int v = std::atoi(c);
        if (v >= 1 && uint32_t(v) <= MAX_CTX) CHUNK = uint32_t(v);
    }

    ie::GgufReader g;
    std::string err = g.open(gguf);
    if (!err.empty()) { std::fprintf(stderr, "gguf: %s\n", err.c_str()); return 1; }
    ie::Qwen3NextConfig cfg;
    if ((err = ie::read_qwen3next_config(g, cfg)) != "") { std::fprintf(stderr, "config: %s\n", err.c_str()); return 1; }
    ie::Tokenizer tok;
    if ((err = tok.load_from_gguf(g)) != "") { std::fprintf(stderr, "tok: %s\n", err.c_str()); return 1; }
    ie::DeviceFleet fleet;
    if ((err = fleet.init(n_gpus)) != "") { std::fprintf(stderr, "fleet: %s\n", err.c_str()); return 1; }
    ie::LayerPlan plan = ie::LayerPlan::contiguous(cfg.hybrid.n_transformer_layers(), n_gpus);
    ie::Qwen3NextModel m;
    if ((err = m.load(fleet, plan, g, cfg, MAX_CTX)) != "") { std::fprintf(stderr, "load: %s\n", err.c_str()); return 1; }
    const uint32_t V = cfg.hybrid.dense.vocab;
    std::vector<sycl::half> logits(V);
    auto argmax = [&]() { int32_t b = 0; float bv = -1e30f;
        for (uint32_t i = 0; i < V; ++i) { float v = float(logits[i]); if (v > bv) { bv = v; b = int32_t(i); } } return b; };

    const std::vector<std::pair<std::string, std::string>> suite = {
        {"short-chat",       pdir + "/01_short_chat.txt"},
        {"long-instruction", pdir + "/02_long_instruction.txt"},
        {"codegen",          pdir + "/03_codegen.txt"},
        {"math-reasoning",   pdir + "/04_math_reasoning.txt"},
        {"long-context",     pdir + "/05_long_context.txt"},
    };

    // One pass of the 5-prompt suite at a given prefill chunk; prints the table
    // and returns {median prefill t/s, median decode t/s}.
    auto run_suite = [&](uint32_t chunk) -> std::pair<double, double> {
        std::printf("ie-qwen3next-bench  model=Qwen3-Next-80B-A3B-Q4_K_M  gpus=%u  int_dot_down=%s  prefill_chunk=%u  decode=%u  runs=%u\n\n",
                    n_gpus, std::getenv("IE_QWEN3NEXT_NO_Q8") ? "off(fp16)" : "ON", chunk, decode_n, n_runs);
        std::printf("%-18s %8s %12s %12s\n", "category", "p_toks", "prefill t/s", "decode t/s");
        std::printf("%-18s %8s %12s %12s\n", "--------", "------", "-----------", "----------");
        std::vector<double> all_pp, all_tg;
        for (const auto& [name, path] : suite) {
            const std::string raw = read_file(path);
            if (raw.empty()) { std::printf("%-18s  [skip: missing %s]\n", name.c_str(), path.c_str()); continue; }
            std::vector<ie::ChatTurn> turns = {{"user", raw}};
            const std::string chatml = ie::build_chatml_prompt(turns, /*gen=*/true, /*think=*/false);
            std::vector<int32_t> ids = tok.encode(chatml, /*allow_special=*/true);
            const uint32_t plen = uint32_t(ids.size());
            if (plen + decode_n + warmup + 8 > MAX_CTX) { std::printf("%-18s  [skip: %u tok > ctx]\n", name.c_str(), plen); continue; }
            std::vector<double> pps, tgs;
            for (uint32_t r = 0; r < n_runs; ++r) {
                const double t0 = now_ms();
                for (uint32_t pos = 0; pos < plen; pos += chunk) {
                    const uint32_t T = std::min(chunk, plen - pos);
                    m.forward(&ids[pos], T, pos, /*reset_kv=*/(pos == 0), logits.data());
                }
                const double pre_ms = now_ms() - t0;
                uint32_t pos = plen; int32_t nxt = argmax();
                for (uint32_t w = 0; w < warmup; ++w) { m.forward(&nxt, 1, pos++, false, logits.data()); nxt = argmax(); }
                const double t1 = now_ms();
                for (uint32_t d = 0; d < decode_n; ++d) { m.forward(&nxt, 1, pos++, false, logits.data()); nxt = argmax(); }
                const double dec_ms = now_ms() - t1;
                pps.push_back(1000.0 * plen / pre_ms);
                tgs.push_back(1000.0 * decode_n / dec_ms);
            }
            const double pp = median(pps), tg = median(tgs);
            all_pp.push_back(pp); all_tg.push_back(tg);
            std::printf("%-18s %8u %12.1f %12.2f\n", name.c_str(), plen, pp, tg);
        }
        const double mpp = median(all_pp), mtg = median(all_tg);
        std::printf("\nmedian across categories:  prefill %.1f tok/s   decode %.2f tok/s\n", mpp, mtg);
        return {mpp, mtg};
    };

    // Synthetic long-prefill timer: prefill `ntok` tokens at `chunk` and return
    // tok/s (median of n_runs, with a warmup). The suite prompts are all <256
    // tokens so they prefill in ONE chunk regardless of cap — the 256-vs-512 win
    // only appears for prompts >256, which this measures directly.
    auto time_prefill = [&](uint32_t ntok, uint32_t chunk) -> double {
        std::vector<int32_t> ids(ntok);
        for (uint32_t i = 0; i < ntok; ++i) ids[i] = int32_t(1 + (i % 4096));
        auto once = [&]() {
            for (uint32_t pos = 0; pos < ntok; pos += chunk) {
                const uint32_t T = std::min(chunk, ntok - pos);
                m.forward(&ids[pos], T, pos, /*reset_kv=*/(pos == 0), logits.data());
            }
        };
        once();   // warmup / JIT
        std::vector<double> ms;
        for (uint32_t r = 0; r < n_runs; ++r) { const double t0 = now_ms(); once(); ms.push_back(now_ms() - t0); }
        return 1000.0 * ntok / median(ms);
    };

    // --ab: order-controlled single-load A/B. First the real suite (256->512->256,
    // discard the first as JIT warmup), then synthetic long prefills (512/1024 tok)
    // where the chunk cap actually changes the call count. Else one pass at CHUNK.
    bool ab = false;
    for (int i = 3; i < argc; ++i) if (std::string(argv[i]) == "--ab") ab = true;
    if (ab) {
        std::printf("### A/B pass 1: chunk 256 (warmup, discard)\n");
        run_suite(256);
        std::printf("\n### A/B pass 2: chunk 512\n");
        auto p512 = run_suite(512);
        std::printf("\n### A/B pass 3: chunk 256\n");
        auto p256 = run_suite(256);
        std::printf("\n=== suite A/B (all prompts <256 tok => one chunk either way) ===\n");
        std::printf("prefill: 512 %.1f vs 256 %.1f  => %.2fx\n", p512.first, p256.first,
                    p256.first > 0 ? p512.first / p256.first : 0.0);
        std::printf("decode : 512 %.2f vs 256 %.2f (parity)\n", p512.second, p256.second);

        std::printf("\n=== synthetic long-prefill (the >256-tok case the cap targets) ===\n");
        std::printf("%-8s %12s %12s %12s %10s\n", "ntok", "c256 t/s", "c512 t/s", "c256 t/s", "512/256");
        for (uint32_t ntok : {512u, 768u, 1024u}) {
            if (ntok + 8 > MAX_CTX) continue;
            const double a = time_prefill(ntok, 256);
            const double b = time_prefill(ntok, 512);
            const double c = time_prefill(ntok, 256);
            const double base = 0.5 * (a + c);   // bracket the 512 run
            std::printf("%-8u %12.1f %12.1f %12.1f %9.2fx\n", ntok, a, b, c,
                        base > 0 ? b / base : 0.0);
        }
        return 0;
    }
    run_suite(CHUNK);
    return 0;
}
