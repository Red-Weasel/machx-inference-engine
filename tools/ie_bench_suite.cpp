// tools/ie_bench_suite.cpp — Deterministic prompt-suite benchmark.
//
// Runs each prompt in benchmark_prompts/ 3 times with identical settings
// (greedy argmax, fixed KV mode, fixed decode budget) and reports
// median / min / max decode tok/s plus top-5 kernel breakdown from the
// median run.  All runs are bit-identical: same prompt → same token ids →
// same KV state → same per-step compute.
//
// Usage:
//   ie-bench-suite [--gguf path] [--prompts-dir dir] [--decode N]
//                  [--runs N] [--warmup N] [--int8-kv]

#include "ie/allocator.hpp"
#include "ie/deltanet_state.hpp"
#include "ie/gguf.hpp"
#include "ie/kernel_profiler.hpp"
#include "ie/kv_cache.hpp"
#include "ie/ops.hpp"
#include "ie/qwen36.hpp"
#include "ie/tokenizer.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) return {};
    return {std::istreambuf_iterator<char>(f), {}};
}

static int32_t argmax_fp16(const std::vector<sycl::half>& v) {
    int32_t best = 0;
    float   bval = float(v[0]);
    for (size_t i = 1; i < v.size(); ++i) {
        float f = float(v[i]);
        if (f > bval) { bval = f; best = int32_t(i); }
    }
    return best;
}

// ---------------------------------------------------------------------------
struct RunResult {
    uint32_t prompt_toks = 0;
    double   prefill_ms  = 0;
    uint32_t decode_toks = 0;
    double   decode_ms   = 0;
    std::vector<ie::KernelProfiler::Stat> decode_stats;

    double prefill_tok_s() const { return prompt_toks && prefill_ms > 0 ? 1000.0 * prompt_toks / prefill_ms : 0; }
    double decode_tok_s()  const { return decode_toks  && decode_ms  > 0 ? 1000.0 * decode_toks  / decode_ms  : 0; }
    double avg_ms_tok()    const { return decode_toks ? decode_ms / decode_toks : 0; }
};

struct TableRow {
    std::string name;
    uint32_t prompt_toks = 0;
    uint32_t gen_toks    = 0;
    double median_tok_s  = 0;
    double min_tok_s     = 0;
    double max_tok_s     = 0;
    std::string top_kernel;
};

// ---------------------------------------------------------------------------
static void print_kernel_table(const std::vector<ie::KernelProfiler::Stat>& stats, int top_n) {
    if (stats.empty()) return;

    auto sorted = stats;
    std::sort(sorted.begin(), sorted.end(),
        [](const auto& a, const auto& b) { return a.total_ns > b.total_ns; });

    uint64_t total_ns = 0;
    for (const auto& s : sorted) total_ns += s.total_ns;

    const char* RED  = "\033[31m";
    const char* YEL  = "\033[33m";
    const char* DIM  = "\033[2m";
    const char* RST  = "\033[0m";
    const char* BOLD = "\033[1m";

    std::printf("  %s%-24s %6s %10s %8s %6s%s\n",
        BOLD, "Kernel", "calls", "total ms", "avg ms", "%", RST);
    std::printf("  ");
    for (int i = 0; i < 58; ++i) std::putchar('-');
    std::printf("\n");

    const int n = std::min((int)sorted.size(), top_n);
    for (int i = 0; i < n; ++i) {
        const auto& s = sorted[i];
        const double pct = total_ns ? double(s.total_ns) * 100.0 / total_ns : 0.0;
        const int bar = std::max(0, std::min(16, int(pct / 100.0 * 16 + 0.5)));
        const char* col = (i == 0) ? RED : (i < 3) ? YEL : (pct < 1.0) ? DIM : RST;
        std::printf("  %s%-24s %6u %10.3f %8.4f %5.1f%%  ",
            col, s.name.c_str(), s.calls, s.total_ms(), s.avg_ms(), pct);
        for (int b = 0; b < bar; ++b) std::printf("\xe2\x96\x88");
        std::printf("%s\n", RST);
    }
    std::printf("  ");
    for (int i = 0; i < 58; ++i) std::putchar('-');
    std::printf("\n");
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    std::string gguf_path   = "/home/weezy/.seal/models/Qwen3.6-35B-A3B-Q4_K_M.gguf";
    std::string prompts_dir = "benchmark_prompts";
    uint32_t    decode_n    = 128;
    uint32_t    n_runs      = 3;
    uint32_t    warmup      = 3;
    bool        int8_kv     = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--gguf"        && i+1 < argc) gguf_path   = argv[++i];
        else if (a == "--prompts-dir" && i+1 < argc) prompts_dir = argv[++i];
        else if (a == "--decode"      && i+1 < argc) decode_n    = uint32_t(std::atoi(argv[++i]));
        else if (a == "--runs"        && i+1 < argc) n_runs      = uint32_t(std::atoi(argv[++i]));
        else if (a == "--warmup"      && i+1 < argc) warmup      = uint32_t(std::atoi(argv[++i]));
        else if (a == "--int8-kv")                   int8_kv     = true;
    }

    // Fixed prompt suite — same files, same order, every run.
    const std::vector<std::pair<std::string,std::string>> suite = {
        {"short-chat",       prompts_dir + "/01_short_chat.txt"},
        {"long-instruction", prompts_dir + "/02_long_instruction.txt"},
        {"codegen",          prompts_dir + "/03_codegen.txt"},
        {"math-reasoning",   prompts_dir + "/04_math_reasoning.txt"},
        {"long-context",     prompts_dir + "/05_long_context.txt"},
    };

    // ── GGUF + tokenizer ─────────────────────────────────────────────────────
    ie::GgufReader g;
    if (auto e = g.open(gguf_path); !e.empty()) {
        std::fprintf(stderr, "gguf: %s\n", e.c_str()); return 1;
    }
    ie::Tokenizer tok;
    if (auto e = tok.load_from_gguf(g); !e.empty()) {
        std::fprintf(stderr, "tokenizer: %s\n", e.c_str()); return 1;
    }

    // ── Model ────────────────────────────────────────────────────────────────
    ie::DeviceAllocator alloc;
    if (auto e = alloc.init(); !e.empty()) {
        std::fprintf(stderr, "alloc: %s\n", e.c_str()); return 1;
    }
    auto& q = alloc.queue();
    const std::string device_name = q.get_device().get_info<sycl::info::device::name>();
    std::fprintf(stderr, "Device: %s\nLoading model...\n", device_name.c_str());

    ie::QwenConfig cfg;
    ie::QwenModel  model;
    if (auto e = model.load(alloc, g, cfg); !e.empty()) {
        std::fprintf(stderr, "model.load: %s\n", e.c_str()); return 1;
    }
    std::fprintf(stderr, "Loaded.\n\n");

    // ── KV cache sized to handle all prompts ─────────────────────────────────
    // 2048 is a safe upper bound: longest prompt ~400 toks + 128 decode + headroom.
    constexpr uint32_t MAX_CTX = 2048;
    const uint32_t L_full = cfg.n_layers / cfg.full_attn_interval;
    const uint32_t L_lin  = cfg.n_layers - L_full;

    ie::KvCache kv;
    {
        ie::KvCacheConfig kvcfg{};
        kvcfg.n_layers_full = L_full;
        kvcfg.n_kv_heads    = cfg.n_kv_heads;
        kvcfg.max_ctx       = MAX_CTX;
        kvcfg.head_dim      = cfg.head_dim;
        kvcfg.use_int8      = int8_kv;
        if (auto e = kv.init(alloc, kvcfg); !e.empty()) {
            std::fprintf(stderr, "kv: %s\n", e.c_str()); return 1;
        }
    }
    ie::DeltaNetState dn;
    if (auto e = dn.init(alloc, ie::DeltaNetStateConfig{
            L_lin, cfg.ssm_n_v_heads, cfg.ssm_head_dim, cfg.ssm_head_dim,
            cfg.ssm_inner * 2, cfg.ssm_conv_kernel}); !e.empty()) {
        std::fprintf(stderr, "dn: %s\n", e.c_str()); return 1;
    }
    if (auto e = model.ensure_workspace(MAX_CTX); !e.empty()) {
        std::fprintf(stderr, "ws: %s\n", e.c_str()); return 1;
    }
    if (auto e = model.ensure_attn_partials(MAX_CTX); !e.empty()) {
        std::fprintf(stderr, "attn_partials: %s\n", e.c_str()); return 1;
    }

    // Shared GPU buffers — allocated once, reused for all prompts.
    auto* d_logits = static_cast<sycl::half*>(alloc.malloc(uint64_t(cfg.vocab) * sizeof(sycl::half)));
    auto* d_tok    = static_cast<int32_t*>(alloc.malloc(sizeof(int32_t)));
    std::vector<sycl::half> h_logits(cfg.vocab);

    ie::KernelProfiler prof;
    std::vector<TableRow> table;
    const std::string kv_mode = int8_kv ? "INT8" : "fp16";

    // ── Print header ─────────────────────────────────────────────────────────
    std::printf("ie-bench-suite   model=Qwen3.6-35B-A3B-Q4_K_M   kv=%s   decode=%u   runs=%u\n",
        kv_mode.c_str(), decode_n, n_runs);
    std::printf("device: %s\n", device_name.c_str());
    std::printf("================================================================\n\n");

    // ── Prompt loop ───────────────────────────────────────────────────────────
    for (const auto& [pname, ppath] : suite) {
        std::printf(">>> %s\n", pname.c_str());
        std::printf("    file: %s\n", ppath.c_str());

        const std::string raw = read_file(ppath);
        if (raw.empty()) {
            std::printf("    [SKIP] file not found or empty\n\n");
            continue;
        }

        // Build ChatML and encode — identical every invocation.
        std::vector<ie::ChatTurn> turns = {{"user", raw}};
        const std::string chatml = ie::build_chatml_prompt(turns, /*gen=*/true, /*think=*/false);
        const std::vector<int32_t> ids = tok.encode(chatml, /*allow_special=*/true);
        const uint32_t plen = uint32_t(ids.size());

        if (plen + decode_n + warmup + 8 > MAX_CTX) {
            std::printf("    [SKIP] prompt too long: %u tokens (MAX_CTX=%u)\n\n", plen, MAX_CTX);
            continue;
        }
        std::printf("    prompt tokens: %u   decode budget: %u\n\n", plen, decode_n);

        auto* d_prompt = static_cast<int32_t*>(alloc.malloc(uint64_t(plen) * sizeof(int32_t)));
        q.memcpy(d_prompt, ids.data(), uint64_t(plen) * sizeof(int32_t)).wait();

        const int32_t start_tok = tok.bos_token_id() >= 0 ? tok.bos_token_id() : 0;

        std::vector<RunResult> results(n_runs);

        for (uint32_t r = 0; r < n_runs; ++r) {
            std::printf("    run %u/%u  prefill...", r + 1, n_runs); std::fflush(stdout);

            // Reset KV + DeltaNet state — identical starting point every run.
            kv.reset();
            dn.reset(q);
            q.wait();

            // ── Prefill ───────────────────────────────────────────────────────
            ie::g_profiler = nullptr;  // don't count prefill in decode kernel stats
            const double t_pre0 = now_ms();
            {
                uint32_t pos = 0;
                constexpr uint32_t CHUNK = 256;
                while (pos + CHUNK <= plen) {
                    model.forward(q, d_prompt + pos, CHUNK, pos, kv, dn, d_logits).wait();
                    pos += CHUNK;
                }
                if (pos < plen)
                    model.forward(q, d_prompt + pos, plen - pos, pos, kv, dn, d_logits).wait();
            }
            const double prefill_ms = now_ms() - t_pre0;

            // ── Warmup decode (not timed, not profiled) ───────────────────────
            std::printf("  warmup..."); std::fflush(stdout);
            uint32_t pos = plen;
            int32_t next_tok = start_tok;
            for (uint32_t w = 0; w < warmup; ++w) {
                q.memcpy(d_tok, &next_tok, sizeof(int32_t)).wait();
                model.forward(q, d_tok, 1, pos, kv, dn, d_logits).wait();
                q.memcpy(h_logits.data(), d_logits, uint64_t(cfg.vocab) * sizeof(sycl::half)).wait();
                next_tok = argmax_fp16(h_logits);
                pos++;
            }

            // ── Timed + profiled decode ───────────────────────────────────────
            std::printf("  decode..."); std::fflush(stdout);
            ie::g_profiler = &prof;
            prof.begin_step();  // accumulate all decode steps in one harvest
            const double t_dec0 = now_ms();
            for (uint32_t d = 0; d < decode_n; ++d) {
                q.memcpy(d_tok, &next_tok, sizeof(int32_t)).wait();
                model.forward(q, d_tok, 1, pos, kv, dn, d_logits).wait();
                q.memcpy(h_logits.data(), d_logits, uint64_t(cfg.vocab) * sizeof(sycl::half)).wait();
                next_tok = argmax_fp16(h_logits);
                pos++;
            }
            const double decode_ms = now_ms() - t_dec0;
            auto decode_stats = prof.harvest();
            ie::g_profiler = nullptr;

            results[r] = {plen, prefill_ms, decode_n, decode_ms, std::move(decode_stats)};

            std::printf("  %.1f ms prefill (%.1f tok/s)   %.2f tok/s decode (%.3f ms/tok)\n",
                prefill_ms, results[r].prefill_tok_s(),
                results[r].decode_tok_s(), results[r].avg_ms_tok());
        }

        alloc.free(d_prompt);

        // ── Compute median / min / max ────────────────────────────────────────
        std::vector<double> ts(n_runs);
        for (uint32_t r = 0; r < n_runs; ++r) ts[r] = results[r].decode_tok_s();

        std::vector<size_t> order(n_runs);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return ts[a] < ts[b]; });

        const size_t med_idx = order[n_runs / 2];
        const double min_ts  = ts[order.front()];
        const double max_ts  = ts[order.back()];
        const double med_ts  = ts[med_idx];
        const RunResult& mr  = results[med_idx];

        std::printf("\n    ── Summary ──────────────────────────────────────────\n");
        std::printf("    Decode:  median %.2f  min %.2f  max %.2f tok/s  "
                    "(spread %.1f%%)\n",
            med_ts, min_ts, max_ts,
            min_ts > 0 ? (max_ts - min_ts) / min_ts * 100.0 : 0.0);
        std::printf("    Prefill (median run): %.1f ms → %.1f tok/s\n",
            mr.prefill_ms, mr.prefill_tok_s());
        std::printf("    Avg ms/tok (median run): %.3f ms\n", mr.avg_ms_tok());

        std::printf("\n    ── Top-5 kernels (median run, %u decode steps aggregated) ──\n",
            decode_n);
        print_kernel_table(mr.decode_stats, 5);

        // Top kernel for table row
        std::string top_kern = "—";
        if (!mr.decode_stats.empty()) {
            auto it = std::max_element(mr.decode_stats.begin(), mr.decode_stats.end(),
                [](const auto& a, const auto& b) { return a.total_ns < b.total_ns; });
            top_kern = it->name;
        }

        table.push_back({pname, plen, decode_n, med_ts, min_ts, max_ts, top_kern});
        std::printf("\n");
    }

    // ── Final markdown table ──────────────────────────────────────────────────
    std::printf("================================================================\n");
    std::printf("## Results  (kv=%s, decode=%u toks, runs=%u)\n\n", kv_mode.c_str(), decode_n, n_runs);
    std::printf("| %-16s | %11s | %8s | %12s | %9s | %9s | %-22s |\n",
        "Prompt", "Prompt toks", "Gen toks", "Median tok/s", "Min tok/s", "Max tok/s", "Top kernel");
    std::printf("| %s | %s | %s | %s | %s | %s | %s |\n",
        "----------------", "-----------:", "-------:", "------------:",
        "---------:", "---------:", "----------------------");
    for (const auto& row : table) {
        std::printf("| %-16s | %11u | %8u | %12.2f | %9.2f | %9.2f | %-22s |\n",
            row.name.c_str(), row.prompt_toks, row.gen_toks,
            row.median_tok_s, row.min_tok_s, row.max_tok_s, row.top_kernel.c_str());
    }
    std::printf("\n");

    return 0;
}
