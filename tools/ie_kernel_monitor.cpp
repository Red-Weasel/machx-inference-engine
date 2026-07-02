// tools/ie_kernel_monitor.cpp — Live per-kernel timing feed for all SYCL kernels.
//
// Shows a refreshing table of every named kernel dispatched during each decode
// step in dispatch order with % bar.  Runs the real model on a fixed real prompt
// so timings are repeatable and comparable across engine tweaks.
//
// Usage:
//   ie-kernel-monitor [--gguf <path>] [--prompt "..."] [--decode N] [--int8-kv] [--warmup 3]

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
#include <deque>
#include <numeric>
#include <signal.h>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Globals
// ---------------------------------------------------------------------------
static volatile bool g_running = true;
static void on_sigint(int) { g_running = false; }

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------
// ANSI helpers
// ---------------------------------------------------------------------------
namespace ansi {
    static const char* RESET  = "\033[0m";
    static const char* BOLD   = "\033[1m";
    static const char* DIM    = "\033[2m";
    static const char* RED    = "\033[31m";
    static const char* GRN    = "\033[32m";
    static const char* YEL    = "\033[33m";
    static const char* CYN    = "\033[36m";
    static const char* WHT    = "\033[97m";

    static void cursor_up(int n) { if (n > 0) std::printf("\033[%dA", n); }
    static void clear_to_end()   { std::printf("\033[J"); }
}

// ---------------------------------------------------------------------------
// Render prefill stats (static, printed once)
// ---------------------------------------------------------------------------
static void render_prefill(
    const std::vector<ie::KernelProfiler::Stat>& stats,
    uint32_t ctx_len,
    double total_ms,
    double tok_per_s)
{
    constexpr int W = 90;
    uint64_t total_ns = 0;
    for (const auto& s : stats) total_ns += s.total_ns;
    const double kernel_ms = double(total_ns) * 1e-6;

    std::printf("\n%s+-- PREFILL  ctx=%-6u  %.1f tok/s  (wall %.2f ms / kernel %.2f ms) ",
        ansi::BOLD, ctx_len, tok_per_s, total_ms, kernel_ms);
    std::printf("%s\n", ansi::RESET);

    std::printf("  %s%-3s  %-22s  %5s  %9s  %7s  %7s  %4s%s\n",
        ansi::BOLD, "#", "Kernel", "calls", "total ms", "avg ms", "max ms", "%", ansi::RESET);
    std::printf("  ");
    for (int i = 0; i < W - 4; ++i) std::putchar('-');
    std::printf("\n");

    const int n_show = std::min((int)stats.size(), 20);
    for (int i = 0; i < n_show; ++i) {
        const auto& s = stats[i];
        const double pct = total_ns ? double(s.total_ns) * 100.0 / total_ns : 0.0;
        const char* col = (i == 0) ? ansi::RED :
                          (i  < 3) ? ansi::YEL :
                          (pct < 1.0) ? ansi::DIM : ansi::RESET;
        std::printf("  %s%3d  %-22s  %5u  %9.3f  %7.4f  %7.4f  %3.0f%%%s\n",
            col, i + 1, s.name.c_str(), s.calls,
            s.total_ms(), s.avg_ms(), s.max_ms(), pct, ansi::RESET);
    }
    std::printf("  ");
    for (int i = 0; i < W - 4; ++i) std::putchar('-');
    std::printf("\n  %sΣ   %-22s              %9.3f ms  →  %.2f tok/s%s\n\n",
        ansi::BOLD, "TOTAL", kernel_ms, tok_per_s, ansi::RESET);
    std::fflush(stdout);
}

// ---------------------------------------------------------------------------
// Render — returns number of lines printed
// ---------------------------------------------------------------------------
static int render(
    const std::vector<ie::KernelProfiler::Stat>& stats,
    const char* device_name,
    const char* kv_mode,
    int step,
    uint32_t ctx_len,
    double step_ms,
    double rolling_tok_s)
{
    int lines = 0;
    constexpr int W = 90;  // display width

    uint64_t total_ns = 0;
    for (const auto& s : stats) total_ns += s.total_ns;
    const double total_ms_kernels = double(total_ns) * 1e-6;

    // ── Header box ──────────���───────────────────────────────────────────────
    auto hline = [&](char l, char m, char r, char fill) {
        std::putchar(l);
        for (int i = 0; i < W - 2; ++i) std::putchar(fill);
        std::putchar(r);
        std::putchar('\n');
        ++lines;
    };

    // Get current time
    auto now = std::chrono::system_clock::now();
    auto t   = std::chrono::system_clock::to_time_t(now);
    char timebuf[16];
    std::strftime(timebuf, sizeof(timebuf), "%H:%M:%S", std::localtime(&t));

    hline('\xe2', '\x95', '\x90', '\xe2');  // ╔═...═╗  — avoid non-ASCII in source
    // Simpler: just use plain ASCII box since terminal support varies
    std::printf("+%s%s ie-kernel-monitor%s  %s%s%s  %s%s%s",
        ansi::BOLD, ansi::WHT, ansi::RESET, ansi::DIM, device_name, ansi::RESET,
        ansi::DIM, timebuf, ansi::RESET);
    // pad to width
    int used = 2 + 19 + 2 + (int)std::strlen(device_name) + 2 + 8 + 2;
    for (int i = used; i < W - 1; ++i) std::putchar(' ');
    std::printf("+\n"); ++lines;

    std::printf("| ctx=%-6u  kv=%-14s  step %-6d  "
                "%s%.1f tok/s%s (kernel total %.2f ms / step %.2f ms) ",
        ctx_len, kv_mode, step,
        ansi::GRN, rolling_tok_s, ansi::RESET,
        total_ms_kernels, step_ms);
    used = 3 + 11 + 20 + 11 + 8 + 40;
    for (int i = used; i < W - 1; ++i) std::putchar(' ');
    std::printf("|\n"); ++lines;

    // Print plain separator
    std::printf("+");
    for (int i = 0; i < W - 2; ++i) std::putchar('-');
    std::printf("+\n"); ++lines;

    // ── Column header ───────────────────────────────���───────────────���───────
    std::printf("\n"); ++lines;
    std::printf("  %s%-3s  %-22s  %5s  %9s  %7s  %7s  %7s  %4s  %s%s\n",
        ansi::BOLD,
        "#", "Kernel", "calls", "total ms", "avg ms", "min ms", "max ms", "%",
        "bar", ansi::RESET);
    std::printf("  ");
    for (int i = 0; i < W - 4; ++i) std::putchar('-');
    std::printf("\n");
    lines += 2;

    // ── Kernel rows ─────────────────────────────────────────────────────────
    constexpr int BAR_W = 20;
    const int n_show = std::min((int)stats.size(), 30);
    for (int i = 0; i < n_show; ++i) {
        const auto& s = stats[i];
        const double pct = total_ns ? double(s.total_ns) * 100.0 / total_ns : 0.0;
        const int bar = std::max(0, std::min(BAR_W, (int)(pct / 100.0 * BAR_W + 0.5)));

        const char* col = (i == 0) ? ansi::RED :
                          (i  < 3) ? ansi::YEL :
                          (pct < 1.0) ? ansi::DIM : ansi::RESET;

        std::printf("  %s%3d  %-22s  %5u  %9.3f  %7.4f  %7.4f  %7.4f  %3.0f%%  ",
            col, i + 1, s.name.c_str(), s.calls,
            s.total_ms(), s.avg_ms(), s.min_ms(), s.max_ms(), pct);
        for (int b = 0; b < bar; ++b) std::printf("\xe2\x96\x88");  // █
        for (int b = bar; b < BAR_W; ++b) std::printf(" ");
        std::printf("%s\n", ansi::RESET);
        ++lines;
    }

    if ((int)stats.size() > n_show) {
        std::printf("  %s  ... %d more kernels (all < 0.1ms)%s\n",
            ansi::DIM, (int)stats.size() - n_show, ansi::RESET);
        ++lines;
    }

    // ── Footer ───────────────────────────────────────────────────────���──────
    std::printf("  ");
    for (int i = 0; i < W - 4; ++i) std::putchar('-');
    std::printf("\n");
    std::printf("  %sΣ   %-22s              %9.3f ms  →  %.2f tok/s%s\n",
        ansi::BOLD, "TOTAL", total_ms_kernels, 1000.0 / step_ms, ansi::RESET);
    std::printf("\n");
    lines += 3;

    std::fflush(stdout);
    return lines;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    std::string gguf_path    = "/home/weezy/.seal/models/Qwen3.6-35B-A3B-Q4_K_M.gguf";
    std::string prompt       = "Explain step by step how the attention mechanism works "
                               "in a transformer model, including queries, keys, and values. "
                               "Then describe how multi-head attention differs from single-head.";
    uint32_t    decode_steps = 128;   // decode window; resets + re-prefills when exhausted
    uint32_t    warmup       = 3;
    bool        int8_kv      = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--gguf"   && i + 1 < argc) gguf_path    = argv[++i];
        else if (a == "--prompt" && i + 1 < argc) prompt       = argv[++i];
        else if (a == "--decode" && i + 1 < argc) decode_steps = uint32_t(std::atoi(argv[++i]));
        else if (a == "--warmup" && i + 1 < argc) warmup       = uint32_t(std::atoi(argv[++i]));
        else if (a == "--int8-kv")                int8_kv      = true;
    }

    signal(SIGINT, on_sigint);

    // ── Open GGUF + tokenizer ────────────────────────────────────────────────
    ie::GgufReader g;
    if (auto e = g.open(gguf_path); !e.empty()) {
        std::fprintf(stderr, "gguf: %s\n", e.c_str()); return 1;
    }
    ie::Tokenizer tok;
    if (auto e = tok.load_from_gguf(g); !e.empty()) {
        std::fprintf(stderr, "tokenizer: %s\n", e.c_str()); return 1;
    }

    // Build ChatML prompt and encode — same tokens every single run.
    std::vector<ie::ChatTurn> turns = {{"user", prompt}};
    const std::string chatml = ie::build_chatml_prompt(turns, /*gen_prompt=*/true, /*think=*/false);
    const std::vector<int32_t> prompt_ids = tok.encode(chatml, /*allow_special=*/true);
    if (prompt_ids.empty()) { std::fprintf(stderr, "empty prompt after encode\n"); return 1; }

    const uint32_t prompt_len = uint32_t(prompt_ids.size());
    const uint32_t max_ctx    = prompt_len + decode_steps + warmup + 4;

    std::fprintf(stderr, "Prompt: \"%s\"\n  → %u tokens\n", prompt.c_str(), prompt_len);

    // ── Load model ───────────────────────────────────────────────────────────
    ie::DeviceAllocator alloc;
    if (auto e = alloc.init(); !e.empty()) {
        std::fprintf(stderr, "alloc: %s\n", e.c_str()); return 1;
    }
    auto& q = alloc.queue();
    std::fprintf(stderr, "Device: %s\nLoading model...\n",
        q.get_device().get_info<sycl::info::device::name>().c_str());

    ie::QwenConfig cfg;
    ie::QwenModel  model;
    if (auto e = model.load(alloc, g, cfg); !e.empty()) {
        std::fprintf(stderr, "model.load: %s\n", e.c_str()); return 1;
    }
    std::fprintf(stderr, "Loaded.\n");

    // ── Caches ───────────────────────────────────────────────────────────────
    const uint32_t L_full = cfg.n_layers / cfg.full_attn_interval;
    const uint32_t L_lin  = cfg.n_layers - L_full;
    ie::KvCache kv;
    {
        ie::KvCacheConfig kvcfg{};
        kvcfg.n_layers_full = L_full;
        kvcfg.n_kv_heads    = cfg.n_kv_heads;
        kvcfg.max_ctx       = max_ctx;
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
    if (auto e = model.ensure_workspace(prompt_len); !e.empty()) {
        std::fprintf(stderr, "ws: %s\n", e.c_str()); return 1;
    }
    if (auto e = model.ensure_attn_partials(max_ctx); !e.empty()) {
        std::fprintf(stderr, "attn_partials: %s\n", e.c_str()); return 1;
    }

    // ── GPU buffers ──────────────────────────────────────────────────────────
    auto* d_prompt = static_cast<int32_t*>(alloc.malloc(uint64_t(prompt_len) * sizeof(int32_t)));
    auto* d_tok    = static_cast<int32_t*>(alloc.malloc(sizeof(int32_t)));
    auto* d_logits = static_cast<sycl::half*>(alloc.malloc(uint64_t(cfg.vocab) * sizeof(sycl::half)));
    q.memcpy(d_prompt, prompt_ids.data(), uint64_t(prompt_len) * sizeof(int32_t)).wait();
    // Fixed decode token — keeps every run identical so timings are comparable.
    const int32_t decode_tok = (tok.bos_token_id() >= 0) ? tok.bos_token_id() : 0;
    q.memcpy(d_tok, &decode_tok, sizeof(int32_t)).wait();

    // ── Prefill lambda (called on startup and after each KV wrap) ────────────
    auto do_prefill = [&]() -> std::pair<std::vector<ie::KernelProfiler::Stat>, double> {
        kv.reset();
        dn.reset(q);
        q.wait();

        ie::KernelProfiler pprof;
        ie::g_profiler = &pprof;
        pprof.begin_step();
        const double t0 = now_ms();

        uint32_t pos = 0;
        const uint32_t chunk = 256;
        while (pos + chunk <= prompt_len) {
            model.forward(q, d_prompt + pos, chunk, pos, kv, dn, d_logits).wait();
            pos += chunk;
        }
        if (pos < prompt_len) {
            model.forward(q, d_prompt + pos, prompt_len - pos, pos, kv, dn, d_logits).wait();
        }

        const double ms = now_ms() - t0;
        auto stats = pprof.harvest();
        ie::g_profiler = nullptr;
        return {stats, ms};
    };

    // ── Initial prefill ──────────────────────────────────────────────────────
    std::fprintf(stderr, "Prefilling...\n");
    auto [pre_stats, pre_ms] = do_prefill();
    render_prefill(pre_stats, prompt_len, pre_ms, double(prompt_len) / pre_ms * 1000.0);

    // ── Warmup ───────────────────────────────────────────────────────────────
    std::fprintf(stderr, "Warming up (%u decode steps)...\n", warmup);
    uint32_t cur_pos = prompt_len;
    for (uint32_t s = 0; s < warmup; ++s) {
        model.forward(q, d_tok, 1, cur_pos, kv, dn, d_logits).wait();
        cur_pos++;
    }
    std::fprintf(stderr, "Live decode monitor — Ctrl+C to stop.\n\n");

    // ── Live decode loop ─────────────────────────────────────────────────────
    const std::string device_name = q.get_device().get_info<sycl::info::device::name>();
    const std::string kv_mode = int8_kv ? "INT8" : "fp16";

    std::deque<double> step_times;
    constexpr int ROLLING_N = 10;
    int prev_lines = 0;
    int step = 0;
    ie::KernelProfiler prof;
    ie::g_profiler = &prof;

    std::printf("\033[2J\033[H");

    while (g_running) {
        // KV full — reset and re-prefill with the same prompt, then re-warm.
        if (cur_pos >= max_ctx - 2) {
            ie::g_profiler = nullptr;
            if (prev_lines > 0) { ansi::cursor_up(prev_lines); ansi::clear_to_end(); }
            std::printf("  [ re-prefilling... ]\n"); std::fflush(stdout);
            auto [_, ms2] = do_prefill(); (void)ms2;
            cur_pos = prompt_len;
            for (uint32_t s = 0; s < warmup; ++s) {
                model.forward(q, d_tok, 1, cur_pos, kv, dn, d_logits).wait();
                cur_pos++;
            }
            step_times.clear();
            prev_lines = 0;
            ie::g_profiler = &prof;
        }

        prof.begin_step();
        const double t0 = now_ms();
        model.forward(q, d_tok, 1, cur_pos, kv, dn, d_logits).wait();
        const double step_ms = now_ms() - t0;

        step_times.push_back(step_ms);
        if ((int)step_times.size() > ROLLING_N) step_times.pop_front();
        const double avg_ms = std::accumulate(step_times.begin(), step_times.end(), 0.0)
                              / step_times.size();

        auto stats = prof.harvest();
        if (prev_lines > 0) { ansi::cursor_up(prev_lines); ansi::clear_to_end(); }
        prev_lines = render(stats, device_name.c_str(), kv_mode.c_str(),
                            step + 1, cur_pos, step_ms, 1000.0 / avg_ms);
        cur_pos++;
        step++;
    }

    ie::g_profiler = nullptr;
    std::printf("\nStopped after %d steps.\n", step);
    return 0;
}
