// tools/ie_kv_scale.cpp — KV-mode context-scaling benchmark.
//
// Runs fp16 and INT8 KV caches at each target context length,
// 3 timed iterations per (ctx, mode) pair.  Reports decode tok/s,
// prefill tok/s, and per-kernel breakdown for fa2_partial, gemv_q6k,
// gemv_q4k at every context size so the INT8 crossover point is visible.
//
// Base context: all benchmark_prompts/ files concatenated + token-repeated
// to reach the largest target context.  Same token sequence is sliced to
// each smaller context — so ctx=256 is a strict prefix of ctx=4096.
//
// Usage:
//   ie-kv-scale [--gguf path] [--prompts-dir dir] [--decode N]
//               [--runs N] [--warmup N] [--max-ctx N]

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
    int32_t best = 0; float bval = float(v[0]);
    for (size_t i = 1; i < v.size(); ++i) {
        float f = float(v[i]);
        if (f > bval) { bval = f; best = int32_t(i); }
    }
    return best;
}

// ---------------------------------------------------------------------------
// Per-run kernel snapshot — contains only the metrics we care about.
struct KSnap {
    double attn_partial_ms  = 0;   // fa2_partial_fp16 or fa2_partial_int8
    double attn_partial_pct = 0;
    double gemv_q6k_ms      = 0;
    double gemv_q6k_pct     = 0;
    double gemv_q4k_ms      = 0;
    double gemv_q4k_pct     = 0;
    double total_kernel_ms  = 0;
};

static KSnap extract(const std::vector<ie::KernelProfiler::Stat>& stats, bool int8_kv) {
    const char* attn_name = int8_kv ? "fa2_partial_int8" : "fa2_partial_fp16";
    uint64_t total_ns = 0;
    for (const auto& s : stats) total_ns += s.total_ns;
    KSnap snap;
    snap.total_kernel_ms = double(total_ns) * 1e-6;
    for (const auto& s : stats) {
        const double pct = total_ns ? double(s.total_ns) * 100.0 / total_ns : 0.0;
        if      (s.name == attn_name)  { snap.attn_partial_ms  = s.total_ms(); snap.attn_partial_pct  = pct; }
        else if (s.name == "gemv_q6k") { snap.gemv_q6k_ms      = s.total_ms(); snap.gemv_q6k_pct      = pct; }
        else if (s.name == "gemv_q4k") { snap.gemv_q4k_ms      = s.total_ms(); snap.gemv_q4k_pct      = pct; }
    }
    return snap;
}

struct RunResult {
    double prefill_ms  = 0;
    double decode_ms   = 0;
    uint32_t decode_n  = 0;
    uint32_t plen      = 0;
    KSnap    kernels;

    double decode_tok_s()  const { return decode_n && decode_ms > 0 ? 1000.0 * decode_n / decode_ms : 0; }
    double prefill_tok_s() const { return plen && prefill_ms > 0 ? 1000.0 * plen / prefill_ms : 0; }
    double avg_ms_tok()    const { return decode_n ? decode_ms / decode_n : 0; }
};

struct ModeResult {
    bool     valid       = false;
    double   med_tok_s   = 0;
    double   min_tok_s   = 0;
    double   max_tok_s   = 0;
    double   prefill_ms  = 0;
    double   prefill_ts  = 0;
    double   avg_ms_tok  = 0;
    KSnap    kernels;
    uint32_t plen        = 0;
};

// ---------------------------------------------------------------------------
// Run one (ctx, kv_mode) cell: N timed iterations, return ModeResult.
// Assumes model, dn, GPU buffers are already set up.
static ModeResult run_cell(
    ie::QwenModel&     model,
    ie::DeviceAllocator& alloc,
    ie::QwenConfig&    cfg,
    const uint32_t     L_full,
    const uint32_t     L_lin,
    ie::DeltaNetState& dn,
    sycl::queue&       q,
    ie::KernelProfiler& prof,
    const int32_t*     d_base,      // full base context on device
    uint32_t           ctx_len,     // tokens to prefill
    uint32_t           decode_n,
    uint32_t           n_runs,
    uint32_t           warmup,
    bool               int8_kv,
    int32_t            start_tok,
    sycl::half*        d_logits,
    int32_t*           d_tok,
    std::vector<sycl::half>& h_logits)
{
    // Init KV cache for this mode + context length.
    const uint32_t max_ctx = ctx_len + decode_n + warmup + 8;
    ie::KvCache kv;
    {
        ie::KvCacheConfig kvcfg{};
        kvcfg.n_layers_full = L_full;
        kvcfg.n_kv_heads    = cfg.n_kv_heads;
        kvcfg.max_ctx       = max_ctx;
        kvcfg.head_dim      = cfg.head_dim;
        kvcfg.use_int8      = int8_kv;
        if (auto e = kv.init(alloc, kvcfg); !e.empty()) {
            std::fprintf(stderr, "kv.init: %s\n", e.c_str());
            return {};
        }
    }

    std::vector<RunResult> runs(n_runs);

    for (uint32_t r = 0; r < n_runs; ++r) {
        std::printf("      run %u/%u  prefill...", r + 1, n_runs); std::fflush(stdout);

        kv.reset();
        dn.reset(q);
        q.wait();

        // Prefill (unprofiled — we want decode kernel breakdown only).
        ie::g_profiler = nullptr;
        const double t_pre0 = now_ms();
        {
            uint32_t pos = 0;
            constexpr uint32_t CHUNK = 256;
            while (pos + CHUNK <= ctx_len) {
                model.forward(q, d_base + pos, CHUNK, pos, kv, dn, d_logits).wait();
                pos += CHUNK;
            }
            if (pos < ctx_len)
                model.forward(q, d_base + pos, ctx_len - pos, pos, kv, dn, d_logits).wait();
        }
        const double prefill_ms = now_ms() - t_pre0;

        // Warmup decode (not profiled, not timed).
        std::printf("  warmup..."); std::fflush(stdout);
        uint32_t pos = ctx_len;
        int32_t next_tok = start_tok;
        for (uint32_t w = 0; w < warmup; ++w) {
            q.memcpy(d_tok, &next_tok, sizeof(int32_t)).wait();
            model.forward(q, d_tok, 1, pos, kv, dn, d_logits).wait();
            q.memcpy(h_logits.data(), d_logits, h_logits.size() * sizeof(sycl::half)).wait();
            next_tok = argmax_fp16(h_logits);
            pos++;
        }

        // Timed + profiled decode — accumulate all steps in one harvest.
        std::printf("  decode..."); std::fflush(stdout);
        ie::g_profiler = &prof;
        prof.begin_step();
        const double t_dec0 = now_ms();
        for (uint32_t d = 0; d < decode_n; ++d) {
            q.memcpy(d_tok, &next_tok, sizeof(int32_t)).wait();
            model.forward(q, d_tok, 1, pos, kv, dn, d_logits).wait();
            q.memcpy(h_logits.data(), d_logits, h_logits.size() * sizeof(sycl::half)).wait();
            next_tok = argmax_fp16(h_logits);
            pos++;
        }
        const double decode_ms = now_ms() - t_dec0;
        auto stats = prof.harvest();
        ie::g_profiler = nullptr;

        runs[r] = {prefill_ms, decode_ms, decode_n, ctx_len, extract(stats, int8_kv)};
        std::printf("  %.2f tok/s (prefill %.1f ms → %.1f tok/s)\n",
            runs[r].decode_tok_s(), prefill_ms, runs[r].prefill_tok_s());
    }

    // Median / min / max.
    std::vector<double> ts(n_runs);
    for (uint32_t r = 0; r < n_runs; ++r) ts[r] = runs[r].decode_tok_s();
    std::vector<size_t> ord(n_runs);
    std::iota(ord.begin(), ord.end(), 0);
    std::sort(ord.begin(), ord.end(), [&](size_t a, size_t b){ return ts[a] < ts[b]; });

    const size_t mi     = ord[n_runs / 2];
    const RunResult& mr = runs[mi];

    ModeResult res;
    res.valid      = true;
    res.med_tok_s  = ts[mi];
    res.min_tok_s  = ts[ord.front()];
    res.max_tok_s  = ts[ord.back()];
    res.prefill_ms = mr.prefill_ms;
    res.prefill_ts = mr.prefill_tok_s();
    res.avg_ms_tok = mr.avg_ms_tok();
    res.kernels    = mr.kernels;
    res.plen       = ctx_len;
    return res;
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    std::string gguf_path   = "/home/weezy/.seal/models/Qwen3.6-35B-A3B-Q4_K_M.gguf";
    std::string prompts_dir = "benchmark_prompts";
    uint32_t    decode_n    = 128;
    uint32_t    n_runs      = 3;
    uint32_t    warmup      = 3;
    uint32_t    max_ctx     = 4096;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--gguf"        && i+1 < argc) gguf_path   = argv[++i];
        else if (a == "--prompts-dir" && i+1 < argc) prompts_dir = argv[++i];
        else if (a == "--decode"      && i+1 < argc) decode_n    = uint32_t(std::atoi(argv[++i]));
        else if (a == "--runs"        && i+1 < argc) n_runs      = uint32_t(std::atoi(argv[++i]));
        else if (a == "--warmup"      && i+1 < argc) warmup      = uint32_t(std::atoi(argv[++i]));
        else if (a == "--max-ctx"     && i+1 < argc) max_ctx     = uint32_t(std::atoi(argv[++i]));
    }

    const std::vector<uint32_t> target_ctxs = {128, 256, 512, 1024, 2048, 4096};

    // ── GGUF + tokenizer ─────────────────────────────────────────────────────
    ie::GgufReader g;
    if (auto e = g.open(gguf_path); !e.empty()) {
        std::fprintf(stderr, "gguf: %s\n", e.c_str()); return 1;
    }
    ie::Tokenizer tok;
    if (auto e = tok.load_from_gguf(g); !e.empty()) {
        std::fprintf(stderr, "tokenizer: %s\n", e.c_str()); return 1;
    }

    // ── Build base context ────────────────────────────────────────────────────
    // Concatenate all prompt files (raw text, no ChatML), tokenize, repeat to
    // reach max_ctx tokens.  ctx=256 is always a strict prefix of ctx=4096.
    std::string base_raw;
    for (const char* fn : {"01_short_chat.txt","02_long_instruction.txt",
                            "03_codegen.txt","04_math_reasoning.txt","05_long_context.txt"}) {
        std::string t = read_file(prompts_dir + "/" + fn);
        if (!t.empty()) { base_raw += t; base_raw += "\n\n"; }
    }
    if (base_raw.empty()) {
        std::fprintf(stderr, "No prompt files found in %s\n", prompts_dir.c_str()); return 1;
    }
    std::vector<int32_t> base_tokens = tok.encode(base_raw, /*allow_special=*/false);
    // Repeat until we have enough for the largest context.
    while (base_tokens.size() < max_ctx) {
        const size_t orig = base_tokens.size();
        base_tokens.resize(orig * 2);
        std::copy(base_tokens.begin(), base_tokens.begin() + orig,
                  base_tokens.begin() + orig);
    }
    base_tokens.resize(max_ctx);
    std::fprintf(stderr, "Base context: %zu tokens (repeated from %zu unique source tokens)\n",
        base_tokens.size(), tok.encode(base_raw, false).size());

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

    const uint32_t L_full = cfg.n_layers / cfg.full_attn_interval;
    const uint32_t L_lin  = cfg.n_layers - L_full;

    // DeltaNet state — sized once for the largest context + headroom.
    ie::DeltaNetState dn;
    if (auto e = dn.init(alloc, ie::DeltaNetStateConfig{
            L_lin, cfg.ssm_n_v_heads, cfg.ssm_head_dim, cfg.ssm_head_dim,
            cfg.ssm_inner * 2, cfg.ssm_conv_kernel}); !e.empty()) {
        std::fprintf(stderr, "dn: %s\n", e.c_str()); return 1;
    }
    if (auto e = model.ensure_workspace(256); !e.empty()) {
        std::fprintf(stderr, "ws: %s\n", e.c_str()); return 1;
    }
    // Attn partials must fit the largest (ctx + decode + warmup).
    if (auto e = model.ensure_attn_partials(max_ctx + decode_n + warmup + 8); !e.empty()) {
        std::fprintf(stderr, "attn_partials: %s\n", e.c_str()); return 1;
    }

    // ── GPU buffers ──────────────────────────────────────────────────────────
    auto* d_base   = static_cast<int32_t*>(alloc.malloc(uint64_t(max_ctx) * sizeof(int32_t)));
    auto* d_tok    = static_cast<int32_t*>(alloc.malloc(sizeof(int32_t)));
    auto* d_logits = static_cast<sycl::half*>(alloc.malloc(uint64_t(cfg.vocab) * sizeof(sycl::half)));
    q.memcpy(d_base, base_tokens.data(), uint64_t(max_ctx) * sizeof(int32_t)).wait();
    std::vector<sycl::half> h_logits(cfg.vocab);

    const int32_t start_tok = tok.bos_token_id() >= 0 ? tok.bos_token_id() : 0;
    ie::KernelProfiler prof;

    // ── Header ───────────────────────────────────────────────────────────────
    std::printf("ie-kv-scale  decode=%u  runs=%u  warmup=%u\n", decode_n, n_runs, warmup);
    std::printf("device: %s\n", device_name.c_str());
    std::printf("model:  Qwen3.6-35B-A3B-Q4_K_M\n");
    std::printf("================================================================\n\n");

    // ── Main benchmark loop ───────────────────────────────────────────────────
    // [ci][0]=fp16  [ci][1]=int8
    std::vector<std::array<ModeResult, 2>> results(target_ctxs.size());

    for (size_t ci = 0; ci < target_ctxs.size(); ++ci) {
        const uint32_t ctx = target_ctxs[ci];

        if (ctx > max_ctx) {
            std::printf(">>> ctx=%u  [SKIP — exceeds --max-ctx %u]\n\n", ctx, max_ctx);
            continue;
        }

        std::printf(">>> ctx=%u tokens\n", ctx);
        for (int mode = 0; mode < 2; ++mode) {
            const bool int8_kv = (mode == 1);
            std::printf("    mode=%s\n", int8_kv ? "INT8" : "fp16");
            results[ci][mode] = run_cell(
                model, alloc, cfg, L_full, L_lin, dn, q, prof,
                d_base, ctx, decode_n, n_runs, warmup, int8_kv,
                start_tok, d_logits, d_tok, h_logits);
        }

        const ModeResult& fp16 = results[ci][0];
        const ModeResult& int8 = results[ci][1];
        if (!fp16.valid || !int8.valid) { std::printf("\n"); continue; }

        const double delta_tok  = fp16.med_tok_s > 0 ?
            (int8.med_tok_s - fp16.med_tok_s) / fp16.med_tok_s * 100.0 : 0.0;
        const double delta_attn = fp16.kernels.attn_partial_ms > 0 ?
            (int8.kernels.attn_partial_ms - fp16.kernels.attn_partial_ms)
            / fp16.kernels.attn_partial_ms * 100.0 : 0.0;

        std::printf("\n    ── ctx=%u summary ─────────────────────────────────\n", ctx);
        std::printf("    FP16:  %.2f tok/s  fa2=%.1f ms (%.1f%%)  "
                    "gemv_q6k=%.1f ms (%.1f%%)  gemv_q4k=%.1f ms (%.1f%%)\n",
            fp16.med_tok_s,
            fp16.kernels.attn_partial_ms, fp16.kernels.attn_partial_pct,
            fp16.kernels.gemv_q6k_ms,     fp16.kernels.gemv_q6k_pct,
            fp16.kernels.gemv_q4k_ms,     fp16.kernels.gemv_q4k_pct);
        std::printf("    INT8:  %.2f tok/s  fa2=%.1f ms (%.1f%%)  "
                    "gemv_q6k=%.1f ms (%.1f%%)  gemv_q4k=%.1f ms (%.1f%%)\n",
            int8.med_tok_s,
            int8.kernels.attn_partial_ms, int8.kernels.attn_partial_pct,
            int8.kernels.gemv_q6k_ms,     int8.kernels.gemv_q6k_pct,
            int8.kernels.gemv_q4k_ms,     int8.kernels.gemv_q4k_pct);
        std::printf("    delta: tok/s %+.1f%%   fa2 %+.1f%%   winner: %s\n\n",
            delta_tok, delta_attn,
            delta_tok > 0.5 ? "INT8" : (delta_tok < -0.5 ? "FP16" : "TIE"));
    }

    // ── Comparison table ──────────────────────────────────────────────────────
    std::printf("================================================================\n");
    std::printf("## KV-mode context scaling  (decode=%u toks, runs=%u)\n\n", decode_n, n_runs);

    // Header
    std::printf("| %12s | %10s | %10s | %12s | %11s | %11s | %11s | %6s |\n",
        "Context toks", "FP16 tok/s", "INT8 tok/s", "INT8 delta %",
        "FP16 fa2 ms", "INT8 fa2 ms", "FA2 delta %", "Winner");
    std::printf("| %s | %s | %s | %s | %s | %s | %s | %s |\n",
        "------------", "----------:", "----------:", "------------:",
        "-----------:", "-----------:", "-----------:", "------");

    for (size_t ci = 0; ci < target_ctxs.size(); ++ci) {
        const uint32_t ctx   = target_ctxs[ci];
        const ModeResult& f  = results[ci][0];
        const ModeResult& i8 = results[ci][1];
        if (!f.valid || !i8.valid) {
            std::printf("| %12u | %10s | %10s | %12s | %11s | %11s | %11s | %6s |\n",
                ctx, "SKIP", "SKIP", "—", "—", "—", "—", "—");
            continue;
        }
        const double dtok  = f.med_tok_s > 0 ? (i8.med_tok_s - f.med_tok_s) / f.med_tok_s * 100.0 : 0;
        const double dfa2  = f.kernels.attn_partial_ms > 0 ?
            (i8.kernels.attn_partial_ms - f.kernels.attn_partial_ms)
            / f.kernels.attn_partial_ms * 100.0 : 0;
        const char* winner = dtok > 0.5 ? "INT8" : (dtok < -0.5 ? "FP16" : "TIE");
        std::printf("| %12u | %10.2f | %10.2f | %+12.1f | %11.1f | %11.1f | %+11.1f | %6s |\n",
            ctx, f.med_tok_s, i8.med_tok_s, dtok,
            f.kernels.attn_partial_ms, i8.kernels.attn_partial_ms, dfa2, winner);
    }
    std::printf("\n");

    // Extended detail table
    std::printf("## Per-mode detail\n\n");
    std::printf("| %12s | %4s | %10s | %9s | %10s | %9s | %9s | %13s |\n",
        "Context toks", "Mode", "Median tok/s", "Prefill ms", "Prefill tok/s",
        "Avg ms/tok", "Total ker ms", "fa2_partial ms");
    std::printf("| %s | %s | %s | %s | %s | %s | %s | %s |\n",
        "------------", "----", "----------:", "---------:", "------------:",
        "---------:", "------------:", "-------------:");
    for (size_t ci = 0; ci < target_ctxs.size(); ++ci) {
        for (int mode = 0; mode < 2; ++mode) {
            const ModeResult& r = results[ci][mode];
            if (!r.valid) continue;
            std::printf("| %12u | %4s | %10.2f | %9.1f | %12.1f | %9.3f | %12.1f | %13.1f |\n",
                target_ctxs[ci], mode == 0 ? "fp16" : "INT8",
                r.med_tok_s, r.prefill_ms, r.prefill_ts,
                r.avg_ms_tok, r.kernels.total_kernel_ms, r.kernels.attn_partial_ms);
        }
    }
    std::printf("\n");

    return 0;
}
