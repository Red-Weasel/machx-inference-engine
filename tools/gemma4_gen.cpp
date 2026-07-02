// tools/gemma4_gen.cpp — Gemma 4 greedy-generation smoke. Loads on one B70,
// tokenizes with the GGUF tokenizer, prefills, greedy-decodes n_new tokens.
// The lead runs it on "The capital of France is" expecting " Paris".
// (Sliding-window layers use full attention — exact for T <= 1024.)
#include "ie/gemma4.hpp"
#include "ie/gguf.hpp"
#include "ie/model_config.hpp"
#include "ie/tokenizer.hpp"
#include "ie/kv_cache.hpp"
#include "ie/allocator.hpp"
#include "ie/kernel_profiler.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
static double median(std::vector<double> v) {
    if (v.empty()) return 0; std::sort(v.begin(), v.end()); return v[v.size() / 2];
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s <gguf> \"<prompt>\" [n_new=8] [bench]\n", argv[0]); return 2; }
    const std::string gguf = argv[1], prompt = argv[2];
    const int n_new = (argc > 3) ? std::atoi(argv[3]) : 8;
    bool bench = false, bench_decode = false, profile = false;
    for (int i = 3; i < argc; ++i) {
        if (std::string(argv[i]) == "bench") bench = true;
        if (std::string(argv[i]) == "bench-decode") bench_decode = true;
        if (std::string(argv[i]) == "profile") profile = true;
    }

    ie::GgufReader g;
    if (auto e = g.open(gguf); !e.empty()) { std::fprintf(stderr, "gguf: %s\n", e.c_str()); return 1; }
    ie::GemmaConfig cfg;
    if (auto e = ie::read_gemma4_config(g, cfg); !e.empty()) { std::fprintf(stderr, "config: %s\n", e.c_str()); return 1; }
    ie::Tokenizer tok;
    if (auto e = tok.load_from_gguf(g); !e.empty()) { std::fprintf(stderr, "tok: %s\n", e.c_str()); return 1; }

    ie::DeviceAllocator alloc;
    if (auto e = alloc.init(); !e.empty()) { std::fprintf(stderr, "alloc: %s\n", e.c_str()); return 1; }
    ie::Gemma4Model m;
    if (auto e = m.load(alloc, g, cfg); !e.empty()) { std::fprintf(stderr, "LOAD: %s\n", e.c_str()); return 1; }

    std::vector<int32_t> ids;
    if (prompt.rfind("ids:", 0) == 0) {            // raw token ids: "ids:2 818 5279 ..."
        std::string s = prompt.substr(4);
        size_t p = 0;
        while (p < s.size()) {
            while (p < s.size() && s[p] == ' ') ++p;
            if (p >= s.size()) break;
            ids.push_back(std::atoi(s.c_str() + p));
            while (p < s.size() && s[p] != ' ') ++p;
        }
    } else {
        ids = tok.encode(prompt, /*allow_special=*/true);
    }
    if (ids.empty()) { std::fprintf(stderr, "encode produced 0 tokens\n"); return 1; }
    const uint32_t T = uint32_t(ids.size());
    const uint32_t V = cfg.vocab;
    const uint32_t max_ctx = T + uint32_t(n_new) + 8;   // +8 covers decode-bench warmup steps

    if (auto e = m.ensure_workspace(std::max<uint32_t>(T, 1)); !e.empty()) { std::fprintf(stderr, "ws: %s\n", e.c_str()); return 1; }
    if (auto e = m.ensure_kv(max_ctx); !e.empty()) { std::fprintf(stderr, "kv: %s\n", e.c_str()); return 1; }

    std::printf("prompt: '%s'\nids (%u, bos=%d):", prompt.c_str(), T, tok.bos_token_id());
    for (int32_t id : ids) std::printf(" %d", id);
    std::printf("\n");

    auto& q = alloc.queue();
    auto* d_ids = sycl::malloc_device<int32_t>(std::max<uint32_t>(T, 1), q);
    auto* d_logits = sycl::malloc_device<sycl::half>(V, q);
    ie::KvCache kv;  // unused — Gemma self-manages KV
    std::vector<sycl::half> logits(V);

    auto argmax = [&]() -> int32_t {
        int32_t bi = 0; float best = -1e30f;
        for (uint32_t i = 0; i < V; ++i) { float v = float(logits[i]); if (v > best) { best = v; bi = int32_t(i); } }
        return bi;
    };
    auto run = [&](const int32_t* host_ids, uint32_t n, uint32_t start_pos) {
        q.memcpy(d_ids, host_ids, n * sizeof(int32_t)).wait();
        m.forward(q, d_ids, n, start_pos, kv, d_logits).wait_and_throw();
        q.memcpy(logits.data(), d_logits, V * sizeof(sycl::half)).wait();
    };

    if (bench) {
        // Prefill-perf A/B: time the T-token prefill (warmup + 3 runs), report
        // tok/s. Compare default (gemm proj) vs IE_GEMMA4_NO_GEMM_PROJ=1 and
        // IE_GEMMA4_EXPERT_SYNC=1. Greedy argmax printed as a coherence check.
        run(ids.data(), T, 0);               // warmup / JIT
        std::vector<double> ms;
        for (int r = 0; r < 3; ++r) { const double t0 = now_ms(); run(ids.data(), T, 0); ms.push_back(now_ms() - t0); }
        const double pre_ms = median(ms);
        std::printf("BENCH prefill: T=%u  %.2f ms  %.1f tok/s   gemm_proj=%s expert_sync=%s  argmax=%d\n",
                    T, pre_ms, 1000.0 * T / pre_ms,
                    std::getenv("IE_GEMMA4_NO_GEMM_PROJ") ? "off" : "ON",
                    std::getenv("IE_GEMMA4_EXPERT_SYNC") ? "ON" : "off", argmax());
        sycl::free(d_ids, q); sycl::free(d_logits, q);
        return 0;
    }

    if (profile) {
        // Per-kernel breakdown (run with IE_QUEUE_PROFILING=1 for real device
        // times). 'profile-decode' arg → profile a single T=1 decode step (after
        // a prefill + a few warmup decodes); else profile the T-token prefill.
        bool prof_decode = false;
        for (int i = 3; i < argc; ++i) if (std::string(argv[i]) == "profile-decode") prof_decode = true;
        if (prof_decode) {
            run(ids.data(), T, 0);
            int32_t nx = argmax(); uint32_t sp = T;
            for (int w = 0; w < 4; ++w) { run(&nx, 1, sp); nx = argmax(); ++sp; }
            ie::KernelProfiler prof; ie::g_profiler = &prof; prof.begin_step();
            run(&nx, 1, sp);
            q.wait();
            auto stats = prof.harvest(); ie::g_profiler = nullptr;
            double tot = 0; for (auto& s : stats) tot += s.total_ms();
            std::sort(stats.begin(), stats.end(),
                      [](const ie::KernelProfiler::Stat& a, const ie::KernelProfiler::Stat& b) {
                          return a.total_ns > b.total_ns; });
            std::printf("PROFILE decode step  total=%.3f ms  (%.1f tok/s)  n_launches=%zu\n",
                        tot, 1000.0 / tot, stats.size());
            for (auto& s : stats)
                std::printf("  %-28s %6u %10.4f %7.1f%%\n",
                            s.name.c_str(), s.calls, s.total_ms(), 100.0 * s.total_ms() / tot);
            sycl::free(d_ids, q); sycl::free(d_logits, q);
            return 0;
        }
        // Per-kernel prefill breakdown. Warm once, then harvest one prefill.
        run(ids.data(), T, 0);                     // warmup / JIT
        ie::KernelProfiler prof; ie::g_profiler = &prof; prof.begin_step();
        run(ids.data(), T, 0);
        q.wait();
        auto stats = prof.harvest(); ie::g_profiler = nullptr;
        double tot = 0; for (auto& s : stats) tot += s.total_ms();
        std::sort(stats.begin(), stats.end(),
                  [](const ie::KernelProfiler::Stat& a, const ie::KernelProfiler::Stat& b) {
                      return a.total_ns > b.total_ns; });
        std::printf("PROFILE prefill T=%u  total=%.2f ms  (%.1f tok/s)\n", T, tot, 1000.0 * T / tot);
        std::printf("  %-28s %6s %10s %8s %8s\n", "kernel", "calls", "total_ms", "avg_ms", "%%");
        for (auto& s : stats)
            std::printf("  %-28s %6u %10.3f %8.4f %7.1f%%\n",
                        s.name.c_str(), s.calls, s.total_ms(), s.avg_ms(),
                        100.0 * s.total_ms() / tot);
        sycl::free(d_ids, q); sycl::free(d_logits, q);
        return 0;
    }

    if (bench_decode) {
        // Decode-perf: prefill once, then time steady-state single-token decode
        // (the real tg number — ie-bench mis-detects gemma4 as qwen35moe). n_new
        // = number of decode steps to time (use >= 64 for a stable number). Feeds
        // the model's own greedy output back so KV/positions advance realistically.
        run(ids.data(), T, 0);
        int32_t next = argmax();
        uint32_t start_pos = T;
        for (int w = 0; w < 4; ++w) {     // warmup decodes (JIT + heat)
            run(&next, 1, start_pos); next = argmax(); ++start_pos;
        }
        const double t0 = now_ms();
        int steps = std::max(n_new, 1);
        for (int s = 0; s < steps; ++s) {
            run(&next, 1, start_pos); next = argmax(); ++start_pos;
        }
        const double dt = now_ms() - t0;
        std::printf("BENCH decode: %d steps  %.2f ms  %.2f tok/s   intdot_proj=%s\n",
                    steps, dt, 1000.0 * steps / dt,
                    std::getenv("IE_GEMMA4_INTDOT_PROJ") ? "ON" : "off");
        sycl::free(d_ids, q); sycl::free(d_logits, q);
        return 0;
    }

    run(ids.data(), T, 0);
    {
        std::vector<uint32_t> idx(V);
        for (uint32_t i = 0; i < V; ++i) idx[i] = i;
        std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                          [&](uint32_t a, uint32_t b) { return float(logits[a]) > float(logits[b]); });
        std::printf("prefill top-5:");
        for (int k = 0; k < 5; ++k) {
            std::vector<int32_t> one{int32_t(idx[k])};
            std::printf(" [%u %.3f '%s']", idx[k], float(logits[idx[k]]), tok.decode(one, true, {}).c_str());
        }
        std::printf("\n");
    }

    std::vector<int32_t> gen;
    int32_t next = argmax(); gen.push_back(next);
    uint32_t start_pos = T;
    for (int s = 1; s < n_new; ++s) {
        if (next == tok.eos_token_id()) break;
        run(&next, 1, start_pos);
        next = argmax(); gen.push_back(next); ++start_pos;
    }
    std::printf("generated ids (%zu):", gen.size());
    for (int32_t id : gen) std::printf(" %d", id);
    std::printf("\ngenerated text: '%s'\n", tok.decode(gen, true, {}).c_str());

    sycl::free(d_ids, q); sycl::free(d_logits, q);
    return 0;
}
