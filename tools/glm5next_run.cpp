// tools/glm5next_run.cpp — GLM-5.3-Flash end-to-end forward smoke + PPL (P2).
//
// Default: greedy continuation of a prompt (correctness smoke — coherent text
// is the cheapest whole-model probe). --ppl <file> runs the chunked NLL
// scorer over a text file: PPL in the single digits on normal text is the
// "graph is right" band; a mis-wired sublayer lands in the hundreds+.
//
// Two-card runs overlap prefill chunks by default. Sustained generation can
// also overlap MTP-drafted stage A with stage B; --no-pipeline and
// --no-pipedraft retain serial execution for comparisons.
//
// usage: ie-glm5next-run <model-00001-of-00006.gguf> [gpu] [--prompt "..."]
//        [--ngen N] [--ppl file] [--chunk T]
#include "ie/gguf.hpp"
#include "ie/glm5next.hpp"
#include "ie/model_config.hpp"
#include "ie/tokenizer.hpp"
#include "ie/kernel_profiler.hpp"

#include <chrono>
#include <csignal>
#include <future>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <string>
#include <vector>

using namespace ie;

namespace {
Glm5NextModel* g_term_m[2] = {};
uint32_t g_term_n = 0;
void on_term(int sig) {
    for (uint32_t i = 0; i < g_term_n; ++i)
        if (g_term_m[i]) g_term_m[i]->shutdown();
    _exit(128 + sig);
}
int32_t argmax_h(sycl::queue& q, const sycl::half* lg, uint32_t V) {
    std::vector<sycl::half> h(V);
    q.memcpy(h.data(), lg, uint64_t(V) * 2).wait();
    float mx = -1e30f; int32_t am = 0;
    for (uint32_t v = 0; v < V; ++v)
        if (float(h[v]) > mx) { mx = float(h[v]); am = int32_t(v); }
    return am;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.gguf> [gpu] [--prompt S] [--ngen N] "
                             "[--ppl file] [--chunk T] [--gpus N] "
                             "[--pipeline|--no-pipeline] [--pipedraft|--no-pipedraft]\n", argv[0]);
        return 2;
    }
    // chunk 1024 (was 512): each prefill chunk re-streams essentially the whole
    // expert bank, so doubling the chunk halves bytes per token — measured
    // 30.8 -> 39.1 tok/s unpinned, 44.4 pinned (2026-09-03). 2048 with a
    // shrunken IE_G5_ECACHE_MB thrashed the per-wave victim scan; 1024 with the
    // default cache is the geometry every long-context run today used.
    uint32_t gpu = 0, ngen = 32, chunk = 1024, n_gpus = 1, spec = 0, ppbench = 0, ctx_req = 0, prefill_n = 0;
    bool pipeline = true, pipedraft = false;
    int pipedraft_request = -1;  // auto unless explicitly enabled/disabled
    std::string prompt = "The capital of France is";
    std::string ppl_file;
    for (int i = 2; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--prompt") && i + 1 < argc) prompt = argv[++i];
        else if (!std::strcmp(argv[i], "--ngen") && i + 1 < argc) ngen = uint32_t(std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "--ppl") && i + 1 < argc) ppl_file = argv[++i];
        else if (!std::strcmp(argv[i], "--chunk") && i + 1 < argc) chunk = uint32_t(std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "--gpus") && i + 1 < argc) n_gpus = uint32_t(std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "--spec") && i + 1 < argc) spec = uint32_t(std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "--ppbench") && i + 1 < argc) ppbench = uint32_t(std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "--pipeline")) pipeline = true;
        else if (!std::strcmp(argv[i], "--no-pipeline")) pipeline = false;
        else if (!std::strcmp(argv[i], "--pipedraft")) pipedraft_request = 1;
        else if (!std::strcmp(argv[i], "--no-pipedraft")) pipedraft_request = 0;
        else if (!std::strcmp(argv[i], "--ctx") && i + 1 < argc) ctx_req = uint32_t(std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "--prefill") && i + 1 < argc) prefill_n = uint32_t(std::atoi(argv[++i]));
        else if (argv[i][0] != '-') gpu = uint32_t(std::atoi(argv[i]));
    }
    if (!chunk) { std::fprintf(stderr, "--chunk must be greater than zero\n"); return 2; }

    GgufReader g;
    if (auto e = g.open(argv[1]); !e.empty()) { std::fprintf(stderr, "open: %s\n", e.c_str()); return 1; }
    Glm5NextConfig cfg;
    if (auto e = read_glm5next_config(g, cfg); !e.empty()) { std::fprintf(stderr, "config: %s\n", e.c_str()); return 1; }
    Tokenizer tok;
    if (auto e = tok.load_from_gguf(g); !e.empty()) { std::fprintf(stderr, "tok: %s\n", e.c_str()); return 1; }

    // --ctx N: the KV/context length to size the runtime for. Default stays the
    // selection width (indexer_top_k + kpool - 1 = 2051), where dense attention
    // IS the sparse answer; above it the DSA indexer engages.
    const uint32_t n_sel_w = cfg.indexer_top_k + cfg.indexer_kpool - 1;
    const uint32_t n_sel = ctx_req ? std::max(ctx_req, n_sel_w) : n_sel_w;
    const uint32_t n_tf = cfg.n_transformer_layers();
    const uint32_t n_st = n_gpus >= 2 ? 2u : 1u;
    // Overlap the two cards for sustained generation. Keep short continuations,
    // scoring, prefill-only benchmarks and the plain-decode profiler free of
    // the MTP kit's load/cache cost. 64 tokens is a conservative dispatch policy,
    // not a measured universal break-even point; explicit switches always win.
    const bool auto_pipedraft = n_st == 2 && ngen >= 64 && !spec && !ppbench &&
        (ppl_file.empty() || prefill_n) && cfg.nextn_predict_layers > 0 &&
        !std::getenv("IE_QUEUE_PROFILING") && !std::getenv("IE_G5_LOGIT_DUMP") &&
        !std::getenv("IE_G5_EP");
    pipedraft = pipedraft_request < 0 ? auto_pipedraft : pipedraft_request != 0;
    // Layer split (2 GPUs): A [0, 23) + embed, B [23, 45) + lm_head — the
    // extra A block offsets B's head/output_norm residency. IE_G5_SPLIT
    // overrides for balance sweeps (stage B lost 1.38 GiB with the MTP kit
    // gated, so the old balance point may have moved).
    uint32_t split = 23;
    if (const char* v = std::getenv("IE_G5_SPLIT")) {
        const uint32_t s = uint32_t(std::atoi(v));
        if (s >= 1 && s < n_tf) split = s;
    }
    // Per-card expert cache: with the resident set halved, most of each
    // card's VRAM goes to slots. Overridable via IE_G5_ECACHE_MB as ever.
    const bool user_ecache = std::getenv("IE_G5_ECACHE_MB") != nullptr;
    if (n_st == 2 && !user_ecache)
        setenv("IE_G5_ECACHE_MB", "23040", 0);   // 22.5 GiB/card: fits VRAM with the
        // budget-exact allocator (25600 spent-in-full oversubscribed stage A:
        // 5.46 w + 24.99 cache > 29.8) AND the 25600/87-79 geometry reproducibly
        // degenerated text at ~tok 45 on 2026-08-29 (rot-vs-bug open; battery
        // step 3 retests the old value post-reboot).
    // MTP kit loads only when spec decode is requested — otherwise the tail
    // stage skips token_embd + blk.45 (was pushing stage B past the card).
    if (spec > 0 || pipedraft) setenv("IE_G5_MTP", "1", 1);
    if (pipedraft) setenv("IE_G5_MTP_STAGE", "head", 0);   // lever 3: draft on card 0 (env overrides)
    // Lever 4 (glm5t sweep, 2026-09-02): under the pipelined draft stage A's
    // link idles while stage B runs, so stage A fills more of its misses
    // (q*_A 0.5 vs 0.25) — its CPU team shrinks and stops contending with
    // stage B's for DRAM: B's round 80 -> 67 ms, ~+11% at equal acceptance.
    if (pipedraft && !std::getenv("IE_G5_QSTAR") && !std::getenv("IE_G5_QSTAR_A"))
        setenv("IE_G5_QSTAR_A", "0.5", 0);

    DeviceAllocator alloc[2];
    Glm5NextModel   m_st[2];
    g_term_m[0] = &m_st[0];
    g_term_m[1] = &m_st[1];
    g_term_n = n_st;
    std::signal(SIGTERM, on_term);
    std::signal(SIGINT, on_term);
    // IE_G5_EP=1: expert-parallel groundwork — both stage allocators join ONE
    // shared P2P context so either card can DMA any host bank and cross-queue
    // events work (the qwen4exp interstage-push machinery, reused).
    std::vector<sycl::device> p2p_gpus;
    std::unique_ptr<sycl::context> p2p_ctx;
    const bool ep = n_st == 2 && std::getenv("IE_G5_EP") != nullptr &&
                    gpu_p2p_shared_context("B70", p2p_gpus, p2p_ctx);
    if (ep) std::printf("[glm5next] EP: shared P2P context up\n");
    for (uint32_t s = 0; s < n_st; ++s) {
        const uint32_t ord = n_st == 2 ? s : gpu;
        if (ep) {
            if (auto e = alloc[s].init_with(*p2p_ctx, p2p_gpus[s]); !e.empty()) { std::fprintf(stderr, "gpu%u: %s\n", ord, e.c_str()); return 1; }
        } else if (auto e = alloc[s].init("B70", ord); !e.empty()) { std::fprintf(stderr, "gpu%u: %s\n", ord, e.c_str()); return 1; }
        const uint32_t lo = (n_st == 2 && s == 1) ? split : 0;
        const uint32_t hi = (n_st == 2 && s == 0) ? split : n_tf;
        std::printf("[glm5next] loading blocks [%u, %u) on GPU %u ...\n", lo, hi, ord);
        if (auto e = m_st[s].load(alloc[s], g, cfg, 0, lo, hi); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
        // Lever 3 (--pipedraft, MTP kit on the head stage): card 0 also carries
        // blk.45 + lm_head (~2 GiB), so its expert cache takes 21 GiB instead
        // of 22.5 unless the user set IE_G5_ECACHE_MB. init_runtime reads the env.
        if (n_st == 2 && !user_ecache) {
            const char* ms = std::getenv("IE_G5_MTP_STAGE");
            const bool head_kit = pipedraft && ms && (ms[0] == 'h' || ms[0] == 'H' || ms[0] == '0');
            setenv("IE_G5_ECACHE_MB", (head_kit && s == 0) ? "21504" : "23040", 1);
        }
        if (auto e = m_st[s].init_runtime(n_sel, std::max(chunk, 8u)); !e.empty()) { std::fprintf(stderr, "rt: %s\n", e.c_str()); return 1; }
        std::printf("[glm5next] stage %u: %.2f GiB device\n", s, m_st[s].device_bytes() / 1073741824.0);
    }
    {
        // Warm every stage's pinned-bank device mappings AFTER all loads
        // (a later stage's host-USM allocs evict the earlier stage's
        // mappings — the 39-48s first-forward anomaly). Both cards sweep
        // concurrently.
        auto tw0 = std::chrono::steady_clock::now();
        std::future<std::string> fw[2];
        for (uint32_t s = 0; s < n_st; ++s)
            fw[s] = std::async(std::launch::async,
                               [&m_st, s] { return m_st[s].warm_banks(); });
        for (uint32_t s = 0; s < n_st; ++s)
            if (auto e = fw[s].get(); !e.empty()) { std::fprintf(stderr, "warm: %s\n", e.c_str()); return 1; }
        std::printf("[glm5next] bank warmup: %.1fs\n",
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - tw0).count());
    }
    // IE_G5_WARM_CACHE=<profile>: pre-fill each stage's expert cache from a
    // recorded usage profile (IE_G5_EPROFILE_OUT) — the first prompt starts
    // at the profile's hit rate instead of 0%.
    if (const char* wp = std::getenv("IE_G5_WARM_CACHE")) {
        auto tw0 = std::chrono::steady_clock::now();
        std::future<std::string> fw[2];
        for (uint32_t s = 0; s < n_st; ++s)
            fw[s] = std::async(std::launch::async,
                               [&m_st, s, wp] { return m_st[s].warm_cache(wp); });
        for (uint32_t s = 0; s < n_st; ++s)
            if (auto e = fw[s].get(); !e.empty()) { std::fprintf(stderr, "warm_cache: %s\n", e.c_str()); return 1; }
        std::printf("[glm5next] cache warmup: %.1fs\n",
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - tw0).count());
    }
    // IE_G5_EP_DECODE=1 (with IE_G5_EP=1): expert-parallel decode — card 0
    // computes even experts of every pinned MoE layer, card 1 odd; both PCIe
    // links fill concurrently. init_runtime already halved each card's own
    // cache budget; ep_enable allocates the peer-layer half-caches.
    if (std::getenv("IE_G5_EP_DECODE")) {
        if (!ep) { std::fprintf(stderr, "IE_G5_EP_DECODE needs IE_G5_EP=1 (shared P2P context)\n"); return 1; }
        if (auto e = m_st[0].ep_enable(m_st[1], 0); !e.empty()) { std::fprintf(stderr, "ep: %s\n", e.c_str()); return 1; }
        if (auto e = m_st[1].ep_enable(m_st[0], 1); !e.empty()) { std::fprintf(stderr, "ep: %s\n", e.c_str()); return 1; }
    }
    Glm5NextModel& m = m_st[n_st - 1];   // tail stage owns logits
    sycl::queue& q = m.queue();
    std::vector<float> wide_h;
    if (n_st == 2) wide_h.resize(uint64_t(std::max(chunk, 8u)) * cfg.hc_count * cfg.hidden);
    auto fwd = [&](const int32_t* toks, uint32_t T, uint32_t pos,
                   float* all_logits) -> std::string {
        if (n_st == 1) return m_st[0].forward(toks, T, pos, nullptr, all_logits);
        if (auto e = m_st[0].forward_range(toks, T, pos, nullptr, wide_h.data(),
                                           nullptr); !e.empty()) return e;
        return m_st[1].forward_range(toks, T, pos, wide_h.data(), nullptr,
                                     nullptr, all_logits);
    };
    auto reset_all = [&] { for (uint32_t s = 0; s < n_st; ++s) m_st[s].reset_state(); };
    std::printf("[glm5next] loaded (%u stage%s), ctx cap %u\n", n_st,
                n_st > 1 ? "s" : "", n_sel);

    // --prefill uses --ppl only as a TOKEN SOURCE, so it must not fall into the
    // scoring branch (which returns before the generation path is ever reached).
    if (!ppl_file.empty() && !ppbench && !prefill_n) {
        std::ifstream f(ppl_file);
        if (!f) { std::fprintf(stderr, "ppl: cannot open %s\n", ppl_file.c_str()); return 1; }
        std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        std::vector<int32_t> ids = tok.encode(text, false);
        const uint32_t n_sc = std::min<uint32_t>(uint32_t(ids.size()), n_sel);
        std::printf("[ppl] %zu tokens encoded, scoring %u (ctx cap), chunk %u\n",
                    ids.size(), n_sc, chunk);
        float* d_logits = static_cast<float*>(
            sycl::malloc_device(uint64_t((chunk + 7) & ~7u) * cfg.vocab * 4, q));
        double nll = 0.0; uint64_t n_pred = 0;
        reset_all();
        std::vector<float> row(cfg.vocab);
        for (uint32_t p0 = 0; p0 + 1 < n_sc; ) {
            // chunk==1 streams the DECODE kernels; the sequence is primed with
            // one 8-token prefill first (a T=1 forward at pos 0 has no context
            // state to attend/scan — NaN logits, 2026-09-01).
            const uint32_t T = (chunk == 1 && p0 == 0) ? std::min(8u, n_sc - p0)
                                                       : std::min(chunk, n_sc - p0);
            if (auto e = fwd(ids.data() + p0, T, p0, d_logits); !e.empty()) {
                std::fprintf(stderr, "forward: %s\n", e.c_str()); return 1;
            }
            // NLL of position p0+t predicts token ids[p0+t+1].
            for (uint32_t t = 0; t < T && p0 + t + 1 < n_sc; ++t) {
                q.memcpy(row.data(), d_logits + uint64_t(t) * cfg.vocab,
                         uint64_t(cfg.vocab) * 4).wait();
                static int dbg_left = 3;
                if (T == 1 && dbg_left > 0 && !std::isfinite(row[0]) ) {   // PPL DEBUG: first NaN entry
                    --dbg_left;
                    std::vector<sycl::half> h16(cfg.vocab);
                    q.memcpy(h16.data(), m_st[n_st - 1].logits(),
                             uint64_t(cfg.vocab) * 2).wait();
                    uint32_t nf32 = 0, nf16 = 0;
                    for (uint32_t v = 0; v < cfg.vocab; ++v) {
                        if (!std::isfinite(row[v])) ++nf32;
                        if (!std::isfinite(float(h16[v]))) ++nf16;
                    }
                    std::printf("  [dbg p0=%u] f32row nonfinite=%u (row[0]=%g)  f16 logits nonfinite=%u (l16[0]=%g)  tok=%d nll_so_far=%g n=%llu\n",
                                p0, nf32, row[0], nf16, float(h16[0]), ids[p0], nll, (unsigned long long)n_pred);
                }
                float mx = -1e30f;
                for (uint32_t v = 0; v < cfg.vocab; ++v) mx = std::max(mx, row[v]);
                double s = 0.0;
                for (uint32_t v = 0; v < cfg.vocab; ++v) s += std::exp(double(row[v]) - mx);
                const int32_t tgt = ids[p0 + t + 1];
                const double nll_t = std::log(s) + double(mx) - double(row[tgt]);
                nll += nll_t;
                ++n_pred;
                // IE_G5_PPL_TRACE=<file>: one "pos target nll" line per prediction —
                // the cross-run determinism ruler (docs/glm53/CAMPAIGN_2026-09-02.md det1).
                static FILE* trace = [] {
                    const char* tp = std::getenv("IE_G5_PPL_TRACE");
                    return tp ? std::fopen(tp, "w") : nullptr;
                }();
                if (trace) std::fprintf(trace, "%u %d %.17g\n", p0 + t, tgt, nll_t);
            }
            if (chunk > 1 || p0 == 0)
                std::printf("  [chunk @%u] running PPL %.4f over %llu preds\n",
                            p0, std::exp(nll / double(n_pred)), (unsigned long long)n_pred);
            p0 += T;
        }
        {   uint64_t svc = 0, svb = 0;
            for (uint32_t s2 = 0; s2 < n_st; ++s2) { svc += m_st[s2].sv_checks_; svb += m_st[s2].sv_bad_; }
            if (svc) std::printf("[ppl] slot-verify: %llu regions compared, %llu WRONG\n",
                                 (unsigned long long)svc, (unsigned long long)svb);
        }
        std::printf("\n[ppl] FINAL: %.4f (NLL %.5f) over %llu predictions\n",
                    std::exp(nll / double(n_pred)), nll / double(n_pred),
                    (unsigned long long)n_pred);
        sycl::free(d_logits, q);
        if (std::getenv("IE_G5_PPL_TRACE")) std::fflush(nullptr);
        return 0;
    }

    if (ppbench) {
        // Chunked prefill bench over wikitext (needs --ppl <file> as the
        // token source). On two cards, the default overlaps stage A on
        // chunk k+1 with stage B on chunk k (both PCIe links pumping).
        if (ppl_file.empty()) { std::fprintf(stderr, "ppbench needs --ppl <file>\n"); return 1; }
        std::ifstream f(ppl_file);
        std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        std::vector<int32_t> ids = tok.encode(text, false);
        const uint32_t N = std::min(ppbench, (n_sel - 1) / chunk);
        if (N * chunk > ids.size()) { std::fprintf(stderr, "corpus too small\n"); return 1; }
        std::printf("[ppb] %u chunks of %u (%s)\n", N, chunk,
                    pipeline && n_st == 2 && N > 1 ? "PIPELINED" : "serial");
        reset_all();
        KernelProfiler pp_prof;
        struct ClearPrefillProfiler { ~ClearPrefillProfiler() { g_profiler = nullptr; } } pp_clear;
        const bool pp_profiling = std::getenv("IE_QUEUE_PROFILING") != nullptr;
        if (pp_profiling) { pp_prof.begin_step(); g_profiler = &pp_prof; }
        auto t0 = std::chrono::steady_clock::now();
        if (pipeline && n_st == 2 && N > 1) {
            std::vector<float> wide2[2];
            wide2[0].resize(uint64_t(chunk) * cfg.hc_count * cfg.hidden);
            wide2[1].resize(uint64_t(chunk) * cfg.hc_count * cfg.hidden);
            auto stage_a = [&](uint32_t k) {
                return m_st[0].forward_range(ids.data() + uint64_t(k) * chunk, chunk,
                                             k * chunk, nullptr,
                                             wide2[k % 2].data(), nullptr);
            };
            std::future<std::string> fa =
                std::async(std::launch::async, stage_a, 0u);
            for (uint32_t k = 0; k < N; ++k) {
                auto tw0 = std::chrono::steady_clock::now();
                if (auto e = fa.get(); !e.empty()) { std::fprintf(stderr, "A: %s\n", e.c_str()); return 1; }
                auto tw1 = std::chrono::steady_clock::now();
                if (k + 1 < N) fa = std::async(std::launch::async, stage_a, k + 1);
                if (auto e = m_st[1].forward_range(ids.data() + uint64_t(k) * chunk, chunk,
                                                   k * chunk, wide2[k % 2].data(),
                                                   nullptr, nullptr); !e.empty()) {
                    std::fprintf(stderr, "B: %s\n", e.c_str()); return 1;
                }
                auto tw2 = std::chrono::steady_clock::now();
                std::printf("[ppb] chunk %u done at %.2fs (A-wait %.2fs, B %.2fs)\n", k,
                            std::chrono::duration<double>(tw2 - t0).count(),
                            std::chrono::duration<double>(tw1 - tw0).count(),
                            std::chrono::duration<double>(tw2 - tw1).count());
            }
        } else {
            for (uint32_t k = 0; k < N; ++k) {
                if (auto e = fwd(ids.data() + uint64_t(k) * chunk, chunk, k * chunk,
                                 nullptr); !e.empty()) {
                    std::fprintf(stderr, "fwd: %s\n", e.c_str()); return 1;
                }
                std::printf("[ppb] chunk %u done at %.2fs\n", k,
                            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            }
        }
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (pp_profiling) {
            g_profiler = nullptr;  // both stages have finished submitting
            auto stats = pp_prof.harvest();
            std::sort(stats.begin(), stats.end(),
                      [](const auto& a, const auto& b) { return a.total_ns > b.total_ns; });
            uint64_t total = 0;
            for (const auto& st : stats) total += st.total_ns;
            std::printf("[pprof] instrumented GPU kernels %.1f ms over %u input tokens\n",
                        double(total) * 1e-6, N * chunk);
            for (size_t d = 0; d < pp_prof.dev_spans.size(); ++d) {
                const auto& ds = pp_prof.dev_spans[d];
                std::printf("[pprof] dev %zu: busy %.1f ms, window %.1f ms\n", d,
                            double(ds.busy_ns) * 1e-6, double(ds.span_ns) * 1e-6);
            }
            for (size_t i = 0; i < stats.size(); ++i)
                std::printf("[pprof] %-24s %9.1f ms %7u calls\n",
                            stats[i].name.c_str(), stats[i].total_ms(), stats[i].calls);
        }
        std::printf("[ppb] TOTAL: %u tok / %.2f s = %.1f tok/s prefill\n",
                    N * chunk, s, double(N) * chunk / s);
        for (uint32_t s2 = 0; s2 < n_st; ++s2)
            std::printf("[ppb] stage %u host: ring_wait %.2fs  pool_copy %.2fs  "
                        "fill_submit %.2fs  compute_submit %.2fs\n", s2,
                        m_st[s2].t_ring_wait, m_st[s2].t_pool_copy,
                        m_st[s2].t_fill_submit, m_st[s2].t_compute_submit);
        for (uint32_t s2 = 0; s2 < n_st; ++s2)
            if (m_st[s2].t_pps_get > 0 || m_st[s2].t_pps_wait > 0)
                std::printf("[ppb] stage %u stream: get %.2fs  landed-wait %.2fs\n",
                            s2, m_st[s2].t_pps_get, m_st[s2].t_pps_wait);
        uint64_t hits = 0, misses = 0;
        for (uint32_t s2 = 0; s2 < n_st; ++s2) { hits += m_st[s2].ecache_hits; misses += m_st[s2].ecache_misses; }
        std::printf("[ppb] ecache: %llu hits / %llu misses (%.1f%% hit)\n",
                    (unsigned long long)hits, (unsigned long long)misses,
                    100.0 * double(hits) / double(std::max<uint64_t>(1, hits + misses)));
        return 0;
    }

    std::printf("[glm5next] execution: prefill %s, decode %s\n",
                pipeline && n_st == 2 ? "pipeline" : "serial",
                spec ? "spec" : pipedraft ? "pipedraft" : "plain");
    // Greedy continuation smoke.
    // --prefill N (with --ppl as the token source): prefill N real tokens in
    // `chunk`-sized pieces, timing each piece, THEN decode --ngen tokens. This
    // is the only way to see cost at DEPTH — every other mode measures near
    // position 0, and the DSA indexer scores n_pools per token per MLA layer,
    // so both prefill and decode cost move with position.
    // Rows in mean_rows() after the prefill. `mean_` holds only the LAST
    // forward's T rows, so with a CHUNKED prefill (--prefill) the last hidden
    // row is at (last chunk's T - 1), NOT at ids.size()-1. Indexing by the
    // prompt length reads past the buffer and hands the MTP drafter garbage —
    // measured 2026-09-04 n3: acceptance 85.3% (5-token prompt, in bounds),
    // 68.4% (2048 prefill), 0.0% (8192 prefill), i.e. --pipedraft silently
    // stopped drafting on exactly the long prompts it is worth the most on.
    uint32_t last_fwd_rows = 0;
    std::vector<int32_t> ids;
    if (prefill_n) {
        if (ppl_file.empty()) { std::fprintf(stderr, "--prefill needs --ppl <file> as the token source\n"); return 1; }
        std::ifstream pf(ppl_file);
        std::string ptext((std::istreambuf_iterator<char>(pf)), std::istreambuf_iterator<char>());
        std::vector<int32_t> all = tok.encode(ptext, false);
        const uint32_t want = std::min<uint32_t>(prefill_n, std::min<uint32_t>(n_sel - 1, uint32_t(all.size())));
        ids.assign(all.begin(), all.begin() + want);
        std::printf("[gen] prefill %u tokens from %s in chunks of %u\n[gen] ",
                    want, ppl_file.c_str(), chunk);
    } else {
        ids = tok.encode(prompt, false);
        std::printf("[gen] prompt: \"%s\" (%zu tokens)\n[gen] ", prompt.c_str(), ids.size());
    }
    if (ids.empty() || ids.size() >= n_sel) {
        std::fprintf(stderr, "prompt must contain between 1 and %u tokens (use --ctx for a larger prompt)\n", n_sel - 1);
        return 2;
    }
    reset_all();
    auto t0 = std::chrono::steady_clock::now();
    if (prefill_n || ids.size() > chunk) {
        auto pt = t0;
        auto pf_line = [&](uint32_t off, uint32_t T) {
            const auto now = std::chrono::steady_clock::now();
            std::printf("[pf] pos %7u..%-7u  %6.2f s  %7.1f tok/s\n", off, off + T,
                        std::chrono::duration<double>(now - pt).count(),
                        double(T) / std::chrono::duration<double>(now - pt).count());
            std::fflush(stdout);
            pt = now;
        };
        if (pipeline && n_st == 2 && ids.size() > chunk) {
            // Two-stage PIPELINED prefill — the --ppbench --pipeline path, wired
            // to real prompts (2026-09-04; it had never served one). Stage A
            // runs chunk k+1 on card 0 while stage B runs chunk k on card 1.
            // Double-buffered wide handoff: A(k+1) writes wide2[(k+1)%2] while
            // B(k) reads wide2[k%2], and A(k+2) is not launched until B(k) has
            // returned. Each stage sees its chunks in order, so per-stage state
            // is exactly the serial sequence; the ragged last chunk carries its
            // own T, and the tail stage's last forward sets last_fwd_rows (the
            // MTP seed row) as the serial loop does. Measured 44.3 -> 58.1 tok/s
            // at 2 x 1024 (n18; the 2-chunk ceiling is 1.33x).
            const uint32_t nck = uint32_t((ids.size() + chunk - 1) / chunk);
            std::vector<float> wide2[2];
            wide2[0].resize(uint64_t(chunk) * cfg.hc_count * cfg.hidden);
            wide2[1].resize(uint64_t(chunk) * cfg.hc_count * cfg.hidden);
            auto ck_T = [&](uint32_t k) {
                return std::min<uint32_t>(chunk, uint32_t(ids.size()) - k * chunk);
            };
            auto stage_a = [&](uint32_t k) {
                return m_st[0].forward_range(ids.data() + uint64_t(k) * chunk, ck_T(k),
                                             k * chunk, nullptr, wide2[k % 2].data(), nullptr);
            };
            std::future<std::string> fa = std::async(std::launch::async, stage_a, 0u);
            for (uint32_t k = 0; k < nck; ++k) {
                const uint32_t T = ck_T(k);
                if (auto e = fa.get(); !e.empty()) {
                    std::fprintf(stderr, "prefill A@%u: %s\n", k * chunk, e.c_str()); return 1;
                }
                if (k + 1 < nck) fa = std::async(std::launch::async, stage_a, k + 1);
                if (auto e = m_st[1].forward_range(ids.data() + uint64_t(k) * chunk, T, k * chunk,
                                                   wide2[k % 2].data(), nullptr, nullptr);
                    !e.empty()) {
                    std::fprintf(stderr, "prefill B@%u: %s\n", k * chunk, e.c_str()); return 1;
                }
                last_fwd_rows = T;
                pf_line(k * chunk, T);
            }
        } else
        for (uint32_t off = 0; off < ids.size(); off += chunk) {
            const uint32_t T = std::min<uint32_t>(chunk, uint32_t(ids.size()) - off);
            if (auto e = fwd(ids.data() + off, T, off, nullptr); !e.empty()) {
                std::fprintf(stderr, "prefill@%u: %s\n", off, e.c_str()); return 1;
            }
            last_fwd_rows = T;
            pf_line(off, T);
        }
    } else {
        if (auto e = fwd(ids.data(), uint32_t(ids.size()), 0, nullptr); !e.empty()) {
            std::fprintf(stderr, "prefill: %s\n", e.c_str()); return 1;
        }
        last_fwd_rows = uint32_t(ids.size());
    }
    auto t1 = std::chrono::steady_clock::now();
    // Zero the host wait counters at the prefill -> decode boundary so the
    // [host] partition printed at the end is DECODE-ONLY. They accumulate from
    // load; every depth partition before 2026-09-04 carried the prefill's
    // waits ("pool 2082 ms/token" at 24K was the prefill's expert staging).
    for (uint32_t s2 = 0; s2 < n_st; ++s2) {
        m_st[s2].t_route_wait = m_st[s2].t_final_wait = m_st[s2].t_wide_wait = 0;
        m_st[s2].t_cpu_join = m_st[s2].t_cpu_compute = 0;
        m_st[s2].t_ring_wait = m_st[s2].t_pool_copy = 0;
        m_st[s2].t_fill_submit = m_st[s2].t_compute_submit = 0;
    }
    uint32_t pos = uint32_t(ids.size());
    int32_t t = argmax_h(q, m.logits(), cfg.vocab);
    uint32_t emitted = 0;
    if (spec > 0 && m.mtp_loaded()) {
        // MTP speculative decode: draft `spec` tokens with the NextN head,
        // verify the whole window in one batched forward, accept the matching
        // prefix. KDA state cannot rewind, so each round snapshots and a
        // partial accept restores + replays the accepted rows (their replay
        // also yields the next committed token).
        const uint32_t H = cfg.hidden, V = cfg.vocab, W = spec + 1;
        float* d_all = static_cast<float*>(
            sycl::malloc_device(uint64_t((W + 7) & ~7u) * V * 4, q));
        std::vector<float> hl(uint64_t(W) * V);
        // mean_ holds only the LAST forward's rows (capacity max_chunk), so with
        // a chunked --prefill this must be that chunk's T, not the prompt
        // length — indexing by ids.size() is an out-of-bounds device read
        // (Codex review 2026-09-04; same bug as the --pipedraft seed).
        uint32_t lastT = last_fwd_rows;
        uint64_t rounds = 0, drafted = 0, accepted = 0;
        std::vector<uint64_t> try_at(spec, 0), acc_at(spec, 0);
        for (uint32_t s = 0; s < n_st; ++s) m_st[s].set_spec_verify(true);
        m.mtp_reset(pos);
        std::vector<int32_t> win;
        double t_snap = 0, t_draft = 0, t_verify = 0, t_commit = 0, t_argmax = 0;
        auto lap = [](std::chrono::steady_clock::time_point& tp) {
            auto now = std::chrono::steady_clock::now();
            double d = std::chrono::duration<double>(now - tp).count();
            tp = now;
            return d;
        };
        while (emitted < ngen) {
            auto tp = std::chrono::steady_clock::now();
            for (uint32_t s = 0; s < n_st; ++s) m_st[s].snapshot_state();
            t_snap += lap(tp);
            win.assign(1, t);
            const float* hid = m.mean_rows() + uint64_t(lastT - 1) * H;
            for (uint32_t d = 0; d < spec; ++d) {
                if (auto e = m.mtp_step(hid, win.back(), pos + d); !e.empty()) {
                    std::fprintf(stderr, "\nmtp: %s\n", e.c_str()); return 1;
                }
                win.push_back(argmax_h(q, m.logits(), V));
                hid = m.mtp_hidden();
            }
            t_draft += lap(tp);
            ++rounds; drafted += spec;
            if (auto e = fwd(win.data(), W, pos, d_all); !e.empty()) {
                std::fprintf(stderr, "\nverify: %s\n", e.c_str()); return 1;
            }
            q.memcpy(hl.data(), d_all, uint64_t(W) * V * 4).wait();
            t_verify += lap(tp);
            uint32_t n_rows = W;
            int32_t next_t = 0;
            for (uint32_t i = 0; i < W; ++i) {
                std::printf("%s", tok.decode(std::span<const int32_t>(&win[i], 1)).c_str());
                ++emitted;
                const float* r = hl.data() + uint64_t(i) * V;
                int32_t am = 0; float mx = -1e30f;
                for (uint32_t v = 0; v < V; ++v) if (r[v] > mx) { mx = r[v]; am = int32_t(v); }
                next_t = am;
                if (i + 1 < W) {
                    ++try_at[i];
                    if (am == win[i + 1] && emitted < ngen) { ++accepted; ++acc_at[i]; continue; }
                    n_rows = i + 1;
                }
                break;
            }
            std::fflush(stdout);
            auto tp2 = std::chrono::steady_clock::now();
            t_argmax += std::chrono::duration<double>(
                tp2 - tp).count();   // host argmax + decode of the window
            if (n_rows == W) {           // full accept: verify state is live
                pos += W; lastT = W; t = next_t;
            } else {                     // partial: commit the accepted prefix
                for (uint32_t s = 0; s < n_st; ++s)
                    if (auto e = m_st[s].commit_verify(n_rows); !e.empty()) {
                        std::fprintf(stderr, "\ncommit: %s\n", e.c_str()); return 1;
                    }
                pos += n_rows; lastT = n_rows;
                t = next_t;              // the verify row already scored it
            }
            t_commit += lap(tp2);
        }
        std::printf("\n[spec] round cost: snap %.2fs draft %.2fs verify %.2fs argmax %.2fs commit %.2fs (%llu rounds)\n",
                    t_snap, t_draft, t_verify, t_argmax, t_commit,
                    (unsigned long long)rounds);
        sycl::free(d_all, q);
        std::printf("\n[spec] K=%u: %llu rounds, %llu drafted, %llu accepted (%.1f%%), %.2f tok/round\n[spec] acc@d:",
                    spec, (unsigned long long)rounds, (unsigned long long)drafted,
                    (unsigned long long)accepted,
                    100.0 * double(accepted) / double(std::max<uint64_t>(1, drafted)),
                    double(emitted) / double(std::max<uint64_t>(1, rounds)));
        for (uint32_t d = 0; d < spec; ++d)
            std::printf(" %llu/%llu", (unsigned long long)acc_at[d], (unsigned long long)try_at[d]);
    } else if (pipedraft && n_st == 2 && (m.mtp_loaded() || m_st[0].mtp_loaded())) {
        // PIPELINED DRAFT decode (docs/glm53/CAMPAIGN_2026-09-02.md G5'): the two
        // stages run SERIALLY per token, so while stage B (card 1) computes token
        // t, card 0 and its PCIe link sit idle. Run stage A for the MTP-drafted
        // NEXT token there, concurrently, behind a KDA snapshot. If stage B's
        // argmax confirms the draft, stage A's work for the next token is
        // already done (bit-identical to what it would compute: same token,
        // same position, same state); if not, restore and redo it. The emitted
        // token always comes from stage B's argmax over the real token stream,
        // so the text is the greedy text by construction. The MTP head stays
        // aligned with the committed stream (stepped once per real token).
        // Lever 3: the MTP kit lives on card 0 (IE_G5_MTP_STAGE=head, the
        // default for this mode) so the draft step runs on the card that idles
        // and overlaps stage B instead of sitting on card 1's critical path;
        // the hidden row it needs (16 KB) hops card 1 -> host -> card 0.
        // Declare the CONCURRENT stages before the first decode token creates
        // the q* CPU workers (the model logs the placement it chooses for this
        // mode and the measured reason; see set_cpu_partition).
        for (uint32_t s = 0; s < n_st; ++s) m_st[s].set_cpu_partition(true);
        Glm5NextModel& md = m_st[0].mtp_loaded() ? m_st[0] : m;   // the draft host
        sycl::queue& qd = md.queue();
        const bool draft_on_a = &md == &m_st[0];   // reported in the summary line (stdout carries the text)
        const uint32_t H = cfg.hidden, V = cfg.vocab;
        std::vector<float> wide2[2];
        wide2[0].assign(uint64_t(cfg.hc_count) * H, 0.f);
        wide2[1].assign(uint64_t(cfg.hc_count) * H, 0.f);
        // hidden-row hop buffers: two host copies (ping-pong by round) + one
        // device row on the draft host's card
        std::vector<float> hid_h[2] = {std::vector<float>(H), std::vector<float>(H)};
        float* hid_d = draft_on_a ? static_cast<float*>(sycl::malloc_device(uint64_t(H) * 4, qd)) : nullptr;
        uint64_t rounds = 0, accepted = 0;
        double t_wait_a = 0, t_b = 0, t_mtp = 0, t_redo = 0;
        auto lap = [](std::chrono::steady_clock::time_point& tp) {
            auto now = std::chrono::steady_clock::now();
            double d = std::chrono::duration<double>(now - tp).count();
            tp = now;
            return d;
        };
        // Draft for position p+1 from the hidden row of p-1 (in hid_h[par]) and
        // the real token at p (the MTP frame: entry p fuses hidden(p-1) with
        // tok(p)). Runs on the draft host's queue; safe from a second thread
        // because it touches only md and its own buffers.
        md.mtp_reset(pos);
        // tail-hosted kit: the hidden row is already on card 1 — the prefill's
        // LAST mean row first (T = ids.size() rows), then row 0 of each T = 1
        // stage-B forward. (A first version passed row 0 of the prefill here;
        // the wrong first draft shifted the whole q* trajectory — 19:28 tail1.)
        const float* tail_hid = m.mean_rows() + uint64_t(last_fwd_rows - 1) * H;
        auto draft = [&](uint32_t par, int32_t tok_id, uint32_t p) -> int32_t {
            const float* hid;
            if (draft_on_a) {
                qd.memcpy(hid_d, hid_h[par].data(), uint64_t(H) * 4).wait();
                hid = hid_d;
            } else {
                // Only the serial tail-hosted path reads tail_hid: stage B
                // updates it while the head-hosted draft runs concurrently.
                hid = tail_hid;
            }
            if (auto e = md.mtp_step(hid, tok_id, p); !e.empty()) {
                std::fprintf(stderr, "\nmtp: %s\n", e.c_str()); return -1;
            }
            return argmax_h(qd, md.logits(), V);
        };
        // seed: hidden(pos-1) = the prefill's last mean row (card 1) -> host
        uint32_t par = 0;
        q.memcpy(hid_h[par].data(), m.mean_rows() + uint64_t(last_fwd_rows - 1) * H, uint64_t(H) * 4).wait();
        uint32_t cur = 0;
        if (auto e = m_st[0].forward_range(&t, 1, pos, nullptr, wide2[cur].data(), nullptr);
            !e.empty()) { std::fprintf(stderr, "\nA: %s\n", e.c_str()); return 1; }
        while (emitted < ngen) {
            std::printf("%s", tok.decode(std::span<const int32_t>(&t, 1)).c_str());
            std::fflush(stdout);
            ++emitted;
            if (emitted >= ngen) break;
            auto tp = std::chrono::steady_clock::now();
            // card 0: draft the token for pos+1, then run speculative stage A on it
            const int32_t t_now = t; const uint32_t par_now = par;
            int32_t d_now = -1;
            double t_mtp_round = 0;
            if (!draft_on_a) {
                // tail-hosted kit: the draft shares card 1 and the tail model
                // object with stage B, so it runs here, serially, before B.
                auto tm = std::chrono::steady_clock::now();
                d_now = draft(par_now, t_now, pos);
                t_mtp_round = std::chrono::duration<double>(std::chrono::steady_clock::now() - tm).count();
                if (d_now < 0) return 1;
            }
            std::future<std::string> fa = std::async(std::launch::async, [&, t_now, par_now] {
                if (draft_on_a) {
                    // head-hosted kit: draft on card 0, overlapping stage B
                    auto tm = std::chrono::steady_clock::now();
                    d_now = draft(par_now, t_now, pos);
                    t_mtp_round = std::chrono::duration<double>(std::chrono::steady_clock::now() - tm).count();
                    if (d_now < 0) return std::string("draft failed");
                }
                if (auto e = m_st[0].snapshot_state(); !e.empty()) return e;
                return m_st[0].forward_range(&d_now, 1, pos + 1, nullptr,
                                             wide2[cur ^ 1].data(), nullptr);
            });
            // card 1: stage B for the real token at pos
            if (auto e = m_st[1].forward_range(&t, 1, pos, wide2[cur].data(), nullptr, nullptr);
                !e.empty()) { std::fprintf(stderr, "\nB: %s\n", e.c_str()); return 1; }
            const int32_t t_next = argmax_h(q, m.logits(), V);
            // hidden(pos) -> host, into the OTHER parity slot (the draft thread may
            // still be reading this round's slot)
            q.memcpy(hid_h[par ^ 1].data(), m.mean_rows(), uint64_t(H) * 4).wait();
            tail_hid = m.mean_rows();   // T = 1: row 0 is hidden(pos)
            t_b += lap(tp);
            if (auto e = fa.get(); !e.empty()) { std::fprintf(stderr, "\nA(spec): %s\n", e.c_str()); return 1; }
            t_mtp += t_mtp_round;
            t_wait_a += lap(tp);
            ++rounds;
            if (d_now == t_next) {
                ++accepted;
            } else {
                if (auto e = m_st[0].restore_state(); !e.empty()) { std::fprintf(stderr, "\nA restore: %s\n", e.c_str()); return 1; }
                if (auto e = m_st[0].forward_range(&t_next, 1, pos + 1, nullptr, wide2[cur ^ 1].data(), nullptr);
                    !e.empty()) { std::fprintf(stderr, "\nA(redo): %s\n", e.c_str()); return 1; }
                t_redo += lap(tp);
            }
            t = t_next; ++pos; cur ^= 1; par ^= 1;
        }
        if (hid_d) sycl::free(hid_d, qd);
        std::printf("\n[pipedraft] %llu rounds, %llu drafts accepted (%.1f%%); B %.2fs  mtp %.2fs (on card %s, overlapped)  A-wait %.2fs  A-redo %.2fs\n",
                    (unsigned long long)rounds, (unsigned long long)accepted,
                    100.0 * double(accepted) / double(std::max<uint64_t>(1, rounds)),
                    t_b, t_mtp, draft_on_a ? "0" : "1", t_wait_a, t_redo);
    } else {
        if (pipedraft) std::fprintf(stderr, "[pipedraft] needs --gpus 2 and the MTP kit (nextn.*) on the tail stage — plain greedy\n");
        // IE_QUEUE_PROFILING=1: per-kernel GPU-time buckets over the decode
        // loop (queue built with profiling; ie::ps feeds g_profiler).
        KernelProfiler prof;
        const bool profiling = std::getenv("IE_QUEUE_PROFILING") != nullptr;
        if (profiling) { g_profiler = &prof; prof.begin_step(); }
        // Host-time partition of the decode loop: fwd() (stage A + stage B host
        // thread time, of which the model's own counters say how much was
        // WAITING on the GPU) + argmax (a 150K-row D2H + CPU scan per token)
        // + decode/print. Sums to the wall clock exactly.
        double t_fwd = 0, t_argmax = 0, t_print = 0;
        for (uint32_t s = 0; s < ngen; ++s) {
            if (profiling) prof.mark_token();
            const auto p0 = std::chrono::steady_clock::now();
            std::printf("%s", tok.decode(std::span<const int32_t>(&t, 1)).c_str());
            std::fflush(stdout);
            const auto p1 = std::chrono::steady_clock::now();
            t_print += std::chrono::duration<double>(p1 - p0).count();
            if (auto e = fwd(&t, 1, pos, nullptr); !e.empty()) {
                std::fprintf(stderr, "\nstep: %s\n", e.c_str()); return 1;
            }
            t_fwd += std::chrono::duration<double>(std::chrono::steady_clock::now() - p1).count();
            ++pos;
            // IE_G5_LOGIT_DUMP=<file>: raw fp16 logits of the FIRST T=1
            // decode step (the EP-vs-baseline numerics gate).
            if (s == 0) {
                if (const char* lf = std::getenv("IE_G5_LOGIT_DUMP")) {
                    std::vector<sycl::half> hl(cfg.vocab);
                    q.memcpy(hl.data(), m.logits(), cfg.vocab * 2).wait();
                    if (FILE* f = std::fopen(lf, "wb")) {
                        std::fwrite(hl.data(), 2, hl.size(), f);
                        std::fclose(f);
                        std::printf("[logits -> %s]", lf);
                    }
                }
            }
            const auto a0 = std::chrono::steady_clock::now();
            t = argmax_h(q, m.logits(), cfg.vocab);
            t_argmax += std::chrono::duration<double>(std::chrono::steady_clock::now() - a0).count();
            ++emitted;
        }
        if (emitted) {
            const double n = double(emitted);
            std::printf("\n[host] per token: fwd %.2f ms | argmax %.2f ms | print %.2f ms\n",
                        1e3 * t_fwd / n, 1e3 * t_argmax / n, 1e3 * t_print / n);
            for (uint32_t s2 = 0; s2 < n_st; ++s2)
                std::printf("[host] stage %u waits/token: route %.2f ms  final %.2f ms  wide %.2f ms  "
                            "cpu-join %.2f ms  ring %.2f ms  pool %.2f ms\n", s2,
                            1e3 * m_st[s2].t_route_wait / n, 1e3 * m_st[s2].t_final_wait / n,
                            1e3 * m_st[s2].t_wide_wait / n, 1e3 * m_st[s2].t_cpu_join / n,
                            1e3 * m_st[s2].t_ring_wait / n, 1e3 * m_st[s2].t_pool_copy / n);
            // Host WORK (not waits): the MoE submit path per token — ensure()
            // (Belady victim scan) + compute_e() per expert, and the fill
            // submits. If this is most of the token, the host is the critical
            // path and launch COUNT / submit COST are the levers (2026-09-04).
            for (uint32_t s2 = 0; s2 < n_st; ++s2)
                std::printf("[host] stage %u submit/token: moe-compute %.2f ms  fill %.2f ms\n", s2,
                            1e3 * m_st[s2].t_compute_submit / n, 1e3 * m_st[s2].t_fill_submit / n);
        }
        if (profiling) {
            auto stats = prof.harvest();
            g_profiler = nullptr;
            std::sort(stats.begin(), stats.end(),
                      [](auto& a, auto& b) { return a.total_ns > b.total_ns; });
            uint64_t all = 0;
            for (auto& s : stats) all += s.total_ns;
            std::printf("\n[prof] decode GPU time %.1f ms over %u tokens (%.2f ms/tok):\n",
                        double(all) * 1e-6, ngen, double(all) * 1e-6 / ngen);
            // Per device: busy = its kernels' summed time; window = first kernel
            // start .. last kernel end WITHIN each token, summed. busy/window <<
            // 1 means the GPU idles inside its own window: the host feeds it
            // slower than it executes (model B); ~1 means the GPU runs
            // back-to-back and the rest of the token is host-only time.
            for (size_t d = 0; d < prof.dev_spans.size(); ++d) {
                const auto& ds = prof.dev_spans[d];
                if (!ds.tokens) continue;
                std::printf("[prof] dev %zu: busy %.2f ms/tok inside a %.2f ms/tok window (%.0f%% busy, %u tok)\n",
                            d, double(ds.busy_ns) * 1e-6 / ds.tokens,
                            double(ds.span_ns) * 1e-6 / ds.tokens,
                            100.0 * double(ds.busy_ns) / double(std::max<uint64_t>(1, ds.span_ns)), ds.tokens);
            }
            for (size_t i = 0; i < stats.size(); ++i)
                std::printf("[prof]  %-22s %8.1f ms  %6u calls  (%4.1f%%)\n",
                            stats[i].name.c_str(), stats[i].total_ms(), stats[i].calls,
                            100.0 * double(stats[i].total_ns) / double(all));
        }
    }
    auto t2 = std::chrono::steady_clock::now();
    const double pp_s = std::chrono::duration<double>(t1 - t0).count();
    const double tg_s = std::chrono::duration<double>(t2 - t1).count();
    std::printf("\n[gen] done: prefill %zu tok / %.2f s = %.1f tok/s | decode %u tok / %.2f s = %.2f tok/s\n",
                ids.size(), pp_s, double(ids.size()) / pp_s, emitted, tg_s, double(emitted) / tg_s);
    uint64_t hits = 0, misses = 0;
    for (uint32_t s = 0; s < n_st; ++s) { hits += m_st[s].ecache_hits; misses += m_st[s].ecache_misses; }
    for (uint32_t s2 = 0; s2 < n_st; ++s2)
        std::printf("[gen] stage %u host: ring_wait %.2fs  stage %.2fs\n", s2,
                    m_st[s2].t_ring_wait, m_st[s2].t_pool_copy);
    std::printf("[gen] ecache: %llu hits / %llu misses (%.1f%% hit)\n",
                (unsigned long long)hits, (unsigned long long)misses,
                100.0 * double(hits) / double(std::max<uint64_t>(1, hits + misses)));
    {   // IE_G5_SLOT_VERIFY positive control: "0 bad" only means something
        // when the checker actually ran.
        uint64_t svc = 0, svb = 0;
        for (uint32_t s2 = 0; s2 < n_st; ++s2) { svc += m_st[s2].sv_checks_; svb += m_st[s2].sv_bad_; }
        if (svc) std::printf("[gen] slot-verify: %llu regions compared, %llu WRONG\n",
                             (unsigned long long)svc, (unsigned long long)svb);
    }
    for (uint32_t s2 = 0; s2 < n_st; ++s2)
        if (m_st[s2].n_cpu_experts)
            std::printf("[gen] stage %u q*: %llu CPU experts, worker %.2fs (%.3f ms/expert), main-thread join wait %.2fs\n",
                        s2, (unsigned long long)m_st[s2].n_cpu_experts, m_st[s2].t_cpu_compute,
                        1e3 * m_st[s2].t_cpu_compute / double(m_st[s2].n_cpu_experts),
                        m_st[s2].t_cpu_join);
    if (std::getenv("IE_G5_SHADOW_COUNT")) {
        uint64_t sh = 0, st = 0;
        for (uint32_t s = 0; s < n_st; ++s) { sh += m_st[s].shadow_hits_; st += m_st[s].shadow_total_; }
        std::printf("[gen] shadow-router: %llu/%llu layer-ahead predictions correct (%.1f%%)\n",
                    (unsigned long long)sh, (unsigned long long)st,
                    100.0 * double(sh) / double(std::max<uint64_t>(1, st)));
        uint64_t s2 = 0, t2 = 0;
        for (uint32_t s = 0; s < n_st; ++s) { s2 += m_st[s].shadow2_hits_; t2 += m_st[s].shadow2_total_; }
        std::printf("[gen] shadow-router depth-2: %llu/%llu correct (%.1f%%)\n",
                    (unsigned long long)s2, (unsigned long long)t2,
                    100.0 * double(s2) / double(std::max<uint64_t>(1, t2)));
        std::printf("[gen] shadow per-layer %%:");
        for (uint32_t s = 0; s < n_st; ++s)
            for (uint32_t L = 0; L < cfg.n_layers; ++L) {
                const auto& lh = m_st[s].shadow_lhits_;
                const auto& lt = m_st[s].shadow_ltotal_;
                if (L < lt.size() && lt[L])
                    std::printf(" L%u:%.0f", L, 100.0 * double(lh[L]) / double(lt[L]));
            }
        std::printf("\n");
    }
    if (std::getenv("IE_G5_PREFETCH")) {
        uint64_t pi = 0, pu = 0;
        for (uint32_t s = 0; s < n_st; ++s) { pi += m_st[s].pf_issued_; pu += m_st[s].pf_used_; }
        std::printf("[gen] prefetch: %llu issued, %llu consumed (%.1f%%)\n",
                    (unsigned long long)pi, (unsigned long long)pu,
                    100.0 * double(pu) / double(std::max<uint64_t>(1, pi)));
    }
    // Routing-skew report: what fraction of each layer's picks land in its
    // top-N experts (N = a typical slot count). The decision number for
    // static hot residency.
    {
        const uint32_t E = cfg.n_experts, N = 55;
        double cov_sum = 0, cov_min = 1e9, cov_max = -1;
        uint32_t n_lay = 0;
        for (uint32_t s = 0; s < n_st; ++s) {
            const auto& c = m_st[s].expert_counts;
            if (c.empty()) continue;
            for (uint32_t L = 0; L < cfg.n_layers; ++L) {
                std::vector<uint64_t> row(c.begin() + uint64_t(L) * E,
                                          c.begin() + uint64_t(L + 1) * E);
                const uint64_t tot = std::accumulate(row.begin(), row.end(), uint64_t(0));
                if (!tot) continue;
                std::partial_sort(row.begin(), row.begin() + N, row.end(),
                                  std::greater<uint64_t>());
                const uint64_t top = std::accumulate(row.begin(), row.begin() + N, uint64_t(0));
                const double cov = double(top) / double(tot);
                cov_sum += cov; cov_min = std::min(cov_min, cov); cov_max = std::max(cov_max, cov);
                ++n_lay;
            }
        }
        if (n_lay)
            std::printf("[gen] top-%u coverage: mean %.1f%%  min %.1f%%  max %.1f%%  (%u MoE layers)\n",
                        N, 100 * cov_sum / n_lay, 100 * cov_min, 100 * cov_max, n_lay);
        if (const char* out = std::getenv("IE_G5_MISSPROF_OUT")) {
            std::vector<uint64_t> merged(uint64_t(cfg.n_layers) * E, 0);
            for (uint32_t s = 0; s < n_st; ++s)
                for (uint64_t i = 0; i < m_st[s].miss_counts.size(); ++i)
                    merged[i] += m_st[s].miss_counts[i];
            if (FILE* f = std::fopen(out, "wb")) {
                std::fwrite(merged.data(), 8, merged.size(), f);
                std::fclose(f);
                std::printf("[gen] miss profile written: %s\n", out);
            }
        }
        if (const char* out = std::getenv("IE_G5_EPROFILE_OUT")) {
            std::vector<uint64_t> merged(uint64_t(cfg.n_layers) * E, 0);
            for (uint32_t s = 0; s < n_st; ++s)
                for (uint64_t i = 0; i < m_st[s].expert_counts.size(); ++i)
                    merged[i] += m_st[s].expert_counts[i];
            if (FILE* f = std::fopen(out, "wb")) {
                std::fwrite(merged.data(), 8, merged.size(), f);
                std::fclose(f);
                std::printf("[gen] expert profile written: %s\n", out);
            }
        }
    }
    return 0;
}
