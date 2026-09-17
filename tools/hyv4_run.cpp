// tools/hyv4_run.cpp — GLM-5.3-Flash end-to-end forward smoke + PPL (P2).
//
// Default: greedy continuation of a prompt (correctness smoke — coherent text
// is the cheapest whole-model probe). --ppl <file> runs the chunked NLL
// scorer over a text file: PPL in the single digits on normal text is the
// "graph is right" band; a mis-wired sublayer lands in the hundreds+.
//
// v0 limits (by design, see hyv4.hpp): ctx <= 2051 (dense MLA == sparse
// selection below n_select), single GPU.
//
// usage: ie-hyv4-run <model-00001-of-00006.gguf> [gpu] [--prompt "..."]
//        [--ngen N] [--ppl file] [--chunk T]
#include "ie/gguf.hpp"
#include "ie/hyv4.hpp"
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
Hyv4Model* g_term_m[2] = {};
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
                             "[--ppl file] [--chunk T]\n", argv[0]);
        return 2;
    }
    uint32_t gpu = 0, ngen = 32, chunk = 512, n_gpus = 1, spec = 0, ppbench = 0;
    bool pipeline = false;
    std::string prompt = "The capital of France is";
    std::string ppl_file, ids_file, logits_out;
    bool print_tokens = false;
    for (int i = 2; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--prompt") && i + 1 < argc) prompt = argv[++i];
        else if (!std::strcmp(argv[i], "--ngen") && i + 1 < argc) ngen = uint32_t(std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "--ppl") && i + 1 < argc) ppl_file = argv[++i];
        else if (!std::strcmp(argv[i], "--ids") && i + 1 < argc) ids_file = argv[++i];
        else if (!std::strcmp(argv[i], "--logits-out") && i + 1 < argc) logits_out = argv[++i];
        else if (!std::strcmp(argv[i], "--print-tokens")) print_tokens = true;
        else if (!std::strcmp(argv[i], "--chunk") && i + 1 < argc) chunk = uint32_t(std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "--gpus") && i + 1 < argc) n_gpus = uint32_t(std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "--spec") && i + 1 < argc) spec = uint32_t(std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "--ppbench") && i + 1 < argc) ppbench = uint32_t(std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "--pipeline")) pipeline = true;
        else if (argv[i][0] != '-') gpu = uint32_t(std::atoi(argv[i]));
    }

    GgufReader g;
    if (auto e = g.open(argv[1]); !e.empty()) { std::fprintf(stderr, "open: %s\n", e.c_str()); return 1; }
    Hyv4Config cfg;
    if (auto e = read_hyv4_config(g, cfg); !e.empty()) { std::fprintf(stderr, "config: %s\n", e.c_str()); return 1; }
    Tokenizer tok;
    if (auto e = tok.load_from_gguf(g); !e.empty()) { std::fprintf(stderr, "tok: %s\n", e.c_str()); return 1; }

    if (print_tokens) {   // tokenizer parity gate: engine ids for --prompt
        std::vector<int32_t> pt = tok.encode(prompt, false);
        std::printf("[tok]");
        for (int32_t t : pt) std::printf(" %d", t);
        std::printf("\n");
        return 0;
    }

    const uint32_t n_sel = cfg.indexer_top_k + cfg.indexer_kpool - 1;
    const uint32_t n_tf = cfg.n_transformer_layers();
    const uint32_t n_st = n_gpus >= 2 ? 2u : 1u;
    // Layer split (2 GPUs): A [0, 23) + embed, B [23, 45) + lm_head — the
    // extra A block offsets B's head/output_norm residency. IE_HY4_SPLIT
    // overrides for balance sweeps (stage B lost 1.38 GiB with the MTP kit
    // gated, so the old balance point may have moved).
    // hyv4: 78 uniform MLA blocks — even split. (The donor's 23 was tuned for
    // GLM's cheap KDA layers.) Stage A additionally carries token_embd, stage B
    // the F16-expanded lm_head (1.45 GiB), so A takes one extra block.
    uint32_t split = n_tf / 2 + 1;
    if (const char* v = std::getenv("IE_HY4_SPLIT")) {
        const uint32_t s = uint32_t(std::atoi(v));
        if (s >= 1 && s < n_tf) split = s;
    }
    // Per-card expert cache: with the resident set halved, most of each
    // card's VRAM goes to slots. Overridable via IE_HY4_ECACHE_MB as ever.
    if (n_st == 2 && std::getenv("IE_HY4_ECACHE_MB") == nullptr)
        setenv("IE_HY4_ECACHE_MB", "14336", 0);   // raw-dense frees ~12 GiB/card (P4);
        // (pre-P4 bring-up note: the F16-expanded dense
        // set is ~25 GiB/card (P4 re-packs it), leaving ~4 GiB for expert slots.
        // (donor comment, GLM geometry: 22.5 GiB/card: fits VRAM with the
        // budget-exact allocator (25600 spent-in-full oversubscribed stage A:
        // 5.46 w + 24.99 cache > 29.8) AND the 25600/87-79 geometry reproducibly
        // degenerated text at ~tok 45 on 2026-08-29 (rot-vs-bug open; battery
        // step 3 retests the old value post-reboot).
    // MTP kit loads only when spec decode is requested — otherwise the tail
    // stage skips token_embd + blk.45 (was pushing stage B past the card).
    if (spec > 0) setenv("IE_HY4_MTP", "1", 1);

    DeviceAllocator alloc[2];
    Hyv4Model   m_st[2];
    g_term_m[0] = &m_st[0];
    g_term_m[1] = &m_st[1];
    g_term_n = n_st;
    std::signal(SIGTERM, on_term);
    std::signal(SIGINT, on_term);
    // IE_HY4_EP=1: expert-parallel groundwork — both stage allocators join ONE
    // shared P2P context so either card can DMA any host bank and cross-queue
    // events work (the qwen4exp interstage-push machinery, reused).
    std::vector<sycl::device> p2p_gpus;
    std::unique_ptr<sycl::context> p2p_ctx;
    const bool ep = n_st == 2 && std::getenv("IE_HY4_EP") != nullptr &&
                    gpu_p2p_shared_context("B70", p2p_gpus, p2p_ctx);
    if (ep) std::printf("[hyv4] EP: shared P2P context up\n");
    for (uint32_t s = 0; s < n_st; ++s) {
        const uint32_t ord = n_st == 2 ? s : gpu;
        if (ep) {
            if (auto e = alloc[s].init_with(*p2p_ctx, p2p_gpus[s]); !e.empty()) { std::fprintf(stderr, "gpu%u: %s\n", ord, e.c_str()); return 1; }
        } else if (auto e = alloc[s].init("B70", ord); !e.empty()) { std::fprintf(stderr, "gpu%u: %s\n", ord, e.c_str()); return 1; }
        const uint32_t lo = (n_st == 2 && s == 1) ? split : 0;
        const uint32_t hi = (n_st == 2 && s == 0) ? split : n_tf;
        std::printf("[hyv4] loading blocks [%u, %u) on GPU %u ...\n", lo, hi, ord);
        if (auto e = m_st[s].load(alloc[s], g, cfg, 0, lo, hi); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
        if (auto e = m_st[s].init_runtime(n_sel, std::max(chunk, 8u)); !e.empty()) { std::fprintf(stderr, "rt: %s\n", e.c_str()); return 1; }
        std::printf("[hyv4] stage %u: %.2f GiB device\n", s, m_st[s].device_bytes() / 1073741824.0);
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
        std::printf("[hyv4] bank warmup: %.1fs\n",
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - tw0).count());
    }
    // IE_HY4_WARM_CACHE=<profile>: pre-fill each stage's expert cache from a
    // recorded usage profile (IE_HY4_EPROFILE_OUT) — the first prompt starts
    // at the profile's hit rate instead of 0%.
    if (const char* wp = std::getenv("IE_HY4_WARM_CACHE")) {
        auto tw0 = std::chrono::steady_clock::now();
        std::future<std::string> fw[2];
        for (uint32_t s = 0; s < n_st; ++s)
            fw[s] = std::async(std::launch::async,
                               [&m_st, s, wp] { return m_st[s].warm_cache(wp); });
        for (uint32_t s = 0; s < n_st; ++s)
            if (auto e = fw[s].get(); !e.empty()) { std::fprintf(stderr, "warm_cache: %s\n", e.c_str()); return 1; }
        std::printf("[hyv4] cache warmup: %.1fs\n",
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - tw0).count());
    }
    // IE_HY4_EP_DECODE=1 (with IE_HY4_EP=1): expert-parallel decode — card 0
    // computes even experts of every pinned MoE layer, card 1 odd; both PCIe
    // links fill concurrently. init_runtime already halved each card's own
    // cache budget; ep_enable allocates the peer-layer half-caches.
    if (std::getenv("IE_HY4_EP_DECODE")) {
        if (!ep) { std::fprintf(stderr, "IE_HY4_EP_DECODE needs IE_HY4_EP=1 (shared P2P context)\n"); return 1; }
        if (auto e = m_st[0].ep_enable(m_st[1], 0); !e.empty()) { std::fprintf(stderr, "ep: %s\n", e.c_str()); return 1; }
        if (auto e = m_st[1].ep_enable(m_st[0], 1); !e.empty()) { std::fprintf(stderr, "ep: %s\n", e.c_str()); return 1; }
    }
    Hyv4Model& m = m_st[n_st - 1];   // tail stage owns logits
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
    std::printf("[hyv4] loaded (%u stage%s), ctx cap %u\n", n_st,
                n_st > 1 ? "s" : "", n_sel);

    if (!ids_file.empty() && !logits_out.empty()) {
        // Parity gate: run EXACTLY these token ids once, dump fp32 logits for
        // every position (same file format as the llama.cpp-hy4 hy4_logits
        // oracle: u32 n_pos, u32 n_vocab, then n_pos*n_vocab fp32).
        std::vector<int32_t> pids;
        { std::ifstream f(ids_file);
          if (!f) { std::fprintf(stderr, "ids: cannot open %s\n", ids_file.c_str()); return 1; }
          long v; while (f >> v) pids.push_back(int32_t(v)); }
        const uint32_t T = uint32_t(pids.size());
        if (T == 0 || T > std::max(chunk, 8u)) { std::fprintf(stderr, "ids: bad count %u\n", T); return 1; }
        float* d_lg = static_cast<float*>(
            sycl::malloc_device(uint64_t((T + 7) & ~7u) * cfg.vocab * 4, q));
        if (auto e = fwd(pids.data(), T, 0, d_lg); !e.empty()) {
            std::fprintf(stderr, "forward: %s\n", e.c_str()); return 1;
        }
        std::vector<float> row(uint64_t(T) * cfg.vocab);
        q.memcpy(row.data(), d_lg, row.size() * 4).wait();
        FILE* o = std::fopen(logits_out.c_str(), "wb");
        if (!o) { std::fprintf(stderr, "logits-out: cannot open\n"); return 1; }
        const uint32_t nv = cfg.vocab;
        std::fwrite(&T, 4, 1, o); std::fwrite(&nv, 4, 1, o);
        std::fwrite(row.data(), 4, row.size(), o);
        std::fclose(o);
        std::printf("[parity] wrote %u x %u logits to %s\n", T, nv, logits_out.c_str());
        return 0;
    }
    if (!ppl_file.empty() && !ppbench) {
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
                nll += std::log(s) + double(mx) - double(row[tgt]);
                ++n_pred;
            }
            if (chunk > 1 || p0 == 0)
                std::printf("  [chunk @%u] running PPL %.4f over %llu preds\n",
                            p0, std::exp(nll / double(n_pred)), (unsigned long long)n_pred);
            p0 += T;
        }
        std::printf("\n[ppl] FINAL: %.4f (NLL %.5f) over %llu predictions\n",
                    std::exp(nll / double(n_pred)), nll / double(n_pred),
                    (unsigned long long)n_pred);
        sycl::free(d_logits, q);
        return 0;
    }

    if (ppbench) {
        // Chunked prefill bench over wikitext (needs --ppl <file> as the
        // token source). Serial by default; --pipeline overlaps stage A on
        // chunk k+1 with stage B on chunk k (both PCIe links pumping).
        if (ppl_file.empty()) { std::fprintf(stderr, "ppbench needs --ppl <file>\n"); return 1; }
        std::ifstream f(ppl_file);
        std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        std::vector<int32_t> ids = tok.encode(text, false);
        const uint32_t N = std::min(ppbench, (n_sel - 1) / chunk);
        if (N * chunk > ids.size()) { std::fprintf(stderr, "corpus too small\n"); return 1; }
        std::printf("[ppb] %u chunks of %u (%s)\n", N, chunk,
                    pipeline && n_st == 2 ? "PIPELINED" : "serial");
        reset_all();
        auto t0 = std::chrono::steady_clock::now();
        if (pipeline && n_st == 2) {
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

    // Greedy continuation smoke.
    std::vector<int32_t> ids = tok.encode(prompt, false);
    std::printf("[gen] prompt: \"%s\" (%zu tokens)\n[gen] ", prompt.c_str(), ids.size());
    reset_all();
    auto t0 = std::chrono::steady_clock::now();
    if (auto e = fwd(ids.data(), uint32_t(ids.size()), 0, nullptr); !e.empty()) {
        std::fprintf(stderr, "prefill: %s\n", e.c_str()); return 1;
    }
    auto t1 = std::chrono::steady_clock::now();
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
        uint32_t lastT = uint32_t(ids.size());
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
    } else {
        // IE_QUEUE_PROFILING=1: per-kernel GPU-time buckets over the decode
        // loop (queue built with profiling; ie::ps feeds g_profiler).
        KernelProfiler prof;
        const bool profiling = std::getenv("IE_QUEUE_PROFILING") != nullptr;
        if (profiling) { g_profiler = &prof; prof.begin_step(); }
        for (uint32_t s = 0; s < ngen; ++s) {
            std::printf("%s", tok.decode(std::span<const int32_t>(&t, 1)).c_str());
            std::fflush(stdout);
            if (auto e = fwd(&t, 1, pos, nullptr); !e.empty()) {
                std::fprintf(stderr, "\nstep: %s\n", e.c_str()); return 1;
            }
            ++pos;
            // IE_HY4_LOGIT_DUMP=<file>: raw fp16 logits of the FIRST T=1
            // decode step (the EP-vs-baseline numerics gate).
            if (s == 0) {
                if (const char* lf = std::getenv("IE_HY4_LOGIT_DUMP")) {
                    std::vector<sycl::half> hl(cfg.vocab);
                    q.memcpy(hl.data(), m.logits(), cfg.vocab * 2).wait();
                    if (FILE* f = std::fopen(lf, "wb")) {
                        std::fwrite(hl.data(), 2, hl.size(), f);
                        std::fclose(f);
                        std::printf("[logits -> %s]", lf);
                    }
                }
            }
            t = argmax_h(q, m.logits(), cfg.vocab);
            ++emitted;
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
            for (size_t i = 0; i < stats.size() && i < 16; ++i)
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
    for (uint32_t s2 = 0; s2 < n_st; ++s2)
        if (m_st[s2].n_cpu_experts)
            std::printf("[gen] stage %u q*: %llu CPU experts, worker %.2fs (%.3f ms/expert), main-thread join wait %.2fs\n",
                        s2, (unsigned long long)m_st[s2].n_cpu_experts, m_st[s2].t_cpu_compute,
                        1e3 * m_st[s2].t_cpu_compute / double(m_st[s2].n_cpu_experts),
                        m_st[s2].t_cpu_join);
    if (std::getenv("IE_HY4_SHADOW_COUNT")) {
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
    if (std::getenv("IE_HY4_PREFETCH")) {
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
        if (const char* out = std::getenv("IE_HY4_MISSPROF_OUT")) {
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
        if (const char* out = std::getenv("IE_HY4_EPROFILE_OUT")) {
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
