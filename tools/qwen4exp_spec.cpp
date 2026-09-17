// tools/qwen4exp_spec.cpp — Flash-Next MTP self-speculative GREEDY decode.
//
// LOSSLESS GATE: the spec loop must emit exactly the plain-greedy token
// sequence (verification corrects every draft miss; drafts only buy speed).
// Also reports the mean accepted length — the tech report's four-step MTP
// measures ~3.4 with QSA, so K=3 should accept ~2.3-2.6 drafts/round if the
// (reconstructed) fusion layout is right; a wrong layout shows ~0.
//
// usage: ie-qwen4exp-spec <model.gguf> <mtp-head.gguf> [-n N] [-p prompt] [gpu]
#include "ie/gguf.hpp"
#include "ie/kernel_profiler.hpp"
#include "ie/model_config.hpp"
#include "ie/qwen4exp.hpp"
#include "ie/tokenizer.hpp"

#include <algorithm>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace ie;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? "\033[32mOK\033[0m" : "\033[31mFAIL\033[0m", what);
    if (!ok) ++g_fail;
}
int32_t host_argmax(sycl::queue& q, const sycl::half* dl, std::vector<sycl::half>& tmp,
                    uint32_t V) {
    q.memcpy(tmp.data(), dl, uint64_t(V) * 2).wait();
    float mx = -1e30f; int32_t am = 0;
    for (uint32_t v = 0; v < V; ++v)
        if (float(tmp[v]) > mx) { mx = float(tmp[v]); am = int32_t(v); }
    return am;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s <model.gguf> <mtp.gguf> [-n N] [-p prompt] [gpu]\n", argv[0]); return 2; }
    uint32_t n_gen = 64, ordinal = 0, K = 3;
    bool prof_verify = false;   // --prof: kernel buckets for one verify fwd
                                // (needs IE_QUEUE_PROFILING=1)
    std::string prompt = "<|im_start|>user\nExplain why the sky is blue in one short paragraph.<|im_end|>\n<|im_start|>assistant\n";
    for (int i = 3; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) n_gen = uint32_t(std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "-p") && i + 1 < argc) prompt = argv[++i];
        else if (!std::strcmp(argv[i], "-k") && i + 1 < argc) K = uint32_t(std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "--prof")) prof_verify = true;
        else ordinal = uint32_t(std::atoi(argv[i]));
    }

    GgufReader g, mg;
    if (auto e = g.open(argv[1]); !e.empty()) { std::fprintf(stderr, "open: %s\n", e.c_str()); return 1; }
    if (auto e = mg.open(argv[2]); !e.empty()) { std::fprintf(stderr, "mtp open: %s\n", e.c_str()); return 1; }
    Qwen4ExpConfig cfg;
    if (auto e = read_qwen4exp_config(g, cfg); !e.empty()) { std::fprintf(stderr, "config: %s\n", e.c_str()); return 1; }
    Tokenizer tok;
    if (auto e = tok.load_from_gguf(g); !e.empty()) { std::fprintf(stderr, "tok: %s\n", e.c_str()); return 1; }
    DeviceAllocator alloc;
    if (auto e = alloc.init("B70", ordinal); !e.empty()) { std::fprintf(stderr, "gpu: %s\n", e.c_str()); return 1; }
    Qwen4ExpModel m;
    if (auto e = m.load(alloc, g, cfg); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
    if (auto e = m.load_mtp(mg); !e.empty()) { std::fprintf(stderr, "mtp: %s\n", e.c_str()); return 1; }
    if (auto e = m.init_runtime(8192, 1024); !e.empty()) { std::fprintf(stderr, "rt: %s\n", e.c_str()); return 1; }
    std::printf("\n\033[1mFlash-Next MTP spec gate\033[0m  K=%u, device %.2f GiB\n\n",
                K, m.device_bytes() / 1073741824.0);

    std::vector<int32_t> ids = tok.encode(prompt, true);
    const uint32_t P0 = uint32_t(ids.size());
    const uint32_t V = cfg.vocab;
    std::vector<sycl::half> ltmp(V);
    sycl::queue& q = m.queue();

    // ---- reference: plain greedy ------------------------------------------
    std::vector<int32_t> ref;
    {
        m.reset_state();
        m.forward(ids.data(), P0, 0, nullptr);
        int32_t t = host_argmax(q, m.logits(), ltmp, V);
        uint32_t pos = P0;
        const auto t0 = std::chrono::steady_clock::now();
        for (uint32_t s = 0; s < n_gen; ++s) {
            ref.push_back(t);
            m.forward(&t, 1, pos, nullptr); ++pos;
            t = host_argmax(q, m.logits(), ltmp, V);
        }
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        std::printf("  plain greedy: %.1f ms/token (%.2f tok/s)\n",
                    ms / n_gen, n_gen * 1000.0 / ms);
    }

    // ---- spec loop --------------------------------------------------------
    std::vector<int32_t> spec;
    uint32_t rounds = 0, accepted_total = 0;
    {
        m.reset_state();
        // prefill in chunks + mtp ingest
        uint32_t pos = 0;
        while (pos < P0) {
            const uint32_t n = std::min(1024u, P0 - pos);
            m.forward(ids.data() + pos, n, pos, nullptr);
            m.mtp_ingest(ids.data() + pos, n, pos);
            pos += n;
        }
        int32_t t = host_argmax(q, m.logits(), ltmp, V);
        // Lossless requires verify batches to be bit-identical to serial
        // decode; only forwards with T in [2,16] change behavior (prefill
        // chunks above stay on the fast batched kernels).
        m.set_spec_verify(true);
        double ms_draft = 0, ms_verify = 0, ms_logits = 0, ms_host = 0, ms_commit = 0;
        auto lap = [last = std::chrono::steady_clock::now()](double& acc) mutable {
            const auto now = std::chrono::steady_clock::now();
            acc += std::chrono::duration<double, std::milli>(now - last).count();
            last = now;
        };
        std::vector<int32_t> draft(K), batch(K + 1);
        std::vector<uint32_t> pos_hits(K, 0);
        std::vector<sycl::half> lrows((K + 1) * size_t(V));
        auto* d_lrows = static_cast<sycl::half*>(alloc.malloc((K + 1) * size_t(V) * 2));
        const auto t0 = std::chrono::steady_clock::now();
        while (spec.size() < n_gen) {
            if (auto e = m.mtp_draft(t, pos, K, draft.data()); !e.empty()) {
                std::fprintf(stderr, "draft: %s\n", e.c_str()); return 1;
            }
            lap(ms_draft);
            batch[0] = t;
            for (uint32_t k = 0; k < K; ++k) batch[k + 1] = draft[k];
            // Verify forward checkpoints DeltaNet/PLE state after every row;
            // commit_verify below adopts the accepted prefix's state — no
            // snapshot/restore and no replay forward.
            KernelProfiler vprof;
            const bool prof_this = prof_verify && rounds == 4;
            if (prof_this) { g_profiler = &vprof; vprof.begin_step(); }
            m.forward(batch.data(), K + 1, pos, nullptr);
            q.wait();
            if (prof_this) {
                auto st = vprof.harvest();
                g_profiler = nullptr;
                double busy = 0;
                for (auto& s : st) busy += s.total_ms();
                std::sort(st.begin(), st.end(),
                          [](const auto& a, const auto& b) { return a.total_ns > b.total_ns; });
                std::printf("  -- verify fwd profile (round 4): GPU busy %.1f ms --\n", busy);
                for (size_t i = 0; i < st.size() && i < 14; ++i)
                    std::printf("     %-22s %6u %10.2f %9.1f\n", st[i].name.c_str(),
                                st[i].calls, st[i].total_ms(), st[i].avg_ms() * 1000.0);
            }
            lap(ms_verify);
            m.logits_rows(K + 1, d_lrows);
            q.memcpy(lrows.data(), d_lrows, lrows.size() * 2).wait();
            lap(ms_logits);
            // a[i] = model's next-token at row i
            std::vector<int32_t> a(K + 1);
            for (uint32_t i = 0; i <= K; ++i) {
                float mx = -1e30f; int32_t am = 0;
                const sycl::half* lr = lrows.data() + uint64_t(i) * V;
                for (uint32_t v = 0; v < V; ++v)
                    if (float(lr[v]) > mx) { mx = float(lr[v]); am = int32_t(v); }
                a[i] = am;
            }
            uint32_t n_ok = 0;
            while (n_ok < K && draft[n_ok] == a[n_ok]) ++n_ok;
            for (uint32_t i = 0; i < K; ++i)
                if (i < n_ok || (i == n_ok && draft[i] == a[i])) ++pos_hits[i];
            // commit: t, then the n_ok accepted drafts; next t = a[n_ok]
            spec.push_back(t);
            for (uint32_t i = 0; i < n_ok && spec.size() < n_gen + 1; ++i)
                spec.push_back(draft[i]);
            ++rounds; accepted_total += n_ok;
            lap(ms_host);
            if (auto e = m.commit_verify(n_ok + 1); !e.empty()) {
                std::fprintf(stderr, "commit: %s\n", e.c_str()); return 1;
            }
            m.mtp_ingest(batch.data(), n_ok + 1, pos);
            pos += n_ok + 1;
            lap(ms_commit);
            t = a[n_ok];
        }
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        if (spec.size() > n_gen) spec.resize(n_gen);
        std::printf("  spec K=%u : %.1f ms/token (%.2f tok/s), rounds %u, "
                    "mean accepted %.2f\n",
                    K, ms / spec.size(), spec.size() * 1000.0 / ms, rounds,
                    rounds ? double(accepted_total) / rounds : 0.0);
        if (rounds)
            std::printf("  round: draft %.1f  verify %.1f  logits %.1f  "
                        "host %.1f  commit+ingest %.1f ms\n",
                        ms_draft / rounds, ms_verify / rounds,
                        ms_logits / rounds, ms_host / rounds,
                        ms_commit / rounds);
        std::printf("  draft hit-rate by depth:");
        for (uint32_t i = 0; i < K; ++i)
            std::printf(" k%u %.0f%%", i, 100.0 * pos_hits[i] / std::max(1u, rounds));
        std::printf("\n");
        std::printf("  ecache: %.1f%% hit (%llu hits, %llu misses)\n",
                    100.0 * m.ecache_hits /
                        std::max<uint64_t>(1, m.ecache_hits + m.ecache_misses),
                    (unsigned long long)m.ecache_hits,
                    (unsigned long long)m.ecache_misses);
        alloc.free(d_lrows);
    }

    uint32_t diff = 0;
    for (uint32_t i = 0; i < n_gen && i < spec.size() && i < ref.size(); ++i)
        if (spec[i] != ref[i]) ++diff;
    check(diff == 0 && spec.size() == ref.size(), "LOSSLESS: spec == plain greedy");
    std::printf("  text: %s\n",
                tok.decode(std::span<const int32_t>(spec), true, {}).substr(0, 160).c_str());
    std::printf("\n\033[1m%s\033[0m\n\n", g_fail ? "\033[31mGATE FAILED\033[0m" : "\033[32mGATE PASSED\033[0m");
    return g_fail ? 1 : 0;
}
