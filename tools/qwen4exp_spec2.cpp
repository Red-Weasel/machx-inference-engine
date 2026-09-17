// tools/qwen4exp_spec2.cpp — Flash-Next MTP spec decode on the 2-GPU
// layer-split pipeline (A: layers [0,split) on GPU0; B: [split,48) + lm_head
// + the MTP layer on GPU1; wide residual bounces through host).
//
// LOSSLESS GATE: the spec loop must emit exactly the 2-GPU plain-greedy
// sequence. Both stages run spec-verify mode, so every verify forward is
// bit-identical per row to the pipeline's own T=1 decode; each stage
// checkpoints its DeltaNet/PLE state and commit_verify() adopts the accepted
// prefix on BOTH cards (no restore/replay).
//
// usage: ie-qwen4exp-spec2 <model.gguf> <mtp.gguf> [-n N] [-k K] [-p prompt] [split]
#include "ie/gguf.hpp"
#include "ie/kernel_profiler.hpp"
#include "ie/ops.hpp"
#include "ie/model_config.hpp"
#include "ie/qwen4exp.hpp"
#include "ie/tokenizer.hpp"

#include <chrono>
#include <future>
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
int32_t host_argmax_buf(const sycl::half* p, uint32_t V) {
    float mx = -1e30f; int32_t am = 0;
    for (uint32_t v = 0; v < V; ++v)
        if (float(p[v]) > mx) { mx = float(p[v]); am = int32_t(v); }
    return am;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s <model.gguf> <mtp.gguf> [-n N] [-k K] [-p prompt] [split]\n", argv[0]); return 2; }
    uint32_t n_gen = 64, K = 3, split = 24;
    std::string prompt = "<|im_start|>user\nExplain why the sky is blue in one short paragraph.<|im_end|>\n<|im_start|>assistant\n";
    for (int i = 3; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) n_gen = uint32_t(std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "-k") && i + 1 < argc) K = uint32_t(std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "-p") && i + 1 < argc) prompt = argv[++i];
        else split = uint32_t(std::atoi(argv[i]));
    }

    GgufReader g, mg;
    if (auto e = g.open(argv[1]); !e.empty()) { std::fprintf(stderr, "open: %s\n", e.c_str()); return 1; }
    if (auto e = mg.open(argv[2]); !e.empty()) { std::fprintf(stderr, "mtp open: %s\n", e.c_str()); return 1; }
    Qwen4ExpConfig cfg;
    if (auto e = read_qwen4exp_config(g, cfg); !e.empty()) { std::fprintf(stderr, "config: %s\n", e.c_str()); return 1; }
    Tokenizer tok;
    if (auto e = tok.load_from_gguf(g); !e.empty()) { std::fprintf(stderr, "tok: %s\n", e.c_str()); return 1; }

    DeviceAllocator a0, a1;
    // IE_P2P=1: shared context + peer access — the A->B wide handoff PUSHES
    // device-to-device (2x per verify round) instead of host-bouncing.
    std::vector<sycl::device> p2p_gpus;
    std::unique_ptr<sycl::context> p2p_ctx;
    const bool p2p = std::getenv("IE_P2P") != nullptr &&
                     gpu_p2p_shared_context("B70", p2p_gpus, p2p_ctx);
    if (std::getenv("IE_P2P") && !p2p)
        std::fprintf(stderr, "IE_P2P requested but unavailable — host bounce\n");
    if (p2p) {
        if (auto e = a0.init_with(*p2p_ctx, p2p_gpus[0]); !e.empty()) { std::fprintf(stderr, "gpu0: %s\n", e.c_str()); return 1; }
        if (auto e = a1.init_with(*p2p_ctx, p2p_gpus[1]); !e.empty()) { std::fprintf(stderr, "gpu1: %s\n", e.c_str()); return 1; }
    } else {
    if (auto e = a0.init("B70", 0); !e.empty()) { std::fprintf(stderr, "gpu0: %s\n", e.c_str()); return 1; }
    if (auto e = a1.init("B70", 1); !e.empty()) { std::fprintf(stderr, "gpu1: %s\n", e.c_str()); return 1; }
    }
    Qwen4ExpModel A, B;
    if (auto e = A.load(a0, g, cfg, 0, 0, split); !e.empty()) { std::fprintf(stderr, "A: %s\n", e.c_str()); return 1; }
    if (auto e = B.load(a1, g, cfg, 0, split, cfg.n_layers); !e.empty()) { std::fprintf(stderr, "B: %s\n", e.c_str()); return 1; }
    if (auto e = B.load_mtp(mg, &g); !e.empty()) { std::fprintf(stderr, "mtp: %s\n", e.c_str()); return 1; }
    if (auto e = A.init_runtime(8192, 1024); !e.empty()) { std::fprintf(stderr, "A rt: %s\n", e.c_str()); return 1; }
    if (auto e = B.init_runtime(8192, 1024); !e.empty()) { std::fprintf(stderr, "B rt: %s\n", e.c_str()); return 1; }
    std::printf("\n\033[1mFlash-Next 2-GPU MTP spec gate\033[0m  split %u, K=%u, "
                "device A %.2f + B %.2f GiB\n\n", split, K,
                A.device_bytes() / 1073741824.0, B.device_bytes() / 1073741824.0);

    const uint32_t V = cfg.vocab;
    std::vector<int32_t> ids = tok.encode(prompt, true);
    const uint32_t P0 = uint32_t(ids.size());
    std::vector<float> wide(uint64_t(1024) * cfg.hc_count * cfg.hidden);
    sycl::queue& qB = B.queue();
    std::vector<sycl::half> l16(V);
    if (p2p) A.set_wide_peer(B.wide_device());

    auto pipe_fwd = [&](const int32_t* toks, uint32_t T, uint32_t pos) -> std::string {
        if (auto e = A.forward_range(toks, T, pos, nullptr, wide.data(), nullptr); !e.empty()) return e;
        return B.forward_range(toks, T, pos, p2p ? nullptr : wide.data(),
                               nullptr, nullptr, /*wide_in_device=*/p2p);
    };
    // Device argmax on BOTH lanes (same kernel, same tie-break on the
    // reference and the verify sides -> losslessness is internal-consistent;
    // saves the 500 KB D2H + host scan per step / per verify row).
    auto* d_am = static_cast<int32_t*>(a1.malloc(8 * sizeof(int32_t)));
    auto argmax_last = [&]() {
        int32_t am = 0;
        sample_argmax(qB, B.logits(), d_am, V);
        qB.memcpy(&am, d_am, sizeof(int32_t)).wait();
        return am;
    };

    // ---- reference: 2-GPU plain greedy -----------------------------------
    std::vector<int32_t> ref;
    {
        A.reset_state(); B.reset_state();
        uint32_t pos = 0;
        while (pos < P0) {
            const uint32_t n = std::min(1024u, P0 - pos);
            if (auto e = pipe_fwd(ids.data() + pos, n, pos); !e.empty()) {
                std::fprintf(stderr, "prefill: %s\n", e.c_str()); return 1;
            }
            pos += n;
        }
        int32_t t = argmax_last();
        const auto t0 = std::chrono::steady_clock::now();
        for (uint32_t s = 0; s < n_gen; ++s) {
            ref.push_back(t);
            if (auto e = pipe_fwd(&t, 1, pos); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
            ++pos;
            t = argmax_last();
        }
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        std::printf("  2-GPU plain greedy: %.1f ms/token (%.2f tok/s)\n",
                    ms / n_gen, n_gen * 1000.0 / ms);
    }

    // ---- spec loop over the pipeline -------------------------------------
    std::vector<int32_t> spec;
    uint32_t rounds = 0, accepted_total = 0;
    {
        A.reset_state(); B.reset_state();
        uint32_t pos = 0;
        while (pos < P0) {
            const uint32_t n = std::min(1024u, P0 - pos);
            if (auto e = pipe_fwd(ids.data() + pos, n, pos); !e.empty()) {
                std::fprintf(stderr, "prefill: %s\n", e.c_str()); return 1;
            }
            if (auto e = B.mtp_ingest(ids.data() + pos, n, pos); !e.empty()) {
                std::fprintf(stderr, "ingest: %s\n", e.c_str()); return 1;
            }
            pos += n;
        }
        int32_t t = argmax_last();
        A.set_spec_verify(true);
        B.set_spec_verify(true);
        double ms_draft = 0, ms_vA = 0, ms_verify = 0, ms_rest = 0;
        auto lap = [last = std::chrono::steady_clock::now()](double& acc) mutable {
            const auto now = std::chrono::steady_clock::now();
            acc += std::chrono::duration<double, std::milli>(now - last).count();
            last = now;
        };
        std::vector<int32_t> draft(K), batch(K + 1);
        std::vector<uint32_t> pos_hits(K, 0);
        std::vector<sycl::half> lrows((K + 1) * size_t(V));
        auto* d_lrows = static_cast<sycl::half*>(a1.malloc((K + 1) * size_t(V) * 2));
        // Verify-phase kernel decomposition (IE_QUEUE_PROFILING=1): per-round
        // named-kernel buckets across BOTH stages, accumulated after the JIT
        // warm-up round. Wall stays valid from the unprofiled runs; this run
        // answers kernels-vs-host-gap inside the verify wall.
        const bool vprof_on = std::getenv("IE_QUEUE_PROFILING") != nullptr;
        KernelProfiler vprof;
        std::vector<KernelProfiler::Stat> vacc;
        uint32_t vrounds = 0;
        sycl::queue& qA = A.queue();
        const auto t0 = std::chrono::steady_clock::now();
        while (spec.size() < n_gen) {
            // (Overlap of A's row-0 stage with the draft was FALSIFIED
            // 2026-08-27: A's T=1 pass costs a full dense-weight read
            // (~33 ms, T-independent) to hide a 12 ms draft — net -6 ms/tok.
            // The checkpoint-append machinery it exercised stays in the
            // model for future streaming shapes.)
            const std::string ed = B.mtp_draft(t, pos, K, draft.data());
            if (!ed.empty()) { std::fprintf(stderr, "draft: %s\n", ed.c_str()); return 1; }
            lap(ms_draft);
            batch[0] = t;
            for (uint32_t k = 0; k < K; ++k) batch[k + 1] = draft[k];
            const bool varm = vprof_on && rounds >= 1;
            if (varm) { g_profiler = &vprof; vprof.begin_step(); }
            if (auto e = A.forward_range(batch.data(), K + 1, pos, nullptr,
                                         wide.data(), nullptr); !e.empty()) {
                std::fprintf(stderr, "verifyA: %s\n", e.c_str()); return 1;
            }
            lap(ms_vA);
            if (auto e = B.forward_range(batch.data(), K + 1, pos,
                                         p2p ? nullptr : wide.data(),
                                         nullptr, nullptr,
                                         /*wide_in_device=*/p2p); !e.empty()) {
                std::fprintf(stderr, "verifyB: %s\n", e.c_str()); return 1;
            }
            if (varm) {
                qA.wait(); qB.wait();
                for (const auto& s : vprof.harvest()) {
                    auto it = std::find_if(vacc.begin(), vacc.end(),
                        [&](const KernelProfiler::Stat& v) { return v.name == s.name; });
                    if (it == vacc.end()) vacc.push_back(s);
                    else { it->calls += s.calls; it->total_ns += s.total_ns; }
                }
                g_profiler = nullptr;
                ++vrounds;
            }
            lap(ms_verify);
            if (auto e = B.logits_rows(K + 1, d_lrows); !e.empty()) {
                std::fprintf(stderr, "rows: %s\n", e.c_str()); return 1;
            }
            std::vector<int32_t> a(K + 1);
            for (uint32_t i = 0; i <= K; ++i)
                sample_argmax(qB, d_lrows + uint64_t(i) * V, d_am + i, V);
            qB.memcpy(a.data(), d_am, (K + 1) * sizeof(int32_t)).wait();
            uint32_t n_ok = 0;
            while (n_ok < K && draft[n_ok] == a[n_ok]) ++n_ok;
            for (uint32_t i = 0; i < K; ++i)
                if (i < n_ok || (i == n_ok && draft[i] == a[i])) ++pos_hits[i];
            spec.push_back(t);
            for (uint32_t i = 0; i < n_ok && spec.size() < n_gen + 1; ++i)
                spec.push_back(draft[i]);
            ++rounds; accepted_total += n_ok;
            if (auto e = A.commit_verify(n_ok + 1); !e.empty()) { std::fprintf(stderr, "commitA: %s\n", e.c_str()); return 1; }
            if (auto e = B.commit_verify(n_ok + 1); !e.empty()) { std::fprintf(stderr, "commitB: %s\n", e.c_str()); return 1; }
            if (auto e = B.mtp_ingest(batch.data(), n_ok + 1, pos); !e.empty()) {
                std::fprintf(stderr, "ingest: %s\n", e.c_str()); return 1;
            }
            pos += n_ok + 1;
            t = a[n_ok];
            lap(ms_rest);
        }
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        if (spec.size() > n_gen) spec.resize(n_gen);
        std::printf("  2-GPU spec K=%u : %.1f ms/token (%.2f tok/s), rounds %u, "
                    "mean accepted %.2f\n",
                    K, ms / spec.size(), spec.size() * 1000.0 / ms, rounds,
                    rounds ? double(accepted_total) / rounds : 0.0);
        if (rounds)
            std::printf("  round: draft %.1f  verifyA %.1f  verifyB %.1f  logits+commit+ingest %.1f ms\n",
                        ms_draft / rounds, ms_vA / rounds, ms_verify / rounds,
                        ms_rest / rounds);
        if (vprof_on && vrounds) {
            std::sort(vacc.begin(), vacc.end(),
                      [](const KernelProfiler::Stat& a, const KernelProfiler::Stat& b) {
                          return a.total_ns > b.total_ns;
                      });
            double busy = 0;
            for (const auto& s : vacc) busy += s.total_ms();
            std::printf("  verify kernel buckets (avg over %u rounds; busy %.1f ms/round):\n",
                        vrounds, busy / vrounds);
            for (size_t i = 0; i < vacc.size() && i < 14; ++i)
                std::printf("    %-22s %7.1f calls/rnd %8.2f ms/rnd\n",
                            vacc[i].name.c_str(),
                            double(vacc[i].calls) / vrounds,
                            vacc[i].total_ms() / vrounds);
        }
        std::printf("  draft hit-rate by depth:");
        for (uint32_t i = 0; i < K; ++i)
            std::printf(" k%u %.0f%%", i, 100.0 * pos_hits[i] / std::max(1u, rounds));
        std::printf("\n  ecache: A %.1f%%  B %.1f%% hit\n",
                    100.0 * A.ecache_hits / std::max<uint64_t>(1, A.ecache_hits + A.ecache_misses),
                    100.0 * B.ecache_hits / std::max<uint64_t>(1, B.ecache_hits + B.ecache_misses));
        a1.free(d_lrows);
    }

    uint32_t diff = 0;
    for (uint32_t i = 0; i < n_gen && i < spec.size() && i < ref.size(); ++i)
        if (spec[i] != ref[i]) ++diff;
    check(diff == 0 && spec.size() == ref.size(), "LOSSLESS: 2-GPU spec == 2-GPU plain greedy");
    std::printf("  text: %s\n",
                tok.decode(std::span<const int32_t>(spec), true, {}).substr(0, 160).c_str());
    std::printf("\n\033[1m%s\033[0m\n\n", g_fail ? "\033[31mGATE FAILED\033[0m" : "\033[32mGATE PASSED\033[0m");
    return g_fail ? 1 : 0;
}
