// tools/qwen4exp_spec3.cpp — Flash-Next MTP spec decode, REPLICATED
// expert-parallel topology (2026-08-27): both cards hold ALL layers (B also
// the MTP head; banks pinned ONCE via the shared-context fleet + bank donor);
// per verify round both cards run the SAME forward concurrently, each
// computing verify-MoE cells only for its expert half; a per-layer cell
// exchange (D2H -> peer H2D -> device merge) completes both sides before the
// ordered reduce.
//
// LOSSLESS GATE: reference = card B alone (full model, plain greedy). The
// split verify is bit-identical to an unsplit one iff A's cells match what B
// would compute — identical kernels on identical replicated inputs from the
// same pinned banks (deterministic in-order queues). The gate decides.
//
// usage: ie-qwen4exp-spec3 <model.gguf> <mtp.gguf> [-n N] [-k K] [-p prompt]
#include "ie/allocator.hpp"
#include "ie/gguf.hpp"
#include "ie/model_config.hpp"
#include "ie/ops.hpp"
#include "ie/qwen4exp.hpp"
#include "ie/tokenizer.hpp"

#include <atomic>
#include <barrier>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <string>
#include <vector>

using namespace ie;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? "\033[32mOK\033[0m" : "\033[31mFAIL\033[0m", what);
    if (!ok) ++g_fail;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s <model.gguf> <mtp.gguf> [-n N] [-k K] [-p prompt]\n", argv[0]); return 2; }
    uint32_t n_gen = 64, K = 3;
    std::string prompt = "<|im_start|>user\nExplain why the sky is blue in one short paragraph.<|im_end|>\n<|im_start|>assistant\n";
    for (int i = 3; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) n_gen = uint32_t(std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "-k") && i + 1 < argc) K = uint32_t(std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "-p") && i + 1 < argc) prompt = argv[++i];
    }

    GgufReader g, mg;
    if (auto e = g.open(argv[1]); !e.empty()) { std::fprintf(stderr, "open: %s\n", e.c_str()); return 1; }
    if (auto e = mg.open(argv[2]); !e.empty()) { std::fprintf(stderr, "mtp open: %s\n", e.c_str()); return 1; }
    Qwen4ExpConfig cfg;
    if (auto e = read_qwen4exp_config(g, cfg); !e.empty()) { std::fprintf(stderr, "config: %s\n", e.c_str()); return 1; }
    Tokenizer tok;
    if (auto e = tok.load_from_gguf(g); !e.empty()) { std::fprintf(stderr, "tok: %s\n", e.c_str()); return 1; }

    DeviceFleet fleet;
    // enable_p2p_transfers=true: the per-layer cell exchange PUSHES
    // device-to-device (the falsifier was the host-bounce exchange).
    if (auto e = fleet.init(2, "B70", true); !e.empty()) { std::fprintf(stderr, "fleet: %s\n", e.c_str()); return 1; }
    if (fleet.size() < 2 || !fleet.shared_ctx()) {
        std::fprintf(stderr, "spec3 needs 2 GPUs in a shared context\n"); return 1;
    }
    Qwen4ExpModel A, B;
    if (auto e = A.load(fleet.dev(0), g, cfg); !e.empty()) { std::fprintf(stderr, "A: %s\n", e.c_str()); return 1; }
    if (auto e = B.load(fleet.dev(1), g, cfg); !e.empty()) { std::fprintf(stderr, "B: %s\n", e.c_str()); return 1; }
    if (auto e = B.load_mtp(mg, &g); !e.empty()) { std::fprintf(stderr, "mtp: %s\n", e.c_str()); return 1; }
    if (auto e = A.init_runtime(8192, 1024); !e.empty()) { std::fprintf(stderr, "A rt: %s\n", e.c_str()); return 1; }
    B.set_bank_donor(&A);
    if (auto e = B.init_runtime(8192, 1024); !e.empty()) { std::fprintf(stderr, "B rt: %s\n", e.c_str()); return 1; }
    const uint32_t EHALF = cfg.n_experts / 2;
    std::printf("\n\033[1mFlash-Next EXPERT-PARALLEL spec gate\033[0m  K=%u, "
                "A experts [0,%u) %.2f GiB, B [%u,%u) %.2f GiB\n\n",
                K, EHALF, A.device_bytes() / 1073741824.0,
                EHALF, cfg.n_experts, B.device_bytes() / 1073741824.0);

    const uint32_t V = cfg.vocab, NU = cfg.n_experts_used, H = cfg.hidden;
    std::vector<int32_t> ids = tok.encode(prompt, true);
    const uint32_t P0 = uint32_t(ids.size());
    sycl::queue& qA = A.queue();
    sycl::queue& qB = B.queue();
    const uint32_t Tv = K + 1;
    const uint64_t cellsN = uint64_t(16) * NU * H;   // Tv <= 16 cap

    // ---- reference: card B alone, plain greedy ---------------------------
    std::vector<int32_t> ref;
    auto* d_amB = static_cast<int32_t*>(fleet.dev(1).malloc(8 * sizeof(int32_t)));
    auto argmaxB = [&](const sycl::half* lg, uint32_t n, int32_t* slot) {
        int32_t am = 0;
        sample_argmax(qB, lg, slot, n);
        qB.memcpy(&am, slot, sizeof(int32_t)).wait();
        return am;
    };
    {
        B.reset_state();
        uint32_t pos = 0;
        while (pos < P0) {
            const uint32_t n = std::min(1024u, P0 - pos);
            if (auto e = B.forward(ids.data() + pos, n, pos, nullptr); !e.empty()) {
                std::fprintf(stderr, "ref prefill: %s\n", e.c_str()); return 1;
            }
            pos += n;
        }
        int32_t t = argmaxB(B.logits(), V, d_amB);
        const auto t0 = std::chrono::steady_clock::now();
        for (uint32_t s = 0; s < n_gen; ++s) {
            ref.push_back(t);
            if (auto e = B.forward(&t, 1, pos, nullptr); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
            ++pos;
            t = argmaxB(B.logits(), V, d_amB);
        }
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        std::printf("  replicated plain greedy (B): %.1f ms/token (%.2f tok/s)\n",
                    ms / n_gen, n_gen * 1000.0 / ms);
    }

    // ---- exchange plumbing ------------------------------------------------
    // P2P DIRECT exchange (2026-08-28): each card PUSHES its own cells
    // straight into the peer's parity-buffered device scratch from its OWN
    // queue (the verified direction on this fabric — peer READS are broken),
    // then merges from LOCAL scratch after the barrier. The old host-bounce
    // exchange (D2H stage -> barrier -> H2D pull, 2 full waits/layer) was
    // the 26.6 ms/round falsifier; IE_SPEC3_HOST=1 restores it for A/B.
    // Parity safety: merge(L) reads scr[c][L&1] async; the next write into
    // that slot is the peer's push at L+2, which the peer cannot even ISSUE
    // until it passed barrier(L+1) — and card c only arrives there after its
    // q.wait(L+1) drained merge(L). One-slot scratch would race at L+1;
    // two parities suffice.
    const bool exch_host = std::getenv("IE_SPEC3_HOST") != nullptr || !fleet.p2p();
    if (exch_host)
        std::fprintf(stderr, "[spec3] HOST-BOUNCE exchange (%s)\n",
                     fleet.p2p() ? "IE_SPEC3_HOST=1" : "no P2P");
    // EVENT-CHAINED mode (default under P2P): no per-layer q.wait()/barrier —
    // those drained both pipelines 48x/round and WERE the 24-27 ms exchange
    // (the copies are ~20 us). Cross-queue sycl events (legal in the shared
    // context) express the real deps: merge(c,L) depends_on peer push(L);
    // push(c,L) depends_on peer merge(L-2) (the parity slot's last reader).
    // Host threads only hand event HANDLES through seq-stamped mailboxes
    // (acquire/release; reader spins ~us). One qA+qB drain per ROUND kills
    // cross-round slot hazards. IE_SPEC3_BULKSYNC=1 restores the old form.
    const bool exch_bulk = exch_host ||
                           std::getenv("IE_SPEC3_BULKSYNC") != nullptr;
    struct ExchSlot {
        std::atomic<uint64_t> seq_push{0}, seq_merge{0};
        sycl::event push_ev, merge_ev;
    };
    static ExchSlot slots[2][64];
    std::atomic<uint64_t> round_id{0};
    sycl::half* stage[2][2] = {{nullptr, nullptr}, {nullptr, nullptr}};
    sycl::half* scr[2][2];
    uint8_t*    d_own[2];
    for (int c = 0; c < 2; ++c) {
        for (int pb = 0; pb < 2; ++pb) {
            scr[c][pb] = static_cast<sycl::half*>(fleet.dev(c).malloc(cellsN * 2));
            if (!scr[c][pb]) { std::fprintf(stderr, "scr alloc\n"); return 1; }
            if (exch_host) {
                stage[c][pb] = static_cast<sycl::half*>(
                    sycl::malloc_host(cellsN * 2, c ? qB : qA));
                if (!stage[c][pb]) { std::fprintf(stderr, "stage alloc\n"); return 1; }
            }
        }
        d_own[c] = static_cast<uint8_t*>(fleet.dev(c).malloc(16 * NU));
        if (!d_own[c]) { std::fprintf(stderr, "own alloc\n"); return 1; }
    }
    std::barrier bar(2);
    double ms_exch = 0;
    auto make_sync = [&](int c, Qwen4ExpModel* self, Qwen4ExpModel* peer) {
        return [&, c, self, peer](uint32_t L) {
            sycl::queue& q = self->queue();
            const int pb = int(L & 1);
            const auto t0 = std::chrono::steady_clock::now();
            if (!exch_bulk) {
                const uint64_t r = round_id.load(std::memory_order_acquire);
                const uint64_t bytes = uint64_t(Tv) * NU * H * 2;
                // Push own cells into the peer's parity slot, gated on the
                // peer's merge of that slot two layers back (its last reader).
                sycl::event push;
                if (L >= 2) {
                    while (slots[1 - c][L - 2].seq_merge.load(std::memory_order_acquire) != r) {}
                    push = q.submit([&](sycl::handler& h) {
                        h.depends_on(slots[1 - c][L - 2].merge_ev);
                        h.memcpy(scr[1 - c][pb], self->verify_cells(), bytes);
                    });
                } else {
                    push = q.memcpy(scr[1 - c][pb], self->verify_cells(), bytes);
                }
                q.memcpy(d_own[c], self->tp_cell_owner_.data(), uint64_t(Tv) * NU);
                // Publish the push, then merge once the PEER's push arrives.
                slots[c][L].push_ev = push;
                slots[c][L].seq_push.store(r, std::memory_order_release);
                while (slots[1 - c][L].seq_push.load(std::memory_order_acquire) != r) {}
                sycl::event peer_push = slots[1 - c][L].push_ev;
                sycl::half* cells = self->verify_cells();
                const sycl::half* sc2 = scr[c][pb];
                const uint8_t* own = d_own[c];
                const uint32_t H_ = H;
                sycl::event mev = q.submit([&](sycl::handler& h) {
                    h.depends_on(peer_push);
                    h.parallel_for(sycl::range<1>(uint64_t(Tv) * NU * H),
                                   [=](sycl::id<1> i) {
                        const uint32_t cell = uint32_t(i / H_);
                        if (!own[cell]) cells[i] = sc2[i];
                    });
                });
                slots[c][L].merge_ev = mev;
                slots[c][L].seq_merge.store(r, std::memory_order_release);
                if (c == 0) ms_exch += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count();
                return;
            }
            if (exch_host) {
                // Own cells done -> stage for the peer; pull after barrier.
                q.memcpy(stage[c][pb], self->verify_cells(),
                         uint64_t(Tv) * NU * H * 2);
                q.wait();
                bar.arrive_and_wait();
                q.memcpy(scr[c][pb], stage[1 - c][pb],
                         uint64_t(Tv) * NU * H * 2);
            } else {
                // PUSH own cells into the peer's parity slot; the wait
                // publishes them before both sides pass the barrier.
                q.memcpy(scr[1 - c][pb], self->verify_cells(),
                         uint64_t(Tv) * NU * H * 2);
                q.wait();
                bar.arrive_and_wait();
            }
            q.memcpy(d_own[c], self->tp_cell_owner_.data(), uint64_t(Tv) * NU);
            {
                sycl::half* cells = self->verify_cells();
                const sycl::half* sc2 = scr[c][pb];
                const uint8_t* own = d_own[c];
                const uint32_t H_ = H;
                q.parallel_for(sycl::range<1>(uint64_t(Tv) * NU * H), [=](sycl::id<1> i) {
                    const uint32_t cell = uint32_t(i / H_);
                    if (!own[cell]) cells[i] = sc2[i];
                });
            }
            if (c == 0) ms_exch += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
        };
    };
    A.set_tp_experts(0, EHALF, make_sync(0, &A, &B));
    B.set_tp_experts(EHALF, cfg.n_experts, make_sync(1, &B, &A));

    // ---- spec loop: replicated verify, expert-split MoE -------------------
    std::vector<int32_t> spec;
    uint32_t rounds = 0, accepted_total = 0;
    {
        A.reset_state(); B.reset_state();
        uint32_t pos = 0;
        while (pos < P0) {
            const uint32_t n = std::min(1024u, P0 - pos);
            auto af = std::async(std::launch::async, [&, n, pos]() {
                return A.forward(ids.data() + pos, n, pos, nullptr);
            });
            if (auto e = B.forward(ids.data() + pos, n, pos, nullptr); !e.empty()) {
                std::fprintf(stderr, "prefill B: %s\n", e.c_str()); return 1;
            }
            if (auto e = af.get(); !e.empty()) { std::fprintf(stderr, "prefill A: %s\n", e.c_str()); return 1; }
            if (auto e = B.mtp_ingest(ids.data() + pos, n, pos); !e.empty()) {
                std::fprintf(stderr, "ingest: %s\n", e.c_str()); return 1;
            }
            pos += n;
        }
        int32_t t = argmaxB(B.logits(), V, d_amB);
        A.set_spec_verify(true);
        B.set_spec_verify(true);
        double ms_draft = 0, ms_verify = 0, ms_rest = 0;
        auto lap = [last = std::chrono::steady_clock::now()](double& acc) mutable {
            const auto now = std::chrono::steady_clock::now();
            acc += std::chrono::duration<double, std::milli>(now - last).count();
            last = now;
        };
        std::vector<int32_t> draft(K), batch(K + 1);
        auto* d_lrows = static_cast<sycl::half*>(fleet.dev(1).malloc(uint64_t(Tv) * V * 2));
        const auto t0 = std::chrono::steady_clock::now();
        while (spec.size() < n_gen) {
            if (auto e = B.mtp_draft(t, pos, K, draft.data()); !e.empty()) {
                std::fprintf(stderr, "draft: %s\n", e.c_str()); return 1;
            }
            lap(ms_draft);
            batch[0] = t;
            for (uint32_t k = 0; k < K; ++k) batch[k + 1] = draft[k];
            if (!exch_bulk) {
                // Round boundary for the event-chained exchange: one drain
                // per ROUND (not per layer) retires every scr reader/writer
                // of the previous round before its slots are reused.
                qA.wait(); qB.wait();
                round_id.store(round_id.load(std::memory_order_relaxed) + 1,
                               std::memory_order_release);
            }
            auto af = std::async(std::launch::async, [&]() {
                return A.forward(batch.data(), K + 1, pos, nullptr);
            });
            const std::string eb = B.forward(batch.data(), K + 1, pos, nullptr);
            const std::string ea = af.get();
            if (!ea.empty()) { std::fprintf(stderr, "verify A: %s\n", ea.c_str()); return 1; }
            if (!eb.empty()) { std::fprintf(stderr, "verify B: %s\n", eb.c_str()); return 1; }
            lap(ms_verify);
            if (auto e = B.logits_rows(K + 1, d_lrows); !e.empty()) {
                std::fprintf(stderr, "rows: %s\n", e.c_str()); return 1;
            }
            std::vector<int32_t> a(K + 1);
            for (uint32_t i = 0; i <= K; ++i)
                sample_argmax(qB, d_lrows + uint64_t(i) * V, d_amB + i, V);
            qB.memcpy(a.data(), d_amB, (K + 1) * sizeof(int32_t)).wait();
            uint32_t n_ok = 0;
            while (n_ok < K && draft[n_ok] == a[n_ok]) ++n_ok;
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
        std::printf("  EP spec K=%u : %.1f ms/token (%.2f tok/s), rounds %u, "
                    "mean accepted %.2f\n",
                    K, ms / spec.size(), spec.size() * 1000.0 / ms, rounds,
                    rounds ? double(accepted_total) / rounds : 0.0);
        if (rounds)
            std::printf("  round: draft %.1f  verify %.1f (exch %.1f)  rest %.1f ms\n",
                        ms_draft / rounds, ms_verify / rounds, ms_exch / rounds,
                        ms_rest / rounds);
    }

    uint32_t diff = 0;
    for (uint32_t i = 0; i < n_gen && i < spec.size() && i < ref.size(); ++i)
        if (spec[i] != ref[i]) ++diff;
    check(diff == 0 && spec.size() == ref.size(),
          "LOSSLESS: EP spec == replicated plain greedy");
    std::printf("  text: %s\n",
                tok.decode(std::span<const int32_t>(spec), true, {}).substr(0, 160).c_str());
    std::printf("\n\033[1m%s\033[0m\n\n", g_fail ? "\033[31mGATE FAILED\033[0m" : "\033[32mGATE PASSED\033[0m");
    return g_fail ? 1 : 0;
}
