// tests/unit/deepseek4_sched_gate_test.cpp — the DECODE SCHEDULE gate.
//
// WHAT DEFECT THIS TEST EXISTS FOR
// --------------------------------
// A two-card decode step on the real model walls at ~75 ms while the two cards
// between them execute ~68 ms of kernel and ~36 ms of DMA.  If compute and
// transfer overlapped and the cards ran concurrently, that would be a ~36 ms
// step.  The difference is HOST time — submission, barriers, cross-card
// handshakes — and it was invisible, because a decode step that is GPU-bound
// and one that is submission-bound produce the same wall and the same kernel
// profile.  The only thing that separates them is where the HOST was.
//
// `Ds4TpForwardTrace` now answers that, and this file is what makes the answer
// trustworthy.  An instrument that reports a decomposition nobody checks is
// worse than no instrument: it will be believed.
//
// WHAT IT CHECKS, AND WITH WHAT EVIDENCE
// --------------------------------------
//   §1  THE TILING.  prologue, span, reduce and epilogue are DISJOINT
//       sub-intervals of [forward entry, forward exit], so their sum cannot
//       exceed the wall and the residual `gap` cannot be negative.  Likewise a
//       card's segment interval is contained in that segment's span, and its
//       barrier waits are contained in its segment.  These are set-containment
//       facts, not tolerances — the bound below is pure floating-point
//       accumulation error and is derived as such.  Checked on every forward of
//       a real (tiny) two-card run, and then on hand-built traces that violate
//       each invariant by a known margin, which is what proves the checker
//       discriminates rather than always passing.
//   §2  THE BARRIER CENSUS.  The schedule says one router D2H and one drain per
//       layer per card, and one extra epilogue drain on card 0 only.  Those are
//       exact integers, and they are what a stray `q.wait()` changes.  A count
//       that drifts is a host round trip somebody added without saying so.
//   §3  THE LOGITS BUFFER IS SIZED BY USE.  `ws_logits_` used to be allocated
//       [max_seq, vocab] fp32 to have one row written into it — 529 MB/card at
//       --max-seq 1024, VRAM the expert cache could have had.  §3 gates that it
//       is one row after load, that a `last_only=false` caller GROWS it rather
//       than overrunning it, and that the grown path returns the same logits.
//
// NO MODEL FILE, NO ENV VAR, ONE GPU.  The fixture is `ds4mini` — a real GGUF
// with the real tensor contract at toy dimensions — and the two "cards" land on
// the same device via `Ds4TpOptions::same_device_rehearsal` unless $DS4_TP_GPUS
// names two.

#include "ie/deepseek4.hpp"
#include "ie/gguf.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../../tools/deepseek4/ds4_mini_gguf.hpp"

namespace {

int g_fail = 0;

void ok(const char* what, bool cond, const std::string& detail = {}) {
    std::printf("%s %s", cond ? "[ ok ]" : "[FAIL]", what);
    if (!detail.empty()) std::printf("  (%s)", detail.c_str());
    std::printf("\n");
    if (!cond) ++g_fail;
}

std::string ms(double s) {
    char b[48];
    std::snprintf(b, sizeof(b), "%.6f ms", s * 1e3);
    return b;
}

// ---------------------------------------------------------------------------
// §1's checker, as a function over a trace so §1 can also run it on traces it
// BUILT to be wrong.  Returns "" when the trace tiles, else the first
// invariant it violates and by how much.
//
// THE BOUND.  Every comparison below is a set-containment fact about intervals
// measured with the same clock: [prologue], each segment's span, each reduction
// and [epilogue] are disjoint and all lie inside [wall0, wall1], so the exact
// arithmetic answer is "zero slack" and no tolerance is needed for the physics.
// What forces a nonzero bound is only that the sums are accumulated in double:
// `span` is a sum of `3 * n_layers` differences, each O(1e-4 s), and IEEE-754
// double carries 2^-53 relative error per operation, so the accumulated
// absolute error over N additions of magnitude M is bounded by N * M * 2^-53.
// For the fixture below (N = 3 * 6 = 18 terms, M <= 1e-1 s) that is
// 18 * 1e-1 * 1.11e-16 = 2.0e-16 s.  The bound used is 1e-9 s — seven orders of
// magnitude above the error model and six below the smallest interval it
// compares — so a real violation cannot hide under it and rounding cannot trip
// it.  It is NOT tuned: the discrimination control at the end of §1 pins it.
constexpr double kTileTol = 1e-9;

std::string check_tiling(const ie::Ds4TpForwardTrace& f, double tol) {
    const double parts =
        f.prologue_seconds + f.span_seconds + f.reduce_seconds + f.epilogue_seconds;
    if (parts > f.wall_seconds + tol)
        return "prologue+span+reduce+epilogue (" + ms(parts) + ") exceeds wall (" +
               ms(f.wall_seconds) + ") by " + ms(parts - f.wall_seconds);
    if (f.span_seconds < -tol) return "span is negative: " + ms(f.span_seconds);
    if (f.reduce_seconds < -tol) return "reduce is negative: " + ms(f.reduce_seconds);
    double phase = 0.0;
    for (double p : f.phase_span) phase += p;
    if (std::fabs(phase - f.span_seconds) > tol)
        return "the three phase spans (" + ms(phase) + ") do not sum to span (" +
               ms(f.span_seconds) + ")";
    for (size_t c = 0; c < f.card_busy.size(); ++c) {
        if (f.card_busy[c] > f.span_seconds + tol)
            return "card " + std::to_string(c) + " busy (" + ms(f.card_busy[c]) +
                   ") exceeds the span it ran inside (" + ms(f.span_seconds) + ")";
        if (c < f.card_waits.size() && f.card_waits[c].total() > f.card_busy[c] + tol)
            return "card " + std::to_string(c) + " waited (" + ms(f.card_waits[c].total()) +
                   ") longer than it was busy (" + ms(f.card_busy[c]) + ")";
    }
    if (f.overlap_seconds > f.span_seconds + tol)
        return "overlap (" + ms(f.overlap_seconds) + ") exceeds span (" + ms(f.span_seconds) + ")";
    if (f.overlapped > f.total)
        return "overlapped segments (" + std::to_string(f.overlapped) + ") exceeds total (" +
               std::to_string(f.total) + ")";
    return {};
}

int32_t argmax(const std::vector<float>& v) {
    return int32_t(std::max_element(v.begin(), v.end()) - v.begin());
}

}  // namespace

int main() {
    std::printf("=== deepseek4_sched_gate_test — the decode schedule ===\n");

    std::vector<sycl::device> gpus;
    for (const auto& d : sycl::device::get_devices())
        if (d.is_gpu() && d.get_backend() == sycl::backend::ext_oneapi_level_zero)
            gpus.push_back(d);
    if (gpus.empty()) {
        std::printf("[FAIL] no Level Zero GPU — this gate needs one and will not pretend"
                    " it passed without it\n");
        return 1;
    }
    std::printf("device: %s\n", gpus[0].get_info<sycl::info::device::name>().c_str());

    // Two "cards" on one device unless the caller names two.
    std::vector<uint32_t> tp_gpus{0u, 0u};
    bool two_device = false;
    if (const char* s = std::getenv("DS4_TP_GPUS")) {
        int a = -1, b = -1;
        if (std::sscanf(s, "%d,%d", &a, &b) == 2 && a >= 0 && b >= 0 &&
            size_t(a) < gpus.size() && size_t(b) < gpus.size() && a != b) {
            tp_gpus = {uint32_t(a), uint32_t(b)};
            two_device = true;
        }
    }
    std::printf("cards: %u,%u (%s)\n", tp_gpus[0], tp_gpus[1],
                two_device ? "two devices" : "same-device rehearsal");

    // ---- the fixture -------------------------------------------------------
    // VOCAB is deliberately LARGE relative to everything else: §3 measures a
    // logits allocation, and a toy vocabulary would make the buffer it is about
    // smaller than every other workspace in the runtime.
    ds4mini::Cfg mc;
    mc.L         = 6;
    mc.VOCAB     = 32768;
    mc.iq3_scale = 5e-4f;   // away from the SwiGLU clamp; see ds4mini::Cfg
    const std::string dir  = std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp";
    const std::string path = dir + "/ds4_sched_gate.gguf";
    std::vector<std::vector<uint8_t>> keep;
    if (std::string e = ds4mini::write_gguf(mc, path, keep); !e.empty()) {
        std::printf("[FAIL] could not write the fixture: %s\n", e.c_str());
        return 1;
    }

    ie::GgufReader rd;
    if (std::string e = rd.open(path); !e.empty()) {
        std::printf("[FAIL] could not open the fixture: %s\n", e.c_str());
        std::remove(path.c_str());
        return 1;
    }
    ie::DeepSeek4Config cfg;
    if (std::string e = ie::read_deepseek4_config(rd, cfg); !e.empty()) {
        std::printf("[FAIL] config: %s\n", e.c_str());
        std::remove(path.c_str());
        return 1;
    }
    ie::DeepSeek4Model model;
    if (std::string e = model.load(rd, cfg); !e.empty()) {
        std::printf("[FAIL] model bind: %s\n", e.c_str());
        std::remove(path.c_str());
        return 1;
    }

    const uint32_t kMaxSeq = 64;
    auto base_opts = [&](uint32_t ord) {
        ie::Ds4Options o;
        o.device_ordinal     = ord;
        o.max_seq            = kMaxSeq;
        o.max_context        = 256;
        o.slots_per_layer    = 4;
        o.expert_cache_bytes = 32ull << 20;
        o.host_pin_cap_bytes = 1ull << 30;
        o.prefetch           = false;   // speculation is timing-dependent
        return o;
    };

    // =======================================================================
    // §1 + §2 — the tiling and the barrier census, on a real two-card drive
    // =======================================================================
    std::printf("\n--- 1: the wall-clock tiling / 2: the barrier census ---\n");
    {
        ie::Ds4TpOptions to;
        to.base                  = base_opts(tp_gpus[0]);
        to.n_cards               = 2;
        to.device_ordinals       = tp_gpus;
        to.same_device_rehearsal = !two_device;
        ie::DeepSeek4TpRuntime tp;
        const std::string le = tp.load(model, to);
        ok("two-card orchestrator load", le.empty(), le);

        if (le.empty()) {
            const uint32_t PT = 16, STEPS = 4;
            std::vector<int32_t> prompt(PT);
            for (uint32_t i = 0; i < PT; ++i) prompt[i] = int32_t(i % cfg.vocab);
            std::vector<float> lg(cfg.vocab, 0.f);

            // The forwards: one prefill and STEPS decode steps.  EVERY one is
            // checked, prefill included — the tiling is a property of the
            // decomposition, not of the shape being decomposed.
            struct Snap { uint32_t T; ie::Ds4TpForwardTrace f; };
            std::vector<Snap> snaps;
            std::string ferr = tp.forward(prompt.data(), PT, 0, lg.data());
            ok("prefill forward", ferr.empty(), ferr);
            if (ferr.empty()) snaps.push_back({PT, tp.forward_trace()});
            int32_t tok = argmax(lg);
            for (uint32_t s = 0; s < STEPS && ferr.empty(); ++s) {
                ferr = tp.forward(&tok, 1, PT + s, lg.data());
                if (!ferr.empty()) { ok("decode forward", false, ferr); break; }
                snaps.push_back({1u, tp.forward_trace()});
                tok = argmax(lg);
            }
            ok("prefill + every decode step completed", snaps.size() == size_t(STEPS) + 1,
               std::to_string(snaps.size()) + " of " + std::to_string(STEPS + 1));

            // ---- 1: on every real trace ----
            bool all_tile = true;
            for (const Snap& sn : snaps) {
                const std::string why = check_tiling(sn.f, kTileTol);
                if (!why.empty()) {
                    all_tile = false;
                    ok("a real trace tiles its wall", false,
                       "T=" + std::to_string(sn.T) + ": " + why);
                }
            }
            ok(("every forward's trace tiles its own wall (bound " + ms(kTileTol) +
                ", derived from double accumulation over the segment sums)").c_str(),
               all_tile);

            // The decomposition, printed.  Not gated — it is the measurement the
            // gate exists to make trustworthy, and its VALUE is the real model's,
            // not this fixture's.
            for (const Snap& sn : snaps) {
                const ie::Ds4TpForwardTrace& f = sn.f;
                const double gap = f.wall_seconds - f.prologue_seconds - f.span_seconds -
                                   f.reduce_seconds - f.epilogue_seconds;
                std::printf("       T=%2u wall=%8.4f ms | pro=%.4f span=%.4f red=%.4f/%llu"
                            " epi=%.4f gap=%.4f | ovl=%.4f in %llu/%llu seg\n",
                            sn.T, f.wall_seconds * 1e3, f.prologue_seconds * 1e3,
                            f.span_seconds * 1e3, f.reduce_seconds * 1e3,
                            (unsigned long long)f.reductions, f.epilogue_seconds * 1e3,
                            gap * 1e3, f.overlap_seconds * 1e3,
                            (unsigned long long)f.overlapped, (unsigned long long)f.total);
                for (size_t c = 0; c < f.card_busy.size() && c < f.card_waits.size(); ++c) {
                    const auto& w = f.card_waits[c];
                    std::printf("            c%zu busy=%.4f submit=%.4f wait=%.4f"
                                " (rt=%.4f/%llu moe=%.4f/%llu post=%.4f/%llu oth=%.4f/%llu)\n",
                                c, f.card_busy[c] * 1e3, (f.card_busy[c] - w.total()) * 1e3,
                                w.total() * 1e3, w.router * 1e3, (unsigned long long)w.n_router,
                                w.moe * 1e3, (unsigned long long)w.n_moe, w.post * 1e3,
                                (unsigned long long)w.n_post, w.other * 1e3,
                                (unsigned long long)w.n_other);
                }
            }

            // ---- 1's NEGATIVE CONTROLS ----
            // A checker that never fails is not a check.  Each control takes a
            // REAL trace and breaks exactly one invariant by a margin far above
            // the bound; each must be caught.  The last one moves a term by less
            // than the bound and must NOT be caught, which is what pins the
            // bound where the comment says it is rather than somewhere lower.
            if (!snaps.empty()) {
                const ie::Ds4TpForwardTrace& good = snaps.back().f;
                ok("control: the unmodified trace passes", check_tiling(good, kTileTol).empty(),
                   check_tiling(good, kTileTol));

                {   // parts exceed wall
                    ie::Ds4TpForwardTrace b = good;
                    b.span_seconds = b.wall_seconds + 1e-3;
                    ok("control: span > wall is REJECTED", !check_tiling(b, kTileTol).empty(),
                       check_tiling(b, kTileTol));
                }
                {   // phase spans no longer sum to span
                    ie::Ds4TpForwardTrace b = good;
                    b.phase_span[1] += 1e-3;
                    ok("control: phase spans not summing to span is REJECTED",
                       !check_tiling(b, kTileTol).empty(), check_tiling(b, kTileTol));
                }
                {   // a card busier than the span that contained it
                    ie::Ds4TpForwardTrace b = good;
                    if (!b.card_busy.empty()) b.card_busy[0] = b.span_seconds + 1e-3;
                    ok("control: card busy > span is REJECTED", !check_tiling(b, kTileTol).empty(),
                       check_tiling(b, kTileTol));
                }
                {   // a card that waited longer than it ran
                    ie::Ds4TpForwardTrace b = good;
                    if (!b.card_waits.empty() && !b.card_busy.empty())
                        b.card_waits[0].post = b.card_busy[0] + 1e-3;
                    ok("control: card wait > card busy is REJECTED",
                       !check_tiling(b, kTileTol).empty(), check_tiling(b, kTileTol));
                }
                {   // overlap larger than the span it was measured inside
                    ie::Ds4TpForwardTrace b = good;
                    b.overlap_seconds = b.span_seconds + 1e-3;
                    ok("control: overlap > span is REJECTED", !check_tiling(b, kTileTol).empty(),
                       check_tiling(b, kTileTol));
                }
                {   // THE DISCRIMINATION CHECK.  A perturbation an order of
                    // magnitude BELOW the bound must pass, or the bound is not
                    // where it was claimed to be and the controls above proved
                    // nothing about its placement.
                    ie::Ds4TpForwardTrace b = good;
                    b.phase_span[0] += kTileTol * 0.1;
                    ok("control: a perturbation 10x below the bound is ACCEPTED"
                       " (the bound is where the error model puts it)",
                       check_tiling(b, kTileTol).empty(), check_tiling(b, kTileTol));
                }
            }

            // ---- 2: the barrier census ----
            // These are EXACT integers read straight off the schedule:
            //   layer_forward_mid drains once for the router D2H       -> n_router
            //   layer_forward_post drains once at the end of the layer -> n_post
            // per layer, per card, per forward.  Nothing conditional touches
            // either.  A count that differs means a host round trip was added or
            // removed, which is the single most expensive thing that can change
            // in this file without changing a single number in the output.
            if (!snaps.empty()) {
                const ie::Ds4TpForwardTrace& f = snaps.back().f;
                bool router_exact = true, post_exact = true, moe_at_least = true;
                for (const auto& w : f.card_waits) {
                    router_exact &= (w.n_router == cfg.n_layers);
                    post_exact   &= (w.n_post == cfg.n_layers);
                    moe_at_least &= (w.n_moe >= cfg.n_layers);
                }
                std::string got;
                for (const auto& w : f.card_waits)
                    got += "rt=" + std::to_string(w.n_router) + " post=" + std::to_string(w.n_post) +
                           " moe=" + std::to_string(w.n_moe) +
                           " oth=" + std::to_string(w.n_other) + "; ";
                ok(("exactly one router D2H barrier per layer per card (" +
                    std::to_string(cfg.n_layers) + ")").c_str(), router_exact, got);
                ok(("exactly one end-of-layer drain per layer per card (" +
                    std::to_string(cfg.n_layers) + ")").c_str(), post_exact, got);
                ok("at least one expert-group barrier per layer per card", moe_at_least, got);
                // Card 0 runs the epilogue and the others do not, so card 0 owes
                // exactly one more `other` barrier than every sibling.  This is
                // the assertion that would catch the epilogue silently running
                // on every card — which would be correct output at n_cards times
                // the lm_head cost, i.e. invisible except here.
                if (f.card_waits.size() >= 2) {
                    bool epi_only_c0 = true;
                    for (size_t c = 1; c < f.card_waits.size(); ++c)
                        epi_only_c0 &= (f.card_waits[0].n_other == f.card_waits[c].n_other + 1);
                    ok("card 0 owes exactly one more `other` barrier than each sibling"
                       " — the epilogue runs on card 0 ALONE", epi_only_c0, got);
                }
            }
        }
    }

    // =======================================================================
    // §3 — the logits buffer is sized by what is used
    // =======================================================================
    std::printf("\n--- 3: ws_logits_ is sized by use, not by max_seq ---\n");
    {
        ie::DeepSeek4Runtime rt;
        const std::string le = rt.load(model, base_opts(tp_gpus[0]));
        ok("single-card load", le.empty(), le);
        if (le.empty()) {
            const uint64_t saved = uint64_t(kMaxSeq - 1) * cfg.vocab * 4ull;
            ok(("after load the logits buffer holds ONE row, not max_seq (" +
                std::to_string(kMaxSeq) + ")").c_str(),
               rt.logits_rows() == 1,
               "rows = " + std::to_string(rt.logits_rows()) + ", VRAM not reserved = " +
                   std::to_string(saved) + " B");

            const uint32_t PT = 16;
            std::vector<int32_t> prompt(PT);
            for (uint32_t i = 0; i < PT; ++i) prompt[i] = int32_t(i % cfg.vocab);

            std::vector<float> last(cfg.vocab, 0.f);
            std::string e = rt.forward(prompt.data(), PT, 0, last.data(), /*last_only=*/true);
            ok("forward(last_only = true)", e.empty(), e);
            ok("a last_only forward does NOT grow the buffer", rt.logits_rows() == 1,
               "rows = " + std::to_string(rt.logits_rows()));

            // The grow path, and the proof it grows rather than overruns: the
            // same forward with last_only=false must return [T, vocab] whose
            // LAST ROW is bit-identical to the [1, vocab] above.  Same weights,
            // same inputs, same kernel — anything but bit-equality means the
            // wide path is not computing the same thing.
            rt.reset_context();
            std::vector<float> all(size_t(PT) * cfg.vocab, 0.f);
            e = rt.forward(prompt.data(), PT, 0, all.data(), /*last_only=*/false);
            ok("forward(last_only = false) after a 1-row allocation", e.empty(), e);
            if (e.empty()) {
                ok("the buffer GREW to T rows rather than overrunning", rt.logits_rows() == PT,
                   "rows = " + std::to_string(rt.logits_rows()) + ", T = " + std::to_string(PT));
                const float* lastrow = all.data() + size_t(PT - 1) * cfg.vocab;
                ok("its last row is BIT-IDENTICAL to the last_only=true logits",
                   std::memcmp(lastrow, last.data(), size_t(cfg.vocab) * 4) == 0);
            }
            // And it must not shrink back and re-grow on every call.
            rt.reset_context();
            std::vector<float> again(cfg.vocab, 0.f);
            e = rt.forward(prompt.data(), PT, 0, again.data(), /*last_only=*/true);
            ok("a later last_only forward still works against the grown buffer", e.empty(), e);
            ok("...and does not shrink it back", rt.logits_rows() == PT,
               "rows = " + std::to_string(rt.logits_rows()));
        }
    }

    std::remove(path.c_str());
    if (g_fail) {
        std::printf("\n%d CHECK(S) FAILED\n", g_fail);
        return 1;
    }
    std::printf("\nall checks passed\n");
    return 0;
}
