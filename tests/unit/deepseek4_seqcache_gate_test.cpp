// tests/unit/deepseek4_seqcache_gate_test.cpp — the SEQUENCE-SLOT gate.
//
// WHAT DEFECT THIS TEST EXISTS FOR
// --------------------------------
// `Ds4Options::seq_slots` lets one loaded runtime hold N independent
// conversations' attention state.  The failure mode it introduces is the worst
// kind this engine can have: if two sequences share any part of a layer cache —
// the sliding ring, a compressor window buffer, an emitted-entry count, the
// CSA overlap slot — the model does not crash and does not produce NaNs.  It
// produces FLUENT, PLAUSIBLE TEXT conditioned on a stranger's conversation.
// Nothing downstream can detect that.  It has to be caught here.
//
// So this file gates exactly two claims, and gates them bitwise:
//
//   §1  ONE SLOT IS UNCHANGED.  A runtime loaded with seq_slots = 4 and driven
//       only on slot 0 produces logits BIT-IDENTICAL to the same runtime loaded
//       with seq_slots = 1.  This is what makes the feature safe to compile in:
//       the shipped single-conversation path is not merely "equivalent", it is
//       the same numbers.
//   §2  SLOTS CANNOT SEE EACH OTHER.  A sequence advanced ALONE and the SAME
//       sequence advanced INTERLEAVED with a different one on another slot
//       produce bit-identical logits at every step.  Checked in BOTH
//       directions, because a contamination that only flows one way is still
//       contamination.
//
// plus the two supporting facts a capacity model depends on:
//
//   §3  THE SCRATCH IS SHARED, THE STATE IS NOT.  Adding a slot must not add
//       any transient scratch (`cache_scratch_bytes()` is invariant to slot
//       traffic), while per-slot state must be genuinely per-slot
//       (`reset_seq(1)` leaves slot 0's context length and bytes alone).  The
//       measured per-slot and scratch figures are PRINTED, because
//       docs/deepseek4/60_CONTINUOUS_BATCHING.md's batch-size ceiling is
//       computed from exactly this ratio and a doc that quotes a number no test
//       prints is a doc nobody can check.
//   §4  OUT-OF-RANGE SLOTS ARE REFUSED, NOT CLAMPED.  A clamped slot id is two
//       conversations sharing one cache — the §2 failure, arrived at through
//       the front door.
//
// §0 IS A PRECONDITION, NOT A FEATURE.  Every check above is bitwise, and
// docs/deepseek4/50_STATUS_AND_NEXT.md:112 records that this engine is
// run-to-run NON-DETERMINISTIC at T > 1 through oneDNN's matmul partitioning.
// So the test forces `IE_DS4_ONEDNN=0` before anything loads, and then PROVES
// determinism inside this process before trusting a single bitwise comparison.
// If §0 fails, the rest of the file is not evidence and says so.
//
// NO MODEL FILE, NO ENV VAR REQUIRED, ONE GPU.  The fixture is `ds4mini` — a
// real GGUF with the real tensor contract at toy dimensions, carrying all three
// attention layer types (ratios 0 / 4 / 128), which is what makes the
// compressor and indexer window state machines actually run.  $DS4_GPU picks
// the device.

#include "ie/deepseek4.hpp"
#include "ie/gguf.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../../tools/deepseek4/ds4_mini_gguf.hpp"

namespace {

int g_fail = 0;

void ok(const std::string& what, bool cond, const std::string& why = {}) {
    std::printf("[%s] %s%s%s\n", cond ? " OK " : "FAIL", what.c_str(),
                cond || why.empty() ? "" : " — ", cond ? "" : why.c_str());
    if (!cond) ++g_fail;
}

// A run's whole logit history: one [vocab] row per forward call.  Compared with
// memcmp, never with a tolerance — the claim under test is bit-identity, and a
// tolerance would pass exactly the contamination this file exists to catch
// (a stranger's context perturbs logits by far less than a loose epsilon).
struct Trace {
    std::vector<std::vector<float>> rows;
    bool                            ok = false;
    std::string                     err;
};

// The first row index at which two traces differ, or -1.  Reported rather than
// a bare bool so a failure names WHEN divergence started, which is the
// difference between "slot state leaked from step 0" and "leaked once the
// compressor emitted its first window".
long first_diff(const Trace& a, const Trace& b) {
    if (a.rows.size() != b.rows.size()) return 0;
    for (size_t i = 0; i < a.rows.size(); ++i) {
        if (a.rows[i].size() != b.rows[i].size()) return long(i);
        if (std::memcmp(a.rows[i].data(), b.rows[i].data(),
                        a.rows[i].size() * sizeof(float)) != 0)
            return long(i);
    }
    return -1;
}

// One sequence's script: a prompt to prefill, then a fixed number of decode
// steps.  The decoded token is TAKEN FROM THE SCRIPT, not from the logits —
// feeding back an argmax would make the two streams converge on the same tokens
// and quietly weaken §2 into a much easier test.
struct Script {
    std::vector<int32_t> prompt;
    std::vector<int32_t> decode;
};

Script make_script(uint32_t vocab, uint32_t prompt_len, uint32_t steps, uint32_t seed) {
    Script s;
    // Distinct, deterministic, and NOT a rotation of each other: two scripts
    // that differ only by an offset would still exercise the same window
    // alignment, and window alignment is where the compressor state machines
    // live.
    uint32_t x = seed * 2654435761u + 12345u;
    auto nxt = [&] { x = x * 1664525u + 1013904223u; return int32_t(x % vocab); };
    for (uint32_t i = 0; i < prompt_len; ++i) s.prompt.push_back(nxt());
    for (uint32_t i = 0; i < steps; ++i)      s.decode.push_back(nxt());
    return s;
}

}  // namespace

int main() {
    // BEFORE ANY LOAD.  ds4 reads this with getenv at load time, and every
    // comparison below is bitwise (see the §0 note in the header).
    ::setenv("IE_DS4_ONEDNN", "0", 1);

    std::printf("=== deepseek4 sequence-slot gate ===\n");

    uint32_t gpu = 0;
    if (const char* g = std::getenv("DS4_GPU")) gpu = uint32_t(std::max(0, std::atoi(g)));

    ds4mini::Cfg mc;
    mc.L         = 6;         // ratios {0,0,4,128,4,128}: all three layer kinds
    mc.VOCAB     = 512;
    mc.iq3_scale = 5e-4f;     // away from the SwiGLU clamp; see ds4mini::Cfg
    const std::string dir  = std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp";
    const std::string path = dir + "/ds4_seqcache_gate.gguf";
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

    // PROMPT_LEN and STEPS are chosen against the fixture's compress ratios, not
    // arbitrarily: 20 prompt tokens is 5 full windows of the ratio-4 CSA layers
    // with no remainder, and 12 decode steps then walk the window boundary three
    // more times AND leave a partial window in the buffer at the end.  A script
    // that never crossed a window boundary would never touch
    // update_compressor_states, update_overlap_state, or the buffer carry —
    // exactly the state that is now per-slot.
    const uint32_t PROMPT_LEN = 20, STEPS = 12;
    const uint32_t V = cfg.vocab;

    auto base_opts = [&](uint32_t slots) {
        ie::Ds4Options o;
        o.device_ordinal     = gpu;
        o.max_seq            = 64;
        o.max_context        = 256;
        o.slots_per_layer    = 4;
        o.expert_cache_bytes = 32ull << 20;
        o.host_pin_cap_bytes = 1ull << 30;
        o.prefetch           = false;   // speculation is timing-dependent
        o.seq_slots          = slots;
        return o;
    };

    // Drives one script on one slot of an already-loaded runtime, appending the
    // logits of every forward to `t`.  Does NOT reset the slot — the caller
    // decides, because "what happens when a slot is NOT reset" is part of §2.
    auto step_prefill = [&](ie::DeepSeek4Runtime& rt, const Script& s, Trace& t) {
        std::vector<float> lg(V, 0.f);
        std::string e = rt.forward(s.prompt.data(), uint32_t(s.prompt.size()), 0, lg.data());
        if (!e.empty()) { t.ok = false; t.err = e; return false; }
        t.rows.push_back(lg);
        return true;
    };
    auto step_decode = [&](ie::DeepSeek4Runtime& rt, const Script& s, uint32_t i, Trace& t) {
        std::vector<float> lg(V, 0.f);
        const uint32_t pos = uint32_t(s.prompt.size()) + i;
        std::string e = rt.forward(&s.decode[i], 1, pos, lg.data());
        if (!e.empty()) { t.ok = false; t.err = e; return false; }
        t.rows.push_back(lg);
        return true;
    };
    // The whole script, start to finish, on the CURRENTLY SELECTED slot.
    auto run_alone = [&](ie::DeepSeek4Runtime& rt, const Script& s) {
        Trace t; t.ok = true;
        if (!step_prefill(rt, s, t)) return t;
        for (uint32_t i = 0; i < STEPS; ++i)
            if (!step_decode(rt, s, i, t)) return t;
        return t;
    };

    const Script A = make_script(V, PROMPT_LEN, STEPS, 1);
    const Script B = make_script(V, PROMPT_LEN, STEPS, 7);
    {
        bool differ = A.prompt != B.prompt && A.decode != B.decode;
        ok("the two scripts are genuinely different token streams", differ);
    }

    // =======================================================================
    // §0 — the precondition: this process is deterministic
    // =======================================================================
    std::printf("\n--- 0: determinism precondition (IE_DS4_ONEDNN=0) ---\n");
    bool deterministic = false;
    Trace one_slot_A;   // §1 compares against this
    {
        ie::DeepSeek4Runtime rt;
        const std::string le = rt.load(model, base_opts(1));
        ok("load with seq_slots = 1", le.empty(), le);
        if (le.empty()) {
            ok("a default load reports exactly one sequence slot", rt.seq_slots() == 1,
               "seq_slots() = " + std::to_string(rt.seq_slots()));
            rt.reset_context();
            one_slot_A = run_alone(rt, A);
            ok("script A on the one-slot runtime", one_slot_A.ok, one_slot_A.err);
            rt.reset_context();
            Trace again = run_alone(rt, A);
            ok("script A again after reset_context()", again.ok, again.err);
            if (one_slot_A.ok && again.ok) {
                const long d = first_diff(one_slot_A, again);
                deterministic = (d < 0);
                ok("the same script twice in one process is BIT-IDENTICAL", deterministic,
                   "diverged at forward #" + std::to_string(d) +
                   " — every bitwise check below is therefore inconclusive, not failing;"
                   " see docs/deepseek4/50_STATUS_AND_NEXT.md:112");
            }
        }
    }

    // =======================================================================
    // §1 — one slot is unchanged
    // =======================================================================
    std::printf("\n--- 1: seq_slots = 4 driven on slot 0 == seq_slots = 1 ---\n");
    {
        ie::DeepSeek4Runtime rt;
        const std::string le = rt.load(model, base_opts(4));
        ok("load with seq_slots = 4", le.empty(), le);
        if (le.empty()) {
            ok("the runtime reports four sequence slots", rt.seq_slots() == 4,
               "seq_slots() = " + std::to_string(rt.seq_slots()));
            ok("slot 0 is active after load", rt.active_seq() == 0);
            rt.reset_context();
            Trace four_slot_A = run_alone(rt, A);
            ok("script A on slot 0 of the four-slot runtime", four_slot_A.ok, four_slot_A.err);
            if (deterministic && one_slot_A.ok && four_slot_A.ok) {
                const long d = first_diff(one_slot_A, four_slot_A);
                ok("BIT-IDENTICAL to the one-slot runtime", d < 0,
                   "diverged at forward #" + std::to_string(d));
            } else {
                std::printf("[SKIP] §1 bitwise comparison — §0 did not establish determinism\n");
            }
        }
    }

    // =======================================================================
    // §2 — slots cannot see each other
    // =======================================================================
    std::printf("\n--- 2: interleaved slots are isolated ---\n");
    {
        ie::DeepSeek4Runtime rt;
        const std::string le = rt.load(model, base_opts(2));
        ok("load with seq_slots = 2", le.empty(), le);
        if (le.empty()) {
            // Each script alone, on its own slot, with the other slot idle and
            // empty.  These are the references.
            rt.reset_context();
            (void)rt.select_seq(0);
            Trace A_alone = run_alone(rt, A);
            ok("script A alone on slot 0", A_alone.ok, A_alone.err);
            rt.reset_context();
            (void)rt.select_seq(1);
            Trace B_alone = run_alone(rt, B);
            ok("script B alone on slot 1", B_alone.ok, B_alone.err);

            // Now INTERLEAVED, one forward at a time, alternating slots.  This is
            // the shape a round-robin scheduler produces, and it is the shape
            // that shares every workspace buffer between the two sequences while
            // their cache state must stay apart.
            rt.reset_context();
            Trace A_int, B_int;
            A_int.ok = B_int.ok = true;
            bool drove = true;
            if (drove) { drove = (rt.select_seq(0).empty()) && step_prefill(rt, A, A_int); }
            if (drove) { drove = (rt.select_seq(1).empty()) && step_prefill(rt, B, B_int); }
            for (uint32_t i = 0; i < STEPS && drove; ++i) {
                drove = rt.select_seq(0).empty() && step_decode(rt, A, i, A_int);
                if (drove) drove = rt.select_seq(1).empty() && step_decode(rt, B, i, B_int);
            }
            ok("interleaved drive completed", drove && A_int.ok && B_int.ok,
               A_int.err.empty() ? B_int.err : A_int.err);

            if (deterministic && A_alone.ok && A_int.ok && B_alone.ok && B_int.ok && drove) {
                const long da = first_diff(A_alone, A_int);
                ok("slot 0 is BIT-IDENTICAL whether or not slot 1 is running", da < 0,
                   "diverged at forward #" + std::to_string(da) +
                   " — sequence B's state reached sequence A");
                const long db = first_diff(B_alone, B_int);
                ok("slot 1 is BIT-IDENTICAL whether or not slot 0 is running", db < 0,
                   "diverged at forward #" + std::to_string(db) +
                   " — sequence A's state reached sequence B");
                // The two streams must not accidentally agree, or the two checks
                // above would pass on a runtime that ignored the slot id entirely.
                ok("A and B produce DIFFERENT logits (the checks above discriminate)",
                   first_diff(A_alone, B_alone) >= 0);
            } else {
                std::printf("[SKIP] §2 bitwise comparison — §0 did not establish determinism\n");
            }
        }
    }

    // =======================================================================
    // §3 — the scratch is shared, the state is not
    // =======================================================================
    std::printf("\n--- 3: shared scratch / per-slot state accounting ---\n");
    {
        ie::DeepSeek4Runtime rt;
        const std::string le = rt.load(model, base_opts(3));
        ok("load with seq_slots = 3", le.empty(), le);
        if (le.empty()) {
            rt.reset_context();
            (void)rt.select_seq(0);
            Trace ta = run_alone(rt, A);
            ok("script A on slot 0", ta.ok, ta.err);

            const uint64_t scratch_after_one = rt.cache_scratch_bytes();
            const uint64_t slot0_state = rt.seq_state_bytes(0);
            const uint64_t slot0_alloc = rt.seq_allocated_bytes(0);
            const uint64_t ctx0        = rt.context_length();
            ok("slot 1 is untouched while slot 0 ran", rt.seq_state_bytes(1) == 0,
               "slot 1 state = " + std::to_string(rt.seq_state_bytes(1)) + " B");

            (void)rt.select_seq(1);
            Trace tb = run_alone(rt, B);
            ok("script B on slot 1", tb.ok, tb.err);
            const uint64_t scratch_after_two = rt.cache_scratch_bytes();

            ok("running a SECOND slot adds NO transient scratch",
               scratch_after_two == scratch_after_one,
               std::to_string(scratch_after_one) + " -> " + std::to_string(scratch_after_two) + " B");

            (void)rt.select_seq(0);
            ok("slot 0's context length survived slot 1's traffic", rt.context_length() == ctx0,
               std::to_string(ctx0) + " -> " + std::to_string(rt.context_length()));
            ok("slot 0's live state survived slot 1's traffic",
               rt.seq_state_bytes(0) == slot0_state);
            ok("slot 0's allocation survived slot 1's traffic",
               rt.seq_allocated_bytes(0) == slot0_alloc);

            // Resetting ONE slot must clear that slot and nothing else.
            ok("reset_seq(1) is accepted", rt.reset_seq(1).empty());
            ok("reset_seq(1) cleared slot 1", rt.seq_state_bytes(1) == 0);
            ok("reset_seq(1) left slot 0 alone", rt.seq_state_bytes(0) == slot0_state &&
                                                 rt.context_length() == ctx0);

            // PRINTED, not asserted against a magic constant: these are the two
            // numbers docs/deepseek4/60_CONTINUOUS_BATCHING.md's ceiling is built
            // from, and the fixture's toy dimensions make their absolute values
            // meaningless — their SEPARATION is the point.
            std::printf("       per-slot state  %llu B, per-slot allocated %llu B,"
                        " SHARED scratch %llu B (over %u layers, max_seq %u)\n",
                        (unsigned long long)slot0_state, (unsigned long long)slot0_alloc,
                        (unsigned long long)scratch_after_two, mc.L, 64u);
        }
    }

    // =======================================================================
    // §4 — out-of-range slots are refused, not clamped
    // =======================================================================
    std::printf("\n--- 4: an out-of-range slot is refused ---\n");
    {
        ie::DeepSeek4Runtime rt;
        const std::string le = rt.load(model, base_opts(2));
        ok("load with seq_slots = 2", le.empty(), le);
        if (le.empty()) {
            ok("select_seq(2) is refused", !rt.select_seq(2).empty());
            ok("a refused select_seq did NOT move the active slot", rt.active_seq() == 0,
               "active = " + std::to_string(rt.active_seq()));
            ok("reset_seq(2) is refused", !rt.reset_seq(2).empty());
            ok("select_seq(1) is accepted", rt.select_seq(1).empty());
            ok("an accepted select_seq DID move the active slot", rt.active_seq() == 1);
            ok("seq_state_bytes on an out-of-range slot is 0, not a read past the end",
               rt.seq_state_bytes(9) == 0 && rt.seq_allocated_bytes(9) == 0);
        }
    }

    std::remove(path.c_str());
    std::printf("\n=== %s (%d failure%s) ===\n", g_fail ? "FAIL" : "PASS", g_fail,
                g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
