// tests/unit/deepseek4_promptcache_gate_test.cpp — the PROMPT-CACHE gate.
//
// WHAT DEFECT THIS TEST EXISTS FOR
// --------------------------------
// The DS4 prompt cache snapshots a conversation's attention state at a prefill
// boundary and restores it on the next turn instead of re-prefilling.  If the
// restore is wrong the model does not crash: it answers the new question while
// attending over a state that is subtly — or completely — not the one those
// tokens produced.  That reads as a slightly-off answer, or as a fluent answer
// to somebody else's conversation.  Nothing downstream can see it.
//
// Four ways this can go wrong, and this file gates all four bitwise:
//
//   §1  RESTORE MUST BE EXACT.  Restoring a snapshot taken at depth N and then
//       prefilling the suffix must be BIT-IDENTICAL to having prefilled [0..N)
//       and the suffix without any snapshot at all.  This is the whole claim.
//       Note the reference does the SAME two-part split at N — the engine
//       splits its prefill at the snapshot boundary whether the cache hits or
//       misses, so a comparison against a single unsplit prefill would be
//       testing a property the engine never relies on.
//   §2  THE SNAPSHOT MUST BE A COPY, NOT AN ALIAS.  Both the snapshot and the
//       live cache are device USM in the same allocations family.  Decoding 12
//       more steps AFTER the snapshot must leave the snapshot untouched: it
//       must still restore to depth N and still reproduce §1's continuation.
//       A snapshot that aliased `compressed_` or the sliding ring would drift
//       silently as generation continued.
//   §3  RESTORE MUST ACTUALLY RESTORE.  A restore that quietly did nothing
//       would pass §1 whenever the suffix dominated the logits.  So the same
//       suffix is also run against a snapshot of a DIFFERENT conversation, and
//       is required to DIFFER.  Without this check the file proves very little.
//   §4  THE FAILURE PATHS MUST BE HONEST.  `free_snapshot()` leaves depth 0 and
//       `restore_context()` returns 0 — never a stale or partial restore.  And
//       snapshot/restore act on the ACTIVE SLOT ONLY: restoring into slot 1
//       must not disturb slot 0, which is the property serving will depend on.
//
// §0 IS A PRECONDITION, NOT A FEATURE.  Every check here is bitwise, and
// docs/deepseek4/50_STATUS_AND_NEXT.md records that this engine is run-to-run
// NON-DETERMINISTIC at T > 1 through oneDNN's matmul partitioning.  So the test
// forces `IE_DS4_ONEDNN=0` before anything loads and PROVES determinism inside
// this process first.  If §0 fails, the rest is inconclusive and says so rather
// than reporting a false pass or a false failure.
//
// NO MODEL FILE, NO ENV VAR REQUIRED, ONE GPU.  The fixture is `ds4mini` — a
// real GGUF with the real tensor contract at toy dimensions, carrying all three
// attention layer kinds (ratios 0 / 4 / 128), which is what makes the
// compressor window buffers, the emitted-entry counters and the CSA overlap
// ping-pong — the parts of the state a naive snapshot forgets — actually run.
// $DS4_GPU picks the device.

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

// A run's logit history: one [vocab] row per forward call.  Compared with
// memcmp, never with a tolerance — the claim is bit-identity, and a tolerance
// would pass exactly the near-miss restore this file exists to catch.
struct Trace {
    std::vector<std::vector<float>> rows;
    bool                            ok = false;
    std::string                     err;
};

// Index of the first differing row between `a` from `a0` and `b` from `b0`, or
// -1 when every compared row matches.  Offsets exist because the uncached
// reference emits one extra row (the prefix prefill) that the cached run skips
// — that skipped forward IS the saving being tested.
long first_diff(const Trace& a, size_t a0, const Trace& b, size_t b0) {
    const size_t na = a.rows.size() - a0, nb = b.rows.size() - b0;
    if (na != nb) return 0;
    for (size_t i = 0; i < na; ++i) {
        const auto& ra = a.rows[a0 + i];
        const auto& rb = b.rows[b0 + i];
        if (ra.size() != rb.size()) return long(i);
        if (std::memcmp(ra.data(), rb.data(), ra.size() * sizeof(float)) != 0) return long(i);
    }
    return -1;
}

// Deterministic pseudo-random token streams.  Distinct seeds give streams that
// are not rotations of each other, so the two conversations in §3 land on
// different compressor window contents and not merely different offsets.
std::vector<int32_t> tokens(uint32_t vocab, uint32_t n, uint32_t seed) {
    std::vector<int32_t> v;
    uint32_t x = seed * 2654435761u + 12345u;
    for (uint32_t i = 0; i < n; ++i) {
        x = x * 1664525u + 1013904223u;
        v.push_back(int32_t(x % vocab));
    }
    return v;
}

}  // namespace

int main() {
    // BEFORE ANY LOAD — see the §0 note in the header.
    ::setenv("IE_DS4_ONEDNN", "0", 1);

    std::printf("=== deepseek4 prompt-cache gate ===\n");

    uint32_t gpu = 0;
    if (const char* g = std::getenv("DS4_GPU")) gpu = uint32_t(std::max(0, std::atoi(g)));

    ds4mini::Cfg mc;
    mc.L         = 6;         // ratios {0,0,4,128,4,128}: all three layer kinds
    mc.VOCAB     = 512;
    mc.iq3_scale = 5e-4f;     // away from the SwiGLU clamp; see ds4mini::Cfg
    const std::string dir  = std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp";
    const std::string path = dir + "/ds4_promptcache_gate.gguf";
    std::vector<std::vector<uint8_t>> keep;
    if (std::string e = ds4mini::write_gguf(mc, path, keep); !e.empty()) {
        std::printf("[FAIL] could not write the fixture: %s\n", e.c_str());
        return 1;
    }
    struct Unlink { const char* p; ~Unlink() { std::remove(p); } } unlink_guard{path.c_str()};

    ie::GgufReader rd;
    if (std::string e = rd.open(path); !e.empty()) {
        std::printf("[FAIL] could not open the fixture: %s\n", e.c_str());
        return 1;
    }
    ie::DeepSeek4Config cfg;
    if (std::string e = ie::read_deepseek4_config(rd, cfg); !e.empty()) {
        std::printf("[FAIL] config: %s\n", e.c_str());
        return 1;
    }
    ie::DeepSeek4Model model;
    if (std::string e = model.load(rd, cfg); !e.empty()) {
        std::printf("[FAIL] model bind: %s\n", e.c_str());
        return 1;
    }

    // N, S and STEPS are chosen against the fixture's compress ratios, not
    // arbitrarily.  N = 20 is five whole windows of the ratio-4 CSA layers with
    // NO remainder, so the snapshot is taken with an EMPTY window buffer and a
    // live overlap slice — the easy case.  S = 14 then leaves a 2-token
    // remainder in the buffer, and the decode steps walk three more boundaries,
    // so §2's later traffic genuinely mutates every piece of state the snapshot
    // had to copy.  A script that never crossed a boundary would leave
    // update_compressor_states, update_overlap_state and the buffer carry
    // untested — which is most of what makes this cache hard.
    const uint32_t N = 20, S = 14, STEPS = 12;
    const uint32_t V = cfg.vocab;

    ie::Ds4Options o;
    o.device_ordinal     = gpu;
    o.max_seq            = 64;
    o.max_context        = 256;
    o.slots_per_layer    = 4;
    o.expert_cache_bytes = 32ull << 20;
    o.host_pin_cap_bytes = 1ull << 30;
    o.prefetch           = false;   // speculation is timing-dependent
    o.seq_slots          = 2;       // §4 needs a second slot; §1-§3 use slot 0

    ie::DeepSeek4Runtime rt;
    if (std::string e = rt.load(model, o); !e.empty()) {
        std::printf("[FAIL] runtime load: %s\n", e.c_str());
        return 1;
    }
    ok("loaded with 2 sequence slots", rt.seq_slots() == 2);

    const std::vector<int32_t> prefixA = tokens(V, N, 1);
    const std::vector<int32_t> prefixB = tokens(V, N, 7);
    const std::vector<int32_t> suffix  = tokens(V, S, 3);
    const std::vector<int32_t> dec     = tokens(V, STEPS, 5);
    ok("the two conversations are genuinely different token streams", prefixA != prefixB);

    // Appends one forward's logits to `t`.  Returns false and latches the error
    // rather than throwing, so a runtime failure is reported as a failed check
    // and not as a crash with no diagnosis.
    auto fwd = [&](const int32_t* ids, uint32_t T, uint32_t pos, Trace& t) {
        std::vector<float> lg(V, 0.f);
        std::string e = rt.forward(ids, T, pos, lg.data());
        if (!e.empty()) { t.ok = false; t.err = e; return false; }
        t.rows.push_back(std::move(lg));
        return true;
    };
    // The suffix prefill plus every decode step, starting from whatever state
    // the cache is already in.  This is the part both arms of §1 share.
    auto run_tail = [&](Trace& t) {
        if (!fwd(suffix.data(), S, N, t)) return false;
        for (uint32_t i = 0; i < STEPS; ++i)
            if (!fwd(&dec[i], 1, N + S + i, t)) return false;
        return true;
    };

    // =======================================================================
    // §0 — the precondition: this process is deterministic
    // =======================================================================
    std::printf("\n--- 0: determinism precondition (IE_DS4_ONEDNN=0) ---\n");
    bool deterministic = false;
    Trace ref;            // §1's reference arm, built here so §0 reuses the work
    {
        ref.ok = true;
        rt.reset_context();
        if (fwd(prefixA.data(), N, 0, ref)) run_tail(ref);
        ok("uncached reference run", ref.ok, ref.err);

        Trace again; again.ok = true;
        rt.reset_context();
        if (fwd(prefixA.data(), N, 0, again)) run_tail(again);
        ok("the same run again after reset_context()", again.ok, again.err);
        if (ref.ok && again.ok) {
            const long d = first_diff(ref, 0, again, 0);
            deterministic = (d < 0);
            ok("the same script twice in one process is BIT-IDENTICAL", deterministic,
               "diverged at forward #" + std::to_string(d) +
               " — every bitwise check below is inconclusive, not failing;"
               " see docs/deepseek4/50_STATUS_AND_NEXT.md");
        }
    }
    const char* skip = " (SKIPPED: §0 failed)";

    // =======================================================================
    // §1 — restore is exact
    // =======================================================================
    std::printf("\n--- 1: snapshot at N, reset, restore, continue == never reset ---\n");
    Trace cached; cached.ok = true;
    {
        // Build the snapshot the way the engine does: prefill exactly to the
        // boundary, then capture.
        rt.reset_context();
        Trace build; build.ok = true;
        ok("prefill to the snapshot boundary", fwd(prefixA.data(), N, 0, build), build.err);
        const std::string se = rt.snapshot_context();
        ok("snapshot_context()", se.empty(), se);
        ok("snapshot_depth() is the boundary", rt.snapshot_depth() == N,
           "depth = " + std::to_string(rt.snapshot_depth()) + ", expected " + std::to_string(N));
        ok("snapshot_bytes() is non-zero", rt.snapshot_bytes() > 0);
        std::printf("       snapshot: depth %u, %.3f MiB on the device\n",
                    rt.snapshot_depth(), double(rt.snapshot_bytes()) / (1024.0 * 1024.0));

        // A fresh request: the engine's `pos == 0 → reset_context()` has fired,
        // and the cache hit restores instead of re-prefilling.
        rt.reset_context();
        ok("context is empty after reset", rt.context_length() == 0);
        const uint32_t m = rt.restore_context();
        ok("restore_context() returns the snapshot depth", m == N,
           "returned " + std::to_string(m));
        ok("context_length() is the restored depth", rt.context_length() == N,
           "length = " + std::to_string(rt.context_length()));
        if (m == N) run_tail(cached);
        ok("continuation after restore", cached.ok, cached.err);

        if (deterministic && ref.ok && cached.ok) {
            // ref.rows[0] is the prefix prefill the cache skipped; compare from
            // the suffix forward onward.
            const long d = first_diff(ref, 1, cached, 0);
            ok("restore + suffix is BIT-IDENTICAL to prefix + suffix", d < 0,
               "diverged at forward #" + std::to_string(d) + " after the boundary");
        } else {
            ok(std::string("restore + suffix is BIT-IDENTICAL to prefix + suffix") + skip, true);
        }
    }

    // =======================================================================
    // §2 — the snapshot is a copy, not an alias
    // =======================================================================
    std::printf("\n--- 2: later generation does not mutate the snapshot ---\n");
    {
        // The runtime is currently sitting at N + S + STEPS, having generated
        // right through every buffer the snapshot copied: the sliding ring has
        // wrapped, the compressor emitted new entries, the overlap slot flipped
        // several times.  The snapshot must be untouched by all of it.
        ok("live context has moved well past the snapshot",
           rt.context_length() > N, "length = " + std::to_string(rt.context_length()));
        ok("snapshot_depth() is still the boundary", rt.snapshot_depth() == N,
           "depth = " + std::to_string(rt.snapshot_depth()));

        rt.reset_context();
        const uint32_t m = rt.restore_context();
        ok("the snapshot still restores to its depth", m == N,
           "returned " + std::to_string(m));
        Trace second; second.ok = true;
        if (m == N) run_tail(second);
        ok("continuation after the second restore", second.ok, second.err);
        if (deterministic && cached.ok && second.ok) {
            const long d = first_diff(cached, 0, second, 0);
            ok("restoring the same snapshot twice gives BIT-IDENTICAL output", d < 0,
               "diverged at forward #" + std::to_string(d));
        } else {
            ok(std::string("restoring the same snapshot twice gives BIT-IDENTICAL output")
               + skip, true);
        }
    }

    // =======================================================================
    // §3 — restore actually restores (the discriminator)
    // =======================================================================
    std::printf("\n--- 3: a snapshot of a DIFFERENT conversation gives different output ---\n");
    {
        rt.reset_context();
        Trace build; build.ok = true;
        fwd(prefixB.data(), N, 0, build);
        const std::string se = rt.snapshot_context();
        ok("snapshot conversation B", se.empty() && build.ok, se.empty() ? build.err : se);

        rt.reset_context();
        const uint32_t m = rt.restore_context();
        ok("restore B", m == N, "returned " + std::to_string(m));
        Trace other; other.ok = true;
        if (m == N) run_tail(other);
        ok("continuation after restoring B", other.ok, other.err);

        if (cached.ok && other.ok) {
            // Same suffix, same positions, same decode script — only the
            // restored prefix differs.  If these matched, the restore would be
            // writing nothing and §1 would be measuring the suffix alone.
            const long d = first_diff(cached, 0, other, 0);
            ok("the same suffix over a different restored prefix DIFFERS", d >= 0,
               "identical — restore is not writing the cached state");
        }
    }

    // =======================================================================
    // §4 — the failure paths are honest, and slots are respected
    // =======================================================================
    std::printf("\n--- 4: free_snapshot, and per-slot isolation ---\n");
    {
        rt.free_snapshot();
        ok("free_snapshot() clears the depth", rt.snapshot_depth() == 0);
        ok("free_snapshot() releases the device bytes", rt.snapshot_bytes() == 0);
        rt.reset_context();
        ok("restore_context() with no snapshot returns 0, not a stale depth",
           rt.restore_context() == 0);

        // Snapshot slot 0, restore into slot 1, and require slot 0 untouched.
        // A snapshot/restore pair that ignored the active slot would be the
        // cross-conversation contamination the seq-slot gate exists to prevent,
        // arrived at through the prompt cache instead.
        ok("select slot 0", rt.select_seq(0).empty());
        rt.reset_context();
        Trace b0; b0.ok = true;
        fwd(prefixA.data(), N, 0, b0);
        const std::string se = rt.snapshot_context();
        ok("snapshot taken on slot 0", se.empty(), se);
        // Advance slot 0 past the boundary so "untouched" is a claim with teeth.
        fwd(suffix.data(), S, N, b0);
        const uint64_t slot0_len = rt.context_length();

        ok("select slot 1", rt.select_seq(1).empty());
        rt.reset_seq(1);
        ok("slot 1 starts empty", rt.context_length() == 0);
        const uint32_t m = rt.restore_context();
        ok("restore into slot 1 returns the depth", m == N, "returned " + std::to_string(m));
        ok("slot 1 now holds the restored context", rt.context_length() == N,
           "length = " + std::to_string(rt.context_length()));

        ok("select slot 0 again", rt.select_seq(0).empty());
        ok("slot 0's context was NOT disturbed by slot 1's restore",
           rt.context_length() == slot0_len,
           "length = " + std::to_string(rt.context_length()) + ", expected " +
           std::to_string(slot0_len));
    }

    // -----------------------------------------------------------------------
    std::printf("\n--- 5: host-resident snapshot (docs/deepseek4/72 Phase L3) ---\n");
    // -----------------------------------------------------------------------
    // The prompt cache now keeps conversations' states in PINNED HOST memory
    // (Ds4HostSnapshot: export_host / import_host per layer).  Same four claims
    // as §1-§3, through the host path: exact, a copy not an alias, actually
    // restoring, and an empty snapshot restores nothing.
    {
        ok("slot 0 selected for §5", rt.select_seq(0).empty());
        rt.reset_context();
        Trace build; build.ok = true;
        ok("prefill A to the boundary", fwd(prefixA.data(), N, 0, build), build.err);
        ie::Ds4HostSnapshot hs;
        const std::string se = rt.snapshot_to_host(hs);
        ok("snapshot_to_host()", se.empty(), se);
        ok("host snapshot depth is the boundary", hs.depth == N,
           "depth = " + std::to_string(hs.depth) + ", expected " + std::to_string(N));
        ok("host snapshot bytes are non-zero", hs.bytes() > 0);
        std::printf("       host snapshot: depth %u, %.3f MiB pinned\n", hs.depth,
                    double(hs.bytes()) / (1024.0 * 1024.0));
        // generation continues past the boundary; the host copy must not move
        Trace live; live.ok = true;
        run_tail(live);
        rt.reset_context();
        ok("context is empty after reset", rt.context_length() == 0);
        const uint32_t m = rt.restore_from_host(hs);
        ok("restore_from_host() returns the snapshot depth", m == N, "returned " + std::to_string(m));
        ok("context_length() is the restored depth", rt.context_length() == N,
           "length = " + std::to_string(rt.context_length()));
        Trace hc; hc.ok = true;
        if (m == N) run_tail(hc);
        ok("continuation after the host restore", hc.ok, hc.err);
        if (deterministic && ref.ok && hc.ok) {
            const long d = first_diff(ref, 1, hc, 0);
            ok("host restore + suffix is BIT-IDENTICAL to prefix + suffix", d < 0,
               "diverged at forward #" + std::to_string(d) + " after the boundary");
        } else {
            ok(std::string("host restore + suffix is BIT-IDENTICAL to prefix + suffix") + skip, true);
        }
        // the same host snapshot restored a second time, after more generation
        rt.reset_context();
        const uint32_t m2 = rt.restore_from_host(hs);
        ok("the host snapshot still restores to its depth", m2 == N, "returned " + std::to_string(m2));
        Trace hc2; hc2.ok = true;
        if (m2 == N) run_tail(hc2);
        if (deterministic && hc.ok && hc2.ok) {
            const long d = first_diff(hc, 0, hc2, 0);
            ok("restoring the same host snapshot twice gives BIT-IDENTICAL output", d < 0,
               "diverged at forward #" + std::to_string(d));
        } else {
            ok(std::string("restoring the same host snapshot twice gives BIT-IDENTICAL output") + skip, true);
        }
        // a host snapshot of the OTHER conversation must give different output
        rt.reset_context();
        Trace buildB; buildB.ok = true;
        fwd(prefixB.data(), N, 0, buildB);
        ie::Ds4HostSnapshot hb;
        const std::string sb = rt.snapshot_to_host(hb);
        ok("host snapshot of conversation B", sb.empty() && buildB.ok, sb.empty() ? buildB.err : sb);
        rt.reset_context();
        const uint32_t mb = rt.restore_from_host(hb);
        ok("restore B from the host", mb == N, "returned " + std::to_string(mb));
        Trace ob; ob.ok = true;
        if (mb == N) run_tail(ob);
        if (hc.ok && ob.ok) {
            const long d = first_diff(hc, 0, ob, 0);
            ok("the same suffix over a different host-restored prefix DIFFERS", d >= 0,
               "identical — the host restore is not writing the cached state");
        }
        // reusing a snapshot object for a new export re-pins as needed and stays exact
        rt.reset_context();
        Trace buildA2; buildA2.ok = true;
        fwd(prefixA.data(), N, 0, buildA2);
        const std::string se2 = rt.snapshot_to_host(hb);   // hb now holds A
        ok("re-export into a used snapshot object", se2.empty() && buildA2.ok, se2);
        rt.reset_context();
        Trace hc3; hc3.ok = true;
        if (rt.restore_from_host(hb) == N) run_tail(hc3);
        if (deterministic && hc.ok && hc3.ok) {
            const long d = first_diff(hc, 0, hc3, 0);
            ok("the re-exported snapshot restores BIT-IDENTICALLY", d < 0,
               "diverged at forward #" + std::to_string(d));
        }
        ie::Ds4HostSnapshot empty;
        ok("an empty host snapshot restores nothing (returns 0)", rt.restore_from_host(empty) == 0);
    }

    std::printf("\n=== %s (%d failure%s) ===\n", g_fail ? "FAILED" : "PASSED",
                g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
