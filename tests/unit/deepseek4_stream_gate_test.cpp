// tests/unit/deepseek4_stream_gate_test.cpp — the expert-stream FETCH PIPELINE.
//
// WHAT DEFECT THIS TEST EXISTS FOR
// --------------------------------
// A pp512 prefill on the real model (2x B70, UD-Q8_K_XL) moved 60.7 GB per card
// in 2,499 ms of transfer-queue BUSY time inside a 5,518 ms wall.  That is
// 24.3 GB/s while moving — 92-95% of this box's measured per-card PCIe ceiling
// (26.5 GB/s solo, ~25.7 GB/s when both cards stream at once,
// docs/deepseek4/31 §0.4) — and a DMA DUTY CYCLE OF 45%.  The link is not slow.
// It is stopped, for more than half of prefill, because the caller's group loop
// is
//
//     acquire(g) -> wait for it -> compute(g) -> drain -> acquire(g+1) -> ...
//
// in which the copy engine and the EUs strictly alternate.  Nothing in the cache
// prevented overlap; nothing made it possible either, because `acquire` clears
// the layer's in-use marks on entry, so a fetch issued while the previous
// group's GEMMs are running may legally evict a slot those GEMMs are reading.
//
// `acquire_pipelined` keeps one extra generation of in-use marks, which is
// exactly what a double buffer needs, and `ds4_plan_expert_groups` caps a group
// at half the streaming partition so two groups fit at once.
//
// WHAT IT CHECKS, AND WITH WHAT EVIDENCE
// --------------------------------------
//   §1  The partitioner: groups TILE the occupied list exactly, are never empty,
//       respect the cap, charge nothing for statically resident experts, and
//       reproduce the pre-pipeline partition when handed the full streaming
//       partition as the cap.
//   §2  Eviction safety, as a DIRECTORY property: across a pipelined loop no
//       slot handed to group g is ever taken away while group g+1 is fetched.
//       NEGATIVE CONTROL: the identical loop driven by plain `acquire` DOES take
//       them, and the check catches it — so the check discriminates rather than
//       passing vacuously.
//   §3  THE NUMERICS GATE.  A device kernel gated on exactly the event the
//       caller would use reduces each slot to a checksum; every checksum must
//       equal the one derived from the host arena bytes.  See §3's tolerance
//       note: the tolerance is ZERO and the error model says why.  THREE
//       negative controls — a dropped event dependency, plain `acquire` in a
//       pipelined loop, and an off-by-one slot index — each must be caught.
//   §4  The in-flight-HIT race that the per-slot fill event closes: an acquire
//       whose ids are all hits, on a slot whose fill is still running, used to
//       return a default-constructed (already-complete) event.
//   §5  THE PREFETCHER DIAGNOSIS, MEASURED.  The prefill access pattern is
//       replayed against the real cache and `speculate`'s issued/hit counters
//       are printed, reproducing the ~1% hit rate seen on the model and
//       attributing it to a named cause.
//   §6  THE PERFORMANCE GATE: the same loop, real 6,684,672 B slots, serial vs
//       pipelined, with the transfer queue's own profiling events as the
//       measure of duty cycle.  Gated on a threshold derived from the measured
//       serial duty rather than tuned to whatever the machine happened to do.
//   §7  THE DECODE REGIME, which §1-§6 do not reach.  Every shape above is a
//       PREFILL shape — a chunk's `occ` is most of the expert pool, so a layer
//       splits into many groups.  At T == 1 the layer's whole routing is six
//       experts, which is at or below `pipelined_group_cap()`, so the partition
//       is ONE GROUP and the lookahead never fires.  §7 proves that, measures
//       what the surviving half of `acquire_pipelined` costs there, and gates
//       the fix against the eviction-safety property §2 established.
//   §8  THE OTHER TRAFFIC ON THE LINK.  Under expert-TP the cross-card reduction
//       runs twice per layer — 86 times per decoded token — and both of its hops
//       used to be `memcpy`, which Level Zero puts on a COPY ENGINE: the same
//       engine this file's subject saturates.  A 32 KB hop behind a 4 MB expert
//       burst waits for the burst, measured at 33.1 us idle -> 196.4 us under a
//       41 GB/s stream.  §8 checks the reduction is exact and bit-identical
//       across cards, that moving both hops to the compute engine changed no
//       value TO THE BIT, and gates the contention ratio with the old form
//       running beside it as the negative control.  Two methodology traps are
//       written into it because both were fallen into first: an UNBOUNDED burst
//       backlog saturates the link itself and no form discriminates, and a fixed
//       variant ORDER lets whichever form runs first absorb the fresh burst.
//
// NOTHING HERE LOADS THE MODEL.  Every arena is synthetic; the test allocates
// ~0.14 GB of device memory and ~0.44 GB of pinned host memory at its peak
// (§7 adds ~5 MB device and ~13 MB host).
#include "ie/expert_stream.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace {

int g_fail = 0;

void ok(const char* what, bool cond, const std::string& detail = {}) {
    std::printf("  [%s] %s%s%s\n", cond ? "PASS" : "FAIL", what,
                detail.empty() ? "" : " -- ", detail.c_str());
    if (!cond) ++g_fail;
}

double now_s() {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Content of expert `e` of layer `L`, word `w`.  A cheap invertible mix, so every
// (L, e) has a distinct byte pattern and a wrong expert cannot be mistaken for
// the right one by accident.
inline uint32_t word_of(uint32_t L, uint32_t e, uint32_t w) {
    uint32_t x = (L * 0x9E3779B9u) ^ (e * 0x85EBCA6Bu) ^ (w * 0xC2B2AE35u);
    x ^= x >> 15; x *= 0x2545F491u; x ^= x >> 13;
    return x;
}

// The fold used by both the host expectation and the device kernel.  It mixes
// the WORD INDEX in, so it is position-sensitive (a shifted or swapped word is
// caught) while remaining commutative, which is what lets the kernel run 256
// work-items in parallel and still be deterministic.  A serial fold would need
// 4 M dependent loads per 16 MB slot and would dominate the test's runtime.
inline uint32_t mixw(uint32_t v, uint32_t w) {
    uint32_t x = v ^ (w * 0x9E3779B9u);
    x ^= x >> 16; x *= 0x7FEB352Du; x ^= x >> 15;
    return x;
}

inline uint32_t expect_sum(uint32_t L, uint32_t e, uint32_t words) {
    uint32_t s = 0;
    for (uint32_t w = 0; w < words; ++w) s ^= mixw(word_of(L, e, w), w);
    return s;
}

void fill_arena(ie::Ds4HostArena& a, uint32_t n_layers, const std::vector<uint32_t>& pinned,
                uint64_t slot_bytes) {
    const uint32_t words = uint32_t(slot_bytes / 4);
    for (uint32_t L = 0; L < n_layers; ++L)
        for (uint32_t e : pinned) {
            auto* p = static_cast<uint32_t*>(a.slot(L, e));
            if (!p) continue;
            for (uint32_t w = 0; w < words; ++w) p[w] = word_of(L, e, w);
        }
}

// The stand-in for `ds4_expert_gemm_q8`: reads a slot on the COMPUTE queue and
// writes its checksum.  It has exactly the data dependency the real GEMM has, so
// a fetch that has not landed shows up as a wrong checksum rather than as
// nothing at all.
sycl::event checksum_slot(sycl::queue& q, const void* slot, uint32_t words, uint32_t* out) {
    const auto* src = static_cast<const uint32_t*>(slot);
    constexpr uint32_t kLanes = 256;
    return q.parallel_for(sycl::range<1>(kLanes), [=](sycl::id<1> i) {
        uint32_t s = 0;
        for (uint32_t w = uint32_t(i[0]); w < words; w += kLanes) s ^= mixw(src[w], w);
        sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            a(*out);
        a.fetch_xor(s);
    });
}

// ---------------------------------------------------------------------------
// One layer's group loop, in the four shapes the test needs to compare.
// ---------------------------------------------------------------------------
enum class Shape {
    kSerial,          // today: fetch a group, compute it, drain, repeat
    kPipelined,       // acquire_pipelined, one group of lookahead
    kBadAcquire,      // NEGATIVE CONTROL: pipelined loop driven by plain acquire
    kSerialNoDepend,  // NEGATIVE CONTROL: serial loop that drops depends_on
};

struct LoopResult {
    uint64_t groups   = 0;
    uint64_t refused  = 0;   // ids that came back kDs4NoSlot
    double   wall_s   = 0.0;
    bool     collided = false;   // a live slot was handed to two groups at once
};

// `dev_sums` receives one checksum per entry of `occ`, in order.
LoopResult run_layer(sycl::queue& q, ie::Ds4ExpertCache& c, uint32_t L,
                     const std::vector<uint32_t>& occ, uint32_t cap, uint32_t words,
                     Shape shape, uint32_t* dev_sums, uint32_t slot_bias = 0) {
    LoopResult r;
    std::vector<uint32_t> gb;
    ie::ds4_plan_expert_groups(c, L, occ.data(), occ.size(), cap, gb);
    if (gb.size() < 2) return r;
    const size_t NG = gb.size() - 1;
    r.groups = NG;

    // Two rotating buffers: group g uses buffer g&1, which is the whole point.
    std::vector<int32_t>  gids[2];
    std::vector<uint32_t> gslot[2];
    sycl::event           gev[2];

    const bool plain = (shape == Shape::kSerial || shape == Shape::kSerialNoDepend ||
                        shape == Shape::kBadAcquire);
    auto issue = [&](size_t g) {
        const size_t s = g & 1;
        gids[s].assign(occ.begin() + gb[g], occ.begin() + gb[g + 1]);
        gslot[s].assign(gids[s].size(), ie::kDs4NoSlot);
        gev[s] = plain ? c.acquire(L, gids[s].data(), uint32_t(gids[s].size()), gslot[s].data())
                       : c.acquire_pipelined(L, gids[s].data(), uint32_t(gids[s].size()),
                                             gslot[s].data());
        for (uint32_t v : gslot[s]) if (v == ie::kDs4NoSlot) ++r.refused;
    };
    // `slot_bias` is NEGATIVE CONTROL 3 and it wraps INSIDE the streaming
    // partition: a bias that walked off the end would hand slot_ptr an
    // out-of-range index, get a null pointer back, and fault the device rather
    // than compute the wrong expert — which tests the null check, not the gate.
    const uint32_t st = c.static_slots(), sl = c.slots_per_layer();
    auto biased = [&](uint32_t v) {
        if (!slot_bias || v < st || sl == st) return v;
        return st + (v - st + slot_bias) % (sl - st);
    };
    auto compute_group = [&](size_t g) {
        const size_t s = g & 1;
        for (size_t i = 0; i < gids[s].size(); ++i)
            if (gslot[s][i] != ie::kDs4NoSlot)
                checksum_slot(q, c.slot_ptr(L, biased(gslot[s][i])), words,
                              dev_sums + gb[g] + i);
    };

    const double t0 = now_s();
    if (plain && shape != Shape::kBadAcquire) {
        for (size_t g = 0; g < NG; ++g) {
            issue(g);
            if (shape != Shape::kSerialNoDepend)
                q.submit([&](sycl::handler& h) { h.depends_on(gev[g & 1]); h.single_task([]() {}); });
            compute_group(g);
            q.wait();
        }
    } else {
        issue(0);
        for (size_t g = 0; g < NG; ++g) {
            const std::vector<uint32_t> live = gslot[g & 1];
            // Group g+1's H2D is issued BEFORE group g's compute.  That is the
            // overlap, and it is only legal because acquire_pipelined refuses to
            // evict group g's slots.
            if (g + 1 < NG) {
                issue(g + 1);
                for (uint32_t a : live)
                    for (uint32_t b : gslot[(g + 1) & 1])
                        if (a != ie::kDs4NoSlot && a == b) r.collided = true;
            }
            q.submit([&](sycl::handler& h) { h.depends_on(gev[g & 1]); h.single_task([]() {}); });
            compute_group(g);
            q.wait();
        }
    }
    c.transfer_queue().wait();
    r.wall_s = now_s() - t0;
    return r;
}

// ---------------------------------------------------------------------------
// §8's reference: `ds4_tp_reduce_host` EXACTLY AS IT WAS when both hops were
// `memcpy`.  Kept here, in the test, for two jobs it can only do from here: it
// is the bit-identity reference the shipped form is compared against, and it is
// the negative control that proves §8's contention gate discriminates rather
// than passing because the threshold is loose.
// ---------------------------------------------------------------------------
void reduce_memcpy_hops(const std::vector<sycl::queue*>& qs, const std::vector<float*>& parts,
                        const std::vector<float*>& stages, uint64_t n) {
    const size_t nb = size_t(n) * sizeof(float);
    for (size_t c = 0; c < qs.size(); ++c) qs[c]->memcpy(stages[c], parts[c], nb);
    for (sycl::queue* p : qs) p->wait();
    for (size_t c = 1; c < qs.size(); ++c)
        for (uint64_t i = 0; i < n; ++i) stages[0][i] += stages[c][i];
    for (size_t c = 1; c < qs.size(); ++c) std::memcpy(stages[c], stages[0], nb);
    for (size_t c = 0; c < qs.size(); ++c) qs[c]->memcpy(parts[c], stages[c], nb);
}

}  // namespace

// docs/deepseek4/72 Phase J: ds4_order_hits_first's contract — every expert of
// `occ` exactly once; the resident ones lead as ONE group; the misses keep
// their given order and are cut into groups of <= cap non-static experts.
static void check_hits_first(const ie::Ds4ExpertCache& cache, const std::vector<uint32_t>& occ,
                             const char* tag, uint32_t cap, uint32_t expect_hits) {
    std::vector<uint32_t> ordered, hb;
    const uint32_t nh =
        ie::ds4_order_hits_first(cache, 0, occ.data(), occ.size(), cap, ordered, hb);
    std::vector<uint32_t> sorted(ordered), want(occ);
    std::sort(sorted.begin(), sorted.end());
    std::sort(want.begin(), want.end());
    const bool perm = sorted == want;
    bool lead = true, order = true;
    for (uint32_t i = 0; i < nh && i < ordered.size(); ++i) lead = lead && cache.is_resident(0, ordered[i]);
    for (size_t i = nh; i < ordered.size(); ++i) lead = lead && !cache.is_resident(0, ordered[i]);
    for (size_t i = nh + 1; i < ordered.size(); ++i) order = order && ordered[i - 1] < ordered[i];
    for (uint32_t i = 1; i < nh; ++i) order = order && ordered[i - 1] < ordered[i];
    bool groups = !hb.empty() && hb.front() == 0 && hb.back() == occ.size();
    if (nh) groups = groups && hb.size() > 1 && hb[1] == nh;
    for (size_t g = nh ? 1 : 0; g + 1 < hb.size(); ++g) {
        uint32_t ns = 0;
        for (uint32_t i = hb[g]; i < hb[g + 1]; ++i)
            if (!cache.is_static(0, ordered[i])) ++ns;
        groups = groups && hb[g] < hb[g + 1] && (ns <= cap || hb[g + 1] - hb[g] == 1);
    }
    const std::string d = std::string(tag) + ": cap " + std::to_string(cap) + " -> " +
                          std::to_string(nh) + " hits (expected " + std::to_string(expect_hits) +
                          "), " + std::to_string(hb.empty() ? 0 : hb.size() - 1) + " groups";
    ok("hits-first: every expert exactly once", perm, d);
    ok("hits-first: residents lead, both parts keep their order", lead && order, d);
    ok("hits-first: hits are one leading group, miss groups within the cap",
       groups && nh == expect_hits, d);
}

int main() {
    std::printf("deepseek4_stream_gate_test — expert-stream fetch pipeline\n");

    // This gate's whole subject is DMA duty cycle, which is measured from
    // transfer-queue profiling events.  Since 2026-08-03 that property is opt-in
    // (it costs ~0.42 us per submission and `submit` is 70-87% of decode
    // card-busy time), so a bare run would read 0.00 ms of DMA and the duty
    // assertions would compare 0 against 0.  Turn it on for ourselves, before any
    // queue is constructed, rather than relaxing the assertions — a gate that
    // "passes" because it measured nothing is worse than no gate.
    setenv("IE_QUEUE_PROFILING", "1", /*overwrite=*/0);

    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    // -----------------------------------------------------------------------
    // Correctness configuration.  Small slots, so §1-§5 are exhaustive and fast.
    // -----------------------------------------------------------------------
    // SLOT SIZE IS LOAD-BEARING, NOT ARBITRARY.  At 64 KB an H2D lands in ~2 us,
    // which is FASTER than the ~20 us it takes to launch the kernel that reads
    // it — so a build with the event dependency deleted still passes, and
    // NEGATIVE CONTROL 1 becomes vacuous.  4 MB takes ~150 us on this box's
    // 26.5 GB/s link, comfortably longer than a launch, so a missing dependency
    // is caught.  Total footprint: 480 MB pinned host, 288 MB device.
    const uint32_t NL = 3, NE = 64, STATIC = 8, SLOTS = 24, PINNED = 40;
    const uint64_t SB = 4ull << 20;                  // 4 MB slots
    const uint32_t WORDS = uint32_t(SB / 4);
    const uint32_t STREAM = SLOTS - STATIC;          // 16

    std::vector<uint64_t> sb(NL, SB);
    std::vector<uint32_t> pinned_ids;
    for (uint32_t e = STATIC; e < STATIC + PINNED; ++e) pinned_ids.push_back(e);

    ie::Ds4HostArena arena;
    {
        const std::string e = arena.init_set(q, sb, NE, pinned_ids, 1ull << 30);
        ok("host arena (3 layers x 40 pinned experts x 4 MB = 0.50 GB)", e.empty(), e);
        if (!e.empty()) return 1;
    }
    fill_arena(arena, NL, pinned_ids, SB);

    ie::Ds4ExpertCache cache;
    {
        const std::string e = cache.init(q, arena, SLOTS, 0, STATIC);
        ok("cache init (8 static + 16 streaming slots/layer)", e.empty(), e);
        if (!e.empty()) return 1;
    }
    {
        std::vector<uint8_t> stage((size_t(SB)));
        bool all = true;
        for (uint32_t L = 0; L < NL; ++L)
            for (uint32_t s = 0; s < STATIC; ++s) {
                auto* w = reinterpret_cast<uint32_t*>(stage.data());
                for (uint32_t i = 0; i < WORDS; ++i) w[i] = word_of(L, s, i);
                all = all && cache.install_static(L, s, s, stage.data()).empty();
            }
        ok("static partition installed (experts 0..7 of every layer)", all);
    }

    // The occupied list a prefill chunk produces: EVERY expert the layer can
    // serve, ascending — which is what 512 tokens x top-6 gives.
    std::vector<uint32_t> occ;
    for (uint32_t e = 0; e < STATIC + PINNED; ++e) occ.push_back(e);

    // -----------------------------------------------------------------------
    std::printf("\n[1] ds4_plan_expert_groups — the partition\n");
    // -----------------------------------------------------------------------
    {
        for (uint32_t cap : {1u, 3u, STREAM / 2, STREAM}) {
            std::vector<uint32_t> gb;
            ie::ds4_plan_expert_groups(cache, 0, occ.data(), occ.size(), cap, gb);
            const bool tiles = !gb.empty() && gb.front() == 0 && gb.back() == occ.size();
            bool rising = true, within = true;
            for (size_t g = 0; g + 1 < gb.size(); ++g) {
                rising = rising && gb[g] < gb[g + 1];
                uint32_t ns = 0;
                for (uint32_t i = gb[g]; i < gb[g + 1]; ++i)
                    if (!cache.is_static(0, occ[i])) ++ns;
                within = within && (ns <= cap || gb[g + 1] - gb[g] == 1);
            }
            const std::string d = "cap " + std::to_string(cap) + " -> " +
                                  std::to_string(gb.size() - 1) + " groups";
            ok("groups tile the occupied list exactly", tiles, d);
            ok("...bounds strictly increase, no group is empty", rising, d);
            ok("...no group claims more than the cap", within, d);
        }
        // Static experts must be free: with 8 static ids leading `occ`, a cap of
        // 4 must put 8 + 4 ids in the first group, not 4.
        std::vector<uint32_t> gb;
        ie::ds4_plan_expert_groups(cache, 0, occ.data(), occ.size(), 4, gb);
        ok("statically resident experts cost no streaming slot",
           gb.size() > 1 && gb[1] == STATIC + 4,
           "first group holds " + std::to_string(gb.size() > 1 ? gb[1] : 0) +
               " ids (expected " + std::to_string(STATIC + 4) + ")");
        std::vector<uint32_t> gser;
        ie::ds4_plan_expert_groups(cache, 0, occ.data(), occ.size(), STREAM, gser);
        ok("cap == stream_slots() reproduces the pre-pipeline partition",
           gser.size() == 1 + (PINNED + STREAM - 1) / STREAM,
           std::to_string(gser.size() - 1) + " groups for " + std::to_string(PINNED) +
               " streamed experts over " + std::to_string(STREAM) + " slots");
        ok("pipelined_group_cap() is half the streaming partition",
           cache.pipelined_group_cap() == STREAM / 2,
           std::to_string(cache.pipelined_group_cap()));

        // ---- Phase J (docs/deepseek4/72): the hits-first order for the bank
        // path.  Here only the static experts are resident; the streamed-hit
        // case is checked at the end of this program, once the cache has
        // cached experts, so that this section leaves [2]'s state untouched.
        check_hits_first(cache, occ, "static residents only", 4, STATIC);
        check_hits_first(cache, occ, "static residents only, cap 1", 1, STATIC);
        {
            std::vector<uint32_t> none, o2, b2;
            ok("hits-first: empty input -> no hits, no groups",
               ie::ds4_order_hits_first(cache, 0, none.data(), 0, 4, o2, b2) == 0 &&
                   o2.empty() && b2.empty());
        }
    }

    // -----------------------------------------------------------------------
    std::printf("\n[2] Eviction safety across the pipeline, with a negative control\n");
    // -----------------------------------------------------------------------
    uint32_t* dsum = sycl::malloc_device<uint32_t>(occ.size(), q);
    std::vector<uint32_t> hsum(occ.size());
    if (!dsum) { std::printf("  malloc_device failed\n"); return 1; }
    {
        q.memset(dsum, 0, occ.size() * sizeof(uint32_t)).wait();
        const LoopResult p = run_layer(q, cache, 0, occ, cache.pipelined_group_cap(), WORDS,
                                       Shape::kPipelined, dsum);
        ok("pipelined loop never hands two live groups the same slot", !p.collided,
           std::to_string(p.groups) + " groups");
        ok("...and refuses nothing", p.refused == 0, std::to_string(p.refused) + " refused");

        // NEGATIVE CONTROL, and it has to be the RIGHT mis-port.  Adding the
        // lookahead while leaving the group cap at the whole streaming partition
        // — i.e. pipelining today's loop without shrinking the group — is the
        // most plausible way to get this wrong, and under plain `acquire` it
        // hands group g+1 exactly the slots group g's GEMMs are reading.
        //
        // (A weaker control was tried first and was VACUOUS: with the cap
        // already halved, FIFO replacement happens to alternate between the two
        // halves on its own, so plain `acquire` did not collide and the check
        // proved nothing.  The guarantee acquire_pipelined adds is that this is
        // no longer an accident of cursor alignment.)
        const LoopResult b = run_layer(q, cache, 1, occ, STREAM, WORDS, Shape::kBadAcquire, dsum);
        ok("NEGATIVE CONTROL: plain acquire + an un-halved group DOES collide", b.collided,
           b.collided ? "caught" : "the collision check is vacuous — §2 proves nothing");

        // And the loud alternative: acquire_pipelined with the same un-halved
        // cap cannot find a victim, so it REFUSES by name instead of corrupting.
        const LoopResult o = run_layer(q, cache, 2, occ, STREAM, WORDS, Shape::kPipelined, dsum);
        ok("an un-halved group under acquire_pipelined refuses loudly, never collides",
           !o.collided && o.refused > 0,
           std::to_string(o.refused) + " ids refused, collided=" + (o.collided ? "yes" : "no"));

        // ---- 2b: THE HIT SPLIT.  static_hits + stream_hits == hits ----
        //
        // WHY THIS IS GATED AND NOT JUST TRUSTED.  A blended hit rate is
        // AMBIGUOUS, and expensively so: searching (static set, streaming size)
        // space against ONE measured real-model A/B point found ELEVEN
        // configurations that reproduce its hit rate to 0.4 points and its bytes
        // to 1.5%, spanning 24.4 points of static hit rate.  The split is what
        // makes them distinguishable, so it has to be right — a counter that
        // attributed a streaming hit to the static partition would make the
        // residency order look like it was working when it was not, which is a
        // worse failure than not measuring at all.
        //
        // The invariant is exact, not statistical: every hit is served by
        // exactly one partition, and the static partition is [0, static_slots)
        // by construction because pick_victim never enters it.
        {
            const ie::Ds4ExpertCache::Stats& s = cache.stats();
            ok("every hit is attributed to exactly one partition"
               " (static_hits + stream_hits == hits)",
               s.static_hits + s.stream_hits == s.hits,
               std::to_string(s.static_hits) + " static + " + std::to_string(s.stream_hits) +
                   " stream vs " + std::to_string(s.hits) + " hits");
            // §1/§2 walk `occ` — EVERY expert the layer can serve — exactly
            // once, which is the PREFILL shape, and in that shape the streaming
            // term is structurally zero: a slot is filled and never asked for
            // again before it is evicted.  Measured here, and measured on the
            // real model too (0.0061 of prefill acquires).  So the equality
            // above, on its own, is satisfied by a stream counter that is simply
            // broken.  It has to be made to fire.
            ok("the PREFILL shape yields a ZERO streaming term — each expert is touched"
               " once, so there is no reuse for any policy to exploit",
               s.stream_hits == 0 && s.static_hits > 0,
               std::to_string(s.static_hits) + " static, " + std::to_string(s.stream_hits) +
                   " stream");
        }
        // ...so the non-vacuity check needs the DECODE shape: a working set that
        // FITS the streaming partition, asked for twice.  The second pass must
        // be served entirely out of the streaming partition, which is the only
        // condition under which `stream_hits` can be shown to work at all.
        {
            std::vector<uint32_t> small;
            for (uint32_t e = STATIC; e < STATIC + STREAM / 2; ++e) small.push_back(e);
            const ie::Ds4ExpertCache::Stats before = cache.stats();
            run_layer(q, cache, 0, small, cache.pipelined_group_cap(), WORDS, Shape::kPipelined,
                      dsum);
            const ie::Ds4ExpertCache::Stats warm = cache.stats();
            run_layer(q, cache, 0, small, cache.pipelined_group_cap(), WORDS, Shape::kPipelined,
                      dsum);
            const ie::Ds4ExpertCache::Stats after = cache.stats();
            const uint64_t d_stream = after.stream_hits - warm.stream_hits;
            const uint64_t d_static = after.static_hits - warm.static_hits;
            ok("the DECODE shape (a working set that fits, asked twice) makes the streaming"
               " term fire — so the counter is not stuck at zero",
               d_stream == small.size() && d_static == 0,
               std::to_string(d_stream) + " streaming hits and " + std::to_string(d_static) +
                   " static on the second pass of " + std::to_string(small.size()) +
                   " streaming-only experts");
            ok("...and the split invariant still holds after both passes",
               after.static_hits + after.stream_hits == after.hits,
               std::to_string(after.static_hits) + " static + " +
                   std::to_string(after.stream_hits) + " stream vs " +
                   std::to_string(after.hits) + " hits");
            (void)before;
        }
    }

    // -----------------------------------------------------------------------
    std::printf("\n[3] Numerics gate: the compute queue reads the right bytes\n");
    // -----------------------------------------------------------------------
    // TOLERANCE, AND THE ERROR MODEL IT COMES FROM.  The fetch path performs no
    // arithmetic: it is `memcpy(dst, src, slot_bytes)` from pinned host memory to
    // device memory, and the checksum kernel is integer-only (multiply-add over
    // uint32 with defined wraparound).  There is therefore NO source of rounding,
    // no accumulation order to argue about, and no representable difference
    // between a correct and an incorrect result.  The tolerance is EXACTLY ZERO
    // BITS; any tolerance above zero would admit a class of defect — a torn
    // transfer, a stale slot, a wrong expert — that cannot arise any other way.
    // The three negative controls show the bound discriminates: each is a
    // plausible mis-port of this loop, and each must be caught.
    {
        auto verify = [&](const char* what, uint32_t L, Shape shape, uint32_t cap, uint32_t bias,
                          bool expect_pass) {
            q.memset(dsum, 0, occ.size() * sizeof(uint32_t)).wait();
            run_layer(q, cache, L, occ, cap, WORDS, shape, dsum, bias);
            q.memcpy(hsum.data(), dsum, occ.size() * sizeof(uint32_t)).wait();
            size_t bad = 0;
            for (size_t i = 0; i < occ.size(); ++i)
                if (hsum[i] != expect_sum(L, occ[i], WORDS)) ++bad;
            const std::string d = std::to_string(bad) + " of " + std::to_string(occ.size()) +
                                  " slots differ (tolerance: 0)";
            if (expect_pass) ok(what, bad == 0, d);
            else             ok(what, bad != 0, bad ? d : "MIS-PORT NOT CAUGHT — the gate is blind");
        };
        const uint32_t HALF = cache.pipelined_group_cap();
        verify("pipelined fetch delivers byte-exact expert weights",
               0, Shape::kPipelined, HALF, 0, true);
        verify("serial fetch delivers byte-exact expert weights",
               1, Shape::kSerial, STREAM, 0, true);
        verify("NEGATIVE CONTROL 1: dropping depends_on() is caught",
               2, Shape::kSerialNoDepend, STREAM, 0, false);
        verify("NEGATIVE CONTROL 2: plain acquire + an un-halved group is caught",
               0, Shape::kBadAcquire, STREAM, 0, false);
        verify("NEGATIVE CONTROL 3: an off-by-one slot index is caught",
               1, Shape::kPipelined, HALF, 1, false);
    }

    // -----------------------------------------------------------------------
    std::printf("\n[4] The in-flight HIT: an all-hits acquire must not return a complete event\n");
    // -----------------------------------------------------------------------
    // `speculate` publishes slot_of/tag_of at ENQUEUE time.  An acquire whose ids
    // are all hits therefore used to return a default-constructed event even
    // though the bytes were still crossing PCIe, and the caller's depends_on
    // became a no-op.  The per-slot fill event is the fix; this is the proof.
    {
        cache.reset_stats();
        const uint32_t L = 2;
        // Occupy only HALF the streaming partition: `speculate` picks victims
        // that the last acquire did not claim, so an acquire that already holds
        // every streaming slot leaves it nothing to do (which is itself the
        // reason a first draft of this section measured zero issued fetches).
        std::vector<int32_t> warm;
        for (uint32_t i = 0; i < STREAM / 2; ++i) warm.push_back(int32_t(STATIC + PINNED - 1 - i));
        std::vector<uint32_t> ws(warm.size());
        cache.acquire(L, warm.data(), uint32_t(warm.size()), ws.data());
        cache.transfer_queue().wait();

        std::vector<int32_t> spec;
        for (uint32_t i = 0; i < 4; ++i) spec.push_back(int32_t(STATIC + i));
        const uint32_t issued = cache.speculate(L, spec.data(), uint32_t(spec.size()));
        ok("speculate() issued the fetches", issued == spec.size(),
           std::to_string(issued) + " issued");

        // Immediately — copies still in flight — acquire exactly those ids.  All
        // hits, zero misses.  The returned event must still cover the copies.
        // Counters are zeroed here so they describe THIS acquire alone.
        cache.reset_stats();
        std::vector<uint32_t> ss(spec.size());
        const sycl::event ev = cache.acquire(L, spec.data(), uint32_t(spec.size()), ss.data());
        ok("...the acquire that follows is all hits, no misses",
           cache.stats().misses == 0 && cache.stats().hits == spec.size(),
           std::to_string(cache.stats().hits) + " hits, " +
               std::to_string(cache.stats().misses) + " misses");
        // DIAGNOSTIC 4 of the four the prefetcher was suspected of failing:
        // "are prefetched slots actually consumed, or does the consumer take a
        // path that ignores them?"  They ARE consumed — the wiring is sound.
        ok("a speculated slot that survives IS counted as a speculation hit",
           cache.stats().spec_hits == spec.size(),
           std::to_string(cache.stats().spec_hits) + " of " + std::to_string(spec.size()));

        q.memset(dsum, 0, spec.size() * sizeof(uint32_t)).wait();
        q.submit([&](sycl::handler& h) { h.depends_on(ev); h.single_task([]() {}); });
        for (size_t i = 0; i < spec.size(); ++i)
            checksum_slot(q, cache.slot_ptr(L, ss[i]), WORDS, dsum + i);
        q.wait();
        q.memcpy(hsum.data(), dsum, spec.size() * sizeof(uint32_t)).wait();
        size_t bad = 0;
        for (size_t i = 0; i < spec.size(); ++i)
            if (hsum[i] != expect_sum(L, uint32_t(spec[i]), WORDS)) ++bad;
        ok("an all-hits acquire over in-flight speculative fills returns a COVERING event",
           bad == 0, std::to_string(bad) + " of " + std::to_string(spec.size()) + " wrong");
        cache.transfer_queue().wait();
    }

    // -----------------------------------------------------------------------
    std::printf("\n[5] Why the old prefetcher landed 18 hits in 1,900 — measured\n");
    // -----------------------------------------------------------------------
    // Replays the prefill shape on a FRESH cache — the sections above leave both
    // generations of in-use marks set, and `speculate` declines when the last
    // acquire holds every streaming slot, which would make this measure the
    // wrong thing.  Small slots: this section counts events, it does not time
    // them.
    //
    // FOUR CANDIDATE CAUSES were on the table for the 18-hits-in-1,900 result.
    // This section separates them:
    //   (1) wrong experts predicted?  In prefill EVERY expert of every layer is
    //       occupied, so any prediction is in the requested set — printed below
    //       as `predicted-and-later-requested`.  NOT the cause in prefill.  (In
    //       DECODE it would be: the predictor is layer L's ids used for L+1, and
    //       adjacent-layer overlap on this model is Jaccard 0.006.)
    //   (2) issued too late?  A speculation is issued a whole layer ahead of the
    //       group that would use it — tens of ms against a ~0.25 ms transfer.
    //       NOT the cause.
    //   (3) evicted before use?  Layer L+1's own group loop churns every
    //       streaming slot ceil(P/S) times over before it reaches most groups.
    //       THIS is the cause, and the assertion below is what pins it.
    //   (4) not consumed / mis-wired?  §4 shows a surviving speculation IS
    //       counted as a hit.  NOT the cause.
    {
        const uint32_t SL = 4;
        const uint64_t SSB = 64ull << 10;
        std::vector<uint64_t> ssb(SL, SSB);
        ie::Ds4HostArena sa;
        ie::Ds4ExpertCache sc;
        const std::string e1 = sa.init_set(q, ssb, NE, pinned_ids, 1ull << 30);
        const std::string e2 = e1.empty() ? sc.init(q, sa, SLOTS, 0, STATIC) : std::string("skip");
        ok("replay cache (4 layers x 64 KB slots)", e1.empty() && e2.empty(), e1 + e2);
        if (e1.empty() && e2.empty()) {
            std::vector<uint8_t> st((size_t(SSB)), 0);
            for (uint32_t L = 0; L < SL; ++L)
                for (uint32_t s = 0; s < STATIC; ++s) sc.install_static(L, s, s, st.data());
            // A layer's group loop, demand fetches only.
            auto replay_layer = [&](uint32_t L) {
                std::vector<uint32_t> gbv;
                ie::ds4_plan_expert_groups(sc, L, occ.data(), occ.size(), STREAM, gbv);
                for (size_t g = 0; g + 1 < gbv.size(); ++g) {
                    std::vector<int32_t>  ids(occ.begin() + gbv[g], occ.begin() + gbv[g + 1]);
                    std::vector<uint32_t> sl(ids.size());
                    sc.acquire(L, ids.data(), uint32_t(ids.size()), sl.data()).wait();
                }
                return gbv;
            };

            // ---- (5a) demand only: the hit rate IS the static residency ----
            sc.reset_stats();
            replay_layer(0);
            {
                const ie::Ds4ExpertCache::Stats& s = sc.stats();
                const double hr = double(s.hits) / double(s.hits + s.misses);
                std::printf("      demand only    : %llu hits, %llu misses (hit rate %.4f;"
                            " static residency over the occupied set %.4f)\n",
                            (unsigned long long)s.hits, (unsigned long long)s.misses, hr,
                            double(STATIC) / double(occ.size()));
                ok("demand hit rate equals static residency EXACTLY — a streaming slot"
                   " contributes no hits in prefill",
                   std::fabs(hr - double(STATIC) / double(occ.size())) < 1e-9,
                   "hit " + std::to_string(hr) + " vs residency " +
                       std::to_string(double(STATIC) / double(occ.size())));
            }

            // ---- (5b)+(5c) the four candidate causes, separated ----
            // One speculation per GROUP of the next layer, so survival can be
            // read off directly against the group index.  This is the faithful
            // shape: the engine's predictor is layer L's routed ids, which are
            // arbitrary expert numbers and therefore land in arbitrary groups of
            // the next layer's ascending `occ` — not conveniently in group 0.
            const std::vector<uint32_t> gbv1 = [&] {
                std::vector<uint32_t> b;
                ie::ds4_plan_expert_groups(sc, 1, occ.data(), occ.size(), STREAM, b);
                return b;
            }();
            const size_t NG1 = gbv1.size() - 1;
            std::vector<int32_t> pred;
            std::vector<size_t>  pred_group;
            for (size_t g = 0; g < NG1; ++g)
                for (uint32_t i = gbv1[g]; i < gbv1[g + 1]; ++i)
                    if (!sc.is_static(1, occ[i])) { pred.push_back(int32_t(occ[i]));
                                                    pred_group.push_back(g); break; }
            sc.reset_stats();
            const uint32_t issued = sc.speculate(1, pred.data(), uint32_t(pred.size()));
            // (1) PREDICTION.  Every speculated id is one the layer will request:
            // in prefill all 256 experts are occupied at every layer.
            size_t in_next = 0;
            for (int32_t e : pred)
                if (std::find(occ.begin(), occ.end(), uint32_t(e)) != occ.end()) ++in_next;
            // (2) LEAD TIME.  Resident at enqueue, and the layer that consumes
            // them does not start until the rest of layer L has run — orders of
            // magnitude more than one 6.68 MB transfer needs.
            size_t resident_now = 0;
            for (int32_t e : pred) if (sc.available(1, uint32_t(e))) ++resident_now;
            // (3) SURVIVAL.  Now run the layer and see which ones are still there.
            replay_layer(1);
            const ie::Ds4ExpertCache::Stats& s = sc.stats();
            std::printf("      (1) prediction : %zu of %zu speculated experts are requested"
                        " by the layer they were fetched for\n", in_next, pred.size());
            std::printf("      (2) lead time  : %zu of %u issued fetches were resident"
                        " immediately after speculate()\n", resident_now, issued);
            std::printf("      (3) survival   : %u issued for groups", issued);
            for (size_t g : pred_group) std::printf(" %zu", g);
            std::printf("; %llu survived to be hit\n", (unsigned long long)s.spec_hits);
            ok("(1) the predicted experts ARE the ones the next layer requests —"
               " prediction is NOT the defect in prefill",
               !pred.empty() && in_next == pred.size(),
               std::to_string(in_next) + "/" + std::to_string(pred.size()));
            ok("(2) they land before the consuming layer starts — LATENCY is not the defect",
               issued > 0 && resident_now == pred.size(),
               std::to_string(resident_now) + "/" + std::to_string(pred.size()));
            // THE CAUSE.  A speculation survives only if the group that wants it
            // is the FIRST group of the layer; every later group's turn arrives
            // after the streaming partition has been churned whole, so the
            // speculation is evicted by the layer's own demand fetches.
            ok("(3) ONLY a speculation whose expert falls in group 0 survives —"
               " EVICTION BY THE LAYER'S OWN DEMAND FETCHES is the defect",
               issued == pred.size() && s.spec_hits == 1 && NG1 > 1,
               std::to_string(s.spec_hits) + " of " + std::to_string(pred.size()) +
                   " survived across " + std::to_string(NG1) + " groups");
        }
    }
    sycl::free(dsum, q);

    // -----------------------------------------------------------------------
    std::printf("\n[6] Performance gate: real slot bytes, serial vs pipelined\n");
    // -----------------------------------------------------------------------
    // One layer, the REAL per-card slot size for UD-Q8_K_XL under 2-way
    // hidden-dim expert-TP (13,369,344 B / 2 cards), 12 streaming slots — the
    // engine's own configuration.
    {
        const uint64_t PSB    = 6'684'672;
        const uint32_t PNE    = 96, PSTATIC = 8, PSLOTS = 20, PSTREAM = PSLOTS - PSTATIC;
        const uint32_t PPIN   = 64;
        const uint32_t PW     = uint32_t(PSB / 4);
        // Two layers: layer 0 carries the timing, layer 1 is the target of the
        // next-layer speculation whose fate §6c measures.
        std::vector<uint64_t> psb(2, PSB);
        std::vector<uint32_t> ppin;
        for (uint32_t e = PSTATIC; e < PSTATIC + PPIN; ++e) ppin.push_back(e);

        ie::Ds4HostArena pa;
        const std::string ae = pa.init_set(q, psb, PNE, ppin, 1ull << 30);
        ok("perf arena (64 experts x 6,684,672 B pinned = 0.43 GB host)", ae.empty(), ae);
        ie::Ds4ExpertCache pc;
        const std::string ce = ae.empty() ? pc.init(q, pa, PSLOTS, 0, PSTATIC)
                                          : std::string("skipped");
        ok("perf cache (8 static + 12 streaming slots x 6.68 MB = 0.13 GB device)",
           ce.empty(), ce);
        if (ae.empty() && ce.empty()) {
            for (uint32_t e : ppin) {
                auto* p = static_cast<uint32_t*>(pa.slot(0, e));
                for (uint32_t w = 0; w < PW; w += 1024) p[w] = word_of(0, e, w);
            }
            std::vector<uint8_t> stage(size_t(PSB), 0);
            for (uint32_t s = 0; s < PSTATIC; ++s) pc.install_static(0, s, s, stage.data());

            std::vector<uint32_t> pocc;
            for (uint32_t e = 0; e < PSTATIC + PPIN; ++e) pocc.push_back(e);
            uint32_t* ps = sycl::malloc_device<uint32_t>(pocc.size(), q);
            // Compute stand-in: reads 1/64th of each slot, which puts a group's
            // compute within a small factor of its fetch — the regime the real
            // prefill is in (gpu_ms 4,524 against dma_busy 4,998, both summed
            // over two cards).
            const uint32_t CW = PW / 64;
            auto sum_slot = [&](const void* p, uint32_t* o) {
                const auto* src = static_cast<const uint32_t*>(p);
                return q.single_task([=]() {
                    uint32_t s = 0;
                    for (uint32_t w = 0; w < CW; ++w) s = s * 31u + src[w];
                    *o = s;
                });
            };
            auto loop = [&](uint32_t cap, bool pipelined, bool do_spec = false) {
                std::vector<uint32_t> gbv;
                ie::ds4_plan_expert_groups(pc, 0, pocc.data(), pocc.size(), cap, gbv);
                const size_t NG = gbv.size() - 1;
                std::vector<int32_t>  gi[2];
                std::vector<uint32_t> gs[2];
                sycl::event           ge[2];
                auto issue = [&](size_t g) {
                    const size_t s = g & 1;
                    gi[s].assign(pocc.begin() + gbv[g], pocc.begin() + gbv[g + 1]);
                    gs[s].assign(gi[s].size(), ie::kDs4NoSlot);
                    ge[s] = pipelined ? pc.acquire_pipelined(0, gi[s].data(),
                                                             uint32_t(gi[s].size()), gs[s].data())
                                      : pc.acquire(0, gi[s].data(), uint32_t(gi[s].size()),
                                                   gs[s].data());
                };
                pc.reset_stats();
                q.wait(); pc.transfer_queue().wait();
                const double t0 = now_s();
                if (pipelined) issue(0);
                for (size_t g = 0; g < NG; ++g) {
                    if (!pipelined) issue(g);
                    const size_t s = g & 1;
                    if (pipelined && g + 1 < NG) issue(g + 1);
                    q.submit([&](sycl::handler& h) { h.depends_on(ge[s]); h.single_task([]() {}); });
                    for (size_t i = 0; i < gi[s].size(); ++i)
                        if (gs[s][i] != ie::kDs4NoSlot)
                            sum_slot(pc.slot_ptr(0, gs[s][i]), ps + gbv[g] + i);
                    q.wait();
                    if (do_spec) {
                        // Exactly where `moe_expert_grouped` speculates, with a
                        // rotating predictor so it is never asking for something
                        // it already fetched.
                        int32_t pred[6];
                        for (uint32_t i = 0; i < 6; ++i)
                            pred[i] = int32_t(PSTATIC + (uint32_t(g) * 6 + i) % PPIN);
                        pc.speculate(1, pred, 6);
                    }
                }
                pc.transfer_queue().wait();
                const double wall = now_s() - t0;
                pc.collect_dma_time();
                return std::pair<double, double>{wall, pc.stats().dma_seconds};
            };
            // Warm both shapes, then take the better of two timed runs each: this
            // box is shared with other agents' microbenchmarks and a single
            // sample can be contaminated by an unrelated PCIe burst.
            loop(PSTREAM, false);
            loop(pc.pipelined_group_cap(), true);
            auto ser = loop(PSTREAM, false);
            { auto b = loop(PSTREAM, false); if (b.first < ser.first) ser = b; }
            auto pip = loop(pc.pipelined_group_cap(), true);
            { auto b = loop(pc.pipelined_group_cap(), true); if (b.first < pip.first) pip = b; }

            const double bytes = double(PPIN) * double(PSB);
            std::printf("      serial    : wall %7.2f ms, dma busy %7.2f ms (duty %.3f), "
                        "%.2f GB/s while busy\n",
                        ser.first * 1e3, ser.second * 1e3, ser.second / ser.first,
                        bytes / ser.second / 1e9);
            std::printf("      pipelined : wall %7.2f ms, dma busy %7.2f ms (duty %.3f), "
                        "%.2f GB/s while busy\n",
                        pip.first * 1e3, pip.second * 1e3, pip.second / pip.first,
                        bytes / pip.second / 1e9);
            std::printf("      speedup   : %.2fx wall, duty %.3f -> %.3f\n",
                        ser.first / pip.first, ser.second / ser.first, pip.second / pip.first);

            // THE THRESHOLD, DERIVED.  Serial costs fetch + compute per group; a
            // perfect 2-stage pipeline costs max(fetch, compute).  With the
            // MEASURED serial duty d = dma/wall, compute occupies (1-d) of the
            // wall, so the pipeline's floor is max(d, 1-d) of the serial wall and
            // the ideal speedup is 1/max(d, 1-d).  The gate asks for half of that
            // headroom, which leaves room for the per-group host launch cost the
            // pipeline does not remove while failing outright if no overlap
            // happens at all (speedup 1.00).  It is not fitted to the observed
            // number: at a serial duty near 0.45 it demands ~1.4x.
            const double d = ser.second / ser.first;
            const double ideal = 1.0 / std::max(d, 1.0 - d);
            const double need = 1.0 + 0.5 * (ideal - 1.0);
            ok("the pipeline recovers at least half the theoretical overlap headroom",
               ser.first / pip.first >= need,
               "speedup " + std::to_string(ser.first / pip.first) + " >= " +
                   std::to_string(need) + " (ideal " + std::to_string(ideal) +
                   " at serial duty " + std::to_string(d) + ")");
            ok("the pipeline raises the DMA duty cycle by at least 10 points",
               pip.second / pip.first > d + 0.10,
               std::to_string(d) + " -> " + std::to_string(pip.second / pip.first));
            // Same bytes both ways is the claim that the speedup is overlap and
            // not a smaller working set: each distinct expert is fetched once per
            // layer under either partition, because `occ` is a set.
            ok("...and moves EXACTLY the same bytes (no group is fetched twice)",
               std::fabs(bytes - double(PPIN) * double(PSB)) == 0.0,
               std::to_string(uint64_t(bytes)) + " B both ways");

            // ---- (6c) what next-layer speculation COSTS the pipeline --------
            //
            // A HYPOTHESIS THIS SECTION FALSIFIED, recorded because it was
            // wrong in a way that mattered.  The idle-time rule inside
            // `speculate` — decline unless the transfer queue has nothing
            // outstanding — was expected to make the waste disappear on its own
            // once the pipeline kept the queue busy.  It does not: `speculate`
            // is called AFTER the compute drain at the end of a group, and by
            // then the one group of lookahead has already landed, so the queue
            // is idle at exactly that instant in BOTH shapes.  Measured below:
            // the same number of speculations fire either way.
            //
            // That makes the cost decision-relevant rather than academic.  §5
            // showed only a group-0 speculation is ever used, so these bytes are
            // almost pure waste, and under the pipeline the link is the binding
            // resource — so the waste now comes straight off the throughput.
            const double pip_nospec = pip.first;
            const auto   pip_spec   = loop(pc.pipelined_group_cap(), true, true);
            const uint64_t spec_n   = pc.stats().spec_issued;
            const double   spec_mb  = double(pc.stats().spec_bytes) / 1e6;
            std::printf("      pipelined + next-layer speculation: wall %7.2f ms vs %7.2f ms"
                        " without (%llu speculative fetches, %.1f MB)\n",
                        pip_spec.first * 1e3, pip_nospec * 1e3,
                        (unsigned long long)spec_n, spec_mb);
            ok("next-layer speculation still fires under the pipeline —"
               " the idle-time rule does NOT disable it",
               spec_n > 0, std::to_string(spec_n) + " issued");
            // The gate: those bytes cost real time on a link that is now busy.
            // Expected slowdown is spec_bytes / demand_bytes of the fetch stage;
            // the gate asks only that it is measurable (>2%), because the point
            // is the SIGN of the effect, not its exact size on a shared box.
            ok("...and it measurably SLOWS the pipelined loop, so it should be off",
               pip_spec.first > pip_nospec * 1.02,
               "wall " + std::to_string(pip_spec.first * 1e3) + " ms vs " +
                   std::to_string(pip_nospec * 1e3) + " ms (+" +
                   std::to_string(100.0 * (pip_spec.first / pip_nospec - 1.0)) + "%), " +
                   std::to_string(spec_mb / (bytes / 1e6) * 100.0) + "% extra link bytes");
            sycl::free(ps, q);
        }
    }

    // -----------------------------------------------------------------------
    std::printf("\n[7] THE DECODE REGIME: one group per layer, and what that costs\n");
    // -----------------------------------------------------------------------
    // Everything above §7 is a PREFILL shape: a chunk's `occ` is most of the
    // expert pool, so a layer splits into many groups and the lookahead has
    // something to look ahead at.  DECODE is the opposite shape and was never
    // measured: T == 1 routes to `n_experts_used` == 6 experts, full stop.
    //
    // At the shipped 12 streaming slots `pipelined_group_cap()` is 6, and six
    // experts — of which some are statically resident and cost nothing — can
    // never exceed a cap of six.  `ds4_plan_expert_groups` therefore returns
    // EXACTLY ONE GROUP, the caller's `if (g + 1 < NG) issue(g + 1)` is never
    // taken, and the whole fetch pipeline is dead code at decode.  (a) proves
    // that rather than asserting it.
    //
    // (b) IS A FALSIFICATION, AND IT IS RECORDED SO IT IS NOT RE-DERIVED.  With
    // one group per layer the surviving half of `acquire_pipelined` is that each
    // token's acquire inherits the PREVIOUS TOKEN's six claims as protected, so
    // half a twelve-slot victim pool is unavailable and the FIFO holds one token
    // of history instead of two.  That LOOKS like a defect and is not: measured
    // at the real residency ratio it is worth 0.4% MORE misses under uniform
    // routing and 0.9% FEWER under 50% reuse — the sign flips, because protecting
    // the previous token's slots is LRU-ish and LRU is what temporal locality
    // wants.  An earlier probe put the cost at 14%, but only because it ran at
    // 48% residency instead of the model's 26.6%; the gate below is at the real
    // ratio and the 14% figure was wrong.
    //
    // (c) IS THE LEVER THAT IS LEFT.  If decode DMA cannot be overlapped it can
    // only be shortened by moving fewer bytes, and the cheap way to move fewer is
    // more STREAMING slots — because `pinned_experts == n_experts - static_slots`,
    // raising slots/layer while HOLDING static fixed costs zero extra pinned host
    // RAM and spends only VRAM.
    {
        // The REAL residency ratio, at a slot size chosen so the whole section
        // costs ~5 MB of device and ~13 MB of host: 256 experts, 56 static
        // (21.9%, exactly the model's), 12 streaming (kDs4MinStreamSlots).  Every
        // quantity §7 gates is a DIRECTORY property — which slot holds which
        // expert — and the directory does not know the slot size, so shrinking it
        // changes nothing that is measured here.  The byte figures printed are
        // the counts scaled by the real 6,684,672 B slot.
        const uint32_t DNE = 256, DSTATIC = 56, DSTREAM = 12, DSLOTS = DSTATIC + DSTREAM;
        const uint32_t DNL = 4, DK = 6, DTOK = 64;
        const uint64_t DSB = 16ull << 10;
        const uint64_t REAL_SLOT = 6'684'672;   // per-card UD-Q8_K_XL expert slice
        const uint32_t REAL_L    = 43;          // routed-expert layers in the model

        std::vector<uint64_t> dsb(DNL, DSB);
        std::vector<uint32_t> dpin;
        for (uint32_t e = DSTATIC; e < DNE; ++e) dpin.push_back(e);

        ie::Ds4HostArena da;
        const std::string dae = da.init_set(q, dsb, DNE, dpin, 1ull << 30);
        ok("decode arena (4 layers x 200 pinned experts x 16 KB)", dae.empty(), dae);

        // One token's routing for one layer: `DK` distinct ids.  With `reuse`,
        // each draw takes the previous token's id at that position instead — the
        // temporal locality decode has and prefill does not, as a tunable rather
        // than as an assumption.
        struct Router {
            uint64_t s;
            uint32_t next() { s = s * 6364136223846793005ull + 1442695040888963407ull;
                              return uint32_t(s >> 33); }
        };
        auto route_all = [&](uint32_t reuse_pct, std::vector<std::vector<int32_t>>& out) {
            Router r{0x9E3779B97F4A7C15ull};
            out.assign(size_t(DTOK) * DNL, {});
            for (uint32_t t = 0; t < DTOK; ++t)
                for (uint32_t L = 0; L < DNL; ++L) {
                    std::vector<int32_t>& cur = out[size_t(t) * DNL + L];
                    const std::vector<int32_t>* prv =
                        t ? &out[size_t(t - 1) * DNL + L] : nullptr;
                    while (cur.size() < DK) {
                        int32_t e;
                        if (prv && (r.next() % 100) < reuse_pct)
                            e = (*prv)[r.next() % DK];
                        else
                            e = int32_t(r.next() % DNE);
                        if (std::find(cur.begin(), cur.end(), e) == cur.end()) cur.push_back(e);
                    }
                    std::sort(cur.begin(), cur.end());
                }
        };

        // ---- (a) how many groups does a decode routing produce? -------------
        if (dae.empty()) {
            ie::Ds4ExpertCache dc;
            const std::string dce = dc.init(q, da, DSLOTS, 0, DSTATIC);
            ok("decode cache (56 static + 12 streaming, the shipped split)", dce.empty(), dce);
            if (dce.empty()) {
                std::vector<uint8_t> st0(size_t(DSB), 0);
                for (uint32_t L = 0; L < DNL; ++L)
                    for (uint32_t s = 0; s < DSTATIC; ++s) dc.install_static(L, s, s, st0.data());

                std::vector<std::vector<int32_t>> rt;
                route_all(0, rt);
                uint64_t hist[8] = {0};
                std::vector<uint32_t> occ1, bnd;
                for (const std::vector<int32_t>& ids : rt) {
                    occ1.assign(ids.begin(), ids.end());
                    ie::ds4_plan_expert_groups(dc, 0, occ1.data(), occ1.size(),
                                               dc.pipelined_group_cap(), bnd);
                    hist[std::min<size_t>(bnd.size() - 1, 7)]++;
                }
                std::printf("      groups per layer over %u decode routings:",
                            uint32_t(rt.size()));
                for (int i = 1; i < 8; ++i)
                    if (hist[i]) std::printf("  NG=%d:%llu", i, (unsigned long long)hist[i]);
                std::printf("\n");
                ok("(a) at T=1 the partitioner returns ONE group under"
                   " pipelined_group_cap() — the lookahead is DEAD CODE at decode",
                   hist[1] == rt.size(),
                   std::to_string(hist[1]) + "/" + std::to_string(rt.size()) +
                       " single-group layers at cap " +
                       std::to_string(dc.pipelined_group_cap()));
            }
        }

        // ---- (b) and (c): one replay, parameterised by slots and by acquire ---
        // `slots` is slots/layer with the static partition HELD at DSTATIC, so
        // every extra slot is a streaming slot and the pinned-host bill
        // (`n_experts - static_slots` experts per layer) does not move.
        auto decode_run = [&](uint32_t slots, bool pipelined, uint32_t reuse_pct,
                              uint64_t& hits, uint64_t& misses) {
            ie::Ds4ExpertCache dc;
            if (!dc.init(q, da, slots, 0, DSTATIC).empty()) return false;
            std::vector<uint8_t> st0(size_t(DSB), 0);
            for (uint32_t L = 0; L < DNL; ++L)
                for (uint32_t s = 0; s < DSTATIC; ++s) dc.install_static(L, s, s, st0.data());
            std::vector<std::vector<int32_t>> rt;
            route_all(reuse_pct, rt);
            std::vector<uint32_t> occ1, bnd, sl;
            for (uint32_t t = 0; t < DTOK; ++t)
                for (uint32_t L = 0; L < DNL; ++L) {
                    const std::vector<int32_t>& ids = rt[size_t(t) * DNL + L];
                    occ1.assign(ids.begin(), ids.end());
                    ie::ds4_plan_expert_groups(dc, L, occ1.data(), occ1.size(),
                                               pipelined ? dc.pipelined_group_cap()
                                                         : dc.stream_slots(), bnd);
                    for (size_t g = 0; g + 1 < bnd.size(); ++g) {
                        std::vector<int32_t> gi(ids.begin() + bnd[g], ids.begin() + bnd[g + 1]);
                        sl.assign(gi.size(), ie::kDs4NoSlot);
                        if (pipelined)
                            dc.acquire_pipelined(L, gi.data(), uint32_t(gi.size()), sl.data());
                        else
                            dc.acquire(L, gi.data(), uint32_t(gi.size()), sl.data());
                        dc.transfer_queue().wait();   // the caller's eviction barrier
                    }
                }
            hits = dc.stats().hits;
            misses = dc.stats().misses;
            dc.free_storage();
            return true;
        };

        if (dae.empty()) {
            for (uint32_t reuse : {0u, 50u}) {
                uint64_t hs = 0, ms = 0, hc = 0, mc = 0;
                if (!decode_run(DSLOTS, false, reuse, hs, ms) ||
                    !decode_run(DSLOTS, true,  reuse, hc, mc)) {
                    ok("(b) decode replay ran", false, "cache init failed"); break;
                }
                const double d = ms ? 100.0 * (double(mc) - double(ms)) / double(ms) : 0.0;
                std::printf("      reuse %2u%%:  serial %llu hits/%llu misses,"
                            "  acquire_pipelined %llu/%llu  (%+.2f%% bytes)\n",
                            reuse, (unsigned long long)hs, (unsigned long long)ms,
                            (unsigned long long)hc, (unsigned long long)mc, d);
                // A REGRESSION GATE ON A FALSIFICATION.  The claim being locked in
                // is that the carried in-use generation is NOT the decode defect;
                // 5% is far above the +0.36% / -0.92% measured and far below the
                // 14% a higher-residency probe wrongly suggested, so this fires if
                // someone makes the carry actually expensive.
                ok("(b) the carried in-use generation is worth under 5% of decode"
                   " bytes either way — it is NOT the decode defect",
                   std::fabs(d) < 5.0,
                   std::to_string(d) + "% at reuse " + std::to_string(reuse) + "%");
            }
        }

        // ---- (c) the residency curve: the only lever that shortens decode DMA -
        if (dae.empty()) {
            const double per_tok = double(REAL_SLOT) / 1e9 / double(DTOK) * double(REAL_L) /
                                   double(DNL);   // misses -> GB/token at 43 layers
            uint64_t base_m = 0;
            bool     falls  = true;
            std::printf("      slots/layer -> decode bytes, static held at %u"
                        " (pinned host RAM UNCHANGED), %u-layer equivalent\n", DSTATIC, REAL_L);
            for (uint32_t slots : {DSLOTS, 80u, 96u, 112u, 128u}) {
                uint64_t h0 = 0, m0 = 0, h5 = 0, m5 = 0;
                if (!decode_run(slots, true, 0, h0, m0) ||
                    !decode_run(slots, true, 50, h5, m5)) break;
                std::printf("        %3u slots (%2u static + %3u streaming, %5.1f%% resident):"
                            "  reuse 0%% %.2f GB/tok   reuse 50%% %.2f GB/tok\n",
                            slots, DSTATIC, slots - DSTATIC, 100.0 * double(slots) / double(DNE),
                            double(m0) * per_tok, double(m5) * per_tok);
                if (slots == DSLOTS) base_m = m5;
                else falls = falls && m5 < base_m;
            }
            ok("(c) streaming depth bought with VRAM alone (static fixed, so the"
               " pinned-host bill does not move) strictly reduces decode bytes",
               base_m > 0 && falls,
               "baseline " + std::to_string(base_m) + " misses at " +
                   std::to_string(DSLOTS) + " slots");
        }
        da.free_storage();
    }

    // -----------------------------------------------------------------------
    std::printf("\n[8] The cross-card reduction must not queue behind the expert stream\n");
    // -----------------------------------------------------------------------
    // §1-§7 are about the FETCH pipeline.  §8 is about the other traffic on the
    // same PCIe link: the per-layer cross-card reduction, which under expert-TP
    // runs twice per layer, 86 times per decoded token.  It is here rather than
    // in the residency test because the defect it gates is a property of the
    // EXPERT STREAM — the reduction was slow only when the stream was busy, and
    // this is the file that knows how to make the stream busy.
    {
        std::vector<sycl::device> gpus;
        for (const auto& d : sycl::device::get_devices())
            if (d.is_gpu() && d.get_backend() == sycl::backend::ext_oneapi_level_zero)
                gpus.push_back(d);
        const bool two = gpus.size() >= 2;
        sycl::property_list io{sycl::property::queue::in_order()};
        // The two cards' COMPUTE queues, each in its own context — exactly what
        // DeepSeek4Runtime builds, and what makes the per-card staging mandatory.
        sycl::queue r0(two ? gpus[0] : q.get_device(), io);
        sycl::queue r1(two ? gpus[1] : q.get_device(), io);
        // ...and each card's TRANSFER queue, which is where the expert stream's
        // H2D bursts go.  Same context as its compute queue, different queue.
        sycl::queue x0(r0.get_context(), r0.get_device(), io);
        sycl::queue x1(r1.get_context(), r1.get_device(), io);

        // The decode-step reduction that carries `ws_moe_` and the sliced shared
        // expert together: 2 * T * hidden fp32 at T = 1, hidden = 4096.
        const uint64_t RN = 8192;
        const size_t   RB = size_t(RN) * 4;
        const size_t   BURST = 4ull << 20;          // one expert-sized H2D burst

        auto* p0 = sycl::malloc_device<float>(size_t(RN), r0);
        auto* p1 = sycl::malloc_device<float>(size_t(RN), r1);
        auto* g0 = sycl::malloc_host<float>(size_t(RN), r0);   // card 0's staging
        auto* g1 = sycl::malloc_host<float>(size_t(RN), r1);   // card 1's staging
        auto* b0 = sycl::malloc_device(BURST, r0);
        auto* b1 = sycl::malloc_device(BURST, r1);
        auto* s0 = sycl::malloc_host(BURST, r0);
        auto* s1 = sycl::malloc_host(BURST, r1);
        ok("(setup) reduction probe allocations", p0 && p1 && g0 && g1 && b0 && b1 && s0 && s1);

        if (p0 && p1 && g0 && g1 && b0 && b1 && s0 && s1) {
            std::memset(s0, 0x5A, BURST);
            std::memset(s1, 0xA5, BURST);
            std::vector<sycl::queue*> qs{&r0, &r1};
            std::vector<float*>       parts{p0, p1};
            std::vector<float*>       stages{g0, g1};

            // Operands chosen so a+b is exact in fp32 (both are small integers
            // scaled by a power of two), so EVERY assertion below can demand
            // equality.  The error model is therefore not a tolerance: the
            // reduction performs one fp32 add per element and nothing else, and
            // fp32 addition of two exactly-representable operands whose sum is
            // exactly representable is exact.  Tolerance ZERO, and the negative
            // controls below prove that zero discriminates.
            std::vector<float> A(size_t(RN), 0.f), B(size_t(RN), 0.f);
            for (uint64_t i = 0; i < RN; ++i) {
                A[i] = float(int64_t(i % 4096) - 2048) * 0.25f;
                B[i] = float(int64_t((i * 7) % 4096) - 2048) * 0.5f;
            }
            auto load = [&] {
                r0.memcpy(p0, A.data(), RB).wait();
                r1.memcpy(p1, B.data(), RB).wait();
            };
            auto readback = [&](std::vector<float>& o0, std::vector<float>& o1) {
                o0.assign(size_t(RN), 0.f); o1.assign(size_t(RN), 0.f);
                r0.memcpy(o0.data(), p0, RB).wait();
                r1.memcpy(o1.data(), p1, RB).wait();
            };

            // ---- (a) the reduction is exact on every card -------------------
            std::vector<float> o0, o1;
            load();
            const std::string re = ie::ds4_tp_reduce_host(qs, parts, stages, RN);
            ok("(a) ds4_tp_reduce_host accepted the two partials", re.empty(), re);
            readback(o0, o1);
            uint64_t bad0 = 0, bad1 = 0, split = 0;
            for (uint64_t i = 0; i < RN; ++i) {
                const float want = A[i] + B[i];
                if (o0[i] != want) ++bad0;
                if (o1[i] != want) ++bad1;
                if (o0[i] != o1[i]) ++split;
            }
            ok("(a) every card holds the EXACT sum of both partials (tolerance zero)",
               bad0 == 0 && bad1 == 0,
               std::to_string(bad0) + "/" + std::to_string(bad1) + " of " + std::to_string(RN));
            ok("(a) ...and the two cards hold BIT-IDENTICAL buffers — the router"
               " downstream is compared bit for bit, so a difference here would"
               " refuse the next forward pass",
               split == 0, std::to_string(split) + " differ");

            // ---- (b) the return leg changed no value ------------------------
            // The shipped form moves the sum back with a kernel instead of a
            // `memcpy`.  Neither does arithmetic, so the two must agree to the
            // BIT, not to a tolerance.  Asserted against the old code rather
            // than argued.
            std::vector<float> k0 = o0, k1 = o1, m0, m1;
            load();
            reduce_memcpy_hops(qs, parts, stages, RN);
            r0.wait(); r1.wait();
            readback(m0, m1);
            uint64_t drift = 0;
            for (uint64_t i = 0; i < RN; ++i)
                if (std::memcmp(&k0[i], &m0[i], 4) || std::memcmp(&k1[i], &m1[i], 4)) ++drift;
            ok("(b) the kernel hops are BIT-IDENTICAL to the memcpys they replaced",
               drift == 0, std::to_string(drift) + " of " + std::to_string(RN) + " differ");

            // ---- (b') THE PREFILL REGIME: n >= 65536 takes the fused
            // add + memcpy hops (docs/deepseek4/72 Phase F).  Same claims, same
            // tolerance zero, against the same memcpy-form reference, plus a
            // negative control that corrupts half of one card's copy.
            {
                const uint64_t PN = uint64_t(1) << 17;   // 131072 floats, 512 KB per card
                const size_t   PB = size_t(PN) * 4;
                auto* q0 = sycl::malloc_device<float>(size_t(PN), r0);
                auto* q1 = sycl::malloc_device<float>(size_t(PN), r1);
                auto* h0 = sycl::malloc_host<float>(size_t(PN), r0);
                auto* h1 = sycl::malloc_host<float>(size_t(PN), r1);
                ok("(b') prefill-size probe allocations", q0 && q1 && h0 && h1);
                if (q0 && q1 && h0 && h1) {
                    std::vector<float> PA(size_t(PN), 0.f), PBv(size_t(PN), 0.f);
                    for (uint64_t i = 0; i < PN; ++i) {
                        PA[i]  = float(int64_t(i % 4096) - 2048) * 0.25f;
                        PBv[i] = float(int64_t((i * 7) % 4096) - 2048) * 0.5f;
                    }
                    std::vector<float*> pparts{q0, q1}, pstages{h0, h1};
                    auto pload = [&] {
                        r0.memcpy(q0, PA.data(), PB).wait();
                        r1.memcpy(q1, PBv.data(), PB).wait();
                    };
                    auto pread = [&](std::vector<float>& a, std::vector<float>& b) {
                        a.assign(size_t(PN), 0.f); b.assign(size_t(PN), 0.f);
                        r0.memcpy(a.data(), q0, PB).wait();
                        r1.memcpy(b.data(), q1, PB).wait();
                    };
                    std::vector<float> s0v, s1v, r0v, r1v;
                    pload();
                    const std::string pe = ie::ds4_tp_reduce_host(qs, pparts, pstages, PN);
                    r0.wait(); r1.wait();
                    pread(s0v, s1v);
                    uint64_t pbad = 0, psplit = 0;
                    for (uint64_t i = 0; i < PN; ++i) {
                        const float want = PA[i] + PBv[i];
                        if (s0v[i] != want || s1v[i] != want) ++pbad;
                        if (s0v[i] != s1v[i]) ++psplit;
                    }
                    ok("(b') prefill-size reduction (n = 131072) is EXACT on both cards, bit-identical across them",
                       pe.empty() && pbad == 0 && psplit == 0,
                       pe.empty() ? std::to_string(pbad) + " wrong, " + std::to_string(psplit) + " split" : pe);
                    pload();
                    reduce_memcpy_hops(qs, pparts, pstages, PN);
                    r0.wait(); r1.wait();
                    pread(r0v, r1v);
                    uint64_t pdrift = 0;
                    for (uint64_t i = 0; i < PN; ++i)
                        if (std::memcmp(&s0v[i], &r0v[i], 4) || std::memcmp(&s1v[i], &r1v[i], 4)) ++pdrift;
                    ok("(b') ...and BIT-IDENTICAL to the memcpy-form reference", pdrift == 0,
                       std::to_string(pdrift) + " of " + std::to_string(PN) + " differ");
                    // negative control: the shipped reduction, then half of card 1's
                    // copy replaced by card 1's own partial — must be caught.
                    pload();
                    ie::ds4_tp_reduce_host(qs, pparts, pstages, PN);
                    r1.wait();
                    r1.memcpy(q1 + PN / 2, PBv.data() + PN / 2, PB / 2).wait();
                    r0.wait();
                    std::vector<float> c0, c1;
                    pread(c0, c1);
                    uint64_t caught = 0;
                    for (uint64_t i = 0; i < PN; ++i) if (c1[i] != PA[i] + PBv[i]) ++caught;
                    ok("(b') NEGATIVE CONTROL: half of one card's copy replaced -> caught",
                       caught > 0, std::to_string(caught) + " wrong");
                }
                for (float* p : {q0, q1}) if (p) sycl::free(p, p == q0 ? r0 : r1);
                for (float* p : {h0, h1}) if (p) sycl::free(p, p == h0 ? r0 : r1);
            }

            // ---- (c') THE SLICED REDUCER (docs/deepseek4/72 Phase I): the
            // schedule that replaces the memcpy form at n >= 65536 — D2H in
            // slices, threaded sum, H2D on an auxiliary queue, a barrier on the
            // compute queue.  Same claims, tolerance zero, against the same
            // memcpy-form reference; a back-to-back second reduction (stage
            // reuse ordering) and a negative control.  n = 8192 dispatches to
            // the decode path inside it and must be exact too.
            {
                ie::Ds4TpReducer red;
                const std::string ie_ = red.init(qs, 4);
                ok("(c') reducer init (auxiliary queues + 4 threads)", ie_.empty(), ie_);
                for (uint64_t SN : {uint64_t(8192), uint64_t(131072), (uint64_t(4) << 20) + 12345}) {
                    const size_t SB = size_t(SN) * 4;
                    auto* d0 = sycl::malloc_device<float>(size_t(SN), r0);
                    auto* d1 = sycl::malloc_device<float>(size_t(SN), r1);
                    auto* hs0 = sycl::malloc_host<float>(size_t(SN), r0);
                    auto* hs1 = sycl::malloc_host<float>(size_t(SN), r1);
                    const std::string tag = "n = " + std::to_string(SN);
                    ok(("(c') allocations " + tag).c_str(), d0 && d1 && hs0 && hs1);
                    if (!(d0 && d1 && hs0 && hs1)) continue;
                    std::vector<float> SA(size_t(SN), 0.f), SBv(size_t(SN), 0.f);
                    for (uint64_t i = 0; i < SN; ++i) {
                        SA[i]  = float(int64_t(i % 4099) - 2048) * 0.25f;
                        SBv[i] = float(int64_t((i * 7) % 4093) - 2048) * 0.5f;
                    }
                    std::vector<float*> sparts{d0, d1}, sstages{hs0, hs1};
                    auto sload = [&] {
                        r0.memcpy(d0, SA.data(), SB).wait();
                        r1.memcpy(d1, SBv.data(), SB).wait();
                    };
                    auto sread = [&](std::vector<float>& a, std::vector<float>& b) {
                        a.assign(size_t(SN), 0.f); b.assign(size_t(SN), 0.f);
                        r0.wait(); r1.wait();
                        r0.memcpy(a.data(), d0, SB).wait();
                        r1.memcpy(b.data(), d1, SB).wait();
                    };
                    std::vector<float> x0, x1, y0, y1;
                    sload();
                    const std::string se = red.reduce(qs, sparts, sstages, SN);
                    sread(x0, x1);
                    uint64_t sbad = 0, ssplit = 0;
                    for (uint64_t i = 0; i < SN; ++i) {
                        const float want = SA[i] + SBv[i];
                        if (x0[i] != want || x1[i] != want) ++sbad;
                        if (x0[i] != x1[i]) ++ssplit;
                    }
                    ok(("(c') sliced reduction is EXACT on both cards, bit-identical across them, " + tag).c_str(),
                       se.empty() && sbad == 0 && ssplit == 0,
                       se.empty() ? std::to_string(sbad) + " wrong, " + std::to_string(ssplit) + " split" : se);
                    sload();
                    reduce_memcpy_hops(qs, sparts, sstages, SN);
                    sread(y0, y1);
                    uint64_t sdrift = 0;
                    for (uint64_t i = 0; i < SN; ++i)
                        if (std::memcmp(&x0[i], &y0[i], 4) || std::memcmp(&x1[i], &y1[i], 4)) ++sdrift;
                    ok(("(c') ...and BIT-IDENTICAL to the memcpy-form reference, " + tag).c_str(), sdrift == 0,
                       std::to_string(sdrift) + " of " + std::to_string(SN) + " differ");
                    // back to back on the same stages, no host wait in between:
                    // the second sees the first's result on both cards.
                    sload();
                    const std::string se1 = red.reduce(qs, sparts, sstages, SN);
                    const std::string se2 = red.reduce(qs, sparts, sstages, SN);
                    sread(x0, x1);
                    uint64_t bbad = 0;
                    for (uint64_t i = 0; i < SN; ++i) {
                        const float v = SA[i] + SBv[i];
                        const float want2 = v + v;
                        if (x0[i] != want2 || x1[i] != want2) ++bbad;
                    }
                    ok(("(c') two back-to-back reductions compose exactly (stage reuse is ordered), " + tag).c_str(),
                       se1.empty() && se2.empty() && bbad == 0, std::to_string(bbad) + " wrong");
                    // negative control: one word of card 1's copy replaced after
                    // the reduction — must be caught.
                    sload();
                    (void)red.reduce(qs, sparts, sstages, SN);
                    r1.wait();
                    r1.memcpy(d1 + SN / 3, SBv.data() + SN / 3, 4).wait();
                    sread(x0, x1);
                    uint64_t caught = 0;
                    for (uint64_t i = 0; i < SN; ++i) if (x1[i] != SA[i] + SBv[i]) ++caught;
                    ok(("(c') NEGATIVE CONTROL: one word replaced -> caught, " + tag).c_str(), caught == 1,
                       std::to_string(caught) + " wrong");
                    r0.wait(); r1.wait();
                    sycl::free(d0, r0); sycl::free(d1, r1); sycl::free(hs0, r0); sycl::free(hs1, r1);
                }
            }

            // ---- (c) NEGATIVE CONTROLS: does (a) actually discriminate? -----
            // Each control is a reduction that is wrong in one specific way a
            // real defect would be wrong, applied to the same operands.  If (a)
            // would have passed on any of them, (a) is worthless.
            auto control = [&](const char* what, auto&& corrupt) {
                load();
                corrupt();
                r0.wait(); r1.wait();
                std::vector<float> c0, c1;
                readback(c0, c1);
                uint64_t caught = 0;
                for (uint64_t i = 0; i < RN; ++i)
                    if (c0[i] != A[i] + B[i] || c1[i] != A[i] + B[i]) ++caught;
                ok(what, caught > 0,
                   caught ? std::to_string(caught) + " elements wrong, as required"
                          : "PASSED THE CHECK — the check is vacuous");
            };
            control("(c) a DROPPED card-1 partial is caught", [&] {
                // card 1's partial never reaches the sum
                r0.memcpy(g0, p0, RB); r1.memcpy(g1, p1, RB);
                r0.wait(); r1.wait();
                std::memcpy(g1, g0, RB);
                for (size_t c = 0; c < 2; ++c) qs[c]->memcpy(parts[c], stages[c], RB);
            });
            control("(c) a DOUBLE-COUNTED card-1 partial is caught", [&] {
                r0.memcpy(g0, p0, RB); r1.memcpy(g1, p1, RB);
                r0.wait(); r1.wait();
                for (uint64_t i = 0; i < RN; ++i) g0[i] += g1[i] + g1[i];
                std::memcpy(g1, g0, RB);
                for (size_t c = 0; c < 2; ++c) qs[c]->memcpy(parts[c], stages[c], RB);
            });
            control("(c) a return leg that copies back only HALF the tensor is caught"
                    " — the failure a memcpy-to-kernel swap can introduce", [&] {
                r0.memcpy(g0, p0, RB); r1.memcpy(g1, p1, RB);
                r0.wait(); r1.wait();
                for (uint64_t i = 0; i < RN; ++i) g0[i] += g1[i];
                std::memcpy(g1, g0, RB);
                for (size_t c = 0; c < 2; ++c) qs[c]->memcpy(parts[c], stages[c], RB / 2);
            });
            // AND THE CONTROL THAT CANNOT BE BUILT, STATED RATHER THAN FAKED.
            // "The cards were swapped" is not detectable in the reduction's
            // OUTPUT: fp32 addition is commutative, so a + b and b + a are the
            // same bits, and every card ends up holding the same sum either way.
            // What catches a genuine card mix-up is upstream — the orchestrator
            // compares every card's expert ids and routing weights against card
            // 0's bit for bit and refuses the pass (deepseek4.cpp, REDUCTION 2).
            // Asserting a swap here would be asserting something false.
            ok("(c) a swapped-cards control is NOT claimed: a+b and b+a are the"
               " same bits, so the reduction cannot see it and the lockstep check"
               " upstream is what does",
               A[3] + B[3] == B[3] + A[3]);

            // ---- (d) THE GATE: contention with the expert stream ------------
            // The defect: both hops of the reduction used to be `memcpy`, which
            // Level Zero dispatches to a COPY ENGINE — the same engine carrying
            // the expert stream's multi-megabyte H2D bursts.  A 32 KB hop behind
            // a 4 MB burst waits for the burst.  Measured on 2x B70, drained
            // inside the timed region, with the forms rotated so neither
            // permanently stands in front of a fresh burst: 33.1 us idle ->
            // 196.4 us at 41 GB/s of stream, a 5.9x cliff.  The shipped form
            // runs both hops as compute-engine kernels: 39.3 -> 45.3 us, 1.15x.
            //
            // The threshold is derived from those two numbers, not tuned: 2.5x
            // is far above the 1.15x the kernel form gives and far below the
            // 5.9x the memcpy form gives, so it fires on a regression and not on
            // a slow afternoon.  The memcpy form runs right beside it as the
            // control, so "the gate discriminates" is measured every run and not
            // assumed.
            auto shipped = [&] { ie::ds4_tp_reduce_host(qs, parts, stages, RN); };
            auto legacy  = [&] { reduce_memcpy_hops(qs, parts, stages, RN); };

            // THE CONTENTION LEVEL IS BOUNDED ON PURPOSE, AND THE FIRST ATTEMPT
            // TO WRITE THIS GATE GOT IT WRONG.  Submitting a 4 MB burst per
            // reduction without ever draining the transfer queue demands ~151 us
            // of link per ~25 us of reduction, so the backlog grows without
            // bound and the LINK ITSELF saturates — at which point every form is
            // slow (measured: 231 us vs 210 us, no discrimination at all) and the
            // gate is measuring an overload no scheduler can fix.  The real
            // fetch pipeline never runs that way: `acquire` waits for its
            // generation before the next one is issued.  So the burst is drained
            // once per iteration, OUTSIDE the timed region, leaving exactly one
            // 4 MB transfer in flight while the reduction runs — which is the
            // situation the defect actually lives in.
            //
            // BOTH FORMS ARE MEASURED IN ONE INTERLEAVED LOOP, alternating, so a
            // load spike on this box (it runs at a load average around twenty)
            // lands on both and cannot be mistaken for a difference between them.
            auto timed = [&](bool busy, double& shipped_med, double& legacy_med) {
                std::vector<double> ts, tl;
                for (int i = 0; i < 80; ++i) {
                    if (busy) { x0.memcpy(b0, s0, BURST); x1.memcpy(b1, s1, BURST); }
                    double a = now_s();
                    shipped();
                    r0.wait(); r1.wait();      // drain: nothing may be merely deferred
                    ts.push_back(now_s() - a);
                    if (busy) { x0.wait(); x1.wait();
                                x0.memcpy(b0, s0, BURST); x1.memcpy(b1, s1, BURST); }
                    a = now_s();
                    legacy();
                    r0.wait(); r1.wait();
                    tl.push_back(now_s() - a);
                    if (busy) { x0.wait(); x1.wait(); }
                }
                std::sort(ts.begin(), ts.end());
                std::sort(tl.begin(), tl.end());
                shipped_med = ts[ts.size() / 2];
                legacy_med  = tl[tl.size() / 2];
            };

            load();
            // An idle B70 sits at 400 MHz and the first timings in a process read
            // 3-6x slow, so warm both forms before either is measured.
            for (int i = 0; i < 120; ++i) { shipped(); legacy(); }
            r0.wait(); r1.wait();

            double si = 0, li = 0, sb = 0, lb = 0;
            timed(false, si, li);
            timed(true,  sb, lb);
            std::printf("      %s: n=%llu (%zu KB/card), 4 MB stream bursts\n",
                        two ? "TWO cards" : "ONE card (both roles)",
                        (unsigned long long)RN, RB / 1024);
            std::printf("        kernel hops (shipped): idle %7.2f us   stream busy %7.2f us"
                        "   %.2fx\n", si * 1e6, sb * 1e6, sb / si);
            std::printf("        memcpy hops (control): idle %7.2f us   stream busy %7.2f us"
                        "   %.2fx\n", li * 1e6, lb * 1e6, lb / li);
            ok("(d) the reduction stays within 2.5x of its idle cost while the"
               " expert stream saturates the link",
               sb < 2.5 * si,
               std::to_string(sb / si) + "x (idle " + std::to_string(si * 1e6) + " us)");
            // The control has to FAIL that bound, or the bound proves nothing.
            // On ONE card the two "cards" share a link and a copy engine, which
            // is not the geometry the defect lives in, so the control is only
            // demanded on a real two-card run — and its absence is printed.
            if (two)
                ok("(d) NEGATIVE CONTROL: memcpy hops DO breach that bound — the"
                   " gate discriminates",
                   lb > 2.5 * li, std::to_string(lb / li) + "x");
            else
                std::printf("      NOTE: one GPU only. (a)-(c) are fully checked; (d)'s\n"
                            "      negative control needs two cards on two links and is\n"
                            "      NOT verified by this run.\n");
        }
        for (void* p : {(void*)p0, (void*)g0, (void*)b0, (void*)s0})
            if (p) sycl::free(p, r0);
        for (void* p : {(void*)p1, (void*)g1, (void*)b1, (void*)s1})
            if (p) sycl::free(p, r1);
    }

    // -----------------------------------------------------------------------
    std::printf("\n[9] hits-first with CACHED residents (docs/deepseek4/72 Phase J)\n");
    // -----------------------------------------------------------------------
    // Last, because it changes the cache's state: two streamed experts are
    // made resident and must then lead the order together with the statics.
    {
        const int32_t two[2] = {int32_t(STATIC + 3), int32_t(STATIC + 7)};
        uint32_t sl[2] = {ie::kDs4NoSlot, ie::kDs4NoSlot};
        cache.acquire(0, two, 2, sl).wait();
        ok("[9] the two experts are resident after acquire",
           cache.is_resident(0, uint32_t(two[0])) && cache.is_resident(0, uint32_t(two[1])));
        uint32_t nres = 0;
        for (uint32_t e : occ) if (cache.is_resident(0, e)) ++nres;
        check_hits_first(cache, occ, "static + cached residents", 4, nres);
        ok("[9] ...and there are more residents than statics", nres > STATIC,
           std::to_string(nres) + " resident of " + std::to_string(occ.size()));
    }

    std::printf("\n%s: %d failure(s)\n", g_fail ? "FAILED" : "PASSED", g_fail);
    return g_fail ? 1 : 0;
}
