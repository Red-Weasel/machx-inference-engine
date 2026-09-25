// include/ie/mimo26_host_rules.hpp — MiMo-V2.6 serving rules that need no device (docs/mimo26/P7_FIX64_FIX70.md): the
// host-slot prefix cache's arithmetic and admission policy (#70) and the DFlash drafter's fp16 range check (#64). No SYCL:
// unit-tested on the CPU by tests/unit/mimo26_host_rules_test.cpp. Built into ie_core (src/model/mimo26_host_rules.cpp).
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ie {

// A ring of R slots holds position p at slot p % R. The slots of positions [p0, p1) -- only the last R of them when the
// range is longer -- as at most two runs [runs[2i], runs[2i + 1]). Returns the run count (0 for an empty range or R == 0).
uint32_t mimo26_ring_runs(uint32_t p0, uint32_t p1, uint32_t R, uint32_t runs[4]);

// How many leading positions of `ids` a state serves. The state holds `have` (the ids at its positions [0, have.size()))
// and wrote up to `written_end` (a verify's rejected rows go past have.size()) with SWA rings of `ring` slots (0 = linear
// caches) over a `window`: their common prefix, at most ids.size() - 1 (the last prompt token always runs, for its
// logits), and 0 when a ring no longer holds the window before it (a ring holds p only for p >= written_end - ring: P3b
// gate finding 3). The rule mimo26_run_ids applied to the live state since P3b, now shared by the host slots.
uint32_t mimo26_servable(const std::vector<int32_t>& have, uint32_t written_end, uint32_t ring, uint32_t window,
                         const std::vector<int32_t>& ids);

// Whether serving `served` positions of a state whose last prompt covered [0, prompt_end) CONTINUES that conversation --
// the request re-reads at most `slack` of that prompt (a new turn re-renders only the generated tail) -- or branches off
// it (prompt positions of that conversation past `served` would be lost).
bool mimo26_continues(uint32_t served, uint32_t prompt_end, uint32_t slack);

// A host slot as the admission policy sees it.
struct Mimo26SlotCost {
    uint64_t bytes = 0;   // host bytes held
    uint64_t tick = 0;    // last use (larger = more recent)
    uint32_t uses = 0;    // requests that continued this conversation; 0 = a one-shot state
    uint32_t tokens = 0;  // positions held: what a re-read of it would prefill
};

// Admitting a new slot of `bytes` whose conversation was continued `uses` times: `victims` = the slots to evict first so
// the store stays within `budget` bytes and MemAvailable (`avail`, read now) stays at or above `keep_free` with the new
// slot held (an evicted slot's bytes return to MemAvailable). Order: one-shot slots, cheapest to rebuild first -- fewest
// `tokens`, then least recently used (#77: the owner's long first turn is one-shot until its first continuation and must
// not go before the short side states that arrive after it); then -- only for a newcomer that was itself continued --
// continued slots, least recently used first (equal ticks: index order). Slot `keep` (-1 = none) is never a victim;
// with `keep_leaves` it is also left out of the BUDGET count (#78: a slot restored for a continuation is dropped once the
// prompt has run), though not out of MemAvailable, whose bytes it holds until then. Anything but kOk: the newcomer does
// not fit even with every slot it may evict gone -- larger than the whole budget, the budget held by slots it may not
// evict, or the MemAvailable floor -- and `victims` is empty (nothing is to be evicted).
enum class Mimo26Admit { kOk, kTooLarge, kBudget, kFloor };
Mimo26Admit mimo26_plan_admit(const std::vector<Mimo26SlotCost>& slots, uint64_t bytes, uint32_t uses, uint64_t budget,
                              uint64_t avail, uint64_t keep_free, int keep, std::vector<size_t>& victims,
                              bool keep_leaves = false);

// #87 (docs/mimo26/P7_FIX64_FIX70.md section 8): the prompt-end snapshot. A reply's decode overwrites the SWA rings, so
// after more than ring - window generated positions the state no longer serves its own prompt: a request that discards
// the reply (a length cut, an interrupt, an error, a regenerate) re-read everything. At the end of a prompt of P
// positions (the state written to `hi`) the rings' slots of positions [s0, P) are kept on the host, s0 = max(P - window -
// margin, hi - ring, 0); written back, they serve every divergence L in [s0 + window, P] (so [P - margin, P] when P is
// long enough). `hi_syn` = s0 + ring is the written_end that says exactly that to mimo26_servable and to
// Mimo26Forward::state_spans: the restored ring holds [hi_syn - ring, P) and no more. False = nothing to keep: no ring
// (linear SWA caches), an empty prompt, a margin of ring - window or more, or hi_syn beyond `max_ctx` -- then no reply the
// capacity allows can push the ring that far, and the live state itself serves [P - margin, P].
bool mimo26_prompt_snapshot(uint32_t P, uint32_t hi, uint32_t ring, uint32_t window, uint32_t margin, uint32_t max_ctx,
                            uint32_t& s0, uint32_t& hi_syn);

// The first element of x[0, n) that fp16 cannot hold -- NaN, +-inf, or a magnitude that rounds (to nearest even) to inf,
// i.e. >= 65520 -- or n when every element fits. Bit tests, not float compares: the host code builds under icpx's default
// fast fp model, where compares against NaN / inf may be folded away.
size_t mimo26_first_non_f16(const float* x, size_t n);

// A bit-level scan (no float compares, so the fast fp model cannot fold it away): how many of x[0, n) are non-finite
// (NaN, +-inf), the first of them, and the largest finite magnitude and where it is (-1 = none). fp16 input as raw bits.
struct Mimo26Scan { uint64_t non_finite = 0; int64_t first_bad = -1; float max_abs = 0.f; int64_t max_at = -1; };
Mimo26Scan mimo26_scan_f32(const float* x, size_t n);
Mimo26Scan mimo26_scan_f16(const uint16_t* x, size_t n);

}  // namespace ie
