// tests/unit/memory_plan_gate_test.cpp — pure predicate for the VRAM planner
// hard gate (Task 2). should_block_oom decides whether Engine::load must refuse
// an over-budget model instead of crashing deep in an OOM. Host-only: no GPU,
// no GGUF — it only reads fields of a fabricated PlacementPlan + an env flag.
#undef NDEBUG
#include "ie/memory_plan.hpp"

#include <cassert>
#include <cstdio>

using ie::PlacementPlan;
using ie::should_block_oom;

// Build a plan with the two fields the predicate reads (fits + forced). `forced`
// is now passed only to PROVE it is ignored by the gate — the predicate inspects
// fits plus the env flag arg (2026-07-08: forced no longer bypasses the fit gate).
static PlacementPlan mk(bool fits, bool forced) {
    PlacementPlan p;
    p.fits   = fits;
    p.forced = forced;
    return p;
}

int main() {
    // Truth table (2026-07-08): block == !allow_oom && !fits. `forced` (user
    // pinned --gpus) is NO LONGER part of the predicate — pinning the card count
    // does not consent to OOM; only IE_ALLOW_OOM does.

    // 1) A plan that fits NEVER blocks — under any forced/allow_oom combination.
    for (bool forced : {false, true})
        for (bool allow_oom : {false, true})
            assert(!should_block_oom(mk(/*fits=*/true, forced), allow_oom) &&
                   "fits==true must never block");

    // 2) Won't fit + override off → BLOCK regardless of forced. Forcing --gpus
    //    pins the card count; it does NOT bypass the fit gate anymore.
    for (bool forced : {false, true})
        assert(should_block_oom(mk(/*fits=*/false, forced), /*allow_oom=*/false) &&
               "over-budget load with override off must block even when forced");

    // 3) Won't fit but IE_ALLOW_OOM override on → attempt anyway, any forced.
    for (bool forced : {false, true})
        assert(!should_block_oom(mk(/*fits=*/false, forced), /*allow_oom=*/true) &&
               "allow_oom override must suppress the block");

    std::puts("memory_plan_gate_test: OK");
    return 0;
}
