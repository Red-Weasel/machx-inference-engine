// tests/unit/prompt_cache_slots_test.cpp -- host-only (no GPU, no model): /props "prompt_cache_slots" per arch
// (include/ie/prompt_cache_slots.hpp, P4 B45 (3)). Dream skips its end-of-turn verifier when the field says 1, so the two
// splits whose FleetPrefixCache holds many conversations must report their entry limit; the others keep 1 / 0.
#undef NDEBUG
#include "ie/prompt_cache_slots.hpp"

#include <cassert>
#include <cstdio>

int main() {
    using ie::prompt_cache_slots_rule;
    // the cache off: 0 on every arch (--no-prompt-cache, IE_NO_PROMPT_CACHE)
    assert(prompt_cache_slots_rule(false, false, 12) == 0);
    assert(prompt_cache_slots_rule(false, true, 40) == 0);
    // a single-card crown, Flash-Next, GLM, the dense archs: the live conversation's prefix = 1 (today's value)
    assert(prompt_cache_slots_rule(true, false, 12) == 1);
    assert(prompt_cache_slots_rule(true, false, 0) == 1);
    // the crown split at 16 lanes with the lanes cache: the load line's "up to 40 entries" (2 x 16 + 8 with the anchor)
    assert(prompt_cache_slots_rule(true, true, 40) == 40);
    // the crown split at --parallel 1 (no lanes budget) and the 27B split's shared-prefix cache: the load's 12 entries
    assert(prompt_cache_slots_rule(true, true, 12) == 12);
    // IE_PROMPT_CACHE_MAX_ENTRIES raised: whatever the cache's limit is
    assert(prompt_cache_slots_rule(true, true, 64) == 64);
    std::puts("prompt_cache_slots_test: all OK");
    return 0;
}
