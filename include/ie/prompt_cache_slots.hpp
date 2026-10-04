#pragma once
// P4 B45 (3): what /props "prompt_cache_slots" reports for the GGUF archs (Engine::prompt_cache_slots; MiMo-V2.6 and
// DeepSeek-V4.1 have their own counts). Dream reads the field and, at 1, SKIPS its end-of-turn verifier ("this engine keeps
// one conversation cached ... would evict this one"), so a server whose prompt cache holds many conversations must say so:
// the crown split's lanes cache (Engine::q35m_lanes_init: "prompt cache for the lanes: up to N entries", N = max(12,
// 2 x lanes + 8) under its VRAM budget, or the load's 12 without one) and the 27B split's shared-prefix cache (12 entries,
// IE_PROMPT_CACHE_MAX_ENTRIES) are FleetPrefixCache entry limits, one conversation each; every other arch keeps today's 1
// (the live conversation's prefix) when the cache is on, 0 when it is off. Host-tested in tests/unit/prompt_cache_slots_test.cpp.
#include <cstdint>

namespace ie {

inline uint32_t prompt_cache_slots_rule(bool cache_on, bool split_fleet_cache, uint32_t fleet_max_entries) {
    if (!cache_on) return 0;
    if (split_fleet_cache) return fleet_max_entries;
    return 1;
}

}  // namespace ie
