#pragma once
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
namespace ie {
// Server policy only. Standalone research runners retain their own defaults.
struct Glm5MemoryPolicy {
    bool pin_banks = true;
    uint64_t cache_bytes = 0; // automatic from live per-card headroom
    uint64_t pin_max_bytes = UINT64_MAX; // still bounded by live host budget
    double floor_gib = 40.0;
    double pin_overhead = 1.0;  // pinned host USM costs 1.00x (measured 2026-09-11; the old 1.37 was the VRAM mirror)
};
inline std::string glm5_memory_policy(const char* pin, const char* cache,
        const char* floor, const char* cap, const char* overhead, Glm5MemoryPolicy& out) {
    out = {};
    if (pin) {
        if (std::string_view(pin)!="0" && std::string_view(pin)!="1") return "invalid IE_G5_PIN_BANKS (use 0 or 1)";
        out.pin_banks = pin[0]=='1';
    }
    if (cache) {
        uint64_t mb=0; std::string_view s(cache);
        auto r=std::from_chars(s.data(),s.data()+s.size(),mb);
        if(r.ec!=std::errc{} || r.ptr!=s.data()+s.size() || !mb || mb>1048576)
            return "invalid IE_G5_ECACHE_MB";
        out.cache_bytes=mb<<20;
    }
    auto number=[](const char* s, double lo, double hi, double& value) {
        if(!s)return true;
        std::string_view text(s);
        auto r=std::from_chars(text.data(),text.data()+text.size(),value);
        return r.ec==std::errc{} && r.ptr==text.data()+text.size() &&
               std::isfinite(value) && value>=lo && value<=hi;
    };
    if(!number(floor,8,1048576,out.floor_gib))return "invalid IE_G5_PIN_FLOOR_GIB";
    double cap_gib=0;
    if(!number(cap,0,1048576,cap_gib))return "invalid IE_G5_PIN_MAX_GIB";
    if(cap)out.pin_max_bytes=uint64_t(cap_gib*1073741824.0);
    if(!number(overhead,1,16,out.pin_overhead))return "invalid IE_G5_PIN_OVERHEAD";
    return {};
}
inline uint64_t glm5_cache_budget(uint64_t available, uint64_t weights,
        uint64_t runtime, uint64_t experts, const Glm5MemoryPolicy& policy) {
    if(weights>available || runtime>available-weights)return 0;
    // Explicit budgets are validated by the caller, never silently reduced.
    if(policy.cache_bytes)return policy.cache_bytes;
    return std::min(available-weights-runtime,experts) & ~((1ull<<20)-1);
}
inline uint64_t glm5_pin_budget(uint64_t available, uint64_t staging,
        uint32_t stages, const Glm5MemoryPolicy& policy) {
    if(!policy.pin_banks || !stages || available<staging)return 0;
    const long double free=static_cast<long double>(available-staging)-policy.floor_gib*1073741824.L;
    if(free<=0)return 0;
    return std::min(policy.pin_max_bytes,uint64_t(free/policy.pin_overhead/stages));
}
}
