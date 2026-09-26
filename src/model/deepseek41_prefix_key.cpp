// src/model/deepseek41_prefix_key.cpp — see include/ie/deepseek41_prefix_key.hpp (#48, docs/deepseek41/103).
#include "ie/deepseek41_prefix_key.hpp"

#include <cstdio>

namespace ie {

uint64_t ds41_fnv1a64(std::string_view s) {
    uint64_t h = 14695981039346656037ull;                      // the FNV-1a 64 offset basis (0xcbf29ce484222325)
    for (const char c : s) h = (h ^ uint8_t(c)) * 1099511628211ull;   // ... and prime (0x100000001b3)
    return h;
}

std::string ds41_numerics_fingerprint(std::string_view manifest) {
    char hex[17];
    std::snprintf(hex, sizeof hex, "%016llx", static_cast<unsigned long long>(ds41_fnv1a64(manifest)));
    return hex;
}

std::string ds41_prefix_disk_key(std::string_view model, uint32_t format, std::string_view manifest, std::string_view runtime) {
    std::string k(model);
    k += " | format " + std::to_string(format);
    k += " | numerics " + ds41_numerics_fingerprint(manifest);
    k += " | runtime ";
    k += runtime;
    return k;
}

}  // namespace ie
