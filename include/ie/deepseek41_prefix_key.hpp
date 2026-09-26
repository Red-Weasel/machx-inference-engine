// include/ie/deepseek41_prefix_key.hpp — #48 (docs/deepseek41/103): the key of a V4.1 disk prompt-cache entry.
//
// A disk entry is the state a prompt prefix left in the caches (Phase 47, docs/deepseek41/87). It may be loaded only by
// a process that computes state the same way: the same model, the same entry format, the same arithmetic. The
// arithmetic used to be keyed by this executable's size and mtime, so every rebuild emptied the cache (a ~30 s cold
// first turn). It is keyed now by
//   - the NUMERICS MANIFEST, written at build time by src/model/deepseek41_numerics.cmake: the SHA-256 of every file
//     whose code can change a cached value (src/model/deepseek41_numerics_inputs.txt lists them, with the reasons),
//     the compiler's version line and the build flags that reach those files. A rebuild that changes none of them
//     keeps the cache; any change to one of them invalidates it;
//   - the RUNTIME that still generates arithmetic after the build: each card's GPU and driver (spir64 kernels are
//     JIT-compiled by the driver) and the oneDNN library the process loaded (its GEMM kernels are generated at run
//     time).
// Pure C++ (no SYCL), so the derivation is unit-tested on the CPU (tests/unit/ds41_prefix_key_test.cpp).
#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace ie {

// The entry file's layout version (the header's `ver` field). Bump it when the layout written by prefix_persist changes.
constexpr uint32_t kDs41PrefixDiskFormat = 1;

// FNV-1a, 64-bit, over `s`. The fingerprint of a manifest: the inputs are the engine's own files, so accidental
// collisions (2^-64 per pair) are the only concern, not adversarial ones.
uint64_t ds41_fnv1a64(std::string_view s);

// The manifest's fingerprint as 16 lowercase hex digits (what the log line prints).
std::string ds41_numerics_fingerprint(std::string_view manifest);

// The disk key: the model's identity (shape + embedding fingerprint), the entry format, the manifest's fingerprint and
// the runtime string, in one readable line. Equal inputs give an equal key; a change in any of them gives another key.
std::string ds41_prefix_disk_key(std::string_view model, uint32_t format, std::string_view manifest, std::string_view runtime);

}  // namespace ie
