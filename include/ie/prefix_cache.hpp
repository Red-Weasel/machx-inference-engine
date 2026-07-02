// include/ie/prefix_cache.hpp — PR #3 stage 2: token-trie prefix cache.
//
// Caches (token_ids → (KvCache slab, DeltaNetState snapshot)) so that
// repeated prompts skip prefill on the cached portion.  Stage 1's
// KvCache::copy_prefix_from and DeltaNetState::copy_from are the
// load-bearing primitives; this file is the multi-entry layer on top.
//
// Structure: token-trie with one edge per token id.  Multiple endpoints
// may sit at different depths along the same root-to-leaf path; longest-
// match lookup picks the deepest endpoint along the requested prefix.
//
// Memory: each endpoint owns a private (KvCache, DeltaNetState) pair,
// sized for that endpoint's depth.  KV is 20 KB × depth; DN is ~62 MB
// flat regardless of depth.  DN dominates so the practical entry count
// is bounded by VRAM headroom (typically 30-80 entries on a 32 GB GPU
// once the model is resident).
//
// Eviction: LRU on last-access timestamp.  Endpoint nodes are kept on
// a flat list for O(N) eviction scan; trie navigation is O(L) per
// lookup where L is prefix length.

#pragma once

#include "ie/allocator.hpp"
#include "ie/deltanet_state.hpp"
#include "ie/kv_cache.hpp"
#include "ie/qwen36.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace ie {

struct PrefixCacheConfig {
    uint32_t max_entries = 32;       // cap on simultaneously-stored endpoints
    uint32_t max_prefix_len = 4096;  // upper bound on a cached prefix length
};

class PrefixCache {
public:
    PrefixCache() = default;
    ~PrefixCache();
    PrefixCache(const PrefixCache&) = delete;
    PrefixCache& operator=(const PrefixCache&) = delete;

    // Init.  Captures sizing from QwenConfig; per-entry memory is allocated
    // lazily on insert.
    std::string init(DeviceAllocator& alloc,
                     const QwenConfig& cfg,
                     const PrefixCacheConfig& pcfg);

    // Result of a longest-match lookup.
    struct LookupResult {
        uint32_t             match_len = 0;
        const KvCache*       kv = nullptr;
        const DeltaNetState* dn = nullptr;
    };

    // Walk the trie along `tokens`; track the deepest endpoint encountered.
    // Returns {0, nullptr, nullptr} if no endpoint along the prefix path.
    // Side-effect: updates last-access timestamp on the matched endpoint.
    LookupResult find_longest_match(const std::vector<int32_t>& tokens);

    // Snapshot the current state of (kv, dn) at depth `tokens.size()` and
    // store it as a new endpoint along the trie path for `tokens`.
    // Triggers LRU eviction if at capacity.  No-op if an endpoint at this
    // exact depth already exists for this token sequence.
    std::string insert(sycl::queue& q,
                       const std::vector<int32_t>& tokens,
                       const KvCache& kv,
                       const DeltaNetState& dn);

    // Stats.
    uint32_t n_entries() const noexcept { return uint32_t(endpoints_.size()); }
    uint64_t total_bytes() const noexcept;
    void     clear() noexcept;

    // Diagnostic: print all endpoints and their depths to stdout.
    void dump(std::FILE* out = nullptr) const;

private:
    struct Node {
        std::map<int32_t, std::unique_ptr<Node>> children;
        // Endpoint payload (only set when is_endpoint == true).
        bool is_endpoint = false;
        std::unique_ptr<KvCache>       kv;
        std::unique_ptr<DeltaNetState> dn;
        uint64_t depth          = 0;
        uint64_t last_access_us = 0;
    };

    // Walk-and-create nodes along `tokens` to depth N; returns the leaf node.
    Node* walk_or_create(const std::vector<int32_t>& tokens);

    // Find the LRU endpoint (caller-provided exclusion to avoid evicting
    // the one we just created).  Returns nullptr if cache is empty.
    Node* lru_endpoint(const Node* exclude = nullptr) const;

    // Tear down an endpoint: drops kv/dn, marks node as non-endpoint,
    // removes from endpoint list.  Doesn't prune empty internal nodes
    // (cheap to leave them; pruning is purely a memory-of-pointers cost).
    void evict_endpoint(Node* n);

    DeviceAllocator*  alloc_ = nullptr;
    QwenConfig        cfg_{};
    PrefixCacheConfig pcfg_{};
    uint32_t          L_full_ = 0;
    uint32_t          L_lin_  = 0;

    std::unique_ptr<Node> root_;
    std::vector<Node*>    endpoints_;  // flat list for O(N) LRU scan
    uint64_t              tick_ = 0;   // monotonic logical clock for LRU
};

}  // namespace ie
