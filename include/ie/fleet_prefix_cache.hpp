// include/ie/fleet_prefix_cache.hpp — prompt/KV prefix cache for the Qwen3-Next-80B
// multi-GPU (fleet) path.
//
// The crown PrefixCache (prefix_cache.hpp) snapshots a single (KvCache,
// DeltaNetState) pair on one GPU. Qwen3NextModel layer-splits across the fleet
// and keeps its per-card state as std::vector<KvCache> kv_ + std::vector<
// DeltaNetState> dn_ (one per device). Those are the EXACT same classes, so this
// cache is "run the crown design once per device": each endpoint owns a vector of
// per-card snapshots, sized to that card's layer mix at the endpoint's depth.
//
// Structure / eviction are identical to PrefixCache: a token-trie with one edge
// per token id, longest-match lookup, LRU eviction on a logical clock. The only
// differences are the per-device payload and that allocations land on each card's
// own DeviceAllocator (fleet->dev(dev)).
//
// Memory: per endpoint ≈ Σ_dev (DeltaNet snapshot ~tens of MB/card + KV slab for
// depth N). Larger per-endpoint footprint than the crown → default max_entries is
// small (12); insert is skipped (not fatal) if a card lacks VRAM headroom.

#pragma once

#include "ie/deltanet_state.hpp"
#include "ie/kv_cache.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace ie {

class Qwen3NextModel;  // fwd — defined in qwen3next.hpp (included by the .cpp)
class DeviceFleet;     // fwd — from allocator.hpp

struct FleetPrefixCacheConfig {
    uint32_t max_entries    = 12;     // smaller than the crown's 32 (per-endpoint = N cards)
    uint32_t max_prefix_len = 8192;   // upper bound on a cached prefix length
};

class FleetPrefixCache {
public:
    FleetPrefixCache() = default;
    ~FleetPrefixCache();
    FleetPrefixCache(const FleetPrefixCache&) = delete;
    FleetPrefixCache& operator=(const FleetPrefixCache&) = delete;

    // Capture n_dev + the fleet from the loaded model. Per-entry memory is
    // allocated lazily on insert (on each card's own allocator).
    std::string init(Qwen3NextModel& m, const FleetPrefixCacheConfig& pcfg);

    // Per-device snapshot payload. Indexed by device; entry is null for a card
    // that has no full-attn (kv) / no linear (dn) layers.
    using KvVec = std::vector<std::unique_ptr<KvCache>>;
    using DnVec = std::vector<std::unique_ptr<DeltaNetState>>;

    struct LookupResult {
        uint32_t      match_len = 0;
        const KvVec*  kv = nullptr;
        const DnVec*  dn = nullptr;
    };

    // Walk the trie along `tokens`; return the deepest endpoint along the prefix.
    // {0, nullptr, nullptr} if none. Refreshes the matched endpoint's LRU stamp.
    LookupResult find_longest_match(const std::vector<int32_t>& tokens);

    // Snapshot the model's CURRENT per-card state at depth tokens.size() and store
    // it as a new endpoint. LRU-evicts at capacity. No-op if an endpoint already
    // exists at this exact depth+sequence. Per-card alloc is sized to this depth.
    std::string insert(Qwen3NextModel& m, const std::vector<int32_t>& tokens);

    uint32_t n_entries() const noexcept { return uint32_t(endpoints_.size()); }
    uint64_t total_bytes() const noexcept;
    void     clear() noexcept;

private:
    struct Node {
        std::map<int32_t, std::unique_ptr<Node>> children;
        bool     is_endpoint = false;
        KvVec    kv;                  // [dev] (null per-card if that card has no KV)
        DnVec    dn;                  // [dev] (null per-card if that card has no DN)
        uint64_t depth          = 0;
        uint64_t last_access_us = 0;
    };

    Node* walk_or_create(const std::vector<int32_t>& tokens);
    Node* lru_endpoint(const Node* exclude = nullptr) const;
    void  evict_endpoint(Node* n);

    DeviceFleet*           fleet_ = nullptr;
    uint32_t               n_dev_ = 0;
    FleetPrefixCacheConfig pcfg_{};

    std::unique_ptr<Node>  root_;
    std::vector<Node*>     endpoints_;  // flat list for O(N) LRU scan
    uint64_t               tick_ = 0;   // monotonic logical clock for LRU
};

}  // namespace ie
