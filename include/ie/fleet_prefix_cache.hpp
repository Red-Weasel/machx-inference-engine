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

class Qwen3NextModel;       // fwd — defined in qwen3next.hpp (included by the .cpp)
class Qwen35MoeSplitModel;  // fwd — crown Q8 2-card split; same per-card surface
class DeviceFleet;          // fwd — from allocator.hpp

struct FleetPrefixCacheConfig {
    uint32_t max_entries    = 12;     // smaller than the crown's 32 (per-endpoint = N cards)
    uint32_t max_prefix_len = 8192;   // upper bound on a cached prefix length
    // P4 B29: a per-card byte budget for the snapshots (0 = none, the old rule: allocate, then evict by count). With a budget,
    // insert evicts LRU endpoints BEFORE it allocates until every card's snapshots + the new one fit, and skips a snapshot
    // that alone does not fit (never evicting for it), so the cache stays inside VRAM the caller measured as free.
    uint64_t max_dev_bytes  = 0;
    // P4 B29 Fix A: an insert that is not a shared prefix drops the non-shared endpoints that are strict prefixes of it (a
    // conversation's earlier turns: its next turn restores the deeper one), BEFORE the budget / count eviction, so a
    // conversation holds one entry instead of one per turn. false = every turn's entry stays until the LRU takes it.
    bool     supersede      = false;
};

class FleetPrefixCache {
public:
    FleetPrefixCache() = default;
    ~FleetPrefixCache();
    FleetPrefixCache(const FleetPrefixCache&) = delete;
    FleetPrefixCache& operator=(const FleetPrefixCache&) = delete;

    // Capture n_dev + the fleet from the loaded model. Per-entry memory is
    // allocated lazily on insert (on each card's own allocator). Templated on the
    // model type: any model exposing fleet()/n_devices()/dev_has_kv/dev_has_dn/
    // kv_cache(dev)/dn_state(dev) works (Qwen3NextModel, Qwen35MoeSplitModel).
    // Defined + explicitly instantiated in the .cpp.
    template <class Model>
    std::string init(Model& m, const FleetPrefixCacheConfig& pcfg);

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
    // {0, nullptr, nullptr} if none. Refreshes the matched endpoint's LRU stamp, and (P4 B36) every anchor endpoint the walk
    // passes: a conversation's anchor stays as fresh as its turns that restore through it.
    LookupResult find_longest_match(const std::vector<int32_t>& tokens);
    // P4 B15: the depth find_longest_match would return, without touching the LRU stamps (host-only, const).
    uint32_t peek_longest_match(const std::vector<int32_t>& tokens) const;

    // Snapshot the model's CURRENT per-card state at depth tokens.size() and store
    // it as a new endpoint. LRU-evicts at capacity. No-op if an endpoint already
    // exists at this exact depth+sequence. Per-card alloc is sized to this depth.
    // shared = a shared prefix (P4 B15's mark): never superseded (P4 B29), and an existing endpoint inserted again as shared
    // becomes shared.
    // anchor (P4 B36) = the end of a conversation's last user query (the crown's lanes: LanesRequest::anchor): a later
    // non-anchor insert does not supersede it, a later anchor insert does (one anchor per conversation); an existing endpoint
    // inserted again as anchor becomes one.
    template <class Model>
    std::string insert(Model& m, const std::vector<int32_t>& tokens, bool shared = false, bool anchor = false);

    uint32_t n_entries() const noexcept { return uint32_t(endpoints_.size()); }
    // P4 B29: change the caps after init (the crown split's request lanes size them once their own VRAM is allocated). Takes
    // effect at the next insert; existing entries are not evicted here.
    void     set_limits(uint32_t max_entries, uint64_t max_dev_bytes) noexcept { pcfg_.max_entries = max_entries; pcfg_.max_dev_bytes = max_dev_bytes; }
    // P4 B45 (4): the prefix bound insert() applies (the 27B's load shrinks it to fit its budget beside a vision tower)
    void     set_max_prefix_len(uint32_t n) noexcept { pcfg_.max_prefix_len = n; }
    void     set_supersede(bool on) noexcept { pcfg_.supersede = on; }
    const FleetPrefixCacheConfig& config() const noexcept { return pcfg_; }
    uint64_t total_bytes() const noexcept;
    void     clear() noexcept;

private:
    struct Node {
        std::map<int32_t, std::unique_ptr<Node>> children;
        bool     is_endpoint = false;
        bool     shared      = false;   // P4 B29: a shared-prefix endpoint (never superseded)
        bool     anchor      = false;   // P4 B36: a last user query's end (superseded only by a later anchor)
        KvVec    kv;                  // [dev] (null per-card if that card has no KV)
        DnVec    dn;                  // [dev] (null per-card if that card has no DN)
        uint64_t depth          = 0;
        uint64_t last_access_us = 0;
        std::vector<int32_t> path;    // an endpoint's tokens (P4 B15: an eviction prunes its trie branch)
    };

    Node* walk_or_create(const std::vector<int32_t>& tokens);
    Node* lru_endpoint(const Node* exclude = nullptr) const;
    void  evict_endpoint(Node* n);
    // P4 B15: drop the trie nodes along `tokens` that no longer lead to an endpoint (deepest first). Without it every insert
    // left depth-many nodes behind after its endpoint was evicted (or its snapshot failed): host memory grew with every
    // unique conversation (~0.3 MiB a request at 1.5K tokens, measured on the crown; B15 j15).
    void  prune(const std::vector<int32_t>& tokens);

    DeviceFleet*           fleet_ = nullptr;
    uint32_t               n_dev_ = 0;
    FleetPrefixCacheConfig pcfg_{};

    std::unique_ptr<Node>  root_;
    std::vector<Node*>     endpoints_;  // flat list for O(N) LRU scan
    uint64_t               tick_ = 0;   // monotonic logical clock for LRU
    const Node*            protect_ = nullptr;   // P4 B15: prune never frees this node (insert's, while it evicts)
};

}  // namespace ie
