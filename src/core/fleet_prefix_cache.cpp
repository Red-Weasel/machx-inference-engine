// src/core/fleet_prefix_cache.cpp — Qwen3-Next-80B fleet prefix cache.
//
// Trie/LRU mirror src/core/prefix_cache.cpp (per-arch copy is the repo norm);
// the per-device payload + per-card allocation are the only real differences.

#include "ie/fleet_prefix_cache.hpp"

#include "ie/allocator.hpp"        // DeviceFleet / DeviceAllocator
#include "ie/qwen3next.hpp"        // Qwen3NextModel (accessors)
#include "ie/qwen35moe_split.hpp"  // Qwen35MoeSplitModel (same per-card surface)
#include "ie/gptoss_tp.hpp"        // GptOssTpModel (KV-only; DeltaNet-free)
#include "ie/qwen3moe_split.hpp"   // Qwen3MoeSplitModel (Coder/Tongyi layer-split; KV-only)
#include "ie/qwen3moe_tp.hpp"      // Qwen3MoeTpModel (Coder/Tongyi tensor-parallel; KV-only)
#include "ie/qwen35_split.hpp"     // Qwen35SplitModel (27B DeltaNet+dense layer-split; KV+DN)

#include <algorithm>

namespace ie {

FleetPrefixCache::~FleetPrefixCache() { clear(); }

template <class Model>
std::string FleetPrefixCache::init(Model& m,
                                   const FleetPrefixCacheConfig& pcfg) {
    fleet_ = m.fleet();
    n_dev_ = m.n_devices();
    pcfg_  = pcfg;
    if (!fleet_ || n_dev_ == 0) return "FleetPrefixCache::init: model not loaded";
    root_  = std::make_unique<Node>();
    return {};
}

FleetPrefixCache::Node* FleetPrefixCache::walk_or_create(
        const std::vector<int32_t>& tokens) {
    Node* cur = root_.get();
    for (size_t i = 0; i < tokens.size(); ++i) {
        auto& slot = cur->children[tokens[i]];
        if (!slot) {
            slot = std::make_unique<Node>();
            slot->depth = i + 1;
        }
        cur = slot.get();
    }
    return cur;
}

FleetPrefixCache::LookupResult FleetPrefixCache::find_longest_match(
        const std::vector<int32_t>& tokens) {
    LookupResult result;
    Node* cur = root_.get();
    Node* best_endpoint = nullptr;

    for (size_t i = 0; i < tokens.size(); ++i) {
        auto it = cur->children.find(tokens[i]);
        if (it == cur->children.end()) break;
        cur = it->second.get();
        if (cur->is_endpoint) best_endpoint = cur;
    }

    if (best_endpoint) {
        ++tick_;
        best_endpoint->last_access_us = tick_;
        result.match_len = uint32_t(best_endpoint->depth);
        result.kv = &best_endpoint->kv;
        result.dn = &best_endpoint->dn;
    }
    return result;
}

uint32_t FleetPrefixCache::peek_longest_match(const std::vector<int32_t>& tokens) const {
    const Node* cur = root_.get();
    uint32_t best = 0;
    for (size_t i = 0; cur && i < tokens.size(); ++i) {
        auto it = cur->children.find(tokens[i]);
        if (it == cur->children.end()) break;
        cur = it->second.get();
        if (cur->is_endpoint) best = uint32_t(cur->depth);
    }
    return best;
}

FleetPrefixCache::Node* FleetPrefixCache::lru_endpoint(const Node* exclude) const {
    Node* lru = nullptr;
    uint64_t lru_t = UINT64_MAX;
    for (Node* ep : endpoints_) {
        if (ep == exclude) continue;
        if (ep->last_access_us < lru_t) {
            lru_t = ep->last_access_us;
            lru = ep;
        }
    }
    return lru;
}

void FleetPrefixCache::evict_endpoint(Node* n) {
    if (!n || !n->is_endpoint) return;
    n->kv.clear();
    n->dn.clear();
    n->is_endpoint = false;
    auto it = std::find(endpoints_.begin(), endpoints_.end(), n);
    if (it != endpoints_.end()) endpoints_.erase(it);
    std::vector<int32_t> path;
    path.swap(n->path);
    prune(path);   // (may free n)
}

void FleetPrefixCache::prune(const std::vector<int32_t>& tokens) {
    if (!root_) return;
    std::vector<Node*> chain;   // chain[i] = the node at depth i (chain[0] = root)
    chain.reserve(tokens.size() + 1);
    Node* cur = root_.get();
    chain.push_back(cur);
    for (int32_t t : tokens) {
        auto it = cur->children.find(t);
        if (it == cur->children.end()) break;
        cur = it->second.get();
        chain.push_back(cur);
    }
    for (size_t d = chain.size() - 1; d > 0; --d) {
        Node* n = chain[d];
        if (n->is_endpoint || !n->children.empty() || n == protect_) break;   // (protect_: insert's node being committed)
        chain[d - 1]->children.erase(tokens[d - 1]);   // frees n
    }
}

template <class Model>
std::string FleetPrefixCache::insert(Model& m,
                                     const std::vector<int32_t>& tokens) {
    if (!fleet_) return "FleetPrefixCache::insert: not initialized";
    if (tokens.empty()) return "FleetPrefixCache::insert: empty token list";
    if (tokens.size() > pcfg_.max_prefix_len)
        return "FleetPrefixCache::insert: tokens.size() exceeds max_prefix_len";

    Node* node = walk_or_create(tokens);
    struct PruneOnFail {   // P4 B15: a failed snapshot leaves no node behind
        FleetPrefixCache* c; const std::vector<int32_t>& t; bool armed = true;
        ~PruneOnFail() { if (armed) c->prune(t); }
    } prune_on_fail{this, tokens};
    if (node->is_endpoint) {
        prune_on_fail.armed = false;
        // Idempotent: refresh access timestamp, don't re-snapshot.
        ++tick_;
        node->last_access_us = tick_;
        return {};
    }

    const uint32_t N = uint32_t(tokens.size());

    // Build the per-card snapshot vectors into LOCALS first; only commit to the
    // node on full success so a mid-loop alloc failure (OOM) frees cleanly and
    // leaves the cache unchanged ("skip insert, don't crash" — there is no free-
    // VRAM query, so a null device alloc surfacing as an init error is the guard).
    KvVec kv_snap(n_dev_);
    DnVec dn_snap(n_dev_);
    for (uint32_t dev = 0; dev < n_dev_; ++dev) {
        DeviceAllocator& a = fleet_->dev(dev);
        sycl::queue&     q = a.queue();
        if (m.dev_has_kv(dev)) {
            KvCacheConfig kc = m.kv_cache(dev).config();
            kc.max_ctx = N;                 // snapshot only needs depth N
            auto snap = std::make_unique<KvCache>();
            if (auto e = snap->init(a, kc); !e.empty())
                return "kv snap dev " + std::to_string(dev) + " init: " + e;
            if (auto e = snap->copy_prefix_from(q, m.kv_cache(dev), N); !e.empty())
                return "kv snap dev " + std::to_string(dev) + " copy: " + e;
            kv_snap[dev] = std::move(snap);
        }
        if (m.dev_has_dn(dev)) {
            auto snap = std::make_unique<DeltaNetState>();
            if (auto e = snap->init(a, m.dn_state(dev).config()); !e.empty())
                return "dn snap dev " + std::to_string(dev) + " init: " + e;
            if (auto e = snap->copy_from(q, m.dn_state(dev)); !e.empty())
                return "dn snap dev " + std::to_string(dev) + " copy: " + e;
            dn_snap[dev] = std::move(snap);
        }
    }

    // Snapshot fully built — now make room and commit.
    protect_ = node;   // an evicted endpoint deeper along node's path must not prune node itself
    while (endpoints_.size() >= pcfg_.max_entries) {
        Node* victim = lru_endpoint(node);
        if (!victim) break;
        evict_endpoint(victim);
    }
    protect_ = nullptr;

    node->kv = std::move(kv_snap);
    node->dn = std::move(dn_snap);
    node->is_endpoint = true;
    node->path = tokens;
    prune_on_fail.armed = false;
    ++tick_;
    node->last_access_us = tick_;
    endpoints_.push_back(node);
    return {};
}

uint64_t FleetPrefixCache::total_bytes() const noexcept {
    uint64_t total = 0;
    for (const Node* ep : endpoints_) {
        for (const auto& kv : ep->kv)
            if (kv) total += kv->bytes_per_layer() * kv->config().n_layers_full * 2;  // K+V
        for (const auto& dn : ep->dn)
            if (dn) {
                total += dn->state_elems_per_layer() * dn->config().n_layers_linear * sizeof(float);
                total += dn->conv_elems_per_layer()  * dn->config().n_layers_linear * sizeof(sycl::half);
            }
    }
    return total;
}

void FleetPrefixCache::clear() noexcept {
    for (Node* ep : endpoints_) {
        ep->kv.clear();
        ep->dn.clear();
        ep->is_endpoint = false;
    }
    endpoints_.clear();
    root_.reset();
    tick_ = 0;
}

// Explicit instantiations — the two multi-GPU hybrid models that own per-card
// std::vector<KvCache> + std::vector<DeltaNetState> and expose the accessor set.
template std::string FleetPrefixCache::init<Qwen3NextModel>(
    Qwen3NextModel&, const FleetPrefixCacheConfig&);
template std::string FleetPrefixCache::insert<Qwen3NextModel>(
    Qwen3NextModel&, const std::vector<int32_t>&);
template std::string FleetPrefixCache::init<Qwen35MoeSplitModel>(
    Qwen35MoeSplitModel&, const FleetPrefixCacheConfig&);
template std::string FleetPrefixCache::insert<Qwen35MoeSplitModel>(
    Qwen35MoeSplitModel&, const std::vector<int32_t>&);
template std::string FleetPrefixCache::init<GptOssTpModel>(
    GptOssTpModel&, const FleetPrefixCacheConfig&);
template std::string FleetPrefixCache::insert<GptOssTpModel>(
    GptOssTpModel&, const std::vector<int32_t>&);
template std::string FleetPrefixCache::init<Qwen3MoeSplitModel>(
    Qwen3MoeSplitModel&, const FleetPrefixCacheConfig&);
template std::string FleetPrefixCache::insert<Qwen3MoeSplitModel>(
    Qwen3MoeSplitModel&, const std::vector<int32_t>&);
template std::string FleetPrefixCache::init<Qwen3MoeTpModel>(
    Qwen3MoeTpModel&, const FleetPrefixCacheConfig&);
template std::string FleetPrefixCache::insert<Qwen3MoeTpModel>(
    Qwen3MoeTpModel&, const std::vector<int32_t>&);
template std::string FleetPrefixCache::init<Qwen35SplitModel>(
    Qwen35SplitModel&, const FleetPrefixCacheConfig&);
template std::string FleetPrefixCache::insert<Qwen35SplitModel>(
    Qwen35SplitModel&, const std::vector<int32_t>&);

}  // namespace ie
