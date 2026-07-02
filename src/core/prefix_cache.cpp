// src/core/prefix_cache.cpp — PR #3 stage 2 implementation.

#include "ie/prefix_cache.hpp"

#include <algorithm>
#include <cstdio>

namespace ie {

PrefixCache::~PrefixCache() { clear(); }

std::string PrefixCache::init(DeviceAllocator& alloc,
                              const QwenConfig& cfg,
                              const PrefixCacheConfig& pcfg) {
    alloc_ = &alloc;
    cfg_   = cfg;
    pcfg_  = pcfg;
    L_full_ = cfg.n_layers / cfg.full_attn_interval;
    L_lin_  = cfg.n_layers - L_full_;
    root_ = std::make_unique<Node>();
    return {};
}

PrefixCache::Node* PrefixCache::walk_or_create(const std::vector<int32_t>& tokens) {
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

PrefixCache::LookupResult PrefixCache::find_longest_match(const std::vector<int32_t>& tokens) {
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
        result.kv = best_endpoint->kv.get();
        result.dn = best_endpoint->dn.get();
    }
    return result;
}

PrefixCache::Node* PrefixCache::lru_endpoint(const Node* exclude) const {
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

void PrefixCache::evict_endpoint(Node* n) {
    if (!n || !n->is_endpoint) return;
    n->kv.reset();
    n->dn.reset();
    n->is_endpoint = false;
    auto it = std::find(endpoints_.begin(), endpoints_.end(), n);
    if (it != endpoints_.end()) endpoints_.erase(it);
}

std::string PrefixCache::insert(sycl::queue& q,
                                const std::vector<int32_t>& tokens,
                                const KvCache& kv,
                                const DeltaNetState& dn) {
    if (!alloc_) return "PrefixCache::insert: not initialized";
    if (tokens.empty()) return "PrefixCache::insert: empty token list";
    if (tokens.size() > pcfg_.max_prefix_len)
        return "PrefixCache::insert: tokens.size() exceeds max_prefix_len";

    Node* node = walk_or_create(tokens);
    if (node->is_endpoint) {
        // Idempotent: refresh access timestamp and return.  Don't re-snapshot.
        ++tick_;
        node->last_access_us = tick_;
        return {};
    }

    // Evict if at capacity.
    while (endpoints_.size() >= pcfg_.max_entries) {
        Node* victim = lru_endpoint(node);
        if (!victim) break;  // cache empty (shouldn't happen if size>=max but be safe)
        evict_endpoint(victim);
    }

    // Allocate per-endpoint storage sized to this depth.
    auto kv_endpoint = std::make_unique<KvCache>();
    auto dn_endpoint = std::make_unique<DeltaNetState>();
    {
        KvCacheConfig kvcfg{};
        kvcfg.n_layers_full = L_full_;
        kvcfg.n_kv_heads    = cfg_.n_kv_heads;
        kvcfg.max_ctx       = uint32_t(tokens.size());
        kvcfg.head_dim      = cfg_.head_dim;
        kvcfg.use_int8      = false;
        if (auto e = kv_endpoint->init(*alloc_, kvcfg); !e.empty())
            return "kv_endpoint init: " + e;
    }
    if (auto e = dn_endpoint->init(*alloc_, DeltaNetStateConfig{
            L_lin_, cfg_.ssm_n_v_heads, cfg_.ssm_head_dim, cfg_.ssm_head_dim,
            cfg_.ssm_inner * 2, cfg_.ssm_conv_kernel}); !e.empty())
        return "dn_endpoint init: " + e;

    // Snapshot live state into the per-endpoint storage.
    if (auto e = kv_endpoint->copy_prefix_from(q, kv, uint32_t(tokens.size())); !e.empty())
        return "kv copy_prefix_from: " + e;
    if (auto e = dn_endpoint->copy_from(q, dn); !e.empty())
        return "dn copy_from: " + e;

    node->kv = std::move(kv_endpoint);
    node->dn = std::move(dn_endpoint);
    node->is_endpoint = true;
    ++tick_;
    node->last_access_us = tick_;
    endpoints_.push_back(node);
    return {};
}

uint64_t PrefixCache::total_bytes() const noexcept {
    uint64_t total = 0;
    for (const Node* ep : endpoints_) {
        if (ep->kv) total += ep->kv->bytes_per_layer() * L_full_ * 2;  // K + V
        if (ep->dn) {
            total += ep->dn->state_elems_per_layer() * L_lin_ * sizeof(float);
            total += ep->dn->conv_elems_per_layer()  * L_lin_ * sizeof(sycl::half);
        }
    }
    return total;
}

void PrefixCache::clear() noexcept {
    for (Node* ep : endpoints_) {
        ep->kv.reset();
        ep->dn.reset();
        ep->is_endpoint = false;
    }
    endpoints_.clear();
    root_.reset();
    tick_ = 0;
}

void PrefixCache::dump(std::FILE* out) const {
    if (!out) out = stdout;
    std::fprintf(out, "PrefixCache: %zu entries, %.1f MiB total\n",
                 endpoints_.size(), double(total_bytes()) / (1024.0 * 1024.0));
    for (size_t i = 0; i < endpoints_.size(); ++i) {
        const Node* ep = endpoints_[i];
        std::fprintf(out, "  [%zu] depth=%llu  last_access=%llu\n",
                     i,
                     (unsigned long long)ep->depth,
                     (unsigned long long)ep->last_access_us);
    }
}

}  // namespace ie
