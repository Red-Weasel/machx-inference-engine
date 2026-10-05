// include/ie/dn_ladder.hpp -- P4 B57: in-place restart points for a hybrid (full attention + DeltaNet) split model.
//
// The KV rows of a conversation are position-indexed: after an edit at token p, or when the prompt cache could not keep
// the conversation's snapshot, the rows below p are still right where they are. What cannot be rewound is the DeltaNet
// state (a running state over everything read). A ladder keeps copies of that state, per card, taken at known depths of
// the LIVE sequence -- every `step` tokens while a prompt is read, at the prompt's end and at the reply's end -- and the
// tokens the live state covers. A next prompt that shares L leading tokens with them restarts in place from the deepest
// copy at or below min(L, prompt - 1): the copy back into the live state, every full-attention layer's KV length set to
// that depth; no KV is copied. Only what follows is read again.
//
// A copy is a DeltaNetState (state fp32 + conv fp16; tens of MB a card, independent of the context length). Slots are
// allocated on first use; a failed allocation turns the ladder off (nothing is refused). Host bookkeeping only here: the
// caller decides when the live state is at a depth (both cards idle) and clears the ladder whenever the live state is
// replaced by anything else (a cache restore, another request's state, a reset).
#pragma once

#include "ie/allocator.hpp"
#include "ie/deltanet_state.hpp"
#include "ie/dn_ladder_plan.hpp"
#include "ie/kv_cache.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace ie {

// The device copies for a model with fleet(), n_devices(), dev_has_dn(d), dn_state(d), dev_has_kv(d), kv_cache(d).
class DnLadder {
public:
    DnLadderPlan plan;
    std::vector<int32_t> live;         // the tokens the live state covers (empty = unknown: no in-place restart)
    bool on = false;
    uint64_t takes = 0, restores = 0;
    void init(uint32_t n_dev, uint32_t step, uint32_t n_regular) {
        plan.init(step, n_regular);
        n_dev_ = n_dev;
        ck_.clear();
        ck_.resize(size_t(plan.slots()) * n_dev);
        live.clear();
        on = true;
    }
    void clear() { plan.clear(); live.clear(); }
    void free_all() { ck_.clear(); plan.clear(); live.clear(); on = false; }
    template <class Model> uint64_t slot_bytes(Model& m, uint32_t dev) const {
        if (!m.dev_has_dn(dev)) return 0;
        const DeltaNetState& s = m.dn_state(dev);
        const uint64_t nl = s.config().n_layers_linear;
        return nl * (s.state_elems_per_layer() * sizeof(float) + s.conv_elems_per_layer() * sizeof(sycl::half));
    }
    // The live state is at `depth` on every card: keep a copy in `slot`. An error turns the ladder off.
    template <class Model> std::string take(Model& m, uint32_t slot, uint32_t depth) {
        if (!on || slot >= plan.slots() || depth == 0) return {};
        plan.depth[slot] = 0;
        for (uint32_t dev = 0; dev < n_dev_; ++dev) {
            if (!m.dev_has_dn(dev)) continue;
            auto& c = ck_[size_t(slot) * n_dev_ + dev];
            if (!c) {
                c = std::make_unique<DeltaNetState>();
                if (auto e = c->init(m.fleet()->dev(dev), m.dn_state(dev).config()); !e.empty()) { free_all(); return "slot alloc: " + e; }
            }
            if (auto e = c->copy_from(m.fleet()->dev(dev).queue(), m.dn_state(dev)); !e.empty()) { free_all(); return "copy: " + e; }
        }
        plan.depth[slot] = depth;
        ++takes;
        return {};
    }
    // The live state back to `slot`'s depth, in place. An error leaves the caller to reset and read everything again.
    template <class Model> std::string restore(Model& m, uint32_t slot) {
        if (!on || slot >= plan.slots() || !plan.depth[slot]) return "no copy in the slot";
        const uint32_t d = plan.depth[slot];
        for (uint32_t dev = 0; dev < n_dev_; ++dev) {
            if (m.dev_has_dn(dev)) {
                const auto& c = ck_[size_t(slot) * n_dev_ + dev];
                if (!c) return "no copy on card " + std::to_string(dev);
                if (auto e = m.dn_state(dev).copy_from(m.fleet()->dev(dev).queue(), *c); !e.empty()) return e;
            }
            if (m.dev_has_kv(dev)) {
                KvCache& kv = m.kv_cache(dev);
                for (uint32_t li = 0; li < kv.config().n_layers_full; ++li) kv.set_length(li, d);
            }
        }
        plan.drop_above(d);
        live.resize(std::min<size_t>(live.size(), d));
        ++restores;
        return {};
    }
private:
    uint32_t n_dev_ = 0;
    std::vector<std::unique_ptr<DeltaNetState>> ck_;   // [slot * n_dev + dev]
};

}  // namespace ie
