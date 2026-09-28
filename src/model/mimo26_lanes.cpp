// src/model/mimo26_lanes.cpp — MiMo-V2.6 lanes' host bookkeeping (P4 B1). See include/ie/mimo26_lanes.hpp.
#include "ie/mimo26_lanes.hpp"

#include <algorithm>
#include <cstdio>

namespace ie {

uint64_t mimo26_lane_kv_bytes(const std::vector<Mimo26LaneLayer>& layers, uint32_t head_dim, uint32_t v_head_dim, uint32_t ring,
                              uint32_t ctx) {
    uint64_t b = 0;
    for (const auto& l : layers) {
        const uint64_t slots = l.swa && ring ? ring : ctx, v_row = l.swa ? v_head_dim : head_dim;
        b += (uint64_t(l.n_kv) * slots + 64) * head_dim * 2 + uint64_t(l.n_kv) * slots * v_row * 2;
    }
    return b;
}

std::string mimo26_lanes_fit(uint64_t free_bytes, uint64_t per_lane, uint32_t extra, uint64_t reserve, uint32_t card, uint32_t lane_ctx) {
    const uint64_t need = per_lane * extra;
    if (need + reserve <= free_bytes) return {};
    char b[320];
    std::snprintf(b, sizeof b, "lanes: %u more lane(s) at ctx %u need %llu MiB on card %u, which has %llu MiB free less a %llu MiB reserve "
                               "-- fewer lanes or a smaller lane context", extra, lane_ctx, (unsigned long long)(need >> 20), card,
                  (unsigned long long)(free_bytes >> 20), (unsigned long long)(reserve >> 20));
    return b;
}

bool mimo26_lane_commit(Mimo26Lane& l, int32_t id, const std::vector<int32_t>& eos, bool ignore_eos) {
    if (!ignore_eos && std::find(eos.begin(), eos.end(), id) != eos.end()) { l.finish = "stop"; l.next = -1; return false; }
    ++l.n_new;
    l.outbox.push_back(id);
    l.recent.push_back(id);
    if (l.recent.size() > Mimo26Lane::kRecent) l.recent.erase(l.recent.begin());
    l.next = id;
    if (l.n_new >= l.max_new) { l.finish = "length"; return false; }
    return true;
}

Mimo26LaneSet::Mimo26LaneSet(uint32_t n_lanes) : lanes_(std::max<uint32_t>(n_lanes, 1)) {
    for (uint32_t i = 0; i < lanes_.size(); ++i) lanes_[i].index = i;
}

uint32_t Mimo26LaneSet::busy() const {
    uint32_t n = 0;
    for (const auto& l : lanes_) n += l.phase != Mimo26Lane::Phase::kIdle;
    return n;
}

int Mimo26LaneSet::admit(std::vector<int32_t> prompt, uint32_t max_new, uint64_t seed, VitalsWindow* vitals) {
    const int i = lru_idle();
    if (i < 0) return -1;
    Mimo26Lane& l = lanes_[size_t(i)];
    l.phase = Mimo26Lane::Phase::kPrefill;
    l.prompt = std::move(prompt);
    l.prefill_at = 0; l.max_new = max_new; l.n_new = 0; l.next = -1;
    l.rng = seed;
    l.recent.assign(l.prompt.end() - std::ptrdiff_t(std::min(l.prompt.size(), Mimo26Lane::kRecent)), l.prompt.end());
    l.vitals = vitals;
    l.outbox.clear(); l.finish.clear();
    l.tick = ++tick_;
    return i;
}

int Mimo26LaneSet::next() {
    const uint32_t n = uint32_t(lanes_.size());
    for (uint32_t k = 1; k <= n; ++k) {
        const uint32_t i = last_ == ~0u ? k - 1 : (last_ + k) % n;
        if (lanes_[i].phase == Mimo26Lane::Phase::kIdle) continue;
        last_ = i; lanes_[i].tick = ++tick_;
        return int(i);
    }
    return -1;
}

void Mimo26LaneSet::release(uint32_t i) {
    if (i >= lanes_.size()) return;
    lanes_[i].phase = Mimo26Lane::Phase::kIdle;
    lanes_[i].next = -1;
    lanes_[i].vitals = nullptr;
}

int Mimo26LaneSet::lru_idle() const {
    int best = -1;
    for (uint32_t i = 0; i < lanes_.size(); ++i)
        if (lanes_[i].phase == Mimo26Lane::Phase::kIdle && (best < 0 || lanes_[i].tick < lanes_[size_t(best)].tick)) best = int(i);
    return best;
}

}  // namespace ie
