// include/ie/mimo26_lanes.hpp — MiMo-V2.6 lanes (P4 B1, ~/ds41_work/p60/p4_mimo_batching_design.md section 2.1): the host
// bookkeeping of several sequences sharing one loaded model. A LANE is one sequence's state: on the device its own full-layer
// K/V, SWA rings and positions (Mimo26Forward::select_lane) and its drafter context ring (Mimo26DFlash::select_lane); on the
// host what is below -- its prompt and live ids, where its prefill stands, its sampler state (rng, the repetition window),
// its stop state, its vitals window and the outbox of committed ids. Lane 0 is the conversation the engine has always had
// (its device buffers are the ones a one-lane build allocates, so one lane is byte-identical to the pre-lane engine).
// B1 steps lanes SERIALLY -- one lane's forward at a time (round robin, Mimo26LaneSet::next) -- so every lane runs exactly
// its solo kernel sequence: with the CPU expert leg off (IE_DS41_CPU_MISS=0) a lane's tokens and logits are bit-identical
// to the same prompt run alone. No SYCL here: unit-tested on the CPU by tests/unit/mimo26_lanes_test.cpp.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ie {

struct VitalsWindow;   // ie/vitals.hpp

// One layer of a card as the lanes' caches see it.
struct Mimo26LaneLayer { bool swa = false; uint32_t n_kv = 0; };
// The device bytes one lane's caches take on a card, exactly as Mimo26Forward allocates them: a full layer (or an SWA layer
// with linear caches, ring == 0) keeps K [n_kv * ctx + 64 rows of slack, head_dim] and V [n_kv * ctx, v_row] with V padded
// to head_dim on the full layers; an SWA layer with a ring keeps `ring` slots per kv head instead of ctx.
uint64_t mimo26_lane_kv_bytes(const std::vector<Mimo26LaneLayer>& layers, uint32_t head_dim, uint32_t v_head_dim, uint32_t ring,
                              uint32_t ctx);
// Whether `extra` more lanes of `per_lane` bytes each fit in `free_bytes` with `reserve` left over: "" = they fit, else the
// refusal with the numbers (MiB) and what to change.
std::string mimo26_lanes_fit(uint64_t free_bytes, uint64_t per_lane, uint32_t extra, uint64_t reserve, uint32_t card, uint32_t lane_ctx);

struct Mimo26Lane {
    enum class Phase : uint8_t { kIdle, kPrefill, kDecode };
    uint32_t             index = 0;
    Phase                phase = Phase::kIdle;
    std::vector<int32_t> prompt;          // the request's ids
    std::vector<int32_t> live;            // the ids at the lane's positions [0, n_pos) -- kept while idle, for the prefix step
    uint32_t             prefill_at = 0;  // the next prompt position to prefill (the prefix step sets it to what it reused)
    uint32_t             max_new = 0, n_new = 0;
    int32_t              next = -1;       // the sampled id whose forward is due (decode)
    uint64_t             rng = 0;         // the sampler's state
    std::vector<int32_t> recent;          // the sampler's repetition window: the last kRecent ids (prompt tail + output)
    VitalsWindow*        vitals = nullptr;
    std::vector<int32_t> outbox;          // committed ids the consumer has not taken yet
    std::string          finish;          // "" while running; "stop" (an eos id) or "length"
    uint64_t             tick = 0;        // the lane's last step (larger = more recent)
    static constexpr size_t kRecent = 512;
};

// A sampled id into the lane (the engine's emit rule): an eos id (unless ignore_eos) ends the lane with "stop" and is not
// committed; otherwise it goes to the outbox and the repetition window, and the lane ends with "length" at max_new.
// Returns true while the lane goes on decoding; the id to feed next is then l.next.
bool mimo26_lane_commit(Mimo26Lane& l, int32_t id, const std::vector<int32_t>& eos, bool ignore_eos);

// The lanes of one loaded model and the serial round robin over the busy ones.
class Mimo26LaneSet {
public:
    explicit Mimo26LaneSet(uint32_t n_lanes);
    uint32_t    size() const { return uint32_t(lanes_.size()); }
    Mimo26Lane& lane(uint32_t i) { return lanes_[i]; }
    const Mimo26Lane& lane(uint32_t i) const { return lanes_[i]; }
    uint32_t    busy() const;   // lanes prefilling or decoding
    // A new sequence into an idle lane -- the least recently used one (its live ids are what it keeps for reuse) -- or -1
    // when every lane is busy. The lane starts prefilling at 0 with a fresh sampler (rng = seed, the window = the prompt's
    // tail) and an empty outbox; its live ids stay until the caller's prefix step cuts them.
    int         admit(std::vector<int32_t> prompt, uint32_t max_new, uint64_t seed, VitalsWindow* vitals = nullptr);
    // The lane to step next: the first busy lane after the one stepped last, in index order (-1 = none busy). Stamps its tick.
    int         next();
    // The lane's sequence has ended (finish set, or cancelled): it goes idle, keeping its live ids.
    void        release(uint32_t i);
    // The least recently used idle lane (-1 = none): the one to park in a host slot when a sequence needs a lane.
    int         lru_idle() const;

private:
    std::vector<Mimo26Lane> lanes_;
    uint32_t last_ = ~0u;
    uint64_t tick_ = 0;
};

}  // namespace ie
