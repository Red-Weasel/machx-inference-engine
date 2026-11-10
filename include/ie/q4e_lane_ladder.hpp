// include/ie/q4e_lane_ladder.hpp -- the bookkeeping of a Flash-Next request lane's in-place restart points (2026-11-10).
//
// The lane's prompt cache is all or nothing: its snapshot is reused only when every one of its tokens is a prefix of the new
// prompt, because the recurrent state cannot be truncated. One changed token anywhere in the history then re-reads the whole
// prompt (the owner's 37 h session at --parallel 4 --spec: 53 full re-reads, 7,711 s, up to 332 s for 215K tokens). The KV and
// indexer rows below the change are still in place; what is missing is the recurrent state at a depth below it. A restart
// point is a copy of that state (Qwen4ExpModel::StateCopy, ~40 MiB a stage, pinned host RAM) at a depth inside the RECORD,
// the tokens the lane's rows hold from position 0.
//
// This class is the record and the slots' depths only -- no device state, so tests/unit/q4e_lane_ladder_test.cpp gates the
// rules on the host. Slots: n_regular regular ones (a ring over the multiples of `step` a prompt read passes), then n_turns
// turn slots (the last prompt boundaries). The engine calls:
//   reset()            the lane starts a new sequence
//   keep(d)            the lane's state was set to depth d (a restore): the record and every slot above d go
//   regular_slot(..)   which slot the copy of a chunk [p0, p0 + n) goes to, or -1 (the chunk passes no multiple of step)
//   append(..)         a chunk finished every stage: its tokens join the record (false: not contiguous, nothing changes);
//                      append_piece takes the chunk's own token pointer (the lane pipe's steps carry that)
//   set(slot, d)       every stage holds the slot's copy at depth d
//   turn_slot(Tp)      the slot for the prompt boundary Tp, or -1 (not in the record, or a slot already holds it)
//   best(ids, &slot)   the deepest slot at or below what `ids` shares with the record, leaving at least one row to read
//   own(ids)           best() as a reason to CHOOSE this lane: only for a prompt that looks like this lane's conversation
#pragma once

#include <algorithm>
#include <cstdint>
#include <span>
#include <vector>

namespace ie {

struct Q4eLaneLadder {
    uint32_t step = 0, n_regular = 0, n_turns = 0;
    std::vector<int32_t> rec;
    std::vector<uint32_t> depth;   // [n_regular + n_turns], 0 = no copy
    uint32_t turn = 0;

    void init(uint32_t step_, uint32_t n_regular_, uint32_t n_turns_) {
        step = step_; n_regular = n_regular_; n_turns = n_turns_;
        rec.clear(); depth.assign(size_t(n_regular) + n_turns, 0u); turn = 0;
    }
    bool on() const { return step != 0 && !depth.empty(); }
    void reset() { rec.clear(); std::fill(depth.begin(), depth.end(), 0u); }
    void keep(uint32_t d) {
        if (rec.size() > d) rec.resize(d);
        for (auto& x : depth) if (x > rec.size()) x = 0;
    }
    bool whole_to(uint32_t pos) const { return rec.size() == pos; }
    int regular_slot(uint32_t p0, uint32_t n) const {
        if (!on() || !n_regular || n == 0) return -1;
        const uint32_t end = p0 + n;
        if (end / step == p0 / step) return -1;
        return int((end / step - 1) % n_regular);
    }
    bool append(const int32_t* ids, uint32_t p0, uint32_t n) {
        if (rec.size() != p0) return false;
        rec.insert(rec.end(), ids + p0, ids + p0 + n);
        return true;
    }
    bool append_piece(const int32_t* piece, uint32_t p0, uint32_t n) {   // the piece's own tokens (piece[0] is position p0)
        if (rec.size() != p0) return false;
        rec.insert(rec.end(), piece, piece + n);
        return true;
    }
    void set(int slot, uint32_t d) {
        if (slot >= 0 && size_t(slot) < depth.size()) depth[size_t(slot)] = d <= rec.size() ? d : 0u;
    }
    int turn_slot(uint32_t Tp) {
        if (!on() || !n_turns || Tp == 0 || rec.size() < Tp) return -1;
        for (uint32_t d : depth) if (d == Tp) return -1;
        const int s = int(n_regular + (turn++ % n_turns));
        depth[size_t(s)] = 0;
        return s;
    }
    uint32_t shared(std::span<const int32_t> ids) const {   // the leading tokens `ids` shares with the record
        const size_t n = std::min(rec.size(), ids.size());
        size_t l = 0;
        while (l < n && rec[l] == ids[l]) ++l;
        return uint32_t(l);
    }
    // best() for the CHOICE of a lane: a restart point makes this lane the prompt's own only when the prompt looks like this
    // lane's conversation -- about as long as the record or longer (a conversation continued with its history changed), or
    // sharing at least half of the record. Otherwise 0: another conversation that merely shares a long prefix (the same
    // document, the same system prompt) must not be steered here and push the live conversation out of its lane. (Once a
    // lane IS chosen, by any rule, best() is used: reading from a copy is then free.)
    uint32_t own(std::span<const int32_t> ids, uint32_t slack = 4096) const {
        const uint32_t b = best(ids);
        if (!b) return 0;
        const size_t S = rec.size();
        if (ids.size() + slack >= S || size_t(shared(ids)) * 2 >= S) return b;
        return 0;
    }
    uint32_t best(std::span<const int32_t> ids, int* slot = nullptr) const {
        if (!on() || ids.empty()) return 0;
        const size_t n = std::min(rec.size(), ids.size());
        size_t l = 0;
        while (l < n && rec[l] == ids[l]) ++l;
        const uint32_t lim = uint32_t(std::min<size_t>(l, ids.size() - 1));
        uint32_t b = 0;
        for (size_t s = 0; s < depth.size(); ++s)
            if (depth[s] && depth[s] <= lim && depth[s] > b) { b = depth[s]; if (slot) *slot = int(s); }
        return b;
    }
};

}  // namespace ie
