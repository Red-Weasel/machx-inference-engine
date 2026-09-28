// include/ie/q27_lanes.hpp -- the Qwen3.8-27B split (qwen35, Qwen35SplitModel) request lanes: the host-side rules (P4 B18,
// docs/lanes/LANES_SERVE.md).
//
// The 27B runs as ONE model object over two cards (embedding + the first half of the layers on card 0, the rest + the head on
// card 1). A request lane is one sequence's KV + DeltaNet state on both cards (Qwen35SplitModel::select_lane; lane 0 = the
// state load() allocated at --ctx, lanes 1..n-1 at --slot-ctx); the shared lanes module (ie/lanes_serve.hpp) serves them. It
// replaces the joint-step path (slot banks + Engine::BatchStepper) unless IE_QWEN35_LANES=0.
//
// --parallel 1's prefill on this model (Engine::generate, prefill_to) is NOT the crown's plain chunk loop: a range longer than
// pf_chunk runs PIPELINED (forward_pipelined: every chunk through the prefill kernels, even a 1-row last chunk), a shorter one
// runs through forward() (a 1-row chunk there takes the DECODE kernels: fa2 decode attention). A lane must cut the same pieces
// AND run each 1-row piece through the same kernels, so the plan carries the kind: pieces of a pipelined range are `pk`
// (prefill kernels). A 1-row pk piece never goes through the lane pipe (the rows mode would group it with decode rows): the
// pipe's part of the prompt ends before it and prompt_end runs the rest in the turn.
//
// No SYCL here: unit-tested on the CPU (tests/unit/q27_lanes_test.cpp).
#pragma once

#include "ie/lanes_serve.hpp"

#include <cstdint>
#include <cstdlib>
#include <span>
#include <string>
#include <vector>

namespace ie {

// The 27B split's prefill chunk at --parallel 1 (Engine::generate, the kQwen35Dense split branch of its pf_chunk rule):
// min(512, max_ctx); IE_QWEN35_PREFILL_CHUNK (1..max_ctx) overrides. The env value is passed in (nullptr = unset).
inline uint32_t q27_prefill_chunk(uint32_t max_ctx, const char* override_chunk) {
    uint32_t c = max_ctx < 512u ? max_ctx : 512u;
    if (override_chunk) {
        const int v = std::atoi(override_chunk);
        if (v >= 1 && uint32_t(v) <= max_ctx) c = uint32_t(v);
    }
    return c;
}

// Lanes 1..n-1's context: --slot-ctx, 0 = 65,536 (the joint-step banks' default), never above --ctx.
inline uint32_t q27_lane_ctx(uint32_t slot_ctx, uint32_t max_ctx) {
    const uint32_t c = slot_ctx ? slot_ctx : 65536u;
    return c < max_ctx ? c : max_ctx;
}

// One prefill piece: rows [pos0, pos0 + rows); pk = the prefill kernels even at 1 row (a piece of a pipelined range).
struct Q27Piece { uint32_t pos0 = 0, rows = 0; bool pk = false; };
inline bool operator==(const Q27Piece& a, const Q27Piece& b) { return a.pos0 == b.pos0 && a.rows == b.rows && a.pk == b.pk; }

// --parallel 1's prefill as a sequence of ops: pieces, and the prompt-cache inserts between them (the shared prefix, the
// snapshot boundary), each at the position the lane's state then ends at.
struct Q27Op {
    enum Kind : uint8_t { kPiece, kShared, kSnap } kind = kPiece;
    Q27Piece piece;   // kPiece
    uint32_t at = 0;  // kShared / kSnap
};

// prefill_to(end) from p: pipelined (pk) when the range is longer than one chunk and the pipeline is on, else forward() chunk
// by chunk; the same pf_chunk-row cuts either way.
inline void q27_range(uint32_t p, uint32_t end, uint32_t pf_chunk, bool pipeline, std::vector<Q27Op>& out) {
    if (end <= p) return;
    const uint32_t piece = pf_chunk ? pf_chunk : 1u;
    const bool pk = pipeline && end - p > piece;
    for (uint32_t q = p; q < end; q += piece) {
        Q27Op o;
        o.piece = Q27Piece{q, end - q < piece ? end - q : piece, pk};
        out.push_back(o);
    }
}

// The whole prefill of a T-token prompt with [0, reused) restored: --parallel 1's prefill_to(share_at) + the shared insert (when
// share_at > reused), prefill_to(snap_at) + the snapshot insert (when the cache is on and snap_at > reused), prefill_to(T).
// share_at: 0 = none (it is < snap_at when set: shared_prefix_boundary).
inline std::vector<Q27Op> q27_prefill_ops(uint32_t T, uint32_t reused, uint32_t snap_at, uint32_t share_at, bool cache_on,
                                          uint32_t pf_chunk, bool pipeline) {
    std::vector<Q27Op> ops;
    uint32_t pos = reused;
    if (share_at > reused) {
        q27_range(pos, share_at, pf_chunk, pipeline, ops);
        pos = share_at;
        Q27Op o; o.kind = Q27Op::kShared; o.at = share_at;
        ops.push_back(o);
    }
    q27_range(pos, snap_at, pf_chunk, pipeline, ops);
    if (snap_at > pos) pos = snap_at;
    if (cache_on && snap_at > reused) { Q27Op o; o.kind = Q27Op::kSnap; o.at = snap_at; ops.push_back(o); }
    q27_range(pos, T, pf_chunk, pipeline, ops);
    return ops;
}

// The lane plan from the ops: the leading pieces go through the lane pipe up to Tp -- the snapshot insert or a 1-row pk piece,
// whichever comes first (else T); a shared insert strictly inside [reused, Tp) is the plan's mark (the module's
// LanesModel::mark), one at Tp or later is prompt_end's.
inline LanesPlan q27_plan(const std::vector<Q27Op>& ops, uint32_t reused, uint32_t T) {
    LanesPlan p;
    p.Tp = T;
    uint32_t pos = reused, shared = 0;
    for (const Q27Op& o : ops) {
        if (o.kind == Q27Op::kSnap) { p.Tp = pos; break; }
        if (o.kind == Q27Op::kShared) { shared = o.at; continue; }
        if (o.piece.rows == 1 && o.piece.pk) { p.Tp = pos; break; }
        p.chunks.emplace_back(o.piece.pos0, o.piece.rows);
        pos = o.piece.pos0 + o.piece.rows;
    }
    if (shared > reused && shared < p.Tp) p.mark = shared;
    return p;
}

// What prompt_end runs, in order: the ops at or past Tp (the pieces starting there, the inserts due there or later).
inline std::vector<Q27Op> q27_rest(const std::vector<Q27Op>& ops, uint32_t Tp) {
    std::vector<Q27Op> r;
    for (const Q27Op& o : ops)
        if (o.kind == Q27Op::kPiece ? o.piece.pos0 >= Tp : o.at >= Tp) r.push_back(o);
    return r;
}

// prompt_end's walk over the rest, in order: each run of consecutive pieces through run(span of pieces) (the arch pipelines the
// cards over them), each insert through insert(op). The first error ends the walk. ran = whether any piece ran.
template <class RunFn, class InsertFn>
std::string q27_walk_rest(const std::vector<Q27Op>& rest, RunFn&& run, InsertFn&& insert, bool& ran) {
    ran = false;
    std::vector<Q27Piece> seg;
    auto flush = [&]() -> std::string {
        if (seg.empty()) return {};
        ran = true;
        std::string e = run(std::span<const Q27Piece>(seg));
        seg.clear();
        return e;
    };
    for (const Q27Op& o : rest) {
        if (o.kind == Q27Op::kPiece) { seg.push_back(o.piece); continue; }
        if (auto e = flush(); !e.empty()) return e;
        insert(o);
    }
    return flush();
}

}  // namespace ie
