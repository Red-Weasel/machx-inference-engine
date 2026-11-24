// tests/unit/dense_lanes_test.cpp -- the dense fleet class's request-lane host rules (include/ie/dense_lanes.hpp). CPU only.
#include "ie/dense_lanes.hpp"

#include <cstdio>

static int fails = 0;
#define CHECK(cond) do { if (!(cond)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++fails; } } while (0)

int main() {
    // Nanbeige4.3-33B: 52 layers, 8 kv heads (4 a card on two cards), head dim 128
    const ie::DenseLaneShape two{52, 4, 128}, one{52, 8, 128};
    CHECK(ie::dense_lane_bytes(two, 16384) == 2ull * 52 * 4 * 16384 * 128 * 2);   // 1.74 GB a card
    CHECK(ie::dense_lane_bytes(one, 16384) == 2 * ie::dense_lane_bytes(two, 16384));
    CHECK(ie::dense_lane_bytes(two, 65536) == 4 * ie::dense_lane_bytes(two, 16384));
    // the lane context
    CHECK(ie::dense_lane_ctx(0, 65536) == 65536);
    CHECK(ie::dense_lane_ctx(0, 32768) == 32768);
    CHECK(ie::dense_lane_ctx(16384, 65536) == 16384);
    CHECK(ie::dense_lane_ctx(100000, 65536) == 65536);
    // the fit: 3 extra lanes of 1.625 GiB (1.74 GB) + a 2 GiB reserve need 6.875 GiB
    const uint64_t lb = ie::dense_lane_bytes(two, 16384), G = 1ull << 30;
    CHECK(ie::dense_lanes_fit(7 * G, lb, 4, 16384, 2 * G, 0).empty());
    CHECK(!ie::dense_lanes_fit(6 * G, lb, 4, 16384, 2 * G, 0).empty());
    CHECK(ie::dense_lanes_fit(0, lb, 1, 16384, 2 * G, 0).empty());                 // one lane allocates nothing
    CHECK(ie::dense_lanes_fit(6 * G, lb, 4, 16384, 2 * G, 1).find("card 1") != std::string::npos);
    // the plan: plain pf_chunk pieces over [reused, Tp)
    {
        const ie::LanesPlan p = ie::dense_lane_plan(1300, 0, 1290, true, 512);      // cold, the cache on: Tp = snap_at
        CHECK(p.Tp == 1290); CHECK(p.mark == 0); CHECK(p.snap == 0);
        CHECK(p.chunks.size() == 3);
        CHECK(p.chunks[0] == ie::LanesChunk(0, 512)); CHECK(p.chunks[1] == ie::LanesChunk(512, 512)); CHECK(p.chunks[2] == ie::LanesChunk(1024, 266));
    }
    {
        const ie::LanesPlan p = ie::dense_lane_plan(1300, 1024, 1290, true, 512);   // 1024 restored: one piece to the boundary
        CHECK(p.Tp == 1290); CHECK(p.chunks.size() == 1); CHECK(p.chunks[0] == ie::LanesChunk(1024, 266));
    }
    {
        // the boundary itself restored: nothing to insert, so the tail goes through the pipe (the 27B's rule: no snap op when
        // snap_at == reused)
        const ie::LanesPlan p = ie::dense_lane_plan(1300, 1290, 1290, true, 512);
        CHECK(p.Tp == 1300); CHECK(p.chunks.size() == 1); CHECK(p.chunks[0] == ie::LanesChunk(1290, 10));
    }
    {
        const ie::LanesPlan p = ie::dense_lane_plan(1300, 0, 1290, false, 512);     // the cache off: the whole prompt
        CHECK(p.Tp == 1300); CHECK(p.chunks.size() == 3); CHECK(p.chunks[2] == ie::LanesChunk(1024, 276));
    }
    {
        const ie::LanesPlan p = ie::dense_lane_plan(1300, 0, 0, true, 512);         // no boundary: the whole prompt
        CHECK(p.Tp == 1300);
    }
    {
        const ie::LanesPlan p = ie::dense_lane_plan(100, 0, 100, true, 0);          // pf_chunk 0 = 1-row pieces
        CHECK(p.chunks.size() == 100); CHECK(p.chunks[99] == ie::LanesChunk(99, 1));
    }
    if (fails) { std::fprintf(stderr, "%d failure(s)\n", fails); return 1; }
    std::printf("dense_lanes_test: PASS\n");
    return 0;
}
