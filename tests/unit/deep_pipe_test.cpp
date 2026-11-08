// tests/unit/deep_pipe_test.cpp -- host-only gate of include/ie/deep_pipe.hpp, the deep prefill pipeline's scheduling
// (2026-11-08). Fake stages carry a value through a ring of N handoff slots the way Flash-Next's stages carry the wide
// state: stage 0 writes chunk k's slot on the calling thread, stage s reads it, checks it is exactly what stage s - 1 left
// for THIS chunk, and rewrites it. Checks:
//   1. every value arrives intact at the tail (no slot overwritten early, every producer before its consumer),
//      every stage sees its chunks in order, no stage ever runs on two threads at once (random per-part sleeps);
//   2. the stages really overlap: stage 1's part of chunk 1 waits on a latch that only stage 2's part of chunk 0 opens --
//      a pipeline that ran the stages in a row would never open it (bounded wait, so a failure cannot hang the suite);
//   3. an error at (stage 2, chunk 3) comes back from drain(), no later part of stage 2 or 3 runs, done() == 3;
//   4. two stages, and a pipe destroyed with work queued (the destructor drains).
#include "ie/deep_pipe.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

using ie::DeepPipe;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? "OK" : "FAIL", what);
    if (!ok) ++g_fail;
}
int64_t value_after(uint32_t stage, size_t chunk) { return int64_t(chunk) * 1000 + stage; }   // what stage s leaves in the slot

struct Fake {
    uint32_t N;
    std::vector<int64_t> ring;                       // the handoff slots
    std::vector<std::atomic<int>> busy;              // [stage] tasks inside run() now
    std::vector<std::vector<size_t>> seen;           // [stage] chunks in the order run() saw them
    std::mutex mu;
    std::atomic<int> bad_value{0}, overlap_violation{0};
    explicit Fake(uint32_t n) : N(n), ring(n, -1), busy(n), seen(n) {}
};

// Runs `chunks` chunks through a pipe; part(s, k) may sleep / block / fail. Returns drain()'s error; *done = done().
template <class Part>
std::string run_chain(Fake& f, size_t chunks, Part part, size_t* done = nullptr) {
    DeepPipe pipe(f.N);
    std::string err;
    for (size_t k = 0; k < chunks; ++k) {
        if (auto e = pipe.slot_free(k); !e.empty()) { err = e; break; }
        f.ring[k % f.N] = value_after(0, k);         // stage 0, on this thread
        pipe.enqueue([&f, k, part](uint32_t s) -> std::string {
            if (f.busy[s].fetch_add(1) != 0) ++f.overlap_violation;
            { std::lock_guard<std::mutex> lk(f.mu); f.seen[s].push_back(k); }
            if (f.ring[k % f.N] != value_after(s - 1, k)) ++f.bad_value;
            std::string e = part(s, k);
            if (e.empty()) f.ring[k % f.N] = value_after(s, k);
            f.busy[s].fetch_sub(1);
            return e;
        });
    }
    const std::string e = pipe.drain();
    if (err.empty()) err = e;
    if (done) *done = pipe.done();
    return err;
}

bool in_order(const std::vector<size_t>& v) {
    for (size_t i = 0; i < v.size(); ++i) if (v[i] != i) return false;
    return true;
}
}  // namespace

int main() {
    std::printf("deep_pipe_test\n");
    {   // 1. values, order, exclusivity
        std::printf("1. four stages, 24 chunks, random part times\n");
        Fake f(4);
        std::mt19937 rng(11);
        std::vector<int> ms(4 * 24);
        for (auto& m : ms) m = int(rng() % 4);
        std::vector<int64_t> tail_seen;
        std::mutex tm;
        const std::string e = run_chain(f, 24, [&](uint32_t s, size_t k) {
            std::this_thread::sleep_for(std::chrono::milliseconds(ms[s * 24 + k]));
            return std::string();
        });
        check(e.empty(), "no error");
        check(f.bad_value == 0, "every stage read exactly what its producer left for that chunk (no slot reused early)");
        check(f.overlap_violation == 0, "no stage ran on two threads at once");
        bool ord = true;
        for (uint32_t s = 1; s < 4; ++s) ord = ord && f.seen[s].size() == 24 && in_order(f.seen[s]);
        check(ord, "every stage saw all 24 chunks, in order");
    }
    {   // 2. overlap
        std::printf("2. the stages overlap\n");
        Fake f(4);
        std::mutex m;
        std::condition_variable cv;
        bool opened = false, timed_out = false;
        const std::string e = run_chain(f, 3, [&](uint32_t s, size_t k) {
            if (s == 2 && k == 0) { std::lock_guard<std::mutex> lk(m); opened = true; cv.notify_all(); }
            if (s == 1 && k == 1) {
                std::unique_lock<std::mutex> lk(m);
                if (!cv.wait_for(lk, std::chrono::seconds(5), [&] { return opened; })) timed_out = true;
            }
            return std::string();
        });
        check(e.empty() && !timed_out, "stage 1's part of chunk 1 ran while stage 2's part of chunk 0 did");
    }
    {   // 3. error
        std::printf("3. an error at stage 2, chunk 3\n");
        Fake f(4);
        size_t done = 0;
        const std::string e = run_chain(f, 10, [&](uint32_t s, size_t k) {
            return (s == 2 && k == 3) ? std::string("boom") : std::string();
        }, &done);
        check(e == "boom", "drain() returns the stage's error");
        bool none_after = true;
        for (size_t k : f.seen[2]) none_after = none_after && k <= 3;
        for (size_t k : f.seen[3]) none_after = none_after && k < 3;
        check(none_after, "no later part of stage 2 ran, stage 3 never saw chunk 3 or later");
        check(done == 3, "done() == 3 (the chunks the tail finished)");
        check(f.overlap_violation == 0 && f.bad_value == 0, "exclusive and intact up to the error");
    }
    {   // 4. two stages; destructor drains
        std::printf("4. two stages; a pipe destroyed with work queued\n");
        Fake f(2);
        const std::string e = run_chain(f, 9, [&](uint32_t, size_t) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            return std::string();
        });
        check(e.empty() && f.bad_value == 0 && in_order(f.seen[1]) && f.seen[1].size() == 9, "two stages: intact, in order");
        std::atomic<int> ran{0};
        {
            DeepPipe p(3);
            for (int k = 0; k < 4; ++k)
                p.enqueue([&ran](uint32_t) { std::this_thread::sleep_for(std::chrono::milliseconds(5)); ++ran; return std::string(); });
        }   // no drain(): the destructor must wait
        check(ran.load() == 8, "the destructor waited for every queued part (4 chunks x 2 stages)");
    }
    std::printf("deep_pipe_test: %s (%d failures)\n", g_fail ? "FAIL" : "OK", g_fail);
    return g_fail ? 1 : 0;
}
