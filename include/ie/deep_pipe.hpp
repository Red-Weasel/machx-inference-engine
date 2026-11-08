// include/ie/deep_pipe.hpp -- the scheduling of the deep prefill pipeline (2026-11-08: Flash-Next's N-stage chain, GitHub
// issue 3's finding 7). Host-only, header-only; tests/unit/deep_pipe_test.cpp is its gate.
//
// Stage 0 reads a chunk on the calling thread; enqueue(run) then queues that chunk on stages 1..N-1, each part on its own
// task, so chunk k is on stage s while chunk k + 1 is on stage s - 1. run(s) is called on stage s's task once
//   (1) its producer -- stage s - 1's part of the same chunk -- has returned, and
//   (2) stage s's own part of the previous chunk has returned,
// and the task waits for both whatever they returned: a stage is never driven by two threads, and no task ends while its
// stage is still busy (the first race the reporter hit). An error travels along the chunk and down the stage: no later part
// of a failed stage runs. The futures are shared and captured by value (the second race: a future read while the calling
// thread reassigned it). The caller's handoff buffer for chunk k is ring slot k % N: slot_free(k) waits until the tail has
// finished chunk k - N, which, by (1) and (2), every stage has. drain() waits for every stage; the destructor drains.
#pragma once

#include <cstdint>
#include <functional>
#include <future>
#include <string>
#include <vector>

namespace ie {

class DeepPipe {
public:
    using Future = std::shared_future<std::string>;
    using Run = std::function<std::string(uint32_t stage)>;

    explicit DeepPipe(uint32_t stages) : n_(stages < 2 ? 2 : stages), last_(n_) {}
    DeepPipe(const DeepPipe&) = delete;
    DeepPipe& operator=(const DeepPipe&) = delete;
    ~DeepPipe() { (void)drain(); }

    uint32_t stages() const noexcept { return n_; }

    // Before stage 0 writes chunk k's ring slot (k % stages()): waits for the tail's part of chunk k - stages() and returns
    // its error, if any ("" when the slot was never used).
    std::string slot_free(size_t k) {
        if (k < n_ || k - n_ >= tail_.size() || !tail_[k - n_].valid()) return {};
        return tail_[k - n_].get();
    }

    // Queues one chunk (stage 0's part done) on stages 1..N-1. Returns the tail's future for it.
    Future enqueue(Run run) {
        Future prev;   // stage 1's producer is stage 0, already done on the calling thread
        for (uint32_t s = 1; s < n_; ++s) {
            const Future self = last_[s];
            prev = std::async(std::launch::async, [s, prev, self, run]() -> std::string {
                const std::string pe = prev.valid() ? prev.get() : std::string();
                const std::string se = self.valid() ? self.get() : std::string();
                if (!se.empty()) return se;
                if (!pe.empty()) return pe;
                return run(s);
            }).share();
            last_[s] = prev;
        }
        tail_.push_back(prev);
        return prev;
    }

    // Every stage idle again: the first error, if any.
    std::string drain() {
        std::string err;
        for (auto& f : last_)
            if (f.valid()) {
                const std::string e = f.get();
                if (err.empty() && !e.empty()) err = e;
            }
        return err;
    }

    // Chunks the tail finished without an error, in order (after drain()).
    size_t done() {
        size_t d = 0;
        for (auto& f : tail_) {
            if (!f.valid() || !f.get().empty()) break;
            ++d;
        }
        return d;
    }

private:
    uint32_t n_;
    std::vector<Future> last_;   // [stage] its latest queued part (stage 0's slot unused)
    std::vector<Future> tail_;   // [chunk] the tail's part
};

}  // namespace ie
