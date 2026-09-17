// include/ie/server_admission.hpp — bounded request admission, the device-fault
// latch and the shutdown flag for the OpenAI server. Header-only and free of
// any Engine dependency so the concurrency contract is unit-testable without a
// model or a GPU (tests/unit/server_admission_test.cpp).
#pragma once
#include <algorithm>
#include <atomic>
#include <cctype>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>

namespace ie {

// Up to `parallel` requests run at once; at most `max_queue` more may WAIT for
// a run slot. Anything beyond is refused immediately (the server answers 429)
// so the HTTP worker pool can never fill up with blocked waiters and /health,
// /props and the 429s themselves keep getting answered under load.
class Admission {
public:
    Admission(uint32_t parallel, uint32_t max_queue)
        : avail_(parallel ? parallel : 1u), max_queue_(max_queue) {}

    // Take a run slot. Returns true once a slot is held (possibly after
    // waiting); returns false WITHOUT waiting when the wait queue is full, and
    // false (waking any waiter early) once shutdown() was called.
    bool acquire() {
        std::unique_lock<std::mutex> l(mu_);
        if (stopping_) return false;
        if (avail_ == 0) {
            if (waiting_ >= max_queue_) return false;
            ++waiting_;
            cv_.wait(l, [&] { return avail_ > 0 || stopping_.load(); });
            --waiting_;
            if (stopping_) return false;
        }
        --avail_;
        ++inflight_;
        return true;
    }
    void release() {
        { std::lock_guard<std::mutex> l(mu_); ++avail_; --inflight_; }
        cv_.notify_one();
    }
    // Wake every waiter with a refusal and refuse all later acquires; in-flight
    // generations poll stopping() through their token callbacks and abort.
    void shutdown() {
        { std::lock_guard<std::mutex> l(mu_); stopping_ = true; }
        cv_.notify_all();
    }
    bool     stopping()  const noexcept { return stopping_.load(); }
    uint32_t inflight()  const noexcept { return inflight_.load(); }
    uint32_t queued()    const noexcept { return waiting_.load(); }
    uint32_t max_queue() const noexcept { return max_queue_; }

private:
    std::mutex              mu_;
    std::condition_variable cv_;
    uint32_t                avail_;
    const uint32_t          max_queue_;
    std::atomic<uint32_t>   waiting_{0};    // written under mu_, read lock-free
    std::atomic<uint32_t>   inflight_{0};   // written under mu_, read lock-free
    std::atomic<bool>       stopping_{false};
};

// Latched process-wide device fault. After a forward reports a lost/reset
// device nothing in this engine re-creates the context, so every later request
// would fail the same way while /health kept saying "ok". Once latched, the
// server answers 503 on /health and on every generation until restarted.
class DeviceFaultLatch {
public:
    // Heuristic on the error text: Level Zero / UR / SYCL spell a lost device
    // as UR_RESULT_ERROR_DEVICE_LOST, ZE_RESULT_ERROR_DEVICE_LOST, "device lost"
    // or a device reset. Out-of-memory and shape errors are NOT device loss.
    static bool looks_like_device_loss(std::string_view msg) {
        std::string s(msg);
        std::transform(s.begin(), s.end(), s.begin(),
                       [](unsigned char c) { return char(std::tolower(c)); });
        for (const char* needle : {"device_lost", "device lost", "device_reset", "device reset"})
            if (s.find(needle) != std::string::npos) return true;
        return false;
    }
    // Latch iff `msg` looks like device loss; returns whether it did.
    bool observe(std::string_view msg) {
        if (!looks_like_device_loss(msg)) return false;
        latch(msg);
        return true;
    }
    void latch(std::string_view reason) {
        std::lock_guard<std::mutex> l(mu_);
        if (!faulted_.exchange(true)) reason_ = std::string(reason);   // first cause wins
    }
    bool        faulted() const noexcept { return faulted_.load(); }
    std::string reason()  const { std::lock_guard<std::mutex> l(mu_); return reason_; }

private:
    std::atomic<bool>  faulted_{false};
    mutable std::mutex mu_;
    std::string        reason_;
};

}  // namespace ie
