// tests/unit/server_admission_test.cpp — Admission (bounded wait queue,
// shutdown wake-up, lock-free stats) and DeviceFaultLatch. Host-only.
#undef NDEBUG
#include "ie/server_admission.hpp"
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

namespace {
// Poll a predicate with a bounded wall-clock budget (no GPU, no model; the
// threads here only block on the Admission condition variable).
template <typename F>
bool eventually(F f, int ms = 2000) {
    const auto t0 = std::chrono::steady_clock::now();
    while (!f()) {
        if (std::chrono::steady_clock::now() - t0 > std::chrono::milliseconds(ms)) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}
}  // namespace

int main() {
    // 1. parallel=1, max_queue=2: one runs, two wait, the fourth is refused
    //    immediately, release() hands the slot to a waiter, shutdown() wakes
    //    the remaining waiter with a refusal and refuses new arrivals.
    {
        ie::Admission adm(1, 2);
        assert(adm.admitted() == 0);
        assert(adm.acquire());                       // runs
        assert(adm.inflight() == 1 && adm.queued() == 0 && adm.admitted() == 1);
        std::atomic<int> got{0}, refused{0}, done{0};
        auto waiter = [&] {
            if (adm.acquire()) ++got; else ++refused;
            ++done;
        };
        std::thread t1(waiter), t2(waiter);
        assert(eventually([&] { return adm.queued() == 2; }));
        const auto t0 = std::chrono::steady_clock::now();
        assert(!adm.acquire());                      // queue full → immediate false
        assert(std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(500));
        assert(adm.queued() == 2 && adm.inflight() == 1 && adm.admitted() == 1);   // a refusal is not admitted
        adm.release();                               // slot → one waiter
        assert(eventually([&] { return got.load() == 1 && adm.queued() == 1; }));
        assert(adm.inflight() == 1 && adm.admitted() == 2);                        // the waiter got the slot
        adm.shutdown();                              // remaining waiter refused
        assert(eventually([&] { return done.load() == 2; }));
        assert(got.load() == 1 && refused.load() == 1);
        assert(!adm.acquire() && adm.stopping());
        assert(adm.admitted() == 2);                                                // neither refusal counted
        t1.join(); t2.join();
        std::puts("admission: bounded queue / release / shutdown / admitted count OK");
    }
    // 2. max_queue=0: never wait — a second arrival is refused at once.
    {
        ie::Admission adm(1, 0);
        assert(adm.acquire());
        assert(!adm.acquire());
        adm.release();
        assert(adm.acquire());
        assert(adm.admitted() == 2);                 // two runs; the refusal between them is not one
        std::puts("admission: max_queue=0 OK");
    }
    // 3. parallel=0 is clamped to 1 (never a deadlocked server).
    {
        ie::Admission adm(0, 1);
        assert(adm.acquire());
        std::puts("admission: parallel clamp OK");
    }
    // 4. Device-fault latch: only device-loss texts latch; first cause wins.
    {
        ie::DeviceFaultLatch f;
        assert(ie::DeviceFaultLatch::looks_like_device_loss("UR_RESULT_ERROR_DEVICE_LOST"));
        assert(ie::DeviceFaultLatch::looks_like_device_loss("ZE_RESULT_ERROR_DEVICE_LOST (0x70000001)"));
        assert(ie::DeviceFaultLatch::looks_like_device_loss("forward: Device lost during submit"));
        assert(ie::DeviceFaultLatch::looks_like_device_loss("error: deepseek4: DEVICE_RESET"));
        assert(!ie::DeviceFaultLatch::looks_like_device_loss("PI_ERROR_OUT_OF_RESOURCES"));
        assert(!ie::DeviceFaultLatch::looks_like_device_loss("error: slot: bank_store failed"));
        assert(!f.faulted());
        assert(!f.observe("out of memory"));
        assert(!f.faulted());
        assert(f.observe("forward: UR_RESULT_ERROR_DEVICE_LOST"));
        assert(f.faulted() && f.reason() == "forward: UR_RESULT_ERROR_DEVICE_LOST");
        f.observe("later: device lost again");
        assert(f.reason() == "forward: UR_RESULT_ERROR_DEVICE_LOST");
        std::puts("fault latch OK");
    }
    std::puts("server_admission_test: all OK");
    return 0;
}
