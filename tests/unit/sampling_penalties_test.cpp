#undef NDEBUG
#include "ie/ops.hpp"
#include <cassert>
#include <cmath>
#include <cstdio>
int main() {
    const auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devices.empty()) return 77;
    sycl::queue q(devices.front()); // out of order: explicit dependencies matter
    auto* logits = sycl::malloc_shared<sycl::half>(4, q);
    auto* ids = sycl::malloc_shared<int32_t>(6, q);
    assert(logits && ids);
    logits[0] = 8; logits[1] = -4; logits[2] = 3; logits[3] = 0;
    ids[0] = 0; ids[1] = 0; ids[2] = 0; ids[3] = 1; ids[4] = -1; ids[5] = 99;
    // Duplicated history IDs incur repetition once, presence once, frequency
    // once per occurrence. Invalid history IDs cannot access outside logits.
    ie::sampling_penalties(q, logits, 4, ids, 6, 2.f, .5f, .25f).wait_and_throw();
    assert(float(logits[0]) == 2.75f); // 8/2 - .5 - 3*.25
    assert(float(logits[1]) == -8.75f); // -4*2 - .5 - .25
    assert(float(logits[2]) == 3.f && float(logits[3]) == 0.f);
    auto filled = q.fill(logits, sycl::half(5), 4);
    ie::sampling_penalties(q, logits, 4, ids, 6, 1.f, -1.f, -.5f, {filled}).wait_and_throw();
    assert(float(logits[0]) == 7.5f && float(logits[1]) == 6.5f);
    assert(float(logits[2]) == 5.f);
    ie::sampling_penalties(q, logits, 4, ids, 0, 2.f, 1.f, 1.f).wait_and_throw();
    assert(float(logits[0]) == 7.5f);
    ie::repetition_penalty(q, logits, 4, ids, 6, 2.f).wait_and_throw();
    assert(float(logits[0]) == 3.75f && float(logits[1]) == 3.25f);
    sycl::free(ids, q); sycl::free(logits, q);
    std::puts("sampling penalties numerical and dependency checks passed");
}
