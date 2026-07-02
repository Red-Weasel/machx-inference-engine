// src/core/allocator.cpp

#include <cstdlib>
#include "ie/allocator.hpp"

#include <algorithm>
#include <string>
#include <vector>

namespace ie {

std::string DeviceAllocator::init(std::string_view name_filter, uint32_t ordinal) {
    // Env override: device names drift across drivers (the old "0xe223" stopped
    // matching the L0-V2 "Intel(R) Arc(TM) Pro B70 Graphics" name). IE_GPU_FILTER
    // lets the user re-point the filter without a rebuild.
    if (const char* f = std::getenv("IE_GPU_FILTER"); f && *f) name_filter = f;
    sycl::device picked;
    bool found = false;
    uint32_t seen = 0;   // count of matching GPUs so far
    for (const auto& d : sycl::device::get_devices()) {
        if (!d.is_gpu()) continue;
        if (name_filter.empty() ||
            d.get_info<sycl::info::device::name>().find(name_filter) != std::string::npos) {
            if (seen == ordinal) { picked = d; found = true; break; }
            ++seen;
        }
    }
    if (!found && ordinal == 0) {
        // Fall back to any GPU (only for the first device — ordinal>0 must match).
        for (const auto& d : sycl::device::get_devices()) {
            if (d.is_gpu()) { picked = d; found = true; break; }
        }
    }
    if (!found) return "no SYCL GPU device available at ordinal " + std::to_string(ordinal);

    try {
        // In-order queue: each submission implicitly depends on the previous,
        // so the model code can drop most per-kernel .wait() calls. Keeps
        // launch latency on a single thread of execution and lets the SYCL
        // runtime overlap host-side dispatch with device-side execution.
        //
        // enable_profiling is OPT-IN (IE_QUEUE_PROFILING=1). On the in-order
        // Level-Zero queue it adds large per-submit HOST overhead (~1.76 ms/kernel
        // measured) → decode is submission-bound, not GPU-bound (crown GPU-busy
        // 11.5 ms but wall 276 ms = 3.6 tok/s vs the ~81 tok/s GPU ceiling). Only
        // the kernel profiler / ie-bench --kprofile need it, so default OFF.
        if (std::getenv("IE_QUEUE_PROFILING"))
            queue_ = sycl::queue(picked,
                                 sycl::property_list{sycl::property::queue::in_order{},
                                                     sycl::property::queue::enable_profiling{}});
        else
            queue_ = sycl::queue(picked, sycl::property_list{sycl::property::queue::in_order{}});
    } catch (sycl::exception& e) {
        return std::string("queue creation failed: ") + e.what();
    }
    return {};
}

void* DeviceAllocator::malloc(size_t nbytes) {
    if (!queue_) return nullptr;
    return sycl::malloc_device(nbytes, *queue_);
}

void DeviceAllocator::free(void* p) noexcept {
    if (!queue_ || !p) return;
    sycl::free(p, *queue_);
}

Tensor DeviceAllocator::alloc_tensor(DType dtype, std::span<const uint64_t> shape) {
    Tensor t{};
    if (shape.empty() || shape.size() > kMaxDims) return t;
    t.dtype = dtype;
    t.device = Device::kGpu;
    t.n_dims = static_cast<uint32_t>(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) t.shape[i] = shape[i];
    uint64_t row_elems = shape[0];
    uint64_t hi = 1;
    for (size_t i = 1; i < shape.size(); ++i) hi *= shape[i];
    t.nbytes = bytes_for(dtype, row_elems) * hi;
    if (t.nbytes == 0) return Tensor{};
    t.data = malloc(t.nbytes);
    if (!t.data) return Tensor{};
    return t;
}

void DeviceAllocator::free_tensor(Tensor& t) noexcept {
    if (t.device == Device::kGpu) {
        free(t.data);
    }
    t.data = nullptr;
    t.nbytes = 0;
    t.n_dims = 0;
}

// ---- DeviceFleet (P-A multi-GPU) --------------------------------------
std::string DeviceFleet::init(uint32_t n_request, std::string_view name_filter) {
    if (n_request == 0) return "DeviceFleet: n_request == 0";
    if (const char* f = std::getenv("IE_GPU_FILTER"); f && *f) name_filter = f;  // see DeviceAllocator::init
    // Count matching GPUs so we bind min(n_request, available).
    uint32_t available = 0;
    for (const auto& d : sycl::device::get_devices()) {
        if (!d.is_gpu()) continue;
        if (name_filter.empty() ||
            d.get_info<sycl::info::device::name>().find(name_filter) != std::string::npos)
            ++available;
    }
    if (available == 0) return "DeviceFleet: no GPU matches filter '" +
                                std::string(name_filter) + "'";
    const uint32_t n = std::min(n_request, available);
    devs_.clear();
    devs_.resize(n);   // DeviceAllocator default-constructs (no queue yet)
    for (uint32_t i = 0; i < n; ++i)
        if (auto e = devs_[i].init(name_filter, i); !e.empty())
            return "DeviceFleet dev " + std::to_string(i) + ": " + e;
    return {};
}

void DeviceFleet::copy_across(uint32_t si, void* dst, uint32_t di,
                              const void* src, size_t nbytes) {
    if (si == di) {                          // same device — direct copy
        devs_[di].queue().memcpy(dst, src, nbytes).wait();
        return;
    }
    // Host bounce: src(dev si) → host → dst(dev di). Robust on any board; the
    // layer-split boundary copy is small so this is cheap. P2P is a later opt.
    std::vector<unsigned char> host(nbytes);
    devs_[si].queue().memcpy(host.data(), src, nbytes).wait();
    devs_[di].queue().memcpy(dst, host.data(), nbytes).wait();
}

void DeviceFleet::ensure_ar_pin(uint64_t n_elem) {
    if (n_elem <= ar_cap_ && !ar_pin_.empty()) return;
    free_ar_pin();
    const uint32_t nd = static_cast<uint32_t>(devs_.size());
    ar_pin_.assign(kArGen, std::vector<sycl::half*>(nd, nullptr));
    for (uint32_t gtmp = 0; gtmp < kArGen; ++gtmp)
        for (uint32_t d = 0; d < nd; ++d)
            ar_pin_[gtmp][d] = sycl::malloc_host<sycl::half>(n_elem, devs_[d].queue());
    ar_acc_.assign(n_elem, 0.0f);
    ar_cap_ = n_elem;
    ar_gen_ = 0;
}

void DeviceFleet::free_ar_pin() {
    for (uint32_t gtmp = 0; gtmp < ar_pin_.size(); ++gtmp)
        for (uint32_t d = 0; d < ar_pin_[gtmp].size() && d < devs_.size(); ++d)
            if (ar_pin_[gtmp][d]) sycl::free(ar_pin_[gtmp][d], devs_[d].queue());
    ar_pin_.clear();
    ar_cap_ = 0;
}

DeviceFleet::~DeviceFleet() {
    // Drain any in-flight async scatters before freeing the pinned staging.
    for (auto& d : devs_) d.queue().wait();
    free_ar_pin();
}

void DeviceFleet::all_reduce_sum_fp16(const std::vector<sycl::half*>& bufs,
                                      uint64_t n_elem) {
    const uint32_t nd = static_cast<uint32_t>(bufs.size());
    if (nd <= 1 || n_elem == 0) return;   // nothing to reduce
    const size_t bytes = n_elem * sizeof(sycl::half);
    ensure_ar_pin(n_elem);
    sycl::half** stage = ar_pin_[ar_gen_].data();   // this generation's per-dev buffers
    ar_gen_ = (ar_gen_ + 1) % kArGen;

    // Gather ALL devices concurrently (overlap the per-card D→H DMAs) into pinned
    // staging, then wait — the CPU fp32 sum must see the gathered data.
    std::vector<sycl::event> ev(nd);
    for (uint32_t d = 0; d < nd; ++d)
        ev[d] = devs_[d].queue().memcpy(stage[d], bufs[d], bytes);
    for (uint32_t d = 0; d < nd; ++d) ev[d].wait();

    // fp32 accumulate (device order 0..nd → BIT-IDENTICAL to the prior version),
    // then write the sum back into each device's own pinned buffer for the scatter.
    std::fill_n(ar_acc_.begin(), n_elem, 0.0f);
    for (uint32_t d = 0; d < nd; ++d) {
        const sycl::half* s = stage[d];
        for (uint64_t i = 0; i < n_elem; ++i) ar_acc_[i] += float(s[i]);
    }
    for (uint32_t d = 0; d < nd; ++d) {
        sycl::half* s = stage[d];
        for (uint64_t i = 0; i < n_elem; ++i) s[i] = sycl::half(ar_acc_[i]);
    }

    // Scatter the sum back to ALL devices — ASYNC: NO host .wait(). Each device's
    // queue is in-order, so its next-submitted consumer (residual_add on the same
    // queue) runs after this scatter; the generation ring keeps the host from
    // overwriting a buffer whose DMA is still in flight. Removes one host-device
    // sync barrier per all-reduce (the per-layer decode hot path). Math unchanged.
    for (uint32_t d = 0; d < nd; ++d)
        devs_[d].queue().memcpy(bufs[d], stage[d], bytes);
}

}  // namespace ie
