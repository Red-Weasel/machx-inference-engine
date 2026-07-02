// include/ie/allocator.hpp — USM device allocator (stub for Phase 1).
//
// Phase 1 only needs a parse-and-list path, so this is the minimum viable
// shape: a sycl::queue + thin wrappers around malloc_device/free. Phase 2
// fleshes it out with bookkeeping (tracking outstanding allocs, peak usage,
// arena-style sub-allocation for KV/state caches).

#pragma once

#include "ie/tensor.hpp"

#include <sycl/sycl.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace ie {

class DeviceAllocator {
public:
    // Picks the `ordinal`-th GPU device whose name contains `name_filter`
    // (ordinal 0 = first match). If `name_filter` is empty, counts all GPUs.
    // Errors (including "fewer than ordinal+1 matching GPUs") are returned as a
    // non-empty string. ordinal defaults to 0 → the historical single-GPU pick.
    std::string init(std::string_view name_filter = "B70", uint32_t ordinal = 0);

    bool             ready() const noexcept { return queue_.has_value(); }
    sycl::queue&     queue() noexcept       { return *queue_; }
    sycl::device     device() const         { return queue_->get_device(); }

    // Raw allocations.
    void*    malloc(size_t nbytes);
    void     free(void* p) noexcept;

    // Allocate a Tensor with the given (dtype, shape) on the device. `shape`
    // is logical shape; nbytes is computed via bytes_for(). The returned
    // tensor's `data` is owned by *this* allocator and freed by free_tensor.
    Tensor   alloc_tensor(DType dtype, std::span<const uint64_t> shape);
    void     free_tensor(Tensor& t) noexcept;

private:
    std::optional<sycl::queue> queue_;
};

// P-A (multi-GPU layer-split): owns one DeviceAllocator per selected GPU. The
// single-GPU path is a fleet of size 1 (bit-identical to today). Cross-device
// transfers use a host bounce buffer (robust on any board; Level Zero P2P is a
// later optimization — the layer-split boundary copy is tiny, ~T·H·2 bytes).
// See docs/superpowers/specs/2026-06-12-multi-gpu-layer-split-design.md.
class DeviceFleet {
public:
    // Bind up to `n_request` GPUs matching `name_filter` (ordinals 0..n-1).
    // Returns error text if zero GPUs match; binds min(n_request, available).
    std::string init(uint32_t n_request, std::string_view name_filter = "B70");

    uint32_t         size() const noexcept { return static_cast<uint32_t>(devs_.size()); }
    DeviceAllocator& dev(uint32_t i)       { return devs_[i]; }

    // Copy `nbytes` from `src` (device `si`) to `dst` (device `di`) via a host
    // bounce. Blocking. Same-device (si==di) is a plain device memcpy.
    void copy_across(uint32_t si, void* dst, uint32_t di, const void* src, size_t nbytes);

    // Tensor-parallel all-reduce (SUM): `bufs[d]` is device d's partial fp16
    // tensor of `n_elem` elements; on return EVERY device's buffer holds the
    // element-wise sum across all devices. Accumulates in fp32 for accuracy.
    // Blocking. Host bounce: the gathers (and the scatters) are submitted to all
    // devices' queues CONCURRENTLY then waited together, so the per-card PCIe
    // transfers overlap instead of serializing; staging is reused across calls.
    // (Level-Zero P2P is NOT supported on this 2×B70 board — can_access_peer=0
    // both directions, PCIe topology — so the host bounce is the path here.)
    void all_reduce_sum_fp16(const std::vector<sycl::half*>& bufs, uint64_t n_elem);

    ~DeviceFleet();

private:
    std::vector<DeviceAllocator> devs_;
    // All-reduce scratch. Staging is PINNED host USM (per device, per generation):
    // a ring of generations lets the SCATTER (H->D) run ASYNC — the consumer kernel
    // on the same in-order queue follows it, so no host .wait() is needed on the
    // scatter (only the gather, which the CPU fp32 sum must wait on). The ring (>=2)
    // keeps the host from overwriting a buffer whose scatter DMA is still in flight
    // (consecutive all-reduces are ms apart; a ~6KB DMA is us). Drops one host-device
    // sync barrier per all-reduce on the per-layer decode hot path.
    static constexpr uint32_t kArGen = 4;
    std::vector<std::vector<sycl::half*>> ar_pin_;   // [gen][dev] pinned staging
    std::vector<float>                    ar_acc_;   // fp32 accumulate (CPU-only)
    uint64_t                              ar_cap_ = 0;  // per-buffer capacity (elems)
    uint32_t                              ar_gen_ = 0;  // round-robin generation
    void ensure_ar_pin(uint64_t n_elem);
    void free_ar_pin();
};

}  // namespace ie
