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
#include <memory>
#include <string>
#include <vector>

namespace ie {

// True iff EVERY ordered pair of `devs` reports Level-Zero peer access, i.e.
// iff a direct device-to-device transfer is a legal thing to attempt at all.
//
// WHY THIS IS PROBED AND NOT ASSERTED IN A COMMENT.  Every cross-card path in
// this engine already goes through pinned host memory, and the reason recorded
// for that is a measurement taken once on one board ("can_access_peer == 0 both
// directions").  A fact that decides which transfer path runs must be checked by
// the process that depends on it, not carried in prose: the answer is a property
// of the PCIe topology (two cards under different root ports with no shared
// upstream bridge cannot do peer DMA at all, and the kernel refuses it), and
// that changes when the cards are moved, not when the source is edited.
//
// `tag` names the subsystem in the one line this prints.  The probe runs — and
// prints — ONCE per process: the verdict cannot change while it runs, and a line
// per transfer would bury it.  Fewer than two devices is not a peer question, so
// it returns true and says nothing.
bool gpu_p2p_available(const std::vector<sycl::device>& devs, const char* tag);

// IE_P2P interstage transport setup: collect the (first two) matching GPUs,
// verify + enable peer access BOTH ways, and build ONE shared context over
// them (PUSH-only bridge — the source device's queue writes into the peer's
// buffer; peer READS are broken on this fabric, 2026-08-27 probe). Returns
// false with untouched outputs when P2P is unavailable — callers fall back
// to per-device contexts and the host-bounce handoff.
bool gpu_p2p_shared_context(std::string_view name_filter,
                            std::vector<sycl::device>& gpus_out,
                            std::unique_ptr<sycl::context>& ctx_out);

// Diagnostic device-allocation registry (IE_ALLOC_AUDIT=1, off otherwise).
// Every DeviceAllocator::malloc is recorded with the tag current on this
// thread. dev_alloc_audit() prints the sorted map per device and flags any
// overlapping pair; dev_alloc_owner() names the allocation containing an
// address, so a corruption report can say WHOSE bytes it is looking at.
bool        dev_alloc_auditing();
void        dev_alloc_tag(const char* tag);
void        dev_alloc_record(const void* dev_key, const void* p, size_t n);
void        dev_alloc_audit(const char* what);
std::string dev_alloc_owner(const void* p);

// Apply process-wide runtime defaults (currently SYCL_CACHE_PERSISTENT=1) with
// overwrite=0. Idempotent; also run from a static initializer in allocator.cpp.
void apply_runtime_defaults();

class DeviceAllocator {
public:
    // Picks the `ordinal`-th GPU device whose name contains `name_filter`
    // (ordinal 0 = first match). If `name_filter` is empty, counts all GPUs.
    // Errors (including "fewer than ordinal+1 matching GPUs") are returned as a
    // non-empty string. ordinal defaults to 0 → the historical single-GPU pick.
    std::string init(std::string_view name_filter = "B70", uint32_t ordinal = 0);

    // Rebind this allocator's queue to (ctx, dev) — used by DeviceFleet to put
    // every fleet device in ONE shared context (legal cross-device host-USM
    // dereference + cross-queue event chaining). Must be called before any
    // allocation; same in-order/profiling properties as init().
    std::string init_with(const sycl::context& ctx, const sycl::device& dev);

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
    // `enable_p2p_transfers`: when true AND the pair probe passes, peer access is
    // enabled and copy_across / all_reduce_sum_fp16 go DIRECT GPU→GPU. Default
    // false — measured 2026-08-15: direct transfers win on the qwen35 TP path
    // (all-reduce/layer: decode 12.6 → 20.8) but REGRESS the split+spec path
    // (22 → 10.6, mechanism unpinned — spec's host-side traffic interacts badly
    // once peer access is enabled), so each model path opts in deliberately.
    // IE_NO_P2P=1 force-disables regardless.
    // `shared_ctx`: bind all queues into ONE sycl::context (needed by the
    // tensor-parallel all-reduce). On this stack a two-device context mirrors
    // every device allocation into host RAM 1:1, so pipeline-split paths, which
    // only copy_across, pass false and keep per-device contexts (2026-09-11:
    // 27B split identical speed/text, 29 GiB less host RAM at load).
    std::string init(uint32_t n_request, std::string_view name_filter = "B70",
                     bool enable_p2p_transfers = false, bool shared_ctx = true);

    uint32_t         size() const noexcept { return static_cast<uint32_t>(devs_.size()); }
    DeviceAllocator& dev(uint32_t i)       { return devs_[i]; }

    // Copy `nbytes` from `src` (device `si`) to `dst` (device `di`). Blocking.
    // Direct GPU→GPU when the fleet probed peer access at init (2026-08-15: live
    // on the 2×B70 once the host-bridge whitelist kernel patch is booted);
    // otherwise the robust host bounce. Same-device (si==di) is a plain memcpy.
    void copy_across(uint32_t si, void* dst, uint32_t di, const void* src, size_t nbytes);

    // True iff every device pair reported Level-Zero peer access at init.
    bool p2p() const noexcept { return p2p_; }

    // True iff all fleet queues share one sycl::context (multi-GPU fleets).
    bool shared_ctx() const noexcept { return shared_ctx_.has_value(); }

    // Tensor-parallel all-reduce (SUM): `bufs[d]` is device d's partial fp16
    // tensor of `n_elem` elements; on return EVERY device's buffer holds the
    // element-wise sum across all devices. Accumulates in fp32 for accuracy.
    // Blocking. Host bounce: the gathers (and the scatters) are submitted to all
    // devices' queues CONCURRENTLY then waited together, so the per-card PCIe
    // transfers overlap instead of serializing; staging is reused across calls.
    // (Level-Zero P2P is NOT supported on this 2×B70 board — can_access_peer=0
    // both directions, PCIe topology — so the host bounce is the path here.)
    // The 2-device P2P fast path event-chains pull→sum→scatter and blocks only
    // on the final scatter, preserving the original operation sequence.
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
    // P2P (Level-Zero peer access, probed + enabled at init when available).
    bool        p2p_ = false;
    sycl::half* ar2_scratch_ = nullptr;   // dev-0 scratch for the 2-dev P2P all-reduce
    uint64_t    ar2_cap_ = 0;             // elems
    // Shared context (2026-08-16): every fleet queue lives in ONE sycl::context,
    // so host USM from any fleet queue is legally dereferenceable by any fleet
    // device's kernels, and events chain ACROSS queues. Prerequisite for the
    // zero-copy all-reduce (the per-device-context variant was UB — see the
    // falsified note in all_reduce_sum_fp16).
    std::optional<sycl::context> shared_ctx_;
    // Zero-copy AR device scratch (2-dev): peer's pinned slot is DMA'd here so
    // the sum kernel reads DEVICE memory only (see the falsified-EU-read note
    // at the zero-copy branch in all_reduce_sum_fp16).
    sycl::half* zc_scr_[2] = {nullptr, nullptr};
    uint64_t    zc_scr_cap_ = 0;   // elems
};

}  // namespace ie
