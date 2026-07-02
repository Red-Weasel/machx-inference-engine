# Step 25 — USM host allocation for `state[]` (diagnostic)

**Date**: 2026-05-03
**Hypothesis under test**: the bug is BMG/Xe2 per-Xe-core L1 staleness for `state[]` reads across recurrence kernel launches. Steps 21/22 (atomic_ref seq_cst+system + atomic_fence) and Step 23 (DMA round-trip via L0 copy engine) all failed to fix it because none of those primitives invalidate per-Xe-core L1 lines.

## What this experiment does

Allocates `dn.state_ptr()`'s backing buffer via `sycl::malloc_host(...)` instead of `sycl::malloc_device(...)`. All GPU-side accesses then go through PCIe to pinned host RAM, bypassing GPU L1/L2/LLC entirely.

This is a **diagnostic, not a fix**. USM host accesses are ~50–100× slower than USM device. If this kills the bug, we have hard evidence it's a GPU-side caching/coherency issue, and we then look for a less-invasive fix (e.g., USM shared, explicit cache-invalidate intrinsics, or a state-read kernel that goes through a specific cache path).

## Outcomes

- **Smoking gun = 0 in 3/3 runs**: bug is GPU-side caching of `state[]`. Next step: find a minimum-overhead fix.
- **Smoking gun ≥ 1**: bug is not (only) GPU-side caching. Pivot back to model-level workaround (state quantize/dequantize between iters).
- **Wall time spike**: expected; not a problem for diagnostic.

## Step 24 priors carried in

- Smoking gun fires once per validator run regardless of chunking, atomic flags, DMA flush, or unitrace.
- Step 24 (chunking T=64) made the cascade much worse (7 → 28 layers) which is consistent with launch-boundary state-visibility being the carrier.
- Bug onset is timing-sensitive (unitrace shifted iter 192 → 1023; Step 24 shifted DN_state[L=22] → DN_state[L=1]).

## Implementation notes

- Single source-file change: `src/core/deltanet_state.cpp` — new flag `IE_DN_STATE_USM_HOST` (default 0).
- USM host pointers work transparently with `q.memset(...)` and `q.memcpy(...)` and the existing kernel's `state[...]` indexing — no kernel changes needed.
- `sycl::free(state_, alloc_->queue())` correctly frees either USM host or USM device.
