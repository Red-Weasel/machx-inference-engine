# Bisect Steps 23–26 — summary and dead-end conclusion

**Period**: 2026-05-03 (post-Step 22)
**Branch**: `main` (uncommitted scaffolding)
**Engine status at end**: default-OFF flags preserve baseline byte-for-byte. Built-in 511-tok PPL = 6.55 (parity with all prior baselines). All experiments below leave their flags in source as default-0 scaffolding.

## What we tested

| Step | Hypothesis | Implementation | Result | Verdict |
|---|---|---|---|---|
| 22 | SYCL atomics seq_cst+system are enough to fix state-RMW visibility | `atomic_ref<seq_cst, system>` + `atomic_fence` | bug persists, slightly worse than Step 21 | ❌ |
| 23 | DMA round-trip on `state[]` via L0 copy engine flushes per-Xe-core L1 | `q.memcpy(scratch, state); q.memcpy(state, scratch)` post-recurrence | bug persists (PPL ~71) | ❌ |
| 24 | Long internal t-loop is the carrier (split T=1024 → 16× T=64 sub-launches) | recursive deltanet_recurrence chunking | bug **worse**: cascade exploded 7→28 layers | rules out option A (long-loop drift); points to option B (between-launch state visibility) or timing sensitivity |
| 25 | GPU-side caching of `state[]` is the racy element | `sycl::malloc_host` for state[] (PCIe-coherent, GPU caches bypassed) | bug persists, smoking-gun = 1 at L14, divergence at iter 511 | ❌ — bug is **not** in state[] caching at any level |
| 26 | Stochastic divergence is small (sub-fp16-ULP); clamping after writeback re-unifies | per-call `float(half(state[idx]))` round-trip + NaN/Inf flush | bug **much worse**: divergence at iter 40 instead of iter 192 | ❌ — fp16 quantization noise amplifies via recurrence dynamics |

## What this rules out

The bug in `deltanet_recurrence` is **not** any of:
- ❌ Long internal t-loop arithmetic drift (Step 24)
- ❌ State[] read-after-write visibility across kernel launches (Steps 21, 22, 23)
- ❌ State[] cache coherency at any level (Step 25; USM host bypasses ALL GPU caches)
- ❌ Drift small enough to be clamped to fp16 ULP (Step 26 made it worse, not better)
- ❌ Native `exp` vs precise `exp` on `α` (Step 15)
- ❌ IGC SIMD width / register allocation variance (Step 16)
- ❌ SLM bank conflicts during k_slm/q_slm reads (Step 18)
- ❌ Per-lane FMA pipelining (Step 19)
- ❌ Kernel internal compute on bit-identical inputs+state=0 (Step 20)

## What's left as the actual cause

Working hypothesis after Step 25: a **HW-level non-deterministic compute event** in the recurrence kernel's FMA path on Xe2 BMG-G31 silicon, sensitive to kernel-launch microarchitectural state (residue from prior launches), that fires stochastically even when:

1. all 5 inputs `(q, k, v, g, β)` are bit-identical at submission;
2. `state[]` is bit-identical at submission and read coherently from host RAM (Step 25 confirms this);
3. compute path is structurally invariant (Step 20 confirms determinism for state=0).

Candidates within this class:
- DPAS-equivalent FMA pipeline transient (research brief: llama.cpp #21893 raised "DPAS synchronization or reordering" as a hypothesis on Battlemage; our path uses scalar FMA, not DPAS, but a similar HW-level pipeline issue could exist in the FMA hardware).
- Cross-Xe-core register or scoreboard transient state that affects FMA rounding non-deterministically when a lane reads state values whose magnitudes exercise specific exponent/mantissa boundaries.
- An IGC code-gen choice (e.g., implicit prefetcher, software-pipelined FMA chain) that produces deterministic code with non-deterministic timing-coupled execution.

Step 24's "more launches → much wider cascade" supports option B (timing/launch-coupled), as do unitrace's iter shift (192 → 1023) and Step 25's iter shift (192 → 511). The bug is timing-sensitive at the kernel-launch level.

## Decision: stop further bisect of this bug; pivot

After 26 single-step bisect experiments, the remaining suspect-set is below software's reach (HW-level FMA/DPAS pipeline non-determinism). Continued single-experiment bisect is unlikely to converge.

**Production status**: the bug only manifests in long-prefill-in-one-call paths (T > ~200). The production workflow already chunks at T=256 externally, where:
- Built-in 511-tok PPL = 6.55 (parity)
- Decode tok/s = ~50 (parity with llama.cpp)
- pp512 = ~150 (5.3× behind llama.cpp — separate perf issue, not this bug)

**The bug does not block shipping the engine for short-prefill / streaming-decode workloads.**

## Next moves (recommended, not yet executed)

1. **Vendor escalation**: file minimal repro against IGC at https://github.com/intel/intel-graphics-compiler/issues. Required: ~200-line standalone SYCL program that reproduces the smoking gun on BMG-G31. Build from `tools/ie_validate_chunking.cpp` + a stripped-down `deltanet_recurrence` clone. Estimated 4–8 h to prepare a minimal repro.

2. **Tighten the production guarantee**: enforce internal-chunking in production code paths, treating large-T forward calls as a programmer error. Add a runtime warning when T > 256 and the caller hasn't opted into long-prefill.

3. **Performance investigation** (orthogonal to the bug): pp512 at 150 tok/s vs llama.cpp's 800. Closing this is independent of the deltanet bug. Per the deep-research brief, knobs we haven't fully exploited:
   - `-O3 -DNDEBUG` with `GGML_SYCL_F16=ON`-equivalent f16 accumulation
   - Tile sizes re-tuned for Xe2's 256 KB unified L1+SLM (33% more than Alchemist)
   - Persistent zero-gap MoE kernel with atomic counters for dynamic expert load balancing

## Files modified in this session (all default-OFF)

- `src/ops/deltanet.cpp` — Step 23 (`IE_DN_STATE_DRAM_FLUSH`), Step 24 (`IE_DN_RECURRENCE_T_CHUNK`), Step 26 (`IE_DN_STATE_FP16_ROUNDTRIP` + `state_fp16_quantize_dequantize_inplace` kernel)
- `src/core/deltanet_state.cpp` — Step 25 (`IE_DN_STATE_USM_HOST`)
- `src/model/qwen36.cpp` — Step 26 call site
- `include/ie/ops.hpp` — Step 26 declaration
- `docs/bisect_step25_usm_host_state.md`, `docs/bisect_step25_26_summary.md` — this and prior log

## Validator state

`build/tools/ie-validate-chunking` and `build/tools/ie-perplexity` are built at default-OFF baseline. Running `./build/tools/ie-perplexity` (built-in corpus) gives PPL = 6.55, confirming engine math is intact at default flags. cmake flags reset: `CMAKE_CXX_FLAGS=""`.
