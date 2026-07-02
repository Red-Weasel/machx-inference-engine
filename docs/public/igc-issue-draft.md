# Run-to-run non-deterministic FMA results in chunked-scan kernel on BMG-G31 (Xe2)

> Draft for https://github.com/intel/intel-graphics-compiler/issues
> (filed from the B70 inference-engine project; internal refs:
> `docs/known_bugs.md` §1, `docs/bisect_step25_26_summary.md`)

## Environment

- GPU: Intel Arc Pro B70, device id 0xe223 (BMG-G31, Xe2-HPG), 32 GB
- Driver: compute-runtime (NEO) **26.14.37833.4**, Level Zero 1.15.37833+4
- IGC: **2.32.7**
- GuC firmware: **70.60.0**
- Compiler: Intel oneAPI DPC++ (icpx) **2026.0.0** (IntelLLVM 2026.0.0), `-fsycl -O2`
  (JIT path, `spir64` target — same as the production engine; if IGC's codegen differs
  under `-fsycl-targets=` AOT, that difference is itself relevant signal)
- OS: Ubuntu, Linux kernel 6.17
- Kernel style: plain SYCL 2020 only — no ESIMD, no 2D block loads, no
  subgroup block reads, no DPAS; scalar fp32 FMA chains.

## Summary

A gated-DeltaNet chunked-scan recurrence kernel (one 128-work-item
work-group per head, 32 heads; each lane holds a 128-element fp32 state
column in private registers; sequential t-loop over a T=512..1024 chunk
with four unrolled 128-deep fp32 multiply-accumulate chains per step,
SLM broadcasts + `group_barrier` between fill and use) produces
**run-to-run different fp32 results on bit-identical inputs** when
launched many times in succession inside a busy inference engine on
BMG-G31. The first divergence is a small numeric difference in the
state buffer that the recurrence then amplifies (perplexity 6.55 → ~74
at chunk T=1024).

The divergence is stochastic and timing-coupled: across 8 parallel
identical chains, the first divergent chain appears anywhere between
launch ~7 and ~500, and the layer at which it fires varies run to run
(L=3, 9, 15, 20, 22 all observed).

## Reproduction

**Honest status:** a standalone single-file repro
(`dn_fma_nondet_repro.cpp`, attached — exact structural clone of the
kernel, synthetic xorshift-seeded inputs at production shape, 200
iterations × 4 chained launches on identical inputs, FNV-1a output
hashing) does **not** fire in isolation: 200/200 identical hashes at
T=512 (three separate processes, identical hash across processes) and
at T=1024 — 800 kernel launches at T=512, plus 800 at T=1024,
1,600 total clean launches. The kernel by itself, in a
quiet in-order queue, is deterministic.

The 28-step bisect closed every software channel by which surrounding
traffic could corrupt this kernel's inputs or outputs: inputs verified
bytewise-identical at every submission via host-side hashing (with a full
bytewise buffer trace captured at a known-divergent target); state
coherency bypassed entirely via `sycl::malloc_host` (PCIe-coherent, no
GPU caching) with the bug still firing; EU stores to state replaced with a
scratch-buffer-plus-memcpy path with the bug still firing. Determinism in
isolation therefore does not point back at the engine's software — it is
consistent with the remaining hypothesis: a microarchitectural condition
(residue of heterogeneous launch traffic) that an isolated kernel does not
recreate.

The bug reproduces reliably only in-engine, where each forward pass
interleaves ~30 other kernel types and 8 logically independent inference
chains, all serialized through one in-order SYCL queue — their kernels
interleave temporally, not concurrently:

```bash
./build/tools/ie-bug-monitor --max-iters 100
# 8 chains, identical token stream, per-layer FNV hashes of all
# persistent state after every step.
# → "✗ DIVERGENCE DETECTED" typically within ~100 iterations;
#   first divergent buffer is always a DeltaNet state layer.
```

This requires a 20 GB Qwen3.6-35B GGUF model. We can supply traces,
the full tool source, or run any requested instrumentation on our
hardware. The isolated-kernel determinism is itself a data point: the
failure needs surrounding kernel-launch traffic (microarchitectural
residue from prior launches), not just the FMA chains.

## Observed vs expected

- **Expected:** identical kernel + bit-identical inputs + bit-identical
  initial state ⇒ bit-identical outputs, every launch. IEEE-754 fp32
  FMA is deterministic; the kernel has a fixed reduction order (private
  per-lane chains, no atomics, no work-group reductions). All inner
  accumulations are `#pragma unroll` with fixed sequential order — no
  atomics, no group reductions, no sub-group shuffles. An
  fp-non-associativity explanation would require the accumulation order
  to change between runs of the same compiled binary, which it cannot.
- **Observed:** occasionally one launch returns a state buffer that
  differs from the reference (initially by a small numeric delta in one
  layer's state). All inputs at submission were verified bytewise
  identical (host-side hash before launch, and a captured-target
  bytewise trace of every input buffer at a known-divergent launch).

## What was ruled out (28-step bisect)

Each step changed exactly one thing; "bug persists" = divergence still
fired with all inputs verified identical:

- **Uninitialized memory** — zero-fill of the suspect workspace (Step 8): persists.
- **native::exp vs precise exp** on the decay gate (Step 15): persists.
- **IGC SIMD width / register-allocation variance** — forced widths (Step 16): persists.
- **SLM bank conflicts** on the broadcast buffers (Step 18): persists.
- **Per-lane FMA pipelining rework** (Step 19): persists.
- **Kernel-internal compute** is deterministic from state=0 (Step 20): clean.
- **Memory ordering** — `atomic_ref<seq_cst, system>` on the state RMW
  plus `atomic_fence` (Steps 21–22): persists.
- **Cache flush via DMA** — L0 copy-engine round-trip of state[] after
  every launch (Step 23): persists.
- **Long internal t-loop drift** — splitting T=1024 into 16× T=64
  sub-launches made the cascade *worse* (7 → 28 affected layers,
  Step 24): more launches ⇒ more divergence, pointing at
  launch-coupled timing, not accumulation length.
- **GPU cache coherency of state[] at any level** — state moved to
  `sycl::malloc_host` (PCIe-coherent, GPU caches bypassed entirely,
  Step 25): persists (divergence iter shifted, confirming timing
  sensitivity).
- **Sub-ULP drift clampable by quantization** — fp16 round-trip on
  state writeback (Step 26) made first divergence *earlier* (iter 192
  → 40): the initial delta is real, not representational.
- **Stale reads of kernel inputs** — bytewise trace of every input at a
  captured divergent (layer 22, iter 4) target (Step 27): inputs identical.
- **EU stores to state[]** — alternative kernel writing to scratch +
  single `memcpy` back (so EUs never store to the persistent buffer):
  persists.

Remaining hypothesis: an FMA-pipeline (or IGC codegen whose execution
is timing-coupled) non-determinism on Xe2 BMG-G31, sensitive to
microarchitectural state left by preceding launches.

Possibly related third-party reports of BMG-G31 SYCL numerical/stability
issues: https://github.com/ginkgo-project/ginkgo/issues/2018
(preconditioner numerical failures on BMG-G31),
https://github.com/hpsim/OGL/issues/170 (DEVICE_LOST in IC/ICT solvers
on Arc Pro B70).

## Production impact

We must externally chunk prefill at T ≤ 256 (the bug does not fire
below ~200 sequential steps per launch sequence), forfeiting
long-chunk prefill throughput; any long single-call prefill (T=1024)
collapses model quality (PPL 6.55 → ~74) when the bug fires.

## Ask

1. Is this a known BMG-G31 / Xe2-HPG erratum (FMA pipeline or
   scoreboard/launch-residue class), or addressable in IGC/NEO?
2. Guidance on what instrumentation (unitrace config, IGC dump flags,
   GuC log, specific NEO debug keys) would let us capture the divergent
   launch in a form useful to you — we can run anything on our box.
3. Can Intel confirm whether NEO 26.09.37435.12 + GuC 70.60.0 (the
   llm-scaler-pinned stack) is known-good for sustained FMA workloads on
   BMG-G31? We can run our in-engine repro on that stack on request.
