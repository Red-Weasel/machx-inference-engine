# V4.1 port — Phase 13 gate criteria: the decode miss path (written BEFORE the build)

**The founder's goal (2026-09-13 17:00): 400+ tok/s prefill, 20+ tok/s decode.**
**Starting line (ab8b578):** prefill 280 tok/s at pp2048, decode 5.0 tok/s at a 2,048-token
context, both bound by expert bytes over one PCIe link at ~22 GB/s: at decode 1,378 MiB per
token = 65 ms of a 195 ms step; at prefill 5,701 pinned experts per 2,048-token chunk = 100 GiB
= 4.6 s of 7.3 s. (docs/30, docs/32, `decode_ship.log`, `resident_p12.log`.)

## The arithmetic, stated before anything is built

Decode per token at the 2,048-token context, measured: 65 ms pinned fetch (92 pinned misses x
17.9 MiB), 26-46 ms disk (15 mmap misses), 12 ms expert GEMMs, 40 ms dense body (floor 26 ms
fp16, 13 ms as FP8), ~12 ms head + setup + host. The levers and what each is worth, from these
numbers (projections, not measurements — each phase measures its own):

| lever | mechanism | decode ms saved | prefill |
|---|---|---|---|
| **A. CPU miss compute (this phase)** | the missed experts are computed on the CPU from the pinned host arena instead of crossing PCIe (FreeToken's q* split, ported on GLM-5.3: +44%, 0.61 ms/expert; this rig's host bandwidth 60 GB/s vs the link's 22) | 65 → ~20-25 | none (prefill's M ~ 140 rows/expert is compute, not bandwidth) |
| B. residency policy | decode hits 61.6% with 45 static + 8 LRU slots per layer; the LRU slots hit 43 of 240 selections with only 8 slots — an LRU over all 53 may beat the prefill-profile static set at decode | up to ~15 | possibly negative |
| C. more pinned RAM | 160 GiB pinned (228/layer); ~40 GiB free during runs → ~285/layer, fewer disk misses | ~15 | ~ |
| D. expert parallel (both links, GEMMs split) | Phase 14 | ~15 (after A) | **4.6 → 2.3 s: ~450-500 tok/s** |
| E. FP8-resident dense | Phase 15 | ~27 | ~ |
| F. batched verify | Phase 16 (needs docs/33's continuation) | dense amortised, MoE bytes not | ~ |

Summed honestly, A through F land single-stream decode near **12-15 tok/s** at a 2,048-token
context, not 20: the expert traffic of a 552B-parameter model with 269 GiB of experts and 64 GB
of VRAM is the wall, and no lever above changes the bytes a token needs — only where they come
from and how fast. 20+ needs one of: a decode hit rate near 90% (lever B's measurement says
whether the working set of a long generation is small enough), a longer context (the LRU
locality rises with it), or hardware. The founder gets this arithmetic with the results, and
each phase's measurement replaces one projection with a number. Prefill 400+ is lever D's.

## Scope, in order; each step measured before the next

1. **Measure lever B before building anything.** The decode test's 36-step benches with the
   tier configured as (a) today (45 static + 8 stream), (b) 0 static + 53 stream (pure LRU),
   (c) 20 static + 33 stream; at the 2,048-token context and after a 4,096-token context
   (docs/32's second CLI session shows the runtime holds 4,096). Report the decode hit rate,
   bytes per token and ms/token for each; the winner is the placement the rest of the phase runs
   on. No code beyond the knobs `ResidentOptions` already has.
2. **The CPU MXFP4 expert GEMV, benchmarked alone.** A host kernel over the pinned arena's slot
   layout (`ds41_slot_layout`: the three nibble planes in the engine's interleaved order, the
   E8M0 scale planes) for one row: gate, up, silu-clamp, down, in fp32; AVX2 (this CPU has no
   AVX-512); a team pinned to a core set. Gate: bit-for-bit equal to the reference dequant
   (`mxfp4_e8m0_half`, `mxfp4_nibble_int` in src/ops/deepseek4_experts.cpp) applied on the host
   for one expert, and within 1e-4 relative of the GPU's `ds4_gemm_mxfp4` M = 1 result on the
   same expert and activation (`tools/cpu_moe_gemv_bench.cpp`'s shape); experts per ms measured
   on 8 P-cores, on 12 E-cores, on all 20, cold (rotating over more experts than the L3 holds).
   That number decides q*.
3. **The split in the tier.** `Ds41ExpertTier::moe` at T = 1: of the m pinned misses in a layer,
   q* x m go over PCIe as today and the rest to a persistent CPU worker (GLM's two lessons:
   a persistent OpenMP team, and the worker's cores partitioned away from the reader threads
   and the host's own thread), whose partial sums land in the layer's output through the same
   scatter. q* from the two measured bandwidths, overridable (`IE_DS41_QSTAR`, `IE_DS41_CPU_MISS`).
   The mmap misses stay on the reader path (a disk read is a disk read either way). Gate: the
   decode test's Phase 9 criteria under their bars with the CPU path on (the arithmetic is the
   same numbers in a different order: forced step 0 judged as in Phase 11's isolation, per layer
   against the golden AND against the all-GPU path); the greedy tokens unchanged; ms/token and
   the tier's bytes over PCIe per token, before and after, and the q* sweep (0.3 / q* / 0.6 /
   1.0) at the 2,048-token context.
4. **Lever C** if step 1's placement leaves disk misses that matter: `host_pin_cap` raised to
   what the RAM rails allow (the memguard's 60 GiB floor and the pinned-RAM livelock record in
   memory decide the cap), the mmap misses per token before and after.

## Pass criteria

1. Step 1's table (three placements x two contexts), with the chosen placement stated.
2. The CPU kernel's exactness against the host reference dequant and its agreement with the
   GPU result; its experts per ms on the three core sets, cold.
3. Correctness unchanged under the split: every Phase 9 criterion under its bar, tokens and
   hashes unchanged, the per-layer isolation against the all-GPU path on the ~1e-2
   amplification scale at worst; the forward, resident, replay, generate, tokenizer, prompt and
   tier tests PASS; unload unchanged.
4. **Measured decode**, steps 4-35 after the golden prompt and after the 2,048-token context:
   ms/token, tok/s, PCIe bytes per token, CPU experts per token, the q* sweep. The number is
   what it is; the projection above (65 → ~20-25 ms on the fetch) is judged against it and the
   arithmetic table is corrected in the results doc.
5. The founder-facing statement: the corrected table of levers with this phase's measurement in
   it, and the honest distance to 20 tok/s.

## Explicitly NOT in this phase

Expert parallelism (Phase 14, the prefill lever), FP8 dense weights, the batched verify and its
continuation prerequisite (docs/33), tool calls.
