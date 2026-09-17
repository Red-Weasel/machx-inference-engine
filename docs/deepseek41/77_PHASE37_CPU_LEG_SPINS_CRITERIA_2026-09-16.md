# V4.1 — Phase 37: the CPU leg keeps its cores warm (criteria before the build)

## The term

At 32k (docs/76, q* 0.30) the CPU leg spans 34.1 ms/token for 26.9 ms of compute: **7 ms/token of overhead**, and
the compute itself runs at 0.600 ms/expert against 0.31-0.35 for the same kernel on the same E-cores in the bench.
Part of the compute gap is DRAM contention with the PCIe leg (measured directional in docs/76); part is the way
the leg waits. Between layers the worker thread sleeps on a condition variable and its OpenMP team has
blocktime 0, so all 12 E-cores drop into deep C-states and cool (~1.5 ms gaps, 40 times a token); inside a layer
each expert opens three parallel regions, each ending with the team going back to sleep; and the SwiGLU and the
fp16 conversion of every expert's row run on one thread.

## The change (the leg only; the GPU side and the split rule untouched)

1. The worker waits for a request by SPINNING (bounded -- a few ms -- then the condition variable as now), and
   the team's blocktime is raised so its threads spin through a decode step's gaps instead of sleeping after every
   region. The cores are the leg's own (8-19), so the spin costs power, not throughput.
2. The per-expert SwiGLU and fp16 row conversion join the parallel work.
3. Then q* is re-swept at {0.15, 0.30}: a faster CPU leg moves the balance toward the CPU, which takes bytes OFF
   the PCIe link and the memory-controller contention with it.

The math per row is unchanged (`row_dot` is the same code on the same operands): bit-identical by construction.

## Criteria

1. **Bit-identity:** the profiled decode test's criterion lines identical to Phase 35's (`p24/w35_decode.log`).
2. **The leg:** at 32k with `IE_DS41_STAGES=1`, span - compute per token below 3 ms (from 7), and ms/expert
   compute below 0.50 at q* 0.30 (from 0.600).
3. **The token:** the 32k long test in default mode against Phase 35's 67.1 ms/token; adopt if >= 2 ms/token
   faster at the best q* of the re-sweep, with the q* value recorded; else the change is kept only if it is not
   slower (criterion 2 met and criterion 3 neutral), and the doc says so.
4. The bench with `IE_DS41_BENCH_GAP_US=1500` (the between-layer gap emulated) sleeping vs spinning, recorded as
   the isolated cost of the cooling.

## Criterion 4 first, and it removes half the plan

`ie-ds41-cpu-expert-bench`, cores 8-19, 12 threads, `IE_DS41_BENCH_GAP_US=1500`: **blocktime 0: 0.48 ms/expert;
blocktime 200 (the team spinning through the gap): 0.61.** Without the gap the same team does 0.35 / 0.32. So the
cooling after a gap costs ~0.13 ms per expert (and at 32k the leg gets ~1.1 experts per layer, so nearly every
expert pays it), but a spinning OpenMP team makes it WORSE, not better -- 12 E-cores spinning are slower at the
work that follows than 12 that slept. The blocktime stays 0. What is built is the other half: the worker's own
wait and the host's join become bounded spins on one core each (the condition variable stays as the fallback).

## Result: FALSIFIED and REVERTED -- the spins make the leg 2x slower

Built: the worker spins (bounded, 20 ms) for the next request before its cv wait, and the host spins the same way
at the join; the team's blocktime untouched at 0. 32k, `IE_DS41_STAGES=1`, same build otherwise:

| | q* 0.30, Phase 36 | q* 0.30, spins | q* 0.15, Phase 36 | q* 0.15, spins |
|---|---|---|---|---|
| CPU leg compute, ms/expert | 0.600 | **1.232** | 0.506 | **0.955** |
| CPU leg span / compute, ms/token | 34.1 / 26.9 | 62.6 / 55.1 | 36.8 / 33.6 | 67.0 / 63.5 |
| token | 68.8 ms | **97.6** | 71.4 | **101.8** |

The leg's COMPUTE doubled, not its wait: the same kernel on the same cores runs at half speed while one E-core
(the worker between requests) and one P-core (the host at the join) spin instead of sleeping. Together with the
bench's spinning-team result (0.61 vs 0.48 after a gap) the direction is consistent: on this part a spinning core
makes the neighbouring working cores slower. **Assumption, not verified:** the package power budget -- a core
spinning at full clock takes the headroom the E-cores' turbo needs. What would verify it: the same run with the
E-cores' frequency read from sysfs during the leg. Not pursued; the code is reverted to the Phase 35 state and
the per-layer cooling cost (~0.13 ms per expert after a gap) stands as measured but unrecovered.

What this closes: any "keep the leg warm by spinning" idea, team or single thread. What it leaves open: the
cooling cost itself (an OS/C-state matter, the founder's call), and folding the three parallel regions per expert
into one (fewer wake-ups per expert; ~10 % of the leg at most by the blocktime bench).
