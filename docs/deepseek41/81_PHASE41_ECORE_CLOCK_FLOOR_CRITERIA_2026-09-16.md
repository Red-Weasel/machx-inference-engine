# V4.1 — Phase 41: the CPU leg's cooling is a CLOCK ramp, not a C-state (criteria before the test)

## The correction

docs/77 called the CPU leg's per-gap cost "cooling into deep C-states". Read from sysfs on 2026-09-16: this
kernel (7.0.12-p2pwl) runs NO cpuidle driver (`current_driver` = none, no idle states on any core), so the cores
never enter a C-state deeper than HLT. What they DO do: `intel_pstate` in `powersave` with EPP `balance_performance`,
E-cores at 800 MHz idle and 4.8 GHz under load, sampled between layers at 800 / 1,139 / 2,477 MHz. The 0.35 -> 0.48
ms/expert after a 1.5 ms gap (docs/77's bench) is the E-cores climbing from 800 MHz, ~1.1 experts per layer paying
it. And "spinning made it worse" fits the same mechanism from the other side: a core spinning at full clock takes
the package power the working cores' turbo needs.

## The test (the founder's approval given 04:34; root commands run by the founder)

Arm A, the decisive one: a clock FLOOR on the leg's cores only -- `cpupower -c 8-19 frequency-set --min 4800MHz`
(idle stays HLT, so the cost is the idle power of 12 E-cores held at clock, not load). Arm B, gentler, if A
helps: EPP `performance` on 8-19 with the floor back at 800 MHz. Restore: `--min 800MHz` / EPP `balance_performance`.

## Criteria

1. The gap bench (`IE_DS41_BENCH_GAP_US=1500`, cores 8-19, 12 threads, blocktime 0): 0.48 ms/expert toward the
   warm 0.35.
2. The 32k long test, default mode, against Phase 39's 62.1 ms/token: keep the recipe if >= 2 ms/token faster;
   with `IE_DS41_STAGES=1` once, to see the CPU leg's ms/expert (0.600 at q* 0.30) move.
3. If it holds, q* re-swept at 0.15 / 0.30 (a faster CPU leg moves the balance off the PCIe link).
4. Nothing persists: the setting is a run-time recipe in this doc and the run script, restored after the test
   unless the founder chooses to keep it.

## Result: FALSIFIED -- the governor moves nothing in situ

Applied by the founder 05:21 (`cpupower` has no build for the p2pwl kernel; the sysfs write to
`scaling_governor` = `performance` on cpus 8-19), verified from sysfs: E-cores mostly at 4.8 GHz idle, though
still sampled dipping to 1.8-3.6 GHz at moments (HWP keeps some say even under `performance`).

| | powersave (before) | performance on 8-19 |
|---|---|---|
| gap bench, 1.5 ms sleep before each expert | 0.48 ms/expert | 0.45 |
| gap bench, no gap | 0.35 | 0.36 |
| 32k long test, default mode | 62.1 ms/token = 16.11 tok/s | **62.5 = 15.99** (noise) |
| 32k with the stage split: CPU leg ms/expert compute | 0.600 (span 34.1, compute 26.9) | **0.601 (34.1, 26.9)** |

Criterion 1 barely moves and criterion 2 fails outright; the leg's in-situ per-expert cost is unchanged to three
digits. So the clock ramp is NOT what separates the leg's in-situ 0.60 ms/expert from the kernel's warm 0.35: the
q* sweep's evidence (docs/76 -- the cost tracks the PCIe share) stands as the explanation, DRAM contention with the
pinned-expert DMA, with the rest in the twelve threads' wake-up after a sleep. Recommendation to the founder:
restore `powersave` (idle power for nothing). Both "keep the leg warm" routes -- spinning (docs/77) and the clock
floor (this) -- are now measured dead; the leg is bound by the memory system it shares with the PCIe leg.
