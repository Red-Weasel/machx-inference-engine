# V4.1 — Phase 36: re-balancing the miss split (q*) after Phases 34/35 (criteria before the measurement)

## Where the MoE stage's time is

From the 32k stage split (docs/75): the MoE stage is 39-42 ms of a 67-69 ms token and holds ~15 ms of kernels.
Its two miss legs run side by side and the layer joins on the slower: the GPU leg (static-tier GEMMs + the pinned
experts over PCIe, "groups" 26.5 ms) and the CPU leg (44.7 experts/token at 0.633 ms each: 28.3 ms of compute in a
35.2 ms span). The CPU leg is the longer one, and the split that sends 30 % of the pinned misses over PCIe
(`q*` = 0.30 "for a whole tier") was tuned before tonight's attention work changed the rest of the token.

Why the CPU leg is slow in situ, measured: the kernel alone on the same E-cores (`ie-ds41-cpu-expert-bench`,
pinned 8-19, 12 threads) does 0.31-0.35 ms/expert = 54-60 GB/s -- blocktime 0 costs ~10 %, the core set nothing.
In the run it does 0.633. The remaining 1.8x is DRAM contention: the PCIe leg streams ~375 MiB/token of pinned
experts through the same memory controller while the CPU leg reads ~800 MiB. Both legs together are near the
practical DDR5 ceiling, so the floor for 1.2 GB/token of miss bytes is ~20 ms; the legs take 27-35.

## The change

None to the code. `IE_DS41_QSTAR` sweeps the PCIe share of the pinned misses at 32k with the stage split on, so
both legs are visible; the best value becomes the default for a whole tier if it beats 0.30 by the bar.

## Criteria

1. Sweep q* in {0.15, 0.30, 0.45, 0.60} on the 32k long test, `IE_DS41_STAGES=1` (legs visible; +2 ms/token of
   waits on every arm alike), 512 decode steps, same ranking, same build.
2. The best arm re-run in default mode (waits out) against the Phase 35 default figure, 67.1 ms/token: adopt if
   >= 2 ms/token faster (the run-to-run spread seen tonight is ~1-2 ms), else the default stays 0.30 and the sweep
   is recorded.
3. Bit-identity is not at stake (the split moves which device computes an expert; the CPU path is fp32 and was
   gated at 3e-4 against the GPU's Q8 path in Phase 13), but the decode test runs on the adopted value.

## Results: 0.30 stays -- no change

| q* | ms/token | tok/s | CPU experts/token | ms/expert compute | pinned experts/token | groups / join / tail (ms) |
|---|---|---|---|---|---|---|
| 0.15 | 71.4 | 14.01 | 66.4 | 0.506 | 8.4 | 14.9 / 3.1 / 22.3 |
| **0.30** | **68.8** | **14.54** | 44.7 | 0.600 | 29.7 | 26.5 / 2.8 / 8.5 |
| 0.45 | 69.1 | 14.47 | 42.5 | 0.615 | 32.3 | 28.3 / 2.7 / 7.3 |
| 0.60 | 78.3 | 12.77 | 21.9 | 0.782 | 53.0 | 40.8 / 2.9 / 3.8 |

(32k, 512 decode steps, `IE_DS41_STAGES=1` on every arm; `p24/qs_*.log`.) Criterion 2 is not met: nothing beats
0.30, and 0.45 barely moves the split at all (the rule keeps a lone miss on the link and splits pairs, so the share
is quantised). Two things the sweep measured on the way:

- **The contention is real and directional.** The CPU leg's cost per expert rises with the PCIe share -- 0.506 ms
  with 8 pinned experts streaming beside it, 0.600 at 30, 0.615 at 32, 0.782 at 53 -- while the kernel alone on
  the same cores does 0.31-0.35. The two miss legs share the memory controller.
- **The PCIe leg saturates.** At q* 0.60 it carries 53 experts = 949 MiB in ~41 ms = ~24 GB/s of the 26.5 GB/s
  link. At 0.30 the legs are balanced (CPU span 34.1 vs groups 26.5 + tail 8.5).

What is left in the MoE stage is not the split: the CPU leg's span exceeds its compute by ~7 ms/token (34.1 vs
26.9) -- per-layer dispatch, wake-up, the serial SwiGLU and fp16 conversion inside `cpu_expert_mxfp4`, the join --
and that is Phase 37.
