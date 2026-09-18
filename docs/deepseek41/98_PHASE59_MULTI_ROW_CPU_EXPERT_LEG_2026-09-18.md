# V4.1 — Phase 59: the CPU expert leg for multi-row decode steps

## The finding that asked for it (docs/97's first A/B)

Prompt-lookup speculation on 42 recorded agent requests, shipping configuration: acceptance was excellent (6.62 of
6.89 drafts per pass) and the result was x1.04, because an 8-row verify cost 777 ms -- 98.5 ms a row, 97 % of a
one-row step. Per step, summed over the 40 layers (`[ds41 lookup]` attribution, `~/ds41_work/p58/serverC.log`):

| | one-row step | 8-row verify |
|---|---|---|
| wall | 107 ms | 767 ms |
| attention | 4.9 ms | 5.7 ms (flat: already batched) |
| router | 15 ms | 42 ms |
| MoE | 83 ms | 712 ms |
| pinned experts over PCIe | 832 MiB | 17,120 MiB (x20.6) |
| pinned misses computed on the CPU | 89 | 0 |

The CPU leg (Phase 20, docs/49) computes ~70 % of a one-row step's pinned misses in place (q* = 0.30 PCIe share); it
served `T == 1` only, so a verify sent the union of eight rows' pinned experts (~24 a layer) over the link.

## What changed (`src/model/deepseek41_experts.cpp`)

- The leg serves 2..8-row decode steps. A work item is (expert slot, packed row, that row's token); the host copy of
  the activations is [T, H]; the pinned row buffer holds 8 x top-k rows.
- A multi-row step sends the experts with the MOST rows over PCIe (a transfer costs the same per expert, the CPU per
  row) and puts the rest on the CPU, with its own share `IE_DS41_QSTAR_MULTI` (default 0.20). One row keeps Phase 20's
  order and share exactly.
- A 2..8-row decode step keeps the core partition (the leg's team owns the E-cores); only prefill spreads the mmap
  readers onto every core.

## The share, swept (`ie-ds41-lookup-test`, CPU miss split ON, `~/ds41_work/p59/lookup_q*.log`)

| PCIe share | 8-row verify | list rewrite | paragraph repeat | code rename |
|---|---|---|---|---|
| (leg off) | ~780 ms | x1.04-ish (docs/97) | | |
| 0.65 | 627 ms | x1.27 | x1.18 | x1.09 |
| 0.50 | 570 ms | x1.39 | x1.23 | x1.13 |
| 0.35 | 502 ms | x1.56 | x1.33 | x1.20 |
| **0.20** | **488 ms** | **x1.61** | **x1.39** | **x1.23** |
| 0.10 | 512 ms | x1.55 | x1.35 | x1.17 |

At 0.20 a verify moves 4.9 GB instead of 17.1 and runs 660 (row, expert) pairs on the CPU; below it the CPU binds.

## Served: x1.47 on the recorded agent traffic

The 42-request replay (docs/97): plain 106.0 / 104.8 ms/token (A / A2), lookup with this leg **71.6 ms/token**
(13.96 tok/s, x1.47); the median 8-row verify 445 ms. One-row steps are unchanged, so plain chat decode is too.

## Correctness

- `ie-ds41-multi-test` 38/38 with the split off: the GPU multi-row path is unchanged (the leg is off with the split).
- With the split ON a verify row and a one-row step put different experts on the CPU (fp32 activations) and the GPU
  (int8 activations), so a greedy near-tie can resolve differently -- the same property the one-row leg has shipped
  with since Phase 20. `ie-ds41-lookup-test` enforces bit-identity with the split off and reports divergence with it on.
