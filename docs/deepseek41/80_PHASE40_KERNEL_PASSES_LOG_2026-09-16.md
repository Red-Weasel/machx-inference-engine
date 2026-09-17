# V4.1 — Phase 40: further kernel passes, measured (a running log)

## MXFP4 expert GEMM (M = 1 grouped, `ds4_gemm_mxfp4`, 8.9 ms/token at 32k, the largest kernel)

**Where it stands (bench, `bench/ds41_mxfp4_gemm_bench.cpp`, six single-row jobs per launch streamed from a
64-expert bank):** gate/up 87.8 us per launch = 428 GB/s of expert bytes, down 86.2 us = 436 GB/s; in situ the
profile's 110 us average over ~9-job launches is ~520 GB/s. Its decode table costs one SLM load per TWO weights
(a byte -> two int8 in one uint16), and its ALU is a few ops per weight, so a table-free decode would triple the
ALU for little SLM relief -- not attempted, on the FP8 relay's arithmetic and this kernel's own ISA note (the
decode, not the dot, is what it issues).

**Falsified: a second sub-block of loads in flight.** The next sub-block's weight words and scale bytes hoisted
ahead of the current sub-block's decode and dot products (bit-identical: gate test bit-exact, output hashes equal).
gate/up 87.9 -> 87.8 us (nothing), down 86.2 -> 93.0 (8 % slower). The compiler already issues those loads early;
the extra live registers cost more than the bytes in flight buy. Reverted the same hour.

This kernel is also on the GPU miss leg, which the CPU leg outlasts at 32k (docs/76): a faster expert GEMM would
not move the token until the split shifts, so it is not the lever it looks like from the profile table.

## Where the serial-path kernels stand after Phases 38-39 (32k, profiled decode table, ms/token)

| kernel | ms/token | against its floor |
|---|---|---|
| q_b / o_b / o_a (FP8 relay) | 3.20 / 3.08 / 2.76 | ~500 GB/s of ~570: 85-90 % |
| head (`gemv_f16_rows`, 1.32 GB fp16) | 2.47 | at the bandwidth wall; only fewer bytes (a lossy head) would move it |
| XMX attention (part + combine + gather) | 1.32 + 0.45 + 0.28 | context-independent; a gather-in-kernel fusion is worth <= 0.4 |
| ds4_router_logits | 1.10 (27.6 us x 40) | 384 columns = 24 sub-groups: latency-bound; a split-K form is ~4x but reorders the sum |
| ds4_rms_norm / hc_mixes / collapse / quant / rope / masks | ~3.5 in total, ~700 launches | at the launch floor each; fusions of a few hundred us apiece |
| ds4_indexer_score | 0.64 (80 us x 8) | already the "headlane" shape (5.9x over its first form, two alternatives falsified in its notes) |
| ds4_indexer_topk (split) | < 0.5 | Phase 39 |

Nothing on this list is worth a phase on its own by tonight's bar (>= 1 ms/token, bit-identical or logits-gated),
and the two largest terms of the token are not kernels: the MoE stage's host-DRAM legs (docs/76) and the
expert-byte wait (docs/70). The kernel passes stop here; the log records what was measured so the next person
starts from the numbers, not the profile table's order.

## Expert parallel at 32k (one env var, measured 2026-09-16 05:42): a LOSS, and a RAM hazard

`IE_DS41_EP=1`, the 32k long test, same build and ranking as the 62.1 ms/token reference: **68.5 ms/token =
14.60 tok/s (-10 %)**. Under EP the split rule sends 60 % of the misses over the two links (pinned 69.7 experts =
889 MiB/token, CPU 16.4) and the MoE stage grows 38.5 -> 44.7 ms (tail 8.5 -> 15.6): two links do not beat one
link plus the CPU leg when all of it drains the same host DRAM. And the two-device context mirrors VRAM into host
RAM: MemAvailable fell from 243 GiB to **42 GiB** during the run (the memguard floor is 60; the pinned-RAM livelock
of 2026-08-27 is the precedent). Not to be run again in this configuration without the mirror fixed for the V4.1
tier (docs of 83554c2 fixed it for single-device paths only).

## DSpark speculation at 24k, re-measured with tonight's kernels (05:50): still a loss, by more

`IE_DS41_SPEC=1` on the 24k held-out prompt, 120 tokens: **6.18 tok/s against the plain 10.26**. 36 passes, mean
3.33 accepted tokens per pass (above docs/59's 2.57), but a pass costs draft 38.6 + verify **499.0** ms: the verify
step's rows (up to 7) each route to their own top-6, the union crosses the link with the CPU split off, and the
drafter's 0.61 GiB plus its own tier cut the static slots from 54/52 to 41 per layer (hit rate down, more misses).
161 ms per committed token. Cheaper attention did not matter: the pass is expert bytes. What would tip it is a CPU
leg that serves multi-row verify (docs/59's own conclusion) at the price of bit-exactness against plain greedy --
the founder's call, not a kernel.
