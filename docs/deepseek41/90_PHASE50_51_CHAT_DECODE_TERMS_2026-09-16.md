# V4.1 — Phases 50-51: where a Dream-style decode token goes, and what was tried against it

Workload: Dream's system prompt + 90 tools + one user turn (17,747 tokens), the chat decode ranking (docs/89), 150-300
greedy tokens, `ie-ds41-run --raw` with `IE_DS41_STAGES=1` -- which now reports the MEAN over every decode step
(docs/88 era tools reported the last step only).

## The token (bash-script prompt, 250 tokens, after Phase 50)

| term | ms/token |
|---|---|
| attention | 17.3 |
| ffn_pre (router) | 3.1 |
| **MoE** | **62.7-69.8** = GPU groups 31.3 (static + pinned, incl. 534 MiB/token of pinned misses over PCIe) + **join 19.8-23.9** (the mmap tier: 6.7 experts, 120 MiB/token of O_DIRECT reads, 6 regions each) + CPU-leg tail 10.1-13.1 (59.9 experts at 0.61-0.66 ms) |
| outside the layers | 5.4-6.2 (prep 2.7, head 2.4) |
| **wall** | **89.7-97.1 (10.3-11.2 tok/s)** |

Experts per token: static 107, pinned 66 (37 already in a stream slot), mmap 6.7, CPU 60; hit rate 60 %.

## Phase 50 -- LANDED (`ad545a1`): the engram gather faults its rows in parallel

The gather parallelised over tokens, so a decode step faulted its 48 random engram rows (2 layers x 24, from two
91.5 GiB FP8 tables) one after another: prep **9.39 -> 2.72 ms**, a token 104.3 -> 98.3 ms, same experts per token,
engram test 5/5, decode 28/28.

## Phase 51 -- measured, NOT kept

| idea | result |
|---|---|
| page-cache warming of the mmap tier (docs/70's -46 ms lever) | not applicable any more: the mmap tier reads O_DIRECT (page cache bypassed) since Phase 15 |
| buffered (page-cache) reads for DECODE steps, so repeated experts are memory reads | A/B/A 90.8 / **91.4** / 97.1 ms/token: no gain (reads per reader 12.3 -> 9.7 ms, join unchanged). Reverted. |
| more stream slots, fewer static (`IE_DS41_STREAM_SLOTS`, kept as an A/B knob) | 8: 89.7 / 90.3 (A-B-A); 16: 91.9; 24: **87.0** (-3 %). Stream hits 36.6 -> 58.8, but each static slot given up moves an expert to the mmap tier (the pinned tier is RAM-capped at 266): join 19.8 -> 25.1. Default stays 8. |

## Reading

Decode on this box at conversational context is bound by expert BYTES in three places at once -- PCIe for pinned
misses, NVMe for the mmap tail, E-core DRAM for the CPU leg -- plus a fixed ~17 ms of attention. docs/70's verdict
(a hardware bound: VRAM and host RAM against a 475 GB checkpoint) holds; what software moved today was host overhead
(the thread leak, the engram faults) and placement (the chat ranking).

## 2026-09-17 00:35-00:58: more stream slots with the expert file re-aligned -- closed

The earlier stream-slot sweep ran with the file's window misaligned. Re-run with the file rebuilt at each arm's disk
boundary (stream 24: [301, 345); 16: [309, 353); 8: [317, 361)):

| | single prompt (bash, 250 tok) | held-out 6 prompts via `ie serve`, A-B-A |
|---|---|---|
| stream 8 | 86.6 ms/token (11.60 tok/s) | **12.22 / 12.52 tok/s** |
| stream 16 | 82.0 / 82.1 ms/token | -- |
| stream 24 | **72.3 / 71.2 ms/token (13.9-14.1 tok/s)** | **11.71 tok/s** |

The single-prompt gain did not generalise: the generated text itself differs between placements from the first
sentence (the CPU and GPU expert paths are not bit-identical), so one prompt measures a different generation per arm.
Over the held-out set stream 24 is 4-6 % SLOWER. Default stays 8; the file is back at [317, 361). Held-out decode on
the shipped configuration is now 12.2-12.5 tok/s (docs/89 measured 10.81 before the engram fix and the expert file).
