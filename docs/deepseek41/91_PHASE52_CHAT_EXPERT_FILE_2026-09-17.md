# V4.1 — Phase 52: the expert tail file rebuilt for the chat ranking, and on by default in `ie serve`

## Why

docs/90: on a Dream-style decode the mmap tier's disk fills (`join`) cost 20-24 ms of a ~90 ms token -- 6.7 experts
per token read O_DIRECT from the safetensors (6 regions each, then a permute). `ie_experts_tail.ieslot` (Phase 15 C2)
holds a rank window in slot layout (one read, no permute), but it had been written for the held-out ranking's window
[297, 341) and the server never used it.

## Done (founder-approved delete + rebuild, 2026-09-16 23:47)

The disk tier at the server's ctx 75,000 starts at rank 319 on card 0 (53 static + 266 pinned) and 317 on card 1
(51 + 266). Old file deleted (33.1 GB); new file `ie-ds41-expert-file <model> <model>/ie_ranking_decode_chat.txt
--cutoff 317 --end 361 --confirm`: 1,760 slots, 33.1 GB in 12.9 s; the volume back at 93 GB free. At load it
**covers 840 of 1,300 (card 0) and 880 of 1,340 (card 1) mmap experts** and verifies a slot byte for byte.

## A/B (bash-script prompt with Dream's 90 tools, 17,747 tokens, 250 greedy tokens, chat ranking, alternating)

| arm | decode ms/token | tok/s | join | permute per reader | prefill tok/s |
|---|---|---|---|---|---|
| off 1 | 91.1 | 11.03 | 21.3 | 2.50 | 264 |
| **on 1** | **85.4** | **11.75** | **16.5** | **0.46** | **277** |
| off 2 | 90.9 | 11.05 | 21.1 | 2.50 | 259 |
| **on 2** | **86.6** | **11.58** | **16.4** | **0.47** | **279** |

**Decode -5.9 % ms/token (+5.9 % tok/s), prefill +6 %**; no overlap between arms; experts per token identical
(106.3 / 66.7 / 7.0 / 59.9), so the placement and the arithmetic inputs are the same bytes.

## Shipped

`Engine::ds41_load` sets `IE_DS41_EXPERT_FILE=<model>/ie_experts_tail.ieslot` when the file exists and the variable is
unset; `IE_DS41_EXPERT_FILE=0` turns it off. Final server smoke (4 Dream-shaped turns): identical replies to the
previous build, decode 10.8-14.8 tok/s, turn-1 prefill 270 tok/s, 157 threads. If the ranking or the context (and so
the tier boundary) changes, coverage drops and the rest of the tier takes the pack path -- rebuild the window then.
