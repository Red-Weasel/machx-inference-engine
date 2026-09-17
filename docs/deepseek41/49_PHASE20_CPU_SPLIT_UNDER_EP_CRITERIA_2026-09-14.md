# V4.1 port — Phase 20 gate criteria: the CPU miss split under expert parallel, the CPU experts' rows exported (written BEFORE the build)

**The term (docs/47, the 2,048-token context, 88.1 ms/token):** the MoE call is ~62 ms of the step and its wall
is the SLOWER card's link time per layer — the E[max] geometry of docs/39: of the ~70 pinned misses per
token, 55 land on whichever card's tier has more in that layer and 15 on the other, so the sum over layers
of max(owner, remote) is 36-39 ms while each link alone carries ~30 ms of bytes. The E-cores serve an
MXFP4 expert from the pinned arena at 2.75 experts/ms with 12 threads (docs/35, 0.36 ms per expert, 52
GB/s of slot bytes) against a link's 1.2 per ms; Phase 13's split (`IE_DS41_CPU_MISS=1`, q* 0.30) was worth
~9 ms/token without EP. Under EP the split is REFUSED in effect: docs/47 measured the combination once
and it died with `DEVICE_LOST` on the first decode step — the CPU leg's final add `y[i] += dp[i]` targets
`y`, which an exporting (remote) tier does not have. The export-mode guard in c771120 skips the leg with a
notice; it has not been run.

**The change, bounded to the tier.** The CPU leg stops producing a weighted fp32 partial sum added into
`y` and instead writes each CPU expert's output as ITS PACKED ROW: fp16 into `bws_.yp[off[e]]`, with the
real weight kept in `weights_packed` (today the row is zeroed and its weight set to 0). One H2D copy of
the n_cpu rows lands before the owner's scatter, or before an exporting tier's `q.wait()` — from then on
a CPU row is indistinguishable from a GPU row to the export protocol (the gather, the host hop, the
import) and to the one scatter that sums the six products in the whole tier's order. Both tiers split
their own misses on their own core partition (8-13 / 14-19, six E-cores each — `split_cores` exists),
and q* is per tier: with half the cores the CPU rate is ~1.4 experts/ms against the link's 1.2, so the
PCIe share is ~0.46 by the arithmetic of docs/35; the measurement decides (a sweep, below). Numerics:
the CPU expert keeps the activation in fp32 where the GPU quantises it to Q8 (docs/35: within 3.1e-4 of
the GPU's M = 1 path on a real expert), and the row is stored as fp16 like the GPU's rows — the forced
digits MOVE, the bars judge (1.2e-2 / 3.5e-2 / logits 5e-2); bit-identity is NOT claimed. The
non-EP split (Phase 13's path) goes through the same rows — its digits are compared with docs/35's.

**The arithmetic, stated before the build.** Per layer the slower card's ~1.4 link misses become ~0.65
on the link and ~0.75 on its six E-cores (concurrently, 0.5 ms), so its per-layer wall drops from ~1.1
ms toward ~0.6 ms where the fetch is the leg; over 40 layers **~10-15 ms/token** if the E-cores are not
starved by the readers (docs/35's gate 13 lesson: a pinned team under a foreign load collapses 15-23x —
the readers are confined to the P-cores by `split_cores`; the prefill cost of that confinement, docs/35's
+8%, is measured too and, if it holds, the pinning moves to decode only).

## Step 0 — before building (the guard, at a phase boundary, the founder aware)

The committed guard build (c771120) with `IE_DS41_EP=1 IE_DS41_CPU_MISS=1` for one decode bench: it must
NOT crash (an exporting tier prints the notice and skips; the owner's tier splits its own misses), and
its ms/token is the "owner-only split" number. If it crashes again, the card is lost until a reboot —
the arm is run right after the gate's verdict, before any long build, with the founder told first.

## Scope, in order

1. The CPU leg's product becomes rows (both modes), one H2D copy, the weight kept; the non-EP digits
   against docs/35's; the tier test's split check (a layer with the CPU leg forced on a random subset
   of experts equals the whole tier within a stated bar).
2. Under EP: both tiers split; q* per tier from a sweep {0.30, 0.45, 0.60} on the decode bench; the
   E[max] line (the slower card's LINK misses must fall; the wall per layer with it).
3. The prefill cost of the core confinement (the resident test, pp2048 cold / warm) — moved to
   decode-only if it costs more than the spread.
4. The founder table updated; the default decided by the measurement (on if ≥ 3 ms/token and the bars
   hold; off otherwise; never a silent fallback — the notice stays where a tier cannot split).

## Pass criteria

1. No crash in any arm; the guard's path exercised once and recorded.
2. Digits: the decode test's forced digits and logits under the bars with the split on (EP and off), the
   four greedy tokens the golden's; the resident test PASS (prefill's path untouched: the split is T = 1);
   generate PASS.
3. Measured on one build: the decode bench EP alone vs EP + split at each q*, a second sample of the
   best; the slower card's link misses and the remote wall reported; ≥ 3 ms/token or the change is
   reverted (the guard stays).
4. Prefill within the spread, or the confinement moved to decode.

## Explicitly NOT in this phase

DSpark (docs/48, the structural lever — this phase's rows path is what a T > 1 pass will also use, since
its union of experts has 3-4x the misses); `o_a` in FP8; any change to the link protocol.
