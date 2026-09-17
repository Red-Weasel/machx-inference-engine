# V4.1 port — Phase 19 gate criteria: the expert-parallel helper's timeline, and the routing union (written BEFORE the build)

**Why this phase comes before DSpark (docs/48).** The design brief's cost model puts DSpark's ceiling on
this hardware at ~18 tok/s even at perfect acceptance (the union of six rows' experts anti-amortises the
link-bound fetch: misses per layer per token 1.79 → 2.81), and ranks two T = 1 levers above it: (a) the
E[max] imbalance of the misses across the two links (55 of 70 per token on the slower card — Phase 20,
docs/49, absorbs the excess on the E-cores) and (b) **the expert-parallel helper's serialisation**:
docs/47's columns (the owner's groups 38.7 + tail 19.4 = 58.1 ms against a remote wall of 36.1-36.8)
are consistent with the remote card's tier starting ~0.5 ms late per layer and with a serial tail
after it — up to ~19 ms/token, "derived from columns, not verified by a run" (docs/48 F.2). What the
code does per layer today (deepseek41_forward.cpp, the EP block): the owner copies x to its pinned
buffer and WAITS; a NEW `std::thread` is spawned for the remote card, which sets its OpenMP block
time, copies x host-to-host, enqueues the H2D, runs the remote tier in export mode, then gathers its
rows with a kernel and a D2H WAIT; the owner's import hook JOINS the thread, copies the rows
host-to-host, enqueues an H2D and the placement kernel and WAITS — all of it serial with the owner's
own tier around the one overlapped stretch. Two things are unknown: how late the remote's tier
actually enters relative to the owner's, and how long the tail after both tiers finish really is.

## Step 0 — measure before building (instrumentation only; no algorithm change)

(a) **The EP timeline per layer**, timestamps taken from the owner's MoE-call start `tm` and summed
per token by the decode bench: the staging D2H done; the helper thread's entry; the remote tier's
`moe()` entry and return; its rows landed on the host; the owner's own tier entry; the import hook's
entry (= the owner's tier is at its scatter) and exit; the call's end. One decode bench at the
2,048-token context on the committed build's path (EP + the file + FP8 + the Phase 18 defaults). The
numbers name the term: the spawn latency, the remote's late start, and the serial tail.
(b) **docs/48's P0, in the same run:** `IE_DS41_DUMP_ROUTING=<file>` writes, per decode step and layer,
the six selected experts with their tier (static / pinned / mmap); a script computes the union of
T consecutive steps' experts per layer for T = 1..6 (and the same for shuffled steps) — the routing
correlation, measured. This decides DSpark's ceiling (docs/48 D.3) and costs nothing here.
(c) **docs/48 F.3:** the attention kernel at T = 1 vs T = 6 on the decode shapes (the existing
`deepseek4_attn` bench, no model load) — the marginal row's cost.

## The change (after step 0 names the term), bounded to the EP block

1. **A persistent helper thread per remote card** (spawned at `init_resident`, parked on a condition
   variable, its OpenMP block time set once), posted a job per layer instead of a `std::thread` per
   layer; the import hook waits on its completion flag instead of joining. Bit-identical (the same
   lambda body runs).
2. **The serial tail**: whatever (a) shows — e.g. the remote's rows D2H and the owner's H2D + placement
   are today two waits in sequence; the rows can be placed by one kernel that reads the remote's
   pinned buffer directly if the owner's queue may read another context's host USM (measured, not
   assumed: a probe), or the two copies overlapped with the owner's own scatter prep.
3. Kill switch `IE_DS41_EP_THREAD=0` (the per-layer thread); measured against it on one build.

## Pass criteria

1. Step 0 reported whatever it says: the timeline's per-token sums and per-layer means, the routing
   union table (consecutive vs shuffled, T = 1..6, with the tier split and the implied link bytes),
   and the attention kernel's T = 1 / T = 6 times.
2. The change bit-identical to the Phase 18 path (the dumps at 0.0 at all 40 layers, the decode test's
   digits), the whole test set PASS.
3. Measured on one build: the decode bench with the persistent helper vs the per-layer thread; the
   timeline's spawn and tail terms moved as predicted, the step by ≥ 2 ms/token or the change is
   reverted (kept only if bit-identical AND measured).
4. Prefill within the spread (the EP block is the same code at prefill; a change there is measured
   on the resident test's pp2048 cold / warm).

## Explicitly NOT in this phase

The CPU miss split (Phase 20, docs/49); DSpark's build (docs/48 P1-P5, after P0's number); any change
to the tiers' partition of experts.
