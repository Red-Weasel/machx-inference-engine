# V4.1 port — DSpark P3 gate criteria: rollback after a partial acceptance (written BEFORE the build)

**The contract** is docs/48 section E, P3. After a T-row verify step at pos0 (P2, docs/55) the loop accepts the
first L ≤ T rows and must return the state to exactly what L one-row steps from the same start would have left:
`Ds41Forward::rollback_to(pos0 + L)`.

**What a T-row step changes, and what the rollback restores (docs/48 C.2, re-derived on the P2 code):**

1. **The window ring**, every layer: the step wrote its T rows at slots (pos0 + r) mod WIN AFTER the attention
   (docs/55 finding 4). The slots of rows r ≥ L must get back the keys they evicted (positions pos0 + r − WIN, or
   nothing before the wrap). The step snapshots the T slots' contents before the write — [T, HD] fp32 per layer,
   12 KB at T = 6 — and the rollback copies rows L..T−1 back.
2. **The latents and index keys**, the kv-source layers: `nc` is a counter — ratio 1: nc_before + L; ratio 2:
   nc_before + the pairs closed by the accepted sequence (the held half, if any, then rows 0..L−1). The rows
   beyond nc are never read.
3. **The ratio-2 partial group** (layers 2, 8, 14): after the rollback the held half is the last element of the
   accepted sequence when its length is odd — row L−1's compressor output (or the pre-step half when L = 0), so
   the step keeps its per-row `ckv_p` / `cg_p` ([T, HD] × 2 per ratio-2 source) and the pre-step half.
4. **The host sequence**: `all_ids_` resized to pos0 + L, `n_pos_` = pos0 + L. The stream and mix carry-ins are
   recomputed per step from the embedding — no state.

Snapshots are taken only when the step has T ≥ 2 rows (a `multi` step): the T = 1 path is untouched. Under
expert parallel the state lives on the owner card of each layer; the rollback restores on that card's queue.

## Pass criteria (docs/48 P3, with the test that judges each: `ie-ds41-rollback-test`)

1. **Bit-identity**, for L ∈ {1 … 6} at pos0 ∈ {2048, 2049} (past the wrap) and {12, 13}: a T = 6 step from
   state S, `rollback_to(pos0 + L)`, then `read_state` for every layer + `n_pos()` + the id sequence equal the
   state after L one-row steps from S. (The reference is reached by a fresh prefill: deterministic.)
2. **Continuation**: from each rolled-back state one further one-row step (the token the reference's L-th step
   produced) gives logits bit-identical to the same step on the never-speculated path, and its state again.
3. **The negative control**: with the ring restore skipped (`set_rollback_noring_diagnostic(true)`) criterion 1
   must FAIL for some L < 6 at pos0 ≥ 128 — the evicted keys are what the restore is for.
4. Snapshot bytes per step and the snapshot / restore time reported; < 1 ms per step (D2D copies of ~100 KB).
5. Prefill and the T = 1 path unchanged (no snapshot at T = 1; the decode test's digits the P2 build's).

## Explicitly NOT in this phase

The loop (P4), the drafter (P1), any tuning of the T-row step.

## Results (10:09 / 10:23, `ie-ds41-rollback-test`; logs `~/ds41_work/p3/`)

**PASS — the wrapped ring 38 of 38, the short prompt 36 of 36** (an earlier draft of this line said 42 / 40: it counted the four `[ds41 …]` diagnostic lines too, the miscount gate P2 found in docs/55). Re-run 11:1x on the P1 build: 38 / 36 again, `rollback_to` 0.225 / 0.230 ms (`IE_DS41_CPU_MISS=0`, docs/55's environment for bit-identity; logs `~/ds41_work/p1/rollback_{long,short}_p1.log`). For L ∈ {1 … 6} at pos0 2048 / 2049 and 12 / 13: after a
T = 6 step and `rollback_to(pos0 + L)` every layer's ring (its live slots), latents, index keys and nc equal L one-row
steps' from the same start, `n_pos` and the id sequence equal; one further one-row step from the rolled-back state
gives the never-speculated path's logits bit for bit and its state again (criterion 2 — this is also what consumes the
restored ratio-2 half, gate P2 finding 4); skipping the ring restore leaves a different state at L = 2 past the wrap on
both parities (criterion 3). `rollback_to` **0.228 ms** mean (12 calls; D2D copies of at most 8 slots per layer plus
the compressor halves); the snapshot per step is 4 memcpys per layer at T = 6 (criterion 4). The T = 1 path takes no
snapshot (criterion 5; the decode test's digits are P2's).

**One test correction:** the first short-prompt run failed at L = 6 on all 40 layers because the test compared whole
rings — below the wrap the slots past `n_pos` hold whatever earlier iterations of the same process wrote there (the
prefill rewrites only its last min(T, window) slots and `reset_state` keeps the rest), and no step ever reads them (the
mask closes positions < 0). The comparison now covers the live slots (positions ≤ n_pos − 1; all of them once wrapped).

## Gate (12:00, an independent evaluator; logs `~/ds41_work/gate_p3p1/`): PASS WITH FINDINGS

The evaluator re-ran both scenarios on the pinned binaries (38 / 38 and 36 / 36; the negative control fires at both
parities; `rollback_to` 0.227 / 0.225 ms) and re-derived criteria 1-5 from the code. Findings taken: (3) the id
sequence was compared only indirectly (through the continuation's engram hashes) — `Ds41Forward::all_ids()` added and
the test compares it directly (38 / 38 + 36 / 36 again on the P4 build, 0.227 / 0.229 ms); (4) only the restore half is
measured — the snapshot is a kernel over the T evicted slots per layer plus 2-4 memcpys on the three ratio-2 sources,
≈ 0.58 MB per T = 6 step (derived), its time inside the step (336.8 / 313.7 ms at 2048 / 2049 on the P1 build against
P2's 344.1 / 312.3 — noise-level, not a measurement; "4 memcpys per layer" above was loose). The evaluator's run also
showed the T = 6 step at 454 / 475 ms after the 12-token prompt (the short regime).
