# V4.1 port — DSpark P4 gate criteria: the loop, greedy, lossless (written BEFORE the build)

**The contract** is docs/48 section E, P4, on the protocol of B.3 and the pieces P1-P3 delivered (docs/55-57): one
pass = one drafter forward (`Ds41Drafter::draft`, 12-18 ms) + one backbone verify at T = 1 + k rows
(`IE_DS41_DECODE_MULTI`'s path, bit-identical to T one-row steps, 333-443 ms at T = 6 with the split off) + the
greedy acceptance + `rollback_to` (0.23 ms) + the drafter's rings re-seeded for the accepted rows. Fixed k = 5.
Temperature 0 only. Opt-in.

## The algebra (docs/48 B.3, re-derived on the P1-P3 code)

State before a pass: the backbone holds positions 0 … p (`n_pos() = p + 1`), the drafter's rings hold main_kv for
the same positions, the last backbone forward produced `main_hidden(p)` (`Ds41Forward::main_hidden()`, the row of
position p) and the logits from which the token t_{p+1} was sampled.

1. `draft(main_hidden(p), t_{p+1}, p)` → `ids = [t_{p+1}, d_1 … d_5]`. t_{p+1} is emitted now, as the plain loop
   would. **Corrected after the gate (finding 2):** `draft` writes NO ring slot — only `seed` does — so every
   position's `main_kv` must be seeded explicitly, the prompt's odd tail row included.
2. Verify: `forward([t_{p+1}, d_1 … d_5], T = 6, pos0 = p + 1)` with all six rows' logits. Row r sits at position
   p + 1 + r and predicts the token at p + 2 + r; its capture `main_hidden()` row r is `main_hidden(p + 1 + r)`.
3. Accept greedily with THE SAME sampler the plain loop uses (`Ds41Generator::sample` at temperature 0 = argmax after
   the repetition penalty over the running `recent`): a_r = sample(row r, recent so far); d_{r+1} is accepted iff
   d_{r+1} == a_r and every earlier draft was; L = 1 + the matched prefix (1 ≤ L ≤ 6). The committed tokens are
   d_1 … d_{L−1} and the bonus a_{L−1}, each pushed through the plain loop's eos / piece / stop machinery IN ORDER —
   an eos or a stop string among them ends the run there and nothing past it is emitted.
4. `rollback_to(p + 1 + L)`: rows 0 … L−1 stay, the rest is undone (docs/56).
5. The rings: `seed(main_hidden rows 0 … L−1, L, p + 1)` (slot p + L is rewritten by the next draft, the same bytes).
   The next pass is at p' = p + L with `main_hidden(p')` = verify row L−1 and t_{p'+1} = the bonus.
6. Near the token budget the verify still runs T = 6 and the emission is cut at `max_new`; the run refuses a prompt
   whose `T + max_new + 6` exceeds the runtime's position capacity (the plain loop's check plus the block).

The prefill seeds the rings once from the prompt's capture (`seed(main_hidden, Te, 0)`: the last min(Te, 128) rows
land; under bounded replay the capture holds exactly those rows). An odd prompt's tail step gives `main_hidden(Te)`.

## The build (bounded)

1. `Ds41Generator` gains the speculative mode: `set_drafter(Ds41Drafter*, sycl::queue*)`; `run` speculates when a
   drafter is attached AND `temperature <= 0`; with a drafter attached at temperature > 0 it prints one line to
   stderr ("speculation off: temperature > 0") and runs the plain loop — no silent fallback. `Ds41GenStats` gains
   `spec_passes`, `spec_hist[7]` (the count of passes at each L = 1 … 6), `spec_offered[5]` / `spec_accepted[5]`
   (per draft position) and `spec_conf` (one (position, confidence, accepted) record per offered draft).
2. `ie-ds41-run` and the server's `ds41_load` (src/engine/ds41_engine.cpp) build the drafter on the head card and
   attach it when `IE_DS41_SPEC=1` (the drafter's tier residency from the card's free VRAM after the backbone's).
3. `ie-ds41-spec-test <model> <golden dir> [lossless|stats]` (a new tool): `lossless` — for each of 6 prompts (the
   golden's 12-token prompt; five English prompts held in the test: a fact question, a short story opening, a code
   request, a list request, an explanation request), 256 greedy tokens plain, then 256 speculative from the same
   state (a fresh prefill), the id streams compared token for token; plus the stop cases (a stop string, the
   eos-within-budget prompt if any of the six ends early). `stats` — 6 prompts × 512 speculative tokens, the
   acceptance histogram and the per-position rates and the confidence records written to a file.

## Pass criteria (`ie-ds41-spec-test`, run with `IE_DS41_EP=1 IE_DS41_EXPERT_FILE`, `IE_DS41_DECODE_MULTI=1`)

1. **Losslessness — the phase's whole point.** Under `IE_DS41_CPU_MISS=0` (docs/55: the CPU miss leg makes a T-row
   step differ from T one-row steps in the last bits) every prompt's speculative stream equals the plain greedy
   stream TOKEN FOR TOKEN over 256 tokens (6 × 256). Any divergence is a hard failure. The same comparison with the
   split ON (the default) is REPORTED (the number of prompts and the first differing position, if any): argmax is
   expected to survive 1e-3 differences except at near-ties; it is information for P5's environment choice.
2. The acceptance histogram over ≥ 1,000 passes (the `stats` mode): the distribution of L, its mean, the per-position
   acceptance rate for draft positions 1-5 — against docs/48 D.3's requirement (E[L] ≈ 4.1 for 20 tok/s in the
   optimistic calibration; 18.1 tok/s at L = 6 in the central one).
3. The confidence head measured, not used: the empirical acceptance rate per draft position bucketed by the raw
   confidence scalar (8 buckets), from the same records. Reported only.
4. Stop conditions identical with speculation on and off: the stop-string case ends at the same token with the same
   text and `stop_reason`; a run that hits eos ends with the same `n_gen`; nothing is emitted past an eos that lands
   mid-block. The test constructs the mid-block case by choosing `max_new` so that the plain run's last token falls
   inside a block.
5. The backbone's own tests unchanged (they do not go through the generator; the decode / resident / multi tests on
   the build); the generate test passes as before (no drafter attached = the switch off).
6. Reported: per pass the drafter ms, the verify ms and the rollback ms; ms per committed token at the 12-token and
   the 2,048-token contexts, next to the plain loop's on the same binary — information for P5 (not a bar here).

## Explicitly NOT in this phase

The confidence scheduler (k stays 5), temperature > 0 (speculative sampling is its own phase), any speed work (P5),
batch > 1, the drafter's pool placement.

## Results (11:47 build; `ie-ds41-spec-test` md5 a65032a0 for the runs below; logs `~/ds41_work/p4/`)

**PASS — lossless on every prompt.** Six invocations, each foreground and under the 10-minute cap, all with
`IE_DS41_EP=1 IE_DS41_EXPERT_FILE IE_DS41_CPU_MISS=0 IE_DS41_DECODE_MULTI=1`:

1. **Losslessness** (`lossless 0-2`, `lossless 3-5`): every prompt's speculative stream equals the plain greedy stream
   token for token — prompt 0 (the golden's 12 tokens) 9 tokens to eos, 1 ("The three primary colors are") 7 to eos,
   2 (the story opening) 144 to eos, 3 (the Python prompt) 0 tokens (eos at once, both ways), 4 (the five reasons)
   200 to eos, 5 (the bill) 19 to eos; `stop_reason` and the text agree on each. The 2,048-token context: 48 tokens
   equal. Criterion 1's "with the split ON" comparison was NOT run (every invocation used the split-off environment);
   it stays open for P5, whose environment choice it informs.
2. **The acceptance histogram**, 1,031 passes over prompts 2, 4, 6 (the 2,048-token text, 1,024 tokens), 7 (its
   first 512 tokens, 676 to eos) and 8 (its first 1,024 tokens, 641 to eos): L = 1 … 6 in **325 / 255 / 194 / 108 /
   55 / 94** passes, **mean 2.61 tokens per pass**; per-position acceptance (judged drafts) **0.685 / 0.640 / 0.574
   / 0.580 / 0.631** (1,031 / 705 / 448 / 257 / 149 judged). Prompt 4 alone reaches 3.85 (0.85 / 0.79 / 0.85 / 0.83 /
   0.75); the story 1.92 (0.60 / 0.49 / 0.14 / 0.00). Against docs/48 D.3: the central calibration wants ~4.1 for 20
   tok/s; the measured 2.6 sits between its L = 2 and L = 3 rows.
3. **The confidence head, measured** (2,590 judged drafts, the raw scalar; buckets [-9,0) [0,1) [1,2) [2,3) [3,4)
   [4,6) [6,20)): position 1 acceptance 0.34 / 0.53 / 0.85 / 0.95 / 0.98 / 1.00 / 1.00 (166 / 380 / 196 / 85 / 45 /
   62 / 97); position 2 0.35 / 0.67 / 0.85 / 0.94 / 1.00 / 1.00 / 1.00; positions 3-5 the same shape. Among drafts
   scored ≥ 1.0 the acceptance is 0.91-1.00 at every position (47 % / 34 % / 25 % / 24 % / 27 % of the drafts); ≥ 2.0:
   0.98-1.00. The head separates well; a threshold scheduler is the obvious P5 lever. Records: `conf_*.tsv`.
4. **Stop conditions**: the stop string "." ends both loops at " Berlin" with `stop`; a budget of 7 (inside the second
   block) ends both after exactly 7 identical tokens with `length`; the eos cases in item 1 end at the same `n_gen`.
5. **The backbone's own tests on the build**: dspark 153 / 153 (the opened-columns check added, gate P1 finding 2),
   rollback 38 / 38 + 36 / 36 with the id sequence compared directly (gate P3 finding 3; 0.227 / 0.229 ms), generate
   15 / 15 (the switch off; 82.0 ms/token after the 2,048-token text), decode with the ranking 28 / 28 (83.9 ms/token,
   the split ON).
6. **Reported, per pass** (draft + verify + rollback): at the 2,048-token context **21.2 + 341.5 + 0.30 ms** (prompt
   6), 20.9 + 390.6 + 0.29 (7), 25.6 + 493.2 + 0.29 (8); at the short prompts the verify is 700-1,160 ms (the
   disk-bound short regime). **ms per committed token: 143.8 at the 2,048-token context** (prompt 6, 1,024 tokens)
   against the plain loop's 82-84 on the same build — speculation is **slower** today: 363 ms per pass / 2.53
   tokens. The plain loop's own number inside the spec test is noisy (130.6 and 351.5 ms/token for the same 48-token
   run in two invocations: the cold steps after a prefill, disk-bound), so P5 measures with warm-up steps discarded.

**What P5 inherits, in order of size (measured, not derived):** (a) the verify step — 341 ms at T = 6 against the
plain step's ~84, i.e. the six-row union of experts over the links (docs/48 D.3's central case predicted 332: the
model was right); (b) acceptance 2.6 of 6 — the confidence threshold (item 3) trades verify rows for acceptance:
verifying only drafts scored ≥ 1.0 keeps ~0.9 acceptance on the rows kept; (c) the drafter's 21 ms (docs/48 assumed
8; its 88-100 pinned experts per stage cross the link every pass).

**Build notes.** The loop lives in `Ds41Generator::run` behind `set_drafter`; the run tool and the server build the
drafter under `IE_DS41_SPEC=1` (they set `IE_DS41_DECODE_MULTI=1` for the verify block themselves). The forward's
`main_hidden` capture is now on only when a drafter consumes it (`set_capture_main_hidden`, gate P1 finding 7). The
seed projects only the last window's rows. The drafter's expert-tier residency differed between invocations (39 / 89,
28 / 100 static / pinned per stage): it takes what the backbone's tiers leave on card 1, which varies with the free
VRAM at init — P5 should pin it.

## Gate (13:39, an independent evaluator on the pinned binaries; logs `~/ds41_work/gate_p4/`): PASS WITH FINDINGS

**VERDICT P4: PASS WITH FINDINGS** — 81 checks over 9 invocations, 0 failures; the evaluator reproduced every
discrete number in the Results above exactly (1,031 passes, the L histogram, the per-position rates, the confidence
buckets, 2,590 records) and found no fabrication. Its findings and what was done, all on one build (13:43-14:15,
`ie-ds41-spec-test` md5 e38104db, `ie-ds41-run` defb598f):

1. **(defect, TAKEN) The shipped `IE_DS41_SPEC=1` switch was lossy in the engine's default environment.** The
   evaluator ran the comparison this doc had left open and prompt 2 **diverged at token 49**: `IE_DS41_CPU_MISS` is
   ON by default (gate 20), the tools set only `IE_DS41_DECODE_MULTI=1`, and the verify agrees with one-row steps
   bit for bit only with the split off (docs/55). **Fixed:** `ie-ds41-run` and `ds41_load` now set
   `IE_DS41_CPU_MISS=0` as well, and both do it BEFORE `init_resident` reads it (the old lines sat after it, so the
   DECODE_MULTI one worked only because the forward reads that per call). The cost is near zero under speculation:
   the CPU leg serves T = 1 misses only (`deepseek41_experts.cpp:517`), so a verify step never used it. An explicit
   `IE_DS41_CPU_MISS` in the environment still wins, and the banner then says the output may differ from plain
   greedy. **Verified:** with the switch on and nothing else set, the tier prints no CPU-miss line and
   " red, blue, and yellow." is byte-identical to the plain loop run the same way (`p4/run_spec_fixed.log`,
   `run_plain_fixed.log`). The guarantee is against plain greedy IN THE SAME environment, which the banner states.
2. **(doc error + quality defect, TAKEN) `draft` writes no ring slot; the odd prompt's tail row was never seeded.**
   The algebra above is corrected, and the generator now seeds the tail position after the odd-tail step — the
   drafter was attending a zeroed slot for up to 128 passes on any odd-length prompt (correctness unaffected: the
   drafter only proposes; the cost was acceptance). Verified on the 5-token prompt.
3. **(defect, TAKEN) Criterion 1's scope, and one vacuous check.** No prompt of 0-5 reached 256 tokens (they end at
   eos: 9 / 7 / 144 / 0 / 200 / 19) and prompt 3's `[ ok ]` compared two empty vectors. **Fixed:** the prompt set
   gained five document continuations (the 2,048-token text and its first 256 / 512 / 768 / 1,024 tokens), the
   default range covers them, and a prompt that emits nothing now reports `[info] NOT evidence`, never `[ ok ]`.
   **Verified: five prompts at exactly 256 tokens each, every stream token-identical** (prompts 6, 7, 8, 9, 11;
   prompt 10 ends at eos after 209, also identical) — docs/48 P4's "≥ 5 prompts × ≥ 256 tokens" is now met, ~1,800
   compared tokens in total.
4. **(defect, TAKEN) No rollback when the run ended inside a block.** On eos, a stop string, a callback stop or the
   token budget the loop broke out before `rollback_to`, leaving the caches, `n_pos` and `all_ids_` describing
   rejected drafts. Harmless today (every `run` re-prefills) and a silent corruption for any future prompt-cache
   reuse. **Fixed:** the block is always rolled back, to `L − 1` when the last accepted token was the one that
   ended the run (an eos is never part of the output, so its position is dropped too).
5. **(doc error, TAKEN) Criterion 5 was narrowed and under-run.** docs/48 P4 lists generate, decode, replay,
   resident and forward; this doc had said "decode / resident / multi". **All of them now run on this build:**
   forward 5 / 5, replay 55 / 55, resident 14 / 14 (pp2048 372.9 cold / 403.9 warm tok/s), generate 15 / 15,
   decode 28 / 28, multi 38 / 38, rollback 38 / 38, dspark 153 / 153.
   **And it found a real cost:** the forward test's "state returns to start within 128 MiB" bar FAILED at 128.453
   MiB (history: 127.70-127.77). The cause was mine — P3's rollback snapshots are ~0.75 MiB of per-card state that
   `ensure_state` allocated on every layer whether or not anything ever speculates. **Fixed by allocating them on
   first use** (`ensure_snapshots`, called only for a decode step with T ≥ 2), not by moving the bar: the test now
   reads **127.699 MiB** and passes, and rollback / multi / lossless still pass on the same build.
6. **(risk, the campaign's decision point) E[L] = 2.60 against docs/48 D.3's ~4.1, and the loop is 1.4-1.6× slower**
   than the plain loop (the evaluator measured 142.3 vs 86.4 ms/token; this build reads 117-139 vs 89-96 on the five
   document prompts). Not a criterion failure — P4's speed criterion is "reported" and P5 owns speed — but the
   founder should see it before P5 is funded. P5's criteria are docs/59.
7-10. **(notes, recorded)** the `stats` mode enforces only "passes > 0" (the ≥ 1,000-pass bar is the operator's
   aggregate across invocations: 1,031 reached); criterion 4's mid-block construction covered the budget but not a
   stop string past the first block — **now tested** (the first candidate the plain loop hits mid-stream: " the" at
   token 8, both loops identical); `main_hidden_`'s "only the last Tl rows are written" invariant is unasserted;
   `Ds41Drafter::seed` mallocs and frees seven device buffers **per pass** (a P5 item, now in docs/59); and the
   `gt-c6` idle check is unreliable on card0 (Xorg and a system monitor hold it) — `pgrep` is the trustworthy
   signal, which the next gate brief should say.
