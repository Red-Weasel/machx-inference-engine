# Arch Living-Authority Docs — Framework & Operating Strategy

This directory holds **one living authority doc per architecture family** (`gemma4.md`,
`crown.md`, `qwen35-27b.md`, `coder.md`, `qwen3next-80b.md`). Each is the **single source
of truth for that arch's optimization state** — its workload envelope, measured standing
vs llama, bottleneck map, the ledger of every lever tried (kept/reverted/specialized/dead),
the live shape-aware dispatch, and the next justified frontier. They are *living*: update
the relevant doc at every milestone for that arch; if reality and the doc disagree, fixing
the doc is part of the change. `MASTER_DEV_PLAN.md` stays the cross-arch roadmap; these are
the per-arch deep state it points into.

---

## 0. Core principle

An inference engine is a **mapping of model computation onto one physical machine** — arch ×
quant × layout × workload shape × GPU µarch × memory hierarchy × compiler codegen × queue/host
scheduling × context × batch × cache state. There is **no universal fastest engine**, only one
optimized for a defined envelope. The goal is **best end-to-end latency / throughput / memory /
quality for that envelope**, not the fastest single kernel.

Kernels *mature* by being trimmed at successive boundaries: correctness → memory (coalesce/pack)
→ work-group (threads/subgroups/registers/SLM) → arithmetic (fuse dequant/act/residual/norm) →
dispatch (fewer, larger launches) → workload (different paths per shape) → regression (stays fast
after compiler/driver/model drift). Every fix exposes the next bottleneck — that is the normal
loop, not failure. A "superhuman" reasoner narrows the search; **hardware measurement selects the
truth** — because mathematically identical kernels differ 5–20× from cache/registers/codegen/
queues that the math doesn't expose, and the config space is combinatorial.

## 1. What "optimized" means here

Not one benchmark number — a **current Pareto frontier over an explicit workload matrix**:

| Regime | Primary metric | Typical bottleneck |
|---|---|---|
| Decode, short ctx | tok/s | weight BW, quant GEMV, lm_head |
| Decode, long ctx | tok/s | KV reads, attention |
| Short prefill | tok/s | quant GEMM, launch count, packing |
| Long prefill | tok/s | attention, KV traffic, compute |
| Batch serving | total tok/s | batching, scheduling, KV layout |
| Warm prefix | TTFT | prefix-cache restore |

"Beat llama" counts **only** when matched: same model + quant file/hash, context, batch, KV dtype,
sampling, prompt tokens, warm/cold state, GPU/driver/power, and the same definition of wall time.
Never declare "fully optimized" — declare *optimized for a specific measured envelope*.

## 2. The operating loop (condensed A→Z)

```
1. CAPTURE   one clean serial GPU profile (--kprofile-decode / --kprofile / -verify).
             Box stable: GPU idle-clocked, one workload at a time, ≥3 trials, median+spread.
2. CLASSIFY  every hotspot into a bottleneck class BEFORE proposing code:
             compute | weight-BW | KV-BW | quant | registers/occupancy | SLM | dispatch | algorithm | scheduling.
3. ROOFLINE  per hot kernel: bytes, FLOPs, arithmetic intensity, effective BW/throughput vs ceiling
             → label memory- / compute- / launch- / occupancy-bound. No tuning without this label.
4. LEDGER    one row per hypothesis: observation → mechanism → scope(files) → expected END-TO-END win →
             risk → FALSIFIER → validation gates → decision. No "might help" without a falsifier.
5. FAN OUT   parallel CPU-only analysis on the ONE captured profile (never concurrent GPU):
             kernel/roofline · host-scheduler · weight-layout/quant · KV/cache · algorithm-path · SKEPTIC.
             The skeptic attacks the top proposals (wrong assumption? already exists? what regresses?
             apples-to-apples?) — the lens that stops 5 agents recommending the same seductive bad tile.
6. RANK      priority = expected_e2e_win × confidence × repeatability ÷ eng_cost ÷ regression_risk.
             Pick ONE primary patch (a second only if it touches a different subsystem + has its own gate).
7. MICROBENCH the exact production shapes (T=1/128/512, real H/quant/output) — not a toy GEMM.
8. PATCH     one objective, minimal files, feature-flag if risky, easy rollback, no unrelated cleanup.
9. VALIDATE  correctness ladder (compile/unit → tensor/dtype → kernel diff → calibration logits → PPL →
             deterministic gen → regression → soak) THEN the 3-run end-to-end matrix (median+spread,
             wall+GPU time, memory, quality, per-context behavior). PPL is necessary, NOT sufficient.
10. DECIDE   KEEP (improves matrix, no bad regression) · REVERT · SPECIALIZE (wins one regime → dispatch
             it only there — often the mature answer). Record the decision + restart at step 1.
```

## 3. Hard rules (non-negotiable)

- **One GPU workload at a time.** Never concurrent benches/profiles/loads — across agents/repos. Gate every
  GPU launch on the prior job's explicit done-marker AND `pgrep -x` empty. (See `feedback_gpu_exclusivity`;
  a 2026-06-25 double-load crashed the box.)
- **Never optimize without a captured profile.** Never accept a microbench win without end-to-end validation.
- **Never compare to llama** unless model/quant/context/prompt/KV/sampling/box are all matched.
- **Never combine unrelated changes** into one perf claim. Don't touch unrelated subsystems mid-experiment.
- **PPL gate holds** (`./build/tools/ie-perplexity` crown ≤ 6.57; per-arch PPL in each doc). Reasoning before code.
- Stop only at a **defined decision gate** (matrix beats llama / no proposal clears the threshold / a quality
  or stability gate fails / a new bottleneck class needs a human design call) — not because a subtask is hard.

## 4. Per-arch doc template (the sections every `<arch>.md` must fill)

```
0. Identity        — family, sizes, GGUF file(s)+role, forward path file, Engine dispatch tag, one-line standing.
1. Envelope        — the benchmark contract: hardware, model+quant, target modes, competitor build+flags, quality gate.
2. Correctness     — PPL gate (number+command), reference oracle, deterministic-gen check, the silent-bug landmines.
3. Pareto frontier — the measured matrix (decode/prefill × ctx/T × batch): ours | llama | WIN/LOSS/PARITY | bottleneck class. Clean-box + dated.
4. Bottleneck map  — per hot regime: dominant kernels (% of wall), roofline class, effective BW/util vs ceiling.
5. Hypothesis ledger — every lever: observation → mechanism → result KEEP/REVERT/SPECIALIZE/DEFER → commit/gate. INCLUDE the dead-ends (so they aren't re-tried).
6. Shape dispatch  — which kernel/path for which T/ctx/batch/dtype + the env gates (default-on/opt-in/opt-out).
7. Layout/sched    — what's reordered/SoA/fused/hoisted; host-stall & launch state; KV mode.
8. Open frontier   — the next justified bottleneck (observation+mechanism+falsifier+expected win); the standing verdict (real lever vs commoditized grind).
```

Keep each doc *strategic and honest*: state where we win, lose, and are at parity, and **why** — grounded in
measured roofline, not aspiration. A dead-end recorded is as valuable as a win shipped.
