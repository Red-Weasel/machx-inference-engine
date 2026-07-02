# Dense-path run-to-run nondeterminism — bisect, root cause, fix (2026-06-10)

**STATUS: RESOLVED — software bug, fixed.** Read/write race in `rope_partial`
(`src/ops/elementwise.cpp`) under the in-place calls every model path makes.
NOT the Xe2 hardware-FMA class from `docs/known_bugs.md` §1 — that verdict was
considered and *eliminated* by isolation.

## Symptom (P2 T8 gate blocker)

`ie-perplexity` on qwen3-8b (dense), identical binary + corpus, default
streaming T=1: PPL 19.06 / 19.42 / 18.98 / 19.22 / 19.03 / 19.16 / 19.13 /
19.25 across 8 runs (range 0.44). fp16 decode (`IE_NO_Q8_DECODE=1`) also
affected (19.32/19.21/18.95). Crown (Qwen3.6-35B) bit-exact on the same
binaries (avg NLL 1.864495 to the last digit, 3 runs).

## Bisect trail (single-variable, hash-checkpointed)

| # | Step | Result | Eliminates / establishes |
|---|---|---|---|
| 1 | 511-tok verbose run ×2, diff per-token NLL | First divergent token = **i=1 (ctx_len=2)**; i=0 bit-stable across 6 runs | Not long-ctx onset. Every op already ran at i=0 (gemv_fp16 ×36, all GEMVs, norms, FA-2 at ctx=1) deterministically |
| 2 | `ie-det-probe`: isolated-kernel loops on frozen inputs, 300 iters, bitwise output hash | `fa2_decode` dense 32/8/128, `fa2_decode` crown 16/2/256, naive `full_attention` T=1, `gemv_fp16` 4096×1024 — **all BIT-STABLE** | Not an intrinsic-kernel flicker at ctx=2; FA-2 split-K and gemv_fp16 exonerated in isolation |
| 3 | `ie-det-probe --test model`: in-process reps of {kv.reset; 8 × T=1 steps}, full-logits hash per step | step 0 (ctx=1) bit-stable; steps 1–7 diverge **every rep** (8 distinct hashes / 8 reps) | In-process repro; ctx=1 vs ctx=2 boundary confirmed at bit level |
| 4 | Per-layer residual dumps at step 1, 4 reps | L00 (embedding) bit-equal; **L01 (after layer 0) diverges**, ~3900/4096 values, max_abs ≤ 0.006; reps 2,3 bit-identical to each other (fixed point) | Culprit inside layer 0; behavior scheduling/timing-dependent, not random per-element |
| 5 | Perturb test: isolated FA-2 with iteration-varying garbage in cache rows ≥ ctx and unused partials scratch | BIT-STABLE | FA-2 reads no out-of-range memory — stale-KV hypothesis dead |
| 6 | `IE_DENSE_HASH=1`: FNV hash of every section output buffer, all 36 layers, 4 process-separate runs | attn_norm, q/k/v_proj, qk_norm all bit-equal; **first divergent buffer = `rope_q` at pos=1**; first affected layer varies run-to-run (L=0 vs L=3) | Surface = `rope_partial` on Q at pos ≥ 1, stochastic low-probability per-launch |

## Mechanism (read of the kernel, confirmed by shape arithmetic)

`rope_partial` used one work-item per (token, head, dim). For a rotary pair
(r, r+half), the cos-side item `d=r` and the sin-side item `d=r+half` **each
read both** `x[r]` and `x[r+half]`, and each writes one of them. Every caller
runs RoPE **in place** (`x == y`): dense `dense_transformer.cpp:329-332`,
crown `qwen36.cpp:965-967`. That is a cross-work-item read/write race.

Shape dependence — why dense fires and crown never did:

| | crown (Qwen3.6) | dense (qwen3-8b) |
|---|---|---|
| head_dim / n_rotary | 256 / 64 | 128 / 128 |
| half | 32 | **64** |
| WG along dim axis | 64 | 64 |
| pair (r, r+half) placement | same 64-wide WG (adjacent subgroups, co-scheduled) | **different work-groups** → freely scheduled across Xe-cores | **(no barrier between the racing reads/writes — intra-WG placement gave no spec-level guarantee; crown's immunity was subgroup-lockstep scheduling luck, not safety. The race was latent in the crown path too, which is why this fix matters beyond dense.)**

Why pos=0 (ctx=1) is bit-stable: angle = 0 → cs=1, sn=0 → both outputs equal
their inputs; the racing write stores the same bits, so the race is
value-invisible. First nonzero position makes it visible — exactly the
observed i=1 boundary. Also explains the session-disjoint PPL bands
(T7: 18.74–18.93 vs T8: 18.98–19.42): race outcomes ride scheduling state.

## Verdict

**SOFTWARE bug** (in-place data race), not the known_bugs.md §1 hardware
class. The HW-class verdict was earned-then-rejected: isolated kernels are
bit-stable, the divergence is value-invisible at angle 0, and the fix flips
the entire path bit-exact.

## Fix

`src/ops/elementwise.cpp` (`rope_partial`): one work-item now owns the whole
pair (r, r+half) — reads both inputs, then writes both outputs. In-place safe
by construction. Launch geometry and per-element arithmetic expressions
unchanged (items d ∈ [half, n_rotary) become no-ops); per-pair trig is
computed once instead of twice.

## Verification

| Check | Result |
|---|---|
| `ie-det-probe --test model --iters 8` | all 8 steps BIT-STABLE (was: 8 distinct hashes at step 1) |
| Dense PPL, builtin 511-tok, default decode, **10 process-separate runs** | avg NLL **2.940491** / PPL **18.9251** — bit-identical 10/10 |
| Dense PPL, fp16 decode (`IE_NO_Q8_DECODE=1`), 2 runs | avg NLL 2.943471 / PPL 18.9816 — bit-identical |
| **Crown invariant**: default `ie-perplexity`, 3 runs | avg NLL **1.864495** / PPL 6.4527 — exact, unchanged |
| `ctest` | 15/15 pass (incl. rope reference test, attention vs scalar ref, dense_forward_smoke greedy) |

New dense PPL gate: see `docs/ppl_baseline_matrix.md` — the dense path is now
deterministic, so the gate is exact equality (2.940491), not a statistical band.

## Tooling added (kept)

* `tools/ie_det_probe.cpp` (`ie-det-probe`) — kernel- and model-level
  determinism probe: frozen-input bitwise hash loops (`--test
  fa2_dense|fa2_crown|naive|gemv_fp16`), out-of-range perturbation mode
  (`--perturb`), in-process full-model repetition (`--test model`), per-step
  layer dumps (`--dump-step N`).
* `IE_DENSE_HASH=1` — env-gated per-section output-buffer hashing in
  `DenseModel::forward` (zero cost unset).

## Possible follow-up (not done here)

The 2026-05-02 chunking-validator finding that naive `full_attention` T>1 was
"intrinsically non-deterministic" (comments in `src/ops/attention.cpp`,
Steps 3–6) predates this fix and was observed on the crown prefill path,
where RoPE runs in place at T>1 over many positions — the same race could
plausibly explain part of that record. Worth a one-shot re-run of
`ie-validate-chunking` before ever re-opening that file's bisect scaffolds.
The DeltaNet recurrence verdict (known_bugs.md §1) is unaffected: DN layers
have no RoPE, and that bisect hash-verified bit-exact *inputs* into the
diverging kernel.
