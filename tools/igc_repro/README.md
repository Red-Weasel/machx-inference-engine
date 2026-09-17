# igc_repro — standalone candidate repro for the BMG-G31 FMA non-determinism

## What this is

`dn_fma_nondet_repro.cpp` is a fully self-contained (single file, plain
SYCL 2020, no engine headers, no ESIMD/block-load extensions) structural
clone of the engine's `deltanet_recurrence` kernel
(`src/ops/deltanet.cpp`), built to reproduce the bug documented in
`docs/known_bugs.md` §1: run-to-run different fp32 results on
bit-identical inputs on Intel Arc Pro B70 (BMG-G31, Xe2-HPG).

It generates xorshift64-seeded synthetic inputs at the production shape
(B=1, chunk T=512, 32 v-heads, head_dim 128; L2-normalized q/k,
alpha=exp(g) decay gates in ~(0.6, 1], sigmoid beta), warms up a
realistic non-zero state from state=0, then runs 200 iterations. Each
iteration resets state to the same reference, launches the kernel 4×
back-to-back (carrying state, mimicking a 4×512 = 2048-token chunked
prefill), and FNV-1a-hashes `out[]` + `state[]`. Identical inputs every
iteration ⇒ every hash must equal iteration 0's.

## Build / run

```bash
icpx -fsycl -O2 tools/igc_repro/dn_fma_nondet_repro.cpp -o /tmp/dn_repro
/tmp/dn_repro                # defaults: --iters 200 --T 512 --chain 4
/tmp/dn_repro --T 1024       # longer-chunk variation
/tmp/dn_repro --wide         # wide-exponent-range variation on v
```

We deliberately use the default JIT path — the production engine does the
same; if IGC's codegen differs under `-fsycl-targets=` AOT, that
difference is itself relevant signal.

Expected output, either:

```
DIVERGED at iter N (hash 0x... vs 0x...)     # exit 1 — bug fired
```

or:

```
200/200 identical                            # exit 0 — bug did not fire
```

## Decision-gate outcome (2026-06-10): the synthetic repro does NOT fire

On NEO 26.14.37833.4 / IGC 2.32.7 / level-zero driver 1.15.37833+4:

| Run | Shape | Result |
|---|---|---|
| default ×3 (separate processes) | T=512, 200 iters × 4 chained launches | `200/200 identical`, same reference hash `0x661d36a3ef2eece9` across all 3 processes |
| variation ×1 | T=1024, 200 iters × 4 chained launches | `200/200 identical` (`0x15caf1c4d369b44e`) |

That is 800 kernel launches at T=512, plus 800 at T=1024 — 1,600 total
clean launches — of the exact kernel structure with zero divergence. Per the pre-agreed stop rule, only one structured variation
(T=1024) was attempted before stopping.

Interpretation: the failure in the engine is timing-/launch-state-
coupled (see `docs/bisect_step25_26_summary.md` — chunking the launch
made it worse, divergence iter shifts with instrumentation). The
isolated kernel in a quiet queue evidently does not recreate the
microarchitectural conditions (kernel-launch residue from ~30 other
kernel types per forward pass, 8 interleaved chains) under which the
bug fires within ~100 iterations in-engine.

## Consequence for the vendor issue

The issue draft (`docs/public/igc-issue-draft.md`) states this honestly
and references the in-engine reproduction instead:

```bash
./build/tools/ie-bug-monitor --max-iters 100
# fires "✗ DIVERGENCE DETECTED" typically within ~100 iters,
# but requires the 20 GB Qwen3.6-35B GGUF model
```

This standalone file is still attached to the issue as (a) the exact
kernel structure under suspicion and (b) evidence that the kernel in
isolation is deterministic — which narrows the bug to interaction with
surrounding launch traffic rather than the kernel's own code.
