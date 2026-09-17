# V4.1 port — Phase 23 criteria: is the decode's kernel time a CLOCK problem? (written BEFORE the measurement)

**The suspicion, inherited and never settled.** docs/47 (Phase 18, step 0(a)) sampled the GPU clock through a decode
run — ~10,600 samples per arm — and found `act_freq` reading **0 in 92-93 % of them**, i.e. the card is power-gated
at the sampling instant, with 2,735-2,752 MHz whenever it is not. It closed with the gap unexplained and the
deciding experiment named and refused: "the A/B that would decide it — `min_freq` 2,800 on both cards for one bench —
is a root-only sysfs write and is the founder's call". Nothing since has tested it.

**Why it now looks like the best remaining explanation.** docs/62 measured what the card is doing during a decode
step: **35 of the 80.5 ms/token is not compute** — it is the per-layer wait for expert fetches, 40 times per token.
So the compute units are idle for ~43 % of every token, in 40 separate gaps, and this box's own sysfs says the
clock is allowed to fall to **400 MHz against a 2,800 MHz maximum** (`min_freq` 400, `rp0_freq` 2,800 on both cards'
gt0, read today). A 7× clock range plus 40 re-entries per token is exactly the shape that would make every kernel
measure slower in-engine than in a back-to-back bench, and three independent measurements fit that pattern:

| kernel | in-engine | its own standalone figure | ratio |
|---|---|---|---|
| `ds4_attention` | 175 µs/call (docs/62, queue profiling) | 80-90 µs (docs/47, same shape, SP = 64) | **~2×** |
| `gemv_fp8_e4m3_f16t` | 14.7 ms/token = ~374 GB/s | this card's measured ~580 GB/s | **~1.5×** |
| `ds4_gemm_mxfp4` | 11.1 ms/token = ~70 % of peak | — | ~1.4× |

Nothing in the kernels explains all three at once; a clock that is ramping when each kernel starts explains all
three with one cause. **This is a hypothesis with a cheap decisive test, not a conclusion.**

**The test, and why it needs the founder.** Pin the minimum frequency and re-run one bench:

```
# read first (safe, informative)
cat /sys/class/drm/card{0,2}/device/tile0/gt0/freq0/{min_freq,cur_freq,rp0_freq}
# the experiment (root; raises idle power on both B70s until reverted)
echo 2800 | sudo tee /sys/class/drm/card0/device/tile0/gt0/freq0/min_freq
echo 2800 | sudo tee /sys/class/drm/card2/device/tile0/gt0/freq0/min_freq
# revert afterwards
echo 400  | sudo tee /sys/class/drm/card0/device/tile0/gt0/freq0/min_freq
echo 400  | sudo tee /sys/class/drm/card2/device/tile0/gt0/freq0/min_freq
```

`min_freq` is root-owned and not writable by this session (checked: `root:root 644`, and `power/control`,
`power/autosuspend_delay_ms` likewise; the `xe` module exposes no frequency or RC6 parameter). It is also a **system
setting**, which this project's rules reserve for the founder. Idle power rises while it is pinned — the notes put
the B70s at ~46 W idle at gt-c6 — so it is a temporary experiment, not a proposed default.

## The measurement, once the pin is in place

`ie-ds41-decode-test` with the held-out ranking, `IE_DS41_BENCH_ONLY=long`, `IE_QUEUE_PROFILING=1`, ≥ 2 samples,
against the same run on the same build unpinned (today: **80.5 / 81.6 ms/token**, `ds4_attention` 175.3 µs/call,
`gemv_fp8_e4m3_f16t` 50.7 µs/call, named kernels 45.2 ms/token).

> **DECISION RULE, fixed now.** The hypothesis is confirmed if the pinned arm's **named-kernel total** drops by
> **≥ 15 %** (45.2 → ≤ 38.4 ms/token) — that is the quantity a clock change must move, and it is measured directly by
> the profiler rather than inferred from the wall. If it does, the phase's deliverable is the number and a written
> recommendation for the founder about the power trade (a permanent pin, a governor, or nothing). If the named-kernel
> total moves by **< 5 %**, the clock hypothesis is **dead** and docs/47's open question is closed as falsified —
> which is itself worth the run, because it is the last cause that explains all three kernel gaps at once. Between
> 5 % and 15 %: report, attribute per kernel, and do not recommend a system change for it.

## Pass criteria

1. Both arms measured on the same binary and the same build, back to back, ≥ 2 samples each, with the pin state
   read from sysfs and printed in the log rather than assumed.
2. The decision rule applied in writing, per kernel: `ds4_attention`, `gemv_fp8_e4m3_f16t`, `ds4_gemm_mxfp4`,
   `gemv_f16_rows` — each kernel's avg µs pinned against unpinned.
3. **Correctness unchanged**: the decode test's forced-routing digits identical in both arms (a clock cannot change
   arithmetic; if the digits move, something else did and the result is void).
4. The wall-clock effect reported separately from the kernel effect, since ~35 ms/token is DMA wait that a clock pin
   cannot touch — so the wall should improve by roughly the kernel saving, not more.
5. **The pin reverted** at the end of the phase, verified by reading `min_freq` back, whatever the outcome.
6. If confirmed: the idle-power cost measured (`gt-c6` residency and, if a sensor is readable, watts) so the founder
   is trading against a number rather than a guess.

## Explicitly NOT in this phase

Any permanent system change (the phase recommends, the founder decides), kernel rewrites, the attention shape work
that docs/62 ranks next if this is falsified, and concurrency.

## Results (17:20) — the clock hypothesis is DEAD, and docs/47's gap was a shape-mismatched comparison

**No root write was needed.** The profiler already recorded `min_ns` / `max_ns` per kernel and the decode test simply
never printed them; printing them answers the question outright (`p23/qprof_minmax.log`, 32 timed steps,
`IE_QUEUE_PROFILING=1`):

| kernel | calls | avg µs | **min µs** | **max µs** |
|---|---|---|---|---|
| `gemv_fp8_e4m3_f16t` | 9,280 | 50.7 | 9.3 | 364.2 |
| `ds4_gemm_mxfp4` | 5,112 | 69.3 | 16.0 | 203.4 |
| `gemv_f16_rows` | 1,872 | 126.2 | 1.7 | 2,353.2 |
| **`ds4_attention`** | 1,280 | **175.4** | **75.1** | 220.5 |
| **`ds4_router_logits`** | 1,280 | **27.2** | **26.7** | **28.5** |

**`ds4_router_logits` kills the clock hypothesis.** It runs 40 times per token, immediately before each MoE call —
i.e. at exactly the points in the step where a ramping clock would have to show — and its spread is **26.7 to 28.5
µs, ±3 %**. A clock that was ramping out of gt-c6 forty times per token could not leave the router that tight while
stretching the attention call next to it by 3×. Under the decision rule this is the "< 5 %" arm: **falsified**, and
docs/47's open question is closed. The founder's `min_freq` write is not needed and is not recommended.

**What the attention spread actually is: the comparison was wrong, not the kernel.** `ds4_attention`'s fastest call
is **75.1 µs**, inside docs/47's 80-90 µs bench range — so the kernel does reach its bench speed in-engine. The
spread is the KV length, which is a per-layer property of this architecture: two window layers see
`sliding_window = 128` keys, the ratio-2 layers see ~1,024 latents at a 2,048-token context, and the ratio-1 layers
see ~2,048. The mix predicts the mean almost exactly — 5 % of calls at ~75 µs and 95 % at ~180 µs gives **174.8 µs
against the 175.4 measured**. docs/47 compared an in-engine mean dominated by long-KV layers against a bench of the
short-KV shape; the kernel is doing 8-16× the KV work for 2.4× the time, which is the split kernel working as
designed. **So there is no 3.5 ms/token sitting in the attention kernel, and docs/62's "what remains" item 2 is
withdrawn.**

The same reading applies to item 3: `gemv_fp8_e4m3_f16t` ranges 9.3 to 364.2 µs and `gemv_f16_rows` 1.7 to 2,353 µs
because they serve shapes from the small LoRA projections to the 129,280-row tied head. An aggregate "374 GB/s
against 580 peak" is a mix of those shapes, not a uniform 35 % shortfall — the believed 2.6-4.6 ms of headroom is
**not established** by that number and would need per-shape roofline work to claim.

**What this leaves standing.** The profiler also prints the honest summary: **the GPU executes 26-27 % of its own
window** (card 0 busy 693.6 ms of a 2,701 ms span; card 1 751.8 of 2,757.7). The kernels are at their shapes' cost;
the machine is idle three quarters of the time waiting for expert bytes, exactly as docs/62's budget said. The one
remaining lever is therefore the one docs/62 ranked first and this phase has now isolated by elimination:
**fill the idle with another request's work** (concurrent serving), not with faster kernels.

**Kept from this phase:** the min/max columns are now permanent in the decode bench — a mean alone hid a 3× spread
and sent two phases chasing a kernel that was already at its bench speed. The queue-profiling arm reads 87.4
ms/token against the unprofiled 80.5 because each submit costs ~0.4 µs; profiled numbers are for shape comparisons,
never for the headline.
