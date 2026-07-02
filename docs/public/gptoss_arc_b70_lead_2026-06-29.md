# gpt-oss-20b on Intel Arc Pro B70 — day-one, and faster than llama.cpp on both axes

*How the engine (Mach X) went from "gpt-oss runs" to "gpt-oss beats llama.cpp's SYCL
backend on prefill **and** decode" on a single Arc Pro B70 — including the one profiler
finding that was worth a 5.5× prefill speedup, and the dead end we keep on the record.*

**Hardware:** 1× Intel Arc Pro B70 (BMG-G31 / Xe2, 32 GB, 608 GB/s).
**Model:** OpenAI `gpt-oss-20b-mxfp4.gguf` (20.9 B params / 3.6 B active, MXFP4 experts,
11.27 GiB).
**Opponent:** llama.cpp **SYCL** backend, same GGUF, same box, run in *its* fastest
single-stream configuration (flash-attention on).
**Measured:** 2026-06-27, same hour, same machine. License: **Apache-2.0 (planned)**.
Engine name "Mach X" is pending confirmation — referred to below as *the engine*.

---

## TL;DR

On one Arc Pro B70, the engine runs OpenAI's **gpt-oss-20b** and beats llama.cpp's SYCL
backend on **both** axes:

- **Prefill: 1.9–4.5× faster**, and the gap *grows* with context (4.5× at 2K tokens).
- **Decode: 1.13–1.16× faster**, and **flat across context** where the naive cost would
  otherwise climb.

This is the cleanest competitive result we have: a brand-new architecture, beaten against
the one real single-stream opponent on Arc, in that opponent's own best config, on the
same GGUF, same hour. The scope is deliberately narrow — **"fastest known gpt-oss on
Intel Arc, both axes, single card,"** not "best engine for Intel GPUs." The honest caveats
(thin decode margin, a one-time opponent snapshot, our dense-model decode loss) are in the
scope section, not buried.

---

## The numbers (1× B70, same GGUF)

| context | engine prefill | llama prefill | **prefill** | engine decode | llama decode | **decode** |
|--------:|---------------:|--------------:|:-----------:|--------------:|-------------:|:----------:|
| 512     | **1795** t/s   | 927 t/s ¹     | **1.94×**   | **58.3** t/s  | 50.3 t/s ¹   | **1.16×**  |
| 2048    | **4147** t/s   | 927 t/s ¹     | **4.47×**   | **57.4** t/s  | 49.9 t/s ¹   | **1.15×**  |
| 4096    | **3428** t/s   | 896 t/s ¹     | **3.83×**   | **55.6** t/s  | 49.4 t/s ¹   | **1.13×**  |

Engine decode is reported at depth = the prefill length; llama decode uses `-d {512,2048,4096}`
to match. Both sides stay flat across depth — neither collapses.

**Method (fairness first):**
- **Engine:** `ie-bench --gguf gpt-oss-20b-mxfp4.gguf --prefill T --decode N --warmup 2`
  (warmup + median; prefill t/s = T / prefill_ms; decode measured at depth T).
- **llama (its best):** `ONEAPI_DEVICE_SELECTOR=level_zero:0 llama-bench -m
  gpt-oss-20b-mxfp4.gguf -ngl 99 -fa 1 -p T -n 0` for prefill and `… -n 128 -d {depth}`
  for depth-matched decode. **Flash-attention ON** is llama's *faster* single-stream config
  on this box, so that is what we compare against.
- Single GPU on both sides (`level_zero:0`; the 20b fits one 32 GB card). Same GGUF file.

**An honesty note on baselines.** An earlier internal comparison used a *stale* llama build
(roughly 607 t/s prefill / 22 t/s decode). The table above is a fresh re-bench against a
*current* llama-SYCL, which is far faster than that old number — and the engine still wins
both axes. Publishing a win against a stale opponent is not a win.

> **¹ Measurement provenance / re-bench status.** The llama baseline (927/927/896 t/s
> prefill, 50.3/49.9/49.4 t/s decode) was measured **once, on 2026-06-27**, against the
> local llama.cpp `build-sycl` checkout at commit `fdc3db9b6` (2026-06-11), with the
> commands above. It is the single load-bearing number under the whole "both axes" claim.
> Per our pre-publish checklist it is flagged **NEEDS CLEAN-BOX RE-BENCH**: this box has
> since drifted (heat-soaked GPU clocks swing ±40 t/s, which is wider than the 1.13–1.16×
> decode margin), and upstream llama may have moved since the snapshot. Re-confirm the
> baseline — and the exact `build-sycl` commit — on a cool, idle box before print.

---

## What gpt-oss actually is

gpt-oss is the engine's **8th architecture**. Under the hood it is a top-4 MoE with three
pieces most transformers don't have, each of which needed its own kernel work:

- **Attention sinks** — a per-head learned scalar that enters the softmax *denominator* as
  a virtual, always-attended key with no value. Attention mass "leaks out" of the real
  keys; get it wrong and every long-context completion drifts.
- **Alternating sliding-window / dense layers** — even layers attend a fixed local window,
  odd layers attend the full context. Two attention kernels, not one.
- **MXFP4 experts** — block-quantized 4-bit weights: a shared exponent byte plus 16
  nibble-bytes (32 FP4 weights) packed into **17 bytes per block**, decoded through a small
  signed mapping. That 17-byte stride turns out to be the single most important number in
  the whole optimization story (below).

Getting it *numerically correct* was step one — validated against a reference forward pass
and the engine's perplexity gate. Making it *fast* was a sequence of kernel-level levers.

---

## How we made it fast — seven engineering notes

A precise count, because an earlier draft mis-stated it: **five of these are speedups; the
sixth is a dead end we keep documented so nobody re-runs it; the seventh is the correctness
gate every one of the first five had to clear.** Throughput deltas inside this section are
**self-improvement vs our own earlier baseline** (engineering progress), not competitive
ratios — the only competitive numbers are in the table above.

### 1. The fused MXFP4 decode GEMV — and the 17-byte trap

At decode the MoE was re-materializing each routed expert's weights to fp16 just to
multiply by a single activation vector (an M=1 GEMM) — many times the real weight size in
memory traffic, every token. The fix is a fused GEMV that reads the packed 4-bit weights
**once**, as a W4A8 integer dot product.

The trap is that 17-byte block stride. A 17-byte stride is coalesced for *no* cross-lane
access pattern — a first cut over the native layout ran at **~64 GB/s, about a tenth of
peak**. The fix is a load-time **Structure-of-Arrays repack**: split each expert into an
aligned nibble plane and an aligned exponent plane, so 16 lanes striding 32-element blocks
read consecutive aligned chunks. A second subtlety: the canonical CUDA kernel maps nibbles
to int8 with a hardware byte-permute that is *software-emulated* on Intel Xe, so we decode
the FP4 value with branchless register bit-math instead — no memory lookup table, no
emulated permute. **Decode 30.5 → 43.0 t/s (1.41×)**, perplexity-neutral (q8 22.52 /
f16 22.51 / oneDNN 22.53).

### 2. Split-K flash-decode with sinks — killing the long-context collapse

Naive decode attention does a serial reduction over the whole KV cache: it is
latency-bound and *collapses* as context grows (down to 25.3 t/s at 2K). We route the
full-attention (odd) layers through a split-K FA-2 flash-decode kernel and fold the
per-head sink in **once**, after the per-chunk partials are merged — the sink contributes
to the denominator but never to the value accumulator, exactly the "mass leaks out"
semantics, now in a parallel kernel. Decode goes **flat** across context:
**@2K 25.3 → 53.9 t/s (2.13×)**, ~1.31× at 512. Perplexity 22.54 vs 22.52 — within noise.

### 3. Wide-tile prefill with sinks

The naive prefill attention sagged as context grew (327 t/s at 4K). The same sink fold,
this time at the end of a query-row tile, dropped into the proven wide-tile prefill kernel
(which reads K/V once per tile instead of once per query). The sink was threaded as a
*compile-time template parameter* so every other model that shares the kernel compiles to
byte-identical code. **Prefill @4K 327 → 716 t/s (2.19×)**, ~1.37× at 2K. Perplexity moves
+0.16% (floating-point reduction order, the same band as our other production tile kernels;
the crown reference stays bit-exact).

### 4. A W8A8 lm_head

The output projection (lm_head) is the single biggest decode GEMV. Keeping it in Q8_0 and
running an int-dot GEMV — instead of expanding to fp16 — halves the weight read and saves
**0.58 GB** of VRAM. **Decode 56.7 → 60.4 t/s (1.06×)**, perplexity +0.09%. The same
"Q8 non-expert" pattern is a lever we expect to reuse to fit the 120B sibling.

### 5. The bug worth 5.5× — per-token GEMVs at prefill

This is the one that mattered most, and it was invisible until we profiled per kernel.

A prefill of 2048 tokens was issuing **196,608 GEMV launches in a single forward pass —
88% of prefill time.** The cause: the engine's batched-GEMM path is gated on dimensions
divisible by 256 (K) and 64 (N), an XMX matrix-tile constraint. gpt-oss's **hidden size is
2880** (2880 % 256 ≠ 0), so the attention q/k/v/o projections silently fell back to the
*decode* path — a per-row GEMV — looped once per token. Fix that, and the **32-expert
router** (N = 32, fails % 64) surfaced as the next-largest offender.

The fix is almost anticlimactic: route those prefill GEMMs through oneDNN directly, which
handles arbitrary dimensions (the MoE expert GEMM already called it with K = 2880). One
helper, two call sites. **Prefill 759 → 4147 t/s at 2K — a 5.5× self-speedup, and the
difference between *losing* prefill and winning it 4.5×.**

**The lesson generalizes:** always per-kernel-profile a new architecture for a per-token
GEMV explosion. A dimension gate written for one matmul backend can silently route a whole
class of weights onto the slow path.

### 6. An honest negative result

We hypothesized that the MoE GEMV's signed×signed dot product (`dp4a_ss`) might be
software-emulated on BMG-G31, and tried the unsigned `dp4a_us` form with an offset
correction. It was **slower**: `dp4a_ss` *is* hardware-lowered here, and the offset's extra
reductions just added work. The MoE GEMV is genuinely ALU-co-limited at its ceiling, not
held back by the dot. We keep this on the record so nobody re-runs it.

### 7. Correctness as a constraint, not an afterthought

Every speedup above held the engine's reference perplexity gate **bit-exact** — a separate
"crown" model at PPL **6.4527 (NLL 1.864495)** — and the regression test suite green
(31/31 as last run 2026-06-27; the source tree now declares 32 tests, so this count is
flagged for a fresh re-run before any "all green" claim). The int-dot decode paths
(W4A8 / W8A8) move gpt-oss's own perplexity by **< 0.1%**; the wide-tile prefill moves it
**0.16%**. Greedy factual completions stay correct through the OpenAI Harmony chat format:
*"The capital of France is **Paris**,"* finishing cleanly on the stop token.

**Cumulative effect (vs our own earlier baseline, not vs llama):** decode
**30.5 → 60.4 t/s at 512 (~1.98×)** across the session; prefill **759 → 4147 t/s at 2K
(5.5×)**.

(The head-to-head table's decode 58.3 and this section's cumulative 60.4 are different
bench harnesses — `ie-bench` median vs the lever endpoint — so don't read the ~2 t/s gap
as drift.)

---

## Quality

gpt-oss is a reasoning-tuned model, so **raw-text perplexity is high for everyone** — its
strength is chain-of-thought, not raw next-token text. With that framing:

- **Streaming PPL: 22.5** on a 256-token bench; the int-dot variants land at 22.51–22.53,
  i.e. within 0.1% of the fp16 path.
- **Forward-validation PPL: 17.9** on a 586-token corpus with plain RoPE at ≤4096 context —
  versus llama's chunked run on the same corpus at 54.7, i.e. **no forward deficit on our
  side**. (This 17.9 is a *different* corpus/depth than the 22.5 streaming number — they
  are not the same measurement and shouldn't be conflated.)
- **Tokenizer parity:** our tokenization matches llama's exactly (586 = 586 tokens on the
  validation text).
- **RoPE:** plain RoPE is best at ≤4096 (17.9 vs 30.5 with YaRN); YaRN is opt-in
  (`IE_GPTOSS_YARN`) for longer contexts.

---

## Honest scope and the losses

The claim is intentionally bounded. Where it doesn't hold, here is where:

- **The decode margin is thin.** 1.13–1.16× is real but slim, and this box's heat-soaked
  clock swings (±40 t/s) are *wider* than that margin. We treat the decode win as
  "confirmed on 2026-06-27, pending a clean-box re-bench," not as a settled headline. The
  prefill win (1.9–4.5×) is comfortably outside the noise.
- **One-time opponent snapshot.** The llama baseline is a single 2026-06-27 measurement
  (footnote 1). It is the load-bearing number, and it is explicitly queued for re-bench.
- **We lose on dense-model decode.** On dense architectures (e.g. Qwen3-Coder) our decode
  currently *trails* llama — the win described here is specific to gpt-oss's MoE + MXFP4
  shape, where the fused 4-bit GEMV and split-K flash-decode pay off. We are not claiming a
  blanket "faster on Arc."
- **Raw-text PPL is high (22.5).** Expected for a reasoning model, but worth stating plainly
  rather than hiding behind a chat demo.

Negative results build trust. None of the above changes the headline; all of it is part of
being skeptic-proof on r/LocalLLaMA or HN.

---

## Why this matters on Arc

For a brand-new architecture on Intel Arc, the competitive field is thin and worth labeling
precisely:

- **llama.cpp Vulkan** lacks the shaders this architecture needs — not a contender here.
- **Intel OpenVINO** is preview-only for this model class.
- **llama.cpp SYCL** is the one real single-stream opponent — and is exactly what the table
  above measures against.

Being **day-one on the architecture** *and* faster than that opponent on both prefill and
decode is the defensible position: **first, and fastest, on new architectures on Arc** —
measured the same hour, on the same card, on the same file.

---

## Reproduce

Same box, same GGUF, single B70:

```sh
# engine
ie-bench --gguf gpt-oss-20b-mxfp4.gguf --prefill 512,2048,4096 --decode 128 --warmup 2

# llama.cpp SYCL, its best single-stream config (flash-attention on)
ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  llama-bench -m gpt-oss-20b-mxfp4.gguf -ngl 99 -fa 1 -p 512,2048,4096 -n 0
ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  llama-bench -m gpt-oss-20b-mxfp4.gguf -ngl 99 -fa 1 -n 128 -d 512,2048,4096
```

llama build for the baseline: local llama.cpp `build-sycl` at commit `fdc3db9b6`
(2026-06-11) — re-confirm this commit and re-run on a cool, idle box before publishing
(footnote 1).

---

*Numbers in the head-to-head table were measured on 2026-06-27 and are flagged for a
clean-box re-bench; the correctness story (crown PPL 6.4527 bit-exact, < 0.1% int-dot PPL
drift, tokenizer parity) is committed and reproducible from the cited build. License:
Apache-2.0 (planned). Engine name "Mach X" pending confirmation.*
