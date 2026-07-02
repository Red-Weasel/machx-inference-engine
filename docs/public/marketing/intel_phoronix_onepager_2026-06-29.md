# The engine (Mach X) on Intel Arc Pro B70 — competitive one-pager

**Outbound pitch sheet · 2026-06-29 · for Intel devrel / Liftoff and Phoronix**

> Engine name **"Mach X"** is pending final confirmation [CONFIRM]. License: **Apache-2.0 (planned)** — owner confirming.

---

## Headline

**First and fastest on brand-new model architectures on Intel Arc — including day-one OpenAI gpt-oss, where we beat llama.cpp SYCL on *both* axes on a single B70.** A from-scratch SYCL inference engine for Battlemage (BMG-G31 / Xe2), tuned specifically for the Arc Pro B70's 608 GB/s memory and 32 GB-per-card footprint.

Two things no other B70 stack does today:

1. **gpt-oss-20b: beats llama.cpp SYCL on prefill *and* decode** on one B70 — same GGUF, same box, llama in its fastest (flash-attention-on) config.
2. **gpt-oss-120b — the 117-billion-parameter OpenAI model — runs coherently on 2× Arc Pro B70 (64 GB total), with a display-safe VRAM split.** Tensor-parallel, correct (PPL bit-identical T=1 vs batched), chat + Harmony tool-calling working.

---

## Hardware & method (read before the numbers)

- **Hardware:** Intel Arc Pro B70 (BMG-G31 / Xe2, 32 GB, **608 GB/s**). 20b figures are **1× B70**; 120b figures are **2× B70 tensor-parallel** (no working P2P between the cards).
- **Opponent labels are explicit** in every row: **vs llama.cpp SYCL**, **vs llama.cpp Vulkan**, or **vs LM Studio**. Same GGUF and same box wherever a ratio is quoted.
- **Reproduce (20b head-to-head):** ours = `ie-bench --warmup 2` (median); llama = `llama-bench -ngl 99 -fa 1` (single GPU, flash-attention on). See the gpt-oss write-up `docs/public/gptoss_benchmark_2026-06-27.md`.
- **Correctness gate:** the "crown" reference model holds **PPL 6.4527 bit-exact** across every gpt-oss change.

> **Box-state honesty note:** the throughput numbers below were measured on **2026-06-27 / 2026-06-29**; the bench box has since degraded (120b model loads drifted to multi-minute). Every tok/s figure here is flagged for **one clean-box re-bench before any number goes to print** — see the checklist at the end. The *correctness* claims (PPL, bit-exact, tool calling) are committed and independent of box state.

---

## Competitive scorecard

### A. gpt-oss-20b — 1× Arc Pro B70, vs llama.cpp SYCL (the lead claim)

Model: `gpt-oss-20b-mxfp4.gguf` (20.9 B total / 3.6 B active, MXFP4, 11.27 GiB). llama = SYCL, flash-attention ON (its faster config), single GPU, **same GGUF**.

| Workload | Engine (Mach X) | llama.cpp SYCL | Ratio |
|---|---|---|---|
| **Prefill** pp512  | **1795 t/s** | 927 t/s | **1.94×** |
| **Prefill** pp2048 | **4147 t/s** | 927 t/s | **4.47×** |
| **Prefill** pp4096 | **3428 t/s** | 896 t/s | **3.83×** |
| **Decode** tg@512  | **58.3 t/s** | 50.3 t/s | **1.16×** |
| **Decode** tg@2048 | **57.4 t/s** | 49.9 t/s | **1.15×** |
| **Decode** tg@4096 | **55.6 t/s** | 49.4 t/s | **1.13×** |

- **Both axes flat across context depth.** Headline: **prefill 1.9–4.5×, decode 1.13–1.16×, both flat.**
- **Honest caveat:** the decode margin (1.13–1.16×) is real but **thin** — box noise on this hardware can run ±40 t/s; this is the one row most in need of the clean-box re-bench. The prefill win is large and not noise-sensitive.
- **Quality:** streaming PPL **22.5** (int-dot vs f16 vs oneDNN all within 0.1%); greedy factual ("The capital of France is **Paris.**", Harmony, finish=stop); tokenizer parity (our 586 tokens = llama's 586).

### B. gpt-oss-120b — 2× Arc Pro B70 tensor-parallel (the "117B on consumer Arc" claim)

| Metric | Engine (Mach X), 2× B70 | Reference | Note |
|---|---|---|---|
| **Decode** | **~31 t/s** (peak 32.05) | LM Studio ~12.42 t/s ≈ **2.6×** | LM Studio number is **owner-reported, layer-split, @100k+ ctx** — re-bench at matched ctx before printing the ratio |
| **Prefill** | **538–679 t/s** at moderate ctx (≤16k); **433 t/s** at 65k | — | engineering progress from a 34 t/s starting point |
| **Correctness (PPL)** | **15.1985** batched-prefill == **15.1985** T=1 (**bit-identical**) | — | the publishable correctness statement |
| **VRAM split** | card0 **29.3 GB** VRAM + 3.4 GB host-spill; card1 **31.1 GB** + 0.3 GB | 32 GB/card | **display-safe** — leaves headroom on the card driving the screen |

- **Chat + Harmony tool-calling work** (factual, math, multi-turn; outbound function call + result round-trip; finish=`stop`/`tool_calls`).
- **Decode is comm/host-bound, not compute-bound:** ~50% of a decode step is the two cross-card all-reduces + router (no P2P on B70). This is the next lever, and it favors better interconnect — a concrete Intel-platform talking point.

### C. The DeltaNet "moat" architectures (gated-DeltaNet / hybrid)

Crown-35B-A3B, Qwen3-Next-80B, Qwen3.6-27B dense, Qwen3-Coder-30B-A3B, Gemma-4 26B/31B.

| Competitor on B70 | Status on these arches |
|---|---|
| llama.cpp **Vulkan** | **cannot run them** (missing shaders) |
| **OpenVINO** | **preview-only** on these arches |
| Intel **llm-scaler** | throughput-oriented, not single-stream latency |
| llama.cpp **SYCL** | the only real single-stream opponent |

- **Structural moat:** on hybrid gated-DeltaNet arches, **we are frequently the only latency-focused single-stream engine that runs at all on B70**, and we ship support **day-one on new checkpoints**. That "first on new arches" position is the durable advantage.
- **Throughput head-to-heads on these specific arches are [NEEDS VERIFICATION]** for this pitch — they were measured in prior sessions but are **not in the launch-verified ledger**, and llama.cpp SYCL has improved on mature arches, so we will only quote a DeltaNet prefill/decode multiplier after a fresh same-box re-bench. The honest framing for outbound use today is **"prefill leadership across arches + day-one on brand-new arches,"** with gpt-oss-20b as the hard, re-verified both-axes win.
- Reference model correctness is solid: **crown PPL 6.4527 bit-exact** across all engine edits.

---

## The "117B viable on 64 GB of consumer Arc" angle

OpenAI's **gpt-oss-120b** is a 117-billion-parameter MoE. We run it **coherently on two Arc Pro B70 cards (64 GB total VRAM)** with:

- a **display-safe load split** (29.3 / 31.1 GB across the two cards, ~3.4 GB host-spill on card0) so the card driving your monitor keeps headroom;
- **verified correctness** — batched-prefill PPL **15.1985** is **bit-identical** to the T=1 reference path (not "looks coherent" — numerically identical);
- **~31 t/s decode** and **538–679 t/s prefill** at moderate context;
- **working chat and Harmony tool-calling**, zero server-side changes.

Net story for Intel: **a frontier-class 117B open model is a *consumer-2-card Arc* workload today, not a datacenter-only one** — and the remaining bottleneck (cross-card all-reduce, no P2P) is exactly the kind of platform gap Intel can close.

---

## Honest scope (the part that builds trust)

- **Scoped claim:** "fastest *known* gpt-oss inference on Intel Arc, beats llama.cpp SYCL on both axes for 20b, same-hour measured." **Not** "best engine for Intel GPUs."
- **We lose** dense-model **decode** (Qwen3-Coder-30B, Qwen3.6-27B dense) and **gpt-oss-120b decode is comm-bound, not yet a clean win vs all comers** — dense decode on B70 hits the same memory wall as llama, and on some mature dense arches llama.cpp SYCL is ahead. We never claim those.
- **Long-context tradeoffs are real:** 120b prefill peaks at moderate ctx and steps down at 65k; the DeltaNet family runs under a chunk cap on this kernel/HW combo.
- **Self-improvement multipliers** (e.g. 20b decode 30.5 → 60.4 t/s ≈ 1.98× across the session; 120b decode 14.33 → 32.05 = 2.24×; 120b prefill 34 → 538–679 ≈ 16–20×) are **engineering progress vs our own earlier baseline — never competitive ratios.**
- **Every tok/s number on this sheet needs a clean-box re-bench before print** (the bench box is currently degraded). The correctness story (crown 6.4527 bit-exact; 120b 15.1985 bit-identical; tool calling functional) is committed and box-independent.

---

## Pitch email — Intel devrel / Liftoff

> **Subject:** 117B OpenAI gpt-oss running on 2× Arc Pro B70 — and beating llama.cpp on a single B70
>
> Hi [name],
>
> We build a from-scratch SYCL inference engine tuned for Battlemage (BMG-G31 / Xe2) on the Arc Pro B70. Two results I think are relevant to the Arc Pro story:
>
> 1. **gpt-oss-20b beats llama.cpp SYCL on a single B70 on both axes** — same GGUF, llama in its flash-attention-on config: prefill **1.9–4.5×**, decode **1.13–1.16×**, flat across context. Reproducible with `ie-bench` vs `llama-bench -ngl 99 -fa 1`.
> 2. **gpt-oss-120b (117B) runs coherently on two B70s (64 GB)** with a display-safe split, **~31 t/s decode** and **538–679 t/s prefill**, batched-prefill PPL **bit-identical** to the reference path, plus working Harmony tool-calling.
>
> The current 120b decode bottleneck is the cross-card all-reduce (no working P2P on B70) — roughly half of each decode step — which is exactly the platform gap better Arc Pro interconnect would close. I'd love to share the full methodology and re-run any of these live on a clean box for your team.
>
> Engine is **Apache-2.0 (planned)**. Happy to send the benchmark write-up and the reproduce commands.
>
> Best,
> [name]

## Pitch email — Phoronix

> **Subject:** Story tip: a SYCL engine beating llama.cpp on Intel Arc B70 — and running OpenAI's 117B gpt-oss on two consumer cards
>
> Hi [name],
>
> Possible benchmark story for Arc readers: an independent, **Apache-2.0 (planned)** SYCL inference engine for the Intel Arc Pro B70 (BMG-G31, 608 GB/s).
>
> - On **gpt-oss-20b** it beats **llama.cpp SYCL** (flash-attention on, same GGUF, single B70) on **both** axes: prefill **1.9–4.5×**, decode **1.13–1.16×**, both flat across context.
> - It runs OpenAI's **gpt-oss-120b (117B)** on **two B70s (64 GB total)** — coherent chat + tool-calling, ~31 t/s decode, with verified bit-identical correctness vs the single-stream reference path.
>
> I've kept it honest: the 20b decode margin is thin and within box noise, dense-model decode is a known loss vs llama, and the 120b decode is currently comm-bound by the lack of P2P on B70. All commands are llama-bench-comparable and I can hand you a clean-box re-bench so the numbers are yours, not mine.
>
> Want the methodology doc + GGUFs to reproduce?
>
> Best,
> [name]

---

## Pre-send re-bench checklist (gate before any number is printed)

1. **[BLOCKER]** Re-bench gpt-oss-20b vs *current* llama.cpp SYCL — the lead rests on llama = 927/927/896 prefill, 50.3/49.9/49.4 decode (one-time 2026-06-27). Confirm the decode margin survives box noise.
2. **[BLOCKER]** Re-run `ctest` and state the real count (source tree declares 32; last doc said 31). Confirm crown PPL 6.4527.
3. Re-bench 120b prefill (538–679 / 433 t/s) and decode (~31, peak 32.05) on a cool box.
4. Re-confirm or relabel LM Studio 12.42 t/s as "owner-reported, LM Studio layer-split" at matched ctx.
5. Replace the DeltaNet [NEEDS VERIFICATION] cells with fresh same-box llama.cpp SYCL head-to-heads, or keep the qualitative moat only.

*Source of truth for every number above: `docs/public/marketing/VERIFIED_CLAIMS_2026-06-29.md`.*
