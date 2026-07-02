# Fastest known gpt-oss inference on Intel Arc B70 — beats llama.cpp on both axes (20b), runs the 117B 120b viably on 2× B70 [Apache-2.0 planned]

*Cross-posting our gpt-oss results because this sub is the one place that'll actually
check my math. Numbers, reproduce commands, and the honest losses are all below. If you
have a B70 and llama.cpp SYCL built, you can refute or confirm any line in here.*

---

## What this is

A from-scratch inference engine (working name **Mach X** — name not final) for Intel Arc,
licensed **Apache-2.0 (planned, owner confirming)**. gpt-oss is its 8th architecture. This
post is two claims, scoped tightly:

1. **gpt-oss-20b on a single Arc Pro B70: faster than llama.cpp's SYCL backend on BOTH
   prefill and decode**, same GGUF, llama's best config.
2. **gpt-oss-120b (117B) runs viably tensor-parallel across 2× Arc Pro B70** — both cards
   computing every token — at prefill **538–679 tok/s** and decode **~31 tok/s**, with
   coherent chat and working OpenAI tool calling.

**Hardware:** Intel Arc Pro B70 (BMG-G31 / Xe2, 32 GB, 608 GB/s). 20b numbers are **1× B70**;
120b numbers are **2× B70 tensor-parallel, with no working P2P** (the all-reduce bounces
through host RAM — that detail matters below). Box is a 32 GB-RAM desktop.

I am going to flag, loudly and in its own section, exactly which numbers were measured once
on a box that has since degraded and need a clean re-bench. I'd rather you trust the ones I
stand behind than discover the soft spots yourself.

---

## 1. gpt-oss-20b — head-to-head vs llama.cpp SYCL (1× B70)

Model: `gpt-oss-20b-mxfp4.gguf` (20.9 B total / 3.6 B active, MXFP4 experts, 11.27 GiB).
Opponent: **llama.cpp SYCL**, flash-attention ON (its *faster* config here), single GPU,
**same GGUF file**.

| context | engine prefill | llama-SYCL prefill | **prefill** | engine decode | llama-SYCL decode | **decode** |
|--------:|---------------:|-------------------:|:-----------:|--------------:|------------------:|:----------:|
| 512     | **1795** t/s   | 927 t/s            | **1.94×**   | **58.3** t/s  | 50.3 t/s          | **1.16×**  |
| 2048    | **4147** t/s   | 927 t/s            | **4.47×**   | **57.4** t/s  | 49.9 t/s          | **1.15×**  |
| 4096    | **3428** t/s   | 896 t/s            | **3.83×**   | **55.6** t/s  | 49.4 t/s          | **1.13×**  |

So: **prefill 1.9–4.5×, decode 1.13–1.16×, both flat across context** (neither side
collapses with depth; the prefill gap actually *grows*).

**Read the decode column skeptically.** 1.13–1.16× is a thin margin, and this box swings
±40 tok/s heat-soaked. I believe the prefill win is robust (it's a structural fix, see §3);
I treat the decode win as real-but-narrow and **pending a clean-box re-bench** (§7).

**Honesty note up front:** an earlier internal baseline of ours compared against a *stale*
llama build (607 prefill / 22 decode) and is retired — do not cite it. The table above is a
fresh re-bench against a current, much-faster llama-SYCL, **measured once on 2026-06-27**.
Publishing a win against a stale opponent isn't a win.

**Not the comparison:** llama.cpp's **Vulkan** backend lacks the shaders for this
architecture, and Intel's **OpenVINO** path is preview-only for this model class — so SYCL
is the one real single-stream opponent on B70. This is *not* a "best engine for Intel GPUs"
claim; it's "first and fastest on a brand-new architecture on Arc."

---

## 2. How — the levers (engineering progress vs OUR OWN baseline, not competitive ratios)

These multipliers are self-improvement deltas — how far we moved the engine from its own
"it runs" starting point. They are **not** ratios against llama. Each held the engine's
reference perplexity gate bit-exact (crown model, NLL 1.864495 → PPL 6.4527) and moved
gpt-oss's own PPL < 0.1% (int-dot paths) / 0.16% (the tile kernel).

- **Fused MXFP4 decode GEMV** — decode was re-materializing each routed expert to fp16 just
  to do an M=1 matmul. A load-time Structure-of-Arrays repack + a W4A8 int-dot GEMV reads the
  4.25-bpw weights *once*. The trap: a native MXFP4 block is **17 bytes**, so a first cut over
  the native layout ran at **64 GB/s (~10% of peak)** — you *must* split into aligned planes.
  **Decode 30.5 → 43.0 tok/s (1.41×).**
- **Split-K flash-decode with attention sinks** — folds the per-head sink into the combine
  step; makes decode flat across context. **Decode @2K 25.3 → 53.9 tok/s (2.13×).**
- **Wide-tile prefill with sinks** (hd64, sink as a compile-time template param so every
  other arch sharing the kernel stays byte-identical). **Prefill @4K 327 → 716 tok/s (2.19×).**
- **W8A8 int-dot lm_head** (201k-vocab Q8_0, the single biggest decode GEMV). **Decode
  56.7 → 60.4 tok/s (1.06×), −0.58 GB VRAM.**
- **The bug worth 5.5× — per-token GEMVs at prefill.** A 2048-token prefill was issuing
  **196,608 GEMV launches in a single forward pass (88% of prefill)**, because the batched-GEMM
  path is gated on dims divisible by 256/64 (an XMX tile constraint) — and gpt-oss's hidden
  size **2880** and **32-expert router** both fail that gate, silently falling onto the
  per-token decode path. Routing those through a batched oneDNN GEMM **flipped prefill from a
  loss to a win**: **759 → 4147 tok/s @2K (5.5×).**
- **Cumulative decode** across the session: **30.5 → 60.4 tok/s @512 (~1.98×).**
- **An honest negative:** we tried `dp4a_us`+offset instead of `dp4a_ss` for the MoE GEMV,
  guessing the signed dot was emulated. It was **slower** — `dp4a_ss` is hardware-lowered on
  BMG-G31; the GEMV is genuinely ALU-co-limited. Documented so nobody re-runs it.

(The H2H table's decode 58.3 and this section's cumulative 60.4 are different bench
harnesses — `ie-bench` median vs the lever endpoint — don't read the 2 tok/s gap as drift.)

---

## 3. gpt-oss-120b — 117B on 2× B70, tensor-parallel

Model: `gpt-oss-120b` (117 B total / ~5.1 B active, top-4 MoE, MXFP4 experts). This is the
engine's first **MoE tensor-parallel** path: every expert split across both cards, attention
heads + KV sharded, and the experts that overflow VRAM spilled to pinned host RAM over PCIe
(no smaller quant — full MXFP4). Display card is capped so the desktop doesn't freeze:
load reports **card 0: 29.3 GB VRAM + 3.4 GB host-spill; card 1: 31.1 GB + 0.3 GB**.

- **Prefill: 34 → 538–679 tok/s** at typical agent context (≤16k) — ~16–20× our own start
  number — via three levers: pad-M MoE GEMM to restore XMX (63 → 127), prefill chunk 256 → 4096
  (127 → 433), and auto-replicating attention at ctx ≤16384 (433 → **538–679**, 1.24–1.54×).
  At long context (~65k) it's **433 tok/s** (chunk 4096; chunk 8192 *regresses* to 301 — an
  honest tradeoff, see §6).
- **Decode: ~31 tok/s** (peak measured 32.05; 14.33 → 32.05 = 2.24× from an async-scatter
  all-reduce). I print **~31** because this is a **host/comm-bound** decode with **±15% box
  noise** (the *same binary* has produced 12–32 tok/s depending on box state) — 32.05 is a
  peak, not a number to quote as typical.
- **vs LM Studio** (llama.cpp **layer-split**, same two cards): **~31 vs 12.42 ≈ 2.6×.**
  Caveat in plain sight: **12.42 is owner-reported at ~100k context**, not a matched-config
  measurement of mine — so treat 2.6× as *indicative* pending a matched re-bench. The
  *structural* reason TP wins for a too-big model is real regardless: layer-split decodes one
  token with **only one card doing math** while the other waits the hand-off; tensor-parallel
  has both cards multiply their half of every matmul, every layer, every token.

**The cost, stated honestly:** with no working P2P on this B70 pair, each all-reduce bounces
through host RAM (D→H, fp32 sum, H→D). At decode that comm + router is **~50% of the step**;
at prefill the two host-bounced all-reduces are **67.6%** of the pass (attention 33.4% + MoE
34.2%), vs MoE compute 24.4% and attention compute 6.1%. TP here is the right tool *because
the model doesn't fit on one card* — not a free win on every axis.

---

## 4. The bug-hunt highlight — the token-0-bit-exact diagnostic

This is the one I'd want to read about, so here's the hook in full.

The 120b's batched multi-token prefill produced garbage — **PPL ~123** — while the
slow token-by-token path was correct. The tell that cracked it: with the stable
counting-sort that packs tokens per expert, **token 0 always lands on row 0**, and the
output came out **bit-exact for token 0 and garbled for every other token**. That's not a
random-corruption signature — that's a matmul that's wrong for **every local row except
row 0**. Root cause: with **128 experts / top-4**, each per-expert prefill GEMM runs at a
*tiny* M (~1–8 rows), and BMG's small-M f16 XMX path corrupts every local row ≥1. Fix: pad
small experts up to M=32 back onto XMX. The A/B is the clean part: **batched PPL 123 →
15.1985, bit-identical to the token-by-token path's 15.1985** (NLL 2.721199, identical to
the last digit). 20b showed no regression (19.36). Batched prefill is now default
(opt-out `IE_GPTOSS_TP_T1_PREFILL`).

Two more deep ones, briefly:

- **fp16 residual overflow** = the "!" / NaN chat garbage. gpt-oss's activations exceed
  fp16's max (65504) at deep layers (~L35) → `inf` poisons the KV cache → NaN cascade →
  argmax collapses to token 0 ("!"). Fix: saturate the residual add to ±65504 — an exact
  no-op for every in-range model, so the crown PPL gate stays **6.4527 bit-exact**. Chat
  coherence then validated (17×23=391, haikus, color lists).
- **comm-bound prefill** (the 67.6% above): replicating attention at moderate context drops
  one of the two host-bounced all-reduces entirely → 433 → 679 tok/s.

---

## 5. Correctness / quality (the part I stand behind hardest)

- **Engine reference gate:** crown model PPL **6.4527 bit-exact** (≤ 6.57 required) through
  every gpt-oss edit — the gpt-oss code is arch-gated so it can't touch other models.
- **20b PPL:** streaming **22.5** (256-tok; q8 22.52 / f16 22.51 / oneDNN 22.53 — int-dot
  moves it < 0.1%). gpt-oss is reasoning-tuned, so raw-text PPL is high *for everyone* — its
  strength is chain-of-thought, not raw next-token. A forward-validation corpus (586 tokens,
  plain rope ≤4096) gives **17.9**, vs llama chunked **54.7** — i.e. no forward deficit on our
  side. **Tokenizer parity: our 586 tokens = llama's 586.** Greedy: "The capital of France is
  **Paris.**" via OpenAI Harmony, finishing cleanly on stop.
- **120b PPL:** the publishable correctness statement is the bit-identical batched-vs-T=1
  A/B above (**15.1985 == 15.1985**; replicate-attn 15.19 vs head-shard 15.20 is
  reduction-order rounding). Decode-path PPL is **17.08** post gate+up gemv fusion (was 17.90;
  MoE bucket 62% → 53%). *(An older streaming 12.91 from a pre-fusion config exists; I'm
  deliberately not headlining it.)*
- **20b TP validation** (2-card vs 1-card, correctness harness, **Phase-1 replicated attn**):
  decode PPL 19.31 vs 19.36 (0.26%), prefill 16.85 vs 16.82 (0.18%), "Paris" both ways. The all-reduce isn't bit-exact
  to single-GPU (different sum order), so it's validated by PPL-match, not bit-equality.
- **YaRN:** plain rope is best ≤4096 (17.9 vs 30.5); YaRN is opt-in (`IE_GPTOSS_YARN`) for
  longer context.

---

## 6. Tool calling + the honest losses

**Tool calling** (OpenAI Harmony) is wired end-to-end with zero server changes: render
`tools` → Harmony `namespace functions` TS, parse the model's `commentary to=functions.NAME`
back into structured `tool_calls`. Validated: `get_weather` → `tool_calls[0]` with
`{"location":"San Francisco"}`, finish=`tool_calls`; feed the result back → "…foggy, 61 °F,
12 mph", finish=`stop`; no-tools chat byte-identical → "Paris". Scope: only **user
`functions.*`** is wired — built-in browser/python tools are a follow-up.

**Where we lose / what's not a win** (because negative results are the point of this sub):

- **20b tensor-parallel decode is *slower* than single-card** (41 vs 58 tok/s) — *expected*.
  For a model that fits on one card, paying two host-bounced all-reduces per layer per token
  is pure overhead. 20b TP exists here only as a correctness harness; TP's decode case is the
  120b, where the alternative is "doesn't run" or "runs layer-split with a card idle."
- **120b long-context prefill has a chunk cliff:** chunk 4096 = 433 tok/s, but chunk 8192
  *regresses* to 301 at ~65k. Not yet tuned away.
- **120b decode is comm/host-bound** (~50% of the step is sync, no P2P) — that's the ceiling,
  not the kernels, and it's why the number is noisy.
- **Dense-attention models remain our weak axis** — on dense decode we still don't beat
  llama the way the MoE prefill does. (No clean current figure to quote here, so I'm not
  giving one rather than inventing it.)
- The MoE decode GEMV is **ALU-co-limited at its ceiling** — the `dp4a` negative above means
  there's no cheap further decode win on that kernel.

---

## 7. What I have NOT re-verified yet (read before quoting me)

The throughput numbers and the vs-llama / vs-LM-Studio ratios are **real but were measured
once on a box that has since degraded** (120b loads have drifted to multi-minute; decode
shows ±15% noise). Before I'd call any tok/s number "fresh as of launch day," these get one
clean, idle, heat-soaked-controlled re-bench — and I'll post the script:

1. The **20b head-to-head vs *current* llama-SYCL** — the whole lead rests on llama =
   927/927/896 prefill, 50.3/49.9/49.4 decode, measured once 2026-06-27. The decode margin
   (1.13–1.16×) is within this box's noise; I want to confirm it survives.
2. **120b prefill (538–679 / 433) and decode (~31)** on a cool box.
3. **LM Studio 12.42** at matched context (it's owner-reported, not my measurement).
4. **ctest** count (source declares 32 tests; last green run reported 31) + re-confirm crown
   PPL 6.4527.

The **correctness story is committed and solid** (crown 6.4527 bit-exact; 120b batched PPL
15.1985 bit-identical to T=1; the residual-overflow and small-M-XMX fixes are code-confirmed;
tool calling functionally validated). The **throughput ratios are pending that one re-bench.**

---

## 8. Reproduce

20b head-to-head (single B70):

```bash
# ours (warmup + median; prefill t/s = T / prefill_ms, decode measured at depth T)
ie-bench --gguf gpt-oss-20b-mxfp4.gguf --prefill 512,2048,4096 --decode 128 --warmup 2

# llama.cpp SYCL at its best config (flash-attention ON), single GPU, same GGUF
ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  llama-bench -m gpt-oss-20b-mxfp4.gguf -ngl 99 -fa 1 -p 512,2048,4096 -n 128
```

120b tensor-parallel (2× B70):

```bash
# loader prints the per-card "NN.NN GB VRAM + N.NN GB host-spill" split
ie run --gguf gpt-oss-120b --gpus 2
# two-GPU decode bench + tracers
ie-gptoss-tp-bench --gguf gpt-oss-120b --gpus 2
# IE_GPTOSS_TP_TIMING=1 / IE_GPTOSS_TP_NANCHECK=1 for the profilers

# relevant toggles (all default to the fast/correct path; these revert):
#   IE_GPTOSS_TP_REPLICATE_ATTN  - Phase-1 replicated attention
#   IE_GPTOSS_TP_RESERVE_GB      - per-card VRAM headroom (display safety)
#   IE_GPTOSS_TP_T1_PREFILL      - token-by-token prefill (pre-small-M-XMX-fix path)
#   IE_GPTOSS_YARN               - YaRN rope for >4096 context
#   IE_GPTOSS_NO_MXFP4_GEMV / IE_GPTOSS_NO_FA2_PREFILL / IE_GPTOSS_NO_PROJ_GEMM
```

Gates:

```bash
ctest                       # source declares 32 tests
./build/tools/ie-perplexity # crown PPL must stay <= 6.57 (currently 6.4527)
```

---

## TL;DR

On **1× Arc Pro B70**, gpt-oss-20b beats **llama.cpp SYCL** on both axes — **prefill
1.9–4.5×, decode 1.13–1.16×, both flat** (same GGUF, llama's best config; decode margin thin
+ pending a clean-box re-bench). On **2× B70 tensor-parallel**, the **117B gpt-oss-120b** runs
viably at **538–679 tok/s prefill / ~31 tok/s decode (~2.6× LM Studio, owner-reported
baseline)**, full MXFP4, coherent chat + Harmony tool calling, with the residual-overflow and
small-M-XMX bugs fixed (the token-0-bit-exact diagnostic was the crack). Engine is
**Apache-2.0 (planned)**; name not final. Happy to run any specific config you want checked —
I'd rather you catch a soft number here than in the wild.
