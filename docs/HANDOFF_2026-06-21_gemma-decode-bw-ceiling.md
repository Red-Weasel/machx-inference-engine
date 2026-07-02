# Handoff — 2026-06-21 — gemma4 decode: BW ceiling characterized, two wins banked

## TL;DR
gemma4-31B decode is **GPU-bandwidth-bound at its ceiling.** Proven by a roofline
diagnostic, not theory. Two measured, bit-exact wins shipped (`baba378`); every
other candidate lever was measured and is a **dead end**. The ~10% gap to current
llama-Vulkan (19.0) is not closable with the available levers — it is llama's
slightly higher sustained gemv BW efficiency, in diminishing-returns territory.

## What shipped (`baba378`, gemma4-only, crown 6.4527 bit-exact, ctest 30/30)
1. **NPWG 32→64** in `gemv_q4_0_soa_q8`. 64 sub-groups/WG measured best on 31B
   (swept 8/16/32/64; 64 > 32 ≈ 16 > 8). Env read hoisted to a read-once
   `soa_npwg()` (was a `getenv` in the 410×/token hot path ≈ 0.7% host overhead).
   `IE_GEMV_NPWG` overrides.
2. **`gemv_q4_0_soa_q8_multi`** — fused multi-bank decode GEMV. q/k/v (shared
   attn_norm input) and gate/up (shared ffn_norm input) now run as **one quant +
   one launch** instead of 3/2. WGs tile the concatenated column space, so the
   tiny GQA k/v columns ride a full grid instead of under-sized launches. Zero
   extra VRAM (separate banks, single launch). Bit-identical token stream.
   Opt-out `IE_GEMMA4_NO_FUSE_QKV`.

Combined: **~+2–3%** (16.9 → ~17.2 tok/s, tg128, 31B, 1×B70). Modest but free.

## The diagnostic that settled it (do not re-litigate)
Method: added a BW-only kernel variant — identical global weight+scale loads, the
dp4a/FMA replaced by a cheap XOR fold (timing only). Compared sustained 128-step
tok/s, order-controlled.

| | tok/s | meaning |
|---|---|---|
| real kernel | 17.2 | — |
| ALU fully stripped | 18.7 | **+8.6% is the entire ALU headroom** |

⇒ decode GEMV is **~91% memory-bound**. The warm-step gemv already hits ~75–80%
of the 608 GB/s peak (the kernel's design band). There is almost nothing to gain
from arithmetic or layout cleverness.

### Confirmed DEAD ENDS (all measured, all neutral — stop chasing these)
- **Interleaved-column SoA repack** (the prior session's headline plan): premise
  was "32 columns sit K/2 apart = scattered reads." FALSE — each sub-group already
  issues a fully-coalesced 256 B SIMD load; the 32 sub-groups are independent
  coalesced streams (good MLP). BW is already near the warm ceiling. A multi-day
  repack would chase the ~9% ALU envelope at best. **Do not build it.**
- **Native fp16 in the gemv hot loop** — IGC already lowers it optimally (prior).
- **Per-layer `q.wait()` removal** — `IE_GEMMA4_NO_LAYER_WAIT` gates all 3
  dense-path waits; measured **neutral** (in-order queue already pipelines host
  submit under GPU compute). Confirms the prior session's finding. Left gated.
- **QKV/gate-up launch fusion** — shipped (bit-exact, cleaner) but only **+0.7%**.
  This is itself proof that launch/inter-kernel overhead is well-hidden: cutting
  ~6 launches/layer barely moved the needle ⇒ the ~1,400 launches/token are
  pipelined, so SYCL-graph record/replay would also not help. We are GPU-bound.
- **uint4 vectorized weight load** — tried, bit-identical, box had drifted (swap
  1 GB) so no clean signal; reverted unmeasured (likely neutral, same as fp16).

### NOT throttling
GPU pinned at **2800 MHz** (max boost) throughout a 400-step decode — the
sustained-vs-warm gap is **not** thermal. The decode-rate decay over a long run
(400 steps → 13.7 tok/s) is **attention growing O(seq_len)**, not the GEMV.

## Open lever (only one left, secondary value)
- **lm_head `gemv_q6_soa`** runs at ~57% BW (known headroom to ~80%, per memory)
  and is ~6–7% of the decode step. Pushing it to 80% ≈ +1–1.5% decode. BUT this
  kernel is **shared with crown + dense lm_head** → touching it risks the 6.4527
  gate. Defer unless the gemma decode number becomes strategically critical.
- Attention: SWA layers run **full** attention (`full_attention_gemma`, no window
  arg). Correct only for ctx ≤ sliding_window (1024); irrelevant at tg128 (ctx<
  150) but it both caps correctness and inflates cost past 1024. A windowed
  attention op would help long-ctx decode (and is a prerequisite for >1024 ctx).

## Strategic read (honest)
gemma4-31B decode is at its BW ceiling; ~10% behind llama-Vulkan and that gap is
not worth a multi-week kernel campaign for a model where our moat is elsewhere.
**Publishing should keep leading with the DeltaNet arches** (crown/80B — llama
literally cannot run them) and the engineering wins (31B fits one card, auto GPU
planner, multi-GPU). Treat gemma as "competitive, at its bandwidth ceiling."

## System state
Box drifted during the session (many back-to-back 20 GB loads → swap 1 GB, heat
soak; warmup tok/s drifted 17.4→16.7). The committed numbers were taken
order-controlled but the absolute values are ±~3% box-state noise. **Re-bench the
final 31B decode on a freshly rebooted box** (swap 0, GPU idle 400 MHz) for the
publishing number. Wins are bit-exact so correctness is independent of box state.
