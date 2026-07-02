# VERIFIED-CLAIMS LEDGER — gpt-oss launch (2026-06-29)

**Purpose.** Every public-facing number/claim about gpt-oss (20b + 120b) and the engine,
with the exact figure, the source (file:line or commit SHA), and a publish-readiness STATUS.
Built for the launch publication review. Be conservative: when a number drifted across the
session, the **current** value is recorded with the drift noted.

**Hardware referenced throughout:** Intel Arc Pro B70 (BMG-G31 / Xe2, 32 GB, 608 GB/s).
20b claims are **1× B70**; 120b claims are **2× B70 tensor-parallel, no working P2P**.

### STATUS legend
- **VERIFIED** = committed and measured/validated this session; reproducible from the cited
  commit; for correctness claims the crown PPL gate (6.4527) and/or a PPL A/B backs it.
- **NEEDS CLEAN-BOX RE-BENCH** = the number is real but was taken on a box that is now
  degraded (the §7 note: 120b loads drifted 67s→426s, "~9-min loads"; clean-box rule), OR it
  is a vs-llama / vs-LM-Studio ratio whose **opponent** may have moved since the one-time
  bench. Re-confirm on an idle, heat-soaked-controlled box before it goes in print.
- **FUNCTIONAL-VERIFIED** = behavior validated (not a throughput number); committed.
- **SUPERSEDED / DO-NOT-USE** = a retired number that must not appear in any artifact.

---

## A. gpt-oss-20b — single-card head-to-head vs llama.cpp SYCL (THE LEAD CLAIM)

Source table appears identically in three docs:
`docs/public/gptoss_benchmark_2026-06-27.md:25-29`,
`docs/public/gptoss_optimization_deepdive_2026-06-27.md:119-123`,
`docs/COMPETITIVE_SCORECARD_2026-06-25.md:18-19`. Model `gpt-oss-20b-mxfp4.gguf`
(20.9 B / 3.6 B active, 11.27 GiB). llama config = SYCL, flash-attention ON (its faster config), single GPU, same GGUF.

| Claim | Number | Source | STATUS |
|---|---|---|---|
| Prefill pp512 ie vs llama | **1795 vs 927 t/s = 1.94×** | benchmark_2026-06-27.md:27 | NEEDS CLEAN-BOX RE-BENCH — the **927 llama baseline is load-bearing**; measured once 2026-06-27; llama-SYCL has a documented history of improving (§ scorecard) |
| Prefill pp2048 ie vs llama | **4147 vs 927 t/s = 4.47×** | :28 | NEEDS CLEAN-BOX RE-BENCH (same 927 baseline; note llama is flat 927 at 512 and 2048 — re-confirm that flatness) |
| Prefill pp4096 ie vs llama | **3428 vs 896 t/s = 3.83×** | :29 | NEEDS CLEAN-BOX RE-BENCH |
| Decode tg@512 ie vs llama | **58.3 vs 50.3 t/s = 1.16×** | :27 | NEEDS CLEAN-BOX RE-BENCH (decode margin is the thin one — 1.13–1.16×; box noise ±40 t/s can erase it) |
| Decode tg@2048 ie vs llama | **57.4 vs 49.9 t/s = 1.15×** | :28 | NEEDS CLEAN-BOX RE-BENCH |
| Decode tg@4096 ie vs llama | **55.6 vs 49.4 t/s = 1.13×** | :29 | NEEDS CLEAN-BOX RE-BENCH |
| Headline summary | "prefill **1.9–4.5×**, decode **1.13–1.16×**, both flat across context" | benchmark TL;DR:14-16; deepdive:8 | NEEDS CLEAN-BOX RE-BENCH before print, otherwise internally consistent |

**Notes.** Both axes flat across depth (neither side collapses). ie measured with
`ie-bench --warmup 2` (median); llama with `llama-bench -ngl 99 -fa 1`. The docs themselves
carry an explicit honesty note that an *earlier* baseline was stale (see §K) and these are a
fresh re-bench — but "fresh as of 2026-06-27" is not "fresh as of publish day." The
publishing rule in `COMPETITIVE_SCORECARD_2026-06-25.md:36-40` is "re-bench EVERY claim vs a
current llama before publishing." This is the single most important re-bench gate in the doc.

---

## B. gpt-oss-20b — the seven optimization levers (self-improvement, 1× B70)

These are **vs our own earlier baseline** (engineering progress), not competitive ratios —
present them as such. All committed, all crown-gated bit-exact, all PPL-neutral unless noted.

| Lever | Number | Source | STATUS |
|---|---|---|---|
| 1. MXFP4 fused decode GEMV (SoA repack + W4A8 int-dot) | decode **30.5 → 43.0 t/s (1.41×)** | commit `86f9a54`; benchmark:60; deepdive:28-41 | VERIFIED (committed, PPL-neutral q8 22.52 / f16 22.51 / oneDNN 22.53) |
| 2. Split-K FA-2 decode + sink | decode @2K **25.3 → 53.9 t/s (2.13×)**, @512 1.31×, flat across ctx | plan §7:99-111; benchmark:61-64 | VERIFIED (committed; PPL 22.54 vs 22.52) |
| 3. Wide-tile prefill + sink (hd64, `SINK` template param) | prefill @4K **327 → 716 t/s (2.19×)**, @2K 1.37× | plan §7:89-98; benchmark:65-67 | VERIFIED (committed; PPL +0.16%, crown bit-exact) |
| 4. W8A8 int-dot lm_head (Q8_0 SoA, 201k vocab) | decode **56.7 → 60.4 t/s (1.06×)**, −0.58 GB VRAM | commit `f05718f`; benchmark:68-70 | VERIFIED (committed; PPL +0.09%) |
| 5. Batched prefill GEMMs + router (THE big find) | prefill **759 → 4147 t/s @2K = 5.5×** self-speedup | commits `2e9a452`/`5fc7d7e`; benchmark:71-77; deepdive:76-97 | VERIFIED (committed; flipped prefill from loss to win) |
| Cumulative decode (session) | **30.5 → 60.4 t/s @512 ≈ 1.98×** | plan §7:86 | VERIFIED (committed milestones) |

**Mechanism facts safe to quote:** the 17-byte MXFP4 block stride → 64 GB/s misaligned if not
SoA-repacked (deepdive:35-37); the per-token-GEMV explosion = **196,608 GEMV launches in one
2048-token forward, 88% of prefill**, caused by the `K%256 && N%64` batched-gemm gate that
hidden=2880 and the 32-expert router both fail (deepdive:80-86). dp4a_us+offset was an honest
negative (slower than dp4a_ss — hardware-lowered) — keep documented (deepdive:99-105).

---

## C. gpt-oss-20b — quality / correctness

| Claim | Number | Source | STATUS |
|---|---|---|---|
| Streaming PPL (256-tok bench) | **22.5** (q8 22.52 / f16 22.51 / oneDNN 22.53) | benchmark:84-88; plan §7:119-120 | VERIFIED (int-dot paths move PPL < 0.1%) |
| Forward-validation streaming PPL (586-tok corpus, plain rope ≤4096) | **17.9** (vs llama chunked 54.7 same corpus = no fwd deficit) | plan §7:1359-1362 | VERIFIED (committed; ≠ the 22.5 number — different corpus/depth, do not conflate) |
| Tokenizer parity | our **586 tok = llama 586 tok** (identical) | plan §7:1359 | VERIFIED |
| Greedy factual | "The capital of France is **Paris.**" via Harmony, finish=stop | benchmark:86-88 | FUNCTIONAL-VERIFIED |
| YaRN | plain rope BEST ≤4096 (PPL 17.9 vs full-YaRN 30.5); YaRN opt-in `IE_GPTOSS_YARN` >4096 | plan §7:1373-1377 | VERIFIED (committed, opt-in) |
| 20b never overflows fp16 residual | (smaller activations than 120b) | plan §7:1266 | VERIFIED (context for the 120b bug) |

---

## D. gpt-oss-120b — decode (2× B70 TP)

| Claim | Number | Source | STATUS |
|---|---|---|---|
| Decode peak | **14.33 → 32.05 t/s (2.24×)** (async-scatter all-reduce) | commit `d3a4aea`; plan header:28-29; §7:1266 | VERIFIED-as-committed; NEEDS CLEAN-BOX RE-BENCH — host/comm-bound decode showed **±15% box noise (same binary read 12–32 t/s)**, so 32.05 is a PEAK, not a steady median |
| Decode "holds" under box load | **~31 t/s** (after gate+up fusion, more contention-robust) | plan header:37-38; 120b doc:7,200 | NEEDS CLEAN-BOX RE-BENCH (this is the conservative print number, not 32.05) |
| vs LM Studio (llama.cpp layer-split, same 2 cards) | **~31 vs 12.42 t/s ≈ 2.6×** (older baseline cited 14.6 vs 12.42) | 120b doc:7,200; plan §7:1268 | NEEDS CLEAN-BOX RE-BENCH — LM Studio's 12.42 is an **owner-reported** number on his box at 100k+ ctx, not re-measured by us this session; confirm the ctx/config match |
| Decode baseline (pre-async) | 14.6 t/s (streaming) / 14.33 (timing-bench) | 120b doc:200; plan §7 | VERIFIED (committed baseline; two numbers = two harnesses, not a drift) |
| Win #2: gate+up MoE gemv fusion | MoE bucket **62% → 53%**, PPL **17.90 → 17.08** | commit `49afc6c`; plan header:37-39 | VERIFIED (committed; opt-out `IE_GPTOSS_NO_MXFP4_X2`) |
| Decode is comm/host-bound (not spill/quant) | ~50% of step = 2 all-reduces + router (Step-0 timing) | commit `d654c32`; plan header:33-34 | VERIFIED (profiling, committed tracer `IE_GPTOSS_TP_TIMING`) |

---

## E. gpt-oss-120b — prefill (2× B70 TP)

| Claim | Number | Source | STATUS |
|---|---|---|---|
| Prefill cumulative | **34 (T=1 workaround) → 538–679 t/s** (~16–20×) at moderate ctx (≤16k) | plan §7:1220-1230; 120b doc:4-6 | NEEDS CLEAN-BOX RE-BENCH — measured 2026-06-29 but "box degraded to ~9-min loads" (§7:1230); let it cool and re-run |
| Prefill @ long ctx (65k) | **433 t/s** (chunk 4096; 8192 spills+regresses to 301) | commit `08cb3a2`; plan §7:1211-1219 | NEEDS CLEAN-BOX RE-BENCH (same degraded box) |
| Lever: pad-M MoE GEMM (restore XMX) | 63 → **127 t/s** | commit `cc6b23a`; plan §7:1213-1214 | VERIFIED-as-committed; throughput NEEDS RE-BENCH |
| Lever: prefill chunk 256 → 4096 | 127 → **433 t/s** | commit `08cb3a2`; plan §7:1214-1217 | VERIFIED-as-committed; throughput NEEDS RE-BENCH |
| Lever: auto-replicate attn (Phase 1) at ctx ≤16384 | 433 → **538–679 t/s (1.24–1.54×)** | commit `34b53c3`; plan §7:1220-1230 | VERIFIED-as-committed; throughput NEEDS RE-BENCH |
| Prefill is comm-bound | **two host-bounced all-reduces = 67.6%** of prefill (attn 33.4 + MoE 34.2); MoE compute 24.4%, attention 6.1% | plan §7:1221-1224 (67.6%); 120b doc:22-24 (rounded "68%") | VERIFIED (profiling, committed). NOTE: doc says "68%", plan says "67.6%" — reconcile to one figure |

---

## F. gpt-oss-120b — the bug-fix story (three deep bugs)

| Bug | Fix + evidence | Source | STATUS |
|---|---|---|---|
| **(1) fp16 residual OVERFLOW** = the Harmony-chat "!" / NaN garbage | gpt-oss massive activations exceed fp16 max (65504) at ~L35 → `inf` poisons that position's KV → NaN cascade → argmax id 0 ("!"). Fix = `residual_add` saturates to ±65504 before the fp16 store. **Code confirmed:** `src/ops/elementwise.cpp:557-559` (`s = fmin(fmax(s,-65504),65504)`). EXACT no-op for in-range models → crown PPL stays 6.4527. | commit `38ea8f5`; plan §7:1266; 120b doc:13-16 | VERIFIED (committed, code-confirmed, crown gate holds; chat coherence validated: 17×23=391, haiku, "red/blue/yellow") |
| **(2) small-M XMX MoE corruption** = batched-prefill garbage (PPL ~123) | At E=128/top-4 each per-expert prefill GEMM runs at tiny M (~1–8 rows); BMG's small-M f16 XMX matmul **corrupts every local row ≥1 while row 0 stays correct**. Stable counting-sort packing puts token 0 at row 0 of every expert → **token 0 bit-exact, all others garbled** (the diagnostic). Fix = pad small experts to M=32 back onto XMX. **A/B: batched PPL 123 → 15.1985 == T=1 15.1985 (avg NLL 2.721199 BIT-IDENTICAL).** | commits `3c06c32` (correctness, per-row int-dot) → `cc6b23a` (pad-M XMX); plan §7:1189-1210; 120b doc:17-21 | VERIFIED (committed, PPL bit-identical A/B; 20b no-regression 19.36; batched prefill now DEFAULT, opt-out `IE_GPTOSS_TP_T1_PREFILL`) |
| **(3) comm-bound prefill (67.6%)** | Two host-bounced all-reduces (P2P HW-blocked on 2× B70) dominate; the **attn all-reduce is pure head-shard overhead** → Phase-1 (replicate attention) drops it → 433 → 679 t/s. Auto-selects at ctx ≤16384 (`gptoss_tp.cpp:359`). | commit `34b53c3`; plan §7:1220-1232; 120b doc:22-24 | VERIFIED (profiling + committed lever); throughput numbers NEEDS RE-BENCH (degraded box) |

---

## G. gpt-oss-120b — correctness / PPL (⚠ DRIFT — three different numbers, three configs)

The 120b PPL appears as **12.91 / 15.19–15.20 / 17.08**. These are NOT one number that drifted —
they are three different harnesses/configs. Record each with its config; do not present a
single "120b PPL." **For publication the cleanest correctness statement is the bit-identical
prefill A/B (15.19 vs 15.20).**

| PPL | Config | Source | STATUS |
|---|---|---|---|
| **12.91** (NLL 2.56) | original streaming (T=1), own corpus, ctx 1024, **pre-fusion** | 120b doc:199; plan §7:1268 | VERIFIED-as-measured but **STALE config** — predates the gate+up fusion (17.08) and the small-M fix. Do NOT headline. |
| **15.1985** (T=1) == **15.1985** (batched) ; **15.19** (replicate-attn) vs **15.20** (head-shard) | batched-prefill correctness A/B, post small-M-XMX fix | plan §7:1201-1202,1227; 120b doc:8-9 | VERIFIED (bit-identical batched-vs-T=1; the publishable correctness claim) |
| **17.08** (was 17.90) | decode-path PPL after gate+up MoE gemv fusion | commit `49afc6c`; plan header:38 | VERIFIED (committed; PPL improved one fewer fp16 round) |
| Load split | card0 **29.3 GB VRAM + 3.4 GB host-spill** (display safe, no crash); card1 31.1 + 0.3 | 120b doc:197-198; plan §7:1268 | VERIFIED (committed; owner's hard "display-safe" requirement met) |

**⚠ INTERNAL CONTRADICTION TO FIX before publishing `gptoss_120b_tp_2026-06-27.md`:** the
doc's **TL;DR (lines 1-51)** says "publish-ready, 538–679 t/s, decode ~31, PPL 15.19, chat
coherent + tool calling" — but the doc's **lower body (lines 190-220)** still says "120B
status (partial — forward correct, generation has open bugs)", quotes the old 12.91 / 14.6
t/s, lists the two now-FIXED bugs as OPEN, and says the benchmark is "held until (a) and (b)
are fixed." The lower section is stale. Reconcile (delete/replace the partial section) or the
artifact contradicts itself.

---

## H. Tool calling (gpt-oss Harmony)

| Claim | Evidence | Source | STATUS |
|---|---|---|---|
| Harmony tool calling wired end-to-end (render + parse + multi-turn agent loop), zero server changes | RENDER `tools`→Harmony `namespace functions` TS block; PARSE `commentary to=functions.NAME …<|call|>` → canonical `<tool_call>` → structured `tool_calls` | commit `fd9de14`; plan header:14-27; §7:1234-1255 | FUNCTIONAL-VERIFIED (committed) |
| Validated outbound call | `get_weather` → `tool_calls[0]=get_weather{"location":"San Francisco"}`, finish=`tool_calls` | plan §7:1250-1251 | FUNCTIONAL-VERIFIED (gpt-oss-20b, `ie serve`, non-stream) |
| Validated round-trip | tool result fed back → "…foggy, 61 °F, 12 mph", finish=`stop` | plan §7:1251 | FUNCTIONAL-VERIFIED |
| No-tools sanity | "Paris", finish=`stop`; no-tools chat byte-identical | plan §7:1252 | FUNCTIONAL-VERIFIED |
| Streaming tool calls in-engine (Harmony SSE buffered emit) | proven by non-stream→SSE bridge returning correct streaming `tool_calls`+content vs live 120b | commit `c5c4650`; plan §7:1256-1263 | FUNCTIONAL-VERIFIED |
| Crown neutrality | every edit `arch_==kGptOss`-gated → **crown PPL 6.4527 bit-exact** | plan §7:1252-1253 | VERIFIED |
| Scope honesty | only **user `functions.*`** wired; built-in browser/python tools = follow-up | plan §7:1262 | (state this limit when claiming tool calling) |

---

## I. Multi-GPU TP — 20b validation harness (proves the path is numerically sound)

| Claim | Number | Source | STATUS |
|---|---|---|---|
| 2-card vs 1-card PPL (Phase 1, MoE-TP) | decode 19.31 vs 19.36 (0.26%); prefill 16.85 vs 16.82 (0.18%) | plan §7:1276 | VERIFIED (committed) |
| 2-card vs 1-card PPL (Phase 2, head-shard) | streaming 19.49 vs 19.36; prefill 16.83 vs 16.82; "Paris" ✓ | plan §7:1280; 120b doc:177-181 | VERIFIED (committed) |
| 20b TP decode < single-card (EXPECTED) | 41 (2-card) < 58 (1-card) — TP overhead on a model that fits one card | plan §7:1276; 120b doc:160-166 | VERIFIED — state honestly: TP is for the 120b (doesn't fit), the 20b runs TP only as a correctness harness |
| Multi-shard GGUF loader | 120b 2-file → **687 tensors merged**; 20b single-file 459 unchanged | plan §7:1279 | VERIFIED (CPU-validated, `ie-inspect`) |

---

## J. Engine-wide gates (every gpt-oss change held these)

| Gate | Value | Source | STATUS |
|---|---|---|---|
| Crown PPL gate (Qwen3.6-35B, hard ≤ 6.57) | **6.4527** bit-exact (NLL **1.864495**) across all gpt-oss edits | plan §7 passim; deepdive:109; CLAUDE.md | VERIFIED (arch-gated edits → provably no-op for crown) |
| ctest | **31/31** (last explicit gpt-oss-session claim, 2026-06-27) | plan §7:1290,1362,1388 | NEEDS RE-RUN — see DRIFT below |
| Build | "build green" on the 2026-06-29 120b fixes (no fresh ctest count restated) | plan §7:1202,1208 | NEEDS RE-RUN |

**⚠ ctest DRIFT (task-flagged 7/30/31):** historical progression **12/12 → 15/15 → 21/21 →
30/30 (80B/Gemma era) → 31/31 (gpt-oss-20b era, 2026-06-27)**. The current source tree
declares **32 `add_test()`** entries (`tests/CMakeLists.txt`, verified this review). The
2026-06-29 small-M-XMX and replicate-attn commits report only "build green," **not** a fresh
ctest count. → **Action: run `ctest` on the current tree and state the real current
number (31 or 32) before publishing any "all tests green" claim.** Do not print "31/31" as
current without re-running.

---

## K. SUPERSEDED / DO-NOT-USE (these must NOT appear in any artifact)

| Retired claim | Why dead | Source |
|---|---|---|
| "gpt-oss-20b beats llama both axes **668/607 prefill (1.10×), 31.9/22.35 decode (1.43×)**" | vs a **STALE** llama build; current llama-SYCL is 908/49 — re-bench gives 1.9–4.5× / 1.13–1.16× (§A) | plan §7:1291; benchmark:42-44 |
| gpt-oss-120b is a "dead-end / doesn't fit on 2× B70" | RETRACTED — it runs (owner ran full MXFP4 @100k, 12.42 t/s; we run it TP now) | commits `03c85c2`/`ab014fe`; plan §7:1295-1301 |
| "1.40× 80B decode win" | DEAD — was vs an old llama (~37 tg); current llama-SYCL does 60.8, we're 7% behind | scorecard:126-131 |
| 120b "2 open generation bugs / benchmark held" | Both FIXED this session (residual overflow `38ea8f5`, small-M XMX `3c06c32`/`cc6b23a`) — but **the 120b public doc still says this in its lower half** (§G contradiction) | 120b doc:190-220 |
| README.md gpt-oss claims | **None exist** — README (44 lines) covers only the 5 older arches; it does **not** mention gpt-oss at all. Stale for launch; flag for doc-hygiene. | README.md |

---

## L. PRE-PUBLISH RE-BENCH CHECKLIST (the conservative gate)

Run on a **clean, idle, heat-soaked-controlled** box (current box degraded to ~9-min loads):

1. **[BLOCKER] Re-bench the 20b head-to-head vs CURRENT llama-SYCL** — the entire lead claim
   rests on llama = **927/927/896 prefill, 50.3/49.9/49.4 decode**. These were measured once
   on 2026-06-27; the scorecard's own rule is "re-bench every claim vs a current llama." The
   decode margin (1.13–1.16×) is within box noise — confirm it survives.
2. **[BLOCKER] Re-run `ctest`** and record the true current count (source declares 32; last
   doc says 31). Confirm crown PPL still 6.4527.
3. **Re-bench 120b prefill (538–679 / 433 t/s)** and **decode (~31 t/s peak 32.05)** on a cool
   box — both were taken on the degraded box; decode is host-bound with ±15% noise.
4. **Re-confirm LM Studio 12.42 t/s** reference (owner-reported, not our measurement) at a
   matched ctx, or label it explicitly as "owner-reported, LM Studio layer-split."
5. **Fix the `gptoss_120b_tp_2026-06-27.md` internal contradiction** (TL;DR publish-ready vs
   lower-body "partial, bugs open, held") before it ships.
6. **Reconcile the prefill-comm figure** to a single value (plan "67.6%" vs doc "68%").
7. Present all self-improvement multipliers (§B, §E levers) as **engineering progress vs our
   own baseline**, never as competitive ratios.

---

### One-line conservative summary for reviewers
The **correctness story is solid and committed** (crown 6.4527 bit-exact; 120b batched PPL
15.1985 bit-identical to T=1; residual-overflow and small-M-XMX fixes code-confirmed; tool
calling functionally validated). The **throughput and vs-llama/vs-LM-Studio ratios are real
but were measured on a now-degraded box and/or against a one-time opponent snapshot** — every
tok/s number and every ratio in §A, §D, §E needs one clean-box re-bench before it goes to
print, with the **20b 927-prefill llama baseline and the ctest count** as the two hard gates.
