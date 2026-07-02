# MASTER DEV PLAN — single source of truth

**This is the authority document for the entire build** (the inference
engine AND its two product offerings). If you are a fresh session — human
or AI — reading this one document plus the files it points to brings you
fully up to speed. Every other planning doc is subordinate to this one.

**Update discipline (non-negotiable):** this file is updated at every
milestone — phase completion, crown change, strategic decision, scope
change. If reality and this document disagree, fixing this document IS
part of the change. (This rule previously lived on PLAN.md; authority has
moved here.)

**Last updated:** 2026-07-02 (**📦 FRICTION-KILL — the engine is now installable by a stranger (`4c8834b`+`26768b2`).**
Owner's strategic pivot from "run every quant" → "make it usable by another person". `ie pull <name|hf-repo> [file]`
(ollama-style GGUF fetch, `src/cli/main.cpp`→`scripts/ie-pull`) + a reproducible **Docker image** (`Dockerfile`,
multi-stage: oneAPI 2026.x from Intel's apt repo — not on Docker Hub — + Intel Arc Level-Zero runtime from the kobuk
PPA; builds pure SPIR-V/JIT, the runtime specializes for the Arc device on first load) + `scripts/ie-docker` run
wrapper (`--device /dev/dri`, models volume, :11435) + `QUICKSTART.md`. **VALIDATED 2×B70**: `ie-docker serve`
Qwen3-4B, chat from host through the container → correct; warmed prefill 142 / decode 75.8 tok/s; image 2.49 GB.
Arc-in-Docker gotchas (build needs SPIR-V-not-AOT to skip ocloc; runtime needs `intel-oneapi-umf` for `libumf.so.1`
+ `LD_LIBRARY_PATH` for UR adapter discovery) → [[project_friction_kill_docker]]. PREVIOUS: 2026-07-02
(**🏆 qwen3moe TENSOR-PARALLEL — long-ctx decode+prefill win on 2×B70 (`1bf67a7`).**
`Qwen3MoeTpModel` (`src/model/qwen3moe_tp.cpp`): OPT-IN `IE_QWEN3MOE_TP=1` + `--gpus>1` (else the layer-split stays
default). Both cards compute EVERY layer concurrently — the fix for the layer-split's long-ctx decode bottleneck
(attention-at-depth, which the split does sequentially). **Attention HEAD-SHARD** (n_q/N + n_kv/N heads per card,
q/k/v col-slice — Q4_K/Q6_K native + Q5_K/Q8_0/F16 dequant-fp16 for bartowski mixes; per-head QK-norm/rope; KV sliced
to the card's kv-heads; o-proj row-parallel → PARTIAL → all_reduce_sum_fp16 #1) + **MoE EXPERT-SHARD** (card c owns
experts [c·E/N,(c+1)·E/N); replicated router → identical top-K; each card runs the qwen3moe int-dot kernels over ONLY
its experts via `expert_offsets+ef0_e` base + `E=efn` + local banks — kernels UNCHANGED, avoids the within-expert
EF/2=384 that breaks the down kernel's E_ffn%256; PARTIAL → all-reduce #2; carries the reduce-memset-for-all-T).
2 all-reduces/layer (host-bounced; P2P hw-blocked). NOT bit-exact (reduction order) → validated by coherence.
**VALIDATED 2×B70 (Tongyi Q6_K): coherent, fits ~13GB/card @130k (experts 24→12/card + head-sharded KV). Long-ctx WIN
vs layer-split @17k: decode 21→27.4 tok/s (1.30×), prefill 124→291 (2.35×). Short-ctx 40 vs 46 (all-reduce overhead
when attn is cheap) → TP is the LONG-session choice.** Layer-split default unregressed; crown untouched. Built by the
inference-engine-expert from a full spec (incl. the Q8_0/Q5_K attn-slice fix I caught in GPU validation); ALL GPU
validation by main-loop. `hermes-engine --tp` flag added (`d4bf74b`). PREVIOUS: 2026-07-02 (**🏆 qwen3moe 2-CARD
LAYER-SPLIT — full-context Tongyi/Coder on 2×B70 (fixes the 130k DEVICE_LOST) + `hermes-engine` one-command launcher.** (1) **`Qwen3MoeSplitModel`** (`src/model/qwen3moe_split.cpp`
+ `include/ie/qwen3moe_split.hpp`, commit `e7ccd0c`): `ie run/serve <qwen3moe> --gpus 2` now layer-splits kQwen3Moe
across the fleet. ROOT of the owner's crash: Tongyi Q6_K @130k = ~25GB weights + ~13GB fp16 KV = ~38GB > 32GB on ONE
card, and kQwen3Moe had NO multi-GPU path (`engine.cpp:51` "single-GPU today"), so `--gpus 2` silently loaded 1 card →
oversubscribe → DEVICE_LOST. Now split ~18.6GB/card (12.54 weights + 6.0 KV). **Approach B** (runs the EXISTING
Q4_K_M/Q6_K GGUFs — no all-Q8_0 needed): mirrors `Qwen35MoeSplitModel`'s DeviceFleet/LayerPlan orchestration but for
the SIMPLER dense QK-norm attention (no DeltaNet, no shared expert) — per-card weights on `fleet.dev(dev)`, per-card
full-attn KV via `kv_local_`, cross-card residual handoff via `copy_across` + the mandatory pre-copy queue drain.
Router = qwen3moe gemv + host `route_from_logits` for ALL T (tightest numeric match to single-card), sidestepping the
E=256-locked `moe_router` + its cross-card static-ptr landmine. Reuses the fixed Q6_K/Q4_K int-dot MoE kernels + CARRIES
the reduce-memset-for-all-T fix. Engine wiring: kQwen3Moe added to the `mgpu` set + a `n_gpus>1` split branch +
`q3moe_split_` dispatch. **VALIDATED on 2×B70:** 2-card load (12.54GB/card), COHERENT generation at ctx 8192 AND
**ctx 130000 (the crash scenario — no DEVICE_LOST)**, single-card `--gpus 1` unchanged (no regression), crown PPL
6.4527 held, ctest 31/31. Built by the inference-engine-expert from a full blueprint; ALL GPU validation done by
main-loop. Open/future (scope, not correctness): int8-KV not threaded on the split (fp16 only); no long-ctx
prefill-chunk clause yet. (2) **`hermes-engine`** (`scripts/hermes-engine`, commit `87e2184`): one command →
interactive model picker (GGUFs under ~/models+~/.seal) + ctx + gpus → starts SearXNG (from ~/searxng) + `ie serve`
(free port, IE_FA2_VEC=1) → launches Hermes Agent NON-DESTRUCTIVELY via env only (HERMES_INFERENCE_PROVIDER=custom /
CUSTOM_BASE_URL / HERMES_INFERENCE_MODEL / SEARXNG_URL + `--provider custom -m`), config.yaml untouched; engine under
setsid (Ctrl-C-safe) with pidfile-tracked teardown. Validated headlessly: discovery, engine lifecycle, engine serves
OpenAI chat (Coder→"345"), Hermes connects + REPL launches; the interactive turn needs a real TTY (owner confirms).
PREVIOUS: 2026-07-01 (**🏆 Tongyi-DeepResearch-30B-A3B (bartowski Q6_K) FUSED MoE — CORRECT + FAST, and a
PRE-EXISTING T>1 PREFILL BUG FOUND + FIXED.** Wrote the missing **Q6_K gate/up int-dot kernel**
`moe_prefill_gate_up_silu_q6k_q8` (`src/ops/moe_qwen3.cpp` — the Q4_K gate/up q8 two-bank schedule + the Q6_K
weight-read/two-scale-fold lifted from `down_q6k_q8_gen_impl`) and wired the qwen3moe dispatch (relaxed the
`unfused` guard so Q6_K gate/up fuses on the int-dot path; re-enabled per-expert SoA for Q6_K gate/up; INVARIANT:
SoA-on ⟺ fused path, so the AoS reference/fp16 fallbacks only run when SoA is off). **Tongyi Q6_K, 1 card: decode
18.7→41.9 tok/s (2.24×), prefill 32.9→595 (18.1×)** (`ie-bench`). **THE REAL STORY — the 2026-06-30 "fast kernel
degrades generation, reverted" was a MISDIAGNOSIS:** the Q6_K kernel was correct all along (the specialist's CPU
bit-exact proof was right); the degradation was a **pre-existing bug in the SHARED T>1 prefill reduce**.
`moe_prefill_reduce` ACCUMULATES (`acc = y[t*H+n]` as its base — moe_fused.cpp), but `ws_ffn_y_` was only memset
for T==1, NEVER T>1. So T>1 prefill added each layer's MoE onto the prior layer's stale output → corrupted hidden
states → wrong first token + poisoned KV → repetition loops in generation. **Masked because ie-perplexity AND
generation stream T==1 (prefill_chunk=0 default)** = the correct path, so PPL never exercised T>1 prefill; only the
correctness-blind throughput bench did. **Bug hit Coder (Q4_K) too** (Coder prefill was silently wrong). FIX = memset
`ws_ffn_y_` for ALL T at both reduce sites (int-dot + XMX) in `qwen3moe.cpp`. **The diagnostic that cracked it** (the
previous attempt lacked it, over-trusted corpus PPL): `IE_DIAG_LOGITS_OUT` in `ie-qwen3moe-test` dumps teacher-forced
T==1 decode OR (IE_DIAG_PREFILL) T>1 last-position logits; compare fused-vs-unfused, calibrate vs Coder's proven
kernel. **VALIDATION:** decode fused==unfused **31/31 argmax** (worst_cos 0.990, tighter than Coder's proven 0.984),
prefill fused==unfused **cos 0.9999** (was −0.02 Tongyi / 0.36 Coder pre-fix), generation coherent (" Paris.",
fused==unfused), Tongyi PPL 10.25 (q8-activation noise vs unfused 10.18), **crown 6.4527 held**, ctest 31/31.
⚠ **NOTE for owner (propose, don't close):** §7 gpt-oss-120b "batched T>1 prefill degraded @E=128 (workaround =
T=1-step prefill)" smells like the SAME stale-accumulator class in `GptOssTpModel` — worth the same memset audit.
⚠ Also observed (pre-existing, OUT of scope, NOT the MoE — fused==unfused both): `--prefill-chunk` PPL > streaming
for BOTH models (Tongyi 10.25→11.43, Coder 11.98→18.19) — a prefill-ATTENTION/chunk-boundary characteristic worth a
future look. Files: `src/ops/moe_qwen3.cpp`, `include/ie/moe_qwen3.hpp`, `src/model/qwen3moe.cpp`,
`tools/qwen3moe_test.cpp`. PREVIOUS: 2026-06-28 (**🔧 gpt-oss TOOL CALLING WIRED + VALIDATED end-to-end.** The blanket
"tool calling not supported (v1)" guard — which errored on SIX arches (Mistral/DeepSeek/Granite/Gemma/gpt-oss/llama),
leaving tools working ONLY on the Qwen `<tool_call>` path — is now broken open for **gpt-oss**. Wired via the OpenAI
**Harmony** tool protocol with ZERO server changes: (1) RENDER the `tools` array as a Harmony `namespace functions {
type NAME = (_: {…}) => any; }` block in the developer message (JSON-Schema→TS) + a commentary-channel system line;
(2) PARSE the model's `commentary to=functions.NAME …<|message|>{json}<|call|>` back into the canonical `<tool_call>`
the server already emits as structured `tool_calls`; (3) multi-turn — prior assistant calls + `role:"tool"` results
re-render in Harmony so the full call→result→continue agent loop works. **VALIDATED (gpt-oss-20b, `ie serve`):**
`get_weather` → `tool_calls[0]=get_weather{"location":"San Francisco"}` finish `tool_calls`; round-trip → "…foggy, 61 °F,
12 mph" finish `stop`; no-tools sanity → "Paris". **Crown PPL 6.4527 bit-exact** (every edit `arch_==kGptOss`-gated → no-tools
chat byte-identical), build green. `tokenizer.{cpp,hpp}` + `engine.cpp`. NEXT: same pattern unlocks the other 5 arches
(llama next). **STREAMING tool calls now wired in-engine** — gpt-oss is channel-structured, so `openai_server.cpp`
buffers the stream and emits the post-processed `r.text` as SSE (a `tool_calls` delta, or clean content), Qwen
live-streaming byte-identical; behavior proven by a non-stream→SSE bridge that returned correct streaming `tool_calls`
+ content against the live 120b. Built-in browser/python tools remain the follow-up. PREVIOUS: **🏆 gpt-oss-120b DECODE HITS 30+ tok/s on 2×B70 — 14.33→32.05 tok/s PEAK
(2.24×, PAST the 30 target), PPL-IDENTICAL, beats LM Studio 12.42 by ~2.6×.** Committed `d3a4aea`. THE FIX
= async-scatter all-reduce (`DeviceFleet::all_reduce_sum_fp16`): the TP all-reduce host-`.wait()`ed on BOTH
the gather (kept — CPU sum needs it) AND the scatter, but the scatter wait was dead time (in-order queues
already order the consumer); removing it + a 4-gen pinned-USM staging ring un-stalls the host pipeline.
**Decode is HOST/COMM-bound, NOT spill/quant-bound** — Step-0 timing proved comm (2 all-reduces + router) =
~50% of the step, so the model-shrink/TurboQuant/EXL3 idea was FALSIFIED for decode (spill ~10%, shrink caps
~+10%). Built `ie-gptoss-tp-bench` (engine's FIRST 2-GPU decode bench) + gated `IE_GPTOSS_TP_TIMING`
breakdown. **WIN #2 (`49afc6c`): fused gate+up MoE decode gemv** (`gemv_mxfp4_soa_q8_x2`) — one launch
for both projections (shared SLM-staged x) + folded biases → MoE bucket 62%→53%, holds 31 tok/s at box
load 5-7 (more contention-robust), PPL even IMPROVES 17.90→17.08 (one fewer fp16 round); opt-out
`IE_GPTOSS_NO_MXFP4_X2`. **NEXT lever: MoE gemv still 53%** — the per-expert DOWN gemv + a grouped-expert
MoE kernel (12 gemv launches → ~3); substantial kernel work, needs a quiet bench box (±15% box noise
already swallows sub-10% gains). ⚠ Decode is host-bound → reliable measurement needs a QUIET box (a stuck
`systemd-resolved` core made the SAME binary read 12–32 tok/s — the clean-box rule). PREVIOUS: **🎯 gpt-oss-120b RUNS on OUR engine
(2×B70 tensor-parallel) — loads safe + decode-CORRECT + fast; 2 open generation bugs.** `GptOssTpModel`
(engine's FIRST MoE-TP path) ships all 3
phases: P1 MoE expert-TP (MXFP4 experts sharded), P2 attention head-shard (**both cards compute**, halved
KV/card), P3 host-RAM expert spill (caps the display card → **no crash**). `--gpus 2` routes here. **120b
VALIDATED:** multi-shard loader opens both shards (687 tensors); spill caps card 0 at 29.3 GB (display
safe); **decode numerically CORRECT — streaming PPL 12.91 (own corpus), 14.6 tok/s BASELINE (now 32 — see
above) > LM Studio's 12.42.** 20b fully validated (2-card PPL≈single both axes, chat "Paris"); crown 6.45; ctest 31/31.
**🎯 120b CHAT NOW COHERENT — bug (b) FIXED (`38ea8f5`).** The Harmony-chat "!" garbage was an
**fp16 residual OVERFLOW**: gpt-oss's massive activations push the residual stream past fp16's finite max
(65504) at ~L35; the first `inf` poisons that position's KV → NaN cascade → all-NaN logits → argmax id 0
("!"). PPL never caught it (plain-corpus tokens never reach the magnitude); 20b never overflows (smaller).
FIX = `residual_add` saturates to ±65504 before the fp16 store (EXACT no-op for every in-range model →
crown PPL still 6.4527; proper follow-up = fp32 residual stream). Validated: 120b chat coherent across
math/creative/factual + multi-turn ("17×23=391", ocean haiku, "red/blue/yellow"). **120b is now USABLE:
coherent chat + 32 tok/s decode (2.24×) + PPL 17.08 → publishable.** 1 REMAINING (non-blocking) BUG:
**(a) batched T>1 prefill MoE degraded at E=128** — workaround active + DEFAULT (correct T=1-step prefill,
opt-in fast path `IE_GPTOSS_TP_FAST_PREFILL`); only costs first-token latency, output is correct.
Localized via the gated `IE_GPTOSS_TP_NANCHECK` residual tracer. Detail + isolation evidence in §7. PREVIOUS: **🎯 gpt-oss-120b CONFIRMED RUNS on 2×B70 (in LM Studio)** — owner ran the FULL MXFP4 across
the two cards at 100k+ ctx, 12.42 tok/s, usable. So the 120b is a REAL target. **NEXT-WORK = build
`GptOssSplitModel` (2-card, ~80% reusable from Qwen35SplitModel) + multi-shard GGUF loading → run the 120b on
OUR engine for the prefill win** (same arch family as the 20b, all kernels carry over; at 100k ctx prefill IS
the first-token latency, where we post 4.5×). De-risk on the 20b (`--gpus 2`) first. ⚠️ I had wrongly written
the 120b off as a "dead-end" autonomously — RETRACTED + scrubbed; see `feedback-no-unilateral-dead-ends`
(propose, never close — the owner decides). A research thread is OPEN (a NEW own method: MoE expert-tiering /
precision-follows-usage, rotor KV) — research+plan phase, not started. **PREVIOUS: 🏆🏆 gpt-oss-20b NOW BEATS
llama-SYCL ON BOTH AXES (re-verified vs a
CURRENT llama build) — prefill 2.3–3× self / 1.6–2.6× vs llama; decode ~1.2× vs llama.** ⚠️ The old
"beats both axes (607/22.35)" banner was vs a STALE llama; a clean re-bench shows the current llama-SYCL
at **908 pp / 49 tg** — much faster. Found + fixed the real prefill deficit: gpt-oss's attn q/k/v/o
projections were running as **per-TOKEN gemvs at prefill** (196k calls = 88% of prefill) because
`dense::gemv_q_T`'s batched branch is gated `K%256==0`, which H=2880 (2880%256=64) FAILS → per-row
fallback. Fix = a gpt-oss-local `proj` helper routing prefill (T>1) fp16 attn projections through
`gemm_fp16_onednn` directly (oneDNN handles K=2880 — the MoE path already does). The **router** had the
SAME bug (N=E=32<64 fails N%64) → per-token (49k calls = 54% of prefill after the q/k/v/o fix); batched it
too. **Prefill 759→1795 @512, 759→4147 @2K (5.5×), 714→3428 @4K.** H2H 1×B70 (verified, vs llama's BEST
FA-on): **pp512 1795 vs 927 = 1.94×, pp2048 4147 vs 927 = 4.47×, pp4096 3428 vs 896 = 3.83×; tg ~60/57/56
vs 50/49.9/49.4 = ~1.13–1.20×.** Decode unchanged (proj/router are T>1 only), crown 6.4527 bit-exact,
ctest 31/31, PPL band-neutral, Paris ✓. Opt-out `IE_GPTOSS_NO_PROJ_GEMM` (covers both). Publishable
write-up: `docs/public/gptoss_benchmark_2026-06-27.md`. PREVIOUS: 2026-06-27 (**🏆 gpt-oss-20b lm_head W8A8 int-dot — decode 56.7→60.4 tok/s (1.06×),
−0.58 GB VRAM, DEFAULT-ON.** The lm_head (Q8_0, 201k vocab) was the single biggest decode gemv
(~2.5 ms, dequant→fp16 read at 460 GB/s); kept it Q8_0 SoA + routed through `gemv_q8_0_soa_q8` (W8A8,
~80% BW, proven in qwen35) — half the weight read. PPL +0.09% (22.56 vs 22.54), crown 6.4527 bit-exact,
ctest, "Paris" ✓. Opt-out `IE_GPTOSS_NO_Q8_LMHEAD`. **The Q8_0-SoA-non-experts pattern is REUSABLE for
the 120b fit** (halves the attn/embd/lm_head VRAM). Cumulative session decode: **30.5→60.4 @512 = 1.98×
(~2.7× vs llama)**. Earlier session levers (dp4a_us+offset for gemv_mxfp4 = FALSIFIED, slower than
dp4a_ss which is hardware-lowered). NEXT: extend Q8_0 SoA to the attn q/k/v/o projections (the rest of
`gemv_fp16`), then 120b. PREVIOUS: 2026-06-27 (**🏆 gpt-oss-20b PREFILL LEVER — wide-tile hd64 attention (sink):
long-ctx prefill 327→716 tok/s @4K = 2.19× (1.37× @2K), DEFAULT-ON.** Completes the gpt-oss attention
campaign (decode #1/#2 below). The naive O(T·ctx) prefill collapsed (606→554→327 @512→2K→4K); routed
gpt-oss prefill (T>1, BOTH full + SWA layers) through `full_attention_gptoss_prefill_tile` =
`fa2_tile_wide_impl<HD=64, SINK=true>` (the proven 27B/gemma wide tile + the per-head sink, threaded as
a `bool SINK` TEMPLATE param so `if constexpr(SINK=false)` leaves crown/gemma/Coder BYTE-IDENTICAL —
crown 6.4527 bit-exact confirmed). Prefill now flat/improving across ctx (637→764→716) vs naive's
collapse. PPL +0.16% (FMA-reorder, same band as the 27B/crown tile's +0.15%), "Paris" ✓, ctest 31/31.
Opt-out `IE_GPTOSS_NO_FA2_PREFILL`. **gpt-oss now fast on BOTH axes at ALL ctx** (this session: decode
30.5→56.1 @512 / ~54 @2K, prefill +2.19× @4K). NEXT: gemv_mxfp4 geometry (154 GB/s) + SWA decode layers
(small) + 120b. PREVIOUS: 2026-06-27 (**🏆 gpt-oss-20b DECODE LEVER #2 — split-K FA-2 decode (sink): long-ctx
decode 25.3→53.9 tok/s @2K = 2.13× (1.31× @512), PPL-NEUTRAL, DEFAULT-ON.** After the MXFP4 GEMV win
(below), `attn_gptoss` became the dominant decode cost (62.7% @2K — the naive O(ctx) per-token reduction
collapses with context). Fix = route the FULL-attention (odd) decode layers through a COPY of the proven
split-K `full_attention_fa2_decode` + the per-head softmax **SINK** folded once in the combine pass
(`m'=max(m,sink); l=l·exp(m−m')+exp(sink−m'); out·=exp(m−m')`); the windowed (even) layers stay on the
bounded naive path (128 keys, already cheap). Decode is now nearly FLAT across ctx (56→54 tok/s
512→2K) vs naive's collapse (42.8→25.3). **Cumulative this session: decode 30.5→56.1 @512 = 1.84×, ~2×
@2K → ~2.5× vs llama-SYCL.** PPL-neutral (22.54 vs 22.52), crown **6.4527 bit-exact**, ctest **31/31**,
"Paris" finish=stop. Also fixed gpt-oss `ensure_attn_partials` (was /256, must be Bc=64 — 4× under-alloc
overflow latent at small --ctx) + wired it in engine.cpp. Opt-out `IE_GPTOSS_NO_FA2_DECODE`. New
`full_attention_fa2_decode_gptoss` in attention.cpp. NEXT: hd64 windowed PREFILL wide-tile (naive prefill
still sags 580→320 @512→4K) + the SWA decode layers (small remaining) + gemv_mxfp4 geometry. PREVIOUS:
2026-06-27 (**🏆 gpt-oss-20b DECODE LEVER — MXFP4 fused decode GEMV: 30.5→43.0 tok/s
= 1.41×, PPL-NEUTRAL, DEFAULT-ON.** Killed the 44% `dequant_mxfp4` decode bottleneck (per-token
full-expert fp16 materialization, ~10.8 GB/tok vs a 1.27 GB packed floor). Fix = a load-time **SoA
repack** of the experts (aligned `qs`/`e` planes, memory-flat 4.25 bpw — the native 17-byte block
stride tanked BW to 64 GB/s misaligned) + a **W4A8 int-dot GEMV** (`gemv_mxfp4_soa_q8`, `dp4a_ss`,
BRANCHLESS FP4 decode — NOT the byte-permute LUT, which is sw-emulated/slow on Xe) reading the weights
ONCE. Prefill routes through `dequant_mxfp4_soa_to_Bt` (raw bank dropped). Decode kprofile: gemv_mxfp4
8.2 ms (was dequant 11.8 + gemm slice), GPU step 27.1→21.97 ms, quant_q8_1 0.6%. **PPL-NEUTRAL** (q8
22.52 / f16 22.51 / oneDNN 22.53 @256-tok, Δ<0.05%); crown **6.4527 bit-exact**, ctest **31/31**;
coherence "Paris" ✓. gpt-oss decode now ~**1.9× vs llama-SYCL** (43.0 vs 22.35). Gates: opt-out
`IE_GPTOSS_NO_MXFP4_GEMV`, bit-faithful `IE_GPTOSS_MXFP4_F16`. New `src/ops/gemv_mxfp4.cpp`; `gptoss.cpp`
SoA repack + decode wiring. REMAINING: push gemv_mxfp4 past 154 GB/s (FP4-decode ALU / geometry) + the
**hd64 windowed decode tile** (attn_gptoss is now co-#1 decode cost at 30%, grows w/ ctx). PREVIOUS:
2026-06-27 (**🏆🏆 gpt-oss-20b BEATS llama-SYCL on BOTH axes + Harmony chat shipped + YaRN
investigated.** H2H (1×B70, `ie-bench` gained kGptOss): **prefill pp512 668 vs 607 = 1.10× WIN; decode tg128
31.9 vs 22.35 = 1.43× WIN** — even with unoptimized kernels. Decode kernel profile (the perf-lever map):
`dequant_mxfp4` **44%** (per-token full-expert dequant) + `gemv_fp16` 28% (oneDNN M=1) + `attn_gptoss`
23%→**55%@2180** (naive O(ctx) attn) → huge headroom (the MXFP4-decode kernel + hd64-tile are the levers,
designs ready). **HARMONY chat SHIPPED** (`4009238`): `build_harmony_prompt` + kGptOss chat dispatch + stop
(`<|call|>`/`<|return|>`) + final-channel extraction → chat "capital of France?" streams [analysis→final],
`res.text`="Paris", finish=stop — gpt-oss now usable as the reasoning model it is. **YaRN** (`b8240c4`):
implemented `rope_yarn` (ggml-exact, mscale=1.34657 verified vs llama-context.cpp) but MEASURED plain rope
base-150000 is BEST ≤4096 (streaming PPL 17.9 vs 30.5 full-YaRN / 22.1 mscale-only) — factor-32 YaRN perturbs
short-ctx; wired OPT-IN `IE_GPTOSS_YARN` for >4096. crown 6.4527 bit-exact, ctest 31/31 throughout. REMAINING:
MXFP4 dp4a decode kernel (44% lever) + hd64 windowed tile (prefill) + 120b fit/split (L3/L4/L5 designs in hand).
PREVIOUS: 2026-06-26 (**🏆 gpt-oss-20b RUNS ON THE ENGINE — 8th arch, forward COMPLETE + validated.**
The OpenAI MoE family (the 120B-on-64GB play) is live on the engine end-to-end. **STEP 1 loader** (`d931b13`):
forked a clean `GptOssModel` (gptoss.hpp/cpp) on the qwen3moe structure — loads every tensor bit-clean (Q8_0
attn q/k/v/o + F32 biases, F32 `attn_sinks[64]`, `post_attention_norm`, F32 router+bias, MXFP4 experts + F32
per-expert biases, Q8_0 embd/lm_head→fp16). **STEP 2 forward** (`a61443a`): the 3 new pieces ALL working —
(1) `full_attention_gptoss` (new kernel, COPY → crown byte-identical) = per-head softmax **SINK** (denom-only)
+ alternating **SWA** (even layers, 128); (2) `swiglu_oai` (new op, ggml-bit-exact α=1.702/limit=7.0);
(3) top-4 **MXFP4 oneDNN** per-expert MoE (dequant_mxfp4_to_Bt→gemm_fp16_onednn, the ONLY path) + router/expert
biases + SOFTMAX_WEIGHT routing. **VALIDATED:** greedy "France is"→" Paris.", "Japan is"→" Tokyo" (correct
factual argmax at pos 4/11); per-token NLL healthy; **tokenizer 586 tok = llama's 586 (identical)**; streaming
PPL **17.9** vs llama-perplexity chunked **54.7** on the same corpus → NO forward deficit (a broken fwd would be
≫50; gpt-oss is reasoning-tuned → high raw-text PPL). crown 6.4527 bit-exact, ctest 31/31. **Decode 38 tok/s**
(naive attn + per-token MXFP4 dequant — functional, perf-tile is a later lever). **RESIDUALS (usability, not
correctness):** Harmony chat template (gpt-oss is reasoning-trained — wants its channel format; raw completion
veers into thinking), YaRN rope (only matters >4096 ctx — plain rope used), prefill/decode perf (naive FA →
v2/wide-tile + MXFP4 int-dot decode kernel), then **120b scale-up** (load-only; disk OK, NVMe 77 GB free).
Spec: `docs/gptoss_arch_spec_2026-06-26.md`. PREVIOUS: 2026-06-26 (**🚧 gpt-oss FAMILY STARTED + MXFP4 dequant
SHIPPED.** `1ee4a2e` block_mxfp4 + dequant_mxfp4[_to_Bt] bit-exact; STEP 0 `bf70cdd` kGptOss + read_gptoss_config.
120B goal: gpt-oss-120b ~63 GB tight on 64 GB; fit levers = EXL3 ~3 bpw / low-bpw GGUF / RAM-overflow; or
Nemotron-H-120B. EXL3-80B archived → `/media/weezy/Data/ai-models-archive/`, NVMe 77 GB free.) PREVIOUS:
2026-06-26 (**🏆 80B oneDNN MoE-PREFILL
LEVER — WIN CONFIRMED 1.53× / 1.91×, DEFAULT-ON.**
The last DeltaNet-moat arch where we lost prefill (0.50×) is fixed end-to-end across 5 commits. **Step 1
(`bb82d44`):** `gemm_onednn.cpp:ctx_for` was a static singleton bound to card 0 → a card-1 GEMM hit
`UR_RESULT_ERROR_DEVICE_LOST` (why oneDNN was force-OFF on every multi-card path). Now one
engine/stream/prim-cache **per SYCL device** (`unordered_map<sycl::device,OnednnCtx>`) — PROVEN by
`ie-onednn-multidev-test` (both B70s interleaved, no DEVICE_LOST, exact). **Step 2 (`2219c58`/`e0c0d13`/
`61a8e1b`/`d48798a`):** ported crown's recipe to `qwen3next.cpp` — DeltaNet recurrence sub-chunk ≤512 (§1-safe,
`--sweep` VERDICT CLEAN) + per-expert dequant Q4_K/Q6_K → fp16 Bt + `gemm_fp16_onednn` (multi-card-safe via
Step 1's per-device map) + engine pf_chunk raise. **CLEAN-BOX A/B (8K prompt, M≈160, new-old-new, 0.6%
baseline variance):** today's default (chunk 512, int-dot) **480** → big chunk 8192 int-dot **601** → big chunk
8192 **oneDNN 919/915** = **1.53× over int-dot at the same chunk, 1.91× end-to-end.** M≈160 was NOT borderline.
**DEFAULT-ON for max_ctx≥8192** (engine raises chunk to min(8192,max_ctx); forward engages oneDNN at T≥MINT
def 6144 = M≥120, the confirmed-winning regime); opt-out `IE_QWEN3NEXT_NO_MOE_ONEDNN` (→596, int-dot);
short-ctx (<8192) UNCHANGED. Default-on auto-engages (no flag → 914.8); crown 6.4527 bit-exact, ctest 30/30,
oneDNN coherent (no NaN/collapse) all 48 layers. **✅ llama-SYCL H2H DONE (same GGUF, 2×B70):
pp8192 ours 915 vs llama 627 = 1.46×; pp16384 722.7 vs 590.3 = 1.22×** — the structural 0.50× prefill loss is
now a prefill WIN over llama-SYCL at both 8K and 16K. §7 has full detail. **NEXT (autonomous):** big-model
exploration → `docs/big_model_roadmap_2026-06-26.md` (gpt-oss-120b is the pick; prototyping on gpt-oss-20b).
PREVIOUS: 2026-06-26 (**🏆 CROWN LONG-CTX PREFILL: oneDNN large-M MoE = 1.48× — DEFAULT-ON.** Reopened a
documented "dead end" (it was a T<2048 fused-MoE dispatch cliff making the large-M regime unreachable, NOT the
kernel math): big engine prefill chunk (8192) → MoE sees M≈256 → per-expert **oneDNN** GEMM (new
`dequant_q4_moe_soa_to_Bt` keeps the DEFAULT SoA experts so **decode is untouched**); DeltaNet recurrence
internally sub-chunked to ≤512 (§1-safe). Crown long-ctx prefill **727→1079 tok/s (1.48×)**, beats llama-SYCL
(ours-real 1058 > llama-synthetic-best 910 = 1.16×), decode 56.55 unchanged, needle-correct, long-ctx PPL Δ+0.15%,
6.4527 bit-exact, ctest 30/30. DEFAULT-ON for `max_ctx≥8192` (opt-out `IE_QWEN36_NO_MOE_ONEDNN`; short-ctx
unchanged). 6 commits (`2c819c3`→`909cbd1`→`26c5e8f`→`c2b4a79`→default-on). CROWN-ONLY (80B 2-card-oneDNN-blocked;
27B dense wins via tile). Also this session: **SG32 wide-tile attention rewrite** shipped OPT-IN
`IE_FA2_TILE_SG32` (subgroup 16→32 via shadow-template; SIMD32 works on BMG-G31; **+5-7% hd128 long-ctx CONFIRMED**
on Coder, hd256/512 gemma correct + mechanism-sound but perf-unconfirmed — gemma bench too noisy; `96aa02f`→
`f9e2c3f`); 16K-prefill ncols lever FALSIFIED (NO-GO, `602b0d6`); gemma 16K is a 1.58× WIN (prior 0.71× was stale);
27B `--spec` usage text corrected (REGRESSES on Q4_K_M, `f5614cf`). **11 commits total.** §7 has full detail.
PREVIOUS: 2026-06-26 (**🏆 LONG-CTX PREFILL SWEEP + PUBLICATION CLAIMS SECURED — 8 commits, all gates green.**
THE HEADLINE: a hd256 wide-tile attention lever (`full_attention_fa2_prefill_tile_gemma`, gated ctx≥6144, opt-out)
ported to all three DeltaNet hybrids' long-ctx full-attn prefill — **27B 16K prefill 0.52× LOSS → 2.13× WIN** (now
wins prefill at EVERY length 1.96–3.08×, `07ab8ef`); **crown +57% @16K** (`bf73093`); **80B +12.5%** bit-identical
(`442af21`). PUBLICATION INTEGRITY: **27B prefill ~2× RE-VERIFIED** vs current llama-SYCL (refuted the HIGH-RISK
"public llama ≈718" landmine, `76f65bf`); **Gemma 31B MTP lossless 1.48× SECURED**, 26B MTP confirmed NOT lossless
(corrected a false claim, `466765c`). Coder decode: opt-in res_rms fusion +2-3% + **launch-fusion line CLOSED**
(commodity intra-kernel grind, `11b0a74`). HONEST RESIDUALS: crown/80B long-ctx prefill REMAIN a loss — it's
**MoE-GEMM-bound** (256 experts/top-8 → 16 rows/expert at the T≤512 chunk = small-M, no GEMM amortizes); the
straight oneDNN port is FALSIFIED by that math (`c96909a`), and the **chunk-cap lift is TESTED + FALSIFIED**
(`0a293c0`: safe on this box but perf only +6.7%@1024 / REGRESSES@2048, + research found the "fix" is timing-luck
+ a DEVICE_LOST hazard — Intel Triton XPU #6658). The ONLY remaining crown/80B long-ctx lever = the L-effort
MoE-batch-decouple (uncertain ROI, not started). Dense-27B/coder decode + crown/80B long-ctx prefill = the honest
losses; everything else (gemma/27B all-length prefill, coder/crown short-ctx, gemma-31B lossless MTP) is the verified
headline. PREVIOUS: 2026-06-25 (**⚖ DELTANET MOAT RE-VERIFIED — honest correction.** Fresh same-box H2H vs CURRENT
llama-SYCL (on-GPU verified, not the ~0.5 CPU-fallback): **Crown 35B-A3B: prefill WIN 1.14× (1101 vs 962), decode
PARITY (78.1 vs 76.9 — the prior "1.10× decode win" is STALE; llama caught up). Qwen3-Next-80B (2×B70): decode
56.6 vs 60.8 = LOSE 0.93× (the famous "1.40×" was vs an OLD llama @37 tg), prefill 311 vs 625 = LOSE 0.50×.**
The DeltaNet win that HOLDS is Crown PREFILL. Decode is BW-bound (profile: GEMVs ~70%, dn_recurrence 3.4%, lm_head
already ~63% BW) → near ceiling, both engines hit the memory wall. **TWO BLOCKERS to re-winning DeltaNet decode
(documented `docs/COMPETITIVE_SCORECARD_2026-06-25.md`):** (1) NO MTP head in the Crown/80B GGUFs (0 `nextn`
tensors — verified; only the 27B blk.64 + gemma's separate mtp file have one) → MTP self-spec is blocked until a
head-containing GGUF is obtained/imported; (2) the spec hooks (all_logits/hidden_pre_norm/DeltaNet ckpt) are
27B-`Qwen35DenseModel`-only — Crown(`QwenModel`)/80B(`Qwen3NextModel`) need them ported. **80B prefill 2× gap is
structural:** oneDNN (the Coder/Gemma MoE-prefill win) is single-card-only (ctx binds card 0 → DEVICE_LOST
multi-card), so the 2-GPU 80B MoE is stuck on int-dot; chunk relax 256→512 measured MARGINAL (+2.5%). Unblock =
per-device oneDNN ctx (scoped, multi-GPU-risky). Scorecard + ledger updated; the prefill wins (Gemma all-lengths,
Coder 4K/8K, Crown 1.14×) are the verified headline. PREVIOUS: 2026-06-24 (**🏆🏆 GEMMA SLIDING-WINDOW ATTENTION → BEATS LLAMA AT EVERY PREFILL LENGTH.**
Profiled gemma 16K prefill = **96.3% attention** (`fa2_tilew_compute`); the SWA layers (head_dim 256, ~5/6 of
layers, window=1024) were doing FULL causal O(T²) — both wasteful AND a correctness mismatch vs llama (which
windows). Added sliding-window to the wide tile kernel (per-WG out-of-window tile SKIP keyed on the WG's first
query + per-key window mask; P·V auto-correct via masked weight=0). Adversarially reviewed (no bugs) + coherence
PASS (windowed answers "Dartmouth" top-1 vs full-causal's echo — windowing is SHARPER/more correct). DEFAULT-ON
(opt-out `IE_GEMMA4_NO_SWA_WINDOW`; global layers stay full-causal). **FINAL H2H (single-B70 vs llama-SYCL): 4K
1642 vs 809 = WIN 2.03×; 8K 1453 vs 761 = WIN 1.91×; 16K 1016 vs 641 = WIN 1.58×.** Gemma prefill: lost ALL THREE
at session start → now WINS ALL THREE by 1.58-2.03×. crown 6.45 bit-exact (qwen3moe passes window=0), ctest 30/30.
KEY: it's a correctness FIX that's also the perf lever — full-causal-on-SWA diverged from the oracle at T>1024.
NEXT: same windowing could help Coder if it has SWA layers (it doesn't — dense attn). PREVIOUS: 2026-06-24
(**⭐⭐⭐ GEMMA ATTENTION LEVERS SPREAD TO CODER — wide tile (REGDOT+SMALLBC) now
DEFAULT-ON for qwen3moe → Coder prefill +6% (4K 1303 / 8K 919 / 16K 547 vs hd128-tile 1248/867/535), FLIPS 8K vs
llama from parity to WIN (919 vs 849 = 1.08×); 4K WIN 1.28×; 16K 0.88× (was 0.85×). crown PPL 6.45 UNCHANGED
(wide kernel numerically equiv at HD=128), ctest 30/30. The `fa2_tile_wide_impl<HD,Bc,REGDOT>` is now general
(128/256/512); dispatcher gained HD=128; qwen3moe routes to it (opt-out `IE_QWEN3MOE_NO_WIDE_TILE`). KEY: regdot
needs small Bc (smallbc+regdot beats baseline; neither alone does). Dense arches (Llama/Granite/Mistral) NOT spread
— they're MLP-bound (attn ~30% even @16K), so the attention lever is ~flat there (v2 was already A/B-flat on granite).
PREVIOUS: 2026-06-24 (**oneDNN MoE-GEMM LEVER PORTED TO GEMMA — gemma 26B-A4B prefill 4K 1.70× /
8K 1.30× over our int-dot SoA baseline (clean-box: 4K 750 vs 440, 8K 248 vs 190 tok/s).** The SAME lever that
put Coder ahead of llama, now on the 2nd arch: per-expert `dequant_q4_0_soa_to_Bt` + `gemm_fp16_onednn` over the
existing fused-MoE packing (`Gemma4Model::moe_onednn_proj`), reusing the dense path's proven Q4_0→fp16 dequant
(bit-identical to AoS, same `repack_q4_0_to_soa`). T-gated ≥4096 (opt-out `IE_GEMMA4_NO_MOE_XMX`,
`IE_GEMMA4_MOE_XMX_MINT`): at T≥4096 Gemma's 128-expert/8-used routing gives rows/expert = T/16 ≥ 256 — exactly
Coder's winning oneDNN regime. The OLD "gemma MoE-XMX is a loss (632→420)" note was at T≈512 (~30 rows/expert,
small-M starvation) AND used hand-rolled `gemm_fp16`, not oneDNN — superseded. Below 4096 the int-dot SoA path
stays (no small-M regression). CORRECTNESS: argmax bit-identical int-dot↔oneDNN at 4K/8K/16K full prefill (a
layout bug would diverge). QUALITY: oneDNN keeps fp16 activations (int-dot quantizes to int8) + bit-identical
fp16 weights → higher fidelity; T=1 streaming-PPL wobble (+0.36% NLL real text) is harness noise (inflated
absolute PPL). crown 6.45 bit-exact, ctest 30/30. ⚠ BENCH LESSON re-confirmed: heat-soaked box gave 4K 1.27×;
cooled box (only light PPL since) gave the true 1.70× — measure 4K/8K, NOT 16K (one 16K prefill = 171s, cooks
the box). ⚠ **LLAMA HEAD-TO-HEAD (single-B70, same file): we LOSE prefill — us 750/248 vs llama-SYCL 831/758 @
4K/8K (0.90× / 0.33×).** Two facts: (1) llama.cpp's SYCL build DOES run `gemma4` on B70 (13.43 GiB, ngl 99, fast)
— the "new arch = moat, llama can't run it" hypothesis is FALSIFIED; always measure, never assume. (2) The 8K
collapse is **ATTENTION O(T²), not MoE**: llama scales 831→758 (−9%) 4K→8K while we collapse 750→248 (−67%) — the
SAME naive-attention problem solved on Coder. MoE-oneDNN was real+necessary (moved 4K from ~0.53× int-dot to
0.90×) but only HALF the playbook. **STEP 2 DONE — v2 attention for SWA layers:** routed Gemma's head_dim-256
SWA layers (the ~5/6 majority) through the proven `full_attention_fa2_prefill_v2` (query-row-block + cooperative
KV SLM tile), numerically identical to naive (argmax bit-identical, same 1/sqrt scale w/ Q pre-scaled). **8K
203→288 tok/s (+42%)**, argmax-identical, crown 6.45 bit-exact, ctest 30/30. T-GATED ≥6144 (env
`IE_GEMMA4_FA2_V2_MINT`, opt-out `IE_GEMMA4_NO_FA2_V2`): v2 REGRESSES small T (4K 624 vs 729 = 0.86×, L2 absorbs
redundancy) — crossover in (4096,6144]. ⚠ STILL BEHIND LLAMA @8K (288 vs 758 = 2.6×) because the **global
head_dim-512 layers stay naive O(T²)** (they overflow v2's `q_vals[16]` reg cap, dpl=HD/16=32). **hd512-v2 TESTED
+ REVERTED (banked negative):** generalized v2 to head_dim ≤512 (`q_vals`/`out_local`→[32], K+V SLM 64 KB <
128 KB/WG cap) + routed the 512 layers through it. CORRECT (argmax bit-identical 8K/16K, crown 6.45, ctest 30/30)
and within-run beat all-naive (8K +11%, 16K +46%) — BUT it does NOT clearly beat the SWA-only build at 8K
(all-v2 279 vs SWA-only 288–297 despite a cooler box) because hd512-v2 has poor occupancy (64 KB SLM → ~2 WG/Xe-core
vs naive-512's full occupancy) — it's the weakest of {hd256-v2, naive-512, hd512-v2}. The 256 SWA layers (5/6 of
all layers) carry the v2 benefit; the 512 layers are better left naive. Reverted hd512-v2;
don't re-chase it. **STEP 3 SHIPPED — WIDE TILE KERNEL (head_dim 256/512):** `full_attention_fa2_prefill_tile_gemma`
(templated `fa2_tile_wide_impl<HD,Bc>`, `<256,64>`/`<512,32>`) — a faithful generalization of llama's flash-attn-tile
structure (per-lane COMPLETE KQ dots, NO per-key subgroup reduce = the v2 floor; half2 coalesced K/V SLM tiles; SLM
76/82 KB < 128 KB cap). Handles BOTH SWA-256 and global-512 layers (v2 couldn't do 512). **8K 435 vs naive 227 =
1.92× (and +45% vs v2 300); 16K 230 vs naive 91 = 2.54×.** argmax bit-identical tile==naive==v2 (100 @8K, 2037
@16K), crown 6.45 bit-exact, ctest 30/30. Adversarially reviewed (inference-engine-expert: no correctness bugs) BEFORE
GPU validation. T-gated ≥6144 (`IE_GEMMA4_FA2_V2_MINT`), opt-out `IE_GEMMA4_NO_FA2_TILE`→v2→naive. hd128 tile
(crown/Coder) UNTOUCHED. **Gemma 8K prefill vs llama: 0.30× (naive, session start) → 0.57× (tile) — closed the gap
from 2.6× to 1.74× behind.** **STEP 4 SHIPPED — REGDOT (register-staged KQ dot in the wide
tile):** the `_regtile` CPE-chunk lever (2 parity fp32 accumulators per dot, bounded staging) ported into
`fa2_tile_wide_impl<HD,Bc,REGDOT>` via `if constexpr`. The lever that REGRESSED at hd128 (0.82×, too-short 64-iter
dot) **WINS at hd256/512** because the long dot (hd512 = 256 serial FMAs, single-acc) is carried-dependency bound:
**8K 442 vs 343 single-acc = +29%; 16K 224 vs 177 = +27%.** argmax bit-identical (100/2037), crown 6.45, ctest 30/30.
Default-ON, opt-out `IE_GEMMA4_NO_TILE_REGDOT`. **Gemma 8K prefill arc this session: naive 227 → v2 300 → tile 343 →
+regdot 442 (1.95× over naive); vs llama 758 = 0.58×.** **STEP 5 SHIPPED — OCCUPANCY (SMALLBC) + LOW TILE GATE → BEAT LLAMA @4K+8K.**
Two findings: (1) the default tile Bc (64/32) put the SLM at 76/82 KB → only 1 WG/Xe-core (128 KB cap) =
occupancy-STARVED. Halving Bc (32 @hd256 / 16 @hd512) → 42/49 KB → 2-3 WG/core: **8K +54%, 16K +59%** (default-on,
opt-out `IE_GEMMA4_NO_TILE_SMALLBC`). (2) The tile kernel does NOT regress at small T (unlike v2), so the 6144 gate
was crippling 4K — at 4K tile is **1366 vs v2/naive 497 = 2.75×**. Gave tile its own low gate (`tile_minT`=2048,
env `IE_GEMMA4_TILE_MINT`); v2 fallback keeps 6144. **FINAL HEAD-TO-HEAD (single-B70, same box, vs llama-SYCL): 4K
1366 vs 785 = WIN 1.74×; 8K 857 vs 737 = WIN 1.16×; 16K 466 vs 658 = lose 0.71×.** argmax bit-identical, crown 6.45,
ctest 30/30. **Gemma prefill went from losing ALL THREE (0.30–0.59×) at session start to BEATING llama in the
interactive 4K-8K range.** Only 16K still behind (occupancy still tighter at long ctx + the MoE/proj fraction grows).
**Gemma is now a BOTH-AXES competitive story: prefill WIN @4-8K + lossless MTP spec-decode 1.46× @decode** — moved
out of the "loses to llama" bucket. NEXT (16K): push hd256→Bc16 too, or the MoE GEMM at 16K. Decode at BW ceiling.
PREVIOUS: 2026-06-24 (**🏆 AHEAD OF LLAMA @4K, PARITY @8K (single-B70 fair) — Coder prefill us 4K 1150 /
8K 810 / 16K 530 vs llama-SYCL 1016 / 849 / 624.** The MoE GEMMs now run through **oneDNN** (`gemm_fp16_onednn` —
its tuned XMX matmul, ~1.65× our hand-rolled `gemm_fp16` which was only 18-29% of B70 peak, + fp16-direct output
killing the cast/fp32-scratch). That's exactly llama's prefill backend (oneMKL/oneDNN); our `gemm_fp16` was the
uniform 1.3× gap. PPL 3.1785 (BETTER than int-dot 3.1815), crown 6.4527 bit-exact, ctest 30/30. ⚠ FAIR-COMPARE
LESSON: an earlier llama-bench used `level_zero:*` (BOTH B70s + iGPU = multi-GPU) — invalid vs our single-GPU
engine; pin `level_zero:0` for 1-card. The remaining gap is ONLY @16K (530 vs 624 = 1.18× behind), where attention
is 61% and at its SG16 ceiling (register-block spilled, GQA wash, faithful-port already) — the last frontier needs
llama's exact nbatch_K bounded register-staging. oneDNN is single-card only (multi-card=DEVICE_LOST; qwen3moe is
1-GPU). Session arc @4K: ~600 (v2-era) → 1150 (ahead of llama). PREVIOUS: 2026-06-24 (**MoE-GEMM-on-XMX DEFAULT-ON
— Coder prefill 4K 777 / 8K 680 / 16K 481
tok/s; llama 16K gap 1.84×→1.30× behind, competitive at 4-8K.** Found by a 5-lens parallel fan-out (kernel-roofline
/ host-scheduler / weight-layout / KV / algorithm-path agents on one captured profile). The algorithm-path lens read
llama's actual dispatch (`MMQ_MAX_BATCH_SIZE=32`, ggml-sycl.cpp:4065): llama switches quantized MoE GEMM to
fp16-on-XMX above batch 32; we ran int-dot dp4a at batch ~256 on a compute-bound (AI~680) matrix-matrix shape. Fix:
per-expert dequant W4/6_K→fp16 + `gemm_fp16` over the existing expert-sorted slices (`moe_xmx_prefill`), reusing
gather/offsets/reduce. T-gated ≥4096 (2048 stays int-dot, no regression), opt-out `IE_QWEN3MOE_NO_MOE_XMX`. A
SoA-aware Q6_K dequant (byte-identical to AoS, GPU-verified) lets it engage with the DEFAULT SoA banks → decode
keeps SoA-Q6K (+13.8%). crown 6.4527 bit-exact, ctest 30/30, +0.17% NLL. Session arc @16K: v2 105 → tile 352 →
+MoE-XMX 481 (**4.6×**). Fan-out NEGATIVES (banked): GQA head-packing WASH (attention ALU-bound not KV-bound — L2
absorbs redundant reads), register-block KQ REGRESSED (spill) → the attention tile kernel is near its practical
SG16 ceiling. NEXT to beat llama: 16K attention (61%, needs SG32 / llama nbatch_K register-staging) + MoE-XMX
fp32-swiglu (PPL) + fuse dequant into gemm. PREVIOUS: 2026-06-23 PM (**⭐⭐ PREFILL WIN — Coder prefill 2.0–3.1×
(16K 105→324, DEFAULT-ON), llama gap 5.9× → 1.9×.** Root cause was a MISFRAME: llama-SYCL uses NO XMX/
joint_matrix for attention ("Todo: use XMX" in their source) — it's a SIMD "tile" kernel. The win = a FAITHFUL
port of llama's tile INNER LOOP (`full_attention_fa2_prefill_tile`): 16 subgroups × 1 q-row, every lane does 4
complete register-resident Q·K dots, full head_dim streamed as half2 in registers, distributed softmax/VKQ,
Q resident — all 256 lanes busy, no per-key reduce. (A 1st high-level-structure port was 0.6× v2 — the perf is
in the inner loop, not the WG geometry.) Measured 4K 2.0× / 8K 2.4× / 16K 3.08× (grows with ctx; reverse-order
confirmed). Coherent + on-topic, crown 6.4527 bit-exact (qwen3moe-gated), opt-out `IE_NO_FA2_PREFILL_TILE`→v2.
SUPERSEDES v2/xmx/orchestration as the Coder prefill kernel. The dead-ends — XMX fused (0.86× v2), gemm-
orchestration (DELETED, 0.68×) — are recorded so they aren't re-chased. NEXT: still 1.9× behind llama (324 vs
626) — port llama's larger ncols (more q-cols/WG) + head-dim sub-batching to close further; wire tile into the
dense full-attn arches (Llama/Gemma/Granite) which also process full-T prefill. PREVIOUS: 2026-06-23 (**PREFILL
PIVOT — the Coder 16K prefill 5.4× gap is **86.7% our SIMD O(T²)
attention running on the vector ALU, not XMX.** Validated the fix: `bench_gemm_fp16` at attn shapes hits
26.7/30.2 TFLOPS (~13-27× our SIMD fa2_v2) → orchestrate prefill attn as gemm_fp16(Q·Kᵀ)+masked-softmax+
gemm_fp16(P·V). Scaffold's CORRECTNESS BUG root-caused+FIXED (`25f1f84`: `gemm_fp16` K-loop was unguarded for K%16≠0 → P·V
gemm read next-row garbage for non-16-aligned prompts; fast-path-preserved guard → crown 6.4527 bit-exact,
Coder == v2 byte-identical). **BUT the now-correct gemm path was 0.68× v2** (shape-misleading "13-27×"; real orchestration
query-blocks → ~5000 launches/layer + materialized S = launch-bound) → **orchestration DELETED** (and a Codex adversarial
review found it was the sole caller hitting `gemm_fp16`'s ragged-N store corruption — removed the dead path, documented the
store-alignment contract). **The real lever = the FUSED XMX flash-attn-2 kernel** (`full_attention_fa2_prefill_xmx`): it
EXISTS + is correct; multi-subgroup re-tiling took it **0.27× → ~parity v2 @8K (0.86× @16K)** — occupancy fixed, residual =
the SLM-staged softmax round-trip → needs a **register-resident softmax** (the scoped next kernel project). NSG now chosen at
runtime from the SLM budget (adversarial-review fix → safe for any head_dim). Default v2 path UNTOUCHED; gated default-off.
Decode 3.85× gap = STRUCTURAL on the COMMODITY axis (8 kernel approaches + a faithful llama
`fattn-vec` port ALL flat → parity; command-graph RULED OUT via clean `ie-attn-profile` isolation —
`fa2_partial` is intrinsically ~40 GB/s even with ZERO inter-kernel host gaps) → DE-PRIORITIZED vs the
prefill win. Strategic note: the whole 2026-06-23 marathon was fought on Coder/qwen3moe = the COMMODITIZED
axis where llama's mature SYCL kernels win; our MOAT (Crown 35B both-axes win, Qwen3-Next-80B decode 1.40×)
is unchanged and still ahead. See §7 top.**) PREVIOUS: 2026-06-22 EVE (**✅ INT8-KV DECODE wired into Coder — 1.29× @8K (`d70065e`); 5-LENS DECODE
FAN-OUT killed command-graph + dtype + WG-geometry hypotheses with evidence; 27B-dense decode gap VERIFIED
REAL (llama 22.65 vs us 16.3 = 1.39×, same box — NOT stale) but it's the commoditized axis. NEXT LEVER =
`fa2_partial` decode-attention (62% of Coder decode @16K, latency-bound at ~11% BW — the decode twin of the
prefill-attn fix). See §7 top.** PREVIOUS: 2026-06-22 PM (**✅ FA-2 PREFILL KERNEL SHIPPED — qwen3moe long-ctx 2.05× @16K (`430ab08`).**
The publish-blocker's worst case (Coder/qwen3moe, pure full-attn → unchunked O(T²) `attn_naive_compute` =
73.3% of prefill, 16K = 317s) is FIXED: new `full_attention_fa2_prefill_v2` (query-row-block — keeps naive's
split-head-dim+reduce inner loop, but 1 query row per SUBGROUP, Br=16 rows/WG sharing one K/V SLM tile →
cuts redundant KV HBM ~Br×). Measured Coder Q4_K_M 1×B70: **512 0.95× / 4096 1.04× / 16384 2.05×** (317s→146s).
KEY FINDING: naive is **L2-bound, not HBM-bound, up to ~4K** (redundant reads L2-served → tiling is parity);
only PAST L2 (16K) does the redundancy hit real HBM where tiling halves it → default-ON **T≥2048** crossover,
opt-out `IE_NO_FA2_PREFILL_V2`. Two variants measured+REJECTED first (PERF_LEDGER): v1 thread-per-row 0.31×
(serialized head_dim onto 1 lane — the reduce is cheaper) + the old shelved tiled kernel. ⚠ SCOPE: only
qwen3moe benefits — crown/27B/qwen3next are DeltaNet-chunked at T≤512 so their attn never leaves the parity
regime (their long-ctx cap is the SEPARATE §1 DeltaNet bug, env-revertable). crown PPL 6.4527 bit-exact
(additive). NEXT: optional — wire v2 into dense full-attn arches (Llama/Gemma dense also process full T).
PREVIOUS: 2026-06-22 (**CLEAN APPLES-TO-APPLES BENCHMARKS + PERF HARNESS + BOTTLENECK DIAGNOSIS.**
Post-oneAPI-2026.0 clean sweep corrected stale "wins": only Crown 35B is a clean both-axes win vs llama
SYCL; 27B wins prefill≤4K/loses decode; Coder+80B LOSE both — **the 80B "1.40× decode win" was STALE
(llama SYCL now 60.6 > our 56.55).** Built `tools/perf/` harness (roofline-classifier + HTML viz + PPL-
gated ledger). 5-lens domain analysis → 10 levers (`docs/optimization_roadmap_2026-06-22.md`). Tournament
round 1: **3 levers measured+REJECTED** (Q4-SoA decode parity [occupancy-bound not BW]; quantize-hoist
neutral [launch-count not the lever]; enabling the shelved tiled-FA2-prefill = REGRESSION 0.72× [bad impl]).
**⭐ PUBLISH-BLOCKER DIAGNOSED: long-ctx prefill collapse = `attn_naive_compute` (naive O(T²) attention)
= 70.9% of prefill @4096 → the fix is a GOOD flash-attention-2 prefill kernel (the existing tiled one is
inadequate). THAT IS THE NEXT BUILD.** Decode gap = launch+occupancy bound (only command-graph / split-K
can move it; micro-levers neutral). Full: `docs/benchmark_results_2026-06-22.md`, memory
`[[bench_clean_apples_2026-06-22]]`. crown PPL gate 6.4527 held bit-exact; oneAPI 2026.0 changed no numerics.
⚠ gemma26 MTP spec is NO LONGER bit-lossless (MoE near-tie routing) — re-validate before claiming.) PREVIOUS:
2026-06-21 (**✅ CODER-30B (qwen3moe) DECODE +13.8% via per-expert SoA-Q6 repack
(`c8dd418`) + CLEAN-BOX RE-BENCH SETTLED THE STANDINGS.** Post-reboot clean-box diagnosis of the two
"tune-before-publish" targets, apples-to-apples vs llama.cpp SYCL on the IDENTICAL files, same session:
**(1) Coder-30B-A3B Q4_K_M, 1×B70:** llama **pp512 1130 / tg 71.5**; ours (before) pp 407 / tg 43.5 →
**we LOSE BOTH axes** (the old "1.94× decode win" record was a STALE-llama artifact — DELETE that claim).
ROOT CAUSE + fix: qwen3moe's MoE Q6_K down experts read AoS (interleaved ql/qh, uncoalesced) = 25.7% of
decode; the crown already repacks experts to per-expert SoA (`ie/quant_soa.hpp`) — that path was scaffolded
in the qwen3moe kernels (`soa` arg) but never wired (soa=false hardcoded). Wired it (copy-not-hoist
`upload_expert_soa`, gated `moe_exps_soa_` default-ON / opt-out IE_NO_MOE_SOA), **DTYPE-CONDITIONAL**: SoA
ONLY Q6_K (down SoA 2.38×, 0.204→0.086 ms/call) — Q4_K REGRESSES under SoA (gate +38%, AoS already
coalesced; same as the dense Q4 finding) so Q4_K stays AoS. Result **decode 43.5→49.5 (+13.8%), prefill
614→630 (+2.6%)**, gap to llama 0.61×→**0.69× decode**. Crown 6.4527 bit-exact, Coder coherent. NEXT Coder
levers (still behind llama): the dense-attn fp16 projections (11.8%, Q4_K→fp16-expanded → native int-dot
like the dense path), prefill GEMM (llama coopmat 1.8× our onednn/dp4a).
**(1b) 27B SPEC tuned + the real wall identified (`70dbba4`,`1157489`):** swept K (verify cost grows
~linearly, acceptance saturates → LOW K wins): **K=2 1.12× / K=3 1.06× / K=4 0.98× (the old default was a
LOSS)**, lossless all K → default flipped to K=2. Verify attention: wired a KV-stationary fa2-decode-LOOP
(naive re-read KV T×; tiled fa2-PREFILL was 4× slower at T=4 — GPU-starved) = +4% verify, lossless-by-
construction, scales with ctx; new `ie-bench --kprofile-verify K` tool. **THE WALL (math): 27B plain decode
14.5 = 40% BW vs llama 23.2 = 64% BW on the IDENTICAL file. Spec is a 1.12× multiplier on OUR base; llama's
base is 1.6× ours → spec cannot close it. Beating llama on 27B = a dense-GEMV-BW project (40→64%, matching
llama MMVQ) OR EXL3-3bit (33% fewer bytes, runtime shipped, needs a 3bpw model), NOT spec tuning.** ⚠ spec
is NET-NEGATIVE on prose at any K>2 and only ~1.12-1.2× at K=2 — do NOT claim a universal 27B spec win.
**(2) 27B-dense Q4_K_M:** NEAR ITS DECODE CEILING. Plain 14.5 tok/s (q4k bulk 53% already at 64% BW =
llama-parity, NOT the lever). SPEC (`ie run --spec`, the ONLY way we run 27B) is **prompt-dependent: ~1.0×
prose (accept 2.35) / ~1.2× code (accept 2.7)** — the remembered "20+" was a code prompt. verify fwd(T=4)
=1.9× a decode (DeltaNet recurrence+ckpt don't amortize) → net ≈ ceiling. Levers all 4–12% w/ losslessness
risk = diminishing returns; the real 27B win is byte-reduction (EXL3-3bit, shipped) not GEMV grinding. NOTE:
K=4/6 spec showed `LOSSLESS:NO` (diverge >tok24), K=8 clean — a §1-DeltaNet flicker to recheck. Box CLEAN.
**Prior — HANDOFF: `docs/HANDOFF_2026-06-21_gemma-mtp-spec-decode.md`** (latest);
prior `docs/HANDOFF_2026-06-21_gemma-decode-bw-ceiling.md`, `docs/HANDOFF_2026-06-21_memory-planner-and-bench.md`.
**✅ GEMMA-4 MTP SPEC DECODE — COMPLETE + STRICTLY LOSSLESS ON BOTH SIZES, BEATS LLAMA (`a78e292`).**
The whole port landed: `read_attention_gemma` (read-only attn over the TARGET's shared KV — head has no wk/wv;
SWA head layers read target L(n-2), global L(n-1)) + `Gemma4Model` accessors + `all_logits` verify output
(batched SoA-Q6 lm_head, read-once) + `proj()` batched-verify path (small T routes SoA-Q4_0 projections through
`gemv_q4_0_soa_q8_batched` — the de-risk kernel, the perf unlock: 0.18×→1.4×) + `tools/ie-gemma4-spec`
draft/verify/accept loop (EAGLE recurrence h_next→inp_h, embed via TARGET tok_embd, fixed draft pos = shared-mem
semantics; NO DeltaNet → rollback implicit via absolute-position KV overwrite). Head fwd correct FIRST build
(acceptance ~0.76 > llama 0.61). **31B dense: LOSSLESS, 1.47× (≈24–25 vs 17 tok/s, beats llama 20.6).**
**26B-A4B MoE: LOSSLESS, 1.46× (≈67 vs 46 tok/s, beats llama ~55)** — needed the per-token verify-router fix
(`a78e292`: batched gemv_q_T(T) flipped a near-tie top-k expert vs the T==1 decode path → diverged ~1 tok/32;
per-token T==1 router in the verify range = byte-identical → strict lossless, GPU-router-fast). Draft head SoA-Q8
int-dot (`cb8bfeb`, 12.4→8.0 ms/round). At the ceiling (net ≈ accepted/(draft+verify) ≈ 2.7/1.8). crown 6.4527
bit-exact throughout. Both heads downloaded (31B `mtp-gemma-4-31B-it-Q8_0.gguf`, 26B `mtp-gemma-4-26B-A4B-it-Q8_0.gguf`).
**✅ PRODUCTIZED — `ie run --spec` now works for gemma4 (`892934a`):** `Gemma4Model::{load_mtp_head,spec_generate}`
+ engine kGemma4 spec-dispatch + CLI `--spec-head` (else auto-finds `mtp-*.gguf` next to the target). Validated
on both sizes (auto-find, coherent output via the product path); default `ie run` byte-unchanged; 26B needs
`IE_GEMMA4_FUSED_MOE=1 IE_GEMMA4_MOE_SOA=1`. NEXT: clean-box re-bench (see `docs/BENCHMARK_METHODOLOGY.md`) +
tune 27B-dense decode (the one weak spot — loses to llama even with spec). Prior milestone —
**🟢 MTP SPEC-DECODE PORT — 3/4 PIECES SHIPPED, forward+loop remain.** De-risk passed (verify amortizes on B70:
T=4 ≈ 1× decode). Committed: verify kernel (`a962423`), `hidden_pre_norm` (`c9f38e7`), `gemma4-assistant` head
loader (`9e34f17`, loads the real 491MB Q8_0 head — config exact, LOAD OK). Remaining = EAGLE head forward
(Q8_0 GEMV + read-only attn over target L58/L59 KV) + draft/verify/accept loop (rollback=KV truncate). Target
~1.4–2× → 24–34 tok/s vs llama 20.6, lossless. Full spec/plan in the handoff. crown 6.4527 bit-exact throughout.
**🟢 GEMMA-4 MTP SPEC DECODE — DE-RISK PASSED, PORT GREEN-LIT (`a962423`):** Gemma 4 ships an official 930M MTP
assistant draft head (`mtp-gemma-4-31B-it.gguf`, arch `gemma4-assistant`, 4L). llama.cpp runs it on B70 at only
20.6 tok/s / 61% accept (+8% — its Vulkan GEMM can't amortize the verify; the "3×" is datacenter marketing). BUT
our new BW-optimal `gemv_q4_0_soa_q8_batched` verify kernel AMORTIZES on B70: **T=4 verify costs 0.83–1.25× a
single decode** (per-token 3.7–5.4× cheaper T1→T8), bit-identical to the decode kernel. ⇒ with 61% acceptance, a
realistic **~1.4–2× decode (24–34 tok/s) that BEATS llama's 20.6** — the port is worth it. Foundation committed
(kernel + `ie-gemv-q4-0-batched-bench`); crown 6.4527 bit-exact. REMAINING (multi-day, own session): expose
`hidden_pre_norm`; `gemma4-assistant` head loader; EAGLE head forward (Q-only attn sharing the target KV; no
DeltaNet → rollback = KV truncation); draft/verify/accept loop (template `tools/ie_qwen35_spec.cpp`). Full plan in
`memory/project_gemma4_arch`. **Prior milestone —**
**✅ GEMMA4 DECODE AT ITS BW CEILING — 2 measured wins banked, all other levers killed by measurement (`baba378`):**
A roofline diagnostic (ALU stripped, identical loads) PROVES the 31B decode GEMV is **~91% memory-bound**
(+8.6% ceiling from removing ALL arithmetic) and already at ~75–80% of warm-step peak BW. So the prior
session's headline **interleaved-column repack is a DEAD END** (its "32 scattered reads" premise is false —
each sub-group already issues a coalesced 256 B SIMD load), as are the fp16 micro-opt and per-layer
`q.wait()` removal (both measured NEUTRAL). Shipped two bit-exact, gemma4-only wins: **(1) `gemv_q4_0_soa`
NPWG 32→64** (swept best; env read hoisted out of the 410×/token hot path) and **(2) `gemv_q4_0_soa_q8_multi`**
— q/k/v (shared attn_norm) and gate/up (shared ffn_norm) fused into ONE quant + ONE multi-bank launch (tiny
GQA k/v columns ride a full grid; zero extra VRAM; bit-identical tokens; opt-out `IE_GEMMA4_NO_FUSE_QKV`).
Combined **~+2–3%** (16.9→~17.2 tg128). Validated: crown **6.4527 bit-exact**, **ctest 30/30**, 31B+26B
greedy "Paris." NOT thermal (GPU pinned 2800 MHz); long-run decay = attention O(seq). **gemma4-31B decode is
at its bandwidth ceiling, ~10% behind llama-Vulkan 19.0 — not worth a multi-week campaign; publish leading
with the DeltaNet moat.** Only remaining lever = lm_head `gemv_q6_soa` 57→80% BW (~+1.5%, but SHARED with the
crown gate → defer). Full analysis + dead-end list in the handoff. **Prior —**
**✅ GEMMA-4 SoA-ONLY RESIDENT LAYOUT (2026-06-21): fixes 31B single-card OOM.** The fast Q4_0 path
kept BOTH an AoS *and* a SoA copy of every dense projection (AoS for prefill onednn-dequant, SoA for
decode gemv) → ~2× projection VRAM. On 31B that hit **32.6 GB peak → `DEVICE_LOST` crash** on a 32 GB
B70. Fix: build SoA streams from the HOST GGUF (`repack_q4_0_to_soa` reads `t->data`) and DROP the AoS
device copy; new `dequant_q4_0_soa_to_Bt` feeds the onednn prefill GEMM bit-identically. **31B now loads
at 20.2 GB (fits 24 GB), 26B at 16.3 GB; both greedy-coherent; crown 6.4527 bit-exact.** Default-on;
`IE_GEMMA4_KEEP_AOS=1` restores both copies for dp4a/xmx A/B.
**✅ VRAM-AWARE PLACEMENT PLANNER (2026-06-21, `8483c36`): the engine now SENSES per-card VRAM at load
and AUTO-PICKS single vs multi-GPU** — `plan_placement()` estimates footprint (Σ tensor bytes + per-arch
KV reserve + 1.5 GB ws) vs `global_mem_size×0.90`; `--gpus 0` (new CLI default) = auto, `--gpus N` forces.
Validated through `ie run` (no `--gpus`): gemma4-31B → `[mem-plan] fits 1 card (17.9/28.7 GB)` single-GPU;
80B Coder-Next → `split across 2 GPUs`, coherent. **⚠ CRITICAL FIX (`b336585`): the GPU name filter
`"0xe223"` matched ZERO current-driver device names → ALL multi-GPU was silently broken (fleet had no
fallback). Now `"B70"` + `IE_GPU_FILTER` env override** ([[project_gpu_filter_drift]]).
**Tier C (universal SoA-only) = NO-OP by investigation**: gemma4 was the only arch with a redundant DEVICE
AoS+SoA double; qwen35_dense already sets `dq={}` when SoA built, crown uploads only SoA from host mmap,
dense/qwen3moe are single-copy. **Tier B (gemma4 multi-GPU) DEFERRED** — gemma4-31B fits one card with
~11 GB headroom and tiny SWA KV, so it never needs 2 cards in practice; not worth a new layer-split class
at publishing time. crown gate 6.4527 bit-exact throughout.
**⚖ GEMMA vs CURRENT llama = CONTESTED (re-benched CLEAN vs llama-Vulkan b8902, 2026-06-21 — the OLD
"beats both axes" numbers were vs a STALE llama build):** 31B prefill ~262 (T512) vs **254.9** ≈ parity;
31B decode (tg128 sustained) **16.9** vs llama **19.0** = llama +12%; 26B prefill 585 vs **1064** = llama
+82%; 26B decode 42–47 vs ~26 = ours wins. **NO regression on our side** — the earlier "3 tok/s" was a
THRASHED box (swap full, 2.4× run variance); the "16% deficit" was burst (single-step 20.4 ≈ historical
20.5) vs 128-step sustained (16.9, heat-soak). llama's Vulkan simply got faster. **SoA-only change
perf-NEUTRAL** (26B A/B + clean 31B); `q.wait()`-removal reverted (no benefit); native-fp16 gemv micro-opt
tried+reverted (IGC already optimal). **DECODE LEVER (not yet done):** `gemv_q4_0_soa` = 81.9% of decode @
~67% BW — an **interleaved-column SoA repack** (coalesce a WG's 32 K/2-apart columns) targets 67→80% BW →
~+12% decode → beats llama; cross-arch. Prefill-26B gap = llama's Vulkan KHR_coopmat matrix GEMM vs our
onednn/dp4a. Roadmap in handoff (`15fa7e4`). LESSON: never bench a thrashed box (verify swap=0 + GPU
idle-clocked); compare vs CURRENT llama, not memory. **Prior —**
**✅ GEMMA-4 31B DENSE: BEATS llama on BOTH axes, ZERO new code — pp 367 vs llama-Vulkan 203 (1.81×),
tg 20.5 vs 15.35 (1.33×), 1×B70, same Q4_0 GGUF.** Ran first-try on the existing gemma4 path (dense FFN every
layer; MoE additive only when a router is present); greedy coherent ("Eiffel Tower → Paris, France. It is one
of the most famous landmarks … in the world"). Fast config `IE_GEMMA4_INTDOT_PROJ=1 IE_GEMMA4_ONEDNN=1`
(SoA-Q4_0 decode default-on). **Bar caveat:** llama SYCL is CRIPPLED on dense gemma4 (59/6 tok/s — falls to a
slow path; only ~18% BW), so the honest bar is llama's best backend = Vulkan. **Decode is gemv BW/ALU-bound
(~59% BW = 361/608 GB/s); the obvious host-sync lever (removing redundant per-layer `q.wait()` on the
in-order queue) measured NEUTRAL (20.13 vs 20.19) → reverted.** Further dense-decode gains need a cross-arch
SoA-gemv rewrite (ALU-co-limited), a separate milestone. QAT-Q4_0 so PPL is high-but-meaningless (same as 26B
— validate by greedy, not PPL). crown 6.4527 untouched (no engine change). **NEXT: 12B (same path/kernels);
then wire gemma4 into ie::Engine + `<start_of_turn>`; cross-arch SoA-gemv decode kernel.** **Prior —**
**✅ GEMMA-4 26B-A4B: DECODE BEATS llama, PREFILL 1.17× behind — pp 708 tok/s,
tg ~55-58 (llama 54.75 ✅).** Full SoA-Q4_0 W4A8 path closed the
decode gap: the MoE experts (42% of decode, ~13GB) were repacked AoS→SoA in-place (no doubling) via
`moe_prefill_proj_q4_0_soa_q8` (bit-identical dp4a fold, lane-coalesced 16B reads) — decode 48.9→54.8-58.1
= AT/ABOVE llama 54.75, AND prefill 655→708 (SoA coalescing helps prefill too; gap to llama 828 = 1.17×).
Built bottom-up this session: weight-stationary fp16/oneDNN dense projections (prefill 1.96×, `b0a8a18`) +
SoA-Q4_0 dense decode (`d3973a9`) + SoA-Q6 lm_head (`decd0dc`) + SoA-Q4_0 MoE (`77ed888`). Measured
negatives kept opt-in: naive XMX per-M-tile re-dequant, MoE-XMX (small-M starve). **Fast config =
`IE_GEMMA4_FUSED_MOE=1 IE_GEMMA4_INTDOT_PROJ=1 IE_GEMMA4_ONEDNN=1 IE_GEMMA4_MOE_SOA=1`** (gemma4 still
tool-only; flip defaults when wired into ie::Engine — MOE_SOA needs FUSED, the unfused path still uses AoS).
**⚠️ PPL — NOT an engine bug, and the prior "beats llama" claim is RETRACTED (was apples-to-oranges).**
DEFINITIVE (2026-06-21, same wikitext-2 + same GGUF): **llama.cpp reference itself = PPL 1178** (20×512,
strided); ours = 2880 (first-512 streaming-T1). BOTH land in the pathological ~1000-3000 band → this is a
property of the **QAT-Q4_0 instruct GGUF**, NOT our engine. Mechanism: QAT preserves argmax (greedy is
coherent, crown bit-exact) but the Q4_0 output projection miscalibrates logit MAGNITUDES → flat softmax →
PPL ~1000+ despite correct predictions; feeding an instruct+multimodal model raw documents adds a 3-5×
mismatch on top. **⇒ PPL is NOT a meaningful quality metric for THIS GGUF — use greedy/task accuracy.** The
2.4× ours-vs-llama gap is measurement protocol (their strided back-window scoring vs our full single-window
incl. high-NLL opening tokens), not a forward error. crown 6.4527 bit-exact throughout. Session arc: **pp 114→708 (6.2×), tg
32→~56 (1.75×).** Memory `project_gemma4_arch`. **NEXT: wire gemma4 into ie::Engine (+`<start_of_turn>`,
flip defaults); 31B dense + 12B (same Q4_0 SoA kernels apply); dedupe dense AoS+SoA.** **Prior —**

**Prior gemma4 milestone — at the bar pp ~655 / tg ~45.** Big perf session on the gemma4 26B-A4B MoE
(QAT Q4_0, 1×B70). **PREFILL 114→655 tok/s (5.7×; gap to llama SYCL 828 = 7.2×→1.26×):** the lever was
weight-stationary fp16/oneDNN projections (`dequant_q4_0_to_Bt`→`gemm_fp16_onednn`, 1.96×, `b0a8a18`) on
top of the int-dot W4A8 projection groundwork; MEASURED NEGATIVES (kept opt-in): naive XMX per-M-tile
re-dequant (`gemm_q4_0_xmx`, slower than dp4a) and MoE-XMX (`moe_prefill_proj_q4_0_xmx`, starves at ~30
rows/expert). **DECODE 32→~45 tok/s:** SoA-Q4_0 W4A8 decode GEMV (`gemv_q4_0_soa_q8`+`repack_q4_0_to_soa`,
dense projections 1.21×, `d3973a9`) + SoA-Q6 lm_head (`decd0dc`); both default-ON, opt-out
IE_GEMMA4_NO_Q4_SOA. **MoE experts (42% of decode, ~13GB AoS) = the remaining decode lever** — needs an
expert-SoA refactor (can't double to SoA). **⚠️ PPL SCARE RESOLVED — NO bug:** the old "engine 2× llama
PPL" was a BROKEN comparison (our streaming-full-context vs llama's windowed-chunk AVERAGE); correct
apples-to-apples (llama `-c N` chunk[1] vs our streaming `--max-tokens N`) = **256-tok ours 2615 vs llama
2883, we're BETTER**; ruled out softcap/tokenizer/shared_kv/KV-path/int-dot. Fast config =
`IE_GEMMA4_FUSED_MOE=1 IE_GEMMA4_INTDOT_PROJ=1 IE_GEMMA4_ONEDNN=1` (gemma4 still tool-only; flip defaults
when wired into ie::Engine). Tools: `ie-gemma4-gen … {bench,bench-decode,profile[ profile-decode]}`. crown
6.4527 bit-exact throughout. Memory `project_gemma4_arch`. **NEXT: MoE-decode expert-SoA; wire gemma4 into
ie::Engine (+`<start_of_turn>`); 31B dense + 12B (same Q4_0 kernels apply).** **Prior banner —**

**✅ GEMMA-4 INT-DOT W4A8 PREFILL PROJECTIONS BUILT — prefill 2.61× (116→303
tok/s), gap to llama SYCL 7.2×→2.74× = the "credible 2-3×" breadth target.** Routed the dense/attn Q4_0
projections (q/k/v/o + shared-FFN) through int-dot W4A8, the lever the prior banner localized. Two commits
(`a036018` Phase A: existing kernel E=1 for K≤3072 = 1.87×/214; `4d48007` Phase B: new split-K
`gemm_q4_0_q8` in `src/ops/gemv_q4_0.cpp`, tiles K in 96-block chunks → handles ANY K, so the large-K
**o-proj** (K=NQ≤8192, was stuck on the per-token gemv fallback) + all 31B-dense projections now run int-dot
= 2.61×/303). **PREFILL-ONLY (`T>1`):** at T==1 the batched kernel is −16% tg, decode keeps the per-token
gemv (decode 34.6 unchanged). Gated **`IE_GEMMA4_INTDOT_PROJ`** (opt-in; pairs with `IE_GEMMA4_FUSED_MOE=1`
which reproduces the 114.4 baseline). **Validated 26B-A4B QAT Q4_0, 1×B70:** greedy "Paris" + argmax
bit-identical (669) across 477 prefill tokens; reprefill PPL OFF 2311 vs ON 2170 (int-dot slightly better,
within the instruct-on-prose noise floor); crown PPL 6.4527 bit-exact; default path byte-identical. Also
**fixed the decode-bench tooling gap**: `ie-gemma4-gen … bench-decode` (real tg — ie-bench mis-detects
gemma4 as qwen35moe) → decode **34.6 vs llama 54.75 = 1.58× behind** (now the bigger relative gap; MoE
decode is BW-bound = the next lever). **NEXT for gemma4:** wire into `ie::Engine` (kGemma4 branch +
`<start_of_turn>` template) and flip `INTDOT_PROJ`/`FUSED_MOE` default-on there; validate 31B dense (split-K
kernel already covers its K); MoE decode GEMV for the decode gap. Memory `project_gemma4_arch`.
**Prior this session —**

**✅ MTP SELF-SPECULATIVE DECODE SHIPPED as `ie run --spec`** — net lossless win. Single-card 27B-Q4
via the native `blk.64` MTP head: `ie-qwen35-spec` K=4 → **cyber 14.34→17.14 = 1.20× LOSSLESS**, code
15.25→16.38 = 1.07×, 0 rollback re-forwards. Built bottom-up (all additive, crown 6.4527 bit-exact):
STEP 0 acceptance 0.5669 (`bac465e`) → 1a verify primitive all-pos-logits+hidden (`ba74240`) → 1b/2
lossless loop proven (`3638f65`) → 3 batched-T int-dot verify kernels (`5046ac2`, verify 1071→120 ms 8.8×)
→ 4 per-position DeltaNet checkpointing (`f8783f6`, kills the rollback re-forward + FIXES the §1
non-determinism lossless flips). Prefill/decode unchanged (default-off ckpt). **NEXT = wire into
`Engine::generate`** (`spec_` path, additive — the only remaining productionization). Ceiling ~1.2-1.5×
(verify W6A8 ALU-co-limit + MTP acceptance 2.6/verify); K=4 optimal. The batched-T int-dot GEMV doubles
as small-batch serving infra. Spec/memory: `project_mtp_spec_decode`. **Prior banner —**

**▶ SPEC-DECODE via the NATIVE MTP head — STEP 0 + 1a DONE, building the
lossless ~1.5–2× decode lever.** Since the dense Q4 kernel is at llama-parity BW (below), the only way
past `BW/bytes` is amortization → spec-decode. The GGUF ships a trained NextN/MTP head (`blk.64`) = a
built-in EAGLE self-draft. **STEP 0 (de-risk) PASSED:** `ie-qwen35-mtp-accept` teacher-forced
**acceptance 0.5669** (commit `bac465e`, concat `[enorm(e);hnorm(h)]` A/B-proven, crown bit-exact,
single-card viable). **STEP 1a DONE:** verify primitive — `Qwen35DenseModel::forward` gained default-null
`all_logits` ([T,vocab]) + `hidden_pre_norm` (h_i) params, default path byte-identical, crown 6.4527
(commit `ba74240`). **NEXT = STEP 1b** (DeltaNet/KV rollback — the correctness gate; snapshot+re-run
correct-first, design in the spec) **→ STEP 2** (engine spec loop: MTP drafts K, verify forward(T=K)
all-pos, accept argmax-match prefix, rollback) **→ STEP 3** (LOSSLESS validation: greedy-spec ==
plain-greedy token-for-token). Cheap bankable extras (separate): SYCL-Graph (~+5%, NOT used today),
fuse gate+up GEMVs. Spec: `docs/superpowers/specs/2026-06-20-qwen35-27b-speculative-decoding.md`;
memory `project_mtp_spec_decode`. **Prior banner —**

**⛔ DENSE-Q4 DECODE IS ALREADY AT llama PARITY — stop tuning the Q4
kernel.** Settled apples-to-apples (built llama.cpp SYCL): on the IDENTICAL 27B-Q4_K_M, 1×B70, we WIN
prefill (348.6 vs 287.7, 1.21×) and trail decode (15.2 vs 23.2). Chased the decode gap with TWO gated
kernel attempts, BOTH measured + REVERTED: (1) int-dot `gemv_q4_K_q8` wire-up = NEUTRAL; (2) full
SoA-Q4 repack kernel (`gemv_q4_soa`, mirroring the Q6-SoA + llama's reorder-MMVQ) = **+6% decode but
−24% prefill → net-negative.** The KEY finding: the AoS `gemv_q4_K` on the big FFN tensors is **already
389 GB/s = 64% of 608 = llama's level** ("Q4 at 46%" was the OVERALL-effective rate incl. F16-expansion
bloat + overhead, NOT the kernel). So you do NOT beat llama by grinding the Q4 GEMV — it's done. The
residual dense gap is BYTES, not %BW: F16-expansion of Q8_0/Q5_K (~10% extra bytes, but gemv_fp16 is
already ~82% BW so killing it only buys ~+3%), Q6 at ~60%, and the fundamental 18 GiB/token of a dense
model. **REAL "beat-llama" decode levers are byte-reduction/amortization: spec-decode (~2× lossless,
the multiplier — NOW the move since the kernel is tuned to parity) + EXL3 3-bit single-card (33% fewer
bytes, ALREADY SHIPPED, needs a 3bpw model + quality opt-in).** Crown 6.4527 bit-exact throughout; tree
clean at HEAD. Memory `dense_q4_parity_2026-06-20`. **Prior banner —**

**✅ PREFILL CHUNK RAISED 256→512 for crown + 27B — the §1 DeltaNet
non-determinism is NOT REPRODUCIBLE on our stack.** After updating NEO 26.14→26.18 (GuC/kernel
unchanged = the diagnosis stack), the bug did not fire: `ie-bug-monitor` 1024-iter ×3 (24 chains) =
0 divergence on BOTH drivers; crown PPL `--prefill-chunk 512` ×3 bit-identical (16.27, no collapse),
27B-Q6 ×3 bit-identical (15.68); crown `ie run` 563-tok → 701.9 tok/s prefill coherent. So crown
(`kQwen35Moe`) + 27B (`kQwen35Dense`) now run T≤512 (env-revert `IE_QWEN35_PREFILL_CHUNK`), joining
qwen3next → **closes the long-context/pp512 weakness** (~1.1× long-prompt prefill). NOT a "26.18
fixed it" claim (clean on 26.14 too) — "no longer reproducible." Crown PPL 6.4527 bit-exact. T≥1024
still unvalidated. Docs: `docs/known_bugs.md` §1, `CLAUDE.md`. **Prior banner —**

**⛔ HYBRID-TP PHASE 0 BUILT → MEASURED → NO-GO — tensor-parallel
does NOT beat the layer-split on this no-P2P board** (commit `e622a3a`; spec
`docs/superpowers/specs/2026-06-20-hybrid-tp-SCOPING.md` §7). New additive `Qwen35TpModel`
(`include/ie/qwen35_tp.{hpp}`, `src/model/qwen35_tp.cpp`): FFN-slice TP — dense SwiGLU sharded
(gate/up column, down row, 1 all-reduce/layer); gated full-attn + gated-DeltaNet REPLICATED full
on both cards (bit-identical residual, no all-reduce). Wired `IE_QWEN35_TP=1` + `--gpus 2` →
`Qwen35TpModel`. **Correctness GREEN:** cyber Q8 2×B70 → "Paris." + coherent 200-word essay;
card0 19.47 GB / card1 16.77 GB. **Perf NO-GO:** clean warm decode (331 tok, profiler off) =
**12.6 tok/s, at/below the ~13 baseline** (profiler-on 12.4 → wait-bracketing cost only ~2%). Decode profiler (`IE_QWEN35_TP_PROFILE`, wait-bracketed → read the breakdown):
**all-reduce ~32%** (the no-P2P host bounce — now the LARGEST phase, ≈ the entire FFN it splits) /
FFN ~28% / DeltaNet ~28% (replicated, redundant) / attn ~9%. **The documented #1 risk
materialized:** ONE all-reduce/layer costs as much as the FFN it parallelizes; Phases 1/2 would each
ADD an all-reduce (128/token) → MORE host bounces, not faster. **DECISION: STOP at Phase 0** (per
the §5 GO/NO-GO gate) — do NOT build the gated-attn / DeltaNet v-head shards. Phase 0 committed as a
measured negative + reusable scaffold (off by default; only useful if a P2P board or a
batched/overlapped/fused-2-block all-reduce changes the math). **Faster-27B-decode = the shipped
Q6_K SoA kernel (single-card ~13, zero all-reduce) and spec-decode (~2×, lossless), NOT TP here.**
`Qwen35DenseModel` / `Qwen35SplitModel` / `DenseModelTP` / crown UNTOUCHED; crown PPL 6.4527
bit-exact (additive). Prior banner below. ⛔ —

**Prior — ✅ FAST Q6_K DECODE KERNEL SHIPPED — 27B Q6 single-GPU decode
5.17 → 12.9 tok/s = 2.50×** (commits `6292b4c`/`233a55d`/`b218d20`; spec
`docs/superpowers/specs/2026-06-20-q6k-fast-decode-kernel.md`). New `src/ops/gemv_q6_soa.cpp`: a
load-time **SoA-Q6 repack** (block_q6_K → per-column natural-order bit-plane streams: q6_lo 4-bit,
q6_hi 2-bit, per-16 int8 scales, fp16 super-scale — **stays ~6.5 bpw / ~22 GB**, single-card) +
**`gemv_q6_soa_q8`** int-dot **W6A8** decode GEMV (split-K, Q8_1 act staged in SLM, mirrors the
proven `gemv_q8_0_soa_q8` ~80% template). Wired into `Qwen35DenseModel` for ALL Q6_K projections
(ffn_gate/up/down, attn_q, attn_qkv, attn_gate, ssm_out, lm_head); AoS Q6_K NOT uploaded on the SoA
branch (no memory doubling). `gemv_q6_soa` = 0.18 ms/call (vs old `gemv_q6k_med` 0.41) = ~344 GB/s =
**57% of 608** (the Q8-SoA template hits ~80%; gap = 6-bit-reconstruct ALU — documented follow-on
headroom to ~17 tok/s). Default-ON, opt-out `IE_QWEN35_NO_Q6_SOA=1`. **GATES:** crown PPL **6.4527
bit-exact**; Q6 coherent ("Paris."); 27B Q6 PPL **5.5578 SoA vs 5.5577 AoS-oracle** = Δ0.00002%
(int-dot numerically indistinguishable). Crown/Q8-SoA/`DenseModelTP`/`Qwen35SplitModel` UNTOUCHED.
**This changes the single-GPU-vs-split tradeoff:** Q6 single-GPU decode 12.9 now ≈ the Q8 2-card split's
~13 — single-card Q6 is viable as a daily driver without needing 2 cards. Prior banner below. ⚠️⚠️ —

**⚠️⚠️ THE "ENGINE-WIDE 81→13 REGRESSION" WAS NOT REAL.**
Re-benched the crown CLEAN: `ie-bench --gguf <crown-Q4_K_M> --prefill 1 --decode 64 --warmup 20`
(profiling OFF, current code) → **TG 80.4 tok/s** = its historical 81. **The crown never regressed.**
The prior "crown 3.6→13.7" numbers were BAD MEASUREMENTS (cold/under-warmed/`ie run`) that snowballed
into a fictional engine-wide-decode-bug narrative — disregard it. **ALL models are at expected speed:**
crown **80.4**, coder-next/qwen3next **~47–50** (= hist 51.8, heat), 27B dense Q8 split **~13**
(CORRECT — dense reads 28.6 GB/token = bandwidth-bound at 608 GB/s/card; the 81 was the SPARSE crown
reading ~1.7 GB/token, NEVER comparable). The `a36c138` profiling-opt-in is still valid + harmless,
but its "3.8×" was vs a broken baseline. **METHOD LESSON: before declaring a regression, re-bench the
SAME model with `ie-bench` + adequate warmup; `ie run` and cold runs lie.**

**THIS SESSION (later, 2026-06-20) — two deliverables, both done + gated:**
(A) **Q6_K ssm-proj loader fix** (`qwen35_dense.cpp` + `qwen35_split.cpp`, additive `dequant_q6_K_row`
helper + Q6_K branch in `upload_f32_proj_fp16`; copy-not-hoist, hot header untouched). Unblocks
`*-Q6_K` GGUFs on the qwen35 path (was hard-failing `ssm proj: expected F32 or Q8_0`). VERIFIED: Q6
27B loads + coherent single-GPU ("Paris"); crown PPL **6.4527 bit-exact**. **⚠ BUT Q6 single-GPU
decode = 5.0 tok/s — SLOWER than the Q8 2-card split's 13** (the 27B dense Q6_K decode GEMV is
unoptimized, ~18% BW; the split rides the fast int-dot Q8-SoA kernel). So "fewer bytes = faster" did
NOT hold; the Q8 split is better as-is. To make Q6 pay off needs a fast Q6_K decode kernel (or
repack→Q8-SoA, which then needs 2 cards anyway).
(B) **Hybrid-TP SCOPED** (`docs/superpowers/specs/2026-06-20-hybrid-tp-SCOPING.md`). Foundation real
(`DenseModelTP` + `all_reduce_sum_fp16` exist). **Corrected the base spec's profile** with measured
data: FFN **50%** / full-attn **29%** / DeltaNet **21%** (spec assumed 65/10/22 → attn TP is 3× more
valuable than it thought). Realistic payoff ~1.4× (no-P2P, 128 all-reduces/token) → 13→~18 tok/s.
Phased de-risk: FFN-slice first (go/no-go on all-reduce cost) → attn head-shard → DeltaNet v-head
shard (novel) → wire+validate. ~1 week.

**HARDWARE (owner-confirmed):** 2× B70 = **608 GB/s VRAM each**, on the ProArt board at **PCIe Gen4
x8/x8** (sysfs reports idle-downtrained `2.5GT/s x1` — ignore it). PCIe is NOT the layer-split decode
bottleneck (cross-card = ~10 KB/token = sub-µs; profiler `head+bounce` <1%; coder-next proves it at
~50 on the same split). The split's real cost is SERIALIZATION (cards run sequentially, not
concurrently) — which is exactly what TP (B) fixes.

**OPEN THREADS (current — full detail in `docs/HANDOFF_2026-06-20_q6kernel-tp-chunklift-publicprep.md`):**
(1) **Spec-decode** (`...speculative-decoding.md`, ~2× lossless) — the main remaining decode lever
(Q6 kernel SHIPPED 5→12.8; hybrid-TP NO-GO). (2) **Dense-Q4 parity — APPLES-TO-APPLES SETTLED 2026-06-20** (built llama.cpp SYCL fresh:
`~/llama.cpp/build-sycl`, icpx/oneAPI, HEAD fdc3db9b6/9598, `GGML_SYCL=ON`; `llama-bench` on the
IDENTICAL `Qwen3.6-27B-Q4_K_M`, 1×B70, `-ngl 99`):

| 27B-Q4_K_M 1×B70 | llama.cpp SYCL | ours | winner |
|---|---|---|---|
| prefill pp512 | 287.7 | **348.6** | **us 1.21×** |
| decode tg | **23.2** | 15.2 | **llama 1.53×** |

**The dense DECODE gap is REAL** (not just a quant mismatch — that only exaggerated it). **We WIN
prefill.** BW: llama **64%** vs us **42%** of the 36 tok/s ceiling → **a faster Q4 decode path IS
worth building** (revises the earlier "gemv_q4_soa not needed"). `gemv_q4_K_q8` measures ~53% BW
alone, so the end-to-end 42% deficit is split between the Q4 GEMV AND the hybrid DeltaNet/attn decode
overhead — PROFILE-localized before choosing the fix. **ROOT-CAUSED (negative result, do not re-run):**
hypothesis "wire the int-dot `gemv_q4_K_q8` crown uses (like the Q6-SoA win)" was BUILT + A/B'd
(new-old-new 15.1/14.9/15.1 = **NEUTRAL**) and **REVERTED** — int-dot `gemv_q4k_q8` (0.1273 ms/call) ≈
plain `gemv_q4k` (0.1289). Crown PPL 6.4527 bit-exact throughout. **The gap is BROAD GEMV bandwidth
efficiency (~44–48% vs llama ~64% of 608 GB/s), GPU-bound (65 ms GPU ≈ 65 ms wall — NOT
launch-bound), not one slow kernel.** Two real levers, both modest + real kernel work: (a) native
Q5_K/Q8_0 decode GEMV to kill the F16-expansion of attn_k/v/output + ssm_out (~13% bytes → est 15.2→17);
(b) match llama's GEMV BW 48%→64% (deep tuning). **Dense decode parity is a kernel-optimization
project, not a wire-up — de-prioritize vs spec-decode (~2× lossless, the better decode lever) unless it
becomes a publish blocker.** We still WIN prefill (1.21×) + BEAT llama on the DeltaNet arches (the
headline). Next: OpenVINO GenAI baseline. See memory `dense_q4_parity_2026-06-20`. (3) ✅ **Q4_K dense re-bench DONE** — 4B-Q4_K =
**79.3 tok/s** (the "4B 2.55" WAS a profiling artifact; generic dense Q4 path healthy at all sizes). (4) **T≥1024 chunk** validation (512 done; 1024 promising, needs repeatability).
(5) **Public cleanup:** profiling-dependent tools (`ie-bug-monitor`) crash without `IE_QUEUE_PROFILING=1`;
gitignore `.claude/agent-memory/`. (6) `ie-bench --gpus N` still missing (used `IE_QWEN35_PROFILE`).

**Prior banner — Qwen3.6-27B Q8 PACKED PATH DONE (Phase 2, `6696b6b`):** weights PACKED as Q8_0-SoA
(int8 qs + fp16 d, de-interleaved, BIT-EXACT) not dequant-to-F16 → **14.28 GB/card** (= true 28.56 GB
Q8 size). `sgemv` decode = int-dot W8A8 (`gemv_q8_0_soa_q8`), prefill = SoA→fp16 + `gemm_fp16`. All in
`qwen35_split.{hpp,cpp}`; single-GPU `Qwen35DenseModel` UNTOUCHED. Coherent on cyber Q8, crown bit-exact.
Prior: **Phase 1 layer-split (`a9704a1`).**
Older banner —
**Qwen3.6-27B MULTI-GPU LAYER-SPLIT DONE (Phase 1).** New additive `Qwen35SplitModel` (`include/ie/qwen35_split.hpp` +
`src/model/qwen35_split.cpp`) mirrors the 80B `Qwen3NextModel` fleet scaffold (per-card
weights/KV/DeltaNet, device-by-device forward, boundary residual copy, host-logits bounce)
with the **dense SwiGLU FFN + 27B DeltaNet conventions** (separate ssm_alpha/beta N-padded
projections, TILE repeat 16→48). Single-GPU `Qwen35DenseModel` byte-for-byte UNTOUCHED → zero
regression. Engine routes `kQwen35Dense` with `--gpus>1` to the split (`qwen35_split_` flag/
member + vocab/forward_step branches); `--gpus 1` unchanged. **LANDMINE fixed:** oneDNN DISABLED
on the split (`gemm_onednn.cpp:ctx_for` caches ONE static engine bound to card 0's queue →
card-1 buffers there = `UR_RESULT_ERROR_DEVICE_LOST`; forces bit-exact `gemv_fp16`/card). Runs
**Qwen3.6-27B-obliterated-cyber Q8_0** (27 GB → F16-dequant ~54 GB → ~27 GB/card on 2×B70):
`ie run --gpus 2` → "capital of France" → **"Paris"** coherent. Crown `ie-perplexity`
**6.4527 bit-exact**. Commit `a9704a1`. **NEXT = Phase 2: native `gemv_q8_0_soa_q8` end-to-end
→ ~13.5 GB/card → big context** (spec `docs/superpowers/specs/2026-06-20-qwen35-27b-multigpu-split.md`).
Prior: **PROMPT/KV CACHE WIRED + F16/Q8 DISPATCH.** **Committed 2026-06-19:** (a) **F16 + Q8_0 projection dispatch**
(`qwen36.cpp`, commit `0a0bcbb`) — additive gated branches in the crown loader (`upload_quant`) +
decode (`gemv_q`) so it can run **F16** (host-transpose [N,K]→[K,N], reuse `gemv_fp16`) and **Q8_0**
(dequant→fp16 at load via `dequant_q8_0_to_Bt`) projection weights; a normal crown GGUF has none so
the branches never fire — crown **6.4527 / NLL 1.864495 bit-exact**. Unblocks loading fine-tuned
models whose ~1% delta Q4_K rounds away (the trainer's projection-precision case). (b) **Prompt/KV
cache wired into `Engine::generate`** (commit `2c6a535`) — the keystone; the PR#3 `PrefixCache`
(KV+DeltaNet snapshot/restore + token-trie) integrated, **gated default-OFF** (`IE_PROMPT_CACHE=1`
or `EngineOptions.prompt_cache`), **crown (`kQwen35Moe`) single-GPU only**. Flow: match →
restore(leave ≥1 tok for fresh logits) → prefill suffix → snapshot. **⚠ CRITICAL GOTCHA (recurs on
the fleet path): the reasoning template injects an empty `<think>\n\n</think>` block (tokens
248068/248069) into the generation prompt, which POISONS the cached prefix → every multi-turn
request misses.** Fixed by snapshotting at the **STABLE conversation boundary** (`cache_prefix_len`
= the prompt rendered with `add_generation_prompt=false`, computed in `chat()`) via a split prefill.
Validated: canonical **6.4527 bit-exact OFF** (before+after the prefill restructure), ctest **29/29**,
identical-req 21 cached 165→11ms (**15×**), multi-turn turn-2 **cached=15** output bit-identical to
fresh. `GenerateResult.cached_tokens` + server `[gen]` log. Still default-OFF (flip-to-default-ON is
a small follow-on). (c) **✅ 80B FLEET PROMPT/KV CACHE DONE + GPU-VALIDATED (this session, NOT yet
committed at the time of this line — see git log):** the prompt cache now extends to the Qwen3-Next-80B
`next_` layer-split path. New `FleetPrefixCache` (`include/ie/fleet_prefix_cache.hpp` +
`src/core/fleet_prefix_cache.cpp`) = per-endpoint `vector<unique_ptr<KvCache>>` +
`vector<unique_ptr<DeltaNetState>>` (one snapshot per card, allocated on `fleet->dev(dev)`); trie/LRU
copied from `PrefixCache` (per-arch-copy norm). `Qwen3NextModel` gained read/write accessors
(`n_devices`/`fleet`/`kv_cache(dev)`/`dn_state(dev)`/`dev_has_kv`/`dev_has_dn`). `Engine::generate` got
a `next_` restore branch (per-card `copy_prefix_from`/`copy_from`, **all-or-nothing** — any card
missing/erroring → `restored=0` so the `pos==0` forward cleanly resets every card) + a `next_` insert
branch; both reuse the existing arch-agnostic `pos=restored`/`cached_tokens`/`snap_at` plumbing and the
crown's stable-boundary `cache_prefix_len` (carried over verbatim — chat() already computes it for
qwen3next via the ChatML branch). Gated on `prompt_cache_on_` (reused) → **default-OFF, OFF-path
byte-identical**. Default **12** entries. **VALIDATED on 2×B70 (Reasoning-Distilled abliterated 80B,
`ie serve --gpus 2`, temp=0):** crown PPL **6.4527 bit-exact** (untouched); ctest **29/29** (serial —
`-j` flake is GPU cross-taint); deterministic 2-turn conversation **ON vs OFF byte-identical both
turns**; turn-2 **`16 cached` ON vs `0 cached` OFF** → per-card KV+DeltaNet restore is exact across
cards. (d) **✅ PROMPT CACHE FLIPPED DEFAULT-ON** — `EngineOptions.prompt_cache` now defaults `true`
(crown + fleet); opt out via `IE_NO_PROMPT_CACHE=1`. Re-validated: crown PPL 6.4527 bit-exact, ctest
29/29, 80B default(no env)→turn-2 `16 cached` vs `IE_NO_PROMPT_CACHE=1`→`0 cached`, byte-identical.
**▶ NEXT (fresh context starts here):** the big engine bet is **EXL3 (QTIP-based SOTA quant) on Arc**
(plan `docs/superpowers/plans/2026-06-19-qtip-quant-engine.md`). See §7.
**Strategic context (memory `[[project_strategy_pivot_2026-06-18]]`):** Seal app dropped →
harness is now a **Pi extension** (`/home/weezy/00 - Mach X Harness/`, builds clean); license → go
**permissive OSS**; the big engine bet = **EXL3 (QTIP-based SOTA quant) on Arc** (plan
`docs/superpowers/plans/2026-06-19-qtip-quant-engine.md` + `docs/exl3_format_notes.md`; exllamav3 ref
cloned at `~/exllamav3-ref`). The canonical crown Q4_K_M was deleted (disk) + redownloaded — gate
green. See §7.**)
(**⚠️ [2026-06-14] PER-MODEL DECODE/PREFILL PERF CAMPAIGN + a
BLOCKER.** A full per-model kernel perf pass (crown UNTOUCHED per rules). **Committed +
pushed wins:** (a) **qwen3next** prefill cap 256→512 (`next_`-gated, 1.08–1.15× long-prefill,
closes pp512 gap; 25/25 bit-identical, no BMG divergence — bbf60c5); (b) **gemma4** batch
gemm-proj + drop per-expert sync = **1.23× prefill** (5fc4a32) + **GPU router** 1.06× (3fc5231);
(c) **qwen3moe** route T==1 decode → int-dot fused path = **+~28% decode** (37.5→48 tok/s),
unlocked by a T==1-only `memset(ws_ffn_y_)` (qwen3next bug-#4 class) — 1c467aa; (d) **dense
Q6_K→Q8_0 repack decode** (new `gemv_q8_0_soa_q8` + `dp4a_ss`): prototype-gate **2.96×** on the
ffn_down cliff (574e59c) → integrated + **8B-validated +19.2% decode** (44.8→53.4), opt-in
`IE_DENSE_Q6K_REPACK=1`, PPL +0.37% (69408a3). **Measured-and-rejected:** qwen3next T==1→fused
is +correct but ~40% SLOWER (2-card overhead, 51ab785). **Characterized:** dense+27B decode are
Q6_K-ALU-bound (not occupancy) → the repack is the lever (split-K isn't). Memory:
[[project_perf_campaign_2026-06-14]]. **✅ CROWN RESTORED (2026-06-14): the crown GGUF had been
deleted externally (disk hit 97% full — a 46 GB Qwen3-Coder-Next download filled it; NOT the perf
work, which is dense-only); re-downloaded via `hf download lmstudio-community/Qwen3.6-35B-A3B-GGUF
Qwen3.6-35B-A3B-Q4_K_M.gguf` into `~/models/lmstudio-community/Qwen3.6-35B-A3B-GGUF/` (symlink
resolves). GATE GREEN: ie-perplexity = 6.4527 / NLL 1.864495 — confirms the file is valid AND the
whole perf campaign is crown bit-exact. ⚠ Disk is now ~7 GB free (100%); free space (e.g. the
46 GB Coder-Next, with owner OK) before more big loads. NEXT (now UNBLOCKED): dense-repack family
rollout — Llama-3/Granite/Phi-4/Mistral/Codestral + 27B + VRAM fit-check + the default-ON
decision. See [[reference_crown_gguf_symlink]].**)
(**CUDA Windows M2 + tuning server-params.** (1) **CUDA
build-config DONE on the Windows/RTX-2080 box** — `ie_core` compiles + links for
`nvptx64 sm_75`, pushed to **`origin/cuda-windows` (dc7e7e0, NOT merged)**. Key finding:
nvptx64 *compilation* needs only oneAPI + CUDA toolkit, **not Codeplay** (Codeplay's plugin
is only for GPU *execution*). Done all default-OFF (Intel build byte-identical): `IE_TARGET_CUDA`
seam, `gemm_q4k_xmx_stub.cpp` (XMX→scalar, crown untouched), scalar `gemm_fp16` for CUDA
(joint_matrix tile invalid on NV), cross-platform `MappedFile` (Win32 + POSIX). **BLOCKED** on
Codeplay's download portal being DOWN (need `ur_adapter_cuda.dll` to *run* on the 2080). M3 =
SG 16→32 kernel port quantified: **78 `reqd_sub_group_size(16)` sites / 12 files** (nvptx
silently uses 32 → compiles but mis-reduces until ported); dense-kLlama3 subset first
(`gemv_*`/attention/elementwise/kv_cache); needs Codeplay runtime to cosine-validate each.
**Merge flow unchanged:** Linux box pulls `cuda-windows`, runs Intel gates, merges. (2)
**Server** `openai_proto.cpp` now parses `min_p` + `repeat_penalty` (alias `repetition_penalty`)
— the sampler already applied them; only the request parse was missing — so the Seal cockpit's
Min-P/Repetition-Penalty/Seed knobs take effect (commit d284c9c). Crown PPL **6.4527 / NLL
1.864495 bit-exact** (server-parse only). See §7 CUDA + branch-flow note.**) (**CODESTRAL SP-track — classic SentencePiece tokenizer
ADDED; the last untested tokenizer family is closed.** Owner roadmap #2. The engine
could NOT load classic-SPM GGUFs (`tokenizer.ggml.model=="llama"`: Llama-1/2, Mistral,
Codestral) — `Tokenizer::load` hard-errored without `tokenizer.ggml.merges`, and
nothing read `tokenizer.ggml.scores`. Gemma 4's "SPM-style" path is rank-ordered BPE
that STILL needs a merges list; classic SPM ships scores + NO merges and merges by
*token score*. Added (additive, gated on `model=="llama"`→`spm_`; every BPE/Gemma path
byte-identical): merges made optional for SPM, `scores_` + `add_space_prefix_` read,
`bpe_merge_spm` (highest-merged-token-score wins, leftmost on tie ≡ llama.cpp
`llm_tokenizer_spm`, `<0xXX>` byte fallback), encode/decode SPM branches (leading dummy
▁, spaces↔U+2581). **Arch was FREE:** Codestral-22B-v0.1 (`general.architecture=="llama"`,
56L/6144/48q-8kv/ffn16384/θ1e6/vocab32768) rides the validated DenseModel `kLlama3` +
Mistral `[INST]` path. **Validated:** tokenizer **9/9 byte-exact vs llama-tokenize**
(BOS+leading-▁, code, café→29113, numbers, tab/newline, 🚀+日本語 byte-fallback, specials),
`ie run` chat coherent + clean `</s>` stop, **PPL 10.89** (NLL 2.387521, deterministic),
crown **6.4527 bit-exact**, ctest 29/29. Tokenizer-only diff (+126 lines). Plan
`docs/superpowers/plans/2026-06-13-codestral-spm-tokenizer.md`; memory
[[project_codestral_spm]]. Unlocks every classic-SPM GGUF (Llama-1/2, Mistral-7B v0.1/
v0.2, Codestral). See §7.**) (**GEMMA 4 (`gemma4`) — 8th arch family, END-TO-END on
the engine.** Google Gemma 4 26B-A4B (MoE, QAT Q4_0) runs tokenizer→forward→`ie run`
chat: "What is the capital of France?" → "Paris". Built from scratch this session:
(1) **Q4_0 GEMV/GEMM** — the engine had ZERO Q4_0 support; new `gemv_q4_0`/`gemm_q4_0`
validated vs real Gemma tensors (max_rel 5e-4); (2) **GemmaConfig + loader** — per-layer
variable head geometry (sliding 256-dim/8-KV +V; global 512-dim/2-KV, V=K), fused
gate_up experts + QAT scales; (3) **forward** — dual-path MoE (shared GeGLU FFN +
128-expert top-8), QK-norm + weightless V-norm, attn scale 1.0, partial RoPE on global
layers, 7 sandwich norms, softcap 30, `full_attention_gemma` (head_dim 512, crown's
full_attention untouched), `geglu` kernel; (4) **SPM tokenizer** — non-byte-encoded BPE
(spaces→U+2581, `<0xXX>` fallback), **byte-exact vs llama-tokenize**; (5) **wired into
`ie run`** (kGemma4 load branch + `<|turn>` template + `<turn|>` stop). PPL same order
as llama-perplexity on the same model+text (instruct-on-prose; not a bug). Crown PPL
6.4527 bit-exact throughout; ctest 29/29. Commits 5e52f23→10ff55b. Tools: ie-gemma4-
{load-test,gen,tok-test,ppl}. Memory: [[project_gemma4_arch]]. **✅ PREFILL PERF 2026-06-14:
1.23× (45.3→55.7 tok/s, T=669 on one B70)** via two additive gemma4-only levers, both bit-
stable (reprefill PPL match; crown 6.4527 bit-exact): (1) batch the per-token gemv_q4_0
projection loop into one `gemm_q4_0` for T>1 when its SLM staging fits (+6.9%; opt-out
IE_GEMMA4_NO_GEMM_PROJ); (2) drop the per-expert `q.wait()` in the MoE loop — the in-order
queue already serializes, the sync only stalled the pipeline (+15%; opt-out IE_GEMMA4_EXPERT_SYNC;
bit-identical). Also speeds decode (T=1 MoE loops K experts). **+ GPU ROUTER 2026-06-14:**
moved the host T×E×H router dot-loop (E=128) to GPU — rms(attn_out)·router_in_scale·(1/√H)
on device then ONE gemv_q_T logits[T,E]=rin@router_w_devᵀ (router_w uploaded TRANSPOSED F16
[H,E]); host keeps only top-k. +1.06× prefill (gemma4's router is only ~6% of prefill, unlike
qwen3moe's 66%, so a smaller win). Faithful: 64-tok reprefill PPL GPU 2694.56 vs host 2683.30
(0.15%), greedy gen identical; opt-out IE_GEMMA4_HOST_ROUTER=1. Crown 6.4527 bit-exact.
NEXT (optional): 31B dense (downloaded), Q4_0 fused-MoE gather/scatter, true SWA >1024 ctx.
See §7.**) (**WAVE-1 GATE: IBM Granite-3.x dense GREEN —
Granite-3.3-8B loads, forwards, and chats.** The 4 Granite scalar multipliers
(embedding 12.0 / residual 0.22 / **attention.scale 0.0078125** / logit 16.0) are
now applied in the dense forward (they were read into config but never consumed);
fixed the DOTTED `granite.attention.scale` key (was `attention_scale` → silently
defaulted); added `TemplateFamily::kGranite` + `build_granite_prompt`
(`<|start_of_role|>` turns, NOT ChatML). All additive + value-gated (every
non-Granite model byte-identical). Greedy " Paris.", per-layer cosine 0.9995–
0.99999 (L01–L39), PPL 10.30, chat coherent (Python+unittest, stops clean).
Crown PPL 6.4527/NLL 1.864495 + dense qwen3-8b NLL 2.940491 bit-exact, ctest 28/28.
Commits 34b0049 (forward) + 13f2fe7 (template). Unlocks the Granite-3.x dense tier;
Granite-4 hybrid-Mamba2 = separate Wave-2 arch. See §7.**) (**qwen3next STEP 2 FORWARD CORRECT — Qwen3-Next-80B runs
end-to-end across 2×B70: greedy "The capital of France is" → " Paris. The capital of Italy
is Rome".** FOUR forward bugs found+fixed by op-by-op VALUE diff vs the saved fp32 oracle
(probes `IE_QWEN3NEXT_DBG`/`_DBG_VALS`/route-dump + `ie-inspect --layer`): (1) down
`down_dt` dispatch (GGUF mixes ffn_down Q6_K/Q4_K per layer — prior "down-kernel-overflow"
was a MISDIAGNOSIS) [402786a]; (2) fused `ssm_ba` per-k-head split `{4,SKH=16}` β=[0:2]/
α=[2:4] not contiguous [d1afa7a]; (3) DeltaNet q/k repeat must INTERLEAVE (h/repeat) not
tile (h%n_in) — added `bool interleave` param, 27B keeps default tile [e7319f7]; (4)
`moe_y` not zeroed before `moe_prefill_reduce` (it accumulates; L0 ok, L1+ MoE ~2.66×
inflated) [1ed07e0]. NEOX-rope suspicion REFUTED. Verified vs oracle: L0 x(out) bit-exact,
dn_g/dn_out/attn_block/L1 moe_y all match, routing top-10 matches `ffn_moe_topk-1`. Crown
PPL 6.4527/NLL 1.864495 bit-exact; all isolated to qwen3next.cpp. **STEP 2 COMPLETE:**
cosine sweep all 48 layers (0.998–1.007 vs oracle) + PPL 4.73 (`ie-qwen3next-ppl`); probes
STRIPPED; **WIRED into ie::Engine — `ie run/serve --gpus 2` chats** ("…is Paris."); int-dot
W4A8 down DEFAULT (pp512 2.5× vs fp16). **vs llama.cpp SYCL** (built GGML_SYCL @ fdc3db9b6,
runs qwen3next ON GPU via its gated_delta_net/ssm ops; same GGUF/2×B70/session): **DECODE
51.8 vs 37.1 tok/s = 1.40× faster**; prefill parity ≤256 tok, ~10% behind at pp512 (our
DeltaNet T≤256 chunking). 7th Engine arch family. commits 402786a→c0338f4. NEXT: `ie serve`
smoke; optional relax T≤256 prefill cap to close the long-prefill gap. See §7.**) (**qwen3moe TIER-3 PREFILL DONE — Qwen3-Coder-30B prefill
80.6→651 tok/s = 8.1× (12×→1.5× behind llama Vulkan).** Two additive, qwen3moe-owned
levers, crown/`moe_fused.cpp` untouched: (1) GPU-gemm router logits — the host
`route_token` dot loop was ~66% of prefill wall, invisible to `--kprofile`; (2) NEW
`src/ops/moe_qwen3.cpp` generalized int-dot W4A8 down kernel `_gen` for any `E_ffn%256==0`
(the crown's int-dot down is `E_ffn==512`-locked) — down_q8_6k 1720→190 ms. PPL 11.98
bit-stable, crown 6.45 bit-exact, ctest 28/28. See §7. Prior milestones below.) (**P3d Qwen3.6-27B FULLY COMPLETE — forward validated to per-layer
cosine + chat bug resolved.** Per-layer cosine vs the llama oracle PASSED: ≥0.9995 on all 64 layers
(embedding 1.000000), exact greedy argmax (' Paris'), top-5 identical — forward correct (the
rel_fro ~0.02 is fp16-residual precision, oneDNN-invariant via the new `IE_QWEN35_NO_ONEDNN` switch,
NOT a defect). **Chat bug was a MISDIAGNOSIS:** temp-0.7 "garbage" does NOT reproduce (114 clean
gens); tokens 248068/248069 are `<think>`/`</think>` (legitimate markers); "state leak" refuted
(forward is temp-independent); the "obvious" suppression fix would have BROKEN reasoning. New tooling:
`ie-qwen35-dump`, `ie-llama-dump` (rebuilt vs master), `IE_QWEN35_NO_ONEDNN`. Crown PPL 6.45 held.
Next: P3a (Llama-3) → 72B/multi-GPU scale. Details: §7.) (**Cycle-4 2026-06-12: P3a Llama-3 DONE; 72B RUNS end-to-end split across 2×B70 (decode 7.2 tok/s); now optimizing decode via TENSOR-PARALLELISM — TP-1 all-reduce done+tested (b481f03), TP-2/TP-3 turnkey-spec'd (`docs/superpowers/specs/2026-06-12-tensor-parallel-decode-design.md`). See §7.**) (**Prior cycle-3: P3d 27B COMPLETE through perf: Qwen3.6-27B runs,
validated (PPL 5.34 oracle-consistent), and BEATS llama.cpp on prefill 1.9× (pp512 577 vs
303; decode parity 10.0 vs 9.72). Tasks 3A→3C forward + Task 4 PPL + Task 5 perf-optimized
(3 stacked prefill opts, all crown-safe via qwen35-only oneDNN flag). Dual-B70 hardware
resolved via ProArt board swap. Build green, ctest 22/22, crown PPL 6.45 bit-exact
throughout. See §7 + Hand-off.md perf ledger.** Prior cycle-2 below.) (**P3d ORACLE RESOLVED + Task 2.5 DONE.** Updated
llama.cpp to master `fdc3db9b6` (+665 commits, brings MTP/NextN support the old HEAD
lacked) → loads Qwen3.6-27B cleanly; full bit-exact oracle restored
(`~/llama.cpp/build-cpu/`). Dumped per-op tensors and **verified the entire qwen35
dataflow vs ground truth** (`docs/qwen35_27b_oracle_dataflow.md`) — confirms the plan's
landmines (conv_channels=10240, 16k/48v heads, joint Q|gate 12288, residual order).
P3d decision: **proceed** (oracle risk eliminated). Also: **IP audit** →
`THIRD_PARTY_LICENSES.md` (engine core original; 3 import-path files are ggml-MIT ports,
now attributed). **72B finding:** engine is single-GPU → 72B inference needs multi-GPU
that doesn't exist yet (see §5.7). Prior context below.) (**BREADTH PIVOT. P3e AWQ+GPTQ import COMPLETE &
PROVEN: `ie import` loads AWQ+GPTQ across Qwen3 AND Qwen2.5 dense (formats llama.cpp
can't), verified end-to-end incl. Qwen2.5-7B 2-shard (PPL 15.48 + correct codegen);
Qwen3-8B dense 2.937037 EXACT (zero regression). P3d Qwen3.6-27B: Tasks 0-2 + Q5_K/Q8_0
dequant kernels DONE; the loader + ~600-line DeltaNet/gated-attn forward + oracle are a
focused multi-session push (no clean llama.cpp oracle — see §7). P3b PAUSED. ctest 22/22,
~29 commits this session, all on `main`.** §7 = read first.)
P2 COMPLETE; Seal P0 + P1a done, P1b started. §7 "In-flight / next-session
pickup" is the first thing to read after this header.

**Pivot rationale (2026-06-11):** the goal is the engine Intel can't ignore →
**breadth** (many models + file formats), not just crown speed. The dense model
people actually deploy is **Qwen3.6-27B** (`qwen35` arch), which the engine could
NOT load. It is the crown's gated-DeltaNet + full-attn family MINUS the MoE, plus
the P2 dense MLP — ~80% reuse of ops we already beat llama.cpp on. And **AWQ**
ingestion (P3e) is a format llama.cpp can't load natively — a clean differentiator.
P3b (dense-8B decode micro-opt) was polishing a model nobody runs; its
`gemv_q4_K_q8` retune returns under P3d Task 5 measured on the 27B's real shapes.

---

## 1. What we are building and why

A from-scratch C++/SYCL LLM inference engine that is the **fastest local
inference on Intel Arc GPUs** — proven, not claimed — shipping as **one
engine, two offerings**:

```
                 ┌─────────────────────────────────────────┐
                 │   THE ENGINE (this repo)                  │
                 │   neutral name TBD · Apache-2.0 at launch │
                 │   C++ API · `ie` CLI · OpenAI-compatible  │
                 │   server (`ie serve`)                     │
                 └────────────────┬────────────────────────┘
              ┌───────────────────┴───────────────────────┐
  OFFERING 1: standalone                      OFFERING 2: Seal Team (proprietary)
  Ollama/LM-Studio-style product              multi-agent app at
  (CLI now, lite Tauri UI at P5)              /home/weezy/SEAL - VERSION 2.1
                                              consumes the engine as a sidecar
                                              via SEAL_RUNTIME=openai-compat
```

**Iron rule:** there is never a second engine codebase. Seal consumes the
same binary the public gets. (The deleted `seal-native-engine` died for
violating this.)

**End goal:** the gold standard for local inference on Intel hardware —
good enough that Intel (or another buyer) can't ignore it. Strategy,
licensing (open-core Apache-2.0), naming criteria, and go-to-market are
specified in `docs/superpowers/specs/2026-06-10-gold-standard-roadmap-design.md`.

**Claim discipline:** until breadth exists, the public claim is exactly
"fastest known inference for Qwen3.6-35B-A3B on Intel Arc — beats
llama.cpp's best backend on both metrics, same-hour measured." Never
"best engine for Intel GPUs."

## 2. Current state (the facts)

**Launch publication (2026-06-29) — PACKAGE COMMITTED (`a9bc899`), gated on a clean-box re-bench:**
- Lead article `docs/public/gptoss_arc_b70_lead_2026-06-29.md` (gpt-oss-20b both-axes win vs llama-SYCL)
  + finalized 120b write-up + marketing set (`docs/public/marketing/`: reddit, HN+X, Intel/Phoronix one-pager).
- `docs/public/marketing/VERIFIED_CLAIMS_2026-06-29.md` = the claims ledger (every number + source + status);
  built ledger-first + adversarially reviewed (`REVIEW_2026-06-29.md`, verdict go-with-fixes → punch-list applied).
- **OPEN GATES before the public push:** (1) clean-box re-bench of the load-bearing 20b-vs-llama (`927`
  prefill baseline) + 120b throughput — box degraded (4 GB swap, ~9-min loads); run `scripts/verify_publish_claims.sh`
  once cool. (2) `ctest` now declares **34** (docs said 31) → needs a pass-run. (3) only **Vulkan** llama-bench on
  box — confirm/locate the SYCL build the "vs-SYCL" claim needs. (4) OWNER: Apache-2.0 + engine-name confirm.
- Correctness is NOT gated (committed + proven): crown 6.4527 bit-exact; 120b batched PPL 15.1985 == T=1.

**Performance — TOTAL CROWN held (2026-06-10, same-hour alternating runs):**
- prefill pp512 **1144 ± 5 vs llama.cpp SYCL master 1064 ± 8 (+7.6%)**
- decode tg128 **84.1 turbo / 81.0 default vs 81.31**
- PPL **6.45** at production defaults (improved from 6.52 on 2026-06-10 by
  the qwen2 pre-tokenizer contraction fix `f508c01` — found by Seal
  dogfooding; hard gate: ≤ 6.57 on every change)
- Reproduce: `scripts/bench_showdown.sh` (order-controlled protocol)

**Product surface (P1) — SHIPPED:**
- `ie::Engine` streaming generate/chat API (`include/ie/engine.hpp`)
- OpenAI-compatible server: `./build/src/ie serve <gguf> [--port 11435]
  [--ctx N]` — /v1/chat/completions (SSE + non-stream), /v1/models,
  /health, 501 /v1/embeddings; **tool calling works** (tools rendered into
  the Qwen template; `<tool_call>` JSON emission)
- CLI: `ie run` (interactive chat), `ie serve`, `ie bench`
- **Seal Team runs real agent tasks on the engine** (verified: 2 tool
  calls, file created+verified). Recipe in `docs/seal-integration.md`.
- Gates green: ctest 12/12, PPL 6.52, crown bands held (1170/83.3),
  server smoke, 1-hour soak 5402 completions / 0 failures / RSS flat (+0.0016%).

**Model breadth (P2) — TWO ARCHITECTURES:**
- Crown: Qwen3.6-35B-A3B (hybrid DeltaNet/MoE) — the perf-crown daily driver.
- **Qwen3-dense (qwen3-8b) validated bit-exact post rope-fix** (`34e4c01`):
  layer parity all 37 comparable residual slots rel_fro ≤ 1.4e-3 / cosine
  1.000000 vs llama.cpp; greedy parity 64/64 tokens vs `--temp 0` reference;
  PPL deterministic to the last bit — avg NLL 2.940491 exactly at
  `--max-tokens 511` (3/3; the constant is invocation-bound — the default
  invocation predicts one more token and scores 2.937037 exactly, 4/4
  across two md5-identical clean rebuilds; both recorded in
  `docs/ppl_baseline_matrix.md`). The ~18.9 PPL level is the model, not
  the engine (llama.cpp scores 24.54 on the same corpus/GGUF). Per-arch
  gate: `scripts/p2_parity_qwen3.sh` — ALL GREEN at T8 close.

**Seal P0 list — COMPLETE & live-validated (2026-06-10):** P0.1 snapshot
fold-in to main (builds green); P0.2 `ie serve` as the cockpit's preferred
native runtime + readiness probes; P0.3 Linux eval harness (leak-proofed);
P0.4 verify→finish close-out (+npm-placeholder guard); P0.5 session resume;
P0.6 context compaction. Live-validation batch (`SEAL .../docs/p0-validation-2026-06-10.md`,
commit `7da43d5`): eval strict passes **1/6 → 4/6** after the engine
tokenizer/max_tokens fixes; resume & compaction both PASS against the live
engine. Engine↔Seal recipe: `SEAL_RUNTIME=openai-compat
SEAL_RUNTIME_URL=http://127.0.0.1:11435 SEAL_MODEL=<gguf-stem>`.

**Scope honesty:** two model architectures (Qwen3.6-MoE crown + Qwen3-dense),
one GPU (B70), single-user, Linux-only today. Dense decode (43.7 tok/s vs
llama.cpp's 77.7 same-hour) is unoptimized — P3b is closing it. Dense prefill
already leads (+15%). Known v1 limits: stateless server requests, no
truncation policy, one generation at a time (mutex), structured `tool_calls`
response field not yet emitted (Seal's fallback parser handles the text form).

## 3. Roadmap — where we are and what's next

| Phase | Status | Content |
|---|---|---|
| Crown | ✅ 2026-06-10 | SoA repack + int-dot MoE prefill; decode crown v1.5 |
| P0 credibility | ✅ | `scripts/bench_showdown.sh`; writeup draft (`docs/public/2026-06-beating-llamacpp-on-intel-arc.md`); IGC issue package (`docs/public/igc-issue-draft.md` + `tools/igc_repro/`) — **filing the IGC issue is a pending human action** |
| P1 product surface | ✅ | Engine API, OpenAI server, CLI, Seal integration, gate battery |
| **P2 model breadth** | ✅ COMPLETE 2026-06-10 | Qwen3-dense second architecture shipped & validated bit-exact (rope_partial race fixed `34e4c01`; layer cosine 1.000000, greedy 64/64, PPL avg NLL bit-exact across runs); T8 battery all green (build 0 err, ctest 15/15, crown PPL 1.864495 exact + bands 1158/81.9, parity script, dual-model server smoke); dense baseline ledgered: pp512 1190 ± 5 vs llama.cpp SYCL same-hour 1036 ± 4 (**+14.9%**), tg 43.7 vs 77.7 (unoptimized, out-of-scope). Details: `docs/benchmark_matrix_2026-06-09.md` §P2. |
| **P3 (engine)** | ⬅ IN PROGRESS | Breadth pivot 2026-06-11. **P3d (Qwen3.6-27B) ✅ · P3e (AWQ import) ✅ · P3a (Llama-3) ✅ · multi-GPU TP ✅ · qwen3moe (Qwen3-Coder-30B) ✅ FUSED MoE + TIER-3 PREFILL 8.1× (pp512 80.6→651, GPU router + generalized int-dot down — see §7) ✅ 2026-06-12** · **qwen3next (Qwen3-Next-80B) ✅ FORWARD CORRECT + WIRED INTO ENGINE 2026-06-13** (layer-split across 2×B70; 4 forward bugs fixed vs oracle; greedy " Paris", cosine sweep all 48 layers, PPL 4.73; `ie run --gpus 2` → "The capital of France is Paris."; §7)** · **gemma4 (Gemma 4 26B-A4B QAT) ✅ END-TO-END 2026-06-13** (new Q4_0 GEMV/GEMM + per-layer head geom + dual-path MoE + SPM tokenizer byte-exact; `ie run` → "Paris"; §7)** → Remaining: P3c (Windows). P3b PAUSED. Engine runs **8 arch families** (qwen35moe crown, qwen3/qwen2-dense, qwen35-hybrid 27B, llama-3, qwen3moe Coder-30B, qwen3next 80B, **gemma4 26B-A4B**) + AWQ/GPTQ import + **multi-GPU** TP (dense) + **layer-split** (qwen3next + **crown all-Q8_0 `kQwen35Moe` 2026-06-30**, coherent on 2×B70, root-caused the cross-card `moe_router` static fault — see §3 DONE) via `ie run/serve --gpus N`. |
| **P3d Qwen3.6-27B** | ✅ COMPLETE (2026-06-12) | `qwen35` dense-hybrid (DeltaNet+full-attn+dense MLP, 64 transformer +1 NextN, hidden 5120, ffn 17408, head_dim 256, 48v/16k DeltaNet). Plan `docs/superpowers/plans/2026-06-11-p3d-*.md`. Tasks 0–5 DONE: loader + unfused DeltaNet/gated-attn forward (3A–3C) + PPL 5.34 oracle-consistent + perf (prefill 1.9× vs llama). **Chat bug RESOLVED as a misdiagnosis (2026-06-12); engine.cpp unchanged. Per-layer cosine PASSED: ≥0.9995 all 64 layers vs llama oracle, exact greedy argmax (' Paris'), top-5 identical — forward validated (fp16-precision rel_fro ~0.02, oneDNN-invariant, NOT a defect).** See §7. |
| **P3e AWQ+GPTQ import** | ✅ WORKING (qwen2+qwen3) | `ie import` loads **AWQ and GPTQ** safetensors directly — two formats **llama.cpp can't load natively** — across **Qwen3 AND Qwen2.5** dense archs. Q4_K+Q6_K encoders, sharded support, all host-tested (ctest 21/21). **VERIFIED on real models:** Qwen3-4B AWQ/GPTQ + **Qwen2.5-3B-AWQ (PPL 18.73 ≈ ref 17.30)**; weights bit-verified faithful (≤1.6%); Qwen3-8B dense **PPL 2.937037 EXACT (zero regression)**. Unlocks the Qwen2.5 AWQ/GPTQ ecosystem incl. **72B**. Detail: `docs/p3e_awq_import_validation.md`. Follow-ups: act-order GPTQ, 7B/72B, vLLM cross-check. |
| ~~P3b dense decode~~ | ⏸ PAUSED | `gemv_q4_K_q8` retune (10.31ms→~7.5ms target) was correct but polishes Qwen3-8B (not deployed). Clean baseline captured; returns under P3d Task 5 on real 27B shapes. Branch `p3b-task1-*` merged to main default-OFF (`e1e1fa5`). |
| P4 public launch | pending | Fresh Apache-2.0 repo, writeup publishes, r/LocalLLaMA + Phoronix + Intel devrel/Liftoff; engine gets its neutral name (candidates + trademark criteria in the spec) |
| P5 UX layer | pending | Lite Tauri UI over the server, `ie pull` (HF download), `ie import` (AWQ/GPTQ/safetensors converters), nightly public benchmark dashboard |
| **SEAL P1 tier** | ⬅ IN PROGRESS | P1a MCP client **COMPLETE** (modulo live gate — see §7); P1b parallel subagents **STARTED** (Task 1.1 pool core done). Plans: `SEAL .../docs/plans/2026-06-10-p1{a,b}-*.md`. Then hooks, vision input (gap-analysis P1 list). |

**Crown defense (ongoing):** rerun `scripts/bench_showdown.sh` whenever
llama.cpp master moves; PPL + order-controlled A/B discipline on every
engine change, forever.

## 4. Document map (what is authoritative for what)

| Document | Role |
|---|---|
| **MASTER_DEV_PLAN.md** (this file) | THE authority: state, roadmap, decisions |
| `Hand-off.md` | Deep engine-internals state: kernel history, perf ledger, measurement rules, tooling. Read second. |
| `docs/superpowers/specs/2026-06-10-gold-standard-roadmap-design.md` | Strategy/productization spec (approved): offerings, licensing, naming criteria, P-phase definitions |
| `docs/superpowers/plans/2026-06-10-p0-*.md`, `...-p1-*.md` | Executed implementation plans (historical record) |
| `PLAN.md` | The engine's original 12-phase technical plan + research index. Historical authority for kernel-phase detail; superseded as top authority by this file. |
| `RELEASE.md` | v1 release record + crown amendments |
| `docs/benchmark_matrix_2026-06-09.md` | The numbers ledger (v1.4→v1.6 sections) — every public number traces here |
| `docs/seal-integration.md` | Seal-on-engine recipe + gap statuses |
| `docs/known_bugs.md` | DN non-determinism (closed, HW-level, T≤256 chunking) — do NOT re-investigate |
| `docs/public/` | Launch artifacts (writeup, IGC issue draft) |
| Seal repo: `docs/seal-gap-analysis-2026-06-10.md`, `docs/repo-stabilization-2026-06-10.md` | Seal competitive gaps + repo-churn inventory. Seal's own `seal-agent-app/MASTER_DEV_PLAN.md` is STALE (predates the engine pivot) — refresh it during Seal P0 to mirror this file. |
| Auto-memory (`~/.claude/.../memory/`) | Session-persistent pointers; this file outranks it on conflicts |

## 5. Open decisions (waiting on the human)

1. ~~P3b re-scope~~ — RESOLVED 2026-06-11: branch merged to main (default-OFF),
   P3b PAUSED, pivoted to breadth (P3d Qwen3.6-27B + P3e AWQ). See header + §7.
2. **File the IGC issue** (from your GitHub account): text at
   `docs/public/igc-issue-draft.md`.
3. ~~Engine name~~ — **RESOLVED 2026-06-13: the engine is "Mach X"**, full
   official name **"Mach X Inference Engine"** (vendor-neutral; avoids the
   Intel "Arc"/"Xe" trademarks). Marketing line: *"we're running Mach10 on dual
   B70s."* GitHub repo (private for now): `Red-Weasel/mach-x-inference-engine`.
   **⚠ LICENSE REOPENED 2026-06-15:** the name "Mach X" stays RESOLVED, but the
   **Apache-2.0 / open-core launch license is NO LONGER settled** — the owner
   questioned it. Open decision: open-core Apache-2.0 (the original plan), another
   OSS license, or **proprietary**. The launch article's license line is left out
   until this is decided.
4. **Windows validation hardware**: dual-boot the B70 box or acquire a
   B580 before P3 sign-off.
5. **Push the Seal repo** (~25+ commits ahead of origin, private) when ready.
6. ~~Seal snapshot fold-in~~ — DONE (merged to main, P0.1).
7. **72B needs engine MULTI-GPU (new finding 2026-06-11).** The engine is
   single-device today (`src/core/allocator.cpp` enumerates but selects one
   GPU). A 72B AWQ/GGUF (~40 GB) does NOT fit one 32 GB B70, so running/
   benchmarking it on the engine across the incoming 2nd B70 (→64 GB) requires
   **tensor/layer device-split offload that does not exist yet** — a substantial
   new capability, NOT just the extra card. llama.cpp already splits (`-sm
   layer/row`), so the *reference* side of a same-hardware A/B works. **Decision
   for the human:** (a) implement engine multi-GPU (big, but unlocks 72B + is a
   real breadth/credibility feature), OR (b) bank "P3e at scale" on a model that
   fits ONE B70 now — **Qwen2.5-32B-Instruct-AWQ (~19 GB)** is runnable +
   benchmarkable today and extends the proven P3e ladder (currently 7B) without
   waiting on multi-GPU. What IS bankable for 72B without multi-GPU: **import
   correctness** (AWQ→GGUF weight-faithful, CPU-verifiable). 72B AWQ download
   staged at `~/models/Qwen2.5-72B-Instruct-AWQ` for when the call is made.
   **PROGRESS 2026-06-12: 72B UNBLOCKED END-TO-END (multi-GPU run + RAM-safe import done).**
   `DeviceFleet` (37ff1d1) + `DenseModelSplit` (c25d14d, prefill+decode 2474874) run a dense
   model split across both B70s, **2-GPU == single-GPU BIT-IDENTICAL** (qwen3-8b: all 151936
   vocab + greedy 16/16). `ie-multi-gpu-run --gguf <g> --gpus N` (379687e) generates across N
   cards (8B: 45.5 tok/s, coherent). Fixed the multi-GPU blocker (`gemv_q_T` device-static
   scratch → DEVICE_LOST → now per-device; zero regression, qwen3+crown PPL bit-exact). The
   72B IMPORT was the other blocker (held the whole ~40GB output in RAM → OOM'd the editor);
   fixed with **streaming import** (`IE_IMPORT_STREAM=1`, fc2a9c5 — byte-identical to the
   proven path, ~8GB anon vs ~40GB). **✅✅ THE 72B RUNS (2026-06-12): Qwen2.5-72B streamed-
   imported to a 39GB GGUF, then `ie-multi-gpu-run --gpus 2` loaded it SPLIT across both B70s
   (20.8GB/card, 80 layers) and generated "The capital of France is" → " Paris." — correct,
   coherent.** decode 7.2 tok/s (unoptimized). Design
   `docs/superpowers/specs/2026-06-12-multi-gpu-layer-split-design.md`; memories
   [[project_multi_gpu]] + [[project_box_hardware_limits]].
   **✅ 72B PPL VALIDATED (2026-06-12): engine PPL 8.97** (avg NLL 2.193958, 511 tokens,
   full causal context, fp16 KV, layer-split across 2×B70) via the new additive tool
   `ie-multi-gpu-ppl` (mirrors `ie-perplexity`'s exact streaming-T=1 NLL + built-in corpus
   over `DenseModelSplit::forward`; zero edits to any validated path). **Correctness proven
   by cross-model ordering:** the same-tokenizer Qwen3-8B scores NLL 2.94 (PPL 18.9) → the
   72B at NLL 2.19 is decisively better, exactly as a 72B must be; a mismapped-tensor import
   would yield PPL >50, not a correctly-ordered 8.97. With the coherent-gen + bit-verified
   AWQ weights (P3e, ≤1.6%) + bit-identical layer-split equality test, the 72B import +
   split-forward are quantitatively validated — TP is de-risked to build on a sound model.
   **✅ IMPORT INTEROP BUG FOUND + FIXED (2026-06-12):** llama.cpp REFUSED our imported GGUFs
   for vocab-padded Qwen2.5 — `token_embd` is [H, **152064**] (Qwen2.5 pads vocab +128) but
   `copy_tokenizer_kvs` copied a 151936-token list from the ref GGUF, and **llama derives
   n_vocab from the token-LIST length** (not the `.vocab_size` KV), so it rejected the 152064-row
   tensor (`check_tensor_dims`). **Fix (`hf_import.cpp`):** emit `<arch>.vocab_size = embedding
   rows` AND pad the token list + token_type to the embedding row count with UNUSED `[PAD]`
   placeholders (stock Qwen2.5 convention). **VALIDATED on Qwen2.5-7B-AWQ (same +128 padding):**
   re-imported → llama-perplexity now **LOADS it (PPL 21.98, was a hard rejection)** + our engine
   PPL **15.48 bit-identical** to P3e (pad rows are inert → zero quality change). The 72B benefits
   identically on re-import (deferred — disk only 56G free, needs ~40G; not run unattended). Our
   loader reads vocab from the (now-padded) token list, so imported padded models now load at the
   true 152064 vocab. The same-file 72B llama PPL anchor is unblocked pending that re-import.
   **DECODE-OPT IN FLIGHT — TENSOR-PARALLELISM (2026-06-12).** Layer-split runs the
   cards SERIALLY (profiled `IE_SPLIT_PROFILE`: dev0 9ms then dev1 9ms, each idle ~50%);
   TP splits every *layer* across both cards so they compute concurrently → target
   ~1.4–2× decode. **TP IS ADDITIVE + NON-bit-exact by design** (different reduction
   order); crown/single-GPU-dense/27B forwards untouched, guarded by a mandatory
   regression battery. **TP-1 DONE+TESTED (b481f03):** `DeviceFleet::all_reduce_sum_fp16`
   (fp32-accum host bounce) validated on 2×B70 (dev0=1+dev1=2→3 every buffer; `ie-multi-gpu-probe`).
   **✅ TP-2 + TP-3 DONE + VALIDATED (2026-06-12).** New additive files `include/ie/dense_tp.hpp`
   + `src/model/dense_tp.cpp` (`DenseModelTP`) — zero edits to crown / single-GPU dense / split /
   27B. TP-2 = Megatron column(q/k/v,gate/up)-then-row(o-proj,ffn-down) split with local re-pack
   helpers (`up_col` Q4_K/Q6_K/F16, `up_row` 256-superblock Q4_K/Q6_K, `up_bias`), GQA head-split,
   embed/lm_head on card0. TP-3 = concurrent forward, each card's pre-reduce work submitted to its
   in-order queue without waiting, 2 host-bounce all-reduces/layer as the sync. **VALIDATED:**
   `ie-multi-gpu-tp-test` on qwen3-8b → prefill cosine **0.999999**, argmax MATCH, greedy **16/16
   identical**, coherent (" Paris. …Washington, D.C."). **72B speedup: decode 10.4 tok/s TP vs 7.2
   layer-split = 1.44×** (`ie-multi-gpu-run --tp`; right in the predicted host-bounce range),
   coherent. **REGRESSION GREEN:** crown `ie-perplexity` **6.45 bit-exact** (NLL 1.864495). TP is
   NOT bit-exact by design (cosine-validated). Scope v1: Q4_K/Q6_K + F16 column weights (qwen3-8b
   attn_v); **qwen3/qwen2 AND llama-3.x** (Q/K un-permute composed with the column split via
   `up_col_perm` — validated Llama-3.2-3B cosine 0.999999, so Llama-3.1-70B runs across 2×B70).
   F16 row-parallel weights are the only TP follow-up (no current target has one).
   **⛔ Level-Zero P2P all-reduce is HARDWARE-BLOCKED on this box** — `ext_oneapi_can_access_peer`
   returns **0 both directions** for the 2×B70 (PCIe topology, no P2P), verified by probe. So the
   ~2× path is unavailable here; the host-bounce all-reduce was instead made **concurrent** (gathers
   + scatters submitted to all queues then waited together → overlapped PCIe; math-identical, TP
   cosine still 0.999999). Clean TP decode baseline = **10.4 tok/s (1.44× layer-split)**; the
   concurrency tweak is correctness-validated, stable cold perf A/B pending (cards were thermally
   throttling after repeated 72B loads → 6/4.7 tok/s declining = heat, not a regression).
   **✅ TP WIRED INTO THE PRODUCT (2026-06-12): `ie run --gpus N` + `ie serve --gpus N`.** Additive
   Engine path (`EngineOptions.n_gpus`; default 1 → byte-identical single-GPU). When `n_gpus>1` &&
   dense arch, `Engine::load` skips the single-GPU `dense_`/`kv_` and loads `DenseModelTP` on a
   `DeviceFleet`; `forward_step` bounces the TP host-logits into `d_logits_` so the existing GPU
   sampler/chat/server stack is unchanged. `forward_step` resets the per-card KV when pos==0.
   VALIDATED: `ie run --gpus 2` qwen3-8b → "The capital of France is Paris." (full chat stack:
   template + GPU sampler + multi-step TP decode); single-GPU `ie run` unchanged ("2 + 2 equals
   4."); crown PPL 6.45 bit-exact. `ie serve --gpus N` rides the same path (OpenAI server →
   72B-as-a-product). v1 caveats: int8-KV unsupported on TP (fp16 only), one generation at a time.
   Other remaining polish: ~~72B PPL~~ (engine PPL 8.97 DONE; llama same-file anchor blocked
   by the vocab-padding import bug above — fix the writer, then re-anchor); Engine/`ie serve`
   `--gpus N`; hybrid-27B split.
   **⛔ The 72B AWQ *import* OOM-crashed VS Code (32 GB RAM ceiling)** — see
   [[project_box_hardware_limits]]; it must NOT be retried autonomously, needs a
   memory-capped scope or a streaming import path. So the bankable-import item is
   itself RAM-blocked on this box; the safe scale demo remains Qwen2.5-32B-AWQ on
   one B70.

## 6. Quick start for a fresh session

```bash
cd "/home/weezy/00 - Inference Engine"
cmake --build build -j                    # clean build (0 errors expected)
ctest --test-dir build                    # host+GPU unit tests (15/15)
./build/tools/ie-perplexity               # crown PPL gate → 6.45 (≤ 6.57)
./scripts/bench_showdown.sh               # crown check vs llama.cpp
./scripts/p2_parity_qwen3.sh              # dense (qwen3-8b) parity gate
./build/src/ie serve ~/.seal/models/Qwen3.6-35B-A3B-Q4_K_M.gguf  # the product
# dense model: ~/.seal/models/Qwen3-8B-Q4_K_M.gguf (symlink to an Ollama blob)
```

Hard rules a fresh session must know: PPL ≤ 6.57 gate on every engine
change (crown is bit-exact: avg NLL **1.864495** ≡ PPL 6.45; dense default
since 2026-06-15 is the Q6_K→Q8_0 repack: avg NLL **2.944273** @
`--max-tokens 511`; the **2.940491** fp16 leak-canary is the opt-out
`IE_DENSE_NO_Q6K_REPACK=1` and is what `p2_parity_qwen3.sh` checks);
order-controlled
A/Bs (new-old-new), discard first run after rebuild (JIT); prefill chunked
at T ≤ 256 (HW bug workaround); never touch ESIMD/block2d paths; never edit
`src/model/qwen36.cpp` or its hot kernels (crown is untouchable — the dense
path COPIES helpers into `src/model/dense_dispatch.hpp`, never hoists);
work happens directly on `main`; every milestone updates THIS file;
**GPU EXCLUSIVITY: one GPU workload at a time across ALL repos and agents**
— the B70 is shared by the engine and Seal; two model loads (~20 GB each on
32 GB) silently taint every perf number and eval latency measured in the
overlap (incident 2026-06-10: P2 T2/T3 overlapped the Seal eval server;
re-measured solo). CPU work (Rust/TS builds, planning, host-only tests) may
parallelize freely; anything that loads a model serializes.

## 7. In-flight / next-session pickup (READ FIRST after a context reset)

### ✅ FIXED 2026-06-29 — gpt-oss-120b BATCHED PREFILL corruption (E=128 small-M XMX) → prefill unblocked, batched DEFAULT
The 120b 2-card TP **batched prefill (T>1) produced garbage** (PPL ~123 vs T=1's correct 15.20) — which is why
prefill ran at decode speed (~34 tok/s): the slow-but-correct **T=1-sequential workaround was the default**.
**ROOT CAUSE (agent-team root-caused via a 5-lens parallel code workflow, then GPU-confirmed): the per-expert
MoE prefill `gemm_fp16_onednn` ran at tiny M** (n_e≈1–8 rows/expert at E=128/top-4); **BMG's small-M f16 XMX
matmul corrupts local rows ≥1 while row 0 stays correct.** The STABLE counting-sort packing makes token 0
always row 0 of every expert → **token 0 was bit-exact at every layer** (THE key diagnostic — token-0 residual
byte-identical batched-vs-T=1), every other token garbled. Fits ALL invariants: 20b (E=32→M~32 large-M-safe)
correct; Coder (per-expert gemm only at T≥4096, large-M) correct; decode (gemv not gemm) correct;
chunk-monotonicity (PPL 42@chunk128 / 19.78@chunk200 = M-magnitude signature). **FIX** (`src/model/gptoss_tp.cpp`
`moe_ffn_card`): replace the T>1 per-expert oneDNN GEMM (gate/up + down else-branches) with the validated **W4A8
int-dot `gemv_mxfp4_soa_q8` looped per packed row** (the decode kernel) — removes the small-M gemm AND the reused
`moe_btf16` dequant scratch in one shot; bit-faithful to the decode path. **VALIDATED: 120b batched PPL 123 →
15.1985 == T=1 15.1985 (avg NLL 2.721199 BIT-IDENTICAL); crown 6.4527 gate; 20b 2-card 19.36 no-regression;
build green.** Batched prefill is now the **DEFAULT** (opt-out `IE_GPTOSS_TP_T1_PREFILL` → slow T=1 fallback).
Per-row gemv re-reads weights (not XMX-optimal at large chunk) → **speed-optimal follow-up = pad-M GEMM**
(Coder's `moe_xmx_prefill` round-up-to-32 + copy-first-n_e pattern, qwen3moe.cpp), once correctness banked.
Added gated diagnostic `IE_GPTOSS_TP_XDUMP` (token-0 per-layer checksum — the tool that cracked it). Confined to
`gptoss_tp.cpp` (2-card gpt-oss only); crown/coder/gemma/27B/80B untouched. ⚠ LESSON: repeated 120b loads
thrash the box (loads degraded 67s→426s) — let it cool between runs; and a gross residual checksum is masked by
gpt-oss's NORMAL massive-activations (the correct 20b spikes bigger at L6), so the DECISIVE diagnostic was the
token-0 bit-exact comparison, not magnitude.
**⚡ PREFILL SPEED (after correctness): 34 → 433 tok/s (12.7×) on the 12k-token bench, 2×B70 @65k ctx.** Three
levers: (1) per-row int-dot gemv fix → correct but launch-bound (63 tok/s, `cc6b23a` superseded); (2) **pad-M
MoE GEMM** — small-M experts (n_e<32) padded to M=32 into `moe_pout` + copy-first-n_e, large experts direct;
restores XMX (weights read once), PPL still 15.1985 bit-identical (`cc6b23a`) → 127 tok/s; (3) **prefill chunk
256→4096** — gpt-oss isn't DeltaNet so the §1 256-cap never applied; the tiny chunks were the wall (re-dequant
all experts + TP all-reduce per chunk). 4096 = measured sweet spot (433); **8192 spills the MoE workspace to
host RAM and REGRESSES (301, decode thrashes to 0.1)** → 4096 is the VRAM ceiling at 65k ctx, opt-out
`IE_GPTOSS_PREFILL_CHUNK`. >500 would need freeing VRAM (shorter ctx) or compute (dequant/attention) work —
the chunk lever is VRAM-capped here. coherent ("Paris"), no OOM at 4096.
**⚡⚡ >500 HIT — prefill 538-679 tok/s DEFAULT at moderate ctx (the "truly viable" bar).** `IE_GPTOSS_TP_TIMING`
profiling of the BATCHED prefill (modified `ie-gptoss-tp-bench` to forward(T>1) + report) showed it is
**COMM-bound, NOT compute**: the two host-bounced all-reduces (P2P HW-blocked on 2×B70) = **67.6%** (attn 33.4 +
MoE 34.2), MoE compute 24.4%, attention 6.1%. The MoE all-reduce is fundamental to expert-TP, but the **attn
all-reduce is pure head-shard overhead** — **Phase 1 (replicate attention) drops it entirely**. AUTO-SELECT
(`gptoss_tp.cpp:359`): replicate attn (Phase 1) when the full KV fits (`max_ctx ≤ 16384`), head-shard (Phase 2,
halved KV) for long ctx; force via `IE_GPTOSS_TP_REPLICATE_ATTN`/`IE_GPTOSS_TP_HEAD_SHARD`. **VALIDATED 120b
2×B70: PPL 15.19 (== head-shard 15.20, TP-rounding); prefill DEFAULT @16k ctx 538 (12k prompt) / 679 (4k bench)
vs 433 head-shard = 1.24-1.54×; decode ~31 unchanged; "attn replicated" auto-fires, coherent, no OOM (card0
29.2GB+2.5 spill). 32k ctx Phase-1 OOMs (full KV) → head-shard, hence the 16384 gate.** Cumulative 120b prefill:
**34 (T=1 workaround) → 538-679 tok/s (~16-20×)** at moderate ctx, batched-correct. ⚠ box degraded to ~9-min
loads from the day's testing (clean-box rule) — let it cool. Further (past ~700): the MoE all-reduce (34%) is
the floor; would need a lower-precision/overlapped all-reduce or P2P (HW-blocked).

### ✅ DONE 2026-06-30 — crown all-Q8_0 on 2×B70 RUNS + COHERENT + BEATS llama BOTH AXES (kQwen35Moe layer-split)
**Goal met:** the Qwen3.6-35B-A3B crown now runs as an **all-Q8_0 GGUF (~36 GB) layer-split across 2×B70**
(`LuffyTheFox/...-Genesis-V4-Q8_0.gguf`), the first MoE-arch layer-split in the engine. New additive class
**`Qwen35MoeSplitModel`** (`qwen35moe_split.{hpp,cpp}`) + a fully GPU-resident **Q8_0-expert MoE** (`moe_q8.cpp`:
`gate_up_silu_q8`/`down_q8`/`reduce_q8`, int-dot over the `block_q8_1x` activation stream, top-k stays on device).
Forked from the 27B `Qwen35SplitModel` orchestration with the crown's DeltaNet+full-attn hybrid math; single-GPU
`QwenModel` is **never touched** (crown PPL 6.4527 gate safe — re-verified this session). Engine wiring: `kQwen35Moe`
added to the mgpu planner + a `--gpus>1` load branch + `forward_step` host-bounce; `ie run --gpus 2 --gguf <q8>` (commit `882207a`).

**The blocking bug (DEVICE_LOST on decode) was ROOT-CAUSED + FIXED — NOT a DeltaNet/§1 issue:** decode hung on
**card-1's first MoE layer (L20)** while prefill across all 40 layers was clean. Cause = `moe_router`'s `T==1`
(decode-only) path held a single process-wide **`static float* logits_buf`**, allocated in **card-0's** L0 context on
the first decode call; card-1's `moe_router` then dereferenced that **foreign-device pointer** →
`UR_RESULT_ERROR_DEVICE_LOST` (host spins in Level-Zero). Confirmed by gdb backtrace (hang in `queue::memcpy` right
after `moe_router`) + the exact card-0-fine / card-1-faults / prefill-clean signature. Fix = key the scratch **per
device** (`unordered_map<sycl::device,float*>`, mirrors `gemm_onednn.cpp:ctx_for`); single-GPU keeps one entry →
byte-identical, and it hardens **any** future multi-device MoE path (gpt-oss TP, etc.). `src/ops/moe.cpp`.

**Validated (2-card Q8):** **PPL 6.3604** (avg NLL 1.8501, 511 streaming T=1 decode steps across both cards) — BELOW
the Q4_K_M crown's 6.4527 as expected (more bits) and under the ≤6.4527 gate; the `Qwen35MoeSplitModel` PPL branch is
now wired into `ie_perplexity` (`14e41c0`, mirrors the gptoss-TP host-bounce — run `ie-perplexity --gguf <q8> --gpus 2`).
Coherent — "capital of France is Paris", 5 correct Mars facts (88-tok decode, no late-position fault), 25×4=100,
tides=Moon+Sun gravity; multi-turn KV+DeltaNet across cards clean; **~60 tok/s decode, 124–424 tok/s prefill**
(host-bounce split). Loads 18.44 GB/card.

**✅ BEATS llama-SYCL on BOTH AXES (2026-06-30, `588e883`).** Head-to-head 2×B70, same Q8 GGUF, layer-split
(`llama-bench -p512 -n128 -sm layer` vs `ie-qwen35moe-bench`): **prefill 963 vs 763 = 1.26× WIN, decode 63 vs 42 =
1.49× WIN.** Decode already won (our int-dot Q8 grouped MoE; llama's DeltaNet decode on SYCL is weak). Prefill was the
gap (510 = 0.67×): `IE_Q35MOE_TIMING` showed MoE = 80-82% of pp512 because the per-token-slot int-dot decode kernel
re-reads each expert's Q8 weights ~16× (TK/E) at T=512. Fix = **expert-batched weight-stationary Q8_0 MoE PREFILL**
(`moe_prefill_gate_up_silu_q8`/`moe_prefill_down_q8` in `moe_q8.cpp`; T>1 only, decode untouched): `moe_router` → host
counting-sort by expert → `moe_gather_rows` → batched GEMM (one WG per (expert,col-chunk) loops the expert's routed
rows in M_TILE=16 tiles, each weight column read once) → `moe_prefill_reduce_sum`. **pp512 509→963 = 1.89×**, decode
unchanged, **PPL bit-exact** (old vs new prefill-chunk avg NLL 2.122657 to 6 decimals; opt-out `IE_Q35MOE_NO_PREFILL_GEMM`).
Harness: `ie-qwen35moe-bench` (2-card pp/tg) + `IE_Q35MOE_TIMING` probe. Single-GPU crown PPL 6.4527 untouched.

### ✅ DONE 2026-06-28 — gpt-oss TOOL CALLING wired + validated end-to-end (Harmony agent loop works)
**The gap was real and broad:** `Engine::chat` hard-errored on `tools_json` for SIX arches
(Mistral/DeepSeek/Granite/Gemma/gpt-oss/llama — "tool calling not supported (v1)"); tools worked ONLY on the Qwen
`<tool_call>` path. That was an unfinished `// v1` shortcut, never a design choice. **gpt-oss is now wired** (the arch
the owner hit the wall on, loaded daily), reusing the OpenAI server's existing structured-`tool_calls` emitter with
**zero server changes**:
- **RENDER** (`tokenizer.cpp`: new `append_harmony_tools` + `harmony_schema_to_ts`, threaded into `build_harmony_prompt`):
  the `tools` array → a Harmony `namespace functions { // desc\n type NAME = (_: {…}) => any; }` block in the DEVELOPER
  message (JSON-Schema→TS for string/number/integer/boolean/array/object/enum; required→optional `?`) + a system-message
  "calls go to the commentary channel" line.
- **PARSE** (`engine.cpp` gpt-oss post-process): decode the generation WITH Harmony markers, detect
  `…to=functions.NAME …<|message|>{json}<|call|>`, translate to the canonical
  `<tool_call>{"name":…,"arguments":…}</tool_call>` the server already parses → structured `tool_calls` + finish `tool_calls`.
- **MULTI-TURN**: prior assistant `<tool_call>`s re-render as commentary `to=functions.NAME …<|call|>`; `role:"tool"` results
  render back on the commentary channel addressed to the assistant → full call→result→continue loop.
- **Guard dropped**; `tools_json` defaulted empty in `build_harmony_prompt` → **no-tools gpt-oss chat is byte-identical**.
**VALIDATED (gpt-oss-20b, `ie serve :8080`, non-stream):** (1) outbound `get_weather` → `tool_calls[0]=get_weather
{"location":"San Francisco"}`, finish `tool_calls`; (2) round-trip (assistant call + `tool` result fed back) →
"…foggy, 61 °F, 12 mph", finish `stop`; (3) no-tools sanity → "Paris", finish `stop`. **Crown PPL 6.4527 bit-exact**
(every edit `arch_==kGptOss`-gated → provably crown-neutral), build green. Files: `tokenizer.cpp`, `include/ie/tokenizer.hpp`,
`engine.cpp`. **NEXT:** the same render+parse pattern unlocks the other 5 guarded arches — **llama is the obvious next**
(its template + a `<tool_call>`-style or native format).
**STREAMING wired in-engine 2026-06-28** (`openai_server.cpp`): gpt-oss is channel-structured, so the SSE provider
buffers (`harmony = arch()==kGptOss`: on_token suppressed) and at end emits the engine's post-processed `r.text` —
`chat_chunk_sse_tool_calls(r.text)` for a tool call, else `chat_chunk_sse(r.text)` for clean content — while every
other arch keeps live token-by-token streaming byte-identical. Behavior proven by a non-stream→SSE bridge
(`harmony-tool-proxy.py`) that returned correct streaming `tool_calls` (Tokyo) + content (Paris) against the live
120b; the in-engine path is the same transform, so the proxy retires on the next `ie serve` restart. Remaining
follow-up: built-in browser/python tools (only user `functions.*` wired); optional live per-token streaming of the
gpt-oss final channel (needs decode-loop channel tracking — deferred, buffered emit is correct for an agent).

### 🎯 IN-FLIGHT 2026-06-27 — gpt-oss-120b RUNS on OUR engine (2×B70 TP): loads safe + decode-CORRECT + fast; 2 open gen bugs
**⭐ UPDATE 2026-06-28 (this session) — 120b is now USABLE + FAST + PUBLISHABLE.** Bug (b) FIXED: the Harmony-chat "!" garbage was an **fp16 RESIDUAL OVERFLOW** (gpt-oss massive activations push the residual stream past fp16 max 65504 at ~L35 → inf poisons that position's KV → NaN cascade → all-NaN logits → argmax id 0 = "!"). Fix = `residual_add` saturates to ±65504 before the fp16 store (`src/ops/elementwise.cpp`, commit `38ea8f5`) — EXACT no-op for every in-range model so crown PPL stays 6.4527; proper follow-up = fp32 residual. Chat validated coherent across math (17×23=391) / creative / factual + multi-turn. **Decode 14.6→32.05 tok/s (2.24×):** async-scatter all-reduce (`d3a4aea`, decode was comm/host-bound) + gate+up MoE gemv fusion (`49afc6c`, MoE 62→53%, PPL→17.08). Built `ie-gptoss-tp-bench` (engine's first 2-GPU decode bench) + gated `IE_GPTOSS_TP_TIMING` / `IE_GPTOSS_TP_NANCHECK` tracers. **Remaining (NON-blocking): bug (a)** batched T>1 prefill @E=128 — DEFAULT T=1-prefill workaround (correct output, only first-token latency; opt-in `IE_GPTOSS_TP_FAST_PREFILL`). **Publish blocker CLEARED.** NEXT: finalize `docs/public/gptoss_120b_tp_2026-06-27.md`; future (idle-box, measured): grouped-MoE kernel toward ~40 tok/s + fp32 residual. Tried+reverted (noise-bound on a busy box): grouped-down kernel, wpk-async. HISTORICAL (pre-fix) account follows. ↓

The full MXFP4 120b (2-shard) now loads + runs on our engine via the TP path. **What's VALIDATED:** multi-shard loader opens it (687 tensors); the **host-RAM spill caps card 0 at 29.3 GB → NO crash, display safe** (the owner's hard requirement); **decode is numerically CORRECT — 120b streaming PPL 12.91 (NLL 2.56), faster than the 20b as expected**; decode **14.6 tok/s > LM Studio's 12.42**. Both cards compute (head-shard). Parallel repack → ~6× faster loads. Bugs found+fixed this session: Phase-2 Bt-transpose (was [N,K], must be [K,N]); `dequant_mxfp4_soa_to_Bt` N%64 nd_range; spill host-fallback on device-OOM; `moe_btf16` reuse deps. **2 OPEN BUGS (the showstoppers for a usable 120b chat):** (a) **batched T>1 prefill MoE is DEGRADED at E=128** — PPL 42 @chunk128(==sliding_window), 19.78 @chunk200, vs streaming 12.91 (should be *better*); 20b prefill is fine at all chunks → it's E=128 + the T>1 MoE prefill, NOT spill/multishard/attn (all ruled out by isolation tests). WORKAROUND active: process prompts as a sequence of correct T=1 steps (opt-out `IE_GPTOSS_TP_FAST_PREFILL`); slower first-token, fixes the prefill PPL. (b) **free-generation produces garbage (constant "!") after the HARMONY CHAT prompt** — and the T=1-prefill workaround does NOT fix it, so it is NOT the prefill; the forward is PPL-correct and the 20b chat works ("Paris"), so it's specific to the 120b's prediction after the Harmony assistant-turn / special tokens (hypothesis: special-token handling at 120b scale, though token_embd is shard-1/validated). NEXT: crack bug (b) (likely add a greedy-gen debug to dump the 120b's argmax right after the Harmony prompt; compare special-token embeddings 20b vs 120b), then fix (a) the T>1 prefill MoE. **COMMITTED 6 commits `5348dca`→`9fd9372`** (P1+P2+P3 + multi-shard loader + parallel repack + the 4 found-and-fixed bugs + the T=1-prefill workaround + the publish-doc draft). Files: `gptoss_tp.{hpp,cpp}`, `gguf_reader.cpp`+`gguf.hpp` (multi-shard), `gemv_mxfp4.cpp` (N%64 fix), `engine.{cpp,hpp}`, `ie_perplexity.cpp` (--gpus), `src/CMakeLists.txt`.

**Reusable deliverables from this session's 5-agent fan-out (not yet applied):** (1) **decode-perf opts** — the 120b decode bottleneck is ~252 host-blocking syncs/token (2 host-bounced all-reduces/layer × 36 + the per-layer routing D→H→H→D bounce), NOT compute; ranked low-risk wins: build the MoE packing once/layer into a persistent pinned buffer + async the per-card upload (drop the `.wait()`), use `sycl::malloc_host` for the all-reduce staging (`allocator.cpp` ar_stage_/ar_acc_/ar_out_ are pageable), iterate only active experts, then attn q/k/v/o Q8_0 decode; est. ~15–25% combined, each must hold cosine≥0.999. (2) **bench matrix** — 11 GPU-serial runs (ctx sweep 512/2k/8k/32k/99k + head-shard A/B new-old-new @32k + spill-reserve sweep @2k + PPL correctness), ~1–1.5 h, ready to run once (a) and (b) are fixed. (3) **publish doc** `docs/public/gptoss_120b_tp_2026-06-27.md` — DRAFT; the chat blocker is now CLEARED (bug (b) fixed) so it is publish-ready pending a finalize pass (drop the "do not publish" banner, fill the results table with 32 tok/s / 2.24× / PPL 17.08 / coherent-chat transcript / the residual-overflow fix as the technical highlight).

**Isolation evidence for bug (a)** (so the next session doesn't re-walk it): 20b prefill is correct at EVERY chunk (256→16.82, 128→20.40, 64→20.27 — all sane); 120b prefill PPL 42 @chunk128 / 19.78 @chunk200 / 12.91 streaming(T=1). `IE_GPTOSS_NO_PROJ_GEMM=1` (proj+router→gemv) stays 41.65 → NOT the attn-proj/router GEMM. Threading the per-expert gemm events through the reused `moe_btf16` dep did NOT change it → oneDNN is in-order, NOT a scratch race. So (a) is specifically the **E=128 batched T>1 MoE-expert path** (`dequant_mxfp4_soa_to_Bt`→per-expert `gemm_fp16_onednn` in `moe_ffn_card`). Next probes: per-layer hidden-norm trace to find the diverging layer; compare a single expert's batched-T GEMM output vs T=1 gemv for the same rows; check the small-M/many-expert oneDNN regime (the same chunk-cap regime the 80B/Gemma MoE-XMX hit). **Bug (b)** next probe: greedy-gen the 120b and dump argmax+top-5 immediately after the Harmony prompt; diff the special-token (`<|start|>`/`<|message|>`/channel) embeddings + lm_head rows 20b vs 120b (token_embd is shard-1, ie-inspect-validated, but the *forward through* the special tokens at 120b scale is the untested bit). Both are OPEN findings — the forward is PPL-correct, so this is a generation/prefill-path bug to crack, not a wall.


### ✅ DONE 2026-06-27 — gpt-oss MULTI-GPU tensor-parallel (Phase 1) VALIDATED + multi-shard GGUF loader
Lead greenlit "wire up multi-GPU TP for gpt-oss" (to run the 120b on OUR engine). The engine's **FIRST MoE tensor-parallel path**. **VALIDATED on the 20b across 2×B70 (de-risk): coherence "Paris" ✓; 2-card TP PPL ≈ single-card on BOTH axes — decode 19.31 vs 19.36 (0.26%), prefill 16.85 vs 16.82 (0.18%) — so the MXFP4 expert sharding + host-bounce all-reduce are numerically correct AND oneDNN stays bit-identical across cards (the one review concern, cleared); crown PPL 6.45 ≤6.57; ctest 31/31; single-card gpt-oss PPL unchanged (the gemv_mxfp4 fix is bit-identical for N=2880).** (20b TP decode 41 tok/s < single-card 58 — EXPECTED: the 20b fits one card so TP only adds all-reduce overhead; TP pays off on the 120b which doesn't fit + at 99k where prefill dominates.)
- **`GptOssTpModel` (NEW: `include/ie/gptoss_tp.hpp` + `src/model/gptoss_tp.cpp`)** — the engine's FIRST MoE tensor-parallel path. **PHASE 1 = MoE expert-TP only, attention REPLICATED** (the proven `qwen35_tp` Phase-0 pattern: x stays bit-identical across cards from replicated attn → NO attention all-reduce). MXFP4 experts sharded: gate/up column-parallel (EF→efc), down row-parallel (EF→efc on 32-blk boundaries) → ONE host-bounce all-reduce/layer; `down_bias` rank-0-only; routing computed once on card 0 + broadcast. New MXFP4 slicers `repack_exps_col`/`repack_exps_row`. 6 engine edits (kGptOss in the `mgpu` predicate @engine.cpp:54; the load branch gates `n_gpus>1`→`gptoss_tp_model_.load`; `forward_step` host-bounce branch; `vocab()`; members). CMake += `model/gptoss_tp.cpp`.
- **Adversarial review (4 agents) found 1 CRITICAL bug → FIXED:** `dequant_mxfp4_soa_to_Bt` (`gemv_mxfp4.cpp:224`) launched `nd_range<2>({bpc,N},{1,64})` requiring N%64==0; the gate/up PREFILL slice N=efc=1440 (1440%64=32) → SYCL throws. Fix = round the dim-1 global up to WG (the dead `if(n>=N)return` tail guard now lives); **provably bit-identical for single-GPU (N=2880, round-up=N)**. Everything else (bias rules, swiglu-on-slice, all-reduce linearity, routing-once, attn/residual replication invariant, all 6 edits, lifecycle, sizing, aliasing) reviewed CORRECT. One open VALIDATION note: the replicated-attn path uses oneDNN (template disabled it) → cross-card cosine must be checked (expected bit-identical on identical B70s).
- **Multi-shard GGUF loader (DONE + CPU-validated):** `gguf_reader.cpp` `open()` refactored into a per-shard `parse_shard(base,size,primary)`; detects `split.count`>1 (filename `-NNNNN-of-MMMMM.gguf` convention + KV cross-check), mmaps siblings (kept alive in `extra_shards_`), merges tensor tables resolving each tensor against ITS shard's base. **`ie-inspect` on the 2-file 120b → 687 tensors merged, blk.22 (straddles boundary) fully resolves, config correct (36L/128E/EF2880); single-file 20b unchanged (459 tensors).**
- ✅ **PHASE 1 + PHASE 2 VALIDATED + committed.** Phase 2 = attention head-shard (q/k/v column, o-proj row, sinks+biases+KV sliced → **both cards compute attention**, halved KV/card, 2 all-reduces/layer). 20b/2-card: head-shard streaming PPL 19.49 vs single 19.36, prefill 16.83 vs 16.82, "Paris" ✓, crown 6.45, ctest 31/31. opt-out `IE_GPTOSS_TP_REPLICATE_ATTN` → Phase 1. **LESSON: the F16 attn slice must emit Bt=[K,N] (out[k*N+n], what `dequant_q8_0_to_Bt` produces + gemm/gemv consume) — NOT [N,K]; a transpose there → worse-than-random PPL (caught by the PPL A/B).** **NEXT: (3) PHASE 3 host-RAM spill** (experts ~28.4GB/card + attn + KV slightly exceed 32GB even with Phase 2 → ~3GB/card spill, matches owner's "~10% to RAM"). Then the 120b run: `./build/src/ie run "<...>-MXFP4-00001-of-00002.gguf" --gpus 2 --ctx 99000` (multi-shard loader DONE → it opens; needs Phase 2+3 to FIT @99k). VRAM math (120b 2-card): experts 56.7GB→28.4/card, attn-F16 replicated ~1.9/card, KV@99k replicated ~7.3/card → Phase-1 alone fits the 20b de-risk only; Phase 2 halves attn+KV, Phase 3 spills the expert overflow. Files: `gptoss_tp.{hpp,cpp}`, `gemv_mxfp4.cpp` (kernel fix), `gguf_reader.cpp`+`gguf.hpp` (multi-shard), `engine.{cpp,hpp}`, `ie_perplexity.cpp` (--gpus), `src/CMakeLists.txt`.

### 🏆🏆 DONE 2026-06-27 — gpt-oss-20b BEATS llama BOTH AXES + 120b RUNS on 2×B70 (port=next-work) + publishing re-verified
**Headline:** gpt-oss-20b now wins both axes on 1×B70 vs CURRENT llama-SYCL (fa1 best): **prefill pp512/2K/4K
1795/4147/3428 vs 927/927/896 = 1.94×/4.47×/3.83×; decode tg 58/57/56 vs 50/50/49 = 1.13–1.16×**, both flat.
9 commits `86f9a54`→`9665f2f`. The 7 levers: MXFP4 fused decode GEMV (SoA repack, the 17-byte-stride trap →
64 GB/s misaligned), split-K FA-2 decode + sink, wide-tile prefill + sink (SINK as `if constexpr` template →
crown byte-identical), W8A8 lm_head, **batched prefill projections + router** (the BIG find: q/k/v/o + router
ran as per-TOKEN gemvs at prefill — 196k+49k calls = >90% of prefill — because `dense::gemv_q_T`'s batched
branch is gated `K%256 && N%64` which H=2880/E=32 fail; fix = `gemm_fp16_onednn` directly → prefill 5.5×).
Crown 6.4527 bit-exact throughout, ctest 31/31. Detail: [[gptoss-prefill-pertoken-gemv]] in agent-mem.
**⚠ The old "beats both axes 668/607, 31.9/22.35" banner was vs a STALE llama** (current llama-SYCL is
908/49, much faster) — re-verified fresh. Publishable write-ups: `docs/public/gptoss_benchmark_2026-06-27.md`
+ `docs/public/gptoss_optimization_deepdive_2026-06-27.md`; COMPETITIVE_SCORECARD updated.

**✅ gpt-oss-120b RUNS on 2×B70 — OWNER CONFIRMED (full MXFP4, 100k+ ctx, 12.42 tok/s decode, usable).** One
B70 maxed, the other ~29/32 GB; the runtime (llama/LM Studio) fit the full ~63 GB across the two cards (KV is
cheap — GQA-8 + windowed). So the 120b is a REAL target. (My earlier "doesn't fit / dead-end" was WRONG — the
0.7×/card margin was not the owner's constraint; he's fine with ~10% spilling to system RAM. Retracted; see the
feedback memory `feedback-no-unilateral-dead-ends`.) **THE OPPORTUNITY: run it on OUR engine — same arch family
as the 20b, every kernel carries over, and at 100k ctx PREFILL dominates first-token latency, exactly where we
post 4.5× MXFP4 prefill. The demo: "117B, 100k ctx, first token in seconds, on a 64 GB box."**
**⚠ OUR ENGINE CANNOT RUN IT YET (verified):** `kGptOss` load (engine.cpp:184) is SINGLE-CARD only — no gpt-oss
layer-split exists (only qwen35/qwen3next do) → 63 GB on one 32 GB card OOMs; AND the 120b is a 3-shard GGUF and
our reader has no multi-shard support. **NEXT-WORK to run it: (1) build `GptOssSplitModel` (~80% reusable from
Qwen35SplitModel — LayerPlan + copy_across host-bounce verbatim; new = gpt-oss Layer + MoE workspace) + multi-
shard GGUF loading; (2) DE-RISK by forcing the 20b onto 2 cards (`--gpus 2`) to prove the split is bit-correct
FIRST; (3) then load the 120b.** Target command: `./build/src/ie run "<...>-mxfp4-00001-of-00003.gguf" --gpus 2
--ctx 99000`. Owner has the full MXFP4 (~63 GB) + a pruned REAP-58B (~41.3 GB) downloaded; other variants:
meshllm UD-Q4_K_XL layer-split, majentik RotorQuant Q8 (~30-piece, Cl(3,0)-rotor KV), noctrex Huihui MXFP4 BF16.
**RESEARCH THREAD (lead-opened — research+plan phase, NOT started, do NOT build unprompted):** our OWN new method
— MoE **expert-tiering / precision-follows-usage** (hot experts resident at working precision, cold experts
compressed harder + spilled to RAM, overlap the fetch — the smart version of "offload a few layers"; a pruned
model is the static version of this) and/or a rotor/geometric KV compression. First de-risking step is CPU-only:
profile the real expert-usage skew on the 120b GGUF (is it Zipfian enough that a small hot-set covers ~90%?).

**Publishing re-verification (the gpt-oss stale lesson generalizes):** llama-SYCL improved ~10-20% on mature
arches. Coder pp4096 1.24× holds but pp512 0.62× + decode 0.65× LOSS; 27B pp512/4K/8K 1.18×/1.63×/2.74× (was
1.96–3.08×) still a long-ctx win, decode 0.72× loss. **gpt-oss-20b is the freshest + strongest claim.** Re-bench
EVERY claim vs a current llama before publishing.

**attn q/k/v/o Q8_0 decode — NOT shipped this session (open).** OBSERVED: a first wiring of `gemv_q8_0_soa_q8`
for the attn dims (K=2880 N=4096 q-proj, K=4096 o-proj) hung at decode, though the same kernel works for the
lm_head (K=2880 N=201088). I reverted to keep the box stable; the hang's cause is unknown and worth a look
(a kernel dimension assumption?) — this is a finding to investigate, not a closed door.
**Candidate next levers (lead's call): gemv_mxfp4 ALU-co-limited (dp4a_us+offset was slower in one A/B — re-test
welcome); SWA decode layers (small); a fitting big-MoE re-verify (80B vs current llama); the 120b via any of
the fit paths above; and the lead-opened RESEARCH thread — a NEW quant / KV-compression / attention / expert-
tiering method (own AWQ/GPTQ/rotor/FA equivalent). All OPEN, none closed.**

### 🏆 DONE 2026-06-26 (night) — gpt-oss family (8th arch): STEP 1 loader + STEP 2 forward COMPLETE + validated
**Goal (user's actual ask):** run a **120B-class model on 64 GB** — the EXL3-revisit thread was about FITTING a
larger model (low-bpw), not the 80B. Survey → `docs/big_model_roadmap_2026-06-26.md`; full arch spec →
`docs/gptoss_arch_spec_2026-06-26.md`. **Decision: build the gpt-oss family**, prototype on **gpt-oss-20b**
(downloaded `~/models/gpt-oss-20b-GGUF/gpt-oss-20b-mxfp4.gguf`, 12 GB). Quality/fit research verdict: gpt-oss-20b
is the 64 GB sweet spot (MMLU 85.3 > Nemotron-70B 83, AIME 98.7, ⅓ the size, 50 GB free, fast — 3.6B active);
gpt-oss-120b (~63 GB) is the quality ceiling but a TIGHT/short-ctx fit on 64 GB (KV is cheap — GQA 8-kv +
windowed — so it's a model-weight squeeze, not a context one); Nemotron-70B is the comfortable drop-in
(`nemotron→llama→kLlama3`, zero code) but a 2024-era 70B.
**Shipped (5 commits, `1ee4a2e`→`a61443a`):**
- ✅ **MXFP4 dequant** (`1ee4a2e`): `block_mxfp4` + `dequant_mxfp4[_to_Bt]`, bit-exact vs llama (ctest 31/31).
- ✅ **STEP 0** (`bf70cdd`): `kGptOss` + `detect_arch` + `GptOssConfig` + `read_gptoss_config`.
- ✅ **STEP 1 loader** (`d931b13`): forked `GptOssModel` (gptoss.hpp/cpp) on the qwen3moe structure (NOT a
  qwen3moe branch — that has zero bias plumbing; fork keeps Coder/crown bit-exact). Loads Q8_0 attn q/k/v/o +
  F32 biases, F32 `attn_sinks[64]`, `post_attention_norm`, F32 router+bias, MXFP4 gate/up/down experts + F32
  per-expert biases, Q8_0 `token_embd` (lookup_q8_0) + Q8_0 lm_head (→fp16). Exempted gpt-oss from the dense
  SWA load guard. Engine wired (kGptOss load/vocab/forward); `ie-gptoss-load-test` proves `LOAD OK vocab=201088`.
- ✅ **STEP 2 forward** (`a61443a`): the 3 new pieces, all working. (1) `full_attention_gptoss` — a COPY of
  `full_attention` (crown's stays byte-identical) with per-head softmax **SINK** folded into the denominator
  only (`l'=l·corr+exp(s−m_f)`, mass leaks out) + sliding **WINDOW** (even layers 128; loop-start `wstart=
  max(0,p+1−w)`); used for prefill AND decode (T=1 naive; split-K+sink is a later perf lever). (2) `swiglu_oai`
  — new op, **bit-exact vs ggml SWIGLU_OAI** (α=1.702, limit=7.0; gate clamped above, up clamped both sides,
  (up+1)). (3) `moe_ffn` — top-4 **MXFP4 oneDNN** per-expert (dequant_mxfp4_to_Bt→gemm_fp16_onednn, the ONLY
  path — MXFP4 has no int-dot) + router/gate/up/down biases + **SOFTMAX_WEIGHT** routing (≡ route_from_logits +
  router bias, verified vs llama-graph.cpp:1502-1578). **DECODE-GARBAGE BUG FOUND+FIXED:** `host_routes_` was
  max_T-sized + persistent; `build_moe_packing` uses `routes.size()` → stale prefill routes corrupted the T=1
  packing (prefill worked because the first call's stale slots were empty). Now resized to exactly T.
**VALIDATED (STEP 3):** greedy "France is"→" Paris.", "Japan is"→" Tokyo" (correct factual argmax at pos 4 & 11,
no repetition); per-token NLL healthy (low on predictable, high on content); position-bucketed NLL has NO SWA
jump + NO YaRN runaway; **our tokenizer 586 tok = llama's 586 (IDENTICAL)**; streaming PPL **17.9** vs
`llama-perplexity` chunked **54.7** on the same corpus → NO forward deficit (broken fwd ≫50; the absolute PPL is
high because gpt-oss is reasoning-tuned). crown **6.4527 bit-exact**, ctest **31/31**, build green. Decode **38
tok/s** (functional). `ie-perplexity` + `ie-gptoss-load-test` gained kGptOss branches.
**UPDATE 2026-06-27 — benchmark WIN + Harmony + YaRN (5 more commits `b8240c4`→`f8ef2af`):**
- ✅ **H2H benchmark — BEATS llama-SYCL both axes** (1×B70, `ie-bench` gained a kGptOss branch == `llama-bench`):
  prefill pp512 **668 vs 607 = 1.10×**, decode tg128 **31.9 vs 22.35 = 1.43×** — with UNoptimized kernels.
  Decode kernel profile (the perf-lever map): `dequant_mxfp4` **44%** + `gemv_fp16` 28% + `attn_gptoss`
  23%→**55%@2180** (naive O(ctx)). `ie-bench`/`ie-perplexity` both have kGptOss branches now.
- ✅ **Harmony chat SHIPPED** (`4009238`): `build_harmony_prompt` (tokenizer.cpp) + kGptOss `Engine::chat`
  dispatch (enable_thinking→effort) + stop set (`<|call|>`=200012; eos `<|return|>`=200002; `<|end|>` NOT a
  stop) + final-channel extraction (`res.text` = tokens after last `<|message|>`). Validated: "capital of
  France?" → stream `[analysis: Need answer: Paris. → final: Paris]`, `res.text`="Paris", finish=stop.
  gpt-oss now usable as the reasoning model it is. (Streaming still shows analysis live — clean-stream is v1 TODO.)
- ⚖ **YaRN investigated** (`b8240c4`): impl `rope_yarn` (ggml-exact NEOX, mscale=1.34657 VERIFIED vs
  llama-context.cpp:158-169) — but MEASURED plain rope base-150000 BEST ≤4096 (streaming PPL **17.9** vs 30.5
  full-YaRN / 22.1 mscale-only on the 586-tok corpus): factor-32 YaRN perturbs short-ctx. **Plain rope DEFAULT,
  YaRN OPT-IN `IE_GPTOSS_YARN`** for >4096 (where plain extrapolation fails). llama's chunked 447(off)/54.7(on)
  is methodology (`--rope-scaling none` breaks it), not a clean baseline. Diagnostic `IE_GPTOSS_ROPE_NOINTERP`.
**STILL REMAINING (perf extends the WIN, + the goal):** (a) ✅ **MXFP4 decode GEMV — DONE 2026-06-27**
(the 44% lever): SoA repack (aligned `qs`/`e` planes, memory-flat) + W4A8 int-dot `gemv_mxfp4_soa_q8`
(`dp4a_ss`, branchless FP4 decode — the byte-permute LUT is sw-emulated/slow on Xe) reading the 4.25-bpw
weights ONCE; prefill via `dequant_mxfp4_soa_to_Bt`. **Decode 30.5→43.0 tok/s = 1.41×** (f16 ref 41.5),
PPL-neutral (q8 22.52 vs oneDNN 22.53), crown 6.4527 bit-exact, ctest 31/31. gemv_mxfp4 8.2 ms (was
dequant 11.8 + gemm slice). gpt-oss decode ~1.9× vs llama. Opt-out `IE_GPTOSS_NO_MXFP4_GEMV`, faithful
`IE_GPTOSS_MXFP4_F16`. New `src/ops/gemv_mxfp4.cpp`. Sub-remaining: gemv_mxfp4 at ~154 GB/s still has
headroom (FP4-decode ALU / MMV-vs-fat-WG geometry / SG32). (b) ✅ **hd64 DECODE FA-2 (sink) DONE 2026-06-27** — `attn_gptoss` was the dominant decode cost (62.7%
@2K). COPY of `full_attention_fa2_decode` + per-head sink (combine fold) for the full-attn layers; SWA
layers stay bounded-naive. **Decode @2K 25.3→53.9 = 2.13× (1.31× @512), flat across ctx**, PPL-neutral,
crown bit-exact, ctest 31/31. Opt-out `IE_GPTOSS_NO_FA2_DECODE`; fixed `ensure_attn_partials` /256→Bc64.
✅ **PREFILL wide-tile DONE 2026-06-27** (`full_attention_gptoss_prefill_tile` = `fa2_tile_wide_impl<64,
SINK=true>`, threaded `bool SINK` template param → crown byte-identical): prefill @4K 327→716 = 2.19×
(1.37× @2K), flat across ctx, crown 6.4527 bit-exact, PPL +0.16%, opt-out `IE_GPTOSS_NO_FA2_PREFILL`.
Sub-remaining: gemv_mxfp4 geometry (154 GB/s), the SWA decode layers (small), prefill-tile Bc=32 occupancy A/B. (c) **120b** — OPEN (lead's call): full MXFP4 ~63 GB is ~31 GB/card of weights (~90% VRAM + ~10% system-RAM
spill is acceptable to the owner), or a pruned/layer-split/alt-quant variant, on a 2-card `GptOssSplitModel`.
(d) gpt-4o tokenizer edge cases (prose matches llama 586=586 today). Note: the SoA repack + signed-dp4a +
branchless-FP4 recipe is REUSABLE for any future MXFP4 arch.

### ✅ NEW 2026-06-26 (night) — per-device oneDNN ctx map: the multi-card DEVICE_LOST landmine is FIXED (STEP 1 of the 80B-prefill unblock)
Lead-greenlit. The block on ever running the crown MoE-prefill win (oneDNN large-M GEMM) on the 2-card
80B/27B paths was a **static singleton** in `src/ops/gemm_onednn.cpp:ctx_for`: it built ONE `dnnl::engine`/
`stream` from the FIRST queue it saw (card 0) and reused it for every call, so a card-1 GEMM routed card-1 USM
through a card-0-bound stream → `UR_RESULT_ERROR_DEVICE_LOST`. **FIX:** `ctx_for` now keys a
`std::unordered_map<sycl::device, OnednnCtx>` — one engine + stream + per-(M,N,K) primitive cache **per SYCL
device** (the user's "two identical oneDNNs"). Each card builds and uses only its own context; zero cross-device
sharing. Correctness rests on the codebase invariant that each device owns exactly one persistent queue
(`DeviceAllocator::queue_`), so device→(context,queue) is stable. **PROVEN:** new `ie-onednn-multidev-test`
runs `gemm_fp16_onednn` on both B70s INTERLEAVED (dev0,dev1,dev0,dev1) — no DEVICE_LOST, all results exact
(maxerr 0). **NO IMPACT on current work** (the user's other constraint): single-GPU = one device = one map
entry = byte-identical to the old singleton → crown **6.4527 bit-exact**, ctest **30/30**, build green. The
`prefer_onednn()=false` guards on the 27B-split/TP paths are **deliberately UNCHANGED** — this is enabling
*infra*, not a perf flip.

### 🏆 NEW 2026-06-26 (night) — 80B oneDNN MoE-prefill lever: WIN CONFIRMED 1.53× / 1.91×, DEFAULT-ON (Step 2).
Built on Step 1. Ported crown's large-M MoE-prefill recipe to the 2-card 80B (`qwen3next.cpp`), in three
validated commits:
- **2a (`2219c58`) — DeltaNet recurrence sub-chunk ≤512.** To feed the MoE a big prefill chunk, `forward()`
  must take T>512, but the recurrence must NOT run >512 steps/launch (§1 BMG-G31 HW-bug). Ported crown's loop:
  split the scan into ≤512 segments threading `state_layer` (bit-identical; NO-OP at T≤512; env
  `IE_QWEN3NEXT_DN_RECUR_CHUNK`). **VALIDATED `ie-qwen3next-ppl --sweep --repeats 3`: VERDICT CLEAN** — c1024
  bit-identical over 3 repeats (no collapse, no divergence) → over-cap prefill is §1-safe for the 80B.
- **2b (`e0c0d13`) — per-expert oneDNN MoE branch.** Dequant each expert's Q4_K gate/up + Q4_K|Q6_K down →
  fp16 Bt (`dequant_q{4,6}_K_to_Bt`, AoS) + `gemm_fp16_onednn` over expert-sorted rows → SAME
  `moe_prefill_reduce`. **Multi-card SAFE for free:** each layer calls `gemm_fp16_onednn(q,…)` on its card's
  queue → Step-1 per-device map gives each card its own engine. New buffer `moe_btf16`; up reuses `moe_h2`;
  per-layer `down_dt` dispatch. Opt-in `IE_QWEN3NEXT_MOE_ONEDNN` (+ `_MINT`, def 2048). **VALIDATED:** at
  chunk 1024 int-dot=0.001196 (byte-identical to 2a → default UNREGRESSED) vs oneDNN=0.001581 — coherent, NO
  NaN/collapse across all 48 layers incl. mixed down dtypes.
- **2c (`61a8e1b`) — engine enablement.** The flag raises the 80B engine pf_chunk to min(8192,max_ctx) so the
  lever engages through `ie run/serve` (default no-flag stays 512 — 80B path unchanged). **VALIDATED:** oneDNN
  chunk 2048 @ DEFAULT MINT → ppl 1.0394 coherent; crown **6.4527 bit-exact**; ctest **30/30**.

**✅ PERF CONFIRMED + DEFAULT-ON (`d48798a`).** Clean-box A/B (80B Q4_K_M, 2×B70, 8K prompt → forward(8192) at
M≈160, `ie-qwen3next-bench` --runs 3 median, new-old-new):
| config (8K prompt) | prefill t/s |
|---|---|
| today's default (chunk 512, int-dot) | 480 |
| big chunk 8192, int-dot | 599 / 603 / 596 (chunking alone = 1.25×) |
| **big chunk 8192, oneDNN MoE** | **919 / 915** |

→ **oneDNN vs int-dot @ same 8192 chunk = 1.53×** (isolated lever); **vs today's default = 1.91×** (end-to-end).
new-old-new int-dot variance 0.6% → real signal, not heat-soak. **M≈160 was NOT borderline — the lever wins
decisively.** Flipped **DEFAULT-ON for max_ctx≥8192**: the engine raises pf_chunk to min(8192,max_ctx) and
forward() engages oneDNN at T≥MINT (def 6144 = M≥120, inside the confirmed-winning band). Opt-out
`IE_QWEN3NEXT_NO_MOE_ONEDNN` (→596, int-dot). Validated: default-on auto-engages (no flag → 914.8); crown
6.4527 bit-exact; ctest 30/30. **Net: the 80B's long-ctx prefill goes from a structural 0.50× loss to ~915
tok/s @8K.** ✅ **llama-SYCL H2H DONE** (`llama-bench` `build-sycl` fdc3db9b6, SAME 80B GGUF, 2×B70,
ONEAPI_DEVICE_SELECTOR=level_zero:0,1): **llama pp8192 = 627.3 ±1.0 t/s; ours 915 = 1.46× FASTER.** The 80B's
structural 0.50× prefill loss is now a **1.46× prefill WIN over llama-SYCL** at 8K (our int-dot baseline 601 ≈
llama parity; the oneDNN lever is the differentiator). Downside of the lever: none — default path byte-identical,
short-ctx unchanged, correctness proven. **The DeltaNet moat now holds the 80B on prefill (1.46×) too.**

### ⚙ NEW 2026-06-26 (late) — SG32 wide-tile attention (opt-in): +5-7% long-ctx prefill, viable on BMG
Lead-greenlit the "attention rewrite." Rewrote the wide-tile FA kernel (`src/ops/attention.cpp`
`fa2_tile_wide_impl`) from subgroup-size 16 → **32** to capture llama's SIMD32 issue-efficiency edge, via a
**shadow template param** (`<HD,Bc,REGDOT,SG=SG_SIZE>` + `constexpr SG_SIZE=SG` shadow + `WG_ITEMS=Br*SG_SIZE`
→ whole body auto-retargets; 4 additive/gated edits). Design vetted by a workflow (register math: hot path
16→9 live fp32/lane, **no spill**, same 16 HW threads/WG). **THE empirical unknown — does IGC accept
`reqd_sub_group_size(32)` + occupy on BMG-G31 — is RESOLVED YES.** MEASURED (Coder hd128, warm A/B): 4K −1.4%
(attention-light), **8K +4.9%, 16K +6.9%**. Quality: Coder PPL **2.3686 == SG16 (IDENTICAL** to 4 dp), coherence
perfect, crown 6.4527 bit-exact (default SG=16; crown is hd256 so untouched), ctest 30/30. **HONEST: modest — does
NOT flip Coder 16K to a win (0.88→~0.92× vs llama), exactly the design's "parity-to-marginal-win, not a crush."**
Shipped **OPT-IN** `IE_FA2_TILE_SG32` (default-OFF; not bit-exact, −1.4% @4K). **✅ EXTENDED TO hd256** (lead-
requested): covers Gemma SWA layers + crown/27B/80B long-ctx tile. Gemma-31B prefill **117.6→120.6 = +2.5%**
overall (the SWA +5-7% diluted by the non-SG32'd hd512 GLOBAL layers + MLP; box was warm — noisy, clean re-bench
pending); coherence-correct, crown 6.4527 bit-exact, ctest 30/30. **✅ ALSO EXTENDED TO hd512** (Bc forced 16→32;
register-safe 21 fp32/lane) → **ALL gemma attention layers now SG32-capable.** hd512 = gemma's FULL-causal O(T²)
GLOBAL layers, which DOMINATE gemma-16K attention → the bigger lever. Correct (gemma all-layers SG32 coherent),
crown 6.4527 bit-exact, ctest 30/30. ⚠ **GEMMA PERF UNCONFIRMABLE (within bench noise):** cleaner-box A/B (box
recovered) gave SG16 115.5 / SG32 120.4 / SG16 120.1 — the two identical SG16 runs differ 4% (warmup drift), SG32
inside that spread → the +2.5-5% gemma gain is below the noise floor (gemma-31B prefill is inherently noisy here).
**The CONFIRMED SG32 number is hd128 +5-7% (Coder); the gemma hd256/512 extension is correct + mechanism-sound but
perf-unconfirmed.** **REMAINING (low-ROI):** `T≥6144` gate + default-on decision; a gemma-16K sweep on a truly
clean box. Detail: `docs/authority/coder.md` §8 #2.

### 🏆 NEW WIN 2026-06-26 (eve) — crown long-ctx prefill: oneDNN large-M MoE = 1.46× (the "dead end" REOPENED)
The earlier "crown/80B long-ctx prefill is a MoE-bound DEAD END" verdict (the `(ii)`/chunk-cap blocks below)
was **WRONG** — it rested on the chunk-cap lift's "regression at 2048", which was an **artifact of a fused-MoE
dispatch cliff**, not a real falsification. PROFILE (`ie-bench --kprofile`, crown): prefill is **~64% MoE running
at <1% of peak FLOPs** (small-M starvation, M=16 rows/expert at the 512-chunk). The fused multi-expert MoE was
hard-capped at **T<2048** (`qwen36.cpp:~1488`); above it the code serialized **per-token (M=1)** → THAT is what
regressed the chunk-cap test, so the large-M regime (M≥256, where a blocked GEMM amortizes the 256-expert weight
read — exactly where oneDNN won on Coder/Gemma) was simply **UNREACHABLE**.

**THE LEVER (validated, OPT-IN this commit):** feed crown a BIG prefill chunk (8192) so the MoE sees M=256, made
§1-safe by internally sub-chunking the DeltaNet recurrence to ≤512 launches (bit-identical sequential-scan split;
no-op at T≤512; **strictly safer** than the falsified chunk-cap lift — launches never exceed the validated 512),
+ raise the fused dispatch ceiling/scratch, + a per-expert **oneDNN MoE** (`dequant_q4_K/q6_K_to_Bt` → fp16 Bt +
`gemm_fp16_onednn` over the EXISTING expert-sorted buffers; output feeds the SAME `moe_prefill_reduce`).
EXPERIMENTS (REAL ~12K wikitext prompt, real routing, `ie-prompt-bench`): big-chunk + **INT-DOT** MoE REGRESSES
763→334 = **0.44×** (int-dot re-streams weights ~linearly with rows — Exp1: 15.1× time for 16× rows); big-chunk +
**oneDNN** MoE = 726.9→**1058.2 = 1.46× WIN**. H2H: llama-SYCL `llama-bench` crown **pp8192 = 910** (synthetic
= dummy routing, llama's BEST case) → our **REAL-routing 1058 WINS 1.16×** (real gap larger — dummy routing hides
the MoE cost; `llama-cli -f` real-prompt CPU-stalled ~20 min on crown's DeltaNet). CORRECTNESS: needle-in-haystack
recall ("ZEBRA-9-MAGENTA" across ~5K tokens) through the oneDNN prefill = CORRECT (a dim-swapped GEMM would corrupt
the hidden state). Crown
**6.4527 bit-exact** (oneDNN is prefill T≥minT only; decode T==1 untouched), **ctest 30/30**.
⚠ The kprofile single-`forward(8192)` looked like "parity" at 1006 tok/s but that was a **dummy-token-routing
artifact** (identical tokens hit ~8 experts = 3% of the weights) — ALWAYS measure MoE on a real prompt.

**CODE (all env-gated, default byte-identical):** `qwen36.cpp`/`qwen36.hpp` — internal recurrence chunk
(`IE_QWEN36_DN_RECUR_CHUNK`, def 512), MoE scratch cap (`IE_QWEN36_MOE_PREFILL_MAXT`, def 4096), fused dispatch
ceiling (`IE_QWEN36_MOE_FUSED_CEIL`, def 2048), oneDNN MoE branch (`IE_QWEN36_MOE_ONEDNN` + `..._MINT` def 4096) +
`ws_moe_up_packed_`/`ws_moe_btf16_`. Repro (SoA, decode-safe): `IE_QWEN35_PREFILL_CHUNK=8192
IE_QWEN36_MOE_PREFILL_MAXT=8192 IE_QWEN36_MOE_FUSED_CEIL=16384 IE_QWEN36_MOE_ONEDNN=1`.

**✅ SoA PRODUCTION PATH DONE (decode-safe) — commit pending:** new additive kernel `dequant_q4_moe_soa_to_Bt`
(crown packed-scale SoA Q4_K → fp16 Bt) + the oneDNN branch is now SoA-aware (gate/up→`dequant_q4_moe_soa_to_Bt`,
down→`dequant_q6_K_soa_to_Bt`, AoS fall-through). oneDNN prefill now runs on the **DEFAULT SoA experts** (no
`IE_NO_MOE_SOA`) → **decode stays SoA, zero regression.** MEASURED (SoA, real 12K prompt): **prefill 727→1079
= 1.48× WIN, decode 56.55 (AoS penalty 52.8 GONE)**, needle-recall PASS, crown 6.4527 bit-exact, ctest 30/30.

**✅ SINGLE-FLAG CONSOLIDATION DONE:** `IE_QWEN36_MOE_ONEDNN=1` ALONE now drives the whole lever — it raises the
MoE scratch cap (→8192) and fused ceiling (→16384) at load + allocates the oneDNN buffers, and the engine
(`engine.cpp`) feeds crown a `pf_chunk=8192` so oneDNN auto-engages for long prompts (granular envs still
override). Validated end-to-end (Exp6, single flag): SHORT prompt "Paris" coherent (oneDNN off <minT, big-chunk
feed harmless); LONG needle prefill 861 tok/s + recalls "ZEBRA-9-MAGENTA". No medium-prompt regression — the
int-dot MoE is ~linear in total rows so big-chunk == 512-chunk for T<minT (the old per-token cliff is gone via
the ceiling). crown 6.4527 bit-exact, ctest 30/30.

**LONG-CTX PPL QUALITY GATE DONE (Exp7, chunk 8192, identical 1023-tok scored set):** int-dot M=256 NLL
**1.3083** (PPL 3.70) vs oneDNN M=256 NLL **1.3102** (PPL 3.71) = **Δ +0.15%** — essentially equal (within
FMA-ordering noise; NOT a quality regression). The engine-wide **6.4527 decode gate is bit-exact** (oneDNN is
prefill-T≥minT only; decode T==1 untouched). VRAM fits (21.5/28.7 GB w/ 8192 scratch).

**✅ DEFAULT-ON SHIPPED (lead-approved 2026-06-26).** The lever is now ON by default for long-ctx crown
configs (`max_ctx ≥ 8192`); opt-out `IE_QWEN36_NO_MOE_ONEDNN`. Short-ctx configs (`max_ctx < 8192`) allocate
nothing extra and behave EXACTLY as before. Buys 1.48× long-ctx prefill at +0.15% prefill-PPL (decode/6.4527
gate bit-exact). **KEY BUG FIXED in the flip:** `forward()` self-calls `ensure_workspace(T)`, which decides
`moe_onednn_on` from the per-forward CHUNK size — so a prompt in [4096, 8192) (fed as one sub-8192 chunk)
would miss the lever AND hit the per-token cliff. Fix: the engine sizes the crown workspace to
`min(8192, max_ctx)` at LOAD when default-on (engine.cpp), so the oneDNN buffers + 8192 scratch are allocated
up-front and ANY prompt ≥4096 engages oneDNN. VALIDATED (Exp8, `ie run --ctx 8192`, NO env): default-on
auto-engages **1167 tok/s** (vs opt-out int-dot 890 = 1.31× on a 4986-tok prompt; 1.48× at 12K), needle-recall
correct both paths; 6.4527 + ctest green. **CROWN-ONLY** (oneDNN ctx is single-card; 80B is 2-card → still
blocked; 27B is dense — no MoE, already wins prefill via the tile).

### 🏆 NEW WIN 2026-06-26 PM — 27B long-ctx prefill collapse FIXED (16K: 0.52× LOSS → 2.13× WIN)
The 27B's head_dim-256 full-attn layers were on **naive O(T²) `full_attention`** for real prefill chunks
(re-reads the whole KV cache T× → 16K prefill took 137 s = 119 tok/s). Routed them through the **proven
Gemma wide-tile kernel** `full_attention_fa2_prefill_tile_gemma` (KV once per Br query-tile, appends k/v
internally, `window=0` full causal, head_dim 256). Numerically equivalent to naive (same 1/sqrt(HD) post-dot
scale on the same unscaled Q; argmax-bit-identical on Gemma; coherence-verified — identical capitals vs
naive). **GATED to ctx≥6144** (`IE_QWEN35_FA2_TILE_MINCTX`, opt-out `IE_QWEN35_NO_FA2_TILE`) so the verified
≤4K naive WIN is untouched. **Measured 1×B70 (same GGUF, vs current llama-SYCL `fdc3db9b6`):**
pp8192 272→**722** (2.65× self) = **WIN 3.08×** vs llama 234.8; pp16384 115→**499** (4.35× self) = **WIN
2.13×** vs llama 233.9. **27B now WINS prefill at EVERY length (1.96–3.08×).** Code: `src/model/qwen35_dense.cpp`
(the prefill `else` branch). Detail: `docs/authority/qwen35-27b.md` §3/§8.

**PORTED TO CROWN (`qwen36.cpp:1074`, gated `IE_QWEN36_NO_FA2_TILE` opt-out, ctx≥6144) — validated but
honest:** crown 6.4527 bit-exact (decode-path), coherent, ctest 30/30; **self pp8192 +22% (285→348),
pp16384 +57% (201→316).** BUT crown long-ctx prefill **REMAINS A LOSS (~0.35× vs llama 902–996)** —
unlike the dense 27B (attention-bound → tile flipped to WIN), the crown collapse is **MoE-GEMM-bound**:
llama's coopmat MoE prefill dominates; our tile fixes only the attention portion. **The real crown
long-ctx lever = MoE-prefill GEMM (oneDNN/coopmat for the crown MoE)** — same class that won Coder/Gemma,
not yet ported to crown's MoE.

**PORTED TO 80B TOO (`qwen3next.cpp`, gated `IE_QWEN3NEXT_NO_FA2_TILE`, multi-card-safe SIMD):** correctness
PERFECT (bit-identical generated tokens vs naive — " Paris. The capital of Germany is Berlin…"), **self
+12.5% @8.4K long-ctx prefill (443→498).** SMALLEST of the three ports because the 80B is the most
MoE-GEMM-dominated → confirms the 80B prefill loss is MoE-bound, not attention. **The tile lever now covers
ALL THREE DeltaNet hybrids: 27B (WIN, attention-bound) / crown (+57%) / 80B (+12.5%).** (Also made
`qwen3next_bench.cpp` MAX_CTX env-configurable via `IE_QWEN3NEXT_BENCH_MAXCTX` for long-ctx benching.)
Detail: `docs/authority/{crown,qwen3next-80b}.md` §3.

**⚠ NEXT-FRONTIER CORRECTION (the "oneDNN MoE-prefill GEMM" lever is FALSIFIED by the math):** crown/80B
MoE = 256 experts / top-8 at the **T≤512 DeltaNet chunk → only 16 rows/expert**. At M=16 the expert
weights cannot be amortized by ANY GEMM kernel (int-dot OR oneDNN/coopmat) — it's weight-BW-bound, the
Gemma small-M trap. llama wins because it batches the MoE over the FULL prefill (M=256 @8K). **So a
straight oneDNN MoE port is a DEAD END at the current chunk — do NOT build it.** The real lever = make
the MoE GEMM **large-M**, via EITHER (i) lift the DeltaNet chunk cap, OR (ii) decouple MoE batching from
the recurrence chunk.
> ⚠ **SUPERSEDED 2026-06-26 eve (see top-§7):** the "do NOT build it" was correct only for the per-CHUNK
> M=16. At a big ENGINE prefill chunk (8192 → M=256), the oneDNN MoE **WINS 1.46×** — the missing piece was
> the unreachable large-M regime (T<2048 fused-dispatch cliff), not the kernel choice.

**(i) CHUNK-CAP LIFT — TESTED + FALSIFIED 2026-06-26 (user-authorized "push through"):** SAFE on this box
(`--prefill-chunk 1024 & 2048` each ×3 bit-identical, no PPL collapse, no crash/dmesg-reset — §1 bug not
firing) BUT **PERF FALSIFIES IT**: crown long-ctx prefill A/B (Engine, ~10.5K prompt, only chunk varies)
= 512→**798.7** / 1024→**852.2 (+6.7%)** / 2048→**316.6 (0.40× REGRESSION)**. The "4× fewer MoE weight
reads → big win" hypothesis is WRONG (best +6.7%; 2048 hits an O(T²)-attn/4K-scratch cliff). Research
confirms the "no-longer-reproducible" is **timing-luck not a fix** + a SEPARATE **DEVICE_LOST hazard**
(Intel Triton XPU #6658: gated-delta recurrence → device-lost on this exact BMG-G31 silicon; CR 26.14
regressions, compute-runtime #922). **VERDICT: dead end — keep default 512, do NOT lift.** Full record:
`crown.md` §8.

**(ii) MoE-batch-decouple = the ONLY remaining real lever** (layer-major prefill, recurrence stays chunked
at 512 but MoE batches full-T → large-M). L-effort qwen36.cpp restructure, uncertain ROI for a narrow
>6K-prompt case. NOT started.
> ✅ **RESOLVED 2026-06-26 eve (see top-§7):** premise was off — the MoE was ALREADY full-T per forward-call,
> so no "layer-major restructure" was needed. The fix = a BIG ENGINE prefill chunk (M=256) + a per-expert
> **oneDNN MoE** + an internal recurrence sub-chunk (≤512 launches, §1-safe). **1.46× WIN, opt-in.**

### ✅ Gemma MTP spec-decode RE-VERIFIED 2026-06-26 PM (31B lossless SECURED; 26B losslessness regression CONFIRMED)
Clean-box own-baseline re-verify (K=4, ntok=128): **31B dense LOSSLESS ✓, net 1.48×** (plain 17.95 →
spec 26.55, accept 2.745) — the gemma decode headline, secured. **26B-A4B NOT lossless** (1.46×, accept
2.953) — confirmed regression; `IE_GEMMA4_NO_BATCHED_VERIFY=1` does NOT restore it and is slower (0.77×),
so it's a genuine near-tie MoE routing flip (verify ↔ T==1), not a flag artifact. **Do NOT claim 26B MTP
as lossless** (one open gemma-MTP correctness item — see `docs/authority/gemma4.md` regression note).
Docs updated: gemma4.md (§one-liner/§decode-table/§regression-note), COMPETITIVE_SCORECARD (resolved
the pending gemma row), this §7.

### ✅ PUBLICATION-INTEGRITY WIN 2026-06-26 PM — 27B prefill ~2× RE-VERIFIED (the HIGH-RISK claim is now DEFENSIBLE)
The scorecard's one remaining HIGH-RISK headline ("27B dense prefill 1.21× — public llama ≈718 may erase
it") is **RESOLVED → REFUTED.** Fresh apples-to-apples H2H, SAME GGUF (16.74 GiB), single B70 pinned
(`level_zero:0`), current llama-SYCL build `fdc3db9b6`/9598, both on-GPU: **ours pp1024 572 vs llama 292.3
= WIN 1.96×; pp4096 577 vs 278.9 = WIN 2.07×.** llama decode tg32 = 21.62 (sane on-GPU, not CPU-fallback)
→ our decode ~16 still LOSES 0.74× (the known commodity-axis loss — never claim it). The old H2H llama 287
was accurate (not a slow config); the ≈718 number never materialized on this box. **The 27B prefill ~2×
win is now a defensible public claim.** Docs updated: `docs/authority/qwen35-27b.md` (§0/§1/§3/§8 caveat→
verified), `docs/COMPETITIVE_SCORECARD_2026-06-25.md` (§1 TL;DR + §4 newly-verified + checklist #1).

### ✅ CLOSED 2026-06-26 PM — Coder (qwen3moe) decode launch-fusion line is settled (NOT a sprint priority)
Re-profiled Coder decode clean (1×B70: ~52 tok/s, GPU 15.34ms / wall 19.3ms) + 6-agent design workflow +
skeptic + GGUF inspection. **The decode gap (49.5 vs llama 71.5 = 0.69×) is NOT closable by a single bounded
lever** — settled by two experiments: (a) on-device routing perf-neutral (`4eff28a`, host bubbles already
overlapped); (b) `res_rms` fusion (attn-out residual_add + ffn_norm → one pass) = **REAL +2-3% decode but
+0.33% PPL** (fused 2-pass fp32 ≠ split numerics; `round_residual`-through-half did NOT recover bit-exactness)
→ **shipped OPT-IN `IE_QWEN3MOE_RES_RMS_FUSED`, default = lossless split.** Crown 6.45 / dense bit-exact (shared
`residual_add_rms_norm_fused` untouched), ctest 30/30. KEY FACTS: small attn GEMVs are **occupancy-floored**
(gemv_fp16 ≈ gemv_q4k per-call despite 3.5× bytes); MoE experts already grouped like llama at ~42-46% BW (vs
llama ~60%); Q/K/V are 3 heterogeneous dtypes (Q4_K/Q8_0/Q6_K) so the real QKV-fusion lever (D14, ~+4-7%, would
beat llama on the attn axis) needs a mixed-dtype kernel (L-effort); MTP amortization blocked (no NextN head in
the Coder GGUF). Even fully removing the ~4ms launch overhead floors at 65 < 71.5 → **kernel-efficiency grind,
the commodity axis.** Full ledger: `docs/authority/coder.md` §8 (#1 CLOSED) + D13/D14. **PIVOTED the sprint to
securing the publishable claims (27B prefill verification — see below).**

### ⚖ FALSIFIED + FOUNDATION 2026-06-26 — Coder (qwen3moe) decode: on-device routing (`4eff28a`)
RESULT: built + measured (1×B70, decode=256, 3-run). **PPL 11.98→11.97 = LOSSLESS. Decode ~50.9→~51.1 =
PERF-NEUTRAL** — the projected +20-40% did NOT materialize. **The per-layer host round-trip was NOT the decode
bottleneck (hypothesis FALSIFIED).** Verified no other per-layer host sync remains (attn block all-GPU) → the
decode is bound by the **KERNEL LAUNCH COUNT itself (~720 launches/token)**, not host stalls. The real lever is
**kernel FUSION** (llama fuses MoE experts to 3 grouped launches/projection); on-device routing is the lossless
PREREQUISITE for it. Kept env-gated `IE_QWEN3MOE_ONDEVICE_ROUTE` default-OFF (default byte-identical); the new
`moe_topk_from_logits` is a reusable generalized device router. **NEXT = launch fusion** (the genuine win). Detail below ↓

### (history) ⏳ IN-FLIGHT 2026-06-26 — Coder (qwen3moe) decode: on-device routing (the launch-bound lever, NOT reorder)
Coder decode loses 49.5 vs llama 71.5. Workflow roofline + llama-source dive SETTLED the mechanism: **NOT
bandwidth-bound** (we run at 16.2% of the 608 GB/s ceiling, llama 23.5% — ~5× headroom) → **launch/host-stall
bound.** `--kprofile-decode`: ~3.6ms/19% non-kernel bubbles/token. Root cause: per decode MoE layer we memcpy
router logits D→H + host `route_from_logits` + memcpy packing H→D + a **per-layer `q.wait()` drain**
(qwen3moe.cpp:444/466/874) = ~144 GPU bubbles/token. llama's whole ~45% edge = dispatch amortization + on-device
router ids + NO host wait (its MoE experts are STANDARD block_q4_K, explicitly NOT reordered). **FIX (drafted+built
rc=0, env `IE_QWEN3MOE_ONDEVICE_ROUTE` default-off):** 2 new device kernels in src/ops/moe.cpp —
`moe_topk_from_logits` (generalize moe_router's E=256-locked stage-2 to E=128, reads the already-on-device
ws_router_logits_, softmax+top-k+renorm+sort matching route_from_logits) + `moe_build_pack_decode` (T=1 packing) —
then delete the per-layer q.wait() (safe: in-order queue). **VALIDATING NOW (serial GPU): Coder PPL lossless (~11.98)
+ decode A/B (target 71.5).** Projected +20-40%. ⛔ REORDER on MoE experts REJECTED (~0%, not BW-bound). [[project_qwen3moe]]

### ✅ SHIPPED 2026-06-25 — Crown MTP spec-decode LOSSLESS PROVEN + 27B Q4_K reorder lever (+4-7%)
Two milestones this session (commits `14596da`→`d15bbcd`; all additive, ctest 30/30, lmstudio crown bit-exact):

**(A) 27B decode — llama reordered-Q4_K layout port (`60aaa49`/`15c7dda`/`15e67a3`).** De-interleaves AoS
`block_q4_K` into 3 pure-contiguous global regions (nibbles|scales|dm) for `gemv_q4_K_reorder_q8` (kills the 144B
header gaps that break AoS coalescing). **CONFIRMED +4-7% on 27B FFN decode** (16.28→16.96, rock-solid). Env-gated
**opt-in `IE_QWEN35_Q4K_REORDER`** — NOT default because it loads the reorder buffer ALONGSIDE the AoS (prefill still
needs AoS), so it doubles the Q4_K FFN VRAM. Measured A/B (decode=256): reorder ~17 BEST; `IE_QWEN35_Q4_SOA` on the
projections **REGRESSES** (~15) — small projection GEMVs don't benefit from a fast-path layout. **So the residual 27B
decode gap to llama (23.2) is NOT more layout.** ROOFLINE (`--kprofile-decode`, reorder on, 57ms/tok): decode is
~87% GEMV-BW-bound; **DeltaNet recurrence is only 1.4%** (recurrence-bottleneck hypothesis FALSIFIED). The two big
GEMVs: Q4_K FFN gate/up `gemv_q4k_reorder` 40.8% (reordered ✓) + **Q6_K FFN-down `gemv_q6_soa` 31.5% (already the
strong ~80%-BW SoA int-dot kernel — NOT a reorder candidate)**; small projections `gemv_q4k` AoS 9.3% (reorder
regresses them — measured). Both big kernels sit ~42-46% effective BW in the real loop vs llama ~60%. **CONCLUSION:
no single remaining layout lever closes 17→23 — it's broad kernel micro-efficiency (occupancy / WG-geometry /
dp4a-loop) on the COMMODITIZED dense-decode axis. (⚠ DeltaNet-decode is NOT a moat to lean on either —
`docs/COMPETITIVE_SCORECARD_2026-06-25.md` RE-VERIFIED crown decode = PARITY 78.1 vs 76.9 and 80B = LOSE 0.93×;
the DeltaNet win that HOLDS is crown PREFILL 1.14×.) Lowest-ROI grind; the reorder already banked the cheap win.** (NB: the kprofile that produced this CRASHED the
box once via a double-load — [[feedback_gpu_exclusivity]] repeat-incident; serialize, gate on DONE-marker + `pgrep -x`.)

**(B) Crown (Qwen3.6-35B-A3B) native MTP spec-decode — ✅ LOSSLESS PROVEN end-to-end (`f534cff`/`d15bbcd`).**
Catches llama's merged `--spec-type mtp`. `QwenModel::{load_mtp_head,mtp_head_forward,spec_generate}` + forward hooks
(all_logits/hidden_pre_norm/per-position-DeltaNet-ckpt, `14596da`) + `ie-qwen36-spec` tool. Head (blk.40) = full-attn
+ MoE. **RESULT: token-for-token == plain greedy, mean accept 1.88/round (head CORRECT).** PERF 0.36× — overhead-
limited (NOT a bug): a 248K-vocab lm_head paid K× in the draft + verify(T=K) can't be amortized when plain decode is
already fast. ⚠ CORRECTION (`docs/COMPETITIVE_SCORECARD_2026-06-25.md`): crown decode is **PARITY** vs llama (78.1 vs
76.9), NOT a win — so MTP *would* be the natural lever to win DeltaNet decode, but at 0.36× net here (and re-verified
0.81× even on the 27B) it does NOT help until the verify path is cheaper. **Ships as a lossless CAPABILITY, not a
speedup.** Validation recipe: `ie-qwen36-spec --gguf <lmstudio crown> --head-gguf <unsloth UD>` — main from lmstudio
crown (loads clean), head from unsloth UD with blk.40 experts dequanted to fp16 (sidesteps unsloth's Q5_K MoE-down,
which has no decode kernel; the unsloth UD quant is otherwise incompatible: Q8_0 attn / F32 ssm / BF16 gate_inp).
Perf-win path if ever needed (deferred): head lm_head reuse main int-dot Q6_K output_ + batched int-dot verify. Full
detail: [[project_crown_mtp_port]].

### ✅✅ SHIPPED 2026-06-23 PM — PREFILL WIN: ported llama SIMD "tile" FA kernel, DEFAULT-ON, 2.0–3.1× over v2
`full_attention_fa2_prefill_tile` (`src/ops/attention.cpp`) is now the **default Coder prefill attention** (opt-out
`IE_NO_FA2_PREFILL_TILE`→v2). **Measured 1×B70: 4K 2.0× / 8K 2.4× / 16K 3.08× over v2** (16K 105→324 tok/s) →
the llama gap shrinks from 5.9× to **1.9×** (llama 626 @16K). Coherent + on-topic, crown 6.4527 bit-exact.
- **The misframe that cost time:** llama-SYCL uses NO XMX/joint_matrix for attention (their `fattn.cpp:194` is the
  CUDA backend; the SYCL path is a SIMD "tile" kernel with a literal `// Todo: Use the XMX kernel if possible`).
  So all the XMX work (fused kernel, multi-subgroup, register-softmax scoping) targeted a unit llama doesn't use.
- **The win = a FAITHFUL port of llama's tile INNER LOOP** (`~/llama.cpp/ggml/src/ggml-sycl/fattn-tile.hpp`,
  `flash_attn_tile_iter_KQ`/`iter`): 16 subgroups × 1 q-row, every lane computes 4 COMPLETE register-resident Q·K
  dots (KV pos {L,L+16,L+32,L+48}), full head_dim streamed as 64 half2 in registers, ONE width-16 max/sum reduce
  per TILE (not per key), distributed softmax + per-lane VKQ (8 out-dims/lane), Q resident in SLM. All 256 lanes
  busy; SLM 38KB (better occupancy than v2). A FIRST high-level-structure port (big WG + half2 loads only) was
  0.6× v2 — proving the perf lives in the inner-loop thread/register mapping, NOT the WG geometry.
- **NEXT to close the remaining 1.9×:** (a) larger `ncols` (more q-cols/WG → less redundant causal KV HBM, llama
  uses up to 32; ours is 16), (b) head-dim sub-batching (`nbatch_K`), (c) wire `_tile` into the dense full-attn
  arches (Llama/Gemma/Granite/Phi all process full-T prefill on the SIMD path). The XMX fused kernel stays gated as
  a future lever ONLY if a register-softmax version ever beats this tile kernel (unproven; deprioritized).

### ▶▶▶ (history) RESOLVED/REDIRECTED 2026-06-23 PM — gemm_fp16-orchestration DELETED; real prefill lever = a FUSED XMX flash-attn-2
**UPDATE (`25f1f84` → adversarial-review cleanup): the orchestration was REJECTED then DELETED — do NOT re-create it.**
- **`full_attention_fa2_prefill_gemm` DELETED** (`db5715a` reverted): it was 0.68× v2 (rejected) AND a Codex adversarial
  review flagged it as the sole caller hitting `gemm_fp16`'s ragged-N store corruption (`joint_matrix_store` writes a full
  TM×TN tile but only bounds-checks the tile origin → a partial tail tile spills into the next C row). Rather than bolt
  occupancy-regressing SLM-staging onto the hot `gemm_fp16` for a dead path, the dead path was removed (function + decl +
  gate). **`gemm_fp16`'s store-alignment contract is now documented** (loads tolerate ragged M/N/K — the ragged-K guard
  stays; the STORE requires C allocated to round_up(M,TM)×round_up(N,TN); all in-tree callers pass tile-aligned weight dims).
- **Why it was rejected (kept for the record):** even once correct it was 0.68× v2 because the orchestration MUST query-block
  to bound the materialized S=[Br×ctx] matrix → ~5000 small gemm+glue launches/layer (launch-bound) + no causal-skip. The
  "validated 13-27×" was SHAPE-MISLEADING (`bench_gemm_fp16` measured ONE big M=16384 gemm). Materializing S + launch-explosion
  is architecturally inferior to a FUSED kernel — which is why real flash-attention fuses QKᵀ→softmax→PV and never writes S to HBM.
- **A correct FUSED XMX FA-2 kernel ALREADY EXISTS** (`full_attention_fa2_prefill_xmx`, gated `IE_FA2_PREFILL_XMX`): QKᵀ →
  online-softmax → P·V in ONE kernel, joint_matrix on `gemm_fp16`'s sanctioned path, online-softmax staged through SLM, never
  materializes S to HBM. Output is byte-identical to v2 at all NSG. **Two bottlenecks found & one fixed (2026-06-23):**
  - **(1) occupancy — FIXED via multi-subgroup.** The original kernel was 1 subgroup/WG (16 work-items → starved the EU) = **0.27× v2**.
    Restructured to **NSG subgroups/WG, each owning Br=16 q-rows, ALL sharing one cooperatively-loaded K/V SLM tile** (`5d2ef40`+):
    NSG=4 → **0.83–0.91× v2**; NSG=5 → **0.99× v2 @8K (parity)**. A 2.2–3.3× jump — occupancy was the dominant cost.
    **NSG is now chosen at runtime from the SLM budget** (adversarial-review fix): hd=128 → NSG=4 (105KB, byte-identical to the
    tuned path); hd=256 → NSG=1 (96KB) instead of failing kernel submission at 184KB. The kernel is no longer hd=128-only.
  - **(2) the residual wall — STRUCTURAL, not fixed.** Even at parity@8K, XMX is **0.86× v2 @16K** — the gap RE-WIDENS at long ctx
    because the **SLM-staged softmax round-trip** (store S → barrier → scalar mask/exp/rescale → store P → barrier → P·V, + scalar
    O-rescale) costs scale with n_tiles (256 tiles @16K). Multi-subgroup parallelizes this but cannot eliminate it; v2 (SIMD) wins
    long-ctx by keeping the whole online-softmax **register-resident in the subgroup** (no SLM round-trip, no per-tile barriers).
- **→ THE genuine fix (a real multi-session kernel, NOT a knob):** make the online-softmax operate on the joint_matrix
  **accumulator fragments in-register** (`get_wi_data` / `joint_matrix_apply` for the elementwise exp/scale + subgroup-shuffle
  reductions for the per-row max/sum across the N-distributed lanes), eliminating the S/P SLM round-trips and the per-tile
  barriers. This is exactly the hard part of FA2-on-tensor-cores that llama's `fattn-mma` solves. Scope it fresh against the
  fast `ie-attn-profile`/`bench_gemm_fp16` harness; the existing XMX kernel is the correct starting scaffold. Both cheaper XMX
  paths (gemm-orchestration; naive-fused + bigger Br) are now PROVEN to lose to v2 — do not re-try them. v2 (106 @16K) is the
  current best and is at the vector-ALU ceiling; only a register-softmax XMX kernel closes the 5.9× gap to llama (626).

### ▶▶▶ #1 GAP 2026-06-23 — decode 3.85× behind llama: REAL, but exact mechanism NOT cleanly isolated yet
**⚠ CORRECTION (intellectual honesty):** an earlier commit (`ff608b4`) claimed "ROOT CAUSE: dispatch-bound,
GPU 46% idle." That number is **NOT supported** — both VTune runs were contaminated: the first by PREFILL
(prefill 8192 ≈ 40s window vs 0.2s decode → the 46% idle was prefill's), the second by MODEL LOAD (Elapsed
31.5s vs GPU 10s, but the 21.5s includes loading 18GB + warmup, not decode gaps). Do NOT trust run-level
idle from a load/prefill-containing VTune run.
**What IS clean (per-kernel VTune, load-independent):** the small decode kernels are LOW-OCCUPANCY — attn
projection `gemv_fp16` ~37% occ / 45% within-kernel-idle (16 WGs under-fill B70's ~160 EUs); the big MoE
kernels are fine (`down_q4k` 85% occ). So decode time is dominated by many small under-filling kernels.
**RESOLVED via clean isolation (`ie-attn-profile` — loops fa2_decode ALONE: no model load, no MoE, no other
kernels, contamination-free):** `fa2_partial` is **intrinsically ~40 GB/s (6.7% BW)** — fp16 partial 0.414ms@8K
/ 0.829ms@16K per layer. It's slow EVEN with zero inter-kernel host gaps → the wall is **WITHIN-KERNEL
(memory-latency: too few in-flight KV requests to hide latency)**, NOT dispatch/host-gaps. **→ COMMAND-GRAPH
IS RULED OUT** (isolation has no gaps yet the kernel is still 40 GB/s). int8 partial = 0.198ms@8K = **2.1×**
faster (half the KV bytes — the shipped int8-KV lever, confirmed; tapers to 1.28× @16K). **NEXT LEVER:** raise
the kernel's intrinsic BW above 40 GB/s — iterate variants IN `ie-attn-profile` (fast + clean, NOT in-engine
A/Bs which are diluted/heat-noisy). Open question = WHY 40 GB/s: prime suspects are the split-K SLM
cooperative-load + barrier overhead, and insufficient memory-level parallelism (more in-flight KV loads
per thread / software prefetch / wider vectorized loads). The 8 in-engine tweaks were flat partly because
in-engine signal is diluted — re-run them in `ie-attn-profile` where a kernel BW change is directly visible.
DO NOT build a command-graph (ruled out). int8-KV already banks the biggest single kernel win (2.1×).
**[history — overstated, corrected above]** ROOT CAUSE FOUND (VTune): during decode the GPU is ~46% IDLE. Cause: T=1 decode kernels are TINY (e.g. attn-proj gemv_fp16 = 16
work-groups, 45% idle, 37% occ) so they can't fill B70's ~160 EUs, AND there are ~900 launches/token on an
in-order Level-Zero queue with host gaps between them. The big MoE kernels are fine (down_q4k: 85% occ, 13%
idle) — the idle is concentrated in the many small kernels + inter-launch gaps. **THIS is why kernel work is
futile: 8 attn-kernel approaches + the faithful fattn-vec port were all flat/parity because the GPU is idle
WAITING, not compute-bound — making a kernel faster does nothing when the EUs sit idle between launches.**
**THE LEVER (engine-level): feed the GPU.** (1) SYCL command-graph — record the decode forward once, replay
per token, collapsing ~900 launches → 1 graph submit (kills the inter-launch host gaps; verify BMG L0
`ext_oneapi_graph` support); and/or (2) kernel FUSION (fewer, larger decode kernels). llama's flat-across-ctx
decode = it keeps the GPU fed this way. Expected: closing the 46% idle ≈ up to ~2× decode. PROFILE METHOD:
`vtune -collect gpu-hotspots` (paranoid sysctls already 0); fa2_partial-only kernels are decode-specific even
in a prefill-containing run. NOT XMX, NOT more attn-kernel tweaks (proven). The BenchScope `host_scheduler`
lane now surfaces this (launches + GPU-idle%) via `tools/perf/benchscope_export.py --gpu-idle`.
**[earlier framing kept below]**
### ▶▶▶ (history) #1 GAP — decode 3.85× behind llama: NOT the attn kernel (proven) → it's the ENGINE
**EXHAUSTIVELY PROVEN the gap is NOT in the attention kernel.** A faithful, P@V-ILP-optimized ground-up port
of llama's `flash_attn_ext_vec` (`full_attention_fa2_decode_vec`, gated `IE_FA2_VEC`, coherent+correct) =
**PARITY with our simple v1** (8K +2.5% / 16K −1%), NOT the 3.85×. Combined with: combine-pass not the
overhead (3.9%), occupancy flat, and 7 prior inner-loop tweaks flat — OUR kernel structure and LLAMA's
structure perform IDENTICALLY on our engine, both stuck at ~9-12% BW reading the same 805MB KV @8K (llama
46%). **So the 3.85× lives in the SURROUNDING ENGINE, not the kernel:** the prime suspects are (1) launch/
dispatch overhead (~900 launches/token on an in-order Level-Zero queue → inter-launch GPU stalls) and/or
(2) the memory subsystem (why does the IDENTICAL 805MB KV read hit 9% BW for us vs 46% for llama — KV cache
layout? L2 behavior? read transaction width at the driver level?). **NEXT INVESTIGATION (not kernel work):**
(a) command-graph / launch-batching experiment (record decode once → replay; cuts ~900→1; BMG Level-Zero
graph support must be verified) — note the prior "llama wins with its graph OFF" only means llama isn't
launch-bound (it has far fewer launches natively), NOT that WE aren't; (b) a VTune/GPU-profiler memory study
of our fa2_partial vs llama's fattn-vec actual HBM/L2 transactions to explain the 9%-vs-46% BW. The
`IE_FA2_VEC` port is a correct scaffold (PPL-gate before any default-on). DO NOT keep tweaking the attn
kernel — proven futile (8 approaches). XMX N/A (T=1 vec).
**[superseded framing below kept for history]**
### ▶▶▶ (superseded) #1 LEVER 2026-06-23 — decode-attn is 3.85× behind llama @16K
**MEASURED same box/file (Coder-30B-A3B Q4_K_M, 1×B70): llama tg64 decode = 84.8 @16K (FLAT from 84 @512);
ours collapses 43 @512 → ~22 @16K = 3.85× gap.** Effective BW: llama ~46% (BW-bound, scales), ours ~12%
(latency-bound, collapses). `fa2_partial` is 62% of our decode @16K. This is the engine's single biggest gap
and our strategic-weak axis (long-ctx decode). NOTE: XMX does NOT apply (decode is T=1 vector-matrix; llama
uses a VEC kernel, not joint_matrix) — so this is a SIMD-vec restructure, not a matrix-engine project.
**5 micro-opts MEASURED FLAT (all the wrong framing — v1-vs-v2 tweaks, not closing the llama gap):**
TARGET_SUPER occupancy, reduce-blocking, softmax-defer, no-SLM. The real difference is STRUCTURAL — likely
llama's warp-split (`nthreads_KQ`/`nthreads_V` in `fattn-vec.hpp`: multiple KV positions scored in parallel
per warp + narrower reduces + per-thread VKQ accumulation), where ours runs gqa=8 subgroups each re-walking
the SAME KV with a full-16-lane reduce PER POSITION (8× redundant reduces, serial positions). **CONCLUSIVE (2026-06-23): the gap is STRUCTURAL — needs a GROUND-UP `fattn-vec` port, not tweaks.**
SEVEN kernel restructurings ALL measured FLAT in clean throughput @8K+16K: TARGET_SUPER occupancy,
reduce-blocking, softmax-defer, no-SLM (direct HBM), manual butterfly-shuffle (vs collective reduce),
tile-size (BLK 16/32), and half8-vectorized KV loads. (Beware: kprofile sometimes shows per-kernel "wins"
that DON'T translate to wall-clock — profiling mode disables the compiler's auto-vectorization, so scalar
v1 looks slow only under profiling. Always confirm with a CLEAN throughput A/B, not kprofile.) Decode is
GPU-bound (clean wall ≈ profiled GPU-sum), so it's the kernel — and our split-K + SLM-staging + combine-pass
+ gqa-redundancy structure carries ~4× more GPU overhead than llama's lean `flash_attn_ext_vec`
(`fattn-vec.hpp` L312-422: per-thread KQ_reg fill → one rescale/tile → per-lane P@V from SLM). The fix is to
PORT THAT STRUCTURE from scratch (a focused cool-box session, NOT incremental edits to our kernel — proven
not to work). Ref: `.claude/agent-memory/inference-engine-expert/project_fa_decode_kernel_sycl.md`. The flat
v2 scaffold (`IE_FA2_DECODE_V2`, gated-off) is OUR structure and should be DISCARDED for the port — start
from llama's fattn-vec. XMX does NOT apply (T=1 decode is vector-matrix; llama uses a vec kernel).

### ▶ SUPERSEDED 2026-06-22 EVE — `fa2_partial` "needs XMX" framing (CORRECTED above: it's a vec rewrite)
**The long-ctx DECODE bottleneck, diagnosed by a 5-lens fan-out + kprofile.** `fa2_partial_fp16` (the
split-K decode attention) is **62% of Coder decode @16K** and runs at only **~11% of peak BW (67 GB/s)**.
**Occupancy is RULED OUT (`114b254`):** TARGET_SUPER sweep 256→1024 WGs = **+1.8% (noise)** → the kernel is
at its design ceiling, NOT occupancy-bound. The 11% BW is **STRUCTURAL** — decode attention over 16K
positions is a reduction-bound sequential KV scan (split-K already parallelizes; the per-position dot +
online-softmax is irreducible). **The real win = a flash-decoding kernel that uses the XMX matrix engine
for QKᵀ/PV** (we have safe `joint_matrix` GEMM kernels already — `gemm_q4k_xmx.cpp`/`gemm_fp16.cpp` — to
template from; NOT ESIMD/block2d). This is a real kernel project to scope fresh, not a tweak. It is also
the lever that would let INT8-KV's win stop tapering (int8 only helped fa2_partial 8% @16K because it's
not BW-bound). Pairs with the prefill-attn fix already shipped (`430ab08`) — attention is THE long-ctx
bottleneck on BOTH axes, and it's our strategic moat (long-ctx + DeltaNet arches).

### ✅ DONE 2026-06-22 EVE — INT8-KV decode wired into Coder (1.29× @8K, `d70065e`)
5-lens decode fan-out (capture-then-fan-out: 1 serial GPU profile → 5 parallel CPU analysis agents).
**Killed 3 prior decode hypotheses with evidence:** (1) command-graph REFUTED — llama wins 1.6× with its
OWN graph default-OFF (`GGML_SYCL_DISABLE_GRAPH=1`); we're 97% GPU-busy. (2) W4A8/dtype REFUTED — llama's
Q4_K is also W4A8 dp4a (`vecdotq.hpp`); we already tested it (parity). (3) WG-geometry (MMV_Y=1) FALSIFIED
BY MEASUREMENT — predicted 1.3–1.5×, measured 0.95× (gated infra `IE_QWEN35_Q4_SOA_MMV`). **27B-dense
decode gap is REAL** (llama tg64 22.65 vs our 16.3 = 1.39×, verified same box/file — NOT stale) but it's
the **commoditized axis** where llama's mature kernels win; residual is weight lane-coalescing (bigger
rewrite, thrice-snakebit). **Pivoted to the strategically-aligned long-ctx lever:** wired the built-but-
unwired `full_attention_fa2_decode_int8` into qwen3moe (Coder, mirrors crown `qwen36.cpp:1037`); added
`--int8-kv` to the `ie` CLI. **Measured (clean, 3× repeat):** Coder decode **8K 1.29×** (31.8→41.1),
crown 8K 1.26×; **tapers to 1.06× @16K** (because fa2_partial is latency-bound — see NEXT LEVER above).
PPL crown 6.4527→**6.46** (+0.1%, holds ≤6.57). Coder coherent. Gated `kv.is_int8()` → default fp16
byte-identical. Opt-in `--int8-kv`. **`docs/` agent refs:** `.claude/agent-memory/.../reference_llama_sycl_q4k_decode.md`.

### ✅ DONE 2026-06-22 PM — FA-2 prefill kernel (qwen3moe long-ctx 2.05× @16K, `430ab08`)
The publish-blocker's worst case is FIXED for the arch that actually collapsed. **What shipped:**
`full_attention_fa2_prefill_v2` (`src/ops/attention.cpp`, declared `ops.hpp`) — query-row-block FA-2:
keeps naive's ALU-efficient split-head-dim + subgroup-reduce inner loop, but runs **1 query row per
SUBGROUP** with Br=16 rows/WG sharing one K/V SLM tile (Br parallel subgroups, 1 q_head/WG → high
occupancy) → cuts naive's redundant KV HBM ~Br×. Wired into qwen3moe (`qwen3moe.cpp` prefill branch),
**default-ON T≥2048**, opt-out `IE_NO_FA2_PREFILL_V2`, force-any-T `IE_FA2_PREFILL_V2` (A/B).
- **Measured (Coder Q4_K_M, 1×B70, clean box):** prefill v2/naive = 512 **0.95×** / 4096 **1.04×** /
  16384 **2.05×** (317s→146s). crown PPL 6.4527 bit-exact, Coder coherent.
- **THE KEY FINDING (corrects the roofline premise):** naive is **L2-bound, not HBM-bound, up to ~4K** —
  its "redundant" KV reads are L2-served (per-kv-head working set ~1MB@4K fits L2), so SLM tiling is
  PARITY there. Only past L2 (16K ≈ 4MB/kv-head) does the redundancy hit real HBM, where tiling halves
  it. Hence the T≥2048 crossover gate (not all-T).
- **Two variants measured+REJECTED first** (PERF_LEDGER): v1 thread-per-row (1 lane=1 row) **0.31×** —
  serialized all head_dim MACs onto one lane (16× per-lane work) + reg spill; the subgroup reduce is
  CHEAPER than serializing head_dim. Plus the old shelved `full_attention_fa2_prefill` (under-occupied +
  serial Br loop).
- **⚠ SCOPE — only qwen3moe benefits:** crown/27B/qwen3next are gated-DeltaNet, prefill-chunked at T≤512
  (the §1 BMG bug workaround), so their full-attn layers NEVER see T≥2048 → always in the parity regime.
  Their long-ctx prefill cap is the SEPARATE DeltaNet chunk (env-revertable, bug not repro on current
  NEO/kernel — see `docs/known_bugs.md`), NOT attention.

### ▶ OPTIONAL NEXT — extend v2 to dense full-attn arches + reassess decode
- **Dense full-attn arches process the whole prefill T too** (Llama/Gemma dense via DenseModel, gemma4),
  so they'd get the same long-ctx win — `full_attention_fa2_prefill_v2` is a drop-in for `full_attention`
  (any head_dim). Wire + A/B the prefill sweep + PPL-gate per family. (Gemma head_dim 256/512: confirm
  SLM fits — Bc=32 → 32/64 KB.) Lower urgency than the decode gap below.
- **Decode gap (the bigger lever, harder)** = launch+occupancy bound; only command-graph (~1850
  launches/token → 1, BMG L0 graph support unverified) or split-K GEMV can move it. Micro-levers
  measured neutral (round-1). See below + `docs/optimization_roadmap_2026-06-22.md`.

**Durable assets from the 2026-06-22 session (use these):**
- `tools/perf/` harness: `roofline.py` (kprofile+VTune → bottleneck classification — it correctly
  called decode latency-bound + would have stopped the SoA mistake), `viz.py` (HTML profile),
  `ledger.py`+`PERF_LEDGER.md` (PPL-gated accept/reject). **Use roofline-FIRST before any kernel work.**
- `docs/benchmark_results_2026-06-22.md` — the honest clean apples-to-apples scorecard (method:
  `ours = ie-bench --prefill 512 --decode 128` == `llama-bench -p 512 -n 128`; gemma's fastest llama
  backend on B70 is **SYCL not Vulkan**; discard cold/JIT first run for BOTH engines).
- `docs/optimization_roadmap_2026-06-22.md` — 10 ranked levers + round-1 results. Other big
  *already-built-but-OFF* levers worth a measured try: **INT8 KV** for the crown family (KV #1, ~2×
  long-ctx KV bytes, default-OFF, PPL-gate); **command-graph** for decode (~1850 launches/token → 1,
  bit-exact, but BMG Level-Zero graph support unverified + quant-hoist neutral is a yellow flag).
- Committed gated-OFF (default path byte-identical): Q4-SoA decode kernel + quantize-hoist (both
  neutral/rejected, kept as infra). VTune unlocked via `sudo sysctl dev.xe.observation_paranoid=0
  dev.i915.perf_stream_paranoid=0` (reverts on reboot).

### ▶ IN PROGRESS — EXL3 (QTIP-based SOTA quant) on Arc (2026-06-19)
Plan: `docs/superpowers/plans/2026-06-19-qtip-quant-engine.md`. Spec: `docs/exl3_format_notes.md`
(§5 oracle RESOLVED, §7 verified corrections). The bet: EXL3 = QTIP-class near-fp16 quant at 3–4
bpw, files downloadable free on HF, decode is pure ALU → fast on Arc is the open niche.
**DONE so far:**
- **Format cracked + spec locked** (source-read pass 2): cb=0 codebook (`codebook.cuh`, lop3
  0x6a=(a&b)^c hand-verified), tail-biting trellis extraction (`exl3_dq.cuh`), **tensor-core tile
  un-permute** (`quantize.py:21-48` — the easily-missed step), 128-pt Sylvester Hadamard + suh/svh.
- **Oracle = CUDA-FREE** (exllamav3 is CUDA-only; resolved): `tools/exl3/make_oracle.py` reproduces
  the decode in pure numpy. Artifacts from `turboderp/Llama-3.2-1B-Instruct-exl3` @3.0bpw (~1.1 GB,
  in `~/models/exl3-test/`); one-layer vectors `tests/data/exl3/onelayer.*` (k_proj K=2048 N=512
  bits=3 cb=0) — **git-ignored (Llama-derived), regenerate via the README**, meta.json tracked.
- **✅ Phase B Task 1 — host decode BIT-EXACT** (`d2bf292`): `include/ie/exl3.hpp` +
  `src/core/exl3_decode.cpp` + `exl3_decode_test` (ctest, SKIPs if vectors absent). maxerr **0.000**
  vs oracle across all 1,048,576 weights. **The format risk is retired.**
- **✅ Phase B Task 2 — GPU prototype: DECODE BIT-EXACT ON ARC** (`f657342`). Two additive kernels +
  `ie-exl3-test`; **zero edits to any model/crown path** (crown **6.4527/NLL 1.864495 bit-exact**,
  ctest **30/30**). The one open risk — the parallel-decode↔subgroup mapping — is **resolved**:
  `src/ops/gemv_exl3.cpp` = one WG per tile-column, 16 SGs × 16 lanes (SG=tile-column, lane=tile-row);
  each lane's tensor-core code index is loop-invariant in ki (computed once from the inverse tile
  permutation built in SLM) so every weight decodes exactly once and the 16 lanes reduce over the
  K-rows. `src/ops/hadamard.cpp` = 128-pt Sylvester WHT (1/√128, radix-2 butterfly). Gate (Intel
  [0xe223], heat-soaked): **[A] decode GEMV cosine 1.0000000** vs host `x@wrot.f16`; **[B] hadamard
  cosine 1.0 + round-trip 1.0**; **[C] FULL FORWARD cosine 0.9999999** vs `x@weight.f16` — validates
  the entire EXL3 forward (Hadamard incoherence + suh/svh + decode) AND confirms the Phase-C forward
  composition is `xh=had128(x⊙suh); acc=xh@W_rot; y=had128(acc)⊙svh` (derived from `make_oracle`'s
  `materialize()`, NOT the imprecise §4 ordering). **[S] speed ~2.7× gemv_q4_K @ T=1** — over the
  ~1.5× soft bar; bottleneck is **decode ALU, not memory** (SLM trellis staging gave no gain, and
  EXL3@3bpw reads LESS memory than Q4_K) → tuneable in a later perf pass (vectorized half2 decode,
  per-bits aligned extraction, more ILP/lane), and the real perf signal is end-to-end decode tok/s on
  a model (Task 6) where the smaller footprint + latency overlap help. **CORRECTNESS RISK RETIRED
  END-TO-END; speed is a known-tuneable gap, not a blocker → GO to Phase C.**
- **✅ Phase C — NATIVE EXL3 END-TO-END (2026-06-19):** a real EXL3 model runs on Arc, EXL3 weights
  decoded natively on-GPU (nothing dequantized). Llama-3.2-1B-3.0bpw → `ie run` →
  **"The capital of France is Paris."** + coherent multi-sentence; **PPL 20.13**. Crown **6.4527** +
  dense qwen3-8b **NLL 2.944273** bit-exact; **ctest 30/30**. Commits: `kEXL3` dtype; native forward
  primitives (`gemv_exl3_forward` = had128(A⊙suh)→decode-MAC→had128⊙svh w/ per-device scratch; scaled
  Hadamard; F16 embed); dense loader+dispatch (`upload_exl3`, `DenseQuantPtr{+suh,+svh,+bits}`, `gemv_q`
  kEXL3 branch → `gemv_q_T` inherits, EXL3 output/lm_head + F16 token_embd); `ie import` EXL3→GGUF
  (verbatim trellis + suh/svh, computed llama3 rope_freqs). **Both landmines handled:** rope_freqs
  COMPUTED (not copied), EXL3 Q/K NOT un-permuted (HF-natural NEOX, like phi3). **⚠ Owner-corrected
  scope mid-build:** the lm_head ships as a genuine EXL3 6bpw tensor → decoded natively (NOT dodged via
  tied embeddings); token_embd kept F16 (not re-quantized) — fully faithful, no model manipulation.
- **✅ Perf pass — decode kernel (2026-06-19):** 8-way ILP unroll of the gemv_exl3 trellis-decode loop
  (the per-weight decode is a latency chain; serializing through one accumulator stalled the pipe →
  8 independent decode+FMA chains, TK%8==0 always since K%128==0). **gemv_exl3 2.71× → ~1.15× Q4_K @T=1**
  (worst-case k_proj N=512, lowest occupancy), cosine still **1.0**; **EXL3 1B end-to-end 31.9 → 19.4
  ms/token (1.64×)**, PPL 20.13 bit-identical. Crown bit-exact by construction (gemv_exl3 is kEXL3-only).
### ▶ IN PROGRESS — Phase D: 80B EXL3 on 2×B70 (2026-06-19, build queued)
**Strategy (owner-approved):** build+validate the EXL3 MoE kernel against a ready-made non-abliterated
EXL3 file FIRST, then quantize an abliterated base ourselves and swap it in (no abliterated 80B EXL3
exists off-the-shelf). **Two downloads done/running:**
- **✅ DOWNLOADED + VERIFIED:** `turboderp/Qwen3-Next-80B-A3B-Instruct-exl3` **4.51bpw** →
  `~/models/Qwen3-Next-80B-Instruct-exl3-4.51bpw` (6/6 shards, 43 GB, 222,377 tensors). The
  kernel-build scaffolding. 4.51bpw = 45 GB ≈ the working Q4_K 80B footprint → fits 64 GB (2×B70).
- **⏳ DOWNLOADING (background, Data drive):** `huihui-ai/Huihui-Qwen3-Coder-Next-abliterated` BF16 base
  (~160 GB) → `/media/weezy/Data/exl3-quant/Huihui-Qwen3-Coder-Next-abliterated`. The abliterated end
  target; we'll quantize it to EXL3 with exllamav3 (LAYER-BY-LAYER → low VRAM, fits the B70s; cost is
  disk+time, ~overnight, GPU-exclusive). Can't shortcut from the Q4_K GGUF (compounds quant loss).
**Arch:** the 80B EXL3 = the SAME `qwen3_next` the engine already runs at Q4_K (`qwen3next.cpp`, `next_`
2-card split) → "make qwen3next decode EXL3", NOT a new arch. `qwen3next.cpp` is NOT the crown (qwen36.cpp
is) → editable. **DESIGN RESOLVED → FUSE experts:** 222k-tensor GGUF is impractical, so the importer
concatenates each layer's 512 experts into per-layer BANKS `blk.L.ffn_{gate,up,down}_exps.{trellis,suh,svh}`
(~432 expert tensors, not 221k). Per-expert gate trellis `[16*bits, N/16, K/16]` (K=hidden 2048,
N=moe_inter 512); suh`[512,K]`, svh`[512,N]`. **Forward REUSES `gemv_exl3_forward`** per active expert via
pointer offsets (trellis bank + e·expert_bytes; suh +e·K; svh +e·N) — fused storage AND kernel reuse, no
new fused kernel yet. **EXL3 tensors:** `linear_attn.{in_proj_qkvz,out_proj}`, the expert banks,
`mlp.shared_expert.{gate,up,down}`, `lm_head`(6bpw). **Raw F16/F32 (NOT EXL3):** `embed_tokens`(F16),
norms, `linear_attn.{A_log,conv1d,dt_bias,in_proj_ba}`, `mlp.gate`(router), `mlp.shared_expert_gate`.
**BUILD STEPS (fresh-context start):** (1) read `qwen3next.cpp` loader to lock the GGUF tensor names it
expects; (2) write the qwen3next EXL3 importer→GGUF (fused expert banks; run+inspect on the downloaded
model — self-contained, testable first); (3) `qwen3next.cpp` loader EXL3 branches (per-card upload for the
2-GPU split); (4) forward: reuse `gemv_exl3_forward` for attn/DeltaNet/shared + per-expert EXL3 MoE loop;
(5) run split 2×B70 — **debug by op/layer-diffing vs the working Q4_K `qwen3next` (same-arch live oracle)**.
**Remaining perf (after correctness):** matched-baseline PPL (EXL3 ≤ Q4_K, needs a Llama-3.2-1B Q4_K
~0.8 GB download); EXL3 prefill is still per-token (gemv_q_T loops gemv_exl3_forward) → batched gemm_exl3;
fused EXL3 MoE kernel. **Disk:** owner deleted ~95 GB AWQ/GPTQ → root `/` now ~95 GB free (NVMe, fast =
where the 80B EXL3 lives); `/media/weezy/Data` 1.7 TB (slow, holds the 160 GB BF16 base). Download CLI =
`~/.local/bin/hf` (the `/tmp/exl3venv` lost `huggingface_hub`). Imported 1B GGUF (Phase C):
`~/models/exl3-test/llama32-1b-exl3.gguf`. Tooling venv: `/tmp/exl3venv` (python only).

#### ▶ RECON RESOLVED (2026-06-19, fresh-context push) — importer spec is byte-locked
Four investigations done (loader contract, existing 1B EXL3 code, scaffold safetensors, live
Q4_K reference via `ie-inspect`). **Reference GGUF (the live oracle to diff against) =**
`~/models/Momix-44/Huihui-Qwen3-Coder-Next-Opus-4.6-Reasoning-Distilled-abliterated/…-Q4_K_M.gguf`
(843 tensors, the exact contract `qwen3next.cpp` reads). **Resolved findings:**
- **⚠ LANDMINE — EXL3 bit-width is VARIABLE PER LAYER, uniform across the 512 experts in a
  layer.** Measured: layers 0/1/5 = 5-bit experts, layers 24/47 = 4-bit; whole model is a 4/5-bit
  mix (→ 4.51 avg), `lm_head` = 6-bit (`head_bits`). So the importer MUST read `bits` per
  (layer, projection) from each trellis and store `ne0=16*bits` in the fused tensor — NOT assume a
  global bits. (Source trellis layout = numpy `[K/16, N/16, 16*bits]` ⇒ `bits = shape[2]/16`,
  `K=shape[0]*16`, `N=shape[1]*16`. ggml `ne` is the reverse: `[16*bits, N/16, K/16]`, bytes verbatim.)
- **Per-expert trellis bytes = K·N·bits/8** (verified: 5-bit gate = 655,360 B). Uniform within a
  layer ⇒ `stride = nbytes/E`. Fused bank shape (4D, experts OUTERMOST = contiguous slab):
  `blk.L.ffn_{gate,up,down}_exps.weight` = kEXL3 `ne=[16*bits, N/16, K/16, E]`,
  `.suh` = F16 `ne=[K,E]`, `.svh` = F16 `ne=[N,E]`. Per-expert: `trellis + e*(nbytes/E)`,
  `suh + e*K`, `svh + e*N`. (Mirrors Q4_K which stores experts 3D `[in,out,E]`, stride `nbytes/E`.)
- **⚠ LANDMINE — DeltaNet `in_proj_qkvz` is ONE fused EXL3 trellis (`[2048→12288]`) and CANNOT be
  split** (trellis is opaque). The Q4_K GGUF splits it into `attn_qkv`(8192)+`attn_gate`(z,4096);
  the EXL3 path must instead emit a SINGLE `attn_qkv.weight` (EXL3, N=12288) and SLICE the
  *activation* after the gemv: `[0:8192]`→qkv (conv path), `[8192:12288]`→z gate. Loader+forward
  EXL3 branch owns this; no `attn_gate` tensor on the EXL3 path. (qkvz order = q2048|k2048|v4096|z4096.)
- **Full-attn `q_proj` is fused q|gate (N=8192)** — emit EXL3 verbatim; forward already splits.
- **Source→GGUF mapping table** (importer = new `import_exl3_qwen3next_to_gguf`, dispatched on
  `model_type==qwen3_next` + `quant_method==exl3`; the existing llama `import_exl3_to_gguf` is
  dense-only and hard-rejects non-llama):
  | source (safetensors) | GGUF name | dtype | note |
  |---|---|---|---|
  | `…in_proj_qkvz.{trellis,suh,svh}` | `blk.L.attn_qkv.weight/.suh/.svh` | kEXL3 | fused N=12288, slice in fwd |
  | `…in_proj_ba.weight` (F16 [64,2048]) | `blk.L.ssm_ba.weight` | F16 | verbatim bytes, ne=[2048,64] |
  | `…out_proj.{trellis,…}` | `blk.L.ssm_out.weight/.suh/.svh` | kEXL3 | K=4096,N=2048 |
  | `…A_log` (BF16 [32]) | `blk.L.ssm_a` | F32 | cast; **name has NO .weight** |
  | `…dt_bias` (BF16 [32]) | `blk.L.ssm_dt.bias` | F32 | cast |
  | `…conv1d.weight` (BF16 [8192,1,4]) | `blk.L.ssm_conv1d.weight` | F32 | cast; squeeze→ne=[4,8192] (same bytes) |
  | `…norm.weight` (BF16 [128]) | `blk.L.ssm_norm.weight` | F32 | cast |
  | `input_layernorm` | `blk.L.attn_norm.weight` | F32 | cast |
  | `post_attention_layernorm` | `blk.L.post_attention_norm.weight` | F32 | cast |
  | `self_attn.{q,k,v,o}_proj.{trellis,…}` | `blk.L.{attn_q,attn_k,attn_v,attn_output}.weight/.suh/.svh` | kEXL3 | full-attn only (L%4==3) |
  | `self_attn.{q,k}_norm` | `blk.L.{attn_q_norm,attn_k_norm}.weight` | F32 | cast |
  | `mlp.gate.weight` (F16 [512,2048]) | `blk.L.ffn_gate_inp.weight` | F32 | router; cast, ne=[2048,512] (same bytes) |
  | `mlp.experts.E.{g,u,d}_proj.*` ×512 | `blk.L.ffn_{gate,up,down}_exps.{weight,suh,svh}` | kEXL3 | FUSE 4D banks |
  | `mlp.shared_expert.{g,u,d}_proj.*` | `blk.L.ffn_{gate,up,down}_shexp.weight/.suh/.svh` | kEXL3 | |
  | `mlp.shared_expert_gate.weight` (F16 [1,2048]) | `blk.L.ffn_gate_inp_shexp.weight` | F32 | cast, squeeze→ne=[2048] |
  | `model.embed_tokens` (F16) | `token_embd.weight` | F16 | verbatim, ne=[2048,vocab] |
  | `model.norm` (F16) | `output_norm.weight` | F32 | cast |
  | `lm_head.{trellis,suh,svh}` | `output.weight/.suh/.svh` | kEXL3 | 6-bit |
  - **Metadata KVs to emit** (from `ie-inspect` of the ref, all `qwen3next.*`): block_count 48,
    context_length 262144, embedding_length 2048, feed_forward_length 5120, attention.head_count 16,
    .head_count_kv 2, .key_length 256, .value_length 256, .layer_norm_rms_epsilon 1e-6,
    expert_count 512, expert_used_count 10, expert_feed_forward_length 512,
    expert_shared_feed_forward_length 512, rope.freq_base 5e6, rope.dimension_count 64,
    full_attention_interval 4, ssm.{state_size 128, conv_kernel 4, group_count 16, time_step_rank 32,
    inner_size 4096}. Tokenizer KVs copied from the ref GGUF (`copy_tokenizer_kvs`, gpt2/qwen2 pre).
    No `rope_freqs` tensor (plain NEOX rope, no llama3 scaling).
  - **TEST (Step-2 gate):** run importer on the scaffold → `ie-inspect` the output → tensor
    list/shape/dtype matches the Q4_K reference EXCEPT the deliberate EXL3 deltas (kEXL3 + .suh/.svh
    siblings; single `attn_qkv` not split; F16 ssm_ba). Self-contained, no GPU.
  - **✅ DONE (2026-06-19) — importer built + validated.** `import_exl3_qwen3next_to_gguf`
    (`hf_import.cpp`, STREAMED two-pass = RAM-safe for 43 GB; dispatched in `ie_import.cpp` on
    `model_type qwen3_next` + `quant_method exl3`). Ran on the scaffold in 7m26s → 43 GB GGUF at
    `/media/weezy/Data/exl3-quant/qwen3next-80b-exl3.gguf`. **1625 tensors, all counts reconcile**:
    EXL3 409 (265 singles + 144 fused trellis), F16 855 (all suh/svh + embed + ssm_ba), F32 361
    (casts); 432 fused-expert banks = 48×3×3. `ie-inspect` L0/L3 shapes all decode correctly
    (attn_qkv `[80,768,128]`=bits5/K2048/N12288 fused; ffn_gate_exps `[80,32,128,512]`=[16b,N/16,K/16,E];
    L3 attn 4-bit while L3 experts 5-bit → per-tensor bits confirmed). The `ie-inspect` group-checker
    shows MISMATCH lines — EXPECTED (it's coded for the Q4_K layout; every EXL3 tensor now has
    .suh/.svh siblings). **⚠ WATCH-ITEM for Step-5 validation:** this EXL3 GGUF has
    `rope.freq_base=1e7` (faithful to the turboderp EXL3 config = official Qwen3-Next) but the Q4_K
    oracle GGUF has `5e6` (a different checkpoint). Full-attn (every 4th) layers WILL diverge from the
    Q4_K oracle on rope alone — match 5e6 temporarily or expect it; DeltaNet layers are rope-free.
  - **✅✅ FORWARD CORRECT — 80B EXL3 COHERENT on 2×B70 (2026-06-19, commit 80eaeef):**
    `ie run --gpus 2` → **"The capital of France is Paris."** + a correct 2-sentence Rayleigh-scattering
    answer. End-to-end EXL3-native across both cards: DeltaNet + full-attn + **per-expert EXL3 MoE
    loop** (gemv_exl3 per active expert via fused-bank offsets) + shared expert + EXL3 lm_head. Loader
    (`5b5f50c`) + forward (`376ad84`) + 3 correctness fixes (`80eaeef`). **The 3 bugs (all found by
    diffing my GGUF tensor stats vs the engine-validated Q4_K GGUF — the transforms stand out even
    across different model weights):** (1) norms needed **+1** (qwen3next `(1+w)` RMSNorm; all norms
    except `linear_attn.norm`) — importer emitted raw HF; (2) `ssm_a` needed **`-exp(A_log)`** (DeltaNet
    decay folded at convert) — `-exp` reproduced the Q4_K range exactly; (3) `slice_qkvz` must
    **de-interleave** the fused `in_proj_qkvz` (raw HF is head-grouped `[q|k|v|z]×16` → contiguous
    `[q_all|k_all|v_all]`+z) — was a naive contiguous slice. Mirrors llama.cpp `qwen.py:297/303/320-331`.
    `conv1d`/`ssm_ba` stay verbatim (HF applies conv post-split → already de-interleaved). Loader dtype
    gaps also closed: F16 `token_embd` (embedding_lookup_f16), EXL3 `lm_head` (output_suh/svh/bits),
    EXL3 shared-expert via `LX`. **▶ REMAINING (Step 5+):** (a) crown PPL gate re-confirm (running);
    (b) the per-expert MoE loop is DELIBERATELY SLOW (correctness-first, K×3 gemv_exl3 launches/token/
    layer — no batched trellis MoE kernel yet) → perf pass = a fused EXL3 MoE kernel; (c) EXL3 PPL
    number; (d) **then quantize the abliterated base** (`/media/weezy/Data/exl3-quant/Huihui-...-
    abliterated`, 149 GB BF16 DOWNLOADED) → EXL3 with exllamav3, swap in (same kernel). GGUF on NVMe:
    `~/models/exl3-80b/qwen3next-80b-exl3.gguf`. Watch-item: rope 1e7 (EXL3) vs 5e6 (Q4_K oracle).
  - **▶ LOADER + FORWARD design RESOLVED (Step 3–4), scoped from the qwen3next forward read:**
    KEY WIN — **all single linears already route through `dense::gemv_q_T`, which dispatches
    `kEXL3 → gemv_exl3_forward` (`dense_dispatch.hpp:458`)**. So attn_qkv, ssm_out, attn_q/k/v/output,
    router(F16), shared gate/up/down ALL inherit EXL3 decode for free once uploaded as `DenseQuantPtr`
    with suh/svh/bits. `LayerW` already stores these as `DenseQuantPtr`. **Loader (Step 3):** (a) a
    linear helper that, when `Tl(L,name).dtype==kEXL3`, finds the `.suh`/`.svh` siblings and calls
    `dense::upload_exl3` (else `upload_weight_auto`) — for attn_qkv/ssm_out/attn_q/k/v/output/shared;
    (b) `attn_gate` becomes CONDITIONAL (absent in EXL3 — the z-gate lives inside the fused qkvz);
    (c) `ssm_ba` is F16 → existing `upload_weight_auto` path (no change); (d) `up_exps` gains a kEXL3
    branch: upload trellis bank (`*_exps`/`*_stride` as today) + new `*_suh`/`*_svh` bank ptrs +
    `*_bits` (new `LayerW` fields). **Forward (Step 4) — only TWO new pieces:** (1) DeltaNet: when
    `attn_qkv.dt==kEXL3`, gemv into a new `[T,12288]` scratch `dn_qkvz`, then slice `[0:8192]→dn_qkv`
    (conv path) and `[8192:12288]→dn_z` (z-gate), SKIP the separate attn_gate gemv — rest of DeltaNet
    unchanged (qkvz order q2048|k2048|v4096|z4096). (2) MoE: when `gate_dt==kEXL3`, a per-expert loop
    (the Q4_K batched kernels can't decode trellis) — for each token's K active experts:
    `h=silu(gemv_exl3(x,gate_e))⊙gemv_exl3(x,up_e)` then `out=gemv_exl3(h,down_e)`, `moe_y+=wgt·out`;
    per-expert offsets trellis `+e*stride`, suh `+e*K`, svh `+e*N` (gate/up K=H,N=EF; down K=EF,N=H).
    Shared expert + router need NO change (gemv_q_T inherits). **Step 5 = GPU validate vs Q4_K oracle
    (trainer idle ⇒ GPU free); mind the rope 1e7-vs-5e6 watch-item on full-attn layers.**

### ▶ STATE 2026-06-20 — 80B EXL3 COHERENT; perf + abliterated-encoder are the frontier
**Done & committed:** 80B Qwen3-Next EXL3 runs coherently on 2×B70 (`80eaeef`, "Paris." + Rayleigh
answer); crown gate 6.4527 bit-exact (no regression). Importer/loader/forward all in. See the Phase D
"FORWARD CORRECT" block above for the 3 bugs fixed.
**▶▶ OWNER DECISION (2026-06-20 EOD — RESOLVED, supersedes the earlier "defer"): SHELVE the EXL3
encoder. The abliterated goal is ALREADY MET by a Q4_K_M GGUF on disk** — the EXL3-for-abliterated
quest was solving a problem we don't have. Data that settled it (same box, 2×B70, same model class):
`~/models/Momix-44/Huihui-Qwen3-Coder-Next-Opus-4.6-Reasoning-Distilled-abliterated-Q4_K_M.gguf`
(46 GB) runs COHERENT on the mature Q4_K fast path at **prefill 326 (→500 long-ctx) / decode 57.9
tok/s** — vs the abliterated-EXL3 path which would be **45 GB (same size), 2 cards (same), prefill 55
/ decode 32** AND cost weeks to build the encoder. EXL3 4.5bpw buys nothing here: no card-count win
(45 GB > 32 GB still needs 2 cards), ~Q5_K quality delta = practical noise, and ~6× slower prefill
(trellis decode is ALU-bound / GEMV-style; closing it = a batched trellis-GEMM, a big lift).
**⇒ For "uncensored 80B on the box" use the Q4_K_M GGUF.** EXL3 stays a SHIPPED CAPABILITY (first/
fastest on Arc; fused MoE kernel done) but is NOT an active workstream. Its only real future niche =
AGGRESSIVE LOW-BPW: EXL3 3.0bpw ≈ 30 GB → fits a SINGLE B70 (or frees VRAM for long-context KV) where
GGUF Q3 quality collapses but EXL3 holds. Revisit ONLY if that 1-card/long-ctx case becomes a priority.
**Earlier-this-session work that stands:** fused EXL3 MoE kernel (item 1 below, decode 1.46×/prefill
1.78×) — keeps the EXL3 capability usable. The W4A16 fallback and the encoder are both moot now.
**Two open work items:**
1. **PERF — fused EXL3 MoE kernel. ✅ DONE (2026-06-20 PM).** Replaced the per-(token,expert) slow loop
   with a BATCHED path over all R=T·K active rows → **11 launches/layer regardless of T,K** (was ~11·T·K).
   Two new primitives, each a direct batched extension of a PROVEN kernel + per-row expert gather
   (lowest-risk route — reuses the bit-exact decode math rather than re-deriving it inside a monolith):
   `gemv_exl3_moe` (`src/ops/gemv_exl3.cpp`, = gemv_exl3 + leading row dim) and `hadamard_transform_moe`
   (`src/ops/hadamard.cpp`, = WHT + per-expert suh/svh gather). suh/svh are PER-EXPERT so the Hadamards
   stay separate launches (can't hoist), but the K·T·3 trellis gemvs collapse to 3 batched launches; the
   output-Hadamard-before-swiglu ordering (which `docs/exl3_fused_moe_design.md` glossed) is handled
   correctly. Wired in `qwen3next.cpp` behind `IE_EXL3_MOE_SLOW=1` (the slow loop kept as oracle); reuses
   `moe_*` scratch + 1 new half buf (`moe_h2`) + 1 int (`moe_rowtok`); handles decode AND prefill.
   **VALIDATED on 2×B70 (same EXL3 GGUF):** greedy output BYTE-IDENTICAL to the slow oracle
   (' Paris. The capital of Germany is Berlin.'); EXL3 PPL **4.7296** ≈ baseline 4.7282 (fp-reduction
   noise); crown gate **6.4527 bit-exact** (≤6.57). **PERF A/B (order-controlled, --warmup 1, 3 runs):
   decode 21.91 → 31.95 tok/s = 1.46×; prefill 31.2 → 55.5 tok/s = 1.78×.** (Note: the older 51.8 tok/s
   in MEMORY is the Q4_K model, NOT this EXL3 model — not a valid baseline; 21.91 is the EXL3-slow number.)
   NEXT-PERF (optional, if more needed): the full inlined 2-kernel monolith from the design doc could push
   further, and the per-layer `q.memcpy(...).wait()` (48 syncs/token) is a candidate to remove.
2. **ABLITERATED EXL3 — ⛔ SHELVED 2026-06-20 EOD (see OWNER DECISION above; goal met by Q4_K_M GGUF).
   The analysis below is kept for the record / future low-bpw revisit only — it is NOT active work.**
   **⚠ NO CLOUD (owner decision, firm).** The 149 GB abliterated BF16 base is
   downloaded (`/media/weezy/Data/exl3-quant/Huihui-Qwen3-Coder-Next-abliterated`). To get it into EXL3
   we must QUANTIZE it. **exllamav3 (the only EXL3 quantizer) is CUDA-only — cannot run on Arc, and
   cloud rental is OFF THE TABLE.** ⚠ I earlier WRONGLY claimed we could quantize "overnight on the
   B70s" — that was an unverified assumption; exllamav3 won't run on Arc at all. **Cloud-free paths:**
   (a) find a pre-made abliterated qwen3-next EXL3 on HF (sidesteps everything — verify first);
   (b) **build our own EXL3 ENCODER** that runs on Arc/CPU (we have the proven DECODER + numpy oracle +
   format cracked; the encoder = QTIP incoherence/Hessian scales + tail-biting trellis Viterbi search —
   the harder half, FEASIBILITY UNDER ASSESSMENT, do NOT promise until verified); (c) abliterated model
   in a quant we already make on Arc (loses EXL3 quality). Owner wants (b) if feasible.
   **✅ ENCODER FEASIBILITY VERDICT (2026-06-20, source-grounded in `~/exllamav3-ref`): FEASIBLE-HARD
   — ⚠ SUPERSEDED IN PART, read the ▼ CORRECTION below before trusting the "one kernel" framing.**
   The encode pipeline is ~90% PORTABLE torch (Hadamard, LDL/Cholesky, regularization, sign-flips,
   packing — device-generic, NOT hardcoded `.cuda()`, runs on CPU as-is;
   `exllamav3/modules/quant/exl3_lib/quantize.py:944+`). **Exactly ONE CUDA-only piece:**
   `quantize_tiles()` — the tail-biting **Viterbi trellis search** (`exllamav3_ext/quant/quantize.cu:14-265`,
   ~200 lines, called at quantize.py:454 & 566). It's a STANDARD Viterbi (QTIP paper arXiv:2406.11235,
   not proprietary) → reimplementable in ~100-150 lines numpy/torch. The `torch.cuda.Stream/Event/
   synchronize` calls are all in multi-GPU branches → stub on CPU. Codebook `decode_3inst`
   (`codebook.cuh:25-56`) must be replicated (we already have cb=0 in our decoder). **COST: the CPU
   Viterbi is SLOW** — naive ~1 week for the 80B/149GB, ~1-2 days with numba/vectorization (ONE-TIME,
   offline, on-box, no cloud). **RISK: FP16 numerical subtlety in the path search** — a naive port may
   round-trip-valid but give worse PPL. **DE-RISK PLAN:** (1) implement CPU `quantize_tiles_cpu` (numpy
   Viterbi + decode_3inst); (2) quantize a SMALL model (e.g. the 1B) on CPU → decode with our proven
   decoder → check PPL near-fp16 BEFORE the 80B (no bit-exact CUDA reference is possible — random
   sign-flips + no CUDA on-box — so validation is end-to-end PPL, not bit-match); (3) then the 80B.
   Build the CPU encoder as a fork/patch of `~/exllamav3-ref` (swap quantize_tiles + stub cuda syncs),
   driven from a venv. **This is a multi-session build — start in fresh context.**
   **▼ CORRECTION (2026-06-20 PM, source-grounded in `convert_model.py:560-727` + `linear.py:389-402`
   + `fp16.py:82-85` + `attn.py`): the "one Viterbi kernel / 90% portable" verdict above was measuring
   ONLY `quantize.py`. It MISSED the conversion DRIVER, which is the real cost and is CUDA-bound.**
   EXL3 conversion is **sequential GPTQ-style error-feedback**, NOT a single fp16 pass: per module N it
   (1) captures Hessians `H=Σxxᵀ` by forwarding a running calibration `state` through the *unquantized*
   module (`convert_model.py:602`, default **250 seqs × 2048 tok**), (2) quantizes with H, (3) reloads
   the module **quantized**, (4) **advances `state` by forwarding through the now-EXL3 weights**
   (`:710`). So H for layer N depends on the quantized outputs of 0..N-1, and the advance-forward runs
   on EXL3 weights → `ext.exl3_mgemm` (CUDA). The calib forward also hits `ext.hgemm` (cuBLAS, on
   dtype-mismatch), `ext.count_inf_nan`, `ext.softcap` — **none have a torch fallback**. ⇒ exllamav3's
   forward CANNOT run on Intel XPU and on CPU the 80B sequential forward over 250×2048 tok is the
   dominant cost (multi-day→week+, likely worse), not the Viterbi. **TWO REAL ARCHITECTURES:**
   **(A) exllamav3-fork-on-CPU** — port Viterbi to numpy, run the rest (incl. forward) on CPU. Simplest
   to wire, but the CPU 80B sequential forward is brutally slow + high-risk. **(B) OUR-ENGINE-AS-FORWARD
   (recommended if we build it):** our engine ALREADY decodes EXL3 on the 2×B70 (Phase C/D) — so it can
   BE the per-layer calibration forward on-GPU: forward `state` with layers 0..N-1 as EXL3 + layer N as
   BF16 (need a BF16 linear path — we have F16 gemv) while capturing `H=Σxxᵀ` per linear input; hand H to
   an offline Python LDL+Viterbi quantizer (the portable part); reload layer N as EXL3; advance; repeat.
   CUDA-free, heavy compute stays on the GPU. Bigger engine-integration effort (the sequential
   orchestration loop + H-capture instrumentation + BF16 linear), and the fidelity risk is matching
   exllamav3's exact H/incoherence/scale conventions. **NET: encoder is a WEEKS-scale R&D build, not the
   "port one kernel" the line above implied. Decision pending with owner; fused MoE kernel goes first
   regardless. Fallback in pocket: abliterated W4A16 exists on HF + we have AWQ import → uncensored 80B
   on Arc now, below EXL3 quality.**

### ▶ SCOPED — speculative decoding (2026-06-20): FEASIBLE + LOSSLESS, drafter exists; DEFERRED for product
Owner asked to scope + verdict. **Verdict: worth it on the merits, but DEFERRED** — 58 tok/s decode is
already comfortably usable for an interactive product, and building the product harness is higher-leverage
right now. Revisit as a perf-v2 once the product exists / long-gen latency becomes a real user complaint.
- **LOSSLESS (the hard constraint, satisfiable):** greedy = accept iff draft==target argmax → BIT-IDENTICAL
  (validate exact token-match spec-on vs spec-off); sampling = Leviathan/Chen modified rejection rule →
  SAME distribution (validate PPL within fp noise). Quality only breaks via a bug → caught by the gate.
- **ENABLERS ALREADY EXIST** (built for the prefix cache): `DeltaNetState::copy_from` (snapshot/restore)
  + `KvCache::set_length` (per-layer rewind) + `FleetPrefixCache` per-card snapshots. forward() handles T>1.
- **HYBRID CATCH (the real complexity):** `deltanet_recurrence` updates the recurrent state ONCE per
  `forward(T=K)` (`qwen3next.cpp:725`), NOT per token → mid-chunk reject needs snapshot-before-verify +
  restore + re-`forward(T=accepted)`. Also `forward()` must gain an all-position-logits mode (today it runs
  lm_head on the last token only, `:943`). On a plain transformer rollback is just a KV pointer rewind; the
  DeltaNet state is what makes this harder here.
- **DRAFTER CONFIRMED on disk:** target abliterated 80B = vocab **151936**, `gpt2`/`qwen2`-pre, eos 151645.
  Same-vocab small Qwen present: **Qwen2.5-0.5B (380MB, best)**, Qwen2.5-3B, Qwen3-4B. ⚠ none abliterated →
  lower acceptance on uncensored content (costs SPEED, not quality); distill/abliterate a drafter later for more.
- **EFFORT ~1 week** (drafter integ via DenseModel path + all-pos logits + reject sampler + state-rollback
  orchestration + engine loop + lossless validation harness). **Payoff ~1.3–2.5× decode** (58→~75–145 tok/s),
  acceptance-rate-dependent. Build greedy-first (simplest, provably bit-lossless), measure real acceptance, then sampling.

### ✅ DONE — 80B `next_` fleet prompt/KV cache (2026-06-19, GPU-validated)
**Built + validated this session.** Crown prompt cache (`2c6a535`) is now extended to the
Qwen3-Next-80B layer-split path. Spec: `docs/superpowers/specs/2026-06-19-prompt-cache-fleet-80b.md`.
What shipped:
- **`FleetPrefixCache`** (`include/ie/fleet_prefix_cache.hpp` + `src/core/fleet_prefix_cache.cpp`,
  registered in `src/CMakeLists.txt`): per-endpoint `vector<unique_ptr<KvCache>>` +
  `vector<unique_ptr<DeltaNetState>>`, one snapshot per card on `fleet->dev(dev)`. Trie/LRU copied
  from `PrefixCache`. `init()` builds per-card snapshots into LOCALS first, commits only on full
  success (an OOM mid-loop frees cleanly and skips the insert — there's no free-VRAM query, so a null
  device alloc surfacing as an init-error string is the "skip, don't crash" guard). Default **12**
  entries (`FleetPrefixCacheConfig`).
- **`Qwen3NextModel` accessors** (`qwen3next.hpp`): `n_devices`/`fleet`/`kv_cache(dev)`/`dn_state(dev)`/
  `dev_has_kv`/`dev_has_dn` (kv/dn non-const so the engine restores INTO live state).
- **`Engine`** (`engine.hpp`/`engine.cpp`): `fleet_cache_` member declared AFTER `next_model_`/`fleet_`
  (destructs first → frees snapshots while the fleet allocators are alive). `load()` inits it under the
  `kQwen3Next` branch (gated, reuses `prompt_cache_on_`). `generate()` got a `next_` RESTORE branch
  (per-card `copy_prefix_from`/`copy_from` on each card's own queue; **all-or-nothing** — any card
  missing a snapshot or erroring → `restored=0`, so the `pos==0` forward cleanly resets every card
  before a clean full prefill; this avoids a half-restored card silently corrupting a hit since the
  `next_` forward does NOT reset on `pos≠0`) + a `next_` INSERT branch. Both reuse the arch-agnostic
  `snap_at`/`pos=restored`/`cached_tokens` plumbing and the crown's stable-boundary `cache_prefix_len`
  (carried over verbatim; `chat()` already computes it for qwen3next via the ChatML branch).
- **Default-OFF, OFF-path byte-identical** (`IE_PROMPT_CACHE=1` / `opts.prompt_cache` to enable).
- **VALIDATION (2×B70, Reasoning-Distilled abliterated 80B `Huihui-Qwen3-Coder-Next-...-Q4_K_M`,
  `ie serve --gpus 2 --ctx 4096`, temp=0):** crown PPL **6.4527 / NLL 1.864495 bit-exact** (crown
  untouched); ctest **29/29** serial (the lone `-j` fail is GPU cross-taint per the exclusivity rule —
  passes isolated); deterministic 2-turn conversation **ON vs OFF byte-identical on BOTH turns**;
  turn-2 `[gen]` **`(16 cached)` ON vs `(0 cached)` OFF** → per-card KV+DeltaNet snapshot/restore is
  exact across cards. Harness: `/tmp/fleet_validate.sh`.

### ✅ DONE — prompt cache flipped DEFAULT-ON (2026-06-19, GPU-validated)
`EngineOptions.prompt_cache` now defaults **`true`** (crown + fleet share it); opt out via env
**`IE_NO_PROMPT_CACHE=1`** (or set the field `false`) — the `load()` guards became
`(opts.prompt_cache || IE_PROMPT_CACHE) && !IE_NO_PROMPT_CACHE`. Only the product (`ie run`/`serve`)
and the two manual load-tests construct an `Engine`; nothing in ctest does, so the suite is unaffected.
Re-validated after the flip: crown PPL **6.4527 bit-exact** (ie-perplexity doesn't go through Engine);
ctest **29/29**; 80B 2-turn **default (no env) → turn-2 `16 cached`** (cache engages with no flag) vs
**`IE_NO_PROMPT_CACHE=1` → `0 cached`**, outputs **byte-identical** both turns. Harness:
`/tmp/fleet_validate_flip.sh`.

### ▶ NEXT TASK (fresh context starts here)
1. **Big engine bet — EXL3 (QTIP-based SOTA quant) on Arc** — plan
   `docs/superpowers/plans/2026-06-19-qtip-quant-engine.md` + `docs/exl3_format_notes.md`; exllamav3
   ref cloned at `~/exllamav3-ref`. Strategic frame in memory `[[project_strategy_pivot_2026-06-18]]`
   (Seal dropped → Pi-extension harness → permissive OSS → EXL3-on-Arc as the first/fastest-on-new-
   arches-on-Arc angle).
- **Gate/infra reminders:** canonical crown Q4_K_M is restored on disk; GPU-exclusivity rule (one
  workload at a time — serialize vs any trainer run); enable the prompt cache with `IE_PROMPT_CACHE=1`
  until it's flipped default-ON.
- **Also committed earlier this session:** F16/Q8 projection dispatch (`0a0bcbb`); EXL3/QTIP plan +
  `docs/exl3_format_notes.md` (`eb50f56`).

### ★ AGENT-HARNESS ENABLEMENT — OpenAI server now drivable by agents (2026-06-18)
Validated the "engine as a backend for a minimal agent harness" path LIVE: **Pi
(`pi.dev`, MIT) pointed at `ie serve` drove Qwen3-Coder-Next 80B agentically across
2×B70** (openai-compat custom provider in `~/.pi/agent/models.json`). The model
reasoned and chose to call the bash tool — the hybrid is real. Found + fixed the one
blocker and two robustness bugs in the OpenAI server (all server/Engine-layer, crown
forward UNTOUCHED):
- **Structured `tool_calls` (the unlock):** the server emitted `<tool_call>` TEXT,
  not the OpenAI `tool_calls` field, so standard agents couldn't EXECUTE calls. New
  `parse_response_tool_calls` in `openai_proto.cpp` converts the model's
  `<tool_call>{…}</tool_call>` → structured `tool_calls` (function name + JSON-string
  args + `finish_reason:"tool_calls"`), non-stream + streaming delta; unit-tested
  (`openai_proto_test`). Turns the engine from "fast chat backend" into "backend any
  openai-compat agent (Pi/Claude Code/Cursor) can drive."
- **`context_length_exceeded` → clean HTTP 400** (was: empty `"length"` response that
  agents read as a generic failure → circuit-breaker). `Engine::generate` returns a
  distinct finish_reason; the server maps it to a proper OpenAI 400 so the agent can
  compact + retry. Does NOT cap context (guard sits at full `max_ctx`).
- **Streaming UTF-8 crash fix:** the new streaming tool-call buffering re-sliced text
  at byte offsets → split a multi-byte char → `json::dump` threw `type_error.316` →
  `terminate`/abort. Fixed with a UTF-8-boundary clamp + `error_handler::replace` on
  all response dumps. Regression-tested.
- **Per-request tok/s logging:** `Engine::generate` times prefill/decode (free —
  phase-boundary clock reads); the server prints `[gen] prefill N tok / Xms = Y tok/s
  | decode M tok / Zms = W tok/s` per request. (Showcase + writeup fuel.)
Files: `src/server/openai_{proto,server}.cpp`, `include/ie/openai_proto.hpp`,
`src/engine/engine.cpp`, `include/ie/engine.hpp`, `tests/unit/openai_proto_test.cpp`.
Gates: build green, `openai_proto_test` all OK, **crown PPL 6.4527 bit-exact**.
**Harness design captured:** `docs/superpowers/specs/2026-06-18-mach-x-harness-design.md`
— fork Pi + context-engineering spine (compaction / tool-output truncation /
sub-agent isolation) + self-extension; **prompt/KV caching in the server is the named
next keystone engine task** (kills the per-turn re-prefill of the ~10k agent prefix —
the "felt slow" tax). NB the engine has NO server-side prompt cache yet (stateless;
re-prefills the full prompt every request).

### ★ OWNER-SET LAUNCH ROADMAP — "Mach X" (2026-06-13)

Product name LOCKED: **Mach X Inference Engine** (§5 #3). Two repos go to GitHub
**now** (before more tuning) so the owner can work the app side on a 2nd PC:
**(a)** engine → private `Red-Weasel/mach-x-inference-engine` (NEW; public Apache-2.0
at P4); **(b)** Seal app → existing private `Red-Weasel/Seal-Team-Application`
(35 commits ahead, scanned clean). Auth via `gh` device login.

Owner-set order AFTER the Seal-app review (do in this sequence):
1. **Gemma 4 QAT models** — Gemma-4-26B-A4B (MoE) + Gemma-4-31B, both QAT-optimized
   (quantization-aware training). New arch family eval — "see what they're about."
2. ~~**Codestral SP-track**~~ — ✅ **DONE 2026-06-13** (classic SentencePiece tokenizer;
   Codestral-22B-v0.1 chats; tokenizer 9/9 byte-exact, PPL 10.89, crown bit-exact). The
   last untested tokenizer family (SentencePiece, scores-based) is now closed. See the
   gate section below + [[project_codestral_spm]].
3. **CUDA + multi-platform** — CUDA is a HARD must-have for NVIDIA before launch
   ("at least say it supports CUDA"). Validation HW: Windows box w/ 8 GB RTX 2080.
   **✅ READINESS AUDIT DONE 2026-06-13** (`docs/cuda_multiplatform_readiness_2026-06-13.md`):
   CUDA is reachable via oneAPI's NVIDIA SYCL backend (`-fsycl-targets=nvptx64-nvidia-cuda`)
   — **not a rewrite** — but **not a free recompile**: the gating work, in effort order,
   is (1) **subgroup width 16→32** (~15 kernels hardcode Intel SG=16 with width-specific
   reductions — the dominant cost; needs SG=32 variants), (2) make **oneDNN** compile-optional
   (`src/CMakeLists.txt:2` is a HARD `find_package(DNNL REQUIRED)`; runtime-opt-in already,
   `gemm_fp16` is the default fallback), (3) generalize device selection (the `name_filter`
   param), (4) add the nvptx64 build target (`IE_SYCL_TARGET` cache var seam already exists;
   sm_75 for the 2080). ESIMD is already off/optional → excluded cleanly. The CMake/target
   seam (items 2-4) is doable on THIS box (~0.5 day, no NVIDIA HW to make the tree *build*
   for CUDA); the SG-width kernel port + bit-correctness validation (~1-2 weeks, dense-only
   far less than crown) needs the RTX box. Honest launch claim after that: "runs on NVIDIA
   via SYCL/CUDA." Also surveyed: Windows-Arc (P3c groundwork), AMD ROCm (same SYCL story,
   amdgcn target), Metal (no SYCL backend, out of scope). **✅ item 2 DONE 2026-06-13
   (commit 704425a): oneDNN made compile-optional** (`option(IE_ENABLE_ONEDNN ON)`; OFF
   swaps `gemm_onednn.cpp`→a `gemm_fp16` stub with the SAME signature so the crown's
   `qwen36.cpp` is UNTOUCHED; verified ON byte-identical + OFF links with zero libdnnl).
   **Corrected 2026-06-13:** the cheap build-config win (item 2, oneDNN) is the only one that
   was both doable AND testable on Intel — done. **ESIMD gating** has macro-coupling with the
   crown's `#if` call site, and **XMX is LOAD-BEARING on the default dense path**
   (`gemm_q4_K_xmx`, not a flag) so it needs a tensor-core path or `gemm_fp16` fallback shim —
   that's part of the kernel port, not trivial build-config. **NEXT belongs on the RTX box**
   (where ESIMD/XMX/SG-width can actually be compiled + validated for nvptx64): wire the
   nvptx64 target, add the XMX fallback shim + ESIMD gate, then the SG 16→32 kernel port +
   bit-correctness battery (item 1). Audit doc has the precise plan.
   **⚠ MULTI-MACHINE BRANCH FLOW (CUDA work happens on the Windows/RTX-2080 box — read this
   FIRST if you are that session):** the engine repo now lives on TWO machines — the Linux box
   (2×B70, the authority for the Intel gate battery) and a Windows box (RTX 2080, 8 GB, sm_75,
   for CUDA bring-up). **CUDA work MUST go on a branch `cuda-windows`, NEVER on `main`.** Reasons:
   (a) the Intel regression gates — crown PPL **6.4527**, ctest **29/29**, `bench_showdown.sh` —
   run only on the Linux box (the Windows box has no Intel GPU and can't verify the Intel path
   stayed byte-identical); (b) the 2080's 8 GB only fits small models, so CUDA can be
   correctness-validated there on a small dense model but the crown can't. **Keep every CUDA
   change default-OFF for non-CUDA builds** so the Intel/Linux `main` build is unaffected.
   **Merge flow:** Windows pushes `cuda-windows` → the Linux box pulls it, runs the full gate
   battery → merges to `main` only if green. The CUDA build is NOT a one-session "finish": first
   session = toolchain (oneAPI DPC++ + Codeplay oneAPI-for-NVIDIA plugin + CUDA Toolkit) →
   nvptx64 configures + compiles → smallest dense (`kLlama3`) model partially runs → SG-width
   breakages discovered/logged. Oracle = a CUDA-built llama.cpp (sm_75) for per-layer cosine.
   **STATUS 2026-06-14: M2 (build-config) DONE on the Windows box — `ie_core` compiles+links for
   nvptx64 sm_75, pushed `origin/cuda-windows` (dc7e7e0, not merged). BLOCKED on Codeplay portal
   down (runtime adapter). M3 SG-port = 78 `reqd_sub_group_size(16)` sites / 12 files, scoped not
   started (needs Codeplay to validate). See the header for full detail. Linux-side action when
   wanted: pull `cuda-windows`, run Intel gates to prove the build-config kept Intel byte-identical
   before any merge.**
4. **1-bit models** — ✅ **INVESTIGATED + SCOPED 2026-06-13**
   (`docs/bitnet_1bit_readiness_2026-06-13.md`). Findings: (a) **TRAP avoided** — the official
   `microsoft/bitnet-b1.58-2B-4T-gguf` is `I2_S` (ggml dtype id 36 = bitnet.cpp *fork-only*;
   mainline ggml + our engine error on it); the right artifact is a **TQ2_0** GGUF converted
   from HF safetensors (`convert_hf_to_gguf.py --outtype tq2_0`), which both our engine and the
   mainline llama.cpp oracle run. (b) **Engine is partially scaffolded** — `kTQ1_0=34`/`kTQ2_0=35`
   dtypes already registered with block sizes matching the oracle exactly, but NO TQ kernels and
   NO `bitnet` arch. (c) **Arch** = LLaMA-like dense (RMSNorm/GQA/NEOX-rope/SwiGLU) + 2 distinctive
   sub-norms/layer (attn_sub_norm/ffn_sub_norm before each ternary proj) + per-tensor weight
   scales → rides ~90% of the DenseModel/`kLlama3` path; new work = TQ2_0 dequant/GEMV kernel +
   sub-norms + scale + arch branch (≈ Gemma-Q4_0-level effort, multi-session). **Recommendation:
   GO as a breadth/credibility item but LOW priority — sequence AFTER #5** (on a B70 the BitNet
   win is memory footprint, not throughput; it's "we support 1-bit," not a perf crown). **Mobile
   deploy = SEPARATE major initiative** (new ARM backend, months) — do NOT let #4 balloon into it.
5. **Perf tuning**, THEN the **benchmark writeup**: one single session running the
   SAME 5 custom benchmarks head-to-head vs llama.cpp SYCL (maybe add a 3rd
   contender), then a publishable/marketing findings write-up. Tagline: *"we're
   running Mach10 on dual B70s."*
   **PREP DONE 2026-06-13: consolidated data skeleton at `docs/public/benchmark_snapshot_2026-06-13.md`**
   (all per-arch numbers in one table, ours vs llama, with claim discipline + the gap list).
   **⚠ The "beats llama.cpp" headline is a MOVING TARGET** — crown took the total crown 2026-06-10
   (+7.6%) but llama.cpp SYCL master then improved (PR #23142 = +70% MoE prefill).
   **✅ RE-MEASURED 2026-06-13 — TOTAL CROWN HOLDS vs current SYCL master:** ran `bench_showdown.sh`
   order-controlled vs llama SYCL built from `fdc3db9b6` (INCLUDES #23142, the +70% MoE prefill that
   threatened the headline). **Prefill: engine ~1047 vs llama ~924–956 → +10–13%. Decode: engine ~73
   vs llama ~70 → +4%. Engine leads BOTH metrics same-hour — the stale-headline risk is RESOLVED;
   llama's MoE-prefill PR did not retake the crown.** Caveat: absolutes were thermally depressed
   (1047/73 + 940/70 vs the 2026-06-10 heat-managed peaks 1144/84; card heat-soaked from back-to-back
   runs) — the RELATIVE standing is valid; the framed #5 run should cool between passes for the
   cleanest absolute headline numbers. **Still TODO for #5:** fully heat-soaked publishable absolutes,
   same-hour perf for the correctness-only arches (Codestral/Gemma-4/Mistral-24B), same-hw llama TP
   run for the 72B, contender-set decision. Safe claim NOW: "beats llama.cpp SYCL master on both
   prefill and decode, same-hour measured 2026-06-13" + "fastest vs llama Vulkan."
   **✅ #5 RELEASE BENCHMARK RUN COMPLETE 2026-06-15** (`docs/public/release_bench_2026-06-15.md`,
   all 6 items): (1) perf FROZEN (defaults are the optimized paths; no risky last-minute tuning
   before measuring). (2) **Crown clean headline (cooled between passes, order-controlled): prefill
   +~8% (1098 vs 1013, peak 1174), decode +~1–3% (83.1 vs 82.2) vs llama SYCL master incl #23142 —
   holds BOTH metrics.** (3) Breadth, HONEST: generic dense BEHIND (Codestral −7%, Mistral −14% —
   dense path not bespoke-optimized), **Gemma-4 ~15× behind (unoptimized MoE; fused-MoE perf never
   done)**. (4) 72B llama-TP BLOCKED (disk 100% full + GGUF gone + re-import RAM-risky → used
   internal 10.4 TP). (5) two llama backends, no viable 3rd (vLLM/OpenVINO lack the crown's DeltaNet
   arch; disk full). (6) publishable findings: **speed crown = the MoE crown ONLY; breadth =
   compatibility, not speed** — frame as "fastest for the crown on Arc + broad-compat for the rest,"
   never "fastest at everything." Post-launch perf levers: dense int-dot/oneDNN + Gemma-4 fused-MoE.
   **Launch article DRAFTED** `docs/public/2026-06-15-mach-x-launch-article.md` (publish-ready
   pending owner: license decision above, repo URL, exact driver string). A punchier marketing
   variant is PENDING — owner wants the Gemma-4 perf fixed first since numbers will update.
   **⚠ NEXT-SESSION PERF TASK #1 — Gemma-4 fused-MoE (diagnosed 2026-06-15):** the ~15× prefill gap
   (ours ~59 vs llama 922 tok/s) is STRUCTURAL, not a tweak. Confirmed via A/B that the existing
   flags (`IE_GEMMA4_NO_GEMM_PROJ` = a correctness variant not perf; `IE_GEMMA4_EXPERT_SYNC` slower)
   give NO free win; default is best. Root cause: gemma4 is **Q4_0 / 128-expert / top-8**, and the
   crown's batched fused int-dot MoE kernels are **Q4_K/Q6_K-only** — so gemma4's Q4_0 experts run
   the **unbatched/unfused** path (per-expert small GEMVs, not token-batched GEMMs). Router is
   already on GPU (NOT the qwen3moe host-router bug). **Fix = a batched-fused Q4_0 MoE path for
   gemma4** (gather tokens-per-expert → batched GEMM, ideally int-dot W4A8), scope ≈ the qwen3moe
   fused-MoE session; PPL-gated + order-controlled A/B. Tool: `ie-gemma4-gen <g> "<prompt>" 1 bench`
   (prints `BENCH prefill ... tok/s`); llama ref `llama-bench -m <g> -p 512 -n 128`.
   **✅ M0 DONE — batched-fp16 fused-MoE (2026-06-15, gated `IE_GEMMA4_FUSED_MOE`, default OFF):**
   replaced the per-token×per-expert `gemv_q4_0` loop (gemma4.cpp:436-451) with expert-sorted
   gather (`build_moe_packing`+`moe_gather_rows`) → ONE batched `gemm_q4_0` per expert for gate_up
   and down → `geglu_rows` (new gemma4-owned kernel, `src/ops/moe_gemma4.cpp`, bit-same gelu-tanh
   over the interleaved [TK,2EF] buffer) → `moe_prefill_reduce` (per-expert `ffn_down.scale` folded
   into the packed routing weight). Reuses qwen3moe's packing/gather/reduce ops; **`moe_fused.cpp`
   UNTOUCHED**. **PERF: prefill T=1262 57.6→79.6 tok/s = 1.38×** (order-controlled, post-JIT). Gates:
   crown PPL **6.4527/NLL 1.864495 bit-exact**; gemma4 decode PPL Δ0.05% NLL, reprefill Δ0.39% NLL
   (better direction; fp16 GEMM-vs-GEMV accumulation order — same "PPL-stable not bit-exact" class as
   the GPU router); prefill top-5 logits identical to 3 decimals. **HONEST: 1.38× still leaves us
   ~11.6× behind llama (~922) — batching alone is modest because `gemm_q4_0` is fp16 dequant-multiply
   bound. M0 is the safe down-payment + the gather/pack/reduce foundation.**
   **✅ M1 DONE — int-dot W4A8 Q4_0 fused-MoE (2026-06-15, default within `IE_GEMMA4_FUSED_MOE`;
   opt-out `IE_GEMMA4_NO_Q8`→M0):** new single-launch kernel `moe_prefill_proj_q4_0_q8`
   (`src/ops/moe_gemma4.cpp`) — `y[TK,N]=q8(x)[TK,K]@Q4_0 W[K,N]` over ALL experts in ONE launch via
   `expert_offsets` (vs M0's per-expert GEMMs); serves BOTH gate_up (K=H,N=2EF) and down (K=EF,N=H),
   `geglu_rows` between; Q4_0 int-dot `acc += d4*(d8*idot − 8*(s0+s1))` (one fp16 scale/32-elem block,
   no sub-scales — simpler than Q4_K), nibble words assembled byte-wise (Q4_0 blocks are 2-aligned);
   activations via `quantize_q8_1s`. `moe_fused.cpp` UNTOUCHED. **PERF: prefill T=1262 57.8→121 tok/s
   = 2.1× (1.41× over M0).** Gates: crown PPL **6.4527/NLL 1.864495 bit-exact**; gemma4 reprefill PPL
   Δ−0.065% NLL (tighter than M0), decode Δ−0.70% (int8 activation; better dir); prefill top-1
   identical, top-5 within ~0.1 logit. Commits 4c7275c (M0) + 42d1390 (M1). **HONEST: 2.1× closes the
   llama gap ~15.6×→~7.6× behind — significant but NOT parity.** Unlike qwen3moe (8.1×, whose dominant
   lever was the host→GPU router = 66% of its prefill), gemma4's router was ALREADY on GPU, so int-dot
   is the ONLY lever → ~2× not 8×. **Gemma stays broad-compat, NOT speed-crown** (article framing
   unchanged). Default-OFF (shifts the documented gemma4 baseline <1%, new shared path) — flip to
   default-ON after a family PPL pass; revisit the T==1 decode int-dot policy then (decode has no perf
   pressure, could stay fp16 for the 0.05% M0 delta). **Optional M2 (more gap):** kernel tuning —
   larger M_TILE via less SLM, SOA repack, reduce MAX_BPL=6 gate_up register pressure. **Article:** the
   punchy crown-led variant is crown-led + cites NO Gemma speed number → unblocked now regardless.

(Engine breadth status: 8 arch families + Granite-3.x dense + Devstral + **Codestral
(classic SPM)** all GREEN; crown PPL 6.4527 bit-exact, tree clean. See the per-family
gate sections below.)

### ✅ CODESTRAL SP-track — classic SentencePiece tokenizer (owner roadmap #2, 2026-06-13)
**The last untested tokenizer family is closed.** The engine could not LOAD a classic-SPM
GGUF (`tokenizer.ggml.model=="llama"`: Llama-1/2, Mistral, Codestral): `Tokenizer::load`
hard-errored without `tokenizer.ggml.merges`, and nothing read `tokenizer.ggml.scores`.
Gemma 4's "SPM-style" path is rank-ordered BPE that STILL needs a merges list — it did
NOT cover this. Classic SPM ships **scores + NO merges** and merges by *token score*.
- **Fix (additive, gated on `model=="llama"`→`spm_`; every BPE/Gemma path byte-identical),
  all in `src/tokenizer/tokenizer.cpp` (+115) + `include/ie/tokenizer.hpp` (+17):** merges
  made optional for SPM; read `scores_` (error if size≠vocab) + `add_space_prefix_`
  (default true); `bpe_merge_spm` = merge the adjacent pair whose MERGED token has the
  highest `scores_[id]` (leftmost on tie ≡ llama.cpp `llm_tokenizer_spm` priority queue),
  `<0xXX>` byte fallback; encode SPM branch (leading dummy ▁ when add_space_prefix, escape
  spaces→U+2581, whole-chunk score-merge, no regex split); decode SPM branch (raw UTF-8
  tokens, `<0xXX>`→byte, U+2581→space).
- **Arch was FREE:** Codestral-22B-v0.1 (`general.architecture=="llama"`, 56L, hidden 6144,
  48q/8kv GQA, ffn 16384, head_dim 128, θ=1e6, vocab 32768) rides the validated DenseModel
  `kLlama3` + Mistral `[INST]` path (same as Mistral-Small-24B). Only the tokenizer was new.
- **Validated:** tokenizer **9/9 byte-exact vs `llama-tokenize`** (BOS + leading-▁ `' The'`,
  code `def reverse(s): return s[::-1]`, café→29113, numbers/floats, tab+newline, 🚀+日本語
  byte-fallback reassembly, leading spaces, camelCase, `<s>[INST] hi [/INST]` special-split);
  `ie run` chat coherent (correct reverse-string code + clean `</s>` stop, no `[INST]`
  leakage); **PPL 10.89** (NLL 2.387521, deterministic; sane for a 22B Q4_K_M code model on
  prose, cf. Llama-3.1-8B 10.79); crown **6.4527 / NLL 1.864495 bit-exact**; **ctest 29/29**.
- **Tool:** `ie-gemma4-tok-test <gguf> <str>...` is already generic (Tokenizer only) — the
  SPM encode dumper for byte-exact diffs. GGUF: `~/models/Codestral-22B-v0.1-GGUF/
  Codestral-22B-v0.1-Q4_K_M.gguf` (13.3 GB, one B70). Plan:
  `docs/superpowers/plans/2026-06-13-codestral-spm-tokenizer.md`.
- **Unlocks every classic-SPM GGUF:** Llama-1/2, Mistral-7B v0.1/v0.2 (pre-tekken),
  Codestral — independent of arch (each rides whatever dense path its
  `general.architecture` selects). Follow-up (optional): a synthetic host ctest for the
  SPM merge (dbrx-style golden ids) — skipped (SPM needs a scored vocab from the GGUF).

### ⏯ Qwen3-Next-80B-A3B (`qwen3next`) — STEP 1 LOADER DONE (2026-06-12)
**The crown family at 80B** (llama.cpp groups `QWEN3NEXT` with `QWEN35MOE`):
gated DeltaNet + gated full-attn (NEOX rope) + MoE (512 experts/10 used, E_ffn
512) + a **shared expert**. Composes from existing additive blocks — `qwen35_dense`
DeltaNet/attn + `qwen3moe` int-dot MoE (E_ffn=512 = the int-dot **base case**) +
the crown's shared-expert pattern + `DenseModelSplit` fleet (46 GB Q4 GGUF does
NOT fit one 32 GB B70 → **must** layer-split across 2×B70 — the one new capability).
- **Model:** `~/models/Qwen3-Next-80B-GGUF/Qwen_Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf`
  (46 GB, bartowski). Oracle: `~/llama.cpp` has `LLM_ARCH_QWEN3NEXT` (+ `SSM_BETA_ALPHA`).
- **DONE + committed:** Task 0 — `kQwen3Next` + `read_qwen3next_config` (commit
  20708d7); full 48-layer tensor-inventory validation + the verified forward spec
  **`docs/qwen3next_80b_dataflow.md`** (`ie-qwen3next-config-test` → "ALL 48 layers
  OK").
- **✅ STEP 1 — FLEET LOADER DONE + GATED (2026-06-12, commits 3ff1f5a→9064e8a).**
  New additive files `include/ie/qwen3next.hpp` + `src/model/qwen3next.cpp`
  (`Qwen3NextModel`) + `tools/qwen3next_load_test.cpp` (`ie-qwen3next-load-test`).
  Layer-parallel pipeline split (24/24 over 2×B70, signed-off architecture review
  `docs/superpowers/specs/2026-06-12-qwen3next-hybrid-split-architecture-review.md`):
  each layer's full weight set (DeltaNet|attn + 512-expert MoE bank + shared expert)
  on its owning card via the existing streaming per-tensor upload (host RAM stays low —
  no 46 GB host buffer), per-card hybrid caches (`DeltaNetState` for the card's linear
  layers + `KvCache` for its attn layers). Composes the validated qwen35_dense
  DeltaNet/attn load + qwen3moe MoE/router load (file-local `upload_weight_auto` →
  `dense::upload_quant_dense_auto`, copy-not-hoist) + crown shared-expert names.
  **GATE: `ie-qwen3next-load-test … 2` → "LOAD OK across 2x B70  layers=48
  experts=10/512  E_ffn=512  card 0: 22.60 GB / card 1: 22.68 GB"** (balanced, fits
  32 GB/card, RAM healthy throughout). Crown PPL **6.4527 / NLL 1.864495 bit-exact**,
  ctest **28/28**. Built/reviewed task-by-task (spec + code-quality per task), zero
  edits to crown / qwen35_dense / qwen3moe / moe_fused / dense_split.
- **Deltas vs the 27B (flagged in the spec ⚠):** fused `ssm_ba` [2048,64]=[β|α] (vs
  separate alpha/beta), repeat 16→32 (n_v=32), **NEOX rope** (not M-RoPE), E_ffn 512,
  + a shared expert. conv_channels=8192, joint Q|gate=8192.
- **✅ STEP 2 ORACLE ESTABLISHED (2026-06-13).** llama.cpp `build-vk` qwen3next runs +
  produces a full per-op dump. **KEY FINDING: llama's Vulkan backend has NO qwen3next
  DeltaNet/SSM ops → it falls back to CPU** (both B70s' VRAM stayed empty the whole run;
  verified via `--list-devices`). It computed on the mmap'd 46 GB file and **stayed
  RAM-safe (26 GB free throughout) because MoE sparsity (10/512 experts/token) keeps the
  per-token working set small** — this is NOT the CPU-mmap pattern that froze the box
  (that was a dense full-weight touch). A complete reference trace (5012 ops, all 48
  layers `blk.0`→`blk.47` → final logits, prompt "The capital of France is") is saved at
  **`docs/qwen3next_oracle_prefill_capital-of-france.log`** for op-by-op diffing.
  Implications: (a) the oracle is **correctness-only** (CPU values = ground truth);
  (b) there is **no fair llama *GPU* perf number** for qwen3next on this box → when our
  engine GPU-runs it, we're the first-mover GPU impl on Intel Arc (story: "runs on GPU
  where llama falls to CPU"), and any perf claim is vs llama-CPU or standalone, not a
  llama-GPU head-to-head. ⚠ Re-confirm a future llama build doesn't add Vulkan qwen3next
  ops before claiming first-mover.
- **✅ STEP 2 FORWARD NaN FIXED (2026-06-13, commit 402786a) — logits now FINITE
  end-to-end across 2×B70.** Full `Qwen3NextModel::forward` (F1–F4: scaffold → gated
  full-attn NEOX → gated DeltaNet fused-ssm_ba/repeat-32 → MoE + shared expert) runs
  with no NaN. **The greedy=NaN root cause was a MISDIAGNOSIS in the prior note (it was
  NOT the int-dot `_gen` down kernel overflowing at E_ffn=512).** The real bug: the
  forward called the **Q6_K** down kernel *unconditionally*, but the Q4_K_M GGUF **mixes
  ffn_down precision per layer** — `ie-inspect` shows `ffn_down_exps` = Q6_K on
  blk.0–5/8/11/14/…/41–47 and **Q4_K on the rest**. From blk.6 (first Q4_K) on, the Q6_K
  kernel misread 144-byte Q4_K blocks as 210-byte Q6_K → garbage super-block scales →
  `moe_out` maxabs 65504/Inf → NaN. Localised with `IE_QWEN3NEXT_DBG`: `x(out)` finite
  L0–L5 (maxabs 10.5), cliff to NaN at L6; both the fp16 AND int-dot Q6_K kernels failed
  identically at L6 (∴ never the kernel), while `gate_up` (always Q4_K, correctly
  dispatched) was always finite. **FIX:** branch the prefill+decode down calls on
  `lw.down_dt` (Q6_K vs Q4_K), mirroring `qwen3moe.cpp`; the loader already captured
  `down_dt` per layer (only the forward ignored it). Default = crown's proven
  fp16-activation packed down; int-dot W4A8 opt-in via `IE_QWEN3NEXT_Q8=1`. Crown PPL
  6.4527/NLL 1.864495 bit-exact; isolated to `qwen3next.cpp` (its own TU).
- **✅ STEP 2 FORWARD CORRECT — greedy = " Paris. The capital of Italy is Rome" (2026-06-13).**
  Qwen3-Next-80B runs correctly end-to-end across 2×B70. FOUR forward bugs found+fixed by
  op-by-op VALUE diff vs `docs/qwen3next_oracle_prefill_capital-of-france.log` (probes
  `IE_QWEN3NEXT_DBG`/`_DBG_VALS`/route-dump + `ie-inspect --layer`): (1) down `down_dt`
  dispatch — GGUF mixes ffn_down Q6_K/Q4_K per layer (402786a); (2) `ssm_ba` per-k-head
  split `{4,SKH=16}` β=[0:2]/α=[2:4] (d1afa7a); (3) DeltaNet q/k repeat INTERLEAVE not tile
  (e7319f7); (4) **`moe_y` not zeroed before `moe_prefill_reduce`** (it accumulates;
  qwen3moe pre-writes y via shexp, qwen3next adds shexp after → reduce read stale y → L1+
  MoE ~2.66× inflated) — memset fix (1ed07e0). Verified: L0 x(out) oracle-bit-exact; L1
  moe_y 1.180≈1.167; routing top-10 matches `ffn_moe_topk-1`. Crown PPL 6.4527/NLL 1.864495
  bit-exact; all isolated to qwen3next.cpp (its own TU). **VALIDATED (2026-06-13):** cosine/sum
  sweep vs oracle — all 48 layers' residual within 0.2–1.6% (ratio 0.998–1.007; L47 ratio 5.4
  = llama's known last-token-only artifact, not a bug); **PPL 4.7282** (avg NLL 1.553554, 256
  tok) via new `ie-qwen3next-ppl` (streaming T=1, commit 022b26f) — healthy/low. **✅ WIRED INTO
  THE ENGINE (commit 1de78bd):** `echo "What is the capital of France?" | ie run <gguf> --gpus 2`
  → "The capital of France is Paris." Additive Engine path (`next_model_` + `next_` flag, mirrors
  the TP host-logits bounce; `kQwen3Next` load branch → fleet.init → LayerPlan::contiguous →
  `next_model_.load(…, opts.max_ctx)`; ChatML/`<|im_end|>` fall-through). Env-gated probes
  STRIPPED. Gates: build green, ctest 28/28, crown 6.4527 bit-exact. **✅ PERF + int-dot DEFAULT
  (commit 1942223):** int-dot W4A8 down validated correct & made default (opt-out
  `IE_QWEN3NEXT_NO_Q8`) — pp512 224→566 tok/s = **2.5×** vs fp16. **HEAD-TO-HEAD vs llama.cpp
  SYCL** (built `-DGGML_SYCL=ON` icpx 2026.0 @ fdc3db9b6 — its SYCL backend HAS the
  gated_delta_net/ssm_conv/ssm_scan GPU ops, so it runs qwen3next ON GPU, unlike Vulkan→CPU;
  same 46 GB GGUF, both 2×B70 `-sm layer`, same session): **DECODE tg128 ours 51.8 vs llama
  37.1 tok/s = 1.40× FASTER**; **PREFILL** ours ahead/parity ≤256 tok (208t 513 vs llama pp256
  474), ~10% behind at pp512 (566 chunked vs 622) — our DeltaNet T≤256 prefill chunking caps
  long-prefill throughput. 5-prompt suite (`ie-qwen3next-bench`, same protocol as ie-bench-suite):
  median pp 336 / tg 51.8 tok/s across short-chat/long-instruction/codegen/math/long-context.
  **✅ PREFILL CAP RELAXED 256→512 (2026-06-14):** qwen3next is empirically CLEAN at T=512 —
  `ie-qwen3next-ppl --sweep --repeats 25` (651-tok diverse corpus, 2×B70) = **25/25 bit-identical
  PPL at chunk-256 AND chunk-512, no collapse / no run-to-run divergence** (each 512-pass runs the
  recurrence 512 steps × 24 linear layers/card). A 512 single-call prefill is **1.08–1.15× faster**
  than 2×256 on >256-tok prompts (`ie-qwen3next-bench --ab`: 512-tok 543→617, 768-tok 536→578,
  1024-tok 532→608 tok/s) — **617 ≈ llama SYCL's pp512 622, closing that gap** — at zero decode/
  short-prompt cost. Baked into `Engine::generate` gated on `next_` (512, env-override
  `IE_QWEN3NEXT_PREFILL_CHUNK`, capped at max_ctx); crown/27B/dense stay at 256. T≥1024 NOT yet
  validated (corpus too short). Crown PPL **6.4527 bit-exact** (change is next_-gated). Note in
  docs/known_bugs.md §1. **NEXT:** `ie serve` smoke; longer-corpus sweep if T>512 is wanted.
- **(history) THREE-bugs checkpoint — L0 oracle-exact before bug #4:**
  Op-by-op value diff vs the oracle (probes `IE_QWEN3NEXT_DBG=<L>` + `IE_QWEN3NEXT_DBG_VALS=1`)
  found+fixed THREE bugs: (1) down `down_dt` dispatch (402786a), (2) `ssm_ba` per-k-head
  split (d1afa7a), (3) **DeltaNet q/k repeat must INTERLEAVE not tile (e7319f7)** — `repeat_interleave_heads`
  did `h_in=h_out%n_in` (tile) but qwen3next's ggml REPEAT is `h_in=h_out/repeat` (k-head h
  → v-heads rep·h..); added a `bool interleave` param (default tile = 27B-safe), qwen3next=true.
  **VERIFIED: L0 `x(out)` now matches the oracle BIT-FOR-BIT** ([0.0201,−0.0252,0.0013] =
  `l_out-0`); dn_out 29.377≈29.379, attn_block −8.164≈−8.163, L0 MoE 2.709≈`ffn_moe_out-0` 2.722.
  **BUG #4 (greedy still wrong): the L1-onward MoE EXPERTS output is ~2.66× too big** (L1
  moe_y experts 3.111 vs `ffn_moe_out-1` 1.167) while **L0 MoE is exact, and L1's DeltaNet +
  shared-expert + router-renorm all VERIFIED correct** (L1 dn_out 1.573≈1.572, gated-shexp
  −0.524 both, oracle renorm confirmed = mine). Router logits match in SUM but differ ~0.46/elem
  → prime suspect = **expert top-10 SELECTION diverging at L1** (logit noise flips borderline
  experts; harmless at L0's routing, not at L1). **NEXT:** dump L1 top-10 expert indices+weights
  vs oracle `ffn_moe_topk-1`/`ffn_moe_weights-1`; if selection differs, chase the router-logit
  ~0.46 error (x_normed(moe-in) per-value vs `attn_post_norm-1`, or router_w precision). Then
  greedy " Paris" → PPL → wire in. Crown 6.4527 bit-exact; all isolated to qwen3next.cpp.
- **(superseded) ssm_ba SPLIT FIXED (commit d1afa7a):**
  Op-by-op sum diff vs the oracle (`IE_QWEN3NEXT_DBG` + new `IE_QWEN3NEXT_DBG_VALS=1`
  first6/last6 previews) localised a real bug: the **fused `ssm_ba`[64] split was
  contiguous (β=[0,32)/α=[32,64)) but must be PER-K-HEAD INTERLEAVED** — oracle reshapes
  to `{4, SKH=16}` and views β=[0:2]/α=[2:4] within each group (layout `[T,SKH,2,SVH/SKH]`,
  β FIRST). Fixed via `split_q_gate_per_head(SKH, SVH/SKH)`. **VERIFIED:** `dn_g` now
  matches the oracle to 4 figs (−5622.24 vs gate-0 −5622.515; was −8931), dn_alpha/β =
  118/367 = oracle a/b. **The prior "NEOX-rope attn over-contribution" suspicion was
  REFUTED** — k(rope) matches the oracle (121.2 vs 121.8); rope is fine. **WHAT REMAINS:**
  the DeltaNet residual still diverges in DIRECTION by L3 (x_normed values ≠ oracle;
  L0 `x(out)` already sign-flips an element; `dn_out(recur)` 27.3 vs oracle attn_output
  29.4 = ~7%). Suspect the **gated delta-rule recurrence / q-k repeat(16→32) / gated-norm
  / ssm_out** path — needs VALUE-level (not sum) diff of L0 dn_qrep/dn_krep→dn_out→dn_gn→
  attn_block vs the oracle. **NEXT:** find that op → greedy " Paris" → PPL → wire into
  Engine. Then re-test int-dot down (`IE_QWEN3NEXT_Q8=1`; `_gen` was never broken — it
  was fed Q4_K data, now dtype-dispatched) for ~3.3× down. Probes env-gated (default OFF).
- **NEXT — STEP 2 (forward), after the down-kernel fix:** diff op-by-op vs the saved
  oracle trace → per-op cosine ≥0.999 → greedy/PPL; then wire the split forward into
  ie-perplexity + Engine + ie-bench. NB the forward must re-init each card's `KvCache`
  at the production `max_ctx` (the loader's gate value is 2048). Compose the per-layer
  forward from `qwen35_dense.cpp` (DeltaNet/gated-attn, with the ⚠ deltas below) +
  `qwen3moe.cpp` (GPU router + int-dot `_gen` MoE) + crown shared-expert, with the one
  inter-card residual `copy_across` at the 24/24 boundary. Read
  `docs/qwen3next_80b_dataflow.md` + the architecture review FIRST. Memory [[project_qwen3moe]] tail.
- **⚠ Disk-care:** the crown PPL gate `~/.seal/…Q4_K_M.gguf` is a SYMLINK into
  `~/models/lmstudio-community/Qwen3.6-35B-A3B-GGUF/` — NOT a deletable dup
  ([[reference_crown_gguf_symlink]]; broke + restored 2026-06-12).


### ✅ qwen3moe TIER-3 PREFILL DONE — 8.1× (GPU router + generalized int-dot down kernel) (2026-06-12)
**The two prefill levers are BUILT and committed; Qwen3-Coder-30B prefill went
80.6 → 651 tok/s (8.1×), from 12× behind llama.cpp Vulkan (~984) to ~1.5×.**
Both additive + qwen3moe-owned (crown `qwen36.cpp` / `moe_fused.cpp` UNTOUCHED).
1. **GPU-gemm router (the hidden lever the Tier-2 profile MISSED).** The
   `--kprofile` "routing 0.3%" was GPU-kernel-only; the real qwen3moe prefill
   bottleneck was the **single-threaded host router** (`route_token`: E=128 ×
   H=2048 half→float dot per token × 512 × 48 layers) = **~4859 ms = ~66% of the
   6354 ms pp512 wall**, invisible to kprofile (GPU total was 2501 ms). Fix:
   upload the F32 router weight transposed to device F16 `[H,E]` at load; logits =
   `ws_xn[T,H] @ router_w_devᵀ` run as ONE `gemv_q_T`/`gemm_fp16` (N=E=128%64==0,
   K=H=2048%256==0), copy `[T,E]` to host, keep only the cheap softmax/top-k on
   host. **pp512 59→196 (3.3×), PPL 11.98 bit-stable.** Opt-out `IE_QWEN3MOE_HOST_ROUTER=1`.
2. **Generalized int-dot W4A8 down kernel** (`src/ops/moe_qwen3.cpp`, NEW file).
   The crown's int-dot down (`moe_fused.cpp down_packed_q{4,6}k_q8`) is hard-locked
   to `E_ffn==512` (one SG = 16 lanes × 1 q8 block × 32 = 512 covers full K).
   `moe_prefill_down_q{6,4}k_q8_gen` generalize it to any `E_ffn%256==0`: each lane
   walks q8 blocks `{lane, lane+16, …} < E_ffn/32` (b_in=j/8, sb=j%8), SG-reduce
   sums over full K. Bit-identical to the crown kernel at E_ffn=512. Plus wired the
   already-E_ffn-parameterized int-dot gate/up. **kprofile: down_q8_6k 1720→190 ms
   (9×), down_q8_4k 354→149 ms, gate 293→253 ms; GPU total 2501→747 ms.**
   Order-controlled warm A/B: **pp512 int-dot 651±5 vs fp16 197±2 = 3.3×.**
- **Gates:** qwen3moe PPL **11.98 / NLL 2.4834 bit-stable** (== fp16 path =
  authoritative correctness); Coder greedy coherent (" Paris. …Germany is Berlin.");
  crown **6.45 / NLL 1.864495 bit-exact**; ctest **28/28**. Default ON; opt-out
  `IE_QWEN3MOE_NO_Q8=1`. Commits this session (2 milestones).
- **Remaining qwen3moe levers:** gate_q8 is now the #1 kernel (34%, ~253 ms — a
  generalized faster gate is a possible follow-up but near-shape-optimal); the
  bigger picture is **decode** (separate documented lever) and **qwen3moe
  multi-GPU → Qwen3-Next-80B**. The two int-dot `_gen` down kernels also unlock
  Next-80B's MoE shapes. Ledger: `docs/benchmark_matrix_2026-06-09.md` §qwen3moe.
- **✅ DECODE int-dot — DONE, +~28% (2026-06-14, task #6).** Fresh decode profile
  (Coder-30B, ctx512): 38.8 tok/s; FA-2 decode already exists (the old "71% naive attn"
  note was STALE); MoE ~36% used the crown fp16-activation `moe_decode_*`. **Fix: route
  T==1 decode through `moe_ffn_fused_prefill`** (int-dot `_gen`, pack 1 token's K experts
  as K rows) — the crown decode kernels were tuned for top-2; the fused `_gen` kernels
  pack top-8 (E_ffn=768) far better. **+~28% (37.5→48 tok/s, heat-robust F/D/F/D), PPL
  bit-stable, coherent ("…Paris.").** Default ON; opt-out `IE_QWEN3MOE_FP16_DECODE=1`.
  **The fix that unlocked it: a T==1-only `memset(ws_ffn_y_)` before `moe_prefill_reduce`**
  (it ACCUMULATES; without zeroing, the decode route added onto stale ws_ffn_y_ ≈5.37 vs
  correct ≈-0.017 → residual explosion → garbage/NaN). Found by op-by-op host probes
  (all stages finite but the MoE output was 300× too large vs the fp16 path). T>1 prefill
  LEFT UNTOUCHED (validated path: PPL 11.98, fused==unfused parity, crown bit-exact).
  ⚠⚠ **LESSON (banked): `ie-bench` only TIMES forwards — it does NOT validate output;** the
  pre-fix route timed +28% on NaN garbage and only the PPL/chat gate caught it (ie-perplexity
  runs qwen3moe in T>1 chunks so it never exercised the T==1 path). **qwen3next: MEASURED, NOT
  applied** — the same T==1→fused route is +CORRECT (PPL 4.7255≈4.7282) but **~40% SLOWER**
  for qwen3next (decode 30 vs 50 tok/s): the 80B is split across 2 CARDS, so the fused path's
  per-layer host packing + gather + cross-card coordination dominates at T==1 (single-card
  qwen3moe wins, 2-card qwen3next loses). qwen3next keeps fp16 decode (already 1.40× ahead of
  llama at 50 tok/s); finding documented in `qwen3next.cpp`. Decode bottleneck #2 = `fa2_partial`
  22% (launch-bound, shared kernel — needs graph/persistent infra).
  **Dense + 27B decode are Q6_K-ffn_down ALU-bound** (140 GB/s cliff): the int-dot
  ktiled already regressed (75 GB/s), and the fast Q4_K GEMV runs the SAME 128 WGs at
  321 GB/s → NOT occupancy-limited → split-K won't help; **a Q6_K weight REPACK is the
  lever**. **✅ PROTOTYPE-GATE PASSED 2026-06-14 (`ie-q8-0-repack-bench`): repack Q6_K
  ffn_down → SoA Q8_0 (int8 qs col-major + fp16 d/32-block) + new `gemv_q8_0_soa_q8`
  (int8 dp4a, no 6-bit unpack) = 542 vs 141 GB/s = 2.96× on the real shape (K=12288,
  N=4096), AND MORE accurate (rel_rms 0.0075 vs Q6_K 0.0178 — 8-bit > 6-bit). Costs +30%
  weight bytes (41→53 MB/layer). **✅ INTEGRATED + 8B-VALIDATED 2026-06-14, OPT-IN
  (`IE_DENSE_Q6K_REPACK=1`):** `DenseModel::load` repacks a Q6_K ffn_down → SoA Q8_0
  (kept alongside the Q6_K, which prefill still uses); the T==1 decode routes through
  `gemv_q8_0_soa_q8`. **Qwen3-8B: decode 44.8 → 53.4 tok/s = +19.2%**, PPL NLL 2.9405 →
  2.9443 (+0.37% — the Q8_0 WEIGHT is more accurate but the decode now int8-quantizes the
  ffn_down ACTIVATION, which the Q6_K-scalar path kept fp16; consistent with the engine's
  existing Q8-decode philosophy). DEFAULT OFF because it shifts a documented bit-stable
  dense baseline + touches the shared dense path (all dense models) — flip to default-ON
  after a family-wide PPL pass. **Crown gate RE-CONFIRMED 2026-06-14 after the crown GGUF
  was deleted+re-downloaded: ie-perplexity = 6.4527 / NLL 1.864495 bit-exact (this change is
  dense-only/crown-isolated). Disk now ~7 GB free.**
  **✅ FAMILY ROLLOUT VALIDATED 2026-06-15 — PPL-stable across ALL dense families; recommend
  default-ON (pending owner nod).** PPL A/B (`ie-perplexity --gguf <m> --max-tokens 256`, default vs
  `IE_DENSE_Q6K_REPACK=1`) — every model triggered the repack (non-zero Δ ⇒ Q6_K ffn_down present)
  and held within ±0.22% NLL: Llama-3.1-8B +0.221%, Llama-3.2-3B +0.081%, Granite-3.3-8B −0.152%,
  Phi-4 +0.121%, Mistral-Small-24B −0.059%, Codestral-22B −0.136% (+ qwen3-8b baseline +0.13% NLL /
  +0.37% PPL, prior). THREE improved, all tiny — decisively PPL-stable. **VRAM fit confirmed to 24B**
  (both 22-24B Q4_K_M + the ffn_down Q8_0 copy load fine on one 32 GB B70; the copy adds ~+1-2 GB on
  ffn_down only, kept alongside the Q6_K prefill uses). **Decode gain transfers + scales with FFN
  size:** Qwen3-8B +19.2% (documented) → Phi-4 ~2.9× (256 streaming-T1 decode 18.15→6.24 s, crude
  t264−t8 isolation; Phi-4's ffn=17920 makes the Q6_K scalar cliff dominate decode, so the 542 GB/s
  Q8_0 GEMV win is amplified). Crown UNAFFECTED (MoE, not the DenseModel path); multi-GPU TP/split
  paths unaffected (separate classes). **DEFAULT-ON DECISION (owner):** evidence is uniform + strong
  → recommend flipping `IE_DENSE_Q6K_REPACK` to default-ON with opt-out. Caveat = it shifts the
  documented bit-exact dense baselines (e.g. qwen3-8b NLL 2.940491→2.9443) referenced in
  `docs/ppl_baseline_matrix.md` / dense parity scripts, so the flip must come WITH a baseline-doc
  refresh (deliberate, not silent). 27B (qwen35 hybrid) uses its own model class, not DenseModel →
  separate follow-up if wanted.**

<details><summary>Tier-2 (historical, 2026-06-12) — oneDNN A/B that led here</summary>

### ⏯ qwen3moe TIER-2 PREFILL: oneDNN is NOT the lever — Q6_K expert GEMM is (2026-06-12)
**Profiled + A/B'd qwen3moe prefill. oneDNN (the 27B's 1.65× lever) gives NO
qwen3moe win, and the profile says why.** pp512 warm kernel profile: **84% of
prefill is the MoE *expert* GEMMs** — `moe_pfl_down_pk6k` (Q6_K expert
down-proj) alone is **69.3%**, `_pk4k` 14.8%, `_gate_q4k` 12.0%; attention is
only **3.1%** and host-routing pack/scatter **0.3%**. `dense::prefer_onednn()`
only routes `gemv_q_T` (attn q/k/v/o + lm_head = <5% of prefill) and never
touches the crown-owned fused expert path (`moe_fused.cpp`), so it has nothing to
bite on (qwen3moe H=2048 vs the 27B's 5120 — the attn GEMMs are tiny).
- **pp512 warm plateau:** in-house gemm **~88** vs oneDNN **~89–92** (±noise) vs
  **llama Vulkan 939** → still **~10.4×** behind. This is a **GPU KERNEL gap, not
  swap**: both engines run the VRAM-resident 18.6 GB model on the same B70 (32 GB
  VRAM), and 96% of qwen3moe prefill is in GPU MoE-GEMM kernels (down 69% + 15% +
  gate 12%, routing 0.3%). The crown beats llama via a bespoke int-dot Q8 MoE
  kernel that hard-requires `E_ffn==512`; Qwen3-Coder's `E_ffn=768` falls back to
  the generic fp16 MoE path, and top-8 (vs crown top-2) is ~4× the expert work →
  the 10× is entirely engine-side kernel efficiency. (The old "16×" was the
  cold/first-forward number, not a RAM-swap effect.)
- **Decision:** oneDNN **not enabled by default** — kept behind opt-in
  `IE_QWEN3MOE_ONEDNN=1` (reproducible A/B); default path bit-identical to prior
  commit. **Part B (device routing) DEFERRED** — host routing is 0.3% of prefill,
  zero to gain.
- **Gates:** build 0-err; qwen3moe PPL **11.98** / NLL 2.483450 (held, both
  paths); crown **6.45 / NLL 1.864495 bit-exact** (no crown files touched).
- **Real next lever (Tier-3, multi-session):** a **tuned int-dot MoE-expert
  kernel** generalizing the crown's int-dot Q8 win to `E_ffn≠512` / `top-k≠2`
  (additive, qwen3moe-owned; SoA repack / split-K Q6_K down-proj), ESIMD-safe
  (1D block reads only) — `moe_fused.cpp` is FORBIDDEN, so it must be a new
  file/op, not an edit. It closes the ~10× prefill gap AND unlocks
  Qwen3-Next-80B. OR qwen3moe multi-GPU → Qwen3-Next-80B.
  Ledger: `docs/benchmark_matrix_2026-06-09.md` §qwen3moe Tier-2.

</details>

### ⏯ WAVE-1 GATE 1/9: Mistral-Small-3.2-24B VALIDATED (2026-06-12)
**First per-family GPU validation gate is GREEN — the tekken + `[INST]` +
DenseModel cluster path is proven end-to-end.** Read/validate task, **zero
engine source edits**, crown re-confirmed **6.45 / NLL 1.864495 bit-exact**.
Model: `~/models/lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-GGUF/
...Q4_K_M.gguf` (14.3 GB, single B70; mmproj vision tower out of scope).
- **Arch route:** GGUF `general.architecture=llama` → **kLlama3** (DenseModel:
  RMSNorm+SwiGLU+GQA 32/8+NEOX rope θ1e9, no QK-norm/bias). 40L, hidden 5120,
  ffn 32768, vocab 131072, ctx_train 131072.
- **SWA guard: CLEAN** (predicted host-side first) — **no `attention.sliding_window`
  KV** in the GGUF (v3.2 dropped global SWA); the `model_config.cpp` load-guard
  correctly does NOT fire. (Had it fired we'd have reported BLOCKED + needed
  windowed attention — it didn't.)
- **Tokenizer (tekken, `pre=tekken`):** engine encode ids **identical** to
  `llama-tokenize` on the prose+code corpus (incl. BOS=1 prepend + multibyte
  `café`→35858). End-to-end confirm of the committed tekken path.
- **Per-layer cosine vs llama.cpp oracle (`fdc3db9b6`):** min **0.99954** over
  L01–L39 (~1.0; shared validated Llama-3.x un-permute/rope path). L40/L41
  "DIVERGED" is the known final-norm/lm_head **dump-shape artifact** (engine
  dumps all 6 tokens, oracle only the last → tail-misalign), NOT a forward bug.
- **PPL:** **7.42** / NLL 2.004174, **bit-exact deterministic** across 2 runs
  (full-causal builtin corpus; sane for a strong 24B Q4_K_M).
- **Chat (`[INST]`):** "Write a Python function to reverse a string" → coherent,
  4 idiomatic variants, renders `[INST] {user} [/INST]`, stops clean at `</s>`
  (no free-continuation). `chat_template` `[INST]` → kMistral → `build_mistral_prompt`.
- **UNLOCKS the Devstral / Codestral / Mistral-Nemo cluster** (same tekken +
  `[INST]` + kLlama3 path). Ledger: `docs/benchmark_matrix_2026-06-09.md`
  §Wave-1. Next Wave-1 gates: R1-Distill ✅ (both qwen2 + llama paths), Phi-4
  ✅VALIDATED (fused-split + Q5_K dense + dbrx — see Gate 3/9; Phi-3.5-mini still
  gated on LongRoPE+hd96), Yi/Nemotron/InternLM/Baichuan
  ✅validate-on-demand (free llama-clone tier — same kLlama3 path), Granite
  (`docs/superpowers/plans/2026-06-12-wave1-breadth-blitz.md`).

### ⏯ WAVE-1 GATE 2/9: DeepSeek-R1-Distill-Qwen-1.5B VALIDATED (2026-06-12)
**Second per-family gate is GREEN — the DeepSeek-R1-Distill `<｜User｜>`/
`<｜Assistant｜>` sentinel template (`kDeepSeek` → `build_deepseek_prompt`, wired
in `c5b12ca`, NEVER tested on a real GGUF until now) is proven end-to-end +
`<think>` reasoning stays ON.** Read/validate task, **one harness-only edit**
(wave1_gate ANSI strip), crown re-confirmed **6.45 / NLL 1.864495 bit-exact**.
Model: `~/models/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/...Q4_K_M.gguf` (1.1 GB,
single B70, kept for re-tests). Oracle: llama.cpp `build-vk`.
- **Arch route:** GGUF `general.architecture=qwen2` → **kQwen3Dense** (the
  already-validated DenseModel path — it's a Qwen2 fine-tune). 28L, hidden 1536,
  ffn 8960, 12q/2kv, head_dim 128, vocab 151936, θ=10000, eps 1e-6.
- **Template family: kDeepSeek** — `classify_template_family` on the GGUF's
  actual `chat_template` returns **kDeepSeek** (verified via standalone link
  against `libie_core.a`, not just a grep). Template uses the `<｜User｜>`/
  `<｜Assistant｜>` sentinels (NOT ChatML, no `<|im_start|>`). No `sliding_window`.
- **Tokenizer (qwen2 BPE):** engine encode ids **identical** to `llama-tokenize`
  on the prose+code corpus (incl. BOS 151646 + multibyte `café`). The DeepSeek
  special tokens are **single ids**: `<｜User｜>`=151644, `<｜Assistant｜>`=151645,
  `<｜end▁of▁sentence｜>`=151643 (= GGUF eos_token_id).
- **`build_deepseek_prompt` render (standalone-verified):** `enable_thinking=true`
  → `<｜User｜>{u}<｜Assistant｜><think>\n` (think block OPEN — reasoning ON);
  `enable_thinking=false` → injects the empty-think convention
  `<think>\n\n</think>\n\n` (direct answer). NOTE the `<think>` gate here is the
  dedicated DeepSeek path's own `enable_thinking` flag, NOT the ChatML
  `model_has_think` template-string gate from `8c568a6` (the template carries no
  literal `<think>`, but the path injects it anyway — correct for R1-Distill).
- **Chat smoke END-TO-END (the key DeepSeek-template test):** "What is 12*13?
  Think step by step." via `ie serve` with `enable_thinking=true` → emits the
  reasoning trace (the prompt opened `<think>`, model reasons then closes it),
  answers **156** (`\boxed{156}`), **`finish_reason: stop`** at
  `<｜end▁of▁sentence｜>` (264 tok, no length-cap, no free-continuation).
  `ie run` (hardcodes `enable_thinking=false`) gives the right answer 156 with the
  empty-think convention — also correct.
- **PPL:** **81.24** / NLL 4.397384, finite/deterministic (high but expected for a
  1.5B Q4_K_M distill — property gate, not bit-exact).
- **Per-layer cosine vs llama oracle:** min cos **0.999404** over the proper-shaped
  layers (~1.0; argmax `ĠParis` correct). diff_layers' `**DIVERGED**` labels are
  rel_fro≥0.05 (Q4_K-vs-oracle quant gap) + the L28 final-norm dump-shape
  tail-artifact, NOT a forward bug (validated qwen2 dense path).
- **R1-Distill-Llama is the SAME template on the llama path** (`general.architecture
  =llama` → kLlama3, same `<｜Assistant｜>` sentinels → kDeepSeek) — this gate proves
  the whole R1-Distill family's template + special-token + `<think>` handling.
  Ledger row: `docs/benchmark_matrix_2026-06-09.md` §Wave-1.

### ⏯ WAVE-1 GATE: DeepSeek-R1-Distill-Llama-8B VALIDATED (2026-06-12)
**The DeepSeek-R1-Distill template is now proven on BOTH paths — qwen2 (Gate 2/9)
AND llama (this gate).** R1-Distill-Llama-8B is a plain Llama-3 fine-tune →
`general.architecture=llama` → **kLlama3** (the validated DenseModel path: Q/K
un-permute + `rope_freqs` + llama-bpe), carrying the same `<｜User｜>`/
`<｜Assistant｜>` sentinel `chat_template` → **kDeepSeek** → `build_deepseek_prompt`.
**Zero engine edits, zero harness edits** (read/validate only). Crown re-confirmed
**6.4527 / NLL 1.864495 bit-exact**. Model:
`~/models/DeepSeek-R1-Distill-Llama-8B-GGUF/...Q4_K_M.gguf` (4.6 GB, single B70).
Oracle: llama.cpp `build-vk`.
- **Arch route:** `general.architecture=llama` → **kLlama3**. 32L, hidden 4096,
  ffn 14336, 32q/8kv, head_dim 128, vocab 128256, θ=500000, eps 1e-5.
  **No `sliding_window`** (plain Llama-3, not a sliding-attn variant).
- **Template family: kDeepSeek on the LLAMA path** — the GGUF chat_template carries
  the `<｜Assistant｜>` sentinel → `classify_template_family` → **kDeepSeek**. Crucially
  `engine.cpp::chat` dispatches the kDeepSeek branch BEFORE the kLlama3 branch, so a
  llama-arch model with the DeepSeek template takes `build_deepseek_prompt`, NOT
  `build_llama3_prompt`. This is the key new-path proof (Gate 2/9 only exercised it
  on qwen2).
- **Tokenizer (llama-bpe, NOT qwen2):** `tokenizer.ggml.pre=llama-bpe`; engine encode
  ids **byte-identical** to `llama-tokenize` (`128000 9906 1917 13 711 282 2120 1680
  471 865 9 17 220 674 220 4513 19 53050`, incl. BOS 128000 + multibyte `café`). DeepSeek
  specials are **single ids**: `<｜User｜>`=128011, `<｜Assistant｜>`=128012,
  `<｜end▁of▁sentence｜>`=128001 (= GGUF eos), `<｜begin▁of▁sentence｜>`=128000 (bos),
  `<think>`=128013, `</think>`=128014.
- **`<think>` / stop:** same `enable_thinking` flag as the qwen2 gate (path-injected,
  not template-string gated). `ie run` hardcodes `enable_thinking=false` → empty-think
  convention `<think>\n\n</think>\n\n` → direct answer. Stops at `<｜end▁of▁sentence｜>`
  (128001).
- **Chat smoke END-TO-END:** "What is 12*13? Think step by step." via `ie run` →
  coherent FOIL derivation → **156** (`\boxed{156}`), stops clean at eos (prompt
  returns, no free-continuation). Renders the DeepSeek template on the llama path.
- **PPL:** **25.66** / NLL 3.244969, finite + **bit-exact deterministic** across 2
  runs (high but expected for an 8B Q4_K_M distill on the short builtin corpus —
  property gate).
- **Per-layer cosine vs llama oracle:** L01–L31 all **≥0.99992**; only L32 reads
  0.998990 — the final-norm/logits **dump-shape tail artifact** (llama.cpp slices it
  to last-token via `inp_out_ids`, so the compare crosses 24576-vs-4096 widths;
  diff_layers' own verdict is WARN rel_fro 0.045, NOT DIVERGED). Forward is correct.
- **UNLOCKS the free llama-clone tier — VALIDATE-ON-DEMAND:** Yi, Nemotron, InternLM,
  Baichuan are all `general.architecture=llama` Llama-3 derivatives that ride this
  exact validated kLlama3 path. No new engine code expected; validate a specific GGUF
  only when first needed. Ledger row: `docs/benchmark_matrix_2026-06-09.md` §Wave-1.

### ✅ WAVE-1 GATE 3/9: Microsoft Phi family — **Phi-4 VALIDATED** / Phi-3.5-mini scoped (updated 2026-06-12)
**Phi-4 (`phi3` arch) now loads, forwards bit-true, and chats — the `phi3` family
is UNBLOCKED for the Phi-4 tier.** Three additive, presence/dtype/`pre`-gated levers
landed (commits `aaab17b`, `aa37fc9`, `27eaa24`, `e917e6d`):
- **(1) Q5_K/Q8_0 dense GEMV** (`dense_dispatch.hpp` `upload_quant_dense_auto` →
  `dequant_q5_K_to_Bt`) — a **reusable lever that unblocks ANY Q5_K dense GGUF**,
  not just Phi; Q4_K/Q6_K/F16 default path byte-identical.
- **(2) Fused-tensor splitter at load** (`dense_dispatch.hpp` + `dense_transformer.cpp`):
  slices fused `attn_qkv`→Q/K/V and `ffn_up`→gate/up row-spans (contiguous
  superblock byte-spans, no requant; NEOX natural order; Q5_K Q/K slice via dequant).
  Presence-gated (`attn_q` absent + `attn_qkv` present; `ffn_up.shape[1]==2·ffn`) →
  every existing dense GGUF untouched.
- **(3) `pre=dbrx` tokenizer flag** (`tokenizer.cpp`): dbrx (cl100k/gpt2) pretokenizes
  identically to llama-bpe (digits_1to3 + ignore_merges) — a flag fold, not a new
  `pretokenize_*`. Host **encode-parity test** vs `llama-tokenize` golden ids
  (`tests/unit/dbrx_pretok_test.cpp`, 10 golden + 4 round-trip) PASSES; verified it
  FAILS without the flag (single-digit split).
- **(4) ChatML stop set** (`engine.cpp`): `tf==kChatML && arch!=kQwen3Dense` →
  `<|im_end|>`+`<|endoftext|>` (Qwen3-dense excluded to keep its prior stop set).

**Phi-4 gate results (single B70 [0xe223], oracle llama.cpp `build-vk` / `ie-llama-dump`):**
config 40L/5120/40q-10kv/hd128/ffn17920/θ2.5e5/vocab100352; greedy=` Paris` (logit
18.44 vs 18.45 oracle, top-5 identical); **per-layer cosine ≥ 0.999999 across
L01..L41**; **PPL 8.2475 / NLL 2.109913** (deterministic ×2); **chat coherent +
stops at `<|im_end|>`** (correct `s[::-1]` reverse-string fn; `build_chatml_prompt`
renders `\n` not Phi's `<|im_sep|>` — model robust, byte-exact separator is a noted
follow-up). **Crown 6.45 / NLL 1.864495 bit-exact + dense NLL 2.940491 bit-exact**
after every shared-file edit. Phi-4 GGUF 9.05 GB; box disk tight (12 GB free).

**Phi-3.5-mini — STILL BLOCKED, scoped as the NEXT phi3 milestone.** The fused-split
+ Q5_K + tokenizer work above also covers Phi-3.5's tensor/dtype blockers, but it
additionally needs **(a) LongRoPE** (`rope_factors_long`/`_short` SU short/long
selection by seq-len; the dense path only does a single linear `rope_freqs` scale)
and **(b) a non-power-of-two head_dim 96**, which trips the FA-2 load gate
(`dense_transformer.cpp` `head_dim must be power-of-two ≥16`). Phi-4 (hd 128, plain
rope) needed neither. Phi-3-small (block-sparse/cl100k) and Phi-3.5-MoE are further out.

<details><summary>Original BLOCKED finding (2026-06-12, pre-unblock) — kept for provenance</summary>

- **The blocker (both models):** llama.cpp's phi3 convert keeps Microsoft's
  *fused* projections fused — one `attn_qkv.weight` (Q‖K‖V) and one `ffn_up.weight`
  (gate‖up) per layer. Our DenseModel loader expects split `attn_q/k/v` +
  `ffn_gate/up` → load fails immediately:
  **`model.load: layer 0 attn_q: tensor not found`**. WORSE: `attn_qkv` is
  quantized **Q5_K**, and the dense GEMV path supports only Q4_K/Q6_K/F16 — so
  even after a split the Q-slice is rejected (the crown/qwen3moe paths DO have a
  `dequant_q5_K_to_Bt` kernel, it's just not wired into `upload_quant_dense`).
- **Phi-4 (~15B):** 40L, hidden 5120, 40q/10kv GQA, hd 128, ffn 17920, θ2.5e5,
  vocab 100352, plain rope. `attn_qkv`[5120,7680]=Q5_K, `ffn_up`[5120,35840]=Q4_K.
  **Tokenizer `pre=dbrx`** (gpt2/tiktoken cl100k BPE) — UNSUPPORTED; engine does
  `llama-bpe`/`qwen2`/`tekken` only, so `dbrx` silently falls to the qwen2
  default branch (wrong regex, no `ignore_merges`) = a 2nd gap. `chat_template` =
  ChatML-with-`<|im_sep|>`.
- **Phi-3.5-mini (~3.8B):** friendlier tokenizer (`pre=default`, SPM `llama`) but
  STRICTLY HARDER arch — same fused Q5_K `attn_qkv` + fused Q4_K `ffn_up`, PLUS
  **LongRoPE** (`rope_factors_long`/`_short`; dense path only does single-set
  linear `rope_freqs`) AND **head_dim 96** which fails the FA-2 load gate
  (`head_dim must be power-of-two ≥16`). NOT downloaded (header proved it).
- **Per-task fallback note:** task said fall back to Phi-3.5-mini if Phi-4's
  *tokenizer* is unsupported. Phi-4 IS tokenizer-blocked (`dbrx`), but Phi-3.5-mini
  is ALSO arch-blocked (and harder), so the fallback can't light up the family
  either — the whole family is gated on the fused/Q5_K work below.
- **Follow-up to unblock Phi (4 items, precise):** (1) fused-tensor splitter in
  `dense_dispatch.hpp` (slice `attn_qkv`→Q/K/V and `ffn_up`→gate/up at row-span
  boundaries — same contiguous-superblock logic as `upload_quant_dense_permuted`;
  un-permute on Q/K slices only; trigger when `attn_q` absent + `attn_qkv`
  present); (2) wire Q5_K into the dense GEMV path (`dequant_q5_K_to_Bt`); (3)
  `pre=="dbrx"` pre-tokenizer branch (cl100k regex + `ignore_merges`, Phi-4 only);
  (4) LongRoPE short/long SU selection + relax the hd power-of-two gate for hd 96
  (Phi-3.5-mini only — Phi-4 needs neither). Items (1)+(2)+(3) = Phi-4; all four
  = Phi-3.5. Ledger: `docs/benchmark_matrix_2026-06-09.md` §Wave-1 Gate 3/9.

</details>

### ✅ WAVE-1 GATE: IBM Granite-3.x dense — **Granite-3.3-8B VALIDATED** (2026-06-13)
**Granite-3.x now loads, forwards correctly, and chats — the `granite` arch is
GREEN.** It rides the kLlama3 dense path but needed the FOUR scalar multipliers
(the only forward edit in the breadth sprint) actually applied in the forward —
they were read into `DenseConfig` (b4a1c05) but never consumed. Two additive,
value-gated commits (34b0049 forward, 13f2fe7 template); every non-Granite model
byte-identical. Model: `~/models/lmstudio-community/granite-3.3-8b-instruct-GGUF/
...Q4_K_M.gguf` (4.7 GB, single B70). Oracle: llama.cpp `fdc3db9b6`.
- **Arch route:** GGUF `general.architecture=granite` → **kLlama3** (40L, hidden
  4096, ffn 12800, 32q/8kv, head_dim 128, vocab 49159, θ=1e7, eps 1e-5).
- **The 4 scalars (verified vs the real GGUF):** `embedding_scale`=12.0 ×token_embd;
  `residual_scale`=0.22 ×each sub-block output before its residual add;
  **`attention.scale`=0.0078125** as the SDPA softmax scale (vs default 1/√128≈0.088
  — ~11× smaller; applied by pre-scaling Q by attention_multiplier·√HD since the
  attention kernels are crown-shared, signature untouchable); `logit_scale`=16.0
  divides the final logits. **KEY-NAME BUG FIXED:** llama.cpp namespaces the attn
  scale `granite.attention.scale` (DOTTED, `LLM_KV_ATTENTION_SCALE`), but the read
  used `granite.attention_scale` → it silently defaulted; fixed in model_config.cpp.
  New dense-local `dense::scale_inplace` helper; 5 value-gated call sites.
- **Forward CORRECT:** greedy "The capital of France is" → " **Paris.**" (argmax id
  2716 'ĠPar' logit 16.4, clean margin — without `attention.scale` the scores are
  ~11× too large → garbage). Per-layer cosine vs llama oracle **0.9995–0.99999 on
  L01–L39**; L40/L41 dip <0.999 = the known final-norm/lm_head dump-shape tail
  artifact (every dense gate); the `rel_fro ~0.07` "DIVERGED" tags are fp16-residual
  precision on one massive-activation channel (@3433, cosine stays ~1.0). **PPL 10.30**
  (sane for an 8B; better than qwen3-8b 18.93).
- **Chat (kGranite template):** `<|start_of_role|>{role}<|end_of_role|>{content}
  <|end_of_text|>` (NOT ChatML) — new `TemplateFamily::kGranite` + `build_granite_prompt`
  + stop on `<|end_of_text|>` (=eos id 0; markers are single ids 49152/49153/0).
  "Write a Python function to reverse a string" → coherent slicing impl + docstring
  + a unittest suite, stops clean (no free-continuation). Before this the ChatML
  fallback made it emit eos immediately (empty output).
- **REGRESSION GREEN:** crown PPL **6.4527 / NLL 1.864495 bit-exact**, dense qwen3-8b
  **NLL 2.940491 bit-exact** (value-gating → byte-identical), ctest **28/28**.
- **Unlocks the Granite-3.x dense tier** (3.1/3.2/3.3 instruct, same scalars+template).
  Granite-4.0 went hybrid-Mamba2 → a separate SSM arch, NOT this path (Wave-2).
  Ledger: `docs/benchmark_matrix_2026-06-09.md` §Wave-1.

### ⏯ SPRINT 2026-06-12 (autonomous): multi-GPU TP + qwen3moe NEW ARCH
**Two big breadth/scale threads landed + committed this sprint (crown PPL 6.45
bit-exact throughout, ctest green):**
1. **Tensor-parallel multi-GPU** — `DenseModelTP` (qwen3/qwen2 + llama), validated
   cosine 0.999999, **72B decode 1.44×** layer-split (10.4 tok/s), 72B/TP PPL 8.98.
   Wired into **`ie run/serve --gpus N`**. P2P hardware-blocked on 2×B70. gguf import
   vocab-padding fix (llama round-trip). See the TP/72B blocks below + [[project_multi_gpu]].
2. **`qwen3moe` NEW ARCH (Qwen3-Coder-30B-A3B)** — arch family #6. `Qwen3MoeModel`
   (`include/ie/qwen3moe.hpp` + `src/model/qwen3moe.cpp`): dense QK-norm attention +
   **unfused top-k MoE FFN** (host softmax router — `moe_router` op hardcodes 256 experts
   so unusable for 128 — + per-expert device GEMVs, gate/up Q4_K, down Q6_K stacked,
   `scaled_add` accumulate; attn auto-dequants per-layer Q5_K/Q8_0→fp16). **VALIDATED:**
   greedy coherent + **PPL 11.99** vs llama oracle 20.19 windowed (same directional ratio
   as 27B/8B → forward correct). Wired into `ie-perplexity` + the **Engine** (`ie run/serve`).
   GGUF at `~/models/Qwen3-Coder-30B-GGUF/...Q4_K_M.gguf` (18.6G). Tool `ie-qwen3moe-test`.
   **✅ Coder CHAT FIXED (8c568a6):** root cause = Qwen3-Coder HAS `<think>` in vocab
   (151667/151668) but its template never uses them (non-reasoning Instruct); the Engine
   injected the empty-think `<think>\n\n</think>` → derail. Fix: gate `model_has_think` on the
   **chat_template** string containing `<think>`, not token presence. Now answers correctly
   ("## Method 1 (string slicing s[::-1])..."). Crown/27B templates use `<think>` → unchanged.
   Tokenizer/template/EOS all verified correct (`IE_DUMP_PROMPT=1` diagnostic added).
   **✅ FUSED MoE FFN DONE (2026-06-12, latest commit):** the unfused per-token loop
   is replaced (behind `IE_QWEN3MOE_UNFUSED=1` as the parity oracle) by a host
   counting-sort token→expert packer + the crown's runtime-parameterized fused ops
   (`moe_gather_rows → moe_prefill_gate_up_silu_q4k → moe_prefill_down_packed_q6k →
   moe_prefill_reduce` for T>1; the 2-launch `moe_decode_*` pair for T==1) — **all
   in `qwen3moe.cpp` only, ZERO edits to `qwen36.cpp`/`moe_fused.cpp`** (copy-not-hoist).
   Gates: fused PPL **11.9981** == unfused **11.9856** (Δ 0.013 < fp16 floor) =
   authoritative correctness check; crown **6.45 bit-exact** (NLL 1.864495); p2/p3
   parity ALL GREEN; ctest **25/25**. Host packer unit-tested (`qwen3moe_pack_test`).
   **PERF (order-controlled A/B, B70, 2026-06-12 — `docs/benchmark_matrix_2026-06-09.md`
   §qwen3moe):** fused **pp512 ~61 / tg128 ~19.3** tok/s; unfused **~58 / ~18.4** →
   *fused is correct, not the perf lever here.* `--kprofile-decode` proves it:
   `attn_naive_compute` = **71.3%** of the decode step, fused MoE decode kernels only
   ~13% — qwen3moe is **naive-attention-bound** (no FA-2/split-K for this arch yet).
   vs llama.cpp Vulkan (master `fdc3db9b6`): pp512 **~984**, tg128 **~58.6** → engine
   is **16× behind prefill / 3× behind decode** (HONEST: NOT a "beats llama" entry;
   crown stays the only such claim; the prefill gap is a GPU MoE-GEMM KERNEL gap
   — fp16 expert path, no int-dot at `E_ffn=768`, top-8 = ~4× work — NOT swap;
   warm-plateau number is ~88 → ~10.4× behind, see §7 Tier-2).
   `ie-bench` taught the `qwen3moe` arch (additive branch, no crown-file edits).
   **NEXT (engine):** Wave-1 breadth blitz (`docs/superpowers/plans/2026-06-12-wave1-breadth-blitz.md`);
   then qwen3moe FA-2/split-K for dense-attention decode (the 71% lever) + **qwen3moe
   multi-GPU** (TP/layer-split) → path to **Qwen3-Next-80B**.
   Scoping: `docs/superpowers/specs/2026-06-12-qwen3-moe-and-next-scoping.md` (Target A DONE).

### ⏯ HARDWARE RESOLVED + P3d Task 3A DONE (2026-06-11 cycle-3)
**2× B70 = 64 GB VRAM now LIVE.** Required a **motherboard swap**: the original
ASUS Z890 AYW had one CPU-direct x16 slot + chipset x4 slots, and the B70's
hardware SR-IOV (TotalVFs=4) forced a PF+VF ≈88 GB BAR reservation onto a ~33 GB
chipset PCIe domain → kernel dropped the card's PF (`xe` "failed to map registers").
Nothing software-side fixed it (BIOS SR-IOV off, `xe.max_vfs=0`, VT-d off,
`pci=realloc` all failed — VF BARs are reserved at PCI enumeration, pre-driver).
New **ASUS ProArt Z890-CREATOR WIFI** gives both cards proper windows: both bind
`xe`, `enable=1`, `totalvfs=0`, render nodes `renderD129`+`renderD130` (iGPU=128).
Engine selects a B70 via name filter `"0xe223"` (excludes iGPU); true multi-GPU
will need an index/affinity selector later. **RAM still 32 GB** — RAM-serialize
loads regardless of VRAM ([[project_box_hardware_limits]]). Leftover cmdline
`pci=realloc xe.max_vfs=0` is harmless and kept (max_vfs=0 = good SR-IOV hygiene).

**✅ P3d Task 3A DONE (loader + scaffolding).** New files `include/ie/qwen35_dense.hpp`
+ `src/model/qwen35_dense.cpp` + `tools/qwen35_load_test.cpp`; additive engine
dispatch (`kQwen35Dense` branch in `engine.{hpp,cpp}`). `Qwen35DenseModel::load`
ingests the bartowski 27B GGUF end-to-end — all 64 transformer layers (48 linear +
16 full), dtype-driven weight loader (Q4_K/Q6_K/F16 → int-dot GEMV; **Q5_K/Q8_0 →
dequant-to-fp16 [K,N]** via the landed kernels, handling bartowski's per-layer mixed
quant), NextN `blk.64` skipped, hybrid caches (KV=16 full, DeltaNet state=48 linear,
conv_channels computed directly =10240 per R1) + workspace allocated on one B70.
**Gate: `ie-qwen35-load-test` → `LOAD OK arch=2 vocab=248320`; crown PPL 6.45
bit-exact (crown files 0 edits — additive only).**

**✅ P3d Task 3B DONE (full-attn + dense-MLP forward runs).** `Qwen35DenseModel::forward`
implemented: embedding → per-layer {pre-norm → full-attn OR DeltaNet → +res →
post-attn-norm → SwiGLU → +res} → final norm → lm_head. Full-attn path = joint
Q|gate proj → `split_q_gate_per_head` → per-head Q/K-norm → partial RoPE (n_rot 64)
→ FA-2 (KV idx = L/interval) → `sigmoid_gate` → out-proj. **The 48 linear (DeltaNet)
layers are STUBBED to a zero attention contribution** (clearly marked) so the
pipeline runs end-to-end. Smoke: `ie-qwen35-load-test … "prompt"` prefills + decodes
8 tokens with **no kernel-shape crashes** (output garbage as expected — ¾ of the
model is stubbed). Change isolated to the new qwen35 forward (0 crown/shared edits).
Task-4 verification checkpoints flagged in-code: norm convention (std vs 1+w),
rope_dim=64, sigmoid_gate direction.

**✅ P3d Task 3C DONE (gated-DeltaNet) + Task 4 PPL oracle cross-anchor PASS.** The 48
linear layers now run the UNFUSED recurrence (split → l2norm[q:1/√128,k:1] → repeat 16→48
→ compute_g_beta_h16 → deltanet_recurrence(48,128,128) → gated_rms_norm → ssm_out),
ssm_alpha/beta F32→fp16 transposed at load. **Greedy "The capital of France is" → "Paris."**
(was garbage with the stub). **`ie-perplexity` taught the qwen35 arch** (additive 3rd branch
alongside dense/crown) → **27B PPL 5.34 (avg NLL 1.675056), deterministic.** Oracle
cross-anchor: `llama-perplexity -c 256` on the SAME corpus = **8.38 ± 1.25**; ours is
lower because we stream full causal context per token vs llama's 256-windows — the SAME
directional relationship the validated dense path showed (ie 18.9 vs llama 24.5). Forward
is end-to-end correct + oracle-consistent. Commits 65056b8 (3A) → 5d3cbdc (3B) → b9db14f (3C).

**✅ 27B CHAT BUG — RESOLVED as a MISDIAGNOSIS (2026-06-12, human-present session).**
The alleged "temp-0.7 chat garbage" **does not reproduce** and the prior root cause was wrong on
every prior revision (vision-token → sampler → intermittent → "order-dependent state leak" — 5
contradictory doc commits = the root cause was never actually pinned). Re-investigated from
primitives under systematic-debugging; the evidence:
- **The flagged tokens 248068/248069 are `<think>` / `</think>`** (USER_DEFINED, NOT CONTROL —
  verified in the GGUF `tokenizer.ggml.token_type`). The model emits them **legitimately** (its
  reasoning mechanism). In RAW (non-chat) context the 27B correctly argmaxes `<think>` to start
  reasoning; it decodes to '' under skip-special, so "invisible special-token argmax" =
  **correct behavior, not garbage.** The earlier "271,248068,271,248069" trace was the raw-generate
  smoke emitting `<think>`/newlines — misattributed to chat.
- **"State leak" REFUTED:** `forward_step` is temperature-independent (temp only enters
  `sample_softmax_topk_topp`, never the forward) — so "greedy works" and "sampler picks rank-0
  garbage" cannot both hold for one prefill; the prior claim was internally contradictory.
  `DeltaNetState::reset()` zeros state+conv (correct); the production `deltanet_recurrence` carries
  NO static/persistent scratch (the only statics are under the default-OFF `IE_DN_RECURRENCE_REWRITE`);
  `kv_.reset()` + `start_pos=0` per `generate()` make stale KV unreadable (causal). No reset-exempt
  state path exists.
- **Empirics:** **114 chat generations clean** — thinking ON *and* OFF, 100+ seeds, temp 0.7, varied
  prompts (`tools/qwen35_load_test.cpp <gguf> <prompt> <base_seed> <n_seeds>` sweep mode + `ie run`).
  Zero garbage; outputs are coherent/high-quality. The reference (llama.cpp 27B) also chats clean.
- **The "obvious fix" would have BROKEN the model:** suppressing special tokens from sampling (the
  natural response to "special tokens emitted") would mask `<think>`/`</think>` — the markers the
  reasoning path REQUIRES. Verifying token types before fixing (Iron Law) averted that regression.
- **Verdict:** the 27B chats correctly at temp 0.7; **no code fix is warranted.** The original
  rare "sd"/"etter" sighting (if real) is at most ordinary temp-0.7 sampling variance, not a
  forward/state defect — not chased further.
**DEPLOYABLE:** `ie serve` + `/v1/chat/completions` and `ie run` both produce correct chat at the
default temp 0.7 — the 27B is a usable product today, full stop.

**✅ P3d COMPLETE (2026-06-12).** All tiers closed:
(a) **per-layer cosine PASSED** — `ie-llama-dump` rebuilt vs llama master `fdc3db9b6` (build-vk
libs), new `ie-qwen35-dump` (engine side, `tools/qwen35_dump.cpp`), `tools/diff_layers.sh` over
"The capital of France is": **cosine ≥0.9995 on ALL 64 layers** (embedding L00 = 1.000000; L01
0.999990; min L53 0.999543), **exact greedy argmax id=11751 ' Paris'** (logit 16.03 vs oracle
16.15), top-5 identical. The per-layer `rel_fro ~0.02` (vs the dense path's 0.0014) is the **fp16
residual stream** (we keep fp16; llama keeps fp32) accumulating over 64 layers + DeltaNet/M-RoPE —
**proven inherent, not a defect**: `IE_QWEN35_NO_ONEDNN=1` (bit-exact gemm_fp16) gives the SAME
cosine (oneDNN-invariant), and exact argmax + PPL 5.34 oracle-consistency confirm correct output.
Concentrated on massive-activation dim 3994, smooth depth-accumulation (no op-localized crater).
(b) ~~Task 5 perf~~ **DONE** (prefill 1.9×). (c) ~~chat-as-product smoke~~ **DONE** (114 clean gens).
Tooling added: `ie-qwen35-dump`, `IE_QWEN35_NO_ONEDNN` parity kill switch, `ie-llama-dump` rebuilt.
**P3d is fully validated and complete; next breadth model = P3a (Llama-3).**

**✅ Task 5 — perf A/B + prefill OPTIMIZED: 27B BEATS llama.cpp on prefill.** `ie-bench`
taught qwen35. Same-hardware (one B70) vs llama.cpp **Vulkan** (build-vk — HAS the GPU
DeltaNet shaders gated_delta_net/ssm_scan/ssm_conv; a fair, tough backend):
```
                  ours (SYCL)        llama.cpp (Vulkan)
  pp512 prefill    577  (was 185.8)   303      ← 1.9× FASTER (was 1.65× behind)
  tg128 decode      10.0               9.72    ← parity
```
THREE prefill opts (all PPL-bit-identical, crown-safe via the qwen35-only prefer_onednn flag):
(1) batch ssm_alpha/beta 186→296; (2) oneDNN on the Q4_K/Q6_K dequant→gemm path 296→521;
(3) oneDNN on the F16 branch too (attn_k/v/o, ssm_out, alpha/beta) 521→577.
ORDER-CONTROLLED (3 alternating same-hour rounds, ours↔llama): ours 504.8/521.8/535.8,
llama 301.2/304.4/303.6 → consistent **1.67–1.77×**, far outside ±40 tok/s heat noise
(ours trends UP as the card heat-soaks; llama flat). Defensible, not a fluke.
**Two profiling-driven prefill optimizations this session (`--kprofile` found each), PPL
held bit-identical (5.3391) throughout:**
1. **Batch ssm_alpha/beta** (`cc52fb9`): they were N=48 → failed the gemm N-gate → a
   per-token serial gemv (49,152 launches = 38% of prefill). Pad to N=64 (SG-aligned;
   gemm's B-load isn't N-bounds-checked) → batched gemm + strided `extract_cols`. 185.8→296.
2. **oneDNN matmul default for qwen35** (`44febe3`): oneDNN is ~1.7× faster than our
   gemm_fp16 at the 27B's shapes (K=5120, ffn 17408). Per-model `dense::prefer_onednn()`
   flag set only by `Qwen35DenseModel::load` — crown/qwen3-dense keep bit-exact gemm_fp16
   (crown PPL re-verified 6.45 bit-exact). 296→**510–521**. (Fused-XMX A/B'd, rejected:
   39.8 — re-streams quant blocks at big-K.)
**Decode parity (10.0 vs 9.72)** — NOT DeltaNet-bound (profiled: dn_recurrence 0.8%). Decode
is GEMV-bound and **Q6_K-dominated**: gemm_q6k ffn_down K=17408 (27%, the SLM-doesn't-fit
scalar fallback) + gemv_q6k_med attn (22%). **Int-dot q8 decode A/B'd and REVERTED** —
gemv_q4_K_q8 measured SLOWER than scalar at the big-N FFN (0.164 vs 0.123 ms/call), and the
Q6_K big-K is the known-hard P3b dead-end (ktiled int-dot was slower there too). So 27B decode
has no easy win; it already beats llama (10.0 vs 9.72) and is left as-is. Real decode upside
would need a repacked-Q6_K or split-K kernel (out of this session's scope). **Claim:** the 27B now **beats llama.cpp on prefill (1.7×) and matches on
decode, correct (PPL 5.34)** — but the headline "beats llama.cpp" still belongs to the CROWN
(35B MoE, bit-exact); the 27B is breadth that now also wins prefill.
**Repro:** llama oracle = `build-vk` rebuilt from master fdc3db9b6 with a 1-line patch
(`ggml-vulkan.cpp:6207` `eMesaHoneykrisp` → `false`; Vulkan-Hpp enum absent on this box,
Asahi-only, irrelevant to Arc) + `-j4` (mul_mm shader OOMs at high -j on 32 GB RAM).

**Honest claim discipline:** we do NOT yet beat llama.cpp on the 27B (decode parity, prefill
behind). The "beats llama.cpp" claim remains the CROWN (35B MoE) only. The 27B win is BREADTH
— it runs, correctly (PPL 5.34), at decode parity, with a clear prefill optimization path.

---
**(historical) The 3C implementation drive — UNFUSED recurrence per the oracle dataflow**
(R2: split → repeat(3) → l2norm×2 → deltanet_recurrence(n_v=48) → gated_rms_norm → out-proj),
oracle-diffed. Fits ONE B70 (18 GB), zero new kernels. Reference material:
- Oracle (rebuilt, working): `~/llama.cpp/build-cpu/bin/llama-eval-callback` (per-op
  dumps) + `llama-perplexity`. Op-by-op spec + verified shapes + landmines:
  **`docs/qwen35_27b_oracle_dataflow.md`** (read this first). Config foundation
  (`Qwen35Config`, `recurrent_layer()`) is complete.
- Build `Qwen35DenseModel` (NEW files only; crown untouchable): loader (dequant
  Q5_K/Q8_0 at load — kernels landed) + DeltaNet/full-attn/dense-MLP orchestration
  per the verified dataflow → CPU-diff each named tensor vs the oracle (Task 2.5
  method) → GPU parity (Task 4, new `scripts/p3d_parity_qwen35.sh`).
- GPU EXCLUSIVITY + always pass `--gguf`; `ps`/`free` before any load.

**Secondary track (now that 2 GPUs exist) — multi-GPU + 72B (decision §5.7).**
72B AWQ download is **COMPLETE** (`~/models/Qwen2.5-72B-Instruct-AWQ`, 41.6 GB,
`qwen2`+AWQ = P3e-supported, import is memory-safe per-tensor). It still can't RUN
on the engine until **multi-GPU device-split offload** exists (the engine selects
one device). Choose: (a) build engine multi-GPU (unlocks 72B, real credibility
feature), or (b) bank "P3e at scale" on **Qwen2.5-32B-AWQ (~19 GB, fits ONE B70,
runnable now)**. The 72B *import* (weight-faithful GGUF) is bankable anytime.

**IP posture (P4 launch): RESOLVED** — `THIRD_PARTY_LICENSES.md` attributes the 3
ggml-MIT import-path ports; engine core is original. `tools/llama_dump.cpp` is
dev-only/not-shipped (exclude from public repo).

---
State as of the 2026-06-10 late-night context reset. Two prongs run in
parallel (engine = C++/GPU; Seal = Rust/TS/CPU), serialized only at GPU
gates. Execution method all session: one agent per plan-task, then a
spec-compliance review + a code-quality review before the next task; plans
themselves are architecture-reviewed before any code. Keep that rhythm.

**✅ RESOLVED — P3b re-scope → BREADTH PIVOT (2026-06-11).** Decision made
(human-directed: "be the engine Intel can't ignore = run many models + file
types"). `p3b-task1-*` branch fast-forwarded into main (default-OFF kernel
parked, `e1e1fa5`/`5ef5f6e`), branch deleted, trunk restored. P3b paused. New
top priority: **P3d Qwen3.6-27B** then **P3e AWQ import**.

**⚠️ GPU-INCIDENT NOTE (2026-06-11):** `ie-bench` has NO `--help` handler and
default-loads the 20 GB crown GGUF — a stray `ie-bench --help` ran concurrently
with an 8B bench and maxed the 32 GB B70, tainting a baseline (read 22 ms vs the
true 10.31 ms). RULE: read tool arg-parsers in source, never probe with `--help`;
always pass explicit `--gguf`; `ps` before any GPU run; foreground + serialized.

**Engine P3d — Qwen3.6-27B (`qwen35` dense-hybrid) — ACTIVE:**
Plan `docs/superpowers/plans/2026-06-11-p3d-qwen35-dense-27b-support.md`.
- **Task 0 DONE** (`55039da`): `ModelArch::kQwen35Dense`, `read_qwen35_config`,
  validated vs the real GGUF (`~/models/bartowski/Qwen3.6-27B-GGUF/Qwen_Qwen3.6-27B-Q4_K_M.gguf`,
  17.98 GB, downloaded). ctest 16/16. **Caught the NextN/MTP layer** (block_count
  65 = 64 transformer + 1 NextN) — captured in `nextn_predict_layers`.
- **Reuse thesis (verified in-tree):** the crown (`qwen35moe`) already implements
  the gated-DeltaNet op (`src/ops/deltanet.cpp`, 128/128 head-dim, runtime head
  counts), the gated full-attn (`ws_q_full_` interleaved Q|gate), the ssm tensor
  loader, partial RoPE. The dense MLP is the P2 path. **27B ≈ crown ops + dense MLP**.
- **Tasks 1-2 DONE:** tensor layout `docs/qwen35_27b_gguf_layout.md`; tokenizer
  resolved by reuse (27B tokenizer byte-identical to the crown: gpt2/qwen35/248320).
- **Task 3 prerequisite DONE (`feat(p3d): Q5_K + Q8_0 transposed dequant kernels`):**
  `dequant_q5_K_to_Bt` + `dequant_q8_0_to_Bt` (27B has Q5_K attn_k/attn_output + Q8_0
  ssm_out the dense path lacked). GPU-tested vs flat dequant (`tests/unit/dequant_bt_test.cpp`).
- **✅ ORACLE RESOLVED (2026-06-11 cycle 2):** the old HEAD `dcad77cc3` simply predated
  llama.cpp's **MTP/NextN support series** (PRs #22673, #24025…). Updated to current master
  (`fdc3db9b6`, build `b9598`, +665 commits) + rebuilt CPU-only at `~/llama.cpp/build-cpu/`;
  its `qwen35.cpp` reads `LLM_KV_NEXTN_PREDICT_LAYERS` and treats `blk.64` as **dense** —
  **loads the bartowski 27B GGUF cleanly and generates coherent text.** Full bit-exact
  oracle now available: `llama-perplexity` (PPL) + `llama-eval-callback` (per-op dumps for
  Task 2.5). NOTE the new `llama-cli` is interactive-only (`-no-cnv` removed); scripted
  oracle runs use `llama-eval-callback`/`llama-perplexity`. ⚠ Box is **32 GB RAM** — an
  18 GB CPU load concurrent with the 72B download + VS Code + LM Studio OOM-crashed the
  editor; **serialize model loads on RAM too** (pause downloads, check `free`, cap threads).
- **THE BIG REMAINING PIECE (focused multi-session):** Task 3 `Qwen35DenseModel`
  loader (upload qwen35 tensors, dequant Q5_K/Q8_0 via the new kernels at load) +
  Task 3b the **~600-line unfused DeltaNet/gated-attn forward** re-derived from the
  leaf ops with CORRECTED shapes. **Op map + the landmines (BAN `SI2=SI*2` → use
  `conv_channels = d_inner + 2·n_k_heads·state = 10240`; FORBID `dn_qkv_split_norm_fused`
  VT=2*KT → unfused split/repeat(3)/l2norm; residual order; SVH=48) are all in the
  plan's "Architecture review — INCORPORATED" + "Progress log" sections.** The leaf
  ops (deltanet_recurrence, depthwise_conv1d_causal, l2_norm_scale,
  repeat_interleave_heads, gated_rms_norm, sigmoid_gate, split_q_gate_per_head,
  rope, full_attention, swiglu) all exist with signatures in `include/ie/ops.hpp`.

**Engine P3e — AWQ+GPTQ import — ✅ COMPLETE & PROVEN (independent of P3d):**
Plan `...-p3e-*.md`; validation `docs/p3e_awq_import_validation.md`; memory
[[project_awq_gptq_import]]. `ie import <hf_dir> <out.gguf> <tok_ref.gguf>` loads
**AWQ + GPTQ** (two formats llama.cpp can't) across **Qwen3 AND Qwen2.5 dense**.
All host-tested (ctest 22/22): safetensors_reader (+ sharded `SafetensorsModel`),
awq (awq+gptq dequant), gguf_writer, quantize_q4k+quantize_q6k (ggml ports),
hf_import (config+name-map+driver, **dtype-aware** to_f32 — Qwen2 F16 vs Qwen3 BF16
embeddings), `ie import` first-class CLI.
- **VERIFIED end-to-end on real models:** Qwen3-4B AWQ (40.68) + GPTQ (32.08≈ref),
  Qwen2.5-3B-AWQ (18.73≈ref 17.30), **Qwen2.5-7B-AWQ (2-shard, 15.48 + correct codegen)**.
  Weights bit-verified vs numpy dequants (≤1.6%). Qwen3-8B dense **2.937037 EXACT
  (zero regression)**. Imported models generate correct coherent output via `ie run`.
  Models in `~/models/Qwen3-4B-{AWQ,GGUF}/`, `~/models/Qwen3-4B-GPTQ/`,
  `~/models/Qwen2.5-{3B-AWQ,3B-GGUF,7B-AWQ}/`.
- **Qwen2 support (new arch):** rides DenseModel; attn QKV bias + QK-norm gated on
  tensor presence (Qwen3 byte-identical); `add_bias` op; `read_dense_config_auto`.
- **Follow-ups (optional):** act-order GPTQ (desc_act=true g_idx); 14B/72B scale;
  self-contained HF tokenizer (drop the ref-GGUF arg); vLLM cross-check PPL.
- **P3a Llama-3.1-8B — ✅ COMPLETE (2026-06-12).** Runs on the DenseModel path
  (`general.architecture=llama` → `kLlama3` → `is_dense_arch`). All deltas landed:
  load-time Q/K inverse row-permute (`llama_qk_unpermute_rows`, host-tested),
  `rope_partial_ff` (rope_freqs scaling, null→bit-identical), llama-bpe tokenizer
  (digit triplets + ignore_merges, qwen3 goldens still 13/13), `build_llama3_prompt`
  + per-arch stop set, add_bos default-true for llama-bpe. **VALIDATED:** per-layer
  cosine **0.99998–1.000000 all 32 layers** vs the llama oracle (un-permute+rope_ff
  proven correct — L01 0.999983); coherent generation ("a city of grandeur…");
  greedy argmax near-tie (' a' vs ' Paris' within 0.1 logit, same top-2 — fp16
  precedent); **PPL 10.79** (sane, deterministic). qwen3 regression ALL GREEN,
  crown PPL 6.45 held. Gate: `scripts/p3_parity_llama3.sh`. Commits a60f967 (Task 1)
  → fa9fe0e (Tasks 2-5) → 2ceb1e3 (validation). **Llama-3.2-3B also validated** (smoke:
  tied embeddings + GQA=3 → " Paris. The capital of Italy is Rome…", coherent). Optional
  follow-ups: 64/64 greedy golden (needs llama-completion build), perf ledger.
- P3c Windows — `...-p3c-windows-groundwork.md` (4 tasks). NOT started.
  Only load-bearing POSIX is `src/loaders/gguf_reader.cpp` mmap → `MappedFile`
  abstraction (Task 1). Diagnostic tools (3 signal()/2 getenv HOME) are
  Windows-build-excluded.

**Seal P1 (`/home/weezy/SEAL - VERSION 2.1`, branch main):**
- P1a MCP client — `docs/plans/2026-06-10-p1a-mcp-client.md` — **all 7 tasks
  done** (commits ba5fe08→a8572c0; 150 tests). Cross-seam review verdict:
  **FEATURE_SOUND**. Two Important follow-ups to track (not blocking): (1)
  `subagents/mod.rs` env-collision at subagent nesting depth ≥2 (SEAL_MCP_DISABLE
  not cleared when setting SEAL_MCP_SERVERS); (2) `main.rs` `mcp tools` CLI
  creates a ghost session file that breaks `--resume last`. **Deferred: the MCP
  live gate** (spawn a real `@modelcontextprotocol/server-filesystem`, exercise
  a tool through Seal, resume-replay it) — batch with other GPU validations.
- P1b parallel subagents — `...-p1b-parallel-subagents.md` (5 tasks) —
  Task 1.1 pool core ✅ committed `fedd073` (155 tests). Tasks 1.2–1.5 next:
  pgid/no-spawn hardening, `spawn_subagents` tool, cockpit protocol (the review
  folded in: `_index` not `#index` prefix for SquadIndicator.tsx:73, add `|pool`
  to hidden-protocol allowlists in App.tsx:2402 + ChatView.tsx:52, add
  `spawn_subagents` to MissionGraphPanel skip), prompts/docs/live gate. Honors
  `ie serve` single-flight (default concurrency 2; overlap is non-decode time).

**GPU-validation backlog (batch under exclusivity, one server session):**
MCP live gate · P1b live gate (parallel subagents vs `ie serve`) · any P3b
re-measurement after the next kernel lands.

**Seal git note:** repo-local identity is the Claude attribution (sandbox
can't commit as the human); Seal `main` is ~25+ commits ahead of origin,
NOT pushed (human pushes when ready). Snapshot branch + memory-pivot branch
preserved, untouched.
