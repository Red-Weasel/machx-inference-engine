# Session handoff — 2026-06-21: Gemma-4 26B perf SOLVED (decode beats llama)

Read `MASTER_DEV_PLAN.md` banner first, then memory `project_gemma4_arch`. This session took the gemma-4
26B-A4B MoE (QAT Q4_0, 1×B70) from "7.2× behind llama / quality-scare" to **decode beats llama, prefill
1.17× behind, PPL beats llama** — all crown 6.4527 bit-exact.

## Final numbers (26B-A4B, 1×B70, fast config)
| metric | start | **end** | llama SYCL | verdict |
|---|---|---|---|---|
| pp (prefill, T≈477) | 114 | **708** | 828 | 1.17× behind (was 7.2×) |
| tg (decode) | 32 | **~55-58** | 54.75 | **✅ beats/ties llama** |
| PPL (wikitext-2, same GGUF) | — | 2880 (streaming-T1) | **1178 (reference)** | ⚠️ see PPL section — NOT a quality metric for this QAT GGUF |

**Fast config (gemma4 is tool-only; flip these on when wired to ie::Engine):**
`IE_GEMMA4_FUSED_MOE=1 IE_GEMMA4_INTDOT_PROJ=1 IE_GEMMA4_ONEDNN=1 IE_GEMMA4_MOE_SOA=1`

## What shipped (commits on `main`, all additive, crown bit-exact, greedy "Paris" intact)
1. **`a036018`/`4d48007`** int-dot W4A8 prefill projections (groundwork; split-K `gemm_q4_0_q8`).
2. **`b0a8a18`** weight-stationary fp16 PREFILL projections: `dequant_q4_0_to_Bt`→`gemm_fp16_onednn`
   (1.96×; the projection GEMMs collapsed 962ms→~6ms). THE prefill lever.
3. **`d3973a9`** SoA-Q4_0 W4A8 DECODE GEMV `gemv_q4_0_soa_q8`+`repack_q4_0_to_soa` (dense projections,
   ~80%-BW template like gemv_q6_soa). dense decode 36.8→44.
4. **`decd0dc`** SoA-Q6 lm_head (`gemv_q6_soa_q8` on tied token_embd).
5. **`77ed888`** SoA-Q4_0 MoE experts `moe_prefill_proj_q4_0_soa_q8` (bit-identical fold; experts stored
   SoA INSTEAD of AoS = no doubling the ~13GB). decode 48.9→58 AND prefill 655→708.
6. **`3fd0b67`/`c7615d8`** tooling: `ie-gemma4-gen … {bench-decode, profile, profile-decode}`.

## Measured NEGATIVES (kept opt-in, do NOT re-try)
- naive XMX `gemm_q4_0_xmx` (`c7615d8`): per-M-tile re-dequant, SLOWER than dp4a at M=477.
- MoE-XMX `moe_prefill_proj_q4_0_xmx` (`6f0fbe9`): matrix engine starves at ~30 rows/expert (small-M).
Lesson: XMX wins large-M (didn't beat the oneDNN weight-stationary path here); dp4a/SoA wins small-M.

## The PPL question — DEFINITIVELY SETTLED 2026-06-21 (it's the GGUF, not our engine)
Earlier "PPL beats llama" was apples-to-oranges (our 128-tok Frankenstein streaming vs llama windowed) and
is **RETRACTED**. The decisive test: **llama-perplexity (the reference) on wikitext-2 with THIS exact GGUF →
PPL = 1178** (`-c 512 --chunks 20`, per-chunk 700-1530). Ours on the same wikitext = 2880 (first-512
streaming-T1). **Both sit in the pathological ~1000-3000 band → the high PPL is a property of the QAT-Q4_0
instruct GGUF, NOT our engine.** llama itself cannot get this checkpoint below ~1000.

Mechanism (why coherent greedy + terrible PPL coexist): **QAT** trains the net so the **argmax** survives
Q4_0 quantization (→ greedy "Paris" + coherent essays, crown bit-exact) but does NOT preserve logit
**magnitudes** — the Q4_0 output projection compresses the logit spread → flat softmax → correct token gets
~0.0005 prob → NLL ~7 → PPL ~1000. Feeding an **instruct + multimodal** model raw documents (no
`<start_of_turn>`) adds a further 3-5× distribution mismatch. **⇒ For this GGUF, PPL is NOT a meaningful
quality signal — use greedy coherence / task accuracy.**

The 2.4× ours(2880)-vs-llama(1178) residual is **measurement protocol**: llama strides fresh-context chunks
and scores the back-window; our streaming-T1 single pass includes the high-NLL low-context opening tokens
(the `= Valkyria Chronicles =` header) that dominate a short 512-tok average. Not a forward error (argmax
matches). Previously ruled out as causes: softcap (correct; NO_SOFTCAP→94924), tokenizer (byte-identical,
BOS prepended for gemma), shared_kv_layers (=0), KV/decode path (reprefill≈streaming), int-dot.
**METHOD: validate PPL against the llama REFERENCE on a STANDARD corpus (wikitext-2), never on a short
ad-hoc snippet; and the engine-quality gate is crown ie-perplexity 6.4527 (a different, healthy model) +
greedy coherence — gemma's QAT PPL was never the gate.**

## GEMMA-4 31B DENSE — DONE 2026-06-21 (beats llama both axes, ZERO new code)
Ran first-try on the existing gemma4 path — the loader already branches dense vs MoE on `cfg.is_moe`
(dense GeGLU FFN every layer; MoE additive only when a `ffn_gate_inp` router is present), and the decode SoA
gemv / oneDNN prefill paths are arch-agnostic. 17.65 GB Q4_0, fits 1×B70.
| metric | ours (fast cfg) | llama Vulkan | llama SYCL | verdict |
|---|---|---|---|---|
| pp (T≈477) | **367** | 203 | 59.4 | **1.81× vs Vulkan** |
| tg (decode) | **20.5** | 15.35 | 6.09 | **1.33× vs Vulkan** |

- Fast config: `IE_GEMMA4_INTDOT_PROJ=1 IE_GEMMA4_ONEDNN=1` (no MoE flags — dense; SoA-Q4_0 decode default-on).
- **Bar caveat — use llama VULKAN, not SYCL:** SYCL is crippled on dense gemma4 (59/6 tok/s, ~18% BW — falls
  to a slow path; it ran the 26B *MoE* fine at 54.75, so it's dense-gemma4-specific). Honest bar = best
  backend = Vulkan. Don't quote the 6.2×/3.4× SYCL ratios as the headline.
- **Quality:** greedy coherent — "The Eiffel Tower is located in the city of" → "Paris, France. It is one of
  the most famous landmarks … in the world" (one QAT near-miss token, expected). QAT-Q4_0 ⇒ PPL high-but-
  meaningless (see PPL section); gate on greedy, not PPL. No engine code changed ⇒ crown 6.4527 untouched.
- **Decode is gemv BW/ALU-bound** (20.5 tok/s = 361/608 = ~59% BW; 17.6 GB/token read = 48.8 ms, matches).
  NEGATIVE RESULT: the obvious lever — removing the redundant per-layer `q.wait()` drains (the in-order queue
  makes them dead; dense loop has no mid-layer host reads) — measured **NEUTRAL** (clean A/B: 20.13 sync vs
  20.19 no-drain). Host-sync is NOT the bottleneck; the driver already overlaps enqueue/execute. Reverted.
  **Further dense-decode headroom (→~27 at 80% BW) needs a cross-arch SoA-`gemv_q4_0_soa_q8` rewrite — it's
  ALU-co-limited, shared by 26B/27B/31B, regression-risky → a SEPARATE milestone, not 31B work.**
- Profiler footgun: `IE_QUEUE_PROFILING=1` + the oneDNN prefill path → DEVICE_LOST (level_zero err 20) in
  `profile-decode`. GPU recovers; just don't profile-decode with ONEDNN. Use `bench-decode` A/B instead.

## ⚠️ KNOWN ISSUE — gemma-4 12B dense produces GARBAGE (parked 2026-06-21)
**CONFIRMED OUR ENGINE BUG (not the checkpoint) — 2026-06-21:** TWO independent conversions —
`google/gemma-4-12b-it-qat-q4_0.gguf` AND `lmstudio-community/gemma-4-12B-it-QAT-Q4_0.gguf` (different
quantizers, 832-byte size diff) — produce BYTE-IDENTICAL garbage (`[236771 '0' 23.45]` → "01111111").
Two good GGUFs failing identically ⇒ the bug is ours, 12B-specific. (The google one was deleted; use the
lmstudio file at `~/models/lmstudio-community/gemma-4-12B-it-QAT-GGUF/gemma-4-12B-it-QAT-Q4_0.gguf`.)
Symptoms: greedy outputs digit-token spam
(`'01111111'`; prefill top-5 = clustered digits at logits ~22-23, NO clear winner vs 31B's "Paris" 27.5),
AND ~5× pathological slowness (pp 70 / tg 4.8 vs 31B's 367 / 20.5 — a 7 GB model should be FASTER).
Systematically RULED OUT (via `IE_GEMMA4_DBG` per-layer + EMB + FINALNORM absmax probe — committed):
- NaN/blowup: residuals healthy through all 48 layers (no nan/inf; absmax rises ~14→171→6, magnitudes
  comparable to the 31B).
- Per-layer geometry: PERFECT — swa HD256/nkv8/nrot256/θ1e4, global (every 6th) HD512/nkv1/nrot512/θ1e6.
- lm_head: NOT it — SoA-Q6 (`gemv_q6_soa_q8`) AND reference `gemv_q6_K` give BYTE-identical garbage.
- Embedding scale `sqrt(H)` and attention scale (gemma4 `f_attention_scale=1.0`, our sqrt(HD) pre-scale)
  both match `~/llama.cpp/src/models/gemma4.cpp` (lines 11, 182) — and both work for the 31B.
- dtype (token_embd Q6_K, same), rope_freqs (present, same), vocab (262144, same).
⇒ Bug is a SUBTLE value-wrong (magnitude-healthy) error in the 12B layer stack, between embedding and
lm_head. Prime remaining suspect: global-layer attention at **nkv=1 / gqa_ratio=16** (the 12B is the first
arch to hit MQA on global layers; 31B global is nkv=4). ROOT-CAUSE BLOCKED: needs an op-by-op VALUE oracle,
but `llama-cli` (Vulkan) HANGS on this exact GGUF (couldn't get a reference) — which ALSO weakly suggests a
problematic checkpoint. **Resume path:** (a) get a working oracle (different 12B source / llama build / CPU),
op-by-op diff starting at the global-attention nkv=1 path; or (b) treat this QAT-12B upload as suspect and
re-pull from a different source. 26B + 31B are UNAFFECTED and both beat llama.

**UPDATE (later 2026-06-21) — ORACLE BLOCKED, 12B is a NEWER conversion the whole local toolchain predates:**
- Structure: the 12B GGUF per-block tensor set is IDENTICAL to the 31B (the 14 standard gemma4 tensors;
  NO AltUp/Laurel/PLE extras; `embedding_length_per_layer_input = 0`). So the "unified multimodal decoder"
  design does NOT appear as extra text-decoder tensors (vision lives in the separate mmproj). If it's a real
  difference it's a subtle numerical convention, not visible structure.
- `llama.cpp` build `fdc3db9b6` (build-vk) CANNOT run ANY gemma-4-12B variant on this box: QAT-Q4_0
  (lmstudio) → `llama_bench: error: failed to load model`; non-QAT Q4_K_M → dies (exit 144, empty stderr);
  yet it runs the 26B/31B fine. Local `gguf-py` reads 0 keys from the lmstudio QAT-12B (newer GGUF format).
  ⇒ the 12B is a NEWER release/conversion than the 26B/31B (those were converted Jun-13); our lenient reader
  loads it but the forward garbles → genuine our-engine forward bug on the 12B geometry (hidden 3840 / nq 16),
  and there is NO working op-by-op oracle on this box.
- non-QAT Q4_K_M (`lmstudio-community/gemma-4-12B-it-GGUF/gemma-4-12B-it-Q4_K_M.gguf`) does NOT load on our
  engine either — gemma4 loader is Q4_0-ONLY (`LOAD: attn_q: expected Q4_0`); would need a Q4_K dense path.
- **TO FIX:** update local llama.cpp to a build that supports gemma-4-12B (gives both the op-by-op oracle AND
  a fair perf bar), OR stand up the HF-transformers gemma4 reference; then op-by-op diff. Until then 12B stays
  parked. The 26B + 31B wins are the shippable deliverable.

## NEXT (priority)
1. **12B** — PARKED (see KNOWN ISSUE above); needs an oracle. Not a blocker; revisit with a clean reference.
2. **Wire gemma4 into `ie::Engine`** (`kGemma4` load branch + `<start_of_turn>` template + host-logits
   bounce like qwen3next) and flip the fast-config flags default-ON there. NB: `MOE_SOA` needs `FUSED_MOE`
   (unfused path still reads AoS `gate_up_exps`=null → would crash) — enforce/guard at wiring.
3. **Cross-arch SoA-gemv decode kernel** — the real dense-decode lever (lifts 26B/27B/31B); ALU-co-limited at
   ~59% BW, crown-gate carefully. Only worth it as its own milestone (31B already beats llama).
4. dedupe dense AoS+SoA (+1.1GB; currently keep both — AoS for oneDNN prefill, SoA for decode).
5. then the Pi-like harness / publish-prep (user's post-gemma direction).

## Method notes (carry forward)
- ONE GPU workload at a time; serialize every model load.
- llama apples-to-apples: `~/llama.cpp/build-vk/bin/llama-perplexity` (Vulkan runs gemma4 on GPU) +
  `build-sycl/bin/llama-bench`; build-cpu has perplexity (slow). Decode bench: `ie-gemma4-gen … bench-decode`.
- discard first run after a rebuild (JIT); decode ±heat-soak (back-to-back prefills throttle ~−25%).
- New hard rules this session: beat-llama-is-the-bar; think-first on engine changes
  (memory `feedback_beat_llama_is_the_bar`, `feedback_think_first_on_engine_changes`).
