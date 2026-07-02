# Gemma-4 — living authority doc (`gemma4` arch)

> One source of truth for the Gemma-4 family optimization state on Intel Arc Pro B70.
> Update at every milestone for this arch. If reality and this doc disagree, fixing the
> doc is part of the change. Cross-arch roadmap stays in `MASTER_DEV_PLAN.md §7`.
> **Last updated:** 2026-06-26 (from the 2026-06-21…25 gemma sessions).

---

## 0. Identity

- **Family:** Gemma 4 (Google, released 2026-04-02, Apache-2.0). Among the most complex
  arches the engine supports (after the crown). `general.architecture = "gemma4"` for
  **both** sizes (dense vs MoE detected per-layer via `blk.N.ffn_gate_inp` presence).
- **Sizes targeted:** **31B dense** + **26B-A4B MoE** (128 experts, top-8, ~4B active).
  (12B dense = parked/broken, see §2; E4B = unsupported gemma-3n matformer, do not retry.)
- **GGUF files + role:** official QAT GGUFs are plain **Q4_0** (only `token_embd`/`lm_head`
  are Q6_K): `google/gemma-4-{31B,26B-A4B}-it-qat-q4_0-gguf`. 31B ≈ 17.65 GB (fits 1×B70),
  26B-A4B ≈ 13.4 GB (fits 1×B70). MTP draft heads are **separate** files:
  `mtp-gemma-4-31B-it-Q8_0.gguf` (491 MB), `mtp-gemma-4-26B-A4B-it-Q8_0.gguf`
  (arch `gemma4-assistant`, 4 layers / hidden 1024; in the model dirs).
- **Forward path:** `src/model/gemma4.cpp` (`Gemma4Model`, 1250 lines; MTP head loader/
  forward/loop at `:1054`/`:1139`/`:1188`). MTP head class = `src/model/gemma4_assistant.cpp`.
- **Engine dispatch tag:** `ModelArch::kGemma4`. Wired into `ie run/serve` (`<start_of_turn>`
  chat template, `<end_of_turn>` stop, host-logits bounce like qwen3next). `ie run --spec`
  auto-finds the `mtp-*.gguf` next to the target (or `--spec-head`).
- **Oracle:** `~/llama.cpp/src/models/gemma4.cpp` (+ `gemma4-assistant.cpp` for the head),
  build `fdc3db9b6` (`build-sycl`/`build-vk`/`build-cpu` all have gemma4).
- **One-line standing (2026-06-26):** **Gemma is a BOTH-AXES competitive story** — prefill
  **WINS llama-SYCL at every length** (26B-A4B 4K **2.03×** / 8K **1.91×** / 16K **1.58×**),
  base decode is at its **BW ceiling** (~par-to-0.78× behind, do not chase), and MTP
  self-spec decode is a **lossless win on 31B (1.48× own / 1.12× over llama — RE-VERIFIED
  2026-06-26)**; **26B MTP is 1.46× speed but CONFIRMED NOT lossless (2026-06-26)** — a
  near-tie MoE routing divergence, NOT a claimable lossless win (§5 + regression note).

---

## 1. Envelope (the benchmark contract)

- **Hardware:** 1× Intel Arc Pro B70 (BMG-G31, 32 GB, **608 GB/s** VRAM, 2800 MHz boost).
  Both sizes fit on a single card → all numbers are **single-B70** unless noted.
- **Model + quant:** the official **QAT Q4_0** GGUFs above (the engine's gemma4 loader is
  **Q4_0-only**; Q4_K dense GGUFs do NOT load — `LOAD: attn_q: expected Q4_0`).
- **Target modes:** interactive single-stream — short/mid prefill (≤8K is the headline win),
  long prefill (16K), decode (tg128). Batch serving / warm-prefix not yet characterized.
- **Competitor build + flags:** llama.cpp `fdc3db9b6`.
  - **llama-SYCL (the fair, faster bar for gemma4):**
    `ONEAPI_DEVICE_SELECTOR=level_zero:0 llama-bench -m <gguf> -p T -n 0 -ngl 99`.
  - **llama-Vulkan:** `GGML_VK_VISIBLE_DEVICES=0 llama-bench … -sm none -mg 0`.
    **Single-GPU pinning is mandatory** (un-pinned llama silently splits across both B70s
    + the iGPU and invalidates the comparison).
  - ⚠ Backend choice matters and has DRIFTED: pre-oneAPI-2026.0, SYCL was *crippled* on
    **dense** gemma4 (31B 6 tok/s) → Vulkan was the bar; **post-oneAPI-2026.0 (2026-06-22)
    SYCL is the faster backend for both sizes** (31B tg 23.7, 26B tg 54.1). Quote SYCL as
    llama's best unless re-measured otherwise.
- **Quality gate:** **greedy coherence**, NOT PPL. The QAT-Q4_0 GGUFs have *meaningless
  absolute PPL* — llama's own reference scores this exact 26B GGUF at **~1178** on
  wikitext-2 (QAT preserves argmax but flattens logit magnitudes). The engine-wide quality
  gate stays **crown `ie-perplexity` = 6.4527 bit-exact** (a different, healthy model);
  every gemma change is validated against it + greedy argmax bit-identical across kernel
  variants + adversarial kernel review. See §2.

---

## 2. Correctness

- **PPL gate:** crown `./build/tools/ie-perplexity` = **6.4527 bit-exact** (held across ALL
  gemma work). **Do NOT gate on gemma's own PPL** — it is ~1000–3000 (QAT artifact, llama
  ref ~1178), DEFINITIVELY SETTLED 2026-06-21 (the old "engine 2235 vs llama 883" and "PPL
  beats llama" were BROKEN comparisons — both RETRACTED). Method lesson: never compare our
  streaming-full-context PPL to llama's windowed multi-chunk *average*; match `-c N` chunk[1]
  to streaming `--max-tokens N`. `ie-gemma4-ppl --reprefill` is BROKEN (start_pos=0 growing-T
  → garbage 8750); use streaming-T1 only, treat ±0.4% NLL wobble as harness noise.
- **Reference oracle:** `~/llama.cpp/src/models/gemma4.cpp` (text model loads fine even when
  the vision mmproj fails — the mmproj abort is NOT the text model). `gemma4-assistant.cpp`
  for the MTP head.
- **Deterministic-gen check:** greedy "The capital of France is" → ids `2 818 5279 529 7001
  563` → top-1 `▁Paris` 21.59 → " Paris." (31B: "Eiffel Tower → Paris, France…"). Spec-decode
  losslessness gate = token-for-token == plain greedy at K=4 & K=8.
- **Silent-bug landmines (all implemented + verified; the ones that silently wreck output):**
  1. **RMSNorm DROPPED the (1+w) convention** — plain `_norm(x)*w` (Gemma 1/2/3 used (1+w)!).
     eps 1e-6 fp32. Needed a plain-weight norm variant (the engine's default (1+w) is WRONG here).
  2. **Attention softmax scale = 1.0** (NOT 1/√d; QK-norm carries magnitude). Engine kernels
     bake 1/√HD → **pre-scale Q by √HD** to net 1.0 (`gemma4.cpp:665`, inverse of the Granite trick).
  3. **Per-layer-variable head geometry** — SWA layers HD=256 (kv 16/8); GLOBAL layers HD=**512**
     (kv 4/2). Query n_head=32/16 constant but Q width varies per layer. KV/attn/RoPE are per-layer-dim.
  4. **Pattern 5 sliding : 1 global**, window=1024, global at idx ≡5 mod6 (last layer global).
  5. **Partial RoPE on GLOBAL layers only** (n_rot=128, θ=1e6, `rope_freqs`); SWA full RoPE
     (256 dims, θ=1e4). `gemma4.cpp:666-672`.
  6. **QK-norm** = per-head weighted RMSNorm on Q and K BEFORE RoPE (`:662-663`).
  7. **V** = `wv@cur` if present else = K; then a **WEIGHTLESS** RMSNorm, no RoPE (`:664`).
  8. **Sandwich norms** (attn_norm→attn→attn_post_norm→+resid; ffn_norm→GeGLU→ffn_post_norm→+).
     MoE layer = dual path (shared dense FFN + experts) then OUTER ffn_post_norm.
  9. **MoE router** on un-normed `attn_out`: weightless-rms × (1/√n_embd) × gate-scale → logits
     → softmax-128 → top-8 → renorm. Shared expert = parallel dense FFN added.
  10. **GeGLU `gelu_pytorch_tanh`** (tanh-approx). 11. **Final logit softcap = 30.0**
     (`tanh(l/30)*30`; `NO_SOFTCAP`→94924, confirms required). 12. **Embedding scale ×√n_embd**
     (bf16), tied lm_head. 13. **gemma SPM tokenizer** (non-byte BPE, spaces→U+2581, forces BOS;
     byte-exact vs `llama-tokenize`). Ruled-out non-bugs: shared_kv_layers=0, per_layer_tok_embd=0,
     out_scale absent, layerscale REQUIRED (`NO_LAYERSCALE`→NaN), gate|up order (`SWAP_GATEUP`→Paris vanishes).
- **⚠ 12B DENSE — BROKEN/PARKED (correctness, not perf).** `gemma-4-12B-it-QAT-Q4_0` outputs
  digit-token garbage + ~5× slow. CONFIRMED an engine bug (two independent conversions → byte-
  identical garbage). Ruled out: NaN, geometry, lm_head, embed/attn scale, dtype, rope_freqs.
  Prime suspect = global-attn at **nkv=1 / gqa=16** (12B is the first MQA-on-global arch; 31B
  global nkv=4, 26B nkv=2). **ROOT-CAUSE BLOCKED: no working op-by-op oracle on this box**
  (local llama `fdc3db9b6` cannot load any gemma-4-12B; it is a newer conversion). 26B + 31B
  unaffected. Resume = newer llama build or HF-transformers oracle, op-by-op diff the nkv=1 path.

---

## 3. Pareto frontier (measured matrix, clean-box + dated)

**Bar = llama-SYCL** (its faster backend post-oneAPI-2026.0) unless noted. Single-B70.

### Prefill (tok/s) — **the headline WIN** (26B-A4B, clean-box 2026-06-24/25)

| T | ours | llama-SYCL | llama-Vulkan | verdict | bottleneck class |
|---|---:|---:|---:|:---:|---|
| 4K  | **1642** | 809 | 302 | **WIN 2.03×** | attention (now windowed) |
| 8K  | **1453** | 761 | collapses | **WIN 1.91×** | attention |
| 16K | **1016** | 641 | collapses | **WIN 1.58×** | attention occupancy + MoE/proj fraction |

- Session arc @16K: ~95 (naive) → **1016** (10.7× over our own start; present as engineering
  progress, NOT a competitive ratio). Vulkan collapses past ~2K = the O(T²) signature of not
  windowing the SWA layers — the exact bug we fixed.
- **31B dense prefill** at long ctx is not separately re-tabulated post-attention-levers; the
  attention kernels are per-layer-HD and apply identically. Pre-levers (2026-06-21): 31B pp(T≈477)
  **367 vs llama-Vulkan 203 = 1.81×** ⚠ STALE (pre-windowed-attention, Vulkan bar).

### Decode (tg128, tok/s) — at BW ceiling (clean-box 2026-06-22, SYCL bar)

| model | ours base | llama-SYCL | verdict | + MTP spec | spec vs llama |
|---|---:|---:|:---:|---:|:---:|
| 31B dense | 18.57 | 23.7 | lose 0.78× (base) | **26.55** (1.48×, **LOSSLESS ✓ RE-VERIFIED 06-26**) | **WIN 1.12×** |
| 26B-A4B | 51.53 | 54.1 | ~par 0.95× (base) | 72.83 (1.46×) | **NOT lossless ❌ (06-26 confirmed)** — speed only, do not claim |

- ✅ **RE-VERIFIED 2026-06-26 (clean box, own baseline, K=4, ntok=128):**
  - **31B dense: LOSSLESS ✓ (token-for-token), net 1.48×** (plain 17.95 → spec 26.55, mean accept 2.745).
    The lossless 31B MTP win is **SECURED**.
  - **26B-A4B: confirmed NOT lossless** (plain 50.04 → spec 72.83 = 1.46×, accept 2.953). The per-token
    verify router (`a78e292`, default-ON) does NOT suffice; **disabling batched-verify
    (`IE_GEMMA4_NO_BATCHED_VERIFY=1`) does NOT restore losslessness AND is slower (0.77×)** — so it's a
    genuine near-tie MoE routing divergence, not a projection/flag issue. **Do NOT claim 26B MTP as
    lossless** — a spec-decode that changes outputs is not a usable lossless win; the 26B row is a speed
    figure only, pending a real routing-determinism fix (§ regression note).
  - 27B MTP spec is separately re-verified 0.81× (LOSSLESS but net slowdown — see `qwen35-27b.md`).
- **The "spec vs llama" column = OUR MTP-spec vs llama's BASE (non-spec) SYCL** (26.66 vs 23.7 = 1.12×).
  For context, llama's OWN MTP on B70 is weak (+8% → 20.6 tok/s, 61% accept — its MTP runs the slow Vulkan
  path), i.e. *below* its own SYCL base 23.7 — so our lossless MTP beats both llama-base and llama-MTP. (Don't
  read the 20.6 < 23.7 as a contradiction; it's the backend split.)
- **Note on backend drift:** the 2026-06-21 handoff measured llama-SYCL dense gemma4 at 6 tok/s
  (crippled) → reported 31B base as a 1.33× WIN vs Vulkan 15.35. Post-oneAPI SYCL fixed itself
  (23.7) → 31B base is now a 0.78× LOSS. Use the 2026-06-22 SYCL numbers; the old Vulkan win is STALE.

---

## 4. Bottleneck map

**Decode (BW-bound, at ceiling).** Profiler kernel maps (2026-06-22):
- **Gemma31** (46.6 ms/tok): `gemv_q4_0_soa_multi` **50.5%** · `gemv_q4_0_soa` **27.2%** ·
  rms_norm 8.3% → **77.7% in two Q4_0-SoA GEMVs**. Roofline diagnostic (BW-only XOR-fold variant,
  `baba378`): **~91% memory-bound** — stripping ALL arithmetic gains only **+8.6%** (17.2→18.7).
  Warm gemv already hits **~75–80% of 608 GB/s**. ⇒ no arithmetic/layout headroom left.
- **Gemma26** (13.8 ms/tok): `moe_pfl_proj_q4_0_soa` 36.9% · `gemv_q4_0_soa` 14.1% ·
  `gemv_q4_0_soa_multi` 13.7% · rms_norm 12.9% · `gemv_q6_soa` (lm_head) 10.0%. MoE ~4B-active
  = BW-bound on the Q4_0 expert reads.
- lm_head `gemv_q6_soa` ≈ 57% BW (~6–7% of the step) — the only known decode headroom, but
  **shared with the crown 6.4527 gate** → defer.
- Decode-rate decay over a long run (400 steps → 13.7) = **attention growing O(seq_len)**, NOT
  thermal (GPU pinned 2800 MHz) and NOT the GEMV.

**Prefill (was attention-bound; now WON).** Profiled gemma 16K = **96.3% attention**
(`fa2_tilew_compute`) → that single profile pinned the target and drove the whole prefill
campaign. After the §5 attention levers, the residual at 16K is attention occupancy + the
growing MoE/projection GEMM fraction.

---

## 5. Hypothesis ledger (every lever; dead-ends INCLUDED so they aren't re-tried)

### Prefill — KEPT (the campaign that flipped gemma from losing all 3 lengths to winning all 3)

| lever | mechanism | result | decision | commit / gate |
|---|---|---|---|---|
| Weight-stationary fp16 proj | `dequant_q4_0_to_Bt`→`gemm_fp16_onednn` for dense/attn projections (the dp4a→oneDNN collapse 962ms→~6ms) | prefill **1.96×** | **KEEP** | `b0a8a18`, `IE_GEMMA4_ONEDNN` (opt-out `_NO_ONEDNN`) |
| int-dot W4A8 split-K proj | `gemm_q4_0_q8` split-K (TILE_BLK=96) handles ANY K incl. large-K o-proj (was the per-token `gemv_q4_0` fallback = T launches/layer) | prefill **2.61×** (116→303); prefill-only (T==1 −16%) | **SPECIALIZE** (T>1) | `a036018`/`4d48007`, `IE_GEMMA4_INTDOT_PROJ` |
| oneDNN MoE-GEMM | per-expert `dequant_q4_0_soa_to_Bt`+`gemm_fp16_onednn` (the Coder lever ported; rows/expert=T·K/E=T/16, ≥256 @T≥4096 = oneDNN's large-M regime) | 4K **1.70×** / 8K 1.30× over int-dot SoA | **SPECIALIZE** (T≥4096) | `83b22c4`, `Gemma4Model::moe_onednn_proj`, opt-out `IE_GEMMA4_NO_MOE_XMX` / `_MOE_XMX_MINT` |
| v2 attention (SWA HD≤256) | route HD-256 SWA layers (5/6) through `full_attention_fa2_prefill_v2` (query-row-block KV-SLM tile) | 8K +42% | **SPECIALIZE** (T≥6144) — superseded by tile | `IE_GEMMA4_NO_FA2_V2`, `_FA2_V2_MINT` |
| wide tile kernel (HD 256/512) | `full_attention_fa2_prefill_tile_gemma` = `fa2_tile_wide_impl<HD,Bc>` (per-lane COMPLETE KQ dots, no per-key subgroup reduce; half2 coalesced K/V SLM) — handles the 512 global layers v2 can't | 8K 1.92× / 16K 2.54× over naive | **KEEP** (gate 2048) | `bcd987c`, opt-out `IE_GEMMA4_NO_FA2_TILE`, `_TILE_MINT` |
| REGDOT (register-staged KQ dot) | `<HD,Bc,REGDOT>` 2-parity-acc CPE-chunk dot; the long hd512 dot (256 serial FMAs) is carried-dep-bound | 8K +29% / 16K +27% | **KEEP** (default-on) | `4177a6f`, opt-out `IE_GEMMA4_NO_TILE_REGDOT` |
| SMALLBC (occupancy) | halve Bc (32/16) → SLM 82→49 KB → 1→2-3 WG/Xe-core (128 KB cap) | 8K +54% / 16K +59% | **KEEP** (default-on) | `77b4233`, opt-out `IE_GEMMA4_NO_TILE_SMALLBC` |
| **sliding-window attention** | per-WG out-of-window tile SKIP + per-key window mask on the SWA layers (window=1024). **A CORRECTNESS FIX that is also the biggest perf lever** — full-causal-on-SWA diverged from the oracle at T>1024 | 4K 2.03× / 8K 1.91× / 16K 1.58× **vs llama** | **KEEP** (default-on) | `3de7887`, opt-out `IE_GEMMA4_NO_SWA_WINDOW` |

### Decode — KEPT (then hit the BW wall)

| lever | mechanism | result | decision | commit / gate |
|---|---|---|---|---|
| SoA-Q4_0 W4A8 decode GEMV | `gemv_q4_0_soa_q8` + `repack_q4_0_to_soa` (the AoS block_q4_0 18B-interleaved scale floored decode ~20% BW) | dense decode 36.8→44 | **KEEP** (default-on) | `d3973a9`, opt-out `IE_GEMMA4_NO_Q4_SOA` |
| SoA-Q6 lm_head | `gemv_q6_soa_q8` on tied token_embd | small | **KEEP** | `decd0dc` |
| SoA-Q4_0 MoE experts | `moe_prefill_proj_q4_0_soa_q8` — banks stored SoA *instead* of AoS (no doubling the ~13GB) | decode 48.9→58, prefill 655→708 | **KEEP** | `77ed888`, `IE_GEMMA4_MOE_SOA` (needs FUSED_MOE) |
| NPWG 32→64 + fused multi-bank GEMV | `gemv_q4_0_soa_q8_multi` (q/k/v share attn_norm, gate/up share ffn_norm → 1 quant + 1 launch; tiny GQA k/v cols ride a full grid) | +2–3% (16.9→17.2) | **KEEP** | `baba378`, opt-out `IE_GEMMA4_NO_FUSE_QKV`, `IE_GEMV_NPWG` |

### Decode — DEAD ENDS (measured neutral/negative; **do NOT re-try**)

| lever | why it failed | evidence |
|---|---|---|
| Interleaved-column SoA repack | premise "32 cols K/2 apart = scattered reads" is FALSE — each sub-group already issues a coalesced 256 B SIMD load; 32 independent coalesced streams = good MLP. BW already near warm ceiling | `baba378` roofline; the multi-day repack chases the ~9% ALU envelope at best |
| Cross-arch Q4_K-SoA int-dot decode | 27B decode is **occupancy/latency-bound** (27% occ, 85% XVE-stall), NOT BW-saturated; the Q4 baseline is already ~60% BW so coalescing has no headroom (Q6-SoA's 2.5× came from a terrible 140 GB/s scalar baseline) | benchmark_results_2026-06-22 #1: SoA 14.68–15.85 vs 15.55 baseline = parity at best |
| Native fp16 in gemv hot loop | IGC already lowers it optimally | prior, re-confirmed `baba378` |
| Per-layer `q.wait()` removal | in-order queue already pipelines host submit under GPU compute; dense loop has no mid-layer host reads | NEUTRAL 20.13 vs 20.19; `IE_GEMMA4_NO_LAYER_WAIT` |
| QKV/gate-up launch fusion (for speed) | only +0.7% → launch/inter-kernel overhead is well-hidden ⇒ SYCL-graph replay also won't help | `baba378` (kept the kernel for cleanliness, not speed) |
| uint4 vectorized weight load | box had drifted (swap 1 GB) → no clean signal; likely neutral like fp16 | reverted unmeasured |

### Prefill — DEAD ENDS / SPECIALIZED-AWAY

| lever | why | evidence |
|---|---|---|
| naive XMX `gemm_q4_0_xmx` | per-M-tile re-dequant SLOWER than dp4a at M≈477 (XMX wins large-M; lost to the oneDNN weight-stationary path) | `c7615d8`, opt-in `IE_GEMMA4_XMX` |
| MoE-XMX `moe_prefill_proj_q4_0_xmx` | matrix engine STARVES at ~30 rows/expert (small-M, T≈512); the old "MoE-XMX is a loss 632→420" was THIS + hand-rolled `gemm_fp16` (18–29% peak), NOT oneDNN — a red herring | `6f0fbe9`, opt-in `IE_GEMMA4_MOE_XMX` |
| hd512-v2 attention | generalizing v2 to HD≤512 was CORRECT but has poor occupancy (64 KB SLM → ~2 WG/Xe-core); weakest of {hd256-v2, naive-512, hd512-v2}; the tile kernel supersedes it | REVERTED `6304094` |

### Spec-decode (MTP self-speculative) — the decode lever

| item | mechanism | result | decision | commit / gate |
|---|---|---|---|---|
| de-risk: batched verify GEMV | `gemv_q4_0_soa_q8_batched` — each weight col read ONCE, dotted vs T staged Q8_1 rows; verify T=4 costs 0.83–1.25× a single decode (amortizes on B70, unlike llama's Vulkan GEMM) | bit-identical to T==1 | green-lit the port | `a962423` |
| EAGLE head forward + loop | embed via TARGET tok_embd (×√5376), `pre_proj`(concat)→4 layers Q-ONLY attn SHARING target KV (no wk/wv; SWA head L→target L(n-2), global→L(n-1); `read_attention_gemma` read-only)→`output_norm`→logits + `post_proj`→h_next. NO DeltaNet ⇒ rollback = implicit KV overwrite | **31B LOSSLESS 1.44×→1.12× vs llama**; head CORRECT first build (acceptance ~0.76 = the signal) | **KEEP** | `6f4682a` |
| wire batched verify into `proj()` | **THE perf unlock** — without it verify(T=K) took the slow oneDNN prefill path → **0.18× (6× SLOWDOWN)**; with it → 1.4× | net 1.4× | **KEEP** | `6f4682a`, opt-out `IE_GEMMA4_NO_BATCHED_VERIFY` |
| SoA-Q8 draft head | repack head Q8_0 weights SoA at load | draft round 12.4→8.0 ms | **KEEP** | `cb8bfeb` |
| per-token verify router (26B) | GPU router's batched `gemv_q_T(T)` differs from T==1 decode → near-tie expert flip → ~1 tok/32 divergence; compute router PER-TOKEN in [2,16] = byte-identical to decode | made 26B strict-lossless (then) | **KEEP but REGRESSED** (§ below) | `a78e292`, opt-out `IE_GEMMA4_NO_ROUTER_PERTOK` |
| productize `ie run --spec` | `load_mtp_head` + `mtp_run` + `spec_generate` (draft/verify/accept, emit() streaming+stops); engine kGemma4 spec-dispatch + CLI `--spec-head` | both sizes coherent via product path | **KEEP** | `892934a`, `f40cc1e` |

**⚠ 26B losslessness REGRESSION (open, CONFIRMED 2026-06-26):** the `a78e292` per-token router
made 26B strictly lossless on 2026-06-21, but it **NO LONGER reproduces.** Fresh clean-box re-verify
(2026-06-26, per-token router default-ON): 26B spec **NOT lossless** (1.46×, accept 2.953).
**Disabling batched-verify (`IE_GEMMA4_NO_BATCHED_VERIFY=1`) does NOT restore losslessness AND is
slower (0.77×)** → the divergence is a genuine near-tie MoE expert routing flip (batched-verify ↔
T==1), not a projection/flag artifact. **The 26B 1.46× is a SPEED figure only — do NOT claim it as a
lossless win** until the verify routing is made bit-deterministic vs T==1 decode (the real fix:
make the verify forward's router selection identical to the T==1 path for near-tie experts —
candidates: argmax tie-break stabilization, or running the router strictly per-token in the verify
forward incl. the projection q8 quant). 31B remains **lossless ✓ (1.48× re-verified)**; 27B lossless
(but net 0.81×). This is the one open gemma-MTP correctness item.

**MTP strategic note (B70):** llama's own MTP on B70 = only +8% (20.6 tok/s, 61% accept) — its
Vulkan GEMM can't amortize verify; the "3×/60%" is datacenter-GPU marketing. Net ceiling ≈
accepted/(draft+verify) ≈ 2.7/1.8 ≈ **1.5×**, we're at it (verify(T=4) ≈ 1.7× a decode in the
FULL forward). The 12B and E4B are NOT viable drafts (12B broken; E4B = unsupported matformer).

---

## 6. Shape dispatch (path per T / ctx / dtype + env gates)

All gemma4 weights are Q4_0 (lm_head/token_embd Q6_K). Resident layout is **SoA-only** by
default (`IE_GEMMA4_NO_Q4_SOA` reverts to AoS; `IE_GEMMA4_KEEP_AOS` keeps both for A/B).

**Projections (q/k/v/o, gate/up/down, shared-FFN):**
- **T == 1 (decode):** `gemv_q4_0_soa_q8` (or fused `gemv_q4_0_soa_q8_multi`) — int-dot W4A8 SoA.
- **T in [2,16] (spec verify):** `gemv_q4_0_soa_q8_batched` (opt-out `IE_GEMMA4_NO_BATCHED_VERIFY`)
  — without this verify falls to the slow prefill path (the 0.18× trap).
- **T > 1 (prefill):** `dequant_q4_0_soa_to_Bt`→`gemm_fp16_onednn` (`IE_GEMMA4_ONEDNN`, default-on);
  int-dot split-K `gemm_q4_0_q8` available via `IE_GEMMA4_INTDOT_PROJ`.

**Attention (prefill), by per-layer head_dim:**
- `T ≥ 2048` AND `HD ∈ {256,512}` → `full_attention_fa2_prefill_tile_gemma` (REGDOT + SMALLBC
  default-on; **SWA layers windowed** at 1024 via `IE_GEMMA4_NO_SWA_WINDOW` opt-out, global full-causal).
- `T ≥ 6144` AND `HD ≤ 256` (fallback if tile off) → `full_attention_fa2_prefill_v2`.
- else → `full_attention_gemma` (naive O(T²); also the **T==1 decode** path — fine at ctx ≤ window).

**MoE experts (prefill):** `T ≥ 4096` → `moe_onednn_proj` (oneDNN, `IE_GEMMA4_NO_MOE_XMX` /
`_MOE_XMX_MINT`); else `moe_prefill_proj_q4_0_soa_q8` (int-dot SoA). **Decode** MoE =
`moe_prefill_proj_q4_0_soa_q8` (needs `FUSED_MOE`); **router** computed per-token in spec-verify range.

**Fast config (now default-on when wired via `ie::Engine`; for the tools set explicitly):**
`IE_GEMMA4_FUSED_MOE=1 IE_GEMMA4_INTDOT_PROJ=1 IE_GEMMA4_ONEDNN=1 IE_GEMMA4_MOE_SOA=1`
(26B needs all four + a chat-templated prompt; 31B dense needs only ONEDNN — MoE flags are no-ops).
Footgun: `IE_QUEUE_PROFILING=1` + oneDNN → DEVICE_LOST in profile-decode; use `bench-decode` A/B.

---

## 7. Layout / sched state

- **Resident weights:** SoA-only Q4_0 by default (no AoS device copy; `repack_q4_0_to_soa` builds
  both dense `W.soa_qs` and MoE `gu_qs`/`dn_qs`). Fixes 31B single-card OOM (`[mem-plan] fits 1
  card 17.9/28.7 GB`). MoE expert banks stored SoA instead of AoS — no doubling the ~13GB.
  Open: dedupe dense AoS+SoA (+1.1GB; currently keep both when KEEP_AOS for oneDNN-prefill A/B).
- **Fused launches:** `gemv_q4_0_soa_q8_multi` fuses q/k/v (shared attn_norm) + gate/up (shared
  ffn_norm) into one quant + one multi-bank launch (decode). Prefill MoE = fused gate+up [H,2·EF]
  GEMM + `geglu_rows` then down [EF,H].
- **Host-stall / launch:** in-order queue already pipelines host submit; per-layer `q.wait()`
  drains are dead (NEUTRAL when removed). ~1,400 launches/token are well-hidden (launch fusion
  +0.7% only) → SYCL-graph replay would not help.
- **KV mode:** per-layer variable-dim KV cache (SWA 16/8 × 256, global 4/2 × 512). SWA prefill
  attention is now **truly windowed** (1024); global stays full-causal. Decode attention is still
  naive full O(seq) — correct at ctx ≤ window, a windowed decode op is a long-ctx (>1024) follow-up.
- **MTP head:** shares the target's KV cache (SWA head L→target L(n-2), global→L(n-1)); writes no
  K/V; rollback is implicit (target overwrites stale draft keys by absolute pos — no DeltaNet state).
- **Multi-GPU caveat:** oneDNN is single-card-only (its static ctx binds to card 0 → DEVICE_LOST
  multi-card). Both gemma sizes fit 1×B70 so this is not a gemma blocker (it IS the 80B-prefill blocker).

---

## 8. Open frontier

**Standing verdict (honest):** Gemma is **out of the "loses to llama" bucket** and is the engine's
cleanest *prefill* win (26B 1.58–2.03× at every length) plus a *lossless* 31B decode win via MTP
(1.12×). The remaining axes split into one **real lever** and several **commoditized grinds**:

1. **★ THE most important open item — re-validate / fix the 26B-A4B MTP-spec losslessness
   regression.** *Observation:* `a78e292` (per-token verify router) made 26B strictly lossless on
   2026-06-21, but the 2026-06-22 clean-box found 26B spec **no longer bit-lossless** post-oneAPI
   (near-tie MoE expert routing diverges batched-verify ↔ T==1, deterministic). *Mechanism:* the
   verify path's router logit differs by a hair from the T==1 decode router → flips a near-tie
   top-8 expert → ~1 tok/32 divergence. *Why it matters:* it is the ONLY thing between gemma and a
   defensible *both-sizes, both-axes, lossless* story — without it the 26B 1.39× decode is a
   "fast-but-not-lossless" asterisk. *Falsifier:* re-run `ie-gemma4-spec` token-for-token == plain
   greedy (K=4 & 8) on the current build; if `IE_GEMMA4_NO_ROUTER_PERTOK`-off still diverges, the
   per-token router no longer covers the regression and the divergence source must be re-bisected
   (`HOST_ROUTER=1`→lossless isolates the router GEMM; `NO_FUSED_MOE=1` isolates the expert GEMM).
   **Verdict: a real, scoped, must-do correctness lever (not grind).**

2. **Base decode → BW ceiling (commoditized grind — do NOT chase).** *Observation:* decode is
   ~91% memory-bound (`baba378` roofline), warm gemv at ~75–80% of 608 GB/s, ~10% behind llama
   (31B) / par (26B). *Verdict:* not closable with available levers; a multi-week kernel campaign
   for ~10% on a non-moat arch is the wrong bet. The only untried decode lever (lm_head Q6-SoA
   57→80% BW, ~+1.5%) is shared with the crown 6.4527 gate → defer. **Lead publishing with the
   prefill wins + the lossless MTP decode, not base-decode parity.**

3. **16K prefill (minor real lever).** 26B 16K is already a WIN (1.58×) but the gap is tighter
   than 4-8K (occupancy still 1 WG/core at long ctx + growing MoE/proj GEMM fraction). *Falsifier:*
   push hd256→Bc16 (more WG/Xe-core) or profile the MoE GEMM share at 16K. Low priority — we already win.

4. **12B dense correctness (completeness blocker, not perf).** Parked digit-garbage bug, prime
   suspect nkv=1/gqa=16 global attn; blocked on no op-by-op oracle. *Falsifier:* get a newer llama
   build or HF-transformers gemma4 reference, op-by-op value-diff the global-attn path.

5. **Publish blocker:** no token-exact cross-engine output match yet (llama-cli wouldn't complete
   the long-prompt run); current gemma validation = coherence + adversarial mask review + crown
   invariant. Close one clean Gemma long-prompt byte-match vs llama before publishing.

**Stale/uncertain flags:** 31B base "1.33× decode WIN" (2026-06-21, Vulkan bar) is STALE — SYCL
caught up, it's now 0.78×. 31B "1.81× prefill" is pre-windowed-attention. The 2026-06-25 scorecard
flagged BOTH gemma spec rows "re-verify (own baseline)"; that re-run is now DONE (2026-06-26, 466765c):
31B spec 26.55 tok/s LOSSLESS 1.48x SECURED, 26B spec 72.83 tok/s 1.46x but CONFIRMED NOT lossless.
