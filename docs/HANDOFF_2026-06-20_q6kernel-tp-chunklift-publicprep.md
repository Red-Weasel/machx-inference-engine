# Session handoff — 2026-06-20: Q6 kernel + hybrid-TP + chunk-lift + public-prep

Read `MASTER_DEV_PLAN.md` banner first. This is the session continuation guide. Strategic frame:
**going public, Intel-facing** — see memory `reference_b70_competitive_landscape`.

## What shipped this session (all committed, gated, on `main`)

1. **⭐ Q6_K fast decode kernel (`gemv_q6_soa`)** — `6292b4c`/`233a55d`/`b218d20`/`3a8b343`.
   27B dense Q6 single-GPU decode **5.0 → 12.8 tok/s (2.5×)**. New `src/ops/gemv_q6_soa.cpp`:
   load-time SoA-Q6 repack (stays ~6.5 bpw / ~22 GB, single-card) + int-dot W6A8 split-K kernel,
   mirroring the Q8-SoA template. Default-ON, opt-out `IE_QWEN35_NO_Q6_SOA=1`. Validated: crown PPL
   6.4527 bit-exact, 27B Q6 PPL Δ0.00002% vs AoS oracle, "Paris" coherent. **Single-card Q6 now ≈
   2-card Q8 split (~13) → daily driver works on ONE card.** Kernel is ALU-bound at 57% BW (headroom
   to ~17 if a cheaper 6-bit unpack / different layout is found — follow-on). Spec:
   `docs/superpowers/specs/2026-06-20-q6k-fast-decode-kernel.md`.

2. **Q6_K ssm-proj loader fix** — `60e452c`. `dequant_q6_K_row` + Q6_K branch in
   `upload_f32_proj_fp16` (qwen35_dense + qwen35_split, copy-not-hoist). Unblocks `*-Q6_K` GGUFs
   (was hard-failing `ssm proj: expected F32 or Q8_0`).

3. **Hybrid tensor-parallel (Phase 0) → measured NO-GO** — `e622a3a`/`88db8de`/`8b3a293`. New additive
   `Qwen35TpModel` (`IE_QWEN35_TP=1`, off by default). FFN-slice TP decodes **12.6 ≤ ~13 baseline**
   because the no-P2P all-reduce is **~32% of decode** (≈ the FFN it parallelizes). Per the spec
   gate, STOPPED at Phase 0 — did not build the DeltaNet v-head shard. **TP is dead on this no-P2P
   board**; faster-27B levers are the shipped Q6 kernel + spec-decode. Kept as a measured negative +
   reusable scaffold. Spec: `docs/superpowers/specs/2026-06-20-hybrid-tp-SCOPING.md` §7.

4. **✅ Prefill chunk RAISED 256→512 for crown + 27B** — `34480a0`. **The §1 BMG DeltaNet
   non-determinism is NOT REPRODUCIBLE on our stack.** After NEO 26.14→**26.18** (user ran the apt;
   GuC `bmg_guc_70.bin` Apr-26 + kernel `6.17.0-35` UNCHANGED = the exact diagnosis stack), the bug
   did not fire: `ie-bug-monitor` 1024-iter ×3 (24 chains) = 0 divergence on BOTH drivers; crown PPL
   `--prefill-chunk 512` ×3 bit-identical (16.27, no collapse); 27B-Q6 ×3 bit-identical (15.68);
   crown `ie run` 563-tok → 701.9 tok/s prefill coherent. Crown (`kQwen35Moe`) + 27B (`kQwen35Dense`)
   now T≤512 in `Engine::generate` (env-revert `IE_QWEN35_PREFILL_CHUNK`). **Closes the long-context/
   pp512 weakness** (~1.1× long-prompt prefill). NOT "26.18 fixed it" (clean on 26.14 too) — "no
   longer reproducible." Workspace self-grows (qwen36:797 / qwen35_dense:550), so no load-time change.

5. **CRITICAL CORRECTION: the "81→13 engine-wide regression" was a MEASUREMENT ARTIFACT.** Crown
   re-benched clean = **80.4 tok/s** (= hist 81, never regressed). All models at expected speed:
   crown 80.4, coder-next ~47-50, 27B dense ~13 (correct — dense reads 28.6 GB/tok). The prior
   session's bad `ie run`/under-warmed numbers snowballed into a false narrative. Memory + dev-plan
   corrected. LESSON: re-bench the SAME model with `ie-bench` + warmup before declaring a regression.

6. **B70 competitive research** (memory `reference_b70_competitive_landscape`): on hybrid DeltaNet
   arches every competitor is weakest on B70 (Vulkan can't run them, OpenVINO preview-only/no-B70
   numbers, llama.cpp SYCL + Intel's llm-scaler cap ~43-55) and WE BEAT ALL (35B ~80, coder-next-80B
   ~50 incl beating Intel's own llm-scaler). ⚠ HONEST GAP: dense 27B (llama SYCL 22.5 vs our ~13 —
   but that's Qwen3.5-27B at unknown quant; NOT apples-to-apples — verify). Windows: decode-on-
   DeltaNet > day-one > multi-GPU-without-P2P > single-binary/multi-format > (long-ctx now improving).

## Hardware / stack truth (owner-confirmed + measured)
- 2× B70 = **608 GB/s VRAM each**, ProArt board **PCIe Gen4 x8/x8** (sysfs idle-downtrains to x1 —
  ignore). PCIe is NOT the layer-split decode bottleneck (cross-card ~10 KB/tok); TP's all-reduce is.
- **NEO 26.18.38308.4** (updated from 26.14), IGC 2.32.7, GuC bmg_70 (Apr-26), kernel 6.17.0-35.
- 30 GB RAM → 21 GB+ models don't stay page-cached (slow 2nd loads). cyber-Q8 27B is on a spinning
  HDD (~74 MB/s) → ~8-10 min cold load.

## Gates / invariants held
Crown `ie-perplexity` **6.4527 bit-exact** across every commit. All changes additive; crown /
`Qwen35DenseModel` / `Qwen35SplitModel` / `DenseModelTP` / `gemv_q8_0_soa_q8` UNTOUCHED. Binary
`build/src/ie`, bench `build/tools/ie-bench`.

## OPEN THREADS (priority order, for next session)
1. **Spec-decode (#3 of the 3 builds)** — the chosen ~2× LOSSLESS decode lever, stacks under
   everything; TP is dead so this is the main remaining decode win. Spec:
   `docs/superpowers/specs/2026-06-20-qwen35-27b-speculative-decoding.md`. Drafters now fast (post Q6
   kernel + profiling fix) — re-bench, pick winner, abliterate.
2. **Benchmark-matrix prep for public** — (a) update **llama.cpp** (stale at `fdc3db9b6`, 06-11) +
   rebuild SYCL; (b) **dense 27B-Q4 apples-to-apples**: our engine vs llama on the SAME
   `Qwen3.6-27B-Q4_K_M` (we have it, bartowski 06-11) — settles the dense-parity question. If our Q4
   is slow, `gemv_q4_soa` mirrors the Q6 win AND speeds every 4B/8B/Llama Q4 dense (big reusable).
   (c) Re-pull Qwen models if updated (owner said they were — don't delete the crown lmstudio dir).
   (d) Add OpenVINO GenAI as a matrix baseline (novel-arch coverage = the headline).
3. **Q4_K dense re-bench** — the "4B at 2.55 tok/s" was a profiling artifact; never re-benched clean.
   Quick `ie-bench` on a 4B Q4_K to know if the generic dense path is healthy.
4. **T≥1024 chunk validation** — 1024 looked clean (1 run, no collapse) but needs the repeatability
   treatment before lifting crown/27B beyond 512. Would further close the long-prefill gap.
5. **Public-readiness cleanup** — (a) profiling-dependent tools (`ie-bug-monitor`) CRASH without
   `IE_QUEUE_PROFILING=1` since the profiling-opt-in change — make them self-enable or not require it.
   (b) gitignore `.claude/agent-memory/` (agent scratch, currently untracked/modified in the tree).
6. **Q6 kernel headroom** (optional) — 57%→~80% BW would take 12.8→~17; needs a cheaper 6-bit unpack.

## Method notes (carry forward)
- `ie run` is JIT-noisy and was NOT the perf path — use `ie-bench` (warmup) / `ie-perplexity` for
  numbers. `ie-bench` is single-GPU only (no `--gpus N`) → 2-GPU split benched via `IE_QWEN35_PROFILE`.
- GPU is a serial resource — ONE workload at a time (exclusivity rule); driver updates need GPU idle.
- One specialist agent per build, gated (crown PPL bit-exact + cosine/PPL + before/after bench), then
  owner-verify before trusting. Worked well this session (Q6 kernel + TP both clean-reported).
