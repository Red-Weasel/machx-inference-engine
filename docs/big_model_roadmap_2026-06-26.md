# Big-model roadmap — 120B-class on 2×B70 (2026-06-26)

## UPDATE 2026-06-28 — gpt-oss-120b RUNS on OUR ENGINE (2×B70 tensor-parallel); 20b ships + beats llama
The gpt-oss family was built + optimized end-to-end on **gpt-oss-20b** — it beats llama-SYCL on BOTH
axes (prefill 1.9–4.5×, decode 1.13–1.16×; see MASTER_DEV_PLAN §7 + `docs/public/gptoss_benchmark_2026-06-27.md`).
**✅ gpt-oss-120b now RUNS on OUR engine** via the new MoE tensor-parallel `GptOssTpModel` (`--gpus 2`) —
built this session (NOT the `GptOssSplitModel` layer-split route originally sketched; TP was chosen so BOTH
cards compute). Loads safe (host-RAM spill caps the display card → no crash), **decode numerically correct
(streaming PPL 12.91) + fast (14.6 tok/s > the LM Studio layer-split 12.42)**, multi-shard loader merges the
2-file GGUF. **2 open generation-path bugs block a usable 120b *chat*** (batched prefill @E=128; garbage
after the Harmony prompt — both localized to generation, NOT the PPL-correct forward) → see MASTER_DEV_PLAN
§7 + `project_gptoss_tp_120b_2026-06-28` memory + `docs/public/gptoss_120b_tp_2026-06-27.md` (DRAFT). (The
owner separately ran the same 120b in LM Studio earlier: full MXFP4, 100k+ ctx, 12.42 tok/s, one card maxed.)
Other quant paths still open if wanted: pruned (mradermacher REAP-58B ~41.3 GB — needs a Q4_K expert path,
NOT the MXFP4 TP route); layer-split (meshllm UD-Q4_K_XL); RotorQuant (majentik Q8,
Cl(3,0)-rotor KV); abliterated (noctrex MXFP4 BF16); or a new own method (the research thread below).
OBSERVED only: standard 120b
GGUF quants are ~62.5–65 GB (the MXFP4 experts are ~90% of params; off-the-shelf Q2/IQ2 only shave the ~0.8 GB
attn/embed). On 2×B70 that's ~31 GB/card of weights. The owner is fine with **~90% on VRAM + ~10% spilling to
system RAM** (the fast prefill offsets the spill) — so this is a real, live option, not a wall. Other live paths:
a **pruned** variant (mradermacher REAP-58B ~41.3 GB, fits 2-card), **layer-split** variants
(meshllm UD-Q4_K_XL-layers), **RotorQuant** (majentik Q8_0 ~30-piece, Cl(3,0)-rotor KV compression),
**noctrex Huihui abliterated MXFP4_MOE** (BF16), or a **new own method** (the lead-opened research thread:
a novel quant / KV-compression / expert-tiering format — build piece-by-piece if it helps). All 120b kernels
carry over from the 20b (same family). `GptOssSplitModel` design mapped, ~80% reusable, ready when wanted.
Also fits today on the Q-quant (oneDNN, non-MXFP4) path: Qwen3-Next-80B Q3 38 GB (already run), GLM-4.5-Air IQ2,
Llama-4-Scout Q2. The 120b quality table below is the target.

## ✅ DECISION + STATUS (2026-06-26 night)
**Goal (corrected):** run a **120B-class model on 64 GB**. The EXL3-revisit ask was about *fitting a larger
model* via low-bpw (not the 80B at 4.5 bpw). **Decision: build the gpt-oss family** (8th arch, novel: attention
sinks), prototype on **gpt-oss-20b** (downloaded, 12 GB). MXFP4 dequant SHIPPED (`1ee4a2e`); kGptOss STEP 0 done
(`bf70cdd`); loader/forward next. See `gptoss_arch_spec_2026-06-26.md` + MASTER_DEV_PLAN §7.

**Quality (real benchmarks):**
| | gpt-oss-120b | gpt-oss-20b | Nemotron-70B |
|---|---|---|---|
| MMLU | 90.0 | **85.3** | 83.0 |
| GPQA | 80.1 | 71.5 | — |
| AIME'25 | 97.9 | **98.7** | — |
| HumanEval/LCB | 88.3 | LCB 70 | tier-match |
| Arena-Hard (chat) | — | — | **85.0** |

**64 GB fit (the deciding axis — KV is *cheap* here: GQA 8-kv + half-windowed):**
- **gpt-oss-120b** ~63 GB → ~0.5–1 GB headroom → **~4–8 K context, brittle** (a model-weight squeeze, not a
  context one). The quality ceiling, but tight. The 120B fit levers: **EXL3 ~3 bpw (~45 GB, fits clean)**, a
  low-bpw GGUF, or slight system-RAM overflow (lose some speed).
- **gpt-oss-20b** ~12 GB → **~50 GB free, 32 K+ context, fast** (3.6B active). Beats Nemotron-70B on MMLU/AIME at
  ⅓ the size → the 64 GB **sweet spot**.
- **Nemotron-70B** Q4 ~42 GB → ~20 GB free, comfortable 8–16 K, dense (slower), 2024-era. The zero-code drop-in.

**Housekeeping done:** EXL3-80B (43 GB) archived → `/media/weezy/Data/ai-models-archive/Qwen3-Next-80B-Instruct-exl3-4.51bpw/`
(intact). gpt-oss-20b-mxfp4.gguf downloaded → `~/models/gpt-oss-20b-GGUF/`. NVMe **77 GB free** (room for a 120B).
Left in place: unsloth-mtp (tool ref), Momix-44 (80B test model), crown gate GGUF, OBLITERATUS/OpenYourMind
(crown variants — archive on request).

---

Decision brief from a 3-lane agent survey (gpt-oss / Nemotron / others+EXL3). Target box:
2×Intel Arc Pro B70 = 64 GB VRAM (32 GB/card, layer-split, NO P2P). Usable budget per the
GPU-first rule ≈ ≤60 GB for the model so KV + workspace fit.

Engine paths available to ride: `DenseModel/kLlama3` (Llama/Mistral/Phi/Granite), `qwen3moe`
(top-k MoE), `qwen3next` (hybrid gated-DeltaNet + MoE, 2-card split), EXL3 (importer+forward,
qwen3next-validated). **oneDNN multi-card landed 2026-06-26** → multi-card MoE prefill now wins.

## Ranked candidates

| Model | Params (act) | Arch novelty vs us | Fits 2×B70 | Engine path | Effort | Verdict |
|---|---|---|---|---|---|---|
| **gpt-oss-120b** | 117B (5.1B, top-4) | **NEW**: attention **sinks** + alternating SWA/dense + clamped gated-SwiGLU + biases + MXFP4 | Q4_K ~63GB (tight) / Q3_K ~62GB | **NEW arch** (closest base = qwen3moe) | **L** | **The flagship the user named + most differentiated ("first on Arc"). Prototype on gpt-oss-20b (same arch, 12GB).** |
| Llama-3.1-Nemotron-70B | 70B dense | none (plain Llama-3.1) | Q4_K ~38GB (easy) | `kLlama3` **drop-in** | **S** | Surest quick "it runs" but 70B not 120B, arch already supported → low novelty. |
| Mistral-Large-2411 (123B) | 123B **dense** | none (Llama-family + tekken, we have it from Mistral-Small wave) | Q3_K ~52GB (Q4_K ~70GB OVERFLOWS) | `kLlama3` + tekken | **S–M** | True 120B-class that could RIDE the existing dense path; needs Q3_K (quality hit) to fit. Verify dense-vs-MoE. |
| GLM-4.5-Air | 106B (17.8B, top-2) | MoE top-2 (reuse qwen3moe router) + **new attn-proj kernel** | Q4_K ~60GB (tight) | qwen3moe-ish + new attn | **M** | High community interest; SYCL path unverified on B70. |
| Nemotron-H-120B | 120B hybrid | **Mamba2** SSM + per-layer hybrid (NOT our DeltaNet) | Q3_K ~45GB | **NEW arch** (Mamba2 path) | **L** | True 120B hybrid; needs a Mamba2 kernel; SYCL coverage uncertain. Frontier. |
| Qwen3-235B (unreleased) | 235B (~100B) | our arch upscaled | ❌ Q3_K ~88GB > 64GB | `qwen3next` direct | S (when it fits) | Zero-effort reuse BUT doesn't fit at sane GGUF quant; only via EXL3 3bpw. Watch for release. |
| EXL3-revisit | n/a (quant) | trellis/Hadamard (ALU-friendly, Arc-good) | 3.0bpw: 100B→~37GB (1-card!) | importer+forward DONE | M (load) / L (encode) | **Defer** unless (a) a pre-quantized EXL3 GGUF of a big model appears, or (b) we pivot to 1-card + long-ctx (3bpw frees a whole card for KV). Q4_K proven faster at 2-card. |

## Recommendation (autonomous pick: gpt-oss)

**Primary = gpt-oss** — it's the model the user named, the most architecturally novel (attention
sinks are in NO other arch we support), and on-brand for the "first/fastest on a brand-new arch on
Arc" strategy. Prototype on **gpt-oss-20b** (identical arch, ~12 GB Q4_K, 1-card, fast iterate),
then the 120B is a load-only scale-up.

**The new pieces to build (vs qwen3moe, which we already have):**
1. **Attention sinks** — a per-head learned scalar added into the softmax denominator (a "virtual"
   always-attended key with no value). Modifies the FA softmax normalization only.
2. **Alternating attention** — even/odd layers swap sliding-window (banded, e.g. 128) vs full dense.
   We already have SWA windowing from the Gemma work (`IE_GEMMA4_NO_SWA_WINDOW` lever) → reuse.
3. **top-k = 4** MoE (vs qwen3moe's 8) — a config value; router + scatter/gather reuse.
4. **Clamped gated-SwiGLU** + **biases** on attn-out/FFN/router — small kernel tweaks.
5. **Quant**: use a **Q4_K / Q6_K** GGUF to avoid MXFP4 entirely (the engine's Q4_K/Q6_K dequant
   handles the re-quantized experts). MXFP4-native is a later optimization, not a blocker.

**Secondary / fallback:** if gpt-oss stalls, **Llama-3.1-Nemotron-70B** is a same-day drop-in
validation (kLlama3), and **Mistral-Large-123B Q3_K** is the lowest-friction *true* 120B (rides the
dense path if it's dense — verify).

## ⚠ Disk is the first blocker (95% full, 46 GB free, 2026-06-26)
A 120B GGUF (≈60–63 GB) does NOT fit in 46 GB free. gpt-oss-**20b** (12 GB) does. To run any
**120B**, free space first. Biggest reclaim candidate (report only — user's call, NOT auto-deleted):
- `~/models/Qwen3-Next-80B-Instruct-exl3-4.51bpw` — **43 GB**, EXL3 80B. EXL3 is SHELVED (the
  Q4_K_M GGUF beat it at the same size + speed). Freeing it → ~89 GB free = a 120B fits.
- `~/models/unsloth-mtp` (22 GB, the MTP GGUF that's quant-incompatible per the Crown-MTP block) —
  secondary candidate; verify before removing.
- OBLITERATUS (21 GB) / OpenYourMind (26 GB) are crown variants — leave unless you don't use them.

## Concrete next steps (per pick)
- **gpt-oss:** FULLY SCOPED → `docs/gptoss_arch_spec_2026-06-26.md` (arch string `gpt-oss`, all
  tensor names + the sinks/SWA/MoE math + the 6-step build checklist). Open decision = the MXFP4 quant
  (engine has no FP4 path) — resolve which non-MXFP4 gpt-oss-20b GGUF to grab, then execute the checklist
  (`kGptOss` register → loader → sinks-FA + alternating SWA + top-4 MoE). Needs disk freed for the 120b.
- **Nemotron-70B:** `hf download` Q4_K_M GGUF → `ie run --gpus 2` (should ride kLlama3 as-is) →
  validate "Paris" + PPL + bench vs llama.
- **Mistral-Large:** verify dense vs MoE first; if dense, Q3_K GGUF → kLlama3 + tekken → run.
