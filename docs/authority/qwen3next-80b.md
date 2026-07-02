# Qwen3-Next-80B-A3B — Living Authority Doc (`qwen3next`)

**Arch family:** `qwen3next` (the crown `qwen35moe` family scaled to 80B — gated DeltaNet
+ gated full-attn + 512-expert MoE + shared expert). **Owner file:** `src/model/qwen3next.cpp`
(`Qwen3NextModel`, its own TU; crown/27B/qwen3moe untouched). **Last updated:** 2026-06-26.

> ⚠️ **HEADLINE STANDING WAS CORRECTED 2026-06-25 (`ec4ec6d`).** The famous "DECODE 1.40× WIN"
> (51.8 vs 37.1, 2026-06-13) is **STALE/DEAD** — it was measured against an OLD llama-SYCL @37 tg.
> Fresh same-box on-GPU head-to-head: **80B now LOSES both axes vs current llama-SYCL** (decode
> 56.6 vs 60.8 = 0.93×; prefill 311 vs 625 = 0.50×). Do **not** quote the 1.40×. The real,
> durable standing is **first-mover GPU on Intel Arc** (we GPU-run it where Vulkan/OpenVINO can't)
> — not a single-stream perf win over llama-SYCL. See §3/§8.

---

## §0 Identity

| Field | Value |
|---|---|
| Family | `qwen3next` — `ModelArch::kQwen3Next`; llama groups `LLM_ARCH_QWEN3NEXT` with `QWEN35MOE` |
| Checkpoint | Qwen3-Next-80B-A3B-Instruct |
| GGUF (daily driver) | `~/models/Qwen3-Next-80B-GGUF/Qwen_Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf` (46 GB, bartowski) |
| GGUF (alt, abliterated) | `Momix-44` Q4_K_M — abliterated-CHAT variant (a DIFFERENT model). **NOT the perf-H2H file:** the §3 head-to-head ran a SINGLE Q4_K_M on BOTH engines (apples-to-apples, §1), never a cross-model compare. |
| EXL3 (niche, shelved) | `~/models/exl3-80b/qwen3next-80b-exl3.gguf` (4.51bpw, 43 GB, imported from turboderp exl3) |
| Forward path | `src/model/qwen3next.cpp` → `Qwen3NextModel::{load,forward}` (additive; own TU) |
| Loader/config | `read_qwen3next_config` (`src/loaders/model_config.cpp:279`); `Qwen3NextConfig` (`include/ie/model_config.hpp:179`) |
| Engine dispatch | `kQwen3Next` branch in `engine.cpp` → `next_model_` / `next_` flag; **`ie run/serve --gpus 2`** |
| Tools | `ie-qwen3next-{gen,ppl,bench}`, `ie-qwen3next-load-test`, `ie-qwen3next-config-test` |
| Topology | **2×B70 layer-split mandatory** (46 GB Q4 does NOT fit one 32 GB card; ~22.6 GB/card) |
| One-line standing | Forward correct + wired + first-mover-on-GPU; **single-stream perf LOSES both axes vs current llama-SYCL** (decode 0.93×, prefill 0.50×). The win is structural (day-one + competitors can't run it), not raw tok/s. |

**Config (`Qwen3NextConfig`, validated `ie-qwen3next-config-test` "ALL 48 layers OK"):**
48 layers · hidden **2048** · vocab 151936 · eps 1e-6.
- **Full-attn layers:** n_q **16**, n_kv **2** (GQA=8), head_dim **256**; joint Q|gate `attn_q` [2048,8192]
  (=2·16·256); `attn_k`/`attn_v` Q8_0 [2048,512]; per-head Q/K RMSNorm [256]; **NEOX partial RoPE n_rot 64, θ=1e7**; post-attn sigmoid gate.
- **DeltaNet (linear) layers:** ssm_n_k_heads **16**, ssm_n_v_heads **32** (n_v; 27B was 48), ssm_state **128**,
  ssm_inner **4096**, conv_ch **8192**, conv_kernel 4; **fused `ssm_ba` [2048,64]=[β(32)|α(32)] per-k-head interleaved**.
- **MoE (every layer):** E **512**, K **10**, expert_ffn (EF) **512** (= the int-dot base case);
  `ffn_{gate,up}_exps` Q4_K, `ffn_down_exps` **MIXED Q6_K/Q4_K per layer**; F32 router.
- **Shared expert (every layer):** `ffn_gate_inp_shexp` F32 sigmoid gate + `ffn_{gate,up,down}_shexp` Q8_0 (shared_expert_ffn 512).

---

## §1 Envelope (benchmark contract)

| Axis | Contract |
|---|---|
| Hardware | 2× Intel Arc Pro B70 (BMG-G31, 32 GB, 608 GB/s each), PCIe Gen4 x8/x8, **no P2P** (HW-blocked on 2×B70) |
| Model + quant | Qwen3-Next-80B-A3B-Instruct **Q4_K_M** (46 GB), layer-split 24/24 across 2 cards |
| KV / state | per-card hybrid: `KvCache` (full-attn layers) + `DeltaNetState` (linear layers); fp16 KV |
| Target modes | **decode tg128** (primary, the BW-bound battlefield) · **prefill pp≤512** (chunk-capped) · multi-turn chat (`ie run/serve --gpus 2`) |
| Competitor build | llama.cpp `-DGGML_SYCL=ON` icpx 2026.0 @ `fdc3db9b6`; its SYCL backend HAS gated_delta_net/ssm_conv/ssm_scan GPU ops → runs qwen3next ON GPU (Vulkan does NOT — CPU fallback). Same 46 GB GGUF, both 2×B70 `-sm layer`, same session. |
| Competitor flags | `llama-bench -m <gguf> -sm layer -ngl 99 -p T -n 128` |
| Quality gate | Crown PPL `6.4527` bit-exact (all qwen3next work isolated to its TU); qwen3next PPL `4.7282` (256 tok, `ie-qwen3next-ppl`) |
| Box discipline | one GPU workload at a time (2×B70 both occupied by one model); order-controlled A/B; discard first JIT run |

**The honest competitive frame (`reference_b70_competitive_landscape`):** on DeltaNet arches
every *other* competitor is weakest on B70 — Vulkan can't run them (no ssm shaders → CPU corruption),
OpenVINO is preview-only (no B70 number), Intel llm-scaler is throughput-not-latency. **llama-SYCL is
the only single-stream opponent** — and as of 2026-06-25 it ties/leads us on 80B decode and doubles us
on long prefill. So the defensible claim is **"day-one, runs-on-GPU-where-others-can't"**, plus the
*Crown* prefill win (1.14×) — NOT an 80B perf win.

---

## §2 Correctness

**PPL gate (crown, global):** `./build/tools/ie-perplexity` → **6.4527 / NLL 1.864495** bit-exact.
Held across every qwen3next milestone (separate TU, additive).

**Arch PPL:** `ie-qwen3next-ppl` (streaming T=1, commit `022b26f`) → **4.7282** (avg NLL 1.553554, 256 tok) —
healthy/low. `--sweep --repeats 25` (651-tok corpus) = **25/25 bit-identical at chunk-256 AND chunk-512**
(the empirical basis for the T≤512 cap, §6).

**Reference oracle:** `docs/qwen3next_oracle_prefill_capital-of-france.log` — llama.cpp **Vulkan** full
op-by-op trace (5012 ops, all 48 layers, prompt "The capital of France is"). ⚠️ **CPU-VALUE oracle only**:
llama-Vulkan has no qwen3next ssm ops → it ran on CPU (both B70 VRAM stayed empty). Correctness ground-truth,
**not** a GPU perf reference. Diffing probes (now stripped): `IE_QWEN3NEXT_DBG=<L>`, `IE_QWEN3NEXT_DBG_VALS=1`,
route-dump + `ie-inspect --layer`.

**Deterministic-gen check:** greedy "The capital of France is" → **" Paris. The capital of Italy is Rome"**;
cosine/sum sweep all 48 layers within **0.998–1.007** of the oracle (L47 ratio 5.4 = llama's known
last-token-only artifact, not a bug). Wired into Engine: `echo "What is the capital of France?" | ie run <gguf> --gpus 2` → "The capital of France is Paris."

**Silent-bug landmines (all found by op-by-op VALUE diff vs the oracle — these are the 4 forward bugs):**
1. **`ffn_down_exps` dtype is MIXED Q6_K/Q4_K per layer.** Dispatching the Q6_K down kernel
   unconditionally misreads 144-byte Q4_K blocks as 210-byte Q6_K → garbage scales → `moe_out`→Inf→NaN
   from blk.6 on. **FIX: branch down on `lw.down_dt`** (loader captured it; forward ignored it). `402786a`.
   The prior "int-dot `_gen` down overflows at E_ffn=512" was a **MISDIAGNOSIS** — the kernel was never wrong.
2. **Fused `ssm_ba` split is PER-K-HEAD INTERLEAVED, not two contiguous halves.** Oracle reshapes to
   `{4, SKH=16}`, views within each group as `[β(2)|α(2)]` (β FIRST). A contiguous `[β(32)|α(32)]` slice
   scrambles heads. FIX: `split_q_gate_per_head(SKH, SVH/SKH)`. `d1afa7a`.
3. **DeltaNet q/k repeat 16→32 must INTERLEAVE** (`h_in = h_out/repeat`), not the 27B's tile
   (`h_in = h_out%n_in`). Added `bool interleave` to `repeat_interleave_heads` (default tile = 27B-safe;
   qwen3next passes `interleave=true`). `e7319f7`. After this, L0 `x(out)` is oracle-bit-exact.
4. **`moe_y` is NOT zeroed before `moe_prefill_reduce`** — the reduce ACCUMULATES. qwen3moe pre-writes y
   via shexp, but qwen3next adds the shared expert AFTER the reduce → stale-y carried → L1+ MoE ~2.66×
   inflated (L0 fine). FIX: `q.memset(w.moe_y, 0, …)` before reduce (`qwen3next.cpp:918`/`831`/`843`/`935`). `1ed07e0`.
5. **NEOX-rope over-contribution was a refuted red herring** — k(rope) matched the oracle (121.2 vs 121.8);
   do not chase rope again.

---

## §3 Pareto frontier (measured matrix)

### CURRENT — RE-VERIFIED 2026-06-25 (`ec4ec6d`), clean box, llama on-GPU verified

| Regime | T/ctx | ours | llama-SYCL | result | bottleneck class |
|---|---|---|---|---|---|
| **Decode** | tg128 | **56.6** | **60.8** | **LOSE 0.93×** | weight-BW (GEMVs ~70% of decode) |
| **Prefill** | pp512 | **311** | **625** | **LOSE 0.50×** | MoE-GEMM kernel gap (2-card int-dot, no oneDNN) + T≤256 chunk legacy |
| **Prefill (long-ctx)** | pp8413 | 443→**498** | — | **+12.5% self** (tile lever) | MoE-GEMM-bound residual |

> **Long-ctx tile lever (2026-06-26, `IE_QWEN3NEXT_NO_FA2_TILE` opt-out, gate ctx≥6144):** ported the
> proven Gemma hd256 wide-tile kernel (same lever as 27B `07ab8ef` WIN / crown `bf73093` +57%) to the
> 80B's head_dim-256 full-attn prefill (`qwen3next.cpp`, multi-card-safe SIMD — no oneDNN landmine).
> **Correctness PERFECT: bit-identical generated tokens vs naive** (" Paris. The capital of Germany is
> Berlin…", identical ids). **Self +12.5% @8.4K** — the SMALLEST of the three ports because the 80B is
> the most MoE-GEMM-dominated (the tile recovers only the attention fraction; the MoE prefill GEMM is
> the dominant residual, see §4). Confirms the 80B prefill loss is MoE-bound, not attention-bound. A
> straight per-device oneDNN MoE-GEMM port is FALSIFIED (`c96909a`): at the T<=512 chunk the MoE is
> small-M (rows/expert = T*K/E ~= 10-16), below oneDNN's large-rows/expert regime -> no win. The 80B
> chunked-prefill loss is structural at this chunk; the tile recovers only the attention fraction.

> These supersede everything below. The "1.40×" decode and "≈parity pp512" prefill numbers from
> 2026-06-13/14 were against an **older, slower llama-SYCL**; current llama-SYCL caught up on decode
> and was always ahead on a single 512-prefill (we chunk).

### STALE — 2026-06-13/14 (`c0338f4`, `bbf60c5`) — kept for history, DO NOT quote competitively

| Regime | T/ctx | ours | llama-SYCL (old, ~slow) | "result" (stale) |
|---|---|---|---|---|
| Decode | tg128 | 51.8 | 37.1 ± 0.2 | ~~1.40× WIN~~ (vs OLD llama) |
| Prefill | pp128 | 336 (@~83 tok) | 334.4 ± 19 | ~parity |
| Prefill | pp256 | 513 (@208 tok) | 473.9 ± 10 | ~ahead |
| Prefill | pp512 | 566 (chunked) / 617 (single-512) | 622.2 ± 7 | 0.91× / ~tie |

5-prompt suite (`ie-qwen3next-bench`, 2×B70, int-dot, decode=128): median **pp 336 / tg 51.8** (the
51.8 was clean/text-independent at the time; the *competitive ratio* is what went stale, not our absolute).

### EXL3 (4.51bpw, niche path — SHELVED)
80B EXL3 coherent on 2×B70 (`80eaeef`); fused/batched EXL3 MoE = **decode 1.46× / prefill 1.78×** vs the
slow EXL3 loop (`01f0a7b`). ⚠️ This is EXL3-vs-EXL3, NOT vs the Q4 fast path. EXL3 at 4.5bpw/2-card buys
nothing over the Q4_K_M GGUF (same size, fast path faster) → shelved; kept only for a future 3bpw/1-card niche.

---

## §4 Bottleneck map

**Decode (the primary regime) — bandwidth-bound, near the memory wall.**
The DeltaNet-family decode profile (Crown-measured, shared structure): **GEMVs ~70%** of decode,
**DeltaNet recurrence only ~3.4%**, **lm_head already ~63% of peak BW**. Both engines hit the same
608 GB/s × 2-card wall → we are parity-to-7%-behind. There is **no measured per-phase % breakdown for
the 80B specifically in the sources** — the `IE_QWEN3NEXT_PROFILE` phase profiler exists (`3af75fb`:
attn/deltanet · router gemm+sync · router HOST cpu · MoE compute · shared expert · other) but its 80B
numbers were not recorded as percentages. **GAP: capture a clean 80B decode roofline before the next lever.**

**Prefill — MoE-GEMM kernel-bound + chunk-capped.** Two structural causes of the 0.50×:
1. **oneDNN is single-card-only** — its static ctx binds to card 0 → DEVICE_LOST on any multi-card path
   (the documented landmine, §7). The Coder/Gemma prefill win (oneDNN MoE GEMMs) therefore **cannot reach
   the 2-card 80B**; the 80B MoE prefill is stuck on the slower int-dot/dp4a path. And even with a
   per-device ctx the chunked-prefill MoE is small-M (rows/expert ~= 10-16, below oneDNN's regime), so
   oneDNN buys nothing here (`c96909a`) — the loss is structural at this chunk, not just the landmine.
2. **DeltaNet T≤256→512 chunk cap** — caps a single long prefill vs llama's one-shot 512. Measured
   **marginal (+2.5%)** when relaxed → NOT the dominant lever (the MoE-GEMM kernel is).

Roofline label: **decode = memory-bound (near ceiling)**; **prefill = compute/kernel-bound (MoE-GEMM gap)**.

---

## §5 Hypothesis ledger (every lever — KEEP / REVERT / SPECIALIZE / DEFER)

| # | Lever | Observation → mechanism | Result | Commit / gate |
|---|---|---|---|---|
| 1 | **Layer-split fleet loader** | 46 GB > 32 GB/card → must split 24/24, per-card hybrid caches | **KEEP** (the enabling capability) | `3ff1f5a`→`9064e8a`; load-test "22.60/22.68 GB" |
| 2 | **`down_dt`-branched MoE down** | GGUF mixes Q6_K/Q4_K ffn_down per layer → unconditional Q6_K = NaN | **KEEP** (correctness) | `402786a` |
| 3 | **`ssm_ba` per-k-head interleaved split** | β/α interleaved within `{4,SKH}` groups, not contiguous | **KEEP** (correctness) | `d1afa7a` |
| 4 | **q/k repeat INTERLEAVE (16→32)** | ggml REPEAT is `h/repeat`, not tile | **KEEP** (correctness; default stays tile for 27B) | `e7319f7` |
| 5 | **`moe_y` memset before reduce** | reduce accumulates; shexp added after → stale carry | **KEEP** (correctness) | `1ed07e0` |
| 6 | **int-dot W4A8 MoE down (`_gen`, E_ffn%256)** | generalize crown's E_ffn==512-locked int-dot down; pp512 224→566 = **2.5×** vs fp16 | **KEEP, default-ON** (opt-out `IE_QWEN3NEXT_NO_Q8`) | `1942223`; PPL bit-stable |
| 7 | **Prefill chunk 256→512** | `--sweep --repeats 25` 25/25 bit-identical at 512; 1.08–1.15× on >256-tok prompts | **KEEP, `next_`-gated default 512** (env `IE_QWEN3NEXT_PREFILL_CHUNK`) | `bbf60c5`; closes pp512 to ~tie vs OLD llama |
| 8 | **int-dot fused T==1 DECODE** | tried routing decode through the fused int-dot path (won on single-card qwen3moe) | **REVERT** — measured **~40% SLOWER on 2-card** (per-layer host-route + cross-card overhead dominates at T=1) → decode stays on fp16 `moe_decode_*` | `51ab785` (measured-and-rejected) |
| 9 | **EXL3 native forward + fused MoE** | 4.51bpw on 2×B70; fused EXL3 MoE decode 1.46×/prefill 1.78× vs slow loop | **SPECIALIZE → then SHELVE** — beats the slow EXL3 loop but not the Q4 fast path; no win at 4.5bpw/2-card | `80eaeef`/`5b5f50c`/`58320b9`/`01f0a7b`; kept behind `IE_EXL3_MOE_SLOW` oracle |
| 10 | **80B fleet prompt/KV cache** | multi-turn restore across 2 cards; DeltaNet state position-dependent → snapshot at conversation boundary | **KEEP** (default-ON, opt-out `IE_NO_PROMPT_CACHE`) | `6355890`; turn-2 "16 cached" byte-identical |
| 11 | **`IE_QWEN3NEXT_PROFILE` phase profiler** | wait-bracketed relative phase breakdown (decode/prefill) | **KEEP** (tooling, opt-in) | `3af75fb` |
| 12 | **Chunk relax 256→512 for the PREFILL GAP** | hoped it closes the 0.50× pp512 gap | **DEFER/insufficient** — re-measured **+2.5% only**; the gap is the MoE-GEMM kernel, not the chunk | scorecard `ec4ec6d` |
| 13 | **SoA Q6_K decode-down repack** | qwen3moe got **2.38×** on Q6_K down via per-expert SoA; qwen3next hardcodes **`soa=false`** (`qwen3next.cpp:939-943`) — never wired | **DEFER (untried on 80B)** — candidate decode lever for the Q6_K-down layers; needs load-time repack + dtype-conditional gate (Q4_K stays AoS) | not committed |

**Dead-ends to NOT re-try:** (a) NEOX-rope as the DeltaNet-divergence cause (refuted, §2-5);
(b) int-dot fused decode on 2-card (REVERTED, #8); (c) chunk relax as the prefill-gap fix (#12);
(d) re-investigating the §1 `deltanet_recurrence` non-determinism root cause (exhaustively bisected,
HW-level, software-irreducible — `docs/known_bugs.md §1`).

---

## §6 Shape dispatch (path per T / ctx / dtype + env gates)

**Prefill (T>1):** chunked at **T≤512** (`next_`-gated in `engine.cpp:609-615`; override
`IE_QWEN3NEXT_PREFILL_CHUNK`, capped at max_ctx; T≥1024 NOT determinism-validated).
- Attention layers → `full_attention` (naive) per chunk; KV slot = card-local `kv_local_[L]`.
- DeltaNet layers → `deltanet_recurrence` (per-layer streaming conv_state + state).
- MoE → GPU-gemm router → host softmax/top-K/renorm → counting-sort packer → `moe_gather` →
  `gate_up_silu_q8` → **`moe_prefill_down_q{6,4}k_q8_gen`** (branched on `lw.down_dt`) → `moe_prefill_reduce`.
  **int-dot W4A8 default-ON**; opt-out `IE_QWEN3NEXT_NO_Q8` → fp16 packed-down.

**Decode (T==1):**
- Attention → `full_attention_fa2_decode` (FA-2 split, uses `w.attn_partials`).
- MoE → **fp16 `moe_decode_gate_up_silu_q4k` + `moe_decode_down_q{6,4}k`** (branched on `down_dt`,
  `soa=false`). The int-dot fused path is **deliberately NOT used at T==1** (#8: 2-card overhead).
- Shared expert runs every layer (gate/up/swiglu/down + per-token σ-gate `scaled_add`).

**dtype gates:** `IE_QWEN3NEXT_NO_Q8` (fp16 MoE down) · `IE_EXL3_MOE_SLOW` (EXL3 oracle loop) ·
`IE_QWEN3NEXT_PREFILL_CHUNK` (chunk size) · `IE_QWEN3NEXT_PROFILE` (phase timer) · `IE_NO_PROMPT_CACHE`.
EXL3 path keyed on `lw.attn_qkv.dt == kEXL3` (fused qkvz trellis → `slice_qkvz`).

**Engine:** `kQwen3Next` load branch → `fleet.init` → `LayerPlan::contiguous` (24/24) →
`next_model_.load(…, opts.max_ctx)`; generate uses `next_` host-logits bounce (mirrors TP);
ChatML / `<|im_end|>` fall-through.

---

## §7 Layout / scheduling state

- **Topology:** layer-parallel pipeline split 24/24 over 2×B70; each layer's full weight set
  (DeltaNet|attn + 512-expert MoE bank + shared expert) on its owning card; **one residual copy per card
  boundary** (`fleet_->copy_across`). Per-card hybrid caches (`DeltaNetState` + `KvCache`).
- **Streaming upload:** per-tensor upload at load (host RAM stays low — no 46 GB host buffer);
  fp16 casts for `ssm_conv1d`/`ssm_norm`; shexp Q8_0 dequant→fp16 at load.
- **Router:** logits via ONE GPU gemm `[T,H]@router_wᵀ{F16}` → host softmax/top-K/renorm (per-token,
  HF `norm_topk_prob` order). Decode keeps routing on host (1 token).
- **MoE int-dot:** activations quantized to `block_q8_1s` (`quantize_q8_1s`) → W4A8 dp4a down.
- **KV:** fp16; per-layer card-local slot; `set_length` per layer; FA-2 partials scratch
  (`ws_attn_partials_`) lazily sized from `max_ctx`.
- **⛔ LANDMINE — oneDNN is single-card-only.** Its static ctx binds to device 0 → **DEVICE_LOST** on any
  multi-card path. So the oneDNN MoE-GEMM prefill win (Coder/Gemma) is **unavailable to the 2-card 80B** —
  this is the structural cause of the prefill 0.50× (§4). Unblock = per-device oneDNN context (§8).
- **Host-stall:** per-decode-layer host route bounce (logits D→H, host softmax/top-K, pack H→D) +
  per-card `q.wait()` at the boundary. Shared with qwen3moe's known decode-bubble pattern (the qwen3moe
  on-device-routing fix is the cross-arch analog — not yet ported here).

---

## §8 Open frontier

**Honest verdict:** on a single-stream tok/s basis vs **current** llama-SYCL, the 80B now **LOSES both
axes** (decode 0.93×, prefill 0.50×). The durable, defensible standing is **structural** — first-mover
on-GPU on Intel Arc (Vulkan/OpenVINO can't run DeltaNet on B70; llm-scaler is throughput-only), plus the
sibling Crown's prefill 1.14× win. The 80B itself is **mostly commoditized grind on decode** (BW-bound,
both engines at the memory wall) — the ONE non-grind lever is amortizing weight reads (MTP self-spec).

**Ranked next bottlenecks (justified):**

1. **MTP self-spec decode — THE single most important item (the only real lever past the decode BW wall).**
   - *Mechanism:* decode is weight-BW-bound (GEMVs ~70%, lm_head ~63% BW). A NextN/MTP self-draft amortizes
     the weight reads across K accepted tokens — the proven path (27B 1.2×, Gemma 1.46× lossless).
   - *BLOCKED on THREE things (`COMPETITIVE_SCORECARD_2026-06-25.md §4`):*
     (0) **The verify path is regressed** — MTP is currently **0.81× even on the 27B** (the one model that
     HAS a head): spec 13.19 vs plain 16.33, lossless, mean-accept 2.63; the batched-T int-dot verify that
     the old "1.20×" relied on appears not engaged. **Must beat 1.0× before anything else.**
     (1) **No MTP head in the 80B GGUF** — `0 nextn` tensors verified in the Q4_K_M (and in Crown's). Must
     obtain/import a Qwen3-Next-80B GGUF that includes the NextN head.
     (2) **Spec hooks are 27B-only** — `all_logits` / `hidden_pre_norm` / DeltaNet `ckpt` exist only in
     `Qwen35DenseModel`; `Qwen3NextModel` needs them ported (the DeltaNet per-position checkpoint is the
     hard, correctness-critical part — and across a 2-card split).
   - *Falsifier:* if, after fixing (0), MTP still nets <1.0× on the 27B at the 80B's mean-accept regime,
     the verify-`forward(T=K)` overhead dominates and MTP is not the 80B decode lever either — stop and
     re-roofline.

2. **Per-device oneDNN MoE-GEMM port — FALSIFIED (`c96909a`).** Not merely the single-card landmine: at
   T<=512 chunk the MoE is small-M (rows/expert ~= 10-16, below oneDNN's regime per the Gemma T*K/E
   finding) -> no prefill win even with a per-device ctx. Do not re-chase.

3. **SoA Q6_K decode-down repack (low-cost probe).**
   - *Mechanism:* qwen3moe got **2.38×** on Q6_K down via per-expert SoA; qwen3next hardcodes `soa=false`.
     For the Q6_K-down layers (mixed per layer) this is a cheap decode candidate (dtype-conditional: Q4_K
     stays AoS — SoA regresses Q4_K, mirrors the dense finding).
   - *Falsifier:* if Q6_K-down is a small % of 80B decode (capture the missing roofline, §4) the win is
     negligible — measure the per-phase % first.

**Immediate gap to close before any of the above:** capture a clean **80B decode roofline** (the
`IE_QWEN3NEXT_PROFILE` per-phase %s are not recorded in the sources) — no tuning without the label.
