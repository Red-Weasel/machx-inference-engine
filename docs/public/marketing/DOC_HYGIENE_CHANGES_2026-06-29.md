# Doc-Hygiene Sweep — 2026-06-29

Edit-in-place pass to align launch-facing docs with the
[VERIFIED_CLAIMS_2026-06-29 ledger](VERIFIED_CLAIMS_2026-06-29.md) (the source of truth for all numbers).
Every number below is taken verbatim from that ledger; nothing was invented.

Hardware framing used throughout: Intel Arc Pro B70 (BMG-G31, 32 GB, 608 GB/s). 20b claims = **1× B70**;
120b claims = **2× B70 tensor-parallel, no working P2P**. Opponents always labeled (llama.cpp SYCL vs
Vulkan vs LM Studio).

---

## What I changed

### 1. `README.md`
- **Status line** (was `v1 shipped; P2 (model breadth) complete`): now reads **8+ architectures supported**
  (Qwen3.6 DeltaNet, Qwen3/Coder MoE, dense Qwen/Llama, Gemma-4, Phi-3, Mistral/Granite, **gpt-oss-20b/120b
  — the 8th arch**) and **multi-GPU tensor-parallel shipped in-product** (`ie run/serve --gpus N`).
- **Added two `gpt-oss` rows** to the supported-models table (after `qwen3moe`):
  - **gpt-oss 20b (1× B70):** WIN both axes vs llama.cpp **SYCL** (FA-on best config, same GGUF): prefill
    1795 / 4147 / 3428 vs 927 / 927 / 896 = **1.94× / 4.47× / 3.83×**; decode 58.3 / 57.4 / 55.6 vs
    50.3 / 49.9 / 49.4 = **1.16× / 1.15× / 1.13×**; streaming PPL 22.5; Harmony chat + tool calling.
    Carries the "measured 2026-06-27, clean-box re-bench pending" caveat and the repro command
    (`ie-bench --warmup 2` vs `llama-bench -ngl 99 -fa 1`). [ledger §A, §C, §H]
  - **gpt-oss 120b (2× B70 TP):** decode ~31 t/s (peak 32.05, ±15% box noise), prefill 538–679 t/s @ ctx
    ≤16k (measured on a degraded box — re-bench pending), batched-prefill **PPL 15.1985 bit-identical to
    T=1**, display-safe load split (card0 29.3 GB + 3.4 GB host-spill), coherent Harmony chat + tool
    calling, vs LM Studio layer-split ~12.42 t/s (owner-reported). [ledger §D, §E, §G, §F]
- **27B (`qwen35`) row decode label fixed** (was bare "decode parity"): now explicitly **parity vs
  Vulkan only**, and a **LOSS vs llama.cpp SYCL** (the faster single-stream opponent — never claim a 27B
  decode win), with a pointer to `COMPETITIVE_SCORECARD_2026-06-25.md §4`.

### 2. `docs/PUBLISHING_READINESS_2026-06-21.md`
- **Added a bold `⚠️ SUPERSEDED (2026-06-29)` banner at the very top.** States the file is retained for
  history only and that its **#1 claim — "qwen3next-80B decode 1.40× vs llama SYCL" is DEAD: now 0.93×
  (a LOSS), ~7% behind current llama-SYCL's 60.8 t/s; the 1.40× was vs an old ~37 t/s llama** [ledger §K].
  Also flags that the Gemma-4 26B MTP result is NOT lossless and the 80B prefill loss has since flipped to
  a win. Points readers to the current scorecard, the ledger, and `MASTER_DEV_PLAN.md`.
- **Inline-marked the #1 entry** in the "strongest claims" list with strikethrough + `❌ DEAD` so a skimming
  reader can't cite it.

---

## Drift noted (not all fixed — flagged per task)

### PPL string drift `6.45 / 6.4527 / 6.54`
- **Standard = `6.4527`** (NLL 1.864495) — the crown PPL gate, bit-exact. `6.45` is just its rounded form;
  `6.54` is a **different model** (the lmhead-q4k crown variant), **not** a drift of the gate — do not
  conflate them.
- In the files I touched: `PUBLISHING_READINESS_2026-06-21.md` already uses `6.4527` consistently (2
  occurrences, both correct — left as-is); `README.md` does not mention the crown PPL gate, so nothing to
  standardize there. **No code/doc-wide normalization performed** (out of scope of the touched files).

### ctest drift `7 / 30 / 31` (source now declares 32)
- Neither `README.md` nor `PUBLISHING_READINESS_2026-06-21.md` quotes a ctest count, so nothing was changed
  in the touched files. For the record: `COMPETITIVE_SCORECARD_2026-06-25.md:110` still says `30/30`; the
  gpt-oss-era docs say `31/31`; the current source tree declares **32 `add_test()`** entries
  (ledger §J/§L). **Not reconciled** — requires an actual `ctest` run on a clean box before any "all tests
  green" claim (ledger §L item 2). Flagged, not fixed (a build/run, not a doc edit).

---

## What remains (out of scope of this sweep)

1. **Clean-box re-bench** of every tok/s number and vs-llama / vs-LM-Studio ratio in README's gpt-oss rows
   (ledger §A, §D, §E are all "NEEDS CLEAN-BOX RE-BENCH"). The 20b 927-prefill llama baseline and the
   decode margin (1.13–1.16×, within ±40 t/s box noise) are the load-bearing gates.
2. **Run `ctest`** and state the real current number (31 or 32) + re-confirm crown PPL 6.4527 (ledger §L).
3. **Internal contradiction in `gptoss_120b_tp_2026-06-27.md`** (TL;DR says publish-ready; lower body still
   says "partial / open bugs / benchmark held" with the old 12.91 / 14.6 numbers) — NOT touched here;
   tracked in ledger §G and §L item 5.
4. **Reconcile prefill-comm figure** (plan "67.6%" vs 120b doc "68%"), ledger §L item 6.
5. **README pre-existing numbers I did not re-verify** (27B PPL 5.34, cosine ≥0.9995, the "1.9× vs Vulkan"
   prefill, the qwen3moe/llama/72B PPL figures): these predate the ledger and were left as-is except for
   the decode-label fix the task called out. They should be reconciled against the scorecard in a later pass.

---

### One-line summary
README now lists gpt-oss-20b/120b with ledger-exact, opponent-labeled numbers and an honest 27B
decode-loss note; the 2026-06-21 publishing audit is banner-marked SUPERSEDED (its dead 1.40× headline
called out); PPL and ctest drifts are documented and standardized where touched, with the rest flagged for
the clean-box re-bench.
