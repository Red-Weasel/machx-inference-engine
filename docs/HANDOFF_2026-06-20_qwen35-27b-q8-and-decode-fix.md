# Session handoff — 2026-06-20: Qwen3.6-27B Q8 split + engine-wide decode fix

Read `MASTER_DEV_PLAN.md` banner first. This is the session-specific continuation guide.

## What shipped (committed, validated)
1. **Qwen3.6-27B-obliterated-cyber Q8 runs on 2×B70.** New additive `Qwen35SplitModel`
   (`include/ie/qwen35_split.{hpp,cpp}`): Phase 1 layer-split (`a9704a1`) + Phase 2 native Q8_0-SoA
   packed weights (`6696b6b`) → **14.28 GB/card** (= the model's true 28.56 GB; NOT requantized,
   bit-exact). `ie run --gpus 2 <cyber-Q8.gguf>` → coherent. Single-GPU `Qwen35DenseModel` untouched.
   - oneDNN forced OFF on the split (multi-card `ctx_for` static-engine = DEVICE_LOST landmine).
   - `ie run --gpus 1 + IE_QWEN35_Q8=1` = single-GPU Q8 route (kept for >32 GB cards; on THIS box it
     host-spills under sustained load — non-viable, 2-GPU is correct).
2. **⭐ ENGINE-WIDE DECODE FIX (`a36c138`) — the big one.** Every queue had `enable_profiling` ALWAYS
   ON → ~1.76 ms host overhead/submit → decode submission-bound. Now opt-in (`IE_QUEUE_PROFILING=1`).
   **Crown 3.6→13.7 tok/s, 4B 2.55→7.9.** Crown PPL 6.4527 bit-exact. Helps ALL models incl the 27B.

## CRITICAL method note
- **`ie run` (chat) is NOT a perf tool** — JIT-noisy, and until `a36c138` it ran with profiling on.
  Every perf number I reported via `ie run` (27B "11 tok/s", single-GPU, no-SLM gemv A/B) is
  UNRELIABLE. **Re-measure with `ie-bench`** (raw `forward`, built-in warmup):
  `./build/tools/ie-bench --gguf <m> --prefill 1 --decode 64 --warmup 20`  → clean TG tok/s.
  `--kprofile-decode` → per-kernel GPU-event timing for one decode step (self-enables profiling).
- `ie-bench` is SINGLE-GPU only → can't yet bench the 2-GPU 27B split. Adding `--gpus N` is open thread (2).

## Open threads (priority order) — all have specs/docs
0. **DONE this session:** the enable_profiling fix (above).
1. **Chase remaining decode host-submit overhead.** Crown is 13.7 vs its historical 81; ~0.4 ms/kernel
   host gap REMAINS (GPU-busy ~11.5 ms vs wall ~73 ms). Investigate: per-submit SYCL event creation,
   L0 command-list flush, kernel count (fusion). Verify the historical 81 wasn't a different config
   (`--fastforward`/`--int8-kv`). FAST loop: `ie-bench --kprofile-decode` on 4B (~20 s load) / crown.
2. **Add `ie-bench --gpus N`** (split-aware bench) → get the 27B Q8 split's REAL post-fix decode.
3. **Speculative decoding** = chosen lever (lossless ~2-2.5×). Spec:
   `docs/superpowers/specs/2026-06-20-qwen35-27b-speculative-decoding.md`. STEP -1 blocker: drafters
   ran slow (partly fixed by (0) — re-bench Qwen2.5-3B / Qwen3-4B now); Qwen2.5-0.5B won't load
   (hidden=896 vs engine hidden%256==0 — relax or pad). Owner: bench drafters for NET speed, then
   abliterate the winner. STEP 0 = cheap acceptance-rate measure gates the whole build.
4. **Hybrid tensor-parallel** spec: `docs/superpowers/specs/2026-06-20-qwen35-27b-hybrid-tensor-
   parallel.md`. DEFERRED (no-P2P board caps ~1.4×; owner deprioritized vs spec-decode).

## Gates / invariants held
Crown `ie-perplexity` **6.4527** bit-exact across every commit this session. All changes additive;
crown / `Qwen35DenseModel` / `DenseModelTP` / `gemv_q8_0_soa_q8` (shared) UNTOUCHED in behavior.
Binary: `build/src/ie`. Bench: `build/tools/ie-bench`.
