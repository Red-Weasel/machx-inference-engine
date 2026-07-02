# Project instructions

**Read `MASTER_DEV_PLAN.md` first — it is the single source of truth for
the entire build** (current state, roadmap, document map, open decisions).
For deep engine internals read `Hand-off.md` second.

Hard rules (full detail in MASTER_DEV_PLAN.md §6):
- PPL gate: every engine change must hold `./build/tools/ie-perplexity` ≤ 6.57.
- Perf A/Bs are order-controlled (new-old-new); discard the first run after
  any rebuild (JIT); GPU swings ±40 tok/s heat-soaked.
- Prefill chunk: the gated-DeltaNet family (crown/`qwen35moe`, 27B/`qwen35`,
  `qwen3next`) runs at T ≤ 512; all other arches at T ≤ 256 (the §1 BMG
  DeltaNet-recurrence HW-bug cap — `docs/known_bugs.md`). The bug is NOT
  reproducible on NEO 26.14/26.18 + kernel 6.17.0-35 (validated 2026-06-20);
  env-revert via `IE_QWEN35_PREFILL_CHUNK` / `IE_QWEN3NEXT_PREFILL_CHUNK`. Still
  do NOT re-investigate the root-cause non-determinism (exhaustively bisected, §1).
- Never touch ESIMD / block2d / `lsc_load` paths (hangs BMG-G31).
- Work happens directly on `main`; commit at every working milestone.
- Every milestone updates `MASTER_DEV_PLAN.md` — if reality and that file
  disagree, fixing the file is part of the change.
