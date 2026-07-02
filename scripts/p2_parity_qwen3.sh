#!/usr/bin/env bash
# p2_parity_qwen3.sh — P2 Task 7: the qwen3-dense correctness oracle battery.
# Per-arch gate every future engine change can run.  Three stages:
#
#   1. layer parity   — ie-dense-dump vs ie-llama-dump on a fixed prompt,
#                       tools/diff_layers.sh slots 0..37 (L00 has no llama
#                       reference for qwen3 — skipped).  PASS = no WARN /
#                       DIVERGED among comparable slots (rel_fro < 5e-3).
#   2. greedy parity  — 64 greedy tokens (temp 0) on the fp16 decode path
#                       (IE_NO_Q8_DECODE=1) vs the recorded llama.cpp b8902
#                       golden continuation.  PASS = byte-identical text.
#                       (Default int-dot decode flips ONE near-tie at token
#                       26, top-2 logit gap ~0.1 — documented, not a bug;
#                       the parity gate pins the fp16 path.)
#   3. PPL gate       — ie-perplexity, builtin corpus, default (int-dot)
#                       decode.  The dense path is DETERMINISTIC post
#                       rope_partial-race fix (34e4c01; bisect in
#                       docs/dense_nondeterminism_2026-06-10.md): PASS =
#                       avg NLL == 2.940491 exactly (PPL 18.9251), same
#                       bit-exact regime as the crown's 1.864495.  Any
#                       deviation is a real numerics change, not noise.
#                       NOTE the constant is tied to the invocation: the
#                       gate was derived with --max-tokens 511 (510
#                       predicted tokens); the default invocation predicts
#                       511 tokens and scores 2.937037 exactly instead
#                       (docs/ppl_baseline_matrix.md).  Keep flag+constant
#                       in sync.  Cross-anchor: llama.cpp b8902 scores
#                       24.54 on the same corpus/GGUF (-c 256 --chunks 2)
#                       — the absolute level is the model, not the engine.
#
# Env overrides: GGUF (model path), BUILD (build dir).
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD:-$ROOT/build}"
GGUF="${GGUF:-$HOME/.seal/models/Qwen3-8B-Q4_K_M.gguf}"
OUT="$(mktemp -d /tmp/p2_parity.XXXXXX)"
PROMPT="The capital of France is the city of"
FAIL=0

say()  { printf '\033[1m%s\033[0m\n' "$*"; }
pass() { printf '  \033[32mPASS\033[0m %s\n' "$*"; }
fail() { printf '  \033[31mFAIL\033[0m %s\n' "$*"; FAIL=1; }

[[ -f "$GGUF" ]] || { echo "SKIP: $GGUF not found"; exit 0; }

# ---- 1. layer parity --------------------------------------------------
say "[1/3] layer parity vs llama.cpp (slots 0..37)"
"$BUILD/tools/ie-dense-dump" --gguf "$GGUF" --dump "$OUT/mine" -p "$PROMPT" \
    > "$OUT/mine.log" 2>&1 || { fail "ie-dense-dump (see $OUT/mine.log)"; }
"$BUILD/tools/ie-llama-dump" -m "$GGUF" --dump "$OUT/ref" --n-layers 36 \
    -p "$PROMPT" > "$OUT/ref.log" 2>&1 || { fail "ie-llama-dump (see $OUT/ref.log)"; }
if [[ $FAIL -eq 0 ]]; then
    bash "$ROOT/tools/diff_layers.sh" "$OUT/mine" "$OUT/ref" 37 | tee "$OUT/diff.txt"
    if grep -qE "DIVERGED|WARN" "$OUT/diff.txt"; then
        fail "layer divergence (see $OUT/diff.txt)"
    else
        pass "all comparable slots rel_fro < 5e-3"
    fi
fi

# ---- 2. greedy parity --------------------------------------------------
say "[2/3] greedy parity vs llama.cpp b8902 (--temp 0, 64 tokens, fp16 decode)"
GOLDEN=" Paris. The capital of Italy is Rome. The capital of Spain is Madrid. The capital of Germany is Berlin. The capital of Belgium is Brussels. The capital of the Netherlands is Amsterdam. The capital of Switzerland is Bern. The capital of Austria is Vienna. The capital of Poland is Warsaw. The capital of Czech Republic"
IE_NO_Q8_DECODE=1 "$BUILD/tools/ie-dense-dump" --gguf "$GGUF" \
    -p "The capital of France is" -n 64 > "$OUT/greedy.log" 2>&1
GOT=$(sed -n "s/^greedy text: '\(.*\)'$/\1/p" "$OUT/greedy.log")
if [[ "$GOT" == "$GOLDEN" ]]; then
    pass "64/64 tokens identical"
else
    fail "greedy mismatch (see $OUT/greedy.log)"
    diff <(echo "$GOLDEN") <(echo "$GOT") || true
fi

# ---- 3. PPL gate -------------------------------------------------------
# Bit-exact leak-canary on the FP16 ffn_down path. Since 2026-06-15 the dense
# decode default is the Q6_K→Q8_0 repack (IE_DENSE_NO_Q6K_REPACK opt-out); the
# repack path is also deterministic but int8-quantizes the activation, so this
# pure-fp16 canary (NLL 2.940491) is held by forcing the opt-out here — it stays
# the most sensitive detector of a stray branch leaking into the shared dense hot path.
say "[3/3] PPL gate (builtin corpus, fp16 opt-out decode) — deterministic: avg NLL == 2.940491 exactly"
IE_DENSE_NO_Q6K_REPACK=1 "$BUILD/tools/ie-perplexity" --gguf "$GGUF" --max-tokens 511 > "$OUT/ppl.log" 2>&1
NLL=$(sed -n 's/^# fp16\t[0-9]*\t\([0-9.]*\)\t[0-9.]*$/\1/p' "$OUT/ppl.log")
if [[ -z "$NLL" ]]; then
    fail "could not parse avg NLL (see $OUT/ppl.log)"
elif [[ "$NLL" == "2.940491" ]]; then
    pass "avg NLL $NLL == 2.940491 (bit-exact)"
else
    fail "avg NLL $NLL != 2.940491 (determinism broken or numerics changed)"
fi

echo
if [[ $FAIL -eq 0 ]]; then
    say "p2_parity_qwen3: ALL GREEN  (artifacts: $OUT)"
else
    say "p2_parity_qwen3: FAILURES — artifacts kept at $OUT"
fi
exit $FAIL
