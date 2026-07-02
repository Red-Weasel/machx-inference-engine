#!/usr/bin/env bash
# p3_parity_llama3.sh — P3a: the Llama-3.x correctness oracle battery.
# Third per-arch gate (clone of p2_parity_qwen3.sh). Three stages:
#
#   1. layer parity   — ie-dense-dump vs ie-llama-dump, tools/diff_layers.sh
#                       slots 0..33 (L00 absent for the llama graph). The dense
#                       path is gemm_fp16 (NOT oneDNN) so per-layer COSINE is
#                       ~1.0; gate on cos_sim >= 0.9985 on every comparable slot
#                       (the final-norm slots 32/33 are last-token tail-compares
#                       and sit ~0.9993 — still well above the gate). A wrong
#                       Q/K un-permute or rope_freqs blows up at L01.
#   2. greedy near-tie — engine argmax vs llama oracle argmax on a fixed prompt.
#                       "The capital of France is" is a near-degenerate position
#                       (' a' vs ' Paris' within ~0.1 logit); fp16-vs-fp32 flips
#                       which is #1 (the documented qwen3 token-26 precedent).
#                       PASS = the two argmaxes are the SAME near-tie set, i.e.
#                       each argmax appears in the other's top-2.
#   3. PPL gate       — ie-perplexity builtin corpus, deterministic. PASS =
#                       avg NLL == 2.378209 exactly at --max-tokens 511
#                       (PPL 10.79; default invocation = 2.380699 / 10.81).
#                       A sane absolute level — Llama-3.1-8B is not reasoning-
#                       tuned (cf. qwen3-8b 18.9).
#
# Env overrides: GGUF (model path), BUILD (build dir). ie-llama-dump needs the
# llama.cpp build-vk libs on LD_LIBRARY_PATH.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD:-$ROOT/build}"
GGUF="${GGUF:-$HOME/models/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$HOME/llama.cpp/build-vk/bin"
OUT="$(mktemp -d /tmp/p3_parity.XXXXXX)"
PROMPT="The capital of France is"
FAIL=0

say()  { printf '\033[1m%s\033[0m\n' "$*"; }
pass() { printf '  \033[32mPASS\033[0m %s\n' "$*"; }
fail() { printf '  \033[31mFAIL\033[0m %s\n' "$*"; FAIL=1; }

[[ -f "$GGUF" ]] || { echo "SKIP: $GGUF not found"; exit 0; }

# ---- 1. layer parity (cosine gate) ------------------------------------
say "[1/3] layer parity vs llama.cpp (slots 0..33, cos_sim >= 0.9985)"
"$BUILD/tools/ie-dense-dump" --gguf "$GGUF" --dump "$OUT/mine" -p "$PROMPT" \
    > "$OUT/mine.log" 2>&1 || fail "ie-dense-dump (see $OUT/mine.log)"
"$BUILD/tools/ie-llama-dump" -m "$GGUF" --dump "$OUT/ref" --n-layers 32 -ngl 0 \
    -p "$PROMPT" > "$OUT/ref.log" 2>&1 || fail "ie-llama-dump (see $OUT/ref.log)"
if [[ $FAIL -eq 0 ]]; then
    bash "$ROOT/tools/diff_layers.sh" "$OUT/mine" "$OUT/ref" 33 | tee "$OUT/diff.txt"
    # Parse cos_sim (4th column) on rows that actually compared (skip MISSING).
    MINCOS=$(awk 'NF>=4 && $1 ~ /^[0-9]+$/ {print $4}' "$OUT/diff.txt" | sort -g | head -1)
    if [[ -z "$MINCOS" ]]; then
        fail "could not parse cos_sim (see $OUT/diff.txt)"
    elif awk "BEGIN{exit !($MINCOS >= 0.9985)}"; then
        pass "min cos_sim $MINCOS >= 0.9985 (un-permute + rope_freqs correct)"
    else
        fail "min cos_sim $MINCOS < 0.9985 — forward divergence"
    fi
fi

# ---- 2. greedy near-tie ------------------------------------------------
say "[2/3] greedy near-tie vs llama oracle ('$PROMPT')"
# Engine top-2 ids from the dump log we already have ($OUT/mine.log).
EARG=$(sed -n 's/^argmax: id=\([0-9]*\).*/\1/p' "$OUT/mine.log" | head -1)
ETOP=$(sed -n 's/^  top[12] id=\([0-9]*\).*/\1/p' "$OUT/mine.log" | tr '\n' ' ')
OARG=$(sed -n 's/^argmax: id=\([0-9]*\).*/\1/p' "$OUT/ref.log" | head -1)
OTOP=$(sed -n 's/^  top[12] id=\( *[0-9]*\).*/\1/p' "$OUT/ref.log" | tr -d ' ' | tr '\n' ' ')
if [[ -n "$EARG" && -n "$OARG" ]] && \
   [[ " $OTOP " == *" $EARG "* ]] && [[ " $ETOP " == *" $OARG "* ]]; then
    pass "argmax near-tie set matches (engine=$EARG oracle=$OARG; each in other's top-2)"
else
    fail "argmax sets differ (engine=$EARG top2[$ETOP] vs oracle=$OARG top2[$OTOP])"
fi

# ---- 3. PPL gate -------------------------------------------------------
say "[3/3] PPL gate (builtin corpus) — deterministic: avg NLL == 2.378209"
"$BUILD/tools/ie-perplexity" --gguf "$GGUF" --max-tokens 511 > "$OUT/ppl.log" 2>&1
NLL=$(sed -n 's/^# fp16\t[0-9]*\t\([0-9.]*\)\t[0-9.]*$/\1/p' "$OUT/ppl.log")
if [[ -z "$NLL" ]]; then
    fail "could not parse avg NLL (see $OUT/ppl.log)"
elif [[ "$NLL" == "2.378209" ]]; then
    pass "avg NLL $NLL == 2.378209 (PPL 10.79, bit-exact)"
else
    fail "avg NLL $NLL != 2.378209 (numerics changed)"
fi

echo
if [[ $FAIL -eq 0 ]]; then say "p3_parity_llama3: ALL GREEN  (artifacts: $OUT)"
else say "p3_parity_llama3: FAILURES — artifacts kept at $OUT"; fi
exit $FAIL
