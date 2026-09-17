#!/usr/bin/env bash
# scripts/validate_int8_kv.sh — INT8 KV cache acceptance gate.
#
# Runs ie-perplexity on both fp16 and INT8 KV modes for the same corpus and
# reports the relative drift.  Used to validate item 8 (PLAN.md) before
# claiming it's ship-ready.
#
# Acceptance: relative perplexity drift < 0.5%.  This number is conservative;
# for symmetric per-row INT8 with a 256-element block, typical drift on
# clean prose is ~0.1-0.3%.  If we see > 0.5% there's almost certainly
# something wrong with the quant or dequant kernel — re-investigate before
# enabling INT8 by default.
#
# Usage:
#   scripts/validate_int8_kv.sh                    # built-in 512-token sample
#   scripts/validate_int8_kv.sh --text wiki.txt --max-tokens 1024
#   scripts/validate_int8_kv.sh --gguf /path/to/other.gguf
#
# All extra args are passed through to BOTH ie-perplexity invocations so the
# fp16 and INT8 runs see identical inputs.

set -euo pipefail

# Project root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PPL_BIN="$ROOT/build/tools/ie-perplexity"
if [[ ! -x "$PPL_BIN" ]]; then
    echo "build ie-perplexity first:" >&2
    echo "  cmake --build $ROOT/build --target ie_perplexity" >&2
    exit 1
fi

# Pull just the TSV trailer line out of an ie-perplexity run.
extract_tsv_line() {
    awk '/^# (fp16|int8|fp16-pfc|int8-pfc)\t/ { print }'
}

tsv_field() {
    local line="$1"
    local field="$2"
    printf '%s\n' "$line" | awk -F '\t' -v field="$field" '{ print $field }'
}

require_min_tokens() {
    local label="$1"
    local tokens="$2"
    local min_tokens="$3"

    if ! [[ "$tokens" =~ ^[0-9]+$ ]]; then
        echo "$label TSV token count is not an integer: '$tokens'" >&2
        exit 1
    fi
    if (( tokens < min_tokens )); then
        echo "$label scored too few tokens: $tokens (need >= $min_tokens)" >&2
        exit 1
    fi
}

# Args common to both runs (passed straight through).
COMMON_ARGS=("$@")
OVERALL_STATUS=0

echo "=== fp16 baseline ==="
FP16_OUT="$(mktemp)"
"$PPL_BIN" "${COMMON_ARGS[@]}" 2>&1 | tee "$FP16_OUT"
FP16_LINE="$(extract_tsv_line < "$FP16_OUT")"

echo
echo "=== INT8 KV ==="
INT8_OUT="$(mktemp)"
"$PPL_BIN" --int8-kv "${COMMON_ARGS[@]}" 2>&1 | tee "$INT8_OUT"
INT8_LINE="$(extract_tsv_line < "$INT8_OUT")"

rm -f "$FP16_OUT" "$INT8_OUT"

if [[ -z "$FP16_LINE" || -z "$INT8_LINE" ]]; then
    echo "could not parse TSV trailer from one of the runs." >&2
    exit 1
fi

# TSV: "# kv_mode\ttokens\tavg_nll\tperplexity"
FP16_TOKENS="$(tsv_field "$FP16_LINE" 2)"
INT8_TOKENS="$(tsv_field "$INT8_LINE" 2)"
FP16_PPL="$(tsv_field "$FP16_LINE" 4)"
INT8_PPL="$(tsv_field "$INT8_LINE" 4)"
require_min_tokens "fp16" "$FP16_TOKENS" 10
require_min_tokens "int8" "$INT8_TOKENS" 10

echo
echo "=== summary ==="
printf '  fp16 tokens : %s\n' "$FP16_TOKENS"
printf '  int8 tokens : %s\n' "$INT8_TOKENS"
printf '  fp16 PPL : %s\n' "$FP16_PPL"
printf '  int8 PPL : %s\n' "$INT8_PPL"

# Relative drift (%).  Bash can't do floats; use awk.
DELTA_PCT="$(awk -v a="$FP16_PPL" -v b="$INT8_PPL" 'BEGIN { printf "%.4f", (b - a) / a * 100.0 }')"
ABS_DELTA_PCT="$(awk -v d="$DELTA_PCT" 'BEGIN { printf "%.4f", (d < 0) ? -d : d }')"

printf '  drift    : %s%%\n' "$DELTA_PCT"

# Acceptance.
PASS="$(awk -v d="$ABS_DELTA_PCT" 'BEGIN { print (d < 0.5) ? "yes" : "no" }')"
if [[ "$PASS" == "yes" ]]; then
    echo "  result   : PASS (|drift| < 0.5%)"
else
    echo "  result   : FAIL — INT8 KV may have a quant/dequant bug; investigate."
    OVERALL_STATUS=2
fi

echo
echo "=== prefill→decode test (--prefill-chunk 64) ==="
echo "    INT8 KV cache must be correct after T>1 prefill"

FP16_PFC_OUT="$(mktemp)"
echo "--- fp16 prefill→decode ---"
"$PPL_BIN" --prefill-chunk 64 "${COMMON_ARGS[@]}" 2>&1 | tee "$FP16_PFC_OUT"
FP16_PFC_LINE="$(extract_tsv_line < "$FP16_PFC_OUT")"
rm -f "$FP16_PFC_OUT"

INT8_PFC_OUT="$(mktemp)"
echo "--- INT8 prefill→decode ---"
"$PPL_BIN" --int8-kv --prefill-chunk 64 "${COMMON_ARGS[@]}" 2>&1 | tee "$INT8_PFC_OUT"
INT8_PFC_LINE="$(extract_tsv_line < "$INT8_PFC_OUT")"
rm -f "$INT8_PFC_OUT"

if [[ -z "$FP16_PFC_LINE" || -z "$INT8_PFC_LINE" ]]; then
    echo "could not parse TSV from prefill→decode runs." >&2
    exit 1
fi

FP16_PFC_TOKENS="$(tsv_field "$FP16_PFC_LINE" 2)"
INT8_PFC_TOKENS="$(tsv_field "$INT8_PFC_LINE" 2)"
FP16_PFC_PPL="$(tsv_field "$FP16_PFC_LINE" 4)"
INT8_PFC_PPL="$(tsv_field "$INT8_PFC_LINE" 4)"
require_min_tokens "fp16-pfc" "$FP16_PFC_TOKENS" 2
require_min_tokens "int8-pfc" "$INT8_PFC_TOKENS" 2

echo
echo "=== prefill→decode summary ==="
printf '  fp16-pfc tokens : %s\n' "$FP16_PFC_TOKENS"
printf '  int8-pfc tokens : %s\n' "$INT8_PFC_TOKENS"
printf '  fp16-pfc PPL : %s\n' "$FP16_PFC_PPL"
printf '  int8-pfc PPL : %s\n' "$INT8_PFC_PPL"

PFC_DELTA="$(awk -v a="$FP16_PFC_PPL" -v b="$INT8_PFC_PPL" \
    'BEGIN { printf "%.4f", (b - a) / a * 100.0 }')"
PFC_ABS="$(awk -v d="$PFC_DELTA" \
    'BEGIN { printf "%.4f", (d < 0) ? -d : d }')"
printf '  drift        : %s%%\n' "$PFC_DELTA"

PFC_PASS="$(awk -v d="$PFC_ABS" 'BEGIN { print (d < 0.5) ? "yes" : "no" }')"
if [[ "$PFC_PASS" == "yes" ]]; then
    echo "  result       : PASS (|drift| < 0.5% — INT8 KV after T>1 prefill is correct)"
else
    echo "  result       : FAIL — INT8 KV after prefill is corrupted (prefill bug not fixed?)."
    OVERALL_STATUS=2
fi

exit "$OVERALL_STATUS"
