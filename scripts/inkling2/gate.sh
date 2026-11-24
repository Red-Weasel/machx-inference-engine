#!/usr/bin/env bash
# scripts/inkling2/gate.sh — Inkling 2 Mini phase-1 gate: ie-inkling2-run against the reference goldens, three ways.
#
#   (a) teacher-forced (each call fed the golden output of the previous call), reference bf16 rounding emulated
#       (--ref-bf16) and the reference's expert picks imposed        -> every call <= 5e-3
#   (b) end to end, engine fp32 arithmetic, reference picks imposed -> every call <= 1.2e-2, logits <= 8e-3, top-1
#   (c) free-running, engine fp32 arithmetic                        -> top-1 152717; calls > 2e-2 listed with swaps
#
# Each run loads the model on ONE card (CARD, default 0) after scripts/inkling2/gpu_check.sh passes.
#   scripts/inkling2/gate.sh [a] [b] [c]          (default: all three)
# Env: CARD, MODEL, GOLD, ROUTES, OUT (dump folder root, default /tmp/inkling2-gate).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="${BIN:-$ROOT/build/tools/ie-inkling2-run}"
MODEL="${MODEL:-$HOME/models/Inkling-2-Mini}"
GOLD="${GOLD:-$HOME/models/inkling2-reference/goldens}"
ROUTES="${ROUTES:-$HOME/models/inkling2-reference/scratch/routes_ref.json}"
PY="${PY:-$HOME/models/inkling2-reference/venv/bin/python}"
OUT="${OUT:-/tmp/inkling2-gate}"
CARD="${CARD:-0}"
MODES=("$@")
[ ${#MODES[@]} -eq 0 ] && MODES=(a b c)
IDS=$("$PY" -I -c "import json,sys; print(','.join(map(str, json.load(open(sys.argv[1]))['ids'])))" "$GOLD/meta.json")
mkdir -p "$OUT"

status=0
for m in "${MODES[@]}"; do
    case "$m" in
        a) args=(--ref-bf16 --teacher "$GOLD" --routes "$ROUTES") ;;
        b) args=(--routes "$ROUTES") ;;
        c) args=(--repeat 2) ;;
        *) echo "unknown mode $m"; exit 2 ;;
    esac
    echo "=== ($m) ${args[*]}"
    rm -rf "${OUT:?}/$m"
    bash "$ROOT/scripts/inkling2/gpu_check.sh" "$CARD" && \
        "$BIN" --model "$MODEL" --card "$CARD" --ids "$IDS" "${args[@]}" --dump-calls "$OUT/$m" 2>&1 | tee "$OUT/$m.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "run ($m) failed"; exit 1; }
    "$PY" -I "$ROOT/scripts/inkling2/compare.py" --dump "$OUT/$m" --goldens "$GOLD" --mode "$m" --routes-ref "$ROUTES" \
        | tee "$OUT/$m.compare" || status=1
done
exit $status
