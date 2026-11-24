#!/usr/bin/env bash
# scripts/inkling2/step_check.sh TAG [BASE_TAG] [bench]
#   Dumps the last-position logits of three runs into /tmp/inkling2-steps/TAG_{400,8k,abl400}/step0.f32:
#     400     the 400-token prompt, no ablation set (free-running picks)
#     8k      goldens2/long8500 with the reference's expert picks imposed, no ablation set
#     abl400  the 400-token prompt on Inkling-2-Mini-abliterated (its ie_ablation.json on)
#   then compares each with BASE_TAG's (relL2, max|d|, top-10) when given, and with "bench" also prints the speed table
#   (decode at context 30 / 2K / 8K / 32K / 131K, prefill at 2K / 8K). Card: CARD (default 0); gpu_check before each run.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TAG="$1"; BASE="${2:-}"; BENCH="${3:-}"
CARD="${CARD:-0}"
BIN="${BIN:-$ROOT/build/tools/ie-inkling2-run}"
PY=$HOME/models/inkling2-reference/venv/bin/python
G=$HOME/models/inkling2-reference/goldens2/long8500
OUT=/tmp/inkling2-steps
P400="${P400:-/tmp/inkling2-p2/p400.json}"
P8K="${P8K:-/tmp/inkling2-p2/p8192.json}"
mkdir -p "$OUT"
chk() { bash "$ROOT/scripts/inkling2/gpu_check.sh" "$CARD" >/dev/null || { echo "GPU check failed: not running"; exit 1; }; }
dump() {   # name, args...
    local n="$1"; shift
    rm -rf "$OUT/${TAG}_$n"; chk
    "$BIN" --card "$CARD" "$@" --dump-steps "$OUT/${TAG}_$n" >/dev/null 2>"$OUT/${TAG}_$n.err" || { echo "run $n failed:"; tail -3 "$OUT/${TAG}_$n.err"; exit 1; }
}
dump 400 --ids-from "$P400" --ablation off
dump 8k --ids-from "$G/meta.json" --generate 1 --step-routes "$G/routes.json" --force-ids meta --ablation off
dump abl400 --model $HOME/models/Inkling-2-Mini-abliterated --ids-from "$P400"
if [ -n "$BASE" ]; then
    for n in 400 8k abl400; do
        printf "%-7s vs %-10s " "$n" "$BASE"
        "$PY" -I "$ROOT/scripts/inkling2/cmp_logits.py" "$OUT/${TAG}_$n/step0.f32" "$OUT/${BASE}_$n/step0.f32" --tol 1e-5
    done
fi
if [ "$BENCH" = bench ]; then
    chk
    "$BIN" --card "$CARD" --ablation off --ids-from "$P8K" --bench 30,2048,8192,32768,131072 --bench-prefill 2048,8192 2>&1 | grep -E "BENCH|memory"
fi
