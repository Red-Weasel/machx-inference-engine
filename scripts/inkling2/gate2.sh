#!/usr/bin/env bash
# scripts/inkling2/gate2.sh — Inkling 2 Mini phase 2 (KV-cached path) against the reference's cached runs (goldens2/,
# made by ~/models/inkling2-reference/oracle_phase2.py on CPU) and against itself.
#
#   tf      golden prompt, 24 decode steps, every layer call of every step teacher-forced from the reference's cached run,
#           reference picks imposed, bf16 rounding emulated     -> every call of every step <= 5e-3   (pins SPEC 6)
#   forced  golden prompt, 24 steps, reference picks + reference ids fed, fp32 -> every step's top-1 = the reference's
#   greedy  golden prompt, free-running greedy 24 tokens        -> report the first differing position (target: none)
#   equiv   400-token prompt: whole / half + one at a time / chunks of 64 / chunks of 100 -> logits relL2 <= 1e-6 (fp32 cache)
#   trunc   400-token prompt: + 100 junk tokens, truncate to 300, refeed -> logits equal to the straight run (relL2 <= 1e-6),
#           and truncate below earliest_restorable() is refused
#   long    goldens2/long1500 and long8500 (if present): prefill with the reference's picks + 3 steps fed the reference's
#           ids -> top-1 = the reference's at every step; and free-running
#
#   scripts/inkling2/gate2.sh [tf] [forced] [greedy] [equiv] [trunc] [long]      (default: all)
# Env: CARD (default 0), G2 (goldens2 root), OUT (default /tmp/inkling2-gate2).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="${BIN:-$ROOT/build/tools/ie-inkling2-run}"
G2="${G2:-$HOME/models/inkling2-reference/goldens2}"
PY="${PY:-$HOME/models/inkling2-reference/venv/bin/python}"
OUT="${OUT:-/tmp/inkling2-gate2}"
CARD="${CARD:-0}"
MODES=("$@")
[ ${#MODES[@]} -eq 0 ] && MODES=(tf forced greedy equiv trunc long)
mkdir -p "$OUT"
DF="$G2/decode_france"
status=0
run() { bash "$ROOT/scripts/inkling2/gpu_check.sh" "$CARD" >/dev/null || { echo "GPU check failed: not running"; exit 1; }
        "$BIN" --card "$CARD" "$@"; }
cmp() { "$PY" -I "$ROOT/scripts/inkling2/compare_steps.py" "$@" || status=1; }

for m in "${MODES[@]}"; do
    echo "=== $m"
    case "$m" in
        tf)
            rm -rf "$OUT/tf"
            run --ref-bf16 --ids-from "$DF/meta.json" --generate 25 --step-routes "$DF/routes.json" --force-ids meta \
                --teacher-steps "$DF" --dump-steps "$OUT/tf" | tail -2
            cmp --dump "$OUT/tf" --golden "$DF" --tol-logits 1e-2 | grep -vE "^   step .* margin" ;;
        forced)
            rm -rf "$OUT/forced"
            run --ids-from "$DF/meta.json" --generate 25 --step-routes "$DF/routes.json" --force-ids meta --dump-steps "$OUT/forced" | tail -2
            cmp --dump "$OUT/forced" --golden "$DF" --tol-logits 2e-2 | grep -vE "^   step .* margin" ;;
        greedy)
            rm -rf "$OUT/greedy"
            run --ids-from "$DF/meta.json" --generate 25 --dump-steps "$OUT/greedy" | tail -3
            "$PY" -I "$ROOT/scripts/inkling2/compare_steps.py" --dump "$OUT/greedy" --golden "$DF" | grep -E "generated|first differing|logits there" ;;
        equiv)
            "$PY" -I -c "import json,sys; m=json.load(open(sys.argv[1])); json.dump({'prompt_ids': m['prompt_ids'][:400]}, open(sys.argv[2], 'w'))" \
                "$G2/long1500/meta.json" "$OUT/p400.json"
            # the fp32 arithmetic's chunking invariance: a bf16 key / value cache rounds values that differ by ~1e-7
            # between chunkings to different bf16 neighbours (logits then differ ~6e-3 relL2, as bf16 vs fp32 does)
            IE_INKLING2_KV=fp32 run --ids-from "$OUT/p400.json" --equiv 200 --max-chunk 512 --chunk 64 --chunk2 100 | tee "$OUT/equiv.log" | tail -4
            "$PY" -I -c "
import re,sys
rel=[float(x) for x in re.findall(r'relL2 ([0-9.e+-]+)', open(sys.argv[1]).read())]
ok=len(rel)==3 and max(rel)<=1e-6 and 'DIFFERENT' not in open(sys.argv[1]).read()
print('RESULT (equiv):', 'PASS' if ok else 'FAIL', '(logits relL2 <= 1e-6, identical top-10)'); sys.exit(0 if ok else 1)" "$OUT/equiv.log" || status=1 ;;
        trunc)
            "$PY" -I -c "import json,sys; m=json.load(open(sys.argv[1])); json.dump({'prompt_ids': m['prompt_ids'][:400]}, open(sys.argv[2], 'w'))" \
                "$G2/long1500/meta.json" "$OUT/p400.json"
            run --ids-from "$OUT/p400.json" --truncate-check 100 2>&1 | tail -3
            [ "${PIPESTATUS[0]}" -eq 0 ] || status=1 ;;
        long)
            for tag in long1500 long8500; do
                G="$G2/$tag"
                [ -f "$G/routes.json" ] || { echo "($tag: no goldens, skipped)"; continue; }
                rm -rf "$OUT/${tag}_forced" "$OUT/${tag}_free"
                run --ids-from "$G/meta.json" --generate 4 --step-routes "$G/routes.json" --force-ids meta --dump-steps "$OUT/${tag}_forced" | tail -2
                cmp --dump "$OUT/${tag}_forced" --golden "$G" --tol-logits 5e-2 | grep -vE "^   step .* margin"
                run --ids-from "$G/meta.json" --generate 4 --dump-steps "$OUT/${tag}_free" | tail -2
                "$PY" -I "$ROOT/scripts/inkling2/compare_steps.py" --dump "$OUT/${tag}_free" --golden "$G" --tol-logits 1 | grep -vE "^   step .* margin"
            done ;;
        *) echo "unknown mode $m"; exit 2 ;;
    esac
done
exit $status
