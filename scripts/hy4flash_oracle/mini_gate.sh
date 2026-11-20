#!/bin/bash
# Hy4-flash mini gate (docs/hy4flash/00_PORT_AND_ABLITERATION_PLAN.md, phase 2): the engine's forward against the reference
# implementation on a tiny random model with the real widths.
#
#   scripts/hy4flash_oracle/mini_gate.sh <work_dir> [reference_python]
#
# Builds <work_dir>/hy4mini (weights + oracle) and <work_dir>/hy4mini_ie (the engine directory) when they are missing, then
# runs build/tools/ie-hy4flash-mini in six regimes with the reference's discrete choices forced (expert picks, recalled
# groups: a random model is full of near-ties), once with a planted fault that must FAIL, and once free for the record.
# Needs both cards free for the two-card regime. ~7 GB of disk, ~3 minutes on an idle machine.
set -u
W="$(cd "$(dirname "$0")/../.." && pwd)"; D="${1:?work directory}"; PY="${2:-$HOME/models/hy4flash-reference/venv/bin/python}"
source "$W/scripts/env.sh" >/dev/null 2>&1
B="$W/build/tools/ie-hy4flash-mini"; O="$W/scripts/hy4flash_oracle"
[ -x "$B" ] || { echo "build ie_hy4flash_mini first"; exit 2; }
[ -f "$D/hy4mini/oracle/routes.i32" ] || "$PY" -I "$O/mini_model.py" build "$D/hy4mini" || exit 2
[ -f "$D/hy4mini_ie/model.safetensors.index.json" ] || "$PY" -I "$O/pack.py" "$D/hy4mini" "$D/hy4mini_ie" || exit 2
fail=0
run() {   # name, expected exit, env, args...
    local name="$1" want="$2" env="$3"; shift 3
    local out; out=$(env $env "$B" "$D/hy4mini_ie" --oracle "$D/hy4mini/oracle" "$@" 2>&1); local rc=$?
    echo "$out" | grep -E "^logits|^discrete|^MINI GATE|^forward|^init|^load" | sed "s/^/  /"
    if [ "$rc" = "$want" ]; then echo "[ok]   $name (exit $rc)"; else echo "[FAIL] $name: exit $rc, expected $want"; fail=1; fi
}
run "full-precision experts, one chunk"                               0 X=0 --force --exact-experts
run "full-precision experts, chunks of 40 to 64, then single tokens"  0 X=0 --force --exact-experts --chunk 40 --decode-from 64
run "4-bit pack, all in VRAM, one chunk"                              0 X=0 --force
run "4-bit pack, 16 in VRAM / 48 in RAM, chunks + single tokens"      0 X=0 --force --static 16 --pinned 48 --chunk 40 --decode-from 64
run "two cards, 16 / 48, chunks of 32 to 48, then single tokens"      0 X=0 --force --cards 2 --static 16 --pinned 48 --chunk 32 --decode-from 48
run "attention in many small row blocks (recall mask, 32 + 64 heads)"  0 IE_HY4F_SCORE_ELEMS=65536 --force
run "planted fault (attention sinks ignored) is caught"               3 IE_HY4F_PLANT=nosink --force
echo "--- for the record, the engine's own discrete choices (near-ties fall either way; not part of the verdict):"
"$B" "$D/hy4mini_ie" --oracle "$D/hy4mini/oracle" 2>&1 | grep -E "^logits|^discrete" | sed "s/^/  /"
[ $fail = 0 ] && echo "HY4-FLASH MINI GATE: PASS" || echo "HY4-FLASH MINI GATE: FAIL"
exit $fail
