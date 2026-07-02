#!/usr/bin/env bash
# diff_layers.sh — compute max-abs diff per layer between two dump dirs.
# Both must use the {prefix}_L00..L41.bin convention.
#
# Usage:
#   diff_layers.sh <my_prefix> <ref_prefix> [max_slot]
# Example:
#   diff_layers.sh /tmp/ie_diff/mine_cmd0 /tmp/ie_diff/ref       # crown: 0..41
#   diff_layers.sh /tmp/p2/mine /tmp/p2/ref 37                   # qwen3-8b
#
# A slot missing on BOTH sides is skipped with a warning (the llama.cpp
# fork's qwen3 graph does not emit model.input_embed, so L00 has no
# reference for qwen3 models).

set -euo pipefail

MINE=${1:?missing my prefix}
REF=${2:?missing ref prefix}
MAX_SLOT=${3:-41}

printf "%5s  %-12s  %-12s  %-10s  %-10s  %s\n" "LAYER" "max_abs" "rel_fro" "cos_sim" "verdict" "(notes)"
for L in $(seq -f "%02g" 0 "$MAX_SLOT"); do
    MB="${MINE}_L${L}.bin"
    RB="${REF}_L${L}.bin"
    if [[ ! -f "$MB" || ! -f "$RB" ]]; then
        printf "%5s  MISSING (skipped)\n" "$L"
        continue
    fi
    python3 - "$MB" "$RB" "$L" <<'PY'
import sys, struct, os, math
mb, rb, L = sys.argv[1], sys.argv[2], sys.argv[3]
def load(p):
    n = os.path.getsize(p) // 4
    with open(p, "rb") as f:
        return list(struct.unpack(f"<{n}f", f.read()))
a = load(mb); b = load(rb)
note = ""
if len(a) != len(b):
    # llama.cpp slices the last layer + result_norm to the LAST TOKEN only
    # (inp_out_ids). Compare the overlapping tail rows.
    n = min(len(a), len(b))
    note = f" tail-compare {n} of ({len(a)} vs {len(b)})"
    a = a[-n:]; b = b[-n:]
mx = 0.0; worst_i = 0
d2 = 0.0; b2 = 0.0; a2 = 0.0; ab = 0.0
for i, (x, y) in enumerate(zip(a, b)):
    d = abs(x - y)
    if d > mx:
        mx = d; worst_i = i
    d2 += (x - y) * (x - y); b2 += y * y; a2 += x * x; ab += x * y
rel_fro = math.sqrt(d2 / b2) if b2 > 0 else float("inf")
cos = ab / math.sqrt(a2 * b2) if a2 * b2 > 0 else float("nan")
# Scale-aware verdict: our residual stream is fp16 (llama.cpp keeps fp32),
# so per-element error scales with activation magnitude — judge the
# layer-wide relative Frobenius error, not absolute element diffs.
verdict = "OK" if rel_fro < 5e-3 else ("WARN" if rel_fro < 5e-2 else "**DIVERGED**")
print(f"   {L}  {mx:<12.5g}  {rel_fro:<12.5g}  {cos:<10.6f}  {verdict:<10}  worst@{worst_i} (mine={a[worst_i]:.4f} ref={b[worst_i]:.4f}){note}")
PY
done
