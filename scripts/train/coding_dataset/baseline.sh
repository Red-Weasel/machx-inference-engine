#!/bin/bash
# Dream baseline: each model does the 52 held-out tasks through Dream's real agent loop, 3 attempts each, graded by the
# hidden tests. Same Dream, same tools, same settings for every model; only the served weights change.
#   baseline.sh LABEL MODEL_DIR [EFFORT]      (records in dream-coding-work/records/base_<LABEL>_<EFFORT>)
#   ABL=set.json: serve with that Inkling 2 direction set (steering / removal) instead of the folder's own
set -u
L="$1"; M="$2"; E="${3:-high}"
H="$(cd "$(dirname "$0")" && pwd)"; ENG="${IE_ENGINE_DIR:-$HOME/00 - Inference Engine}"
W="${IE_DATA_DRIVE:-/media/$USER/Secondary Drive}/dream-coding-work"; LOG="$W/logs/base_${L}_${E}"
bash "$H/../../inkling2/gpu_check.sh" 0 && bash "$H/../../inkling2/gpu_check.sh" 1 || { echo "cards not clear"; exit 3; }
source "$H/../../env.sh" >/dev/null 2>&1
systemd-run --user --unit=coding-ie --collect -q -p "StandardOutput=file:$LOG.ie.log" -p "StandardError=file:$LOG.ie.log" \
  --setenv=LD_LIBRARY_PATH="$LD_LIBRARY_PATH" --setenv=PATH="$PATH" ${ABL:+--setenv=IE_INKLING2_ABLATION="$ABL"} \
  "$ENG/build/src/ie" serve "$M" --gpus 2 --parallel "${PAR:-8}" --ctx 131072 --slot-ctx "${SLOTCTX:-65536}" --port 11435 --reasoning-effort "$E"
for i in $(seq 1 120); do curl -s -m 2 http://127.0.0.1:11435/health | grep -qi ok && break; sleep 3; done
curl -s -m 2 http://127.0.0.1:11435/health | grep -qi ok || { echo "engine did not come up"; systemctl --user stop coding-ie; exit 4; }
python3 -I "$H/run.py" --cond nopage --split heldout --label "base_${L}_${E}" --attempts 3 --stop-after 99 --workers "${PAR:-8}" \
  --minutes "${MINUTES:-15}" --metrics "$W/base_${L}_${E}.metrics.jsonl" > "$LOG.log" 2>&1
rc=$?
systemctl --user stop coding-ie
exit $rc
