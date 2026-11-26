#!/bin/bash
# Watch a model work in Dream on TRAINING tasks (never held-out): the sessions the coach reads.
#   watch.sh LABEL MODEL_DIR EFFORT IDS [ATTEMPTS]     records in dream-coding-work/records/LABEL
set -u
L="$1"; M="$2"; E="$3"; IDS="$4"; N="${5:-2}"
H="$(cd "$(dirname "$0")" && pwd)"; ENG="${IE_ENGINE_DIR:-$HOME/00 - Inference Engine}"
W="${IE_DATA_DRIVE:-/media/$USER/Secondary Drive}/dream-coding-work"; LOG="$W/logs/$L"
bash "$H/../../inkling2/gpu_check.sh" 0 && bash "$H/../../inkling2/gpu_check.sh" 1 || { echo "cards not clear"; exit 3; }
source "$H/../../env.sh" >/dev/null 2>&1
systemd-run --user --unit=coding-ie --collect -q -p "StandardOutput=file:$LOG.ie.log" -p "StandardError=file:$LOG.ie.log" \
  --setenv=LD_LIBRARY_PATH="$LD_LIBRARY_PATH" --setenv=PATH="$PATH" \
  "$ENG/build/src/ie" serve "$M" --gpus 2 --parallel "${PAR:-8}" --ctx 131072 --slot-ctx "${SLOTCTX:-65536}" --port 11435 --reasoning-effort "$E"
for i in $(seq 1 120); do curl -s -m 2 http://127.0.0.1:11435/health | grep -qi ok && break; sleep 3; done
curl -s -m 2 http://127.0.0.1:11435/health | grep -qi ok || { echo "engine did not come up"; systemctl --user stop coding-ie; exit 4; }
python3 -I "$H/run.py" --cond nopage --split train --ids "$IDS" --label "$L" --attempts "$N" --stop-after 99 --workers 12 \
  --minutes 15 ${NOTES:+--notes "$NOTES"} > "$LOG.log" 2>&1
rc=$?
systemctl --user stop coding-ie
exit $rc
