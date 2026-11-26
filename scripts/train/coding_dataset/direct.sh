#!/bin/bash
# The held-out tasks answered directly through the engine, no Dream: direct.sh LABEL MODEL_DIR EFFORT
set -u
L="$1"; M="$2"; E="$3"
H="$(cd "$(dirname "$0")" && pwd)"; ENG="${IE_ENGINE_DIR:-$HOME/00 - Inference Engine}"; W="${IE_DATA_DRIVE:-/media/$USER/Secondary Drive}/dream-coding-work"
bash "$H/../../inkling2/gpu_check.sh" 0 && bash "$H/../../inkling2/gpu_check.sh" 1 || { echo "cards not clear"; exit 3; }
source "$H/../../env.sh" >/dev/null 2>&1
systemd-run --user --unit=coding-ie --collect -q -p "StandardOutput=file:$W/logs/$L.ie.log" -p "StandardError=file:$W/logs/$L.ie.log" \
  --setenv=LD_LIBRARY_PATH="$LD_LIBRARY_PATH" --setenv=PATH="$PATH" \
  "$ENG/build/src/ie" serve "$M" --gpus 2 --parallel "${PAR:-8}" --ctx 131072 --slot-ctx "${SLOTCTX:-65536}" --port 11435 --reasoning-effort "$E"
for i in $(seq 1 120); do curl -s -m 2 http://127.0.0.1:11435/health | grep -qi ok && break; sleep 3; done
"$HOME"/models/inkling2-train/venv/bin/python -I "$H/direct_eval.py" --tasks "${IE_DATA_DRIVE:-/media/$USER/Secondary Drive}/dream-dataset/grpo-coding-v1/heldout.grpo.jsonl" --label "$L" > "$W/logs/$L.log" 2>&1
rc=$?; systemctl --user stop coding-ie; exit $rc
