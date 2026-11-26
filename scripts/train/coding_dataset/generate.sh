#!/bin/bash
# The generator service: the missing no-page held-out baseline runs beside one pass (then retries) over the train tasks.
cd "$(dirname "$0")"
W="${IE_DATA_DRIVE:-/media/$USER/Secondary Drive}/dream-coding-work"
python3 run.py --cond nopage --split heldout --workers 2 --slot-base 12 >> "$W/logs/gen_heldout.log" 2>&1 &
H=$!
python3 run.py --cond nopage --split train --attempts 3 --stop-after 1 --hours 2.8 --workers 12 --slot-base 0 \
    --metrics "$W/gen_metrics.jsonl" >> "$W/logs/gen_train.log" 2>&1
T=$?
wait $H; R=$?
if [ "$T" = 3 ] || [ "$R" = 3 ]; then systemctl --user stop coding-ie; exit 3; fi
exit $T
