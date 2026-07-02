#!/usr/bin/env bash
# soak_1h.sh — 1 hour of continuous chat against `ie serve`; fails on any
# request error, empty completion, or server death.  Logs RSS every ~5 min.
set -euo pipefail
PORT="${PORT:-11436}"
DUR="${DUR:-3600}"
END=$((SECONDS + DUR)); N=0
while [ $SECONDS -lt $END ]; do
  R=$(curl -fsS -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"max_tokens":128,"messages":[{"role":"user","content":"Write two sentences about prime numbers."}]}') \
    || { echo "FAIL: request error at iter $N"; exit 1; }
  echo "$R" | grep -q '"content":"[^"]' || { echo "FAIL: empty completion at iter $N"; exit 1; }
  N=$((N + 1))
  if [ $((N % 20)) -eq 0 ]; then
    RSS=$(ps -o rss= -p "$(pgrep -f 'ie serve' | head -1)" 2>/dev/null || echo "?")
    echo "iter $N rss_kb $RSS elapsed_s $SECONDS"
  fi
done
echo "soak: $N completions in ${DUR}s, all OK"
