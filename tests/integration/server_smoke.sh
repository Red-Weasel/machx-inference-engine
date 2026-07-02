#!/usr/bin/env bash
# server_smoke.sh — boots `ie serve` and exercises every endpoint.
# Requires the daily-driver GGUF + GPU. ~2 min total (model load dominates).
set -euo pipefail
GGUF="${GGUF:-$HOME/.seal/models/Qwen3.6-35B-A3B-Q4_K_M.gguf}"
PORT="${PORT:-11436}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

"$ROOT/build/src/ie" serve "$GGUF" --port "$PORT" --ctx 4096 & SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT
up=0
for i in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then up=1; break; fi
  kill -0 $SRV 2>/dev/null || { echo "FAIL: server died during startup"; exit 1; }
  sleep 1
done
[[ $up == 1 ]] || { echo "FAIL: server not up after 120s"; exit 1; }

curl -fsS "http://127.0.0.1:$PORT/v1/models" | grep -q '"object":"list"' \
  && echo "models: OK"

BODY='{"model":"x","max_tokens":24,"temperature":0,"messages":[{"role":"user","content":"Reply with exactly the word: ping"}]}'
STREAM_BODY='{"stream":true,"model":"x","max_tokens":24,"temperature":0,"messages":[{"role":"user","content":"Reply with exactly the word: ping"}]}'

RESP=$(curl -fsS -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' -d "$BODY")
echo "$RESP" | grep -q '"finish_reason"' || { echo "FAIL: non-stream missing finish_reason"; exit 1; }
echo "$RESP" | grep -q '"content":"[^"]' || { echo "FAIL: non-stream empty content"; exit 1; }
echo "$RESP" | grep -qi "thinking process" && { echo "FAIL: thinking leak in content"; exit 1; }
echo "non-stream: OK"

STREAM=$(curl -fsS -N -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "$STREAM_BODY")
echo "$STREAM" | grep -q "^data: " || { echo "FAIL: no SSE frames"; exit 1; }
echo "$STREAM" | grep -q "data: \[DONE\]" || { echo "FAIL: missing [DONE]"; exit 1; }
echo "stream: OK"

CODE=$(curl -s -o /dev/null -w '%{http_code}' -X POST \
  "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d '{bad')
[[ "$CODE" == "400" ]] || { echo "FAIL: garbage body gave $CODE, want 400"; exit 1; }
echo "400-on-garbage: OK"

CODE=$(curl -s -o /dev/null -w '%{http_code}' -X POST \
  "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"messages":[]}')
[[ "$CODE" == "400" ]] || { echo "FAIL: empty messages gave $CODE, want 400"; exit 1; }
echo "400-on-empty-messages: OK"
echo "server_smoke: all OK"
