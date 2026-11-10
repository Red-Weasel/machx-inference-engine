#!/bin/bash
# Soak gate: start `ie serve` with a real launch line, run scripts/soak/soak_serve.py against it for N minutes, stop it the
# orderly way, then read its log. Exit 0 only if the traffic checks passed AND the log holds no error / device-lost line.
#   soak.sh <gguf> <minutes> [out_dir] -- <serve args...>        env: IE_BIN (default build/src/ie), SOAK_PORT (11497)
# Example (the owner's Flash-Next launch):
#   soak.sh "$G" 60 /tmp/soak -- --gpus 2 --ctx 262000 --parallel 4 --slot-ctx 75000 --thinking on --spec --spec-k 3 --prefill-chunk 2048
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(cd "$HERE/../.." && pwd)"
G="$1"; MIN="$2"; shift 2
OUT=/tmp/ie-soak
if [ "${1:-}" != "--" ] && [ $# -gt 0 ]; then OUT="$1"; shift; fi
[ "${1:-}" = "--" ] && shift
PORT="${SOAK_PORT:-11497}"; IE="${IE_BIN:-$ROOT/build/src/ie}"
mkdir -p "$OUT"; cd "$ROOT" || exit 2; source scripts/env.sh >/dev/null 2>&1
"$IE" serve "$G" --host 127.0.0.1 --port "$PORT" "$@" > "$OUT/serve.log" 2>&1 &
SP=$!
up=0; for i in $(seq 1 300); do kill -0 $SP 2>/dev/null || break; curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { up=1; break; }; sleep 2; done
if [ $up -ne 1 ]; then echo "soak: the server did not come up"; tail -5 "$OUT/serve.log"; kill $SP 2>/dev/null; exit 2; fi
echo "soak: ie $(md5sum "$IE" | cut -c1-8), server pid $SP, $MIN min, $(date +%T)"
python3 "$HERE/soak_serve.py" --base-url "http://127.0.0.1:$PORT" --minutes "$MIN" --doc "$ROOT/README.md" --pid $SP > "$OUT/soak.json" 2> "$OUT/soak.err"; rc=$?
kill -TERM $SP 2>/dev/null; for i in $(seq 1 120); do kill -0 $SP 2>/dev/null || break; sleep 1; done
kill -0 $SP 2>/dev/null && { echo "soak: the server did not stop in 120 s (left running, pid $SP)"; rc=1; }
bad=$(sed 's/\x1b\[[0-9;]*m//g' "$OUT/serve.log" | grep -ciE "DEVICE_LOST|generation error|EXCEPTION|load failed")
echo "soak: traffic rc=$rc, error lines in the server log: $bad"
sed 's/\x1b\[[0-9;]*m//g' "$OUT/serve.log" | grep -E "restart points:|device routing:|stopped in" | cut -c1-220
echo "soak: restart-point restores $(grep -c 'restart point at' "$OUT/serve.log"), full re-reads over 20K tokens $(grep -E '\[gen\] prefill [0-9]{5,} tok \(0 cached\)' "$OUT/serve.log" | grep -vc ' / 0 ms')"
[ $rc -eq 0 ] && [ "$bad" -eq 0 ]
