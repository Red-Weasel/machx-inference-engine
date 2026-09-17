#!/usr/bin/env bash
# bench_showdown.sh — order-controlled engine vs llama.cpp SYCL showdown.
#
# Protocol (docs/benchmark_matrix_2026-06-09.md):
#  * one discarded engine warmup run (JIT) after any rebuild
#  * then ROUNDS alternating pairs: engine pp512 -> llama-bench pp512
#  * decode (tg) pass is optional via --tg
# Env overrides:
#  GGUF        path to the model (default: the daily-driver Qwen3.6 Q4_K_M)
#  LLAMA_BENCH path to a SYCL llama-bench binary
#  ROUNDS      alternating pairs (default 3)
set -euo pipefail

GGUF="${GGUF:-$HOME/.seal/models/Qwen3.6-35B-A3B-Q4_K_M.gguf}"
LLAMA_BENCH="${LLAMA_BENCH:-/tmp/lcpp-master/build-sycl/bin/llama-bench}"
ROUNDS="${ROUNDS:-3}"
RUN_TG=0
[[ "${1:-}" == "--tg" ]] && RUN_TG=1

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IE_BENCH="$ROOT/build/tools/ie-bench"

[[ -x "$IE_BENCH" ]]     || { echo "missing $IE_BENCH — build first (cmake --build build -j)"; exit 1; }
[[ -x "$LLAMA_BENCH" ]]  || { echo "missing llama-bench at $LLAMA_BENCH (set LLAMA_BENCH=...)"; exit 1; }
[[ -f "$GGUF" ]]         || { echo "missing model at $GGUF (set GGUF=...)"; exit 1; }

engine_pp() {  # prints pp512 tok/s
  local v
  v=$("$IE_BENCH" --gguf "$GGUF" --ctx 1024 --prefill 512 --decode 0 2>/dev/null \
    | awk -F'\t' '/^# 512/{print $3}')
  [[ -n "$v" ]] || { echo "ERROR: engine pp512 extraction empty (ie-bench output changed?)" >&2; exit 1; }
  echo "$v"
}
engine_tg() {  # prints tg tok/s (96 steps after a 256 prefill)
  local v
  v=$("$IE_BENCH" --gguf "$GGUF" --ctx 1024 --prefill 256 --decode 96 --warmup 8 2>/dev/null \
    | awk -F'\t' '/^# 256/{print $5}')
  [[ -n "$v" ]] || { echo "ERROR: engine tg extraction empty (ie-bench output changed?)" >&2; exit 1; }
  echo "$v"
}
llama_run() {  # $1 = "-p 512 -n 0" or "-p 0 -n 128"; prints tok/s "mean ± sd"
  local v
  v=$( source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
       "$LLAMA_BENCH" -m "$GGUF" -ngl 99 -sm none -mg 0 $1 2>/dev/null \
         | awk -F'|' '/pp512|tg128/{gsub(/^ +| +$/,"",$(NF-1)); print $(NF-1)}' )
  [[ -n "$v" ]] || { echo "ERROR: llama-bench produced no tok/s (output format changed?)" >&2; exit 1; }
  echo "$v"
}

echo "== showdown: $(date -Is) =="
echo "model: $GGUF"
echo "llama-bench: $LLAMA_BENCH ($("$LLAMA_BENCH" --version 2>/dev/null || echo unknown))"
echo "-- engine JIT warmup (discarded) --"
engine_pp >/dev/null

printf "%-8s %-14s %-20s\n" "round" "engine pp512" "llama.cpp pp512"
for r in $(seq 1 "$ROUNDS"); do
  e=$(engine_pp); l=$(llama_run "-p 512 -n 0")
  printf "%-8s %-14s %-20s\n" "$r" "$e" "$l"
done

if [[ "$RUN_TG" == 1 ]]; then
  printf "%-8s %-14s %-20s\n" "round" "engine tg" "llama.cpp tg128"
  for r in $(seq 1 "$ROUNDS"); do
    e=$(engine_tg); l=$(llama_run "-p 0 -n 128")
    printf "%-8s %-14s %-20s\n" "$r" "$e" "$l"
  done
fi
echo "== done. Heat-soak caveat: treat cross-round drift > ~3% as thermal. =="
