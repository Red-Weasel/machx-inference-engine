#!/bin/bash
# kmap.sh — kernel-level tuning map. For each model, dumps the per-kernel GPU
# breakdown for DECODE (T=1) and, for the spec/DeltaNet models, the VERIFY (T=K)
# forward — the exact map used to find tuning levers (which kernel owns the time,
# %, calls, avg ms). Built on the existing KernelProfiler (ie::ps + g_profiler)
# via ie-bench --kprofile-decode / --kprofile-verify.
#
# CAVEAT (learned 2026-06-22): IE_QUEUE_PROFILING inflates per-SUBMIT cost, so
# these numbers are for RELATIVE kernel ranking (find the lever), NOT absolute
# wall time — cross-check a candidate win with a real-wall ie-bench/spec run.
# Warm box is fine here (relative breakdown is box-robust). Run:
#   ./tools/kmap.sh 2>&1 | tee kmap_$(date +%F).log
set +e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT" || exit 1
B=./build/tools/ie-bench
CROWN=/home/weezy/.seal/models/Qwen3.6-35B-A3B-Q4_K_M.gguf
D27=/home/weezy/models/bartowski/Qwen3.6-27B-GGUF/Qwen_Qwen3.6-27B-Q4_K_M.gguf
CODER=/home/weezy/models/Qwen3-Coder-30B-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf
echo "==== kmap: per-kernel decode/verify tuning map ($(date)) ===="
for tag in "crown:$CROWN:dec" "27B:$D27:decver" "coder:$CODER:dec"; do
  name=${tag%%:*}; rest=${tag#*:}; g=${rest%:*}; mode=${rest##*:}
  echo "############### $name ###############"
  echo "--- DECODE (T=1) ---"
  $B --gguf "$g" --prefill 1024 --decode 6 --warmup 8 --kprofile-decode 2>&1 | grep -E "DECODE kernel|gemv|fa2|moe_|dn_|rms|GPU total|TOTAL" | head -30
  if [ "$mode" = "decver" ]; then
    echo "--- VERIFY (T=4) ---"
    $B --gguf "$g" --prefill 1024 --decode 6 --warmup 8 --kprofile-verify 4 2>&1 | grep -E "VERIFY kernel|gemv|fa2|attn|moe_|dn_|gemm|GPU total|TOTAL" | head -30
  fi
  echo
done
echo "==== kmap DONE $(date) ===="
