#!/usr/bin/env bash
# scripts/verify_publish_claims.sh — the §L pre-publish re-bench gate, automated.
#
# Run this on a CLEAN, IDLE, heat-soaked box before the gpt-oss launch goes to print.
# It re-measures every load-bearing number in docs/public/marketing/VERIFIED_CLAIMS_2026-06-29.md
# and prints MEASURED vs PUBLISHED so a human can confirm each claim survives a fresh bench.
#
# WHY a gate: the launch numbers were measured on a box that degraded to ~9-min loads + 4 GB
# swap during the day's work; the clean-box rule (a thrashed box gave a FAKE 6× regression once)
# means throughput numbers are not trustworthy until the box is idle. Correctness checks (ctest,
# crown PPL) are NOT perf-sensitive and are valid any time.
#
# Usage:
#   LLAMA_BENCH=/path/to/SYCL/llama-bench ./scripts/verify_publish_claims.sh
#   (LLAMA_BENCH is REQUIRED for the head-to-head — see GAP note below.)

set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
source scripts/env.sh 2>/dev/null || true

GO20B=/home/weezy/models/gpt-oss-20b-GGUF/gpt-oss-20b-mxfp4.gguf
GO120B=/home/weezy/models/lmstudio-community/gpt-oss-120b-GGUF/gpt-oss-120b-MXFP4-00001-of-00002.gguf
IEBENCH=./build/tools/ie-bench
IEPPL=./build/tools/ie-perplexity
TPBENCH=./build/tools/ie-gptoss-tp-bench

hr(){ printf '%.0s─' {1..78}; echo; }
say(){ printf '\n\033[1m%s\033[0m\n' "$*"; }

# ── 0. clean-box pre-flight (the clean-box rule) ─────────────────────────────
say "0. CLEAN-BOX PRE-FLIGHT"
SWAP=$(free -g | awk '/Swap/{print $3}')
LOAD=$(awk '{print $1}' /proc/loadavg)
echo "  swap_used = ${SWAP} GB   (want ≈0; >1 GB ⇒ box memory-pressured, throughput suspect)"
echo "  loadavg   = ${LOAD}      (want <0.5 idle; a busy core fakes a slowdown)"
[ "${SWAP:-0}" -gt 1 ] && echo "  ⚠ NOT CLEAN — reboot or wait for swap to drain before trusting tok/s below."
echo "  (Intel GPU idle-clock check: run 'xpu-smi stats' / 'intel_gpu_top' separately if available.)"

# ── 1. ctest — the real current count + pass (CORRECTNESS, box-state-independent) ──
say "1. ctest  (PUBLISHED last said 31/31; source now declares $(grep -c add_test tests/CMakeLists.txt) add_test)"
ctest --test-dir build -j1 --output-on-failure 2>&1 | tail -25

# ── 2. crown PPL gate — the hard ≤6.57 correctness gate (CLAUDE.md) ───────────
say "2. CROWN PPL  (PUBLISHED 6.4527, NLL 1.864495; HARD GATE ≤ 6.57)"
"$IEPPL" 2>&1 | grep -iE "perplexity|avg NLL|nll" | head -4

# ── 3. gpt-oss-20b head-to-head — THE LEAD CLAIM (throughput; needs clean box) ─
say "3. gpt-oss-20b 1×B70 — OURS  (PUBLISHED prefill 1795/4147/3428 @512/2048/4096, decode 58.3/57.4/55.6)"
for P in 512 2048 4096; do
  echo "  -- pp$P / tg128 --"
  "$IEBENCH" --gguf "$GO20B" --prefill "$P" --decode 128 --warmup 2 2>&1 | grep -iE "prefill|decode|tok/s" | head -4
done

say "3b. gpt-oss-20b 1×B70 — llama  (PUBLISHED SYCL 927/927/896 prefill, 50.3/49.9/49.4 decode)"
echo "  ⚠ GAP: only Vulkan llama-bench builds were found on this box (llama-b8902 / llama_vulkan)."
echo "         The docs claim 'vs llama-SYCL'. Set LLAMA_BENCH to the SYCL build, or RE-LABEL the"
echo "         comparison as vs-Vulkan after measuring. Vulkan path (for reference only):"
echo "         /home/weezy/llama.cpp-vulkan/llama-b8902/llama-bench"
if [ -n "${LLAMA_BENCH:-}" ] && [ -x "${LLAMA_BENCH:-}" ]; then
  for P in 512 2048 4096; do
    "$LLAMA_BENCH" -m "$GO20B" -p "$P" -n 128 -ngl 99 -fa 1 2>&1 | grep -iE "pp$P|tg128" | head -4
  done
else
  echo "  ↳ LLAMA_BENCH not set/executable — skipping the head-to-head opponent (THE blocking gate)."
fi

# ── 4. gpt-oss-120b 2×B70 TP — prefill + decode (throughput; needs clean box) ──
say "4. gpt-oss-120b 2×B70 TP  (PUBLISHED prefill 538-679 @≤16k / 433 @long; decode ~31 peak 32.05)"
echo "  -- moderate ctx (auto Phase-1 replicate, should print 'attn replicated') --"
"$TPBENCH" --gguf "$GO120B" --gpus 2 --ctx 16000 --prefill 4096 --decode 32 --warmup 8 2>&1 \
  | grep -iE "attn (replicated|sharded)|prefill .*tok|DECODE:" | head -4
echo "  -- long ctx (auto Phase-2 head-shard) --  [optional; slow load]"
echo "     ${TPBENCH} --gguf <120b> --gpus 2 --ctx 65000 --prefill 4096 --decode 32 --warmup 8"

hr
say "DONE. Compare each MEASURED block to its PUBLISHED target above."
echo "Gate passes when: ctest all-green (record the real count), crown ≤6.57, the 20b H2H prefill"
echo "stays >1.5× and decode >1.0× vs a CURRENT same-box llama, and the 120b numbers reproduce"
echo "within box noise on a CLEAN box. Until then, every tok/s in the launch docs is 'measured"
echo "2026-06-27/29, re-bench pending'."
