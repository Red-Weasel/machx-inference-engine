#!/usr/bin/env bash
# glm53-goal-stamp — verification plan for 15 decode / 300 prefill.
# Run ONLY on a boot with zero `journalctl -k -b` "engine reset" hits.
set -uo pipefail
ROOT="${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
M="${M:-$HOME/models/GLM-5.3-Flash-GGUF/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf}"
BIN="${BIN:-./build/tools/ie-glm5next-run}"
WIKI="${WIKI:-$HOME/llama.cpp/wikitext-2-raw/wiki.test.raw}"
OUT="${OUT:-${SCRATCH:-$ROOT/results}/glm53-goal-stamp}"
mkdir -p "$OUT"

if [ "$(journalctl -k -b --no-pager 2>/dev/null | grep -ci 'engine reset')" -gt 0 ]; then
    echo "xe engine resets in THIS boot — refuse. Reboot first."
    journalctl -k -b --no-pager 2>/dev/null | grep -i 'engine reset' | tee "$OUT/gpu-health.log"
    exit 1
fi
if [ ! -x "$BIN" ]; then echo "missing $BIN"; exit 1; fi

RUN=(scripts/ie-run-guarded --timeout 3600 "$BIN" "$M" --gpus 2)
export IE_G5_CPU_MISS="${IE_G5_CPU_MISS:-1}"
export IE_G5_GROUPED="${IE_G5_GROUPED:-1}"

echo "stamp OUT=$OUT CPU_MISS=$IE_G5_CPU_MISS GROUPED=$IE_G5_GROUPED"
"${RUN[@]}" --ngen 96 2>&1 | tee "$OUT/decode.log"
"${RUN[@]}" --ngen 96 2>&1 | tee "$OUT/decode-2.log"
# Prefill: whole-layer stream + pipeline split. Decode must not inherit PP_STREAM
# (it would allocate 2×~5 GiB layer buffers and cut ecache).
IE_G5_PP_STREAM=1 "${RUN[@]}" --ppbench 2 --chunk 1024 --ppl "$WIKI" --pipeline \
    2>&1 | tee "$OUT/prefill.log"
IE_G5_PP_STREAM=1 "${RUN[@]}" --ppbench 2 --chunk 1024 --ppl "$WIKI" --pipeline \
    2>&1 | tee "$OUT/prefill-2.log"
"${RUN[@]}" --ppl "$WIKI" --chunk 512 2>&1 | tee "$OUT/ppl.log"

echo "════ SUMMARY ════"
grep -h "\[gen\] done:\|TOTAL:\|\[ppl\] FINAL:" "$OUT"/decode.log "$OUT"/decode-2.log \
    "$OUT"/prefill.log "$OUT"/prefill-2.log "$OUT"/ppl.log | sort -u
echo "full: $OUT"
