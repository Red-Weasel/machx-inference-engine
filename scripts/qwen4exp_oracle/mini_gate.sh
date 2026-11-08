#!/bin/bash
# Mini qwen4exp end-to-end gate (2026-11-08): for each variant, mini_model.py builds a tiny random model from the HF reference
# classes (GGUF in llama.cpp's layout + the reference's fp32 logits), ie-q4e-mini-logits runs the engine on the file
# (decode, prefill, split prefill), mini_model.py compare scores them. No download; ~1-2 GB of scratch per variant.
#
#   scripts/qwen4exp_oracle/mini_gate.sh <out-dir> [variant ...]      (default: every variant)
#
# Variants (mini_model.py VARIANTS): flashnext (Flash-Next's shape), kv4 (4 KV heads), noqsa (no indexer keys / tensors),
# h5120 (hidden 5120), dense2560 / dense (a dense SwiGLU FFN, no experts; dense = the Qwen4-27B reading: hidden 5120,
# 4 KV heads, no QSA), arch_qwen4, and the wider sweep (hd128, q32kv8, moe64, ssm32, nople, hc8, lowrank512), fused_gu (one ffn_gate_up_exps bank).
# neg_qknorm is the negative control (a wrong norm transform in one block): it must FAIL; not in the default list.
# Rows where the reference's k-th / (k+1)-th router logits nearly tie are held to a looser cosine (the engine's F16 routing
# may pick the other expert there), and argmax only counts where the reference's top-1 is decisive (mini_model.py compare).
# IE_MINI_GPU=<ordinal> picks the card (default 0); IE_MINI_KEEP=1 keeps each variant's mini.gguf (default: deleted). Needs a built tree (ninja -C build ie_q4e_mini_logits) and the parity venv ($HOME/venv, or IE_ORACLE_PY=<python>).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="${1:?usage: mini_gate.sh <out-dir> [variant ...]}"; shift
VARIANTS=("$@"); [ ${#VARIANTS[@]} -eq 0 ] && VARIANTS=(flashnext kv4 noqsa h5120 dense2560 dense arch_qwen4 hd128 q32kv8 moe64 ssm32 nople hc8 lowrank512 fused_gu)
PY="${IE_ORACLE_PY:-$HOME/venv/bin/python}"
BIN="$ROOT/build/tools/ie-q4e-mini-logits"
[ -x "$BIN" ] || { echo "missing $BIN (ninja -C build ie_q4e_mini_logits)"; exit 2; }
mkdir -p "$OUT"
fails=0
for v in "${VARIANTS[@]}"; do
  D="$OUT/$v"
  (cd "$HERE" && "$PY" mini_model.py build --variant "$v" --out-dir "$D" 2>&1 | grep -v grid_thw | tail -1)
  timeout 900 "$BIN" "$D/mini.gguf" "$D/ids.i32" "$D/engine_logits.f32" "${IE_MINI_GPU:-0}" > "$D/engine.log" 2>&1; rc=$?
  sed 's/\x1b\[[0-9;]*m//g' "$D/engine.log" | grep -E "^mini:|load:|config:|init_runtime|error|abort" | tail -2
  if [ $rc -ne 0 ]; then echo "[$v] FAIL (engine rc=$rc; $D/engine.log)"; fails=$((fails + 1)); continue; fi
  (cd "$HERE" && "$PY" mini_model.py compare --out-dir "$D") || fails=$((fails + 1))
  # each mini GGUF is 0.7-4 GB and rebuilds in ~30 s: dropped after scoring unless IE_MINI_KEEP=1 (2026-11-08: a day of
  # gates left 70 GB on a 98 %-full root drive)
  [ "${IE_MINI_KEEP:-0}" = 1 ] || rm -f "$D/mini.gguf"
done
echo "mini gate: ${#VARIANTS[@]} variants, $fails failed"
[ $fails -eq 0 ]
