#!/bin/bash
# Hy4-flash: the whole expert pack at a low-bit format, one layer at a time.
#
#   pack_lowbit_all.sh <source_dir> <ie_dir> <bits_gu> <bits_dn> <cov.npz> [--replace-q8] [--layers "1 2 3 ..."] [--threads N]
#
# Per layer: pack_lowbit_trial.py rounds the experts (gate / up with GPTQ under the recorded input covariance, down
# plainly at its own width) into a scratch directory in the 8-bit layout, repack_lowbit.py rewrites those integers at
# their real width as <ie_dir>/experts-q<g><d>-LNN.bin, and the scratch file is removed. A layer whose file exists with
# the right size is skipped, so the script can be stopped and restarted.
#
# --replace-q8 removes <ie_dir>/experts-q8-LNN.bin after its layer is written (the 8-bit pack is 147 GiB and the main
# drive cannot hold both; pack_q8.py rebuilds it from the source in about 25 minutes). WITHOUT the flag nothing is
# deleted and the script stops when the drive has less than 8 GiB free. Sizes: 4,8 -> 2.46 GiB a layer (101 GiB in all),
# 5,5 -> 2.33 GiB (95 GiB), 4,4 -> 1.80 GiB (74 GiB). About 3-5 minutes a layer on 14 threads.
# Run with the reference venv active directory as cwd (it calls venv/bin/python).
set -euo pipefail
SRC="$1"; IE="$2"; BG="$3"; BD="$4"; COV="$5"; shift 5
REPLACE=0; LAYERS=""; THREADS=14
while [ $# -gt 0 ]; do
  case "$1" in
    --replace-q8) REPLACE=1 ;;
    --layers) LAYERS="$2"; shift ;;
    --threads) THREADS="$2"; shift ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac; shift
done
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-venv/bin/python}"
[ -n "$LAYERS" ] || LAYERS="$(seq 1 41)"
TAG="q${BG}${BD}"; [ "$BG$BD" = "88" ] && TAG="q8"
SCRATCH="$IE/.lowbit-scratch"
row() { case "$1" in 8) echo $(( $2 )) ;; 5) echo $(( $2 / 2 + $2 / 8 )) ;; *) echo $(( $2 / 2 )) ;; esac; }
E=512; MI=1536; LAT=1536
RAW=$(( 2 * MI * $(row "$BG" $LAT) + LAT * $(row "$BD" $MI) + 2 * MI * (LAT / 32) * 2 + LAT * (MI / 32) * 2 ))
SLOT=$(( (RAW + 63) / 64 * 64 )); WANT=$(( E * SLOT ))
# <cov> ending in .npz = the old pooled file (--cov); anything else = the engine recorder's prefix (--cov-engine).
# FROM_Q8=1: no BF16 source -- the source weights are <ie_dir>'s 8-bit pack, dequantized (pass <ie_dir> as <source_dir>).
case "$COV" in *.npz) COVARG="--cov $COV" ;; *) COVARG="--cov-engine $COV" ;; esac
FROMQ8=""; [ "${FROM_Q8:-0}" = 1 ] && FROMQ8="--from-q8"
t_all=$(date +%s)
for L in $LAYERS; do
  LL=$(printf "%02d" "$L"); OUT="$IE/experts-$TAG-L$LL.bin"
  if [ -f "$OUT" ] && [ "$(stat -c %s "$OUT")" = "$WANT" ]; then echo "layer $L: exists, skipped"; continue; fi
  FREE=$(df --output=avail -B1 "$IE" | tail -1)
  if [ "$FREE" -lt $(( 8 * 1024 * 1024 * 1024 )) ]; then echo "layer $L: less than 8 GiB free on the target drive, stopping (nothing deleted)"; exit 3; fi
  rm -rf "$SCRATCH"; t0=$(date +%s)
  EXTRA=""; [ "$BD" != "$BG" ] && EXTRA="--down-bits $BD"
  "$PY" -I "$HERE/pack_lowbit_trial.py" "$SRC" "$IE" "$SCRATCH" --layers "$L" --bits "$BG" --method gptq $EXTRA $COVARG $FROMQ8 --threads "$THREADS" | grep "^layer"
  "$PY" -I "$HERE/repack_lowbit.py" "$SCRATCH" "$IE" --layers "$L" --bits "$BG,$BD" | grep "^layer"
  rm -rf "$SCRATCH"
  [ "$(stat -c %s "$OUT")" = "$WANT" ] || { echo "layer $L: $OUT has the wrong size"; exit 4; }
  if [ "$REPLACE" = 1 ] && [ -f "$IE/experts-q8-L$LL.bin" ] && [ "$TAG" != "q8" ]; then rm -f "$IE/experts-q8-L$LL.bin"; echo "layer $L: the 8-bit file removed"; fi
  echo "layer $L: done in $(( $(date +%s) - t0 )) s"
done
echo "pack $TAG: finished in $(( ($(date +%s) - t_all) / 60 )) min"
