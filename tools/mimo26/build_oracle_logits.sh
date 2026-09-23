#!/bin/bash
# Builds tools/mimo26/oracle_logits.cpp against the upstream llama.cpp checkout (docs/mimo26/00_PORT_PLAN.md, P2 gate).
set -eu
L=${LLAMA_CPP:-$HOME/llama.cpp-mimo}
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT=${1:-$L/build/bin/mimo26-oracle-logits}
g++ -O2 -std=c++17 -I"$L/include" -I"$L/ggml/include" "$HERE/oracle_logits.cpp" -o "$OUT" \
    -L"$L/build/bin" -lllama -lggml -lggml-base -Wl,-rpath,"$L/build/bin"
echo "built $OUT"
