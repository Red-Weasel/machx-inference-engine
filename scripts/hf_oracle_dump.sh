#!/usr/bin/env bash
# hf_oracle_dump.sh — thin wrapper around scripts/hf_oracle_dump.py + diff_layers.sh.
#
# Validates an engine per-layer dump against the HF/PyTorch reference (the REAL
# ground truth that ships with every model on release day), enabling day-one
# correctness validation of a NEW architecture on Intel Arc — without waiting
# for llama.cpp to support it. See docs/hf_reference_oracle.md.
#
# This wrapper:
#   1. Runs the HF reference dump on CPU (scripts/hf_oracle_dump.py).
#   2. Cosine-compares it against an EXISTING engine dump dir via
#      tools/diff_layers.sh.
#
# PREREQUISITE (produced separately, on the GPU, in a different session):
#   The engine-side dump, e.g.
#       build/tools/ie-dense-dump  --gguf <model.gguf> --dump <ENGINE_PREFIX> -p "<PROMPT>"
#       build/tools/ie-qwen35-dump --gguf <model.gguf> --dump <ENGINE_PREFIX> -p "<PROMPT>"
#   This wrapper does NOT run any ie-* tool (GPU-exclusivity rule) and does NOT
#   touch the GPU. It only consumes the engine dump that already exists.
#
# Usage:
#   hf_oracle_dump.sh <HF_MODEL_DIR> <ENGINE_DUMP_PREFIX> [MAX_SLOT] [PROMPT]
#
# Example:
#   # (GPU session, separately) produce build dump:
#   #   build/tools/ie-dense-dump --gguf llama3.gguf --dump /tmp/eng/mine -p "The capital of France is"
#   # (this CPU session) produce HF ref + diff:
#   hf_oracle_dump.sh ~/models/Meta-Llama-3.1-8B-Instruct /tmp/eng/mine 33 "The capital of France is"
#
# Env overrides:
#   PY       python interpreter (default: python3; point at your CPU-torch venv)
#   HF_OUT   HF dump prefix (default: a mktemp dir)
#   NOSPECIAL=1  pass --no-special (do not add BOS) to match the engine tokenizer
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-python3}"

HF_MODEL="${1:?usage: hf_oracle_dump.sh <HF_MODEL_DIR> <ENGINE_DUMP_PREFIX> [MAX_SLOT] [PROMPT]}"
ENGINE_PREFIX="${2:?missing ENGINE_DUMP_PREFIX (an existing engine dump, e.g. /tmp/eng/mine)}"
MAX_SLOT="${3:-}"
PROMPT="${4:-The capital of France is}"

HF_OUT="${HF_OUT:-$(mktemp -d /tmp/hf_oracle.XXXXXX)/ref}"
NOSPECIAL_FLAG=""
[[ "${NOSPECIAL:-0}" == "1" ]] && NOSPECIAL_FLAG="--no-special"

say() { printf '\033[1m%s\033[0m\n' "$*"; }

# Sanity: the engine dump must already exist (we do NOT produce it here).
if [[ ! -f "${ENGINE_PREFIX}_L00.bin" ]]; then
    echo "ERROR: engine dump '${ENGINE_PREFIX}_L00.bin' not found." >&2
    echo "Produce it FIRST in a GPU session with ie-dense-dump / ie-qwen35-dump," >&2
    echo "then re-run this wrapper. (This script never touches the GPU.)" >&2
    exit 1
fi

say "[1/2] HF reference dump (CPU) → ${HF_OUT}_LNN.bin"
"$PY" "$ROOT/scripts/hf_oracle_dump.py" \
    --model "$HF_MODEL" --prompt "$PROMPT" --out "$HF_OUT" $NOSPECIAL_FLAG || {
        echo "HF dump failed (need CPU torch+transformers; see the script's hint)." >&2
        exit 2
    }

# If MAX_SLOT was not given, derive it from the highest HF slot we wrote.
if [[ -z "$MAX_SLOT" ]]; then
    MAX_SLOT=$(ls "${HF_OUT}"_L*.bin 2>/dev/null \
        | sed -n 's/.*_L\([0-9]\+\)\.bin/\1/p' | sort -n | tail -1)
    MAX_SLOT=$((10#${MAX_SLOT:-0}))
fi

say "[2/2] cosine diff: engine (${ENGINE_PREFIX}) vs HF ref (${HF_OUT}), slots 0..${MAX_SLOT}"
bash "$ROOT/tools/diff_layers.sh" "$ENGINE_PREFIX" "$HF_OUT" "$MAX_SLOT"

echo
say "done. HF reference artifacts: ${HF_OUT}_LNN.bin"
echo "Gate: every comparable slot should show cos_sim ~1.0 (>= 0.9985 like the"
echo "      llama.cpp oracle). A wrong rope/permute/norm blows up at L01."
