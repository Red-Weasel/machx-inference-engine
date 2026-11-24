#!/usr/bin/env bash
# scripts/inkling2/gpu_check.sh [CARD] — exit 0 only when card CARD (default 0) is free for a short test run:
#   - no process whose executable is the owner's installed engine ($HOME/00 - Inference Engine/build/src/ie,
#     judged by /proc/<pid>/exe; an `ie` built in a worktree, e.g. another agent's on another card, does not count),
#   - no DeepSWE evaluation running,
#   - card CARD under 1500 MiB used.
# Chain it with && in front of anything that loads a model.
#
# The DeepSWE pattern is written "deepswe_eval[ ]run" on purpose: a plain `pgrep -f "deepswe_eval run"` typed in a
# shell matches that shell's own command line and always reports "busy" (the memguard self-match trap).
set -u
CARD="${1:-0}"
INSTALLED="$HOME/00 - Inference Engine/build/src/ie"
for pid in $(pgrep -x ie); do
    exe=$(readlink "/proc/$pid/exe" 2>/dev/null || true)
    exe="${exe% (deleted)}"
    if [ "$exe" = "$INSTALLED" ]; then echo "gpu_check: the installed engine is running (pid $pid)"; exit 1; fi
    if [ -z "$exe" ]; then echo "gpu_check: cannot read the executable of ie pid $pid"; exit 1; fi
done
if pgrep -f "deepswe_eval[ ]run" >/dev/null; then echo "gpu_check: a DeepSWE evaluation is running"; exit 1; fi
used=$(xpu-smi stats -d "$CARD" 2>/dev/null | grep "GPU Memory Used" | sed -E 's/.*current: *([0-9]+).*/\1/')
if [ -z "$used" ]; then echo "gpu_check: cannot read card $CARD memory"; exit 1; fi
if [ "$used" -ge 1500 ]; then echo "gpu_check: card $CARD has $used MiB in use"; exit 1; fi
echo "gpu_check: card $CARD $used MiB used"
exit 0
