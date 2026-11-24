#!/usr/bin/env bash
# scripts/inkling2/card1.sh CMD...   — run one GPU job on card 1 under the shared-card protocol (2026-11-22): hold an exclusive
#   flock on /tmp/ink2_card1.lock for the whole run, and check the card is under 1500 MiB (gpu_check.sh) once the lock is
#   held. Blocks until the lock is free. Exit status: the command's (or 1 if the card check fails).
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
exec flock /tmp/ink2_card1.lock bash -c 'bash "$0/scripts/inkling2/gpu_check.sh" 1 >/dev/null || { echo "card 1 not clear: not running"; exit 1; }; shift; exec "$@"' "$ROOT" _ "$@"
