#!/bin/bash
# Head-to-head on ONE box and ONE GGUF: Mach X (`ie serve`) against llama.cpp builds (llama-bench), prompt reading and decode.
# Prints a markdown table plus everything needed to reproduce it (commits, driver, launch lines). Losses are printed too.
#
#   head_to_head.sh <gguf> [prompt_tokens=4096] [gen_tokens=256] -- [ie serve args...]
#   env: LLAMA_BINS="vulkan=/path/build-vk/bin sycl=/path/build-sycl/bin"   (name=dir pairs; a dir needs llama-bench)
#        IE_BIN (default build/src/ie), LLAMA_NGL (99), REPS (3), H2H_PORT (11496)
#
# What is compared: prompt tokens/s reading a cold prompt of about <prompt_tokens>, and decode tokens/s over <gen_tokens>.
# Mach X numbers are the server's own "[gen] prefill .. | decode .." line for a greedy chat request (median of REPS, each on
# a fresh conversation); llama.cpp's are llama-bench's avg_ts for -p / -n (REPS repetitions). Not compared: quality, memory,
# concurrent requests, time to first token. A model llama.cpp cannot load is reported as "does not load", not left out.
set -u
G="${1:?gguf}"; PT="${2:-4096}"; GT="${3:-256}"; shift; [ $# -gt 0 ] && [ "$1" != "--" ] && shift; [ $# -gt 0 ] && [ "$1" != "--" ] && shift; [ "${1:-}" = "--" ] && shift
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(cd "$HERE/../.." && pwd)"
IE="${IE_BIN:-$ROOT/build/src/ie}"; PORT="${H2H_PORT:-11496}"; REPS="${REPS:-3}"; NGL="${LLAMA_NGL:-99}"
OUT="$(mktemp -d "${TMPDIR:-/tmp}/h2h.XXXXXX")"
cd "$ROOT" || exit 2; source scripts/env.sh >/dev/null 2>&1
rows=()
# ---- Mach X ---------------------------------------------------------------------------------------------------------------
"$IE" serve "$G" --host 127.0.0.1 --port "$PORT" "$@" > "$OUT/ie.log" 2>&1 &
SP=$!
up=0; for i in $(seq 1 300); do kill -0 $SP 2>/dev/null || break; curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { up=1; break; }; sleep 2; done
if [ $up -eq 1 ]; then
  for r in $(seq 1 "$REPS"); do
    python3 - "$PT" "$GT" "$r" > "$OUT/body.json" <<'PY'
import json, sys
pt, gt, r = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
words = ("The engine reads the prompt and writes the reply one token at a time. " * (pt // 12 + 8)).split()
text = f"Run {r}. " + " ".join(words[: int(pt * 0.78)])
print(json.dumps({"messages": [{"role": "user", "content": text + "\n\nWrite a long story about a lighthouse."}], "max_tokens": gt, "temperature": 0, "ignore_eos": True}))
PY
    curl -s -m 1800 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d @"$OUT/body.json" > "$OUT/ie_r$r.json"
  done
  kill -TERM $SP 2>/dev/null; wait $SP 2>/dev/null
  read -r PP TG NP <<< "$(sed 's/\x1b\[[0-9;]*m//g' "$OUT/ie.log" | python3 -c "
import re, sys, statistics as st
pp, tg, n = [], [], []
for l in sys.stdin:
    m = re.search(r'\[gen\] prefill (\d+) tok \((\d+) cached\) / (\d+) ms.*decode (\d+) tok / (\d+) ms', l)
    if m and int(m[3]) > 0 and int(m[5]) > 0:
        pp.append((int(m[1]) - int(m[2])) / int(m[3]) * 1000); tg.append(int(m[4]) / int(m[5]) * 1000); n.append(int(m[1]))
print(f'{st.median(pp):.1f} {st.median(tg):.1f} {int(st.median(n))}' if pp else 'NA NA NA')")"
  rows+=("| Mach X | $(md5sum "$IE" | cut -c1-8) | $NP | $PP | $TG | \`ie serve $*\` |")
else
  kill $SP 2>/dev/null; wait $SP 2>/dev/null
  rows+=("| Mach X | $(md5sum "$IE" | cut -c1-8) | - | does not load | does not load | $(grep -m1 -iE 'load failed|error' "$OUT/ie.log" | cut -c1-80) |")
fi
# ---- llama.cpp builds -----------------------------------------------------------------------------------------------------
for pair in ${LLAMA_BINS:-}; do
  name="${pair%%=*}"; dir="${pair#*=}"
  if [ ! -x "$dir/llama-bench" ]; then rows+=("| llama.cpp $name | - | - | no llama-bench in $dir | - | |"); continue; fi
  ver="$(git -C "$dir/../.." log -1 --format=%h 2>/dev/null || echo '?')"
  if "$dir/llama-bench" -m "$G" -ngl "$NGL" -p "$PT" -n "$GT" -r "$REPS" -o json > "$OUT/lb_$name.json" 2> "$OUT/lb_$name.err"; then
    read -r PP TG <<< "$(python3 -c "
import json, sys
d = json.load(open('$OUT/lb_$name.json')); pp = [x['avg_ts'] for x in d if x.get('n_prompt') and not x.get('n_gen')]; tg = [x['avg_ts'] for x in d if x.get('n_gen') and not x.get('n_prompt')]
print(f'{pp[0]:.1f}' if pp else 'NA', f'{tg[0]:.1f}' if tg else 'NA')")"
    rows+=("| llama.cpp $name | $ver | $PT | $PP | $TG | \`llama-bench -ngl $NGL -p $PT -n $GT -r $REPS\` |")
  else
    rows+=("| llama.cpp $name | $ver | - | does not load | does not load | $(tail -1 "$OUT/lb_$name.err" | cut -c1-80) |")
  fi
done
echo "## Head-to-head: $(basename "$G")"
echo
echo "Box: $(lscpu | awk -F: '/Model name/{gsub(/^ +/,"",$2); print $2; exit}'), $(lspci | grep -ciE 'vga.*(e223|arc)') Arc card(s), kernel $(uname -r), $(date +%F)."
echo
echo "| Engine | Build | Prompt tokens | Prompt tok/s | Decode tok/s | Launch |"
echo "|---|---|---|---|---|---|"
printf '%s\n' "${rows[@]}"
echo
echo "Raw logs: $OUT"
