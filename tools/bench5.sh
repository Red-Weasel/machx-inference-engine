#!/bin/bash
# bench5.sh — the REAL publish benchmark. EVERY model row reports the THREE
# mandatory metrics, the SAME way for both engines:
#   (1) PROMPT PROCESSING  — prefill tok/s   (prompt_tokens / prefill_time)
#   (2) TOK/S              — decode  tok/s   (gen_tokens   / decode_time)
#   (3) PPL               — perplexity on wikitext-2 test (quality)
# ONE protocol, BOTH engines, the SAME 5 prompts (benchmark_prompts/) for pp/tg,
# the SAME corpus (wiki.test.raw) for PPL. Greedy (temp 0), decode budget 128,
# 3 runs / discard run 1 / median. Where an engine can't run a model, the cell
# says so (N/A) — never silently skipped.
#
# OURS  pp/tg = ie-prompt-bench (ie::Engine product path; auto 1/2-GPU; --spec where intended)
# OURS  PPL   = ie-perplexity / ie-gemma4-ppl / ie-qwen3next-ppl (per arch)
# LLAMA pp/tg = llama-cli -f <prompt> -n N --temp 0   (raw prompt+BOS, == ours' raw encode)
# LLAMA PPL   = llama-perplexity -f wiki.test.raw     (SYCL for DeltaNet/MoE, Vulkan for gemma)
# OpenVINO    = N/A (not installed; IR-convert per model to add a column)
#
# CLEAN-BOX FIRST (non-negotiable): reboot, swap≈0, GPU idle-clocked, one workload at a time.
# Run:  ./tools/bench5.sh 2>&1 | tee bench5_$(date +%F).log
set +e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT" || exit 1
PD=benchmark_prompts
CORPUS=~/llama.cpp/wikitext-2-raw/wiki.test.raw
PPLTOK=4096                       # PPL eval tokens (same budget both engines)
N=128; RUNS=3; WARM=1
OURS=./build/tools/ie-prompt-bench
IEPPL=./build/tools/ie-perplexity
IEPPL_G=./build/tools/ie-gemma4-ppl
IEPPL_N=./build/tools/ie-qwen3next-ppl
LCLI_S=~/llama.cpp/build-sycl/bin/llama-cli
LCLI_V=~/llama.cpp/build-vk/bin/llama-cli
LPPL_S=~/llama.cpp/build-sycl/bin/llama-perplexity
LPPL_V=~/llama.cpp/build-vk/bin/llama-perplexity

CROWN=/home/weezy/.seal/models/Qwen3.6-35B-A3B-Q4_K_M.gguf
D27=/home/weezy/models/bartowski/Qwen3.6-27B-GGUF/Qwen_Qwen3.6-27B-Q4_K_M.gguf
CODER=/home/weezy/models/Qwen3-Coder-30B-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf
G31=/home/weezy/models/google/gemma-4-31B-it-qat-q4_0-gguf/gemma-4-31B_q4_0-it.gguf
G31H=/home/weezy/models/google/gemma-4-31B-it-qat-q4_0-gguf/mtp-gemma-4-31B-it-Q8_0.gguf
G26=/home/weezy/models/google/gemma-4-26B-A4B-it-qat-q4_0-gguf/gemma-4-26B_q4_0-it.gguf
G26H=/home/weezy/models/google/gemma-4-26B-A4B-it-qat-q4_0-gguf/mtp-gemma-4-26B-A4B-it-Q8_0.gguf
NEXT80=/home/weezy/models/Momix-44/Huihui-Qwen3-Coder-Next-Opus-4.6-Reasoning-Distilled-abliterated/Huihui-Qwen3-Coder-Next-Opus-4.6-Reasoning-Distilled-abliterated-Q4_K_M.gguf

median(){ printf '%s\n' "$@" | sort -n | awk '{a[NR]=$1} END{if(NR==0)print 0;else if(NR%2)print a[(NR+1)/2];else print(a[NR/2]+a[NR/2+1])/2}'; }

# (1)+(2) pp/tg, llama side: same 5 prompts, 3 runs/discard-1/median.
llama_pp_tg(){ local BIN="$1" M="$2" CTX="$3"; shift 3; local EXTRA="$*"; local PM=() TM=()
  for p in "$PD"/*.txt; do local pps=() tgs=()
    for r in $(seq 1 $RUNS); do local o; o=$("$BIN" -m "$M" -f "$p" -n $N --temp 0 --ignore-eos -no-cnv -ngl 99 -c "$CTX" $EXTRA 2>&1)
      [ "$r" -le "$WARM" ] && continue
      pps+=("$(printf '%s' "$o"|grep -i 'prompt eval time'|grep -oiE '[0-9.]+ tokens per second'|grep -oE '^[0-9.]+'||echo 0)")
      tgs+=("$(printf '%s' "$o"|grep -iE 'eval time'|grep -vi 'prompt eval'|grep -oiE '[0-9.]+ tokens per second'|grep -oE '^[0-9.]+'||echo 0)")
    done
    PM+=("$(median "${pps[@]}")"); TM+=("$(median "${tgs[@]}")")
  done
  printf "    llama  pp_median %s  tg_median %s\n" "$(median "${PM[@]}")" "$(median "${TM[@]}")"
}
# (3) PPL: ours tool varies by arch; llama-perplexity same corpus.
ours_ppl(){ echo "    ours  PPL: $("$@" 2>&1 | grep -iE 'perplexity|ppl' | tail -1)"; }
llama_ppl(){ local BIN="$1" M="$2"; shift 2; local EXTRA="$*"
  echo "    llama PPL: $("$BIN" -m "$M" -f "$CORPUS" -ngl 99 -c 2048 $EXTRA 2>&1 | grep -iE 'Final estimate|perplexity' | tail -1)"; }

hdr(){ echo "================================================================"; echo ">>> $1"; }

echo "==== bench5 — pp / tg / PPL, both engines, same 5 prompts + wiki corpus ($(date)) ===="
echo "decode=$N runs=$RUNS(discard 1) greedy. PPL=wikitext-2 ($PPLTOK tok). OpenVINO=N/A(not installed). Box: $(uptime)"

hdr "crown 35B-A3B (qwen35moe) 1xB70 Q4_K_M [plain]"
$OURS --gguf $CROWN --decode $N --runs $RUNS --warmup $WARM --max-ctx 16384
ours_ppl $IEPPL --gguf $CROWN --text $CORPUS --max-tokens $PPLTOK
llama_pp_tg $LCLI_S $CROWN 16384
llama_ppl $LPPL_S $CROWN

hdr "27B dense (qwen35) 1xB70 Q4_K_M [plain]"
$OURS --gguf $D27 --decode $N --runs $RUNS --warmup $WARM --max-ctx 16384
ours_ppl $IEPPL --gguf $D27 --text $CORPUS --max-tokens $PPLTOK
llama_pp_tg $LCLI_S $D27 16384
llama_ppl $LPPL_S $D27

hdr "27B dense (qwen35) 1xB70 Q4_K_M [SPEC ours blk.64 MTP K=2; PPL == plain (spec lossless)]"
$OURS --gguf $D27 --spec --spec-k 2 --decode $N --runs $RUNS --warmup $WARM --max-ctx 16384
echo "    llama spec: N/A (no llama path for the embedded blk.64 MTP) -> llama-plain row above is the bar"

hdr "Coder-30B-A3B (qwen3moe) 1xB70 Q4_K_M [plain]"
$OURS --gguf $CODER --decode $N --runs $RUNS --warmup $WARM --max-ctx 16384
ours_ppl $IEPPL --gguf $CODER --text $CORPUS --max-tokens $PPLTOK
llama_pp_tg $LCLI_S $CODER 16384
llama_ppl $LPPL_S $CODER

hdr "gemma-4-31B dense (gemma4) 1xB70 QAT-Q4_0 [SPEC ours MTP K=4 vs llama --model-draft same head]"
$OURS --gguf $G31 --spec --spec-k 4 --spec-head $G31H --decode $N --runs $RUNS --warmup $WARM --max-ctx 16384
ours_ppl $IEPPL_G --gguf $G31 --text $CORPUS --max-tokens $PPLTOK
echo "    (gemma QAT-Q4_0 PPL is high-but-meaningless — argmax-preserving; report it, judge by task acc)"
llama_pp_tg $LCLI_V $G31 16384 --model-draft $G31H --draft-max 4 --draft-min 1
llama_ppl $LPPL_V $G31

hdr "gemma-4-26B-A4B MoE (gemma4) 1xB70 QAT-Q4_0 [SPEC ours MTP K=4 vs llama --model-draft same head]"
IE_GEMMA4_FUSED_MOE=1 IE_GEMMA4_MOE_SOA=1 $OURS --gguf $G26 --spec --spec-k 4 --spec-head $G26H --decode $N --runs $RUNS --warmup $WARM --max-ctx 16384
ours_ppl $IEPPL_G --gguf $G26 --text $CORPUS --max-tokens $PPLTOK
llama_pp_tg $LCLI_V $G26 16384 --model-draft $G26H --draft-max 4 --draft-min 1
llama_ppl $LPPL_V $G26

hdr "Qwen3-Coder-Next-80B (qwen3next) 2xB70 Q4_K_M [plain]"
$OURS --gguf $NEXT80 --gpus 2 --decode $N --runs $RUNS --warmup $WARM --max-ctx 16384
ours_ppl $IEPPL_N $NEXT80 2 --text $CORPUS --max-tokens $PPLTOK
ONEAPI_DEVICE_SELECTOR=level_zero:0,1 llama_pp_tg $LCLI_S $NEXT80 16384 -sm layer
ONEAPI_DEVICE_SELECTOR=level_zero:0,1 llama_ppl $LPPL_S $NEXT80 -sm layer

echo "==== bench5 DONE $(date) ===="
