#!/bin/bash
# run1.sh — run ONE measurement for ONE model, then exit (frees VRAM).
# Short commands (no paste-wrap), timeout-guarded (no hang/freeze).
# usage:  ./tools/run1.sh <model> <metric>
#   model : crown | 27b | coder | gemma31 | gemma26 | next80
#   metric: ours | ours-real | ours-spec | llama | ppl-ours | ppl-llama | kmap | kmap-verify
# Run them one at a time, paste the output back.
#
# APPLES-TO-APPLES (the head-to-head vs llama):
#   ours  == ie-bench  --prefill 512 --decode 128   (synthetic pp512/tg128)
#   llama == llama-bench -p 512 -n 128              (synthetic pp512/tg128)
#   SAME GGUF, SAME single card, SAME synthetic workload. THIS is the comparison.
# ours-real = the 5 real prompts (our-engine color ONLY; llama-cli hangs on the
#   DeltaNet arches so it can NOT run the identical prompts -> never a head-to-head).
# Spec numbers come from ie-qwen35-spec / ie-gemma4-spec (validated lossless),
#   NOT ie-prompt-bench --spec (spec isn't wired into Engine::generate -> that path is slow).
set +e
cd "/home/weezy/00 - Inference Engine" || exit 1
M="$1"; W="$2"
CORPUS=~/llama.cpp/wikitext-2-raw/wiki.test.raw
LBENCH=~/llama.cpp/build-sycl/bin/llama-bench
LPPL=~/llama.cpp/build-sycl/bin/llama-perplexity
LBENCH_V=~/llama.cpp/build-vk/bin/llama-bench
LPPL_V=~/llama.cpp/build-vk/bin/llama-perplexity

case "$M" in
  crown)  G=/home/weezy/.seal/models/Qwen3.6-35B-A3B-Q4_K_M.gguf; DEV=0; VK=0; SPECF="";;
  27b)    G=/home/weezy/models/bartowski/Qwen3.6-27B-GGUF/Qwen_Qwen3.6-27B-Q4_K_M.gguf; DEV=0; VK=0; SPECF="--spec --spec-k 2";;
  coder)  G=/home/weezy/models/Qwen3-Coder-30B-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf; DEV=0; VK=0; SPECF="";;
  gemma31)G=/home/weezy/models/google/gemma-4-31B-it-qat-q4_0-gguf/gemma-4-31B_q4_0-it.gguf; H=/home/weezy/models/google/gemma-4-31B-it-qat-q4_0-gguf/mtp-gemma-4-31B-it-Q8_0.gguf; DEV=0; VK=1; SPECF="--spec --spec-k 4 --spec-head $H";;
  gemma26)G=/home/weezy/models/google/gemma-4-26B-A4B-it-qat-q4_0-gguf/gemma-4-26B_q4_0-it.gguf; H=/home/weezy/models/google/gemma-4-26B-A4B-it-qat-q4_0-gguf/mtp-gemma-4-26B-A4B-it-Q8_0.gguf; DEV=0; VK=1; SPECF="--spec --spec-k 4 --spec-head $H"; export IE_GEMMA4_FUSED_MOE=1 IE_GEMMA4_MOE_SOA=1;;
  next80) G=/home/weezy/models/Momix-44/Huihui-Qwen3-Coder-Next-Opus-4.6-Reasoning-Distilled-abliterated/Huihui-Qwen3-Coder-Next-Opus-4.6-Reasoning-Distilled-abliterated-Q4_K_M.gguf; DEV="0,1"; VK=0; SPECF="--gpus 2";;
  *) echo "unknown model: $M (crown|27b|coder|gemma31|gemma26|next80)"; exit 2;;
esac

OURSGPU=""; [ "$M" = "next80" ] && OURSGPU="--gpus 2"
LB=$LBENCH; LP=$LPPL; SM=""; [ "$M" = "next80" ] && SM="-sm layer"
[ "$VK" = "1" ] && { LB=$LBENCH_V; LP=$LPPL_V; }

echo "### $M / $W ($(basename $G)) ###"
case "$W" in
  ours)       # apples-to-apples synthetic pp512/tg128 == llama-bench -p 512 -n 128 (SAME workload)
    if [ "$M" = "next80" ]; then
      # ie-bench has no 2-card path; 80B head-to-head falls back to the --gpus 2 prompt tool
      # (NOTE: that is the 5-prompt workload, not pp512-synthetic — flagged in the writeup).
      timeout 900 ./build/tools/ie-prompt-bench --gguf "$G" --gpus 2 --decode 128 --runs 3 --warmup 1 --max-ctx 16384 2>&1 | grep -ivE "^\[|loading|mem-plan|^Device|warming" | tail -9
    else
      timeout 700 ./build/tools/ie-bench --gguf "$G" --prefill 512 --decode 128 --warmup 4 2>&1 | grep -ivE "^\[|loading|^  loading|mem-plan|warming" | tail -12
    fi ;;
  ours-real)  # 5 real prompts — our-engine color/appendix ONLY (NOT comparable to llama-bench)
    timeout 700 ./build/tools/ie-prompt-bench --gguf "$G" $OURSGPU --decode 128 --runs 3 --warmup 1 --max-ctx 16384 2>&1 | grep -ivE "^\[|loading|mem-plan|^Device|warming" | tail -9 ;;
  ours-spec)  timeout 900 ./build/tools/ie-prompt-bench --gguf "$G" $SPECF --decode 128 --runs 3 --warmup 1 --max-ctx 16384 2>&1 | grep -ivE "^\[|loading|mem-plan|^Device|warming" | tail -9 ;;
  llama)      ONEAPI_DEVICE_SELECTOR=level_zero:$DEV timeout 600 $LB -m "$G" -ngl 99 -p 512 -n 128 $SM 2>&1 | grep -E "pp512|tg128|t/s|model" | tail -4 ;;
  ppl-ours)   timeout 900 ./build/tools/ie-perplexity --gguf "$G" --text "$CORPUS" --max-tokens 4096 2>&1 | grep -iE "perplexity|nll|fp16" | tail -3 ;;
  ppl-llama)  ONEAPI_DEVICE_SELECTOR=level_zero:$DEV timeout 1800 $LP -m "$G" -f "$CORPUS" -ngl 99 -c 2048 $SM 2>&1 | grep -iE "Final estimate|perplexity" | tail -2 ;;
  kmap)       # per-kernel DECODE map (the tuning tracker). RELATIVE ranking only
              # (profiling inflates per-submit) — find the lever, verify with a real run.
    if [ "$M" = "gemma31" ] || [ "$M" = "gemma26" ]; then
      timeout 500 ./build/tools/ie-gemma4-gen "$G" "Explain in detail how a transformer language model works." 8 profile profile-decode 2>&1 | grep -iE "kernel|gemv|moe_|fa2|geglu|GPU total|TOTAL" | tail -30
    elif [ "$M" = "next80" ]; then
      echo "kmap: N/A — 80B is 2-card; ie-bench --kprofile has no --gpus path. (decode map per-card via the split forward)"
    else
      timeout 500 ./build/tools/ie-bench --gguf "$G" --prefill 1024 --decode 6 --warmup 8 --kprofile-decode 2>&1 | grep -E "DECODE kernel|gemv|fa2|moe_|dn_|rms|quant|GPU total|TOTAL" | tail -30
    fi ;;
  kmap-verify) # per-kernel VERIFY (T=4) map — spec models (27b). Where spec cost lives.
    timeout 600 ./build/tools/ie-bench --gguf "$G" --prefill 1024 --decode 6 --warmup 8 --kprofile-verify 4 2>&1 | grep -E "VERIFY kernel|gemv|fa2|attn|moe_|dn_|gemm|GPU total|TOTAL" | tail -30 ;;
  *) echo "unknown metric: $W (ours|ours-real|ours-spec|llama|ppl-ours|ppl-llama|kmap|kmap-verify)"; exit 2;;
esac
echo "### done $M/$W ###"
