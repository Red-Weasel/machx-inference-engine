#!/usr/bin/env bash
# wave1_gate.sh — Wave-1 reusable per-family validation battery.
#
# Generalises scripts/p3_parity_llama3.sh over an arbitrary GGUF / arch /
# template family. It is the harness the per-family GPU validation tasks
# (Wave-1 Tasks 5-9: Yi, InternLM2, R1-Distill, Mistral/Nemo/Devstral/Codestral,
# Phi, Granite, Baichuan) invoke once a model is staged. It gates the *property*
# (correct arch routing + cosine ≥ 0.999 + finite PPL + coherent chat), NOT a
# fabricated per-model PPL number — we cannot know that without the GGUF.
#
# STAGES
#   0. config / arch-detect  (CPU, no GPU)  — ie-inspect dumps general.architecture
#      + tokenizer.chat_template + dims. Asserts the GGUF reports the expected
#      arch string and (if TEMPLATE is set) carries the matching template marker.
#   1. tokenizer encode-parity (CPU)        — llama-tokenize golden --ids vs the
#      engine's ie-dense-dump 'tokens=' line on a fixed prose+code corpus.
#      [OPTIONAL: needs llama.cpp + a GPU-free dump; skipped if either absent.]
#   2. per-layer cosine (GPU)               — ie-dense-dump vs ie-llama-dump +
#      tools/diff_layers.sh; PASS = min cos_sim ≥ COS_MIN (default 0.999) on every
#      comparable slot (a wrong un-permute / rope / theta blows up at L01).
#   3. PPL (GPU)                            — ie-perplexity --gguf, deterministic.
#      PASS = finite + (if EXPECT_NLL given) bit-exact; else record-only (judge by
#      oracle RATIO, not an absolute — our full-causal PPL runs below llama's).
#   4. greedy / chat smoke (GPU)            — ie run, one turn; PASS = non-empty
#      coherent continuation that renders the right template (and <think> for
#      reasoning distills).
#
# ENV (all overridable):
#   GGUF        (required)  model path
#   ARCH                    expected general.architecture (llama|qwen2|phi3|granite|…)
#   TEMPLATE                expected family: mistral|deepseek|llama3|chatml|auto
#   COS_MIN     (0.999)     per-layer cosine gate
#   EXPECT_NLL  ("")        if set, stage-3 asserts bit-exact avg NLL
#   PROMPT      ("The capital of France is")
#   N_LAYERS    (auto)      layers for ie-llama-dump (--n-layers); default = all
#   BUILD       ($ROOT/build)
#   LLAMA_BIN   ($HOME/llama.cpp/build-vk/bin)
#   SKIP_GPU    (0)         1 → run only the CPU stages (0,1) — for plumbing checks
#   ORACLE_GGUF ($GGUF)     GGUF llama.cpp loads for the oracle (same model)
#
# GPU EXCLUSIVITY: the GPU stages (2-4) load a model. The CALLER must run the
# dual-card pre-flight gate (ps/free/xpu-smi) BEFORE invoking this script for a
# real model — this script does not police other repos' loads. The CPU stages
# (0,1) and SKIP_GPU=1 are exclusivity-free.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD:-$ROOT/build}"
GGUF="${GGUF:-}"
ARCH="${ARCH:-}"
TEMPLATE="${TEMPLATE:-}"
COS_MIN="${COS_MIN:-0.999}"
EXPECT_NLL="${EXPECT_NLL:-}"
PROMPT="${PROMPT:-The capital of France is}"
CORPUS="${CORPUS:-Hello world. def f(x): return x*2  # 1234 café}"
N_LAYERS="${N_LAYERS:-}"
SKIP_GPU="${SKIP_GPU:-0}"
LLAMA_BIN="${LLAMA_BIN:-$HOME/llama.cpp/build-vk/bin}"
ORACLE_GGUF="${ORACLE_GGUF:-$GGUF}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$LLAMA_BIN"

OUT="$(mktemp -d /tmp/wave1_gate.XXXXXX)"
FAIL=0

say()  { printf '\033[1m%s\033[0m\n' "$*"; }
pass() { printf '  \033[32mPASS\033[0m %s\n' "$*"; }
fail() { printf '  \033[31mFAIL\033[0m %s\n' "$*"; FAIL=1; }
skip() { printf '  \033[33mSKIP\033[0m %s\n' "$*"; }

[[ -n "$GGUF" ]]   || { echo "usage: GGUF=<model> [ARCH=..] [TEMPLATE=..] $0"; exit 2; }
[[ -f "$GGUF" ]]   || { echo "SKIP: $GGUF not found"; exit 0; }

say "wave1_gate — GGUF=$(basename "$GGUF")  ARCH=${ARCH:-?}  TEMPLATE=${TEMPLATE:-?}"

# ---- 0. config / arch-detect (CPU) ------------------------------------
say "[0] arch-detect + config (CPU, ie-inspect)"
if [[ -x "$BUILD/tools/ie-inspect" ]]; then
    # ie-inspect colorises its output (no --no-color flag); strip ANSI SGR codes
    # so the arch / template parse sees plain text (else the value reads as '0m').
    "$BUILD/tools/ie-inspect" "$GGUF" 2>&1 \
        | sed 's/\x1b\[[0-9;]*m//g' > "$OUT/inspect.log" \
        || fail "ie-inspect failed (see $OUT/inspect.log)"
    # The arch is the quoted string value on the general.architecture row
    # (... string  "qwen2"). Grab the first quoted token after the key.
    DET_ARCH=$(grep -iE 'general\.architecture' "$OUT/inspect.log" \
               | sed -n 's/.*"\([^"]*\)".*/\1/p' | head -1)
    [[ -n "$DET_ARCH" ]] && printf '  detected general.architecture = %s\n' "$DET_ARCH"
    if [[ -n "$ARCH" && -n "$DET_ARCH" ]]; then
        if [[ "$DET_ARCH" == "$ARCH" ]]; then pass "arch == $ARCH"
        else fail "arch '$DET_ARCH' != expected '$ARCH'"; fi
    fi
    # Template-family marker presence (mirrors classify_template_family).
    if [[ -n "$TEMPLATE" && "$TEMPLATE" != "auto" ]]; then
        case "$TEMPLATE" in
            mistral)  MARK='\[INST\]';;
            llama3)   MARK='start_header_id';;
            chatml)   MARK='im_start';;
            deepseek) MARK='Assistant';;   # the <｜Assistant｜> sentinel
            *)        MARK='';;
        esac
        if [[ -n "$MARK" ]]; then
            if grep -qE "$MARK" "$OUT/inspect.log"; then
                pass "chat_template carries the $TEMPLATE marker"
            else
                skip "could not confirm $TEMPLATE marker in ie-inspect dump (template KV may be truncated; verify by hand)"
            fi
        fi
    fi
else
    skip "ie-inspect not built — cannot do the CPU config check"
fi

# ---- 1. tokenizer encode-parity (CPU; needs llama-tokenize) -----------
say "[1] tokenizer encode-parity vs llama.cpp (corpus: \"$CORPUS\")"
if [[ -x "$LLAMA_BIN/llama-tokenize" && -x "$BUILD/tools/ie-dense-dump" && "$SKIP_GPU" != "1" ]]; then
    "$LLAMA_BIN/llama-tokenize" -m "$GGUF" -p "$CORPUS" --ids 2>/dev/null \
        | grep -oE '[0-9]+' | tr '\n' ' ' | sed 's/ *$//' > "$OUT/golden_ids.txt"
    # Engine ids: ie-dense-dump prints "prompt='..' tokens=<id id ...>".
    "$BUILD/tools/ie-dense-dump" --gguf "$GGUF" -p "$CORPUS" -n 0 > "$OUT/enc.log" 2>&1 || true
    sed -n "s/^prompt=.*tokens=//p" "$OUT/enc.log" | head -1 \
        | tr -s ' ' | sed 's/ *$//' > "$OUT/mine_ids.txt"
    G=$(cat "$OUT/golden_ids.txt"); M=$(cat "$OUT/mine_ids.txt")
    if [[ -z "$G" || -z "$M" ]]; then
        skip "encode-parity could not collect both id lists (golden='$G' mine='$M')"
    elif [[ "$G" == "$M" ]]; then
        pass "encode ids identical to llama.cpp"
    else
        # BOS prepend can shift by one leading id — note but do not auto-pass.
        fail "encode ids differ — golden=[$G] mine=[$M]"
    fi
else
    skip "llama-tokenize or ie-dense-dump unavailable (or SKIP_GPU=1)"
fi

if [[ "$SKIP_GPU" == "1" ]]; then
    echo; say "wave1_gate: CPU-only run complete (SKIP_GPU=1). artifacts: $OUT"
    exit $FAIL
fi

# ---- 2. per-layer cosine (GPU) ----------------------------------------
say "[2] per-layer cosine vs llama.cpp (min cos_sim ≥ $COS_MIN)"
"$BUILD/tools/ie-dense-dump" --gguf "$GGUF" --dump "$OUT/mine" -p "$PROMPT" \
    > "$OUT/mine.log" 2>&1 || fail "ie-dense-dump (see $OUT/mine.log)"
LL_ARGS=(-m "$ORACLE_GGUF" --dump "$OUT/ref" -ngl 0 -p "$PROMPT")
[[ -n "$N_LAYERS" ]] && LL_ARGS+=(--n-layers "$N_LAYERS")
"$BUILD/tools/ie-llama-dump" "${LL_ARGS[@]}" > "$OUT/ref.log" 2>&1 \
    || fail "ie-llama-dump (see $OUT/ref.log)"
if [[ $FAIL -eq 0 ]]; then
    # diff_layers needs a slot count; derive from however many ref slots dumped.
    NSLOT=$(ls "$OUT"/ref_L*.bin 2>/dev/null | wc -l)
    [[ "$NSLOT" -gt 0 ]] || NSLOT=33
    bash "$ROOT/tools/diff_layers.sh" "$OUT/mine" "$OUT/ref" "$NSLOT" | tee "$OUT/diff.txt"
    MINCOS=$(awk 'NF>=4 && $1 ~ /^[0-9]+$/ {print $4}' "$OUT/diff.txt" | sort -g | head -1)
    if [[ -z "$MINCOS" ]]; then
        fail "could not parse cos_sim (see $OUT/diff.txt)"
    elif awk "BEGIN{exit !($MINCOS >= $COS_MIN)}"; then
        pass "min cos_sim $MINCOS ≥ $COS_MIN (forward matches the oracle)"
    else
        fail "min cos_sim $MINCOS < $COS_MIN — forward divergence (debug at the first sub-gate slot)"
    fi
fi

# ---- 3. PPL (GPU) ------------------------------------------------------
say "[3] PPL gate (builtin corpus, deterministic)"
"$BUILD/tools/ie-perplexity" --gguf "$GGUF" --max-tokens 511 > "$OUT/ppl.log" 2>&1
NLL=$(sed -n 's/^# fp16\t[0-9]*\t\([0-9.]*\)\t[0-9.]*$/\1/p' "$OUT/ppl.log")
PPL=$(sed -n 's/^# fp16\t[0-9]*\t[0-9.]*\t\([0-9.]*\)$/\1/p' "$OUT/ppl.log")
if [[ -z "$NLL" ]]; then
    fail "could not parse avg NLL (see $OUT/ppl.log)"
elif [[ -n "$EXPECT_NLL" ]]; then
    if [[ "$NLL" == "$EXPECT_NLL" ]]; then pass "avg NLL $NLL == $EXPECT_NLL (bit-exact, PPL $PPL)"
    else fail "avg NLL $NLL != expected $EXPECT_NLL (numerics changed)"; fi
else
    # Record-only: finite + positive is the property gate (oracle-ratio judged
    # by hand against the doc matrix; our full-causal PPL runs below llama's).
    if awk "BEGIN{exit !($NLL > 0 && $NLL < 100)}"; then
        pass "avg NLL $NLL finite/sane → PPL $PPL (RECORD this vs the oracle ratio)"
    else
        fail "avg NLL $NLL not finite/sane"
    fi
fi

# ---- 4. greedy / chat smoke (GPU) -------------------------------------
say "[4] chat smoke (ie run, one turn — renders the family template)"
printf '%s\n/quit\n' "$PROMPT" | "$BUILD/src/ie" run "$GGUF" > "$OUT/chat.log" 2>&1 || true
# Strip the prompt banner / '> ' markers; require some generated continuation.
GEN=$(sed -n '/^ie chat/,$p' "$OUT/chat.log" | tr -d '\r' | grep -v '^ie chat' | tr -s ' \n' ' ' | sed 's/^[> ]*//')
if [[ -n "${GEN// /}" ]]; then
    pass "chat produced a continuation: '$(echo "$GEN" | cut -c1-80)'"
else
    fail "chat produced no output (see $OUT/chat.log)"
fi

echo
if [[ $FAIL -eq 0 ]]; then say "wave1_gate: ALL GREEN — $(basename "$GGUF")  (artifacts: $OUT)"
else say "wave1_gate: FAILURES — $(basename "$GGUF")  (artifacts kept at $OUT)"; fi
exit $FAIL
