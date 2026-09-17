#!/usr/bin/env bash
# scripts/exl3_lowbpw_ppl_ab.sh
# =============================================================================
# EXL3 low-bpw niche — the make-or-break matched-size PPL A/B (2026-07-07).
#
# Decides the ONE unproven claim the whole EXL3 low-bpw single-card niche rests
# on (docs/exl3_lowbpw_niche_plan_2026-07-07.md §2, §5): does EXL3 @ ~3.0bpw
# beat a same-base sub-4bit GGUF on QUALITY (perplexity)? If yes -> pursue the
# 70B-on-one-card headline. If no -> the niche collapses.
#
# Vehicle: the cheapest convenient DENSE model IE runs on its kLlama3/qwen3 path
# (dense_transformer.cpp) -> Qwen3-8B. Both quants are the SAME base checkpoint
# (Qwen/Qwen3-8B), so tokenizer + weights match; only the quant differs.
#
#   A) EXL3  turboderp/Qwen3-8B-exl3 @ 3.0bpw  (~3.2 GB)  -> imported to a kEXL3 GGUF
#   B) GGUF  bartowski Qwen_Qwen3-8B-Q3_K_S.gguf (~3.77 GB, kQ3_K)
#
# Both PPLs are produced by the SAME tool (build/tools/ie-perplexity), SAME
# corpus, SAME tokenizer -> directly comparable. That apples-to-apples identity
# is the entire point; see the "COMPARABILITY" notes below.
#
# THIS SCRIPT IS NOT RUN BY THE AUTHORING AGENT. It needs the GPU (2xB70) and a
# clean box; run it later, deliberately. It is CPU-safe up to the ie-perplexity
# steps (download + import + preflight are CPU/host-only).
#
# Usage:
#   ./scripts/exl3_lowbpw_ppl_ab.sh            # full run (download->import->PPL A/B)
#   IE_AB_SKIP_DOWNLOAD=1 ./scripts/...        # assume models already on disk
#   IE_AB_MAXTOK=1024 IE_AB_TEXT=/path/wiki.txt ./scripts/...   # bigger/custom corpus
# =============================================================================

# NOTE: intentionally NOT `set -e`. `ie preflight` on the EXL3 GGUF exits 3 today
# (a known preflight whitelist gap, see PREFLIGHT NOTE below) even though the
# engine loads the file fine — an abort there would be a false failure.
set -uo pipefail

# -----------------------------------------------------------------------------
# 0. Config (all overridable via env)
# -----------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${IE_AB_WORK:-$HOME/models/exl3-lowbpw-ab}"

# --- Model A: EXL3 3.0bpw (dense Qwen3 -> dense_transformer.cpp) --------------
EXL3_REPO="${IE_AB_EXL3_REPO:-turboderp/Qwen3-8B-exl3}"
EXL3_REV="${IE_AB_EXL3_REV:-3.0bpw}"          # HF branch = bits-per-weight
EXL3_SRC="$WORK/qwen3-8b-exl3-3.0bpw"          # downloaded safetensors dir
EXL3_GGUF="$WORK/qwen3-8b-exl3-3.0bpw.gguf"    # imported kEXL3 GGUF (what IE runs)

# --- Model B: matched-base GGUF, IE-loadable sub-4bit (kQ3_K) -----------------
# NOTE: IE CANNOT load IQ3_XXS/IQ3_XS/IQ3_S (dtype id exists but is NOT in the
# preflight is_loadable_weight_dtype whitelist / has no kernel). The IE-runnable
# same-family sub-4bit baseline is Q3_K_S (loads via the universal host
# dequant->fp16 fallback). Q3_K_S is ~3.9 bpw i.e. slightly LARGER than EXL3
# 3.0bpw -> a mildly GENEROUS GGUF baseline (see COMPARABILITY risk below).
GGUF_REPO="${IE_AB_GGUF_REPO:-bartowski/Qwen_Qwen3-8B-GGUF}"
GGUF_FILE="${IE_AB_GGUF_FILE:-Qwen_Qwen3-8B-Q3_K_S.gguf}"
GGUF_PATH="$WORK/$GGUF_FILE"

# The GGUF above ALSO serves as the tokenizer reference for the EXL3 import
# (`ie_import <hf_dir> <out.gguf> <tok_ref.gguf>`), which GUARANTEES both models
# carry the byte-identical tokenizer -> the cleanest possible apples-to-apples.
TOK_REF="$GGUF_PATH"

# --- Corpus / budget (identical for BOTH models = the comparability contract) -
MAXTOK="${IE_AB_MAXTOK:-512}"
CTX="${IE_AB_CTX:-4096}"
TEXT_ARG=()
[[ -n "${IE_AB_TEXT:-}" ]] && TEXT_ARG=(--text "$IE_AB_TEXT")

# --- Tools -------------------------------------------------------------------
IE="$REPO_ROOT/build/src/ie"
IE_IMPORT="$REPO_ROOT/build/tools/ie_import"   # EXL3-aware importer (NOT `ie import`, see below)
IE_PPL="$REPO_ROOT/build/tools/ie-perplexity"
HF="${HF:-hf}"                                  # huggingface CLI (~/.local/bin/hf)

# IMPORTANT: the main CLI `ie import` hard-codes the AWQ importer
# (src/cli/main.cpp:80 -> import_awq_to_gguf, no method routing). ONLY
# build/tools/ie_import peeks config.json quant_method and routes exl3 ->
# import_exl3_to_gguf (dense) / import_exl3_qwen3next_to_gguf. Use ie_import.

log()  { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
note() { printf '   \033[33m! %s\033[0m\n' "$*"; }
gb()   { awk -v b="$1" 'BEGIN{printf "%.2f GB", b/1073741824}'; }

# -----------------------------------------------------------------------------
# 1. oneAPI env (SYCL runtime). setvars.sh returns exit 3 but the env IS set;
#    chain with `;` not `&&` (scripts/setup gotcha). Needed for the GPU steps.
# -----------------------------------------------------------------------------
if [[ -f /opt/intel/oneapi/setvars.sh ]]; then
    log "source oneAPI setvars (needed for the GPU ie-perplexity steps)"
    source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
fi

mkdir -p "$WORK"

# -----------------------------------------------------------------------------
# 2. Download both models (GUARDED — skip if already present). CPU/network only.
# -----------------------------------------------------------------------------
if [[ "${IE_AB_SKIP_DOWNLOAD:-0}" != "1" ]]; then
    if [[ ! -f "$EXL3_SRC/config.json" ]]; then
        log "download EXL3 $EXL3_REPO @ $EXL3_REV -> $EXL3_SRC"
        "$HF" download "$EXL3_REPO" --revision "$EXL3_REV" --local-dir "$EXL3_SRC" \
            || { echo "EXL3 download failed — verify the '$EXL3_REV' branch exists on $EXL3_REPO"; exit 1; }
    else
        note "EXL3 source already present ($EXL3_SRC) — skipping download"
    fi

    if [[ ! -f "$GGUF_PATH" ]]; then
        log "download GGUF $GGUF_REPO / $GGUF_FILE -> $GGUF_PATH"
        "$HF" download "$GGUF_REPO" "$GGUF_FILE" --local-dir "$WORK" \
            || { echo "GGUF download failed — verify $GGUF_FILE exists in $GGUF_REPO"; exit 1; }
        # hf may nest under the repo path; normalize to $GGUF_PATH.
        [[ -f "$GGUF_PATH" ]] || GGUF_PATH="$(find "$WORK" -name "$GGUF_FILE" | head -1)"
        TOK_REF="$GGUF_PATH"
    else
        note "GGUF already present ($GGUF_PATH) — skipping download"
    fi
else
    note "IE_AB_SKIP_DOWNLOAD=1 — assuming both models already on disk"
fi

# -----------------------------------------------------------------------------
# 3. Import EXL3 safetensors -> kEXL3 GGUF (GUARDED). CPU/host-only.
#    Dense arch (qwen3) -> import_exl3_to_gguf. tok_ref = the Q3_K_S GGUF so the
#    tokenizer is identical to Model B.
# -----------------------------------------------------------------------------
if [[ ! -f "$EXL3_GGUF" ]]; then
    log "import EXL3 -> GGUF (build/tools/ie_import, EXL3-aware router)"
    "$IE_IMPORT" "$EXL3_SRC" "$EXL3_GGUF" "$TOK_REF" \
        || { echo "ie_import failed"; exit 1; }
else
    note "EXL3 GGUF already imported ($EXL3_GGUF) — skipping import"
fi

# -----------------------------------------------------------------------------
# 4. On-disk size + 1-card fit verdict (preflight). CPU/host-only.
# -----------------------------------------------------------------------------
log "on-disk sizes"
EXL3_BYTES=$(stat -c %s "$EXL3_GGUF" 2>/dev/null || echo 0)
GGUF_BYTES=$(stat -c %s "$GGUF_PATH" 2>/dev/null || echo 0)
printf "   EXL3 3.0bpw GGUF : %s (%s)\n" "$(gb "$EXL3_BYTES")" "$EXL3_GGUF"
printf "   GGUF  Q3_K_S     : %s (%s)\n" "$(gb "$GGUF_BYTES")" "$GGUF_PATH"

# PREFLIGHT NOTE: `ie preflight` reports the correct VRAM numbers for a kEXL3
# GGUF (estimate_weight_bytes sums real packed tensor nbytes), BUT its will_load
# verdict is WRONG for EXL3: is_loadable_weight_dtype (src/loaders/preflight.cpp:31)
# does NOT list DType::kEXL3, so it prints "UNSUPPORTED EXL3 ... will not load"
# and exits 3 — even though DenseModel loads kEXL3 fine (dense_transformer.cpp:148).
# => For EXL3, read the printed "VRAM need" line; IGNORE the exit-3 verdict.
# (One-line orchestrator fix: add `case DType::kEXL3:` to the whitelist.)
log "preflight EXL3 GGUF (1 GPU) — expect correct VRAM line, bogus exit 3 (kEXL3 whitelist gap)"
"$IE" preflight "$EXL3_GGUF" --ctx "$CTX" --gpus 1 || note "exit 3 here is the known kEXL3 whitelist gap, NOT a real fit failure"

log "preflight GGUF Q3_K_S (1 GPU) — expect exit 0 = will load"
"$IE" preflight "$GGUF_PATH" --ctx "$CTX" --gpus 1
echo "   (8B models are ~3-4 GB -> both fit one 32 GB B70 trivially; the real 1-card"
echo "    fit story is the 70B headline, which this small A/B only DE-RISKS.)"

# -----------------------------------------------------------------------------
# 5. THE A/B — perplexity on each, SAME tool + corpus + tokenizer. NEEDS GPU.
# -----------------------------------------------------------------------------
run_ppl () {   # $1=label  $2=gguf  -> echoes the perplexity number
    local label="$1" gguf="$2" logf="$WORK/ppl_${1}.log"
    log "ie-perplexity [$label]  (max_tokens=$MAXTOK ctx=$CTX)"
    "$IE_PPL" --gguf "$gguf" --max-tokens "$MAXTOK" --ctx "$CTX" "${TEXT_ARG[@]}" \
        2>&1 | tee "$logf"
    # ie-perplexity emits a TSV data line: `# fp16 <tokens> <avg_nll> <perplexity>`
    grep -E '^# fp16' "$logf" | tail -1 | awk '{print $NF}'
}

PPL_EXL3=$(run_ppl "exl3_3.0bpw" "$EXL3_GGUF")
PPL_GGUF=$(run_ppl "gguf_q3ks"   "$GGUF_PATH")

# -----------------------------------------------------------------------------
# 6. Verdict
# -----------------------------------------------------------------------------
log "A/B RESULT (lower PPL = better; SAME corpus/tokenizer/tool = directly comparable)"
printf "   EXL3  Qwen3-8B 3.0bpw : PPL = %s\n" "${PPL_EXL3:-<none>}"
printf "   GGUF  Qwen3-8B Q3_K_S : PPL = %s   (~3.9 bpw, i.e. a GENEROUS GGUF baseline)\n" "${PPL_GGUF:-<none>}"

if [[ -z "${PPL_EXL3:-}" || -z "${PPL_GGUF:-}" ]]; then
    echo ""
    echo "   INCONCLUSIVE — one or both PPL numbers failed to parse. Check the *.log in $WORK."
    exit 1
fi

if awk -v e="$PPL_EXL3" -v g="$PPL_GGUF" 'BEGIN{exit !(e < g)}'; then
    printf "\n   \033[32mPASS\033[0m — EXL3 3.0bpw beats a LARGER Q3_K_S at quality.\n"
    echo   "          => The niche is real: PURSUE the 70B-on-one-card headline"
    echo   "             (download Llama-3.3/3.1-70B-exl3 @ 3.0bpw, 1-card + int8-kv)."
else
    printf "\n   \033[31mFAIL\033[0m — EXL3 3.0bpw did NOT beat Q3_K_S.\n"
    echo   "          => Before declaring the niche dead, run the TRUE same-size"
    echo   "             cross-check below (Q3_K_S is ~0.9 bpw larger than EXL3 3.0bpw,"
    echo   "             so a narrow loss here may be a size handicap, not a quant loss)."
fi

# -----------------------------------------------------------------------------
# 7. OPTIONAL manual cross-check — true SAME-bpw point via llama.cpp.
#    NOT directly comparable to the IE numbers above (different harness: llama
#    strides a sliding window over the corpus, different tokenizer plumbing) —
#    treat it ONLY as a secondary directional sanity signal, never mix into the
#    PASS/FAIL above.
# -----------------------------------------------------------------------------
cat <<EOF

-------------------------------------------------------------------------------
OPTIONAL cross-check (run by hand; NOT apples-to-apples with the IE PPL above):
  The true same-size GGUF is IQ3_XXS (~3.06 bpw ~= EXL3 3.0bpw), but IE CANNOT
  load IQ3_XXS (no kernel / not whitelisted). Score it on llama.cpp instead:

    $HF download $GGUF_REPO Qwen_Qwen3-8B-IQ3_XXS.gguf --local-dir "$WORK"
    ~/llama.cpp/build-sycl/bin/llama-perplexity \\
        -m "$WORK/Qwen_Qwen3-8B-IQ3_XXS.gguf" \\
        -f <wikitext-2-raw/wiki.test.raw> -c 512

  For a matched llama.cpp reference on the EXL3 side you would need the EXL3
  file scored by exllamav3 (~/exllamav3-ref); llama.cpp cannot read kEXL3.
  => The DECISIVE comparison is the IE-vs-IE A/B in step 6; this is colour only.
-------------------------------------------------------------------------------
EOF
